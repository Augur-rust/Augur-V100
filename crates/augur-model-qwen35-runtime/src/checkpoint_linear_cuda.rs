//! A complete recurrent transformer layer for explicit safetensors profiles.
//! Uses the original-activation scalar projection baseline, not the mixed
//! artifact's Q8 assumptions. All operations are bound once; state and scratch
//! have one stream owner and no per-token allocation or intermediate D2H.
use crate::checkpoint_ffn_cuda::{CheckpointFfnContext, CheckpointFfnProgram};
use crate::checkpoint_projection_cuda::{
    CheckpointProjectionPolicy, CheckpointScalarCudaProjection, CheckpointSmallGatePair,
};
use augur_backend::{
    CudaAddF32ArgsV1, CudaF32View, CudaQwen35LinearProjectionPostprocessArgsV1,
    CudaQwen35LinearStepArgsV1, CudaRmsnormF32ArgsV1, CudaSiluMulF32ArgsV1, OwnedCudaBuffer,
    OwnedCudaStream,
};
use augur_core::PhysicalLayoutId;
use augur_model_qwen35::CheckpointArtifact;
use std::{collections::BTreeMap, error::Error, rc::Rc};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
type Buffer = Rc<OwnedCudaBuffer>;
enum Operation {
    Norm(CudaRmsnormF32ArgsV1),
    Projection(CheckpointScalarCudaProjection),
    SmallGatePair(CheckpointSmallGatePair),
    #[cfg(feature = "qwen35-nerkyor-dflash-research")]
    NerkyorBf16GatePair(crate::checkpoint_projection_cuda::nerkyor_gate::NerkyorBf16GatePair),
    Gates(CudaQwen35LinearProjectionPostprocessArgsV1),
    Recurrent(augur_backend::CudaQwen35CheckpointLinearStepArgsV1),
    Add(CudaAddF32ArgsV1),
    Silu(CudaSiluMulF32ArgsV1),
    QpnFfn(CheckpointFfnProgram),
}
impl Operation {
    fn submit(&self, stream: &OwnedCudaStream) -> Result<()> {
        match self {
            Self::Projection(op) => op.submit_for_plan()?,
            Self::SmallGatePair(op) => op.submit_for_plan()?,
            #[cfg(feature = "qwen35-nerkyor-dflash-research")]
            Self::NerkyorBf16GatePair(op) => op.submit_for_plan()?,
            Self::Norm(op) => op.submit_for_plan(stream)?,
            Self::Gates(op) => op.submit_for_plan(stream)?,
            Self::Recurrent(op) => op.submit_for_plan(stream)?,
            Self::Add(op) => op.submit_for_plan(stream)?,
            Self::Silu(op) => op.submit_for_plan(stream)?,
            Self::QpnFfn(op) => op.submit(stream)?,
        }
        Ok(())
    }
}

pub struct CheckpointLinearCudaLayer {
    // Destroy the executable before the retained operands and stream it references.
    #[cfg(feature = "qwen35-augur-http")]
    b1_graph: Option<augur_backend::OwnedCudaGraphExec>,
    #[cfg(feature = "qwen35-prefill-cublas")]
    input_norm_qpn: Option<augur_backend::cuda_qwen35_prefill_cublas::CudaQwen35NormQpnGroupV1>,
    #[cfg(feature = "qwen35-prefill-cublas")]
    post_norm_qpn: Option<augur_backend::cuda_qwen35_prefill_cublas::CudaQwen35NormQpnGroupV1>,
    stream: Rc<OwnedCudaStream>,
    operations: Vec<Operation>,
    buffers: BTreeMap<&'static str, Buffer>,
    tokens: usize,
    poisoned: bool,
}

pub(super) fn view(buffer: &OwnedCudaBuffer) -> Result<CudaF32View> {
    Ok(CudaF32View::new(
        buffer.view(0, buffer.len_bytes())?,
        PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
        buffer.len_bytes() / 4,
    )?)
}
pub(super) fn upload(stream: &OwnedCudaStream, data: &[f32]) -> Result<Buffer> {
    if data.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite layer vector".into());
    }
    let buffer = OwnedCudaBuffer::allocate(stream.device_ordinal(), data.len() * 4)?;
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    buffer.copy_from_host_sync(stream, &bytes)?;
    Ok(Rc::new(buffer))
}
pub(super) fn empty(stream: &OwnedCudaStream, elements: usize) -> Result<Buffer> {
    let mut buffer = OwnedCudaBuffer::allocate(stream.device_ordinal(), elements * 4)?;
    buffer.zero_sync(stream)?;
    Ok(Rc::new(buffer))
}

impl CheckpointLinearCudaLayer {
    pub fn load(
        reader: &mut CheckpointArtifact,
        layer: usize,
        stream: Rc<OwnedCudaStream>,
        input: Buffer,
        output: Buffer,
    ) -> Result<Self> {
        Self::load_with_policy(
            reader,
            layer,
            stream,
            input,
            output,
            CheckpointProjectionPolicy::OriginalF32,
        )
    }
    pub fn load_with_policy(
        reader: &mut CheckpointArtifact,
        layer: usize,
        stream: Rc<OwnedCudaStream>,
        input: Buffer,
        output: Buffer,
        policy: CheckpointProjectionPolicy,
    ) -> Result<Self> {
        let ffn = CheckpointFfnContext::new(Rc::clone(&stream), policy)?;
        Self::load_with_ffn_context(reader, layer, stream, input, output, &ffn)
    }
    pub(crate) fn load_with_ffn_context(
        reader: &mut CheckpointArtifact,
        layer: usize,
        stream: Rc<OwnedCudaStream>,
        input: Buffer,
        output: Buffer,
        ffn: &CheckpointFfnContext,
    ) -> Result<Self> {
        let policy = ffn.policy();
        if layer >= 64 || (layer + 1) % 4 == 0 {
            return Err("expected one of the 48 recurrent layers".into());
        }
        if input.len_bytes() != 5120 * 4
            || output.len_bytes() != 5120 * 4
            || Rc::ptr_eq(&input, &output)
        {
            return Err("layer requires distinct exact hidden buffers".into());
        }
        let prefix = format!("model.language_model.layers.{layer}");
        let mut b = BTreeMap::from([("input", input), ("output", output)]);
        for (name, size) in [
            ("normalized", 5120),
            ("qkv", 10240),
            ("z", 6144),
            ("alpha", 48),
            ("beta_raw", 48),
            ("decay", 48),
            ("beta", 48),
            ("history", 10240 * 3),
            ("state", 48 * 128 * 128),
            ("activated", 10240),
            ("recurrent", 6144),
            ("linear_output", 6144),
            ("attention", 5120),
            ("attention_residual", 5120),
            ("post_norm", 5120),
            ("gate", 17408),
            ("up", 17408),
            ("swiglu", 17408),
            ("ffn_output", 5120),
        ] {
            if policy.qpn_ffn_layer(layer) && ["gate", "up", "swiglu"].contains(&name) {
                continue;
            }
            b.insert(name, empty(&stream, size)?);
        }
        for (name, suffix) in [
            ("input_norm_weight", "input_layernorm.weight"),
            ("post_norm_weight", "post_attention_layernorm.weight"),
            ("conv", "linear_attn.conv1d.weight"),
            ("norm", "linear_attn.norm.weight"),
            ("ssm_a", "linear_attn.A_log"),
            ("dt_bias", "linear_attn.dt_bias"),
        ] {
            b.insert(
                name,
                upload(
                    &stream,
                    &reader.load_dense_execution(&format!("{prefix}.{suffix}"))?,
                )?,
            );
        }
        let v = |name| view(&b[name]);
        let epsilon = reader.config().rms_norm_eps as f32;
        let mut operations = vec![Operation::Norm(CudaRmsnormF32ArgsV1::bind(
            &stream,
            v("input")?,
            v("input_norm_weight")?,
            v("normalized")?,
            epsilon,
        )?)];
        let projection = |reader: &mut CheckpointArtifact,
                          suffix: &str,
                          input: &str,
                          output: &str|
         -> Result<Operation> {
            let weight = ffn.upload_projection(
                Rc::clone(&stream), reader.load_linear(&format!("{prefix}.{suffix}"))?,
            )?;
            Ok(Operation::Projection(weight.bind(
                Rc::clone(&b[input]),
                0,
                Rc::clone(&b[output]),
                0,
            )?))
        };
        for (suffix, output) in [
            ("in_proj_qkv", "qkv"),
            ("in_proj_z", "z"),

        ] {
            operations.push(projection(
                reader,
                &format!("linear_attn.{suffix}"),
                "normalized",
                output,
            )?);
        }
        let alpha = projection(reader, "linear_attn.in_proj_a", "normalized", "alpha")?;
        let beta = projection(reader, "linear_attn.in_proj_b", "normalized", "beta_raw")?;
        if policy.small_gate_pair_schedule().is_some() {
            let (Operation::Projection(alpha),Operation::Projection(beta)) = (alpha,beta) else {
                return Err("small-gate pair projection binding changed".into());
            };
            operations.push(Operation::SmallGatePair(alpha.into_small_gate_pair(beta)?));
        } else {
            #[cfg(feature = "qwen35-nerkyor-dflash-research")]
            if policy.uses_nerkyor_gate_f32_cta_research() {
                let (Operation::Projection(alpha), Operation::Projection(beta)) = (alpha, beta) else {
                    return Err("research BF16 gate projection binding changed".into());
                };
                operations.push(Operation::NerkyorBf16GatePair(
                    crate::checkpoint_projection_cuda::nerkyor_gate::NerkyorBf16GatePair::bind(alpha, beta)?));
            } else {
                operations.push(alpha);
                operations.push(beta);
            }
            #[cfg(not(feature = "qwen35-nerkyor-dflash-research"))]
            {
                operations.push(alpha);
                operations.push(beta);
            }
        }
        operations.push(Operation::Gates(
            CudaQwen35LinearProjectionPostprocessArgsV1::bind(
                &stream,
                v("alpha")?,
                v("beta_raw")?,
                v("dt_bias")?,
                v("ssm_a")?,
                v("decay")?,
                v("beta")?,
            )?,
        ));
        operations.push(Operation::Recurrent(
            CudaQwen35LinearStepArgsV1::bind(
                &stream,
                v("qkv")?,
                v("z")?,
                v("conv")?,
                v("history")?,
                v("decay")?,
                v("beta")?,
                v("state")?,
                v("norm")?,
                v("activated")?,
                v("recurrent")?,
                v("linear_output")?,
                epsilon,
            )?
            .for_checkpoint_source(),
        ));
        operations.push(projection(
            reader,
            "linear_attn.out_proj",
            "linear_output",
            "attention",
        )?);
        operations.push(Operation::Add(CudaAddF32ArgsV1::bind(
            &stream,
            v("input")?,
            v("attention")?,
            v("attention_residual")?,
        )?));
        operations.push(Operation::Norm(CudaRmsnormF32ArgsV1::bind(
            &stream,
            v("attention_residual")?,
            v("post_norm_weight")?,
            v("post_norm")?,
            epsilon,
        )?));
        if policy.qpn_ffn_layer(layer) {
            operations.push(Operation::QpnFfn(ffn.bind(
                reader,
                layer,
                &stream,
                Rc::clone(&b["post_norm"]),
                Rc::clone(&b["ffn_output"]),
            )?));
        } else {
            operations.push(projection(reader, "mlp.gate_proj", "post_norm", "gate")?);
            operations.push(projection(reader, "mlp.up_proj", "post_norm", "up")?);
            operations.push(Operation::Silu(CudaSiluMulF32ArgsV1::bind(
                &stream,
                v("gate")?,
                v("up")?,
                v("swiglu")?,
            )?));
            operations.push(projection(reader, "mlp.down_proj", "swiglu", "ffn_output")?);
        }
        operations.push(Operation::Add(CudaAddF32ArgsV1::bind(
            &stream,
            v("attention_residual")?,
            v("ffn_output")?,
            v("output")?,
        )?));
        Ok(Self {
            #[cfg(feature = "qwen35-augur-http")]
            b1_graph: None,
            stream,
            operations,
            #[cfg(feature = "qwen35-prefill-cublas")]
            input_norm_qpn: None,
            #[cfg(feature = "qwen35-prefill-cublas")]
            post_norm_qpn: None,
            buffers: b,
            tokens: 0,
            poisoned: false,
        })
    }

    /// Test-only actual attention-input capture, outside every timed lane.
    #[cfg(test)]
    pub(crate) fn diagnostic_normalized_buffer(&self) -> &OwnedCudaBuffer {
        &self.buffers["normalized"]
    }

    #[cfg(all(test, feature = "qwen35-nerkyor-dflash-research"))]
    pub(crate) fn diagnostic_bf16_gate_outputs(&self) -> (&OwnedCudaBuffer, &OwnedCudaBuffer) {
        (&self.buffers["alpha"], &self.buffers["beta_raw"])
    }

    pub fn submitted_tokens(&self) -> usize {
        self.tokens
    }

    /// Enqueue a whole layer. A failed partial submission cannot be continued.
    pub fn submit_for_plan(&mut self) -> Result<()> {
        if self.poisoned {
            return Err("checkpoint layer owner is poisoned".into());
        }
        let next = self.tokens.checked_add(1).ok_or("token counter overflow")?;
        self.poisoned = true;
        #[cfg(feature = "qwen35-augur-http")]
        if let Some(graph) = &self.b1_graph {
            graph.launch(&self.stream)?;
        } else {
            self.submit_b1_operations()?;
        }
        #[cfg(not(feature = "qwen35-augur-http"))]
        self.submit_b1_operations()?;
        self.tokens = next;
        self.poisoned = false;
        Ok(())
    }

    fn submit_b1_operations(&self) -> Result<()> {
        let start = self.submit_input_prefix()?;
        self.submit_complete_ops(start)
    }

    /// Capture only position-independent B1 operations after final norm/QPN setup.
    /// Capture enqueues graph nodes without executing or advancing recurrent state.
    #[cfg(feature = "qwen35-augur-http")]
    pub(crate) fn prepare_b1_graph(&self) -> Result<augur_backend::OwnedCudaGraphExec> {
        if self.tokens != 0 || self.poisoned || self.b1_graph.is_some()
            || self.input_norm_qpn.is_none() || self.post_norm_qpn.is_none() {
            return Err("linear B1 graph requires a fresh finalized fused layer".into());
        }
        let mut submit_error = None;
        let graph = augur_backend::OwnedCudaGraphExec::capture(&self.stream, || {
            self.submit_b1_operations().map_err(|error| {
                submit_error = Some(error);
                augur_backend::CudaRuntimeError::invalid_metadata("checkpoint_linear_b1_graph_capture")
            })
        });
        if let Some(error) = submit_error { return Err(error); }
        Ok(graph?)
    }

    #[cfg(feature = "qwen35-augur-http")]
    pub(crate) fn install_b1_graph(&mut self, graph: augur_backend::OwnedCudaGraphExec) {
        self.b1_graph = Some(graph);
    }

    #[cfg(feature = "qwen35-prefill-cublas")]
    pub(crate) fn ffn_batch_binding(
        &self,
    ) -> Result<(
        crate::checkpoint_ffn_cuda::CheckpointFfnBatchSource<'_>,
        Buffer,
        Buffer,
    )> {
        use crate::checkpoint_ffn_cuda::CheckpointFfnBatchSource;
        match self.operations.as_slice() {
            [.., Operation::QpnFfn(program), Operation::Add(_)] => Ok((
                CheckpointFfnBatchSource::Qpn(program),
                Rc::clone(&self.buffers["post_norm"]),
                Rc::clone(&self.buffers["attention_residual"]),
            )),
            [.., Operation::Projection(gate), Operation::Projection(up), Operation::Silu(_), Operation::Projection(down), Operation::Add(_)] => {
                Ok((
                    CheckpointFfnBatchSource::Exact([gate, up, down]),
                    Rc::clone(&self.buffers["post_norm"]),
                    Rc::clone(&self.buffers["attention_residual"]),
                ))
            }
            _ => Err("checkpoint FFN batch tail contract".into()),
        }
    }

    /// Private partial layer submission. The enclosing target stays poisoned
    /// until it finishes every batch FFN/residual; no partial token is visible.
    #[cfg(feature = "qwen35-prefill-cublas")]
    pub(crate) fn submit_attention_for_ffn_batch(&mut self) -> Result<()> {
        let tail = self.ffn_batch_binding()?.0.tail_operations();
        if self.poisoned {
            return Err("checkpoint layer owner is poisoned".into());
        }
        let next = self.tokens.checked_add(1).ok_or("token counter overflow")?;
        self.poisoned = true;
        let start=self.submit_input_prefix()?;
        for op in &self.operations[start..self.operations.len() - tail] {
            op.submit(&self.stream)?;
        }
        self.tokens = next;
        self.poisoned = false;
        Ok(())
    }

    pub fn synchronize(&mut self) -> Result<()> {
        if let Err(e) = self.stream.synchronize() {
            self.poisoned = true;
            return Err(e.into());
        }
        Ok(())
    }

    /// Explicitly start an empty history; not a continuation or rollback API.
    #[cfg(feature = "qwen35-prefill-cublas")]
    pub(crate) fn prepare_input_norm_qpn(&self) -> Result<augur_backend::cuda_qwen35_prefill_cublas::CudaQwen35NormQpnGroupV1> {
        if self.tokens!=0 || self.poisoned || self.input_norm_qpn.is_some() {
            return Err("fused input staging requires a fresh layer".into());
        }
        let Some(Operation::Norm(norm))=self.operations.first() else {
            return Err("input norm prefix changed".into());
        };
        let mut projections=Vec::with_capacity(2);
        for op in self.operations.get(1..3).ok_or("input projection prefix missing")? {
            let Operation::Projection(projection)=op else {return Err("input projection prefix changed".into())};
            projections.push(projection.retained_qpn()?);
        }
        Ok(augur_backend::cuda_qwen35_prefill_cublas::CudaQwen35NormQpnGroupV1::bind(
            *norm,Rc::clone(&self.buffers["input"]),Rc::clone(&self.buffers["input_norm_weight"]),projections)?)
    }
    #[cfg(feature = "qwen35-prefill-cublas")]
    pub(crate) fn install_input_norm_qpn(&mut self,program:augur_backend::cuda_qwen35_prefill_cublas::CudaQwen35NormQpnGroupV1) {
        self.input_norm_qpn=Some(program);
    }
    fn submit_input_prefix(&self) -> Result<usize> {
        #[cfg(feature = "qwen35-prefill-cublas")]
        if let Some(program)=&self.input_norm_qpn {
            program.submit_for_plan()?;
            return Ok(3);
        }
        Ok(0)
    }

    #[cfg(feature = "qwen35-prefill-cublas")]
    pub(crate) fn prepare_post_norm_qpn(&self) -> Result<augur_backend::cuda_qwen35_prefill_cublas::CudaQwen35NormQpnGroupV1> {
        if self.tokens!=0 || self.poisoned || self.post_norm_qpn.is_some() {
            return Err("fused FFN norm requires a fresh layer".into());
        }
        let [.., Operation::Norm(norm), Operation::QpnFfn(program), Operation::Add(_)] = self.operations.as_slice()
            else {return Err("post norm/QPN FFN tail changed".into());};
        Ok(program.retained().bind_post_norm_group(*norm,
            Rc::clone(&self.buffers["attention_residual"]),Rc::clone(&self.buffers["post_norm_weight"]))?)
    }
    #[cfg(feature = "qwen35-prefill-cublas")]
    pub(crate) fn install_post_norm_qpn(&mut self, program:augur_backend::cuda_qwen35_prefill_cublas::CudaQwen35NormQpnGroupV1) {
        self.post_norm_qpn=Some(program);
    }
    fn submit_complete_ops(&self, start:usize) -> Result<()> {
        #[cfg(feature = "qwen35-prefill-cublas")]
        if let Some(group)=&self.post_norm_qpn {
            let tail=self.operations.len().checked_sub(3).ok_or("short FFN tail")?;
            if start>tail {return Err("overlapping fused FFN prefix".into());}
            let Operation::QpnFfn(program)=&self.operations[tail+1] else {return Err("FFN binding drift".into());};
            for op in &self.operations[start..tail] {op.submit(&self.stream)?;}
            group.submit_for_plan()?;
            program.retained().submit_activation_and_down_on(&self.stream)?;
            self.operations[tail+2].submit(&self.stream)?;
            return Ok(());
        }
        for op in &self.operations[start..] {op.submit(&self.stream)?;}
        Ok(())
    }

    pub(crate) fn norm_bindings_mut(&mut self) -> Vec<&mut CudaRmsnormF32ArgsV1> {
        self.operations.iter_mut().filter_map(|op| match op {
            Operation::Norm(norm) => Some(norm), _ => None,
        }).collect()
    }

    pub fn reset_empty(&mut self) -> Result<()> {
        self.poisoned = true;
        self.stream.synchronize()?;
        for name in ["history", "state"] {
            let buffer = &self.buffers[name];
            // SAFETY: reset has drained this layer's sole execution stream.
            // Retained B8 state/history aliases are inactive; the memset drains
            // the same stream before returning. Preserve complete physical zero,
            // including poisoned/partially submitted state, without host H2D slabs.
            unsafe { buffer.zero_sync_unchecked(&self.stream)?; }
        }
        self.tokens = 0;
        self.poisoned = false;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn readback(&mut self, stage: &str) -> Result<Vec<f32>> {
        self.synchronize()?;
        let buffer = self.buffers.get(stage).ok_or("unknown layer stage")?;
        let mut bytes = vec![0; buffer.len_bytes()];
        buffer.copy_to_host_sync(&self.stream, &mut bytes)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect())
    }
}
impl Drop for CheckpointLinearCudaLayer {
    fn drop(&mut self) {
        // Owner teardown only; the hot submit path has no host wait.
        let _ = self.stream.synchronize();
    }
}

#[cfg(test)]
mod tests;

#[cfg(feature = "qwen35-prefill-cublas")]
pub(crate) mod prefill_prefix;

#[cfg(feature = "qwen35-augur-dflash-bridge")]
pub(crate) mod augur_prefill;

#[cfg(all(windows, feature="qwen35-augur-http"))]
impl CheckpointLinearCudaLayer {
    pub(crate) fn checkpoint_buffers(&self)->[(&'static str,&OwnedCudaBuffer);2] {
        [("conv",&self.buffers["history"]),("gdn",&self.buffers["state"])]
    }
    pub(crate) fn checkpoint_resume(&mut self,consumed:usize)->Result<()> {
        if self.poisoned || self.tokens!=0 || consumed==0 {return Err("GDN checkpoint fresh cursor".into());}
        self.tokens=consumed;Ok(())
    }
}

#[cfg(all(feature = "qwen35-prefill-cublas", feature = "qwen35-nerkyor-dflash-research"))]
pub(crate) mod nerkyor_batch;
