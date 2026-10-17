//! Raw-checkpoint full-attention/FFN and output-head scalar baseline.
//! Compiled only with explicit split-half target RoPE and fused head norm.
//! Explicit short-reference FP32 KV or resident KV8 with shared RoPE tables;
//! a capacity allocation alone does not qualify actual long-history inference.
use crate::checkpoint_ffn_cuda::{CheckpointFfnContext, CheckpointFfnProgram};
use crate::checkpoint_linear_cuda::{empty, upload, view};
use crate::checkpoint_projection_cuda::{
    CheckpointProjectionPolicy, CheckpointScalarCudaProjection, CheckpointScalarCudaWeight,
};
use augur_backend::{
    CudaAddF32ArgsV1, CudaF32View, CudaQwen35FullAttentionArgsV1, CudaQwen35FullAttentionViewsV1,
    CudaQwen35FusedHeadNormRopeArgsV1, CudaQwen35FusedHeadNormRopeViewsV1, CudaRmsnormF32ArgsV1,
    CudaSiluMulF32ArgsV1, OwnedCudaBuffer, OwnedCudaStream, QWEN35_FULL_ATTENTION_MAX_CAPACITY,
};
use augur_core::PhysicalLayoutId;
use augur_model_qwen35::CheckpointArtifact;
use std::{collections::BTreeMap, error::Error, rc::Rc};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
type Buffer = Rc<OwnedCudaBuffer>;
mod kv_reset;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CheckpointKvPolicy {
    #[default]
    F32,
    Q8_0,
}
impl CheckpointKvPolicy {
    pub fn label(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::Q8_0 => "Q8_0",
        }
    }
    pub fn max_capacity(self) -> Result<usize> {
        match self {
            Self::F32 => Ok(QWEN35_FULL_ATTENTION_MAX_CAPACITY),
            Self::Q8_0 => {
                #[cfg(feature = "qwen35-kv8-full-attention")]
                {
                    Ok(augur_backend::QWEN35_KV8_FULL_ATTENTION_MAX_CAPACITY)
                }
                #[cfg(not(feature = "qwen35-kv8-full-attention"))]
                {
                    Err("checkpoint KV8 feature absent".into())
                }
            }
        }
    }
    pub fn row_bytes(self) -> usize {
        match self {
            Self::F32 => 1024 * 4,
            Self::Q8_0 => 1088,
        }
    }
}

/// One immutable absolute-position table shared by all16 attention layers.
/// Weight-dependent Q/K normalization remains owned by each layer.
pub(crate) struct CheckpointRopeTables {
    stream: Rc<OwnedCudaStream>,
    cos: Buffer,
    sin: Buffer,
    capacity: usize,
    absolute_start: usize,
    theta: f64,
    #[cfg(feature = "qwen35-augur-vision")]
    text: text_rope::RetainedTextRope,
}
impl CheckpointRopeTables {
    pub(crate) fn new(
        reader: &CheckpointArtifact,
        stream: Rc<OwnedCudaStream>,
        capacity: usize,
        absolute_start: usize,
    ) -> Result<Self> {
        if capacity == 0
            || reader.config().rope_dimension != 64
            || absolute_start
                .checked_add(capacity)
                .is_none_or(|n| n > reader.config().max_position_embeddings)
        {
            return Err("checkpoint RoPE table bounds".into());
        }
        let theta = reader.config().rope_theta;
        let mut cos = Vec::with_capacity(capacity * 32);
        let mut sin = Vec::with_capacity(capacity * 32);
        for position in absolute_start..absolute_start + capacity {
            for pair in 0..32 {
                let inverse = theta.powf(-2.0 * pair as f64 / 64.0) as f32;
                let phase = position as f32 * inverse;
                cos.push(phase.cos());
                sin.push(phase.sin());
            }
        }
        #[cfg(feature = "qwen35-augur-vision")]
        let text = text_rope::RetainedTextRope::from_f32(&cos, &sin)?;
        #[cfg(feature = "qwen35-augur-vision")]
        let (cos, sin) = text.upload(&stream)?;
        #[cfg(not(feature = "qwen35-augur-vision"))]
        let (cos, sin) = (upload(&stream, &cos)?, upload(&stream, &sin)?);
        Ok(Self {
            cos,
            sin,
            stream,
            capacity,
            absolute_start,
            theta,
            #[cfg(feature = "qwen35-augur-vision")]
            text,
        })
    }
}

#[cfg(feature = "qwen35-augur-vision")]
impl CheckpointRopeTables {
    pub(crate) fn install_image(&self, plan:&augur_model_qwen35::Qwen35ImageMropePlan)->Result<()> {
        if self.absolute_start!=0 || plan.capacity()!=self.capacity || self.theta!=10_000_000.0 {
            return Err("image MRoPE table owner/capacity/theta mismatch".into());
        }
        let (cos,sin)=plan.tables()?;
        self.cos.copy_from_host_sync(&self.stream,&cos.iter().flat_map(|x|x.to_le_bytes()).collect::<Vec<_>>())?;
        self.sin.copy_from_host_sync(&self.stream,&sin.iter().flat_map(|x|x.to_le_bytes()).collect::<Vec<_>>())?;
        Ok(())
    }
    pub(crate) fn restore_text(&self)->Result<()> {
        // Keep the bound GPU allocations and restore their exact startup bytes.
        self.text.restore_into(&self.stream, &self.cos, &self.sin)
    }
}

#[cfg(feature = "qwen35-augur-vision")]
mod text_rope;

fn subview(buffer: &OwnedCudaBuffer, first: usize, count: usize) -> Result<CudaF32View> {
    Ok(CudaF32View::new(
        buffer.view(
            first.checked_mul(4).ok_or("offset overflow")?,
            count.checked_mul(4).ok_or("length overflow")?,
        )?,
        PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
        count,
    )?)
}
enum Operation {
    Norm(CudaRmsnormF32ArgsV1),
    Projection(CheckpointScalarCudaProjection),
    Add(CudaAddF32ArgsV1),
    Silu(CudaSiluMulF32ArgsV1),
    QpnFfn(CheckpointFfnProgram),
}
impl Operation {
    fn submit(&self, stream: &OwnedCudaStream) -> Result<()> {
        match self {
            Self::Norm(op) => op.submit_for_plan(stream)?,
            Self::Projection(op) => op.submit_for_plan()?,
            Self::Add(op) => op.submit_for_plan(stream)?,
            Self::Silu(op) => op.submit_for_plan(stream)?,
            Self::QpnFfn(op) => op.submit(stream)?,
        }
        Ok(())
    }
}
struct Step {
    heads: CudaQwen35FusedHeadNormRopeArgsV1,
    attention: CudaQwen35FullAttentionArgsV1,
}
pub struct CheckpointFullCudaLayer {
    // Graphs borrow operation addresses and must drop before their owners.
    #[cfg(feature = "qwen35-augur-http")]
    b1_graphs: Option<b1_graph::FullB1Graphs>,
    #[cfg(feature = "qwen35-prefill-cublas")]
    input_norm_qpn: Option<augur_backend::cuda_qwen35_prefill_cublas::CudaQwen35NormQpnGroupV1>,
    #[cfg(feature = "qwen35-prefill-cublas")]
    post_norm_qpn: Option<augur_backend::cuda_qwen35_prefill_cublas::CudaQwen35NormQpnGroupV1>,
    stream: Rc<OwnedCudaStream>,
    input_ops: Vec<Operation>,
    output_ops: Vec<Operation>,
    steps: Vec<Step>,
    capacity: usize,
    source_epsilon: f64,
    rope_theta: f64,
    absolute_start: usize,
    #[cfg(feature = "qwen35-kv8-full-attention")]
    kv8: Option<augur_backend::CudaQwen35Kv8FullAttentionArgsV3>,
    #[cfg(feature = "qwen35-kv8-full-attention")]
    heads_template: CudaQwen35FusedHeadNormRopeViewsV1,
    #[cfg(feature = "qwen35-kv8-full-attention")]
    epsilon: f32,
    buffers: BTreeMap<&'static str, Buffer>,
    kv_dirty: kv_reset::DirtyPrefix,
    tokens: usize,
    poisoned: bool,
}
impl CheckpointFullCudaLayer {
    pub fn load(
        reader: &mut CheckpointArtifact,
        layer: usize,
        capacity: usize,
        absolute_start: usize,
        stream: Rc<OwnedCudaStream>,
        input: Buffer,
        output: Buffer,
    ) -> Result<Self> {
        Self::load_with_policy(
            reader,
            layer,
            capacity,
            absolute_start,
            stream,
            input,
            output,
            CheckpointProjectionPolicy::OriginalF32,
        )
    }
    pub fn load_with_policy(
        reader: &mut CheckpointArtifact,
        layer: usize,
        capacity: usize,
        absolute_start: usize,
        stream: Rc<OwnedCudaStream>,
        input: Buffer,
        output: Buffer,
        policy: CheckpointProjectionPolicy,
    ) -> Result<Self> {
        let ffn = CheckpointFfnContext::new(Rc::clone(&stream), policy)?;
        Self::load_with_ffn_context(
            reader,
            layer,
            capacity,
            absolute_start,
            stream,
            input,
            output,
            &ffn,
        )
    }
    pub(crate) fn load_with_ffn_context(
        reader: &mut CheckpointArtifact,
        layer: usize,
        capacity: usize,
        absolute_start: usize,
        stream: Rc<OwnedCudaStream>,
        input: Buffer,
        output: Buffer,
        ffn: &CheckpointFfnContext,
    ) -> Result<Self> {
        Self::load_with_cache_context(
            reader,
            layer,
            capacity,
            absolute_start,
            stream,
            input,
            output,
            ffn,
            CheckpointKvPolicy::F32,
            None,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load_with_cache_context(
        reader: &mut CheckpointArtifact,
        layer: usize,
        capacity: usize,
        absolute_start: usize,
        stream: Rc<OwnedCudaStream>,
        input: Buffer,
        output: Buffer,
        ffn: &CheckpointFfnContext,
        kv_policy: CheckpointKvPolicy,
        shared_rope: Option<&CheckpointRopeTables>,
    ) -> Result<Self> {
        let policy = ffn.policy();
        if layer >= 64 || (layer + 1) % 4 != 0 {
            return Err("expected a full-attention layer".into());
        }
        if capacity == 0
            || capacity > kv_policy.max_capacity()?
            || absolute_start
                .checked_add(capacity)
                .is_none_or(|n| n > reader.config().max_position_embeddings)
        {
            return Err("checkpoint KV/absolute-position bounds".into());
        }
        if input.len_bytes() != 5120 * 4
            || output.len_bytes() != 5120 * 4
            || Rc::ptr_eq(&input, &output)
        {
            return Err("full layer requires distinct exact hidden buffers".into());
        }
        let p = format!("model.language_model.layers.{layer}");
        let mut b = BTreeMap::from([("input", input), ("output", output)]);
        for (name, n) in [
            ("normalized", 5120),
            ("q_full", 12288),
            ("q_normed", 6144),
            ("query", 6144),
            ("k_raw", 1024),
            ("k_normed", 1024),
            ("key", 1024),
            ("value", 1024),
            ("attention", 6144),
            ("gated", 6144),
            ("attention_output", 5120),
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
            b.insert(name, empty(&stream, n)?);
        }
        for name in ["key_cache", "value_cache"] {
            b.insert(name, empty(&stream, capacity * kv_policy.row_bytes() / 4)?);
        }
        #[cfg(feature = "qwen35-kv8-full-attention")]
        if kv_policy == CheckpointKvPolicy::Q8_0 {
            b.insert(
                "kv8_workspace",
                empty(
                    &stream,
                    augur_backend::QWEN35_FULL_ATTENTION_WORKSPACE_ELEMENTS_V2,
                )?,
            );
        }
        for (name, suffix, repeats) in [
            ("input_norm_weight", "input_layernorm.weight", 1),
            ("post_norm_weight", "post_attention_layernorm.weight", 1),
            ("q_norm_weight", "self_attn.q_norm.weight", 24),
            ("k_norm_weight", "self_attn.k_norm.weight", 4),
        ] {
            b.insert(
                name,
                upload(
                    &stream,
                    &reader
                        .load_dense_execution(&format!("{p}.{suffix}"))?
                        .repeat(repeats),
                )?,
            );
        }
        // Frequency rounded once to F32, then F32 absolute-position phase.
        // This is recorded separately from the F64 whole-layer reference.
        let owned_rope;
        let rope = if let Some(shared) = shared_rope {
            shared
        } else {
            owned_rope =
                CheckpointRopeTables::new(reader, Rc::clone(&stream), capacity, absolute_start)?;
            &owned_rope
        };
        if !Rc::ptr_eq(&stream, &rope.stream)
            || rope.capacity != capacity
            || rope.absolute_start != absolute_start
            || rope.theta.to_bits() != reader.config().rope_theta.to_bits()
        {
            return Err("shared RoPE table identity drift".into());
        }
        b.insert("rope_cos", Rc::clone(&rope.cos));
        b.insert("rope_sin", Rc::clone(&rope.sin));
        let v = |name| view(&b[name]);
        let eps = reader.config().rms_norm_eps as f32;
        let projection = |reader: &mut CheckpointArtifact,
                          suffix: &str,
                          input: &str,
                          output: &str|
         -> Result<Operation> {
            let source = reader.load_linear(&format!("{p}.{suffix}"))?;
            let source = if suffix == "self_attn.q_proj" {
                source.separate_query_gate_rows()?
            } else {
                source
            };
            let weight =
                ffn.upload_projection(Rc::clone(&stream), source)?;
            Ok(Operation::Projection(weight.bind(
                Rc::clone(&b[input]),
                0,
                Rc::clone(&b[output]),
                0,
            )?))
        };
        let mut input_ops = vec![Operation::Norm(CudaRmsnormF32ArgsV1::bind(
            &stream,
            v("input")?,
            v("input_norm_weight")?,
            v("normalized")?,
            eps,
        )?)];
        for (suffix, out) in [
            ("q_proj", "q_full"),
            ("k_proj", "k_raw"),
            ("v_proj", "value"),
        ] {
            input_ops.push(projection(
                reader,
                &format!("self_attn.{suffix}"),
                "normalized",
                out,
            )?);
        }
        #[cfg(feature = "qwen35-kv8-full-attention")]
        let heads_template = CudaQwen35FusedHeadNormRopeViewsV1 {
            q_input: subview(&b["q_full"], 0, 6144)?,
            q_weight: v("q_norm_weight")?,
            q_normed: v("q_normed")?,
            q_output: v("query")?,
            k_input: v("k_raw")?,
            k_weight: v("k_norm_weight")?,
            k_normed: v("k_normed")?,
            k_output: v("key")?,
            rope_cos: subview(&b["rope_cos"], 0, 32)?,
            rope_sin: subview(&b["rope_sin"], 0, 32)?,
        };
        let mut steps = Vec::new();
        if kv_policy == CheckpointKvPolicy::F32 {
            for position in 0..capacity {
                let heads = CudaQwen35FusedHeadNormRopeArgsV1::bind(
                    &stream,
                    CudaQwen35FusedHeadNormRopeViewsV1 {
                        q_input: subview(&b["q_full"], 0, 6144)?,
                        q_weight: v("q_norm_weight")?,
                        q_normed: v("q_normed")?,
                        q_output: v("query")?,
                        k_input: v("k_raw")?,
                        k_weight: v("k_norm_weight")?,
                        k_normed: v("k_normed")?,
                        k_output: v("key")?,
                        rope_cos: subview(&b["rope_cos"], position * 32, 32)?,
                        rope_sin: subview(&b["rope_sin"], position * 32, 32)?,
                    },
                    eps,
                )?;
                let attention = CudaQwen35FullAttentionArgsV1::bind(
                    &stream,
                    CudaQwen35FullAttentionViewsV1 {
                        query: v("query")?,
                        gate: subview(&b["q_full"], 6144, 6144)?,
                        key: v("key")?,
                        value: v("value")?,
                        key_cache: v("key_cache")?,
                        value_cache: v("value_cache")?,
                        attention: v("attention")?,
                        gated: v("gated")?,
                    },
                    capacity,
                    position,
                )?;
                steps.push(Step { heads, attention });
            }
        }
        #[cfg(feature = "qwen35-kv8-full-attention")]
        let kv8 = if kv_policy == CheckpointKvPolicy::Q8_0 {
            Some(augur_backend::CudaQwen35Kv8FullAttentionArgsV3::bind(
                &stream,
                augur_backend::CudaQwen35Kv8FullAttentionViewsV3 {
                    query: v("query")?,
                    gate: subview(&b["q_full"], 6144, 6144)?,
                    key: v("key")?,
                    value: v("value")?,
                    key_cache: b["key_cache"].view(0, b["key_cache"].len_bytes())?,
                    value_cache: b["value_cache"].view(0, b["value_cache"].len_bytes())?,
                    attention: v("attention")?,
                    gated: v("gated")?,
                    workspace: v("kv8_workspace")?,
                },
                capacity,
                0,
            )?)
        } else {
            None
        };
        let mut output_ops = vec![
            projection(reader, "self_attn.o_proj", "gated", "attention_output")?,
            Operation::Add(CudaAddF32ArgsV1::bind(
                &stream,
                v("input")?,
                v("attention_output")?,
                v("attention_residual")?,
            )?),
            Operation::Norm(CudaRmsnormF32ArgsV1::bind(
                &stream,
                v("attention_residual")?,
                v("post_norm_weight")?,
                v("post_norm")?,
                eps,
            )?),
        ];
        if policy.qpn_ffn_layer(layer) {
            output_ops.push(Operation::QpnFfn(ffn.bind(
                reader,
                layer,
                &stream,
                Rc::clone(&b["post_norm"]),
                Rc::clone(&b["ffn_output"]),
            )?));
        } else {
            output_ops.push(projection(reader, "mlp.gate_proj", "post_norm", "gate")?);
            output_ops.push(projection(reader, "mlp.up_proj", "post_norm", "up")?);
            output_ops.push(Operation::Silu(CudaSiluMulF32ArgsV1::bind(
                &stream,
                v("gate")?,
                v("up")?,
                v("swiglu")?,
            )?));
            output_ops.push(projection(reader, "mlp.down_proj", "swiglu", "ffn_output")?);
        }
        output_ops.push(Operation::Add(CudaAddF32ArgsV1::bind(
            &stream,
            v("attention_residual")?,
            v("ffn_output")?,
            v("output")?,
        )?));
        Ok(Self {
            stream,
            input_ops,
            #[cfg(feature = "qwen35-augur-http")]
            b1_graphs: None,
            #[cfg(feature = "qwen35-prefill-cublas")]
            input_norm_qpn: None,
            #[cfg(feature = "qwen35-prefill-cublas")]
            post_norm_qpn: None,
            output_ops,
            steps,
            capacity,
            source_epsilon: reader.config().rms_norm_eps,
            rope_theta: reader.config().rope_theta,
            absolute_start,
            #[cfg(feature = "qwen35-kv8-full-attention")]
            kv8,
            #[cfg(feature = "qwen35-kv8-full-attention")]
            heads_template,
            #[cfg(feature = "qwen35-kv8-full-attention")]
            epsilon: eps,
            buffers: b,
            kv_dirty: kv_reset::DirtyPrefix::new(capacity, kv_policy.row_bytes())?,
            tokens: 0,
            poisoned: false,
        })
    }
    #[cfg(feature = "qwen35-prefill-cublas")]
    pub(crate) fn prepare_input_norm_qpn(&self) -> Result<augur_backend::cuda_qwen35_prefill_cublas::CudaQwen35NormQpnGroupV1> {
        if self.tokens!=0 || self.poisoned || self.input_norm_qpn.is_some() {
            return Err("fused input staging requires a fresh layer".into());
        }
        let Some(Operation::Norm(norm))=self.input_ops.first() else {
            return Err("input norm prefix changed".into());
        };
        let mut projections=Vec::with_capacity(3);
        for op in self.input_ops.get(1..4).ok_or("input projection prefix missing")? {
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
            return Ok(4);
        }
        Ok(0)
    }

    #[cfg(feature = "qwen35-prefill-cublas")]
    pub(crate) fn prepare_post_norm_qpn(&self) -> Result<augur_backend::cuda_qwen35_prefill_cublas::CudaQwen35NormQpnGroupV1> {
        if self.tokens!=0 || self.poisoned || self.post_norm_qpn.is_some() {
            return Err("fused FFN norm requires a fresh layer".into());
        }
        let [.., Operation::Norm(norm), Operation::QpnFfn(program), Operation::Add(_)] = self.output_ops.as_slice()
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
            let tail=self.output_ops.len().checked_sub(3).ok_or("short FFN tail")?;
            if start>tail {return Err("overlapping fused FFN prefix".into());}
            let Operation::QpnFfn(program)=&self.output_ops[tail+1] else {return Err("FFN binding drift".into());};
            for op in &self.output_ops[start..tail] {op.submit(&self.stream)?;}
            group.submit_for_plan()?;
            program.retained().submit_activation_and_down_on(&self.stream)?;
            self.output_ops[tail+2].submit(&self.stream)?;
            return Ok(());
        }
        for op in &self.output_ops[start..] {op.submit(&self.stream)?;}
        Ok(())
    }

    pub(crate) fn norm_bindings_mut(&mut self) -> Vec<&mut CudaRmsnormF32ArgsV1> {
        self.input_ops.iter_mut().chain(self.output_ops.iter_mut()).filter_map(|op| match op {
            Operation::Norm(norm) => Some(norm), _ => None,
        }).collect()
    }

    /// Test-only actual attention-input capture, outside every timed lane.
    #[cfg(test)]
    pub(crate) fn diagnostic_normalized_buffer(&self) -> &OwnedCudaBuffer {
        &self.buffers["normalized"]
    }

    pub fn submitted_tokens(&self) -> usize {
        self.tokens
    }
    pub fn submit_for_plan(&mut self) -> Result<()> {
        if self.poisoned || self.tokens >= self.capacity {
            return Err("poisoned/full checkpoint KV owner".into());
        }
        self.kv_dirty.mark(self.tokens, 1, 1)?;
        self.poisoned = true;
        let start=self.submit_input_prefix()?;
        for op in &self.input_ops[start..] {
            op.submit(&self.stream)?;
        }
        self.submit_attention_position()?;
        self.submit_complete_ops(0)?;
        self.tokens += 1;
        self.poisoned = false;
        Ok(())
    }
    fn submit_attention_position(&self) -> Result<()> {
        #[cfg(feature = "qwen35-kv8-full-attention")]
        if let Some(attention) = self.kv8 {
            let mut heads = self.heads_template;
            heads.rope_cos = subview(&self.buffers["rope_cos"], self.tokens * 32, 32)?;
            heads.rope_sin = subview(&self.buffers["rope_sin"], self.tokens * 32, 32)?;
            CudaQwen35FusedHeadNormRopeArgsV1::bind(&self.stream, heads, self.epsilon)?
                .submit_for_plan(&self.stream)?;
            attention
                .at_position(self.tokens)?
                .submit_for_plan(&self.stream)?;
            return Ok(());
        }
        self.steps[self.tokens]
            .heads
            .submit_for_plan(&self.stream)?;
        self.steps[self.tokens]
            .attention
            .submit_for_plan(&self.stream)?;
        Ok(())
    }
    pub fn synchronize(&mut self) -> Result<()> {
        if let Err(e) = self.stream.synchronize() {
            self.poisoned = true;
            return Err(e.into());
        }
        Ok(())
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
        match self.output_ops.as_slice() {
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
    #[cfg(feature = "qwen35-prefill-cublas")]
    pub(crate) fn submit_attention_for_ffn_batch(&mut self) -> Result<()> {
        let tail = self.ffn_batch_binding()?.0.tail_operations();
        if self.poisoned || self.tokens >= self.capacity {
            return Err("poisoned/full checkpoint KV owner".into());
        }
        self.kv_dirty.mark(self.tokens, 1, 1)?;
        self.poisoned = true;
        let start=self.submit_input_prefix()?;
        for op in &self.input_ops[start..] {
            op.submit(&self.stream)?;
        }
        self.submit_attention_position()?;
        for op in &self.output_ops[..self.output_ops.len() - tail] {
            op.submit(&self.stream)?;
        }
        self.tokens += 1;
        self.poisoned = false;
        Ok(())
    }
    pub fn reset_empty(&mut self) -> Result<()> {
        kv_reset::reset_buffers(
            &self.stream,
            &mut self.buffers,
            &mut self.kv_dirty,
            &mut self.tokens,
            &mut self.poisoned,
        )
    }
    #[cfg(test)]
    pub(crate) fn readback(&mut self, stage: &str) -> Result<Vec<f32>> {
        self.synchronize()?;
        let buffer = self.buffers.get(stage).ok_or("unknown stage")?;
        let mut bytes = vec![0; buffer.len_bytes()];
        buffer.copy_to_host_sync(&self.stream, &mut bytes)?;
        #[cfg(feature = "qwen35-kv8-full-attention")]
        if self.kv8.is_some() && ["key_cache", "value_cache"].contains(&stage) {
            let mut decoded = vec![0.; bytes.len() / 34 * 32];
            for (block, out) in bytes.chunks_exact(34).zip(decoded.chunks_exact_mut(32)) {
                augur_model_qwen35::dequantize_q8_0_block(block, out)?;
            }
            return Ok(decoded);
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
            .collect())
    }
}
impl Drop for CheckpointFullCudaLayer {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
    }
}

pub struct CheckpointCudaOutputHead {
    #[cfg(feature = "qwen35-augur-http")]
    b1_graph: Option<augur_backend::OwnedCudaGraphExec>,
    stream: Rc<OwnedCudaStream>,
    norm: CudaRmsnormF32ArgsV1,
    projection: CheckpointScalarCudaProjection,
    _input: Buffer,
    _norm_weight: Buffer,
    normalized: Buffer,
}
impl CheckpointCudaOutputHead {


    #[cfg(feature = "qwen35-augur-dflash-bridge")]
    pub(crate) fn bind_shared_normalized_bf16(
        &self, stream: &Rc<OwnedCudaStream>, input: Buffer, input_offset: usize,
        output: Buffer, output_offset: usize,
    ) -> Result<CheckpointScalarCudaProjection> {
        self.projection.rebind_shared_bf16_head(stream, input, input_offset, output, output_offset)
    }
    #[cfg(all(feature="qwen35-prefill-cublas",feature="qwen35-augur-dflash-bridge"))]
    pub(crate) fn bind_shared_normalized_bf16_rows(&self,stream:&Rc<OwnedCudaStream>,input:Buffer,output:Buffer,rows:usize)
        -> Result<crate::checkpoint_projection_cuda::CheckpointSharedBf16HeadRows> {
        self.projection.rebind_shared_bf16_head_rows(stream,input,output,rows)
    }
    #[cfg(all(feature="qwen35-prefill-cublas",feature="qwen35-augur-dflash-bridge"))]
    pub(crate) fn bind_shared_normalized_bf16_b8(&self,stream:&Rc<OwnedCudaStream>,input:Buffer,output:Buffer)
        -> Result<crate::checkpoint_projection_cuda::CheckpointSharedBf16HeadRows> {
        self.bind_shared_normalized_bf16_rows(stream,input,output,8)
    }
    #[cfg(feature="qwen35-augur-dflash-bridge")]
    pub(crate) fn bind_target_norm_rows(&self,input:&OwnedCudaBuffer,output:&OwnedCudaBuffer,epsilon:f32,rows:usize)
        -> Result<augur_backend::cuda_qwen35_prefill_cublas::CudaQwen35PrefillRmsnormV1> {
        if !(1..=8).contains(&rows) { return Err("target norm row width must be in 1..=8".into()); }
        Ok(augur_backend::cuda_qwen35_prefill_cublas::CudaQwen35PrefillRmsnormV1::bind(&self.stream,
            view(input)?,view(&self._norm_weight)?,view(output)?,rows,5120,epsilon)?)
    }
    #[cfg(feature="qwen35-augur-dflash-bridge")]
    pub(crate) fn bind_b8_target_norm(&self,input:&OwnedCudaBuffer,output:&OwnedCudaBuffer,epsilon:f32)
        -> Result<augur_backend::cuda_qwen35_prefill_cublas::CudaQwen35PrefillRmsnormV1> {
        self.bind_target_norm_rows(input,output,epsilon,8)
    }
    pub(crate) fn norm_binding_mut(&mut self) -> &mut CudaRmsnormF32ArgsV1 { &mut self.norm }
    pub fn load(
        reader: &mut CheckpointArtifact,
        stream: Rc<OwnedCudaStream>,
        input: Buffer,
        logits: Buffer,
    ) -> Result<Self> {
        Self::load_timed(reader, stream, input, logits).map(|(head, _)| head)
    }
    pub(crate) fn load_timed(
        reader: &mut CheckpointArtifact,
        stream: Rc<OwnedCudaStream>,
        input: Buffer,
        logits: Buffer,
    ) -> Result<(Self, serde_json::Value)> {
        let started = std::time::Instant::now();
        if input.len_bytes() != 5120 * 4
            || logits.len_bytes() != 248320 * 4
            || Rc::ptr_eq(&input, &logits)
        {
            return Err("output head requires exact hidden/vocabulary buffers".into());
        }
        let norm_weight = upload(
            &stream,
            &reader.load_dense_execution("model.language_model.norm.weight")?,
        )?;
        let normalized = empty(&stream, 5120)?;
        let norm = CudaRmsnormF32ArgsV1::bind(
            &stream,
            view(&input)?,
            view(&norm_weight)?,
            view(&normalized)?,
            reader.config().rms_norm_eps as f32,
        )?;
        let norm_setup_s = started.elapsed().as_secs_f64();
        #[cfg(feature = "qwen35-augur-http")]
        let streamed = crate::checkpoint_projection_cuda::startup_bf16::enabled()?;
        #[cfg(not(feature = "qwen35-augur-http"))]
        let streamed = false;
        let (weight, transfer_detail) = if streamed {
            #[cfg(feature = "qwen35-augur-http")]
            { CheckpointScalarCudaWeight::load_streamed_bf16(reader, Rc::clone(&stream), "lm_head")? }
            #[cfg(not(feature = "qwen35-augur-http"))]
            { unreachable!() }
        } else {
            let tick = std::time::Instant::now();
            let source = reader.load_linear("lm_head")?;
            let pinned_payload_read_s = tick.elapsed().as_secs_f64();
            let tick = std::time::Instant::now();
            let weight = CheckpointScalarCudaWeight::upload(Rc::clone(&stream), source)?;
            (weight, serde_json::json!({"mode":"authenticated_bf16_pageable_v1",
                "pinned_payload_read_s":pinned_payload_read_s,
                "validate_allocate_upload_s":tick.elapsed().as_secs_f64()}))
        };
        let projection = weight.bind(Rc::clone(&normalized), 0, logits, 0)?;
        let detail = serde_json::json!({"norm_setup_s":norm_setup_s,
            "transfer":transfer_detail,
            "complete_s":started.elapsed().as_secs_f64()});
        Ok((Self {
            #[cfg(feature = "qwen35-augur-http")]
            b1_graph: None,
            stream,
            norm,
            projection,
            _input: input,
            _norm_weight: norm_weight,
            normalized,
        }, detail))
    }
    pub fn submit_for_plan(&self) -> Result<()> {
        #[cfg(feature = "qwen35-augur-http")]
        if let Some(graph) = &self.b1_graph {
            graph.launch(&self.stream)?;
            return Ok(());
        }
        self.submit_head_operations()
    }
    fn submit_head_operations(&self) -> Result<()> {
        self.norm.submit_for_plan(&self.stream)?;
        self.projection.submit_for_plan()?;
        Ok(())
    }
    #[cfg(test)]
    fn normalized(&self) -> Result<Vec<f32>> {
        let mut bytes = vec![0; 5120 * 4];
        self.normalized
            .copy_to_host_sync(&self.stream, &mut bytes)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
            .collect())
    }
}
impl Drop for CheckpointCudaOutputHead {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
    }
}

#[cfg(test)]
mod tests;

#[cfg(feature = "qwen35-augur-dflash-bridge")]
pub(crate) mod augur_prefill;
#[cfg(feature = "qwen35-augur-http")]
pub(crate) mod augur_decode;

#[cfg(all(windows, feature="qwen35-augur-http"))]
impl CheckpointFullCudaLayer {
    pub(crate) fn checkpoint_buffers(&self)->[(&'static str,&OwnedCudaBuffer);2] {
        [("key",&self.buffers["key_cache"]),("value",&self.buffers["value_cache"])]
    }
    pub(crate) fn checkpoint_prepare_restore(&mut self,consumed:usize)->Result<()> {
        if self.poisoned || self.tokens!=0 || consumed==0 || consumed>=self.capacity {
            return Err("KV checkpoint fresh restore span".into());}
        // Direct checkpoint copies can fail before checkpoint_resume commits a
        // cursor. Reset must already know every physical row they may touch.
        self.kv_dirty.mark(0,consumed,256)
    }
    pub(crate) fn checkpoint_resume(&mut self,consumed:usize)->Result<()> {
        if self.poisoned || self.tokens!=0 || consumed==0 || consumed>=self.capacity {
            return Err("KV checkpoint fresh cursor".into());}
        self.kv_dirty.mark(0,consumed,256)?;self.tokens=consumed;Ok(())
    }
}

#[cfg(feature = "qwen35-augur-http")]
mod b1_graph;
