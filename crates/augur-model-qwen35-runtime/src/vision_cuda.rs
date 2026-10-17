//! Opt-in, bounded Qwen3.5 Vision encoder. Not a language-model image ingress.
//! FP16 affine storage/inputs, FP32 affine outputs and control. Workspace owners
//! are request-local; learned weights and the stream/handle stay encoder-resident.
use augur_backend::{
    CudaBufferView, CudaF32View, CudaQwen35PrefillCublasPlanV1, CudaQwen35VisionAffineViewsV1,
    CudaQwen35VisionControlOp as Op, CudaQwen35VisionControlV1, CudaQwen35VisionControlViews,
    OwnedCudaBuffer, OwnedCudaStream,
};
use augur_core::PhysicalLayoutId;
use augur_model_qwen35::{
    convert_qwen35_vision_bf16, Qwen35PreparedF16Image, Qwen35VisionGeometry, Qwen35VisionProfile,
    Qwen35VisionStorage, Qwen35VisionWeightMap, Qwen35VisionWorkspacePlan, SafetensorsManifest,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, error::Error, path::Path};
type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;
const PREFIX: &str = "model.visual.";

#[derive(Clone, Debug)]
struct Tensor {
    offset: usize,
    bytes: usize,
    shape: Vec<usize>,
    storage: Qwen35VisionStorage,
}

pub struct Qwen35VisionCudaEncoder {
    tensors: BTreeMap<String, Tensor>,
    weights: OwnedCudaBuffer,
    plan: CudaQwen35PrefillCublasPlanV1,
    stream: OwnedCudaStream,
    config_sha256: String,
    source_selected_sha256: String,
    converted_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct Qwen35VisionCudaOutput {
    pub grid_thw: [usize; 3],
    pub merged_tokens: usize,
    pub hidden_size: usize,
    #[serde(skip_serializing)]
    pub embeddings_f32: Vec<f32>,
    pub prepared_payload_sha256: String,
    pub source_config_sha256: String,
    pub source_selected_payload_sha256: String,
    pub converted_weights_sha256: String,
    pub explicit_workspace_bytes: usize,
    pub encoder_wall_ms: f64,
    pub executed_blocks: usize,
    pub vision_forward_executed: bool,
    pub target_image_consumed: bool,
    pub product_ready: bool,
}
struct Drain<'a> {
    stream: &'a OwnedCudaStream,
    pending: bool,
}
impl Drop for Drain<'_> {
    fn drop(&mut self) {
        if self.pending {
            let _ = self.stream.synchronize();
        }
    }
}
impl Qwen35VisionCudaEncoder {
    /// Reads only the 333 fixed Vision tensors, converts once, and uploads one
    /// packed owner. An expected selected-payload digest is optional for local
    /// development; qualification callers pass their independent source anchor.
    pub fn load(
        config: &[u8],
        source: &Path,
        expected_selected_sha256: Option<&str>,
        ordinal: i32,
    ) -> Result<Self> {
        let profile = Qwen35VisionProfile::from_config_bytes(config)?;
        let manifest = SafetensorsManifest::inspect(source, false)?;
        let map = Qwen35VisionWeightMap::from_manifest(&profile, &manifest)?;
        let mut packed =
            Vec::with_capacity(map.selected_payload_bytes() as usize + 16 * 1024 * 1024);
        let mut tensors = BTreeMap::new();
        let mut source_hash = Sha256::new();
        for tensor in map.bindings() {
            let raw = manifest.read_tensor(&tensor.name)?;
            source_hash.update(&raw);
            let storage = Qwen35VisionStorage::for_binding(tensor);
            let converted = convert_qwen35_vision_bf16(&raw, storage, 128 * 1024 * 1024)?;
            let descriptor = Tensor {
                offset: packed.len(),
                bytes: converted.bytes.len(),
                shape: tensor.shape.iter().map(|&x| x as usize).collect(),
                storage,
            };
            packed.extend_from_slice(&converted.bytes);
            tensors.insert(tensor.name.clone(), descriptor);
        }
        let source_selected_sha256 = format!("{:x}", source_hash.finalize());
        if expected_selected_sha256.is_some_and(|expected| expected != source_selected_sha256) {
            return Err("Vision selected-source digest mismatch".into());
        }
        let after = SafetensorsManifest::inspect(source, false)?;
        let after_map = Qwen35VisionWeightMap::from_manifest(&profile, &after)?;
        if serde_json::to_vec(&map)? != serde_json::to_vec(&after_map)?
            || manifest.file_size != after.file_size
        {
            return Err("Vision source descriptor changed during load".into());
        }
        let converted_sha256 = format!("{:x}", Sha256::digest(&packed));
        let stream = OwnedCudaStream::create(ordinal)?;
        let weights = OwnedCudaBuffer::allocate(ordinal, packed.len())?;
        weights.copy_from_host_sync(&stream, &packed)?;
        let plan = CudaQwen35PrefillCublasPlanV1::new_for_resident_vision_affine(&stream, 4096)?;
        Ok(Self {
            tensors,
            weights,
            plan,
            stream,
            config_sha256: profile.config_sha256().to_owned(),
            source_selected_sha256,
            converted_sha256,
        })
    }
    fn tensor(&self, name: &str) -> Result<(CudaBufferView, &Tensor)> {
        let tensor = self
            .tensors
            .get(&(PREFIX.to_owned() + name))
            .ok_or("missing fixed Vision tensor")?;
        Ok((self.weights.view(tensor.offset, tensor.bytes)?, tensor))
    }
    fn vector(&self, name: &str) -> Result<CudaBufferView> {
        let (view, tensor) = self.tensor(name)?;
        if tensor.storage != Qwen35VisionStorage::F32 {
            return Err("Vision vector/position storage drift".into());
        }
        Ok(view)
    }
    fn affine(
        &self,
        name: &str,
        tokens: usize,
        input: CudaBufferView,
        output: CudaBufferView,
    ) -> Result<()> {
        let (weight, tensor) = self.tensor(&(name.to_owned() + ".weight"))?;
        if tensor.storage != Qwen35VisionStorage::F16 {
            return Err("Vision affine storage drift".into());
        }
        let rows = tensor.shape[0];
        let cols = tensor.shape[1..].iter().product();
        let bias = self.vector(&(name.to_owned() + ".bias"))?;
        let views = CudaQwen35VisionAffineViewsV1 {
            weight_f16: weight,
            input_f16: input,
            bias: f32_view(bias)?,
            output: f32_view(output)?,
        };
        let operation =
            self.plan
                .bind_vision_f16_affine(&self.stream, views, tokens, rows, cols)?;
        self.plan
            .submit_vision_f16_affine(&self.stream, &operation)?;
        Ok(())
    }
    fn control(
        &self,
        op: Op,
        tokens: usize,
        cols: usize,
        inputs: [Option<CudaBufferView>; 5],
        output: CudaBufferView,
        status: CudaBufferView,
    ) -> Result<()> {
        CudaQwen35VisionControlV1::bind(
            &self.stream,
            op,
            tokens,
            cols,
            CudaQwen35VisionControlViews {
                inputs,
                output,
                status,
            },
        )?
        .submit(&self.stream)?;
        Ok(())
    }
    fn upload(&self, bytes: &[u8]) -> Result<OwnedCudaBuffer> {
        let buffer = OwnedCudaBuffer::allocate(self.stream.device_ordinal(), bytes.len())?;
        buffer.copy_from_host_sync(&self.stream, bytes)?;
        Ok(buffer)
    }
    /// Real encoder forward, then one checked embedding readback. These embeddings
    /// are not fed to a target model here. Callers cannot claim multimodal text
    /// generation from this API alone. Cancellation rejects publication and drains.
    pub fn encode(
        &mut self,
        image: &Qwen35PreparedF16Image,
        cancelled: impl Fn() -> bool,
    ) -> Result<Qwen35VisionCudaOutput> {
        self.encode_traced(image, cancelled, |_, _, _| Ok(()))
    }
    // The normal monomorphization has a no-op observer. Only the ignored
    // diagnostic test supplies D2H/file IO; never use traced timings as TPS.
    fn encode_traced(
        &mut self,
        image: &Qwen35PreparedF16Image,
        cancelled: impl Fn() -> bool,
        mut observe: impl FnMut(&str, &OwnedCudaStream, CudaBufferView) -> Result<()>,
    ) -> Result<Qwen35VisionCudaOutput> {
        let geometry = Qwen35VisionGeometry::new(image.grid, 4 * 1024 * 1024)?;
        let n = geometry.patch_count();
        let layout = Qwen35VisionWorkspacePlan::new(n, 192 * 1024 * 1024)?;
        if image.pixel_values_f16_le.len() != n * 1536 * 2 {
            return Err("Vision input extent drift".into());
        }
        image.validate_payload_identity()?;
        if cancelled() {
            return Err("Vision request cancelled before allocation".into());
        }
        let started = std::time::Instant::now();
        let ordinal = self.stream.device_ordinal();
        let alloc = |bytes| OwnedCudaBuffer::allocate(ordinal, bytes);
        let hidden = alloc(n * 1152 * 4)?;
        let projected = alloc(n * 1152 * 4)?;
        let affine_scratch = alloc(n * 4304 * 4)?;
        let half_scratch = alloc(n * 4304 * 2)?;
        let output = alloc(n / 4 * 5120 * 4)?;
        let pixels = self.upload(&image.pixel_values_f16_le)?;
        let indices = self.upload(&u32_bytes(
            geometry.bilinear_indices().iter().flatten().copied(),
        ))?;
        let positions = self.upload(&f32_bytes(
            geometry.bilinear_weights().iter().flatten().copied(),
        ))?;
        let cosine = self.upload(&f32_bytes(geometry.rotary_cos().iter().copied()))?;
        let sine = self.upload(&f32_bytes(geometry.rotary_sin().iter().copied()))?;
        let status = self.upload(&[0; 4])?;
        // Declared after every workspace owner: on any error it drains before
        // local buffers drop, including enqueues that succeeded before a later bind failure.
        let mut drain = Drain {
            stream: &self.stream,
            pending: true,
        };
        let h = hidden.view(0, n * 1152 * 4)?;
        let p = projected.view(0, n * 1152 * 4)?;
        let flag = status.view(0, 4)?;
        let half_hidden = half_scratch.view(0, n * 1152 * 2)?;
        self.affine("patch_embed.proj", n, pixels.view(0, n * 1536 * 2)?, h)?;
        observe("patch", &self.stream, h)?;
        self.control(
            Op::PositionAdd,
            n,
            1152,
            [
                Some(h),
                Some(self.vector("pos_embed.weight")?),
                Some(indices.view(0, n * 16)?),
                Some(positions.view(0, n * 16)?),
                None,
            ],
            h,
            flag,
        )?;
        observe("position", &self.stream, h)?;
        for block in 0..27 {
            if cancelled() {
                return Err("Vision request cancelled during block submission".into());
            }
            let prefix = format!("blocks.{block}.");
            self.control(
                Op::LayerNormToF16,
                n,
                1152,
                [
                    Some(h),
                    Some(self.vector(&(prefix.clone() + "norm1.weight"))?),
                    Some(self.vector(&(prefix.clone() + "norm1.bias"))?),
                    None,
                    None,
                ],
                half_hidden,
                flag,
            )?;
            let qkv = affine_scratch.view(0, n * 3456 * 4)?;
            self.affine(&(prefix.clone() + "attn.qkv"), n, half_hidden, qkv)?;
            self.control(
                Op::RotaryQkInPlace,
                n,
                72,
                [
                    Some(qkv),
                    Some(cosine.view(0, n * 72 * 4)?),
                    Some(sine.view(0, n * 72 * 4)?),
                    None,
                    None,
                ],
                qkv,
                flag,
            )?;
            self.control(
                Op::OnlineAttention,
                n,
                72,
                [Some(qkv), None, None, None, None],
                p,
                flag,
            )?;
            self.control(
                Op::CastToF16,
                n,
                1152,
                [Some(p), None, None, None, None],
                half_hidden,
                flag,
            )?;
            self.affine(&(prefix.clone() + "attn.proj"), n, half_hidden, p)?;
            self.control(
                Op::ResidualAdd,
                n,
                1152,
                [Some(h), Some(p), None, None, None],
                h,
                flag,
            )?;
            observe(&format!("block_{block:02}_attention"), &self.stream, h)?;
            self.control(
                Op::LayerNormToF16,
                n,
                1152,
                [
                    Some(h),
                    Some(self.vector(&(prefix.clone() + "norm2.weight"))?),
                    Some(self.vector(&(prefix.clone() + "norm2.bias"))?),
                    None,
                    None,
                ],
                half_hidden,
                flag,
            )?;
            let up = affine_scratch.view(0, n * 4304 * 4)?;
            let activated = half_scratch.view(0, n * 4304 * 2)?;
            self.affine(&(prefix.clone() + "mlp.linear_fc1"), n, half_hidden, up)?;
            self.control(
                Op::GeluTanhToF16,
                n,
                4304,
                [Some(up), None, None, None, None],
                activated,
                flag,
            )?;
            self.affine(&(prefix + "mlp.linear_fc2"), n, activated, p)?;
            self.control(
                Op::ResidualAdd,
                n,
                1152,
                [Some(h), Some(p), None, None, None],
                h,
                flag,
            )?;
            observe(&format!("block_{block:02}"), &self.stream, h)?;
        }
        self.control(
            Op::LayerNormToF16,
            n,
            1152,
            [
                Some(h),
                Some(self.vector("merger.norm.weight")?),
                Some(self.vector("merger.norm.bias")?),
                None,
                None,
            ],
            half_hidden,
            flag,
        )?;
        let groups = n / 4;
        let merged = affine_scratch.view(0, groups * 4608 * 4)?;
        self.affine("merger.linear_fc1", groups, half_hidden, merged)?;
        self.control(
            Op::GeluErfToF16,
            groups,
            4608,
            [Some(merged), None, None, None, None],
            half_hidden,
            flag,
        )?;
        let embedding_view = output.view(0, groups * 5120 * 4)?;
        self.affine("merger.linear_fc2", groups, half_hidden, embedding_view)?;
        observe("merger", &self.stream, embedding_view)?;
        let mut errors = [0u8; 4];
        status.copy_to_host_sync(&self.stream, &mut errors)?;
        drain.pending = false;
        if errors != [0; 4] {
            return Err(format!(
                "Vision numeric control rejected: {}",
                u32::from_le_bytes(errors)
            )
            .into());
        }
        if cancelled() {
            return Err("Vision request cancelled before embedding readback".into());
        }
        let mut raw = vec![0u8; groups * 5120 * 4];
        output.copy_to_host_sync(&self.stream, &mut raw)?;
        let embeddings_f32: Vec<f32> = raw
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        if !embeddings_f32.iter().all(|x| x.is_finite()) {
            return Err("non-finite Vision embedding".into());
        }
        if cancelled() {
            return Err("Vision request cancelled before embedding publication".into());
        }
        Ok(Qwen35VisionCudaOutput {
            grid_thw: geometry.grid_thw(),
            merged_tokens: groups,
            hidden_size: 5120,
            embeddings_f32,
            prepared_payload_sha256: image.payload_sha256.clone(),
            source_config_sha256: self.config_sha256.clone(),
            source_selected_payload_sha256: self.source_selected_sha256.clone(),
            converted_weights_sha256: self.converted_sha256.clone(),
            explicit_workspace_bytes: layout.explicit_workspace_bytes,
            encoder_wall_ms: started.elapsed().as_secs_f64() * 1000.0,
            executed_blocks: 27,
            vision_forward_executed: true,
            target_image_consumed: false,
            product_ready: false,
        })
    }
    pub fn close(self) -> Result<()> {
        let Self {
            plan,
            weights,
            mut stream,
            ..
        } = self;
        stream.synchronize()?;
        drop(plan);
        drop(weights);
        stream.close()?;
        Ok(())
    }
}
fn f32_view(view: CudaBufferView) -> Result<CudaF32View> {
    Ok(CudaF32View::new(
        view,
        PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
        view.len_bytes() / 4,
    )?)
}
fn f32_bytes(values: impl Iterator<Item = f32>) -> Vec<u8> {
    values.flat_map(f32::to_le_bytes).collect()
}
fn u32_bytes(values: impl Iterator<Item = u32>) -> Vec<u8> {
    values.flat_map(u32::to_le_bytes).collect()
}

#[cfg(test)]
mod diagnostic_tests;

/// Exact converted Vision weight bytes retained in host RAM between image phases.
/// No CUDA allocation is hidden in this owner. Payloads come from the target's
/// already-authenticated pinned descriptors, not from a separate legacy model.
pub struct Qwen35VisionHostWeights {
    tensors: BTreeMap<String, Tensor>, packed: Vec<u8>, config_sha256: String,
    source_selected_sha256: String, converted_sha256: String,
}
impl Qwen35VisionHostWeights {
    pub fn from_checkpoint(reader: &mut augur_model_qwen35::CheckpointArtifact) -> Result<Self> {
        let (config, payloads) = Self::read_checkpoint(reader)?;
        Self::pack_checkpoint(config, payloads)
    }
    fn read_checkpoint(reader: &mut augur_model_qwen35::CheckpointArtifact)
        -> Result<(String, Vec<(augur_model_qwen35::Qwen35VisionTensorBinding, Vec<u8>)>)> {
        let map = reader.vision_weight_map()?;
        let mut payloads = Vec::with_capacity(map.bindings().len());
        for tensor in map.bindings() {
            let mut raw = vec![0; usize::try_from(tensor.byte_len)?];
            reader.read_tensor_range_into(&tensor.name, 0, &mut raw)?;
            payloads.push((tensor.clone(), raw));
        }
        Ok((reader.identity().config_sha256.clone(), payloads))
    }
    fn pack_checkpoint(config_sha256: String,
        payloads: Vec<(augur_model_qwen35::Qwen35VisionTensorBinding, Vec<u8>)>) -> Result<Self> {
        let mut packed = Vec::with_capacity(928 * 1024 * 1024);
        let mut tensors = BTreeMap::new(); let mut hash = Sha256::new();
        for (t, raw) in payloads {
            hash.update(&raw);
            let storage=Qwen35VisionStorage::for_binding(&t);
            let converted=convert_qwen35_vision_bf16(&raw,storage,128*1024*1024)?;
            tensors.insert(t.name.clone(),Tensor {offset:packed.len(),bytes:converted.bytes.len(),
                shape:t.shape.iter().map(|&x|x as usize).collect(),storage});
            packed.extend_from_slice(&converted.bytes);
        }
        let converted_sha256=format!("{:x}",Sha256::digest(&packed));
        Ok(Self {tensors,packed,config_sha256,
            source_selected_sha256:format!("{:x}",hash.finalize()),converted_sha256})
    }
    pub fn info(&self) -> serde_json::Value {
        serde_json::json!({"host_weight_bytes":self.packed.len(),"tensors":self.tensors.len(),
            "source_config_sha256":self.config_sha256,"source_selected_sha256":self.source_selected_sha256,
            "converted_sha256":self.converted_sha256,"gpu_weights_resident":false,
            "source":"authenticated_target_pinned_shard","product_ready":false})
    }
    pub fn upload(&self, ordinal:i32) -> Result<Qwen35VisionCudaEncoder> {
        let stream=OwnedCudaStream::create(ordinal)?;
        let weights=OwnedCudaBuffer::allocate(ordinal,self.packed.len())?;
        weights.copy_from_host_sync(&stream,&self.packed)?;
        let plan=CudaQwen35PrefillCublasPlanV1::new_for_resident_vision_affine(&stream,4096)?;
        Ok(Qwen35VisionCudaEncoder {tensors:self.tensors.clone(),weights,plan,stream,
            config_sha256:self.config_sha256.clone(),source_selected_sha256:self.source_selected_sha256.clone(),
            converted_sha256:self.converted_sha256.clone()})
    }
}

/// CPU-only conversion overlaps language-weight upload after the shared source
/// authentication. Read from pinned descriptors before spawning; never reopen
/// an unverified path, move a CUDA owner to the worker, or detach on failure.
pub(crate) struct VisionHostLoad {
    pending: Option<std::thread::JoinHandle<std::result::Result<(Qwen35VisionHostWeights, f64), String>>>,
    raw_read_s: f64,
}
impl VisionHostLoad {
    pub(crate) fn begin(reader: &mut augur_model_qwen35::CheckpointArtifact) -> Result<Self> {
        let started = std::time::Instant::now();
        let (config, payloads) = Qwen35VisionHostWeights::read_checkpoint(reader)?;
        let raw_read_s = started.elapsed().as_secs_f64();
        let pending = std::thread::Builder::new().name("augur-vision-host".into()).spawn(move || {
            let started = std::time::Instant::now();
            let host = Qwen35VisionHostWeights::pack_checkpoint(config, payloads).map_err(|e| e.to_string())?;
            Ok((host, started.elapsed().as_secs_f64()))
        })?;
        Ok(Self { pending: Some(pending), raw_read_s })
    }
    pub(crate) fn finish(mut self) -> Result<(Qwen35VisionHostWeights, serde_json::Value)> {
        let started = std::time::Instant::now();
        let (host, cpu_s) = self.pending.take().ok_or("missing Vision host worker")?
            .join().map_err(|_| "Vision host worker panicked")?.map_err(|e| e.to_string())?;
        let detail = serde_json::json!({"parallel_cpu_conversion":true,"pinned_raw_read_s":self.raw_read_s,
            "worker_conversion_hash_s":cpu_s,"join_wait_s":started.elapsed().as_secs_f64(),
            "host_weights_ready":true,"gpu_weights_resident":false,"encoder_smoke_executed":false});
        Ok((host, detail))
    }
}
impl Drop for VisionHostLoad {
    fn drop(&mut self) { if let Some(worker) = self.pending.take() { let _ = worker.join(); } }
}
