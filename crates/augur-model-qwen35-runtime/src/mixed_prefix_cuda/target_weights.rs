//! Immutable target payloads. No request cursor, KV/GDN state, scratch or bound
//! operation is stored here. Leases are private to this single-threaded lane.
use super::*;
use std::rc::Rc;

type Result<T> = std::result::Result<T, Qwen35MixedPrefixCudaError>;
pub(super) type Weight = Rc<OwnedCudaBuffer>;

pub(super) fn qpn_ffn_switch(value: Option<&std::ffi::OsStr>) -> Result<bool> {
    match value {
        None => Ok(false),
        Some(v) if v == "1" => Ok(true),
        _ => Err(invalid("AUGUR_QWEN35_QPN_FFN requires exactly 1")),
    }
}

pub(super) fn require_ffn_owner_profile(requested_qpn: bool, resident_qpn: Option<bool>) -> Result<()> {
    if requested_qpn && resident_qpn.is_none() {
        return Err(invalid("QPN FFN requires resident target weights; cold-reader QPN is not implemented"));
    }
    if resident_qpn.is_some_and(|profile| profile != requested_qpn) {
        return Err(invalid("FFN profile changed; a new immutable target owner is required"));
    }
    Ok(())
}

#[cfg(feature = "qwen35-prefill-cublas")]
pub(super) type QpnFfnWeights = [Rc<augur_backend::cuda_qwen35_qpn::OwnedCudaQwen35QpnWeightV1>; 3];

/// One scalar workspace for a whole serial target session, not per layer.
#[cfg(feature = "qwen35-prefill-cublas")]
pub(super) struct QpnScalarSession {
    stream: Rc<OwnedCudaStream>,
    plan: Rc<CudaQwen35PrefillCublasPlanV1>,
    workspace: augur_backend::cuda_qwen35_prefill_cublas::CudaQwen35QpnFfnWorkspaceV1,
}
#[cfg(feature = "qwen35-prefill-cublas")]
impl QpnScalarSession {
    pub fn new(stream: Rc<OwnedCudaStream>) -> Result<Self> {
        // Small batches do not expand weights. Reserve only the native plan's
        // nonempty sentinel, not another 170MiB FP16 matrix per scalar layer.
        let plan = Rc::new(CudaQwen35PrefillCublasPlanV1::new(&stream, 1, linear::INTERMEDIATE, 1)?);
        let workspace = augur_backend::cuda_qwen35_prefill_cublas::CudaQwen35QpnFfnWorkspaceV1::new(
            Rc::clone(&stream), linear::INTERMEDIATE, 1)?;
        Ok(Self { stream, plan, workspace })
    }
    pub fn bind(&self, stream: &OwnedCudaStream, weights: &QpnFfnWeights,
        input: Weight, output: Weight) -> Result<linear::BoundFfn> {
        if self.stream.owner_instance_key()? != stream.owner_instance_key()? {
            return Err(invalid("QPN scalar workspace stream mismatch"));
        }
        Ok(linear::BoundFfn::Qpn(
            augur_backend::cuda_qwen35_prefill_cublas::CudaQwen35QpnFfnProgramV1::bind(
                Rc::clone(&self.plan), Rc::clone(&self.stream), weights.clone(), &self.workspace,
                input, 0, output, 0, 1)?))
    }
}

pub(super) fn pinned_startup_enabled() -> Result<bool> {
    match std::env::var_os("AUGUR_QWEN35_PINNED_STARTUP") {
        None => Ok(false),
        Some(value) if value == "1" => Ok(true),
        Some(_) => Err(invalid("pinned startup requires value1")),
    }
}

pub(super) fn overlapped_startup_enabled(pinned: bool) -> Result<bool> {
    match std::env::var_os("AUGUR_QWEN35_OVERLAPPED_STARTUP") {
        None => Ok(false),
        Some(value) if value == "1" && pinned => Ok(true),
        Some(_) => Err(invalid("overlapped startup requires value1 and pinned startup")),
    }
}

pub(super) fn startup_component_enabled(key: &str, pinned: bool) -> Result<bool> {
    match std::env::var_os(key) {
        None => Ok(false),
        Some(value) if value == "1" && pinned => Ok(true),
        Some(_) => Err(invalid(format!("{key} requires value1 and pinned startup"))),
    }
}

fn upload(stream: &OwnedCudaStream, bytes: &[u8]) -> Result<Weight> {
    let buffer = OwnedCudaBuffer::allocate(stream.device_ordinal(), bytes.len())?;
    buffer.copy_from_host_sync(stream, bytes)?;
    Ok(Rc::new(buffer))
}

fn dense(stream: &OwnedCudaStream, values: &[f32]) -> Result<Weight> {
    upload(stream, &f32_bytes(values))
}

fn input_upload(s:&OwnedCudaStream,bytes:&[u8],rows:usize,enabled:bool,name:&str,
    receipts:&mut Vec<serde_json::Value>)->Result<Weight>{
    let started=Instant::now();
    let converted=enabled.then(||row_input::encode(bytes,rows,5120)).transpose()?;
    let relayout_ms=if enabled {started.elapsed().as_secs_f64()*1000.0}else{0.0};
    let physical=converted.as_deref().unwrap_or(bytes);
    if enabled {receipts.push(serde_json::json!({"name":name,"rows":rows,"cols":5120,
        "layout":row_input::layout(true).0,"bytes":physical.len(),"physical_sha256":sha256_hex(physical),
        "relayout_ms_excluding_hash_and_upload":relayout_ms}));}
    upload(s,physical)
}

pub(super) struct FfnWeights {
    pub gate: Weight,
    pub up: Weight,
    pub down: Weight,
    pub weight_type: u32,
    pub gate_scale: f32,
    pub up_scale: f32,
    pub down_scale: f32,
    #[cfg(feature = "qwen35-prefill-cublas")]
    pub qpn: Option<QpnFfnWeights>,
}

impl FfnWeights {
    fn upload(s: &OwnedCudaStream, m: &FfnMaterial) -> Result<Self> {
        let selected = qpn_ffn_switch(std::env::var_os("AUGUR_QWEN35_QPN_FFN").as_deref())?;
        #[cfg(not(feature = "qwen35-prefill-cublas"))]
        if selected { return Err(invalid("QPN FFN requires the prefill CUDA feature")); }
        #[cfg(feature = "qwen35-prefill-cublas")]
        if selected && m.weight_type == GGML_TYPE_NVFP4 {
            let upload = |t: &LoadedTensor, rows, cols, scale| {
                augur_backend::cuda_qwen35_qpn::OwnedCudaQwen35QpnWeightV1::upload_ggml(
                    s, &t.payload, rows, cols, scale).map(Rc::new)
            };
            let qpn = [upload(&m.gate, linear::INTERMEDIATE, linear::HIDDEN, m.gate_scale)?,
                upload(&m.up, linear::INTERMEDIATE, linear::HIDDEN, m.up_scale)?,
                upload(&m.down, linear::HIDDEN, linear::INTERMEDIATE, m.down_scale)?];
            return Ok(Self {
                gate: qpn[0].shared_storage(), up: qpn[1].shared_storage(), down: qpn[2].shared_storage(),
                weight_type: m.weight_type, gate_scale: m.gate_scale, up_scale: m.up_scale,
                down_scale: m.down_scale, qpn: Some(qpn),
            });
        }
        Ok(Self {
            gate: upload(s, &m.gate.payload)?,
            up: upload(s, &m.up.payload)?,
            down: upload(s, &m.down.payload)?,
            weight_type: m.weight_type,
            gate_scale: m.gate_scale,
            up_scale: m.up_scale,
            down_scale: m.down_scale,
            #[cfg(feature = "qwen35-prefill-cublas")]
            qpn: None,
        })
    }
    pub fn qpn_enabled(&self) -> bool {
        #[cfg(feature = "qwen35-prefill-cublas")]
        { return self.qpn.is_some(); }
        #[cfg(not(feature = "qwen35-prefill-cublas"))]
        { false }
    }
}

pub(super) struct ProjectionWeights {
    pub input_layout: PhysicalLayoutId,
    pub input_receipts: Vec<serde_json::Value>,
    pub attn_norm_weight: Weight,
    pub qkv_weight: Weight,
    pub z_weight: Weight,
    pub alpha_weight: Weight,
    pub beta_weight: Weight,
    pub dt_bias: Weight,
    pub ssm_a: Weight,
}

pub(super) struct LinearWeights {
    pub spec: Qwen35LayerSpec,
    pub conv: Weight,
    pub ssm_norm: Weight,
    pub output_weight: Weight,
    pub post_norm_weight: Weight,
    pub ffn: FfnWeights,
    pub projection: Option<ProjectionWeights>,
}

impl LinearWeights {
    pub fn upload(s: &OwnedCudaStream, m: &LinearMaterial, projection: bool) -> Result<Self> {
        let enabled=row_input::selected()?;
        let mut input_receipts=Vec::new();
        Ok(Self {
            spec: m.spec.clone(),
            conv: dense(s, &m.conv_weights)?,
            ssm_norm: dense(s, &m.ssm_norm)?,
            output_weight: upload(s, &m.ssm_out.payload)?,
            post_norm_weight: dense(s, &m.post_norm)?,
            ffn: FfnWeights::upload(s, &m.ffn)?,
            projection: if projection {
                Some(ProjectionWeights {
                    input_layout: row_input::layout(enabled),
                    attn_norm_weight: dense(s, &m.attn_norm)?,
                    qkv_weight: input_upload(s, &m.qkv.payload,10240,enabled,"qkv",&mut input_receipts)?,
                    z_weight: input_upload(s, &m.z.payload,6144,enabled,"z",&mut input_receipts)?,
                    input_receipts,
                    alpha_weight: upload(s, &m.alpha.payload)?,
                    beta_weight: upload(s, &m.beta.payload)?,
                    dt_bias: dense(s, &m.dt_bias)?,
                    ssm_a: dense(s, &m.ssm_a)?,
                })
            } else {
                None
            },
        })
    }
}

pub(super) struct FullWeights {
    pub input_layout: PhysicalLayoutId,
    pub input_receipts: Vec<serde_json::Value>,
    pub spec: Qwen35LayerSpec,
    pub attn_norm_weight: Weight,
    pub q_weight: Weight,
    pub q_norm_weight: Weight,
    pub k_weight: Weight,
    pub k_norm_weight: Weight,
    pub v_weight: Weight,
    pub output_weight: Weight,
    pub post_norm_weight: Weight,
    pub ffn: FfnWeights,
}

impl FullWeights {
    pub fn upload(s: &OwnedCudaStream, m: &FullMaterial) -> Result<Self> {
        let enabled=row_input::selected()?;
        let mut input_receipts=Vec::new();
        let q = full::reorder_interleaved_query_gate_q8_0_rows(
            &m.q.descriptor,
            &m.q.payload,
            m.spec.query_heads,
            m.spec.head_dim,
        )?;
        Ok(Self {
            input_layout: row_input::layout(enabled),
            spec: m.spec.clone(),
            attn_norm_weight: dense(s, &m.attn_norm)?,
            q_weight: input_upload(s,&q,12288,enabled,"q_reordered",&mut input_receipts)?,
            q_norm_weight: dense(s, &repeat_norm_values(&m.q_norm, m.spec.query_heads)?)?,
            k_weight: input_upload(s,&m.k.payload,1024,enabled,"k",&mut input_receipts)?,
            k_norm_weight: dense(s, &repeat_norm_values(&m.k_norm, m.spec.kv_heads)?)?,
            v_weight: input_upload(s,&m.v.payload,1024,enabled,"v",&mut input_receipts)?,
            input_receipts,
            output_weight: upload(s, &m.attn_output.payload)?,
            post_norm_weight: dense(s, &m.post_norm)?,
            ffn: FfnWeights::upload(s, &m.ffn)?,
        })
    }
}

pub(super) struct HeadWeights {
    pub norm_weight: Weight,
    pub output_weight: Weight,
    pub hidden: usize,
    pub vocabulary: usize,
    pub payload_sha256: String,
    pub output_layout: PhysicalLayoutId,
    pub physical_sha256: String,
    pub relayout_ms: f64,
}

fn planar_head_switch(value: Option<&std::ffi::OsStr>, supported: bool) -> Result<bool> {
    match value {
        None => Ok(false),
        Some(v) if v == "0" => Ok(false),
        Some(v) if v == "1" && supported => Ok(true),
        _ => Err(invalid("AUGUR_QWEN35_PLANAR_Q8_HEAD requires 0/1 and prefill backend")),
    }
}

// Pure permutation: keep original scale bits, signed code bytes and extent.
// The temporary host payload is dropped after one synchronous upload. There
// is no second resident allocation and no decode-time conversion.
fn encode_planar_head(bytes: &[u8], rows: usize, cols: usize) -> Result<Vec<u8>> {
    let codes = rows.checked_mul(cols).ok_or_else(|| invalid("planar head extent overflow"))?;
    let expected = codes.checked_div(32).and_then(|b| b.checked_mul(34));
    if rows == 0 || cols == 0 || cols % 32 != 0 || expected != Some(bytes.len()) {
        return Err(invalid("planar head codec extent"));
    }
    let mut output = vec![0u8; bytes.len()];
    let (values, scales) = output.split_at_mut(codes);
    for (index, block) in bytes.chunks_exact(34).enumerate() {
        values[index*32..(index+1)*32].copy_from_slice(&block[2..]);
        scales[index*2..index*2+2].copy_from_slice(&block[..2]);
    }
    Ok(output)
}

impl HeadWeights {
    pub fn upload(
        s: &OwnedCudaStream,
        m: &OutputHeadMaterial,
        hidden: usize,
        vocabulary: usize,
    ) -> Result<Self> {
        if m.norm.len() != hidden
            || m.weight.descriptor.dimensions != [hidden as u64, vocabulary as u64]
            || m.weight.descriptor.ggml_type != GGML_TYPE_Q8_0
        {
            return Err(invalid("output-head material geometry"));
        }
        let planar = planar_head_switch(std::env::var_os("AUGUR_QWEN35_PLANAR_Q8_HEAD").as_deref(),
            cfg!(feature = "qwen35-prefill-cublas"))?;
        if planar && (hidden != 5120 || vocabulary != 248320) {
            return Err(invalid("planar head requires Qwen35 248320x5120"));
        }
        let started = Instant::now();
        let converted = planar.then(|| encode_planar_head(&m.weight.payload, vocabulary, hidden)).transpose()?;
        let relayout_ms = if planar { started.elapsed().as_secs_f64()*1000.0 } else { 0.0 };
        let resident_bytes = converted.as_deref().unwrap_or(&m.weight.payload);
        let logical_sha = sha256_hex(&m.weight.payload);
        let physical_sha = if planar { sha256_hex(resident_bytes) } else { logical_sha.clone() };
        Ok(Self {
            norm_weight: dense(s, &m.norm)?,
            output_weight: upload(s, resident_bytes)?,
            hidden,
            vocabulary,
            payload_sha256: logical_sha,
            output_layout: if planar { PhysicalLayoutId::DEVICE_QWEN35_Q8_CODE_SCALE_PLANES_V1 }
                else { PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_0_CUDA_V1 },
            physical_sha256: physical_sha,
            relayout_ms,
        })
    }
}

pub(super) enum LayerWeights {
    Linear(LinearWeights),
    Full(FullWeights),
}

#[cfg(feature = "qwen35-draft-target-verify")]
fn load_target_layers(reader: &mut augur_model_qwen35::GgufTensorReader<'_>, manifest: &GgufManifest,
    model: &Qwen35Config, stream: &OwnedCudaStream, pipeline: bool) -> Result<(Vec<LayerWeights>, serde_json::Value)> {
    let upload_layer = |material: LayerMaterial| -> Result<LayerWeights> {
        Ok(match material {
            LayerMaterial::Linear(m) => LayerWeights::Linear(LinearWeights::upload(stream, &m, true)?),
            LayerMaterial::Full(m) => LayerWeights::Full(FullWeights::upload(stream, &m)?),
        })
    };
    let read_layer = |reader: &mut augur_model_qwen35::GgufTensorReader<'_>, layer: usize| -> Result<_> {
        let start = Instant::now();
        let mut materials = load_layer_material_range(reader, manifest, model, layer..layer + 1)?;
        let material = materials.pop().ok_or_else(|| invalid("missing resident layer"))?;
        Ok((material, start.elapsed().as_secs_f64()*1000.0))
    };
    let mut details = Vec::with_capacity(model.num_hidden_layers);
    let layers = if pipeline {
        std::thread::scope(|scope| -> Result<_> {
            // Rendezvous: at most the caller's current material and one next
            // material exist. Only CPU material crosses this channel.
            let (send, receive) = std::sync::mpsc::sync_channel(0);
            let worker = scope.spawn(move || -> Result<()> {
                for layer in 0..model.num_hidden_layers {
                    let (material, read_ms) = read_layer(reader, layer)?;
                    if send.send((layer, material, read_ms)).is_err() { return Ok(()); }
                }
                Ok(())
            });
            let loaded = (|| -> Result<_> {
                let mut layers = Vec::with_capacity(model.num_hidden_layers);
                for layer in 0..model.num_hidden_layers {
                    let wait = Instant::now();
                    let (actual, material, read_ms) = receive.recv().map_err(|_| invalid("layer preparation channel closed"))?;
                    let wait_ms = wait.elapsed().as_secs_f64()*1000.0;
                    if actual != layer { return Err(invalid("layer preparation order drift")); }
                    let upload = Instant::now();
                    layers.push(upload_layer(material)?);
                    details.push(serde_json::json!({"layer":layer,"cpu_read_material_and_hash_ms":read_ms,
                        "relayout_and_upload_host_ms":upload.elapsed().as_secs_f64()*1000.0,"consumer_wait_ms":wait_ms}));
                }
                Ok(layers)
            })();
            // Drop the receiver BEFORE joining on a CUDA/consumer failure, so a
            // sender waiting at rendezvous wakes instead of deadlocking.
            drop(receive);
            worker.join().map_err(|_| invalid("layer preparation worker panicked"))??;
            loaded
        })?
    } else {
        let mut layers = Vec::with_capacity(model.num_hidden_layers);
        for layer in 0..model.num_hidden_layers {
            let (material, read_ms) = read_layer(reader, layer)?;
            let upload = Instant::now();
            layers.push(upload_layer(material)?);
            details.push(serde_json::json!({"layer":layer,"cpu_read_material_and_hash_ms":read_ms,
                "relayout_and_upload_host_ms":upload.elapsed().as_secs_f64()*1000.0,"consumer_wait_ms":0.0}));
        }
        layers
    };
    Ok((layers, serde_json::json!({"schema":"qwen35-startup-layer-host-bill-v1","pipeline_enabled":pipeline,
        "gpu_device_time_measured":false,"nested_parts_overlap":pipeline,"prepared_layer_count":details.len(),
        "maximum_cpu_layer_materials":if pipeline {2} else {1},"parts_not_additive_with_target_wall":true,"layers":details})))
}

#[cfg(feature = "qwen35-draft-target-verify")]
fn resident_tokenizer_switch(value: Option<&std::ffi::OsStr>) -> Result<bool> {
    match value {
        None => Ok(false),
        Some(v) if v == "0" => Ok(false),
        Some(v) if v == "1" => Ok(true),
        Some(_) => Err(invalid("resident tokenizer switch requires0 or1")),
    }
}

// Immutable CPU parse owner, not a prompt/token/KV cache. Every lease still
// validates freshly stable-read file bytes before encoding or decoding.
#[cfg(feature = "qwen35-draft-target-verify")]
struct ResidentTokenizer {
    parsed: Arc<Tokenizer>,
    sha256: String,
    source_bytes: usize,
    leases: std::cell::Cell<u64>,
}

#[cfg(feature = "qwen35-draft-target-verify")]
impl ResidentTokenizer {
    fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Ok(Self {
            parsed: Arc::new(Tokenizer::from_bytes(bytes).map_err(|e| invalid(e.to_string()))?),
            sha256: sha256_hex(bytes), source_bytes: bytes.len(), leases: std::cell::Cell::new(0),
        })
    }
    fn lease(&self, bytes: &[u8]) -> Result<Arc<Tokenizer>> {
        self.lease_with_digest(bytes).map(|(parsed, _)| parsed)
    }
    // Return the digest computed from THIS lease's freshly read bytes. This is
    // not a caller-supplied digest or a bypass of the per-request content check.
    fn lease_with_digest(&self, bytes: &[u8]) -> Result<(Arc<Tokenizer>, String)> {
        if bytes.len() != self.source_bytes {
            return Err(invalid("resident tokenizer bytes changed; new owner required"));
        }
        let digest = sha256_hex(bytes);
        if digest != self.sha256 {
            return Err(invalid("resident tokenizer bytes changed; new owner required"));
        }
        self.leases.set(self.leases.get().checked_add(1).ok_or_else(|| invalid("tokenizer lease counter overflow"))?);
        Ok((Arc::clone(&self.parsed), digest))
    }
    fn report(&self) -> serde_json::Value {
        serde_json::json!({"enabled":true,"tokenizer_sha256":self.sha256,"source_bytes":self.source_bytes,
            "parse_count":1,"successful_request_leases":self.leases.get(),
            "content_checked_each_lease":true,"prompt_cache":false,"token_policy_changed":false,
            "parsed_heap_bytes":serde_json::Value::Null,"scope":"immutable_target_owner_lifetime"})
    }
}

/// Pinned load-time GGUF identity. Host payloads are released one layer at a
/// time, before the next layer is read; there is no second host model copy.
#[cfg(feature = "qwen35-draft-target-verify")]
pub(super) struct TargetWeights {
    pub manifest: GgufManifest,
    pub layers: Vec<LayerWeights>,
    pub head: HeadWeights,
    pub embedding: Weight,
    // CPU prefill consumes individual BF16 embedding rows. Keeping only this
    // bounded payload avoids both per-request full-file rehash and GPU D2H.
    host_embedding: Vec<u8>,
    pub config_sha256: String,
    pub gguf_path: PathBuf,
    pub device: i32,
    pub resident_bytes: u64,
    pub resident_allocations: u64,
    tokenizer: Option<ResidentTokenizer>,
    pinned_startup: bool,
    overlapped_startup: bool,
    pub startup_timing: serde_json::Value,
}

#[cfg(feature = "qwen35-draft-target-verify")]
impl TargetWeights {
    pub fn qpn_ffn_enabled(&self) -> bool {
        self.layers.iter().any(|layer| match layer {
            LayerWeights::Linear(w) => w.ffn.qpn_enabled(),
            LayerWeights::Full(w) => w.ffn.qpn_enabled(),
        })
    }
    pub fn open(config: &Qwen35AutoregressiveCudaConfig) -> Result<Self> {
        let startup = Instant::now();
        let tokenizer = if resident_tokenizer_switch(std::env::var_os("AUGUR_QWEN35_RESIDENT_TOKENIZER").as_deref())? {
            Some(ResidentTokenizer::from_bytes(&read_stable_regular_file(&config.tokenizer_path, "resident tokenizer")?)?)
        } else { None };
        let bytes = read_stable_regular_file(&config.config_path, "resident target config")?;
        let model = Qwen35Config::parse(&bytes)?;
        model.validate_exact_profile()?;
        if config.device_ordinal != 0
            || cuda_device_properties(0)?.compute_capability
                != augur_backend::CudaComputeCapability::SM70
        {
            return Err(invalid("resident target requires SM70 device zero"));
        }
        let validation_end_ms = startup.elapsed().as_secs_f64() * 1000.0;
        let pinned_startup = pinned_startup_enabled()?;
        let overlapped_startup = overlapped_startup_enabled(pinned_startup)?;
        let pipelined_layers = startup_component_enabled("AUGUR_QWEN35_PIPELINED_LAYER_STARTUP", pinned_startup)?;
        let before = cuda_device_allocation_snapshot();
        let stream = OwnedCudaStream::create(config.device_ordinal)?;
        // Identical payload transformations/uploads in all paths. The overlapped
        // path changes only authentication scheduling; it cannot return usable
        // TargetWeights until the complete file hash has joined successfully.
        let prepare = |manifest: &GgufManifest, reader: &mut augur_model_qwen35::GgufTensorReader<'_>| -> Result<_> {
            Qwen35TensorSchema::validate_manifest_allowing_mtp(&model, manifest)?;
            let reader_end_ms = startup.elapsed().as_secs_f64() * 1000.0;
            let (layers, layer_bill) = load_target_layers(reader, manifest, &model, &stream, pipelined_layers)?;
            let layers_end_ms = startup.elapsed().as_secs_f64() * 1000.0;
            let material = load_output_head_material(reader, manifest, &model)?;
            let head = HeadWeights::upload(&stream, &material, model.hidden_size, model.vocab_size)?;
            drop(material);
            let head_end_ms = startup.elapsed().as_secs_f64() * 1000.0;
            let descriptor = manifest.tensor("token_embd.weight").ok_or_else(|| invalid("resident embedding missing"))?;
            let payload = reader.read_tensor_bytes("token_embd.weight")?;
            if descriptor.ggml_type != GGML_TYPE_BF16 || payload.len() != model.vocab_size * model.hidden_size * 2 {
                return Err(invalid("resident BF16 embedding geometry"));
            }
            let embedding = upload(&stream, &payload)?;
            Ok((layers, head, embedding, payload, layer_bill, reader_end_ms, layers_end_ms, head_end_ms,
                startup.elapsed().as_secs_f64() * 1000.0))
        };
        let (prepared, manifest, manifest_end_ms, overlap_timing) = if overlapped_startup {
            let mut manifest_end_ms = 0.0;
            let (prepared, artifact, timing) = augur_model_qwen35::GgufPinnedArtifact::load_overlapped(
                &config.gguf_path, GgufAuditOptions::default(), |m, r| {
                    manifest_end_ms = startup.elapsed().as_secs_f64() * 1000.0;
                    prepare(m, r)
                })?;
            (prepared, artifact.manifest().clone(), manifest_end_ms, Some(timing))
        } else {
            let mut artifact = if pinned_startup {
                Some(augur_model_qwen35::GgufPinnedArtifact::open(&config.gguf_path, GgufAuditOptions::default())?)
            } else { None };
            let manifest = if let Some(a) = &artifact { a.manifest().clone() }
                else { GgufManifest::inspect(&config.gguf_path, GgufAuditOptions::default())? };
            let manifest_end_ms = startup.elapsed().as_secs_f64() * 1000.0;
            let mut reader = if let Some(a) = artifact.as_mut() { a.tensor_reader()? }
                else { manifest.open_tensor_reader(&config.gguf_path)? };
            let prepared = prepare(&manifest, &mut reader)?;
            drop(reader);
            (prepared, manifest, manifest_end_ms, None)
        };
        if manifest.file_sha256.is_none() { return Err(invalid("resident target hash missing")); }
        let (layers, head, embedding, payload, layer_bill, reader_end_ms, layers_end_ms, head_end_ms, embedding_end_ms) = prepared;
        let qpn_selected = qpn_ffn_switch(std::env::var_os("AUGUR_QWEN35_QPN_FFN").as_deref())?;
        for layer in &layers {
            let ffn = match layer { LayerWeights::Linear(w) => &w.ffn, LayerWeights::Full(w) => &w.ffn };
            if ffn.qpn_enabled() != (qpn_selected && ffn.weight_type == GGML_TYPE_NVFP4) {
                return Err(invalid("FFN numerical profile changed while loading immutable target"));
            }
        }
        let authentication_end_ms = startup.elapsed().as_secs_f64() * 1000.0;
        stream.synchronize()?;
        let sync_end_ms = startup.elapsed().as_secs_f64() * 1000.0;
        let after = cuda_device_allocation_snapshot();
        Ok(Self {
            manifest,
            layers,
            head,
            embedding,
            host_embedding: payload,
            config_sha256: sha256_hex(&bytes),
            gguf_path: config.gguf_path.clone(),
            device: config.device_ordinal,
            resident_bytes: after.live_bytes - before.live_bytes,
            resident_allocations: after.live_allocations - before.live_allocations,
            tokenizer,
            pinned_startup,
            overlapped_startup,
            startup_timing: serde_json::json!({"schema":"qwen35-target-startup-host-timing-v2",
                "clock":"host_monotonic_wall","gpu_device_time_measured":false,"same_clock_disjoint_partition":true,
                "scope":"target_open_entry_through_final_upload_sync","enclosing_ms":sync_end_ms,
                "full_file_hash_passes":if pinned_startup {1} else {2},
                "authentication_overlapped":overlapped_startup,"overlap_nested_walls_ms":overlap_timing,
                "layer_preparation_bill":layer_bill,
                "inference_and_ready_after_full_authentication":true,
                "parts_ms":{"config_tokenizer_and_device_validation":validation_end_ms,
                    "manifest_and_front_authentication":manifest_end_ms-validation_end_ms,
                    "stream_and_reader_binding":reader_end_ms-manifest_end_ms,
                    "layer_read_relayout_hash_and_upload":layers_end_ms-reader_end_ms,
                    "output_head_read_relayout_hash_and_upload":head_end_ms-layers_end_ms,
                    "embedding_read_and_upload":embedding_end_ms-head_end_ms,
                    "postload_authentication_barrier":authentication_end_ms-embedding_end_ms,
                    "final_sync":sync_end_ms-authentication_end_ms}}),
        })
    }

    pub fn validate_request(
        &self,
        config: &Qwen35AutoregressiveCudaConfig,
        bytes: &[u8],
    ) -> Result<()> {
        if self.gguf_path != config.gguf_path
            || self.device != config.device_ordinal
            || self.config_sha256 != sha256_hex(bytes)
        {
            return Err(invalid("resident target artifact/config/device mismatch"));
        }
        Ok(())
    }

    pub fn host_embedding_bytes(&self) -> usize {
        self.host_embedding.len()
    }

    pub fn request_tokenizer(&self, current_bytes: &[u8]) -> Result<Arc<Tokenizer>> {
        match &self.tokenizer {
            Some(owner) => owner.lease(current_bytes),
            None => Ok(Arc::new(Tokenizer::from_bytes(current_bytes).map_err(|e| invalid(e.to_string()))?)),
        }
    }

    pub fn request_tokenizer_with_digest(&self, current_bytes: &[u8]) -> Result<(Arc<Tokenizer>, String)> {
        self.tokenizer.as_ref()
            .ok_or_else(|| invalid("tokenizer digest reuse requires resident parsed owner"))?
            .lease_with_digest(current_bytes)
    }

    pub fn input_layout_report(&self) -> serde_json::Value {
        let layers:Vec<_>=self.layers.iter().enumerate().map(|(layer,w)|match w {
            LayerWeights::Linear(w)=>w.projection.as_ref().map(|p|serde_json::json!({
                "layer":layer,"layout":p.input_layout.0,"weights":p.input_receipts})).unwrap_or(serde_json::Value::Null),
            LayerWeights::Full(w)=>serde_json::json!({"layer":layer,"layout":w.input_layout.0,"weights":w.input_receipts}),
        }).collect();
        serde_json::json!({"schema":"qwen35-target-input-weight-layout-v1","layers":layers,
            "physical_hash_scope":"uploaded_QKV_Z_and_reordered_Q_K_V_bytes_only",
            "startup_conversion_only":true,"second_gpu_weight_copy":false,"product_ready":false})
    }

    pub fn resident_report(&self) -> serde_json::Value {
        serde_json::json!({
            "schema":"qwen35-immutable-resident-target-v1",
            "weight_bytes":self.resident_bytes,
            "weight_allocations":self.resident_allocations,
            "input_weight_layout":self.input_layout_report(),
            "ffn_numerical_profile":if self.qpn_ffn_enabled() {"qpn_folded_f16_original_activation_v1"}
                else {"legacy_ggml_mixed_ffn_v1"},
            "ffn_qpn_weight_copy_count":if self.qpn_ffn_enabled() {1} else {0},
            "ffn_input_scale_applied":false,
            "host_embedding_bytes":self.host_embedding_bytes(),
            "target_gguf_sha256":self.manifest.file_sha256,
            "config_sha256":self.config_sha256,
            "source_identity_policy":"load_time_authenticated_bytes_pinned_until_owner_drop",
            "weight_upload_policy":"reuse_load_time_owner_no_request_upload",
            "startup_reader_policy":if self.overlapped_startup {"windows_locked_descriptor_overlapped_full_hash_v1"}
                else if self.pinned_startup {"windows_locked_descriptor_single_hash_v1"} else {"legacy_reopened_descriptor_double_hash_v1"},
            "startup_timing":self.startup_timing,
            "host_policy":if self.tokenizer.is_some() {"bf16_embedding_and_immutable_tokenizer_no_layer_payload_retention"}
                else {"bf16_embedding_only_no_layer_payload_retention"},
            "tokenizer_cache":self.tokenizer.as_ref().map(ResidentTokenizer::report),
            "fresh_request_state":true,"product_ready":false
        })
    }

    pub fn embedding_row(&self, token: usize) -> Result<Vec<f32>> {
        decode_embedding_row(
            &self.host_embedding,
            self.head.hidden,
            self.head.vocabulary,
            token,
        )
    }
}

fn decode_embedding_row(
    bytes: &[u8],
    hidden: usize,
    vocabulary: usize,
    token: usize,
) -> Result<Vec<f32>> {
    let row_bytes = hidden
        .checked_mul(2)
        .ok_or_else(|| invalid("embedding row overflow"))?;
    let expected = row_bytes
        .checked_mul(vocabulary)
        .ok_or_else(|| invalid("embedding extent overflow"))?;
    if hidden == 0 || token >= vocabulary || bytes.len() != expected {
        return Err(invalid("resident embedding row bounds"));
    }
    let start = token * row_bytes;
    let values: Vec<_> = bytes[start..start + row_bytes]
        .chunks_exact(2)
        .map(|b| half::bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
        .collect();
    if values.iter().any(|v| !v.is_finite()) {
        return Err(invalid("resident embedding finite"));
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn qpn_profile_requires_matching_resident_owner_without_cold_fallback() {
        for (requested, resident, accepted) in [
            (false, None, true), (false, Some(false), true), (true, Some(true), true),
            (true, None, false), (false, Some(true), false), (true, Some(false), false),
        ] {
            assert_eq!(require_ffn_owner_profile(requested, resident).is_ok(), accepted);
        }
    }
    #[test]
    fn qpn_profile_switch_is_explicit() {
        use std::ffi::OsStr;
        assert!(!qpn_ffn_switch(None).unwrap());
        assert!(qpn_ffn_switch(Some(OsStr::new("1"))).unwrap());
        for value in ["", "0", "true", "01", "1 ", " 1"] {
            assert!(qpn_ffn_switch(Some(OsStr::new(value))).is_err());
        }
    }

    #[cfg(feature = "qwen35-draft-target-verify")]
    #[test]
    #[ignore = "exclusive SM70; AUGUR_QPN_TARGET_INPUT pins mixed-model scalar smoke inputs"]
    fn qpn_scalar_full_target_smoke() {
        let path = PathBuf::from(std::env::var_os("AUGUR_QPN_TARGET_INPUT").unwrap());
        let input_bytes = std::fs::read(&path).unwrap();
        let m: serde_json::Value = serde_json::from_slice(&input_bytes).unwrap();
        let config = Qwen35AutoregressiveCudaConfig {
            gguf_path: PathBuf::from(m["model"].as_str().unwrap()),
            config_path: PathBuf::from(m["config"].as_str().unwrap()),
            tokenizer_path: PathBuf::from(m["tokenizer"].as_str().unwrap()),
            capacity: 256, ..Default::default()
        };
        let config_bytes = std::fs::read(&config.config_path).unwrap();
        let tokenizer_bytes = std::fs::read(&config.tokenizer_path).unwrap();
        assert_eq!(sha256_hex(&config_bytes), m["config_sha256"].as_str().unwrap());
        assert_eq!(sha256_hex(&tokenizer_bytes), m["tokenizer_sha256"].as_str().unwrap());
        let model = Qwen35Config::parse(&config_bytes).unwrap();
        let tokenizer = Tokenizer::from_bytes(&tokenizer_bytes).unwrap();
        let stop_tokens = [tokenizer.token_to_id("<|im_end|>").expect("current tokenizer im_end"),
            tokenizer.token_to_id("<|endoftext|>").expect("current tokenizer endoftext")];
        assert_ne!(stop_tokens[0], stop_tokens[1]);
        assert!(stop_tokens.iter().all(|id| (*id as usize) < model.vocab_size));
        let selected = qpn_ffn_switch(std::env::var_os("AUGUR_QWEN35_QPN_FFN").as_deref()).unwrap();
        let before = cuda_device_allocation_snapshot();
        let mut reports = Vec::new();
        let resident;
        {
            let stream = Rc::new(OwnedCudaStream::create(0).unwrap());
            let qpn = selected.then(|| QpnScalarSession::new(Rc::clone(&stream)).unwrap());
            eprintln!("QPN_TARGET_PROGRESS=opening_target profile_qpn={selected}");
            let weights = TargetWeights::open(&config).unwrap();
            assert_eq!(weights.manifest.file_sha256.as_deref(), m["model_sha256"].as_str());
            assert_eq!(weights.qpn_ffn_enabled(), selected);
            resident = weights.resident_report();
            eprintln!("QPN_TARGET_PROGRESS=target_loaded layers={}", weights.layers.len());
            let mut plans = Vec::<DevicePlan>::with_capacity(64);
            let tables = Arc::new(FullRopeTables::new(&stream, model.rope_theta as f32, config.capacity, full::ROTARY_DIM).unwrap());
            let zero = vec![0.0; linear::HIDDEN];
            for layer in &weights.layers {
                let source = plans.last().map(DevicePlan::final_hidden_f32_view).transpose().unwrap();
                let plan = match layer {
                    LayerWeights::Linear(w) => DevicePlan::Linear(bind_linear_plan(&stream, w, &zero,
                        None, true, source, qpn.as_ref()).unwrap()),
                    LayerWeights::Full(w) => DevicePlan::Full(bind_full_plan(&stream, w, &model,
                        config.capacity, config.capacity, source, Some(Arc::clone(&tables)), qpn.as_ref()).unwrap()),
                };
                plans.push(plan);
            }
            assert_eq!(plans.len(), 64);
            let head = bind_device_output_head_plan(&stream, &weights.head,
                plans.last().unwrap().final_hidden_view().unwrap(), model.rms_norm_eps as f32).unwrap();
            for case in m["cases"].as_array().unwrap() {
                let rendered = case["rendered_prompt"].as_str().unwrap();
                let prompt = tokenizer.encode(rendered, false).unwrap().get_ids().to_vec();
                let limit = case["max_new_tokens"].as_u64().unwrap() as usize;
                assert!(!prompt.is_empty() && limit > 0 && prompt.len() + limit <= config.capacity);
                let mut replay = Vec::new();
                let mut generated = Vec::new();
                for pass in 0..2 {
                    for plan in &mut plans { plan.reset(&stream).unwrap(); }
                    let mut tokens = Vec::<u32>::new();
                    let mut top2s = Vec::new();
                    let mut logit_bytes = vec![0; model.vocab_size * 4];
                    let allocations = cuda_device_allocation_snapshot().successful_allocations;
                    let started = Instant::now();
                    for position in 0..prompt.len() + limit - 1 {
                        let token = if position < prompt.len() { prompt[position] }
                            else { tokens[position - prompt.len()] };
                        let hidden = weights.embedding_row(token as usize).unwrap();
                        plans[0].upload(&stream, None, &hidden, true).unwrap();
                        for plan in &plans { plan.submit_resident(&stream, position).unwrap(); }
                        if position + 1 >= prompt.len() {
                            head.submit(&stream).unwrap();
                            head.logits.copy_to_host_sync(&stream, &mut logit_bytes).unwrap();
                            let logits: Vec<_> = logit_bytes.chunks_exact(4)
                                .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
                            let top2 = cuda_qwen35_finite_top2_cpu_v1(&logits).unwrap();
                            tokens.push(top2.token_id);
                            top2s.push(top2);
                            if stop_tokens.contains(&top2.token_id) { break; }
                        }
                    }
                    stream.synchronize().unwrap();
                    assert_eq!(cuda_device_allocation_snapshot().successful_allocations, allocations,
                        "no submit-time owned allocation");
                    if pass == 0 {
                        replay = top2s;
                        generated = tokens;
                        reports.push(serde_json::json!({"name":case["name"],"prompt_tokens":prompt,
                            "generated_token_ids":generated,"text":tokenizer.decode(&generated, true).unwrap(),
                            "stop_reason":if generated.last().is_some_and(|id| stop_tokens.contains(id)) {"eos"} else {"max_new_tokens"},
                            "terminal_logits_sha256":sha256_hex(&logit_bytes),
                            "diagnostic_wall_ms_including_cpu_embedding_and_full_logits_readback":started.elapsed().as_secs_f64()*1000.0}));
                    } else {
                        assert_eq!(tokens, generated, "fresh-state replay tokens");
                        assert_eq!(top2s, replay, "fresh-state replay top2 bits");
                    }
                }
                eprintln!("QPN_TARGET_PROGRESS=case_pass name={}", case["name"]);
            }
            stream.synchronize().unwrap();
        }
        let after = cuda_device_allocation_snapshot();
        assert_eq!(after.live_bytes, before.live_bytes);
        assert_eq!(after.live_allocations, before.live_allocations);
        assert_eq!(after.successful_allocations-before.successful_allocations,
            after.successful_frees-before.successful_frees);
        eprintln!("QPN_TARGET_RESULT={}", serde_json::json!({"input_sha256":sha256_hex(&input_bytes),
            "profile_qpn":selected,"resident":resident,"cases":reports,
            "fresh_state_token_and_top2_replay_exact":true,"owner_cleanup_pass":true,
            "peak_engine_owned_cuda_bytes":after.peak_live_bytes,"model_load_count":1,
            "capacity":256,"dflash":false,"vision_forward":false,
            "tokenizer_bound_stop_tokens":stop_tokens,
            "product_ready":false,"semantic_qualified":false,"visible_tps":false}));
    }
    #[cfg(feature = "qwen35-draft-target-verify")]
    #[test]
    fn resident_tokenizer_switch_is_explicit() {
        use std::ffi::OsStr;
        assert!(!resident_tokenizer_switch(None).unwrap());
        assert!(!resident_tokenizer_switch(Some(OsStr::new("0"))).unwrap());
        assert!(resident_tokenizer_switch(Some(OsStr::new("1"))).unwrap());
        for value in ["", "true", "01", "1 ", "2"] {
            assert!(resident_tokenizer_switch(Some(OsStr::new(value))).is_err());
        }
    }
    #[cfg(feature = "qwen35-draft-target-verify")]
    #[test]
    fn resident_tokenizer_lease_exact_content_and_owner_lifetime() {
        let raw = br#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"[UNK]":0,"hello":1,"world":2},"unk_token":"[UNK]"}}"#;
        let cache = ResidentTokenizer::from_bytes(raw).unwrap();
        let direct = Tokenizer::from_bytes(raw).unwrap();
        let weak = Arc::downgrade(&cache.parsed);
        let lease = cache.lease(raw).unwrap();
        for text in ["hello world", "中文 hello", "fn world() {}", ""] {
            let a = direct.encode(text, false).unwrap(); let b = lease.encode(text, false).unwrap();
            assert_eq!(a.get_ids(), b.get_ids());
            assert_eq!(direct.decode(a.get_ids(), false).unwrap(), lease.decode(b.get_ids(), false).unwrap());
        }
        assert_eq!(cache.leases.get(),1);
        let mut changed=raw.to_vec();let at=changed.iter().position(|b|*b==b'2').unwrap();changed[at]=b'3';
        assert!(cache.lease(&changed).is_err());
        assert!(cache.lease(&raw[..raw.len()-1]).is_err());
        assert_eq!(cache.leases.get(),1);
        drop(cache);
        assert!(weak.upgrade().is_some());
        assert_eq!(lease.decode(&[1,2],false).unwrap(),"hello world");
        drop(lease);
        assert!(weak.upgrade().is_none());
        assert!(ResidentTokenizer::from_bytes(b"{}").is_err());
    }
    #[cfg(feature = "qwen35-draft-target-verify")]
    #[test]
    fn request_tokenizer_digest_is_checked_per_lease_not_trusted_by_path_or_length() {
        let raw = br#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"[UNK]":0,"hello":1,"world":2},"unk_token":"[UNK]"}}"#;
        let owner = ResidentTokenizer::from_bytes(raw).unwrap();
        let (parsed, digest) = owner.lease_with_digest(raw).unwrap();
        assert_eq!(digest, sha256_hex(raw));
        assert!(Arc::ptr_eq(&parsed, &owner.parsed));
        let direct = Tokenizer::from_bytes(raw).unwrap();
        for text in ["hello world", "中文 hello", "fn world() {}", ""] {
            let a = direct.encode(text, false).unwrap();
            let b = parsed.encode(text, false).unwrap();
            assert_eq!(a.get_ids(), b.get_ids());
            assert_eq!(direct.decode(a.get_ids(), false).unwrap(), parsed.decode(b.get_ids(), false).unwrap());
        }
        let mut altered = raw.to_vec();
        let index = altered.iter().position(|&b| b == b'2').unwrap();
        altered[index] = b'3';
        assert!(owner.lease_with_digest(&altered).is_err());
        assert!(owner.lease_with_digest(&raw[..raw.len()-1]).is_err());
        assert!(owner.lease_with_digest(&[]).is_err());
        assert_eq!(owner.leases.get(), 1);
        assert_eq!(owner.lease_with_digest(raw).unwrap().1, digest);
        assert_eq!(owner.leases.get(), 2);
        assert!(owner.report()["content_checked_each_lease"].as_bool().unwrap());
        owner.leases.set(u64::MAX);
        assert!(owner.lease_with_digest(raw).is_err());
        assert_eq!(owner.leases.get(), u64::MAX);
        let weak = Arc::downgrade(&owner.parsed);
        drop(owner);
        assert!(weak.upgrade().is_some());
        assert_eq!(parsed.decode(&[1,2], false).unwrap(), "hello world");
        drop(parsed);
        assert!(weak.upgrade().is_none());
    }
    #[test]
    fn planar_head_switch_is_explicit_and_feature_bounded() {
        use std::ffi::OsStr;
        assert!(!planar_head_switch(None,false).unwrap());
        assert!(!planar_head_switch(Some(OsStr::new("0")),false).unwrap());
        assert!(planar_head_switch(Some(OsStr::new("1")),true).unwrap());
        assert!(planar_head_switch(Some(OsStr::new("1")),false).is_err());
        for v in ["", "true", "01", "1 ", " 1"] { assert!(planar_head_switch(Some(OsStr::new(v)),true).is_err()); }
    }
    #[test]
    fn planar_head_codec_preserves_all_code_scale_bits_and_roundtrips() {
        let bytes: Vec<_> = (0..34*8).map(|i| (i*139+37) as u8).collect();
        let p = encode_planar_head(&bytes,2,128).unwrap();
        assert_eq!(p.len(),bytes.len());
        let mut recovered = Vec::new();
        for b in 0..8 { recovered.extend_from_slice(&p[256+b*2..256+b*2+2]); recovered.extend_from_slice(&p[b*32..b*32+32]); }
        assert_eq!(recovered,bytes);
        assert_ne!(p,bytes);
        for (r,c) in [(0,128),(2,127),(usize::MAX,128),(1,128)] { assert!(encode_planar_head(&bytes,r,c).is_err()); }
    }
    #[test]
    fn resident_embedding_rows_are_exact_bf16_and_bounded() {
        let values = [1.0f32, -0.5, 0.0, 4.0];
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|v| half::bf16::from_f32(*v).to_bits().to_le_bytes())
            .collect();
        assert_eq!(decode_embedding_row(&bytes, 2, 2, 0).unwrap(), values[..2]);
        assert_eq!(decode_embedding_row(&bytes, 2, 2, 1).unwrap(), values[2..]);
        assert!(decode_embedding_row(&bytes, 2, 2, 2).is_err());
        assert!(decode_embedding_row(&bytes[..7], 2, 2, 0).is_err());
        assert!(decode_embedding_row(&bytes, usize::MAX, 2, 0).is_err());
        assert!(decode_embedding_row(&bytes, 0, 2, 0).is_err());
        assert!(decode_embedding_row(&[0x80, 0x7f], 1, 1, 0).is_err());
    }
}
