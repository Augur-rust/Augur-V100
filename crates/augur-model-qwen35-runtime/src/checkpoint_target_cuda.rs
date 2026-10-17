//! Complete resident text target for explicit authenticated safetensors.
//! Scalar baseline plus explicit B1..B8 FFN batching, short FP32 or resident
//! KV8 capacity. Each numerical/KV profile needs its own qualification;
//! no DFlash/Vision/HTTP/product admission. All language weights stay in HBM.
use crate::checkpoint_ffn_cuda::CheckpointFfnContext;
use crate::checkpoint_full_cuda::{
    CheckpointCudaOutputHead, CheckpointFullCudaLayer, CheckpointKvPolicy, CheckpointRopeTables,
};
use crate::checkpoint_linear_cuda::{empty, view, CheckpointLinearCudaLayer};
use crate::checkpoint_projection_cuda::CheckpointProjectionPolicy;
use augur_backend::{
    cuda_device_allocation_snapshot, cuda_device_properties, CudaBf16TokenRowGatherArgsV1,
    CudaBf16WeightViewV1, CudaComputeCapability, CudaQwen35VocabTop1ArgsV1, OwnedCudaBuffer,
    OwnedCudaStream,
};
use augur_core::PhysicalLayoutId;
use augur_model_qwen35::{CheckpointArtifact, CheckpointExpectedIdentity, Qwen35CheckpointProfile};
use std::{error::Error, path::Path, rc::Rc, time::Instant};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
#[cfg(feature = "qwen35-augur-dflash-capture")]
pub mod dflash_capture;
#[cfg(feature = "qwen35-augur-dflash-bridge")]
pub mod dflash_bridge;
#[cfg(feature = "qwen35-augur-http")]
mod sampling;
enum Layer {
    Linear(CheckpointLinearCudaLayer),
    Full(CheckpointFullCudaLayer),
}
impl Layer {
    fn submit(&mut self) -> Result<()> {
        match self {
            Self::Linear(l) => l.submit_for_plan(),
            Self::Full(l) => l.submit_for_plan(),
        }
    }
    fn reset(&mut self) -> Result<()> {
        match self {
            Self::Linear(l) => l.reset_empty(),
            Self::Full(l) => l.reset_empty(),
        }
    }
    fn tokens(&self) -> usize {
        match self {
            Self::Linear(l) => l.submitted_tokens(),
            Self::Full(l) => l.submitted_tokens(),
        }
    }
}

#[cfg(feature = "qwen35-augur-dflash-bridge")]
pub(crate) mod augur_prefill;

#[cfg(feature = "qwen35-augur-vision")]
pub mod image_prefill;

pub struct CheckpointCudaTarget {
    #[cfg(feature = "qwen35-augur-vision")]
    vision_host: Option<crate::Qwen35VisionHostWeights>,
    #[cfg(feature = "qwen35-augur-vision")]
    vision_cache: image_prefill::residency::Cache,
    #[cfg(feature = "qwen35-augur-vision")]
    image_rope_active: bool,
    #[cfg(feature = "qwen35-augur-vision")]
    rope: CheckpointRopeTables,
    #[cfg(feature = "qwen35-augur-http")]
    augur_decode: Option<crate::checkpoint_full_cuda::augur_decode::Decode>,
    #[cfg(feature = "qwen35-augur-dflash-bridge")]
    augur_rotated_kv: bool,
    // Same-owner immutable padded source weights outlive request phases, not
    // activation/KV values. Drop their retained layer views before the layers.
    #[cfg(feature = "qwen35-augur-dflash-bridge")]
    augur_sources: Option<std::collections::BTreeMap<usize, Rc<crate::checkpoint_linear_cuda::augur_prefill::Source>>>,
    stream: Rc<OwnedCudaStream>,
    layers: Vec<Layer>,
    head: CheckpointCudaOutputHead,
    gather: CudaBf16TokenRowGatherArgsV1,
    token: OwnedCudaBuffer,
    prompt_tokens: OwnedCudaBuffer,
    prompt_gathers: Vec<CudaBf16TokenRowGatherArgsV1>,
    top1: CudaQwen35VocabTop1ArgsV1,
    selected: Option<u32>,
    #[cfg(feature = "qwen35-augur-http")]
    sampler: Option<sampling::Sampler>,
    #[cfg(feature = "qwen35-augur-http")]
    sampling_trace: crate::augur_sampling::Trace,
    _embedding: OwnedCudaBuffer,
    hidden: [Rc<OwnedCudaBuffer>; 2],
    logits: Rc<OwnedCudaBuffer>,
    _artifact: CheckpointArtifact,
    capacity: usize,
    consumed: usize,
    poisoned: bool,
    logits_current: bool,
    info: serde_json::Value,
    #[cfg(feature = "qwen35-prefill-cublas")]
    ffn_batch: Option<ffn_batch::Batch>,
    #[cfg(feature = "qwen35-augur-dflash-capture")]
    dflash_capture: Option<dflash_capture::Capture>,
}
impl CheckpointCudaTarget {
    pub fn open(
        root: impl AsRef<Path>,
        profile: Qwen35CheckpointProfile,
        identity: CheckpointExpectedIdentity,
        capacity: usize,
    ) -> Result<Self> {
        Self::open_with_projection_policy(
            root,
            profile,
            identity,
            capacity,
            CheckpointProjectionPolicy::OriginalF32,
        )
    }
    pub fn open_with_projection_policy(
        root: impl AsRef<Path>,
        profile: Qwen35CheckpointProfile,
        identity: CheckpointExpectedIdentity,
        capacity: usize,
        policy: CheckpointProjectionPolicy,
    ) -> Result<Self> {
        Self::open_with_execution_policy(
            root,
            profile,
            identity,
            capacity,
            policy,
            CheckpointKvPolicy::F32,
        )
    }
    pub fn open_with_execution_policy(
        root: impl AsRef<Path>,
        profile: Qwen35CheckpointProfile,
        identity: CheckpointExpectedIdentity,
        capacity: usize,
        policy: CheckpointProjectionPolicy,
        kv_policy: CheckpointKvPolicy,
    ) -> Result<Self> {
        if policy.uses_qpn_projections() && profile != Qwen35CheckpointProfile::QuasarNvfp4 {
            return Err("expanded QPN target policy is explicitly Augur-only".into());
        }
        if policy.uses_fp8_warp() && profile != Qwen35CheckpointProfile::NvidiaMixed {
            return Err("FP8 warp target policy is explicitly NVIDIA-only".into());
        }
        if capacity == 0 || capacity > kv_policy.max_capacity()? {
            return Err("checkpoint target capacity exceeds selected KV format".into());
        }
        #[cfg(feature = "qwen35-augur-http")]
        let qpn_startup = crate::checkpoint_projection_cuda::startup_qpn::mode()?;
        if cuda_device_properties(0)?.compute_capability != CudaComputeCapability::SM70 {
            return Err("checkpoint scalar target currently audited on SM70 only".into());
        }
        let started = Instant::now();
        #[cfg(feature = "qwen35-augur-http")]
        let shard_workers = if profile == Qwen35CheckpointProfile::QuasarNvfp4 {
            match std::env::var("AUGUR_SHARD_WORKERS") {
                Ok(value) => value.parse::<usize>()?,
                Err(std::env::VarError::NotPresent) => 5,
                Err(e) => return Err(e.into()),
            }
        } else { 1 };
        #[cfg(not(feature = "qwen35-augur-http"))]
        let shard_workers = 1;
        let mut reader = CheckpointArtifact::open_with_shard_workers(root, profile, identity, shard_workers)?;
        let hash_open_s = started.elapsed().as_secs_f64();
        #[cfg(feature = "qwen35-augur-vision")]
        let vision_load = if profile == Qwen35CheckpointProfile::QuasarNvfp4 {
            Some(crate::vision_cuda::VisionHostLoad::begin(&mut reader).map_err(|e| e.to_string())?)
        } else { None };
        let before = cuda_device_allocation_snapshot();
        let stream = Rc::new(OwnedCudaStream::create(0)?);
        let hidden = [empty(&stream, 5120)?, empty(&stream, 5120)?];
        let ffn = CheckpointFfnContext::new(Rc::clone(&stream), policy)?;
        let rope = CheckpointRopeTables::new(&reader, Rc::clone(&stream), capacity, 0)?;
        let pre_layers_s = started.elapsed().as_secs_f64() - hash_open_s;
        let layers_started = Instant::now();
        let mut layers = Vec::with_capacity(64);
        for index in 0..64 {
            let input = Rc::clone(&hidden[index % 2]);
            let output = Rc::clone(&hidden[(index + 1) % 2]);
            layers.push(if (index + 1) % 4 == 0 {
                Layer::Full(CheckpointFullCudaLayer::load_with_cache_context(
                    &mut reader,
                    index,
                    capacity,
                    0,
                    Rc::clone(&stream),
                    input,
                    output,
                    &ffn,
                    kv_policy,
                    Some(&rope),
                )?)
            } else {
                Layer::Linear(CheckpointLinearCudaLayer::load_with_ffn_context(
                    &mut reader,
                    index,
                    Rc::clone(&stream),
                    input,
                    output,
                    &ffn,
                )?)
            });
            if index % 8 == 7 {
                eprintln!("CHECKPOINT_TARGET_LOAD={profile:?} layers={}", index + 1);
            }
        }
        let layers_s = layers_started.elapsed().as_secs_f64();
        let head_started = Instant::now();
        let logits = empty(&stream, 248320)?;
        let (head, head_detail) = CheckpointCudaOutputHead::load_timed(
            &mut reader,
            Rc::clone(&stream),
            Rc::clone(&hidden[0]),
            Rc::clone(&logits),
        )?;
        let head_load_s = head_started.elapsed().as_secs_f64();
        let embedding_started = Instant::now();
        let embedding_name = "model.language_model.embed_tokens.weight";
        let length = usize::try_from(reader.tensor(embedding_name)?.byte_len()?)?;
        if length != 248320 * 5120 * 2 {
            return Err("complete embedding extent drift".into());
        }
        #[cfg(feature = "qwen35-augur-http")]
        let streamed = crate::checkpoint_projection_cuda::startup_bf16::enabled()?;
        #[cfg(not(feature = "qwen35-augur-http"))]
        let streamed = false;
        let (embedding, embedding_detail) = if streamed {
            #[cfg(feature = "qwen35-augur-http")]
            { crate::checkpoint_projection_cuda::startup_bf16::upload_authenticated(
                &mut reader, embedding_name, &stream,
            )? }
            #[cfg(not(feature = "qwen35-augur-http"))]
            { unreachable!() }
        } else {
            let mut payload = vec![0; length];
            reader.read_tensor_range_into(embedding_name, 0, &mut payload)?;
            let embedding_read_s = embedding_started.elapsed().as_secs_f64();
            let tick = Instant::now();
            if !crate::checkpoint_projection_cuda::bf16_payload_is_finite(&payload) {
                return Err("nonfinite resident embedding".into());
            }
            let embedding_validate_s = tick.elapsed().as_secs_f64();
            let tick = Instant::now();
            let embedding = OwnedCudaBuffer::allocate(0, length)?;
            let embedding_allocate_s = tick.elapsed().as_secs_f64();
            let tick = Instant::now();
            embedding.copy_from_host_sync(&stream, &payload)?;
            let embedding_upload_s = tick.elapsed().as_secs_f64();
            drop(payload);
            (embedding, serde_json::json!({"mode":"authenticated_bf16_pageable_v1",
                "pinned_payload_read_s":embedding_read_s,"finite_scan_s":embedding_validate_s,
                "device_allocate_s":embedding_allocate_s,"synchronous_upload_s":embedding_upload_s,
                "original_bytes":length}))
        };
        let embedding_load_s = embedding_started.elapsed().as_secs_f64();
        let bindings_started = Instant::now();
        let token = OwnedCudaBuffer::allocate(0, 4)?;
        let gather = CudaBf16TokenRowGatherArgsV1::bind(
            &stream,
            CudaBf16WeightViewV1::new(
                embedding.view(0, length)?,
                PhysicalLayoutId::DEVICE_CONTIGUOUS_BF16_CUDA_V1,
                248320,
                5120,
            )?,
            token.view(0, 4)?,
            view(&hidden[0])?,
        )?;
        // Only token IDs are staged, not activations or a second embedding.
        // Prebind every address so ordinary sequence submission allocates no
        // GPU buffer and does not reconstruct native pointer contracts.
        let prompt_tokens = OwnedCudaBuffer::allocate(0, capacity * 4)?;
        let mut prompt_gathers = Vec::with_capacity(capacity);
        for i in 0..capacity {
            prompt_gathers.push(CudaBf16TokenRowGatherArgsV1::bind(
                &stream,
                CudaBf16WeightViewV1::new(
                    embedding.view(0, length)?,
                    PhysicalLayoutId::DEVICE_CONTIGUOUS_BF16_CUDA_V1,
                    248320,
                    5120,
                )?,
                prompt_tokens.view(i * 4, 4)?,
                view(&hidden[0])?,
            )?);
        }
        let top1 = CudaQwen35VocabTop1ArgsV1::bind(&stream, view(&logits)?, token.view(0, 4)?, 1)?;
        stream.synchronize()?;
        let after = cuda_device_allocation_snapshot();
        let info = serde_json::json!({"profile":profile,"policy":policy.label(),
            "source_identity":reader.identity(),"text_layers":64,"text_linears":497,
            "all_language_weights_in_HBM":true,"embedding_resident_bytes":length,
            "owned_live_bytes_including_KV_state_and_scratch":after.live_bytes-before.live_bytes,
            "owned_live_allocations":after.live_allocations-before.live_allocations,
            "shard_auth_workers":shard_workers,"full_shard_hashes_verified":true,
            "startup_detail":{"open_and_hash_s":hash_open_s,"stream_workspace_rope_s":pre_layers_s,
                "layer_load_s":layers_s,"head_load_s":head_load_s,"embedding_load_s":embedding_load_s,
                "head":head_detail,"embedding":embedding_detail,
                "bindings_and_drain_s":bindings_started.elapsed().as_secs_f64()},
            "open_and_hash_s":hash_open_s,"layer_load_s":layers_s,"complete_load_s":started.elapsed().as_secs_f64(),
            "kv_capacity":capacity,"kv_dtype":kv_policy.label(),"activation_dtype":"F32",
            "kv_cache_bytes":capacity*kv_policy.row_bytes()*2*16,
            "shared_rope_table_bytes":capacity*32*4*2,
            "kv8_position_binding_reused":kv_policy==CheckpointKvPolicy::Q8_0,
            "ffn_operand_staging":if policy.qpn_residual() {"F16_hi_plus_scaled_residual"} else if policy.uses_qpn_ffn() {"F16"} else {"F32"},
            "ffn_shared_workspace":policy.uses_qpn_ffn(),
            "qpn_ffn_layers":(0..64).filter(|&i|policy.qpn_ffn_layer(i)).collect::<Vec<_>>(),
            "qpn_global_scale_after_f32_accumulation":policy.qpn_postscale(),
            "qpn_non_ffn_projection_count":if policy.uses_qpn_projections() {208} else {0},
            "qpn_small_48row_gates_remain_original":policy.uses_qpn_projections() && !policy.uses_small_gate_split16(),
            "small_gate_split16_count":if policy.uses_small_gate_split16() {96} else {0},
            "small_gate_numeric_contract":if policy.uses_small_gate_split16() {"f32_pair_contiguous_K16_f64_ordered_split_fold_then_warp_v1"} else {"original_f32_pair_serial_K_f64_then_warp_v1"},
            "small_gate_added_hbm_workspace_bytes":0,
            "small_gate_pair_count":if policy.small_gate_pair_schedule().is_some() {48} else {0},
            "small_gate_pair_schedule":policy.small_gate_pair_schedule().map(|s|format!("{s:?}")),
            "qpn_activation_residual_repair":policy.qpn_residual(),
            "qpn_residual_scale":if policy.qpn_residual() {4096} else {0},
            "qpn_finite_low_component_limit":if policy.qpn_residual() {65504} else {0},
            "qpn_activation_overflow_clamped":false,
            "qpn_f64_chunk_groups":match policy {CheckpointProjectionPolicy::AugurQpnAllRefinedF64 | CheckpointProjectionPolicy::AugurQpnAllWorkspaceF64|CheckpointProjectionPolicy::AugurQpnAllWorkspaceCtaFoldF64|CheckpointProjectionPolicy::AugurQpnAllWorkspaceShapeF64|CheckpointProjectionPolicy::AugurQpnAllWorkspaceShapeSmallF64|CheckpointProjectionPolicy::AugurQpnAllWorkspaceShapeSmallPairF64|CheckpointProjectionPolicy::AugurQpnAllWorkspaceShapeSmallPairPrefetchF64=>4,CheckpointProjectionPolicy::AugurQpnAllRefinedFastF64=>16,_=>0},
            "qpn_residual_chunk_groups":if matches!(policy,CheckpointProjectionPolicy::AugurQpnAllRefinedF64|CheckpointProjectionPolicy::AugurQpnAllWorkspaceF64|CheckpointProjectionPolicy::AugurQpnAllWorkspaceCtaFoldF64|CheckpointProjectionPolicy::AugurQpnAllWorkspaceShapeF64|CheckpointProjectionPolicy::AugurQpnAllWorkspaceShapeSmallF64|CheckpointProjectionPolicy::AugurQpnAllWorkspaceShapeSmallPairF64|CheckpointProjectionPolicy::AugurQpnAllWorkspaceShapeSmallPairPrefetchF64) {4} else {0},
            "qpn_hi_lo_schedule":if policy.qpn_workspace() {"B1..B4_packed_MMA_rows_B5..B8_parallel_CTAs"} else {"in_warp"},
            "qpn_split_k_shape_policy":if policy.qpn_shape_split() {"k/v1024x5120_out5120x6144=32;FFN_qkv_z_q=16;other=8"} else {"gate/up17408x5120=16;other=8"},
            "qpn_partial_component_warps_folded":if matches!(policy,CheckpointProjectionPolicy::AugurQpnAllWorkspaceCtaFoldF64|CheckpointProjectionPolicy::AugurQpnAllWorkspaceShapeF64|CheckpointProjectionPolicy::AugurQpnAllWorkspaceShapeSmallF64|CheckpointProjectionPolicy::AugurQpnAllWorkspaceShapeSmallPairF64|CheckpointProjectionPolicy::AugurQpnAllWorkspaceShapeSmallPairPrefetchF64) {4} else {1},
            "qpn_scalar_f64_partial_workspace_bytes":if policy.qpn_workspace() {17408*16*16} else {0},
            "rmsnorm_schedule":"canonical_shared256_v1","rmsnorm_added_hbm_bytes":0,
            "qpn_b1_live_row_schedule":if policy.qpn_shape_split() || policy==CheckpointProjectionPolicy::AugurQpnAllWorkspaceCtaFoldF64 {"workspace_CTA_fold_live_hi_lo_rows_v1"} else {"not_applicable"},
            "qpn_b1_fused_schedule":if policy.qpn_shape_split() {"profile8_B1_gate_up_Q_CTA512_Z_CTA256_same_fourwarp_fold_v1"} else {"not_applicable"},
            "qpn_input_staging_schedule":"independent_projection_v1",
            "qpn_input_staging_added_hbm_bytes":0,
            "draft_enabled":false,"vision_enabled":false,"native_w4a4":false,"product_ready":false});
        #[cfg(feature = "qwen35-augur-vision")]
        let (vision_host, info) = {
            let mut info = info;
            let host = if let Some(load) = vision_load {
                let (host, detail) = load.finish().map_err(|e| e.to_string())?;
                info["vision_host"] = host.info();
                info["startup_detail"]["vision_host"] = detail;
                Some(host)
            } else { None };
            info["complete_load_s"] = started.elapsed().as_secs_f64().into();
            (host, info)
        };
        #[cfg(feature = "qwen35-augur-http")]
        let info = {
            let mut info = info;
            info["startup_detail"]["qpn_weight_transfer"] = qpn_startup.label().into();
            info
        };
        Ok(Self {
            #[cfg(feature = "qwen35-augur-vision")]
            vision_host,
            #[cfg(feature = "qwen35-augur-vision")]
            vision_cache: image_prefill::residency::Cache::from_env()?,
            #[cfg(feature = "qwen35-augur-vision")]
            image_rope_active: false,
            #[cfg(feature = "qwen35-augur-vision")]
            rope,
            #[cfg(feature = "qwen35-augur-dflash-bridge")]
            augur_rotated_kv: false,
            #[cfg(feature = "qwen35-augur-http")]
            augur_decode: None,
            stream,
            layers,
            head,
            gather,
            token,
            prompt_tokens,
            prompt_gathers,
            top1,
            selected: None,
            #[cfg(feature = "qwen35-augur-http")]
            sampler: None,
            #[cfg(feature = "qwen35-augur-http")]
            sampling_trace: crate::augur_sampling::Trace::default(),
            _embedding: embedding,
            hidden,
            logits,
            _artifact: reader,
            capacity,
            consumed: 0,
            poisoned: false,
            logits_current: false,
            info,
            #[cfg(feature = "qwen35-prefill-cublas")]
            ffn_batch: None,
            #[cfg(feature = "qwen35-augur-dflash-capture")]
            dflash_capture: None,
            #[cfg(feature = "qwen35-augur-dflash-bridge")]
            augur_sources: None,
        })
    }
    pub fn info(&self) -> &serde_json::Value {
        &self.info
    }
    pub fn consumed_tokens(&self) -> usize {
        self.consumed
    }

    /// Change only129 existing hidden5120 norm schedules, atomically before use.
    /// Preserves the original per-lane F32 sums and256-lane addition tree.
    pub fn enable_exact_warp_tree_norm(&mut self) -> Result<()> {
        if self.poisoned || self.consumed != 0
            || self.info["rmsnorm_schedule"] != "canonical_shared256_v1" {
            return Err("norm schedule setup requires a fresh canonical target".into());
        }
        #[cfg(feature = "qwen35-prefill-cublas")]
        if self.ffn_batch.is_some() {
            return Err("choose norm schedule before FFN batching".into());
        }
        let mut norms = Vec::with_capacity(129);
        for layer in &mut self.layers {
            norms.extend(match layer {
                Layer::Linear(l) => l.norm_bindings_mut(),
                Layer::Full(l) => l.norm_bindings_mut(),
            });
        }
        norms.push(self.head.norm_binding_mut());
        if norms.len() != 129 {
            return Err("complete target norm inventory changed".into());
        }
        let replacements = norms.iter().map(|norm| (**norm).with_exact_warp_tree5120())
            .collect::<std::result::Result<Vec<_>, _>>()?;
        for (norm, replacement) in norms.into_iter().zip(replacements) { *norm = replacement; }
        self.info["rmsnorm_schedule"] = serde_json::json!("same_tree_warp_cached5120_v1");
        Ok(())
    }

    /// Fuse only the64 B1 input norms with their144 attention/GDN QPN consumers.
    /// Prepares ALL immutable bindings before mutation. FFN batching stays as-is:
    /// no row-local staging into a B8 arena with the wrong residual-plane stride.
    #[cfg(feature = "qwen35-prefill-cublas")]
    pub fn enable_fused_input_qpn_norm(&mut self) -> Result<()> {
        if self.poisoned || self.consumed!=0 || self.ffn_batch.is_some()
            || self.info["rmsnorm_schedule"]!="same_tree_warp_cached5120_v1"
            || self.info["qpn_input_staging_schedule"]!="independent_projection_v1" {
            return Err("fused input staging requires a fresh cached-norm target before FFN setup".into());
        }
        let programs=self.layers.iter().map(|layer| match layer {
            Layer::Linear(l)=>l.prepare_input_norm_qpn(),
            Layer::Full(l)=>l.prepare_input_norm_qpn(),
        }).collect::<Result<Vec<_>>>()?;
        let post_programs=self.layers.iter().map(|layer| match layer {
            Layer::Linear(l)=>l.prepare_post_norm_qpn(),
            Layer::Full(l)=>l.prepare_post_norm_qpn(),
        }).collect::<Result<Vec<_>>>()?;
        if programs.len()!=64 || post_programs.len()!=64 {return Err("norm inventory changed".into());}
        for ((layer,program),post) in self.layers.iter_mut().zip(programs).zip(post_programs) {
            match layer {
                Layer::Linear(l)=>{l.install_input_norm_qpn(program);l.install_post_norm_qpn(post);},
                Layer::Full(l)=>{l.install_input_norm_qpn(program);l.install_post_norm_qpn(post);},
            }
        }
        self.info["qpn_input_staging_schedule"]=serde_json::json!("fused_cached_norm_qkv_v1");
        self.info["qpn_fused_input_norm_count"]=serde_json::json!(64);
        self.info["qpn_fused_input_projection_count"]=serde_json::json!(144);
        self.info["qpn_fused_post_norm_count"]=serde_json::json!(64);
        Ok(())
    }

    /// Explicit setup-time opt-in, not a product default. Attention/GDN stays
    /// scalar and causal; only FFN projections run at widths1..8. No weight
    /// duplication, new quantization, history truncation or padded token.
    #[cfg(feature = "qwen35-prefill-cublas")]
    pub fn enable_batched_ffn8(&mut self) -> Result<()> {
        if self.poisoned || self.consumed != 0 || self.ffn_batch.is_some() {
            return Err("FFN batch setup requires a fresh empty target".into());
        }
        let started = Instant::now();
        let before = cuda_device_allocation_snapshot();
        let batch = ffn_batch::Batch::new(self)?;
        self.stream.synchronize()?;
        let after = cuda_device_allocation_snapshot();
        let bytes = after.live_bytes - before.live_bytes;
        let allocations = after.live_allocations - before.live_allocations;
        self.info["ffn_batch_max_rows"] = serde_json::json!(8);
        self.info["ffn_batch_numeric_contract"] =
            serde_json::json!("same_per_layer_profile_no_weight_or_activation_conversion_v1");
        self.info["ffn_batch_setup_s"] = serde_json::json!(started.elapsed().as_secs_f64());
        self.info["ffn_batch_added_live_bytes"] = serde_json::json!(bytes);
        self.info["qpn_batch_f64_partial_workspace_bytes"] = serde_json::json!(
            self.info["qpn_scalar_f64_partial_workspace_bytes"]
                .as_u64()
                .unwrap_or(0)
                * 8
        );
        self.info["owned_live_bytes_including_KV_state_and_scratch"] = serde_json::json!(
            self.info["owned_live_bytes_including_KV_state_and_scratch"]
                .as_u64()
                .unwrap()
                + bytes
        );
        self.info["owned_live_allocations"] =
            serde_json::json!(self.info["owned_live_allocations"].as_u64().unwrap() + allocations);
        self.ffn_batch = Some(batch);
        Ok(())
    }

    /// Original target step. Captures are explicit diagnostic D2H and must not
    /// be used for timing. Empty capture list keeps all layer intermediates HBM.
    pub fn consume(
        &mut self,
        token_id: u32,
        with_logits: bool,
        capture_layers: &[usize],
    ) -> Result<Vec<(usize, Vec<f32>)>> {
        if self.poisoned
            || self.consumed >= self.capacity
            || token_id >= 248320
            || capture_layers.iter().any(|&i| i >= 64)
            || capture_layers.windows(2).any(|w| w[0] >= w[1])
        {
            return Err("invalid target token/capacity/capture or poisoned owner".into());
        }
        if self.layers.iter().any(|l| l.tokens() != self.consumed) {
            self.poisoned = true;
            return Err("layer/session cursor mismatch".into());
        }
        self.poisoned = true;
        self.logits_current = false;
        self.selected = None;
        #[cfg(feature = "qwen35-augur-dflash-capture")]
        if let Some(capture) = &mut self.dflash_capture {
            capture.begin(self.consumed, 1)?;
        }
        self.token
            .copy_from_host_sync(&self.stream, &token_id.to_le_bytes())?;
        self.gather.submit_for_plan(&self.stream)?;
        let mut captured = Vec::with_capacity(capture_layers.len());
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.submit()?;
            #[cfg(feature = "qwen35-augur-dflash-capture")]
            if let Some(capture) = &mut self.dflash_capture {
                capture.layer(i, self.consumed, 1, &self.hidden[(i + 1) % 2])?;
            }
            if capture_layers.binary_search(&i).is_ok() {
                captured.push((i, read_f32(&self.stream, &self.hidden[(i + 1) % 2])?));
            }
        }
        if with_logits {
            self.head.submit_for_plan()?;
        }
        self.stream.synchronize()?;
        #[cfg(feature = "qwen35-augur-dflash-capture")]
        if let Some(capture) = &mut self.dflash_capture { capture.commit()?; }
        self.consumed += 1;
        self.logits_current = with_logits;
        self.poisoned = false;
        Ok(captured)
    }

    /// Append an exact known token sequence with one H2D upload and one final
    /// stream barrier. Attention/GDN stays scalar and ordered. Explicit FFN
    /// batching groups widths1..8; this is not fully batched attention/prefill.
    /// All IDs/capacity are checked before altering the owner.
    pub fn consume_sequence(&mut self, tokens: &[u32]) -> Result<()> {
        self.require_unrotated_cache()?;
        validate_sequence(self.consumed, self.capacity, tokens)?;
        self.require_unrotated_cache()?;
        self.validate_cursor()?;
        self.poisoned = true;
        self.logits_current = false;
        self.selected = None;
        #[cfg(feature = "qwen35-augur-dflash-capture")]
        if let Some(capture) = &mut self.dflash_capture {
            capture.begin(self.consumed, tokens.len())?;
        }
        let bytes: Vec<u8> = tokens.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.prompt_tokens
            .copy_from_host_sync(&self.stream, &bytes)?;
        #[cfg(feature = "qwen35-prefill-cublas")]
        let batched = if let Some(batch) = &self.ffn_batch {
            batch.submit(
                &self.stream,
                &mut self.layers,
                &self.prompt_gathers,
                &self.hidden,
                tokens.len(),
                self.consumed,
                #[cfg(feature = "qwen35-augur-dflash-capture")]
                self.dflash_capture.as_mut(),
            )?;
            true
        } else {
            false
        };
        #[cfg(not(feature = "qwen35-prefill-cublas"))]
        let batched = false;
        if !batched {
            for i in 0..tokens.len() {
                self.prompt_gathers[i].submit_for_plan(&self.stream)?;
                for (index, layer) in self.layers.iter_mut().enumerate() {
                    layer.submit()?;
                    #[cfg(feature = "qwen35-augur-dflash-capture")]
                    if let Some(capture) = &mut self.dflash_capture {
                        capture.layer(index, self.consumed + i, 1, &self.hidden[(index + 1) % 2])?;
                    }
                    let _ = index;
                }
            }
        }
        self.head.submit_for_plan()?;
        self.stream.synchronize()?;
        #[cfg(feature = "qwen35-augur-dflash-capture")]
        if let Some(capture) = &mut self.dflash_capture { capture.commit()?; }
        self.consumed += tokens.len();
        self.logits_current = true;
        self.poisoned = false;
        Ok(())
    }

    fn require_unrotated_cache(&self) -> Result<()> {
        #[cfg(feature = "qwen35-augur-dflash-bridge")]
        if self.augur_rotated_kv {
            return Err("Augur rotated KV is prefill-only; reset before ordinary scalar/HTTP decode".into());
        }
        Ok(())
    }

    fn validate_cursor(&mut self) -> Result<()> {
        if self.poisoned {
            return Err("checkpoint target owner is poisoned".into());
        }
        if self.layers.iter().any(|l| l.tokens() != self.consumed) {
            self.poisoned = true;
            return Err("layer/session cursor mismatch".into());
        }
        Ok(())
    }

    /// Same greedy rule as the independent host oracle: lowest ID on a tie,
    /// and any nonfinite vocabulary entry rejects the entire selection.
    /// Reads only4 bytes, draining all prior work on this exact owner's stream.
    pub fn greedy_next_token(&mut self) -> Result<u32> {
        #[cfg(feature = "qwen35-augur-http")]
        if self.sampler.is_some() {return Err("greedy diagnostic called on sampled target".into());}
        self.select_next_token()
    }

    /// Select under the configured target law; cached selections also own the device token slot.
    pub fn select_next_token(&mut self) -> Result<u32> {
        self.validate_cursor()?;
        if !self.logits_current {
            return Err("no current committed logits".into());
        }
        if let Some(id) = self.selected {
            #[cfg(feature="qwen35-augur-http")]
            if self.sampler.is_some(){self.sampling_trace.record(self.consumed,id,None)?;}
            return Ok(id);
        }
        self.poisoned = true;
        #[cfg(feature = "qwen35-augur-http")]
        if let Some(sampler)=&self.sampler {
            sampler.submit()?;let selection=sampler.read(self.consumed)?[0];
            self.cache_sampled_selection(selection)?;self.poisoned=false;return Ok(selection.id);
        }
        self.top1.submit(&self.stream)?;
        let mut bytes = [0; 4];
        self.token.copy_to_host_sync(&self.stream, &mut bytes)?;
        let id = u32::from_le_bytes(bytes);
        if id >= 248320 {
            return Err("nonfinite or invalid greedy target selection".into());
        }
        self.selected = Some(id);
        self.poisoned = false;
        Ok(id)
    }

    /// Consume the last selected token directly from the retained device slot.
    /// This enqueues work; the next selection/snapshot/reset drains it. EOS is
    /// a caller stop policy, not silently consumed by this method.
    pub fn consume_selected(&mut self) -> Result<()> {
        self.require_unrotated_cache()?;
        self.validate_cursor()?;
        if self.consumed >= self.capacity || !self.logits_current || self.selected.is_none() {
            return Err("missing/stale selection or full checkpoint target".into());
        }
        self.poisoned = true;
        self.logits_current = false;
        self.selected = None;
        #[cfg(feature = "qwen35-augur-dflash-capture")]
        if let Some(capture) = &mut self.dflash_capture { capture.begin(self.consumed, 1)?; }
        self.gather.submit_for_plan(&self.stream)?;
        for (index, layer) in self.layers.iter_mut().enumerate() {
            layer.submit()?;
            #[cfg(feature = "qwen35-augur-dflash-capture")]
            if let Some(capture) = &mut self.dflash_capture {
                capture.layer(index, self.consumed, 1, &self.hidden[(index + 1) % 2])?;
            }
            let _ = index;
        }
        self.head.submit_for_plan()?;
        #[cfg(feature = "qwen35-augur-dflash-capture")]
        if let Some(capture) = &mut self.dflash_capture { capture.commit()?; }
        self.consumed += 1;
        self.logits_current = true;
        self.poisoned = false;
        Ok(())
    }
    #[cfg(feature = "qwen35-augur-http")]
    pub(crate) fn enable_augur_decode(&mut self) -> Result<()> {
        self.validate_cursor()?;
        if self.augur_decode.is_some() || self.consumed != 0 || self.augur_rotated_kv
            || self.dflash_capture.is_some() {
            return Err("rotated decoder install requires a fresh empty owner before feature capture".into());
        }
        self.enable_fixed_b1_graphs()?;
        self.augur_decode = Some(crate::checkpoint_full_cuda::augur_decode::Decode::new(Rc::clone(&self.stream), self.capacity)?);
        self.info["decode_attention"] = self.augur_decode.as_ref().unwrap().label().into();
        self.info["attention_mode"] = self.attention_mode().label().into();
        self.info["kv_identity"] = self.attention_mode().kv_identity().into();
        Ok(())
    }
    #[cfg(feature = "qwen35-augur-http")]
    pub(crate) fn consume_selected_augur(&mut self) -> Result<()> {
        self.validate_cursor()?;
        if !self.augur_rotated_kv || self.augur_decode.is_none()
            || self.info["active_kv_format"] != self.attention_mode().kv_identity()
            || self.consumed >= self.capacity || !self.logits_current || self.selected.is_none() {
            return Err("missing rotated KV/decoder/selection or full target".into());
        }
        self.poisoned = true;
        self.logits_current = false;
        self.selected = None;
        if let Some(c) = &mut self.dflash_capture { c.begin(self.consumed, 1)?; }
        self.gather.submit_for_plan(&self.stream)?;
        for (index, layer) in self.layers.iter_mut().enumerate() {
            match layer {
                Layer::Linear(l) => l.submit_for_plan()?,
                Layer::Full(l) => l.submit_augur_decode(self.augur_decode.as_ref().unwrap())?,
            }
            if let Some(c) = &mut self.dflash_capture { c.layer(index, self.consumed, 1, &self.hidden[(index + 1) % 2])?; }
        }
        self.head.submit_for_plan()?;
        if let Some(c) = &mut self.dflash_capture { c.commit()?; }
        self.consumed += 1;
        self.logits_current = true;
        self.poisoned = false;
        Ok(())
    }
    pub fn logits_snapshot(&mut self) -> Result<Vec<f32>> {
        if self.poisoned || !self.logits_current {
            return Err("no current committed logits".into());
        }
        match read_f32(&self.stream, &self.logits) {
            Ok(values) if values.iter().all(|v| v.is_finite()) => Ok(values),
            Ok(_) => {
                self.poisoned = true;
                Err("nonfinite complete target logits".into())
            }
            Err(e) => {
                self.poisoned = true;
                Err(e)
            }
        }
    }
    /// Owner cancellation/transport retirement barrier; never clears poison or
    /// resets the cursor. Only the existing explicit reset may clear state.
    #[cfg(feature = "qwen35-augur-session")]
    pub(crate) fn drain_session_work(&mut self) -> Result<()> {
        self.stream.synchronize().map_err(|error| {
            self.poisoned = true;
            error.into()
        })
    }

    pub fn reset_empty(&mut self) -> Result<()> {
        #[cfg(feature="qwen35-augur-http")]
        {self.sampling_trace=Default::default();}
        self.poisoned = true;
        self.logits_current = false;
        self.selected = None;
        self.stream.synchronize()?;
        for layer in &mut self.layers {
            layer.reset()?;
        }
        #[cfg(feature = "qwen35-augur-dflash-capture")]
        if let Some(capture) = &mut self.dflash_capture { capture.reset()?; }
        #[cfg(feature = "qwen35-augur-dflash-bridge")]
        {
            if self.augur_rotated_kv {
                self.info.as_object_mut().ok_or("target metadata is not an object")?.remove("active_kv_format");
            }
            self.augur_rotated_kv = false;
        }
        #[cfg(feature = "qwen35-augur-vision")]
        if self.image_rope_active {
            self.rope.restore_text()?;
            self.image_rope_active=false;
            self.info.as_object_mut().ok_or("target metadata")?.remove("active_image_input");
        }
        self.consumed = 0;
        self.poisoned = false;
        Ok(())
    }
}

fn validate_sequence(consumed: usize, capacity: usize, tokens: &[u32]) -> Result<()> {
    if tokens.is_empty()
        || tokens.iter().any(|&id| id >= 248320)
        || consumed
            .checked_add(tokens.len())
            .is_none_or(|end| end > capacity)
    {
        return Err("invalid checkpoint sequence IDs or capacity".into());
    }
    Ok(())
}
fn read_f32(stream: &OwnedCudaStream, buffer: &OwnedCudaBuffer) -> Result<Vec<f32>> {
    let mut bytes = vec![0; buffer.len_bytes()];
    buffer.copy_to_host_sync(stream, &mut bytes)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}
impl Drop for CheckpointCudaTarget {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
    }
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod transfer_tests;
#[cfg(test)]
mod context_timing_tests;
#[cfg(all(test, feature = "qwen35-augur-http"))]
mod native_decode_tests;

#[cfg(feature = "qwen35-prefill-cublas")]
mod ffn_batch {
    use super::*;
    use crate::checkpoint_ffn_cuda::{
        CheckpointFfnBatchProgram, CheckpointFfnBatchSource, CheckpointFfnBatchWorkspace,
    };
    use augur_backend::{CudaAddF32ArgsV1, CudaF32View};
    const H: usize = 5120;
    const B: usize = 8;
    type Buffer = Rc<OwnedCudaBuffer>;

    pub(super) struct Batch {
        // All plans/weights/IO allocations are retained; Drop drains on error.
        stream: Rc<OwnedCudaStream>,
        hidden: [Buffer; 2],
        normalized: Buffer,
        residual: Buffer,
        _ffn_output: Buffer,
        sources: Vec<(Buffer, Buffer)>,
        gdn_prefixes: Option<Vec<Option<crate::checkpoint_linear_cuda::prefill_prefix::Program>>>,
        programs: Vec<Vec<CheckpointFfnBatchProgram>>,
        _workspace: CheckpointFfnBatchWorkspace,
        adds: [Vec<CudaAddF32ArgsV1>; 2],
        #[cfg(test)]
        pub(super) fail_after_layer: Option<usize>,
        #[cfg(all(test, feature = "qwen35-augur-dflash-capture"))]
        pub(super) diagnostic_capture: Option<std::cell::RefCell<Vec<(usize, usize, Vec<f32>)>>>,
    }
    fn binding(layer: &Layer) -> Result<(CheckpointFfnBatchSource<'_>, Buffer, Buffer)> {
        match layer {
            Layer::Linear(l) => l.ffn_batch_binding(),
            Layer::Full(l) => l.ffn_batch_binding(),
        }
    }
    fn slice(buffer: &OwnedCudaBuffer, rows: usize) -> Result<CudaF32View> {
        Ok(CudaF32View::new(
            buffer.view(0, rows * H * 4)?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            rows * H,
        )?)
    }
    fn copy(
        stream: &OwnedCudaStream,
        destination: &OwnedCudaBuffer,
        dst_row: usize,
        source: &OwnedCudaBuffer,
        src_row: usize,
    ) -> Result<()> {
        // SAFETY: target and Batch retain exact owners on this single stream;
        // neither is mutated elsewhere. Both Drop implementations drain even
        // after partial submission errors. Rows are checked by owned views.
        unsafe {
            destination.copy_from_device_async_unchecked(
                stream,
                dst_row * H * 4,
                source.view(src_row * H * 4, H * 4)?,
            )?;
        }
        Ok(())
    }
    impl Batch {
        pub(super) fn new(target: &CheckpointCudaTarget) -> Result<Self> {
            // Preserve each layer's numerical profile; share a small arena,
            // never upload a second copy of its retained weights.
            let mut need_qpn = false;
            let mut need_exact = false;
            let mut need_residual = false;
            let mut need_partial_workspace = false;
            for layer in &target.layers {
                need_residual |= binding(layer)?.0.is_residual();
                need_partial_workspace |= binding(layer)?.0.uses_partial_workspace();
                if binding(layer)?.0.is_qpn() {
                    need_qpn = true;
                } else {
                    need_exact = true;
                }
            }
            let stream = Rc::clone(&target.stream);
            let workspace = CheckpointFfnBatchWorkspace::new(
                Rc::clone(&stream),
                need_qpn,
                need_exact,
                need_residual,
                need_partial_workspace,
            )?;
            let hidden = [empty(&stream, B * H)?, empty(&stream, B * H)?];
            let normalized = empty(&stream, B * H)?;
            let residual = empty(&stream, B * H)?;
            let output = empty(&stream, B * H)?;
            let mut programs = Vec::with_capacity(64);
            let mut sources = Vec::with_capacity(64);
            for layer in &target.layers {
                let (original, norm, res) = binding(layer)?;
                let mut widths = Vec::with_capacity(B);
                for width in 1..=B {
                    widths.push(workspace.bind(
                        original,
                        Rc::clone(&normalized),
                        Rc::clone(&output),
                        width,
                    )?);
                }
                programs.push(widths);
                sources.push((norm, res));
            }
            let mut adds = [Vec::with_capacity(B), Vec::with_capacity(B)];
            for parity in 0..2 {
                for width in 1..=B {
                    adds[parity].push(CudaAddF32ArgsV1::bind(
                        &stream,
                        slice(&residual, width)?,
                        slice(&output, width)?,
                        slice(&hidden[parity], width)?,
                    )?);
                }
            }
            Ok(Self {
                stream,
                hidden,
                normalized,
                residual,
                _ffn_output: output,
                sources,
                gdn_prefixes: None,
                programs,
                _workspace: workspace,
                adds,
                #[cfg(test)]
                fail_after_layer: None,
                #[cfg(all(test, feature = "qwen35-augur-dflash-capture"))]
                diagnostic_capture: None,
            })
        }
        /// Diagnostic comparison runner: same current scalar-prefix/B8-FFN
        /// programs and residuals, without embedding/head or host readback.
        #[cfg(all(test, feature = "qwen35-augur-dflash-capture"))]
        pub(super) fn diagnostic_one_layer(&self, stream:&OwnedCudaStream, index:usize,
            layer:&mut Layer, scalar:&[Buffer;2], input:&OwnedCudaBuffer, output:&OwnedCudaBuffer, tokens:usize)->Result<()> {
            if index>=self.programs.len() || tokens==0 {return Err("invalid old layer comparison".into());}
            for first in (0..tokens).step_by(B) {
                let width=B.min(tokens-first);
                for row in 0..width {
                    copy(stream,&scalar[index%2],0,input,first+row)?;
                    match layer {Layer::Linear(l)=>l.submit_attention_for_ffn_batch()?,Layer::Full(l)=>l.submit_attention_for_ffn_batch()?}
                    copy(stream,&self.normalized,row,&self.sources[index].0,0)?;
                    copy(stream,&self.residual,row,&self.sources[index].1,0)?;
                }
                self.programs[index][width-1].submit_for_plan_on(stream)?;
                self.adds[(index+1)%2][width-1].submit_for_plan(stream)?;
                // SAFETY: caller and Batch retain all exact stream owners.
                unsafe { output.copy_from_device_async_unchecked(stream,first*H*4,
                    self.hidden[(index+1)%2].view(0,width*H*4)?)?; }
            }
            Ok(())
        }
        pub(super) fn enable_gdn_prefixes(&mut self, target: &CheckpointCudaTarget) -> Result<()> {
            use crate::checkpoint_linear_cuda::prefill_prefix::{Workspace, Program};
            if self.gdn_prefixes.is_some() { return Err("GDN prefill prefix already installed".into()); }
            let workspace = Rc::new(Workspace::new(Rc::clone(&self.stream))?);
            let mut programs = Vec::with_capacity(target.layers.len());
            for (index, layer) in target.layers.iter().enumerate() {
                programs.push(match layer {
                    Layer::Linear(layer) => Some(Program::bind(layer, Rc::clone(&workspace),
                        Rc::clone(&self.hidden[index % 2]), target._artifact.config().rms_norm_eps as f32)?),
                    Layer::Full(_) => None,
                });
            }
            self.gdn_prefixes = Some(programs);
            Ok(())
        }
        pub(super) fn submit(
            &self,
            stream: &OwnedCudaStream,
            layers: &mut [Layer],
            gathers: &[CudaBf16TokenRowGatherArgsV1],
            scalar: &[Buffer; 2],
            tokens: usize,
            absolute_start: usize,
            #[cfg(feature = "qwen35-augur-dflash-capture")]
            mut capture: Option<&mut dflash_capture::Capture>,
        ) -> Result<()> {
            let _ = absolute_start;
            if tokens == 0 || tokens > gathers.len() || layers.len() != self.programs.len() {
                return Err("invalid FFN batch submission geometry".into());
            }
            for first in (0..tokens).step_by(B) {
                let width = B.min(tokens - first);
                for row in 0..width {
                    gathers[first + row].submit_for_plan(stream)?;
                    copy(stream, &self.hidden[0], row, &scalar[0], 0)?;
                }
                for (index, layer) in layers.iter_mut().enumerate() {
                    // Width1 keeps the already-fused scalar prefix: the measured
                    // projection-only B1 tie cannot justify extra row copies.
                    let prefix = self.gdn_prefixes.as_ref().and_then(|p| p[index].as_ref());
                    if let (Some(prefix), Layer::Linear(linear)) = (prefix.filter(|_| width > 1), &mut *layer) {
                        prefix.submit_rows(linear, width, &self.normalized, &self.residual)?;
                    } else {
                    for row in 0..width {
                        copy(stream, &scalar[index % 2], 0, &self.hidden[index % 2], row)?;

                        match layer {
                            Layer::Linear(l) => l.submit_attention_for_ffn_batch()?,
                            Layer::Full(l) => l.submit_attention_for_ffn_batch()?,
                        }

                        copy(stream, &self.normalized, row, &self.sources[index].0, 0)?;
                        copy(stream, &self.residual, row, &self.sources[index].1, 0)?;
                    }
                    }

                    self.programs[index][width - 1].submit_for_plan_on(stream)?;


                    self.adds[(index + 1) % 2][width - 1].submit_for_plan(stream)?;

                    #[cfg(feature = "qwen35-augur-dflash-capture")]
                    if let Some(capture) = capture.as_deref_mut() {
                        capture.layer(index, absolute_start + first, width, &self.hidden[(index + 1) % 2])?;
                    }
                    // Independent diagnostic reference: D2H of actual post-add
                    // rows. Test-only, never in a product/performance binary.
                    #[cfg(all(test, feature = "qwen35-augur-dflash-capture"))]
                    if let Some(reference) = &self.diagnostic_capture {
                        if crate::DFLASH_TARGET_CAPTURE_LAYER_IDS.contains(&index) {
                            let values = read_f32(stream, &self.hidden[(index + 1) % 2])?;
                            for row in 0..width {
                                reference.borrow_mut().push((absolute_start + first + row, index,
                                    values[row * H..(row + 1) * H].to_vec()));
                            }
                        }
                    }
                    #[cfg(test)]
                    if self.fail_after_layer == Some(index) {
                        return Err("injected partial FFN batch submission failure".into());
                    }
                }
            }
            copy(stream, &scalar[0], 0, &self.hidden[0], (tokens - 1) % B)?;
            Ok(())
        }
    }
    impl Drop for Batch {
        fn drop(&mut self) {
            let _ = self.stream.synchronize();
        }
    }
}

#[cfg(all(windows, feature="qwen35-augur-http"))]
pub(crate) mod checkpoint_io;

#[cfg(feature = "qwen35-augur-http")]
fn b1_linear_graph_switch(value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        _ => Err("AUGUR_B1_LINEAR_GRAPH must be 0 or 1".into()),
    }
}

#[cfg(feature = "qwen35-augur-http")]
mod b1_graph;
