#![cfg(feature = "qwen35-full-attention")]

//! Real-weight Qwen35 full-attention CUDA component.
//!
//! This boundary deliberately keeps Q/K/V projection and normalization on the
//! CPU reference side.  The device executes the causal GQA cache append,
//! softmax, output gate, and the real Q8_0 attention-output projection.  It is
//! therefore useful for measuring the attention hardware cost without
//! pretending that a complete 64-layer decoder already exists.

use std::{error::Error, fmt, hint::spin_loop, path::PathBuf, time::Instant};

use augur_backend::{
    cuda_device_allocation_snapshot, cuda_device_properties, cuda_host_boundary_snapshot,
    CudaF32View, CudaQ8_0F32MatvecArgsV1, CudaQ8_0WeightViewV1, CudaQwen35FullAttentionArgsV1,
    CudaQwen35FullAttentionViewsV1, CudaRuntimeError, OwnedCudaBuffer, OwnedCudaEvent,
    OwnedCudaStream,
};
#[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
use augur_backend::{
    CudaQ8_1ActivationQuantizeArgsV1, CudaQ8_1ActivationViewV1, CudaQwen35FullAttentionArgsV2,
    CudaQwen35FullAttentionViewsV2, QWEN35_FULL_ATTENTION_FUSED_Q8_BYTES_V2,
    QWEN35_FULL_ATTENTION_MERGE_SEGMENTS_V2, QWEN35_FULL_ATTENTION_MERGE_THREADS_V2,
    QWEN35_FULL_ATTENTION_PARTIAL_ELEMENTS_V2, QWEN35_FULL_ATTENTION_SPLITS_V2,
    QWEN35_FULL_ATTENTION_STAGE1_THREADS_V2, QWEN35_FULL_ATTENTION_TILE_POSITIONS_V2,
    QWEN35_FULL_ATTENTION_WORKSPACE_ELEMENTS_V2,
};
#[cfg(feature = "qwen35-kv8-full-attention")]
use augur_backend::{
    CudaQwen35Kv8FullAttentionArgsV3, CudaQwen35Kv8FullAttentionViewsV3, QWEN35_KV8_CACHE_ROW_BYTES,
};
use augur_core::PhysicalLayoutId;
use augur_model_qwen35::{
    dequantize_bf16, dequantize_tensor, quantized_matvec_fused_simd, rms_norm_last_dim,
    split_interleaved_query_gate, target_rope_last_dim, GgufAuditOptions, GgufManifest, GgufTensor,
    Qwen35Config, Qwen35Error, Qwen35FullAttentionState, ReferenceTensor, GGML_TYPE_BF16,
    GGML_TYPE_F32, GGML_TYPE_Q8_0, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMENTS,
};
use serde::Serialize;
use sha2::{Digest, Sha256};

pub const QWEN35_REAL_FULL_ATTENTION_CUDA_SCHEMA: &str =
    "crystal-quant.qwen35.real-full-attention-cuda.v1";

pub(crate) const HIDDEN: usize = 5_120;
pub(crate) const VOCAB: usize = 248_320;
pub(crate) const QUERY_HEADS: usize = 24;
pub(crate) const KV_HEADS: usize = 4;
pub(crate) const HEAD_DIM: usize = 256;
pub(crate) const ROTARY_DIM: usize = 64;
pub(crate) const QUERY_VALUES: usize = QUERY_HEADS * HEAD_DIM;
pub(crate) const QUERY_GATE_VALUES: usize = QUERY_VALUES * 2;
pub(crate) const KV_VALUES: usize = KV_HEADS * HEAD_DIM;
const STRICT_PROFILE_TENSOR_COUNT: u64 = 1_187;
const MAX_COMPLETION_POLLS: u64 = 2_000_000;
#[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
const GROUPED_GQA_WORKSPACE_BYTES: usize = QWEN35_FULL_ATTENTION_WORKSPACE_ELEMENTS_V2 * 4;
#[cfg(not(feature = "qwen35-grouped-gqa-online-softmax"))]
const GROUPED_GQA_WORKSPACE_BYTES: usize = 0;

pub(crate) const EMBEDDING: &str = "token_embd.weight";
pub(crate) const ATTN_NORM: &str = "attn_norm.weight";
pub(crate) const Q: &str = "attn_q.weight";
pub(crate) const K: &str = "attn_k.weight";
pub(crate) const V: &str = "attn_v.weight";
pub(crate) const Q_NORM: &str = "attn_q_norm.weight";
pub(crate) const K_NORM: &str = "attn_k_norm.weight";
pub(crate) const OUTPUT: &str = "attn_output.weight";
pub(crate) const POST_ATTN_NORM: &str = "post_attention_norm.weight";

#[derive(Clone, Debug)]
pub struct Qwen35RealFullAttentionCudaConfig {
    pub gguf_path: PathBuf,
    pub config_path: PathBuf,
    /// Zero-based layer index. The selected layer must be in the authenticated
    /// full-attention partition.
    pub layer: usize,
    pub token_id: usize,
    pub device_ordinal: i32,
    pub capacity: usize,
    pub warmup: usize,
    pub iterations: usize,
    pub source_commit: String,
}

impl Default for Qwen35RealFullAttentionCudaConfig {
    fn default() -> Self {
        Self {
            gguf_path: PathBuf::new(),
            config_path: PathBuf::new(),
            layer: 3,
            token_id: 1,
            device_ordinal: 0,
            capacity: 2,
            warmup: 2,
            iterations: 20,
            source_commit: "UNBOUND_COMPONENT_SOURCE".to_owned(),
        }
    }
}

#[derive(Debug)]
pub enum Qwen35RealFullAttentionCudaError {
    Runtime(CudaRuntimeError),
    Model(Qwen35Error),
    InvalidConfiguration(&'static str),
    CompletionPollLimit,
}

impl fmt::Display for Qwen35RealFullAttentionCudaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Runtime(error) => error.fmt(formatter),
            Self::Model(error) => error.fmt(formatter),
            Self::InvalidConfiguration(field) => {
                write!(
                    formatter,
                    "invalid Qwen full-attention configuration: {field}"
                )
            }
            Self::CompletionPollLimit => {
                formatter.write_str("CUDA completion polling exceeded its bound")
            }
        }
    }
}

impl Error for Qwen35RealFullAttentionCudaError {}

impl From<CudaRuntimeError> for Qwen35RealFullAttentionCudaError {
    fn from(error: CudaRuntimeError) -> Self {
        Self::Runtime(error)
    }
}

impl From<Qwen35Error> for Qwen35RealFullAttentionCudaError {
    fn from(error: Qwen35Error) -> Self {
        Self::Model(error)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Qwen35RealFullAttentionCudaReport {
    pub schema: String,
    pub status: String,
    pub component_only: bool,
    pub source_commit: String,
    pub binary_path: String,
    pub binary_sha256: String,
    pub device: DeviceReport,
    pub config_sha256: String,
    pub gguf_path: String,
    pub gguf_file_sha256: Option<String>,
    pub artifact: ArtifactReport,
    pub token_ids: Vec<usize>,
    pub tensors: Vec<TensorReport>,
    pub geometry: GeometryReport,
    pub timing: TimingReport,
    pub completion: CompletionReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fused_q8_parity: Option<FusedQ8ParityReport>,
    pub numerical: Vec<NumericalReport>,
    pub resources: ResourceReport,
    pub claims: ClaimsReport,
}

#[derive(Clone, Debug, Serialize)]
pub struct DeviceReport {
    pub ordinal: i32,
    pub name: String,
    pub compute_capability: [u16; 2],
    pub total_global_memory_bytes: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ArtifactReport {
    pub manifest_tensor_count: u64,
    pub audited_tensor_count: usize,
    pub strict_profile_expected_tensor_count: u64,
    pub strict_profile_count_match: bool,
    pub strict_profile_qualified: bool,
    pub classification: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct TensorReport {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub ggml_type: u32,
    pub descriptor_payload_bytes: u64,
    pub payload_sha256: String,
    pub decoded_elements: Option<usize>,
}

#[derive(Clone, Debug, Serialize)]
pub struct GeometryReport {
    pub layer: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub query_values: usize,
    pub query_gate_values: usize,
    pub kv_values: usize,
    pub kv_capacity: usize,
    pub grouped_gqa_online_softmax_v2: bool,
    pub attention_workspace_bytes: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grouped_gqa_v2: Option<GroupedGqaV2GeometryReport>,
}

#[derive(Clone, Debug, Serialize)]
pub struct GroupedGqaV2GeometryReport {
    pub split_count: usize,
    pub stage1_threads: usize,
    pub tile_positions: usize,
    pub merge_segments: usize,
    pub merge_threads: usize,
    pub partial_elements: usize,
    pub workspace_bytes: usize,
    pub fused_q8_bytes: usize,
    pub current_row_kv_store_fused: bool,
    pub fused_q8_workspace_tail: bool,
    pub direct_gate_alias: bool,
}

#[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
pub(crate) fn grouped_gqa_v2_geometry(
    direct_gate_alias: bool,
) -> Option<GroupedGqaV2GeometryReport> {
    Some(GroupedGqaV2GeometryReport {
        split_count: QWEN35_FULL_ATTENTION_SPLITS_V2,
        stage1_threads: QWEN35_FULL_ATTENTION_STAGE1_THREADS_V2,
        tile_positions: QWEN35_FULL_ATTENTION_TILE_POSITIONS_V2,
        merge_segments: QWEN35_FULL_ATTENTION_MERGE_SEGMENTS_V2,
        merge_threads: QWEN35_FULL_ATTENTION_MERGE_THREADS_V2,
        partial_elements: QWEN35_FULL_ATTENTION_PARTIAL_ELEMENTS_V2,
        workspace_bytes: QWEN35_FULL_ATTENTION_WORKSPACE_ELEMENTS_V2 * core::mem::size_of::<f32>(),
        fused_q8_bytes: QWEN35_FULL_ATTENTION_FUSED_Q8_BYTES_V2,
        current_row_kv_store_fused: true,
        fused_q8_workspace_tail: true,
        direct_gate_alias,
    })
}

#[cfg(not(feature = "qwen35-grouped-gqa-online-softmax"))]
pub(crate) fn grouped_gqa_v2_geometry(
    _direct_gate_alias: bool,
) -> Option<GroupedGqaV2GeometryReport> {
    None
}

#[derive(Clone, Debug, Serialize)]
pub struct TimingReport {
    pub warmup_passes: usize,
    pub steady_passes: usize,
    pub steady_cuda_event_ms: f32,
    pub steady_batch_wall_ms: f64,
    pub per_pass_cuda_event_ms: f64,
    pub reference_ms: f64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct CompletionReport {
    pub poll_iterations: u64,
    pub nonblocking_poll_only: bool,
    pub owner_epoch_proof: bool,
    pub gpu_checkpoint_replay_exact: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct FusedQ8ParityReport {
    pub bytes_per_step: usize,
    pub audited_steps: usize,
    pub fused_sha256: Vec<String>,
    pub standalone_sha256: Vec<String>,
    pub all_steps_byte_exact: bool,
    pub checkpoint_replay_byte_exact: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct NumericalReport {
    pub name: String,
    pub elements: usize,
    pub finite_count: usize,
    pub max_abs_diff: f32,
    pub max_rel_diff: f32,
    pub absolute_tolerance: f32,
    pub relative_tolerance: f32,
    pub output_sha256: String,
    pub expected_sha256: String,
    pub pass: bool,
}

#[derive(Clone, Debug, Default, Serialize, PartialEq, Eq)]
pub struct BoundaryReport {
    pub host_waits: u64,
    pub stream_synchronizations: u64,
    pub event_synchronizations: u64,
    pub synchronous_h2d_copies: u64,
    pub synchronous_h2d_bytes: u64,
    pub synchronous_d2h_copies: u64,
    pub synchronous_d2h_bytes: u64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct AllocationReport {
    pub successful_allocations: u64,
    pub successful_frees: u64,
    pub live_allocations: u64,
    pub live_bytes: u64,
    pub peak_live_allocations: u64,
    pub peak_live_bytes: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ResourceReport {
    pub setup_boundaries: BoundaryReport,
    pub setup_allocations: AllocationReport,
    pub steady_boundaries: BoundaryReport,
    pub steady_allocations: AllocationReport,
    pub sequence_input_boundaries: BoundaryReport,
    pub completion_boundaries: BoundaryReport,
    pub readback_boundaries: BoundaryReport,
    pub cleanup_boundaries: BoundaryReport,
    pub total_boundaries: BoundaryReport,
    pub boundary_ledger_closed: bool,
    pub final_allocation_delta: AllocationReport,
}

#[derive(Clone, Debug, Serialize)]
pub struct ClaimsReport {
    pub real_weight_attention_executed: bool,
    pub real_q8_output_projection_executed: bool,
    pub two_token_checkpoint_replay: bool,
    pub complete_transformer_executed: bool,
    pub decode_executed: bool,
    pub semantic_parity_established: bool,
    pub product_tps_permitted: bool,
    pub default_cli_enabled: bool,
}

#[derive(Clone)]
pub(crate) struct ProjectedToken {
    pub(crate) query: Vec<f32>,
    pub(crate) gate: Vec<f32>,
    pub(crate) key: Vec<f32>,
    pub(crate) value: Vec<f32>,
}

#[derive(Clone)]
pub(crate) struct CpuTokenResult {
    pub(crate) attention: Vec<f32>,
    pub(crate) gated: Vec<f32>,
    pub(crate) output: Vec<f32>,
}

#[derive(Clone)]
pub(crate) struct DeviceReadback {
    pub(crate) attention: Vec<f32>,
    pub(crate) gated: Vec<f32>,
    pub(crate) output: Vec<f32>,
    pub(crate) key_cache: Vec<f32>,
    pub(crate) value_cache: Vec<f32>,
    #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
    pub(crate) fused_q8_1: Vec<u8>,
    #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
    pub(crate) standalone_q8_1: Vec<u8>,
    pub(crate) polls: u64,
    pub(crate) nonblocking: bool,
    pub(crate) proof: bool,
    pub(crate) completion_boundaries: BoundaryReport,
    pub(crate) readback_boundaries: BoundaryReport,
}

/// Execute the bounded selected full-attention component for two real tokens.
pub fn run_qwen35_real_full_attention_cuda(
    config: &Qwen35RealFullAttentionCudaConfig,
) -> Result<Qwen35RealFullAttentionCudaReport, Qwen35RealFullAttentionCudaError> {
    validate_config(config)?;
    let binary_path = std::env::current_exe().map_err(Qwen35Error::from)?;
    let binary_sha256 = sha256_hex(&std::fs::read(&binary_path).map_err(Qwen35Error::from)?);
    let config_bytes = std::fs::read(&config.config_path).map_err(Qwen35Error::from)?;
    let config_sha256 = sha256_hex(&config_bytes);
    let model_config = Qwen35Config::parse(&config_bytes)?;
    if model_config.hidden_size != HIDDEN
        || model_config.vocab_size != VOCAB
        || model_config.num_hidden_layers != 64
        || model_config.num_attention_heads != QUERY_HEADS
        || model_config.num_key_value_heads != KV_HEADS
        || model_config.head_dim != HEAD_DIM
        || model_config.rope_dimension != ROTARY_DIM
        || !model_config.full_attention_layers.contains(&config.layer)
    {
        return Err(Qwen35RealFullAttentionCudaError::InvalidConfiguration(
            "Qwen full-attention geometry",
        ));
    }

    let manifest = GgufManifest::inspect(&config.gguf_path, GgufAuditOptions::default())?;
    let descriptors = required_descriptors(&manifest, config.layer)?;
    let strict_profile_count_match = manifest.tensor_count == STRICT_PROFILE_TENSOR_COUNT;
    let artifact_classification = if strict_profile_count_match {
        "count_matches_strict_profile_but_full_model_qualification_not_run"
    } else {
        "extended_or_combined_manifest"
    };

    let mut reader = manifest.open_tensor_reader(&config.gguf_path)?;
    let second_token_id = config.token_id.checked_add(1).ok_or(
        Qwen35RealFullAttentionCudaError::InvalidConfiguration("token id"),
    )?;
    if second_token_id >= model_config.vocab_size {
        return Err(Qwen35RealFullAttentionCudaError::InvalidConfiguration(
            "token id leaves vocabulary",
        ));
    }
    let embedding = descriptors[0].clone();
    let hidden_a = read_embedding_row(&mut reader, &embedding, config.token_id)?;
    let hidden_b = read_embedding_row(&mut reader, &embedding, second_token_id)?;
    let mut payloads = Vec::with_capacity(descriptors.len());
    for descriptor in &descriptors {
        payloads.push(reader.read_tensor_bytes(&descriptor.name)?);
    }
    let tensor_reports = descriptors
        .iter()
        .zip(payloads.iter())
        .enumerate()
        .map(|(index, (descriptor, payload))| TensorReport {
            name: descriptor.name.clone(),
            dimensions: descriptor.dimensions.clone(),
            ggml_type: descriptor.ggml_type,
            descriptor_payload_bytes: descriptor.byte_len,
            payload_sha256: sha256_hex(payload),
            decoded_elements: if matches!(index, 0 | 1 | 5 | 6 | 8) {
                descriptor
                    .dimensions
                    .iter()
                    .try_fold(1usize, |count, dimension| {
                        count.checked_mul(*dimension as usize)
                    })
            } else {
                None
            },
        })
        .collect::<Vec<_>>();

    let attn_norm = dequantize_tensor(&descriptors[1], &payloads[1])?;
    let q_norm = dequantize_tensor(&descriptors[5], &payloads[5])?;
    let k_norm = dequantize_tensor(&descriptors[6], &payloads[6])?;
    let _post_attention_norm = dequantize_tensor(&descriptors[8], &payloads[8])?;
    let epsilon = model_config.rms_norm_eps as f32;
    let reference_started = Instant::now();
    let token_a = project_token(
        &hidden_a,
        &attn_norm,
        &descriptors[2],
        &payloads[2],
        &descriptors[3],
        &payloads[3],
        &descriptors[4],
        &payloads[4],
        &q_norm,
        &k_norm,
        0,
        &model_config,
        epsilon,
    )?;
    let token_b = project_token(
        &hidden_b,
        &attn_norm,
        &descriptors[2],
        &payloads[2],
        &descriptors[3],
        &payloads[3],
        &descriptors[4],
        &payloads[4],
        &q_norm,
        &k_norm,
        1,
        &model_config,
        epsilon,
    )?;
    let output_a = cpu_token_result(
        &token_a,
        &descriptors[7],
        &payloads[7],
        config.capacity,
        epsilon,
    )?;
    let mut cpu_state = Qwen35FullAttentionState::zeros(KV_HEADS, HEAD_DIM, config.capacity)?;
    let _ = cpu_state.step(&token_a.query, &token_a.key, &token_a.value, QUERY_HEADS)?;
    let checkpoint = cpu_state.checkpoint();
    let output_b = cpu_token_result_with_state(
        &mut cpu_state,
        &token_b,
        &descriptors[7],
        &payloads[7],
        epsilon,
    )?;
    let state_after_b = cpu_state.checkpoint();
    cpu_state.restore(&checkpoint)?;
    let replay_b = cpu_token_result_with_state(
        &mut cpu_state,
        &token_b,
        &descriptors[7],
        &payloads[7],
        epsilon,
    )?;
    let reference_ms = reference_started.elapsed().as_secs_f64() * 1_000.0;
    let checkpoint_replay_exact = output_b.attention == replay_b.attention
        && output_b.gated == replay_b.gated
        && output_b.output == replay_b.output
        && cpu_state.checkpoint() == state_after_b;

    let device = cuda_device_properties(config.device_ordinal)?;
    if device.compute_capability.major < 7 {
        return Err(Qwen35RealFullAttentionCudaError::InvalidConfiguration(
            "SM70 or newer device required",
        ));
    }
    let stream = OwnedCudaStream::create(config.device_ordinal)?;
    let allocation_start = cuda_device_allocation_snapshot();
    let boundary_start = cuda_host_boundary_snapshot();
    let query = OwnedCudaBuffer::allocate(config.device_ordinal, QUERY_VALUES * 4)?;
    let gate = OwnedCudaBuffer::allocate(config.device_ordinal, QUERY_VALUES * 4)?;
    let key = OwnedCudaBuffer::allocate(config.device_ordinal, KV_VALUES * 4)?;
    let value = OwnedCudaBuffer::allocate(config.device_ordinal, KV_VALUES * 4)?;
    let cache_bytes = config
        .capacity
        .checked_mul(KV_VALUES)
        .and_then(|elements| elements.checked_mul(4))
        .ok_or(Qwen35RealFullAttentionCudaError::InvalidConfiguration(
            "cache size",
        ))?;
    let mut key_cache = OwnedCudaBuffer::allocate(config.device_ordinal, cache_bytes)?;
    let mut value_cache = OwnedCudaBuffer::allocate(config.device_ordinal, cache_bytes)?;
    let attention = OwnedCudaBuffer::allocate(config.device_ordinal, QUERY_VALUES * 4)?;
    let gated = OwnedCudaBuffer::allocate(config.device_ordinal, QUERY_VALUES * 4)?;
    #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
    let attention_workspace = OwnedCudaBuffer::allocate(
        config.device_ordinal,
        QWEN35_FULL_ATTENTION_WORKSPACE_ELEMENTS_V2 * 4,
    )?;
    #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
    let standalone_gated_q8_1 = OwnedCudaBuffer::allocate(
        config.device_ordinal,
        QWEN35_FULL_ATTENTION_FUSED_Q8_BYTES_V2,
    )?;
    let output = OwnedCudaBuffer::allocate(config.device_ordinal, HIDDEN * 4)?;
    let output_weight = OwnedCudaBuffer::allocate(config.device_ordinal, payloads[7].len())?;
    #[cfg(not(feature = "qwen35-grouped-gqa-online-softmax"))]
    let attention_args_a = bind_attention(
        &stream,
        &query,
        &gate,
        &key,
        &value,
        &key_cache,
        &value_cache,
        &attention,
        &gated,
        config.capacity,
        0,
    )?;
    #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
    let attention_args_a = bind_attention_v2(
        &stream,
        &query,
        &gate,
        &key,
        &value,
        &key_cache,
        &value_cache,
        &attention,
        &gated,
        &attention_workspace,
        config.capacity,
        0,
    )?;
    #[cfg(not(feature = "qwen35-grouped-gqa-online-softmax"))]
    let attention_args_b = bind_attention(
        &stream,
        &query,
        &gate,
        &key,
        &value,
        &key_cache,
        &value_cache,
        &attention,
        &gated,
        config.capacity,
        1,
    )?;
    #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
    let attention_args_b = bind_attention_v2(
        &stream,
        &query,
        &gate,
        &key,
        &value,
        &key_cache,
        &value_cache,
        &attention,
        &gated,
        &attention_workspace,
        config.capacity,
        1,
    )?;
    let output_args = bind_output_projection(&stream, &output_weight, &gated, &output)?;
    #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
    let standalone_gated_quantize_args = CudaQ8_1ActivationQuantizeArgsV1::bind(
        &stream,
        CudaF32View::new(
            gated.view(0, gated.len_bytes())?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            QUERY_VALUES,
        )?,
        CudaQ8_1ActivationViewV1::new(
            standalone_gated_q8_1.view(0, standalone_gated_q8_1.len_bytes())?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_1_ACTIVATION_CUDA_V1,
            QUERY_VALUES,
        )?,
    )?;

    output_weight.copy_from_host_sync(&stream, &payloads[7])?;
    upload_token(&stream, &query, &gate, &key, &value, &token_a)?;
    key_cache.zero_sync(&stream)?;
    value_cache.zero_sync(&stream)?;
    for _ in 0..config.warmup {
        submit_pair(&stream, &attention_args_a, &output_args)?;
    }
    stream.synchronize()?;
    key_cache.zero_sync(&stream)?;
    value_cache.zero_sync(&stream)?;

    let setup_allocations = allocation_delta(allocation_start, cuda_device_allocation_snapshot())?;
    let setup_boundaries = boundary_delta(boundary_start, cuda_host_boundary_snapshot())?;

    let steady_alloc_start = cuda_device_allocation_snapshot();
    let steady_boundary_start = cuda_host_boundary_snapshot();
    let started = OwnedCudaEvent::create(config.device_ordinal)?;
    let stopped = OwnedCudaEvent::create(config.device_ordinal)?;
    let wall_start = Instant::now();
    started.record(&stream)?;
    for _ in 0..config.iterations {
        submit_pair(&stream, &attention_args_a, &output_args)?;
    }
    stopped.record(&stream)?;
    stopped.synchronize()?;
    let steady_batch_wall_ms = wall_start.elapsed().as_secs_f64() * 1_000.0;
    let steady_cuda_event_ms = started.elapsed_ms(&stopped)?;
    let steady_allocations =
        allocation_delta(steady_alloc_start, cuda_device_allocation_snapshot())?;
    let steady_boundaries = boundary_delta(steady_boundary_start, cuda_host_boundary_snapshot())?;

    let input_start = cuda_host_boundary_snapshot();
    key_cache.zero_sync(&stream)?;
    value_cache.zero_sync(&stream)?;
    upload_token(&stream, &query, &gate, &key, &value, &token_a)?;
    let mut sequence_input_boundaries = boundary_delta(input_start, cuda_host_boundary_snapshot())?;
    let audited_a = execute_and_readback(
        &stream,
        &attention_args_a,
        &output_args,
        &attention,
        &gated,
        &output,
        &key_cache,
        &value_cache,
        #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
        &attention_workspace,
        #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
        &standalone_gated_q8_1,
        #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
        &standalone_gated_quantize_args,
    )?;
    let input_start = cuda_host_boundary_snapshot();
    upload_token(&stream, &query, &gate, &key, &value, &token_b)?;
    sequence_input_boundaries = add_boundaries(
        sequence_input_boundaries,
        boundary_delta(input_start, cuda_host_boundary_snapshot())?,
    );
    let audited_b = execute_and_readback(
        &stream,
        &attention_args_b,
        &output_args,
        &attention,
        &gated,
        &output,
        &key_cache,
        &value_cache,
        #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
        &attention_workspace,
        #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
        &standalone_gated_q8_1,
        #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
        &standalone_gated_quantize_args,
    )?;
    let input_start = cuda_host_boundary_snapshot();
    key_cache.zero_sync(&stream)?;
    value_cache.zero_sync(&stream)?;
    upload_token(&stream, &query, &gate, &key, &value, &token_a)?;
    sequence_input_boundaries = add_boundaries(
        sequence_input_boundaries,
        boundary_delta(input_start, cuda_host_boundary_snapshot())?,
    );
    let replay_a = execute_and_readback(
        &stream,
        &attention_args_a,
        &output_args,
        &attention,
        &gated,
        &output,
        &key_cache,
        &value_cache,
        #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
        &attention_workspace,
        #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
        &standalone_gated_q8_1,
        #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
        &standalone_gated_quantize_args,
    )?;
    let mut gpu_checkpoint_replay_exact = equal_f32_bits(&audited_a.attention, &replay_a.attention)
        && equal_f32_bits(&audited_a.gated, &replay_a.gated)
        && equal_f32_bits(&audited_a.output, &replay_a.output)
        && equal_f32_bits(&audited_a.key_cache, &replay_a.key_cache)
        && equal_f32_bits(&audited_a.value_cache, &replay_a.value_cache);
    #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
    {
        gpu_checkpoint_replay_exact &= audited_a.fused_q8_1 == replay_a.fused_q8_1
            && audited_a.standalone_q8_1 == replay_a.standalone_q8_1;
    }

    #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
    let fused_q8_parity = {
        let fused = [
            &audited_a.fused_q8_1,
            &audited_b.fused_q8_1,
            &replay_a.fused_q8_1,
        ];
        let standalone = [
            &audited_a.standalone_q8_1,
            &audited_b.standalone_q8_1,
            &replay_a.standalone_q8_1,
        ];
        Some(FusedQ8ParityReport {
            bytes_per_step: QWEN35_FULL_ATTENTION_FUSED_Q8_BYTES_V2,
            audited_steps: fused.len(),
            fused_sha256: fused.iter().map(|bytes| sha256_hex(bytes)).collect(),
            standalone_sha256: standalone.iter().map(|bytes| sha256_hex(bytes)).collect(),
            all_steps_byte_exact: fused
                .iter()
                .zip(standalone.iter())
                .all(|(fused, standalone)| fused == standalone),
            checkpoint_replay_byte_exact: audited_a.fused_q8_1 == replay_a.fused_q8_1,
        })
    };
    #[cfg(not(feature = "qwen35-grouped-gqa-online-softmax"))]
    let fused_q8_parity: Option<FusedQ8ParityReport> = None;

    let numerical = vec![
        compare(
            "token_a.attention",
            &audited_a.attention,
            &output_a.attention,
            2e-3,
            4e-3,
        ),
        compare(
            "token_a.gated",
            &audited_a.gated,
            &output_a.gated,
            2e-3,
            4e-3,
        ),
        compare(
            "token_a.output",
            &audited_a.output,
            &output_a.output,
            5e-3,
            3e-3,
        ),
        compare(
            "token_b.attention",
            &audited_b.attention,
            &output_b.attention,
            2e-3,
            4e-3,
        ),
        compare(
            "token_b.gated",
            &audited_b.gated,
            &output_b.gated,
            2e-3,
            4e-3,
        ),
        compare(
            "token_b.output",
            &audited_b.output,
            &output_b.output,
            5e-3,
            3e-3,
        ),
        compare(
            "token_b.key_cache",
            &audited_b.key_cache,
            &expected_cache(&token_a, &token_b, config.capacity, true),
            2e-3,
            4e-3,
        ),
        compare(
            "token_b.value_cache",
            &audited_b.value_cache,
            &expected_cache(&token_a, &token_b, config.capacity, false),
            2e-3,
            4e-3,
        ),
    ];
    let all_numerical_pass = numerical.iter().all(|entry| entry.pass);
    let completion = CompletionReport {
        poll_iterations: audited_a
            .polls
            .saturating_add(audited_b.polls)
            .saturating_add(replay_a.polls),
        nonblocking_poll_only: audited_a.nonblocking
            && audited_b.nonblocking
            && replay_a.nonblocking,
        owner_epoch_proof: audited_a.proof && audited_b.proof && replay_a.proof,
        gpu_checkpoint_replay_exact,
    };
    let readback_boundaries = add_boundaries(
        add_boundaries(audited_a.readback_boundaries, audited_b.readback_boundaries),
        replay_a.readback_boundaries,
    );
    let completion_boundaries = add_boundaries(
        add_boundaries(
            audited_a.completion_boundaries,
            audited_b.completion_boundaries,
        ),
        replay_a.completion_boundaries,
    );
    let cleanup_start = cuda_host_boundary_snapshot();
    stream.synchronize()?;
    let before_cleanup = cuda_device_allocation_snapshot();
    drop(output_weight);
    drop(output);
    drop(gated);
    drop(attention);
    #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
    drop(standalone_gated_q8_1);
    #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
    drop(attention_workspace);
    drop(value_cache);
    drop(key_cache);
    drop(value);
    drop(key);
    drop(gate);
    drop(query);
    drop(stream);
    let cleanup_boundaries = boundary_delta(cleanup_start, cuda_host_boundary_snapshot())?;
    let after_cleanup = cuda_device_allocation_snapshot();
    let final_allocation_delta = allocation_delta(before_cleanup, after_cleanup)?;
    let total_boundaries = boundary_delta(boundary_start, cuda_host_boundary_snapshot())?;
    let accounted_boundaries = add_boundaries(
        add_boundaries(
            add_boundaries(
                add_boundaries(
                    add_boundaries(setup_boundaries.clone(), steady_boundaries.clone()),
                    sequence_input_boundaries.clone(),
                ),
                completion_boundaries.clone(),
            ),
            readback_boundaries.clone(),
        ),
        cleanup_boundaries.clone(),
    );
    let boundary_ledger_closed = accounted_boundaries == total_boundaries;
    let cleanup_zero =
        final_allocation_delta.live_allocations == 0 && final_allocation_delta.live_bytes == 0;
    let pass = all_numerical_pass
        && checkpoint_replay_exact
        && gpu_checkpoint_replay_exact
        && completion.nonblocking_poll_only
        && completion.owner_epoch_proof
        && fused_q8_parity.as_ref().map_or(true, |parity| {
            parity.all_steps_byte_exact && parity.checkpoint_replay_byte_exact
        })
        && boundary_ledger_closed
        && cleanup_zero;
    let report = Qwen35RealFullAttentionCudaReport {
        schema: QWEN35_REAL_FULL_ATTENTION_CUDA_SCHEMA.to_owned(),
        status: if pass {
            "PASS_COMPONENT_REAL_FULL_ATTENTION_CUDA".to_owned()
        } else {
            "FAIL_COMPONENT_REAL_FULL_ATTENTION_CUDA".to_owned()
        },
        component_only: true,
        source_commit: config.source_commit.clone(),
        binary_path: binary_path.display().to_string(),
        binary_sha256,
        device: DeviceReport {
            ordinal: device.ordinal,
            name: device.name,
            compute_capability: [
                device.compute_capability.major,
                device.compute_capability.minor,
            ],
            total_global_memory_bytes: device.total_global_memory_bytes,
        },
        config_sha256,
        gguf_path: config.gguf_path.display().to_string(),
        gguf_file_sha256: manifest.file_sha256.clone(),
        artifact: ArtifactReport {
            manifest_tensor_count: manifest.tensor_count,
            audited_tensor_count: descriptors.len(),
            strict_profile_expected_tensor_count: STRICT_PROFILE_TENSOR_COUNT,
            strict_profile_count_match,
            strict_profile_qualified: false,
            classification: artifact_classification.to_owned(),
        },
        token_ids: vec![config.token_id, second_token_id],
        tensors: tensor_reports,
        geometry: GeometryReport {
            layer: config.layer,
            query_heads: QUERY_HEADS,
            kv_heads: KV_HEADS,
            head_dim: HEAD_DIM,
            rotary_dim: ROTARY_DIM,
            query_values: QUERY_VALUES,
            query_gate_values: QUERY_GATE_VALUES,
            kv_values: KV_VALUES,
            kv_capacity: config.capacity,
            grouped_gqa_online_softmax_v2: cfg!(feature = "qwen35-grouped-gqa-online-softmax"),
            attention_workspace_bytes: GROUPED_GQA_WORKSPACE_BYTES,
            grouped_gqa_v2: grouped_gqa_v2_geometry(false),
        },
        timing: TimingReport {
            warmup_passes: config.warmup,
            steady_passes: config.iterations,
            steady_cuda_event_ms,
            steady_batch_wall_ms,
            per_pass_cuda_event_ms: f64::from(steady_cuda_event_ms)
                / config.iterations.max(1) as f64,
            reference_ms,
        },
        completion,
        fused_q8_parity,
        numerical,
        resources: ResourceReport {
            setup_boundaries,
            setup_allocations,
            steady_boundaries,
            steady_allocations,
            sequence_input_boundaries,
            completion_boundaries,
            readback_boundaries,
            cleanup_boundaries,
            total_boundaries,
            boundary_ledger_closed,
            final_allocation_delta,
        },
        claims: ClaimsReport {
            real_weight_attention_executed: true,
            real_q8_output_projection_executed: true,
            two_token_checkpoint_replay: checkpoint_replay_exact,
            complete_transformer_executed: false,
            decode_executed: false,
            semantic_parity_established: false,
            product_tps_permitted: false,
            default_cli_enabled: false,
        },
    };
    Ok(report)
}

fn validate_config(
    config: &Qwen35RealFullAttentionCudaConfig,
) -> Result<(), Qwen35RealFullAttentionCudaError> {
    if config.gguf_path.as_os_str().is_empty() || config.config_path.as_os_str().is_empty() {
        return Err(Qwen35RealFullAttentionCudaError::InvalidConfiguration(
            "gguf/config path",
        ));
    }
    if config.device_ordinal < 0 || config.capacity < 2 || config.capacity > 4096 {
        return Err(Qwen35RealFullAttentionCudaError::InvalidConfiguration(
            "device/capacity (two-token replay requires capacity >= 2)",
        ));
    }
    if config.iterations == 0 || config.warmup > 1000 {
        return Err(Qwen35RealFullAttentionCudaError::InvalidConfiguration(
            "warmup/iterations",
        ));
    }
    Ok(())
}

pub(crate) fn required_descriptors(
    manifest: &GgufManifest,
    layer: usize,
) -> Result<Vec<GgufTensor>, Qwen35RealFullAttentionCudaError> {
    let specs = vec![
        (
            EMBEDDING.to_owned(),
            vec![HIDDEN as u64, VOCAB as u64],
            GGML_TYPE_BF16,
        ),
        (
            layer_tensor_name(layer, ATTN_NORM),
            vec![HIDDEN as u64],
            GGML_TYPE_F32,
        ),
        (
            layer_tensor_name(layer, Q),
            vec![HIDDEN as u64, QUERY_GATE_VALUES as u64],
            GGML_TYPE_Q8_0,
        ),
        (
            layer_tensor_name(layer, K),
            vec![HIDDEN as u64, KV_VALUES as u64],
            GGML_TYPE_Q8_0,
        ),
        (
            layer_tensor_name(layer, V),
            vec![HIDDEN as u64, KV_VALUES as u64],
            GGML_TYPE_Q8_0,
        ),
        (
            layer_tensor_name(layer, Q_NORM),
            vec![HEAD_DIM as u64],
            GGML_TYPE_F32,
        ),
        (
            layer_tensor_name(layer, K_NORM),
            vec![HEAD_DIM as u64],
            GGML_TYPE_F32,
        ),
        (
            layer_tensor_name(layer, OUTPUT),
            vec![QUERY_VALUES as u64, HIDDEN as u64],
            GGML_TYPE_Q8_0,
        ),
        (
            layer_tensor_name(layer, POST_ATTN_NORM),
            vec![HIDDEN as u64],
            GGML_TYPE_F32,
        ),
    ];
    specs
        .into_iter()
        .map(|(name, dimensions, ggml_type)| {
            let descriptor = manifest.tensor(&name).ok_or(
                Qwen35RealFullAttentionCudaError::InvalidConfiguration("required tensor missing"),
            )?;
            if descriptor.dimensions != dimensions || descriptor.ggml_type != ggml_type {
                return Err(Qwen35RealFullAttentionCudaError::InvalidConfiguration(
                    "required tensor geometry/type",
                ));
            }
            Ok(descriptor.clone())
        })
        .collect()
}

pub(crate) fn layer_tensor_name(layer: usize, suffix: &str) -> String {
    format!("blk.{layer}.{suffix}")
}

pub(crate) fn reorder_interleaved_query_gate_q8_0_rows(
    descriptor: &GgufTensor,
    payload: &[u8],
    query_heads: usize,
    head_dim: usize,
) -> Result<Vec<u8>, Qwen35RealFullAttentionCudaError> {
    if descriptor.ggml_type != GGML_TYPE_Q8_0 || descriptor.dimensions.len() != 2 {
        return Err(Qwen35RealFullAttentionCudaError::InvalidConfiguration(
            "query/gate Q8_0 tensor type/rank",
        ));
    }
    let input_width = usize::try_from(descriptor.dimensions[0]).map_err(|_| {
        Qwen35RealFullAttentionCudaError::InvalidConfiguration("query/gate input width")
    })?;
    let output_rows = usize::try_from(descriptor.dimensions[1]).map_err(|_| {
        Qwen35RealFullAttentionCudaError::InvalidConfiguration("query/gate output rows")
    })?;
    let head_stride =
        head_dim
            .checked_mul(2)
            .ok_or(Qwen35RealFullAttentionCudaError::InvalidConfiguration(
                "query/gate head stride",
            ))?;
    let expected_rows = query_heads.checked_mul(head_stride).ok_or(
        Qwen35RealFullAttentionCudaError::InvalidConfiguration("query/gate row extent"),
    )?;
    if query_heads == 0
        || head_dim == 0
        || output_rows != expected_rows
        || !input_width.is_multiple_of(Q8_0_BLOCK_ELEMENTS)
    {
        return Err(Qwen35RealFullAttentionCudaError::InvalidConfiguration(
            "query/gate row geometry",
        ));
    }
    let row_bytes = input_width
        .checked_div(Q8_0_BLOCK_ELEMENTS)
        .and_then(|blocks| blocks.checked_mul(Q8_0_BLOCK_BYTES))
        .ok_or(Qwen35RealFullAttentionCudaError::InvalidConfiguration(
            "query/gate row bytes",
        ))?;
    let expected_bytes = output_rows.checked_mul(row_bytes).ok_or(
        Qwen35RealFullAttentionCudaError::InvalidConfiguration("query/gate payload extent"),
    )?;
    if payload.len() != expected_bytes || descriptor.byte_len != expected_bytes as u64 {
        return Err(Qwen35RealFullAttentionCudaError::InvalidConfiguration(
            "query/gate payload bytes",
        ));
    }

    // The model stores [Q_head, gate_head] pairs. Reordering weight rows once
    // at plan construction preserves a contiguous per-token CUDA projection
    // result without adding 48 device copies to every full-attention layer.
    let mut reordered = Vec::with_capacity(payload.len());
    for half in 0..2 {
        for head in 0..query_heads {
            let first_row = (head * 2 + half) * head_dim;
            let first_byte = first_row * row_bytes;
            let end_byte = first_byte + head_dim * row_bytes;
            reordered.extend_from_slice(&payload[first_byte..end_byte]);
        }
    }
    Ok(reordered)
}

pub(crate) fn read_embedding_row(
    reader: &mut augur_model_qwen35::GgufTensorReader<'_>,
    descriptor: &GgufTensor,
    token_id: usize,
) -> Result<Vec<f32>, Qwen35RealFullAttentionCudaError> {
    if token_id >= VOCAB {
        return Err(Qwen35RealFullAttentionCudaError::InvalidConfiguration(
            "embedding token id",
        ));
    }
    let row_bytes = HIDDEN * 2;
    let offset = token_id.checked_mul(row_bytes).ok_or(
        Qwen35RealFullAttentionCudaError::InvalidConfiguration("embedding offset"),
    )?;
    let bytes = reader.read_tensor_slice(&descriptor.name, offset as u64, row_bytes as u64)?;
    let mut values = vec![0.0; HIDDEN];
    dequantize_bf16(&bytes, &mut values)?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(Qwen35RealFullAttentionCudaError::InvalidConfiguration(
            "embedding finite",
        ));
    }
    Ok(values)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn project_token(
    hidden: &[f32],
    attn_norm: &[f32],
    q_descriptor: &GgufTensor,
    q_payload: &[u8],
    k_descriptor: &GgufTensor,
    k_payload: &[u8],
    v_descriptor: &GgufTensor,
    v_payload: &[u8],
    q_norm: &[f32],
    k_norm: &[f32],
    position: usize,
    model_config: &Qwen35Config,
    epsilon: f32,
) -> Result<ProjectedToken, Qwen35RealFullAttentionCudaError> {
    let normalized = rms_norm_last_dim(
        &ReferenceTensor::new(vec![1, HIDDEN], hidden.to_vec())?,
        &ReferenceTensor::new(vec![HIDDEN], attn_norm.to_vec())?,
        epsilon,
    )?;
    let q_full = quantized_matvec_fused_simd(q_descriptor, q_payload, normalized.values())?;
    let k_raw = quantized_matvec_fused_simd(k_descriptor, k_payload, normalized.values())?;
    let v_raw = quantized_matvec_fused_simd(v_descriptor, v_payload, normalized.values())?;
    if q_full.len() != QUERY_GATE_VALUES || k_raw.len() != KV_VALUES || v_raw.len() != KV_VALUES {
        return Err(Qwen35RealFullAttentionCudaError::InvalidConfiguration(
            "projection output geometry",
        ));
    }
    let (q_raw, gate) = split_interleaved_query_gate(&q_full, QUERY_HEADS, HEAD_DIM)?;
    let q = rms_norm_last_dim(
        &ReferenceTensor::new(vec![QUERY_HEADS, HEAD_DIM], q_raw)?,
        &ReferenceTensor::new(vec![HEAD_DIM], q_norm.to_vec())?,
        epsilon,
    )?;
    let k = rms_norm_last_dim(
        &ReferenceTensor::new(vec![KV_HEADS, HEAD_DIM], k_raw)?,
        &ReferenceTensor::new(vec![HEAD_DIM], k_norm.to_vec())?,
        epsilon,
    )?;
    let q = target_rope_last_dim(&q, position, ROTARY_DIM, model_config.rope_theta as f32)?;
    let k = target_rope_last_dim(&k, position, ROTARY_DIM, model_config.rope_theta as f32)?;
    if q.values()
        .iter()
        .chain(k.values())
        .chain(v_raw.iter())
        .chain(gate.iter())
        .any(|value| !value.is_finite())
    {
        return Err(Qwen35RealFullAttentionCudaError::InvalidConfiguration(
            "projection finite values",
        ));
    }
    Ok(ProjectedToken {
        query: q.into_values(),
        gate,
        key: k.into_values(),
        value: v_raw,
    })
}

fn stable_sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let e = value.exp();
        e / (1.0 + e)
    }
}

pub(crate) fn cpu_token_result(
    token: &ProjectedToken,
    output_descriptor: &GgufTensor,
    output_payload: &[u8],
    capacity: usize,
    _epsilon: f32,
) -> Result<CpuTokenResult, Qwen35RealFullAttentionCudaError> {
    let mut state = Qwen35FullAttentionState::zeros(KV_HEADS, HEAD_DIM, capacity)?;
    cpu_token_result_with_state(&mut state, token, output_descriptor, output_payload, 1.0e-6)
}

pub(crate) fn cpu_token_result_with_state(
    state: &mut Qwen35FullAttentionState,
    token: &ProjectedToken,
    output_descriptor: &GgufTensor,
    output_payload: &[u8],
    _epsilon: f32,
) -> Result<CpuTokenResult, Qwen35RealFullAttentionCudaError> {
    let attention = state.step(&token.query, &token.key, &token.value, QUERY_HEADS)?;
    let gated = attention
        .iter()
        .zip(&token.gate)
        .map(|(value, gate)| value * stable_sigmoid(*gate))
        .collect::<Vec<_>>();
    let output = quantized_matvec_fused_simd(output_descriptor, output_payload, &gated)?;
    Ok(CpuTokenResult {
        attention,
        gated,
        output,
    })
}

fn expected_cache(
    token_a: &ProjectedToken,
    token_b: &ProjectedToken,
    capacity: usize,
    keys: bool,
) -> Vec<f32> {
    let mut result = vec![0.0; capacity * KV_VALUES];
    result[..KV_VALUES].copy_from_slice(if keys { &token_a.key } else { &token_a.value });
    if capacity > 1 {
        result[KV_VALUES..2 * KV_VALUES].copy_from_slice(if keys {
            &token_b.key
        } else {
            &token_b.value
        });
    }
    result
}

pub(crate) fn bind_attention(
    stream: &OwnedCudaStream,
    query: &OwnedCudaBuffer,
    gate: &OwnedCudaBuffer,
    key: &OwnedCudaBuffer,
    value: &OwnedCudaBuffer,
    key_cache: &OwnedCudaBuffer,
    value_cache: &OwnedCudaBuffer,
    attention: &OwnedCudaBuffer,
    gated: &OwnedCudaBuffer,
    capacity: usize,
    position: usize,
) -> Result<CudaQwen35FullAttentionArgsV1, Qwen35RealFullAttentionCudaError> {
    let view = |buffer: &OwnedCudaBuffer| -> Result<CudaF32View, CudaRuntimeError> {
        CudaF32View::new(
            buffer.view(0, buffer.len_bytes())?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            buffer.len_bytes() / 4,
        )
    };
    Ok(CudaQwen35FullAttentionArgsV1::bind(
        stream,
        CudaQwen35FullAttentionViewsV1 {
            query: view(query)?,
            gate: view(gate)?,
            key: view(key)?,
            value: view(value)?,
            key_cache: view(key_cache)?,
            value_cache: view(value_cache)?,
            attention: view(attention)?,
            gated: view(gated)?,
        },
        capacity,
        position,
    )?)
}

#[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn bind_attention_v2(
    stream: &OwnedCudaStream,
    query: &OwnedCudaBuffer,
    gate: &OwnedCudaBuffer,
    key: &OwnedCudaBuffer,
    value: &OwnedCudaBuffer,
    key_cache: &OwnedCudaBuffer,
    value_cache: &OwnedCudaBuffer,
    attention: &OwnedCudaBuffer,
    gated: &OwnedCudaBuffer,
    workspace: &OwnedCudaBuffer,
    capacity: usize,
    position: usize,
) -> Result<CudaQwen35FullAttentionArgsV2, Qwen35RealFullAttentionCudaError> {
    let gate_view = CudaF32View::new(
        gate.view(0, gate.len_bytes())?,
        PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
        gate.len_bytes() / 4,
    )?;
    bind_attention_v2_with_gate_view(
        stream,
        query,
        gate_view,
        key,
        value,
        key_cache,
        value_cache,
        attention,
        gated,
        workspace,
        capacity,
        position,
    )
}

#[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn bind_attention_v2_with_gate_view(
    stream: &OwnedCudaStream,
    query: &OwnedCudaBuffer,
    gate: CudaF32View,
    key: &OwnedCudaBuffer,
    value: &OwnedCudaBuffer,
    key_cache: &OwnedCudaBuffer,
    value_cache: &OwnedCudaBuffer,
    attention: &OwnedCudaBuffer,
    gated: &OwnedCudaBuffer,
    workspace: &OwnedCudaBuffer,
    capacity: usize,
    position: usize,
) -> Result<CudaQwen35FullAttentionArgsV2, Qwen35RealFullAttentionCudaError> {
    let view = |buffer: &OwnedCudaBuffer| -> Result<CudaF32View, CudaRuntimeError> {
        CudaF32View::new(
            buffer.view(0, buffer.len_bytes())?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            buffer.len_bytes() / 4,
        )
    };
    if workspace.len_bytes() != QWEN35_FULL_ATTENTION_WORKSPACE_ELEMENTS_V2 * 4 {
        return Err(Qwen35RealFullAttentionCudaError::InvalidConfiguration(
            "grouped-GQA workspace extent",
        ));
    }
    Ok(CudaQwen35FullAttentionArgsV2::bind(
        stream,
        CudaQwen35FullAttentionViewsV2 {
            query: view(query)?,
            gate,
            key: view(key)?,
            value: view(value)?,
            key_cache: view(key_cache)?,
            value_cache: view(value_cache)?,
            attention: view(attention)?,
            gated: view(gated)?,
            workspace: view(workspace)?,
        },
        capacity,
        position,
    )?)
}

#[cfg(feature = "qwen35-kv8-full-attention")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn bind_attention_v3_with_gate_view(
    stream: &OwnedCudaStream,
    query: &OwnedCudaBuffer,
    gate: CudaF32View,
    key: &OwnedCudaBuffer,
    value: &OwnedCudaBuffer,
    key_cache: &OwnedCudaBuffer,
    value_cache: &OwnedCudaBuffer,
    attention: &OwnedCudaBuffer,
    gated: &OwnedCudaBuffer,
    workspace: &OwnedCudaBuffer,
    capacity: usize,
    position: usize,
) -> Result<CudaQwen35Kv8FullAttentionArgsV3, Qwen35RealFullAttentionCudaError> {
    let view = |buffer: &OwnedCudaBuffer| -> Result<CudaF32View, CudaRuntimeError> {
        CudaF32View::new(
            buffer.view(0, buffer.len_bytes())?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            buffer.len_bytes() / core::mem::size_of::<f32>(),
        )
    };
    let cache_bytes = capacity.checked_mul(QWEN35_KV8_CACHE_ROW_BYTES).ok_or(
        Qwen35RealFullAttentionCudaError::InvalidConfiguration("KV8 cache extent overflow"),
    )?;
    if key_cache.len_bytes() != cache_bytes
        || value_cache.len_bytes() != cache_bytes
        || workspace.len_bytes() != QWEN35_FULL_ATTENTION_WORKSPACE_ELEMENTS_V2 * 4
    {
        return Err(Qwen35RealFullAttentionCudaError::InvalidConfiguration(
            "KV8 grouped-GQA workspace or cache extent",
        ));
    }
    Ok(CudaQwen35Kv8FullAttentionArgsV3::bind(
        stream,
        CudaQwen35Kv8FullAttentionViewsV3 {
            query: view(query)?,
            gate,
            key: view(key)?,
            value: view(value)?,
            key_cache: key_cache.view(0, key_cache.len_bytes())?,
            value_cache: value_cache.view(0, value_cache.len_bytes())?,
            attention: view(attention)?,
            gated: view(gated)?,
            workspace: view(workspace)?,
        },
        capacity,
        position,
    )?)
}

pub(crate) fn bind_output_projection(
    stream: &OwnedCudaStream,
    weight: &OwnedCudaBuffer,
    input: &OwnedCudaBuffer,
    output: &OwnedCudaBuffer,
) -> Result<CudaQ8_0F32MatvecArgsV1, Qwen35RealFullAttentionCudaError> {
    Ok(CudaQ8_0F32MatvecArgsV1::bind(
        stream,
        CudaQ8_0WeightViewV1::new(
            weight.view(0, weight.len_bytes())?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_0_CUDA_V1,
            HIDDEN,
            QUERY_VALUES,
        )?,
        CudaF32View::new(
            input.view(0, input.len_bytes())?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            QUERY_VALUES,
        )?,
        CudaF32View::new(
            output.view(0, output.len_bytes())?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            HIDDEN,
        )?,
    )?)
}

pub(crate) fn upload_token(
    stream: &OwnedCudaStream,
    query: &OwnedCudaBuffer,
    gate: &OwnedCudaBuffer,
    key: &OwnedCudaBuffer,
    value: &OwnedCudaBuffer,
    token: &ProjectedToken,
) -> Result<(), Qwen35RealFullAttentionCudaError> {
    query.copy_from_host_sync(stream, &f32_bytes(&token.query))?;
    gate.copy_from_host_sync(stream, &f32_bytes(&token.gate))?;
    key.copy_from_host_sync(stream, &f32_bytes(&token.key))?;
    value.copy_from_host_sync(stream, &f32_bytes(&token.value))?;
    Ok(())
}

pub(crate) trait FullAttentionSubmit {
    fn submit_attention(
        &self,
        stream: &OwnedCudaStream,
    ) -> Result<(), Qwen35RealFullAttentionCudaError>;
}

impl FullAttentionSubmit for CudaQwen35FullAttentionArgsV1 {
    fn submit_attention(
        &self,
        stream: &OwnedCudaStream,
    ) -> Result<(), Qwen35RealFullAttentionCudaError> {
        self.submit_for_plan(stream)?;
        Ok(())
    }
}

#[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
impl FullAttentionSubmit for CudaQwen35FullAttentionArgsV2 {
    fn submit_attention(
        &self,
        stream: &OwnedCudaStream,
    ) -> Result<(), Qwen35RealFullAttentionCudaError> {
        self.submit_for_plan(stream)?;
        Ok(())
    }
}

pub(crate) fn submit_pair<A: FullAttentionSubmit>(
    stream: &OwnedCudaStream,
    attention: &A,
    output: &CudaQ8_0F32MatvecArgsV1,
) -> Result<(), Qwen35RealFullAttentionCudaError> {
    attention.submit_attention(stream)?;
    output.submit_for_plan(stream)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_and_readback<A: FullAttentionSubmit>(
    stream: &OwnedCudaStream,
    attention_args: &A,
    output_args: &CudaQ8_0F32MatvecArgsV1,
    attention: &OwnedCudaBuffer,
    gated: &OwnedCudaBuffer,
    output: &OwnedCudaBuffer,
    key_cache: &OwnedCudaBuffer,
    value_cache: &OwnedCudaBuffer,
    #[cfg(feature = "qwen35-grouped-gqa-online-softmax")] attention_workspace: &OwnedCudaBuffer,
    #[cfg(feature = "qwen35-grouped-gqa-online-softmax")] standalone_gated_q8_1: &OwnedCudaBuffer,
    #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
    standalone_gated_quantize_args: &CudaQ8_1ActivationQuantizeArgsV1,
) -> Result<DeviceReadback, Qwen35RealFullAttentionCudaError> {
    let completion_before = cuda_host_boundary_snapshot();
    submit_pair(stream, attention_args, output_args)?;
    #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
    standalone_gated_quantize_args.submit_for_plan(stream)?;
    let receipt = stream.submission_receipt()?;
    let event = OwnedCudaEvent::create(stream.device_ordinal())?;
    event.record_for_submission(stream, receipt)?;
    let mut polls = 0_u64;
    let proof = loop {
        if let Some(proof) = stream.query_event_completion(&event, receipt)? {
            break proof;
        }
        polls = polls.saturating_add(1);
        if polls >= MAX_COMPLETION_POLLS {
            return Err(Qwen35RealFullAttentionCudaError::CompletionPollLimit);
        }
        spin_loop();
    };
    let proof_ok = proof.covers_submission(receipt) && proof.belongs_to_stream(stream)?;
    if !proof_ok {
        return Err(Qwen35RealFullAttentionCudaError::InvalidConfiguration(
            "owner epoch proof",
        ));
    }
    let completion_after = cuda_host_boundary_snapshot();
    let completion_boundaries = boundary_delta(completion_before, completion_after)?;
    let readback_before = cuda_host_boundary_snapshot();
    let mut attention_bytes = vec![0_u8; QUERY_VALUES * 4];
    let mut gated_bytes = vec![0_u8; QUERY_VALUES * 4];
    let mut output_bytes = vec![0_u8; HIDDEN * 4];
    let mut key_cache_bytes = vec![0_u8; key_cache.len_bytes()];
    let mut value_cache_bytes = vec![0_u8; value_cache.len_bytes()];
    #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
    let mut fused_q8_1 = vec![0_u8; QWEN35_FULL_ATTENTION_FUSED_Q8_BYTES_V2];
    #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
    let mut standalone_q8_1 = vec![0_u8; QWEN35_FULL_ATTENTION_FUSED_Q8_BYTES_V2];
    attention.copy_to_host_sync_with_submission_proof(stream, &mut attention_bytes, receipt)?;
    gated.copy_to_host_sync_with_submission_proof(stream, &mut gated_bytes, receipt)?;
    output.copy_to_host_sync_with_submission_proof(stream, &mut output_bytes, receipt)?;
    key_cache.copy_to_host_sync_with_submission_proof(stream, &mut key_cache_bytes, receipt)?;
    value_cache.copy_to_host_sync_with_submission_proof(stream, &mut value_cache_bytes, receipt)?;
    #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
    attention_workspace
        .view(
            QWEN35_FULL_ATTENTION_PARTIAL_ELEMENTS_V2 * core::mem::size_of::<f32>(),
            QWEN35_FULL_ATTENTION_FUSED_Q8_BYTES_V2,
        )?
        .copy_to_host_sync_with_submission_proof(stream, &mut fused_q8_1, receipt)?;
    #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
    standalone_gated_q8_1.copy_to_host_sync_with_submission_proof(
        stream,
        &mut standalone_q8_1,
        receipt,
    )?;
    let readback_after = cuda_host_boundary_snapshot();
    Ok(DeviceReadback {
        attention: f32_vec(&attention_bytes),
        gated: f32_vec(&gated_bytes),
        output: f32_vec(&output_bytes),
        key_cache: f32_vec(&key_cache_bytes),
        value_cache: f32_vec(&value_cache_bytes),
        #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
        fused_q8_1,
        #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
        standalone_q8_1,
        polls,
        nonblocking: true,
        proof: proof_ok,
        completion_boundaries,
        readback_boundaries: boundary_delta(readback_before, readback_after)?,
    })
}

fn compare(
    name: &str,
    actual: &[f32],
    expected: &[f32],
    absolute_tolerance: f32,
    relative_tolerance: f32,
) -> NumericalReport {
    let mut max_abs_diff = 0.0_f32;
    let mut max_rel_diff = 0.0_f32;
    let finite_count = actual.iter().filter(|value| value.is_finite()).count();
    for (actual, expected) in actual.iter().zip(expected) {
        let abs = (*actual - *expected).abs();
        max_abs_diff = max_abs_diff.max(abs);
        max_rel_diff = max_rel_diff.max(abs / expected.abs().max(1.0e-6));
    }
    NumericalReport {
        name: name.to_owned(),
        elements: actual.len(),
        finite_count,
        max_abs_diff,
        max_rel_diff,
        absolute_tolerance,
        relative_tolerance,
        output_sha256: sha256_f32_bits(actual),
        expected_sha256: sha256_f32_bits(expected),
        pass: actual.len() == expected.len()
            && finite_count == actual.len()
            && actual.iter().zip(expected).all(|(actual, expected)| {
                (*actual - *expected).abs()
                    <= absolute_tolerance + relative_tolerance * expected.abs()
            }),
    }
}

fn equal_f32_bits(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

fn allocation_delta(
    earlier: augur_backend::CudaDeviceAllocationSnapshot,
    later: augur_backend::CudaDeviceAllocationSnapshot,
) -> Result<AllocationReport, Qwen35RealFullAttentionCudaError> {
    Ok(AllocationReport {
        successful_allocations: later
            .successful_allocations
            .checked_sub(earlier.successful_allocations)
            .ok_or(Qwen35RealFullAttentionCudaError::InvalidConfiguration(
                "allocation counters",
            ))?,
        successful_frees: later
            .successful_frees
            .checked_sub(earlier.successful_frees)
            .ok_or(Qwen35RealFullAttentionCudaError::InvalidConfiguration(
                "allocation counters",
            ))?,
        live_allocations: later.live_allocations,
        live_bytes: later.live_bytes,
        peak_live_allocations: later.peak_live_allocations,
        peak_live_bytes: later.peak_live_bytes,
    })
}

fn boundary_delta(
    earlier: augur_backend::CudaHostBoundarySnapshot,
    later: augur_backend::CudaHostBoundarySnapshot,
) -> Result<BoundaryReport, Qwen35RealFullAttentionCudaError> {
    let delta = later.checked_delta_since(earlier).ok_or(
        Qwen35RealFullAttentionCudaError::InvalidConfiguration("host counters"),
    )?;
    Ok(BoundaryReport {
        host_waits: delta.host_waits,
        stream_synchronizations: delta.stream_synchronizations,
        event_synchronizations: delta.event_synchronizations,
        synchronous_h2d_copies: delta.synchronous_h2d_copies,
        synchronous_h2d_bytes: delta.synchronous_h2d_bytes,
        synchronous_d2h_copies: delta.synchronous_d2h_copies,
        synchronous_d2h_bytes: delta.synchronous_d2h_bytes,
    })
}

fn add_boundaries(left: BoundaryReport, right: BoundaryReport) -> BoundaryReport {
    BoundaryReport {
        host_waits: left.host_waits.saturating_add(right.host_waits),
        stream_synchronizations: left
            .stream_synchronizations
            .saturating_add(right.stream_synchronizations),
        event_synchronizations: left
            .event_synchronizations
            .saturating_add(right.event_synchronizations),
        synchronous_h2d_copies: left
            .synchronous_h2d_copies
            .saturating_add(right.synchronous_h2d_copies),
        synchronous_h2d_bytes: left
            .synchronous_h2d_bytes
            .saturating_add(right.synchronous_h2d_bytes),
        synchronous_d2h_copies: left
            .synchronous_d2h_copies
            .saturating_add(right.synchronous_d2h_copies),
        synchronous_d2h_bytes: left
            .synchronous_d2h_bytes
            .saturating_add(right.synchronous_d2h_bytes),
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect()
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

fn sha256_f32_bits(values: &[f32]) -> String {
    sha256_hex(&f32_bytes(values))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_tensor_names_are_scoped_without_rewriting_suffixes() {
        assert_eq!(layer_tensor_name(3, ATTN_NORM), "blk.3.attn_norm.weight");
        assert_eq!(layer_tensor_name(7, OUTPUT), "blk.7.attn_output.weight");
        assert_eq!(layer_tensor_name(63, Q_NORM), "blk.63.attn_q_norm.weight");
    }

    #[test]
    fn default_full_attention_component_targets_layer_three() {
        assert_eq!(Qwen35RealFullAttentionCudaConfig::default().layer, 3);
    }

    #[test]
    fn q8_query_gate_rows_are_reordered_once_for_contiguous_cuda_views() {
        let row_bytes = Q8_0_BLOCK_BYTES;
        let descriptor = GgufTensor {
            name: "test.q_gate".into(),
            dimensions: vec![Q8_0_BLOCK_ELEMENTS as u64, 8],
            ggml_type: GGML_TYPE_Q8_0,
            relative_offset: 0,
            absolute_offset: 0,
            byte_len: (row_bytes * 8) as u64,
        };
        let mut payload = vec![0_u8; row_bytes * 8];
        for row in 0..8 {
            payload[row * row_bytes..(row + 1) * row_bytes].fill(row as u8);
        }

        let reordered =
            reorder_interleaved_query_gate_q8_0_rows(&descriptor, &payload, 2, 2).unwrap();
        let row_ids = reordered
            .chunks_exact(row_bytes)
            .map(|row| row[0])
            .collect::<Vec<_>>();
        assert_eq!(row_ids, vec![0, 1, 4, 5, 2, 3, 6, 7]);

        let mut truncated = payload;
        truncated.pop();
        assert!(reorder_interleaved_query_gate_q8_0_rows(&descriptor, &truncated, 2, 2).is_err());
    }
}
