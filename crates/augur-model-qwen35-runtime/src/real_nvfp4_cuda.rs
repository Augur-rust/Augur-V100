#![cfg(feature = "cuda")]

//! Real-weight GGML NVFP4 projection evidence for the Qwen lane.
//!
//! This runner uploads the layer-0 `ffn_gate`, `ffn_up`, and `ffn_down`
//! matrices without changing their 36-byte GGML block representation.  The
//! CPU fused implementation is the numerical oracle.  The runner executes a
//! GPU `SiLU(gate) * up` stage between the three projections and therefore
//! remains a component diagnostic; it is not a complete transformer, decode
//! loop, semantic, or product path.

use std::{error::Error, fmt, hint::spin_loop, path::PathBuf, time::Instant};

use augur_backend::{
    cuda_device_allocation_snapshot, cuda_device_properties, cuda_host_boundary_snapshot,
    CudaF32View, CudaNvfp4Sm70F32MatvecArgsV1, CudaNvfp4Sm70Q8_1MatvecArgsV1,
    CudaNvfp4Sm70WeightViewV1, CudaQ8_1ActivationQuantizeArgsV1, CudaQ8_1ActivationViewV1,
    CudaRuntimeError, CudaSiluMulF32ArgsV1, OwnedCudaBuffer, OwnedCudaEvent, OwnedCudaStream,
    Q8_1_ACTIVATION_BLOCK_BYTES, Q8_1_ACTIVATION_BLOCK_ELEMENTS,
};
use augur_core::PhysicalLayoutId;
use augur_model_qwen35::{
    dequantize_bf16, dequantize_tensor, quantize_q8_0_activation, quantized_matvec_activation_q8_0,
    quantized_matvec_fused_simd, rms_norm_last_dim, GgufAuditOptions, GgufManifest, GgufTensor,
    Qwen35Config, Qwen35Error, ReferenceTensor, GGML_TYPE_BF16, GGML_TYPE_F32, GGML_TYPE_NVFP4,
    NVFP4_BLOCK_BYTES, NVFP4_BLOCK_ELEMENTS,
};
use serde::Serialize;
use sha2::{Digest, Sha256};

pub const QWEN35_REAL_NVFP4_CUDA_SCHEMA: &str =
    "crystal-quant.qwen35.real-nvfp4-cuda-silu-chain.v2";

const HIDDEN: usize = 5_120;
const INTERMEDIATE: usize = 17_408;
const VOCAB: usize = 248_320;
const LAYER: usize = 0;
const EMBEDDING: &str = "token_embd.weight";
const ATTN_NORM: &str = "blk.0.attn_norm.weight";
const FFN_GATE: &str = "blk.0.ffn_gate.weight";
const FFN_UP: &str = "blk.0.ffn_up.weight";
const FFN_DOWN: &str = "blk.0.ffn_down.weight";
const FFN_GATE_SCALE: &str = "blk.0.ffn_gate.scale";
const FFN_UP_SCALE: &str = "blk.0.ffn_up.scale";
const FFN_DOWN_SCALE: &str = "blk.0.ffn_down.scale";
const FFN_SILU_MUL: &str = "ffn_silu_mul";
const MAX_COMPLETION_POLLS: u64 = 2_000_000;

/// Configuration for the real layer-0 NVFP4 projection boundary.
#[derive(Clone, Debug)]
pub struct Qwen35RealNvfp4CudaConfig {
    pub gguf_path: PathBuf,
    pub config_path: PathBuf,
    pub token_id: usize,
    pub device_ordinal: i32,
    pub warmup: usize,
    pub iterations: usize,
    pub source_commit: String,
}

impl Default for Qwen35RealNvfp4CudaConfig {
    fn default() -> Self {
        Self {
            gguf_path: PathBuf::new(),
            config_path: PathBuf::new(),
            token_id: 1,
            device_ordinal: 0,
            warmup: 2,
            iterations: 20,
            source_commit: "UNBOUND_COMPONENT_SOURCE".to_owned(),
        }
    }
}

#[derive(Debug)]
pub enum Qwen35RealNvfp4CudaError {
    Runtime(CudaRuntimeError),
    Model(Qwen35Error),
    InvalidConfiguration(&'static str),
    CompletionPollLimit,
}

impl fmt::Display for Qwen35RealNvfp4CudaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Runtime(error) => error.fmt(formatter),
            Self::Model(error) => error.fmt(formatter),
            Self::InvalidConfiguration(field) => {
                write!(
                    formatter,
                    "invalid Qwen real NVFP4 CUDA configuration: {field}"
                )
            }
            Self::CompletionPollLimit => {
                formatter.write_str("CUDA completion polling exceeded its bound")
            }
        }
    }
}

impl Error for Qwen35RealNvfp4CudaError {}

impl From<CudaRuntimeError> for Qwen35RealNvfp4CudaError {
    fn from(error: CudaRuntimeError) -> Self {
        Self::Runtime(error)
    }
}

impl From<Qwen35Error> for Qwen35RealNvfp4CudaError {
    fn from(error: Qwen35Error) -> Self {
        Self::Model(error)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Qwen35RealNvfp4CudaReport {
    pub schema: String,
    pub status: String,
    pub component_only: bool,
    pub source_commit: String,
    pub config_path: String,
    pub config_sha256: String,
    pub device: Nvfp4CudaDeviceReport,
    pub source: Nvfp4CudaSourceReport,
    pub input: Nvfp4CudaInputReport,
    pub shape: Nvfp4CudaShapeReport,
    pub timing: Nvfp4CudaTimingReport,
    pub q8_1_dp4a: Nvfp4CudaQ8_1Dp4aReport,
    pub completion: Nvfp4CudaCompletionReport,
    pub numerical: Vec<Nvfp4CudaNumericalReport>,
    pub resources: Nvfp4CudaResourceReport,
    pub claims: Nvfp4CudaClaimReport,
}

#[derive(Clone, Debug, Serialize)]
pub struct Nvfp4CudaDeviceReport {
    pub ordinal: i32,
    pub name: String,
    pub compute_capability: [u16; 2],
    pub total_global_memory_bytes: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Nvfp4CudaTensorReport {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub ggml_type: u32,
    pub descriptor_payload_bytes: u64,
    pub payload_bytes_read: usize,
    pub payload_sha256: String,
    pub tensor_level_scale: Option<f32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Nvfp4CudaSourceReport {
    pub path: String,
    pub file_sha256: String,
    pub tensors: Vec<Nvfp4CudaTensorReport>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Nvfp4CudaInputReport {
    pub token_id: usize,
    pub source: String,
    pub normalized_elements: usize,
    pub normalized_input_sha256: String,
    pub down_input_sha256: String,
    pub down_input_source: String,
    pub norm_epsilon: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct Nvfp4CudaProjectionShape {
    pub name: String,
    pub rows: usize,
    pub columns: usize,
    pub blocks_per_row: usize,
    pub row_bytes: usize,
    pub payload_bytes: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct Nvfp4CudaShapeReport {
    pub block_elements: usize,
    pub block_bytes: usize,
    pub projections: Vec<Nvfp4CudaProjectionShape>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Nvfp4CudaTimingReport {
    pub warmup_passes: usize,
    pub steady_passes: usize,
    pub projections_per_pass: usize,
    pub activation_ops_per_pass: usize,
    pub cuda_event_ms: f32,
    pub batch_wall_ms: f64,
    pub per_pass_event_ms: f64,
    pub per_projection_event_ms: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Nvfp4CudaQ8_1Dp4aReport {
    pub activation_layout: String,
    pub activation_block_elements: usize,
    pub activation_block_bytes: usize,
    pub normalized_workspace_bytes: usize,
    pub down_workspace_bytes: usize,
    pub gate_up_shared_input_quantization: bool,
    pub timing: Nvfp4CudaQ8_1TimingReport,
    pub activation_compatibility: Vec<Nvfp4CudaQ8_1CompatibilityReport>,
    pub numerical: Vec<Nvfp4CudaNumericalReport>,
    pub completion: Nvfp4CudaCompletionReport,
    pub steady_allocations: AllocationDeltaReport,
    pub steady_host_boundaries: HostBoundaryDeltaReport,
    pub readback_host_boundaries: HostBoundaryDeltaReport,
}

#[derive(Clone, Debug, Serialize)]
pub struct Nvfp4CudaQ8_1TimingReport {
    pub warmup_passes: usize,
    pub steady_passes: usize,
    pub activation_quantizations_per_pass: usize,
    pub projections_per_pass: usize,
    pub activation_ops_per_pass: usize,
    pub cuda_event_ms: f32,
    pub batch_wall_ms: f64,
    pub per_pass_event_ms: f64,
    pub speedup_over_f32_activation: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Nvfp4CudaQ8_1CompatibilityReport {
    pub name: String,
    pub elements: usize,
    pub q8_1_payload_bytes: usize,
    pub q8_0_projection_bytes: usize,
    pub metadata_sum_bytes_ignored: usize,
    pub q8_0_projection_sha256: String,
    pub expected_q8_0_sha256: String,
    pub exact_scale_and_code_bytes: bool,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Nvfp4CudaCompletionReport {
    pub poll_iterations: u64,
    pub nonblocking_poll_only: bool,
    pub completion_proof_required: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct Nvfp4CudaNumericalReport {
    pub name: String,
    pub rows: usize,
    pub finite_count: usize,
    pub expected_count: usize,
    pub max_abs_diff: f32,
    pub max_relative_diff: f32,
    pub absolute_tolerance: f32,
    pub relative_tolerance: f32,
    pub output_sha256: String,
    pub expected_sha256: String,
    pub pass: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct Nvfp4CudaResourceReport {
    pub setup_allocations: AllocationDeltaReport,
    pub steady_allocations: AllocationDeltaReport,
    pub completion_poll_host_boundaries: HostBoundaryDeltaReport,
    pub setup_host_boundaries: HostBoundaryDeltaReport,
    pub steady_host_boundaries: HostBoundaryDeltaReport,
    pub readback_host_boundaries: HostBoundaryDeltaReport,
    pub final_allocation_delta_after_cleanup: AllocationDeltaReport,
}

#[derive(Clone, Debug, Serialize)]
pub struct AllocationDeltaReport {
    pub successful_allocations: u64,
    pub successful_frees: u64,
    pub live_allocations_after: u64,
    pub live_bytes_after: u64,
    pub peak_live_allocations_observed: u64,
    pub peak_live_bytes_observed: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct HostBoundaryDeltaReport {
    pub host_waits: u64,
    pub stream_synchronizations: u64,
    pub event_synchronizations: u64,
    pub synchronous_h2d_copies: u64,
    pub synchronous_h2d_bytes: u64,
    pub synchronous_d2h_copies: u64,
    pub synchronous_d2h_bytes: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Nvfp4CudaClaimReport {
    pub real_weight_nvfp4_projection_boundary_executed: bool,
    pub q8_1_activation_dp4a_boundary_executed: bool,
    pub gpu_silu_mul_boundary_executed: bool,
    pub complete_transformer_executed: bool,
    pub decode_executed: bool,
    pub semantic_parity_established: bool,
    pub product_tps_permitted: bool,
    pub default_cli_enabled: bool,
}

/// Execute the authenticated real-weight layer-0 NVFP4 projection boundary.
pub fn run_qwen35_real_nvfp4_cuda(
    config: &Qwen35RealNvfp4CudaConfig,
) -> Result<Qwen35RealNvfp4CudaReport, Qwen35RealNvfp4CudaError> {
    validate_config(config)?;
    let config_bytes = std::fs::read(&config.config_path).map_err(Qwen35Error::from)?;
    let config_sha256 = sha256_hex(&config_bytes);
    let model_config = Qwen35Config::parse(&config_bytes)?;
    if model_config.hidden_size != HIDDEN
        || model_config.vocab_size != VOCAB
        || model_config.num_hidden_layers != 64
        || !model_config.linear_attention_layers.contains(&LAYER)
    {
        return Err(Qwen35RealNvfp4CudaError::InvalidConfiguration(
            "Qwen config geometry",
        ));
    }

    let manifest = GgufManifest::inspect(&config.gguf_path, GgufAuditOptions::default())?;
    let embedding = require_tensor(
        &manifest,
        EMBEDDING,
        &[HIDDEN as u64, VOCAB as u64],
        GGML_TYPE_BF16,
    )?;
    let attn_norm = require_tensor(&manifest, ATTN_NORM, &[HIDDEN as u64], GGML_TYPE_F32)?;
    let gate = require_tensor(
        &manifest,
        FFN_GATE,
        &[HIDDEN as u64, INTERMEDIATE as u64],
        GGML_TYPE_NVFP4,
    )?;
    let up = require_tensor(
        &manifest,
        FFN_UP,
        &[HIDDEN as u64, INTERMEDIATE as u64],
        GGML_TYPE_NVFP4,
    )?;
    let down = require_tensor(
        &manifest,
        FFN_DOWN,
        &[INTERMEDIATE as u64, HIDDEN as u64],
        GGML_TYPE_NVFP4,
    )?;
    let gate_scale_desc = require_tensor(&manifest, FFN_GATE_SCALE, &[1], GGML_TYPE_F32)?;
    let up_scale_desc = require_tensor(&manifest, FFN_UP_SCALE, &[1], GGML_TYPE_F32)?;
    let down_scale_desc = require_tensor(&manifest, FFN_DOWN_SCALE, &[1], GGML_TYPE_F32)?;

    let mut reader = manifest.open_tensor_reader(&config.gguf_path)?;
    let embedding_row_bytes =
        HIDDEN
            .checked_mul(2)
            .ok_or(Qwen35RealNvfp4CudaError::InvalidConfiguration(
                "embedding row bytes",
            ))?;
    let embedding_offset = config.token_id.checked_mul(embedding_row_bytes).ok_or(
        Qwen35RealNvfp4CudaError::InvalidConfiguration("embedding offset"),
    )?;
    let embedding_bytes = reader.read_tensor_slice(
        EMBEDDING,
        u64::try_from(embedding_offset)
            .map_err(|_| Qwen35RealNvfp4CudaError::InvalidConfiguration("embedding offset"))?,
        u64::try_from(embedding_row_bytes)
            .map_err(|_| Qwen35RealNvfp4CudaError::InvalidConfiguration("embedding row bytes"))?,
    )?;
    let mut embedding_values = vec![0.0_f32; HIDDEN];
    dequantize_bf16(&embedding_bytes, &mut embedding_values)?;
    let norm_bytes = reader.read_tensor_bytes(ATTN_NORM)?;
    let norm_values = dequantize_tensor(&attn_norm, &norm_bytes)?;
    let normalized = rms_norm_last_dim(
        &ReferenceTensor::new(vec![1, HIDDEN], embedding_values)?,
        &ReferenceTensor::new(vec![HIDDEN], norm_values)?,
        model_config.rms_norm_eps as f32,
    )?
    .into_values();
    if normalized.len() != HIDDEN || normalized.iter().any(|value| !value.is_finite()) {
        return Err(Qwen35RealNvfp4CudaError::InvalidConfiguration(
            "normalized embedding",
        ));
    }

    let gate_payload = reader.read_tensor_bytes(FFN_GATE)?;
    let up_payload = reader.read_tensor_bytes(FFN_UP)?;
    let down_payload = reader.read_tensor_bytes(FFN_DOWN)?;
    let gate_scale = positive_scale(
        &dequantize_tensor(&gate_scale_desc, &reader.read_tensor_bytes(FFN_GATE_SCALE)?)?,
        "ffn_gate.scale",
    )?;
    let up_scale = positive_scale(
        &dequantize_tensor(&up_scale_desc, &reader.read_tensor_bytes(FFN_UP_SCALE)?)?,
        "ffn_up.scale",
    )?;
    let down_scale = positive_scale(
        &dequantize_tensor(&down_scale_desc, &reader.read_tensor_bytes(FFN_DOWN_SCALE)?)?,
        "ffn_down.scale",
    )?;

    let gate_raw = quantized_matvec_fused_simd(&gate, &gate_payload, &normalized)?;
    let up_raw = quantized_matvec_fused_simd(&up, &up_payload, &normalized)?;
    let gate_expected = scale_values(&gate_raw, gate_scale, "gate output")?;
    let up_expected = scale_values(&up_raw, up_scale, "up output")?;
    let mut down_input = Vec::with_capacity(INTERMEDIATE);
    for (gate_value, up_value) in gate_expected.iter().zip(&up_expected) {
        let sigmoid = stable_sigmoid(*gate_value);
        let hidden = *up_value * (*gate_value * sigmoid);
        if !hidden.is_finite() {
            return Err(Qwen35RealNvfp4CudaError::InvalidConfiguration(
                "SwiGLU input",
            ));
        }
        down_input.push(hidden);
    }
    let down_raw = quantized_matvec_fused_simd(&down, &down_payload, &down_input)?;
    let down_expected = scale_values(&down_raw, down_scale, "down output")?;

    // Mirror llama.cpp/SGLang's B=1 quantized GEMV boundary: gate and up
    // consume one shared activation quantization, while down consumes the
    // post-SwiGLU activation quantized independently.
    let normalized_q8_0_expected = quantize_q8_0_activation(&normalized)?;
    let gate_q8_raw = quantized_matvec_activation_q8_0(&gate, &gate_payload, &normalized)?;
    let up_q8_raw = quantized_matvec_activation_q8_0(&up, &up_payload, &normalized)?;
    let gate_q8_expected = scale_values(&gate_q8_raw, gate_scale, "Q8_1 gate output")?;
    let up_q8_expected = scale_values(&up_q8_raw, up_scale, "Q8_1 up output")?;
    let mut down_q8_input = Vec::with_capacity(INTERMEDIATE);
    for (gate_value, up_value) in gate_q8_expected.iter().zip(&up_q8_expected) {
        let hidden = *up_value * (*gate_value * stable_sigmoid(*gate_value));
        if !hidden.is_finite() {
            return Err(Qwen35RealNvfp4CudaError::InvalidConfiguration(
                "Q8_1 SwiGLU input",
            ));
        }
        down_q8_input.push(hidden);
    }
    let down_q8_0_expected = quantize_q8_0_activation(&down_q8_input)?;
    let down_q8_raw = quantized_matvec_activation_q8_0(&down, &down_payload, &down_q8_input)?;
    let down_q8_expected = scale_values(&down_q8_raw, down_scale, "Q8_1 down output")?;

    let device = cuda_device_properties(config.device_ordinal)?;
    let stream = OwnedCudaStream::create(config.device_ordinal)?;
    let allocation_start = cuda_device_allocation_snapshot();
    let boundary_start = cuda_host_boundary_snapshot();

    let normalized_bytes = f32_bytes(&normalized);
    let down_input_bytes = f32_bytes(&down_input);
    let gate_weight = OwnedCudaBuffer::allocate(config.device_ordinal, gate_payload.len())?;
    let up_weight = OwnedCudaBuffer::allocate(config.device_ordinal, up_payload.len())?;
    let down_weight = OwnedCudaBuffer::allocate(config.device_ordinal, down_payload.len())?;
    let normalized_buffer =
        OwnedCudaBuffer::allocate(config.device_ordinal, normalized_bytes.len())?;
    let gate_output = OwnedCudaBuffer::allocate(
        config.device_ordinal,
        INTERMEDIATE
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or(Qwen35RealNvfp4CudaError::InvalidConfiguration(
                "gate output bytes",
            ))?,
    )?;
    let up_output = OwnedCudaBuffer::allocate(
        config.device_ordinal,
        INTERMEDIATE
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or(Qwen35RealNvfp4CudaError::InvalidConfiguration(
                "up output bytes",
            ))?,
    )?;
    let activation_output =
        OwnedCudaBuffer::allocate(config.device_ordinal, down_input_bytes.len())?;
    let down_output = OwnedCudaBuffer::allocate(
        config.device_ordinal,
        HIDDEN.checked_mul(core::mem::size_of::<f32>()).ok_or(
            Qwen35RealNvfp4CudaError::InvalidConfiguration("down output bytes"),
        )?,
    )?;
    let normalized_q8_1_bytes = q8_1_workspace_bytes(HIDDEN)?;
    let down_q8_1_bytes = q8_1_workspace_bytes(INTERMEDIATE)?;
    let normalized_q8_1 = OwnedCudaBuffer::allocate(config.device_ordinal, normalized_q8_1_bytes)?;
    let down_q8_1 = OwnedCudaBuffer::allocate(config.device_ordinal, down_q8_1_bytes)?;
    let gate_q8_output = OwnedCudaBuffer::allocate(
        config.device_ordinal,
        INTERMEDIATE
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or(Qwen35RealNvfp4CudaError::InvalidConfiguration(
                "Q8_1 gate output bytes",
            ))?,
    )?;
    let up_q8_output = OwnedCudaBuffer::allocate(
        config.device_ordinal,
        INTERMEDIATE
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or(Qwen35RealNvfp4CudaError::InvalidConfiguration(
                "Q8_1 up output bytes",
            ))?,
    )?;
    let activation_q8_output =
        OwnedCudaBuffer::allocate(config.device_ordinal, down_input_bytes.len())?;
    let down_q8_output = OwnedCudaBuffer::allocate(
        config.device_ordinal,
        HIDDEN.checked_mul(core::mem::size_of::<f32>()).ok_or(
            Qwen35RealNvfp4CudaError::InvalidConfiguration("Q8_1 down output bytes"),
        )?,
    )?;

    gate_weight.copy_from_host_sync(&stream, &gate_payload)?;
    up_weight.copy_from_host_sync(&stream, &up_payload)?;
    down_weight.copy_from_host_sync(&stream, &down_payload)?;
    normalized_buffer.copy_from_host_sync(&stream, &normalized_bytes)?;

    let normalized_view = CudaF32View::new(
        normalized_buffer.view(0, normalized_bytes.len())?,
        PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
        HIDDEN,
    )?;
    let activation_view = CudaF32View::new(
        activation_output.view(0, down_input_bytes.len())?,
        PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
        INTERMEDIATE,
    )?;
    let gate_args = bind_args(
        &stream,
        &gate_weight,
        &gate_output,
        INTERMEDIATE,
        HIDDEN,
        normalized_view,
        gate_scale,
    )?;
    let up_args = bind_args(
        &stream,
        &up_weight,
        &up_output,
        INTERMEDIATE,
        HIDDEN,
        normalized_view,
        up_scale,
    )?;
    let silu_args = CudaSiluMulF32ArgsV1::bind(
        &stream,
        CudaF32View::new(
            gate_output.view(0, down_input_bytes.len())?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            INTERMEDIATE,
        )?,
        CudaF32View::new(
            up_output.view(0, down_input_bytes.len())?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            INTERMEDIATE,
        )?,
        activation_view,
    )?;
    let down_args = bind_args(
        &stream,
        &down_weight,
        &down_output,
        HIDDEN,
        INTERMEDIATE,
        activation_view,
        down_scale,
    )?;
    let normalized_q8_1_view = CudaQ8_1ActivationViewV1::new(
        normalized_q8_1.view(0, normalized_q8_1_bytes)?,
        PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_1_ACTIVATION_CUDA_V1,
        HIDDEN,
    )?;
    let down_q8_1_view = CudaQ8_1ActivationViewV1::new(
        down_q8_1.view(0, down_q8_1_bytes)?,
        PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_1_ACTIVATION_CUDA_V1,
        INTERMEDIATE,
    )?;
    let normalized_q8_quantize_args =
        CudaQ8_1ActivationQuantizeArgsV1::bind(&stream, normalized_view, normalized_q8_1_view)?;
    let gate_q8_args = bind_q8_1_args(
        &stream,
        &gate_weight,
        &gate_q8_output,
        INTERMEDIATE,
        HIDDEN,
        normalized_q8_1_view,
        gate_scale,
    )?;
    let up_q8_args = bind_q8_1_args(
        &stream,
        &up_weight,
        &up_q8_output,
        INTERMEDIATE,
        HIDDEN,
        normalized_q8_1_view,
        up_scale,
    )?;
    let activation_q8_view = CudaF32View::new(
        activation_q8_output.view(0, down_input_bytes.len())?,
        PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
        INTERMEDIATE,
    )?;
    let silu_q8_args = CudaSiluMulF32ArgsV1::bind(
        &stream,
        CudaF32View::new(
            gate_q8_output.view(0, down_input_bytes.len())?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            INTERMEDIATE,
        )?,
        CudaF32View::new(
            up_q8_output.view(0, down_input_bytes.len())?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            INTERMEDIATE,
        )?,
        activation_q8_view,
    )?;
    let down_q8_quantize_args =
        CudaQ8_1ActivationQuantizeArgsV1::bind(&stream, activation_q8_view, down_q8_1_view)?;
    let down_q8_args = bind_q8_1_args(
        &stream,
        &down_weight,
        &down_q8_output,
        HIDDEN,
        INTERMEDIATE,
        down_q8_1_view,
        down_scale,
    )?;

    let setup_allocations = delta_allocations(allocation_start, cuda_device_allocation_snapshot())?;
    let setup_host_boundaries = delta_boundaries(boundary_start, cuda_host_boundary_snapshot())?;

    for _ in 0..config.warmup {
        submit_all(&stream, &gate_args, &up_args, &silu_args, &down_args)?;
    }
    stream.synchronize()?;

    let steady_allocation_start = cuda_device_allocation_snapshot();
    let steady_boundary_start = cuda_host_boundary_snapshot();
    let started = OwnedCudaEvent::create(config.device_ordinal)?;
    let stopped = OwnedCudaEvent::create(config.device_ordinal)?;
    let wall_start = Instant::now();
    started.record(&stream)?;
    for _ in 0..config.iterations {
        submit_all(&stream, &gate_args, &up_args, &silu_args, &down_args)?;
    }
    stopped.record(&stream)?;
    stopped.synchronize()?;
    let batch_wall_ms = wall_start.elapsed().as_secs_f64() * 1_000.0;
    let cuda_event_ms = started.elapsed_ms(&stopped)?;
    let steady_allocations =
        delta_allocations(steady_allocation_start, cuda_device_allocation_snapshot())?;
    let steady_host_boundaries =
        delta_boundaries(steady_boundary_start, cuda_host_boundary_snapshot())?;

    for _ in 0..config.warmup {
        submit_q8_1_all(
            &stream,
            &normalized_q8_quantize_args,
            &gate_q8_args,
            &up_q8_args,
            &silu_q8_args,
            &down_q8_quantize_args,
            &down_q8_args,
        )?;
    }
    stream.synchronize()?;

    let q8_1_steady_allocation_start = cuda_device_allocation_snapshot();
    let q8_1_steady_boundary_start = cuda_host_boundary_snapshot();
    let q8_1_started = OwnedCudaEvent::create(config.device_ordinal)?;
    let q8_1_stopped = OwnedCudaEvent::create(config.device_ordinal)?;
    let q8_1_wall_start = Instant::now();
    q8_1_started.record(&stream)?;
    for _ in 0..config.iterations {
        submit_q8_1_all(
            &stream,
            &normalized_q8_quantize_args,
            &gate_q8_args,
            &up_q8_args,
            &silu_q8_args,
            &down_q8_quantize_args,
            &down_q8_args,
        )?;
    }
    q8_1_stopped.record(&stream)?;
    q8_1_stopped.synchronize()?;
    let q8_1_batch_wall_ms = q8_1_wall_start.elapsed().as_secs_f64() * 1_000.0;
    let q8_1_cuda_event_ms = q8_1_started.elapsed_ms(&q8_1_stopped)?;
    let q8_1_steady_allocations = delta_allocations(
        q8_1_steady_allocation_start,
        cuda_device_allocation_snapshot(),
    )?;
    let q8_1_steady_host_boundaries =
        delta_boundaries(q8_1_steady_boundary_start, cuda_host_boundary_snapshot())?;

    let completion_event = OwnedCudaEvent::create(config.device_ordinal)?;
    let completion_boundary_start = cuda_host_boundary_snapshot();
    submit_all(&stream, &gate_args, &up_args, &silu_args, &down_args)?;
    submit_q8_1_all(
        &stream,
        &normalized_q8_quantize_args,
        &gate_q8_args,
        &up_q8_args,
        &silu_q8_args,
        &down_q8_quantize_args,
        &down_q8_args,
    )?;
    let receipt = stream.submission_receipt()?;
    completion_event.record_for_submission(&stream, receipt)?;
    let poll_iterations = poll_completion(&stream, &completion_event, receipt)?;
    let completion_poll_host_boundaries =
        delta_boundaries(completion_boundary_start, cuda_host_boundary_snapshot())?;

    let readback_boundary_start = cuda_host_boundary_snapshot();
    let mut gate_output_bytes = vec![0_u8; INTERMEDIATE * core::mem::size_of::<f32>()];
    let mut up_output_bytes = vec![0_u8; INTERMEDIATE * core::mem::size_of::<f32>()];
    let mut activation_output_bytes = vec![0_u8; INTERMEDIATE * core::mem::size_of::<f32>()];
    let mut down_output_bytes = vec![0_u8; HIDDEN * core::mem::size_of::<f32>()];
    let mut normalized_q8_1_payload = vec![0_u8; normalized_q8_1_bytes];
    let mut down_q8_1_payload = vec![0_u8; down_q8_1_bytes];
    let mut gate_q8_output_bytes = vec![0_u8; INTERMEDIATE * core::mem::size_of::<f32>()];
    let mut up_q8_output_bytes = vec![0_u8; INTERMEDIATE * core::mem::size_of::<f32>()];
    let mut activation_q8_output_bytes = vec![0_u8; INTERMEDIATE * core::mem::size_of::<f32>()];
    let mut down_q8_output_bytes = vec![0_u8; HIDDEN * core::mem::size_of::<f32>()];
    gate_output.copy_to_host_sync_with_submission_proof(
        &stream,
        &mut gate_output_bytes,
        receipt,
    )?;
    up_output.copy_to_host_sync_with_submission_proof(&stream, &mut up_output_bytes, receipt)?;
    activation_output.copy_to_host_sync_with_submission_proof(
        &stream,
        &mut activation_output_bytes,
        receipt,
    )?;
    down_output.copy_to_host_sync_with_submission_proof(
        &stream,
        &mut down_output_bytes,
        receipt,
    )?;
    normalized_q8_1.copy_to_host_sync_with_submission_proof(
        &stream,
        &mut normalized_q8_1_payload,
        receipt,
    )?;
    down_q8_1.copy_to_host_sync_with_submission_proof(&stream, &mut down_q8_1_payload, receipt)?;
    gate_q8_output.copy_to_host_sync_with_submission_proof(
        &stream,
        &mut gate_q8_output_bytes,
        receipt,
    )?;
    up_q8_output.copy_to_host_sync_with_submission_proof(
        &stream,
        &mut up_q8_output_bytes,
        receipt,
    )?;
    activation_q8_output.copy_to_host_sync_with_submission_proof(
        &stream,
        &mut activation_q8_output_bytes,
        receipt,
    )?;
    down_q8_output.copy_to_host_sync_with_submission_proof(
        &stream,
        &mut down_q8_output_bytes,
        receipt,
    )?;
    let readback_host_boundaries =
        delta_boundaries(readback_boundary_start, cuda_host_boundary_snapshot())?;

    let gate_actual = f32_vec_from_bytes(&gate_output_bytes);
    let up_actual = f32_vec_from_bytes(&up_output_bytes);
    let activation_actual = f32_vec_from_bytes(&activation_output_bytes);
    let down_actual = f32_vec_from_bytes(&down_output_bytes);
    let gate_q8_actual = f32_vec_from_bytes(&gate_q8_output_bytes);
    let up_q8_actual = f32_vec_from_bytes(&up_q8_output_bytes);
    let activation_q8_actual = f32_vec_from_bytes(&activation_q8_output_bytes);
    let down_q8_actual = f32_vec_from_bytes(&down_q8_output_bytes);
    let numerical = vec![
        compare_output(FFN_GATE, &gate_actual, &gate_expected, &gate_output_bytes),
        compare_output(FFN_UP, &up_actual, &up_expected, &up_output_bytes),
        compare_output_with_tolerances(
            FFN_SILU_MUL,
            &activation_actual,
            &down_input,
            &activation_output_bytes,
            5.0e-5,
            5.0e-5,
        ),
        compare_output(FFN_DOWN, &down_actual, &down_expected, &down_output_bytes),
    ];
    let activation_compatibility = vec![
        compare_q8_1_activation_payload(
            "normalized",
            HIDDEN,
            &normalized_q8_1_payload,
            &normalized_q8_0_expected,
        )?,
        compare_q8_1_activation_payload(
            "post_swiglu",
            INTERMEDIATE,
            &down_q8_1_payload,
            &down_q8_0_expected,
        )?,
    ];
    let q8_1_numerical = vec![
        compare_output(
            "blk.0.ffn_gate.weight:q8_1_dp4a",
            &gate_q8_actual,
            &gate_q8_expected,
            &gate_q8_output_bytes,
        ),
        compare_output(
            "blk.0.ffn_up.weight:q8_1_dp4a",
            &up_q8_actual,
            &up_q8_expected,
            &up_q8_output_bytes,
        ),
        compare_output_with_tolerances(
            "ffn_silu_mul:q8_1_dp4a",
            &activation_q8_actual,
            &down_q8_input,
            &activation_q8_output_bytes,
            5.0e-5,
            5.0e-5,
        ),
        compare_output(
            "blk.0.ffn_down.weight:q8_1_dp4a",
            &down_q8_actual,
            &down_q8_expected,
            &down_q8_output_bytes,
        ),
    ];
    let pass = numerical.iter().all(|result| result.pass)
        && q8_1_numerical.iter().all(|result| result.pass)
        && activation_compatibility
            .iter()
            .all(|result| result.exact_scale_and_code_bytes);

    drop(completion_event);
    drop(q8_1_stopped);
    drop(q8_1_started);
    drop(stopped);
    drop(started);
    drop(down_output);
    drop(down_q8_output);
    drop(activation_q8_output);
    drop(up_q8_output);
    drop(gate_q8_output);
    drop(down_q8_1);
    drop(normalized_q8_1);
    drop(up_output);
    drop(gate_output);
    drop(activation_output);
    drop(normalized_buffer);
    drop(down_weight);
    drop(up_weight);
    drop(gate_weight);
    drop(stream);
    let final_allocation_delta_after_cleanup =
        delta_allocations(allocation_start, cuda_device_allocation_snapshot())?;

    let source_file_sha256 =
        manifest
            .file_sha256
            .clone()
            .ok_or(Qwen35RealNvfp4CudaError::InvalidConfiguration(
                "GGUF file hash",
            ))?;
    let tensors = vec![
        tensor_report(&embedding, &embedding_bytes, None),
        tensor_report(&attn_norm, &norm_bytes, None),
        tensor_report(&gate, &gate_payload, Some(gate_scale)),
        tensor_report(&up, &up_payload, Some(up_scale)),
        tensor_report(&down, &down_payload, Some(down_scale)),
    ];

    Ok(Qwen35RealNvfp4CudaReport {
        schema: QWEN35_REAL_NVFP4_CUDA_SCHEMA.to_owned(),
        status: if pass {
            "PASS_COMPONENT_REAL_NVFP4_FFN_SWIGLU_CHAIN_CUDA_Q8_1_DP4A".to_owned()
        } else {
            "FAIL_COMPONENT_REAL_NVFP4_FFN_SWIGLU_CHAIN_CUDA_NUMERICAL".to_owned()
        },
        component_only: true,
        source_commit: config.source_commit.clone(),
        config_path: config.config_path.display().to_string(),
        config_sha256,
        device: Nvfp4CudaDeviceReport {
            ordinal: device.ordinal,
            name: device.name,
            compute_capability: [
                device.compute_capability.major,
                device.compute_capability.minor,
            ],
            total_global_memory_bytes: device.total_global_memory_bytes,
        },
        source: Nvfp4CudaSourceReport {
            path: config.gguf_path.display().to_string(),
            file_sha256: source_file_sha256,
            tensors,
        },
        input: Nvfp4CudaInputReport {
            token_id: config.token_id,
            source: "real_token_embedding_row_then_layer0_attn_rmsnorm".to_owned(),
            normalized_elements: normalized.len(),
            normalized_input_sha256: sha256_f32_bits(&normalized),
            down_input_sha256: sha256_f32_bits(&down_input),
            down_input_source: "gpu_cuda_silu_mul_f32_unclamped".to_owned(),
            norm_epsilon: model_config.rms_norm_eps as f32,
        },
        shape: Nvfp4CudaShapeReport {
            block_elements: NVFP4_BLOCK_ELEMENTS,
            block_bytes: NVFP4_BLOCK_BYTES,
            projections: vec![
                projection_shape(FFN_GATE, &gate, &gate_payload)?,
                projection_shape(FFN_UP, &up, &up_payload)?,
                projection_shape(FFN_DOWN, &down, &down_payload)?,
            ],
        },
        timing: Nvfp4CudaTimingReport {
            warmup_passes: config.warmup,
            steady_passes: config.iterations,
            projections_per_pass: 3,
            activation_ops_per_pass: 1,
            cuda_event_ms,
            batch_wall_ms,
            per_pass_event_ms: f64::from(cuda_event_ms) / config.iterations as f64,
            per_projection_event_ms: f64::from(cuda_event_ms) / (config.iterations * 3) as f64,
        },
        q8_1_dp4a: Nvfp4CudaQ8_1Dp4aReport {
            activation_layout: "llama_cpp_block_q8_1_fp16_scale_fp16_sum_i8x32".to_owned(),
            activation_block_elements: Q8_1_ACTIVATION_BLOCK_ELEMENTS,
            activation_block_bytes: Q8_1_ACTIVATION_BLOCK_BYTES,
            normalized_workspace_bytes: normalized_q8_1_bytes,
            down_workspace_bytes: down_q8_1_bytes,
            gate_up_shared_input_quantization: true,
            timing: Nvfp4CudaQ8_1TimingReport {
                warmup_passes: config.warmup,
                steady_passes: config.iterations,
                activation_quantizations_per_pass: 2,
                projections_per_pass: 3,
                activation_ops_per_pass: 1,
                cuda_event_ms: q8_1_cuda_event_ms,
                batch_wall_ms: q8_1_batch_wall_ms,
                per_pass_event_ms: f64::from(q8_1_cuda_event_ms) / config.iterations as f64,
                speedup_over_f32_activation: f64::from(cuda_event_ms)
                    / f64::from(q8_1_cuda_event_ms),
            },
            activation_compatibility,
            numerical: q8_1_numerical,
            completion: Nvfp4CudaCompletionReport {
                poll_iterations,
                nonblocking_poll_only: true,
                completion_proof_required: true,
            },
            steady_allocations: q8_1_steady_allocations,
            steady_host_boundaries: q8_1_steady_host_boundaries,
            readback_host_boundaries: readback_host_boundaries.clone(),
        },
        completion: Nvfp4CudaCompletionReport {
            poll_iterations,
            nonblocking_poll_only: true,
            completion_proof_required: true,
        },
        numerical,
        resources: Nvfp4CudaResourceReport {
            setup_allocations,
            steady_allocations,
            completion_poll_host_boundaries,
            setup_host_boundaries,
            steady_host_boundaries,
            readback_host_boundaries,
            final_allocation_delta_after_cleanup,
        },
        claims: Nvfp4CudaClaimReport {
            real_weight_nvfp4_projection_boundary_executed: true,
            q8_1_activation_dp4a_boundary_executed: true,
            gpu_silu_mul_boundary_executed: true,
            complete_transformer_executed: false,
            decode_executed: false,
            semantic_parity_established: false,
            product_tps_permitted: false,
            default_cli_enabled: false,
        },
    })
}

fn validate_config(config: &Qwen35RealNvfp4CudaConfig) -> Result<(), Qwen35RealNvfp4CudaError> {
    if config.gguf_path.as_os_str().is_empty() {
        return Err(Qwen35RealNvfp4CudaError::InvalidConfiguration("gguf_path"));
    }
    if config.config_path.as_os_str().is_empty() {
        return Err(Qwen35RealNvfp4CudaError::InvalidConfiguration(
            "config_path",
        ));
    }
    if config.device_ordinal < 0 {
        return Err(Qwen35RealNvfp4CudaError::InvalidConfiguration(
            "device_ordinal",
        ));
    }
    if config.token_id >= VOCAB {
        return Err(Qwen35RealNvfp4CudaError::InvalidConfiguration("token_id"));
    }
    if config.iterations == 0 {
        return Err(Qwen35RealNvfp4CudaError::InvalidConfiguration("iterations"));
    }
    Ok(())
}

fn require_tensor(
    manifest: &GgufManifest,
    name: &str,
    dimensions: &[u64],
    ggml_type: u32,
) -> Result<GgufTensor, Qwen35RealNvfp4CudaError> {
    let tensor = manifest
        .tensor(name)
        .ok_or(Qwen35RealNvfp4CudaError::InvalidConfiguration(
            "required tensor missing",
        ))?
        .clone();
    if tensor.dimensions != dimensions || tensor.ggml_type != ggml_type {
        return Err(Qwen35RealNvfp4CudaError::InvalidConfiguration(
            "required tensor geometry or type",
        ));
    }
    Ok(tensor)
}

fn positive_scale(values: &[f32], field: &'static str) -> Result<f32, Qwen35RealNvfp4CudaError> {
    if values.len() != 1 || !values[0].is_finite() || values[0] <= 0.0 {
        return Err(Qwen35RealNvfp4CudaError::InvalidConfiguration(field));
    }
    Ok(values[0])
}

fn scale_values(
    values: &[f32],
    scale: f32,
    field: &'static str,
) -> Result<Vec<f32>, Qwen35RealNvfp4CudaError> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err(Qwen35RealNvfp4CudaError::InvalidConfiguration(field));
    }
    values
        .iter()
        .map(|value| {
            let scaled = *value * scale;
            if scaled.is_finite() {
                Ok(scaled)
            } else {
                Err(Qwen35RealNvfp4CudaError::InvalidConfiguration(field))
            }
        })
        .collect()
}

fn stable_sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponent = value.exp();
        exponent / (1.0 + exponent)
    }
}

fn bind_args(
    stream: &OwnedCudaStream,
    weight_buffer: &OwnedCudaBuffer,
    output_buffer: &OwnedCudaBuffer,
    rows: usize,
    columns: usize,
    input: CudaF32View,
    tensor_scale: f32,
) -> Result<CudaNvfp4Sm70F32MatvecArgsV1, Qwen35RealNvfp4CudaError> {
    let weight_bytes = rows
        .checked_mul(columns / NVFP4_BLOCK_ELEMENTS)
        .and_then(|blocks| blocks.checked_mul(NVFP4_BLOCK_BYTES))
        .ok_or(Qwen35RealNvfp4CudaError::InvalidConfiguration(
            "NVFP4 weight bytes",
        ))?;
    let output_bytes = rows.checked_mul(core::mem::size_of::<f32>()).ok_or(
        Qwen35RealNvfp4CudaError::InvalidConfiguration("NVFP4 output bytes"),
    )?;
    Ok(CudaNvfp4Sm70F32MatvecArgsV1::bind(
        stream,
        CudaNvfp4Sm70WeightViewV1::new(
            weight_buffer.view(0, weight_bytes)?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_NVFP4_CUDA_SM70_V1,
            rows,
            columns,
        )?,
        input,
        CudaF32View::new(
            output_buffer.view(0, output_bytes)?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            rows,
        )?,
        tensor_scale,
    )?)
}

fn bind_q8_1_args(
    stream: &OwnedCudaStream,
    weight_buffer: &OwnedCudaBuffer,
    output_buffer: &OwnedCudaBuffer,
    rows: usize,
    columns: usize,
    input: CudaQ8_1ActivationViewV1,
    tensor_scale: f32,
) -> Result<CudaNvfp4Sm70Q8_1MatvecArgsV1, Qwen35RealNvfp4CudaError> {
    let weight_bytes = rows
        .checked_mul(columns / NVFP4_BLOCK_ELEMENTS)
        .and_then(|blocks| blocks.checked_mul(NVFP4_BLOCK_BYTES))
        .ok_or(Qwen35RealNvfp4CudaError::InvalidConfiguration(
            "NVFP4 Q8_1 weight bytes",
        ))?;
    let output_bytes = rows.checked_mul(core::mem::size_of::<f32>()).ok_or(
        Qwen35RealNvfp4CudaError::InvalidConfiguration("NVFP4 Q8_1 output bytes"),
    )?;
    Ok(CudaNvfp4Sm70Q8_1MatvecArgsV1::bind(
        stream,
        CudaNvfp4Sm70WeightViewV1::new(
            weight_buffer.view(0, weight_bytes)?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_NVFP4_CUDA_SM70_V1,
            rows,
            columns,
        )?,
        input,
        CudaF32View::new(
            output_buffer.view(0, output_bytes)?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            rows,
        )?,
        tensor_scale,
    )?)
}

fn submit_all(
    stream: &OwnedCudaStream,
    gate: &CudaNvfp4Sm70F32MatvecArgsV1,
    up: &CudaNvfp4Sm70F32MatvecArgsV1,
    silu: &CudaSiluMulF32ArgsV1,
    down: &CudaNvfp4Sm70F32MatvecArgsV1,
) -> Result<(), Qwen35RealNvfp4CudaError> {
    gate.submit_for_plan(stream)?;
    up.submit_for_plan(stream)?;
    silu.submit_for_plan(stream)?;
    down.submit_for_plan(stream)?;
    Ok(())
}

fn submit_q8_1_all(
    stream: &OwnedCudaStream,
    normalized_quantize: &CudaQ8_1ActivationQuantizeArgsV1,
    gate: &CudaNvfp4Sm70Q8_1MatvecArgsV1,
    up: &CudaNvfp4Sm70Q8_1MatvecArgsV1,
    silu: &CudaSiluMulF32ArgsV1,
    down_quantize: &CudaQ8_1ActivationQuantizeArgsV1,
    down: &CudaNvfp4Sm70Q8_1MatvecArgsV1,
) -> Result<(), Qwen35RealNvfp4CudaError> {
    normalized_quantize.submit_for_plan(stream)?;
    gate.submit_for_plan(stream)?;
    up.submit_for_plan(stream)?;
    silu.submit_for_plan(stream)?;
    down_quantize.submit_for_plan(stream)?;
    down.submit_for_plan(stream)?;
    Ok(())
}

fn q8_1_workspace_bytes(elements: usize) -> Result<usize, Qwen35RealNvfp4CudaError> {
    if elements == 0 || !elements.is_multiple_of(Q8_1_ACTIVATION_BLOCK_ELEMENTS) {
        return Err(Qwen35RealNvfp4CudaError::InvalidConfiguration(
            "Q8_1 activation elements",
        ));
    }
    elements
        .checked_div(Q8_1_ACTIVATION_BLOCK_ELEMENTS)
        .and_then(|blocks| blocks.checked_mul(Q8_1_ACTIVATION_BLOCK_BYTES))
        .ok_or(Qwen35RealNvfp4CudaError::InvalidConfiguration(
            "Q8_1 activation workspace bytes",
        ))
}

fn compare_q8_1_activation_payload(
    name: &str,
    elements: usize,
    q8_1_payload: &[u8],
    expected_q8_0: &[u8],
) -> Result<Nvfp4CudaQ8_1CompatibilityReport, Qwen35RealNvfp4CudaError> {
    let expected_q8_1_bytes = q8_1_workspace_bytes(elements)?;
    let block_count = elements / Q8_1_ACTIVATION_BLOCK_ELEMENTS;
    let expected_q8_0_bytes = block_count
        .checked_mul(Q8_1_ACTIVATION_BLOCK_BYTES - core::mem::size_of::<u16>())
        .ok_or(Qwen35RealNvfp4CudaError::InvalidConfiguration(
            "Q8_0 compatibility bytes",
        ))?;
    if q8_1_payload.len() != expected_q8_1_bytes || expected_q8_0.len() != expected_q8_0_bytes {
        return Err(Qwen35RealNvfp4CudaError::InvalidConfiguration(
            "Q8_1 compatibility payload extent",
        ));
    }

    let mut projected_q8_0 = Vec::with_capacity(expected_q8_0_bytes);
    for block in q8_1_payload.chunks_exact(Q8_1_ACTIVATION_BLOCK_BYTES) {
        projected_q8_0.extend_from_slice(&block[..core::mem::size_of::<u16>()]);
        projected_q8_0.extend_from_slice(&block[2 * core::mem::size_of::<u16>()..]);
    }
    Ok(Nvfp4CudaQ8_1CompatibilityReport {
        name: name.to_owned(),
        elements,
        q8_1_payload_bytes: q8_1_payload.len(),
        q8_0_projection_bytes: projected_q8_0.len(),
        metadata_sum_bytes_ignored: block_count * core::mem::size_of::<u16>(),
        q8_0_projection_sha256: sha256_hex(&projected_q8_0),
        expected_q8_0_sha256: sha256_hex(expected_q8_0),
        exact_scale_and_code_bytes: projected_q8_0 == expected_q8_0,
    })
}

fn poll_completion(
    stream: &OwnedCudaStream,
    event: &OwnedCudaEvent,
    receipt: augur_backend::CudaStreamSubmissionReceipt,
) -> Result<u64, Qwen35RealNvfp4CudaError> {
    let mut polls = 0_u64;
    loop {
        if stream.query_event_completion(event, receipt)?.is_some() {
            return Ok(polls);
        }
        polls = polls.saturating_add(1);
        if polls > MAX_COMPLETION_POLLS {
            return Err(Qwen35RealNvfp4CudaError::CompletionPollLimit);
        }
        spin_loop();
    }
}

fn projection_shape(
    name: &str,
    tensor: &GgufTensor,
    payload: &[u8],
) -> Result<Nvfp4CudaProjectionShape, Qwen35RealNvfp4CudaError> {
    let columns = usize::try_from(tensor.dimensions[0])
        .map_err(|_| Qwen35RealNvfp4CudaError::InvalidConfiguration("NVFP4 columns"))?;
    let rows = usize::try_from(tensor.dimensions[1])
        .map_err(|_| Qwen35RealNvfp4CudaError::InvalidConfiguration("NVFP4 rows"))?;
    let blocks_per_row = columns / NVFP4_BLOCK_ELEMENTS;
    let row_bytes = blocks_per_row.checked_mul(NVFP4_BLOCK_BYTES).ok_or(
        Qwen35RealNvfp4CudaError::InvalidConfiguration("NVFP4 row bytes"),
    )?;
    Ok(Nvfp4CudaProjectionShape {
        name: name.to_owned(),
        rows,
        columns,
        blocks_per_row,
        row_bytes,
        payload_bytes: payload.len(),
    })
}

fn tensor_report(
    tensor: &GgufTensor,
    payload: &[u8],
    tensor_level_scale: Option<f32>,
) -> Nvfp4CudaTensorReport {
    Nvfp4CudaTensorReport {
        name: tensor.name.clone(),
        dimensions: tensor.dimensions.clone(),
        ggml_type: tensor.ggml_type,
        descriptor_payload_bytes: tensor.byte_len,
        payload_bytes_read: payload.len(),
        payload_sha256: sha256_hex(payload),
        tensor_level_scale,
    }
}

fn compare_output(
    name: &str,
    actual: &[f32],
    expected: &[f32],
    output_bytes: &[u8],
) -> Nvfp4CudaNumericalReport {
    compare_output_with_tolerances(name, actual, expected, output_bytes, 3.0e-3, 2.0e-4)
}

fn compare_output_with_tolerances(
    name: &str,
    actual: &[f32],
    expected: &[f32],
    output_bytes: &[u8],
    absolute_tolerance: f32,
    relative_tolerance: f32,
) -> Nvfp4CudaNumericalReport {
    let mut max_abs_diff = 0.0_f32;
    let mut max_relative_diff = 0.0_f32;
    let mut pass = actual.len() == expected.len();
    for (actual_value, expected_value) in actual.iter().zip(expected) {
        if !actual_value.is_finite() || !expected_value.is_finite() {
            pass = false;
            continue;
        }
        let absolute = (*actual_value - *expected_value).abs();
        let relative = absolute / expected_value.abs().max(1.0e-12);
        max_abs_diff = max_abs_diff.max(absolute);
        max_relative_diff = max_relative_diff.max(relative);
        if absolute > absolute_tolerance + relative_tolerance * expected_value.abs() {
            pass = false;
        }
    }
    Nvfp4CudaNumericalReport {
        name: name.to_owned(),
        rows: actual.len(),
        finite_count: actual.iter().filter(|value| value.is_finite()).count(),
        expected_count: expected.len(),
        max_abs_diff,
        max_relative_diff,
        absolute_tolerance,
        relative_tolerance,
        output_sha256: sha256_hex(output_bytes),
        expected_sha256: sha256_f32_bits(expected),
        pass,
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * core::mem::size_of::<f32>());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn f32_vec_from_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(core::mem::size_of::<f32>())
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("f32 chunk")))
        .collect()
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

fn sha256_f32_bits(values: &[f32]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn delta_allocations(
    before: augur_backend::CudaDeviceAllocationSnapshot,
    after: augur_backend::CudaDeviceAllocationSnapshot,
) -> Result<AllocationDeltaReport, Qwen35RealNvfp4CudaError> {
    Ok(AllocationDeltaReport {
        successful_allocations: after
            .successful_allocations
            .checked_sub(before.successful_allocations)
            .ok_or(Qwen35RealNvfp4CudaError::InvalidConfiguration(
                "allocation counters",
            ))?,
        successful_frees: after
            .successful_frees
            .checked_sub(before.successful_frees)
            .ok_or(Qwen35RealNvfp4CudaError::InvalidConfiguration(
                "allocation counters",
            ))?,
        live_allocations_after: after.live_allocations,
        live_bytes_after: after.live_bytes,
        peak_live_allocations_observed: after.peak_live_allocations,
        peak_live_bytes_observed: after.peak_live_bytes,
    })
}

fn delta_boundaries(
    before: augur_backend::CudaHostBoundarySnapshot,
    after: augur_backend::CudaHostBoundarySnapshot,
) -> Result<HostBoundaryDeltaReport, Qwen35RealNvfp4CudaError> {
    let delta =
        after
            .checked_delta_since(before)
            .ok_or(Qwen35RealNvfp4CudaError::InvalidConfiguration(
                "host counters",
            ))?;
    Ok(HostBoundaryDeltaReport {
        host_waits: delta.host_waits,
        stream_synchronizations: delta.stream_synchronizations,
        event_synchronizations: delta.event_synchronizations,
        synchronous_h2d_copies: delta.synchronous_h2d_copies,
        synchronous_h2d_bytes: delta.synchronous_h2d_bytes,
        synchronous_d2h_copies: delta.synchronous_d2h_copies,
        synchronous_d2h_bytes: delta.synchronous_d2h_bytes,
    })
}
