#![cfg(feature = "cuda")]

//! Real-weight, pre-recurrent layer-0 projection evidence for the Qwen lane.
//!
//! The runner authenticates one Qwen GGUF and executes the four projections
//! that consume the layer-0 normalized embedding on SM70: `attn_qkv`,
//! `attn_gate`, `ssm_alpha`, and `ssm_beta`.  The output is compared with the
//! checked CPU quantized reference.  It intentionally stops before the
//! recurrent state update, FFN, output head, decode loop, and token
//! publication, so this is a component boundary rather than an inference
//! runner.

use std::{error::Error, fmt, hint::spin_loop, path::PathBuf, time::Instant};

use augur_backend::{
    cuda_device_allocation_snapshot, cuda_device_properties, cuda_host_boundary_snapshot,
    CudaBf16F32MatvecArgsV1, CudaBf16WeightViewV1, CudaDeviceAllocationSnapshot, CudaF32View,
    CudaHostBoundarySnapshot, CudaQ8_0F32MatvecArgsV1, CudaQ8_0WeightViewV1, CudaRuntimeError,
    OwnedCudaBuffer, OwnedCudaEvent, OwnedCudaStream,
};
use augur_core::PhysicalLayoutId;
use augur_model_qwen35::{
    dequantize_bf16, dequantize_tensor, quantized_matvec_fused_simd, rms_norm_last_dim,
    GgufAuditOptions, GgufManifest, GgufTensor, Qwen35Config, Qwen35Error, ReferenceTensor,
    GGML_TYPE_BF16, GGML_TYPE_F32, GGML_TYPE_Q8_0,
};
use serde::Serialize;
use sha2::{Digest, Sha256};

pub const QWEN35_REAL_LAYER0_PROJECTION_CUDA_SCHEMA: &str =
    "crystal-quant.qwen35.real-layer0-projection-cuda.v1";

const HIDDEN: usize = 5_120;
const VOCAB: usize = 248_320;
const QKV_ROWS: usize = 10_240;
const GATE_ROWS: usize = 6_144;
const VALUE_HEADS: usize = 48;
const MAX_COMPLETION_POLLS: u64 = 2_000_000;

const ATTN_NORM: &str = "blk.0.attn_norm.weight";
const QKV: &str = "blk.0.attn_qkv.weight";
const GATE: &str = "blk.0.attn_gate.weight";
const ALPHA: &str = "blk.0.ssm_alpha.weight";
const BETA: &str = "blk.0.ssm_beta.weight";
const EMBEDDING: &str = "token_embd.weight";

/// Configuration for the real layer-0 projection boundary.
#[derive(Clone, Debug)]
pub struct Qwen35RealLayer0ProjectionCudaConfig {
    pub gguf_path: PathBuf,
    pub config_path: PathBuf,
    pub token_id: usize,
    pub device_ordinal: i32,
    pub warmup: usize,
    pub iterations: usize,
    pub source_commit: String,
}

impl Default for Qwen35RealLayer0ProjectionCudaConfig {
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
pub enum Qwen35RealLayer0ProjectionCudaError {
    Runtime(CudaRuntimeError),
    Model(Qwen35Error),
    InvalidConfiguration(&'static str),
    CompletionPollLimit,
}

impl fmt::Display for Qwen35RealLayer0ProjectionCudaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Runtime(error) => error.fmt(formatter),
            Self::Model(error) => error.fmt(formatter),
            Self::InvalidConfiguration(field) => {
                write!(
                    formatter,
                    "invalid Qwen layer-0 CUDA projection configuration: {field}"
                )
            }
            Self::CompletionPollLimit => {
                formatter.write_str("CUDA completion polling exceeded its bound")
            }
        }
    }
}

impl Error for Qwen35RealLayer0ProjectionCudaError {}

impl From<CudaRuntimeError> for Qwen35RealLayer0ProjectionCudaError {
    fn from(error: CudaRuntimeError) -> Self {
        Self::Runtime(error)
    }
}

impl From<Qwen35Error> for Qwen35RealLayer0ProjectionCudaError {
    fn from(error: Qwen35Error) -> Self {
        Self::Model(error)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Qwen35RealLayer0ProjectionCudaReport {
    pub schema: String,
    pub status: String,
    pub component_only: bool,
    pub source_commit: String,
    pub device: ProjectionCudaDeviceReport,
    pub config: ProjectionCudaConfigReport,
    pub source: ProjectionCudaSourceReport,
    pub input: ProjectionCudaInputReport,
    pub timing: ProjectionCudaTimingReport,
    pub completion: ProjectionCudaCompletionReport,
    pub numerical: Vec<ProjectionCudaNumericalReport>,
    pub resources: ProjectionCudaResourceReport,
    pub claims: ProjectionCudaClaimReport,
}

#[derive(Clone, Debug, Serialize)]
pub struct ProjectionCudaDeviceReport {
    pub ordinal: i32,
    pub name: String,
    pub compute_capability: [u16; 2],
    pub total_global_memory_bytes: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ProjectionCudaConfigReport {
    pub path: String,
    pub sha256: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct ProjectionCudaTensorReport {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub ggml_type: u32,
    pub descriptor_payload_bytes: u64,
    pub read_bytes: usize,
    pub read_sha256: String,
    pub read_offset_bytes: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ProjectionCudaSourceReport {
    pub path: String,
    pub file_sha256: String,
    pub tensors: Vec<ProjectionCudaTensorReport>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ProjectionCudaInputReport {
    pub token_id: usize,
    pub embedding_row_bytes: usize,
    pub normalized_elements: usize,
    pub normalized_input_sha256: String,
    pub norm_epsilon: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct ProjectionCudaTimingReport {
    pub warmup_passes: usize,
    pub steady_passes: usize,
    pub cuda_event_ms: f32,
    pub batch_wall_ms: f64,
    pub per_pass_event_ms: f64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ProjectionCudaCompletionReport {
    pub poll_iterations: u64,
    pub nonblocking_poll_only: bool,
    pub completion_proof_required: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct ProjectionCudaNumericalReport {
    pub name: String,
    pub rows: usize,
    pub columns: usize,
    pub finite_count: usize,
    pub expected_count: usize,
    pub max_abs_diff: f32,
    pub absolute_tolerance: f32,
    pub relative_tolerance: f32,
    pub output_sha256: String,
    pub expected_sha256: String,
    pub pass: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct ProjectionCudaResourceReport {
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
pub struct ProjectionCudaClaimReport {
    pub real_weight_projection_boundary_executed: bool,
    pub complete_transformer_executed: bool,
    pub decode_executed: bool,
    pub semantic_parity_established: bool,
    pub product_tps_permitted: bool,
    pub default_cli_enabled: bool,
}

/// Execute the authenticated layer-0 pre-recurrent projection boundary.
pub fn run_qwen35_real_layer0_projection_cuda(
    config: &Qwen35RealLayer0ProjectionCudaConfig,
) -> Result<Qwen35RealLayer0ProjectionCudaReport, Qwen35RealLayer0ProjectionCudaError> {
    validate_config(config)?;
    let config_bytes = std::fs::read(&config.config_path).map_err(Qwen35Error::from)?;
    let config_sha256 = sha256_hex(&config_bytes);
    let model_config = Qwen35Config::parse(&config_bytes)?;
    if model_config.hidden_size != HIDDEN
        || model_config.vocab_size != VOCAB
        || model_config.num_hidden_layers != 64
    {
        return Err(Qwen35RealLayer0ProjectionCudaError::InvalidConfiguration(
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
    let qkv = require_tensor(
        &manifest,
        QKV,
        &[HIDDEN as u64, QKV_ROWS as u64],
        GGML_TYPE_Q8_0,
    )?;
    let gate = require_tensor(
        &manifest,
        GATE,
        &[HIDDEN as u64, GATE_ROWS as u64],
        GGML_TYPE_Q8_0,
    )?;
    let alpha = require_tensor(
        &manifest,
        ALPHA,
        &[HIDDEN as u64, VALUE_HEADS as u64],
        GGML_TYPE_BF16,
    )?;
    let beta = require_tensor(
        &manifest,
        BETA,
        &[HIDDEN as u64, VALUE_HEADS as u64],
        GGML_TYPE_BF16,
    )?;

    let row_bytes =
        HIDDEN
            .checked_mul(2)
            .ok_or(Qwen35RealLayer0ProjectionCudaError::InvalidConfiguration(
                "embedding row bytes",
            ))?;
    let embedding_offset = config.token_id.checked_mul(row_bytes).ok_or(
        Qwen35RealLayer0ProjectionCudaError::InvalidConfiguration("embedding offset"),
    )?;
    let embedding_offset_u64 = u64::try_from(embedding_offset).map_err(|_| {
        Qwen35RealLayer0ProjectionCudaError::InvalidConfiguration("embedding offset")
    })?;
    let mut reader = manifest.open_tensor_reader(&config.gguf_path)?;
    let embedding_bytes = reader.read_tensor_slice(
        EMBEDDING,
        embedding_offset_u64,
        u64::try_from(row_bytes).map_err(|_| {
            Qwen35RealLayer0ProjectionCudaError::InvalidConfiguration("embedding row bytes")
        })?,
    )?;
    let mut embedding_values = vec![0.0_f32; HIDDEN];
    dequantize_bf16(&embedding_bytes, &mut embedding_values)?;
    let norm_bytes = reader.read_tensor_bytes(ATTN_NORM)?;
    let qkv_payload = reader.read_tensor_bytes(QKV)?;
    let gate_payload = reader.read_tensor_bytes(GATE)?;
    let alpha_payload = reader.read_tensor_bytes(ALPHA)?;
    let beta_payload = reader.read_tensor_bytes(BETA)?;
    let norm_values = dequantize_tensor(&attn_norm, &norm_bytes)?;
    let normalized = rms_norm_last_dim(
        &ReferenceTensor::new(vec![1, HIDDEN], embedding_values)?,
        &ReferenceTensor::new(vec![HIDDEN], norm_values)?,
        model_config.rms_norm_eps as f32,
    )?
    .into_values();
    if normalized.iter().any(|value| !value.is_finite()) {
        return Err(Qwen35RealLayer0ProjectionCudaError::InvalidConfiguration(
            "normalized embedding",
        ));
    }

    let qkv_expected = quantized_matvec_fused_simd(&qkv, &qkv_payload, &normalized)?;
    let gate_expected = quantized_matvec_fused_simd(&gate, &gate_payload, &normalized)?;
    let alpha_expected = quantized_matvec_fused_simd(&alpha, &alpha_payload, &normalized)?;
    let beta_expected = quantized_matvec_fused_simd(&beta, &beta_payload, &normalized)?;

    let device = cuda_device_properties(config.device_ordinal)?;
    let stream = OwnedCudaStream::create(config.device_ordinal)?;
    let allocation_start = cuda_device_allocation_snapshot();
    let boundary_start = cuda_host_boundary_snapshot();

    let input_bytes = f32_bytes(&normalized);
    let qkv_weight = OwnedCudaBuffer::allocate(config.device_ordinal, qkv_payload.len())?;
    let gate_weight = OwnedCudaBuffer::allocate(config.device_ordinal, gate_payload.len())?;
    let alpha_weight = OwnedCudaBuffer::allocate(config.device_ordinal, alpha_payload.len())?;
    let beta_weight = OwnedCudaBuffer::allocate(config.device_ordinal, beta_payload.len())?;
    let input = OwnedCudaBuffer::allocate(config.device_ordinal, input_bytes.len())?;
    let qkv_output = OwnedCudaBuffer::allocate(
        config.device_ordinal,
        QKV_ROWS * core::mem::size_of::<f32>(),
    )?;
    let gate_output = OwnedCudaBuffer::allocate(
        config.device_ordinal,
        GATE_ROWS * core::mem::size_of::<f32>(),
    )?;
    let alpha_output = OwnedCudaBuffer::allocate(
        config.device_ordinal,
        VALUE_HEADS * core::mem::size_of::<f32>(),
    )?;
    let beta_output = OwnedCudaBuffer::allocate(
        config.device_ordinal,
        VALUE_HEADS * core::mem::size_of::<f32>(),
    )?;

    qkv_weight.copy_from_host_sync(&stream, &qkv_payload)?;
    gate_weight.copy_from_host_sync(&stream, &gate_payload)?;
    alpha_weight.copy_from_host_sync(&stream, &alpha_payload)?;
    beta_weight.copy_from_host_sync(&stream, &beta_payload)?;
    input.copy_from_host_sync(&stream, &input_bytes)?;

    let input_view = CudaF32View::new(
        input.view(0, input_bytes.len())?,
        PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
        HIDDEN,
    )?;
    let qkv_args = bind_q8_args(
        &stream,
        &qkv_weight,
        &qkv_output,
        QKV_ROWS,
        HIDDEN,
        input_view,
    )?;
    let gate_args = bind_q8_args(
        &stream,
        &gate_weight,
        &gate_output,
        GATE_ROWS,
        HIDDEN,
        input_view,
    )?;
    let alpha_args = bind_bf16_args(
        &stream,
        &alpha_weight,
        &alpha_output,
        VALUE_HEADS,
        HIDDEN,
        input_view,
    )?;
    let beta_args = bind_bf16_args(
        &stream,
        &beta_weight,
        &beta_output,
        VALUE_HEADS,
        HIDDEN,
        input_view,
    )?;

    let setup_allocations = delta_allocations(allocation_start, cuda_device_allocation_snapshot())?;
    let setup_host_boundaries = delta_boundaries(boundary_start, cuda_host_boundary_snapshot())?;

    for _ in 0..config.warmup {
        submit_all(&stream, &qkv_args, &gate_args, &alpha_args, &beta_args)?;
    }
    stream.synchronize()?;

    let steady_allocation_start = cuda_device_allocation_snapshot();
    let steady_boundary_start = cuda_host_boundary_snapshot();
    let started = OwnedCudaEvent::create(config.device_ordinal)?;
    let stopped = OwnedCudaEvent::create(config.device_ordinal)?;
    let wall_start = Instant::now();
    started.record(&stream)?;
    for _ in 0..config.iterations {
        submit_all(&stream, &qkv_args, &gate_args, &alpha_args, &beta_args)?;
    }
    stopped.record(&stream)?;
    stopped.synchronize()?;
    let batch_wall_ms = wall_start.elapsed().as_secs_f64() * 1_000.0;
    let cuda_event_ms = started.elapsed_ms(&stopped)?;
    let steady_allocations =
        delta_allocations(steady_allocation_start, cuda_device_allocation_snapshot())?;
    let steady_host_boundaries =
        delta_boundaries(steady_boundary_start, cuda_host_boundary_snapshot())?;

    let completion_event = OwnedCudaEvent::create(config.device_ordinal)?;
    let completion_boundary_start = cuda_host_boundary_snapshot();
    submit_all(&stream, &qkv_args, &gate_args, &alpha_args, &beta_args)?;
    let receipt = stream.submission_receipt()?;
    completion_event.record_for_submission(&stream, receipt)?;
    let (poll_iterations, completion_proof) = poll_completion(&stream, &completion_event, receipt)?;
    if !completion_proof.covers_submission(receipt)
        || !completion_proof.belongs_to_stream(&stream)?
    {
        return Err(Qwen35RealLayer0ProjectionCudaError::InvalidConfiguration(
            "completion proof binding",
        ));
    }
    let completion_poll_host_boundaries =
        delta_boundaries(completion_boundary_start, cuda_host_boundary_snapshot())?;

    let readback_boundary_start = cuda_host_boundary_snapshot();
    let mut qkv_output_bytes = vec![0_u8; QKV_ROWS * core::mem::size_of::<f32>()];
    let mut gate_output_bytes = vec![0_u8; GATE_ROWS * core::mem::size_of::<f32>()];
    let mut alpha_output_bytes = vec![0_u8; VALUE_HEADS * core::mem::size_of::<f32>()];
    let mut beta_output_bytes = vec![0_u8; VALUE_HEADS * core::mem::size_of::<f32>()];
    qkv_output.copy_to_host_sync_with_submission_proof(&stream, &mut qkv_output_bytes, receipt)?;
    gate_output.copy_to_host_sync_with_submission_proof(
        &stream,
        &mut gate_output_bytes,
        receipt,
    )?;
    alpha_output.copy_to_host_sync_with_submission_proof(
        &stream,
        &mut alpha_output_bytes,
        receipt,
    )?;
    beta_output.copy_to_host_sync_with_submission_proof(
        &stream,
        &mut beta_output_bytes,
        receipt,
    )?;
    let readback_host_boundaries =
        delta_boundaries(readback_boundary_start, cuda_host_boundary_snapshot())?;
    let qkv_actual = f32_vec_from_bytes(&qkv_output_bytes);
    let gate_actual = f32_vec_from_bytes(&gate_output_bytes);
    let alpha_actual = f32_vec_from_bytes(&alpha_output_bytes);
    let beta_actual = f32_vec_from_bytes(&beta_output_bytes);
    let numerical = vec![
        compare_projection(
            QKV,
            QKV_ROWS,
            HIDDEN,
            &qkv_actual,
            &qkv_expected,
            5e-3,
            2e-4,
            &qkv_output_bytes,
        ),
        compare_projection(
            GATE,
            GATE_ROWS,
            HIDDEN,
            &gate_actual,
            &gate_expected,
            5e-3,
            2e-4,
            &gate_output_bytes,
        ),
        compare_projection(
            ALPHA,
            VALUE_HEADS,
            HIDDEN,
            &alpha_actual,
            &alpha_expected,
            2e-3,
            2e-4,
            &alpha_output_bytes,
        ),
        compare_projection(
            BETA,
            VALUE_HEADS,
            HIDDEN,
            &beta_actual,
            &beta_expected,
            2e-3,
            2e-4,
            &beta_output_bytes,
        ),
    ];
    let pass = numerical.iter().all(|result| result.pass);

    drop(completion_event);
    drop(stopped);
    drop(started);
    drop(beta_output);
    drop(alpha_output);
    drop(gate_output);
    drop(qkv_output);
    drop(input);
    drop(beta_weight);
    drop(alpha_weight);
    drop(gate_weight);
    drop(qkv_weight);
    drop(stream);
    let final_allocation_delta_after_cleanup =
        delta_allocations(allocation_start, cuda_device_allocation_snapshot())?;

    let tensor_reports = [
        (&embedding, &embedding_bytes, embedding_offset_u64),
        (&attn_norm, &norm_bytes, 0_u64),
        (&qkv, &qkv_payload, 0_u64),
        (&gate, &gate_payload, 0_u64),
        (&alpha, &alpha_payload, 0_u64),
        (&beta, &beta_payload, 0_u64),
    ]
    .into_iter()
    .map(
        |(descriptor, payload, read_offset_bytes)| ProjectionCudaTensorReport {
            name: descriptor.name.clone(),
            dimensions: descriptor.dimensions.clone(),
            ggml_type: descriptor.ggml_type,
            descriptor_payload_bytes: descriptor.byte_len,
            read_bytes: payload.len(),
            read_sha256: sha256_hex(payload),
            read_offset_bytes,
        },
    )
    .collect();

    Ok(Qwen35RealLayer0ProjectionCudaReport {
        schema: QWEN35_REAL_LAYER0_PROJECTION_CUDA_SCHEMA.to_owned(),
        status: if pass {
            "PASS_COMPONENT_REAL_LAYER0_PRE_RECURRENT_PROJECTION_CUDA".to_owned()
        } else {
            "FAIL_COMPONENT_REAL_LAYER0_PRE_RECURRENT_PROJECTION_CUDA_NUMERICAL".to_owned()
        },
        component_only: true,
        source_commit: config.source_commit.clone(),
        device: ProjectionCudaDeviceReport {
            ordinal: device.ordinal,
            name: device.name,
            compute_capability: [
                device.compute_capability.major,
                device.compute_capability.minor,
            ],
            total_global_memory_bytes: device.total_global_memory_bytes,
        },
        config: ProjectionCudaConfigReport {
            path: config.config_path.display().to_string(),
            sha256: config_sha256,
        },
        source: ProjectionCudaSourceReport {
            path: config.gguf_path.display().to_string(),
            file_sha256: manifest.file_sha256.clone().ok_or(
                Qwen35RealLayer0ProjectionCudaError::InvalidConfiguration("GGUF file hash"),
            )?,
            tensors: tensor_reports,
        },
        input: ProjectionCudaInputReport {
            token_id: config.token_id,
            embedding_row_bytes: embedding_bytes.len(),
            normalized_elements: normalized.len(),
            normalized_input_sha256: sha256_f32_bits(&normalized),
            norm_epsilon: model_config.rms_norm_eps as f32,
        },
        timing: ProjectionCudaTimingReport {
            warmup_passes: config.warmup,
            steady_passes: config.iterations,
            cuda_event_ms,
            batch_wall_ms,
            per_pass_event_ms: f64::from(cuda_event_ms) / config.iterations as f64,
        },
        completion: ProjectionCudaCompletionReport {
            poll_iterations,
            nonblocking_poll_only: true,
            completion_proof_required: true,
        },
        numerical,
        resources: ProjectionCudaResourceReport {
            setup_allocations,
            steady_allocations,
            completion_poll_host_boundaries,
            setup_host_boundaries,
            steady_host_boundaries,
            readback_host_boundaries,
            final_allocation_delta_after_cleanup,
        },
        claims: ProjectionCudaClaimReport {
            real_weight_projection_boundary_executed: true,
            complete_transformer_executed: false,
            decode_executed: false,
            semantic_parity_established: false,
            product_tps_permitted: false,
            default_cli_enabled: false,
        },
    })
}

fn validate_config(
    config: &Qwen35RealLayer0ProjectionCudaConfig,
) -> Result<(), Qwen35RealLayer0ProjectionCudaError> {
    if config.gguf_path.as_os_str().is_empty() {
        return Err(Qwen35RealLayer0ProjectionCudaError::InvalidConfiguration(
            "gguf_path",
        ));
    }
    if config.config_path.as_os_str().is_empty() {
        return Err(Qwen35RealLayer0ProjectionCudaError::InvalidConfiguration(
            "config_path",
        ));
    }
    if config.device_ordinal < 0 {
        return Err(Qwen35RealLayer0ProjectionCudaError::InvalidConfiguration(
            "device_ordinal",
        ));
    }
    if config.token_id >= VOCAB {
        return Err(Qwen35RealLayer0ProjectionCudaError::InvalidConfiguration(
            "token_id",
        ));
    }
    if config.iterations == 0 {
        return Err(Qwen35RealLayer0ProjectionCudaError::InvalidConfiguration(
            "iterations",
        ));
    }
    Ok(())
}

fn require_tensor(
    manifest: &GgufManifest,
    name: &str,
    dimensions: &[u64],
    ggml_type: u32,
) -> Result<GgufTensor, Qwen35RealLayer0ProjectionCudaError> {
    let tensor = manifest
        .tensor(name)
        .ok_or(Qwen35RealLayer0ProjectionCudaError::InvalidConfiguration(
            "required tensor missing",
        ))?
        .clone();
    if tensor.dimensions != dimensions || tensor.ggml_type != ggml_type {
        return Err(Qwen35RealLayer0ProjectionCudaError::InvalidConfiguration(
            "required tensor geometry or type",
        ));
    }
    Ok(tensor)
}

fn bind_q8_args(
    stream: &OwnedCudaStream,
    weight: &OwnedCudaBuffer,
    output: &OwnedCudaBuffer,
    rows: usize,
    columns: usize,
    input_view: CudaF32View,
) -> Result<CudaQ8_0F32MatvecArgsV1, Qwen35RealLayer0ProjectionCudaError> {
    Ok(CudaQ8_0F32MatvecArgsV1::bind(
        stream,
        CudaQ8_0WeightViewV1::new(
            weight.view(0, weight.len_bytes())?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_0_CUDA_V1,
            rows,
            columns,
        )?,
        input_view,
        CudaF32View::new(
            output.view(0, rows * core::mem::size_of::<f32>())?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            rows,
        )?,
    )?)
}

fn bind_bf16_args(
    stream: &OwnedCudaStream,
    weight: &OwnedCudaBuffer,
    output: &OwnedCudaBuffer,
    rows: usize,
    columns: usize,
    input_view: CudaF32View,
) -> Result<CudaBf16F32MatvecArgsV1, Qwen35RealLayer0ProjectionCudaError> {
    Ok(CudaBf16F32MatvecArgsV1::bind(
        stream,
        CudaBf16WeightViewV1::new(
            weight.view(0, weight.len_bytes())?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_BF16_CUDA_V1,
            rows,
            columns,
        )?,
        input_view,
        CudaF32View::new(
            output.view(0, rows * core::mem::size_of::<f32>())?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            rows,
        )?,
    )?)
}

fn submit_all(
    stream: &OwnedCudaStream,
    qkv: &CudaQ8_0F32MatvecArgsV1,
    gate: &CudaQ8_0F32MatvecArgsV1,
    alpha: &CudaBf16F32MatvecArgsV1,
    beta: &CudaBf16F32MatvecArgsV1,
) -> Result<(), Qwen35RealLayer0ProjectionCudaError> {
    qkv.submit_for_plan(stream)?;
    gate.submit_for_plan(stream)?;
    alpha.submit_for_plan(stream)?;
    beta.submit_for_plan(stream)?;
    Ok(())
}

fn poll_completion(
    stream: &OwnedCudaStream,
    event: &OwnedCudaEvent,
    receipt: augur_backend::CudaStreamSubmissionReceipt,
) -> Result<(u64, augur_backend::CudaStreamCompletionProof), Qwen35RealLayer0ProjectionCudaError> {
    let mut polls = 0_u64;
    loop {
        if let Some(proof) = stream.query_event_completion(event, receipt)? {
            return Ok((polls, proof));
        }
        polls = polls.saturating_add(1);
        if polls > MAX_COMPLETION_POLLS {
            return Err(Qwen35RealLayer0ProjectionCudaError::CompletionPollLimit);
        }
        spin_loop();
    }
}

fn compare_projection(
    name: &str,
    rows: usize,
    columns: usize,
    actual: &[f32],
    expected: &[f32],
    absolute_tolerance: f32,
    relative_tolerance: f32,
    actual_bytes: &[u8],
) -> ProjectionCudaNumericalReport {
    let max_abs_diff = actual
        .iter()
        .zip(expected)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max);
    let pass = actual.len() == expected.len()
        && actual.iter().zip(expected).all(|(left, right)| {
            left.is_finite()
                && (*left - *right).abs() <= absolute_tolerance + relative_tolerance * right.abs()
        });
    ProjectionCudaNumericalReport {
        name: name.to_owned(),
        rows,
        columns,
        finite_count: actual.iter().filter(|value| value.is_finite()).count(),
        expected_count: expected.len(),
        max_abs_diff,
        absolute_tolerance,
        relative_tolerance,
        output_sha256: sha256_hex(actual_bytes),
        expected_sha256: sha256_f32_bits(expected),
        pass,
    }
}

fn delta_allocations(
    before: CudaDeviceAllocationSnapshot,
    after: CudaDeviceAllocationSnapshot,
) -> Result<AllocationDeltaReport, Qwen35RealLayer0ProjectionCudaError> {
    Ok(AllocationDeltaReport {
        successful_allocations: after
            .successful_allocations
            .checked_sub(before.successful_allocations)
            .ok_or(Qwen35RealLayer0ProjectionCudaError::InvalidConfiguration(
                "allocation counters",
            ))?,
        successful_frees: after
            .successful_frees
            .checked_sub(before.successful_frees)
            .ok_or(Qwen35RealLayer0ProjectionCudaError::InvalidConfiguration(
                "allocation counters",
            ))?,
        live_allocations_after: after.live_allocations,
        live_bytes_after: after.live_bytes,
        peak_live_allocations_observed: after.peak_live_allocations,
        peak_live_bytes_observed: after.peak_live_bytes,
    })
}

fn delta_boundaries(
    before: CudaHostBoundarySnapshot,
    after: CudaHostBoundarySnapshot,
) -> Result<HostBoundaryDeltaReport, Qwen35RealLayer0ProjectionCudaError> {
    let delta = after.checked_delta_since(before).ok_or(
        Qwen35RealLayer0ProjectionCudaError::InvalidConfiguration("host counters"),
    )?;
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

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
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

#[cfg(test)]
mod tests {
    use super::{compare_projection, validate_config, Qwen35RealLayer0ProjectionCudaConfig};

    #[test]
    fn configuration_requires_both_authenticated_paths() {
        let config = Qwen35RealLayer0ProjectionCudaConfig::default();
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn comparison_applies_relative_tolerance_without_accepting_nonfinite_values() {
        let actual = [100.001_f32, f32::INFINITY];
        let expected = [100.0_f32, 0.0];
        let report = compare_projection("fixture", 2, 1, &actual, &expected, 1e-3, 1e-4, &[0; 8]);
        assert!(!report.pass);
        assert_eq!(report.finite_count, 1);
    }
}
