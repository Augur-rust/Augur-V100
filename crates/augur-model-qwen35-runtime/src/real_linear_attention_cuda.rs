#![cfg(feature = "qwen35-linear-attention")]

//! Real-weight Qwen3.8 layer-0 linear-attention output boundary.
//!
//! This opt-in component extends the separately audited recurrent-step path by
//! placing the real `ssm_out` Q8_0 projection immediately after the recurrent
//! kernels on the same CUDA stream.  It intentionally stops at the attention
//! output: post-attention residual/norm, FFN, later layers, token publication,
//! and product/decode paths remain outside this boundary.

use std::{error::Error, fmt, hint::spin_loop, path::PathBuf, time::Instant};

use augur_backend::{
    cuda_device_allocation_snapshot, cuda_device_properties, cuda_host_boundary_snapshot,
    CudaF32View, CudaQ8_0F32MatvecArgsV1, CudaQ8_0WeightViewV1, CudaQwen35LinearStepArgsV1,
    CudaRuntimeError, OwnedCudaBuffer, OwnedCudaEvent, OwnedCudaStream,
};
use augur_core::PhysicalLayoutId;
use augur_model_qwen35::{
    dequantize_bf16, dequantize_tensor, quantized_matvec_fused_simd, rms_norm_last_dim,
    GgufAuditOptions, GgufManifest, GgufTensor, Qwen35Config, Qwen35Error,
    Qwen35LinearAttentionBlock, Qwen35LinearAttentionInputs, ReferenceTensor, GGML_TYPE_BF16,
    GGML_TYPE_F32, GGML_TYPE_Q8_0,
};
use serde::Serialize;
use sha2::{Digest, Sha256};

pub const QWEN35_REAL_LINEAR_ATTENTION_CUDA_SCHEMA: &str =
    "crystal-quant.qwen35.real-linear-attention-cuda.v1";

const HIDDEN: usize = 5_120;
const VOCAB: usize = 248_320;
const KEY_HEADS: usize = 16;
const VALUE_HEADS: usize = 48;
const HEAD_DIM: usize = 128;
const KERNEL_SIZE: usize = 4;
const QKV_ELEMENTS: usize = 10_240;
const RECURRENT_ELEMENTS: usize = 6_144;
const CONV_CHANNELS: usize = QKV_ELEMENTS;
const CONV_HISTORY_ELEMENTS: usize = CONV_CHANNELS * (KERNEL_SIZE - 1);
const CONV_WEIGHT_ELEMENTS: usize = CONV_CHANNELS * KERNEL_SIZE;
const STATE_ELEMENTS: usize = VALUE_HEADS * HEAD_DIM * HEAD_DIM;
const STRICT_PROFILE_TENSOR_COUNT: u64 = 1_187;
const MAX_COMPLETION_POLLS: u64 = 2_000_000;

const EMBEDDING: &str = "token_embd.weight";
const ATTN_NORM: &str = "blk.0.attn_norm.weight";
const QKV: &str = "blk.0.attn_qkv.weight";
const GATE: &str = "blk.0.attn_gate.weight";
const ALPHA: &str = "blk.0.ssm_alpha.weight";
const BETA: &str = "blk.0.ssm_beta.weight";
const SSM_OUT: &str = "blk.0.ssm_out.weight";
const CONV: &str = "blk.0.ssm_conv1d.weight";
const SSM_A: &str = "blk.0.ssm_a";
const DT_BIAS: &str = "blk.0.ssm_dt.bias";
const SSM_NORM: &str = "blk.0.ssm_norm.weight";

/// Configuration for the real layer-0 linear-attention output boundary.
#[derive(Clone, Debug)]
pub struct Qwen35RealLinearAttentionCudaConfig {
    pub gguf_path: PathBuf,
    pub config_path: PathBuf,
    pub token_id: usize,
    pub device_ordinal: i32,
    pub warmup: usize,
    pub iterations: usize,
    pub source_commit: String,
}

impl Default for Qwen35RealLinearAttentionCudaConfig {
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
pub enum Qwen35RealLinearAttentionCudaError {
    Runtime(CudaRuntimeError),
    Model(Qwen35Error),
    InvalidConfiguration(&'static str),
    CompletionPollLimit,
}

impl fmt::Display for Qwen35RealLinearAttentionCudaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Runtime(error) => error.fmt(formatter),
            Self::Model(error) => error.fmt(formatter),
            Self::InvalidConfiguration(field) => {
                write!(
                    formatter,
                    "invalid Qwen linear-attention configuration: {field}"
                )
            }
            Self::CompletionPollLimit => {
                formatter.write_str("CUDA completion polling exceeded its bound")
            }
        }
    }
}

impl Error for Qwen35RealLinearAttentionCudaError {}

impl From<CudaRuntimeError> for Qwen35RealLinearAttentionCudaError {
    fn from(error: CudaRuntimeError) -> Self {
        Self::Runtime(error)
    }
}

impl From<Qwen35Error> for Qwen35RealLinearAttentionCudaError {
    fn from(error: Qwen35Error) -> Self {
        Self::Model(error)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Qwen35RealLinearAttentionCudaReport {
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
    pub decoded_elements: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct GeometryReport {
    pub key_heads: usize,
    pub value_heads: usize,
    pub head_dim: usize,
    pub kernel_size: usize,
    pub qkv_elements: usize,
    pub recurrent_elements: usize,
    pub conv_history_elements: usize,
    pub conv_weight_elements: usize,
    pub state_elements: usize,
    pub output_projection_rows: usize,
    pub output_projection_columns: usize,
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
pub struct ClaimsReport {
    pub real_weight_linear_attention_output_executed: bool,
    pub recurrent_step_executed: bool,
    pub two_token_checkpoint_replay: bool,
    pub complete_transformer_executed: bool,
    pub decode_executed: bool,
    pub semantic_parity_established: bool,
    pub product_tps_permitted: bool,
    pub default_cli_enabled: bool,
}

#[derive(Clone)]
struct ProjectedToken {
    qkv: Vec<f32>,
    z: Vec<f32>,
    decay_gate: Vec<f32>,
    beta: Vec<f32>,
}

/// Execute recurrent state update and `ssm_out` projection as one ordered
/// device component for two real embedding tokens.
pub fn run_qwen35_real_linear_attention_cuda(
    config: &Qwen35RealLinearAttentionCudaConfig,
) -> Result<Qwen35RealLinearAttentionCudaReport, Qwen35RealLinearAttentionCudaError> {
    validate_config(config)?;
    let binary_path = std::env::current_exe().map_err(Qwen35Error::from)?;
    let binary_sha256 = sha256_hex(&std::fs::read(&binary_path).map_err(Qwen35Error::from)?);
    let config_bytes = std::fs::read(&config.config_path).map_err(Qwen35Error::from)?;
    let config_sha256 = sha256_hex(&config_bytes);
    let model_config = Qwen35Config::parse(&config_bytes)?;
    if model_config.hidden_size != HIDDEN
        || model_config.vocab_size != VOCAB
        || model_config.num_hidden_layers != 64
        || !model_config.linear_attention_layers.contains(&0)
        || model_config.linear_num_key_heads != KEY_HEADS
        || model_config.linear_num_value_heads != VALUE_HEADS
        || model_config.linear_key_head_dim != HEAD_DIM
        || model_config.linear_value_head_dim != HEAD_DIM
        || model_config.linear_conv_kernel_dim != KERNEL_SIZE
    {
        return Err(Qwen35RealLinearAttentionCudaError::InvalidConfiguration(
            "Qwen linear-attention geometry",
        ));
    }

    let manifest = GgufManifest::inspect(&config.gguf_path, GgufAuditOptions::default())?;
    let descriptors = required_descriptors(&manifest)?;
    let strict_profile_count_match = manifest.tensor_count == STRICT_PROFILE_TENSOR_COUNT;
    let artifact_classification = if strict_profile_count_match {
        "count_matches_strict_profile_but_full_model_qualification_not_run"
    } else {
        "extended_or_combined_manifest"
    };
    let mut reader = manifest.open_tensor_reader(&config.gguf_path)?;
    let second_token_id = config.token_id.checked_add(1).ok_or(
        Qwen35RealLinearAttentionCudaError::InvalidConfiguration("token id"),
    )?;
    let hidden_a = read_embedding_row(&mut reader, &descriptors[0], config.token_id)?;
    let hidden_b = read_embedding_row(&mut reader, &descriptors[0], second_token_id)?;
    let mut payloads = Vec::with_capacity(descriptors.len());
    for descriptor in &descriptors {
        payloads.push(reader.read_tensor_bytes(&descriptor.name)?);
    }
    let tensor_reports = descriptors
        .iter()
        .zip(payloads.iter())
        .map(|(descriptor, payload)| {
            let decoded_elements = descriptor
                .dimensions
                .iter()
                .try_fold(1usize, |count, dimension| {
                    count.checked_mul(*dimension as usize)
                })
                .ok_or(Qwen35RealLinearAttentionCudaError::InvalidConfiguration(
                    "tensor element count",
                ))?;
            Ok(TensorReport {
                name: descriptor.name.clone(),
                dimensions: descriptor.dimensions.clone(),
                ggml_type: descriptor.ggml_type,
                descriptor_payload_bytes: descriptor.byte_len,
                payload_sha256: sha256_hex(payload),
                decoded_elements,
            })
        })
        .collect::<Result<Vec<_>, Qwen35RealLinearAttentionCudaError>>()?;

    let attn_norm = dequantize_tensor(&descriptors[1], &payloads[1])?;
    let conv_raw = dequantize_tensor(&descriptors[7], &payloads[7])?;
    let conv_weights = convert_conv_layout(&conv_raw)?;
    let ssm_a = dequantize_tensor(&descriptors[8], &payloads[8])?;
    let dt_bias = dequantize_tensor(&descriptors[9], &payloads[9])?;
    let ssm_norm = dequantize_tensor(&descriptors[10], &payloads[10])?;
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
        &descriptors[5],
        &payloads[5],
        &dt_bias,
        &ssm_a,
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
        &descriptors[5],
        &payloads[5],
        &dt_bias,
        &ssm_a,
        epsilon,
    )?;
    let mut cpu_block =
        Qwen35LinearAttentionBlock::zeros(KERNEL_SIZE, KEY_HEADS, VALUE_HEADS, HEAD_DIM)?;
    let initial_checkpoint = cpu_block.checkpoint();
    let recurrent_a = cpu_block.step(Qwen35LinearAttentionInputs {
        qkv_mixed: &token_a.qkv,
        conv_weights: &conv_weights,
        z: &token_a.z,
        decay_gate: &token_a.decay_gate,
        beta: &token_a.beta,
        norm_weight: &ssm_norm,
        epsilon,
    })?;
    let attention_a = quantized_matvec_fused_simd(&descriptors[6], &payloads[6], &recurrent_a)?;
    let checkpoint_a = cpu_block.checkpoint();
    let recurrent_b = cpu_block.step(Qwen35LinearAttentionInputs {
        qkv_mixed: &token_b.qkv,
        conv_weights: &conv_weights,
        z: &token_b.z,
        decay_gate: &token_b.decay_gate,
        beta: &token_b.beta,
        norm_weight: &ssm_norm,
        epsilon,
    })?;
    let attention_b = quantized_matvec_fused_simd(&descriptors[6], &payloads[6], &recurrent_b)?;
    let checkpoint_b = cpu_block.checkpoint();
    let reference_ms = reference_started.elapsed().as_secs_f64() * 1_000.0;

    let device = cuda_device_properties(config.device_ordinal)?;
    let stream = OwnedCudaStream::create(config.device_ordinal)?;
    let allocation_start = cuda_device_allocation_snapshot();
    let boundary_start = cuda_host_boundary_snapshot();
    let qkv = OwnedCudaBuffer::allocate(config.device_ordinal, QKV_ELEMENTS * 4)?;
    let z = OwnedCudaBuffer::allocate(config.device_ordinal, RECURRENT_ELEMENTS * 4)?;
    let conv_weights_buffer =
        OwnedCudaBuffer::allocate(config.device_ordinal, CONV_WEIGHT_ELEMENTS * 4)?;
    let conv_history = OwnedCudaBuffer::allocate(config.device_ordinal, CONV_HISTORY_ELEMENTS * 4)?;
    let decay_gate = OwnedCudaBuffer::allocate(config.device_ordinal, VALUE_HEADS * 4)?;
    let beta = OwnedCudaBuffer::allocate(config.device_ordinal, VALUE_HEADS * 4)?;
    let state = OwnedCudaBuffer::allocate(config.device_ordinal, STATE_ELEMENTS * 4)?;
    let norm_weight = OwnedCudaBuffer::allocate(config.device_ordinal, HEAD_DIM * 4)?;
    let activated = OwnedCudaBuffer::allocate(config.device_ordinal, QKV_ELEMENTS * 4)?;
    let recurrent = OwnedCudaBuffer::allocate(config.device_ordinal, RECURRENT_ELEMENTS * 4)?;
    let linear_output = OwnedCudaBuffer::allocate(config.device_ordinal, RECURRENT_ELEMENTS * 4)?;
    let attention_output = OwnedCudaBuffer::allocate(config.device_ordinal, HIDDEN * 4)?;
    let output_weight = OwnedCudaBuffer::allocate(config.device_ordinal, payloads[6].len())?;
    let recurrent_step = bind_linear_step(
        &stream,
        &qkv,
        &z,
        &conv_weights_buffer,
        &conv_history,
        &decay_gate,
        &beta,
        &state,
        &norm_weight,
        &activated,
        &recurrent,
        &linear_output,
        epsilon,
    )?;
    let output_projection =
        bind_output_projection(&stream, &output_weight, &linear_output, &attention_output)?;

    upload_token_a(
        &stream,
        &qkv,
        &z,
        &conv_weights_buffer,
        &decay_gate,
        &beta,
        &norm_weight,
        &output_weight,
        &token_a,
        &conv_weights,
        &ssm_norm,
        &payloads[6],
    )?;
    let mut conv_history = conv_history;
    let mut state = state;
    conv_history.zero_sync(&stream)?;
    state.zero_sync(&stream)?;
    for _ in 0..config.warmup {
        submit_pair(&stream, &recurrent_step, &output_projection)?;
    }
    stream.synchronize()?;
    conv_history.zero_sync(&stream)?;
    state.zero_sync(&stream)?;

    // Include all initialization, warm-up, and state-reset work in the setup
    // interval before opening the steady timing window.
    let setup_allocations = allocation_delta(allocation_start, cuda_device_allocation_snapshot())?;
    let setup_boundaries = boundary_delta(boundary_start, cuda_host_boundary_snapshot())?;

    let steady_alloc_start = cuda_device_allocation_snapshot();
    let steady_boundary_start = cuda_host_boundary_snapshot();
    let started = OwnedCudaEvent::create(config.device_ordinal)?;
    let stopped = OwnedCudaEvent::create(config.device_ordinal)?;
    let wall_start = Instant::now();
    started.record(&stream)?;
    for _ in 0..config.iterations {
        submit_pair(&stream, &recurrent_step, &output_projection)?;
    }
    stopped.record(&stream)?;
    stopped.synchronize()?;
    let steady_batch_wall_ms = wall_start.elapsed().as_secs_f64() * 1_000.0;
    let steady_cuda_event_ms = started.elapsed_ms(&stopped)?;
    let steady_allocations =
        allocation_delta(steady_alloc_start, cuda_device_allocation_snapshot())?;
    let steady_boundaries = boundary_delta(steady_boundary_start, cuda_host_boundary_snapshot())?;

    let input_start = cuda_host_boundary_snapshot();
    conv_history.zero_sync(&stream)?;
    state.zero_sync(&stream)?;
    let mut sequence_input_boundaries = boundary_delta(input_start, cuda_host_boundary_snapshot())?;
    let audited_a = execute_and_readback(
        &stream,
        &recurrent_step,
        &output_projection,
        &state,
        &linear_output,
        &attention_output,
        &conv_history,
    )?;

    let input_start = cuda_host_boundary_snapshot();
    upload_token(&stream, &qkv, &z, &decay_gate, &beta, &token_b)?;
    sequence_input_boundaries = add_boundaries(
        sequence_input_boundaries,
        boundary_delta(input_start, cuda_host_boundary_snapshot())?,
    );
    let audited_b = execute_and_readback(
        &stream,
        &recurrent_step,
        &output_projection,
        &state,
        &linear_output,
        &attention_output,
        &conv_history,
    )?;

    let input_start = cuda_host_boundary_snapshot();
    conv_history.zero_sync(&stream)?;
    state.zero_sync(&stream)?;
    upload_token(&stream, &qkv, &z, &decay_gate, &beta, &token_a)?;
    sequence_input_boundaries = add_boundaries(
        sequence_input_boundaries,
        boundary_delta(input_start, cuda_host_boundary_snapshot())?,
    );
    let replay_a = execute_and_readback(
        &stream,
        &recurrent_step,
        &output_projection,
        &state,
        &linear_output,
        &attention_output,
        &conv_history,
    )?;

    let gpu_replay_exact = equal_f32_bits(&audited_a.recurrent, &replay_a.recurrent)
        && equal_f32_bits(&audited_a.attention, &replay_a.attention)
        && equal_f32_bits(&audited_a.state, &replay_a.state)
        && equal_f32_bits(&audited_a.history, &replay_a.history);
    let numerical = vec![
        compare(
            "token_a.recurrent",
            &audited_a.recurrent,
            &recurrent_a,
            2e-3,
            4e-3,
        ),
        compare(
            "token_b.recurrent",
            &audited_b.recurrent,
            &recurrent_b,
            2e-3,
            4e-3,
        ),
        compare(
            "token_a.ssm_out",
            &audited_a.attention,
            &attention_a,
            5e-3,
            2e-4,
        ),
        compare(
            "token_b.ssm_out",
            &audited_b.attention,
            &attention_b,
            5e-3,
            2e-4,
        ),
        compare(
            "token_a.state",
            &audited_a.state,
            checkpoint_a.gated_delta_values(),
            2e-3,
            4e-3,
        ),
        compare(
            "token_b.state",
            &audited_b.state,
            checkpoint_b.gated_delta_values(),
            2e-3,
            4e-3,
        ),
        compare(
            "token_a.conv_history",
            &audited_a.history,
            checkpoint_a.conv_history(),
            2e-4,
            2e-3,
        ),
        compare(
            "token_b.conv_history",
            &audited_b.history,
            checkpoint_b.conv_history(),
            2e-4,
            2e-3,
        ),
    ];
    let cpu_replay_exact = {
        let mut replay = cpu_block;
        replay.restore(&initial_checkpoint)?;
        let recurrent = replay.step(Qwen35LinearAttentionInputs {
            qkv_mixed: &token_a.qkv,
            conv_weights: &conv_weights,
            z: &token_a.z,
            decay_gate: &token_a.decay_gate,
            beta: &token_a.beta,
            norm_weight: &ssm_norm,
            epsilon,
        })?;
        let attention = quantized_matvec_fused_simd(&descriptors[6], &payloads[6], &recurrent)?;
        equal_f32_bits(&recurrent, &recurrent_a) && equal_f32_bits(&attention, &attention_a)
    };
    let completion = CompletionReport {
        poll_iterations: audited_a.polls + audited_b.polls + replay_a.polls,
        nonblocking_poll_only: audited_a.nonblocking
            && audited_b.nonblocking
            && replay_a.nonblocking,
        owner_epoch_proof: audited_a.proof && audited_b.proof && replay_a.proof,
    };
    let completion_boundaries = add_boundaries(
        add_boundaries(
            audited_a.completion_boundaries,
            audited_b.completion_boundaries,
        ),
        replay_a.completion_boundaries,
    );
    let readback_boundaries = add_boundaries(
        add_boundaries(audited_a.readback_boundaries, audited_b.readback_boundaries),
        replay_a.readback_boundaries,
    );

    let cleanup_boundary_start = cuda_host_boundary_snapshot();
    stream.synchronize()?;
    let cleanup_boundaries = boundary_delta(cleanup_boundary_start, cuda_host_boundary_snapshot())?;
    let before_cleanup = cuda_device_allocation_snapshot();
    drop(output_weight);
    drop(attention_output);
    drop(linear_output);
    drop(recurrent);
    drop(activated);
    drop(norm_weight);
    drop(state);
    drop(decay_gate);
    drop(beta);
    drop(conv_history);
    drop(conv_weights_buffer);
    drop(z);
    drop(qkv);
    drop(stream);
    let after_cleanup = cuda_device_allocation_snapshot();
    let final_allocation_delta = allocation_delta(before_cleanup, after_cleanup)?;
    let total_boundaries = boundary_delta(boundary_start, cuda_host_boundary_snapshot())?;
    let accounted_boundaries = add_boundaries(
        add_boundaries(
            add_boundaries(setup_boundaries.clone(), steady_boundaries.clone()),
            sequence_input_boundaries.clone(),
        ),
        add_boundaries(
            add_boundaries(completion_boundaries.clone(), readback_boundaries.clone()),
            cleanup_boundaries.clone(),
        ),
    );
    let boundary_ledger_closed = accounted_boundaries == total_boundaries;
    let cleanup_zero =
        final_allocation_delta.live_allocations == 0 && final_allocation_delta.live_bytes == 0;
    let pass = numerical.iter().all(|item| item.pass)
        && gpu_replay_exact
        && cpu_replay_exact
        && completion.nonblocking_poll_only
        && completion.owner_epoch_proof
        && boundary_ledger_closed
        && cleanup_zero;

    Ok(Qwen35RealLinearAttentionCudaReport {
        schema: QWEN35_REAL_LINEAR_ATTENTION_CUDA_SCHEMA.to_owned(),
        status: if pass {
            "PASS_COMPONENT_REAL_LINEAR_ATTENTION_OUTPUT_CUDA"
        } else {
            "FAIL_COMPONENT_REAL_LINEAR_ATTENTION_OUTPUT_CUDA"
        }
        .to_owned(),
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
            key_heads: KEY_HEADS,
            value_heads: VALUE_HEADS,
            head_dim: HEAD_DIM,
            kernel_size: KERNEL_SIZE,
            qkv_elements: QKV_ELEMENTS,
            recurrent_elements: RECURRENT_ELEMENTS,
            conv_history_elements: CONV_HISTORY_ELEMENTS,
            conv_weight_elements: CONV_WEIGHT_ELEMENTS,
            state_elements: STATE_ELEMENTS,
            output_projection_rows: HIDDEN,
            output_projection_columns: RECURRENT_ELEMENTS,
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
            real_weight_linear_attention_output_executed: true,
            recurrent_step_executed: true,
            two_token_checkpoint_replay: gpu_replay_exact && cpu_replay_exact,
            complete_transformer_executed: false,
            decode_executed: false,
            semantic_parity_established: false,
            product_tps_permitted: false,
            default_cli_enabled: false,
        },
    })
}

fn validate_config(
    config: &Qwen35RealLinearAttentionCudaConfig,
) -> Result<(), Qwen35RealLinearAttentionCudaError> {
    if config.gguf_path.as_os_str().is_empty() || !config.gguf_path.is_file() {
        return Err(Qwen35RealLinearAttentionCudaError::InvalidConfiguration(
            "GGUF path",
        ));
    }
    if config.config_path.as_os_str().is_empty() || !config.config_path.is_file() {
        return Err(Qwen35RealLinearAttentionCudaError::InvalidConfiguration(
            "config path",
        ));
    }
    if config.device_ordinal < 0 || config.iterations == 0 || config.token_id >= VOCAB - 1 {
        return Err(Qwen35RealLinearAttentionCudaError::InvalidConfiguration(
            "runner bounds",
        ));
    }
    Ok(())
}

fn required_descriptors(
    manifest: &GgufManifest,
) -> Result<Vec<GgufTensor>, Qwen35RealLinearAttentionCudaError> {
    let specs = [
        (EMBEDDING, vec![HIDDEN as u64, VOCAB as u64], GGML_TYPE_BF16),
        (ATTN_NORM, vec![HIDDEN as u64], GGML_TYPE_F32),
        (
            QKV,
            vec![HIDDEN as u64, QKV_ELEMENTS as u64],
            GGML_TYPE_Q8_0,
        ),
        (
            GATE,
            vec![HIDDEN as u64, RECURRENT_ELEMENTS as u64],
            GGML_TYPE_Q8_0,
        ),
        (
            ALPHA,
            vec![HIDDEN as u64, VALUE_HEADS as u64],
            GGML_TYPE_BF16,
        ),
        (
            BETA,
            vec![HIDDEN as u64, VALUE_HEADS as u64],
            GGML_TYPE_BF16,
        ),
        (
            SSM_OUT,
            vec![RECURRENT_ELEMENTS as u64, HIDDEN as u64],
            GGML_TYPE_Q8_0,
        ),
        (
            CONV,
            vec![KERNEL_SIZE as u64, CONV_CHANNELS as u64],
            GGML_TYPE_F32,
        ),
        (SSM_A, vec![VALUE_HEADS as u64], GGML_TYPE_F32),
        (DT_BIAS, vec![VALUE_HEADS as u64], GGML_TYPE_F32),
        (SSM_NORM, vec![HEAD_DIM as u64], GGML_TYPE_F32),
    ];
    specs
        .into_iter()
        .map(|(name, dimensions, ggml_type)| {
            let tensor = manifest.tensor(name).ok_or(
                Qwen35RealLinearAttentionCudaError::InvalidConfiguration("required tensor missing"),
            )?;
            if tensor.dimensions != dimensions || tensor.ggml_type != ggml_type {
                return Err(Qwen35RealLinearAttentionCudaError::InvalidConfiguration(
                    "required tensor geometry/type",
                ));
            }
            Ok(tensor.clone())
        })
        .collect()
}

fn read_embedding_row(
    reader: &mut augur_model_qwen35::GgufTensorReader<'_>,
    descriptor: &GgufTensor,
    token_id: usize,
) -> Result<Vec<f32>, Qwen35RealLinearAttentionCudaError> {
    let row_bytes = HIDDEN * 2;
    let offset = token_id.checked_mul(row_bytes).ok_or(
        Qwen35RealLinearAttentionCudaError::InvalidConfiguration("embedding offset"),
    )?;
    let bytes = reader.read_tensor_slice(&descriptor.name, offset as u64, row_bytes as u64)?;
    let mut values = vec![0.0; HIDDEN];
    dequantize_bf16(&bytes, &mut values)?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(Qwen35RealLinearAttentionCudaError::InvalidConfiguration(
            "embedding finite",
        ));
    }
    Ok(values)
}

#[allow(clippy::too_many_arguments)]
fn project_token(
    hidden: &[f32],
    attn_norm: &[f32],
    qkv_descriptor: &GgufTensor,
    qkv_payload: &[u8],
    gate_descriptor: &GgufTensor,
    gate_payload: &[u8],
    alpha_descriptor: &GgufTensor,
    alpha_payload: &[u8],
    beta_descriptor: &GgufTensor,
    beta_payload: &[u8],
    dt_bias: &[f32],
    ssm_a: &[f32],
    epsilon: f32,
) -> Result<ProjectedToken, Qwen35RealLinearAttentionCudaError> {
    let normalized = rms_norm_last_dim(
        &ReferenceTensor::new(vec![1, HIDDEN], hidden.to_vec())?,
        &ReferenceTensor::new(vec![HIDDEN], attn_norm.to_vec())?,
        epsilon,
    )?;
    let input = normalized.values();
    let qkv = quantized_matvec_fused_simd(qkv_descriptor, qkv_payload, input)?;
    let z = quantized_matvec_fused_simd(gate_descriptor, gate_payload, input)?;
    let alpha_raw = quantized_matvec_fused_simd(alpha_descriptor, alpha_payload, input)?;
    let beta_raw = quantized_matvec_fused_simd(beta_descriptor, beta_payload, input)?;
    if qkv.len() != QKV_ELEMENTS
        || z.len() != RECURRENT_ELEMENTS
        || alpha_raw.len() != VALUE_HEADS
        || beta_raw.len() != VALUE_HEADS
        || dt_bias.len() != VALUE_HEADS
        || ssm_a.len() != VALUE_HEADS
    {
        return Err(Qwen35RealLinearAttentionCudaError::InvalidConfiguration(
            "projection output geometry",
        ));
    }
    let mut decay_gate = Vec::with_capacity(VALUE_HEADS);
    let mut beta = Vec::with_capacity(VALUE_HEADS);
    for index in 0..VALUE_HEADS {
        let value = alpha_raw[index] + dt_bias[index];
        let softplus = if value > 20.0 {
            value
        } else if value < -20.0 {
            value.exp()
        } else {
            (1.0 + value.exp()).ln()
        };
        decay_gate.push(softplus * ssm_a[index]);
        let gate = beta_raw[index];
        beta.push(if gate >= 0.0 {
            1.0 / (1.0 + (-gate).exp())
        } else {
            let exponent = gate.exp();
            exponent / (1.0 + exponent)
        });
    }
    if qkv
        .iter()
        .chain(z.iter())
        .chain(decay_gate.iter())
        .chain(beta.iter())
        .any(|value| !value.is_finite())
    {
        return Err(Qwen35RealLinearAttentionCudaError::InvalidConfiguration(
            "projection finite values",
        ));
    }
    Ok(ProjectedToken {
        qkv,
        z,
        decay_gate,
        beta,
    })
}

fn convert_conv_layout(raw: &[f32]) -> Result<Vec<f32>, Qwen35RealLinearAttentionCudaError> {
    if raw.len() != CONV_WEIGHT_ELEMENTS {
        return Err(Qwen35RealLinearAttentionCudaError::InvalidConfiguration(
            "conv weight extent",
        ));
    }
    let mut converted = vec![0.0; raw.len()];
    for channel in 0..CONV_CHANNELS {
        for tap in 0..KERNEL_SIZE {
            converted[channel * KERNEL_SIZE + tap] = raw[tap + channel * KERNEL_SIZE];
        }
    }
    if converted.iter().any(|value| !value.is_finite()) {
        return Err(Qwen35RealLinearAttentionCudaError::InvalidConfiguration(
            "conv weights finite",
        ));
    }
    Ok(converted)
}

#[allow(clippy::too_many_arguments)]
fn bind_linear_step(
    stream: &OwnedCudaStream,
    qkv: &OwnedCudaBuffer,
    z: &OwnedCudaBuffer,
    conv_weights: &OwnedCudaBuffer,
    conv_history: &OwnedCudaBuffer,
    decay_gate: &OwnedCudaBuffer,
    beta: &OwnedCudaBuffer,
    state: &OwnedCudaBuffer,
    norm_weight: &OwnedCudaBuffer,
    activated: &OwnedCudaBuffer,
    recurrent: &OwnedCudaBuffer,
    output: &OwnedCudaBuffer,
    epsilon: f32,
) -> Result<CudaQwen35LinearStepArgsV1, Qwen35RealLinearAttentionCudaError> {
    let view = |buffer: &OwnedCudaBuffer| -> Result<CudaF32View, CudaRuntimeError> {
        CudaF32View::new(
            buffer.view(0, buffer.len_bytes())?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            buffer.len_bytes() / 4,
        )
    };
    Ok(CudaQwen35LinearStepArgsV1::bind(
        stream,
        view(qkv)?,
        view(z)?,
        view(conv_weights)?,
        view(conv_history)?,
        view(decay_gate)?,
        view(beta)?,
        view(state)?,
        view(norm_weight)?,
        view(activated)?,
        view(recurrent)?,
        view(output)?,
        epsilon,
    )?)
}

fn bind_output_projection(
    stream: &OwnedCudaStream,
    weight: &OwnedCudaBuffer,
    input: &OwnedCudaBuffer,
    output: &OwnedCudaBuffer,
) -> Result<CudaQ8_0F32MatvecArgsV1, Qwen35RealLinearAttentionCudaError> {
    Ok(CudaQ8_0F32MatvecArgsV1::bind(
        stream,
        CudaQ8_0WeightViewV1::new(
            weight.view(0, weight.len_bytes())?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_0_CUDA_V1,
            HIDDEN,
            RECURRENT_ELEMENTS,
        )?,
        CudaF32View::new(
            input.view(0, input.len_bytes())?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            RECURRENT_ELEMENTS,
        )?,
        CudaF32View::new(
            output.view(0, output.len_bytes())?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            HIDDEN,
        )?,
    )?)
}

#[allow(clippy::too_many_arguments)]
fn upload_token_a(
    stream: &OwnedCudaStream,
    qkv: &OwnedCudaBuffer,
    z: &OwnedCudaBuffer,
    conv_weights: &OwnedCudaBuffer,
    decay_gate: &OwnedCudaBuffer,
    beta: &OwnedCudaBuffer,
    norm_weight: &OwnedCudaBuffer,
    output_weight: &OwnedCudaBuffer,
    token: &ProjectedToken,
    conv: &[f32],
    norm: &[f32],
    output_payload: &[u8],
) -> Result<(), Qwen35RealLinearAttentionCudaError> {
    qkv.copy_from_host_sync(stream, &f32_bytes(&token.qkv))?;
    z.copy_from_host_sync(stream, &f32_bytes(&token.z))?;
    conv_weights.copy_from_host_sync(stream, &f32_bytes(conv))?;
    decay_gate.copy_from_host_sync(stream, &f32_bytes(&token.decay_gate))?;
    beta.copy_from_host_sync(stream, &f32_bytes(&token.beta))?;
    norm_weight.copy_from_host_sync(stream, &f32_bytes(norm))?;
    output_weight.copy_from_host_sync(stream, output_payload)?;
    Ok(())
}

fn upload_token(
    stream: &OwnedCudaStream,
    qkv: &OwnedCudaBuffer,
    z: &OwnedCudaBuffer,
    decay_gate: &OwnedCudaBuffer,
    beta: &OwnedCudaBuffer,
    token: &ProjectedToken,
) -> Result<(), Qwen35RealLinearAttentionCudaError> {
    qkv.copy_from_host_sync(stream, &f32_bytes(&token.qkv))?;
    z.copy_from_host_sync(stream, &f32_bytes(&token.z))?;
    decay_gate.copy_from_host_sync(stream, &f32_bytes(&token.decay_gate))?;
    beta.copy_from_host_sync(stream, &f32_bytes(&token.beta))?;
    Ok(())
}

fn submit_pair(
    stream: &OwnedCudaStream,
    recurrent: &CudaQwen35LinearStepArgsV1,
    output: &CudaQ8_0F32MatvecArgsV1,
) -> Result<(), Qwen35RealLinearAttentionCudaError> {
    recurrent.submit(stream)?;
    output.submit_for_plan(stream)?;
    Ok(())
}

struct DeviceReadback {
    recurrent: Vec<f32>,
    attention: Vec<f32>,
    state: Vec<f32>,
    history: Vec<f32>,
    polls: u64,
    nonblocking: bool,
    proof: bool,
    completion_boundaries: BoundaryReport,
    readback_boundaries: BoundaryReport,
}

#[allow(clippy::too_many_arguments)]
fn execute_and_readback(
    stream: &OwnedCudaStream,
    recurrent_step: &CudaQwen35LinearStepArgsV1,
    output_projection: &CudaQ8_0F32MatvecArgsV1,
    state: &OwnedCudaBuffer,
    output: &OwnedCudaBuffer,
    attention_output: &OwnedCudaBuffer,
    history: &OwnedCudaBuffer,
) -> Result<DeviceReadback, Qwen35RealLinearAttentionCudaError> {
    let completion_before = cuda_host_boundary_snapshot();
    submit_pair(stream, recurrent_step, output_projection)?;
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
            return Err(Qwen35RealLinearAttentionCudaError::CompletionPollLimit);
        }
        spin_loop();
    };
    let proof_ok = proof.covers_submission(receipt) && proof.belongs_to_stream(stream)?;
    if !proof_ok {
        return Err(Qwen35RealLinearAttentionCudaError::InvalidConfiguration(
            "owner epoch proof",
        ));
    }
    let completion_after = cuda_host_boundary_snapshot();
    let completion_boundaries = boundary_delta(completion_before, completion_after)?;
    let readback_before = cuda_host_boundary_snapshot();
    let mut recurrent_bytes = vec![0_u8; RECURRENT_ELEMENTS * 4];
    let mut attention_bytes = vec![0_u8; HIDDEN * 4];
    let mut state_bytes = vec![0_u8; STATE_ELEMENTS * 4];
    let mut history_bytes = vec![0_u8; CONV_HISTORY_ELEMENTS * 4];
    output.copy_to_host_sync_with_submission_proof(stream, &mut recurrent_bytes, receipt)?;
    attention_output.copy_to_host_sync_with_submission_proof(
        stream,
        &mut attention_bytes,
        receipt,
    )?;
    state.copy_to_host_sync_with_submission_proof(stream, &mut state_bytes, receipt)?;
    history.copy_to_host_sync_with_submission_proof(stream, &mut history_bytes, receipt)?;
    let readback_after = cuda_host_boundary_snapshot();
    Ok(DeviceReadback {
        recurrent: f32_vec(&recurrent_bytes),
        attention: f32_vec(&attention_bytes),
        state: f32_vec(&state_bytes),
        history: f32_vec(&history_bytes),
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

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
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

fn allocation_delta(
    earlier: augur_backend::CudaDeviceAllocationSnapshot,
    later: augur_backend::CudaDeviceAllocationSnapshot,
) -> Result<AllocationReport, Qwen35RealLinearAttentionCudaError> {
    Ok(AllocationReport {
        successful_allocations: later
            .successful_allocations
            .checked_sub(earlier.successful_allocations)
            .ok_or(Qwen35RealLinearAttentionCudaError::InvalidConfiguration(
                "allocation counters",
            ))?,
        successful_frees: later
            .successful_frees
            .checked_sub(earlier.successful_frees)
            .ok_or(Qwen35RealLinearAttentionCudaError::InvalidConfiguration(
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
) -> Result<BoundaryReport, Qwen35RealLinearAttentionCudaError> {
    let delta = later.checked_delta_since(earlier).ok_or(
        Qwen35RealLinearAttentionCudaError::InvalidConfiguration("host counters"),
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
