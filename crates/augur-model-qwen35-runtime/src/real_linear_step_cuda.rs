#![cfg(feature = "qwen35-linear-step")]

//! Real-weight Qwen3.8 layer-0 recurrent-step component evidence.
//!
//! This lane starts from CPU-authenticated projection values and executes the
//! causal convolution, Gated DeltaNet state update, and gated RMSNorm on SM70.
//! It intentionally stops before `ssm_out`, the remaining transformer layers,
//! token publication, and any product/decode path.

use std::{error::Error, fmt, hint::spin_loop, path::PathBuf, time::Instant};

use augur_backend::{
    cuda_device_allocation_snapshot, cuda_device_properties, cuda_host_boundary_snapshot,
    CudaF32View, CudaQwen35LinearStepArgsV1, CudaRuntimeError, OwnedCudaBuffer, OwnedCudaEvent,
    OwnedCudaStream,
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

pub const QWEN35_REAL_LINEAR_STEP_CUDA_SCHEMA: &str =
    "crystal-quant.qwen35.real-linear-step-cuda.v1";

const HIDDEN: usize = 5_120;
const VOCAB: usize = 248_320;
const KEY_HEADS: usize = 16;
const VALUE_HEADS: usize = 48;
const HEAD_DIM: usize = 128;
const KERNEL_SIZE: usize = 4;
const QKV_ELEMENTS: usize = 10_240;
const OUTPUT_ELEMENTS: usize = 6_144;
const CONV_CHANNELS: usize = 10_240;
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
const CONV: &str = "blk.0.ssm_conv1d.weight";
const SSM_A: &str = "blk.0.ssm_a";
const DT_BIAS: &str = "blk.0.ssm_dt.bias";
const SSM_NORM: &str = "blk.0.ssm_norm.weight";

#[derive(Clone, Debug)]
pub struct Qwen35RealLinearStepCudaConfig {
    pub gguf_path: PathBuf,
    pub config_path: PathBuf,
    pub token_id: usize,
    pub device_ordinal: i32,
    pub warmup: usize,
    pub iterations: usize,
    pub source_commit: String,
}

impl Default for Qwen35RealLinearStepCudaConfig {
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
pub enum Qwen35RealLinearStepCudaError {
    Runtime(CudaRuntimeError),
    Model(Qwen35Error),
    InvalidConfiguration(&'static str),
    CompletionPollLimit,
}

impl fmt::Display for Qwen35RealLinearStepCudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Runtime(error) => error.fmt(f),
            Self::Model(error) => error.fmt(f),
            Self::InvalidConfiguration(field) => {
                write!(f, "invalid Qwen linear-step configuration: {field}")
            }
            Self::CompletionPollLimit => f.write_str("CUDA completion polling exceeded its bound"),
        }
    }
}

impl Error for Qwen35RealLinearStepCudaError {}

impl From<CudaRuntimeError> for Qwen35RealLinearStepCudaError {
    fn from(error: CudaRuntimeError) -> Self {
        Self::Runtime(error)
    }
}

impl From<Qwen35Error> for Qwen35RealLinearStepCudaError {
    fn from(error: Qwen35Error) -> Self {
        Self::Model(error)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Qwen35RealLinearStepCudaReport {
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
pub struct ArtifactReport {
    pub manifest_tensor_count: u64,
    pub audited_tensor_count: usize,
    pub strict_profile_expected_tensor_count: u64,
    pub strict_profile_count_match: bool,
    pub strict_profile_qualified: bool,
    pub classification: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct DeviceReport {
    pub ordinal: i32,
    pub name: String,
    pub compute_capability: [u16; 2],
    pub total_global_memory_bytes: u64,
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
    pub output_elements: usize,
    pub conv_channels: usize,
    pub conv_history_elements: usize,
    pub conv_weight_elements: usize,
    pub state_elements: usize,
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
    pub completion_boundaries: BoundaryReport,
    pub readback_boundaries: BoundaryReport,
    pub final_allocation_delta: AllocationReport,
}

#[derive(Clone, Debug, Serialize)]
pub struct AllocationReport {
    pub successful_allocations: u64,
    pub successful_frees: u64,
    pub live_allocations: u64,
    pub live_bytes: u64,
    pub peak_live_allocations: u64,
    pub peak_live_bytes: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct BoundaryReport {
    pub host_waits: u64,
    pub stream_synchronizations: u64,
    pub event_synchronizations: u64,
    pub synchronous_h2d_copies: u64,
    pub synchronous_h2d_bytes: u64,
    pub synchronous_d2h_copies: u64,
    pub synchronous_d2h_bytes: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ClaimsReport {
    pub real_weight_inputs_loaded: bool,
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

/// Execute two real-weight recurrent steps and compare them with the CPU
/// reference, including a state checkpoint/replay.
pub fn run_qwen35_real_linear_step_cuda(
    config: &Qwen35RealLinearStepCudaConfig,
) -> Result<Qwen35RealLinearStepCudaReport, Qwen35RealLinearStepCudaError> {
    validate_config(config)?;
    let binary_path = std::env::current_exe().map_err(Qwen35Error::from)?;
    let binary_bytes = std::fs::read(&binary_path).map_err(Qwen35Error::from)?;
    let binary_sha256 = sha256_hex(&binary_bytes);
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
        return Err(Qwen35RealLinearStepCudaError::InvalidConfiguration(
            "Qwen linear-attention geometry",
        ));
    }
    let manifest = GgufManifest::inspect(&config.gguf_path, GgufAuditOptions::default())?;
    let descriptors = required_descriptors(&manifest)?;
    let strict_profile_count_match = manifest.tensor_count == STRICT_PROFILE_TENSOR_COUNT;
    let artifact_classification = if strict_profile_count_match {
        "count_matches_strict_profile_but_validation_not_run"
    } else {
        "extended_or_combined_manifest"
    };
    let mut reader = manifest.open_tensor_reader(&config.gguf_path)?;
    let embedding = read_embedding_row(&mut reader, &descriptors[0], config.token_id)?;
    let second_token_id = config.token_id.checked_add(1).ok_or(
        Qwen35RealLinearStepCudaError::InvalidConfiguration("token id"),
    )?;
    let embedding_second = read_embedding_row(&mut reader, &descriptors[0], second_token_id)?;
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
                .ok_or(Qwen35RealLinearStepCudaError::InvalidConfiguration(
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
        .collect::<Result<Vec<_>, Qwen35RealLinearStepCudaError>>()?;

    let attn_norm = dequantize_tensor(&descriptors[1], &payloads[1])?;
    let conv_raw = dequantize_tensor(&descriptors[6], &payloads[6])?;
    let conv_weights = convert_conv_layout(&conv_raw)?;
    let ssm_a = dequantize_tensor(&descriptors[7], &payloads[7])?;
    let dt_bias = dequantize_tensor(&descriptors[8], &payloads[8])?;
    let ssm_norm = dequantize_tensor(&descriptors[9], &payloads[9])?;
    let epsilon = model_config.rms_norm_eps as f32;
    let reference_started = Instant::now();
    let token_a = project_token(
        &embedding,
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
        &embedding_second,
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
    let expected_a = cpu_block.step(Qwen35LinearAttentionInputs {
        qkv_mixed: &token_a.qkv,
        conv_weights: &conv_weights,
        z: &token_a.z,
        decay_gate: &token_a.decay_gate,
        beta: &token_a.beta,
        norm_weight: &ssm_norm,
        epsilon,
    })?;
    let checkpoint_a = cpu_block.checkpoint();
    let expected_b = cpu_block.step(Qwen35LinearAttentionInputs {
        qkv_mixed: &token_b.qkv,
        conv_weights: &conv_weights,
        z: &token_b.z,
        decay_gate: &token_b.decay_gate,
        beta: &token_b.beta,
        norm_weight: &ssm_norm,
        epsilon,
    })?;
    let checkpoint_b = cpu_block.checkpoint();
    let reference_ms = reference_started.elapsed().as_secs_f64() * 1_000.0;

    let device = cuda_device_properties(config.device_ordinal)?;
    let stream = OwnedCudaStream::create(config.device_ordinal)?;
    let allocation_start = cuda_device_allocation_snapshot();
    let boundary_start = cuda_host_boundary_snapshot();
    let qkv = OwnedCudaBuffer::allocate(config.device_ordinal, QKV_ELEMENTS * 4)?;
    let z = OwnedCudaBuffer::allocate(config.device_ordinal, OUTPUT_ELEMENTS * 4)?;
    let conv_weights_buffer =
        OwnedCudaBuffer::allocate(config.device_ordinal, CONV_WEIGHT_ELEMENTS * 4)?;
    let conv_history = OwnedCudaBuffer::allocate(config.device_ordinal, CONV_HISTORY_ELEMENTS * 4)?;
    let decay_gate = OwnedCudaBuffer::allocate(config.device_ordinal, VALUE_HEADS * 4)?;
    let beta = OwnedCudaBuffer::allocate(config.device_ordinal, VALUE_HEADS * 4)?;
    let state = OwnedCudaBuffer::allocate(config.device_ordinal, STATE_ELEMENTS * 4)?;
    let norm_weight = OwnedCudaBuffer::allocate(config.device_ordinal, HEAD_DIM * 4)?;
    let activated = OwnedCudaBuffer::allocate(config.device_ordinal, QKV_ELEMENTS * 4)?;
    let recurrent = OwnedCudaBuffer::allocate(config.device_ordinal, OUTPUT_ELEMENTS * 4)?;
    let output = OwnedCudaBuffer::allocate(config.device_ordinal, OUTPUT_ELEMENTS * 4)?;
    let input = bind_views(
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
        &output,
        epsilon,
    )?;
    qkv.copy_from_host_sync(&stream, &f32_bytes(&token_a.qkv))?;
    z.copy_from_host_sync(&stream, &f32_bytes(&token_a.z))?;
    conv_weights_buffer.copy_from_host_sync(&stream, &f32_bytes(&conv_weights))?;
    decay_gate.copy_from_host_sync(&stream, &f32_bytes(&token_a.decay_gate))?;
    beta.copy_from_host_sync(&stream, &f32_bytes(&token_a.beta))?;
    norm_weight.copy_from_host_sync(&stream, &f32_bytes(&ssm_norm))?;
    let setup_allocations = allocation_delta(allocation_start, cuda_device_allocation_snapshot())?;
    let setup_boundaries = boundary_delta(boundary_start, cuda_host_boundary_snapshot())?;
    let mut conv_history = conv_history;
    let mut state = state;
    conv_history.zero_sync(&stream)?;
    state.zero_sync(&stream)?;
    for _ in 0..config.warmup {
        input.submit(&stream)?;
    }
    stream.synchronize()?;
    conv_history.zero_sync(&stream)?;
    state.zero_sync(&stream)?;

    let steady_alloc_start = cuda_device_allocation_snapshot();
    let steady_boundary_start = cuda_host_boundary_snapshot();
    let started = OwnedCudaEvent::create(config.device_ordinal)?;
    let stopped = OwnedCudaEvent::create(config.device_ordinal)?;
    let wall_start = Instant::now();
    started.record(&stream)?;
    for _ in 0..config.iterations {
        input.submit(&stream)?;
    }
    stopped.record(&stream)?;
    stopped.synchronize()?;
    let steady_batch_wall_ms = wall_start.elapsed().as_secs_f64() * 1_000.0;
    let steady_cuda_event_ms = started.elapsed_ms(&stopped)?;
    let steady_allocations =
        allocation_delta(steady_alloc_start, cuda_device_allocation_snapshot())?;
    let steady_boundaries = boundary_delta(steady_boundary_start, cuda_host_boundary_snapshot())?;

    // Re-zero after the timing loop and execute the two-token audited sequence.
    conv_history.zero_sync(&stream)?;
    state.zero_sync(&stream)?;
    let (
        actual_a,
        actual_state_a,
        actual_history_a,
        completion_a,
        completion_boundary_a,
        readback_boundary_a,
    ) = execute_and_readback(
        &stream,
        &input,
        &qkv,
        &z,
        &decay_gate,
        &beta,
        &token_a,
        &state,
        &output,
        &conv_history,
    )?;
    // Update the device input vectors for the second token.  The copies are
    // intentionally outside the recurrent kernel and are visible in the
    // boundary ledger.
    qkv.copy_from_host_sync(&stream, &f32_bytes(&token_b.qkv))?;
    z.copy_from_host_sync(&stream, &f32_bytes(&token_b.z))?;
    decay_gate.copy_from_host_sync(&stream, &f32_bytes(&token_b.decay_gate))?;
    beta.copy_from_host_sync(&stream, &f32_bytes(&token_b.beta))?;
    let (
        actual_b,
        actual_state_b,
        actual_history_b,
        completion_b,
        completion_boundary_b,
        readback_boundary_b,
    ) = execute_and_readback(
        &stream,
        &input,
        &qkv,
        &z,
        &decay_gate,
        &beta,
        &token_b,
        &state,
        &output,
        &conv_history,
    )?;

    // A device-side reset followed by token A is the GPU checkpoint/replay
    // proof.  It is intentionally separate from the CPU oracle replay below.
    conv_history.zero_sync(&stream)?;
    state.zero_sync(&stream)?;
    qkv.copy_from_host_sync(&stream, &f32_bytes(&token_a.qkv))?;
    z.copy_from_host_sync(&stream, &f32_bytes(&token_a.z))?;
    decay_gate.copy_from_host_sync(&stream, &f32_bytes(&token_a.decay_gate))?;
    beta.copy_from_host_sync(&stream, &f32_bytes(&token_a.beta))?;
    let (
        replay_gpu,
        replay_gpu_state,
        replay_gpu_history,
        completion_replay,
        completion_boundary_replay,
        readback_boundary_replay,
    ) = execute_and_readback(
        &stream,
        &input,
        &qkv,
        &z,
        &decay_gate,
        &beta,
        &token_a,
        &state,
        &output,
        &conv_history,
    )?;
    let gpu_replay_exact = replay_gpu
        .iter()
        .zip(actual_a.iter())
        .all(|(left, right)| left.to_bits() == right.to_bits())
        && replay_gpu_state
            .iter()
            .zip(actual_state_a.iter())
            .all(|(left, right)| left.to_bits() == right.to_bits())
        && replay_gpu_history
            .iter()
            .zip(actual_history_a.iter())
            .all(|(left, right)| left.to_bits() == right.to_bits());

    let numerical = vec![
        compare("token_a.output", &actual_a, &expected_a, 5e-4, 2e-3),
        compare("token_b.output", &actual_b, &expected_b, 5e-4, 2e-3),
        compare(
            "token_a.state",
            &actual_state_a,
            checkpoint_a.gated_delta_values(),
            2e-3,
            4e-3,
        ),
        compare(
            "token_b.state",
            &actual_state_b,
            checkpoint_b.gated_delta_values(),
            2e-3,
            4e-3,
        ),
        compare(
            "token_a.conv_history",
            &actual_history_a,
            checkpoint_a.conv_history(),
            2e-4,
            2e-3,
        ),
        compare(
            "token_b.conv_history",
            &actual_history_b,
            checkpoint_b.conv_history(),
            2e-4,
            2e-3,
        ),
    ];
    let mut replay = cpu_block;
    replay.restore(&initial_checkpoint)?;
    let replay_a = replay.step(Qwen35LinearAttentionInputs {
        qkv_mixed: &token_a.qkv,
        conv_weights: &conv_weights,
        z: &token_a.z,
        decay_gate: &token_a.decay_gate,
        beta: &token_a.beta,
        norm_weight: &ssm_norm,
        epsilon,
    })?;
    let replay_exact = replay_a
        .iter()
        .zip(expected_a.iter())
        .all(|(left, right)| left.to_bits() == right.to_bits());
    let completion = CompletionReport {
        poll_iterations: completion_a.0 + completion_b.0 + completion_replay.0,
        nonblocking_poll_only: completion_a.2 && completion_b.2 && completion_replay.2,
        owner_epoch_proof: completion_a.1 && completion_b.1 && completion_replay.1,
    };
    let completion_boundaries = add_boundaries(
        add_boundaries(completion_boundary_a, completion_boundary_b),
        completion_boundary_replay,
    );
    let readback_boundaries = add_boundaries(
        add_boundaries(readback_boundary_a, readback_boundary_b),
        readback_boundary_replay,
    );
    stream.synchronize()?;
    let before_cleanup = cuda_device_allocation_snapshot();
    drop(qkv);
    drop(z);
    drop(conv_weights_buffer);
    drop(conv_history);
    drop(decay_gate);
    drop(beta);
    drop(state);
    drop(norm_weight);
    drop(activated);
    drop(recurrent);
    drop(output);
    drop(stream);
    let after_cleanup = cuda_device_allocation_snapshot();
    let resources = ResourceReport {
        setup_boundaries,
        setup_allocations,
        steady_boundaries,
        steady_allocations,
        completion_boundaries,
        readback_boundaries,
        final_allocation_delta: allocation_delta(before_cleanup, after_cleanup)?,
    };
    let cleanup_zero = resources.final_allocation_delta.live_allocations == 0
        && resources.final_allocation_delta.live_bytes == 0;
    let pass = numerical.iter().all(|item| item.pass)
        && replay_exact
        && gpu_replay_exact
        && completion.owner_epoch_proof
        && cleanup_zero;
    Ok(Qwen35RealLinearStepCudaReport {
        schema: QWEN35_REAL_LINEAR_STEP_CUDA_SCHEMA.to_owned(),
        status: if pass {
            "PASS_COMPONENT"
        } else {
            "FAIL_COMPONENT"
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
            output_elements: OUTPUT_ELEMENTS,
            conv_channels: CONV_CHANNELS,
            conv_history_elements: CONV_HISTORY_ELEMENTS,
            conv_weight_elements: CONV_WEIGHT_ELEMENTS,
            state_elements: STATE_ELEMENTS,
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
        resources,
        claims: ClaimsReport {
            real_weight_inputs_loaded: true,
            recurrent_step_executed: true,
            two_token_checkpoint_replay: gpu_replay_exact,
            complete_transformer_executed: false,
            decode_executed: false,
            semantic_parity_established: false,
            product_tps_permitted: false,
            default_cli_enabled: false,
        },
    })
}

fn validate_config(
    config: &Qwen35RealLinearStepCudaConfig,
) -> Result<(), Qwen35RealLinearStepCudaError> {
    if config.gguf_path.as_os_str().is_empty() || !config.gguf_path.is_file() {
        return Err(Qwen35RealLinearStepCudaError::InvalidConfiguration(
            "GGUF path",
        ));
    }
    if config.config_path.as_os_str().is_empty() || !config.config_path.is_file() {
        return Err(Qwen35RealLinearStepCudaError::InvalidConfiguration(
            "config path",
        ));
    }
    if config.device_ordinal < 0 || config.iterations == 0 || config.token_id >= VOCAB - 1 {
        return Err(Qwen35RealLinearStepCudaError::InvalidConfiguration(
            "runner bounds",
        ));
    }
    Ok(())
}

fn required_descriptors(
    manifest: &GgufManifest,
) -> Result<Vec<GgufTensor>, Qwen35RealLinearStepCudaError> {
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
            vec![HIDDEN as u64, OUTPUT_ELEMENTS as u64],
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
                Qwen35RealLinearStepCudaError::InvalidConfiguration("required tensor missing"),
            )?;
            if tensor.dimensions != dimensions || tensor.ggml_type != ggml_type {
                return Err(Qwen35RealLinearStepCudaError::InvalidConfiguration(
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
) -> Result<Vec<f32>, Qwen35RealLinearStepCudaError> {
    let row_bytes = HIDDEN * 2;
    let offset = token_id.checked_mul(row_bytes).ok_or(
        Qwen35RealLinearStepCudaError::InvalidConfiguration("embedding offset"),
    )?;
    let bytes = reader.read_tensor_slice(&descriptor.name, offset as u64, row_bytes as u64)?;
    let mut values = vec![0.0; HIDDEN];
    dequantize_bf16(&bytes, &mut values)?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(Qwen35RealLinearStepCudaError::InvalidConfiguration(
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
) -> Result<ProjectedToken, Qwen35RealLinearStepCudaError> {
    if hidden.len() != HIDDEN
        || attn_norm.len() != HIDDEN
        || dt_bias.len() != VALUE_HEADS
        || ssm_a.len() != VALUE_HEADS
    {
        return Err(Qwen35RealLinearStepCudaError::InvalidConfiguration(
            "projection input geometry",
        ));
    }
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
        || z.len() != OUTPUT_ELEMENTS
        || alpha_raw.len() != VALUE_HEADS
        || beta_raw.len() != VALUE_HEADS
    {
        return Err(Qwen35RealLinearStepCudaError::InvalidConfiguration(
            "projection output geometry",
        ));
    }
    let mut decay_gate = Vec::with_capacity(VALUE_HEADS);
    let mut beta = Vec::with_capacity(VALUE_HEADS);
    for index in 0..VALUE_HEADS {
        let alpha_projection = alpha_raw[index] + dt_bias[index];
        let softplus = if alpha_projection > 20.0 {
            alpha_projection
        } else if alpha_projection < -20.0 {
            alpha_projection.exp()
        } else {
            (1.0 + alpha_projection.exp()).ln()
        };
        decay_gate.push(softplus * ssm_a[index]);
        let value = beta_raw[index];
        beta.push(if value >= 0.0 {
            1.0 / (1.0 + (-value).exp())
        } else {
            let exponent = value.exp();
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
        return Err(Qwen35RealLinearStepCudaError::InvalidConfiguration(
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

fn convert_conv_layout(raw: &[f32]) -> Result<Vec<f32>, Qwen35RealLinearStepCudaError> {
    if raw.len() != CONV_WEIGHT_ELEMENTS {
        return Err(Qwen35RealLinearStepCudaError::InvalidConfiguration(
            "conv weight extent",
        ));
    }
    // GGUF is [kernel, channel] with dimension zero contiguous; the native
    // binding consumes [channel][tap].
    let mut converted = vec![0.0; raw.len()];
    for channel in 0..CONV_CHANNELS {
        for tap in 0..KERNEL_SIZE {
            converted[channel * KERNEL_SIZE + tap] = raw[tap + channel * KERNEL_SIZE];
        }
    }
    Ok(converted)
}

#[allow(clippy::too_many_arguments)]
fn bind_views(
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
) -> Result<CudaQwen35LinearStepArgsV1, Qwen35RealLinearStepCudaError> {
    let f32_view =
        |buffer: &OwnedCudaBuffer| -> Result<CudaF32View, Qwen35RealLinearStepCudaError> {
            Ok(CudaF32View::new(
                buffer.view(0, buffer.len_bytes())?,
                PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                buffer.len_bytes() / 4,
            )?)
        };
    Ok(CudaQwen35LinearStepArgsV1::bind(
        stream,
        f32_view(qkv)?,
        f32_view(z)?,
        f32_view(conv_weights)?,
        f32_view(conv_history)?,
        f32_view(decay_gate)?,
        f32_view(beta)?,
        f32_view(state)?,
        f32_view(norm_weight)?,
        f32_view(activated)?,
        f32_view(recurrent)?,
        f32_view(output)?,
        epsilon,
    )?)
}

#[allow(clippy::too_many_arguments)]
fn execute_and_readback(
    stream: &OwnedCudaStream,
    input: &CudaQwen35LinearStepArgsV1,
    _qkv: &OwnedCudaBuffer,
    _z: &OwnedCudaBuffer,
    _decay_gate: &OwnedCudaBuffer,
    _beta: &OwnedCudaBuffer,
    _token: &ProjectedToken,
    state: &OwnedCudaBuffer,
    output: &OwnedCudaBuffer,
    conv_history: &OwnedCudaBuffer,
) -> Result<
    (
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        (u64, bool, bool),
        BoundaryReport,
        BoundaryReport,
    ),
    Qwen35RealLinearStepCudaError,
> {
    let completion_boundary_before = cuda_host_boundary_snapshot();
    input.submit(stream)?;
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
            return Err(Qwen35RealLinearStepCudaError::CompletionPollLimit);
        }
        spin_loop();
    };
    let proof_ok = proof.covers_submission(receipt) && proof.belongs_to_stream(stream)?;
    if !proof_ok {
        return Err(Qwen35RealLinearStepCudaError::InvalidConfiguration(
            "owner epoch proof",
        ));
    }
    let completion_boundary_after = cuda_host_boundary_snapshot();
    let completion_boundaries =
        boundary_delta(completion_boundary_before, completion_boundary_after)?;
    let readback_boundary_before = cuda_host_boundary_snapshot();
    let mut output_bytes = vec![0_u8; OUTPUT_ELEMENTS * 4];
    let mut state_bytes = vec![0_u8; STATE_ELEMENTS * 4];
    let mut history_bytes = vec![0_u8; CONV_HISTORY_ELEMENTS * 4];
    output.copy_to_host_sync_with_submission_proof(stream, &mut output_bytes, receipt)?;
    state.copy_to_host_sync_with_submission_proof(stream, &mut state_bytes, receipt)?;
    conv_history.copy_to_host_sync_with_submission_proof(stream, &mut history_bytes, receipt)?;
    let readback_boundary_after = cuda_host_boundary_snapshot();
    Ok((
        f32_vec(&output_bytes),
        f32_vec(&state_bytes),
        f32_vec(&history_bytes),
        (polls, true, proof_ok),
        completion_boundaries,
        boundary_delta(readback_boundary_before, readback_boundary_after)?,
    ))
}

fn compare(
    name: &str,
    actual: &[f32],
    expected: &[f32],
    abs_tol: f32,
    rel_tol: f32,
) -> NumericalReport {
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    let mut finite = 0_usize;
    for (left, right) in actual.iter().zip(expected.iter()) {
        if left.is_finite() {
            finite += 1;
        }
        let abs = (left - right).abs();
        let rel = abs / right.abs().max(1.0e-6);
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
    }
    NumericalReport {
        name: name.to_owned(),
        elements: actual.len(),
        finite_count: finite,
        max_abs_diff: max_abs,
        max_rel_diff: max_rel,
        absolute_tolerance: abs_tol,
        relative_tolerance: rel_tol,
        output_sha256: sha256_f32_bits(actual),
        expected_sha256: sha256_f32_bits(expected),
        pass: actual.len() == expected.len()
            && finite == actual.len()
            && max_abs
                <= abs_tol.max(rel_tol * expected.iter().map(|v| v.abs()).fold(0.0, f32::max)),
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
) -> Result<AllocationReport, Qwen35RealLinearStepCudaError> {
    Ok(AllocationReport {
        successful_allocations: later
            .successful_allocations
            .checked_sub(earlier.successful_allocations)
            .ok_or(Qwen35RealLinearStepCudaError::InvalidConfiguration(
                "allocation counters",
            ))?,
        successful_frees: later
            .successful_frees
            .checked_sub(earlier.successful_frees)
            .ok_or(Qwen35RealLinearStepCudaError::InvalidConfiguration(
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
) -> Result<BoundaryReport, Qwen35RealLinearStepCudaError> {
    let delta = later.checked_delta_since(earlier).ok_or(
        Qwen35RealLinearStepCudaError::InvalidConfiguration("host counters"),
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
