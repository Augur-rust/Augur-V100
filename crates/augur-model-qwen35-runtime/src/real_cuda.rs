#![cfg(feature = "cuda")]

//! Bounded real-weight CUDA evidence for the Qwen35 lane.
//!
//! This module intentionally stops at a small BF16 embedding slice.  It is a
//! device/data-boundary probe, not a transformer runner: it never publishes a
//! token and it cannot grant decode, semantic, or throughput claims.

use std::{error::Error, fmt, hint::spin_loop, path::PathBuf, time::Instant};

use augur_backend::{
    cuda_device_allocation_snapshot, cuda_device_properties, cuda_host_boundary_snapshot,
    CudaBf16F32MatvecArgsV1, CudaBf16TokenRowGatherArgsV1, CudaBf16WeightViewV1, CudaF32View,
    CudaRuntimeError, OwnedCudaBuffer, OwnedCudaEvent, OwnedCudaStream,
};
use augur_core::PhysicalLayoutId;
use augur_model_qwen35::{
    dequantize_bf16, GgufAuditOptions, GgufManifest, Qwen35Error, GGML_TYPE_BF16,
};
use serde::Serialize;
use sha2::{Digest, Sha256};

pub const QWEN35_REAL_CUDA_SLICE_SCHEMA: &str = "crystal-quant.qwen35.real-bf16-cuda-slice.v1";

const EMBEDDING_TENSOR: &str = "token_embd.weight";
const BF16_BYTES_PER_ELEMENT: usize = 2;
const MAX_ROWS: usize = 32;
const MAX_COMPLETION_POLLS: u64 = 1_000_000;

/// Configuration for the bounded real-weight slice probe.
#[derive(Clone, Debug)]
pub struct Qwen35RealCudaSliceConfig {
    pub gguf_path: PathBuf,
    pub device_ordinal: i32,
    pub token_id: usize,
    pub rows: usize,
    pub warmup: usize,
    pub iterations: usize,
    pub source_commit: String,
}

impl Default for Qwen35RealCudaSliceConfig {
    fn default() -> Self {
        Self {
            gguf_path: PathBuf::new(),
            device_ordinal: 0,
            token_id: 17,
            rows: 4,
            warmup: 10,
            iterations: 100,
            source_commit: "UNBOUND_COMPONENT_SOURCE".to_owned(),
        }
    }
}

#[derive(Debug)]
pub enum Qwen35RealCudaSliceError {
    Runtime(CudaRuntimeError),
    Model(Qwen35Error),
    InvalidConfiguration(&'static str),
    CompletionPollLimit,
}

impl fmt::Display for Qwen35RealCudaSliceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Runtime(error) => error.fmt(formatter),
            Self::Model(error) => error.fmt(formatter),
            Self::InvalidConfiguration(field) => {
                write!(
                    formatter,
                    "invalid Qwen real CUDA slice configuration: {field}"
                )
            }
            Self::CompletionPollLimit => {
                formatter.write_str("CUDA completion polling exceeded its bound")
            }
        }
    }
}

impl Error for Qwen35RealCudaSliceError {}

impl From<CudaRuntimeError> for Qwen35RealCudaSliceError {
    fn from(error: CudaRuntimeError) -> Self {
        Self::Runtime(error)
    }
}

impl From<Qwen35Error> for Qwen35RealCudaSliceError {
    fn from(error: Qwen35Error) -> Self {
        Self::Model(error)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Qwen35RealCudaSliceReport {
    pub schema: String,
    pub status: String,
    pub component_only: bool,
    pub source_commit: String,
    pub device: RealCudaDeviceReport,
    pub source: RealCudaSourceReport,
    pub shape: RealCudaShapeReport,
    pub timing: RealCudaTimingReport,
    pub completion: RealCudaCompletionReport,
    pub numerical: RealCudaNumericalReport,
    pub matvec: RealCudaMatvecReport,
    pub resources: RealCudaResourceReport,
    pub claims: RealCudaClaimReport,
}

#[derive(Clone, Debug, Serialize)]
pub struct RealCudaDeviceReport {
    pub ordinal: i32,
    pub name: String,
    pub compute_capability: [u16; 2],
    pub total_global_memory_bytes: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct RealCudaSourceReport {
    pub path: String,
    pub file_sha256: String,
    pub tensor_name: String,
    pub tensor_dimensions: Vec<u64>,
    pub ggml_type: u32,
    pub base_token_id: usize,
    pub requested_token_id: usize,
    pub local_token_id: usize,
    pub rows: usize,
    pub slice_offset_bytes: u64,
    pub slice_bytes: usize,
    pub slice_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct RealCudaShapeReport {
    pub rows: usize,
    pub columns: usize,
    pub row_bytes: usize,
    pub output_elements: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct RealCudaTimingReport {
    pub cuda_event_ms: f32,
    pub batch_wall_ms: f64,
    pub per_submission_event_ms: f64,
    pub matvec_cuda_event_ms: f32,
    pub matvec_batch_wall_ms: f64,
    pub matvec_per_submission_event_ms: f64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct RealCudaCompletionReport {
    pub poll_iterations: u64,
    pub matvec_poll_iterations: u64,
    pub nonblocking_poll_only: bool,
    pub completion_proof_required: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct RealCudaNumericalReport {
    pub finite_count: usize,
    pub expected_count: usize,
    pub max_abs_diff: f32,
    pub tolerance: f32,
    pub output_sha256: String,
    pub expected_sha256: String,
    pub pass: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct RealCudaMatvecReport {
    pub rows: usize,
    pub columns: usize,
    pub input_sha256: String,
    pub finite_count: usize,
    pub expected_count: usize,
    pub max_abs_diff: f32,
    pub tolerance: f32,
    pub output_sha256: String,
    pub expected_sha256: String,
    pub pass: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct RealCudaResourceReport {
    pub setup_allocations: AllocationDeltaReport,
    pub steady_allocations: AllocationDeltaReport,
    pub steady_matvec_allocations: AllocationDeltaReport,
    pub setup_host_boundaries: HostBoundaryDeltaReport,
    pub steady_host_boundaries: HostBoundaryDeltaReport,
    pub steady_matvec_host_boundaries: HostBoundaryDeltaReport,
    pub completion_poll_host_boundaries: HostBoundaryDeltaReport,
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
pub struct RealCudaClaimReport {
    pub real_weight_slice_uploaded: bool,
    pub complete_transformer_executed: bool,
    pub decode_executed: bool,
    pub semantic_parity_established: bool,
    pub product_tps_permitted: bool,
    pub default_cli_enabled: bool,
}

/// Upload and gather a small, manifest-bound real BF16 embedding window.
pub fn run_qwen35_real_bf16_cuda_slice(
    config: &Qwen35RealCudaSliceConfig,
) -> Result<Qwen35RealCudaSliceReport, Qwen35RealCudaSliceError> {
    validate_config(config)?;
    let manifest = GgufManifest::inspect(&config.gguf_path, GgufAuditOptions::default())?;
    let tensor = manifest
        .tensor(EMBEDDING_TENSOR)
        .ok_or(Qwen35RealCudaSliceError::InvalidConfiguration(
            "token_embd.weight is missing",
        ))?
        .clone();
    if tensor.ggml_type != GGML_TYPE_BF16 || tensor.dimensions.len() != 2 {
        return Err(Qwen35RealCudaSliceError::InvalidConfiguration(
            "token_embd.weight must be rank-2 BF16",
        ));
    }
    let columns = usize::try_from(tensor.dimensions[0])
        .map_err(|_| Qwen35RealCudaSliceError::InvalidConfiguration("embedding columns"))?;
    let vocabulary = usize::try_from(tensor.dimensions[1])
        .map_err(|_| Qwen35RealCudaSliceError::InvalidConfiguration("embedding rows"))?;
    if columns == 0 || vocabulary == 0 || config.token_id >= vocabulary || config.rows > vocabulary
    {
        return Err(Qwen35RealCudaSliceError::InvalidConfiguration(
            "embedding geometry or token_id",
        ));
    }
    let row_bytes = columns.checked_mul(BF16_BYTES_PER_ELEMENT).ok_or(
        Qwen35RealCudaSliceError::InvalidConfiguration("row byte size"),
    )?;
    let base_token_id = choose_window_base(config.token_id, config.rows, vocabulary);
    let local_token_id = config.token_id - base_token_id;
    let slice_offset = base_token_id.checked_mul(row_bytes).ok_or(
        Qwen35RealCudaSliceError::InvalidConfiguration("slice offset"),
    )?;
    let slice_bytes_len = config.rows.checked_mul(row_bytes).ok_or(
        Qwen35RealCudaSliceError::InvalidConfiguration("slice byte size"),
    )?;
    let mut reader = manifest.open_tensor_reader(&config.gguf_path)?;
    let slice_bytes = reader.read_tensor_slice(
        EMBEDDING_TENSOR,
        u64::try_from(slice_offset)
            .map_err(|_| Qwen35RealCudaSliceError::InvalidConfiguration("slice offset"))?,
        u64::try_from(slice_bytes_len)
            .map_err(|_| Qwen35RealCudaSliceError::InvalidConfiguration("slice byte size"))?,
    )?;
    if slice_bytes.len() != slice_bytes_len {
        return Err(Qwen35RealCudaSliceError::InvalidConfiguration(
            "slice reader length",
        ));
    }
    let mut decoded = vec![0.0_f32; config.rows * columns];
    dequantize_bf16(&slice_bytes, &mut decoded)?;
    let expected_start = local_token_id.checked_mul(columns).ok_or(
        Qwen35RealCudaSliceError::InvalidConfiguration("expected row offset"),
    )?;
    let expected = decoded[expected_start..expected_start + columns].to_vec();
    let matvec_input_values = make_matvec_input(columns);
    let matvec_expected = matvec_reference(&decoded, config.rows, columns, &matvec_input_values);
    let matvec_input_bytes = f32_bytes(&matvec_input_values);

    let device = cuda_device_properties(config.device_ordinal)?;
    let stream = OwnedCudaStream::create(config.device_ordinal)?;
    let allocation_start = cuda_device_allocation_snapshot();
    let boundary_start = cuda_host_boundary_snapshot();
    let embedding = OwnedCudaBuffer::allocate(config.device_ordinal, slice_bytes.len())?;
    let token = OwnedCudaBuffer::allocate(config.device_ordinal, core::mem::size_of::<u32>())?;
    let output =
        OwnedCudaBuffer::allocate(config.device_ordinal, columns * core::mem::size_of::<f32>())?;
    let matvec_input = OwnedCudaBuffer::allocate(config.device_ordinal, matvec_input_bytes.len())?;
    let matvec_output = OwnedCudaBuffer::allocate(
        config.device_ordinal,
        config.rows * core::mem::size_of::<f32>(),
    )?;
    let local_token_u32 = u32::try_from(local_token_id)
        .map_err(|_| Qwen35RealCudaSliceError::InvalidConfiguration("local token id"))?;
    embedding.copy_from_host_sync(&stream, &slice_bytes)?;
    token.copy_from_host_sync(&stream, &local_token_u32.to_le_bytes())?;
    matvec_input.copy_from_host_sync(&stream, &matvec_input_bytes)?;
    let args = CudaBf16TokenRowGatherArgsV1::bind(
        &stream,
        CudaBf16WeightViewV1::new(
            embedding.view(0, slice_bytes.len())?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_BF16_CUDA_V1,
            config.rows,
            columns,
        )?,
        token.view(0, core::mem::size_of::<u32>())?,
        CudaF32View::new(
            output.view(0, columns * core::mem::size_of::<f32>())?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            columns,
        )?,
    )?;
    let matvec_args = CudaBf16F32MatvecArgsV1::bind(
        &stream,
        CudaBf16WeightViewV1::new(
            embedding.view(0, slice_bytes.len())?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_BF16_CUDA_V1,
            config.rows,
            columns,
        )?,
        CudaF32View::new(
            matvec_input.view(0, matvec_input_bytes.len())?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            columns,
        )?,
        CudaF32View::new(
            matvec_output.view(0, config.rows * core::mem::size_of::<f32>())?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            config.rows,
        )?,
    )?;
    let setup_allocations = delta_allocations(allocation_start, cuda_device_allocation_snapshot())?;
    let setup_host_boundaries = delta_boundaries(boundary_start, cuda_host_boundary_snapshot())?;

    for _ in 0..config.warmup {
        args.submit_for_plan(&stream)?;
    }
    stream.synchronize()?;
    for _ in 0..config.warmup {
        matvec_args.submit_for_plan(&stream)?;
    }
    stream.synchronize()?;

    let steady_allocation_start = cuda_device_allocation_snapshot();
    let steady_boundary_start = cuda_host_boundary_snapshot();
    let started = OwnedCudaEvent::create(config.device_ordinal)?;
    let stopped = OwnedCudaEvent::create(config.device_ordinal)?;
    let wall_start = Instant::now();
    started.record(&stream)?;
    for _ in 0..config.iterations {
        args.submit_for_plan(&stream)?;
    }
    stopped.record(&stream)?;
    stopped.synchronize()?;
    let batch_wall_ms = wall_start.elapsed().as_secs_f64() * 1_000.0;
    let cuda_event_ms = started.elapsed_ms(&stopped)?;
    let steady_allocations =
        delta_allocations(steady_allocation_start, cuda_device_allocation_snapshot())?;
    let steady_host_boundaries =
        delta_boundaries(steady_boundary_start, cuda_host_boundary_snapshot())?;

    let steady_matvec_allocation_start = cuda_device_allocation_snapshot();
    let steady_matvec_boundary_start = cuda_host_boundary_snapshot();
    let matvec_started = OwnedCudaEvent::create(config.device_ordinal)?;
    let matvec_stopped = OwnedCudaEvent::create(config.device_ordinal)?;
    let matvec_wall_start = Instant::now();
    matvec_started.record(&stream)?;
    for _ in 0..config.iterations {
        matvec_args.submit_for_plan(&stream)?;
    }
    matvec_stopped.record(&stream)?;
    matvec_stopped.synchronize()?;
    let matvec_batch_wall_ms = matvec_wall_start.elapsed().as_secs_f64() * 1_000.0;
    let matvec_cuda_event_ms = matvec_started.elapsed_ms(&matvec_stopped)?;
    let steady_matvec_allocations = delta_allocations(
        steady_matvec_allocation_start,
        cuda_device_allocation_snapshot(),
    )?;
    let steady_matvec_host_boundaries =
        delta_boundaries(steady_matvec_boundary_start, cuda_host_boundary_snapshot())?;

    let completion_event = OwnedCudaEvent::create(config.device_ordinal)?;
    let completion_boundary_start = cuda_host_boundary_snapshot();
    let poll_iterations =
        submit_and_poll(&stream, &completion_event, || args.submit_for_plan(&stream))?;
    let matvec_poll_iterations = submit_and_poll(&stream, &completion_event, || {
        matvec_args.submit_for_plan(&stream)
    })?;
    let completion_poll_host_boundaries =
        delta_boundaries(completion_boundary_start, cuda_host_boundary_snapshot())?;

    let mut output_bytes = vec![0_u8; columns * core::mem::size_of::<f32>()];
    output.copy_to_host_sync(&stream, &mut output_bytes)?;
    let mut matvec_output_bytes = vec![0_u8; config.rows * core::mem::size_of::<f32>()];
    matvec_output.copy_to_host_sync(&stream, &mut matvec_output_bytes)?;
    let actual = f32_vec_from_bytes(&output_bytes);
    let matvec_actual = f32_vec_from_bytes(&matvec_output_bytes);
    let tolerance = 2.0e-5_f32;
    let gather_max_abs_diff = max_abs_diff(&actual, &expected);
    let pass = finite_and_within(&actual, &expected, tolerance);
    let output_sha256 = sha256_hex(&output_bytes);
    let expected_sha256 = sha256_f32_bits(&expected);
    let matvec_max_abs_diff = max_abs_diff(&matvec_actual, &matvec_expected);
    let matvec_tolerance = 5.0e-4_f32;
    let matvec_pass = finite_and_within(&matvec_actual, &matvec_expected, matvec_tolerance);
    let matvec_output_sha256 = sha256_hex(&matvec_output_bytes);
    let matvec_expected_sha256 = sha256_f32_bits(&matvec_expected);
    let matvec_input_sha256 = sha256_hex(&matvec_input_bytes);
    let slice_sha256 = sha256_hex(&slice_bytes);

    drop(completion_event);
    drop(stopped);
    drop(started);
    drop(matvec_stopped);
    drop(matvec_started);
    drop(matvec_output);
    drop(matvec_input);
    drop(output);
    drop(token);
    drop(embedding);
    drop(stream);
    let final_allocation_delta_after_cleanup =
        delta_allocations(allocation_start, cuda_device_allocation_snapshot())?;

    Ok(Qwen35RealCudaSliceReport {
        schema: QWEN35_REAL_CUDA_SLICE_SCHEMA.to_owned(),
        status: if pass && matvec_pass {
            "PASS_COMPONENT_REAL_BF16_CUDA_SLICE".to_owned()
        } else {
            "FAIL_COMPONENT_REAL_BF16_CUDA_SLICE_NUMERICAL".to_owned()
        },
        component_only: true,
        source_commit: config.source_commit.clone(),
        device: RealCudaDeviceReport {
            ordinal: device.ordinal,
            name: device.name,
            compute_capability: [
                device.compute_capability.major,
                device.compute_capability.minor,
            ],
            total_global_memory_bytes: device.total_global_memory_bytes,
        },
        source: RealCudaSourceReport {
            path: config.gguf_path.display().to_string(),
            file_sha256: manifest.file_sha256.clone().ok_or(
                Qwen35RealCudaSliceError::InvalidConfiguration("GGUF file hash"),
            )?,
            tensor_name: EMBEDDING_TENSOR.to_owned(),
            tensor_dimensions: tensor.dimensions,
            ggml_type: tensor.ggml_type,
            base_token_id,
            requested_token_id: config.token_id,
            local_token_id,
            rows: config.rows,
            slice_offset_bytes: u64::try_from(slice_offset)
                .map_err(|_| Qwen35RealCudaSliceError::InvalidConfiguration("slice offset"))?,
            slice_bytes: slice_bytes.len(),
            slice_sha256,
        },
        shape: RealCudaShapeReport {
            rows: config.rows,
            columns,
            row_bytes,
            output_elements: actual.len(),
        },
        timing: RealCudaTimingReport {
            cuda_event_ms,
            batch_wall_ms,
            per_submission_event_ms: f64::from(cuda_event_ms) / config.iterations as f64,
            matvec_cuda_event_ms,
            matvec_batch_wall_ms,
            matvec_per_submission_event_ms: f64::from(matvec_cuda_event_ms)
                / config.iterations as f64,
        },
        completion: RealCudaCompletionReport {
            poll_iterations,
            matvec_poll_iterations,
            nonblocking_poll_only: true,
            completion_proof_required: true,
        },
        numerical: RealCudaNumericalReport {
            finite_count: actual.iter().filter(|value| value.is_finite()).count(),
            expected_count: expected.len(),
            max_abs_diff: gather_max_abs_diff,
            tolerance,
            output_sha256,
            expected_sha256,
            pass,
        },
        matvec: RealCudaMatvecReport {
            rows: config.rows,
            columns,
            input_sha256: matvec_input_sha256,
            finite_count: matvec_actual
                .iter()
                .filter(|value| value.is_finite())
                .count(),
            expected_count: matvec_expected.len(),
            max_abs_diff: matvec_max_abs_diff,
            tolerance: matvec_tolerance,
            output_sha256: matvec_output_sha256,
            expected_sha256: matvec_expected_sha256,
            pass: matvec_pass,
        },
        resources: RealCudaResourceReport {
            setup_allocations,
            steady_allocations,
            steady_matvec_allocations,
            setup_host_boundaries,
            steady_host_boundaries,
            steady_matvec_host_boundaries,
            completion_poll_host_boundaries,
            final_allocation_delta_after_cleanup,
        },
        claims: RealCudaClaimReport {
            real_weight_slice_uploaded: true,
            complete_transformer_executed: false,
            decode_executed: false,
            semantic_parity_established: false,
            product_tps_permitted: false,
            default_cli_enabled: false,
        },
    })
}

fn validate_config(config: &Qwen35RealCudaSliceConfig) -> Result<(), Qwen35RealCudaSliceError> {
    if config.gguf_path.as_os_str().is_empty() {
        return Err(Qwen35RealCudaSliceError::InvalidConfiguration("gguf_path"));
    }
    if config.device_ordinal < 0 {
        return Err(Qwen35RealCudaSliceError::InvalidConfiguration(
            "device_ordinal",
        ));
    }
    if config.rows == 0 || config.rows > MAX_ROWS {
        return Err(Qwen35RealCudaSliceError::InvalidConfiguration("rows"));
    }
    if config.iterations == 0 {
        return Err(Qwen35RealCudaSliceError::InvalidConfiguration("iterations"));
    }
    Ok(())
}

fn choose_window_base(token_id: usize, rows: usize, vocabulary: usize) -> usize {
    let half = rows / 2;
    let centered = token_id.saturating_sub(half);
    centered.min(vocabulary.saturating_sub(rows))
}

fn submit_and_poll(
    stream: &OwnedCudaStream,
    event: &OwnedCudaEvent,
    submit: impl FnOnce() -> Result<(), CudaRuntimeError>,
) -> Result<u64, Qwen35RealCudaSliceError> {
    submit()?;
    let receipt = stream.submission_receipt()?;
    event.record_for_submission(stream, receipt)?;
    let mut polls = 0_u64;
    loop {
        if stream.query_event_completion(event, receipt)?.is_some() {
            return Ok(polls);
        }
        polls = polls.saturating_add(1);
        if polls > MAX_COMPLETION_POLLS {
            return Err(Qwen35RealCudaSliceError::CompletionPollLimit);
        }
        spin_loop();
    }
}

fn delta_allocations(
    before: augur_backend::CudaDeviceAllocationSnapshot,
    after: augur_backend::CudaDeviceAllocationSnapshot,
) -> Result<AllocationDeltaReport, Qwen35RealCudaSliceError> {
    Ok(AllocationDeltaReport {
        successful_allocations: after
            .successful_allocations
            .checked_sub(before.successful_allocations)
            .ok_or(Qwen35RealCudaSliceError::InvalidConfiguration(
                "allocation counters",
            ))?,
        successful_frees: after
            .successful_frees
            .checked_sub(before.successful_frees)
            .ok_or(Qwen35RealCudaSliceError::InvalidConfiguration(
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
) -> Result<HostBoundaryDeltaReport, Qwen35RealCudaSliceError> {
    let delta =
        after
            .checked_delta_since(before)
            .ok_or(Qwen35RealCudaSliceError::InvalidConfiguration(
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

fn f32_vec_from_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(core::mem::size_of::<f32>())
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("f32 chunk")))
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn make_matvec_input(columns: usize) -> Vec<f32> {
    (0..columns)
        .map(|index| ((index % 53) as f32 - 26.0) / 37.0)
        .collect()
}

fn matvec_reference(decoded: &[f32], rows: usize, columns: usize, input: &[f32]) -> Vec<f32> {
    (0..rows)
        .map(|row| {
            let start = row * columns;
            let end = start + columns;
            decoded[start..end]
                .iter()
                .zip(input)
                .map(|(weight, value)| f64::from(*weight) * f64::from(*value))
                .sum::<f64>() as f32
        })
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

fn max_abs_diff(actual: &[f32], expected: &[f32]) -> f32 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max)
}

fn finite_and_within(actual: &[f32], expected: &[f32], tolerance: f32) -> bool {
    actual
        .iter()
        .zip(expected)
        .all(|(actual, expected)| actual.is_finite() && (*actual - *expected).abs() <= tolerance)
}

#[cfg(test)]
mod tests {
    use super::{choose_window_base, make_matvec_input, matvec_reference};

    #[test]
    fn window_stays_inside_vocabulary_and_centers_requested_row() {
        assert_eq!(choose_window_base(17, 4, 100), 15);
        assert_eq!(choose_window_base(1, 4, 100), 0);
        assert_eq!(choose_window_base(99, 4, 100), 96);
    }

    #[test]
    fn matvec_reference_uses_row_major_window() {
        let input = make_matvec_input(3);
        let decoded = [1.0_f32, 2.0, 3.0, -1.0, 0.5, 2.0];
        let output = matvec_reference(&decoded, 2, 3, &input);
        assert_eq!(output.len(), 2);
        assert!(output.iter().all(|value| value.is_finite()));
    }
}
