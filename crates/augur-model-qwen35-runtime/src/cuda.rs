#![cfg(feature = "cuda")]

use std::{error::Error, fmt, hint::spin_loop, time::Instant};

use augur_backend::{
    cuda_device_allocation_snapshot, cuda_device_properties, cuda_host_boundary_snapshot,
    CudaBf16TokenRowGatherArgsV1, CudaBf16WeightViewV1, CudaF32View, CudaF8E4M3F32MatvecArgsV1,
    CudaF8E4M3WeightViewV1, CudaRuntimeError, OwnedCudaBuffer, OwnedCudaEvent, OwnedCudaStream,
};
use augur_core::PhysicalLayoutId;
use serde::Serialize;
use sha2::{Digest, Sha256};

pub const QWEN35_CUDA_COMPONENT_SCHEMA: &str = "crystal-quant.qwen35.cuda-component.v1";

const EMBEDDING_ROWS: usize = 257;
const EMBEDDING_COLS: usize = 256;
const F8_ROWS: usize = 256;
const F8_COLS: usize = 256;
const MAX_COMPLETION_POLLS: u64 = 1_000_000;

/// Configuration for the isolated component probe.
///
/// `warmup` and `iterations` affect only the bounded component loop.  They do
/// not imply a model-level decode sample count or grant a performance claim.
#[derive(Clone, Debug)]
pub struct Qwen35CudaProbeConfig {
    pub device_ordinal: i32,
    pub warmup: usize,
    pub iterations: usize,
    pub source_commit: String,
}

impl Default for Qwen35CudaProbeConfig {
    fn default() -> Self {
        Self {
            device_ordinal: 0,
            warmup: 10,
            iterations: 100,
            source_commit: "UNBOUND_COMPONENT_SOURCE".to_owned(),
        }
    }
}

#[derive(Debug)]
pub enum Qwen35CudaComponentError {
    Runtime(CudaRuntimeError),
    InvalidConfiguration(&'static str),
    CompletionPollLimit,
}

impl fmt::Display for Qwen35CudaComponentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Runtime(error) => error.fmt(formatter),
            Self::InvalidConfiguration(field) => {
                write!(
                    formatter,
                    "invalid Qwen CUDA component configuration: {field}"
                )
            }
            Self::CompletionPollLimit => {
                formatter.write_str("CUDA completion polling exceeded its bound")
            }
        }
    }
}

impl Error for Qwen35CudaComponentError {}

impl From<CudaRuntimeError> for Qwen35CudaComponentError {
    fn from(error: CudaRuntimeError) -> Self {
        Self::Runtime(error)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Qwen35CudaComponentReport {
    pub schema: String,
    pub status: String,
    pub component_only: bool,
    pub source_commit: String,
    pub device: DeviceReport,
    pub shape: ShapeReport,
    pub timing: TimingReport,
    pub completion: CompletionReport,
    pub numerical: NumericalReport,
    pub resources: ResourceReport,
    pub claims: ClaimReport,
}

#[derive(Clone, Debug, Serialize)]
pub struct DeviceReport {
    pub ordinal: i32,
    pub name: String,
    pub compute_capability: [u16; 2],
    pub total_global_memory_bytes: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ShapeReport {
    pub embedding_rows: usize,
    pub embedding_cols: usize,
    pub embedding_token_id: u32,
    pub f8_rows: usize,
    pub f8_cols: usize,
    pub f8_scale_count: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct TimingReport {
    pub gather_cuda_event_ms: f32,
    pub f8_cuda_event_ms: f32,
    pub gather_batch_wall_ms: f64,
    pub f8_batch_wall_ms: f64,
    pub gather_per_submission_event_ms: f64,
    pub f8_per_submission_event_ms: f64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct CompletionReport {
    pub gather_poll_iterations: u64,
    pub f8_poll_iterations: u64,
    pub nonblocking_poll_only: bool,
    pub completion_proof_required: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct NumericalReport {
    pub gather_finite_count: usize,
    pub gather_expected_count: usize,
    pub gather_max_abs_diff: f32,
    pub f8_finite_count: usize,
    pub f8_expected_count: usize,
    pub f8_max_abs_diff: f32,
    pub gather_output_sha256: String,
    pub f8_output_sha256: String,
    pub tolerance: f32,
    pub gather_pass: bool,
    pub f8_pass: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct ResourceReport {
    pub setup_allocations: AllocationDeltaReport,
    pub steady_gather_allocations: AllocationDeltaReport,
    pub steady_f8_allocations: AllocationDeltaReport,
    pub setup_host_boundaries: HostBoundaryDeltaReport,
    pub steady_host_boundaries: HostBoundaryDeltaReport,
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
pub struct ClaimReport {
    pub complete_transformer_executed: bool,
    pub decode_executed: bool,
    pub semantic_parity_established: bool,
    pub product_tps_permitted: bool,
    pub default_cli_enabled: bool,
}

/// Run a deterministic BF16 gather and FP8 E4M3 matvec on a real CUDA device.
///
/// The probe deliberately uses small synthetic buffers.  It proves that the
/// existing typed CUDA ownership/launch boundary can be consumed by the Qwen
/// lane; it does not execute Qwen's 64-layer transformer or publish tokens.
pub fn run_qwen35_cuda_component(
    config: &Qwen35CudaProbeConfig,
) -> Result<Qwen35CudaComponentReport, Qwen35CudaComponentError> {
    if config.device_ordinal < 0 {
        return Err(Qwen35CudaComponentError::InvalidConfiguration(
            "device_ordinal",
        ));
    }
    if config.iterations == 0 {
        return Err(Qwen35CudaComponentError::InvalidConfiguration(
            "iterations must be non-zero",
        ));
    }

    let device = cuda_device_properties(config.device_ordinal)?;
    let stream = OwnedCudaStream::create(config.device_ordinal)?;
    let allocation_start = cuda_device_allocation_snapshot();
    let boundary_start = cuda_host_boundary_snapshot();

    let embedding_bytes = make_bf16_embedding_bytes(EMBEDDING_ROWS, EMBEDDING_COLS);
    let embedding = OwnedCudaBuffer::allocate(config.device_ordinal, embedding_bytes.len())?;
    let token_id = u32::try_from(EMBEDDING_ROWS / 2)
        .map_err(|_| Qwen35CudaComponentError::InvalidConfiguration("embedding rows"))?;
    let token = OwnedCudaBuffer::allocate(config.device_ordinal, core::mem::size_of::<u32>())?;
    let gather_output = OwnedCudaBuffer::allocate(
        config.device_ordinal,
        EMBEDDING_COLS * core::mem::size_of::<f32>(),
    )?;

    let f8_weights_bytes = make_f8_weights_bytes(F8_ROWS, F8_COLS);
    let f8_weights = OwnedCudaBuffer::allocate(config.device_ordinal, f8_weights_bytes.len())?;
    let f8_scale_values = make_f8_scales(F8_ROWS, F8_COLS);
    let f8_scales = OwnedCudaBuffer::allocate(
        config.device_ordinal,
        f8_scale_values.len() * core::mem::size_of::<f32>(),
    )?;
    let f8_input_values = make_f32_input(F8_COLS);
    let f8_input = OwnedCudaBuffer::allocate(
        config.device_ordinal,
        f8_input_values.len() * core::mem::size_of::<f32>(),
    )?;
    let f8_output =
        OwnedCudaBuffer::allocate(config.device_ordinal, F8_ROWS * core::mem::size_of::<f32>())?;

    embedding.copy_from_host_sync(&stream, &embedding_bytes)?;
    token.copy_from_host_sync(&stream, &token_id.to_le_bytes())?;
    f8_weights.copy_from_host_sync(&stream, &f8_weights_bytes)?;
    f8_scales.copy_from_host_sync(&stream, &f32_bytes(&f8_scale_values))?;
    f8_input.copy_from_host_sync(&stream, &f32_bytes(&f8_input_values))?;

    let gather_args = CudaBf16TokenRowGatherArgsV1::bind(
        &stream,
        CudaBf16WeightViewV1::new(
            embedding.view(0, embedding_bytes.len())?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_BF16_CUDA_V1,
            EMBEDDING_ROWS,
            EMBEDDING_COLS,
        )?,
        token.view(0, core::mem::size_of::<u32>())?,
        CudaF32View::new(
            gather_output.view(0, EMBEDDING_COLS * core::mem::size_of::<f32>())?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            EMBEDDING_COLS,
        )?,
    )?;
    let f8_args = CudaF8E4M3F32MatvecArgsV1::bind(
        &stream,
        CudaF8E4M3WeightViewV1::new(
            f8_weights.view(0, f8_weights_bytes.len())?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F8E4M3_CUDA_V1,
            F8_ROWS,
            F8_COLS,
        )?,
        CudaF32View::new(
            f8_scales.view(0, f8_scale_values.len() * core::mem::size_of::<f32>())?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            f8_scale_values.len(),
        )?,
        CudaF32View::new(
            f8_input.view(0, f8_input_values.len() * core::mem::size_of::<f32>())?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            F8_COLS,
        )?,
        CudaF32View::new(
            f8_output.view(0, F8_ROWS * core::mem::size_of::<f32>())?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            F8_ROWS,
        )?,
    )?;

    let setup_allocations = delta_allocations(allocation_start, cuda_device_allocation_snapshot())?;
    let setup_host_boundaries = delta_boundaries(boundary_start, cuda_host_boundary_snapshot())?;

    for _ in 0..config.warmup {
        gather_args.submit_for_plan(&stream)?;
    }
    stream.synchronize()?;
    for _ in 0..config.warmup {
        f8_args.submit_bitdecode_for_plan(&stream)?;
    }
    stream.synchronize()?;

    let steady_gather_allocation_start = cuda_device_allocation_snapshot();
    let steady_gather_boundary_start = cuda_host_boundary_snapshot();
    let gather_started = OwnedCudaEvent::create(config.device_ordinal)?;
    let gather_stopped = OwnedCudaEvent::create(config.device_ordinal)?;
    let gather_wall_start = Instant::now();
    gather_started.record(&stream)?;
    for _ in 0..config.iterations {
        gather_args.submit_for_plan(&stream)?;
    }
    gather_stopped.record(&stream)?;
    gather_stopped.synchronize()?;
    let gather_batch_wall_ms = gather_wall_start.elapsed().as_secs_f64() * 1_000.0;
    let gather_cuda_event_ms = gather_started.elapsed_ms(&gather_stopped)?;
    let steady_gather_allocations = delta_allocations(
        steady_gather_allocation_start,
        cuda_device_allocation_snapshot(),
    )?;
    let steady_gather_boundaries =
        delta_boundaries(steady_gather_boundary_start, cuda_host_boundary_snapshot())?;

    let steady_f8_allocation_start = cuda_device_allocation_snapshot();
    let steady_f8_boundary_start = cuda_host_boundary_snapshot();
    let f8_started = OwnedCudaEvent::create(config.device_ordinal)?;
    let f8_stopped = OwnedCudaEvent::create(config.device_ordinal)?;
    let f8_wall_start = Instant::now();
    f8_started.record(&stream)?;
    for _ in 0..config.iterations {
        f8_args.submit_bitdecode_for_plan(&stream)?;
    }
    f8_stopped.record(&stream)?;
    f8_stopped.synchronize()?;
    let f8_batch_wall_ms = f8_wall_start.elapsed().as_secs_f64() * 1_000.0;
    let f8_cuda_event_ms = f8_started.elapsed_ms(&f8_stopped)?;
    let steady_f8_allocations = delta_allocations(
        steady_f8_allocation_start,
        cuda_device_allocation_snapshot(),
    )?;
    let steady_f8_boundaries =
        delta_boundaries(steady_f8_boundary_start, cuda_host_boundary_snapshot())?;

    // Exercise the non-blocking completion proof on one replay of each
    // component.  The final readback below is the only intentional D2H wait.
    let completion_event = OwnedCudaEvent::create(config.device_ordinal)?;
    let completion_poll_boundary_start = cuda_host_boundary_snapshot();
    let gather_poll_iterations = submit_and_poll(&stream, &completion_event, || {
        gather_args.submit_for_plan(&stream)
    })?;
    let f8_poll_iterations = submit_and_poll(&stream, &completion_event, || {
        f8_args.submit_bitdecode_for_plan(&stream)
    })?;
    let completion_poll_host_boundaries = delta_boundaries(
        completion_poll_boundary_start,
        cuda_host_boundary_snapshot(),
    )?;

    let mut gather_output_bytes = vec![0_u8; EMBEDDING_COLS * core::mem::size_of::<f32>()];
    gather_output.copy_to_host_sync(&stream, &mut gather_output_bytes)?;
    let mut f8_output_bytes = vec![0_u8; F8_ROWS * core::mem::size_of::<f32>()];
    f8_output.copy_to_host_sync(&stream, &mut f8_output_bytes)?;

    let gather_actual = f32_vec_from_bytes(&gather_output_bytes);
    let gather_expected = expected_embedding_row(&embedding_bytes, token_id, EMBEDDING_COLS);
    let f8_actual = f32_vec_from_bytes(&f8_output_bytes);
    let f8_expected = f8_reference(&f8_weights_bytes, &f8_scale_values, &f8_input_values);
    let tolerance = 2.0e-4_f32;
    let gather_max_abs_diff = max_abs_diff(&gather_actual, &gather_expected);
    let f8_max_abs_diff = max_abs_diff(&f8_actual, &f8_expected);
    let gather_pass = finite_and_within(&gather_actual, &gather_expected, tolerance);
    let f8_pass = finite_and_within(&f8_actual, &f8_expected, tolerance);
    let gather_output_sha256 = sha256_hex(&gather_output_bytes);
    let f8_output_sha256 = sha256_hex(&f8_output_bytes);

    drop(completion_event);
    drop(f8_stopped);
    drop(f8_started);
    drop(gather_stopped);
    drop(gather_started);
    drop(f8_output);
    drop(f8_input);
    drop(f8_scales);
    drop(f8_weights);
    drop(gather_output);
    drop(token);
    drop(embedding);
    drop(stream);
    let final_allocation_delta_after_cleanup =
        delta_allocations(allocation_start, cuda_device_allocation_snapshot())?;

    Ok(Qwen35CudaComponentReport {
        schema: QWEN35_CUDA_COMPONENT_SCHEMA.to_owned(),
        status: if gather_pass && f8_pass {
            "PASS_COMPONENT_CUDA_SM70_BOUNDARY".to_owned()
        } else {
            "FAIL_COMPONENT_CUDA_NUMERICAL_BOUNDARY".to_owned()
        },
        component_only: true,
        source_commit: config.source_commit.clone(),
        device: DeviceReport {
            ordinal: device.ordinal,
            name: device.name,
            compute_capability: [
                device.compute_capability.major,
                device.compute_capability.minor,
            ],
            total_global_memory_bytes: device.total_global_memory_bytes,
        },
        shape: ShapeReport {
            embedding_rows: EMBEDDING_ROWS,
            embedding_cols: EMBEDDING_COLS,
            embedding_token_id: token_id,
            f8_rows: F8_ROWS,
            f8_cols: F8_COLS,
            f8_scale_count: f8_scale_values.len(),
        },
        timing: TimingReport {
            gather_cuda_event_ms,
            f8_cuda_event_ms,
            gather_batch_wall_ms,
            f8_batch_wall_ms,
            gather_per_submission_event_ms: f64::from(gather_cuda_event_ms)
                / config.iterations as f64,
            f8_per_submission_event_ms: f64::from(f8_cuda_event_ms) / config.iterations as f64,
        },
        completion: CompletionReport {
            gather_poll_iterations,
            f8_poll_iterations,
            nonblocking_poll_only: true,
            completion_proof_required: true,
        },
        numerical: NumericalReport {
            gather_finite_count: gather_actual
                .iter()
                .filter(|value| value.is_finite())
                .count(),
            gather_expected_count: gather_expected.len(),
            gather_max_abs_diff,
            f8_finite_count: f8_actual.iter().filter(|value| value.is_finite()).count(),
            f8_expected_count: f8_expected.len(),
            f8_max_abs_diff,
            gather_output_sha256,
            f8_output_sha256,
            tolerance,
            gather_pass,
            f8_pass,
        },
        resources: ResourceReport {
            setup_allocations,
            steady_gather_allocations,
            steady_f8_allocations,
            setup_host_boundaries,
            steady_host_boundaries: merge_boundaries(
                steady_gather_boundaries,
                steady_f8_boundaries,
            ),
            completion_poll_host_boundaries,
            final_allocation_delta_after_cleanup,
        },
        claims: ClaimReport {
            complete_transformer_executed: false,
            decode_executed: false,
            semantic_parity_established: false,
            product_tps_permitted: false,
            default_cli_enabled: false,
        },
    })
}

fn submit_and_poll(
    stream: &OwnedCudaStream,
    event: &OwnedCudaEvent,
    submit: impl FnOnce() -> Result<(), CudaRuntimeError>,
) -> Result<u64, Qwen35CudaComponentError> {
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
            return Err(Qwen35CudaComponentError::CompletionPollLimit);
        }
        spin_loop();
    }
}

fn delta_allocations(
    before: augur_backend::CudaDeviceAllocationSnapshot,
    after: augur_backend::CudaDeviceAllocationSnapshot,
) -> Result<AllocationDeltaReport, Qwen35CudaComponentError> {
    Ok(AllocationDeltaReport {
        successful_allocations: after
            .successful_allocations
            .checked_sub(before.successful_allocations)
            .ok_or(Qwen35CudaComponentError::InvalidConfiguration(
                "allocation counters",
            ))?,
        successful_frees: after
            .successful_frees
            .checked_sub(before.successful_frees)
            .ok_or(Qwen35CudaComponentError::InvalidConfiguration(
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
) -> Result<HostBoundaryDeltaReport, Qwen35CudaComponentError> {
    let delta =
        after
            .checked_delta_since(before)
            .ok_or(Qwen35CudaComponentError::InvalidConfiguration(
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

fn merge_boundaries(
    first: HostBoundaryDeltaReport,
    second: HostBoundaryDeltaReport,
) -> HostBoundaryDeltaReport {
    HostBoundaryDeltaReport {
        host_waits: first.host_waits + second.host_waits,
        stream_synchronizations: first.stream_synchronizations + second.stream_synchronizations,
        event_synchronizations: first.event_synchronizations + second.event_synchronizations,
        synchronous_h2d_copies: first.synchronous_h2d_copies + second.synchronous_h2d_copies,
        synchronous_h2d_bytes: first.synchronous_h2d_bytes + second.synchronous_h2d_bytes,
        synchronous_d2h_copies: first.synchronous_d2h_copies + second.synchronous_d2h_copies,
        synchronous_d2h_bytes: first.synchronous_d2h_bytes + second.synchronous_d2h_bytes,
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

fn f32_vec_from_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(core::mem::size_of::<f32>())
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("f32 chunk")))
        .collect()
}

fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    (rounded >> 16) as u16
}

fn make_bf16_embedding_bytes(rows: usize, cols: usize) -> Vec<u8> {
    (0..rows * cols)
        .flat_map(|index| bf16_bits(((index % 37) as f32 - 18.0) / 17.0).to_le_bytes())
        .collect()
}

fn expected_embedding_row(bytes: &[u8], row: u32, cols: usize) -> Vec<f32> {
    let start = row as usize * cols * 2;
    bytes[start..start + cols * 2]
        .chunks_exact(2)
        .map(|chunk| f32::from_bits(u32::from(u16::from_le_bytes([chunk[0], chunk[1]])) << 16))
        .collect()
}

fn make_f8_weights_bytes(rows: usize, cols: usize) -> Vec<u8> {
    (0..rows * cols)
        .map(|index| {
            let exponent = ((index / 11) % 14 + 1) as u8;
            let mantissa = (index % 7) as u8;
            let sign = if index % 5 == 0 { 0x80 } else { 0 };
            sign | (exponent << 3) | mantissa
        })
        .collect()
}

fn make_f8_scales(rows: usize, cols: usize) -> Vec<f32> {
    (0..(rows / 128) * (cols / 128))
        .map(|index| 0.25 + (index % 5) as f32 * 0.125)
        .collect()
}

fn make_f32_input(cols: usize) -> Vec<f32> {
    (0..cols)
        .map(|index| ((index % 29) as f32 - 14.0) / 19.0)
        .collect()
}

fn f8_to_f32(value: u8) -> f32 {
    let sign = u32::from(value & 0x80) << 24;
    let exponent = (value >> 3) & 0x0f;
    let mantissa = value & 0x07;
    if exponent == 0x0f && mantissa == 0x07 {
        return f32::NAN;
    }
    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign);
        }
        let magnitude = f32::from(mantissa) * 2f32.powi(-9);
        return f32::from_bits(magnitude.to_bits() | sign);
    }
    f32::from_bits(sign | ((u32::from(exponent) + 120) << 23) | (u32::from(mantissa) << 20))
}

fn f8_reference(weights: &[u8], scales: &[f32], input: &[f32]) -> Vec<f32> {
    (0..F8_ROWS)
        .map(|row| {
            let mut sum = 0.0_f32;
            for col in 0..F8_COLS {
                let scale = scales[(row / 128) * (F8_COLS / 128) + col / 128];
                sum += f8_to_f32(weights[row * F8_COLS + col]) * scale * input[col];
            }
            sum
        })
        .collect()
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
    use super::{f8_to_f32, make_f8_scales, make_f8_weights_bytes};

    #[test]
    fn f8_fixture_is_finite_and_has_expected_scale_grid() {
        let weights = make_f8_weights_bytes(256, 256);
        assert!(weights.iter().all(|value| *value != 0x7f && *value != 0xff));
        assert_eq!(make_f8_scales(256, 256).len(), 4);
        assert!(f8_to_f32(0x38).is_finite());
    }

    #[test]
    fn f8_special_value_is_rejected_by_numerical_gate() {
        assert!(f8_to_f32(0x7f).is_nan());
    }
}
