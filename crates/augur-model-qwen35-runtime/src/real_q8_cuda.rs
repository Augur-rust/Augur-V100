#![cfg(feature = "cuda")]

//! Bounded real-weight Q8_0 CUDA evidence for the Qwen35 lane.
//!
//! This probe intentionally stops at one finite row window from the real
//! GGUF. It validates a quantized device boundary and a matrix-vector result;
//! it is not a transformer, decode, semantic, or product-throughput runner.

use std::{error::Error, fmt, hint::spin_loop, path::PathBuf, time::Instant};

use augur_backend::{
    cuda_device_allocation_snapshot, cuda_device_properties, cuda_host_boundary_snapshot,
    CudaF32View, CudaQ8_0F32MatvecArgsV1, CudaQ8_0WeightViewV1, CudaRuntimeError, OwnedCudaBuffer,
    OwnedCudaEvent, OwnedCudaStream,
};
use augur_core::PhysicalLayoutId;
use augur_model_qwen35::{
    GgufAuditOptions, GgufManifest, Qwen35Error, GGML_TYPE_Q8_0, Q8_0_BLOCK_BYTES,
    Q8_0_BLOCK_ELEMENTS,
};
use serde::Serialize;
use sha2::{Digest, Sha256};

pub const QWEN35_REAL_Q8_CUDA_SLICE_SCHEMA: &str = "crystal-quant.qwen35.real-q8-0-cuda-slice.v1";

const DEFAULT_Q8_TENSOR: &str = "blk.0.attn_qkv.weight";
const MAX_ROWS: usize = 512;
const MAX_COMPLETION_POLLS: u64 = 1_000_000;

/// Configuration for a bounded contiguous Q8_0 row-window probe.
#[derive(Clone, Debug)]
pub struct Qwen35RealQ8CudaSliceConfig {
    pub gguf_path: PathBuf,
    /// Exact GGUF tensor name. GGUF uses [input_width, output_rows].
    pub tensor_name: String,
    pub device_ordinal: i32,
    pub row_start: usize,
    pub rows: usize,
    pub warmup: usize,
    pub iterations: usize,
    pub source_commit: String,
}

impl Default for Qwen35RealQ8CudaSliceConfig {
    fn default() -> Self {
        Self {
            gguf_path: PathBuf::new(),
            tensor_name: DEFAULT_Q8_TENSOR.to_owned(),
            device_ordinal: 0,
            row_start: 0,
            rows: 128,
            warmup: 10,
            iterations: 200,
            source_commit: "UNBOUND_COMPONENT_SOURCE".to_owned(),
        }
    }
}

#[derive(Debug)]
pub enum Qwen35RealQ8CudaSliceError {
    Runtime(CudaRuntimeError),
    Model(Qwen35Error),
    InvalidConfiguration(&'static str),
    CompletionPollLimit,
}

impl fmt::Display for Qwen35RealQ8CudaSliceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Runtime(error) => error.fmt(formatter),
            Self::Model(error) => error.fmt(formatter),
            Self::InvalidConfiguration(field) => {
                write!(
                    formatter,
                    "invalid Qwen real Q8 CUDA slice configuration: {field}"
                )
            }
            Self::CompletionPollLimit => {
                formatter.write_str("CUDA completion polling exceeded its bound")
            }
        }
    }
}

impl Error for Qwen35RealQ8CudaSliceError {}

impl From<CudaRuntimeError> for Qwen35RealQ8CudaSliceError {
    fn from(error: CudaRuntimeError) -> Self {
        Self::Runtime(error)
    }
}

impl From<Qwen35Error> for Qwen35RealQ8CudaSliceError {
    fn from(error: Qwen35Error) -> Self {
        Self::Model(error)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Qwen35RealQ8CudaSliceReport {
    pub schema: String,
    pub status: String,
    pub component_only: bool,
    pub source_commit: String,
    pub device: Q8CudaDeviceReport,
    pub source: Q8CudaSourceReport,
    pub shape: Q8CudaShapeReport,
    pub timing: Q8CudaTimingReport,
    pub completion: Q8CudaCompletionReport,
    pub numerical: Q8CudaNumericalReport,
    pub resources: Q8CudaResourceReport,
    pub claims: Q8CudaClaimReport,
}

#[derive(Clone, Debug, Serialize)]
pub struct Q8CudaDeviceReport {
    pub ordinal: i32,
    pub name: String,
    pub compute_capability: [u16; 2],
    pub total_global_memory_bytes: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Q8CudaSourceReport {
    pub path: String,
    pub file_sha256: String,
    pub tensor_name: String,
    pub tensor_dimensions: Vec<u64>,
    pub ggml_type: u32,
    pub row_start: usize,
    pub rows: usize,
    pub slice_offset_bytes: u64,
    pub slice_bytes: usize,
    pub slice_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct Q8CudaShapeReport {
    pub rows: usize,
    pub columns: usize,
    pub blocks_per_row: usize,
    pub row_bytes: usize,
    pub output_elements: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct Q8CudaTimingReport {
    pub cuda_event_ms: f32,
    pub batch_wall_ms: f64,
    pub per_submission_event_ms: f64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Q8CudaCompletionReport {
    pub poll_iterations: u64,
    pub nonblocking_poll_only: bool,
    pub completion_proof_required: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct Q8CudaNumericalReport {
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
pub struct Q8CudaResourceReport {
    pub setup_allocations: AllocationDeltaReport,
    pub steady_allocations: AllocationDeltaReport,
    pub completion_poll_host_boundaries: HostBoundaryDeltaReport,
    pub setup_host_boundaries: HostBoundaryDeltaReport,
    pub steady_host_boundaries: HostBoundaryDeltaReport,
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
pub struct Q8CudaClaimReport {
    pub real_q8_slice_uploaded: bool,
    pub q8_matvec_executed: bool,
    pub complete_transformer_executed: bool,
    pub decode_executed: bool,
    pub semantic_parity_established: bool,
    pub product_tps_permitted: bool,
    pub default_cli_enabled: bool,
}

/// Upload and execute a bounded Q8_0 row window from the real GGUF.
pub fn run_qwen35_real_q8_cuda_slice(
    config: &Qwen35RealQ8CudaSliceConfig,
) -> Result<Qwen35RealQ8CudaSliceReport, Qwen35RealQ8CudaSliceError> {
    validate_config(config)?;
    let manifest = GgufManifest::inspect(&config.gguf_path, GgufAuditOptions::default())?;
    let tensor = manifest
        .tensor(&config.tensor_name)
        .ok_or(Qwen35RealQ8CudaSliceError::InvalidConfiguration(
            "requested Q8 tensor is missing",
        ))?
        .clone();
    if tensor.ggml_type != GGML_TYPE_Q8_0 || tensor.dimensions.len() != 2 {
        return Err(Qwen35RealQ8CudaSliceError::InvalidConfiguration(
            "requested tensor must be rank-2 Q8_0",
        ));
    }

    let columns = usize::try_from(tensor.dimensions[0])
        .map_err(|_| Qwen35RealQ8CudaSliceError::InvalidConfiguration("tensor columns"))?;
    let tensor_rows = usize::try_from(tensor.dimensions[1])
        .map_err(|_| Qwen35RealQ8CudaSliceError::InvalidConfiguration("tensor rows"))?;
    if columns == 0
        || tensor_rows == 0
        || !columns.is_multiple_of(Q8_0_BLOCK_ELEMENTS)
        || config.row_start > tensor_rows
        || config.rows > tensor_rows.saturating_sub(config.row_start)
    {
        return Err(Qwen35RealQ8CudaSliceError::InvalidConfiguration(
            "Q8 tensor geometry or row window",
        ));
    }
    let blocks_per_row = columns / Q8_0_BLOCK_ELEMENTS;
    let row_bytes = blocks_per_row.checked_mul(Q8_0_BLOCK_BYTES).ok_or(
        Qwen35RealQ8CudaSliceError::InvalidConfiguration("row bytes"),
    )?;
    let slice_offset = config.row_start.checked_mul(row_bytes).ok_or(
        Qwen35RealQ8CudaSliceError::InvalidConfiguration("slice offset"),
    )?;
    let slice_bytes_len = config.rows.checked_mul(row_bytes).ok_or(
        Qwen35RealQ8CudaSliceError::InvalidConfiguration("slice bytes"),
    )?;
    let mut reader = manifest.open_tensor_reader(&config.gguf_path)?;
    let slice_bytes = reader.read_tensor_slice(
        &config.tensor_name,
        u64::try_from(slice_offset)
            .map_err(|_| Qwen35RealQ8CudaSliceError::InvalidConfiguration("slice offset"))?,
        u64::try_from(slice_bytes_len)
            .map_err(|_| Qwen35RealQ8CudaSliceError::InvalidConfiguration("slice bytes"))?,
    )?;
    if slice_bytes.len() != slice_bytes_len {
        return Err(Qwen35RealQ8CudaSliceError::InvalidConfiguration(
            "slice reader length",
        ));
    }

    let input_values = make_matvec_input(columns);
    let expected = q8_matvec_reference(&slice_bytes, config.rows, columns, &input_values)?;
    let input_bytes = f32_bytes(&input_values);

    let device = cuda_device_properties(config.device_ordinal)?;
    let stream = OwnedCudaStream::create(config.device_ordinal)?;
    let allocation_start = cuda_device_allocation_snapshot();
    let boundary_start = cuda_host_boundary_snapshot();
    let weight = OwnedCudaBuffer::allocate(config.device_ordinal, slice_bytes.len())?;
    let input = OwnedCudaBuffer::allocate(config.device_ordinal, input_bytes.len())?;
    let output = OwnedCudaBuffer::allocate(
        config.device_ordinal,
        config.rows.checked_mul(core::mem::size_of::<f32>()).ok_or(
            Qwen35RealQ8CudaSliceError::InvalidConfiguration("output bytes"),
        )?,
    )?;
    weight.copy_from_host_sync(&stream, &slice_bytes)?;
    input.copy_from_host_sync(&stream, &input_bytes)?;
    let args = CudaQ8_0F32MatvecArgsV1::bind(
        &stream,
        CudaQ8_0WeightViewV1::new(
            weight.view(0, slice_bytes.len())?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_0_CUDA_V1,
            config.rows,
            columns,
        )?,
        CudaF32View::new(
            input.view(0, input_bytes.len())?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            columns,
        )?,
        CudaF32View::new(
            output.view(0, config.rows * core::mem::size_of::<f32>())?,
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

    let completion_event = OwnedCudaEvent::create(config.device_ordinal)?;
    let completion_boundary_start = cuda_host_boundary_snapshot();
    let poll_iterations =
        submit_and_poll(&stream, &completion_event, || args.submit_for_plan(&stream))?;
    let completion_poll_host_boundaries =
        delta_boundaries(completion_boundary_start, cuda_host_boundary_snapshot())?;

    let mut output_bytes = vec![0_u8; config.rows * core::mem::size_of::<f32>()];
    output.copy_to_host_sync(&stream, &mut output_bytes)?;
    let actual = f32_vec_from_bytes(&output_bytes);
    let tolerance = 2.0e-3_f32;
    let max_abs_diff = max_abs_diff(&actual, &expected);
    let pass = finite_and_within(&actual, &expected, tolerance);
    let output_sha256 = sha256_hex(&output_bytes);
    let expected_sha256 = sha256_f32_bits(&expected);
    let input_sha256 = sha256_hex(&input_bytes);
    let slice_sha256 = sha256_hex(&slice_bytes);

    drop(completion_event);
    drop(stopped);
    drop(started);
    drop(output);
    drop(input);
    drop(weight);
    drop(stream);
    let final_allocation_delta_after_cleanup =
        delta_allocations(allocation_start, cuda_device_allocation_snapshot())?;

    Ok(Qwen35RealQ8CudaSliceReport {
        schema: QWEN35_REAL_Q8_CUDA_SLICE_SCHEMA.to_owned(),
        status: if pass {
            "PASS_COMPONENT_REAL_Q8_CUDA_SLICE".to_owned()
        } else {
            "FAIL_COMPONENT_REAL_Q8_CUDA_SLICE_NUMERICAL".to_owned()
        },
        component_only: true,
        source_commit: config.source_commit.clone(),
        device: Q8CudaDeviceReport {
            ordinal: device.ordinal,
            name: device.name,
            compute_capability: [
                device.compute_capability.major,
                device.compute_capability.minor,
            ],
            total_global_memory_bytes: device.total_global_memory_bytes,
        },
        source: Q8CudaSourceReport {
            path: config.gguf_path.display().to_string(),
            file_sha256: manifest.file_sha256.clone().ok_or(
                Qwen35RealQ8CudaSliceError::InvalidConfiguration("GGUF file hash"),
            )?,
            tensor_name: config.tensor_name.clone(),
            tensor_dimensions: tensor.dimensions,
            ggml_type: tensor.ggml_type,
            row_start: config.row_start,
            rows: config.rows,
            slice_offset_bytes: u64::try_from(slice_offset)
                .map_err(|_| Qwen35RealQ8CudaSliceError::InvalidConfiguration("slice offset"))?,
            slice_bytes: slice_bytes.len(),
            slice_sha256,
        },
        shape: Q8CudaShapeReport {
            rows: config.rows,
            columns,
            blocks_per_row,
            row_bytes,
            output_elements: actual.len(),
        },
        timing: Q8CudaTimingReport {
            cuda_event_ms,
            batch_wall_ms,
            per_submission_event_ms: f64::from(cuda_event_ms) / config.iterations as f64,
        },
        completion: Q8CudaCompletionReport {
            poll_iterations,
            nonblocking_poll_only: true,
            completion_proof_required: true,
        },
        numerical: Q8CudaNumericalReport {
            input_sha256,
            finite_count: actual.iter().filter(|value| value.is_finite()).count(),
            expected_count: expected.len(),
            max_abs_diff,
            tolerance,
            output_sha256,
            expected_sha256,
            pass,
        },
        resources: Q8CudaResourceReport {
            setup_allocations,
            steady_allocations,
            completion_poll_host_boundaries,
            setup_host_boundaries,
            steady_host_boundaries,
            final_allocation_delta_after_cleanup,
        },
        claims: Q8CudaClaimReport {
            real_q8_slice_uploaded: true,
            q8_matvec_executed: true,
            complete_transformer_executed: false,
            decode_executed: false,
            semantic_parity_established: false,
            product_tps_permitted: false,
            default_cli_enabled: false,
        },
    })
}

fn validate_config(config: &Qwen35RealQ8CudaSliceConfig) -> Result<(), Qwen35RealQ8CudaSliceError> {
    if config.gguf_path.as_os_str().is_empty() {
        return Err(Qwen35RealQ8CudaSliceError::InvalidConfiguration(
            "gguf_path",
        ));
    }
    if config.tensor_name.trim().is_empty() {
        return Err(Qwen35RealQ8CudaSliceError::InvalidConfiguration(
            "tensor_name",
        ));
    }
    if config.device_ordinal < 0 {
        return Err(Qwen35RealQ8CudaSliceError::InvalidConfiguration(
            "device_ordinal",
        ));
    }
    if config.rows == 0 || config.rows > MAX_ROWS {
        return Err(Qwen35RealQ8CudaSliceError::InvalidConfiguration("rows"));
    }
    if config.iterations == 0 {
        return Err(Qwen35RealQ8CudaSliceError::InvalidConfiguration(
            "iterations",
        ));
    }
    Ok(())
}

fn submit_and_poll(
    stream: &OwnedCudaStream,
    event: &OwnedCudaEvent,
    submit: impl FnOnce() -> Result<(), CudaRuntimeError>,
) -> Result<u64, Qwen35RealQ8CudaSliceError> {
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
            return Err(Qwen35RealQ8CudaSliceError::CompletionPollLimit);
        }
        spin_loop();
    }
}

fn q8_matvec_reference(
    bytes: &[u8],
    rows: usize,
    columns: usize,
    input: &[f32],
) -> Result<Vec<f32>, Qwen35RealQ8CudaSliceError> {
    if rows == 0
        || columns == 0
        || !columns.is_multiple_of(Q8_0_BLOCK_ELEMENTS)
        || input.len() != columns
    {
        return Err(Qwen35RealQ8CudaSliceError::InvalidConfiguration(
            "Q8 reference geometry",
        ));
    }
    let blocks_per_row = columns / Q8_0_BLOCK_ELEMENTS;
    let row_bytes = blocks_per_row.checked_mul(Q8_0_BLOCK_BYTES).ok_or(
        Qwen35RealQ8CudaSliceError::InvalidConfiguration("Q8 reference row bytes"),
    )?;
    let expected_bytes =
        rows.checked_mul(row_bytes)
            .ok_or(Qwen35RealQ8CudaSliceError::InvalidConfiguration(
                "Q8 reference payload",
            ))?;
    if bytes.len() != expected_bytes || input.iter().any(|value| !value.is_finite()) {
        return Err(Qwen35RealQ8CudaSliceError::InvalidConfiguration(
            "Q8 reference payload or input",
        ));
    }
    let mut output = vec![0.0_f32; rows];
    for (row, destination) in output.iter_mut().enumerate() {
        let row_start = row * row_bytes;
        let mut sum = 0.0_f64;
        for block in 0..blocks_per_row {
            let block_start = row_start + block * Q8_0_BLOCK_BYTES;
            let scale_bits = u16::from_le_bytes([bytes[block_start], bytes[block_start + 1]]);
            let scale = q8_f16_to_f32(scale_bits);
            if !scale.is_finite() {
                return Err(Qwen35RealQ8CudaSliceError::InvalidConfiguration("Q8 scale"));
            }
            let input_start = block * Q8_0_BLOCK_ELEMENTS;
            for lane in 0..Q8_0_BLOCK_ELEMENTS {
                let code = i8::from_ne_bytes([bytes[block_start + 2 + lane]]);
                sum += f64::from(code) * f64::from(scale) * f64::from(input[input_start + lane]);
            }
        }
        let value = sum as f32;
        if !value.is_finite() {
            return Err(Qwen35RealQ8CudaSliceError::InvalidConfiguration(
                "Q8 reference output",
            ));
        }
        *destination = value;
    }
    Ok(output)
}

fn q8_f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let mantissa = u32::from(bits & 0x03ff);
    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign);
        }
        let mut normalized = mantissa;
        let mut shift = 0_i32;
        while normalized & 0x0400 == 0 {
            normalized <<= 1;
            shift -= 1;
        }
        normalized &= 0x03ff;
        let f32_exponent = (127 - 15 + 1 + shift) as u32;
        return f32::from_bits(sign | (f32_exponent << 23) | (normalized << 13));
    }
    if exponent == 0x1f {
        return f32::from_bits(sign | 0x7f80_0000 | (mantissa << 13));
    }
    f32::from_bits(sign | ((exponent + 112) << 23) | (mantissa << 13))
}

fn delta_allocations(
    before: augur_backend::CudaDeviceAllocationSnapshot,
    after: augur_backend::CudaDeviceAllocationSnapshot,
) -> Result<AllocationDeltaReport, Qwen35RealQ8CudaSliceError> {
    Ok(AllocationDeltaReport {
        successful_allocations: after
            .successful_allocations
            .checked_sub(before.successful_allocations)
            .ok_or(Qwen35RealQ8CudaSliceError::InvalidConfiguration(
                "allocation counters",
            ))?,
        successful_frees: after
            .successful_frees
            .checked_sub(before.successful_frees)
            .ok_or(Qwen35RealQ8CudaSliceError::InvalidConfiguration(
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
) -> Result<HostBoundaryDeltaReport, Qwen35RealQ8CudaSliceError> {
    let delta = after.checked_delta_since(before).ok_or(
        Qwen35RealQ8CudaSliceError::InvalidConfiguration("host counters"),
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

fn make_matvec_input(columns: usize) -> Vec<f32> {
    (0..columns)
        .map(|index| ((index % 53) as f32 - 26.0) / 37.0)
        .collect()
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
    use super::{make_matvec_input, q8_f16_to_f32, q8_matvec_reference};

    #[test]
    fn half_scale_conversion_matches_basic_values() {
        assert_eq!(q8_f16_to_f32(0x0000), 0.0);
        assert_eq!(q8_f16_to_f32(0x3c00), 1.0);
        assert_eq!(q8_f16_to_f32(0xbc00), -1.0);
    }

    #[test]
    fn reference_accepts_one_q8_row() {
        let mut bytes = vec![0_u8; 34];
        bytes[0..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        bytes[2] = 2;
        let input = make_matvec_input(32);
        let output = q8_matvec_reference(&bytes, 1, 32, &input).unwrap();
        assert_eq!(output.len(), 1);
        assert!(output[0].is_finite());
    }
}
