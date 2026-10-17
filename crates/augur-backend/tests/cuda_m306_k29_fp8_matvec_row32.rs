#![cfg(feature = "cuda")]

use augur_backend::{
    CudaF32View, CudaF8E4M3F32MatvecArgsV1, CudaF8E4M3F32MatvecRow32ArgsV1,
    CudaF8E4M3Row32WeightViewV1, CudaF8E4M3WeightViewV1, CudaRuntimeError, OwnedCudaBuffer,
    OwnedCudaEvent, OwnedCudaStream,
};
use augur_core::PhysicalLayoutId;

const ROWS: usize = 8192;
const COLS: usize = 1024;
const CALLS: usize = 21;
const PAIRS: usize = 15;

fn row32_interleave(source: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(source.len(), rows * cols);
    assert!(rows.is_multiple_of(32));
    let mut result = vec![0_u8; source.len()];
    for row in 0..rows {
        let tile_base = (row / 32) * cols * 32;
        let lane = row % 32;
        for col in 0..cols {
            result[tile_base + col * 32 + lane] = source[row * cols + col];
        }
    }
    result
}

fn row32_deinterleave(source: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(source.len(), rows * cols);
    assert!(rows.is_multiple_of(32));
    let mut result = vec![0_u8; source.len()];
    for row in 0..rows {
        let tile_base = (row / 32) * cols * 32;
        let lane = row % 32;
        for col in 0..cols {
            result[row * cols + col] = source[tile_base + col * 32 + lane];
        }
    }
    result
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn elapsed_ms(stream: &OwnedCudaStream, launch: &dyn Fn() -> Result<(), CudaRuntimeError>) -> f64 {
    let start = OwnedCudaEvent::create(0).unwrap();
    let stop = OwnedCudaEvent::create(0).unwrap();
    start.record(stream).unwrap();
    launch().unwrap();
    stop.record(stream).unwrap();
    stop.synchronize().unwrap();
    f64::from(start.elapsed_ms(&stop).unwrap())
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

fn p95(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[(sorted.len() - 1) * 95 / 100]
}

fn coefficient_of_variation(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt() / mean
}

struct Fixture {
    stream: OwnedCudaStream,
    control_weight: OwnedCudaBuffer,
    candidate_weight: OwnedCudaBuffer,
    scales: OwnedCudaBuffer,
    input: OwnedCudaBuffer,
    control_output: OwnedCudaBuffer,
    candidate_output: OwnedCudaBuffer,
}

impl Fixture {
    fn new(rows: usize, cols: usize, include_nan_codes: bool) -> Self {
        let row_major = (0..rows * cols)
            .map(|index| {
                let mut value = (index as u8).wrapping_mul(73).wrapping_add(19);
                if !include_nan_codes && (value & 0x7f) == 0x7f {
                    value = (value & 0x80) | 0x7e;
                }
                value
            })
            .collect::<Vec<_>>();
        let row32 = row32_interleave(&row_major, rows, cols);
        assert_eq!(row32_deinterleave(&row32, rows, cols), row_major);
        let scales = (0..(rows / 128) * (cols / 128))
            .map(|index| match index % 4 {
                0 => 0.25_f32,
                1 => -0.5_f32,
                2 => 1.0_f32,
                _ => 2.0_f32,
            })
            .collect::<Vec<_>>();
        let input = (0..cols)
            .map(|index| ((index % 257) as f32 - 128.0) * 0.000_976_562_5)
            .collect::<Vec<_>>();
        let stream = OwnedCudaStream::create(0).unwrap();
        let control_weight = OwnedCudaBuffer::allocate(0, row_major.len()).unwrap();
        let candidate_weight = OwnedCudaBuffer::allocate(0, row32.len()).unwrap();
        let scales_buffer = OwnedCudaBuffer::allocate(0, scales.len() * 4).unwrap();
        let input_buffer = OwnedCudaBuffer::allocate(0, input.len() * 4).unwrap();
        let control_output = OwnedCudaBuffer::allocate(0, rows * 4).unwrap();
        let candidate_output = OwnedCudaBuffer::allocate(0, rows * 4).unwrap();
        control_weight
            .copy_from_host_sync(&stream, &row_major)
            .unwrap();
        candidate_weight
            .copy_from_host_sync(&stream, &row32)
            .unwrap();
        scales_buffer
            .copy_from_host_sync(&stream, &f32_bytes(&scales))
            .unwrap();
        input_buffer
            .copy_from_host_sync(&stream, &f32_bytes(&input))
            .unwrap();
        Self {
            stream,
            control_weight,
            candidate_weight,
            scales: scales_buffer,
            input: input_buffer,
            control_output,
            candidate_output,
        }
    }

    fn args(
        &self,
        rows: usize,
        cols: usize,
    ) -> (
        CudaF8E4M3F32MatvecArgsV1,
        CudaF8E4M3F32MatvecRow32ArgsV1,
        CudaF8E4M3F32MatvecArgsV1,
    ) {
        let scales = CudaF32View::new(
            self.scales
                .view(0, (rows / 128) * (cols / 128) * 4)
                .unwrap(),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            (rows / 128) * (cols / 128),
        )
        .unwrap();
        let input = CudaF32View::new(
            self.input.view(0, cols * 4).unwrap(),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            cols,
        )
        .unwrap();
        let control_output = CudaF32View::new(
            self.control_output.view(0, rows * 4).unwrap(),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            rows,
        )
        .unwrap();
        let candidate_output = CudaF32View::new(
            self.candidate_output.view(0, rows * 4).unwrap(),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            rows,
        )
        .unwrap();
        let control_weight = CudaF8E4M3WeightViewV1::new(
            self.control_weight.view(0, rows * cols).unwrap(),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F8E4M3_CUDA_V1,
            rows,
            cols,
        )
        .unwrap();
        let candidate_weight = CudaF8E4M3Row32WeightViewV1::new(
            self.candidate_weight.view(0, rows * cols).unwrap(),
            PhysicalLayoutId::DEVICE_ROW32_F8E4M3_CUDA_V1,
            rows,
            cols,
        )
        .unwrap();
        (
            CudaF8E4M3F32MatvecArgsV1::bind(
                &self.stream,
                control_weight,
                scales,
                input,
                control_output,
            )
            .unwrap(),
            CudaF8E4M3F32MatvecRow32ArgsV1::bind(
                &self.stream,
                candidate_weight,
                scales,
                input,
                candidate_output,
            )
            .unwrap(),
            CudaF8E4M3F32MatvecArgsV1::bind(
                &self.stream,
                control_weight,
                scales,
                input,
                candidate_output,
            )
            .unwrap(),
        )
    }

    

    fn outputs(&self, rows: usize) -> (Vec<u8>, Vec<u8>) {
        let mut control = vec![0_u8; rows * 4];
        let mut candidate = vec![0_u8; rows * 4];
        self.control_output
            .copy_to_host_sync(&self.stream, &mut control)
            .unwrap();
        self.candidate_output
            .copy_to_host_sync(&self.stream, &mut candidate)
            .unwrap();
        (control, candidate)
    }
}

#[test]
fn row32_fp8_decode_is_byte_exact_for_all_codes_and_replay() {
    const TEST_ROWS: usize = 128;
    const TEST_COLS: usize = 256;
    let fixture = Fixture::new(TEST_ROWS, TEST_COLS, true);
    let (control, candidate, bitdecode) = fixture.args(TEST_ROWS, TEST_COLS);
    control.submit_for_plan(&fixture.stream).unwrap();
    candidate.submit_for_plan(&fixture.stream).unwrap();
    let (control_bytes, first_candidate) = fixture.outputs(TEST_ROWS);
    assert_eq!(control_bytes, first_candidate);
    candidate.submit_for_plan(&fixture.stream).unwrap();
    assert_eq!(first_candidate, fixture.outputs(TEST_ROWS).1);

    bitdecode
        .submit_bitdecode_for_plan(&fixture.stream)
        .unwrap();
    assert_eq!(control_bytes, fixture.outputs(TEST_ROWS).1);
    bitdecode
        .submit_bitdecode_for_plan(&fixture.stream)
        .unwrap();
    assert_eq!(control_bytes, fixture.outputs(TEST_ROWS).1);

    assert!(CudaF8E4M3Row32WeightViewV1::new(
        fixture
            .candidate_weight
            .view(0, TEST_ROWS * TEST_COLS)
            .unwrap(),
        PhysicalLayoutId::DEVICE_CONTIGUOUS_F8E4M3_CUDA_V1,
        TEST_ROWS,
        TEST_COLS,
    )
    .is_err());
    assert!(CudaF8E4M3Row32WeightViewV1::new(
        fixture
            .candidate_weight
            .view(0, TEST_ROWS * TEST_COLS - 1)
            .unwrap(),
        PhysicalLayoutId::DEVICE_ROW32_F8E4M3_CUDA_V1,
        TEST_ROWS,
        TEST_COLS,
    )
    .is_err());
    assert!(CudaF8E4M3Row32WeightViewV1::new(
        fixture
            .candidate_weight
            .view(0, TEST_ROWS * TEST_COLS)
            .unwrap(),
        PhysicalLayoutId::DEVICE_ROW32_F8E4M3_CUDA_V1,
        TEST_ROWS - 1,
        TEST_COLS,
    )
    .is_err());
}

#[test]
fn row32_fp8_matvec_reproduces_frozen_source_shape_no_go() {
    let fixture = Fixture::new(ROWS, COLS, false);
    let (control, candidate, _) = fixture.args(ROWS, COLS);
    control.submit_for_plan(&fixture.stream).unwrap();
    candidate.submit_for_plan(&fixture.stream).unwrap();
    let (control_output, candidate_output) = fixture.outputs(ROWS);
    assert_eq!(control_output, candidate_output);

    let control_launch = || {
        for _ in 0..CALLS {
            control.submit_for_plan(&fixture.stream)?;
        }
        Ok(())
    };
    let candidate_launch = || {
        for _ in 0..CALLS {
            candidate.submit_for_plan(&fixture.stream)?;
        }
        Ok(())
    };
    for _ in 0..5 {
        control_launch().unwrap();
        candidate_launch().unwrap();
    }
    fixture.stream.synchronize().unwrap();

    let mut control_ms = Vec::with_capacity(PAIRS);
    let mut candidate_ms = Vec::with_capacity(PAIRS);
    for pair in 0..PAIRS {
        if pair.is_multiple_of(2) {
            control_ms.push(elapsed_ms(&fixture.stream, &control_launch));
            candidate_ms.push(elapsed_ms(&fixture.stream, &candidate_launch));
        } else {
            candidate_ms.push(elapsed_ms(&fixture.stream, &candidate_launch));
            control_ms.push(elapsed_ms(&fixture.stream, &control_launch));
        }
    }
    let control_median = median(&control_ms);
    let candidate_median = median(&candidate_ms);
    let speedups = control_ms
        .iter()
        .zip(&candidate_ms)
        .map(|(control, candidate)| control / candidate)
        .collect::<Vec<_>>();
    let paired_cv = coefficient_of_variation(&speedups);
    eprintln!(
        "K29_R1 control_ms={control_median:.6} candidate_ms={candidate_median:.6} speedup={:.6} control_p95_ms={:.6} candidate_p95_ms={:.6} paired_cv_pct={:.6}",
        control_median / candidate_median,
        p95(&control_ms),
        p95(&candidate_ms),
        paired_cv * 100.0,
    );
    let passes = control_median / candidate_median >= 2.0
        && paired_cv <= 0.03
        && p95(&candidate_ms) <= p95(&control_ms);
    assert!(!passes, "K29-R1 unexpectedly passed its frozen gate");
}

#[test]
fn bitdecode_fp8_matvec_meets_frozen_source_shape_gate() {
    let fixture = Fixture::new(ROWS, COLS, false);
    let (control, _, candidate) = fixture.args(ROWS, COLS);
    control.submit_for_plan(&fixture.stream).unwrap();
    candidate
        .submit_bitdecode_for_plan(&fixture.stream)
        .unwrap();
    let (control_output, candidate_output) = fixture.outputs(ROWS);
    assert_eq!(control_output, candidate_output);

    let control_launch = || {
        for _ in 0..CALLS {
            control.submit_for_plan(&fixture.stream)?;
        }
        Ok(())
    };
    let candidate_launch = || {
        for _ in 0..CALLS {
            candidate.submit_bitdecode_for_plan(&fixture.stream)?;
        }
        Ok(())
    };
    for _ in 0..5 {
        control_launch().unwrap();
        candidate_launch().unwrap();
    }
    fixture.stream.synchronize().unwrap();

    let mut control_ms = Vec::with_capacity(PAIRS);
    let mut candidate_ms = Vec::with_capacity(PAIRS);
    for pair in 0..PAIRS {
        if pair.is_multiple_of(2) {
            control_ms.push(elapsed_ms(&fixture.stream, &control_launch));
            candidate_ms.push(elapsed_ms(&fixture.stream, &candidate_launch));
        } else {
            candidate_ms.push(elapsed_ms(&fixture.stream, &candidate_launch));
            control_ms.push(elapsed_ms(&fixture.stream, &control_launch));
        }
    }
    let control_median = median(&control_ms);
    let candidate_median = median(&candidate_ms);
    let speedups = control_ms
        .iter()
        .zip(&candidate_ms)
        .map(|(control, candidate)| control / candidate)
        .collect::<Vec<_>>();
    let paired_cv = coefficient_of_variation(&speedups);
    eprintln!(
        "K29_R2 control_ms={control_median:.6} candidate_ms={candidate_median:.6} speedup={:.6} control_p95_ms={:.6} candidate_p95_ms={:.6} paired_cv_pct={:.6}",
        control_median / candidate_median,
        p95(&control_ms),
        p95(&candidate_ms),
        paired_cv * 100.0,
    );
    assert!(control_median / candidate_median >= 1.10);
    assert!(paired_cv <= 0.03);
    assert!(p95(&candidate_ms) <= p95(&control_ms));
}


