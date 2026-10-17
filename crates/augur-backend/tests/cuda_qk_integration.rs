#![cfg(feature = "cuda")]

use augur_backend::{
    submit_m120_q5k_exact_sm70, CudaF32View, CudaQ5kF32DecodeArgsV1, CudaQ5kF32DecodeGrid32ArgsV2,
    CudaQ5kWeightViewV1, CudaQ6kF32DecodeArgsV1, CudaQ6kWeightViewV1, OwnedCudaBuffer,
    OwnedCudaEvent, OwnedCudaStream,
};
use augur_core::PhysicalLayoutId;
use augur_formats::{Q5KCompactBlock, Q6KCompactBlock, ReferenceDecode};
use augur_repack::{repack_q5k_compact_to_cuda_v1, repack_q6k_compact_to_cuda_v1};

fn input_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn run_q5(compact: &[u8], input: &[f32]) -> Vec<f32> {
    let hot = repack_q5k_compact_to_cuda_v1(compact, 1, 256).unwrap();
    let stream = OwnedCudaStream::create(0).unwrap();
    let weight = OwnedCudaBuffer::allocate(0, hot.payload_bytes().len()).unwrap();
    let input_buffer = OwnedCudaBuffer::allocate(0, input.len() * 4).unwrap();
    let output = OwnedCudaBuffer::allocate(0, 4).unwrap();
    weight
        .copy_from_host_sync(&stream, hot.payload_bytes())
        .unwrap();
    input_buffer
        .copy_from_host_sync(&stream, &input_bytes(input))
        .unwrap();
    let w = CudaQ5kWeightViewV1::new(
        weight.view(0, hot.payload_bytes().len()).unwrap(),
        PhysicalLayoutId::Q5K_HOT_CUDA_V1,
        1,
        256,
    )
    .unwrap();
    let x = CudaF32View::new(
        input_buffer.view(0, input.len() * 4).unwrap(),
        PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
        256,
    )
    .unwrap();
    let y = CudaF32View::new(
        output.view(0, 4).unwrap(),
        PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
        1,
    )
    .unwrap();
    CudaQ5kF32DecodeArgsV1::bind(&stream, w, x, y)
        .unwrap()
        .launch(&stream)
        .unwrap();
    let mut bytes = [0_u8; 4];
    output.copy_to_host_sync(&stream, &mut bytes).unwrap();
    vec![f32::from_le_bytes(bytes)]
}

fn run_q6(compact: &[u8], input: &[f32]) -> Vec<f32> {
    let hot = repack_q6k_compact_to_cuda_v1(compact, 1, 256).unwrap();
    let stream = OwnedCudaStream::create(0).unwrap();
    let weight = OwnedCudaBuffer::allocate(0, hot.payload_bytes().len()).unwrap();
    let input_buffer = OwnedCudaBuffer::allocate(0, input.len() * 4).unwrap();
    let output = OwnedCudaBuffer::allocate(0, 4).unwrap();
    weight
        .copy_from_host_sync(&stream, hot.payload_bytes())
        .unwrap();
    input_buffer
        .copy_from_host_sync(&stream, &input_bytes(input))
        .unwrap();
    let w = CudaQ6kWeightViewV1::new(
        weight.view(0, hot.payload_bytes().len()).unwrap(),
        PhysicalLayoutId::Q6K_HOT_CUDA_V1,
        1,
        256,
    )
    .unwrap();
    let x = CudaF32View::new(
        input_buffer.view(0, input.len() * 4).unwrap(),
        PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
        256,
    )
    .unwrap();
    let y = CudaF32View::new(
        output.view(0, 4).unwrap(),
        PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
        1,
    )
    .unwrap();
    CudaQ6kF32DecodeArgsV1::bind(&stream, w, x, y)
        .unwrap()
        .launch(&stream)
        .unwrap();
    let mut bytes = [0_u8; 4];
    output.copy_to_host_sync(&stream, &mut bytes).unwrap();
    vec![f32::from_le_bytes(bytes)]
}

fn q5_compact_fixture(rows: usize, cols: usize) -> Vec<u8> {
    let blocks = rows * (cols / 256);
    let mut compact = vec![0_u8; blocks * 176];
    for block_index in 0..blocks {
        let block = &mut compact[block_index * 176..(block_index + 1) * 176];
        block[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        block[4..16].fill((block_index as u8 & 0x07) + 1);
        for (index, value) in block[48..].iter_mut().enumerate() {
            let seed = index.wrapping_add(block_index * 13);
            *value = ((seed * 3) as u8 & 0x0f) | (((seed * 5) as u8 & 0x0f) << 4);
        }
    }
    compact
}

fn median(mut samples: Vec<f32>) -> f32 {
    samples.sort_by(f32::total_cmp);
    samples[samples.len() / 2]
}

fn elapsed_ms(
    stream: &OwnedCudaStream,
    calls: usize,
    mut submit: impl FnMut() -> Result<(), augur_backend::CudaRuntimeError>,
) -> f32 {
    let started = OwnedCudaEvent::create(stream.device_ordinal()).unwrap();
    let stopped = OwnedCudaEvent::create(stream.device_ordinal()).unwrap();
    started.record(stream).unwrap();
    for _ in 0..calls {
        submit().unwrap();
    }
    stopped.record(stream).unwrap();
    stopped.synchronize().unwrap();
    started.elapsed_ms(&stopped).unwrap()
}

#[test]
fn q5k_sm70_decode_matches_reference_and_replays() {
    let mut compact = vec![0_u8; 176];
    compact[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
    compact[4..16].fill(1);
    for (index, value) in compact[48..].iter_mut().enumerate() {
        *value = ((index * 3) as u8 & 0x0f) | (((index * 5) as u8 & 0x0f) << 4);
    }
    let input = (0..256)
        .map(|index| index as f32 / 256.0)
        .collect::<Vec<_>>();
    let block = Q5KCompactBlock::from_le_bytes(&compact).unwrap();
    let mut decoded = [0.0; 256];
    block.decode_ref(&mut decoded).unwrap();
    let expected = decoded.iter().zip(&input).map(|(a, b)| a * b).sum::<f32>();
    let first = run_q5(&compact, &input)[0];
    let second = run_q5(&compact, &input)[0];
    assert_eq!(first, second);
    assert!((first - expected).abs() <= expected.abs().max(1.0) * 1.0e-4);
}

#[test]
fn q6k_sm70_decode_matches_reference_and_replays() {
    let mut compact = vec![0_u8; 210];
    compact[192..208].fill(1);
    compact[208..210].copy_from_slice(&0x3c00_u16.to_le_bytes());
    for (index, value) in compact[..128].iter_mut().enumerate() {
        *value = (index as u8).wrapping_mul(7);
    }
    for (index, value) in compact[128..192].iter_mut().enumerate() {
        *value = (index as u8).wrapping_mul(11);
    }
    let input = (0..256)
        .map(|index| 1.0 - index as f32 / 512.0)
        .collect::<Vec<_>>();
    let block = Q6KCompactBlock::from_le_bytes(&compact).unwrap();
    let mut decoded = [0.0; 256];
    block.decode_ref(&mut decoded).unwrap();
    let expected = decoded.iter().zip(&input).map(|(a, b)| a * b).sum::<f32>();
    let first = run_q6(&compact, &input)[0];
    let second = run_q6(&compact, &input)[0];
    assert_eq!(first, second);
    assert!((first - expected).abs() <= expected.abs().max(1.0) * 1.0e-4);
}

#[test]
fn q5k_grid32_matches_v1_raw_bytes_and_replays() {
    const COLS: usize = 256;
    for rows in [1, 31, 32, 33] {
        let compact = q5_compact_fixture(rows, COLS);
        let hot = repack_q5k_compact_to_cuda_v1(&compact, rows, COLS).unwrap();
        let input = (0..COLS)
            .map(|index| (index as f32 - 127.0) / 64.0)
            .collect::<Vec<_>>();
        let stream = OwnedCudaStream::create(0).unwrap();
        let weight = OwnedCudaBuffer::allocate(0, hot.payload_bytes().len()).unwrap();
        let input_buffer = OwnedCudaBuffer::allocate(0, input.len() * 4).unwrap();
        let control_output = OwnedCudaBuffer::allocate(0, rows * 4).unwrap();
        let candidate_output = OwnedCudaBuffer::allocate(0, rows * 4).unwrap();
        weight
            .copy_from_host_sync(&stream, hot.payload_bytes())
            .unwrap();
        input_buffer
            .copy_from_host_sync(&stream, &input_bytes(&input))
            .unwrap();
        let weights = CudaQ5kWeightViewV1::new(
            weight.view(0, hot.payload_bytes().len()).unwrap(),
            PhysicalLayoutId::Q5K_HOT_CUDA_V1,
            rows,
            COLS,
        )
        .unwrap();
        let input = CudaF32View::new(
            input_buffer.view(0, COLS * 4).unwrap(),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            COLS,
        )
        .unwrap();
        let control_output_view = CudaF32View::new(
            control_output.view(0, rows * 4).unwrap(),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            rows,
        )
        .unwrap();
        let candidate_output_view = CudaF32View::new(
            candidate_output.view(0, rows * 4).unwrap(),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            rows,
        )
        .unwrap();
        let control =
            CudaQ5kF32DecodeArgsV1::bind(&stream, weights, input, control_output_view).unwrap();
        let candidate =
            CudaQ5kF32DecodeGrid32ArgsV2::bind(&stream, weights, input, candidate_output_view)
                .unwrap();

        let mut outputs = Vec::new();
        for _ in 0..2 {
            control.launch(&stream).unwrap();
            let mut bytes = vec![0_u8; rows * 4];
            control_output
                .copy_to_host_sync(&stream, &mut bytes)
                .unwrap();
            outputs.push(bytes);
        }
        for _ in 0..2 {
            candidate.launch(&stream).unwrap();
            let mut bytes = vec![0_u8; rows * 4];
            candidate_output
                .copy_to_host_sync(&stream, &mut bytes)
                .unwrap();
            outputs.push(bytes);
        }
        assert_eq!(outputs[0], outputs[1]);
        assert_eq!(outputs[0], outputs[2]);
        assert_eq!(outputs[2], outputs[3]);
        assert!(outputs[0]
            .chunks_exact(4)
            .all(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()).is_finite()));
    }
}

#[test]
#[ignore = "M306-K5-R1 hardware performance gate; run explicitly"]
fn q5k_grid32_real_shared_shape_meets_component_gate() {
    const ROWS: usize = 2_048;
    const COLS: usize = 4_096;
    const WARMUP_CALLS: usize = 4;
    const CALLS_PER_SAMPLE: usize = 20;
    const SAMPLES: usize = 7;

    let compact = q5_compact_fixture(ROWS, COLS);
    let hot = repack_q5k_compact_to_cuda_v1(&compact, ROWS, COLS).unwrap();
    let input = (0..COLS)
        .map(|index| ((index * 17 % 257) as f32 - 128.0) / 97.0)
        .collect::<Vec<_>>();
    let stream = OwnedCudaStream::create(0).unwrap();
    let weight = OwnedCudaBuffer::allocate(0, hot.payload_bytes().len()).unwrap();
    let input_buffer = OwnedCudaBuffer::allocate(0, input.len() * 4).unwrap();
    let control_output = OwnedCudaBuffer::allocate(0, ROWS * 4).unwrap();
    let candidate_output = OwnedCudaBuffer::allocate(0, ROWS * 4).unwrap();
    weight
        .copy_from_host_sync(&stream, hot.payload_bytes())
        .unwrap();
    input_buffer
        .copy_from_host_sync(&stream, &input_bytes(&input))
        .unwrap();
    let weights = CudaQ5kWeightViewV1::new(
        weight.view(0, hot.payload_bytes().len()).unwrap(),
        PhysicalLayoutId::Q5K_HOT_CUDA_V1,
        ROWS,
        COLS,
    )
    .unwrap();
    let input = CudaF32View::new(
        input_buffer.view(0, COLS * 4).unwrap(),
        PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
        COLS,
    )
    .unwrap();
    let control = CudaQ5kF32DecodeArgsV1::bind(
        &stream,
        weights,
        input,
        CudaF32View::new(
            control_output.view(0, ROWS * 4).unwrap(),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            ROWS,
        )
        .unwrap(),
    )
    .unwrap();
    let candidate = CudaQ5kF32DecodeGrid32ArgsV2::bind(
        &stream,
        weights,
        input,
        CudaF32View::new(
            candidate_output.view(0, ROWS * 4).unwrap(),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            ROWS,
        )
        .unwrap(),
    )
    .unwrap();

    for _ in 0..WARMUP_CALLS {
        control.submit_for_plan(&stream).unwrap();
        candidate.submit_for_plan(&stream).unwrap();
    }
    stream.synchronize().unwrap();

    let mut control_samples = Vec::with_capacity(SAMPLES);
    let mut candidate_samples = Vec::with_capacity(SAMPLES);
    for sample in 0..SAMPLES {
        if sample % 2 == 0 {
            control_samples.push(elapsed_ms(&stream, CALLS_PER_SAMPLE, || {
                control.submit_for_plan(&stream)
            }));
            candidate_samples.push(elapsed_ms(&stream, CALLS_PER_SAMPLE, || {
                candidate.submit_for_plan(&stream)
            }));
        } else {
            candidate_samples.push(elapsed_ms(&stream, CALLS_PER_SAMPLE, || {
                candidate.submit_for_plan(&stream)
            }));
            control_samples.push(elapsed_ms(&stream, CALLS_PER_SAMPLE, || {
                control.submit_for_plan(&stream)
            }));
        }
    }
    control.launch(&stream).unwrap();
    candidate.launch(&stream).unwrap();
    let mut control_bytes = vec![0_u8; ROWS * 4];
    let mut candidate_bytes = vec![0_u8; ROWS * 4];
    control_output
        .copy_to_host_sync(&stream, &mut control_bytes)
        .unwrap();
    candidate_output
        .copy_to_host_sync(&stream, &mut candidate_bytes)
        .unwrap();
    assert_eq!(control_bytes, candidate_bytes);

    for sample in &mut control_samples {
        *sample /= CALLS_PER_SAMPLE as f32;
    }
    for sample in &mut candidate_samples {
        *sample /= CALLS_PER_SAMPLE as f32;
    }
    let control_median = median(control_samples.clone());
    let candidate_median = median(candidate_samples.clone());
    let ratio = candidate_median / control_median;
    eprintln!(
        "M306_K5_Q5_GRID32 control_samples_ms={control_samples:?} candidate_samples_ms={candidate_samples:?} control_median_ms={control_median:.6} candidate_median_ms={candidate_median:.6} ratio={ratio:.6}"
    );
    assert!(
        ratio <= 0.55,
        "Q5 Grid32 component keep gate failed: {ratio:.6} > 0.550000"
    );
}

#[test]
fn m120_exact_q5k_rejects_nonmatching_shape() {
    let compact = vec![0_u8; 176];
    let input = vec![0.0_f32; 256];
    let hot = repack_q5k_compact_to_cuda_v1(&compact, 1, 256).unwrap();
    let stream = OwnedCudaStream::create(0).unwrap();
    let weight = OwnedCudaBuffer::allocate(0, hot.payload_bytes().len()).unwrap();
    let input_buffer = OwnedCudaBuffer::allocate(0, input.len() * 4).unwrap();
    let output = OwnedCudaBuffer::allocate(0, 4).unwrap();
    weight
        .copy_from_host_sync(&stream, hot.payload_bytes())
        .unwrap();
    input_buffer
        .copy_from_host_sync(&stream, &input_bytes(&input))
        .unwrap();
    let args = CudaQ5kF32DecodeArgsV1::bind(
        &stream,
        CudaQ5kWeightViewV1::new(
            weight.view(0, hot.payload_bytes().len()).unwrap(),
            PhysicalLayoutId::Q5K_HOT_CUDA_V1,
            1,
            256,
        )
        .unwrap(),
        CudaF32View::new(
            input_buffer.view(0, input.len() * 4).unwrap(),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            256,
        )
        .unwrap(),
        CudaF32View::new(
            output.view(0, 4).unwrap(),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            1,
        )
        .unwrap(),
    )
    .unwrap();
    assert!(submit_m120_q5k_exact_sm70(&stream, &args).is_err());
}
