#![cfg(feature = "cuda")]

use augur_backend::{
    CudaBf16F32MatvecArgsV1, CudaBf16TokenRowGatherArgsV1, CudaBf16WeightViewV1, CudaF32View,
    OwnedCudaBuffer, OwnedCudaStream,
};
use augur_core::PhysicalLayoutId;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    (rounded >> 16) as u16
}

fn run(rows: usize, cols: usize, weights: &[u16], input: &[f32]) -> Vec<f32> {
    let stream = OwnedCudaStream::create(0).unwrap();
    let weight_bytes = weights
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let weight = OwnedCudaBuffer::allocate(0, weight_bytes.len()).unwrap();
    let input_buffer = OwnedCudaBuffer::allocate(0, input.len() * 4).unwrap();
    let output = OwnedCudaBuffer::allocate(0, rows * 4).unwrap();
    weight.copy_from_host_sync(&stream, &weight_bytes).unwrap();
    input_buffer
        .copy_from_host_sync(&stream, &f32_bytes(input))
        .unwrap();
    let args = CudaBf16F32MatvecArgsV1::bind(
        &stream,
        CudaBf16WeightViewV1::new(
            weight.view(0, weight_bytes.len()).unwrap(),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_BF16_CUDA_V1,
            rows,
            cols,
        )
        .unwrap(),
        CudaF32View::new(
            input_buffer.view(0, input.len() * 4).unwrap(),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            cols,
        )
        .unwrap(),
        CudaF32View::new(
            output.view(0, rows * 4).unwrap(),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            rows,
        )
        .unwrap(),
    )
    .unwrap();
    args.launch(&stream).unwrap();
    let mut bytes = vec![0_u8; rows * 4];
    output.copy_to_host_sync(&stream, &mut bytes).unwrap();
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn reference(rows: usize, cols: usize, weights: &[u16], input: &[f32]) -> Vec<f32> {
    (0..rows)
        .map(|row| {
            let mut sum = 0.0_f64;
            for col in 0..cols {
                let weight = f32::from_bits(u32::from(weights[row * cols + col]) << 16);
                sum += f64::from(weight * input[col]);
            }
            sum as f32
        })
        .collect()
}

fn fixture(rows: usize, cols: usize) -> (Vec<u16>, Vec<f32>) {
    let weights = (0..rows * cols)
        .map(|index| bf16_bits(((index % 29) as f32 - 14.0) / 8.0))
        .collect::<Vec<_>>();
    let input = (0..cols)
        .map(|index| ((index % 23) as f32 - 11.0) / 16.0)
        .collect::<Vec<_>>();
    (weights, input)
}

#[test]
fn staged_bf16_matvec_matches_reference_and_replays() {
    let (weights, input) = fixture(257, 64);
    let expected = reference(257, 64, &weights, &input);
    let first = run(257, 64, &weights, &input);
    let second = run(257, 64, &weights, &input);
    assert_eq!(first, second);
    assert_eq!(first, expected);
}

#[test]
fn non_tile_shape_fallback_matches_reference() {
    let (weights, input) = fixture(3, 63);
    let expected = reference(3, 63, &weights, &input);
    assert_eq!(run(3, 63, &weights, &input), expected);
}

#[test]
fn bf16_weight_view_rejects_wrong_layout() {
    let buffer = OwnedCudaBuffer::allocate(0, 128).unwrap();
    assert!(CudaBf16WeightViewV1::new(
        buffer.view(0, 128).unwrap(),
        PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
        1,
        64,
    )
    .is_err());
}

#[test]
fn token_row_gather_decodes_selected_bf16_row_and_replays() {
    let stream = OwnedCudaStream::create(0).unwrap();
    let rows = 7;
    let cols = 256;
    let weights = (0..rows * cols)
        .map(|index| bf16_bits((index as f32 - 500.0) / 128.0))
        .collect::<Vec<_>>();
    let weight_bytes = weights
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let weight = OwnedCudaBuffer::allocate(0, weight_bytes.len()).unwrap();
    let token = OwnedCudaBuffer::allocate(0, 4).unwrap();
    let output = OwnedCudaBuffer::allocate(0, cols * 4).unwrap();
    weight.copy_from_host_sync(&stream, &weight_bytes).unwrap();
    token
        .copy_from_host_sync(&stream, &3_u32.to_le_bytes())
        .unwrap();
    let args = CudaBf16TokenRowGatherArgsV1::bind(
        &stream,
        CudaBf16WeightViewV1::new(
            weight.view(0, weight_bytes.len()).unwrap(),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_BF16_CUDA_V1,
            rows,
            cols,
        )
        .unwrap(),
        token.view(0, 4).unwrap(),
        CudaF32View::new(
            output.view(0, cols * 4).unwrap(),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            cols,
        )
        .unwrap(),
    )
    .unwrap();
    let mut captures = Vec::new();
    for _ in 0..2 {
        args.launch(&stream).unwrap();
        let mut bytes = vec![0_u8; cols * 4];
        output.copy_to_host_sync(&stream, &mut bytes).unwrap();
        captures.push(bytes);
    }
    assert_eq!(captures[0], captures[1]);
    for (col, bytes) in captures[0].chunks_exact(4).enumerate() {
        let actual = f32::from_le_bytes(bytes.try_into().unwrap());
        let expected = f32::from_bits(u32::from(weights[3 * cols + col]) << 16);
        assert_eq!(actual, expected);
    }
}

#[test]
fn token_row_gather_marks_out_of_range_token_nonfinite() {
    let stream = OwnedCudaStream::create(0).unwrap();
    let rows = 2;
    let cols = 16;
    let weight = OwnedCudaBuffer::allocate(0, rows * cols * 2).unwrap();
    let token = OwnedCudaBuffer::allocate(0, 4).unwrap();
    let output = OwnedCudaBuffer::allocate(0, cols * 4).unwrap();
    weight
        .copy_from_host_sync(&stream, &vec![0_u8; rows * cols * 2])
        .unwrap();
    token
        .copy_from_host_sync(&stream, &u32::MAX.to_le_bytes())
        .unwrap();
    CudaBf16TokenRowGatherArgsV1::bind(
        &stream,
        CudaBf16WeightViewV1::new(
            weight.view(0, rows * cols * 2).unwrap(),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_BF16_CUDA_V1,
            rows,
            cols,
        )
        .unwrap(),
        token.view(0, 4).unwrap(),
        CudaF32View::new(
            output.view(0, cols * 4).unwrap(),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            cols,
        )
        .unwrap(),
    )
    .unwrap()
    .launch(&stream)
    .unwrap();
    let mut bytes = vec![0_u8; cols * 4];
    output.copy_to_host_sync(&stream, &mut bytes).unwrap();
    assert!(bytes
        .chunks_exact(4)
        .all(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()).is_nan()));
}
