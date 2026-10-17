#![cfg(feature = "cuda")]

use augur_backend::{
    CudaF32View, CudaQ4kF32DecodeArgsV1, CudaQ4kWeightViewV1, OwnedCudaBuffer, OwnedCudaStream,
};
use augur_core::{F16Bits, PhysicalLayoutId};
use augur_formats::{Q4KCompactBlock, Q4K_QS_BYTES, Q4K_SCALE_BYTES};
use augur_kernel_api::{Q4KHotCpuBlock, Q4KHotCpuTensor, Q4KHotCudaStagingTensorV1};
use augur_kernels::portable::qmatmul_q4k_f32_portable;
use augur_repack::{repack_q4k_block_cpu_ref, repack_q4k_hot_cpu_to_cuda_v1};

const TOLERANCE_ABS: f32 = 1.0e-4;

fn assert_q4k_close(actual: f32, expected: f32, row: usize) {
    let abs = (actual - expected).abs();
    let scale = expected.abs().max(1.0);
    assert!(
        abs <= TOLERANCE_ABS * scale,
        "row={row}: actual={actual}, expected={expected}, abs={abs}, tol={}",
        TOLERANCE_ABS * scale,
    );
}

fn f32_bytes_le(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn output_f32(device_bytes: &[u8], rows: usize) -> Vec<f32> {
    (0..rows)
        .map(|row| {
            let start = row * 4;
            f32::from_le_bytes(device_bytes[start..start + 4].try_into().unwrap())
        })
        .collect()
}

fn cuda_weight_and_hot(rows: usize, cols: usize) -> (Q4KHotCpuTensor, Q4KHotCudaStagingTensorV1) {
    assert!(cols.is_multiple_of(Q4KHotCpuBlock::DECODED_LEN));
    let blocks_per_row = cols / Q4KHotCpuBlock::DECODED_LEN;
    let block_count = rows * blocks_per_row;

    let compact = (0..block_count)
        .map(|index| {
            let mut scale = [0_u8; Q4K_SCALE_BYTES];
            let mut qs = [0_u8; Q4K_QS_BYTES];

            for (byte, value) in scale.iter_mut().enumerate() {
                *value = (index.wrapping_mul(7).wrapping_add(byte)) as u8;
            }
            for (nibble, value) in qs.iter_mut().enumerate() {
                let low = (index.wrapping_add(nibble * 3) % 15) as u8;
                let high = (index.wrapping_add(nibble * 5 + 7) % 15) as u8;
                *value = low | (high << 4);
            }

            Q4KCompactBlock {
                d: F16Bits::from_f32(0.5 + (index % 4) as f32 * 0.125),
                dmin: F16Bits::from_f32(0.125 + (index % 3) as f32 * 0.0625),
                scales: scale,
                qs,
            }
        })
        .collect::<Vec<_>>();

    let hot_blocks = compact
        .iter()
        .map(repack_q4k_block_cpu_ref)
        .collect::<Vec<_>>();

    let hot = Q4KHotCpuTensor::new(rows, cols, hot_blocks).unwrap();
    let cuda = repack_q4k_hot_cpu_to_cuda_v1(&hot).unwrap();

    (hot, cuda)
}

fn run_kernel_on_device(
    _weight_hot: &Q4KHotCpuTensor,
    cuda_staging: &Q4KHotCudaStagingTensorV1,
    input_f32: &[f32],
) -> Vec<f32> {
    let rows = cuda_staging.rows();
    let cols = cuda_staging.cols();
    let weight_bytes = cuda_staging.payload_bytes().to_vec();
    let weight_len = weight_bytes.len();
    let input_len = input_f32.len() * 4;
    let output_len = rows * 4;

    let stream = OwnedCudaStream::create(0).unwrap();
    let weight_buf = OwnedCudaBuffer::allocate(0, weight_len).unwrap();
    let input_buf = OwnedCudaBuffer::allocate(0, input_len).unwrap();
    let output_buf = OwnedCudaBuffer::allocate(0, output_len).unwrap();

    weight_buf
        .copy_from_host_sync(&stream, &weight_bytes)
        .unwrap();
    input_buf
        .copy_from_host_sync(&stream, &f32_bytes_le(input_f32))
        .unwrap();

    let weight_view = CudaQ4kWeightViewV1::new(
        weight_buf.view(0, weight_len).unwrap(),
        PhysicalLayoutId::Q4K_HOT_CUDA_V1,
        rows,
        cols,
    )
    .unwrap();

    let input_view = CudaF32View::new(
        input_buf.view(0, input_len).unwrap(),
        PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
        input_f32.len(),
    )
    .unwrap();

    let output_view = CudaF32View::new(
        output_buf.view(0, output_len).unwrap(),
        PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
        rows,
    )
    .unwrap();

    let args = CudaQ4kF32DecodeArgsV1::bind(&stream, weight_view, input_view, output_view).unwrap();

    args.launch(&stream).unwrap();

    let mut output_bytes = vec![0_u8; output_len];
    output_buf
        .copy_to_host_sync(&stream, &mut output_bytes)
        .unwrap();

    output_f32(&output_bytes, rows)
}

fn cpu_reference(weight: &Q4KHotCpuTensor, input: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0_f32; weight.rows()];
    qmatmul_q4k_f32_portable(weight, input, &mut output).unwrap();
    output
}

// ── Numerical correctness ────────────────────────────────────────────────

#[test]
fn cuda_q4k_one_block_matches_cpu_reference() {
    let (hot, cuda) = cuda_weight_and_hot(1, 256);
    let input = (0..256)
        .map(|i| 0.5 - (i as f32) / 512.0)
        .collect::<Vec<_>>();

    let device = run_kernel_on_device(&hot, &cuda, &input);
    let host = cpu_reference(&hot, &input);

    assert_eq!(device.len(), host.len());
    for row in 0..device.len() {
        assert_q4k_close(device[row], host[row], row);
    }
}

#[test]
fn cuda_q4k_one_block_constant_activation() {
    let (hot, cuda) = cuda_weight_and_hot(1, 256);
    let input = vec![1.0_f32; 256];

    let device = run_kernel_on_device(&hot, &cuda, &input);
    let host = cpu_reference(&hot, &input);

    assert_eq!(device.len(), host.len());
    for row in 0..device.len() {
        assert_q4k_close(device[row], host[row], row);
    }
}

#[test]
fn cuda_q4k_multi_row_matches_cpu_reference() {
    for (rows, cols) in [(1, 512), (2, 256), (2, 512)] {
        let (hot, cuda) = cuda_weight_and_hot(rows, cols);
        let input = (0..cols)
            .map(|i| 0.5 - (i as f32) / (cols as f32 * 2.0))
            .collect::<Vec<_>>();

        let device = run_kernel_on_device(&hot, &cuda, &input);
        let host = cpu_reference(&hot, &input);

        assert_eq!(device.len(), rows);
        for row in 0..rows {
            assert_q4k_close(device[row], host[row], row);
        }
    }
}

// ── Repeated launch ─────────────────────────────────────────────────────

#[test]
fn cuda_q4k_repeated_launch_consistent() {
    let (hot, cuda) = cuda_weight_and_hot(1, 256);
    let input = (0..256)
        .map(|i| 0.5 - (i as f32) / 512.0)
        .collect::<Vec<_>>();

    let first = run_kernel_on_device(&hot, &cuda, &input);
    for _ in 0..8 {
        let repeat = run_kernel_on_device(&hot, &cuda, &input);
        assert_eq!(repeat.len(), first.len());
        for row in 0..repeat.len() {
            assert_eq!(repeat[row], first[row]);
        }
    }
}

// Invalid ABI version test moved to cuda_q4k.rs internal test module
// where pub(crate) raw() is accessible.
