#![allow(clippy::needless_range_loop)]

use augur_core::F16Bits;
use augur_formats::{pack_q4k_scales_ref, Q4KCompactBlock, Q4KScales};
use augur_kernel_api::{Q4KHotCpuBlock, Q4KHotCpuTensor};
use augur_kernels::portable::{
    decode_q4k_hot_f32, dot_q4k_hot_f32, dot_q4k_hot_f32_direct, qmatmul_q4k_f32_bias_portable,
    qmatmul_q4k_f32_bias_silu_portable, qmatmul_q4k_f32_portable, silu_f32,
};
use augur_kernels::KernelError;
use augur_repack::{decode_q4k_hot_cpu_block_ref, repack_q4k_block_cpu_ref, TensorError};

fn sample_block(seed: u8) -> Q4KCompactBlock {
    let scales = Q4KScales {
        scale: [1 + (seed & 3), 2, 3, 4, 17, 18, 35, 63],
        min: [5, 6, 7, 8, 9, 26, 42, 61],
    };
    let mut qs = [0u8; 128];
    for (i, q) in qs.iter_mut().enumerate() {
        let lo = ((i as u8).wrapping_add(seed)) & 0x0f;
        let hi = (15u8.wrapping_sub(i as u8).wrapping_add(seed)) & 0x0f;
        *q = lo | (hi << 4);
    }
    Q4KCompactBlock {
        d: F16Bits::from_bits(0x3c00),    // 1.0
        dmin: F16Bits::from_bits(0x3800), // 0.5
        scales: pack_q4k_scales_ref(scales).unwrap(),
        qs,
    }
}

/// Build a small 2×256 Q4KHotCpuTensor (2 rows, 256 cols = 1 block/row).
fn sample_tensor_2x256() -> Q4KHotCpuTensor {
    let blocks = vec![
        repack_q4k_block_cpu_ref(&sample_block(0)),
        repack_q4k_block_cpu_ref(&sample_block(7)),
    ];
    Q4KHotCpuTensor::new(2, 256, blocks).unwrap()
}

fn sample_activation_256() -> Vec<f32> {
    (0..256).map(|i| ((i % 17) as f32 - 8.0) * 0.125).collect()
}

// ── Existing M3–M6 tests ─────────────────────────────────────────────────

#[test]
fn decode_q4k_hot_f32_matches_hot_block_decode() {
    let compact0 = sample_block(0);
    let compact1 = sample_block(3);
    let hot = vec![
        repack_q4k_block_cpu_ref(&compact0),
        repack_q4k_block_cpu_ref(&compact1),
    ];

    let mut kernel_out = vec![0.0f32; hot.len() * Q4KHotCpuBlock::DECODED_LEN];
    decode_q4k_hot_f32(&hot, &mut kernel_out).unwrap();

    let mut expected = vec![0.0f32; hot.len() * Q4KHotCpuBlock::DECODED_LEN];
    decode_q4k_hot_cpu_block_ref(&hot[0], &mut expected[0..Q4KHotCpuBlock::DECODED_LEN]).unwrap();
    decode_q4k_hot_cpu_block_ref(&hot[1], &mut expected[Q4KHotCpuBlock::DECODED_LEN..]).unwrap();

    assert_eq!(kernel_out, expected);
}

#[test]
fn decode_q4k_hot_f32_rejects_short_output() {
    let hot = vec![repack_q4k_block_cpu_ref(&sample_block(0))];
    let mut out = vec![0.0f32; Q4KHotCpuBlock::DECODED_LEN - 1];
    assert_eq!(
        decode_q4k_hot_f32(&hot, &mut out),
        Err(KernelError::BadOutputLength {
            expected: Q4KHotCpuBlock::DECODED_LEN,
            actual: Q4KHotCpuBlock::DECODED_LEN - 1,
        })
    );
}

#[test]
fn dot_q4k_hot_f32_matches_dense_reference() {
    let compact0 = sample_block(0);
    let compact1 = sample_block(7);
    let hot = vec![
        repack_q4k_block_cpu_ref(&compact0),
        repack_q4k_block_cpu_ref(&compact1),
    ];
    let n = hot.len() * Q4KHotCpuBlock::DECODED_LEN;
    let activation: Vec<f32> = (0..n).map(|i| ((i % 17) as f32 - 8.0) * 0.125).collect();

    let mut dense_w = vec![0.0f32; n];
    decode_q4k_hot_f32(&hot, &mut dense_w).unwrap();
    let expected: f32 = dense_w
        .iter()
        .zip(activation.iter())
        .map(|(w, x)| w * x)
        .sum();
    let got = dot_q4k_hot_f32(&hot, &activation).unwrap();
    assert_eq!(got, expected);
}

#[test]
fn dot_q4k_hot_f32_direct_matches_decode_buffer_bit_exact() {
    let compact0 = sample_block(0);
    let compact1 = sample_block(7);
    let hot = vec![
        repack_q4k_block_cpu_ref(&compact0),
        repack_q4k_block_cpu_ref(&compact1),
    ];
    let n = hot.len() * Q4KHotCpuBlock::DECODED_LEN;
    let activation: Vec<f32> = (0..n).map(|i| ((i % 17) as f32 - 8.0) * 0.125).collect();

    let buffered = dot_q4k_hot_f32(&hot, &activation).unwrap();
    let direct = dot_q4k_hot_f32_direct(&hot, &activation).unwrap();

    assert_eq!(direct, buffered);
}

#[test]
fn dot_q4k_hot_f32_direct_rejects_bad_activation_length() {
    let hot = vec![repack_q4k_block_cpu_ref(&sample_block(0))];
    let activation = vec![0.0f32; Q4KHotCpuBlock::DECODED_LEN - 1];
    assert_eq!(
        dot_q4k_hot_f32_direct(&hot, &activation),
        Err(KernelError::BadActivationLength {
            expected: Q4KHotCpuBlock::DECODED_LEN,
            actual: Q4KHotCpuBlock::DECODED_LEN - 1,
        })
    );
}

#[test]
fn dot_q4k_hot_f32_rejects_bad_activation_length() {
    let hot = vec![repack_q4k_block_cpu_ref(&sample_block(0))];
    let activation = vec![0.0f32; Q4KHotCpuBlock::DECODED_LEN - 1];
    assert_eq!(
        dot_q4k_hot_f32(&hot, &activation),
        Err(KernelError::BadActivationLength {
            expected: Q4KHotCpuBlock::DECODED_LEN,
            actual: Q4KHotCpuBlock::DECODED_LEN - 1,
        })
    );
}

// ── M7 fused qmatmul tests ───────────────────────────────────────────────

#[test]
fn qmatmul_q4k_f32_portable_matches_row_dot() {
    let tensor = sample_tensor_2x256();
    let act = sample_activation_256();
    let mut out = vec![0.0f32; 2];

    qmatmul_q4k_f32_portable(&tensor, &act, &mut out).unwrap();

    // Compare with per-row dot_q4k_hot_f32_direct
    for row in 0..2 {
        let row_blocks = tensor.row_blocks(row).unwrap();
        let expected = dot_q4k_hot_f32_direct(row_blocks, &act).unwrap();
        assert_eq!(out[row], expected);
    }
}

#[test]
fn qmatmul_q4k_f32_bias_portable_matches_dense_reference() {
    let tensor = sample_tensor_2x256();
    let act = sample_activation_256();
    let bias = vec![1.5_f32, -0.5_f32];
    let mut out = vec![0.0f32; 2];

    qmatmul_q4k_f32_bias_portable(&tensor, &act, &bias, &mut out).unwrap();

    for row in 0..2 {
        let row_blocks = tensor.row_blocks(row).unwrap();
        let expected = dot_q4k_hot_f32_direct(row_blocks, &act).unwrap() + bias[row];
        assert_eq!(out[row], expected);
    }
}

#[test]
fn qmatmul_q4k_f32_bias_silu_portable_matches_dense_reference() {
    let tensor = sample_tensor_2x256();
    let act = sample_activation_256();
    let bias = vec![1.5_f32, -0.5_f32];
    let mut out = vec![0.0f32; 2];

    qmatmul_q4k_f32_bias_silu_portable(&tensor, &act, &bias, &mut out).unwrap();

    for row in 0..2 {
        let row_blocks = tensor.row_blocks(row).unwrap();
        let x = dot_q4k_hot_f32_direct(row_blocks, &act).unwrap() + bias[row];
        let expected = silu_f32(x);
        assert_eq!(out[row], expected);
    }
}

// ── M7 error-path tests ──────────────────────────────────────────────────

#[test]
fn qmatmul_q4k_f32_rejects_bad_input_length() {
    let tensor = sample_tensor_2x256();
    let act = vec![0.0f32; 128]; // too short
    let mut out = vec![0.0f32; 2];
    assert_eq!(
        qmatmul_q4k_f32_portable(&tensor, &act, &mut out),
        Err(KernelError::BadActivationLength {
            expected: 256,
            actual: 128,
        })
    );
}

#[test]
fn qmatmul_q4k_f32_bias_rejects_short_bias() {
    let tensor = sample_tensor_2x256();
    let act = sample_activation_256();
    let bias = vec![0.0f32; 1]; // too short
    let mut out = vec![0.0f32; 2];
    assert_eq!(
        qmatmul_q4k_f32_bias_portable(&tensor, &act, &bias, &mut out),
        Err(KernelError::BadBiasLength {
            expected: 2,
            actual: 1,
        })
    );
}

#[test]
fn qmatmul_q4k_f32_rejects_short_output() {
    let tensor = sample_tensor_2x256();
    let act = sample_activation_256();
    let mut out = vec![0.0f32; 1]; // too short
    assert_eq!(
        qmatmul_q4k_f32_portable(&tensor, &act, &mut out),
        Err(KernelError::BadOutputLength {
            expected: 2,
            actual: 1,
        })
    );
}

#[test]
fn q4k_hot_cpu_tensor_rejects_bad_block_count() {
    let blocks = vec![repack_q4k_block_cpu_ref(&sample_block(0))]; // only 1, need 2
    let err = Q4KHotCpuTensor::new(2, 256, blocks).unwrap_err();
    assert_eq!(
        err,
        TensorError::BadBlockCount {
            expected: 2,
            actual: 1,
        }
    );
}

// ── M8 bit-exact tests (2×512: 2 rows × 2 blocks/row) ───────────────────

/// Build a 2×512 tensor (4 blocks total).
fn sample_tensor_2x512() -> Q4KHotCpuTensor {
    let blocks = vec![
        repack_q4k_block_cpu_ref(&sample_block(0)),
        repack_q4k_block_cpu_ref(&sample_block(7)),
        repack_q4k_block_cpu_ref(&sample_block(3)),
        repack_q4k_block_cpu_ref(&sample_block(11)),
    ];
    Q4KHotCpuTensor::new(2, 512, blocks).unwrap()
}

fn sample_activation_512() -> Vec<f32> {
    (0..512).map(|i| ((i % 19) as f32 - 9.0) * 0.0625).collect()
}

#[test]
fn qmatmul_q4k_f32_portable_matches_public_row_dot_bit_exact() {
    let weight = sample_tensor_2x512();
    let input = sample_activation_512();
    let mut out = vec![0.0f32; 2];

    qmatmul_q4k_f32_portable(&weight, &input, &mut out).unwrap();

    let row0 = dot_q4k_hot_f32_direct(weight.row_blocks(0).unwrap(), &input).unwrap();
    let row1 = dot_q4k_hot_f32_direct(weight.row_blocks(1).unwrap(), &input).unwrap();

    assert_eq!(out[0], row0);
    assert_eq!(out[1], row1);
}

#[test]
fn qmatmul_q4k_f32_bias_portable_matches_public_row_dot_bit_exact() {
    let weight = sample_tensor_2x512();
    let input: Vec<f32> = (0..512)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.03125)
        .collect();
    let bias = vec![0.25f32, -0.75f32];
    let mut out = vec![0.0f32; 2];

    qmatmul_q4k_f32_bias_portable(&weight, &input, &bias, &mut out).unwrap();

    let row0 = dot_q4k_hot_f32_direct(weight.row_blocks(0).unwrap(), &input).unwrap() + bias[0];
    let row1 = dot_q4k_hot_f32_direct(weight.row_blocks(1).unwrap(), &input).unwrap() + bias[1];

    assert_eq!(out[0], row0);
    assert_eq!(out[1], row1);
}

#[test]
fn qmatmul_q4k_f32_bias_silu_portable_matches_public_row_dot_bit_exact() {
    let weight = sample_tensor_2x512();
    let input: Vec<f32> = (0..512)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.015625)
        .collect();
    let bias = vec![0.125f32, -0.25f32];
    let mut out = vec![0.0f32; 2];

    qmatmul_q4k_f32_bias_silu_portable(&weight, &input, &bias, &mut out).unwrap();

    let row0 =
        silu_f32(dot_q4k_hot_f32_direct(weight.row_blocks(0).unwrap(), &input).unwrap() + bias[0]);
    let row1 =
        silu_f32(dot_q4k_hot_f32_direct(weight.row_blocks(1).unwrap(), &input).unwrap() + bias[1]);

    assert_eq!(out[0], row0);
    assert_eq!(out[1], row1);
}
