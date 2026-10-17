#![allow(clippy::needless_range_loop)]

use augur_core::F16Bits;
use augur_formats::{
    pack_q4k_scales_ref, quantize_q8k_activation_ref, quantize_q8k_activation_vector_ref,
    Q4KCompactBlock, Q4KScales, ReferenceDecode, Q8K_BLOCK_SIZE,
};
use augur_kernel_api::Q4KHotCpuTensor;
use augur_kernels::portable::{
    dot_q4k_hot_f32_direct, qmatmul_q4k_f32_bias_portable, qmatmul_q4k_f32_bias_silu_portable,
    qmatmul_q4k_f32_portable, silu_f32,
};
use augur_reference::{
    dot_q4k_block_q8k_block_ref, dot_q4k_hot_q8k_ref, qmatmul_q4k_q8k_bias_ref,
    qmatmul_q4k_q8k_bias_silu_ref, qmatmul_q4k_q8k_ref, ReferenceQ8Error,
};
use augur_repack::{decode_q4k_hot_cpu_block_ref, repack_q4k_block_cpu_ref};

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
        d: F16Bits::from_bits(0x3c00),
        dmin: F16Bits::from_bits(0x3800),
        scales: pack_q4k_scales_ref(scales).unwrap(),
        qs,
    }
}

fn sample_weight_2x512() -> Q4KHotCpuTensor {
    let blocks = vec![
        repack_q4k_block_cpu_ref(&sample_block(0)),
        repack_q4k_block_cpu_ref(&sample_block(1)),
        repack_q4k_block_cpu_ref(&sample_block(2)),
        repack_q4k_block_cpu_ref(&sample_block(3)),
    ];

    Q4KHotCpuTensor::new(2, Q8K_BLOCK_SIZE * 2, blocks).unwrap()
}

fn assert_relative_close(a: f32, b: f32, rel_tol: f32) {
    let diff = (a - b).abs();
    let denom = a.abs().max(b.abs()).max(1.0);
    let rel = diff / denom;

    assert!(
        rel <= rel_tol,
        "a={a} b={b} diff={diff} rel={rel} tol={rel_tol}"
    );
}

// ── M9 tests ─────────────────────────────────────────────────────────────

#[test]
fn dot_q4k_hot_q8k_ref_rejects_block_count_mismatch() {
    let compact = sample_block(0);
    let hot = repack_q4k_block_cpu_ref(&compact);
    let weight = vec![hot.clone(), hot];
    let act = vec![quantize_q8k_activation_ref(&vec![0.0f32; 256]).unwrap()]; // only 1

    assert_eq!(
        dot_q4k_hot_q8k_ref(&weight, &act),
        Err(ReferenceQ8Error::BadActivationBlockCount {
            expected: 2,
            actual: 1,
        })
    );
}

#[test]
fn dot_q4k_block_q8k_block_int_ref_matches_f32_dot_close() {
    let compact = sample_block(3);
    let hot = repack_q4k_block_cpu_ref(&compact);
    let mut w_f32 = vec![0.0f32; 256];
    decode_q4k_hot_cpu_block_ref(&hot, &mut w_f32).unwrap();

    let x_f32: Vec<f32> = (0..256).map(|i| ((i % 29) as f32 - 14.0) * 0.37).collect();
    let q8k = quantize_q8k_activation_ref(&x_f32).unwrap();

    let int_dot = dot_q4k_block_q8k_block_ref(&hot, &q8k);

    let mut x_decoded = vec![0.0f32; 256];
    q8k.decode_ref(&mut x_decoded).unwrap();
    let dense_dot: f32 = w_f32.iter().zip(x_decoded.iter()).map(|(w, x)| w * x).sum();

    assert_relative_close(int_dot, dense_dot, 1e-3);
}

#[test]
fn dot_q4k_hot_q8k_int_ref_matches_dense_reference_close() {
    let compact0 = sample_block(0);
    let compact1 = sample_block(7);
    let hot0 = repack_q4k_block_cpu_ref(&compact0);
    let hot1 = repack_q4k_block_cpu_ref(&compact1);

    let mut w_f32 = vec![0.0f32; 512];
    decode_q4k_hot_cpu_block_ref(&hot0, &mut w_f32[0..256]).unwrap();
    decode_q4k_hot_cpu_block_ref(&hot1, &mut w_f32[256..512]).unwrap();

    let x_f32: Vec<f32> = (0..512).map(|i| ((i % 19) as f32 - 9.0) * 0.21).collect();
    let q8k0 = quantize_q8k_activation_ref(&x_f32[0..256]).unwrap();
    let q8k1 = quantize_q8k_activation_ref(&x_f32[256..512]).unwrap();

    let int_dot =
        dot_q4k_hot_q8k_ref(&[hot0.clone(), hot1.clone()], &[q8k0.clone(), q8k1.clone()]).unwrap();

    let mut x_decoded = vec![0.0f32; 512];
    q8k0.decode_ref(&mut x_decoded[0..256]).unwrap();
    q8k1.decode_ref(&mut x_decoded[256..512]).unwrap();
    let dense_dot: f32 = w_f32.iter().zip(x_decoded.iter()).map(|(w, x)| w * x).sum();

    assert_relative_close(int_dot, dense_dot, 1e-3);
}

#[test]
fn dot_q4k_block_q8k_block_int_ref_agrees_with_f32_direct_q4k_dot() {
    let compact = sample_block(5);
    let hot = repack_q4k_block_cpu_ref(&compact);

    let x_f32: Vec<f32> = (0..256)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.03125)
        .collect();
    let q8k = quantize_q8k_activation_ref(&x_f32).unwrap();

    let int_dot = dot_q4k_block_q8k_block_ref(&hot, &q8k);

    let mut x_decoded = vec![0.0f32; 256];
    q8k.decode_ref(&mut x_decoded).unwrap();
    let direct_dot = dot_q4k_hot_f32_direct(&[hot], &x_decoded).unwrap();

    assert_relative_close(int_dot, direct_dot, 1e-3);
}

// ── M10 Q8K qmatmul tests ────────────────────────────────────────────────

#[test]
fn qmatmul_q4k_q8k_ref_matches_f32_q8k_decoded_reference_close() {
    let weight = sample_weight_2x512();

    let src: Vec<f32> = (0..Q8K_BLOCK_SIZE * 2)
        .map(|i| ((i % 37) as f32 - 18.0) * 0.03125)
        .collect();

    let q8 = quantize_q8k_activation_vector_ref(&src).unwrap();

    let mut decoded = vec![0.0f32; src.len()];
    q8.decode_ref(&mut decoded).unwrap();

    let mut expected = vec![0.0f32; 2];
    qmatmul_q4k_f32_portable(&weight, &decoded, &mut expected).unwrap();

    let mut got = vec![0.0f32; 2];
    qmatmul_q4k_q8k_ref(&weight, &q8, &mut got).unwrap();

    assert_relative_close(got[0], expected[0], 1e-3);
    assert_relative_close(got[1], expected[1], 1e-3);
}

#[test]
fn qmatmul_q4k_q8k_bias_ref_matches_f32_q8k_decoded_reference_close() {
    let weight = sample_weight_2x512();

    let src: Vec<f32> = (0..Q8K_BLOCK_SIZE * 2)
        .map(|i| ((i % 41) as f32 - 20.0) * 0.015625)
        .collect();

    let q8 = quantize_q8k_activation_vector_ref(&src).unwrap();

    let mut decoded = vec![0.0f32; src.len()];
    q8.decode_ref(&mut decoded).unwrap();

    let bias = vec![0.25f32, -0.5f32];

    let mut expected = vec![0.0f32; 2];
    qmatmul_q4k_f32_bias_portable(&weight, &decoded, &bias, &mut expected).unwrap();

    let mut got = vec![0.0f32; 2];
    qmatmul_q4k_q8k_bias_ref(&weight, &q8, &bias, &mut got).unwrap();

    assert_relative_close(got[0], expected[0], 1e-3);
    assert_relative_close(got[1], expected[1], 1e-3);
}

#[test]
fn qmatmul_q4k_q8k_bias_silu_ref_matches_f32_q8k_decoded_reference_close() {
    let weight = sample_weight_2x512();

    let src: Vec<f32> = (0..Q8K_BLOCK_SIZE * 2)
        .map(|i| ((i % 43) as f32 - 21.0) * 0.015625)
        .collect();

    let q8 = quantize_q8k_activation_vector_ref(&src).unwrap();

    let mut decoded = vec![0.0f32; src.len()];
    q8.decode_ref(&mut decoded).unwrap();

    let bias = vec![0.125f32, -0.25f32];

    let mut expected = vec![0.0f32; 2];
    qmatmul_q4k_f32_bias_silu_portable(&weight, &decoded, &bias, &mut expected).unwrap();

    let mut got = vec![0.0f32; 2];
    qmatmul_q4k_q8k_bias_silu_ref(&weight, &q8, &bias, &mut got).unwrap();

    assert_relative_close(got[0], expected[0], 1e-3);
    assert_relative_close(got[1], expected[1], 1e-3);
}

#[test]
fn qmatmul_q4k_q8k_rejects_activation_length_mismatch() {
    let weight = sample_weight_2x512();

    let src = vec![0.0f32; Q8K_BLOCK_SIZE];
    let q8 = quantize_q8k_activation_vector_ref(&src).unwrap();

    let mut out = vec![0.0f32; 2];

    assert_eq!(
        qmatmul_q4k_q8k_ref(&weight, &q8, &mut out),
        Err(ReferenceQ8Error::BadActivationLength {
            expected: Q8K_BLOCK_SIZE * 2,
            actual: Q8K_BLOCK_SIZE,
        })
    );
}

// ── M11 BitExact tests ───────────────────────────────────────────────────

#[test]
fn qmatmul_q4k_q8k_ref_matches_checked_row_dot_bit_exact() {
    let weight = sample_weight_2x512();

    let src: Vec<f32> = (0..Q8K_BLOCK_SIZE * 2)
        .map(|i| ((i % 47) as f32 - 23.0) * 0.015625)
        .collect();

    let q8 = quantize_q8k_activation_vector_ref(&src).unwrap();

    let mut got = vec![0.0f32; 2];
    qmatmul_q4k_q8k_ref(&weight, &q8, &mut got).unwrap();

    let row0 = dot_q4k_hot_q8k_ref(weight.row_blocks(0).unwrap(), q8.blocks()).unwrap();
    let row1 = dot_q4k_hot_q8k_ref(weight.row_blocks(1).unwrap(), q8.blocks()).unwrap();

    assert_eq!(got[0], row0);
    assert_eq!(got[1], row1);
}

#[test]
fn qmatmul_q4k_q8k_bias_ref_matches_checked_row_dot_bit_exact() {
    let weight = sample_weight_2x512();

    let src: Vec<f32> = (0..Q8K_BLOCK_SIZE * 2)
        .map(|i| ((i % 53) as f32 - 26.0) * 0.0078125)
        .collect();

    let q8 = quantize_q8k_activation_vector_ref(&src).unwrap();
    let bias = vec![0.25f32, -0.75f32];

    let mut got = vec![0.0f32; 2];
    qmatmul_q4k_q8k_bias_ref(&weight, &q8, &bias, &mut got).unwrap();

    let row0 = dot_q4k_hot_q8k_ref(weight.row_blocks(0).unwrap(), q8.blocks()).unwrap();
    let row1 = dot_q4k_hot_q8k_ref(weight.row_blocks(1).unwrap(), q8.blocks()).unwrap();

    assert_eq!(got[0], row0 + bias[0]);
    assert_eq!(got[1], row1 + bias[1]);
}

#[test]
fn qmatmul_q4k_q8k_bias_silu_ref_matches_checked_row_dot_bit_exact() {
    let weight = sample_weight_2x512();

    let src: Vec<f32> = (0..Q8K_BLOCK_SIZE * 2)
        .map(|i| ((i % 59) as f32 - 29.0) * 0.0078125)
        .collect();

    let q8 = quantize_q8k_activation_vector_ref(&src).unwrap();
    let bias = vec![0.125f32, -0.5f32];

    let mut got = vec![0.0f32; 2];
    qmatmul_q4k_q8k_bias_silu_ref(&weight, &q8, &bias, &mut got).unwrap();

    let row0 = dot_q4k_hot_q8k_ref(weight.row_blocks(0).unwrap(), q8.blocks()).unwrap();
    let row1 = dot_q4k_hot_q8k_ref(weight.row_blocks(1).unwrap(), q8.blocks()).unwrap();

    assert_eq!(got[0], silu_f32(row0 + bias[0]));
    assert_eq!(got[1], silu_f32(row1 + bias[1]));
}
