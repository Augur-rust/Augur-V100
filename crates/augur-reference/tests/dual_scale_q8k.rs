#![allow(clippy::needless_range_loop)]

use augur_core::F16Bits;
use augur_formats::dual_scale::decode_dual_scale_main_block_ref;
use augur_formats::{
    encode_dual_scale_block_ref, quantize_q8k_activation_ref, quantize_q8k_activation_vector_ref,
    Q8KActivationBlock, ReferenceDecode, DUAL_SCALE_BLOCK_SIZE, Q8K_BLOCK_SIZE,
};
use augur_kernel_api::{
    DualScaleHotCpuBlock, DualScaleHotCpuTensor, DualScaleQ8HotCpuLaunch, OutlierCsrHotCpu,
    Q8KActivationHotCpuBlock, Q8KActivationHotCpuVector,
};
use augur_kernels::portable::{
    dot_dual_scale_block_q8k_hot_cpu_grouped_ref, dot_dual_scale_hot_f32_grouped_ref,
    dot_dual_scale_hot_f32_main_ref, dot_dual_scale_hot_q8k_hot_cpu_grouped_ref,
    launch_qmatvec_dual_scale_q8k_hot_cpu_portable, qmatvec_dual_scale_f32_portable,
    qmatvec_dual_scale_q8k_hot_cpu_portable,
};
use augur_kernels::KernelError;
use augur_reference::{
    dot_dual_scale_block_q8k_grouped_ref, dot_dual_scale_hot_q8k_grouped_ref,
    qmatvec_dual_scale_q8k_ref, ReferenceQ8Error,
};
use augur_repack::{
    repack_dual_scale_block_cpu_ref, repack_dual_scale_tensor_cpu_ref,
    repack_q8k_activation_block_cpu_ref, repack_q8k_activation_vector_cpu_ref,
};

fn make_source_block(seed: usize, outlier_idx: usize, outlier_value: f32) -> Vec<f32> {
    let mut src: Vec<f32> = (0..DUAL_SCALE_BLOCK_SIZE)
        .map(|i| (((i + seed) % 37) as f32 - 18.0) * 0.03125)
        .collect();

    src[outlier_idx] = outlier_value;
    src
}

fn make_compact_2x512() -> Vec<augur_formats::DualScaleCompactBlock> {
    let blocks = [
        make_source_block(0, 5, 48.0),
        make_source_block(1, 37, -56.0),
        make_source_block(2, 91, 64.0),
        make_source_block(3, 123, -72.0),
    ];

    blocks
        .iter()
        .map(|src| encode_dual_scale_block_ref(src).unwrap())
        .collect()
}

fn activation_512() -> Vec<f32> {
    (0..DUAL_SCALE_BLOCK_SIZE * 2)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.0625)
        .collect()
}

fn make_source_block_no_outlier(seed: usize) -> Vec<f32> {
    (0..DUAL_SCALE_BLOCK_SIZE)
        .map(|i| (((i + seed) % 37) as f32 - 18.0) * 0.03125)
        .collect()
}

fn assert_close(a: f32, b: f32, abs_tol: f32, rel_tol: f32) {
    let diff = (a - b).abs();
    let scale = a.abs().max(b.abs()).max(1.0);
    let rel = diff / scale;

    assert!(
        diff <= abs_tol || rel <= rel_tol,
        "a={a} b={b} diff={diff} rel={rel} abs_tol={abs_tol} rel_tol={rel_tol}"
    );
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

// ── Main dot tests ────────────────────────────────────────────────────────

#[test]
fn dual_scale_main_dot_matches_main_decode_dense_dot_bit_exact() {
    let src = make_source_block(7, 17, 48.0);
    let compact = encode_dual_scale_block_ref(&src).unwrap();
    let hot = repack_dual_scale_block_cpu_ref(&compact);

    let activation: Vec<f32> = (0..DUAL_SCALE_BLOCK_SIZE)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.03125)
        .collect();

    let got = dot_dual_scale_hot_f32_main_ref(std::slice::from_ref(&hot), &activation);

    let mut decoded = vec![0.0f32; DUAL_SCALE_BLOCK_SIZE];
    decode_dual_scale_main_block_ref(&compact, &mut decoded).unwrap();

    let mut expected = 0.0f32;
    for i in 0..DUAL_SCALE_BLOCK_SIZE {
        expected += decoded[i] * activation[i];
    }

    assert_eq!(got, expected);
}

#[test]
fn dual_scale_main_row_dot_matches_two_block_main_decode_dense_dot_bit_exact() {
    let compact = make_compact_2x512();
    let hot = repack_dual_scale_tensor_cpu_ref(2, DUAL_SCALE_BLOCK_SIZE * 2, &compact).unwrap();

    let activation = activation_512();
    let row_blocks = hot.row_blocks(0).unwrap();

    let got = dot_dual_scale_hot_f32_main_ref(row_blocks, &activation);

    let mut expected = 0.0f32;

    for block_col in 0..2 {
        let mut decoded = vec![0.0f32; DUAL_SCALE_BLOCK_SIZE];
        decode_dual_scale_main_block_ref(&compact[block_col], &mut decoded).unwrap();

        let col_start = block_col * DUAL_SCALE_BLOCK_SIZE;

        for i in 0..DUAL_SCALE_BLOCK_SIZE {
            expected += decoded[i] * activation[col_start + i];
        }
    }

    assert_eq!(got, expected);
}

// ── Full matvec tests ─────────────────────────────────────────────────────

#[test]
fn dual_scale_qmatvec_matches_full_compact_decode_dense_reference_close() {
    let rows = 2;
    let cols = DUAL_SCALE_BLOCK_SIZE * 2;

    let compact = make_compact_2x512();
    let hot = repack_dual_scale_tensor_cpu_ref(rows, cols, &compact).unwrap();
    let activation = activation_512();

    let mut got = vec![0.0f32; rows];
    qmatvec_dual_scale_f32_portable(&hot, &activation, &mut got).unwrap();

    let mut expected = vec![0.0f32; rows];

    for row in 0..rows {
        for block_col in 0..2 {
            let block_index = row * 2 + block_col;
            let col_start = block_col * DUAL_SCALE_BLOCK_SIZE;

            let mut decoded = vec![0.0f32; DUAL_SCALE_BLOCK_SIZE];
            compact[block_index].decode_ref(&mut decoded).unwrap();

            for i in 0..DUAL_SCALE_BLOCK_SIZE {
                expected[row] += decoded[i] * activation[col_start + i];
            }
        }
    }

    assert_relative_close(got[0], expected[0], 1e-5);
    assert_relative_close(got[1], expected[1], 1e-5);
}

#[test]
fn dual_scale_qmatvec_rejects_bad_activation_length() {
    let rows = 2;
    let cols = DUAL_SCALE_BLOCK_SIZE * 2;

    let compact = make_compact_2x512();
    let hot = repack_dual_scale_tensor_cpu_ref(rows, cols, &compact).unwrap();

    let activation = vec![0.0f32; cols - 1];
    let mut out = vec![0.0f32; rows];

    assert_eq!(
        qmatvec_dual_scale_f32_portable(&hot, &activation, &mut out),
        Err(KernelError::BadActivationLength {
            expected: cols,
            actual: cols - 1,
        })
    );
}

#[test]
fn dual_scale_qmatvec_rejects_short_output() {
    let rows = 2;
    let cols = DUAL_SCALE_BLOCK_SIZE * 2;

    let compact = make_compact_2x512();
    let hot = repack_dual_scale_tensor_cpu_ref(rows, cols, &compact).unwrap();

    let activation = activation_512();
    let mut out = vec![0.0f32; rows - 1];

    assert_eq!(
        qmatvec_dual_scale_f32_portable(&hot, &activation, &mut out),
        Err(KernelError::BadOutputLength {
            expected: rows,
            actual: rows - 1,
        })
    );
}

#[test]
fn dual_scale_qmatvec_error_leaves_output_unchanged() {
    let rows = 2;
    let cols = DUAL_SCALE_BLOCK_SIZE * 2;

    let compact = make_compact_2x512();
    let hot = repack_dual_scale_tensor_cpu_ref(rows, cols, &compact).unwrap();

    let activation = vec![0.0f32; cols - 1];
    let mut out = vec![123.0f32, 456.0];

    assert_eq!(
        qmatvec_dual_scale_f32_portable(&hot, &activation, &mut out),
        Err(KernelError::BadActivationLength {
            expected: cols,
            actual: cols - 1,
        })
    );

    assert_eq!(out, vec![123.0f32, 456.0]);
}

#[test]
fn dual_scale_qmatvec_no_outlier_matches_public_row_dot_bit_exact() {
    let rows = 2;
    let cols = DUAL_SCALE_BLOCK_SIZE * 2;

    let sources = [
        make_source_block_no_outlier(0),
        make_source_block_no_outlier(1),
        make_source_block_no_outlier(2),
        make_source_block_no_outlier(3),
    ];

    let compact: Vec<_> = sources
        .iter()
        .map(|src| encode_dual_scale_block_ref(src).unwrap())
        .collect();

    for block in &compact {
        assert_eq!(block.outlier_count, 0);
    }

    let hot = repack_dual_scale_tensor_cpu_ref(rows, cols, &compact).unwrap();
    assert_eq!(hot.outliers().row_ptr(), vec![0, 0, 0]);
    assert!(hot.outliers().col_idx().is_empty());
    assert!(hot.outliers().delta().is_empty());

    let activation = activation_512();

    let mut got = vec![0.0f32; rows];
    qmatvec_dual_scale_f32_portable(&hot, &activation, &mut got).unwrap();

    let row0 = dot_dual_scale_hot_f32_main_ref(hot.row_blocks(0).unwrap(), &activation);
    let row1 = dot_dual_scale_hot_f32_main_ref(hot.row_blocks(1).unwrap(), &activation);

    assert_eq!(got[0], row0);
    assert_eq!(got[1], row1);
}

// ── M18 grouped dot tests ────────────────────────────────────────────────

#[test]
fn dual_scale_grouped_dot_matches_elementwise_main_dot_close_one_block() {
    let src = make_source_block(11, 23, 48.0);
    let compact = encode_dual_scale_block_ref(&src).unwrap();
    let hot = repack_dual_scale_block_cpu_ref(&compact);

    let activation: Vec<f32> = (0..DUAL_SCALE_BLOCK_SIZE)
        .map(|i| ((i % 31) as f32 - 15.0) * 0.03125)
        .collect();

    let expected = dot_dual_scale_hot_f32_main_ref(std::slice::from_ref(&hot), &activation);
    let got = dot_dual_scale_hot_f32_grouped_ref(std::slice::from_ref(&hot), &activation);

    assert_close(got, expected, 1e-5, 1e-5);
}

#[test]
fn dual_scale_grouped_dot_matches_elementwise_main_dot_close_two_block_row() {
    let compact = make_compact_2x512();
    let hot = repack_dual_scale_tensor_cpu_ref(2, DUAL_SCALE_BLOCK_SIZE * 2, &compact).unwrap();

    let activation = activation_512();
    let row_blocks = hot.row_blocks(0).unwrap();

    let expected = dot_dual_scale_hot_f32_main_ref(row_blocks, &activation);
    let got = dot_dual_scale_hot_f32_grouped_ref(row_blocks, &activation);

    assert_close(got, expected, 1e-5, 1e-5);
}

#[test]
fn dual_scale_grouped_dot_zero_scales_return_canonical_zero() {
    // qs = 0x00 decodes to q = -8 for every lane. With zero scales, the
    // grouped path must skip scale application and return canonical +0.0
    // instead of producing signed zero through 0.0 * negative_sum.
    //
    // 0xaa is 10101010b. With LSB-first membership bits, this alternates
    // lo/hi lanes and exercises both zero-scale guards.
    let block = DualScaleHotCpuBlock {
        scale_lo: [F16Bits::ZERO; 8],
        scale_hi: [F16Bits::ZERO; 8],
        membership: [0xaau8; 32],
        qs: [0x00u8; 128],
    };

    let activation: Vec<f32> = (0..DUAL_SCALE_BLOCK_SIZE)
        .map(|i| (i as f32 + 1.0) * 0.125)
        .collect();

    let got = dot_dual_scale_hot_f32_grouped_ref(std::slice::from_ref(&block), &activation);

    assert_eq!(got, 0.0);
    assert!(!got.is_sign_negative());
}

// ── M19 DualScale × Q8K grouped integer tests ────────────────────────────

#[test]
fn dual_scale_q8k_block_grouped_dot_matches_decoded_q8k_grouped_f32_close() {
    let src = make_source_block(13, 19, 48.0);
    let compact = encode_dual_scale_block_ref(&src).unwrap();
    let hot = repack_dual_scale_block_cpu_ref(&compact);

    let activation_src: Vec<f32> = (0..DUAL_SCALE_BLOCK_SIZE)
        .map(|i| ((i % 31) as f32 - 15.0) * 0.03125)
        .collect();

    let q8 = quantize_q8k_activation_vector_ref(&activation_src).unwrap();

    let mut decoded_activation = vec![0.0f32; DUAL_SCALE_BLOCK_SIZE];
    q8.decode_ref(&mut decoded_activation).unwrap();

    let expected =
        dot_dual_scale_hot_f32_grouped_ref(std::slice::from_ref(&hot), &decoded_activation);

    let got = dot_dual_scale_block_q8k_grouped_ref(&hot, &q8.blocks()[0]);

    assert_close(got, expected, 1e-4, 1e-4);
}

#[test]
fn dual_scale_q8k_row_grouped_dot_matches_decoded_q8k_grouped_f32_close() {
    let rows = 2;
    let cols = DUAL_SCALE_BLOCK_SIZE * 2;

    let compact = make_compact_2x512();
    let hot = repack_dual_scale_tensor_cpu_ref(rows, cols, &compact).unwrap();

    let activation_src = activation_512();
    let q8 = quantize_q8k_activation_vector_ref(&activation_src).unwrap();

    let mut decoded_activation = vec![0.0f32; cols];
    q8.decode_ref(&mut decoded_activation).unwrap();

    let row_blocks = hot.row_blocks(0).unwrap();

    let expected = dot_dual_scale_hot_f32_grouped_ref(row_blocks, &decoded_activation);
    let got = dot_dual_scale_hot_q8k_grouped_ref(row_blocks, &q8);

    assert_close(got, expected, 1e-4, 1e-4);
}

#[test]
fn dual_scale_q8k_grouped_dot_zero_activation_returns_canonical_zero() {
    let src = make_source_block_no_outlier(5);
    let compact = encode_dual_scale_block_ref(&src).unwrap();
    let hot = repack_dual_scale_block_cpu_ref(&compact);

    let q8 = Q8KActivationBlock {
        d: F16Bits::ZERO,
        qs: [-127i8; Q8K_BLOCK_SIZE],
    };

    let got = dot_dual_scale_block_q8k_grouped_ref(&hot, &q8);

    assert_eq!(got, 0.0);
    assert!(!got.is_sign_negative());
}

#[test]
fn dual_scale_q8k_qmatvec_matches_full_decode_dense_reference_close() {
    let rows = 2;
    let cols = DUAL_SCALE_BLOCK_SIZE * 2;

    let compact = make_compact_2x512();
    let hot = repack_dual_scale_tensor_cpu_ref(rows, cols, &compact).unwrap();

    let activation_src = activation_512();
    let q8 = quantize_q8k_activation_vector_ref(&activation_src).unwrap();

    let mut got = vec![0.0f32; rows];
    qmatvec_dual_scale_q8k_ref(&hot, &q8, &mut got).unwrap();

    let mut decoded_activation = vec![0.0f32; cols];
    q8.decode_ref(&mut decoded_activation).unwrap();

    let mut expected = vec![0.0f32; rows];

    for row in 0..rows {
        for block_col in 0..2 {
            let block_index = row * 2 + block_col;
            let col_start = block_col * DUAL_SCALE_BLOCK_SIZE;

            let mut decoded_weight = vec![0.0f32; DUAL_SCALE_BLOCK_SIZE];
            compact[block_index]
                .decode_ref(&mut decoded_weight)
                .unwrap();

            for i in 0..DUAL_SCALE_BLOCK_SIZE {
                expected[row] += decoded_weight[i] * decoded_activation[col_start + i];
            }
        }
    }

    assert_close(got[0], expected[0], 1e-3, 1e-3);
    assert_close(got[1], expected[1], 1e-3, 1e-3);
}

#[test]
fn dual_scale_q8k_qmatvec_rejects_bad_activation_length() {
    let rows = 2;
    let cols = DUAL_SCALE_BLOCK_SIZE * 2;

    let compact = make_compact_2x512();
    let hot = repack_dual_scale_tensor_cpu_ref(rows, cols, &compact).unwrap();

    let activation_src = vec![0.0f32; DUAL_SCALE_BLOCK_SIZE];
    let q8 = quantize_q8k_activation_vector_ref(&activation_src).unwrap();

    let mut out = vec![0.0f32; rows];

    assert_eq!(
        qmatvec_dual_scale_q8k_ref(&hot, &q8, &mut out),
        Err(ReferenceQ8Error::BadActivationLength {
            expected: cols,
            actual: DUAL_SCALE_BLOCK_SIZE,
        })
    );
}

#[test]
fn dual_scale_q8k_qmatvec_rejects_short_output() {
    let rows = 2;
    let cols = DUAL_SCALE_BLOCK_SIZE * 2;

    let compact = make_compact_2x512();
    let hot = repack_dual_scale_tensor_cpu_ref(rows, cols, &compact).unwrap();

    let activation_src = activation_512();
    let q8 = quantize_q8k_activation_vector_ref(&activation_src).unwrap();

    let mut out = vec![0.0f32; rows - 1];

    assert_eq!(
        qmatvec_dual_scale_q8k_ref(&hot, &q8, &mut out),
        Err(ReferenceQ8Error::BadOutputLength {
            expected: rows,
            actual: rows - 1,
        })
    );
}

#[test]
fn dual_scale_q8k_qmatvec_error_leaves_output_unchanged() {
    let rows = 2;
    let cols = DUAL_SCALE_BLOCK_SIZE * 2;

    let compact = make_compact_2x512();
    let hot = repack_dual_scale_tensor_cpu_ref(rows, cols, &compact).unwrap();

    let activation_src = vec![0.0f32; DUAL_SCALE_BLOCK_SIZE];
    let q8 = quantize_q8k_activation_vector_ref(&activation_src).unwrap();

    let mut out = vec![123.0f32, 456.0];

    assert_eq!(
        qmatvec_dual_scale_q8k_ref(&hot, &q8, &mut out),
        Err(ReferenceQ8Error::BadActivationLength {
            expected: cols,
            actual: DUAL_SCALE_BLOCK_SIZE,
        })
    );

    assert_eq!(out, vec![123.0f32, 456.0]);
}

// ── M21 DualScale × Q8KHot CPU portable tests ────────────────────────────

#[test]
fn dual_scale_q8k_hot_block_grouped_dot_matches_reference_q8k_bit_exact() {
    let src = make_source_block(17, 23, 48.0);
    let compact = encode_dual_scale_block_ref(&src).unwrap();
    let hot_weight = repack_dual_scale_block_cpu_ref(&compact);

    let activation_src: Vec<f32> = (0..Q8K_BLOCK_SIZE)
        .map(|i| ((i % 31) as f32 - 15.0) * 0.03125)
        .collect();

    let q8 = quantize_q8k_activation_ref(&activation_src).unwrap();
    let q8_hot = repack_q8k_activation_block_cpu_ref(&q8);

    let expected = dot_dual_scale_block_q8k_grouped_ref(&hot_weight, &q8);
    let got = dot_dual_scale_block_q8k_hot_cpu_grouped_ref(&hot_weight, &q8_hot);

    assert_eq!(got, expected);
}

#[test]
fn dual_scale_q8k_hot_row_grouped_dot_matches_reference_q8k_bit_exact() {
    let rows = 2;
    let cols = DUAL_SCALE_BLOCK_SIZE * 2;

    let compact = make_compact_2x512();
    let hot_weight = repack_dual_scale_tensor_cpu_ref(rows, cols, &compact).unwrap();

    let activation = activation_512();
    let q8 = quantize_q8k_activation_vector_ref(&activation).unwrap();
    let q8_hot = repack_q8k_activation_vector_cpu_ref(&q8).unwrap();

    let row_blocks = hot_weight.row_blocks(0).unwrap();

    let expected = dot_dual_scale_hot_q8k_grouped_ref(row_blocks, &q8);
    let got = dot_dual_scale_hot_q8k_hot_cpu_grouped_ref(row_blocks, &q8_hot);

    assert_eq!(got, expected);
}

#[test]
fn dual_scale_q8k_hot_qmatvec_matches_reference_q8k_path_bit_exact() {
    let rows = 2;
    let cols = DUAL_SCALE_BLOCK_SIZE * 2;

    let compact = make_compact_2x512();
    let hot_weight = repack_dual_scale_tensor_cpu_ref(rows, cols, &compact).unwrap();

    let activation = activation_512();
    let q8 = quantize_q8k_activation_vector_ref(&activation).unwrap();
    let q8_hot = repack_q8k_activation_vector_cpu_ref(&q8).unwrap();

    let mut expected = vec![0.0f32; rows];
    qmatvec_dual_scale_q8k_ref(&hot_weight, &q8, &mut expected).unwrap();

    let mut got = vec![0.0f32; rows];
    qmatvec_dual_scale_q8k_hot_cpu_portable(&hot_weight, &q8_hot, &mut got).unwrap();

    // BitExact because the hot activation block is a bit-exact repack of the
    // reference Q8K block and the qmatvec paths use the same accumulation
    // order.
    assert_eq!(got, expected);
}

#[test]
fn dual_scale_q8k_hot_qmatvec_matches_full_decode_dense_reference_close() {
    let rows = 2;
    let cols = DUAL_SCALE_BLOCK_SIZE * 2;

    let compact = make_compact_2x512();
    let hot_weight = repack_dual_scale_tensor_cpu_ref(rows, cols, &compact).unwrap();

    let activation = activation_512();
    let q8 = quantize_q8k_activation_vector_ref(&activation).unwrap();
    let q8_hot = repack_q8k_activation_vector_cpu_ref(&q8).unwrap();

    let mut got = vec![0.0f32; rows];
    qmatvec_dual_scale_q8k_hot_cpu_portable(&hot_weight, &q8_hot, &mut got).unwrap();

    let mut decoded_activation = vec![0.0f32; cols];
    q8.decode_ref(&mut decoded_activation).unwrap();

    let mut expected = vec![0.0f32; rows];

    for row in 0..rows {
        for block_col in 0..2 {
            let block_index = row * 2 + block_col;
            let col_start = block_col * DUAL_SCALE_BLOCK_SIZE;

            let mut decoded_weight = vec![0.0f32; DUAL_SCALE_BLOCK_SIZE];
            compact[block_index]
                .decode_ref(&mut decoded_weight)
                .unwrap();

            for i in 0..DUAL_SCALE_BLOCK_SIZE {
                expected[row] += decoded_weight[i] * decoded_activation[col_start + i];
            }
        }
    }

    assert_close(got[0], expected[0], 1e-3, 1e-3);
    assert_close(got[1], expected[1], 1e-3, 1e-3);
}

#[test]
fn dual_scale_q8k_hot_qmatvec_rejects_bad_activation_length() {
    let rows = 2;
    let cols = DUAL_SCALE_BLOCK_SIZE * 2;

    let compact = make_compact_2x512();
    let hot_weight = repack_dual_scale_tensor_cpu_ref(rows, cols, &compact).unwrap();

    let activation = vec![0.0f32; DUAL_SCALE_BLOCK_SIZE];
    let q8 = quantize_q8k_activation_vector_ref(&activation).unwrap();
    let q8_hot = repack_q8k_activation_vector_cpu_ref(&q8).unwrap();

    let mut out = vec![0.0f32; rows];

    assert_eq!(
        qmatvec_dual_scale_q8k_hot_cpu_portable(&hot_weight, &q8_hot, &mut out),
        Err(KernelError::BadActivationLength {
            expected: cols,
            actual: DUAL_SCALE_BLOCK_SIZE,
        })
    );
}

#[test]
fn dual_scale_q8k_hot_qmatvec_rejects_short_output() {
    let rows = 2;
    let cols = DUAL_SCALE_BLOCK_SIZE * 2;

    let compact = make_compact_2x512();
    let hot_weight = repack_dual_scale_tensor_cpu_ref(rows, cols, &compact).unwrap();

    let activation = activation_512();
    let q8 = quantize_q8k_activation_vector_ref(&activation).unwrap();
    let q8_hot = repack_q8k_activation_vector_cpu_ref(&q8).unwrap();

    let mut out = vec![0.0f32; rows - 1];

    assert_eq!(
        qmatvec_dual_scale_q8k_hot_cpu_portable(&hot_weight, &q8_hot, &mut out),
        Err(KernelError::BadOutputLength {
            expected: rows,
            actual: rows - 1,
        })
    );
}

#[test]
fn dual_scale_q8k_hot_qmatvec_error_leaves_output_unchanged() {
    let rows = 2;
    let cols = DUAL_SCALE_BLOCK_SIZE * 2;

    let compact = make_compact_2x512();
    let hot_weight = repack_dual_scale_tensor_cpu_ref(rows, cols, &compact).unwrap();

    let activation = vec![0.0f32; DUAL_SCALE_BLOCK_SIZE];
    let q8 = quantize_q8k_activation_vector_ref(&activation).unwrap();
    let q8_hot = repack_q8k_activation_vector_cpu_ref(&q8).unwrap();

    let mut out = vec![123.0f32, 456.0];

    assert_eq!(
        qmatvec_dual_scale_q8k_hot_cpu_portable(&hot_weight, &q8_hot, &mut out),
        Err(KernelError::BadActivationLength {
            expected: cols,
            actual: DUAL_SCALE_BLOCK_SIZE,
        })
    );

    assert_eq!(out, vec![123.0f32, 456.0]);
}

// ── M22 Q8KHot indexed CSR correction tests ─────────────────────────────

#[test]
fn dual_scale_q8k_hot_qmatvec_with_outliers_matches_reference_q8k_path_bit_exact() {
    let rows = 2;
    let cols = DUAL_SCALE_BLOCK_SIZE * 2;

    let compact = make_compact_2x512();

    let nnz: usize = compact
        .iter()
        .map(|block| block.outlier_count as usize)
        .sum();
    assert!(nnz > 0);

    let hot_weight = repack_dual_scale_tensor_cpu_ref(rows, cols, &compact).unwrap();
    assert_eq!(hot_weight.outliers().delta().len(), nnz);

    let activation = activation_512();
    let q8 = quantize_q8k_activation_vector_ref(&activation).unwrap();
    let q8_hot = repack_q8k_activation_vector_cpu_ref(&q8).unwrap();

    let mut expected = vec![0.0_f32; rows];
    qmatvec_dual_scale_q8k_ref(&hot_weight, &q8, &mut expected).unwrap();

    let mut got = vec![0.0_f32; rows];
    qmatvec_dual_scale_q8k_hot_cpu_portable(&hot_weight, &q8_hot, &mut got).unwrap();

    // BitExact because M22 indexed CSR correction decodes the same Q8K d/qs
    // values in the same CSR iteration order as the M19 decoded-vector path.
    assert_eq!(got, expected);
}

#[test]
fn dual_scale_q8k_hot_qmatvec_zero_activation_scale_zeroes_main_and_csr() {
    let rows = 2;
    let cols = DUAL_SCALE_BLOCK_SIZE * 2;

    let compact = make_compact_2x512();

    let nnz: usize = compact
        .iter()
        .map(|block| block.outlier_count as usize)
        .sum();
    assert!(nnz > 0);

    let hot_weight = repack_dual_scale_tensor_cpu_ref(rows, cols, &compact).unwrap();

    let zero_q8_blocks = (0..hot_weight.blocks_per_row())
        .map(|_| Q8KActivationHotCpuBlock {
            d: F16Bits::ZERO,
            qs: [-127_i8; Q8K_BLOCK_SIZE],
        })
        .collect::<Vec<_>>();

    let q8_hot = Q8KActivationHotCpuVector::new(cols, zero_q8_blocks).unwrap();

    let mut got = vec![99.0_f32; rows];
    qmatvec_dual_scale_q8k_hot_cpu_portable(&hot_weight, &q8_hot, &mut got).unwrap();

    assert_eq!(got, vec![0.0_f32; rows]);
    assert!(!got[0].is_sign_negative());
    assert!(!got[1].is_sign_negative());
}

// ── M26-D accumulator extreme value tests ───────────────────────────────────

#[test]
fn dual_scale_q8hot_i32_lo_accumulator_bound_is_exact() {
    let weight = DualScaleHotCpuBlock {
        scale_lo: [F16Bits::from_f32(1.0); 8],
        scale_hi: [F16Bits::ZERO; 8],
        membership: [0x00; 32], // every lane → lo_acc
        qs: [0x00; 128],        // every nibble decodes to q4 = -8
    };
    let activation = Q8KActivationHotCpuBlock {
        d: F16Bits::from_f32(1.0),
        qs: [-128; Q8K_BLOCK_SIZE], // complete i8 domain
    };
    let actual = dot_dual_scale_block_q8k_hot_cpu_grouped_ref(&weight, &activation);
    // Per group: 32 × (-8) × (-128) = 32,768
    // Eight groups: 8 × 32,768 = 262,144
    assert_eq!(actual, 262_144.0);
}

#[test]
fn dual_scale_q8hot_i32_hi_accumulator_bound_is_exact() {
    let weight = DualScaleHotCpuBlock {
        scale_lo: [F16Bits::ZERO; 8],
        scale_hi: [F16Bits::from_f32(1.0); 8],
        membership: [0xff; 32], // every lane → hi_acc
        qs: [0x00; 128],        // every nibble decodes to q4 = -8
    };
    let activation = Q8KActivationHotCpuBlock {
        d: F16Bits::from_f32(1.0),
        qs: [-128; Q8K_BLOCK_SIZE],
    };
    let actual = dot_dual_scale_block_q8k_hot_cpu_grouped_ref(&weight, &activation);
    assert_eq!(actual, 262_144.0);
}

#[test]
fn dual_scale_q8hot_typed_launch_matches_direct_kernel() {
    let rows = 2;
    let cols = DUAL_SCALE_BLOCK_SIZE * 2;
    let compact = make_compact_2x512();
    let weight = repack_dual_scale_tensor_cpu_ref(rows, cols, &compact).unwrap();
    let q8 = quantize_q8k_activation_vector_ref(&activation_512()).unwrap();
    let q8_hot = repack_q8k_activation_vector_cpu_ref(&q8).unwrap();

    let launch: DualScaleQ8HotCpuLaunch = launch_qmatvec_dual_scale_q8k_hot_cpu_portable;
    let mut direct = vec![0.0_f32; rows];
    qmatvec_dual_scale_q8k_hot_cpu_portable(&weight, &q8_hot, &mut direct).unwrap();

    let mut launched = vec![0.0_f32; rows];
    launch(&weight, &q8_hot, &mut launched).unwrap();
    assert_eq!(launched, direct);
}

#[test]
fn dual_scale_q8hot_overwrites_rows_and_preserves_tail() {
    let rows = 2;
    let cols = DUAL_SCALE_BLOCK_SIZE * 2;
    let compact = make_compact_2x512();
    let weight = repack_dual_scale_tensor_cpu_ref(rows, cols, &compact).unwrap();
    let q8 = quantize_q8k_activation_vector_ref(&activation_512()).unwrap();
    let q8_hot = repack_q8k_activation_vector_cpu_ref(&q8).unwrap();

    let mut expected = vec![0.0_f32; rows];
    qmatvec_dual_scale_q8k_hot_cpu_portable(&weight, &q8_hot, &mut expected).unwrap();

    let mut output = vec![91.0_f32, 92.0, 93.0, 94.0];
    qmatvec_dual_scale_q8k_hot_cpu_portable(&weight, &q8_hot, &mut output).unwrap();

    assert_eq!(&output[..rows], expected.as_slice());
    assert_eq!(&output[rows..], &[93.0, 94.0]);
}

#[test]
fn dual_scale_q8hot_csr_last_lane_of_last_block_is_safe() {
    let rows = 1;
    let cols = DUAL_SCALE_BLOCK_SIZE * 2;

    let outliers = OutlierCsrHotCpu::new(
        rows,
        cols,
        vec![0, 1],
        vec![u32::try_from(cols - 1).unwrap()],
        vec![F16Bits::from_f32(1.0)],
    )
    .unwrap();

    let zero_weight = DualScaleHotCpuBlock {
        scale_lo: [F16Bits::ZERO; 8],
        scale_hi: [F16Bits::ZERO; 8],
        membership: [0; 32],
        qs: [0; 128],
    };

    let weight =
        DualScaleHotCpuTensor::new(rows, cols, vec![zero_weight.clone(), zero_weight], outliers)
            .unwrap();

    let mut activation_blocks = vec![Q8KActivationHotCpuBlock::ZERO; 2];
    activation_blocks[1].d = F16Bits::from_f32(1.0);
    activation_blocks[1].qs[255] = 7;

    let activation = Q8KActivationHotCpuVector::new(cols, activation_blocks).unwrap();

    let mut output = vec![0.0_f32; rows];
    qmatvec_dual_scale_q8k_hot_cpu_portable(&weight, &activation, &mut output).unwrap();

    // Main path is zero.
    // CSR correction = delta 1 × last activation lane 7.
    assert_eq!(output, vec![7.0]);
}
