use augur_core::F16Bits;
use augur_formats::dual_scale::decode_dual_scale_main_block_ref;
use augur_formats::{
    add_outlier_csr_correction_ref, encode_dual_scale_block_ref,
    extract_dual_scale_outlier_csr_ref, FormatError, OutlierCsrRef, ReferenceDecode,
    DUAL_SCALE_BLOCK_SIZE,
};

fn assert_relative_close(a: f32, b: f32, rel_tol: f32) {
    let diff = (a - b).abs();
    let denom = a.abs().max(b.abs()).max(1.0);
    let rel = diff / denom;

    assert!(
        rel <= rel_tol,
        "a={a} b={b} diff={diff} rel={rel} tol={rel_tol}"
    );
}

fn make_block_with_outlier(local_idx: usize, outlier: f32) -> augur_formats::DualScaleCompactBlock {
    let mut src = vec![0.25f32; DUAL_SCALE_BLOCK_SIZE];
    src[local_idx] = outlier;
    encode_dual_scale_block_ref(&src).unwrap()
}

// ── Extraction tests ──────────────────────────────────────────────────────

#[test]
fn dual_scale_outlier_csr_extracts_row_major_indices() {
    let rows = 2;
    let cols = DUAL_SCALE_BLOCK_SIZE * 2;

    let blocks = vec![
        make_block_with_outlier(3, 40.0),
        make_block_with_outlier(5, -50.0),
        make_block_with_outlier(7, 60.0),
        make_block_with_outlier(11, -70.0),
    ];

    let csr = extract_dual_scale_outlier_csr_ref(rows, cols, &blocks).unwrap();

    assert_eq!(csr.rows, rows);
    assert_eq!(csr.cols, cols);
    assert_eq!(csr.row_ptr, vec![0, 2, 4]);
    assert_eq!(
        csr.col_idx,
        vec![
            3,
            DUAL_SCALE_BLOCK_SIZE as u32 + 5,
            7,
            DUAL_SCALE_BLOCK_SIZE as u32 + 11,
        ]
    );
    assert_eq!(csr.delta.len(), 4);
}

#[test]
fn dual_scale_outlier_csr_rejects_bad_cols_shape() {
    let blocks = vec![make_block_with_outlier(3, 40.0)];

    assert_eq!(
        extract_dual_scale_outlier_csr_ref(1, DUAL_SCALE_BLOCK_SIZE + 1, &blocks),
        Err(FormatError::BadLength)
    );
}

#[test]
fn dual_scale_outlier_csr_rejects_block_count_mismatch() {
    let rows = 2;
    let cols = DUAL_SCALE_BLOCK_SIZE * 2;
    let blocks = vec![make_block_with_outlier(3, 40.0)];

    assert_eq!(
        extract_dual_scale_outlier_csr_ref(rows, cols, &blocks),
        Err(FormatError::InputLengthMismatch {
            expected: 4,
            actual: 1,
        })
    );
}

#[test]
fn dual_scale_outlier_csr_rejects_duplicate_outlier_idx() {
    let mut block = make_block_with_outlier(3, 40.0);
    block.outlier_count = 2;
    block.outlier_idx = [3, 3];

    assert_eq!(
        extract_dual_scale_outlier_csr_ref(1, DUAL_SCALE_BLOCK_SIZE, &[block]),
        Err(FormatError::BadBlock)
    );
}

#[test]
fn dual_scale_outlier_csr_rejects_non_finite_delta() {
    let mut block = make_block_with_outlier(3, 40.0);
    block.outlier_delta[0] = F16Bits::from_bits(0x7c00); // +inf

    assert_eq!(
        extract_dual_scale_outlier_csr_ref(1, DUAL_SCALE_BLOCK_SIZE, &[block]),
        Err(FormatError::BadBlock)
    );
}

// ── Correction tests ─────────────────────────────────────────────────────

#[test]
fn outlier_csr_correction_rejects_bad_x_length() {
    let csr = OutlierCsrRef {
        rows: 1,
        cols: 4,
        row_ptr: vec![0, 1],
        col_idx: vec![2],
        delta: vec![F16Bits::from_f32(3.0)],
    };

    let x = vec![1.0f32; 3];
    let mut out = vec![0.0f32; 1];

    assert_eq!(
        add_outlier_csr_correction_ref(&csr, &x, &mut out),
        Err(FormatError::InputLengthMismatch {
            expected: 4,
            actual: 3,
        })
    );
}

#[test]
fn outlier_csr_correction_rejects_short_output() {
    let csr = OutlierCsrRef {
        rows: 2,
        cols: 4,
        row_ptr: vec![0, 0, 0],
        col_idx: vec![],
        delta: vec![],
    };

    let x = vec![1.0f32; 4];
    let mut out = vec![0.0f32; 1];

    assert_eq!(
        add_outlier_csr_correction_ref(&csr, &x, &mut out),
        Err(FormatError::OutputTooSmall {
            expected: 2,
            actual: 1,
        })
    );
}

#[test]
fn outlier_csr_correction_rejects_invalid_row_ptr() {
    let csr = OutlierCsrRef {
        rows: 2,
        cols: 4,
        row_ptr: vec![0, 2, 1],
        col_idx: vec![0],
        delta: vec![F16Bits::from_f32(1.0)],
    };

    let x = vec![1.0f32; 4];
    let mut out = vec![0.0f32; 2];

    assert_eq!(
        add_outlier_csr_correction_ref(&csr, &x, &mut out),
        Err(FormatError::BadBlock)
    );
}

#[test]
fn outlier_csr_correction_rejects_col_out_of_range() {
    let csr = OutlierCsrRef {
        rows: 1,
        cols: 4,
        row_ptr: vec![0, 1],
        col_idx: vec![4],
        delta: vec![F16Bits::from_f32(1.0)],
    };

    let x = vec![1.0f32; 4];
    let mut out = vec![0.0f32; 1];

    assert_eq!(
        add_outlier_csr_correction_ref(&csr, &x, &mut out),
        Err(FormatError::BadBlock)
    );
}

#[test]
fn outlier_csr_correction_error_leaves_output_unchanged() {
    let csr = OutlierCsrRef {
        rows: 1,
        cols: 4,
        row_ptr: vec![0, 1],
        col_idx: vec![4],
        delta: vec![F16Bits::from_f32(1.0)],
    };

    let x = vec![1.0f32; 4];
    let mut out = vec![123.0f32];

    assert_eq!(
        add_outlier_csr_correction_ref(&csr, &x, &mut out),
        Err(FormatError::BadBlock)
    );

    assert_eq!(out, vec![123.0f32]);
}

#[test]
fn outlier_csr_correction_accumulates_into_existing_output() {
    let csr = OutlierCsrRef {
        rows: 2,
        cols: 4,
        row_ptr: vec![0, 1, 2],
        col_idx: vec![1, 3],
        delta: vec![F16Bits::from_f32(2.0), F16Bits::from_f32(-3.0)],
    };

    let x = vec![10.0f32, 4.0, 5.0, 6.0];
    let mut out = vec![100.0f32, 200.0];

    add_outlier_csr_correction_ref(&csr, &x, &mut out).unwrap();

    assert_eq!(out[0], 100.0 + 2.0 * 4.0);
    assert_eq!(out[1], 200.0 + -3.0 * 6.0);
}

#[test]
fn dual_scale_dense_matvec_matches_main_matvec_plus_csr_correction() {
    let rows = 2;
    let cols = DUAL_SCALE_BLOCK_SIZE * 2;

    let outlier_positions = [5usize, 37, 91, 123];
    let outlier_values = [48.0f32, -56.0, 64.0, -72.0];

    let mut dense_blocks = Vec::new();

    for block_id in 0..4 {
        let mut src: Vec<f32> = (0..DUAL_SCALE_BLOCK_SIZE)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.03125)
            .collect();

        src[outlier_positions[block_id]] = outlier_values[block_id];
        dense_blocks.push(src);
    }

    let blocks: Vec<_> = dense_blocks
        .iter()
        .map(|src| encode_dual_scale_block_ref(src).unwrap())
        .collect();

    let x: Vec<f32> = (0..cols)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.0625)
        .collect();

    let mut full = vec![0.0f32; rows];
    let mut main = vec![0.0f32; rows];

    for row in 0..rows {
        for block_col in 0..2 {
            let block_index = row * 2 + block_col;
            let col_start = block_col * DUAL_SCALE_BLOCK_SIZE;

            let mut full_decoded = vec![0.0f32; DUAL_SCALE_BLOCK_SIZE];
            blocks[block_index].decode_ref(&mut full_decoded).unwrap();

            let mut main_decoded = vec![0.0f32; DUAL_SCALE_BLOCK_SIZE];
            decode_dual_scale_main_block_ref(&blocks[block_index], &mut main_decoded).unwrap();

            for i in 0..DUAL_SCALE_BLOCK_SIZE {
                full[row] += full_decoded[i] * x[col_start + i];
                main[row] += main_decoded[i] * x[col_start + i];
            }
        }
    }

    let csr = extract_dual_scale_outlier_csr_ref(rows, cols, &blocks).unwrap();
    add_outlier_csr_correction_ref(&csr, &x, &mut main).unwrap();

    assert_relative_close(main[0], full[0], 1e-5);
    assert_relative_close(main[1], full[1], 1e-5);
}
