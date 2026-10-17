#![allow(clippy::needless_range_loop)]

use augur_core::F16Bits;
use augur_formats::{
    encode_dual_scale_block_ref, extract_dual_scale_outlier_csr_ref, FormatError, OutlierCsrRef,
    DUAL_SCALE_BLOCK_SIZE,
};
use augur_kernel_api::{DualScaleHotCpuTensor, OutlierCsrHotCpu};
use augur_repack::{
    repack_dual_scale_block_cpu_ref, repack_dual_scale_tensor_cpu_ref, RepackError, TensorError,
};

fn make_block(
    seed: usize,
    outlier_idx: usize,
    outlier_value: f32,
) -> augur_formats::DualScaleCompactBlock {
    let mut src: Vec<f32> = (0..DUAL_SCALE_BLOCK_SIZE)
        .map(|i| (((i + seed) % 31) as f32 - 15.0) * 0.03125)
        .collect();
    src[outlier_idx] = outlier_value;
    encode_dual_scale_block_ref(&src).unwrap()
}

fn hot_outliers(src: OutlierCsrRef) -> OutlierCsrHotCpu {
    OutlierCsrHotCpu::new(src.rows, src.cols, src.row_ptr, src.col_idx, src.delta).unwrap()
}

#[test]
fn dual_scale_block_repack_preserves_main_fields_bit_exact() {
    let compact = make_block(3, 17, 48.0);
    let hot = repack_dual_scale_block_cpu_ref(&compact);
    assert_eq!(hot.scale_lo, compact.scale_lo);
    assert_eq!(hot.scale_hi, compact.scale_hi);
    assert_eq!(hot.membership, compact.membership);
    assert_eq!(hot.qs, compact.qs);
}

#[test]
fn dual_scale_tensor_repack_preserves_shape_and_blocks() {
    let rows = 2;
    let cols = DUAL_SCALE_BLOCK_SIZE * 2;
    let compact = vec![
        make_block(0, 5, 48.0),
        make_block(1, 37, -56.0),
        make_block(2, 91, 64.0),
        make_block(3, 123, -72.0),
    ];
    let hot = repack_dual_scale_tensor_cpu_ref(rows, cols, &compact).unwrap();
    assert_eq!(hot.rows(), rows);
    assert_eq!(hot.cols(), cols);
    assert_eq!(hot.blocks_per_row(), 2);
    assert_eq!(hot.blocks().len(), 4);
    for (hot_block, compact_block) in hot.blocks().iter().zip(compact.iter()) {
        assert_eq!(hot_block.scale_lo, compact_block.scale_lo);
        assert_eq!(hot_block.scale_hi, compact_block.scale_hi);
        assert_eq!(hot_block.membership, compact_block.membership);
        assert_eq!(hot_block.qs, compact_block.qs);
    }
}

#[test]
fn dual_scale_tensor_repack_extracts_outliers_to_csr_side_data() {
    let rows = 2;
    let cols = DUAL_SCALE_BLOCK_SIZE * 2;
    let compact = vec![
        make_block(0, 5, 48.0),
        make_block(1, 37, -56.0),
        make_block(2, 91, 64.0),
        make_block(3, 123, -72.0),
    ];
    let expected = extract_dual_scale_outlier_csr_ref(rows, cols, &compact).unwrap();
    let hot = repack_dual_scale_tensor_cpu_ref(rows, cols, &compact).unwrap();
    let actual = hot.outliers();
    assert_eq!(actual.rows(), expected.rows);
    assert_eq!(actual.cols(), expected.cols);
    assert_eq!(actual.row_ptr(), expected.row_ptr.as_slice());
    assert_eq!(actual.col_idx(), expected.col_idx.as_slice());
    assert_eq!(actual.delta(), expected.delta.as_slice());
}

#[test]
fn dual_scale_hot_tensor_row_blocks_returns_row_slice() {
    let rows = 2;
    let cols = DUAL_SCALE_BLOCK_SIZE * 2;
    let compact = vec![
        make_block(0, 5, 48.0),
        make_block(1, 37, -56.0),
        make_block(2, 91, 64.0),
        make_block(3, 123, -72.0),
    ];
    let hot = repack_dual_scale_tensor_cpu_ref(rows, cols, &compact).unwrap();
    let row0 = hot.row_blocks(0).unwrap();
    let row1 = hot.row_blocks(1).unwrap();
    assert_eq!(row0.len(), 2);
    assert_eq!(row1.len(), 2);
    assert_eq!(row0, &hot.blocks()[0..2]);
    assert_eq!(row1, &hot.blocks()[2..4]);
    assert!(hot.row_blocks(2).is_none());
}

#[test]
fn dual_scale_hot_tensor_new_rejects_bad_cols_shape() {
    let compact = vec![make_block(0, 5, 48.0)];
    let outliers = hot_outliers(
        extract_dual_scale_outlier_csr_ref(1, DUAL_SCALE_BLOCK_SIZE, &compact).unwrap(),
    );
    assert_eq!(
        DualScaleHotCpuTensor::new(
            1,
            DUAL_SCALE_BLOCK_SIZE + 1,
            vec![repack_dual_scale_block_cpu_ref(&compact[0])],
            outliers
        ),
        Err(TensorError::BadShape)
    );
}

#[test]
fn dual_scale_hot_tensor_new_rejects_bad_block_count() {
    let compact = vec![make_block(0, 5, 48.0), make_block(1, 37, -56.0)];
    let outliers = hot_outliers(
        extract_dual_scale_outlier_csr_ref(1, DUAL_SCALE_BLOCK_SIZE * 2, &compact).unwrap(),
    );
    assert_eq!(
        DualScaleHotCpuTensor::new(
            1,
            DUAL_SCALE_BLOCK_SIZE * 2,
            vec![repack_dual_scale_block_cpu_ref(&compact[0])],
            outliers
        ),
        Err(TensorError::BadBlockCount {
            expected: 2,
            actual: 1
        })
    );
}

#[test]
fn dual_scale_hot_tensor_new_rejects_outlier_shape_mismatch() {
    let compact = vec![make_block(0, 5, 48.0)];
    let outliers = hot_outliers(
        extract_dual_scale_outlier_csr_ref(1, DUAL_SCALE_BLOCK_SIZE, &compact).unwrap(),
    );
    assert_eq!(
        DualScaleHotCpuTensor::new(
            2,
            DUAL_SCALE_BLOCK_SIZE,
            vec![
                repack_dual_scale_block_cpu_ref(&compact[0]),
                repack_dual_scale_block_cpu_ref(&compact[0])
            ],
            outliers
        ),
        Err(TensorError::BadShape)
    );
}

#[test]
fn dual_scale_hot_tensor_new_rejects_malformed_outlier_csr() {
    assert_eq!(
        OutlierCsrHotCpu::new(
            1,
            DUAL_SCALE_BLOCK_SIZE,
            vec![0, 2],
            vec![5],
            vec![F16Bits::from_f32(1.0)]
        ),
        Err(augur_kernel_api::HotLayoutError::BadOutlierCsr)
    );
}

#[test]
fn dual_scale_tensor_repack_rejects_compact_shape_errors() {
    let compact = vec![make_block(0, 5, 48.0)];
    assert_eq!(
        repack_dual_scale_tensor_cpu_ref(1, DUAL_SCALE_BLOCK_SIZE * 2, &compact),
        Err(RepackError::Format(FormatError::InputLengthMismatch {
            expected: 2,
            actual: 1
        }))
    );
}
