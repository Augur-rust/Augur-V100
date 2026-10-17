use crate::{RepackError, TensorError};
use augur_formats::dual_scale::{DUAL_SCALE_MEMBERSHIP_BYTES, DUAL_SCALE_QS_BYTES};
use augur_formats::{
    extract_dual_scale_outlier_csr_ref, DualScaleCompactBlock, DUAL_SCALE_BLOCK_SIZE,
    DUAL_SCALE_GROUPS,
};
use augur_kernel_api::{
    DualScaleHotCpuBlock, DualScaleHotCpuTensor, OutlierCsrHotCpu, DUAL_SCALE_HOT_CPU_BLOCK_LEN,
    DUAL_SCALE_HOT_CPU_GROUPS, DUAL_SCALE_HOT_CPU_MEMBERSHIP_BYTES, DUAL_SCALE_HOT_CPU_QS_BYTES,
};

fn repack_outlier_csr_cpu(src: augur_formats::OutlierCsrRef) -> Result<OutlierCsrHotCpu, TensorError> {
    OutlierCsrHotCpu::new(src.rows, src.cols, src.row_ptr, src.col_idx, src.delta)
}

pub fn repack_dual_scale_block_cpu_ref(src: &DualScaleCompactBlock) -> DualScaleHotCpuBlock {
    debug_assert_eq!(
        DUAL_SCALE_MEMBERSHIP_BYTES,
        DUAL_SCALE_HOT_CPU_MEMBERSHIP_BYTES
    );
    debug_assert_eq!(DUAL_SCALE_QS_BYTES, DUAL_SCALE_HOT_CPU_QS_BYTES);
    debug_assert_eq!(DUAL_SCALE_GROUPS, DUAL_SCALE_HOT_CPU_GROUPS);
    DualScaleHotCpuBlock {
        scale_lo: src.scale_lo,
        scale_hi: src.scale_hi,
        membership: src.membership,
        qs: src.qs,
    }
}

pub fn repack_dual_scale_tensor_cpu_ref(
    rows: usize,
    cols: usize,
    src: &[DualScaleCompactBlock],
) -> Result<DualScaleHotCpuTensor, RepackError> {
    debug_assert_eq!(DUAL_SCALE_BLOCK_SIZE, DUAL_SCALE_HOT_CPU_BLOCK_LEN);
    let outliers_ref = extract_dual_scale_outlier_csr_ref(rows, cols, src)?;
    let outliers = repack_outlier_csr_cpu(outliers_ref)?;
    let blocks = src
        .iter()
        .map(repack_dual_scale_block_cpu_ref)
        .collect::<Vec<_>>();
    Ok(DualScaleHotCpuTensor::new(rows, cols, blocks, outliers)?)
}
