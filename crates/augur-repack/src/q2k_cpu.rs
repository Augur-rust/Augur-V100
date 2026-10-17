use augur_formats::{Q2KCompactBlock, Q2K_GROUPS};
use augur_kernel_api::{Q2KHotCpuBlock, Q2KHotCpuTensor, Q2K_HOT_CPU_GROUPS};

use crate::TensorError;

pub fn repack_q2k_block_cpu_ref(source: &Q2KCompactBlock) -> Q2KHotCpuBlock {
    debug_assert_eq!(Q2K_GROUPS, Q2K_HOT_CPU_GROUPS);
    let mut scale = [0_u8; Q2K_HOT_CPU_GROUPS];
    let mut min = [0_u8; Q2K_HOT_CPU_GROUPS];
    for index in 0..Q2K_HOT_CPU_GROUPS {
        scale[index] = source.scales[index] & 0x0f;
        min[index] = source.scales[index] >> 4;
    }
    Q2KHotCpuBlock {
        d: source.d,
        dmin: source.dmin,
        scale,
        min,
        qs: source.qs,
    }
}

pub fn repack_q2k_tensor_cpu_ref(
    source: &[Q2KCompactBlock],
    rows: usize,
    cols: usize,
) -> Result<Q2KHotCpuTensor, TensorError> {
    let blocks = source
        .iter()
        .map(repack_q2k_block_cpu_ref)
        .collect::<Vec<_>>();
    Q2KHotCpuTensor::new(rows, cols, blocks)
}
