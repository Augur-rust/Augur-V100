use augur_formats::{Q8_0CompactBlock, Q8_0_BLOCK_SIZE};
use augur_kernel_api::{Q8_0HotCpuBlock, Q8_0HotCpuTensor, Q8_0_HOT_CPU_BLOCK_LEN};

use crate::TensorError;

pub fn repack_q8_0_block_cpu_ref(source: &Q8_0CompactBlock) -> Q8_0HotCpuBlock {
    debug_assert_eq!(Q8_0_BLOCK_SIZE, Q8_0_HOT_CPU_BLOCK_LEN);
    Q8_0HotCpuBlock {
        d: source.d,
        qs: source.qs,
    }
}

pub fn repack_q8_0_tensor_cpu_ref(
    source: &[Q8_0CompactBlock],
    rows: usize,
    cols: usize,
) -> Result<Q8_0HotCpuTensor, TensorError> {
    let blocks = source
        .iter()
        .map(repack_q8_0_block_cpu_ref)
        .collect::<Vec<_>>();
    Q8_0HotCpuTensor::new(rows, cols, blocks)
}
