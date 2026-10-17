use augur_formats::Iq2XxsCompactBlock;
use augur_kernel_api::{Iq2XxsHotCpuBlock, Iq2XxsHotCpuTensor};

use crate::TensorError;

pub fn repack_iq2_xxs_block_cpu_ref(source: &Iq2XxsCompactBlock) -> Iq2XxsHotCpuBlock {
    Iq2XxsHotCpuBlock {
        d: source.d,
        qs: source.qs,
    }
}

pub fn repack_iq2_xxs_tensor_cpu_ref(
    source: &[Iq2XxsCompactBlock],
    rows: usize,
    cols: usize,
) -> Result<Iq2XxsHotCpuTensor, TensorError> {
    let blocks = source
        .iter()
        .map(repack_iq2_xxs_block_cpu_ref)
        .collect::<Vec<_>>();
    Iq2XxsHotCpuTensor::new(rows, cols, blocks)
}
