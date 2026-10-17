use crate::TensorError;
use augur_formats::{Q8KActivationBlock, Q8KActivationVector, Q8K_BLOCK_SIZE};
use augur_kernel_api::{Q8KActivationHotCpuBlock, Q8KActivationHotCpuVector, Q8K_HOT_CPU_BLOCK_LEN};

pub fn repack_q8k_activation_block_cpu_ref(src: &Q8KActivationBlock) -> Q8KActivationHotCpuBlock {
    debug_assert_eq!(Q8K_BLOCK_SIZE, Q8K_HOT_CPU_BLOCK_LEN);
    Q8KActivationHotCpuBlock {
        d: src.d,
        qs: src.qs,
    }
}

pub fn repack_q8k_activation_vector_cpu_ref(
    src: &Q8KActivationVector,
) -> Result<Q8KActivationHotCpuVector, TensorError> {
    let blocks = src
        .blocks()
        .iter()
        .map(repack_q8k_activation_block_cpu_ref)
        .collect::<Vec<_>>();
    Q8KActivationHotCpuVector::new(src.len_elements(), blocks)
}
