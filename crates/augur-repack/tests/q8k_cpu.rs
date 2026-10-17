use augur_formats::{quantize_q8k_activation_ref, quantize_q8k_activation_vector_ref, Q8K_BLOCK_SIZE};

use augur_kernel_api::Q8KActivationHotCpuVector;
use augur_repack::{
    repack_q8k_activation_block_cpu_ref, repack_q8k_activation_vector_cpu_ref, TensorError,
};

fn activation_src(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i % 37) as f32 - 18.0) * 0.03125)
        .collect()
}

#[test]
fn q8k_activation_block_repack_preserves_fields_bit_exact() {
    let src = activation_src(Q8K_BLOCK_SIZE);
    let compact = quantize_q8k_activation_ref(&src).unwrap();

    let hot = repack_q8k_activation_block_cpu_ref(&compact);

    assert_eq!(hot.d, compact.d);
    assert_eq!(hot.qs, compact.qs);
}

#[test]
fn q8k_activation_vector_repack_preserves_shape_and_blocks() {
    let src = activation_src(Q8K_BLOCK_SIZE * 2);
    let compact = quantize_q8k_activation_vector_ref(&src).unwrap();

    let hot = repack_q8k_activation_vector_cpu_ref(&compact).unwrap();

    assert_eq!(hot.len_elements(), Q8K_BLOCK_SIZE * 2);
    assert_eq!(hot.block_count(), 2);
    assert_eq!(hot.blocks().len(), 2);

    for (hot_block, compact_block) in hot.blocks().iter().zip(compact.blocks().iter()) {
        assert_eq!(hot_block.d, compact_block.d);
        assert_eq!(hot_block.qs, compact_block.qs);
    }
}

#[test]
fn q8k_activation_hot_vector_block_returns_optional_block() {
    let src = activation_src(Q8K_BLOCK_SIZE * 2);
    let compact = quantize_q8k_activation_vector_ref(&src).unwrap();

    let hot = repack_q8k_activation_vector_cpu_ref(&compact).unwrap();

    assert_eq!(hot.block(0), Some(&hot.blocks()[0]));
    assert_eq!(hot.block(1), Some(&hot.blocks()[1]));
    assert_eq!(hot.block(2), None);
}

#[test]
fn q8k_activation_hot_vector_new_rejects_bad_length_shape() {
    let src = activation_src(Q8K_BLOCK_SIZE);
    let compact = quantize_q8k_activation_vector_ref(&src).unwrap();

    let blocks = compact
        .blocks()
        .iter()
        .map(repack_q8k_activation_block_cpu_ref)
        .collect::<Vec<_>>();

    assert_eq!(
        Q8KActivationHotCpuVector::new(Q8K_BLOCK_SIZE + 1, blocks),
        Err(TensorError::BadShape)
    );
}

#[test]
fn q8k_activation_hot_vector_new_rejects_empty_length() {
    assert_eq!(
        Q8KActivationHotCpuVector::new(0, Vec::new()),
        Err(TensorError::BadShape)
    );
}

#[test]
fn q8k_activation_hot_vector_new_rejects_block_count_mismatch() {
    let src = activation_src(Q8K_BLOCK_SIZE);
    let compact = quantize_q8k_activation_vector_ref(&src).unwrap();

    let blocks = compact
        .blocks()
        .iter()
        .map(repack_q8k_activation_block_cpu_ref)
        .collect::<Vec<_>>();

    assert_eq!(
        Q8KActivationHotCpuVector::new(Q8K_BLOCK_SIZE * 2, blocks),
        Err(TensorError::BadBlockCount {
            expected: 2,
            actual: 1,
        })
    );
}
