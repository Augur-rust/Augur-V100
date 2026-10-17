use augur_formats::{
    quantize_q8k_activation_vector_ref, FormatError, Q8KActivationVector, Q8K_BLOCK_SIZE,
};

#[test]
fn q8k_activation_vector_quantizes_two_blocks() {
    let src: Vec<f32> = (0..Q8K_BLOCK_SIZE * 2)
        .map(|i| ((i % 31) as f32 - 15.0) * 0.125)
        .collect();

    let q = quantize_q8k_activation_vector_ref(&src).unwrap();

    assert_eq!(q.len_elements(), Q8K_BLOCK_SIZE * 2);
    assert_eq!(q.block_count(), 2);
    assert_eq!(q.blocks().len(), 2);
}

#[test]
fn q8k_activation_vector_rejects_non_multiple_of_block() {
    let src = vec![0.0f32; Q8K_BLOCK_SIZE + 1];

    assert_eq!(
        quantize_q8k_activation_vector_ref(&src),
        Err(FormatError::InputLengthMismatch {
            expected: Q8K_BLOCK_SIZE * 2,
            actual: Q8K_BLOCK_SIZE + 1,
        })
    );
}

#[test]
fn q8k_activation_vector_decode_rejects_short_output() {
    let src: Vec<f32> = (0..Q8K_BLOCK_SIZE)
        .map(|i| (i as f32 - 128.0) * 0.01)
        .collect();

    let q = quantize_q8k_activation_vector_ref(&src).unwrap();
    let mut out = vec![0.0f32; Q8K_BLOCK_SIZE - 1];

    assert_eq!(
        q.decode_ref(&mut out),
        Err(FormatError::OutputTooSmall {
            expected: Q8K_BLOCK_SIZE,
            actual: Q8K_BLOCK_SIZE - 1,
        })
    );
}

#[test]
fn q8k_activation_vector_new_rejects_block_count_mismatch() {
    let src = vec![0.0f32; Q8K_BLOCK_SIZE];
    let q = quantize_q8k_activation_vector_ref(&src).unwrap();

    assert_eq!(
        Q8KActivationVector::new(q.blocks().to_vec(), Q8K_BLOCK_SIZE * 2),
        Err(FormatError::InputLengthMismatch {
            expected: 2,
            actual: 1,
        })
    );
}
