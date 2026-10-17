use augur_formats::{Q4KCompactBlock, ReferenceDecode};

use crate::ReferenceQ8Error;

/// Independent Compact-layout Q4K reference GEMM.
///
/// `weight` is `[n, k / 256]`, `input` is row-major `[m, k]`, and `output`
/// is row-major `[m, n]`.
pub fn q4k_f32_gemm_reference(
    weight: &[Q4KCompactBlock],
    m: usize,
    n: usize,
    k: usize,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), ReferenceQ8Error> {
    if m == 0 || n == 0 || k == 0 || !k.is_multiple_of(Q4KCompactBlock::DECODED_LEN) {
        return Err(ReferenceQ8Error::BadWeightShape);
    }
    let blocks_per_row = k / Q4KCompactBlock::DECODED_LEN;
    let expected_blocks = n
        .checked_mul(blocks_per_row)
        .ok_or(ReferenceQ8Error::BadWeightShape)?;
    if weight.len() != expected_blocks {
        return Err(ReferenceQ8Error::BadWeightBlockCount {
            expected: expected_blocks,
            actual: weight.len(),
        });
    }
    let expected_input = m.checked_mul(k).ok_or(ReferenceQ8Error::BadWeightShape)?;
    if input.len() != expected_input {
        return Err(ReferenceQ8Error::BadActivationLength {
            expected: expected_input,
            actual: input.len(),
        });
    }
    let expected_output = m.checked_mul(n).ok_or(ReferenceQ8Error::BadWeightShape)?;
    if output.len() < expected_output {
        return Err(ReferenceQ8Error::BadOutputLength {
            expected: expected_output,
            actual: output.len(),
        });
    }

    let mut decoded = [0.0_f32; Q4KCompactBlock::DECODED_LEN];
    for sample in 0..m {
        for row in 0..n {
            let mut acc = 0.0_f32;
            for block_index in 0..blocks_per_row {
                weight[row * blocks_per_row + block_index]
                    .decode_ref(&mut decoded)
                    .map_err(|_| ReferenceQ8Error::BadWeight)?;
                let input_start = sample * k + block_index * Q4KCompactBlock::DECODED_LEN;
                for lane in 0..Q4KCompactBlock::DECODED_LEN {
                    acc += decoded[lane] * input[input_start + lane];
                }
            }
            output[sample * n + row] = acc;
        }
    }
    Ok(())
}
