//! Reference Q4K HotWeight × reference-format Q8K arithmetic.
//!
//! This module preserves the historical integer-domain decomposition used to
//! validate Q4K × Q8K arithmetic. It is a correctness oracle, not a production
//! kernel and not a registered execution path.

use augur_formats::{Q8KActivationBlock, Q8KActivationVector, Q8K_BLOCK_SIZE};
use augur_kernel_api::{Q4KHotCpuBlock, Q4KHotCpuTensor};

use crate::ReferenceQ8Error;

const _: [(); Q4KHotCpuBlock::DECODED_LEN] = [(); Q8K_BLOCK_SIZE];

/// Integer-domain dot product for one Q4K/Q8K block pair.
///
/// Numerical contract: NumericallyClose against elementwise decoded f32 dot.
/// The integer accumulation and floating reconstruction order are frozen.
pub fn dot_q4k_block_q8k_block_ref(
    weight: &Q4KHotCpuBlock,
    activation: &Q8KActivationBlock,
) -> f32 {
    let activation_scale = activation.d.to_f32();
    let weight_scale = weight.d.to_f32();
    let weight_min_scale = weight.dmin.to_f32();

    let mut q_dot = [0_i32; 8];
    let mut activation_sum = [0_i32; 8];

    for index in 0..Q4KHotCpuBlock::DECODED_LEN {
        let group = index / 32;
        let q4 = weight.quant_at(index) as i32;

        let q8 = activation.qs[index] as i32;

        q_dot[group] += q4 * q8;
        activation_sum[group] += q8;
    }

    let mut output = 0.0_f32;

    for group in 0..8 {
        output +=
            weight_scale * activation_scale * (weight.scale[group] as f32) * (q_dot[group] as f32);

        output -= weight_min_scale
            * activation_scale
            * (weight.min[group] as f32)
            * (activation_sum[group] as f32);
    }

    output
}

/// Checked Q4K/Q8K row dot.
pub fn dot_q4k_hot_q8k_ref(
    weight: &[Q4KHotCpuBlock],
    activation: &[Q8KActivationBlock],
) -> Result<f32, ReferenceQ8Error> {
    if weight.len() != activation.len() {
        return Err(ReferenceQ8Error::BadActivationBlockCount {
            expected: weight.len(),
            actual: activation.len(),
        });
    }

    Ok(dot_q4k_hot_q8k_ref_unchecked(weight, activation))
}

/// Reference Q4K/Q8K matrix-vector multiplication.
pub fn qmatmul_q4k_q8k_ref(
    weight: &Q4KHotCpuTensor,
    activation: &Q8KActivationVector,
    output: &mut [f32],
) -> Result<(), ReferenceQ8Error> {
    qmatmul_q4k_q8k_checked(weight, activation, None, output, |value, _| value)
}

/// Reference Q4K/Q8K matrix-vector multiplication with bias.
pub fn qmatmul_q4k_q8k_bias_ref(
    weight: &Q4KHotCpuTensor,
    activation: &Q8KActivationVector,
    bias: &[f32],
    output: &mut [f32],
) -> Result<(), ReferenceQ8Error> {
    qmatmul_q4k_q8k_checked(weight, activation, Some(bias), output, |value, bias| {
        value + bias
    })
}

/// Reference Q4K/Q8K matrix-vector multiplication with BiasSiLU.
pub fn qmatmul_q4k_q8k_bias_silu_ref(
    weight: &Q4KHotCpuTensor,
    activation: &Q8KActivationVector,
    bias: &[f32],
    output: &mut [f32],
) -> Result<(), ReferenceQ8Error> {
    qmatmul_q4k_q8k_checked(weight, activation, Some(bias), output, |value, bias| {
        silu_f32_ref(value + bias)
    })
}

fn qmatmul_q4k_q8k_checked(
    weight: &Q4KHotCpuTensor,
    activation: &Q8KActivationVector,
    bias: Option<&[f32]>,
    output: &mut [f32],
    epilogue: impl Fn(f32, f32) -> f32,
) -> Result<(), ReferenceQ8Error> {
    validate_qmatmul_q8k_args(weight, activation, output, bias)?;

    let activation_blocks = activation.blocks();
    let row_width = weight.blocks_per_row();
    for (row, (row_blocks, dst)) in weight
        .blocks()
        .chunks_exact(row_width)
        .zip(output.iter_mut())
        .enumerate()
    {
        let value = dot_q4k_hot_q8k_ref_unchecked(row_blocks, activation_blocks);
        *dst = epilogue(value, bias.map_or(0.0, |values| values[row]));
    }
    Ok(())
}

fn dot_q4k_hot_q8k_ref_unchecked(
    weight: &[Q4KHotCpuBlock],
    activation: &[Q8KActivationBlock],
) -> f32 {
    debug_assert_eq!(weight.len(), activation.len());

    let mut output = 0.0_f32;

    for (weight_block, activation_block) in weight.iter().zip(activation.iter()) {
        output += dot_q4k_block_q8k_block_ref(weight_block, activation_block);
    }

    output
}

fn validate_qmatmul_q8k_args(
    weight: &Q4KHotCpuTensor,
    activation: &Q8KActivationVector,
    output: &[f32],
    bias: Option<&[f32]>,
) -> Result<(), ReferenceQ8Error> {
    validate_q4k_weight_shape(weight)?;

    if activation.len_elements() != weight.cols() {
        return Err(ReferenceQ8Error::BadActivationLength {
            expected: weight.cols(),
            actual: activation.len_elements(),
        });
    }

    if activation.block_count() != weight.blocks_per_row() {
        return Err(ReferenceQ8Error::BadActivationBlockCount {
            expected: weight.blocks_per_row(),
            actual: activation.block_count(),
        });
    }

    if output.len() < weight.rows() {
        return Err(ReferenceQ8Error::BadOutputLength {
            expected: weight.rows(),
            actual: output.len(),
        });
    }

    if let Some(bias) = bias {
        if bias.len() < weight.rows() {
            return Err(ReferenceQ8Error::BadBiasLength {
                expected: weight.rows(),
                actual: bias.len(),
            });
        }
    }

    // Preflight every row before the first output write. This preserves the
    // strong output-unchanged guarantee even if public shape fields were
    // mutated after construction.
    for row in 0..weight.rows() {
        weight.row_blocks(row).ok_or(ReferenceQ8Error::BadWeight)?;
    }

    Ok(())
}

fn validate_q4k_weight_shape(weight: &Q4KHotCpuTensor) -> Result<(), ReferenceQ8Error> {
    let expected_cols = weight
        .blocks_per_row()
        .checked_mul(Q4KHotCpuBlock::DECODED_LEN)
        .ok_or(ReferenceQ8Error::BadWeightShape)?;

    if weight.cols() != expected_cols {
        return Err(ReferenceQ8Error::BadWeightShape);
    }

    let expected_blocks = weight
        .rows()
        .checked_mul(weight.blocks_per_row())
        .ok_or(ReferenceQ8Error::BadWeightShape)?;

    if weight.blocks().len() != expected_blocks {
        return Err(ReferenceQ8Error::BadWeightBlockCount {
            expected: expected_blocks,
            actual: weight.blocks().len(),
        });
    }

    Ok(())
}

#[inline]
fn silu_f32_ref(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}
