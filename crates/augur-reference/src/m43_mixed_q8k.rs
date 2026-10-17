//! Integer-domain Q8_0/Q2_K weight x Q8K activation reference arithmetic.

use augur_formats::{Q8KActivationBlock, Q8KActivationVector, Q8K_BLOCK_SIZE};
use augur_kernel_api::{Q2KHotCpuBlock, Q2KHotCpuTensor, Q8_0HotCpuBlock, Q8_0HotCpuTensor};

use crate::ReferenceQ8Error;

const Q8_0_BLOCKS_PER_Q8K: usize = Q8K_BLOCK_SIZE / Q8_0HotCpuBlock::DECODED_LEN;

pub fn dot_q8_0_hot_q8k_ref(
    weight: &[Q8_0HotCpuBlock],
    activation: &[Q8KActivationBlock],
) -> Result<f32, ReferenceQ8Error> {
    let expected = activation
        .len()
        .checked_mul(Q8_0_BLOCKS_PER_Q8K)
        .ok_or(ReferenceQ8Error::BadWeightShape)?;
    if weight.len() != expected {
        return Err(ReferenceQ8Error::BadWeightBlockCount {
            expected,
            actual: weight.len(),
        });
    }
    let mut output = 0.0_f32;
    for (activation_block, weight_group) in activation
        .iter()
        .zip(weight.chunks_exact(Q8_0_BLOCKS_PER_Q8K))
    {
        let activation_scale = activation_block.d.to_f32();
        for (group, weight_block) in weight_group.iter().enumerate() {
            let mut integer_dot = 0_i32;
            let activation_start = group * Q8_0HotCpuBlock::DECODED_LEN;
            for lane in 0..Q8_0HotCpuBlock::DECODED_LEN {
                integer_dot += (weight_block.qs[lane] as i32)
                    * (activation_block.qs[activation_start + lane] as i32);
            }
            output += weight_block.d.to_f32() * activation_scale * (integer_dot as f32);
        }
    }
    Ok(output)
}

pub fn dot_q2k_hot_q8k_ref(
    weight: &[Q2KHotCpuBlock],
    activation: &[Q8KActivationBlock],
) -> Result<f32, ReferenceQ8Error> {
    if weight.len() != activation.len() {
        return Err(ReferenceQ8Error::BadActivationBlockCount {
            expected: weight.len(),
            actual: activation.len(),
        });
    }
    let mut output = 0.0_f32;
    for (weight_block, activation_block) in weight.iter().zip(activation.iter()) {
        let activation_scale = activation_block.d.to_f32();
        let super_scale = weight_block.d.to_f32();
        let super_min = weight_block.dmin.to_f32();
        for group in 0..16 {
            let mut quant_dot = 0_i32;
            let mut activation_sum = 0_i32;
            let start = group * 16;
            for lane in 0..16 {
                let activation_quant = activation_block.qs[start + lane] as i32;
                quant_dot += (weight_block.quant_at(start + lane) as i32) * activation_quant;
                activation_sum += activation_quant;
            }
            output += super_scale
                * activation_scale
                * (weight_block.scale[group] as f32)
                * (quant_dot as f32);
            output -= super_min
                * activation_scale
                * (weight_block.min[group] as f32)
                * (activation_sum as f32);
        }
    }
    Ok(output)
}

pub fn qmatvec_q8_0_q8k_ref(
    weight: &Q8_0HotCpuTensor,
    activation: &Q8KActivationVector,
    output: &mut [f32],
) -> Result<(), ReferenceQ8Error> {
    validate_q8_0(weight, activation, output)?;
    for (row, value) in output.iter_mut().take(weight.rows()).enumerate() {
        let row_blocks = weight.row_blocks(row).ok_or(ReferenceQ8Error::BadWeight)?;
        *value = dot_q8_0_hot_q8k_ref(row_blocks, activation.blocks())?;
    }
    Ok(())
}

pub fn qmatvec_q2k_q8k_ref(
    weight: &Q2KHotCpuTensor,
    activation: &Q8KActivationVector,
    output: &mut [f32],
) -> Result<(), ReferenceQ8Error> {
    validate_q2k(weight, activation, output)?;
    for (row, value) in output.iter_mut().take(weight.rows()).enumerate() {
        let row_blocks = weight.row_blocks(row).ok_or(ReferenceQ8Error::BadWeight)?;
        *value = dot_q2k_hot_q8k_ref(row_blocks, activation.blocks())?;
    }
    Ok(())
}

fn validate_q8_0(
    weight: &Q8_0HotCpuTensor,
    activation: &Q8KActivationVector,
    output: &[f32],
) -> Result<(), ReferenceQ8Error> {
    validate_common(weight.rows(), weight.cols(), activation, output)?;
    let expected = activation
        .block_count()
        .checked_mul(Q8_0_BLOCKS_PER_Q8K)
        .ok_or(ReferenceQ8Error::BadWeightShape)?;
    if weight.blocks_per_row() != expected {
        return Err(ReferenceQ8Error::BadWeightBlockCount {
            expected,
            actual: weight.blocks_per_row(),
        });
    }
    for row in 0..weight.rows() {
        weight.row_blocks(row).ok_or(ReferenceQ8Error::BadWeight)?;
    }
    Ok(())
}

fn validate_q2k(
    weight: &Q2KHotCpuTensor,
    activation: &Q8KActivationVector,
    output: &[f32],
) -> Result<(), ReferenceQ8Error> {
    validate_common(weight.rows(), weight.cols(), activation, output)?;
    if weight.blocks_per_row() != activation.block_count() {
        return Err(ReferenceQ8Error::BadActivationBlockCount {
            expected: weight.blocks_per_row(),
            actual: activation.block_count(),
        });
    }
    for row in 0..weight.rows() {
        weight.row_blocks(row).ok_or(ReferenceQ8Error::BadWeight)?;
    }
    Ok(())
}

fn validate_common(
    rows: usize,
    cols: usize,
    activation: &Q8KActivationVector,
    output: &[f32],
) -> Result<(), ReferenceQ8Error> {
    if activation.len_elements() != cols {
        return Err(ReferenceQ8Error::BadActivationLength {
            expected: cols,
            actual: activation.len_elements(),
        });
    }
    if output.len() < rows {
        return Err(ReferenceQ8Error::BadOutputLength {
            expected: rows,
            actual: output.len(),
        });
    }
    Ok(())
}
