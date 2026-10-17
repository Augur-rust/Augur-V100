//! Portable integer-domain M43 weight kernels.

use augur_kernel_api::{
    KernelLaunchError, Q2KHotCpuBlock, Q2KHotCpuTensor, Q8KActivationHotCpuBlock,
    Q8KActivationHotCpuVector, Q8_0HotCpuBlock, Q8_0HotCpuTensor,
};

use crate::KernelError;

const Q8_0_BLOCKS_PER_Q8K: usize = 256 / Q8_0HotCpuBlock::DECODED_LEN;

pub fn qmatvec_q8_0_q8hot_cpu_portable(
    weight: &Q8_0HotCpuTensor,
    activation: &Q8KActivationHotCpuVector,
    output: &mut [f32],
) -> Result<(), KernelError> {
    validate_q8_0(weight, activation, output)?;
    for row in 0..weight.rows() {
        output[row] = dot_q8_0_unchecked(
            weight.row_blocks(row).ok_or(KernelError::BadWeight)?,
            activation.blocks(),
        );
    }
    Ok(())
}

pub fn qmatvec_q2k_q8hot_cpu_portable(
    weight: &Q2KHotCpuTensor,
    activation: &Q8KActivationHotCpuVector,
    output: &mut [f32],
) -> Result<(), KernelError> {
    validate_q2k(weight, activation, output)?;
    for row in 0..weight.rows() {
        output[row] = dot_q2k_unchecked(
            weight.row_blocks(row).ok_or(KernelError::BadWeight)?,
            activation.blocks(),
        );
    }
    Ok(())
}

pub fn launch_qmatvec_q8_0_q8hot_cpu_portable(
    weight: &Q8_0HotCpuTensor,
    activation: &Q8KActivationHotCpuVector,
    output: &mut [f32],
) -> Result<(), KernelLaunchError> {
    qmatvec_q8_0_q8hot_cpu_portable(weight, activation, output)
        .map_err(KernelError::into_launch_error)
}

pub fn launch_qmatvec_q2k_q8hot_cpu_portable(
    weight: &Q2KHotCpuTensor,
    activation: &Q8KActivationHotCpuVector,
    output: &mut [f32],
) -> Result<(), KernelLaunchError> {
    qmatvec_q2k_q8hot_cpu_portable(weight, activation, output)
        .map_err(KernelError::into_launch_error)
}

fn dot_q8_0_unchecked(weight: &[Q8_0HotCpuBlock], activation: &[Q8KActivationHotCpuBlock]) -> f32 {
    let mut output = 0.0_f32;
    for (activation_block, weight_group) in activation
        .iter()
        .zip(weight.chunks_exact(Q8_0_BLOCKS_PER_Q8K))
    {
        let activation_scale = activation_block.d.to_f32();
        for (group, weight_block) in weight_group.iter().enumerate() {
            let start = group * Q8_0HotCpuBlock::DECODED_LEN;
            let mut integer_dot = 0_i32;
            for lane in 0..Q8_0HotCpuBlock::DECODED_LEN {
                integer_dot +=
                    (weight_block.qs[lane] as i32) * (activation_block.qs[start + lane] as i32);
            }
            output += weight_block.d.to_f32() * activation_scale * (integer_dot as f32);
        }
    }
    output
}

fn dot_q2k_unchecked(weight: &[Q2KHotCpuBlock], activation: &[Q8KActivationHotCpuBlock]) -> f32 {
    let mut output = 0.0_f32;
    for (weight_block, activation_block) in weight.iter().zip(activation.iter()) {
        let activation_scale = activation_block.d.to_f32();
        let super_scale = weight_block.d.to_f32();
        let super_min = weight_block.dmin.to_f32();
        for group in 0..16 {
            let start = group * 16;
            let mut quant_dot = 0_i32;
            let mut activation_sum = 0_i32;
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
    output
}

fn validate_q8_0(
    weight: &Q8_0HotCpuTensor,
    activation: &Q8KActivationHotCpuVector,
    output: &[f32],
) -> Result<(), KernelError> {
    validate_common(weight.rows(), weight.cols(), activation, output)?;
    let expected = activation
        .block_count()
        .checked_mul(Q8_0_BLOCKS_PER_Q8K)
        .ok_or(KernelError::BadWeightShape)?;
    if weight.blocks_per_row() != expected {
        return Err(KernelError::BadWeightBlockCount {
            expected,
            actual: weight.blocks_per_row(),
        });
    }
    Ok(())
}

fn validate_q2k(
    weight: &Q2KHotCpuTensor,
    activation: &Q8KActivationHotCpuVector,
    output: &[f32],
) -> Result<(), KernelError> {
    validate_common(weight.rows(), weight.cols(), activation, output)?;
    if weight.blocks_per_row() != activation.block_count() {
        return Err(KernelError::BadActivationBlockCount {
            expected: weight.blocks_per_row(),
            actual: activation.block_count(),
        });
    }
    Ok(())
}

fn validate_common(
    rows: usize,
    cols: usize,
    activation: &Q8KActivationHotCpuVector,
    output: &[f32],
) -> Result<(), KernelError> {
    if activation.len_elements() != cols {
        return Err(KernelError::BadActivationLength {
            expected: cols,
            actual: activation.len_elements(),
        });
    }
    if output.len() < rows {
        return Err(KernelError::BadOutputLength {
            expected: rows,
            actual: output.len(),
        });
    }
    Ok(())
}
