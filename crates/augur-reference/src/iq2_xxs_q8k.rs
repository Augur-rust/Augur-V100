//! Independent scalar IQ2_XXS weight x Q8K activation reference arithmetic.

use augur_formats::{iq2_xxs_grid_value, iq2_xxs_sign_mask, Q8KActivationBlock, Q8KActivationVector};
use augur_kernel_api::{Iq2XxsHotCpuBlock, Iq2XxsHotCpuTensor};

use crate::ReferenceQ8Error;

pub fn dot_iq2_xxs_hot_q8k_ref(
    weight: &[Iq2XxsHotCpuBlock],
    activation: &[Q8KActivationBlock],
) -> Result<f32, ReferenceQ8Error> {
    if weight.len() != activation.len() {
        return Err(ReferenceQ8Error::BadActivationBlockCount {
            expected: weight.len(),
            actual: activation.len(),
        });
    }
    let mut output = 0.0_f32;
    for (weight_block, activation_block) in weight.iter().zip(activation) {
        for group in 0..8 {
            let words = weight_block
                .group_words(group)
                .ok_or(ReferenceQ8Error::BadWeight)?;
            let [grid0, grid1] = words[0].to_le_bytes();
            let [grid2, grid3] = words[1].to_le_bytes();
            let grids = [grid0, grid1, grid2, grid3];
            let sign_scale = u32::from_le_bytes([
                words[2].to_le_bytes()[0],
                words[2].to_le_bytes()[1],
                words[3].to_le_bytes()[0],
                words[3].to_le_bytes()[1],
            ]);
            let group_scale = (2 * (sign_scale >> 28) + 1) as f32;
            let mut integer_dot = 0_i32;
            for (sub_group, grid) in grids.iter().enumerate() {
                let sign_index = ((sign_scale >> (7 * sub_group)) & 0x7f) as u8;
                let sign_mask = iq2_xxs_sign_mask(sign_index);
                for lane in 0..8 {
                    let magnitude = iq2_xxs_grid_value(*grid, lane) as i32;
                    let signed = if sign_mask & (1 << lane) == 0 {
                        magnitude
                    } else {
                        -magnitude
                    };
                    integer_dot +=
                        signed * (activation_block.qs[group * 32 + sub_group * 8 + lane] as i32);
                }
            }
            output += weight_block.d.to_f32()
                * activation_block.d.to_f32()
                * group_scale
                * integer_dot as f32
                * 0.125;
        }
    }
    Ok(output)
}

pub fn qmatvec_iq2_xxs_q8k_ref(
    weight: &Iq2XxsHotCpuTensor,
    activation: &Q8KActivationVector,
    output: &mut [f32],
) -> Result<(), ReferenceQ8Error> {
    if activation.len_elements() != weight.cols() {
        return Err(ReferenceQ8Error::BadActivationLength {
            expected: weight.cols(),
            actual: activation.len_elements(),
        });
    }
    if output.len() < weight.rows() {
        return Err(ReferenceQ8Error::BadOutputLength {
            expected: weight.rows(),
            actual: output.len(),
        });
    }
    for (row, value) in output.iter_mut().take(weight.rows()).enumerate() {
        *value = dot_iq2_xxs_hot_q8k_ref(
            weight.row_blocks(row).ok_or(ReferenceQ8Error::BadWeight)?,
            activation.blocks(),
        )?;
    }
    Ok(())
}
