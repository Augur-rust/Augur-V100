//! Reference DualScale × Q8K arithmetic.
//!
//! This module consumes the reference-format Q8K activation representation
//! from `augur-formats`. It is a correctness oracle only and is not registered
//! or reachable from the production ExecutionPlan path.

use augur_formats::{Q8KActivationBlock, Q8KActivationVector, Q8K_BLOCK_SIZE};
use augur_kernel_api::{
    decode_dual_scale_signed_q4, dual_scale_hot_membership_bit, dual_scale_hot_q4_code,
    DualScaleHotCpuBlock, DualScaleHotCpuTensor, OutlierCsrHotCpu, DUAL_SCALE_HOT_CPU_BLOCK_LEN,
    DUAL_SCALE_HOT_CPU_GROUPS, DUAL_SCALE_HOT_CPU_GROUP_SIZE,
};

use crate::ReferenceQ8Error;

const _: [(); DualScaleHotCpuBlock::DECODED_LEN] = [(); Q8K_BLOCK_SIZE];

/// Reference-format Q8K DualScale block dot.
///
/// The integer-domain accumulation is grouped into separate lo/hi
/// accumulators according to the DualScale membership mask.
pub fn dot_dual_scale_block_q8k_grouped_ref(
    weight: &DualScaleHotCpuBlock,
    activation: &Q8KActivationBlock,
) -> f32 {
    let activation_scale = activation.d.to_f32();

    if activation_scale == 0.0 {
        return 0.0;
    }

    let mut output = 0.0_f32;

    for group in 0..DUAL_SCALE_HOT_CPU_GROUPS {
        let start = group * DUAL_SCALE_HOT_CPU_GROUP_SIZE;
        let end = start + DUAL_SCALE_HOT_CPU_GROUP_SIZE;

        let mut lo_acc = 0_i32;
        let mut hi_acc = 0_i32;

        for index in start..end {
            let q4 = i32::from(decode_dual_scale_signed_q4(dual_scale_hot_q4_code(
                &weight.qs, index,
            )));
            let q8 = i32::from(activation.qs[index]);
            let product = q4 * q8;

            if dual_scale_hot_membership_bit(&weight.membership, index) {
                hi_acc += product;
            } else {
                lo_acc += product;
            }
        }

        let scale_lo = weight.scale_lo[group].to_f32();
        if scale_lo != 0.0 {
            output += (scale_lo * activation_scale) * (lo_acc as f32);
        }

        let scale_hi = weight.scale_hi[group].to_f32();
        if scale_hi != 0.0 {
            output += (scale_hi * activation_scale) * (hi_acc as f32);
        }
    }

    output
}

/// Reference-format Q8K DualScale row dot.
pub fn dot_dual_scale_hot_q8k_grouped_ref(
    row_blocks: &[DualScaleHotCpuBlock],
    activation: &Q8KActivationVector,
) -> f32 {
    debug_assert_eq!(row_blocks.len(), activation.block_count());
    debug_assert_eq!(
        row_blocks.len() * DualScaleHotCpuBlock::DECODED_LEN,
        activation.len_elements()
    );

    let mut output = 0.0_f32;

    for (weight_block, activation_block) in row_blocks.iter().zip(activation.blocks().iter()) {
        output += dot_dual_scale_block_q8k_grouped_ref(weight_block, activation_block);
    }

    output
}

/// Reference-format Q8K DualScale qmatvec.
///
/// All recoverable validation happens before the first write to `output`.
/// The reference path may allocate temporary storage.
pub fn qmatvec_dual_scale_q8k_ref(
    weight: &DualScaleHotCpuTensor,
    activation: &Q8KActivationVector,
    output: &mut [f32],
) -> Result<(), ReferenceQ8Error> {
    validate_qmatvec_args(weight, activation, output)?;

    let mut temporary = vec![0.0_f32; weight.rows()];

    for (row, value) in temporary.iter_mut().enumerate() {
        let row_blocks = weight.row_blocks(row).ok_or(ReferenceQ8Error::BadWeight)?;
        *value = dot_dual_scale_hot_q8k_grouped_ref(row_blocks, activation);
    }

    let mut decoded_activation = vec![0.0_f32; activation.len_elements()];
    decode_q8k_vector_prevalidated(activation, &mut decoded_activation);

    add_outlier_csr_correction(weight.outliers(), &decoded_activation, &mut temporary)?;

    output[..weight.rows()].copy_from_slice(&temporary);
    Ok(())
}

fn validate_qmatvec_args(
    weight: &DualScaleHotCpuTensor,
    activation: &Q8KActivationVector,
    output: &[f32],
) -> Result<(), ReferenceQ8Error> {
    validate_weight_shape(weight)?;

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

    let outliers = weight.outliers();
    if outliers.rows() != weight.rows() || outliers.cols() != weight.cols() {
        return Err(ReferenceQ8Error::BadWeightShape);
    }

    for row in 0..weight.rows() {
        weight.row_blocks(row).ok_or(ReferenceQ8Error::BadWeight)?;
    }

    Ok(())
}

fn validate_weight_shape(weight: &DualScaleHotCpuTensor) -> Result<(), ReferenceQ8Error> {
    let expected_cols = weight
        .blocks_per_row()
        .checked_mul(DUAL_SCALE_HOT_CPU_BLOCK_LEN)
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

fn decode_q8k_vector_prevalidated(activation: &Q8KActivationVector, output: &mut [f32]) {
    debug_assert_eq!(output.len(), activation.len_elements());

    for (block_index, block) in activation.blocks().iter().enumerate() {
        let start = block_index * Q8K_BLOCK_SIZE;
        let scale = block.d.to_f32();

        for lane in 0..Q8K_BLOCK_SIZE {
            output[start + lane] = scale * f32::from(block.qs[lane]);
        }
    }
}

fn add_outlier_csr_correction(
    csr: &OutlierCsrHotCpu,
    activation: &[f32],
    output: &mut [f32],
) -> Result<(), ReferenceQ8Error> {
    if activation.len() != csr.cols() {
        return Err(ReferenceQ8Error::BadActivationLength {
            expected: csr.cols(),
            actual: activation.len(),
        });
    }

    if output.len() < csr.rows() {
        return Err(ReferenceQ8Error::BadOutputLength {
            expected: csr.rows(),
            actual: output.len(),
        });
    }

    for (row, value) in output.iter_mut().enumerate().take(csr.rows()) {
        let range = csr.row_range(row).ok_or(ReferenceQ8Error::BadWeight)?;

        let mut correction = 0.0_f32;

        for position in range {
            let column = csr.col_idx()[position] as usize;

            if column >= activation.len() {
                return Err(ReferenceQ8Error::BadWeight);
            }

            correction += csr.delta()[position].to_f32() * activation[column];
        }

        *value += correction;
    }

    Ok(())
}
