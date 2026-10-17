//! DualScale portable f32 reference kernel.
//!
//! M16 scope:
//! - consumes `DualScaleHotCpuTensor` from augur-kernel-api
//! - computes f32 main-path matvec from hot blocks
//! - applies tensor-level outlier CSR correction
//! - no Graph / ExecutionPlan integration
//! - no epilogue fusion
//! - no SIMD / unsafe
//!
//! Mathematical contract:
//!
//! For each 256-element DualScale block:
//! - `group = i / 32`
//! - `q = decode_signed_q4_hot(code[i])` in `[-8, 7]`
//! - `scale = membership[i] ? scale_hi[group] : scale_lo[group]`
//! - `w_main = 0.0` if `scale == 0.0`, otherwise `scale * q`
//! - `acc += w_main * x[col + i]`
//!
//! Full matvec:
//! - `out[row] = main_dot(row, x)`
//! - out += CSR(delta * x)

use augur_kernel_api::{
    decode_dual_scale_signed_q4, dual_scale_hot_membership_bit, dual_scale_hot_q4_code,
    DualScaleHotCpuBlock, DualScaleHotCpuTensor, OutlierCsrHotCpuRow, ValidatedOutlierCsrRows,
    DUAL_SCALE_HOT_CPU_GROUPS, DUAL_SCALE_HOT_CPU_GROUP_SIZE,
};

use crate::KernelError;

// ── Hot CSR correction ─────────────────────────────────────────────────────

fn outlier_row_correction_f32_hot(row: OutlierCsrHotCpuRow<'_>, activation: &[f32]) -> f32 {
    row.entries()
        .map(|(&col, &delta)| delta.to_f32() * activation[col as usize])
        .sum()
}

// ── Main-path row dot ─────────────────────────────────────────────────────

/// Main-path DualScale row dot (caller-validated, no length checks).
pub fn dot_dual_scale_hot_f32_main_ref(
    row_blocks: &[DualScaleHotCpuBlock],
    activation: &[f32],
) -> f32 {
    debug_assert_eq!(
        activation.len(),
        row_blocks.len() * DualScaleHotCpuBlock::DECODED_LEN
    );

    let mut acc = 0.0_f32;
    for (block_idx, block) in row_blocks.iter().enumerate() {
        let start = block_idx * DualScaleHotCpuBlock::DECODED_LEN;
        let act = &activation[start..start + DualScaleHotCpuBlock::DECODED_LEN];
        acc += dot_dual_scale_block_f32_main_ref(block, act);
    }
    acc
}

fn dot_dual_scale_block_f32_main_ref(block: &DualScaleHotCpuBlock, activation: &[f32]) -> f32 {
    debug_assert_eq!(activation.len(), DualScaleHotCpuBlock::DECODED_LEN);

    let mut acc = 0.0_f32;
    for i in 0..DualScaleHotCpuBlock::DECODED_LEN {
        let group = i / DUAL_SCALE_HOT_CPU_GROUP_SIZE;
        let scale = if dual_scale_hot_membership_bit(&block.membership, i) {
            block.scale_hi[group].to_f32()
        } else {
            block.scale_lo[group].to_f32()
        };
        if scale == 0.0 {
            continue;
        }
        let q = decode_dual_scale_signed_q4(dual_scale_hot_q4_code(&block.qs, i)) as f32;
        let w = scale * q;
        acc += w * activation[i];
    }
    acc
}

// ── M18 grouped main-path row dot ──────────────────────────────────────────

/// Grouped main-path DualScale row dot (scaffold for integer-domain path).
pub fn dot_dual_scale_hot_f32_grouped_ref(
    row_blocks: &[DualScaleHotCpuBlock],
    activation: &[f32],
) -> f32 {
    debug_assert_eq!(
        activation.len(),
        row_blocks.len() * DualScaleHotCpuBlock::DECODED_LEN
    );

    let mut acc = 0.0_f32;
    for (block_idx, block) in row_blocks.iter().enumerate() {
        let start = block_idx * DualScaleHotCpuBlock::DECODED_LEN;
        let act = &activation[start..start + DualScaleHotCpuBlock::DECODED_LEN];
        acc += dot_dual_scale_block_f32_grouped_ref(block, act);
    }
    acc
}

fn dot_dual_scale_block_f32_grouped_ref(block: &DualScaleHotCpuBlock, activation: &[f32]) -> f32 {
    debug_assert_eq!(activation.len(), DualScaleHotCpuBlock::DECODED_LEN);

    let mut acc = 0.0_f32;
    for group in 0..DUAL_SCALE_HOT_CPU_GROUPS {
        let start = group * DUAL_SCALE_HOT_CPU_GROUP_SIZE;
        let end = start + DUAL_SCALE_HOT_CPU_GROUP_SIZE;

        let mut lo_sum = 0.0_f32;
        let mut hi_sum = 0.0_f32;

        for i in start..end {
            let q = decode_dual_scale_signed_q4(dual_scale_hot_q4_code(&block.qs, i)) as f32;
            if dual_scale_hot_membership_bit(&block.membership, i) {
                hi_sum += q * activation[i];
            } else {
                lo_sum += q * activation[i];
            }
        }

        let scale_lo = block.scale_lo[group].to_f32();
        if scale_lo != 0.0 {
            acc += scale_lo * lo_sum;
        }
        let scale_hi = block.scale_hi[group].to_f32();
        if scale_hi != 0.0 {
            acc += scale_hi * hi_sum;
        }
    }
    acc
}

pub fn qmatvec_dual_scale_f32_portable(
    weight: &DualScaleHotCpuTensor,
    activation: &[f32],
    out: &mut [f32],
) -> Result<(), KernelError> {
    let csr = validate_qmatvec_dual_scale_args(weight, activation, out)?;

    for ((row_blocks, csr_row), dst) in weight
        .blocks()
        .chunks_exact(weight.blocks_per_row())
        .zip(csr.iter())
        .zip(out.iter_mut())
    {
        *dst = dot_dual_scale_hot_f32_main_ref(row_blocks, activation)
            + outlier_row_correction_f32_hot(csr_row, activation);
    }
    Ok(())
}

fn validate_qmatvec_dual_scale_args<'a>(
    weight: &'a DualScaleHotCpuTensor,
    activation: &[f32],
    out: &[f32],
) -> Result<ValidatedOutlierCsrRows<'a>, KernelError> {
    validate_dual_scale_weight_shape(weight)?;
    if activation.len() != weight.cols() {
        return Err(KernelError::BadActivationLength {
            expected: weight.cols(),
            actual: activation.len(),
        });
    }
    if out.len() < weight.rows() {
        return Err(KernelError::BadOutputLength {
            expected: weight.rows(),
            actual: out.len(),
        });
    }
    let outliers = weight.outliers();
    if outliers.rows() != weight.rows() || outliers.cols() != weight.cols() {
        return Err(KernelError::BadWeightShape);
    }
    let csr = outliers.validate_rows().ok_or(KernelError::BadWeight)?;
    for row in 0..weight.rows() {
        weight.row_blocks(row).ok_or(KernelError::BadWeight)?;
    }
    Ok(csr)
}

fn validate_dual_scale_weight_shape(weight: &DualScaleHotCpuTensor) -> Result<(), KernelError> {
    let expected_cols = weight
        .blocks_per_row()
        .checked_mul(DualScaleHotCpuBlock::DECODED_LEN)
        .ok_or(KernelError::BadWeightShape)?;
    if weight.cols() != expected_cols {
        return Err(KernelError::BadWeightShape);
    }
    let expected_blocks = weight
        .rows()
        .checked_mul(weight.blocks_per_row())
        .ok_or(KernelError::BadWeightShape)?;
    if weight.blocks().len() != expected_blocks {
        return Err(KernelError::BadWeightBlockCount {
            expected: expected_blocks,
            actual: weight.blocks().len(),
        });
    }
    Ok(())
}
