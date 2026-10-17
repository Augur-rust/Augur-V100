//! Allocation-free CPU portable DualScale x Q8Hot decode qmatvec.
//!
//! (module doc truncated for brevity — same as before)
use crate::KernelError;
use augur_kernel_api::{
    decode_dual_scale_signed_q4, dual_scale_hot_membership_bit, dual_scale_hot_q4_code,
    BoundDualScaleQ8Hot, DualScaleHotCpuBlock, DualScaleHotCpuTensor, KernelLaunchError,
    Q8KActivationHotCpuBlock, Q8KActivationHotCpuVector, ValidatedOutlierCsrRows,
    DUAL_SCALE_HOT_CPU_GROUPS, DUAL_SCALE_HOT_CPU_GROUP_SIZE,
};

const _: [(); DualScaleHotCpuBlock::DECODED_LEN] = [(); Q8KActivationHotCpuBlock::DECODED_LEN];

pub fn dot_dual_scale_block_q8k_hot_cpu_grouped_ref(
    weight: &DualScaleHotCpuBlock,
    activation: &Q8KActivationHotCpuBlock,
) -> f32 {
    let dx = activation.d.to_f32();
    if dx == 0.0 {
        return 0.0;
    }
    let mut out = 0.0_f32;
    for g in 0..DUAL_SCALE_HOT_CPU_GROUPS {
        let start = g * DUAL_SCALE_HOT_CPU_GROUP_SIZE;
        let end = start + DUAL_SCALE_HOT_CPU_GROUP_SIZE;
        let mut lo = 0_i32;
        let mut hi = 0_i32;
        for i in start..end {
            let p = i32::from(decode_dual_scale_signed_q4(dual_scale_hot_q4_code(
                &weight.qs, i,
            ))) * i32::from(activation.qs[i]);
            if dual_scale_hot_membership_bit(&weight.membership, i) {
                hi += p;
            } else {
                lo += p;
            }
        }
        let sl = weight.scale_lo[g].to_f32();
        if sl != 0.0 {
            out += (sl * dx) * (lo as f32);
        }
        let sh = weight.scale_hi[g].to_f32();
        if sh != 0.0 {
            out += (sh * dx) * (hi as f32);
        }
    }
    out
}

pub fn dot_dual_scale_hot_q8k_hot_cpu_grouped_ref(
    row_blocks: &[DualScaleHotCpuBlock],
    activation: &Q8KActivationHotCpuVector,
) -> f32 {
    let mut out = 0.0_f32;
    for (wb, ab) in row_blocks.iter().zip(activation.blocks().iter()) {
        out += dot_dual_scale_block_q8k_hot_cpu_grouped_ref(wb, ab);
    }
    out
}

/// Allocation-free CPU portable DualScale x Q8Hot qmatvec.
///
/// All recoverable errors are detected before the first output write.
pub fn qmatvec_dual_scale_q8k_hot_cpu_portable(
    weight: &DualScaleHotCpuTensor,
    activation: &Q8KActivationHotCpuVector,
    output: &mut [f32],
) -> Result<(), KernelError> {
    let csr = validate_and_preflight(weight, activation, output)?;
    execute_validated(weight, activation, output, csr);
    Ok(())
}

pub fn launch_qmatvec_dual_scale_q8k_hot_cpu_portable(
    weight: &DualScaleHotCpuTensor,
    activation: &Q8KActivationHotCpuVector,
    output: &mut [f32],
) -> Result<(), KernelLaunchError> {
    qmatvec_dual_scale_q8k_hot_cpu_portable(weight, activation, output)
        .map_err(KernelError::into_launch_error)
}

/// Infallible launch for binder-validated owners and views.
pub fn launch_qmatvec_dual_scale_q8k_hot_cpu_portable_prevalidated(
    bound: &BoundDualScaleQ8Hot,
    activation: &Q8KActivationHotCpuVector,
    output: &mut [f32],
) {
    let weight = bound.weight();
    debug_assert_eq!(activation.len_elements(), weight.cols());
    debug_assert_eq!(activation.block_count(), weight.blocks_per_row());
    debug_assert!(output.len() >= weight.rows());
    let csr = weight.outliers().prevalidated_rows();
    execute_validated(weight, activation, output, csr);
}

fn validate_and_preflight<'a>(
    weight: &'a DualScaleHotCpuTensor,
    activation: &Q8KActivationHotCpuVector,
    output: &[f32],
) -> Result<ValidatedOutlierCsrRows<'a>, KernelError> {
    validate_weight_shape(weight)?;
    if activation.len_elements() != weight.cols() {
        return Err(KernelError::BadActivationLength {
            expected: weight.cols(),
            actual: activation.len_elements(),
        });
    }
    if activation.block_count() != weight.blocks_per_row() {
        return Err(KernelError::BadActivationBlockCount {
            expected: weight.blocks_per_row(),
            actual: activation.block_count(),
        });
    }
    if output.len() < weight.rows() {
        return Err(KernelError::BadOutputLength {
            expected: weight.rows(),
            actual: output.len(),
        });
    }
    let o = weight.outliers();
    if o.rows() != weight.rows() || o.cols() != weight.cols() {
        return Err(KernelError::BadWeightShape);
    }
    let csr = o.validate_rows().ok_or(KernelError::BadWeight)?;
    // Prove every weight row's block slice is reachable.
    for row in 0..weight.rows() {
        weight.row_blocks(row).ok_or(KernelError::BadWeight)?;
    }
    Ok(csr)
}

fn validate_weight_shape(weight: &DualScaleHotCpuTensor) -> Result<(), KernelError> {
    let ec = weight
        .blocks_per_row()
        .checked_mul(DualScaleHotCpuBlock::DECODED_LEN)
        .ok_or(KernelError::BadWeightShape)?;
    if weight.cols() != ec {
        return Err(KernelError::BadWeightShape);
    }
    let eb = weight
        .rows()
        .checked_mul(weight.blocks_per_row())
        .ok_or(KernelError::BadWeightShape)?;
    if weight.blocks().len() != eb {
        return Err(KernelError::BadWeightBlockCount {
            expected: eb,
            actual: weight.blocks().len(),
        });
    }
    Ok(())
}

/// Infallible write phase — returns `()`.
///
/// Preflight guarantees:
///
/// - `output.len() >= weight.rows()` (validated)
/// - `weight.blocks().len() == rows * blocks_per_row` (validated)
/// - every CSR row view reachable (validated CSR token)
/// - CSR constructor guarantees all col indexes within activation range
/// - validated token immutably borrows the CSR, preventing mutation
///
/// The write phase uses only direct chunking on `weight.blocks()` to avoid
/// calling `row_blocks()` or other fallible access. All three iterators
/// (weight blocks, CSR rows, output) have matching lengths established by
/// preflight.
fn execute_validated(
    weight: &DualScaleHotCpuTensor,
    activation: &Q8KActivationHotCpuVector,
    output: &mut [f32],
    csr: ValidatedOutlierCsrRows<'_>,
) {
    let mut row_iter = weight.blocks().chunks_exact(weight.blocks_per_row());
    let mut csr_iter = csr.iter();
    let mut dst_iter = output.iter_mut();

    while let (Some(row_blocks), Some(csr_row), Some(dst)) =
        (row_iter.next(), csr_iter.next(), dst_iter.next())
    {
        let main = dot_dual_scale_hot_q8k_hot_cpu_grouped_ref(row_blocks, activation);
        let correction = outlier_row_correction_q8hot(csr_row, activation);
        *dst = main + correction;
    }
}

fn outlier_row_correction_q8hot(
    row: augur_kernel_api::OutlierCsrHotCpuRow<'_>,
    activation: &Q8KActivationHotCpuVector,
) -> f32 {
    let mut correction = 0.0_f32;
    for (&column, &delta) in row.entries() {
        let column = column as usize;
        // CSR constructor guarantees column < cols == activation.len_elements().
        // Validated token prevents mutation. Indexing is safe.
        let activation_value = decode_q8hot_at_validated(activation, column);
        correction += delta.to_f32() * activation_value;
    }
    correction
}

fn decode_q8hot_at_validated(activation: &Q8KActivationHotCpuVector, column: usize) -> f32 {
    debug_assert!(column < activation.len_elements());

    let block_index = column / Q8KActivationHotCpuBlock::DECODED_LEN;
    let lane = column % Q8KActivationHotCpuBlock::DECODED_LEN;

    debug_assert!(block_index < activation.block_count());
    debug_assert!(lane < Q8KActivationHotCpuBlock::DECODED_LEN);

    // SAFETY:
    //
    // OutlierCsrHotCpu::new validates:
    //
    //     column < csr.cols()
    //
    // validate_qmatvec_args establishes:
    //
    //     csr.cols() == weight.cols()
    //     weight.cols() == activation.len_elements()
    //
    // Q8KActivationHotCpuVector construction guarantees:
    //
    //     activation.blocks().len()
    //         == activation.len_elements()
    //            / Q8KActivationHotCpuBlock::DECODED_LEN
    //
    // Therefore block_index is within activation.blocks().
    //
    // CSR and activation are immutably borrowed throughout the validated
    // write phase, so these invariants cannot change before this access.
    let block = unsafe { activation.blocks().get_unchecked(block_index) };

    // SAFETY:
    //
    // lane is column modulo Q8KActivationHotCpuBlock::DECODED_LEN.
    // The qs field contains exactly DECODED_LEN lanes.
    let quantized = unsafe { *block.qs.get_unchecked(lane) };

    let scale = block.d.to_f32();

    if scale == 0.0 {
        return 0.0;
    }

    scale * f32::from(quantized)
}
