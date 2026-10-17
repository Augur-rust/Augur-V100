use augur_kernel_api::{KernelLaunchError, Q4KHotCpuBlock, Q4KHotCpuTensor};

use crate::KernelError;

// ── Internal decode helper (replaces removed Q4KHotCpuBlock::decode_ref) ──

fn decode_q4k_hot_block_f32(block: &Q4KHotCpuBlock, dst: &mut [f32]) {
    debug_assert!(dst.len() >= Q4KHotCpuBlock::DECODED_LEN);

    let d = block.d.to_f32();
    let dmin = block.dmin.to_f32();

    for (i, out) in dst.iter_mut().take(Q4KHotCpuBlock::DECODED_LEN).enumerate() {
        let group = i / 32;
        let q = block.quant_at(i);
        *out = d * (block.scale[group] as f32) * (q as f32) - dmin * (block.min[group] as f32);
    }
}

// ── Per-block helpers (M3–M6) ────────────────────────────────────────────

pub fn decode_q4k_hot_f32(weight: &[Q4KHotCpuBlock], dst: &mut [f32]) -> Result<(), KernelError> {
    let per = Q4KHotCpuBlock::DECODED_LEN;
    let expected = weight
        .len()
        .checked_mul(per)
        .ok_or(KernelError::BadOutputLength {
            expected: usize::MAX,
            actual: dst.len(),
        })?;
    if dst.len() < expected {
        return Err(KernelError::BadOutputLength {
            expected,
            actual: dst.len(),
        });
    }

    for (bi, block) in weight.iter().enumerate() {
        decode_q4k_hot_block_f32(block, &mut dst[bi * per..(bi + 1) * per]);
    }
    Ok(())
}

pub fn dot_q4k_hot_f32(weight: &[Q4KHotCpuBlock], activation: &[f32]) -> Result<f32, KernelError> {
    let per = Q4KHotCpuBlock::DECODED_LEN;
    let expected = weight
        .len()
        .checked_mul(per)
        .ok_or(KernelError::BadActivationLength {
            expected: usize::MAX,
            actual: activation.len(),
        })?;
    if activation.len() != expected {
        return Err(KernelError::BadActivationLength {
            expected,
            actual: activation.len(),
        });
    }

    let mut acc = 0.0f32;
    let mut decoded = [0.0f32; 256];

    for (bi, block) in weight.iter().enumerate() {
        decode_q4k_hot_block_f32(block, &mut decoded);
        let start = bi * per;
        for i in 0..per {
            acc += decoded[i] * activation[start + i];
        }
    }
    Ok(acc)
}

// ── M8 internal unchecked row-dot helper ─────────────────────────────────

/// Internal M8 helper — direct decode+dot **without** length checks.
///
/// # Preconditions (caller MUST guarantee)
///
/// - `activation.len() == weight.len() * Q4KHotCpuBlock::DECODED_LEN`
/// - `weight` is a contiguous row slice from a validated [`Q4KHotCpuTensor`]
/// - Caller has already validated qmatmul args via [`validate_qmatmul_args`]
///
/// # Bit-exact contract
///
/// The loop order and arithmetic order intentionally match
/// [`dot_q4k_hot_f32_direct`] to preserve bit-exact results. Do **not**
/// reorder loops, fuse groups, or switch to `f32::mul_add` without updating
/// the bit-exact test suite.
///
/// This function is **private** — it must never be exposed to external
/// crates. Public entry points that need unchecked performance should
/// validate once at the entry boundary and then call this helper.
fn dot_q4k_hot_f32_direct_unchecked(weight: &[Q4KHotCpuBlock], activation: &[f32]) -> f32 {
    debug_assert_eq!(activation.len(), weight.len() * Q4KHotCpuBlock::DECODED_LEN);

    let mut acc = 0.0_f32;

    for (block_idx, block) in weight.iter().enumerate() {
        let d = block.d.to_f32();
        let dmin = block.dmin.to_f32();
        let start = block_idx * Q4KHotCpuBlock::DECODED_LEN;
        let act = &activation[start..start + Q4KHotCpuBlock::DECODED_LEN];

        for i in 0..Q4KHotCpuBlock::DECODED_LEN {
            let group = i / 32;
            let q = block.quant_at(i);

            // Keep operation order bit-exact with Q4KHotCpuBlock::decode_ref
            // and dot_q4k_hot_f32_direct.
            let w = d * (block.scale[group] as f32) * (q as f32) - dmin * (block.min[group] as f32);

            acc += w * act[i];
        }
    }

    acc
}

/// Direct decode+dot — bit-exact with [`dot_q4k_hot_f32`].
///
/// Eliminates the per-block `decoded[256]` temporary buffer while keeping
/// the exact same element order and f32 expression order.
/// Must remain bit-exact with `dot_q4k_hot_f32` at `assert_eq!` level.
///
/// M8: delegates to `dot_q4k_hot_f32_direct_unchecked` after public
/// entry-point validation.
pub fn dot_q4k_hot_f32_direct(
    weight: &[Q4KHotCpuBlock],
    activation: &[f32],
) -> Result<f32, KernelError> {
    let expected = weight
        .len()
        .checked_mul(Q4KHotCpuBlock::DECODED_LEN)
        .ok_or(KernelError::BadActivationLength {
            expected: usize::MAX,
            actual: activation.len(),
        })?;

    if activation.len() != expected {
        return Err(KernelError::BadActivationLength {
            expected,
            actual: activation.len(),
        });
    }

    Ok(dot_q4k_hot_f32_direct_unchecked(weight, activation))
}

// ── Shared weight-shape validation ───────────────────────────────────────

pub(crate) fn validate_q4k_weight_shape(weight: &Q4KHotCpuTensor) -> Result<(), KernelError> {
    let expected_cols = weight
        .blocks_per_row()
        .checked_mul(Q4KHotCpuBlock::DECODED_LEN)
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

// ── M7 matrix-level fused portable qmatmul ──────────────────────────────

#[inline]
pub fn silu_f32(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Validate input/out/bias shapes against a [`Q4KHotCpuTensor`].
///
/// `input.len()` must equal `weight.cols()` exactly.
/// `out.len()` and `bias.len()` must be ≥ `weight.rows()`.
/// Also checks internal tensor invariants as a defence-in-depth measure.
pub(crate) fn validate_qmatmul_args(
    weight: &Q4KHotCpuTensor,
    input: &[f32],
    out: &[f32],
    bias: Option<&[f32]>,
) -> Result<(), KernelError> {
    validate_q4k_weight_shape(weight)?;

    if input.len() != weight.cols() {
        return Err(KernelError::BadActivationLength {
            expected: weight.cols(),
            actual: input.len(),
        });
    }

    if out.len() < weight.rows() {
        return Err(KernelError::BadOutputLength {
            expected: weight.rows(),
            actual: out.len(),
        });
    }

    if let Some(b) = bias {
        if b.len() < weight.rows() {
            return Err(KernelError::BadBiasLength {
                expected: weight.rows(),
                actual: b.len(),
            });
        }
    }

    Ok(())
}

// ── Public fused qmatmul entry points (M8: row loops use unchecked) ─────

pub fn qmatmul_q4k_f32_portable(
    weight: &Q4KHotCpuTensor,
    input: &[f32],
    out: &mut [f32],
) -> Result<(), KernelError> {
    validate_qmatmul_args(weight, input, out, None)?;

    for row in 0..weight.rows() {
        let row_blocks = weight.row_blocks(row).ok_or(KernelError::BadWeight)?;
        out[row] = dot_q4k_hot_f32_direct_unchecked(row_blocks, input);
    }

    Ok(())
}

/// Tiled Q4K x f32 prefill GEMM over row-major activations and outputs.
///
/// This is a distinct matrix kernel: each decoded weight value is reused
/// across an activation tile. It does not invoke or wrap the decode GEMV
/// entry point.
pub fn qmatmul_q4k_f32_prefill_portable(
    weight: &Q4KHotCpuTensor,
    input: &[f32],
    m: usize,
    out: &mut [f32],
) -> Result<(), KernelError> {
    validate_q4k_weight_shape(weight)?;
    let expected_input = m
        .checked_mul(weight.cols())
        .ok_or(KernelError::BadActivationLength {
            expected: usize::MAX,
            actual: input.len(),
        })?;
    let expected_output = m
        .checked_mul(weight.rows())
        .ok_or(KernelError::BadOutputLength {
            expected: usize::MAX,
            actual: out.len(),
        })?;
    if m == 0 || input.len() != expected_input {
        return Err(KernelError::BadActivationLength {
            expected: expected_input,
            actual: input.len(),
        });
    }
    if out.len() < expected_output {
        return Err(KernelError::BadOutputLength {
            expected: expected_output,
            actual: out.len(),
        });
    }

    const TILE_M: usize = 4;
    for m0 in (0..m).step_by(TILE_M) {
        let tile_m = (m - m0).min(TILE_M);
        for row in 0..weight.rows() {
            let row_blocks = weight.row_blocks(row).ok_or(KernelError::BadWeight)?;
            let mut acc = [0.0_f32; TILE_M];
            for (block_idx, block) in row_blocks.iter().enumerate() {
                let d = block.d.to_f32();
                let dmin = block.dmin.to_f32();
                let k0 = block_idx * Q4KHotCpuBlock::DECODED_LEN;
                for lane in 0..Q4KHotCpuBlock::DECODED_LEN {
                    let group = lane / 32;
                    let q = block.quant_at(lane);
                    let w = d * (block.scale[group] as f32) * (q as f32)
                        - dmin * (block.min[group] as f32);
                    for tile_row in 0..tile_m {
                        acc[tile_row] += w * input[(m0 + tile_row) * weight.cols() + k0 + lane];
                    }
                }
            }
            for tile_row in 0..tile_m {
                out[(m0 + tile_row) * weight.rows() + row] = acc[tile_row];
            }
        }
    }
    Ok(())
}

pub fn qmatmul_q4k_f32_bias_portable(
    weight: &Q4KHotCpuTensor,
    input: &[f32],
    bias: &[f32],
    out: &mut [f32],
) -> Result<(), KernelError> {
    validate_qmatmul_args(weight, input, out, Some(bias))?;

    for row in 0..weight.rows() {
        let row_blocks = weight.row_blocks(row).ok_or(KernelError::BadWeight)?;
        out[row] = dot_q4k_hot_f32_direct_unchecked(row_blocks, input) + bias[row];
    }

    Ok(())
}

pub fn qmatmul_q4k_f32_bias_silu_portable(
    weight: &Q4KHotCpuTensor,
    input: &[f32],
    bias: &[f32],
    out: &mut [f32],
) -> Result<(), KernelError> {
    validate_qmatmul_args(weight, input, out, Some(bias))?;

    for row in 0..weight.rows() {
        let row_blocks = weight.row_blocks(row).ok_or(KernelError::BadWeight)?;
        let x = dot_q4k_hot_f32_direct_unchecked(row_blocks, input) + bias[row];
        out[row] = silu_f32(x);
    }

    Ok(())
}

// ── Bound launch wrappers (Batch 6) ───────────────────────────────────────

/// ExecutionPlan launch wrapper for the portable plain Q4K × f32 kernel.
pub fn launch_qmatmul_q4k_f32_portable(
    weight: &Q4KHotCpuTensor,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), KernelLaunchError> {
    qmatmul_q4k_f32_portable(weight, input, output).map_err(KernelError::into_launch_error)
}

pub fn launch_qmatmul_q4k_f32_prefill_portable(
    weight: &Q4KHotCpuTensor,
    input: &[f32],
    m: usize,
    output: &mut [f32],
) -> Result<(), KernelLaunchError> {
    qmatmul_q4k_f32_prefill_portable(weight, input, m, output)
        .map_err(KernelError::into_launch_error)
}

/// ExecutionPlan launch wrapper for portable Q4K × f32 + Bias.
pub fn launch_qmatmul_q4k_f32_bias_portable(
    weight: &Q4KHotCpuTensor,
    input: &[f32],
    bias: &[f32],
    output: &mut [f32],
) -> Result<(), KernelLaunchError> {
    qmatmul_q4k_f32_bias_portable(weight, input, bias, output)
        .map_err(KernelError::into_launch_error)
}

/// ExecutionPlan launch wrapper for portable Q4K × f32 + Bias + SiLU.
pub fn launch_qmatmul_q4k_f32_bias_silu_portable(
    weight: &Q4KHotCpuTensor,
    input: &[f32],
    bias: &[f32],
    output: &mut [f32],
) -> Result<(), KernelLaunchError> {
    qmatmul_q4k_f32_bias_silu_portable(weight, input, bias, output)
        .map_err(KernelError::into_launch_error)
}
