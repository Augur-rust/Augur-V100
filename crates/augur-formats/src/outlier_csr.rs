//! Reference CSR representation for sparse outlier deltas.
//!
//! # M14 scope
//!
//! - reference-side sparse representation only
//! - stores outlier deltas, not replacement values
//! - no backend hot layout
//! - no kernel execution
//! - no device pointers
//!
//! # Mathematical contract
//!
//! ```text
//! W = W_main + W_outlier_delta
//! ```
//!
//! Correction semantics:
//! ```text
//! out[row] += Σ delta[p] × x[col_idx[p]]
//! ```

use crate::{
    validate_dual_scale_compact_block, DualScaleCompactBlock, FormatError, DUAL_SCALE_BLOCK_SIZE,
};
use augur_core::F16Bits;

// ── CSR struct ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OutlierCsrRef {
    pub rows: usize,
    pub cols: usize,
    pub row_ptr: Vec<u32>,
    pub col_idx: Vec<u32>,
    pub delta: Vec<F16Bits>,
}

// ── Extraction ────────────────────────────────────────────────────────────

pub fn extract_dual_scale_outlier_csr_ref(
    rows: usize,
    cols: usize,
    blocks: &[DualScaleCompactBlock],
) -> Result<OutlierCsrRef, FormatError> {
    if cols % DUAL_SCALE_BLOCK_SIZE != 0 {
        return Err(FormatError::BadLength);
    }

    let blocks_per_row = cols / DUAL_SCALE_BLOCK_SIZE;
    let expected_blocks = rows
        .checked_mul(blocks_per_row)
        .ok_or(FormatError::BadLength)?;

    if blocks.len() != expected_blocks {
        return Err(FormatError::InputLengthMismatch {
            expected: expected_blocks,
            actual: blocks.len(),
        });
    }

    let row_ptr_len = rows.checked_add(1).ok_or(FormatError::BadLength)?;
    let mut row_ptr = Vec::with_capacity(row_ptr_len);
    let mut col_idx = Vec::new();
    let mut delta = Vec::new();

    row_ptr.push(0);

    for row in 0..rows {
        for block_col in 0..blocks_per_row {
            let block_index = row
                .checked_mul(blocks_per_row)
                .and_then(|base| base.checked_add(block_col))
                .ok_or(FormatError::BadLength)?;

            let block = &blocks[block_index];

            let count = validate_dual_scale_compact_block(block)?;

            for k in 0..count {
                let local_idx = block.outlier_idx[k] as usize;

                if local_idx >= DUAL_SCALE_BLOCK_SIZE {
                    return Err(FormatError::BadBlock);
                }

                let col = block_col
                    .checked_mul(DUAL_SCALE_BLOCK_SIZE)
                    .and_then(|base| base.checked_add(local_idx))
                    .ok_or(FormatError::BadLength)?;

                if col >= cols {
                    return Err(FormatError::BadBlock);
                }

                if col_idx.len() == u32::MAX as usize {
                    return Err(FormatError::BadBlock);
                }

                col_idx.push(col as u32);
                delta.push(block.outlier_delta[k]);
            }
        }

        row_ptr.push(u32::try_from(col_idx.len()).map_err(|_| FormatError::BadBlock)?);
    }

    Ok(OutlierCsrRef {
        rows,
        cols,
        row_ptr,
        col_idx,
        delta,
    })
}

// ── In-place correction ───────────────────────────────────────────────────

/// Applies sparse outlier correction in-place.
///
/// Semantics:
/// ```text
/// out[row] += Σ delta[p] × x[col_idx[p]]
/// ```
///
/// If validation fails, `out` is left unchanged.
///
/// This function does not zero or overwrite `out`; callers are expected to
/// initialize it with the main-path matvec result.
pub fn add_outlier_csr_correction_ref(
    csr: &OutlierCsrRef,
    x: &[f32],
    out: &mut [f32],
) -> Result<(), FormatError> {
    validate_outlier_csr_ref(csr)?;

    if x.len() != csr.cols {
        return Err(FormatError::InputLengthMismatch {
            expected: csr.cols,
            actual: x.len(),
        });
    }

    if out.len() < csr.rows {
        return Err(FormatError::OutputTooSmall {
            expected: csr.rows,
            actual: out.len(),
        });
    }

    // Compute correction into a temporary buffer so that `out` is left
    // unchanged on error (reference-path strong guarantee).
    let mut correction = vec![0.0f32; csr.rows];

    for (row, corr) in correction.iter_mut().enumerate() {
        let start = csr.row_ptr[row] as usize;
        let end = csr.row_ptr[row + 1] as usize;

        for p in start..end {
            let col = csr.col_idx[p] as usize;
            *corr += csr.delta[p].to_f32() * x[col];
        }
    }

    for row in 0..csr.rows {
        out[row] += correction[row];
    }

    Ok(())
}

// ── Validation ────────────────────────────────────────────────────────────

pub fn validate_outlier_csr_ref(csr: &OutlierCsrRef) -> Result<(), FormatError> {
    if csr.row_ptr.len() != csr.rows.checked_add(1).ok_or(FormatError::BadBlock)? {
        return Err(FormatError::BadBlock);
    }

    if csr.col_idx.len() != csr.delta.len() {
        return Err(FormatError::BadBlock);
    }

    let nnz = u32::try_from(csr.col_idx.len()).map_err(|_| FormatError::BadBlock)?;

    if csr.row_ptr.first().copied() != Some(0) {
        return Err(FormatError::BadBlock);
    }

    if csr.row_ptr.last().copied() != Some(nnz) {
        return Err(FormatError::BadBlock);
    }

    for pair in csr.row_ptr.windows(2) {
        if pair[0] > pair[1] {
            return Err(FormatError::BadBlock);
        }
    }

    for &col in &csr.col_idx {
        if col as usize >= csr.cols {
            return Err(FormatError::BadBlock);
        }
    }

    Ok(())
}
