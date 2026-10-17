//! DualScale reference compact format.
//!
//! # Format contract
//!
//! - **compact layout**:
//!   Logical Rust reference struct `DualScaleCompactBlock`. This is not a
//!   stable ABI or serialized byte layout. Stable bytes are deferred to a
//!   future `DualScaleCompactBytesV1`.
//! - **hot layout**:
//!   deferred. Future hot layout should separate main q4 / membership / scales
//!   from matrix-level outlier CSR.
//! - **encode/decode reference**:
//!   `encode_dual_scale_block_ref` and `ReferenceDecode::decode_ref`.
//! - **repack logic**:
//!   deferred. Future repack should build DualScaleHot plus OutlierCsr.
//! - **kernel consumption**:
//!   none in M13. Future main kernel consumes signed q4 + membership + dual
//!   per-group scales; outliers are consumed by CSR correction.
//! - **precision metric**:
//!   M13 tests use max error / MSE style checks. Future model-level validation
//!   should use layer L2 / PPL.
//! - **decode cost**:
//!   per block: 256 nibble unpack, 256 membership bit reads, 256 f16 scale
//!   selects / multiplies, plus at most 2 outlier delta adds.
//!
//! # Numerical definition
//!
//! - block size: 256
//! - group size: 32
//! - groups per block: 8
//! - q code: signed q4, q ∈ [-8, 7], stored as `code = q + 8`
//! - membership bit: 0 → `scale_lo[group]`, 1 → `scale_hi[group]`
//! - decode main: `scale * q`
//! - outlier: sparse delta, so final value is `main + delta`

use crate::{FormatError, ReferenceDecode};
use augur_core::F16Bits;

pub const DUAL_SCALE_BLOCK_SIZE: usize = 256;
pub const DUAL_SCALE_GROUP_SIZE: usize = 32;
pub const DUAL_SCALE_GROUPS: usize = 8;
pub const DUAL_SCALE_QS_BYTES: usize = DUAL_SCALE_BLOCK_SIZE / 2;
pub const DUAL_SCALE_MEMBERSHIP_BYTES: usize = DUAL_SCALE_BLOCK_SIZE / 8;
pub const DUAL_SCALE_MAX_OUTLIERS: usize = 2;

// ── Block struct ──────────────────────────────────────────────────────────

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DualScaleCompactBlock {
    pub scale_lo: [F16Bits; DUAL_SCALE_GROUPS],
    pub scale_hi: [F16Bits; DUAL_SCALE_GROUPS],
    pub membership: [u8; DUAL_SCALE_MEMBERSHIP_BYTES],
    pub qs: [u8; DUAL_SCALE_QS_BYTES],
    pub outlier_count: u8,
    pub outlier_idx: [u8; DUAL_SCALE_MAX_OUTLIERS],
    pub outlier_delta: [F16Bits; DUAL_SCALE_MAX_OUTLIERS],
}

// ── Encode ────────────────────────────────────────────────────────────────

pub fn encode_dual_scale_block_ref(src: &[f32]) -> Result<DualScaleCompactBlock, FormatError> {
    if src.len() != DUAL_SCALE_BLOCK_SIZE {
        return Err(FormatError::InputLengthMismatch {
            expected: DUAL_SCALE_BLOCK_SIZE,
            actual: src.len(),
        });
    }

    for (i, &v) in src.iter().enumerate() {
        if !v.is_finite() {
            return Err(FormatError::NonFiniteInput { index: i });
        }
    }

    let outliers = select_outliers(src);
    let mut is_outlier = [false; DUAL_SCALE_BLOCK_SIZE];

    for &(idx, _) in &outliers {
        is_outlier[idx] = true;
    }

    let mut membership = [0u8; DUAL_SCALE_MEMBERSHIP_BYTES];

    for g in 0..DUAL_SCALE_GROUPS {
        let start = g * DUAL_SCALE_GROUP_SIZE;
        let end = start + DUAL_SCALE_GROUP_SIZE;

        let mut sum_abs = 0.0f64;
        let mut count = 0usize;

        for i in start..end {
            if !is_outlier[i] {
                sum_abs += f64::from(src[i].abs());
                count += 1;
            }
        }

        let threshold = if count == 0 {
            0.0
        } else {
            sum_abs / count as f64
        };

        for i in start..end {
            if !is_outlier[i] && f64::from(src[i].abs()) > threshold {
                set_membership_bit(&mut membership, i, true);
            }
        }
    }

    let mut scale_lo = [F16Bits::ZERO; DUAL_SCALE_GROUPS];
    let mut scale_hi = [F16Bits::ZERO; DUAL_SCALE_GROUPS];

    for g in 0..DUAL_SCALE_GROUPS {
        scale_lo[g] = finite_f16(compute_group_scale(src, &membership, &is_outlier, g, false))?;
        scale_hi[g] = finite_f16(compute_group_scale(src, &membership, &is_outlier, g, true))?;
    }

    let mut qs = [0u8; DUAL_SCALE_QS_BYTES];

    for i in 0..DUAL_SCALE_BLOCK_SIZE {
        let g = i / DUAL_SCALE_GROUP_SIZE;
        let scale = if get_membership_bit(&membership, i) {
            scale_hi[g].to_f32()
        } else {
            scale_lo[g].to_f32()
        };

        let q = quantize_signed_q4(src[i], scale);
        set_q4_code(&mut qs, i, encode_signed_q4(q));
    }

    let mut outlier_idx = [0u8; DUAL_SCALE_MAX_OUTLIERS];
    let mut outlier_delta = [F16Bits::ZERO; DUAL_SCALE_MAX_OUTLIERS];

    for (slot, &(idx, _)) in outliers.iter().enumerate() {
        let main = decode_main_at(idx, &scale_lo, &scale_hi, &membership, &qs);
        outlier_idx[slot] = idx as u8;
        outlier_delta[slot] = finite_f16(src[idx] - main)?;
    }

    Ok(DualScaleCompactBlock {
        scale_lo,
        scale_hi,
        membership,
        qs,
        outlier_count: outliers.len() as u8,
        outlier_idx,
        outlier_delta,
    })
}

// ── Decode ────────────────────────────────────────────────────────────────

impl ReferenceDecode for DualScaleCompactBlock {
    type Elem = f32;

    fn decoded_len(&self) -> usize {
        DUAL_SCALE_BLOCK_SIZE
    }

    fn decode_ref(&self, dst: &mut [Self::Elem]) -> Result<(), FormatError> {
        if dst.len() < DUAL_SCALE_BLOCK_SIZE {
            return Err(FormatError::OutputTooSmall {
                expected: DUAL_SCALE_BLOCK_SIZE,
                actual: dst.len(),
            });
        }

        let count = validate_dual_scale_compact_block(self)?;

        // Main decode pass.
        for i in 0..DUAL_SCALE_BLOCK_SIZE {
            dst[i] = decode_main_at(
                i,
                &self.scale_lo,
                &self.scale_hi,
                &self.membership,
                &self.qs,
            );
        }

        // Sparse outlier delta correction.
        for k in 0..count {
            let idx = self.outlier_idx[k] as usize;
            dst[idx] += self.outlier_delta[k].to_f32();
        }

        Ok(())
    }
}

/// Decodes only the DualScale main path, without applying outlier deltas.
///
/// This is a reference decomposition helper for CSR correction tests / repack
/// validation. It is intentionally not re-exported from the crate root.
pub fn decode_dual_scale_main_block_ref(
    block: &DualScaleCompactBlock,
    dst: &mut [f32],
) -> Result<(), FormatError> {
    if dst.len() < DUAL_SCALE_BLOCK_SIZE {
        return Err(FormatError::OutputTooSmall {
            expected: DUAL_SCALE_BLOCK_SIZE,
            actual: dst.len(),
        });
    }

    validate_dual_scale_compact_block(block)?;

    for i in 0..DUAL_SCALE_BLOCK_SIZE {
        dst[i] = decode_main_at(
            i,
            &block.scale_lo,
            &block.scale_hi,
            &block.membership,
            &block.qs,
        );
    }

    Ok(())
}

// ── Membership bit helpers ────────────────────────────────────────────────

pub fn get_membership_bit(membership: &[u8; DUAL_SCALE_MEMBERSHIP_BYTES], idx: usize) -> bool {
    debug_assert!(idx < DUAL_SCALE_BLOCK_SIZE);
    let byte = membership[idx / 8];
    let bit = idx % 8;
    ((byte >> bit) & 1) != 0
}

pub fn set_membership_bit(
    membership: &mut [u8; DUAL_SCALE_MEMBERSHIP_BYTES],
    idx: usize,
    value: bool,
) {
    debug_assert!(idx < DUAL_SCALE_BLOCK_SIZE);
    let byte = &mut membership[idx / 8];
    let mask = 1u8 << (idx % 8);

    if value {
        *byte |= mask;
    } else {
        *byte &= !mask;
    }
}

// ── Q4 nibble helpers ─────────────────────────────────────────────────────

pub fn get_q4_code(qs: &[u8; DUAL_SCALE_QS_BYTES], idx: usize) -> u8 {
    debug_assert!(idx < DUAL_SCALE_BLOCK_SIZE);
    let byte = qs[idx / 2];

    if idx & 1 == 0 {
        byte & 0x0f
    } else {
        byte >> 4
    }
}

pub fn set_q4_code(qs: &mut [u8; DUAL_SCALE_QS_BYTES], idx: usize, code: u8) {
    debug_assert!(idx < DUAL_SCALE_BLOCK_SIZE);
    debug_assert!(code <= 0x0f);

    let byte = &mut qs[idx / 2];

    if idx & 1 == 0 {
        *byte = (*byte & 0xf0) | (code & 0x0f);
    } else {
        *byte = (*byte & 0x0f) | ((code & 0x0f) << 4);
    }
}

// ── Signed q4 encode/decode ───────────────────────────────────────────────

pub fn encode_signed_q4(q: i8) -> u8 {
    debug_assert!((-8..=7).contains(&q));
    (q + 8) as u8
}

pub fn decode_signed_q4(code: u8) -> i8 {
    debug_assert!(code <= 0x0f);
    code as i8 - 8
}

/// Rust `f32::round` rounds half-way cases away from zero.
/// This is part of the M13 reference quantization contract.
fn quantize_signed_q4(value: f32, scale: f32) -> i8 {
    if scale == 0.0 {
        return 0;
    }
    (value / scale).round().clamp(-8.0, 7.0) as i8
}

// ── Internal helpers ──────────────────────────────────────────────────────

fn compute_group_scale(
    src: &[f32],
    membership: &[u8; DUAL_SCALE_MEMBERSHIP_BYTES],
    is_outlier: &[bool; DUAL_SCALE_BLOCK_SIZE],
    group: usize,
    hi: bool,
) -> f32 {
    let start = group * DUAL_SCALE_GROUP_SIZE;
    let end = start + DUAL_SCALE_GROUP_SIZE;

    let mut max_pos = 0.0f32;
    let mut max_neg = 0.0f32;

    for i in start..end {
        if is_outlier[i] || get_membership_bit(membership, i) != hi {
            continue;
        }

        let v = src[i];
        if v >= 0.0 {
            max_pos = max_pos.max(v);
        } else {
            max_neg = max_neg.max(-v);
        }
    }

    (max_pos / 7.0).max(max_neg / 8.0)
}

pub fn validate_dual_scale_compact_block(
    block: &DualScaleCompactBlock,
) -> Result<usize, FormatError> {
    let count = block.outlier_count as usize;
    if count > DUAL_SCALE_MAX_OUTLIERS {
        return Err(FormatError::BadBlock);
    }
    if block
        .scale_lo
        .iter()
        .chain(&block.scale_hi)
        .any(|value| !value.to_f32().is_finite())
        || block.outlier_delta[..count]
            .iter()
            .any(|value| !value.to_f32().is_finite())
    {
        return Err(FormatError::BadBlock);
    }
    for slot in 0..count {
        if block.outlier_idx[..slot].contains(&block.outlier_idx[slot]) {
            return Err(FormatError::BadBlock);
        }
    }
    Ok(count)
}

fn finite_f16(value: f32) -> Result<F16Bits, FormatError> {
    let encoded = F16Bits::from_f32(value);
    if encoded.to_f32().is_finite() {
        Ok(encoded)
    } else {
        Err(FormatError::NonFiniteValue)
    }
}

fn decode_main_at(
    idx: usize,
    scale_lo: &[F16Bits; DUAL_SCALE_GROUPS],
    scale_hi: &[F16Bits; DUAL_SCALE_GROUPS],
    membership: &[u8; DUAL_SCALE_MEMBERSHIP_BYTES],
    qs: &[u8; DUAL_SCALE_QS_BYTES],
) -> f32 {
    let group = idx / DUAL_SCALE_GROUP_SIZE;
    let scale = if get_membership_bit(membership, idx) {
        scale_hi[group].to_f32()
    } else {
        scale_lo[group].to_f32()
    };

    // Canonical zero when scale is zero, regardless of q code.
    if scale == 0.0 {
        return 0.0;
    }

    let q = decode_signed_q4(get_q4_code(qs, idx)) as f32;
    scale * q
}

fn select_outliers(src: &[f32]) -> Vec<(usize, f32)> {
    let mean_abs =
        src.iter().map(|v| f64::from(v.abs())).sum::<f64>() / DUAL_SCALE_BLOCK_SIZE as f64;
    let threshold = 2.5 * mean_abs;

    let mut candidates: Vec<(usize, f32)> = src
        .iter()
        .enumerate()
        .filter_map(|(idx, &v)| {
            let abs = v.abs();
            if f64::from(abs) > threshold && abs > 0.0 {
                Some((idx, abs))
            } else {
                None
            }
        })
        .collect();

    candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    candidates.truncate(DUAL_SCALE_MAX_OUTLIERS);
    candidates
}

// ── Module-internal unit tests ────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_signed_q4_ties_round_away_from_zero() {
        assert_eq!(quantize_signed_q4(0.5, 1.0), 1);
        assert_eq!(quantize_signed_q4(-0.5, 1.0), -1);
        assert_eq!(quantize_signed_q4(1.5, 1.0), 2);
        assert_eq!(quantize_signed_q4(-1.5, 1.0), -2);
    }

    #[test]
    fn quantize_signed_q4_clamps_to_range() {
        assert_eq!(quantize_signed_q4(100.0, 1.0), 7);
        assert_eq!(quantize_signed_q4(-100.0, 1.0), -8);
    }

    #[test]
    fn quantize_signed_q4_zero_scale_returns_zero() {
        assert_eq!(quantize_signed_q4(5.0, 0.0), 0);
        assert_eq!(quantize_signed_q4(-5.0, 0.0), 0);
    }

    #[test]
    fn encode_rejects_finite_values_not_representable_in_f16() {
        let src = [f32::MAX; DUAL_SCALE_BLOCK_SIZE];
        assert_eq!(
            encode_dual_scale_block_ref(&src),
            Err(FormatError::NonFiniteValue)
        );
    }

    #[test]
    fn decode_preflight_rejects_nonfinite_scale_delta_and_duplicate_outlier() {
        let mut block = encode_dual_scale_block_ref(&[1.0; DUAL_SCALE_BLOCK_SIZE]).unwrap();
        let mut dst = [7.0; DUAL_SCALE_BLOCK_SIZE];
        block.scale_lo[0] = F16Bits::from_bits(0x7c00);
        assert_eq!(block.decode_ref(&mut dst), Err(FormatError::BadBlock));
        assert_eq!(dst, [7.0; DUAL_SCALE_BLOCK_SIZE]);

        block.scale_lo[0] = F16Bits::ZERO;
        block.outlier_count = 2;
        block.outlier_idx = [3, 3];
        block.outlier_delta[0] = F16Bits::from_bits(0x7c00);
        assert_eq!(block.decode_ref(&mut dst), Err(FormatError::BadBlock));
        assert_eq!(dst, [7.0; DUAL_SCALE_BLOCK_SIZE]);
    }
}
