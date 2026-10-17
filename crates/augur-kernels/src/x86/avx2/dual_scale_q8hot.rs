//! AVX2 DualScale × Q8Hot block and row-dot primitives.
//!
//! M28-B scope:
//!
//! - consume the existing `DUAL_SCALE_HOT_CPU_PORTABLE_V1` weight layout;
//! - consume the existing `Q8K_ACTIVATION_HOT_CPU_PORTABLE_V1` activation;
//! - compute one block or one complete row main-path dot;
//! - use separate lo/hi integer accumulators;
//! - perform no allocation;
//! - perform no registry lookup;
//! - perform no execution routing;
//! - perform no CSR correction.
//!
//! This module deliberately does not contain:
//!
//! - a complete checked qmatvec launch;
//! - a static kernel registration;
//! - compiler/lowering integration;
//! - ExecutionPlan integration;
//! - Compact/reference Q8K consumers.
//!
//! # Integer arithmetic
//!
//! For every 32-lane group:
//!
//! ```text
//! q4 ∈ [-8, 7]
//! q8 ∈ [-128, 127]
//!
//! lo_acc = Σ membership=lo (q4 × q8)
//! hi_acc = Σ membership=hi (q4 × q8)
//! ```
//!
//! Each i16 lane product satisfies:
//!
//! ```text
//! abs(q4 × q8) <= 8 × 128 = 1,024
//! ```
//!
//! Each lo/hi i32 accumulator satisfies:
//!
//! ```text
//! abs(acc) <= 32 × 1,024 = 32,768
//! ```
//!
//! No i16 multiplication or i32 accumulation can overflow.
//!
//! # Numerical contract
//!
//! M28-B is initially admitted as `NumericallyClose` against the existing
//! portable DualScale × Q8Hot grouped primitive.
//!
//! Integer products and integer accumulation are exact. Floating-point
//! differences may still arise from compiler code generation or scale-apply
//! contraction. BitExact may be promoted only after the complete M28 test and
//! assembly matrix passes.

use core::arch::x86_64::{
    __m128i, __m256i, _mm256_add_epi32, _mm256_and_si256, _mm256_andnot_si256,
    _mm256_castsi256_si128, _mm256_cmpeq_epi16, _mm256_cvtepi8_epi16, _mm256_extracti128_si256,
    _mm256_madd_epi16, _mm256_set1_epi16, _mm256_setr_epi16, _mm256_setzero_si256,
    _mm256_xor_si256, _mm_add_epi32, _mm_and_si128, _mm_cvtsi128_si32, _mm_hadd_epi32,
    _mm_loadu_si128, _mm_set1_epi8, _mm_srli_epi16, _mm_sub_epi8, _mm_unpackhi_epi8,
    _mm_unpacklo_epi8,
};

use crate::KernelError;

use augur_kernel_api::{
    BoundDualScaleQ8Hot, DualScaleHotCpuBlock, DualScaleHotCpuTensor, KernelLaunchError,
    OutlierCsrHotCpuRow, Q8KActivationHotCpuBlock, Q8KActivationHotCpuVector,
    ValidatedOutlierCsrRows, DUAL_SCALE_HOT_CPU_GROUPS, DUAL_SCALE_HOT_CPU_GROUP_SIZE,
};

const PACKED_Q4_BYTES_PER_GROUP: usize = DUAL_SCALE_HOT_CPU_GROUP_SIZE / 2;
const MEMBERSHIP_BYTES_PER_GROUP: usize = DUAL_SCALE_HOT_CPU_GROUP_SIZE / 8;
const LANES_PER_HALF: usize = 16;

const _: [(); DualScaleHotCpuBlock::DECODED_LEN] = [(); Q8KActivationHotCpuBlock::DECODED_LEN];

/// Expands 16 packed q4 bytes into two logical 16-lane unsigned-code vectors.
///
/// Input byte order:
///
/// ```text
/// byte[j] = code[2*j] | (code[2*j + 1] << 4)
/// ```
///
/// Output:
///
/// ```text
/// first  = code[0..16]
/// second = code[16..32]
/// ```
///
/// Each output lane is in `[0, 15]`.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn expand_q4_group_codes(packed: __m128i) -> (__m128i, __m128i) {
    let nibble_mask = _mm_set1_epi8(0x0f);

    let low_nibbles = _mm_and_si128(packed, nibble_mask);
    let high_nibbles = _mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask);

    let first_16 = _mm_unpacklo_epi8(low_nibbles, high_nibbles);
    let second_16 = _mm_unpackhi_epi8(low_nibbles, high_nibbles);

    (first_16, second_16)
}

/// Converts 16 unsigned q4 codes `[0, 15]` into signed q4 i16 lanes `[-8, 7]`.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn signed_q4_i16(codes: __m128i) -> __m256i {
    let signed_q4_i8 = _mm_sub_epi8(codes, _mm_set1_epi8(8));
    _mm256_cvtepi8_epi16(signed_q4_i8)
}

/// Expands 16 membership bits into 16 i16 masks.
///
/// Bit order is the frozen LSB-first protocol:
///
/// ```text
/// bit 0  → lane 0
/// bit 1  → lane 1
/// ...
/// bit 15 → lane 15
/// ```
///
/// Returned lanes are:
///
/// ```text
/// membership = 0 → 0x0000
/// membership = 1 → 0xffff
/// ```
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn membership_mask_i16(bits: u16) -> __m256i {
    let broadcast = _mm256_set1_epi16(bits as i16);

    #[allow(overflowing_literals)]
    let lane_bits = _mm256_setr_epi16(
        0x0001,
        0x0002,
        0x0004,
        0x0008,
        0x0010,
        0x0020,
        0x0040,
        0x0080,
        0x0100,
        0x0200,
        0x0400,
        0x0800,
        0x1000,
        0x2000,
        0x4000,
        0x8000_u16 as i16,
    );

    let selected = _mm256_and_si256(broadcast, lane_bits);
    let zero = _mm256_setzero_si256();

    // cmpeq produces all-one lanes where the membership bit is zero.
    let lo_mask = _mm256_cmpeq_epi16(selected, zero);

    // Invert to obtain all-one lanes where the membership bit is one.
    _mm256_xor_si256(lo_mask, _mm256_set1_epi16(-1))
}

/// Accumulates 16 lanes into separate lo/hi i32 vectors.
///
/// Each returned i32 lane contains the sum of two adjacent i16 products.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn accumulate_sixteen(
    q4_codes: __m128i,
    q8_values: __m128i,
    membership_bits: u16,
    lo_acc: __m256i,
    hi_acc: __m256i,
) -> (__m256i, __m256i) {
    let q4 = unsafe { signed_q4_i16(q4_codes) };
    let q8 = _mm256_cvtepi8_epi16(q8_values);

    let hi_mask = unsafe { membership_mask_i16(membership_bits) };

    let q8_hi = _mm256_and_si256(q8, hi_mask);
    let q8_lo = _mm256_andnot_si256(hi_mask, q8);

    let lo_products = _mm256_madd_epi16(q4, q8_lo);
    let hi_products = _mm256_madd_epi16(q4, q8_hi);

    (
        _mm256_add_epi32(lo_acc, lo_products),
        _mm256_add_epi32(hi_acc, hi_products),
    )
}

/// Horizontally reduces eight i32 lanes.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_i32x8(value: __m256i) -> i32 {
    let low = _mm256_castsi256_si128(value);
    let high = _mm256_extracti128_si256(value, 1);

    let sum_128 = _mm_add_epi32(low, high);
    let pair_sum = _mm_hadd_epi32(sum_128, sum_128);
    let total = _mm_hadd_epi32(pair_sum, pair_sum);

    _mm_cvtsi128_si32(total)
}

/// Loads one little-endian two-byte membership half.
#[inline]
unsafe fn load_membership_u16(pointer: *const u8) -> u16 {
    u16::from_le(unsafe { core::ptr::read_unaligned(pointer.cast::<u16>()) })
}

/// Computes one DualScale Hot × Q8Hot block main-path dot.
///
/// This function performs no allocation and no recoverable validation.
///
/// # Safety
///
/// The caller must guarantee:
///
/// - the current CPU supports AVX2;
/// - `weight` is a valid `DUAL_SCALE_HOT_CPU_PORTABLE_V1` block;
/// - `activation` is a valid `Q8K_ACTIVATION_HOT_CPU_PORTABLE_V1` block;
/// - both owners remain valid and immutable for the complete call;
/// - the frozen block/group constants remain consistent:
///   256 lanes, eight groups, 32 lanes per group;
/// - the documented i16/i32 accumulator bounds hold.
///
/// This function uses unaligned loads and does not require 16-byte or
/// 32-byte owner alignment.
#[doc(hidden)]
#[target_feature(enable = "avx2")]
pub unsafe fn dot_dual_scale_block_q8k_hot_avx2_prevalidated(
    weight: &DualScaleHotCpuBlock,
    activation: &Q8KActivationHotCpuBlock,
) -> f32 {
    let activation_scale = activation.d.to_f32();

    if activation_scale == 0.0 {
        return 0.0;
    }

    let mut output = 0.0_f32;

    for group in 0..DUAL_SCALE_HOT_CPU_GROUPS {
        let packed_offset = group * PACKED_Q4_BYTES_PER_GROUP;
        let packed_ptr = unsafe { weight.qs.as_ptr().add(packed_offset).cast::<__m128i>() };
        let packed = unsafe { _mm_loadu_si128(packed_ptr) };

        let (first_q4_codes, second_q4_codes) = unsafe { expand_q4_group_codes(packed) };

        let activation_offset = group * DUAL_SCALE_HOT_CPU_GROUP_SIZE;
        let first_q8_ptr = unsafe { activation.qs.as_ptr().add(activation_offset) };
        let second_q8_ptr = unsafe { first_q8_ptr.add(LANES_PER_HALF) };

        let first_q8 = unsafe { _mm_loadu_si128(first_q8_ptr.cast::<__m128i>()) };
        let second_q8 = unsafe { _mm_loadu_si128(second_q8_ptr.cast::<__m128i>()) };

        let mem_offset = group * MEMBERSHIP_BYTES_PER_GROUP;
        let mem_ptr = unsafe { weight.membership.as_ptr().add(mem_offset) };

        let first_membership = unsafe { load_membership_u16(mem_ptr) };
        let second_membership =
            unsafe { load_membership_u16(mem_ptr.add(MEMBERSHIP_BYTES_PER_GROUP / 2)) };

        let zero = _mm256_setzero_si256();

        let (lo_acc, hi_acc) =
            unsafe { accumulate_sixteen(first_q4_codes, first_q8, first_membership, zero, zero) };
        let (lo_acc, hi_acc) = unsafe {
            accumulate_sixteen(
                second_q4_codes,
                second_q8,
                second_membership,
                lo_acc,
                hi_acc,
            )
        };

        let lo = unsafe { horizontal_sum_i32x8(lo_acc) };
        let hi = unsafe { horizontal_sum_i32x8(hi_acc) };

        let scale_lo = weight.scale_lo[group].to_f32();
        if scale_lo != 0.0 {
            output += (scale_lo * activation_scale) * (lo as f32);
        }

        let scale_hi = weight.scale_hi[group].to_f32();
        if scale_hi != 0.0 {
            output += (scale_hi * activation_scale) * (hi as f32);
        }
    }

    output
}

/// Computes one complete DualScale Hot × Q8Hot row main-path dot.
///
/// CSR correction is deliberately not part of M28-B.
///
/// # Safety
///
/// The caller must guarantee:
///
/// - the current CPU supports AVX2;
/// - `row_blocks.len() == activation.block_count()`;
/// - `activation.len_elements()` equals
///   `row_blocks.len() * DualScaleHotCpuBlock::DECODED_LEN`;
/// - `row_blocks` is one complete row from a validated `DualScaleHotCpuTensor`;
/// - every activation block corresponds to the same logical column block as the
///   matched weight block;
/// - all owners remain valid and immutable for the complete call;
/// - all preconditions of [`dot_dual_scale_block_q8k_hot_avx2_prevalidated`] hold.
///
/// This function performs no allocation, no feature detection, no registry lookup
/// and no bounds validation in its block loop.
#[doc(hidden)]
#[target_feature(enable = "avx2")]
pub unsafe fn dot_dual_scale_hot_q8k_hot_avx2_prevalidated(
    row_blocks: &[DualScaleHotCpuBlock],
    activation: &Q8KActivationHotCpuVector,
) -> f32 {
    debug_assert_eq!(row_blocks.len(), activation.block_count());
    debug_assert_eq!(
        row_blocks
            .len()
            .checked_mul(DualScaleHotCpuBlock::DECODED_LEN),
        Some(activation.len_elements()),
    );

    let mut output = 0.0_f32;

    for (weight_block, activation_block) in row_blocks.iter().zip(activation.blocks().iter()) {
        output += unsafe {
            dot_dual_scale_block_q8k_hot_avx2_prevalidated(weight_block, activation_block)
        };
    }

    output
}

// ── M28-C: complete checked qmatvec ------------------------------------------

pub fn qmatvec_dual_scale_q8k_hot_cpu_avx2(
    weight: &DualScaleHotCpuTensor,
    activation: &Q8KActivationHotCpuVector,
    output: &mut [f32],
) -> Result<(), KernelError> {
    let validated_csr = validate_qmatvec_dual_scale_q8hot_avx2(weight, activation, output)?;

    if !std::is_x86_feature_detected!("avx2") {
        return Err(KernelError::MissingCpuFeatures);
    }

    unsafe {
        qmatvec_dual_scale_q8k_hot_cpu_avx2_prevalidated(weight, activation, output, validated_csr);
    }
    Ok(())
}

pub fn launch_qmatvec_dual_scale_q8k_hot_cpu_avx2(
    weight: &DualScaleHotCpuTensor,
    activation: &Q8KActivationHotCpuVector,
    output: &mut [f32],
) -> Result<(), KernelLaunchError> {
    qmatvec_dual_scale_q8k_hot_cpu_avx2(weight, activation, output)
        .map_err(KernelError::into_launch_error)
}

/// Infallible AVX2 launch for a binder-validated owner on an AVX2 plan.
///
/// # Panics
///
/// Panics if the current CPU does not support AVX2 or if `activation`/`output`
/// do not match the bound weight shape. The binder guarantees these
/// preconditions on production plans, so the checks are a once-per-launch
/// capability guard, never an inner-loop cost; they exist so this safe entry
/// cannot reach the `unsafe` kernel with a violated invariant in release
/// builds.
pub fn launch_qmatvec_dual_scale_q8k_hot_cpu_avx2_prevalidated(
    bound: &BoundDualScaleQ8Hot,
    activation: &Q8KActivationHotCpuVector,
    output: &mut [f32],
) {
    let weight = bound.weight();
    assert!(std::is_x86_feature_detected!("avx2"));
    assert_eq!(activation.len_elements(), weight.cols());
    assert_eq!(activation.block_count(), weight.blocks_per_row());
    assert!(output.len() >= weight.rows());
    unsafe {
        qmatvec_dual_scale_q8k_hot_cpu_avx2_prevalidated(
            weight,
            activation,
            output,
            weight.outliers().prevalidated_rows(),
        );
    }
}

fn validate_qmatvec_dual_scale_q8hot_avx2<'a>(
    weight: &'a DualScaleHotCpuTensor,
    activation: &Q8KActivationHotCpuVector,
    output: &[f32],
) -> Result<ValidatedOutlierCsrRows<'a>, KernelError> {
    validate_dual_scale_weight_shape_avx2(weight)?;
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
    let outliers = weight.outliers();
    if outliers.rows() != weight.rows() || outliers.cols() != weight.cols() {
        return Err(KernelError::BadWeightShape);
    }
    let validated_csr = outliers.validate_rows().ok_or(KernelError::BadWeight)?;
    for row in 0..weight.rows() {
        weight.row_blocks(row).ok_or(KernelError::BadWeight)?;
    }
    Ok(validated_csr)
}

fn validate_dual_scale_weight_shape_avx2(
    weight: &DualScaleHotCpuTensor,
) -> Result<(), KernelError> {
    if weight.rows() == 0 || weight.cols() == 0 || weight.blocks_per_row() == 0 {
        return Err(KernelError::BadWeightShape);
    }
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

#[target_feature(enable = "avx2")]
unsafe fn qmatvec_dual_scale_q8k_hot_cpu_avx2_prevalidated(
    weight: &DualScaleHotCpuTensor,
    activation: &Q8KActivationHotCpuVector,
    output: &mut [f32],
    validated_csr: ValidatedOutlierCsrRows<'_>,
) {
    debug_assert!(weight.rows() > 0);
    debug_assert!(weight.cols() > 0);
    debug_assert!(weight.blocks_per_row() > 0);
    let row_chunks = weight.blocks().chunks_exact(weight.blocks_per_row());
    let csr_rows = validated_csr.iter();
    for ((row_blocks, csr_row), dst) in row_chunks
        .zip(csr_rows)
        .zip(output[..weight.rows()].iter_mut())
    {
        let main = unsafe { dot_dual_scale_hot_q8k_hot_avx2_prevalidated(row_blocks, activation) };
        let correction = outlier_row_correction_q8hot_avx2(csr_row, activation);
        *dst = main + correction;
    }
}

fn outlier_row_correction_q8hot_avx2(
    row: OutlierCsrHotCpuRow<'_>,
    activation: &Q8KActivationHotCpuVector,
) -> f32 {
    let mut correction = 0.0_f32;
    for (&column, &delta) in row.entries() {
        let column = column as usize;
        let activation_value = decode_q8hot_at_validated_avx2(activation, column);
        correction += delta.to_f32() * activation_value;
    }
    correction
}

fn decode_q8hot_at_validated_avx2(activation: &Q8KActivationHotCpuVector, column: usize) -> f32 {
    debug_assert!(column < activation.len_elements());
    let block_index = column / Q8KActivationHotCpuBlock::DECODED_LEN;
    let lane = column % Q8KActivationHotCpuBlock::DECODED_LEN;
    debug_assert!(block_index < activation.block_count());
    debug_assert!(lane < Q8KActivationHotCpuBlock::DECODED_LEN);
    let block = unsafe { activation.blocks().get_unchecked(block_index) };
    let quantized = unsafe { *block.qs.get_unchecked(lane) };
    let scale = block.d.to_f32();
    if scale == 0.0 {
        return 0.0;
    }
    scale * f32::from(quantized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::arch::x86_64::_mm256_storeu_si256;

    fn supported() -> bool {
        std::is_x86_feature_detected!("avx2")
    }

    #[target_feature(enable = "avx2")]
    unsafe fn expanded_membership(bits: u16) -> [i16; 16] {
        let mask = unsafe { membership_mask_i16(bits) };
        let mut output = [0_i16; 16];
        unsafe { _mm256_storeu_si256(output.as_mut_ptr().cast::<__m256i>(), mask) };
        output
    }

    #[test]
    fn membership_mask_is_lsb_first_and_bit_exact() {
        if !supported() {
            return;
        }
        for lane in 0..16 {
            let bits = 1_u16 << lane;
            let mask = unsafe { expanded_membership(bits) };
            for (actual_lane, &value) in mask.iter().enumerate() {
                let expected = if actual_lane == lane { -1 } else { 0 };
                assert_eq!(
                    value, expected,
                    "source_lane={lane} actual_lane={actual_lane}"
                );
            }
        }
    }

    #[test]
    fn membership_mask_handles_all_lo_and_all_hi() {
        if !supported() {
            return;
        }
        let all_lo = unsafe { expanded_membership(0x0000) };
        assert_eq!(all_lo, [0; 16]);
        let all_hi = unsafe { expanded_membership(0xffff) };
        assert_eq!(all_hi, [-1; 16]);
    }

    #[test]
    fn membership_mask_handles_alternating_patterns() {
        if !supported() {
            return;
        }
        let alt_aa = unsafe { expanded_membership(0xaaaa) };
        let alt_55 = unsafe { expanded_membership(0x5555) };
        for lane in 0..16 {
            assert_eq!(alt_aa[lane], if lane % 2 == 1 { -1 } else { 0 });
            assert_eq!(alt_55[lane], if lane % 2 == 0 { -1 } else { 0 });
        }
    }
}
