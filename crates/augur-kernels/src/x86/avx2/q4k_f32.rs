use core::arch::x86_64::{
    __m128i, __m256, _mm256_add_ps, _mm256_castps256_ps128, _mm256_cvtepi32_ps,
    _mm256_cvtepu8_epi32, _mm256_extractf128_ps, _mm256_fmadd_ps, _mm256_loadu_ps,
    _mm256_setzero_ps, _mm_add_ps, _mm_and_si128, _mm_cvtss_f32, _mm_hadd_ps, _mm_loadu_si128,
    _mm_set1_epi8, _mm_srli_epi16, _mm_srli_si128,
};
use augur_kernel_api::{
    KernelLaunchError, Q4KHotCpuBlock, Q4KHotCpuTensor, Q4K_HOT_CPU_GROUPS, Q4K_HOT_CPU_GROUP_SIZE,
};

use crate::portable::q4k::validate_qmatmul_args;
use crate::KernelError;

const PACKED_BYTES_PER_GROUP_PAIR: usize = Q4K_HOT_CPU_GROUP_SIZE;
const LANES_PER_VECTOR: usize = 8;

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_f32x8(value: __m256) -> f32 {
    let low = _mm256_castps256_ps128(value);
    let high = _mm256_extractf128_ps(value, 1);
    let sum128 = _mm_add_ps(low, high);
    let pair_sum = _mm_hadd_ps(sum128, sum128);
    let total = _mm_hadd_ps(pair_sum, pair_sum);
    _mm_cvtss_f32(total)
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn extract_q4_nibbles(packed: __m128i, high: bool) -> __m128i {
    let nibble_mask = _mm_set1_epi8(0x0f);
    if high {
        _mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask)
    } else {
        _mm_and_si128(packed, nibble_mask)
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn accumulate_eight(
    q8: __m128i,
    activation: *const f32,
    q_dot: __m256,
    x_sum: __m256,
) -> (__m256, __m256) {
    let q_i32 = _mm256_cvtepu8_epi32(q8);
    let q_f32 = _mm256_cvtepi32_ps(q_i32);
    let x = unsafe { _mm256_loadu_ps(activation) };
    (_mm256_fmadd_ps(q_f32, x, q_dot), _mm256_add_ps(x_sum, x))
}

/// Computes a Q4K Hot × contiguous f32 row dot using AVX2 and FMA.
///
/// This is an unregistered M27-B primitive. It directly consumes
/// `Q4K_HOT_CPU_PORTABLE_V1`.
///
/// # Safety
///
/// The caller must guarantee:
///
/// - the current CPU supports AVX2 and FMA;
/// - `activation.len() == weight.len() * 256`;
/// - `weight` is a complete row of valid `Q4KHotCpuBlock` values;
/// - both slices remain valid and immutable for the call.
///
/// M27-B benchmark callers must detect AVX2/FMA before entering their timed
/// loop. Production capability enforcement belongs to M27-D lowering.
#[doc(hidden)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_q4k_hot_f32_avx2_prevalidated(
    weight: &[Q4KHotCpuBlock],
    activation: &[f32],
) -> f32 {
    debug_assert_eq!(
        weight.len().checked_mul(Q4KHotCpuBlock::DECODED_LEN),
        Some(activation.len()),
    );

    let mut result = 0.0_f32;

    for (block_index, block) in weight.iter().enumerate() {
        let d = block.d.to_f32();
        let dmin = block.dmin.to_f32();
        let block_offset = block_index * Q4KHotCpuBlock::DECODED_LEN;
        let block_activation = unsafe { activation.as_ptr().add(block_offset) };

        for group in 0..Q4K_HOT_CPU_GROUPS {
            let packed_offset = (group / 2) * PACKED_BYTES_PER_GROUP_PAIR;
            let first_packed =
                unsafe { _mm_loadu_si128(block.qs.as_ptr().add(packed_offset).cast::<__m128i>()) };
            let second_packed = unsafe {
                _mm_loadu_si128(block.qs.as_ptr().add(packed_offset + 16).cast::<__m128i>())
            };
            let high = group & 1 != 0;
            let first_16 = unsafe { extract_q4_nibbles(first_packed, high) };
            let second_16 = unsafe { extract_q4_nibbles(second_packed, high) };

            let q_chunks = [
                first_16,
                _mm_srli_si128(first_16, 8),
                second_16,
                _mm_srli_si128(second_16, 8),
            ];

            let mut q_dot = _mm256_setzero_ps();
            let mut x_sum = _mm256_setzero_ps();

            for (chunk, q8) in q_chunks.into_iter().enumerate() {
                let activation_offset = group * Q4K_HOT_CPU_GROUP_SIZE + chunk * LANES_PER_VECTOR;
                let x_ptr = unsafe { block_activation.add(activation_offset) };
                (q_dot, x_sum) = unsafe { accumulate_eight(q8, x_ptr, q_dot, x_sum) };
            }

            let q_dot_scalar = unsafe { horizontal_sum_f32x8(q_dot) };
            let x_sum_scalar = unsafe { horizontal_sum_f32x8(x_sum) };

            let scaled_q = d * f32::from(block.scale[group]) * q_dot_scalar;
            let scaled_min = dmin * f32::from(block.min[group]) * x_sum_scalar;

            result += scaled_q - scaled_min;
        }
    }

    result
}

// ── M27-C: complete qmatmul ──────────────────────────────────────────────

/// Executes the complete Q4K × f32 Identity QMatmul after all launch
/// invariants and CPU features have been validated.
///
/// # Safety
///
/// The caller must guarantee:
///
/// - the current CPU supports AVX2 and FMA;
/// - `weight.rows() > 0`;
/// - `weight.cols() > 0`;
/// - `weight.blocks_per_row() > 0`;
/// - `weight.cols() == weight.blocks_per_row() * 256`;
/// - `weight.blocks().len() == weight.rows() * weight.blocks_per_row()`;
/// - `activation.len() == weight.cols()`;
/// - `output.len() >= weight.rows()`;
/// - weight, activation and output do not alias incompatibly.
///
/// The function performs no allocation and no recoverable validation.
#[target_feature(enable = "avx2,fma")]
unsafe fn qmatmul_q4k_f32_avx2_prevalidated(
    weight: &Q4KHotCpuTensor,
    activation: &[f32],
    output: &mut [f32],
) {
    debug_assert!(weight.rows() > 0);
    debug_assert!(weight.cols() > 0);
    debug_assert!(weight.blocks_per_row() > 0);
    debug_assert_eq!(
        weight
            .blocks_per_row()
            .checked_mul(Q4KHotCpuBlock::DECODED_LEN),
        Some(weight.cols()),
    );
    debug_assert_eq!(
        weight.rows().checked_mul(weight.blocks_per_row()),
        Some(weight.blocks().len()),
    );
    debug_assert_eq!(activation.len(), weight.cols());
    debug_assert!(output.len() >= weight.rows());

    let row_chunks = weight.blocks().chunks_exact(weight.blocks_per_row());
    debug_assert!(row_chunks.remainder().is_empty());
    debug_assert_eq!(row_chunks.len(), weight.rows());

    for (out, row_blocks) in output[..weight.rows()].iter_mut().zip(row_chunks) {
        *out = unsafe { dot_q4k_hot_f32_avx2_prevalidated(row_blocks, activation) };
    }
}

/// Complete, checked Q4K × f32 AVX2/FMA Identity QMatmul.
///
/// Validation and feature detection complete before the first output write.
/// A failed call therefore leaves `output` unchanged.
///
/// This M27-C entry point is not registered and is not selected by lowering.
pub fn qmatmul_q4k_f32_avx2(
    weight: &Q4KHotCpuTensor,
    activation: &[f32],
    output: &mut [f32],
) -> Result<(), KernelError> {
    validate_qmatmul_args(weight, activation, output, None)?;

    // Portable shared validation accepts some zero-dimensional owners, but
    // chunks_exact(0) would panic. M27-C rejects these before writing.
    if weight.rows() == 0 || weight.cols() == 0 || weight.blocks_per_row() == 0 {
        return Err(KernelError::BadWeightShape);
    }

    if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
        return Err(KernelError::MissingCpuFeatures);
    }

    unsafe { qmatmul_q4k_f32_avx2_prevalidated(weight, activation, output) }

    Ok(())
}

/// Typed launch-boundary wrapper for the M27-C AVX2/FMA Identity kernel.
///
/// The function matches `QMatmulF32Launch`, but M27-C does not register it.
/// Registration and lowering selection belong to M27-D.
pub fn launch_qmatmul_q4k_f32_avx2(
    weight: &Q4KHotCpuTensor,
    activation: &[f32],
    output: &mut [f32],
) -> Result<(), KernelLaunchError> {
    qmatmul_q4k_f32_avx2(weight, activation, output).map_err(KernelError::into_launch_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::arch::x86_64::_mm_storeu_si128;

    fn supported() -> bool {
        std::is_x86_feature_detected!("avx2")
    }

    #[target_feature(enable = "avx2")]
    unsafe fn extract(packed_bytes: [u8; 16], high: bool) -> [u8; 16] {
        let packed = unsafe { _mm_loadu_si128(packed_bytes.as_ptr().cast::<__m128i>()) };
        let values = unsafe { extract_q4_nibbles(packed, high) };
        let mut output = [0_u8; 16];
        unsafe { _mm_storeu_si128(output.as_mut_ptr().cast::<__m128i>(), values) };
        output
    }

    fn expected(packed: &[u8; 16], high: bool) -> [u8; 16] {
        let mut output = [0_u8; 16];
        for (index, byte) in packed.iter().copied().enumerate() {
            output[index] = if high { byte >> 4 } else { byte & 0x0f };
        }
        output
    }

    #[test]
    fn nibble_expand_is_bit_exact() {
        if !supported() {
            return;
        }

        let cases = [
            [0x00_u8; 16],
            [0xff_u8; 16],
            [0x0f_u8; 16],
            [0xf0_u8; 16],
            core::array::from_fn(|i| if i % 2 == 0 { 0x0f } else { 0xf0 }),
            core::array::from_fn(|i| {
                let low = (i * 2) as u8 & 0x0f;
                let high = (i * 2 + 1) as u8 & 0x0f;
                low | (high << 4)
            }),
        ];

        for packed in cases {
            for high in [false, true] {
                let got = unsafe { extract(packed, high) };
                assert_eq!(got, expected(&packed, high));
            }
        }
    }
}
