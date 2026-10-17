//! Allocation-free CPU portable Q8Hot activation preparation.
//!
//! This implementation writes directly into an existing fixed-shape
//! `Q8KActivationHotCpuVector`.
//!
//! It does not:
//!
//! - allocate a reference Q8K owner;
//! - allocate a Hot owner;
//! - call `augur-repack`;
//! - consume Compact/reference Q8 types;
//! - resize the destination;
//! - perform registry lookup;
//! - use unsafe code.
//!
//! # Execution phases
//!
//! ```text
//! 1. validate input/output logical length
//! 2. global finite/range preflight
//! 3. obtain fixed-shape writer
//! 4. calculate one max_abs per block
//! 5. quantize and overwrite existing blocks
//! ```
//!
//! All recoverable errors occur before the writer is created and before the
//! first destination block is modified.

use augur_core::F16Bits;
use augur_kernel_api::{
    KernelLaunchError, Q8KActivationHotCpuBlock, Q8KActivationHotCpuVector, Q8K_HOT_CPU_BLOCK_LEN,
    Q8K_HOT_CPU_MAX_INPUT_ABS,
};

use crate::KernelError;

/// Direct allocation-free f32 -> Q8Hot CPU portable quantizer.
///
/// # Numerical contract
///
/// For inputs in the production accepted domain, output is BitExact with:
///
/// ```text
/// quantize_q8k_activation_vector_ref
/// -> repack_q8k_activation_vector_cpu_ref
/// ```
///
/// Accepted input domain:
///
/// - input length exactly matches the fixed output owner;
/// - every input value is finite;
/// - every input absolute value is at most
///   `Q8K_HOT_CPU_MAX_INPUT_ABS`.
///
/// # Error guarantee
///
/// On any returned error, `output` is left unchanged.
pub fn quantize_q8k_hot_cpu_portable_into(
    input: &[f32],
    output: &mut Q8KActivationHotCpuVector,
) -> Result<(), KernelError> {
    validate_length(input, output)?;
    preflight_input(input)?;

    // Writer creation occurs only after every recoverable input error has
    // been ruled out.
    let mut writer = output.writer();

    for (source, destination) in input
        .chunks_exact(Q8K_HOT_CPU_BLOCK_LEN)
        .zip(writer.iter_mut())
    {
        *destination = quantize_prevalidated_block(source);
    }

    Ok(())
}

/// Typed launch wrapper for the portable direct Q8Hot quantizer.
pub fn launch_quantize_q8k_hot_cpu_portable(
    input: &[f32],
    output: &mut Q8KActivationHotCpuVector,
) -> Result<(), KernelLaunchError> {
    quantize_q8k_hot_cpu_portable_into(input, output).map_err(KernelError::into_launch_error)
}

fn validate_length(input: &[f32], output: &Q8KActivationHotCpuVector) -> Result<(), KernelError> {
    let expected = output.len_elements();
    if input.len() != expected {
        return Err(KernelError::BadActivationLength {
            expected,
            actual: input.len(),
        });
    }
    Ok(())
}

/// Validates the complete activation before any output write.
///
/// This pass deliberately does not calculate per-block max_abs. It verifies
/// that every value is finite and within the production accepted input
/// domain. These conditions guarantee that the later per-block f16 scale is
/// finite.
///
/// The first offending logical element is reported.
fn preflight_input(input: &[f32]) -> Result<(), KernelError> {
    for (index, &value) in input.iter().enumerate() {
        if !value.is_finite() {
            return Err(KernelError::NonFiniteActivation { index });
        }
        // The accepted domain is inclusive. A value exactly equal to
        // Q8K_HOT_CPU_MAX_INPUT_ABS produces the largest finite f16 scale.
        if value.abs() > Q8K_HOT_CPU_MAX_INPUT_ABS {
            return Err(KernelError::ActivationValueOutOfRange { index });
        }
    }
    Ok(())
}

/// Quantizes one prevalidated logical Q8K block.
///
/// Preconditions established by the public entry point:
///
/// - `source.len() == Q8K_HOT_CPU_BLOCK_LEN`;
/// - every element is finite;
/// - every element is within the production accepted range.
///
/// This function returns no recoverable error.
fn quantize_prevalidated_block(source: &[f32]) -> Q8KActivationHotCpuBlock {
    debug_assert_eq!(source.len(), Q8K_HOT_CPU_BLOCK_LEN);

    let max_abs = source.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);

    if max_abs == 0.0 {
        return Q8KActivationHotCpuBlock::ZERO;
    }

    let scale = F16Bits::from_f32(max_abs / 127.0);
    let scale_f32 = scale.to_f32();

    if scale_f32 == 0.0 {
        return Q8KActivationHotCpuBlock::ZERO;
    }

    // Established by global preflight and the accepted-domain bound.
    debug_assert!(scale_f32.is_finite());
    debug_assert!(scale_f32 > 0.0);

    let inverse_scale = 1.0 / scale_f32;
    debug_assert!(inverse_scale.is_finite());
    debug_assert!(inverse_scale > 0.0);

    let mut quantized = [0_i8; Q8K_HOT_CPU_BLOCK_LEN];

    for index in 0..Q8K_HOT_CPU_BLOCK_LEN {
        quantized[index] = quantize_prevalidated_lane(source[index], inverse_scale);
    }

    Q8KActivationHotCpuBlock {
        d: scale,
        qs: quantized,
    }
}

#[inline]
fn quantize_prevalidated_lane(value: f32, inverse_scale: f32) -> i8 {
    debug_assert!(value.is_finite());
    debug_assert!(inverse_scale.is_finite());
    debug_assert!(inverse_scale > 0.0);

    let rounded = (value * inverse_scale).round().clamp(-127.0, 127.0);

    debug_assert!(rounded.is_finite());
    debug_assert!((-127.0..=127.0).contains(&rounded));

    rounded as i8
}
