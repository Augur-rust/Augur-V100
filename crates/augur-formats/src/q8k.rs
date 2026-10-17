//! Q8K activation reference format.
//!
//! # Format contract
//!
//! - **compact layout**:
//!   `Q8KActivationBlock { d: F16Bits, qs: [i8; 256] }` and
//!   `Q8KActivationVector { blocks, len_elements }` for one decode/GEMV vector.
//! - **hot layout**:
//!   deferred. M10 has no backend-specific Q8K hot layout. Future AVX2/CUDA
//!   paths should introduce repacked activation layouts through `augur-repack`.
//! - **encode/decode reference**:
//!   `quantize_q8k_activation_ref`, `quantize_q8k_activation_vector_ref`,
//!   and `decode_ref`.
//! - **repack logic**:
//!   deferred. M10 consumes the reference vector only in portable kernels.
//!   No scale predecode is needed yet because each block stores one `d`.
//! - **kernel consumption**:
//!   portable Q4K × Q8K kernels consume `Q8KActivationVector` directly as a
//!   staged exception for math validation.
//! - **precision metric**:
//!   NumericallyClose against Q4K × decoded-Q8K f32 reference, relative
//!   tolerance 1e-3 in current tests.
//! - **decode cost**:
//!   per block: one f16-to-f32 scale conversion, 256 i8-to-f32 conversions,
//!   and 256 f32 multiplies. Vector decode is linear in block count.
//!
//! `Q8KActivationVector` represents a single logical vector `[cols]` split
//! into 256-element blocks. It does not represent batch/prefill activations.

use crate::{FormatError, ReferenceDecode};
use augur_core::F16Bits;

pub const Q8K_BLOCK_SIZE: usize = 256;

// ── Q8K Activation Block ──────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Q8KActivationBlock {
    pub d: F16Bits,
    pub qs: [i8; Q8K_BLOCK_SIZE],
}

impl ReferenceDecode for Q8KActivationBlock {
    type Elem = f32;

    fn decoded_len(&self) -> usize {
        Q8K_BLOCK_SIZE
    }

    fn decode_ref(&self, dst: &mut [Self::Elem]) -> Result<(), FormatError> {
        if dst.len() < Q8K_BLOCK_SIZE {
            return Err(FormatError::OutputTooSmall {
                expected: Q8K_BLOCK_SIZE,
                actual: dst.len(),
            });
        }

        let d = self.d.to_f32();

        for (out, &q) in dst.iter_mut().take(Q8K_BLOCK_SIZE).zip(self.qs.iter()) {
            *out = d * (q as f32);
        }

        Ok(())
    }
}

// ── Quantize (single block) ───────────────────────────────────────────────

pub fn quantize_q8k_activation_ref(src: &[f32]) -> Result<Q8KActivationBlock, FormatError> {
    if src.len() != Q8K_BLOCK_SIZE {
        return Err(FormatError::InputLengthMismatch {
            expected: Q8K_BLOCK_SIZE,
            actual: src.len(),
        });
    }

    for (i, &v) in src.iter().enumerate() {
        if !v.is_finite() {
            return Err(FormatError::NonFiniteInput { index: i });
        }
    }

    let max_abs = src.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

    if max_abs == 0.0 {
        return Ok(Q8KActivationBlock {
            d: F16Bits::ZERO,
            qs: [0; Q8K_BLOCK_SIZE],
        });
    }

    let d = F16Bits::from_f32(max_abs / 127.0);
    let d_f32 = d.to_f32();

    // Guard: if the f16-rounded scale underflows to zero, quantize as all-zero
    // so decode is self-consistent (d == 0 → all outputs zero).
    if d_f32 == 0.0 {
        return Ok(Q8KActivationBlock {
            d: F16Bits::ZERO,
            qs: [0; Q8K_BLOCK_SIZE],
        });
    }

    let inv_d = 1.0 / d_f32;
    let mut qs = [0i8; Q8K_BLOCK_SIZE];

    for i in 0..Q8K_BLOCK_SIZE {
        let q = (src[i] * inv_d).round().clamp(-127.0, 127.0) as i32;
        qs[i] = q as i8;
    }

    Ok(Q8KActivationBlock { d, qs })
}

// ── Q8K Activation Vector ─────────────────────────────────────────────────

/// A single sequence of [`Q8KActivationBlock`]s forming one logical
/// activation vector of `len_elements` floats.
///
/// `len_elements` must be a multiple of [`Q8K_BLOCK_SIZE`].
/// `blocks.len()` must equal `len_elements / Q8K_BLOCK_SIZE`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Q8KActivationVector {
    blocks: Vec<Q8KActivationBlock>,
    len_elements: usize,
}

impl Q8KActivationVector {
    pub fn new(blocks: Vec<Q8KActivationBlock>, len_elements: usize) -> Result<Self, FormatError> {
        if len_elements == 0 {
            return Err(FormatError::BadLength);
        }
        if len_elements % Q8K_BLOCK_SIZE != 0 {
            return Err(FormatError::InputLengthMismatch {
                expected: round_up_to_block_for_error_msg(len_elements),
                actual: len_elements,
            });
        }

        let expected_blocks = len_elements / Q8K_BLOCK_SIZE;

        if blocks.len() != expected_blocks {
            return Err(FormatError::InputLengthMismatch {
                expected: expected_blocks,
                actual: blocks.len(),
            });
        }

        Ok(Self {
            blocks,
            len_elements,
        })
    }

    pub fn blocks(&self) -> &[Q8KActivationBlock] {
        &self.blocks
    }

    pub fn len_elements(&self) -> usize {
        self.len_elements
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Decodes into the first `len_elements` entries of `dst`.
    /// If `dst` is longer, trailing entries are left unchanged.
    pub fn decode_ref(&self, dst: &mut [f32]) -> Result<(), FormatError> {
        if dst.len() < self.len_elements {
            return Err(FormatError::OutputTooSmall {
                expected: self.len_elements,
                actual: dst.len(),
            });
        }

        for (block_idx, block) in self.blocks.iter().enumerate() {
            let start = block_idx * Q8K_BLOCK_SIZE;
            let end = start + Q8K_BLOCK_SIZE;
            block.decode_ref(&mut dst[start..end])?;
        }

        Ok(())
    }
}

impl ReferenceDecode for Q8KActivationVector {
    type Elem = f32;

    fn decoded_len(&self) -> usize {
        self.len_elements()
    }

    fn decode_ref(&self, dst: &mut [Self::Elem]) -> Result<(), FormatError> {
        Q8KActivationVector::decode_ref(self, dst)
    }
}

// ── Quantize (vector) ─────────────────────────────────────────────────────

pub fn quantize_q8k_activation_vector_ref(src: &[f32]) -> Result<Q8KActivationVector, FormatError> {
    if src.is_empty() {
        return Err(FormatError::BadLength);
    }
    if src.len() % Q8K_BLOCK_SIZE != 0 {
        return Err(FormatError::InputLengthMismatch {
            expected: round_up_to_block_for_error_msg(src.len()),
            actual: src.len(),
        });
    }

    let mut blocks = Vec::with_capacity(src.len() / Q8K_BLOCK_SIZE);

    for chunk in src.chunks_exact(Q8K_BLOCK_SIZE) {
        blocks.push(quantize_q8k_activation_ref(chunk)?);
    }

    Q8KActivationVector::new(blocks, src.len())
}

// ── Helpers ───────────────────────────────────────────────────────────────

/// Only used for `InputLengthMismatch.expected` in diagnostics.
fn round_up_to_block_for_error_msg(n: usize) -> usize {
    n.checked_add(Q8K_BLOCK_SIZE - 1)
        .map(|value| value / Q8K_BLOCK_SIZE * Q8K_BLOCK_SIZE)
        .unwrap_or(usize::MAX)
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8k_quantize_zero_block() {
        let src = vec![0.0f32; 256];
        let block = quantize_q8k_activation_ref(&src).unwrap();
        assert_eq!(block.d, F16Bits::ZERO);
        assert!(block.qs.iter().all(|&q| q == 0));
    }

    #[test]
    fn q8k_quantize_rejects_bad_length() {
        let src = vec![0.0f32; 128];
        assert_eq!(
            quantize_q8k_activation_ref(&src),
            Err(FormatError::InputLengthMismatch {
                expected: 256,
                actual: 128,
            })
        );
        assert_eq!(round_up_to_block_for_error_msg(usize::MAX), usize::MAX);
        assert_eq!(
            Q8KActivationVector::new(Vec::new(), 0),
            Err(FormatError::BadLength)
        );
        assert_eq!(
            quantize_q8k_activation_vector_ref(&[]),
            Err(FormatError::BadLength)
        );
    }

    #[test]
    fn q8k_quantize_rejects_non_finite() {
        let mut src = vec![0.0f32; 256];
        src[42] = f32::NAN;
        assert_eq!(
            quantize_q8k_activation_ref(&src),
            Err(FormatError::NonFiniteInput { index: 42 })
        );
    }

    #[test]
    fn q8k_decode_rejects_short_output() {
        let src = vec![0.0f32; 256];
        let block = quantize_q8k_activation_ref(&src).unwrap();
        let mut dst = vec![0.0f32; 128];
        assert_eq!(
            block.decode_ref(&mut dst),
            Err(FormatError::OutputTooSmall {
                expected: 256,
                actual: 128,
            })
        );
    }

    #[test]
    fn q8k_quantize_decode_reasonable_error() {
        let src: Vec<f32> = (0..256).map(|i| ((i % 29) as f32 - 14.0) * 0.37).collect();
        let block = quantize_q8k_activation_ref(&src).unwrap();
        let mut decoded = vec![0.0f32; 256];
        block.decode_ref(&mut decoded).unwrap();

        let mut max_err = 0.0f32;
        for i in 0..256 {
            let err = (decoded[i] - src[i]).abs();
            if err > max_err {
                max_err = err;
            }
        }
        assert!(max_err < 0.1, "max_err={max_err} too large");
    }
}
