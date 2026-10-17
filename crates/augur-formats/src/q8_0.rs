//! Q8_0 weight format semantics.

use augur_core::{F16Bits, FormatVersion, PhysicalLayoutId, QuantFormatId};

use crate::{FormatError, ReferenceDecode};

pub const Q8_0_BLOCK_SIZE: usize = 32;
pub const Q8_0_BLOCK_BYTES: usize = 34;
pub const Q8_0_FORMAT_ID: QuantFormatId = QuantFormatId::Q8_0;
pub const Q8_0_FORMAT_VERSION: FormatVersion = FormatVersion(1);
pub const Q8_0_COMPACT_LAYOUT: PhysicalLayoutId = PhysicalLayoutId::Q8_0_COMPACT_V1;

#[repr(C)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Q8_0CompactBlock {
    pub d: F16Bits,
    pub qs: [i8; Q8_0_BLOCK_SIZE],
}

impl Q8_0CompactBlock {
    pub const DECODED_LEN: usize = Q8_0_BLOCK_SIZE;

    pub const ZERO: Self = Self {
        d: F16Bits::ZERO,
        qs: [0; Q8_0_BLOCK_SIZE],
    };

    pub fn to_le_bytes(&self) -> [u8; Q8_0_BLOCK_BYTES] {
        let mut output = [0_u8; Q8_0_BLOCK_BYTES];
        output[..2].copy_from_slice(&self.d.to_bits().to_le_bytes());
        for (destination, source) in output[2..].iter_mut().zip(self.qs.iter()) {
            *destination = *source as u8;
        }
        output
    }

    pub fn from_le_bytes(bytes: &[u8]) -> Result<Self, FormatError> {
        if bytes.len() != Q8_0_BLOCK_BYTES {
            return Err(FormatError::InputLengthMismatch {
                expected: Q8_0_BLOCK_BYTES,
                actual: bytes.len(),
            });
        }
        let d = F16Bits::from_bits(u16::from_le_bytes([bytes[0], bytes[1]]));
        if !d.to_f32().is_finite() {
            return Err(FormatError::NonFiniteValue);
        }
        let mut qs = [0_i8; Q8_0_BLOCK_SIZE];
        for (destination, source) in qs.iter_mut().zip(bytes[2..].iter()) {
            *destination = *source as i8;
        }
        Ok(Self { d, qs })
    }
}

impl ReferenceDecode for Q8_0CompactBlock {
    type Elem = f32;

    fn decoded_len(&self) -> usize {
        Q8_0_BLOCK_SIZE
    }

    fn decode_ref(&self, destination: &mut [f32]) -> Result<(), FormatError> {
        if destination.len() < Q8_0_BLOCK_SIZE {
            return Err(FormatError::OutputTooSmall {
                expected: Q8_0_BLOCK_SIZE,
                actual: destination.len(),
            });
        }
        let scale = self.d.to_f32();
        if !scale.is_finite() {
            return Err(FormatError::NonFiniteValue);
        }
        for (output, quant) in destination.iter_mut().zip(self.qs.iter()) {
            *output = scale * (*quant as f32);
        }
        Ok(())
    }
}

/// Deterministic Q8_0 reference quantizer.
///
/// Quantized lanes use the unrounded f32 inverse scale, while the persisted
/// scale is f16. This order matches the bound llama.cpp Q8_0 protocol.
pub fn quantize_q8_0_block_ref(source: &[f32]) -> Result<Q8_0CompactBlock, FormatError> {
    if source.len() != Q8_0_BLOCK_SIZE {
        return Err(FormatError::InputLengthMismatch {
            expected: Q8_0_BLOCK_SIZE,
            actual: source.len(),
        });
    }
    for (index, value) in source.iter().enumerate() {
        if !value.is_finite() {
            return Err(FormatError::NonFiniteInput { index });
        }
    }
    let maximum = source
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f32, f32::max);
    if maximum == 0.0 {
        return Ok(Q8_0CompactBlock::ZERO);
    }
    let scale = maximum / 127.0;
    let persisted_scale = q8_scale_to_f16(scale);
    if !persisted_scale.to_f32().is_finite() {
        return Err(FormatError::NonFiniteValue);
    }
    let inverse_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };
    if !inverse_scale.is_finite() {
        // The C reference has no defined integer result for infinite products.
        return Err(FormatError::NonFiniteValue);
    }
    let mut qs = [0_i8; Q8_0_BLOCK_SIZE];
    for (destination, value) in qs.iter_mut().zip(source.iter()) {
        *destination = (value * inverse_scale).round().clamp(-127.0, 127.0) as i8;
    }
    Ok(Q8_0CompactBlock {
        d: persisted_scale,
        qs,
    })
}

// Keep the offline Q8_0 wire conversion independent of the legacy F16Bits
// constructor. This is the same portable nearest-even conversion used by the
// model crate and ggml; it retains half subnormals and exponent carries.
fn q8_scale_to_f16(value: f32) -> F16Bits {
    let mut base = (value.abs() * f32::from_bits(0x7780_0000))
        * f32::from_bits(0x0880_0000);
    let bits = value.to_bits();
    let shifted = bits.wrapping_add(bits);
    let bias = (shifted & 0xff00_0000).max(0x7100_0000);
    base += f32::from_bits((bias >> 1).wrapping_add(0x0780_0000));
    let rounded = base.to_bits();
    let nonsign = ((rounded >> 13) & 0x7c00) + (rounded & 0x0fff);
    F16Bits::from_bits(((bits >> 16) & 0x8000 | nonsign) as u16)
}

pub fn quantize_q8_0_tensor_ref(
    source: &[f32],
    rows: usize,
    cols: usize,
) -> Result<Vec<Q8_0CompactBlock>, FormatError> {
    if rows == 0 || cols == 0 || !cols.is_multiple_of(Q8_0_BLOCK_SIZE) {
        return Err(FormatError::BadLength);
    }
    let expected = rows.checked_mul(cols).ok_or(FormatError::BadLength)?;
    if source.len() != expected {
        return Err(FormatError::InputLengthMismatch {
            expected,
            actual: source.len(),
        });
    }
    source
        .chunks_exact(Q8_0_BLOCK_SIZE)
        .map(quantize_q8_0_block_ref)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_wire_size_and_known_decode_are_exact() {
        assert_eq!(core::mem::size_of::<Q8_0CompactBlock>(), Q8_0_BLOCK_BYTES);
        let block = Q8_0CompactBlock {
            d: F16Bits::from_f32(0.5),
            qs: core::array::from_fn(|index| index as i8 - 16),
        };
        let wire = block.to_le_bytes();
        assert_eq!(Q8_0CompactBlock::from_le_bytes(&wire), Ok(block.clone()));
        let mut decoded = [0.0_f32; Q8_0_BLOCK_SIZE];
        block.decode_ref(&mut decoded).unwrap();
        assert_eq!(decoded[0], -8.0);
        assert_eq!(decoded[31], 7.5);
    }

    #[test]
    fn zero_and_extrema_quantize_deterministically() {
        assert_eq!(
            quantize_q8_0_block_ref(&[0.0; Q8_0_BLOCK_SIZE]),
            Ok(Q8_0CompactBlock::ZERO)
        );
        let source: [f32; Q8_0_BLOCK_SIZE] = core::array::from_fn(|index| index as f32 - 16.0);
        let first = quantize_q8_0_block_ref(&source).unwrap();
        let second = quantize_q8_0_block_ref(&source).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.qs[0], -127);
        assert_eq!(first.qs[31], 119);
    }

    #[test]
    fn llama_cpp_b10001_q8_0_wire_vector_is_bit_exact() {
        // Independently derived from ggml-quants.c::quantize_row_q8_0_ref.
        let source: [f32; Q8_0_BLOCK_SIZE] = core::array::from_fn(|index| index as f32 - 16.0);
        let expected = [
            0x08, 0x30, 0x81, 0x89, 0x91, 0x99, 0xa1, 0xa9, 0xb1, 0xb9, 0xc0, 0xc8, 0xd0, 0xd8,
            0xe0, 0xe8, 0xf0, 0xf8, 0x00, 0x08, 0x10, 0x18, 0x20, 0x28, 0x30, 0x38, 0x40, 0x47,
            0x4f, 0x57, 0x5f, 0x67, 0x6f, 0x77,
        ];
        assert_eq!(
            quantize_q8_0_block_ref(&source).unwrap().to_le_bytes(),
            expected
        );
    }

    #[test]
    fn offline_q8_scale_uses_nearest_even_subnormals_and_carries() {
        for (scale, expected) in [
            (2.0_f32.powi(-24), 0x0001),
            (2.0_f32.powi(-25), 0x0000),
            (3.0 * 2.0_f32.powi(-25), 0x0002),
            (1.0 + 2.0_f32.powi(-11), 0x3c00),
            (1.0 + 3.0 * 2.0_f32.powi(-11), 0x3c02),
            (2.0 - 2.0_f32.powi(-11), 0x4000),
        ] {
            let mut source = [0.0; Q8_0_BLOCK_SIZE];
            source[0] = scale * 127.0;
            source[1] = -source[0];
            let block = quantize_q8_0_block_ref(&source).unwrap();
            assert_eq!(block.d.to_bits(), expected, "scale={scale}");
            assert_eq!(&block.qs[..2], &[127, -127]);
        }
    }

    #[test]
    fn offline_q8_rejects_unrepresentable_scales_and_nonfinite_inverse() {
        assert_eq!(quantize_q8_0_block_ref(&[f32::MAX; 32]), Err(FormatError::NonFiniteValue));
        assert_eq!(quantize_q8_0_block_ref(&[1.0e-37; 32]), Err(FormatError::NonFiniteValue));
        // Underflow to an f32 zero scale follows the explicit zero branch in ggml.
        assert_eq!(quantize_q8_0_block_ref(&[f32::from_bits(1); 32]), Ok(Q8_0CompactBlock::ZERO));
    }

    #[test]
    fn invalid_inputs_do_not_produce_a_block() {
        assert_eq!(
            quantize_q8_0_block_ref(&[0.0; 31]),
            Err(FormatError::InputLengthMismatch {
                expected: 32,
                actual: 31
            })
        );
        let mut source = [0.0; Q8_0_BLOCK_SIZE];
        source[7] = f32::NAN;
        assert_eq!(
            quantize_q8_0_block_ref(&source),
            Err(FormatError::NonFiniteInput { index: 7 })
        );
        let mut wire = [0_u8; Q8_0_BLOCK_BYTES];
        wire[..2].copy_from_slice(&0x7c00_u16.to_le_bytes());
        assert_eq!(
            Q8_0CompactBlock::from_le_bytes(&wire),
            Err(FormatError::NonFiniteValue)
        );
    }
}
