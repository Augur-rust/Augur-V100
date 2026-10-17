//! Q2_K weight-format semantics.

use augur_core::{F16Bits, FormatVersion, PhysicalLayoutId, QuantFormatId};

use crate::{FormatError, ReferenceDecode};

pub const Q2K_BLOCK_SIZE: usize = 256;
pub const Q2K_GROUPS: usize = 16;
pub const Q2K_GROUP_SIZE: usize = 16;
pub const Q2K_SCALE_BYTES: usize = 16;
pub const Q2K_QS_BYTES: usize = 64;
pub const Q2K_BLOCK_BYTES: usize = 84;
pub const Q2K_FORMAT_ID: QuantFormatId = QuantFormatId::Q2K;
pub const Q2K_FORMAT_VERSION: FormatVersion = FormatVersion(1);
pub const Q2K_COMPACT_LAYOUT: PhysicalLayoutId = PhysicalLayoutId::Q2K_COMPACT_V1;

#[repr(C)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Q2KCompactBlock {
    pub scales: [u8; Q2K_SCALE_BYTES],
    pub qs: [u8; Q2K_QS_BYTES],
    pub d: F16Bits,
    pub dmin: F16Bits,
}

impl Q2KCompactBlock {
    pub const DECODED_LEN: usize = Q2K_BLOCK_SIZE;

    pub const ZERO: Self = Self {
        scales: [0; Q2K_SCALE_BYTES],
        qs: [0; Q2K_QS_BYTES],
        d: F16Bits::ZERO,
        dmin: F16Bits::ZERO,
    };

    pub fn to_le_bytes(&self) -> [u8; Q2K_BLOCK_BYTES] {
        let mut output = [0_u8; Q2K_BLOCK_BYTES];
        output[..Q2K_SCALE_BYTES].copy_from_slice(&self.scales);
        output[Q2K_SCALE_BYTES..Q2K_SCALE_BYTES + Q2K_QS_BYTES].copy_from_slice(&self.qs);
        output[80..82].copy_from_slice(&self.d.to_bits().to_le_bytes());
        output[82..84].copy_from_slice(&self.dmin.to_bits().to_le_bytes());
        output
    }

    pub fn from_le_bytes(bytes: &[u8]) -> Result<Self, FormatError> {
        if bytes.len() != Q2K_BLOCK_BYTES {
            return Err(FormatError::InputLengthMismatch {
                expected: Q2K_BLOCK_BYTES,
                actual: bytes.len(),
            });
        }
        let mut scales = [0_u8; Q2K_SCALE_BYTES];
        scales.copy_from_slice(&bytes[..Q2K_SCALE_BYTES]);
        let mut qs = [0_u8; Q2K_QS_BYTES];
        qs.copy_from_slice(&bytes[Q2K_SCALE_BYTES..Q2K_SCALE_BYTES + Q2K_QS_BYTES]);
        let d = F16Bits::from_bits(u16::from_le_bytes([bytes[80], bytes[81]]));
        let dmin = F16Bits::from_bits(u16::from_le_bytes([bytes[82], bytes[83]]));
        if !d.to_f32().is_finite() || !dmin.to_f32().is_finite() {
            return Err(FormatError::NonFiniteValue);
        }
        Ok(Self {
            scales,
            qs,
            d,
            dmin,
        })
    }

    #[inline]
    pub fn quant_at(&self, index: usize) -> u8 {
        debug_assert!(index < Q2K_BLOCK_SIZE);
        let half = index / 128;
        let within_half = index % 128;
        let quarter = within_half / 32;
        let lane = within_half % 32;
        (self.qs[half * 32 + lane] >> (quarter * 2)) & 0x03
    }
}

impl ReferenceDecode for Q2KCompactBlock {
    type Elem = f32;

    fn decoded_len(&self) -> usize {
        Q2K_BLOCK_SIZE
    }

    fn decode_ref(&self, destination: &mut [f32]) -> Result<(), FormatError> {
        if destination.len() < Q2K_BLOCK_SIZE {
            return Err(FormatError::OutputTooSmall {
                expected: Q2K_BLOCK_SIZE,
                actual: destination.len(),
            });
        }
        let super_scale = self.d.to_f32();
        let super_min = self.dmin.to_f32();
        if !super_scale.is_finite() || !super_min.is_finite() {
            return Err(FormatError::NonFiniteValue);
        }
        for (index, output) in destination.iter_mut().take(Q2K_BLOCK_SIZE).enumerate() {
            let group = index / Q2K_GROUP_SIZE;
            let metadata = self.scales[group];
            let scale = super_scale * ((metadata & 0x0f) as f32);
            let minimum = super_min * ((metadata >> 4) as f32);
            *output = scale * (self.quant_at(index) as f32) - minimum;
        }
        Ok(())
    }
}

fn nearest_int(value: f32) -> i32 {
    debug_assert!(value.abs() <= 4_194_303.0);
    let bits = (value + 12_582_912.0).to_bits();
    ((bits & 0x007f_ffff) as i32) - 0x0040_0000
}

fn make_group_quants(
    values: &[f32; Q2K_GROUP_SIZE],
    quantized: &mut [u8; Q2K_GROUP_SIZE],
) -> (f32, f32) {
    let weights = values.map(f32::abs);
    let mut minimum = values[0];
    let mut maximum = values[0];
    let mut sum_weight = weights[0];
    let mut sum_x = weights[0] * values[0];
    for index in 1..Q2K_GROUP_SIZE {
        minimum = minimum.min(values[index]);
        maximum = maximum.max(values[index]);
        sum_weight += weights[index];
        sum_x += weights[index] * values[index];
    }
    minimum = minimum.min(0.0);
    if maximum == minimum {
        quantized.fill(0);
        return (0.0, -minimum);
    }

    let mut inverse_scale = 3.0 / (maximum - minimum);
    let mut scale = 1.0 / inverse_scale;
    let mut best_error = 0.0_f32;
    for index in 0..Q2K_GROUP_SIZE {
        let level = nearest_int(inverse_scale * (values[index] - minimum)).clamp(0, 3);
        quantized[index] = level as u8;
        best_error += weights[index] * (scale * (level as f32) + minimum - values[index]).abs();
    }

    let mut candidate = [0_u8; Q2K_GROUP_SIZE];
    for step in 0..=15 {
        inverse_scale = (-0.5 + 0.1 * (step as f32) + 3.0) / (maximum - minimum);
        let mut sum_l = 0.0_f32;
        let mut sum_l2 = 0.0_f32;
        let mut sum_xl = 0.0_f32;
        for index in 0..Q2K_GROUP_SIZE {
            let level = nearest_int(inverse_scale * (values[index] - minimum)).clamp(0, 3);
            candidate[index] = level as u8;
            let weight = weights[index];
            let level_f32 = level as f32;
            sum_l += weight * level_f32;
            sum_l2 += weight * level_f32 * level_f32;
            sum_xl += weight * level_f32 * values[index];
        }
        let determinant = sum_weight * sum_l2 - sum_l * sum_l;
        if determinant <= 0.0 {
            continue;
        }
        let mut candidate_scale = (sum_weight * sum_xl - sum_x * sum_l) / determinant;
        let mut candidate_minimum = (sum_l2 * sum_x - sum_l * sum_xl) / determinant;
        if candidate_minimum > 0.0 {
            candidate_minimum = 0.0;
            candidate_scale = sum_xl / sum_l2;
        }
        let mut candidate_error = 0.0_f32;
        for index in 0..Q2K_GROUP_SIZE {
            let difference =
                candidate_scale * (candidate[index] as f32) + candidate_minimum - values[index];
            candidate_error += weights[index] * difference.abs();
        }
        if candidate_error < best_error {
            quantized.copy_from_slice(&candidate);
            best_error = candidate_error;
            scale = candidate_scale;
            minimum = candidate_minimum;
        }
    }
    (scale, -minimum)
}

pub fn quantize_q2k_block_ref(source: &[f32]) -> Result<Q2KCompactBlock, FormatError> {
    if source.len() != Q2K_BLOCK_SIZE {
        return Err(FormatError::InputLengthMismatch {
            expected: Q2K_BLOCK_SIZE,
            actual: source.len(),
        });
    }
    for (index, value) in source.iter().enumerate() {
        if !value.is_finite() {
            return Err(FormatError::NonFiniteInput { index });
        }
    }

    let mut levels = [0_u8; Q2K_BLOCK_SIZE];
    let mut group_scales = [0.0_f32; Q2K_GROUPS];
    let mut group_mins = [0.0_f32; Q2K_GROUPS];
    for group in 0..Q2K_GROUPS {
        let values: [f32; Q2K_GROUP_SIZE] = source
            [group * Q2K_GROUP_SIZE..(group + 1) * Q2K_GROUP_SIZE]
            .try_into()
            .expect("fixed Q2_K group");
        let quantized: &mut [u8; Q2K_GROUP_SIZE] = levels
            .get_mut(group * Q2K_GROUP_SIZE..(group + 1) * Q2K_GROUP_SIZE)
            .expect("fixed Q2_K group")
            .try_into()
            .expect("fixed Q2_K group");
        (group_scales[group], group_mins[group]) = make_group_quants(&values, quantized);
    }

    let maximum_scale = group_scales.iter().copied().fold(0.0_f32, f32::max);
    let maximum_min = group_mins.iter().copied().fold(0.0_f32, f32::max);
    let d = if maximum_scale > 0.0 {
        F16Bits::from_f32(maximum_scale / 15.0)
    } else {
        F16Bits::ZERO
    };
    let dmin = if maximum_min > 0.0 {
        F16Bits::from_f32(maximum_min / 15.0)
    } else {
        F16Bits::ZERO
    };
    if !d.to_f32().is_finite() || !dmin.to_f32().is_finite() {
        return Err(FormatError::NonFiniteValue);
    }

    let mut scales = [0_u8; Q2K_SCALE_BYTES];
    for group in 0..Q2K_GROUPS {
        let scale_level = if maximum_scale > 0.0 {
            nearest_int(15.0 / maximum_scale * group_scales[group]).clamp(0, 15)
        } else {
            0
        };
        let min_level = if maximum_min > 0.0 {
            nearest_int(15.0 / maximum_min * group_mins[group]).clamp(0, 15)
        } else {
            0
        };
        scales[group] = (scale_level as u8) | ((min_level as u8) << 4);

        let effective_scale = d.to_f32() * (scale_level as f32);
        if effective_scale != 0.0 {
            let effective_min = dmin.to_f32() * (min_level as f32);
            for lane in 0..Q2K_GROUP_SIZE {
                let index = group * Q2K_GROUP_SIZE + lane;
                levels[index] = nearest_int((source[index] + effective_min) / effective_scale)
                    .clamp(0, 3) as u8;
            }
        }
    }

    let mut qs = [0_u8; Q2K_QS_BYTES];
    for half in 0..2 {
        let start = half * 128;
        for lane in 0..32 {
            qs[half * 32 + lane] = levels[start + lane]
                | (levels[start + lane + 32] << 2)
                | (levels[start + lane + 64] << 4)
                | (levels[start + lane + 96] << 6);
        }
    }
    Ok(Q2KCompactBlock {
        scales,
        qs,
        d,
        dmin,
    })
}

pub fn quantize_q2k_tensor_ref(
    source: &[f32],
    rows: usize,
    cols: usize,
) -> Result<Vec<Q2KCompactBlock>, FormatError> {
    if rows == 0 || cols == 0 || !cols.is_multiple_of(Q2K_BLOCK_SIZE) {
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
        .chunks_exact(Q2K_BLOCK_SIZE)
        .map(quantize_q2k_block_ref)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_layout_and_known_decode_are_exact() {
        assert_eq!(core::mem::size_of::<Q2KCompactBlock>(), Q2K_BLOCK_BYTES);
        let mut block = Q2KCompactBlock::ZERO;
        block.d = F16Bits::from_f32(1.0);
        block.dmin = F16Bits::from_f32(0.5);
        block.scales[0] = 0x12;
        block.scales[1] = 0x03;
        for lane in 0..32 {
            block.qs[lane] = 0x01;
        }
        let wire = block.to_le_bytes();
        assert_eq!(Q2KCompactBlock::from_le_bytes(&wire), Ok(block.clone()));
        let mut decoded = [0.0_f32; Q2K_BLOCK_SIZE];
        block.decode_ref(&mut decoded).unwrap();
        assert_eq!(decoded[0], 1.5);
        assert_eq!(decoded[15], 1.5);
        assert_eq!(decoded[16], 3.0);
    }

    #[test]
    fn quantization_is_deterministic_and_bounded() {
        let source: [f32; Q2K_BLOCK_SIZE] = core::array::from_fn(|index| {
            let x = index as f32 - 127.5;
            (x * 0.03125).sin() * (1.0 + (index % 7) as f32)
        });
        let first = quantize_q2k_block_ref(&source).unwrap();
        let second = quantize_q2k_block_ref(&source).unwrap();
        assert_eq!(first, second);
        let mut decoded = [0.0_f32; Q2K_BLOCK_SIZE];
        first.decode_ref(&mut decoded).unwrap();
        assert!(decoded.iter().all(|value| value.is_finite()));
        assert!(first.qs.iter().any(|value| *value != 0));
    }

    #[test]
    fn llama_cpp_b10001_q2k_packing_vector_is_bit_exact() {
        // Independently derived from ggml-quants.c::quantize_row_q2_K_ref.
        let source: [f32; Q2K_BLOCK_SIZE] = core::array::from_fn(|index| (index % 4) as f32);
        let block = quantize_q2k_block_ref(&source).unwrap();
        assert_eq!(block.scales, [0x0f; Q2K_SCALE_BYTES]);
        assert_eq!(
            block.qs,
            [
                0x00, 0x55, 0xaa, 0xff, 0x00, 0x55, 0xaa, 0xff, 0x00, 0x55, 0xaa, 0xff, 0x00, 0x55,
                0xaa, 0xff, 0x00, 0x55, 0xaa, 0xff, 0x00, 0x55, 0xaa, 0xff, 0x00, 0x55, 0xaa, 0xff,
                0x00, 0x55, 0xaa, 0xff, 0x00, 0x55, 0xaa, 0xff, 0x00, 0x55, 0xaa, 0xff, 0x00, 0x55,
                0xaa, 0xff, 0x00, 0x55, 0xaa, 0xff, 0x00, 0x55, 0xaa, 0xff, 0x00, 0x55, 0xaa, 0xff,
                0x00, 0x55, 0xaa, 0xff, 0x00, 0x55, 0xaa, 0xff,
            ]
        );
        assert_eq!(block.d.to_bits(), 0x2c44);
        assert_eq!(block.dmin.to_bits(), 0);
    }

    #[test]
    fn llama_cpp_b10001_q2k_min_vector_is_bit_exact() {
        let block = quantize_q2k_block_ref(&[-1.0; Q2K_BLOCK_SIZE]).unwrap();
        assert_eq!(block.scales, [0xf0; Q2K_SCALE_BYTES]);
        assert_eq!(block.qs, [0; Q2K_QS_BYTES]);
        assert_eq!(block.d.to_bits(), 0);
        assert_eq!(block.dmin.to_bits(), 0x2c44);
    }

    #[test]
    fn invalid_inputs_are_rejected_before_output_exists() {
        assert_eq!(
            quantize_q2k_block_ref(&[0.0; 255]),
            Err(FormatError::InputLengthMismatch {
                expected: 256,
                actual: 255
            })
        );
        let mut source = [0.0; Q2K_BLOCK_SIZE];
        source[99] = f32::INFINITY;
        assert_eq!(
            quantize_q2k_block_ref(&source),
            Err(FormatError::NonFiniteInput { index: 99 })
        );
    }
}
