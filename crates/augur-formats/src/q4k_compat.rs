use augur_core::{FormatVersion, PhysicalLayoutId, QuantFormatId};

use crate::{FormatError, ReferenceDecode};
use augur_core::F16Bits;

pub const Q4K_BLOCK_SIZE: usize = 256;
pub const Q4K_GROUPS: usize = 8;
pub const Q4K_GROUP_SIZE: usize = 32;
pub const Q4K_SCALE_BYTES: usize = 12;
pub const Q4K_QS_BYTES: usize = Q4K_BLOCK_SIZE / 2;
pub const Q4K_BLOCK_BYTES: usize = 2 * size_of::<u16>() + Q4K_SCALE_BYTES + Q4K_QS_BYTES;

pub const Q4K_FORMAT_ID: QuantFormatId = QuantFormatId::Q4K;
pub const Q4K_FORMAT_VERSION: FormatVersion = FormatVersion(1);
pub const Q4K_COMPACT_LAYOUT: PhysicalLayoutId = PhysicalLayoutId::Q4K_COMPACT_V1;

#[repr(C)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Q4KCompactBlock {
    pub d: F16Bits,
    pub dmin: F16Bits,
    pub scales: [u8; Q4K_SCALE_BYTES],
    pub qs: [u8; Q4K_QS_BYTES],
}

impl Q4KCompactBlock {
    pub const DECODED_LEN: usize = Q4K_BLOCK_SIZE;

    pub const ZERO: Self = Self {
        d: F16Bits::ZERO,
        dmin: F16Bits::ZERO,
        scales: [0; Q4K_SCALE_BYTES],
        qs: [0; Q4K_QS_BYTES],
    };

    pub fn scales_ref(&self) -> Q4KScales {
        unpack_q4k_scales_ref(&self.scales)
    }

    #[inline]
    pub fn quant_at(&self, index: usize) -> u8 {
        debug_assert!(index < Q4K_BLOCK_SIZE);
        let chunk = index / 64;
        let within_chunk = index % 64;
        let packed = self.qs[chunk * 32 + within_chunk % 32];
        if within_chunk < 32 {
            packed & 0x0f
        } else {
            packed >> 4
        }
    }

    pub fn to_le_bytes(&self) -> [u8; Q4K_BLOCK_BYTES] {
        let mut output = [0_u8; Q4K_BLOCK_BYTES];
        output[..2].copy_from_slice(&self.d.to_bits().to_le_bytes());
        output[2..4].copy_from_slice(&self.dmin.to_bits().to_le_bytes());
        output[4..16].copy_from_slice(&self.scales);
        output[16..].copy_from_slice(&self.qs);
        output
    }

    pub fn from_le_bytes(bytes: &[u8]) -> Result<Self, FormatError> {
        if bytes.len() != Q4K_BLOCK_BYTES {
            return Err(FormatError::InputLengthMismatch {
                expected: Q4K_BLOCK_BYTES,
                actual: bytes.len(),
            });
        }
        let d = F16Bits::from_bits(u16::from_le_bytes([bytes[0], bytes[1]]));
        let dmin = F16Bits::from_bits(u16::from_le_bytes([bytes[2], bytes[3]]));
        if !d.to_f32().is_finite() || !dmin.to_f32().is_finite() {
            return Err(FormatError::NonFiniteValue);
        }
        let mut scales = [0_u8; Q4K_SCALE_BYTES];
        scales.copy_from_slice(&bytes[4..16]);
        let mut qs = [0_u8; Q4K_QS_BYTES];
        qs.copy_from_slice(&bytes[16..]);
        Ok(Self {
            d,
            dmin,
            scales,
            qs,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Q4KScales {
    pub scale: [u8; Q4K_GROUPS],
    pub min: [u8; Q4K_GROUPS],
}

impl Q4KScales {
    pub fn validate(self) -> Result<(), FormatError> {
        for (i, &v) in self.scale.iter().enumerate() {
            if v > 63 {
                return Err(FormatError::ScaleOutOfRange { index: i, value: v });
            }
        }
        for (i, &v) in self.min.iter().enumerate() {
            if v > 63 {
                return Err(FormatError::MinOutOfRange { index: i, value: v });
            }
        }
        Ok(())
    }
}

pub fn unpack_q4k_scales_ref(scales: &[u8; Q4K_SCALE_BYTES]) -> Q4KScales {
    let mut scale = [0u8; Q4K_GROUPS];
    let mut min = [0u8; Q4K_GROUPS];

    for j in 0..4 {
        scale[j] = scales[j] & 0x3f;
        min[j] = scales[j + 4] & 0x3f;
    }

    for j in 4..8 {
        scale[j] = (scales[j + 4] & 0x0f) | ((scales[j - 4] >> 6) << 4);
        min[j] = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
    }

    Q4KScales { scale, min }
}

pub fn pack_q4k_scales_ref(decoded: Q4KScales) -> Result<[u8; Q4K_SCALE_BYTES], FormatError> {
    decoded.validate()?;

    let mut scales = [0u8; Q4K_SCALE_BYTES];

    for j in 0..4 {
        scales[j] = (decoded.scale[j] & 0x3f) | ((decoded.scale[j + 4] >> 4) << 6);
        scales[j + 4] = (decoded.min[j] & 0x3f) | ((decoded.min[j + 4] >> 4) << 6);
        scales[j + 8] = (decoded.scale[j + 4] & 0x0f) | ((decoded.min[j + 4] & 0x0f) << 4);
    }

    Ok(scales)
}

fn nearest_int(value: f32) -> i32 {
    debug_assert!(value.abs() <= 4_194_303.0);
    let bits = (value + 12_582_912.0).to_bits();
    ((bits & 0x007f_ffff) as i32) - 0x0040_0000
}

fn make_q4k_group_quants(
    values: &[f32; Q4K_GROUP_SIZE],
    levels: &mut [u8; Q4K_GROUP_SIZE],
) -> (f32, f32) {
    let mut sum_x2 = 0.0_f32;
    for value in values {
        sum_x2 += value * value;
    }
    let average = (sum_x2 / Q4K_GROUP_SIZE as f32).sqrt();
    let weights = values.map(|value| average + value.abs());
    let mut minimum = values[0];
    let mut maximum = values[0];
    let mut sum_weight = weights[0];
    let mut sum_x = weights[0] * values[0];
    for index in 1..Q4K_GROUP_SIZE {
        minimum = minimum.min(values[index]);
        maximum = maximum.max(values[index]);
        sum_weight += weights[index];
        sum_x += weights[index] * values[index];
    }
    minimum = minimum.min(0.0);
    if maximum == minimum {
        levels.fill(0);
        return (0.0, -minimum);
    }

    let mut inverse_scale = 15.0 / (maximum - minimum);
    let mut scale = 1.0 / inverse_scale;
    let mut best_error = 0.0_f32;
    for index in 0..Q4K_GROUP_SIZE {
        let level = nearest_int(inverse_scale * (values[index] - minimum)).clamp(0, 15);
        levels[index] = level as u8;
        let difference = scale * (level as f32) + minimum - values[index];
        best_error += weights[index] * difference * difference;
    }

    let mut candidate = [0_u8; Q4K_GROUP_SIZE];
    for step in 0..=20 {
        inverse_scale = (-1.0 + 0.1 * (step as f32) + 15.0) / (maximum - minimum);
        let mut sum_l = 0.0_f32;
        let mut sum_l2 = 0.0_f32;
        let mut sum_xl = 0.0_f32;
        for index in 0..Q4K_GROUP_SIZE {
            let level = nearest_int(inverse_scale * (values[index] - minimum)).clamp(0, 15);
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
        for index in 0..Q4K_GROUP_SIZE {
            let difference =
                candidate_scale * (candidate[index] as f32) + candidate_minimum - values[index];
            candidate_error += weights[index] * difference * difference;
        }
        if candidate_error < best_error {
            levels.copy_from_slice(&candidate);
            best_error = candidate_error;
            scale = candidate_scale;
            minimum = candidate_minimum;
        }
    }
    (scale, -minimum)
}

pub fn quantize_q4k_block_ref(source: &[f32]) -> Result<Q4KCompactBlock, FormatError> {
    if source.len() != Q4K_BLOCK_SIZE {
        return Err(FormatError::InputLengthMismatch {
            expected: Q4K_BLOCK_SIZE,
            actual: source.len(),
        });
    }
    for (index, value) in source.iter().enumerate() {
        if !value.is_finite() {
            return Err(FormatError::NonFiniteInput { index });
        }
    }
    let mut levels = [0_u8; Q4K_BLOCK_SIZE];
    let mut group_scales = [0.0_f32; Q4K_GROUPS];
    let mut group_mins = [0.0_f32; Q4K_GROUPS];
    for group in 0..Q4K_GROUPS {
        let values: [f32; Q4K_GROUP_SIZE] = source
            [group * Q4K_GROUP_SIZE..(group + 1) * Q4K_GROUP_SIZE]
            .try_into()
            .expect("fixed Q4_K group");
        let group_levels: &mut [u8; Q4K_GROUP_SIZE] = levels
            .get_mut(group * Q4K_GROUP_SIZE..(group + 1) * Q4K_GROUP_SIZE)
            .expect("fixed Q4_K group")
            .try_into()
            .expect("fixed Q4_K group");
        (group_scales[group], group_mins[group]) = make_q4k_group_quants(&values, group_levels);
    }
    let maximum_scale = group_scales.iter().copied().fold(0.0_f32, f32::max);
    let maximum_min = group_mins.iter().copied().fold(0.0_f32, f32::max);
    let inverse_scale = if maximum_scale > 0.0 {
        63.0 / maximum_scale
    } else {
        0.0
    };
    let inverse_min = if maximum_min > 0.0 {
        63.0 / maximum_min
    } else {
        0.0
    };
    let mut scale_levels = [0_u8; Q4K_GROUPS];
    let mut min_levels = [0_u8; Q4K_GROUPS];
    for group in 0..Q4K_GROUPS {
        scale_levels[group] = nearest_int(inverse_scale * group_scales[group]).clamp(0, 63) as u8;
        min_levels[group] = nearest_int(inverse_min * group_mins[group]).clamp(0, 63) as u8;
    }
    let scales = pack_q4k_scales_ref(Q4KScales {
        scale: scale_levels,
        min: min_levels,
    })?;
    let d = if maximum_scale > 0.0 {
        F16Bits::from_f32(maximum_scale / 63.0)
    } else {
        F16Bits::ZERO
    };
    let dmin = if maximum_min > 0.0 {
        F16Bits::from_f32(maximum_min / 63.0)
    } else {
        F16Bits::ZERO
    };
    if !d.to_f32().is_finite() || !dmin.to_f32().is_finite() {
        return Err(FormatError::NonFiniteValue);
    }
    for group in 0..Q4K_GROUPS {
        let effective_scale = d.to_f32() * (scale_levels[group] as f32);
        if effective_scale == 0.0 {
            continue;
        }
        let effective_min = dmin.to_f32() * (min_levels[group] as f32);
        for lane in 0..Q4K_GROUP_SIZE {
            let index = group * Q4K_GROUP_SIZE + lane;
            levels[index] =
                nearest_int((source[index] + effective_min) / effective_scale).clamp(0, 15) as u8;
        }
    }
    let mut qs = [0_u8; Q4K_QS_BYTES];
    for chunk in 0..4 {
        let start = chunk * 64;
        for lane in 0..32 {
            qs[chunk * 32 + lane] = levels[start + lane] | (levels[start + lane + 32] << 4);
        }
    }
    Ok(Q4KCompactBlock {
        d,
        dmin,
        scales,
        qs,
    })
}

pub fn quantize_q4k_tensor_ref(
    source: &[f32],
    rows: usize,
    cols: usize,
) -> Result<Vec<Q4KCompactBlock>, FormatError> {
    if rows == 0 || cols == 0 || !cols.is_multiple_of(Q4K_BLOCK_SIZE) {
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
        .chunks_exact(Q4K_BLOCK_SIZE)
        .map(quantize_q4k_block_ref)
        .collect()
}

impl ReferenceDecode for Q4KCompactBlock {
    type Elem = f32;

    fn decoded_len(&self) -> usize {
        Self::DECODED_LEN
    }

    fn decode_ref(&self, dst: &mut [Self::Elem]) -> Result<(), FormatError> {
        if dst.len() < Self::DECODED_LEN {
            return Err(FormatError::OutputTooSmall {
                expected: Self::DECODED_LEN,
                actual: dst.len(),
            });
        }

        let scales = self.scales_ref();
        let d = self.d.to_f32();
        let dmin = self.dmin.to_f32();

        for (i, out) in dst.iter_mut().take(Self::DECODED_LEN).enumerate() {
            let group = i / Q4K_GROUP_SIZE;
            let q = self.quant_at(i);

            *out =
                d * (scales.scale[group] as f32) * (q as f32) - dmin * (scales.min[group] as f32);
        }

        Ok(())
    }
}
