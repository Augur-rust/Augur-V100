//! GGUF-compatible IQ2_XXS weight-format semantics.

use augur_core::{F16Bits, FormatVersion, PhysicalLayoutId, QuantFormatId};
use std::sync::OnceLock;

use crate::{FormatError, ReferenceDecode};

pub const IQ2_XXS_BLOCK_SIZE: usize = 256;
pub const IQ2_XXS_GROUPS: usize = 8;
pub const IQ2_XXS_GROUP_SIZE: usize = 32;
pub const IQ2_XXS_CODES: usize = 32;
pub const IQ2_XXS_BLOCK_BYTES: usize = 66;
pub const IQ2_XXS_FORMAT_ID: QuantFormatId = QuantFormatId::IQ2_XXS;
pub const IQ2_XXS_FORMAT_VERSION: FormatVersion = FormatVersion(1);
pub const IQ2_XXS_COMPACT_LAYOUT: PhysicalLayoutId = PhysicalLayoutId::IQ2_XXS_COMPACT_V1;

/// Allowed GGML IQ2_XXS magnitude grids, encoded as eight 2-bit levels.
///
/// The source is the MIT-licensed DwarfStar/GGML-compatible `kgrid` table.
pub const IQ2_XXS_GRID_CODES: [u16; 256] = [
    0, 2, 5, 8, 10, 17, 20, 32, 34, 40, 42, 65, 68, 80, 88, 97, 100, 128, 130, 138, 162, 257, 260,
    272, 277, 320, 388, 408, 512, 514, 546, 642, 1025, 1028, 1040, 1057, 1060, 1088, 1090, 1096,
    1120, 1153, 1156, 1168, 1188, 1280, 1282, 1288, 1312, 1350, 1385, 1408, 1425, 1545, 1552, 1600,
    1668, 1700, 2048, 2053, 2056, 2068, 2088, 2113, 2116, 2128, 2130, 2184, 2308, 2368, 2562, 2580,
    4097, 4100, 4112, 4129, 4160, 4192, 4228, 4240, 4245, 4352, 4360, 4384, 4432, 4442, 4480, 4644,
    4677, 5120, 5128, 5152, 5157, 5193, 5248, 5400, 5474, 5632, 5654, 6145, 6148, 6160, 6208, 6273,
    6400, 6405, 6560, 6737, 8192, 8194, 8202, 8260, 8289, 8320, 8322, 8489, 8520, 8704, 8706, 9217,
    9220, 9232, 9280, 9302, 9472, 9537, 9572, 9872, 10248, 10272, 10388, 10820, 16385, 16388,
    16400, 16408, 16417, 16420, 16448, 16456, 16470, 16480, 16513, 16516, 16528, 16640, 16672,
    16737, 16768, 16773, 16897, 16912, 16968, 16982, 17000, 17408, 17416, 17440, 17536, 17561,
    17682, 17700, 17920, 18433, 18436, 18448, 18496, 18501, 18688, 18776, 18785, 18818, 19013,
    19088, 20480, 20488, 20497, 20505, 20512, 20608, 20616, 20740, 20802, 20900, 21137, 21648,
    21650, 21770, 22017, 22100, 22528, 22545, 22553, 22628, 22848, 23048, 24580, 24592, 24640,
    24680, 24832, 24917, 25112, 25184, 25600, 25605, 25872, 25874, 25988, 26690, 32768, 32770,
    32778, 32833, 32898, 33028, 33048, 33088, 33297, 33793, 33796, 33808, 33813, 33856, 33888,
    34048, 34118, 34196, 34313, 34368, 34400, 34818, 35076, 35345, 36868, 36880, 36900, 36928,
    37025, 37142, 37248, 37445, 37888, 37922, 37956, 38225, 39041, 39200, 40962, 41040, 41093,
    41225, 41472, 42008, 43088, 43268,
];

fn nearest_int(value: f32) -> i32 {
    debug_assert!(value.abs() <= 4_194_303.0);
    let bits = (value + 12_582_912.0).to_bits();
    ((bits & 0x007f_ffff) as i32) - 0x0040_0000
}

fn grid_levels(grid_index: usize) -> [u8; 8] {
    let code = IQ2_XXS_GRID_CODES[grid_index];
    core::array::from_fn(|lane| ((code >> (2 * lane)) & 0x03) as u8)
}

struct GridSearch {
    direct: Vec<i16>,
    neighbours: Vec<Box<[u16]>>,
}

fn grid_search() -> &'static GridSearch {
    static SEARCH: OnceLock<GridSearch> = OnceLock::new();
    SEARCH.get_or_init(|| {
        const MAP_SIZE: usize = 43_692;
        let mut direct = vec![-1_i16; MAP_SIZE];
        for (index, code) in IQ2_XXS_GRID_CODES.into_iter().enumerate() {
            direct[code as usize] = index as i16;
        }
        let mut neighbours = Vec::with_capacity(MAP_SIZE);
        for code in 0..MAP_SIZE {
            if direct[code] >= 0 {
                neighbours.push(Box::default());
                continue;
            }
            let proposed: [u8; 8] =
                core::array::from_fn(|lane| ((code >> (2 * lane)) & 0x03) as u8);
            let distances: [i32; 256] = core::array::from_fn(|index| {
                grid_levels(index)
                    .into_iter()
                    .zip(proposed)
                    .map(|(grid, candidate)| {
                        let delta = 2 * (grid as i32 - candidate as i32);
                        delta * delta
                    })
                    .sum()
            });
            let first = distances.iter().copied().min().unwrap();
            let second = distances
                .iter()
                .copied()
                .filter(|distance| *distance > first)
                .min()
                .unwrap();
            let selected = distances
                .iter()
                .enumerate()
                .filter_map(|(index, distance)| (*distance == first).then_some(index as u16))
                .chain(
                    distances
                        .iter()
                        .enumerate()
                        .filter_map(|(index, distance)| {
                            (*distance == second).then_some(index as u16)
                        }),
                )
                .collect::<Vec<_>>()
                .into_boxed_slice();
            neighbours.push(selected);
        }
        GridSearch { direct, neighbours }
    })
}

fn direct_grid_index(code: u16) -> Option<usize> {
    let index = grid_search().direct[code as usize];
    (index >= 0).then_some(index as usize)
}

fn nearest_grid_index(code: u16, values: &[f32], weights: &[f32], scale: f32) -> usize {
    let mut best_index = 0;
    let mut best_error = f32::MAX;
    for &index in &grid_search().neighbours[code as usize] {
        let index = index as usize;
        let levels = grid_levels(index);
        let mut error = 0.0_f32;
        for lane in 0..8 {
            let quant = (2 * levels[lane] + 1) as f32;
            let difference = scale * quant - values[lane];
            error += weights[lane] * difference * difference;
        }
        if error < best_error {
            best_error = error;
            best_index = index;
        }
    }
    best_index
}

fn make_qp_quants(values: &[f32; 32], levels: &mut [u8; 32], weights: &[f32; 32]) -> f32 {
    let maximum = values.iter().copied().fold(0.0_f32, f32::max);
    if maximum < 1.0e-15 {
        levels.fill(0);
        return 0.0;
    }
    let mut inverse_scale = 4.0 / maximum;
    for index in 0..32 {
        levels[index] = nearest_int(inverse_scale * values[index]) as u8;
    }
    let mut best_error = 0.0_f32;
    let scale = 1.0 / inverse_scale;
    for index in 0..32 {
        let difference = values[index] - scale * levels[index] as f32;
        best_error += weights[index] * difference * difference;
    }
    for step in -4..=4 {
        if step == 0 {
            continue;
        }
        let candidate_inverse = (0.1 * step as f32 + 4.0) / maximum;
        let candidate_scale = 1.0 / candidate_inverse;
        let mut error = 0.0_f32;
        for index in 0..32 {
            let level = nearest_int(candidate_inverse * values[index]).min(4);
            let difference = values[index] - candidate_scale * level as f32;
            error += weights[index] * difference * difference;
        }
        if error < best_error {
            best_error = error;
            inverse_scale = candidate_inverse;
        }
    }

    let mut sum_level_value = 0.0_f32;
    let mut sum_level_squared = 0.0_f32;
    for index in 0..32 {
        let level = nearest_int(inverse_scale * values[index]).min(4) as u8;
        levels[index] = level;
        let weight = weights[index];
        sum_level_value += weight * values[index] * level as f32;
        sum_level_squared += weight * (level as f32) * (level as f32);
    }
    for _ in 0..5 {
        let mut changed = 0;
        for index in 0..32 {
            let weight = weights[index];
            let old_level = levels[index] as f32;
            let mut candidate_lx = sum_level_value - weight * values[index] * old_level;
            let mut candidate_l2 = sum_level_squared - weight * old_level * old_level;
            if candidate_lx > 0.0 && candidate_l2 > 0.0 {
                let new_level = nearest_int(values[index] * candidate_l2 / candidate_lx).min(4);
                if new_level != levels[index] as i32 {
                    candidate_lx += weight * values[index] * new_level as f32;
                    candidate_l2 += weight * (new_level as f32) * (new_level as f32);
                    if candidate_lx * candidate_lx * sum_level_squared
                        > sum_level_value * sum_level_value * candidate_l2
                    {
                        levels[index] = new_level as u8;
                        sum_level_value = candidate_lx;
                        sum_level_squared = candidate_l2;
                        changed += 1;
                    }
                }
            }
        }
        if changed == 0 {
            break;
        }
    }
    if sum_level_squared > 0.0 {
        sum_level_value / sum_level_squared
    } else {
        0.0
    }
}

pub fn quantize_iq2_xxs_block_weighted_ref(
    source: &[f32],
    importance: &[f32],
) -> Result<Iq2XxsCompactBlock, FormatError> {
    if source.len() != IQ2_XXS_BLOCK_SIZE {
        return Err(FormatError::InputLengthMismatch {
            expected: IQ2_XXS_BLOCK_SIZE,
            actual: source.len(),
        });
    }
    if importance.len() != IQ2_XXS_BLOCK_SIZE {
        return Err(FormatError::InputLengthMismatch {
            expected: IQ2_XXS_BLOCK_SIZE,
            actual: importance.len(),
        });
    }
    for (index, value) in source.iter().enumerate() {
        if !value.is_finite() {
            return Err(FormatError::NonFiniteInput { index });
        }
    }
    if importance
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(FormatError::BadBlock);
    }

    let sigma2 = source.iter().map(|value| value * value).sum::<f32>() / 256.0;
    let mut packed = [0_u32; 16];
    let mut scales = [0.0_f32; 8];
    let mut maximum_scale = 0.0_f32;

    for group in 0..8 {
        let source_group = &source[group * 32..group * 32 + 32];
        let importance_group = &importance[group * 32..group * 32 + 32];
        let mut weights = [0.0_f32; 32];
        let mut neighbour_weights = [0.0_f32; 32];
        let mut values = [0.0_f32; 32];
        let mut levels = [0_u8; 32];
        let mut candidate_levels = [0_u8; 32];
        let mut signs = [0_u8; 4];

        for index in 0..32 {
            weights[index] =
                importance_group[index] * (sigma2 + source_group[index].powi(2)).sqrt();
            neighbour_weights[index] = weights[index].sqrt();
        }
        for sub_group in 0..4 {
            let mut negative_count = 0;
            let mut sign = 0_u8;
            for lane in 0..8 {
                let index = sub_group * 8 + lane;
                let value = source_group[index];
                if value >= 0.0 {
                    values[index] = value;
                } else {
                    values[index] = -value;
                    negative_count += 1;
                    sign |= 1 << lane;
                }
            }
            if negative_count % 2 != 0 {
                let mut minimum_index = sub_group * 8;
                let mut minimum = weights[minimum_index]
                    * source_group[minimum_index]
                    * source_group[minimum_index];
                for lane in 1..8 {
                    let index = sub_group * 8 + lane;
                    let candidate = weights[index] * source_group[index] * source_group[index];
                    if candidate < minimum {
                        minimum = candidate;
                        minimum_index = index;
                    }
                }
                values[minimum_index] = -values[minimum_index];
                sign ^= 1 << (minimum_index - sub_group * 8);
            }
            signs[sub_group] = sign & 0x7f;
        }

        let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if maximum < 1.0e-15 {
            continue;
        }
        let mut scale = make_qp_quants(&values, &mut levels, &weights);
        let effective_maximum = scale * 3.0;
        if effective_maximum <= 0.0 {
            continue;
        }

        let mut best = 0.0_f32;
        for scale_step in -6..=6 {
            let inverse = (5.0 + scale_step as f32 * 0.1) / effective_maximum;
            let candidate_scale = 1.0 / inverse;
            for sub_group in 0..4 {
                let mut code = 0_u16;
                for lane in 0..8 {
                    let index = sub_group * 8 + lane;
                    let level = nearest_int(0.5 * (inverse * values[index] - 1.0)).clamp(0, 2);
                    candidate_levels[index] = level as u8;
                    code |= (level as u16) << (2 * lane);
                }
                if direct_grid_index(code).is_none() {
                    let grid = nearest_grid_index(
                        code,
                        &values[sub_group * 8..sub_group * 8 + 8],
                        &neighbour_weights[sub_group * 8..sub_group * 8 + 8],
                        candidate_scale,
                    );
                    candidate_levels[sub_group * 8..sub_group * 8 + 8]
                        .copy_from_slice(&grid_levels(grid));
                }
            }
            let mut sum_qx = 0.0_f32;
            let mut sum_q2 = 0.0_f32;
            for index in 0..32 {
                let quant = (2 * candidate_levels[index] + 1) as f32;
                sum_qx += weights[index] * values[index] * quant;
                sum_q2 += weights[index] * quant * quant;
            }
            if sum_q2 > 0.0 && sum_qx * sum_qx > best * sum_q2 {
                scale = sum_qx / sum_q2;
                best = scale * sum_qx;
                levels.copy_from_slice(&candidate_levels);
            }
        }

        if scale > 0.0 {
            let inverse = 1.0 / scale;
            for sub_group in 0..4 {
                let mut code = 0_u16;
                for lane in 0..8 {
                    let index = sub_group * 8 + lane;
                    let level = nearest_int(0.5 * (inverse * values[index] - 1.0)).clamp(0, 2);
                    code |= (level as u16) << (2 * lane);
                }
                let grid = direct_grid_index(code).unwrap_or_else(|| {
                    nearest_grid_index(
                        code,
                        &values[sub_group * 8..sub_group * 8 + 8],
                        &neighbour_weights[sub_group * 8..sub_group * 8 + 8],
                        scale,
                    )
                });
                levels[sub_group * 8..sub_group * 8 + 8].copy_from_slice(&grid_levels(grid));
            }
            let mut sum_qx = 0.0_f32;
            let mut sum_q2 = 0.0_f32;
            for index in 0..32 {
                let quant = (2 * levels[index] + 1) as f32;
                sum_qx += weights[index] * values[index] * quant;
                sum_q2 += weights[index] * quant * quant;
            }
            if sum_q2 > 0.0 {
                scale = sum_qx / sum_q2;
            }
        }
        if scale < 0.0 {
            scale = -scale;
            for sign in &mut signs {
                *sign = !*sign & 0x7f;
            }
        }

        for sub_group in 0..4 {
            let mut code = 0_u16;
            for lane in 0..8 {
                code |= (levels[sub_group * 8 + lane] as u16) << (2 * lane);
            }
            let grid = direct_grid_index(code).ok_or(FormatError::BadBlock)?;
            packed[2 * group] |= (grid as u32) << (8 * sub_group);
            packed[2 * group + 1] |= (signs[sub_group] as u32) << (7 * sub_group);
        }
        scales[group] = scale;
        maximum_scale = maximum_scale.max(scale);
    }

    if maximum_scale == 0.0 {
        return Ok(Iq2XxsCompactBlock::ZERO);
    }
    let block_scale = maximum_scale / 31.0;
    let d = F16Bits::from_f32(block_scale);
    if !d.to_f32().is_finite() {
        return Err(FormatError::NonFiniteValue);
    }
    let inverse = 1.0 / block_scale;
    for group in 0..8 {
        let level = nearest_int(0.5 * (inverse * scales[group] - 1.0)).clamp(0, 15);
        packed[2 * group + 1] |= (level as u32) << 28;
    }
    let mut qs = [0_u16; IQ2_XXS_CODES];
    for (index, value) in packed.into_iter().enumerate() {
        let bytes = value.to_le_bytes();
        qs[index * 2] = u16::from_le_bytes([bytes[0], bytes[1]]);
        qs[index * 2 + 1] = u16::from_le_bytes([bytes[2], bytes[3]]);
    }
    Ok(Iq2XxsCompactBlock { d, qs })
}

pub fn quantize_iq2_xxs_block_ref(source: &[f32]) -> Result<Iq2XxsCompactBlock, FormatError> {
    quantize_iq2_xxs_block_weighted_ref(source, &[1.0; IQ2_XXS_BLOCK_SIZE])
}

pub fn quantize_iq2_xxs_tensor_ref(
    source: &[f32],
    rows: usize,
    cols: usize,
) -> Result<Vec<Iq2XxsCompactBlock>, FormatError> {
    let importance = iq2_xxs_weight_energy_importance_ref(source, rows, cols)?;
    quantize_iq2_xxs_tensor_weighted_ref(source, rows, cols, &importance)
}

pub fn iq2_xxs_weight_energy_importance_ref(
    source: &[f32],
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>, FormatError> {
    if rows == 0 || cols == 0 || !cols.is_multiple_of(IQ2_XXS_BLOCK_SIZE) {
        return Err(FormatError::BadLength);
    }
    let expected = rows.checked_mul(cols).ok_or(FormatError::BadLength)?;
    if source.len() != expected {
        return Err(FormatError::InputLengthMismatch {
            expected,
            actual: source.len(),
        });
    }
    let mut importance = vec![0.0_f32; cols];
    for row in source.chunks_exact(cols) {
        for (total, value) in importance.iter_mut().zip(row) {
            if !value.is_finite() {
                return Err(FormatError::NonFiniteValue);
            }
            *total += value * value;
        }
    }
    if importance.iter().any(|value| !value.is_finite()) {
        return Err(FormatError::NonFiniteValue);
    }
    Ok(importance)
}

pub fn quantize_iq2_xxs_tensor_weighted_ref(
    source: &[f32],
    rows: usize,
    cols: usize,
    importance: &[f32],
) -> Result<Vec<Iq2XxsCompactBlock>, FormatError> {
    if rows == 0 || cols == 0 || !cols.is_multiple_of(IQ2_XXS_BLOCK_SIZE) {
        return Err(FormatError::BadLength);
    }
    let expected = rows.checked_mul(cols).ok_or(FormatError::BadLength)?;
    if source.len() != expected {
        return Err(FormatError::InputLengthMismatch {
            expected,
            actual: source.len(),
        });
    }
    if importance.len() != cols {
        return Err(FormatError::InputLengthMismatch {
            expected: cols,
            actual: importance.len(),
        });
    }
    let mut output = Vec::with_capacity(rows * (cols / IQ2_XXS_BLOCK_SIZE));
    for row in source.chunks_exact(cols) {
        for (block, block_importance) in row
            .chunks_exact(IQ2_XXS_BLOCK_SIZE)
            .zip(importance.chunks_exact(IQ2_XXS_BLOCK_SIZE))
        {
            output.push(quantize_iq2_xxs_block_weighted_ref(
                block,
                block_importance,
            )?);
        }
    }
    Ok(output)
}

#[repr(C)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Iq2XxsCompactBlock {
    pub d: F16Bits,
    pub qs: [u16; IQ2_XXS_CODES],
}

impl Iq2XxsCompactBlock {
    pub const DECODED_LEN: usize = IQ2_XXS_BLOCK_SIZE;
    pub const ZERO: Self = Self {
        d: F16Bits::ZERO,
        qs: [0; IQ2_XXS_CODES],
    };

    pub fn to_le_bytes(&self) -> [u8; IQ2_XXS_BLOCK_BYTES] {
        let mut output = [0_u8; IQ2_XXS_BLOCK_BYTES];
        output[..2].copy_from_slice(&self.d.to_bits().to_le_bytes());
        for (index, code) in self.qs.iter().enumerate() {
            output[2 + index * 2..4 + index * 2].copy_from_slice(&code.to_le_bytes());
        }
        output
    }

    pub fn from_le_bytes(bytes: &[u8]) -> Result<Self, FormatError> {
        if bytes.len() != IQ2_XXS_BLOCK_BYTES {
            return Err(FormatError::InputLengthMismatch {
                expected: IQ2_XXS_BLOCK_BYTES,
                actual: bytes.len(),
            });
        }
        let d = F16Bits::from_bits(u16::from_le_bytes([bytes[0], bytes[1]]));
        if !d.to_f32().is_finite() {
            return Err(FormatError::NonFiniteValue);
        }
        let mut qs = [0_u16; IQ2_XXS_CODES];
        for (index, code) in qs.iter_mut().enumerate() {
            let offset = 2 + index * 2;
            *code = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
        }
        Ok(Self { d, qs })
    }

    #[inline]
    pub fn group_metadata(&self, group: usize) -> Option<([u8; 4], u32)> {
        if group >= IQ2_XXS_GROUPS {
            return None;
        }
        let words = &self.qs[group * 4..group * 4 + 4];
        let [grid0, grid1] = words[0].to_le_bytes();
        let [grid2, grid3] = words[1].to_le_bytes();
        let sign_scale = u32::from_le_bytes([
            words[2].to_le_bytes()[0],
            words[2].to_le_bytes()[1],
            words[3].to_le_bytes()[0],
            words[3].to_le_bytes()[1],
        ]);
        Some(([grid0, grid1, grid2, grid3], sign_scale))
    }
}

#[inline]
pub const fn iq2_xxs_sign_mask(sign_index: u8) -> u8 {
    let low = sign_index & 0x7f;
    low | (((low.count_ones() & 1) as u8) << 7)
}

/// Scalar value used by the canonical GGML IQ2_XXS dot table before `/8`.
#[inline]
pub const fn iq2_xxs_grid_value(grid_index: u8, lane: usize) -> i8 {
    let level = (IQ2_XXS_GRID_CODES[grid_index as usize] >> (2 * lane)) & 0x03;
    match level {
        0 => 8,
        1 => 25,
        2 => 43,
        _ => 0,
    }
}

impl ReferenceDecode for Iq2XxsCompactBlock {
    type Elem = f32;

    fn decoded_len(&self) -> usize {
        IQ2_XXS_BLOCK_SIZE
    }

    fn decode_ref(&self, destination: &mut [f32]) -> Result<(), FormatError> {
        if destination.len() < IQ2_XXS_BLOCK_SIZE {
            return Err(FormatError::OutputTooSmall {
                expected: IQ2_XXS_BLOCK_SIZE,
                actual: destination.len(),
            });
        }
        let d = self.d.to_f32();
        if !d.is_finite() {
            return Err(FormatError::NonFiniteValue);
        }
        for group in 0..IQ2_XXS_GROUPS {
            let (grids, sign_scale) = self.group_metadata(group).ok_or(FormatError::BadLength)?;
            let group_scale = (2 * (sign_scale >> 28) + 1) as f32;
            for sub_group in 0..4 {
                let sign_index = ((sign_scale >> (7 * sub_group)) & 0x7f) as u8;
                let sign_mask = iq2_xxs_sign_mask(sign_index);
                for lane in 0..8 {
                    let magnitude = iq2_xxs_grid_value(grids[sub_group], lane) as f32;
                    let sign = if sign_mask & (1 << lane) == 0 {
                        1.0
                    } else {
                        -1.0
                    };
                    destination[group * 32 + sub_group * 8 + lane] =
                        d * group_scale * magnitude * sign * 0.125;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_layout_and_sign_parity_are_exact() {
        assert_eq!(
            core::mem::size_of::<Iq2XxsCompactBlock>(),
            IQ2_XXS_BLOCK_BYTES
        );
        let mut block = Iq2XxsCompactBlock::ZERO;
        block.d = F16Bits::from_f32(1.0);
        block.qs[0] = 0x0100;
        block.qs[2] = 1;
        let wire = block.to_le_bytes();
        assert_eq!(Iq2XxsCompactBlock::from_le_bytes(&wire), Ok(block.clone()));
        assert_eq!(iq2_xxs_sign_mask(0), 0);
        assert_eq!(iq2_xxs_sign_mask(1), 0x81);
        assert_eq!(iq2_xxs_sign_mask(3), 0x03);
    }

    #[test]
    fn known_group_decodes_with_canonical_adjusted_grid() {
        let mut block = Iq2XxsCompactBlock::ZERO;
        block.d = F16Bits::from_f32(1.0);
        block.qs[0] = 0x0200;
        block.qs[2] = 1;
        let mut decoded = [0.0_f32; IQ2_XXS_BLOCK_SIZE];
        block.decode_ref(&mut decoded).unwrap();
        assert_eq!(decoded[0], -1.0);
        assert_eq!(decoded[7], -1.0);
        assert_eq!(decoded[8], 3.125);
        assert_eq!(decoded[15], 1.0);
    }

    #[test]
    fn malformed_wire_and_non_finite_scale_fail_closed() {
        assert!(matches!(
            Iq2XxsCompactBlock::from_le_bytes(&[0; 65]),
            Err(FormatError::InputLengthMismatch { .. })
        ));
        let mut bytes = [0_u8; IQ2_XXS_BLOCK_BYTES];
        bytes[..2].copy_from_slice(&0x7c00_u16.to_le_bytes());
        assert_eq!(
            Iq2XxsCompactBlock::from_le_bytes(&bytes),
            Err(FormatError::NonFiniteValue)
        );
    }

    #[test]
    fn quantization_is_deterministic_finite_and_nontrivial() {
        let source: [f32; IQ2_XXS_BLOCK_SIZE] = core::array::from_fn(|index| {
            let x = index as f32 - 127.5;
            (x * 0.037).sin() * (1.0 + (index % 11) as f32)
        });
        let first = quantize_iq2_xxs_block_ref(&source).unwrap();
        let second = quantize_iq2_xxs_block_ref(&source).unwrap();
        assert_eq!(first, second);
        assert_ne!(first.d, F16Bits::ZERO);
        assert!(first.qs.iter().any(|value| *value != 0));
        let mut decoded = [0.0_f32; IQ2_XXS_BLOCK_SIZE];
        first.decode_ref(&mut decoded).unwrap();
        assert!(decoded.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn quantization_rejects_invalid_source_and_importance() {
        assert!(matches!(
            quantize_iq2_xxs_block_ref(&[0.0; 255]),
            Err(FormatError::InputLengthMismatch { .. })
        ));
        let mut source = [0.0; IQ2_XXS_BLOCK_SIZE];
        source[7] = f32::NAN;
        assert_eq!(
            quantize_iq2_xxs_block_ref(&source),
            Err(FormatError::NonFiniteInput { index: 7 })
        );
        assert_eq!(
            quantize_iq2_xxs_block_weighted_ref(
                &[0.0; IQ2_XXS_BLOCK_SIZE],
                &[-1.0; IQ2_XXS_BLOCK_SIZE]
            ),
            Err(FormatError::BadBlock)
        );
    }

    #[test]
    fn tensor_quantizer_uses_dwarfstar_weight_energy_fallback() {
        let source = (0..2 * IQ2_XXS_BLOCK_SIZE)
            .map(|index| (index % 17) as f32 - 8.0)
            .collect::<Vec<_>>();
        let importance =
            iq2_xxs_weight_energy_importance_ref(&source, 2, IQ2_XXS_BLOCK_SIZE).unwrap();
        for col in 0..IQ2_XXS_BLOCK_SIZE {
            assert_eq!(
                importance[col],
                source[col] * source[col]
                    + source[IQ2_XXS_BLOCK_SIZE + col] * source[IQ2_XXS_BLOCK_SIZE + col]
            );
        }
        assert_eq!(
            quantize_iq2_xxs_tensor_ref(&source, 2, IQ2_XXS_BLOCK_SIZE).unwrap(),
            quantize_iq2_xxs_tensor_weighted_ref(&source, 2, IQ2_XXS_BLOCK_SIZE, &importance)
                .unwrap()
        );
    }
}
