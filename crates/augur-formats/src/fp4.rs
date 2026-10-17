//! M46 floating-point compact format semantics.

use augur_core::{FormatVersion, PhysicalLayoutId, QuantFormatId};

use crate::{FormatError, ReferenceDecode};

pub const E2M1_VALUES: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

pub const MXFP4_BLOCK_SIZE: usize = 32;
pub const MXFP4_PACKED_BYTES: usize = 16;
pub const MXFP4_FORMAT_ID: QuantFormatId = QuantFormatId::MXFP4;
pub const MXFP4_FORMAT_VERSION: FormatVersion = FormatVersion(1);
pub const MXFP4_COMPACT_LAYOUT: PhysicalLayoutId = PhysicalLayoutId::MXFP4_COMPACT_V1;

pub const NVFP4_BLOCK_SIZE: usize = 16;
pub const NVFP4_PACKED_BYTES: usize = 8;
pub const NVFP4_FORMAT_ID: QuantFormatId = QuantFormatId::NVFP4;
pub const NVFP4_FORMAT_VERSION: FormatVersion = FormatVersion(1);
pub const NVFP4_COMPACT_LAYOUT: PhysicalLayoutId = PhysicalLayoutId::NVFP4_COMPACT_V1;

pub const FP8_E4M3_BLOCK_SIZE: usize = 128;
pub const FP8_E4M3_FORMAT_ID: QuantFormatId = QuantFormatId::FP8E4M3Block;
pub const FP8_E4M3_FORMAT_VERSION: FormatVersion = FormatVersion(1);
pub const FP8_E4M3_COMPACT_LAYOUT: PhysicalLayoutId = PhysicalLayoutId::FP8_E4M3_BLOCK_COMPACT_V1;

pub const FP8_E4M3_SCALE_TILE: usize = 128;
pub const FP8_E4M3_SCALE_GRID_FORMAT_ID: QuantFormatId = QuantFormatId::FP8E4M3ScaleGrid;
pub const FP8_E4M3_SCALE_GRID_FORMAT_VERSION: FormatVersion = FormatVersion(1);
pub const FP8_E4M3_SCALE_GRID_COMPACT_LAYOUT: PhysicalLayoutId =
    PhysicalLayoutId::FP8_E4M3_SCALE_GRID_COMPACT_V1;

#[inline]
pub fn decode_e2m1(value: u8) -> f32 {
    E2M1_VALUES[(value & 0x0f) as usize]
}

pub fn decode_e8m0(value: u8) -> Result<f32, FormatError> {
    if value == u8::MAX {
        return Err(FormatError::NonFiniteValue);
    }
    Ok(2.0_f32.powi(i32::from(value) - 127))
}

pub fn decode_e4m3fn(value: u8) -> Result<f32, FormatError> {
    let sign = if value & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = (value >> 3) & 0x0f;
    let mantissa = value & 0x07;
    if exponent == 0x0f && mantissa == 0x07 {
        return Err(FormatError::NonFiniteValue);
    }
    let magnitude = if exponent == 0 {
        2.0_f32.powi(-6) * (f32::from(mantissa) / 8.0)
    } else {
        2.0_f32.powi(i32::from(exponent) - 7) * (1.0 + f32::from(mantissa) / 8.0)
    };
    Ok(sign * magnitude)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Mxfp4CompactBlock {
    pub qs: [u8; MXFP4_PACKED_BYTES],
    pub scale_e8m0: u8,
}

impl ReferenceDecode for Mxfp4CompactBlock {
    type Elem = f32;

    fn decoded_len(&self) -> usize {
        MXFP4_BLOCK_SIZE
    }

    fn decode_ref(&self, destination: &mut [f32]) -> Result<(), FormatError> {
        if destination.len() < MXFP4_BLOCK_SIZE {
            return Err(FormatError::OutputTooSmall {
                expected: MXFP4_BLOCK_SIZE,
                actual: destination.len(),
            });
        }
        let scale = decode_e8m0(self.scale_e8m0)?;
        for (index, output) in destination.iter_mut().take(MXFP4_BLOCK_SIZE).enumerate() {
            let byte = self.qs[index / 2];
            let nibble = if index % 2 == 0 { byte } else { byte >> 4 };
            *output = decode_e2m1(nibble) * scale;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Nvfp4CompactBlock {
    pub qs: [u8; NVFP4_PACKED_BYTES],
    pub scale_e4m3: u8,
    pub global_scale: f32,
}

impl ReferenceDecode for Nvfp4CompactBlock {
    type Elem = f32;

    fn decoded_len(&self) -> usize {
        NVFP4_BLOCK_SIZE
    }

    fn decode_ref(&self, destination: &mut [f32]) -> Result<(), FormatError> {
        if destination.len() < NVFP4_BLOCK_SIZE {
            return Err(FormatError::OutputTooSmall {
                expected: NVFP4_BLOCK_SIZE,
                actual: destination.len(),
            });
        }
        if !self.global_scale.is_finite() {
            return Err(FormatError::NonFiniteValue);
        }
        let scale = decode_e4m3fn(self.scale_e4m3)? * self.global_scale;
        if !scale.is_finite() {
            return Err(FormatError::NonFiniteValue);
        }
        for (index, output) in destination.iter_mut().take(NVFP4_BLOCK_SIZE).enumerate() {
            let byte = self.qs[index / 2];
            let nibble = if index % 2 == 0 { byte } else { byte >> 4 };
            *output = decode_e2m1(nibble) * scale;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Fp8E4M3CompactBlock {
    pub qs: [u8; FP8_E4M3_BLOCK_SIZE],
    pub scale_e8m0: u8,
}

impl ReferenceDecode for Fp8E4M3CompactBlock {
    type Elem = f32;

    fn decoded_len(&self) -> usize {
        FP8_E4M3_BLOCK_SIZE
    }

    fn decode_ref(&self, destination: &mut [f32]) -> Result<(), FormatError> {
        if destination.len() < FP8_E4M3_BLOCK_SIZE {
            return Err(FormatError::OutputTooSmall {
                expected: FP8_E4M3_BLOCK_SIZE,
                actual: destination.len(),
            });
        }
        let scale = decode_e8m0(self.scale_e8m0)?;
        for (output, value) in destination.iter_mut().zip(self.qs.iter()) {
            *output = decode_e4m3fn(*value)? * scale;
        }
        Ok(())
    }
}

/// Source-faithful FP8 tensor with one E8M0 scale per 128x128 tile.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Fp8E4M3ScaleGridCompactTensor {
    rows: usize,
    cols: usize,
    scale_rows: usize,
    scale_cols: usize,
    values: Vec<u8>,
    scales_e8m0: Vec<u8>,
}

impl Fp8E4M3ScaleGridCompactTensor {
    pub fn new(
        rows: usize,
        cols: usize,
        values: Vec<u8>,
        scales_e8m0: Vec<u8>,
    ) -> Result<Self, FormatError> {
        if rows == 0 || cols == 0 || !cols.is_multiple_of(FP8_E4M3_SCALE_TILE) {
            return Err(FormatError::BadLength);
        }
        let value_count = rows.checked_mul(cols).ok_or(FormatError::BadLength)?;
        if values.len() != value_count {
            return Err(FormatError::InputLengthMismatch {
                expected: value_count,
                actual: values.len(),
            });
        }
        let scale_rows = rows.div_ceil(FP8_E4M3_SCALE_TILE);
        let scale_cols = cols / FP8_E4M3_SCALE_TILE;
        let scale_count = scale_rows
            .checked_mul(scale_cols)
            .ok_or(FormatError::BadLength)?;
        if scales_e8m0.len() != scale_count {
            return Err(FormatError::InputLengthMismatch {
                expected: scale_count,
                actual: scales_e8m0.len(),
            });
        }
        Ok(Self {
            rows,
            cols,
            scale_rows,
            scale_cols,
            values,
            scales_e8m0,
        })
    }

    pub const fn rows(&self) -> usize {
        self.rows
    }
    pub const fn cols(&self) -> usize {
        self.cols
    }
    pub const fn scale_rows(&self) -> usize {
        self.scale_rows
    }
    pub const fn scale_cols(&self) -> usize {
        self.scale_cols
    }
    pub fn values(&self) -> &[u8] {
        &self.values
    }
    pub fn scales_e8m0(&self) -> &[u8] {
        &self.scales_e8m0
    }
    pub fn row_values(&self, row: usize) -> Option<&[u8]> {
        if row >= self.rows {
            return None;
        }
        let start = row.checked_mul(self.cols)?;
        self.values.get(start..start + self.cols)
    }
    pub fn scale_e8m0(&self, row: usize, col_block: usize) -> Option<u8> {
        if row >= self.rows || col_block >= self.scale_cols {
            return None;
        }
        let scale_row = row / FP8_E4M3_SCALE_TILE;
        self.scales_e8m0
            .get(scale_row * self.scale_cols + col_block)
            .copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e2m1_table_matches_bound_nvidia_converter() {
        for (index, expected) in E2M1_VALUES.iter().enumerate() {
            assert_eq!(decode_e2m1(index as u8), *expected);
        }
    }

    #[test]
    fn e8m0_and_e4m3_boundaries_are_strict() {
        assert_eq!(decode_e8m0(127), Ok(1.0));
        assert_eq!(decode_e8m0(128), Ok(2.0));
        assert!(decode_e8m0(255).is_err());
        assert_eq!(decode_e4m3fn(0x38), Ok(1.0));
        assert_eq!(decode_e4m3fn(0x7e), Ok(448.0));
        assert_eq!(decode_e4m3fn(0xfe), Ok(-448.0));
        assert!(decode_e4m3fn(0x7f).is_err());
    }

    #[test]
    fn mxfp4_and_nvfp4_use_distinct_scale_protocols() {
        let qs = [0x21; MXFP4_PACKED_BYTES];
        let mut mx = [0.0; MXFP4_BLOCK_SIZE];
        Mxfp4CompactBlock {
            qs,
            scale_e8m0: 128,
        }
        .decode_ref(&mut mx)
        .unwrap();
        assert_eq!(&mx[..4], &[1.0, 2.0, 1.0, 2.0]);

        let mut nv = [0.0; NVFP4_BLOCK_SIZE];
        Nvfp4CompactBlock {
            qs: [0x21; NVFP4_PACKED_BYTES],
            scale_e4m3: 0x38,
            global_scale: 2.0,
        }
        .decode_ref(&mut nv)
        .unwrap();
        assert_eq!(&nv[..4], &[1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn fp8_block_rejects_nan_and_decodes_scale() {
        let mut block = Fp8E4M3CompactBlock {
            qs: [0x38; FP8_E4M3_BLOCK_SIZE],
            scale_e8m0: 128,
        };
        let mut output = [0.0; FP8_E4M3_BLOCK_SIZE];
        block.decode_ref(&mut output).unwrap();
        assert!(output.iter().all(|value| *value == 2.0));
        block.qs[17] = 0x7f;
        assert!(block.decode_ref(&mut output).is_err());
    }

    #[test]
    fn fp8_scale_grid_preserves_two_dimensional_scale_identity() {
        let values = vec![0x38; 129 * 256];
        let scales = vec![127, 128, 129, 130];
        let tensor = Fp8E4M3ScaleGridCompactTensor::new(129, 256, values, scales).unwrap();
        assert_eq!((tensor.scale_rows(), tensor.scale_cols()), (2, 2));
        assert_eq!(tensor.scale_e8m0(0, 1), Some(128));
        assert_eq!(tensor.scale_e8m0(127, 1), Some(128));
        assert_eq!(tensor.scale_e8m0(128, 0), Some(129));
    }
}
