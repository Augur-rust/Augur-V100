use augur_core::PhysicalLayoutId;

use crate::HotLayoutError;

pub const FP8_E4M3_SCALE_TILE: usize = 128;

/// CPU Hot owner preserving the source 128x128 scale grid.
#[derive(Clone, Debug, PartialEq)]
pub struct Fp8E4M3ScaleGridHotCpuTensor {
    rows: usize,
    cols: usize,
    scale_rows: usize,
    scale_cols: usize,
    values: Vec<u8>,
    scales: Vec<f32>,
}

impl Fp8E4M3ScaleGridHotCpuTensor {
    pub const PHYSICAL_LAYOUT: PhysicalLayoutId =
        PhysicalLayoutId::FP8_E4M3_SCALE_GRID_HOT_CPU_PORTABLE_V1;

    pub fn new(
        rows: usize,
        cols: usize,
        values: Vec<u8>,
        scales: Vec<f32>,
    ) -> Result<Self, HotLayoutError> {
        if rows == 0 || cols == 0 || !cols.is_multiple_of(FP8_E4M3_SCALE_TILE) {
            return Err(HotLayoutError::BadShape);
        }
        let value_count = rows.checked_mul(cols).ok_or(HotLayoutError::BadShape)?;
        if values.len() != value_count {
            return Err(HotLayoutError::BadPayloadBytes {
                expected: value_count,
                actual: values.len(),
            });
        }
        if let Some((index, _)) = values
            .iter()
            .enumerate()
            .find(|(_, value)| (**value & 0x7f) == 0x7f)
        {
            return Err(HotLayoutError::NonFinitePayload { index });
        }
        let scale_rows = rows.div_ceil(FP8_E4M3_SCALE_TILE);
        let scale_cols = cols / FP8_E4M3_SCALE_TILE;
        let scale_count = scale_rows
            .checked_mul(scale_cols)
            .ok_or(HotLayoutError::BadShape)?;
        if scales.len() != scale_count {
            return Err(HotLayoutError::BadBlockCount {
                expected: scale_count,
                actual: scales.len(),
            });
        }
        if let Some((index, _)) = scales
            .iter()
            .enumerate()
            .find(|(_, scale)| !scale.is_finite() || **scale <= 0.0)
        {
            return Err(HotLayoutError::NonFinitePayload { index });
        }
        Ok(Self {
            rows,
            cols,
            scale_rows,
            scale_cols,
            values,
            scales,
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
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }
    pub fn row_values(&self, row: usize) -> Option<&[u8]> {
        if row >= self.rows {
            return None;
        }
        let start = row.checked_mul(self.cols)?;
        self.values.get(start..start + self.cols)
    }
    pub fn scale(&self, row: usize, col_block: usize) -> Option<f32> {
        if row >= self.rows || col_block >= self.scale_cols {
            return None;
        }
        self.scales
            .get(row / FP8_E4M3_SCALE_TILE * self.scale_cols + col_block)
            .copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hot_grid_rejects_nonfinite_codes_and_scales() {
        assert!(matches!(
            Fp8E4M3ScaleGridHotCpuTensor::new(1, 128, vec![0x7f; 128], vec![1.0]),
            Err(HotLayoutError::NonFinitePayload { .. })
        ));
        assert!(matches!(
            Fp8E4M3ScaleGridHotCpuTensor::new(1, 128, vec![0x38; 128], vec![f32::NAN]),
            Err(HotLayoutError::NonFinitePayload { .. })
        ));
    }
}
