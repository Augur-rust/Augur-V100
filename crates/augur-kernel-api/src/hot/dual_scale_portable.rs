use crate::{HotLayoutError, OutlierCsrHotCpu};
use augur_core::F16Bits;

pub const DUAL_SCALE_HOT_CPU_BLOCK_LEN: usize = 256;
pub const DUAL_SCALE_HOT_CPU_GROUP_SIZE: usize = 32;
pub const DUAL_SCALE_HOT_CPU_GROUPS: usize = 8;
pub const DUAL_SCALE_HOT_CPU_QS_BYTES: usize = 128;
pub const DUAL_SCALE_HOT_CPU_MEMBERSHIP_BYTES: usize = 32;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DualScaleHotCpuBlock {
    pub scale_lo: [F16Bits; DUAL_SCALE_HOT_CPU_GROUPS],
    pub scale_hi: [F16Bits; DUAL_SCALE_HOT_CPU_GROUPS],
    pub membership: [u8; DUAL_SCALE_HOT_CPU_MEMBERSHIP_BYTES],
    pub qs: [u8; DUAL_SCALE_HOT_CPU_QS_BYTES],
}

impl DualScaleHotCpuBlock {
    pub const DECODED_LEN: usize = DUAL_SCALE_HOT_CPU_BLOCK_LEN;
}

#[inline]
pub fn dual_scale_hot_membership_bit(
    membership: &[u8; DUAL_SCALE_HOT_CPU_MEMBERSHIP_BYTES],
    index: usize,
) -> bool {
    debug_assert!(index < DUAL_SCALE_HOT_CPU_BLOCK_LEN);
    ((membership[index / 8] >> (index % 8)) & 1) != 0
}

#[inline]
pub fn dual_scale_hot_q4_code(qs: &[u8; DUAL_SCALE_HOT_CPU_QS_BYTES], index: usize) -> u8 {
    debug_assert!(index < DUAL_SCALE_HOT_CPU_BLOCK_LEN);
    let byte = qs[index / 2];
    if index & 1 == 0 {
        byte & 0x0f
    } else {
        byte >> 4
    }
}

#[inline]
pub fn decode_dual_scale_signed_q4(code: u8) -> i8 {
    debug_assert!(code <= 0x0f);
    code as i8 - 8
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DualScaleHotCpuTensor {
    rows: usize,
    cols: usize,
    blocks_per_row: usize,
    blocks: Vec<DualScaleHotCpuBlock>,
    outliers: OutlierCsrHotCpu,
}

impl DualScaleHotCpuTensor {
    pub fn new(
        rows: usize,
        cols: usize,
        blocks: Vec<DualScaleHotCpuBlock>,
        outliers: OutlierCsrHotCpu,
    ) -> Result<Self, HotLayoutError> {
        if rows == 0 || cols == 0 || !cols.is_multiple_of(DualScaleHotCpuBlock::DECODED_LEN) {
            return Err(HotLayoutError::BadShape);
        }
        let blocks_per_row = cols / DualScaleHotCpuBlock::DECODED_LEN;
        let expected_blocks = rows
            .checked_mul(blocks_per_row)
            .ok_or(HotLayoutError::BadShape)?;
        if blocks.len() != expected_blocks {
            return Err(HotLayoutError::BadBlockCount {
                expected: expected_blocks,
                actual: blocks.len(),
            });
        }
        if outliers.rows() != rows || outliers.cols() != cols {
            return Err(HotLayoutError::BadShape);
        }
        Ok(Self {
            rows,
            cols,
            blocks_per_row,
            blocks,
            outliers,
        })
    }

    pub const fn rows(&self) -> usize {
        self.rows
    }

    pub const fn cols(&self) -> usize {
        self.cols
    }

    pub const fn blocks_per_row(&self) -> usize {
        self.blocks_per_row
    }

    pub fn blocks(&self) -> &[DualScaleHotCpuBlock] {
        &self.blocks
    }

    pub fn row_blocks(&self, row: usize) -> Option<&[DualScaleHotCpuBlock]> {
        if row >= self.rows {
            return None;
        }
        let start = row.checked_mul(self.blocks_per_row)?;
        let end = start.checked_add(self.blocks_per_row)?;
        self.blocks.get(start..end)
    }

    pub fn outliers(&self) -> &OutlierCsrHotCpu {
        &self.outliers
    }
}

#[cfg(test)]
mod layout_tests {
    use super::*;

    #[test]
    fn dual_scale_hot_cpu_block_current_payload_size_is_192() {
        assert_eq!(core::mem::size_of::<DualScaleHotCpuBlock>(), 192);
    }

    #[test]
    fn dual_scale_hot_cpu_block_stride_is_avx2_vector_multiple() {
        assert_eq!(core::mem::size_of::<DualScaleHotCpuBlock>() % 32, 0);
    }

    #[test]
    fn tensor_rejects_zero_dimensions() {
        let outliers = OutlierCsrHotCpu::new(0, 256, vec![0], vec![], vec![]).unwrap();
        assert_eq!(
            DualScaleHotCpuTensor::new(0, 256, vec![], outliers),
            Err(HotLayoutError::BadShape)
        );
    }

    #[test]
    fn malformed_internal_shape_makes_row_blocks_return_none() {
        let outliers = OutlierCsrHotCpu::new(1, 256, vec![0, 0], vec![], vec![]).unwrap();
        let mut tensor = DualScaleHotCpuTensor::new(
            1,
            256,
            vec![DualScaleHotCpuBlock {
                scale_lo: [F16Bits::ZERO; DUAL_SCALE_HOT_CPU_GROUPS],
                scale_hi: [F16Bits::ZERO; DUAL_SCALE_HOT_CPU_GROUPS],
                membership: [0; DUAL_SCALE_HOT_CPU_MEMBERSHIP_BYTES],
                qs: [0; DUAL_SCALE_HOT_CPU_QS_BYTES],
            }],
            outliers,
        )
        .unwrap();
        tensor.blocks_per_row = usize::MAX;
        assert_eq!(tensor.row_blocks(0), None);
    }
}
