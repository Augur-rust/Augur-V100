use augur_core::{F16Bits, PhysicalLayoutId};

use crate::HotLayoutError;

pub const Q8_0_HOT_CPU_BLOCK_LEN: usize = 32;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Q8_0HotCpuBlock {
    pub d: F16Bits,
    pub qs: [i8; Q8_0_HOT_CPU_BLOCK_LEN],
}

impl Q8_0HotCpuBlock {
    pub const DECODED_LEN: usize = Q8_0_HOT_CPU_BLOCK_LEN;
    pub const PHYSICAL_LAYOUT: PhysicalLayoutId = PhysicalLayoutId::Q8_0_HOT_CPU_PORTABLE_V1;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Q8_0HotCpuTensor {
    rows: usize,
    cols: usize,
    blocks_per_row: usize,
    blocks: Vec<Q8_0HotCpuBlock>,
}

impl Q8_0HotCpuTensor {
    pub const PHYSICAL_LAYOUT: PhysicalLayoutId = PhysicalLayoutId::Q8_0_HOT_CPU_PORTABLE_V1;

    pub fn new(
        rows: usize,
        cols: usize,
        blocks: Vec<Q8_0HotCpuBlock>,
    ) -> Result<Self, HotLayoutError> {
        if rows == 0 || cols == 0 || !cols.is_multiple_of(Q8_0_HOT_CPU_BLOCK_LEN) {
            return Err(HotLayoutError::BadShape);
        }
        let blocks_per_row = cols / Q8_0_HOT_CPU_BLOCK_LEN;
        let expected = rows
            .checked_mul(blocks_per_row)
            .ok_or(HotLayoutError::BadShape)?;
        if blocks.len() != expected {
            return Err(HotLayoutError::BadBlockCount {
                expected,
                actual: blocks.len(),
            });
        }
        Ok(Self {
            rows,
            cols,
            blocks_per_row,
            blocks,
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

    pub fn blocks(&self) -> &[Q8_0HotCpuBlock] {
        &self.blocks
    }

    pub fn row_blocks(&self, row: usize) -> Option<&[Q8_0HotCpuBlock]> {
        if row >= self.rows {
            return None;
        }
        let start = row.checked_mul(self.blocks_per_row)?;
        self.blocks.get(start..start + self.blocks_per_row)
    }
}
