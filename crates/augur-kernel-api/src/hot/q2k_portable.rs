use augur_core::{F16Bits, PhysicalLayoutId};

use crate::HotLayoutError;

pub const Q2K_HOT_CPU_BLOCK_LEN: usize = 256;
pub const Q2K_HOT_CPU_GROUPS: usize = 16;
pub const Q2K_HOT_CPU_QS_BYTES: usize = 64;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Q2KHotCpuBlock {
    pub d: F16Bits,
    pub dmin: F16Bits,
    pub scale: [u8; Q2K_HOT_CPU_GROUPS],
    pub min: [u8; Q2K_HOT_CPU_GROUPS],
    pub qs: [u8; Q2K_HOT_CPU_QS_BYTES],
}

impl Q2KHotCpuBlock {
    pub const DECODED_LEN: usize = Q2K_HOT_CPU_BLOCK_LEN;
    pub const PHYSICAL_LAYOUT: PhysicalLayoutId = PhysicalLayoutId::Q2K_HOT_CPU_PORTABLE_V1;

    #[inline]
    pub fn quant_at(&self, index: usize) -> u8 {
        debug_assert!(index < Q2K_HOT_CPU_BLOCK_LEN);
        let half = index / 128;
        let within_half = index % 128;
        let quarter = within_half / 32;
        let lane = within_half % 32;
        (self.qs[half * 32 + lane] >> (quarter * 2)) & 0x03
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Q2KHotCpuTensor {
    rows: usize,
    cols: usize,
    blocks_per_row: usize,
    blocks: Vec<Q2KHotCpuBlock>,
}

impl Q2KHotCpuTensor {
    pub const PHYSICAL_LAYOUT: PhysicalLayoutId = PhysicalLayoutId::Q2K_HOT_CPU_PORTABLE_V1;

    pub fn new(
        rows: usize,
        cols: usize,
        blocks: Vec<Q2KHotCpuBlock>,
    ) -> Result<Self, HotLayoutError> {
        if rows == 0 || cols == 0 || !cols.is_multiple_of(Q2K_HOT_CPU_BLOCK_LEN) {
            return Err(HotLayoutError::BadShape);
        }
        let blocks_per_row = cols / Q2K_HOT_CPU_BLOCK_LEN;
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

    pub fn blocks(&self) -> &[Q2KHotCpuBlock] {
        &self.blocks
    }

    pub fn row_blocks(&self, row: usize) -> Option<&[Q2KHotCpuBlock]> {
        if row >= self.rows {
            return None;
        }
        let start = row.checked_mul(self.blocks_per_row)?;
        self.blocks.get(start..start + self.blocks_per_row)
    }
}
