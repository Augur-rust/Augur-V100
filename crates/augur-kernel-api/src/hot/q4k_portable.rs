use crate::HotLayoutError;
use augur_core::{F16Bits, PhysicalLayoutId};

pub const Q4K_HOT_CPU_BLOCK_LEN: usize = 256;
pub const Q4K_HOT_CPU_GROUPS: usize = 8;
pub const Q4K_HOT_CPU_GROUP_SIZE: usize = 32;
pub const Q4K_HOT_CPU_QS_BYTES: usize = 128;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Q4KHotCpuBlock {
    pub d: F16Bits,
    pub dmin: F16Bits,
    pub scale: [u8; Q4K_HOT_CPU_GROUPS],
    pub min: [u8; Q4K_HOT_CPU_GROUPS],
    pub qs: [u8; Q4K_HOT_CPU_QS_BYTES],
}

impl Q4KHotCpuBlock {
    pub const DECODED_LEN: usize = Q4K_HOT_CPU_BLOCK_LEN;
    pub const PHYSICAL_LAYOUT: PhysicalLayoutId = PhysicalLayoutId::Q4K_HOT_CPU_PORTABLE_V1;

    #[inline]
    pub fn quant_at(&self, index: usize) -> u8 {
        debug_assert!(index < Q4K_HOT_CPU_BLOCK_LEN);
        let chunk = index / 64;
        let within_chunk = index % 64;
        let packed = self.qs[chunk * 32 + within_chunk % 32];
        if within_chunk < 32 {
            packed & 0x0f
        } else {
            packed >> 4
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Q4KHotCpuTensor {
    rows: usize,
    cols: usize,
    blocks_per_row: usize,
    blocks: Vec<Q4KHotCpuBlock>,
}

impl Q4KHotCpuTensor {
    pub const PHYSICAL_LAYOUT: PhysicalLayoutId = PhysicalLayoutId::Q4K_HOT_CPU_PORTABLE_V1;

    pub fn new(
        rows: usize,
        cols: usize,
        blocks: Vec<Q4KHotCpuBlock>,
    ) -> Result<Self, HotLayoutError> {
        if rows == 0 || cols == 0 || !cols.is_multiple_of(Q4KHotCpuBlock::DECODED_LEN) {
            return Err(HotLayoutError::BadShape);
        }
        let blocks_per_row = cols / Q4KHotCpuBlock::DECODED_LEN;
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

    pub fn blocks(&self) -> &[Q4KHotCpuBlock] {
        &self.blocks
    }

    pub fn row_blocks(&self, row: usize) -> Option<&[Q4KHotCpuBlock]> {
        if row >= self.rows {
            return None;
        }
        let start = row.checked_mul(self.blocks_per_row)?;
        let end = start.checked_add(self.blocks_per_row)?;
        self.blocks.get(start..end)
    }
}
