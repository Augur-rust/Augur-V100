use augur_core::{F16Bits, PhysicalLayoutId};

use crate::HotLayoutError;

pub const IQ2_XXS_HOT_CPU_BLOCK_LEN: usize = 256;
pub const IQ2_XXS_HOT_CPU_GROUPS: usize = 8;
pub const IQ2_XXS_HOT_CPU_CODES: usize = 32;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Iq2XxsHotCpuBlock {
    pub d: F16Bits,
    pub qs: [u16; IQ2_XXS_HOT_CPU_CODES],
}

impl Iq2XxsHotCpuBlock {
    pub const DECODED_LEN: usize = IQ2_XXS_HOT_CPU_BLOCK_LEN;
    pub const PHYSICAL_LAYOUT: PhysicalLayoutId = PhysicalLayoutId::IQ2_XXS_HOT_CPU_PORTABLE_V1;

    #[inline]
    pub fn group_words(&self, group: usize) -> Option<&[u16]> {
        if group >= IQ2_XXS_HOT_CPU_GROUPS {
            return None;
        }
        self.qs.get(group * 4..group * 4 + 4)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Iq2XxsHotCpuTensor {
    rows: usize,
    cols: usize,
    blocks_per_row: usize,
    blocks: Vec<Iq2XxsHotCpuBlock>,
}

impl Iq2XxsHotCpuTensor {
    pub const PHYSICAL_LAYOUT: PhysicalLayoutId = PhysicalLayoutId::IQ2_XXS_HOT_CPU_PORTABLE_V1;

    pub fn new(
        rows: usize,
        cols: usize,
        blocks: Vec<Iq2XxsHotCpuBlock>,
    ) -> Result<Self, HotLayoutError> {
        if rows == 0 || cols == 0 || !cols.is_multiple_of(IQ2_XXS_HOT_CPU_BLOCK_LEN) {
            return Err(HotLayoutError::BadShape);
        }
        let blocks_per_row = cols / IQ2_XXS_HOT_CPU_BLOCK_LEN;
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

    pub fn blocks(&self) -> &[Iq2XxsHotCpuBlock] {
        &self.blocks
    }

    pub fn row_blocks(&self, row: usize) -> Option<&[Iq2XxsHotCpuBlock]> {
        if row >= self.rows {
            return None;
        }
        let start = row.checked_mul(self.blocks_per_row)?;
        self.blocks.get(start..start + self.blocks_per_row)
    }
}
