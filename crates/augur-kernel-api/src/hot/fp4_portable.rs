use augur_core::PhysicalLayoutId;

use crate::HotLayoutError;

pub const MXFP4_HOT_CPU_BLOCK_LEN: usize = 32;
pub const NVFP4_HOT_CPU_BLOCK_LEN: usize = 16;
pub const FP8_E4M3_HOT_CPU_BLOCK_LEN: usize = 128;

#[derive(Clone, Debug, PartialEq)]
pub struct Mxfp4HotCpuBlock {
    pub scale: f32,
    pub qs: [u8; 16],
}

#[derive(Clone, Debug, PartialEq)]
pub struct Nvfp4HotCpuBlock {
    pub scale: f32,
    pub qs: [u8; 8],
}

#[derive(Clone, Debug, PartialEq)]
pub struct Fp8E4M3HotCpuBlock {
    pub scale: f32,
    pub qs: [u8; FP8_E4M3_HOT_CPU_BLOCK_LEN],
}

macro_rules! fp4_tensor {
    ($name:ident, $block:ty, $len:expr, $layout:expr) => {
        #[derive(Clone, Debug, PartialEq)]
        pub struct $name {
            rows: usize,
            cols: usize,
            blocks_per_row: usize,
            blocks: Vec<$block>,
        }

        impl $name {
            pub const PHYSICAL_LAYOUT: PhysicalLayoutId = $layout;

            pub fn new(
                rows: usize,
                cols: usize,
                blocks: Vec<$block>,
            ) -> Result<Self, HotLayoutError> {
                if rows == 0 || cols == 0 || !cols.is_multiple_of($len) {
                    return Err(HotLayoutError::BadShape);
                }
                let blocks_per_row = cols / $len;
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
            pub fn blocks(&self) -> &[$block] {
                &self.blocks
            }
            pub fn row_blocks(&self, row: usize) -> Option<&[$block]> {
                if row >= self.rows {
                    return None;
                }
                let start = row.checked_mul(self.blocks_per_row)?;
                self.blocks.get(start..start + self.blocks_per_row)
            }
        }
    };
}

fp4_tensor!(
    Mxfp4HotCpuTensor,
    Mxfp4HotCpuBlock,
    MXFP4_HOT_CPU_BLOCK_LEN,
    PhysicalLayoutId::MXFP4_HOT_CPU_PORTABLE_V1
);
fp4_tensor!(
    Nvfp4HotCpuTensor,
    Nvfp4HotCpuBlock,
    NVFP4_HOT_CPU_BLOCK_LEN,
    PhysicalLayoutId::NVFP4_HOT_CPU_PORTABLE_V1
);
fp4_tensor!(
    Fp8E4M3HotCpuTensor,
    Fp8E4M3HotCpuBlock,
    FP8_E4M3_HOT_CPU_BLOCK_LEN,
    PhysicalLayoutId::FP8_E4M3_BLOCK_HOT_CPU_PORTABLE_V1
);
