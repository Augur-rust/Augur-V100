use augur_core::PhysicalLayoutId;

use crate::{
    HotLayoutError, Q4K_HOT_CUDA_V1_BLOCK_BYTES, Q4K_HOT_CUDA_V1_RESERVED_BYTES,
    Q4K_HOT_CUDA_V1_RESERVED_OFFSET, Q5K_HOT_CUDA_V1_BLOCK_BYTES, Q5K_HOT_CUDA_V1_RESERVED_BYTES,
    Q5K_HOT_CUDA_V1_RESERVED_OFFSET, QK_HOT_CUDA_V1_DECODED_LEN,
};

pub const EXPERT_ROW32_CUDA_V2_ROWS_PER_GROUP: usize = 32;
pub const EXPERT_ROW32_CUDA_V2_VECTOR_BYTES: usize = 16;

/// Host-owned staging bytes for the expert-only row32 CUDA protocols.
///
/// The payload contains the same per-block bytes as the corresponding v1 Hot
/// layout, but its address order is `[row_group][block][vector][row_lane]`.
/// It is deliberately a separate type so row-major v1 views cannot alias it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QkExpertRow32CudaStagingTensorV2 {
    layout: PhysicalLayoutId,
    source_layout: PhysicalLayoutId,
    block_bytes: usize,
    vectors_per_block: usize,
    rows: usize,
    cols: usize,
    blocks_per_row: usize,
    bytes: Vec<u8>,
}

impl QkExpertRow32CudaStagingTensorV2 {
    pub fn new(
        layout: PhysicalLayoutId,
        rows: usize,
        cols: usize,
        bytes: Vec<u8>,
    ) -> Result<Self, HotLayoutError> {
        let (source_layout, block_bytes) = match layout {
            PhysicalLayoutId::Q4K_EXPERT_ROW32_CUDA_V2 => (
                PhysicalLayoutId::Q4K_HOT_CUDA_V1,
                Q4K_HOT_CUDA_V1_BLOCK_BYTES,
            ),
            PhysicalLayoutId::Q5K_EXPERT_ROW32_CUDA_V2 => (
                PhysicalLayoutId::Q5K_HOT_CUDA_V1,
                Q5K_HOT_CUDA_V1_BLOCK_BYTES,
            ),
            _ => return Err(HotLayoutError::BadShape),
        };
        if rows == 0
            || !rows.is_multiple_of(EXPERT_ROW32_CUDA_V2_ROWS_PER_GROUP)
            || cols == 0
            || !cols.is_multiple_of(QK_HOT_CUDA_V1_DECODED_LEN)
            || !block_bytes.is_multiple_of(EXPERT_ROW32_CUDA_V2_VECTOR_BYTES)
        {
            return Err(HotLayoutError::BadShape);
        }

        let blocks_per_row = cols / QK_HOT_CUDA_V1_DECODED_LEN;
        let block_count = rows
            .checked_mul(blocks_per_row)
            .ok_or(HotLayoutError::BadShape)?;
        let expected = block_count
            .checked_mul(block_bytes)
            .ok_or(HotLayoutError::BadShape)?;
        if bytes.len() != expected {
            return Err(HotLayoutError::BadPayloadBytes {
                expected,
                actual: bytes.len(),
            });
        }

        let vectors_per_block = block_bytes / EXPERT_ROW32_CUDA_V2_VECTOR_BYTES;
        let tensor = Self {
            layout,
            source_layout,
            block_bytes,
            vectors_per_block,
            rows,
            cols,
            blocks_per_row,
            bytes,
        };
        tensor.validate_reserved_bytes()?;
        Ok(tensor)
    }

    pub const fn layout(&self) -> PhysicalLayoutId {
        self.layout
    }

    pub const fn source_layout(&self) -> PhysicalLayoutId {
        self.source_layout
    }

    pub const fn block_bytes(&self) -> usize {
        self.block_bytes
    }

    pub const fn vectors_per_block(&self) -> usize {
        self.vectors_per_block
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

    pub const fn block_count(&self) -> usize {
        self.rows * self.blocks_per_row
    }

    pub fn payload_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn payload_len_bytes(&self) -> usize {
        self.bytes.len()
    }

    /// Returns the physical offset for one 16-byte vector.
    ///
    /// The checked arithmetic is the contract formula:
    /// `((((row/32)*blocks_per_row+block)*vectors_per_block+vector)*32
    ///   +(row%32))*16)`.
    pub fn vector_offset(&self, row: usize, block_index: usize, vector: usize) -> Option<usize> {
        if row >= self.rows
            || block_index >= self.blocks_per_row
            || vector >= self.vectors_per_block
        {
            return None;
        }
        let row_group = row / EXPERT_ROW32_CUDA_V2_ROWS_PER_GROUP;
        let row_lane = row % EXPERT_ROW32_CUDA_V2_ROWS_PER_GROUP;
        row_group
            .checked_mul(self.blocks_per_row)?
            .checked_add(block_index)?
            .checked_mul(self.vectors_per_block)?
            .checked_add(vector)?
            .checked_mul(EXPERT_ROW32_CUDA_V2_ROWS_PER_GROUP)?
            .checked_add(row_lane)?
            .checked_mul(EXPERT_ROW32_CUDA_V2_VECTOR_BYTES)
    }

    pub fn vector_bytes(&self, row: usize, block_index: usize, vector: usize) -> Option<&[u8]> {
        let start = self.vector_offset(row, block_index, vector)?;
        let end = start.checked_add(EXPERT_ROW32_CUDA_V2_VECTOR_BYTES)?;
        self.bytes.get(start..end)
    }

    fn validate_reserved_bytes(&self) -> Result<(), HotLayoutError> {
        let (reserved_offset, reserved_bytes) = match self.layout {
            PhysicalLayoutId::Q4K_EXPERT_ROW32_CUDA_V2 => (
                Q4K_HOT_CUDA_V1_RESERVED_OFFSET,
                Q4K_HOT_CUDA_V1_RESERVED_BYTES,
            ),
            PhysicalLayoutId::Q5K_EXPERT_ROW32_CUDA_V2 => (
                Q5K_HOT_CUDA_V1_RESERVED_OFFSET,
                Q5K_HOT_CUDA_V1_RESERVED_BYTES,
            ),
            _ => return Err(HotLayoutError::BadShape),
        };
        let reserved_vector = reserved_offset / EXPERT_ROW32_CUDA_V2_VECTOR_BYTES;
        let reserved_start = reserved_offset % EXPERT_ROW32_CUDA_V2_VECTOR_BYTES;
        let reserved_end = reserved_start + reserved_bytes;
        if reserved_end > EXPERT_ROW32_CUDA_V2_VECTOR_BYTES {
            return Err(HotLayoutError::BadShape);
        }
        for row in 0..self.rows {
            for block_index in 0..self.blocks_per_row {
                let vector = self
                    .vector_bytes(row, block_index, reserved_vector)
                    .ok_or(HotLayoutError::BadShape)?;
                if vector[reserved_start..reserved_end]
                    .iter()
                    .any(|byte| *byte != 0)
                {
                    return Err(HotLayoutError::BadReservedBytes {
                        block: row * self.blocks_per_row + block_index,
                    });
                }
            }
        }
        Ok(())
    }
}

const _: () = {
    assert!(Q4K_HOT_CUDA_V1_BLOCK_BYTES.is_multiple_of(EXPERT_ROW32_CUDA_V2_VECTOR_BYTES));
    assert!(Q5K_HOT_CUDA_V1_BLOCK_BYTES.is_multiple_of(EXPERT_ROW32_CUDA_V2_VECTOR_BYTES));
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructor_rejects_row_major_and_non_grouped_shapes() {
        assert_eq!(
            QkExpertRow32CudaStagingTensorV2::new(
                PhysicalLayoutId::Q4K_HOT_CUDA_V1,
                32,
                256,
                vec![0; 32 * Q4K_HOT_CUDA_V1_BLOCK_BYTES],
            ),
            Err(HotLayoutError::BadShape),
        );
        assert_eq!(
            QkExpertRow32CudaStagingTensorV2::new(
                PhysicalLayoutId::Q4K_EXPERT_ROW32_CUDA_V2,
                31,
                256,
                vec![0; 31 * Q4K_HOT_CUDA_V1_BLOCK_BYTES],
            ),
            Err(HotLayoutError::BadShape),
        );
        assert_eq!(
            QkExpertRow32CudaStagingTensorV2::new(
                PhysicalLayoutId::Q5K_EXPERT_ROW32_CUDA_V2,
                32,
                255,
                vec![0; 32 * Q5K_HOT_CUDA_V1_BLOCK_BYTES],
            ),
            Err(HotLayoutError::BadShape),
        );
    }

    #[test]
    fn vector_offset_matches_frozen_formula() {
        let tensor = QkExpertRow32CudaStagingTensorV2::new(
            PhysicalLayoutId::Q5K_EXPERT_ROW32_CUDA_V2,
            64,
            512,
            vec![0; 64 * 2 * Q5K_HOT_CUDA_V1_BLOCK_BYTES],
        )
        .unwrap();
        let row = 47;
        let block = 1;
        let vector = 7;
        let expected = ((((row / 32) * 2 + block) * 12 + vector) * 32 + (row % 32)) * 16;
        assert_eq!(tensor.vector_offset(row, block, vector), Some(expected));
        assert_eq!(tensor.vector_offset(64, block, vector), None);
        assert_eq!(tensor.vector_offset(row, 2, vector), None);
        assert_eq!(tensor.vector_offset(row, block, 12), None);
    }
}
