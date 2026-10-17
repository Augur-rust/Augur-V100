use augur_core::PhysicalLayoutId;

use crate::HotLayoutError;

pub const Q5K_HOT_CUDA_V1_BLOCK_BYTES: usize = 192;
pub const Q5K_HOT_CUDA_V1_COMPACT_BYTES: usize = 176;
pub const Q5K_HOT_CUDA_V1_RESERVED_OFFSET: usize = 20;
pub const Q5K_HOT_CUDA_V1_RESERVED_BYTES: usize = 12;
pub const Q5K_HOT_CUDA_V1_QH_OFFSET: usize = 32;
pub const Q5K_HOT_CUDA_V1_QS_OFFSET: usize = 64;
pub const Q5K_HOT_CUDA_V1_QS_BYTES: usize = 128;
pub const Q6K_HOT_CUDA_V1_BLOCK_BYTES: usize = 224;
pub const Q6K_HOT_CUDA_V1_COMPACT_BYTES: usize = 210;
pub const QK_HOT_CUDA_V1_DECODED_LEN: usize = 256;
pub const QK_HOT_CUDA_V1_BLOCK_ALIGNMENT: usize = 32;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QkHotCudaStagingTensorV1 {
    layout: PhysicalLayoutId,
    block_bytes: usize,
    rows: usize,
    cols: usize,
    bytes: Vec<u8>,
}

impl QkHotCudaStagingTensorV1 {
    pub fn new(
        layout: PhysicalLayoutId,
        block_bytes: usize,
        compact_bytes: usize,
        rows: usize,
        cols: usize,
        bytes: Vec<u8>,
    ) -> Result<Self, HotLayoutError> {
        if rows == 0 || cols == 0 || !cols.is_multiple_of(QK_HOT_CUDA_V1_DECODED_LEN) {
            return Err(HotLayoutError::BadShape);
        }
        if !matches!(
            (layout, block_bytes, compact_bytes),
            (PhysicalLayoutId::Q5K_HOT_CUDA_V1, 192, 176)
                | (PhysicalLayoutId::Q6K_HOT_CUDA_V1, 224, 210)
        ) {
            return Err(HotLayoutError::BadShape);
        }
        let blocks = rows
            .checked_mul(cols / QK_HOT_CUDA_V1_DECODED_LEN)
            .ok_or(HotLayoutError::BadShape)?;
        let expected = blocks
            .checked_mul(block_bytes)
            .ok_or(HotLayoutError::BadShape)?;
        if bytes.len() != expected {
            return Err(HotLayoutError::BadPayloadBytes {
                expected,
                actual: bytes.len(),
            });
        }
        for (block, payload) in bytes.chunks_exact(block_bytes).enumerate() {
            let reserved = match layout {
                PhysicalLayoutId::Q5K_HOT_CUDA_V1 => {
                    &payload[Q5K_HOT_CUDA_V1_RESERVED_OFFSET
                        ..Q5K_HOT_CUDA_V1_RESERVED_OFFSET + Q5K_HOT_CUDA_V1_RESERVED_BYTES]
                }
                PhysicalLayoutId::Q6K_HOT_CUDA_V1 => &payload[compact_bytes..],
                _ => unreachable!(),
            };
            if reserved.iter().any(|byte| *byte != 0) {
                return Err(HotLayoutError::BadReservedBytes { block });
            }
        }
        Ok(Self {
            layout,
            block_bytes,
            rows,
            cols,
            bytes,
        })
    }

    pub const fn layout(&self) -> PhysicalLayoutId {
        self.layout
    }
    pub const fn block_bytes(&self) -> usize {
        self.block_bytes
    }
    pub const fn rows(&self) -> usize {
        self.rows
    }
    pub const fn cols(&self) -> usize {
        self.cols
    }
    pub fn payload_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn into_payload_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

const _: () = {
    assert!(Q5K_HOT_CUDA_V1_RESERVED_OFFSET + Q5K_HOT_CUDA_V1_RESERVED_BYTES == 32);
    assert!(Q5K_HOT_CUDA_V1_QH_OFFSET == 32);
    assert!(Q5K_HOT_CUDA_V1_QS_OFFSET == 64);
    assert!(Q5K_HOT_CUDA_V1_QS_OFFSET + Q5K_HOT_CUDA_V1_QS_BYTES == Q5K_HOT_CUDA_V1_BLOCK_BYTES);
};
