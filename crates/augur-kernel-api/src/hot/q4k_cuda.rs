use augur_core::PhysicalLayoutId;

use crate::HotLayoutError;

pub const Q4K_HOT_CUDA_V1_DECODED_LEN: usize = 256;
pub const Q4K_HOT_CUDA_V1_GROUPS: usize = 8;
pub const Q4K_HOT_CUDA_V1_QS_BYTES: usize = 128;

pub const Q4K_HOT_CUDA_V1_BLOCK_ALIGNMENT: usize = 32;
pub const Q4K_HOT_CUDA_V1_BLOCK_BYTES: usize = 160;

pub const Q4K_HOT_CUDA_V1_D_OFFSET: usize = 0;
pub const Q4K_HOT_CUDA_V1_DMIN_OFFSET: usize = 2;
pub const Q4K_HOT_CUDA_V1_SCALE_OFFSET: usize = 4;
pub const Q4K_HOT_CUDA_V1_MIN_OFFSET: usize = 12;
pub const Q4K_HOT_CUDA_V1_RESERVED_OFFSET: usize = 20;
pub const Q4K_HOT_CUDA_V1_RESERVED_BYTES: usize = 12;
pub const Q4K_HOT_CUDA_V1_QS_OFFSET: usize = 32;

/// Host-owned serialized staging payload for `Q4K_HOT_CUDA_V1`.
///
/// This owner contains an explicit byte protocol. It is suitable as the
/// source of an H2D transfer, but it is not CUDA device memory.
///
/// It is deliberately separate from `Q4KHotCpuTensor`:
///
/// - no Rust struct representation crosses FFI;
/// - padding bytes are canonical zero;
/// - all multi-byte scalar fields use little-endian encoding;
/// - physical block stride is fixed independently of Rust ABI.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Q4KHotCudaStagingTensorV1 {
    rows: usize,
    cols: usize,
    blocks_per_row: usize,
    bytes: Vec<u8>,
}

impl Q4KHotCudaStagingTensorV1 {
    pub const PHYSICAL_LAYOUT: PhysicalLayoutId = PhysicalLayoutId::Q4K_HOT_CUDA_V1;

    pub fn new(rows: usize, cols: usize, bytes: Vec<u8>) -> Result<Self, HotLayoutError> {
        if rows == 0 || cols == 0 || !cols.is_multiple_of(Q4K_HOT_CUDA_V1_DECODED_LEN) {
            return Err(HotLayoutError::BadShape);
        }

        let blocks_per_row = cols / Q4K_HOT_CUDA_V1_DECODED_LEN;
        let block_count = rows
            .checked_mul(blocks_per_row)
            .ok_or(HotLayoutError::BadShape)?;

        let expected_bytes = block_count
            .checked_mul(Q4K_HOT_CUDA_V1_BLOCK_BYTES)
            .ok_or(HotLayoutError::BadShape)?;

        if bytes.len() != expected_bytes {
            return Err(HotLayoutError::BadPayloadBytes {
                expected: expected_bytes,
                actual: bytes.len(),
            });
        }

        for (block_index, block) in bytes.chunks_exact(Q4K_HOT_CUDA_V1_BLOCK_BYTES).enumerate() {
            let reserved_end = Q4K_HOT_CUDA_V1_RESERVED_OFFSET + Q4K_HOT_CUDA_V1_RESERVED_BYTES;

            if block[Q4K_HOT_CUDA_V1_RESERVED_OFFSET..reserved_end]
                .iter()
                .any(|&byte| byte != 0)
            {
                return Err(HotLayoutError::BadReservedBytes { block: block_index });
            }
        }

        Ok(Self {
            rows,
            cols,
            blocks_per_row,
            bytes,
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

    pub fn block_count(&self) -> usize {
        self.bytes.len() / Q4K_HOT_CUDA_V1_BLOCK_BYTES
    }

    pub fn payload_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn into_payload_bytes(self) -> Vec<u8> {
        self.bytes
    }

    pub fn payload_len_bytes(&self) -> usize {
        self.bytes.len()
    }

    pub fn block_bytes(&self, block_index: usize) -> Option<&[u8]> {
        let start = block_index.checked_mul(Q4K_HOT_CUDA_V1_BLOCK_BYTES)?;
        let end = start.checked_add(Q4K_HOT_CUDA_V1_BLOCK_BYTES)?;
        self.bytes.get(start..end)
    }
}

const _: () = {
    assert!(Q4K_HOT_CUDA_V1_D_OFFSET == 0);
    assert!(Q4K_HOT_CUDA_V1_DMIN_OFFSET == 2);
    assert!(Q4K_HOT_CUDA_V1_SCALE_OFFSET == 4);
    assert!(Q4K_HOT_CUDA_V1_MIN_OFFSET == 12);
    assert!(Q4K_HOT_CUDA_V1_QS_OFFSET == 32);
    assert!(Q4K_HOT_CUDA_V1_QS_OFFSET + Q4K_HOT_CUDA_V1_QS_BYTES == Q4K_HOT_CUDA_V1_BLOCK_BYTES);
    assert!(Q4K_HOT_CUDA_V1_BLOCK_BYTES.is_multiple_of(Q4K_HOT_CUDA_V1_BLOCK_ALIGNMENT));
};
