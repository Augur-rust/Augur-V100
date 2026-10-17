#![cfg(feature = "cuda")]

use crate::{CudaBufferView, CudaRuntimeError, OwnedCudaStream};
pub use augur_cuda_sys::AugurCudaM292HbmArgsV1 as CudaM292RawHbmArgsV1;

const OP: &str = "cuda_m292_hbm_submit_v1";

pub fn cuda_m292_l2_cache_bytes(device_ordinal: i32) -> Result<u64, CudaRuntimeError> {
    if device_ordinal < 0 {
        return Err(CudaRuntimeError::invalid_metadata("m292_l2_cache_bytes"));
    }
    let mut bytes = 0_u64;
    let result = unsafe { augur_cuda_sys::augur_cuda_m292_l2_cache_bytes_v1(device_ordinal, &mut bytes) };
    CudaRuntimeError::from_result("cuda_m292_l2_cache_bytes", result)?;
    if bytes == 0 {
        return Err(CudaRuntimeError::invalid_metadata("m292_l2_cache_bytes"));
    }
    Ok(bytes)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CudaM292HbmOperation {
    ReadOnly,
    ReadWrite,
}

impl CudaM292HbmOperation {
    const fn raw(self) -> u32 {
        match self {
            Self::ReadOnly => augur_cuda_sys::AUGUR_CUDA_M292_HBM_READ_ONLY,
            Self::ReadWrite => augur_cuda_sys::AUGUR_CUDA_M292_HBM_READ_WRITE,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaM292HbmByteCount {
    pub read_bytes: u64,
    pub write_bytes: u64,
}

impl CudaM292HbmByteCount {
    pub fn total_bytes(self) -> Result<u64, CudaRuntimeError> {
        self.read_bytes
            .checked_add(self.write_bytes)
            .ok_or_else(CudaRuntimeError::range_overflow)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CudaM292HbmV1 {
    raw: CudaM292RawHbmArgsV1,
    device_ordinal: i32,
    byte_count: CudaM292HbmByteCount,
}

impl CudaM292HbmV1 {
    pub fn bind(
        operation: CudaM292HbmOperation,
        source: CudaBufferView,
        destination: CudaBufferView,
        blocks: u32,
        repeats: u32,
    ) -> Result<Self, CudaRuntimeError> {
        if source.device_ordinal() != destination.device_ordinal()
            || source.len_bytes() == 0
            || !source.len_bytes().is_multiple_of(16)
            || !source.offset_bytes().is_multiple_of(16)
            || !destination.offset_bytes().is_multiple_of(16)
            || blocks == 0
            || repeats == 0
        {
            return Err(CudaRuntimeError::invalid_metadata("m292_hbm_binding"));
        }
        let source_start = source.checked_address(OP)?;
        let destination_start = destination.checked_address(OP)?;
        source.checked_range(OP)?;
        destination.checked_range(OP)?;
        if !source_start.is_multiple_of(16) || !destination_start.is_multiple_of(16) {
            return Err(CudaRuntimeError::invalid_metadata("m292_hbm_alignment"));
        }
        let expected_destination = match operation {
            CudaM292HbmOperation::ReadOnly => usize::try_from(blocks)
                .ok()
                .and_then(|value| value.checked_mul(16))
                .ok_or_else(CudaRuntimeError::range_overflow)?,
            CudaM292HbmOperation::ReadWrite => source.len_bytes(),
        };
        if destination.len_bytes() != expected_destination || ranges_overlap(source, destination)? {
            return Err(CudaRuntimeError::invalid_metadata("m292_hbm_destination"));
        }
        let source_bytes =
            u64::try_from(source.len_bytes()).map_err(|_| CudaRuntimeError::range_overflow())?;
        let destination_bytes = u64::try_from(destination.len_bytes())
            .map_err(|_| CudaRuntimeError::range_overflow())?;
        let repeats_u64 = u64::from(repeats);
        let read_bytes = source_bytes
            .checked_mul(repeats_u64)
            .ok_or_else(CudaRuntimeError::range_overflow)?;
        let write_bytes = match operation {
            CudaM292HbmOperation::ReadOnly => destination_bytes,
            CudaM292HbmOperation::ReadWrite => source_bytes
                .checked_mul(repeats_u64)
                .ok_or_else(CudaRuntimeError::range_overflow)?,
        };

        let mut raw = CudaM292RawHbmArgsV1::new();
        raw.operation = operation.raw();
        raw.blocks = blocks;
        raw.source_base = source.device_ptr().raw_value();
        raw.source_allocation_extent_bytes = u64::try_from(source.allocation_len_bytes())
            .map_err(|_| CudaRuntimeError::range_overflow())?;
        raw.source_view_offset_bytes =
            u64::try_from(source.offset_bytes()).map_err(|_| CudaRuntimeError::range_overflow())?;
        raw.source_view_extent_bytes = source_bytes;
        raw.destination_base = destination.device_ptr().raw_value();
        raw.destination_allocation_extent_bytes = u64::try_from(destination.allocation_len_bytes())
            .map_err(|_| CudaRuntimeError::range_overflow())?;
        raw.destination_view_offset_bytes = u64::try_from(destination.offset_bytes())
            .map_err(|_| CudaRuntimeError::range_overflow())?;
        raw.destination_view_extent_bytes = destination_bytes;
        raw.repeats = repeats;
        Ok(Self {
            raw,
            device_ordinal: source.device_ordinal(),
            byte_count: CudaM292HbmByteCount {
                read_bytes,
                write_bytes,
            },
        })
    }

    pub const fn raw(&self) -> &CudaM292RawHbmArgsV1 {
        &self.raw
    }

    pub const fn byte_count(&self) -> CudaM292HbmByteCount {
        self.byte_count
    }

    pub fn submit(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.device_ordinal {
            return Err(CudaRuntimeError::device_mismatch());
        }
        let result = unsafe {
            augur_cuda_sys::augur_cuda_m292_hbm_submit_v1(self.device_ordinal, stream.raw()?, &self.raw)
        };
        CudaRuntimeError::from_result(OP, result)
    }
}

fn ranges_overlap(left: CudaBufferView, right: CudaBufferView) -> Result<bool, CudaRuntimeError> {
    let left_start = left
        .device_ptr()
        .raw_value()
        .checked_add(
            u64::try_from(left.offset_bytes()).map_err(|_| CudaRuntimeError::range_overflow())?,
        )
        .ok_or_else(CudaRuntimeError::range_overflow)?;
    let left_end = left_start
        .checked_add(
            u64::try_from(left.len_bytes()).map_err(|_| CudaRuntimeError::range_overflow())?,
        )
        .ok_or_else(CudaRuntimeError::range_overflow)?;
    let right_start = right
        .device_ptr()
        .raw_value()
        .checked_add(
            u64::try_from(right.offset_bytes()).map_err(|_| CudaRuntimeError::range_overflow())?,
        )
        .ok_or_else(CudaRuntimeError::range_overflow)?;
    let right_end = right_start
        .checked_add(
            u64::try_from(right.len_bytes()).map_err(|_| CudaRuntimeError::range_overflow())?,
        )
        .ok_or_else(CudaRuntimeError::range_overflow)?;
    Ok(left_start < right_end && right_start < left_end)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_runtime::test_buffer_view;

    #[test]
    fn read_only_binding_accounts_payload_and_checksum_bytes() {
        let value = CudaM292HbmV1::bind(
            CudaM292HbmOperation::ReadOnly,
            test_buffer_view(0x1000, 0, 32, 4096),
            test_buffer_view(0x10_0000, 0, 64, 320),
            20,
            4,
        )
        .unwrap();
        assert_eq!(value.raw.source_view_offset_bytes, 32);
        assert_eq!(value.raw.destination_view_offset_bytes, 64);
        assert_eq!(value.byte_count.read_bytes, 16_384);
        assert_eq!(value.byte_count.write_bytes, 320);
    }

    #[test]
    fn read_write_rejects_alias_and_requires_equal_extents() {
        let source = test_buffer_view(0x1000, 0, 0, 4096);
        let alias = test_buffer_view(0x1000, 0, 2048, 4096);
        assert!(
            CudaM292HbmV1::bind(CudaM292HbmOperation::ReadWrite, source, alias, 80, 1,).is_err()
        );
        let destination = test_buffer_view(0x20_0000, 0, 0, 4096);
        let value =
            CudaM292HbmV1::bind(CudaM292HbmOperation::ReadWrite, source, destination, 80, 3)
                .unwrap();
        assert_eq!(value.byte_count.total_bytes().unwrap(), 24_576);
    }

    #[test]
    fn binding_rejects_allocation_address_overflow() {
        let source = test_buffer_view(u64::MAX - 15, 0, 0, 4096);
        let destination = test_buffer_view(0x20_0000, 0, 0, 4096);
        let error =
            CudaM292HbmV1::bind(CudaM292HbmOperation::ReadWrite, source, destination, 80, 1)
                .unwrap_err();
        assert_eq!(error.kind, crate::CudaRuntimeErrorKind::RangeOverflow);
    }
}
