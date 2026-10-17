#![cfg(feature = "cuda")]

use crate::{CudaBufferView, CudaRuntimeError, OwnedCudaStream};
pub use augur_cuda_sys::AugurCudaM285FillSpanArgsV1 as CudaM285RawFillSpanArgsV1;

const OP: &str = "cuda_m285_fill_span_submit_v1";

#[derive(Clone, Copy, Debug)]
pub struct CudaM285FillSpanV1 {
    raw: CudaM285RawFillSpanArgsV1,
    device_ordinal: i32,
}

impl CudaM285FillSpanV1 {
    pub fn bind(destination: CudaBufferView, pattern_seed: u64) -> Result<Self, CudaRuntimeError> {
        if destination.len_bytes() == 0 {
            return Err(CudaRuntimeError::out_of_bounds());
        }
        destination.checked_range(OP)?;
        let mut raw = CudaM285RawFillSpanArgsV1::new();
        raw.allocation_base = destination.device_ptr().raw_value();
        raw.allocation_extent_bytes = u64::try_from(destination.allocation_len_bytes())
            .map_err(|_| CudaRuntimeError::range_overflow())?;
        raw.view_offset_bytes = u64::try_from(destination.offset_bytes())
            .map_err(|_| CudaRuntimeError::range_overflow())?;
        raw.view_extent_bytes = u64::try_from(destination.len_bytes())
            .map_err(|_| CudaRuntimeError::range_overflow())?;
        raw.pattern_seed = pattern_seed;
        Ok(Self {
            raw,
            device_ordinal: destination.device_ordinal(),
        })
    }

    pub const fn raw(&self) -> &CudaM285RawFillSpanArgsV1 {
        &self.raw
    }

    pub fn submit(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.device_ordinal {
            return Err(CudaRuntimeError::device_mismatch());
        }
        // SAFETY: `&self.raw` was populated by `bind` from a non-empty view whose
        // range was proven in-allocation by checked_range; the ordinal guard pins the
        // stream to the view's device. Submission is asynchronous, so the caller must
        // keep the destination buffer alive until the stream drains (see findings).
        let result = unsafe {
            augur_cuda_sys::augur_cuda_m285_fill_span_submit_v1(
                self.device_ordinal,
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result(OP, result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_runtime::test_buffer_view;

    #[test]
    fn binding_preserves_allocation_and_nonzero_view_offset() {
        let value = CudaM285FillSpanV1::bind(test_buffer_view(0x1000, 2, 64, 512), 17)
            .expect("valid M285 fill span");
        assert_eq!(value.device_ordinal, 2);
        assert_eq!(value.raw.allocation_base, 0x1000);
        assert_eq!(value.raw.allocation_extent_bytes, 576);
        assert_eq!(value.raw.view_offset_bytes, 64);
        assert_eq!(value.raw.view_extent_bytes, 512);
        assert_eq!(value.raw.pattern_seed, 17);
        assert_eq!(value.raw.reserved, 0);
    }

    #[test]
    fn binding_rejects_allocation_address_overflow() {
        let error =
            CudaM285FillSpanV1::bind(test_buffer_view(u64::MAX - 127, 0, 0, 256), 17).unwrap_err();
        assert_eq!(error.kind, crate::CudaRuntimeErrorKind::RangeOverflow);
    }
}
