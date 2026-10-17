#![cfg(feature = "cuda")]

use augur_cuda_sys::AugurCudaAddF32ArgsV1;

use crate::{CudaBufferView, CudaF32View, CudaRuntimeError, CudaRuntimeErrorKind, OwnedCudaStream};

const OPERATION: &str = "cuda_add_f32_v1";

#[derive(Clone, Copy, Debug)]
pub struct CudaAddF32ArgsV1 {
    raw: AugurCudaAddF32ArgsV1,
    device_ordinal: i32,
}

impl CudaAddF32ArgsV1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        a: CudaF32View,
        b: CudaF32View,
        output: CudaF32View,
    ) -> Result<Self, CudaRuntimeError> {
        let ordinal = stream.device_ordinal();
        for view in [a.buffer_view(), b.buffer_view(), output.buffer_view()] {
            if view.device_ordinal() != ordinal {
                return Err(validation(CudaRuntimeErrorKind::DeviceMismatch));
            }
        }
        let elements = a.element_count();
        if elements == 0 || b.element_count() != elements || output.element_count() != elements {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        if overlaps(output.buffer_view(), a.buffer_view())?
            || overlaps(output.buffer_view(), b.buffer_view())?
        {
            return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
        }

        let mut raw = AugurCudaAddF32ArgsV1::new();
        raw.a = address(a.buffer_view())?;
        raw.a_len_bytes = to_u64(a.buffer_view().len_bytes())?;
        raw.b = address(b.buffer_view())?;
        raw.b_len_bytes = to_u64(b.buffer_view().len_bytes())?;
        raw.output = address(output.buffer_view())?;
        raw.output_len_bytes = to_u64(output.buffer_view().len_bytes())?;
        raw.elements = u32::try_from(elements)
            .map_err(|_| validation(CudaRuntimeErrorKind::InvalidDimensions))?;
        Ok(Self {
            raw,
            device_ordinal: ordinal,
        })
    }

    pub const fn as_raw(&self) -> &AugurCudaAddF32ArgsV1 {
        &self.raw
    }

    pub fn launch(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        // SAFETY: `&self.raw` is owned args built by `bind`, which checked device
        // ordinals, equal element counts, and output/input non-overlap (`overlaps`);
        // validate_stream re-pins the bound device and `stream.raw()?` is a live handle.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_add_f32_v1(stream.device_ordinal(), stream.raw()?, &self.raw)
        };
        CudaRuntimeError::from_result(OPERATION, result)
    }

    #[doc(hidden)]
    pub fn submit_for_plan(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        // SAFETY: Same bind-established contract as `launch`; completion is asynchronous,
        // so the caller must keep the viewed buffers alive until the stream drains
        // (CudaBufferView carries no lifetime — see findings).
        let result = unsafe {
            augur_cuda_sys::augur_cuda_add_f32_submit_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result("cuda_add_f32_submit_v1", result)
    }

    fn validate_stream(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if self.device_ordinal != stream.device_ordinal() {
            return Err(validation(CudaRuntimeErrorKind::DeviceMismatch));
        }
        Ok(())
    }
}

fn validation(kind: CudaRuntimeErrorKind) -> CudaRuntimeError {
    CudaRuntimeError::validation(OPERATION, kind)
}

fn to_u64(value: usize) -> Result<u64, CudaRuntimeError> {
    u64::try_from(value).map_err(|_| validation(CudaRuntimeErrorKind::RangeOverflow))
}

fn address(view: CudaBufferView) -> Result<u64, CudaRuntimeError> {
    view.device_ptr()
        .raw_value()
        .checked_add(to_u64(view.offset_bytes())?)
        .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))
}

fn overlaps(a: CudaBufferView, b: CudaBufferView) -> Result<bool, CudaRuntimeError> {
    if a.device_ptr() != b.device_ptr() {
        return Ok(false);
    }
    let a_start = a.offset_bytes();
    let a_end = a_start
        .checked_add(a.len_bytes())
        .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))?;
    let b_start = b.offset_bytes();
    let b_end = b_start
        .checked_add(b.len_bytes())
        .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))?;
    Ok(a_start < b_end && b_start < a_end)
}
