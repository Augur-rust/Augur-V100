#![cfg(feature = "cuda")]

use augur_cuda_sys::AugurCudaGqaRepeatF32ArgsV1;

use crate::{CudaBufferView, CudaF32View, CudaRuntimeError, CudaRuntimeErrorKind, OwnedCudaStream};

const OPERATION: &str = "cuda_gqa_repeat_f32_v1";

#[derive(Clone, Copy, Debug)]
pub struct CudaGqaRepeatF32ArgsV1 {
    raw: AugurCudaGqaRepeatF32ArgsV1,
    device_ordinal: i32,
}

impl CudaGqaRepeatF32ArgsV1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        input: CudaF32View,
        output: CudaF32View,
        head_dim: usize,
        repeats: usize,
    ) -> Result<Self, CudaRuntimeError> {
        let ordinal = stream.device_ordinal();
        if input.buffer_view().device_ordinal() != ordinal
            || output.buffer_view().device_ordinal() != ordinal
            || !valid_geometry(
                input.element_count(),
                output.element_count(),
                head_dim,
                repeats,
            )
        {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        if overlaps(input.buffer_view(), output.buffer_view())? {
            return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
        }

        let mut raw = AugurCudaGqaRepeatF32ArgsV1::new();
        raw.input = address(input.buffer_view())?;
        raw.input_len_bytes = to_u64(input.buffer_view().len_bytes())?;
        raw.output = address(output.buffer_view())?;
        raw.output_len_bytes = to_u64(output.buffer_view().len_bytes())?;
        raw.source_elements = u32::try_from(input.element_count())
            .map_err(|_| validation(CudaRuntimeErrorKind::InvalidDimensions))?;
        raw.head_dim = u32::try_from(head_dim)
            .map_err(|_| validation(CudaRuntimeErrorKind::InvalidDimensions))?;
        raw.repeats = u32::try_from(repeats)
            .map_err(|_| validation(CudaRuntimeErrorKind::InvalidDimensions))?;
        Ok(Self {
            raw,
            device_ordinal: ordinal,
        })
    }

    pub const fn as_raw(&self) -> &AugurCudaGqaRepeatF32ArgsV1 {
        &self.raw
    }

    pub fn launch(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        // SAFETY: `raw` is constructed by `bind`, which validates device identity,
        // grouped geometry, extents, alignment, and non-overlap. The stream is live.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_gqa_repeat_f32_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result(OPERATION, result)
    }

    #[doc(hidden)]
    pub fn submit_for_plan(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        // SAFETY: Same bind-established contract as `launch`; buffers remain alive
        // until the caller drains the stream.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_gqa_repeat_f32_submit_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result("cuda_gqa_repeat_f32_submit_v1", result)
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

fn valid_geometry(
    source_elements: usize,
    output_elements: usize,
    head_dim: usize,
    repeats: usize,
) -> bool {
    source_elements != 0
        && head_dim != 0
        && repeats != 0
        && source_elements % head_dim == 0
        && source_elements.checked_mul(repeats) == Some(output_elements)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binding_accepts_grouped_geometry() {
        assert!(valid_geometry(1_024, 4_096, 128, 4));
    }

    #[test]
    fn binding_rejects_invalid_grouping_and_alias() {
        assert!(!valid_geometry(1_024, 4_096, 127, 4));
    }

    #[test]
    fn binding_rejects_output_extent_mismatch() {
        assert!(!valid_geometry(1_024, 2_048, 128, 4));
    }
}
