#![cfg(feature = "cuda")]

use augur_cuda_sys::AugurCudaSwigluF32ArgsV1;

use crate::{CudaBufferView, CudaF32View, CudaRuntimeError, CudaRuntimeErrorKind, OwnedCudaStream};

const OPERATION: &str = "cuda_swiglu_f32_v1";

#[derive(Clone, Copy, Debug)]
pub struct CudaSwigluF32ArgsV1 {
    raw: AugurCudaSwigluF32ArgsV1,
    device_ordinal: i32,
}

impl CudaSwigluF32ArgsV1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        gate: CudaF32View,
        up: CudaF32View,
        output: CudaF32View,
        limit: f32,
    ) -> Result<Self, CudaRuntimeError> {
        let ordinal = stream.device_ordinal();
        for view in [gate.buffer_view(), up.buffer_view(), output.buffer_view()] {
            if view.device_ordinal() != ordinal {
                return Err(validation(CudaRuntimeErrorKind::DeviceMismatch));
            }
        }
        let elements = gate.element_count();
        if elements == 0
            || up.element_count() != elements
            || output.element_count() != elements
            || !limit.is_finite()
            || limit <= 0.0
        {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let mut raw = AugurCudaSwigluF32ArgsV1::new();
        raw.gate = address(gate.buffer_view())?;
        raw.gate_len_bytes = gate.buffer_view().len_bytes() as u64;
        raw.up = address(up.buffer_view())?;
        raw.up_len_bytes = up.buffer_view().len_bytes() as u64;
        raw.output = address(output.buffer_view())?;
        raw.output_len_bytes = output.buffer_view().len_bytes() as u64;
        raw.elements = elements
            .try_into()
            .map_err(|_| validation(CudaRuntimeErrorKind::InvalidDimensions))?;
        raw.limit_bits = limit.to_bits();
        Ok(Self {
            raw,
            device_ordinal: ordinal,
        })
    }

    pub fn launch(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        // SAFETY: `&self.raw` is owned args built by `bind`, which checked equal element
        // counts, a finite positive limit, and per-view device ordinals; addresses come
        // from bounds-checked CudaF32Views and validate_stream re-pins the device.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_swiglu_f32_v1(stream.device_ordinal(), stream.raw()?, &self.raw)
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
            augur_cuda_sys::augur_cuda_swiglu_f32_submit_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result("cuda_swiglu_f32_submit_v1", result)
    }

    fn validate_stream(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.device_ordinal {
            return Err(validation(CudaRuntimeErrorKind::DeviceMismatch));
        }
        Ok(())
    }
}

fn validation(kind: CudaRuntimeErrorKind) -> CudaRuntimeError {
    CudaRuntimeError::validation(OPERATION, kind)
}

fn address(view: CudaBufferView) -> Result<u64, CudaRuntimeError> {
    view.device_ptr()
        .raw_value()
        .checked_add(view.offset_bytes() as u64)
        .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))
}
