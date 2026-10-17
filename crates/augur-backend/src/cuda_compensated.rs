#![cfg(feature = "cuda")]

use augur_cuda_sys::{self, AugurCudaCompensatedAccumF32ArgsV1};

use crate::{CudaF32View, CudaRuntimeError, CudaRuntimeErrorKind, OwnedCudaStream};

#[derive(Clone, Copy, Debug)]
pub struct CudaCompensatedAccumF32ArgsV1 {
    raw: AugurCudaCompensatedAccumF32ArgsV1,
    device_ordinal: i32,
}

impl CudaCompensatedAccumF32ArgsV1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        input: CudaF32View,
        output: CudaF32View,
        compensation: CudaF32View,
        weight: f32,
        replace: bool,
    ) -> Result<Self, CudaRuntimeError> {
        const OP: &str = "cuda_compensated_accum_f32_v1";
        if input.element_count() == 0
            || input.element_count() != output.element_count()
            || input.element_count() != compensation.element_count()
            || input.buffer_view().device_ordinal() != stream.device_ordinal()
            || output.buffer_view().device_ordinal() != stream.device_ordinal()
            || compensation.buffer_view().device_ordinal() != stream.device_ordinal()
            || input.buffer_view().device_ptr() == output.buffer_view().device_ptr()
            || input.buffer_view().device_ptr() == compensation.buffer_view().device_ptr()
            || output.buffer_view().device_ptr() == compensation.buffer_view().device_ptr()
            || !weight.is_finite()
        {
            return Err(CudaRuntimeError::validation(
                OP,
                CudaRuntimeErrorKind::InvalidDimensions,
            ));
        }
        let mut raw = AugurCudaCompensatedAccumF32ArgsV1::new();
        raw.input_f32 = address(input)?;
        raw.input_len_bytes = input.buffer_view().len_bytes() as u64;
        raw.output_f32 = address(output)?;
        raw.output_len_bytes = output.buffer_view().len_bytes() as u64;
        raw.compensation_f32 = address(compensation)?;
        raw.compensation_len_bytes = compensation.buffer_view().len_bytes() as u64;
        raw.elements = input.element_count() as u32;
        raw.weight_bits = weight.to_bits();
        raw.replace = u32::from(replace);
        Ok(Self {
            raw,
            device_ordinal: stream.device_ordinal(),
        })
    }

    pub fn submit_for_plan(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate(stream)?;
        let result = unsafe {
            augur_cuda_sys::augur_cuda_compensated_accum_f32_submit_v1(
                self.device_ordinal,
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result("cuda_compensated_accum_f32_submit_v1", result)
    }

    pub fn launch(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate(stream)?;
        let result = unsafe {
            augur_cuda_sys::augur_cuda_compensated_accum_f32_v1(
                self.device_ordinal,
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result("cuda_compensated_accum_f32_v1", result)
    }

    fn validate(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.device_ordinal {
            return Err(CudaRuntimeError::validation(
                "cuda_compensated_accum_f32_v1",
                CudaRuntimeErrorKind::DeviceMismatch,
            ));
        }
        Ok(())
    }
}

fn address(view: CudaF32View) -> Result<u64, CudaRuntimeError> {
    view.buffer_view()
        .device_ptr()
        .raw_value()
        .checked_add(view.buffer_view().offset_bytes() as u64)
        .ok_or_else(|| {
            CudaRuntimeError::validation(
                "cuda_compensated_accum_f32_v1",
                CudaRuntimeErrorKind::RangeOverflow,
            )
        })
}
