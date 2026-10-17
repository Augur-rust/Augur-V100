#![cfg(feature = "cuda")]

use crate::{CudaBufferView, CudaF32View, CudaRuntimeError, CudaRuntimeErrorKind, OwnedCudaStream};
use augur_core::PhysicalLayoutId;
use augur_cuda_sys::AugurCudaF8E4M3F32MatvecArgsV1;

const OP: &str = "cuda_f8e4m3_f32_matvec_v1";
const ROW32_OP: &str = "cuda_f8e4m3_f32_matvec_row32_v1";
const ROW_SCALE_OP: &str = "cuda_f8e4m3_f32_matvec_row_scale_v1";


#[derive(Clone, Copy, Debug)]
pub struct CudaF8E4M3WeightViewV1 {
    view: CudaBufferView,
    rows: u32,
    cols: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct CudaF8E4M3Row32WeightViewV1 {
    view: CudaBufferView,
    rows: u32,
    cols: u32,
}
impl CudaF8E4M3Row32WeightViewV1 {
    pub fn new(
        view: CudaBufferView,
        layout: PhysicalLayoutId,
        rows: usize,
        cols: usize,
    ) -> Result<Self, CudaRuntimeError> {
        let bytes = rows
            .checked_mul(cols)
            .ok_or_else(|| row32_error(CudaRuntimeErrorKind::RangeOverflow))?;
        if layout != PhysicalLayoutId::DEVICE_ROW32_F8E4M3_CUDA_V1
            || rows == 0
            || cols == 0
            || !rows.is_multiple_of(128)
            || !cols.is_multiple_of(128)
            || view.len_bytes() != bytes
        {
            return Err(row32_error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        Ok(Self {
            view,
            rows: rows
                .try_into()
                .map_err(|_| row32_error(CudaRuntimeErrorKind::InvalidDimensions))?,
            cols: cols
                .try_into()
                .map_err(|_| row32_error(CudaRuntimeErrorKind::InvalidDimensions))?,
        })
    }
}
impl CudaF8E4M3WeightViewV1 {
    pub fn new(
        view: CudaBufferView,
        layout: PhysicalLayoutId,
        rows: usize,
        cols: usize,
    ) -> Result<Self, CudaRuntimeError> {
        let bytes = rows
            .checked_mul(cols)
            .ok_or_else(|| error(CudaRuntimeErrorKind::RangeOverflow))?;
        if layout != PhysicalLayoutId::DEVICE_CONTIGUOUS_F8E4M3_CUDA_V1
            || rows == 0
            || cols == 0
            || !rows.is_multiple_of(128)
            || !cols.is_multiple_of(128)
            || view.len_bytes() != bytes
        {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        Ok(Self {
            view,
            rows: rows
                .try_into()
                .map_err(|_| error(CudaRuntimeErrorKind::InvalidDimensions))?,
            cols: cols
                .try_into()
                .map_err(|_| error(CudaRuntimeErrorKind::InvalidDimensions))?,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CudaF8E4M3F32MatvecArgsV1 {
    raw: AugurCudaF8E4M3F32MatvecArgsV1,
    ordinal: i32,
    #[cfg_attr(
        not(feature = "p0-prefill-v22-projection-resume-batch-exact"),
        allow(dead_code)
    )]
    tokens: u32,
}
impl CudaF8E4M3F32MatvecArgsV1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        weight: CudaF8E4M3WeightViewV1,
        scales: CudaF32View,
        input: CudaF32View,
        output: CudaF32View,
    ) -> Result<Self, CudaRuntimeError> {
        let expected_scales = (weight.rows as usize / 128) * (weight.cols as usize / 128);
        if scales.element_count() != expected_scales
            || input.element_count() != weight.cols as usize
            || output.element_count() != weight.rows as usize
            || [
                weight.view,
                scales.buffer_view(),
                input.buffer_view(),
                output.buffer_view(),
            ]
            .iter()
            .any(|v| v.device_ordinal() != stream.device_ordinal())
        {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let mut raw = AugurCudaF8E4M3F32MatvecArgsV1::new();
        raw.weight = addr(weight.view)?;
        raw.weight_len_bytes = weight.view.len_bytes() as u64;
        raw.scales = addr(scales.buffer_view())?;
        raw.scales_len_bytes = scales.buffer_view().len_bytes() as u64;
        raw.input = addr(input.buffer_view())?;
        raw.input_len_bytes = input.buffer_view().len_bytes() as u64;
        raw.output = addr(output.buffer_view())?;
        raw.output_len_bytes = output.buffer_view().len_bytes() as u64;
        raw.rows = weight.rows;
        raw.cols = weight.cols;
        Ok(Self {
            raw,
            ordinal: stream.device_ordinal(),
            tokens: 1,
        })
    }

    
    pub fn launch(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.ordinal {
            return Err(error(CudaRuntimeErrorKind::DeviceMismatch));
        }
        let result = unsafe {
            augur_cuda_sys::augur_cuda_f8e4m3_f32_matvec_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result(OP, result)
    }

    pub fn submit_for_plan(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.ordinal {
            return Err(error(CudaRuntimeErrorKind::DeviceMismatch));
        }
        let result = unsafe {
            augur_cuda_sys::augur_cuda_f8e4m3_f32_matvec_submit_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result("cuda_f8e4m3_f32_matvec_submit_v1", result)
    }

    

    pub fn launch_bitdecode(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.ordinal {
            return Err(error(CudaRuntimeErrorKind::DeviceMismatch));
        }
        let result = unsafe {
            augur_cuda_sys::augur_cuda_f8e4m3_f32_matvec_bitdecode_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result("cuda_f8e4m3_f32_matvec_bitdecode_v1", result)
    }

    pub fn submit_bitdecode_for_plan(
        &self,
        stream: &OwnedCudaStream,
    ) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.ordinal {
            return Err(error(CudaRuntimeErrorKind::DeviceMismatch));
        }
        let result = unsafe {
            augur_cuda_sys::augur_cuda_f8e4m3_f32_matvec_bitdecode_submit_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result("cuda_f8e4m3_f32_matvec_bitdecode_submit_v1", result)
    }

    /// Explicit warp-reduced F32 profile over128x128 block scales. This reuses
    /// the native bit-exact E4M3 decoder but changes summation order relative to
    /// the serial row kernel. It is not a transparent numerical replacement.
    pub fn submit_warp_fp32_for_plan(
        &self,
        stream: &OwnedCudaStream,
    ) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.ordinal {
            return Err(error(CudaRuntimeErrorKind::DeviceMismatch));
        }
        // SAFETY: bind validated exact scalar/tile geometry and device ranges;
        // retained callers own them through stream completion. Native checks
        // packed-load alignment before enqueueing. This call does not wait.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_f8e4m3_f32_matvec_warp_fp32_submit_v1(
                stream.device_ordinal(), stream.raw()?, &self.raw,
            )
        };
        CudaRuntimeError::from_result("cuda_f8e4m3_f32_matvec_warp_fp32_submit_v1", result)
    }

    

    
}

#[derive(Clone, Copy, Debug)]
pub struct CudaF8E4M3RowScaleF32MatvecArgsV1 {
    raw: AugurCudaF8E4M3F32MatvecArgsV1,
    ordinal: i32,
}

impl CudaF8E4M3RowScaleF32MatvecArgsV1 {
    /// Bind a contiguous row-major F8_E4M3 matrix with one FP32 scale per row.
    /// This is the native Hugging Face safetensors convention used by the
    /// Qwen output head, separate from the 128x128 tile-scale ABI.
    pub fn bind(
        stream: &OwnedCudaStream,
        weight: CudaF8E4M3WeightViewV1,
        row_scales: CudaF32View,
        input: CudaF32View,
        output: CudaF32View,
    ) -> Result<Self, CudaRuntimeError> {
        if row_scales.element_count() != weight.rows as usize
            || input.element_count() != weight.cols as usize
            || output.element_count() != weight.rows as usize
            || [
                weight.view,
                row_scales.buffer_view(),
                input.buffer_view(),
                output.buffer_view(),
            ]
            .iter()
            .any(|view| view.device_ordinal() != stream.device_ordinal())
        {
            return Err(row_scale_error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let mut raw = AugurCudaF8E4M3F32MatvecArgsV1::new();
        raw.weight = addr(weight.view)?;
        raw.weight_len_bytes = weight.view.len_bytes() as u64;
        raw.scales = addr(row_scales.buffer_view())?;
        raw.scales_len_bytes = row_scales.buffer_view().len_bytes() as u64;
        raw.input = addr(input.buffer_view())?;
        raw.input_len_bytes = input.buffer_view().len_bytes() as u64;
        raw.output = addr(output.buffer_view())?;
        raw.output_len_bytes = output.buffer_view().len_bytes() as u64;
        raw.rows = weight.rows;
        raw.cols = weight.cols;
        Ok(Self {
            raw,
            ordinal: stream.device_ordinal(),
        })
    }

    pub fn launch(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.ordinal {
            return Err(row_scale_error(CudaRuntimeErrorKind::DeviceMismatch));
        }
        let result = unsafe {
            augur_cuda_sys::augur_cuda_f8e4m3_f32_matvec_row_scale_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result(ROW_SCALE_OP, result)
    }

    pub fn submit_for_plan(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.ordinal {
            return Err(row_scale_error(CudaRuntimeErrorKind::DeviceMismatch));
        }
        let result = unsafe {
            augur_cuda_sys::augur_cuda_f8e4m3_f32_matvec_row_scale_submit_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result("cuda_f8e4m3_f32_matvec_row_scale_submit_v1", result)
    }
}
#[derive(Clone, Copy, Debug)]
pub struct CudaF8E4M3F32MatvecRow32ArgsV1 {
    raw: AugurCudaF8E4M3F32MatvecArgsV1,
    ordinal: i32,
}
impl CudaF8E4M3F32MatvecRow32ArgsV1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        weight: CudaF8E4M3Row32WeightViewV1,
        scales: CudaF32View,
        input: CudaF32View,
        output: CudaF32View,
    ) -> Result<Self, CudaRuntimeError> {
        let expected_scales = (weight.rows as usize / 128) * (weight.cols as usize / 128);
        if scales.element_count() != expected_scales
            || input.element_count() != weight.cols as usize
            || output.element_count() != weight.rows as usize
            || [
                weight.view,
                scales.buffer_view(),
                input.buffer_view(),
                output.buffer_view(),
            ]
            .iter()
            .any(|view| view.device_ordinal() != stream.device_ordinal())
        {
            return Err(row32_error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let mut raw = AugurCudaF8E4M3F32MatvecArgsV1::new();
        raw.weight = addr(weight.view)?;
        raw.weight_len_bytes = weight.view.len_bytes() as u64;
        raw.scales = addr(scales.buffer_view())?;
        raw.scales_len_bytes = scales.buffer_view().len_bytes() as u64;
        raw.input = addr(input.buffer_view())?;
        raw.input_len_bytes = input.buffer_view().len_bytes() as u64;
        raw.output = addr(output.buffer_view())?;
        raw.output_len_bytes = output.buffer_view().len_bytes() as u64;
        raw.rows = weight.rows;
        raw.cols = weight.cols;
        Ok(Self {
            raw,
            ordinal: stream.device_ordinal(),
        })
    }

    pub fn launch(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.ordinal {
            return Err(row32_error(CudaRuntimeErrorKind::DeviceMismatch));
        }
        let result = unsafe {
            augur_cuda_sys::augur_cuda_f8e4m3_f32_matvec_row32_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result(ROW32_OP, result)
    }

    pub fn submit_for_plan(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.ordinal {
            return Err(row32_error(CudaRuntimeErrorKind::DeviceMismatch));
        }
        let result = unsafe {
            augur_cuda_sys::augur_cuda_f8e4m3_f32_matvec_row32_submit_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result("cuda_f8e4m3_f32_matvec_row32_submit_v1", result)
    }
}
fn error(kind: CudaRuntimeErrorKind) -> CudaRuntimeError {
    CudaRuntimeError::validation(OP, kind)
}
fn row_scale_error(kind: CudaRuntimeErrorKind) -> CudaRuntimeError {
    CudaRuntimeError::validation(ROW_SCALE_OP, kind)
}
fn row32_error(kind: CudaRuntimeErrorKind) -> CudaRuntimeError {
    CudaRuntimeError::validation(ROW32_OP, kind)
}
fn addr(view: CudaBufferView) -> Result<u64, CudaRuntimeError> {
    view.device_ptr()
        .raw_value()
        .checked_add(view.offset_bytes() as u64)
        .ok_or_else(|| error(CudaRuntimeErrorKind::RangeOverflow))
}
