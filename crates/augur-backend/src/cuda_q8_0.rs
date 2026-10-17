#![cfg(feature = "cuda")]

//! Checked binding for a row-major GGML Q8_0 matrix-vector product.
//!
//! The view deliberately preserves the GGML wire bytes on device.  Each
//! logical row is `cols / 32` blocks of two-byte FP16 scale plus 32 signed
//! int8 codes.  This is a bounded quantized component primitive; it does not
//! imply a complete model or decode path.

use crate::{CudaBufferView, CudaF32View, CudaRuntimeError, CudaRuntimeErrorKind, OwnedCudaStream};
use augur_core::PhysicalLayoutId;
use augur_cuda_sys::{
    AugurCudaQ8_0F32MatvecArgsV1, AugurCudaQ8_0Q8_1MatvecArgsV1, AugurCudaQ8_1ActivationQuantizeArgsV1,
};

const OPERATION: &str = "cuda_q8_0_f32_matvec_v1";
const Q8_1_QUANTIZE_OPERATION: &str = "cuda_q8_1_activation_quantize_v1";
const Q8_1_MATVEC_OPERATION: &str = "cuda_q8_0_q8_1_matvec_v1";
const BLOCK_ELEMENTS: usize = 32;
const BLOCK_BYTES: usize = 34;
pub const Q8_1_ACTIVATION_BLOCK_ELEMENTS: usize = 32;
pub const Q8_1_ACTIVATION_BLOCK_BYTES: usize = 36;
/// Shared activation primitive build identity; diagnostic-only, default OFF.
pub const Q8_1_DIAGNOSTIC_TIES_AWAY: bool = false;
pub const QWEN35_DIAGNOSTIC_SPLIT_ROPE: bool = false;
pub const QWEN35_TARGET_SPLIT_ROPE: bool = cfg!(feature = "qwen35-target-split-rope");

fn validation(kind: CudaRuntimeErrorKind) -> CudaRuntimeError {
    CudaRuntimeError::validation(OPERATION, kind)
}

fn q8_1_validation(operation: &'static str, kind: CudaRuntimeErrorKind) -> CudaRuntimeError {
    CudaRuntimeError::validation(operation, kind)
}

/// A device view containing one llama.cpp-compatible Q8_1 activation vector.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaQ8_1ActivationViewV1 {
    view: CudaBufferView,
    elements: u32,
}

impl CudaQ8_1ActivationViewV1 {
    pub fn new(
        view: CudaBufferView,
        layout: PhysicalLayoutId,
        elements: usize,
    ) -> Result<Self, CudaRuntimeError> {
        if layout != PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_1_ACTIVATION_CUDA_V1 {
            return Err(q8_1_validation(
                Q8_1_QUANTIZE_OPERATION,
                CudaRuntimeErrorKind::InvalidLayout,
            ));
        }
        if elements == 0 || !elements.is_multiple_of(Q8_1_ACTIVATION_BLOCK_ELEMENTS) {
            return Err(q8_1_validation(
                Q8_1_QUANTIZE_OPERATION,
                CudaRuntimeErrorKind::InvalidDimensions,
            ));
        }
        let expected_bytes = elements
            .checked_div(Q8_1_ACTIVATION_BLOCK_ELEMENTS)
            .and_then(|blocks| blocks.checked_mul(Q8_1_ACTIVATION_BLOCK_BYTES))
            .ok_or_else(|| {
                q8_1_validation(Q8_1_QUANTIZE_OPERATION, CudaRuntimeErrorKind::RangeOverflow)
            })?;
        if view.len_bytes() != expected_bytes {
            return Err(q8_1_validation(
                Q8_1_QUANTIZE_OPERATION,
                CudaRuntimeErrorKind::OutOfBounds,
            ));
        }
        let address = address_for(Q8_1_QUANTIZE_OPERATION, view)?;
        if address % core::mem::align_of::<u32>() as u64 != 0 {
            return Err(q8_1_validation(
                Q8_1_QUANTIZE_OPERATION,
                CudaRuntimeErrorKind::MisalignedDeviceAddress,
            ));
        }
        Ok(Self {
            view,
            elements: u32::try_from(elements).map_err(|_| {
                q8_1_validation(
                    Q8_1_QUANTIZE_OPERATION,
                    CudaRuntimeErrorKind::InvalidDimensions,
                )
            })?,
        })
    }

    pub const fn elements(self) -> usize {
        self.elements as usize
    }

    pub const fn buffer_view(self) -> CudaBufferView {
        self.view
    }
}

/// Bound arguments for asynchronous f32-to-Q8_1 activation conversion.
#[derive(Clone, Copy, Debug)]
pub struct CudaQ8_1ActivationQuantizeArgsV1 {
    raw: AugurCudaQ8_1ActivationQuantizeArgsV1,
    device_ordinal: i32,
}

impl CudaQ8_1ActivationQuantizeArgsV1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        input: CudaF32View,
        output: CudaQ8_1ActivationViewV1,
    ) -> Result<Self, CudaRuntimeError> {
        let ordinal = stream.device_ordinal();
        let input_view = input.buffer_view();
        let output_view = output.buffer_view();
        if input_view.device_ordinal() != ordinal
            || output_view.device_ordinal() != ordinal
            || input.element_count() != output.elements()
        {
            return Err(q8_1_validation(
                Q8_1_QUANTIZE_OPERATION,
                CudaRuntimeErrorKind::InvalidDimensions,
            ));
        }
        if input_view.overlaps(output_view, Q8_1_QUANTIZE_OPERATION)? {
            return Err(q8_1_validation(
                Q8_1_QUANTIZE_OPERATION,
                CudaRuntimeErrorKind::OverlappingDeviceRanges,
            ));
        }
        let mut raw = AugurCudaQ8_1ActivationQuantizeArgsV1::new();
        raw.input_f32 = address_for(Q8_1_QUANTIZE_OPERATION, input_view)?;
        raw.input_len_bytes = u64::try_from(input_view.len_bytes()).map_err(|_| {
            q8_1_validation(Q8_1_QUANTIZE_OPERATION, CudaRuntimeErrorKind::RangeOverflow)
        })?;
        raw.output_q8_1 = address_for(Q8_1_QUANTIZE_OPERATION, output_view)?;
        raw.output_len_bytes = u64::try_from(output_view.len_bytes()).map_err(|_| {
            q8_1_validation(Q8_1_QUANTIZE_OPERATION, CudaRuntimeErrorKind::RangeOverflow)
        })?;
        raw.elements = output.elements;
        Ok(Self {
            raw,
            device_ordinal: ordinal,
        })
    }

    pub const fn as_raw(&self) -> &AugurCudaQ8_1ActivationQuantizeArgsV1 {
        &self.raw
    }

    pub fn launch(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        // SAFETY: binding checks device ownership, exact extents, layout, and
        // disjoint ranges. The synchronous entry point completes before return.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_q8_1_activation_quantize_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result(Q8_1_QUANTIZE_OPERATION, result)
    }

    pub fn submit_for_plan(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        // SAFETY: the owners behind the checked views outlive stream completion.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_q8_1_activation_quantize_submit_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result("cuda_q8_1_activation_quantize_submit_v1", result)
    }

    fn validate_stream(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.device_ordinal {
            return Err(q8_1_validation(
                Q8_1_QUANTIZE_OPERATION,
                CudaRuntimeErrorKind::DeviceMismatch,
            ));
        }
        Ok(())
    }
}

/// A device view containing exactly `rows` GGML Q8_0 rows.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaQ8_0WeightViewV1 {
    view: CudaBufferView,
    rows: u32,
    cols: u32,
    blocks_per_row: u32,
}

impl CudaQ8_0WeightViewV1 {
    pub fn new(
        view: CudaBufferView,
        layout: PhysicalLayoutId,
        rows: usize,
        cols: usize,
    ) -> Result<Self, CudaRuntimeError> {
        if layout != PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_0_CUDA_V1 {
            return Err(validation(CudaRuntimeErrorKind::InvalidLayout));
        }
        if rows == 0 || cols == 0 || !cols.is_multiple_of(BLOCK_ELEMENTS) {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let blocks_per_row = cols
            .checked_div(BLOCK_ELEMENTS)
            .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))?;
        let block_count = rows
            .checked_mul(blocks_per_row)
            .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))?;
        let expected_bytes = block_count
            .checked_mul(BLOCK_BYTES)
            .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))?;
        if view.len_bytes() != expected_bytes {
            return Err(validation(CudaRuntimeErrorKind::OutOfBounds));
        }
        let address = view
            .device_ptr()
            .raw_value()
            .checked_add(
                u64::try_from(view.offset_bytes())
                    .map_err(|_| validation(CudaRuntimeErrorKind::RangeOverflow))?,
            )
            .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))?;
        if address % core::mem::align_of::<u16>() as u64 != 0 {
            return Err(validation(CudaRuntimeErrorKind::MisalignedDeviceAddress));
        }
        Ok(Self {
            view,
            rows: u32::try_from(rows)
                .map_err(|_| validation(CudaRuntimeErrorKind::InvalidDimensions))?,
            cols: u32::try_from(cols)
                .map_err(|_| validation(CudaRuntimeErrorKind::InvalidDimensions))?,
            blocks_per_row: u32::try_from(blocks_per_row)
                .map_err(|_| validation(CudaRuntimeErrorKind::InvalidDimensions))?,
        })
    }

    pub const fn rows(self) -> u32 {
        self.rows
    }

    pub const fn cols(self) -> u32 {
        self.cols
    }

    pub const fn blocks_per_row(self) -> u32 {
        self.blocks_per_row
    }

    pub const fn buffer_view(self) -> CudaBufferView {
        self.view
    }
}

/// Bound arguments for one asynchronous Q8_0 f32 matvec submission.
#[derive(Clone, Copy, Debug)]
pub struct CudaQ8_0F32MatvecArgsV1 {
    raw: AugurCudaQ8_0F32MatvecArgsV1,
    device_ordinal: i32,
}

impl CudaQ8_0F32MatvecArgsV1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        weight: CudaQ8_0WeightViewV1,
        input: CudaF32View,
        output: CudaF32View,
    ) -> Result<Self, CudaRuntimeError> {
        let ordinal = stream.device_ordinal();
        let weight_view = weight.view;
        let input_view = input.buffer_view();
        let output_view = output.buffer_view();
        if [weight_view, input_view, output_view]
            .iter()
            .any(|view| view.device_ordinal() != ordinal)
            || input.element_count() != weight.cols as usize
            || output.element_count() < weight.rows as usize
        {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        if weight_view.overlaps(input_view, OPERATION)?
            || weight_view.overlaps(output_view, OPERATION)?
            || input_view.overlaps(output_view, OPERATION)?
        {
            return Err(validation(CudaRuntimeErrorKind::OverlappingDeviceRanges));
        }

        let mut raw = AugurCudaQ8_0F32MatvecArgsV1::new();
        raw.weight = address(weight_view)?;
        raw.weight_len_bytes = u64::try_from(weight_view.len_bytes())
            .map_err(|_| validation(CudaRuntimeErrorKind::RangeOverflow))?;
        raw.input = address(input_view)?;
        raw.input_len_bytes = u64::try_from(input_view.len_bytes())
            .map_err(|_| validation(CudaRuntimeErrorKind::RangeOverflow))?;
        raw.output = address(output_view)?;
        raw.output_len_bytes = u64::try_from(output_view.len_bytes())
            .map_err(|_| validation(CudaRuntimeErrorKind::RangeOverflow))?;
        raw.rows = weight.rows;
        raw.cols = weight.cols;
        Ok(Self {
            raw,
            device_ordinal: ordinal,
        })
    }

    pub const fn as_raw(&self) -> &AugurCudaQ8_0F32MatvecArgsV1 {
        &self.raw
    }

    pub fn launch(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        // SAFETY: `raw` was populated from checked non-owning views. The
        // caller keeps the owning buffers alive until the synchronous launch
        // returns; the native shim repeats ABI and extent validation.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_q8_0_f32_matvec_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result(OPERATION, result)
    }

    /// Enqueue the kernel without a host synchronization.
    pub fn submit_for_plan(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        // SAFETY: same binding contract as `launch`; owners must outlive the
        // stream completion proof for this asynchronous submission.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_q8_0_f32_matvec_submit_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result("cuda_q8_0_f32_matvec_submit_v1", result)
    }

    fn validate_stream(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.device_ordinal {
            return Err(validation(CudaRuntimeErrorKind::DeviceMismatch));
        }
        Ok(())
    }
}

/// Bound arguments for a Q8_0-weight by Q8_1-activation DP4A matvec.
#[derive(Clone, Copy, Debug)]
pub struct CudaQ8_0Q8_1MatvecArgsV1 {
    raw: AugurCudaQ8_0Q8_1MatvecArgsV1,
    device_ordinal: i32,
}

impl CudaQ8_0Q8_1MatvecArgsV1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        weight: CudaQ8_0WeightViewV1,
        input: CudaQ8_1ActivationViewV1,
        output: CudaF32View,
    ) -> Result<Self, CudaRuntimeError> {
        let ordinal = stream.device_ordinal();
        let weight_view = weight.buffer_view();
        let input_view = input.buffer_view();
        let output_view = output.buffer_view();
        if [weight_view, input_view, output_view]
            .iter()
            .any(|view| view.device_ordinal() != ordinal)
            || input.elements() != weight.cols as usize
            || output.element_count() < weight.rows as usize
        {
            return Err(q8_1_validation(
                Q8_1_MATVEC_OPERATION,
                CudaRuntimeErrorKind::InvalidDimensions,
            ));
        }
        if weight_view.overlaps(input_view, Q8_1_MATVEC_OPERATION)?
            || weight_view.overlaps(output_view, Q8_1_MATVEC_OPERATION)?
            || input_view.overlaps(output_view, Q8_1_MATVEC_OPERATION)?
        {
            return Err(q8_1_validation(
                Q8_1_MATVEC_OPERATION,
                CudaRuntimeErrorKind::OverlappingDeviceRanges,
            ));
        }

        let mut raw = AugurCudaQ8_0Q8_1MatvecArgsV1::new();
        raw.weight = address_for(Q8_1_MATVEC_OPERATION, weight_view)?;
        raw.weight_len_bytes = u64::try_from(weight_view.len_bytes()).map_err(|_| {
            q8_1_validation(Q8_1_MATVEC_OPERATION, CudaRuntimeErrorKind::RangeOverflow)
        })?;
        raw.input_q8_1 = address_for(Q8_1_MATVEC_OPERATION, input_view)?;
        raw.input_len_bytes = u64::try_from(input_view.len_bytes()).map_err(|_| {
            q8_1_validation(Q8_1_MATVEC_OPERATION, CudaRuntimeErrorKind::RangeOverflow)
        })?;
        raw.output = address_for(Q8_1_MATVEC_OPERATION, output_view)?;
        raw.output_len_bytes = u64::try_from(output_view.len_bytes()).map_err(|_| {
            q8_1_validation(Q8_1_MATVEC_OPERATION, CudaRuntimeErrorKind::RangeOverflow)
        })?;
        raw.rows = weight.rows;
        raw.cols = weight.cols;
        Ok(Self {
            raw,
            device_ordinal: ordinal,
        })
    }

    pub const fn as_raw(&self) -> &AugurCudaQ8_0Q8_1MatvecArgsV1 {
        &self.raw
    }

    pub fn launch(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        // SAFETY: all device ranges and dimensions were checked at bind time.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_q8_0_q8_1_matvec_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result(Q8_1_MATVEC_OPERATION, result)
    }

    pub fn submit_for_plan(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        // SAFETY: owners behind the bound views remain alive through completion.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_q8_0_q8_1_matvec_submit_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result("cuda_q8_0_q8_1_matvec_submit_v1", result)
    }

    fn validate_stream(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.device_ordinal {
            return Err(q8_1_validation(
                Q8_1_MATVEC_OPERATION,
                CudaRuntimeErrorKind::DeviceMismatch,
            ));
        }
        Ok(())
    }
}

fn address(view: CudaBufferView) -> Result<u64, CudaRuntimeError> {
    view.device_ptr()
        .raw_value()
        .checked_add(
            u64::try_from(view.offset_bytes())
                .map_err(|_| validation(CudaRuntimeErrorKind::RangeOverflow))?,
        )
        .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))
}

fn address_for(operation: &'static str, view: CudaBufferView) -> Result<u64, CudaRuntimeError> {
    view.device_ptr()
        .raw_value()
        .checked_add(
            u64::try_from(view.offset_bytes())
                .map_err(|_| q8_1_validation(operation, CudaRuntimeErrorKind::RangeOverflow))?,
        )
        .ok_or_else(|| q8_1_validation(operation, CudaRuntimeErrorKind::RangeOverflow))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_runtime::test_buffer_view;

    #[test]
    fn exact_q8_layout_binds() {
        let view = test_buffer_view(0x1000, 0, 0, 2 * 2 * BLOCK_BYTES);
        let weight = CudaQ8_0WeightViewV1::new(
            view,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_0_CUDA_V1,
            2,
            64,
        )
        .unwrap();
        assert_eq!(weight.rows(), 2);
        assert_eq!(weight.cols(), 64);
        assert_eq!(weight.blocks_per_row(), 2);
    }

    #[test]
    fn cpu_layout_is_rejected() {
        let error = CudaQ8_0WeightViewV1::new(
            test_buffer_view(0x1000, 0, 0, 68),
            PhysicalLayoutId::Q8_0_COMPACT_V1,
            1,
            32,
        )
        .unwrap_err();
        assert_eq!(error.kind, CudaRuntimeErrorKind::InvalidLayout);
    }

    #[test]
    fn columns_must_be_q8_block_multiple() {
        let error = CudaQ8_0WeightViewV1::new(
            test_buffer_view(0x1000, 0, 0, 68),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_0_CUDA_V1,
            1,
            31,
        )
        .unwrap_err();
        assert_eq!(error.kind, CudaRuntimeErrorKind::InvalidDimensions);
    }

    #[test]
    fn weight_payload_length_must_be_exact() {
        let error = CudaQ8_0WeightViewV1::new(
            test_buffer_view(0x1000, 0, 0, 67),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_0_CUDA_V1,
            1,
            32,
        )
        .unwrap_err();
        assert_eq!(error.kind, CudaRuntimeErrorKind::OutOfBounds);
    }

    #[test]
    fn weight_address_must_be_two_byte_aligned() {
        let error = CudaQ8_0WeightViewV1::new(
            test_buffer_view(0x1001, 0, 0, 34),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_0_CUDA_V1,
            1,
            32,
        )
        .unwrap_err();
        assert_eq!(error.kind, CudaRuntimeErrorKind::MisalignedDeviceAddress);
    }

    #[test]
    fn exact_q8_1_activation_layout_binds() {
        let view = CudaQ8_1ActivationViewV1::new(
            test_buffer_view(0x2000, 0, 0, 2 * Q8_1_ACTIVATION_BLOCK_BYTES),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_1_ACTIVATION_CUDA_V1,
            64,
        )
        .unwrap();
        assert_eq!(view.elements(), 64);
        assert_eq!(view.buffer_view().len_bytes(), 72);
    }

    #[test]
    fn q8_1_activation_requires_exact_extent_and_layout() {
        let wrong_extent = CudaQ8_1ActivationViewV1::new(
            test_buffer_view(0x2000, 0, 0, 71),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_1_ACTIVATION_CUDA_V1,
            64,
        )
        .unwrap_err();
        assert_eq!(wrong_extent.kind, CudaRuntimeErrorKind::OutOfBounds);

        let wrong_layout = CudaQ8_1ActivationViewV1::new(
            test_buffer_view(0x2000, 0, 0, 72),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_0_CUDA_V1,
            64,
        )
        .unwrap_err();
        assert_eq!(wrong_layout.kind, CudaRuntimeErrorKind::InvalidLayout);
    }

    #[test]
    fn q8_1_activation_requires_four_byte_alignment() {
        let error = CudaQ8_1ActivationViewV1::new(
            test_buffer_view(0x2002, 0, 0, Q8_1_ACTIVATION_BLOCK_BYTES),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_1_ACTIVATION_CUDA_V1,
            32,
        )
        .unwrap_err();
        assert_eq!(error.kind, CudaRuntimeErrorKind::MisalignedDeviceAddress);
    }
}
