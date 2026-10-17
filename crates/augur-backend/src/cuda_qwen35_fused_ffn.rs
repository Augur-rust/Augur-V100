//! Checked Qwen35 gate/up projection fusion for the SM70 diagnostic lane.

use crate::{
    CudaBufferView, CudaF32View, CudaNvfp4Sm70WeightViewV1, CudaQ8_0WeightViewV1,
    CudaQ8_1ActivationViewV1, CudaRuntimeError, CudaRuntimeErrorKind, OwnedCudaStream,
};
use augur_cuda_sys::{AugurCudaNvfp4Sm70Q8_1FusedSwigluArgsV1, AugurCudaQ8_0Q8_1FusedSwigluArgsV1};

const Q8_OPERATION: &str = "cuda_q8_0_q8_1_fused_swiglu_v1";
const NVFP4_OPERATION: &str = "cuda_nvfp4_sm70_q8_1_fused_swiglu_v1";

fn validation(operation: &'static str, kind: CudaRuntimeErrorKind) -> CudaRuntimeError {
    CudaRuntimeError::validation(operation, kind)
}

fn address(operation: &'static str, view: CudaBufferView) -> Result<u64, CudaRuntimeError> {
    view.device_ptr()
        .raw_value()
        .checked_add(
            u64::try_from(view.offset_bytes())
                .map_err(|_| validation(operation, CudaRuntimeErrorKind::RangeOverflow))?,
        )
        .ok_or_else(|| validation(operation, CudaRuntimeErrorKind::RangeOverflow))
}

fn extent(operation: &'static str, bytes: usize) -> Result<u64, CudaRuntimeError> {
    u64::try_from(bytes).map_err(|_| validation(operation, CudaRuntimeErrorKind::RangeOverflow))
}

fn validate_views(
    operation: &'static str,
    ordinal: i32,
    views: [CudaBufferView; 4],
) -> Result<(), CudaRuntimeError> {
    if views.iter().any(|view| view.device_ordinal() != ordinal) {
        return Err(validation(
            operation,
            CudaRuntimeErrorKind::InvalidDimensions,
        ));
    }
    for first in 0..views.len() {
        for second in first + 1..views.len() {
            if views[first].overlaps(views[second], operation)? {
                return Err(validation(
                    operation,
                    CudaRuntimeErrorKind::OverlappingDeviceRanges,
                ));
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub struct CudaQ8_0Q8_1FusedSwigluArgsV1 {
    raw: AugurCudaQ8_0Q8_1FusedSwigluArgsV1,
    device_ordinal: i32,
}

impl CudaQ8_0Q8_1FusedSwigluArgsV1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        gate_weight: CudaQ8_0WeightViewV1,
        up_weight: CudaQ8_0WeightViewV1,
        input: CudaQ8_1ActivationViewV1,
        output: CudaF32View,
    ) -> Result<Self, CudaRuntimeError> {
        if gate_weight.rows() != up_weight.rows()
            || gate_weight.cols() != up_weight.cols()
            || input.elements() != gate_weight.cols() as usize
            || output.element_count() != gate_weight.rows() as usize
        {
            return Err(validation(
                Q8_OPERATION,
                CudaRuntimeErrorKind::InvalidDimensions,
            ));
        }
        let gate_view = gate_weight.buffer_view();
        let up_view = up_weight.buffer_view();
        let input_view = input.buffer_view();
        let output_view = output.buffer_view();
        let ordinal = stream.device_ordinal();
        validate_views(
            Q8_OPERATION,
            ordinal,
            [gate_view, up_view, input_view, output_view],
        )?;

        let mut raw = AugurCudaQ8_0Q8_1FusedSwigluArgsV1::new();
        raw.gate_weight = address(Q8_OPERATION, gate_view)?;
        raw.gate_weight_len_bytes = extent(Q8_OPERATION, gate_view.len_bytes())?;
        raw.up_weight = address(Q8_OPERATION, up_view)?;
        raw.up_weight_len_bytes = extent(Q8_OPERATION, up_view.len_bytes())?;
        raw.input_q8_1 = address(Q8_OPERATION, input_view)?;
        raw.input_len_bytes = extent(Q8_OPERATION, input_view.len_bytes())?;
        raw.output = address(Q8_OPERATION, output_view)?;
        raw.output_len_bytes = extent(Q8_OPERATION, output_view.len_bytes())?;
        raw.rows = gate_weight.rows();
        raw.cols = gate_weight.cols();
        Ok(Self {
            raw,
            device_ordinal: ordinal,
        })
    }

    pub const fn as_raw(&self) -> &AugurCudaQ8_0Q8_1FusedSwigluArgsV1 {
        &self.raw
    }

    pub fn submit_for_plan(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        // SAFETY: bind checked exact geometry, ownership, and disjoint ranges.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_q8_0_q8_1_fused_swiglu_submit_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result("cuda_q8_0_q8_1_fused_swiglu_submit_v1", result)
    }

    fn validate_stream(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.device_ordinal {
            return Err(validation(
                Q8_OPERATION,
                CudaRuntimeErrorKind::DeviceMismatch,
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CudaNvfp4Sm70Q8_1FusedSwigluArgsV1 {
    raw: AugurCudaNvfp4Sm70Q8_1FusedSwigluArgsV1,
    device_ordinal: i32,
}

impl CudaNvfp4Sm70Q8_1FusedSwigluArgsV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn bind(
        stream: &OwnedCudaStream,
        gate_weight: CudaNvfp4Sm70WeightViewV1,
        up_weight: CudaNvfp4Sm70WeightViewV1,
        input: CudaQ8_1ActivationViewV1,
        output: CudaF32View,
        gate_tensor_scale: f32,
        up_tensor_scale: f32,
    ) -> Result<Self, CudaRuntimeError> {
        if gate_weight.rows() != up_weight.rows()
            || gate_weight.cols() != up_weight.cols()
            || input.elements() != gate_weight.cols() as usize
            || output.element_count() != gate_weight.rows() as usize
        {
            return Err(validation(
                NVFP4_OPERATION,
                CudaRuntimeErrorKind::InvalidDimensions,
            ));
        }
        if !gate_tensor_scale.is_finite()
            || gate_tensor_scale <= 0.0
            || !up_tensor_scale.is_finite()
            || up_tensor_scale <= 0.0
        {
            return Err(validation(
                NVFP4_OPERATION,
                CudaRuntimeErrorKind::InvalidMetadata,
            ));
        }
        let gate_view = gate_weight.buffer_view();
        let up_view = up_weight.buffer_view();
        let input_view = input.buffer_view();
        let output_view = output.buffer_view();
        let ordinal = stream.device_ordinal();
        validate_views(
            NVFP4_OPERATION,
            ordinal,
            [gate_view, up_view, input_view, output_view],
        )?;
        if address(NVFP4_OPERATION, gate_view)? % core::mem::align_of::<u32>() as u64 != 0
            || address(NVFP4_OPERATION, up_view)? % core::mem::align_of::<u32>() as u64 != 0
        {
            return Err(validation(
                NVFP4_OPERATION,
                CudaRuntimeErrorKind::MisalignedDeviceAddress,
            ));
        }

        let mut raw = AugurCudaNvfp4Sm70Q8_1FusedSwigluArgsV1::new();
        raw.gate_weight = address(NVFP4_OPERATION, gate_view)?;
        raw.gate_weight_len_bytes = extent(NVFP4_OPERATION, gate_view.len_bytes())?;
        raw.up_weight = address(NVFP4_OPERATION, up_view)?;
        raw.up_weight_len_bytes = extent(NVFP4_OPERATION, up_view.len_bytes())?;
        raw.input_q8_1 = address(NVFP4_OPERATION, input_view)?;
        raw.input_len_bytes = extent(NVFP4_OPERATION, input_view.len_bytes())?;
        raw.output = address(NVFP4_OPERATION, output_view)?;
        raw.output_len_bytes = extent(NVFP4_OPERATION, output_view.len_bytes())?;
        raw.rows = gate_weight.rows();
        raw.cols = gate_weight.cols();
        raw.gate_tensor_scale = gate_tensor_scale;
        raw.up_tensor_scale = up_tensor_scale;
        Ok(Self {
            raw,
            device_ordinal: ordinal,
        })
    }

    pub const fn as_raw(&self) -> &AugurCudaNvfp4Sm70Q8_1FusedSwigluArgsV1 {
        &self.raw
    }

    pub fn submit_for_plan(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        // SAFETY: bind checked exact geometry, ownership, and disjoint ranges.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_nvfp4_sm70_q8_1_fused_swiglu_submit_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result("cuda_nvfp4_sm70_q8_1_fused_swiglu_submit_v1", result)
    }

    fn validate_stream(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.device_ordinal {
            return Err(validation(
                NVFP4_OPERATION,
                CudaRuntimeErrorKind::DeviceMismatch,
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_runtime::test_buffer_view;

    #[test]
    fn fused_abi_layout_is_fixed() {
        assert_eq!(core::mem::size_of::<AugurCudaQ8_0Q8_1FusedSwigluArgsV1>(), 80);
        assert_eq!(
            core::mem::size_of::<AugurCudaNvfp4Sm70Q8_1FusedSwigluArgsV1>(),
            88
        );
    }

    #[test]
    fn fused_views_require_one_device_and_pairwise_disjoint_ranges() {
        let views = [
            test_buffer_view(0x1000, 0, 0, 0x20),
            test_buffer_view(0x1020, 0, 0, 0x20),
            test_buffer_view(0x1040, 0, 0, 0x20),
            test_buffer_view(0x1060, 0, 0, 0x20),
        ];
        validate_views(Q8_OPERATION, 0, views).unwrap();

        let mut overlapping = views;
        overlapping[3] = test_buffer_view(0x101f, 0, 0, 0x20);
        let error = validate_views(Q8_OPERATION, 0, overlapping).unwrap_err();
        assert_eq!(error.kind, CudaRuntimeErrorKind::OverlappingDeviceRanges);

        let mut foreign = views;
        foreign[2] = test_buffer_view(0x1040, 1, 0, 0x20);
        let error = validate_views(Q8_OPERATION, 0, foreign).unwrap_err();
        assert_eq!(error.kind, CudaRuntimeErrorKind::InvalidDimensions);
    }

    #[test]
    fn fused_address_rejects_pointer_end_overflow() {
        let view = test_buffer_view(u64::MAX - 3, 0, 4, 1);
        let error = address(NVFP4_OPERATION, view).unwrap_err();
        assert_eq!(error.kind, CudaRuntimeErrorKind::RangeOverflow);
    }
}
