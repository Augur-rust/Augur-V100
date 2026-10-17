#![cfg(feature = "qwen35-linear-step")]

//! Checked binding for the Qwen35 linear-attention projection postprocess.
//!
//! The operation applies the stable softplus/sigmoid gate transforms on 48
//! device-resident FP32 values.  It is intentionally fixed-width and is only
//! a component primitive; callers must retain all owning buffers until a
//! completion proof covers the submission.

use crate::{CudaBufferView, CudaF32View, CudaRuntimeError, CudaRuntimeErrorKind, OwnedCudaStream};
use augur_cuda_sys::{
    AugurCudaQwen35LinearProjectionPostprocessArgsV1,
    AUGUR_CUDA_QWEN35_LINEAR_PROJECTION_POSTPROCESS_ARGS_VERSION_V1,
    AUGUR_CUDA_QWEN35_LINEAR_PROJECTION_POSTPROCESS_ELEMENTS_V1,
};

const OPERATION: &str = "cuda_qwen35_linear_projection_postprocess_v1";
const ELEMENTS: usize = AUGUR_CUDA_QWEN35_LINEAR_PROJECTION_POSTPROCESS_ELEMENTS_V1;

fn validation(kind: CudaRuntimeErrorKind) -> CudaRuntimeError {
    CudaRuntimeError::validation(OPERATION, kind)
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

fn exact_f32(view: CudaF32View) -> Result<(), CudaRuntimeError> {
    let expected = ELEMENTS
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))?;
    if view.element_count() != ELEMENTS || view.buffer_view().len_bytes() != expected {
        return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
    }
    Ok(())
}

/// Bound arguments for an asynchronous projection gate postprocess.
#[derive(Clone, Copy, Debug)]
pub struct CudaQwen35LinearProjectionPostprocessArgsV1 {
    raw: AugurCudaQwen35LinearProjectionPostprocessArgsV1,
    ordinal: i32,
}

impl CudaQwen35LinearProjectionPostprocessArgsV1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        alpha_raw: CudaF32View,
        beta_raw: CudaF32View,
        dt_bias: CudaF32View,
        ssm_a: CudaF32View,
        decay_gate: CudaF32View,
        beta_out: CudaF32View,
    ) -> Result<Self, CudaRuntimeError> {
        let views = [
            alpha_raw.buffer_view(),
            beta_raw.buffer_view(),
            dt_bias.buffer_view(),
            ssm_a.buffer_view(),
            decay_gate.buffer_view(),
            beta_out.buffer_view(),
        ];
        let ordinal = stream.device_ordinal();
        if views.iter().any(|view| view.device_ordinal() != ordinal) {
            return Err(validation(CudaRuntimeErrorKind::DeviceMismatch));
        }
        for view in [alpha_raw, beta_raw, dt_bias, ssm_a, decay_gate, beta_out] {
            exact_f32(view)?;
        }
        for (index, view) in views.iter().enumerate() {
            for other in views.iter().skip(index + 1) {
                if view.overlaps(*other, OPERATION)? {
                    return Err(validation(CudaRuntimeErrorKind::OverlappingDeviceRanges));
                }
            }
        }

        let addresses = views
            .map(address)
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        let lengths = views.map(|view| view.len_bytes());
        let mut raw = AugurCudaQwen35LinearProjectionPostprocessArgsV1::new();
        raw.abi_version = AUGUR_CUDA_QWEN35_LINEAR_PROJECTION_POSTPROCESS_ARGS_VERSION_V1;
        raw.struct_size =
            core::mem::size_of::<AugurCudaQwen35LinearProjectionPostprocessArgsV1>() as u32;
        raw.alpha_raw_f32 = addresses[0];
        raw.alpha_raw_len_bytes = lengths[0] as u64;
        raw.beta_raw_f32 = addresses[1];
        raw.beta_raw_len_bytes = lengths[1] as u64;
        raw.dt_bias_f32 = addresses[2];
        raw.dt_bias_len_bytes = lengths[2] as u64;
        raw.ssm_a_f32 = addresses[3];
        raw.ssm_a_len_bytes = lengths[3] as u64;
        raw.decay_gate_f32 = addresses[4];
        raw.decay_gate_len_bytes = lengths[4] as u64;
        raw.beta_out_f32 = addresses[5];
        raw.beta_out_len_bytes = lengths[5] as u64;
        raw.elements = ELEMENTS as u32;
        raw.reserved = 0;
        Ok(Self { raw, ordinal })
    }

    pub const fn as_raw(&self) -> &AugurCudaQwen35LinearProjectionPostprocessArgsV1 {
        &self.raw
    }

    pub fn submit_for_plan(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        // SAFETY: all pointers and exact extents were checked at bind time;
        // owners must outlive a completion proof for this asynchronous work.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_linear_projection_postprocess_submit_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result(
            "cuda_qwen35_linear_projection_postprocess_submit_v1",
            result,
        )
    }

    pub fn launch(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        // SAFETY: same binding contract as submit_for_plan; the synchronous
        // shim drains the stream before returning.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_linear_projection_postprocess_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result(OPERATION, result)
    }

    fn validate_stream(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.ordinal {
            return Err(validation(CudaRuntimeErrorKind::DeviceMismatch));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_runtime::test_buffer_view;
    use augur_core::PhysicalLayoutId;

    fn view(address: u64, bytes: usize) -> CudaF32View {
        CudaF32View::new(
            test_buffer_view(address, 0, 0, bytes),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            bytes / core::mem::size_of::<f32>(),
        )
        .unwrap()
    }

    #[test]
    fn abi_geometry_is_fixed() {
        assert_eq!(ELEMENTS, 48);
        assert_eq!(
            core::mem::size_of::<AugurCudaQwen35LinearProjectionPostprocessArgsV1>(),
            112
        );
        assert_eq!(
            core::mem::align_of::<AugurCudaQwen35LinearProjectionPostprocessArgsV1>(),
            8
        );
    }

    #[test]
    fn exact_extent_rejects_extra_values() {
        assert!(exact_f32(view(0x1000, (ELEMENTS + 1) * 4)).is_err());
    }
}
