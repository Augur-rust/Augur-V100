#![cfg(feature = "qwen35-linear-step")]

//! Typed SM70 binding for one Qwen3.8 linear-attention recurrent step.
//!
//! The binding is deliberately fixed-width.  It is a component primitive for
//! the audited layer-0 geometry, not a general-purpose recurrent executor.
//! All views are device-resident FP32 and the state view is updated in place.

use crate::{CudaBufferView, CudaF32View, CudaRuntimeError, CudaRuntimeErrorKind, OwnedCudaStream};
use augur_cuda_sys::{
    AugurCudaQwen35LinearStepArgsV1, AUGUR_CUDA_QWEN35_LINEAR_STEP_ARGS_VERSION_V1,
    AUGUR_CUDA_QWEN35_LINEAR_STEP_CONV_HISTORY_ELEMENTS_V1,
    AUGUR_CUDA_QWEN35_LINEAR_STEP_CONV_WEIGHT_ELEMENTS_V1, AUGUR_CUDA_QWEN35_LINEAR_STEP_HEAD_DIM_V1,
    AUGUR_CUDA_QWEN35_LINEAR_STEP_OUTPUT_ELEMENTS_V1, AUGUR_CUDA_QWEN35_LINEAR_STEP_QKV_ELEMENTS_V1,
    AUGUR_CUDA_QWEN35_LINEAR_STEP_STATE_ELEMENTS_V1, AUGUR_CUDA_QWEN35_LINEAR_STEP_VALUE_HEADS_V1,
};

const OPERATION: &str = "cuda_qwen35_linear_step_v1";

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

fn exact_f32(view: CudaF32View, elements: usize) -> Result<(), CudaRuntimeError> {
    if view.element_count() != elements
        || view.buffer_view().len_bytes() != elements.saturating_mul(core::mem::size_of::<f32>())
    {
        return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
    }
    Ok(())
}

/// Bound arguments for one asynchronous Qwen35 recurrent step.
#[derive(Clone, Copy, Debug)]
pub struct CudaQwen35LinearStepArgsV1 {
    raw: AugurCudaQwen35LinearStepArgsV1,
    ordinal: i32,
}

impl CudaQwen35LinearStepArgsV1 {
    /// Explicit raw-checkpoint interpretation: grouped Q/K heads and
    /// sum-of-squares + 1e-6 L2 normalization. Never select for GGUF inputs.
    pub fn for_checkpoint_source(self) -> CudaQwen35CheckpointLinearStepArgsV1 {
        CudaQwen35CheckpointLinearStepArgsV1(self)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn bind(
        stream: &OwnedCudaStream,
        qkv: CudaF32View,
        z: CudaF32View,
        conv_weights: CudaF32View,
        conv_history: CudaF32View,
        decay_gate: CudaF32View,
        beta: CudaF32View,
        state: CudaF32View,
        norm_weight: CudaF32View,
        activated: CudaF32View,
        recurrent: CudaF32View,
        output: CudaF32View,
        epsilon: f32,
    ) -> Result<Self, CudaRuntimeError> {
        let views = [
            qkv.buffer_view(),
            z.buffer_view(),
            conv_weights.buffer_view(),
            conv_history.buffer_view(),
            decay_gate.buffer_view(),
            beta.buffer_view(),
            state.buffer_view(),
            norm_weight.buffer_view(),
            activated.buffer_view(),
            recurrent.buffer_view(),
            output.buffer_view(),
        ];
        let ordinal = stream.device_ordinal();
        if views.iter().any(|view| view.device_ordinal() != ordinal) {
            return Err(validation(CudaRuntimeErrorKind::DeviceMismatch));
        }
        exact_f32(qkv, AUGUR_CUDA_QWEN35_LINEAR_STEP_QKV_ELEMENTS_V1)?;
        exact_f32(z, AUGUR_CUDA_QWEN35_LINEAR_STEP_OUTPUT_ELEMENTS_V1)?;
        exact_f32(
            conv_weights,
            AUGUR_CUDA_QWEN35_LINEAR_STEP_CONV_WEIGHT_ELEMENTS_V1,
        )?;
        exact_f32(
            conv_history,
            AUGUR_CUDA_QWEN35_LINEAR_STEP_CONV_HISTORY_ELEMENTS_V1,
        )?;
        exact_f32(decay_gate, AUGUR_CUDA_QWEN35_LINEAR_STEP_VALUE_HEADS_V1)?;
        exact_f32(beta, AUGUR_CUDA_QWEN35_LINEAR_STEP_VALUE_HEADS_V1)?;
        exact_f32(state, AUGUR_CUDA_QWEN35_LINEAR_STEP_STATE_ELEMENTS_V1)?;
        exact_f32(norm_weight, AUGUR_CUDA_QWEN35_LINEAR_STEP_HEAD_DIM_V1)?;
        exact_f32(activated, AUGUR_CUDA_QWEN35_LINEAR_STEP_QKV_ELEMENTS_V1)?;
        exact_f32(recurrent, AUGUR_CUDA_QWEN35_LINEAR_STEP_OUTPUT_ELEMENTS_V1)?;
        exact_f32(output, AUGUR_CUDA_QWEN35_LINEAR_STEP_OUTPUT_ELEMENTS_V1)?;
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }

        // State is intentionally in-place, but it must not alias any other
        // view.  Every other pair is required to be disjoint as well.
        for (index, view) in views.iter().enumerate() {
            for other in views.iter().skip(index + 1) {
                if view.overlaps(*other, OPERATION)? {
                    return Err(validation(CudaRuntimeErrorKind::OverlappingDeviceRanges));
                }
            }
        }

        let mut raw = AugurCudaQwen35LinearStepArgsV1::new();
        let addresses = views
            .map(address)
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        let lengths = views.map(|view| view.len_bytes());
        raw.abi_version = AUGUR_CUDA_QWEN35_LINEAR_STEP_ARGS_VERSION_V1;
        raw.struct_size = core::mem::size_of::<AugurCudaQwen35LinearStepArgsV1>() as u32;
        raw.qkv_f32 = addresses[0];
        raw.qkv_len_bytes = lengths[0] as u64;
        raw.z_f32 = addresses[1];
        raw.z_len_bytes = lengths[1] as u64;
        raw.conv_weights_f32 = addresses[2];
        raw.conv_weights_len_bytes = lengths[2] as u64;
        raw.conv_history_f32 = addresses[3];
        raw.conv_history_len_bytes = lengths[3] as u64;
        raw.decay_gate_f32 = addresses[4];
        raw.decay_gate_len_bytes = lengths[4] as u64;
        raw.beta_f32 = addresses[5];
        raw.beta_len_bytes = lengths[5] as u64;
        raw.state_f32 = addresses[6];
        raw.state_len_bytes = lengths[6] as u64;
        raw.norm_weight_f32 = addresses[7];
        raw.norm_weight_len_bytes = lengths[7] as u64;
        raw.activated_f32 = addresses[8];
        raw.activated_len_bytes = lengths[8] as u64;
        raw.recurrent_f32 = addresses[9];
        raw.recurrent_len_bytes = lengths[9] as u64;
        raw.output_f32 = addresses[10];
        raw.output_len_bytes = lengths[10] as u64;
        raw.epsilon_bits = epsilon.to_bits();
        raw.reserved = 0;
        Ok(Self { raw, ordinal })
    }

    pub const fn as_raw(&self) -> &AugurCudaQwen35LinearStepArgsV1 {
        &self.raw
    }

    pub fn submit(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        // SAFETY: all pointers and exact extents were checked at bind time;
        // callers retain the owning buffers until the covering completion
        // proof.  The native shim repeats the ABI and range checks.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_linear_step_submit_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result("cuda_qwen35_linear_step_submit_v1", result)
    }

    pub fn launch(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        // SAFETY: same binding contract as `submit`; the native sync wrapper
        // drains the stream before returning.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_linear_step_v1(
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

#[derive(Clone, Copy, Debug)]
pub struct CudaQwen35CheckpointLinearStepArgsV1(CudaQwen35LinearStepArgsV1);
impl CudaQwen35CheckpointLinearStepArgsV1 {
    pub fn submit_for_plan(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.0.validate_stream(stream)?;
        // SAFETY: original binding checked exact extents, disjoint ranges and
        // device; callers retain all owners through stream completion.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_checkpoint_linear_step_submit_v1(
                stream.device_ordinal(), stream.raw()?, &self.0.raw)
        };
        CudaRuntimeError::from_result("cuda_qwen35_checkpoint_linear_step_submit_v1", result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_runtime::test_buffer_view;
    use augur_core::PhysicalLayoutId;
    use augur_cuda_sys::AUGUR_CUDA_QWEN35_LINEAR_STEP_KEY_HEADS_V1;

    fn view(address: u64, bytes: usize) -> CudaF32View {
        CudaF32View::new(
            test_buffer_view(address, 0, 0, bytes),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            bytes / core::mem::size_of::<f32>(),
        )
        .unwrap()
    }

    #[test]
    fn geometry_constants_are_fixed() {
        assert_eq!(AUGUR_CUDA_QWEN35_LINEAR_STEP_KEY_HEADS_V1, 16);
        assert_eq!(AUGUR_CUDA_QWEN35_LINEAR_STEP_VALUE_HEADS_V1, 48);
        assert_eq!(AUGUR_CUDA_QWEN35_LINEAR_STEP_HEAD_DIM_V1, 128);
        assert_eq!(AUGUR_CUDA_QWEN35_LINEAR_STEP_QKV_ELEMENTS_V1, 10_240);
        assert_eq!(AUGUR_CUDA_QWEN35_LINEAR_STEP_STATE_ELEMENTS_V1, 786_432);
        assert_eq!(AUGUR_CUDA_QWEN35_LINEAR_STEP_CONV_HISTORY_ELEMENTS_V1, 30_720);
        assert_eq!(AUGUR_CUDA_QWEN35_LINEAR_STEP_CONV_WEIGHT_ELEMENTS_V1, 40_960);
    }

    #[test]
    fn exact_f32_rejects_extra_extent() {
        let bytes = (AUGUR_CUDA_QWEN35_LINEAR_STEP_QKV_ELEMENTS_V1 + 1) * 4;
        assert!(exact_f32(
            view(0x1000, bytes),
            AUGUR_CUDA_QWEN35_LINEAR_STEP_QKV_ELEMENTS_V1
        )
        .is_err());
    }
}
