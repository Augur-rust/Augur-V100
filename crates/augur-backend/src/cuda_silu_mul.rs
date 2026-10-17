#![cfg(feature = "cuda")]

//! Checked binding for the unclamped FP32 `SiLU(gate) * up` primitive.
//!
//! This is intentionally separate from [`CudaSwigluF32ArgsV1`].  The older
//! binding carries a positive clamp limit and therefore represents a different
//! numerical contract.  This operation only accepts equal, non-empty FP32
//! views on one device and leaves all activation values unclamped.

use augur_cuda_sys::AugurCudaSiluMulF32ArgsV1;

use crate::{CudaBufferView, CudaF32View, CudaRuntimeError, CudaRuntimeErrorKind, OwnedCudaStream};

const OPERATION: &str = "cuda_silu_mul_f32_v1";

#[derive(Clone, Copy, Debug)]
pub struct CudaSiluMulF32ArgsV1 {
    raw: AugurCudaSiluMulF32ArgsV1,
    device_ordinal: i32,
}

impl CudaSiluMulF32ArgsV1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        gate: CudaF32View,
        up: CudaF32View,
        output: CudaF32View,
    ) -> Result<Self, CudaRuntimeError> {
        Self::bind_layout(stream, gate, up, output, false)
    }

    /// Explicit exact-alias gate overwrite for the elementwise runtime kernel.
    /// Each thread reads gate[i]/up[i] before writing gate[i]; up must be disjoint.
    /// Ordinary bind still rejects overlap, and partial alias is never allowed.
    pub fn bind_in_place_gate(
        stream: &OwnedCudaStream,
        gate: CudaF32View,
        up: CudaF32View,
    ) -> Result<Self, CudaRuntimeError> {
        Self::bind_layout(stream, gate, up, gate, true)
    }

    fn bind_layout(
        stream: &OwnedCudaStream,
        gate: CudaF32View,
        up: CudaF32View,
        output: CudaF32View,
        in_place_gate: bool,
    ) -> Result<Self, CudaRuntimeError> {
        let ordinal = stream.device_ordinal();
        let views = [gate.buffer_view(), up.buffer_view(), output.buffer_view()];
        if views.iter().any(|view| view.device_ordinal() != ordinal) {
            return Err(validation(CudaRuntimeErrorKind::DeviceMismatch));
        }
        let elements = gate.element_count();
        if elements == 0 || up.element_count() != elements || output.element_count() != elements {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        if !valid_output_layout(
            gate.buffer_view(),
            up.buffer_view(),
            output.buffer_view(),
            in_place_gate,
        )? {
            return Err(validation(CudaRuntimeErrorKind::OverlappingDeviceRanges));
        }
        let mut raw = AugurCudaSiluMulF32ArgsV1::new();
        raw.gate = address(gate.buffer_view())?;
        raw.gate_len_bytes = to_u64(gate.buffer_view().len_bytes())?;
        raw.up = address(up.buffer_view())?;
        raw.up_len_bytes = to_u64(up.buffer_view().len_bytes())?;
        raw.output = address(output.buffer_view())?;
        raw.output_len_bytes = to_u64(output.buffer_view().len_bytes())?;
        raw.elements = u32::try_from(elements)
            .map_err(|_| validation(CudaRuntimeErrorKind::InvalidDimensions))?;
        Ok(Self {
            raw,
            device_ordinal: ordinal,
        })
    }

    pub const fn as_raw(&self) -> &AugurCudaSiluMulF32ArgsV1 {
        &self.raw
    }

    pub fn launch(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        // SAFETY: `raw` was built from checked views. The caller keeps all
        // owners alive until this synchronous operation returns.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_silu_mul_f32_v1(stream.device_ordinal(), stream.raw()?, &self.raw)
        };
        CudaRuntimeError::from_result(OPERATION, result)
    }

    /// Enqueue the unclamped activation without a host synchronization.
    pub fn submit_for_plan(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        // SAFETY: same binding contract as `launch`; owners must outlive the
        // stream completion proof for this asynchronous submission.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_silu_mul_f32_submit_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result("cuda_silu_mul_f32_submit_v1", result)
    }

    pub(crate) fn validate_stream(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if self.device_ordinal != stream.device_ordinal() {
            return Err(validation(CudaRuntimeErrorKind::DeviceMismatch));
        }
        Ok(())
    }

    /// Explicit B512 prefix-only numerical alternative; never used by default.
    #[cfg(feature = "qwen35-prefill-cublas")]
    pub fn preflight_llama_prefix(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        self.require_disjoint_llama()?;
        // SAFETY: immutable equal-sized live F32 views from bind, retained by
        // the prefill workspace. Native validates the bounded shape and ranges.
        CudaRuntimeError::from_result(OPERATION, unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_llama_swiglu_preflight_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        })
    }

    /// Separate exact-input activation extent, matching the 1..=512 input-row
    /// family. Legacy prefix and default decode entry points remain unchanged.
    #[cfg(feature = "qwen35-prefill-cublas")]
    pub fn preflight_llama_input_rows(
        &self,
        stream: &OwnedCudaStream,
    ) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        self.require_disjoint_llama()?;
        // SAFETY: the immutable binding retains exact, disjoint F32 extents;
        // native preflight additionally checks the input-row family and device.
        CudaRuntimeError::from_result(OPERATION, unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_llama_swiglu_rows_preflight_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        })
    }

    #[cfg(feature = "qwen35-prefill-cublas")]
    pub fn submit_llama_input_rows(
        &self,
        stream: &OwnedCudaStream,
    ) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        self.require_disjoint_llama()?;
        // SAFETY: same preflighted binding/owner; all allocations must remain
        // live through stream completion, exactly as for submit_llama_prefix.
        CudaRuntimeError::from_result(OPERATION, unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_llama_swiglu_rows_prevalidated_submit_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        })
    }

    #[cfg(feature = "qwen35-prefill-cublas")]
    pub fn submit_llama_prefix(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        self.require_disjoint_llama()?;
        // SAFETY: called by the same prefix owner after preflight; workspace
        // owners outlive stream completion. This does not alter the default op.
        CudaRuntimeError::from_result(OPERATION, unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_llama_swiglu_prevalidated_submit_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        })
    }
    #[cfg(feature = "qwen35-prefill-cublas")]
    fn require_disjoint_llama(&self) -> Result<(), CudaRuntimeError> {
        if self.raw.output == self.raw.gate {
            return Err(validation(CudaRuntimeErrorKind::OverlappingDeviceRanges));
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

fn valid_output_layout(
    gate: CudaBufferView,
    up: CudaBufferView,
    output: CudaBufferView,
    in_place_gate: bool,
) -> Result<bool, CudaRuntimeError> {
    let gate_ok = if in_place_gate {
        address(gate)? == address(output)? && gate.len_bytes() == output.len_bytes()
    } else {
        !overlaps(gate, output)?
    };
    Ok(gate_ok && !overlaps(up, output)?)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_runtime::test_buffer_view;

    #[test]
    fn rejects_output_overlap() {
        let gate = test_buffer_view(0x1000, 0, 0, 16);
        let up = test_buffer_view(0x2000, 0, 0, 16);
        let output = test_buffer_view(0x1000, 0, 8, 16);
        // Binding needs a live stream, so exercise the pure overlap helper.
        assert!(overlaps(gate, output).unwrap());
        assert!(!overlaps(gate, up).unwrap());
    }
}

#[cfg(test)]
mod in_place_tests {
    use super::*;
    use crate::cuda_runtime::test_buffer_view;
    #[test]
    fn exact_gate_alias_only_and_default_contract_unchanged() {
        let gate = test_buffer_view(0x1000, 0, 0, 16);
        let up = test_buffer_view(0x1000, 0, 32, 16);
        let partial = test_buffer_view(0x1000, 0, 8, 16);
        assert!(valid_output_layout(gate, up, gate, true).unwrap());
        assert!(!valid_output_layout(gate, up, gate, false).unwrap());
        assert!(!valid_output_layout(gate, up, partial, true).unwrap());
        assert!(!valid_output_layout(gate, gate, gate, true).unwrap());
        assert!(!valid_output_layout(gate, up, up, true).unwrap());
    }
}

#[cfg(all(test, feature = "qwen35-prefill-cublas"))]
mod in_place_dispatch_tests {
    use super::*;
    #[test]
    fn inplace_runtime_binding_cannot_enter_disjoint_llama_alternative() {
        let mut raw = AugurCudaSiluMulF32ArgsV1::new();
        raw.gate = 0x1000;
        raw.output = 0x1000;
        let mut args = CudaSiluMulF32ArgsV1 {
            raw,
            device_ordinal: 0,
        };
        assert!(args.require_disjoint_llama().is_err());
        args.raw.output = 0x2000;
        assert!(args.require_disjoint_llama().is_ok());
    }
}
