#![cfg(feature = "cuda")]

use crate::{
    CudaBufferView, CudaF32View, CudaRuntimeError, CudaRuntimeErrorKind, OwnedCudaBuffer,
    OwnedCudaStream,
};
use augur_cuda_sys::{
    AugurCudaDflashRopeF32ArgsV1, AugurCudaRmsnormF32ArgsV1, AugurCudaRopeF32ArgsV1,
    AugurCudaSplitRmsnormF32ArgsV1,
};

#[derive(Clone, Copy, Debug)]
pub struct CudaRmsnormF32ArgsV1 {
    raw: AugurCudaRmsnormF32ArgsV1,
    ordinal: i32,
    exact_warp_tree5120: bool,
}
impl CudaRmsnormF32ArgsV1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        input: CudaF32View,
        weight: CudaF32View,
        output: CudaF32View,
        epsilon: f32,
    ) -> Result<Self, CudaRuntimeError> {
        if input.element_count() == 0
            || input.element_count() != weight.element_count()
            || output.element_count() != input.element_count()
            || !epsilon.is_finite()
            || epsilon <= 0.0
            || [
                input.buffer_view(),
                weight.buffer_view(),
                output.buffer_view(),
            ]
            .iter()
            .any(|v| v.device_ordinal() != stream.device_ordinal())
        {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        if output
            .buffer_view()
            .overlaps(input.buffer_view(), "cuda_rmsnorm_f32_v1")?
            || output
                .buffer_view()
                .overlaps(weight.buffer_view(), "cuda_rmsnorm_f32_v1")?
        {
            return Err(error(CudaRuntimeErrorKind::OverlappingDeviceRanges));
        }
        let mut raw = AugurCudaRmsnormF32ArgsV1::new();
        raw.input = addr(input.buffer_view())?;
        raw.input_len_bytes = input.buffer_view().len_bytes() as u64;
        raw.weight = addr(weight.buffer_view())?;
        raw.weight_len_bytes = weight.buffer_view().len_bytes() as u64;
        raw.output = addr(output.buffer_view())?;
        raw.output_len_bytes = output.buffer_view().len_bytes() as u64;
        raw.elements = input
            .element_count()
            .try_into()
            .map_err(|_| error(CudaRuntimeErrorKind::InvalidDimensions))?;
        raw.epsilon_bits = epsilon.to_bits();
        Ok(Self {
            raw,
            ordinal: stream.device_ordinal(),
            exact_warp_tree5120: false,
        })
    }
    /// Opt-in fixed5120 schedule. No new operands, workspace, scale or reduction tree.
    pub fn with_exact_warp_tree5120(mut self) -> Result<Self, CudaRuntimeError> {
        if self.raw.elements != 5120 {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        self.exact_warp_tree5120 = true;
        Ok(self)
    }

    #[cfg(feature = "qwen35-prefill-cublas")]
    pub(crate) fn qpn_fused_descriptor(&self) -> Result<AugurCudaRmsnormF32ArgsV1, CudaRuntimeError> {
        if !self.exact_warp_tree5120 || self.raw.elements != 5120 {
            return Err(error(CudaRuntimeErrorKind::InvalidMetadata));
        }
        Ok(self.raw)
    }

    pub fn launch(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if self.exact_warp_tree5120 {
            self.submit_for_plan(stream)?;
            return stream.synchronize();
        }
        if stream.device_ordinal() != self.ordinal {
            return Err(error(CudaRuntimeErrorKind::DeviceMismatch));
        }
        let result = unsafe {
            augur_cuda_sys::augur_cuda_rmsnorm_f32_v1(stream.device_ordinal(), stream.raw()?, &self.raw)
        };
        CudaRuntimeError::from_result("cuda_rmsnorm_f32_v1", result)
    }

    /// Enqueue RMSNorm without synchronizing the host.
    ///
    /// The caller must retain every buffer referenced by the bound views until
    /// a completion proof covering this submission has been observed.
    pub fn submit_for_plan(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.ordinal {
            return Err(error(CudaRuntimeErrorKind::DeviceMismatch));
        }
        let result = unsafe {
            if self.exact_warp_tree5120 {
                augur_cuda_sys::augur_cuda_rmsnorm_f32_warp_tree5120_submit_v1(
                    stream.device_ordinal(), stream.raw()?, &self.raw)
            } else {
                augur_cuda_sys::augur_cuda_rmsnorm_f32_submit_v1(
                    stream.device_ordinal(), stream.raw()?, &self.raw)
            }
        };
        CudaRuntimeError::from_result("cuda_rmsnorm_f32_submit_v1", result)
    }
}

pub struct CudaSplitRmsnormF32V1 {
    raw: AugurCudaSplitRmsnormF32ArgsV1,
    ordinal: i32,
    _scale_workspace: OwnedCudaBuffer,
}

impl CudaSplitRmsnormF32V1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        input: CudaF32View,
        weight: CudaF32View,
        output: CudaF32View,
        epsilon: f32,
    ) -> Result<Self, CudaRuntimeError> {
        if input.element_count() == 0
            || input.element_count() != weight.element_count()
            || output.element_count() != input.element_count()
            || !epsilon.is_finite()
            || epsilon <= 0.0
            || [
                input.buffer_view(),
                weight.buffer_view(),
                output.buffer_view(),
            ]
            .iter()
            .any(|view| view.device_ordinal() != stream.device_ordinal())
        {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let operation = "cuda_split_rmsnorm_f32_v1";
        if output
            .buffer_view()
            .overlaps(input.buffer_view(), operation)?
            || output
                .buffer_view()
                .overlaps(weight.buffer_view(), operation)?
        {
            return Err(error(CudaRuntimeErrorKind::OverlappingDeviceRanges));
        }

        let scale_workspace =
            OwnedCudaBuffer::allocate(stream.device_ordinal(), core::mem::size_of::<f32>())?;
        let scale_view = scale_workspace.view(0, core::mem::size_of::<f32>())?;
        let mut raw = AugurCudaSplitRmsnormF32ArgsV1::new();
        raw.input = addr(input.buffer_view())?;
        raw.input_len_bytes = input.buffer_view().len_bytes() as u64;
        raw.weight = addr(weight.buffer_view())?;
        raw.weight_len_bytes = weight.buffer_view().len_bytes() as u64;
        raw.output = addr(output.buffer_view())?;
        raw.output_len_bytes = output.buffer_view().len_bytes() as u64;
        raw.scale_workspace = addr(scale_view)?;
        raw.scale_workspace_len_bytes = scale_view.len_bytes() as u64;
        raw.elements = input
            .element_count()
            .try_into()
            .map_err(|_| error(CudaRuntimeErrorKind::InvalidDimensions))?;
        raw.epsilon_bits = epsilon.to_bits();
        Ok(Self {
            raw,
            ordinal: stream.device_ordinal(),
            _scale_workspace: scale_workspace,
        })
    }

    pub fn launch(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        let result = unsafe {
            augur_cuda_sys::augur_cuda_split_rmsnorm_f32_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result("cuda_split_rmsnorm_f32_v1", result)
    }

    /// Enqueue the exact-order reduction and parallel output stages.
    pub fn submit_for_plan(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        let result = unsafe {
            augur_cuda_sys::augur_cuda_split_rmsnorm_f32_submit_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result("cuda_split_rmsnorm_f32_submit_v1", result)
    }

    fn validate_stream(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.ordinal {
            return Err(error(CudaRuntimeErrorKind::DeviceMismatch));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CudaRopeF32ArgsV1 {
    raw: AugurCudaRopeF32ArgsV1,
    ordinal: i32,
}
impl CudaRopeF32ArgsV1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        input: CudaF32View,
        output: CudaF32View,
        cos: CudaF32View,
        sin: CudaF32View,
        dimensions: usize,
    ) -> Result<Self, CudaRuntimeError> {
        if dimensions == 0
            || !dimensions.is_multiple_of(2)
            || dimensions > input.element_count()
            || output.element_count() != input.element_count()
            || cos.element_count() != dimensions / 2
            || sin.element_count() != dimensions / 2
            || [
                input.buffer_view(),
                output.buffer_view(),
                cos.buffer_view(),
                sin.buffer_view(),
            ]
            .iter()
            .any(|v| v.device_ordinal() != stream.device_ordinal())
        {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let mut raw = AugurCudaRopeF32ArgsV1::new();
        raw.input = addr(input.buffer_view())?;
        raw.input_len_bytes = input.buffer_view().len_bytes() as u64;
        raw.output = addr(output.buffer_view())?;
        raw.output_len_bytes = output.buffer_view().len_bytes() as u64;
        raw.cos = addr(cos.buffer_view())?;
        raw.cos_len_bytes = cos.buffer_view().len_bytes() as u64;
        raw.sin = addr(sin.buffer_view())?;
        raw.sin_len_bytes = sin.buffer_view().len_bytes() as u64;
        raw.elements = input
            .element_count()
            .try_into()
            .map_err(|_| error(CudaRuntimeErrorKind::InvalidDimensions))?;
        raw.dimensions = dimensions
            .try_into()
            .map_err(|_| error(CudaRuntimeErrorKind::InvalidDimensions))?;
        Ok(Self {
            raw,
            ordinal: stream.device_ordinal(),
        })
    }
    pub fn launch(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.ordinal {
            return Err(error(CudaRuntimeErrorKind::DeviceMismatch));
        }
        let result = unsafe {
            augur_cuda_sys::augur_cuda_rope_f32_v1(stream.device_ordinal(), stream.raw()?, &self.raw)
        };
        CudaRuntimeError::from_result("cuda_rope_f32_v1", result)
    }

    /// Enqueue RoPE without synchronizing the host.
    ///
    /// The caller must retain every buffer referenced by the bound views until
    /// a completion proof covering this submission has been observed.
    pub fn submit_for_plan(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.ordinal {
            return Err(error(CudaRuntimeErrorKind::DeviceMismatch));
        }
        // Keep the existing synchronous `launch` contract intact while using
        // the dedicated submit ABI for plans that keep all stages on one
        // stream.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_rope_f32_submit_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result("cuda_rope_f32_submit_v1", result)
    }
}

/// DFlash-specific NeoX RoPE. Unlike the generic adjacent-pair kernel, this
/// operator pairs the first and second halves of every head and computes the
/// angle from an absolute token position. The operator is intentionally kept
/// separate so a generic Qwen RoPE cannot silently be used for DFlash K/V.
/// Frequencies are evaluated in FP64 and rounded once to F32 before the F32
/// position product, matching the CPU reference across long absolute positions.
#[derive(Clone, Copy, Debug)]
pub struct CudaDflashRopeF32ArgsV1 {
    raw: AugurCudaDflashRopeF32ArgsV1,
    ordinal: i32,
}

impl CudaDflashRopeF32ArgsV1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        input: CudaF32View,
        output: CudaF32View,
        dimensions: usize,
        rotary_dimensions: usize,
        position: usize,
        theta: f32,
    ) -> Result<Self, CudaRuntimeError> {
        if dimensions == 0
            || !dimensions.is_multiple_of(2)
            || rotary_dimensions == 0
            || !rotary_dimensions.is_multiple_of(2)
            || rotary_dimensions > dimensions
            || dimensions > input.element_count()
            || output.element_count() != input.element_count()
            || !theta.is_finite()
            || theta <= 1.0
            || [input.buffer_view(), output.buffer_view()]
                .iter()
                .any(|view| view.device_ordinal() != stream.device_ordinal())
        {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let mut raw = AugurCudaDflashRopeF32ArgsV1::new();
        raw.input = addr(input.buffer_view())?;
        raw.input_len_bytes = input.buffer_view().len_bytes() as u64;
        raw.output = addr(output.buffer_view())?;
        raw.output_len_bytes = output.buffer_view().len_bytes() as u64;
        raw.elements = input
            .element_count()
            .try_into()
            .map_err(|_| error(CudaRuntimeErrorKind::InvalidDimensions))?;
        raw.dimensions = dimensions
            .try_into()
            .map_err(|_| error(CudaRuntimeErrorKind::InvalidDimensions))?;
        raw.rotary_dimensions = rotary_dimensions
            .try_into()
            .map_err(|_| error(CudaRuntimeErrorKind::InvalidDimensions))?;
        raw.position = position
            .try_into()
            .map_err(|_| error(CudaRuntimeErrorKind::RangeOverflow))?;
        raw.theta = theta;
        Ok(Self {
            raw,
            ordinal: stream.device_ordinal(),
        })
    }

    pub fn launch(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.ordinal {
            return Err(error(CudaRuntimeErrorKind::DeviceMismatch));
        }
        let result = unsafe {
            augur_cuda_sys::augur_cuda_dflash_rope_f32_v1(stream.device_ordinal(), stream.raw()?, &self.raw)
        };
        CudaRuntimeError::from_result("cuda_dflash_rope_f32_v1", result)
    }

    pub fn submit_for_plan(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.ordinal {
            return Err(error(CudaRuntimeErrorKind::DeviceMismatch));
        }
        let result = unsafe {
            augur_cuda_sys::augur_cuda_dflash_rope_f32_submit_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result("cuda_dflash_rope_f32_submit_v1", result)
    }
}

fn error(kind: CudaRuntimeErrorKind) -> CudaRuntimeError {
    CudaRuntimeError::validation("cuda_norm", kind)
}
fn addr(view: CudaBufferView) -> Result<u64, CudaRuntimeError> {
    view.device_ptr()
        .raw_value()
        .checked_add(view.offset_bytes() as u64)
        .ok_or_else(|| error(CudaRuntimeErrorKind::RangeOverflow))
}


#[cfg(test)]
mod warp_tree_tests {
    use super::*;
    #[test]
    fn exact_warp_tree_selection_keeps_geometry_and_original_default() {
        for n in [0,128,5120,5121] {
            let mut raw = AugurCudaRmsnormF32ArgsV1::new(); raw.elements=n;
            let op=CudaRmsnormF32ArgsV1 {raw,ordinal:0,exact_warp_tree5120:false};
            assert!(!op.exact_warp_tree5120);
            let selected=op.with_exact_warp_tree5120();
            assert_eq!(selected.is_ok(),n==5120);
            if let Ok(candidate)=selected {
                assert!(candidate.exact_warp_tree5120);
                assert_eq!(candidate.raw.elements,op.raw.elements);
                assert_eq!(candidate.raw.epsilon_bits,op.raw.epsilon_bits);
            }
        }
    }

    #[test]
    #[ignore="exclusive SM70; direct native norm identity and owner cleanup"]
    fn real_warp_tree_norm_identity_cleanup() {
        use augur_core::PhysicalLayoutId;
        use crate::cuda_device_allocation_snapshot;
        let baseline=cuda_device_allocation_snapshot();
        {
            let stream=OwnedCudaStream::create(0).unwrap();
            let x=OwnedCudaBuffer::allocate(0,5120*4).unwrap();
            let w=OwnedCudaBuffer::allocate(0,5120*4).unwrap();
            let control=OwnedCudaBuffer::allocate(0,5120*4).unwrap();
            let candidate=OwnedCudaBuffer::allocate(0,5120*4).unwrap();
            let view=|b:&OwnedCudaBuffer| CudaF32View::new(b.view(0,5120*4).unwrap(),
                PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,5120).unwrap();
            let original=CudaRmsnormF32ArgsV1::bind(&stream,view(&x),view(&w),view(&control),1e-6).unwrap();
            let fast=CudaRmsnormF32ArgsV1::bind(&stream,view(&x),view(&w),view(&candidate),1e-6)
                .unwrap().with_exact_warp_tree5120().unwrap();
            let weights:Vec<u8>=(0..5120).flat_map(|i|(1.0f32+(i as f32*0.13).sin()*0.7).to_le_bytes()).collect();
            w.copy_from_host_sync(&stream,&weights).unwrap();
            for scale in [0.0,1e-20,0.1,1.0,13.0,1e8] {
                let values:Vec<u8>=(0..5120).flat_map(|i|((i as f32*0.123).sin()*scale).to_le_bytes()).collect();
                x.copy_from_host_sync(&stream,&values).unwrap();
                original.submit_for_plan(&stream).unwrap();
                fast.submit_for_plan(&stream).unwrap();
                let mut a=vec![0;5120*4];let mut b=vec![0;5120*4];
                control.copy_to_host_sync(&stream,&mut a).unwrap();
                candidate.copy_to_host_sync(&stream,&mut b).unwrap();assert_eq!(a,b);
                fast.launch(&stream).unwrap();candidate.copy_to_host_sync(&stream,&mut b).unwrap();assert_eq!(a,b);
            }
            let mut bad=fast.raw;bad.elements=5119;
            let rejected=unsafe {augur_cuda_sys::augur_cuda_rmsnorm_f32_warp_tree5120_submit_v1(0,stream.raw().unwrap(),&bad)};
            assert!(CudaRuntimeError::from_result("wrong norm shape",rejected).is_err());
            stream.synchronize().unwrap();
        }
        let after=cuda_device_allocation_snapshot();
        assert_eq!(after.live_bytes,baseline.live_bytes);
        assert_eq!(after.live_allocations,baseline.live_allocations);
        assert_eq!(after.successful_allocations-baseline.successful_allocations,
                   after.successful_frees-baseline.successful_frees);
    }
}
