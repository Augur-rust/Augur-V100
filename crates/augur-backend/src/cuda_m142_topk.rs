#![cfg(feature = "cuda")]

use crate::{CudaBufferView, CudaF32View, CudaRuntimeError, CudaRuntimeErrorKind, OwnedCudaStream};
use augur_cuda_sys::AugurCudaM142OrderedTopkArgsV1;

const OP: &str = "cuda_m142_ordered_topk_v1";

#[derive(Clone, Copy, Debug)]
pub struct CudaM142OrderedTopkViewsV1 {
    pub scores: CudaF32View,
    pub indices: CudaBufferView,
    pub selected: CudaF32View,
}

#[derive(Clone, Copy, Debug)]
pub struct CudaM142OrderedTopkConfigV1 {
    pub tokens: usize,
    pub candidates: usize,
    pub top_k: usize,
    pub causal_ratio: usize,
    pub index_base: i32,
    pub padding: i32,
}

#[derive(Clone, Copy, Debug)]
pub struct CudaM142OrderedTopkArgsV1 {
    raw: AugurCudaM142OrderedTopkArgsV1,
    ordinal: i32,
}

impl CudaM142OrderedTopkArgsV1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        views: CudaM142OrderedTopkViewsV1,
        config: CudaM142OrderedTopkConfigV1,
    ) -> Result<Self, CudaRuntimeError> {
        Self::bind_for_ordinal(stream.device_ordinal(), views, config)
    }

    fn bind_for_ordinal(
        ordinal: i32,
        views: CudaM142OrderedTopkViewsV1,
        config: CudaM142OrderedTopkConfigV1,
    ) -> Result<Self, CudaRuntimeError> {
        let CudaM142OrderedTopkViewsV1 {
            scores,
            indices,
            selected,
        } = views;
        let CudaM142OrderedTopkConfigV1 {
            tokens,
            candidates,
            top_k,
            causal_ratio,
            index_base,
            padding,
        } = config;
        let score_elements = tokens
            .checked_mul(candidates)
            .ok_or_else(|| error(CudaRuntimeErrorKind::RangeOverflow))?;
        let output_elements = tokens
            .checked_mul(top_k)
            .ok_or_else(|| error(CudaRuntimeErrorKind::RangeOverflow))?;
        let score_bytes = score_elements
            .checked_mul(4)
            .ok_or_else(|| error(CudaRuntimeErrorKind::RangeOverflow))?;
        let out_bytes = output_elements
            .checked_mul(4)
            .ok_or_else(|| error(CudaRuntimeErrorKind::RangeOverflow))?;
        if tokens == 0
            || candidates == 0
            || top_k == 0
            || top_k > candidates
            || causal_ratio == 0
            || index_base < 0
            || padding >= index_base
            || scores.element_count() != score_elements
            || selected.element_count() != output_elements
            || indices.len_bytes() != out_bytes
            || scores.buffer_view().device_ordinal() != ordinal
            || indices.device_ordinal() != ordinal
            || selected.buffer_view().device_ordinal() != ordinal
            || scores.buffer_view().len_bytes() != score_bytes
            || selected.buffer_view().len_bytes() != out_bytes
        {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        for view in [scores.buffer_view(), indices, selected.buffer_view()] {
            if view.checked_address(OP)? % 4 != 0 {
                return Err(error(CudaRuntimeErrorKind::MisalignedDeviceAddress));
            }
            view.checked_range(OP)?;
        }
        let resources = [scores.buffer_view(), indices, selected.buffer_view()];
        for left in 0..resources.len() {
            for right in left + 1..resources.len() {
                if resources[left].overlaps(resources[right], OP)? {
                    return Err(error(CudaRuntimeErrorKind::OverlappingDeviceRanges));
                }
            }
        }
        let mut raw = AugurCudaM142OrderedTopkArgsV1::new();
        raw.scores = scores.buffer_view().checked_address(OP)?;
        raw.scores_len_bytes =
            u64::try_from(score_bytes).map_err(|_| error(CudaRuntimeErrorKind::RangeOverflow))?;
        raw.indices = indices.checked_address(OP)?;
        raw.indices_len_bytes =
            u64::try_from(out_bytes).map_err(|_| error(CudaRuntimeErrorKind::RangeOverflow))?;
        raw.selected_scores = selected.buffer_view().checked_address(OP)?;
        raw.selected_scores_len_bytes =
            u64::try_from(out_bytes).map_err(|_| error(CudaRuntimeErrorKind::RangeOverflow))?;
        raw.tokens =
            u32::try_from(tokens).map_err(|_| error(CudaRuntimeErrorKind::InvalidDimensions))?;
        raw.candidates = u32::try_from(candidates)
            .map_err(|_| error(CudaRuntimeErrorKind::InvalidDimensions))?;
        raw.top_k =
            u32::try_from(top_k).map_err(|_| error(CudaRuntimeErrorKind::InvalidDimensions))?;
        raw.causal_ratio = u32::try_from(causal_ratio)
            .map_err(|_| error(CudaRuntimeErrorKind::InvalidDimensions))?;
        raw.index_base = index_base;
        raw.padding = padding;
        Ok(Self { raw, ordinal })
    }
    pub fn launch(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.launch_sync(stream)
    }

    /// Launches the frozen M142 operation and waits for `stream` before returning.
    pub fn launch_sync(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.ordinal {
            return Err(CudaRuntimeError::device_mismatch());
        }
        let result = unsafe {
            augur_cuda_sys::augur_cuda_m142_ordered_topk_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result(OP, result)?;
        crate::cuda_runtime::record_stream_synchronization();
        Ok(())
    }

    pub fn submit_for_plan(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.ordinal {
            return Err(CudaRuntimeError::device_mismatch());
        }
        let result = unsafe {
            augur_cuda_sys::augur_cuda_m142_ordered_topk_submit_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result("cuda_m142_ordered_topk_submit_v1", result)
    }

    
}
fn error(kind: CudaRuntimeErrorKind) -> CudaRuntimeError {
    CudaRuntimeError::validation(OP, kind)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{cuda_host_boundary_snapshot, cuda_runtime::test_buffer_view, OwnedCudaBuffer};
    use augur_core::PhysicalLayoutId;

    fn valid_views() -> CudaM142OrderedTopkViewsV1 {
        CudaM142OrderedTopkViewsV1 {
            scores: CudaF32View::new(
                test_buffer_view(0x10_0000, 0, 0, 32),
                PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                8,
            )
            .unwrap(),
            indices: test_buffer_view(0x20_0000, 0, 0, 8),
            selected: CudaF32View::new(
                test_buffer_view(0x30_0000, 0, 0, 8),
                PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                2,
            )
            .unwrap(),
        }
    }

    fn valid_config() -> CudaM142OrderedTopkConfigV1 {
        CudaM142OrderedTopkConfigV1 {
            tokens: 1,
            candidates: 8,
            top_k: 2,
            causal_ratio: 1,
            index_base: 1,
            padding: 0,
        }
    }

    #[test]
    fn causal_ratio_must_fit_the_native_abi() {
        if usize::BITS <= 32 {
            return;
        }
        let mut config = valid_config();
        config.causal_ratio = u32::MAX as usize + 1;
        let error =
            CudaM142OrderedTopkArgsV1::bind_for_ordinal(0, valid_views(), config).unwrap_err();
        assert_eq!(error.kind, CudaRuntimeErrorKind::InvalidDimensions);
    }

    #[test]
    fn output_alias_is_rejected() {
        let mut views = valid_views();
        views.indices = test_buffer_view(0x10_0000, 0, 0, 8);
        let error =
            CudaM142OrderedTopkArgsV1::bind_for_ordinal(0, views, valid_config()).unwrap_err();
        assert_eq!(error.kind, CudaRuntimeErrorKind::OverlappingDeviceRanges);
    }

    #[test]
    fn dimension_multiplication_overflow_is_structured() {
        let mut config = valid_config();
        config.tokens = usize::MAX;
        config.candidates = 2;
        let error =
            CudaM142OrderedTopkArgsV1::bind_for_ordinal(0, valid_views(), config).unwrap_err();
        assert_eq!(error.kind, CudaRuntimeErrorKind::RangeOverflow);
    }

    

    #[test]
    #[ignore = "requires a CUDA device"]
    fn launch_sync_records_the_native_stream_wait() -> Result<(), Box<dyn std::error::Error>> {
        let stream = OwnedCudaStream::create(0)?;
        let scores = OwnedCudaBuffer::allocate(0, 32)?;
        let indices = OwnedCudaBuffer::allocate(0, 8)?;
        let selected = OwnedCudaBuffer::allocate(0, 8)?;
        scores.copy_from_host_sync(&stream, &[0_u8; 32])?;
        let args = CudaM142OrderedTopkArgsV1::bind(
            &stream,
            CudaM142OrderedTopkViewsV1 {
                scores: CudaF32View::new(
                    scores.view(0, 32)?,
                    PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                    8,
                )?,
                indices: indices.view(0, 8)?,
                selected: CudaF32View::new(
                    selected.view(0, 8)?,
                    PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                    2,
                )?,
            },
            valid_config(),
        )?;

        let before = cuda_host_boundary_snapshot();
        args.launch_sync(&stream)?;
        let delta = cuda_host_boundary_snapshot()
            .checked_delta_since(before)
            .ok_or("CUDA host-boundary counter regressed")?;

        assert_eq!(delta.host_waits, 1);
        assert_eq!(delta.stream_synchronizations, 1);
        assert_eq!(delta.event_synchronizations, 0);
        assert_eq!(delta.synchronous_h2d_copies, 0);
        assert_eq!(delta.synchronous_d2h_copies, 0);
        Ok(())
    }
}
