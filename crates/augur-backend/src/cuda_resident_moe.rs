#![cfg(feature = "cuda")]

use augur_cuda_sys::AugurCudaResidentMoeF32ArgsV1;

use crate::{CudaBufferView, CudaRuntimeError, CudaRuntimeErrorKind, OwnedCudaStream};

const OPERATION: &str = "cuda_resident_moe_f32_v1";

#[derive(Clone, Copy, Debug)]
pub struct CudaResidentMoeF32ArgsV1 {
    raw: AugurCudaResidentMoeF32ArgsV1,
    device_ordinal: i32,
}

impl CudaResidentMoeF32ArgsV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn bind(
        stream: &OwnedCudaStream,
        weights: CudaBufferView,
        input: CudaBufferView,
        output: CudaBufferView,
        router_logits: CudaBufferView,
        selected_ids: CudaBufferView,
        selected_weights: CudaBufferView,
        scratch: CudaBufferView,
        access_counts: CudaBufferView,
        tokens: usize,
        hidden: usize,
        intermediate: usize,
        expert_count: usize,
        top_k: usize,
        routed_scale: f32,
    ) -> Result<Self, CudaRuntimeError> {
        let ordinal = stream.device_ordinal();
        for view in [
            weights,
            input,
            output,
            router_logits,
            selected_ids,
            selected_weights,
            scratch,
            access_counts,
        ] {
            if view.device_ordinal() != ordinal {
                return Err(validation(CudaRuntimeErrorKind::DeviceMismatch));
            }
        }
        if tokens == 0
            || hidden == 0
            || intermediate == 0
            || !matches!(expert_count, 4 | 8)
            || !matches!(top_k, 2 | 4)
            || top_k > expert_count
            || !routed_scale.is_finite()
            || routed_scale <= 0.0
        {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let mut raw = AugurCudaResidentMoeF32ArgsV1::new();
        set_view(&mut raw.weights, &mut raw.weights_len_bytes, weights)?;
        set_view(&mut raw.input, &mut raw.input_len_bytes, input)?;
        set_view(&mut raw.output, &mut raw.output_len_bytes, output)?;
        set_view(
            &mut raw.router_logits,
            &mut raw.router_logits_len_bytes,
            router_logits,
        )?;
        set_view(
            &mut raw.selected_ids,
            &mut raw.selected_ids_len_bytes,
            selected_ids,
        )?;
        set_view(
            &mut raw.selected_weights,
            &mut raw.selected_weights_len_bytes,
            selected_weights,
        )?;
        set_view(&mut raw.scratch, &mut raw.scratch_len_bytes, scratch)?;
        set_view(
            &mut raw.access_counts,
            &mut raw.access_counts_len_bytes,
            access_counts,
        )?;
        raw.tokens = to_u32(tokens)?;
        raw.hidden = to_u32(hidden)?;
        raw.intermediate = to_u32(intermediate)?;
        raw.expert_count = to_u32(expert_count)?;
        raw.top_k = to_u32(top_k)?;
        raw.routed_scale = routed_scale;
        Ok(Self {
            raw,
            device_ordinal: ordinal,
        })
    }

    pub const fn as_raw(&self) -> &AugurCudaResidentMoeF32ArgsV1 {
        &self.raw
    }

    pub fn launch(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        let result = unsafe {
            augur_cuda_sys::augur_cuda_resident_moe_f32_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result(OPERATION, result)
    }

    pub fn submit_for_plan(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        let result = unsafe {
            augur_cuda_sys::augur_cuda_resident_moe_f32_submit_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result("cuda_resident_moe_f32_submit_v1", result)
    }

    fn validate_stream(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if self.device_ordinal != stream.device_ordinal() {
            return Err(validation(CudaRuntimeErrorKind::DeviceMismatch));
        }
        Ok(())
    }
}

fn set_view(
    address_out: &mut u64,
    length_out: &mut u64,
    view: CudaBufferView,
) -> Result<(), CudaRuntimeError> {
    *address_out = view
        .device_ptr()
        .raw_value()
        .checked_add(
            u64::try_from(view.offset_bytes())
                .map_err(|_| validation(CudaRuntimeErrorKind::RangeOverflow))?,
        )
        .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))?;
    *length_out = u64::try_from(view.len_bytes())
        .map_err(|_| validation(CudaRuntimeErrorKind::RangeOverflow))?;
    Ok(())
}

fn to_u32(value: usize) -> Result<u32, CudaRuntimeError> {
    u32::try_from(value).map_err(|_| validation(CudaRuntimeErrorKind::InvalidDimensions))
}

fn validation(kind: CudaRuntimeErrorKind) -> CudaRuntimeError {
    CudaRuntimeError::validation(OPERATION, kind)
}
