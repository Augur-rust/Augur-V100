#![cfg(feature = "cuda")]

use crate::{CudaRuntimeError, OwnedCudaBuffer, OwnedCudaStream};

#[derive(Clone, Copy, Debug)]
pub struct CudaM121Q4kDp4aArgsV1 {
    raw: augur_cuda_sys::AugurCudaM121Q4kDp4aArgsV1,
    device_ordinal: i32,
}

impl CudaM121Q4kDp4aArgsV1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        weight: &OwnedCudaBuffer,
        q8_input: &OwnedCudaBuffer,
        q8_scales: &OwnedCudaBuffer,
        output: &OwnedCudaBuffer,
    ) -> Result<Self, CudaRuntimeError> {
        let raw = augur_cuda_sys::AugurCudaM121Q4kDp4aArgsV1 {
            abi_version: augur_cuda_sys::AUGUR_CUDA_M121_Q4K_DP4A_ARGS_VERSION_V1,
            struct_size: core::mem::size_of::<augur_cuda_sys::AugurCudaM121Q4kDp4aArgsV1>() as u32,
            weight: weight
                .device_ptr()
                .ok_or_else(|| CudaRuntimeError::closed("m121 weight"))?
                .raw_value(),
            weight_len_bytes: weight.len_bytes() as u64,
            q8_input: q8_input
                .device_ptr()
                .ok_or_else(|| CudaRuntimeError::closed("m121 q8 input"))?
                .raw_value(),
            q8_input_len_bytes: q8_input.len_bytes() as u64,
            q8_scales: q8_scales
                .device_ptr()
                .ok_or_else(|| CudaRuntimeError::closed("m121 q8 scales"))?
                .raw_value(),
            q8_scales_len_bytes: q8_scales.len_bytes() as u64,
            output: output
                .device_ptr()
                .ok_or_else(|| CudaRuntimeError::closed("m121 output"))?
                .raw_value(),
            output_len_bytes: output.len_bytes() as u64,
            rows: 2048,
            cols: 4096,
            blocks_per_row: 16,
            group_size: 32,
        };
        Ok(Self {
            raw,
            device_ordinal: stream.device_ordinal(),
        })
    }

    pub fn submit_for_plan(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.device_ordinal {
            return Err(CudaRuntimeError::device_mismatch());
        }
        let result = unsafe {
            augur_cuda_sys::augur_cuda_m121_q4k_dp4a_submit_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result("cuda_m121_q4k_dp4a", result)
    }
}
