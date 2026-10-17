#![cfg(feature = "cuda")]

use crate::{
    CudaQ4kF32DecodeArgsV1, CudaQ5kF32DecodeArgsV1, CudaQ6kF32DecodeArgsV1, CudaRuntimeError,
    OwnedCudaStream,
};

pub fn submit_m120_q4k_exact_sm70(
    stream: &OwnedCudaStream,
    args: &CudaQ4kF32DecodeArgsV1,
) -> Result<(), CudaRuntimeError> {
    let result = unsafe {
        augur_cuda_sys::augur_cuda_m120_q4k_exact_f32_decode_submit_v1(
            stream.device_ordinal(),
            stream.raw()?,
            args.as_raw(),
        )
    };
    CudaRuntimeError::from_result("cuda_m120_q4k_exact_sm70", result)
}

pub fn submit_m120_q5k_exact_sm70(
    stream: &OwnedCudaStream,
    args: &CudaQ5kF32DecodeArgsV1,
) -> Result<(), CudaRuntimeError> {
    let result = unsafe {
        augur_cuda_sys::augur_cuda_m120_q5k_exact_f32_decode_submit_v1(
            stream.device_ordinal(),
            stream.raw()?,
            args.as_raw(),
        )
    };
    CudaRuntimeError::from_result("cuda_m120_q5k_exact_sm70", result)
}

pub fn submit_m120_q6k_exact_sm70(
    stream: &OwnedCudaStream,
    args: &CudaQ6kF32DecodeArgsV1,
) -> Result<(), CudaRuntimeError> {
    let result = unsafe {
        augur_cuda_sys::augur_cuda_m120_q6k_exact_f32_decode_submit_v1(
            stream.device_ordinal(),
            stream.raw()?,
            args.as_raw(),
        )
    };
    CudaRuntimeError::from_result("cuda_m120_q6k_exact_sm70", result)
}
