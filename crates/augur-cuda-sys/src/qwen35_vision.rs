//! Fixed Vision ABI; opt-in alongside the existing cuBLAS translation unit.
use crate::{AugurCudaResultV1, AugurCudaStream, AugurCudaQwen35PrefillCublasPlanV1};
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AugurCudaQwen35VisionF16AffineArgsV1 {
    pub abi_version:u32, pub struct_size:u32, pub tokens:u32, pub rows:u32, pub cols:u32, pub reserved:u32,
    pub weight_f16:u64, pub weight_len_bytes:u64, pub input_f16:u64, pub input_len_bytes:u64,
    pub bias_f32:u64, pub bias_len_bytes:u64, pub output_f32:u64, pub output_len_bytes:u64,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AugurCudaQwen35VisionControlArgsV1 {
    pub abi_version:u32, pub struct_size:u32, pub operation:u32, pub tokens:u32, pub cols:u32, pub reserved:u32,
    pub input_a:u64, pub input_a_bytes:u64, pub input_b:u64, pub input_b_bytes:u64,
    pub input_c:u64, pub input_c_bytes:u64, pub input_d:u64, pub input_d_bytes:u64,
    pub input_e:u64, pub input_e_bytes:u64, pub output:u64, pub output_bytes:u64,
    pub status:u64, pub status_bytes:u64,
}
unsafe extern "C" {
    pub fn augur_cuda_qwen35_vision_f16_affine_preflight_v1(ordinal:i32,stream:AugurCudaStream,
        plan:*mut AugurCudaQwen35PrefillCublasPlanV1,args:*const AugurCudaQwen35VisionF16AffineArgsV1)->AugurCudaResultV1;
    pub fn augur_cuda_qwen35_vision_f16_affine_submit_v1(ordinal:i32,stream:AugurCudaStream,
        plan:*mut AugurCudaQwen35PrefillCublasPlanV1,args:*const AugurCudaQwen35VisionF16AffineArgsV1)->AugurCudaResultV1;
    pub fn augur_cuda_qwen35_vision_control_preflight_v1(ordinal:i32,stream:AugurCudaStream,
        args:*const AugurCudaQwen35VisionControlArgsV1)->AugurCudaResultV1;
    pub fn augur_cuda_qwen35_vision_control_submit_v1(ordinal:i32,stream:AugurCudaStream,
        args:*const AugurCudaQwen35VisionControlArgsV1)->AugurCudaResultV1;
}
