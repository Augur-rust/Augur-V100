//! Distinct QPN physical and numerical ABI; never reinterpret GGML arguments.
use crate::{AugurCudaQwen35PrefillCublasPlanV1, AugurCudaResultV1, AugurCudaStream};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AugurCudaQwen35QpnPackArgsV1 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub source: u64,
    pub source_bytes: u64,
    pub destination: u64,
    pub destination_bytes: u64,
    pub rows: u32,
    pub cols: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AugurCudaQwen35QpnStageArgsV1 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub source: u64,
    pub source_bytes: u64,
    pub destination: u64,
    pub destination_bytes: u64,
    pub elements: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AugurCudaQwen35QpnExpandArgsV1 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub source: u64,
    pub source_bytes: u64,
    pub destination: u64,
    pub destination_bytes: u64,
    pub rows: u32,
    pub cols: u32,
    pub tensor_scale: f32,
    pub numerical_profile: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AugurCudaQwen35QpnProjectArgsV1 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub weight: u64,
    pub weight_bytes: u64,
    pub input: u64,
    pub input_bytes: u64,
    pub output: u64,
    pub output_bytes: u64,
    pub rows: u32,
    pub cols: u32,
    pub tokens: u32,
    pub split_k: u32,
    pub tensor_scale: f32,
    pub numerical_profile: u32,
}
macro_rules! abi {
    ($ty:ty, $size:expr) => {
        const _: () = assert!(core::mem::size_of::<$ty>() == $size);
        const _: () = assert!(core::mem::align_of::<$ty>() == 8);
        impl $ty {
            pub fn new() -> Self {
                Self {
                    abi_version: 1,
                    struct_size: $size,
                    ..Self::default()
                }
            }
        }
    };
}
abi!(AugurCudaQwen35QpnPackArgsV1, 48);
abi!(AugurCudaQwen35QpnStageArgsV1, 48);
abi!(AugurCudaQwen35QpnExpandArgsV1, 56);
abi!(AugurCudaQwen35QpnProjectArgsV1, 80);
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AugurCudaQwen35QpnPrefillArgsV1 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub weight: u64,
    pub weight_bytes: u64,
    pub input_f32: u64,
    pub input_bytes: u64,
    pub output_f32: u64,
    pub output_bytes: u64,
    pub rows: u32,
    pub cols: u32,
    pub tokens: u32,
    pub split_k: u32,
    pub tensor_scale: f32,
    pub numerical_profile: u32,
}
abi!(AugurCudaQwen35QpnPrefillArgsV1, 80);
extern "C" {
    pub fn augur_cuda_qwen35_qpn_prefill_preflight_v1(
        ordinal: i32,
        stream: AugurCudaStream,
        plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
        args: *const AugurCudaQwen35QpnPrefillArgsV1,
    ) -> AugurCudaResultV1;
    pub fn augur_cuda_qwen35_qpn_prefill_prevalidated_submit_v1(
        ordinal: i32,
        stream: AugurCudaStream,
        plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
        args: *const AugurCudaQwen35QpnPrefillArgsV1,
    ) -> AugurCudaResultV1;
    pub fn augur_cuda_qwen35_qpn_prefill_pair_prevalidated_submit_v1(
        ordinal: i32,
        stream: AugurCudaStream,
        plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
        first: *const AugurCudaQwen35QpnPrefillArgsV1,
        second: *const AugurCudaQwen35QpnPrefillArgsV1,
    ) -> AugurCudaResultV1;
    pub fn augur_cuda_qwen35_qpn_silu_down_direct_b8_prevalidated_submit_v1(
        ordinal: i32,
        stream: AugurCudaStream,
        plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
        activation: *const crate::AugurCudaSiluMulF32ArgsV1,
        down: *const AugurCudaQwen35QpnPrefillArgsV1,
    ) -> AugurCudaResultV1;
}
extern "C" {
    pub fn augur_cuda_qwen35_qpn_norm_group_preflight_v1(
        ordinal: i32, stream: AugurCudaStream, plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
        norm: *const crate::AugurCudaRmsnormF32ArgsV1,
        projections: *const AugurCudaQwen35QpnPrefillArgsV1, count: u32,
    ) -> AugurCudaResultV1;
    pub fn augur_cuda_qwen35_qpn_norm_group_prevalidated_submit_v1(
        ordinal: i32, stream: AugurCudaStream, plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
        norm: *const crate::AugurCudaRmsnormF32ArgsV1,
        projections: *const AugurCudaQwen35QpnPrefillArgsV1, count: u32,
    ) -> AugurCudaResultV1;
}
/// Direct-B8 norm -> stage -> grouped profile2 projections (see the native ABI header).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AugurCudaQwen35QpnNormGroupB8ArgsV1 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub count: u32,
    pub residual: u32,
    pub norm: crate::AugurCudaQwen35PrefillRmsnormArgsV1,
    pub projections: [AugurCudaQwen35QpnPrefillArgsV1; 4],
    pub out_cols: [u32; 4],
    pub records: [u64; 4],
    pub record_bytes: [u64; 4],
    pub add_a: u64,
    pub add_b: u64,
    pub add_len_bytes: u64,
}
abi!(AugurCudaQwen35QpnNormGroupB8ArgsV1, 512);
extern "C" {
    pub fn augur_cuda_qwen35_qpn_norm_group_b8_preflight_v1(
        ordinal: i32, stream: AugurCudaStream, plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
        args: *const AugurCudaQwen35QpnNormGroupB8ArgsV1,
    ) -> AugurCudaResultV1;
    pub fn augur_cuda_qwen35_qpn_norm_group_b8_prevalidated_submit_v1(
        ordinal: i32, stream: AugurCudaStream, plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
        args: *const AugurCudaQwen35QpnNormGroupB8ArgsV1,
    ) -> AugurCudaResultV1;
}
macro_rules! entry {
    ($preflight:ident, $submit:ident, $ty:ty) => {
        extern "C" {
            pub fn $preflight(
                ordinal: i32,
                stream: AugurCudaStream,
                args: *const $ty,
            ) -> AugurCudaResultV1;
            pub fn $submit(ordinal: i32, stream: AugurCudaStream, args: *const $ty) -> AugurCudaResultV1;
        }
    };
}
entry!(
    augur_cuda_qwen35_qpn_pack_preflight_v1,
    augur_cuda_qwen35_qpn_pack_prevalidated_submit_v1,
    AugurCudaQwen35QpnPackArgsV1
);
entry!(
    augur_cuda_qwen35_qpn_pack_checkpoint_preflight_v1,
    augur_cuda_qwen35_qpn_pack_checkpoint_prevalidated_submit_v1,
    AugurCudaQwen35QpnPackArgsV1
);
entry!(
    augur_cuda_qwen35_qpn_stage_preflight_v1,
    augur_cuda_qwen35_qpn_stage_prevalidated_submit_v1,
    AugurCudaQwen35QpnStageArgsV1
);
entry!(
    augur_cuda_qwen35_qpn_expand_preflight_v1,
    augur_cuda_qwen35_qpn_expand_prevalidated_submit_v1,
    AugurCudaQwen35QpnExpandArgsV1
);
entry!(
    augur_cuda_qwen35_qpn_project_preflight_v1,
    augur_cuda_qwen35_qpn_project_prevalidated_submit_v1,
    AugurCudaQwen35QpnProjectArgsV1
);


#[cfg(feature = "qwen35-prefill-cublas")]
extern "C" {
 pub fn augur_cuda_qwen35_augur_prefill_preflight_v1(d:i32,s:AugurCudaStream,p:*mut AugurCudaQwen35PrefillCublasPlanV1,a:*const AugurCudaQwen35QpnPrefillArgsV1)->AugurCudaResultV1;
 pub fn augur_cuda_qwen35_augur_prefill_submit_v1(d:i32,s:AugurCudaStream,p:*mut AugurCudaQwen35PrefillCublasPlanV1,a:*const AugurCudaQwen35QpnPrefillArgsV1)->AugurCudaResultV1;
 pub fn augur_cuda_qwen35_augur_silu_down_submit_v1(
     d:i32,s:AugurCudaStream,p:*mut AugurCudaQwen35PrefillCublasPlanV1,
     a:*const crate::AugurCudaSiluMulF32ArgsV1,down:*const AugurCudaQwen35QpnPrefillArgsV1)->AugurCudaResultV1;
 pub fn augur_cuda_qwen35_augur_prefill_group_submit_v1(d:i32,s:AugurCudaStream,p:*mut AugurCudaQwen35PrefillCublasPlanV1,a:*const AugurCudaQwen35QpnPrefillArgsV1,count:u32)->AugurCudaResultV1;
}

#[cfg(feature = "qwen35-prefill-cublas")]
extern "C" {
    pub fn augur_cuda_qwen35_qpn_silu_down_prevalidated_submit_v1(
        ordinal: i32, stream: AugurCudaStream, plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
        activation: *const crate::AugurCudaSiluMulF32ArgsV1,
        down: *const AugurCudaQwen35QpnPrefillArgsV1,
    ) -> AugurCudaResultV1;
}
