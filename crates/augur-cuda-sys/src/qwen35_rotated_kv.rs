//! Explicit rotated-cache ABI; incompatible with unrotated V3/V4 KV bytes.
use crate::{AugurCudaResultV1, AugurCudaStream};

pub const AUGUR_QWEN35_ROTATED_KV_FORMAT_V1: u32 = 1;
pub const AUGUR_QWEN35_ROTATED_KV_PLANAR_FORMAT_V2: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AugurCudaQwen35RotatedKvArgsV1 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub query: u64,
    pub query_bytes: u64,
    pub gate: u64,
    pub gate_bytes: u64,
    pub key: u64,
    pub key_bytes: u64,
    pub value: u64,
    pub value_bytes: u64,
    pub key_cache: u64,
    pub key_cache_bytes: u64,
    pub value_cache: u64,
    pub value_cache_bytes: u64,
    pub attention: u64,
    pub attention_bytes: u64,
    pub gated: u64,
    pub gated_bytes: u64,
    pub workspace: u64,
    pub workspace_bytes: u64,
    pub capacity: u32,
    pub position: u32,
    pub rows: u32,
    pub cache_format: u32,
}

impl AugurCudaQwen35RotatedKvArgsV1 {
    pub fn new() -> Self {
        Self {
            abi_version: 1,
            struct_size: core::mem::size_of::<Self>() as u32,
            cache_format: AUGUR_QWEN35_ROTATED_KV_FORMAT_V1,
            ..Self::default()
        }
    }
}

const _: () = assert!(core::mem::size_of::<AugurCudaQwen35RotatedKvArgsV1>() == 168);
const _: () = assert!(core::mem::align_of::<AugurCudaQwen35RotatedKvArgsV1>() == 8);

unsafe extern "C" {
    pub fn augur_cuda_qwen35_rotated_kv_planar_gqa8_c4_configure_v1(ordinal: i32) -> AugurCudaResultV1;
    pub fn augur_cuda_qwen35_rotated_kv_planar_gqa8_c4_submit_v1(
        ordinal: i32, stream: AugurCudaStream, args: *const AugurCudaQwen35RotatedKvArgsV1,
    ) -> AugurCudaResultV1;
    pub fn augur_cuda_qwen35_rotated_kv_planar_gqa8_configure_v1(ordinal: i32) -> AugurCudaResultV1;
    pub fn augur_cuda_qwen35_rotated_kv_planar_gqa8_submit_v1(
        ordinal: i32, stream: AugurCudaStream, args: *const AugurCudaQwen35RotatedKvArgsV1,
    ) -> AugurCudaResultV1;
    pub fn augur_cuda_qwen35_rotated_kv_planar_input_configure_v1(ordinal: i32) -> AugurCudaResultV1;
    pub fn augur_cuda_qwen35_rotated_kv_planar_input_submit_v1(
        ordinal: i32, stream: AugurCudaStream, args: *const AugurCudaQwen35RotatedKvArgsV1,
    ) -> AugurCudaResultV1;
    pub fn augur_cuda_qwen35_rotated_kv_gqa8_configure_v1(ordinal: i32) -> AugurCudaResultV1;
    pub fn augur_cuda_qwen35_rotated_kv_gqa8_validate_v1(
        args: *const AugurCudaQwen35RotatedKvArgsV1,
    ) -> AugurCudaResultV1;
    pub fn augur_cuda_qwen35_rotated_kv_gqa8_submit_v1(
        ordinal: i32, stream: AugurCudaStream, args: *const AugurCudaQwen35RotatedKvArgsV1,
    ) -> AugurCudaResultV1;
    pub fn augur_cuda_qwen35_rotated_kv_validate_v1(
        args: *const AugurCudaQwen35RotatedKvArgsV1,
    ) -> AugurCudaResultV1;
    pub fn augur_cuda_qwen35_rotated_kv_configure_v1(ordinal: i32) -> AugurCudaResultV1;
    pub fn augur_cuda_qwen35_rotated_kv_submit_v1(
        ordinal: i32,
        stream: AugurCudaStream,
        args: *const AugurCudaQwen35RotatedKvArgsV1,
    ) -> AugurCudaResultV1;
    pub fn augur_cuda_qwen35_rotated_kv_small_validate_v1(
        args: *const AugurCudaQwen35RotatedKvArgsV1,
    ) -> AugurCudaResultV1;
    pub fn augur_cuda_qwen35_rotated_kv_small_configure_v1(ordinal: i32) -> AugurCudaResultV1;
    pub fn augur_cuda_qwen35_rotated_kv_small_submit_v1(
        ordinal: i32,
        stream: AugurCudaStream,
        args: *const AugurCudaQwen35RotatedKvArgsV1,
    ) -> AugurCudaResultV1;
    pub fn augur_cuda_qwen35_rotated_kv_input_validate_v1(
        args: *const AugurCudaQwen35RotatedKvArgsV1,
    ) -> AugurCudaResultV1;
    pub fn augur_cuda_qwen35_rotated_kv_small_vector_submit_v1(
        ordinal: i32,
        stream: AugurCudaStream,
        args: *const AugurCudaQwen35RotatedKvArgsV1,
    ) -> AugurCudaResultV1;
    pub fn augur_cuda_qwen35_rotated_kv_small_tile8_submit_v1(
        ordinal: i32,
        stream: AugurCudaStream,
        args: *const AugurCudaQwen35RotatedKvArgsV1,
    ) -> AugurCudaResultV1;
    pub fn augur_cuda_qwen35_rotated_kv_input_configure_v1(ordinal: i32) -> AugurCudaResultV1;
    pub fn augur_cuda_qwen35_rotated_kv_input_submit_v1(
        ordinal: i32,
        stream: AugurCudaStream,
        args: *const AugurCudaQwen35RotatedKvArgsV1,
    ) -> AugurCudaResultV1;
}
