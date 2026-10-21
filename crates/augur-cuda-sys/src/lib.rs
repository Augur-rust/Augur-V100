#![deny(unsafe_op_in_unsafe_fn)]

//! Raw Rust/CUDA ABI: this is a trust boundary, not a safe ownership API.
//!
//! Each foreign call requires valid host argument storage, matching ABI/layout,
//! live native handles, and live device allocation incarnations. A non-null
//! pointer, successful preflight, or submission status is not a completion or
//! lifetime proof. See the per-function `# Safety` contracts and the repository's
//! `docs/UNSAFE_TRUST_BOUNDARIES.md` (TB-ABI, TB-ASYNC, GAP-VIEW).
//! Native CUDA/C++ kernels, cuBLAS, the CUDA runtime and the driver remain part
//! of the trusted computing base; Rust's borrow checker does not validate them.

#[doc(hidden)]
pub const AUGUR_CUDA_SYS_RESOLVED_FEATURES: &str = env!("AUGUR_CUDA_SYS_RESOLVED_FEATURES");

#[cfg(feature = "cuda")]
mod ffi {
    use core::ffi::{c_char, c_int, c_void};

    pub const AUGUR_CUDA_ABI_VERSION_V1: u32 = 1;
    pub const AUGUR_CUDA_DEVICE_NAME_CAPACITY: usize = 256;

    pub const AUGUR_CUDA_STATUS_OK: i32 = 0;
    pub const AUGUR_CUDA_STATUS_INVALID_ARGUMENT: i32 = -1;
    pub const AUGUR_CUDA_STATUS_UNSUPPORTED_DEVICE: i32 = -2;
    pub const AUGUR_CUDA_STATUS_OUT_OF_MEMORY: i32 = -3;
    pub const AUGUR_CUDA_STATUS_RUNTIME_FAILURE: i32 = -4;
    pub const AUGUR_CUDA_STATUS_SUBMISSION_FAILURE: i32 = -5;
    pub const AUGUR_CUDA_STATUS_COMPLETION_FAILURE: i32 = -6;
    pub const AUGUR_CUDA_STATUS_ABI_VERSION_MISMATCH: i32 = -7;
    pub const AUGUR_CUDA_STATUS_DIAGNOSTIC_FAULT_INJECTED: i32 = -8;

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    pub struct AugurCudaD2dCopyEntryV1 {
        pub source: u64,
        pub destination: u64,
        pub bytes: u64,
    }
    
    
    
    

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaResultV1 {
        pub status: i32,
        pub raw_cuda_status: i32,
    }

    impl AugurCudaResultV1 {
        pub const fn is_ok(self) -> bool {
            self.status == AUGUR_CUDA_STATUS_OK
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct AugurCudaDevicePropertiesV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub ordinal: i32,
        pub compute_major: u16,
        pub compute_minor: u16,
        pub total_global_memory_bytes: u64,
        pub name: [c_char; AUGUR_CUDA_DEVICE_NAME_CAPACITY],
    }

    impl AugurCudaDevicePropertiesV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_ABI_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                ordinal: -1,
                compute_major: 0,
                compute_minor: 0,
                total_global_memory_bytes: 0,
                name: [0; AUGUR_CUDA_DEVICE_NAME_CAPACITY],
            }
        }
    }

    impl Default for AugurCudaDevicePropertiesV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    #[repr(C)]
    pub struct AugurCudaStreamHandle {
        _private: [u8; 0],
    }

    pub type AugurCudaStream = *mut AugurCudaStreamHandle;

    #[repr(C)]
    pub struct AugurCudaEventHandle {
        _private: [u8; 0],
    }

    pub type AugurCudaEvent = *mut AugurCudaEventHandle;

    #[repr(C)]
    pub struct AugurCudaGraphExecHandle {
        _private: [u8; 0],
    }

    pub type AugurCudaGraphExec = *mut AugurCudaGraphExecHandle;

    pub type AugurCudaDevicePtr = u64;

    #[cfg(feature = "qwen35-kv8-full-attention")]
    pub const AUGUR_CUDA_QWEN35_KV8_SEGMENT_PARTIAL_ARGS_VERSION_V2: u32 = 2;

    #[cfg(feature = "qwen35-kv8-full-attention")]
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaQwen35Kv8SegmentPartialArgsV2 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub query_f32: AugurCudaDevicePtr,
        pub query_len_bytes: u64,
        pub key_cache_q8_0: AugurCudaDevicePtr,
        pub key_cache_len_bytes: u64,
        pub value_cache_q8_0: AugurCudaDevicePtr,
        pub value_cache_len_bytes: u64,
        pub partial_max_f32: AugurCudaDevicePtr,
        pub partial_max_len_bytes: u64,
        pub partial_denominator_f32: AugurCudaDevicePtr,
        pub partial_denominator_len_bytes: u64,
        pub partial_weighted_value_f32: AugurCudaDevicePtr,
        pub partial_weighted_value_len_bytes: u64,
        pub tokens: u32,
        pub query_heads: u32,
        pub kv_heads: u32,
        pub head_dim: u32,
        pub cache_position_base: u32,
        pub segment_begin: u32,
        pub segment_end: u32,
        pub slot_capacity_positions: u32,
        pub visible_position_end: u32,
        pub mode: u32,
        pub reserved0: u32,
        pub reserved1: u32,
    }

    #[cfg(feature = "qwen35-kv8-full-attention")]
    const _: () = assert!(core::mem::size_of::<AugurCudaQwen35Kv8SegmentPartialArgsV2>() == 152);

    #[cfg(feature = "qwen35-kv8-full-attention")]
    pub const AUGUR_CUDA_QWEN35_KV8_PAGE_DIGEST_ARGS_VERSION_V1: u32 = 1;

    #[cfg(feature = "qwen35-kv8-full-attention")]
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaQwen35Kv8PageDigestArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub input: AugurCudaDevicePtr,
        pub input_len_bytes: u64,
        pub output: AugurCudaDevicePtr,
        pub output_len_bytes: u64,
        pub reserved0: u32,
        pub reserved1: u32,
    }

    #[cfg(feature = "qwen35-kv8-full-attention")]
    impl AugurCudaQwen35Kv8PageDigestArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_QWEN35_KV8_PAGE_DIGEST_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                input: 0,
                input_len_bytes: 0,
                output: 0,
                output_len_bytes: 32,
                reserved0: 0,
                reserved1: 0,
            }
        }
    }

    #[cfg(feature = "qwen35-kv8-full-attention")]
    impl Default for AugurCudaQwen35Kv8PageDigestArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    #[cfg(feature = "qwen35-kv8-full-attention")]
    const _: () = assert!(core::mem::size_of::<AugurCudaQwen35Kv8PageDigestArgsV1>() == 48);

    pub const AUGUR_CUDA_Q4K_F32_DECODE_ARGS_VERSION_V1: u32 = 1;
    pub const AUGUR_CUDA_Q4K_F32_PREFILL_ARGS_VERSION_V1: u32 = 1;
    pub const AUGUR_CUDA_ADD_F32_ARGS_VERSION_V1: u32 = 1;
    pub const AUGUR_CUDA_GQA_REPEAT_F32_ARGS_VERSION_V1: u32 = 1;
    pub const AUGUR_CUDA_Q4K_BLOCK_DECODED_LEN_V1: u32 = 256;
    pub const AUGUR_CUDA_Q4K_BLOCK_BYTES_V1: u32 = 160;
    pub const AUGUR_CUDA_Q4K_BLOCK_ALIGNMENT_V1: u32 = 32;
    
    
    
    pub const AUGUR_CUDA_Q5K_F32_DECODE_ARGS_VERSION_V1: u32 = 1;
    
    pub const AUGUR_CUDA_Q5K_BLOCK_BYTES_V1: u32 = 192;
    pub const AUGUR_CUDA_Q5K_BLOCK_ALIGNMENT_V1: u32 = 32;
    const _: () = assert!(core::mem::size_of::<AugurCudaQ8_0F32MatvecArgsV1>() == 64);
    const _: () = assert!(core::mem::align_of::<AugurCudaQ8_0F32MatvecArgsV1>() == 8);
    const _: () = assert!(core::mem::offset_of!(AugurCudaQ8_0F32MatvecArgsV1, weight) == 8);
    const _: () = assert!(core::mem::offset_of!(AugurCudaQ8_0F32MatvecArgsV1, input) == 24);
    const _: () = assert!(core::mem::offset_of!(AugurCudaQ8_0F32MatvecArgsV1, output) == 40);
    const _: () = assert!(core::mem::size_of::<AugurCudaQ8_1ActivationQuantizeArgsV1>() == 48);
    const _: () = assert!(core::mem::align_of::<AugurCudaQ8_1ActivationQuantizeArgsV1>() == 8);
    const _: () = assert!(core::mem::size_of::<AugurCudaQ8_0Q8_1MatvecArgsV1>() == 64);
    const _: () = assert!(core::mem::align_of::<AugurCudaQ8_0Q8_1MatvecArgsV1>() == 8);
    const _: () = assert!(core::mem::size_of::<AugurCudaSplitRmsnormF32ArgsV1>() == 80);
    const _: () = assert!(core::mem::align_of::<AugurCudaSplitRmsnormF32ArgsV1>() == 8);
    #[cfg(feature = "qwen35-fused-ffn-swiglu")]
    const _: () = assert!(core::mem::size_of::<AugurCudaQ8_0Q8_1FusedSwigluArgsV1>() == 80);
    #[cfg(feature = "qwen35-fused-ffn-swiglu")]
    const _: () = assert!(core::mem::align_of::<AugurCudaQ8_0Q8_1FusedSwigluArgsV1>() == 8);
    const _: () = assert!(core::mem::size_of::<AugurCudaNvfp4Sm70F32MatvecArgsV1>() == 72);
    const _: () = assert!(core::mem::align_of::<AugurCudaNvfp4Sm70F32MatvecArgsV1>() == 8);
    const _: () = assert!(core::mem::size_of::<AugurCudaNvfp4Sm70Q8_1MatvecArgsV1>() == 72);
    const _: () = assert!(core::mem::align_of::<AugurCudaNvfp4Sm70Q8_1MatvecArgsV1>() == 8);
    #[cfg(feature = "qwen35-fused-ffn-swiglu")]
    const _: () = assert!(core::mem::size_of::<AugurCudaNvfp4Sm70Q8_1FusedSwigluArgsV1>() == 88);
    #[cfg(feature = "qwen35-fused-ffn-swiglu")]
    const _: () = assert!(core::mem::align_of::<AugurCudaNvfp4Sm70Q8_1FusedSwigluArgsV1>() == 8);
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    pub const AUGUR_CUDA_Q6K_F32_DECODE_ARGS_VERSION_V1: u32 = 1;
    
    pub const AUGUR_CUDA_Q6K_BLOCK_BYTES_V1: u32 = 224;
    pub const AUGUR_CUDA_Q6K_BLOCK_ALIGNMENT_V1: u32 = 32;
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    pub const AUGUR_CUDA_M121_Q4K_DP4A_ARGS_VERSION_V1: u32 = 1;
    
    pub const AUGUR_CUDA_M142_ORDERED_TOPK_ARGS_VERSION_V1: u32 = 1;
    
    pub const AUGUR_CUDA_M283_KV_TRANSITION_ARGS_VERSION_V1: u32 = 1;
    pub const AUGUR_CUDA_M283_KV_CAPACITY_V1: u32 = 128;
    pub const AUGUR_CUDA_M283_KV_VARIANT_RAW_V1: u32 = 0;
    
    
    pub const AUGUR_CUDA_M285_FILL_SPAN_ARGS_VERSION_V1: u32 = 1;
    pub const AUGUR_CUDA_KV8_DSA_QUANTIZE_ARGS_VERSION_V1: u32 = 1;
    pub const AUGUR_CUDA_KV8_DSA_QUANTIZE_M128_ARGS_VERSION_V1: u32 = 1;
    pub const AUGUR_CUDA_KV8_DSA_QUANTIZE_M128_ROWS_V1: u32 = 128;
    pub const AUGUR_CUDA_KV8_DSA_STATUS_COPY_ARGS_VERSION_V1: u32 = 1;
    pub const AUGUR_CUDA_KV8_DSA_MAX_ROWS_V1: usize = 3;
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    pub const AUGUR_CUDA_KV8_DSA_SELECT_ARGS_VERSION_V1: u32 = 1;
    
    
    
    pub const AUGUR_CUDA_KV8_DSA_SELECT_TOP_K_V1: u32 = 10;
    pub const AUGUR_CUDA_KV8_DSA_SELECT_INDEX_BASE_V1: u32 = 128;
    pub const AUGUR_CUDA_KV8_DSA_SELECT_PADDING_V1: i32 = -1;
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    pub const AUGUR_CUDA_KV8_DSA_ATTENTION_ARGS_VERSION_V1: u32 = 1;
    
    
    
    
    pub const AUGUR_CUDA_KV8_DSA_ATTENTION_FAMILY_RAW_V1: u32 = 0;
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    pub const AUGUR_CUDA_KV8_DSA_ATTENTION_SPLIT_ARGS_VERSION_V2: u32 = 2;
    
    
    
    
    
    
    
    
    pub const AUGUR_CUDA_M292_HBM_ARGS_VERSION_V1: u32 = 1;
    pub const AUGUR_CUDA_M292_HBM_READ_ONLY: u32 = 1;
    pub const AUGUR_CUDA_M292_HBM_READ_WRITE: u32 = 2;
    pub const AUGUR_CUDA_M306_K7_HC_FP64_ARGS_VERSION_V1: u32 = 1;
    pub const AUGUR_CUDA_M306_K7_HC_HIDDEN_V1: u32 = 16_384;
    pub const AUGUR_CUDA_M306_K7_HC_OUTPUTS_V1: u32 = 24;
    pub const AUGUR_CUDA_M306_K7_HC_CHUNKS_V1: u32 = 64;
    pub const AUGUR_CUDA_M306_K7_HC_THREADS_V1: u32 = 256;
    
    
    
    pub const AUGUR_CUDA_M306_K7_HC_FUNCTION_BYTES_V1: u64 = 1_572_864;
    
    
    pub const AUGUR_CUDA_M306_K11_HC_EXACT_ARGS_VERSION_V1: u32 = 1;
    pub const AUGUR_CUDA_M306_K11_HC_EXACT_HIDDEN_V1: u32 = 16_384;
    pub const AUGUR_CUDA_M306_K11_HC_EXACT_OUTPUTS_V1: u32 = 24;
    pub const AUGUR_CUDA_M306_K11_HC_EXACT_ACCUMULATORS_V1: u32 = 25;
    pub const AUGUR_CUDA_M306_K11_HC_EXACT_THREADS_V1: u32 = 32;
    
    
    
    pub const AUGUR_CUDA_M306_K11_HC_EXACT_FUNCTION_BYTES_V1: u64 = 1_572_864;
    
    
    
    
    
    
    
    
    
    
    pub const AUGUR_CUDA_M306_K23_HC_LANE_ARGS_VERSION_V1: u32 = 1;
    pub const AUGUR_CUDA_M306_K23_HC_LANE_HIDDEN_V1: u32 = 16_384;
    pub const AUGUR_CUDA_M306_K23_HC_LANE_OUTPUTS_V1: u32 = 24;
    pub const AUGUR_CUDA_M306_K23_HC_LANE_ACCUMULATORS_V1: u32 = 25;
    pub const AUGUR_CUDA_M306_K23_HC_LANE_THREADS_V1: u32 = 32;
    pub const AUGUR_CUDA_M306_K23_HC_LANE_FUNCTION_LAYOUT_SOURCE_MAJOR_V1: u32 = 1;
    
    
    pub const AUGUR_CUDA_M306_K23_HC_LANE_FUNCTION_BYTES_V1: u64 = 1_572_864;
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    pub const AUGUR_CUDA_M304_R3_GREEDY_RECEIPT_ARGS_VERSION_V1: u32 = 1;
    pub const AUGUR_CUDA_M304_R3_VOCAB_V1: u32 = 129_280;
    pub const AUGUR_CUDA_M304_R3_PARTIAL_COUNT_V1: u32 = 128;
    
    pub const AUGUR_CUDA_M304_R3_LOGITS_BYTES_V1: u64 = 517_120;
    pub const AUGUR_CUDA_M304_R3_PARTIAL_BYTES_V1: u64 = 4_096;
    pub const AUGUR_CUDA_M304_R3_RECEIPT_BYTES_V1: u64 = 64;
    
    
    
    
    
    #[cfg(feature = "qwen35-device-token-handoff")]
    pub const AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_ARGS_VERSION_V1: u32 = 1;
    #[cfg(feature = "qwen35-device-token-handoff")]
    pub const AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_VOCAB_V1: u32 = 248_320;
    #[cfg(feature = "qwen35-device-token-handoff")]
    pub const AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_PARTIAL_COUNT_V1: u32 = 256;
    #[cfg(feature = "qwen35-device-token-handoff")]
    pub const AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_THREADS_V1: u32 = 256;
    #[cfg(feature = "qwen35-device-token-handoff")]
    pub const AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_LOGITS_BYTES_V1: u64 = 993_280;
    #[cfg(feature = "qwen35-device-token-handoff")]
    pub const AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_PARTIAL_BYTES_V1: u64 = 8_192;
    #[cfg(feature = "qwen35-device-token-handoff")]
    pub const AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_RECEIPT_BYTES_V1: u64 = 72;
    #[cfg(feature = "qwen35-device-token-handoff")]
    pub const AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_STATUS_UNWRITTEN_V1: u32 = 0;
    #[cfg(feature = "qwen35-device-token-handoff")]
    pub const AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_STATUS_OK_V1: u32 = 1;
    #[cfg(feature = "qwen35-device-token-handoff")]
    pub const AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_STATUS_ERROR_V1: u32 = 2;
    #[cfg(feature = "qwen35-device-token-handoff")]
    pub const AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_ERROR_NONE_V1: u32 = 0;
    #[cfg(feature = "qwen35-device-token-handoff")]
    pub const AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_ERROR_NON_FINITE_LOGIT_V1: u32 = 1;
    #[cfg(feature = "qwen35-device-token-handoff")]
    pub const AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_ERROR_INCOMPLETE_REDUCTION_V1: u32 = 2;
    #[cfg(feature = "qwen35-device-token-handoff")]
    pub const AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_ARGS_VERSION_V1: u32 = 1;
    #[cfg(feature = "qwen35-device-token-handoff")]
    pub const AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_MAX_DRAFT_TOKENS_V1: u32 = 8;
    #[cfg(feature = "qwen35-device-token-handoff")]
    pub const AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_VOCAB_V1: u32 = 248_320;
    #[cfg(feature = "qwen35-device-token-handoff")]
    pub const AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_RECEIPT_BYTES_V1: u64 = 64;
    #[cfg(feature = "qwen35-device-token-handoff")]
    pub const AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_STATUS_UNWRITTEN_V1: u32 = 0;
    #[cfg(feature = "qwen35-device-token-handoff")]
    pub const AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_STATUS_OK_V1: u32 = 1;
    #[cfg(feature = "qwen35-device-token-handoff")]
    pub const AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_STATUS_ERROR_V1: u32 = 2;
    #[cfg(feature = "qwen35-device-token-handoff")]
    pub const AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_ERROR_NONE_V1: u32 = 0;
    #[cfg(feature = "qwen35-device-token-handoff")]
    pub const AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_ERROR_INVALID_TOKEN_V1: u32 = 1;
    #[cfg(feature = "qwen35-device-token-handoff")]
    pub const AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_ERROR_NON_FINITE_LOGIT_V1: u32 = 2;
    #[cfg(feature = "qwen35-device-token-handoff")]
    pub const AUGUR_CUDA_QWEN35_DFLASH_LOGITS_ACCEPTANCE_ARGS_VERSION_V1: u32 = 1;
    pub const AUGUR_CUDA_M304_R3_EXPERT_ROUTE_ARGS_VERSION_V1: u32 = 1;
    
    pub const AUGUR_CUDA_M304_R3_ROUTE_K_V1: u32 = 6;
    
    pub const AUGUR_CUDA_M304_R3_EXPERT_SLOT_TABLE_BYTES_V1: u64 = 6_144;
    pub const AUGUR_CUDA_M304_R3_ROUTE_STEP_RECEIPT_BYTES_V1: u64 = 1_440;
    pub const AUGUR_CUDA_M304_R3_AGGREGATION_ROUTE_RANK_V1: u32 = 0;
    
    
    
    
    pub const AUGUR_CUDA_M304_R3_ROUTE_ERROR_NONE_V1: u32 = 0;
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    pub const AUGUR_CUDA_M304_R3_EXPERT_GROUP2_ARGS_VERSION_V2: u32 = 2;
    pub const AUGUR_CUDA_M304_R3_EXPERT_GROUP2_ROWS_V2: u32 = 2;
    
    
    pub const AUGUR_CUDA_M304_R3_EXPERT_GROUP4_ARGS_VERSION_V3: u32 = 3;
    pub const AUGUR_CUDA_M304_R3_EXPERT_GROUP4_ROWS_V3: u32 = 4;
    
    
    
    pub const AUGUR_CUDA_M304_R3_EXPERT_GROUP4_TABLE_STATUS_UNWRITTEN_V3: u32 = 0;
    
    
    pub const AUGUR_CUDA_M304_R3_EXPERT_GROUP4_UNION_TABLE_BYTES_V3: u64 = 216;
    pub const AUGUR_CUDA_M306_K4_R2_EXPERT_ROUTE_ARGS_VERSION_V2: u32 = 2;
    pub const AUGUR_CUDA_M306_K4_R2_EXPERT_SLOT_VERSION_V2: u32 = 2;
    pub const AUGUR_CUDA_M306_K4_R2_Q4K_EXPERT_ROW32_CUDA_V2: u32 = 110;
    pub const AUGUR_CUDA_M306_K4_R2_Q5K_EXPERT_ROW32_CUDA_V2: u32 = 111;
    pub const AUGUR_CUDA_M306_K4_R2_PROJECTION_INPUT_V2: u32 = 1;
    pub const AUGUR_CUDA_M306_K4_R2_PROJECTION_DOWN_V2: u32 = 2;
    pub const AUGUR_CUDA_M306_K4_R2_SLOT_PROJECTION_W1_V2: u32 = 1;
    pub const AUGUR_CUDA_M306_K4_R2_SLOT_PROJECTION_W2_V2: u32 = 2;
    pub const AUGUR_CUDA_M306_K4_R2_SLOT_PROJECTION_W3_V2: u32 = 3;
    pub const AUGUR_CUDA_M306_K4_R2_EXPERTS_V2: u32 = 256;
    pub const AUGUR_CUDA_M306_K4_R2_ROUTE_K_V2: u32 = 6;
    pub const AUGUR_CUDA_M306_K4_R2_INPUT_ROWS_V2: u32 = 2_048;
    pub const AUGUR_CUDA_M306_K4_R2_INPUT_COLS_V2: u32 = 4_096;
    pub const AUGUR_CUDA_M306_K4_R2_INPUT_MATRICES_PER_EXPERT_V2: u32 = 2;
    pub const AUGUR_CUDA_M306_K4_R2_DOWN_ROWS_V2: u32 = 4_096;
    pub const AUGUR_CUDA_M306_K4_R2_DOWN_COLS_V2: u32 = 2_048;
    pub const AUGUR_CUDA_M306_K4_R2_DOWN_MATRICES_PER_EXPERT_V2: u32 = 1;
    pub const AUGUR_CUDA_M306_K4_R2_ROW_GROUP_V2: u32 = 32;
    
    pub const AUGUR_CUDA_M306_K4_R2_Q4_WEIGHT_BYTES_V2: u64 = 5_242_880;
    pub const AUGUR_CUDA_M306_K4_R2_Q5_WEIGHT_BYTES_V2: u64 = 6_291_456;
    pub const AUGUR_CUDA_M306_K4_R2_EXPERT_SLOT_TABLE_BYTES_V2: u64 = 22_528;
    
    
    
    
    
    
    
    
    
    
    
    
    
    pub const AUGUR_CUDA_M306_K4_R3_EXPERT_ROUTE_ARGS_VERSION_V3: u32 = 3;
    pub const AUGUR_CUDA_M306_K4_R3_EXPERT_SLOT_VERSION_V3: u32 = 3;
    pub const AUGUR_CUDA_M306_K4_R3_Q4K_HOT_CUDA_V1: u32 = 103;
    pub const AUGUR_CUDA_M306_K4_R3_Q5K_HOT_CUDA_V1: u32 = 105;
    pub const AUGUR_CUDA_M306_K4_R3_PROJECTION_INPUT_V3: u32 = 1;
    pub const AUGUR_CUDA_M306_K4_R3_PROJECTION_DOWN_V3: u32 = 2;
    pub const AUGUR_CUDA_M306_K4_R3_SLOT_PROJECTION_W1_V3: u32 = 1;
    pub const AUGUR_CUDA_M306_K4_R3_SLOT_PROJECTION_W2_V3: u32 = 2;
    pub const AUGUR_CUDA_M306_K4_R3_SLOT_PROJECTION_W3_V3: u32 = 3;
    pub const AUGUR_CUDA_M306_K4_R3_EXPERTS_V3: u32 = 256;
    pub const AUGUR_CUDA_M306_K4_R3_ROUTE_K_V3: u32 = 6;
    pub const AUGUR_CUDA_M306_K4_R3_INPUT_ROWS_V3: u32 = 2_048;
    pub const AUGUR_CUDA_M306_K4_R3_INPUT_COLS_V3: u32 = 4_096;
    pub const AUGUR_CUDA_M306_K4_R3_INPUT_MATRICES_PER_EXPERT_V3: u32 = 2;
    pub const AUGUR_CUDA_M306_K4_R3_DOWN_ROWS_V3: u32 = 4_096;
    pub const AUGUR_CUDA_M306_K4_R3_DOWN_COLS_V3: u32 = 2_048;
    pub const AUGUR_CUDA_M306_K4_R3_DOWN_MATRICES_PER_EXPERT_V3: u32 = 1;
    pub const AUGUR_CUDA_M306_K4_R3_BLOCK_THREADS_V3: u32 = 128;
    pub const AUGUR_CUDA_M306_K4_R3_ROWS_PER_BLOCK_V3: u32 = 4;
    pub const AUGUR_CUDA_M306_K4_R3_PARTIAL_PRECISION_BITS_V3: u32 = 64;
    pub const AUGUR_CUDA_M306_K4_R3_PRODUCT_PRECISION_BITS_V3: u32 = 32;
    pub const AUGUR_CUDA_M306_K4_R3_PRODUCT_FMA_POLICY_SEPARATE_MUL_RN_V3: u32 = 0;
    pub const AUGUR_CUDA_M306_K4_R3_AFFINE_FMA_POLICY_FMA_RN_V3: u32 = 1;
    pub const AUGUR_CUDA_M306_K4_R3_FTZ_POLICY_IEEE_V3: u32 = 0;
    pub const AUGUR_CUDA_M306_K4_R3_FINAL_CAST_F64_TO_F32_RN_BF16_RNE_V3: u32 = 1;
    
    pub const AUGUR_CUDA_M306_K4_R3_Q4_ROW_BYTES_V3: u32 = 2_560;
    pub const AUGUR_CUDA_M306_K4_R3_Q5_ROW_BYTES_V3: u32 = 1_536;
    pub const AUGUR_CUDA_M306_K4_R3_Q4_WEIGHT_BYTES_V3: u64 = 5_242_880;
    pub const AUGUR_CUDA_M306_K4_R3_Q5_WEIGHT_BYTES_V3: u64 = 6_291_456;
    pub const AUGUR_CUDA_M306_K4_R3_EXPERT_SLOT_TABLE_BYTES_V3: u64 = 22_528;
    pub const AUGUR_CUDA_M306_K4_R3_VALIDATION_RECORD_BYTES_V3: u64 = 288;
    
    
    
    
    
    
    
    
    
    
    pub const AUGUR_CUDA_M306_K4_R3_REDUCTION_TREE_OFFSETS_V3: [u8; 8] = [16, 8, 4, 2, 1, 0, 0, 0];
    pub const AUGUR_CUDA_M304_R3_COMBINED_COMMIT_ARGS_VERSION_V2: u32 = 2;
    pub const AUGUR_CUDA_M304_R3_COMBINED_COMMIT_BYTES_V2: u64 = 64;
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    pub const AUGUR_CUDA_M289_ROPE_POSITION_ARGS_VERSION_V1: u32 = 1;
    pub const AUGUR_CUDA_M289_INDEX_SCORE_ARGS_VERSION_V1: u32 = 1;
    pub const AUGUR_CUDA_M289_ATTENTION_ARGS_VERSION_V1: u32 = 1;
    pub const AUGUR_CUDA_M289_CAPACITY_V1: u32 = 128;
    pub const AUGUR_CUDA_M289_HEADS_V1: u32 = 64;
    pub const AUGUR_CUDA_M289_HEAD_DIM_V1: u32 = 512;
    pub const AUGUR_CUDA_M289_INDEX_HEAD_DIM_V1: u32 = 128;
    pub const AUGUR_CUDA_M289_VARIANT_RAW_V1: u32 = 0;
    pub const AUGUR_CUDA_M289_VARIANT_RATIO4_V1: u32 = 1;
    
    pub const AUGUR_CUDA_M289_RATIO_RAW_V1: u32 = 0;
    
    
    pub const AUGUR_CUDA_M289_INDEX_BASE_V1: u32 = 128;
    

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaQ4kF32DecodeArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,

        pub weight: AugurCudaDevicePtr,
        pub weight_len_bytes: u64,

        pub input: AugurCudaDevicePtr,
        pub input_len_bytes: u64,

        pub output: AugurCudaDevicePtr,
        pub output_len_bytes: u64,

        pub rows: u32,
        pub cols: u32,
        pub blocks_per_row: u32,
        pub weight_block_stride_bytes: u32,
    }

    impl AugurCudaQ4kF32DecodeArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_Q4K_F32_DECODE_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                weight: 0,
                weight_len_bytes: 0,
                input: 0,
                input_len_bytes: 0,
                output: 0,
                output_len_bytes: 0,
                rows: 0,
                cols: 0,
                blocks_per_row: 0,
                weight_block_stride_bytes: AUGUR_CUDA_Q4K_BLOCK_BYTES_V1,
            }
        }
    }

    impl Default for AugurCudaQ4kF32DecodeArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaQ4kF32PrefillArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub weight: AugurCudaDevicePtr,
        pub weight_len_bytes: u64,
        pub input: AugurCudaDevicePtr,
        pub input_len_bytes: u64,
        pub output: AugurCudaDevicePtr,
        pub output_len_bytes: u64,
        pub m: u32,
        pub rows: u32,
        pub cols: u32,
        pub blocks_per_row: u32,
        pub weight_block_stride_bytes: u32,
    }

    impl AugurCudaQ4kF32PrefillArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_Q4K_F32_PREFILL_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                weight: 0,
                weight_len_bytes: 0,
                input: 0,
                input_len_bytes: 0,
                output: 0,
                output_len_bytes: 0,
                m: 0,
                rows: 0,
                cols: 0,
                blocks_per_row: 0,
                weight_block_stride_bytes: AUGUR_CUDA_Q4K_BLOCK_BYTES_V1,
            }
        }
    }

    impl Default for AugurCudaQ4kF32PrefillArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    

    

    

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaQ5kF32DecodeArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub weight: AugurCudaDevicePtr,
        pub weight_len_bytes: u64,
        pub input: AugurCudaDevicePtr,
        pub input_len_bytes: u64,
        pub output: AugurCudaDevicePtr,
        pub output_len_bytes: u64,
        pub rows: u32,
        pub cols: u32,
        pub blocks_per_row: u32,
        pub weight_block_stride_bytes: u32,
    }
    impl AugurCudaQ5kF32DecodeArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_Q5K_F32_DECODE_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                weight: 0,
                weight_len_bytes: 0,
                input: 0,
                input_len_bytes: 0,
                output: 0,
                output_len_bytes: 0,
                rows: 0,
                cols: 0,
                blocks_per_row: 0,
                weight_block_stride_bytes: AUGUR_CUDA_Q5K_BLOCK_BYTES_V1,
            }
        }
    }
    impl Default for AugurCudaQ5kF32DecodeArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    

    

    

    
    
    

    
    
    

    
    
    

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaQ6kF32DecodeArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub weight: AugurCudaDevicePtr,
        pub weight_len_bytes: u64,
        pub input: AugurCudaDevicePtr,
        pub input_len_bytes: u64,
        pub output: AugurCudaDevicePtr,
        pub output_len_bytes: u64,
        pub rows: u32,
        pub cols: u32,
        pub blocks_per_row: u32,
        pub weight_block_stride_bytes: u32,
    }
    impl AugurCudaQ6kF32DecodeArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_Q6K_F32_DECODE_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                weight: 0,
                weight_len_bytes: 0,
                input: 0,
                input_len_bytes: 0,
                output: 0,
                output_len_bytes: 0,
                rows: 0,
                cols: 0,
                blocks_per_row: 0,
                weight_block_stride_bytes: AUGUR_CUDA_Q6K_BLOCK_BYTES_V1,
            }
        }
    }
    impl Default for AugurCudaQ6kF32DecodeArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    

    

    

    

    

    

    

    

    

    
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaM142OrderedTopkArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub scores: AugurCudaDevicePtr,
        pub scores_len_bytes: u64,
        pub indices: AugurCudaDevicePtr,
        pub indices_len_bytes: u64,
        pub selected_scores: AugurCudaDevicePtr,
        pub selected_scores_len_bytes: u64,
        pub tokens: u32,
        pub candidates: u32,
        pub top_k: u32,
        pub causal_ratio: u32,
        pub index_base: i32,
        pub padding: i32,
    }
    impl AugurCudaM142OrderedTopkArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_M142_ORDERED_TOPK_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                scores: 0,
                scores_len_bytes: 0,
                indices: 0,
                indices_len_bytes: 0,
                selected_scores: 0,
                selected_scores_len_bytes: 0,
                tokens: 0,
                candidates: 0,
                top_k: 0,
                causal_ratio: 0,
                index_base: 0,
                padding: -1,
            }
        }
    }
    impl Default for AugurCudaM142OrderedTopkArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    
    
    
    
    

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
    pub struct AugurCudaM283DeviceViewV1 {
        pub base: AugurCudaDevicePtr,
        pub allocation_extent_bytes: u64,
        pub view_offset_bytes: u64,
        pub view_extent_bytes: u64,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaM283KvTransitionArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub variant: u32,
        pub reserved0: u32,
        pub layer: u32,
        pub capacity: u32,
        pub position: u32,
        pub next_position: u32,
        pub compression_ratio: u32,
        pub source_start_position: u32,
        pub source_rows: u32,
        pub compressed_row_index: u32,
        pub compressed_position: u32,
        pub compressed_main_rows_before: u32,
        pub compressed_index_rows_before: u32,
        pub reserved1: u32,
        pub raw_current: AugurCudaM283DeviceViewV1,
        pub raw_cache: AugurCudaM283DeviceViewV1,
        pub compressed_main_current: AugurCudaM283DeviceViewV1,
        pub compressed_main_cache: AugurCudaM283DeviceViewV1,
        pub compressed_index_current: AugurCudaM283DeviceViewV1,
        pub compressed_index_cache: AugurCudaM283DeviceViewV1,
    }

    impl AugurCudaM283KvTransitionArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_M283_KV_TRANSITION_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                variant: AUGUR_CUDA_M283_KV_VARIANT_RAW_V1,
                reserved0: 0,
                layer: 0,
                capacity: AUGUR_CUDA_M283_KV_CAPACITY_V1,
                position: 0,
                next_position: 0,
                compression_ratio: 0,
                source_start_position: 0,
                source_rows: 1,
                compressed_row_index: 0,
                compressed_position: 0,
                compressed_main_rows_before: 0,
                compressed_index_rows_before: 0,
                reserved1: 0,
                raw_current: AugurCudaM283DeviceViewV1::default(),
                raw_cache: AugurCudaM283DeviceViewV1::default(),
                compressed_main_current: AugurCudaM283DeviceViewV1::default(),
                compressed_main_cache: AugurCudaM283DeviceViewV1::default(),
                compressed_index_current: AugurCudaM283DeviceViewV1::default(),
                compressed_index_cache: AugurCudaM283DeviceViewV1::default(),
            }
        }
    }

    impl Default for AugurCudaM283KvTransitionArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaM285FillSpanArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub allocation_base: AugurCudaDevicePtr,
        pub allocation_extent_bytes: u64,
        pub view_offset_bytes: u64,
        pub view_extent_bytes: u64,
        pub pattern_seed: u64,
        pub reserved: u64,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
    pub struct AugurCudaKv8DsaDeviceViewV1 {
        pub base: AugurCudaDevicePtr,
        pub allocation_extent_bytes: u64,
        pub view_offset_bytes: u64,
        pub view_extent_bytes: u64,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
    pub struct AugurCudaKv8DsaRowV1 {
        pub role: u32,
        pub values: u32,
        pub source: AugurCudaKv8DsaDeviceViewV1,
        pub destination: AugurCudaKv8DsaDeviceViewV1,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaKv8DsaQuantizeArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub row_count: u32,
        pub reset_status: u32,
        pub layer: u32,
        pub position: u32,
        pub reserved0: u32,
        pub reserved1: u32,
        pub status: AugurCudaKv8DsaDeviceViewV1,
        pub rows: [AugurCudaKv8DsaRowV1; AUGUR_CUDA_KV8_DSA_MAX_ROWS_V1],
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaKv8DsaQuantizeM128ArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub layer: u32,
        pub family: u32,
        pub absolute_start: u32,
        pub rows: u32,
        pub reset_status: u32,
        pub reserved0: u32,
        pub raw_source: AugurCudaKv8DsaDeviceViewV1,
        pub raw_destination: AugurCudaKv8DsaDeviceViewV1,
        pub compressed_main_source: AugurCudaKv8DsaDeviceViewV1,
        pub compressed_main_destination: AugurCudaKv8DsaDeviceViewV1,
        pub compressed_index_source: AugurCudaKv8DsaDeviceViewV1,
        pub compressed_index_destination: AugurCudaKv8DsaDeviceViewV1,
        pub status: AugurCudaKv8DsaDeviceViewV1,
    }

    impl AugurCudaKv8DsaQuantizeM128ArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_KV8_DSA_QUANTIZE_M128_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                layer: 0,
                family: AUGUR_CUDA_KV8_DSA_ATTENTION_FAMILY_RAW_V1,
                absolute_start: 0,
                rows: AUGUR_CUDA_KV8_DSA_QUANTIZE_M128_ROWS_V1,
                reset_status: 0,
                reserved0: 0,
                raw_source: Default::default(),
                raw_destination: Default::default(),
                compressed_main_source: Default::default(),
                compressed_main_destination: Default::default(),
                compressed_index_source: Default::default(),
                compressed_index_destination: Default::default(),
                status: Default::default(),
            }
        }
    }

    impl Default for AugurCudaKv8DsaQuantizeM128ArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    impl AugurCudaKv8DsaQuantizeArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_KV8_DSA_QUANTIZE_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                row_count: 0,
                reset_status: 0,
                layer: 0,
                position: 0,
                reserved0: 0,
                reserved1: 0,
                status: Default::default(),
                rows: [Default::default(); AUGUR_CUDA_KV8_DSA_MAX_ROWS_V1],
            }
        }
    }

    impl Default for AugurCudaKv8DsaQuantizeArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaKv8DsaStatusCopyArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub status: AugurCudaKv8DsaDeviceViewV1,
        pub host_ptr: u64,
        pub host_len_bytes: u64,
    }

    impl AugurCudaKv8DsaStatusCopyArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_KV8_DSA_STATUS_COPY_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                status: Default::default(),
                host_ptr: 0,
                host_len_bytes: 0,
            }
        }
    }

    impl Default for AugurCudaKv8DsaStatusCopyArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
    pub struct AugurCudaKv8DsaSelectReceiptV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub layer: u32,
        pub position: u32,
        pub candidate_count: u32,
        pub top_k: u32,
        pub selected_count: u32,
        pub index_base: u32,
        pub reservation_epoch: u64,
        pub padding: i32,
        pub reserved: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaKv8DsaSelectArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub layer: u32,
        pub position: u32,
        pub candidate_count: u32,
        pub top_k: u32,
        pub index_base: u32,
        pub padding: i32,
        pub reservation_epoch: u64,
        pub reserved0: u32,
        pub reserved1: u32,
        pub query: AugurCudaKv8DsaDeviceViewV1,
        pub index_rows: AugurCudaKv8DsaDeviceViewV1,
        pub head_weight: AugurCudaKv8DsaDeviceViewV1,
        pub scores: AugurCudaKv8DsaDeviceViewV1,
        pub partial_ids: AugurCudaKv8DsaDeviceViewV1,
        pub partial_scores: AugurCudaKv8DsaDeviceViewV1,
        pub selected_ids: AugurCudaKv8DsaDeviceViewV1,
        pub selected_scores: AugurCudaKv8DsaDeviceViewV1,
        pub receipt: AugurCudaKv8DsaDeviceViewV1,
        pub status: AugurCudaKv8DsaDeviceViewV1,
    }

    impl AugurCudaKv8DsaSelectArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_KV8_DSA_SELECT_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                layer: 0,
                position: 0,
                candidate_count: 0,
                top_k: AUGUR_CUDA_KV8_DSA_SELECT_TOP_K_V1,
                index_base: AUGUR_CUDA_KV8_DSA_SELECT_INDEX_BASE_V1,
                padding: AUGUR_CUDA_KV8_DSA_SELECT_PADDING_V1,
                reservation_epoch: 0,
                reserved0: 0,
                reserved1: 0,
                query: Default::default(),
                index_rows: Default::default(),
                head_weight: Default::default(),
                scores: Default::default(),
                partial_ids: Default::default(),
                partial_scores: Default::default(),
                selected_ids: Default::default(),
                selected_scores: Default::default(),
                receipt: Default::default(),
                status: Default::default(),
            }
        }
    }

    impl Default for AugurCudaKv8DsaSelectArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    

    

    

    

    

    

    

    

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaKv8DsaAttentionArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub layer: u32,
        pub position: u32,
        pub family: u32,
        pub raw_rows_visible: u32,
        pub raw_start: u32,
        pub compressed_rows_visible: u32,
        pub selected_count: u32,
        pub top_k: u32,
        pub index_base: i32,
        pub padding: i32,
        pub reservation_epoch: u64,
        pub reserved0: u32,
        pub reserved1: u32,
        pub query: AugurCudaKv8DsaDeviceViewV1,
        pub raw_main: AugurCudaKv8DsaDeviceViewV1,
        pub compressed_main: AugurCudaKv8DsaDeviceViewV1,
        pub selected_ids: AugurCudaKv8DsaDeviceViewV1,
        pub receipt: AugurCudaKv8DsaDeviceViewV1,
        pub sink: AugurCudaKv8DsaDeviceViewV1,
        pub weighted: AugurCudaKv8DsaDeviceViewV1,
        pub inverse: AugurCudaKv8DsaDeviceViewV1,
        pub status: AugurCudaKv8DsaDeviceViewV1,
    }

    impl AugurCudaKv8DsaAttentionArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_KV8_DSA_ATTENTION_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                layer: 0,
                position: 0,
                family: AUGUR_CUDA_KV8_DSA_ATTENTION_FAMILY_RAW_V1,
                raw_rows_visible: 1,
                raw_start: 0,
                compressed_rows_visible: 0,
                selected_count: 0,
                top_k: 0,
                index_base: AUGUR_CUDA_KV8_DSA_SELECT_INDEX_BASE_V1 as i32,
                padding: AUGUR_CUDA_KV8_DSA_SELECT_PADDING_V1,
                reservation_epoch: 0,
                reserved0: 0,
                reserved1: 0,
                query: Default::default(),
                raw_main: Default::default(),
                compressed_main: Default::default(),
                selected_ids: Default::default(),
                receipt: Default::default(),
                sink: Default::default(),
                weighted: Default::default(),
                inverse: Default::default(),
                status: Default::default(),
            }
        }
    }

    impl Default for AugurCudaKv8DsaAttentionArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaKv8DsaAttentionSplitArgsV2 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub base: AugurCudaKv8DsaAttentionArgsV1,
        pub workspace: AugurCudaKv8DsaDeviceViewV1,
        pub split_count: u32,
        pub reserved: u32,
    }

    

    

    

    impl AugurCudaKv8DsaAttentionSplitArgsV2 {
        pub fn new(
            base: AugurCudaKv8DsaAttentionArgsV1,
            workspace: AugurCudaKv8DsaDeviceViewV1,
            split_count: u32,
        ) -> Self {
            Self {
                abi_version: AUGUR_CUDA_KV8_DSA_ATTENTION_SPLIT_ARGS_VERSION_V2,
                struct_size: core::mem::size_of::<Self>() as u32,
                base,
                workspace,
                split_count,
                reserved: 0,
            }
        }
    }

    impl Default for AugurCudaKv8DsaAttentionSplitArgsV2 {
        fn default() -> Self {
            Self::new(Default::default(), Default::default(), 0)
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaM292HbmArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub operation: u32,
        pub blocks: u32,
        pub source_base: AugurCudaDevicePtr,
        pub source_allocation_extent_bytes: u64,
        pub source_view_offset_bytes: u64,
        pub source_view_extent_bytes: u64,
        pub destination_base: AugurCudaDevicePtr,
        pub destination_allocation_extent_bytes: u64,
        pub destination_view_offset_bytes: u64,
        pub destination_view_extent_bytes: u64,
        pub repeats: u32,
        pub reserved: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaM306K7HcFp64ArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub rows: u32,
        pub hidden_elements: u32,
        pub outputs: u32,
        pub chunks: u32,
        pub threads: u32,
        pub reserved: u32,
        pub hidden: AugurCudaDevicePtr,
        pub hidden_len_bytes: u64,
        pub function: AugurCudaDevicePtr,
        pub function_len_bytes: u64,
        pub mix: AugurCudaDevicePtr,
        pub mix_len_bytes: u64,
        pub scratch: AugurCudaDevicePtr,
        pub scratch_len_bytes: u64,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaM306K11HcExactArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub rows: u32,
        pub hidden_elements: u32,
        pub outputs: u32,
        pub accumulators: u32,
        pub threads: u32,
        pub reserved: u32,
        pub hidden: AugurCudaDevicePtr,
        pub hidden_len_bytes: u64,
        pub function: AugurCudaDevicePtr,
        pub function_len_bytes: u64,
        pub mix: AugurCudaDevicePtr,
        pub mix_len_bytes: u64,
        pub scratch: AugurCudaDevicePtr,
        pub scratch_len_bytes: u64,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaM306K23HcLaneArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub rows: u32,
        pub hidden_elements: u32,
        pub outputs: u32,
        pub accumulators: u32,
        pub threads: u32,
        pub function_layout: u32,
        pub hidden: AugurCudaDevicePtr,
        pub hidden_len_bytes: u64,
        pub function: AugurCudaDevicePtr,
        pub function_len_bytes: u64,
        pub mix: AugurCudaDevicePtr,
        pub mix_len_bytes: u64,
        pub scratch: AugurCudaDevicePtr,
        pub scratch_len_bytes: u64,
    }

    

    // C ABI tag: augur_cuda_m306_k12_learned_router_exact_args_v2
    

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
    pub struct AugurCudaM304R3GreedyReceiptV1 {
        pub token_id: u32,
        pub top1_value_bits: u32,
        pub runner_up_id: u32,
        pub runner_up_value_bits: u32,
        pub status: u32,
        pub error_code: u32,
        pub step_generation: u64,
        pub plan_digest: [u8; 32],
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaM304R3GreedyReceiptArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub logits: AugurCudaDevicePtr,
        pub logits_len_bytes: u64,
        pub partials: AugurCudaDevicePtr,
        pub partials_len_bytes: u64,
        pub receipt: AugurCudaDevicePtr,
        pub receipt_len_bytes: u64,
        pub step_generation: u64,
        pub vocab: u32,
        pub partial_count: u32,
        pub plan_digest: [u8; 32],
    }

    #[cfg(feature = "qwen35-device-token-handoff")]
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
    pub struct AugurCudaQwen35TokenHandoffReceiptV1 {
        pub token_id: u32,
        pub top1_value_bits: u32,
        pub runner_up_id: u32,
        pub runner_up_value_bits: u32,
        pub finite_count: u32,
        pub first_non_finite_id: u32,
        pub status: u32,
        pub error_code: u32,
        pub step_generation: u64,
        pub plan_digest: [u8; 32],
    }

    #[cfg(feature = "qwen35-device-token-handoff")]
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaQwen35TokenHandoffArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub logits: AugurCudaDevicePtr,
        pub logits_len_bytes: u64,
        pub partials: AugurCudaDevicePtr,
        pub partials_len_bytes: u64,
        pub receipt: AugurCudaDevicePtr,
        pub receipt_len_bytes: u64,
        pub step_generation: u64,
        pub vocab: u32,
        pub partial_count: u32,
        pub plan_digest: [u8; 32],
    }

    #[cfg(feature = "qwen35-device-token-handoff")]
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
    pub struct AugurCudaQwen35DflashAcceptanceReceiptV1 {
        pub draft_tokens: u32,
        pub accepted_tokens: u32,
        pub rejected_position: u32,
        pub replacement_token: u32,
        pub status: u32,
        pub error_code: u32,
        pub step_generation: u64,
        pub plan_digest: [u8; 32],
    }

    #[cfg(feature = "qwen35-device-token-handoff")]
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaQwen35DflashAcceptanceArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub draft_tokens: u32,
        pub reserved0: u32,
        pub draft_ids: AugurCudaDevicePtr,
        pub draft_ids_len_bytes: u64,
        pub target_ids: AugurCudaDevicePtr,
        pub target_ids_len_bytes: u64,
        pub receipt: AugurCudaDevicePtr,
        pub receipt_len_bytes: u64,
        pub step_generation: u64,
        pub plan_digest: [u8; 32],
    }

    #[cfg(feature = "qwen35-device-token-handoff")]
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaQwen35DflashLogitsAcceptanceArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub draft_tokens: u32,
        pub reserved0: u32,
        pub draft_ids: AugurCudaDevicePtr,
        pub draft_ids_len_bytes: u64,
        pub target_logits: AugurCudaDevicePtr,
        pub target_logits_len_bytes: u64,
        pub target_ids: AugurCudaDevicePtr,
        pub target_ids_len_bytes: u64,
        pub receipt: AugurCudaDevicePtr,
        pub receipt_len_bytes: u64,
        pub step_generation: u64,
        pub vocab: u32,
        pub reserved1: u32,
        pub plan_digest: [u8; 32],
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
    pub struct AugurCudaM304R3ExpertSlotV1 {
        pub w1: AugurCudaDevicePtr,
        pub w3: AugurCudaDevicePtr,
        pub w2: AugurCudaDevicePtr,
    }

    #[cfg(feature = "qwen35-device-token-handoff")]
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaQwen35VocabTop1ArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub rows: u32,
        pub vocab: u32,
        pub logits: AugurCudaDevicePtr,
        pub logits_len_bytes: u64,
        pub ids: AugurCudaDevicePtr,
        pub ids_len_bytes: u64,
    }

    #[cfg(feature = "qwen35-device-token-handoff")]
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaQwen35VocabTopkArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub rows: u32,
        pub vocab: u32,
        pub top_k: u32,
        pub reserved: u32,
        pub logits: AugurCudaDevicePtr,
        pub logits_len_bytes: u64,
        pub ids: AugurCudaDevicePtr,
        pub ids_len_bytes: u64,
        pub scores: AugurCudaDevicePtr,
        pub scores_len_bytes: u64,
    }

    #[cfg(feature = "qwen35-device-token-handoff")]
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaQwen35DflashSelectorArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub rows: u32,
        pub top_k: u32,
        pub rank: u32,
        pub hidden: u32,
        pub vocab: u32,
        pub anchor_token: u32,
        pub hidden_rows: AugurCudaDevicePtr,
        pub hidden_rows_len_bytes: u64,
        pub gate_weight: AugurCudaDevicePtr,
        pub gate_weight_len_bytes: u64,
        pub predecessor: AugurCudaDevicePtr,
        pub predecessor_len_bytes: u64,
        pub successor: AugurCudaDevicePtr,
        pub successor_len_bytes: u64,
        pub candidate_ids: AugurCudaDevicePtr,
        pub candidate_ids_len_bytes: u64,
        pub unary_scores: AugurCudaDevicePtr,
        pub unary_scores_len_bytes: u64,
        pub gates: AugurCudaDevicePtr,
        pub gates_len_bytes: u64,
        pub transition_scores: AugurCudaDevicePtr,
        pub transition_scores_len_bytes: u64,
        pub path_ids: AugurCudaDevicePtr,
        pub path_ids_len_bytes: u64,
    }

    #[cfg(feature = "qwen35-device-token-handoff")]
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaQwen35DflashQ8SelectorArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub rows: u32,
        pub top_k: u32,
        pub rank: u32,
        pub vocab: u32,
        pub anchor_token: u32,
        pub reserved: u32,
        pub predecessor: AugurCudaDevicePtr,
        pub predecessor_len_bytes: u64,
        pub successor: AugurCudaDevicePtr,
        pub successor_len_bytes: u64,
        pub candidate_ids: AugurCudaDevicePtr,
        pub candidate_ids_len_bytes: u64,
        pub unary_scores: AugurCudaDevicePtr,
        pub unary_scores_len_bytes: u64,
        pub gates: AugurCudaDevicePtr,
        pub gates_len_bytes: u64,
        pub transition_scores: AugurCudaDevicePtr,
        pub transition_scores_len_bytes: u64,
        pub path_ids: AugurCudaDevicePtr,
        pub path_ids_len_bytes: u64,
    }

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
    pub struct AugurCudaM304R3RouteLayerReceiptV1 {
        pub ids: [u16; 6],
        pub layer: u16,
        pub position: u16,
        pub status: u32,
        pub error_code: u32,
        pub selected_rank_mask: u64,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaM304R3RouteStepReceiptV1 {
        pub magic: u32,
        pub status: u32,
        pub first_error_layer: u32,
        pub error_code: u32,
        pub step_generation: u64,
        pub completed_layer_mask: u64,
        pub plan_digest: [u8; 32],
        pub layers: [AugurCudaM304R3RouteLayerReceiptV1; 43],
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaM304R3ExpertRouteArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub slot_table: AugurCudaDevicePtr,
        pub slot_table_len_bytes: u64,
        pub ids_u16: AugurCudaDevicePtr,
        pub ids_len_bytes: u64,
        pub router_status_i32: AugurCudaDevicePtr,
        pub router_status_len_bytes: u64,
        pub input_f32: AugurCudaDevicePtr,
        pub input_len_bytes: u64,
        pub output_f32: AugurCudaDevicePtr,
        pub output_len_bytes: u64,
        pub step_receipt: AugurCudaDevicePtr,
        pub step_receipt_len_bytes: u64,
        pub step_generation: u64,
        pub layer: u32,
        pub position: u32,
        pub aggregation_order: u32,
        pub reserved: u32,
        pub plan_digest: [u8; 32],
    }

    

    

    

    

    

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaM304R3ExpertGroup2ArgsV2 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub row_count: u32,
        pub projection: u32,
        pub group_plan_digest: [u8; 32],
        pub rows: [AugurCudaM304R3ExpertRouteArgsV1; 2],
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaM304R3ExpertGroup4UnionEntryV3 {
        pub expert: u16,
        pub row_mask: u8,
        pub reserved: u8,
        pub route_rank: [i8; 4],
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaM304R3ExpertGroup4UnionTableV3 {
        pub status: u32,
        pub count: u32,
        pub route_error: [u32; 4],
        pub entries: [AugurCudaM304R3ExpertGroup4UnionEntryV3; 24],
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaM304R3ExpertGroup4ArgsV3 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub row_count: u32,
        pub projection: u32,
        pub group_plan_digest: [u8; 32],
        pub union_table: AugurCudaDevicePtr,
        pub union_table_len_bytes: u64,
        pub rows: [AugurCudaM304R3ExpertRouteArgsV1; 4],
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaM306K4R2ExpertSlotV2 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub w1: AugurCudaDevicePtr,
        pub w1_len_bytes: u64,
        pub w3: AugurCudaDevicePtr,
        pub w3_len_bytes: u64,
        pub w2: AugurCudaDevicePtr,
        pub w2_len_bytes: u64,
        pub w1_layout: u32,
        pub w3_layout: u32,
        pub w2_layout: u32,
        pub w1_projection: u32,
        pub w3_projection: u32,
        pub w2_projection: u32,
        pub row_group: u32,
        pub reserved: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaM306K4R2ExpertRouteArgsV2 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub slot_table: AugurCudaDevicePtr,
        pub slot_table_len_bytes: u64,
        pub ids_u16: AugurCudaDevicePtr,
        pub ids_len_bytes: u64,
        pub router_status_i32: AugurCudaDevicePtr,
        pub router_status_len_bytes: u64,
        pub input_f32: AugurCudaDevicePtr,
        pub input_len_bytes: u64,
        pub output_f32: AugurCudaDevicePtr,
        pub output_len_bytes: u64,
        pub step_receipt: AugurCudaDevicePtr,
        pub step_receipt_len_bytes: u64,
        pub step_generation: u64,
        pub layer: u32,
        pub position: u32,
        pub aggregation_order: u32,
        pub projection: u32,
        pub slot_count: u32,
        pub route_k: u32,
        pub rows: u32,
        pub cols: u32,
        pub matrices_per_expert: u32,
        pub row_group: u32,
        pub input_layout: u32,
        pub down_layout: u32,
        pub slot_version: u32,
        pub slot_struct_size: u32,
        pub reserved: u32,
        pub plan_digest: [u8; 32],
    }

    

    

    

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaM306K4R3ExpertSlotV3 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub w1: AugurCudaDevicePtr,
        pub w1_len_bytes: u64,
        pub w3: AugurCudaDevicePtr,
        pub w3_len_bytes: u64,
        pub w2: AugurCudaDevicePtr,
        pub w2_len_bytes: u64,
        pub w1_layout: u32,
        pub w3_layout: u32,
        pub w2_layout: u32,
        pub w1_projection: u32,
        pub w3_projection: u32,
        pub w2_projection: u32,
        pub q4_row_bytes: u32,
        pub q5_row_bytes: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaM306K4R3ValidationRecordV3 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub magic: u32,
        pub status: u32,
        pub error_code: u32,
        pub projection: u32,
        pub route_k: u32,
        pub reserved0: u32,
        pub step_generation: u64,
        pub plan_digest: [u8; 32],
        pub ids: [u16; 6],
        pub reserved1: u32,
        pub w1: [AugurCudaDevicePtr; 6],
        pub w3: [AugurCudaDevicePtr; 6],
        pub w2: [AugurCudaDevicePtr; 6],
        pub input_layout: u32,
        pub down_layout: u32,
        pub slot_version: u32,
        pub partial_precision_bits: u32,
        pub block_threads: u32,
        pub rows_per_block: u32,
        pub reduction_tree_offsets: [u8; 8],
        pub product_precision_bits: u32,
        pub product_fma_policy: u32,
        pub affine_fma_policy: u32,
        pub ftz_policy: u32,
        pub final_cast_policy: u32,
        pub reserved2: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaM306K4R3ExpertRouteArgsV3 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub slot_table: AugurCudaDevicePtr,
        pub slot_table_len_bytes: u64,
        pub ids_u16: AugurCudaDevicePtr,
        pub ids_len_bytes: u64,
        pub router_status_i32: AugurCudaDevicePtr,
        pub router_status_len_bytes: u64,
        pub input_f32: AugurCudaDevicePtr,
        pub input_len_bytes: u64,
        pub output_f32: AugurCudaDevicePtr,
        pub output_len_bytes: u64,
        pub step_receipt: AugurCudaDevicePtr,
        pub step_receipt_len_bytes: u64,
        pub validation_record: AugurCudaDevicePtr,
        pub validation_record_len_bytes: u64,
        pub step_generation: u64,
        pub layer: u32,
        pub position: u32,
        pub aggregation_order: u32,
        pub projection: u32,
        pub slot_count: u32,
        pub route_k: u32,
        pub rows: u32,
        pub cols: u32,
        pub matrices_per_expert: u32,
        pub block_threads: u32,
        pub rows_per_block: u32,
        pub input_layout: u32,
        pub down_layout: u32,
        pub slot_version: u32,
        pub slot_struct_size: u32,
        pub partial_precision_bits: u32,
        pub product_precision_bits: u32,
        pub product_fma_policy: u32,
        pub affine_fma_policy: u32,
        pub ftz_policy: u32,
        pub final_cast_policy: u32,
        pub reduction_tree_offsets: [u8; 8],
        pub reserved: u32,
        pub plan_digest: [u8; 32],
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
    pub struct AugurCudaM304R3CombinedCommitV2 {
        pub token_id: u32,
        pub top1_value_bits: u32,
        pub runner_up_id: u32,
        pub runner_up_value_bits: u32,
        pub status: u32,
        pub error_code: u32,
        pub step_generation: u64,
        pub plan_digest: [u8; 32],
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaM304R3CombinedCommitArgsV2 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub logits: AugurCudaDevicePtr,
        pub logits_len_bytes: u64,
        pub route_step_receipt: AugurCudaDevicePtr,
        pub route_step_receipt_len_bytes: u64,
        pub partials: AugurCudaDevicePtr,
        pub partials_len_bytes: u64,
        pub commit: AugurCudaDevicePtr,
        pub commit_len_bytes: u64,
        pub step_generation: u64,
        pub vocab: u32,
        pub partial_count: u32,
        pub expected_position: u32,
        pub reserved: u32,
        pub plan_digest: [u8; 32],
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
    pub struct AugurCudaM289DeviceViewV1 {
        pub base: AugurCudaDevicePtr,
        pub allocation_extent_bytes: u64,
        pub view_offset_bytes: u64,
        pub view_extent_bytes: u64,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaM289RopePositionArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub layer: u32,
        pub capacity: u32,
        pub position: u32,
        pub rows: u32,
        pub head_dim: u32,
        pub rotary_dim: u32,
        pub repeat: u32,
        pub flags: u32,
        pub reserved0: u32,
        pub reserved1: u32,
        pub reserved2: [u64; 2],
        pub input: AugurCudaM289DeviceViewV1,
        pub output: AugurCudaM289DeviceViewV1,
    }
    impl AugurCudaM289RopePositionArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_M289_ROPE_POSITION_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                layer: 0,
                capacity: AUGUR_CUDA_M289_CAPACITY_V1,
                position: 0,
                rows: 1,
                head_dim: AUGUR_CUDA_M289_HEAD_DIM_V1,
                rotary_dim: 64,
                repeat: 1,
                flags: 0,
                reserved0: 0,
                reserved1: 0,
                reserved2: [0; 2],
                input: Default::default(),
                output: Default::default(),
            }
        }
    }
    impl Default for AugurCudaM289RopePositionArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaM289IndexScoreArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub layer: u32,
        pub capacity: u32,
        pub position: u32,
        pub candidate_rows: u32,
        pub top_k: u32,
        pub heads: u32,
        pub head_dim: u32,
        pub index_base: u32,
        pub variant: u32,
        pub flags: u32,
        pub reserved0: u32,
        pub reserved1: u64,
        pub query: AugurCudaM289DeviceViewV1,
        pub compressed: AugurCudaM289DeviceViewV1,
        pub head_weight: AugurCudaM289DeviceViewV1,
        pub scores: AugurCudaM289DeviceViewV1,
    }
    impl AugurCudaM289IndexScoreArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_M289_INDEX_SCORE_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                layer: 0,
                capacity: AUGUR_CUDA_M289_CAPACITY_V1,
                position: 0,
                candidate_rows: 0,
                top_k: 10,
                heads: AUGUR_CUDA_M289_HEADS_V1,
                head_dim: AUGUR_CUDA_M289_INDEX_HEAD_DIM_V1,
                index_base: AUGUR_CUDA_M289_INDEX_BASE_V1,
                variant: AUGUR_CUDA_M289_VARIANT_RATIO4_V1,
                flags: 0,
                reserved0: 0,
                reserved1: 0,
                query: Default::default(),
                compressed: Default::default(),
                head_weight: Default::default(),
                scores: Default::default(),
            }
        }
    }
    impl Default for AugurCudaM289IndexScoreArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaM289AttentionArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub layer: u32,
        pub capacity: u32,
        pub position: u32,
        pub query_position: u32,
        pub variant: u32,
        pub compression_ratio: u32,
        pub raw_rows_visible: u32,
        pub compressed_rows_visible: u32,
        pub ids_row_index: u32,
        pub top_k: u32,
        pub heads: u32,
        pub head_dim: u32,
        pub flags: u32,
        pub reserved0: u32,
        pub query: AugurCudaM289DeviceViewV1,
        pub raw: AugurCudaM289DeviceViewV1,
        pub compressed: AugurCudaM289DeviceViewV1,
        pub ids: AugurCudaM289DeviceViewV1,
        pub sink: AugurCudaM289DeviceViewV1,
        pub weighted: AugurCudaM289DeviceViewV1,
        pub inverse: AugurCudaM289DeviceViewV1,
    }
    impl AugurCudaM289AttentionArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_M289_ATTENTION_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                layer: 0,
                capacity: AUGUR_CUDA_M289_CAPACITY_V1,
                position: 0,
                query_position: 0,
                variant: AUGUR_CUDA_M289_VARIANT_RAW_V1,
                compression_ratio: AUGUR_CUDA_M289_RATIO_RAW_V1,
                raw_rows_visible: 1,
                compressed_rows_visible: 0,
                ids_row_index: 0,
                top_k: 0,
                heads: AUGUR_CUDA_M289_HEADS_V1,
                head_dim: AUGUR_CUDA_M289_HEAD_DIM_V1,
                flags: 0,
                reserved0: 0,
                query: Default::default(),
                raw: Default::default(),
                compressed: Default::default(),
                ids: Default::default(),
                sink: Default::default(),
                weighted: Default::default(),
                inverse: Default::default(),
            }
        }
    }
    impl Default for AugurCudaM289AttentionArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    impl AugurCudaM285FillSpanArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_M285_FILL_SPAN_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                allocation_base: 0,
                allocation_extent_bytes: 0,
                view_offset_bytes: 0,
                view_extent_bytes: 0,
                pattern_seed: 0,
                reserved: 0,
            }
        }
    }

    impl Default for AugurCudaM285FillSpanArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    impl AugurCudaM292HbmArgsV1 {
        pub const fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_M292_HBM_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                operation: 0,
                blocks: 0,
                source_base: 0,
                source_allocation_extent_bytes: 0,
                source_view_offset_bytes: 0,
                source_view_extent_bytes: 0,
                destination_base: 0,
                destination_allocation_extent_bytes: 0,
                destination_view_offset_bytes: 0,
                destination_view_extent_bytes: 0,
                repeats: 0,
                reserved: 0,
            }
        }
    }

    impl Default for AugurCudaM292HbmArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    impl AugurCudaM306K7HcFp64ArgsV1 {
        pub const fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_M306_K7_HC_FP64_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                rows: 0,
                hidden_elements: AUGUR_CUDA_M306_K7_HC_HIDDEN_V1,
                outputs: AUGUR_CUDA_M306_K7_HC_OUTPUTS_V1,
                chunks: AUGUR_CUDA_M306_K7_HC_CHUNKS_V1,
                threads: AUGUR_CUDA_M306_K7_HC_THREADS_V1,
                reserved: 0,
                hidden: 0,
                hidden_len_bytes: 0,
                function: 0,
                function_len_bytes: AUGUR_CUDA_M306_K7_HC_FUNCTION_BYTES_V1,
                mix: 0,
                mix_len_bytes: 0,
                scratch: 0,
                scratch_len_bytes: 0,
            }
        }
    }

    impl Default for AugurCudaM306K7HcFp64ArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    impl AugurCudaM306K11HcExactArgsV1 {
        pub const fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_M306_K11_HC_EXACT_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                rows: 0,
                hidden_elements: AUGUR_CUDA_M306_K11_HC_EXACT_HIDDEN_V1,
                outputs: AUGUR_CUDA_M306_K11_HC_EXACT_OUTPUTS_V1,
                accumulators: AUGUR_CUDA_M306_K11_HC_EXACT_ACCUMULATORS_V1,
                threads: AUGUR_CUDA_M306_K11_HC_EXACT_THREADS_V1,
                reserved: 0,
                hidden: 0,
                hidden_len_bytes: 0,
                function: 0,
                function_len_bytes: AUGUR_CUDA_M306_K11_HC_EXACT_FUNCTION_BYTES_V1,
                mix: 0,
                mix_len_bytes: 0,
                scratch: 0,
                scratch_len_bytes: 0,
            }
        }
    }

    impl Default for AugurCudaM306K11HcExactArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    impl AugurCudaM306K23HcLaneArgsV1 {
        pub const fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_M306_K23_HC_LANE_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                rows: 0,
                hidden_elements: AUGUR_CUDA_M306_K23_HC_LANE_HIDDEN_V1,
                outputs: AUGUR_CUDA_M306_K23_HC_LANE_OUTPUTS_V1,
                accumulators: AUGUR_CUDA_M306_K23_HC_LANE_ACCUMULATORS_V1,
                threads: AUGUR_CUDA_M306_K23_HC_LANE_THREADS_V1,
                function_layout: AUGUR_CUDA_M306_K23_HC_LANE_FUNCTION_LAYOUT_SOURCE_MAJOR_V1,
                hidden: 0,
                hidden_len_bytes: 0,
                function: 0,
                function_len_bytes: AUGUR_CUDA_M306_K23_HC_LANE_FUNCTION_BYTES_V1,
                mix: 0,
                mix_len_bytes: 0,
                scratch: 0,
                scratch_len_bytes: 0,
            }
        }
    }

    impl Default for AugurCudaM306K23HcLaneArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    

    

    

    

    impl AugurCudaM304R3GreedyReceiptArgsV1 {
        pub const fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_M304_R3_GREEDY_RECEIPT_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                logits: 0,
                logits_len_bytes: AUGUR_CUDA_M304_R3_LOGITS_BYTES_V1,
                partials: 0,
                partials_len_bytes: AUGUR_CUDA_M304_R3_PARTIAL_BYTES_V1,
                receipt: 0,
                receipt_len_bytes: AUGUR_CUDA_M304_R3_RECEIPT_BYTES_V1,
                step_generation: 0,
                vocab: AUGUR_CUDA_M304_R3_VOCAB_V1,
                partial_count: AUGUR_CUDA_M304_R3_PARTIAL_COUNT_V1,
                plan_digest: [0; 32],
            }
        }
    }

    impl Default for AugurCudaM304R3GreedyReceiptArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    #[cfg(feature = "qwen35-device-token-handoff")]
    impl AugurCudaQwen35TokenHandoffArgsV1 {
        pub const fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                logits: 0,
                logits_len_bytes: AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_LOGITS_BYTES_V1,
                partials: 0,
                partials_len_bytes: AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_PARTIAL_BYTES_V1,
                receipt: 0,
                receipt_len_bytes: AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_RECEIPT_BYTES_V1,
                step_generation: 0,
                vocab: AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_VOCAB_V1,
                partial_count: AUGUR_CUDA_QWEN35_TOKEN_HANDOFF_PARTIAL_COUNT_V1,
                plan_digest: [0; 32],
            }
        }
    }

    #[cfg(feature = "qwen35-device-token-handoff")]
    impl Default for AugurCudaQwen35TokenHandoffArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    #[cfg(feature = "qwen35-device-token-handoff")]
    impl AugurCudaQwen35DflashAcceptanceArgsV1 {
        pub const fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                draft_tokens: 0,
                reserved0: 0,
                draft_ids: 0,
                draft_ids_len_bytes: 0,
                target_ids: 0,
                target_ids_len_bytes: 0,
                receipt: 0,
                receipt_len_bytes: AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_RECEIPT_BYTES_V1,
                step_generation: 0,
                plan_digest: [0; 32],
            }
        }
    }

    #[cfg(feature = "qwen35-device-token-handoff")]
    impl Default for AugurCudaQwen35DflashAcceptanceArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    #[cfg(feature = "qwen35-device-token-handoff")]
    impl AugurCudaQwen35DflashLogitsAcceptanceArgsV1 {
        pub const fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_QWEN35_DFLASH_LOGITS_ACCEPTANCE_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                draft_tokens: 0,
                reserved0: 0,
                draft_ids: 0,
                draft_ids_len_bytes: 0,
                target_logits: 0,
                target_logits_len_bytes: 0,
                target_ids: 0,
                target_ids_len_bytes: 0,
                receipt: 0,
                receipt_len_bytes: AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_RECEIPT_BYTES_V1,
                step_generation: 0,
                vocab: AUGUR_CUDA_QWEN35_DFLASH_ACCEPTANCE_VOCAB_V1,
                reserved1: 0,
                plan_digest: [0; 32],
            }
        }
    }

    #[cfg(feature = "qwen35-device-token-handoff")]
    impl Default for AugurCudaQwen35DflashLogitsAcceptanceArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    

    

    impl AugurCudaM304R3ExpertRouteArgsV1 {
        pub const fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_M304_R3_EXPERT_ROUTE_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                slot_table: 0,
                slot_table_len_bytes: AUGUR_CUDA_M304_R3_EXPERT_SLOT_TABLE_BYTES_V1,
                ids_u16: 0,
                ids_len_bytes: (AUGUR_CUDA_M304_R3_ROUTE_K_V1 * 2) as u64,
                router_status_i32: 0,
                router_status_len_bytes: 4,
                input_f32: 0,
                input_len_bytes: 0,
                output_f32: 0,
                output_len_bytes: 0,
                step_receipt: 0,
                step_receipt_len_bytes: AUGUR_CUDA_M304_R3_ROUTE_STEP_RECEIPT_BYTES_V1,
                step_generation: 0,
                layer: 0,
                position: 0,
                aggregation_order: AUGUR_CUDA_M304_R3_AGGREGATION_ROUTE_RANK_V1,
                reserved: 0,
                plan_digest: [0; 32],
            }
        }
    }

    impl Default for AugurCudaM304R3ExpertRouteArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    impl AugurCudaM304R3ExpertGroup2ArgsV2 {
        pub const fn new(projection: u32) -> Self {
            Self {
                abi_version: AUGUR_CUDA_M304_R3_EXPERT_GROUP2_ARGS_VERSION_V2,
                struct_size: core::mem::size_of::<Self>() as u32,
                row_count: AUGUR_CUDA_M304_R3_EXPERT_GROUP2_ROWS_V2,
                projection,
                group_plan_digest: [0; 32],
                rows: [AugurCudaM304R3ExpertRouteArgsV1::new(); 2],
            }
        }
    }

    impl AugurCudaM304R3ExpertGroup4UnionEntryV3 {
        pub const fn new() -> Self {
            Self {
                expert: 0,
                row_mask: 0,
                reserved: 0,
                route_rank: [-1; 4],
            }
        }
    }

    impl Default for AugurCudaM304R3ExpertGroup4UnionEntryV3 {
        fn default() -> Self {
            Self::new()
        }
    }

    impl AugurCudaM304R3ExpertGroup4UnionTableV3 {
        pub const fn new() -> Self {
            Self {
                status: AUGUR_CUDA_M304_R3_EXPERT_GROUP4_TABLE_STATUS_UNWRITTEN_V3,
                count: 0,
                route_error: [AUGUR_CUDA_M304_R3_ROUTE_ERROR_NONE_V1; 4],
                entries: [AugurCudaM304R3ExpertGroup4UnionEntryV3::new(); 24],
            }
        }
    }

    impl Default for AugurCudaM304R3ExpertGroup4UnionTableV3 {
        fn default() -> Self {
            Self::new()
        }
    }

    impl AugurCudaM304R3ExpertGroup4ArgsV3 {
        pub const fn new(projection: u32) -> Self {
            Self {
                abi_version: AUGUR_CUDA_M304_R3_EXPERT_GROUP4_ARGS_VERSION_V3,
                struct_size: core::mem::size_of::<Self>() as u32,
                row_count: AUGUR_CUDA_M304_R3_EXPERT_GROUP4_ROWS_V3,
                projection,
                group_plan_digest: [0; 32],
                union_table: 0,
                union_table_len_bytes: AUGUR_CUDA_M304_R3_EXPERT_GROUP4_UNION_TABLE_BYTES_V3,
                rows: [AugurCudaM304R3ExpertRouteArgsV1::new(); 4],
            }
        }
    }

    impl AugurCudaM306K4R2ExpertSlotV2 {
        pub const fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_M306_K4_R2_EXPERT_SLOT_VERSION_V2,
                struct_size: core::mem::size_of::<Self>() as u32,
                w1: 0,
                w1_len_bytes: AUGUR_CUDA_M306_K4_R2_Q4_WEIGHT_BYTES_V2,
                w3: 0,
                w3_len_bytes: AUGUR_CUDA_M306_K4_R2_Q4_WEIGHT_BYTES_V2,
                w2: 0,
                w2_len_bytes: AUGUR_CUDA_M306_K4_R2_Q5_WEIGHT_BYTES_V2,
                w1_layout: AUGUR_CUDA_M306_K4_R2_Q4K_EXPERT_ROW32_CUDA_V2,
                w3_layout: AUGUR_CUDA_M306_K4_R2_Q4K_EXPERT_ROW32_CUDA_V2,
                w2_layout: AUGUR_CUDA_M306_K4_R2_Q5K_EXPERT_ROW32_CUDA_V2,
                w1_projection: AUGUR_CUDA_M306_K4_R2_SLOT_PROJECTION_W1_V2,
                w3_projection: AUGUR_CUDA_M306_K4_R2_SLOT_PROJECTION_W3_V2,
                w2_projection: AUGUR_CUDA_M306_K4_R2_SLOT_PROJECTION_W2_V2,
                row_group: AUGUR_CUDA_M306_K4_R2_ROW_GROUP_V2,
                reserved: 0,
            }
        }
    }

    impl Default for AugurCudaM306K4R2ExpertSlotV2 {
        fn default() -> Self {
            Self::new()
        }
    }

    impl AugurCudaM306K4R2ExpertRouteArgsV2 {
        const fn new_for(
            projection: u32,
            rows: u32,
            cols: u32,
            matrices_per_expert: u32,
            input_len_bytes: u64,
            output_len_bytes: u64,
        ) -> Self {
            Self {
                abi_version: AUGUR_CUDA_M306_K4_R2_EXPERT_ROUTE_ARGS_VERSION_V2,
                struct_size: core::mem::size_of::<Self>() as u32,
                slot_table: 0,
                slot_table_len_bytes: AUGUR_CUDA_M306_K4_R2_EXPERT_SLOT_TABLE_BYTES_V2,
                ids_u16: 0,
                ids_len_bytes: (AUGUR_CUDA_M306_K4_R2_ROUTE_K_V2 * 2) as u64,
                router_status_i32: 0,
                router_status_len_bytes: 4,
                input_f32: 0,
                input_len_bytes,
                output_f32: 0,
                output_len_bytes,
                step_receipt: 0,
                step_receipt_len_bytes: AUGUR_CUDA_M304_R3_ROUTE_STEP_RECEIPT_BYTES_V1,
                step_generation: 0,
                layer: 0,
                position: 0,
                aggregation_order: AUGUR_CUDA_M304_R3_AGGREGATION_ROUTE_RANK_V1,
                projection,
                slot_count: AUGUR_CUDA_M306_K4_R2_EXPERTS_V2,
                route_k: AUGUR_CUDA_M306_K4_R2_ROUTE_K_V2,
                rows,
                cols,
                matrices_per_expert,
                row_group: AUGUR_CUDA_M306_K4_R2_ROW_GROUP_V2,
                input_layout: AUGUR_CUDA_M306_K4_R2_Q4K_EXPERT_ROW32_CUDA_V2,
                down_layout: AUGUR_CUDA_M306_K4_R2_Q5K_EXPERT_ROW32_CUDA_V2,
                slot_version: AUGUR_CUDA_M306_K4_R2_EXPERT_SLOT_VERSION_V2,
                slot_struct_size: core::mem::size_of::<AugurCudaM306K4R2ExpertSlotV2>() as u32,
                reserved: 0,
                plan_digest: [0; 32],
            }
        }

        pub const fn new_input() -> Self {
            Self::new_for(
                AUGUR_CUDA_M306_K4_R2_PROJECTION_INPUT_V2,
                AUGUR_CUDA_M306_K4_R2_INPUT_ROWS_V2,
                AUGUR_CUDA_M306_K4_R2_INPUT_COLS_V2,
                AUGUR_CUDA_M306_K4_R2_INPUT_MATRICES_PER_EXPERT_V2,
                (AUGUR_CUDA_M306_K4_R2_INPUT_COLS_V2 * 4) as u64,
                (AUGUR_CUDA_M306_K4_R2_ROUTE_K_V2
                    * AUGUR_CUDA_M306_K4_R2_INPUT_MATRICES_PER_EXPERT_V2
                    * AUGUR_CUDA_M306_K4_R2_INPUT_ROWS_V2
                    * 4) as u64,
            )
        }

        pub const fn new_down() -> Self {
            Self::new_for(
                AUGUR_CUDA_M306_K4_R2_PROJECTION_DOWN_V2,
                AUGUR_CUDA_M306_K4_R2_DOWN_ROWS_V2,
                AUGUR_CUDA_M306_K4_R2_DOWN_COLS_V2,
                AUGUR_CUDA_M306_K4_R2_DOWN_MATRICES_PER_EXPERT_V2,
                (AUGUR_CUDA_M306_K4_R2_ROUTE_K_V2
                    * AUGUR_CUDA_M306_K4_R2_INPUT_MATRICES_PER_EXPERT_V2
                    * AUGUR_CUDA_M306_K4_R2_INPUT_ROWS_V2
                    * 4) as u64,
                (AUGUR_CUDA_M306_K4_R2_ROUTE_K_V2 * AUGUR_CUDA_M306_K4_R2_DOWN_ROWS_V2 * 4) as u64,
            )
        }
    }

    

    

    

    

    

    impl AugurCudaM306K4R3ExpertSlotV3 {
        pub const fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_M306_K4_R3_EXPERT_SLOT_VERSION_V3,
                struct_size: core::mem::size_of::<Self>() as u32,
                w1: 0,
                w1_len_bytes: AUGUR_CUDA_M306_K4_R3_Q4_WEIGHT_BYTES_V3,
                w3: 0,
                w3_len_bytes: AUGUR_CUDA_M306_K4_R3_Q4_WEIGHT_BYTES_V3,
                w2: 0,
                w2_len_bytes: AUGUR_CUDA_M306_K4_R3_Q5_WEIGHT_BYTES_V3,
                w1_layout: AUGUR_CUDA_M306_K4_R3_Q4K_HOT_CUDA_V1,
                w3_layout: AUGUR_CUDA_M306_K4_R3_Q4K_HOT_CUDA_V1,
                w2_layout: AUGUR_CUDA_M306_K4_R3_Q5K_HOT_CUDA_V1,
                w1_projection: AUGUR_CUDA_M306_K4_R3_SLOT_PROJECTION_W1_V3,
                w3_projection: AUGUR_CUDA_M306_K4_R3_SLOT_PROJECTION_W3_V3,
                w2_projection: AUGUR_CUDA_M306_K4_R3_SLOT_PROJECTION_W2_V3,
                q4_row_bytes: AUGUR_CUDA_M306_K4_R3_Q4_ROW_BYTES_V3,
                q5_row_bytes: AUGUR_CUDA_M306_K4_R3_Q5_ROW_BYTES_V3,
            }
        }
    }

    impl Default for AugurCudaM306K4R3ExpertSlotV3 {
        fn default() -> Self {
            Self::new()
        }
    }

    impl AugurCudaM306K4R3ExpertRouteArgsV3 {
        const fn new_for(
            projection: u32,
            rows: u32,
            cols: u32,
            matrices_per_expert: u32,
            input_len_bytes: u64,
            output_len_bytes: u64,
        ) -> Self {
            Self {
                abi_version: AUGUR_CUDA_M306_K4_R3_EXPERT_ROUTE_ARGS_VERSION_V3,
                struct_size: core::mem::size_of::<Self>() as u32,
                slot_table: 0,
                slot_table_len_bytes: AUGUR_CUDA_M306_K4_R3_EXPERT_SLOT_TABLE_BYTES_V3,
                ids_u16: 0,
                ids_len_bytes: (AUGUR_CUDA_M306_K4_R3_ROUTE_K_V3 * 2) as u64,
                router_status_i32: 0,
                router_status_len_bytes: 4,
                input_f32: 0,
                input_len_bytes,
                output_f32: 0,
                output_len_bytes,
                step_receipt: 0,
                step_receipt_len_bytes: AUGUR_CUDA_M304_R3_ROUTE_STEP_RECEIPT_BYTES_V1,
                validation_record: 0,
                validation_record_len_bytes: AUGUR_CUDA_M306_K4_R3_VALIDATION_RECORD_BYTES_V3,
                step_generation: 0,
                layer: 0,
                position: 0,
                aggregation_order: AUGUR_CUDA_M304_R3_AGGREGATION_ROUTE_RANK_V1,
                projection,
                slot_count: AUGUR_CUDA_M306_K4_R3_EXPERTS_V3,
                route_k: AUGUR_CUDA_M306_K4_R3_ROUTE_K_V3,
                rows,
                cols,
                matrices_per_expert,
                block_threads: AUGUR_CUDA_M306_K4_R3_BLOCK_THREADS_V3,
                rows_per_block: AUGUR_CUDA_M306_K4_R3_ROWS_PER_BLOCK_V3,
                input_layout: AUGUR_CUDA_M306_K4_R3_Q4K_HOT_CUDA_V1,
                down_layout: AUGUR_CUDA_M306_K4_R3_Q5K_HOT_CUDA_V1,
                slot_version: AUGUR_CUDA_M306_K4_R3_EXPERT_SLOT_VERSION_V3,
                slot_struct_size: core::mem::size_of::<AugurCudaM306K4R3ExpertSlotV3>() as u32,
                partial_precision_bits: AUGUR_CUDA_M306_K4_R3_PARTIAL_PRECISION_BITS_V3,
                product_precision_bits: AUGUR_CUDA_M306_K4_R3_PRODUCT_PRECISION_BITS_V3,
                product_fma_policy: AUGUR_CUDA_M306_K4_R3_PRODUCT_FMA_POLICY_SEPARATE_MUL_RN_V3,
                affine_fma_policy: AUGUR_CUDA_M306_K4_R3_AFFINE_FMA_POLICY_FMA_RN_V3,
                ftz_policy: AUGUR_CUDA_M306_K4_R3_FTZ_POLICY_IEEE_V3,
                final_cast_policy: AUGUR_CUDA_M306_K4_R3_FINAL_CAST_F64_TO_F32_RN_BF16_RNE_V3,
                reduction_tree_offsets: AUGUR_CUDA_M306_K4_R3_REDUCTION_TREE_OFFSETS_V3,
                reserved: 0,
                plan_digest: [0; 32],
            }
        }

        pub const fn new_input() -> Self {
            Self::new_for(
                AUGUR_CUDA_M306_K4_R3_PROJECTION_INPUT_V3,
                AUGUR_CUDA_M306_K4_R3_INPUT_ROWS_V3,
                AUGUR_CUDA_M306_K4_R3_INPUT_COLS_V3,
                AUGUR_CUDA_M306_K4_R3_INPUT_MATRICES_PER_EXPERT_V3,
                (AUGUR_CUDA_M306_K4_R3_INPUT_COLS_V3 * 4) as u64,
                (AUGUR_CUDA_M306_K4_R3_ROUTE_K_V3
                    * AUGUR_CUDA_M306_K4_R3_INPUT_MATRICES_PER_EXPERT_V3
                    * AUGUR_CUDA_M306_K4_R3_INPUT_ROWS_V3
                    * 4) as u64,
            )
        }

        pub const fn new_down() -> Self {
            Self::new_for(
                AUGUR_CUDA_M306_K4_R3_PROJECTION_DOWN_V3,
                AUGUR_CUDA_M306_K4_R3_DOWN_ROWS_V3,
                AUGUR_CUDA_M306_K4_R3_DOWN_COLS_V3,
                AUGUR_CUDA_M306_K4_R3_DOWN_MATRICES_PER_EXPERT_V3,
                (AUGUR_CUDA_M306_K4_R3_ROUTE_K_V3
                    * AUGUR_CUDA_M306_K4_R3_INPUT_MATRICES_PER_EXPERT_V3
                    * AUGUR_CUDA_M306_K4_R3_INPUT_ROWS_V3
                    * 4) as u64,
                (AUGUR_CUDA_M306_K4_R3_ROUTE_K_V3 * AUGUR_CUDA_M306_K4_R3_DOWN_ROWS_V3 * 4) as u64,
            )
        }
    }

    impl AugurCudaM304R3CombinedCommitArgsV2 {
        pub const fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_M304_R3_COMBINED_COMMIT_ARGS_VERSION_V2,
                struct_size: core::mem::size_of::<Self>() as u32,
                logits: 0,
                logits_len_bytes: AUGUR_CUDA_M304_R3_LOGITS_BYTES_V1,
                route_step_receipt: 0,
                route_step_receipt_len_bytes: AUGUR_CUDA_M304_R3_ROUTE_STEP_RECEIPT_BYTES_V1,
                partials: 0,
                partials_len_bytes: AUGUR_CUDA_M304_R3_PARTIAL_BYTES_V1,
                commit: 0,
                commit_len_bytes: AUGUR_CUDA_M304_R3_COMBINED_COMMIT_BYTES_V2,
                step_generation: 0,
                vocab: AUGUR_CUDA_M304_R3_VOCAB_V1,
                partial_count: AUGUR_CUDA_M304_R3_PARTIAL_COUNT_V1,
                expected_position: 0,
                reserved: 0,
                plan_digest: [0; 32],
            }
        }
    }

    impl Default for AugurCudaM304R3CombinedCommitArgsV2 {
        fn default() -> Self {
            Self::new()
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaM121Q4kDp4aArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub weight: AugurCudaDevicePtr,
        pub weight_len_bytes: u64,
        pub q8_input: AugurCudaDevicePtr,
        pub q8_input_len_bytes: u64,
        pub q8_scales: AugurCudaDevicePtr,
        pub q8_scales_len_bytes: u64,
        pub output: AugurCudaDevicePtr,
        pub output_len_bytes: u64,
        pub rows: u32,
        pub cols: u32,
        pub blocks_per_row: u32,
        pub group_size: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaAddF32ArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub a: AugurCudaDevicePtr,
        pub a_len_bytes: u64,
        pub b: AugurCudaDevicePtr,
        pub b_len_bytes: u64,
        pub output: AugurCudaDevicePtr,
        pub output_len_bytes: u64,
        pub elements: u32,
        pub reserved: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaGqaRepeatF32ArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub input: AugurCudaDevicePtr,
        pub input_len_bytes: u64,
        pub output: AugurCudaDevicePtr,
        pub output_len_bytes: u64,
        pub source_elements: u32,
        pub head_dim: u32,
        pub repeats: u32,
        pub reserved: u32,
    }

    pub const AUGUR_CUDA_SWIGLU_F32_ARGS_VERSION_V1: u32 = 1;

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct AugurCudaSwigluF32ArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub gate: AugurCudaDevicePtr,
        pub gate_len_bytes: u64,
        pub up: AugurCudaDevicePtr,
        pub up_len_bytes: u64,
        pub output: AugurCudaDevicePtr,
        pub output_len_bytes: u64,
        pub elements: u32,
        pub limit_bits: u32,
    }

    impl AugurCudaSwigluF32ArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_SWIGLU_F32_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                gate: 0,
                gate_len_bytes: 0,
                up: 0,
                up_len_bytes: 0,
                output: 0,
                output_len_bytes: 0,
                elements: 0,
                limit_bits: 0,
            }
        }
    }
    impl Default for AugurCudaSwigluF32ArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    pub const AUGUR_CUDA_SILU_MUL_F32_ARGS_VERSION_V1: u32 = 1;

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaSiluMulF32ArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub gate: AugurCudaDevicePtr,
        pub gate_len_bytes: u64,
        pub up: AugurCudaDevicePtr,
        pub up_len_bytes: u64,
        pub output: AugurCudaDevicePtr,
        pub output_len_bytes: u64,
        pub elements: u32,
        pub reserved: u32,
    }

    impl AugurCudaSiluMulF32ArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_SILU_MUL_F32_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                gate: 0,
                gate_len_bytes: 0,
                up: 0,
                up_len_bytes: 0,
                output: 0,
                output_len_bytes: 0,
                elements: 0,
                reserved: 0,
            }
        }
    }

    impl Default for AugurCudaSiluMulF32ArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    const _: () = assert!(core::mem::size_of::<AugurCudaSiluMulF32ArgsV1>() == 64);
    const _: () = assert!(core::mem::align_of::<AugurCudaSiluMulF32ArgsV1>() == 8);

    pub const AUGUR_CUDA_SILU_MUL_Q8_1_ARGS_VERSION_V1: u32 = 1;

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaSiluMulQ8_1ArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub gate: AugurCudaDevicePtr,
        pub gate_len_bytes: u64,
        pub up: AugurCudaDevicePtr,
        pub up_len_bytes: u64,
        pub output_q8_1: AugurCudaDevicePtr,
        pub output_len_bytes: u64,
        pub elements: u32,
        pub reserved: u32,
    }

    impl AugurCudaSiluMulQ8_1ArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_SILU_MUL_Q8_1_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                gate: 0,
                gate_len_bytes: 0,
                up: 0,
                up_len_bytes: 0,
                output_q8_1: 0,
                output_len_bytes: 0,
                elements: 0,
                reserved: 0,
            }
        }
    }

    impl Default for AugurCudaSiluMulQ8_1ArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    const _: () = assert!(core::mem::size_of::<AugurCudaSiluMulQ8_1ArgsV1>() == 64);
    const _: () = assert!(core::mem::align_of::<AugurCudaSiluMulQ8_1ArgsV1>() == 8);

    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    pub struct AugurCudaQwen35HeadPrepareB8ArgsV1 {
        pub abi_version: u32, pub struct_size: u32, pub mode: u32, pub epsilon_bits: u32,
        pub input: u64, pub input_len_bytes: u64,
        pub weight: u64, pub weight_len_bytes: u64,
        pub normalized: u64, pub normalized_len_bytes: u64,
        pub quantized: u64, pub quantized_len_bytes: u64,
        pub last_hidden: u64, pub last_hidden_len_bytes: u64,
        pub last_normalized: u64, pub last_normalized_len_bytes: u64,
        pub last_quantized: u64, pub last_quantized_len_bytes: u64,
    }
    impl AugurCudaQwen35HeadPrepareB8ArgsV1 {
        pub const fn new() -> Self {
            Self { abi_version: 1, struct_size: 128, mode: 0, epsilon_bits: 0,
                input: 0, input_len_bytes: 0, weight: 0, weight_len_bytes: 0,
                normalized: 0, normalized_len_bytes: 0, quantized: 0, quantized_len_bytes: 0,
                last_hidden: 0, last_hidden_len_bytes: 0, last_normalized: 0, last_normalized_len_bytes: 0,
                last_quantized: 0, last_quantized_len_bytes: 0 }
        }
    }
    impl Default for AugurCudaQwen35HeadPrepareB8ArgsV1 { fn default() -> Self { Self::new() } }
    const _: () = assert!(core::mem::size_of::<AugurCudaQwen35HeadPrepareB8ArgsV1>() == 128);
    const _: () = assert!(core::mem::align_of::<AugurCudaQwen35HeadPrepareB8ArgsV1>() == 8);

    #[cfg(feature = "qwen35-linear-step")]
    pub const AUGUR_CUDA_QWEN35_LINEAR_STEP_ARGS_VERSION_V1: u32 = 1;

    #[cfg(feature = "qwen35-linear-step")]
    pub const AUGUR_CUDA_QWEN35_LINEAR_STEP_KEY_HEADS_V1: usize = 16;
    #[cfg(feature = "qwen35-linear-step")]
    pub const AUGUR_CUDA_QWEN35_LINEAR_STEP_VALUE_HEADS_V1: usize = 48;
    #[cfg(feature = "qwen35-linear-step")]
    pub const AUGUR_CUDA_QWEN35_LINEAR_STEP_HEAD_DIM_V1: usize = 128;
    #[cfg(feature = "qwen35-linear-step")]
    pub const AUGUR_CUDA_QWEN35_LINEAR_STEP_KERNEL_SIZE_V1: usize = 4;
    #[cfg(feature = "qwen35-linear-step")]
    pub const AUGUR_CUDA_QWEN35_LINEAR_STEP_QKV_ELEMENTS_V1: usize = 10_240;
    #[cfg(feature = "qwen35-linear-step")]
    pub const AUGUR_CUDA_QWEN35_LINEAR_STEP_OUTPUT_ELEMENTS_V1: usize = 6_144;
    #[cfg(feature = "qwen35-linear-step")]
    pub const AUGUR_CUDA_QWEN35_LINEAR_STEP_STATE_ELEMENTS_V1: usize = 786_432;
    #[cfg(feature = "qwen35-linear-step")]
    pub const AUGUR_CUDA_QWEN35_LINEAR_STEP_CONV_HISTORY_ELEMENTS_V1: usize = 30_720;
    #[cfg(feature = "qwen35-linear-step")]
    pub const AUGUR_CUDA_QWEN35_LINEAR_STEP_CONV_WEIGHT_ELEMENTS_V1: usize = 40_960;

    #[cfg(feature = "qwen35-linear-step")]
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaQwen35LinearStepArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub qkv_f32: AugurCudaDevicePtr,
        pub qkv_len_bytes: u64,
        pub z_f32: AugurCudaDevicePtr,
        pub z_len_bytes: u64,
        pub conv_weights_f32: AugurCudaDevicePtr,
        pub conv_weights_len_bytes: u64,
        pub conv_history_f32: AugurCudaDevicePtr,
        pub conv_history_len_bytes: u64,
        pub decay_gate_f32: AugurCudaDevicePtr,
        pub decay_gate_len_bytes: u64,
        pub beta_f32: AugurCudaDevicePtr,
        pub beta_len_bytes: u64,
        pub state_f32: AugurCudaDevicePtr,
        pub state_len_bytes: u64,
        pub norm_weight_f32: AugurCudaDevicePtr,
        pub norm_weight_len_bytes: u64,
        pub activated_f32: AugurCudaDevicePtr,
        pub activated_len_bytes: u64,
        pub recurrent_f32: AugurCudaDevicePtr,
        pub recurrent_len_bytes: u64,
        pub output_f32: AugurCudaDevicePtr,
        pub output_len_bytes: u64,
        pub epsilon_bits: u32,
        pub reserved: u32,
    }

    #[cfg(feature = "qwen35-linear-step")]
    impl AugurCudaQwen35LinearStepArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_QWEN35_LINEAR_STEP_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                qkv_f32: 0,
                qkv_len_bytes: 0,
                z_f32: 0,
                z_len_bytes: 0,
                conv_weights_f32: 0,
                conv_weights_len_bytes: 0,
                conv_history_f32: 0,
                conv_history_len_bytes: 0,
                decay_gate_f32: 0,
                decay_gate_len_bytes: 0,
                beta_f32: 0,
                beta_len_bytes: 0,
                state_f32: 0,
                state_len_bytes: 0,
                norm_weight_f32: 0,
                norm_weight_len_bytes: 0,
                activated_f32: 0,
                activated_len_bytes: 0,
                recurrent_f32: 0,
                recurrent_len_bytes: 0,
                output_f32: 0,
                output_len_bytes: 0,
                epsilon_bits: 0,
                reserved: 0,
            }
        }
    }

    #[cfg(feature = "qwen35-linear-step")]
    impl Default for AugurCudaQwen35LinearStepArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    #[cfg(feature = "qwen35-linear-step")]
    const _: () = assert!(core::mem::size_of::<AugurCudaQwen35LinearStepArgsV1>() == 192);
    #[cfg(feature = "qwen35-linear-step")]
    const _: () = assert!(core::mem::align_of::<AugurCudaQwen35LinearStepArgsV1>() == 8);

    #[cfg(feature = "qwen35-linear-step")]
    pub const AUGUR_CUDA_QWEN35_LINEAR_PROJECTION_POSTPROCESS_ARGS_VERSION_V1: u32 = 1;
    #[cfg(feature = "qwen35-linear-step")]
    pub const AUGUR_CUDA_QWEN35_LINEAR_PROJECTION_POSTPROCESS_ELEMENTS_V1: usize = 48;

    #[cfg(feature = "qwen35-linear-step")]
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaQwen35LinearProjectionPostprocessArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub alpha_raw_f32: AugurCudaDevicePtr,
        pub alpha_raw_len_bytes: u64,
        pub beta_raw_f32: AugurCudaDevicePtr,
        pub beta_raw_len_bytes: u64,
        pub dt_bias_f32: AugurCudaDevicePtr,
        pub dt_bias_len_bytes: u64,
        pub ssm_a_f32: AugurCudaDevicePtr,
        pub ssm_a_len_bytes: u64,
        pub decay_gate_f32: AugurCudaDevicePtr,
        pub decay_gate_len_bytes: u64,
        pub beta_out_f32: AugurCudaDevicePtr,
        pub beta_out_len_bytes: u64,
        pub elements: u32,
        pub reserved: u32,
    }

    #[cfg(feature = "qwen35-linear-step")]
    impl AugurCudaQwen35LinearProjectionPostprocessArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_QWEN35_LINEAR_PROJECTION_POSTPROCESS_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                alpha_raw_f32: 0,
                alpha_raw_len_bytes: 0,
                beta_raw_f32: 0,
                beta_raw_len_bytes: 0,
                dt_bias_f32: 0,
                dt_bias_len_bytes: 0,
                ssm_a_f32: 0,
                ssm_a_len_bytes: 0,
                decay_gate_f32: 0,
                decay_gate_len_bytes: 0,
                beta_out_f32: 0,
                beta_out_len_bytes: 0,
                elements: AUGUR_CUDA_QWEN35_LINEAR_PROJECTION_POSTPROCESS_ELEMENTS_V1 as u32,
                reserved: 0,
            }
        }
    }

    #[cfg(feature = "qwen35-linear-step")]
    impl Default for AugurCudaQwen35LinearProjectionPostprocessArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    #[cfg(feature = "qwen35-linear-step")]
    const _: () =
        assert!(core::mem::size_of::<AugurCudaQwen35LinearProjectionPostprocessArgsV1>() == 112);
    #[cfg(feature = "qwen35-linear-step")]
    const _: () =
        assert!(core::mem::align_of::<AugurCudaQwen35LinearProjectionPostprocessArgsV1>() == 8);

    #[cfg(feature = "qwen35-fused-head-norm-rope")]
    pub const AUGUR_CUDA_QWEN35_FUSED_HEAD_NORM_ROPE_ARGS_VERSION_V1: u32 = 1;

    #[cfg(feature = "qwen35-fused-head-norm-rope")]
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaQwen35BatchedHeadNormRopeArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub rows: u32,
        pub epsilon_bits: u32,
        pub reserved0: u32,
        pub reserved1: u32,
        pub addresses: [u64; 11],
        pub lengths: [u64; 11],
    }
    #[cfg(feature = "qwen35-fused-head-norm-rope")]
    const _: () = assert!(core::mem::size_of::<AugurCudaQwen35BatchedHeadNormRopeArgsV1>() == 200);
    #[cfg(feature = "qwen35-fused-head-norm-rope")]
    const _: () = assert!(core::mem::align_of::<AugurCudaQwen35BatchedHeadNormRopeArgsV1>() == 8);

    #[cfg(feature = "qwen35-fused-head-norm-rope")]
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaQwen35FusedHeadNormRopeArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub q_input_f32: AugurCudaDevicePtr,
        pub q_input_len_bytes: u64,
        pub q_weight_f32: AugurCudaDevicePtr,
        pub q_weight_len_bytes: u64,
        pub q_normed_f32: AugurCudaDevicePtr,
        pub q_normed_len_bytes: u64,
        pub q_output_f32: AugurCudaDevicePtr,
        pub q_output_len_bytes: u64,
        pub k_input_f32: AugurCudaDevicePtr,
        pub k_input_len_bytes: u64,
        pub k_weight_f32: AugurCudaDevicePtr,
        pub k_weight_len_bytes: u64,
        pub k_normed_f32: AugurCudaDevicePtr,
        pub k_normed_len_bytes: u64,
        pub k_output_f32: AugurCudaDevicePtr,
        pub k_output_len_bytes: u64,
        pub rope_cos_f32: AugurCudaDevicePtr,
        pub rope_cos_len_bytes: u64,
        pub rope_sin_f32: AugurCudaDevicePtr,
        pub rope_sin_len_bytes: u64,
        pub query_heads: u32,
        pub kv_heads: u32,
        pub head_dim: u32,
        pub rotary_dim: u32,
        pub epsilon_bits: u32,
        pub reserved: u32,
    }

    #[cfg(feature = "qwen35-fused-head-norm-rope")]
    impl AugurCudaQwen35FusedHeadNormRopeArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_QWEN35_FUSED_HEAD_NORM_ROPE_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                q_input_f32: 0,
                q_input_len_bytes: 0,
                q_weight_f32: 0,
                q_weight_len_bytes: 0,
                q_normed_f32: 0,
                q_normed_len_bytes: 0,
                q_output_f32: 0,
                q_output_len_bytes: 0,
                k_input_f32: 0,
                k_input_len_bytes: 0,
                k_weight_f32: 0,
                k_weight_len_bytes: 0,
                k_normed_f32: 0,
                k_normed_len_bytes: 0,
                k_output_f32: 0,
                k_output_len_bytes: 0,
                rope_cos_f32: 0,
                rope_cos_len_bytes: 0,
                rope_sin_f32: 0,
                rope_sin_len_bytes: 0,
                query_heads: AUGUR_CUDA_QWEN35_FULL_ATTENTION_QUERY_HEADS_V1 as u32,
                kv_heads: AUGUR_CUDA_QWEN35_FULL_ATTENTION_KV_HEADS_V1 as u32,
                head_dim: AUGUR_CUDA_QWEN35_FULL_ATTENTION_HEAD_DIM_V1 as u32,
                rotary_dim: AUGUR_CUDA_QWEN35_FULL_ATTENTION_ROTARY_DIM_V1 as u32,
                epsilon_bits: 0,
                reserved: 0,
            }
        }
    }

    #[cfg(feature = "qwen35-fused-head-norm-rope")]
    impl Default for AugurCudaQwen35FusedHeadNormRopeArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    #[cfg(feature = "qwen35-fused-head-norm-rope")]
    const _: () = assert!(core::mem::size_of::<AugurCudaQwen35FusedHeadNormRopeArgsV1>() == 192);
    #[cfg(feature = "qwen35-fused-head-norm-rope")]
    const _: () = assert!(core::mem::align_of::<AugurCudaQwen35FusedHeadNormRopeArgsV1>() == 8);

    pub const AUGUR_CUDA_QWEN35_FULL_ATTENTION_ARGS_VERSION_V1: u32 = 1;
    pub const AUGUR_CUDA_QWEN35_FULL_ATTENTION_QUERY_HEADS_V1: usize = 24;
    pub const AUGUR_CUDA_QWEN35_FULL_ATTENTION_KV_HEADS_V1: usize = 4;
    pub const AUGUR_CUDA_QWEN35_FULL_ATTENTION_HEAD_DIM_V1: usize = 256;
    pub const AUGUR_CUDA_QWEN35_FULL_ATTENTION_ROTARY_DIM_V1: usize = 64;
    pub const AUGUR_CUDA_QWEN35_FULL_ATTENTION_QUERY_ELEMENTS_V1: usize = 6_144;
    pub const AUGUR_CUDA_QWEN35_FULL_ATTENTION_KV_ELEMENTS_V1: usize = 1_024;
    pub const AUGUR_CUDA_QWEN35_FULL_ATTENTION_MAX_CAPACITY_V1: usize = 4_096;

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaQwen35FullAttentionArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub query_f32: AugurCudaDevicePtr,
        pub query_len_bytes: u64,
        pub gate_f32: AugurCudaDevicePtr,
        pub gate_len_bytes: u64,
        pub key_f32: AugurCudaDevicePtr,
        pub key_len_bytes: u64,
        pub value_f32: AugurCudaDevicePtr,
        pub value_len_bytes: u64,
        pub key_cache_f32: AugurCudaDevicePtr,
        pub key_cache_len_bytes: u64,
        pub value_cache_f32: AugurCudaDevicePtr,
        pub value_cache_len_bytes: u64,
        pub attention_f32: AugurCudaDevicePtr,
        pub attention_len_bytes: u64,
        pub gated_f32: AugurCudaDevicePtr,
        pub gated_len_bytes: u64,
        pub capacity: u32,
        pub position: u32,
        pub query_heads: u32,
        pub kv_heads: u32,
        pub head_dim: u32,
        pub rotary_dim: u32,
        pub reserved: u32,
    }

    impl AugurCudaQwen35FullAttentionArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_QWEN35_FULL_ATTENTION_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                query_f32: 0,
                query_len_bytes: 0,
                gate_f32: 0,
                gate_len_bytes: 0,
                key_f32: 0,
                key_len_bytes: 0,
                value_f32: 0,
                value_len_bytes: 0,
                key_cache_f32: 0,
                key_cache_len_bytes: 0,
                value_cache_f32: 0,
                value_cache_len_bytes: 0,
                attention_f32: 0,
                attention_len_bytes: 0,
                gated_f32: 0,
                gated_len_bytes: 0,
                capacity: 0,
                position: 0,
                query_heads: AUGUR_CUDA_QWEN35_FULL_ATTENTION_QUERY_HEADS_V1 as u32,
                kv_heads: AUGUR_CUDA_QWEN35_FULL_ATTENTION_KV_HEADS_V1 as u32,
                head_dim: AUGUR_CUDA_QWEN35_FULL_ATTENTION_HEAD_DIM_V1 as u32,
                rotary_dim: AUGUR_CUDA_QWEN35_FULL_ATTENTION_ROTARY_DIM_V1 as u32,
                reserved: 0,
            }
        }
    }

    impl Default for AugurCudaQwen35FullAttentionArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    const _: () = assert!(core::mem::size_of::<AugurCudaQwen35FullAttentionArgsV1>() == 168);
    const _: () = assert!(core::mem::align_of::<AugurCudaQwen35FullAttentionArgsV1>() == 8);

    #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
    pub const AUGUR_CUDA_QWEN35_FULL_ATTENTION_ARGS_VERSION_V2: u32 = 2;
    #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
    pub const AUGUR_CUDA_QWEN35_FULL_ATTENTION_SPLITS_V2: usize = 72;
    #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
    pub const AUGUR_CUDA_QWEN35_FULL_ATTENTION_STAGE1_THREADS_V2: usize = 256;
    #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
    pub const AUGUR_CUDA_QWEN35_FULL_ATTENTION_TILE_POSITIONS_V2: usize = 8;
    #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
    pub const AUGUR_CUDA_QWEN35_FULL_ATTENTION_MERGE_SEGMENTS_V2: usize = 2;
    #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
    pub const AUGUR_CUDA_QWEN35_FULL_ATTENTION_MERGE_THREADS_V2: usize = 128;
    #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
    pub const AUGUR_CUDA_QWEN35_FULL_ATTENTION_PARTIAL_STRIDE_V2: usize = 258;
    #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
    pub const AUGUR_CUDA_QWEN35_FULL_ATTENTION_PARTIAL_ELEMENTS_V2: usize = 445_824;
    #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
    pub const AUGUR_CUDA_QWEN35_FULL_ATTENTION_FUSED_Q8_BYTES_V2: usize = 6_912;
    #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
    pub const AUGUR_CUDA_QWEN35_FULL_ATTENTION_WORKSPACE_ELEMENTS_V2: usize = 447_552;

    #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaQwen35FullAttentionArgsV2 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub query_f32: AugurCudaDevicePtr,
        pub query_len_bytes: u64,
        pub gate_f32: AugurCudaDevicePtr,
        pub gate_len_bytes: u64,
        pub key_f32: AugurCudaDevicePtr,
        pub key_len_bytes: u64,
        pub value_f32: AugurCudaDevicePtr,
        pub value_len_bytes: u64,
        pub key_cache_f32: AugurCudaDevicePtr,
        pub key_cache_len_bytes: u64,
        pub value_cache_f32: AugurCudaDevicePtr,
        pub value_cache_len_bytes: u64,
        pub attention_f32: AugurCudaDevicePtr,
        pub attention_len_bytes: u64,
        pub gated_f32: AugurCudaDevicePtr,
        pub gated_len_bytes: u64,
        pub workspace_f32: AugurCudaDevicePtr,
        pub workspace_len_bytes: u64,
        pub capacity: u32,
        pub position: u32,
        pub query_heads: u32,
        pub kv_heads: u32,
        pub head_dim: u32,
        pub rotary_dim: u32,
        pub split_count: u32,
        pub reserved: u32,
    }

    #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
    impl AugurCudaQwen35FullAttentionArgsV2 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_QWEN35_FULL_ATTENTION_ARGS_VERSION_V2,
                struct_size: core::mem::size_of::<Self>() as u32,
                query_f32: 0,
                query_len_bytes: 0,
                gate_f32: 0,
                gate_len_bytes: 0,
                key_f32: 0,
                key_len_bytes: 0,
                value_f32: 0,
                value_len_bytes: 0,
                key_cache_f32: 0,
                key_cache_len_bytes: 0,
                value_cache_f32: 0,
                value_cache_len_bytes: 0,
                attention_f32: 0,
                attention_len_bytes: 0,
                gated_f32: 0,
                gated_len_bytes: 0,
                workspace_f32: 0,
                workspace_len_bytes: 0,
                capacity: 0,
                position: 0,
                query_heads: AUGUR_CUDA_QWEN35_FULL_ATTENTION_QUERY_HEADS_V1 as u32,
                kv_heads: AUGUR_CUDA_QWEN35_FULL_ATTENTION_KV_HEADS_V1 as u32,
                head_dim: AUGUR_CUDA_QWEN35_FULL_ATTENTION_HEAD_DIM_V1 as u32,
                rotary_dim: AUGUR_CUDA_QWEN35_FULL_ATTENTION_ROTARY_DIM_V1 as u32,
                split_count: AUGUR_CUDA_QWEN35_FULL_ATTENTION_SPLITS_V2 as u32,
                reserved: 0,
            }
        }
    }

    #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
    impl Default for AugurCudaQwen35FullAttentionArgsV2 {
        fn default() -> Self {
            Self::new()
        }
    }

    #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
    const _: () = assert!(core::mem::size_of::<AugurCudaQwen35FullAttentionArgsV2>() == 184);
    #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
    const _: () = assert!(core::mem::align_of::<AugurCudaQwen35FullAttentionArgsV2>() == 8);

    #[cfg(feature = "qwen35-kv8-full-attention")]
    pub const AUGUR_CUDA_QWEN35_KV8_FULL_ATTENTION_ARGS_VERSION_V3: u32 = 3;
    #[cfg(feature = "qwen35-kv8-full-attention")]
    pub const AUGUR_CUDA_QWEN35_KV8_FULL_ATTENTION_MAX_CAPACITY_V3: usize = 262_144;
    #[cfg(feature = "qwen35-kv8-full-attention")]
    pub const AUGUR_CUDA_QWEN35_KV8_BLOCK_ELEMENTS_V3: usize = 32;
    #[cfg(feature = "qwen35-kv8-full-attention")]
    pub const AUGUR_CUDA_QWEN35_KV8_BLOCK_BYTES_V3: usize = 34;
    #[cfg(feature = "qwen35-kv8-full-attention")]
    pub const AUGUR_CUDA_QWEN35_KV8_BLOCKS_PER_HEAD_V3: usize = 8;
    #[cfg(feature = "qwen35-kv8-full-attention")]
    pub const AUGUR_CUDA_QWEN35_KV8_HEAD_ROW_BYTES_V3: usize = 272;
    #[cfg(feature = "qwen35-kv8-full-attention")]
    pub const AUGUR_CUDA_QWEN35_KV8_CACHE_ROW_BYTES_V3: usize = 1_088;

    #[cfg(feature = "qwen35-kv8-full-attention")]
    pub const AUGUR_CUDA_QWEN35_KV8_BATCHED_ATTENTION_ARGS_VERSION_V4: u32 = 4;
    #[cfg(feature = "qwen35-kv8-full-attention")]
    pub const AUGUR_CUDA_QWEN35_KV8_BATCHED_ATTENTION_MAX_TOKENS_V4: usize = 8;

    #[cfg(feature = "qwen35-kv8-full-attention")]
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaQwen35Kv8FullAttentionArgsV3 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub query_f32: AugurCudaDevicePtr,
        pub query_len_bytes: u64,
        pub gate_f32: AugurCudaDevicePtr,
        pub gate_len_bytes: u64,
        pub key_f32: AugurCudaDevicePtr,
        pub key_len_bytes: u64,
        pub value_f32: AugurCudaDevicePtr,
        pub value_len_bytes: u64,
        pub key_cache_q8_0: AugurCudaDevicePtr,
        pub key_cache_len_bytes: u64,
        pub value_cache_q8_0: AugurCudaDevicePtr,
        pub value_cache_len_bytes: u64,
        pub attention_f32: AugurCudaDevicePtr,
        pub attention_len_bytes: u64,
        pub gated_f32: AugurCudaDevicePtr,
        pub gated_len_bytes: u64,
        pub workspace_f32: AugurCudaDevicePtr,
        pub workspace_len_bytes: u64,
        pub capacity: u32,
        pub position: u32,
        pub query_heads: u32,
        pub kv_heads: u32,
        pub head_dim: u32,
        pub rotary_dim: u32,
        pub split_count: u32,
        pub reserved: u32,
    }

    #[cfg(feature = "qwen35-kv8-full-attention")]
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaQwen35Kv8BatchedAttentionArgsV4 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub query_f32: AugurCudaDevicePtr,
        pub query_len_bytes: u64,
        pub gate_f32: AugurCudaDevicePtr,
        pub gate_len_bytes: u64,
        pub key_f32: AugurCudaDevicePtr,
        pub key_len_bytes: u64,
        pub value_f32: AugurCudaDevicePtr,
        pub value_len_bytes: u64,
        pub key_cache_q8_0: AugurCudaDevicePtr,
        pub key_cache_len_bytes: u64,
        pub value_cache_q8_0: AugurCudaDevicePtr,
        pub value_cache_len_bytes: u64,
        pub attention_f32: AugurCudaDevicePtr,
        pub attention_len_bytes: u64,
        pub gated_f32: AugurCudaDevicePtr,
        pub gated_len_bytes: u64,
        pub workspace_f32: AugurCudaDevicePtr,
        pub workspace_len_bytes: u64,
        pub capacity: u32,
        pub position_base: u32,
        pub tokens: u32,
        pub query_heads: u32,
        pub kv_heads: u32,
        pub head_dim: u32,
        pub rotary_dim: u32,
        pub split_count: u32,
        pub reserved: u32,
    }

    #[cfg(feature = "qwen35-kv8-full-attention")]
    impl AugurCudaQwen35Kv8BatchedAttentionArgsV4 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_QWEN35_KV8_BATCHED_ATTENTION_ARGS_VERSION_V4,
                struct_size: core::mem::size_of::<Self>() as u32,
                query_f32: 0,
                query_len_bytes: 0,
                gate_f32: 0,
                gate_len_bytes: 0,
                key_f32: 0,
                key_len_bytes: 0,
                value_f32: 0,
                value_len_bytes: 0,
                key_cache_q8_0: 0,
                key_cache_len_bytes: 0,
                value_cache_q8_0: 0,
                value_cache_len_bytes: 0,
                attention_f32: 0,
                attention_len_bytes: 0,
                gated_f32: 0,
                gated_len_bytes: 0,
                workspace_f32: 0,
                workspace_len_bytes: 0,
                capacity: 0,
                position_base: 0,
                tokens: 0,
                query_heads: AUGUR_CUDA_QWEN35_FULL_ATTENTION_QUERY_HEADS_V1 as u32,
                kv_heads: AUGUR_CUDA_QWEN35_FULL_ATTENTION_KV_HEADS_V1 as u32,
                head_dim: AUGUR_CUDA_QWEN35_FULL_ATTENTION_HEAD_DIM_V1 as u32,
                rotary_dim: AUGUR_CUDA_QWEN35_FULL_ATTENTION_ROTARY_DIM_V1 as u32,
                split_count: AUGUR_CUDA_QWEN35_FULL_ATTENTION_SPLITS_V2 as u32,
                reserved: 0,
            }
        }
    }

    #[cfg(feature = "qwen35-kv8-full-attention")]
    impl Default for AugurCudaQwen35Kv8BatchedAttentionArgsV4 {
        fn default() -> Self {
            Self::new()
        }
    }

    #[cfg(feature = "qwen35-kv8-full-attention")]
    const _: () = assert!(core::mem::size_of::<AugurCudaQwen35Kv8BatchedAttentionArgsV4>() == 192);

    // Same field layout, independently versioned input-only prefill operation.
    #[cfg(all(
        feature = "qwen35-prefill-cublas",
        feature = "qwen35-kv8-full-attention"
    ))]
    pub type AugurCudaQwen35CanonicalPrefillArgsV1 = AugurCudaQwen35Kv8BatchedAttentionArgsV4;
    #[cfg(all(
        feature = "qwen35-prefill-cublas",
        feature = "qwen35-kv8-full-attention"
    ))]
    pub const AUGUR_CUDA_QWEN35_CANONICAL_PREFILL_ARGS_VERSION_V1: u32 = 1;
    #[cfg(all(
        feature = "qwen35-prefill-cublas",
        feature = "qwen35-kv8-full-attention"
    ))]
    pub const AUGUR_CUDA_QWEN35_CANONICAL_PREFILL_MAX_TOKENS_V1: usize = 128;

    #[cfg(feature = "qwen35-kv8-full-attention")]
    impl AugurCudaQwen35Kv8FullAttentionArgsV3 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_QWEN35_KV8_FULL_ATTENTION_ARGS_VERSION_V3,
                struct_size: core::mem::size_of::<Self>() as u32,
                query_f32: 0,
                query_len_bytes: 0,
                gate_f32: 0,
                gate_len_bytes: 0,
                key_f32: 0,
                key_len_bytes: 0,
                value_f32: 0,
                value_len_bytes: 0,
                key_cache_q8_0: 0,
                key_cache_len_bytes: 0,
                value_cache_q8_0: 0,
                value_cache_len_bytes: 0,
                attention_f32: 0,
                attention_len_bytes: 0,
                gated_f32: 0,
                gated_len_bytes: 0,
                workspace_f32: 0,
                workspace_len_bytes: 0,
                capacity: 0,
                position: 0,
                query_heads: AUGUR_CUDA_QWEN35_FULL_ATTENTION_QUERY_HEADS_V1 as u32,
                kv_heads: AUGUR_CUDA_QWEN35_FULL_ATTENTION_KV_HEADS_V1 as u32,
                head_dim: AUGUR_CUDA_QWEN35_FULL_ATTENTION_HEAD_DIM_V1 as u32,
                rotary_dim: AUGUR_CUDA_QWEN35_FULL_ATTENTION_ROTARY_DIM_V1 as u32,
                split_count: AUGUR_CUDA_QWEN35_FULL_ATTENTION_SPLITS_V2 as u32,
                reserved: 0,
            }
        }
    }

    #[cfg(feature = "qwen35-kv8-full-attention")]
    impl Default for AugurCudaQwen35Kv8FullAttentionArgsV3 {
        fn default() -> Self {
            Self::new()
        }
    }

    #[cfg(feature = "qwen35-kv8-full-attention")]
    const _: () = assert!(core::mem::size_of::<AugurCudaQwen35Kv8FullAttentionArgsV3>() == 184);
    #[cfg(feature = "qwen35-kv8-full-attention")]
    const _: () = assert!(core::mem::align_of::<AugurCudaQwen35Kv8FullAttentionArgsV3>() == 8);

    #[cfg(all(
        feature = "qwen35-prefill-cublas",
        feature = "qwen35-kv8-full-attention"
    ))]
    pub const AUGUR_CUDA_QWEN35_PREFILL_KV8_CACHE_QUANTIZE_ARGS_VERSION_V1: u32 = 1;

    #[cfg(all(
        feature = "qwen35-prefill-cublas",
        feature = "qwen35-kv8-full-attention"
    ))]
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaQwen35PrefillKv8CacheQuantizeArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub key_cache_f32: AugurCudaDevicePtr,
        pub key_cache_f32_len_bytes: u64,
        pub value_cache_f32: AugurCudaDevicePtr,
        pub value_cache_f32_len_bytes: u64,
        pub key_cache_q8_0: AugurCudaDevicePtr,
        pub key_cache_q8_0_len_bytes: u64,
        pub value_cache_q8_0: AugurCudaDevicePtr,
        pub value_cache_q8_0_len_bytes: u64,
        pub capacity: u32,
        pub position_base: u32,
        pub tokens: u32,
        pub kv_heads: u32,
        pub head_dim: u32,
        pub reserved: u32,
    }

    #[cfg(all(
        feature = "qwen35-prefill-cublas",
        feature = "qwen35-kv8-full-attention"
    ))]
    impl AugurCudaQwen35PrefillKv8CacheQuantizeArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_QWEN35_PREFILL_KV8_CACHE_QUANTIZE_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                key_cache_f32: 0,
                key_cache_f32_len_bytes: 0,
                value_cache_f32: 0,
                value_cache_f32_len_bytes: 0,
                key_cache_q8_0: 0,
                key_cache_q8_0_len_bytes: 0,
                value_cache_q8_0: 0,
                value_cache_q8_0_len_bytes: 0,
                capacity: 0,
                position_base: 0,
                tokens: 0,
                kv_heads: AUGUR_CUDA_QWEN35_FULL_ATTENTION_KV_HEADS_V1 as u32,
                head_dim: AUGUR_CUDA_QWEN35_FULL_ATTENTION_HEAD_DIM_V1 as u32,
                reserved: 0,
            }
        }
    }

    #[cfg(all(
        feature = "qwen35-prefill-cublas",
        feature = "qwen35-kv8-full-attention"
    ))]
    impl Default for AugurCudaQwen35PrefillKv8CacheQuantizeArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    #[cfg(all(
        feature = "qwen35-prefill-cublas",
        feature = "qwen35-kv8-full-attention"
    ))]
    const _: () = assert!(core::mem::size_of::<AugurCudaQwen35PrefillKv8CacheQuantizeArgsV1>() == 96);
    #[cfg(all(
        feature = "qwen35-prefill-cublas",
        feature = "qwen35-kv8-full-attention"
    ))]
    const _: () = assert!(core::mem::align_of::<AugurCudaQwen35PrefillKv8CacheQuantizeArgsV1>() == 8);

    pub const AUGUR_CUDA_BF16_F32_MATVEC_ARGS_VERSION_V1: u32 = 1;

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaBf16F32MatvecArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub weight: AugurCudaDevicePtr,
        pub weight_len_bytes: u64,
        pub input: AugurCudaDevicePtr,
        pub input_len_bytes: u64,
        pub output: AugurCudaDevicePtr,
        pub output_len_bytes: u64,
        pub rows: u32,
        pub cols: u32,
    }
    impl AugurCudaBf16F32MatvecArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_BF16_F32_MATVEC_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                weight: 0,
                weight_len_bytes: 0,
                input: 0,
                input_len_bytes: 0,
                output: 0,
                output_len_bytes: 0,
                rows: 0,
                cols: 0,
            }
        }
    }
    impl Default for AugurCudaBf16F32MatvecArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    pub const AUGUR_CUDA_Q8_0_F32_MATVEC_ARGS_VERSION_V1: u32 = 1;

    /// GGML Q8_0 row-major weight blocks consumed directly by the SM70
    /// component kernel.  A row consists of `cols / 32` blocks, each 34 bytes
    /// (FP16 scale followed by 32 signed int8 codes).
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaQ8_0F32MatvecArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub weight: AugurCudaDevicePtr,
        pub weight_len_bytes: u64,
        pub input: AugurCudaDevicePtr,
        pub input_len_bytes: u64,
        pub output: AugurCudaDevicePtr,
        pub output_len_bytes: u64,
        pub rows: u32,
        pub cols: u32,
    }

    impl AugurCudaQ8_0F32MatvecArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_Q8_0_F32_MATVEC_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                weight: 0,
                weight_len_bytes: 0,
                input: 0,
                input_len_bytes: 0,
                output: 0,
                output_len_bytes: 0,
                rows: 0,
                cols: 0,
            }
        }
    }

    impl Default for AugurCudaQ8_0F32MatvecArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    pub const AUGUR_CUDA_Q8_1_ACTIVATION_QUANTIZE_ARGS_VERSION_V1: u32 = 1;
    
    

    /// Device-side f32 to llama.cpp-compatible Q8_1 activation conversion.
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaQ8_1ActivationQuantizeArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub input_f32: AugurCudaDevicePtr,
        pub input_len_bytes: u64,
        pub output_q8_1: AugurCudaDevicePtr,
        pub output_len_bytes: u64,
        pub elements: u32,
        pub reserved: u32,
    }

    impl AugurCudaQ8_1ActivationQuantizeArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_Q8_1_ACTIVATION_QUANTIZE_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                input_f32: 0,
                input_len_bytes: 0,
                output_q8_1: 0,
                output_len_bytes: 0,
                elements: 0,
                reserved: 0,
            }
        }
    }

    impl Default for AugurCudaQ8_1ActivationQuantizeArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    pub const AUGUR_CUDA_Q8_0_Q8_1_MATVEC_ARGS_VERSION_V1: u32 = 1;

    /// GGML Q8_0 weights multiplied by one device Q8_1 activation vector.
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaQ8_0Q8_1MatvecArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub weight: AugurCudaDevicePtr,
        pub weight_len_bytes: u64,
        pub input_q8_1: AugurCudaDevicePtr,
        pub input_len_bytes: u64,
        pub output: AugurCudaDevicePtr,
        pub output_len_bytes: u64,
        pub rows: u32,
        pub cols: u32,
    }

    impl AugurCudaQ8_0Q8_1MatvecArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_Q8_0_Q8_1_MATVEC_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                weight: 0,
                weight_len_bytes: 0,
                input_q8_1: 0,
                input_len_bytes: 0,
                output: 0,
                output_len_bytes: 0,
                rows: 0,
                cols: 0,
            }
        }
    }

    impl Default for AugurCudaQ8_0Q8_1MatvecArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    #[cfg(feature = "qwen35-fused-ffn-swiglu")]
    pub const AUGUR_CUDA_Q8_0_Q8_1_FUSED_SWIGLU_ARGS_VERSION_V1: u32 = 1;

    #[cfg(feature = "qwen35-fused-ffn-swiglu")]
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaQ8_0Q8_1FusedSwigluArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub gate_weight: AugurCudaDevicePtr,
        pub gate_weight_len_bytes: u64,
        pub up_weight: AugurCudaDevicePtr,
        pub up_weight_len_bytes: u64,
        pub input_q8_1: AugurCudaDevicePtr,
        pub input_len_bytes: u64,
        pub output: AugurCudaDevicePtr,
        pub output_len_bytes: u64,
        pub rows: u32,
        pub cols: u32,
    }

    #[cfg(feature = "qwen35-fused-ffn-swiglu")]
    impl AugurCudaQ8_0Q8_1FusedSwigluArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_Q8_0_Q8_1_FUSED_SWIGLU_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                gate_weight: 0,
                gate_weight_len_bytes: 0,
                up_weight: 0,
                up_weight_len_bytes: 0,
                input_q8_1: 0,
                input_len_bytes: 0,
                output: 0,
                output_len_bytes: 0,
                rows: 0,
                cols: 0,
            }
        }
    }

    #[cfg(feature = "qwen35-fused-ffn-swiglu")]
    impl Default for AugurCudaQ8_0Q8_1FusedSwigluArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    pub const AUGUR_CUDA_NVFP4_SM70_F32_MATVEC_ARGS_VERSION_V1: u32 = 1;

    /// GGML NVFP4 row-major blocks consumed directly by the SM70 component
    /// kernel.  Every 64 values occupy four UE4M3 scale bytes followed by 32
    /// packed E2M1 bytes; the tensor-level scale is carried separately.
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct AugurCudaNvfp4Sm70F32MatvecArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub weight: AugurCudaDevicePtr,
        pub weight_len_bytes: u64,
        pub input: AugurCudaDevicePtr,
        pub input_len_bytes: u64,
        pub output: AugurCudaDevicePtr,
        pub output_len_bytes: u64,
        pub rows: u32,
        pub cols: u32,
        pub tensor_scale: f32,
        pub reserved: u32,
    }

    impl AugurCudaNvfp4Sm70F32MatvecArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_NVFP4_SM70_F32_MATVEC_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                weight: 0,
                weight_len_bytes: 0,
                input: 0,
                input_len_bytes: 0,
                output: 0,
                output_len_bytes: 0,
                rows: 0,
                cols: 0,
                tensor_scale: 0.0,
                reserved: 0,
            }
        }
    }

    impl Default for AugurCudaNvfp4Sm70F32MatvecArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    pub const AUGUR_CUDA_NVFP4_SM70_Q8_1_MATVEC_ARGS_VERSION_V1: u32 = 1;

    /// GGML NVFP4 weights multiplied by one device Q8_1 activation vector.
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct AugurCudaNvfp4Sm70Q8_1MatvecArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub weight: AugurCudaDevicePtr,
        pub weight_len_bytes: u64,
        pub input_q8_1: AugurCudaDevicePtr,
        pub input_len_bytes: u64,
        pub output: AugurCudaDevicePtr,
        pub output_len_bytes: u64,
        pub rows: u32,
        pub cols: u32,
        pub tensor_scale: f32,
        pub reserved: u32,
    }

    impl AugurCudaNvfp4Sm70Q8_1MatvecArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_NVFP4_SM70_Q8_1_MATVEC_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                weight: 0,
                weight_len_bytes: 0,
                input_q8_1: 0,
                input_len_bytes: 0,
                output: 0,
                output_len_bytes: 0,
                rows: 0,
                cols: 0,
                tensor_scale: 0.0,
                reserved: 0,
            }
        }
    }

    impl Default for AugurCudaNvfp4Sm70Q8_1MatvecArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    #[cfg(feature = "qwen35-fused-ffn-swiglu")]
    pub const AUGUR_CUDA_NVFP4_SM70_Q8_1_FUSED_SWIGLU_ARGS_VERSION_V1: u32 = 1;

    #[cfg(feature = "qwen35-fused-ffn-swiglu")]
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct AugurCudaNvfp4Sm70Q8_1FusedSwigluArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub gate_weight: AugurCudaDevicePtr,
        pub gate_weight_len_bytes: u64,
        pub up_weight: AugurCudaDevicePtr,
        pub up_weight_len_bytes: u64,
        pub input_q8_1: AugurCudaDevicePtr,
        pub input_len_bytes: u64,
        pub output: AugurCudaDevicePtr,
        pub output_len_bytes: u64,
        pub rows: u32,
        pub cols: u32,
        pub gate_tensor_scale: f32,
        pub up_tensor_scale: f32,
    }

    #[cfg(feature = "qwen35-fused-ffn-swiglu")]
    impl AugurCudaNvfp4Sm70Q8_1FusedSwigluArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_NVFP4_SM70_Q8_1_FUSED_SWIGLU_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                gate_weight: 0,
                gate_weight_len_bytes: 0,
                up_weight: 0,
                up_weight_len_bytes: 0,
                input_q8_1: 0,
                input_len_bytes: 0,
                output: 0,
                output_len_bytes: 0,
                rows: 0,
                cols: 0,
                gate_tensor_scale: 0.0,
                up_tensor_scale: 0.0,
            }
        }
    }

    #[cfg(feature = "qwen35-fused-ffn-swiglu")]
    impl Default for AugurCudaNvfp4Sm70Q8_1FusedSwigluArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    #[cfg(feature = "qwen35-prefill-cublas")]
    pub const AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_PLAN_ARGS_VERSION_V1: u32 = 1;
    #[cfg(feature = "qwen35-prefill-cublas")]
    pub const AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_PROJECTION_ARGS_VERSION_V1: u32 = 1;
    #[cfg(feature = "qwen35-prefill-cublas")]
    pub const AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1: u32 = 1;
    #[cfg(feature = "qwen35-prefill-cublas")]
    pub const AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_NVFP4_V1: u32 = 2;
    #[cfg(feature = "qwen35-prefill-cublas")]
    pub const AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_BF16_V1: u32 = 3;
    pub const AUGUR_CUDA_DFLASH_SELECTOR_GATE_WEIGHT_F32_V1: u32 = 4;
    pub const AUGUR_CUDA_DFLASH_SELECTOR_GATE_F64_ACCUM_V1: u32 = 5;
    pub const AUGUR_CUDA_QWEN35_PLANAR_Q8_HEAD_WEIGHT_V1: u32 = 4;
    pub const AUGUR_CUDA_QWEN35_ROW_Q8_INPUT_WEIGHT_V1: u32 = 5;
    
    
    #[cfg(feature = "qwen35-prefill-cublas")]
    pub const AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_MAX_TOKENS_V1: u32 = 1024;
    
    #[cfg(feature = "qwen35-prefill-cublas")]
    pub const AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_COMPILED_MAX_TOKENS_V1: u32 =
        if cfg!(feature = "qwen35-augur-large-prefill") {6400} else {AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_MAX_TOKENS_V1};
    #[cfg(feature = "qwen35-prefill-cublas")]
    pub const AUGUR_CUDA_QWEN35_PREFILL_RMSNORM_ARGS_VERSION_V1: u32 = 1;
    #[cfg(feature = "qwen35-prefill-cublas")]
    pub const AUGUR_CUDA_QWEN35_PREFILL_LINEAR_CHUNK_ARGS_VERSION_V1: u32 = 1;
    #[cfg(feature = "qwen35-prefill-cublas")]
    pub const AUGUR_CUDA_QWEN35_PREFILL_FULL_ATTENTION_ARGS_VERSION_V1: u32 = 1;
    #[cfg(all(
        feature = "qwen35-prefill-cublas",
        feature = "qwen35-kv8-full-attention"
    ))]
    pub const AUGUR_CUDA_QWEN35_PREFILL_KV8_CHUNK_ATTENTION_ARGS_VERSION_V1: u32 = 1;
    
    
    

    #[cfg(feature = "qwen35-prefill-cublas")]
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaQwen35PrefillCublasPlanArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub max_tokens: u32,
        pub reserved0: u32,
        pub decoded_weight_f16: AugurCudaDevicePtr,
        pub decoded_weight_len_bytes: u64,
        pub decoded_input_f16: AugurCudaDevicePtr,
        pub decoded_input_len_bytes: u64,
        pub reserved: [u64; 2],
    }

    #[cfg(feature = "qwen35-prefill-cublas")]
    impl AugurCudaQwen35PrefillCublasPlanArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_PLAN_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                max_tokens: 0,
                reserved0: 0,
                decoded_weight_f16: 0,
                decoded_weight_len_bytes: 0,
                decoded_input_f16: 0,
                decoded_input_len_bytes: 0,
                reserved: [0; 2],
            }
        }
    }

    #[cfg(feature = "qwen35-prefill-cublas")]
    impl Default for AugurCudaQwen35PrefillCublasPlanArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    #[cfg(feature = "qwen35-prefill-cublas")]
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct AugurCudaQwen35PrefillCublasProjectionArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub weight_format: u32,
        pub tokens: u32,
        pub rows: u32,
        pub cols: u32,
        pub weight: AugurCudaDevicePtr,
        pub weight_len_bytes: u64,
        pub input_q8_1: AugurCudaDevicePtr,
        pub input_len_bytes: u64,
        pub output_f32: AugurCudaDevicePtr,
        pub output_len_bytes: u64,
        pub tensor_scale: f32,
        pub reserved: u32,
    }

    #[cfg(feature = "qwen35-prefill-cublas")]
    impl AugurCudaQwen35PrefillCublasProjectionArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_PROJECTION_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                weight_format: 0,
                tokens: 0,
                rows: 0,
                cols: 0,
                weight: 0,
                weight_len_bytes: 0,
                input_q8_1: 0,
                input_len_bytes: 0,
                output_f32: 0,
                output_len_bytes: 0,
                tensor_scale: 0.0,
                reserved: 0,
            }
        }
    }

    #[cfg(feature = "qwen35-prefill-cublas")]
    impl Default for AugurCudaQwen35PrefillCublasProjectionArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Separate direct-F32 ABI; never aliases Q8_1 activation storage.
    #[cfg(feature = "qwen35-prefill-cublas")]
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct AugurCudaQwen35PrefillF32ProjectionArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub weight_format: u32,
        pub tokens: u32,
        pub rows: u32,
        pub cols: u32,
        pub weight: AugurCudaDevicePtr,
        pub weight_len_bytes: u64,
        pub input_f32: AugurCudaDevicePtr,
        pub input_len_bytes: u64,
        pub output_f32: AugurCudaDevicePtr,
        pub output_len_bytes: u64,
        pub tensor_scale: f32,
        pub reserved: u32,
    }

    

    

    

    #[cfg(feature = "qwen35-prefill-cublas")]
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaQwen35PrefillRmsnormArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub tokens: u32,
        pub cols: u32,
        pub input_f32: AugurCudaDevicePtr,
        pub input_len_bytes: u64,
        pub weight_f32: AugurCudaDevicePtr,
        pub weight_len_bytes: u64,
        pub output_f32: AugurCudaDevicePtr,
        pub output_len_bytes: u64,
        pub epsilon_bits: u32,
        pub reserved: u32,
    }

    #[cfg(feature = "qwen35-prefill-cublas")]
    impl AugurCudaQwen35PrefillRmsnormArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_QWEN35_PREFILL_RMSNORM_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                tokens: 0,
                cols: 0,
                input_f32: 0,
                input_len_bytes: 0,
                weight_f32: 0,
                weight_len_bytes: 0,
                output_f32: 0,
                output_len_bytes: 0,
                epsilon_bits: 0,
                reserved: 0,
            }
        }
    }

    #[cfg(feature = "qwen35-prefill-cublas")]
    impl Default for AugurCudaQwen35PrefillRmsnormArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    #[cfg(feature = "qwen35-prefill-cublas")]
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaQwen35PrefillLinearChunkArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub tokens: u32,
        pub reserved0: u32,
        pub qkv_f32: AugurCudaDevicePtr,
        pub qkv_len_bytes: u64,
        pub z_f32: AugurCudaDevicePtr,
        pub z_len_bytes: u64,
        pub conv_weights_f32: AugurCudaDevicePtr,
        pub conv_weights_len_bytes: u64,
        pub conv_history_f32: AugurCudaDevicePtr,
        pub conv_history_len_bytes: u64,
        pub alpha_raw_f32: AugurCudaDevicePtr,
        pub alpha_raw_len_bytes: u64,
        pub beta_raw_f32: AugurCudaDevicePtr,
        pub beta_raw_len_bytes: u64,
        pub dt_bias_f32: AugurCudaDevicePtr,
        pub dt_bias_len_bytes: u64,
        pub ssm_a_f32: AugurCudaDevicePtr,
        pub ssm_a_len_bytes: u64,
        pub state_f32: AugurCudaDevicePtr,
        pub state_len_bytes: u64,
        pub norm_weight_f32: AugurCudaDevicePtr,
        pub norm_weight_len_bytes: u64,
        pub activated_f32: AugurCudaDevicePtr,
        pub activated_len_bytes: u64,
        pub recurrent_f32: AugurCudaDevicePtr,
        pub recurrent_len_bytes: u64,
        pub output_f32: AugurCudaDevicePtr,
        pub output_len_bytes: u64,
        pub epsilon_bits: u32,
        pub reserved1: u32,
    }

    #[cfg(feature = "qwen35-prefill-cublas")]
    impl AugurCudaQwen35PrefillLinearChunkArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_QWEN35_PREFILL_LINEAR_CHUNK_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                tokens: 0,
                reserved0: 0,
                qkv_f32: 0,
                qkv_len_bytes: 0,
                z_f32: 0,
                z_len_bytes: 0,
                conv_weights_f32: 0,
                conv_weights_len_bytes: 0,
                conv_history_f32: 0,
                conv_history_len_bytes: 0,
                alpha_raw_f32: 0,
                alpha_raw_len_bytes: 0,
                beta_raw_f32: 0,
                beta_raw_len_bytes: 0,
                dt_bias_f32: 0,
                dt_bias_len_bytes: 0,
                ssm_a_f32: 0,
                ssm_a_len_bytes: 0,
                state_f32: 0,
                state_len_bytes: 0,
                norm_weight_f32: 0,
                norm_weight_len_bytes: 0,
                activated_f32: 0,
                activated_len_bytes: 0,
                recurrent_f32: 0,
                recurrent_len_bytes: 0,
                output_f32: 0,
                output_len_bytes: 0,
                epsilon_bits: 0,
                reserved1: 0,
            }
        }
    }

    #[cfg(feature = "qwen35-prefill-cublas")]
    impl Default for AugurCudaQwen35PrefillLinearChunkArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    #[cfg(feature = "qwen35-prefill-cublas")]
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaQwen35PrefillFullAttentionArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub q_full_f32: AugurCudaDevicePtr,
        pub q_full_len_bytes: u64,
        pub k_raw_f32: AugurCudaDevicePtr,
        pub k_raw_len_bytes: u64,
        pub value_f32: AugurCudaDevicePtr,
        pub value_len_bytes: u64,
        pub q_norm_weight_f32: AugurCudaDevicePtr,
        pub q_norm_weight_len_bytes: u64,
        pub k_norm_weight_f32: AugurCudaDevicePtr,
        pub k_norm_weight_len_bytes: u64,
        pub rope_cos_f32: AugurCudaDevicePtr,
        pub rope_cos_len_bytes: u64,
        pub rope_sin_f32: AugurCudaDevicePtr,
        pub rope_sin_len_bytes: u64,
        pub key_cache_f32: AugurCudaDevicePtr,
        pub key_cache_len_bytes: u64,
        pub value_cache_f32: AugurCudaDevicePtr,
        pub value_cache_len_bytes: u64,
        pub query_f32: AugurCudaDevicePtr,
        pub query_len_bytes: u64,
        pub gate_f32: AugurCudaDevicePtr,
        pub gate_len_bytes: u64,
        pub attention_f32: AugurCudaDevicePtr,
        pub attention_len_bytes: u64,
        pub gated_f32: AugurCudaDevicePtr,
        pub gated_len_bytes: u64,
        pub tokens: u32,
        pub position_base: u32,
        pub capacity: u32,
        pub query_heads: u32,
        pub kv_heads: u32,
        pub head_dim: u32,
        pub rotary_dim: u32,
        pub epsilon_bits: u32,
        pub reserved0: u32,
    }

    #[cfg(feature = "qwen35-prefill-cublas")]
    impl AugurCudaQwen35PrefillFullAttentionArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_QWEN35_PREFILL_FULL_ATTENTION_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                q_full_f32: 0,
                q_full_len_bytes: 0,
                k_raw_f32: 0,
                k_raw_len_bytes: 0,
                value_f32: 0,
                value_len_bytes: 0,
                q_norm_weight_f32: 0,
                q_norm_weight_len_bytes: 0,
                k_norm_weight_f32: 0,
                k_norm_weight_len_bytes: 0,
                rope_cos_f32: 0,
                rope_cos_len_bytes: 0,
                rope_sin_f32: 0,
                rope_sin_len_bytes: 0,
                key_cache_f32: 0,
                key_cache_len_bytes: 0,
                value_cache_f32: 0,
                value_cache_len_bytes: 0,
                query_f32: 0,
                query_len_bytes: 0,
                gate_f32: 0,
                gate_len_bytes: 0,
                attention_f32: 0,
                attention_len_bytes: 0,
                gated_f32: 0,
                gated_len_bytes: 0,
                tokens: 0,
                position_base: 0,
                capacity: 0,
                query_heads: 24,
                kv_heads: 4,
                head_dim: 256,
                rotary_dim: 64,
                epsilon_bits: 0,
                reserved0: 0,
            }
        }
    }

    #[cfg(feature = "qwen35-prefill-cublas")]
    impl Default for AugurCudaQwen35PrefillFullAttentionArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    

    

    

    #[cfg(feature = "qwen35-prefill-cublas")]
    #[repr(C)]
    pub struct AugurCudaQwen35PrefillCublasPlanV1 {
        _private: [u8; 0],
    }

    #[cfg(feature = "qwen35-prefill-cublas")]
    const _: () = assert!(core::mem::size_of::<AugurCudaQwen35PrefillCublasPlanArgsV1>() == 64);
    #[cfg(feature = "qwen35-prefill-cublas")]
    const _: () = assert!(core::mem::align_of::<AugurCudaQwen35PrefillCublasPlanArgsV1>() == 8);
    #[cfg(feature = "qwen35-prefill-cublas")]
    const _: () = assert!(core::mem::size_of::<AugurCudaQwen35PrefillCublasProjectionArgsV1>() == 80);
    #[cfg(feature = "qwen35-prefill-cublas")]
    const _: () = assert!(core::mem::align_of::<AugurCudaQwen35PrefillCublasProjectionArgsV1>() == 8);
    #[cfg(feature = "qwen35-prefill-cublas")]
    const _: () = assert!(core::mem::size_of::<AugurCudaQwen35PrefillRmsnormArgsV1>() == 72);
    #[cfg(feature = "qwen35-prefill-cublas")]
    const _: () = assert!(core::mem::align_of::<AugurCudaQwen35PrefillRmsnormArgsV1>() == 8);
    #[cfg(feature = "qwen35-prefill-cublas")]
    const _: () = assert!(core::mem::size_of::<AugurCudaQwen35PrefillLinearChunkArgsV1>() == 232);
    #[cfg(feature = "qwen35-prefill-cublas")]
    const _: () = assert!(core::mem::align_of::<AugurCudaQwen35PrefillLinearChunkArgsV1>() == 8);
    #[cfg(feature = "qwen35-prefill-cublas")]
    const _: () = assert!(core::mem::size_of::<AugurCudaQwen35PrefillFullAttentionArgsV1>() == 256);
    #[cfg(feature = "qwen35-prefill-cublas")]
    const _: () = assert!(core::mem::align_of::<AugurCudaQwen35PrefillFullAttentionArgsV1>() == 8);
    
    

    #[cfg(all(
        feature = "qwen35-prefill-cublas",
        feature = "qwen35-kv8-full-attention"
    ))]
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaQwen35PrefillKv8ChunkAttentionArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub q_full_f32: AugurCudaDevicePtr,
        pub q_full_len_bytes: u64,
        pub k_raw_f32: AugurCudaDevicePtr,
        pub k_raw_len_bytes: u64,
        pub value_f32: AugurCudaDevicePtr,
        pub value_len_bytes: u64,
        pub q_norm_weight_f32: AugurCudaDevicePtr,
        pub q_norm_weight_len_bytes: u64,
        pub k_norm_weight_f32: AugurCudaDevicePtr,
        pub k_norm_weight_len_bytes: u64,
        pub rope_cos_f32: AugurCudaDevicePtr,
        pub rope_cos_len_bytes: u64,
        pub rope_sin_f32: AugurCudaDevicePtr,
        pub rope_sin_len_bytes: u64,
        pub key_work_f32: AugurCudaDevicePtr,
        pub key_work_len_bytes: u64,
        pub value_work_f32: AugurCudaDevicePtr,
        pub value_work_len_bytes: u64,
        pub key_cache_q8_0: AugurCudaDevicePtr,
        pub key_cache_len_bytes: u64,
        pub value_cache_q8_0: AugurCudaDevicePtr,
        pub value_cache_len_bytes: u64,
        pub query_f32: AugurCudaDevicePtr,
        pub query_len_bytes: u64,
        pub gate_f32: AugurCudaDevicePtr,
        pub gate_len_bytes: u64,
        pub attention_f32: AugurCudaDevicePtr,
        pub attention_len_bytes: u64,
        pub gated_f32: AugurCudaDevicePtr,
        pub gated_len_bytes: u64,
        pub tokens: u32,
        pub position_base: u32,
        pub capacity: u32,
        pub query_heads: u32,
        pub kv_heads: u32,
        pub head_dim: u32,
        pub rotary_dim: u32,
        pub epsilon_bits: u32,
        pub reserved0: u32,
    }

    #[cfg(all(
        feature = "qwen35-prefill-cublas",
        feature = "qwen35-kv8-full-attention"
    ))]
    impl AugurCudaQwen35PrefillKv8ChunkAttentionArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_QWEN35_PREFILL_KV8_CHUNK_ATTENTION_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                q_full_f32: 0,
                q_full_len_bytes: 0,
                k_raw_f32: 0,
                k_raw_len_bytes: 0,
                value_f32: 0,
                value_len_bytes: 0,
                q_norm_weight_f32: 0,
                q_norm_weight_len_bytes: 0,
                k_norm_weight_f32: 0,
                k_norm_weight_len_bytes: 0,
                rope_cos_f32: 0,
                rope_cos_len_bytes: 0,
                rope_sin_f32: 0,
                rope_sin_len_bytes: 0,
                key_work_f32: 0,
                key_work_len_bytes: 0,
                value_work_f32: 0,
                value_work_len_bytes: 0,
                key_cache_q8_0: 0,
                key_cache_len_bytes: 0,
                value_cache_q8_0: 0,
                value_cache_len_bytes: 0,
                query_f32: 0,
                query_len_bytes: 0,
                gate_f32: 0,
                gate_len_bytes: 0,
                attention_f32: 0,
                attention_len_bytes: 0,
                gated_f32: 0,
                gated_len_bytes: 0,
                tokens: 0,
                position_base: 0,
                capacity: 0,
                query_heads: AUGUR_CUDA_QWEN35_FULL_ATTENTION_QUERY_HEADS_V1 as u32,
                kv_heads: AUGUR_CUDA_QWEN35_FULL_ATTENTION_KV_HEADS_V1 as u32,
                head_dim: AUGUR_CUDA_QWEN35_FULL_ATTENTION_HEAD_DIM_V1 as u32,
                rotary_dim: AUGUR_CUDA_QWEN35_FULL_ATTENTION_ROTARY_DIM_V1 as u32,
                epsilon_bits: 0,
                reserved0: 0,
            }
        }
    }

    #[cfg(all(
        feature = "qwen35-prefill-cublas",
        feature = "qwen35-kv8-full-attention"
    ))]
    impl Default for AugurCudaQwen35PrefillKv8ChunkAttentionArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    #[cfg(all(
        feature = "qwen35-prefill-cublas",
        feature = "qwen35-kv8-full-attention"
    ))]
    const _: () =
        assert!(core::mem::size_of::<AugurCudaQwen35PrefillKv8ChunkAttentionArgsV1>() == 288);
    #[cfg(all(
        feature = "qwen35-prefill-cublas",
        feature = "qwen35-kv8-full-attention"
    ))]
    const _: () = assert!(core::mem::align_of::<AugurCudaQwen35PrefillKv8ChunkAttentionArgsV1>() == 8);

    pub const AUGUR_CUDA_BF16_TOKEN_ROW_GATHER_ARGS_VERSION_V1: u32 = 1;

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaBf16TokenRowGatherArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub weight: AugurCudaDevicePtr,
        pub weight_len_bytes: u64,
        pub token_source_u32: AugurCudaDevicePtr,
        pub token_source_len_bytes: u64,
        pub output_f32: AugurCudaDevicePtr,
        pub output_len_bytes: u64,
        pub rows: u32,
        pub cols: u32,
    }
    impl AugurCudaBf16TokenRowGatherArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_BF16_TOKEN_ROW_GATHER_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                weight: 0,
                weight_len_bytes: 0,
                token_source_u32: 0,
                token_source_len_bytes: 0,
                output_f32: 0,
                output_len_bytes: 0,
                rows: 0,
                cols: 0,
            }
        }
    }
    impl Default for AugurCudaBf16TokenRowGatherArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    pub const AUGUR_CUDA_M95_ROUTER_GATHER_ARGS_VERSION_V1: u32 = 1;

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct AugurCudaM95RouterGatherArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub logits_f32: AugurCudaDevicePtr,
        pub logits_len_bytes: u64,
        pub bias_f32: AugurCudaDevicePtr,
        pub bias_len_bytes: u64,
        pub output_ids_u16: AugurCudaDevicePtr,
        pub output_ids_len_bytes: u64,
        pub output_weights_f32: AugurCudaDevicePtr,
        pub output_weights_len_bytes: u64,
        pub counts_u32: AugurCudaDevicePtr,
        pub counts_len_bytes: u64,
        pub offsets_u32: AugurCudaDevicePtr,
        pub offsets_len_bytes: u64,
        pub gathered_tokens_u16: AugurCudaDevicePtr,
        pub gathered_tokens_len_bytes: u64,
        pub gathered_weights_f32: AugurCudaDevicePtr,
        pub gathered_weights_len_bytes: u64,
        pub status_i32: AugurCudaDevicePtr,
        pub status_len_bytes: u64,
        pub tokens: u32,
        pub experts: u32,
        pub top_k: u32,
        pub routed_scale: f32,
        pub reserved: u32,
    }

    impl AugurCudaM95RouterGatherArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_M95_ROUTER_GATHER_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                logits_f32: 0,
                logits_len_bytes: 0,
                bias_f32: 0,
                bias_len_bytes: 0,
                output_ids_u16: 0,
                output_ids_len_bytes: 0,
                output_weights_f32: 0,
                output_weights_len_bytes: 0,
                counts_u32: 0,
                counts_len_bytes: 0,
                offsets_u32: 0,
                offsets_len_bytes: 0,
                gathered_tokens_u16: 0,
                gathered_tokens_len_bytes: 0,
                gathered_weights_f32: 0,
                gathered_weights_len_bytes: 0,
                status_i32: 0,
                status_len_bytes: 0,
                tokens: 0,
                experts: 256,
                top_k: 6,
                routed_scale: 1.5,
                reserved: 0,
            }
        }
    }

    impl Default for AugurCudaM95RouterGatherArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    pub const AUGUR_CUDA_M102_DECODE_ROUTER_ARGS_VERSION_V1: u32 = 1;

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct AugurCudaM102DecodeRouterArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub weight_bf16: AugurCudaDevicePtr,
        pub weight_len_bytes: u64,
        pub input_f32: AugurCudaDevicePtr,
        pub input_len_bytes: u64,
        pub bias_f32: AugurCudaDevicePtr,
        pub bias_len_bytes: u64,
        pub logits_f32: AugurCudaDevicePtr,
        pub logits_len_bytes: u64,
        pub output_ids_u16: AugurCudaDevicePtr,
        pub output_ids_len_bytes: u64,
        pub output_weights_f32: AugurCudaDevicePtr,
        pub output_weights_len_bytes: u64,
        pub counts_u32: AugurCudaDevicePtr,
        pub counts_len_bytes: u64,
        pub offsets_u32: AugurCudaDevicePtr,
        pub offsets_len_bytes: u64,
        pub gathered_tokens_u16: AugurCudaDevicePtr,
        pub gathered_tokens_len_bytes: u64,
        pub gathered_weights_f32: AugurCudaDevicePtr,
        pub gathered_weights_len_bytes: u64,
        pub status_i32: AugurCudaDevicePtr,
        pub status_len_bytes: u64,
        pub rows: u32,
        pub cols: u32,
        pub experts: u32,
        pub top_k: u32,
        pub routed_scale: f32,
        pub reserved: u32,
    }

    impl AugurCudaM102DecodeRouterArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_M102_DECODE_ROUTER_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                weight_bf16: 0,
                weight_len_bytes: 0,
                input_f32: 0,
                input_len_bytes: 0,
                bias_f32: 0,
                bias_len_bytes: 0,
                logits_f32: 0,
                logits_len_bytes: 0,
                output_ids_u16: 0,
                output_ids_len_bytes: 0,
                output_weights_f32: 0,
                output_weights_len_bytes: 0,
                counts_u32: 0,
                counts_len_bytes: 0,
                offsets_u32: 0,
                offsets_len_bytes: 0,
                gathered_tokens_u16: 0,
                gathered_tokens_len_bytes: 0,
                gathered_weights_f32: 0,
                gathered_weights_len_bytes: 0,
                status_i32: 0,
                status_len_bytes: 0,
                rows: 256,
                cols: 4096,
                experts: 256,
                top_k: 6,
                routed_scale: 1.5,
                reserved: 0,
            }
        }
    }

    impl Default for AugurCudaM102DecodeRouterArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    pub const AUGUR_CUDA_M111_SPARSE_ATTENTION_F32_ARGS_VERSION_V1: u32 = 1;

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct AugurCudaM111SparseAttentionF32ArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub query_f32: AugurCudaDevicePtr,
        pub query_len_bytes: u64,
        pub kv_f32: AugurCudaDevicePtr,
        pub kv_len_bytes: u64,
        pub sink_f32: AugurCudaDevicePtr,
        pub sink_len_bytes: u64,
        pub indices_i32: AugurCudaDevicePtr,
        pub indices_len_bytes: u64,
        pub rope_cos_f32: AugurCudaDevicePtr,
        pub rope_cos_len_bytes: u64,
        pub rope_sin_f32: AugurCudaDevicePtr,
        pub rope_sin_len_bytes: u64,
        pub output_f32: AugurCudaDevicePtr,
        pub output_len_bytes: u64,
        pub tokens: u32,
        pub heads: u32,
        pub head_dim: u32,
        pub selected: u32,
    }

    impl AugurCudaM111SparseAttentionF32ArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_M111_SPARSE_ATTENTION_F32_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                query_f32: 0,
                query_len_bytes: 0,
                kv_f32: 0,
                kv_len_bytes: 0,
                sink_f32: 0,
                sink_len_bytes: 0,
                indices_i32: 0,
                indices_len_bytes: 0,
                rope_cos_f32: 0,
                rope_cos_len_bytes: 0,
                rope_sin_f32: 0,
                rope_sin_len_bytes: 0,
                output_f32: 0,
                output_len_bytes: 0,
                tokens: 8,
                heads: 64,
                head_dim: 512,
                selected: 10,
            }
        }
    }

    impl Default for AugurCudaM111SparseAttentionF32ArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    pub const AUGUR_CUDA_M131_STATEFUL_ATTENTION_F32_ARGS_VERSION_V1: u32 = 1;

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct AugurCudaM131StatefulAttentionF32ArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub query_f32: AugurCudaDevicePtr,
        pub query_len_bytes: u64,
        pub current_kv_f32: AugurCudaDevicePtr,
        pub current_kv_len_bytes: u64,
        pub kv_window_f32: AugurCudaDevicePtr,
        pub kv_window_len_bytes: u64,
        pub sink_f32: AugurCudaDevicePtr,
        pub sink_len_bytes: u64,
        pub indices_i32: AugurCudaDevicePtr,
        pub indices_len_bytes: u64,
        pub rope_cos_f32: AugurCudaDevicePtr,
        pub rope_cos_len_bytes: u64,
        pub rope_sin_f32: AugurCudaDevicePtr,
        pub rope_sin_len_bytes: u64,
        pub output_f32: AugurCudaDevicePtr,
        pub output_len_bytes: u64,
        pub position: u32,
        pub visible_kv: u32,
        pub heads: u32,
        pub head_dim: u32,
        pub selected: u32,
        pub reserved: u32,
    }
    impl AugurCudaM131StatefulAttentionF32ArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_M131_STATEFUL_ATTENTION_F32_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                query_f32: 0,
                query_len_bytes: 0,
                current_kv_f32: 0,
                current_kv_len_bytes: 0,
                kv_window_f32: 0,
                kv_window_len_bytes: 0,
                sink_f32: 0,
                sink_len_bytes: 0,
                indices_i32: 0,
                indices_len_bytes: 0,
                rope_cos_f32: 0,
                rope_cos_len_bytes: 0,
                rope_sin_f32: 0,
                rope_sin_len_bytes: 0,
                output_f32: 0,
                output_len_bytes: 0,
                position: 0,
                visible_kv: 1,
                heads: 64,
                head_dim: 512,
                selected: 10,
                reserved: 0,
            }
        }
    }
    impl Default for AugurCudaM131StatefulAttentionF32ArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    pub const AUGUR_CUDA_WEIGHTED_ACCUM_F32_ARGS_VERSION_V1: u32 = 1;

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaWeightedAccumF32ArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub input_f32: AugurCudaDevicePtr,
        pub input_len_bytes: u64,
        pub output_f32: AugurCudaDevicePtr,
        pub output_len_bytes: u64,
        pub elements: u32,
        pub weight_bits: u32,
        pub replace: u32,
        pub reserved: u32,
    }

    impl AugurCudaWeightedAccumF32ArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_WEIGHTED_ACCUM_F32_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                input_f32: 0,
                input_len_bytes: 0,
                output_f32: 0,
                output_len_bytes: 0,
                elements: 0,
                weight_bits: 0,
                replace: 0,
                reserved: 0,
            }
        }
    }

    impl Default for AugurCudaWeightedAccumF32ArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    pub const AUGUR_CUDA_DFLASH_GROUPED_CONV_F32_ARGS_VERSION_V1: u32 = 1;

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaDflashGroupedConvF32ArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub hidden_f32: AugurCudaDevicePtr,
        pub hidden_len_bytes: u64,
        pub dynamic_f32: AugurCudaDevicePtr,
        pub dynamic_len_bytes: u64,
        pub base_f32: AugurCudaDevicePtr,
        pub base_len_bytes: u64,
        pub output_f32: AugurCudaDevicePtr,
        pub output_len_bytes: u64,
        pub hidden_elements: u32,
        pub groups: u32,
        pub group_size: u32,
        pub kernel_size: u32,
        pub side: u32,
        pub reserved: u32,
    }

    impl AugurCudaDflashGroupedConvF32ArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_DFLASH_GROUPED_CONV_F32_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                hidden_f32: 0,
                hidden_len_bytes: 0,
                dynamic_f32: 0,
                dynamic_len_bytes: 0,
                base_f32: 0,
                base_len_bytes: 0,
                output_f32: 0,
                output_len_bytes: 0,
                hidden_elements: 0,
                groups: 0,
                group_size: 0,
                kernel_size: 0,
                side: 0,
                reserved: 0,
            }
        }
    }

    impl Default for AugurCudaDflashGroupedConvF32ArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    pub const AUGUR_CUDA_DFLASH_GROUPED_CONV_HISTORY_F32_ARGS_VERSION_V1: u32 = 1;

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaDflashGroupedConvHistoryF32ArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub hidden_f32: AugurCudaDevicePtr,
        pub hidden_len_bytes: u64,
        pub history_f32: AugurCudaDevicePtr,
        pub history_len_bytes: u64,
        pub dynamic_f32: AugurCudaDevicePtr,
        pub dynamic_len_bytes: u64,
        pub base_f32: AugurCudaDevicePtr,
        pub base_len_bytes: u64,
        pub output_f32: AugurCudaDevicePtr,
        pub output_len_bytes: u64,
        pub hidden_elements: u32,
        pub groups: u32,
        pub group_size: u32,
        pub kernel_size: u32,
        pub side: u32,
        pub reserved: u32,
    }

    impl AugurCudaDflashGroupedConvHistoryF32ArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_DFLASH_GROUPED_CONV_HISTORY_F32_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                hidden_f32: 0,
                hidden_len_bytes: 0,
                history_f32: 0,
                history_len_bytes: 0,
                dynamic_f32: 0,
                dynamic_len_bytes: 0,
                base_f32: 0,
                base_len_bytes: 0,
                output_f32: 0,
                output_len_bytes: 0,
                hidden_elements: 0,
                groups: 0,
                group_size: 0,
                kernel_size: 0,
                side: 0,
                reserved: 0,
            }
        }
    }

    impl Default for AugurCudaDflashGroupedConvHistoryF32ArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    pub const AUGUR_CUDA_DFLASH_GROUPED_CONV_HISTORY_ROWS_F32_ARGS_VERSION_V2: u32 = 2;

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaDflashGroupedConvHistoryRowsF32ArgsV2 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub hidden_f32: AugurCudaDevicePtr,
        pub hidden_len_bytes: u64,
        pub zero_f32: AugurCudaDevicePtr,
        pub zero_len_bytes: u64,
        pub dynamic_f32: AugurCudaDevicePtr,
        pub dynamic_len_bytes: u64,
        pub base_f32: AugurCudaDevicePtr,
        pub base_len_bytes: u64,
        pub output_f32: AugurCudaDevicePtr,
        pub output_len_bytes: u64,
        pub rows: u32,
        pub hidden_elements: u32,
        pub groups: u32,
        pub group_size: u32,
        pub kernel_size: u32,
        pub side: u32,
        pub reserved: u32,
    }

    impl AugurCudaDflashGroupedConvHistoryRowsF32ArgsV2 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_DFLASH_GROUPED_CONV_HISTORY_ROWS_F32_ARGS_VERSION_V2,
                struct_size: core::mem::size_of::<Self>() as u32,
                hidden_f32: 0,
                hidden_len_bytes: 0,
                zero_f32: 0,
                zero_len_bytes: 0,
                dynamic_f32: 0,
                dynamic_len_bytes: 0,
                base_f32: 0,
                base_len_bytes: 0,
                output_f32: 0,
                output_len_bytes: 0,
                rows: 0,
                hidden_elements: 0,
                groups: 0,
                group_size: 0,
                kernel_size: 0,
                side: 0,
                reserved: 0,
            }
        }
    }

    impl Default for AugurCudaDflashGroupedConvHistoryRowsF32ArgsV2 {
        fn default() -> Self {
            Self::new()
        }
    }

    pub const AUGUR_CUDA_COMPENSATED_ACCUM_F32_ARGS_VERSION_V1: u32 = 1;

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaCompensatedAccumF32ArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub input_f32: AugurCudaDevicePtr,
        pub input_len_bytes: u64,
        pub output_f32: AugurCudaDevicePtr,
        pub output_len_bytes: u64,
        pub compensation_f32: AugurCudaDevicePtr,
        pub compensation_len_bytes: u64,
        pub elements: u32,
        pub weight_bits: u32,
        pub replace: u32,
        pub reserved: u32,
    }

    impl AugurCudaCompensatedAccumF32ArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_COMPENSATED_ACCUM_F32_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                input_f32: 0,
                input_len_bytes: 0,
                output_f32: 0,
                output_len_bytes: 0,
                compensation_f32: 0,
                compensation_len_bytes: 0,
                elements: 0,
                weight_bits: 0,
                replace: 0,
                reserved: 0,
            }
        }
    }

    impl Default for AugurCudaCompensatedAccumF32ArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    pub const AUGUR_CUDA_RMSNORM_F32_ARGS_VERSION_V1: u32 = 1;
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaRmsnormF32ArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub input: AugurCudaDevicePtr,
        pub input_len_bytes: u64,
        pub weight: AugurCudaDevicePtr,
        pub weight_len_bytes: u64,
        pub output: AugurCudaDevicePtr,
        pub output_len_bytes: u64,
        pub elements: u32,
        pub epsilon_bits: u32,
    }
    impl AugurCudaRmsnormF32ArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_RMSNORM_F32_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                input: 0,
                input_len_bytes: 0,
                weight: 0,
                weight_len_bytes: 0,
                output: 0,
                output_len_bytes: 0,
                elements: 0,
                epsilon_bits: 0,
            }
        }
    }
    impl Default for AugurCudaRmsnormF32ArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    pub const AUGUR_CUDA_SPLIT_RMSNORM_F32_ARGS_VERSION_V1: u32 = 1;
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaSplitRmsnormF32ArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub input: AugurCudaDevicePtr,
        pub input_len_bytes: u64,
        pub weight: AugurCudaDevicePtr,
        pub weight_len_bytes: u64,
        pub output: AugurCudaDevicePtr,
        pub output_len_bytes: u64,
        pub scale_workspace: AugurCudaDevicePtr,
        pub scale_workspace_len_bytes: u64,
        pub elements: u32,
        pub epsilon_bits: u32,
    }
    impl AugurCudaSplitRmsnormF32ArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_SPLIT_RMSNORM_F32_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                input: 0,
                input_len_bytes: 0,
                weight: 0,
                weight_len_bytes: 0,
                output: 0,
                output_len_bytes: 0,
                scale_workspace: 0,
                scale_workspace_len_bytes: 0,
                elements: 0,
                epsilon_bits: 0,
            }
        }
    }
    impl Default for AugurCudaSplitRmsnormF32ArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    pub const AUGUR_CUDA_ROPE_F32_ARGS_VERSION_V1: u32 = 1;
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaRopeF32ArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub input: AugurCudaDevicePtr,
        pub input_len_bytes: u64,
        pub output: AugurCudaDevicePtr,
        pub output_len_bytes: u64,
        pub cos: AugurCudaDevicePtr,
        pub cos_len_bytes: u64,
        pub sin: AugurCudaDevicePtr,
        pub sin_len_bytes: u64,
        pub elements: u32,
        pub dimensions: u32,
    }
    impl AugurCudaRopeF32ArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_ROPE_F32_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                input: 0,
                input_len_bytes: 0,
                output: 0,
                output_len_bytes: 0,
                cos: 0,
                cos_len_bytes: 0,
                sin: 0,
                sin_len_bytes: 0,
                elements: 0,
                dimensions: 0,
            }
        }
    }
    impl Default for AugurCudaRopeF32ArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    pub const AUGUR_CUDA_DFLASH_ROPE_F32_ARGS_VERSION_V1: u32 = 1;
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct AugurCudaDflashRopeF32ArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub input: AugurCudaDevicePtr,
        pub input_len_bytes: u64,
        pub output: AugurCudaDevicePtr,
        pub output_len_bytes: u64,
        pub elements: u32,
        pub dimensions: u32,
        pub rotary_dimensions: u32,
        pub reserved: u32,
        pub position: u64,
        pub theta: f32,
    }
    impl AugurCudaDflashRopeF32ArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_DFLASH_ROPE_F32_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                input: 0,
                input_len_bytes: 0,
                output: 0,
                output_len_bytes: 0,
                elements: 0,
                dimensions: 0,
                rotary_dimensions: 0,
                reserved: 0,
                position: 0,
                theta: 0.0,
            }
        }
    }
    impl Default for AugurCudaDflashRopeF32ArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    pub const AUGUR_CUDA_DFLASH_CACHE_ATTENTION_F32_ARGS_VERSION_V1: u32 = 1;
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct AugurCudaDflashCacheAttentionF32ArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub query: AugurCudaDevicePtr,
        pub query_len_bytes: u64,
        pub key_cache: AugurCudaDevicePtr,
        pub key_cache_len_bytes: u64,
        pub value_cache: AugurCudaDevicePtr,
        pub value_cache_len_bytes: u64,
        pub output: AugurCudaDevicePtr,
        pub output_len_bytes: u64,
        pub capacity_positions: u32,
        pub visible_position_start: u32,
        pub visible_position_end: u32,
        pub query_heads: u32,
        pub kv_heads: u32,
        pub head_dim: u32,
        pub reserved: u32,
        pub scale: f32,
    }
    impl AugurCudaDflashCacheAttentionF32ArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_DFLASH_CACHE_ATTENTION_F32_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                query: 0,
                query_len_bytes: 0,
                key_cache: 0,
                key_cache_len_bytes: 0,
                value_cache: 0,
                value_cache_len_bytes: 0,
                output: 0,
                output_len_bytes: 0,
                capacity_positions: 0,
                visible_position_start: 0,
                visible_position_end: 0,
                query_heads: 0,
                kv_heads: 0,
                head_dim: 0,
                reserved: 0,
                scale: 0.0,
            }
        }
    }
    impl Default for AugurCudaDflashCacheAttentionF32ArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }
    const _: () = assert!(core::mem::size_of::<AugurCudaDflashCacheAttentionF32ArgsV1>() == 104);
    const _: () = assert!(core::mem::align_of::<AugurCudaDflashCacheAttentionF32ArgsV1>() == 8);

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct AugurCudaDflashBlockAttentionF32ArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub query: AugurCudaDevicePtr,
        pub query_len_bytes: u64,
        pub key: AugurCudaDevicePtr,
        pub key_len_bytes: u64,
        pub value: AugurCudaDevicePtr,
        pub value_len_bytes: u64,
        pub output: AugurCudaDevicePtr,
        pub output_len_bytes: u64,
        pub context_len: u32,
        pub context_start: u32,
        pub block_start: u32,
        pub sliding_window: u32,
        pub block_size: u32,
        pub query_heads: u32,
        pub kv_heads: u32,
        pub head_dim: u32,
        pub scale: f32,
        pub reserved: u32,
    }
    const _: () = assert!(core::mem::size_of::<AugurCudaDflashBlockAttentionF32ArgsV1>() == 112);
    const _: () = assert!(core::mem::align_of::<AugurCudaDflashBlockAttentionF32ArgsV1>() == 8);

    pub const AUGUR_CUDA_F8E4M3_F32_MATVEC_ARGS_VERSION_V1: u32 = 1;
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AugurCudaF8E4M3F32MatvecArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub weight: AugurCudaDevicePtr,
        pub weight_len_bytes: u64,
        pub scales: AugurCudaDevicePtr,
        pub scales_len_bytes: u64,
        pub input: AugurCudaDevicePtr,
        pub input_len_bytes: u64,
        pub output: AugurCudaDevicePtr,
        pub output_len_bytes: u64,
        pub rows: u32,
        pub cols: u32,
    }
    impl AugurCudaF8E4M3F32MatvecArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_F8E4M3_F32_MATVEC_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                weight: 0,
                weight_len_bytes: 0,
                scales: 0,
                scales_len_bytes: 0,
                input: 0,
                input_len_bytes: 0,
                output: 0,
                output_len_bytes: 0,
                rows: 0,
                cols: 0,
            }
        }
    }
    impl Default for AugurCudaF8E4M3F32MatvecArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    impl AugurCudaAddF32ArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_ADD_F32_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                a: 0,
                a_len_bytes: 0,
                b: 0,
                b_len_bytes: 0,
                output: 0,
                output_len_bytes: 0,
                elements: 0,
                reserved: 0,
            }
        }
    }

    impl Default for AugurCudaAddF32ArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    impl AugurCudaGqaRepeatF32ArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_GQA_REPEAT_F32_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                input: 0,
                input_len_bytes: 0,
                output: 0,
                output_len_bytes: 0,
                source_elements: 0,
                head_dim: 0,
                repeats: 0,
                reserved: 0,
            }
        }
    }

    impl Default for AugurCudaGqaRepeatF32ArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    pub const AUGUR_CUDA_RESIDENT_MOE_F32_ARGS_VERSION_V1: u32 = 1;

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct AugurCudaResidentMoeF32ArgsV1 {
        pub abi_version: u32,
        pub struct_size: u32,
        pub weights: AugurCudaDevicePtr,
        pub weights_len_bytes: u64,
        pub input: AugurCudaDevicePtr,
        pub input_len_bytes: u64,
        pub output: AugurCudaDevicePtr,
        pub output_len_bytes: u64,
        pub router_logits: AugurCudaDevicePtr,
        pub router_logits_len_bytes: u64,
        pub selected_ids: AugurCudaDevicePtr,
        pub selected_ids_len_bytes: u64,
        pub selected_weights: AugurCudaDevicePtr,
        pub selected_weights_len_bytes: u64,
        pub scratch: AugurCudaDevicePtr,
        pub scratch_len_bytes: u64,
        pub access_counts: AugurCudaDevicePtr,
        pub access_counts_len_bytes: u64,
        pub tokens: u32,
        pub hidden: u32,
        pub intermediate: u32,
        pub expert_count: u32,
        pub top_k: u32,
        pub routed_scale: f32,
    }

    impl AugurCudaResidentMoeF32ArgsV1 {
        pub fn new() -> Self {
            Self {
                abi_version: AUGUR_CUDA_RESIDENT_MOE_F32_ARGS_VERSION_V1,
                struct_size: core::mem::size_of::<Self>() as u32,
                weights: 0,
                weights_len_bytes: 0,
                input: 0,
                input_len_bytes: 0,
                output: 0,
                output_len_bytes: 0,
                router_logits: 0,
                router_logits_len_bytes: 0,
                selected_ids: 0,
                selected_ids_len_bytes: 0,
                selected_weights: 0,
                selected_weights_len_bytes: 0,
                scratch: 0,
                scratch_len_bytes: 0,
                access_counts: 0,
                access_counts_len_bytes: 0,
                tokens: 0,
                hidden: 0,
                intermediate: 0,
                expert_count: 0,
                top_k: 0,
                routed_scale: 0.0,
            }
        }
    }

    impl Default for AugurCudaResidentMoeF32ArgsV1 {
        fn default() -> Self {
            Self::new()
        }
    }

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    

    

    

    

    

    

    

    

    

    

    

    

    

    
    

    

    

    
    

    
    

    
    

    
    

    
    

    
    

    
    

    
    

    

    
    

    

    
    

    

    
    

    

    

    const _: () = {
        assert!(core::mem::size_of::<AugurCudaQ4kF32DecodeArgsV1>() == 72);
        assert!(core::mem::align_of::<AugurCudaQ4kF32DecodeArgsV1>() == 8);
        assert!(core::mem::size_of::<AugurCudaQ4kF32PrefillArgsV1>() == 80);
        assert!(core::mem::align_of::<AugurCudaQ4kF32PrefillArgsV1>() == 8);
        assert!(core::mem::size_of::<AugurCudaM304R3ExpertRouteArgsV1>() == 160);
        assert!(core::mem::align_of::<AugurCudaM304R3ExpertRouteArgsV1>() == 8);
        
        
        
        
        
        
        
        
        
        assert!(core::mem::size_of::<AugurCudaM304R3ExpertGroup2ArgsV2>() == 368);
        assert!(core::mem::align_of::<AugurCudaM304R3ExpertGroup2ArgsV2>() == 8);
        assert!(core::mem::offset_of!(AugurCudaM304R3ExpertGroup2ArgsV2, rows) == 48);
        assert!(core::mem::size_of::<AugurCudaM304R3ExpertGroup4UnionEntryV3>() == 8);
        assert!(core::mem::align_of::<AugurCudaM304R3ExpertGroup4UnionEntryV3>() == 2);
        assert!(core::mem::offset_of!(AugurCudaM304R3ExpertGroup4UnionEntryV3, expert) == 0);
        assert!(core::mem::offset_of!(AugurCudaM304R3ExpertGroup4UnionEntryV3, row_mask) == 2);
        assert!(core::mem::offset_of!(AugurCudaM304R3ExpertGroup4UnionEntryV3, reserved) == 3);
        assert!(core::mem::offset_of!(AugurCudaM304R3ExpertGroup4UnionEntryV3, route_rank) == 4);
        assert!(core::mem::size_of::<AugurCudaM304R3ExpertGroup4UnionTableV3>() == 216);
        assert!(core::mem::align_of::<AugurCudaM304R3ExpertGroup4UnionTableV3>() == 4);
        assert!(core::mem::offset_of!(AugurCudaM304R3ExpertGroup4UnionTableV3, status) == 0);
        assert!(core::mem::offset_of!(AugurCudaM304R3ExpertGroup4UnionTableV3, count) == 4);
        assert!(core::mem::offset_of!(AugurCudaM304R3ExpertGroup4UnionTableV3, route_error) == 8);
        assert!(core::mem::offset_of!(AugurCudaM304R3ExpertGroup4UnionTableV3, entries) == 24);
        assert!(core::mem::size_of::<AugurCudaM304R3ExpertGroup4ArgsV3>() == 704);
        assert!(core::mem::align_of::<AugurCudaM304R3ExpertGroup4ArgsV3>() == 8);
        assert!(core::mem::offset_of!(AugurCudaM304R3ExpertGroup4ArgsV3, abi_version) == 0);
        assert!(core::mem::offset_of!(AugurCudaM304R3ExpertGroup4ArgsV3, struct_size) == 4);
        assert!(core::mem::offset_of!(AugurCudaM304R3ExpertGroup4ArgsV3, row_count) == 8);
        assert!(core::mem::offset_of!(AugurCudaM304R3ExpertGroup4ArgsV3, projection) == 12);
        assert!(core::mem::offset_of!(AugurCudaM304R3ExpertGroup4ArgsV3, group_plan_digest) == 16);
        assert!(core::mem::offset_of!(AugurCudaM304R3ExpertGroup4ArgsV3, union_table) == 48);
        assert!(core::mem::offset_of!(AugurCudaM304R3ExpertGroup4ArgsV3, union_table_len_bytes) == 56);
        assert!(core::mem::offset_of!(AugurCudaM304R3ExpertGroup4ArgsV3, rows) == 64);
        assert!(core::mem::size_of::<AugurCudaAddF32ArgsV1>() == 64);
        assert!(core::mem::align_of::<AugurCudaAddF32ArgsV1>() == 8);
        assert!(core::mem::size_of::<AugurCudaGqaRepeatF32ArgsV1>() == 56);
        assert!(core::mem::align_of::<AugurCudaGqaRepeatF32ArgsV1>() == 8);
        assert!(core::mem::size_of::<AugurCudaBf16TokenRowGatherArgsV1>() == 64);
        assert!(core::mem::align_of::<AugurCudaBf16TokenRowGatherArgsV1>() == 8);
        assert!(core::mem::size_of::<AugurCudaResidentMoeF32ArgsV1>() == 160);
        assert!(core::mem::align_of::<AugurCudaResidentMoeF32ArgsV1>() == 8);
        assert!(core::mem::size_of::<AugurCudaM95RouterGatherArgsV1>() == 176);
        assert!(core::mem::align_of::<AugurCudaM95RouterGatherArgsV1>() == 8);
        assert!(core::mem::size_of::<AugurCudaM102DecodeRouterArgsV1>() == 208);
        assert!(core::mem::align_of::<AugurCudaM102DecodeRouterArgsV1>() == 8);
        assert!(core::mem::size_of::<AugurCudaM111SparseAttentionF32ArgsV1>() == 136);
        assert!(core::mem::align_of::<AugurCudaM111SparseAttentionF32ArgsV1>() == 8);
        assert!(core::mem::size_of::<AugurCudaM131StatefulAttentionF32ArgsV1>() == 160);
        assert!(core::mem::align_of::<AugurCudaM131StatefulAttentionF32ArgsV1>() == 8);
        assert!(core::mem::size_of::<AugurCudaM283DeviceViewV1>() == 32);
        assert!(core::mem::align_of::<AugurCudaM283DeviceViewV1>() == 8);
        assert!(core::mem::size_of::<AugurCudaM283KvTransitionArgsV1>() == 256);
        assert!(core::mem::align_of::<AugurCudaM283KvTransitionArgsV1>() == 8);
        assert!(core::mem::offset_of!(AugurCudaM283DeviceViewV1, base) == 0);
        assert!(core::mem::offset_of!(AugurCudaM283DeviceViewV1, allocation_extent_bytes) == 8);
        assert!(core::mem::offset_of!(AugurCudaM283DeviceViewV1, view_offset_bytes) == 16);
        assert!(core::mem::offset_of!(AugurCudaM283DeviceViewV1, view_extent_bytes) == 24);
        assert!(core::mem::offset_of!(AugurCudaM283KvTransitionArgsV1, abi_version) == 0);
        assert!(core::mem::offset_of!(AugurCudaM283KvTransitionArgsV1, struct_size) == 4);
        assert!(core::mem::offset_of!(AugurCudaM283KvTransitionArgsV1, variant) == 8);
        assert!(core::mem::offset_of!(AugurCudaM283KvTransitionArgsV1, reserved0) == 12);
        assert!(core::mem::offset_of!(AugurCudaM283KvTransitionArgsV1, layer) == 16);
        assert!(core::mem::offset_of!(AugurCudaM283KvTransitionArgsV1, capacity) == 20);
        assert!(core::mem::offset_of!(AugurCudaM283KvTransitionArgsV1, position) == 24);
        assert!(core::mem::offset_of!(AugurCudaM283KvTransitionArgsV1, next_position) == 28);
        assert!(core::mem::offset_of!(AugurCudaM283KvTransitionArgsV1, compression_ratio) == 32);
        assert!(core::mem::offset_of!(AugurCudaM283KvTransitionArgsV1, source_start_position) == 36);
        assert!(core::mem::offset_of!(AugurCudaM283KvTransitionArgsV1, source_rows) == 40);
        assert!(core::mem::offset_of!(AugurCudaM283KvTransitionArgsV1, compressed_row_index) == 44);
        assert!(core::mem::offset_of!(AugurCudaM283KvTransitionArgsV1, compressed_position) == 48);
        assert!(
            core::mem::offset_of!(AugurCudaM283KvTransitionArgsV1, compressed_main_rows_before) == 52
        );
        assert!(
            core::mem::offset_of!(AugurCudaM283KvTransitionArgsV1, compressed_index_rows_before) == 56
        );
        assert!(core::mem::offset_of!(AugurCudaM283KvTransitionArgsV1, reserved1) == 60);
        assert!(core::mem::offset_of!(AugurCudaM283KvTransitionArgsV1, raw_current) == 64);
        assert!(core::mem::offset_of!(AugurCudaM283KvTransitionArgsV1, raw_cache) == 96);
        assert!(
            core::mem::offset_of!(AugurCudaM283KvTransitionArgsV1, compressed_main_current) == 128
        );
        assert!(core::mem::offset_of!(AugurCudaM283KvTransitionArgsV1, compressed_main_cache) == 160);
        assert!(
            core::mem::offset_of!(AugurCudaM283KvTransitionArgsV1, compressed_index_current) == 192
        );
        assert!(core::mem::offset_of!(AugurCudaM283KvTransitionArgsV1, compressed_index_cache) == 224);
        assert!(core::mem::size_of::<AugurCudaM285FillSpanArgsV1>() == 56);
        assert!(core::mem::align_of::<AugurCudaM285FillSpanArgsV1>() == 8);
        assert!(core::mem::offset_of!(AugurCudaM285FillSpanArgsV1, abi_version) == 0);
        assert!(core::mem::offset_of!(AugurCudaM285FillSpanArgsV1, struct_size) == 4);
        assert!(core::mem::offset_of!(AugurCudaM285FillSpanArgsV1, allocation_base) == 8);
        assert!(core::mem::offset_of!(AugurCudaM285FillSpanArgsV1, allocation_extent_bytes) == 16);
        assert!(core::mem::offset_of!(AugurCudaM285FillSpanArgsV1, view_offset_bytes) == 24);
        assert!(core::mem::offset_of!(AugurCudaM285FillSpanArgsV1, view_extent_bytes) == 32);
        assert!(core::mem::offset_of!(AugurCudaM285FillSpanArgsV1, pattern_seed) == 40);
        assert!(core::mem::offset_of!(AugurCudaM285FillSpanArgsV1, reserved) == 48);
        assert!(core::mem::size_of::<AugurCudaKv8DsaDeviceViewV1>() == 32);
        assert!(core::mem::align_of::<AugurCudaKv8DsaDeviceViewV1>() == 8);
        assert!(core::mem::size_of::<AugurCudaKv8DsaRowV1>() == 72);
        assert!(core::mem::align_of::<AugurCudaKv8DsaRowV1>() == 8);
        assert!(core::mem::size_of::<AugurCudaKv8DsaQuantizeArgsV1>() == 280);
        assert!(core::mem::align_of::<AugurCudaKv8DsaQuantizeArgsV1>() == 8);
        assert!(core::mem::offset_of!(AugurCudaKv8DsaQuantizeArgsV1, status) == 32);
        assert!(core::mem::offset_of!(AugurCudaKv8DsaQuantizeArgsV1, rows) == 64);
        assert!(core::mem::size_of::<AugurCudaKv8DsaQuantizeM128ArgsV1>() == 256);
        assert!(core::mem::align_of::<AugurCudaKv8DsaQuantizeM128ArgsV1>() == 8);
        assert!(core::mem::offset_of!(AugurCudaKv8DsaQuantizeM128ArgsV1, raw_source) == 32);
        assert!(core::mem::offset_of!(AugurCudaKv8DsaQuantizeM128ArgsV1, status) == 224);
        assert!(core::mem::size_of::<AugurCudaKv8DsaStatusCopyArgsV1>() == 56);
        assert!(core::mem::align_of::<AugurCudaKv8DsaStatusCopyArgsV1>() == 8);
        assert!(core::mem::offset_of!(AugurCudaKv8DsaStatusCopyArgsV1, status) == 8);
        assert!(core::mem::offset_of!(AugurCudaKv8DsaStatusCopyArgsV1, host_ptr) == 40);
        assert!(core::mem::size_of::<AugurCudaKv8DsaSelectReceiptV1>() == 48);
        assert!(core::mem::align_of::<AugurCudaKv8DsaSelectReceiptV1>() == 8);
        assert!(core::mem::offset_of!(AugurCudaKv8DsaSelectReceiptV1, reservation_epoch) == 32);
        assert!(core::mem::offset_of!(AugurCudaKv8DsaSelectReceiptV1, padding) == 40);
        assert!(core::mem::size_of::<AugurCudaKv8DsaSelectArgsV1>() == 368);
        assert!(core::mem::align_of::<AugurCudaKv8DsaSelectArgsV1>() == 8);
        assert!(core::mem::offset_of!(AugurCudaKv8DsaSelectArgsV1, reservation_epoch) == 32);
        assert!(core::mem::offset_of!(AugurCudaKv8DsaSelectArgsV1, query) == 48);
        assert!(core::mem::offset_of!(AugurCudaKv8DsaSelectArgsV1, index_rows) == 80);
        assert!(core::mem::offset_of!(AugurCudaKv8DsaSelectArgsV1, selected_ids) == 240);
        assert!(core::mem::offset_of!(AugurCudaKv8DsaSelectArgsV1, receipt) == 304);
        assert!(core::mem::offset_of!(AugurCudaKv8DsaSelectArgsV1, status) == 336);
        assert!(core::mem::size_of::<AugurCudaKv8DsaAttentionArgsV1>() == 352);
        assert!(core::mem::align_of::<AugurCudaKv8DsaAttentionArgsV1>() == 8);
        assert!(core::mem::offset_of!(AugurCudaKv8DsaAttentionArgsV1, reservation_epoch) == 48);
        assert!(core::mem::offset_of!(AugurCudaKv8DsaAttentionArgsV1, query) == 64);
        assert!(core::mem::offset_of!(AugurCudaKv8DsaAttentionArgsV1, compressed_main) == 128);
        assert!(core::mem::offset_of!(AugurCudaKv8DsaAttentionArgsV1, receipt) == 192);
        assert!(core::mem::offset_of!(AugurCudaKv8DsaAttentionArgsV1, status) == 320);
        assert!(core::mem::size_of::<AugurCudaKv8DsaAttentionSplitArgsV2>() == 400);
        assert!(core::mem::align_of::<AugurCudaKv8DsaAttentionSplitArgsV2>() == 8);
        assert!(core::mem::offset_of!(AugurCudaKv8DsaAttentionSplitArgsV2, base) == 8);
        assert!(core::mem::offset_of!(AugurCudaKv8DsaAttentionSplitArgsV2, workspace) == 360);
        assert!(core::mem::offset_of!(AugurCudaKv8DsaAttentionSplitArgsV2, split_count) == 392);
        assert!(core::mem::size_of::<AugurCudaM292HbmArgsV1>() == 88);
        assert!(core::mem::align_of::<AugurCudaM292HbmArgsV1>() == 8);
        assert!(core::mem::offset_of!(AugurCudaM292HbmArgsV1, operation) == 8);
        assert!(core::mem::offset_of!(AugurCudaM292HbmArgsV1, source_base) == 16);
        assert!(core::mem::offset_of!(AugurCudaM292HbmArgsV1, destination_base) == 48);
        assert!(core::mem::offset_of!(AugurCudaM292HbmArgsV1, repeats) == 80);
        assert!(core::mem::size_of::<AugurCudaM306K7HcFp64ArgsV1>() == 96);
        assert!(core::mem::align_of::<AugurCudaM306K7HcFp64ArgsV1>() == 8);
        assert!(core::mem::offset_of!(AugurCudaM306K7HcFp64ArgsV1, abi_version) == 0);
        assert!(core::mem::offset_of!(AugurCudaM306K7HcFp64ArgsV1, struct_size) == 4);
        assert!(core::mem::offset_of!(AugurCudaM306K7HcFp64ArgsV1, rows) == 8);
        assert!(core::mem::offset_of!(AugurCudaM306K7HcFp64ArgsV1, hidden) == 32);
        assert!(core::mem::offset_of!(AugurCudaM306K7HcFp64ArgsV1, function) == 48);
        assert!(core::mem::offset_of!(AugurCudaM306K7HcFp64ArgsV1, mix) == 64);
        assert!(core::mem::offset_of!(AugurCudaM306K7HcFp64ArgsV1, scratch) == 80);
        assert!(core::mem::size_of::<AugurCudaM306K11HcExactArgsV1>() == 96);
        assert!(core::mem::align_of::<AugurCudaM306K11HcExactArgsV1>() == 8);
        assert!(core::mem::offset_of!(AugurCudaM306K11HcExactArgsV1, abi_version) == 0);
        assert!(core::mem::offset_of!(AugurCudaM306K11HcExactArgsV1, struct_size) == 4);
        assert!(core::mem::offset_of!(AugurCudaM306K11HcExactArgsV1, rows) == 8);
        assert!(core::mem::offset_of!(AugurCudaM306K11HcExactArgsV1, hidden_elements) == 12);
        assert!(core::mem::offset_of!(AugurCudaM306K11HcExactArgsV1, outputs) == 16);
        assert!(core::mem::offset_of!(AugurCudaM306K11HcExactArgsV1, accumulators) == 20);
        assert!(core::mem::offset_of!(AugurCudaM306K11HcExactArgsV1, threads) == 24);
        assert!(core::mem::offset_of!(AugurCudaM306K11HcExactArgsV1, reserved) == 28);
        assert!(core::mem::offset_of!(AugurCudaM306K11HcExactArgsV1, hidden) == 32);
        assert!(core::mem::offset_of!(AugurCudaM306K11HcExactArgsV1, hidden_len_bytes) == 40);
        assert!(core::mem::offset_of!(AugurCudaM306K11HcExactArgsV1, function) == 48);
        assert!(core::mem::offset_of!(AugurCudaM306K11HcExactArgsV1, function_len_bytes) == 56);
        assert!(core::mem::offset_of!(AugurCudaM306K11HcExactArgsV1, mix) == 64);
        assert!(core::mem::offset_of!(AugurCudaM306K11HcExactArgsV1, mix_len_bytes) == 72);
        assert!(core::mem::offset_of!(AugurCudaM306K11HcExactArgsV1, scratch) == 80);
        assert!(core::mem::offset_of!(AugurCudaM306K11HcExactArgsV1, scratch_len_bytes) == 88);
        assert!(core::mem::size_of::<AugurCudaM306K23HcLaneArgsV1>() == 96);
        assert!(core::mem::align_of::<AugurCudaM306K23HcLaneArgsV1>() == 8);
        assert!(core::mem::offset_of!(AugurCudaM306K23HcLaneArgsV1, abi_version) == 0);
        assert!(core::mem::offset_of!(AugurCudaM306K23HcLaneArgsV1, struct_size) == 4);
        assert!(core::mem::offset_of!(AugurCudaM306K23HcLaneArgsV1, rows) == 8);
        assert!(core::mem::offset_of!(AugurCudaM306K23HcLaneArgsV1, hidden_elements) == 12);
        assert!(core::mem::offset_of!(AugurCudaM306K23HcLaneArgsV1, outputs) == 16);
        assert!(core::mem::offset_of!(AugurCudaM306K23HcLaneArgsV1, accumulators) == 20);
        assert!(core::mem::offset_of!(AugurCudaM306K23HcLaneArgsV1, threads) == 24);
        assert!(core::mem::offset_of!(AugurCudaM306K23HcLaneArgsV1, function_layout) == 28);
        assert!(core::mem::offset_of!(AugurCudaM306K23HcLaneArgsV1, hidden) == 32);
        assert!(core::mem::offset_of!(AugurCudaM306K23HcLaneArgsV1, hidden_len_bytes) == 40);
        assert!(core::mem::offset_of!(AugurCudaM306K23HcLaneArgsV1, function) == 48);
        assert!(core::mem::offset_of!(AugurCudaM306K23HcLaneArgsV1, function_len_bytes) == 56);
        assert!(core::mem::offset_of!(AugurCudaM306K23HcLaneArgsV1, mix) == 64);
        assert!(core::mem::offset_of!(AugurCudaM306K23HcLaneArgsV1, mix_len_bytes) == 72);
        assert!(core::mem::offset_of!(AugurCudaM306K23HcLaneArgsV1, scratch) == 80);
        assert!(core::mem::offset_of!(AugurCudaM306K23HcLaneArgsV1, scratch_len_bytes) == 88);
        assert!(core::mem::size_of::<AugurCudaM304R3GreedyReceiptV1>() == 64);
        assert!(core::mem::align_of::<AugurCudaM304R3GreedyReceiptV1>() == 8);
        assert!(core::mem::offset_of!(AugurCudaM304R3GreedyReceiptV1, token_id) == 0);
        assert!(core::mem::offset_of!(AugurCudaM304R3GreedyReceiptV1, top1_value_bits) == 4);
        assert!(core::mem::offset_of!(AugurCudaM304R3GreedyReceiptV1, runner_up_id) == 8);
        assert!(core::mem::offset_of!(AugurCudaM304R3GreedyReceiptV1, runner_up_value_bits) == 12);
        assert!(core::mem::offset_of!(AugurCudaM304R3GreedyReceiptV1, status) == 16);
        assert!(core::mem::offset_of!(AugurCudaM304R3GreedyReceiptV1, error_code) == 20);
        assert!(core::mem::offset_of!(AugurCudaM304R3GreedyReceiptV1, step_generation) == 24);
        assert!(core::mem::offset_of!(AugurCudaM304R3GreedyReceiptV1, plan_digest) == 32);
        assert!(core::mem::size_of::<AugurCudaM304R3GreedyReceiptArgsV1>() == 104);
        assert!(core::mem::align_of::<AugurCudaM304R3GreedyReceiptArgsV1>() == 8);
        assert!(core::mem::offset_of!(AugurCudaM304R3GreedyReceiptArgsV1, abi_version) == 0);
        assert!(core::mem::offset_of!(AugurCudaM304R3GreedyReceiptArgsV1, struct_size) == 4);
        assert!(core::mem::offset_of!(AugurCudaM304R3GreedyReceiptArgsV1, logits) == 8);
        assert!(core::mem::offset_of!(AugurCudaM304R3GreedyReceiptArgsV1, logits_len_bytes) == 16);
        assert!(core::mem::offset_of!(AugurCudaM304R3GreedyReceiptArgsV1, partials) == 24);
        assert!(core::mem::offset_of!(AugurCudaM304R3GreedyReceiptArgsV1, partials_len_bytes) == 32);
        assert!(core::mem::offset_of!(AugurCudaM304R3GreedyReceiptArgsV1, receipt) == 40);
        assert!(core::mem::offset_of!(AugurCudaM304R3GreedyReceiptArgsV1, receipt_len_bytes) == 48);
        assert!(core::mem::offset_of!(AugurCudaM304R3GreedyReceiptArgsV1, step_generation) == 56);
        assert!(core::mem::offset_of!(AugurCudaM304R3GreedyReceiptArgsV1, vocab) == 64);
        assert!(core::mem::offset_of!(AugurCudaM304R3GreedyReceiptArgsV1, partial_count) == 68);
        assert!(core::mem::offset_of!(AugurCudaM304R3GreedyReceiptArgsV1, plan_digest) == 72);
        #[cfg(feature = "qwen35-device-token-handoff")]
        {
            assert!(core::mem::size_of::<AugurCudaQwen35TokenHandoffReceiptV1>() == 72);
            assert!(core::mem::align_of::<AugurCudaQwen35TokenHandoffReceiptV1>() == 8);
            assert!(core::mem::offset_of!(AugurCudaQwen35TokenHandoffReceiptV1, token_id) == 0);
            assert!(core::mem::offset_of!(AugurCudaQwen35TokenHandoffReceiptV1, finite_count) == 16);
            assert!(core::mem::offset_of!(AugurCudaQwen35TokenHandoffReceiptV1, status) == 24);
            assert!(
                core::mem::offset_of!(AugurCudaQwen35TokenHandoffReceiptV1, step_generation) == 32
            );
            assert!(core::mem::offset_of!(AugurCudaQwen35TokenHandoffReceiptV1, plan_digest) == 40);
            assert!(core::mem::size_of::<AugurCudaQwen35TokenHandoffArgsV1>() == 104);
            assert!(core::mem::align_of::<AugurCudaQwen35TokenHandoffArgsV1>() == 8);
            assert!(core::mem::offset_of!(AugurCudaQwen35TokenHandoffArgsV1, logits) == 8);
            assert!(core::mem::offset_of!(AugurCudaQwen35TokenHandoffArgsV1, partials) == 24);
            assert!(core::mem::offset_of!(AugurCudaQwen35TokenHandoffArgsV1, receipt) == 40);
            assert!(core::mem::offset_of!(AugurCudaQwen35TokenHandoffArgsV1, step_generation) == 56);
            assert!(core::mem::offset_of!(AugurCudaQwen35TokenHandoffArgsV1, vocab) == 64);
            assert!(core::mem::offset_of!(AugurCudaQwen35TokenHandoffArgsV1, plan_digest) == 72);
            assert!(core::mem::size_of::<AugurCudaQwen35DflashAcceptanceReceiptV1>() == 64);
            assert!(core::mem::align_of::<AugurCudaQwen35DflashAcceptanceReceiptV1>() == 8);
            assert!(
                core::mem::offset_of!(AugurCudaQwen35DflashAcceptanceReceiptV1, draft_tokens) == 0
            );
            assert!(
                core::mem::offset_of!(AugurCudaQwen35DflashAcceptanceReceiptV1, step_generation) == 24
            );
            assert!(
                core::mem::offset_of!(AugurCudaQwen35DflashAcceptanceReceiptV1, plan_digest) == 32
            );
            assert!(core::mem::size_of::<AugurCudaQwen35DflashAcceptanceArgsV1>() == 104);
            assert!(core::mem::align_of::<AugurCudaQwen35DflashAcceptanceArgsV1>() == 8);
            assert!(core::mem::offset_of!(AugurCudaQwen35DflashAcceptanceArgsV1, draft_ids) == 16);
            assert!(core::mem::offset_of!(AugurCudaQwen35DflashAcceptanceArgsV1, target_ids) == 32);
            assert!(core::mem::offset_of!(AugurCudaQwen35DflashAcceptanceArgsV1, receipt) == 48);
            assert!(core::mem::offset_of!(AugurCudaQwen35DflashAcceptanceArgsV1, plan_digest) == 72);
            assert!(core::mem::size_of::<AugurCudaQwen35DflashLogitsAcceptanceArgsV1>() == 128);
            assert!(core::mem::align_of::<AugurCudaQwen35DflashLogitsAcceptanceArgsV1>() == 8);
            assert!(
                core::mem::offset_of!(AugurCudaQwen35DflashLogitsAcceptanceArgsV1, draft_ids) == 16
            );
            assert!(
                core::mem::offset_of!(AugurCudaQwen35DflashLogitsAcceptanceArgsV1, target_logits)
                    == 32
            );
            assert!(
                core::mem::offset_of!(AugurCudaQwen35DflashLogitsAcceptanceArgsV1, target_ids) == 48
            );
            assert!(core::mem::offset_of!(AugurCudaQwen35DflashLogitsAcceptanceArgsV1, receipt) == 64);
            assert!(
                core::mem::offset_of!(AugurCudaQwen35DflashLogitsAcceptanceArgsV1, step_generation)
                    == 80
            );
            assert!(core::mem::offset_of!(AugurCudaQwen35DflashLogitsAcceptanceArgsV1, vocab) == 88);
            assert!(
                core::mem::offset_of!(AugurCudaQwen35DflashLogitsAcceptanceArgsV1, plan_digest) == 96
            );
        }
        assert!(core::mem::size_of::<AugurCudaM304R3ExpertSlotV1>() == 24);
        assert!(core::mem::align_of::<AugurCudaM304R3ExpertSlotV1>() == 8);
        
        assert!(core::mem::size_of::<AugurCudaM304R3RouteLayerReceiptV1>() == 32);
        assert!(core::mem::align_of::<AugurCudaM304R3RouteLayerReceiptV1>() == 8);
        assert!(core::mem::offset_of!(AugurCudaM304R3RouteLayerReceiptV1, status) == 16);
        assert!(core::mem::offset_of!(AugurCudaM304R3RouteLayerReceiptV1, selected_rank_mask) == 24);
        assert!(core::mem::size_of::<AugurCudaM304R3RouteStepReceiptV1>() == 1_440);
        assert!(core::mem::align_of::<AugurCudaM304R3RouteStepReceiptV1>() == 8);
        assert!(core::mem::offset_of!(AugurCudaM304R3RouteStepReceiptV1, layers) == 64);
        assert!(core::mem::size_of::<AugurCudaM304R3ExpertRouteArgsV1>() == 160);
        assert!(core::mem::align_of::<AugurCudaM304R3ExpertRouteArgsV1>() == 8);
        assert!(core::mem::offset_of!(AugurCudaM304R3ExpertRouteArgsV1, slot_table) == 8);
        assert!(core::mem::offset_of!(AugurCudaM304R3ExpertRouteArgsV1, ids_u16) == 24);
        assert!(core::mem::offset_of!(AugurCudaM304R3ExpertRouteArgsV1, step_receipt) == 88);
        assert!(core::mem::offset_of!(AugurCudaM304R3ExpertRouteArgsV1, step_generation) == 104);
        assert!(core::mem::offset_of!(AugurCudaM304R3ExpertRouteArgsV1, plan_digest) == 128);
        
        
        assert!(core::mem::size_of::<AugurCudaM306K4R2ExpertSlotV2>() == 88);
        assert!(core::mem::align_of::<AugurCudaM306K4R2ExpertSlotV2>() == 8);
        assert!(core::mem::offset_of!(AugurCudaM306K4R2ExpertSlotV2, w1) == 8);
        assert!(core::mem::offset_of!(AugurCudaM306K4R2ExpertSlotV2, w3) == 24);
        assert!(core::mem::offset_of!(AugurCudaM306K4R2ExpertSlotV2, w2) == 40);
        assert!(core::mem::offset_of!(AugurCudaM306K4R2ExpertSlotV2, w1_layout) == 56);
        assert!(core::mem::offset_of!(AugurCudaM306K4R2ExpertSlotV2, row_group) == 80);
        assert!(core::mem::size_of::<AugurCudaM306K4R2ExpertRouteArgsV2>() == 208);
        assert!(core::mem::align_of::<AugurCudaM306K4R2ExpertRouteArgsV2>() == 8);
        assert!(core::mem::offset_of!(AugurCudaM306K4R2ExpertRouteArgsV2, slot_table) == 8);
        assert!(core::mem::offset_of!(AugurCudaM306K4R2ExpertRouteArgsV2, ids_u16) == 24);
        assert!(core::mem::offset_of!(AugurCudaM306K4R2ExpertRouteArgsV2, step_receipt) == 88);
        assert!(core::mem::offset_of!(AugurCudaM306K4R2ExpertRouteArgsV2, step_generation) == 104);
        assert!(core::mem::offset_of!(AugurCudaM306K4R2ExpertRouteArgsV2, projection) == 124);
        assert!(core::mem::offset_of!(AugurCudaM306K4R2ExpertRouteArgsV2, input_layout) == 152);
        assert!(core::mem::offset_of!(AugurCudaM306K4R2ExpertRouteArgsV2, plan_digest) == 172);
        let input = AugurCudaM306K4R2ExpertRouteArgsV2::new_input();
        let down = AugurCudaM306K4R2ExpertRouteArgsV2::new_down();
        assert!(input.projection == AUGUR_CUDA_M306_K4_R2_PROJECTION_INPUT_V2);
        assert!(input.rows == AUGUR_CUDA_M306_K4_R2_INPUT_ROWS_V2);
        assert!(input.cols == AUGUR_CUDA_M306_K4_R2_INPUT_COLS_V2);
        assert!(down.projection == AUGUR_CUDA_M306_K4_R2_PROJECTION_DOWN_V2);
        assert!(down.rows == AUGUR_CUDA_M306_K4_R2_DOWN_ROWS_V2);
        assert!(down.cols == AUGUR_CUDA_M306_K4_R2_DOWN_COLS_V2);
        assert!(input.input_layout == 110 && input.down_layout == 111);
        assert!(core::mem::size_of::<AugurCudaM306K4R3ExpertSlotV3>() == 88);
        assert!(core::mem::align_of::<AugurCudaM306K4R3ExpertSlotV3>() == 8);
        assert!(core::mem::offset_of!(AugurCudaM306K4R3ExpertSlotV3, w1) == 8);
        assert!(core::mem::offset_of!(AugurCudaM306K4R3ExpertSlotV3, w3) == 24);
        assert!(core::mem::offset_of!(AugurCudaM306K4R3ExpertSlotV3, w2) == 40);
        assert!(core::mem::offset_of!(AugurCudaM306K4R3ExpertSlotV3, w1_layout) == 56);
        assert!(core::mem::offset_of!(AugurCudaM306K4R3ExpertSlotV3, q4_row_bytes) == 80);
        assert!(core::mem::size_of::<AugurCudaM306K4R3ValidationRecordV3>() == 288);
        assert!(core::mem::align_of::<AugurCudaM306K4R3ValidationRecordV3>() == 8);
        assert!(core::mem::offset_of!(AugurCudaM306K4R3ValidationRecordV3, step_generation) == 32);
        assert!(core::mem::offset_of!(AugurCudaM306K4R3ValidationRecordV3, ids) == 72);
        assert!(core::mem::offset_of!(AugurCudaM306K4R3ValidationRecordV3, w1) == 88);
        assert!(core::mem::offset_of!(AugurCudaM306K4R3ValidationRecordV3, w3) == 136);
        assert!(core::mem::offset_of!(AugurCudaM306K4R3ValidationRecordV3, w2) == 184);
        assert!(core::mem::offset_of!(AugurCudaM306K4R3ValidationRecordV3, input_layout) == 232);
        assert!(
            core::mem::offset_of!(AugurCudaM306K4R3ValidationRecordV3, reduction_tree_offsets) == 256
        );
        assert!(core::mem::size_of::<AugurCudaM306K4R3ExpertRouteArgsV3>() == 256);
        assert!(core::mem::align_of::<AugurCudaM306K4R3ExpertRouteArgsV3>() == 8);
        assert!(core::mem::offset_of!(AugurCudaM306K4R3ExpertRouteArgsV3, slot_table) == 8);
        assert!(core::mem::offset_of!(AugurCudaM306K4R3ExpertRouteArgsV3, ids_u16) == 24);
        assert!(core::mem::offset_of!(AugurCudaM306K4R3ExpertRouteArgsV3, step_receipt) == 88);
        assert!(core::mem::offset_of!(AugurCudaM306K4R3ExpertRouteArgsV3, validation_record) == 104);
        assert!(core::mem::offset_of!(AugurCudaM306K4R3ExpertRouteArgsV3, step_generation) == 120);
        assert!(core::mem::offset_of!(AugurCudaM306K4R3ExpertRouteArgsV3, projection) == 140);
        assert!(core::mem::offset_of!(AugurCudaM306K4R3ExpertRouteArgsV3, block_threads) == 164);
        assert!(core::mem::offset_of!(AugurCudaM306K4R3ExpertRouteArgsV3, input_layout) == 172);
        assert!(
            core::mem::offset_of!(AugurCudaM306K4R3ExpertRouteArgsV3, partial_precision_bits) == 188
        );
        assert!(
            core::mem::offset_of!(AugurCudaM306K4R3ExpertRouteArgsV3, reduction_tree_offsets) == 212
        );
        assert!(core::mem::offset_of!(AugurCudaM306K4R3ExpertRouteArgsV3, plan_digest) == 224);
        let input = AugurCudaM306K4R3ExpertRouteArgsV3::new_input();
        let down = AugurCudaM306K4R3ExpertRouteArgsV3::new_down();
        assert!(input.projection == AUGUR_CUDA_M306_K4_R3_PROJECTION_INPUT_V3);
        assert!(input.rows == AUGUR_CUDA_M306_K4_R3_INPUT_ROWS_V3);
        assert!(input.cols == AUGUR_CUDA_M306_K4_R3_INPUT_COLS_V3);
        assert!(down.projection == AUGUR_CUDA_M306_K4_R3_PROJECTION_DOWN_V3);
        assert!(down.rows == AUGUR_CUDA_M306_K4_R3_DOWN_ROWS_V3);
        assert!(down.cols == AUGUR_CUDA_M306_K4_R3_DOWN_COLS_V3);
        assert!(input.input_layout == 103 && input.down_layout == 105);
        assert!(input.reduction_tree_offsets[0] == 16);
        assert!(input.reduction_tree_offsets[1] == 8);
        assert!(input.reduction_tree_offsets[2] == 4);
        assert!(input.reduction_tree_offsets[3] == 2);
        assert!(input.reduction_tree_offsets[4] == 1);
        assert!(input.reduction_tree_offsets[5] == 0);
        assert!(input.reduction_tree_offsets[6] == 0);
        assert!(input.reduction_tree_offsets[7] == 0);
        assert!(core::mem::size_of::<AugurCudaM304R3CombinedCommitV2>() == 64);
        assert!(core::mem::align_of::<AugurCudaM304R3CombinedCommitV2>() == 8);
        assert!(core::mem::offset_of!(AugurCudaM304R3CombinedCommitV2, status) == 16);
        assert!(core::mem::offset_of!(AugurCudaM304R3CombinedCommitV2, step_generation) == 24);
        assert!(core::mem::offset_of!(AugurCudaM304R3CombinedCommitV2, plan_digest) == 32);
        assert!(core::mem::size_of::<AugurCudaM304R3CombinedCommitArgsV2>() == 128);
        assert!(core::mem::align_of::<AugurCudaM304R3CombinedCommitArgsV2>() == 8);
        assert!(core::mem::offset_of!(AugurCudaM304R3CombinedCommitArgsV2, logits) == 8);
        assert!(core::mem::offset_of!(AugurCudaM304R3CombinedCommitArgsV2, route_step_receipt) == 24);
        assert!(core::mem::offset_of!(AugurCudaM304R3CombinedCommitArgsV2, partials) == 40);
        assert!(core::mem::offset_of!(AugurCudaM304R3CombinedCommitArgsV2, commit) == 56);
        assert!(core::mem::offset_of!(AugurCudaM304R3CombinedCommitArgsV2, step_generation) == 72);
        assert!(core::mem::offset_of!(AugurCudaM304R3CombinedCommitArgsV2, expected_position) == 88);
        assert!(core::mem::offset_of!(AugurCudaM304R3CombinedCommitArgsV2, plan_digest) == 96);
        assert!(core::mem::size_of::<AugurCudaM289DeviceViewV1>() == 32);
        assert!(core::mem::align_of::<AugurCudaM289DeviceViewV1>() == 8);
        assert!(core::mem::size_of::<AugurCudaM289RopePositionArgsV1>() == 128);
        assert!(core::mem::align_of::<AugurCudaM289RopePositionArgsV1>() == 8);
        assert!(core::mem::size_of::<AugurCudaM289IndexScoreArgsV1>() == 192);
        assert!(core::mem::align_of::<AugurCudaM289IndexScoreArgsV1>() == 8);
        assert!(core::mem::size_of::<AugurCudaM289AttentionArgsV1>() == 288);
        assert!(core::mem::align_of::<AugurCudaM289AttentionArgsV1>() == 8);
        assert!(core::mem::offset_of!(AugurCudaM289RopePositionArgsV1, input) == 64);
        assert!(core::mem::offset_of!(AugurCudaM289RopePositionArgsV1, output) == 96);
        assert!(core::mem::offset_of!(AugurCudaM289IndexScoreArgsV1, query) == 64);
        assert!(core::mem::offset_of!(AugurCudaM289IndexScoreArgsV1, compressed) == 96);
        assert!(core::mem::offset_of!(AugurCudaM289IndexScoreArgsV1, head_weight) == 128);
        assert!(core::mem::offset_of!(AugurCudaM289IndexScoreArgsV1, scores) == 160);
        assert!(core::mem::offset_of!(AugurCudaM289AttentionArgsV1, query) == 64);
        assert!(core::mem::offset_of!(AugurCudaM289AttentionArgsV1, raw) == 96);
        assert!(core::mem::offset_of!(AugurCudaM289AttentionArgsV1, compressed) == 128);
        assert!(core::mem::offset_of!(AugurCudaM289AttentionArgsV1, ids) == 160);
        assert!(core::mem::offset_of!(AugurCudaM289AttentionArgsV1, sink) == 192);
        assert!(core::mem::offset_of!(AugurCudaM289AttentionArgsV1, weighted) == 224);
        assert!(core::mem::offset_of!(AugurCudaM289AttentionArgsV1, inverse) == 256);
        assert!(core::mem::size_of::<AugurCudaWeightedAccumF32ArgsV1>() == 56);
        assert!(core::mem::align_of::<AugurCudaWeightedAccumF32ArgsV1>() == 8);
    };

    

    // Hand-written bindings over native/augur_cuda_runtime.h (implemented in
    // native/augur_cuda_runtime.cu plus the per-kernel native/*.cu files).
    //
    // # Safety — contract shared by every declaration in this block
    //
    // * Status semantics: fallible entry points return `AugurCudaResultV1` by
    //   value; `status == AUGUR_CUDA_STATUS_OK` (0) is the only success value and
    //   `raw_cuda_status` carries the underlying `cudaError_t`. The native side
    //   never unwinds across this boundary; errors surface only via the return.
    // * Null handling: the native implementations reject null `stream`/`args`/
    //   out-pointers, negative ordinals, and stream-vs-ordinal mismatches with
    //   `AUGUR_CUDA_STATUS_INVALID_ARGUMENT` before any dereference (validation
    //   prologue pattern in augur_cuda_runtime.cu and each kernel .cu). NOTE: this
    //   tolerance is implemented but not promised by augur_cuda_runtime.h itself —
    //   see the unsafe-hygiene findings.
    // * Handle validity: `AugurCudaStream`/`AugurCudaEvent`/`AugurCudaGraphExec` are
    //   host-heap handles owned by the native runtime. Only null-ness is
    //   checked; a destroyed handle is still dereferenced (`stream->ordinal`),
    //   which is host use-after-free. Callers must pass either null or a live
    //   handle from the matching `*_create`/`*_capture_end_v1` call and must
    //   destroy each handle at most once.
    // * `AugurCudaDevicePtr` is an opaque device-address token (u64), not a host
    //   pointer; host code must never dereference it (augur_cuda_runtime.h,
    //   `augur_cuda_device_ptr_t` note). Kernel arg structs carry such tokens plus
    //   caller-claimed extents, audited natively before any launch.
    // * ABI layout: the `#[repr(C)]` arg structs above are pinned to the C
    //   layouts by the `const _` size/align/offset assertions in this module,
    //   and every `new()` constructor seeds the `abi_version`/`struct_size`
    //   fields that the native validators require to match.
    extern "C" {
        /// Total function: no preconditions, no pointers. Used for the
        /// ABI-version handshake against `AUGUR_CUDA_ABI_VERSION_V1`.
        pub fn augur_cuda_abi_version() -> u32;

        /// # Safety
        /// `out_count` must be null (rejected) or valid for a `c_int` write for
        /// the duration of the call; the native side writes it only on success.
        pub fn augur_cuda_device_count(out_count: *mut c_int) -> AugurCudaResultV1;

        /// # Safety
        /// `out_properties` must be null (rejected) or point to a writable
        /// struct whose `abi_version`/`struct_size` were pre-seeded, as done by
        /// `AugurCudaDevicePropertiesV1::new()`; the native side audits both
        /// before filling the remaining fields.
        pub fn augur_cuda_get_device_properties(
            ordinal: c_int,
            out_properties: *mut AugurCudaDevicePropertiesV1,
        ) -> AugurCudaResultV1;
        /// # Safety
        /// `out_uuid` must be null (rejected) or valid for writes of
        /// `out_uuid_bytes` bytes; the length is the caller's claim and cannot
        /// be checked natively beyond its minimum.
        pub fn augur_cuda_get_device_uuid_v1(
            ordinal: c_int,
            out_uuid: *mut u8,
            out_uuid_bytes: u32,
        ) -> AugurCudaResultV1;
        /// # Safety
        /// Both out-pointers must be null (rejected) or valid for a `u64`
        /// write each for the duration of the call.
        pub fn augur_cuda_mem_get_info_v1(
            ordinal: c_int,
            out_free_bytes: *mut u64,
            out_total_bytes: *mut u64,
        ) -> AugurCudaResultV1;

        /// NVTX range markers: no pointer parameters, no preconditions.
        pub fn augur_cuda_nvtx_m306_steady_token_begin_v1() -> AugurCudaResultV1;
        pub fn augur_cuda_nvtx_m306_steady_token_end_v1() -> AugurCudaResultV1;
        /// Fixed BATCH-01 row markers. `row_index` must be in `0..4`, exactly
        /// one row marker may be active, and the M306 outer range must own the
        /// calling thread. Invalid state is rejected without pushing/popping.
        pub fn augur_cuda_nvtx_batch01_b4_row_main_begin_v1(row_index: u32) -> AugurCudaResultV1;
        pub fn augur_cuda_nvtx_batch01_b4_row_main_end_v1(row_index: u32) -> AugurCudaResultV1;
        /// # Safety
        /// `out_stream` must be null (rejected) or valid for a pointer write;
        /// the native side stores a null sentinel first and a live heap handle
        /// only on `is_ok()`. Ownership transfers to the caller, who must pass
        /// the handle to `augur_cuda_stream_destroy` exactly once.
        pub fn augur_cuda_stream_create(
            ordinal: c_int,
            out_stream: *mut AugurCudaStream,
        ) -> AugurCudaResultV1;

        /// # Safety
        /// `stream` must be null (rejected) or a live handle from
        /// `augur_cuda_stream_create`; destroyed handles are dereferenced.
        pub fn augur_cuda_stream_synchronize(stream: AugurCudaStream) -> AugurCudaResultV1;

        /// # Safety
        /// Both handles must be null (rejected) or live and created on the same
        /// ordinal; the native prologue checks agreement before submitting.
        pub fn augur_cuda_stream_wait_event_v1(
            stream: AugurCudaStream,
            event: AugurCudaEvent,
        ) -> AugurCudaResultV1;

        /// # Safety
        /// Consumes the handle: `stream` must be null (rejected) or a live
        /// handle from `augur_cuda_stream_create`, not passed to any call again
        /// afterward. The native side frees the host handle on success.
        pub fn augur_cuda_stream_destroy(stream: AugurCudaStream) -> AugurCudaResultV1;

        /// # Safety
        /// `stream` must be null (rejected) or a live handle created on
        /// `ordinal`; capture state is owned natively per stream.
        pub fn augur_cuda_graph_capture_begin_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
        ) -> AugurCudaResultV1;

        /// # Safety
        /// `out_graph` must be null (rejected) or valid for a pointer write. On
        /// `is_ok()` it receives a live graph handle owned by the caller, to be
        /// released via `augur_cuda_graph_destroy_v1` exactly once.
        pub fn augur_cuda_graph_capture_end_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            out_graph: *mut AugurCudaGraphExec,
        ) -> AugurCudaResultV1;

        /// # Safety
        /// Same handle contract as `augur_cuda_graph_capture_begin_v1`.
        pub fn augur_cuda_graph_capture_abort_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
        ) -> AugurCudaResultV1;

        /// # Safety
        /// `graph` and `stream` must each be null (rejected) or live handles;
        /// the graph must have been captured on the same ordinal.
        pub fn augur_cuda_graph_launch_v1(
            ordinal: c_int,
            graph: AugurCudaGraphExec,
            stream: AugurCudaStream,
        ) -> AugurCudaResultV1;

        /// # Safety
        /// `graph` must be null (rejected) or live; `out_node_count` must be
        /// null (rejected) or valid for a `u64` write.
        pub fn augur_cuda_graph_node_count_v1(
            graph: AugurCudaGraphExec,
            out_node_count: *mut u64,
        ) -> AugurCudaResultV1;

        /// # Safety
        /// Consumes the handle from `augur_cuda_graph_capture_end_v1`; it must not
        /// be used again after this call.
        pub fn augur_cuda_graph_destroy_v1(graph: AugurCudaGraphExec) -> AugurCudaResultV1;

        /// # Safety
        /// Same out-pointer/ownership contract as `augur_cuda_stream_create`:
        /// caller owns the handle and must `augur_cuda_event_destroy` it once.
        pub fn augur_cuda_event_create(ordinal: c_int, out_event: *mut AugurCudaEvent) -> AugurCudaResultV1;

        /// # Safety
        /// Both handles must be null (rejected) or live and on the same
        /// ordinal; the native prologue checks agreement first.
        pub fn augur_cuda_event_record(event: AugurCudaEvent, stream: AugurCudaStream) -> AugurCudaResultV1;

        /// # Safety
        /// `event` must be null (rejected) or a live handle;
        /// `out_complete` must be null (rejected) or valid for a `u8` write.
        /// Both `cudaSuccess` and `cudaErrorNotReady` are returned as ABI OK;
        /// the output byte distinguishes complete (`1`) from pending (`0`).
        pub fn augur_cuda_event_query_complete_v1(
            event: AugurCudaEvent,
            out_complete: *mut u8,
        ) -> AugurCudaResultV1;

        /// # Safety
        /// `event` must be null (rejected) or a live handle.
        pub fn augur_cuda_event_synchronize(event: AugurCudaEvent) -> AugurCudaResultV1;

        /// # Safety
        /// Handles must be null (rejected) or live on the same ordinal;
        /// `out_milliseconds` must be null (rejected) or valid for an `f32`
        /// write (a 0.0 sentinel is stored before the query).
        pub fn augur_cuda_event_elapsed_ms(
            started: AugurCudaEvent,
            stopped: AugurCudaEvent,
            out_milliseconds: *mut f32,
        ) -> AugurCudaResultV1;

        /// # Safety
        /// Consumes the handle; must not be used again after this call.
        pub fn augur_cuda_event_destroy(event: AugurCudaEvent) -> AugurCudaResultV1;

        /// # Safety
        /// `out_ptr` must be null (rejected) or valid for a `u64` write; a 0
        /// sentinel is stored before allocation. The token written on success
        /// is device-owned memory of `len_bytes`; release it only through
        /// `augur_cuda_device_free` with the same `ordinal`, exactly once.
        pub fn augur_cuda_device_alloc(
            ordinal: c_int,
            len_bytes: usize,
            out_ptr: *mut AugurCudaDevicePtr,
        ) -> AugurCudaResultV1;

        /// # Safety
        /// `ptr` must be 0 (rejected) or a token from `augur_cuda_device_alloc`
        /// on the same `ordinal` that has not been freed; no in-flight work may
        /// still reference the allocation.
        pub fn augur_cuda_device_free(ordinal: c_int, ptr: AugurCudaDevicePtr) -> AugurCudaResultV1;

        /// # Safety
        /// `stream` must be a live handle for `ordinal`. `dst` must name a live
        /// allocation of at least `dst_len_bytes` on the same device, with no
        /// concurrent readers or writers. The call synchronizes `stream`
        /// before returning, so the initialized bytes are immediately visible.
        pub fn augur_cuda_memset_zero_sync_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            dst: AugurCudaDevicePtr,
            dst_len_bytes: usize,
        ) -> AugurCudaResultV1;

        /// # Safety
        /// `out_ptr` must be null (rejected) or valid for a pointer write. On
        /// success it receives page-locked HOST memory of `len_bytes`, fully
        /// initialized to zero before publication and owned by the caller
        /// until `augur_cuda_host_free_pinned_v1` on the same ordinal.
        pub fn augur_cuda_host_alloc_pinned_v1(
            ordinal: c_int,
            len_bytes: usize,
            out_ptr: *mut *mut c_void,
        ) -> AugurCudaResultV1;

        /// # Safety
        /// Both out-pointers must be valid writable slots. On success,
        /// `out_host_ptr` owns one page-locked host allocation and
        /// `out_device_ptr` is the kernel-visible address of the same bytes.
        /// Free only the host pointer with
        /// `augur_cuda_host_free_mapped_sync_v1`; never pass the device alias to
        /// either host or device release APIs.
        pub fn augur_cuda_host_alloc_mapped_v1(
            ordinal: c_int,
            len_bytes: usize,
            out_host_ptr: *mut *mut c_void,
            out_device_ptr: *mut AugurCudaDevicePtr,
        ) -> AugurCudaResultV1;

        /// # Safety
        /// `ptr` must be null (rejected) or the exact pointer returned by
        /// `augur_cuda_host_alloc_pinned_v1`, freed at most once, with no pending
        /// async copies still reading or writing it.
        pub fn augur_cuda_host_free_pinned_v1(ordinal: c_int, ptr: *mut c_void) -> AugurCudaResultV1;

        /// # Safety
        /// `ptr` must be the exact live host pointer returned by
        /// `augur_cuda_host_alloc_mapped_v1`, freed at most once. The call fences
        /// all device consumers on `ordinal` before releasing the allocation.
        pub fn augur_cuda_host_free_mapped_sync_v1(ordinal: c_int, ptr: *mut c_void)
            -> AugurCudaResultV1;

        /// # Safety
        /// ASYNCHRONOUS: the native side calls `cudaMemcpyAsync` and returns at
        /// enqueue, so `src` must stay valid and unmodified for
        /// `copy_len_bytes` until the stream has been synchronized (or an event
        /// recorded after this copy has completed) — NOT merely for the call.
        /// This lifetime requirement is not stated in augur_cuda_runtime.h (only
        /// the `_sync` variants document release-immediately); see findings.
        pub fn augur_cuda_memcpy_h2d_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            dst: AugurCudaDevicePtr,
            dst_offset_bytes: usize,
            dst_len_bytes: usize,
            src: *const c_void,
            src_len_bytes: usize,
            copy_len_bytes: usize,
        ) -> AugurCudaResultV1;

        

        /// # Safety
        /// Extent-complete synchronous successor (augur_cuda_runtime.h): `src`
        /// must be readable for `src_len_bytes` during the call only; offset,
        /// destination extent and copy length are cross-checked natively.
        pub fn augur_cuda_memcpy_h2d_sync_v2(
            ordinal: c_int,
            stream: AugurCudaStream,
            dst: AugurCudaDevicePtr,
            dst_offset_bytes: usize,
            dst_len_bytes: usize,
            src: *const u8,
            src_len_bytes: usize,
            copy_len_bytes: usize,
        ) -> AugurCudaResultV1;

        

        /// # Safety
        /// ASYNCHRONOUS: `dst` must be page-locked host memory writable for
        /// `dst_len_bytes` and must remain live and unmodified until a
        /// completion event recorded after this submission has been proven.
        /// `src` must name a live same-device allocation with the declared
        /// extent. The native call enqueues `cudaMemcpyAsync` and never waits
        /// on the stream.
        pub fn augur_cuda_memcpy_d2h_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            dst: *mut c_void,
            dst_len_bytes: usize,
            src: AugurCudaDevicePtr,
            src_offset_bytes: usize,
            src_len_bytes: usize,
            copy_len_bytes: usize,
        ) -> AugurCudaResultV1;

        /// # Safety
        /// Extent-complete synchronous successor: `dst` must be writable for
        /// `dst_len_bytes` during the call only; all extents cross-checked.
        pub fn augur_cuda_memcpy_d2h_sync_v2(
            ordinal: c_int,
            stream: AugurCudaStream,
            dst: *mut u8,
            dst_len_bytes: usize,
            src: AugurCudaDevicePtr,
            src_offset_bytes: usize,
            src_len_bytes: usize,
            copy_len_bytes: usize,
        ) -> AugurCudaResultV1;

        

        /// # Safety
        /// Device-to-device: both operands are device tokens with explicit
        /// extents, audited natively; no host memory is involved, so only the
        /// stream-handle and token-liveness contracts above apply.
        pub fn augur_cuda_memcpy_d2d_independent_batch_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            entries: *const AugurCudaD2dCopyEntryV1,
            count: u32,
        ) -> AugurCudaResultV1;

        pub fn augur_cuda_memcpy_d2d_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            dst: AugurCudaDevicePtr,
            dst_offset_bytes: usize,
            dst_len_bytes: usize,
            src: AugurCudaDevicePtr,
            src_offset_bytes: usize,
            src_len_bytes: usize,
            copy_len_bytes: usize,
        ) -> AugurCudaResultV1;

        /// # Safety
        /// Bitwise in-place row-left-shift; tail untouched. Allocation and
        /// stream must remain live, with no unordered concurrent access.
        pub fn augur_cuda_shift_rows_u32_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            data: AugurCudaDevicePtr,
            len_bytes: usize,
            row_bytes: usize,
            retained_rows: usize,
            shift_rows: usize,
        ) -> AugurCudaResultV1;

        /// # Safety
        /// Device-to-device 2D copy. Both allocation tokens and their complete
        /// extents must remain live through stream completion. The native shim
        /// checks pitches and the final addressed row before submission.
        pub fn augur_cuda_memcpy_d2d_2d_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            dst: AugurCudaDevicePtr,
            dst_offset_bytes: usize,
            dst_len_bytes: usize,
            dst_pitch_bytes: usize,
            src: AugurCudaDevicePtr,
            src_offset_bytes: usize,
            src_len_bytes: usize,
            src_pitch_bytes: usize,
            width_bytes: usize,
            height: usize,
        ) -> AugurCudaResultV1;

        // # Safety — kernel-launch family
        //
        // Every declaration below of the shape (`ordinal`, `stream`,
        // `args: *const ...ArgsV*`) shares this contract:
        // * `args` must be null (rejected) or valid for reads of the full
        //   struct for the duration of the call; the native side copies it by
        //   value at launch, so a stack temporary is sufficient and nothing is
        //   retained after return.
        // * The native validation prologue audits `abi_version`,
        //   `struct_size`, extents, alignment and (where applicable) overlap
        //   before any kernel launch; device tokens inside `args` must
        //   nevertheless denote live allocations on `ordinal` large enough for
        //   the claimed `*_len_bytes`, or submitted kernels fault device-side.
        // * `*_submit_*` variants return at enqueue: the referenced device
        //   allocations must stay alive until the stream is synchronized.
        //   Non-submit variants synchronize the stream before returning.
        pub fn augur_cuda_q4k_f32_decode_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQ4kF32DecodeArgsV1,
        ) -> AugurCudaResultV1;

        pub fn augur_cuda_q4k_f32_decode_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQ4kF32DecodeArgsV1,
        ) -> AugurCudaResultV1;

        pub fn augur_cuda_q4k_f32_prefill_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQ4kF32PrefillArgsV1,
        ) -> AugurCudaResultV1;

        pub fn augur_cuda_q4k_f32_prefill_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQ4kF32PrefillArgsV1,
        ) -> AugurCudaResultV1;

        

        pub fn augur_cuda_q5k_f32_decode_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQ5kF32DecodeArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_q5k_f32_decode_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQ5kF32DecodeArgsV1,
        ) -> AugurCudaResultV1;
        
        pub fn augur_cuda_q5k_f32_decode_grid32_v2(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQ5kF32DecodeArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_q5k_f32_decode_grid32_submit_v2(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQ5kF32DecodeArgsV1,
        ) -> AugurCudaResultV1;
        
        
        
        
        
        
        
        
        
        
        
        
        
        pub fn augur_cuda_q6k_f32_decode_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQ6kF32DecodeArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_q6k_f32_decode_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQ6kF32DecodeArgsV1,
        ) -> AugurCudaResultV1;
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        pub fn augur_cuda_m120_q4k_exact_f32_decode_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQ4kF32DecodeArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_m120_q5k_exact_f32_decode_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQ5kF32DecodeArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_m120_q6k_exact_f32_decode_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQ6kF32DecodeArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_m121_q4k_dp4a_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaM121Q4kDp4aArgsV1,
        ) -> AugurCudaResultV1;

        pub fn augur_cuda_add_f32_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaAddF32ArgsV1,
        ) -> AugurCudaResultV1;

        pub fn augur_cuda_add_f32_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaAddF32ArgsV1,
        ) -> AugurCudaResultV1;

        pub fn augur_cuda_gqa_repeat_f32_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaGqaRepeatF32ArgsV1,
        ) -> AugurCudaResultV1;

        pub fn augur_cuda_gqa_repeat_f32_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaGqaRepeatF32ArgsV1,
        ) -> AugurCudaResultV1;

        
        
        
        
        
        pub fn augur_cuda_m142_ordered_topk_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaM142OrderedTopkArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_m142_ordered_topk_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaM142OrderedTopkArgsV1,
        ) -> AugurCudaResultV1;
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        pub fn augur_cuda_m285_fill_span_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaM285FillSpanArgsV1,
        ) -> AugurCudaResultV1;
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        pub fn augur_cuda_m292_hbm_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaM292HbmArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_m292_l2_cache_bytes_v1(
            ordinal: c_int,
            out_l2_cache_bytes: *mut u64,
        ) -> AugurCudaResultV1;
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        #[cfg(feature = "qwen35-device-token-handoff")]
        pub fn augur_cuda_qwen35_token_handoff_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35TokenHandoffArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-device-token-handoff")]
        pub fn augur_cuda_qwen35_dflash_acceptance_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35DflashAcceptanceArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-device-token-handoff")]
        pub fn augur_cuda_qwen35_dflash_logits_acceptance_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35DflashLogitsAcceptanceArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-device-token-handoff")]
        pub fn augur_cuda_qwen35_vocab_top1_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35VocabTop1ArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-device-token-handoff")]
        pub fn augur_cuda_qwen35_vocab_topk_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35VocabTopkArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-device-token-handoff")]
        pub fn augur_cuda_qwen35_vocab_topk_tiled_submit_v1(
            ordinal: c_int, stream: AugurCudaStream,
            args: *const AugurCudaQwen35VocabTopkArgsV1,
            scratch: u64, scratch_len_bytes: u64, parts: u32,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-device-token-handoff")]
        pub fn augur_cuda_qwen35_dflash_selector_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35DflashSelectorArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-device-token-handoff")]
        pub fn augur_cuda_qwen35_dflash_bf16_codebooks_submit_v1(
            ordinal: c_int,stream: AugurCudaStream,args: *const AugurCudaQwen35DflashQ8SelectorArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-device-token-handoff")]
        pub fn augur_cuda_qwen35_dflash_q8_selector_phase_submit_v1(
            ordinal:c_int,stream:AugurCudaStream,args:*const AugurCudaQwen35DflashQ8SelectorArgsV1,
            after_gates:u64,after_bytes:u64,close_token:u32,
        )->AugurCudaResultV1;
        #[cfg(feature = "qwen35-device-token-handoff")]
        pub fn augur_cuda_qwen35_dflash_q8_selector_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35DflashQ8SelectorArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-device-token-handoff")]
        pub fn augur_cuda_qwen35_dflash_q8_selector_scaled_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35DflashQ8SelectorArgsV1,
            transition_scale: f32,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_qwen35_dflash_q8_selector_dp_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35DflashQ8SelectorArgsV1,
            transition_scale: f32,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_qwen35_dflash_q8_selector_tail_scaled_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35DflashQ8SelectorArgsV1,
            transition_scale: f32,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_qwen35_dflash_q8_selector_position_scaled_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35DflashQ8SelectorArgsV1,
            first_scale: f32,
            second_scale: f32,
            third_scale: f32,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_qwen35_dflash_q8_selector_probability_submit_v2(
            ordinal:c_int, stream:AugurCudaStream, args:*const AugurCudaQwen35DflashQ8SelectorArgsV1,
            q:u64, q_bytes:u64, absolute_start:u64, positions:u32,
            temperature234:f32, pair_scale234:f32,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_qwen35_dflash_q8_selector_probability_submit_v4(
            ordinal:c_int, stream:AugurCudaStream, args:*const AugurCudaQwen35DflashQ8SelectorArgsV1,
            q:u64, q_bytes:u64, absolute_start:u64, positions:u32,
            temperatures7:*const f32, pair_scales7:*const f32, seed:u64,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_qwen35_dflash_q8_selector_probability_submit_v3(
            ordinal:c_int, stream:AugurCudaStream, args:*const AugurCudaQwen35DflashQ8SelectorArgsV1,
            q:u64, q_bytes:u64, absolute_start:u64, positions:u32,
            temperatures7:*const f32, pair_scales7:*const f32,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_qwen35_dflash_q8_selector_probability_submit_v1(
            ordinal:c_int, stream:AugurCudaStream, args:*const AugurCudaQwen35DflashQ8SelectorArgsV1,
            q:u64, q_bytes:u64, absolute_start:u64, positions:u32,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_qwen35_dflash_q8_selector_early_scaled_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35DflashQ8SelectorArgsV1,
            transition_scale: f32,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_qwen35_dflash_q8_selector_rank_comp_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35DflashQ8SelectorArgsV1,
            rank_bonus: f32,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_qwen35_dflash_q8_selector_tail_rank_comp_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35DflashQ8SelectorArgsV1,
            transition_scale: f32,
            rank_bonus: f32,
        ) -> AugurCudaResultV1;
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        pub fn augur_cuda_swiglu_f32_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaSwigluF32ArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_swiglu_f32_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaSwigluF32ArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_silu_mul_f32_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaSiluMulF32ArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_silu_mul_f32_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaSiluMulF32ArgsV1,
        ) -> AugurCudaResultV1;
        
        pub fn augur_cuda_silu_mul_q8_1_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaSiluMulQ8_1ArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_silu_q8_preserved_b8_preflight_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            silu: *const AugurCudaSiluMulF32ArgsV1,
            quantize: *const AugurCudaQ8_1ActivationQuantizeArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_silu_q8_preserved_b8_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            silu: *const AugurCudaSiluMulF32ArgsV1,
            quantize: *const AugurCudaQ8_1ActivationQuantizeArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_qwen35_head_prepare_b8_preflight_v1(
            ordinal: c_int, stream: AugurCudaStream, args: *const AugurCudaQwen35HeadPrepareB8ArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_qwen35_head_prepare_b8_submit_v1(
            ordinal: c_int, stream: AugurCudaStream, args: *const AugurCudaQwen35HeadPrepareB8ArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-linear-step")]
        pub fn augur_cuda_qwen35_linear_step_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35LinearStepArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-linear-step")]
        pub fn augur_cuda_qwen35_linear_step_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35LinearStepArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-linear-step")]
        pub fn augur_cuda_qwen35_checkpoint_linear_step_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35LinearStepArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-linear-step")]
        pub fn augur_cuda_qwen35_linear_projection_postprocess_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35LinearProjectionPostprocessArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-linear-step")]
        pub fn augur_cuda_qwen35_linear_projection_postprocess_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35LinearProjectionPostprocessArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-fused-head-norm-rope")]
        pub fn augur_cuda_qwen35_fused_head_norm_rope_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35FusedHeadNormRopeArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-fused-head-norm-rope")]
        pub fn augur_cuda_qwen35_fused_head_norm_rope_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35FusedHeadNormRopeArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-fused-head-norm-rope")]
        pub fn augur_cuda_qwen35_batched_head_norm_rope_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35BatchedHeadNormRopeArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_qwen35_full_attention_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35FullAttentionArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_qwen35_full_attention_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35FullAttentionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
        pub fn augur_cuda_qwen35_full_attention_v2(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35FullAttentionArgsV2,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
        pub fn augur_cuda_qwen35_full_attention_submit_v2(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35FullAttentionArgsV2,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-kv8-full-attention")]
        pub fn augur_cuda_qwen35_kv8_full_attention_v3(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35Kv8FullAttentionArgsV3,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-kv8-full-attention")]
        pub fn augur_cuda_qwen35_kv8_full_attention_submit_v3(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35Kv8FullAttentionArgsV3,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-kv8-full-attention")]
        pub fn augur_cuda_qwen35_kv8_batched_attention_v4(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35Kv8BatchedAttentionArgsV4,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-kv8-full-attention")]
        pub fn augur_cuda_qwen35_kv8_batched_attention_submit_v4(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35Kv8BatchedAttentionArgsV4,
        ) -> AugurCudaResultV1;
        
        #[cfg(feature = "qwen35-kv8-full-attention")]
        pub fn augur_cuda_qwen35_kv8_segment_partial_submit_v2(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35Kv8SegmentPartialArgsV2,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-kv8-full-attention")]
        pub fn augur_cuda_qwen35_kv8_lse_merge_submit_v2(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const core::ffi::c_void,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-kv8-full-attention")]
        pub fn augur_cuda_qwen35_kv8_page_digest_preflight_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35Kv8PageDigestArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-kv8-full-attention")]
        pub fn augur_cuda_qwen35_kv8_page_digest_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35Kv8PageDigestArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-kv8-full-attention")]
        pub fn augur_cuda_qwen35_kv8_page_digest_parallel_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35Kv8PageDigestArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(all(
            feature = "qwen35-prefill-cublas",
            feature = "qwen35-kv8-full-attention"
        ))]
        pub fn augur_cuda_qwen35_prefill_kv8_cache_quantize_preflight_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35PrefillKv8CacheQuantizeArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(all(
            feature = "qwen35-prefill-cublas",
            feature = "qwen35-kv8-full-attention"
        ))]
        pub fn augur_cuda_qwen35_prefill_kv8_cache_quantize_prevalidated_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35PrefillKv8CacheQuantizeArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_bf16_f32_matvec_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaBf16F32MatvecArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_bf16_f32_matvec_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaBf16F32MatvecArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_nerkyor_bf16_gate_pair_f32_cta_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            a: *const AugurCudaBf16F32MatvecArgsV1,
            b: *const AugurCudaBf16F32MatvecArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_bf16_f32_matvec_rows_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaBf16F32MatvecArgsV1,
            tokens: u32,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_q8_0_f32_matvec_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQ8_0F32MatvecArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_q8_0_f32_matvec_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQ8_0F32MatvecArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_dflash_q8_context_rows_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQ8_0F32MatvecArgsV1,
            tokens: u32,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_dflash_q8_rows_b8_specialized_submit_v1(
            ordinal: i32,
            stream: AugurCudaStream,
            args: *const AugurCudaQ8_0F32MatvecArgsV1,
            tokens: u32,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_dflash_q8_rows_tile4_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQ8_0F32MatvecArgsV1,
            tokens: u32,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_dflash_norm_context_rows_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaRmsnormF32ArgsV1,
            rows: u32,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_dflash_q8_history_reuse_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQ8_0F32MatvecArgsV1,
            tokens: u32,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_dflash_q8_history_panel_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQ8_0F32MatvecArgsV1,
            tokens: u32,
            scratch: u64,
            scratch_bytes: u64,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_dflash_q8_history_panel_token8_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQ8_0F32MatvecArgsV1,
            tokens: u32,
            scratch: u64,
            scratch_bytes: u64,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_dflash_q8_history_fc_slice_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQ8_0F32MatvecArgsV1,
            tokens: u32,
            scratch: u64,
            scratch_bytes: u64,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_dflash_key_norm_rope_context_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            norm: *const AugurCudaRmsnormF32ArgsV1,
            rope: *const AugurCudaDflashRopeF32ArgsV1,
            tokens: u32,
            heads: u32,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_dflash_rope_context_rows_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaDflashRopeF32ArgsV1,
            tokens: u32,
            heads: u32,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_q8_1_activation_quantize_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQ8_1ActivationQuantizeArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_q8_1_activation_quantize_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQ8_1ActivationQuantizeArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_q8_0_q8_1_matvec_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQ8_0Q8_1MatvecArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_q8_0_q8_1_matvec_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQ8_0Q8_1MatvecArgsV1,
        ) -> AugurCudaResultV1;
        
        #[cfg(feature = "qwen35-fused-ffn-swiglu")]
        pub fn augur_cuda_q8_0_q8_1_fused_swiglu_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQ8_0Q8_1FusedSwigluArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_nvfp4_sm70_f32_matvec_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaNvfp4Sm70F32MatvecArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_nvfp4_sm70_f32_matvec_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaNvfp4Sm70F32MatvecArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_nvfp4_sm70_qpn_source_pair_submit_v1(
            ordinal: c_int, stream: AugurCudaStream, args: *const AugurCudaNvfp4Sm70F32MatvecArgsV1,
            mode: u32, splits: u32, partial: u64, partial_bytes: u64,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_nvfp4_sm70_large_split8_submit_v1(
            ordinal: c_int, stream: AugurCudaStream,
            args: *const AugurCudaNvfp4Sm70F32MatvecArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_nvfp4_sm70_small_gate_split16_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaNvfp4Sm70F32MatvecArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_nvfp4_sm70_small_gate_pair_prevalidate_v1(
            ordinal: c_int, stream: AugurCudaStream,
            a: *const AugurCudaNvfp4Sm70F32MatvecArgsV1,
            b: *const AugurCudaNvfp4Sm70F32MatvecArgsV1, schedule: u32,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_nvfp4_sm70_small_gate_pair_submit_v1(
            ordinal: c_int, stream: AugurCudaStream,
            a: *const AugurCudaNvfp4Sm70F32MatvecArgsV1,
            b: *const AugurCudaNvfp4Sm70F32MatvecArgsV1, schedule: u32,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_nvfp4_sm70_f32_batch_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaNvfp4Sm70F32MatvecArgsV1,
            batch_rows: u32,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_nvfp4_sm70_q8_1_matvec_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaNvfp4Sm70Q8_1MatvecArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_nvfp4_sm70_q8_1_matvec_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaNvfp4Sm70Q8_1MatvecArgsV1,
        ) -> AugurCudaResultV1;
        
        #[cfg(feature = "qwen35-fused-ffn-swiglu")]
        pub fn augur_cuda_nvfp4_sm70_q8_1_fused_swiglu_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaNvfp4Sm70Q8_1FusedSwigluArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_prefill_cublas_compiled_max_tokens_v1() -> u32;
        
        
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_prefill_cublas_plan_create_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35PrefillCublasPlanArgsV1,
            out_plan: *mut *mut AugurCudaQwen35PrefillCublasPlanV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_prefill_cublas_projection_preflight_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
            args: *const AugurCudaQwen35PrefillCublasProjectionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_prefill_cublas_projection_prevalidated_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
            args: *const AugurCudaQwen35PrefillCublasProjectionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_prefill_integer_mma_scratch_bytes_v1(
            format: u32,
            rows: u32,
            cols: u32,
            tokens: u32,
        ) -> u64;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_prefill_input_integer_scratch_bytes_v1(
            format: u32,
            rows: u32,
            cols: u32,
            tokens: u32,
        ) -> u64;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_prefill_input_integer_prevalidated_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
            args: *const AugurCudaQwen35PrefillCublasProjectionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_prefill_input_llama_prevalidated_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
            args: *const AugurCudaQwen35PrefillCublasProjectionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_prefill_input_original_preflight_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
            args: *const AugurCudaQwen35PrefillF32ProjectionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_prefill_input_original_prevalidated_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
            args: *const AugurCudaQwen35PrefillF32ProjectionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_dflash_llama_bf16_head_preflight_v1(
            ordinal:c_int, stream:AugurCudaStream, args:*const AugurCudaBf16F32MatvecArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_dflash_llama_bf16_head_submit_v1(
            ordinal:c_int, stream:AugurCudaStream, args:*const AugurCudaBf16F32MatvecArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_dflash_bf16_head_f32_rw4_preflight_v1(
            ordinal:c_int, stream:AugurCudaStream, args:*const AugurCudaBf16F32MatvecArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_dflash_bf16_head_f32_rw4_submit_v1(
            ordinal:c_int, stream:AugurCudaStream, args:*const AugurCudaBf16F32MatvecArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_dflash_selector_gate_f32_preflight_v1(
            ordinal:c_int, stream:AugurCudaStream, plan:*mut AugurCudaQwen35PrefillCublasPlanV1,
            args:*const AugurCudaQwen35PrefillF32ProjectionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_dflash_selector_gate_f32_submit_v1(
            ordinal:c_int, stream:AugurCudaStream, plan:*mut AugurCudaQwen35PrefillCublasPlanV1,
            args:*const AugurCudaQwen35PrefillF32ProjectionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_dflash_llama_projection_preflight_v1(
            ordinal: c_int, stream: AugurCudaStream,
            plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
            args: *const AugurCudaQwen35PrefillF32ProjectionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_dflash_llama_projection_submit_v1(
            ordinal: c_int, stream: AugurCudaStream,
            plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
            args: *const AugurCudaQwen35PrefillF32ProjectionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_dflash_q8_b8_direct_preflight_v1(
            ordinal: c_int, stream: AugurCudaStream,
            plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
            args: *const AugurCudaQwen35PrefillF32ProjectionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_dflash_q8_b8_direct_submit_v1(
            ordinal: c_int, stream: AugurCudaStream,
            plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
            args: *const AugurCudaQwen35PrefillF32ProjectionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_prefill_llama_projection_rows_preflight_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
            args: *const AugurCudaQwen35PrefillF32ProjectionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_prefill_llama_projection_rows_prevalidated_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
            args: *const AugurCudaQwen35PrefillF32ProjectionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_prefill_llama_projection_group_rows_prevalidated_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
            args: *const AugurCudaQwen35PrefillF32ProjectionArgsV1,
            count: u32,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_prefill_llama_swiglu_rows_preflight_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaSiluMulF32ArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_prefill_llama_swiglu_rows_prevalidated_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaSiluMulF32ArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_prefill_llama_projection_preflight_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
            args: *const AugurCudaQwen35PrefillF32ProjectionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_prefill_llama_projection_prevalidated_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
            args: *const AugurCudaQwen35PrefillF32ProjectionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_prefill_llama_swiglu_preflight_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaSiluMulF32ArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_prefill_llama_swiglu_prevalidated_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaSiluMulF32ArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_prefill_integer_mma_prevalidated_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
            args: *const AugurCudaQwen35PrefillCublasProjectionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_prefill_down_tensor_tile_rows_v1(
            format: u32,
            rows: u32,
            cols: u32,
            tokens: u32,
        ) -> u32;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_prefill_down_tensor_prevalidated_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
            args: *const AugurCudaQwen35PrefillCublasProjectionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_prefill_down_compensated_tile_rows_v1(
            format: u32,
            rows: u32,
            cols: u32,
            tokens: u32,
        ) -> u32;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_prefill_down_compensated_prevalidated_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
            args: *const AugurCudaQwen35PrefillCublasProjectionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_packed_small_batch_projection_preflight_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35PrefillCublasProjectionArgsV1,
            exact_tiles: u32,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_packed_small_batch_projection_prevalidated_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35PrefillCublasProjectionArgsV1,
            exact_tiles: u32,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_exact_projection_tile_width_v1(
            format: u32,
            rows: u32,
            cols: u32,
            tokens: u32,
        ) -> u32;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_planar_q8_head_preflight_v1(
            ordinal: c_int, stream: AugurCudaStream,
            args: *const AugurCudaQwen35PrefillCublasProjectionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_planar_q8_head_prevalidated_submit_v1(
            ordinal: c_int, stream: AugurCudaStream,
            args: *const AugurCudaQwen35PrefillCublasProjectionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_row_q8_input_binding_preflight_v1(
            ordinal: c_int, stream: AugurCudaStream, plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
            args: *const AugurCudaQwen35PrefillCublasProjectionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_row_q8_input_packed_preflight_v1(
            ordinal: c_int, stream: AugurCudaStream,
            args: *const AugurCudaQwen35PrefillCublasProjectionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_row_q8_input_packed_submit_v1(
            ordinal: c_int, stream: AugurCudaStream,
            args: *const AugurCudaQwen35PrefillCublasProjectionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_row_q8_input_original_preflight_v1(
            ordinal: c_int, stream: AugurCudaStream, plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
            args: *const AugurCudaQwen35PrefillF32ProjectionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_row_q8_input_original_submit_v1(
            ordinal: c_int, stream: AugurCudaStream, plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
            args: *const AugurCudaQwen35PrefillF32ProjectionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_row_q8_input_group_submit_v1(
            ordinal: c_int, stream: AugurCudaStream, plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
            args: *const AugurCudaQwen35PrefillF32ProjectionArgsV1, count: u32,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_exact_projection_tiles_prevalidated_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
            args: *const AugurCudaQwen35PrefillCublasProjectionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_projection_row_reuse_width_v1(
            format: u32,
            rows: u32,
            cols: u32,
            tokens: u32,
        ) -> u32;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_projection_row_reuse_prevalidated_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
            args: *const AugurCudaQwen35PrefillCublasProjectionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_ffn_down_row_reuse_prevalidated_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
            args: *const AugurCudaQwen35PrefillCublasProjectionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_ffn_nvfp4_pair_preflight_v1(
            ordinal: c_int, stream: AugurCudaStream,
            plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
            gate: *const AugurCudaQwen35PrefillCublasProjectionArgsV1,
            up: *const AugurCudaQwen35PrefillCublasProjectionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_ffn_nvfp4_pair_prevalidated_submit_v1(
            ordinal: c_int, stream: AugurCudaStream,
            plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
            gate: *const AugurCudaQwen35PrefillCublasProjectionArgsV1,
            up: *const AugurCudaQwen35PrefillCublasProjectionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_canonical_prefill_grid_prevalidated_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
            args: *const AugurCudaQwen35PrefillCublasProjectionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_quantized_small_batch_prevalidated_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
            args: *const AugurCudaQwen35PrefillCublasProjectionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_compare_summary_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            candidate: u64,
            reference: u64,
            input_bytes: u64,
            output: u64,
            output_bytes: u64,
            format: u32,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_prefill_cublas_projection_profile_stage_prevalidated_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
            args: *const AugurCudaQwen35PrefillCublasProjectionArgsV1,
            stage: u32,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_prefill_cublas_projection_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
            args: *const AugurCudaQwen35PrefillCublasProjectionArgsV1,
        ) -> AugurCudaResultV1;
        
        
        
        
        
        
        
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_prefill_cublas_projection_pair_prevalidated_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
            args: *const AugurCudaQwen35PrefillCublasProjectionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_prefill_cublas_projection_pair_profile_stage_prevalidated_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
            args: *const AugurCudaQwen35PrefillCublasProjectionArgsV1,
            stage: u32,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_prefill_cublas_projection_quad_prevalidated_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
            args: *const AugurCudaQwen35PrefillCublasProjectionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_prefill_cublas_projection_quad_profile_stage_prevalidated_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
            args: *const AugurCudaQwen35PrefillCublasProjectionArgsV1,
            stage: u32,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_prefill_rmsnorm_preflight_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35PrefillRmsnormArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_prefill_rmsnorm_prevalidated_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35PrefillRmsnormArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_prefill_llama_input_rmsnorm_prevalidated_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35PrefillRmsnormArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_prefill_linear_chunk_preflight_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35PrefillLinearChunkArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_prefill_linear_chunk_prevalidated_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35PrefillLinearChunkArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_prefill_linear_chunk_shared_scalars_prevalidated_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35PrefillLinearChunkArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_checkpoint_prefill_linear_chunk_prevalidated_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35PrefillLinearChunkArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(all(
            feature = "qwen35-prefill-cublas",
            feature = "qwen35-kv8-full-attention"
        ))]
        pub fn augur_cuda_qwen35_canonical_prefill_scores_preflight_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
            args: *const AugurCudaQwen35CanonicalPrefillArgsV1,
            scores: AugurCudaDevicePtr,
            score_bytes: u64,
            score_policy: u32,
        ) -> AugurCudaResultV1;
        #[cfg(all(
            feature = "qwen35-prefill-cublas",
            feature = "qwen35-kv8-full-attention"
        ))]
        pub fn augur_cuda_qwen35_canonical_prefill_scores_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
            args: *const AugurCudaQwen35CanonicalPrefillArgsV1,
            scores: AugurCudaDevicePtr,
            score_bytes: u64,
            score_policy: u32,
        ) -> AugurCudaResultV1;
        
        
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_prefill_linear_chunk_fold_prevalidated_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35PrefillLinearChunkArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_prefill_linear_chunk_fold_shared_prevalidated_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35PrefillLinearChunkArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_prefill_linear_checkpoint_fold_preflight_v1(
            ordinal: c_int, stream: AugurCudaStream,
            args: *const AugurCudaQwen35PrefillLinearChunkArgsV1,
            history: u64, history_bytes: u64, state: u64, state_bytes: u64,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_prefill_linear_checkpoint_fold_prevalidated_submit_v1(
            ordinal: c_int, stream: AugurCudaStream,
            args: *const AugurCudaQwen35PrefillLinearChunkArgsV1,
            history: u64, history_bytes: u64, state: u64, state_bytes: u64,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_prefill_linear_chunk_precise_prevalidated_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35PrefillLinearChunkArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_nerkyor_prefill_full_attention_canonical_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35PrefillFullAttentionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_prefill_full_attention_preflight_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35PrefillFullAttentionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_prefill_full_attention_prevalidated_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35PrefillFullAttentionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(all(
            feature = "qwen35-prefill-cublas",
            feature = "qwen35-kv8-full-attention"
        ))]
        pub fn augur_cuda_qwen35_prefill_kv8_chunk_attention_preflight_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35PrefillKv8ChunkAttentionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(all(
            feature = "qwen35-prefill-cublas",
            feature = "qwen35-kv8-full-attention"
        ))]
        pub fn augur_cuda_qwen35_prefill_kv8_chunk_attention_prevalidated_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35PrefillKv8ChunkAttentionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(all(
            feature = "qwen35-prefill-cublas",
            feature = "qwen35-kv8-full-attention"
        ))]
        pub fn augur_cuda_qwen35_prefill_kv8_tiled_dgemm_preflight_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
            args: *const AugurCudaQwen35PrefillKv8ChunkAttentionArgsV1,
            score_address: u64,
            score_bytes: u64,
        ) -> AugurCudaResultV1;
        #[cfg(all(
            feature = "qwen35-prefill-cublas",
            feature = "qwen35-kv8-full-attention"
        ))]
        pub fn augur_cuda_qwen35_prefill_kv8_tiled_dgemm_prevalidated_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
            args: *const AugurCudaQwen35PrefillKv8ChunkAttentionArgsV1,
            score_address: u64,
            score_bytes: u64,
        ) -> AugurCudaResultV1;
        #[cfg(all(
            feature = "qwen35-prefill-cublas",
            feature = "qwen35-kv8-full-attention"
        ))]
        pub fn augur_cuda_qwen35_prefill_kv8_tiled_scores_preflight_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35PrefillKv8ChunkAttentionArgsV1,
            score_address: u64,
            score_bytes: u64,
        ) -> AugurCudaResultV1;
        #[cfg(all(
            feature = "qwen35-prefill-cublas",
            feature = "qwen35-kv8-full-attention"
        ))]
        pub fn augur_cuda_qwen35_prefill_kv8_tiled_cached_query_prevalidated_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35PrefillKv8ChunkAttentionArgsV1,
            score_address: u64,
            score_bytes: u64,
        ) -> AugurCudaResultV1;
        #[cfg(all(
            feature = "qwen35-prefill-cublas",
            feature = "qwen35-kv8-full-attention"
        ))]
        pub fn augur_cuda_qwen35_prefill_kv8_tiled_scores_prevalidated_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35PrefillKv8ChunkAttentionArgsV1,
            score_address: u64,
            score_bytes: u64,
        ) -> AugurCudaResultV1;
        #[cfg(all(
            feature = "qwen35-prefill-cublas",
            feature = "qwen35-kv8-full-attention"
        ))]
        pub fn augur_cuda_qwen35_prefill_kv8_staged_scores_preflight_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35PrefillKv8ChunkAttentionArgsV1,
            score_address: u64,
            score_bytes: u64,
        ) -> AugurCudaResultV1;
        #[cfg(all(
            feature = "qwen35-prefill-cublas",
            feature = "qwen35-kv8-full-attention"
        ))]
        pub fn augur_cuda_qwen35_prefill_kv8_staged_scores_prevalidated_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35PrefillKv8ChunkAttentionArgsV1,
            score_address: u64,
            score_bytes: u64,
        ) -> AugurCudaResultV1;
        #[cfg(all(
            feature = "qwen35-prefill-cublas",
            feature = "qwen35-kv8-full-attention"
        ))]
        pub fn augur_cuda_qwen35_prefill_kv8_chunk_parallel_scales_prevalidated_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35PrefillKv8ChunkAttentionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(all(
            feature = "qwen35-prefill-cublas",
            feature = "qwen35-kv8-full-attention"
        ))]
        pub fn augur_cuda_qwen35_prefill_kv8_chunk_warp_shared_kv_prevalidated_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35PrefillKv8ChunkAttentionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(all(
            feature = "qwen35-prefill-cublas",
            feature = "qwen35-kv8-full-attention"
        ))]
        pub fn augur_cuda_qwen35_prefill_kv8_chunk_warp_ordered_prevalidated_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35PrefillKv8ChunkAttentionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(all(
            feature = "qwen35-prefill-cublas",
            feature = "qwen35-kv8-full-attention"
        ))]
        pub fn augur_cuda_qwen35_prefill_kv8_chunk_register_softmax_prevalidated_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35PrefillKv8ChunkAttentionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(all(
            feature = "qwen35-prefill-cublas",
            feature = "qwen35-kv8-full-attention"
        ))]
        pub fn augur_cuda_qwen35_prefill_kv8_chunk_shared_softmax_prevalidated_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35PrefillKv8ChunkAttentionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(all(
            feature = "qwen35-prefill-cublas",
            feature = "qwen35-kv8-full-attention"
        ))]
        pub fn augur_cuda_qwen35_prefill_kv8_chunk_shared_softmax_tile16_prevalidated_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35PrefillKv8ChunkAttentionArgsV1,
        ) -> AugurCudaResultV1;
        #[cfg(all(
            feature = "qwen35-prefill-cublas",
            feature = "qwen35-kv8-full-attention"
        ))]
        pub fn augur_cuda_qwen35_prefill_kv8_chunk_shared_softmax_unit_scale_prevalidated_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaQwen35PrefillKv8ChunkAttentionArgsV1,
        ) -> AugurCudaResultV1;
        
        #[cfg(feature = "qwen35-prefill-cublas")]
        pub fn augur_cuda_qwen35_prefill_cublas_plan_destroy_v1(
            plan: *mut AugurCudaQwen35PrefillCublasPlanV1,
        );
        
        pub fn augur_cuda_bf16_token_row_gather_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaBf16TokenRowGatherArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_bf16_token_row_gather_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaBf16TokenRowGatherArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_bf16_token_rows_gather_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaBf16TokenRowGatherArgsV1,
            tokens: u32,
        ) -> AugurCudaResultV1;
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        pub fn augur_cuda_weighted_accum_f32_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaWeightedAccumF32ArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_weighted_accum_f32_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaWeightedAccumF32ArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_dflash_grouped_conv_f32_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaDflashGroupedConvF32ArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_dflash_grouped_conv_f32_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaDflashGroupedConvF32ArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_dflash_grouped_conv_history_f32_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaDflashGroupedConvHistoryF32ArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_dflash_grouped_conv_history_f32_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaDflashGroupedConvHistoryF32ArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_dflash_grouped_conv_history_rows_f32_submit_v2(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaDflashGroupedConvHistoryRowsF32ArgsV2,
        ) -> AugurCudaResultV1;
        
        pub fn augur_cuda_compensated_accum_f32_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaCompensatedAccumF32ArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_compensated_accum_f32_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaCompensatedAccumF32ArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_rmsnorm_f32_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaRmsnormF32ArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_rmsnorm_f32_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaRmsnormF32ArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_rmsnorm_f32_warp_tree5120_submit_v1(
            ordinal: i32,
            stream: AugurCudaStream,
            args: *const AugurCudaRmsnormF32ArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_rmsnorm_f32_rows_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaRmsnormF32ArgsV1,
            rows: u32,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_split_rmsnorm_f32_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaSplitRmsnormF32ArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_split_rmsnorm_f32_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaSplitRmsnormF32ArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_rope_f32_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaRopeF32ArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_rope_f32_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaRopeF32ArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_dflash_rope_f32_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaDflashRopeF32ArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_dflash_rope_f32_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaDflashRopeF32ArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_dflash_cache_attention_f32_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaDflashCacheAttentionF32ArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_dflash_cache_attention_f32_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaDflashCacheAttentionF32ArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_dflash_block_attention_f32_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaDflashBlockAttentionF32ArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_f8e4m3_f32_matvec_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaF8E4M3F32MatvecArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_f8e4m3_f32_matvec_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaF8E4M3F32MatvecArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_f8e4m3_f32_matvec_bitdecode_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaF8E4M3F32MatvecArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_f8e4m3_f32_matvec_bitdecode_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaF8E4M3F32MatvecArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_f8e4m3_f32_matvec_warp_fp32_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaF8E4M3F32MatvecArgsV1,
        ) -> AugurCudaResultV1;
        
        
        
        pub fn augur_cuda_f8e4m3_f32_matvec_row32_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaF8E4M3F32MatvecArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_f8e4m3_f32_matvec_row32_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaF8E4M3F32MatvecArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_f8e4m3_f32_matvec_row_scale_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaF8E4M3F32MatvecArgsV1,
        ) -> AugurCudaResultV1;
        pub fn augur_cuda_f8e4m3_f32_matvec_row_scale_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaF8E4M3F32MatvecArgsV1,
        ) -> AugurCudaResultV1;

        pub fn augur_cuda_resident_moe_f32_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaResidentMoeF32ArgsV1,
        ) -> AugurCudaResultV1;

        pub fn augur_cuda_resident_moe_f32_submit_v1(
            ordinal: c_int,
            stream: AugurCudaStream,
            args: *const AugurCudaResidentMoeF32ArgsV1,
        ) -> AugurCudaResultV1;

        

        

        

        

        

        

        

        

        

        

        

        

        

        

        

        

        

        

        

        

        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        

        

        

        

        

        

        

        

        

        

        

        

        

        

        

        

        fn input_args_with_host_valid_ranges() -> AugurCudaM306K4R2ExpertRouteArgsV2 {
            let mut args = AugurCudaM306K4R2ExpertRouteArgsV2::new_input();
            args.slot_table = 0x1_0000;
            args.ids_u16 = 0x1_6000;
            args.router_status_i32 = 0x1_7000;
            args.input_f32 = 0x1_8000;
            args.output_f32 = 0x2_0000;
            args.step_receipt = 0x4_0000;
            args.step_generation = 1;
            args.plan_digest[0] = 1;
            args
        }

        fn r3_input_args_with_host_valid_ranges() -> AugurCudaM306K4R3ExpertRouteArgsV3 {
            let mut args = AugurCudaM306K4R3ExpertRouteArgsV3::new_input();
            args.slot_table = 0x1_0000;
            args.ids_u16 = 0x1_6000;
            args.router_status_i32 = 0x1_7000;
            args.input_f32 = 0x1_8000;
            args.output_f32 = 0x2_0000;
            args.step_receipt = 0x4_0000;
            args.validation_record = 0x4_1000;
            args.step_generation = 1;
            args.plan_digest[0] = 1;
            args
        }

        

        

        

        

        

        
    }
}

#[cfg(feature = "cuda")]
pub use ffi::*;

#[cfg(all(feature = "cuda", feature = "qwen35-prefill-cublas"))]
mod qwen35_rotated_kv;
#[cfg(all(feature = "cuda", feature = "qwen35-prefill-cublas"))]
pub use qwen35_rotated_kv::*;

#[cfg(all(feature = "cuda", feature = "qwen35-prefill-cublas"))]
mod qwen35_qpn;
#[cfg(all(feature = "cuda", feature = "qwen35-prefill-cublas"))]
pub use qwen35_qpn::*;

#[cfg(feature = "qwen35-vision-affine")]
mod qwen35_vision;
#[cfg(feature = "qwen35-vision-affine")]
pub use qwen35_vision::*;
