#![deny(unsafe_op_in_unsafe_fn)]

#[doc(hidden)]
pub const AUGUR_BACKEND_RESOLVED_FEATURES: &str = env!("AUGUR_BACKEND_RESOLVED_FEATURES");

#[cfg(feature = "cuda")]
#[doc(hidden)]
pub const AUGUR_CUDA_SYS_RESOLVED_FEATURES: &str = augur_cuda_sys::AUGUR_CUDA_SYS_RESOLVED_FEATURES;

pub mod capabilities;
pub mod cpu;
pub mod cpu_identity;
pub mod cuda;

#[cfg(feature = "cuda")]
pub mod cuda_add;
#[cfg(feature = "cuda")]
pub mod cuda_bf16;

#[cfg(feature = "cuda")]
pub mod cuda_compensated;
#[cfg(feature = "cuda")]
pub mod cuda_dflash_block_attention;
#[cfg(feature = "cuda")]
pub mod cuda_dflash_cache_attention;
#[cfg(feature = "cuda")]
pub mod cuda_dflash_context_rows;
#[cfg(feature = "cuda")]
pub mod cuda_dflash_grouped_conv;
#[cfg(feature = "cuda")]
pub mod cuda_f8;
#[cfg(feature = "cuda")]
pub mod cuda_gqa_repeat;
#[cfg(feature = "cuda")]
pub mod cuda_m120_qk;
#[cfg(feature = "cuda")]
pub mod cuda_m121_dp4a;
#[cfg(feature = "cuda")]
pub mod cuda_m142_topk;
#[cfg(feature = "cuda")]
pub mod cuda_m285_kv;
#[cfg(feature = "cuda")]
pub mod cuda_m292_hbm;

#[cfg(feature = "cuda")]
pub mod cuda_norm;

#[cfg(feature = "cuda")]
pub mod cuda_nvfp4_sm70;

#[cfg(all(feature = "cuda", feature = "qwen35-prefill-cublas"))]
pub mod cuda_qwen35_qpn;

#[cfg(feature = "cuda")]
pub mod cuda_q4k;
#[cfg(feature = "cuda")]
pub mod cuda_q8_0;
#[cfg(feature = "cuda")]
pub mod cuda_qk;
#[cfg(feature = "cuda")]
pub mod cuda_qk_expert;
#[cfg(feature = "qwen35-device-token-handoff")]
pub mod cuda_qwen35_dflash_acceptance;
#[cfg(feature = "qwen35-device-token-handoff")]
pub mod cuda_qwen35_dflash_q8_selector;
#[cfg(feature = "qwen35-device-token-handoff")]
pub mod cuda_qwen35_dflash_selector;
#[cfg(feature = "cuda")]
pub mod cuda_qwen35_full_attention;
#[cfg(feature = "qwen35-fused-ffn-swiglu")]
pub mod cuda_qwen35_fused_ffn;
#[cfg(feature = "qwen35-fused-head-norm-rope")]
pub mod cuda_qwen35_fused_head_norm_rope;
#[cfg(feature = "qwen35-kv8-full-attention")]
pub mod cuda_qwen35_kv8_segment;
#[cfg(feature = "qwen35-linear-step")]
pub mod cuda_qwen35_linear_projection;
#[cfg(feature = "qwen35-linear-step")]
pub mod cuda_qwen35_linear_step;
#[cfg(feature = "qwen35-kv8-full-attention")]
pub mod cuda_qwen35_page_digest;
#[cfg(feature = "qwen35-prefill-cublas")]
pub mod cuda_qwen35_prefill_cublas;
#[cfg(feature = "qwen35-prefill-cublas")]
pub mod cuda_qwen35_rotated_kv;
#[cfg(feature = "qwen35-device-token-handoff")]
pub mod cuda_qwen35_token_handoff;
#[cfg(feature = "qwen35-device-token-handoff")]
pub mod cuda_qwen35_vocab_topk;
#[cfg(feature = "cuda")]
pub mod cuda_resident_moe;
#[cfg(feature = "cuda")]
pub mod cuda_runtime;
#[cfg(feature = "cuda")]
pub mod cuda_silu_mul;
#[cfg(feature = "cuda")]
pub mod cuda_silu_mul_q8;
#[cfg(feature = "cuda")]
pub mod cuda_swiglu;
#[cfg(feature = "cuda")]
pub mod cuda_weighted;
pub mod r#trait;

pub use capabilities::*;
pub use cpu::*;
pub use cpu_identity::*;
pub use r#trait::*;

// M30-B2 safe CUDA ownership layer
#[cfg(feature = "cuda")]
pub use cuda_resident_moe::CudaResidentMoeF32ArgsV1;
#[cfg(feature = "cuda")]
pub use cuda_runtime::{
    cuda_device_allocation_snapshot, cuda_device_properties, cuda_device_uuid,
    cuda_host_boundary_snapshot, cuda_memory_info, CudaBatch01B4RowMainNvtxRange, CudaBufferView,
    CudaComputeCapability, CudaD2dBatch, CudaD2dOwnerHandle, CudaDeviceAllocationSnapshot,
    CudaDeviceProperties, CudaDevicePtr, CudaHostBoundarySnapshot, CudaMemoryInfo, CudaNvtxRange,
    CudaPinnedH2dDependencyReceipt, CudaProfilerCapture, CudaRuntimeError, CudaRuntimeErrorKind,
    CudaStreamCloseReceipt, CudaStreamCompletionProof, CudaStreamSubmissionReceipt,
    OwnedCudaBuffer, OwnedCudaEvent, OwnedCudaGraphExec, OwnedCudaHostBuffer,
    OwnedCudaMappedHostBuffer, OwnedCudaStream, OwnedPinnedD2hBuffer, OwnedPinnedH2dBuffer,
    OwnedPinnedH2dSegments, OwnedPinnedH2dSlot, PinnedH2dLane,
    AUGUR_BATCH01_B4_ROW_MAIN_NVTX_RANGES_V1, AUGUR_M306_STEADY_TOKEN_NVTX_RANGE_V1,
    M306_K13_STREAM_CAPABILITY_RECEIPT_ACTIVE, OWNED_PINNED_H2D_MAX_SEGMENTS,
};

#[cfg(feature = "cuda")]
pub use cuda_add::CudaAddF32ArgsV1;
#[cfg(feature = "cuda")]
pub use cuda_bf16::{CudaNerkyorBf16GatePairF32CtaArgsV1, CudaBf16F32MatvecArgsV1, CudaBf16F32MatvecRowsArgsV1, CudaBf16TokenRowGatherArgsV1, CudaBf16TokenRowsGatherArgsV1, CudaBf16WeightViewV1};
#[cfg(feature = "cuda")]
pub use cuda_compensated::*;
#[cfg(feature = "cuda")]
pub use cuda_dflash_block_attention::{
    CudaDflashBlockAttentionF32ArgsV1, CudaDflashBlockAttentionViewsV1,
};
#[cfg(feature = "cuda")]
pub use cuda_dflash_cache_attention::CudaDflashCacheAttentionF32ArgsV1;
#[cfg(feature = "cuda")]
pub use cuda_dflash_grouped_conv::{
    CudaDflashGroupedConvF32ArgsV1, CudaDflashGroupedConvHistoryF32ArgsV1,
    CudaDflashGroupedConvHistoryRowsF32ArgsV2,
};
#[cfg(feature = "cuda")]
pub use cuda_f8::{
    CudaF8E4M3F32MatvecArgsV1, CudaF8E4M3F32MatvecRow32ArgsV1, CudaF8E4M3Row32WeightViewV1,
    CudaF8E4M3WeightViewV1, CudaF8E4M3RowScaleF32MatvecArgsV1,
};
#[cfg(feature = "cuda")]
pub use cuda_gqa_repeat::CudaGqaRepeatF32ArgsV1;
#[cfg(feature = "cuda")]
pub use cuda_m120_qk::*;
#[cfg(feature = "cuda")]
pub use cuda_m121_dp4a::*;
#[cfg(feature = "cuda")]
pub use cuda_m142_topk::*;
#[cfg(feature = "cuda")]
pub use cuda_m285_kv::*;
#[cfg(feature = "cuda")]
pub use cuda_m292_hbm::*;
#[cfg(feature = "cuda")]
pub use cuda_norm::{
    CudaDflashRopeF32ArgsV1, CudaRmsnormF32ArgsV1, CudaRopeF32ArgsV1, CudaSplitRmsnormF32V1,
};
#[cfg(feature = "cuda")]
pub use cuda_nvfp4_sm70::{
    CudaNvfp4SmallGatePairSchedule, CudaNvfp4Sm70SmallGatePairArgsV1,
    CudaNvfp4Sm70F32BatchArgsV1, CudaNvfp4Sm70F32MatvecArgsV1, CudaNvfp4Sm70Q8_1MatvecArgsV1, CudaNvfp4Sm70WeightViewV1,
};

#[cfg(feature = "qwen35-device-token-handoff")]
pub use cuda_qwen35_dflash_acceptance::*;
#[cfg(feature = "qwen35-device-token-handoff")]
pub use cuda_qwen35_dflash_q8_selector::*;
#[cfg(feature = "qwen35-device-token-handoff")]
pub use cuda_qwen35_dflash_selector::*;
#[cfg(feature = "cuda")]
pub use cuda_qwen35_full_attention::{
    CudaQwen35FullAttentionArgsV1, CudaQwen35FullAttentionViewsV1, QWEN35_FULL_ATTENTION_HEAD_DIM,
    QWEN35_FULL_ATTENTION_KV_ELEMENTS, QWEN35_FULL_ATTENTION_KV_HEADS,
    QWEN35_FULL_ATTENTION_MAX_CAPACITY, QWEN35_FULL_ATTENTION_QUERY_ELEMENTS,
    QWEN35_FULL_ATTENTION_QUERY_HEADS, QWEN35_FULL_ATTENTION_ROTARY_DIM,
};
#[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
pub use cuda_qwen35_full_attention::{
    CudaQwen35FullAttentionArgsV2, CudaQwen35FullAttentionViewsV2,
    QWEN35_FULL_ATTENTION_FUSED_Q8_BYTES_V2, QWEN35_FULL_ATTENTION_MERGE_SEGMENTS_V2,
    QWEN35_FULL_ATTENTION_MERGE_THREADS_V2, QWEN35_FULL_ATTENTION_PARTIAL_ELEMENTS_V2,
    QWEN35_FULL_ATTENTION_PARTIAL_STRIDE_V2, QWEN35_FULL_ATTENTION_SPLITS_V2,
    QWEN35_FULL_ATTENTION_STAGE1_THREADS_V2, QWEN35_FULL_ATTENTION_TILE_POSITIONS_V2,
    QWEN35_FULL_ATTENTION_WORKSPACE_ELEMENTS_V2,
};
#[cfg(feature = "qwen35-kv8-batched-attention")]
pub use cuda_qwen35_full_attention::{
    CudaQwen35Kv8BatchedAttentionArgsV4, CudaQwen35Kv8BatchedAttentionViewsV4,
};
#[cfg(feature = "qwen35-kv8-full-attention")]
pub use cuda_qwen35_full_attention::{
    CudaQwen35Kv8FullAttentionArgsV3, CudaQwen35Kv8FullAttentionViewsV3,
    QWEN35_KV8_BLOCKS_PER_HEAD, QWEN35_KV8_BLOCK_BYTES, QWEN35_KV8_BLOCK_ELEMENTS,
    QWEN35_KV8_CACHE_ROW_BYTES, QWEN35_KV8_FULL_ATTENTION_MAX_CAPACITY, QWEN35_KV8_HEAD_ROW_BYTES,
};
#[cfg(feature = "qwen35-fused-head-norm-rope")]
pub use cuda_qwen35_fused_head_norm_rope::{
    CudaQwen35BatchedHeadNormRopeArgsV1, CudaQwen35BatchedHeadNormRopeViewsV1,
    CudaQwen35FusedHeadNormRopeArgsV1, CudaQwen35FusedHeadNormRopeViewsV1,
};
#[cfg(feature = "qwen35-kv8-full-attention")]
pub use cuda_qwen35_kv8_segment::{
    merge_partial_lse_v2, submit_segment_partial_v2, validate_segment_args_v2,
    AugurCudaQwen35Kv8SegmentPartialArgsV2, MergedLseV2, PartialLseViewV2, Qwen35Kv8SegmentGeometryV2,
    SegmentContractError, SegmentExtentsV2, AUGUR_CUDA_QWEN35_KV8_SEGMENT_PARTIAL_ARGS_VERSION_V2,
    QWEN35_KV8_BLOCKS_PER_HEAD_V2, QWEN35_KV8_BLOCK_BYTES_V2, QWEN35_KV8_BLOCK_ELEMENTS_V2,
    QWEN35_KV8_CACHE_ROW_BYTES_V2, QWEN35_KV8_HEAD_DIM_V2, QWEN35_KV8_HEAD_ROW_BYTES_V2,
    QWEN35_KV8_KV_HEADS_V2, QWEN35_KV8_QUERY_HEADS_V2, QWEN35_KV8_SEGMENT_PARTIAL_SCHEMA_V2,
};
#[cfg(feature = "qwen35-linear-step")]
pub use cuda_qwen35_linear_projection::CudaQwen35LinearProjectionPostprocessArgsV1;
#[cfg(feature = "qwen35-linear-step")]
pub use cuda_qwen35_linear_step::{CudaQwen35LinearStepArgsV1, CudaQwen35CheckpointLinearStepArgsV1};
#[cfg(feature = "qwen35-kv8-full-attention")]
pub use cuda_qwen35_page_digest::{
    submit_page_digest_input_view_into_view_v1, submit_page_digest_into_view_v1,
    submit_page_digest_parallel_input_view_into_view_v1, submit_page_digest_parallel_into_view_v1,
    submit_page_digest_v1, QWEN35_KV8_PAGE_DIGEST_BYTES,
};
#[cfg(feature = "qwen35-prefill-cublas")]
pub use cuda_qwen35_prefill_cublas::{
    CudaQwen35PlanarQ8HeadProjectionV1, CudaQwen35PlanarQ8HeadWeightV1,
    CudaQwen35RowQ8InputProjectionV1, CudaQwen35RowQ8InputWeightV1,
};
#[cfg(all(
    feature = "qwen35-prefill-cublas",
    feature = "qwen35-kv8-batched-attention"
))]
pub use cuda_qwen35_prefill_cublas::{
    CudaQwen35CanonicalPrefillV1, Qwen35CanonicalPrefillScorePolicy,
};
#[cfg(feature = "qwen35-device-token-handoff")]
pub use cuda_qwen35_token_handoff::*;
#[cfg(feature = "qwen35-prefill-cublas")]
pub use cuda_qwen35_prefill_cublas::CudaQwen35HeadPrepareB8V1;
#[cfg(feature = "qwen35-device-token-handoff")]
pub use cuda_qwen35_vocab_topk::*;
#[cfg(feature = "cuda")]
pub use cuda_silu_mul::CudaSiluMulF32ArgsV1;
#[cfg(feature = "cuda")]
pub use cuda_silu_mul_q8::{CudaSiluMulQ8_1ArgsV1, CudaSiluQ8PreservedB8V1};
#[cfg(feature = "cuda")]
pub use cuda_swiglu::CudaSwigluF32ArgsV1;
#[cfg(feature = "cuda")]
pub use cuda_weighted::CudaWeightedAccumF32ArgsV1;

// M30-C2 typed CUDA launch args

#[cfg(feature = "cuda")]
pub use cuda_q4k::{
    CudaF32View, CudaQ4kF32DecodeArgsV1, CudaQ4kF32PrefillArgsV1, CudaQ4kWeightViewV1,
};
#[cfg(feature = "cuda")]
pub use cuda_q8_0::{
    CudaQ8_0F32MatvecArgsV1, CudaQ8_0Q8_1MatvecArgsV1, CudaQ8_0WeightViewV1,
    CudaQ8_1ActivationQuantizeArgsV1, CudaQ8_1ActivationViewV1, Q8_1_ACTIVATION_BLOCK_BYTES,
    Q8_1_ACTIVATION_BLOCK_ELEMENTS,
};

#[cfg(feature = "cuda")]
pub use cuda_qk::{
    CudaQ5kF32DecodeArgsV1, CudaQ5kF32DecodeGrid32ArgsV2, CudaQ5kWeightViewV1,
    CudaQ6kF32DecodeArgsV1, CudaQ6kWeightViewV1,
};
#[cfg(feature = "cuda")]
pub use cuda_qk_expert::{
    CudaCompleteQkFfnF32V1, CudaQkExpertF32V1, CudaQkExpertWeightsV1, CudaQkSharedExpertF32V1,
    CudaQkSharedExpertWeightsV1, CudaRouteWeightedSixExpertF32V1, M78_SWIGLU_LIMIT,
};
#[cfg(feature = "qwen35-fused-ffn-swiglu")]
pub use cuda_qwen35_fused_ffn::{
    CudaNvfp4Sm70Q8_1FusedSwigluArgsV1, CudaQ8_0Q8_1FusedSwigluArgsV1,
};

#[cfg(feature = "qwen35-prefill-cublas")]
pub use cuda_qwen35_prefill_cublas::{
    CudaQwen35PackedSmallBatchProjectionV1, CudaQwen35PrefillCublasPlanV1,
    CudaQwen35PrefillCublasProjectionV1, CudaQwen35PrefillFullAttentionV1,
    CudaQwen35PrefillFullAttentionViewsV1, CudaQwen35PrefillLinearChunkV1,
    CudaQwen35PrefillLinearChunkViewsV1, CudaQwen35PrefillRmsnormV1, CudaQwen35PrefillWeightV1,
    QWEN35_PREFILL_FULL_HEAD_DIM, QWEN35_PREFILL_FULL_KV_ELEMENTS, QWEN35_PREFILL_FULL_KV_HEADS,
    QWEN35_PREFILL_FULL_QUERY_ELEMENTS, QWEN35_PREFILL_FULL_QUERY_HEADS,
    QWEN35_PREFILL_FULL_ROTARY_DIM, QWEN35_PREFILL_LINEAR_CONV_WEIGHT_ELEMENTS,
    QWEN35_PREFILL_LINEAR_HEAD_DIM, QWEN35_PREFILL_LINEAR_HISTORY_ELEMENTS,
    QWEN35_PREFILL_LINEAR_KEY_HEADS, QWEN35_PREFILL_LINEAR_OUTPUT_ELEMENTS,
    QWEN35_PREFILL_LINEAR_QKV_ELEMENTS, QWEN35_PREFILL_LINEAR_STATE_ELEMENTS,
    QWEN35_PREFILL_LINEAR_VALUE_HEADS,
};
#[cfg(all(
    feature = "qwen35-prefill-cublas",
    feature = "qwen35-kv8-full-attention"
))]
pub use cuda_qwen35_prefill_cublas::{
    CudaQwen35PrefillKv8CacheQuantizeV1, CudaQwen35PrefillKv8CacheQuantizeViewsV1,
};
#[cfg(all(
    feature = "qwen35-prefill-cublas",
    feature = "qwen35-kv8-full-attention"
))]
pub use cuda_qwen35_prefill_cublas::{
    CudaQwen35PrefillKv8ChunkAttentionV1, CudaQwen35PrefillKv8ChunkAttentionViewsV1,
};

// Explicit re-exports to avoid name collisions between GPU backends.
pub use cuda::{CudaBackend, CudaBuffer, CudaDevice, CudaEvent, CudaStream};

#[cfg(feature = "qwen35-vision-affine")]
pub use cuda_qwen35_prefill_cublas::{CudaQwen35VisionAffineViewsV1, CudaQwen35VisionF16AffineV1};
#[cfg(feature = "qwen35-vision-affine")]
mod cuda_qwen35_vision_control;
#[cfg(feature = "qwen35-vision-affine")]
pub use cuda_qwen35_vision_control::{CudaQwen35VisionControlOp, CudaQwen35VisionControlV1, CudaQwen35VisionControlViews};
