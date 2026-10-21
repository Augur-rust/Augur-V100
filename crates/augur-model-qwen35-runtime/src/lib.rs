#![recursion_limit = "512"]
#![deny(unsafe_op_in_unsafe_fn)]

//! Isolated Qwen3.8/27B execution groundwork.
//!
//! The crate deliberately has no dependency on `augur-infer` or the DeepSeek
//! runtime.  The CUDA feature exposes only bounded component probes until a
//! complete target-only transformer path has a numerical and user-wall
//! qualification record.

mod active_cold;
pub mod augur_sampling;
mod augur_speculative_sampling;
#[cfg(feature = "cuda")]
pub mod checkpoint_projection_cuda;
#[cfg(feature = "cuda")]
mod checkpoint_ffn_cuda;
#[cfg(feature = "qwen35-linear-step")]
pub mod checkpoint_linear_cuda;
#[cfg(all(feature = "qwen35-fused-head-norm-rope", feature = "qwen35-target-split-rope"))]
pub mod checkpoint_full_cuda;
#[cfg(all(feature = "qwen35-fused-head-norm-rope", feature = "qwen35-target-split-rope"))]
pub mod checkpoint_target_cuda;
pub mod build_identity;
#[cfg(any(
    feature = "qwen35-device-token-handoff",
    feature = "qwen35-mixed-prefix-cuda",
    test
))]
mod completion_poll;
mod dflash_artifact;
mod dflash_backbone;
mod dflash_block_attention;
mod dflash_block_conv;
mod dflash_block_producer;
mod dflash_draft_contract;
mod dflash_q8_backbone;
#[cfg(feature = "cuda")]
mod dflash_q8_backbone_cuda;
mod dflash_rope;
pub mod resident_session_profile;
pub mod augur_session;
#[cfg(feature = "qwen35-augur-session")]
pub mod augur_session_cuda;

mod dflash_hot_window;
mod dflash_lifecycle;
mod dflash_producer;
pub mod dflash_q8_features;
mod dflash_selector;
mod dflash_source_seed;
mod dflash_target_capture;
#[cfg(any(test, feature = "qwen35-draft-target-verify"))]
mod dflash_trajectory;
mod gguf_boundary;
mod gguf_prefix;
mod product;
mod resident_growing_history;
mod segmented_attention_oracle;
pub mod session_cancellation;
#[cfg(feature = "qwen35-session-checkpoint")]
pub mod session_checkpoint;
#[cfg(any(test, feature = "qwen35-session-continuation"))]
mod session_seal_hash;
mod target_only;

pub use active_cold::{
    ActiveColdKvPlan, KvPageDescriptor, KvPageLease, KvPageResidence, PageDigest, PageProofReceipt,
    PageProofRing,
};
pub use dflash_artifact::{
    inspect_dflash_artifact, DFlashArtifactError, DFlashArtifactInspection, DFlashRamResident,
};
pub use dflash_backbone::{
    DFlashBackboneError, DFlashBackboneManifest, DFlashTensorBinding,
    DFLASH_BACKBONE_MANIFEST_SCHEMA,
};
pub use dflash_block_attention::{
    apply_block_causal_gqa_attention, apply_block_noncausal_gqa_attention,
    DFlashBlockAttentionError, DFlashBlockWindow, DFLASH_BLOCK_ATTENTION_SCHEMA,
    DFLASH_BLOCK_NONCAUSAL_ATTENTION_SCHEMA,
};
pub use dflash_block_conv::{
    apply_block_local_grouped_conv, DFlashBlockConvError, DFLASH_BLOCK_CONV_SCHEMA,
};
pub use dflash_block_producer::{
    produce_dflash_block_cpu, produce_dflash_block_with_attention, DFlashBlockAttentionInput,
    DFlashBlockProducerError, DFlashBlockProducerInput, DFlashBlockProducerOutput,
    DFLASH_BLOCK_INJECTED_ATTENTION_SCHEMA, DFLASH_BLOCK_PRODUCER_SCHEMA,
};
pub use dflash_draft_contract::{
    DFlashDraftBlockInput, DFlashDraftContractError, DFlashLearnedQueryBlock,
    DFlashLearnedQueryFrame, DFLASH_DRAFT_BLOCK_SCHEMA, DFLASH_DRAFT_CONTRACT_SCHEMA,
};
pub use dflash_q8_backbone::{
    produce_dflash_q8_block_cpu, DFlashQ8Backbone, DFLASH_Q8_BLOCK_PRODUCER_SCHEMA,
};
#[cfg(feature = "qwen35-output-head-acceptance")]
pub use dflash_q8_backbone_cuda::DFlashQ8CudaCandidateDiagnostics;
#[cfg(feature = "cuda")]
pub use dflash_q8_backbone_cuda::{
    DFlashQ8CudaBlock, DFlashQ8CudaDiagnosticOutput, DFlashQ8CudaWeights,
};
pub use dflash_rope::{
    apply_dflash_neox_rope, DFLASH_ROPE_HEAD_DIM, DFLASH_ROPE_LAYOUT, DFLASH_ROPE_THETA,
};

pub use dflash_hot_window::{
    DFlashHotWindow, DFlashHotWindowConfig, DFlashHotWindowError, DFlashHotWindowLease,
    DFlashHotWindowMemoryError, DFlashHotWindowMemoryPlan, DFlashHotWindowTerminal,
    DFLASH_HOT_WINDOW_SCHEMA, QWEN35_KV8_ACTIVE_WINDOW_POSITIONS, QWEN35_KV8_CACHE_ROW_BYTES,
    QWEN35_KV8_FULL_ATTENTION_PLANES, QWEN35_KV8_PAGE_POSITIONS,
};
pub use dflash_lifecycle::{
    DFlashArtifactIdentity, DFlashCompletionProof, DFlashDType, DFlashLifecycle,
    DFlashLifecycleBudget, DFlashLifecycleError, DFlashLifecycleState, DFlashPrefillHandoffReceipt,
    DFlashPrefillReleaseObservation, DFlashRequestIdentity, DFlashSubmissionReceipt,
    DFLASH_LIFECYCLE_SCHEMA,
};
pub use dflash_producer::{
    DFlashProducerError, DFlashProducerPrefixOutput, DFlashRealProducerPrefix,
    DFLASH_HIDDEN_ELEMENTS, DFLASH_PRODUCER_PREFIX_SCHEMA, DFLASH_RMS_NORM_EPS,
    DFLASH_TARGET_FEATURE_ELEMENTS,
};
pub use dflash_selector::{
    select_dflash2_path, DFlashSelectorArtifact, DFlashSelectorBlock, DFlashSelectorError,
    DFlashSelectorPath, DFLASH_SELECTOR_SCHEMA,
};
pub use dflash_source_seed::{embed_dflash_seed, DFlashSourceSeed, DFLASH_SEED_MASK_TOKEN};
pub use dflash_target_capture::{
    DFlashTargetFeatureCaptureError, DFlashTargetFeatureCaptureReceipt,
    DFLASH_TARGET_CAPTURE_FEATURE_WIDTH, DFLASH_TARGET_CAPTURE_HIDDEN_SIZE,
    DFLASH_TARGET_CAPTURE_LAYER_IDS, DFLASH_TARGET_CAPTURE_SCHEMA,
};
pub use gguf_boundary::GgufTargetOnlyBoundary;
pub use gguf_prefix::{
    GgufFullAttentionExecutor, GgufHybridPrefixExecutor, GgufLinearLayer0Executor,
    GgufLinearPrefixExecutor,
};
pub use product::{
    admit_qwen35_product_request, plan_qwen35_32k_component_run, Qwen35Product32kComponentPlan,
    Qwen35Product32kComponentPlanError, Qwen35ProductAdmission, Qwen35ProductAdmissionError,
    Qwen35ProductKvPolicy, Qwen35ProductProfile, Qwen35ProductRequest, QWEN35_32K_CONTEXT_CAPACITY,
    QWEN35_32K_GENERATION_MAX, QWEN35_32K_PAGE_POSITIONS,
};
pub use resident_growing_history::{
    GrowingHistory, GrowingHistoryConfig, GrowingHistoryError, GrowingHistoryPlan,
    DEFAULT_GROWING_HISTORY_BLOCK, DEFAULT_GROWING_HISTORY_CAPACITY, GROWING_HISTORY_SCHEMA,
};
pub use segmented_attention_oracle::{
    compare_page_partition, merge_partials, normalize_partial, partial_for_segment,
    AttentionGeometry, AttentionPartial, SegmentedAttentionOracleError, SegmentedAttentionResult,
};
pub use target_only::{
    greedy_argmax, DecodeOutput, PrefillOutput, PrefillToken, TargetOnlyCheckpoint,
    TargetOnlyError, TargetOnlyLayerExecutor, TargetOnlyLogitHead, TargetOnlyPhase,
    TargetOnlySession,
};

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
mod dflash_real_prefix_cuda;
#[cfg(feature = "cuda")]
mod dflash_upload;
#[cfg(feature = "qwen35-mixed-prefix-cuda")]
mod mixed_prefix_cuda;
#[cfg(feature = "cuda")]
mod real_cuda;
#[cfg(feature = "qwen35-full-attention")]
mod real_full_attention_cuda;

#[cfg(feature = "qwen35-linear-attention")]
mod real_linear_attention_cuda;
#[cfg(feature = "qwen35-linear-layer")]
mod real_linear_layer_cuda;
#[cfg(feature = "qwen35-linear-step")]
mod real_linear_step_cuda;
#[cfg(feature = "cuda")]
mod real_nvfp4_cuda;
#[cfg(feature = "cuda")]
mod real_projection_cuda;
#[cfg(feature = "cuda")]
mod real_q8_cuda;
#[cfg(feature = "cuda")]
mod real_target_output_head_cuda;

#[cfg(feature = "cuda")]
pub use cuda::{
    run_qwen35_cuda_component, Qwen35CudaComponentError, Qwen35CudaComponentReport,
    Qwen35CudaProbeConfig, QWEN35_CUDA_COMPONENT_SCHEMA,
};
#[cfg(feature = "cuda")]
pub use dflash_real_prefix_cuda::{run_dflash_real_prefix_cuda_ab, DFlashRealPrefixCudaAb};
#[cfg(feature = "cuda")]
pub use dflash_upload::{
    upload_dflash_after_handoff, upload_dflash_after_handoff_from_ram, DFlashUploadConfig,
    DFlashUploadError, DFlashUploadReport, DFLASH_UPLOAD_LANES, DFLASH_UPLOAD_SCHEMA,
};
#[cfg(feature = "qwen35-draft-target-verify")]
pub use mixed_prefix_cuda::inspect_qwen35_vision_payload;
#[cfg(feature = "qwen35-draft-target-verify")]
pub use mixed_prefix_cuda::run_qwen35_q8_draft_target_verify_cuda;
#[cfg(feature = "qwen35-draft-target-verify")]
pub use mixed_prefix_cuda::{
    run_qwen35_q8_draft_target_product_rolling_cuda_with_vision,
    run_qwen35_q8_draft_target_rolling_cuda_with_options,
    run_qwen35_q8_draft_target_rolling_cuda_with_options_and_sink,
    run_qwen35_q8_draft_target_verify_rolling_cuda,
    run_qwen35_q8_draft_target_verify_rolling_cuda_with_vision, Qwen35DraftTargetRequestOptions,
    Qwen35DraftTargetRollingOptions, Qwen35ResidentDraftCuda, Qwen35VerifiedCommit,
    Qwen35VerifiedCommitSink,
};

#[cfg(feature = "qwen35-autoregressive-cuda")]
pub use mixed_prefix_cuda::{
    run_qwen35_autoregressive_cuda, Qwen35AutoregressiveCudaConfig, Qwen35AutoregressiveCudaReport,
    QWEN35_AUTOREGRESSIVE_CUDA_SCHEMA,
};
#[cfg(feature = "qwen35-batched-prefill-handoff")]
pub use mixed_prefix_cuda::{
    run_qwen35_batched_prefill_device_token_handoff_cuda,
    QWEN35_BATCHED_PREFILL_DEVICE_TOKEN_HANDOFF_AUTOREGRESSIVE_CUDA_SCHEMA,
};
#[cfg(all(
    feature = "qwen35-batched-prefill-handoff",
    feature = "qwen35-kv8-full-attention"
))]
pub use mixed_prefix_cuda::{
    run_qwen35_batched_prefill_kv8_ram_roundtrip_cuda, Kv8RamPersistenceReport,
    Kv8RamPersistenceStreamOwnerReport, LinearStatePersistenceReport,
    QWEN35_KV8_RAM_ROUNDTRIP_AUTOREGRESSIVE_CUDA_SCHEMA,
};
#[cfg(feature = "qwen35-mixed-prefix-cuda")]
pub use mixed_prefix_cuda::{
    run_qwen35_device_resident_handoff_cuda, run_qwen35_full_target_cuda,
    run_qwen35_mixed_prefix_cuda, HandoffReport, Qwen35MixedPrefixCudaConfig,
    Qwen35MixedPrefixCudaError, Qwen35MixedPrefixCudaReport,
    QWEN35_DEVICE_RESIDENT_HANDOFF_CUDA_SCHEMA, QWEN35_FULL_TARGET_CUDA_SCHEMA,
    QWEN35_MIXED_PREFIX_CUDA_SCHEMA,
};
#[cfg(feature = "qwen35-device-token-handoff")]
pub use mixed_prefix_cuda::{
    run_qwen35_device_token_handoff_cuda, DeviceTokenHandoffReceiptReport,
    DeviceTokenHandoffReport, QWEN35_DEVICE_TOKEN_HANDOFF_AUTOREGRESSIVE_CUDA_SCHEMA,
};

#[cfg(feature = "qwen35-prefill-cublas")]
pub use mixed_prefix_cuda::{
    run_qwen35_prefill_backbone_cuda, run_qwen35_prefill_chunked_cuda,
    run_qwen35_prefill_full_layer_cuda, run_qwen35_prefill_linear_layer_cuda,
    AllocationCleanupReport, AllocationIntervalReport, PrefillBackboneClaimsReport,
    PrefillBackboneNumericalReport, PrefillBackboneSemanticReport, PrefillBackboneTimingReport,
    PrefillChunkScheduleTimingReport, PrefillChunkedClaimsReport, PrefillChunkedNumericalContract,
    PrefillChunkedResourceReport, PrefillChunkedTimingReport, PrefillFullLayerClaimsReport,
    PrefillFullLayerNumericalReport, PrefillLayerResourceReport, PrefillLayerTimingReport,
    PrefillLinearLayerClaimsReport, PrefillLinearLayerNumericalReport,
    Qwen35PrefillBackboneCudaConfig, Qwen35PrefillBackboneCudaReport,
    Qwen35PrefillChunkedCudaConfig, Qwen35PrefillChunkedCudaReport,
    Qwen35PrefillFullLayerCudaConfig, Qwen35PrefillFullLayerCudaReport,
    Qwen35PrefillLinearLayerCudaConfig, Qwen35PrefillLinearLayerCudaReport,
    QWEN35_PREFILL_BACKBONE_CUDA_SCHEMA, QWEN35_PREFILL_CHUNKED_CUDA_SCHEMA,
    QWEN35_PREFILL_FULL_LAYER_CUDA_SCHEMA, QWEN35_PREFILL_LINEAR_LAYER_CUDA_SCHEMA,
};
#[cfg(feature = "qwen35-kv8-chunk-prefill")]
pub use mixed_prefix_cuda::{
    run_qwen35_prefill_dgemm_layer_probe, Qwen35PrefillDgemmLayerProbeConfig,
};

#[cfg(feature = "cuda")]
pub use real_cuda::{
    run_qwen35_real_bf16_cuda_slice, Qwen35RealCudaSliceConfig, Qwen35RealCudaSliceError,
    Qwen35RealCudaSliceReport, QWEN35_REAL_CUDA_SLICE_SCHEMA,
};
#[cfg(feature = "qwen35-full-attention")]
pub use real_full_attention_cuda::{
    run_qwen35_real_full_attention_cuda, Qwen35RealFullAttentionCudaConfig,
    Qwen35RealFullAttentionCudaError, Qwen35RealFullAttentionCudaReport,
    QWEN35_REAL_FULL_ATTENTION_CUDA_SCHEMA,
};

#[cfg(feature = "qwen35-linear-attention")]
pub use real_linear_attention_cuda::{
    run_qwen35_real_linear_attention_cuda, Qwen35RealLinearAttentionCudaConfig,
    Qwen35RealLinearAttentionCudaError, Qwen35RealLinearAttentionCudaReport,
    QWEN35_REAL_LINEAR_ATTENTION_CUDA_SCHEMA,
};
#[cfg(feature = "qwen35-linear-layer")]
pub use real_linear_layer_cuda::{
    run_qwen35_real_linear_layer_cuda, Qwen35RealLinearLayerCudaConfig,
    Qwen35RealLinearLayerCudaError, Qwen35RealLinearLayerCudaReport,
    QWEN35_REAL_LINEAR_LAYER_CUDA_SCHEMA,
};
#[cfg(feature = "qwen35-linear-step")]
pub use real_linear_step_cuda::{
    run_qwen35_real_linear_step_cuda, Qwen35RealLinearStepCudaConfig,
    Qwen35RealLinearStepCudaError, Qwen35RealLinearStepCudaReport,
    QWEN35_REAL_LINEAR_STEP_CUDA_SCHEMA,
};
#[cfg(feature = "cuda")]
pub use real_nvfp4_cuda::{
    run_qwen35_real_nvfp4_cuda, Qwen35RealNvfp4CudaConfig, Qwen35RealNvfp4CudaError,
    Qwen35RealNvfp4CudaReport, QWEN35_REAL_NVFP4_CUDA_SCHEMA,
};
#[cfg(feature = "cuda")]
pub use real_projection_cuda::{
    run_qwen35_real_layer0_projection_cuda, Qwen35RealLayer0ProjectionCudaConfig,
    Qwen35RealLayer0ProjectionCudaError, Qwen35RealLayer0ProjectionCudaReport,
    QWEN35_REAL_LAYER0_PROJECTION_CUDA_SCHEMA,
};
#[cfg(feature = "cuda")]
pub use real_q8_cuda::{
    run_qwen35_real_q8_cuda_slice, Qwen35RealQ8CudaSliceConfig, Qwen35RealQ8CudaSliceError,
    Qwen35RealQ8CudaSliceReport, QWEN35_REAL_Q8_CUDA_SLICE_SCHEMA,
};
#[cfg(feature = "cuda")]
pub use real_target_output_head_cuda::{
    run_qwen35_real_target_output_head_cuda, run_qwen35_real_target_output_head_cuda_with_hidden,
    Qwen35RealTargetOutputHeadCudaConfig, Qwen35RealTargetOutputHeadCudaError,
    Qwen35RealTargetOutputHeadCudaReport, Qwen35RealTargetOutputHeadCudaSession,
    QWEN35_REAL_TARGET_OUTPUT_HEAD_CUDA_SCHEMA, QWEN35_REAL_TARGET_OUTPUT_HEAD_HIDDEN,
    QWEN35_REAL_TARGET_OUTPUT_HEAD_VOCAB,
};

#[cfg(feature = "qwen35-augur-vision")]
mod vision_cuda;
#[cfg(feature = "qwen35-augur-vision")]
pub use vision_cuda::{Qwen35VisionCudaEncoder, Qwen35VisionCudaOutput, Qwen35VisionHostWeights};

// The new model is an explicitly bounded research adapter, never a QUASAR default.
pub mod nerkyor_session_manifest;
#[cfg(feature = "qwen35-nerkyor-session-research")]
pub mod nerkyor_session_cuda;
