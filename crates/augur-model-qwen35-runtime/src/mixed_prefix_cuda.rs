#![cfg(feature = "qwen35-mixed-prefix-cuda")]

//! Shared-stream four-layer CUDA composition for the isolated Qwen35 lane.
//!
//! This is deliberately an opt-in component qualification boundary. It
//! executes a contiguous `0..4` prefix with real GGUF weights, but it does not
//! own a tokenizer, decoder, HTTP service, or product claim. The full-target
//! diagnostic additionally runs an offline CPU-streamed output head after the
//! measured resident path so accumulated hidden drift can be evaluated at the
//! logits boundary without contaminating the component TPS clock. The
//! Linear-attention input projection and every inter-layer hidden handoff stay
//! explicit. The terminal full-attention layer additionally exercises its real
//! Q/K/V projection, per-head normalization, and RoPE on the device so the host
//! projection cost cannot be mistaken for the native attention path.

use std::sync::Arc;
use std::rc::Rc;
use std::{error::Error, fmt, hint::spin_loop, path::PathBuf, time::Instant};
mod full_rope;


#[cfg(not(feature = "qwen35-fused-head-norm-rope"))]
use augur_backend::{CudaRmsnormF32ArgsV1, CudaRopeF32ArgsV1};
use full_rope::FullRopeTables;
mod target_weights;
mod row_input;
#[cfg(feature = "qwen35-prefill-cublas")]
mod prefill_ffn;
#[cfg(feature = "qwen35-prefill-cublas")]
use prefill_ffn::{PrefillFfn, LegacyPrefillFfn};
use target_weights::{FullWeights, HeadWeights, LinearWeights, ProjectionWeights, Weight};

mod ffn_operator_audit;
use ffn_operator_audit::{FfnIntermediateReadback, FfnOperatorAuditReport};
mod attention_operator_audit;
use attention_operator_audit::{AttentionIntermediateReadback, AttentionOperatorAuditReport};
mod full_attention_operator_audit;
use full_attention_operator_audit::{
    FullAttentionIntermediateReadback, FullAttentionOperatorAuditReport,
};
#[cfg(feature = "qwen35-draft-target-verify")]
mod draft_target;
mod resident_auxiliary;
#[cfg(feature = "qwen35-draft-target-verify")]
pub use draft_target::run_qwen35_q8_draft_target_verify_cuda;
#[cfg(feature = "qwen35-draft-target-verify")]
pub use draft_target::{
    run_qwen35_q8_draft_target_product_rolling_cuda_with_vision,
    run_qwen35_q8_draft_target_rolling_cuda_with_options,
    run_qwen35_q8_draft_target_rolling_cuda_with_options_and_sink,
    run_qwen35_q8_draft_target_verify_rolling_cuda,
    run_qwen35_q8_draft_target_verify_rolling_cuda_with_vision, Qwen35DraftTargetRequestOptions,
    Qwen35DraftTargetRollingOptions, Qwen35ResidentDraftCuda, Qwen35VerifiedCommit,
    Qwen35VerifiedCommitSink,
};
#[cfg(feature = "qwen35-draft-target-verify")]
pub use resident_auxiliary::inspect_qwen35_vision_payload;
use resident_auxiliary::{ResidentAuxiliaryOwners, ResidentAuxiliaryReport};






#[cfg(feature = "qwen35-kv8-chunk-prefill")]
mod prefill_stage_profile;
#[cfg(feature = "qwen35-kv8-chunk-prefill")]
mod sparse_prefill_profile;
#[cfg(feature = "qwen35-kv8-chunk-prefill")]
mod prefill_dgemm_layer_probe;
#[cfg(feature = "qwen35-kv8-chunk-prefill")]
mod prefill_dgemm_scores;
#[cfg(feature = "qwen35-kv8-chunk-prefill")]
pub use prefill_dgemm_layer_probe::{run_qwen35_prefill_dgemm_layer_probe, Qwen35PrefillDgemmLayerProbeConfig};
#[cfg(feature = "qwen35-kv8-chunk-prefill")]
pub use prefill_stage_profile::PrefillStageProfileReport;





use crate::{
    completion_poll::{poll_submission, SubmissionPollError},
    greedy_argmax, real_full_attention_cuda as full, real_linear_layer_cuda as linear,
    GgufHybridPrefixExecutor, TargetOnlyError,
};
#[cfg(not(feature = "qwen35-grouped-gqa-online-softmax"))]
use augur_backend::CudaQwen35FullAttentionArgsV1;
#[cfg(all(
    feature = "qwen35-grouped-gqa-online-softmax",
    not(feature = "qwen35-kv8-full-attention")
))]
use augur_backend::CudaQwen35FullAttentionArgsV2;
#[cfg(not(feature = "qwen35-split-rmsnorm"))]
use augur_backend::CudaRmsnormF32ArgsV1 as CudaPlanRmsnormF32;
#[cfg(feature = "qwen35-split-rmsnorm")]
use augur_backend::CudaSplitRmsnormF32V1 as CudaPlanRmsnormF32;

use augur_backend::{
    cuda_device_allocation_snapshot, cuda_device_properties, cuda_host_boundary_snapshot,
    CudaAddF32ArgsV1, CudaBf16F32MatvecArgsV1, CudaBf16WeightViewV1, CudaBufferView, CudaF32View,
    CudaQ8_0Q8_1MatvecArgsV1, CudaQ8_0WeightViewV1, CudaQ8_1ActivationQuantizeArgsV1,
    CudaQ8_1ActivationViewV1, CudaQwen35LinearProjectionPostprocessArgsV1, CudaRuntimeError,
    CudaSiluMulF32ArgsV1, CudaSiluMulQ8_1ArgsV1, CudaStreamSubmissionReceipt, OwnedCudaBuffer,
    OwnedCudaEvent, OwnedCudaStream, Q8_1_ACTIVATION_BLOCK_BYTES, Q8_1_ACTIVATION_BLOCK_ELEMENTS,
};
#[cfg(feature = "qwen35-device-token-handoff")]
use augur_backend::{
    cuda_qwen35_finite_top2_cpu_v1, parse_cuda_qwen35_token_handoff_receipt_v1,
    CudaBf16TokenRowGatherArgsV1, CudaProfilerCapture, CudaQwen35TokenHandoffArgsV1,
    CudaQwen35TokenHandoffOutcomeV1, CudaQwen35TokenHandoffViewsV1, CudaQwen35Top2V1,
    QWEN35_TOKEN_HANDOFF_LOGITS_BYTES, QWEN35_TOKEN_HANDOFF_PARTIAL_BYTES,
    QWEN35_TOKEN_HANDOFF_RECEIPT_BYTES, QWEN35_TOKEN_HANDOFF_VOCAB,
};
#[cfg(all(
    feature = "qwen35-device-token-handoff",
    feature = "qwen35-kv8-full-attention"
))]
use augur_backend::{
    submit_page_digest_into_view_v1, submit_page_digest_v1, QWEN35_KV8_PAGE_DIGEST_BYTES,
};
#[cfg(feature = "qwen35-prefill-cublas")]
use augur_backend::{
    CudaNvfp4Sm70WeightViewV1, CudaQwen35PrefillCublasPlanV1, CudaQwen35PrefillCublasProjectionV1,
    CudaQwen35PrefillLinearChunkV1, CudaQwen35PrefillLinearChunkViewsV1,
    CudaQwen35PrefillRmsnormV1, CudaQwen35PrefillWeightV1,
};
#[cfg(feature = "qwen35-fused-head-norm-rope")]
use augur_backend::{CudaQwen35FusedHeadNormRopeArgsV1, CudaQwen35FusedHeadNormRopeViewsV1};
#[cfg(feature = "qwen35-kv8-full-attention")]
use augur_backend::{
    CudaQwen35Kv8FullAttentionArgsV3, QWEN35_KV8_CACHE_ROW_BYTES,
    QWEN35_KV8_FULL_ATTENTION_MAX_CAPACITY,
};
#[cfg(all(
    feature = "qwen35-prefill-cublas",
    not(feature = "qwen35-kv8-chunk-prefill")
))]
use augur_backend::{CudaQwen35PrefillFullAttentionV1, CudaQwen35PrefillFullAttentionViewsV1};
#[cfg(all(
    feature = "qwen35-prefill-cublas",
    feature = "qwen35-kv8-full-attention",
    not(feature = "qwen35-kv8-chunk-prefill")
))]
use augur_backend::{CudaQwen35PrefillKv8CacheQuantizeV1, CudaQwen35PrefillKv8CacheQuantizeViewsV1};
#[cfg(feature = "qwen35-kv8-chunk-prefill")]
use augur_backend::{CudaQwen35PrefillKv8ChunkAttentionV1, CudaQwen35PrefillKv8ChunkAttentionViewsV1};

#[cfg(all(
    feature = "qwen35-batched-prefill-handoff",
    feature = "qwen35-kv8-full-attention"
))]
use augur_backend::{OwnedPinnedD2hBuffer, OwnedPinnedH2dBuffer};
#[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
use augur_backend::{
    QWEN35_FULL_ATTENTION_FUSED_Q8_BYTES_V2, QWEN35_FULL_ATTENTION_PARTIAL_ELEMENTS_V2,
    QWEN35_FULL_ATTENTION_WORKSPACE_ELEMENTS_V2,
};
use augur_core::PhysicalLayoutId;
#[cfg(feature = "qwen35-kv8-full-attention")]
use augur_model_qwen35::{dequantize_q8_0_block, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMENTS};
use augur_model_qwen35::{
    dequantize_tensor, GgufAuditOptions, GgufManifest, GgufProjectionNumerics,
    GgufStreamedOutputHead, GgufTensor, GgufTensorReader, Qwen35Config, Qwen35Error,
    Qwen35LayerKind, Qwen35LayerOutput, Qwen35LayerSpec, Qwen35LayerState, Qwen35TensorSchema,
    GGML_TYPE_BF16, GGML_TYPE_F32, GGML_TYPE_NVFP4, GGML_TYPE_Q8_0,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

pub const QWEN35_MIXED_PREFIX_CUDA_SCHEMA: &str = "crystal-quant.qwen35.mixed-prefix-cuda.v1";
pub const QWEN35_DEVICE_RESIDENT_HANDOFF_CUDA_SCHEMA: &str =
    "crystal-quant.qwen35.device-resident-handoff-cuda.v1";
pub const QWEN35_FULL_TARGET_CUDA_SCHEMA: &str = "crystal-quant.qwen35.full-target-cuda.v9";
pub const QWEN35_AUTOREGRESSIVE_CUDA_SCHEMA: &str = "crystal-quant.qwen35.autoregressive-cuda.v4";
pub const QWEN35_DEVICE_TOKEN_HANDOFF_AUTOREGRESSIVE_CUDA_SCHEMA: &str =
    "crystal-quant.qwen35.autoregressive-cuda.device-token-handoff.v1";
pub const QWEN35_BATCHED_PREFILL_DEVICE_TOKEN_HANDOFF_AUTOREGRESSIVE_CUDA_SCHEMA: &str =
    "crystal-quant.qwen35.autoregressive-cuda.batched-prefill-device-token-handoff.v1";
pub const QWEN35_KV8_RAM_ROUNDTRIP_AUTOREGRESSIVE_CUDA_SCHEMA: &str =
    "crystal-quant.qwen35.autoregressive-cuda.kv8-ram-roundtrip.v1";
#[cfg(feature = "qwen35-prefill-cublas")]
pub const QWEN35_PREFILL_LINEAR_LAYER_CUDA_SCHEMA: &str =
    "crystal-quant.qwen35.prefill-linear-layer-cublas-sm70.diagnostic.v1";
#[cfg(feature = "qwen35-prefill-cublas")]
pub const QWEN35_PREFILL_FULL_LAYER_CUDA_SCHEMA: &str =
    "crystal-quant.qwen35.prefill-full-layer-cublas-sm70.diagnostic.v1";
#[cfg(feature = "qwen35-prefill-cublas")]
pub const QWEN35_PREFILL_BACKBONE_CUDA_SCHEMA: &str =
    "crystal-quant.qwen35.prefill-64-layer-backbone-cublas-sm70.diagnostic.v2";
#[cfg(feature = "qwen35-prefill-cublas")]
pub const QWEN35_PREFILL_CHUNKED_CUDA_SCHEMA: &str =
    "crystal-quant.qwen35.prefill-chunked-64-layer-cublas-sm70.diagnostic.v2";



const MIXED_PREFIX_LAYER_COUNT: usize = 4;
const FULL_TARGET_LAYER_COUNT: usize = 64;
const TOKEN_COUNT: usize = 2;
const SEMANTIC_OUTPUT_HEAD_CHUNK_ROWS: usize = 4_096;
const SEMANTIC_TOP_K: usize = 50;
// A full-capacity SM70 attention submission can legitimately outlive the
// short-prefix polling budget by orders of magnitude.  Keep the limit finite
// (fail-closed) but large enough for a 192K terminal submission; the chunked
// prefill path retains its separate 50M per-chunk bound below.
const MAX_COMPLETION_POLLS: u64 = 64_000_000;
#[cfg(feature = "qwen35-prefill-cublas")]
const MAX_CHUNKED_PREFILL_COMPLETION_POLLS: u64 = 50_000_000;

#[inline]
fn bounded_completion_poll_backoff(polls: u64) {
    const YIELD_INTERVAL: u64 = 16_384;
    if polls >= YIELD_INTERVAL && polls.is_multiple_of(YIELD_INTERVAL) {
        std::thread::yield_now();
    } else {
        spin_loop();
    }
}
#[cfg(all(
    feature = "qwen35-batched-prefill-handoff",
    feature = "qwen35-kv8-full-attention"
))]
const KV8_RAM_PERSISTENCE_LANES: usize = 4;
const KV8_RAM_PERSISTENCE_PAGE_POSITIONS: usize = 512;

#[derive(Clone, Debug)]
pub struct Qwen35MixedPrefixCudaConfig {
    pub gguf_path: PathBuf,
    pub config_path: PathBuf,
    pub token_id: usize,
    pub device_ordinal: i32,
    pub capacity: usize,
    pub warmup: usize,
    pub ffn_operator_audit: bool,
    pub attention_operator_audit: bool,
    pub source_commit: String,
}

impl Default for Qwen35MixedPrefixCudaConfig {
    fn default() -> Self {
        Self {
            gguf_path: PathBuf::new(),
            config_path: PathBuf::new(),
            token_id: 1,
            device_ordinal: 0,
            capacity: TOKEN_COUNT,
            warmup: 1,
            ffn_operator_audit: false,
            attention_operator_audit: false,
            source_commit: "UNBOUND_MIXED_PREFIX_SOURCE".to_owned(),
        }
    }
}

#[cfg(feature = "qwen35-prefill-cublas")]
#[derive(Clone, Debug)]
pub struct Qwen35PrefillLinearLayerCudaConfig {
    pub gguf_path: PathBuf,
    pub config_path: PathBuf,
    pub tokens: usize,
    pub device_ordinal: i32,
    pub warmup: usize,
    pub iterations: usize,
    pub source_commit: String,
}

#[cfg(feature = "qwen35-prefill-cublas")]
impl Default for Qwen35PrefillLinearLayerCudaConfig {
    fn default() -> Self {
        Self {
            gguf_path: PathBuf::new(),
            config_path: PathBuf::new(),
            tokens: 128,
            device_ordinal: 0,
            warmup: 1,
            iterations: 3,
            source_commit: "DIRTY_COMPONENT_WORKTREE".to_owned(),
        }
    }
}

#[cfg(feature = "qwen35-prefill-cublas")]
#[derive(Clone, Debug)]
pub struct Qwen35PrefillFullLayerCudaConfig {
    pub gguf_path: PathBuf,
    pub config_path: PathBuf,
    pub tokens: usize,
    pub device_ordinal: i32,
    pub warmup: usize,
    pub iterations: usize,
    pub source_commit: String,
}

#[cfg(feature = "qwen35-prefill-cublas")]
impl Default for Qwen35PrefillFullLayerCudaConfig {
    fn default() -> Self {
        Self {
            gguf_path: PathBuf::new(),
            config_path: PathBuf::new(),
            tokens: 128,
            device_ordinal: 0,
            warmup: 1,
            iterations: 3,
            source_commit: "DIRTY_COMPONENT_WORKTREE".to_owned(),
        }
    }
}





#[cfg(feature = "qwen35-prefill-cublas")]
#[derive(Clone, Debug)]
pub struct Qwen35PrefillBackboneCudaConfig {
    pub gguf_path: PathBuf,
    pub config_path: PathBuf,
    pub tokens: usize,
    pub device_ordinal: i32,
    pub warmup: usize,
    pub iterations: usize,
    pub source_commit: String,
}

#[cfg(feature = "qwen35-prefill-cublas")]
impl Default for Qwen35PrefillBackboneCudaConfig {
    fn default() -> Self {
        Self {
            gguf_path: PathBuf::new(),
            config_path: PathBuf::new(),
            tokens: 128,
            device_ordinal: 0,
            warmup: 1,
            iterations: 3,
            source_commit: "DIRTY_COMPONENT_WORKTREE".to_owned(),
        }
    }
}

#[cfg(feature = "qwen35-prefill-cublas")]
#[derive(Clone, Debug)]
pub struct Qwen35PrefillChunkedCudaConfig {
    pub gguf_path: PathBuf,
    pub config_path: PathBuf,
    pub total_tokens: usize,
    pub chunk_tokens: usize,
    pub reference_chunk_tokens: usize,
    pub device_ordinal: i32,
    pub warmup: usize,
    pub iterations: usize,
    pub tensor_core_prefix_attention: bool,
    pub reference_tensor_core_prefix_attention: bool,
    pub layer_major_window: bool,
    pub direct_post_attention_activation: bool,
    pub reference_direct_post_attention_activation: bool,
    /// Candidate only; requires KV8 chunk prefill and never changes the reference.
    pub simt_shared_softmax: bool,
    /// Shared-softmax tile16 candidate; otherwise the original tile8 geometry.
    pub simt_softmax_tile16: bool,
    /// Exact exp(0)=1 shortcut for finite unchanged running maxima, tile8 only.
    pub simt_unit_previous_scale: bool,
    /// Candidate-only two-token KV tile reuse; requires shared tile8/unit scale.
    pub simt_token_pair: bool,
    /// Candidate-only ordered GDN scalar reuse; the reference remains original.
    pub simt_shared_gdn_scalars: bool,
    /// Separate instrumented pass; never contributes samples to timed TPS.
    pub profile_simt_stages: bool,
    /// Additional FFN decode/GEMM events only in the independent profile replay.
    pub profile_ffn_projections: bool,
    /// Paired scalar-equivalent weight decode for candidate prefill projections.
    pub paired_weight_decode: bool,
    pub quad_weight_decode: bool,
    /// Explicit research admission, not input batch size; ordinary admission stays1024.
    pub wide_prefill_microbatch: bool,
    /// V2: wide opt-in defaults to2048;4096 requires a further explicit choice.
    pub wide_prefill_microbatch_limit: usize,
    pub source_commit: String,
}

#[cfg(feature = "qwen35-prefill-cublas")]
impl Default for Qwen35PrefillChunkedCudaConfig {
    fn default() -> Self {
        Self {
            gguf_path: PathBuf::new(),
            config_path: PathBuf::new(),
            total_tokens: 3_584,
            chunk_tokens: 512,
            reference_chunk_tokens: 256,
            device_ordinal: 0,
            warmup: 1,
            iterations: 3,
            tensor_core_prefix_attention: false,
            reference_tensor_core_prefix_attention: false,
            layer_major_window: false,
            direct_post_attention_activation: false,
            reference_direct_post_attention_activation: false,
            simt_shared_softmax: false,
            simt_softmax_tile16: false,
            simt_unit_previous_scale: false,
            simt_token_pair: false,
            simt_shared_gdn_scalars: false,
            profile_simt_stages: false,
            profile_ffn_projections: false,
            paired_weight_decode: false,
            quad_weight_decode: false,
            wide_prefill_microbatch: false,
            wide_prefill_microbatch_limit: 2048,
            source_commit: "DIRTY_COMPONENT_WORKTREE".to_owned(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Qwen35AutoregressiveCudaConfig {
    pub gguf_path: PathBuf,
    pub config_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub prompt_text: String,
    pub prompt_repeat: usize,
    pub device_ordinal: i32,
    pub capacity: usize,
    pub generation_tokens: usize,
    pub warmup: usize,
    pub profile_device_handoff_decode: bool,
    pub profile_device_handoff_layer_timing: bool,
    /// Candidate handoff prefill only. Primary/replay keep the baseline
    /// kernel, so the existing token/top-2/state exactness gate remains an A/B.
    pub prefill_shared_softmax: bool,
    /// Replace terminal full-state D2H audit copies with bounded device-side
    /// digests. This is an opt-in diagnostic transport experiment; it never
    /// enables product publication or weakens the default numerical audit.
    pub terminal_state_receipt_only: bool,
    /// Opt-in CPU readback of the terminal hidden for the resident Q8
    /// selector numerical A/B. Product transport leaves this disabled and
    /// relies on the GPU token receipt plus terminal state digest instead.
    pub validate_q8_selector_terminal: bool,
    /// Optional complete Q8 DFlash artifact kept resident in HBM for the
    /// entire autoregressive session. This currently establishes physical
    /// residency only; the draft consumer has its own numerical gate.
    pub dflash_q8_gguf_path: Option<PathBuf>,
    /// Optional safetensors file from which all `model.visual.*` payloads are
    /// extracted and kept resident in HBM for the session. Vision execution is
    /// intentionally a separate semantic gate.
    pub vision_safetensors_path: Option<PathBuf>,
    pub source_commit: String,
}

impl Default for Qwen35AutoregressiveCudaConfig {
    fn default() -> Self {
        Self {
            gguf_path: PathBuf::new(),
            config_path: PathBuf::new(),
            tokenizer_path: PathBuf::new(),
            prompt_text: String::new(),
            prompt_repeat: 0,
            device_ordinal: 0,
            capacity: 4_096,
            generation_tokens: 128,
            warmup: 1,
            profile_device_handoff_decode: false,
            profile_device_handoff_layer_timing: false,
            prefill_shared_softmax: false,
            terminal_state_receipt_only: false,
            validate_q8_selector_terminal: false,
            dflash_q8_gguf_path: None,
            vision_safetensors_path: None,
            source_commit: "UNBOUND_QWEN35_AUTOREGRESSIVE_SOURCE".to_owned(),
        }
    }
}

#[derive(Debug)]
pub enum Qwen35MixedPrefixCudaError {
    Runtime(CudaRuntimeError),
    Model(Qwen35Error),
    Lifecycle(crate::DFlashLifecycleError),
    InvalidConfiguration(String),
    TargetOnly(String),
    CompletionPollLimit,
}

impl fmt::Display for Qwen35MixedPrefixCudaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Runtime(error) => error.fmt(formatter),
            Self::Model(error) => error.fmt(formatter),
            Self::Lifecycle(error) => error.fmt(formatter),
            Self::InvalidConfiguration(field) => {
                write!(
                    formatter,
                    "invalid Qwen mixed-prefix configuration: {field}"
                )
            }
            Self::TargetOnly(detail) => write!(formatter, "CPU oracle error: {detail}"),
            Self::CompletionPollLimit => {
                formatter.write_str("CUDA completion polling exceeded its bound")
            }
        }
    }
}

impl Error for Qwen35MixedPrefixCudaError {}

impl From<CudaRuntimeError> for Qwen35MixedPrefixCudaError {
    fn from(error: CudaRuntimeError) -> Self {
        Self::Runtime(error)
    }
}

impl From<Qwen35Error> for Qwen35MixedPrefixCudaError {
    fn from(error: Qwen35Error) -> Self {
        Self::Model(error)
    }
}

impl From<crate::DFlashLifecycleError> for Qwen35MixedPrefixCudaError {
    fn from(error: crate::DFlashLifecycleError) -> Self {
        Self::Lifecycle(error)
    }
}

impl From<TargetOnlyError> for Qwen35MixedPrefixCudaError {
    fn from(error: TargetOnlyError) -> Self {
        Self::TargetOnly(error.to_string())
    }
}

impl From<linear::Qwen35RealLinearLayerCudaError> for Qwen35MixedPrefixCudaError {
    fn from(error: linear::Qwen35RealLinearLayerCudaError) -> Self {
        match error {
            linear::Qwen35RealLinearLayerCudaError::Runtime(error) => Self::Runtime(error),
            linear::Qwen35RealLinearLayerCudaError::Model(error) => Self::Model(error),
            linear::Qwen35RealLinearLayerCudaError::InvalidConfiguration(field) => {
                Self::InvalidConfiguration(field.to_owned())
            }
            linear::Qwen35RealLinearLayerCudaError::CompletionPollLimit => {
                Self::CompletionPollLimit
            }
        }
    }
}

impl From<full::Qwen35RealFullAttentionCudaError> for Qwen35MixedPrefixCudaError {
    fn from(error: full::Qwen35RealFullAttentionCudaError) -> Self {
        match error {
            full::Qwen35RealFullAttentionCudaError::Runtime(error) => Self::Runtime(error),
            full::Qwen35RealFullAttentionCudaError::Model(error) => Self::Model(error),
            full::Qwen35RealFullAttentionCudaError::InvalidConfiguration(field) => {
                Self::InvalidConfiguration(field.to_owned())
            }
            full::Qwen35RealFullAttentionCudaError::CompletionPollLimit => {
                Self::CompletionPollLimit
            }
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Qwen35MixedPrefixCudaReport {
    pub schema: String,
    pub status: String,
    pub component_only: bool,
    pub source_commit: String,
    pub binary_path: String,
    pub binary_sha256: String,
    pub device: DeviceReport,
    pub config_sha256: String,
    pub gguf_path: String,
    pub gguf_file_sha256: Option<String>,
    pub layer_indices: Vec<usize>,
    pub layer_kinds: Vec<String>,
    pub token_ids: Vec<usize>,
    pub capacity: usize,
    pub kernel_features: KernelFeatureReport,
    pub oracle: OracleReport,
    pub timing: TimingReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resident_fast_path: Option<ResidentFastPathReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trajectory: Option<TrajectoryReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic_logits: Option<SemanticLogitsReport>,
    pub completion: CompletionReport,
    pub layers: Vec<LayerReport>,
    pub resources: ResourceReport,
    pub checks: CheckReport,
    pub claims: ClaimsReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub handoff: Option<HandoffReport>,
}

#[derive(Clone, Debug, Serialize)]
pub struct HandoffReport {
    pub device_resident: bool,
    pub device_linear_projection: bool,
    pub inter_layer_d2d_copy_count: usize,
    pub inter_layer_d2d_bytes: usize,
    pub host_projection_token_count: usize,
    pub host_hidden_handoff_token_count: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct DeviceReport {
    pub ordinal: i32,
    pub name: String,
    pub compute_capability: [u16; 2],
    pub total_global_memory_bytes: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct KernelFeatureReport {
    pub all_projection_q8_1: bool,
    pub bf16_warp_row_fp64: bool,
    pub q8_packed_load: bool,
    pub nvfp4_fused_lb10: bool,
    pub gdn_warp_row: bool,
    pub fused_head_norm_rope: bool,
    pub full_attention_parallel_exp: bool,
    pub grouped_gqa_online_softmax_v2: bool,
    pub kv8_full_attention_v3: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grouped_gqa_v2: Option<full::GroupedGqaV2GeometryReport>,
    pub split_rmsnorm: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct OracleReport {
    pub cpu_hybrid_prefix: bool,
    pub projection_numerics: String,
    pub replay_exact: bool,
    pub final_hidden_sha256: Vec<String>,
    pub state_checkpoint_sha256: Vec<Vec<String>>,
}

#[derive(Clone, Debug, Serialize)]
pub struct TimingReport {
    pub warmup_sequences: usize,
    pub token_count: usize,
    pub sequence_host_wall_ms: f64,
    pub replay_host_wall_ms: f64,
    pub layer_host_wall_ms: Vec<f64>,
    pub reference_ms: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ResidentFastPathReport {
    pub scope: String,
    pub token_count: usize,
    pub layer_submission_count: usize,
    pub inter_layer_d2d_copy_count: usize,
    pub inter_layer_d2d_bytes: usize,
    pub token_host_wall_ms: Vec<f64>,
    pub token_backbone_cuda_event_ms: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_output_head_cuda_event_ms: Option<Vec<f64>>,
    pub token_cuda_event_ms: Vec<f64>,
    pub token_layer_cuda_event_ms: Vec<Vec<f64>>,
    pub aggregate_host_wall_ms: f64,
    pub aggregate_backbone_cuda_event_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aggregate_output_head_cuda_event_ms: Option<f64>,
    pub aggregate_cuda_event_ms: f64,
    pub host_wall_tokens_per_second: f64,
    pub backbone_cuda_event_tokens_per_second: f64,
    pub cuda_event_tokens_per_second: f64,
    pub terminal_poll_iterations: u64,
    pub owner_epoch_proof: bool,
    pub final_hidden_sha256: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_norm_sha256: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logits_sha256: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_weight_payload_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_weight_payload_sha256: Option<String>,
    pub matches_audited_sequence: bool,
    pub input_boundaries: BoundaryReport,
    pub hot_path_boundaries: BoundaryReport,
    pub readback_boundaries: BoundaryReport,
    pub output_head_executed: bool,
    pub decode_executed: bool,
    pub product_tps_permitted: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct CompletionReport {
    pub layer_submission_count: usize,
    pub poll_iterations: u64,
    pub nonblocking_poll_only: bool,
    pub owner_epoch_proof: bool,
    pub final_sequence_proof: bool,
    pub replay_exact: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct LayerReport {
    pub layer: usize,
    pub kind: String,
    pub native_projection_device: bool,
    pub token_count: usize,
    pub input_hidden_sha256: Vec<String>,
    pub final_hidden_sha256: Vec<String>,
    pub state_sha256: Vec<String>,
    pub state_expected_sha256: Vec<String>,
    pub input_hidden_max_abs_diff: f32,
    pub input_hidden_max_rel_diff: f32,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub ffn_operator_audit: Vec<FfnOperatorAuditReport>,
    pub attention_operator_audit: Vec<AttentionOperatorAuditReport>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub full_attention_operator_audit: Vec<FullAttentionOperatorAuditReport>,
    pub numerical: Vec<NumericalReport>,
    pub handoff_from_previous: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct NumericalReport {
    pub name: String,
    pub elements: usize,
    pub finite_count: usize,
    pub max_abs_diff: f32,
    pub max_rel_diff: f32,
    pub max_abs_index: Option<usize>,
    pub actual_at_max_abs: Option<f32>,
    pub expected_at_max_abs: Option<f32>,
    pub absolute_tolerance: f32,
    pub relative_tolerance: f32,
    pub output_sha256: String,
    pub expected_sha256: String,
    pub pass: bool,
}

#[cfg(feature = "qwen35-prefill-cublas")]
#[derive(Clone, Debug, Serialize)]
pub struct PrefillLinearLayerNumericalReport {
    pub output: TrajectorySampleReport,
    pub recurrent_state: TrajectorySampleReport,
    pub convolution_history: TrajectorySampleReport,
    pub relative_l2_max: f64,
    pub cosine_similarity_min: f64,
    pub pass: bool,
}

#[cfg(feature = "qwen35-prefill-cublas")]
#[derive(Clone, Debug, Serialize)]
pub struct PrefillLayerTimingReport {
    pub warmup: usize,
    pub iterations: usize,
    pub tokens_per_chunk: usize,
    pub cuda_event_total_ms: f64,
    pub cuda_event_mean_ms: f64,
    pub host_wall_total_ms: f64,
    pub host_wall_mean_ms: f64,
    pub layer_matrix_tokens_per_second: f64,
    pub reset_excluded_from_cuda_event: bool,
    pub reset_included_in_host_wall: bool,
    pub input_upload_excluded_from_cuda_event: bool,
}

#[cfg(feature = "qwen35-prefill-cublas")]
#[derive(Clone, Debug, Serialize)]
pub struct PrefillLayerResourceReport {
    pub setup_boundaries: BoundaryReport,
    pub sequential_oracle_boundaries: BoundaryReport,
    pub batched_numerical_boundaries: BoundaryReport,
    pub timed_boundaries: BoundaryReport,
    pub allocation_delta: AllocationReport,
    pub completion_poll_iterations: u64,
    pub owner_epoch_proof: bool,
}

#[cfg(feature = "qwen35-prefill-cublas")]
#[derive(Clone, Debug, Serialize)]
pub struct PrefillLinearLayerClaimsReport {
    pub real_model_weights_executed: bool,
    pub complete_linear_layer_executed: bool,
    pub sequential_gpu_oracle_executed: bool,
    pub complete_transformer_executed: bool,
    pub model_prefill_tps_permitted: bool,
    pub semantic_parity_established: bool,
    pub product_tps_permitted: bool,
    pub default_cli_enabled: bool,
}

#[cfg(feature = "qwen35-prefill-cublas")]
#[derive(Clone, Debug, Serialize)]
pub struct Qwen35PrefillLinearLayerCudaReport {
    pub schema: String,
    pub status: String,
    pub component_only: bool,
    pub source_commit: String,
    pub binary_path: String,
    pub binary_sha256: String,
    pub device: DeviceReport,
    pub config_sha256: String,
    pub gguf_path: String,
    pub gguf_file_sha256: Option<String>,
    pub layer: usize,
    pub token_ids: Vec<usize>,
    pub input_hidden_sha256: String,
    pub numerical: PrefillLinearLayerNumericalReport,
    pub timing: PrefillLayerTimingReport,
    pub resources: PrefillLayerResourceReport,
    pub claims: PrefillLinearLayerClaimsReport,
}

#[cfg(feature = "qwen35-prefill-cublas")]
#[derive(Clone, Debug, Serialize)]
pub struct PrefillFullLayerNumericalReport {
    pub output: TrajectorySampleReport,
    pub key_cache: TrajectorySampleReport,
    pub value_cache: TrajectorySampleReport,
    pub relative_l2_max: f64,
    pub cosine_similarity_min: f64,
    pub pass: bool,
}

#[cfg(feature = "qwen35-prefill-cublas")]
#[derive(Clone, Debug, Serialize)]
pub struct PrefillFullLayerClaimsReport {
    pub real_model_weights_executed: bool,
    pub complete_full_attention_layer_executed: bool,
    pub batched_causal_gqa_executed: bool,
    pub sequential_gpu_oracle_executed: bool,
    pub complete_transformer_executed: bool,
    pub model_prefill_tps_permitted: bool,
    pub semantic_parity_established: bool,
    pub product_tps_permitted: bool,
    pub default_cli_enabled: bool,
}

#[cfg(feature = "qwen35-prefill-cublas")]
#[derive(Clone, Debug, Serialize)]
pub struct Qwen35PrefillFullLayerCudaReport {
    pub schema: String,
    pub status: String,
    pub component_only: bool,
    pub source_commit: String,
    pub binary_path: String,
    pub binary_sha256: String,
    pub device: DeviceReport,
    pub config_sha256: String,
    pub gguf_path: String,
    pub gguf_file_sha256: Option<String>,
    pub layer: usize,
    pub token_ids: Vec<usize>,
    pub input_hidden_sha256: String,
    pub numerical: PrefillFullLayerNumericalReport,
    pub timing: PrefillLayerTimingReport,
    pub resources: PrefillLayerResourceReport,
    pub claims: PrefillFullLayerClaimsReport,
}







#[cfg(feature = "qwen35-prefill-cublas")]
#[derive(Clone, Debug, Serialize)]
pub struct AllocationIntervalReport {
    pub before: AllocationReport,
    pub after: AllocationReport,
    pub successful_allocations_delta: u64,
    pub successful_frees_delta: u64,
    pub live_allocations_unchanged: bool,
    pub live_bytes_unchanged: bool,
    /// True when the interval intentionally releases prefill-only owners.
    /// A lifecycle interval may therefore reduce live bytes, but it must not
    /// allocate new device storage and must never increase the live set.
    pub intentional_lifecycle_release: bool,
    pub pass: bool,
}

#[cfg(feature = "qwen35-prefill-cublas")]
#[derive(Clone, Debug, Serialize)]
pub struct AllocationCleanupReport {
    pub before: AllocationReport,
    pub after: AllocationReport,
    pub successful_allocations_delta: u64,
    pub successful_frees_delta: u64,
    pub pass: bool,
}







#[cfg(feature = "qwen35-prefill-cublas")]
#[derive(Clone, Debug, Serialize)]
pub struct PrefillBackboneNumericalReport {
    pub final_hidden: TrajectorySampleReport,
    pub state_aggregate: TrajectoryAggregateReport,
    pub states: Vec<TrajectorySampleReport>,
    pub final_hidden_relative_l2_max: f64,
    pub state_relative_l2_max: f64,
    pub cosine_similarity_min: f64,
    pub pass: bool,
}

#[cfg(feature = "qwen35-prefill-cublas")]
#[derive(Clone, Debug, Serialize)]
pub struct PrefillBackboneTimingReport {
    pub warmup: usize,
    pub iterations: usize,
    pub tokens_per_chunk: usize,
    pub cuda_event_total_ms: f64,
    pub cuda_event_mean_ms: f64,
    pub host_wall_total_ms: f64,
    pub host_wall_mean_ms: f64,
    pub diagnostic_backbone_tokens_per_second: f64,
    pub host_wall_backbone_tokens_per_second: f64,
    pub reset_excluded_from_cuda_event: bool,
    pub reset_included_in_host_wall: bool,
    pub input_upload_excluded_from_cuda_event: bool,
    pub terminal_output_head_included_in_cuda_event: bool,
}

#[cfg(feature = "qwen35-prefill-cublas")]
#[derive(Clone, Debug, Serialize)]
pub struct PrefillBackboneSemanticReport {
    pub terminal_position: usize,
    pub terminal_input_token_id: usize,
    pub vocabulary_size: usize,
    pub output_head_numerics: String,
    pub normalized_hidden_operator: TrajectorySampleReport,
    pub logits_operator: TrajectorySampleReport,
    pub logits_backbone: TrajectorySampleReport,
    pub logits_end_to_end: TrajectorySampleReport,
    pub batched_argmax: usize,
    pub sequential_argmax: usize,
    pub device_argmax: usize,
    pub all_argmax_equal: bool,
    pub sequential_top1_margin: f64,
    pub device_end_to_end_max_abs_error: f64,
    pub sequential_argmax_stability_certified: bool,
    pub pass: bool,
}

#[cfg(feature = "qwen35-prefill-cublas")]
#[derive(Clone, Debug, Serialize)]
pub struct PrefillBackboneClaimsReport {
    pub real_model_weights_executed: bool,
    pub complete_64_layer_backbone_executed: bool,
    pub linear_layer_count: usize,
    pub full_attention_layer_count: usize,
    pub sequential_gpu_oracle_executed: bool,
    pub output_head_executed: bool,
    pub complete_transformer_executed: bool,
    pub model_prefill_tps_permitted: bool,
    pub semantic_parity_established: bool,
    pub product_tps_permitted: bool,
    pub default_cli_enabled: bool,
}

#[cfg(feature = "qwen35-prefill-cublas")]
#[derive(Clone, Debug, Serialize)]
pub struct Qwen35PrefillBackboneCudaReport {
    pub schema: String,
    pub status: String,
    pub component_only: bool,
    pub source_commit: String,
    pub binary_path: String,
    pub binary_sha256: String,
    pub device: DeviceReport,
    pub config_sha256: String,
    pub gguf_path: String,
    pub gguf_file_sha256: Option<String>,
    pub token_ids: Vec<usize>,
    pub input_hidden_sha256: String,
    pub numerical: PrefillBackboneNumericalReport,
    pub semantic: PrefillBackboneSemanticReport,
    pub timing: PrefillBackboneTimingReport,
    pub resources: PrefillLayerResourceReport,
    pub claims: PrefillBackboneClaimsReport,
}

#[cfg(feature = "qwen35-prefill-cublas")]
#[derive(Clone, Debug, Serialize)]
pub struct PrefillChunkScheduleTimingReport {
    pub chunk_tokens: usize,
    pub chunk_count: usize,
    pub iterations: usize,
    pub cuda_event_total_ms: f64,
    pub cuda_event_mean_sequence_ms: f64,
    pub host_wall_total_ms: f64,
    pub host_wall_mean_sequence_ms: f64,
    pub cuda_event_tokens_per_second: f64,
    pub host_wall_tokens_per_second: f64,
}

#[cfg(feature = "qwen35-prefill-cublas")]
#[derive(Clone, Debug, Serialize)]
pub struct PrefillChunkedTimingReport {
    pub warmup: usize,
    pub primary_numerical: PrefillChunkScheduleTimingReport,
    pub reference_numerical: PrefillChunkScheduleTimingReport,
    pub primary_timed: PrefillChunkScheduleTimingReport,
    pub primary_timed_host_samples_ms: Vec<f64>,
    pub primary_timed_cuda_samples_ms: Vec<f64>,
    pub primary_full_attention_path: String,
    pub reference_full_attention_path: String,
    pub primary_linear_recurrence_path: String,
    pub reference_linear_recurrence_path: String,
    pub primary_simt_attention_tile_positions: Option<usize>,
    pub reference_simt_attention_tile_positions: Option<usize>,
    pub primary_projection_weight_decode: String,
    pub reference_projection_weight_decode: String,
    pub primary_softmax_unit_previous_scale: bool,
    pub reset_excluded_from_cuda_event: bool,
    pub reset_included_in_host_wall: bool,
    pub input_upload_excluded_from_cuda_event: bool,
    pub input_upload_included_in_host_wall: bool,
    pub terminal_output_head_included_in_cuda_event: bool,
}

#[cfg(feature = "qwen35-prefill-cublas")]
#[derive(Clone, Debug, Serialize)]
pub struct PrefillChunkedResourceReport {
    pub setup_boundaries: BoundaryReport,
    pub primary_numerical_boundaries: BoundaryReport,
    pub reference_numerical_boundaries: BoundaryReport,
    pub timed_boundaries: BoundaryReport,
    pub timed_submission_boundaries: BoundaryReport,
    pub timed_allocation_interval: AllocationIntervalReport,
    pub cleanup_allocation_lifecycle: AllocationCleanupReport,
    pub allocation_delta: AllocationReport,
    pub completion_poll_iterations: u64,
    pub owner_epoch_proof: bool,
    pub primary_progress_receipt_sequence_valid: bool,
    pub reference_progress_receipt_sequence_valid: bool,
}

#[cfg(feature = "qwen35-prefill-cublas")]
#[derive(Clone, Debug, Serialize)]
pub struct PrefillChunkedClaimsReport {
    pub real_model_weights_executed: bool,
    pub complete_64_layer_transformer_executed: bool,
    pub linear_layer_count: usize,
    pub full_attention_layer_count: usize,
    pub primary_chunk_schedule_executed: bool,
    pub reference_chunk_schedule_executed: bool,
    pub tensor_core_prefix_attention_executed: bool,
    pub matched_simt_reference_executed: bool,
    pub stable_lse_merge_executed: bool,
    pub terminal_output_head_executed: bool,
    pub cross_chunk_numerical_gate_passed: bool,
    pub same_chunk_numerical_gate_passed: bool,
    pub diagnostic_model_prefill_tps_permitted: bool,
    pub performance_gate_300_tps_passed: bool,
    pub llama_355_tps_gate_passed: bool,
    pub stretch_500_tps_gate_passed: bool,
    pub semantic_parity_established: bool,
    pub product_tps_permitted: bool,
    pub default_cli_enabled: bool,
}

#[cfg(feature = "qwen35-prefill-cublas")]
#[derive(Clone, Debug, Serialize)]
pub struct PrefillChunkedNumericalContract {
    pub final_hidden_relative_l2_max: f64,
    pub final_hidden_cosine_min: f64,
    pub terminal_logits_relative_l2_max: f64,
    pub state_relative_l2_max: f64,
    pub state_cosine_min: f64,
}

#[cfg(feature = "qwen35-prefill-cublas")]
#[derive(Clone, Debug, Serialize)]
pub struct PrefillActivationPolicyComparisonReport {
    pub schema: String,
    pub primary_policy: String,
    pub reference_policy: String,
    pub comparison_kind: String,
    pub post_attention_direct_projection_count_per_layer: usize,
    pub primary_linear_input_direct_projection_count: usize,
    pub reference_linear_input_direct_projection_count: usize,
    pub full_attention_input_projection_policy_unchanged: bool,
    pub upstream_attention_projection_policy_unchanged: bool,
    pub reference_is_retained_q8: bool,
    pub comparison_passed: bool,
    pub measured_retained_reference_gate_passed: bool,
    pub external_semantic_parity_established: bool,
    pub same_policy_pass_does_not_establish_retained_compatibility: bool,
}

#[cfg(feature = "qwen35-prefill-cublas")]
fn prefill_activation_policy_report(
    config: &Qwen35PrefillChunkedCudaConfig,
    numerical_pass: bool,
    semantic_pass: bool,
) -> Option<PrefillActivationPolicyComparisonReport> {
    let primary = config.direct_post_attention_activation;
    let reference = config.reference_direct_post_attention_activation;
    if !primary && !reference {
        return None;
    }
    let policy = |direct| {
        if direct {
            "direct_f32_linear_input_and_post_attention_full_input_retained_q8"
        } else {
            "retained_q8_input_and_post_attention"
        }
        .to_owned()
    };
    Some(PrefillActivationPolicyComparisonReport {
        schema: "crystal-quant.qwen35.prefill-activation-policy.component.v2".to_owned(),
        primary_policy: policy(primary),
        reference_policy: policy(reference),
        comparison_kind: if primary == reference {
            "same_policy_or_control"
        } else {
            "cross_policy"
        }
        .to_owned(),
        post_attention_direct_projection_count_per_layer: if primary { 4 } else { 0 },
        primary_linear_input_direct_projection_count: if primary { 4 } else { 0 },
        reference_linear_input_direct_projection_count: if reference { 4 } else { 0 },
        full_attention_input_projection_policy_unchanged: true,
        upstream_attention_projection_policy_unchanged: false,
        reference_is_retained_q8: !reference,
        comparison_passed: numerical_pass && semantic_pass,
        measured_retained_reference_gate_passed: !reference && numerical_pass && semantic_pass,
        external_semantic_parity_established: false,
        same_policy_pass_does_not_establish_retained_compatibility: true,
    })
}

#[cfg(feature = "qwen35-prefill-cublas")]
fn prefill_chunk_comparison_gates(
    config: &Qwen35PrefillChunkedCudaConfig,
    numerical_pass: bool,
) -> (bool, bool) {
    // Two different requested limits can execute the identical single partial
    // chunk. Such a run proves a same-width comparison, not cross-width carry.
    let same = config.chunk_tokens.min(config.total_tokens)
        == config.reference_chunk_tokens.min(config.total_tokens);
    (numerical_pass && !same, numerical_pass && same)
}

#[cfg(feature = "qwen35-prefill-cublas")]
impl PrefillChunkedNumericalContract {
    fn for_attention_path(tensor_core: bool) -> Self {
        Self {
            final_hidden_relative_l2_max: if tensor_core { 0.015 } else { 0.02 },
            final_hidden_cosine_min: if tensor_core { 0.9999 } else { 0.999 },
            terminal_logits_relative_l2_max: if tensor_core { 0.015 } else { 0.02 },
            state_relative_l2_max: 0.05,
            state_cosine_min: 0.999,
        }
    }

    fn logits_pass(&self, logits: &TrajectorySampleReport) -> bool {
        prefill_trajectory_pass(logits, self.terminal_logits_relative_l2_max, 0.999)
    }
}

#[cfg(feature = "qwen35-prefill-cublas")]
#[derive(Clone, Debug, Serialize)]
pub struct Qwen35PrefillChunkedCudaReport {
    pub schema: String,
    pub status: String,
    pub component_only: bool,
    pub source_commit: String,
    pub binary_path: String,
    pub binary_sha256: String,
    pub device: DeviceReport,
    pub config_sha256: String,
    pub gguf_path: String,
    pub gguf_file_sha256: Option<String>,
    pub total_tokens: usize,
    pub token_ids: Vec<usize>,
    pub input_hidden_sha256: String,
    pub fused_silu_mul_q8: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub microbatch_admission: Option<PrefillMicrobatchAdmissionReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub same_width_attention_replay_bit_exact: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub same_width_linear_replay_bit_exact: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub same_width_weight_decode_replay_bit_exact: Option<bool>,
    #[cfg(feature = "qwen35-kv8-chunk-prefill")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage_profile: Option<PrefillStageProfileReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schedule_comparison: Option<PrefillScheduleComparisonReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub activation_policy_comparison: Option<PrefillActivationPolicyComparisonReport>,
    /// Per-chunk execution receipts.  These are evidence of ordered progress
    /// and ownership only; they do not by themselves establish semantic parity.
    pub primary_progress_receipts: Vec<PrefillChunkProgressReceipt>,
    pub reference_progress_receipts: Vec<PrefillChunkProgressReceipt>,
    pub common_boundary_comparisons: Vec<PrefillChunkBoundaryComparison>,
    pub numerical_contract: PrefillChunkedNumericalContract,
    pub numerical: PrefillBackboneNumericalReport,
    pub semantic: PrefillBackboneSemanticReport,
    pub timing: PrefillChunkedTimingReport,
    pub resources: PrefillChunkedResourceReport,
    pub claims: PrefillChunkedClaimsReport,
}

#[cfg(feature = "qwen35-prefill-cublas")]
#[derive(Clone, Debug, Serialize)]
pub struct PrefillChunkProgressReceipt {
    pub chunk_index: usize,
    pub start_position: usize,
    pub end_position_exclusive: usize,
    pub token_count: usize,
    pub submission_epoch: u64,
    pub owner_epoch_proof: bool,
    pub terminal: bool,
    pub input_hidden_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_hidden_sha256: Option<String>,
    /// Digest of the persistent recurrent state or full-attention KV buffers
    /// after this chunk.  Present only on evidence replays because obtaining
    /// it entails a bounded host readback of the audited device state.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state_cache_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state_cache_layer_sha256: Option<Vec<String>>,
}

#[cfg(feature = "qwen35-prefill-cublas")]
#[derive(Clone, Debug, Serialize)]
pub struct PrefillChunkBoundaryComparison {
    pub end_position_exclusive: usize,
    pub primary_chunk_index: usize,
    pub reference_chunk_index: usize,
    pub output_hidden_exact: Option<bool>,
    pub state_cache_exact: Option<bool>,
    pub first_different_layer: Option<usize>,
    pub different_layer_count: usize,
}

#[cfg(feature = "qwen35-prefill-cublas")]
#[derive(Clone, Debug, Serialize)]
pub struct PrefillMicrobatchAdmissionReport {
    pub explicit_wide_opt_in: bool,
    pub ordinary_runtime_limit: usize,
    pub compiled_native_limit: usize,
    pub admitted_runtime_limit: usize,
    pub primary_computation_rows: usize,
    pub reference_computation_rows: usize,
    pub independent_input_batch_ubatch_api: bool,
}

#[cfg(feature = "qwen35-prefill-cublas")]
#[derive(Clone, Debug, Serialize)]
pub struct PrefillScheduleComparisonReport {
    pub primary_schedule: String,
    pub window_capacity_tokens: usize,
    pub primary_linear_microbatch_limit: usize,
    pub primary_full_attention_batch_limit: usize,
    pub reference_chunk_tokens: usize,
    pub reference_tensor_core_attention: bool,
    pub same_attention_algorithm: bool,
    pub same_activation_policy: bool,
    pub recurrent_state_order: String,
    pub scratch_reuse_order: String,
    pub external_semantic_parity_established: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct TrajectoryReport {
    pub reference: String,
    pub sample_count: usize,
    pub all: TrajectoryAggregateReport,
    pub final_hidden: TrajectoryAggregateReport,
    pub state: TrajectoryAggregateReport,
    pub llama_cuda_reference_bound: bool,
    pub thresholds_calibrated: bool,
    pub qualification_permitted: bool,
    pub samples: Vec<TrajectorySampleReport>,
}

#[derive(Clone, Debug, Serialize)]
pub struct TrajectoryAggregateReport {
    pub sample_count: usize,
    pub finite_metric_count: usize,
    pub max_relative_l2_error: Option<f64>,
    pub min_cosine_similarity: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct TrajectorySampleReport {
    pub name: String,
    pub elements: usize,
    pub shape_match: bool,
    pub finite_pair_count: usize,
    pub actual_sum: Option<f64>,
    pub expected_sum: Option<f64>,
    pub actual_l2_norm: Option<f64>,
    pub expected_l2_norm: Option<f64>,
    pub error_l2_norm: Option<f64>,
    pub rmse: Option<f64>,
    pub relative_l2_error: Option<f64>,
    pub cosine_similarity: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SemanticLogitsReport {
    pub scope: String,
    pub reference: String,
    pub head_numerics: String,
    pub chunk_rows: usize,
    pub chunk_count: usize,
    pub vocabulary_size: usize,
    pub setup_wall_ms: f64,
    pub total_wall_ms: f64,
    pub output_head_executed: bool,
    pub excluded_from_resident_fast_path_timing: bool,
    pub all_logits_finite: bool,
    pub all_argmax_equal: bool,
    pub all_argmax_stability_certified: bool,
    pub logits: TrajectoryAggregateReport,
    pub device_operator_logits: TrajectoryAggregateReport,
    pub device_end_to_end_logits: TrajectoryAggregateReport,
    pub all_device_argmax_equal: bool,
    pub all_device_argmax_stability_certified: bool,
    pub llama_cuda_reference_bound: bool,
    pub thresholds_calibrated: bool,
    pub qualification_permitted: bool,
    pub tokens: Vec<SemanticTokenReport>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SemanticTokenReport {
    pub position: usize,
    pub input_token_id: usize,
    pub actual_final_hidden_sha256: String,
    pub oracle_final_hidden_sha256: String,
    pub actual_normalized_hidden_sha256: String,
    pub oracle_normalized_hidden_sha256: String,
    pub actual_logits_sha256: String,
    pub oracle_logits_sha256: String,
    pub device_normalized_hidden_sha256: String,
    pub device_logits_sha256: String,
    pub actual_wall_ms: f64,
    pub oracle_wall_ms: f64,
    pub normalized_hidden: TrajectorySampleReport,
    pub logits: TrajectorySampleReport,
    pub device_normalized_hidden_operator: TrajectorySampleReport,
    pub device_logits_operator: TrajectorySampleReport,
    pub device_logits_end_to_end: TrajectorySampleReport,
    pub max_abs_logits_error: f64,
    pub device_operator_max_abs_logits_error: f64,
    pub device_end_to_end_max_abs_logits_error: f64,
    pub actual_argmax: usize,
    pub oracle_argmax: usize,
    pub device_argmax: usize,
    pub argmax_equal: bool,
    pub device_argmax_equals_actual: bool,
    pub device_argmax_equals_oracle: bool,
    pub actual_top1_margin: f64,
    pub oracle_top1_margin: f64,
    pub oracle_argmax_stability_certified: bool,
    pub device_oracle_argmax_stability_certified: bool,
    pub top10_overlap_count: usize,
    pub top10_overlap_ratio: f64,
    pub top50_overlap_count: usize,
    pub top50_overlap_ratio: f64,
    pub device_oracle_top10_overlap_count: usize,
    pub device_oracle_top10_overlap_ratio: f64,
    pub device_oracle_top50_overlap_count: usize,
    pub device_oracle_top50_overlap_ratio: f64,
    pub actual_top50: Vec<RankedLogitReport>,
    pub oracle_top50: Vec<RankedLogitReport>,
    pub device_top50: Vec<RankedLogitReport>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RankedLogitReport {
    pub token_id: usize,
    pub logit: f32,
}

#[derive(Clone, Debug, Default, Serialize, PartialEq, Eq)]
pub struct BoundaryReport {
    pub host_waits: u64,
    pub stream_synchronizations: u64,
    pub event_synchronizations: u64,
    pub synchronous_h2d_copies: u64,
    pub synchronous_h2d_bytes: u64,
    pub synchronous_d2h_copies: u64,
    pub synchronous_d2h_bytes: u64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct AllocationReport {
    pub successful_allocations: u64,
    pub successful_frees: u64,
    pub live_allocations: u64,
    pub live_bytes: u64,
    pub peak_live_allocations: u64,
    pub peak_live_bytes: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ResourceReport {
    pub setup_boundaries: BoundaryReport,
    pub sequence_input_boundaries: BoundaryReport,
    pub completion_boundaries: BoundaryReport,
    pub readback_boundaries: BoundaryReport,
    pub cleanup_boundaries: BoundaryReport,
    pub total_boundaries: BoundaryReport,
    pub boundary_ledger_closed: bool,
    pub setup_allocations: AllocationReport,
    pub final_allocation_delta: AllocationReport,
}

#[derive(Clone, Debug, Serialize)]
pub struct CheckReport {
    pub profile_valid: bool,
    pub layer_partition_valid: bool,
    pub oracle_replay_exact: bool,
    pub all_layer_numerical_pass: bool,
    pub final_hidden_numerical_pass: bool,
    pub state_numerical_pass: bool,
    pub completion_proof_valid: bool,
    pub boundary_ledger_closed: bool,
    pub cleanup_zero: bool,
    pub same_stream_handoff: bool,
    pub native_full_attention_projection_device: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resident_fast_path_matches_audited_sequence: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic_logits_finite: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic_argmax_equal: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic_argmax_stability_certified: Option<bool>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ClaimsReport {
    pub real_weight_four_layer_mixed_prefix_cuda: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub real_weight_full_64_layer_cuda: Option<bool>,
    pub cpu_projection_handoff: bool,
    pub native_full_attention_projection_device: bool,
    pub complete_transformer_executed: bool,
    pub tokenizer_executed: bool,
    pub decode_executed: bool,
    pub semantic_parity_established: bool,
    pub product_tps_permitted: bool,
    pub default_cli_enabled: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct Qwen35AutoregressiveCudaReport {
    pub schema: String,
    pub status: String,
    pub diagnostic_only: bool,
    pub source_commit: String,
    pub binary_path: String,
    pub binary_sha256: String,
    pub device: DeviceReport,
    pub config_sha256: String,
    pub gguf_path: String,
    pub gguf_file_sha256: Option<String>,
    pub tokenizer_path: String,
    pub tokenizer_sha256: String,
    pub tokenizer_vocab_size: usize,
    pub prompt_text_sha256: String,
    pub prompt_token_ids: Vec<usize>,
    pub generated_token_ids: Vec<usize>,
    pub generated_text: String,
    pub generated_text_sha256: String,
    pub capacity: usize,
    pub bound_position_count: usize,
    pub generation_tokens: usize,
    pub kernel_features: AutoregressiveKernelFeatureReport,
    pub timing: AutoregressiveTimingReport,
    pub primary: AutoregressiveSequenceReport,
    pub replay: AutoregressiveSequenceReport,
    pub determinism: AutoregressiveDeterminismReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_token_handoff: Option<DeviceTokenHandoffReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resident_auxiliary: Option<ResidentAuxiliaryReport>,
    pub resources: AutoregressiveResourceReport,
    pub checks: AutoregressiveCheckReport,
    pub claims: AutoregressiveClaimsReport,
}

#[derive(Clone, Debug, Serialize)]
pub struct AutoregressiveKernelFeatureReport {
    pub all_projection_q8_1: bool,
    pub bf16_warp_row_fp64: bool,
    pub q8_packed_load: bool,
    pub nvfp4_fused_lb10: bool,
    pub gdn_warp_row: bool,
    pub fused_head_norm_rope: bool,
    pub warp_full_attention_fp64: bool,
    pub full_attention_parallel_exp: bool,
    pub grouped_gqa_online_softmax_v2: bool,
    pub kv8_full_attention_v3: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grouped_gqa_v2: Option<full::GroupedGqaV2GeometryReport>,
    pub split_rmsnorm: bool,
    pub fused_ffn_swiglu: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_token_handoff: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub zero_copy_layer_handoff: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batched_prefill_cublas: Option<bool>,
}

#[derive(Clone, Debug, Serialize)]
pub struct DeviceTokenHandoffReport {
    pub scope: String,
    pub embedding_resident: bool,
    pub embedding_payload_bytes: usize,
    pub embedding_payload_sha256: String,
    pub plan_digest: String,
    pub generation_tokens: usize,
    pub cuda_profiler_capture_requested: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer_timing: Option<DeviceTokenHandoffLayerTimingReport>,
    pub prefill_host_wall_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batched_prefill: Option<AutoregressiveBatchedPrefillReport>,
    pub aggregate_decode_host_wall_ms: f64,
    pub aggregate_cuda_event_ms: f64,
    pub visible_tokens_per_second: f64,
    pub cuda_event_tokens_per_second: f64,
    pub top2_kernel_count: usize,
    pub embedding_gather_count: usize,
    pub inter_layer_d2d_copy_count: usize,
    pub inter_layer_d2d_bytes: usize,
    pub full_logits_d2h_bytes_avoided: usize,
    pub recurrent_embedding_h2d_bytes_avoided: usize,
    pub compact_receipt_d2h_bytes: usize,
    pub receipt_sha256: String,
    pub terminal_poll_iterations: u64,
    pub owner_epoch_proof: bool,
    pub generated_token_ids: Vec<usize>,
    pub receipts: Vec<DeviceTokenHandoffReceiptReport>,
    pub generated_tokens_match_audit: bool,
    pub top2_bits_match_audit: bool,
    pub final_hidden_matches_audit: bool,
    pub final_state_matches_audit: bool,
    pub terminal_state_receipt_contract: bool,
    pub complete_numerical_contract: bool,
    pub measured_boundary_contract_closed: bool,
    pub prefill_boundaries: BoundaryReport,
    pub execution_boundaries: BoundaryReport,
    pub receipt_readback_boundaries: BoundaryReport,
    pub state_readback_boundaries: BoundaryReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resident_auxiliary: Option<ResidentAuxiliaryReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub terminal_state_receipt: Option<TerminalStateReceiptReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub terminal_state_receipt_expected_digest_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv8_ram_persistence: Option<Kv8RamPersistenceReport>,
    pub full_logits_readback_per_token: bool,
    pub cpu_argmax_on_measured_path: bool,
    pub host_embedding_lookup_after_first_output: bool,
    pub product_tps_permitted: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct TerminalStateReceiptReport {
    pub algorithm: String,
    pub component_count: usize,
    pub payload_bytes: usize,
    pub d2h_bytes: usize,
    pub digest_sha256: String,
    pub state_readback_avoided: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct Kv8RamPersistenceReport {
    pub scope: String,
    pub session_id: String,
    pub generation: u64,
    pub identity_sha256: String,
    pub page_ledger_sha256: String,
    pub capacity_positions: usize,
    pub populated_positions: usize,
    pub full_attention_layer_indices: Vec<usize>,
    pub plane_count: usize,
    pub row_bytes: usize,
    pub plane_bytes: usize,
    pub page_positions: usize,
    pub page_bytes: usize,
    pub pages_per_plane: usize,
    pub page_count: usize,
    pub lane_count: usize,
    pub total_payload_bytes: usize,
    pub source_snapshot_sha256: String,
    pub restored_snapshot_sha256: String,
    pub restored_mismatched_pages: usize,
    pub zero_sampled_pages: usize,
    pub zero_sample_nonzero_bytes: u64,
    pub export_host_wall_ms: f64,
    pub clear_host_wall_ms: f64,
    pub zero_sample_host_wall_ms: f64,
    pub restore_host_wall_ms: f64,
    pub verify_host_wall_ms: f64,
    pub total_host_wall_ms: f64,
    pub export_gb_per_second: f64,
    pub restore_gb_per_second: f64,
    pub stream_owners: Vec<Kv8RamPersistenceStreamOwnerReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub linear_state_persistence: Option<LinearStatePersistenceReport>,
    pub export_boundaries: BoundaryReport,
    pub clear_boundaries: BoundaryReport,
    pub zero_sample_boundaries: BoundaryReport,
    pub restore_boundaries: BoundaryReport,
    pub verify_boundaries: BoundaryReport,
    pub stream_close_boundaries: BoundaryReport,
    pub total_boundaries: BoundaryReport,
    pub boundary_ledger_closed: bool,
    pub allocation_cleanup_zero: bool,
    pub q8_cache_geometry_exact: bool,
    pub source_export_complete: bool,
    pub clear_sample_zero: bool,
    pub restore_byte_exact: bool,
    pub transfer_copy_path_async: bool,
    pub stream_owner_identity_closed: bool,
    pub first_suffix_token_matches_control: bool,
    pub first_suffix_top2_bits_match_control: bool,
    pub final_hidden_matches_control: bool,
    pub final_state_matches_control: bool,
    pub complete_numerical_contract: bool,
    pub persistence_wall_excluded_from_decode_timing: bool,
    pub real_kv8_cache_roundtrip_executed: bool,
    pub inactive_session_product_ready: bool,
    pub warm_ttft_qualified: bool,
    pub product_tps_permitted: bool,
    pub default_cli_enabled: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct LinearStatePersistenceReport {
    pub scope: String,
    pub history_buffer_count: usize,
    pub state_buffer_count: usize,
    pub buffer_count: usize,
    pub history_buffer_bytes: usize,
    pub state_buffer_bytes: usize,
    pub total_payload_bytes: usize,
    pub source_snapshot_sha256: String,
    pub restored_snapshot_sha256: String,
    pub restored_mismatched_buffers: usize,
    pub zero_sampled_buffers: usize,
    pub zero_sample_nonzero_bytes: u64,
    pub export_host_wall_ms: f64,
    pub clear_host_wall_ms: f64,
    pub zero_sample_host_wall_ms: f64,
    pub restore_host_wall_ms: f64,
    pub verify_host_wall_ms: f64,
    pub total_host_wall_ms: f64,
    pub export_gb_per_second: f64,
    pub restore_gb_per_second: f64,
    pub export_boundaries: BoundaryReport,
    pub clear_boundaries: BoundaryReport,
    pub zero_sample_boundaries: BoundaryReport,
    pub restore_boundaries: BoundaryReport,
    pub verify_boundaries: BoundaryReport,
    pub total_boundaries: BoundaryReport,
    pub boundary_ledger_closed: bool,
    pub allocation_cleanup_zero: bool,
    pub geometry_exact: bool,
    pub source_export_complete: bool,
    pub clear_sample_zero: bool,
    pub restore_byte_exact: bool,
    pub transfer_copy_path_async: bool,
    pub complete_numerical_contract: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct Kv8RamPersistenceStreamOwnerReport {
    pub lane: usize,
    pub owner_instance_key: usize,
    pub close_epoch: u64,
    pub close_receipt_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct DeviceTokenHandoffLayerTimingReport {
    pub scope: String,
    pub generation_tokens: usize,
    pub layers_per_token: usize,
    pub linear_layers_per_token: usize,
    pub full_attention_layers_per_token: usize,
    pub full_attention_layer_indices: Vec<usize>,
    pub event_records_per_token: usize,
    pub token_instrumented_cuda_event_ms: Vec<f64>,
    pub token_layer_cuda_event_ms: Vec<Vec<f64>>,
    pub token_full_attention_component_cuda_event_ms: Vec<Vec<f64>>,
    pub token_post_backbone_cuda_event_ms: Vec<f64>,
    pub aggregate_instrumented_cuda_event_ms: f64,
    pub aggregate_layer_cuda_event_ms: f64,
    pub aggregate_linear_layer_cuda_event_ms: f64,
    pub aggregate_full_layer_cuda_event_ms: f64,
    pub aggregate_full_attention_component_cuda_event_ms: f64,
    pub aggregate_post_backbone_cuda_event_ms: f64,
    pub accounting_relative_error: f64,
    pub measurement_perturbs_tps: bool,
    pub product_tps_permitted: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct DeviceTokenHandoffReceiptReport {
    pub generation_index: usize,
    pub position: usize,
    pub input_token_id: usize,
    pub token_id: usize,
    pub top1_value_bits: u32,
    pub runner_up_id: usize,
    pub runner_up_value_bits: u32,
    pub top1_margin: f64,
    pub finite_count: usize,
    pub step_generation: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct AutoregressiveTimingReport {
    pub input_and_model_setup_host_wall_ms: f64,
    pub warmup_sequences: usize,
    pub prompt_token_count: usize,
    pub unmeasured_prefill_token_count: usize,
    pub generated_token_count: usize,
    pub primary_prefill_host_wall_ms: f64,
    pub primary_decode_host_wall_ms: f64,
    pub primary_visible_tokens_per_second: f64,
    pub primary_p50_token_ms: f64,
    pub primary_p95_token_ms: f64,
    pub replay_prefill_host_wall_ms: f64,
    pub replay_decode_host_wall_ms: f64,
    pub replay_visible_tokens_per_second: f64,
    pub replay_p50_token_ms: f64,
    pub replay_p95_token_ms: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct AutoregressiveSequenceReport {
    pub prompt_prefix_tokens_processed: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batched_prefill: Option<AutoregressiveBatchedPrefillReport>,
    pub generated_token_count: usize,
    pub aggregate_decode_host_wall_ms: f64,
    pub aggregate_cuda_event_ms: f64,
    pub visible_tokens_per_second: f64,
    pub cuda_event_tokens_per_second: f64,
    pub p50_token_ms: f64,
    pub p95_token_ms: f64,
    pub terminal_poll_iterations: u64,
    pub owner_epoch_proof: bool,
    pub inter_layer_d2d_copy_count: usize,
    pub inter_layer_d2d_bytes: usize,
    pub final_hidden_sha256: String,
    pub final_state_sha256: Vec<String>,
    pub top1_margin_distribution: AutoregressiveMarginDistributionReport,
    pub tokens: Vec<AutoregressiveTokenReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resident_auxiliary: Option<ResidentAuxiliaryReport>,
}

/// Prefill execution and owner release, separate from model loading, schedule
/// construction, auxiliary upload and decode. Snapshots are taken only at
/// quiescent boundaries in this single-owner diagnostic runner.
#[derive(Clone, Debug, Serialize)]
pub struct AutoregressiveBatchedPrefillReport {
    pub attention_path: String,
    pub prefix_tokens: usize,
    pub chunk_tokens: usize,
    pub chunk_count: usize,
    pub compute_host_wall_ms: f64,
    pub cuda_event_ms: f64,
    pub execution_boundaries: BoundaryReport,
    pub submission_boundaries: BoundaryReport,
    pub execution_allocations: AllocationReport,
    pub execution_allocation_stable: bool,
    pub resident_live_bytes_before_build: u64,
    pub resident_live_allocations_before_build: u64,
    pub live_bytes_before_release: u64,
    pub live_allocations_before_release: u64,
    pub workspace_release_host_wall_ms: f64,
    pub workspace_released_bytes: u64,
    pub workspace_released_allocations: u64,
    pub live_bytes_after_release: u64,
    pub workspace_released_before_auxiliary_and_decode: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct AutoregressiveMarginDistributionReport {
    pub minimum: f64,
    pub p01: f64,
    pub p05: f64,
    pub p50: f64,
    pub p95: f64,
    pub below_0_01_count: usize,
    pub below_0_05_count: usize,
    pub below_0_10_count: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct AutoregressiveTokenReport {
    pub generation_index: usize,
    pub position: usize,
    pub input_token_id: usize,
    pub output_token_id: usize,
    pub host_wall_ms: f64,
    pub cuda_event_ms: f64,
    pub top1_margin: f64,
    pub logits_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct AutoregressiveDeterminismReport {
    pub generated_tokens_exact: bool,
    pub logits_exact: bool,
    pub final_hidden_exact: bool,
    pub final_state_exact: bool,
    pub complete_replay_exact: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct AutoregressiveResourceReport {
    pub setup_boundaries: BoundaryReport,
    pub warmup_boundaries: BoundaryReport,
    pub primary_reset_boundaries: BoundaryReport,
    pub primary_execution_boundaries: BoundaryReport,
    pub primary_state_readback_boundaries: BoundaryReport,
    pub replay_reset_boundaries: BoundaryReport,
    pub replay_execution_boundaries: BoundaryReport,
    pub replay_state_readback_boundaries: BoundaryReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_token_handoff_reset_boundaries: Option<BoundaryReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_token_handoff_prefill_boundaries: Option<BoundaryReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_token_handoff_kv8_persistence_boundaries: Option<BoundaryReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_token_handoff_execution_boundaries: Option<BoundaryReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_token_handoff_receipt_readback_boundaries: Option<BoundaryReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_token_handoff_state_readback_boundaries: Option<BoundaryReport>,
    pub cleanup_boundaries: BoundaryReport,
    pub total_boundaries: BoundaryReport,
    pub boundary_ledger_closed: bool,
    pub setup_allocations: AllocationReport,
    pub final_allocation_delta: AllocationReport,
}

#[derive(Clone, Debug, Serialize)]
pub struct AutoregressiveCheckReport {
    pub exact_model_profile: bool,
    pub tokenizer_bound: bool,
    pub position_extent_valid: bool,
    pub all_kernel_features_enabled: bool,
    pub all_logits_finite: bool,
    pub all_tokens_in_vocabulary: bool,
    pub owner_epoch_proof: bool,
    pub generated_token_count_valid: bool,
    pub replay_exact: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_token_handoff_numerical_contract: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub zero_copy_layer_handoff_contract: Option<bool>,
    pub boundary_ledger_closed: bool,
    pub cleanup_zero: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct AutoregressiveClaimsReport {
    pub diagnostic_autoregressive_executed: bool,
    pub semantic_parity_established: bool,
    pub llama_parity_established: bool,
    pub product_tps_permitted: bool,
    pub mtp_or_dflash_executed: bool,
    pub default_cli_enabled: bool,
}

struct LoadedTensor {
    descriptor: GgufTensor,
    payload: Vec<u8>,
}

impl LoadedTensor {
    fn decode(&self) -> Result<Vec<f32>, Qwen35Error> {
        Ok(dequantize_tensor(&self.descriptor, &self.payload)?)
    }
}

struct FfnMaterial {
    gate: LoadedTensor,
    up: LoadedTensor,
    down: LoadedTensor,
    weight_type: u32,
    gate_scale: f32,
    up_scale: f32,
    down_scale: f32,
    // NVFP4 layers require these metadata tensors, but the current fused
    // numerical ABI intentionally leaves the input scales unapplied. Q8_0
    // layers have no companion metadata and use unit multipliers.
    _input_scales: [f32; 3],
}

struct LinearMaterial {
    spec: Qwen35LayerSpec,
    attn_norm: Vec<f32>,
    qkv: LoadedTensor,
    z: LoadedTensor,
    alpha: LoadedTensor,
    beta: LoadedTensor,
    conv_weights: Vec<f32>,
    dt_bias: Vec<f32>,
    ssm_a: Vec<f32>,
    ssm_norm: Vec<f32>,
    ssm_out: LoadedTensor,
    post_norm: Vec<f32>,
    ffn: FfnMaterial,
}

struct FullMaterial {
    spec: Qwen35LayerSpec,
    attn_norm: Vec<f32>,
    q: LoadedTensor,
    k: LoadedTensor,
    v: LoadedTensor,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    attn_output: LoadedTensor,
    post_norm: Vec<f32>,
    ffn: FfnMaterial,
}

struct OutputHeadMaterial {
    norm: Vec<f32>,
    weight: LoadedTensor,
}

enum LayerMaterial {
    Linear(LinearMaterial),
    Full(FullMaterial),
}

enum ProjectedInput {
    Linear(linear::ProjectedToken),
}

impl LayerMaterial {
    fn kind(&self) -> Qwen35LayerKind {
        match self {
            Self::Linear(_) => Qwen35LayerKind::LinearAttention,
            Self::Full(_) => Qwen35LayerKind::FullAttention,
        }
    }

    fn spec(&self) -> &Qwen35LayerSpec {
        match self {
            Self::Linear(material) => &material.spec,
            Self::Full(material) => &material.spec,
        }
    }
}

fn project_linear(
    material: &LinearMaterial,
    hidden: &[f32],
    epsilon: f32,
) -> Result<linear::ProjectedToken, Qwen35MixedPrefixCudaError> {
    Ok(linear::project_token(
        hidden,
        &material.attn_norm,
        &material.qkv.descriptor,
        &material.qkv.payload,
        &material.z.descriptor,
        &material.z.payload,
        &material.alpha.descriptor,
        &material.alpha.payload,
        &material.beta.descriptor,
        &material.beta.payload,
        &material.dt_bias,
        &material.ssm_a,
        epsilon,
    )?)
}

fn invalid(detail: impl Into<String>) -> Qwen35MixedPrefixCudaError {
    Qwen35MixedPrefixCudaError::InvalidConfiguration(detail.into())
}

fn require_tensor(
    manifest: &GgufManifest,
    name: &str,
    dimensions: &[u64],
    ggml_type: u32,
) -> Result<GgufTensor, Qwen35MixedPrefixCudaError> {
    let descriptor = manifest
        .tensor(name)
        .ok_or_else(|| invalid(format!("missing tensor {name:?}")))?;
    if descriptor.dimensions != dimensions {
        return Err(invalid(format!(
            "tensor {name:?} dimensions: expected {dimensions:?}, got {:?}",
            descriptor.dimensions
        )));
    }
    if descriptor.ggml_type != ggml_type {
        return Err(invalid(format!(
            "tensor {name:?} type: expected {ggml_type}, got {}",
            descriptor.ggml_type
        )));
    }
    Ok(descriptor.clone())
}

fn load_tensor(
    reader: &mut GgufTensorReader<'_>,
    manifest: &GgufManifest,
    name: &str,
    dimensions: &[u64],
    ggml_type: u32,
) -> Result<LoadedTensor, Qwen35MixedPrefixCudaError> {
    let descriptor = require_tensor(manifest, name, dimensions, ggml_type)?;
    let payload = reader.read_tensor_bytes(&descriptor.name)?;
    Ok(LoadedTensor {
        descriptor,
        payload,
    })
}

fn load_dense(
    reader: &mut GgufTensorReader<'_>,
    manifest: &GgufManifest,
    name: &str,
    dimensions: &[u64],
    ggml_type: u32,
) -> Result<Vec<f32>, Qwen35MixedPrefixCudaError> {
    let tensor = load_tensor(reader, manifest, name, dimensions, ggml_type)?;
    let values = tensor.decode()?;
    let expected = dimensions.iter().try_fold(1usize, |acc, value| {
        acc.checked_mul(usize::try_from(*value).ok()?)
    });
    if expected != Some(values.len()) || values.iter().any(|value| !value.is_finite()) {
        return Err(invalid(format!(
            "tensor {name:?} decoded geometry or finiteness"
        )));
    }
    Ok(values)
}

fn load_output_head_material(
    reader: &mut GgufTensorReader<'_>,
    manifest: &GgufManifest,
    config: &Qwen35Config,
) -> Result<OutputHeadMaterial, Qwen35MixedPrefixCudaError> {
    let hidden = config.hidden_size;
    let vocabulary = config.vocab_size;
    Ok(OutputHeadMaterial {
        norm: load_dense(
            reader,
            manifest,
            "output_norm.weight",
            &[hidden as u64],
            GGML_TYPE_F32,
        )?,
        weight: load_tensor(
            reader,
            manifest,
            "output.weight",
            &[hidden as u64, vocabulary as u64],
            GGML_TYPE_Q8_0,
        )?,
    })
}

fn positive_scale(values: &[f32], name: &str) -> Result<f32, Qwen35MixedPrefixCudaError> {
    if values.len() != 1 || !values[0].is_finite() || values[0] <= 0.0 {
        return Err(invalid(format!(
            "{name} must be one finite positive scalar"
        )));
    }
    Ok(values[0])
}

fn load_ffn(
    reader: &mut GgufTensorReader<'_>,
    manifest: &GgufManifest,
    config: &Qwen35Config,
    layer: usize,
    spec: &Qwen35LayerSpec,
) -> Result<FfnMaterial, Qwen35MixedPrefixCudaError> {
    let hidden = spec.hidden_size;
    let intermediate = spec.intermediate_size;
    let ffn_type = Qwen35TensorSchema::expected_ffn_weight_ggml_type(config, layer)?;
    if ffn_type != GGML_TYPE_NVFP4 && ffn_type != GGML_TYPE_Q8_0 {
        return Err(invalid(format!(
            "prefix layer {layer} has unsupported FFN type {ffn_type}"
        )));
    }
    let prefix = format!("blk.{layer}");
    let gate = load_tensor(
        reader,
        manifest,
        &format!("{prefix}.ffn_gate.weight"),
        &[hidden as u64, intermediate as u64],
        ffn_type,
    )?;
    let up = load_tensor(
        reader,
        manifest,
        &format!("{prefix}.ffn_up.weight"),
        &[hidden as u64, intermediate as u64],
        ffn_type,
    )?;
    let down = load_tensor(
        reader,
        manifest,
        &format!("{prefix}.ffn_down.weight"),
        &[intermediate as u64, hidden as u64],
        ffn_type,
    )?;
    let (gate_scale, up_scale, down_scale, input_scales) = if ffn_type == GGML_TYPE_NVFP4 {
        let gate_scale = positive_scale(
            &load_dense(
                reader,
                manifest,
                &format!("{prefix}.ffn_gate.scale"),
                &[1],
                GGML_TYPE_F32,
            )?,
            &format!("{prefix}.ffn_gate.scale"),
        )?;
        let up_scale = positive_scale(
            &load_dense(
                reader,
                manifest,
                &format!("{prefix}.ffn_up.scale"),
                &[1],
                GGML_TYPE_F32,
            )?,
            &format!("{prefix}.ffn_up.scale"),
        )?;
        let down_scale = positive_scale(
            &load_dense(
                reader,
                manifest,
                &format!("{prefix}.ffn_down.scale"),
                &[1],
                GGML_TYPE_F32,
            )?,
            &format!("{prefix}.ffn_down.scale"),
        )?;
        let input_scales = [
            positive_scale(
                &load_dense(
                    reader,
                    manifest,
                    &format!("{prefix}.ffn_gate.input_scale"),
                    &[1],
                    GGML_TYPE_F32,
                )?,
                &format!("{prefix}.ffn_gate.input_scale"),
            )?,
            positive_scale(
                &load_dense(
                    reader,
                    manifest,
                    &format!("{prefix}.ffn_up.input_scale"),
                    &[1],
                    GGML_TYPE_F32,
                )?,
                &format!("{prefix}.ffn_up.input_scale"),
            )?,
            positive_scale(
                &load_dense(
                    reader,
                    manifest,
                    &format!("{prefix}.ffn_down.input_scale"),
                    &[1],
                    GGML_TYPE_F32,
                )?,
                &format!("{prefix}.ffn_down.input_scale"),
            )?,
        ];
        (gate_scale, up_scale, down_scale, input_scales)
    } else {
        // Q8_0 encodes its per-block scales in the weight payload.  The
        // converted artifact deliberately has no tensor-level or input-scale
        // metadata for these final layers.
        (1.0, 1.0, 1.0, [1.0; 3])
    };
    Ok(FfnMaterial {
        gate,
        up,
        down,
        weight_type: ffn_type,
        gate_scale,
        up_scale,
        down_scale,
        _input_scales: input_scales,
    })
}

fn load_layer_materials(
    reader: &mut GgufTensorReader<'_>,
    manifest: &GgufManifest,
    config: &Qwen35Config,
    layer_count: usize,
) -> Result<Vec<LayerMaterial>, Qwen35MixedPrefixCudaError> {
    load_layer_material_range(reader, manifest, config, 0..layer_count)
}

fn validate_material_range(
    range: &std::ops::Range<usize>,
    layer_count: usize,
) -> Result<(), Qwen35MixedPrefixCudaError> {
    if range.start >= range.end || range.end > layer_count {
        return Err(invalid("layer material range is outside the model profile"));
    }
    Ok(())
}

// A single-layer diagnostic must not load every preceding layer merely to
// index the final material. Full-chain callers retain the exact 0..N order.
fn load_layer_material_range(
    reader: &mut GgufTensorReader<'_>,
    manifest: &GgufManifest,
    config: &Qwen35Config,
    range: std::ops::Range<usize>,
) -> Result<Vec<LayerMaterial>, Qwen35MixedPrefixCudaError> {
    validate_material_range(&range, config.num_hidden_layers)?;
    let mut materials = Vec::with_capacity(range.len());
    for layer in range {
        let spec = Qwen35LayerSpec::from_config(config, layer)?;
        let prefix = format!("blk.{layer}");
        match spec.kind {
            Qwen35LayerKind::LinearAttention => {
                let ffn_type = Qwen35TensorSchema::expected_ffn_weight_ggml_type(config, layer)?;
                let _ = linear::required_descriptors(manifest, layer, ffn_type)?;
                let key_values = spec.linear_key_heads * spec.linear_head_dim;
                let value_values = spec.linear_value_heads * spec.linear_head_dim;
                let qkv_width = key_values * 2 + value_values;
                let attn_norm = load_dense(
                    reader,
                    manifest,
                    &format!("{prefix}.attn_norm.weight"),
                    &[spec.hidden_size as u64],
                    GGML_TYPE_F32,
                )?;
                let qkv = load_tensor(
                    reader,
                    manifest,
                    &format!("{prefix}.attn_qkv.weight"),
                    &[spec.hidden_size as u64, qkv_width as u64],
                    GGML_TYPE_Q8_0,
                )?;
                let z = load_tensor(
                    reader,
                    manifest,
                    &format!("{prefix}.attn_gate.weight"),
                    &[spec.hidden_size as u64, value_values as u64],
                    GGML_TYPE_Q8_0,
                )?;
                let alpha = load_tensor(
                    reader,
                    manifest,
                    &format!("{prefix}.ssm_alpha.weight"),
                    &[spec.hidden_size as u64, spec.linear_value_heads as u64],
                    GGML_TYPE_BF16,
                )?;
                let beta = load_tensor(
                    reader,
                    manifest,
                    &format!("{prefix}.ssm_beta.weight"),
                    &[spec.hidden_size as u64, spec.linear_value_heads as u64],
                    GGML_TYPE_BF16,
                )?;
                let ssm_out = load_tensor(
                    reader,
                    manifest,
                    &format!("{prefix}.ssm_out.weight"),
                    &[value_values as u64, spec.hidden_size as u64],
                    GGML_TYPE_Q8_0,
                )?;
                let conv_raw = load_dense(
                    reader,
                    manifest,
                    &format!("{prefix}.ssm_conv1d.weight"),
                    &[spec.linear_conv_kernel as u64, qkv_width as u64],
                    GGML_TYPE_F32,
                )?;
                let conv_weights = linear::convert_conv_layout(&conv_raw)?;
                let ssm_a = load_dense(
                    reader,
                    manifest,
                    &format!("{prefix}.ssm_a"),
                    &[spec.linear_value_heads as u64],
                    GGML_TYPE_F32,
                )?;
                let dt_bias = load_dense(
                    reader,
                    manifest,
                    &format!("{prefix}.ssm_dt.bias"),
                    &[spec.linear_value_heads as u64],
                    GGML_TYPE_F32,
                )?;
                let ssm_norm = load_dense(
                    reader,
                    manifest,
                    &format!("{prefix}.ssm_norm.weight"),
                    &[spec.linear_head_dim as u64],
                    GGML_TYPE_F32,
                )?;
                let post_norm = load_dense(
                    reader,
                    manifest,
                    &format!("{prefix}.post_attention_norm.weight"),
                    &[spec.hidden_size as u64],
                    GGML_TYPE_F32,
                )?;
                let ffn = load_ffn(reader, manifest, config, layer, &spec)?;
                materials.push(LayerMaterial::Linear(LinearMaterial {
                    spec,
                    attn_norm,
                    qkv,
                    z,
                    alpha,
                    beta,
                    conv_weights,
                    dt_bias,
                    ssm_a,
                    ssm_norm,
                    ssm_out,
                    post_norm,
                    ffn,
                }));
            }
            Qwen35LayerKind::FullAttention => {
                let _ = full::required_descriptors(manifest, layer)?;
                let query_values = spec.query_heads * spec.head_dim;
                let kv_values = spec.kv_heads * spec.head_dim;
                let attn_norm = load_dense(
                    reader,
                    manifest,
                    &format!("{prefix}.attn_norm.weight"),
                    &[spec.hidden_size as u64],
                    GGML_TYPE_F32,
                )?;
                let q = load_tensor(
                    reader,
                    manifest,
                    &format!("{prefix}.attn_q.weight"),
                    &[spec.hidden_size as u64, (query_values * 2) as u64],
                    GGML_TYPE_Q8_0,
                )?;
                let k = load_tensor(
                    reader,
                    manifest,
                    &format!("{prefix}.attn_k.weight"),
                    &[spec.hidden_size as u64, kv_values as u64],
                    GGML_TYPE_Q8_0,
                )?;
                let v = load_tensor(
                    reader,
                    manifest,
                    &format!("{prefix}.attn_v.weight"),
                    &[spec.hidden_size as u64, kv_values as u64],
                    GGML_TYPE_Q8_0,
                )?;
                let q_norm = load_dense(
                    reader,
                    manifest,
                    &format!("{prefix}.attn_q_norm.weight"),
                    &[spec.head_dim as u64],
                    GGML_TYPE_F32,
                )?;
                let k_norm = load_dense(
                    reader,
                    manifest,
                    &format!("{prefix}.attn_k_norm.weight"),
                    &[spec.head_dim as u64],
                    GGML_TYPE_F32,
                )?;
                let attn_output = load_tensor(
                    reader,
                    manifest,
                    &format!("{prefix}.attn_output.weight"),
                    &[query_values as u64, spec.hidden_size as u64],
                    GGML_TYPE_Q8_0,
                )?;
                let post_norm = load_dense(
                    reader,
                    manifest,
                    &format!("{prefix}.post_attention_norm.weight"),
                    &[spec.hidden_size as u64],
                    GGML_TYPE_F32,
                )?;
                let ffn = load_ffn(reader, manifest, config, layer, &spec)?;
                materials.push(LayerMaterial::Full(FullMaterial {
                    spec,
                    attn_norm,
                    q,
                    k,
                    v,
                    q_norm,
                    k_norm,
                    attn_output,
                    post_norm,
                    ffn,
                }));
            }
        }
    }
    Ok(materials)
}

// The plan owns every device buffer referenced by its bound operations. Some
// fields are intentionally never read after binding; dropping the plan is the
// ownership proof that releases those buffers together.
#[allow(dead_code)]
struct LinearPlan {
    qkv: OwnedCudaBuffer,
    z: OwnedCudaBuffer,
    conv: Weight,
    history: OwnedCudaBuffer,
    decay: OwnedCudaBuffer,
    beta: OwnedCudaBuffer,
    state: OwnedCudaBuffer,
    ssm_norm: Weight,
    activated: OwnedCudaBuffer,
    recurrent: OwnedCudaBuffer,
    linear_output: OwnedCudaBuffer,
    attention: OwnedCudaBuffer,
    input: OwnedCudaBuffer,
    input_view: CudaF32View,
    input_is_alias: bool,
    attention_residual: OwnedCudaBuffer,
    post_norm: Weight,
    ffn_gate_weight: Weight,
    ffn_up_weight: Weight,
    ffn_gate_output: OwnedCudaBuffer,
    ffn_up_output: OwnedCudaBuffer,
    ffn_activation: OwnedCudaBuffer,
    ffn_input_q8_1: OwnedCudaBuffer,
    ffn_down_q8_1: OwnedCudaBuffer,
    ffn_down_weight: Weight,
    ffn_output: Weight,
    final_hidden: OwnedCudaBuffer,
    output_weight: Weight,
    output_q8_1: OwnedCudaBuffer,
    post_norm_weight: Weight,
    ffn_weight_type: u32,
    ffn_gate_scale: f32,
    ffn_up_scale: f32,
    ffn_down_scale: f32,
    ffn_is_qpn: bool,
    #[cfg(feature = "qwen35-prefill-cublas")]
    qpn_weights: Option<[Rc<augur_backend::cuda_qwen35_qpn::OwnedCudaQwen35QpnWeightV1>; 3]>,
    projection: Option<LinearProjectionPlan>,
    chain: linear::BoundChain,
}

#[allow(dead_code)]
struct LinearProjectionPlan {
    input_weight_layout: PhysicalLayoutId,
    attn_norm_weight: Weight,
    qkv_weight: Weight,
    z_weight: Weight,
    alpha_weight: Weight,
    beta_weight: Weight,
    normalized: OwnedCudaBuffer,
    normalized_q8_1: OwnedCudaBuffer,
    alpha_raw: OwnedCudaBuffer,
    beta_raw: OwnedCudaBuffer,
    dt_bias: Weight,
    ssm_a: Weight,
    attn_norm_op: CudaPlanRmsnormF32,
    normalized_quantize_op: CudaQ8_1ActivationQuantizeArgsV1,
    qkv_op: row_input::Scalar,
    z_op: row_input::Scalar,
    alpha_op: CudaBf16F32MatvecArgsV1,
    beta_op: CudaBf16F32MatvecArgsV1,
    postprocess_op: CudaQwen35LinearProjectionPostprocessArgsV1,
}

#[allow(dead_code)]
struct FullPlan {
    input_weight_layout: PhysicalLayoutId,
    #[cfg(feature = "qwen35-prefill-cublas")]
    rotated_attention: Option<Arc<RotatedFullAttention>>,
    // Native projection workspace. The public query/gate/key/value buffers
    // below are retained as the attention ABI inputs and are filled by these
    // device-side operations for every token.
    attn_norm_weight: Weight,
    normalized: OwnedCudaBuffer,
    normalized_q8_1: OwnedCudaBuffer,
    q_weight: Weight,
    q_full: OwnedCudaBuffer,
    q_norm_weight: Weight,
    q_normed: OwnedCudaBuffer,
    k_weight: Weight,
    k_raw: OwnedCudaBuffer,
    k_norm_weight: Weight,
    k_normed: OwnedCudaBuffer,
    v_weight: Weight,
    // Immutable, request-local; all layer leases outlive final completion.
    rope_tables: Arc<FullRopeTables>,
    query: OwnedCudaBuffer,
    #[cfg(not(feature = "qwen35-grouped-gqa-online-softmax"))]
    gate: OwnedCudaBuffer,
    key: OwnedCudaBuffer,
    value: OwnedCudaBuffer,
    key_cache: OwnedCudaBuffer,
    value_cache: OwnedCudaBuffer,
    #[cfg(all(
        feature = "qwen35-prefill-cublas",
        feature = "qwen35-kv8-full-attention",
        not(feature = "qwen35-kv8-chunk-prefill")
    ))]
    prefill_key_cache_f32: OwnedCudaBuffer,
    #[cfg(all(
        feature = "qwen35-prefill-cublas",
        feature = "qwen35-kv8-full-attention",
        not(feature = "qwen35-kv8-chunk-prefill")
    ))]
    prefill_value_cache_f32: OwnedCudaBuffer,
    #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
    attention_workspace: OwnedCudaBuffer,
    attention: OwnedCudaBuffer,
    gated: OwnedCudaBuffer,
    #[cfg(not(feature = "qwen35-grouped-gqa-online-softmax"))]
    gated_q8_1: OwnedCudaBuffer,
    output: OwnedCudaBuffer,
    input: OwnedCudaBuffer,
    input_view: CudaF32View,
    input_is_alias: bool,
    attention_residual: OwnedCudaBuffer,
    post_norm: Weight,
    ffn_gate_weight: Weight,
    ffn_up_weight: Weight,
    ffn_gate_output: OwnedCudaBuffer,
    ffn_up_output: OwnedCudaBuffer,
    ffn_activation: OwnedCudaBuffer,
    ffn_input_q8_1: OwnedCudaBuffer,
    ffn_down_q8_1: OwnedCudaBuffer,
    ffn_down_weight: Weight,
    ffn_output: Weight,
    final_hidden: OwnedCudaBuffer,
    output_weight: Weight,
    post_norm_weight: Weight,
    ffn_weight_type: u32,
    ffn_gate_scale: f32,
    ffn_up_scale: f32,
    ffn_down_scale: f32,
    ffn_is_qpn: bool,
    #[cfg(feature = "qwen35-prefill-cublas")]
    qpn_weights: Option<[Rc<augur_backend::cuda_qwen35_qpn::OwnedCudaQwen35QpnWeightV1>; 3]>,
    #[cfg(not(feature = "qwen35-grouped-gqa-online-softmax"))]
    attention_args: Vec<CudaQwen35FullAttentionArgsV1>,
    #[cfg(all(
        feature = "qwen35-grouped-gqa-online-softmax",
        not(feature = "qwen35-kv8-full-attention")
    ))]
    attention_args: Vec<CudaQwen35FullAttentionArgsV2>,
    #[cfg(feature = "qwen35-kv8-full-attention")]
    attention_args: Vec<CudaQwen35Kv8FullAttentionArgsV3>,
    output_args: CudaQ8_0Q8_1MatvecArgsV1,
    attn_norm_op: CudaPlanRmsnormF32,
    qkv_quantize_op: CudaQ8_1ActivationQuantizeArgsV1,
    q_projection_op: row_input::Scalar,
    k_projection_op: row_input::Scalar,
    v_projection_op: row_input::Scalar,
    #[cfg(not(feature = "qwen35-fused-head-norm-rope"))]
    q_norm_ops: Vec<CudaRmsnormF32ArgsV1>,
    #[cfg(not(feature = "qwen35-fused-head-norm-rope"))]
    k_norm_ops: Vec<CudaRmsnormF32ArgsV1>,
    #[cfg(not(feature = "qwen35-fused-head-norm-rope"))]
    q_rope_ops: Vec<Vec<CudaRopeF32ArgsV1>>,
    #[cfg(not(feature = "qwen35-fused-head-norm-rope"))]
    k_rope_ops: Vec<Vec<CudaRopeF32ArgsV1>>,
    #[cfg(feature = "qwen35-fused-head-norm-rope")]
    fused_head_norm_rope_ops: Vec<CudaQwen35FusedHeadNormRopeArgsV1>,
    #[cfg(not(feature = "qwen35-grouped-gqa-online-softmax"))]
    output_quantize_op: CudaQ8_1ActivationQuantizeArgsV1,
    attention_residual_op: CudaAddF32ArgsV1,
    post_norm_op: CudaPlanRmsnormF32,
    ffn: linear::BoundFfn,
    final_residual_op: CudaAddF32ArgsV1,
}

#[allow(dead_code)]
enum HeadScalarProjection {
    Block(CudaQ8_0Q8_1MatvecArgsV1),
    #[cfg(feature = "qwen35-prefill-cublas")]
    Planar(augur_backend::CudaQwen35PlanarQ8HeadProjectionV1),
}
impl HeadScalarProjection {
    fn submit_for_plan(&self, stream: &OwnedCudaStream) -> std::result::Result<(), augur_backend::CudaRuntimeError> {
        match self {
            Self::Block(p) => p.submit_for_plan(stream),
            #[cfg(feature = "qwen35-prefill-cublas")]
            Self::Planar(p) => p.submit(stream),
        }
    }
}

#[allow(dead_code)]
struct DeviceOutputHeadPlan {
    norm_weight: Weight,
    normalized_hidden: OwnedCudaBuffer,
    activation_q8_1: OwnedCudaBuffer,
    output_weight: Weight,
    logits: OwnedCudaBuffer,
    norm_op: CudaPlanRmsnormF32,
    quantize_op: CudaQ8_1ActivationQuantizeArgsV1,
    projection_op: HeadScalarProjection,
    output_weight_layout: PhysicalLayoutId,
    output_weight_physical_sha256: String,
    output_weight_relayout_ms: f64,
    output_weight_payload_bytes: usize,
    output_weight_payload_sha256: String,
}

impl DeviceOutputHeadPlan {
    fn submit(&self, stream: &OwnedCudaStream) -> Result<(), Qwen35MixedPrefixCudaError> {
        self.norm_op.submit_for_plan(stream)?;
        self.quantize_op.submit_for_plan(stream)?;
        self.projection_op.submit_for_plan(stream)?;
        Ok(())
    }
}

enum DevicePlan {
    Linear(LinearPlan),
    Full(FullPlan),
}

#[cfg(feature = "qwen35-prefill-cublas")]
enum PrefillRopeBuffer {
    Owned(OwnedCudaBuffer),
    // Retain the complete immutable owner, never only its device pointer.
    Shared { tables: Arc<FullRopeTables>, cosine: bool },
}

#[cfg(feature = "qwen35-prefill-cublas")]
impl std::ops::Deref for PrefillRopeBuffer {
    type Target = OwnedCudaBuffer;
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Owned(buffer) => buffer,
            Self::Shared { tables, cosine } => if *cosine { &tables.cos } else { &tables.sin },
        }
    }
}

#[cfg(feature = "qwen35-prefill-cublas")]
struct PrefillWorkspace {
    max_tokens: usize,
    rope_capacity: usize,
    hidden_a: OwnedCudaBuffer,
    hidden_b: OwnedCudaBuffer,
    normalized: OwnedCudaBuffer,
    normalized_q8_1: OwnedCudaBuffer,
    qkv: OwnedCudaBuffer,
    z: OwnedCudaBuffer,
    alpha_raw: OwnedCudaBuffer,
    beta_raw: OwnedCudaBuffer,
    activated: OwnedCudaBuffer,
    recurrent: OwnedCudaBuffer,
    attention_output: OwnedCudaBuffer,
    attention_residual: OwnedCudaBuffer,
    post_norm: Rc<OwnedCudaBuffer>,
    ffn_gate: Rc<OwnedCudaBuffer>,
    ffn_up: Rc<OwnedCudaBuffer>,
    ffn_activation: Rc<OwnedCudaBuffer>,
    ffn_input_q8_1: OwnedCudaBuffer,
    ffn_down_q8_1: OwnedCudaBuffer,
    ffn_output: Rc<OwnedCudaBuffer>,
    rope_cos: PrefillRopeBuffer,
    rope_sin: PrefillRopeBuffer,
    #[cfg(feature = "qwen35-kv8-chunk-prefill")]
    kv8_key_work: OwnedCudaBuffer,
    #[cfg(feature = "qwen35-kv8-chunk-prefill")]
    kv8_value_work: OwnedCudaBuffer,
    #[cfg(feature = "qwen35-kv8-chunk-prefill")]
    dgemm_score_scratch: Option<OwnedCudaBuffer>,
    
    
    cublas: Rc<CudaQwen35PrefillCublasPlanV1>,
    qpn: Option<(Rc<OwnedCudaStream>, augur_backend::cuda_qwen35_prefill_cublas::CudaQwen35QpnFfnWorkspaceV1)>,
}

#[cfg(feature = "qwen35-prefill-cublas")]
struct PrefillLinearOps {
    simt_shared_gdn_scalars: bool,
    simt_shared_gdn_scalar_submissions: std::cell::Cell<u64>,
    attn_norm: CudaQwen35PrefillRmsnormV1,
    normalized_quantize: CudaQ8_1ActivationQuantizeArgsV1,
    qkv: CudaQwen35PrefillCublasProjectionV1,
    z: CudaQwen35PrefillCublasProjectionV1,
    alpha: CudaQwen35PrefillCublasProjectionV1,
    beta: CudaQwen35PrefillCublasProjectionV1,
    // Only the opt-in shadow verifier installs these canonical F32-input rows.
    canonical_gdn_scalars: Option<Vec<[CudaBf16F32MatvecArgsV1; 2]>>,
    // Exact token-grid scheduling for wide prefix and canonical B8. Serial
    // bindings remain the numerical reference and the small-width fallback.
    canonical_gdn_prefill_batch: Option<[augur_backend::CudaBf16F32MatvecRowsArgsV1; 2]>,
    canonical_gdn_token_grid_submissions: std::cell::Cell<u64>,
    recurrent: CudaQwen35PrefillLinearChunkV1,
    recurrent_quantize: CudaQ8_1ActivationQuantizeArgsV1,
    output: CudaQwen35PrefillCublasProjectionV1,
    attention_residual: CudaAddF32ArgsV1,
    post_norm: CudaQwen35PrefillRmsnormV1,
    ffn: PrefillFfn,
    final_residual: CudaAddF32ArgsV1,
    
    
}





#[cfg(feature = "qwen35-prefill-cublas")]
struct PrefillFullOps {
    #[cfg(feature = "qwen35-kv8-chunk-prefill")]
    simt_shared_softmax: bool,
    #[cfg(feature = "qwen35-kv8-chunk-prefill")]
    simt_softmax_tile16: bool,
    #[cfg(feature = "qwen35-kv8-chunk-prefill")]
    simt_unit_previous_scale: bool,
    
    attn_norm: CudaQwen35PrefillRmsnormV1,
    normalized_quantize: CudaQ8_1ActivationQuantizeArgsV1,
    q: CudaQwen35PrefillCublasProjectionV1,
    k: CudaQwen35PrefillCublasProjectionV1,
    v: CudaQwen35PrefillCublasProjectionV1,
    #[cfg(not(feature = "qwen35-kv8-chunk-prefill"))]
    attention: CudaQwen35PrefillFullAttentionV1,
    #[cfg(feature = "qwen35-kv8-chunk-prefill")]
    attention: CudaQwen35PrefillKv8ChunkAttentionV1,
    #[cfg(feature = "qwen35-kv8-chunk-prefill")]
    dgemm_attention: Option<CudaQwen35PrefillKv8ChunkAttentionV1>,
    
    #[cfg(all(
        feature = "qwen35-kv8-full-attention",
        not(feature = "qwen35-kv8-chunk-prefill")
    ))]
    cache_quantize: CudaQwen35PrefillKv8CacheQuantizeV1,
    gated_quantize: CudaQ8_1ActivationQuantizeArgsV1,
    output: CudaQwen35PrefillCublasProjectionV1,
    attention_residual: CudaAddF32ArgsV1,
    post_norm: CudaQwen35PrefillRmsnormV1,
    ffn: PrefillFfn,
    final_residual: CudaAddF32ArgsV1,
    
}





#[cfg(feature = "qwen35-prefill-cublas")]
enum PrefillLayerOps {
    Linear(PrefillLinearOps),
    Full(PrefillFullOps),
    
}

#[cfg(feature = "qwen35-prefill-cublas")]
struct PrefillBackboneOracle {
    final_hidden: Vec<f32>,
    states: Vec<(String, Vec<f32>)>,
    completion_poll_iterations: u64,
    owner_epoch_proof: bool,
}

#[cfg(feature = "qwen35-prefill-cublas")]
struct PrefillChunkSchedule {
    chunk_tokens: usize,
    chunks: Vec<PrefillChunk>,
    #[cfg(feature = "qwen35-kv8-chunk-prefill")]
    sparse_profile: Option<sparse_prefill_profile::SparsePrefillProfile>,
}

#[cfg(feature = "qwen35-prefill-cublas")]
struct PrefillChunk {
    tokens: usize,
    operations: Vec<PrefillLayerOps>,
}

#[cfg(feature = "qwen35-prefill-cublas")]
struct PrefillChunkRun {
    final_hidden: Option<Vec<f32>>,
    states: Option<Vec<(String, Vec<f32>)>>,
    normalized_hidden: Option<Vec<f32>>,
    logits: Option<Vec<f32>>,
    cuda_event_ms: f64,
    host_wall_ms: f64,
    submission_boundaries: BoundaryReport,
    completion_poll_iterations: u64,
    owner_epoch_proof: bool,
    progress_receipts: Vec<PrefillChunkProgressReceipt>,
}

#[cfg(feature = "qwen35-prefill-cublas")]
fn progress_receipt_sequence_valid(
    receipts: &[PrefillChunkProgressReceipt],
    total_tokens: usize,
) -> bool {
    if receipts.is_empty() || total_tokens == 0 {
        return false;
    }
    let mut next_position = 0_usize;
    let mut previous_epoch = None;
    for (index, receipt) in receipts.iter().enumerate() {
        if receipt.chunk_index != index
            || receipt.start_position != next_position
            || receipt.end_position_exclusive <= receipt.start_position
            || receipt.token_count
                != receipt
                    .end_position_exclusive
                    .saturating_sub(receipt.start_position)
            || !receipt.owner_epoch_proof
            || receipt.terminal != (index + 1 == receipts.len())
            || previous_epoch.is_some_and(|epoch| receipt.submission_epoch <= epoch)
        {
            return false;
        }
        next_position = receipt.end_position_exclusive;
        previous_epoch = Some(receipt.submission_epoch);
    }
    next_position == total_tokens
}

#[cfg(feature = "qwen35-prefill-cublas")]
fn compare_progress_receipts(
    primary: &[PrefillChunkProgressReceipt],
    reference: &[PrefillChunkProgressReceipt],
) -> Vec<PrefillChunkBoundaryComparison> {
    let mut comparisons = Vec::new();
    for primary_receipt in primary {
        let Some(reference_receipt) = reference.iter().find(|candidate| {
            candidate.end_position_exclusive == primary_receipt.end_position_exclusive
        }) else {
            continue;
        };
        let (state_cache_exact, first_different_layer, different_layer_count) = match (
            primary_receipt.state_cache_layer_sha256.as_deref(),
            reference_receipt.state_cache_layer_sha256.as_deref(),
        ) {
            (Some(primary_layers), Some(reference_layers))
                if primary_layers.len() == reference_layers.len() =>
            {
                let mut first = None;
                let mut count = 0_usize;
                for (layer, (actual, expected)) in
                    primary_layers.iter().zip(reference_layers).enumerate()
                {
                    if actual != expected {
                        first.get_or_insert(layer);
                        count += 1;
                    }
                }
                (Some(count == 0), first, count)
            }
            _ => (None, None, 0),
        };
        comparisons.push(PrefillChunkBoundaryComparison {
            end_position_exclusive: primary_receipt.end_position_exclusive,
            primary_chunk_index: primary_receipt.chunk_index,
            reference_chunk_index: reference_receipt.chunk_index,
            output_hidden_exact: primary_receipt
                .output_hidden_sha256
                .as_ref()
                .zip(reference_receipt.output_hidden_sha256.as_ref())
                .map(|(actual, expected)| actual == expected),
            state_cache_exact,
            first_different_layer,
            different_layer_count,
        });
    }
    comparisons
}

#[cfg(feature = "qwen35-prefill-cublas")]
impl PrefillWorkspace {
    fn enable_qpn(&mut self, stream: Rc<OwnedCudaStream>) -> Result<(), Qwen35MixedPrefixCudaError> {
        if self.qpn.is_some() { return Err(invalid("QPN batch workspace already bound")); }
        let workspace = augur_backend::cuda_qwen35_prefill_cublas::CudaQwen35QpnFfnWorkspaceV1::from_shared_buffers(
            Rc::clone(&stream), linear::INTERMEDIATE, self.max_tokens,
            Rc::clone(&self.ffn_gate), Rc::clone(&self.ffn_up), Rc::clone(&self.ffn_activation))?;
        self.qpn = Some((stream, workspace));
        Ok(())
    }

    fn bind_qpn_ffn(&self,
        weights: &[Rc<augur_backend::cuda_qwen35_qpn::OwnedCudaQwen35QpnWeightV1>; 3], tokens: usize,
    ) -> Result<PrefillFfn, Qwen35MixedPrefixCudaError> {
        let (stream, workspace) = self.qpn.as_ref()
            .ok_or_else(|| invalid("QPN batch weights require retained same-stream workspace"))?;
        Ok(PrefillFfn::Qpn(augur_backend::cuda_qwen35_prefill_cublas::CudaQwen35QpnFfnProgramV1::bind(
            Rc::clone(&self.cublas), Rc::clone(stream), weights.clone(), workspace,
            Rc::clone(&self.post_norm), 0, Rc::clone(&self.ffn_output), 0, tokens)?))
    }

    fn new(
        stream: &OwnedCudaStream,
        max_tokens: usize,
    ) -> Result<Self, Qwen35MixedPrefixCudaError> {
        Self::new_with_rope_capacity(stream, max_tokens, max_tokens)
    }

    fn new_with_rope_capacity(
        stream: &OwnedCudaStream,
        max_tokens: usize,
        rope_capacity: usize,
    ) -> Result<Self, Qwen35MixedPrefixCudaError> {
        Self::new_with_rope_lease(stream, max_tokens, rope_capacity, None)
    }

    fn new_with_rope_lease(
        stream: &OwnedCudaStream,
        max_tokens: usize,
        rope_capacity: usize,
        shared_rope: Option<(Arc<FullRopeTables>, f32)>,
    ) -> Result<Self, Qwen35MixedPrefixCudaError> {
        if max_tokens == 0 || rope_capacity < max_tokens {
            return Err(invalid("prefill workspace token capacity"));
        }
        if let Some((tables, theta)) = &shared_rope {
            tables.validate(stream, *theta, rope_capacity, full::ROTARY_DIM)?;
        }
        const MAX_QKV: usize = full::QUERY_VALUES * 2;
        let device = stream.device_ordinal();
        let batch = |width: usize| {
            max_tokens
                .checked_mul(width)
                .ok_or_else(|| invalid("prefill workspace extent overflow"))
        };
        let max_weight_elements = linear::INTERMEDIATE
            .checked_mul(linear::HIDDEN)
            .ok_or_else(|| invalid("prefill cuBLAS weight extent overflow"))?;
        Ok(Self {
            max_tokens,
            rope_capacity,
            hidden_a: alloc_f32(device, batch(linear::HIDDEN)?)?,
            hidden_b: alloc_f32(device, batch(linear::HIDDEN)?)?,
            normalized: alloc_f32(device, batch(linear::HIDDEN)?)?,
            normalized_q8_1: alloc_q8_1(device, batch(linear::HIDDEN)?)?,
            qkv: alloc_f32(device, batch(MAX_QKV)?)?,
            z: alloc_f32(device, batch(linear::RECURRENT_ELEMENTS)?)?,
            alpha_raw: alloc_f32(device, batch(linear::VALUE_HEADS)?)?,
            beta_raw: alloc_f32(device, batch(linear::VALUE_HEADS)?)?,
            activated: alloc_f32(device, batch(linear::QKV_ELEMENTS)?)?,
            recurrent: alloc_f32(device, batch(linear::RECURRENT_ELEMENTS)?)?,
            attention_output: alloc_f32(device, batch(full::QUERY_VALUES)?)?,
            attention_residual: alloc_f32(device, batch(linear::HIDDEN)?)?,
            post_norm: Rc::new(alloc_f32(device, batch(linear::HIDDEN)?)?),
            ffn_gate: Rc::new(alloc_f32(device, batch(linear::INTERMEDIATE)?)?),
            ffn_up: Rc::new(alloc_f32(device, batch(linear::INTERMEDIATE)?)?),
            ffn_activation: Rc::new(alloc_f32(device, batch(linear::INTERMEDIATE)?)?),
            ffn_input_q8_1: alloc_q8_1(device, batch(linear::HIDDEN)?)?,
            ffn_down_q8_1: alloc_q8_1(device, batch(linear::INTERMEDIATE)?)?,
            ffn_output: Rc::new(alloc_f32(device, batch(linear::HIDDEN)?)?),
            rope_cos: if let Some((tables, _)) = &shared_rope {
                PrefillRopeBuffer::Shared { tables: Arc::clone(tables), cosine: true }
            } else { PrefillRopeBuffer::Owned(alloc_f32(
                device,
                rope_capacity
                    .checked_mul(full::ROTARY_DIM / 2)
                    .ok_or_else(|| invalid("prefill RoPE workspace extent overflow"))?,
            )?) },
            rope_sin: if let Some((tables, _)) = &shared_rope {
                PrefillRopeBuffer::Shared { tables: Arc::clone(tables), cosine: false }
            } else { PrefillRopeBuffer::Owned(alloc_f32(
                device,
                rope_capacity
                    .checked_mul(full::ROTARY_DIM / 2)
                    .ok_or_else(|| invalid("prefill RoPE workspace extent overflow"))?,
            )?) },
            #[cfg(feature = "qwen35-kv8-chunk-prefill")]
            kv8_key_work: alloc_f32(device, batch(full::KV_VALUES)?)?,
            #[cfg(feature = "qwen35-kv8-chunk-prefill")]
            kv8_value_work: alloc_f32(device, batch(full::KV_VALUES)?)?,
            #[cfg(feature = "qwen35-kv8-chunk-prefill")]
            dgemm_score_scratch: None,
            
            
            qpn: None,
            cublas: Rc::new(CudaQwen35PrefillCublasPlanV1::new(
                stream,
                max_weight_elements,
                linear::INTERMEDIATE,
                max_tokens,
            )?),
        })
    }

    fn hidden_input(
        &self,
        layer: usize,
        tokens: usize,
    ) -> Result<CudaF32View, Qwen35MixedPrefixCudaError> {
        prefill_f32_view(
            if layer.is_multiple_of(2) {
                &self.hidden_a
            } else {
                &self.hidden_b
            },
            tokens
                .checked_mul(linear::HIDDEN)
                .ok_or_else(|| invalid("prefill hidden input extent overflow"))?,
        )
    }

    fn hidden_output(
        &self,
        layer: usize,
        tokens: usize,
    ) -> Result<CudaF32View, Qwen35MixedPrefixCudaError> {
        prefill_f32_view(
            if layer.is_multiple_of(2) {
                &self.hidden_b
            } else {
                &self.hidden_a
            },
            tokens
                .checked_mul(linear::HIDDEN)
                .ok_or_else(|| invalid("prefill hidden output extent overflow"))?,
        )
    }

    fn build_linear_ops(
        &self,
        stream: &OwnedCudaStream,
        plan: &LinearPlan,
        layer: usize,
        tokens: usize,
        epsilon: f32,
    ) -> Result<PrefillLinearOps, Qwen35MixedPrefixCudaError> {
        self.build_linear_ops_at(stream, plan, layer, tokens, 0, epsilon)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_linear_ops_at(
        &self,
        stream: &OwnedCudaStream,
        plan: &LinearPlan,
        layer: usize,
        tokens: usize,
        row_offset: usize,
        epsilon: f32,
    ) -> Result<PrefillLinearOps, Qwen35MixedPrefixCudaError> {
if tokens == 0
            || row_offset
                .checked_add(tokens)
                .is_none_or(|end| end > self.max_tokens)
        {
            return Err(invalid("prefill linear token extent"));
        }
        let projection = plan
            .projection
            .as_ref()
            .ok_or_else(|| invalid("prefill linear projection plan"))?;
        let hidden_elements = tokens
            .checked_mul(linear::HIDDEN)
            .ok_or_else(|| invalid("prefill hidden extent overflow"))?;
        let hidden_offset = row_offset
            .checked_mul(linear::HIDDEN)
            .ok_or_else(|| invalid("prefill hidden offset overflow"))?;
        let (input_owner, output_owner) = if layer.is_multiple_of(2) {
            (&self.hidden_a, &self.hidden_b)
        } else {
            (&self.hidden_b, &self.hidden_a)
        };
        let input = plan_f32_subview(input_owner, hidden_offset, hidden_elements)?;
        let output = plan_f32_subview(output_owner, hidden_offset, hidden_elements)?;
        let qkv_elements = tokens
            .checked_mul(linear::QKV_ELEMENTS)
            .ok_or_else(|| invalid("prefill qkv extent overflow"))?;
        let recurrent_elements = tokens
            .checked_mul(linear::RECURRENT_ELEMENTS)
            .ok_or_else(|| invalid("prefill recurrent extent overflow"))?;
        let gate_elements = tokens
            .checked_mul(linear::VALUE_HEADS)
            .ok_or_else(|| invalid("prefill gate extent overflow"))?;
        let intermediate_elements = tokens
            .checked_mul(linear::INTERMEDIATE)
            .ok_or_else(|| invalid("prefill intermediate extent overflow"))?;
        let normalized = prefill_f32_view(&self.normalized, hidden_elements)?;
        let normalized_q8_1 = prefill_q8_1_view(&self.normalized_q8_1, hidden_elements)?;
        let qkv = prefill_f32_view(&self.qkv, qkv_elements)?;
        let z = prefill_f32_view(&self.z, recurrent_elements)?;
        let alpha_raw = prefill_f32_view(&self.alpha_raw, gate_elements)?;
        let beta_raw = prefill_f32_view(&self.beta_raw, gate_elements)?;
        let recurrent = prefill_f32_view(&self.recurrent, recurrent_elements)?;
        let attention_output = prefill_f32_view(&self.attention_output, hidden_elements)?;
        let attention_residual = prefill_f32_view(&self.attention_residual, hidden_elements)?;
        let post_norm = prefill_f32_view(&self.post_norm, hidden_elements)?;
        let ffn_gate = prefill_f32_view(&self.ffn_gate, intermediate_elements)?;
        let ffn_up = prefill_f32_view(&self.ffn_up, intermediate_elements)?;
        let ffn_activation = prefill_f32_view(&self.ffn_activation, intermediate_elements)?;
        let ffn_input_q8_1 = prefill_q8_1_view(&self.ffn_input_q8_1, hidden_elements)?;
        let ffn_down_q8_1 = prefill_q8_1_view(&self.ffn_down_q8_1, intermediate_elements)?;
        let ffn_output = prefill_f32_view(&self.ffn_output, hidden_elements)?;

        let q8 = |buffer: &OwnedCudaBuffer, rows: usize, cols: usize| {
            Ok::<_, Qwen35MixedPrefixCudaError>(CudaQwen35PrefillWeightV1::Q8_0(
                CudaQ8_0WeightViewV1::new(
                    buffer.view(0, buffer.len_bytes())?,
                    PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_0_CUDA_V1,
                    rows,
                    cols,
                )?,
            ))
        };
        let bf16 = |buffer: &OwnedCudaBuffer, rows: usize, cols: usize| {
            Ok::<_, Qwen35MixedPrefixCudaError>(CudaQwen35PrefillWeightV1::Bf16(
                CudaBf16WeightViewV1::new(
                    buffer.view(0, buffer.len_bytes())?,
                    PhysicalLayoutId::DEVICE_CONTIGUOUS_BF16_CUDA_V1,
                    rows,
                    cols,
                )?,
            ))
        };
        let ffn = |buffer: &OwnedCudaBuffer, rows: usize, cols: usize, scale: f32| {
            prefill_ffn_weight(buffer, plan.ffn_weight_type, rows, cols, scale)
        };

        
        

        Ok(PrefillLinearOps {
            simt_shared_gdn_scalars: false,
            simt_shared_gdn_scalar_submissions: std::cell::Cell::new(0),
            canonical_gdn_scalars: None,
            canonical_gdn_prefill_batch: None,
            canonical_gdn_token_grid_submissions: std::cell::Cell::new(0),
            
            
            attn_norm: CudaQwen35PrefillRmsnormV1::bind(
                stream,
                input,
                plan_f32_view(&projection.attn_norm_weight, linear::HIDDEN)?,
                normalized,
                tokens,
                linear::HIDDEN,
                epsilon,
            )?,
            normalized_quantize: CudaQ8_1ActivationQuantizeArgsV1::bind(
                stream,
                normalized,
                normalized_q8_1,
            )?,
            qkv: self.cublas.bind_projection(
                stream,
                row_input::prefill_weight(&projection.qkv_weight, projection.input_weight_layout, linear::QKV_ELEMENTS, linear::HIDDEN)?,
                normalized_q8_1,
                qkv,
                tokens,
            )?,
            z: self.cublas.bind_projection(
                stream,
                row_input::prefill_weight(
                    &projection.z_weight, projection.input_weight_layout,
                    linear::RECURRENT_ELEMENTS,
                    linear::HIDDEN,
                )?,
                normalized_q8_1,
                z,
                tokens,
            )?,
            alpha: self.cublas.bind_projection(
                stream,
                bf16(
                    &projection.alpha_weight,
                    linear::VALUE_HEADS,
                    linear::HIDDEN,
                )?,
                normalized_q8_1,
                alpha_raw,
                tokens,
            )?,
            beta: self.cublas.bind_projection(
                stream,
                bf16(&projection.beta_weight, linear::VALUE_HEADS, linear::HIDDEN)?,
                normalized_q8_1,
                beta_raw,
                tokens,
            )?,
            recurrent: CudaQwen35PrefillLinearChunkV1::bind(
                stream,
                CudaQwen35PrefillLinearChunkViewsV1 {
                    qkv,
                    z,
                    conv_weights: plan_f32_view(&plan.conv, linear::CONV_WEIGHT_ELEMENTS)?,
                    conv_history: plan_f32_view(&plan.history, linear::CONV_HISTORY_ELEMENTS)?,
                    alpha_raw,
                    beta_raw,
                    dt_bias: plan_f32_view(&projection.dt_bias, linear::VALUE_HEADS)?,
                    ssm_a: plan_f32_view(&projection.ssm_a, linear::VALUE_HEADS)?,
                    state: plan_f32_view(&plan.state, linear::STATE_ELEMENTS)?,
                    norm_weight: plan_f32_view(&plan.ssm_norm, linear::HEAD_DIM)?,
                    activated: prefill_f32_view(&self.activated, qkv_elements)?,
                    recurrent,
                    output: prefill_f32_view(&self.attention_output, recurrent_elements)?,
                },
                tokens,
                epsilon,
            )?,
            recurrent_quantize: CudaQ8_1ActivationQuantizeArgsV1::bind(
                stream,
                prefill_f32_view(&self.attention_output, recurrent_elements)?,
                prefill_q8_1_view(&self.ffn_down_q8_1, recurrent_elements)?,
            )?,
            output: self.cublas.bind_projection(
                stream,
                q8(
                    &plan.output_weight,
                    linear::HIDDEN,
                    linear::RECURRENT_ELEMENTS,
                )?,
                prefill_q8_1_view(&self.ffn_down_q8_1, recurrent_elements)?,
                attention_output,
                tokens,
            )?,
            attention_residual: CudaAddF32ArgsV1::bind(
                stream,
                attention_output,
                input,
                attention_residual,
            )?,
            post_norm: CudaQwen35PrefillRmsnormV1::bind(
                stream,
                attention_residual,
                plan_f32_view(&plan.post_norm_weight, linear::HIDDEN)?,
                post_norm,
                tokens,
                linear::HIDDEN,
                epsilon,
            )?,
            ffn: if let Some(weights) = &plan.qpn_weights {
                self.bind_qpn_ffn(weights, tokens)?
            } else { PrefillFfn::Legacy(LegacyPrefillFfn {
            ffn_input_quantize: CudaQ8_1ActivationQuantizeArgsV1::bind(
                stream,
                post_norm,
                ffn_input_q8_1,
            )?,
            ffn_gate: self.cublas.bind_projection(
                stream,
                ffn(
                    &plan.ffn_gate_weight,
                    linear::INTERMEDIATE,
                    linear::HIDDEN,
                    plan.ffn_gate_scale,
                )?,
                ffn_input_q8_1,
                ffn_gate,
                tokens,
            )?,
            ffn_up: self.cublas.bind_projection(
                stream,
                ffn(
                    &plan.ffn_up_weight,
                    linear::INTERMEDIATE,
                    linear::HIDDEN,
                    plan.ffn_up_scale,
                )?,
                ffn_input_q8_1,
                ffn_up,
                tokens,
            )?,
            silu_mul: CudaSiluMulF32ArgsV1::bind(stream, ffn_gate, ffn_up, ffn_activation)?,
            preserved_silu_q8: None,
            ffn_down_quantize: CudaQ8_1ActivationQuantizeArgsV1::bind(
                stream,
                ffn_activation,
                ffn_down_q8_1,
            )?,
            ffn_down: self.cublas.bind_projection(
                stream,
                ffn(
                    &plan.ffn_down_weight,
                    linear::HIDDEN,
                    linear::INTERMEDIATE,
                    plan.ffn_down_scale,
                )?,
                ffn_down_q8_1,
                ffn_output,
                tokens,
            )?,
            }) },
            final_residual: CudaAddF32ArgsV1::bind(stream, attention_residual, ffn_output, output)?,
        })
    }

    fn upload_rope_tables(
        &self,
        stream: &OwnedCudaStream,
        tokens: usize,
        position_base: usize,
        theta: f32,
    ) -> Result<(), Qwen35MixedPrefixCudaError> {
        if position_base != 0
            || tokens == 0
            || position_base
                .checked_add(tokens)
                .is_none_or(|end| end > self.rope_capacity)
        {
            return Err(invalid("prefill RoPE upload must cover a prefix"));
        }
        match (&self.rope_cos, &self.rope_sin) {
            (PrefillRopeBuffer::Owned(_), PrefillRopeBuffer::Owned(_)) => {}
            (PrefillRopeBuffer::Shared { tables, cosine: true },
             PrefillRopeBuffer::Shared { tables: sine, cosine: false }) => {
                if !Arc::ptr_eq(tables, sine) {
                    return Err(invalid("shared workspace RoPE owner drift"));
                }
                // The canonical table already covers every requested position.
                // Never regenerate, upload, or overwrite immutable shared bytes.
                return tables.validate(stream, theta, self.rope_capacity, full::ROTARY_DIM);
            }
            _ => return Err(invalid("shared workspace RoPE requires paired leases")),
        }
        let elements = tokens
            .checked_mul(full::ROTARY_DIM / 2)
            .ok_or_else(|| invalid("prefill RoPE extent overflow"))?;
        let mut cosine = Vec::with_capacity(elements);
        let mut sine = Vec::with_capacity(elements);
        for token in 0..tokens {
            let position = position_base
                .checked_add(token)
                .ok_or_else(|| invalid("prefill RoPE position overflow"))?;
            let (token_cosine, token_sine) = native_rope_tables(theta, position, full::ROTARY_DIM)?;
            cosine.extend(token_cosine);
            sine.extend(token_sine);
        }
        self.rope_cos
            .copy_from_host_sync(stream, &f32_bytes(&cosine))?;
        self.rope_sin
            .copy_from_host_sync(stream, &f32_bytes(&sine))?;
        Ok(())
    }

    fn rope_reuse_report(&self) -> serde_json::Value {
        let shared = matches!(&self.rope_cos, PrefillRopeBuffer::Shared { .. });
        serde_json::json!({
            "shared_request_table":shared,
            "positions":self.rope_capacity,"rotary_dim":full::ROTARY_DIM,
            "avoided_gpu_allocations":if shared {2} else {0},
            "avoided_gpu_allocation_bytes":if shared {self.rope_cos.len_bytes()+self.rope_sin.len_bytes()} else {0},
            "avoided_table_h2d_bytes":if shared {self.rope_cos.len_bytes()+self.rope_sin.len_bytes()} else {0},
            "scope":"one_workspace_construction_not_per_transaction_savings",
            "immutable_owner_retained":shared,"arithmetic_changed":false
        })
    }

    fn build_full_ops(
        &self,
        stream: &OwnedCudaStream,
        plan: &FullPlan,
        layer: usize,
        tokens: usize,
        position_base: usize,
        epsilon: f32,
    ) -> Result<PrefillFullOps, Qwen35MixedPrefixCudaError> {
if tokens == 0 || tokens > self.max_tokens {
            return Err(invalid("prefill full-attention token extent"));
        }
        #[cfg(feature = "qwen35-kv8-chunk-prefill")]
        let capacity = {
            if !plan
                .key_cache
                .len_bytes()
                .is_multiple_of(QWEN35_KV8_CACHE_ROW_BYTES)
                || plan.key_cache.len_bytes() != plan.value_cache.len_bytes()
            {
                return Err(invalid("prefill KV8 chunk cache geometry"));
            }
            plan.key_cache.len_bytes() / QWEN35_KV8_CACHE_ROW_BYTES
        };
        #[cfg(not(feature = "qwen35-kv8-chunk-prefill"))]
        let (prefill_key_cache, prefill_value_cache, capacity) = {
            #[cfg(not(feature = "qwen35-kv8-full-attention"))]
            let prefill_key_cache = &plan.key_cache;
            #[cfg(not(feature = "qwen35-kv8-full-attention"))]
            let prefill_value_cache = &plan.value_cache;
            #[cfg(feature = "qwen35-kv8-full-attention")]
            let prefill_key_cache = &plan.prefill_key_cache_f32;
            #[cfg(feature = "qwen35-kv8-full-attention")]
            let prefill_value_cache = &plan.prefill_value_cache_f32;
            let cache_row_bytes = full::KV_VALUES
                .checked_mul(core::mem::size_of::<f32>())
                .ok_or_else(|| invalid("prefill full-attention cache row overflow"))?;
            if cache_row_bytes == 0
                || !prefill_key_cache
                    .len_bytes()
                    .is_multiple_of(cache_row_bytes)
                || prefill_key_cache.len_bytes() != prefill_value_cache.len_bytes()
            {
                return Err(invalid("prefill full-attention cache geometry"));
            }
            (
                prefill_key_cache,
                prefill_value_cache,
                prefill_key_cache.len_bytes() / cache_row_bytes,
            )
        };
        if position_base
            .checked_add(tokens)
            .is_none_or(|end| end > capacity)
        {
            return Err(invalid("prefill full-attention causal extent"));
        }

        let batch = |width: usize| {
            tokens
                .checked_mul(width)
                .ok_or_else(|| invalid("prefill full-attention extent overflow"))
        };
        let input = self.hidden_input(layer, tokens)?;
        let output = self.hidden_output(layer, tokens)?;
        let hidden_elements = batch(linear::HIDDEN)?;
        let query_elements = batch(full::QUERY_VALUES)?;
        let query_gate_elements = batch(full::QUERY_GATE_VALUES)?;
        let kv_elements = batch(full::KV_VALUES)?;
        let intermediate_elements = batch(linear::INTERMEDIATE)?;
        let rope_elements = batch(full::ROTARY_DIM / 2)?;
        let normalized = prefill_f32_view(&self.normalized, hidden_elements)?;
        let normalized_q8_1 = prefill_q8_1_view(&self.normalized_q8_1, hidden_elements)?;
        let q_full = prefill_f32_view(&self.qkv, query_gate_elements)?;
        let k_raw = prefill_f32_view(&self.z, kv_elements)?;
        let value = prefill_f32_view(&self.recurrent, kv_elements)?;
        let query = prefill_f32_view(&self.activated, query_elements)?;
        let gate = prefill_f32_view(&self.attention_output, query_elements)?;
        let attention = prefill_f32_view(&self.ffn_gate, query_elements)?;
        let gated = prefill_f32_view(&self.ffn_up, query_elements)?;
        let gated_q8_1 = prefill_q8_1_view(&self.ffn_down_q8_1, query_elements)?;
        let attention_output = prefill_f32_view(&self.attention_output, hidden_elements)?;
        let attention_residual = prefill_f32_view(&self.attention_residual, hidden_elements)?;
        let post_norm = prefill_f32_view(&self.post_norm, hidden_elements)?;
        let ffn_input_q8_1 = prefill_q8_1_view(&self.ffn_input_q8_1, hidden_elements)?;
        let ffn_gate = prefill_f32_view(&self.ffn_gate, intermediate_elements)?;
        let ffn_up = prefill_f32_view(&self.ffn_up, intermediate_elements)?;
        let ffn_activation = prefill_f32_view(&self.ffn_activation, intermediate_elements)?;
        let ffn_down_q8_1 = prefill_q8_1_view(&self.ffn_down_q8_1, intermediate_elements)?;
        let ffn_output = prefill_f32_view(&self.ffn_output, hidden_elements)?;

        let q8 = |buffer: &OwnedCudaBuffer, rows: usize, cols: usize| {
            Ok::<_, Qwen35MixedPrefixCudaError>(CudaQwen35PrefillWeightV1::Q8_0(
                CudaQ8_0WeightViewV1::new(
                    buffer.view(0, buffer.len_bytes())?,
                    PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_0_CUDA_V1,
                    rows,
                    cols,
                )?,
            ))
        };
        let ffn = |buffer: &OwnedCudaBuffer, rows: usize, cols: usize, scale: f32| {
            prefill_ffn_weight(buffer, plan.ffn_weight_type, rows, cols, scale)
        };

        

        Ok(PrefillFullOps {
            #[cfg(feature = "qwen35-kv8-chunk-prefill")]
            simt_shared_softmax: false,
            #[cfg(feature = "qwen35-kv8-chunk-prefill")]
            dgemm_attention: None,
            #[cfg(feature = "qwen35-kv8-chunk-prefill")]
            simt_softmax_tile16: false,
            #[cfg(feature = "qwen35-kv8-chunk-prefill")]
            simt_unit_previous_scale: false,
            
            attn_norm: CudaQwen35PrefillRmsnormV1::bind(
                stream,
                input,
                plan_f32_view(&plan.attn_norm_weight, linear::HIDDEN)?,
                normalized,
                tokens,
                linear::HIDDEN,
                epsilon,
            )?,
            normalized_quantize: CudaQ8_1ActivationQuantizeArgsV1::bind(
                stream,
                normalized,
                normalized_q8_1,
            )?,
            q: self.cublas.bind_projection(
                stream,
                row_input::prefill_weight(&plan.q_weight, plan.input_weight_layout, full::QUERY_GATE_VALUES, linear::HIDDEN)?,
                normalized_q8_1,
                q_full,
                tokens,
            )?,
            k: self.cublas.bind_projection(
                stream,
                row_input::prefill_weight(&plan.k_weight, plan.input_weight_layout, full::KV_VALUES, linear::HIDDEN)?,
                normalized_q8_1,
                k_raw,
                tokens,
            )?,
            v: self.cublas.bind_projection(
                stream,
                row_input::prefill_weight(&plan.v_weight, plan.input_weight_layout, full::KV_VALUES, linear::HIDDEN)?,
                normalized_q8_1,
                value,
                tokens,
            )?,
            #[cfg(not(feature = "qwen35-kv8-chunk-prefill"))]
            attention: CudaQwen35PrefillFullAttentionV1::bind(
                stream,
                CudaQwen35PrefillFullAttentionViewsV1 {
                    q_full,
                    k_raw,
                    value,
                    q_norm_weight: plan_f32_subview(&plan.q_norm_weight, 0, full::HEAD_DIM)?,
                    k_norm_weight: plan_f32_subview(&plan.k_norm_weight, 0, full::HEAD_DIM)?,
                    rope_cos: plan_f32_subview(
                        &self.rope_cos,
                        position_base
                            .checked_mul(full::ROTARY_DIM / 2)
                            .ok_or_else(|| invalid("prefill RoPE offset overflow"))?,
                        rope_elements,
                    )?,
                    rope_sin: plan_f32_subview(
                        &self.rope_sin,
                        position_base
                            .checked_mul(full::ROTARY_DIM / 2)
                            .ok_or_else(|| invalid("prefill RoPE offset overflow"))?,
                        rope_elements,
                    )?,
                    key_cache: plan_f32_view(prefill_key_cache, capacity * full::KV_VALUES)?,
                    value_cache: plan_f32_view(prefill_value_cache, capacity * full::KV_VALUES)?,
                    query,
                    gate,
                    attention,
                    gated,
                },
                tokens,
                position_base,
                capacity,
                epsilon,
            )?,
            #[cfg(feature = "qwen35-kv8-chunk-prefill")]
            attention: CudaQwen35PrefillKv8ChunkAttentionV1::bind(
                stream,
                CudaQwen35PrefillKv8ChunkAttentionViewsV1 {
                    q_full,
                    k_raw,
                    value,
                    q_norm_weight: plan_f32_subview(&plan.q_norm_weight, 0, full::HEAD_DIM)?,
                    k_norm_weight: plan_f32_subview(&plan.k_norm_weight, 0, full::HEAD_DIM)?,
                    rope_cos: plan_f32_subview(
                        &self.rope_cos,
                        position_base
                            .checked_mul(full::ROTARY_DIM / 2)
                            .ok_or_else(|| invalid("prefill RoPE offset overflow"))?,
                        rope_elements,
                    )?,
                    rope_sin: plan_f32_subview(
                        &self.rope_sin,
                        position_base
                            .checked_mul(full::ROTARY_DIM / 2)
                            .ok_or_else(|| invalid("prefill RoPE offset overflow"))?,
                        rope_elements,
                    )?,
                    key_work: prefill_f32_view(&self.kv8_key_work, kv_elements)?,
                    value_work: prefill_f32_view(&self.kv8_value_work, kv_elements)?,
                    key_cache: plan.key_cache.view(0, plan.key_cache.len_bytes())?,
                    value_cache: plan.value_cache.view(0, plan.value_cache.len_bytes())?,
                    query,
                    gate,
                    attention,
                    gated,
                },
                tokens,
                position_base,
                capacity,
                epsilon,
            )?,
            
            #[cfg(all(
                feature = "qwen35-kv8-full-attention",
                not(feature = "qwen35-kv8-chunk-prefill")
            ))]
            cache_quantize: CudaQwen35PrefillKv8CacheQuantizeV1::bind(
                stream,
                CudaQwen35PrefillKv8CacheQuantizeViewsV1 {
                    key_cache_f32: plan_f32_view(prefill_key_cache, capacity * full::KV_VALUES)?,
                    value_cache_f32: plan_f32_view(
                        prefill_value_cache,
                        capacity * full::KV_VALUES,
                    )?,
                    key_cache_q8_0: plan.key_cache.view(0, plan.key_cache.len_bytes())?,
                    value_cache_q8_0: plan.value_cache.view(0, plan.value_cache.len_bytes())?,
                },
                capacity,
                position_base,
                tokens,
            )?,
            gated_quantize: CudaQ8_1ActivationQuantizeArgsV1::bind(stream, gated, gated_q8_1)?,
            output: self.cublas.bind_projection(
                stream,
                q8(&plan.output_weight, linear::HIDDEN, full::QUERY_VALUES)?,
                gated_q8_1,
                attention_output,
                tokens,
            )?,
            attention_residual: CudaAddF32ArgsV1::bind(
                stream,
                attention_output,
                input,
                attention_residual,
            )?,
            post_norm: CudaQwen35PrefillRmsnormV1::bind(
                stream,
                attention_residual,
                plan_f32_view(&plan.post_norm_weight, linear::HIDDEN)?,
                post_norm,
                tokens,
                linear::HIDDEN,
                epsilon,
            )?,
            ffn: if let Some(weights) = &plan.qpn_weights {
                self.bind_qpn_ffn(weights, tokens)?
            } else { PrefillFfn::Legacy(LegacyPrefillFfn {
            ffn_input_quantize: CudaQ8_1ActivationQuantizeArgsV1::bind(
                stream,
                post_norm,
                ffn_input_q8_1,
            )?,
            ffn_gate: self.cublas.bind_projection(
                stream,
                ffn(
                    &plan.ffn_gate_weight,
                    linear::INTERMEDIATE,
                    linear::HIDDEN,
                    plan.ffn_gate_scale,
                )?,
                ffn_input_q8_1,
                ffn_gate,
                tokens,
            )?,
            ffn_up: self.cublas.bind_projection(
                stream,
                ffn(
                    &plan.ffn_up_weight,
                    linear::INTERMEDIATE,
                    linear::HIDDEN,
                    plan.ffn_up_scale,
                )?,
                ffn_input_q8_1,
                ffn_up,
                tokens,
            )?,
            silu_mul: CudaSiluMulF32ArgsV1::bind(stream, ffn_gate, ffn_up, ffn_activation)?,
            preserved_silu_q8: None,
            ffn_down_quantize: CudaQ8_1ActivationQuantizeArgsV1::bind(
                stream,
                ffn_activation,
                ffn_down_q8_1,
            )?,
            ffn_down: self.cublas.bind_projection(
                stream,
                ffn(
                    &plan.ffn_down_weight,
                    linear::HIDDEN,
                    linear::INTERMEDIATE,
                    plan.ffn_down_scale,
                )?,
                ffn_down_q8_1,
                ffn_output,
                tokens,
            )?,
            }) },
            final_residual: CudaAddF32ArgsV1::bind(stream, attention_residual, ffn_output, output)?,
            
        })
    }
}

#[cfg(feature = "qwen35-prefill-cublas")]
impl PrefillLinearOps {
    fn submit_gdn_scalars(
        &self,
        stream: &OwnedCudaStream,
        cublas: &CudaQwen35PrefillCublasPlanV1,
    ) -> Result<(), Qwen35MixedPrefixCudaError> {
        if self.alpha.llama_full_original_enabled() || self.beta.llama_full_original_enabled() {
            if !(self.alpha.llama_full_original_enabled() && self.beta.llama_full_original_enabled()) {
                return Err(invalid("partial llama BF16 scalar binding"));
            }
            // The opt-in plan may batch the two large-row BF16 projections.
            // Disabled, small-row and non-original paths keep exact old dispatch.
            cublas.submit_shared_input_group(stream, &[&self.alpha, &self.beta])?;
        } else if let Some([alpha, beta]) = &self.canonical_gdn_prefill_batch {
            alpha.submit_for_plan(stream)?;
            beta.submit_for_plan(stream)?;
            self.canonical_gdn_token_grid_submissions.set(
                self.canonical_gdn_token_grid_submissions.get().checked_add(2)
                    .ok_or_else(|| invalid("GDN token-grid submission counter overflow"))?,
            );
        } else if let Some(rows) = &self.canonical_gdn_scalars {
            for [alpha, beta] in rows {
                alpha.submit_for_plan(stream)?;
                beta.submit_for_plan(stream)?;
            }
        } else {
            cublas.submit(stream, &self.alpha)?;
            cublas.submit(stream, &self.beta)?;
        }
        Ok(())
    }

    fn submit_recurrence(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        // Reuse the already-bound output scratch only on the measured original-
        // input prefix path. Native stream ordering finishes the scalar consumer
        // before gated norm overwrites that output. No new owner/allocation.
        // B8 is separately explicit; replay has its own state-only implementation.
        if use_shared_gdn_prefix_scalars(
            self.simt_shared_gdn_scalars,
            self.qkv.llama_full_original_enabled(),
            self.qkv.raw().tokens,
        ) {
            self.recurrent.submit_shared_scalars(stream)?;
            self.simt_shared_gdn_scalar_submissions.set(self.simt_shared_gdn_scalar_submissions.get() + 1);
            Ok(())
        } else {
            self.recurrent.submit(stream)
        }
    }

    fn submit(
        &self,
        stream: &OwnedCudaStream,
        cublas: &CudaQwen35PrefillCublasPlanV1,
    ) -> Result<(), Qwen35MixedPrefixCudaError> {
        self.submit_with_activation_policy(stream, cublas, false)
    }

    fn submit_with_activation_policy(
        &self,
        stream: &OwnedCudaStream,
        cublas: &CudaQwen35PrefillCublasPlanV1,
        direct_post_attention: bool,
    ) -> Result<(), Qwen35MixedPrefixCudaError> {
        self.submit_with_activation_policy_and_events(stream, cublas, direct_post_attention, None)
    }

    fn submit_with_activation_policy_and_events(
        &self,
        stream: &OwnedCudaStream,
        cublas: &CudaQwen35PrefillCublasPlanV1,
        direct_post_attention: bool,
        phase_events: Option<&[OwnedCudaEvent; 5]>,
    ) -> Result<(), Qwen35MixedPrefixCudaError> {
        if direct_post_attention && phase_events.is_some() {
            return Err(invalid("canonical phase profile rejects direct activation"));
        }
        
        if direct_post_attention {
            return Err(invalid(
                "direct activation requires Tensor Core prefix feature",
            ));
        }
        self.attn_norm.submit(stream)?;
        
        
        {
            // Both projections and both scalar branches must use original F32
            // before this Q8 materialization is dead. Native B8 still packs.
            cublas.submit_required_q8_input(stream, &self.normalized_quantize,
                &[&self.qkv, &self.z, &self.alpha, &self.beta])?;
            cublas.submit_shared_input_group(stream, &[&self.qkv, &self.z])?;
            self.submit_gdn_scalars(stream, cublas)?;
        }
        record_canonical_layer_phase(stream, phase_events, 0)?;
        self.submit_recurrence(stream)?;
        record_canonical_layer_phase(stream, phase_events, 1)?;
        
        cublas.submit_required_q8_input(stream, &self.recurrent_quantize, &[&self.output])?;
        cublas.submit(stream, &self.output)?;
        self.attention_residual.submit_for_plan(stream)?;
        self.post_norm.submit(stream)?;
        record_canonical_layer_phase(stream, phase_events, 2)?;
        self.ffn.submit_gate_up(stream, cublas, true)?;
        record_canonical_layer_phase(stream, phase_events, 3)?;
        
        
        self.ffn.submit_activation(stream, cublas)?;
        record_canonical_layer_phase(stream, phase_events, 4)?;
        self.ffn.submit_down(stream, cublas)?;
        self.final_residual.submit_for_plan(stream)?;
        Ok(())
    }
}

#[cfg(feature = "qwen35-prefill-cublas")]
const fn use_shared_gdn_prefix_scalars(explicit: bool, llama_original: bool, tokens: u32) -> bool {
    explicit || (llama_original && tokens >= 64)
}

#[cfg(all(test, feature = "qwen35-prefill-cublas"))]
#[test]
fn shared_gdn_prefix_dispatch_preserves_small_and_legacy_paths() {
    for tokens in [1, 7, 8, 31, 63] {
        assert!(!use_shared_gdn_prefix_scalars(false, true, tokens));
    }
    for tokens in [64, 128, 511, 512] {
        assert!(use_shared_gdn_prefix_scalars(false, true, tokens));
        assert!(!use_shared_gdn_prefix_scalars(false, false, tokens));
    }
    assert!(use_shared_gdn_prefix_scalars(true, false, 8));
}

#[cfg(feature = "qwen35-prefill-cublas")]
fn record_canonical_layer_phase(
    stream: &OwnedCudaStream,
    events: Option<&[OwnedCudaEvent; 5]>,
    index: usize,
) -> Result<(), Qwen35MixedPrefixCudaError> {
    if let Some(events) = events {
        let event = events
            .get(index)
            .ok_or_else(|| invalid("canonical phase event extent"))?;
        event.record_for_submission(stream, stream.submission_receipt()?)?;
    }
    Ok(())
}

#[cfg(feature = "qwen35-prefill-cublas")]
impl PrefillFullOps {
    #[cfg(feature = "qwen35-kv8-chunk-prefill")]
    fn submit_shared_simt_attention(
        &self,
        stream: &OwnedCudaStream,
        cublas: &CudaQwen35PrefillCublasPlanV1,
    ) -> Result<(), CudaRuntimeError> {
        if let Some(attention) = &self.dgemm_attention {
            return attention.submit_tiled_dgemm(stream, cublas);
        }
        
        if self.simt_unit_previous_scale {
            self.attention.submit_shared_softmax_unit_scale(stream)
        } else if self.simt_softmax_tile16 {
            self.attention.submit_shared_softmax_tile16(stream)
        } else {
            self.attention.submit_shared_softmax(stream)
        }
    }

    fn submit_projection_prefix(
        &self,
        stream: &OwnedCudaStream,
        cublas: &CudaQwen35PrefillCublasPlanV1,
    ) -> Result<(), Qwen35MixedPrefixCudaError> {
        self.attn_norm.submit(stream)?;
        cublas.submit_required_q8_input(stream, &self.normalized_quantize,
            &[&self.q, &self.k, &self.v])?;
        cublas.submit_shared_input_group(stream, &[&self.q, &self.k, &self.v])?;
        Ok(())
    }

    

    fn submit_post_attention(
        &self,
        stream: &OwnedCudaStream,
        cublas: &CudaQwen35PrefillCublasPlanV1,
    ) -> Result<(), Qwen35MixedPrefixCudaError> {
        self.submit_post_attention_with_activation_policy(stream, cublas, false)
    }

    fn submit_post_attention_with_activation_policy(
        &self,
        stream: &OwnedCudaStream,
        cublas: &CudaQwen35PrefillCublasPlanV1,
        direct_post_attention: bool,
    ) -> Result<(), Qwen35MixedPrefixCudaError> {
        if direct_post_attention {
            
            
            return Err(invalid(
                "direct activation requires Tensor Core prefix feature",
            ));
        }
        #[cfg(all(
            feature = "qwen35-kv8-full-attention",
            not(feature = "qwen35-kv8-chunk-prefill")
        ))]
        self.cache_quantize.submit(stream)?;
        cublas.submit_required_q8_input(stream, &self.gated_quantize, &[&self.output])?;
        cublas.submit(stream, &self.output)?;
        self.attention_residual.submit_for_plan(stream)?;
        self.post_norm.submit(stream)?;
        self.ffn.submit_gate_up(stream, cublas, true)?;
        
        
        self.ffn.submit_activation(stream, cublas)?;
        self.ffn.submit_down(stream, cublas)?;
        self.final_residual.submit_for_plan(stream)?;
        Ok(())
    }

    fn submit(
        &self,
        stream: &OwnedCudaStream,
        cublas: &CudaQwen35PrefillCublasPlanV1,
    ) -> Result<(), Qwen35MixedPrefixCudaError> {
        self.submit_projection_prefix(stream, cublas)?;
        #[cfg(feature = "qwen35-kv8-chunk-prefill")]
        if self.simt_shared_softmax {
            self.submit_shared_simt_attention(stream, cublas)?;
            return self.submit_post_attention(stream, cublas);
        }
        self.attention.submit(stream)?;
        self.submit_post_attention(stream, cublas)
    }

    

    

    

    
}

#[cfg(feature = "qwen35-prefill-cublas")]
impl PrefillLayerOps {
    fn submit(
        &self,
        stream: &OwnedCudaStream,
        cublas: &CudaQwen35PrefillCublasPlanV1,
    ) -> Result<(), Qwen35MixedPrefixCudaError> {
        match self {
            Self::Linear(operations) => operations.submit(stream, cublas),
            Self::Full(operations) => operations.submit(stream, cublas),
            
        }
    }

    fn submit_with_attention_path(
        &self,
        stream: &OwnedCudaStream,
        cublas: &CudaQwen35PrefillCublasPlanV1,
        tensor_core_prefix_attention: bool,
        direct_post_attention: bool,
    ) -> Result<(), Qwen35MixedPrefixCudaError> {
        if !tensor_core_prefix_attention && !direct_post_attention {
            return self.submit(stream, cublas);
        }
        match self {
            Self::Linear(operations) => {
                operations.submit_with_activation_policy(stream, cublas, direct_post_attention)
            }
            
            Self::Full(operations) => {
                
                
                {
                    let _ = operations;
                    Err(invalid("Tensor Core prefix attention feature"))
                }
            }
        }
    }
}

#[cfg(feature = "qwen35-prefill-cublas")]
fn prefill_ffn_weight(
    buffer: &OwnedCudaBuffer,
    weight_type: u32,
    rows: usize,
    cols: usize,
    scale: f32,
) -> Result<CudaQwen35PrefillWeightV1, Qwen35MixedPrefixCudaError> {
    match weight_type {
        GGML_TYPE_Q8_0 => Ok(CudaQwen35PrefillWeightV1::Q8_0(CudaQ8_0WeightViewV1::new(
            buffer.view(0, buffer.len_bytes())?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_0_CUDA_V1,
            rows,
            cols,
        )?)),
        GGML_TYPE_NVFP4 => Ok(CudaQwen35PrefillWeightV1::Nvfp4 {
            view: CudaNvfp4Sm70WeightViewV1::new(
                buffer.view(0, buffer.len_bytes())?,
                PhysicalLayoutId::DEVICE_CONTIGUOUS_NVFP4_CUDA_SM70_V1,
                rows,
                cols,
            )?,
            tensor_scale: scale,
        }),
        _ => Err(invalid("prefill FFN weight type")),
    }
}

#[cfg(feature = "qwen35-prefill-cublas")]
fn prefill_f32_view(
    buffer: &OwnedCudaBuffer,
    elements: usize,
) -> Result<CudaF32View, Qwen35MixedPrefixCudaError> {
    let bytes = elements
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| invalid("prefill F32 view extent overflow"))?;
    if bytes > buffer.len_bytes() {
        return Err(invalid("prefill F32 view exceeds workspace"));
    }
    Ok(CudaF32View::new(
        buffer.view(0, bytes)?,
        PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
        elements,
    )?)
}

#[cfg(feature = "qwen35-prefill-cublas")]
fn prefill_q8_1_view(
    buffer: &OwnedCudaBuffer,
    elements: usize,
) -> Result<CudaQ8_1ActivationViewV1, Qwen35MixedPrefixCudaError> {
    if elements == 0 || !elements.is_multiple_of(Q8_1_ACTIVATION_BLOCK_ELEMENTS) {
        return Err(invalid("prefill Q8_1 view geometry"));
    }
    let bytes = elements
        .checked_div(Q8_1_ACTIVATION_BLOCK_ELEMENTS)
        .and_then(|blocks| blocks.checked_mul(Q8_1_ACTIVATION_BLOCK_BYTES))
        .ok_or_else(|| invalid("prefill Q8_1 view extent overflow"))?;
    if bytes > buffer.len_bytes() {
        return Err(invalid("prefill Q8_1 view exceeds workspace"));
    }
    Ok(CudaQ8_1ActivationViewV1::new(
        buffer.view(0, bytes)?,
        PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_1_ACTIVATION_CUDA_V1,
        elements,
    )?)
}

fn alloc_f32(device: i32, elements: usize) -> Result<OwnedCudaBuffer, Qwen35MixedPrefixCudaError> {
    let bytes = elements
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| invalid("F32 allocation extent overflow"))?;
    Ok(OwnedCudaBuffer::allocate(device, bytes)?)
}

fn alloc_q8_1(device: i32, elements: usize) -> Result<OwnedCudaBuffer, Qwen35MixedPrefixCudaError> {
    if elements == 0 || !elements.is_multiple_of(Q8_1_ACTIVATION_BLOCK_ELEMENTS) {
        return Err(invalid("Q8_1 activation allocation geometry"));
    }
    let bytes = elements
        .checked_div(Q8_1_ACTIVATION_BLOCK_ELEMENTS)
        .and_then(|blocks| blocks.checked_mul(Q8_1_ACTIVATION_BLOCK_BYTES))
        .ok_or_else(|| invalid("Q8_1 activation allocation extent overflow"))?;
    Ok(OwnedCudaBuffer::allocate(device, bytes)?)
}

fn build_device_output_head_plan(
    stream: &OwnedCudaStream,
    material: &OutputHeadMaterial,
    source_hidden: CudaBufferView,
    hidden: usize,
    vocabulary: usize,
    epsilon: f32,
) -> Result<DeviceOutputHeadPlan, Qwen35MixedPrefixCudaError> {
    let weights = HeadWeights::upload(stream, material, hidden, vocabulary)?;
    bind_device_output_head_plan(stream, &weights, source_hidden, epsilon)
}

fn bind_device_output_head_plan(
    stream: &OwnedCudaStream,
    weights: &HeadWeights,
    source_hidden: CudaBufferView,
    epsilon: f32,
) -> Result<DeviceOutputHeadPlan, Qwen35MixedPrefixCudaError> {
    let hidden = weights.hidden;
    let vocabulary = weights.vocabulary;
    let source_bytes = hidden
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| invalid("output-head source extent overflow"))?;
    if source_hidden.len_bytes() != source_bytes {
        return Err(invalid("output-head material geometry"));
    }

    let device = stream.device_ordinal();
    let norm_weight = weights.norm_weight.clone();
    let normalized_hidden = alloc_f32(device, hidden)?;
    let activation_q8_1 = alloc_q8_1(device, hidden)?;
    let output_weight = weights.output_weight.clone();
    let logits = alloc_f32(device, vocabulary)?;

    let source = CudaF32View::new(
        source_hidden,
        PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
        hidden,
    )?;
    let normalized = plan_f32_view(&normalized_hidden, hidden)?;
    let activation = linear::q8_1_activation_view(&activation_q8_1, hidden)?;
    let norm_op = CudaPlanRmsnormF32::bind(
        stream,
        source,
        plan_f32_view(&norm_weight, hidden)?,
        normalized,
        epsilon,
    )?;
    let quantize_op = CudaQ8_1ActivationQuantizeArgsV1::bind(stream, normalized, activation)?;
    let projection_op = if weights.output_layout == PhysicalLayoutId::DEVICE_QWEN35_Q8_CODE_SCALE_PLANES_V1 {
        #[cfg(feature = "qwen35-prefill-cublas")]
        { HeadScalarProjection::Planar(augur_backend::CudaQwen35PlanarQ8HeadProjectionV1::bind(
            stream, augur_backend::CudaQwen35PlanarQ8HeadWeightV1::new(
                output_weight.view(0, output_weight.len_bytes())?, weights.output_layout, vocabulary, hidden)?,
            activation, plan_f32_view(&logits, vocabulary)?, 1)?) }
        #[cfg(not(feature = "qwen35-prefill-cublas"))]
        { return Err(invalid("planar head backend missing")); }
    } else { HeadScalarProjection::Block(CudaQ8_0Q8_1MatvecArgsV1::bind(
        stream,
        CudaQ8_0WeightViewV1::new(
            output_weight.view(0, output_weight.len_bytes())?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_0_CUDA_V1,
            vocabulary,
            hidden,
        )?,
        activation,
        plan_f32_view(&logits, vocabulary)?,
    )?) };

    Ok(DeviceOutputHeadPlan {
        norm_weight,
        normalized_hidden,
        activation_q8_1,
        output_weight,
        logits,
        norm_op,
        quantize_op,
        projection_op,
        output_weight_payload_bytes: weights.output_weight.len_bytes(),
        output_weight_payload_sha256: weights.payload_sha256.clone(),
        output_weight_layout: weights.output_layout,
        output_weight_physical_sha256: weights.physical_sha256.clone(),
        output_weight_relayout_ms: weights.relayout_ms,
    })
}

#[cfg(feature = "qwen35-device-token-handoff")]
#[allow(clippy::too_many_arguments)]
fn build_device_token_handoff_plan(
    stream: &OwnedCudaStream,
    reader: &mut GgufTensorReader<'_>,
    embedding_descriptor: &GgufTensor,
    output_head: &DeviceOutputHeadPlan,
    hidden: usize,
    vocabulary: usize,
    max_tokens: usize,
    source_commit: &str,
    binary_sha256: &str,
    config_sha256: &str,
    gguf_file_sha256: Option<&str>,
    prompt_text_sha256: &str,
    capacity: usize,
) -> Result<DeviceTokenHandoffPlan, Qwen35MixedPrefixCudaError> {
    let expected_embedding_bytes = hidden
        .checked_mul(vocabulary)
        .and_then(|elements| elements.checked_mul(2))
        .ok_or_else(|| invalid("device token-handoff embedding extent overflow"))?;
    let expected_receipt_bytes = max_tokens
        .checked_mul(QWEN35_TOKEN_HANDOFF_RECEIPT_BYTES)
        .ok_or_else(|| invalid("device token-handoff receipt extent overflow"))?;
    if hidden != linear::HIDDEN
        || vocabulary != QWEN35_TOKEN_HANDOFF_VOCAB
        || max_tokens == 0
        || embedding_descriptor.name != "token_embd.weight"
        || embedding_descriptor.ggml_type != GGML_TYPE_BF16
        || embedding_descriptor.dimensions != [hidden as u64, vocabulary as u64]
        || embedding_descriptor.byte_len != expected_embedding_bytes as u64
        || output_head.logits.len_bytes() != QWEN35_TOKEN_HANDOFF_LOGITS_BYTES
    {
        return Err(invalid("device token-handoff material geometry"));
    }

    let embedding_payload = reader.read_tensor_bytes(&embedding_descriptor.name)?;
    if embedding_payload.len() != expected_embedding_bytes {
        return Err(invalid("device token-handoff embedding payload extent"));
    }
    let embedding_payload_sha256 = sha256_hex(&embedding_payload);
    let mut digest = Sha256::new();
    digest.update(b"crystal-quant.qwen35.device-token-handoff-plan.v1\0");
    for value in [
        source_commit,
        binary_sha256,
        config_sha256,
        gguf_file_sha256.unwrap_or("UNBOUND_GGUF_SHA256"),
        prompt_text_sha256,
        &embedding_payload_sha256,
        &output_head.output_weight_payload_sha256,
    ] {
        digest.update((value.len() as u64).to_le_bytes());
        digest.update(value.as_bytes());
    }
    digest.update((hidden as u64).to_le_bytes());
    digest.update((vocabulary as u64).to_le_bytes());
    digest.update((max_tokens as u64).to_le_bytes());
    digest.update((capacity as u64).to_le_bytes());
    let plan_digest: [u8; 32] = digest.finalize().into();
    if plan_digest == [0; 32] {
        return Err(invalid("device token-handoff plan digest"));
    }

    let device = stream.device_ordinal();
    let embedding = OwnedCudaBuffer::allocate(device, expected_embedding_bytes)?;
    let partials = OwnedCudaBuffer::allocate(device, QWEN35_TOKEN_HANDOFF_PARTIAL_BYTES)?;
    let receipts = OwnedCudaBuffer::allocate(device, expected_receipt_bytes)?;
    embedding.copy_from_host_sync(stream, &embedding_payload)?;

    Ok(DeviceTokenHandoffPlan {
        embedding,
        partials,
        receipts,
        embedding_payload_bytes: expected_embedding_bytes,
        embedding_payload_sha256,
        plan_digest,
        max_tokens,
    })
}

#[cfg(feature = "qwen35-device-token-handoff")]
impl DeviceTokenHandoffPlan {
    fn step_generation(
        &self,
        run_epoch: u32,
        generation_index: usize,
    ) -> Result<u64, Qwen35MixedPrefixCudaError> {
        if run_epoch == 0 || generation_index >= self.max_tokens {
            return Err(invalid("device token-handoff generation identity"));
        }
        let index = u32::try_from(generation_index)
            .map_err(|_| invalid("device token-handoff generation index"))?;
        Ok((u64::from(run_epoch) << 32) | u64::from(index + 1))
    }

    fn receipt_view(
        &self,
        generation_index: usize,
    ) -> Result<CudaBufferView, Qwen35MixedPrefixCudaError> {
        if generation_index >= self.max_tokens {
            return Err(invalid("device token-handoff receipt index"));
        }
        let offset = generation_index
            .checked_mul(QWEN35_TOKEN_HANDOFF_RECEIPT_BYTES)
            .ok_or_else(|| invalid("device token-handoff receipt offset overflow"))?;
        Ok(self
            .receipts
            .view(offset, QWEN35_TOKEN_HANDOFF_RECEIPT_BYTES)?)
    }

    fn bind_top2(
        &self,
        stream: &OwnedCudaStream,
        logits: &OwnedCudaBuffer,
        run_epoch: u32,
        generation_index: usize,
    ) -> Result<CudaQwen35TokenHandoffArgsV1, Qwen35MixedPrefixCudaError> {
        Ok(CudaQwen35TokenHandoffArgsV1::bind(
            stream,
            CudaQwen35TokenHandoffViewsV1 {
                logits: logits.view(0, logits.len_bytes())?,
                partials: self.partials.view(0, self.partials.len_bytes())?,
                receipt: self.receipt_view(generation_index)?,
            },
            self.step_generation(run_epoch, generation_index)?,
            self.plan_digest,
        )?)
    }

    fn bind_embedding_gather(
        &self,
        stream: &OwnedCudaStream,
        generation_index: usize,
        output: CudaF32View,
    ) -> Result<CudaBf16TokenRowGatherArgsV1, Qwen35MixedPrefixCudaError> {
        let embedding = CudaBf16WeightViewV1::new(
            self.embedding.view(0, self.embedding.len_bytes())?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_BF16_CUDA_V1,
            QWEN35_TOKEN_HANDOFF_VOCAB,
            linear::HIDDEN,
        )?;
        Ok(CudaBf16TokenRowGatherArgsV1::bind(
            stream,
            embedding,
            self.receipt_view(generation_index)?,
            output,
        )?)
    }
}

#[allow(clippy::too_many_arguments)]
fn bind_scalar_ffn(
    stream: &OwnedCudaStream, material: &target_weights::FfnWeights,
    post_norm: &Weight, output: &Weight,
    gate_output: &OwnedCudaBuffer, up_output: &OwnedCudaBuffer,
    activation: &OwnedCudaBuffer, input_q8: &OwnedCudaBuffer, down_q8: &OwnedCudaBuffer,
    #[cfg(feature = "qwen35-prefill-cublas")]
    qpn_session: Option<&target_weights::QpnScalarSession>,
) -> Result<linear::BoundFfn, Qwen35MixedPrefixCudaError> {
    #[cfg(feature = "qwen35-prefill-cublas")]
    if let Some(weights) = &material.qpn {
        return qpn_session.ok_or_else(|| invalid("QPN weights require an explicit same-stream scalar workspace"))?
            .bind(stream, weights, Rc::clone(post_norm), Rc::clone(output));
    }
    Ok(linear::bind_legacy_ffn(stream, post_norm, &material.gate, &material.up,
        gate_output, up_output, activation, &material.down, output,
        Some(input_q8), Some(down_q8), material.gate.len_bytes(), material.up.len_bytes(),
        material.down.len_bytes(), material.weight_type, material.gate_scale,
        material.up_scale, material.down_scale)?)
}

fn build_linear_plan(
    stream: &OwnedCudaStream,
    material: &LinearMaterial,
    first_hidden: &[f32],
    first_projection: &linear::ProjectedToken,
    device_projection: bool,
    input_source: Option<CudaF32View>,
) -> Result<LinearPlan, Qwen35MixedPrefixCudaError> {
    let weights = LinearWeights::upload(stream, material, device_projection)?;
    bind_linear_plan(
        stream,
        &weights,
        first_hidden,
        Some(first_projection),
        device_projection,
        input_source,
        #[cfg(feature = "qwen35-prefill-cublas")]
        None,
    )
}

fn bind_linear_plan(
    stream: &OwnedCudaStream,
    material: &LinearWeights,
    first_hidden: &[f32],
    first_projection: Option<&linear::ProjectedToken>,
    device_projection: bool,
    input_source: Option<CudaF32View>,
    #[cfg(feature = "qwen35-prefill-cublas")]
    qpn_session: Option<&target_weights::QpnScalarSession>,
) -> Result<LinearPlan, Qwen35MixedPrefixCudaError> {
    let device = stream.device_ordinal();
    let hidden = material.spec.hidden_size;
    let intermediate = material.spec.intermediate_size;
    let qkv_elements = material.spec.linear_key_heads * material.spec.linear_head_dim * 2
        + material.spec.linear_value_heads * material.spec.linear_head_dim;
    let recurrent_elements = material.spec.linear_value_heads * material.spec.linear_head_dim;
    let state_elements = material.spec.linear_value_heads
        * material.spec.linear_head_dim
        * material.spec.linear_head_dim;
    let conv_channels = qkv_elements;
    let history_elements = conv_channels * (material.spec.linear_conv_kernel - 1);

    let qkv = alloc_f32(device, qkv_elements)?;
    let z = alloc_f32(device, recurrent_elements)?;
    let conv = material.conv.clone();
    let mut history = alloc_f32(device, history_elements)?;
    let decay = alloc_f32(device, material.spec.linear_value_heads)?;
    let beta = alloc_f32(device, material.spec.linear_value_heads)?;
    let mut state = alloc_f32(device, state_elements)?;
    let ssm_norm = material.ssm_norm.clone();
    let activated = alloc_f32(device, qkv_elements)?;
    let recurrent = alloc_f32(device, recurrent_elements)?;
    let linear_output = alloc_f32(device, recurrent_elements)?;
    let attention = alloc_f32(device, hidden)?;
    let input = alloc_f32(device, hidden)?;
    let (input_view, input_is_alias) = match input_source {
        Some(view) => {
            if view.element_count() != hidden || view.buffer_view().device_ordinal() != device {
                return Err(invalid("linear input alias geometry"));
            }
            (view, true)
        }
        None => (plan_f32_view(&input, hidden)?, false),
    };
    let attention_residual = alloc_f32(device, hidden)?;
    let post_norm = Rc::new(alloc_f32(device, hidden)?);
    let ffn_gate_weight = material.ffn.gate.clone();
    let ffn_up_weight = material.ffn.up.clone();
    let ffn_gate_output = alloc_f32(device, intermediate)?;
    let ffn_up_output = alloc_f32(device, intermediate)?;
    let ffn_activation = alloc_f32(device, intermediate)?;
    let ffn_input_q8_1 = alloc_q8_1(device, hidden)?;
    let ffn_down_q8_1 = alloc_q8_1(device, intermediate)?;
    let ffn_down_weight = material.ffn.down.clone();
    let ffn_output = Rc::new(alloc_f32(device, hidden)?);
    let final_hidden = alloc_f32(device, hidden)?;
    let output_weight = material.output_weight.clone();
    let output_q8_1 = alloc_q8_1(device, recurrent_elements)?;
    let post_norm_weight = material.post_norm_weight.clone();

    if let Some(token) = first_projection {
        qkv.copy_from_host_sync(stream, &f32_bytes(&token.qkv))?;
        z.copy_from_host_sync(stream, &f32_bytes(&token.z))?;
        decay.copy_from_host_sync(stream, &f32_bytes(&token.decay_gate))?;
        beta.copy_from_host_sync(stream, &f32_bytes(&token.beta))?;
    } else if !device_projection {
        return Err(invalid("uninitialized host-projected linear request"));
    }
    if !input_is_alias {
        input.copy_from_host_sync(stream, &f32_bytes(first_hidden))?;
    }
    history.zero_sync(stream)?;
    state.zero_sync(stream)?;

    let ffn = bind_scalar_ffn(stream, &material.ffn, &post_norm, &ffn_output,
        &ffn_gate_output, &ffn_up_output, &ffn_activation, &ffn_input_q8_1, &ffn_down_q8_1,
        #[cfg(feature = "qwen35-prefill-cublas")]
        qpn_session)?;
    let chain = linear::bind_chain_with_ffn(
        stream,
        &qkv,
        &z,
        &conv,
        &history,
        &decay,
        &beta,
        &state,
        &ssm_norm,
        &activated,
        &recurrent,
        &linear_output,
        &output_weight,
        Some(&output_q8_1),
        &attention,
        input_view,
        &attention_residual,
        &post_norm,
        &post_norm_weight,
        &ffn_output,
        &final_hidden,
        material.output_weight.len_bytes(),
        material.spec.rms_norm_eps,
        ffn,
    )?;
    let projection = if device_projection {
        Some(build_linear_projection(
            stream,
            &material.spec,
            material
                .projection
                .as_ref()
                .ok_or_else(|| invalid("missing resident projection"))?,
            input_view,
            &qkv,
            &z,
            &decay,
            &beta,
        )?)
    } else {
        None
    };
    Ok(LinearPlan {
        qkv,
        z,
        conv,
        history,
        decay,
        beta,
        state,
        ssm_norm,
        activated,
        recurrent,
        linear_output,
        attention,
        input,
        input_view,
        input_is_alias,
        attention_residual,
        post_norm,
        ffn_gate_weight,
        ffn_up_weight,
        ffn_gate_output,
        ffn_up_output,
        ffn_activation,
        ffn_input_q8_1,
        ffn_down_q8_1,
        ffn_down_weight,
        ffn_output,
        final_hidden,
        output_weight,
        output_q8_1,
        post_norm_weight,
        ffn_weight_type: material.ffn.weight_type,
        ffn_gate_scale: material.ffn.gate_scale,
        ffn_up_scale: material.ffn.up_scale,
        ffn_down_scale: material.ffn.down_scale,
        ffn_is_qpn: material.ffn.qpn_enabled(),
        #[cfg(feature = "qwen35-prefill-cublas")]
        qpn_weights: material.ffn.qpn.clone(),
        projection,
        chain,
    })
}

fn build_linear_projection(
    stream: &OwnedCudaStream,
    spec: &Qwen35LayerSpec,
    weights: &ProjectionWeights,
    input: CudaF32View,
    qkv: &OwnedCudaBuffer,
    z: &OwnedCudaBuffer,
    decay: &OwnedCudaBuffer,
    beta: &OwnedCudaBuffer,
) -> Result<LinearProjectionPlan, Qwen35MixedPrefixCudaError> {
    let device = stream.device_ordinal();
    let hidden = spec.hidden_size;
    let qkv_elements = spec.linear_key_heads * spec.linear_head_dim * 2
        + spec.linear_value_heads * spec.linear_head_dim;
    let recurrent_elements = spec.linear_value_heads * spec.linear_head_dim;
    let value_heads = spec.linear_value_heads;
    let attn_norm_weight = weights.attn_norm_weight.clone();
    let qkv_weight = weights.qkv_weight.clone();
    let z_weight = weights.z_weight.clone();
    let alpha_weight = weights.alpha_weight.clone();
    let beta_weight = weights.beta_weight.clone();
    let normalized = alloc_f32(device, hidden)?;
    let normalized_q8_1 = alloc_q8_1(device, hidden)?;
    let alpha_raw = alloc_f32(device, value_heads)?;
    let beta_raw = alloc_f32(device, value_heads)?;
    let dt_bias = weights.dt_bias.clone();
    let ssm_a = weights.ssm_a.clone();

    let attn_norm_op = CudaPlanRmsnormF32::bind(
        stream,
        input,
        plan_f32_view(&attn_norm_weight, hidden)?,
        plan_f32_view(&normalized, hidden)?,
        spec.rms_norm_eps,
    )?;
    let normalized_q8_1_view = linear::q8_1_activation_view(&normalized_q8_1, hidden)?;
    let normalized_quantize_op = CudaQ8_1ActivationQuantizeArgsV1::bind(
        stream,
        plan_f32_view(&normalized, hidden)?,
        normalized_q8_1_view,
    )?;
    let qkv_op = row_input::Scalar::bind(
        stream, &qkv_weight, weights.input_layout, qkv_elements, hidden,
        normalized_q8_1_view,
        plan_f32_view(qkv, qkv_elements)?,
    )?;
    let z_op = row_input::Scalar::bind(
        stream, &z_weight, weights.input_layout, recurrent_elements, hidden,
        normalized_q8_1_view,
        plan_f32_view(z, recurrent_elements)?,
    )?;
    let alpha_op = CudaBf16F32MatvecArgsV1::bind(
        stream,
        CudaBf16WeightViewV1::new(
            alpha_weight.view(0, alpha_weight.len_bytes())?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_BF16_CUDA_V1,
            value_heads,
            hidden,
        )?,
        plan_f32_view(&normalized, hidden)?,
        plan_f32_view(&alpha_raw, value_heads)?,
    )?;
    let beta_op = CudaBf16F32MatvecArgsV1::bind(
        stream,
        CudaBf16WeightViewV1::new(
            beta_weight.view(0, beta_weight.len_bytes())?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_BF16_CUDA_V1,
            value_heads,
            hidden,
        )?,
        plan_f32_view(&normalized, hidden)?,
        plan_f32_view(&beta_raw, value_heads)?,
    )?;
    let postprocess_op = CudaQwen35LinearProjectionPostprocessArgsV1::bind(
        stream,
        plan_f32_view(&alpha_raw, value_heads)?,
        plan_f32_view(&beta_raw, value_heads)?,
        plan_f32_view(&dt_bias, value_heads)?,
        plan_f32_view(&ssm_a, value_heads)?,
        plan_f32_view(decay, value_heads)?,
        plan_f32_view(beta, value_heads)?,
    )?;
    Ok(LinearProjectionPlan {
        input_weight_layout: weights.input_layout,
        attn_norm_weight,
        qkv_weight,
        z_weight,
        alpha_weight,
        beta_weight,
        normalized,
        normalized_q8_1,
        alpha_raw,
        beta_raw,
        dt_bias,
        ssm_a,
        attn_norm_op,
        normalized_quantize_op,
        qkv_op,
        z_op,
        alpha_op,
        beta_op,
        postprocess_op,
    })
}

fn plan_f32_view(
    buffer: &OwnedCudaBuffer,
    elements: usize,
) -> Result<CudaF32View, Qwen35MixedPrefixCudaError> {
    let bytes = elements
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| invalid("F32 view extent overflow"))?;
    if buffer.len_bytes() != bytes {
        return Err(invalid("F32 view extent does not match allocation"));
    }
    Ok(CudaF32View::new(
        buffer.view(0, bytes)?,
        PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
        elements,
    )?)
}

fn plan_f32_subview(
    buffer: &OwnedCudaBuffer,
    offset_elements: usize,
    elements: usize,
) -> Result<CudaF32View, Qwen35MixedPrefixCudaError> {
    let offset = offset_elements
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| invalid("F32 subview offset overflow"))?;
    let bytes = elements
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| invalid("F32 subview extent overflow"))?;
    let end = offset
        .checked_add(bytes)
        .ok_or_else(|| invalid("F32 subview range overflow"))?;
    if end > buffer.len_bytes() {
        return Err(invalid("F32 subview exceeds allocation"));
    }
    Ok(CudaF32View::new(
        buffer.view(offset, bytes)?,
        PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
        elements,
    )?)
}

fn repeat_norm_values(
    values: &[f32],
    repeats: usize,
) -> Result<Vec<f32>, Qwen35MixedPrefixCudaError> {
    if values.is_empty() || repeats == 0 || values.iter().any(|value| !value.is_finite()) {
        return Err(invalid("full-attention normalization weights"));
    }
    let count = values
        .len()
        .checked_mul(repeats)
        .ok_or_else(|| invalid("full-attention normalization extent"))?;
    let mut result = Vec::with_capacity(count);
    for _ in 0..repeats {
        result.extend_from_slice(values);
    }
    Ok(result)
}

fn native_rope_tables(
    theta: f32,
    position: usize,
    rotary_dim: usize,
) -> Result<(Vec<f32>, Vec<f32>), Qwen35MixedPrefixCudaError> {
    if !theta.is_finite() || theta <= 1.0 || rotary_dim == 0 || !rotary_dim.is_multiple_of(2) {
        return Err(invalid("full-attention RoPE configuration"));
    }
    let mut cos = Vec::with_capacity(rotary_dim / 2);
    let mut sin = Vec::with_capacity(rotary_dim / 2);
    for pair in 0..rotary_dim / 2 {
        let exponent = -2.0_f32 * pair as f32 / rotary_dim as f32;
        let angle = position as f32 * theta.powf(exponent);
        let (s, c) = angle.sin_cos();
        if !s.is_finite() || !c.is_finite() {
            return Err(invalid("full-attention RoPE table"));
        }
        cos.push(c);
        sin.push(s);
    }
    Ok((cos, sin))
}

fn native_rope_table_span(
    theta: f32,
    token_count: usize,
    rotary_dim: usize,
) -> Result<(Vec<f32>, Vec<f32>), Qwen35MixedPrefixCudaError> {
    if token_count == 0
        || !theta.is_finite()
        || theta <= 1.0
        || rotary_dim == 0
        || !rotary_dim.is_multiple_of(2)
    {
        return Err(invalid("full-attention packed RoPE configuration"));
    }
    let elements = token_count
        .checked_mul(rotary_dim / 2)
        .filter(|count| {
            count
                .checked_mul(core::mem::size_of::<f32>())
                .is_some_and(|bytes| bytes <= isize::MAX as usize)
        })
        .ok_or_else(|| invalid("full-attention packed RoPE extent overflow"))?;
    let mut cosine = Vec::with_capacity(elements);
    let mut sine = Vec::with_capacity(elements);
    for position in 0..token_count {
        // Preserve the existing per-position FP32 math exactly. This change
        // coalesces allocation and upload, not the numerical contract.
        let (cos, sin) = native_rope_tables(theta, position, rotary_dim)?;
        cosine.extend(cos);
        sine.extend(sin);
    }
    Ok((cosine, sine))
}

fn build_full_plan(
    stream: &OwnedCudaStream,
    material: &FullMaterial,
    model_config: &Qwen35Config,
    capacity: usize,
    token_count: usize,
    input_source: Option<CudaF32View>,
) -> Result<FullPlan, Qwen35MixedPrefixCudaError> {
    build_full_plan_with_rope(
        stream,
        material,
        model_config,
        capacity,
        token_count,
        input_source,
        None,
    )
}

fn build_full_plan_with_rope(
    stream: &OwnedCudaStream,
    material: &FullMaterial,
    model_config: &Qwen35Config,
    capacity: usize,
    token_count: usize,
    input_source: Option<CudaF32View>,
    shared_rope: Option<Arc<FullRopeTables>>,
) -> Result<FullPlan, Qwen35MixedPrefixCudaError> {
    let weights = FullWeights::upload(stream, material)?;
    bind_full_plan(
        stream,
        &weights,
        model_config,
        capacity,
        token_count,
        input_source,
        shared_rope,
        #[cfg(feature = "qwen35-prefill-cublas")]
        None,
    )
}

fn bind_full_plan(
    stream: &OwnedCudaStream,
    material: &FullWeights,
    model_config: &Qwen35Config,
    capacity: usize,
    token_count: usize,
    input_source: Option<CudaF32View>,
    shared_rope: Option<Arc<FullRopeTables>>,
    #[cfg(feature = "qwen35-prefill-cublas")]
    qpn_session: Option<&target_weights::QpnScalarSession>,
) -> Result<FullPlan, Qwen35MixedPrefixCudaError> {
    bind_full_plan_with_position_schedule(
        stream, material, model_config, capacity, token_count, input_source, shared_rope, None,
        #[cfg(feature = "qwen35-prefill-cublas")]
        qpn_session,
    )
}

#[allow(clippy::too_many_arguments)]
fn bind_full_plan_with_position_schedule(
    stream: &OwnedCudaStream,
    material: &FullWeights,
    model_config: &Qwen35Config,
    capacity: usize,
    token_count: usize,
    input_source: Option<CudaF32View>,
    shared_rope: Option<Arc<FullRopeTables>>,
    position_schedule: Option<&augur_model_qwen35::Qwen35ImageMropePlan>,
    #[cfg(feature = "qwen35-prefill-cublas")]
    qpn_session: Option<&target_weights::QpnScalarSession>,
) -> Result<FullPlan, Qwen35MixedPrefixCudaError> {
    if token_count == 0 || token_count > capacity {
        return Err(invalid("full-attention token count exceeds cache capacity"));
    }
    let rope_tables = if let Some(tables) = shared_rope {
        if let Some(schedule) = position_schedule {
            tables.validate_image(
                stream, model_config.rope_theta as f32, token_count, material.spec.rotary_dim, schedule,
            )?;
        } else {
            tables.validate(
                stream, model_config.rope_theta as f32, token_count, material.spec.rotary_dim,
            )?;
        }
        tables
    } else if let Some(schedule) = position_schedule {
        if schedule.capacity() != token_count
            || model_config.rope_theta as f32 != 10_000_000.0
            || material.spec.rotary_dim != 64 {
            return Err(invalid("image MRoPE model/plan geometry mismatch"));
        }
        Arc::new(FullRopeTables::new_image(stream, schedule)?)
    } else {
        Arc::new(FullRopeTables::new(
            stream,
            model_config.rope_theta as f32,
            token_count,
            material.spec.rotary_dim,
        )?)
    };
    let device = stream.device_ordinal();
    let hidden = material.spec.hidden_size;
    let intermediate = material.spec.intermediate_size;
    let query_values = material.spec.query_heads * material.spec.head_dim;
    let kv_values = material.spec.kv_heads * material.spec.head_dim;
    let attn_norm_weight = material.attn_norm_weight.clone();
    let normalized = alloc_f32(device, hidden)?;
    let normalized_q8_1 = alloc_q8_1(device, hidden)?;
    let q_weight = material.q_weight.clone();
    let q_full = alloc_f32(device, query_values * 2)?;
    let q_norm_weight = material.q_norm_weight.clone();
    let q_normed = alloc_f32(device, query_values)?;
    let k_weight = material.k_weight.clone();
    let k_raw = alloc_f32(device, kv_values)?;
    let k_norm_weight = material.k_norm_weight.clone();
    let k_normed = alloc_f32(device, kv_values)?;
    let v_weight = material.v_weight.clone();
    let query = alloc_f32(device, query_values)?;
    #[cfg(not(feature = "qwen35-grouped-gqa-online-softmax"))]
    let gate = alloc_f32(device, query_values)?;
    let key = alloc_f32(device, kv_values)?;
    let value = alloc_f32(device, kv_values)?;
    #[cfg(not(feature = "qwen35-kv8-full-attention"))]
    let cache_elements = capacity
        .checked_mul(kv_values)
        .ok_or_else(|| invalid("full-attention cache extent overflow"))?;
    #[cfg(not(feature = "qwen35-kv8-full-attention"))]
    let mut key_cache = alloc_f32(device, cache_elements)?;
    #[cfg(not(feature = "qwen35-kv8-full-attention"))]
    let mut value_cache = alloc_f32(device, cache_elements)?;
    #[cfg(feature = "qwen35-kv8-full-attention")]
    let cache_bytes = capacity
        .checked_mul(QWEN35_KV8_CACHE_ROW_BYTES)
        .ok_or_else(|| invalid("full-attention KV8 cache extent overflow"))?;
    #[cfg(feature = "qwen35-kv8-full-attention")]
    let mut key_cache = OwnedCudaBuffer::allocate(device, cache_bytes)?;
    #[cfg(feature = "qwen35-kv8-full-attention")]
    let mut value_cache = OwnedCudaBuffer::allocate(device, cache_bytes)?;
    #[cfg(all(
        feature = "qwen35-prefill-cublas",
        feature = "qwen35-kv8-full-attention",
        not(feature = "qwen35-kv8-chunk-prefill")
    ))]
    let prefill_cache_elements = capacity
        .checked_mul(kv_values)
        .ok_or_else(|| invalid("full-attention prefill F32 cache extent overflow"))?;
    #[cfg(all(
        feature = "qwen35-prefill-cublas",
        feature = "qwen35-kv8-full-attention",
        not(feature = "qwen35-kv8-chunk-prefill")
    ))]
    let mut prefill_key_cache_f32 = alloc_f32(device, prefill_cache_elements)?;
    #[cfg(all(
        feature = "qwen35-prefill-cublas",
        feature = "qwen35-kv8-full-attention",
        not(feature = "qwen35-kv8-chunk-prefill")
    ))]
    let mut prefill_value_cache_f32 = alloc_f32(device, prefill_cache_elements)?;
    #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
    let attention_workspace = alloc_f32(device, QWEN35_FULL_ATTENTION_WORKSPACE_ELEMENTS_V2)?;
    let attention = alloc_f32(device, query_values)?;
    let gated = alloc_f32(device, query_values)?;
    #[cfg(not(feature = "qwen35-grouped-gqa-online-softmax"))]
    let gated_q8_1 = alloc_q8_1(device, query_values)?;
    let output = alloc_f32(device, hidden)?;
    let mut input = alloc_f32(device, hidden)?;
    let (input_view, input_is_alias) = match input_source {
        Some(view) => {
            if view.element_count() != hidden || view.buffer_view().device_ordinal() != device {
                return Err(invalid("full-attention input alias geometry"));
            }
            (view, true)
        }
        None => (plan_f32_view(&input, hidden)?, false),
    };
    let attention_residual = alloc_f32(device, hidden)?;
    let post_norm = Rc::new(alloc_f32(device, hidden)?);
    let ffn_gate_weight = material.ffn.gate.clone();
    let ffn_up_weight = material.ffn.up.clone();
    let ffn_gate_output = alloc_f32(device, intermediate)?;
    let ffn_up_output = alloc_f32(device, intermediate)?;
    let ffn_activation = alloc_f32(device, intermediate)?;
    let ffn_input_q8_1 = alloc_q8_1(device, hidden)?;
    let ffn_down_q8_1 = alloc_q8_1(device, intermediate)?;
    let ffn_down_weight = material.ffn.down.clone();
    let ffn_output = Rc::new(alloc_f32(device, hidden)?);
    let final_hidden = alloc_f32(device, hidden)?;
    let output_weight = material.output_weight.clone();
    let post_norm_weight = material.post_norm_weight.clone();

    key_cache.zero_sync(stream)?;
    value_cache.zero_sync(stream)?;
    #[cfg(all(
        feature = "qwen35-prefill-cublas",
        feature = "qwen35-kv8-full-attention",
        not(feature = "qwen35-kv8-chunk-prefill")
    ))]
    {
        prefill_key_cache_f32.zero_sync(stream)?;
        prefill_value_cache_f32.zero_sync(stream)?;
    }
    if !input_is_alias {
        input.zero_sync(stream)?;
    }

    let attn_norm_op = CudaPlanRmsnormF32::bind(
        stream,
        input_view,
        plan_f32_view(&attn_norm_weight, hidden)?,
        plan_f32_view(&normalized, hidden)?,
        material.spec.rms_norm_eps,
    )?;
    let normalized_q8_1_view = linear::q8_1_activation_view(&normalized_q8_1, hidden)?;
    let qkv_quantize_op = CudaQ8_1ActivationQuantizeArgsV1::bind(
        stream,
        plan_f32_view(&normalized, hidden)?,
        normalized_q8_1_view,
    )?;
    let q_projection_op = row_input::Scalar::bind(
        stream, &q_weight, material.input_layout, query_values * 2, hidden,
        normalized_q8_1_view,
        plan_f32_view(&q_full, query_values * 2)?,
    )?;
    let k_projection_op = row_input::Scalar::bind(
        stream, &k_weight, material.input_layout, kv_values, hidden,
        normalized_q8_1_view,
        plan_f32_view(&k_raw, kv_values)?,
    )?;
    let v_projection_op = row_input::Scalar::bind(
        stream, &v_weight, material.input_layout, kv_values, hidden,
        normalized_q8_1_view,
        plan_f32_view(&value, kv_values)?,
    )?;
    #[cfg(not(feature = "qwen35-fused-head-norm-rope"))]
    let mut q_norm_ops = Vec::with_capacity(material.spec.query_heads);
    #[cfg(not(feature = "qwen35-fused-head-norm-rope"))]
    for head in 0..material.spec.query_heads {
        let offset = head * material.spec.head_dim;
        q_norm_ops.push(CudaRmsnormF32ArgsV1::bind(
            stream,
            plan_f32_subview(&q_full, offset, material.spec.head_dim)?,
            plan_f32_subview(&q_norm_weight, offset, material.spec.head_dim)?,
            plan_f32_subview(&q_normed, offset, material.spec.head_dim)?,
            material.spec.rms_norm_eps,
        )?);
    }
    #[cfg(not(feature = "qwen35-fused-head-norm-rope"))]
    let mut k_norm_ops = Vec::with_capacity(material.spec.kv_heads);
    #[cfg(not(feature = "qwen35-fused-head-norm-rope"))]
    for head in 0..material.spec.kv_heads {
        let offset = head * material.spec.head_dim;
        k_norm_ops.push(CudaRmsnormF32ArgsV1::bind(
            stream,
            plan_f32_subview(&k_raw, offset, material.spec.head_dim)?,
            plan_f32_subview(&k_norm_weight, offset, material.spec.head_dim)?,
            plan_f32_subview(&k_normed, offset, material.spec.head_dim)?,
            material.spec.rms_norm_eps,
        )?);
    }
    #[cfg(not(feature = "qwen35-fused-head-norm-rope"))]
    let mut q_rope_ops = Vec::with_capacity(token_count);
    #[cfg(not(feature = "qwen35-fused-head-norm-rope"))]
    let mut k_rope_ops = Vec::with_capacity(token_count);
    #[cfg(feature = "qwen35-fused-head-norm-rope")]
    let mut fused_head_norm_rope_ops = Vec::with_capacity(token_count);
    for position in 0..token_count {
        let row_elements = material.spec.rotary_dim / 2;
        let cos = plan_f32_subview(&rope_tables.cos, position * row_elements, row_elements)?;
        let sin = plan_f32_subview(&rope_tables.sin, position * row_elements, row_elements)?;
        #[cfg(not(feature = "qwen35-fused-head-norm-rope"))]
        {
            let mut q_ops = Vec::with_capacity(material.spec.query_heads);
            for head in 0..material.spec.query_heads {
                let offset = head * material.spec.head_dim;
                q_ops.push(CudaRopeF32ArgsV1::bind(
                    stream,
                    plan_f32_subview(&q_normed, offset, material.spec.head_dim)?,
                    plan_f32_subview(&query, offset, material.spec.head_dim)?,
                    cos,
                    sin,
                    material.spec.rotary_dim,
                )?);
            }
            let mut k_ops = Vec::with_capacity(material.spec.kv_heads);
            for head in 0..material.spec.kv_heads {
                let offset = head * material.spec.head_dim;
                k_ops.push(CudaRopeF32ArgsV1::bind(
                    stream,
                    plan_f32_subview(&k_normed, offset, material.spec.head_dim)?,
                    plan_f32_subview(&key, offset, material.spec.head_dim)?,
                    cos,
                    sin,
                    material.spec.rotary_dim,
                )?);
            }
            q_rope_ops.push(q_ops);
            k_rope_ops.push(k_ops);
        }
        #[cfg(feature = "qwen35-fused-head-norm-rope")]
        fused_head_norm_rope_ops.push(CudaQwen35FusedHeadNormRopeArgsV1::bind(
            stream,
            CudaQwen35FusedHeadNormRopeViewsV1 {
                q_input: plan_f32_subview(&q_full, 0, query_values)?,
                q_weight: plan_f32_view(&q_norm_weight, query_values)?,
                q_normed: plan_f32_view(&q_normed, query_values)?,
                q_output: plan_f32_view(&query, query_values)?,
                k_input: plan_f32_view(&k_raw, kv_values)?,
                k_weight: plan_f32_view(&k_norm_weight, kv_values)?,
                k_normed: plan_f32_view(&k_normed, kv_values)?,
                k_output: plan_f32_view(&key, kv_values)?,
                rope_cos: cos,
                rope_sin: sin,
            },
            material.spec.rms_norm_eps,
        )?);
    }

    let mut attention_args = Vec::with_capacity(token_count);
    for position in 0..token_count {
        #[cfg(not(feature = "qwen35-grouped-gqa-online-softmax"))]
        attention_args.push(full::bind_attention(
            stream,
            &query,
            &gate,
            &key,
            &value,
            &key_cache,
            &value_cache,
            &attention,
            &gated,
            capacity,
            position,
        )?);
        #[cfg(all(
            feature = "qwen35-grouped-gqa-online-softmax",
            not(feature = "qwen35-kv8-full-attention")
        ))]
        attention_args.push(full::bind_attention_v2_with_gate_view(
            stream,
            &query,
            plan_f32_subview(&q_full, query_values, query_values)?,
            &key,
            &value,
            &key_cache,
            &value_cache,
            &attention,
            &gated,
            &attention_workspace,
            capacity,
            position,
        )?);
        #[cfg(feature = "qwen35-kv8-full-attention")]
        attention_args.push(full::bind_attention_v3_with_gate_view(
            stream,
            &query,
            plan_f32_subview(&q_full, query_values, query_values)?,
            &key,
            &value,
            &key_cache,
            &value_cache,
            &attention,
            &gated,
            &attention_workspace,
            capacity,
            position,
        )?);
    }
    #[cfg(not(feature = "qwen35-grouped-gqa-online-softmax"))]
    let gated_q8_1_view = linear::q8_1_activation_view(&gated_q8_1, query_values)?;
    #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
    let gated_q8_1_view = CudaQ8_1ActivationViewV1::new(
        attention_workspace.view(
            QWEN35_FULL_ATTENTION_PARTIAL_ELEMENTS_V2 * core::mem::size_of::<f32>(),
            QWEN35_FULL_ATTENTION_FUSED_Q8_BYTES_V2,
        )?,
        PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_1_ACTIVATION_CUDA_V1,
        query_values,
    )?;
    #[cfg(not(feature = "qwen35-grouped-gqa-online-softmax"))]
    let output_quantize_op = CudaQ8_1ActivationQuantizeArgsV1::bind(
        stream,
        plan_f32_view(&gated, query_values)?,
        gated_q8_1_view,
    )?;
    let output_args = CudaQ8_0Q8_1MatvecArgsV1::bind(
        stream,
        CudaQ8_0WeightViewV1::new(
            output_weight.view(0, output_weight.len_bytes())?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_0_CUDA_V1,
            hidden,
            query_values,
        )?,
        gated_q8_1_view,
        plan_f32_view(&output, hidden)?,
    )?;
    let attention_residual_op = CudaAddF32ArgsV1::bind(
        stream,
        plan_f32_view(&output, hidden)?,
        input_view,
        plan_f32_view(&attention_residual, hidden)?,
    )?;
    let post_norm_op = CudaPlanRmsnormF32::bind(
        stream,
        plan_f32_view(&attention_residual, hidden)?,
        plan_f32_view(&post_norm_weight, hidden)?,
        plan_f32_view(&post_norm, hidden)?,
        material.spec.rms_norm_eps,
    )?;
    let ffn = bind_scalar_ffn(stream, &material.ffn, &post_norm, &ffn_output,
        &ffn_gate_output, &ffn_up_output, &ffn_activation, &ffn_input_q8_1, &ffn_down_q8_1,
        #[cfg(feature = "qwen35-prefill-cublas")]
        qpn_session)?;
    let final_residual_op = CudaAddF32ArgsV1::bind(
        stream,
        plan_f32_view(&attention_residual, hidden)?,
        plan_f32_view(&ffn_output, hidden)?,
        plan_f32_view(&final_hidden, hidden)?,
    )?;
    Ok(FullPlan {
        input_weight_layout: material.input_layout,
        #[cfg(feature = "qwen35-prefill-cublas")]
        rotated_attention: None,
        attn_norm_weight,
        normalized,
        normalized_q8_1,
        q_weight,
        q_full,
        q_norm_weight,
        q_normed,
        k_weight,
        k_raw,
        k_norm_weight,
        k_normed,
        v_weight,
        rope_tables,
        query,
        #[cfg(not(feature = "qwen35-grouped-gqa-online-softmax"))]
        gate,
        key,
        value,
        key_cache,
        value_cache,
        #[cfg(all(
            feature = "qwen35-prefill-cublas",
            feature = "qwen35-kv8-full-attention",
            not(feature = "qwen35-kv8-chunk-prefill")
        ))]
        prefill_key_cache_f32,
        #[cfg(all(
            feature = "qwen35-prefill-cublas",
            feature = "qwen35-kv8-full-attention",
            not(feature = "qwen35-kv8-chunk-prefill")
        ))]
        prefill_value_cache_f32,
        #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
        attention_workspace,
        attention,
        gated,
        #[cfg(not(feature = "qwen35-grouped-gqa-online-softmax"))]
        gated_q8_1,
        output,
        input,
        input_view,
        input_is_alias,
        attention_residual,
        post_norm,
        ffn_gate_weight,
        ffn_up_weight,
        ffn_gate_output,
        ffn_up_output,
        ffn_activation,
        ffn_input_q8_1,
        ffn_down_q8_1,
        ffn_down_weight,
        ffn_output,
        final_hidden,
        output_weight,
        post_norm_weight,
        ffn_weight_type: material.ffn.weight_type,
        ffn_gate_scale: material.ffn.gate_scale,
        ffn_up_scale: material.ffn.up_scale,
        ffn_down_scale: material.ffn.down_scale,
        ffn_is_qpn: material.ffn.qpn_enabled(),
        #[cfg(feature = "qwen35-prefill-cublas")]
        qpn_weights: material.ffn.qpn.clone(),
        attention_args,
        output_args,
        attn_norm_op,
        qkv_quantize_op,
        q_projection_op,
        k_projection_op,
        v_projection_op,
        #[cfg(not(feature = "qwen35-fused-head-norm-rope"))]
        q_norm_ops,
        #[cfg(not(feature = "qwen35-fused-head-norm-rope"))]
        k_norm_ops,
        #[cfg(not(feature = "qwen35-fused-head-norm-rope"))]
        q_rope_ops,
        #[cfg(not(feature = "qwen35-fused-head-norm-rope"))]
        k_rope_ops,
        #[cfg(feature = "qwen35-fused-head-norm-rope")]
        fused_head_norm_rope_ops,
        #[cfg(not(feature = "qwen35-grouped-gqa-online-softmax"))]
        output_quantize_op,
        attention_residual_op,
        post_norm_op,
        ffn,
        final_residual_op,
    })
}

impl LinearPlan {
    fn reset(&mut self, stream: &OwnedCudaStream) -> Result<(), Qwen35MixedPrefixCudaError> {
        self.history.zero_sync(stream)?;
        self.state.zero_sync(stream)?;
        Ok(())
    }

    fn upload(
        &self,
        stream: &OwnedCudaStream,
        projected: Option<&linear::ProjectedToken>,
        hidden: &[f32],
        device_projection: bool,
    ) -> Result<(), Qwen35MixedPrefixCudaError> {
        if self.input_is_alias {
            return Err(invalid("cannot upload an aliased linear input"));
        }
        if device_projection {
            self.input.copy_from_host_sync(stream, &f32_bytes(hidden))?;
        } else {
            let projected = projected.ok_or_else(|| invalid("linear projection is missing"))?;
            linear::upload_token(
                stream,
                &self.qkv,
                &self.z,
                &self.decay,
                &self.beta,
                &self.input,
                projected,
                hidden,
            )?;
        }
        Ok(())
    }

    fn execute(
        &self,
        stream: &OwnedCudaStream,
    ) -> Result<DeviceLayerReadback, Qwen35MixedPrefixCudaError> {
        if let Some(projection) = &self.projection {
            projection.submit(stream)?;
        }
        let readback = linear::execute_and_readback(
            stream,
            &self.chain,
            &self.state,
            &self.history,
            &self.attention,
            &self.attention_residual,
            &self.post_norm,
            &self.ffn_output,
            &self.final_hidden,
            &self.recurrent,
        )?;
        Ok(DeviceLayerReadback {
            input_hidden: Vec::new(),
            ffn_intermediates: None,
            attention_intermediates: None,
            full_attention_intermediates: None,
            attention_output: readback.attention,
            attention_residual: readback.attention_residual,
            post_norm: readback.post_norm,
            ffn_output: readback.ffn_output,
            final_hidden: readback.final_hidden,
            state: readback.state,
            history: readback.history,
            polls: readback.polls,
            proof: readback.proof,
            completion_boundaries: convert_boundary(readback.completion_boundaries),
            readback_boundaries: convert_boundary(readback.readback_boundaries),
        })
    }

    fn submit_resident(&self, stream: &OwnedCudaStream) -> Result<(), Qwen35MixedPrefixCudaError> {
        if let Some(projection) = &self.projection {
            projection.submit(stream)?;
        }
        self.chain.submit(stream)?;
        Ok(())
    }
}

impl LinearProjectionPlan {
    fn submit(&self, stream: &OwnedCudaStream) -> Result<(), Qwen35MixedPrefixCudaError> {
        self.attn_norm_op.submit_for_plan(stream)?;
        self.normalized_quantize_op.submit_for_plan(stream)?;
        self.qkv_op.submit_for_plan(stream)?;
        self.z_op.submit_for_plan(stream)?;
        self.alpha_op.submit_for_plan(stream)?;
        self.beta_op.submit_for_plan(stream)?;
        self.postprocess_op.submit_for_plan(stream)?;
        Ok(())
    }
}

/// One owner-stream arena shared serially by all sixteen full-attention layers.
/// The cache representation belongs to the live plan, not a per-submit switch.
#[cfg(feature = "qwen35-prefill-cublas")]
struct RotatedFullAttention {
    tile8_rows: bool,
    planar: bool,
    gqa8: Option<augur_backend::cuda_qwen35_rotated_kv::CudaQwen35RotatedKvGqa8PlanV1>,
    prefix: augur_backend::cuda_qwen35_rotated_kv::CudaQwen35RotatedKvPlanV1,
    small: augur_backend::cuda_qwen35_rotated_kv::CudaQwen35RotatedKvSmallPlanV1,
    input: Option<augur_backend::cuda_qwen35_rotated_kv::CudaQwen35RotatedKvInputPlanV1>,
    workspace: OwnedCudaBuffer,
    capacity: usize,
}

#[cfg(feature = "qwen35-prefill-cublas")]
impl RotatedFullAttention {
    fn planar_switch(value: Option<&std::ffi::OsStr>, gqa8: bool)
        -> Result<bool, Qwen35MixedPrefixCudaError> {
        match value {
            None => Ok(false),
            Some(v) if v == "1" && gqa8 => Ok(true),
            _ => Err(invalid("planar KV requires owner-fixed GQA8 and exact input consumers")),
        }
    }
    fn gqa8_switch(value: Option<&std::ffi::OsStr>, exact_input: bool, conflicting: bool)
        -> Result<bool, Qwen35MixedPrefixCudaError> {
        match value {
            None => Ok(false),
            Some(v) if v == "1" && exact_input && !conflicting => Ok(true),
            _ => Err(invalid("GQA8 requires exact-input owner and excludes tile8/vector/serial overrides")),
        }
    }
    fn tile8_switch(value: Option<&std::ffi::OsStr>, eligible: bool) -> Result<bool, Qwen35MixedPrefixCudaError> {
        match value {
            None => Ok(false),
            Some(v) if v == "1" && eligible => Ok(true),
            _ => Err(invalid("tile8 row contract requires exact-input rotated owner and excludes other row overrides")),
        }
    }

    fn new(stream: &OwnedCudaStream, capacity: usize, exact_input: bool) -> Result<Arc<Self>, Qwen35MixedPrefixCudaError> {
        let gqa8_enabled = Self::gqa8_switch(std::env::var_os("AUGUR_QWEN35_SMALL_ATTENTION_GQA8_MMA").as_deref(),
            exact_input, ["AUGUR_QWEN35_SMALL_ATTENTION_TILE8_ROWS", "AUGUR_QWEN35_VERIFY_ATTENTION_VECTOR_ROWS",
                "AUGUR_QWEN35_VERIFY_ATTENTION_SERIAL_ROWS"].iter().any(|k| std::env::var_os(k).is_some()))?;
        let tile8_rows = Self::tile8_switch(std::env::var_os("AUGUR_QWEN35_SMALL_ATTENTION_TILE8_ROWS").as_deref(),
            exact_input && std::env::var_os("AUGUR_QWEN35_VERIFY_ATTENTION_VECTOR_ROWS").is_none()
                && std::env::var_os("AUGUR_QWEN35_VERIFY_ATTENTION_SERIAL_ROWS").is_none())?;
        let planar = Self::planar_switch(std::env::var_os("AUGUR_QWEN35_ROTATED_KV_PLANAR").as_deref(), gqa8_enabled)?;
        use augur_backend::cuda_qwen35_rotated_kv::{CudaQwen35RotatedKvPlanV1 as Prefix, CudaQwen35RotatedKvSmallPlanV1 as Small, CudaQwen35RotatedKvInputPlanV1 as Input};
        let mut bytes = Prefix::workspace_bytes(512)?.max(Small::workspace_bytes(capacity, 8)?);
        let gqa8 = if gqa8_enabled {
            use augur_backend::cuda_qwen35_rotated_kv::CudaQwen35RotatedKvGqa8PlanV1 as Gqa8;
            bytes = bytes.max(Gqa8::workspace_bytes(capacity, 8)?);
            Some(if planar { Gqa8::new_planar(stream)? } else { Gqa8::new(stream)? })
        } else { None };
        let input = if exact_input {
            bytes = bytes.max(Input::workspace_bytes(capacity, 512)?);
            Some(if planar { Input::new_planar(stream)? } else { Input::new(stream)? })
        } else { None };
        Ok(Arc::new(Self {
            tile8_rows,
            planar,
            gqa8,
            prefix: Prefix::new(stream)?,
            small: Small::new(stream)?,
            input,
            workspace: OwnedCudaBuffer::allocate(stream.device_ordinal(), bytes)?,
            capacity,
        }))
    }

    fn submit(
        &self, stream: &OwnedCudaStream, plan: &FullPlan, position: usize, rows: usize,
        query: CudaBufferView, gate: CudaBufferView, key: CudaBufferView,
        value: CudaBufferView, attention: CudaBufferView, gated: CudaBufferView,
    ) -> Result<(), Qwen35MixedPrefixCudaError> {
        self.submit_impl(stream, plan, position, rows, query, gate, key, value, attention, gated, false, false)
    }

    fn submit_vector_rows(
        &self, stream: &OwnedCudaStream, plan: &FullPlan, position: usize, rows: usize,
        query: CudaBufferView, gate: CudaBufferView, key: CudaBufferView,
        value: CudaBufferView, attention: CudaBufferView, gated: CudaBufferView,
    ) -> Result<(), Qwen35MixedPrefixCudaError> {
        if !(1..=8).contains(&rows) { return Err(invalid("vector verify attention row admission")); }
        self.submit_impl(stream, plan, position, rows, query, gate, key, value, attention, gated, false, true)
    }

    // Input-only admission. Verification and ordinary decode retain their old
    // row contract. This is not whole-model semantic or product qualification.
    fn submit_exact_input(
        &self, stream: &OwnedCudaStream, plan: &FullPlan, position: usize, rows: usize,
        query: CudaBufferView, gate: CudaBufferView, key: CudaBufferView,
        value: CudaBufferView, attention: CudaBufferView, gated: CudaBufferView,
    ) -> Result<(), Qwen35MixedPrefixCudaError> {
        self.submit_impl(stream, plan, position, rows, query, gate, key, value, attention, gated, true, false)
    }

    fn validate_rows(rows: usize, exact_input: bool) -> Result<(), Qwen35MixedPrefixCudaError> {
        if if exact_input { !(1..=512).contains(&rows) } else { !matches!(rows, 1..=8 | 512) } {
            return Err(invalid("rotated runtime row admission"));
        }
        Ok(())
    }

    fn submit_impl(
        &self, stream: &OwnedCudaStream, plan: &FullPlan, position: usize, rows: usize,
        query: CudaBufferView, gate: CudaBufferView, key: CudaBufferView,
        value: CudaBufferView, attention: CudaBufferView, gated: CudaBufferView,
        exact_input: bool, vector_rows: bool,
    ) -> Result<(), Qwen35MixedPrefixCudaError> {
        use augur_backend::cuda_qwen35_rotated_kv::{CudaQwen35RotatedKvPlanV1 as Prefix, CudaQwen35RotatedKvSmallPlanV1 as Small, CudaQwen35RotatedKvInputPlanV1 as Input, Qwen35RotatedKvViews};
        Self::validate_rows(rows, exact_input)?;
        if self.planar && rows > 8 && !exact_input {
            return Err(invalid("planar KV rejects legacy prefix consumer"));
        }
        let bytes = if self.gqa8.is_some() && rows <= 8 {
            augur_backend::cuda_qwen35_rotated_kv::CudaQwen35RotatedKvGqa8PlanV1::workspace_bytes(self.capacity, rows)?
        } else if rows <= 8 { Small::workspace_bytes(self.capacity, rows)? }
            else if exact_input { Input::workspace_bytes(self.capacity, rows)? }
            else { Prefix::workspace_bytes(rows)? };
        let views = Qwen35RotatedKvViews {
            query, gate, key, value, attention, gated,
            key_cache: plan.key_cache.view(0, plan.key_cache.len_bytes())?,
            value_cache: plan.value_cache.view(0, plan.value_cache.len_bytes())?,
            workspace: self.workspace.view(0, bytes)?,
        };
        if let Some(gqa8) = self.gqa8.as_ref().filter(|_| rows <= 8) {
            gqa8.submit(stream, views, self.capacity, position, rows)?;
        }
        else if self.tile8_rows && rows <= 8 { self.small.submit_tile8_rows(stream, views, self.capacity, position, rows)?; }
        else if vector_rows { self.small.submit_vector_rows(stream, views, self.capacity, position, rows)?; }
        else if rows <= 8 { self.small.submit(stream, views, self.capacity, position, rows)?; }
        else if exact_input {
            self.input.as_ref().ok_or_else(|| invalid("exact input attention plan not configured"))?
                .submit(stream, views, self.capacity, position, rows)?;
        }
        else { self.prefix.submit(stream, views, self.capacity, position, rows)?; }
        Ok(())
    }
}

#[cfg(all(test, feature = "qwen35-prefill-cublas"))]
mod tile8_row_contract_tests {
    use super::*;
    #[test]
    fn planar_format_is_owner_fixed_and_requires_gqa8() {
        use std::ffi::OsStr;
        for gqa8 in [false, true] {
            assert!(!RotatedFullAttention::planar_switch(None, gqa8).unwrap());
            assert_eq!(RotatedFullAttention::planar_switch(Some(OsStr::new("1")), gqa8).is_ok(), gqa8);
            for v in ["0", "", "true", "2"] {
                assert!(RotatedFullAttention::planar_switch(Some(OsStr::new(v)), gqa8).is_err());
            }
        }
    }
    #[test]
    fn gqa8_contract_rejects_mixed_owner_arithmetic() {
        use std::ffi::OsStr;
        for exact in [false, true] {
            for conflict in [false, true] {
                assert!(!RotatedFullAttention::gqa8_switch(None, exact, conflict).unwrap());
                assert_eq!(RotatedFullAttention::gqa8_switch(Some(OsStr::new("1")), exact, conflict).is_ok(),
                    exact && !conflict);
            }
        }
        for v in ["0", "", "true", "1 "] {
            assert!(RotatedFullAttention::gqa8_switch(Some(OsStr::new(v)), true, false).is_err());
        }
    }
    #[test]
    fn tile8_contract_is_explicit_and_owner_bound() {
        use std::ffi::OsStr;
        assert!(!RotatedFullAttention::tile8_switch(None, false).unwrap());
        assert!(RotatedFullAttention::tile8_switch(Some(OsStr::new("1")), true).unwrap());
        assert!(RotatedFullAttention::tile8_switch(Some(OsStr::new("1")), false).is_err());
        for value in ["0", "", "true", "1 "] {
            assert!(RotatedFullAttention::tile8_switch(Some(OsStr::new(value)), true).is_err());
        }
    }
}

impl FullPlan {
    fn submit_attention(&self, stream: &OwnedCudaStream, position: usize) -> Result<(), Qwen35MixedPrefixCudaError> {
        if position >= self.attention_args.len() {
            return Err(invalid("full-attention position exceeds bound"));
        }
        #[cfg(feature = "qwen35-prefill-cublas")]
        if let Some(rotated) = &self.rotated_attention {
            let q = full::QUERY_VALUES;
            rotated.submit(stream, self, position, 1,
                self.query.view(0, q * 4)?, self.q_full.view(q * 4, q * 4)?,
                self.key.view(0, full::KV_VALUES * 4)?, self.value.view(0, full::KV_VALUES * 4)?,
                self.attention.view(0, q * 4)?, self.gated.view(0, q * 4)?)?;
            #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
            CudaQ8_1ActivationQuantizeArgsV1::bind(stream,
                plan_f32_view(&self.gated, q)?,
                CudaQ8_1ActivationViewV1::new(self.attention_workspace.view(
                    QWEN35_FULL_ATTENTION_PARTIAL_ELEMENTS_V2 * 4,
                    QWEN35_FULL_ATTENTION_FUSED_Q8_BYTES_V2)?,
                    PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_1_ACTIVATION_CUDA_V1, q)?)?
                .submit_for_plan(stream)?;
            return Ok(());
        }
        self.attention_args[position].submit_for_plan(stream)?;
        Ok(())
    }

    fn reset(&mut self, stream: &OwnedCudaStream) -> Result<(), Qwen35MixedPrefixCudaError> {
        self.key_cache.zero_sync(stream)?;
        self.value_cache.zero_sync(stream)?;
        #[cfg(all(
            feature = "qwen35-prefill-cublas",
            feature = "qwen35-kv8-full-attention",
            not(feature = "qwen35-kv8-chunk-prefill")
        ))]
        {
            self.prefill_key_cache_f32.zero_sync(stream)?;
            self.prefill_value_cache_f32.zero_sync(stream)?;
        }
        Ok(())
    }

    fn upload(
        &self,
        stream: &OwnedCudaStream,
        hidden: &[f32],
    ) -> Result<(), Qwen35MixedPrefixCudaError> {
        if self.input_is_alias {
            return Err(invalid("cannot upload an aliased full-attention input"));
        }
        // Full-attention Q/K/V are projected by the native device operations
        // submitted in `execute`; the CPU projection is retained only for the
        // independent oracle and is never uploaded here.
        self.input.copy_from_host_sync(stream, &f32_bytes(hidden))?;
        Ok(())
    }

    fn execute(
        &self,
        stream: &OwnedCudaStream,
        position: usize,
    ) -> Result<DeviceLayerReadback, Qwen35MixedPrefixCudaError> {
        let completion_before = cuda_host_boundary_snapshot();
        self.submit_resident(stream, position)?;
        let receipt = stream.submission_receipt()?;
        let event = OwnedCudaEvent::create(stream.device_ordinal())?;
        event.record_for_submission(stream, receipt)?;
        let mut polls = 0_u64;
        let proof = loop {
            if let Some(proof) = stream.query_event_completion(&event, receipt)? {
                break proof;
            }
            polls = polls.saturating_add(1);
            if polls >= MAX_COMPLETION_POLLS {
                return Err(Qwen35MixedPrefixCudaError::CompletionPollLimit);
            }
            bounded_completion_poll_backoff(polls);
        };
        let proof_ok = proof.covers_submission(receipt) && proof.belongs_to_stream(stream)?;
        if !proof_ok {
            return Err(invalid("full-attention owner epoch proof"));
        }
        let completion_boundaries =
            boundary_delta(completion_before, cuda_host_boundary_snapshot())?;
        let readback_before = cuda_host_boundary_snapshot();
        let attention_output = read_f32_buffer(stream, &self.output, receipt)?;
        let attention_residual = read_f32_buffer(stream, &self.attention_residual, receipt)?;
        let post_norm = read_f32_buffer(stream, &self.post_norm, receipt)?;
        let ffn_output = read_f32_buffer(stream, &self.ffn_output, receipt)?;
        let final_hidden = read_f32_buffer(stream, &self.final_hidden, receipt)?;
        let mut state = read_full_cache_buffer(stream, &self.key_cache, receipt)?;
        state.extend(read_full_cache_buffer(stream, &self.value_cache, receipt)?);
        Ok(DeviceLayerReadback {
            input_hidden: Vec::new(),
            ffn_intermediates: None,
            attention_intermediates: None,
            full_attention_intermediates: None,
            attention_output,
            attention_residual,
            post_norm,
            ffn_output,
            final_hidden,
            state,
            history: Vec::new(),
            polls,
            proof: proof_ok,
            completion_boundaries,
            readback_boundaries: boundary_delta(readback_before, cuda_host_boundary_snapshot())?,
        })
    }

    fn submit_resident(
        &self,
        stream: &OwnedCudaStream,
        position: usize,
    ) -> Result<(), Qwen35MixedPrefixCudaError> {
        self.submit_resident_with_attention_timing(stream, position, None)
    }

    fn submit_resident_with_attention_timing(
        &self,
        stream: &OwnedCudaStream,
        position: usize,
        attention_timing: Option<(&OwnedCudaEvent, &OwnedCudaEvent)>,
    ) -> Result<(), Qwen35MixedPrefixCudaError> {
        if position >= self.attention_args.len() {
            return Err(invalid("full-attention position exceeds bound"));
        }
        // Project and prepare Q/K/V on the device. Each Q/K head receives its
        // own RMSNorm operation; a global norm over the concatenated vectors
        // would change the model's attention geometry.
        self.attn_norm_op.submit_for_plan(stream)?;
        self.qkv_quantize_op.submit_for_plan(stream)?;
        self.q_projection_op.submit_for_plan(stream)?;
        self.k_projection_op.submit_for_plan(stream)?;
        self.v_projection_op.submit_for_plan(stream)?;
        #[cfg(feature = "qwen35-fused-head-norm-rope")]
        self.fused_head_norm_rope_ops[position].submit_for_plan(stream)?;
        #[cfg(not(feature = "qwen35-fused-head-norm-rope"))]
        {
            for op in &self.q_norm_ops {
                op.submit_for_plan(stream)?;
            }
            for op in &self.k_norm_ops {
                op.submit_for_plan(stream)?;
            }
            for op in &self.q_rope_ops[position] {
                op.submit_for_plan(stream)?;
            }
            for op in &self.k_rope_ops[position] {
                op.submit_for_plan(stream)?;
            }
        }
        // The V1 boundary owns a separate gate buffer. The grouped-GQA V2
        // boundary binds the already-contiguous q_full tail directly.
        #[cfg(not(feature = "qwen35-grouped-gqa-online-softmax"))]
        {
            let gate_source = self.q_full.view(
                full::QUERY_VALUES * core::mem::size_of::<f32>(),
                full::QUERY_VALUES * core::mem::size_of::<f32>(),
            )?;
            // SAFETY: both ranges belong to this plan, are disjoint, and remain
            // live through the completion proof below.
            unsafe {
                self.gate
                    .copy_from_device_async_unchecked(stream, 0, gate_source)?;
            }
        }
        if let Some((started, _)) = attention_timing {
            started.record(stream)?;
        }
        self.submit_attention(stream, position)?;
        if let Some((_, stopped)) = attention_timing {
            stopped.record(stream)?;
        }
        #[cfg(not(feature = "qwen35-grouped-gqa-online-softmax"))]
        self.output_quantize_op.submit_for_plan(stream)?;
        self.output_args.submit_for_plan(stream)?;
        self.attention_residual_op.submit_for_plan(stream)?;
        self.post_norm_op.submit_for_plan(stream)?;
        self.ffn.submit(stream)?;
        self.final_residual_op.submit_for_plan(stream)?;
        Ok(())
    }
}

impl DevicePlan {
    fn reset(&mut self, stream: &OwnedCudaStream) -> Result<(), Qwen35MixedPrefixCudaError> {
        match self {
            Self::Linear(plan) => plan.reset(stream),
            Self::Full(plan) => plan.reset(stream),
        }
    }

    fn upload(
        &self,
        stream: &OwnedCudaStream,
        projected: Option<&ProjectedInput>,
        hidden: &[f32],
        device_projection: bool,
    ) -> Result<(), Qwen35MixedPrefixCudaError> {
        match self {
            Self::Linear(plan) => match projected {
                Some(ProjectedInput::Linear(projected)) => {
                    plan.upload(stream, Some(projected), hidden, device_projection)
                }
                None if device_projection => plan.upload(stream, None, hidden, true),
                _ => Err(invalid("linear layer projection is missing")),
            },
            Self::Full(plan) => {
                // Native full-attention projection consumes only the hidden
                // input; no CPU projected vector is uploaded.
                let _ = device_projection;
                plan.upload(stream, hidden)
            }
        }
    }

    fn final_hidden_view(&self) -> Result<CudaBufferView, Qwen35MixedPrefixCudaError> {
        match self {
            Self::Linear(plan) => Ok(plan.final_hidden.view(0, plan.final_hidden.len_bytes())?),
            Self::Full(plan) => Ok(plan.final_hidden.view(0, plan.final_hidden.len_bytes())?),
        }
    }

    fn final_hidden_f32_view(&self) -> Result<CudaF32View, Qwen35MixedPrefixCudaError> {
        Ok(CudaF32View::new(
            self.final_hidden_view()?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            linear::HIDDEN,
        )?)
    }

    fn final_hidden_buffer(&self) -> &OwnedCudaBuffer {
        match self {
            Self::Linear(plan) => &plan.final_hidden,
            Self::Full(plan) => &plan.final_hidden,
        }
    }

    #[cfg(feature = "qwen35-device-token-handoff")]
    fn input_f32_view(&self) -> Result<CudaF32View, Qwen35MixedPrefixCudaError> {
        match self {
            Self::Linear(plan) => Ok(plan.input_view),
            Self::Full(plan) => Ok(plan.input_view),
        }
    }

    fn copy_input_from_device(
        &self,
        stream: &OwnedCudaStream,
        source: CudaBufferView,
    ) -> Result<bool, Qwen35MixedPrefixCudaError> {
        match self {
            Self::Linear(plan) => {
                if plan.input_is_alias {
                    if plan.input_view.buffer_view() != source {
                        return Err(invalid("linear input alias source drift"));
                    }
                    return Ok(false);
                }
                // SAFETY: source is an earlier plan's final hidden allocation;
                // both plans and the stream remain live until the covering
                // completion proof in `execute`.
                unsafe {
                    plan.input
                        .copy_from_device_async_unchecked(stream, 0, source)?;
                }
            }
            Self::Full(plan) => {
                if plan.input_is_alias {
                    if plan.input_view.buffer_view() != source {
                        return Err(invalid("full-attention input alias source drift"));
                    }
                    return Ok(false);
                }
                // SAFETY: same lifetime and same-stream ordering proof as the
                // linear destination above.
                unsafe {
                    plan.input
                        .copy_from_device_async_unchecked(stream, 0, source)?;
                }
            }
        }
        Ok(true)
    }

    fn execute(
        &self,
        stream: &OwnedCudaStream,
        position: usize,
    ) -> Result<DeviceLayerReadback, Qwen35MixedPrefixCudaError> {
        match self {
            Self::Linear(plan) => plan.execute(stream),
            Self::Full(plan) => plan.execute(stream, position),
        }
    }

    fn submit_resident(
        &self,
        stream: &OwnedCudaStream,
        position: usize,
    ) -> Result<(), Qwen35MixedPrefixCudaError> {
        match self {
            Self::Linear(plan) => plan.submit_resident(stream),
            Self::Full(plan) => plan.submit_resident(stream, position),
        }
    }

    fn submit_resident_with_attention_timing(
        &self,
        stream: &OwnedCudaStream,
        position: usize,
        attention_timing: Option<(&OwnedCudaEvent, &OwnedCudaEvent)>,
    ) -> Result<(), Qwen35MixedPrefixCudaError> {
        match self {
            Self::Linear(plan) => {
                if attention_timing.is_some() {
                    return Err(invalid(
                        "linear layer received full-attention timing events",
                    ));
                }
                plan.submit_resident(stream)
            }
            Self::Full(plan) => {
                plan.submit_resident_with_attention_timing(stream, position, attention_timing)
            }
        }
    }

    fn state_sha256(
        &self,
        stream: &OwnedCudaStream,
        receipt: CudaStreamSubmissionReceipt,
    ) -> Result<String, Qwen35MixedPrefixCudaError> {
        let mut digest = Sha256::new();
        match self {
            Self::Linear(plan) => {
                update_device_buffer_digest(&mut digest, stream, &plan.history, receipt)?;
                update_device_buffer_digest(&mut digest, stream, &plan.state, receipt)?;
            }
            Self::Full(plan) => {
                update_device_buffer_digest(&mut digest, stream, &plan.key_cache, receipt)?;
                update_device_buffer_digest(&mut digest, stream, &plan.value_cache, receipt)?;
            }
        }
        Ok(format!("{:x}", digest.finalize()))
    }

    #[cfg(all(
        feature = "qwen35-device-token-handoff",
        feature = "qwen35-kv8-full-attention"
    ))]
    fn terminal_state_buffers(&self) -> [&OwnedCudaBuffer; 2] {
        match self {
            Self::Linear(plan) => [&plan.history, &plan.state],
            Self::Full(plan) => [&plan.key_cache, &plan.value_cache],
        }
    }
}

#[cfg(all(
    feature = "qwen35-device-token-handoff",
    feature = "qwen35-kv8-full-attention"
))]
fn submit_terminal_state_receipt(
    plans: &[DevicePlan],
    stream: &OwnedCudaStream,
    _prior_receipt: CudaStreamSubmissionReceipt,
) -> Result<TerminalStateReceiptReport, Qwen35MixedPrefixCudaError> {
    if plans.is_empty() {
        return Err(invalid("terminal state receipt requires plans"));
    }
    let component_count = plans
        .len()
        .checked_mul(2)
        .ok_or_else(|| invalid("terminal state receipt component overflow"))?;
    let payload_bytes = component_count
        .checked_mul(QWEN35_KV8_PAGE_DIGEST_BYTES)
        .ok_or_else(|| invalid("terminal state receipt payload overflow"))?;
    let output = OwnedCudaBuffer::allocate(stream.device_ordinal(), payload_bytes)?;
    let aggregate =
        OwnedCudaBuffer::allocate(stream.device_ordinal(), QWEN35_KV8_PAGE_DIGEST_BYTES)?;
    let mut component = 0_usize;
    for plan in plans {
        for state in plan.terminal_state_buffers() {
            let offset = component
                .checked_mul(QWEN35_KV8_PAGE_DIGEST_BYTES)
                .ok_or_else(|| invalid("terminal state receipt offset overflow"))?;
            let view = output.view(offset, QWEN35_KV8_PAGE_DIGEST_BYTES)?;
            submit_page_digest_into_view_v1(stream, state, view)?;
            component += 1;
        }
    }
    submit_page_digest_v1(stream, &output, &aggregate)?;
    let digest_receipt = stream.submission_receipt()?;
    let mut bytes = vec![0_u8; QWEN35_KV8_PAGE_DIGEST_BYTES];
    aggregate.copy_to_host_sync_with_submission_proof(stream, &mut bytes, digest_receipt)?;
    Ok(TerminalStateReceiptReport {
        algorithm: "qwen35-page-digest-v1-per-state-plane".to_owned(),
        component_count,
        payload_bytes,
        d2h_bytes: QWEN35_KV8_PAGE_DIGEST_BYTES,
        digest_sha256: sha256_hex(&bytes),
        state_readback_avoided: true,
    })
}

#[cfg(all(
    feature = "qwen35-device-token-handoff",
    feature = "qwen35-kv8-full-attention"
))]
#[cfg(all(
    feature = "qwen35-batched-prefill-handoff",
    feature = "qwen35-kv8-full-attention"
))]
#[derive(Clone, Copy, Debug)]
struct Kv8RamPersistenceGeometry {
    capacity_positions: usize,
    populated_positions: usize,
    plane_bytes: usize,
    page_bytes: usize,
    pages_per_plane: usize,
    plane_count: usize,
    page_count: usize,
    total_payload_bytes: usize,
}

#[cfg(all(
    feature = "qwen35-batched-prefill-handoff",
    feature = "qwen35-kv8-full-attention"
))]
impl Kv8RamPersistenceGeometry {
    fn bind(
        plans: &[DevicePlan],
        capacity_positions: usize,
        populated_positions: usize,
    ) -> Result<(Self, Vec<usize>), Qwen35MixedPrefixCudaError> {
        if capacity_positions == 0
            || populated_positions == 0
            || populated_positions > capacity_positions
            || !capacity_positions.is_multiple_of(KV8_RAM_PERSISTENCE_PAGE_POSITIONS)
        {
            return Err(invalid("KV8 RAM persistence position geometry"));
        }
        let full_attention_layer_indices = plans
            .iter()
            .enumerate()
            .filter_map(|(layer, plan)| matches!(plan, DevicePlan::Full(_)).then_some(layer))
            .collect::<Vec<_>>();
        let expected_layers = (3..FULL_TARGET_LAYER_COUNT).step_by(4).collect::<Vec<_>>();
        if full_attention_layer_indices != expected_layers {
            return Err(invalid("KV8 RAM persistence full-attention layer set"));
        }
        let plane_count = full_attention_layer_indices
            .len()
            .checked_mul(2)
            .ok_or_else(|| invalid("KV8 RAM persistence plane count overflow"))?;
        let plane_bytes = capacity_positions
            .checked_mul(QWEN35_KV8_CACHE_ROW_BYTES)
            .ok_or_else(|| invalid("KV8 RAM persistence plane byte overflow"))?;
        let page_bytes = KV8_RAM_PERSISTENCE_PAGE_POSITIONS
            .checked_mul(QWEN35_KV8_CACHE_ROW_BYTES)
            .ok_or_else(|| invalid("KV8 RAM persistence page byte overflow"))?;
        let pages_per_plane = capacity_positions / KV8_RAM_PERSISTENCE_PAGE_POSITIONS;
        let page_count = plane_count
            .checked_mul(pages_per_plane)
            .ok_or_else(|| invalid("KV8 RAM persistence page count overflow"))?;
        let total_payload_bytes = plane_count
            .checked_mul(plane_bytes)
            .ok_or_else(|| invalid("KV8 RAM persistence total byte overflow"))?;
        if plane_count != 32
            || page_count == 0
            || !page_count.is_multiple_of(KV8_RAM_PERSISTENCE_LANES)
            || page_count.checked_mul(page_bytes) != Some(total_payload_bytes)
        {
            return Err(invalid("KV8 RAM persistence tiled geometry"));
        }
        for plan in plans {
            if let DevicePlan::Full(plan) = plan {
                if plan.key_cache.len_bytes() != plane_bytes
                    || plan.value_cache.len_bytes() != plane_bytes
                {
                    return Err(invalid("KV8 RAM persistence cache allocation extent"));
                }
            }
        }
        Ok((
            Self {
                capacity_positions,
                populated_positions,
                plane_bytes,
                page_bytes,
                pages_per_plane,
                plane_count,
                page_count,
                total_payload_bytes,
            },
            full_attention_layer_indices,
        ))
    }

    fn snapshot_range(
        self,
        page_index: usize,
    ) -> Result<std::ops::Range<usize>, Qwen35MixedPrefixCudaError> {
        if page_index >= self.page_count {
            return Err(invalid("KV8 RAM persistence page index"));
        }
        let start = page_index
            .checked_mul(self.page_bytes)
            .ok_or_else(|| invalid("KV8 RAM persistence page offset overflow"))?;
        Ok(start..start + self.page_bytes)
    }
}

#[cfg(all(
    feature = "qwen35-batched-prefill-handoff",
    feature = "qwen35-kv8-full-attention"
))]
struct Kv8CachePlane<'a> {
    buffer: &'a OwnedCudaBuffer,
}

#[cfg(all(
    feature = "qwen35-batched-prefill-handoff",
    feature = "qwen35-kv8-full-attention"
))]
fn kv8_cache_planes(plans: &[DevicePlan]) -> Vec<Kv8CachePlane<'_>> {
    let mut planes = Vec::with_capacity(32);
    for plan in plans {
        if let DevicePlan::Full(plan) = plan {
            planes.push(Kv8CachePlane {
                buffer: &plan.key_cache,
            });
            planes.push(Kv8CachePlane {
                buffer: &plan.value_cache,
            });
        }
    }
    planes
}

#[cfg(all(
    feature = "qwen35-batched-prefill-handoff",
    feature = "qwen35-kv8-full-attention"
))]
fn kv8_persistence_identity(
    plan_digest: [u8; 32],
    generation: u64,
    geometry: Kv8RamPersistenceGeometry,
) -> (String, String) {
    let mut digest = Sha256::new();
    digest.update(b"crystal-quant.qwen35.kv8-ram-persistence-identity.v1\0");
    digest.update(plan_digest);
    digest.update(generation.to_le_bytes());
    digest.update((geometry.capacity_positions as u64).to_le_bytes());
    digest.update((geometry.populated_positions as u64).to_le_bytes());
    digest.update((geometry.plane_count as u64).to_le_bytes());
    digest.update((geometry.total_payload_bytes as u64).to_le_bytes());
    let identity = format!("{:x}", digest.finalize());
    let session_id = format!("qwen35-kv8-{}", &identity[..16]);
    (session_id, identity)
}

#[cfg(all(
    feature = "qwen35-batched-prefill-handoff",
    feature = "qwen35-kv8-full-attention"
))]
fn kv8_page_ledger_sha256(
    identity_sha256: &str,
    full_attention_layer_indices: &[usize],
    geometry: Kv8RamPersistenceGeometry,
    snapshot: &[u8],
) -> Result<String, Qwen35MixedPrefixCudaError> {
    if snapshot.len() != geometry.total_payload_bytes {
        return Err(invalid("KV8 RAM persistence ledger snapshot extent"));
    }
    let mut digest = Sha256::new();
    digest.update(b"crystal-quant.qwen35.kv8-ram-persistence-page-ledger.v1\0");
    digest.update(identity_sha256.as_bytes());
    for page_index in 0..geometry.page_count {
        let plane_index = page_index / geometry.pages_per_plane;
        let layer_slot = plane_index / 2;
        let model_layer = *full_attention_layer_indices
            .get(layer_slot)
            .ok_or_else(|| invalid("KV8 RAM persistence ledger layer slot"))?;
        let page_in_plane = page_index % geometry.pages_per_plane;
        digest.update((page_index as u64).to_le_bytes());
        digest.update((plane_index as u64).to_le_bytes());
        digest.update((model_layer as u64).to_le_bytes());
        digest.update([u8::from(plane_index % 2 == 1)]);
        digest.update((page_in_plane as u64).to_le_bytes());
        let range = geometry.snapshot_range(page_index)?;
        digest.update(Sha256::digest(&snapshot[range]));
    }
    Ok(format!("{:x}", digest.finalize()))
}

#[cfg(all(
    feature = "qwen35-batched-prefill-handoff",
    feature = "qwen35-kv8-full-attention"
))]
fn read_kv8_cache_pages(
    plans: &[DevicePlan],
    streams: &[OwnedCudaStream],
    geometry: Kv8RamPersistenceGeometry,
    mut consume: impl FnMut(usize, &[u8]) -> Result<(), Qwen35MixedPrefixCudaError>,
) -> Result<(f64, BoundaryReport), Qwen35MixedPrefixCudaError> {
    if streams.len() != KV8_RAM_PERSISTENCE_LANES {
        return Err(invalid("KV8 RAM persistence D2H lane count"));
    }
    let planes = kv8_cache_planes(plans);
    if planes.len() != geometry.plane_count {
        return Err(invalid("KV8 RAM persistence D2H plane count"));
    }
    let mut lanes = (0..streams.len())
        .map(|_| OwnedPinnedD2hBuffer::allocate(streams[0].device_ordinal(), geometry.page_bytes))
        .collect::<Result<Vec<_>, _>>()?;
    let mut pending_pages = vec![None::<usize>; lanes.len()];
    let boundary_before = cuda_host_boundary_snapshot();
    let started = Instant::now();
    for page_index in 0..geometry.page_count {
        let lane_index = page_index % lanes.len();
        if let Some(previous_page) = pending_pages[lane_index].take() {
            consume(previous_page, lanes[lane_index].wait_and_read()?)?;
        }
        let plane_index = page_index / geometry.pages_per_plane;
        let page_in_plane = page_index % geometry.pages_per_plane;
        let source = planes[plane_index]
            .buffer
            .view(page_in_plane * geometry.page_bytes, geometry.page_bytes)?;
        // SAFETY: every cache allocation remains live and immutable until its
        // lane's event-backed wait drains the exact submitted page.
        unsafe {
            lanes[lane_index].submit_from_view(&streams[lane_index], source)?;
        }
        pending_pages[lane_index] = Some(page_index);
    }
    for lane_index in 0..lanes.len() {
        if let Some(previous_page) = pending_pages[lane_index].take() {
            consume(previous_page, lanes[lane_index].wait_and_read()?)?;
        }
    }
    let wall_ms = started.elapsed().as_secs_f64() * 1_000.0;
    drop(lanes);
    Ok((
        wall_ms,
        boundary_delta(boundary_before, cuda_host_boundary_snapshot())?,
    ))
}

#[cfg(all(
    feature = "qwen35-batched-prefill-handoff",
    feature = "qwen35-kv8-full-attention"
))]
fn export_kv8_cache_snapshot(
    plans: &[DevicePlan],
    streams: &[OwnedCudaStream],
    geometry: Kv8RamPersistenceGeometry,
) -> Result<(Vec<u8>, f64, BoundaryReport), Qwen35MixedPrefixCudaError> {
    let mut snapshot = vec![0_u8; geometry.total_payload_bytes];
    let (wall_ms, boundaries) =
        read_kv8_cache_pages(plans, streams, geometry, |page_index, bytes| {
            let range = geometry.snapshot_range(page_index)?;
            snapshot[range].copy_from_slice(bytes);
            Ok(())
        })?;
    Ok((snapshot, wall_ms, boundaries))
}

#[cfg(all(
    feature = "qwen35-batched-prefill-handoff",
    feature = "qwen35-kv8-full-attention"
))]
fn verify_kv8_cache_snapshot(
    plans: &[DevicePlan],
    streams: &[OwnedCudaStream],
    geometry: Kv8RamPersistenceGeometry,
    expected: &[u8],
) -> Result<(String, usize, f64, BoundaryReport), Qwen35MixedPrefixCudaError> {
    if expected.len() != geometry.total_payload_bytes {
        return Err(invalid("KV8 RAM persistence verify extent"));
    }
    let mut digest = Sha256::new();
    let mut mismatched_pages = 0_usize;
    let mut next_page = 0_usize;
    let (wall_ms, boundaries) =
        read_kv8_cache_pages(plans, streams, geometry, |page_index, bytes| {
            if page_index != next_page {
                return Err(invalid("KV8 RAM persistence verify page order"));
            }
            let range = geometry.snapshot_range(page_index)?;
            if bytes != &expected[range] {
                mismatched_pages = mismatched_pages.saturating_add(1);
            }
            digest.update(bytes);
            next_page = next_page.saturating_add(1);
            Ok(())
        })?;
    if next_page != geometry.page_count {
        return Err(invalid("KV8 RAM persistence verify page count"));
    }
    Ok((
        format!("{:x}", digest.finalize()),
        mismatched_pages,
        wall_ms,
        boundaries,
    ))
}

#[cfg(all(
    feature = "qwen35-batched-prefill-handoff",
    feature = "qwen35-kv8-full-attention"
))]
fn clear_kv8_caches(
    plans: &mut [DevicePlan],
    stream: &OwnedCudaStream,
) -> Result<(f64, BoundaryReport), Qwen35MixedPrefixCudaError> {
    let boundary_before = cuda_host_boundary_snapshot();
    let started = Instant::now();
    for plan in plans {
        if let DevicePlan::Full(plan) = plan {
            plan.key_cache.zero_sync(stream)?;
            plan.value_cache.zero_sync(stream)?;
        }
    }
    Ok((
        started.elapsed().as_secs_f64() * 1_000.0,
        boundary_delta(boundary_before, cuda_host_boundary_snapshot())?,
    ))
}

#[cfg(all(
    feature = "qwen35-batched-prefill-handoff",
    feature = "qwen35-kv8-full-attention"
))]
fn sample_zeroed_kv8_caches(
    plans: &[DevicePlan],
    stream: &OwnedCudaStream,
    geometry: Kv8RamPersistenceGeometry,
) -> Result<(usize, u64, f64, BoundaryReport), Qwen35MixedPrefixCudaError> {
    let planes = kv8_cache_planes(plans);
    let mut lane = OwnedPinnedD2hBuffer::allocate(stream.device_ordinal(), geometry.page_bytes)?;
    let boundary_before = cuda_host_boundary_snapshot();
    let started = Instant::now();
    let mut sampled_pages = 0_usize;
    let mut nonzero_bytes = 0_u64;
    for plane in planes {
        for page_in_plane in [0, geometry.pages_per_plane - 1] {
            let source = plane
                .buffer
                .view(page_in_plane * geometry.page_bytes, geometry.page_bytes)?;
            // SAFETY: this cache page remains live and immutable through the
            // immediate event-backed wait before the next sample is submitted.
            unsafe {
                lane.submit_from_view(stream, source)?;
            }
            nonzero_bytes = nonzero_bytes.saturating_add(
                lane.wait_and_read()?
                    .iter()
                    .map(|byte| u64::from(*byte != 0))
                    .sum::<u64>(),
            );
            sampled_pages = sampled_pages.saturating_add(1);
        }
    }
    let wall_ms = started.elapsed().as_secs_f64() * 1_000.0;
    drop(lane);
    Ok((
        sampled_pages,
        nonzero_bytes,
        wall_ms,
        boundary_delta(boundary_before, cuda_host_boundary_snapshot())?,
    ))
}

#[cfg(all(
    feature = "qwen35-batched-prefill-handoff",
    feature = "qwen35-kv8-full-attention"
))]
fn restore_kv8_cache_snapshot(
    plans: &[DevicePlan],
    streams: &[OwnedCudaStream],
    geometry: Kv8RamPersistenceGeometry,
    snapshot: &[u8],
) -> Result<(f64, BoundaryReport), Qwen35MixedPrefixCudaError> {
    if streams.len() != KV8_RAM_PERSISTENCE_LANES || snapshot.len() != geometry.total_payload_bytes
    {
        return Err(invalid("KV8 RAM persistence H2D input extent"));
    }
    let planes = kv8_cache_planes(plans);
    if planes.len() != geometry.plane_count {
        return Err(invalid("KV8 RAM persistence H2D plane count"));
    }
    let boundary_before = cuda_host_boundary_snapshot();
    let started = Instant::now();
    let mut lanes = streams
        .iter()
        .map(|stream| {
            OwnedPinnedH2dBuffer::allocate_host_only(stream.device_ordinal(), geometry.page_bytes)
        })
        .collect::<Result<Vec<_>, _>>()?;
    for global_page in 0..geometry.page_count {
        let lane_index = global_page % lanes.len();
        let plane_index = global_page / geometry.pages_per_plane;
        let page_in_plane = global_page % geometry.pages_per_plane;
        let range = geometry.snapshot_range(global_page)?;
        lanes[lane_index]
            .writable()?
            .copy_from_slice(&snapshot[range]);
        let destination = planes[plane_index]
            .buffer
            .view(page_in_plane * geometry.page_bytes, geometry.page_bytes)?;
        // SAFETY: `plane.buffer` remains live and its page is not reused until
        // this lane's completion event is waited below.
        unsafe {
            lanes[lane_index].submit_to_view(&streams[lane_index], destination)?;
        }
    }
    for lane in &mut lanes {
        lane.wait()?;
    }
    Ok((
        started.elapsed().as_secs_f64() * 1_000.0,
        boundary_delta(boundary_before, cuda_host_boundary_snapshot())?,
    ))
}

#[cfg(all(
    feature = "qwen35-batched-prefill-handoff",
    feature = "qwen35-kv8-full-attention"
))]
fn linear_persistent_buffers(
    plans: &[DevicePlan],
) -> Result<(Vec<&OwnedCudaBuffer>, Vec<&OwnedCudaBuffer>), Qwen35MixedPrefixCudaError> {
    let mut history = Vec::new();
    let mut state = Vec::new();
    for plan in plans {
        if let DevicePlan::Linear(plan) = plan {
            history.push(&plan.history);
            state.push(&plan.state);
        }
    }
    if history.len() != 48 || state.len() != 48 {
        return Err(invalid("Qwen35 linear persistent buffer count"));
    }
    let history_bytes = history[0].len_bytes();
    let state_bytes = state[0].len_bytes();
    if history_bytes == 0
        || state_bytes == 0
        || history
            .iter()
            .any(|buffer| buffer.len_bytes() != history_bytes)
        || state.iter().any(|buffer| buffer.len_bytes() != state_bytes)
    {
        return Err(invalid("Qwen35 linear persistent buffer geometry"));
    }
    Ok((history, state))
}

#[cfg(all(
    feature = "qwen35-batched-prefill-handoff",
    feature = "qwen35-kv8-full-attention"
))]
fn read_linear_buffer_group(
    buffers: &[&OwnedCudaBuffer],
    streams: &[OwnedCudaStream],
    mut consume: impl FnMut(usize, &[u8]) -> Result<(), Qwen35MixedPrefixCudaError>,
) -> Result<(f64, BoundaryReport), Qwen35MixedPrefixCudaError> {
    if buffers.is_empty()
        || streams.len() != KV8_RAM_PERSISTENCE_LANES
        || !buffers.len().is_multiple_of(streams.len())
    {
        return Err(invalid("Qwen35 linear persistent D2H group extent"));
    }
    let buffer_bytes = buffers[0].len_bytes();
    if buffer_bytes == 0
        || buffers
            .iter()
            .any(|buffer| buffer.len_bytes() != buffer_bytes)
    {
        return Err(invalid("Qwen35 linear persistent D2H buffer extent"));
    }
    let mut lanes = (0..streams.len())
        .map(|_| OwnedPinnedD2hBuffer::allocate(streams[0].device_ordinal(), buffer_bytes))
        .collect::<Result<Vec<_>, _>>()?;
    let mut pending = vec![None::<usize>; lanes.len()];
    let boundary_before = cuda_host_boundary_snapshot();
    let started = Instant::now();
    for buffer_index in 0..buffers.len() {
        let lane_index = buffer_index % lanes.len();
        if let Some(previous) = pending[lane_index].take() {
            consume(previous, lanes[lane_index].wait_and_read()?)?;
        }
        let source = buffers[buffer_index].view(0, buffer_bytes)?;
        // SAFETY: each source buffer remains live and immutable until its
        // lane's receipt-backed wait drains the submitted copy.
        unsafe {
            lanes[lane_index].submit_from_view(&streams[lane_index], source)?;
        }
        pending[lane_index] = Some(buffer_index);
    }
    for lane_index in 0..lanes.len() {
        if let Some(previous) = pending[lane_index].take() {
            consume(previous, lanes[lane_index].wait_and_read()?)?;
        }
    }
    Ok((
        started.elapsed().as_secs_f64() * 1_000.0,
        boundary_delta(boundary_before, cuda_host_boundary_snapshot())?,
    ))
}

#[cfg(all(
    feature = "qwen35-batched-prefill-handoff",
    feature = "qwen35-kv8-full-attention"
))]
fn export_linear_state_snapshot(
    plans: &[DevicePlan],
    streams: &[OwnedCudaStream],
) -> Result<(Vec<u8>, usize, usize, usize, usize, f64, BoundaryReport), Qwen35MixedPrefixCudaError>
{
    let (history, state) = linear_persistent_buffers(plans)?;
    let history_bytes = history[0].len_bytes();
    let state_bytes = state[0].len_bytes();
    let history_total = history
        .len()
        .checked_mul(history_bytes)
        .ok_or_else(|| invalid("Qwen35 linear history snapshot overflow"))?;
    let state_total = state
        .len()
        .checked_mul(state_bytes)
        .ok_or_else(|| invalid("Qwen35 linear state snapshot overflow"))?;
    let total = history_total
        .checked_add(state_total)
        .ok_or_else(|| invalid("Qwen35 linear snapshot total overflow"))?;
    let mut snapshot = vec![0_u8; total];
    let (history_wall, history_boundaries) =
        read_linear_buffer_group(&history, streams, |index, bytes| {
            let start = index * history_bytes;
            snapshot[start..start + history_bytes].copy_from_slice(bytes);
            Ok(())
        })?;
    let (state_wall, state_boundaries) =
        read_linear_buffer_group(&state, streams, |index, bytes| {
            let start = history_total + index * state_bytes;
            snapshot[start..start + state_bytes].copy_from_slice(bytes);
            Ok(())
        })?;
    Ok((
        snapshot,
        history.len(),
        state.len(),
        history_bytes,
        state_bytes,
        history_wall + state_wall,
        add_boundaries(history_boundaries, state_boundaries),
    ))
}

#[cfg(all(
    feature = "qwen35-batched-prefill-handoff",
    feature = "qwen35-kv8-full-attention"
))]
fn verify_linear_state_snapshot(
    plans: &[DevicePlan],
    streams: &[OwnedCudaStream],
    expected: &[u8],
) -> Result<(String, usize, f64, BoundaryReport), Qwen35MixedPrefixCudaError> {
    let (history, state) = linear_persistent_buffers(plans)?;
    let history_bytes = history[0].len_bytes();
    let state_bytes = state[0].len_bytes();
    let history_total = history.len() * history_bytes;
    let total = history_total + state.len() * state_bytes;
    if expected.len() != total {
        return Err(invalid("Qwen35 linear snapshot verification extent"));
    }
    let mut digest = Sha256::new();
    let mut mismatched = 0_usize;
    let mut next = 0_usize;
    let (history_wall, history_boundaries) =
        read_linear_buffer_group(&history, streams, |index, bytes| {
            if index != next {
                return Err(invalid("Qwen35 linear history verification order"));
            }
            let start = index * history_bytes;
            if bytes != &expected[start..start + history_bytes] {
                mismatched = mismatched.saturating_add(1);
            }
            digest.update(bytes);
            next += 1;
            Ok(())
        })?;
    let mut state_next = 0_usize;
    let (state_wall, state_boundaries) =
        read_linear_buffer_group(&state, streams, |index, bytes| {
            if index != state_next {
                return Err(invalid("Qwen35 linear state verification order"));
            }
            let start = history_total + index * state_bytes;
            if bytes != &expected[start..start + state_bytes] {
                mismatched = mismatched.saturating_add(1);
            }
            digest.update(bytes);
            state_next += 1;
            Ok(())
        })?;
    if next != history.len() || state_next != state.len() {
        return Err(invalid("Qwen35 linear snapshot verification count"));
    }
    Ok((
        format!("{:x}", digest.finalize()),
        mismatched,
        history_wall + state_wall,
        add_boundaries(history_boundaries, state_boundaries),
    ))
}

#[cfg(all(
    feature = "qwen35-batched-prefill-handoff",
    feature = "qwen35-kv8-full-attention"
))]
fn clear_linear_persistent_buffers(
    plans: &mut [DevicePlan],
    stream: &OwnedCudaStream,
) -> Result<(f64, BoundaryReport), Qwen35MixedPrefixCudaError> {
    let boundary_before = cuda_host_boundary_snapshot();
    let started = Instant::now();
    let mut count = 0_usize;
    for plan in plans {
        if let DevicePlan::Linear(plan) = plan {
            plan.history.zero_sync(stream)?;
            plan.state.zero_sync(stream)?;
            count += 1;
        }
    }
    if count != 48 {
        return Err(invalid("Qwen35 linear persistent clear count"));
    }
    Ok((
        started.elapsed().as_secs_f64() * 1_000.0,
        boundary_delta(boundary_before, cuda_host_boundary_snapshot())?,
    ))
}

#[cfg(all(
    feature = "qwen35-batched-prefill-handoff",
    feature = "qwen35-kv8-full-attention"
))]
fn sample_zeroed_linear_buffers(
    plans: &[DevicePlan],
    streams: &[OwnedCudaStream],
) -> Result<(usize, u64, f64, BoundaryReport), Qwen35MixedPrefixCudaError> {
    let (history, state) = linear_persistent_buffers(plans)?;
    let mut sampled = 0_usize;
    let mut nonzero = 0_u64;
    let (history_wall, history_boundaries) =
        read_linear_buffer_group(&history, streams, |_index, bytes| {
            nonzero =
                nonzero.saturating_add(bytes.iter().map(|byte| u64::from(*byte != 0)).sum::<u64>());
            sampled += 1;
            Ok(())
        })?;
    let (state_wall, state_boundaries) =
        read_linear_buffer_group(&state, streams, |_index, bytes| {
            nonzero =
                nonzero.saturating_add(bytes.iter().map(|byte| u64::from(*byte != 0)).sum::<u64>());
            sampled += 1;
            Ok(())
        })?;
    Ok((
        sampled,
        nonzero,
        history_wall + state_wall,
        add_boundaries(history_boundaries, state_boundaries),
    ))
}

#[cfg(all(
    feature = "qwen35-batched-prefill-handoff",
    feature = "qwen35-kv8-full-attention"
))]
fn restore_linear_state_snapshot(
    plans: &[DevicePlan],
    streams: &[OwnedCudaStream],
    snapshot: &[u8],
) -> Result<(f64, BoundaryReport), Qwen35MixedPrefixCudaError> {
    let (history, state) = linear_persistent_buffers(plans)?;
    let history_bytes = history[0].len_bytes();
    let state_bytes = state[0].len_bytes();
    let history_total = history.len() * history_bytes;
    if snapshot.len() != history_total + state.len() * state_bytes {
        return Err(invalid("Qwen35 linear snapshot restore extent"));
    }
    let boundary_before = cuda_host_boundary_snapshot();
    let started = Instant::now();
    for (buffers, bytes, base) in [
        (&history, history_bytes, 0_usize),
        (&state, state_bytes, history_total),
    ] {
        let mut lanes = streams
            .iter()
            .map(|stream| OwnedPinnedH2dBuffer::allocate_host_only(stream.device_ordinal(), bytes))
            .collect::<Result<Vec<_>, _>>()?;
        for index in 0..buffers.len() {
            let lane_index = index % lanes.len();
            let start = base + index * bytes;
            lanes[lane_index]
                .writable()?
                .copy_from_slice(&snapshot[start..start + bytes]);
            let destination = buffers[index].view(0, bytes)?;
            // SAFETY: the linear buffer remains live and unchanged until the
            // lane's completion event is waited below.
            unsafe {
                lanes[lane_index].submit_to_view(&streams[lane_index], destination)?;
            }
        }
        for lane in &mut lanes {
            lane.wait()?;
        }
    }
    Ok((
        started.elapsed().as_secs_f64() * 1_000.0,
        boundary_delta(boundary_before, cuda_host_boundary_snapshot())?,
    ))
}

#[cfg(all(
    feature = "qwen35-batched-prefill-handoff",
    feature = "qwen35-kv8-full-attention"
))]
fn run_linear_state_persistence_roundtrip(
    plans: &mut [DevicePlan],
    compute_stream: &OwnedCudaStream,
    streams: &[OwnedCudaStream],
) -> Result<LinearStatePersistenceReport, Qwen35MixedPrefixCudaError> {
    if streams.len() != KV8_RAM_PERSISTENCE_LANES
        || streams
            .iter()
            .any(|stream| stream.device_ordinal() != compute_stream.device_ordinal())
    {
        return Err(invalid("Qwen35 linear persistent transfer lane geometry"));
    }
    let total_boundary_before = cuda_host_boundary_snapshot();
    let allocation_before = cuda_device_allocation_snapshot();
    let total_started = Instant::now();
    let (
        source,
        history_count,
        state_count,
        history_bytes,
        state_bytes,
        export_host_wall_ms,
        export_boundaries,
    ) = export_linear_state_snapshot(plans, streams)?;
    let source_snapshot_sha256 = sha256_hex(&source);
    let (clear_host_wall_ms, clear_boundaries) =
        clear_linear_persistent_buffers(plans, compute_stream)?;
    let (
        zero_sampled_buffers,
        zero_sample_nonzero_bytes,
        zero_sample_host_wall_ms,
        zero_sample_boundaries,
    ) = sample_zeroed_linear_buffers(plans, streams)?;
    let (restore_host_wall_ms, restore_boundaries) =
        restore_linear_state_snapshot(plans, streams, &source)?;
    let (
        restored_snapshot_sha256,
        restored_mismatched_buffers,
        verify_host_wall_ms,
        verify_boundaries,
    ) = verify_linear_state_snapshot(plans, streams, &source)?;
    let total_host_wall_ms = total_started.elapsed().as_secs_f64() * 1_000.0;
    let total_boundaries = boundary_delta(total_boundary_before, cuda_host_boundary_snapshot())?;
    let accounted_boundaries = [
        export_boundaries.clone(),
        clear_boundaries.clone(),
        zero_sample_boundaries.clone(),
        restore_boundaries.clone(),
        verify_boundaries.clone(),
    ]
    .into_iter()
    .fold(BoundaryReport::default(), add_boundaries);
    let allocation_after = cuda_device_allocation_snapshot();
    let history_total = history_count
        .checked_mul(history_bytes)
        .ok_or_else(|| invalid("Qwen35 linear history payload overflow"))?;
    let state_total = state_count
        .checked_mul(state_bytes)
        .ok_or_else(|| invalid("Qwen35 linear state payload overflow"))?;
    let expected_total = history_total
        .checked_add(state_total)
        .ok_or_else(|| invalid("Qwen35 linear payload total overflow"))?;
    let source_export_complete = source.len() == expected_total;
    let clear_sample_zero =
        zero_sampled_buffers == history_count + state_count && zero_sample_nonzero_bytes == 0;
    let restore_byte_exact =
        restored_mismatched_buffers == 0 && restored_snapshot_sha256 == source_snapshot_sha256;
    let transfer_copy_path_async = [
        &export_boundaries,
        &zero_sample_boundaries,
        &restore_boundaries,
        &verify_boundaries,
    ]
    .iter()
    .all(|boundary| {
        boundary.synchronous_h2d_copies == 0
            && boundary.synchronous_h2d_bytes == 0
            && boundary.synchronous_d2h_copies == 0
            && boundary.synchronous_d2h_bytes == 0
    });
    let boundary_ledger_closed = total_boundaries == accounted_boundaries;
    let allocation_cleanup_zero = allocation_after.live_allocations
        == allocation_before.live_allocations
        && allocation_after.live_bytes == allocation_before.live_bytes;
    let bytes_gb = source.len() as f64 / 1_000_000_000.0;
    let geometry_exact = history_count == 48
        && state_count == 48
        && history_bytes > 0
        && state_bytes > 0
        && source.len() == expected_total;
    let complete_numerical_contract = geometry_exact
        && source_export_complete
        && clear_sample_zero
        && restore_byte_exact
        && transfer_copy_path_async
        && boundary_ledger_closed
        && allocation_cleanup_zero;
    Ok(LinearStatePersistenceReport {
        scope: "real_48_layer_deltanet_history_and_state_export_clear_restore_verify".to_owned(),
        history_buffer_count: history_count,
        state_buffer_count: state_count,
        buffer_count: history_count + state_count,
        history_buffer_bytes: history_bytes,
        state_buffer_bytes: state_bytes,
        total_payload_bytes: source.len(),
        source_snapshot_sha256,
        restored_snapshot_sha256,
        restored_mismatched_buffers,
        zero_sampled_buffers,
        zero_sample_nonzero_bytes,
        export_host_wall_ms,
        clear_host_wall_ms,
        zero_sample_host_wall_ms,
        restore_host_wall_ms,
        verify_host_wall_ms,
        total_host_wall_ms,
        export_gb_per_second: bytes_gb * 1_000.0 / export_host_wall_ms,
        restore_gb_per_second: bytes_gb * 1_000.0 / restore_host_wall_ms,
        export_boundaries,
        clear_boundaries,
        zero_sample_boundaries,
        restore_boundaries,
        verify_boundaries,
        total_boundaries,
        boundary_ledger_closed,
        allocation_cleanup_zero,
        geometry_exact,
        source_export_complete,
        clear_sample_zero,
        restore_byte_exact,
        transfer_copy_path_async,
        complete_numerical_contract,
    })
}

#[cfg(all(
    feature = "qwen35-batched-prefill-handoff",
    feature = "qwen35-kv8-full-attention"
))]
fn run_kv8_ram_persistence_roundtrip(
    plans: &mut [DevicePlan],
    compute_stream: &OwnedCudaStream,
    plan_digest: [u8; 32],
    generation: u64,
    capacity_positions: usize,
    populated_positions: usize,
) -> Result<Kv8RamPersistenceReport, Qwen35MixedPrefixCudaError> {
    let (geometry, full_attention_layer_indices) =
        Kv8RamPersistenceGeometry::bind(plans, capacity_positions, populated_positions)?;
    let (session_id, identity_sha256) = kv8_persistence_identity(plan_digest, generation, geometry);
    let total_boundary_before = cuda_host_boundary_snapshot();
    let allocation_before = cuda_device_allocation_snapshot();
    let total_started = Instant::now();
    let mut streams = (0..KV8_RAM_PERSISTENCE_LANES)
        .map(|_| OwnedCudaStream::create(compute_stream.device_ordinal()))
        .collect::<Result<Vec<_>, _>>()?;
    let owner_keys = streams
        .iter()
        .map(OwnedCudaStream::owner_instance_key)
        .collect::<Result<Vec<_>, _>>()?;
    let linear_state_persistence =
        run_linear_state_persistence_roundtrip(plans, compute_stream, &streams)?;

    let (snapshot, export_host_wall_ms, export_boundaries) =
        export_kv8_cache_snapshot(plans, &streams, geometry)?;
    let source_snapshot_sha256 = sha256_hex(&snapshot);
    let page_ledger_sha256 = kv8_page_ledger_sha256(
        &identity_sha256,
        &full_attention_layer_indices,
        geometry,
        &snapshot,
    )?;
    let (clear_host_wall_ms, clear_boundaries) = clear_kv8_caches(plans, compute_stream)?;
    let (
        zero_sampled_pages,
        zero_sample_nonzero_bytes,
        zero_sample_host_wall_ms,
        zero_sample_boundaries,
    ) = sample_zeroed_kv8_caches(plans, &streams[0], geometry)?;
    let (restore_host_wall_ms, restore_boundaries) =
        restore_kv8_cache_snapshot(plans, &streams, geometry, &snapshot)?;
    let (
        restored_snapshot_sha256,
        restored_mismatched_pages,
        verify_host_wall_ms,
        verify_boundaries,
    ) = verify_kv8_cache_snapshot(plans, &streams, geometry, &snapshot)?;

    let stream_close_before = cuda_host_boundary_snapshot();
    let mut stream_owners = Vec::with_capacity(streams.len());
    for (lane, stream) in streams.iter_mut().enumerate() {
        let receipt = stream.close_with_receipt()?;
        stream_owners.push(Kv8RamPersistenceStreamOwnerReport {
            lane,
            owner_instance_key: owner_keys[lane],
            close_epoch: receipt.close_epoch(),
            close_receipt_sha256: hex_bytes(&receipt.receipt_sha256()),
        });
    }
    let stream_close_boundaries =
        boundary_delta(stream_close_before, cuda_host_boundary_snapshot())?;
    let total_host_wall_ms = total_started.elapsed().as_secs_f64() * 1_000.0;
    let total_boundaries = boundary_delta(total_boundary_before, cuda_host_boundary_snapshot())?;
    let accounted_boundaries = [
        linear_state_persistence.total_boundaries.clone(),
        export_boundaries.clone(),
        clear_boundaries.clone(),
        zero_sample_boundaries.clone(),
        restore_boundaries.clone(),
        verify_boundaries.clone(),
        stream_close_boundaries.clone(),
    ]
    .into_iter()
    .fold(BoundaryReport::default(), add_boundaries);
    let allocation_after = cuda_device_allocation_snapshot();
    let mut unique_owner_keys = owner_keys;
    unique_owner_keys.sort_unstable();
    unique_owner_keys.dedup();
    let stream_owner_identity_closed = unique_owner_keys.len() == KV8_RAM_PERSISTENCE_LANES
        && stream_owners.len() == KV8_RAM_PERSISTENCE_LANES
        && stream_owners
            .iter()
            .all(|owner| owner.close_epoch != 0 && owner.close_receipt_sha256.len() == 64);
    let source_export_complete = snapshot.len() == geometry.total_payload_bytes;
    let clear_sample_zero =
        zero_sampled_pages == geometry.plane_count * 2 && zero_sample_nonzero_bytes == 0;
    let restore_byte_exact =
        restored_mismatched_pages == 0 && restored_snapshot_sha256 == source_snapshot_sha256;
    let transfer_copy_path_async = [
        &linear_state_persistence.export_boundaries,
        &linear_state_persistence.zero_sample_boundaries,
        &linear_state_persistence.restore_boundaries,
        &linear_state_persistence.verify_boundaries,
        &export_boundaries,
        &zero_sample_boundaries,
        &restore_boundaries,
        &verify_boundaries,
    ]
    .iter()
    .all(|boundary| {
        boundary.synchronous_h2d_copies == 0
            && boundary.synchronous_h2d_bytes == 0
            && boundary.synchronous_d2h_copies == 0
            && boundary.synchronous_d2h_bytes == 0
    });
    let boundary_ledger_closed = total_boundaries == accounted_boundaries;
    let allocation_cleanup_zero = allocation_after.live_allocations
        == allocation_before.live_allocations
        && allocation_after.live_bytes == allocation_before.live_bytes;
    let bytes_gb = geometry.total_payload_bytes as f64 / 1_000_000_000.0;

    Ok(Kv8RamPersistenceReport {
        scope: "real_16_layer_q8_0_kv_cache_export_clear_restore_verify_before_suffix".to_owned(),
        session_id,
        generation,
        identity_sha256,
        page_ledger_sha256,
        capacity_positions: geometry.capacity_positions,
        populated_positions: geometry.populated_positions,
        full_attention_layer_indices,
        plane_count: geometry.plane_count,
        row_bytes: QWEN35_KV8_CACHE_ROW_BYTES,
        plane_bytes: geometry.plane_bytes,
        page_positions: KV8_RAM_PERSISTENCE_PAGE_POSITIONS,
        page_bytes: geometry.page_bytes,
        pages_per_plane: geometry.pages_per_plane,
        page_count: geometry.page_count,
        lane_count: KV8_RAM_PERSISTENCE_LANES,
        total_payload_bytes: geometry.total_payload_bytes,
        source_snapshot_sha256,
        restored_snapshot_sha256,
        restored_mismatched_pages,
        zero_sampled_pages,
        zero_sample_nonzero_bytes,
        export_host_wall_ms,
        clear_host_wall_ms,
        zero_sample_host_wall_ms,
        restore_host_wall_ms,
        verify_host_wall_ms,
        total_host_wall_ms,
        export_gb_per_second: bytes_gb * 1_000.0 / export_host_wall_ms,
        restore_gb_per_second: bytes_gb * 1_000.0 / restore_host_wall_ms,
        stream_owners,
        linear_state_persistence: Some(linear_state_persistence),
        export_boundaries,
        clear_boundaries,
        zero_sample_boundaries,
        restore_boundaries,
        verify_boundaries,
        stream_close_boundaries,
        total_boundaries,
        boundary_ledger_closed,
        allocation_cleanup_zero,
        q8_cache_geometry_exact: true,
        source_export_complete,
        clear_sample_zero,
        restore_byte_exact,
        transfer_copy_path_async,
        stream_owner_identity_closed,
        first_suffix_token_matches_control: false,
        first_suffix_top2_bits_match_control: false,
        final_hidden_matches_control: false,
        final_state_matches_control: false,
        complete_numerical_contract: false,
        persistence_wall_excluded_from_decode_timing: true,
        real_kv8_cache_roundtrip_executed: true,
        inactive_session_product_ready: false,
        warm_ttft_qualified: false,
        product_tps_permitted: false,
        default_cli_enabled: false,
    })
}

struct DeviceLayerReadback {
    input_hidden: Vec<f32>,
    ffn_intermediates: Option<FfnIntermediateReadback>,
    attention_intermediates: Option<AttentionIntermediateReadback>,
    full_attention_intermediates: Option<FullAttentionIntermediateReadback>,
    attention_output: Vec<f32>,
    attention_residual: Vec<f32>,
    post_norm: Vec<f32>,
    ffn_output: Vec<f32>,
    final_hidden: Vec<f32>,
    state: Vec<f32>,
    history: Vec<f32>,
    polls: u64,
    proof: bool,
    completion_boundaries: BoundaryReport,
    readback_boundaries: BoundaryReport,
}

struct DeviceSequence {
    layers: Vec<Vec<DeviceLayerReadback>>,
    host_wall_ms: f64,
    layer_host_wall_ms: Vec<f64>,
    completion_boundaries: BoundaryReport,
    readback_boundaries: BoundaryReport,
    polls: u64,
    proof: bool,
    inter_layer_d2d_copy_count: usize,
    inter_layer_d2d_bytes: usize,
    host_projection_token_count: usize,
    host_hidden_handoff_token_count: usize,
}

struct ResidentFastSequence {
    token_host_wall_ms: Vec<f64>,
    token_backbone_cuda_event_ms: Vec<f64>,
    token_output_head_cuda_event_ms: Vec<f64>,
    token_cuda_event_ms: Vec<f64>,
    token_layer_cuda_event_ms: Vec<Vec<f64>>,
    final_hidden: Vec<Vec<f32>>,
    final_norm: Vec<Vec<f32>>,
    logits: Vec<Vec<f32>>,
    input_boundaries: BoundaryReport,
    hot_path_boundaries: BoundaryReport,
    readback_boundaries: BoundaryReport,
    polls: u64,
    proof: bool,
    inter_layer_d2d_copy_count: usize,
    inter_layer_d2d_bytes: usize,
}

struct AutoregressiveForward {
    receipt: CudaStreamSubmissionReceipt,
    cuda_event_ms: f64,
    polls: u64,
    proof: bool,
    inter_layer_d2d_copy_count: usize,
    inter_layer_d2d_bytes: usize,
}

struct PendingAutoregressiveToken {
    generation_index: usize,
    position: usize,
    input_token_id: usize,
    output_token_id: usize,
    host_wall_ms: f64,
    cuda_event_ms: f64,
    logits: Vec<f32>,
}

struct AutoregressiveSequence {
    prompt_prefix_tokens_processed: usize,
    prefill_host_wall_ms: f64,
    batched_prefill: Option<AutoregressiveBatchedPrefillReport>,
    decode_host_wall_ms: f64,
    tokens: Vec<AutoregressiveTokenReport>,
    generated_token_ids: Vec<usize>,
    #[cfg(feature = "qwen35-device-token-handoff")]
    token_top2: Vec<CudaQwen35Top2V1>,
    final_hidden_sha256: String,
    final_state_sha256: Vec<String>,
    resident_auxiliary: Option<ResidentAuxiliaryReport>,
    #[cfg(feature = "qwen35-device-token-handoff")]
    terminal_state_receipt_digest: Option<String>,
    execution_boundaries: BoundaryReport,
    state_readback_boundaries: BoundaryReport,
    polls: u64,
    proof: bool,
    inter_layer_d2d_copy_count: usize,
    inter_layer_d2d_bytes: usize,
}

enum AutoregressivePrefillMode {
    Sequential,
    #[cfg(feature = "qwen35-batched-prefill-handoff")]
    Batched(AutoregressiveBatchedPrefill),
}

impl AutoregressivePrefillMode {
    /// Release token-sized prefill storage as soon as its submission has
    /// completed. The decode owner only needs layer plans and KV state; it
    /// must not retain the chunk schedule, rope tables, or host input slab.
    fn release_workspace(&mut self) {
        let mode = std::mem::replace(self, Self::Sequential);
        drop(mode);
    }

    fn release_workspace_with_report(
        &mut self,
        run: &mut AutoregressivePrefillRun,
    ) -> Result<(), Qwen35MixedPrefixCudaError> {
        if !run.proof {
            return Err(invalid("prefill release requires terminal owner proof"));
        }
        let before = cuda_device_allocation_snapshot();
        let started = Instant::now();
        self.release_workspace();
        let release_ms = started.elapsed().as_secs_f64() * 1_000.0;
        let after = cuda_device_allocation_snapshot();
        if let Some(report) = run.batched_report.as_mut() {
            // Paired experimental runners can own both modes concurrently;
            // report their different global baseline without asserting it.
            // The single-owner autoregressive runner gates this result below.
            let closed = prefill_release_closed(report, before, after);
            report.workspace_release_host_wall_ms = release_ms;
            report.workspace_released_bytes = before
                .live_bytes
                .checked_sub(after.live_bytes)
                .ok_or_else(|| invalid("prefill release byte count increased"))?;
            report.workspace_released_allocations = before
                .live_allocations
                .checked_sub(after.live_allocations)
                .ok_or_else(|| invalid("prefill release allocation count increased"))?;
            report.live_bytes_after_release = after.live_bytes;
            report.workspace_released_before_auxiliary_and_decode = closed;
        }
        Ok(())
    }
}

fn prefill_release_closed(
    report: &AutoregressiveBatchedPrefillReport,
    before: augur_backend::CudaDeviceAllocationSnapshot,
    after: augur_backend::CudaDeviceAllocationSnapshot,
) -> bool {
    report.execution_allocation_stable
        && before.live_bytes == report.live_bytes_before_release
        && before.live_allocations == report.live_allocations_before_release
        && after.live_bytes == report.resident_live_bytes_before_build
        && after.live_allocations == report.resident_live_allocations_before_build
        && before.live_bytes > after.live_bytes
        && before.live_allocations > after.live_allocations
        && after.successful_allocations == before.successful_allocations
        && after.successful_frees.checked_sub(before.successful_frees)
            == before.live_allocations.checked_sub(after.live_allocations)
}

#[cfg(feature = "qwen35-batched-prefill-handoff")]
struct AutoregressiveBatchedPrefill {
    workspace: PrefillWorkspace,
    schedule: PrefillChunkSchedule,
    input_hidden: Vec<f32>,
    tensor_core_prefix_attention: bool,
    direct_activation: bool,
    simt_shared_softmax: bool,
    resident_before_build: augur_backend::CudaDeviceAllocationSnapshot,
}

struct AutoregressivePrefillRun {
    host_wall_ms: f64,
    batched_report: Option<AutoregressiveBatchedPrefillReport>,
    polls: u64,
    proof: bool,
    inter_layer_d2d_copy_count: usize,
    inter_layer_d2d_bytes: usize,
}

#[cfg(feature = "qwen35-device-token-handoff")]
struct DeviceTokenHandoffPlan {
    embedding: OwnedCudaBuffer,
    partials: OwnedCudaBuffer,
    receipts: OwnedCudaBuffer,
    embedding_payload_bytes: usize,
    embedding_payload_sha256: String,
    plan_digest: [u8; 32],
    max_tokens: usize,
}

#[cfg(feature = "qwen35-device-token-handoff")]
struct DeviceTokenHandoffSequence {
    prefill_host_wall_ms: f64,
    batched_prefill: Option<AutoregressiveBatchedPrefillReport>,
    decode_host_wall_ms: f64,
    cuda_event_ms: f64,
    cuda_profiler_capture_requested: bool,
    layer_timing: Option<DeviceTokenHandoffLayerTimingReport>,
    generated_token_ids: Vec<usize>,
    receipts: Vec<DeviceTokenHandoffReceiptReport>,
    raw_receipt_sha256: String,
    final_hidden_sha256: String,
    final_state_sha256: Vec<String>,
    resident_auxiliary: Option<ResidentAuxiliaryReport>,
    prefill_boundaries: BoundaryReport,
    kv8_ram_persistence: Option<Kv8RamPersistenceReport>,
    execution_boundaries: BoundaryReport,
    receipt_readback_boundaries: BoundaryReport,
    state_readback_boundaries: BoundaryReport,
    terminal_state_receipt: Option<TerminalStateReceiptReport>,
    polls: u64,
    proof: bool,
    inter_layer_d2d_copy_count: usize,
    inter_layer_d2d_bytes: usize,
}

#[cfg(feature = "qwen35-device-token-handoff")]
struct DeviceTokenHandoffLayerTimingEvents {
    token_started: Vec<OwnedCudaEvent>,
    token_stopped: Vec<OwnedCudaEvent>,
    layer_stopped: Vec<Vec<OwnedCudaEvent>>,
    full_attention_component: Vec<Vec<Option<(OwnedCudaEvent, OwnedCudaEvent)>>>,
}

#[cfg(feature = "qwen35-device-token-handoff")]
impl DeviceTokenHandoffLayerTimingEvents {
    fn new(
        device_ordinal: i32,
        plans: &[DevicePlan],
        generation_tokens: usize,
    ) -> Result<Self, Qwen35MixedPrefixCudaError> {
        let make_events = |count: usize| {
            (0..count)
                .map(|_| OwnedCudaEvent::create(device_ordinal))
                .collect::<Result<Vec<_>, _>>()
        };
        let token_started = make_events(generation_tokens)?;
        let token_stopped = make_events(generation_tokens)?;
        let layer_stopped = (0..generation_tokens)
            .map(|_| make_events(plans.len()))
            .collect::<Result<Vec<_>, _>>()?;
        let full_attention_component = (0..generation_tokens)
            .map(|_| {
                plans
                    .iter()
                    .map(|plan| match plan {
                        DevicePlan::Linear(_) => Ok(None),
                        DevicePlan::Full(_) => Ok(Some((
                            OwnedCudaEvent::create(device_ordinal)?,
                            OwnedCudaEvent::create(device_ordinal)?,
                        ))),
                    })
                    .collect::<Result<Vec<_>, CudaRuntimeError>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            token_started,
            token_stopped,
            layer_stopped,
            full_attention_component,
        })
    }

    fn attention_events(
        &self,
        token: usize,
        layer: usize,
    ) -> Option<(&OwnedCudaEvent, &OwnedCudaEvent)> {
        self.full_attention_component[token][layer]
            .as_ref()
            .map(|(started, stopped)| (started, stopped))
    }

    fn report(
        &self,
        plans: &[DevicePlan],
    ) -> Result<DeviceTokenHandoffLayerTimingReport, Qwen35MixedPrefixCudaError> {
        let full_attention_layer_indices = plans
            .iter()
            .enumerate()
            .filter_map(|(layer, plan)| matches!(plan, DevicePlan::Full(_)).then_some(layer))
            .collect::<Vec<_>>();
        let linear_layers_per_token = plans.len() - full_attention_layer_indices.len();
        let mut token_instrumented_cuda_event_ms = Vec::with_capacity(self.token_started.len());
        let mut token_layer_cuda_event_ms = Vec::with_capacity(self.token_started.len());
        let mut token_full_attention_component_cuda_event_ms =
            Vec::with_capacity(self.token_started.len());
        let mut token_post_backbone_cuda_event_ms = Vec::with_capacity(self.token_started.len());
        let mut aggregate_layer_cuda_event_ms = 0.0_f64;
        let mut aggregate_linear_layer_cuda_event_ms = 0.0_f64;
        let mut aggregate_full_layer_cuda_event_ms = 0.0_f64;
        let mut aggregate_full_attention_component_cuda_event_ms = 0.0_f64;
        let mut aggregate_post_backbone_cuda_event_ms = 0.0_f64;

        for token in 0..self.token_started.len() {
            let token_ms =
                f64::from(self.token_started[token].elapsed_ms(&self.token_stopped[token])?);
            if !token_ms.is_finite() || token_ms <= 0.0 {
                return Err(invalid("device-handoff instrumented token timing"));
            }
            token_instrumented_cuda_event_ms.push(token_ms);

            let mut layer_times = Vec::with_capacity(plans.len());
            for (layer, plan) in plans.iter().enumerate() {
                let started = if layer == 0 {
                    &self.token_started[token]
                } else {
                    &self.layer_stopped[token][layer - 1]
                };
                let elapsed_ms = f64::from(started.elapsed_ms(&self.layer_stopped[token][layer])?);
                if !elapsed_ms.is_finite() || elapsed_ms <= 0.0 {
                    return Err(invalid("device-handoff instrumented layer timing"));
                }
                aggregate_layer_cuda_event_ms += elapsed_ms;
                match plan {
                    DevicePlan::Linear(_) => aggregate_linear_layer_cuda_event_ms += elapsed_ms,
                    DevicePlan::Full(_) => aggregate_full_layer_cuda_event_ms += elapsed_ms,
                }
                layer_times.push(elapsed_ms);
            }
            token_layer_cuda_event_ms.push(layer_times);

            let mut full_attention_times = Vec::with_capacity(full_attention_layer_indices.len());
            for layer in &full_attention_layer_indices {
                let (started, stopped) = self.full_attention_component[token][*layer]
                    .as_ref()
                    .ok_or_else(|| invalid("device-handoff full-attention timing event"))?;
                let elapsed_ms = f64::from(started.elapsed_ms(stopped)?);
                if !elapsed_ms.is_finite() || elapsed_ms <= 0.0 {
                    return Err(invalid(
                        "device-handoff instrumented full-attention component timing",
                    ));
                }
                aggregate_full_attention_component_cuda_event_ms += elapsed_ms;
                full_attention_times.push(elapsed_ms);
            }
            token_full_attention_component_cuda_event_ms.push(full_attention_times);

            let last_layer = self.layer_stopped[token]
                .last()
                .ok_or_else(|| invalid("device-handoff instrumented layer extent"))?;
            let post_backbone_ms = f64::from(last_layer.elapsed_ms(&self.token_stopped[token])?);
            if !post_backbone_ms.is_finite() || post_backbone_ms <= 0.0 {
                return Err(invalid("device-handoff instrumented post-backbone timing"));
            }
            aggregate_post_backbone_cuda_event_ms += post_backbone_ms;
            token_post_backbone_cuda_event_ms.push(post_backbone_ms);
        }

        let aggregate_instrumented_cuda_event_ms =
            token_instrumented_cuda_event_ms.iter().sum::<f64>();
        let accounted_ms = aggregate_layer_cuda_event_ms + aggregate_post_backbone_cuda_event_ms;
        let accounting_relative_error = (accounted_ms - aggregate_instrumented_cuda_event_ms).abs()
            / aggregate_instrumented_cuda_event_ms.max(f64::EPSILON);
        if !accounting_relative_error.is_finite() || accounting_relative_error > 1.0e-3 {
            return Err(invalid("device-handoff instrumented timing conservation"));
        }

        Ok(DeviceTokenHandoffLayerTimingReport {
            scope:
                "diagnostic_same_stream_layer_boundaries_and_full_attention_kv_store_plus_kernel"
                    .to_owned(),
            generation_tokens: self.token_started.len(),
            layers_per_token: plans.len(),
            linear_layers_per_token,
            full_attention_layers_per_token: full_attention_layer_indices.len(),
            full_attention_layer_indices,
            event_records_per_token: 2 + plans.len() + 2 * (plans.len() - linear_layers_per_token),
            token_instrumented_cuda_event_ms,
            token_layer_cuda_event_ms,
            token_full_attention_component_cuda_event_ms,
            token_post_backbone_cuda_event_ms,
            aggregate_instrumented_cuda_event_ms,
            aggregate_layer_cuda_event_ms,
            aggregate_linear_layer_cuda_event_ms,
            aggregate_full_layer_cuda_event_ms,
            aggregate_full_attention_component_cuda_event_ms,
            aggregate_post_backbone_cuda_event_ms,
            accounting_relative_error,
            measurement_perturbs_tps: true,
            product_tps_permitted: false,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn build_autoregressive_prefill_mode(
    enabled: bool,
    stream: &OwnedCudaStream,
    plans: &[DevicePlan],
    reader: &mut GgufTensorReader<'_>,
    embedding: &GgufTensor,
    prompt_token_ids: &[usize],
    model_config: &Qwen35Config,
    capacity: usize,
    shared_softmax: bool,
) -> Result<AutoregressivePrefillMode, Qwen35MixedPrefixCudaError> {
    build_autoregressive_prefill_mode_for_policy(
        enabled,
        stream,
        plans,
        reader,
        embedding,
        prompt_token_ids,
        model_config,
        capacity,
        512,
        false,
        false,
        shared_softmax,
        false,
        false,
        false,
        false,
        2048,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_autoregressive_prefill_mode_for_policy(
    enabled: bool,
    stream: &OwnedCudaStream,
    plans: &[DevicePlan],
    reader: &mut GgufTensorReader<'_>,
    embedding: &GgufTensor,
    prompt_token_ids: &[usize],
    model_config: &Qwen35Config,
    capacity: usize,
    requested_chunk: usize,
    tensor_core_prefix_attention: bool,
    direct_activation: bool,
    simt_shared_softmax: bool,
    simt_shared_gdn_scalars: bool,
    simt_unit_previous_scale: bool,
    quad_weight_decode: bool,
    wide_prefill_microbatch: bool,
    wide_prefill_microbatch_limit: usize,
) -> Result<AutoregressivePrefillMode, Qwen35MixedPrefixCudaError> {
    build_prefill_mode_from_embeddings(
        enabled,
        stream,
        None,
        plans,
        &mut |token| Ok(linear::read_embedding_row(reader, embedding, token)?),
        prompt_token_ids,
        model_config,
        capacity,
        requested_chunk,
        tensor_core_prefix_attention,
        direct_activation,
        simt_shared_softmax,
        simt_shared_gdn_scalars,
        simt_unit_previous_scale,
        quad_weight_decode,
        wide_prefill_microbatch,
        wide_prefill_microbatch_limit,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_prefill_mode_from_embeddings(
    enabled: bool,
    stream: &OwnedCudaStream,
    retained_stream: Option<Rc<OwnedCudaStream>>,
    plans: &[DevicePlan],
    read_embedding: &mut dyn FnMut(usize) -> Result<Vec<f32>, Qwen35MixedPrefixCudaError>,
    prompt_token_ids: &[usize],
    model_config: &Qwen35Config,
    capacity: usize,
    requested_chunk: usize,
    tensor_core_prefix_attention: bool,
    direct_activation: bool,
    simt_shared_softmax: bool,
    simt_shared_gdn_scalars: bool,
    simt_unit_previous_scale: bool,
    quad_weight_decode: bool,
    wide_prefill_microbatch: bool,
    wide_prefill_microbatch_limit: usize,
) -> Result<AutoregressivePrefillMode, Qwen35MixedPrefixCudaError> {
    if wide_prefill_microbatch
        && (!false
            || !matches!(wide_prefill_microbatch_limit, 2048 | 4096))
    {
        return Err(invalid("autoregressive wide microbatch feature/limit"));
    }
    if !wide_prefill_microbatch && wide_prefill_microbatch_limit != 2048 {
        return Err(invalid(
            "autoregressive wide limit requires explicit opt-in",
        ));
    }
    if tensor_core_prefix_attention
        && (simt_shared_softmax
            || simt_shared_gdn_scalars
            || simt_unit_previous_scale
            || quad_weight_decode
            || wide_prefill_microbatch)
    {
        return Err(invalid("TC autoregressive mode cannot use SIMT tuning"));
    }
    if simt_unit_previous_scale && !simt_shared_softmax {
        return Err(invalid("unit scale requires shared softmax"));
    }
    let allowed_chunk = if tensor_core_prefix_attention {
        1024
    } else if wide_prefill_microbatch {
        wide_prefill_microbatch_limit
    } else {
        512
    };
    if requested_chunk == 0
        || requested_chunk > allowed_chunk
        || (direct_activation && !tensor_core_prefix_attention)
        || (tensor_core_prefix_attention && !false)
    {
        return Err(invalid("autoregressive prefill policy bounds/feature"));
    }
    if !enabled || prompt_token_ids.len() <= 1 {
        return Ok(AutoregressivePrefillMode::Sequential);
    }

    #[cfg(not(feature = "qwen35-batched-prefill-handoff"))]
    {
        let _ = (stream, plans, read_embedding, model_config, capacity);
        return Err(invalid(
            "qwen35-batched-prefill-handoff feature is required",
        ));
    }

    #[cfg(feature = "qwen35-batched-prefill-handoff")]
    {
        let prefix_tokens = prompt_token_ids.len() - 1;
        if prefix_tokens > capacity || prefix_tokens > model_config.max_position_embeddings {
            return Err(invalid("autoregressive batched prefill position extent"));
        }
        let mut input_hidden = Vec::with_capacity(
            prefix_tokens
                .checked_mul(linear::HIDDEN)
                .ok_or_else(|| invalid("autoregressive batched prefill input extent"))?,
        );
        for token_id in &prompt_token_ids[..prefix_tokens] {
            input_hidden.extend(read_embedding(*token_id)?);
        }
        let resident_before_build = cuda_device_allocation_snapshot();
        let chunk_tokens = requested_chunk.min(prefix_tokens);
        let mut workspace = PrefillWorkspace::new_with_rope_capacity(stream, chunk_tokens, capacity)?;
        if plans.iter().any(|p| match p {
            DevicePlan::Linear(p) => p.ffn_is_qpn, DevicePlan::Full(p) => p.ffn_is_qpn,
        }) {
            let retained = retained_stream.ok_or_else(|| invalid("QPN prefill requires retained stream"))?;
            if retained.owner_instance_key()? != stream.owner_instance_key()? {
                return Err(invalid("QPN prefill sequencing stream mismatch"));
            }
            workspace.enable_qpn(retained)?;
        }
        workspace.upload_rope_tables(stream, prefix_tokens, 0, model_config.rope_theta as f32)?;
        let mut schedule = build_prefill_chunk_schedule(
            &workspace,
            plans,
            stream,
            prefix_tokens,
            chunk_tokens,
            model_config.rms_norm_eps as f32,
        )?;
        #[cfg(feature = "qwen35-kv8-chunk-prefill")]
        if !tensor_core_prefix_attention {
            for chunk in &mut schedule.chunks {
                for operation in &mut chunk.operations {
                    match operation {
                        PrefillLayerOps::Full(full) => {
                            full.simt_shared_softmax = simt_shared_softmax;
                            full.simt_unit_previous_scale = simt_unit_previous_scale;
                        }
                        PrefillLayerOps::Linear(linear) => {
                            linear.simt_shared_gdn_scalars = simt_shared_gdn_scalars;
                        }
                        
                    }
                    if quad_weight_decode {
                        match operation {
                            PrefillLayerOps::Linear(p) => {
                                let ffn = p.ffn.legacy_mut()?;
for projection in [
                                    &mut p.qkv,
                                    &mut p.z,
                                    &mut p.alpha,
                                    &mut p.beta,
                                    &mut p.output,
                                    &mut ffn.ffn_gate,
                                    &mut ffn.ffn_up,
                                    &mut ffn.ffn_down,
                                ] {
                                    projection.enable_quad_weight_decode();
                                }
                            }
                            PrefillLayerOps::Full(p) => {
                                let ffn = p.ffn.legacy_mut()?;
for projection in [
                                    &mut p.q,
                                    &mut p.k,
                                    &mut p.v,
                                    &mut p.output,
                                    &mut ffn.ffn_gate,
                                    &mut ffn.ffn_up,
                                    &mut ffn.ffn_down,
                                ] {
                                    projection.enable_quad_weight_decode();
                                }
                            }
                            
                        }
                    }
                }
            }
        }
        Ok(AutoregressivePrefillMode::Batched(
            AutoregressiveBatchedPrefill {
                workspace,
                schedule,
                input_hidden,
                tensor_core_prefix_attention,
                direct_activation,
                simt_shared_softmax,
                resident_before_build,
            },
        ))
    }
}

fn submit_autoregressive_token(
    plans: &[DevicePlan],
    output_head: Option<&DeviceOutputHeadPlan>,
    hidden: &[f32],
    stream: &OwnedCudaStream,
    position: usize,
    started_event: &OwnedCudaEvent,
    stopped_event: &OwnedCudaEvent,
    terminal_event: &OwnedCudaEvent,
) -> Result<AutoregressiveForward, Qwen35MixedPrefixCudaError> {
    if plans.len() != FULL_TARGET_LAYER_COUNT || hidden.len() != linear::HIDDEN {
        return Err(invalid("autoregressive full-target geometry"));
    }

    plans[0].upload(stream, None, hidden, true)?;
    started_event.record(stream)?;
    let mut inter_layer_d2d_copy_count = 0_usize;
    let mut inter_layer_d2d_bytes = 0_usize;
    for layer in 0..plans.len() {
        if layer > 0 {
            let (before, after) = plans.split_at(layer);
            let source = before[layer - 1].final_hidden_view()?;
            if after[0].copy_input_from_device(stream, source)? {
                inter_layer_d2d_copy_count = inter_layer_d2d_copy_count.saturating_add(1);
                inter_layer_d2d_bytes = inter_layer_d2d_bytes
                    .saturating_add(hidden.len().saturating_mul(core::mem::size_of::<f32>()));
            }
        }
        plans[layer].submit_resident(stream, position)?;
    }
    if let Some(output_head) = output_head {
        output_head.submit(stream)?;
    }
    stopped_event.record(stream)?;
    let receipt = stream.submission_receipt()?;
    terminal_event.record_for_submission(stream, receipt)?;

    let poll_limit = MAX_COMPLETION_POLLS.saturating_mul(plans.len() as u64);
    let mut polls = 0_u64;
    let completion = loop {
        if let Some(completion) = stream.query_event_completion(terminal_event, receipt)? {
            break completion;
        }
        polls = polls.saturating_add(1);
        if polls >= poll_limit {
            return Err(Qwen35MixedPrefixCudaError::CompletionPollLimit);
        }
        bounded_completion_poll_backoff(polls);
    };
    let proof = completion.covers_submission(receipt) && completion.belongs_to_stream(stream)?;
    if !proof {
        return Err(invalid("autoregressive owner epoch proof"));
    }
    let cuda_event_ms = f64::from(started_event.elapsed_ms(stopped_event)?);
    if !cuda_event_ms.is_finite() || cuda_event_ms <= 0.0 {
        return Err(invalid("autoregressive CUDA event timing"));
    }

    Ok(AutoregressiveForward {
        receipt,
        cuda_event_ms,
        polls,
        proof,
        inter_layer_d2d_copy_count,
        inter_layer_d2d_bytes,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_autoregressive_prefill(
    plans: &mut [DevicePlan],
    mode: &AutoregressivePrefillMode,
    reader: &mut GgufTensorReader<'_>,
    embedding: &GgufTensor,
    prompt_token_ids: &[usize],
    stream: &OwnedCudaStream,
    started_event: &OwnedCudaEvent,
    stopped_event: &OwnedCudaEvent,
    terminal_event: &OwnedCudaEvent,
) -> Result<AutoregressivePrefillRun, Qwen35MixedPrefixCudaError> {
    run_prefill_from_embeddings(
        plans,
        mode,
        &mut |token| Ok(linear::read_embedding_row(reader, embedding, token)?),
        prompt_token_ids,
        stream,
        started_event,
        stopped_event,
        terminal_event,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_prefill_from_embeddings(
    plans: &mut [DevicePlan],
    mode: &AutoregressivePrefillMode,
    read_embedding: &mut dyn FnMut(usize) -> Result<Vec<f32>, Qwen35MixedPrefixCudaError>,
    prompt_token_ids: &[usize],
    stream: &OwnedCudaStream,
    started_event: &OwnedCudaEvent,
    stopped_event: &OwnedCudaEvent,
    terminal_event: &OwnedCudaEvent,
) -> Result<AutoregressivePrefillRun, Qwen35MixedPrefixCudaError> {
    let prefill_started = Instant::now();
    match mode {
        AutoregressivePrefillMode::Sequential => {
            let mut polls = 0_u64;
            let mut proof = true;
            let mut inter_layer_d2d_copy_count = 0_usize;
            let mut inter_layer_d2d_bytes = 0_usize;
            for (position, token_id) in prompt_token_ids[..prompt_token_ids.len() - 1]
                .iter()
                .copied()
                .enumerate()
            {
                let hidden = read_embedding(token_id)?;
                let forward = submit_autoregressive_token(
                    plans,
                    None,
                    &hidden,
                    stream,
                    position,
                    started_event,
                    stopped_event,
                    terminal_event,
                )?;
                polls = polls.saturating_add(forward.polls);
                proof &= forward.proof;
                inter_layer_d2d_copy_count =
                    inter_layer_d2d_copy_count.saturating_add(forward.inter_layer_d2d_copy_count);
                inter_layer_d2d_bytes =
                    inter_layer_d2d_bytes.saturating_add(forward.inter_layer_d2d_bytes);
            }
            Ok(AutoregressivePrefillRun {
                host_wall_ms: prefill_started.elapsed().as_secs_f64() * 1_000.0,
                batched_report: None,
                polls,
                proof,
                inter_layer_d2d_copy_count,
                inter_layer_d2d_bytes,
            })
        }
        #[cfg(feature = "qwen35-batched-prefill-handoff")]
        AutoregressivePrefillMode::Batched(prefill) => {
            let allocations_before = cuda_device_allocation_snapshot();
            let boundaries_before = cuda_host_boundary_snapshot();
            let run = run_prefill_chunk_schedule(
                plans,
                &prefill.workspace,
                &prefill.schedule,
                prefill.tensor_core_prefix_attention,
                prefill.direct_activation,
                stream,
                None,
                started_event,
                stopped_event,
                terminal_event,
                &prefill.input_hidden,
                false,
                false,
            )?;
            let allocations_after = cuda_device_allocation_snapshot();
            let execution_allocations = allocation_delta(allocations_before, allocations_after)?;
            let execution_allocation_stable = execution_allocations.successful_allocations == 0
                && execution_allocations.successful_frees == 0
                && allocations_before.live_bytes == allocations_after.live_bytes
                && allocations_before.live_allocations == allocations_after.live_allocations;
            let batched_report = AutoregressiveBatchedPrefillReport {
                attention_path: if cfg!(feature = "qwen35-kv8-chunk-prefill") && {
                    #[cfg(feature = "qwen35-kv8-chunk-prefill")]
                    { prefill.workspace.dgemm_score_scratch.is_some() }
                    #[cfg(not(feature = "qwen35-kv8-chunk-prefill"))]
                    { false }
                } { "hybrid_fp64_dgemm_scores_ordered_fp32_v1".to_owned() } else { prefill_attention_path_label(
                    prefill.tensor_core_prefix_attention,
                    cfg!(feature = "qwen35-kv8-chunk-prefill"),
                    prefill.simt_shared_softmax,
                )
                .to_owned() },
                prefix_tokens: prompt_token_ids.len() - 1,
                chunk_tokens: prefill.schedule.chunk_tokens,
                chunk_count: prefill.schedule.chunks.len(),
                compute_host_wall_ms: run.host_wall_ms,
                cuda_event_ms: run.cuda_event_ms,
                execution_boundaries: boundary_delta(
                    boundaries_before,
                    cuda_host_boundary_snapshot(),
                )?,
                submission_boundaries: run.submission_boundaries,
                execution_allocations,
                execution_allocation_stable,
                resident_live_bytes_before_build: prefill.resident_before_build.live_bytes,
                resident_live_allocations_before_build: prefill
                    .resident_before_build
                    .live_allocations,
                live_bytes_before_release: allocations_after.live_bytes,
                live_allocations_before_release: allocations_after.live_allocations,
                workspace_release_host_wall_ms: 0.0,
                workspace_released_bytes: 0,
                workspace_released_allocations: 0,
                live_bytes_after_release: allocations_after.live_bytes,
                workspace_released_before_auxiliary_and_decode: false,
            };
            Ok(AutoregressivePrefillRun {
                host_wall_ms: run.host_wall_ms,
                batched_report: Some(batched_report),
                polls: run.completion_poll_iterations,
                proof: run.owner_epoch_proof,
                inter_layer_d2d_copy_count: 0,
                inter_layer_d2d_bytes: 0,
            })
        }
    }
}

fn run_autoregressive_sequence(
    plans: &mut [DevicePlan],
    output_head: &DeviceOutputHeadPlan,
    reader: &mut GgufTensorReader,
    embedding: &GgufTensor,
    prompt_token_ids: &[usize],
    generation_tokens: usize,
    stream: &OwnedCudaStream,
    capture_terminal_state: bool,
    prefill_mode: &mut AutoregressivePrefillMode,
    q8_dflash_path: Option<&std::path::Path>,
    vision_safetensors_path: Option<&std::path::Path>,
    #[cfg(feature = "qwen35-device-token-handoff")] terminal_state_receipt_only: bool,
    validate_q8_selector_terminal: bool,
) -> Result<AutoregressiveSequence, Qwen35MixedPrefixCudaError> {
    if plans.len() != FULL_TARGET_LAYER_COUNT
        || prompt_token_ids.is_empty()
        || generation_tokens == 0
    {
        return Err(invalid("autoregressive sequence extent"));
    }

    let started_event = OwnedCudaEvent::create(stream.device_ordinal())?;
    let stopped_event = OwnedCudaEvent::create(stream.device_ordinal())?;
    let terminal_event = OwnedCudaEvent::create(stream.device_ordinal())?;
    let execution_before = cuda_host_boundary_snapshot();
    let mut prefill = run_autoregressive_prefill(
        plans,
        prefill_mode,
        reader,
        embedding,
        prompt_token_ids,
        stream,
        &started_event,
        &stopped_event,
        &terminal_event,
    )?;
    prefill_mode.release_workspace_with_report(&mut prefill)?;
    let prefill_host_wall_ms = prefill.host_wall_ms;
    let mut proof = prefill.proof;
    let mut inter_layer_d2d_copy_count = prefill.inter_layer_d2d_copy_count;
    let mut inter_layer_d2d_bytes = prefill.inter_layer_d2d_bytes;
    let resident_auxiliary_before = cuda_device_allocation_snapshot();
    let mut resident_auxiliary_owners = ResidentAuxiliaryOwners::load(
        q8_dflash_path,
        vision_safetensors_path,
        stream.device_ordinal(),
        stream,
    )
    .map_err(invalid)?;
    let mut resident_auxiliary = resident_auxiliary_owners
        .as_ref()
        .map(ResidentAuxiliaryOwners::report);

    let mut pending_tokens = Vec::with_capacity(generation_tokens);
    let mut generated_token_ids = Vec::with_capacity(generation_tokens);
    let mut polls = 0_u64;
    let mut input_token_id = *prompt_token_ids
        .last()
        .ok_or_else(|| invalid("autoregressive prompt token"))?;
    let first_decode_position = prompt_token_ids.len() - 1;
    let decode_started = Instant::now();
    let mut final_receipt = None;
    for generation_index in 0..generation_tokens {
        let position = first_decode_position
            .checked_add(generation_index)
            .ok_or_else(|| invalid("autoregressive position overflow"))?;
        let token_started = Instant::now();
        let hidden = linear::read_embedding_row(reader, embedding, input_token_id)?;
        let forward = submit_autoregressive_token(
            plans,
            Some(output_head),
            &hidden,
            stream,
            position,
            &started_event,
            &stopped_event,
            &terminal_event,
        )?;
        if let Some(owners) = resident_auxiliary_owners.as_mut() {
            let hidden_view = CudaF32View::new(
                plans
                    .last()
                    .ok_or_else(|| invalid("autoregressive terminal layer"))?
                    .final_hidden_view()?,
                PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                linear::HIDDEN,
            )?;
            owners
                .consume_q8_device(generation_index, hidden_view, stream)
                .map_err(invalid)?;
        }
        let logits = read_f32_buffer(stream, &output_head.logits, forward.receipt)?;
        let output_token_id = usize::try_from(greedy_argmax(&logits, logits.len())?)
            .map_err(|_| invalid("autoregressive argmax conversion"))?;
        let host_wall_ms = token_started.elapsed().as_secs_f64() * 1_000.0;

        polls = polls.saturating_add(forward.polls);
        proof &= forward.proof;
        inter_layer_d2d_copy_count =
            inter_layer_d2d_copy_count.saturating_add(forward.inter_layer_d2d_copy_count);
        inter_layer_d2d_bytes = inter_layer_d2d_bytes.saturating_add(forward.inter_layer_d2d_bytes);
        generated_token_ids.push(output_token_id);
        pending_tokens.push(PendingAutoregressiveToken {
            generation_index,
            position,
            input_token_id,
            output_token_id,
            host_wall_ms,
            cuda_event_ms: forward.cuda_event_ms,
            logits,
        });
        input_token_id = output_token_id;
        final_receipt = Some(forward.receipt);
    }
    if let Some(owners) = resident_auxiliary_owners.as_mut() {
        // Finish the selector before taking the state-readback interval so its
        // terminal/output work is accounted in the execution interval.
        // The audited primary path keeps the full terminal CPU-oracle
        // readback.  It is intentionally separate from the receipt-only
        // handoff transport path below.
        owners.finish_consumers(stream, true).map_err(invalid)?;
        resident_auxiliary = Some(owners.report());
    }
    let decode_host_wall_ms = decode_started.elapsed().as_secs_f64() * 1_000.0;
    let execution_boundaries = boundary_delta(execution_before, cuda_host_boundary_snapshot())?;
    let state_readback_before = cuda_host_boundary_snapshot();
    let mut final_hidden_sha256 = String::new();
    let mut final_state_sha256 = Vec::new();
    let mut final_hidden_for_q8 = None;
    #[cfg(feature = "qwen35-device-token-handoff")]
    let terminal_receipt_requested = terminal_state_receipt_only;
    #[cfg(not(feature = "qwen35-device-token-handoff"))]
    let terminal_receipt_requested = false;
    #[cfg(feature = "qwen35-device-token-handoff")]
    let mut terminal_state_receipt_digest = None;
    if capture_terminal_state
        || (resident_auxiliary_owners.is_some() && validate_q8_selector_terminal)
    {
        let receipt = final_receipt.ok_or_else(|| invalid("autoregressive terminal receipt"))?;
        let final_hidden = read_f32_buffer(
            stream,
            plans
                .last()
                .ok_or_else(|| invalid("autoregressive terminal layer"))?
                .final_hidden_buffer(),
            receipt,
        )?;
        if capture_terminal_state {
            final_hidden_sha256 = sha256_f32_bits(&final_hidden);
        }
        final_hidden_for_q8 = Some(final_hidden);
    }
    if validate_q8_selector_terminal {
        if let (Some(owners), Some(final_hidden)) = (
            resident_auxiliary_owners.as_mut(),
            final_hidden_for_q8.as_deref(),
        ) {
            owners.validate_q8_terminal(final_hidden).map_err(invalid)?;
            resident_auxiliary = Some(owners.report());
        }
    }
    if capture_terminal_state {
        let receipt = final_receipt.ok_or_else(|| invalid("autoregressive terminal receipt"))?;
        final_state_sha256.reserve(plans.len());
        for plan in &mut *plans {
            final_state_sha256.push(plan.state_sha256(stream, receipt)?);
        }
    }
    if terminal_receipt_requested {
        let receipt = final_receipt.ok_or_else(|| invalid("autoregressive terminal receipt"))?;
        #[cfg(feature = "qwen35-device-token-handoff")]
        {
            #[cfg(feature = "qwen35-kv8-full-attention")]
            {
                terminal_state_receipt_digest =
                    Some(submit_terminal_state_receipt(plans, stream, receipt)?.digest_sha256);
            }
            #[cfg(not(feature = "qwen35-kv8-full-attention"))]
            {
                return Err(invalid(
                    "terminal state receipt requires qwen35-kv8-full-attention",
                ));
            }
        }
    }
    let state_readback_boundaries =
        boundary_delta(state_readback_before, cuda_host_boundary_snapshot())?;

    drop(resident_auxiliary_owners);
    if let Some(report) = resident_auxiliary.as_mut() {
        let after = cuda_device_allocation_snapshot();
        report.cleanup_zero = after.live_bytes == resident_auxiliary_before.live_bytes
            && after.live_allocations == resident_auxiliary_before.live_allocations;
    }

    #[cfg(feature = "qwen35-device-token-handoff")]
    let token_top2 = pending_tokens
        .iter()
        .map(|token| cuda_qwen35_finite_top2_cpu_v1(&token.logits))
        .collect::<Result<Vec<_>, _>>()?;
    let tokens = pending_tokens
        .into_iter()
        .map(|token| {
            let top1_margin = top1_margin_for_argmax(&token.logits, token.output_token_id);
            AutoregressiveTokenReport {
                generation_index: token.generation_index,
                position: token.position,
                input_token_id: token.input_token_id,
                output_token_id: token.output_token_id,
                host_wall_ms: token.host_wall_ms,
                cuda_event_ms: token.cuda_event_ms,
                top1_margin,
                logits_sha256: sha256_f32_bits(&token.logits),
            }
        })
        .collect();

    Ok(AutoregressiveSequence {
        prompt_prefix_tokens_processed: prompt_token_ids.len() - 1,
        prefill_host_wall_ms,
        batched_prefill: prefill.batched_report,
        decode_host_wall_ms,
        tokens,
        generated_token_ids,
        #[cfg(feature = "qwen35-device-token-handoff")]
        token_top2,
        final_hidden_sha256,
        final_state_sha256,
        resident_auxiliary,
        #[cfg(feature = "qwen35-device-token-handoff")]
        terminal_state_receipt_digest,
        execution_boundaries,
        state_readback_boundaries,
        polls,
        proof,
        inter_layer_d2d_copy_count,
        inter_layer_d2d_bytes,
    })
}

#[cfg(feature = "qwen35-device-token-handoff")]
#[allow(clippy::too_many_arguments)]
fn run_device_token_handoff_sequence(
    plans: &mut [DevicePlan],
    output_head: &DeviceOutputHeadPlan,
    handoff: &DeviceTokenHandoffPlan,
    reader: &mut GgufTensorReader<'_>,
    embedding: &GgufTensor,
    prompt_token_ids: &[usize],
    generation_tokens: usize,
    capacity_positions: usize,
    stream: &OwnedCudaStream,
    run_epoch: u32,
    prefill_mode: &mut AutoregressivePrefillMode,
    kv8_ram_roundtrip: bool,
    profile_decode: bool,
    profile_layer_timing: bool,
    q8_dflash_path: Option<&std::path::Path>,
    vision_safetensors_path: Option<&std::path::Path>,
    terminal_state_receipt_only: bool,
    validate_q8_selector_terminal: bool,
) -> Result<DeviceTokenHandoffSequence, Qwen35MixedPrefixCudaError> {
    if plans.len() != FULL_TARGET_LAYER_COUNT
        || prompt_token_ids.is_empty()
        || generation_tokens == 0
        || generation_tokens != handoff.max_tokens
        || capacity_positions == 0
        || run_epoch == 0
    {
        return Err(invalid("device token-handoff sequence extent"));
    }

    let started_event = OwnedCudaEvent::create(stream.device_ordinal())?;
    let stopped_event = OwnedCudaEvent::create(stream.device_ordinal())?;
    let terminal_event = OwnedCudaEvent::create(stream.device_ordinal())?;
    let prefill_boundary_before = cuda_host_boundary_snapshot();
    let mut prefill = run_autoregressive_prefill(
        plans,
        prefill_mode,
        reader,
        embedding,
        prompt_token_ids,
        stream,
        &started_event,
        &stopped_event,
        &terminal_event,
    )?;
    prefill_mode.release_workspace_with_report(&mut prefill)?;
    let prefill_host_wall_ms = prefill.host_wall_ms;
    let mut polls = prefill.polls;
    let mut proof = prefill.proof;
    let mut inter_layer_d2d_copy_count = prefill.inter_layer_d2d_copy_count;
    let mut inter_layer_d2d_bytes = prefill.inter_layer_d2d_bytes;
    let resident_auxiliary_before = cuda_device_allocation_snapshot();
    let mut resident_auxiliary_owners = ResidentAuxiliaryOwners::load(
        q8_dflash_path,
        vision_safetensors_path,
        stream.device_ordinal(),
        stream,
    )
    .map_err(invalid)?;
    let mut resident_auxiliary = resident_auxiliary_owners
        .as_ref()
        .map(ResidentAuxiliaryOwners::report);
    // The optional resident owners are loaded after prefill completes but
    // before handoff decode. Keep their H2D/synchronize work in this ledger.
    let prefill_boundaries =
        boundary_delta(prefill_boundary_before, cuda_host_boundary_snapshot())?;

    #[cfg(all(
        feature = "qwen35-batched-prefill-handoff",
        feature = "qwen35-kv8-full-attention"
    ))]
    let kv8_ram_persistence = if kv8_ram_roundtrip {
        Some(run_kv8_ram_persistence_roundtrip(
            plans,
            stream,
            handoff.plan_digest,
            u64::from(run_epoch),
            capacity_positions,
            prompt_token_ids.len() - 1,
        )?)
    } else {
        None
    };
    #[cfg(not(all(
        feature = "qwen35-batched-prefill-handoff",
        feature = "qwen35-kv8-full-attention"
    )))]
    let kv8_ram_persistence = {
        if kv8_ram_roundtrip {
            return Err(invalid(
                "KV8 RAM roundtrip requires batched prefill and KV8 full attention",
            ));
        }
        None
    };

    let layer_timing_events = profile_layer_timing
        .then(|| {
            DeviceTokenHandoffLayerTimingEvents::new(
                stream.device_ordinal(),
                plans,
                generation_tokens,
            )
        })
        .transpose()?;
    let mut profiler_capture = profile_decode
        .then(CudaProfilerCapture::start)
        .transpose()?;
    let execution_boundary_before = cuda_host_boundary_snapshot();
    let decode_started = Instant::now();
    let first_input_token = *prompt_token_ids
        .last()
        .ok_or_else(|| invalid("device token-handoff prompt token"))?;
    let first_hidden = linear::read_embedding_row(reader, embedding, first_input_token)?;
    plans[0].upload(stream, None, &first_hidden, true)?;
    started_event.record(stream)?;
    for generation_index in 0..generation_tokens {
        if let Some(timing) = &layer_timing_events {
            timing.token_started[generation_index].record(stream)?;
        }
        let position = prompt_token_ids
            .len()
            .checked_sub(1)
            .and_then(|first| first.checked_add(generation_index))
            .ok_or_else(|| invalid("device token-handoff position overflow"))?;
        for layer in 0..plans.len() {
            if layer > 0 {
                let (before, after) = plans.split_at(layer);
                let source = before[layer - 1].final_hidden_view()?;
                if after[0].copy_input_from_device(stream, source)? {
                    inter_layer_d2d_copy_count = inter_layer_d2d_copy_count.saturating_add(1);
                    inter_layer_d2d_bytes = inter_layer_d2d_bytes
                        .saturating_add(linear::HIDDEN.saturating_mul(core::mem::size_of::<f32>()));
                }
            }
            let attention_timing = layer_timing_events
                .as_ref()
                .and_then(|timing| timing.attention_events(generation_index, layer));
            plans[layer].submit_resident_with_attention_timing(
                stream,
                position,
                attention_timing,
            )?;
            if let Some(owners) = resident_auxiliary_owners.as_mut() {
                let hidden_view = CudaF32View::new(
                    plans[layer].final_hidden_view()?,
                    PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                    linear::HIDDEN,
                )?;
                owners
                    .capture_q8_feature_layer(layer, hidden_view, stream)
                    .map_err(invalid)?;
            }
            if let Some(timing) = &layer_timing_events {
                timing.layer_stopped[generation_index][layer].record(stream)?;
            }
        }
        output_head.submit(stream)?;
        if let Some(owners) = resident_auxiliary_owners.as_mut() {
            let hidden_view = CudaF32View::new(
                plans
                    .last()
                    .ok_or_else(|| invalid("device token-handoff terminal layer"))?
                    .final_hidden_view()?,
                PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                linear::HIDDEN,
            )?;
            owners
                .consume_q8_device(generation_index, hidden_view, stream)
                .map_err(invalid)?;
            owners
                .consume_q8_feature_device(generation_index, position, stream)
                .map_err(invalid)?;
        }
        handoff
            .bind_top2(stream, &output_head.logits, run_epoch, generation_index)?
            .submit(stream)?;
        if generation_index + 1 < generation_tokens {
            handoff
                .bind_embedding_gather(stream, generation_index, plans[0].input_f32_view()?)?
                .submit_for_plan(stream)?;
        }
        if let Some(timing) = &layer_timing_events {
            timing.token_stopped[generation_index].record(stream)?;
        }
    }
    stopped_event.record(stream)?;
    let receipt = stream.submission_receipt()?;
    terminal_event.record_for_submission(stream, receipt)?;

    let poll_limit = MAX_COMPLETION_POLLS
        .saturating_mul(plans.len() as u64)
        .saturating_mul(generation_tokens as u64);
    let (completion, decode_polls) = match poll_submission(
        poll_limit,
        || stream.query_event_completion(&terminal_event, receipt),
        bounded_completion_poll_backoff,
    ) {
        Ok(value) => value,
        Err(SubmissionPollError::Query(error)) => return Err(error.into()),
        Err(SubmissionPollError::Exhausted { .. } | SubmissionPollError::InvalidBudget) => {
            return Err(Qwen35MixedPrefixCudaError::CompletionPollLimit)
        }
    };
    // Prefill and this decode submission have independent budgets. Keep their
    // counts additive for telemetry only; never charge completed prefill polls
    // against the decode submission's bound.
    let polls = prefill.polls.saturating_add(decode_polls);
    proof &= completion.covers_submission(receipt) && completion.belongs_to_stream(stream)?;
    if !proof {
        return Err(invalid("device token-handoff owner epoch proof"));
    }
    if let Some(capture) = profiler_capture.as_mut() {
        capture.stop()?;
    }
    let layer_timing = layer_timing_events
        .as_ref()
        .map(|events| events.report(plans))
        .transpose()?;
    let cuda_event_ms = f64::from(started_event.elapsed_ms(&stopped_event)?);
    let decode_host_wall_ms = decode_started.elapsed().as_secs_f64() * 1_000.0;
    if !cuda_event_ms.is_finite()
        || cuda_event_ms <= 0.0
        || !decode_host_wall_ms.is_finite()
        || decode_host_wall_ms <= 0.0
    {
        return Err(invalid("device token-handoff timing"));
    }
    if let Some(owners) = resident_auxiliary_owners.as_mut() {
        // The selector output is a diagnostic component readback. Close it
        // before the handoff execution interval so the compact receipt copy
        // remains isolated in receipt_readback_boundaries.
        // Receipt-only handoff must not let the optional Q8 component oracle
        // readback pollute the measured decode transport ledger.  Keep the
        // diagnostic terminal readback only when explicitly requested (or
        // when selector validation needs the output on the host).
        owners
            .finish_consumers_with_policy(
                stream,
                !terminal_state_receipt_only || validate_q8_selector_terminal,
                !terminal_state_receipt_only,
            )
            .map_err(invalid)?;
        resident_auxiliary = Some(owners.report());
    }
    let execution_boundaries =
        boundary_delta(execution_boundary_before, cuda_host_boundary_snapshot())?;
    let receipt_readback_before = cuda_host_boundary_snapshot();
    let mut receipt_bytes = vec![0_u8; handoff.receipts.len_bytes()];
    handoff.receipts.copy_to_host_sync_with_submission_proof(
        stream,
        &mut receipt_bytes,
        receipt,
    )?;
    let receipt_readback_boundaries =
        boundary_delta(receipt_readback_before, cuda_host_boundary_snapshot())?;
    let raw_receipt_sha256 = sha256_hex(&receipt_bytes);

    let mut generated_token_ids = Vec::with_capacity(generation_tokens);
    let mut receipt_reports = Vec::with_capacity(generation_tokens);
    let first_decode_position = prompt_token_ids.len() - 1;
    for generation_index in 0..generation_tokens {
        let offset = generation_index
            .checked_mul(QWEN35_TOKEN_HANDOFF_RECEIPT_BYTES)
            .ok_or_else(|| invalid("device token-handoff parse offset overflow"))?;
        let step_generation = handoff.step_generation(run_epoch, generation_index)?;
        let outcome = parse_cuda_qwen35_token_handoff_receipt_v1(
            &receipt_bytes[offset..offset + QWEN35_TOKEN_HANDOFF_RECEIPT_BYTES],
            step_generation,
            handoff.plan_digest,
        )?;
        let (top2, finite_count) = match outcome {
            CudaQwen35TokenHandoffOutcomeV1::Selected {
                top2, finite_count, ..
            } => (top2, finite_count),
            CudaQwen35TokenHandoffOutcomeV1::NonFiniteLogit {
                first_non_finite_id,
                ..
            } => {
                return Err(invalid(format!(
                    "device token-handoff non-finite logit at {first_non_finite_id}"
                )))
            }
        };
        let input_token_id = generated_token_ids
            .last()
            .copied()
            .unwrap_or(first_input_token);
        let token_id = usize::try_from(top2.token_id)
            .map_err(|_| invalid("device token-handoff token conversion"))?;
        let runner_up_id = usize::try_from(top2.runner_up_id)
            .map_err(|_| invalid("device token-handoff runner-up conversion"))?;
        generated_token_ids.push(token_id);
        receipt_reports.push(DeviceTokenHandoffReceiptReport {
            generation_index,
            position: first_decode_position + generation_index,
            input_token_id,
            token_id,
            top1_value_bits: top2.top1_value_bits,
            runner_up_id,
            runner_up_value_bits: top2.runner_up_value_bits,
            top1_margin: f64::from(top2.top1_value()) - f64::from(top2.runner_up_value()),
            finite_count: finite_count as usize,
            step_generation,
        });
    }

    let state_readback_before = cuda_host_boundary_snapshot();
    // Receipt-only handoff deliberately suppresses the terminal hidden digest.
    // The selector A/B still needs the hidden bytes, so validation may read them
    // back, but that read must not silently turn receipt-only transport into a
    // hidden-publication contract.  The terminal state receipt remains the
    // parity witness for this mode.
    let final_hidden = if terminal_state_receipt_only && !validate_q8_selector_terminal {
        None
    } else {
        Some(read_f32_buffer(
            stream,
            plans
                .last()
                .ok_or_else(|| invalid("device token-handoff terminal layer"))?
                .final_hidden_buffer(),
            receipt,
        )?)
    };
    if let Some(final_hidden) = final_hidden.as_deref() {
        if validate_q8_selector_terminal {
            if let Some(owners) = resident_auxiliary_owners.as_mut() {
                owners.validate_q8_terminal(final_hidden).map_err(invalid)?;
                resident_auxiliary = Some(owners.report());
            }
        }
    }
    let final_hidden_sha256 = if terminal_state_receipt_only {
        String::new()
    } else {
        sha256_f32_bits(
            final_hidden
                .as_deref()
                .ok_or_else(|| invalid("device token-handoff terminal hidden"))?,
        )
    };
    let mut final_state_sha256 = Vec::with_capacity(plans.len());
    let terminal_state_receipt = if terminal_state_receipt_only {
        #[cfg(feature = "qwen35-kv8-full-attention")]
        {
            Some(submit_terminal_state_receipt(plans, stream, receipt)?)
        }
        #[cfg(not(feature = "qwen35-kv8-full-attention"))]
        {
            return Err(invalid(
                "terminal state receipt requires qwen35-kv8-full-attention",
            ));
        }
    } else {
        for plan in plans {
            final_state_sha256.push(plan.state_sha256(stream, receipt)?);
        }
        None
    };
    let state_readback_boundaries =
        boundary_delta(state_readback_before, cuda_host_boundary_snapshot())?;

    drop(resident_auxiliary_owners);
    if let Some(report) = resident_auxiliary.as_mut() {
        let after = cuda_device_allocation_snapshot();
        report.cleanup_zero = after.live_bytes == resident_auxiliary_before.live_bytes
            && after.live_allocations == resident_auxiliary_before.live_allocations;
    }

    Ok(DeviceTokenHandoffSequence {
        prefill_host_wall_ms,
        batched_prefill: prefill.batched_report,
        decode_host_wall_ms,
        cuda_event_ms,
        cuda_profiler_capture_requested: profile_decode,
        layer_timing,
        generated_token_ids,
        receipts: receipt_reports,
        raw_receipt_sha256,
        final_hidden_sha256,
        final_state_sha256,
        resident_auxiliary,
        prefill_boundaries,
        kv8_ram_persistence,
        execution_boundaries,
        receipt_readback_boundaries,
        state_readback_boundaries,
        terminal_state_receipt,
        polls,
        proof,
        inter_layer_d2d_copy_count,
        inter_layer_d2d_bytes,
    })
}

#[cfg(feature = "qwen35-device-token-handoff")]
fn device_token_handoff_report(
    plan: &DeviceTokenHandoffPlan,
    sequence: &DeviceTokenHandoffSequence,
    audit: &AutoregressiveSequence,
) -> Result<DeviceTokenHandoffReport, Qwen35MixedPrefixCudaError> {
    let generation_tokens = sequence.generated_token_ids.len();
    let receipt_d2h_bytes = generation_tokens
        .checked_mul(QWEN35_TOKEN_HANDOFF_RECEIPT_BYTES)
        .ok_or_else(|| invalid("device token-handoff receipt byte total overflow"))?;
    let full_logits_d2h_bytes_avoided = generation_tokens
        .checked_mul(QWEN35_TOKEN_HANDOFF_LOGITS_BYTES)
        .ok_or_else(|| invalid("device token-handoff logits byte total overflow"))?;
    let embedding_gather_count = generation_tokens.saturating_sub(1);
    let recurrent_embedding_h2d_bytes_avoided = embedding_gather_count
        .checked_mul(linear::HIDDEN)
        .and_then(|elements| elements.checked_mul(core::mem::size_of::<f32>()))
        .ok_or_else(|| invalid("device token-handoff embedding byte total overflow"))?;
    let first_upload_bytes = linear::HIDDEN
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| invalid("device token-handoff first upload extent overflow"))?;

    let generated_tokens_match_audit = sequence.generated_token_ids == audit.generated_token_ids;
    let top2_bits_match_audit = audit.token_top2.len() == sequence.receipts.len()
        && audit
            .token_top2
            .iter()
            .zip(&sequence.receipts)
            .all(|(expected, actual)| {
                expected.token_id as usize == actual.token_id
                    && expected.top1_value_bits == actual.top1_value_bits
                    && expected.runner_up_id as usize == actual.runner_up_id
                    && expected.runner_up_value_bits == actual.runner_up_value_bits
            });
    let final_hidden_matches_audit = sequence.final_hidden_sha256 == audit.final_hidden_sha256;
    let final_state_matches_audit = sequence
        .terminal_state_receipt
        .as_ref()
        .zip(audit.terminal_state_receipt_digest.as_ref())
        .map_or_else(
            || sequence.final_state_sha256 == audit.final_state_sha256,
            |(actual, expected)| actual.digest_sha256 == *expected,
        );
    let terminal_state_receipt_contract = sequence.terminal_state_receipt.is_some()
        && sequence.final_hidden_sha256.is_empty()
        && final_state_matches_audit;
    let mut kv8_ram_persistence = sequence.kv8_ram_persistence.clone();
    if let Some(persistence) = kv8_ram_persistence.as_mut() {
        persistence.first_suffix_token_matches_control = sequence
            .generated_token_ids
            .first()
            .zip(audit.generated_token_ids.first())
            .is_some_and(|(actual, expected)| actual == expected);
        persistence.first_suffix_top2_bits_match_control = sequence
            .receipts
            .first()
            .zip(audit.token_top2.first())
            .is_some_and(|(actual, expected)| {
                actual.token_id == expected.token_id as usize
                    && actual.top1_value_bits == expected.top1_value_bits
                    && actual.runner_up_id == expected.runner_up_id as usize
                    && actual.runner_up_value_bits == expected.runner_up_value_bits
            });
        persistence.final_hidden_matches_control = final_hidden_matches_audit;
        persistence.final_state_matches_control = final_state_matches_audit;
        let linear_state_contract = persistence
            .linear_state_persistence
            .as_ref()
            .is_some_and(|report| report.complete_numerical_contract);
        persistence.complete_numerical_contract = persistence.q8_cache_geometry_exact
            && persistence.source_export_complete
            && persistence.clear_sample_zero
            && persistence.restore_byte_exact
            && persistence.transfer_copy_path_async
            && linear_state_contract
            && persistence.stream_owner_identity_closed
            && persistence.boundary_ledger_closed
            && persistence.allocation_cleanup_zero
            && persistence.first_suffix_token_matches_control
            && persistence.first_suffix_top2_bits_match_control
            && persistence.final_hidden_matches_control
            && persistence.final_state_matches_control;
    }
    let receipt_identity_closed = sequence.receipts.len() == plan.max_tokens
        && sequence
            .receipts
            .iter()
            .enumerate()
            .all(|(index, receipt)| {
                receipt.generation_index == index
                    && receipt.finite_count == QWEN35_TOKEN_HANDOFF_VOCAB
                    && receipt.step_generation == ((1_u64 << 32) | (index as u64 + 1))
            });
    let selector_output_d2h_bytes = sequence
        .resident_auxiliary
        .as_ref()
        .and_then(|report| report.q8_selector_consumer.as_ref())
        .filter(|report| {
            report
                .get("component_consumed")
                .and_then(|value| value.as_bool())
                == Some(true)
                && report
                    .get("terminal_readback_skipped")
                    .and_then(|value| value.as_bool())
                    != Some(true)
        })
        .map_or(0_u64, |_| 256_u64 * 4);
    let selector_output_d2h_copies = u64::from(selector_output_d2h_bytes != 0);
    let measured_boundary_contract_closed = sequence.execution_boundaries.stream_synchronizations
        == 0
        && sequence.execution_boundaries.event_synchronizations == 0
        && sequence.execution_boundaries.synchronous_h2d_copies == 1
        && sequence.execution_boundaries.synchronous_h2d_bytes == first_upload_bytes as u64
        && sequence.execution_boundaries.synchronous_d2h_copies == selector_output_d2h_copies
        && sequence.execution_boundaries.synchronous_d2h_bytes == selector_output_d2h_bytes
        && sequence.receipt_readback_boundaries.synchronous_h2d_copies == 0
        && sequence.receipt_readback_boundaries.synchronous_h2d_bytes == 0
        && sequence.receipt_readback_boundaries.synchronous_d2h_copies == 1
        && sequence.receipt_readback_boundaries.synchronous_d2h_bytes == receipt_d2h_bytes as u64;
    let kv8_ram_persistence_contract = kv8_ram_persistence
        .as_ref()
        .is_none_or(|report| report.complete_numerical_contract);
    let complete_numerical_contract = generated_tokens_match_audit
        && top2_bits_match_audit
        && (final_hidden_matches_audit || terminal_state_receipt_contract)
        && final_state_matches_audit
        && receipt_identity_closed
        && sequence.proof
        && kv8_ram_persistence_contract;

    Ok(DeviceTokenHandoffReport {
        scope: "diagnostic_device_top2_receipt_to_resident_bf16_embedding_same_stream".to_owned(),
        embedding_resident: true,
        embedding_payload_bytes: plan.embedding_payload_bytes,
        embedding_payload_sha256: plan.embedding_payload_sha256.clone(),
        plan_digest: hex_bytes(&plan.plan_digest),
        generation_tokens,
        cuda_profiler_capture_requested: sequence.cuda_profiler_capture_requested,
        layer_timing: sequence.layer_timing.clone(),
        prefill_host_wall_ms: sequence.prefill_host_wall_ms,
        batched_prefill: sequence.batched_prefill.clone(),
        aggregate_decode_host_wall_ms: sequence.decode_host_wall_ms,
        aggregate_cuda_event_ms: sequence.cuda_event_ms,
        visible_tokens_per_second: generation_tokens as f64 * 1_000.0
            / sequence.decode_host_wall_ms,
        cuda_event_tokens_per_second: generation_tokens as f64 * 1_000.0 / sequence.cuda_event_ms,
        top2_kernel_count: generation_tokens,
        embedding_gather_count,
        inter_layer_d2d_copy_count: sequence.inter_layer_d2d_copy_count,
        inter_layer_d2d_bytes: sequence.inter_layer_d2d_bytes,
        full_logits_d2h_bytes_avoided,
        recurrent_embedding_h2d_bytes_avoided,
        compact_receipt_d2h_bytes: receipt_d2h_bytes,
        receipt_sha256: sequence.raw_receipt_sha256.clone(),
        terminal_poll_iterations: sequence.polls,
        owner_epoch_proof: sequence.proof,
        generated_token_ids: sequence.generated_token_ids.clone(),
        receipts: sequence.receipts.clone(),
        generated_tokens_match_audit,
        top2_bits_match_audit,
        final_hidden_matches_audit,
        final_state_matches_audit,
        terminal_state_receipt_contract,
        complete_numerical_contract,
        measured_boundary_contract_closed,
        prefill_boundaries: sequence.prefill_boundaries.clone(),
        execution_boundaries: sequence.execution_boundaries.clone(),
        receipt_readback_boundaries: sequence.receipt_readback_boundaries.clone(),
        state_readback_boundaries: sequence.state_readback_boundaries.clone(),
        resident_auxiliary: sequence.resident_auxiliary.clone(),
        terminal_state_receipt: sequence.terminal_state_receipt.clone(),
        terminal_state_receipt_expected_digest_sha256: audit.terminal_state_receipt_digest.clone(),
        kv8_ram_persistence,
        full_logits_readback_per_token: false,
        cpu_argmax_on_measured_path: false,
        host_embedding_lookup_after_first_output: false,
        product_tps_permitted: false,
    })
}

fn run_resident_fast_sequence(
    plans: &[DevicePlan],
    output_head: Option<&DeviceOutputHeadPlan>,
    embeddings: &[Vec<f32>],
    stream: &OwnedCudaStream,
    layer_count: usize,
) -> Result<ResidentFastSequence, Qwen35MixedPrefixCudaError> {
    if layer_count == 0 || plans.len() != layer_count || embeddings.is_empty() {
        return Err(invalid("resident fast-path geometry"));
    }
    let started_event = OwnedCudaEvent::create(stream.device_ordinal())?;
    let stopped_event = OwnedCudaEvent::create(stream.device_ordinal())?;
    let terminal_event = OwnedCudaEvent::create(stream.device_ordinal())?;
    let layer_stopped_events = (0..layer_count)
        .map(|_| OwnedCudaEvent::create(stream.device_ordinal()))
        .collect::<Result<Vec<_>, _>>()?;
    let poll_limit = MAX_COMPLETION_POLLS.saturating_mul(layer_count as u64);
    let mut token_host_wall_ms = Vec::with_capacity(embeddings.len());
    let mut token_backbone_cuda_event_ms = Vec::with_capacity(embeddings.len());
    let mut token_output_head_cuda_event_ms = Vec::with_capacity(embeddings.len());
    let mut token_cuda_event_ms = Vec::with_capacity(embeddings.len());
    let mut token_layer_cuda_event_ms = Vec::with_capacity(embeddings.len());
    let mut final_hidden = Vec::with_capacity(embeddings.len());
    let mut final_norm = Vec::with_capacity(embeddings.len());
    let mut logits = Vec::with_capacity(embeddings.len());
    let mut input_boundaries = BoundaryReport::default();
    let mut hot_path_boundaries = BoundaryReport::default();
    let mut readback_boundaries = BoundaryReport::default();
    let mut polls = 0_u64;
    let mut proof = true;
    let mut inter_layer_d2d_copy_count = 0_usize;
    let mut inter_layer_d2d_bytes = 0_usize;

    for (position, embedding) in embeddings.iter().enumerate() {
        let token_started = Instant::now();
        let input_before = cuda_host_boundary_snapshot();
        plans[0].upload(stream, None, embedding, true)?;
        input_boundaries = add_boundaries(
            input_boundaries,
            boundary_delta(input_before, cuda_host_boundary_snapshot())?,
        );

        started_event.record(stream)?;
        let hot_before = cuda_host_boundary_snapshot();
        for layer in 0..layer_count {
            if layer > 0 {
                let (before, after) = plans.split_at(layer);
                let source = before[layer - 1].final_hidden_view()?;
                if after[0].copy_input_from_device(stream, source)? {
                    inter_layer_d2d_copy_count = inter_layer_d2d_copy_count.saturating_add(1);
                    inter_layer_d2d_bytes = inter_layer_d2d_bytes.saturating_add(
                        embedding.len().saturating_mul(core::mem::size_of::<f32>()),
                    );
                }
            }
            plans[layer].submit_resident(stream, position)?;
            layer_stopped_events[layer].record(stream)?;
        }
        if let Some(output_head) = output_head {
            output_head.submit(stream)?;
        }
        stopped_event.record(stream)?;
        let receipt = stream.submission_receipt()?;
        terminal_event.record_for_submission(stream, receipt)?;
        let mut token_polls = 0_u64;
        let completion = loop {
            if let Some(completion) = stream.query_event_completion(&terminal_event, receipt)? {
                break completion;
            }
            token_polls = token_polls.saturating_add(1);
            if token_polls >= poll_limit {
                return Err(Qwen35MixedPrefixCudaError::CompletionPollLimit);
            }
            bounded_completion_poll_backoff(token_polls);
        };
        let token_proof =
            completion.covers_submission(receipt) && completion.belongs_to_stream(stream)?;
        if !token_proof {
            return Err(invalid("resident fast-path owner epoch proof"));
        }
        hot_path_boundaries = add_boundaries(
            hot_path_boundaries,
            boundary_delta(hot_before, cuda_host_boundary_snapshot())?,
        );
        let backbone_cuda_event_ms =
            f64::from(started_event.elapsed_ms(&layer_stopped_events[layer_count - 1])?);
        let cuda_event_ms = f64::from(started_event.elapsed_ms(&stopped_event)?);
        if !backbone_cuda_event_ms.is_finite()
            || backbone_cuda_event_ms <= 0.0
            || !cuda_event_ms.is_finite()
            || cuda_event_ms <= 0.0
        {
            return Err(invalid("resident fast-path CUDA event timing"));
        }
        if output_head.is_some() {
            let output_head_cuda_event_ms =
                f64::from(layer_stopped_events[layer_count - 1].elapsed_ms(&stopped_event)?);
            if !output_head_cuda_event_ms.is_finite() || output_head_cuda_event_ms <= 0.0 {
                return Err(invalid("resident output-head CUDA event timing"));
            }
            token_output_head_cuda_event_ms.push(output_head_cuda_event_ms);
        }
        token_host_wall_ms.push(token_started.elapsed().as_secs_f64() * 1_000.0);
        token_backbone_cuda_event_ms.push(backbone_cuda_event_ms);
        token_cuda_event_ms.push(cuda_event_ms);
        let mut layer_cuda_event_ms = Vec::with_capacity(layer_count);
        for layer in 0..layer_count {
            let layer_started_event = if layer == 0 {
                &started_event
            } else {
                &layer_stopped_events[layer - 1]
            };
            let elapsed_ms =
                f64::from(layer_started_event.elapsed_ms(&layer_stopped_events[layer])?);
            if !elapsed_ms.is_finite() || elapsed_ms <= 0.0 {
                return Err(invalid("resident fast-path per-layer CUDA event timing"));
            }
            layer_cuda_event_ms.push(elapsed_ms);
        }
        token_layer_cuda_event_ms.push(layer_cuda_event_ms);
        polls = polls.saturating_add(token_polls);
        proof &= token_proof;

        let readback_before = cuda_host_boundary_snapshot();
        final_hidden.push(read_f32_buffer(
            stream,
            plans[layer_count - 1].final_hidden_buffer(),
            receipt,
        )?);
        if let Some(output_head) = output_head {
            final_norm.push(read_f32_buffer(
                stream,
                &output_head.normalized_hidden,
                receipt,
            )?);
            logits.push(read_f32_buffer(stream, &output_head.logits, receipt)?);
        }
        readback_boundaries = add_boundaries(
            readback_boundaries,
            boundary_delta(readback_before, cuda_host_boundary_snapshot())?,
        );
    }

    Ok(ResidentFastSequence {
        token_host_wall_ms,
        token_backbone_cuda_event_ms,
        token_output_head_cuda_event_ms,
        token_cuda_event_ms,
        token_layer_cuda_event_ms,
        final_hidden,
        final_norm,
        logits,
        input_boundaries,
        hot_path_boundaries,
        readback_boundaries,
        polls,
        proof,
        inter_layer_d2d_copy_count,
        inter_layer_d2d_bytes,
    })
}

fn run_device_sequence(
    plans: &mut [DevicePlan],
    materials: &[LayerMaterial],
    embeddings: &[Vec<f32>],
    model_config: &Qwen35Config,
    stream: &OwnedCudaStream,
    input_boundaries: &mut BoundaryReport,
    device_resident: bool,
    layer_count: usize,
    audit_ffn: bool,
    audit_attention: bool,
) -> Result<DeviceSequence, Qwen35MixedPrefixCudaError> {
    if layer_count == 0 || plans.len() != layer_count || materials.len() != layer_count {
        return Err(invalid("device plan/material layer count mismatch"));
    }
    let started = Instant::now();
    let mut layers: Vec<Vec<DeviceLayerReadback>> = (0..layer_count)
        .map(|_| Vec::with_capacity(TOKEN_COUNT))
        .collect();
    let mut layer_host_wall_ms = vec![0.0_f64; layer_count];
    let mut completion_boundaries = BoundaryReport::default();
    let mut readback_boundaries = BoundaryReport::default();
    let mut polls = 0_u64;
    let mut proof = true;
    let mut inter_layer_d2d_copy_count = 0_usize;
    let mut inter_layer_d2d_bytes = 0_usize;
    let mut host_projection_token_count = 0_usize;
    let mut host_hidden_handoff_token_count = 0_usize;

    for (position, embedding) in embeddings.iter().enumerate() {
        let mut hidden = embedding.clone();
        for layer in 0..layer_count {
            let input_hidden = hidden.clone();
            let projected = if device_resident {
                None
            } else {
                match &materials[layer] {
                    LayerMaterial::Linear(material) => {
                        host_projection_token_count = host_projection_token_count.saturating_add(1);
                        Some(ProjectedInput::Linear(project_linear(
                            material,
                            &input_hidden,
                            model_config.rms_norm_eps as f32,
                        )?))
                    }
                    LayerMaterial::Full(_) => None,
                }
            };
            let input_before = cuda_host_boundary_snapshot();
            if device_resident && layer > 0 {
                let (before, after) = plans.split_at_mut(layer);
                let source = before[layer - 1].final_hidden_view()?;
                if after[0].copy_input_from_device(stream, source)? {
                    inter_layer_d2d_copy_count = inter_layer_d2d_copy_count.saturating_add(1);
                    inter_layer_d2d_bytes = inter_layer_d2d_bytes.saturating_add(
                        input_hidden
                            .len()
                            .saturating_mul(core::mem::size_of::<f32>()),
                    );
                }
            } else {
                host_hidden_handoff_token_count = host_hidden_handoff_token_count.saturating_add(1);
                plans[layer].upload(stream, projected.as_ref(), &input_hidden, device_resident)?;
            }
            *input_boundaries = add_boundaries(
                input_boundaries.clone(),
                boundary_delta(input_before, cuda_host_boundary_snapshot())?,
            );
            let layer_started = Instant::now();
            let mut readback = plans[layer].execute(stream, position)?;
            layer_host_wall_ms[layer] += layer_started.elapsed().as_secs_f64() * 1_000.0;
            if audit_ffn {
                let before = cuda_host_boundary_snapshot();
                readback.ffn_intermediates =
                    Some(ffn_operator_audit::readback(&plans[layer], stream)?);
                readback.readback_boundaries = add_boundaries(
                    readback.readback_boundaries,
                    boundary_delta(before, cuda_host_boundary_snapshot())?,
                );
            }
            readback.input_hidden = input_hidden;
            if audit_attention {
                let before = cuda_host_boundary_snapshot();
                match &plans[layer] {
                    DevicePlan::Full(_) => {
                        readback.full_attention_intermediates =
                            full_attention_operator_audit::readback(
                                &plans[layer],
                                stream,
                                position,
                            )?;
                    }
                    DevicePlan::Linear(_) => {
                        readback.attention_intermediates =
                            attention_operator_audit::readback(&plans[layer], stream)?;
                    }
                }
                readback.readback_boundaries = add_boundaries(
                    readback.readback_boundaries,
                    boundary_delta(before, cuda_host_boundary_snapshot())?,
                );
            }
            hidden = readback.final_hidden.clone();
            polls = polls.saturating_add(readback.polls);
            proof &= readback.proof;
            completion_boundaries = add_boundaries(
                completion_boundaries,
                readback.completion_boundaries.clone(),
            );
            readback_boundaries =
                add_boundaries(readback_boundaries, readback.readback_boundaries.clone());
            layers[layer].push(readback);
        }
    }
    Ok(DeviceSequence {
        layers,
        host_wall_ms: started.elapsed().as_secs_f64() * 1_000.0,
        layer_host_wall_ms,
        completion_boundaries,
        readback_boundaries,
        polls,
        proof,
        inter_layer_d2d_copy_count,
        inter_layer_d2d_bytes,
        host_projection_token_count,
        host_hidden_handoff_token_count,
    })
}

struct ExpectedLayerResult {
    input_hidden: Vec<f32>,
    output: Qwen35LayerOutput,
    state: Vec<f32>,
}

struct ExpectedSequence {
    layers: Vec<Vec<ExpectedLayerResult>>,
    final_hidden: Vec<Vec<f32>>,
}

fn run_cpu_sequence(
    executor: &mut GgufHybridPrefixExecutor,
    embeddings: &[Vec<f32>],
    capacity: usize,
    layer_count: usize,
) -> Result<ExpectedSequence, Qwen35MixedPrefixCudaError> {
    if layer_count == 0 || layer_count > executor.layer_count() {
        return Err(invalid("CPU oracle layer count is outside the executor"));
    }
    let specs = executor.specs();
    let mut states = specs
        .iter()
        .map(|spec| Qwen35LayerState::for_spec(spec, capacity))
        .collect::<Result<Vec<_>, _>>()?;
    let mut layers: Vec<Vec<ExpectedLayerResult>> = (0..layer_count)
        .map(|_| Vec::with_capacity(TOKEN_COUNT))
        .collect();
    let mut final_hidden = Vec::with_capacity(TOKEN_COUNT);
    for (position, embedding) in embeddings.iter().enumerate() {
        let mut hidden = embedding.clone();
        for layer in 0..layer_count {
            let input_hidden = hidden.clone();
            let output = executor.forward_layer_output(
                layer,
                &mut states[layer],
                &input_hidden,
                position,
            )?;
            let state = states[layer].checkpoint().state_values();
            hidden = output.hidden.clone();
            layers[layer].push(ExpectedLayerResult {
                input_hidden,
                output,
                state,
            });
        }
        final_hidden.push(hidden);
    }
    Ok(ExpectedSequence {
        layers,
        final_hidden,
    })
}

fn expected_sequences_equal(left: &ExpectedSequence, right: &ExpectedSequence) -> bool {
    if left.final_hidden != right.final_hidden || left.layers.len() != right.layers.len() {
        return false;
    }
    left.layers.iter().zip(&right.layers).all(|(left, right)| {
        left.len() == right.len()
            && left.iter().zip(right).all(|(left, right)| {
                left.input_hidden == right.input_hidden
                    && left.output == right.output
                    && left.state == right.state
            })
    })
}

fn expected_full_state(values: &[f32], capacity: usize, kv_values: usize) -> Vec<f32> {
    let Some(region_elements) = kv_values.checked_mul(2) else {
        return Vec::new();
    };
    if region_elements == 0 || !values.len().is_multiple_of(region_elements) {
        return Vec::new();
    }
    let occupied = (values.len() / region_elements).min(capacity);
    let Some(value_start) = capacity.checked_mul(kv_values) else {
        return Vec::new();
    };
    let Some(result_elements) = value_start.checked_mul(2) else {
        return Vec::new();
    };
    let Some(key_values) = occupied.checked_mul(kv_values) else {
        return Vec::new();
    };
    let source_value_start = values.len() / 2;
    if key_values > values.len().saturating_sub(source_value_start) {
        return Vec::new();
    }
    let mut result = vec![0.0_f32; result_elements];
    result[..key_values].copy_from_slice(&values[..key_values]);
    result[value_start..value_start + key_values]
        .copy_from_slice(&values[source_value_start..source_value_start + key_values]);
    result
}

fn compare(
    name: &str,
    actual: &[f32],
    expected: &[f32],
    absolute_tolerance: f32,
    relative_tolerance: f32,
) -> NumericalReport {
    let mut max_abs_diff = 0.0_f32;
    let mut max_rel_diff = 0.0_f32;
    let mut max_abs_index = None;
    let mut actual_at_max_abs = None;
    let mut expected_at_max_abs = None;
    let finite_count = actual.iter().filter(|value| value.is_finite()).count();
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let absolute = (*actual - *expected).abs();
        if absolute > max_abs_diff {
            max_abs_diff = absolute;
            max_abs_index = Some(index);
            actual_at_max_abs = Some(*actual);
            expected_at_max_abs = Some(*expected);
        }
        max_rel_diff = max_rel_diff.max(absolute / expected.abs().max(1.0e-6));
    }
    NumericalReport {
        name: name.to_owned(),
        elements: actual.len(),
        finite_count,
        max_abs_diff,
        max_rel_diff,
        max_abs_index,
        actual_at_max_abs,
        expected_at_max_abs,
        absolute_tolerance,
        relative_tolerance,
        output_sha256: sha256_f32_bits(actual),
        expected_sha256: sha256_f32_bits(expected),
        pass: actual.len() == expected.len()
            && finite_count == actual.len()
            && actual.iter().zip(expected).all(|(actual, expected)| {
                (*actual - *expected).abs()
                    <= absolute_tolerance + relative_tolerance * expected.abs()
            }),
    }
}

fn trajectory_sample(name: &str, actual: &[f32], expected: &[f32]) -> TrajectorySampleReport {
    let shape_match = actual.len() == expected.len();
    let finite_pair_count = actual
        .iter()
        .zip(expected)
        .filter(|(actual, expected)| actual.is_finite() && expected.is_finite())
        .count();
    let metrics_valid = shape_match && !actual.is_empty() && finite_pair_count == actual.len();

    let metrics = metrics_valid.then(|| {
        let mut actual_sum = 0.0_f64;
        let mut expected_sum = 0.0_f64;
        let mut actual_squared = 0.0_f64;
        let mut expected_squared = 0.0_f64;
        let mut error_squared = 0.0_f64;
        let mut dot = 0.0_f64;
        for (actual, expected) in actual.iter().zip(expected) {
            let actual = f64::from(*actual);
            let expected = f64::from(*expected);
            let error = actual - expected;
            actual_sum += actual;
            expected_sum += expected;
            actual_squared = actual.mul_add(actual, actual_squared);
            expected_squared = expected.mul_add(expected, expected_squared);
            error_squared = error.mul_add(error, error_squared);
            dot = actual.mul_add(expected, dot);
        }
        let actual_l2_norm = actual_squared.sqrt();
        let expected_l2_norm = expected_squared.sqrt();
        let error_l2_norm = error_squared.sqrt();
        let rmse = (error_squared / actual.len() as f64).sqrt();
        let relative_l2_error = if expected_l2_norm > 0.0 {
            error_l2_norm / expected_l2_norm
        } else if error_l2_norm == 0.0 {
            0.0
        } else {
            f64::NAN
        };
        let cosine_similarity = if actual_l2_norm > 0.0 && expected_l2_norm > 0.0 {
            dot / (actual_l2_norm * expected_l2_norm)
        } else if actual_l2_norm == 0.0 && expected_l2_norm == 0.0 {
            1.0
        } else {
            f64::NAN
        };
        (
            actual_sum,
            expected_sum,
            actual_l2_norm,
            expected_l2_norm,
            error_l2_norm,
            rmse,
            relative_l2_error,
            cosine_similarity,
        )
    });
    let finite_metric = |value: f64| value.is_finite().then_some(value);

    TrajectorySampleReport {
        name: name.to_owned(),
        elements: actual.len(),
        shape_match,
        finite_pair_count,
        actual_sum: metrics.and_then(|values| finite_metric(values.0)),
        expected_sum: metrics.and_then(|values| finite_metric(values.1)),
        actual_l2_norm: metrics.and_then(|values| finite_metric(values.2)),
        expected_l2_norm: metrics.and_then(|values| finite_metric(values.3)),
        error_l2_norm: metrics.and_then(|values| finite_metric(values.4)),
        rmse: metrics.and_then(|values| finite_metric(values.5)),
        relative_l2_error: metrics.and_then(|values| finite_metric(values.6)),
        cosine_similarity: metrics.and_then(|values| finite_metric(values.7)),
    }
}

fn aggregate_trajectory<'a>(
    samples: impl Iterator<Item = &'a TrajectorySampleReport>,
) -> TrajectoryAggregateReport {
    let mut sample_count = 0_usize;
    let mut finite_metric_count = 0_usize;
    let mut max_relative_l2_error = None::<f64>;
    let mut min_cosine_similarity = None::<f64>;
    for sample in samples {
        sample_count += 1;
        let (Some(relative_l2_error), Some(cosine_similarity)) =
            (sample.relative_l2_error, sample.cosine_similarity)
        else {
            continue;
        };
        finite_metric_count += 1;
        max_relative_l2_error = Some(
            max_relative_l2_error
                .map_or(relative_l2_error, |current| current.max(relative_l2_error)),
        );
        min_cosine_similarity = Some(
            min_cosine_similarity
                .map_or(cosine_similarity, |current| current.min(cosine_similarity)),
        );
    }
    TrajectoryAggregateReport {
        sample_count,
        finite_metric_count,
        max_relative_l2_error,
        min_cosine_similarity,
    }
}

fn ranked_logits(logits: &[f32], limit: usize) -> Vec<RankedLogitReport> {
    let mut indices = (0..logits.len()).collect::<Vec<_>>();
    indices.sort_unstable_by(|left, right| {
        logits[*right]
            .total_cmp(&logits[*left])
            .then_with(|| left.cmp(right))
    });
    indices.truncate(limit.min(indices.len()));
    indices
        .into_iter()
        .map(|token_id| RankedLogitReport {
            token_id,
            logit: logits[token_id],
        })
        .collect()
}

fn ranked_overlap(
    actual: &[RankedLogitReport],
    oracle: &[RankedLogitReport],
    limit: usize,
) -> usize {
    let actual = &actual[..limit.min(actual.len())];
    let oracle = &oracle[..limit.min(oracle.len())];
    actual
        .iter()
        .filter(|candidate| {
            oracle
                .iter()
                .any(|entry| entry.token_id == candidate.token_id)
        })
        .count()
}

fn top1_margin(ranked: &[RankedLogitReport]) -> f64 {
    if ranked.len() < 2 {
        return 0.0;
    }
    f64::from(ranked[0].logit) - f64::from(ranked[1].logit)
}

fn max_abs_error(actual: &[f32], expected: &[f32]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (f64::from(*actual) - f64::from(*expected)).abs())
        .fold(0.0_f64, f64::max)
}

fn run_semantic_logits_probe(
    reader: &mut GgufTensorReader<'_>,
    model_config: &Qwen35Config,
    actual: &DeviceSequence,
    resident: &ResidentFastSequence,
    oracle: &ExpectedSequence,
    token_ids: &[usize],
) -> Result<SemanticLogitsReport, Qwen35MixedPrefixCudaError> {
    let actual_layers = actual
        .layers
        .last()
        .ok_or_else(|| invalid("semantic probe has no CUDA layer"))?;
    if actual_layers.len() != token_ids.len()
        || resident.final_hidden.len() != token_ids.len()
        || resident.final_norm.len() != token_ids.len()
        || resident.logits.len() != token_ids.len()
        || oracle.final_hidden.len() != token_ids.len()
    {
        return Err(invalid("semantic probe token extent"));
    }

    let probe_started = Instant::now();
    let setup_started = Instant::now();
    let head = GgufStreamedOutputHead::for_qwen35_with_numerics(
        reader,
        model_config,
        SEMANTIC_OUTPUT_HEAD_CHUNK_ROWS,
        GgufProjectionNumerics::GgmlQ8_0Activation,
    )?;
    let setup_wall_ms = setup_started.elapsed().as_secs_f64() * 1_000.0;
    let mut tokens = Vec::with_capacity(token_ids.len());

    for (position, input_token_id) in token_ids.iter().copied().enumerate() {
        let actual_hidden = &actual_layers[position].final_hidden;
        let oracle_hidden = &oracle.final_hidden[position];
        if !equal_f32_bits(actual_hidden, &resident.final_hidden[position]) {
            return Err(invalid(
                "semantic probe resident and audited hidden identity",
            ));
        }
        let device_normalized = &resident.final_norm[position];
        let device_logits = &resident.logits[position];

        let actual_started = Instant::now();
        let actual_normalized = head.normalized_hidden(actual_hidden)?;
        let actual_logits = head.logits(reader, actual_hidden)?;
        let actual_wall_ms = actual_started.elapsed().as_secs_f64() * 1_000.0;

        let oracle_started = Instant::now();
        let oracle_normalized = head.normalized_hidden(oracle_hidden)?;
        let oracle_logits = head.logits(reader, oracle_hidden)?;
        let oracle_wall_ms = oracle_started.elapsed().as_secs_f64() * 1_000.0;

        let actual_top50 = ranked_logits(&actual_logits, SEMANTIC_TOP_K);
        let oracle_top50 = ranked_logits(&oracle_logits, SEMANTIC_TOP_K);
        let device_top50 = ranked_logits(device_logits, SEMANTIC_TOP_K);
        if actual_top50.len() < 2 || oracle_top50.len() < 2 || device_top50.len() < 2 {
            return Err(invalid("semantic probe vocabulary is too small"));
        }
        let top10_extent = 10.min(actual_top50.len()).min(oracle_top50.len());
        let top50_extent = actual_top50.len().min(oracle_top50.len());
        let top10_overlap_count = ranked_overlap(&actual_top50, &oracle_top50, top10_extent);
        let top50_overlap_count = ranked_overlap(&actual_top50, &oracle_top50, top50_extent);
        let device_oracle_top10_overlap_count =
            ranked_overlap(&device_top50, &oracle_top50, top10_extent);
        let device_oracle_top50_overlap_count =
            ranked_overlap(&device_top50, &oracle_top50, top50_extent);
        let actual_argmax = actual_top50[0].token_id;
        let oracle_argmax = oracle_top50[0].token_id;
        let device_argmax = device_top50[0].token_id;
        let actual_top1_margin = top1_margin(&actual_top50);
        let oracle_top1_margin = top1_margin(&oracle_top50);
        let max_abs_logits_error = max_abs_error(&actual_logits, &oracle_logits);
        let device_operator_max_abs_logits_error = max_abs_error(device_logits, &actual_logits);
        let device_end_to_end_max_abs_logits_error = max_abs_error(device_logits, &oracle_logits);

        tokens.push(SemanticTokenReport {
            position,
            input_token_id,
            actual_final_hidden_sha256: sha256_f32_bits(actual_hidden),
            oracle_final_hidden_sha256: sha256_f32_bits(oracle_hidden),
            actual_normalized_hidden_sha256: sha256_f32_bits(&actual_normalized),
            oracle_normalized_hidden_sha256: sha256_f32_bits(&oracle_normalized),
            actual_logits_sha256: sha256_f32_bits(&actual_logits),
            oracle_logits_sha256: sha256_f32_bits(&oracle_logits),
            device_normalized_hidden_sha256: sha256_f32_bits(device_normalized),
            device_logits_sha256: sha256_f32_bits(device_logits),
            actual_wall_ms,
            oracle_wall_ms,
            normalized_hidden: trajectory_sample(
                &format!("token{position}.normalized_hidden"),
                &actual_normalized,
                &oracle_normalized,
            ),
            logits: trajectory_sample(
                &format!("token{position}.logits"),
                &actual_logits,
                &oracle_logits,
            ),
            device_normalized_hidden_operator: trajectory_sample(
                &format!("token{position}.device_normalized_hidden_operator"),
                device_normalized,
                &actual_normalized,
            ),
            device_logits_operator: trajectory_sample(
                &format!("token{position}.device_logits_operator"),
                device_logits,
                &actual_logits,
            ),
            device_logits_end_to_end: trajectory_sample(
                &format!("token{position}.device_logits_end_to_end"),
                device_logits,
                &oracle_logits,
            ),
            max_abs_logits_error,
            device_operator_max_abs_logits_error,
            device_end_to_end_max_abs_logits_error,
            actual_argmax,
            oracle_argmax,
            device_argmax,
            argmax_equal: actual_argmax == oracle_argmax,
            device_argmax_equals_actual: device_argmax == actual_argmax,
            device_argmax_equals_oracle: device_argmax == oracle_argmax,
            actual_top1_margin,
            oracle_top1_margin,
            oracle_argmax_stability_certified: oracle_top1_margin > 2.0 * max_abs_logits_error,
            device_oracle_argmax_stability_certified: oracle_top1_margin
                > 2.0 * device_end_to_end_max_abs_logits_error,
            top10_overlap_count,
            top10_overlap_ratio: top10_overlap_count as f64 / top10_extent as f64,
            top50_overlap_count,
            top50_overlap_ratio: top50_overlap_count as f64 / top50_extent as f64,
            device_oracle_top10_overlap_count,
            device_oracle_top10_overlap_ratio: device_oracle_top10_overlap_count as f64
                / top10_extent as f64,
            device_oracle_top50_overlap_count,
            device_oracle_top50_overlap_ratio: device_oracle_top50_overlap_count as f64
                / top50_extent as f64,
            actual_top50,
            oracle_top50,
            device_top50,
        });
    }

    Ok(SemanticLogitsReport {
        scope: "offline_cpu_streamed_output_head_excluded_from_resident_fast_path".to_owned(),
        reference: "same_q8_0_activation_streamed_head_on_cuda_and_cpu_final_hidden".to_owned(),
        head_numerics: "ggml_q8_0_activation".to_owned(),
        chunk_rows: head.chunk_rows(),
        chunk_count: head.chunk_count(),
        vocabulary_size: head.vocab_size(),
        setup_wall_ms,
        total_wall_ms: probe_started.elapsed().as_secs_f64() * 1_000.0,
        output_head_executed: true,
        excluded_from_resident_fast_path_timing: true,
        all_logits_finite: tokens.iter().all(|token| {
            token.logits.finite_pair_count == token.logits.elements
                && token.logits.relative_l2_error.is_some()
                && token.logits.cosine_similarity.is_some()
                && token.device_logits_operator.finite_pair_count
                    == token.device_logits_operator.elements
                && token.device_logits_operator.relative_l2_error.is_some()
                && token.device_logits_operator.cosine_similarity.is_some()
                && token.device_logits_end_to_end.finite_pair_count
                    == token.device_logits_end_to_end.elements
                && token.device_logits_end_to_end.relative_l2_error.is_some()
                && token.device_logits_end_to_end.cosine_similarity.is_some()
        }),
        all_argmax_equal: tokens.iter().all(|token| token.argmax_equal),
        all_argmax_stability_certified: tokens
            .iter()
            .all(|token| token.oracle_argmax_stability_certified),
        logits: aggregate_trajectory(tokens.iter().map(|token| &token.logits)),
        device_operator_logits: aggregate_trajectory(
            tokens.iter().map(|token| &token.device_logits_operator),
        ),
        device_end_to_end_logits: aggregate_trajectory(
            tokens.iter().map(|token| &token.device_logits_end_to_end),
        ),
        all_device_argmax_equal: tokens.iter().all(|token| token.device_argmax_equals_oracle),
        all_device_argmax_stability_certified: tokens
            .iter()
            .all(|token| token.device_oracle_argmax_stability_certified),
        llama_cuda_reference_bound: false,
        thresholds_calibrated: false,
        qualification_permitted: false,
        tokens,
    })
}

fn read_f32_buffer(
    stream: &OwnedCudaStream,
    buffer: &OwnedCudaBuffer,
    receipt: augur_backend::CudaStreamSubmissionReceipt,
) -> Result<Vec<f32>, Qwen35MixedPrefixCudaError> {
    let mut bytes = vec![0_u8; buffer.len_bytes()];
    buffer.copy_to_host_sync_with_submission_proof(stream, &mut bytes, receipt)?;
    Ok(f32_vec(&bytes))
}

fn read_full_cache_buffer(
    stream: &OwnedCudaStream,
    buffer: &OwnedCudaBuffer,
    receipt: augur_backend::CudaStreamSubmissionReceipt,
) -> Result<Vec<f32>, Qwen35MixedPrefixCudaError> {
    #[cfg(not(feature = "qwen35-kv8-full-attention"))]
    {
        read_f32_buffer(stream, buffer, receipt)
    }
    #[cfg(feature = "qwen35-kv8-full-attention")]
    {
        read_q8_0_buffer(stream, buffer, receipt)
    }
}

#[cfg(feature = "qwen35-kv8-full-attention")]
fn read_q8_0_buffer(
    stream: &OwnedCudaStream,
    buffer: &OwnedCudaBuffer,
    receipt: augur_backend::CudaStreamSubmissionReceipt,
) -> Result<Vec<f32>, Qwen35MixedPrefixCudaError> {
    if !buffer.len_bytes().is_multiple_of(Q8_0_BLOCK_BYTES) {
        return Err(invalid("Q8_0 cache byte extent"));
    }
    let mut bytes = vec![0_u8; buffer.len_bytes()];
    buffer.copy_to_host_sync_with_submission_proof(stream, &mut bytes, receipt)?;
    let mut output = vec![0.0_f32; bytes.len() / Q8_0_BLOCK_BYTES * Q8_0_BLOCK_ELEMENTS];
    for (block, decoded) in bytes
        .chunks_exact(Q8_0_BLOCK_BYTES)
        .zip(output.chunks_exact_mut(Q8_0_BLOCK_ELEMENTS))
    {
        dequantize_q8_0_block(block, decoded)?;
    }
    Ok(output)
}

#[cfg(feature = "qwen35-prefill-cublas")]
fn read_f32_view(
    stream: &OwnedCudaStream,
    view: CudaBufferView,
    receipt: CudaStreamSubmissionReceipt,
) -> Result<Vec<f32>, Qwen35MixedPrefixCudaError> {
    if !view.len_bytes().is_multiple_of(core::mem::size_of::<f32>()) {
        return Err(invalid("F32 readback view alignment"));
    }
    let mut bytes = vec![0_u8; view.len_bytes()];
    let proof = view.copy_to_host_sync_with_submission_proof(stream, &mut bytes, receipt)?;
    if !proof.covers_submission(receipt) || !proof.belongs_to_stream(stream)? {
        return Err(invalid("F32 readback owner epoch proof"));
    }
    Ok(f32_vec(&bytes))
}

fn update_device_buffer_digest(
    digest: &mut Sha256,
    stream: &OwnedCudaStream,
    buffer: &OwnedCudaBuffer,
    receipt: CudaStreamSubmissionReceipt,
) -> Result<(), Qwen35MixedPrefixCudaError> {
    let mut bytes = vec![0_u8; buffer.len_bytes()];
    buffer.copy_to_host_sync_with_submission_proof(stream, &mut bytes, receipt)?;
    digest.update(bytes);
    Ok(())
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte f32")))
        .collect()
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

#[cfg(feature = "qwen35-device-token-handoff")]
fn hex_bytes(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn sha256_f32_bits(values: &[f32]) -> String {
    sha256_hex(&f32_bytes(values))
}

fn top1_margin_for_argmax(logits: &[f32], argmax: usize) -> f64 {
    if logits.len() < 2 || argmax >= logits.len() {
        return 0.0;
    }
    let mut second = f32::NEG_INFINITY;
    for (index, value) in logits.iter().copied().enumerate() {
        if index != argmax && value > second {
            second = value;
        }
    }
    f64::from(logits[argmax]) - f64::from(second)
}

fn nearest_rank_percentile(values: &[f64], percentile: f64) -> f64 {
    if values.is_empty() || !percentile.is_finite() || !(0.0..=1.0).contains(&percentile) {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let rank = (percentile * sorted.len() as f64).ceil() as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

fn autoregressive_margin_distribution(
    tokens: &[AutoregressiveTokenReport],
) -> AutoregressiveMarginDistributionReport {
    let margins = tokens
        .iter()
        .map(|token| token.top1_margin)
        .collect::<Vec<_>>();
    AutoregressiveMarginDistributionReport {
        minimum: margins.iter().copied().reduce(f64::min).unwrap_or(0.0),
        p01: nearest_rank_percentile(&margins, 0.01),
        p05: nearest_rank_percentile(&margins, 0.05),
        p50: nearest_rank_percentile(&margins, 0.50),
        p95: nearest_rank_percentile(&margins, 0.95),
        below_0_01_count: margins.iter().filter(|margin| **margin < 0.01).count(),
        below_0_05_count: margins.iter().filter(|margin| **margin < 0.05).count(),
        below_0_10_count: margins.iter().filter(|margin| **margin < 0.10).count(),
    }
}

fn autoregressive_sequence_report(
    sequence: &AutoregressiveSequence,
) -> AutoregressiveSequenceReport {
    let host_wall = sequence
        .tokens
        .iter()
        .map(|token| token.host_wall_ms)
        .collect::<Vec<_>>();
    let aggregate_cuda_event_ms = sequence
        .tokens
        .iter()
        .map(|token| token.cuda_event_ms)
        .sum::<f64>();
    let generated_token_count = sequence.tokens.len();
    AutoregressiveSequenceReport {
        prompt_prefix_tokens_processed: sequence.prompt_prefix_tokens_processed,
        batched_prefill: sequence.batched_prefill.clone(),
        generated_token_count,
        aggregate_decode_host_wall_ms: sequence.decode_host_wall_ms,
        aggregate_cuda_event_ms,
        visible_tokens_per_second: generated_token_count as f64 * 1_000.0
            / sequence.decode_host_wall_ms,
        cuda_event_tokens_per_second: generated_token_count as f64 * 1_000.0
            / aggregate_cuda_event_ms,
        p50_token_ms: nearest_rank_percentile(&host_wall, 0.50),
        p95_token_ms: nearest_rank_percentile(&host_wall, 0.95),
        terminal_poll_iterations: sequence.polls,
        owner_epoch_proof: sequence.proof,
        inter_layer_d2d_copy_count: sequence.inter_layer_d2d_copy_count,
        inter_layer_d2d_bytes: sequence.inter_layer_d2d_bytes,
        final_hidden_sha256: sequence.final_hidden_sha256.clone(),
        final_state_sha256: sequence.final_state_sha256.clone(),
        top1_margin_distribution: autoregressive_margin_distribution(&sequence.tokens),
        tokens: sequence.tokens.clone(),
        resident_auxiliary: sequence.resident_auxiliary.clone(),
    }
}

fn zero_copy_layer_handoff_counts_closed(
    enabled: bool,
    primary: (usize, usize),
    replay: (usize, usize),
    device_handoff: Option<(usize, usize)>,
) -> bool {
    !enabled
        || ([primary, replay]
            .into_iter()
            .chain(device_handoff)
            .all(|(copies, bytes)| copies == 0 && bytes == 0))
}

fn read_stable_regular_file(
    path: &std::path::Path,
    label: &str,
) -> Result<Vec<u8>, Qwen35MixedPrefixCudaError> {
    let before = std::fs::symlink_metadata(path).map_err(Qwen35Error::from)?;
    if !before.is_file() || before.file_type().is_symlink() {
        return Err(invalid(format!(
            "{label} must be a regular non-symlink file"
        )));
    }
    let bytes = std::fs::read(path).map_err(Qwen35Error::from)?;
    let after = std::fs::symlink_metadata(path).map_err(Qwen35Error::from)?;
    if before.len() != after.len() || before.modified().ok() != after.modified().ok() {
        return Err(invalid(format!("{label} changed during read")));
    }
    Ok(bytes)
}

fn equal_f32_bits(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

fn convert_boundary(value: linear::BoundaryReport) -> BoundaryReport {
    BoundaryReport {
        host_waits: value.host_waits,
        stream_synchronizations: value.stream_synchronizations,
        event_synchronizations: value.event_synchronizations,
        synchronous_h2d_copies: value.synchronous_h2d_copies,
        synchronous_h2d_bytes: value.synchronous_h2d_bytes,
        synchronous_d2h_copies: value.synchronous_d2h_copies,
        synchronous_d2h_bytes: value.synchronous_d2h_bytes,
    }
}

fn boundary_delta(
    earlier: augur_backend::CudaHostBoundarySnapshot,
    later: augur_backend::CudaHostBoundarySnapshot,
) -> Result<BoundaryReport, Qwen35MixedPrefixCudaError> {
    let delta = later
        .checked_delta_since(earlier)
        .ok_or_else(|| invalid("host boundary counters moved backwards"))?;
    Ok(BoundaryReport {
        host_waits: delta.host_waits,
        stream_synchronizations: delta.stream_synchronizations,
        event_synchronizations: delta.event_synchronizations,
        synchronous_h2d_copies: delta.synchronous_h2d_copies,
        synchronous_h2d_bytes: delta.synchronous_h2d_bytes,
        synchronous_d2h_copies: delta.synchronous_d2h_copies,
        synchronous_d2h_bytes: delta.synchronous_d2h_bytes,
    })
}

fn add_boundaries(left: BoundaryReport, right: BoundaryReport) -> BoundaryReport {
    BoundaryReport {
        host_waits: left.host_waits.saturating_add(right.host_waits),
        stream_synchronizations: left
            .stream_synchronizations
            .saturating_add(right.stream_synchronizations),
        event_synchronizations: left
            .event_synchronizations
            .saturating_add(right.event_synchronizations),
        synchronous_h2d_copies: left
            .synchronous_h2d_copies
            .saturating_add(right.synchronous_h2d_copies),
        synchronous_h2d_bytes: left
            .synchronous_h2d_bytes
            .saturating_add(right.synchronous_h2d_bytes),
        synchronous_d2h_copies: left
            .synchronous_d2h_copies
            .saturating_add(right.synchronous_d2h_copies),
        synchronous_d2h_bytes: left
            .synchronous_d2h_bytes
            .saturating_add(right.synchronous_d2h_bytes),
    }
}

fn allocation_delta(
    earlier: augur_backend::CudaDeviceAllocationSnapshot,
    later: augur_backend::CudaDeviceAllocationSnapshot,
) -> Result<AllocationReport, Qwen35MixedPrefixCudaError> {
    Ok(AllocationReport {
        successful_allocations: later
            .successful_allocations
            .checked_sub(earlier.successful_allocations)
            .ok_or_else(|| invalid("allocation counter moved backwards"))?,
        successful_frees: later
            .successful_frees
            .checked_sub(earlier.successful_frees)
            .ok_or_else(|| invalid("free counter moved backwards"))?,
        live_allocations: later.live_allocations,
        live_bytes: later.live_bytes,
        peak_live_allocations: later.peak_live_allocations,
        peak_live_bytes: later.peak_live_bytes,
    })
}

#[cfg(feature = "qwen35-prefill-cublas")]
fn allocation_snapshot_report(
    snapshot: augur_backend::CudaDeviceAllocationSnapshot,
) -> AllocationReport {
    AllocationReport {
        successful_allocations: snapshot.successful_allocations,
        successful_frees: snapshot.successful_frees,
        live_allocations: snapshot.live_allocations,
        live_bytes: snapshot.live_bytes,
        peak_live_allocations: snapshot.peak_live_allocations,
        peak_live_bytes: snapshot.peak_live_bytes,
    }
}

#[cfg(feature = "qwen35-prefill-cublas")]
fn allocation_interval_report(
    before: augur_backend::CudaDeviceAllocationSnapshot,
    after: augur_backend::CudaDeviceAllocationSnapshot,
) -> Result<AllocationIntervalReport, Qwen35MixedPrefixCudaError> {
    let delta = allocation_delta(before, after)?;
    let live_allocations_unchanged = before.live_allocations == after.live_allocations;
    let live_bytes_unchanged = before.live_bytes == after.live_bytes;
    let pass = delta.successful_allocations == 0
        && delta.successful_frees == 0
        && live_allocations_unchanged
        && live_bytes_unchanged;
    Ok(AllocationIntervalReport {
        before: allocation_snapshot_report(before),
        after: allocation_snapshot_report(after),
        successful_allocations_delta: delta.successful_allocations,
        successful_frees_delta: delta.successful_frees,
        live_allocations_unchanged,
        live_bytes_unchanged,
        intentional_lifecycle_release: false,
        pass,
    })
}

#[cfg(feature = "qwen35-prefill-cublas")]
fn allocation_lifecycle_release_interval_report(
    before: augur_backend::CudaDeviceAllocationSnapshot,
    after: augur_backend::CudaDeviceAllocationSnapshot,
) -> Result<AllocationIntervalReport, Qwen35MixedPrefixCudaError> {
    let delta = allocation_delta(before, after)?;
    let live_allocations_unchanged = before.live_allocations == after.live_allocations;
    let live_bytes_unchanged = before.live_bytes == after.live_bytes;
    let live_set_did_not_grow =
        after.live_allocations <= before.live_allocations && after.live_bytes <= before.live_bytes;
    let pass = delta.successful_allocations == 0 && live_set_did_not_grow;
    Ok(AllocationIntervalReport {
        before: allocation_snapshot_report(before),
        after: allocation_snapshot_report(after),
        successful_allocations_delta: delta.successful_allocations,
        successful_frees_delta: delta.successful_frees,
        live_allocations_unchanged,
        live_bytes_unchanged,
        intentional_lifecycle_release: true,
        pass,
    })
}

#[cfg(feature = "qwen35-prefill-cublas")]
fn allocation_cleanup_report(
    before: augur_backend::CudaDeviceAllocationSnapshot,
    after: augur_backend::CudaDeviceAllocationSnapshot,
) -> Result<AllocationCleanupReport, Qwen35MixedPrefixCudaError> {
    let delta = allocation_delta(before, after)?;
    Ok(AllocationCleanupReport {
        before: allocation_snapshot_report(before),
        after: allocation_snapshot_report(after),
        successful_allocations_delta: delta.successful_allocations,
        successful_frees_delta: delta.successful_frees,
        pass: delta.successful_allocations == delta.successful_frees
            && before.live_allocations == after.live_allocations
            && before.live_bytes == after.live_bytes,
    })
}

fn reset_plans(
    plans: &mut [DevicePlan],
    stream: &OwnedCudaStream,
) -> Result<BoundaryReport, Qwen35MixedPrefixCudaError> {
    let before = cuda_host_boundary_snapshot();
    for plan in plans {
        plan.reset(stream)?;
    }
    boundary_delta(before, cuda_host_boundary_snapshot())
}

#[cfg(feature = "qwen35-prefill-cublas")]
pub fn run_qwen35_prefill_linear_layer_cuda(
    config: &Qwen35PrefillLinearLayerCudaConfig,
) -> Result<Qwen35PrefillLinearLayerCudaReport, Qwen35MixedPrefixCudaError> {
    const LAYER: usize = 0;
    const RELATIVE_L2_MAX: f64 = 0.02;
    const COSINE_SIMILARITY_MIN: f64 = 0.999;

    if config.gguf_path.as_os_str().is_empty() || !config.gguf_path.is_file() {
        return Err(invalid("prefill linear-layer GGUF path"));
    }
    if config.config_path.as_os_str().is_empty() || !config.config_path.is_file() {
        return Err(invalid("prefill linear-layer config path"));
    }
    if config.tokens == 0
        || config.tokens > 512
        || config.device_ordinal < 0
        || config.warmup > 10
        || config.iterations == 0
        || config.iterations > 100
    {
        return Err(invalid("prefill linear-layer scalar bounds"));
    }

    let binary_path = std::env::current_exe().map_err(Qwen35Error::from)?;
    let binary_bytes = std::fs::read(&binary_path).map_err(Qwen35Error::from)?;
    let config_bytes = read_stable_regular_file(&config.config_path, "Qwen config")?;
    let model_config = Qwen35Config::parse(&config_bytes)?;
    model_config.validate_exact_profile()?;
    let manifest = GgufManifest::inspect(&config.gguf_path, GgufAuditOptions::default())?;
    let schema_summary =
        Qwen35TensorSchema::validate_manifest_allowing_mtp(&model_config, &manifest)?;
    if schema_summary.actual_tensor_count < 1_187 || model_config.num_hidden_layers != 64 {
        return Err(invalid("prefill linear-layer audited model profile"));
    }
    let spec = Qwen35LayerSpec::from_config(&model_config, LAYER)?;
    if spec.kind != Qwen35LayerKind::LinearAttention
        || spec.hidden_size != linear::HIDDEN
        || spec.intermediate_size != linear::INTERMEDIATE
    {
        return Err(invalid("prefill linear-layer geometry"));
    }

    let device_properties = cuda_device_properties(config.device_ordinal)?;
    if device_properties.compute_capability.major != 7
        || device_properties.compute_capability.minor != 0
    {
        return Err(invalid("prefill linear-layer diagnostic requires SM70"));
    }
    let device = DeviceReport {
        ordinal: device_properties.ordinal,
        name: device_properties.name,
        compute_capability: [
            device_properties.compute_capability.major as u16,
            device_properties.compute_capability.minor as u16,
        ],
        total_global_memory_bytes: device_properties.total_global_memory_bytes,
    };

    let mut reader = manifest.open_tensor_reader(&config.gguf_path)?;
    let embedding = manifest
        .tensor(linear::EMBEDDING)
        .ok_or_else(|| invalid("prefill linear-layer embedding descriptor"))?
        .clone();
    let usable_vocabulary = model_config
        .vocab_size
        .checked_sub(1)
        .ok_or_else(|| invalid("prefill linear-layer vocabulary"))?;
    let token_ids = (0..config.tokens)
        .map(|index| 1 + (index * 7_919) % usable_vocabulary)
        .collect::<Vec<_>>();
    let mut input_hidden = Vec::with_capacity(
        config
            .tokens
            .checked_mul(linear::HIDDEN)
            .ok_or_else(|| invalid("prefill linear-layer input extent"))?,
    );
    for token_id in &token_ids {
        input_hidden.extend(linear::read_embedding_row(
            &mut reader,
            &embedding,
            *token_id,
        )?);
    }

    let allocation_start = cuda_device_allocation_snapshot();
    let setup_boundary_start = cuda_host_boundary_snapshot();
    let stream = OwnedCudaStream::create(config.device_ordinal)?;
    let materials = load_layer_materials(&mut reader, &manifest, &model_config, 1)?;
    let material = match materials.first() {
        Some(LayerMaterial::Linear(material)) => material,
        _ => return Err(invalid("prefill layer zero is not linear attention")),
    };
    let first_hidden = &input_hidden[..linear::HIDDEN];
    let first_projection =
        project_linear(material, first_hidden, model_config.rms_norm_eps as f32)?;
    let mut plan = build_linear_plan(
        &stream,
        material,
        first_hidden,
        &first_projection,
        true,
        None,
    )?;
    let workspace = PrefillWorkspace::new(&stream, config.tokens)?;
    let operations = workspace.build_linear_ops(
        &stream,
        &plan,
        LAYER,
        config.tokens,
        model_config.rms_norm_eps as f32,
    )?;
    drop(materials);
    let started_event = OwnedCudaEvent::create(config.device_ordinal)?;
    let stopped_event = OwnedCudaEvent::create(config.device_ordinal)?;
    let terminal_event = OwnedCudaEvent::create(config.device_ordinal)?;
    let setup_boundaries = boundary_delta(setup_boundary_start, cuda_host_boundary_snapshot())?;

    let sequential_boundary_start = cuda_host_boundary_snapshot();
    plan.reset(&stream)?;
    let mut sequential_output = Vec::with_capacity(input_hidden.len());
    let mut sequential_state = Vec::new();
    let mut sequential_history = Vec::new();
    for hidden in input_hidden.chunks_exact(linear::HIDDEN) {
        plan.upload(&stream, None, hidden, true)?;
        let readback = plan.execute(&stream)?;
        sequential_output.extend_from_slice(&readback.final_hidden);
        sequential_state = readback.state;
        sequential_history = readback.history;
    }
    if sequential_output.len() != input_hidden.len()
        || sequential_state.len() != linear::STATE_ELEMENTS
        || sequential_history.len() != linear::CONV_HISTORY_ELEMENTS
    {
        return Err(invalid("prefill sequential oracle readback extent"));
    }
    let sequential_oracle_boundaries =
        boundary_delta(sequential_boundary_start, cuda_host_boundary_snapshot())?;

    let batched_boundary_start = cuda_host_boundary_snapshot();
    plan.reset(&stream)?;
    workspace
        .hidden_a
        .copy_from_host_sync(&stream, &f32_bytes(&input_hidden))?;
    let (batched_receipt, numerical_polls, numerical_proof, _) = submit_prefill_linear_with_proof(
        &stream,
        &operations,
        &workspace.cublas,
        &started_event,
        &stopped_event,
        &terminal_event,
    )?;
    let batched_output = read_f32_buffer(&stream, &workspace.hidden_b, batched_receipt)?;
    let batched_state = read_f32_buffer(&stream, &plan.state, batched_receipt)?;
    let batched_history = read_f32_buffer(&stream, &plan.history, batched_receipt)?;
    let batched_numerical_boundaries =
        boundary_delta(batched_boundary_start, cuda_host_boundary_snapshot())?;

    let output = trajectory_sample("final_hidden", &batched_output, &sequential_output);
    let recurrent_state = trajectory_sample("recurrent_state", &batched_state, &sequential_state);
    let convolution_history =
        trajectory_sample("convolution_history", &batched_history, &sequential_history);
    let numerical_pass = numerical_proof
        && prefill_trajectory_pass(&output, RELATIVE_L2_MAX, COSINE_SIMILARITY_MIN)
        && prefill_trajectory_pass(&recurrent_state, RELATIVE_L2_MAX, COSINE_SIMILARITY_MIN)
        && prefill_trajectory_pass(&convolution_history, RELATIVE_L2_MAX, COSINE_SIMILARITY_MIN);

    let mut completion_poll_iterations = numerical_polls;
    let mut owner_epoch_proof = numerical_proof;
    for _ in 0..config.warmup {
        plan.reset(&stream)?;
        let (_, polls, proof, _) = submit_prefill_linear_with_proof(
            &stream,
            &operations,
            &workspace.cublas,
            &started_event,
            &stopped_event,
            &terminal_event,
        )?;
        completion_poll_iterations = completion_poll_iterations.saturating_add(polls);
        owner_epoch_proof &= proof;
    }
    let timed_boundary_start = cuda_host_boundary_snapshot();
    let timed_wall_start = Instant::now();
    let mut cuda_event_total_ms = 0.0_f64;
    for _ in 0..config.iterations {
        plan.reset(&stream)?;
        let (_, polls, proof, cuda_event_ms) = submit_prefill_linear_with_proof(
            &stream,
            &operations,
            &workspace.cublas,
            &started_event,
            &stopped_event,
            &terminal_event,
        )?;
        completion_poll_iterations = completion_poll_iterations.saturating_add(polls);
        owner_epoch_proof &= proof;
        cuda_event_total_ms += cuda_event_ms;
    }
    let host_wall_total_ms = timed_wall_start.elapsed().as_secs_f64() * 1_000.0;
    let timed_boundaries = boundary_delta(timed_boundary_start, cuda_host_boundary_snapshot())?;
    let cuda_event_mean_ms = cuda_event_total_ms / config.iterations as f64;
    let host_wall_mean_ms = host_wall_total_ms / config.iterations as f64;
    if !cuda_event_mean_ms.is_finite() || cuda_event_mean_ms <= 0.0 {
        return Err(invalid("prefill linear-layer CUDA timing"));
    }

    let status = if numerical_pass && owner_epoch_proof {
        "PASS_COMPONENT_QWEN35_PREFILL_LINEAR_LAYER_CUBLAS"
    } else {
        "FAIL_COMPONENT_QWEN35_PREFILL_LINEAR_LAYER_CUBLAS"
    };
    Ok(Qwen35PrefillLinearLayerCudaReport {
        schema: QWEN35_PREFILL_LINEAR_LAYER_CUDA_SCHEMA.to_owned(),
        status: status.to_owned(),
        component_only: true,
        source_commit: config.source_commit.clone(),
        binary_path: binary_path.display().to_string(),
        binary_sha256: sha256_hex(&binary_bytes),
        device,
        config_sha256: sha256_hex(&config_bytes),
        gguf_path: config.gguf_path.display().to_string(),
        gguf_file_sha256: manifest.file_sha256.clone(),
        layer: LAYER,
        token_ids,
        input_hidden_sha256: sha256_f32_bits(&input_hidden),
        numerical: PrefillLinearLayerNumericalReport {
            output,
            recurrent_state,
            convolution_history,
            relative_l2_max: RELATIVE_L2_MAX,
            cosine_similarity_min: COSINE_SIMILARITY_MIN,
            pass: numerical_pass,
        },
        timing: PrefillLayerTimingReport {
            warmup: config.warmup,
            iterations: config.iterations,
            tokens_per_chunk: config.tokens,
            cuda_event_total_ms,
            cuda_event_mean_ms,
            host_wall_total_ms,
            host_wall_mean_ms,
            layer_matrix_tokens_per_second: config.tokens as f64 * 1_000.0 / cuda_event_mean_ms,
            reset_excluded_from_cuda_event: true,
            reset_included_in_host_wall: true,
            input_upload_excluded_from_cuda_event: true,
        },
        resources: PrefillLayerResourceReport {
            setup_boundaries,
            sequential_oracle_boundaries,
            batched_numerical_boundaries,
            timed_boundaries,
            allocation_delta: allocation_delta(
                allocation_start,
                cuda_device_allocation_snapshot(),
            )?,
            completion_poll_iterations,
            owner_epoch_proof,
        },
        claims: PrefillLinearLayerClaimsReport {
            real_model_weights_executed: true,
            complete_linear_layer_executed: true,
            sequential_gpu_oracle_executed: true,
            complete_transformer_executed: false,
            model_prefill_tps_permitted: false,
            semantic_parity_established: false,
            product_tps_permitted: false,
            default_cli_enabled: false,
        },
    })
}

#[cfg(feature = "qwen35-prefill-cublas")]
fn submit_prefill_linear_with_proof(
    stream: &OwnedCudaStream,
    operations: &PrefillLinearOps,
    cublas: &CudaQwen35PrefillCublasPlanV1,
    started_event: &OwnedCudaEvent,
    stopped_event: &OwnedCudaEvent,
    terminal_event: &OwnedCudaEvent,
) -> Result<(CudaStreamSubmissionReceipt, u64, bool, f64), Qwen35MixedPrefixCudaError> {
    started_event.record(stream)?;
    operations.submit(stream, cublas)?;
    stopped_event.record(stream)?;
    let receipt = stream.submission_receipt()?;
    terminal_event.record_for_submission(stream, receipt)?;
    let mut polls = 0_u64;
    let completion = loop {
        if let Some(completion) = stream.query_event_completion(terminal_event, receipt)? {
            break completion;
        }
        polls = polls.saturating_add(1);
        if polls >= MAX_COMPLETION_POLLS {
            return Err(Qwen35MixedPrefixCudaError::CompletionPollLimit);
        }
        bounded_completion_poll_backoff(polls);
    };
    let proof = completion.covers_submission(receipt) && completion.belongs_to_stream(stream)?;
    let cuda_event_ms = f64::from(started_event.elapsed_ms(stopped_event)?);
    Ok((receipt, polls, proof, cuda_event_ms))
}

#[cfg(feature = "qwen35-prefill-cublas")]
fn prefill_trajectory_pass(
    sample: &TrajectorySampleReport,
    relative_l2_max: f64,
    cosine_similarity_min: f64,
) -> bool {
    sample.shape_match
        && sample.elements > 0
        && sample.finite_pair_count == sample.elements
        && sample
            .relative_l2_error
            .is_some_and(|value| value <= relative_l2_max)
        && sample
            .cosine_similarity
            .is_some_and(|value| value >= cosine_similarity_min)
}

#[cfg(feature = "qwen35-prefill-cublas")]
pub fn run_qwen35_prefill_full_layer_cuda(
    config: &Qwen35PrefillFullLayerCudaConfig,
) -> Result<Qwen35PrefillFullLayerCudaReport, Qwen35MixedPrefixCudaError> {
    const LAYER: usize = 3;
    const RELATIVE_L2_MAX: f64 = 0.02;
    const COSINE_SIMILARITY_MIN: f64 = 0.999;

    if config.gguf_path.as_os_str().is_empty() || !config.gguf_path.is_file() {
        return Err(invalid("prefill full-layer GGUF path"));
    }
    if config.config_path.as_os_str().is_empty() || !config.config_path.is_file() {
        return Err(invalid("prefill full-layer config path"));
    }
    if config.tokens == 0
        || config.tokens > 512
        || config.device_ordinal < 0
        || config.warmup > 10
        || config.iterations == 0
        || config.iterations > 100
    {
        return Err(invalid("prefill full-layer scalar bounds"));
    }

    let binary_path = std::env::current_exe().map_err(Qwen35Error::from)?;
    let binary_bytes = std::fs::read(&binary_path).map_err(Qwen35Error::from)?;
    let config_bytes = read_stable_regular_file(&config.config_path, "Qwen config")?;
    let model_config = Qwen35Config::parse(&config_bytes)?;
    model_config.validate_exact_profile()?;
    let manifest = GgufManifest::inspect(&config.gguf_path, GgufAuditOptions::default())?;
    let schema_summary =
        Qwen35TensorSchema::validate_manifest_allowing_mtp(&model_config, &manifest)?;
    if schema_summary.actual_tensor_count < 1_187 || model_config.num_hidden_layers != 64 {
        return Err(invalid("prefill full-layer audited model profile"));
    }
    let spec = Qwen35LayerSpec::from_config(&model_config, LAYER)?;
    if spec.kind != Qwen35LayerKind::FullAttention
        || spec.hidden_size != linear::HIDDEN
        || spec.intermediate_size != linear::INTERMEDIATE
    {
        return Err(invalid("prefill full-layer geometry"));
    }

    let device_properties = cuda_device_properties(config.device_ordinal)?;
    if device_properties.compute_capability.major != 7
        || device_properties.compute_capability.minor != 0
    {
        return Err(invalid("prefill full-layer diagnostic requires SM70"));
    }
    let device = DeviceReport {
        ordinal: device_properties.ordinal,
        name: device_properties.name,
        compute_capability: [
            device_properties.compute_capability.major as u16,
            device_properties.compute_capability.minor as u16,
        ],
        total_global_memory_bytes: device_properties.total_global_memory_bytes,
    };

    let mut reader = manifest.open_tensor_reader(&config.gguf_path)?;
    let embedding = manifest
        .tensor(linear::EMBEDDING)
        .ok_or_else(|| invalid("prefill full-layer embedding descriptor"))?
        .clone();
    let usable_vocabulary = model_config
        .vocab_size
        .checked_sub(1)
        .ok_or_else(|| invalid("prefill full-layer vocabulary"))?;
    let token_ids = (0..config.tokens)
        .map(|index| 1 + (index * 7_919) % usable_vocabulary)
        .collect::<Vec<_>>();
    let mut input_hidden = Vec::with_capacity(
        config
            .tokens
            .checked_mul(linear::HIDDEN)
            .ok_or_else(|| invalid("prefill full-layer input extent"))?,
    );
    for token_id in &token_ids {
        input_hidden.extend(linear::read_embedding_row(
            &mut reader,
            &embedding,
            *token_id,
        )?);
    }

    let allocation_start = cuda_device_allocation_snapshot();
    let setup_boundary_start = cuda_host_boundary_snapshot();
    let stream = OwnedCudaStream::create(config.device_ordinal)?;
    let materials = load_layer_materials(&mut reader, &manifest, &model_config, LAYER + 1)?;
    let material = match materials.get(LAYER) {
        Some(LayerMaterial::Full(material)) => material,
        _ => return Err(invalid("prefill layer three is not full attention")),
    };
    let mut plan = build_full_plan(
        &stream,
        material,
        &model_config,
        config.tokens,
        config.tokens,
        None,
    )?;
    let workspace = PrefillWorkspace::new(&stream, config.tokens)?;
    workspace.upload_rope_tables(&stream, config.tokens, 0, model_config.rope_theta as f32)?;
    let operations = workspace.build_full_ops(
        &stream,
        &plan,
        LAYER,
        config.tokens,
        0,
        model_config.rms_norm_eps as f32,
    )?;
    drop(materials);
    let started_event = OwnedCudaEvent::create(config.device_ordinal)?;
    let stopped_event = OwnedCudaEvent::create(config.device_ordinal)?;
    let terminal_event = OwnedCudaEvent::create(config.device_ordinal)?;
    let setup_boundaries = boundary_delta(setup_boundary_start, cuda_host_boundary_snapshot())?;

    let sequential_boundary_start = cuda_host_boundary_snapshot();
    plan.reset(&stream)?;
    let mut sequential_output = Vec::with_capacity(input_hidden.len());
    let mut sequential_state = Vec::new();
    for (position, hidden) in input_hidden.chunks_exact(linear::HIDDEN).enumerate() {
        plan.upload(&stream, hidden)?;
        let readback = plan.execute(&stream, position)?;
        sequential_output.extend_from_slice(&readback.final_hidden);
        sequential_state = readback.state;
    }
    let cache_elements = config
        .tokens
        .checked_mul(full::KV_VALUES)
        .ok_or_else(|| invalid("prefill full-layer cache extent overflow"))?;
    if sequential_output.len() != input_hidden.len() || sequential_state.len() != cache_elements * 2
    {
        return Err(invalid("prefill full-layer sequential oracle extent"));
    }
    let sequential_oracle_boundaries =
        boundary_delta(sequential_boundary_start, cuda_host_boundary_snapshot())?;

    let batched_boundary_start = cuda_host_boundary_snapshot();
    plan.reset(&stream)?;
    workspace
        .hidden_b
        .copy_from_host_sync(&stream, &f32_bytes(&input_hidden))?;
    let (batched_receipt, numerical_polls, numerical_proof, _) = submit_prefill_full_with_proof(
        &stream,
        &operations,
        &workspace.cublas,
        &started_event,
        &stopped_event,
        &terminal_event,
    )?;
    let batched_output = read_f32_buffer(&stream, &workspace.hidden_a, batched_receipt)?;
    let batched_key_cache = read_full_cache_buffer(&stream, &plan.key_cache, batched_receipt)?;
    let batched_value_cache = read_full_cache_buffer(&stream, &plan.value_cache, batched_receipt)?;
    let batched_numerical_boundaries =
        boundary_delta(batched_boundary_start, cuda_host_boundary_snapshot())?;

    let output = trajectory_sample("final_hidden", &batched_output, &sequential_output);
    let key_cache = trajectory_sample(
        "key_cache",
        &batched_key_cache,
        &sequential_state[..cache_elements],
    );
    let value_cache = trajectory_sample(
        "value_cache",
        &batched_value_cache,
        &sequential_state[cache_elements..],
    );
    let numerical_pass = numerical_proof
        && prefill_trajectory_pass(&output, RELATIVE_L2_MAX, COSINE_SIMILARITY_MIN)
        && prefill_trajectory_pass(&key_cache, RELATIVE_L2_MAX, COSINE_SIMILARITY_MIN)
        && prefill_trajectory_pass(&value_cache, RELATIVE_L2_MAX, COSINE_SIMILARITY_MIN);

    let mut completion_poll_iterations = numerical_polls;
    let mut owner_epoch_proof = numerical_proof;
    for _ in 0..config.warmup {
        plan.reset(&stream)?;
        let (_, polls, proof, _) = submit_prefill_full_with_proof(
            &stream,
            &operations,
            &workspace.cublas,
            &started_event,
            &stopped_event,
            &terminal_event,
        )?;
        completion_poll_iterations = completion_poll_iterations.saturating_add(polls);
        owner_epoch_proof &= proof;
    }
    let timed_boundary_start = cuda_host_boundary_snapshot();
    let timed_wall_start = Instant::now();
    let mut cuda_event_total_ms = 0.0_f64;
    for _ in 0..config.iterations {
        plan.reset(&stream)?;
        let (_, polls, proof, cuda_event_ms) = submit_prefill_full_with_proof(
            &stream,
            &operations,
            &workspace.cublas,
            &started_event,
            &stopped_event,
            &terminal_event,
        )?;
        completion_poll_iterations = completion_poll_iterations.saturating_add(polls);
        owner_epoch_proof &= proof;
        cuda_event_total_ms += cuda_event_ms;
    }
    let host_wall_total_ms = timed_wall_start.elapsed().as_secs_f64() * 1_000.0;
    let timed_boundaries = boundary_delta(timed_boundary_start, cuda_host_boundary_snapshot())?;
    let cuda_event_mean_ms = cuda_event_total_ms / config.iterations as f64;
    let host_wall_mean_ms = host_wall_total_ms / config.iterations as f64;
    if !cuda_event_mean_ms.is_finite() || cuda_event_mean_ms <= 0.0 {
        return Err(invalid("prefill full-layer CUDA timing"));
    }

    let status = if numerical_pass && owner_epoch_proof {
        "PASS_COMPONENT_QWEN35_PREFILL_FULL_LAYER_CUBLAS"
    } else {
        "FAIL_COMPONENT_QWEN35_PREFILL_FULL_LAYER_CUBLAS"
    };
    Ok(Qwen35PrefillFullLayerCudaReport {
        schema: QWEN35_PREFILL_FULL_LAYER_CUDA_SCHEMA.to_owned(),
        status: status.to_owned(),
        component_only: true,
        source_commit: config.source_commit.clone(),
        binary_path: binary_path.display().to_string(),
        binary_sha256: sha256_hex(&binary_bytes),
        device,
        config_sha256: sha256_hex(&config_bytes),
        gguf_path: config.gguf_path.display().to_string(),
        gguf_file_sha256: manifest.file_sha256.clone(),
        layer: LAYER,
        token_ids,
        input_hidden_sha256: sha256_f32_bits(&input_hidden),
        numerical: PrefillFullLayerNumericalReport {
            output,
            key_cache,
            value_cache,
            relative_l2_max: RELATIVE_L2_MAX,
            cosine_similarity_min: COSINE_SIMILARITY_MIN,
            pass: numerical_pass,
        },
        timing: PrefillLayerTimingReport {
            warmup: config.warmup,
            iterations: config.iterations,
            tokens_per_chunk: config.tokens,
            cuda_event_total_ms,
            cuda_event_mean_ms,
            host_wall_total_ms,
            host_wall_mean_ms,
            layer_matrix_tokens_per_second: config.tokens as f64 * 1_000.0 / cuda_event_mean_ms,
            reset_excluded_from_cuda_event: true,
            reset_included_in_host_wall: true,
            input_upload_excluded_from_cuda_event: true,
        },
        resources: PrefillLayerResourceReport {
            setup_boundaries,
            sequential_oracle_boundaries,
            batched_numerical_boundaries,
            timed_boundaries,
            allocation_delta: allocation_delta(
                allocation_start,
                cuda_device_allocation_snapshot(),
            )?,
            completion_poll_iterations,
            owner_epoch_proof,
        },
        claims: PrefillFullLayerClaimsReport {
            real_model_weights_executed: true,
            complete_full_attention_layer_executed: true,
            batched_causal_gqa_executed: true,
            sequential_gpu_oracle_executed: true,
            complete_transformer_executed: false,
            model_prefill_tps_permitted: false,
            semantic_parity_established: false,
            product_tps_permitted: false,
            default_cli_enabled: false,
        },
    })
}























#[cfg(feature = "qwen35-prefill-cublas")]
fn submit_prefill_full_with_proof(
    stream: &OwnedCudaStream,
    operations: &PrefillFullOps,
    cublas: &CudaQwen35PrefillCublasPlanV1,
    started_event: &OwnedCudaEvent,
    stopped_event: &OwnedCudaEvent,
    terminal_event: &OwnedCudaEvent,
) -> Result<(CudaStreamSubmissionReceipt, u64, bool, f64), Qwen35MixedPrefixCudaError> {
    submit_prefill_full_with_proof_budget(
        stream,
        operations,
        cublas,
        started_event,
        stopped_event,
        terminal_event,
        MAX_COMPLETION_POLLS,
    )
}

#[cfg(feature = "qwen35-prefill-cublas")]
fn submit_prefill_full_with_proof_budget(
    stream: &OwnedCudaStream,
    operations: &PrefillFullOps,
    cublas: &CudaQwen35PrefillCublasPlanV1,
    started_event: &OwnedCudaEvent,
    stopped_event: &OwnedCudaEvent,
    terminal_event: &OwnedCudaEvent,
    poll_limit: u64,
) -> Result<(CudaStreamSubmissionReceipt, u64, bool, f64), Qwen35MixedPrefixCudaError> {
    if poll_limit == 0 {
        return Err(invalid("prefill completion poll budget"));
    }
    started_event.record(stream)?;
    operations.submit(stream, cublas)?;
    stopped_event.record(stream)?;
    let receipt = stream.submission_receipt()?;
    terminal_event.record_for_submission(stream, receipt)?;
    let mut polls = 0_u64;
    let completion = loop {
        if let Some(completion) = stream.query_event_completion(terminal_event, receipt)? {
            break completion;
        }
        polls = polls.saturating_add(1);
        if polls >= poll_limit {
            return Err(Qwen35MixedPrefixCudaError::CompletionPollLimit);
        }
        bounded_completion_poll_backoff(polls);
    };
    let proof = completion.covers_submission(receipt) && completion.belongs_to_stream(stream)?;
    let cuda_event_ms = f64::from(started_event.elapsed_ms(stopped_event)?);
    Ok((receipt, polls, proof, cuda_event_ms))
}

#[cfg(feature = "qwen35-prefill-cublas")]
fn read_prefill_backbone_states(
    plans: &[DevicePlan],
    stream: &OwnedCudaStream,
    receipt: CudaStreamSubmissionReceipt,
) -> Result<Vec<(String, Vec<f32>)>, Qwen35MixedPrefixCudaError> {
    let mut states = Vec::with_capacity(plans.len() * 2);
    for (layer, plan) in plans.iter().enumerate() {
        match plan {
            DevicePlan::Linear(plan) => {
                states.push((
                    format!("layer{layer}.recurrent_state"),
                    read_f32_buffer(stream, &plan.state, receipt)?,
                ));
                states.push((
                    format!("layer{layer}.convolution_history"),
                    read_f32_buffer(stream, &plan.history, receipt)?,
                ));
            }
            DevicePlan::Full(plan) => {
                states.push((
                    format!("layer{layer}.key_cache"),
                    read_full_cache_buffer(stream, &plan.key_cache, receipt)?,
                ));
                states.push((
                    format!("layer{layer}.value_cache"),
                    read_full_cache_buffer(stream, &plan.value_cache, receipt)?,
                ));
            }
        }
    }
    Ok(states)
}

#[cfg(feature = "qwen35-prefill-cublas")]
fn run_prefill_backbone_sequential_oracle(
    plans: &mut [DevicePlan],
    stream: &OwnedCudaStream,
    terminal_event: &OwnedCudaEvent,
    input_hidden: &[f32],
    tokens: usize,
) -> Result<PrefillBackboneOracle, Qwen35MixedPrefixCudaError> {
    if plans.len() != FULL_TARGET_LAYER_COUNT
        || input_hidden.len()
            != tokens
                .checked_mul(linear::HIDDEN)
                .ok_or_else(|| invalid("prefill oracle input extent overflow"))?
    {
        return Err(invalid("prefill oracle geometry"));
    }

    let _ = reset_plans(plans, stream)?;
    let mut final_hidden = Vec::with_capacity(input_hidden.len());
    let mut completion_poll_iterations = 0_u64;
    let mut owner_epoch_proof = true;
    let mut last_receipt = None;
    for (position, hidden) in input_hidden.chunks_exact(linear::HIDDEN).enumerate() {
        plans[0].upload(stream, None, hidden, true)?;
        for layer in 0..plans.len() {
            if layer > 0 {
                let source = plans[layer - 1].final_hidden_view()?;
                let _ = plans[layer].copy_input_from_device(stream, source)?;
            }
            plans[layer].submit_resident(stream, position)?;
        }
        let receipt = stream.submission_receipt()?;
        terminal_event.record_for_submission(stream, receipt)?;
        let mut polls = 0_u64;
        let completion = loop {
            if let Some(completion) = stream.query_event_completion(terminal_event, receipt)? {
                break completion;
            }
            polls = polls.saturating_add(1);
            if polls >= MAX_COMPLETION_POLLS {
                return Err(Qwen35MixedPrefixCudaError::CompletionPollLimit);
            }
            bounded_completion_poll_backoff(polls);
        };
        let proof =
            completion.covers_submission(receipt) && completion.belongs_to_stream(stream)?;
        completion_poll_iterations = completion_poll_iterations.saturating_add(polls);
        owner_epoch_proof &= proof;
        final_hidden.extend(read_f32_buffer(
            stream,
            plans
                .last()
                .ok_or_else(|| invalid("prefill oracle terminal layer"))?
                .final_hidden_buffer(),
            receipt,
        )?);
        last_receipt = Some(receipt);
    }

    let states = read_prefill_backbone_states(
        plans,
        stream,
        last_receipt.ok_or_else(|| invalid("prefill oracle terminal receipt"))?,
    )?;
    Ok(PrefillBackboneOracle {
        final_hidden,
        states,
        completion_poll_iterations,
        owner_epoch_proof,
    })
}

#[cfg(feature = "qwen35-prefill-cublas")]
fn submit_prefill_backbone_with_proof(
    stream: &OwnedCudaStream,
    operations: &[PrefillLayerOps],
    cublas: &CudaQwen35PrefillCublasPlanV1,
    terminal_source: CudaBufferView,
    terminal_hidden: &OwnedCudaBuffer,
    output_head: &DeviceOutputHeadPlan,
    started_event: &OwnedCudaEvent,
    stopped_event: &OwnedCudaEvent,
    terminal_event: &OwnedCudaEvent,
) -> Result<(CudaStreamSubmissionReceipt, u64, bool, f64), Qwen35MixedPrefixCudaError> {
    if operations.len() != FULL_TARGET_LAYER_COUNT {
        return Err(invalid("prefill backbone operation count"));
    }
    started_event.record(stream)?;
    for operation in operations {
        operation.submit(stream, cublas)?;
    }
    // SAFETY: the source is the final row of the shared batched workspace and
    // the destination is a disjoint plan-owned buffer. Both remain live until
    // the terminal event proves this same-stream submission complete.
    unsafe {
        terminal_hidden.copy_from_device_async_unchecked(stream, 0, terminal_source)?;
    }
    output_head.submit(stream)?;
    stopped_event.record(stream)?;
    let receipt = stream.submission_receipt()?;
    terminal_event.record_for_submission(stream, receipt)?;
    let mut polls = 0_u64;
    let completion = loop {
        if let Some(completion) = stream.query_event_completion(terminal_event, receipt)? {
            break completion;
        }
        polls = polls.saturating_add(1);
        if polls >= MAX_COMPLETION_POLLS {
            return Err(Qwen35MixedPrefixCudaError::CompletionPollLimit);
        }
        bounded_completion_poll_backoff(polls);
    };
    let proof = completion.covers_submission(receipt) && completion.belongs_to_stream(stream)?;
    let cuda_event_ms = f64::from(started_event.elapsed_ms(stopped_event)?);
    Ok((receipt, polls, proof, cuda_event_ms))
}

/// Execute the real 64-layer text backbone twice: first through the existing
/// one-token CUDA path as an oracle, then as one batched causal chunk. The
/// measured batched path includes final norm and full-vocabulary output head
/// execution for the terminal prompt token.
#[cfg(feature = "qwen35-prefill-cublas")]
pub fn run_qwen35_prefill_backbone_cuda(
    config: &Qwen35PrefillBackboneCudaConfig,
) -> Result<Qwen35PrefillBackboneCudaReport, Qwen35MixedPrefixCudaError> {
    const FINAL_HIDDEN_RELATIVE_L2_MAX: f64 = 0.05;
    const STATE_RELATIVE_L2_MAX: f64 = 0.05;
    const COSINE_SIMILARITY_MIN: f64 = 0.995;

    if config.gguf_path.as_os_str().is_empty() || !config.gguf_path.is_file() {
        return Err(invalid("prefill backbone GGUF path"));
    }
    if config.config_path.as_os_str().is_empty() || !config.config_path.is_file() {
        return Err(invalid("prefill backbone config path"));
    }
    if config.tokens == 0
        || config.tokens > 512
        || config.device_ordinal < 0
        || config.warmup > 10
        || config.iterations == 0
        || config.iterations > 100
    {
        return Err(invalid("prefill backbone scalar bounds"));
    }

    let binary_path = std::env::current_exe().map_err(Qwen35Error::from)?;
    let binary_bytes = std::fs::read(&binary_path).map_err(Qwen35Error::from)?;
    let config_bytes = read_stable_regular_file(&config.config_path, "Qwen config")?;
    let model_config = Qwen35Config::parse(&config_bytes)?;
    model_config.validate_exact_profile()?;
    let manifest = GgufManifest::inspect(&config.gguf_path, GgufAuditOptions::default())?;
    let schema_summary =
        Qwen35TensorSchema::validate_manifest_allowing_mtp(&model_config, &manifest)?;
    let specs = (0..FULL_TARGET_LAYER_COUNT)
        .map(|layer| Qwen35LayerSpec::from_config(&model_config, layer))
        .collect::<Result<Vec<_>, _>>()?;
    let linear_layer_count = specs
        .iter()
        .filter(|spec| spec.kind == Qwen35LayerKind::LinearAttention)
        .count();
    let full_attention_layer_count = specs
        .iter()
        .filter(|spec| spec.kind == Qwen35LayerKind::FullAttention)
        .count();
    if schema_summary.actual_tensor_count < 1_187
        || model_config.num_hidden_layers != FULL_TARGET_LAYER_COUNT
        || linear_layer_count != 48
        || full_attention_layer_count != 16
        || specs.iter().any(|spec| {
            spec.hidden_size != linear::HIDDEN || spec.intermediate_size != linear::INTERMEDIATE
        })
    {
        return Err(invalid("prefill backbone audited model profile"));
    }

    let device_properties = cuda_device_properties(config.device_ordinal)?;
    if device_properties.compute_capability.major != 7
        || device_properties.compute_capability.minor != 0
    {
        return Err(invalid("prefill backbone diagnostic requires SM70"));
    }
    let device = DeviceReport {
        ordinal: device_properties.ordinal,
        name: device_properties.name,
        compute_capability: [
            device_properties.compute_capability.major as u16,
            device_properties.compute_capability.minor as u16,
        ],
        total_global_memory_bytes: device_properties.total_global_memory_bytes,
    };

    let mut reader = manifest.open_tensor_reader(&config.gguf_path)?;
    let embedding = manifest
        .tensor(linear::EMBEDDING)
        .ok_or_else(|| invalid("prefill backbone embedding descriptor"))?
        .clone();
    let usable_vocabulary = model_config
        .vocab_size
        .checked_sub(1)
        .ok_or_else(|| invalid("prefill backbone vocabulary"))?;
    let token_ids = (0..config.tokens)
        .map(|index| 1 + (index * 7_919) % usable_vocabulary)
        .collect::<Vec<_>>();
    let mut input_hidden = Vec::with_capacity(
        config
            .tokens
            .checked_mul(linear::HIDDEN)
            .ok_or_else(|| invalid("prefill backbone input extent"))?,
    );
    for token_id in &token_ids {
        input_hidden.extend(linear::read_embedding_row(
            &mut reader,
            &embedding,
            *token_id,
        )?);
    }
    let first_hidden = input_hidden
        .get(..linear::HIDDEN)
        .ok_or_else(|| invalid("prefill backbone first hidden"))?;

    let allocation_start = cuda_device_allocation_snapshot();
    let setup_boundary_start = cuda_host_boundary_snapshot();
    let stream = OwnedCudaStream::create(config.device_ordinal)?;
    let materials = load_layer_materials(
        &mut reader,
        &manifest,
        &model_config,
        FULL_TARGET_LAYER_COUNT,
    )?;
    let output_head_material = load_output_head_material(&mut reader, &manifest, &model_config)?;
    let streamed_output_head = GgufStreamedOutputHead::for_qwen35_with_numerics(
        &mut reader,
        &model_config,
        SEMANTIC_OUTPUT_HEAD_CHUNK_ROWS,
        GgufProjectionNumerics::GgmlQ8_0Activation,
    )?;
    let zero_projection = linear::ProjectedToken {
        qkv: vec![0.0; linear::QKV_ELEMENTS],
        z: vec![0.0; linear::RECURRENT_ELEMENTS],
        decay_gate: vec![0.0; linear::VALUE_HEADS],
        beta: vec![0.0; linear::VALUE_HEADS],
    };
    let mut plans = Vec::with_capacity(FULL_TARGET_LAYER_COUNT);
    for material in &materials {
        match material {
            LayerMaterial::Linear(material) => plans.push(DevicePlan::Linear(build_linear_plan(
                &stream,
                material,
                first_hidden,
                &zero_projection,
                true,
                None,
            )?)),
            LayerMaterial::Full(material) => plans.push(DevicePlan::Full(build_full_plan(
                &stream,
                material,
                &model_config,
                config.tokens,
                config.tokens,
                None,
            )?)),
        }
    }
    let workspace = PrefillWorkspace::new(&stream, config.tokens)?;
    workspace.upload_rope_tables(&stream, config.tokens, 0, model_config.rope_theta as f32)?;
    let terminal_hidden = alloc_f32(config.device_ordinal, linear::HIDDEN)?;
    let terminal_row_bytes = linear::HIDDEN
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| invalid("prefill terminal hidden extent overflow"))?;
    let terminal_source = workspace.hidden_a.view(
        config
            .tokens
            .checked_sub(1)
            .and_then(|row| row.checked_mul(terminal_row_bytes))
            .ok_or_else(|| invalid("prefill terminal hidden offset overflow"))?,
        terminal_row_bytes,
    )?;
    let output_head = build_device_output_head_plan(
        &stream,
        &output_head_material,
        terminal_hidden.view(0, terminal_row_bytes)?,
        model_config.hidden_size,
        model_config.vocab_size,
        model_config.rms_norm_eps as f32,
    )?;
    let operations = plans
        .iter()
        .enumerate()
        .map(|(layer, plan)| match plan {
            DevicePlan::Linear(plan) => workspace
                .build_linear_ops(
                    &stream,
                    plan,
                    layer,
                    config.tokens,
                    model_config.rms_norm_eps as f32,
                )
                .map(PrefillLayerOps::Linear),
            DevicePlan::Full(plan) => workspace
                .build_full_ops(
                    &stream,
                    plan,
                    layer,
                    config.tokens,
                    0,
                    model_config.rms_norm_eps as f32,
                )
                .map(PrefillLayerOps::Full),
        })
        .collect::<Result<Vec<_>, _>>()?;
    drop(output_head_material);
    drop(materials);
    let started_event = OwnedCudaEvent::create(config.device_ordinal)?;
    let stopped_event = OwnedCudaEvent::create(config.device_ordinal)?;
    let terminal_event = OwnedCudaEvent::create(config.device_ordinal)?;
    let setup_boundaries = boundary_delta(setup_boundary_start, cuda_host_boundary_snapshot())?;

    let sequential_boundary_start = cuda_host_boundary_snapshot();
    let sequential = run_prefill_backbone_sequential_oracle(
        &mut plans,
        &stream,
        &terminal_event,
        &input_hidden,
        config.tokens,
    )?;
    let sequential_oracle_boundaries =
        boundary_delta(sequential_boundary_start, cuda_host_boundary_snapshot())?;

    let batched_boundary_start = cuda_host_boundary_snapshot();
    let _ = reset_plans(&mut plans, &stream)?;
    workspace
        .hidden_a
        .copy_from_host_sync(&stream, &f32_bytes(&input_hidden))?;
    let (batched_receipt, numerical_polls, numerical_proof, _) =
        submit_prefill_backbone_with_proof(
            &stream,
            &operations,
            &workspace.cublas,
            terminal_source,
            &terminal_hidden,
            &output_head,
            &started_event,
            &stopped_event,
            &terminal_event,
        )?;
    let batched_output = read_f32_buffer(&stream, &workspace.hidden_a, batched_receipt)?;
    let batched_states = read_prefill_backbone_states(&plans, &stream, batched_receipt)?;
    let device_normalized_hidden =
        read_f32_buffer(&stream, &output_head.normalized_hidden, batched_receipt)?;
    let device_logits = read_f32_buffer(&stream, &output_head.logits, batched_receipt)?;
    let batched_numerical_boundaries =
        boundary_delta(batched_boundary_start, cuda_host_boundary_snapshot())?;

    if batched_states.len() != sequential.states.len()
        || batched_states
            .iter()
            .zip(&sequential.states)
            .any(|(actual, expected)| actual.0 != expected.0)
    {
        return Err(invalid("prefill backbone state inventory"));
    }
    let final_hidden = trajectory_sample("final_hidden", &batched_output, &sequential.final_hidden);
    let states = batched_states
        .iter()
        .zip(&sequential.states)
        .map(|(actual, expected)| trajectory_sample(&actual.0, &actual.1, &expected.1))
        .collect::<Vec<_>>();
    let state_aggregate = aggregate_trajectory(states.iter());
    let terminal_element_offset = config
        .tokens
        .checked_sub(1)
        .and_then(|row| row.checked_mul(linear::HIDDEN))
        .ok_or_else(|| invalid("prefill semantic terminal offset overflow"))?;
    let batched_terminal_hidden = batched_output
        .get(terminal_element_offset..terminal_element_offset + linear::HIDDEN)
        .ok_or_else(|| invalid("prefill batched terminal hidden extent"))?;
    let sequential_terminal_hidden = sequential
        .final_hidden
        .get(terminal_element_offset..terminal_element_offset + linear::HIDDEN)
        .ok_or_else(|| invalid("prefill sequential terminal hidden extent"))?;
    let batched_normalized_hidden =
        streamed_output_head.normalized_hidden(batched_terminal_hidden)?;
    let batched_logits = streamed_output_head.logits(&mut reader, batched_terminal_hidden)?;
    let sequential_logits = streamed_output_head.logits(&mut reader, sequential_terminal_hidden)?;
    let normalized_hidden_operator = trajectory_sample(
        "terminal.normalized_hidden_operator",
        &device_normalized_hidden,
        &batched_normalized_hidden,
    );
    let logits_operator =
        trajectory_sample("terminal.logits_operator", &device_logits, &batched_logits);
    let logits_backbone = trajectory_sample(
        "terminal.logits_backbone",
        &batched_logits,
        &sequential_logits,
    );
    let logits_end_to_end = trajectory_sample(
        "terminal.logits_end_to_end",
        &device_logits,
        &sequential_logits,
    );
    let batched_argmax = usize::try_from(greedy_argmax(&batched_logits, model_config.vocab_size)?)
        .map_err(|_| invalid("prefill batched argmax overflow"))?;
    let sequential_argmax =
        usize::try_from(greedy_argmax(&sequential_logits, model_config.vocab_size)?)
            .map_err(|_| invalid("prefill sequential argmax overflow"))?;
    let device_argmax = usize::try_from(greedy_argmax(&device_logits, model_config.vocab_size)?)
        .map_err(|_| invalid("prefill device argmax overflow"))?;
    let all_argmax_equal =
        batched_argmax == sequential_argmax && device_argmax == sequential_argmax;
    let sequential_top1_margin = top1_margin_for_argmax(&sequential_logits, sequential_argmax);
    let device_end_to_end_max_abs_error = max_abs_error(&device_logits, &sequential_logits);
    let sequential_argmax_stability_certified =
        sequential_top1_margin > 2.0 * device_end_to_end_max_abs_error;
    let semantic_pass = all_argmax_equal
        && sequential_argmax_stability_certified
        && prefill_trajectory_pass(&normalized_hidden_operator, 0.02, 0.999)
        && prefill_trajectory_pass(&logits_operator, 0.02, 0.999)
        && prefill_trajectory_pass(&logits_backbone, 0.05, COSINE_SIMILARITY_MIN)
        && prefill_trajectory_pass(&logits_end_to_end, 0.05, COSINE_SIMILARITY_MIN);
    let numerical_pass = numerical_proof
        && sequential.owner_epoch_proof
        && prefill_trajectory_pass(
            &final_hidden,
            FINAL_HIDDEN_RELATIVE_L2_MAX,
            COSINE_SIMILARITY_MIN,
        )
        && states.iter().all(|sample| {
            prefill_trajectory_pass(sample, STATE_RELATIVE_L2_MAX, COSINE_SIMILARITY_MIN)
        });

    let mut completion_poll_iterations = sequential
        .completion_poll_iterations
        .saturating_add(numerical_polls);
    let mut owner_epoch_proof = sequential.owner_epoch_proof && numerical_proof;
    for _ in 0..config.warmup {
        let _ = reset_plans(&mut plans, &stream)?;
        workspace
            .hidden_a
            .copy_from_host_sync(&stream, &f32_bytes(&input_hidden))?;
        let (_, polls, proof, _) = submit_prefill_backbone_with_proof(
            &stream,
            &operations,
            &workspace.cublas,
            terminal_source,
            &terminal_hidden,
            &output_head,
            &started_event,
            &stopped_event,
            &terminal_event,
        )?;
        completion_poll_iterations = completion_poll_iterations.saturating_add(polls);
        owner_epoch_proof &= proof;
    }

    let timed_boundary_start = cuda_host_boundary_snapshot();
    let timed_wall_start = Instant::now();
    let mut cuda_event_total_ms = 0.0_f64;
    for _ in 0..config.iterations {
        let _ = reset_plans(&mut plans, &stream)?;
        workspace
            .hidden_a
            .copy_from_host_sync(&stream, &f32_bytes(&input_hidden))?;
        let (_, polls, proof, cuda_event_ms) = submit_prefill_backbone_with_proof(
            &stream,
            &operations,
            &workspace.cublas,
            terminal_source,
            &terminal_hidden,
            &output_head,
            &started_event,
            &stopped_event,
            &terminal_event,
        )?;
        completion_poll_iterations = completion_poll_iterations.saturating_add(polls);
        owner_epoch_proof &= proof;
        cuda_event_total_ms += cuda_event_ms;
    }
    let host_wall_total_ms = timed_wall_start.elapsed().as_secs_f64() * 1_000.0;
    let timed_boundaries = boundary_delta(timed_boundary_start, cuda_host_boundary_snapshot())?;
    let cuda_event_mean_ms = cuda_event_total_ms / config.iterations as f64;
    let host_wall_mean_ms = host_wall_total_ms / config.iterations as f64;
    if !cuda_event_mean_ms.is_finite()
        || cuda_event_mean_ms <= 0.0
        || !host_wall_mean_ms.is_finite()
        || host_wall_mean_ms <= 0.0
    {
        return Err(invalid("prefill backbone timing"));
    }

    let status = if numerical_pass && semantic_pass && owner_epoch_proof {
        "PASS_COMPONENT_QWEN35_PREFILL_64_LAYER_BACKBONE_CUBLAS"
    } else {
        "FAIL_COMPONENT_QWEN35_PREFILL_64_LAYER_BACKBONE_CUBLAS"
    };
    let terminal_input_token_id = token_ids[config.tokens - 1];
    Ok(Qwen35PrefillBackboneCudaReport {
        schema: QWEN35_PREFILL_BACKBONE_CUDA_SCHEMA.to_owned(),
        status: status.to_owned(),
        component_only: true,
        source_commit: config.source_commit.clone(),
        binary_path: binary_path.display().to_string(),
        binary_sha256: sha256_hex(&binary_bytes),
        device,
        config_sha256: sha256_hex(&config_bytes),
        gguf_path: config.gguf_path.display().to_string(),
        gguf_file_sha256: manifest.file_sha256.clone(),
        token_ids,
        input_hidden_sha256: sha256_f32_bits(&input_hidden),
        numerical: PrefillBackboneNumericalReport {
            final_hidden,
            state_aggregate,
            states,
            final_hidden_relative_l2_max: FINAL_HIDDEN_RELATIVE_L2_MAX,
            state_relative_l2_max: STATE_RELATIVE_L2_MAX,
            cosine_similarity_min: COSINE_SIMILARITY_MIN,
            pass: numerical_pass,
        },
        semantic: PrefillBackboneSemanticReport {
            terminal_position: config.tokens - 1,
            terminal_input_token_id,
            vocabulary_size: model_config.vocab_size,
            output_head_numerics: "ggml_q8_0_activation".to_owned(),
            normalized_hidden_operator,
            logits_operator,
            logits_backbone,
            logits_end_to_end,
            batched_argmax,
            sequential_argmax,
            device_argmax,
            all_argmax_equal,
            sequential_top1_margin,
            device_end_to_end_max_abs_error,
            sequential_argmax_stability_certified,
            pass: semantic_pass,
        },
        timing: PrefillBackboneTimingReport {
            warmup: config.warmup,
            iterations: config.iterations,
            tokens_per_chunk: config.tokens,
            cuda_event_total_ms,
            cuda_event_mean_ms,
            host_wall_total_ms,
            host_wall_mean_ms,
            diagnostic_backbone_tokens_per_second: config.tokens as f64 * 1_000.0
                / cuda_event_mean_ms,
            host_wall_backbone_tokens_per_second: config.tokens as f64 * 1_000.0
                / host_wall_mean_ms,
            reset_excluded_from_cuda_event: true,
            reset_included_in_host_wall: true,
            input_upload_excluded_from_cuda_event: true,
            terminal_output_head_included_in_cuda_event: true,
        },
        resources: PrefillLayerResourceReport {
            setup_boundaries,
            sequential_oracle_boundaries,
            batched_numerical_boundaries,
            timed_boundaries,
            allocation_delta: allocation_delta(
                allocation_start,
                cuda_device_allocation_snapshot(),
            )?,
            completion_poll_iterations,
            owner_epoch_proof,
        },
        claims: PrefillBackboneClaimsReport {
            real_model_weights_executed: true,
            complete_64_layer_backbone_executed: true,
            linear_layer_count,
            full_attention_layer_count,
            sequential_gpu_oracle_executed: true,
            output_head_executed: true,
            complete_transformer_executed: true,
            model_prefill_tps_permitted: false,
            semantic_parity_established: false,
            product_tps_permitted: false,
            default_cli_enabled: false,
        },
    })
}

#[cfg(feature = "qwen35-prefill-cublas")]
fn build_prefill_chunk_schedule(
    workspace: &PrefillWorkspace,
    plans: &[DevicePlan],
    stream: &OwnedCudaStream,
    total_tokens: usize,
    chunk_tokens: usize,
    epsilon: f32,
) -> Result<PrefillChunkSchedule, Qwen35MixedPrefixCudaError> {
    if plans.len() != FULL_TARGET_LAYER_COUNT
        || total_tokens == 0
        || chunk_tokens == 0
        || chunk_tokens.min(total_tokens) > workspace.max_tokens
    {
        return Err(invalid("prefill chunk schedule geometry"));
    }
    let mut chunks = Vec::with_capacity(total_tokens.div_ceil(chunk_tokens));
    for position_base in (0..total_tokens).step_by(chunk_tokens) {
        let tokens = chunk_tokens.min(total_tokens - position_base);
        let operations = plans
            .iter()
            .enumerate()
            .map(|(layer, plan)| match plan {
                DevicePlan::Linear(plan) => workspace
                    .build_linear_ops(stream, plan, layer, tokens, epsilon)
                    .map(PrefillLayerOps::Linear),
                DevicePlan::Full(plan) => workspace
                    .build_full_ops(stream, plan, layer, tokens, position_base, epsilon)
                    .map(PrefillLayerOps::Full),
            })
            .collect::<Result<Vec<_>, _>>()?;
        chunks.push(PrefillChunk { tokens, operations });
    }
    Ok(PrefillChunkSchedule {
        chunk_tokens,
        chunks,
        #[cfg(feature = "qwen35-kv8-chunk-prefill")]
        sparse_profile: None,
    })
}





// CPU-only admission; no CUDA query, allocation or file load is permitted here.
#[cfg(feature = "qwen35-prefill-cublas")]
fn prefill_attention_path_label(tc: bool, kv8: bool, shared: bool) -> &'static str {
    if tc {
        "tensor_core_prefix_current_stable_lse_merge"
    } else if kv8 && shared {
        "simt_kv8_grouped_gqa_shared_softmax_ordered"
    } else if kv8 {
        "simt_kv8_grouped_gqa_online_softmax"
    } else {
        "simt_f32_cache_materialized_softmax"
    }
}

#[cfg(feature = "qwen35-prefill-cublas")]
fn validate_prefill_chunked_config(
    config: &Qwen35PrefillChunkedCudaConfig,
) -> Result<usize, Qwen35MixedPrefixCudaError> {
    if config.wide_prefill_microbatch && !false {
        return Err(invalid("wide prefill microbatch feature"));
    }
    if !matches!(config.wide_prefill_microbatch_limit, 2048 | 4096)
        || (!config.wide_prefill_microbatch && config.wide_prefill_microbatch_limit != 2048)
    {
        return Err(invalid(
            "wide microbatch limit requires explicit admission and2048/4096",
        ));
    }
    if config.paired_weight_decode && config.quad_weight_decode {
        return Err(invalid(
            "paired and quad weight decode are mutually exclusive",
        ));
    }
    if config.profile_ffn_projections && !config.profile_simt_stages {
        return Err(invalid(
            "FFN projection profile requires independent SIMT stage profile",
        ));
    }
    if false && config.profile_ffn_projections {
        return Err(invalid(
            "fused SiLU-Q8 excludes materialized FFN projection profile",
        ));
    }
    if config.simt_softmax_tile16 && !config.simt_shared_softmax {
        return Err(invalid("tile16 requires shared SIMT softmax"));
    }
    if config.simt_token_pair
        && (!false
            || !config.simt_shared_softmax
            || !config.simt_unit_previous_scale
            || config.simt_softmax_tile16)
    {
        return Err(invalid(
            "token pair requires its feature and shared SIMT tile8/unit scale",
        ));
    }
    if config.simt_unit_previous_scale
        && (!config.simt_shared_softmax || config.simt_softmax_tile16)
    {
        return Err(invalid("unit previous scale requires shared SIMT tile8"));
    }
    if (config.simt_shared_softmax
        || config.simt_shared_gdn_scalars
        || config.profile_simt_stages
        || config.paired_weight_decode
        || config.quad_weight_decode
        || config.wide_prefill_microbatch)
        && (!cfg!(feature = "qwen35-kv8-chunk-prefill")
            || config.tensor_core_prefix_attention
            || config.reference_tensor_core_prefix_attention
            || config.direct_post_attention_activation
            || config.reference_direct_post_attention_activation
            || config.layer_major_window)
    {
        return Err(invalid(
            "shared SIMT/stage profile requires plain KV8 and retained activation",
        ));
    }
    
    if config.tensor_core_prefix_attention || config.reference_tensor_core_prefix_attention {
        return Err(invalid("Tensor Core prefix attention feature"));
    }
    
    if config.layer_major_window {
        return Err(invalid("layer-major window feature"));
    }
    if config.reference_tensor_core_prefix_attention && !config.tensor_core_prefix_attention {
        return Err(invalid("TC reference requires primary TC path"));
    }
    if config.layer_major_window
        && (!config.tensor_core_prefix_attention
            || config.chunk_tokens != 1024
            || config.reference_chunk_tokens > 512)
    {
        return Err(invalid(
            "layer-major requires TC window1024 and reference <=512",
        ));
    }
    let max_chunk_tokens = config.chunk_tokens.max(config.reference_chunk_tokens);
    // The compiled research capability never silently widens ordinary or TC
    // admission. Wide SIMT requires BOTH the feature and an explicit request.
    let max_chunk_limit = if config.wide_prefill_microbatch {
        config.wide_prefill_microbatch_limit
    } else {
        1024
    };
    let total_token_limit = if config.tensor_core_prefix_attention {
        131_072
    } else {
        4_096
    };
    if config.total_tokens == 0
        || config.total_tokens > total_token_limit
        || config.chunk_tokens == 0
        || config.reference_chunk_tokens == 0
        || max_chunk_tokens > max_chunk_limit
        || (!config.tensor_core_prefix_attention
            && !config.simt_shared_softmax
            && !config.simt_shared_gdn_scalars
            && !config.paired_weight_decode
            && !config.quad_weight_decode
            && config.chunk_tokens == config.reference_chunk_tokens)
        || config.device_ordinal < 0
        || config.warmup > 10
        || config.iterations == 0
        || config.iterations > 20
        || (config.direct_post_attention_activation && !config.tensor_core_prefix_attention)
        || (config.reference_direct_post_attention_activation
            && !config.tensor_core_prefix_attention)
    {
        return Err(invalid("chunked prefill scalar bounds"));
    }
    // A short final/only window must not bind more query rows than the cache
    // contains. Validate requested limits above before clamping actual storage.
    Ok(max_chunk_tokens.min(config.total_tokens))
}

#[cfg(feature = "qwen35-prefill-cublas")]
#[allow(clippy::too_many_arguments)]
fn run_prefill_chunk_schedule(
    plans: &mut [DevicePlan],
    workspace: &PrefillWorkspace,
    schedule: &PrefillChunkSchedule,
    tensor_core_prefix_attention: bool,
    direct_post_attention_activation: bool,
    stream: &OwnedCudaStream,
    terminal: Option<(&OwnedCudaBuffer, &DeviceOutputHeadPlan)>,
    started_event: &OwnedCudaEvent,
    stopped_event: &OwnedCudaEvent,
    terminal_event: &OwnedCudaEvent,
    input_hidden: &[f32],
    collect_evidence: bool,
    reset_before_run: bool,
) -> Result<PrefillChunkRun, Qwen35MixedPrefixCudaError> {
    #[cfg(feature = "qwen35-kv8-chunk-prefill")]
    if schedule.sparse_profile.is_some()
        && (tensor_core_prefix_attention || direct_post_attention_activation)
    {
        return Err(invalid(
            "sparse profile requires unchanged retained SIMT operations",
        ));
    }
    let total_tokens = schedule
        .chunks
        .iter()
        .try_fold(0_usize, |total, chunk| total.checked_add(chunk.tokens))
        .ok_or_else(|| invalid("prefill chunk schedule extent overflow"))?;
    if schedule.chunks.is_empty()
        || schedule.chunks.iter().any(|chunk| {
            chunk.tokens == 0
                || chunk.tokens > schedule.chunk_tokens
                || chunk.operations.len() != FULL_TARGET_LAYER_COUNT
        })
        || input_hidden.len()
            != total_tokens
                .checked_mul(linear::HIDDEN)
                .ok_or_else(|| invalid("prefill chunk input extent overflow"))?
        || (collect_evidence && terminal.is_none())
    {
        return Err(invalid("prefill chunk execution geometry"));
    }

    let host_started = Instant::now();
    if reset_before_run {
        let _ = reset_plans(plans, stream)?;
    }
    let row_bytes = linear::HIDDEN
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| invalid("prefill terminal row extent overflow"))?;
    let mut final_hidden = collect_evidence.then(|| Vec::with_capacity(input_hidden.len()));
    let mut last_receipt = None;
    let mut cuda_event_ms = 0.0_f64;
    let mut completion_poll_iterations = 0_u64;
    let mut owner_epoch_proof = true;
    let mut input_start = 0_usize;
    let mut submission_boundaries = BoundaryReport::default();
    let mut progress_receipts = Vec::with_capacity(schedule.chunks.len());
    for (chunk_index, chunk) in schedule.chunks.iter().enumerate() {
        let chunk_elements = chunk
            .tokens
            .checked_mul(linear::HIDDEN)
            .ok_or_else(|| invalid("prefill chunk hidden extent overflow"))?;
        let chunk_bytes = chunk_elements
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or_else(|| invalid("prefill chunk hidden byte extent overflow"))?;
        let input_end = input_start
            .checked_add(chunk_elements)
            .ok_or_else(|| invalid("prefill chunk input range overflow"))?;
        let input_chunk = input_hidden
            .get(input_start..input_end)
            .ok_or_else(|| invalid("prefill chunk input range"))?;
        let input_hidden_sha256 = sha256_f32_bits(input_chunk);
        workspace
            .hidden_a
            .copy_from_host_sync(stream, &f32_bytes(input_chunk))?;

        let submission_boundary_start = cuda_host_boundary_snapshot();
        started_event.record(stream)?;
        for (_layer, operation) in chunk.operations.iter().enumerate() {
            #[cfg(feature = "qwen35-kv8-chunk-prefill")]
            if let Some(profile) = &schedule.sparse_profile {
                if profile.selected(chunk_index) {
                    profile.submit(chunk_index, _layer, operation, stream, &workspace.cublas)?;
                    continue;
                }
            }
            operation.submit_with_attention_path(
                stream,
                &workspace.cublas,
                tensor_core_prefix_attention,
                direct_post_attention_activation,
            )?;
        }
        if chunk_index + 1 == schedule.chunks.len() {
            if let Some((terminal_hidden, output_head)) = terminal {
                let terminal_source_offset = chunk
                    .tokens
                    .checked_sub(1)
                    .and_then(|row| row.checked_mul(row_bytes))
                    .ok_or_else(|| invalid("prefill terminal row offset overflow"))?;
                let source = workspace.hidden_a.view(terminal_source_offset, row_bytes)?;
                // SAFETY: source and destination are disjoint live allocations and
                // the terminal event below covers this same-stream copy and head.
                unsafe {
                    terminal_hidden.copy_from_device_async_unchecked(stream, 0, source)?;
                }
                output_head.submit(stream)?;
            }
        }
        submission_boundaries = add_boundaries(
            submission_boundaries,
            boundary_delta(submission_boundary_start, cuda_host_boundary_snapshot())?,
        );
        stopped_event.record(stream)?;
        let receipt = stream.submission_receipt()?;
        terminal_event.record_for_submission(stream, receipt)?;
        let mut polls = 0_u64;
        let completion = loop {
            if let Some(completion) = stream.query_event_completion(terminal_event, receipt)? {
                break completion;
            }
            polls = polls.saturating_add(1);
            if polls >= MAX_CHUNKED_PREFILL_COMPLETION_POLLS {
                // Very large SM70 chunks can outlive a pure spin-poll budget.
                // Quiesce this stream explicitly, then re-query the exact
                // submission/event pair.  The synchronization is observable
                // in the boundary ledger; it is a diagnostic fallback, not a
                // fabricated completion or a product async claim.
                stream.synchronize()?;
                if let Some(completion) = stream.query_event_completion(terminal_event, receipt)? {
                    break completion;
                }
                return Err(Qwen35MixedPrefixCudaError::CompletionPollLimit);
            }
            bounded_completion_poll_backoff(polls);
        };
        let proof =
            completion.covers_submission(receipt) && completion.belongs_to_stream(stream)?;
        completion_poll_iterations = completion_poll_iterations.saturating_add(polls);
        owner_epoch_proof &= proof;
        let chunk_cuda_ms = f64::from(started_event.elapsed_ms(stopped_event)?);
        cuda_event_ms += chunk_cuda_ms;
        #[cfg(feature = "qwen35-kv8-chunk-prefill")]
        if let Some(profile) = &schedule.sparse_profile {
            if profile.selected(chunk_index) {
                profile.finish_chunk(chunk_index, stream, chunk_cuda_ms, proof)?;
            }
        }
        let output_hidden = if collect_evidence {
            Some(read_f32_view(
                stream,
                workspace.hidden_a.view(0, chunk_bytes)?,
                receipt,
            )?)
        } else {
            None
        };
        if let Some(output) = &mut final_hidden {
            let output_hidden = output_hidden
                .as_ref()
                .ok_or_else(|| invalid("prefill chunk output evidence"))?;
            output.extend(output_hidden.iter().copied());
        }
        let state_cache_sha256 = if collect_evidence {
            let layer_digests = plans
                .iter()
                .map(|plan| plan.state_sha256(stream, receipt))
                .collect::<Result<Vec<_>, _>>()?;
            let mut digest = Sha256::new();
            for layer_digest in &layer_digests {
                digest.update(layer_digest.as_bytes());
            }
            Some((format!("{:x}", digest.finalize()), layer_digests))
        } else {
            None
        };
        let (state_cache_sha256, state_cache_layer_sha256) = state_cache_sha256
            .map_or((None, None), |(aggregate, layers)| {
                (Some(aggregate), Some(layers))
            });
        let start_position = input_start / linear::HIDDEN;
        let end_position_exclusive = start_position
            .checked_add(chunk.tokens)
            .ok_or_else(|| invalid("prefill chunk position extent overflow"))?;
        progress_receipts.push(PrefillChunkProgressReceipt {
            chunk_index,
            start_position,
            end_position_exclusive,
            token_count: chunk.tokens,
            submission_epoch: receipt.submission_epoch(),
            owner_epoch_proof: proof,
            terminal: chunk_index + 1 == schedule.chunks.len(),
            input_hidden_sha256,
            output_hidden_sha256: output_hidden.as_deref().map(sha256_f32_bits),
            state_cache_sha256,
            state_cache_layer_sha256,
        });
        last_receipt = Some(receipt);
        input_start = input_end;
    }

    let (states, normalized_hidden, logits) = if collect_evidence {
        let receipt = last_receipt.ok_or_else(|| invalid("prefill chunk terminal receipt"))?;
        let (_, output_head) = terminal.ok_or_else(|| invalid("prefill terminal output head"))?;
        (
            Some(read_prefill_backbone_states(plans, stream, receipt)?),
            Some(read_f32_buffer(
                stream,
                &output_head.normalized_hidden,
                receipt,
            )?),
            Some(read_f32_buffer(stream, &output_head.logits, receipt)?),
        )
    } else {
        (None, None, None)
    };
    Ok(PrefillChunkRun {
        final_hidden,
        states,
        normalized_hidden,
        logits,
        cuda_event_ms,
        host_wall_ms: host_started.elapsed().as_secs_f64() * 1_000.0,
        submission_boundaries,
        completion_poll_iterations,
        owner_epoch_proof,
        progress_receipts,
    })
}

#[cfg(feature = "qwen35-prefill-cublas")]
fn prefill_chunk_timing_report(
    total_tokens: usize,
    schedule: &PrefillChunkSchedule,
    iterations: usize,
    cuda_event_total_ms: f64,
    host_wall_total_ms: f64,
) -> Result<PrefillChunkScheduleTimingReport, Qwen35MixedPrefixCudaError> {
    if iterations == 0
        || !cuda_event_total_ms.is_finite()
        || cuda_event_total_ms <= 0.0
        || !host_wall_total_ms.is_finite()
        || host_wall_total_ms <= 0.0
    {
        return Err(invalid("prefill chunk timing"));
    }
    let cuda_event_mean_sequence_ms = cuda_event_total_ms / iterations as f64;
    let host_wall_mean_sequence_ms = host_wall_total_ms / iterations as f64;
    Ok(PrefillChunkScheduleTimingReport {
        chunk_tokens: schedule.chunk_tokens,
        chunk_count: schedule.chunks.len(),
        iterations,
        cuda_event_total_ms,
        cuda_event_mean_sequence_ms,
        host_wall_total_ms,
        host_wall_mean_sequence_ms,
        cuda_event_tokens_per_second: total_tokens as f64 * 1_000.0 / cuda_event_mean_sequence_ms,
        host_wall_tokens_per_second: total_tokens as f64 * 1_000.0 / host_wall_mean_sequence_ms,
    })
}

/// Execute a complete 64-layer prefill as a bounded sequence of causal chunks.
/// The primary chunk width is checked against a second width over every hidden
/// row and every persistent state/cache, then timed with only terminal logits
/// retained on the device.
#[cfg(feature = "qwen35-prefill-cublas")]
pub fn run_qwen35_prefill_chunked_cuda(
    config: &Qwen35PrefillChunkedCudaConfig,
) -> Result<Qwen35PrefillChunkedCudaReport, Qwen35MixedPrefixCudaError> {
    let max_chunk_tokens = validate_prefill_chunked_config(config)?;
    let numerical_contract =
        PrefillChunkedNumericalContract::for_attention_path(config.tensor_core_prefix_attention);

    if config.gguf_path.as_os_str().is_empty() || !config.gguf_path.is_file() {
        return Err(invalid("chunked prefill GGUF path"));
    }
    if config.config_path.as_os_str().is_empty() || !config.config_path.is_file() {
        return Err(invalid("chunked prefill config path"));
    }
    let binary_path = std::env::current_exe().map_err(Qwen35Error::from)?;
    let binary_bytes = std::fs::read(&binary_path).map_err(Qwen35Error::from)?;
    let config_bytes = read_stable_regular_file(&config.config_path, "Qwen config")?;
    let model_config = Qwen35Config::parse(&config_bytes)?;
    model_config.validate_exact_profile()?;
    let manifest = GgufManifest::inspect(&config.gguf_path, GgufAuditOptions::default())?;
    let schema_summary =
        Qwen35TensorSchema::validate_manifest_allowing_mtp(&model_config, &manifest)?;
    let specs = (0..FULL_TARGET_LAYER_COUNT)
        .map(|layer| Qwen35LayerSpec::from_config(&model_config, layer))
        .collect::<Result<Vec<_>, _>>()?;
    let linear_layer_count = specs
        .iter()
        .filter(|spec| spec.kind == Qwen35LayerKind::LinearAttention)
        .count();
    let full_attention_layer_count = specs
        .iter()
        .filter(|spec| spec.kind == Qwen35LayerKind::FullAttention)
        .count();
    if schema_summary.actual_tensor_count < 1_187
        || model_config.num_hidden_layers != FULL_TARGET_LAYER_COUNT
        || config.total_tokens > model_config.max_position_embeddings
        || linear_layer_count != 48
        || full_attention_layer_count != 16
        || specs.iter().any(|spec| {
            spec.hidden_size != linear::HIDDEN || spec.intermediate_size != linear::INTERMEDIATE
        })
    {
        return Err(invalid("chunked prefill audited model profile"));
    }

    let device_properties = cuda_device_properties(config.device_ordinal)?;
    if device_properties.compute_capability.major != 7
        || device_properties.compute_capability.minor != 0
    {
        return Err(invalid("chunked prefill diagnostic requires SM70"));
    }
    let device = DeviceReport {
        ordinal: device_properties.ordinal,
        name: device_properties.name,
        compute_capability: [
            device_properties.compute_capability.major as u16,
            device_properties.compute_capability.minor as u16,
        ],
        total_global_memory_bytes: device_properties.total_global_memory_bytes,
    };

    let mut reader = manifest.open_tensor_reader(&config.gguf_path)?;
    let embedding = manifest
        .tensor(linear::EMBEDDING)
        .ok_or_else(|| invalid("chunked prefill embedding descriptor"))?
        .clone();
    let usable_vocabulary = model_config
        .vocab_size
        .checked_sub(1)
        .ok_or_else(|| invalid("chunked prefill vocabulary"))?;
    let token_ids = (0..config.total_tokens)
        .map(|index| 1 + (index * 7_919) % usable_vocabulary)
        .collect::<Vec<_>>();
    let mut input_hidden = Vec::with_capacity(
        config
            .total_tokens
            .checked_mul(linear::HIDDEN)
            .ok_or_else(|| invalid("chunked prefill input extent"))?,
    );
    for token_id in &token_ids {
        input_hidden.extend(linear::read_embedding_row(
            &mut reader,
            &embedding,
            *token_id,
        )?);
    }
    let first_hidden = input_hidden
        .get(..linear::HIDDEN)
        .ok_or_else(|| invalid("chunked prefill first hidden"))?;

    let allocation_start = cuda_device_allocation_snapshot();
    let setup_boundary_start = cuda_host_boundary_snapshot();
    let stream = OwnedCudaStream::create(config.device_ordinal)?;
    let materials = load_layer_materials(
        &mut reader,
        &manifest,
        &model_config,
        FULL_TARGET_LAYER_COUNT,
    )?;
    let output_head_material = load_output_head_material(&mut reader, &manifest, &model_config)?;
    let streamed_output_head = GgufStreamedOutputHead::for_qwen35_with_numerics(
        &mut reader,
        &model_config,
        SEMANTIC_OUTPUT_HEAD_CHUNK_ROWS,
        GgufProjectionNumerics::GgmlQ8_0Activation,
    )?;
    let zero_projection = linear::ProjectedToken {
        qkv: vec![0.0; linear::QKV_ELEMENTS],
        z: vec![0.0; linear::RECURRENT_ELEMENTS],
        decay_gate: vec![0.0; linear::VALUE_HEADS],
        beta: vec![0.0; linear::VALUE_HEADS],
    };
    let mut plans = Vec::with_capacity(FULL_TARGET_LAYER_COUNT);
    for material in &materials {
        match material {
            LayerMaterial::Linear(material) => plans.push(DevicePlan::Linear(build_linear_plan(
                &stream,
                material,
                first_hidden,
                &zero_projection,
                true,
                None,
            )?)),
            LayerMaterial::Full(material) => plans.push(DevicePlan::Full(build_full_plan(
                &stream,
                material,
                &model_config,
                config.total_tokens,
                max_chunk_tokens,
                None,
            )?)),
        }
    }
    let workspace =
        PrefillWorkspace::new_with_rope_capacity(&stream, max_chunk_tokens, config.total_tokens)?;
    workspace.upload_rope_tables(
        &stream,
        config.total_tokens,
        0,
        model_config.rope_theta as f32,
    )?;
    let chunk_major_schedule = || {
        build_prefill_chunk_schedule(
            &workspace,
            &plans,
            &stream,
            config.total_tokens,
            config.chunk_tokens,
            model_config.rms_norm_eps as f32,
        )
    };
    
    
    let primary_schedule = chunk_major_schedule()?;
    #[cfg(feature = "qwen35-kv8-chunk-prefill")]
    let mut primary_schedule = primary_schedule;
    #[cfg(feature = "qwen35-kv8-chunk-prefill")]
    if config.simt_shared_softmax {
        for chunk in &mut primary_schedule.chunks {
            for operation in &mut chunk.operations {
                if let PrefillLayerOps::Full(full) = operation {
                    full.simt_shared_softmax = true;
                    full.simt_softmax_tile16 = config.simt_softmax_tile16;
                    full.simt_unit_previous_scale = config.simt_unit_previous_scale;
                    
                }
            }
        }
    }
    #[cfg(feature = "qwen35-kv8-chunk-prefill")]
    if config.simt_shared_gdn_scalars {
        for chunk in &mut primary_schedule.chunks {
            for operation in &mut chunk.operations {
                if let PrefillLayerOps::Linear(linear) = operation {
                    linear.simt_shared_gdn_scalars = true;
                }
            }
        }
    }
    #[cfg(feature = "qwen35-kv8-chunk-prefill")]
    if config.paired_weight_decode || config.quad_weight_decode {
        for chunk in &mut primary_schedule.chunks {
            for op in &mut chunk.operations {
                match op {
                    PrefillLayerOps::Linear(p) => {
                        let ffn = p.ffn.legacy_mut()?;
for projection in [
                            &mut p.qkv,
                            &mut p.z,
                            &mut p.alpha,
                            &mut p.beta,
                            &mut p.output,
                            &mut ffn.ffn_gate,
                            &mut ffn.ffn_up,
                            &mut ffn.ffn_down,
                        ] {
                            if config.quad_weight_decode {
                                projection.enable_quad_weight_decode();
                            } else {
                                projection.enable_paired_weight_decode();
                            }
                        }
                    }
                    PrefillLayerOps::Full(p) => {
                        let ffn = p.ffn.legacy_mut()?;
for projection in [
                            &mut p.q,
                            &mut p.k,
                            &mut p.v,
                            &mut p.output,
                            &mut ffn.ffn_gate,
                            &mut ffn.ffn_up,
                            &mut ffn.ffn_down,
                        ] {
                            if config.quad_weight_decode {
                                projection.enable_quad_weight_decode();
                            } else {
                                projection.enable_paired_weight_decode();
                            }
                        }
                    }
                    
                }
            }
        }
    }
    let reference_schedule = build_prefill_chunk_schedule(
        &workspace,
        &plans,
        &stream,
        config.total_tokens,
        config.reference_chunk_tokens,
        model_config.rms_norm_eps as f32,
    )?;
    let terminal_row_bytes = linear::HIDDEN
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| invalid("chunked prefill terminal hidden extent overflow"))?;
    let terminal_hidden = alloc_f32(config.device_ordinal, linear::HIDDEN)?;
    let output_head = build_device_output_head_plan(
        &stream,
        &output_head_material,
        terminal_hidden.view(0, terminal_row_bytes)?,
        model_config.hidden_size,
        model_config.vocab_size,
        model_config.rms_norm_eps as f32,
    )?;
    drop(output_head_material);
    drop(materials);
    let started_event = OwnedCudaEvent::create(config.device_ordinal)?;
    let stopped_event = OwnedCudaEvent::create(config.device_ordinal)?;
    let terminal_event = OwnedCudaEvent::create(config.device_ordinal)?;
    let setup_boundaries = boundary_delta(setup_boundary_start, cuda_host_boundary_snapshot())?;

    let primary_boundary_start = cuda_host_boundary_snapshot();
    let primary = run_prefill_chunk_schedule(
        &mut plans,
        &workspace,
        &primary_schedule,
        config.tensor_core_prefix_attention,
        config.direct_post_attention_activation,
        &stream,
        Some((&terminal_hidden, &output_head)),
        &started_event,
        &stopped_event,
        &terminal_event,
        &input_hidden,
        true,
        true,
    )?;
    let primary_numerical_boundaries =
        boundary_delta(primary_boundary_start, cuda_host_boundary_snapshot())?;
    let reference_boundary_start = cuda_host_boundary_snapshot();
    let reference = run_prefill_chunk_schedule(
        &mut plans,
        &workspace,
        &reference_schedule,
        config.reference_tensor_core_prefix_attention,
        config.reference_direct_post_attention_activation,
        &stream,
        Some((&terminal_hidden, &output_head)),
        &started_event,
        &stopped_event,
        &terminal_event,
        &input_hidden,
        true,
        true,
    )?;
    let reference_numerical_boundaries =
        boundary_delta(reference_boundary_start, cuda_host_boundary_snapshot())?;
    let primary_progress_receipt_sequence_valid =
        progress_receipt_sequence_valid(&primary.progress_receipts, config.total_tokens);
    let reference_progress_receipt_sequence_valid =
        progress_receipt_sequence_valid(&reference.progress_receipts, config.total_tokens);
    let common_boundary_comparisons =
        compare_progress_receipts(&primary.progress_receipts, &reference.progress_receipts);

    let primary_hidden = primary
        .final_hidden
        .as_ref()
        .ok_or_else(|| invalid("chunked prefill primary hidden evidence"))?;
    let reference_hidden = reference
        .final_hidden
        .as_ref()
        .ok_or_else(|| invalid("chunked prefill reference hidden evidence"))?;
    let primary_states = primary
        .states
        .as_ref()
        .ok_or_else(|| invalid("chunked prefill primary state evidence"))?;
    let reference_states = reference
        .states
        .as_ref()
        .ok_or_else(|| invalid("chunked prefill reference state evidence"))?;
    if primary_states.len() != reference_states.len()
        || primary_states
            .iter()
            .zip(reference_states)
            .any(|(actual, expected)| actual.0 != expected.0)
    {
        return Err(invalid("chunked prefill state inventory"));
    }
    let final_hidden = trajectory_sample("final_hidden", primary_hidden, reference_hidden);
    let states = primary_states
        .iter()
        .zip(reference_states)
        .map(|(actual, expected)| trajectory_sample(&actual.0, &actual.1, &expected.1))
        .collect::<Vec<_>>();
    let state_aggregate = aggregate_trajectory(states.iter());
    let same_width_candidate_replay_bit_exact = ((config.simt_shared_softmax
        || config.simt_shared_gdn_scalars
        || config.paired_weight_decode
        || config.quad_weight_decode)
        && config.chunk_tokens.min(config.total_tokens)
            == config.reference_chunk_tokens.min(config.total_tokens))
    .then(|| {
        equal_f32_bits(primary_hidden, reference_hidden)
            && primary_states
                .iter()
                .zip(reference_states)
                .all(|(a, b)| equal_f32_bits(&a.1, &b.1))
            && primary
                .logits
                .as_ref()
                .zip(reference.logits.as_ref())
                .is_some_and(|(a, b)| equal_f32_bits(a, b))
    });
    let same_width_attention_replay_bit_exact =
        same_width_candidate_replay_bit_exact.filter(|_| config.simt_shared_softmax);
    let same_width_linear_replay_bit_exact =
        same_width_candidate_replay_bit_exact.filter(|_| config.simt_shared_gdn_scalars);
    let same_width_weight_decode_replay_bit_exact = same_width_candidate_replay_bit_exact
        .filter(|_| config.paired_weight_decode || config.quad_weight_decode);
    let numerical_pass = primary.owner_epoch_proof
        && reference.owner_epoch_proof
        && primary_progress_receipt_sequence_valid
        && reference_progress_receipt_sequence_valid
        && same_width_candidate_replay_bit_exact != Some(false)
        && prefill_trajectory_pass(
            &final_hidden,
            numerical_contract.final_hidden_relative_l2_max,
            numerical_contract.final_hidden_cosine_min,
        )
        && states.iter().all(|sample| {
            prefill_trajectory_pass(
                sample,
                numerical_contract.state_relative_l2_max,
                numerical_contract.state_cosine_min,
            )
        });

    let terminal_element_offset = config
        .total_tokens
        .checked_sub(1)
        .and_then(|row| row.checked_mul(linear::HIDDEN))
        .ok_or_else(|| invalid("chunked prefill semantic offset overflow"))?;
    let primary_terminal_hidden = primary_hidden
        .get(terminal_element_offset..terminal_element_offset + linear::HIDDEN)
        .ok_or_else(|| invalid("chunked prefill primary terminal hidden"))?;
    let reference_terminal_hidden = reference_hidden
        .get(terminal_element_offset..terminal_element_offset + linear::HIDDEN)
        .ok_or_else(|| invalid("chunked prefill reference terminal hidden"))?;
    let primary_normalized = streamed_output_head.normalized_hidden(primary_terminal_hidden)?;
    let primary_logits = streamed_output_head.logits(&mut reader, primary_terminal_hidden)?;
    let reference_logits = streamed_output_head.logits(&mut reader, reference_terminal_hidden)?;
    let device_normalized = primary
        .normalized_hidden
        .as_ref()
        .ok_or_else(|| invalid("chunked prefill device normalized evidence"))?;
    let device_logits = primary
        .logits
        .as_ref()
        .ok_or_else(|| invalid("chunked prefill device logits evidence"))?;
    let normalized_hidden_operator = trajectory_sample(
        "terminal.normalized_hidden_operator",
        device_normalized,
        &primary_normalized,
    );
    let logits_operator =
        trajectory_sample("terminal.logits_operator", device_logits, &primary_logits);
    let logits_backbone = trajectory_sample(
        "terminal.logits_chunk_schedule",
        &primary_logits,
        &reference_logits,
    );
    let logits_end_to_end = trajectory_sample(
        "terminal.logits_end_to_end",
        device_logits,
        &reference_logits,
    );
    let batched_argmax = usize::try_from(greedy_argmax(&primary_logits, model_config.vocab_size)?)
        .map_err(|_| invalid("chunked prefill primary argmax overflow"))?;
    let sequential_argmax =
        usize::try_from(greedy_argmax(&reference_logits, model_config.vocab_size)?)
            .map_err(|_| invalid("chunked prefill reference argmax overflow"))?;
    let device_argmax = usize::try_from(greedy_argmax(device_logits, model_config.vocab_size)?)
        .map_err(|_| invalid("chunked prefill device argmax overflow"))?;
    let all_argmax_equal =
        batched_argmax == sequential_argmax && device_argmax == sequential_argmax;
    let sequential_top1_margin = top1_margin_for_argmax(&reference_logits, sequential_argmax);
    let device_end_to_end_max_abs_error = max_abs_error(device_logits, &reference_logits);
    let sequential_argmax_stability_certified =
        sequential_top1_margin > 2.0 * device_end_to_end_max_abs_error;
    let semantic_pass = all_argmax_equal
        && sequential_argmax_stability_certified
        && prefill_trajectory_pass(&normalized_hidden_operator, 0.02, 0.999)
        && prefill_trajectory_pass(&logits_operator, 0.02, 0.999)
        && numerical_contract.logits_pass(&logits_backbone)
        && numerical_contract.logits_pass(&logits_end_to_end);

    let mut completion_poll_iterations = primary
        .completion_poll_iterations
        .saturating_add(reference.completion_poll_iterations);
    let mut owner_epoch_proof = primary.owner_epoch_proof && reference.owner_epoch_proof;
    owner_epoch_proof &=
        primary_progress_receipt_sequence_valid && reference_progress_receipt_sequence_valid;
    for _ in 0..config.warmup {
        let run = run_prefill_chunk_schedule(
            &mut plans,
            &workspace,
            &primary_schedule,
            config.tensor_core_prefix_attention,
            config.direct_post_attention_activation,
            &stream,
            Some((&terminal_hidden, &output_head)),
            &started_event,
            &stopped_event,
            &terminal_event,
            &input_hidden,
            false,
            true,
        )?;
        completion_poll_iterations =
            completion_poll_iterations.saturating_add(run.completion_poll_iterations);
        owner_epoch_proof &= run.owner_epoch_proof;
    }
    let timed_allocation_start = cuda_device_allocation_snapshot();
    let timed_boundary_start = cuda_host_boundary_snapshot();
    let mut timed_cuda_event_total_ms = 0.0_f64;
    let mut timed_host_wall_total_ms = 0.0_f64;
    let mut timed_submission_boundaries = BoundaryReport::default();
    let mut primary_timed_host_samples_ms = Vec::with_capacity(config.iterations);
    let mut primary_timed_cuda_samples_ms = Vec::with_capacity(config.iterations);
    for _ in 0..config.iterations {
        let run = run_prefill_chunk_schedule(
            &mut plans,
            &workspace,
            &primary_schedule,
            config.tensor_core_prefix_attention,
            config.direct_post_attention_activation,
            &stream,
            Some((&terminal_hidden, &output_head)),
            &started_event,
            &stopped_event,
            &terminal_event,
            &input_hidden,
            false,
            true,
        )?;
        timed_cuda_event_total_ms += run.cuda_event_ms;
        timed_host_wall_total_ms += run.host_wall_ms;
        primary_timed_host_samples_ms.push(run.host_wall_ms);
        primary_timed_cuda_samples_ms.push(run.cuda_event_ms);
        timed_submission_boundaries =
            add_boundaries(timed_submission_boundaries, run.submission_boundaries);
        completion_poll_iterations =
            completion_poll_iterations.saturating_add(run.completion_poll_iterations);
        owner_epoch_proof &= run.owner_epoch_proof;
    }
    let timed_boundaries = boundary_delta(timed_boundary_start, cuda_host_boundary_snapshot())?;
    let timed_allocation_interval =
        allocation_interval_report(timed_allocation_start, cuda_device_allocation_snapshot())?;
    let primary_numerical_timing = prefill_chunk_timing_report(
        config.total_tokens,
        &primary_schedule,
        1,
        primary.cuda_event_ms,
        primary.host_wall_ms,
    )?;
    let reference_numerical_timing = prefill_chunk_timing_report(
        config.total_tokens,
        &reference_schedule,
        1,
        reference.cuda_event_ms,
        reference.host_wall_ms,
    )?;
    let primary_timed = prefill_chunk_timing_report(
        config.total_tokens,
        &primary_schedule,
        config.iterations,
        timed_cuda_event_total_ms,
        timed_host_wall_total_ms,
    )?;
    // Instrumentation is a separate replay, after all uninstrumented timing
    // intervals have closed. It must exactly reproduce the primary outputs.
    #[cfg(feature = "qwen35-kv8-chunk-prefill")]
    let stage_profile = if config.profile_simt_stages {
        Some(prefill_stage_profile::run(
            &mut plans,
            &workspace,
            &primary_schedule,
            &stream,
            &terminal_hidden,
            &output_head,
            &input_hidden,
            &primary,
            config.profile_ffn_projections,
        )?)
    } else {
        None
    };
    #[cfg(feature = "qwen35-kv8-chunk-prefill")]
    let stage_profile_pass = stage_profile.as_ref().is_none_or(|p| p.pass);
    #[cfg(not(feature = "qwen35-kv8-chunk-prefill"))]
    let stage_profile_pass = true;
    let resident_allocation_delta =
        allocation_delta(allocation_start, cuda_device_allocation_snapshot())?;
    // Every run above ended with the owner/epoch terminal event completed.
    // Drop sealed operations before their buffers, then sample only after the
    // sole submission owner is quiescent; allocation counters are not atomic
    // multi-field snapshots and must not be interpreted during concurrent work.
    drop(primary_schedule);
    drop(reference_schedule);
    drop(output_head);
    drop(terminal_hidden);
    drop(workspace);
    drop(plans);
    drop(started_event);
    drop(stopped_event);
    drop(terminal_event);
    drop(stream);
    let cleanup_allocation_lifecycle =
        allocation_cleanup_report(allocation_start, cuda_device_allocation_snapshot())?;
    let diagnostic_model_prefill_tps_permitted = numerical_pass
        && semantic_pass
        && stage_profile_pass
        && owner_epoch_proof
        && timed_allocation_interval.pass
        && timed_submission_boundaries == BoundaryReport::default()
        && cleanup_allocation_lifecycle.pass;
    let performance_gate_300_tps_passed = diagnostic_model_prefill_tps_permitted
        && primary_timed.host_wall_tokens_per_second >= 300.0;
    let llama_355_tps_gate_passed = diagnostic_model_prefill_tps_permitted
        && primary_timed.host_wall_tokens_per_second >= 355.0;
    let stretch_500_tps_gate_passed = diagnostic_model_prefill_tps_permitted
        && primary_timed.host_wall_tokens_per_second >= 500.0;
    let status = if diagnostic_model_prefill_tps_permitted {
        if config.tensor_core_prefix_attention {
            "PASS_COMPONENT_QWEN35_TC_PREFIX_CHUNKED_CUBLAS"
        } else {
            "PASS_COMPONENT_QWEN35_PREFILL_CHUNKED_CUBLAS"
        }
    } else {
        if config.tensor_core_prefix_attention {
            "FAIL_COMPONENT_QWEN35_TC_PREFIX_CHUNKED_CUBLAS"
        } else {
            "FAIL_COMPONENT_QWEN35_PREFILL_CHUNKED_CUBLAS"
        }
    };
    
    
    let schema = QWEN35_PREFILL_CHUNKED_CUDA_SCHEMA;
    let schema = if config.simt_token_pair {
        "CQ-QWEN35-PREFILL-SIMT-TOKEN-PAIR-V1"
    } else if config.wide_prefill_microbatch {
        "CQ-QWEN35-PREFILL-WIDE-MICROBATCH-V2"
    } else if config.simt_unit_previous_scale {
        "CQ-QWEN35-PREFILL-SIMT-UNIT-PREVIOUS-SCALE-V1"
    } else if config.quad_weight_decode {
        "CQ-QWEN35-PREFILL-QUAD-WEIGHT-DECODE-V1"
    } else if config.paired_weight_decode {
        "CQ-QWEN35-PREFILL-PAIRED-WEIGHT-DECODE-V1"
    } else if false {
        "CQ-QWEN35-PREFILL-FUSED-SILU-Q8-V1"
    } else if config.simt_softmax_tile16 {
        "CQ-QWEN35-PREFILL-SIMT-SHARED-TILE16-V1"
    } else if config.simt_shared_gdn_scalars {
        "CQ-QWEN35-PREFILL-SIMT-SHARED-GDN-SCALARS-V1"
    } else if config.simt_shared_softmax {
        "CQ-QWEN35-PREFILL-SIMT-SHARED-SOFTMAX-V1"
    } else if config.layer_major_window {
        "CQ-QWEN35-PREFILL-LAYER-MAJOR-WINDOW-CUDA-V2"
    } else if config.reference_tensor_core_prefix_attention {
        "CQ-QWEN35-PREFILL-TC-REFERENCE-CUDA-V2"
    } else if config.direct_post_attention_activation
        || config.reference_direct_post_attention_activation
    {
        "crystal-quant.qwen35.tc-prefix-chunked-64-layer-cublas-sm70.component.v2"
    } else {
        schema
    };
    let terminal_input_token_id = token_ids[config.total_tokens - 1];
    let (cross_chunk_numerical_gate_passed, same_chunk_numerical_gate_passed) =
        prefill_chunk_comparison_gates(config, numerical_pass);

    Ok(Qwen35PrefillChunkedCudaReport {
        schema: schema.to_owned(),
        status: status.to_owned(),
        component_only: true,
        source_commit: config.source_commit.clone(),
        binary_path: binary_path.display().to_string(),
        binary_sha256: sha256_hex(&binary_bytes),
        device,
        config_sha256: sha256_hex(&config_bytes),
        gguf_path: config.gguf_path.display().to_string(),
        gguf_file_sha256: manifest.file_sha256.clone(),
        total_tokens: config.total_tokens,
        token_ids,
        input_hidden_sha256: sha256_f32_bits(&input_hidden),
        fused_silu_mul_q8: false,
        microbatch_admission: config.wide_prefill_microbatch.then(|| {
            PrefillMicrobatchAdmissionReport {
                explicit_wide_opt_in: true,
                ordinary_runtime_limit: 1024,
                compiled_native_limit: CudaQwen35PrefillCublasPlanV1::compiled_max_tokens(),
                admitted_runtime_limit: config.wide_prefill_microbatch_limit,
                primary_computation_rows: config.chunk_tokens.min(config.total_tokens),
                reference_computation_rows: config.reference_chunk_tokens.min(config.total_tokens),
                independent_input_batch_ubatch_api: false,
            }
        }),
        same_width_attention_replay_bit_exact,
        same_width_linear_replay_bit_exact,
        same_width_weight_decode_replay_bit_exact,
        #[cfg(feature = "qwen35-kv8-chunk-prefill")]
        stage_profile,
        schedule_comparison: (config.layer_major_window
            || config.reference_tensor_core_prefix_attention)
            .then(|| PrefillScheduleComparisonReport {
                primary_schedule: if config.layer_major_window {
                    "layer_major_linear512_full1024"
                } else {
                    "chunk_major"
                }
                .to_owned(),
                window_capacity_tokens: config.chunk_tokens.min(config.total_tokens),
                primary_linear_microbatch_limit: if config.layer_major_window {
                    512
                } else {
                    config.chunk_tokens
                },
                primary_full_attention_batch_limit: config.chunk_tokens,
                reference_chunk_tokens: config.reference_chunk_tokens,
                reference_tensor_core_attention: config.reference_tensor_core_prefix_attention,
                same_attention_algorithm: config.tensor_core_prefix_attention
                    == config.reference_tensor_core_prefix_attention,
                same_activation_policy: config.direct_post_attention_activation
                    == config.reference_direct_post_attention_activation,
                recurrent_state_order: "per_layer_persistent_owner_in_increasing_token_order"
                    .to_owned(),
                scratch_reuse_order: "single_stream_no_reuse_before_prior_consumers".to_owned(),
                external_semantic_parity_established: false,
            }),
        activation_policy_comparison: prefill_activation_policy_report(
            config,
            numerical_pass,
            semantic_pass,
        ),
        primary_progress_receipts: primary.progress_receipts.clone(),
        reference_progress_receipts: reference.progress_receipts.clone(),
        common_boundary_comparisons,
        numerical: PrefillBackboneNumericalReport {
            final_hidden,
            state_aggregate,
            states,
            final_hidden_relative_l2_max: numerical_contract.final_hidden_relative_l2_max,
            state_relative_l2_max: numerical_contract.state_relative_l2_max,
            cosine_similarity_min: numerical_contract.state_cosine_min,
            pass: numerical_pass,
        },
        numerical_contract,
        semantic: PrefillBackboneSemanticReport {
            terminal_position: config.total_tokens - 1,
            terminal_input_token_id,
            vocabulary_size: model_config.vocab_size,
            output_head_numerics: "ggml_q8_0_activation".to_owned(),
            normalized_hidden_operator,
            logits_operator,
            logits_backbone,
            logits_end_to_end,
            batched_argmax,
            sequential_argmax,
            device_argmax,
            all_argmax_equal,
            sequential_top1_margin,
            device_end_to_end_max_abs_error,
            sequential_argmax_stability_certified,
            pass: semantic_pass,
        },
        timing: PrefillChunkedTimingReport {
            warmup: config.warmup,
            primary_numerical: primary_numerical_timing,
            reference_numerical: reference_numerical_timing,
            primary_timed,
            primary_timed_host_samples_ms,
            primary_timed_cuda_samples_ms,
            primary_full_attention_path: if config.simt_token_pair {
                "simt_kv8_token_pair_shared_tile8_unit_scale_ordered"
            } else {
                prefill_attention_path_label(
                    config.tensor_core_prefix_attention,
                    cfg!(feature = "qwen35-kv8-chunk-prefill"),
                    config.simt_shared_softmax,
                )
            }
            .to_owned(),
            reference_full_attention_path: prefill_attention_path_label(
                config.reference_tensor_core_prefix_attention,
                cfg!(feature = "qwen35-kv8-chunk-prefill"),
                false,
            )
            .to_owned(),
            primary_linear_recurrence_path: if config.simt_shared_gdn_scalars {
                "simt_gdn_shared_scalars_ordered"
            } else {
                "simt_gdn_redundant_scalars_ordered"
            }
            .to_owned(),
            reference_linear_recurrence_path: "simt_gdn_redundant_scalars_ordered".to_owned(),
            primary_simt_attention_tile_positions: (cfg!(feature = "qwen35-kv8-chunk-prefill")
                && !config.tensor_core_prefix_attention)
                .then_some(if config.simt_softmax_tile16 { 16 } else { 8 }),
            reference_simt_attention_tile_positions: (cfg!(feature = "qwen35-kv8-chunk-prefill")
                && !config.reference_tensor_core_prefix_attention)
                .then_some(8),
            primary_projection_weight_decode: if config.quad_weight_decode {
                "quad_scalar_fp32_rn_f16_store"
            } else if config.paired_weight_decode {
                "paired_scalar_fp32_rn_f16_store"
            } else {
                "scalar_fp32_rn_f16_store"
            }
            .to_owned(),
            reference_projection_weight_decode: "scalar_fp32_rn_f16_store".to_owned(),
            primary_softmax_unit_previous_scale: config.simt_unit_previous_scale,
            reset_excluded_from_cuda_event: true,
            reset_included_in_host_wall: true,
            input_upload_excluded_from_cuda_event: true,
            input_upload_included_in_host_wall: true,
            terminal_output_head_included_in_cuda_event: true,
        },
        resources: PrefillChunkedResourceReport {
            setup_boundaries,
            primary_numerical_boundaries,
            reference_numerical_boundaries,
            timed_boundaries,
            timed_submission_boundaries,
            timed_allocation_interval,
            cleanup_allocation_lifecycle,
            allocation_delta: resident_allocation_delta,
            completion_poll_iterations,
            owner_epoch_proof,
            primary_progress_receipt_sequence_valid,
            reference_progress_receipt_sequence_valid,
        },
        claims: PrefillChunkedClaimsReport {
            real_model_weights_executed: true,
            complete_64_layer_transformer_executed: true,
            linear_layer_count,
            full_attention_layer_count,
            primary_chunk_schedule_executed: true,
            reference_chunk_schedule_executed: true,
            tensor_core_prefix_attention_executed: config.tensor_core_prefix_attention,
            matched_simt_reference_executed: config.tensor_core_prefix_attention
                && !config.reference_tensor_core_prefix_attention,
            stable_lse_merge_executed: config.tensor_core_prefix_attention,
            terminal_output_head_executed: true,
            cross_chunk_numerical_gate_passed,
            same_chunk_numerical_gate_passed,
            diagnostic_model_prefill_tps_permitted,
            performance_gate_300_tps_passed,
            llama_355_tps_gate_passed,
            stretch_500_tps_gate_passed,
            semantic_parity_established: false,
            product_tps_permitted: false,
            default_cli_enabled: false,
        },
    })
}

/// Execute the authenticated real-weight four-layer prefix using the original
/// CPU-projection/host-handoff baseline.
pub fn run_qwen35_mixed_prefix_cuda(
    config: &Qwen35MixedPrefixCudaConfig,
) -> Result<Qwen35MixedPrefixCudaReport, Qwen35MixedPrefixCudaError> {
    run_qwen35_mixed_prefix_cuda_with_mode(config, false)
}

/// Execute the opt-in device-resident handoff experiment.  This keeps hidden
/// activations on one CUDA stream between layers and performs linear-attention
/// input projection on the device; it remains a component-only diagnostic.
pub fn run_qwen35_device_resident_handoff_cuda(
    config: &Qwen35MixedPrefixCudaConfig,
) -> Result<Qwen35MixedPrefixCudaReport, Qwen35MixedPrefixCudaError> {
    run_qwen35_mixed_prefix_cuda_with_mode(config, true)
}

fn run_qwen35_mixed_prefix_cuda_with_mode(
    config: &Qwen35MixedPrefixCudaConfig,
    device_resident: bool,
) -> Result<Qwen35MixedPrefixCudaReport, Qwen35MixedPrefixCudaError> {
    run_qwen35_layer_stack_cuda(config, device_resident, MIXED_PREFIX_LAYER_COUNT)
}

/// Execute all 64 authenticated transformer blocks with device-resident
/// hidden handoff. After the measured resident path, an offline CPU-streamed
/// final norm/output head compares CUDA and CPU-oracle hidden states over the
/// full vocabulary. This remains a component diagnostic and publishes neither
/// tokens nor a product TPS claim.
pub fn run_qwen35_full_target_cuda(
    config: &Qwen35MixedPrefixCudaConfig,
) -> Result<Qwen35MixedPrefixCudaReport, Qwen35MixedPrefixCudaError> {
    run_qwen35_layer_stack_cuda(config, true, FULL_TARGET_LAYER_COUNT)
}

/// Execute a tokenizer-bound greedy target-only loop over the resident 64-layer
/// CUDA stack. This is an opt-in diagnostic: its visible wall includes full
/// logits readback and CPU argmax, while terminal state hashing is outside the
/// measured interval.
pub fn run_qwen35_autoregressive_cuda(
    config: &Qwen35AutoregressiveCudaConfig,
) -> Result<Qwen35AutoregressiveCudaReport, Qwen35MixedPrefixCudaError> {
    run_qwen35_autoregressive_cuda_with_mode(config, false, false, false)
}

/// Execute the same audited autoregressive sequence plus a separate measured
/// device token-handoff replay. The replay keeps top-2 selection and recurrent
/// embedding lookup on one CUDA stream; it remains diagnostic-only.
#[cfg(feature = "qwen35-device-token-handoff")]
pub fn run_qwen35_device_token_handoff_cuda(
    config: &Qwen35AutoregressiveCudaConfig,
) -> Result<Qwen35AutoregressiveCudaReport, Qwen35MixedPrefixCudaError> {
    run_qwen35_autoregressive_cuda_with_mode(config, true, false, false)
}

/// Populate recurrent state and full-attention KV with chunked cuBLAS prefill,
/// then continue through the same device token-handoff decode owner. This path
/// is opt-in and remains diagnostic-only.
#[cfg(feature = "qwen35-batched-prefill-handoff")]
pub fn run_qwen35_batched_prefill_device_token_handoff_cuda(
    config: &Qwen35AutoregressiveCudaConfig,
) -> Result<Qwen35AutoregressiveCudaReport, Qwen35MixedPrefixCudaError> {
    run_qwen35_autoregressive_cuda_with_mode(config, true, true, false)
}

/// Execute the real 16-layer KV8 cache persistence boundary after batched
/// prefill terminal completion, then decode against the no-roundtrip audit.
/// Persistence wall time is reported separately from steady decode.
#[cfg(all(
    feature = "qwen35-batched-prefill-handoff",
    feature = "qwen35-kv8-full-attention"
))]
pub fn run_qwen35_batched_prefill_kv8_ram_roundtrip_cuda(
    config: &Qwen35AutoregressiveCudaConfig,
) -> Result<Qwen35AutoregressiveCudaReport, Qwen35MixedPrefixCudaError> {
    run_qwen35_autoregressive_cuda_with_mode(config, true, true, true)
}

fn validate_autoregressive_prefill_policy(
    config: &Qwen35AutoregressiveCudaConfig,
    device_token_handoff: bool,
    batched_prefill: bool,
) -> Result<(), Qwen35MixedPrefixCudaError> {
    if config.prefill_shared_softmax
        && (!device_token_handoff
            || !batched_prefill
            || !cfg!(feature = "qwen35-kv8-chunk-prefill"))
    {
        return Err(invalid(
            "prefill shared softmax requires batched device handoff and KV8 chunk feature",
        ));
    }
    Ok(())
}

fn run_qwen35_autoregressive_cuda_with_mode(
    config: &Qwen35AutoregressiveCudaConfig,
    device_token_handoff: bool,
    batched_prefill: bool,
    kv8_ram_roundtrip: bool,
) -> Result<Qwen35AutoregressiveCudaReport, Qwen35MixedPrefixCudaError> {
    let input_and_model_setup_started = Instant::now();
    validate_autoregressive_prefill_policy(config, device_token_handoff, batched_prefill)?;
    if !cfg!(feature = "qwen35-autoregressive-cuda") {
        return Err(invalid("qwen35-autoregressive-cuda feature is required"));
    }
    if device_token_handoff && !cfg!(feature = "qwen35-device-token-handoff") {
        return Err(invalid("qwen35-device-token-handoff feature is required"));
    }
    if batched_prefill && !cfg!(feature = "qwen35-batched-prefill-handoff") {
        return Err(invalid(
            "qwen35-batched-prefill-handoff feature is required",
        ));
    }
    if batched_prefill && !device_token_handoff {
        return Err(invalid("batched prefill requires device token handoff"));
    }
    if kv8_ram_roundtrip
        && (!batched_prefill
            || !cfg!(feature = "qwen35-kv8-full-attention")
            || !config
                .capacity
                .is_multiple_of(KV8_RAM_PERSISTENCE_PAGE_POSITIONS))
    {
        return Err(invalid(
            "KV8 RAM roundtrip requires KV8 batched prefill and 512-position capacity pages",
        ));
    }
    if config.profile_device_handoff_decode && !device_token_handoff {
        return Err(invalid(
            "device-handoff decode profiling requires device token handoff",
        ));
    }
    if config.profile_device_handoff_layer_timing && !device_token_handoff {
        return Err(invalid(
            "device-handoff layer timing requires device token handoff",
        ));
    }
    if config.profile_device_handoff_layer_timing && config.generation_tokens > 32 {
        return Err(invalid(
            "device-handoff layer timing is bounded to 32 generation tokens",
        ));
    }
    if config.gguf_path.as_os_str().is_empty() || !config.gguf_path.is_file() {
        return Err(invalid("autoregressive GGUF path"));
    }
    if config.config_path.as_os_str().is_empty() || !config.config_path.is_file() {
        return Err(invalid("autoregressive config path"));
    }
    if config.tokenizer_path.as_os_str().is_empty() || !config.tokenizer_path.is_file() {
        return Err(invalid("autoregressive tokenizer path"));
    }
    #[cfg(feature = "qwen35-kv8-chunk-prefill")]
    let capacity_limit = QWEN35_KV8_FULL_ATTENTION_MAX_CAPACITY;
    #[cfg(not(feature = "qwen35-kv8-chunk-prefill"))]
    let capacity_limit = 4_096;
    if config.prompt_text.is_empty()
        || config.device_ordinal < 0
        || config.capacity == 0
        || config.capacity > capacity_limit
        || config.generation_tokens == 0
        || config.generation_tokens > 1_024
        || config.warmup > 100
    {
        return Err(invalid("autoregressive scalar bounds"));
    }

    let binary_path = std::env::current_exe().map_err(Qwen35Error::from)?;
    let binary_bytes = std::fs::read(&binary_path).map_err(Qwen35Error::from)?;
    let binary_sha256 = sha256_hex(&binary_bytes);
    let config_bytes = read_stable_regular_file(&config.config_path, "Qwen config")?;
    let config_sha256 = sha256_hex(&config_bytes);
    let prompt_text = if config.prompt_repeat == 0 {
        config.prompt_text.clone()
    } else {
        let unit = format!("{} ", config.prompt_text);
        unit.repeat(config.prompt_repeat).trim_end().to_owned()
    };
    let prompt_text_sha256 = sha256_hex(prompt_text.as_bytes());
    let model_config = Qwen35Config::parse(&config_bytes)?;
    model_config.validate_exact_profile()?;

    let tokenizer_bytes = read_stable_regular_file(&config.tokenizer_path, "Qwen tokenizer")?;
    let tokenizer_sha256 = sha256_hex(&tokenizer_bytes);
    let tokenizer = Tokenizer::from_bytes(&tokenizer_bytes)
        .map_err(|error| invalid(format!("tokenizer parse failed: {error}")))?;
    let tokenizer_vocab_size = tokenizer.get_vocab_size(false);
    if tokenizer_vocab_size > model_config.vocab_size {
        return Err(invalid("tokenizer vocabulary exceeds model vocabulary"));
    }
    let encoding = tokenizer
        .encode(prompt_text.as_str(), false)
        .map_err(|error| invalid(format!("tokenizer encode failed: {error}")))?;
    let prompt_token_ids = encoding
        .get_ids()
        .iter()
        .map(|token_id| usize::try_from(*token_id).map_err(|_| invalid("prompt token overflow")))
        .collect::<Result<Vec<_>, _>>()?;
    if prompt_token_ids.is_empty()
        || prompt_token_ids
            .iter()
            .any(|token_id| *token_id >= model_config.vocab_size)
    {
        return Err(invalid("tokenizer produced invalid prompt tokens"));
    }
    let bound_position_count = prompt_token_ids
        .len()
        .checked_sub(1)
        .and_then(|count| count.checked_add(config.generation_tokens))
        .ok_or_else(|| invalid("autoregressive position extent overflow"))?;
    if bound_position_count > config.capacity
        || bound_position_count > model_config.max_position_embeddings
    {
        return Err(invalid(
            "prompt prefix plus generation exceeds position capacity",
        ));
    }

    let manifest = GgufManifest::inspect(&config.gguf_path, GgufAuditOptions::default())?;
    let schema_summary =
        Qwen35TensorSchema::validate_manifest_allowing_mtp(&model_config, &manifest)?;
    if schema_summary.actual_tensor_count < 1_187 {
        return Err(invalid(
            "autoregressive runner requires the audited 1,187 tensor artifact",
        ));
    }
    let specs = (0..FULL_TARGET_LAYER_COUNT)
        .map(|layer| Qwen35LayerSpec::from_config(&model_config, layer))
        .collect::<Result<Vec<_>, _>>()?;
    if model_config.num_hidden_layers != FULL_TARGET_LAYER_COUNT
        || specs
            .iter()
            .filter(|spec| spec.kind == Qwen35LayerKind::LinearAttention)
            .count()
            != 48
        || specs
            .iter()
            .filter(|spec| spec.kind == Qwen35LayerKind::FullAttention)
            .count()
            != 16
    {
        return Err(invalid("autoregressive 48/16 layer partition"));
    }

    let mut reader = manifest.open_tensor_reader(&config.gguf_path)?;
    let embedding = manifest
        .tensor("token_embd.weight")
        .ok_or_else(|| invalid("autoregressive embedding descriptor"))?
        .clone();
    let first_hidden = linear::read_embedding_row(&mut reader, &embedding, prompt_token_ids[0])?;
    let device_properties = cuda_device_properties(config.device_ordinal)?;
    if device_properties.compute_capability.major < 7 {
        return Err(invalid("SM70 or newer device required"));
    }
    let device = DeviceReport {
        ordinal: device_properties.ordinal,
        name: device_properties.name,
        compute_capability: [
            device_properties.compute_capability.major as u16,
            device_properties.compute_capability.minor as u16,
        ],
        total_global_memory_bytes: device_properties.total_global_memory_bytes,
    };

    let stream = OwnedCudaStream::create(config.device_ordinal)?;
    let allocation_start = cuda_device_allocation_snapshot();
    let boundary_start = cuda_host_boundary_snapshot();
    let materials = load_layer_materials(
        &mut reader,
        &manifest,
        &model_config,
        FULL_TARGET_LAYER_COUNT,
    )?;
    let output_head_material = load_output_head_material(&mut reader, &manifest, &model_config)?;
    let mut plans: Vec<DevicePlan> = Vec::with_capacity(FULL_TARGET_LAYER_COUNT);
    for material in &materials {
        let input_source = if cfg!(feature = "qwen35-zero-copy-layer-handoff") {
            plans
                .last()
                .map(DevicePlan::final_hidden_f32_view)
                .transpose()?
        } else {
            None
        };
        match material {
            LayerMaterial::Linear(material) => {
                let projected =
                    project_linear(material, &first_hidden, model_config.rms_norm_eps as f32)?;
                plans.push(DevicePlan::Linear(build_linear_plan(
                    &stream,
                    material,
                    &first_hidden,
                    &projected,
                    true,
                    input_source,
                )?));
            }
            LayerMaterial::Full(material) => {
                plans.push(DevicePlan::Full(build_full_plan(
                    &stream,
                    material,
                    &model_config,
                    config.capacity,
                    bound_position_count,
                    input_source,
                )?));
            }
        }
    }
    let source_hidden = plans
        .last()
        .ok_or_else(|| invalid("autoregressive output-head source plan"))?
        .final_hidden_view()?;
    let output_head = build_device_output_head_plan(
        &stream,
        &output_head_material,
        source_hidden,
        model_config.hidden_size,
        model_config.vocab_size,
        model_config.rms_norm_eps as f32,
    )?;
    drop(output_head_material);
    drop(materials);
    #[cfg(feature = "qwen35-device-token-handoff")]
    let device_token_handoff_plan = if device_token_handoff {
        Some(build_device_token_handoff_plan(
            &stream,
            &mut reader,
            &embedding,
            &output_head,
            model_config.hidden_size,
            model_config.vocab_size,
            config.generation_tokens,
            &config.source_commit,
            &binary_sha256,
            &config_sha256,
            manifest.file_sha256.as_deref(),
            &prompt_text_sha256,
            config.capacity,
        )?)
    } else {
        None
    };
    let setup_allocations = allocation_delta(allocation_start, cuda_device_allocation_snapshot())?;
    let setup_boundaries = boundary_delta(boundary_start, cuda_host_boundary_snapshot())?;
    let mut prefill_mode_build_boundaries = BoundaryReport::default();
    let input_and_model_setup_host_wall_ms =
        input_and_model_setup_started.elapsed().as_secs_f64() * 1_000.0;

    let mut warmup_boundaries = BoundaryReport::default();
    for _ in 0..config.warmup {
        warmup_boundaries = add_boundaries(warmup_boundaries, reset_plans(&mut plans, &stream)?);
        let mode_build_before = cuda_host_boundary_snapshot();
        let mut warmup_prefill_mode = build_autoregressive_prefill_mode(
            batched_prefill,
            &stream,
            &plans,
            &mut reader,
            &embedding,
            &prompt_token_ids,
            &model_config,
            config.capacity,
            false,
        )?;
        prefill_mode_build_boundaries = add_boundaries(
            prefill_mode_build_boundaries,
            boundary_delta(mode_build_before, cuda_host_boundary_snapshot())?,
        );
        let warmup = run_autoregressive_sequence(
            &mut plans,
            &output_head,
            &mut reader,
            &embedding,
            &prompt_token_ids,
            1,
            &stream,
            false,
            &mut warmup_prefill_mode,
            config.dflash_q8_gguf_path.as_deref(),
            config.vision_safetensors_path.as_deref(),
            #[cfg(feature = "qwen35-device-token-handoff")]
            false,
            false,
        )?;
        warmup_boundaries = add_boundaries(warmup_boundaries, warmup.execution_boundaries);
        warmup_boundaries = add_boundaries(warmup_boundaries, warmup.state_readback_boundaries);
    }

    let primary_reset_boundaries = reset_plans(&mut plans, &stream)?;
    let primary_mode_build_before = cuda_host_boundary_snapshot();
    let mut primary_prefill_mode = build_autoregressive_prefill_mode(
        batched_prefill,
        &stream,
        &plans,
        &mut reader,
        &embedding,
        &prompt_token_ids,
        &model_config,
        config.capacity,
        false,
    )?;
    prefill_mode_build_boundaries = add_boundaries(
        prefill_mode_build_boundaries,
        boundary_delta(primary_mode_build_before, cuda_host_boundary_snapshot())?,
    );
    let primary = run_autoregressive_sequence(
        &mut plans,
        &output_head,
        &mut reader,
        &embedding,
        &prompt_token_ids,
        config.generation_tokens,
        &stream,
        !config.terminal_state_receipt_only,
        &mut primary_prefill_mode,
        config.dflash_q8_gguf_path.as_deref(),
        config.vision_safetensors_path.as_deref(),
        #[cfg(feature = "qwen35-device-token-handoff")]
        config.terminal_state_receipt_only,
        config.validate_q8_selector_terminal,
    )?;
    let replay_reset_boundaries = reset_plans(&mut plans, &stream)?;
    let replay_mode_build_before = cuda_host_boundary_snapshot();
    let mut replay_prefill_mode = build_autoregressive_prefill_mode(
        batched_prefill,
        &stream,
        &plans,
        &mut reader,
        &embedding,
        &prompt_token_ids,
        &model_config,
        config.capacity,
        false,
    )?;
    prefill_mode_build_boundaries = add_boundaries(
        prefill_mode_build_boundaries,
        boundary_delta(replay_mode_build_before, cuda_host_boundary_snapshot())?,
    );
    let replay = run_autoregressive_sequence(
        &mut plans,
        &output_head,
        &mut reader,
        &embedding,
        &prompt_token_ids,
        config.generation_tokens,
        &stream,
        !config.terminal_state_receipt_only,
        &mut replay_prefill_mode,
        config.dflash_q8_gguf_path.as_deref(),
        config.vision_safetensors_path.as_deref(),
        #[cfg(feature = "qwen35-device-token-handoff")]
        config.terminal_state_receipt_only,
        config.validate_q8_selector_terminal,
    )?;

    #[cfg(feature = "qwen35-device-token-handoff")]
    let (device_token_handoff_reset_boundaries, device_token_handoff_sequence) =
        if let Some(handoff) = device_token_handoff_plan.as_ref() {
            let reset = reset_plans(&mut plans, &stream)?;
            let handoff_mode_build_before = cuda_host_boundary_snapshot();
            let mut handoff_prefill_mode = build_autoregressive_prefill_mode(
                batched_prefill,
                &stream,
                &plans,
                &mut reader,
                &embedding,
                &prompt_token_ids,
                &model_config,
                config.capacity,
                config.prefill_shared_softmax,
            )?;
            prefill_mode_build_boundaries = add_boundaries(
                prefill_mode_build_boundaries,
                boundary_delta(handoff_mode_build_before, cuda_host_boundary_snapshot())?,
            );
            let sequence = run_device_token_handoff_sequence(
                &mut plans,
                &output_head,
                handoff,
                &mut reader,
                &embedding,
                &prompt_token_ids,
                config.generation_tokens,
                config.capacity,
                &stream,
                1,
                &mut handoff_prefill_mode,
                kv8_ram_roundtrip,
                config.profile_device_handoff_decode,
                config.profile_device_handoff_layer_timing,
                config.dflash_q8_gguf_path.as_deref(),
                config.vision_safetensors_path.as_deref(),
                config.terminal_state_receipt_only,
                config.validate_q8_selector_terminal,
            )?;
            (Some(reset), Some(sequence))
        } else {
            (None, None)
        };

    let generated_tokens_exact = primary.generated_token_ids == replay.generated_token_ids;
    let logits_exact = primary
        .tokens
        .iter()
        .map(|token| &token.logits_sha256)
        .eq(replay.tokens.iter().map(|token| &token.logits_sha256));
    let final_hidden_exact = primary.final_hidden_sha256 == replay.final_hidden_sha256;
    let final_state_exact = primary.final_state_sha256 == replay.final_state_sha256;
    let complete_replay_exact =
        generated_tokens_exact && logits_exact && final_hidden_exact && final_state_exact;
    let primary_report = autoregressive_sequence_report(&primary);
    let replay_report = autoregressive_sequence_report(&replay);
    #[cfg(feature = "qwen35-device-token-handoff")]
    let device_token_handoff_report = match (
        device_token_handoff_plan.as_ref(),
        device_token_handoff_sequence.as_ref(),
    ) {
        (Some(plan), Some(sequence)) => {
            Some(device_token_handoff_report(plan, sequence, &primary)?)
        }
        (None, None) => None,
        _ => return Err(invalid("device token-handoff plan/sequence closure")),
    };
    #[cfg(not(feature = "qwen35-device-token-handoff"))]
    let device_token_handoff_report = None::<DeviceTokenHandoffReport>;

    let cleanup_boundary_start = cuda_host_boundary_snapshot();
    let cleanup_allocation_start = cuda_device_allocation_snapshot();
    stream.synchronize()?;
    #[cfg(feature = "qwen35-device-token-handoff")]
    drop(device_token_handoff_plan);
    drop(output_head);
    drop(plans);
    drop(stream);
    let cleanup_boundaries = boundary_delta(cleanup_boundary_start, cuda_host_boundary_snapshot())?;
    let cleanup_allocation_end = cuda_device_allocation_snapshot();
    let final_allocation_delta =
        allocation_delta(cleanup_allocation_start, cleanup_allocation_end)?;
    let total_boundaries = boundary_delta(boundary_start, cuda_host_boundary_snapshot())?;
    let accounted_boundaries = [
        setup_boundaries.clone(),
        prefill_mode_build_boundaries.clone(),
        warmup_boundaries.clone(),
        primary_reset_boundaries.clone(),
        primary.execution_boundaries.clone(),
        primary.state_readback_boundaries.clone(),
        replay_reset_boundaries.clone(),
        replay.execution_boundaries.clone(),
        replay.state_readback_boundaries.clone(),
        cleanup_boundaries.clone(),
    ]
    .into_iter()
    .fold(BoundaryReport::default(), add_boundaries);
    #[cfg(feature = "qwen35-device-token-handoff")]
    let accounted_boundaries = {
        let mut accounted_boundaries = accounted_boundaries;
        if let Some(reset) = device_token_handoff_reset_boundaries.as_ref() {
            accounted_boundaries = add_boundaries(accounted_boundaries, reset.clone());
        }
        if let Some(sequence) = device_token_handoff_sequence.as_ref() {
            if let Some(persistence) = sequence.kv8_ram_persistence.as_ref() {
                accounted_boundaries =
                    add_boundaries(accounted_boundaries, persistence.total_boundaries.clone());
            }
            for boundaries in [
                &sequence.prefill_boundaries,
                &sequence.execution_boundaries,
                &sequence.receipt_readback_boundaries,
                &sequence.state_readback_boundaries,
            ] {
                accounted_boundaries = add_boundaries(accounted_boundaries, boundaries.clone());
            }
        }
        accounted_boundaries
    };
    let boundary_ledger_closed = total_boundaries == accounted_boundaries;
    let cleanup_zero = cleanup_allocation_end.live_allocations == allocation_start.live_allocations
        && cleanup_allocation_end.live_bytes == allocation_start.live_bytes;

    let generated_u32 = primary
        .generated_token_ids
        .iter()
        .map(|token_id| u32::try_from(*token_id).map_err(|_| invalid("generated token conversion")))
        .collect::<Result<Vec<_>, _>>()?;
    let generated_text = tokenizer
        .decode(&generated_u32, true)
        .map_err(|error| invalid(format!("tokenizer decode failed: {error}")))?;
    let generated_text_sha256 = sha256_hex(generated_text.as_bytes());
    let all_kernel_features_enabled = cfg!(feature = "qwen35-bf16-warp-row")
        && cfg!(feature = "qwen35-q8-packed-load")
        && cfg!(feature = "qwen35-gdn-warp-row")
        && cfg!(feature = "qwen35-fused-head-norm-rope")
        && cfg!(feature = "qwen35-warp-full-attention")
        && cfg!(feature = "qwen35-fused-ffn-swiglu")
        && (!batched_prefill || cfg!(feature = "qwen35-batched-prefill-handoff"))
        && (!kv8_ram_roundtrip || cfg!(feature = "qwen35-kv8-full-attention"));
    let device_token_handoff_contract = if device_token_handoff {
        device_token_handoff_report.as_ref().is_some_and(|report| {
            report.complete_numerical_contract && report.measured_boundary_contract_closed
        })
    } else {
        device_token_handoff_report.is_none()
    };
    let kv8_ram_persistence_contract = if kv8_ram_roundtrip {
        device_token_handoff_report
            .as_ref()
            .and_then(|report| report.kv8_ram_persistence.as_ref())
            .is_some_and(|report| report.complete_numerical_contract)
    } else {
        device_token_handoff_report
            .as_ref()
            .is_none_or(|report| report.kv8_ram_persistence.is_none())
    };
    let device_token_handoff_tokens_in_vocabulary =
        device_token_handoff_report.as_ref().is_none_or(|report| {
            report
                .generated_token_ids
                .iter()
                .all(|token_id| *token_id < model_config.vocab_size)
        });
    let device_token_handoff_count_valid = device_token_handoff_report
        .as_ref()
        .is_none_or(|report| report.generated_token_ids.len() == config.generation_tokens);
    let device_token_handoff_owner_proof = device_token_handoff_report
        .as_ref()
        .is_none_or(|report| report.owner_epoch_proof);
    let zero_copy_layer_handoff_enabled = cfg!(feature = "qwen35-zero-copy-layer-handoff");
    let zero_copy_layer_handoff_contract = zero_copy_layer_handoff_counts_closed(
        zero_copy_layer_handoff_enabled,
        (
            primary.inter_layer_d2d_copy_count,
            primary.inter_layer_d2d_bytes,
        ),
        (
            replay.inter_layer_d2d_copy_count,
            replay.inter_layer_d2d_bytes,
        ),
        device_token_handoff_report.as_ref().map(|report| {
            (
                report.inter_layer_d2d_copy_count,
                report.inter_layer_d2d_bytes,
            )
        }),
    );
    let all_tokens_in_vocabulary = prompt_token_ids
        .iter()
        .chain(primary.generated_token_ids.iter())
        .chain(replay.generated_token_ids.iter())
        .all(|token_id| *token_id < model_config.vocab_size)
        && device_token_handoff_tokens_in_vocabulary;
    let generated_token_count_valid = primary.generated_token_ids.len() == config.generation_tokens
        && replay.generated_token_ids.len() == config.generation_tokens
        && device_token_handoff_count_valid;
    let owner_epoch_proof = primary.proof && replay.proof && device_token_handoff_owner_proof;
    let batched_prefill_release_contract = if batched_prefill && prompt_token_ids.len() > 1 {
        [
            primary.batched_prefill.as_ref(),
            replay.batched_prefill.as_ref(),
            device_token_handoff_report
                .as_ref()
                .and_then(|report| report.batched_prefill.as_ref()),
        ]
        .into_iter()
        .all(|report| {
            report.is_some_and(|report| report.workspace_released_before_auxiliary_and_decode)
        })
    } else {
        true
    };
    let structural_pass = all_kernel_features_enabled
        && batched_prefill_release_contract
        && all_tokens_in_vocabulary
        && generated_token_count_valid
        && owner_epoch_proof
        && complete_replay_exact
        && device_token_handoff_contract
        && kv8_ram_persistence_contract
        && zero_copy_layer_handoff_contract
        && boundary_ledger_closed
        && cleanup_zero;

    Ok(Qwen35AutoregressiveCudaReport {
        schema: if kv8_ram_roundtrip {
            QWEN35_KV8_RAM_ROUNDTRIP_AUTOREGRESSIVE_CUDA_SCHEMA.to_owned()
        } else if batched_prefill {
            QWEN35_BATCHED_PREFILL_DEVICE_TOKEN_HANDOFF_AUTOREGRESSIVE_CUDA_SCHEMA.to_owned()
        } else if device_token_handoff {
            QWEN35_DEVICE_TOKEN_HANDOFF_AUTOREGRESSIVE_CUDA_SCHEMA.to_owned()
        } else {
            QWEN35_AUTOREGRESSIVE_CUDA_SCHEMA.to_owned()
        },
        status: if structural_pass && kv8_ram_roundtrip {
            "PASS_DIAGNOSTIC_KV8_RAM_ROUNDTRIP_AUTOREGRESSIVE_CUDA".to_owned()
        } else if structural_pass && batched_prefill {
            "PASS_DIAGNOSTIC_BATCHED_PREFILL_DEVICE_TOKEN_HANDOFF_AUTOREGRESSIVE_CUDA".to_owned()
        } else if structural_pass && device_token_handoff {
            "PASS_DIAGNOSTIC_DEVICE_TOKEN_HANDOFF_AUTOREGRESSIVE_CUDA".to_owned()
        } else if structural_pass {
            "PASS_DIAGNOSTIC_REAL_WEIGHT_AUTOREGRESSIVE_CUDA".to_owned()
        } else if kv8_ram_roundtrip {
            "NO_GO_DIAGNOSTIC_KV8_RAM_ROUNDTRIP_CONTRACT".to_owned()
        } else if batched_prefill {
            "NO_GO_DIAGNOSTIC_BATCHED_PREFILL_DEVICE_TOKEN_HANDOFF_CONTRACT".to_owned()
        } else if device_token_handoff {
            "NO_GO_DIAGNOSTIC_DEVICE_TOKEN_HANDOFF_CONTRACT".to_owned()
        } else {
            "NO_GO_DIAGNOSTIC_AUTOREGRESSIVE_CONTRACT".to_owned()
        },
        diagnostic_only: true,
        source_commit: config.source_commit.clone(),
        binary_path: binary_path.display().to_string(),
        binary_sha256,
        device,
        config_sha256,
        gguf_path: config.gguf_path.display().to_string(),
        gguf_file_sha256: manifest.file_sha256.clone(),
        tokenizer_path: config.tokenizer_path.display().to_string(),
        tokenizer_sha256,
        tokenizer_vocab_size,
        prompt_text_sha256,
        prompt_token_ids,
        generated_token_ids: primary.generated_token_ids.clone(),
        generated_text,
        generated_text_sha256,
        capacity: config.capacity,
        bound_position_count,
        generation_tokens: config.generation_tokens,
        kernel_features: AutoregressiveKernelFeatureReport {
            all_projection_q8_1: true,
            bf16_warp_row_fp64: cfg!(feature = "qwen35-bf16-warp-row"),
            q8_packed_load: cfg!(feature = "qwen35-q8-packed-load"),
            nvfp4_fused_lb10: cfg!(feature = "qwen35-nvfp4-fused-lb10"),
            gdn_warp_row: cfg!(feature = "qwen35-gdn-warp-row"),
            fused_head_norm_rope: cfg!(feature = "qwen35-fused-head-norm-rope"),
            warp_full_attention_fp64: cfg!(feature = "qwen35-warp-full-attention"),
            full_attention_parallel_exp: cfg!(feature = "qwen35-full-attention-parallel-exp"),
            grouped_gqa_online_softmax_v2: cfg!(feature = "qwen35-grouped-gqa-online-softmax"),
            kv8_full_attention_v3: cfg!(feature = "qwen35-kv8-full-attention"),
            grouped_gqa_v2: full::grouped_gqa_v2_geometry(true),
            split_rmsnorm: cfg!(feature = "qwen35-split-rmsnorm"),
            fused_ffn_swiglu: cfg!(feature = "qwen35-fused-ffn-swiglu"),
            device_token_handoff: device_token_handoff
                .then_some(cfg!(feature = "qwen35-device-token-handoff")),
            zero_copy_layer_handoff: zero_copy_layer_handoff_enabled.then_some(true),
            batched_prefill_cublas: batched_prefill.then_some(true),
        },
        timing: AutoregressiveTimingReport {
            input_and_model_setup_host_wall_ms,
            warmup_sequences: config.warmup,
            prompt_token_count: primary.prompt_prefix_tokens_processed + 1,
            unmeasured_prefill_token_count: primary.prompt_prefix_tokens_processed,
            generated_token_count: config.generation_tokens,
            primary_prefill_host_wall_ms: primary.prefill_host_wall_ms,
            primary_decode_host_wall_ms: primary.decode_host_wall_ms,
            primary_visible_tokens_per_second: primary_report.visible_tokens_per_second,
            primary_p50_token_ms: primary_report.p50_token_ms,
            primary_p95_token_ms: primary_report.p95_token_ms,
            replay_prefill_host_wall_ms: replay.prefill_host_wall_ms,
            replay_decode_host_wall_ms: replay.decode_host_wall_ms,
            replay_visible_tokens_per_second: replay_report.visible_tokens_per_second,
            replay_p50_token_ms: replay_report.p50_token_ms,
            replay_p95_token_ms: replay_report.p95_token_ms,
        },
        primary: primary_report,
        replay: replay_report,
        determinism: AutoregressiveDeterminismReport {
            generated_tokens_exact,
            logits_exact,
            final_hidden_exact,
            final_state_exact,
            complete_replay_exact,
        },
        device_token_handoff: device_token_handoff_report,
        resources: AutoregressiveResourceReport {
            setup_boundaries,
            warmup_boundaries,
            primary_reset_boundaries,
            primary_execution_boundaries: primary.execution_boundaries,
            primary_state_readback_boundaries: primary.state_readback_boundaries,
            replay_reset_boundaries,
            replay_execution_boundaries: replay.execution_boundaries,
            replay_state_readback_boundaries: replay.state_readback_boundaries,
            device_token_handoff_reset_boundaries: {
                #[cfg(feature = "qwen35-device-token-handoff")]
                {
                    device_token_handoff_reset_boundaries
                }
                #[cfg(not(feature = "qwen35-device-token-handoff"))]
                {
                    None
                }
            },
            device_token_handoff_prefill_boundaries: {
                #[cfg(feature = "qwen35-device-token-handoff")]
                {
                    device_token_handoff_sequence
                        .as_ref()
                        .map(|sequence| sequence.prefill_boundaries.clone())
                }
                #[cfg(not(feature = "qwen35-device-token-handoff"))]
                {
                    None
                }
            },
            device_token_handoff_kv8_persistence_boundaries: {
                #[cfg(feature = "qwen35-device-token-handoff")]
                {
                    device_token_handoff_sequence.as_ref().and_then(|sequence| {
                        sequence
                            .kv8_ram_persistence
                            .as_ref()
                            .map(|persistence| persistence.total_boundaries.clone())
                    })
                }
                #[cfg(not(feature = "qwen35-device-token-handoff"))]
                {
                    None
                }
            },
            device_token_handoff_execution_boundaries: {
                #[cfg(feature = "qwen35-device-token-handoff")]
                {
                    device_token_handoff_sequence
                        .as_ref()
                        .map(|sequence| sequence.execution_boundaries.clone())
                }
                #[cfg(not(feature = "qwen35-device-token-handoff"))]
                {
                    None
                }
            },
            device_token_handoff_receipt_readback_boundaries: {
                #[cfg(feature = "qwen35-device-token-handoff")]
                {
                    device_token_handoff_sequence
                        .as_ref()
                        .map(|sequence| sequence.receipt_readback_boundaries.clone())
                }
                #[cfg(not(feature = "qwen35-device-token-handoff"))]
                {
                    None
                }
            },
            device_token_handoff_state_readback_boundaries: {
                #[cfg(feature = "qwen35-device-token-handoff")]
                {
                    device_token_handoff_sequence
                        .as_ref()
                        .map(|sequence| sequence.state_readback_boundaries.clone())
                }
                #[cfg(not(feature = "qwen35-device-token-handoff"))]
                {
                    None
                }
            },
            cleanup_boundaries,
            total_boundaries,
            boundary_ledger_closed,
            setup_allocations,
            final_allocation_delta,
        },
        checks: AutoregressiveCheckReport {
            exact_model_profile: true,
            tokenizer_bound: true,
            position_extent_valid: bound_position_count <= config.capacity,
            all_kernel_features_enabled,
            all_logits_finite: true,
            all_tokens_in_vocabulary,
            owner_epoch_proof,
            generated_token_count_valid,
            replay_exact: complete_replay_exact,
            device_token_handoff_numerical_contract: device_token_handoff
                .then_some(device_token_handoff_contract),
            zero_copy_layer_handoff_contract: zero_copy_layer_handoff_enabled
                .then_some(zero_copy_layer_handoff_contract),
            boundary_ledger_closed,
            cleanup_zero,
        },
        resident_auxiliary: primary.resident_auxiliary.clone(),
        claims: AutoregressiveClaimsReport {
            diagnostic_autoregressive_executed: generated_token_count_valid && owner_epoch_proof,
            semantic_parity_established: false,
            llama_parity_established: false,
            product_tps_permitted: false,
            mtp_or_dflash_executed: false,
            default_cli_enabled: false,
        },
    })
}

fn run_qwen35_layer_stack_cuda(
    config: &Qwen35MixedPrefixCudaConfig,
    device_resident: bool,
    layer_count: usize,
) -> Result<Qwen35MixedPrefixCudaReport, Qwen35MixedPrefixCudaError> {
    if config.gguf_path.as_os_str().is_empty() || !config.gguf_path.is_file() {
        return Err(invalid("GGUF path"));
    }
    if config.config_path.as_os_str().is_empty() || !config.config_path.is_file() {
        return Err(invalid("config path"));
    }
    if config.ffn_operator_audit && cfg!(feature = "qwen35-fused-ffn-swiglu") {
        return Err(invalid(
            "FFN operator audit requires materialized gate/up buffers",
        ));
    }
    if config.device_ordinal < 0
        || config.capacity < TOKEN_COUNT
        || config.capacity > 4096
        || config.token_id >= linear::VOCAB - TOKEN_COUNT + 1
        || config.warmup > 1000
        || layer_count == 0
        || layer_count > FULL_TARGET_LAYER_COUNT
    {
        return Err(invalid("device, token, capacity, or warmup bounds"));
    }

    let binary_path = std::env::current_exe().map_err(Qwen35Error::from)?;
    let binary_bytes = std::fs::read(&binary_path).map_err(Qwen35Error::from)?;
    let binary_sha256 = sha256_hex(&binary_bytes);
    let config_bytes = std::fs::read(&config.config_path).map_err(Qwen35Error::from)?;
    let config_sha256 = sha256_hex(&config_bytes);
    let model_config = Qwen35Config::parse(&config_bytes)?;
    model_config.validate_exact_profile()?;
    let manifest = GgufManifest::inspect(&config.gguf_path, GgufAuditOptions::default())?;
    let schema_summary =
        Qwen35TensorSchema::validate_manifest_allowing_mtp(&model_config, &manifest)?;
    if schema_summary.actual_tensor_count < 1_187 {
        return Err(invalid(
            "layer stack requires the audited 1,187 tensor artifact",
        ));
    }
    let specs = (0..layer_count)
        .map(|layer| Qwen35LayerSpec::from_config(&model_config, layer))
        .collect::<Result<Vec<_>, _>>()?;
    let expected_kinds = specs.iter().map(|spec| spec.kind).collect::<Vec<_>>();
    if layer_count == MIXED_PREFIX_LAYER_COUNT {
        let expected_four = [
            Qwen35LayerKind::LinearAttention,
            Qwen35LayerKind::LinearAttention,
            Qwen35LayerKind::LinearAttention,
            Qwen35LayerKind::FullAttention,
        ];
        if expected_kinds.as_slice() != expected_four {
            return Err(invalid(
                "Qwen four-layer partition is not linear, linear, linear, full",
            ));
        }
    } else if layer_count == FULL_TARGET_LAYER_COUNT
        && model_config.num_hidden_layers != FULL_TARGET_LAYER_COUNT
    {
        return Err(invalid("Qwen full target profile must contain 64 layers"));
    }

    let second_token_id = config
        .token_id
        .checked_add(1)
        .ok_or_else(|| invalid("second token id overflow"))?;
    let mut reader = manifest.open_tensor_reader(&config.gguf_path)?;
    let embedding = manifest
        .tensor("token_embd.weight")
        .ok_or_else(|| invalid("embedding descriptor"))?
        .clone();
    let hidden_a = linear::read_embedding_row(&mut reader, &embedding, config.token_id)?;
    let hidden_b = linear::read_embedding_row(&mut reader, &embedding, second_token_id)?;
    let embeddings = vec![hidden_a, hidden_b];

    let reference_started = Instant::now();
    let mut oracle_reader = manifest.open_tensor_reader(&config.gguf_path)?;
    // Host QKV/Z/alpha/beta projections remain native F32, while the current
    // CUDA chain quantizes the SSM output and FFN activations. Use the explicit
    // selective oracle contract rather than the all-projection GGML mode.
    let mut oracle = GgufHybridPrefixExecutor::for_prefix_with_numerics(
        &mut oracle_reader,
        &model_config,
        layer_count,
        if device_resident {
            GgufProjectionNumerics::GgmlQ8_0Activation
        } else {
            GgufProjectionNumerics::GgmlOutputFfnQ8_0Activation
        },
    )?;
    let expected = run_cpu_sequence(&mut oracle, &embeddings, config.capacity, layer_count)?;
    let replay_expected = run_cpu_sequence(&mut oracle, &embeddings, config.capacity, layer_count)?;
    let oracle_replay_exact = expected_sequences_equal(&expected, &replay_expected);
    let reference_ms = reference_started.elapsed().as_secs_f64() * 1_000.0;

    let device_properties = cuda_device_properties(config.device_ordinal)?;
    if device_properties.compute_capability.major < 7 {
        return Err(invalid("SM70 or newer device required"));
    }
    let device = DeviceReport {
        ordinal: device_properties.ordinal,
        name: device_properties.name,
        compute_capability: [
            device_properties.compute_capability.major as u16,
            device_properties.compute_capability.minor as u16,
        ],
        total_global_memory_bytes: device_properties.total_global_memory_bytes,
    };
    let stream = OwnedCudaStream::create(config.device_ordinal)?;
    let allocation_start = cuda_device_allocation_snapshot();
    let boundary_start = cuda_host_boundary_snapshot();
    let materials = load_layer_materials(&mut reader, &manifest, &model_config, layer_count)?;
    let output_head_material = if layer_count == FULL_TARGET_LAYER_COUNT {
        Some(load_output_head_material(
            &mut reader,
            &manifest,
            &model_config,
        )?)
    } else {
        None
    };
    let mut plans = Vec::with_capacity(layer_count);
    for material in &materials {
        match material {
            LayerMaterial::Linear(material) => {
                let projected =
                    project_linear(material, &embeddings[0], model_config.rms_norm_eps as f32)?;
                plans.push(DevicePlan::Linear(build_linear_plan(
                    &stream,
                    material,
                    &embeddings[0],
                    &projected,
                    device_resident,
                    None,
                )?));
            }
            LayerMaterial::Full(material) => {
                plans.push(DevicePlan::Full(build_full_plan(
                    &stream,
                    material,
                    &model_config,
                    config.capacity,
                    TOKEN_COUNT,
                    None,
                )?));
            }
        }
    }
    let output_head = output_head_material
        .as_ref()
        .map(|material| {
            let source_hidden = plans
                .last()
                .ok_or_else(|| invalid("output-head source plan"))?
                .final_hidden_view()?;
            build_device_output_head_plan(
                &stream,
                material,
                source_hidden,
                model_config.hidden_size,
                model_config.vocab_size,
                model_config.rms_norm_eps as f32,
            )
        })
        .transpose()?;
    drop(output_head_material);

    // Warmup is part of setup accounting. It is reset before every warmup so
    // no state from this diagnostic phase can enter a measured sequence.
    for _ in 0..config.warmup {
        let _ = reset_plans(&mut plans, &stream)?;
        let mut ignored_input = BoundaryReport::default();
        let _ = run_device_sequence(
            &mut plans,
            &materials,
            &embeddings,
            &model_config,
            &stream,
            &mut ignored_input,
            device_resident,
            layer_count,
            false,
            false,
        )?;
        if let Some(output_head) = &output_head {
            output_head.submit(&stream)?;
            stream.synchronize()?;
        }
    }
    let setup_allocations = allocation_delta(allocation_start, cuda_device_allocation_snapshot())?;
    let setup_boundaries = boundary_delta(boundary_start, cuda_host_boundary_snapshot())?;

    let mut sequence_input_boundaries = BoundaryReport::default();
    sequence_input_boundaries =
        add_boundaries(sequence_input_boundaries, reset_plans(&mut plans, &stream)?);
    let first = run_device_sequence(
        &mut plans,
        &materials,
        &embeddings,
        &model_config,
        &stream,
        &mut sequence_input_boundaries,
        device_resident,
        layer_count,
        config.ffn_operator_audit,
        config.attention_operator_audit,
    )?;
    sequence_input_boundaries =
        add_boundaries(sequence_input_boundaries, reset_plans(&mut plans, &stream)?);
    let replay = run_device_sequence(
        &mut plans,
        &materials,
        &embeddings,
        &model_config,
        &stream,
        &mut sequence_input_boundaries,
        device_resident,
        layer_count,
        config.ffn_operator_audit,
        config.attention_operator_audit,
    )?;
    let resident_fast_sequence = if device_resident {
        sequence_input_boundaries =
            add_boundaries(sequence_input_boundaries, reset_plans(&mut plans, &stream)?);
        let sequence = run_resident_fast_sequence(
            &plans,
            output_head.as_ref(),
            &embeddings,
            &stream,
            layer_count,
        )?;
        sequence_input_boundaries =
            add_boundaries(sequence_input_boundaries, sequence.input_boundaries.clone());
        Some(sequence)
    } else {
        None
    };

    let mut all_layer_numerical_pass = true;
    let mut final_hidden_numerical_pass = true;
    let mut state_numerical_pass = true;
    let mut layer_reports = Vec::with_capacity(layer_count);
    let mut state_checkpoint_sha256 = Vec::with_capacity(layer_count);
    let mut trajectory_samples = Vec::with_capacity(layer_count * TOKEN_COUNT * 6);
    for layer in 0..layer_count {
        let kind = materials[layer].kind();
        let mut numerical = Vec::new();
        let ffn_operator_audit = first.layers[layer]
            .iter()
            .enumerate()
            .filter(|(_, actual)| actual.ffn_intermediates.is_some())
            .map(|(token, actual)| {
                ffn_operator_audit::audit(&materials[layer], actual, layer, token)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let attention_operator_audit = first.layers[layer]
            .iter()
            .enumerate()
            .filter(|(_, actual)| actual.attention_intermediates.is_some())
            .map(|(token, actual)| {
                attention_operator_audit::audit(
                    &materials[layer],
                    actual,
                    token.checked_sub(1).map(|p| &first.layers[layer][p]),
                    &expected.layers[layer][token],
                    layer,
                    token,
                    model_config.rms_norm_eps as f32,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let full_attention_operator_audit = first.layers[layer]
            .iter()
            .enumerate()
            .filter(|(_, actual)| actual.full_attention_intermediates.is_some())
            .map(|(token, actual)| {
                full_attention_operator_audit::audit(&materials[layer], actual, layer, token)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let input_hidden_max_abs_diff = first.layers[layer]
            .iter()
            .zip(&expected.layers[layer])
            .map(|(actual, expected)| {
                compare(
                    "input_hidden",
                    &actual.input_hidden,
                    &expected.input_hidden,
                    f32::INFINITY,
                    0.0,
                )
            })
            .map(|report| report.max_abs_diff)
            .fold(0.0_f32, f32::max);
        let input_hidden_max_rel_diff = first.layers[layer]
            .iter()
            .zip(&expected.layers[layer])
            .map(|(actual, expected)| {
                compare(
                    "input_hidden",
                    &actual.input_hidden,
                    &expected.input_hidden,
                    f32::INFINITY,
                    0.0,
                )
            })
            .map(|report| report.max_rel_diff)
            .fold(0.0_f32, f32::max);
        let mut actual_state_hashes = Vec::with_capacity(TOKEN_COUNT);
        let mut expected_state_hashes = Vec::with_capacity(TOKEN_COUNT);
        for token in 0..TOKEN_COUNT {
            let actual = &first.layers[layer][token];
            let expected_layer = &expected.layers[layer][token];
            let (attention_abs, attention_rel) = if kind == Qwen35LayerKind::FullAttention {
                (2.0e-3, 4.0e-3)
            } else {
                (5.0e-3, 2.0e-4)
            };
            let mut record = |suffix: &str,
                              actual: &[f32],
                              expected: &[f32],
                              absolute_tolerance: f32,
                              relative_tolerance: f32| {
                let name = format!("layer{layer}.token{token}.{suffix}");
                trajectory_samples.push(trajectory_sample(&name, actual, expected));
                compare(
                    &name,
                    actual,
                    expected,
                    absolute_tolerance,
                    relative_tolerance,
                )
            };
            let mut entries = vec![
                record(
                    "attention_output",
                    &actual.attention_output,
                    &expected_layer.output.attention_output,
                    attention_abs,
                    attention_rel,
                ),
                record(
                    "attention_residual",
                    &actual.attention_residual,
                    &expected_layer.output.attention_residual,
                    1.0e-2,
                    5.0e-4,
                ),
                record(
                    "post_attention_norm",
                    &actual.post_norm,
                    &expected_layer.output.post_attention_norm,
                    1.0e-2,
                    5.0e-4,
                ),
                record(
                    "ffn_output",
                    &actual.ffn_output,
                    &expected_layer.output.ffn_output,
                    1.0e-2,
                    5.0e-4,
                ),
                record(
                    "final_hidden",
                    &actual.final_hidden,
                    &expected_layer.output.hidden,
                    1.0e-2,
                    5.0e-4,
                ),
            ];
            let expected_state = if kind == Qwen35LayerKind::FullAttention {
                expected_full_state(
                    &expected_layer.state,
                    config.capacity,
                    materials[layer].spec().kv_heads * materials[layer].spec().head_dim,
                )
            } else {
                expected_layer.state.clone()
            };
            entries.push(record(
                "state",
                &state_values(actual),
                &expected_state,
                1.0e-2,
                5.0e-4,
            ));
            let entry_pass = entries.iter().all(|entry| entry.pass);
            all_layer_numerical_pass &= entry_pass;
            final_hidden_numerical_pass &= entries[4].pass;
            state_numerical_pass &= entries[5].pass;
            let actual_state = state_values(actual);
            actual_state_hashes.push(sha256_f32_bits(&actual_state));
            expected_state_hashes.push(sha256_f32_bits(&expected_state));
            numerical.extend(entries);
        }
        state_checkpoint_sha256.push(expected_state_hashes.clone());
        layer_reports.push(LayerReport {
            layer,
            kind: match kind {
                Qwen35LayerKind::LinearAttention => "linear_attention".to_owned(),
                Qwen35LayerKind::FullAttention => "full_attention".to_owned(),
            },
            native_projection_device: device_resident || kind == Qwen35LayerKind::FullAttention,
            token_count: TOKEN_COUNT,
            input_hidden_sha256: first.layers[layer]
                .iter()
                .map(|entry| sha256_f32_bits(&entry.input_hidden))
                .collect(),
            final_hidden_sha256: first.layers[layer]
                .iter()
                .map(|entry| sha256_f32_bits(&entry.final_hidden))
                .collect(),
            state_sha256: actual_state_hashes,
            state_expected_sha256: expected_state_hashes,
            input_hidden_max_abs_diff,
            input_hidden_max_rel_diff,
            ffn_operator_audit,
            attention_operator_audit,
            full_attention_operator_audit,
            numerical,
            handoff_from_previous: layer > 0,
        });
    }

    let replay_exact = first
        .layers
        .iter()
        .zip(&replay.layers)
        .all(|(left, right)| {
            left.len() == right.len()
                && left.iter().zip(right).all(|(left, right)| {
                    equal_f32_bits(&left.attention_output, &right.attention_output)
                        && equal_f32_bits(&left.attention_residual, &right.attention_residual)
                        && equal_f32_bits(&left.post_norm, &right.post_norm)
                        && equal_f32_bits(&left.ffn_output, &right.ffn_output)
                        && equal_f32_bits(&left.final_hidden, &right.final_hidden)
                        && equal_f32_bits(&left.state, &right.state)
                        && equal_f32_bits(&left.history, &right.history)
                })
        });
    let mut completion_boundaries = add_boundaries(
        first.completion_boundaries.clone(),
        replay.completion_boundaries.clone(),
    );
    let mut readback_boundaries = add_boundaries(
        first.readback_boundaries.clone(),
        replay.readback_boundaries.clone(),
    );
    if let Some(fast) = &resident_fast_sequence {
        completion_boundaries =
            add_boundaries(completion_boundaries, fast.hot_path_boundaries.clone());
        readback_boundaries = add_boundaries(readback_boundaries, fast.readback_boundaries.clone());
    }
    let completion_proof_valid = first.proof
        && replay.proof
        && resident_fast_sequence
            .as_ref()
            .is_none_or(|sequence| sequence.proof);
    let resident_fast_path = resident_fast_sequence.as_ref().map(|sequence| {
        let aggregate_host_wall_ms = sequence.token_host_wall_ms.iter().sum::<f64>();
        let aggregate_backbone_cuda_event_ms =
            sequence.token_backbone_cuda_event_ms.iter().sum::<f64>();
        let aggregate_output_head_cuda_event_ms =
            (!sequence.token_output_head_cuda_event_ms.is_empty())
                .then(|| sequence.token_output_head_cuda_event_ms.iter().sum::<f64>());
        let aggregate_cuda_event_ms = sequence.token_cuda_event_ms.iter().sum::<f64>();
        let matches_audited_sequence = sequence.final_hidden.len() == TOKEN_COUNT
            && sequence
                .final_hidden
                .iter()
                .zip(&first.layers[layer_count - 1])
                .all(|(fast, audited)| equal_f32_bits(fast, &audited.final_hidden));
        ResidentFastPathReport {
            scope: if output_head.is_some() {
                format!("resident_{layer_count}_layer_stack_with_gpu_output_head")
            } else {
                format!("resident_{layer_count}_layer_stack_without_output_head")
            },
            token_count: sequence.final_hidden.len(),
            layer_submission_count: layer_count.saturating_mul(sequence.final_hidden.len()),
            inter_layer_d2d_copy_count: sequence.inter_layer_d2d_copy_count,
            inter_layer_d2d_bytes: sequence.inter_layer_d2d_bytes,
            token_host_wall_ms: sequence.token_host_wall_ms.clone(),
            token_backbone_cuda_event_ms: sequence.token_backbone_cuda_event_ms.clone(),
            token_output_head_cuda_event_ms: output_head
                .as_ref()
                .map(|_| sequence.token_output_head_cuda_event_ms.clone()),
            token_cuda_event_ms: sequence.token_cuda_event_ms.clone(),
            token_layer_cuda_event_ms: sequence.token_layer_cuda_event_ms.clone(),
            aggregate_host_wall_ms,
            aggregate_backbone_cuda_event_ms,
            aggregate_output_head_cuda_event_ms,
            aggregate_cuda_event_ms,
            host_wall_tokens_per_second: sequence.final_hidden.len() as f64 * 1_000.0
                / aggregate_host_wall_ms,
            backbone_cuda_event_tokens_per_second: sequence.final_hidden.len() as f64 * 1_000.0
                / aggregate_backbone_cuda_event_ms,
            cuda_event_tokens_per_second: sequence.final_hidden.len() as f64 * 1_000.0
                / aggregate_cuda_event_ms,
            terminal_poll_iterations: sequence.polls,
            owner_epoch_proof: sequence.proof,
            final_hidden_sha256: sequence
                .final_hidden
                .iter()
                .map(|hidden| sha256_f32_bits(hidden))
                .collect(),
            final_norm_sha256: output_head.as_ref().map(|_| {
                sequence
                    .final_norm
                    .iter()
                    .map(|hidden| sha256_f32_bits(hidden))
                    .collect()
            }),
            logits_sha256: output_head.as_ref().map(|_| {
                sequence
                    .logits
                    .iter()
                    .map(|logits| sha256_f32_bits(logits))
                    .collect()
            }),
            output_weight_payload_bytes: output_head
                .as_ref()
                .map(|plan| plan.output_weight_payload_bytes),
            output_weight_payload_sha256: output_head
                .as_ref()
                .map(|plan| plan.output_weight_payload_sha256.clone()),
            matches_audited_sequence,
            input_boundaries: sequence.input_boundaries.clone(),
            hot_path_boundaries: sequence.hot_path_boundaries.clone(),
            readback_boundaries: sequence.readback_boundaries.clone(),
            output_head_executed: output_head.is_some()
                && sequence.final_norm.len() == TOKEN_COUNT
                && sequence.logits.len() == TOKEN_COUNT,
            decode_executed: false,
            product_tps_permitted: false,
        }
    });
    let resident_fast_path_matches_audited_sequence = resident_fast_path
        .as_ref()
        .map(|report| report.matches_audited_sequence);
    let trajectory = (layer_count == FULL_TARGET_LAYER_COUNT).then(|| TrajectoryReport {
        reference: "independent_cpu_ggml_q8_all_projection_activation_recursive_trajectory"
            .to_owned(),
        sample_count: trajectory_samples.len(),
        all: aggregate_trajectory(trajectory_samples.iter()),
        final_hidden: aggregate_trajectory(
            trajectory_samples
                .iter()
                .filter(|sample| sample.name.ends_with(".final_hidden")),
        ),
        state: aggregate_trajectory(
            trajectory_samples
                .iter()
                .filter(|sample| sample.name.ends_with(".state")),
        ),
        llama_cuda_reference_bound: false,
        thresholds_calibrated: false,
        qualification_permitted: false,
        samples: trajectory_samples,
    });
    let semantic_logits = if layer_count == FULL_TARGET_LAYER_COUNT {
        let resident = resident_fast_sequence
            .as_ref()
            .ok_or_else(|| invalid("full-target semantic probe requires resident output"))?;
        Some(run_semantic_logits_probe(
            &mut reader,
            &model_config,
            &first,
            resident,
            &expected,
            &[config.token_id, second_token_id],
        )?)
    } else {
        None
    };
    let semantic_logits_finite = semantic_logits
        .as_ref()
        .map(|report| report.all_logits_finite);
    let semantic_argmax_equal = semantic_logits
        .as_ref()
        .map(|report| report.all_device_argmax_equal);
    let semantic_argmax_stability_certified = semantic_logits
        .as_ref()
        .map(|report| report.all_device_argmax_stability_certified);
    let cleanup_boundary_start = cuda_host_boundary_snapshot();
    let cleanup_allocation_start = cuda_device_allocation_snapshot();
    stream.synchronize()?;
    drop(output_head);
    drop(plans);
    drop(materials);
    drop(stream);
    let cleanup_boundaries = boundary_delta(cleanup_boundary_start, cuda_host_boundary_snapshot())?;
    let cleanup_allocation_end = cuda_device_allocation_snapshot();
    let final_allocation_delta =
        allocation_delta(cleanup_allocation_start, cleanup_allocation_end)?;
    let total_boundaries = boundary_delta(boundary_start, cuda_host_boundary_snapshot())?;
    let accounted_boundaries = add_boundaries(
        add_boundaries(
            add_boundaries(
                add_boundaries(setup_boundaries.clone(), sequence_input_boundaries.clone()),
                completion_boundaries.clone(),
            ),
            readback_boundaries.clone(),
        ),
        cleanup_boundaries.clone(),
    );
    let boundary_ledger_closed = total_boundaries == accounted_boundaries;
    let cleanup_zero = cleanup_allocation_end.live_allocations == allocation_start.live_allocations
        && cleanup_allocation_end.live_bytes == allocation_start.live_bytes;
    let handoff = device_resident.then(|| HandoffReport {
        device_resident: true,
        device_linear_projection: true,
        inter_layer_d2d_copy_count: first
            .inter_layer_d2d_copy_count
            .saturating_add(replay.inter_layer_d2d_copy_count)
            .saturating_add(
                resident_fast_sequence
                    .as_ref()
                    .map_or(0, |sequence| sequence.inter_layer_d2d_copy_count),
            ),
        inter_layer_d2d_bytes: first
            .inter_layer_d2d_bytes
            .saturating_add(replay.inter_layer_d2d_bytes)
            .saturating_add(
                resident_fast_sequence
                    .as_ref()
                    .map_or(0, |sequence| sequence.inter_layer_d2d_bytes),
            ),
        host_projection_token_count: first
            .host_projection_token_count
            .saturating_add(replay.host_projection_token_count),
        host_hidden_handoff_token_count: first
            .host_hidden_handoff_token_count
            .saturating_add(replay.host_hidden_handoff_token_count)
            .saturating_add(resident_fast_sequence.as_ref().map_or(0, |_| TOKEN_COUNT)),
    });

    Ok(Qwen35MixedPrefixCudaReport {
        schema: if layer_count == FULL_TARGET_LAYER_COUNT {
            QWEN35_FULL_TARGET_CUDA_SCHEMA.to_owned()
        } else if device_resident {
            QWEN35_DEVICE_RESIDENT_HANDOFF_CUDA_SCHEMA.to_owned()
        } else {
            QWEN35_MIXED_PREFIX_CUDA_SCHEMA.to_owned()
        },
        status: if layer_count == FULL_TARGET_LAYER_COUNT {
            "DIAGNOSTIC_REAL_WEIGHT_FULL_64_LAYER_TARGET_CUDA".to_owned()
        } else if device_resident {
            "DIAGNOSTIC_REAL_WEIGHT_FOUR_LAYER_DEVICE_RESIDENT_HANDOFF_CUDA".to_owned()
        } else {
            "DIAGNOSTIC_REAL_WEIGHT_FOUR_LAYER_MIXED_PREFIX_CUDA".to_owned()
        },
        component_only: true,
        source_commit: config.source_commit.clone(),
        binary_path: binary_path.display().to_string(),
        binary_sha256,
        device,
        config_sha256,
        gguf_path: config.gguf_path.display().to_string(),
        gguf_file_sha256: manifest.file_sha256.clone(),
        layer_indices: (0..layer_count).collect(),
        layer_kinds: expected_kinds
            .iter()
            .map(|kind| match kind {
                Qwen35LayerKind::LinearAttention => "linear_attention".to_owned(),
                Qwen35LayerKind::FullAttention => "full_attention".to_owned(),
            })
            .collect(),
        token_ids: vec![config.token_id, second_token_id],
        capacity: config.capacity,
        kernel_features: KernelFeatureReport {
            all_projection_q8_1: true,
            bf16_warp_row_fp64: cfg!(feature = "qwen35-bf16-warp-row"),
            q8_packed_load: cfg!(feature = "qwen35-q8-packed-load"),
            nvfp4_fused_lb10: cfg!(feature = "qwen35-nvfp4-fused-lb10"),
            gdn_warp_row: cfg!(feature = "qwen35-gdn-warp-row"),
            fused_head_norm_rope: cfg!(feature = "qwen35-fused-head-norm-rope"),
            full_attention_parallel_exp: cfg!(feature = "qwen35-full-attention-parallel-exp"),
            grouped_gqa_online_softmax_v2: cfg!(feature = "qwen35-grouped-gqa-online-softmax"),
            kv8_full_attention_v3: cfg!(feature = "qwen35-kv8-full-attention"),
            grouped_gqa_v2: full::grouped_gqa_v2_geometry(true),
            split_rmsnorm: cfg!(feature = "qwen35-split-rmsnorm"),
        },
        oracle: OracleReport {
            cpu_hybrid_prefix: true,
            projection_numerics: if device_resident {
                "ggml_q8_0_activation_all_projections"
            } else {
                "layer_aware_linear_input_f32_full_input_q8_output_ffn_q8"
            }
            .to_owned(),
            replay_exact: oracle_replay_exact,
            final_hidden_sha256: expected
                .final_hidden
                .iter()
                .map(|hidden| sha256_f32_bits(hidden))
                .collect(),
            state_checkpoint_sha256,
        },
        timing: TimingReport {
            warmup_sequences: config.warmup,
            token_count: TOKEN_COUNT,
            sequence_host_wall_ms: first.host_wall_ms,
            replay_host_wall_ms: replay.host_wall_ms,
            layer_host_wall_ms: first.layer_host_wall_ms,
            reference_ms,
        },
        resident_fast_path,
        trajectory,
        semantic_logits,
        completion: CompletionReport {
            layer_submission_count: layer_count
                * TOKEN_COUNT
                * (2 + usize::from(resident_fast_sequence.is_some())),
            poll_iterations: first.polls.saturating_add(replay.polls).saturating_add(
                resident_fast_sequence
                    .as_ref()
                    .map_or(0, |sequence| sequence.polls),
            ),
            nonblocking_poll_only: completion_proof_valid,
            owner_epoch_proof: completion_proof_valid,
            final_sequence_proof: completion_proof_valid,
            replay_exact,
        },
        layers: layer_reports,
        resources: ResourceReport {
            setup_boundaries,
            sequence_input_boundaries,
            completion_boundaries,
            readback_boundaries,
            cleanup_boundaries,
            total_boundaries,
            boundary_ledger_closed,
            setup_allocations,
            final_allocation_delta,
        },
        checks: CheckReport {
            profile_valid: true,
            layer_partition_valid: true,
            oracle_replay_exact,
            all_layer_numerical_pass,
            final_hidden_numerical_pass,
            state_numerical_pass,
            completion_proof_valid,
            boundary_ledger_closed,
            cleanup_zero,
            same_stream_handoff: true,
            native_full_attention_projection_device: true,
            resident_fast_path_matches_audited_sequence,
            semantic_logits_finite,
            semantic_argmax_equal,
            semantic_argmax_stability_certified,
        },
        claims: ClaimsReport {
            real_weight_four_layer_mixed_prefix_cuda: layer_count == MIXED_PREFIX_LAYER_COUNT
                && all_layer_numerical_pass
                && completion_proof_valid,
            real_weight_full_64_layer_cuda: (layer_count == FULL_TARGET_LAYER_COUNT).then_some(
                all_layer_numerical_pass
                    && completion_proof_valid
                    && resident_fast_path_matches_audited_sequence == Some(true),
            ),
            cpu_projection_handoff: !device_resident,
            native_full_attention_projection_device: true,
            complete_transformer_executed: false,
            tokenizer_executed: false,
            decode_executed: false,
            semantic_parity_established: false,
            product_tps_permitted: false,
            default_cli_enabled: false,
        },
        handoff,
    })
}

fn state_values(readback: &DeviceLayerReadback) -> Vec<f32> {
    if readback.history.is_empty() {
        readback.state.clone()
    } else {
        let mut values = readback.history.clone();
        values.extend_from_slice(&readback.state);
        values
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn material_range_keeps_original_layer_indices() {
        validate_material_range(&(0..64), 64).unwrap();
        for layer in [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63] {
            let range = layer..layer + 1;
            validate_material_range(&range, 64).unwrap();
            assert_eq!(range.collect::<Vec<_>>(), vec![layer]);
        }
    }

    #[test]
    fn material_range_rejects_empty_reversed_or_outside_model() {
        for range in [0..0, 64..64, 7..3, 63..65, 64..65, 0..usize::MAX] {
            assert!(validate_material_range(&range, 64).is_err());
        }
        assert!(validate_material_range(&(0..1), 0).is_err());
    }

    #[cfg(feature = "qwen35-prefill-cublas")]
    #[test]
    fn prefill_schedule_options_are_default_off() {
        let config = Qwen35PrefillChunkedCudaConfig::default();
        assert!(!config.layer_major_window);
        assert!(!config.reference_tensor_core_prefix_attention);
        assert!(!config.simt_shared_softmax);
        assert!(!config.simt_softmax_tile16);
        assert!(!config.simt_unit_previous_scale);
        assert!(!config.simt_shared_gdn_scalars);
        assert!(!config.profile_simt_stages);
        assert!(!config.profile_ffn_projections);
        assert!(!config.paired_weight_decode);
        assert!(!config.quad_weight_decode);
        assert_eq!(validate_prefill_chunked_config(&config).unwrap(), 512);
        assert!(prefill_activation_policy_report(&config, true, true).is_none());
    }

    #[cfg(feature = "qwen35-prefill-cublas")]
    #[test]
    fn prefill_attention_labels_distinguish_f32_kv8_and_tc() {
        assert_eq!(
            prefill_attention_path_label(false, false, false),
            "simt_f32_cache_materialized_softmax"
        );
        assert_eq!(
            prefill_attention_path_label(false, true, false),
            "simt_kv8_grouped_gqa_online_softmax"
        );
        assert_eq!(
            prefill_attention_path_label(false, true, true),
            "simt_kv8_grouped_gqa_shared_softmax_ordered"
        );
        assert_eq!(
            prefill_attention_path_label(true, true, false),
            "tensor_core_prefix_current_stable_lse_merge"
        );
    }

    #[cfg(feature = "qwen35-prefill-cublas")]
    #[test]
    fn prefill_shared_softmax_requires_kv8_and_retained_non_tc_paths() {
        let mut c = Qwen35PrefillChunkedCudaConfig {
            chunk_tokens: 1024,
            reference_chunk_tokens: 1024,
            simt_shared_softmax: true,
            ..Default::default()
        };
        assert_eq!(
            validate_prefill_chunked_config(&c).is_ok(),
            cfg!(feature = "qwen35-kv8-chunk-prefill")
        );
        c.tensor_core_prefix_attention = true;
        assert!(validate_prefill_chunked_config(&c).is_err());
        c.tensor_core_prefix_attention = false;
        c.direct_post_attention_activation = true;
        assert!(validate_prefill_chunked_config(&c).is_err());
        c.direct_post_attention_activation = false;
        c.reference_direct_post_attention_activation = true;
        assert!(validate_prefill_chunked_config(&c).is_err());
        c.reference_direct_post_attention_activation = false;
        c.reference_tensor_core_prefix_attention = true;
        assert!(validate_prefill_chunked_config(&c).is_err());
        c.reference_tensor_core_prefix_attention = false;
        c.layer_major_window = true;
        assert!(validate_prefill_chunked_config(&c).is_err());
    }

    #[cfg(feature = "qwen35-prefill-cublas")]
    #[test]
    fn prefill_shared_gdn_scalars_is_bounded_retained_and_feature_gated() {
        let mut c = Qwen35PrefillChunkedCudaConfig {
            chunk_tokens: 1024,
            reference_chunk_tokens: 1024,
            simt_shared_gdn_scalars: true,
            ..Default::default()
        };
        let enabled = cfg!(feature = "qwen35-kv8-chunk-prefill");
        assert_eq!(validate_prefill_chunked_config(&c).is_ok(), enabled);
        c.simt_shared_softmax = true;
        c.profile_simt_stages = true;
        assert_eq!(validate_prefill_chunked_config(&c).is_ok(), enabled);
        c.total_tokens = 4097;
        assert!(validate_prefill_chunked_config(&c).is_err());
        c.total_tokens = 129;
        c.chunk_tokens = 64;
        c.reference_chunk_tokens = 64;
        assert_eq!(validate_prefill_chunked_config(&c).is_ok(), enabled);
        for flag in 0..5 {
            let mut bad = c.clone();
            match flag {
                0 => bad.tensor_core_prefix_attention = true,
                1 => bad.reference_tensor_core_prefix_attention = true,
                2 => bad.direct_post_attention_activation = true,
                3 => bad.reference_direct_post_attention_activation = true,
                _ => bad.layer_major_window = true,
            }
            assert!(validate_prefill_chunked_config(&bad).is_err());
        }
    }

    #[cfg(feature = "qwen35-prefill-cublas")]
    #[test]
    fn prefill_tile16_requires_shared_softmax_and_retains_all_admission_gates() {
        let mut c = Qwen35PrefillChunkedCudaConfig {
            simt_softmax_tile16: true,
            chunk_tokens: 1024,
            reference_chunk_tokens: 1024,
            ..Default::default()
        };
        assert!(validate_prefill_chunked_config(&c).is_err());
        c.simt_shared_softmax = true;
        assert_eq!(
            validate_prefill_chunked_config(&c).is_ok(),
            cfg!(feature = "qwen35-kv8-chunk-prefill")
        );
        c.tensor_core_prefix_attention = true;
        assert!(validate_prefill_chunked_config(&c).is_err());
        c.tensor_core_prefix_attention = false;
        c.total_tokens = 4097;
        assert!(validate_prefill_chunked_config(&c).is_err());
    }

    #[cfg(feature = "qwen35-prefill-cublas")]
    #[test]
    fn ffn_profile_requires_separate_retained_simt_replay() {
        let mut c = Qwen35PrefillChunkedCudaConfig {
            profile_ffn_projections: true,
            ..Default::default()
        };
        assert!(validate_prefill_chunked_config(&c).is_err());
        c.profile_simt_stages = true;
        assert_eq!(
            validate_prefill_chunked_config(&c).is_ok(),
            cfg!(feature = "qwen35-kv8-chunk-prefill")
                && !cfg!(feature = "qwen35-prefill-fused-silu-q8")
        );
        c.tensor_core_prefix_attention = true;
        assert!(validate_prefill_chunked_config(&c).is_err());
        c.tensor_core_prefix_attention = false;
        c.direct_post_attention_activation = true;
        assert!(validate_prefill_chunked_config(&c).is_err());
    }

    #[cfg(feature = "qwen35-prefill-cublas")]
    #[test]
    fn paired_weight_decode_is_bounded_retained_and_reference_independent() {
        let mut c = Qwen35PrefillChunkedCudaConfig {
            paired_weight_decode: true,
            chunk_tokens: 64,
            reference_chunk_tokens: 64,
            total_tokens: 129,
            ..Default::default()
        };
        assert_eq!(
            validate_prefill_chunked_config(&c).is_ok(),
            cfg!(feature = "qwen35-kv8-chunk-prefill")
        );
        assert!(!c.simt_shared_softmax && !c.simt_shared_gdn_scalars);
        c.total_tokens = 4097;
        assert!(validate_prefill_chunked_config(&c).is_err());
        c.total_tokens = 129;
        c.reference_tensor_core_prefix_attention = true;
        assert!(validate_prefill_chunked_config(&c).is_err());
        c.reference_tensor_core_prefix_attention = false;
        c.reference_direct_post_attention_activation = true;
        assert!(validate_prefill_chunked_config(&c).is_err());
        c.reference_direct_post_attention_activation = false;
        c.layer_major_window = true;
        assert!(validate_prefill_chunked_config(&c).is_err());
    }

    #[cfg(feature = "qwen35-prefill-cublas")]
    #[test]
    fn unit_previous_scale_requires_shared_tile8_and_retained_feature() {
        let mut c = Qwen35PrefillChunkedCudaConfig {
            simt_unit_previous_scale: true,
            ..Default::default()
        };
        assert!(validate_prefill_chunked_config(&c).is_err());
        c.simt_shared_softmax = true;
        assert_eq!(
            validate_prefill_chunked_config(&c).is_ok(),
            cfg!(feature = "qwen35-kv8-chunk-prefill")
        );
        c.simt_softmax_tile16 = true;
        assert!(validate_prefill_chunked_config(&c).is_err());
        c.simt_softmax_tile16 = false;
        c.tensor_core_prefix_attention = true;
        assert!(validate_prefill_chunked_config(&c).is_err());
    }

    #[cfg(feature = "qwen35-prefill-cublas")]
    #[test]
    fn token_pair_requires_feature_and_plain_shared_tile8_unit_scale() {
        assert!(!Qwen35PrefillChunkedCudaConfig::default().simt_token_pair);
        let mut c = Qwen35PrefillChunkedCudaConfig {
            simt_token_pair: true,
            ..Default::default()
        };
        assert!(validate_prefill_chunked_config(&c).is_err());
        c.simt_shared_softmax = true;
        assert!(validate_prefill_chunked_config(&c).is_err());
        c.simt_unit_previous_scale = true;
        assert_eq!(
            validate_prefill_chunked_config(&c).is_ok(),
            cfg!(feature = "qwen35-kv8-token-pair")
        );
        for incompatible in 0..6 {
            let mut bad = c.clone();
            match incompatible {
                0 => bad.simt_softmax_tile16 = true,
                1 => bad.tensor_core_prefix_attention = true,
                2 => bad.reference_tensor_core_prefix_attention = true,
                3 => bad.direct_post_attention_activation = true,
                4 => bad.reference_direct_post_attention_activation = true,
                _ => bad.layer_major_window = true,
            }
            assert!(validate_prefill_chunked_config(&bad).is_err());
        }
    }

    #[cfg(feature = "qwen35-prefill-cublas")]
    #[test]
    fn quad_decode_requires_retained_feature_and_excludes_pair_selection() {
        let mut c = Qwen35PrefillChunkedCudaConfig {
            quad_weight_decode: true,
            chunk_tokens: 64,
            reference_chunk_tokens: 64,
            ..Default::default()
        };
        assert_eq!(
            validate_prefill_chunked_config(&c).is_ok(),
            cfg!(feature = "qwen35-kv8-chunk-prefill")
        );
        c.paired_weight_decode = true;
        assert!(validate_prefill_chunked_config(&c).is_err());
        c.paired_weight_decode = false;
        c.direct_post_attention_activation = true;
        assert!(validate_prefill_chunked_config(&c).is_err());
    }

    #[cfg(feature = "qwen35-prefill-cublas")]
    #[test]
    fn prefill_plain_1024_bound_does_not_enable_long_context_or_tc() {
        let mut c = Qwen35PrefillChunkedCudaConfig {
            chunk_tokens: 1024,
            reference_chunk_tokens: 512,
            ..Default::default()
        };
        assert_eq!(validate_prefill_chunked_config(&c).unwrap(), 1024);
        c.chunk_tokens = 1025;
        assert!(validate_prefill_chunked_config(&c).is_err());
        c.chunk_tokens = 1024;
        c.total_tokens = 4097;
        assert!(validate_prefill_chunked_config(&c).is_err());
    }

    #[cfg(feature = "qwen35-prefill-cublas")]
    #[test]
    fn wide_microbatch_requires_explicit_feature_and_request() {
        assert!(!Qwen35PrefillChunkedCudaConfig::default().wide_prefill_microbatch);
        let mut c = Qwen35PrefillChunkedCudaConfig {
            chunk_tokens: 2048,
            reference_chunk_tokens: 1024,
            ..Default::default()
        };
        assert!(validate_prefill_chunked_config(&c).is_err());
        c.wide_prefill_microbatch = true;
        assert_eq!(
            validate_prefill_chunked_config(&c).is_ok(),
            cfg!(feature = "qwen35-wide-prefill-microbatch")
        );
    }

    #[cfg(feature = "qwen35-prefill-cublas")]
    #[test]
    fn wide_microbatch_limit_does_not_implicitly_enable_wide_admission() {
        let c = Qwen35PrefillChunkedCudaConfig {
            wide_prefill_microbatch_limit: 4096,
            ..Default::default()
        };
        assert!(validate_prefill_chunked_config(&c).is_err());
        for limit in [0, 1024, 2049, 4097, usize::MAX] {
            let c = Qwen35PrefillChunkedCudaConfig {
                wide_prefill_microbatch: true,
                wide_prefill_microbatch_limit: limit,
                ..Default::default()
            };
            assert!(validate_prefill_chunked_config(&c).is_err());
        }
    }

    

    

    

    #[cfg(feature = "qwen35-prefill-cublas")]
    #[test]
    fn prefill_stage_profile_requires_kv8_retained_and_no_tc() {
        let mut c = Qwen35PrefillChunkedCudaConfig {
            profile_simt_stages: true,
            ..Default::default()
        };
        assert_eq!(
            validate_prefill_chunked_config(&c).is_ok(),
            cfg!(feature = "qwen35-kv8-chunk-prefill")
        );
        c.tensor_core_prefix_attention = true;
        assert!(validate_prefill_chunked_config(&c).is_err());
        c.tensor_core_prefix_attention = false;
        c.direct_post_attention_activation = true;
        assert!(validate_prefill_chunked_config(&c).is_err());
        c.direct_post_attention_activation = false;
        c.layer_major_window = true;
        assert!(validate_prefill_chunked_config(&c).is_err());
    }

    #[cfg(feature = "qwen35-prefill-cublas")]
    #[test]
    fn prefill_partial_chunk_claim_uses_executed_width_not_requested_limit() {
        for total in [1_usize, 128, 512, 513, 1024, 3584] {
            let config = Qwen35PrefillChunkedCudaConfig {
                total_tokens: total,
                chunk_tokens: 1024,
                reference_chunk_tokens: 512,
                ..Default::default()
            };
            assert_eq!(
                prefill_chunk_comparison_gates(&config, true),
                (total > 512, total <= 512)
            );
            assert_eq!(
                prefill_chunk_comparison_gates(&config, false),
                (false, false)
            );
        }
    }

    #[cfg(feature = "qwen35-prefill-cublas")]
    #[test]
    fn prefill_direct_policy_never_claims_unmeasured_retained_compatibility() {
        for primary in [false, true] {
            for reference in [false, true] {
                if !primary && !reference {
                    continue;
                }
                let config = Qwen35PrefillChunkedCudaConfig {
                    direct_post_attention_activation: primary,
                    reference_direct_post_attention_activation: reference,
                    ..Default::default()
                };
                for numerical in [false, true] {
                    for semantic in [false, true] {
                        let r =
                            prefill_activation_policy_report(&config, numerical, semantic).unwrap();
                        assert_eq!(
                            r.measured_retained_reference_gate_passed,
                            !reference && numerical && semantic
                        );
                        assert_eq!(r.reference_is_retained_q8, !reference);
                        assert_eq!(
                            r.primary_linear_input_direct_projection_count,
                            if primary { 4 } else { 0 }
                        );
                        assert_eq!(
                            r.reference_linear_input_direct_projection_count,
                            if reference { 4 } else { 0 }
                        );
                        assert!(!r.upstream_attention_projection_policy_unchanged);
                        assert!(r.full_attention_input_projection_policy_unchanged);
                        assert!(!r.external_semantic_parity_established);
                    }
                }
            }
        }
    }

    #[cfg(feature = "qwen35-prefill-cublas")]
    #[test]
    fn prefill_schedule_rejects_invalid_scalar_bounds_without_cuda() {
        let base = Qwen35PrefillChunkedCudaConfig::default();
        for config in [
            Qwen35PrefillChunkedCudaConfig {
                total_tokens: 0,
                ..base.clone()
            },
            Qwen35PrefillChunkedCudaConfig {
                chunk_tokens: 0,
                ..base.clone()
            },
            Qwen35PrefillChunkedCudaConfig {
                reference_chunk_tokens: 0,
                ..base.clone()
            },
            Qwen35PrefillChunkedCudaConfig {
                chunk_tokens: 1025,
                ..base.clone()
            },
            Qwen35PrefillChunkedCudaConfig {
                total_tokens: usize::MAX,
                ..base.clone()
            },
            Qwen35PrefillChunkedCudaConfig {
                device_ordinal: -1,
                ..base.clone()
            },
            Qwen35PrefillChunkedCudaConfig {
                iterations: 0,
                ..base.clone()
            },
            Qwen35PrefillChunkedCudaConfig {
                reference_tensor_core_prefix_attention: true,
                ..base
            },
        ] {
            assert!(validate_prefill_chunked_config(&config).is_err());
        }
    }

    #[cfg(feature = "qwen35-prefill-cublas")]
    #[test]
    fn prefill_layer_major_rejects_feature_off_without_cuda() {
        let config = Qwen35PrefillChunkedCudaConfig {
            tensor_core_prefix_attention: true,
            layer_major_window: true,
            chunk_tokens: 1024,
            ..Default::default()
        };
        assert!(validate_prefill_chunked_config(&config).is_err());
    }

    

    

    

    

    

    #[cfg(feature = "qwen35-prefill-cublas")]
    #[test]
    fn progress_receipts_require_contiguous_ordered_terminal() {
        let receipt = |index, start, end, epoch, terminal| PrefillChunkProgressReceipt {
            chunk_index: index,
            start_position: start,
            end_position_exclusive: end,
            token_count: end - start,
            submission_epoch: epoch,
            owner_epoch_proof: true,
            terminal,
            input_hidden_sha256: "0".repeat(64),
            output_hidden_sha256: None,
            state_cache_sha256: None,
            state_cache_layer_sha256: None,
        };
        let valid = vec![
            receipt(0, 0, 512, 11, false),
            receipt(1, 512, 1024, 12, true),
        ];
        assert!(progress_receipt_sequence_valid(&valid, 1024));

        let mut gap = valid.clone();
        gap[1].start_position = 513;
        assert!(!progress_receipt_sequence_valid(&gap, 1024));

        let mut stale_epoch = valid.clone();
        stale_epoch[1].submission_epoch = 11;
        assert!(!progress_receipt_sequence_valid(&stale_epoch, 1024));

        let mut early_terminal = valid;
        early_terminal[0].terminal = true;
        assert!(!progress_receipt_sequence_valid(&early_terminal, 1024));
    }

    #[cfg(feature = "qwen35-prefill-cublas")]
    #[test]
    fn progress_receipt_comparison_reports_first_divergent_layer() {
        let make = |index, end, epoch, layer_digests| PrefillChunkProgressReceipt {
            chunk_index: index,
            start_position: end - 512,
            end_position_exclusive: end,
            token_count: 512,
            submission_epoch: epoch,
            owner_epoch_proof: true,
            terminal: true,
            input_hidden_sha256: "0".repeat(64),
            output_hidden_sha256: Some("1".repeat(64)),
            state_cache_sha256: Some("2".repeat(64)),
            state_cache_layer_sha256: Some(layer_digests),
        };
        let primary = make(0, 512, 1, vec!["a".into(), "b".into(), "c".into()]);
        let reference = make(0, 512, 2, vec!["a".into(), "x".into(), "c".into()]);
        let comparisons = compare_progress_receipts(&[primary], &[reference]);
        assert_eq!(comparisons.len(), 1);
        assert_eq!(comparisons[0].first_different_layer, Some(1));
        assert_eq!(comparisons[0].different_layer_count, 1);
        assert_eq!(comparisons[0].state_cache_exact, Some(false));
        assert_eq!(comparisons[0].output_hidden_exact, Some(true));
    }

    

    

    

    

    

    #[cfg(feature = "qwen35-prefill-cublas")]
    #[test]
    fn tc_chunk_contract_rejects_observed_512_token_logits_drift() {
        let tc = PrefillChunkedNumericalContract::for_attention_path(true);
        let simt = PrefillChunkedNumericalContract::for_attention_path(false);
        let mut sample = trajectory_sample("logits", &[1.0, 2.0], &[1.0, 2.0]);
        sample.relative_l2_error = Some(0.0150199029453541);
        assert!(!tc.logits_pass(&sample));
        assert!(simt.logits_pass(&sample));
        sample.relative_l2_error = Some(0.015);
        assert!(tc.logits_pass(&sample));
        sample.relative_l2_error = Some(f64::NAN);
        assert!(!tc.logits_pass(&sample));
        sample.relative_l2_error = None;
        assert!(!tc.logits_pass(&sample));
        assert_eq!(tc.final_hidden_relative_l2_max, 0.015);
        assert_eq!(tc.final_hidden_cosine_min, 0.9999);
    }

    #[cfg(feature = "qwen35-prefill-cublas")]
    #[test]
    fn prefill_allocation_interval_allows_resident_owners_but_not_churn() {
        let start = augur_backend::CudaDeviceAllocationSnapshot {
            successful_allocations: 10,
            successful_frees: 2,
            live_allocations: 8,
            live_bytes: 4096,
            peak_live_allocations: 10,
            peak_live_bytes: 8192,
        };
        assert!(allocation_interval_report(start, start).unwrap().pass);
        let mut end = start;
        end.successful_allocations += 1;
        end.successful_frees += 1;
        assert!(!allocation_interval_report(start, end).unwrap().pass);
        assert!(allocation_cleanup_report(start, end).unwrap().pass);
        end = start;
        end.live_bytes += 1;
        assert!(!allocation_interval_report(start, end).unwrap().pass);
        assert!(!allocation_cleanup_report(start, end).unwrap().pass);
        end = start;
        end.live_allocations += 1;
        assert!(!allocation_interval_report(start, end).unwrap().pass);
        assert!(!allocation_cleanup_report(start, end).unwrap().pass);
        end = start;
        end.successful_allocations -= 1;
        assert!(allocation_interval_report(start, end).is_err());
        assert!(allocation_cleanup_report(start, end).is_err());
        end = start;
        end.successful_frees += 1;
        assert!(!allocation_cleanup_report(start, end).unwrap().pass);
    }

    #[test]
    fn autoregressive_shared_prefill_is_opt_in_and_rejects_unconsumed_modes() {
        let mut config = Qwen35AutoregressiveCudaConfig::default();
        assert!(!config.prefill_shared_softmax);
        assert!(validate_autoregressive_prefill_policy(&config, false, false).is_ok());
        config.prefill_shared_softmax = true;
        for (handoff, batched) in [(false, false), (true, false), (false, true)] {
            assert!(validate_autoregressive_prefill_policy(&config, handoff, batched).is_err());
        }
        assert_eq!(
            validate_autoregressive_prefill_policy(&config, true, true).is_ok(),
            cfg!(feature = "qwen35-kv8-chunk-prefill"),
        );
    }

    #[test]
    fn packed_rope_rows_preserve_native_bits_across_chunk_boundaries() {
        for count in [1, 513, 2049] {
            let (cos, sin) = native_rope_table_span(10_000_000.0, count, 64).unwrap();
            assert_eq!(cos.len(), count * 32);
            assert_eq!(sin.len(), count * 32);
            for position in 0..count {
                let (expected_cos, expected_sin) =
                    native_rope_tables(10_000_000.0, position, 64).unwrap();
                let row = position * 32..(position + 1) * 32;
                assert!(cos[row.clone()]
                    .iter()
                    .zip(expected_cos)
                    .all(|(a, b)| a.to_bits() == b.to_bits()));
                assert!(sin[row]
                    .iter()
                    .zip(expected_sin)
                    .all(|(a, b)| a.to_bits() == b.to_bits()));
            }
        }
    }

    #[test]
    fn packed_rope_rejects_invalid_or_overflowing_extents_without_cuda() {
        for (theta, count, dim) in [
            (1.0, 1, 64),
            (f32::NAN, 1, 64),
            (f32::INFINITY, 1, 64),
            (10_000_000.0, 0, 64),
            (10_000_000.0, 1, 0),
            (10_000_000.0, 1, 63),
            (10_000_000.0, usize::MAX, 64),
        ] {
            assert!(native_rope_table_span(theta, count, dim).is_err());
        }
    }

    #[test]
    fn prefill_release_requires_exact_resident_baseline_and_free_conservation() {
        let before = augur_backend::CudaDeviceAllocationSnapshot {
            successful_allocations: 12,
            successful_frees: 2,
            live_allocations: 10,
            live_bytes: 8192,
            peak_live_allocations: 12,
            peak_live_bytes: 16384,
        };
        let after = augur_backend::CudaDeviceAllocationSnapshot {
            successful_frees: 6,
            live_allocations: 6,
            live_bytes: 4096,
            ..before
        };
        let mut report = AutoregressiveBatchedPrefillReport {
            attention_path: "kv8_simt_shared_scalar_softmax".to_owned(),
            prefix_tokens: 512,
            chunk_tokens: 512,
            chunk_count: 1,
            compute_host_wall_ms: 1.0,
            cuda_event_ms: 1.0,
            execution_boundaries: BoundaryReport::default(),
            submission_boundaries: BoundaryReport::default(),
            execution_allocations: AllocationReport::default(),
            execution_allocation_stable: true,
            resident_live_bytes_before_build: 4096,
            resident_live_allocations_before_build: 6,
            live_bytes_before_release: 8192,
            live_allocations_before_release: 10,
            workspace_release_host_wall_ms: 0.0,
            workspace_released_bytes: 0,
            workspace_released_allocations: 0,
            live_bytes_after_release: 8192,
            workspace_released_before_auxiliary_and_decode: false,
        };
        assert!(prefill_release_closed(&report, before, after));
        for bad in [
            augur_backend::CudaDeviceAllocationSnapshot {
                live_bytes: 4097,
                ..after
            },
            augur_backend::CudaDeviceAllocationSnapshot {
                live_bytes: 4095,
                ..after
            },
            augur_backend::CudaDeviceAllocationSnapshot {
                live_allocations: 5,
                ..after
            },
            augur_backend::CudaDeviceAllocationSnapshot {
                successful_frees: 5,
                ..after
            },
            augur_backend::CudaDeviceAllocationSnapshot {
                successful_allocations: 13,
                ..after
            },
        ] {
            assert!(!prefill_release_closed(&report, before, bad));
        }
        report.execution_allocation_stable = false;
        assert!(!prefill_release_closed(&report, before, after));
    }

    #[test]
    fn state_hash_payload_order_is_stable() {
        let readback = DeviceLayerReadback {
            input_hidden: Vec::new(),
            ffn_intermediates: None,
            attention_intermediates: None,
            full_attention_intermediates: None,
            attention_output: Vec::new(),
            attention_residual: Vec::new(),
            post_norm: Vec::new(),
            ffn_output: Vec::new(),
            final_hidden: Vec::new(),
            state: vec![1.0, 2.0],
            history: vec![3.0],
            polls: 0,
            proof: true,
            completion_boundaries: BoundaryReport::default(),
            readback_boundaries: BoundaryReport::default(),
        };
        assert_eq!(state_values(&readback), vec![3.0, 1.0, 2.0]);
    }

    #[test]
    fn full_cache_padding_keeps_key_and_value_regions_separate() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_eq!(
            expected_full_state(&values, 2, 2),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );
    }

    #[test]
    fn malformed_full_cache_state_fails_closed() {
        assert!(expected_full_state(&[1.0, 2.0, 3.0], 2, 2).is_empty());
        assert!(expected_full_state(&[1.0, 2.0, 3.0, 4.0], usize::MAX, 2).is_empty());
        assert!(expected_full_state(&[1.0, 2.0], 2, 0).is_empty());
    }

    #[test]
    fn exact_trajectory_sample_has_unit_cosine_and_zero_error() {
        let sample = trajectory_sample("exact", &[1.0, -2.0, 3.0], &[1.0, -2.0, 3.0]);
        assert!(sample.shape_match);
        assert_eq!(sample.finite_pair_count, 3);
        assert_eq!(sample.rmse, Some(0.0));
        assert_eq!(sample.relative_l2_error, Some(0.0));
        assert_eq!(sample.cosine_similarity, Some(1.0));
    }

    #[test]
    fn trajectory_aggregate_preserves_worst_sample() {
        let exact = trajectory_sample("exact", &[1.0, 0.0], &[1.0, 0.0]);
        let orthogonal = trajectory_sample("orthogonal", &[0.0, 1.0], &[1.0, 0.0]);
        let aggregate = aggregate_trajectory([&exact, &orthogonal].into_iter());
        assert_eq!(aggregate.sample_count, 2);
        assert_eq!(aggregate.finite_metric_count, 2);
        assert_eq!(aggregate.max_relative_l2_error, Some(2.0_f64.sqrt()));
        assert_eq!(aggregate.min_cosine_similarity, Some(0.0));
    }

    #[test]
    fn ranked_logits_break_ties_toward_lowest_token_id() {
        let ranked = ranked_logits(&[1.0, 3.0, 3.0, -2.0], 3);
        assert_eq!(
            ranked
                .iter()
                .map(|entry| entry.token_id)
                .collect::<Vec<_>>(),
            vec![1, 2, 0]
        );
        assert_eq!(top1_margin(&ranked), 0.0);
    }

    #[test]
    fn ranked_overlap_and_error_bound_are_replayed_from_values() {
        let actual = ranked_logits(&[9.0, 8.5, 1.0, 7.0], 3);
        let oracle = ranked_logits(&[8.8, 8.6, 7.5, 0.0], 3);
        assert_eq!(ranked_overlap(&actual, &oracle, 3), 2);
        assert!((max_abs_error(&[9.0, 8.5], &[8.8, 8.6]) - 0.2).abs() < 1.0e-5);
        let oracle_margin = top1_margin(&oracle);
        assert!(oracle_margin <= 2.0 * max_abs_error(&[9.0, 8.5], &[8.8, 8.6]));
    }

    #[test]
    fn nearest_rank_percentile_covers_boundaries_and_invalid_inputs() {
        let values = [4.0, 1.0, 3.0, 2.0];
        assert_eq!(nearest_rank_percentile(&values, 0.0), 1.0);
        assert_eq!(nearest_rank_percentile(&values, 0.5), 2.0);
        assert_eq!(nearest_rank_percentile(&values, 0.95), 4.0);
        assert_eq!(nearest_rank_percentile(&values, 1.0), 4.0);
        assert_eq!(nearest_rank_percentile(&[], 0.5), 0.0);
        assert_eq!(nearest_rank_percentile(&values, f64::NAN), 0.0);
        assert_eq!(nearest_rank_percentile(&values, -0.1), 0.0);
        assert_eq!(nearest_rank_percentile(&values, 1.1), 0.0);
    }

    #[test]
    fn argmax_margin_reports_ties_and_rejects_invalid_extents() {
        assert_eq!(top1_margin_for_argmax(&[1.0, 3.0, 2.5], 1), 0.5);
        assert_eq!(top1_margin_for_argmax(&[3.0, 3.0, 1.0], 0), 0.0);
        assert_eq!(top1_margin_for_argmax(&[1.0], 0), 0.0);
        assert_eq!(top1_margin_for_argmax(&[1.0, 2.0], 2), 0.0);
    }

    #[test]
    fn autoregressive_margin_distribution_counts_near_ties() {
        let token = |generation_index: usize, margin: f64| AutoregressiveTokenReport {
            generation_index,
            position: generation_index,
            input_token_id: generation_index,
            output_token_id: generation_index + 1,
            host_wall_ms: 1.0,
            cuda_event_ms: 0.5,
            top1_margin: margin,
            logits_sha256: format!("logits-{generation_index}"),
        };
        let report = autoregressive_margin_distribution(&[
            token(0, 0.005),
            token(1, 0.02),
            token(2, 0.075),
            token(3, 1.0),
        ]);
        assert_eq!(report.minimum, 0.005);
        assert_eq!(report.p50, 0.02);
        assert_eq!(report.p95, 1.0);
        assert_eq!(report.below_0_01_count, 1);
        assert_eq!(report.below_0_05_count, 2);
        assert_eq!(report.below_0_10_count, 3);
    }

    #[test]
    fn zero_copy_layer_handoff_contract_rejects_any_copy_residue() {
        assert!(zero_copy_layer_handoff_counts_closed(
            false,
            (63, 1_290_240),
            (63, 1_290_240),
            Some((63, 1_290_240)),
        ));
        assert!(zero_copy_layer_handoff_counts_closed(
            true,
            (0, 0),
            (0, 0),
            Some((0, 0)),
        ));
        assert!(!zero_copy_layer_handoff_counts_closed(
            true,
            (1, 20_480),
            (0, 0),
            None,
        ));
        assert!(!zero_copy_layer_handoff_counts_closed(
            true,
            (0, 0),
            (0, 4),
            None,
        ));
        assert!(!zero_copy_layer_handoff_counts_closed(
            true,
            (0, 0),
            (0, 0),
            Some((1, 20_480)),
        ));
    }

    #[cfg(all(
        feature = "qwen35-batched-prefill-handoff",
        feature = "qwen35-kv8-full-attention"
    ))]
    #[test]
    fn kv8_ram_persistence_identity_binds_generation_and_extent() {
        let geometry = Kv8RamPersistenceGeometry {
            capacity_positions: 4_096,
            populated_positions: 3_583,
            plane_bytes: 4_456_448,
            page_bytes: 557_056,
            pages_per_plane: 8,
            plane_count: 32,
            page_count: 256,
            total_payload_bytes: 142_606_336,
        };
        let (session_a, identity_a) = kv8_persistence_identity([0x5a; 32], 1, geometry);
        let (session_b, identity_b) = kv8_persistence_identity([0x5a; 32], 2, geometry);
        let (_, identity_c) = kv8_persistence_identity(
            [0x5a; 32],
            1,
            Kv8RamPersistenceGeometry {
                populated_positions: 3_582,
                ..geometry
            },
        );
        assert_eq!(session_a.len(), "qwen35-kv8-".len() + 16);
        assert_ne!(session_a, session_b);
        assert_ne!(identity_a, identity_b);
        assert_ne!(identity_a, identity_c);
        assert_eq!(geometry.snapshot_range(255).unwrap().end, 142_606_336);
        assert!(geometry.snapshot_range(256).is_err());
    }

    #[cfg(all(
        feature = "qwen35-batched-prefill-handoff",
        feature = "qwen35-kv8-full-attention"
    ))]
    #[test]
    fn kv8_ram_page_ledger_binds_payload_and_coordinates() {
        let geometry = Kv8RamPersistenceGeometry {
            capacity_positions: 2,
            populated_positions: 1,
            plane_bytes: 8,
            page_bytes: 4,
            pages_per_plane: 2,
            plane_count: 8,
            page_count: 16,
            total_payload_bytes: 64,
        };
        let layers = [3, 7, 11, 15];
        let mut snapshot = (0_u8..64).collect::<Vec<_>>();
        let first = kv8_page_ledger_sha256("identity", &layers, geometry, &snapshot).unwrap();
        snapshot[17] ^= 0x80;
        let mutated = kv8_page_ledger_sha256("identity", &layers, geometry, &snapshot).unwrap();
        let rebound = kv8_page_ledger_sha256("other", &layers, geometry, &snapshot).unwrap();
        assert_ne!(first, mutated);
        assert_ne!(mutated, rebound);
    }
}
