#![cfg(feature = "qwen35-linear-layer")]

//! Real-weight linear-attention CUDA composition for the Qwen35 lane.
//!
//! The probe joins the already audited Gated DeltaNet/output projection with
//! the post-attention residual, RMSNorm, mixed-quantized SwiGLU FFN, and final
//! residual on one CUDA stream.  Input projections are intentionally CPU-derived for
//! this first composition boundary.  The result is a numerical component
//! receipt, not a complete transformer or a token-generation path.

use std::{error::Error, fmt, hint::spin_loop, path::PathBuf, time::Instant};

#[cfg(not(feature = "qwen35-split-rmsnorm"))]
use augur_backend::CudaRmsnormF32ArgsV1 as CudaPlanRmsnormF32;
#[cfg(feature = "qwen35-split-rmsnorm")]
use augur_backend::CudaSplitRmsnormF32V1 as CudaPlanRmsnormF32;
use augur_backend::{
    cuda_device_allocation_snapshot, cuda_device_properties, cuda_host_boundary_snapshot,
    CudaAddF32ArgsV1, CudaF32View, CudaNvfp4Sm70F32MatvecArgsV1, CudaNvfp4Sm70Q8_1MatvecArgsV1,
    CudaNvfp4Sm70WeightViewV1, CudaQ8_0F32MatvecArgsV1, CudaQ8_0Q8_1MatvecArgsV1,
    CudaQ8_0WeightViewV1, CudaQ8_1ActivationQuantizeArgsV1, CudaQ8_1ActivationViewV1,
    CudaQwen35LinearStepArgsV1, CudaRuntimeError, CudaSiluMulF32ArgsV1, OwnedCudaBuffer,
    OwnedCudaEvent, OwnedCudaStream, Q8_1_ACTIVATION_BLOCK_BYTES, Q8_1_ACTIVATION_BLOCK_ELEMENTS,
};
#[cfg(feature = "qwen35-fused-ffn-swiglu")]
use augur_backend::{CudaNvfp4Sm70Q8_1FusedSwigluArgsV1, CudaQ8_0Q8_1FusedSwigluArgsV1};
use augur_core::PhysicalLayoutId;
use augur_model_qwen35::{
    dequantize_bf16, dequantize_tensor, l2_norm_last_dim, quantized_matvec_fused_simd,
    qwen35_layer_forward, rms_norm_last_dim, silu, CausalConv1dState, GatedDeltaNetState,
    GgufAuditOptions, GgufManifest, GgufProjection, GgufTensor, Qwen35Config, Qwen35Error,
    Qwen35FfnWeights, Qwen35LayerSpec, Qwen35LayerState, Qwen35LayerWeights,
    Qwen35LinearLayerWeights, ReferenceTensor, GGML_TYPE_BF16, GGML_TYPE_F32, GGML_TYPE_NVFP4,
    GGML_TYPE_Q8_0, NVFP4_BLOCK_BYTES, NVFP4_BLOCK_ELEMENTS, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMENTS,
};
use serde::Serialize;
use sha2::{Digest, Sha256};

pub const QWEN35_REAL_LINEAR_LAYER_CUDA_SCHEMA: &str =
    "crystal-quant.qwen35.real-linear-layer-cuda.v1";

pub(crate) const HIDDEN: usize = 5_120;
pub(crate) const VOCAB: usize = 248_320;
pub(crate) const KEY_HEADS: usize = 16;
pub(crate) const VALUE_HEADS: usize = 48;
pub(crate) const HEAD_DIM: usize = 128;
pub(crate) const KERNEL_SIZE: usize = 4;
pub(crate) const QKV_ELEMENTS: usize = 10_240;
pub(crate) const RECURRENT_ELEMENTS: usize = 6_144;
pub(crate) const CONV_CHANNELS: usize = QKV_ELEMENTS;
pub(crate) const CONV_HISTORY_ELEMENTS: usize = CONV_CHANNELS * (KERNEL_SIZE - 1);
pub(crate) const CONV_WEIGHT_ELEMENTS: usize = CONV_CHANNELS * KERNEL_SIZE;
pub(crate) const STATE_ELEMENTS: usize = VALUE_HEADS * HEAD_DIM * HEAD_DIM;
pub(crate) const INTERMEDIATE: usize = 17_408;
const STRICT_PROFILE_TENSOR_COUNT: u64 = 1_187;
const MAX_COMPLETION_POLLS: u64 = 2_000_000;

pub(crate) const EMBEDDING: &str = "token_embd.weight";
pub(crate) const ATTN_NORM: &str = "attn_norm.weight";
pub(crate) const QKV: &str = "attn_qkv.weight";
pub(crate) const GATE: &str = "attn_gate.weight";
pub(crate) const ALPHA: &str = "ssm_alpha.weight";
pub(crate) const BETA: &str = "ssm_beta.weight";
pub(crate) const SSM_OUT: &str = "ssm_out.weight";
pub(crate) const CONV: &str = "ssm_conv1d.weight";
pub(crate) const SSM_A: &str = "ssm_a";
pub(crate) const DT_BIAS: &str = "ssm_dt.bias";
pub(crate) const SSM_NORM: &str = "ssm_norm.weight";
pub(crate) const POST_NORM: &str = "post_attention_norm.weight";
pub(crate) const FFN_GATE: &str = "ffn_gate.weight";
pub(crate) const FFN_UP: &str = "ffn_up.weight";
pub(crate) const FFN_DOWN: &str = "ffn_down.weight";
pub(crate) const FFN_GATE_SCALE: &str = "ffn_gate.scale";
pub(crate) const FFN_UP_SCALE: &str = "ffn_up.scale";
pub(crate) const FFN_DOWN_SCALE: &str = "ffn_down.scale";
pub(crate) const FFN_GATE_INPUT_SCALE: &str = "ffn_gate.input_scale";
pub(crate) const FFN_UP_INPUT_SCALE: &str = "ffn_up.input_scale";
pub(crate) const FFN_DOWN_INPUT_SCALE: &str = "ffn_down.input_scale";

pub(crate) const I_EMBEDDING: usize = 0;
pub(crate) const I_ATTN_NORM: usize = 1;
pub(crate) const I_QKV: usize = 2;
pub(crate) const I_GATE: usize = 3;
pub(crate) const I_ALPHA: usize = 4;
pub(crate) const I_BETA: usize = 5;
pub(crate) const I_SSM_OUT: usize = 6;
pub(crate) const I_CONV: usize = 7;
pub(crate) const I_SSM_A: usize = 8;
pub(crate) const I_DT_BIAS: usize = 9;
pub(crate) const I_SSM_NORM: usize = 10;
pub(crate) const I_POST_NORM: usize = 11;
pub(crate) const I_FFN_GATE: usize = 12;
pub(crate) const I_FFN_UP: usize = 13;
pub(crate) const I_FFN_DOWN: usize = 14;
pub(crate) const I_FFN_GATE_SCALE: usize = 15;
pub(crate) const I_FFN_UP_SCALE: usize = 16;
pub(crate) const I_FFN_DOWN_SCALE: usize = 17;
pub(crate) const I_FFN_GATE_INPUT_SCALE: usize = 18;
pub(crate) const I_FFN_UP_INPUT_SCALE: usize = 19;
pub(crate) const I_FFN_DOWN_INPUT_SCALE: usize = 20;

/// Configuration for the opt-in linear-attention composition probe.
#[derive(Clone, Debug)]
pub struct Qwen35RealLinearLayerCudaConfig {
    pub gguf_path: PathBuf,
    pub config_path: PathBuf,
    /// Zero-based layer index. The selected layer must be a linear-attention
    /// layer whose FFN weights use the audited NVFP4 profile.
    pub layer: usize,
    pub token_id: usize,
    pub device_ordinal: i32,
    pub warmup: usize,
    pub iterations: usize,
    pub source_commit: String,
}

impl Default for Qwen35RealLinearLayerCudaConfig {
    fn default() -> Self {
        Self {
            gguf_path: PathBuf::new(),
            config_path: PathBuf::new(),
            layer: 0,
            token_id: 1,
            device_ordinal: 0,
            warmup: 2,
            iterations: 20,
            source_commit: "UNBOUND_COMPONENT_SOURCE".to_owned(),
        }
    }
}

#[derive(Debug)]
pub enum Qwen35RealLinearLayerCudaError {
    Runtime(CudaRuntimeError),
    Model(Qwen35Error),
    InvalidConfiguration(&'static str),
    CompletionPollLimit,
}

impl fmt::Display for Qwen35RealLinearLayerCudaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Runtime(error) => error.fmt(formatter),
            Self::Model(error) => error.fmt(formatter),
            Self::InvalidConfiguration(field) => {
                write!(
                    formatter,
                    "invalid Qwen linear-layer configuration: {field}"
                )
            }
            Self::CompletionPollLimit => {
                formatter.write_str("CUDA completion polling exceeded its bound")
            }
        }
    }
}

impl Error for Qwen35RealLinearLayerCudaError {}

impl From<CudaRuntimeError> for Qwen35RealLinearLayerCudaError {
    fn from(error: CudaRuntimeError) -> Self {
        Self::Runtime(error)
    }
}

impl From<Qwen35Error> for Qwen35RealLinearLayerCudaError {
    fn from(error: Qwen35Error) -> Self {
        Self::Model(error)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Qwen35RealLinearLayerCudaReport {
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
    pub artifact: ArtifactReport,
    pub token_ids: Vec<usize>,
    pub geometry: GeometryReport,
    pub tensors: Vec<TensorReport>,
    pub timing: TimingReport,
    pub completion: CompletionReport,
    pub numerical: Vec<NumericalReport>,
    pub resources: ResourceReport,
    pub checks: CheckReport,
    pub claims: ClaimsReport,
}

#[derive(Clone, Debug, Serialize)]
pub struct DeviceReport {
    pub ordinal: i32,
    pub name: String,
    pub compute_capability: [u16; 2],
    pub total_global_memory_bytes: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ArtifactReport {
    pub manifest_tensor_count: u64,
    pub audited_tensor_count: usize,
    pub strict_profile_expected_tensor_count: u64,
    pub strict_profile_count_match: bool,
    pub strict_profile_qualified: bool,
    pub classification: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct TensorReport {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub ggml_type: u32,
    pub descriptor_payload_bytes: u64,
    pub payload_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct GeometryReport {
    pub layer: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub head_dim: usize,
    pub qkv_elements: usize,
    pub recurrent_elements: usize,
    pub state_elements: usize,
    pub ffn_stages_per_pass: usize,
    pub ordered_operations_per_pass: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct TimingReport {
    pub warmup_passes: usize,
    pub steady_passes: usize,
    pub ordered_operations_per_pass: usize,
    pub steady_cuda_event_ms: f32,
    pub steady_batch_wall_ms: f64,
    pub per_pass_cuda_event_ms: f64,
    pub reference_ms: f64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct CompletionReport {
    pub poll_iterations: u64,
    pub nonblocking_poll_only: bool,
    pub owner_epoch_proof: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct NumericalReport {
    pub name: String,
    pub elements: usize,
    pub finite_count: usize,
    pub max_abs_diff: f32,
    pub max_rel_diff: f32,
    pub absolute_tolerance: f32,
    pub relative_tolerance: f32,
    pub output_sha256: String,
    pub expected_sha256: String,
    pub pass: bool,
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
    pub setup_allocations: AllocationReport,
    pub steady_boundaries: BoundaryReport,
    pub steady_allocations: AllocationReport,
    pub sequence_input_boundaries: BoundaryReport,
    pub completion_boundaries: BoundaryReport,
    pub readback_boundaries: BoundaryReport,
    pub cleanup_boundaries: BoundaryReport,
    pub total_boundaries: BoundaryReport,
    pub boundary_ledger_closed: bool,
    pub final_allocation_delta: AllocationReport,
}

#[derive(Clone, Debug, Serialize)]
pub struct CheckReport {
    pub geometry_valid: bool,
    pub finite_outputs: bool,
    pub linear_state_changed: bool,
    pub attention_residual_shape_valid: bool,
    pub post_norm_shape_valid: bool,
    pub ffn_shape_valid: bool,
    pub final_hidden_shape_valid: bool,
    pub checkpoint_replay_exact: bool,
    pub completion_proof_valid: bool,
    pub steady_no_host_boundary: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct ClaimsReport {
    pub real_weight_linear_attention_and_ffn_chain_executed: bool,
    pub input_projections_cpu_derived: bool,
    pub complete_transformer_executed: bool,
    pub decode_executed: bool,
    pub semantic_parity_established: bool,
    pub product_tps_permitted: bool,
    pub default_cli_enabled: bool,
}

#[derive(Clone)]
pub(crate) struct ProjectedToken {
    pub(crate) qkv: Vec<f32>,
    pub(crate) z: Vec<f32>,
    pub(crate) decay_gate: Vec<f32>,
    pub(crate) beta: Vec<f32>,
}

/// CPU oracle for the raw DeltaNet query projection exposed by the CUDA step.
///
/// The public Qwen reference block returns the value after recurrent RMSNorm
/// and the elementwise SiLU gate.  The CUDA ABI intentionally exposes both
/// that final value (`output_f32`) and the pre-gate projection (`recurrent_f32`)
/// for auditing, so the latter needs a matching stateful oracle.
struct RawLinearAttentionOracle {
    conv: CausalConv1dState,
    gated_delta: GatedDeltaNetState,
}

impl RawLinearAttentionOracle {
    fn zeros() -> Result<Self, Qwen35Error> {
        Ok(Self {
            conv: CausalConv1dState::zeros(KERNEL_SIZE, QKV_ELEMENTS)?,
            gated_delta: GatedDeltaNetState::zeros(VALUE_HEADS, HEAD_DIM)?,
        })
    }

    fn step(
        &mut self,
        qkv_mixed: &[f32],
        conv_weights: &[f32],
        decay_gate: &[f32],
        beta: &[f32],
        epsilon: f32,
    ) -> Result<Vec<f32>, Qwen35Error> {
        let conv_output = self.conv.step(qkv_mixed, conv_weights)?;
        let activated = silu(&ReferenceTensor::new(vec![QKV_ELEMENTS], conv_output)?)?;
        let key_values = KEY_HEADS * HEAD_DIM;
        let q = l2_norm_last_dim(
            &ReferenceTensor::new(
                vec![KEY_HEADS, HEAD_DIM],
                activated.values()[..key_values].to_vec(),
            )?,
            epsilon,
        )?;
        let k = l2_norm_last_dim(
            &ReferenceTensor::new(
                vec![KEY_HEADS, HEAD_DIM],
                activated.values()[key_values..key_values * 2].to_vec(),
            )?,
            epsilon,
        )?;
        self.gated_delta.step(
            q.values(),
            k.values(),
            &activated.values()[key_values * 2..],
            decay_gate,
            beta,
        )
    }

    fn state_checkpoint(&self) -> Vec<f32> {
        self.gated_delta.checkpoint()
    }
}

pub(crate) struct DeviceReadback {
    pub(crate) attention: Vec<f32>,
    pub(crate) attention_residual: Vec<f32>,
    pub(crate) post_norm: Vec<f32>,
    pub(crate) ffn_output: Vec<f32>,
    pub(crate) final_hidden: Vec<f32>,
    pub(crate) recurrent: Vec<f32>,
    pub(crate) state: Vec<f32>,
    pub(crate) history: Vec<f32>,
    pub(crate) polls: u64,
    pub(crate) proof: bool,
    pub(crate) completion_boundaries: BoundaryReport,
    pub(crate) readback_boundaries: BoundaryReport,
}

pub(crate) struct BoundChain {
    recurrent: CudaQwen35LinearStepArgsV1,
    output_projection_quantize: Option<CudaQ8_1ActivationQuantizeArgsV1>,
    output_projection: Q8ProjectionOp,
    attention_residual: CudaAddF32ArgsV1,
    post_norm: CudaPlanRmsnormF32,
    pub(crate) ffn: BoundFfn,
    final_residual: CudaAddF32ArgsV1,
}

/// One whole FFN profile, never a mix of differently represented projections.
pub(crate) enum BoundFfn {
    Legacy(LegacyFfn),
    #[cfg(feature = "qwen35-prefill-cublas")]
    Qpn(augur_backend::cuda_qwen35_prefill_cublas::CudaQwen35QpnFfnProgramV1),
}
pub(crate) struct LegacyFfn {
    pub input_quantize: Option<CudaQ8_1ActivationQuantizeArgsV1>,
    pub gate: FfnMatvecOp,
    pub up: FfnMatvecOp,
    pub silu_mul: CudaSiluMulF32ArgsV1,
    #[cfg(feature = "qwen35-fused-ffn-swiglu")]
    pub fused_gate_up: Option<FfnFusedGateUpOp>,
    pub down_quantize: Option<CudaQ8_1ActivationQuantizeArgsV1>,
    pub down: FfnMatvecOp,
}

#[derive(Clone, Copy, Debug)]
enum Q8ProjectionOp {
    F32(CudaQ8_0F32MatvecArgsV1),
    Q8_1(CudaQ8_0Q8_1MatvecArgsV1),
}

impl Q8ProjectionOp {
    fn submit(&self, stream: &OwnedCudaStream) -> Result<(), Qwen35RealLinearLayerCudaError> {
        match self {
            Self::F32(op) => op.submit_for_plan(stream)?,
            Self::Q8_1(op) => op.submit_for_plan(stream)?,
        }
        Ok(())
    }
}

/// One checked FFN matvec operation.  Qwen's audited artifact uses NVFP4 for
/// layers 0..55 and Q8_0 for layers 56..63; keeping the variants explicit
/// prevents silently treating Q8 payloads as NVFP4 (or inventing scales).
#[derive(Clone, Copy, Debug)]
pub(crate) enum FfnMatvecOp {
    Nvfp4F32(CudaNvfp4Sm70F32MatvecArgsV1),
    Q8_0F32(CudaQ8_0F32MatvecArgsV1),
    Nvfp4Q8_1(CudaNvfp4Sm70Q8_1MatvecArgsV1),
    Q8_0Q8_1(CudaQ8_0Q8_1MatvecArgsV1),
}

#[cfg(feature = "qwen35-fused-ffn-swiglu")]
#[derive(Clone, Copy, Debug)]
pub(crate) enum FfnFusedGateUpOp {
    Nvfp4(CudaNvfp4Sm70Q8_1FusedSwigluArgsV1),
    Q8_0(CudaQ8_0Q8_1FusedSwigluArgsV1),
}

#[cfg(feature = "qwen35-fused-ffn-swiglu")]
impl FfnFusedGateUpOp {
    pub(crate) fn submit(
        &self,
        stream: &OwnedCudaStream,
    ) -> Result<(), Qwen35RealLinearLayerCudaError> {
        match self {
            Self::Nvfp4(op) => op.submit_for_plan(stream)?,
            Self::Q8_0(op) => op.submit_for_plan(stream)?,
        }
        Ok(())
    }
}

impl FfnMatvecOp {
    pub(crate) fn submit(
        &self,
        stream: &OwnedCudaStream,
    ) -> Result<(), Qwen35RealLinearLayerCudaError> {
        match self {
            Self::Nvfp4F32(op) => op.submit_for_plan(stream)?,
            Self::Q8_0F32(op) => op.submit_for_plan(stream)?,
            Self::Nvfp4Q8_1(op) => op.submit_for_plan(stream)?,
            Self::Q8_0Q8_1(op) => op.submit_for_plan(stream)?,
        }
        Ok(())
    }
}

impl BoundChain {
    pub(crate) fn submit(
        &self,
        stream: &OwnedCudaStream,
    ) -> Result<(), Qwen35RealLinearLayerCudaError> {
        self.recurrent.submit(stream)?;
        if let Some(op) = &self.output_projection_quantize {
            op.submit_for_plan(stream)?;
        }
        self.output_projection.submit(stream)?;
        self.attention_residual.submit_for_plan(stream)?;
        self.post_norm.submit_for_plan(stream)?;
        self.ffn.submit(stream)?;
        self.final_residual.submit_for_plan(stream)?;
        Ok(())
    }
}
impl BoundFfn {
    pub(crate) fn submit(&self, stream: &OwnedCudaStream) -> Result<(), Qwen35RealLinearLayerCudaError> {
        match self {
            Self::Legacy(ffn) => ffn.submit(stream),
            #[cfg(feature = "qwen35-prefill-cublas")]
            Self::Qpn(ffn) => Ok(ffn.submit_for_plan_on(stream)?),
        }
    }
}
impl LegacyFfn {
    fn submit(&self, stream: &OwnedCudaStream) -> Result<(), Qwen35RealLinearLayerCudaError> {
        if let Some(op) = &self.input_quantize {
            op.submit_for_plan(stream)?;
        }
        #[cfg(feature = "qwen35-fused-ffn-swiglu")]
        match &self.fused_gate_up {
            Some(op) => op.submit(stream)?,
            None => {
                self.gate.submit(stream)?;
                self.up.submit(stream)?;
                self.silu_mul.submit_for_plan(stream)?;
            }
        }
        #[cfg(not(feature = "qwen35-fused-ffn-swiglu"))]
        {
            self.gate.submit(stream)?;
            self.up.submit(stream)?;
            self.silu_mul.submit_for_plan(stream)?;
        }
        if let Some(op) = &self.down_quantize {
            op.submit_for_plan(stream)?;
        }
        self.down.submit(stream)?;
        Ok(())
    }
}

/// Execute the authenticated selected linear-attention plus FFN composition.
pub fn run_qwen35_real_linear_layer_cuda(
    config: &Qwen35RealLinearLayerCudaConfig,
) -> Result<Qwen35RealLinearLayerCudaReport, Qwen35RealLinearLayerCudaError> {
    validate_config(config)?;
    let binary_path = std::env::current_exe().map_err(Qwen35Error::from)?;
    let binary_sha256 = sha256_hex(&std::fs::read(&binary_path).map_err(Qwen35Error::from)?);
    let config_bytes = std::fs::read(&config.config_path).map_err(Qwen35Error::from)?;
    let config_sha256 = sha256_hex(&config_bytes);
    let model_config = Qwen35Config::parse(&config_bytes)?;
    if model_config.hidden_size != HIDDEN
        || model_config.vocab_size != VOCAB
        || model_config.num_hidden_layers != 64
        || !model_config.linear_attention_layers.contains(&config.layer)
        || model_config.linear_num_key_heads != KEY_HEADS
        || model_config.linear_num_value_heads != VALUE_HEADS
        || model_config.linear_key_head_dim != HEAD_DIM
        || model_config.linear_value_head_dim != HEAD_DIM
        || model_config.linear_conv_kernel_dim != KERNEL_SIZE
        || model_config.intermediate_size != INTERMEDIATE
    {
        return Err(Qwen35RealLinearLayerCudaError::InvalidConfiguration(
            "Qwen linear-attention layer geometry",
        ));
    }

    let manifest = GgufManifest::inspect(&config.gguf_path, GgufAuditOptions::default())?;
    let ffn_type = augur_model_qwen35::Qwen35TensorSchema::expected_ffn_weight_ggml_type(
        &model_config,
        config.layer,
    )?;
    if ffn_type != GGML_TYPE_NVFP4 {
        return Err(Qwen35RealLinearLayerCudaError::InvalidConfiguration(
            "standalone linear-layer runner requires an NVFP4 FFN layer",
        ));
    }
    let descriptors = required_descriptors(&manifest, config.layer, ffn_type)?;
    let strict_profile_count_match = manifest.tensor_count == STRICT_PROFILE_TENSOR_COUNT;
    let artifact_classification = if strict_profile_count_match {
        "count_matches_strict_profile_but_full_model_qualification_not_run"
    } else {
        "extended_or_combined_manifest"
    };
    let mut reader = manifest.open_tensor_reader(&config.gguf_path)?;
    let second_token_id = config.token_id.checked_add(1).ok_or(
        Qwen35RealLinearLayerCudaError::InvalidConfiguration("token id"),
    )?;
    let hidden_a = read_embedding_row(&mut reader, &descriptors[I_EMBEDDING], config.token_id)?;
    let hidden_b = read_embedding_row(&mut reader, &descriptors[I_EMBEDDING], second_token_id)?;
    let mut payloads = Vec::with_capacity(descriptors.len());
    for descriptor in &descriptors {
        payloads.push(reader.read_tensor_bytes(&descriptor.name)?);
    }
    let tensors = descriptors
        .iter()
        .zip(payloads.iter())
        .map(|(descriptor, payload)| TensorReport {
            name: descriptor.name.clone(),
            dimensions: descriptor.dimensions.clone(),
            ggml_type: descriptor.ggml_type,
            descriptor_payload_bytes: descriptor.byte_len,
            payload_sha256: sha256_hex(payload),
        })
        .collect::<Vec<_>>();

    let attn_norm = dequantize_tensor(&descriptors[I_ATTN_NORM], &payloads[I_ATTN_NORM])?;
    let conv_raw = dequantize_tensor(&descriptors[I_CONV], &payloads[I_CONV])?;
    let conv_weights = convert_conv_layout(&conv_raw)?;
    let ssm_a = dequantize_tensor(&descriptors[I_SSM_A], &payloads[I_SSM_A])?;
    let dt_bias = dequantize_tensor(&descriptors[I_DT_BIAS], &payloads[I_DT_BIAS])?;
    let ssm_norm = dequantize_tensor(&descriptors[I_SSM_NORM], &payloads[I_SSM_NORM])?;
    let post_norm = dequantize_tensor(&descriptors[I_POST_NORM], &payloads[I_POST_NORM])?;
    let gate_scale = positive_scale(
        &dequantize_tensor(&descriptors[I_FFN_GATE_SCALE], &payloads[I_FFN_GATE_SCALE])?,
        "ffn_gate.scale",
    )?;
    let up_scale = positive_scale(
        &dequantize_tensor(&descriptors[I_FFN_UP_SCALE], &payloads[I_FFN_UP_SCALE])?,
        "ffn_up.scale",
    )?;
    let down_scale = positive_scale(
        &dequantize_tensor(&descriptors[I_FFN_DOWN_SCALE], &payloads[I_FFN_DOWN_SCALE])?,
        "ffn_down.scale",
    )?;
    let _gate_input_scale = positive_scale(
        &dequantize_tensor(
            &descriptors[I_FFN_GATE_INPUT_SCALE],
            &payloads[I_FFN_GATE_INPUT_SCALE],
        )?,
        "ffn_gate.input_scale",
    )?;
    let _up_input_scale = positive_scale(
        &dequantize_tensor(
            &descriptors[I_FFN_UP_INPUT_SCALE],
            &payloads[I_FFN_UP_INPUT_SCALE],
        )?,
        "ffn_up.input_scale",
    )?;
    let _down_input_scale = positive_scale(
        &dequantize_tensor(
            &descriptors[I_FFN_DOWN_INPUT_SCALE],
            &payloads[I_FFN_DOWN_INPUT_SCALE],
        )?,
        "ffn_down.input_scale",
    )?;
    let epsilon = model_config.rms_norm_eps as f32;

    // Build the independent CPU oracle through the same checked layer contract
    // used by the existing hybrid smoke.  The CUDA path below receives only
    // the pre-attention projection vectors from this oracle.
    let reference_started = Instant::now();
    let qkv = GgufProjection::new(&descriptors[I_QKV], &payloads[I_QKV])?;
    let z = GgufProjection::new(&descriptors[I_GATE], &payloads[I_GATE])?;
    let alpha = GgufProjection::new(&descriptors[I_ALPHA], &payloads[I_ALPHA])?;
    let beta = GgufProjection::new(&descriptors[I_BETA], &payloads[I_BETA])?;
    let ssm_out = GgufProjection::new(&descriptors[I_SSM_OUT], &payloads[I_SSM_OUT])?;
    let ffn_gate = GgufProjection::with_multipliers(
        &descriptors[I_FFN_GATE],
        &payloads[I_FFN_GATE],
        1.0,
        gate_scale,
    )?;
    let ffn_up = GgufProjection::with_multipliers(
        &descriptors[I_FFN_UP],
        &payloads[I_FFN_UP],
        1.0,
        up_scale,
    )?;
    let ffn_down = GgufProjection::with_multipliers(
        &descriptors[I_FFN_DOWN],
        &payloads[I_FFN_DOWN],
        1.0,
        down_scale,
    )?;
    let spec = Qwen35LayerSpec::from_config(&model_config, config.layer)?;
    let layer_weights = Qwen35LayerWeights::Linear(Qwen35LinearLayerWeights {
        attn_norm: &attn_norm,
        qkv: &qkv,
        z: &z,
        alpha: &alpha,
        beta: &beta,
        conv_weights: &conv_weights,
        dt_bias: &dt_bias,
        ssm_a: &ssm_a,
        ssm_norm: &ssm_norm,
        ssm_out: &ssm_out,
        post_attention_norm: &post_norm,
        ffn: Qwen35FfnWeights {
            gate: &ffn_gate,
            up: &ffn_up,
            down: &ffn_down,
            gate_multiplier: 1.0,
            up_multiplier: 1.0,
            down_multiplier: 1.0,
        },
    });
    let mut cpu_state = Qwen35LayerState::for_spec(&spec, 2)?;
    let initial_layer_checkpoint = cpu_state.checkpoint();
    let expected_a = qwen35_layer_forward(&spec, &layer_weights, &mut cpu_state, &hidden_a, 0)?;
    let layer_checkpoint_a = cpu_state.checkpoint();
    let expected_b = qwen35_layer_forward(&spec, &layer_weights, &mut cpu_state, &hidden_b, 1)?;
    let _layer_checkpoint_b = cpu_state.checkpoint();
    let reference_ms = reference_started.elapsed().as_secs_f64() * 1_000.0;

    let token_a = project_token(
        &hidden_a,
        &attn_norm,
        &descriptors[I_QKV],
        &payloads[I_QKV],
        &descriptors[I_GATE],
        &payloads[I_GATE],
        &descriptors[I_ALPHA],
        &payloads[I_ALPHA],
        &descriptors[I_BETA],
        &payloads[I_BETA],
        &dt_bias,
        &ssm_a,
        epsilon,
    )?;
    let token_b = project_token(
        &hidden_b,
        &attn_norm,
        &descriptors[I_QKV],
        &payloads[I_QKV],
        &descriptors[I_GATE],
        &payloads[I_GATE],
        &descriptors[I_ALPHA],
        &payloads[I_ALPHA],
        &descriptors[I_BETA],
        &payloads[I_BETA],
        &dt_bias,
        &ssm_a,
        epsilon,
    )?;

    // Keep a separate raw recurrent oracle.  The CUDA step exposes the
    // pre-RMSNorm/query projection in `recurrent_f32`; the full layer oracle
    // above remains responsible for the gated recurrent output, residual, and
    // FFN expectations.
    let mut raw_recurrent_oracle = RawLinearAttentionOracle::zeros()?;
    let raw_recurrent_a = raw_recurrent_oracle.step(
        &token_a.qkv,
        &conv_weights,
        &token_a.decay_gate,
        &token_a.beta,
        epsilon,
    )?;
    let raw_state_a = raw_recurrent_oracle.state_checkpoint();
    let raw_recurrent_b = raw_recurrent_oracle.step(
        &token_b.qkv,
        &conv_weights,
        &token_b.decay_gate,
        &token_b.beta,
        epsilon,
    )?;
    let raw_state_b = raw_recurrent_oracle.state_checkpoint();

    let device = cuda_device_properties(config.device_ordinal)?;
    let stream = OwnedCudaStream::create(config.device_ordinal)?;
    let allocation_start = cuda_device_allocation_snapshot();
    let boundary_start = cuda_host_boundary_snapshot();

    let qkv_buffer = OwnedCudaBuffer::allocate(config.device_ordinal, QKV_ELEMENTS * 4)?;
    let z_buffer = OwnedCudaBuffer::allocate(config.device_ordinal, RECURRENT_ELEMENTS * 4)?;
    let conv_buffer = OwnedCudaBuffer::allocate(config.device_ordinal, CONV_WEIGHT_ELEMENTS * 4)?;
    let mut history_buffer =
        OwnedCudaBuffer::allocate(config.device_ordinal, CONV_HISTORY_ELEMENTS * 4)?;
    let decay_buffer = OwnedCudaBuffer::allocate(config.device_ordinal, VALUE_HEADS * 4)?;
    let beta_buffer = OwnedCudaBuffer::allocate(config.device_ordinal, VALUE_HEADS * 4)?;
    let mut state_buffer = OwnedCudaBuffer::allocate(config.device_ordinal, STATE_ELEMENTS * 4)?;
    let ssm_norm_buffer = OwnedCudaBuffer::allocate(config.device_ordinal, HEAD_DIM * 4)?;
    let activated_buffer = OwnedCudaBuffer::allocate(config.device_ordinal, QKV_ELEMENTS * 4)?;
    let recurrent_buffer =
        OwnedCudaBuffer::allocate(config.device_ordinal, RECURRENT_ELEMENTS * 4)?;
    let linear_output_buffer =
        OwnedCudaBuffer::allocate(config.device_ordinal, RECURRENT_ELEMENTS * 4)?;
    let attention_buffer = OwnedCudaBuffer::allocate(config.device_ordinal, HIDDEN * 4)?;
    let input_buffer = OwnedCudaBuffer::allocate(config.device_ordinal, HIDDEN * 4)?;
    let attention_residual_buffer = OwnedCudaBuffer::allocate(config.device_ordinal, HIDDEN * 4)?;
    let post_norm_buffer = OwnedCudaBuffer::allocate(config.device_ordinal, HIDDEN * 4)?;
    let ffn_gate_output_buffer =
        OwnedCudaBuffer::allocate(config.device_ordinal, INTERMEDIATE * 4)?;
    let ffn_up_output_buffer = OwnedCudaBuffer::allocate(config.device_ordinal, INTERMEDIATE * 4)?;
    let ffn_activation_buffer = OwnedCudaBuffer::allocate(config.device_ordinal, INTERMEDIATE * 4)?;
    let ffn_output_buffer = OwnedCudaBuffer::allocate(config.device_ordinal, HIDDEN * 4)?;
    let final_hidden_buffer = OwnedCudaBuffer::allocate(config.device_ordinal, HIDDEN * 4)?;
    let output_weight_buffer =
        OwnedCudaBuffer::allocate(config.device_ordinal, payloads[I_SSM_OUT].len())?;
    let post_norm_weight_buffer = OwnedCudaBuffer::allocate(config.device_ordinal, HIDDEN * 4)?;
    let ffn_gate_weight_buffer =
        OwnedCudaBuffer::allocate(config.device_ordinal, payloads[I_FFN_GATE].len())?;
    let ffn_up_weight_buffer =
        OwnedCudaBuffer::allocate(config.device_ordinal, payloads[I_FFN_UP].len())?;
    let ffn_down_weight_buffer =
        OwnedCudaBuffer::allocate(config.device_ordinal, payloads[I_FFN_DOWN].len())?;

    upload_static(
        &stream,
        &qkv_buffer,
        &z_buffer,
        &conv_buffer,
        &decay_buffer,
        &beta_buffer,
        &ssm_norm_buffer,
        &output_weight_buffer,
        &post_norm_weight_buffer,
        &ffn_gate_weight_buffer,
        &ffn_up_weight_buffer,
        &ffn_down_weight_buffer,
        &token_a,
        &conv_weights,
        &ssm_norm,
        &post_norm,
        &payloads[I_SSM_OUT],
        &payloads[I_FFN_GATE],
        &payloads[I_FFN_UP],
        &payloads[I_FFN_DOWN],
    )?;
    // The warmup chain consumes the residual input just like a steady pass.
    // Initialize it before the first submission so warmup never reads an
    // undefined device allocation.
    input_buffer.copy_from_host_sync(&stream, &f32_bytes(&hidden_a))?;
    history_buffer.zero_sync(&stream)?;
    state_buffer.zero_sync(&stream)?;

    let chain = bind_chain(
        &stream,
        &qkv_buffer,
        &z_buffer,
        &conv_buffer,
        &history_buffer,
        &decay_buffer,
        &beta_buffer,
        &state_buffer,
        &ssm_norm_buffer,
        &activated_buffer,
        &recurrent_buffer,
        &linear_output_buffer,
        &output_weight_buffer,
        None,
        &attention_buffer,
        f32_view(&input_buffer, HIDDEN)?,
        &attention_residual_buffer,
        &post_norm_buffer,
        &post_norm_weight_buffer,
        &ffn_gate_weight_buffer,
        &ffn_up_weight_buffer,
        &ffn_gate_output_buffer,
        &ffn_up_output_buffer,
        &ffn_activation_buffer,
        &ffn_down_weight_buffer,
        &ffn_output_buffer,
        &final_hidden_buffer,
        None,
        None,
        payloads[I_SSM_OUT].len(),
        payloads[I_FFN_GATE].len(),
        payloads[I_FFN_UP].len(),
        payloads[I_FFN_DOWN].len(),
        GGML_TYPE_NVFP4,
        gate_scale,
        up_scale,
        down_scale,
        epsilon,
    )?;

    for _ in 0..config.warmup {
        chain.submit(&stream)?;
    }
    stream.synchronize()?;
    history_buffer.zero_sync(&stream)?;
    state_buffer.zero_sync(&stream)?;

    let setup_allocations = allocation_delta(allocation_start, cuda_device_allocation_snapshot())?;
    let setup_boundaries = boundary_delta(boundary_start, cuda_host_boundary_snapshot())?;

    let (steady_cuda_event_ms, steady_batch_wall_ms, steady_allocations, steady_boundaries) = {
        let steady_alloc_start = cuda_device_allocation_snapshot();
        let steady_boundary_start = cuda_host_boundary_snapshot();
        let started = OwnedCudaEvent::create(config.device_ordinal)?;
        let stopped = OwnedCudaEvent::create(config.device_ordinal)?;
        let wall_start = Instant::now();
        started.record(&stream)?;
        for _ in 0..config.iterations {
            chain.submit(&stream)?;
        }
        stopped.record(&stream)?;
        stopped.synchronize()?;
        let wall_ms = wall_start.elapsed().as_secs_f64() * 1_000.0;
        let event_ms = started.elapsed_ms(&stopped)?;
        let steady_allocations =
            allocation_delta(steady_alloc_start, cuda_device_allocation_snapshot())?;
        let steady_boundaries =
            boundary_delta(steady_boundary_start, cuda_host_boundary_snapshot())?;
        (event_ms, wall_ms, steady_allocations, steady_boundaries)
    };

    let mut sequence_input_boundaries = BoundaryReport::default();
    let before = cuda_host_boundary_snapshot();
    history_buffer.zero_sync(&stream)?;
    state_buffer.zero_sync(&stream)?;
    upload_token(
        &stream,
        &qkv_buffer,
        &z_buffer,
        &decay_buffer,
        &beta_buffer,
        &input_buffer,
        &token_a,
        &hidden_a,
    )?;
    sequence_input_boundaries = add_boundaries(
        sequence_input_boundaries,
        boundary_delta(before, cuda_host_boundary_snapshot())?,
    );
    let audited_a = execute_and_readback(
        &stream,
        &chain,
        &state_buffer,
        &history_buffer,
        &attention_buffer,
        &attention_residual_buffer,
        &post_norm_buffer,
        &ffn_output_buffer,
        &final_hidden_buffer,
        &recurrent_buffer,
    )?;

    let before = cuda_host_boundary_snapshot();
    upload_token(
        &stream,
        &qkv_buffer,
        &z_buffer,
        &decay_buffer,
        &beta_buffer,
        &input_buffer,
        &token_b,
        &hidden_b,
    )?;
    sequence_input_boundaries = add_boundaries(
        sequence_input_boundaries,
        boundary_delta(before, cuda_host_boundary_snapshot())?,
    );
    let audited_b = execute_and_readback(
        &stream,
        &chain,
        &state_buffer,
        &history_buffer,
        &attention_buffer,
        &attention_residual_buffer,
        &post_norm_buffer,
        &ffn_output_buffer,
        &final_hidden_buffer,
        &recurrent_buffer,
    )?;

    let before = cuda_host_boundary_snapshot();
    history_buffer.zero_sync(&stream)?;
    state_buffer.zero_sync(&stream)?;
    upload_token(
        &stream,
        &qkv_buffer,
        &z_buffer,
        &decay_buffer,
        &beta_buffer,
        &input_buffer,
        &token_a,
        &hidden_a,
    )?;
    sequence_input_boundaries = add_boundaries(
        sequence_input_boundaries,
        boundary_delta(before, cuda_host_boundary_snapshot())?,
    );
    let replay_a = execute_and_readback(
        &stream,
        &chain,
        &state_buffer,
        &history_buffer,
        &attention_buffer,
        &attention_residual_buffer,
        &post_norm_buffer,
        &ffn_output_buffer,
        &final_hidden_buffer,
        &recurrent_buffer,
    )?;

    let replay_exact = readback_equal(&audited_a, &replay_a);
    let completion = CompletionReport {
        poll_iterations: audited_a.polls + audited_b.polls + replay_a.polls,
        nonblocking_poll_only: audited_a.proof && audited_b.proof && replay_a.proof,
        owner_epoch_proof: audited_a.proof && audited_b.proof && replay_a.proof,
    };
    let completion_proof_valid = completion.owner_epoch_proof;
    let completion_boundaries = add_boundaries(
        add_boundaries(
            audited_a.completion_boundaries.clone(),
            audited_b.completion_boundaries.clone(),
        ),
        replay_a.completion_boundaries.clone(),
    );
    let readback_boundaries = add_boundaries(
        add_boundaries(
            audited_a.readback_boundaries.clone(),
            audited_b.readback_boundaries.clone(),
        ),
        replay_a.readback_boundaries.clone(),
    );

    let numerical = vec![
        compare(
            "token_a.attention",
            &audited_a.attention,
            &expected_a.attention_output,
            5e-3,
            2e-4,
        ),
        compare(
            "token_b.attention",
            &audited_b.attention,
            &expected_b.attention_output,
            5e-3,
            2e-4,
        ),
        compare(
            "token_a.attention_residual",
            &audited_a.attention_residual,
            &expected_a.attention_residual,
            1e-2,
            5e-4,
        ),
        compare(
            "token_b.attention_residual",
            &audited_b.attention_residual,
            &expected_b.attention_residual,
            1e-2,
            5e-4,
        ),
        compare(
            "token_a.post_attention_norm",
            &audited_a.post_norm,
            &expected_a.post_attention_norm,
            1e-2,
            5e-4,
        ),
        compare(
            "token_b.post_attention_norm",
            &audited_b.post_norm,
            &expected_b.post_attention_norm,
            1e-2,
            5e-4,
        ),
        compare(
            "token_a.ffn_output",
            &audited_a.ffn_output,
            &expected_a.ffn_output,
            1e-2,
            5e-4,
        ),
        compare(
            "token_b.ffn_output",
            &audited_b.ffn_output,
            &expected_b.ffn_output,
            1e-2,
            5e-4,
        ),
        compare(
            "token_a.final_hidden",
            &audited_a.final_hidden,
            &expected_a.hidden,
            1e-2,
            5e-4,
        ),
        compare(
            "token_b.final_hidden",
            &audited_b.final_hidden,
            &expected_b.hidden,
            1e-2,
            5e-4,
        ),
        compare(
            "token_a.recurrent",
            &audited_a.recurrent,
            &raw_recurrent_a,
            2e-3,
            4e-3,
        ),
        compare(
            "token_b.recurrent",
            &audited_b.recurrent,
            &raw_recurrent_b,
            2e-3,
            4e-3,
        ),
        compare("token_a.state", &audited_a.state, &raw_state_a, 2e-3, 4e-3),
        compare("token_b.state", &audited_b.state, &raw_state_b, 2e-3, 4e-3),
    ];

    let finite_outputs = [
        &audited_a.attention,
        &audited_a.attention_residual,
        &audited_a.post_norm,
        &audited_a.ffn_output,
        &audited_a.final_hidden,
        &audited_b.attention,
        &audited_b.attention_residual,
        &audited_b.post_norm,
        &audited_b.ffn_output,
        &audited_b.final_hidden,
    ]
    .into_iter()
    .all(|values| values.iter().all(|value| value.is_finite()));

    let cleanup_boundary_start = cuda_host_boundary_snapshot();
    stream.synchronize()?;
    let cleanup_boundaries = boundary_delta(cleanup_boundary_start, cuda_host_boundary_snapshot())?;
    let before_cleanup = cuda_device_allocation_snapshot();
    drop(chain);
    drop(final_hidden_buffer);
    drop(ffn_output_buffer);
    drop(ffn_down_weight_buffer);
    drop(ffn_activation_buffer);
    drop(ffn_up_output_buffer);
    drop(ffn_up_weight_buffer);
    drop(ffn_gate_output_buffer);
    drop(ffn_gate_weight_buffer);
    drop(post_norm_buffer);
    drop(attention_residual_buffer);
    drop(input_buffer);
    drop(post_norm_weight_buffer);
    drop(attention_buffer);
    drop(output_weight_buffer);
    drop(linear_output_buffer);
    drop(recurrent_buffer);
    drop(activated_buffer);
    drop(ssm_norm_buffer);
    drop(state_buffer);
    drop(beta_buffer);
    drop(decay_buffer);
    drop(history_buffer);
    drop(conv_buffer);
    drop(z_buffer);
    drop(qkv_buffer);
    drop(stream);
    let final_allocation_delta =
        allocation_delta(before_cleanup, cuda_device_allocation_snapshot())?;
    let total_boundaries = boundary_delta(boundary_start, cuda_host_boundary_snapshot())?;
    let accounted_boundaries = add_boundaries(
        add_boundaries(
            add_boundaries(setup_boundaries.clone(), steady_boundaries.clone()),
            sequence_input_boundaries.clone(),
        ),
        add_boundaries(
            add_boundaries(completion_boundaries.clone(), readback_boundaries.clone()),
            cleanup_boundaries.clone(),
        ),
    );
    let boundary_ledger_closed = accounted_boundaries == total_boundaries;
    let cleanup_zero =
        final_allocation_delta.live_allocations == 0 && final_allocation_delta.live_bytes == 0;
    let steady_no_host_boundary = steady_boundaries.host_waits == 0
        && steady_boundaries.stream_synchronizations == 0
        && steady_boundaries.synchronous_h2d_copies == 0
        && steady_boundaries.synchronous_d2h_copies == 0;
    let numerical_pass = numerical.iter().all(|item| item.pass);
    let pass = numerical_pass
        && replay_exact
        && completion.nonblocking_poll_only
        && completion.owner_epoch_proof
        && boundary_ledger_closed
        && cleanup_zero;

    Ok(Qwen35RealLinearLayerCudaReport {
        schema: QWEN35_REAL_LINEAR_LAYER_CUDA_SCHEMA.to_owned(),
        status: if pass {
            "PASS_COMPONENT_REAL_LINEAR_LAYER_CUDA"
        } else {
            "FAIL_COMPONENT_REAL_LINEAR_LAYER_CUDA"
        }
        .to_owned(),
        component_only: true,
        source_commit: config.source_commit.clone(),
        binary_path: binary_path.display().to_string(),
        binary_sha256,
        device: DeviceReport {
            ordinal: device.ordinal,
            name: device.name,
            compute_capability: [
                device.compute_capability.major,
                device.compute_capability.minor,
            ],
            total_global_memory_bytes: device.total_global_memory_bytes,
        },
        config_sha256,
        gguf_path: config.gguf_path.display().to_string(),
        gguf_file_sha256: manifest.file_sha256.clone(),
        artifact: ArtifactReport {
            manifest_tensor_count: manifest.tensor_count,
            audited_tensor_count: descriptors.len(),
            strict_profile_expected_tensor_count: STRICT_PROFILE_TENSOR_COUNT,
            strict_profile_count_match,
            strict_profile_qualified: false,
            classification: artifact_classification.to_owned(),
        },
        token_ids: vec![config.token_id, second_token_id],
        geometry: GeometryReport {
            layer: config.layer,
            hidden: HIDDEN,
            intermediate: INTERMEDIATE,
            key_heads: KEY_HEADS,
            value_heads: VALUE_HEADS,
            head_dim: HEAD_DIM,
            qkv_elements: QKV_ELEMENTS,
            recurrent_elements: RECURRENT_ELEMENTS,
            state_elements: STATE_ELEMENTS,
            ffn_stages_per_pass: 4,
            ordered_operations_per_pass: 9,
        },
        tensors,
        timing: TimingReport {
            warmup_passes: config.warmup,
            steady_passes: config.iterations,
            ordered_operations_per_pass: 9,
            steady_cuda_event_ms,
            steady_batch_wall_ms,
            per_pass_cuda_event_ms: f64::from(steady_cuda_event_ms)
                / config.iterations.max(1) as f64,
            reference_ms,
        },
        completion,
        numerical,
        resources: ResourceReport {
            setup_boundaries,
            setup_allocations,
            steady_boundaries,
            steady_allocations,
            sequence_input_boundaries,
            completion_boundaries,
            readback_boundaries,
            cleanup_boundaries,
            total_boundaries,
            boundary_ledger_closed,
            final_allocation_delta,
        },
        checks: CheckReport {
            geometry_valid: true,
            finite_outputs,
            linear_state_changed: initial_layer_checkpoint != layer_checkpoint_a,
            attention_residual_shape_valid: expected_a.attention_residual.len() == HIDDEN
                && audited_a.attention_residual.len() == HIDDEN,
            post_norm_shape_valid: expected_a.post_attention_norm.len() == HIDDEN
                && audited_a.post_norm.len() == HIDDEN,
            ffn_shape_valid: expected_a.ffn_output.len() == HIDDEN
                && audited_a.ffn_output.len() == HIDDEN,
            final_hidden_shape_valid: expected_a.hidden.len() == HIDDEN
                && audited_a.final_hidden.len() == HIDDEN,
            checkpoint_replay_exact: replay_exact,
            completion_proof_valid,
            steady_no_host_boundary,
        },
        claims: ClaimsReport {
            real_weight_linear_attention_and_ffn_chain_executed: true,
            input_projections_cpu_derived: true,
            complete_transformer_executed: false,
            decode_executed: false,
            semantic_parity_established: false,
            product_tps_permitted: false,
            default_cli_enabled: false,
        },
    })
}

fn validate_config(
    config: &Qwen35RealLinearLayerCudaConfig,
) -> Result<(), Qwen35RealLinearLayerCudaError> {
    if config.gguf_path.as_os_str().is_empty() || !config.gguf_path.is_file() {
        return Err(Qwen35RealLinearLayerCudaError::InvalidConfiguration(
            "GGUF path",
        ));
    }
    if config.config_path.as_os_str().is_empty() || !config.config_path.is_file() {
        return Err(Qwen35RealLinearLayerCudaError::InvalidConfiguration(
            "config path",
        ));
    }
    if config.device_ordinal < 0 || config.iterations == 0 || config.token_id >= VOCAB - 1 {
        return Err(Qwen35RealLinearLayerCudaError::InvalidConfiguration(
            "runner bounds",
        ));
    }
    Ok(())
}

pub(crate) fn required_descriptors(
    manifest: &GgufManifest,
    layer: usize,
    ffn_type: u32,
) -> Result<Vec<GgufTensor>, Qwen35RealLinearLayerCudaError> {
    let specs = vec![
        (
            EMBEDDING.to_owned(),
            vec![HIDDEN as u64, VOCAB as u64],
            GGML_TYPE_BF16,
        ),
        (
            layer_tensor_name(layer, ATTN_NORM),
            vec![HIDDEN as u64],
            GGML_TYPE_F32,
        ),
        (
            layer_tensor_name(layer, QKV),
            vec![HIDDEN as u64, QKV_ELEMENTS as u64],
            GGML_TYPE_Q8_0,
        ),
        (
            layer_tensor_name(layer, GATE),
            vec![HIDDEN as u64, RECURRENT_ELEMENTS as u64],
            GGML_TYPE_Q8_0,
        ),
        (
            layer_tensor_name(layer, ALPHA),
            vec![HIDDEN as u64, VALUE_HEADS as u64],
            GGML_TYPE_BF16,
        ),
        (
            layer_tensor_name(layer, BETA),
            vec![HIDDEN as u64, VALUE_HEADS as u64],
            GGML_TYPE_BF16,
        ),
        (
            layer_tensor_name(layer, SSM_OUT),
            vec![RECURRENT_ELEMENTS as u64, HIDDEN as u64],
            GGML_TYPE_Q8_0,
        ),
        (
            layer_tensor_name(layer, CONV),
            vec![KERNEL_SIZE as u64, CONV_CHANNELS as u64],
            GGML_TYPE_F32,
        ),
        (
            layer_tensor_name(layer, SSM_A),
            vec![VALUE_HEADS as u64],
            GGML_TYPE_F32,
        ),
        (
            layer_tensor_name(layer, DT_BIAS),
            vec![VALUE_HEADS as u64],
            GGML_TYPE_F32,
        ),
        (
            layer_tensor_name(layer, SSM_NORM),
            vec![HEAD_DIM as u64],
            GGML_TYPE_F32,
        ),
        (
            layer_tensor_name(layer, POST_NORM),
            vec![HIDDEN as u64],
            GGML_TYPE_F32,
        ),
        (
            layer_tensor_name(layer, FFN_GATE),
            vec![HIDDEN as u64, INTERMEDIATE as u64],
            ffn_type,
        ),
        (
            layer_tensor_name(layer, FFN_UP),
            vec![HIDDEN as u64, INTERMEDIATE as u64],
            ffn_type,
        ),
        (
            layer_tensor_name(layer, FFN_DOWN),
            vec![INTERMEDIATE as u64, HIDDEN as u64],
            ffn_type,
        ),
    ];
    let mut specs = specs;
    if ffn_type == GGML_TYPE_NVFP4 {
        specs.extend([
            (
                layer_tensor_name(layer, FFN_GATE_SCALE),
                vec![1],
                GGML_TYPE_F32,
            ),
            (
                layer_tensor_name(layer, FFN_UP_SCALE),
                vec![1],
                GGML_TYPE_F32,
            ),
            (
                layer_tensor_name(layer, FFN_DOWN_SCALE),
                vec![1],
                GGML_TYPE_F32,
            ),
            (
                layer_tensor_name(layer, FFN_GATE_INPUT_SCALE),
                vec![1],
                GGML_TYPE_F32,
            ),
            (
                layer_tensor_name(layer, FFN_UP_INPUT_SCALE),
                vec![1],
                GGML_TYPE_F32,
            ),
            (
                layer_tensor_name(layer, FFN_DOWN_INPUT_SCALE),
                vec![1],
                GGML_TYPE_F32,
            ),
        ]);
    } else if ffn_type != GGML_TYPE_Q8_0 {
        return Err(Qwen35RealLinearLayerCudaError::InvalidConfiguration(
            "unsupported FFN weight type",
        ));
    }
    specs
        .into_iter()
        .map(|(name, dimensions, ggml_type)| {
            let tensor = manifest.tensor(&name).ok_or(
                Qwen35RealLinearLayerCudaError::InvalidConfiguration("required tensor missing"),
            )?;
            if tensor.dimensions != dimensions || tensor.ggml_type != ggml_type {
                return Err(Qwen35RealLinearLayerCudaError::InvalidConfiguration(
                    "required tensor geometry/type",
                ));
            }
            Ok(tensor.clone())
        })
        .collect()
}

fn layer_tensor_name(layer: usize, suffix: &str) -> String {
    format!("blk.{layer}.{suffix}")
}

pub(crate) fn read_embedding_row(
    reader: &mut augur_model_qwen35::GgufTensorReader<'_>,
    descriptor: &GgufTensor,
    token_id: usize,
) -> Result<Vec<f32>, Qwen35RealLinearLayerCudaError> {
    let row_bytes = HIDDEN * 2;
    let offset = token_id.checked_mul(row_bytes).ok_or(
        Qwen35RealLinearLayerCudaError::InvalidConfiguration("embedding offset"),
    )?;
    let bytes = reader.read_tensor_slice(&descriptor.name, offset as u64, row_bytes as u64)?;
    let mut values = vec![0.0_f32; HIDDEN];
    dequantize_bf16(&bytes, &mut values)?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(Qwen35RealLinearLayerCudaError::InvalidConfiguration(
            "embedding finite",
        ));
    }
    Ok(values)
}

pub(crate) fn positive_scale(
    values: &[f32],
    field: &'static str,
) -> Result<f32, Qwen35RealLinearLayerCudaError> {
    if values.len() != 1 || !values[0].is_finite() || values[0] <= 0.0 {
        return Err(Qwen35RealLinearLayerCudaError::InvalidConfiguration(field));
    }
    Ok(values[0])
}

pub(crate) fn checked_f32_extent(elements: usize) -> Result<usize, Qwen35RealLinearLayerCudaError> {
    elements.checked_mul(core::mem::size_of::<f32>()).ok_or(
        Qwen35RealLinearLayerCudaError::InvalidConfiguration("F32 extent"),
    )
}

pub(crate) fn checked_nvfp4_weight_extent(
    rows: usize,
    columns: usize,
) -> Result<usize, Qwen35RealLinearLayerCudaError> {
    if rows == 0 || columns == 0 || !columns.is_multiple_of(NVFP4_BLOCK_ELEMENTS) {
        return Err(Qwen35RealLinearLayerCudaError::InvalidConfiguration(
            "NVFP4 column geometry",
        ));
    }
    rows.checked_mul(columns / NVFP4_BLOCK_ELEMENTS)
        .and_then(|blocks| blocks.checked_mul(NVFP4_BLOCK_BYTES))
        .ok_or(Qwen35RealLinearLayerCudaError::InvalidConfiguration(
            "NVFP4 weight bytes",
        ))
}

pub(crate) fn checked_q8_weight_extent(
    rows: usize,
    columns: usize,
) -> Result<usize, Qwen35RealLinearLayerCudaError> {
    if rows == 0 || columns == 0 || !columns.is_multiple_of(Q8_0_BLOCK_ELEMENTS) {
        return Err(Qwen35RealLinearLayerCudaError::InvalidConfiguration(
            "Q8_0 column geometry",
        ));
    }
    rows.checked_mul(columns / Q8_0_BLOCK_ELEMENTS)
        .and_then(|blocks| blocks.checked_mul(Q8_0_BLOCK_BYTES))
        .ok_or(Qwen35RealLinearLayerCudaError::InvalidConfiguration(
            "Q8_0 weight bytes",
        ))
}

fn require_exact_extent(
    actual: usize,
    expected: usize,
    field: &'static str,
) -> Result<(), Qwen35RealLinearLayerCudaError> {
    if actual != expected {
        return Err(Qwen35RealLinearLayerCudaError::InvalidConfiguration(field));
    }
    Ok(())
}

pub(crate) fn project_token(
    hidden: &[f32],
    attn_norm: &[f32],
    qkv_descriptor: &GgufTensor,
    qkv_payload: &[u8],
    gate_descriptor: &GgufTensor,
    gate_payload: &[u8],
    alpha_descriptor: &GgufTensor,
    alpha_payload: &[u8],
    beta_descriptor: &GgufTensor,
    beta_payload: &[u8],
    dt_bias: &[f32],
    ssm_a: &[f32],
    epsilon: f32,
) -> Result<ProjectedToken, Qwen35RealLinearLayerCudaError> {
    let normalized = rms_norm_last_dim(
        &ReferenceTensor::new(vec![1, HIDDEN], hidden.to_vec())?,
        &ReferenceTensor::new(vec![HIDDEN], attn_norm.to_vec())?,
        epsilon,
    )?;
    let input = normalized.values();
    let qkv = quantized_matvec_fused_simd(qkv_descriptor, qkv_payload, input)?;
    let z = quantized_matvec_fused_simd(gate_descriptor, gate_payload, input)?;
    let alpha_raw = quantized_matvec_fused_simd(alpha_descriptor, alpha_payload, input)?;
    let beta_raw = quantized_matvec_fused_simd(beta_descriptor, beta_payload, input)?;
    if qkv.len() != QKV_ELEMENTS
        || z.len() != RECURRENT_ELEMENTS
        || alpha_raw.len() != VALUE_HEADS
        || beta_raw.len() != VALUE_HEADS
        || dt_bias.len() != VALUE_HEADS
        || ssm_a.len() != VALUE_HEADS
    {
        return Err(Qwen35RealLinearLayerCudaError::InvalidConfiguration(
            "projection output geometry",
        ));
    }
    let mut decay_gate = Vec::with_capacity(VALUE_HEADS);
    let mut beta = Vec::with_capacity(VALUE_HEADS);
    for index in 0..VALUE_HEADS {
        let value = alpha_raw[index] + dt_bias[index];
        let softplus = if value > 20.0 {
            value
        } else if value < -20.0 {
            value.exp()
        } else {
            (1.0 + value.exp()).ln()
        };
        decay_gate.push(softplus * ssm_a[index]);
        let gate = beta_raw[index];
        beta.push(if gate >= 0.0 {
            1.0 / (1.0 + (-gate).exp())
        } else {
            let exponent = gate.exp();
            exponent / (1.0 + exponent)
        });
    }
    if qkv
        .iter()
        .chain(z.iter())
        .chain(decay_gate.iter())
        .chain(beta.iter())
        .any(|value| !value.is_finite())
    {
        return Err(Qwen35RealLinearLayerCudaError::InvalidConfiguration(
            "projection finite",
        ));
    }
    Ok(ProjectedToken {
        qkv,
        z,
        decay_gate,
        beta,
    })
}

pub(crate) fn convert_conv_layout(raw: &[f32]) -> Result<Vec<f32>, Qwen35RealLinearLayerCudaError> {
    if raw.len() != CONV_WEIGHT_ELEMENTS || raw.iter().any(|value| !value.is_finite()) {
        return Err(Qwen35RealLinearLayerCudaError::InvalidConfiguration(
            "conv weights",
        ));
    }
    let mut converted = vec![0.0_f32; raw.len()];
    for channel in 0..CONV_CHANNELS {
        for tap in 0..KERNEL_SIZE {
            converted[channel * KERNEL_SIZE + tap] = raw[tap + channel * KERNEL_SIZE];
        }
    }
    Ok(converted)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn upload_static(
    stream: &OwnedCudaStream,
    qkv: &OwnedCudaBuffer,
    z: &OwnedCudaBuffer,
    conv: &OwnedCudaBuffer,
    decay: &OwnedCudaBuffer,
    beta: &OwnedCudaBuffer,
    ssm_norm: &OwnedCudaBuffer,
    output_weight: &OwnedCudaBuffer,
    post_norm_weight: &OwnedCudaBuffer,
    ffn_gate_weight: &OwnedCudaBuffer,
    ffn_up_weight: &OwnedCudaBuffer,
    ffn_down_weight: &OwnedCudaBuffer,
    token: &ProjectedToken,
    conv_values: &[f32],
    ssm_norm_values: &[f32],
    post_norm_values: &[f32],
    output_weight_bytes: &[u8],
    ffn_gate_weight_bytes: &[u8],
    ffn_up_weight_bytes: &[u8],
    ffn_down_weight_bytes: &[u8],
) -> Result<(), Qwen35RealLinearLayerCudaError> {
    qkv.copy_from_host_sync(stream, &f32_bytes(&token.qkv))?;
    z.copy_from_host_sync(stream, &f32_bytes(&token.z))?;
    conv.copy_from_host_sync(stream, &f32_bytes(conv_values))?;
    decay.copy_from_host_sync(stream, &f32_bytes(&token.decay_gate))?;
    beta.copy_from_host_sync(stream, &f32_bytes(&token.beta))?;
    ssm_norm.copy_from_host_sync(stream, &f32_bytes(ssm_norm_values))?;
    output_weight.copy_from_host_sync(stream, output_weight_bytes)?;
    post_norm_weight.copy_from_host_sync(stream, &f32_bytes(post_norm_values))?;
    ffn_gate_weight.copy_from_host_sync(stream, ffn_gate_weight_bytes)?;
    ffn_up_weight.copy_from_host_sync(stream, ffn_up_weight_bytes)?;
    ffn_down_weight.copy_from_host_sync(stream, ffn_down_weight_bytes)?;
    Ok(())
}

pub(crate) fn upload_token(
    stream: &OwnedCudaStream,
    qkv: &OwnedCudaBuffer,
    z: &OwnedCudaBuffer,
    decay: &OwnedCudaBuffer,
    beta: &OwnedCudaBuffer,
    input: &OwnedCudaBuffer,
    token: &ProjectedToken,
    hidden: &[f32],
) -> Result<(), Qwen35RealLinearLayerCudaError> {
    qkv.copy_from_host_sync(stream, &f32_bytes(&token.qkv))?;
    z.copy_from_host_sync(stream, &f32_bytes(&token.z))?;
    decay.copy_from_host_sync(stream, &f32_bytes(&token.decay_gate))?;
    beta.copy_from_host_sync(stream, &f32_bytes(&token.beta))?;
    input.copy_from_host_sync(stream, &f32_bytes(hidden))?;
    Ok(())
}

pub(crate) fn f32_view(
    buffer: &OwnedCudaBuffer,
    elements: usize,
) -> Result<CudaF32View, Qwen35RealLinearLayerCudaError> {
    let bytes = checked_f32_extent(elements)?;
    require_exact_extent(buffer.len_bytes(), bytes, "F32 view extent")?;
    Ok(CudaF32View::new(
        buffer.view(0, bytes)?,
        PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
        elements,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn bind_chain(
    stream: &OwnedCudaStream,
    qkv: &OwnedCudaBuffer,
    z: &OwnedCudaBuffer,
    conv: &OwnedCudaBuffer,
    history: &OwnedCudaBuffer,
    decay: &OwnedCudaBuffer,
    beta: &OwnedCudaBuffer,
    state: &OwnedCudaBuffer,
    ssm_norm: &OwnedCudaBuffer,
    activated: &OwnedCudaBuffer,
    recurrent: &OwnedCudaBuffer,
    linear_output: &OwnedCudaBuffer,
    output_weight: &OwnedCudaBuffer,
    output_q8_1: Option<&OwnedCudaBuffer>,
    attention: &OwnedCudaBuffer,
    input: CudaF32View,
    attention_residual: &OwnedCudaBuffer,
    post_norm: &OwnedCudaBuffer,
    post_norm_weight: &OwnedCudaBuffer,
    ffn_gate_weight: &OwnedCudaBuffer,
    ffn_up_weight: &OwnedCudaBuffer,
    ffn_gate_output: &OwnedCudaBuffer,
    ffn_up_output: &OwnedCudaBuffer,
    ffn_activation: &OwnedCudaBuffer,
    ffn_down_weight: &OwnedCudaBuffer,
    ffn_output: &OwnedCudaBuffer,
    final_hidden: &OwnedCudaBuffer,
    ffn_input_q8_1: Option<&OwnedCudaBuffer>,
    ffn_down_q8_1: Option<&OwnedCudaBuffer>,
    output_weight_bytes: usize,
    gate_weight_bytes: usize,
    up_weight_bytes: usize,
    down_weight_bytes: usize,
    ffn_type: u32,
    gate_scale: f32,
    up_scale: f32,
    down_scale: f32,
    epsilon: f32,
) -> Result<BoundChain, Qwen35RealLinearLayerCudaError> {
    let ffn = bind_legacy_ffn(stream, post_norm, ffn_gate_weight, ffn_up_weight,
        ffn_gate_output, ffn_up_output, ffn_activation, ffn_down_weight, ffn_output,
        ffn_input_q8_1, ffn_down_q8_1, gate_weight_bytes, up_weight_bytes,
        down_weight_bytes, ffn_type, gate_scale, up_scale, down_scale)?;
    bind_chain_with_ffn(stream, qkv, z, conv, history, decay, beta, state,
        ssm_norm, activated, recurrent, linear_output, output_weight, output_q8_1,
        attention, input, attention_residual, post_norm, post_norm_weight,
        ffn_output, final_hidden, output_weight_bytes, epsilon, ffn)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn bind_legacy_ffn(
    stream: &OwnedCudaStream, post_norm: &OwnedCudaBuffer,
    ffn_gate_weight: &OwnedCudaBuffer, ffn_up_weight: &OwnedCudaBuffer,
    ffn_gate_output: &OwnedCudaBuffer, ffn_up_output: &OwnedCudaBuffer,
    ffn_activation: &OwnedCudaBuffer, ffn_down_weight: &OwnedCudaBuffer,
    ffn_output: &OwnedCudaBuffer,
    ffn_input_q8_1: Option<&OwnedCudaBuffer>, ffn_down_q8_1: Option<&OwnedCudaBuffer>,
    gate_weight_bytes: usize, up_weight_bytes: usize, down_weight_bytes: usize,
    ffn_type: u32, gate_scale: f32, up_scale: f32, down_scale: f32,
) -> Result<BoundFfn, Qwen35RealLinearLayerCudaError> {
    let (ffn_input_quantize, gate_op, up_op, ffn_down_quantize, down_op) =
        match (ffn_input_q8_1, ffn_down_q8_1) {
            (None, None) => (
                None,
                bind_ffn(
                    stream,
                    ffn_gate_weight,
                    gate_weight_bytes,
                    ffn_gate_output,
                    INTERMEDIATE,
                    HIDDEN,
                    f32_view(post_norm, HIDDEN)?,
                    ffn_type,
                    gate_scale,
                )?,
                bind_ffn(
                    stream,
                    ffn_up_weight,
                    up_weight_bytes,
                    ffn_up_output,
                    INTERMEDIATE,
                    HIDDEN,
                    f32_view(post_norm, HIDDEN)?,
                    ffn_type,
                    up_scale,
                )?,
                None,
                bind_ffn(
                    stream,
                    ffn_down_weight,
                    down_weight_bytes,
                    ffn_output,
                    HIDDEN,
                    INTERMEDIATE,
                    f32_view(ffn_activation, INTERMEDIATE)?,
                    ffn_type,
                    down_scale,
                )?,
            ),
            (Some(input_workspace), Some(down_workspace)) => {
                let input_q8_1 = q8_1_activation_view(input_workspace, HIDDEN)?;
                let down_q8_1 = q8_1_activation_view(down_workspace, INTERMEDIATE)?;
                (
                    Some(CudaQ8_1ActivationQuantizeArgsV1::bind(
                        stream,
                        f32_view(post_norm, HIDDEN)?,
                        input_q8_1,
                    )?),
                    bind_ffn_q8_1(
                        stream,
                        ffn_gate_weight,
                        gate_weight_bytes,
                        ffn_gate_output,
                        INTERMEDIATE,
                        HIDDEN,
                        input_q8_1,
                        ffn_type,
                        gate_scale,
                    )?,
                    bind_ffn_q8_1(
                        stream,
                        ffn_up_weight,
                        up_weight_bytes,
                        ffn_up_output,
                        INTERMEDIATE,
                        HIDDEN,
                        input_q8_1,
                        ffn_type,
                        up_scale,
                    )?,
                    Some(CudaQ8_1ActivationQuantizeArgsV1::bind(
                        stream,
                        f32_view(ffn_activation, INTERMEDIATE)?,
                        down_q8_1,
                    )?),
                    bind_ffn_q8_1(
                        stream,
                        ffn_down_weight,
                        down_weight_bytes,
                        ffn_output,
                        HIDDEN,
                        INTERMEDIATE,
                        down_q8_1,
                        ffn_type,
                        down_scale,
                    )?,
                )
            }
            _ => {
                return Err(Qwen35RealLinearLayerCudaError::InvalidConfiguration(
                    "both Q8_1 FFN workspaces must be supplied together",
                ));
            }
        };
    let silu_mul_op = CudaSiluMulF32ArgsV1::bind(
        stream,
        f32_view(ffn_gate_output, INTERMEDIATE)?,
        f32_view(ffn_up_output, INTERMEDIATE)?,
        f32_view(ffn_activation, INTERMEDIATE)?,
    )?;
    #[cfg(feature = "qwen35-fused-ffn-swiglu")]
    let fused_gate_up = match ffn_input_q8_1 {
        Some(input_workspace) => Some(bind_fused_ffn_q8_1(
            stream,
            ffn_gate_weight,
            gate_weight_bytes,
            ffn_up_weight,
            up_weight_bytes,
            ffn_activation,
            INTERMEDIATE,
            HIDDEN,
            q8_1_activation_view(input_workspace, HIDDEN)?,
            ffn_type,
            gate_scale,
            up_scale,
        )?),
        None => None,
    };
    Ok(BoundFfn::Legacy(LegacyFfn {
        input_quantize: ffn_input_quantize, gate: gate_op, up: up_op, silu_mul: silu_mul_op,
        #[cfg(feature = "qwen35-fused-ffn-swiglu")]
        fused_gate_up,
        down_quantize: ffn_down_quantize, down: down_op,
    }))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn bind_chain_with_ffn(
    stream: &OwnedCudaStream, qkv: &OwnedCudaBuffer, z: &OwnedCudaBuffer,
    conv: &OwnedCudaBuffer, history: &OwnedCudaBuffer, decay: &OwnedCudaBuffer,
    beta: &OwnedCudaBuffer, state: &OwnedCudaBuffer, ssm_norm: &OwnedCudaBuffer,
    activated: &OwnedCudaBuffer, recurrent: &OwnedCudaBuffer,
    linear_output: &OwnedCudaBuffer, output_weight: &OwnedCudaBuffer,
    output_q8_1: Option<&OwnedCudaBuffer>, attention: &OwnedCudaBuffer,
    input: CudaF32View, attention_residual: &OwnedCudaBuffer, post_norm: &OwnedCudaBuffer,
    post_norm_weight: &OwnedCudaBuffer, ffn_output: &OwnedCudaBuffer,
    final_hidden: &OwnedCudaBuffer, output_weight_bytes: usize, epsilon: f32,
    ffn: BoundFfn,
) -> Result<BoundChain, Qwen35RealLinearLayerCudaError> {
    let recurrent_op = CudaQwen35LinearStepArgsV1::bind(
        stream,
        f32_view(qkv, QKV_ELEMENTS)?,
        f32_view(z, RECURRENT_ELEMENTS)?,
        f32_view(conv, CONV_WEIGHT_ELEMENTS)?,
        f32_view(history, CONV_HISTORY_ELEMENTS)?,
        f32_view(decay, VALUE_HEADS)?,
        f32_view(beta, VALUE_HEADS)?,
        f32_view(state, STATE_ELEMENTS)?,
        f32_view(ssm_norm, HEAD_DIM)?,
        f32_view(activated, QKV_ELEMENTS)?,
        f32_view(recurrent, RECURRENT_ELEMENTS)?,
        f32_view(linear_output, RECURRENT_ELEMENTS)?,
        epsilon,
    )?;
    let output_weight_view = CudaQ8_0WeightViewV1::new(
        output_weight_view(output_weight, output_weight_bytes)?,
        PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_0_CUDA_V1,
        HIDDEN,
        RECURRENT_ELEMENTS,
    )?;
    let (output_projection_quantize, output_projection_op) = match output_q8_1 {
        None => (
            None,
            Q8ProjectionOp::F32(CudaQ8_0F32MatvecArgsV1::bind(
                stream,
                output_weight_view,
                f32_view(linear_output, RECURRENT_ELEMENTS)?,
                f32_view(attention, HIDDEN)?,
            )?),
        ),
        Some(workspace) => {
            let activation = q8_1_activation_view(workspace, RECURRENT_ELEMENTS)?;
            (
                Some(CudaQ8_1ActivationQuantizeArgsV1::bind(
                    stream,
                    f32_view(linear_output, RECURRENT_ELEMENTS)?,
                    activation,
                )?),
                Q8ProjectionOp::Q8_1(CudaQ8_0Q8_1MatvecArgsV1::bind(
                    stream,
                    output_weight_view,
                    activation,
                    f32_view(attention, HIDDEN)?,
                )?),
            )
        }
    };
    let attention_residual_op = CudaAddF32ArgsV1::bind(
        stream,
        f32_view(attention, HIDDEN)?,
        input,
        f32_view(attention_residual, HIDDEN)?,
    )?;
    let post_norm_op = CudaPlanRmsnormF32::bind(
        stream,
        f32_view(attention_residual, HIDDEN)?,
        f32_view(post_norm_weight, HIDDEN)?,
        f32_view(post_norm, HIDDEN)?,
        epsilon,
    )?;
    let final_residual_op = CudaAddF32ArgsV1::bind(
        stream,
        f32_view(attention_residual, HIDDEN)?,
        f32_view(ffn_output, HIDDEN)?,
        f32_view(final_hidden, HIDDEN)?,
    )?;
    Ok(BoundChain {
        recurrent: recurrent_op, output_projection_quantize,
        output_projection: output_projection_op, attention_residual: attention_residual_op,
        post_norm: post_norm_op, ffn, final_residual: final_residual_op,
    })
}

fn output_weight_view(
    buffer: &OwnedCudaBuffer,
    bytes: usize,
) -> Result<augur_backend::CudaBufferView, Qwen35RealLinearLayerCudaError> {
    Ok(buffer.view(0, bytes)?)
}

pub(crate) fn bind_nvfp4(
    stream: &OwnedCudaStream,
    weight_buffer: &OwnedCudaBuffer,
    weight_bytes: usize,
    output_buffer: &OwnedCudaBuffer,
    rows: usize,
    columns: usize,
    input: CudaF32View,
    tensor_scale: f32,
) -> Result<CudaNvfp4Sm70F32MatvecArgsV1, Qwen35RealLinearLayerCudaError> {
    let expected_weight_bytes = checked_nvfp4_weight_extent(rows, columns)?;
    let expected_output_bytes = checked_f32_extent(rows)?;
    require_exact_extent(weight_bytes, expected_weight_bytes, "NVFP4 weight extent")?;
    require_exact_extent(
        weight_buffer.len_bytes(),
        expected_weight_bytes,
        "NVFP4 weight buffer extent",
    )?;
    require_exact_extent(
        output_buffer.len_bytes(),
        expected_output_bytes,
        "NVFP4 output extent",
    )?;
    Ok(CudaNvfp4Sm70F32MatvecArgsV1::bind(
        stream,
        CudaNvfp4Sm70WeightViewV1::new(
            weight_buffer.view(0, expected_weight_bytes)?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_NVFP4_CUDA_SM70_V1,
            rows,
            columns,
        )?,
        input,
        CudaF32View::new(
            output_buffer.view(0, expected_output_bytes)?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            rows,
        )?,
        tensor_scale,
    )?)
}

pub(crate) fn bind_q8(
    stream: &OwnedCudaStream,
    weight_buffer: &OwnedCudaBuffer,
    weight_bytes: usize,
    output_buffer: &OwnedCudaBuffer,
    rows: usize,
    columns: usize,
    input: CudaF32View,
) -> Result<CudaQ8_0F32MatvecArgsV1, Qwen35RealLinearLayerCudaError> {
    let expected_weight_bytes = checked_q8_weight_extent(rows, columns)?;
    let expected_output_bytes = checked_f32_extent(rows)?;
    require_exact_extent(weight_bytes, expected_weight_bytes, "Q8_0 weight extent")?;
    require_exact_extent(
        weight_buffer.len_bytes(),
        expected_weight_bytes,
        "Q8_0 weight buffer extent",
    )?;
    require_exact_extent(
        output_buffer.len_bytes(),
        expected_output_bytes,
        "Q8_0 output extent",
    )?;
    Ok(CudaQ8_0F32MatvecArgsV1::bind(
        stream,
        CudaQ8_0WeightViewV1::new(
            weight_buffer.view(0, expected_weight_bytes)?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_0_CUDA_V1,
            rows,
            columns,
        )?,
        input,
        CudaF32View::new(
            output_buffer.view(0, expected_output_bytes)?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            rows,
        )?,
    )?)
}

pub(crate) fn bind_ffn(
    stream: &OwnedCudaStream,
    weight_buffer: &OwnedCudaBuffer,
    weight_bytes: usize,
    output_buffer: &OwnedCudaBuffer,
    rows: usize,
    columns: usize,
    input: CudaF32View,
    ffn_type: u32,
    tensor_scale: f32,
) -> Result<FfnMatvecOp, Qwen35RealLinearLayerCudaError> {
    match ffn_type {
        GGML_TYPE_NVFP4 => Ok(FfnMatvecOp::Nvfp4F32(bind_nvfp4(
            stream,
            weight_buffer,
            weight_bytes,
            output_buffer,
            rows,
            columns,
            input,
            tensor_scale,
        )?)),
        GGML_TYPE_Q8_0 => {
            if tensor_scale != 1.0 {
                return Err(Qwen35RealLinearLayerCudaError::InvalidConfiguration(
                    "Q8_0 FFN scale must be one",
                ));
            }
            Ok(FfnMatvecOp::Q8_0F32(bind_q8(
                stream,
                weight_buffer,
                weight_bytes,
                output_buffer,
                rows,
                columns,
                input,
            )?))
        }
        _ => Err(Qwen35RealLinearLayerCudaError::InvalidConfiguration(
            "unsupported FFN weight type",
        )),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn bind_ffn_q8_1(
    stream: &OwnedCudaStream,
    weight_buffer: &OwnedCudaBuffer,
    weight_bytes: usize,
    output_buffer: &OwnedCudaBuffer,
    rows: usize,
    columns: usize,
    input: CudaQ8_1ActivationViewV1,
    ffn_type: u32,
    tensor_scale: f32,
) -> Result<FfnMatvecOp, Qwen35RealLinearLayerCudaError> {
    let expected_output_bytes = checked_f32_extent(rows)?;
    require_exact_extent(
        output_buffer.len_bytes(),
        expected_output_bytes,
        "Q8_1 FFN output extent",
    )?;
    let output = CudaF32View::new(
        output_buffer.view(0, expected_output_bytes)?,
        PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
        rows,
    )?;
    match ffn_type {
        GGML_TYPE_NVFP4 => {
            let expected_weight_bytes = checked_nvfp4_weight_extent(rows, columns)?;
            require_exact_extent(
                weight_bytes,
                expected_weight_bytes,
                "NVFP4 Q8_1 weight extent",
            )?;
            require_exact_extent(
                weight_buffer.len_bytes(),
                expected_weight_bytes,
                "NVFP4 Q8_1 weight buffer extent",
            )?;
            Ok(FfnMatvecOp::Nvfp4Q8_1(CudaNvfp4Sm70Q8_1MatvecArgsV1::bind(
                stream,
                CudaNvfp4Sm70WeightViewV1::new(
                    weight_buffer.view(0, expected_weight_bytes)?,
                    PhysicalLayoutId::DEVICE_CONTIGUOUS_NVFP4_CUDA_SM70_V1,
                    rows,
                    columns,
                )?,
                input,
                output,
                tensor_scale,
            )?))
        }
        GGML_TYPE_Q8_0 => {
            if tensor_scale != 1.0 {
                return Err(Qwen35RealLinearLayerCudaError::InvalidConfiguration(
                    "Q8_0 Q8_1 FFN scale must be one",
                ));
            }
            let expected_weight_bytes = checked_q8_weight_extent(rows, columns)?;
            require_exact_extent(
                weight_bytes,
                expected_weight_bytes,
                "Q8_0 Q8_1 weight extent",
            )?;
            require_exact_extent(
                weight_buffer.len_bytes(),
                expected_weight_bytes,
                "Q8_0 Q8_1 weight buffer extent",
            )?;
            Ok(FfnMatvecOp::Q8_0Q8_1(CudaQ8_0Q8_1MatvecArgsV1::bind(
                stream,
                CudaQ8_0WeightViewV1::new(
                    weight_buffer.view(0, expected_weight_bytes)?,
                    PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_0_CUDA_V1,
                    rows,
                    columns,
                )?,
                input,
                output,
            )?))
        }
        _ => Err(Qwen35RealLinearLayerCudaError::InvalidConfiguration(
            "unsupported Q8_1 FFN weight type",
        )),
    }
}

#[cfg(feature = "qwen35-fused-ffn-swiglu")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn bind_fused_ffn_q8_1(
    stream: &OwnedCudaStream,
    gate_weight_buffer: &OwnedCudaBuffer,
    gate_weight_bytes: usize,
    up_weight_buffer: &OwnedCudaBuffer,
    up_weight_bytes: usize,
    output_buffer: &OwnedCudaBuffer,
    rows: usize,
    columns: usize,
    input: CudaQ8_1ActivationViewV1,
    ffn_type: u32,
    gate_tensor_scale: f32,
    up_tensor_scale: f32,
) -> Result<FfnFusedGateUpOp, Qwen35RealLinearLayerCudaError> {
    let expected_output_bytes = checked_f32_extent(rows)?;
    require_exact_extent(
        output_buffer.len_bytes(),
        expected_output_bytes,
        "fused SwiGLU output extent",
    )?;
    let output = CudaF32View::new(
        output_buffer.view(0, expected_output_bytes)?,
        PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
        rows,
    )?;
    match ffn_type {
        GGML_TYPE_NVFP4 => {
            let expected_weight_bytes = checked_nvfp4_weight_extent(rows, columns)?;
            require_exact_extent(
                gate_weight_bytes,
                expected_weight_bytes,
                "fused NVFP4 gate weight extent",
            )?;
            require_exact_extent(
                up_weight_bytes,
                expected_weight_bytes,
                "fused NVFP4 up weight extent",
            )?;
            Ok(FfnFusedGateUpOp::Nvfp4(
                CudaNvfp4Sm70Q8_1FusedSwigluArgsV1::bind(
                    stream,
                    CudaNvfp4Sm70WeightViewV1::new(
                        gate_weight_buffer.view(0, expected_weight_bytes)?,
                        PhysicalLayoutId::DEVICE_CONTIGUOUS_NVFP4_CUDA_SM70_V1,
                        rows,
                        columns,
                    )?,
                    CudaNvfp4Sm70WeightViewV1::new(
                        up_weight_buffer.view(0, expected_weight_bytes)?,
                        PhysicalLayoutId::DEVICE_CONTIGUOUS_NVFP4_CUDA_SM70_V1,
                        rows,
                        columns,
                    )?,
                    input,
                    output,
                    gate_tensor_scale,
                    up_tensor_scale,
                )?,
            ))
        }
        GGML_TYPE_Q8_0 => {
            if gate_tensor_scale != 1.0 || up_tensor_scale != 1.0 {
                return Err(Qwen35RealLinearLayerCudaError::InvalidConfiguration(
                    "Q8_0 fused SwiGLU scales must be one",
                ));
            }
            let expected_weight_bytes = checked_q8_weight_extent(rows, columns)?;
            require_exact_extent(
                gate_weight_bytes,
                expected_weight_bytes,
                "fused Q8_0 gate weight extent",
            )?;
            require_exact_extent(
                up_weight_bytes,
                expected_weight_bytes,
                "fused Q8_0 up weight extent",
            )?;
            Ok(FfnFusedGateUpOp::Q8_0(CudaQ8_0Q8_1FusedSwigluArgsV1::bind(
                stream,
                CudaQ8_0WeightViewV1::new(
                    gate_weight_buffer.view(0, expected_weight_bytes)?,
                    PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_0_CUDA_V1,
                    rows,
                    columns,
                )?,
                CudaQ8_0WeightViewV1::new(
                    up_weight_buffer.view(0, expected_weight_bytes)?,
                    PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_0_CUDA_V1,
                    rows,
                    columns,
                )?,
                input,
                output,
            )?))
        }
        _ => Err(Qwen35RealLinearLayerCudaError::InvalidConfiguration(
            "unsupported fused SwiGLU weight type",
        )),
    }
}

pub(crate) fn q8_1_activation_view(
    buffer: &OwnedCudaBuffer,
    elements: usize,
) -> Result<CudaQ8_1ActivationViewV1, Qwen35RealLinearLayerCudaError> {
    if elements == 0 || !elements.is_multiple_of(Q8_1_ACTIVATION_BLOCK_ELEMENTS) {
        return Err(Qwen35RealLinearLayerCudaError::InvalidConfiguration(
            "Q8_1 activation elements",
        ));
    }
    let bytes = elements
        .checked_div(Q8_1_ACTIVATION_BLOCK_ELEMENTS)
        .and_then(|blocks| blocks.checked_mul(Q8_1_ACTIVATION_BLOCK_BYTES))
        .ok_or(Qwen35RealLinearLayerCudaError::InvalidConfiguration(
            "Q8_1 activation extent",
        ))?;
    require_exact_extent(buffer.len_bytes(), bytes, "Q8_1 activation buffer extent")?;
    Ok(CudaQ8_1ActivationViewV1::new(
        buffer.view(0, bytes)?,
        PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_1_ACTIVATION_CUDA_V1,
        elements,
    )?)
}

pub(crate) fn execute_and_readback(
    stream: &OwnedCudaStream,
    chain: &BoundChain,
    state: &OwnedCudaBuffer,
    history: &OwnedCudaBuffer,
    attention: &OwnedCudaBuffer,
    attention_residual: &OwnedCudaBuffer,
    post_norm: &OwnedCudaBuffer,
    ffn_output: &OwnedCudaBuffer,
    final_hidden: &OwnedCudaBuffer,
    recurrent: &OwnedCudaBuffer,
) -> Result<DeviceReadback, Qwen35RealLinearLayerCudaError> {
    let completion_before = cuda_host_boundary_snapshot();
    chain.submit(stream)?;
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
            return Err(Qwen35RealLinearLayerCudaError::CompletionPollLimit);
        }
        spin_loop();
    };
    let proof_ok = proof.covers_submission(receipt) && proof.belongs_to_stream(stream)?;
    if !proof_ok {
        return Err(Qwen35RealLinearLayerCudaError::InvalidConfiguration(
            "owner epoch proof",
        ));
    }
    let completion_boundaries = boundary_delta(completion_before, cuda_host_boundary_snapshot())?;
    let readback_before = cuda_host_boundary_snapshot();
    let mut attention_bytes = vec![0_u8; HIDDEN * 4];
    let mut attention_residual_bytes = vec![0_u8; HIDDEN * 4];
    let mut post_norm_bytes = vec![0_u8; HIDDEN * 4];
    let mut ffn_output_bytes = vec![0_u8; HIDDEN * 4];
    let mut final_hidden_bytes = vec![0_u8; HIDDEN * 4];
    let mut recurrent_bytes = vec![0_u8; RECURRENT_ELEMENTS * 4];
    let mut state_bytes = vec![0_u8; STATE_ELEMENTS * 4];
    let mut history_bytes = vec![0_u8; CONV_HISTORY_ELEMENTS * 4];
    attention.copy_to_host_sync_with_submission_proof(stream, &mut attention_bytes, receipt)?;
    attention_residual.copy_to_host_sync_with_submission_proof(
        stream,
        &mut attention_residual_bytes,
        receipt,
    )?;
    post_norm.copy_to_host_sync_with_submission_proof(stream, &mut post_norm_bytes, receipt)?;
    ffn_output.copy_to_host_sync_with_submission_proof(stream, &mut ffn_output_bytes, receipt)?;
    final_hidden.copy_to_host_sync_with_submission_proof(
        stream,
        &mut final_hidden_bytes,
        receipt,
    )?;
    recurrent.copy_to_host_sync_with_submission_proof(stream, &mut recurrent_bytes, receipt)?;
    state.copy_to_host_sync_with_submission_proof(stream, &mut state_bytes, receipt)?;
    history.copy_to_host_sync_with_submission_proof(stream, &mut history_bytes, receipt)?;
    Ok(DeviceReadback {
        attention: f32_vec(&attention_bytes),
        attention_residual: f32_vec(&attention_residual_bytes),
        post_norm: f32_vec(&post_norm_bytes),
        ffn_output: f32_vec(&ffn_output_bytes),
        final_hidden: f32_vec(&final_hidden_bytes),
        recurrent: f32_vec(&recurrent_bytes),
        state: f32_vec(&state_bytes),
        history: f32_vec(&history_bytes),
        polls,
        proof: proof_ok,
        completion_boundaries,
        readback_boundaries: boundary_delta(readback_before, cuda_host_boundary_snapshot())?,
    })
}

fn readback_equal(left: &DeviceReadback, right: &DeviceReadback) -> bool {
    equal_f32_bits(&left.attention, &right.attention)
        && equal_f32_bits(&left.attention_residual, &right.attention_residual)
        && equal_f32_bits(&left.post_norm, &right.post_norm)
        && equal_f32_bits(&left.ffn_output, &right.ffn_output)
        && equal_f32_bits(&left.final_hidden, &right.final_hidden)
        && equal_f32_bits(&left.recurrent, &right.recurrent)
        && equal_f32_bits(&left.state, &right.state)
        && equal_f32_bits(&left.history, &right.history)
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
    let finite_count = actual.iter().filter(|value| value.is_finite()).count();
    for (actual, expected) in actual.iter().zip(expected) {
        let absolute = (*actual - *expected).abs();
        max_abs_diff = max_abs_diff.max(absolute);
        max_rel_diff = max_rel_diff.max(absolute / expected.abs().max(1e-6));
    }
    NumericalReport {
        name: name.to_owned(),
        elements: actual.len(),
        finite_count,
        max_abs_diff,
        max_rel_diff,
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

fn equal_f32_bits(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.to_bits() == right.to_bits())
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

fn sha256_f32_bits(values: &[f32]) -> String {
    sha256_hex(&f32_bytes(values))
}

fn allocation_delta(
    earlier: augur_backend::CudaDeviceAllocationSnapshot,
    later: augur_backend::CudaDeviceAllocationSnapshot,
) -> Result<AllocationReport, Qwen35RealLinearLayerCudaError> {
    Ok(AllocationReport {
        successful_allocations: later
            .successful_allocations
            .checked_sub(earlier.successful_allocations)
            .ok_or(Qwen35RealLinearLayerCudaError::InvalidConfiguration(
                "allocation counters",
            ))?,
        successful_frees: later
            .successful_frees
            .checked_sub(earlier.successful_frees)
            .ok_or(Qwen35RealLinearLayerCudaError::InvalidConfiguration(
                "allocation counters",
            ))?,
        live_allocations: later.live_allocations,
        live_bytes: later.live_bytes,
        peak_live_allocations: later.peak_live_allocations,
        peak_live_bytes: later.peak_live_bytes,
    })
}

fn boundary_delta(
    earlier: augur_backend::CudaHostBoundarySnapshot,
    later: augur_backend::CudaHostBoundarySnapshot,
) -> Result<BoundaryReport, Qwen35RealLinearLayerCudaError> {
    let delta = later.checked_delta_since(earlier).ok_or(
        Qwen35RealLinearLayerCudaError::InvalidConfiguration("host counters"),
    )?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_extent_is_exact_and_checked() {
        assert_eq!(checked_f32_extent(0).unwrap(), 0);
        assert_eq!(checked_f32_extent(7).unwrap(), 28);
        assert!(checked_f32_extent(usize::MAX).is_err());
    }

    #[test]
    fn nvfp4_extent_matches_block_geometry_and_rejects_overflow() {
        assert_eq!(
            checked_nvfp4_weight_extent(2, NVFP4_BLOCK_ELEMENTS * 2).unwrap(),
            2 * 2 * NVFP4_BLOCK_BYTES
        );
        assert!(checked_nvfp4_weight_extent(2, NVFP4_BLOCK_ELEMENTS + 1).is_err());
        assert!(checked_nvfp4_weight_extent(0, NVFP4_BLOCK_ELEMENTS).is_err());
        assert!(checked_nvfp4_weight_extent(usize::MAX, NVFP4_BLOCK_ELEMENTS).is_err());
    }

    #[test]
    fn q8_extent_matches_block_geometry_and_rejects_non_multiple_columns() {
        assert_eq!(
            checked_q8_weight_extent(2, Q8_0_BLOCK_ELEMENTS * 2).unwrap(),
            2 * 2 * Q8_0_BLOCK_BYTES
        );
        assert!(checked_q8_weight_extent(2, Q8_0_BLOCK_ELEMENTS + 1).is_err());
        assert!(checked_q8_weight_extent(0, Q8_0_BLOCK_ELEMENTS).is_err());
        assert!(checked_q8_weight_extent(usize::MAX, Q8_0_BLOCK_ELEMENTS).is_err());
    }

    #[test]
    fn positive_scale_requires_one_finite_positive_value() {
        assert_eq!(positive_scale(&[1.25], "scale").unwrap(), 1.25);
        assert!(positive_scale(&[], "scale").is_err());
        assert!(positive_scale(&[0.0], "scale").is_err());
        assert!(positive_scale(&[-1.0], "scale").is_err());
        assert!(positive_scale(&[f32::NAN], "scale").is_err());
        assert!(positive_scale(&[f32::INFINITY], "scale").is_err());
        assert!(positive_scale(&[1.0, 2.0], "scale").is_err());
    }

    #[test]
    fn exact_extent_rejects_short_and_long_buffers() {
        assert!(require_exact_extent(36, 36, "extent").is_ok());
        assert!(require_exact_extent(35, 36, "extent").is_err());
        assert!(require_exact_extent(37, 36, "extent").is_err());
    }

    #[test]
    fn layer_tensor_names_are_scoped_without_rewriting_suffixes() {
        assert_eq!(layer_tensor_name(0, ATTN_NORM), "blk.0.attn_norm.weight");
        assert_eq!(layer_tensor_name(2, FFN_DOWN), "blk.2.ffn_down.weight");
        assert_eq!(layer_tensor_name(63, SSM_A), "blk.63.ssm_a");
    }

    #[test]
    fn default_linear_component_targets_layer_zero() {
        assert_eq!(Qwen35RealLinearLayerCudaConfig::default().layer, 0);
    }
}
