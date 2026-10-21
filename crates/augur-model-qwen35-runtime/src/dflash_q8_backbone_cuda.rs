//! Resident five-layer Q8 DFlash2 block. All arithmetic is GPU-side;
//! optional terminal diagnostics are explicitly outside the submit boundary.
//! This owner is not yet wired to target verification or publication.

use crate::dflash_block_producer::validate_dflash_block_input;
use crate::resident_growing_history::{GrowingHistory, GrowingHistoryPlan};
use crate::{
    DFlashBlockProducerInput, DFlashDraftBlockInput, DFlashQ8Backbone,
    DFlashTargetFeatureCaptureReceipt, DFLASH_ROPE_THETA,
};
use augur_backend::{
    CudaAddF32ArgsV1, CudaBufferView, CudaDflashBlockAttentionF32ArgsV1,
    CudaDflashBlockAttentionViewsV1, CudaDflashGroupedConvHistoryF32ArgsV1,
    CudaDflashRopeF32ArgsV1, CudaF32View, CudaQ8_0F32MatvecArgsV1, CudaQ8_0WeightViewV1,
    CudaRmsnormF32ArgsV1, CudaRuntimeError, CudaSiluMulF32ArgsV1, OwnedCudaBuffer,
    OwnedCudaGraphExec, OwnedCudaStream,
};
use augur_core::PhysicalLayoutId;
use augur_model_qwen35::DFlashConfig;
use std::{collections::BTreeMap, rc::Rc};
#[cfg(any(
    feature = "qwen35-augur-dflash-bridge",
    feature = "qwen35-nerkyor-dflash-research"
))]
pub(crate) mod augur_owner;
pub(crate) mod capture_fc;
mod selector_gate;

#[cfg(feature = "qwen35-output-head-acceptance")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DraftFeatureIdentity {
    PayloadSha256([u8; 32]),
    #[cfg(any(
        feature = "qwen35-augur-dflash-bridge",
        feature = "qwen35-nerkyor-dflash-research"
    ))]
    OwnerLease(crate::checkpoint_target_cuda::dflash_capture::OwnerFeatureIdentity),
}
#[cfg(feature = "qwen35-output-head-acceptance")]
impl DraftFeatureIdentity {
    fn payload_sha256(self) -> Result<[u8; 32]> {
        match self {
            Self::PayloadSha256(sha) => Ok(sha),
            #[cfg(any(
                feature = "qwen35-augur-dflash-bridge",
                feature = "qwen35-nerkyor-dflash-research"
            ))]
            Self::OwnerLease(_) => Err("owner lease is not a feature-payload SHA256".into()),
        }
    }
}
enum InitialNoise<'a> {
    Host(&'a [f32]),
    #[cfg(any(
        feature = "qwen35-augur-dflash-bridge",
        feature = "qwen35-nerkyor-dflash-research"
    ))]
    Device(CudaBufferView),
}

#[cfg(feature = "qwen35-draft-private-state")]
pub(crate) mod private_state;

const H: usize = 5120;
const F: usize = 17408;
const Q: usize = 32 * 128;
const KV: usize = 8 * 128;
const B: usize = 8;
const DYN: usize = 1280;
type Result<T> = std::result::Result<T, String>;
fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn context_reuse_prefix(
    prior: &DFlashTargetFeatureCaptureReceipt,
    next: &DFlashTargetFeatureCaptureReceipt,
    committed: usize,
) -> Result<usize> {
    if !(1..=B).contains(&committed)
        || !(1..=2048).contains(&prior.context_len)
        || prior.context_len != next.context_len
        || prior.context_start_position.checked_add(committed) != Some(next.context_start_position)
        || prior.request_generation.checked_add(1) != Some(next.request_generation)
        || prior.source_target_model_sha256 != next.source_target_model_sha256
        || prior.source_target_model_sha256 == [0; 32]
        || prior.target_layer_ids != next.target_layer_ids
        || prior.feature_width != next.feature_width
    {
        return Err("committed context reuse identity/position/generation drift".into());
    }
    Ok(prior.context_len.saturating_sub(committed))
}

fn validate_growing_rebind(
    prior: &DFlashTargetFeatureCaptureReceipt,
    next: &DFlashTargetFeatureCaptureReceipt,
    plan: GrowingHistoryPlan,
    history: GrowingHistory,
    physical_capacity: usize,
    committed: usize,
) -> Result<()> {
    let prior_end = prior
        .context_start_position
        .checked_add(prior.context_len)
        .ok_or("prior context position overflow")?;
    let next_end = next
        .context_start_position
        .checked_add(next.context_len)
        .ok_or("next context position overflow")?;
    let expected_next_end = prior_end
        .checked_add(committed)
        .ok_or("committed context position overflow")?;
    // A session suffix may contain several B8 blocks plus a scalar tail. The
    // aggregate rebind still copies only the final retained/new rows, so its
    // admission bound is the physical ring rather than one draft block.
    if !(1..=physical_capacity).contains(&committed)
        || plan.append_rows != committed
        || plan.old_start_position != prior.context_start_position
        || plan.old_active_rows != prior.context_len
        || plan.new_start_position != next.context_start_position
        || plan.new_active_rows != next.context_len
        || plan.retained_rows.checked_add(plan.append_rows) != Some(plan.new_active_rows)
        || plan.evicted_rows.checked_add(plan.retained_rows) != Some(plan.old_active_rows)
        || plan.shift_rows != plan.evicted_rows
        || plan.append_offset != plan.retained_rows
        || plan.old_active_rows == 0
        || plan.old_active_rows > physical_capacity
        || plan.old_start_position.checked_add(plan.evicted_rows) != Some(plan.new_start_position)
        || plan
            .old_active_rows
            .checked_add(committed)
            .map(|rows| rows.min(physical_capacity))
            != Some(plan.new_active_rows)
        || plan.new_active_rows > physical_capacity
        || history.capacity_rows() != physical_capacity
        || history.start_position() != plan.new_start_position
        || history.active_rows() != plan.new_active_rows
        || history.cursor() != next_end
        || history.generation() != plan.generation_after
        || plan.generation_after <= plan.generation_before
        || next_end != expected_next_end
        || next.request_generation <= prior.request_generation
        || prior.source_target_model_sha256 != next.source_target_model_sha256
        || prior.source_target_model_sha256 == [0; 32]
        || prior.target_layer_ids != next.target_layer_ids
        || prior.feature_width != next.feature_width
    {
        return Err("growing context reuse identity/plan mismatch".into());
    }
    Ok(())
}

fn validate_growing_refresh_phase(
    graph_owned: bool,
    prepared: bool,
    submitted: bool,
    poisoned: bool,
    context_first_row: usize,
    context_rows: usize,
) -> Result<()> {
    // A completed block and an unprepared incremental rebind are both owned
    // same-stream boundaries. Full refresh discards ALL derived projections;
    // it must not require the previous rebind to have projected its dirty tail.
    // Half-prepared/submitted, graph-bound and poisoned owners still reject.
    if graph_owned
        || poisoned
        || prepared != submitted
        || context_rows == 0
        || context_rows > 2048
        || context_first_row > context_rows
    {
        return Err("growing full-refresh owner is not reusable".into());
    }
    Ok(())
}

fn growing_projection_start(
    prepared: bool,
    submitted: bool,
    prior_first: usize,
    old_rows: usize,
    evicted: usize,
    retained: usize,
) -> Result<usize> {
    if prepared != submitted
        || old_rows == 0
        || old_rows > 2048
        || prior_first > old_rows
        || evicted.checked_add(retained) != Some(old_rows)
    {
        return Err("growing projection reuse extent/phase mismatch".into());
    }
    // An unprepared chained rebind has a dirty suffix. Its features survived
    // the prior copy, but its K/V projections did not run yet. Preserve that
    // invalid extent when shifting, rather than treating it as cached K/V.
    let valid_prefix = if prepared { old_rows } else { prior_first };
    Ok(valid_prefix.saturating_sub(evicted).min(retained))
}

fn validate_growing_refresh(
    prior: &DFlashTargetFeatureCaptureReceipt,
    next: &DFlashTargetFeatureCaptureReceipt,
    history: GrowingHistory,
    physical_capacity: usize,
) -> Result<()> {
    let prior_end = prior
        .context_start_position
        .checked_add(prior.context_len)
        .ok_or("prior refresh extent overflow")?;
    let next_end = next
        .context_start_position
        .checked_add(next.context_len)
        .ok_or("next refresh extent overflow")?;
    if history.capacity_rows() != physical_capacity
        || physical_capacity != 2048
        || prior.context_len == 0
        || prior.context_len > physical_capacity
        || next.context_len != history.active_rows()
        || next.context_start_position != history.start_position()
        || next_end != history.cursor()
        || next_end <= prior_end
        || next.request_generation <= prior.request_generation
        || prior.source_target_model_sha256 == [0; 32]
        || prior.source_target_model_sha256 != next.source_target_model_sha256
        || prior.target_layer_ids != next.target_layer_ids
        || prior.target_hidden_size != next.target_hidden_size
        || prior.feature_width != next.feature_width
    {
        return Err("growing session full-refresh identity/extent mismatch".into());
    }
    Ok(())
}
#[cfg(feature = "qwen35-output-head-acceptance")]
mod candidates;
#[cfg(feature = "qwen35-output-head-acceptance")]
pub use candidates::DFlashQ8CudaCandidateDiagnostics;

/// Immutable, upload-complete Q8 weights. Clones retain the same allocations;
/// request streams, feature/KV histories and workspaces are never shared.
/// Intentionally thread-local: this is not a concurrent request scheduler.
#[derive(Clone)]
pub struct DFlashQ8CudaWeights {
    inner: Rc<DFlashQ8CudaWeightPayload>,
}

struct DFlashQ8CudaWeightPayload {
    owners: BTreeMap<String, OwnedCudaBuffer>,
    config: DFlashConfig,
    device: i32,
    file_sha256: String,
    file_bytes: u64,
    selector_gate_f32_sha256: Option<String>,
    selector_gate_fp64: bool,
    selector_codebooks_bf16_sha256: Option<String>,
}

impl DFlashQ8CudaWeights {
    pub fn upload(weights: &DFlashQ8Backbone, device: i32) -> Result<Self> {
        let file_sha256 =
            weights.manifest().file_sha256.clone().ok_or_else(|| {
                "resident Q8 weights require authenticated source bytes".to_owned()
            })?;
        let stream = OwnedCudaStream::create(device).map_err(err)?;
        let mut owners = BTreeMap::new();
        for (name, _, bytes) in weights.tensors() {
            if name == "candidate_selector.hidden_projection.weight"
                && weights.embedded_selector_gate().is_some()
            {
                continue;
            }
            let owner = OwnedCudaBuffer::allocate(device, bytes.len()).map_err(err)?;
            owner.copy_from_host_sync(&stream, bytes).map_err(err)?;
            owners.insert(name.to_owned(), owner);
        }
        let (selector_gate_f32_sha256, selector_gate_fp64) = if let Some((bytes, sha, fp64)) =
            selector_gate::load(weights.embedded_selector_gate())?
        {
            let owner = OwnedCudaBuffer::allocate(device, bytes.len()).map_err(err)?;
            owner.copy_from_host_sync(&stream, &bytes).map_err(err)?;
            owners.insert(selector_gate::KEY.to_owned(), owner);
            (Some(sha), fp64)
        } else {
            (None, false)
        };
        let selector_codebooks_bf16_sha256 = if let Some((bytes, sha)) =
            selector_gate::load_codebooks(selector_gate_f32_sha256.is_some())?
        {
            let owner = OwnedCudaBuffer::allocate(device, bytes.len()).map_err(err)?;
            owner.copy_from_host_sync(&stream, &bytes).map_err(err)?;
            owners.insert(selector_gate::CODEBOOK_KEY.to_owned(), owner);
            Some(sha)
        } else {
            None
        };
        // Publication only after upload completion. Every request creates a
        // fresh stream and can then read these immutable allocations safely.
        stream.synchronize().map_err(err)?;
        Ok(Self {
            inner: Rc::new(DFlashQ8CudaWeightPayload {
                owners,
                config: weights.config().clone(),
                device,
                file_sha256,
                file_bytes: weights.manifest().file_size,
                selector_gate_f32_sha256,
                selector_gate_fp64,
                selector_codebooks_bf16_sha256,
            }),
        })
    }

    pub(crate) fn embedded_selector_gate(&self) -> bool {
        self.get("candidate_selector.hidden_projection.weight")
            .is_none()
    }
    pub fn config(&self) -> &DFlashConfig {
        &self.inner.config
    }
    pub fn device_ordinal(&self) -> i32 {
        self.inner.device
    }
    pub fn file_sha256(&self) -> &str {
        &self.inner.file_sha256
    }
    pub fn file_bytes(&self) -> u64 {
        self.inner.file_bytes
    }
    pub fn selector_gate_f32_sha256(&self) -> Option<&str> {
        self.inner.selector_gate_f32_sha256.as_deref()
    }
    pub fn selector_codebooks_bf16_sha256(&self) -> Option<&str> {
        self.inner.selector_codebooks_bf16_sha256.as_deref()
    }
    pub fn selector_gate_precision(&self) -> &'static str {
        if self.inner.selector_gate_fp64 {
            "f32_weights_f64_accum"
        } else if self.inner.selector_gate_f32_sha256.is_some() {
            "f32_cublas_pedantic"
        } else {
            "q8_0"
        }
    }
    pub fn resident_bytes(&self) -> usize {
        self.inner.owners.values().map(|w| w.len_bytes()).sum()
    }

    // No public mutable owner or writable device view is exposed.
    fn get(&self, name: &str) -> Option<&OwnedCudaBuffer> {
        self.inner.owners.get(name)
    }
}

pub struct DFlashQ8CudaBlock {
    // The graph must be destroyed before the device buffers it retains.
    // Keep this field before `weights`/`buffers` so Rust drop order preserves
    // that ownership rule.  It is opt-in and remains outside the product
    // submission path until a complete semantic/rolling A/B qualifies it.
    graph: Option<OwnedCudaGraphExec>,
    #[cfg(feature = "qwen35-prefill-cublas")]
    llama_projection_plan: Option<augur_backend::CudaQwen35PrefillCublasPlanV1>,
    /// Per-round B8 block projections use the Augur direct Q8_0 kernel
    /// (`AUGUR_DFLASH_Q8_KERNEL=direct`); context rows keep llama dispatch.
    q8_b8_direct: bool,
    stream: OwnedCudaStream,
    config: DFlashConfig,
    weights: DFlashQ8CudaWeights,
    buffers: BTreeMap<String, OwnedCudaBuffer>,
    // Transient weight-decode scratch, not a checkpointed semantic plane.
    history_weight_panel: Option<OwnedCudaBuffer>,
    history_panel_token8: bool,
    history_panel_submissions: std::cell::Cell<u64>,
    history_panel_kernel_launches: std::cell::Cell<u64>,
    capture_fc_ready: bool,
    capture_fc_report: Option<serde_json::Value>,
    /// Fixed allocation extent. `context` below is the active logical extent;
    /// the two are intentionally separate for the opt-in growing-history path.
    physical_capacity: usize,
    context: usize,
    start: usize,
    window: usize,
    prepared: bool,
    submitted: bool,
    poisoned: bool,
    cached_capture: Option<DFlashTargetFeatureCaptureReceipt>,
    context_first_row: usize,
    feature_shift_submissions: u64,
    context_rows_launch: bool,
    block_rows_launch: bool,
    q8_tile4: bool,
    q8_b8_specialized: bool,
    q8_b8_specialized_submissions: std::cell::Cell<u64>,
    history_projection_reuse: bool,
    context_key_norm_rope: bool,
    selector_gate_enabled: bool,
    selector_phase_gate: bool,
    history_projection_reuse_submissions: std::cell::Cell<u64>,
    context_key_norm_rope_submissions: std::cell::Cell<u64>,
    #[cfg(feature = "qwen35-output-head-acceptance")]
    anchor_token: u32,
    #[cfg(feature = "qwen35-output-head-acceptance")]
    request_generation: u64,
    #[cfg(feature = "qwen35-output-head-acceptance")]
    feature_identity: DraftFeatureIdentity,
    #[cfg(feature = "qwen35-output-head-acceptance")]
    candidates: Option<candidates::ResidentCandidates>,
}

#[derive(Debug)]
pub struct DFlashQ8CudaDiagnosticOutput {
    pub hidden: Vec<f32>,
    pub selector_hidden: Vec<f32>,
    pub layer_hidden: Vec<Vec<f32>>,
    pub d2h_bytes: usize,
}

impl DFlashQ8CudaBlock {
    pub(crate) fn select_resident_gate(&mut self, enabled: bool, phase: bool) -> Result<()> {
        if self.poisoned
            || self.submitted
            || self.graph.is_some()
            || (enabled && self.weights.selector_gate_f32_sha256().is_none())
        {
            return Err(
                "selector gate switch requires an idle uncaptured draft and resident F32 weights"
                    .into(),
            );
        }
        if self.weights.embedded_selector_gate() && (!enabled || phase) {
            return Err(
                "embedded gate requires global FP32 scope; no Q8 fallback gate is present".into(),
            );
        }
        self.selector_gate_enabled = enabled;
        self.selector_phase_gate = enabled && phase;
        Ok(())
    }

    #[cfg(feature = "qwen35-session-checkpoint")]
    pub(crate) fn checkpoint_launch_policy(&self) -> Result<(bool, bool, bool)> {
        if self.poisoned || self.graph.is_some() || self.llama_projection_workspace_bytes() != 0 {
            return Err(
                "checkpoint cannot capture poisoned, graph-owned or experimental llama draft state"
                    .into(),
            );
        }
        Ok((
            self.context_rows_launch,
            self.block_rows_launch,
            self.q8_tile4,
        ))
    }

    #[cfg(feature = "qwen35-raw-committed-state")]
    pub(crate) fn raw_state_inventory(&self) -> Result<serde_json::Value> {
        if self.graph.is_some() || self.candidates.is_some() || self.poisoned {
            return Err("raw inventory rejects graph/candidate owners or poison".into());
        }
        self.stream.synchronize().map_err(err)?;
        let buffers=self.buffers.iter().map(|(name,buffer)| {
            let valid_rows=if name=="features" {Some(self.context)}
                else if name.starts_with('k') || name.starts_with('v') {
                    if name.len()==2 && name.as_bytes()[1].is_ascii_digit() {
                        Some(if self.submitted {self.context+B} else if self.prepared {self.context} else {self.context_first_row})
                    } else {None}
                } else {None};
            serde_json::json!({"name":name,"physical_bytes":buffer.len_bytes(),"valid_rows":valid_rows,
                "category":if name=="features" {"source_features"} else if valid_rows.is_some(){"draft_context_kv"}
                    else if name=="hidden0" || name=="zero" {"initialized_seed_or_constant"} else {"scratch_write_history_required"},
                "raw_bytes_captured":false})
        }).collect::<Vec<_>>();
        if buffers.len() != 40 {
            return Err("raw draft inventory count drift".into());
        }
        Ok(
            serde_json::json!({"schema":"qwen35-draft-private-state-phase-inventory-v1",
            "context":self.context,"start":self.start,"window":self.window,
            "prepared":self.prepared,"submitted":self.submitted,"poisoned":self.poisoned,
            "context_first_row":self.context_first_row,"cached_capture":self.cached_capture,
            "anchor_token":self.anchor_token,"request_generation":self.request_generation,
            "context_rows_launch":self.context_rows_launch,"block_rows_launch":self.block_rows_launch,
            "q8_tile4":self.q8_tile4,"buffers":buffers,"workspace_bytes":self.workspace_bytes(),
            "graph_present":false,"resident_candidates_present":false,"draft_equivalence_qualified":false}),
        )
    }

    /// Weights, inputs and every scratch allocation are owned before execution.
    /// Input identity is checked against the same source-seed/capture contract
    /// as the CPU producer. No model weights are uploaded per draft block.
    pub fn upload(
        weights: &DFlashQ8Backbone,
        input: &DFlashBlockProducerInput<'_>,
        device: i32,
    ) -> Result<Self> {
        validate_dflash_block_input(weights.config(), input).map_err(err)?;
        let context = input.target_features.len() / (5 * H);
        if !(1..=2048).contains(&context) || !(1..=2048).contains(&input.sliding_window) {
            return Err("CUDA block requires 1..2048 context rows/window".into());
        }
        let resident = DFlashQ8CudaWeights::upload(weights, device)?;
        Self::from_resident_weights(&resident, input)
    }

    /// New request-local state over the immutable resident weight allocation.
    pub fn from_resident_weights(
        weights: &DFlashQ8CudaWeights,
        input: &DFlashBlockProducerInput<'_>,
    ) -> Result<Self> {
        validate_dflash_block_input(weights.config(), input).map_err(err)?;
        let context = input.target_features.len() / (5 * H);
        if !(1..=2048).contains(&context) || !(1..=2048).contains(&input.sliding_window) {
            return Err("CUDA block requires 1..2048 context rows/window".into());
        }
        let device = weights.device_ordinal();
        let stream = OwnedCudaStream::create(device).map_err(err)?;
        let mut buffers = BTreeMap::new();
        for (name, count) in [
            ("features", context * 5 * H),
            // Rolling committed-window rebind shifts retained feature rows
            // inside the resident owner and only copies the newly committed
            // suffix from the target producer.  Keep one bounded chunk
            // scratch owner so that the shift never relies on overlapping
            // cudaMemcpy semantics and never materializes another context
            // sized feature ring.
            ("feature_shift_scratch", B * 5 * H),
            ("target_fc", context * H),
            ("target", context * H),
            ("kv_raw", (context + B) * KV),
            ("kv_norm", (context + B) * KV),
            ("norm", B * H),
            ("dynamic", B * DYN),
            ("conv", B * H),
            ("q_raw", B * Q),
            ("q_norm", B * Q),
            ("q_rope", B * Q),
            ("attention", B * Q),
            ("projected", B * H),
            ("conv_out", B * H),
            ("ffn_in", B * H),
            ("gate", B * F),
            ("up", B * F),
            ("activation", B * F),
            ("final", B * H),
            ("selector", B * 256),
            ("zero", H),
        ] {
            buffers.insert(
                name.to_owned(),
                OwnedCudaBuffer::allocate(device, count * 4).map_err(err)?,
            );
        }
        for layer in 0..=5 {
            buffers.insert(
                format!("hidden{layer}"),
                OwnedCudaBuffer::allocate(device, B * H * 4).map_err(err)?,
            );
            if layer < 5 {
                for kind in ['k', 'v'] {
                    buffers.insert(
                        format!("{kind}{layer}"),
                        OwnedCudaBuffer::allocate(device, (context + B) * KV * 4).map_err(err)?,
                    );
                }
            }
        }
        for (name, input) in [
            ("features", input.target_features),
            ("hidden0", input.noise_embedding),
        ] {
            let bytes = input
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>();
            buffers[name]
                .copy_from_host_sync(&stream, &bytes)
                .map_err(err)?;
        }
        buffers["zero"]
            .copy_from_host_sync(&stream, &vec![0; H * 4])
            .map_err(err)?;
        #[cfg(feature = "qwen35-draft-target-verify")]
        for (name, bytes) in [("audit_parts", 17 * 32), ("audit_digest", 32)] {
            buffers.insert(
                name.into(),
                OwnedCudaBuffer::allocate(device, bytes).map_err(err)?,
            );
        }
        Ok(Self {
            stream,
            config: weights.config().clone(),
            graph: None,
            #[cfg(feature = "qwen35-prefill-cublas")]
            llama_projection_plan: None,
            q8_b8_direct: false,
            weights: weights.clone(),
            buffers,
            physical_capacity: context,
            history_weight_panel: None,
            capture_fc_ready: false,
            capture_fc_report: None,
            history_panel_token8: false,
            history_panel_submissions: std::cell::Cell::new(0),
            history_panel_kernel_launches: std::cell::Cell::new(0),
            context,
            start: input.context_start_position,
            window: input.sliding_window,
            prepared: false,
            submitted: false,
            poisoned: false,
            cached_capture: None,
            context_first_row: 0,
            feature_shift_submissions: 0,
            context_rows_launch: false,
            block_rows_launch: false,
            q8_tile4: false,
            q8_b8_specialized: false,
            q8_b8_specialized_submissions: std::cell::Cell::new(0),
            history_projection_reuse: false,
            context_key_norm_rope: false,
            selector_gate_enabled: weights.selector_gate_f32_sha256().is_some(),
            selector_phase_gate: false,
            history_projection_reuse_submissions: std::cell::Cell::new(0),
            context_key_norm_rope_submissions: std::cell::Cell::new(0),
            #[cfg(feature = "qwen35-output-head-acceptance")]
            anchor_token: input.seed.block_tokens[0],
            #[cfg(feature = "qwen35-output-head-acceptance")]
            request_generation: input.seed.request_generation,
            #[cfg(feature = "qwen35-output-head-acceptance")]
            feature_identity: DraftFeatureIdentity::PayloadSha256(input.seed.target_feature_sha256),
            #[cfg(feature = "qwen35-output-head-acceptance")]
            candidates: None,
        })
    }

    /// Upload a block whose target feature rows already live on a CUDA
    /// device.  This is the product-oriented handoff: the target producer
    /// publishes a bounded feature receipt, and the DFlash owner copies the
    /// rows device-to-device without materializing a 200 MiB host buffer.
    ///
    /// `source_stream` must be quiescent before this call.  The current
    /// producer invokes it after the target feature capture has completed;
    /// a future asynchronous producer should replace that precondition with
    /// an explicit cross-stream event dependency.
    pub fn upload_device_features(
        weights: &DFlashQ8Backbone,
        seed: &DFlashDraftBlockInput,
        target_capture: &DFlashTargetFeatureCaptureReceipt,
        target_features: CudaBufferView,
        noise_embedding: &[f32],
        context_start_position: usize,
        sliding_window: usize,
        device: i32,
        source_stream: &OwnedCudaStream,
    ) -> Result<Self> {
        validate_device_handoff(
            weights.config(),
            seed,
            target_capture,
            target_features,
            noise_embedding,
            context_start_position,
            sliding_window,
            device,
            source_stream.device_ordinal(),
        )?;
        let resident = DFlashQ8CudaWeights::upload(weights, device)?;
        Self::from_resident_device_features(
            &resident,
            seed,
            target_capture,
            target_features,
            noise_embedding,
            context_start_position,
            sliding_window,
            source_stream,
        )
    }

    /// Request-local device-feature handoff with no weight upload. The block
    /// retains its own weight lease until its stream and optional graph drain.
    pub fn from_resident_device_features(
        weights: &DFlashQ8CudaWeights,
        seed: &DFlashDraftBlockInput,
        target_capture: &DFlashTargetFeatureCaptureReceipt,
        target_features: CudaBufferView,
        noise_embedding: &[f32],
        context_start_position: usize,
        sliding_window: usize,
        source_stream: &OwnedCudaStream,
    ) -> Result<Self> {
        let context = target_capture.context_len;
        Self::from_resident_device_features_with_capacity(
            weights,
            seed,
            target_capture,
            target_features,
            noise_embedding,
            context_start_position,
            sliding_window,
            context,
            source_stream,
            None,
        )
    }

    pub(crate) fn from_resident_growing_device_features(
        weights: &DFlashQ8CudaWeights,
        seed: &DFlashDraftBlockInput,
        target_capture: &DFlashTargetFeatureCaptureReceipt,
        target_features: CudaBufferView,
        noise_embedding: &[f32],
        history: GrowingHistory,
        sliding_window: usize,
        source_stream: &OwnedCudaStream,
    ) -> Result<Self> {
        if history.active_rows() != target_capture.context_len
            || history.start_position() != target_capture.context_start_position
        {
            return Err("growing history extent differs from capture receipt".into());
        }
        Self::from_resident_device_features_with_capacity(
            weights,
            seed,
            target_capture,
            target_features,
            noise_embedding,
            history.start_position(),
            sliding_window,
            history.capacity_rows(),
            source_stream,
            None,
        )
    }

    pub(crate) fn from_resident_growing_device_features_with_fc(
        weights: &DFlashQ8CudaWeights,
        seed: &DFlashDraftBlockInput,
        capture: &DFlashTargetFeatureCaptureReceipt,
        features: CudaBufferView,
        noise: &[f32],
        history: GrowingHistory,
        window: usize,
        source: &OwnedCudaStream,
        fc: capture_fc::PreparedFc,
    ) -> Result<Self> {
        if history.active_rows() != capture.context_len
            || history.start_position() != capture.context_start_position
        {
            return Err("capture FC growing history extent".into());
        }
        Self::from_resident_device_features_with_capacity(
            weights,
            seed,
            capture,
            features,
            noise,
            history.start_position(),
            window,
            history.capacity_rows(),
            source,
            Some(fc),
        )
    }

    fn from_resident_device_features_with_capacity(
        weights: &DFlashQ8CudaWeights,
        seed: &DFlashDraftBlockInput,
        target_capture: &DFlashTargetFeatureCaptureReceipt,
        target_features: CudaBufferView,
        noise_embedding: &[f32],
        context_start_position: usize,
        sliding_window: usize,
        physical_capacity: usize,
        source_stream: &OwnedCudaStream,
        prefetched_fc: Option<capture_fc::PreparedFc>,
    ) -> Result<Self> {
        let context = validate_device_handoff(
            weights.config(),
            seed,
            target_capture,
            target_features,
            noise_embedding,
            context_start_position,
            sliding_window,
            weights.device_ordinal(),
            source_stream.device_ordinal(),
        )?;
        if physical_capacity < context || !(1..=2048).contains(&physical_capacity) {
            return Err("device feature physical capacity must contain active rows".into());
        }
        source_stream.synchronize().map_err(err)?;

        if let Some(fc) = &prefetched_fc {
            fc.validate(weights, target_capture, source_stream)?;
            if physical_capacity != context {
                return Err("capture FC physical extent differs".into());
            }
        }
        Self::allocate_device_features(
            weights,
            target_features,
            InitialNoise::Host(noise_embedding),
            context,
            context_start_position,
            sliding_window,
            physical_capacity,
            Some(target_capture.clone()),
            seed.block_tokens[0],
            seed.request_generation,
            #[cfg(feature = "qwen35-output-head-acceptance")]
            DraftFeatureIdentity::PayloadSha256(seed.target_feature_sha256),
            prefetched_fc,
        )
    }

    // Only called after the public SHA contract or the sealed owner-lease
    // contract has independently validated the initialized extent and source.
    fn allocate_device_features(
        weights: &DFlashQ8CudaWeights,
        target_features: CudaBufferView,
        noise: InitialNoise<'_>,
        context: usize,
        context_start_position: usize,
        sliding_window: usize,
        physical_capacity: usize,
        cached_capture: Option<DFlashTargetFeatureCaptureReceipt>,
        anchor_token: u32,
        request_generation: u64,
        #[cfg(feature = "qwen35-output-head-acceptance")] feature_identity: DraftFeatureIdentity,
        prefetched_fc: Option<capture_fc::PreparedFc>,
    ) -> Result<Self> {
        let (mut fc_output, fc_panel, capture_fc_report) = match prefetched_fc {
            Some(fc) => (Some(fc.output), Some(fc.panel), Some(fc.report)),
            None => (None, None, None),
        };
        let capture_fc_ready = fc_output.is_some();

        let device = weights.device_ordinal();
        let stream = OwnedCudaStream::create(device).map_err(err)?;
        let mut buffers = BTreeMap::new();
        for (name, count) in [
            ("features", physical_capacity * 5 * H),
            ("feature_shift_scratch", B * 5 * H),
            ("target_fc", physical_capacity * H),
            ("target", physical_capacity * H),
            ("kv_raw", (physical_capacity + B) * KV),
            ("kv_norm", (physical_capacity + B) * KV),
            ("norm", B * H),
            ("dynamic", B * DYN),
            ("conv", B * H),
            ("q_raw", B * Q),
            ("q_norm", B * Q),
            ("q_rope", B * Q),
            ("attention", B * Q),
            ("projected", B * H),
            ("conv_out", B * H),
            ("ffn_in", B * H),
            ("gate", B * F),
            ("up", B * F),
            ("activation", B * F),
            ("final", B * H),
            ("selector", B * 256),
            ("zero", H),
        ] {
            if name == "target_fc" {
                if let Some(output) = fc_output.take() {
                    buffers.insert(name.to_owned(), output);
                    continue;
                }
            }
            buffers.insert(
                name.to_owned(),
                OwnedCudaBuffer::allocate(device, count * 4).map_err(err)?,
            );
        }
        for layer in 0..=5 {
            buffers.insert(
                format!("hidden{layer}"),
                OwnedCudaBuffer::allocate(device, B * H * 4).map_err(err)?,
            );
            if layer < 5 {
                for kind in ['k', 'v'] {
                    buffers.insert(
                        format!("{kind}{layer}"),
                        OwnedCudaBuffer::allocate(device, (physical_capacity + B) * KV * 4)
                            .map_err(err)?,
                    );
                }
            }
        }
        // Drain on every exit, including failure after one borrowed-source copy.
        // The source lease/legacy caller stays alive through this entire call.
        let copy_result = (|| -> Result<()> {
            unsafe {
                buffers["features"]
                    .copy_from_device_async_unchecked(&stream, 0, target_features)
                    .map_err(err)?;
            }
            match noise {
                InitialNoise::Host(values) => {
                    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                    buffers["hidden0"]
                        .copy_from_host_sync(&stream, &bytes)
                        .map_err(err)?;
                }
                #[cfg(any(
                    feature = "qwen35-augur-dflash-bridge",
                    feature = "qwen35-nerkyor-dflash-research"
                ))]
                InitialNoise::Device(view) => unsafe {
                    buffers["hidden0"]
                        .copy_from_device_async_unchecked(&stream, 0, view)
                        .map_err(err)?;
                },
            }
            Ok(())
        })();
        let drain_result = stream.synchronize().map_err(err);
        copy_result?;
        drain_result?;
        buffers["zero"]
            .copy_from_host_sync(&stream, &vec![0; H * 4])
            .map_err(err)?;
        Ok(Self {
            #[cfg(feature = "qwen35-draft-target-verify")]
            buffers: {
                for (name, bytes) in [("audit_parts", 17 * 32), ("audit_digest", 32)] {
                    buffers.insert(
                        name.into(),
                        OwnedCudaBuffer::allocate(device, bytes).map_err(err)?,
                    );
                }
                buffers
            },
            #[cfg(not(feature = "qwen35-draft-target-verify"))]
            buffers,
            stream,
            config: weights.config().clone(),
            graph: None,
            #[cfg(feature = "qwen35-prefill-cublas")]
            llama_projection_plan: None,
            q8_b8_direct: false,
            weights: weights.clone(),
            physical_capacity,
            context,
            start: context_start_position,
            history_weight_panel: fc_panel,
            capture_fc_ready,
            capture_fc_report,
            history_panel_token8: false,
            history_panel_submissions: std::cell::Cell::new(0),
            history_panel_kernel_launches: std::cell::Cell::new(0),
            window: sliding_window,
            prepared: false,
            submitted: false,
            poisoned: false,
            cached_capture,
            context_first_row: 0,
            feature_shift_submissions: 0,
            context_rows_launch: false,
            block_rows_launch: false,
            q8_tile4: false,
            q8_b8_specialized: false,
            q8_b8_specialized_submissions: std::cell::Cell::new(0),
            history_projection_reuse: false,
            context_key_norm_rope: false,
            selector_gate_enabled: weights.selector_gate_f32_sha256().is_some(),
            selector_phase_gate: false,
            history_projection_reuse_submissions: std::cell::Cell::new(0),
            context_key_norm_rope_submissions: std::cell::Cell::new(0),
            #[cfg(feature = "qwen35-output-head-acceptance")]
            anchor_token,
            #[cfg(feature = "qwen35-output-head-acceptance")]
            request_generation,
            #[cfg(feature = "qwen35-output-head-acceptance")]
            feature_identity,
            #[cfg(feature = "qwen35-output-head-acceptance")]
            candidates: None,
        })
    }

    pub fn resident_weight_bytes(&self) -> usize {
        self.weights.resident_bytes()
    }

    pub(crate) fn feature_shift_submissions(&self) -> u64 {
        self.feature_shift_submissions
    }
    pub fn workspace_bytes(&self) -> usize {
        self.buffers.values().map(|w| w.len_bytes()).sum::<usize>()
            + self
                .history_weight_panel
                .as_ref()
                .map_or(0, |w| w.len_bytes())
            + self.llama_projection_workspace_bytes()
    }

    /// Rebind one resident block owner for the next rolling draft.
    ///
    /// Weight owners and scratch buffers stay on device. Only the sliding
    /// target-feature ring and the first noise embedding row are replaced;
    /// callers must have already consumed the previous block's compact
    /// receipt before invoking this method. This is intentionally a bounded
    /// diagnostic seam until the feature ring itself is produced on device.
    pub fn rebind(&mut self, input: &DFlashBlockProducerInput<'_>) -> Result<()> {
        #[cfg(feature = "qwen35-output-head-acceptance")]
        self.feature_identity.payload_sha256()?;
        if self.graph.is_some() {
            return Err("graph-captured owner cannot be rebound".into());
        }
        if !self.submitted || self.poisoned {
            return Err("resident block is not reusable after submission".into());
        }
        validate_dflash_block_input(&self.config, input).map_err(err)?;
        let context = input.target_features.len() / (5 * H);
        if context != self.context || input.sliding_window != self.window {
            return Err("rolling rebind geometry drift".into());
        }
        self.poisoned = true;
        let result = (|| {
            let feature_bytes = input
                .target_features
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>();
            let noise_bytes = input
                .noise_embedding
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>();
            self.buffers["features"]
                .copy_from_host_sync(&self.stream, &feature_bytes)
                .map_err(err)?;
            self.buffers["hidden0"]
                .copy_from_host_sync(&self.stream, &noise_bytes)
                .map_err(err)?;
            self.start = input.context_start_position;
            self.cached_capture = None;
            self.context_first_row = 0;
            self.prepared = false;
            self.submitted = false;
            #[cfg(feature = "qwen35-output-head-acceptance")]
            {
                self.anchor_token = input.seed.block_tokens[0];
                self.request_generation = input.seed.request_generation;
                self.feature_identity =
                    DraftFeatureIdentity::PayloadSha256(input.seed.target_feature_sha256);
            }
            Ok(())
        })();
        if result.is_ok() {
            self.poisoned = false;
        }
        result
    }

    /// Rebind using a target-owner device feature ring. This keeps the full
    /// context off the host transport path; only the small noise row remains a
    /// host upload until device-side seed embedding is wired.
    pub fn rebind_device_features(
        &mut self,
        seed: &DFlashDraftBlockInput,
        target_capture: &DFlashTargetFeatureCaptureReceipt,
        target_features: augur_backend::CudaBufferView,
        context_start_position: usize,
        noise_embedding: &[f32],
        sliding_window: usize,
    ) -> Result<()> {
        #[cfg(feature = "qwen35-output-head-acceptance")]
        self.feature_identity.payload_sha256()?;
        if self.graph.is_some() {
            return Err("graph-captured owner cannot be rebound".into());
        }
        if !self.submitted || self.poisoned {
            return Err("resident block is not reusable after submission".into());
        }
        seed.validate(&self.config).map_err(err)?;
        target_capture.validate(&self.config).map_err(err)?;
        let expected_bytes = self.context * 5 * H * 4;
        if target_features.len_bytes() != expected_bytes
            || target_capture.context_len != self.context
            || target_capture.context_start_position != context_start_position
            || target_capture.feature_sha256 != seed.target_feature_sha256
            || sliding_window != self.window
            || noise_embedding.len() != B * H
            || noise_embedding.iter().any(|v| !v.is_finite())
        {
            return Err(format!(
                "rolling device feature rebind contract mismatch: bytes={} expected={} capture_len={} context={} capture_start={} expected_start={} digest_match={} window={} expected_window={} noise_len={}",
                target_features.len_bytes(),
                expected_bytes,
                target_capture.context_len,
                self.context,
                target_capture.context_start_position,
                context_start_position,
                target_capture.feature_sha256 == seed.target_feature_sha256,
                sliding_window,
                self.window,
                noise_embedding.len(),
            ));
        }
        self.poisoned = true;
        let result = (|| {
            // SAFETY: both owners are same-device, live, non-overlapping
            // allocations retained by the caller until this stream drains.
            unsafe {
                self.buffers["features"]
                    .copy_from_device_async_unchecked(&self.stream, 0, target_features)
                    .map_err(err)?;
            }
            let noise_bytes = noise_embedding
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>();
            self.buffers["hidden0"]
                .copy_from_host_sync(&self.stream, &noise_bytes)
                .map_err(err)?;
            self.start = context_start_position;
            self.cached_capture = None;
            self.context_first_row = 0;
            self.prepared = false;
            self.submitted = false;
            #[cfg(feature = "qwen35-output-head-acceptance")]
            {
                self.anchor_token = seed.block_tokens[0];
                self.request_generation = seed.request_generation;
                self.feature_identity =
                    DraftFeatureIdentity::PayloadSha256(seed.target_feature_sha256);
            }
            Ok(())
        })();
        if result.is_ok() {
            self.poisoned = false;
        }
        result
    }

    /// Device-only variant of rolling rebind. The target-side embedding
    /// gather already produced the `[anchor, mask x 7]` F32 rows; no host
    /// noise upload is performed here.
    pub fn rebind_device_features_with_noise(
        &mut self,
        seed: &DFlashDraftBlockInput,
        target_capture: &DFlashTargetFeatureCaptureReceipt,
        target_features: augur_backend::CudaBufferView,
        noise_embedding: augur_backend::CudaBufferView,
        context_start_position: usize,
        sliding_window: usize,
    ) -> Result<()> {
        #[cfg(feature = "qwen35-output-head-acceptance")]
        self.feature_identity.payload_sha256()?;
        if self.graph.is_some() {
            return Err("graph-captured owner cannot be rebound".into());
        }
        if !self.submitted || self.poisoned {
            return Err("resident block is not reusable after submission".into());
        }
        seed.validate(&self.config).map_err(err)?;
        target_capture
            .validate_for_block(
                &self.config,
                seed,
                target_capture.feature_elements,
                seed.target_feature_sha256,
            )
            .map_err(err)?;
        let expected_feature_bytes = self.context * 5 * H * 4;
        let expected_noise_bytes = B * H * 4;
        if target_features.len_bytes() != expected_feature_bytes
            || noise_embedding.len_bytes() != expected_noise_bytes
            || target_capture.context_len != self.context
            || target_capture.context_start_position != context_start_position
            || target_capture.feature_sha256 != seed.target_feature_sha256
            || sliding_window != self.window
        {
            return Err("rolling device feature/noise rebind contract mismatch".into());
        }
        self.poisoned = true;
        let result = (|| {
            unsafe {
                self.buffers["features"]
                    .copy_from_device_async_unchecked(&self.stream, 0, target_features)
                    .map_err(err)?;
                self.buffers["hidden0"]
                    .copy_from_device_async_unchecked(&self.stream, 0, noise_embedding)
                    .map_err(err)?;
            }
            self.start = context_start_position;
            self.cached_capture = Some(target_capture.clone());
            self.context_first_row = 0;
            self.prepared = false;
            self.submitted = false;
            #[cfg(feature = "qwen35-output-head-acceptance")]
            {
                self.anchor_token = seed.block_tokens[0];
                self.request_generation = seed.request_generation;
                self.feature_identity =
                    DraftFeatureIdentity::PayloadSha256(seed.target_feature_sha256);
            }
            Ok(())
        })();
        if result.is_ok() {
            self.poisoned = false;
        }
        result
    }

    /// Rebuild all derived draft projections after a teacher-forced user suffix.
    /// The target window is authoritative; immutable weights/allocation stay resident.
    /// Unlike block reuse, multiple input rows may have advanced since the last
    /// submitted draft, so no final one-row plan can describe this transition.
    /// An input-only append can leave a partial incremental rebind unprepared;
    /// full refresh overwrites its features and resets the projection extent to
    /// zero on the same stream, without ever consuming the stale derived rows.
    pub(crate) fn refresh_growing_device_features_with_noise(
        &mut self,
        seed: &DFlashDraftBlockInput,
        capture: &DFlashTargetFeatureCaptureReceipt,
        features: CudaBufferView,
        noise: CudaBufferView,
        history: GrowingHistory,
    ) -> Result<()> {
        #[cfg(feature = "qwen35-output-head-acceptance")]
        self.feature_identity.payload_sha256()?;
        validate_growing_refresh_phase(
            self.graph.is_some(),
            self.prepared,
            self.submitted,
            self.poisoned,
            self.context_first_row,
            self.context,
        )?;
        let prior = self
            .cached_capture
            .as_ref()
            .ok_or("missing prior draft capture")?;
        validate_growing_refresh(prior, capture, history, self.physical_capacity)?;
        seed.validate(&self.config).map_err(err)?;
        capture
            .validate_for_block(
                &self.config,
                seed,
                capture.feature_elements,
                seed.target_feature_sha256,
            )
            .map_err(err)?;
        if self.start != prior.context_start_position
            || self.context != prior.context_len
            || features.len_bytes() != history.active_rows() * 5 * H * 4
            || noise.len_bytes() != B * H * 4
            || self.window != 2048
            || features.device_ordinal() != self.weights.device_ordinal()
            || noise.device_ordinal() != self.weights.device_ordinal()
        {
            return Err("growing full-refresh physical handoff mismatch".into());
        }
        self.poisoned = true;
        let result = (|| {
            // SAFETY: both owners remain alive through the draft stream terminal;
            // the caller has inserted the same-owner target handoff event wait.
            unsafe {
                self.buffers["features"]
                    .copy_from_device_async_unchecked(&self.stream, 0, features)
                    .map_err(err)?;
                self.buffers["hidden0"]
                    .copy_from_device_async_unchecked(&self.stream, 0, noise)
                    .map_err(err)?;
            }
            self.start = history.start_position();
            self.context = history.active_rows();
            self.cached_capture = Some(capture.clone());
            self.context_first_row = 0;
            self.prepared = false;
            self.submitted = false;
            #[cfg(feature = "qwen35-output-head-acceptance")]
            {
                self.anchor_token = seed.block_tokens[0];
                self.request_generation = seed.request_generation;
                self.feature_identity =
                    DraftFeatureIdentity::PayloadSha256(seed.target_feature_sha256);
            }
            Ok(())
        })();
        if result.is_ok() {
            self.poisoned = false;
        }
        result
    }

    /// Same-owner committed-window rebind. The driver must have completed
    /// the old block and preserved every retained feature row. No external
    /// capture/CLI file is admitted through this crate-private reuse seam.
    pub(crate) fn rebind_committed_device_features_with_noise(
        &mut self,
        seed: &DFlashDraftBlockInput,
        target_capture: &DFlashTargetFeatureCaptureReceipt,
        target_features: CudaBufferView,
        noise_embedding: CudaBufferView,
        committed: usize,
    ) -> Result<()> {
        #[cfg(feature = "qwen35-output-head-acceptance")]
        self.feature_identity.payload_sha256()?;
        if self.graph.is_some() {
            return Err("graph-captured owner cannot be rebound".into());
        }
        if !self.prepared || !self.submitted || self.poisoned {
            return Err("context projection owner is not reusable".into());
        }
        seed.validate(&self.config).map_err(err)?;
        target_capture
            .validate_for_block(
                &self.config,
                seed,
                target_capture.feature_elements,
                seed.target_feature_sha256,
            )
            .map_err(err)?;
        let prior = self
            .cached_capture
            .as_ref()
            .ok_or("missing cached source identity")?;
        let retained = context_reuse_prefix(prior, target_capture, committed)?;
        if prior.context_start_position != self.start || prior.context_len != self.context {
            return Err("cached projection position drift".into());
        }
        let expected_feature_bytes = self.context * 5 * H * 4;
        let expected_noise_bytes = B * H * 4;
        if target_features.len_bytes() != expected_feature_bytes
            || noise_embedding.len_bytes() != expected_noise_bytes
            || target_capture.context_len != self.context
            || target_capture.feature_sha256 != seed.target_feature_sha256
        {
            return Err("committed device feature/noise rebind contract mismatch".into());
        }
        self.poisoned = true;
        let result = (|| {
            if committed > self.context {
                return Err("committed feature window exceeds resident context".into());
            }

            // The target producer has already advanced its own feature ring
            // before the handoff event observed by this stream.  Preserve the
            // old resident prefix with one bitwise in-place row shift,
            // then import only the newly committed suffix.  The previous
            // implementation copied all `context * 5 * H` feature elements
            // here even though all but `committed` rows were already resident.
            let row_bytes = 5 * H * 4;
            if retained > 0 && committed > 0 {
                // SAFETY: this owner is poisoned during rebind; prior work and
                // handoff are ordered on this stream, buffers outlive its drain.
                unsafe {
                    self.buffers["features"]
                        .shift_rows_u32_async_unchecked(
                            &self.stream,
                            row_bytes,
                            retained,
                            committed,
                        )
                        .map_err(err)?;
                }
                self.feature_shift_submissions += 1;
            }
            if committed > 0 {
                let bytes = committed * row_bytes;
                let source = target_features
                    .subview(retained * row_bytes, bytes)
                    .map_err(err)?;
                // SAFETY: target_features is owned by the producer and the
                // caller has inserted the cross-stream handoff before this
                // method. The destination feature ring is a separate owner.
                unsafe {
                    self.buffers["features"]
                        .copy_from_device_async_unchecked(
                            &self.stream,
                            retained * row_bytes,
                            source,
                        )
                        .map_err(err)?;
                }
            }

            for layer in 0..5 {
                for kind in ['k', 'v'] {
                    let owner = &self.buffers[&format!("{kind}{layer}")];
                    let bytes = retained * KV * 4;
                    // SAFETY: distinct live allocations on this owner stream;
                    // previous work completed before rebind. Existing kv_raw is
                    // transient and will be overwritten by the new row projections.
                    if retained > 0 {
                        unsafe {
                            self.buffers["kv_raw"]
                                .copy_from_device_async_unchecked(
                                    &self.stream,
                                    0,
                                    owner.view(committed * KV * 4, bytes).map_err(err)?,
                                )
                                .map_err(err)?;
                            owner
                                .copy_from_device_async_unchecked(
                                    &self.stream,
                                    0,
                                    self.buffers["kv_raw"].view(0, bytes).map_err(err)?,
                                )
                                .map_err(err)?;
                        }
                    }
                }
            }

            unsafe {
                self.buffers["hidden0"]
                    .copy_from_device_async_unchecked(&self.stream, 0, noise_embedding)
                    .map_err(err)?;
            }
            self.start = target_capture.context_start_position;
            self.cached_capture = Some(target_capture.clone());
            self.context_first_row = retained;
            self.prepared = false;
            self.submitted = false;
            #[cfg(feature = "qwen35-output-head-acceptance")]
            {
                self.anchor_token = seed.block_tokens[0];
                self.request_generation = seed.request_generation;
                self.feature_identity =
                    DraftFeatureIdentity::PayloadSha256(seed.target_feature_sha256);
            }
            Ok(())
        })();
        if result.is_ok() {
            self.poisoned = false;
        }
        result
    }

    pub(crate) fn rebind_growing_committed_device_features_with_noise(
        &mut self,
        seed: &DFlashDraftBlockInput,
        target_capture: &DFlashTargetFeatureCaptureReceipt,
        target_features: CudaBufferView,
        noise_embedding: CudaBufferView,
        plan: GrowingHistoryPlan,
        history: GrowingHistory,
    ) -> Result<()> {
        #[cfg(feature = "qwen35-output-head-acceptance")]
        self.feature_identity.payload_sha256()?;
        // A prior rebind leaves this owner unprepared with its old projected
        // rows still resident.  It is safe to chain another aggregate rebind
        // on the same stream; only a half-submitted block is forbidden.
        if self.graph.is_some() || self.prepared != self.submitted || self.poisoned {
            return Err("growing context owner is not reusable".into());
        }
        let projection_start = growing_projection_start(
            self.prepared,
            self.submitted,
            self.context_first_row,
            self.context,
            plan.evicted_rows,
            plan.retained_rows,
        )?;
        seed.validate(&self.config).map_err(err)?;
        let prior = self
            .cached_capture
            .as_ref()
            .ok_or("missing cached source identity")?;
        validate_growing_rebind(
            prior,
            target_capture,
            plan,
            history,
            self.physical_capacity,
            plan.append_rows,
        )?;
        target_capture
            .validate_for_block(
                &self.config,
                seed,
                target_capture.feature_elements,
                seed.target_feature_sha256,
            )
            .map_err(err)?;
        let expected_feature_bytes = plan
            .new_active_rows
            .checked_mul(5 * H * 4)
            .ok_or("growing feature extent overflow")?;
        if self.start != plan.old_start_position
            || self.context != plan.old_active_rows
            || target_features.len_bytes() != expected_feature_bytes
            || noise_embedding.len_bytes() != B * H * 4
            || target_capture.feature_sha256 != seed.target_feature_sha256
            || target_features.device_ordinal() != self.weights.device_ordinal()
            || noise_embedding.device_ordinal() != self.weights.device_ordinal()
        {
            return Err("growing device feature/noise handoff mismatch".into());
        }

        self.poisoned = true;
        let result = (|| {
            let row_bytes = 5 * H * 4;
            let shift_rows = if plan.evicted_rows == 0 {
                0
            } else {
                plan.retained_rows
            };
            if shift_rows > 0 {
                // SAFETY: rebind phase owns the feature buffer; all accesses
                // are stream ordered and the owner lives through completion.
                unsafe {
                    self.buffers["features"]
                        .shift_rows_u32_async_unchecked(
                            &self.stream,
                            row_bytes,
                            shift_rows,
                            plan.evicted_rows,
                        )
                        .map_err(err)?;
                }
                self.feature_shift_submissions += 1;
            }
            let append_bytes = plan.append_rows * row_bytes;
            if append_bytes > 0 {
                let source = target_features
                    .subview(plan.append_offset * row_bytes, append_bytes)
                    .map_err(err)?;
                unsafe {
                    self.buffers["features"]
                        .copy_from_device_async_unchecked(
                            &self.stream,
                            plan.append_offset * row_bytes,
                            source,
                        )
                        .map_err(err)?;
                }
            }

            for layer in 0..5 {
                for kind in ['k', 'v'] {
                    let owner = &self.buffers[&format!("{kind}{layer}")];
                    let bytes = plan.retained_rows * KV * 4;
                    if bytes > 0 && plan.evicted_rows > 0 {
                        unsafe {
                            self.buffers["kv_raw"]
                                .copy_from_device_async_unchecked(
                                    &self.stream,
                                    0,
                                    owner.view(plan.evicted_rows * KV * 4, bytes).map_err(err)?,
                                )
                                .map_err(err)?;
                            owner
                                .copy_from_device_async_unchecked(
                                    &self.stream,
                                    0,
                                    self.buffers["kv_raw"].view(0, bytes).map_err(err)?,
                                )
                                .map_err(err)?;
                        }
                    }
                }
            }
            unsafe {
                self.buffers["hidden0"]
                    .copy_from_device_async_unchecked(&self.stream, 0, noise_embedding)
                    .map_err(err)?;
            }
            self.start = plan.new_start_position;
            self.context = plan.new_active_rows;
            self.cached_capture = Some(target_capture.clone());
            self.context_first_row = projection_start;
            self.prepared = false;
            self.submitted = false;
            #[cfg(feature = "qwen35-output-head-acceptance")]
            {
                self.anchor_token = seed.block_tokens[0];
                self.request_generation = seed.request_generation;
                self.feature_identity =
                    DraftFeatureIdentity::PayloadSha256(seed.target_feature_sha256);
            }
            Ok(())
        })();
        if result.is_ok() {
            self.poisoned = false;
        }
        result
    }

    pub(crate) fn context_projection_rows(&self) -> usize {
        self.context - self.context_first_row
    }

    #[cfg(feature = "qwen35-draft-target-verify")]
    pub(crate) fn enable_context_rows_launch(&mut self) -> Result<()> {
        if self.prepared || self.submitted || self.poisoned || self.context_first_row != 0 {
            return Err("context launch policy requires unused owner".into());
        }
        self.context_rows_launch = true;
        Ok(())
    }

    /// Independent of context preparation; never changes the cached-history axis.
    pub(crate) fn enable_block_rows_launch(&mut self) -> Result<()> {
        if self.prepared || self.submitted || self.poisoned {
            return Err("block launch policy requires unused owner".into());
        }
        self.block_rows_launch = true;
        Ok(())
    }

    #[cfg(feature = "qwen35-draft-target-verify")]
    pub(crate) fn enable_q8_tile4(&mut self) -> Result<()> {
        if self.prepared
            || self.submitted
            || self.poisoned
            || self.context_first_row != 0
            || !self.context_rows_launch
            || !self.block_rows_launch
        {
            return Err("Q8 tile4 requires unused owner and both row launch policies".into());
        }
        self.q8_tile4 = true;
        Ok(())
    }

    #[cfg(feature = "qwen35-draft-target-verify")]
    pub(crate) fn enable_q8_b8_specialized(&mut self, enabled: bool) -> Result<()> {
        if !enabled {
            return Ok(());
        }
        if self.prepared
            || self.submitted
            || self.poisoned
            || self.graph.is_some()
            || self.context_first_row != 0
            || !self.q8_tile4
            || !self.context_rows_launch
            || !self.block_rows_launch
        {
            return Err("specialized B8 projection requires unused tile4 row owner".into());
        }
        self.q8_b8_specialized = true;
        Ok(())
    }

    pub(crate) fn block_projection_kernel(&self) -> &'static str {
        #[cfg(feature = "qwen35-prefill-cublas")]
        if self.llama_projection_plan.is_some() {
            return if self.q8_b8_direct {
                "augur_q8_b8_direct_f32_v1"
            } else {
                "llama_q8_1_mmvq8_v1"
            };
        }
        "context_rows_block_rows_q8_tile4"
    }
    pub(crate) fn llama_projection_workspace_bytes(&self) -> usize {
        #[cfg(feature = "qwen35-prefill-cublas")]
        if self.llama_projection_plan.is_some() {
            return (H * 5 * H + 512 * 5 * H) * 2;
        }
        0
    }
    pub(crate) fn enable_llama_projection(&mut self) -> Result<()> {
        if self.prepared
            || self.submitted
            || self.poisoned
            || self.graph.is_some()
            || self.context_first_row != 0
            || !self.q8_tile4
            || !self.context_rows_launch
            || !self.block_rows_launch
            || self.history_weight_panel.is_some()
            || self.capture_fc_ready
        {
            return Err("llama projection requires a fresh row-launched draft owner".into());
        }
        #[cfg(feature = "qwen35-prefill-cublas")]
        {
            if self.llama_projection_plan.is_some() {
                return Err("duplicate llama draft plan".into());
            }
            // One serial scratch arena: max FC weight25600x5120, max512 rows.
            // Never materialize all draft matrices, never allocate at submit.
            self.llama_projection_plan = Some(
                augur_backend::CudaQwen35PrefillCublasPlanV1::new(
                    &self.stream,
                    H * 5 * H,
                    5 * H,
                    512,
                )
                .map_err(err)?,
            );
            self.q8_b8_direct = match std::env::var("AUGUR_DFLASH_Q8_KERNEL").as_deref() {
                Err(std::env::VarError::NotPresent) | Ok("llama") => false,
                Ok("direct") => true,
                _ => return Err("AUGUR_DFLASH_Q8_KERNEL must be llama or direct".into()),
            };
            Ok(())
        }
        #[cfg(not(feature = "qwen35-prefill-cublas"))]
        Err("llama draft projection requires qwen35-prefill-cublas".into())
    }
    #[cfg(feature = "qwen35-prefill-cublas")]
    fn project_llama(
        &self,
        name: &str,
        rows: usize,
        cols: usize,
        input: &str,
        output: &str,
        first: usize,
        count: usize,
        out_first: usize,
    ) -> Result<()> {
        let plan = self
            .llama_projection_plan
            .as_ref()
            .ok_or("missing llama draft plan")?;
        let owner = self
            .weights
            .get(name)
            .ok_or_else(|| format!("missing weight {name}"))?;
        let weight = CudaQ8_0WeightViewV1::new(
            owner.view(0, owner.len_bytes()).map_err(err)?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_0_CUDA_V1,
            rows,
            cols,
        )
        .map_err(err)?;
        for offset in (0..count).step_by(512) {
            let n = (count - offset).min(512);
            plan.submit_dflash_llama_projection(
                &self.stream,
                weight,
                self.view(input, (first + offset) * cols, n * cols)?,
                self.view(output, (out_first + offset) * rows, n * rows)?,
                n,
            )
            .map_err(err)?;
        }
        Ok(())
    }

    fn prepare_context_rows(&self, first: usize, count: usize) -> Result<()> {
        use augur_backend::cuda_dflash_context_rows::CudaDflashContextRowsV1 as Rows;
        let project =
            |name: &str, rows: usize, cols: usize, input: &str, output: &str| -> Result<()> {
                #[cfg(feature = "qwen35-prefill-cublas")]
                if self.llama_projection_plan.is_some() {
                    return self
                        .project_llama(name, rows, cols, input, output, first, count, first);
                }
                let owner = self
                    .weights
                    .get(name)
                    .ok_or_else(|| format!("missing weight {name}"))?;
                let weight = CudaQ8_0WeightViewV1::new(
                    owner.view(0, owner.len_bytes()).map_err(err)?,
                    PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_0_CUDA_V1,
                    rows,
                    cols,
                )
                .map_err(err)?;
                let reuse = self.history_projection_reuse
                    && Rows::history_projection_supported(count, rows, cols);
                let panel = self
                    .history_weight_panel
                    .as_ref()
                    .zip(Rows::history_panel_elements(count, rows, cols));
                let specialized =
                    self.q8_b8_specialized && Rows::specialized_b8_supported(count, rows, cols);
                let projection = if specialized {
                    Rows::projection_b8_specialized
                } else if reuse {
                    Rows::projection_history_reuse
                } else if self.q8_tile4 {
                    Rows::projection_tile4
                } else {
                    Rows::projection
                };
                let input = self.view(input, first * cols, count * cols)?;
                let output = self.view(output, first * rows, count * rows)?;
                let bound = if let Some((scratch, elements)) = panel {
                    let scratch = CudaF32View::new(
                        scratch.view(0, elements * 4).map_err(err)?,
                        PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                        elements,
                    )
                    .map_err(err)?;
                    if self.history_panel_token8 {
                        Rows::projection_history_panel_token8(
                            &self.stream,
                            weight,
                            input,
                            output,
                            scratch,
                            count,
                        )
                    } else {
                        Rows::projection_history_panel(
                            &self.stream,
                            weight,
                            input,
                            output,
                            scratch,
                            count,
                        )
                    }
                } else {
                    projection(&self.stream, weight, input, output, count)
                };
                bound
                    .map_err(err)?
                    .submit_for_plan(&self.stream)
                    .map_err(err)?;
                if specialized {
                    self.q8_b8_specialized_submissions
                        .set(self.q8_b8_specialized_submissions.get() + 1);
                }
                if let Some((_, elements)) = panel {
                    self.history_panel_submissions
                        .set(self.history_panel_submissions.get() + 1);
                    self.history_panel_kernel_launches.set(
                        self.history_panel_kernel_launches.get()
                            + (2 * rows / (elements / cols)) as u64,
                    );
                } else if reuse {
                    self.history_projection_reuse_submissions
                        .set(self.history_projection_reuse_submissions.get() + 1);
                }
                Ok(())
            };
        if self.capture_fc_ready {
            if first != 0 || count != self.context || self.context != 2048 {
                return Err("capture FC initial consumption extent".into());
            }
        } else {
            project("fc.weight", H, 5 * H, "features", "target_fc")?;
        }
        Rows::norm(
            &self.stream,
            self.view("target_fc", first * H, count * H)?,
            self.weight_f32("hidden_norm.weight", H)?,
            self.view("target", first * H, count * H)?,
            count,
            1e-6,
        )
        .map_err(err)?
        .submit_for_plan(&self.stream)
        .map_err(err)?;
        for layer in 0..5 {
            let p = format!("layers.{layer}.self_attn");
            project(&format!("{p}.k_proj.weight"), KV, H, "target", "kv_raw")?;
            if self.context_key_norm_rope {
                Rows::key_norm_rope(
                    &self.stream,
                    self.view("kv_raw", first * KV, count * KV)?,
                    self.weight_f32(&format!("{p}.k_norm.weight"), 128)?,
                    self.view("kv_norm", first * KV, count * KV)?,
                    self.view(&format!("k{layer}"), first * KV, count * KV)?,
                    count,
                    8,
                    self.start + first,
                    DFLASH_ROPE_THETA,
                    1e-6,
                )
                .map_err(err)?
                .submit_for_plan(&self.stream)
                .map_err(err)?;
                self.context_key_norm_rope_submissions
                    .set(self.context_key_norm_rope_submissions.get() + 1);
            } else {
                Rows::norm(
                    &self.stream,
                    self.view("kv_raw", first * KV, count * KV)?,
                    self.weight_f32(&format!("{p}.k_norm.weight"), 128)?,
                    self.view("kv_norm", first * KV, count * KV)?,
                    count * 8,
                    1e-6,
                )
                .map_err(err)?
                .submit_for_plan(&self.stream)
                .map_err(err)?;
                Rows::rope(
                    &self.stream,
                    self.view("kv_norm", first * KV, count * KV)?,
                    self.view(&format!("k{layer}"), first * KV, count * KV)?,
                    count,
                    8,
                    self.start + first,
                    DFLASH_ROPE_THETA,
                )
                .map_err(err)?
                .submit_for_plan(&self.stream)
                .map_err(err)?;
            }
            project(
                &format!("{p}.v_proj.weight"),
                KV,
                H,
                "target",
                &format!("v{layer}"),
            )?;
        }
        Ok(())
    }

    #[cfg(feature = "qwen35-draft-target-verify")]
    pub(crate) fn enable_history_prepare(
        &mut self,
        projection_reuse: bool,
        key_norm_rope: bool,
    ) -> Result<()> {
        if !projection_reuse && !key_norm_rope {
            return Ok(());
        }
        if self.prepared
            || self.submitted
            || self.poisoned
            || self.context_first_row != 0
            || !self.context_rows_launch
            || (projection_reuse && !self.q8_tile4)
        {
            return Err("history preparation policy requires unused compatible owner".into());
        }
        self.history_projection_reuse = projection_reuse;
        self.context_key_norm_rope = key_norm_rope;
        Ok(())
    }

    #[cfg(feature = "qwen35-draft-target-verify")]
    pub(crate) fn enable_history_weight_panels(
        &mut self,
        enabled: bool,
        token8: bool,
    ) -> Result<()> {
        if token8 && !enabled {
            return Err("history panel token8 requires weight panels".into());
        }
        if !enabled {
            return Ok(());
        }
        if self.prepared
            || self.submitted
            || self.poisoned
            || self.graph.is_some()
            || self.context_first_row != 0
            || !self.context_rows_launch
            || !self.q8_tile4
            || !self.history_projection_reuse
            || (self.history_weight_panel.is_some() && !self.capture_fc_ready)
        {
            return Err("history weight panels require unused compatible owner".into());
        }
        if self.capture_fc_ready {
            if !token8
                || self.history_weight_panel.as_ref().map(|p| p.len_bytes())
                    != Some(128 * 25600 * 4)
            {
                return Err("capture FC requires retained token8 panel".into());
            }
        } else {
            self.history_weight_panel = Some(
                OwnedCudaBuffer::allocate(self.stream.device_ordinal(), 128 * 25600 * 4)
                    .map_err(err)?,
            );
        }
        self.history_panel_token8 = token8;
        Ok(())
    }

    #[cfg(feature = "qwen35-draft-target-verify")]
    pub(crate) fn history_prepare_report(&self) -> serde_json::Value {
        serde_json::json!({
            "projection_reuse_requested":self.history_projection_reuse,
            "q8_b8_specialized_requested":self.q8_b8_specialized,
            "q8_b8_specialized_submissions":self.q8_b8_specialized_submissions.get(),
            "key_norm_rope_requested":self.context_key_norm_rope,
            "projection_reuse_submissions":self.history_projection_reuse_submissions.get(),
            "key_norm_rope_submissions":self.context_key_norm_rope_submissions.get(),
            "scope":"draft_owner_successful_host_enqueues_not_graph_replay_node_counts",
            "weight_panels_enabled":self.history_weight_panel.is_some(),
            "weight_panel_token8_requested":self.history_panel_token8,
            "weight_panel_token_tile":if self.history_panel_token8 {8}else{4},
            "weight_panel_token8_submissions":if self.history_panel_token8 {self.history_panel_submissions.get()}else{0},
            "weight_panel_submissions":self.history_panel_submissions.get(),
            "weight_panel_kernel_launches":self.history_panel_kernel_launches.get(),
            "weight_panel_fc_rows":128,"weight_panel_kv_rows":256,
            "weight_panel_decode_included":true,
            "extra_gpu_workspace_bytes":self.history_weight_panel.as_ref().map_or(0, |w| w.len_bytes()),
            "normalized_scratch_preserved":true
            ,"capture_fc":self.capture_fc_report
        })
    }

    /// Target feature encoding and five-layer context K/V projection.
    /// The same stream orders this before the block; no intermediate readback.
    pub fn prepare_context(&mut self) -> Result<()> {
        if self.prepared || self.poisoned || self.submitted {
            return Err("context owner already used/poisoned".into());
        }
        if self.capture_fc_ready && (!self.context_rows_launch || !self.history_panel_token8) {
            return Err("capture FC consumption requires row/panel path".into());
        }
        self.poisoned = true;
        let result = (|| {
            let first = self.context_first_row;
            let count = self
                .context
                .checked_sub(first)
                .ok_or("context projection frontier exceeds initialized extent")?;
            if count == 0 {
                // A same-epoch repeated proposal may reuse every initialized
                // context row. No zero-row projection kernel is launched.
                return Ok(());
            }
            if self.context_rows_launch {
                return self.prepare_context_rows(first, count);
            }
            self.matvec_range(
                "fc.weight",
                H,
                5 * H,
                "features",
                "target_fc",
                count,
                first * 5 * H,
                first * H,
            )?;
            self.norm_range("hidden_norm.weight", "target_fc", "target", first, count, H)?;
            for layer in 0..5 {
                let p = format!("layers.{layer}.self_attn");
                self.matvec_range(
                    &format!("{p}.k_proj.weight"),
                    KV,
                    H,
                    "target",
                    "kv_raw",
                    count,
                    first * H,
                    first * KV,
                )?;
                self.head_norm_rope(
                    &format!("{p}.k_norm.weight"),
                    "kv_raw",
                    "kv_norm",
                    &format!("k{layer}"),
                    count,
                    8,
                    self.start + first,
                    first * KV,
                )?;
                self.matvec_range(
                    &format!("{p}.v_proj.weight"),
                    KV,
                    H,
                    "target",
                    &format!("v{layer}"),
                    count,
                    first * H,
                    first * KV,
                )?;
            }
            Ok(())
        })();
        if result.is_ok() {
            self.prepared = true;
            self.capture_fc_ready = false;
            self.poisoned = false;
        }
        result
    }

    /// Submit all five learned draft layers, final norm and selector projection.
    /// No CUDA allocation, H2D, D2H or host wait is performed inside this method.
    pub fn submit(&mut self) -> Result<()> {
        if !self.prepared || self.submitted || self.poisoned {
            return Err("block owner not prepared or already used/poisoned".into());
        }
        self.poisoned = true;
        let result = self.submit_ops();
        if result.is_ok() {
            self.submitted = true;
            self.poisoned = false;
        }
        result
    }

    fn submit_ops(&self) -> Result<()> {
        let result = (|| {
            for layer in 0..5 {
                let p = format!("layers.{layer}");
                let hidden = format!("hidden{layer}");
                self.norm(
                    &format!("{p}.input_layernorm.weight"),
                    &hidden,
                    "norm",
                    B,
                    H,
                )?;
                self.matvec(
                    &format!("{p}.attention_conv.kernel_projection.weight"),
                    DYN,
                    H,
                    "norm",
                    "dynamic",
                    B,
                    0,
                )?;
                self.conv(
                    &format!("{p}.attention_conv.base_kernel"),
                    "norm",
                    "conv",
                    0,
                )?;
                self.matvec(
                    &format!("{p}.self_attn.q_proj.weight"),
                    Q,
                    H,
                    "conv",
                    "q_raw",
                    B,
                    0,
                )?;
                self.block_head_norm_rope(
                    &format!("{p}.self_attn.q_norm.weight"),
                    "q_raw",
                    "q_norm",
                    "q_rope",
                    B,
                    32,
                    self.start + self.context,
                    0,
                )?;
                // Raw noise rows use their absolute dense K/V destination offset.
                self.matvec(
                    &format!("{p}.self_attn.k_proj.weight"),
                    KV,
                    H,
                    "conv",
                    "kv_raw",
                    B,
                    self.context * KV,
                )?;
                self.block_head_norm_rope(
                    &format!("{p}.self_attn.k_norm.weight"),
                    "kv_raw",
                    "kv_norm",
                    &format!("k{layer}"),
                    B,
                    8,
                    self.start + self.context,
                    self.context * KV,
                )?;
                self.matvec(
                    &format!("{p}.self_attn.v_proj.weight"),
                    KV,
                    H,
                    "conv",
                    &format!("v{layer}"),
                    B,
                    self.context * KV,
                )?;
                CudaDflashBlockAttentionF32ArgsV1::bind(
                    &self.stream,
                    CudaDflashBlockAttentionViewsV1 {
                        query: self.view("q_rope", 0, B * Q)?.buffer_view(),
                        key: self
                            .view(&format!("k{layer}"), 0, (self.context + B) * KV)?
                            .buffer_view(),
                        value: self
                            .view(&format!("v{layer}"), 0, (self.context + B) * KV)?
                            .buffer_view(),
                        output: self.view("attention", 0, B * Q)?.buffer_view(),
                    },
                    self.context,
                    self.start,
                    self.start + self.context,
                    self.window,
                )
                .map_err(err)?
                .submit(&self.stream)
                .map_err(err)?;
                self.matvec(
                    &format!("{p}.self_attn.o_proj.weight"),
                    H,
                    Q,
                    "attention",
                    "projected",
                    B,
                    0,
                )?;
                self.conv(
                    &format!("{p}.attention_conv.base_kernel"),
                    "projected",
                    "conv_out",
                    1,
                )?;
                self.add(&hidden, "conv_out", "ffn_in")?;
                self.norm(
                    &format!("{p}.post_attention_layernorm.weight"),
                    "ffn_in",
                    "norm",
                    B,
                    H,
                )?;
                self.matvec(
                    &format!("{p}.mlp_conv.kernel_projection.weight"),
                    DYN,
                    H,
                    "norm",
                    "dynamic",
                    B,
                    0,
                )?;
                self.conv(&format!("{p}.mlp_conv.base_kernel"), "norm", "conv", 0)?;
                self.matvec(
                    &format!("{p}.mlp.gate_proj.weight"),
                    F,
                    H,
                    "conv",
                    "gate",
                    B,
                    0,
                )?;
                self.matvec(&format!("{p}.mlp.up_proj.weight"), F, H, "conv", "up", B, 0)?;
                CudaSiluMulF32ArgsV1::bind(
                    &self.stream,
                    self.view("gate", 0, B * F)?,
                    self.view("up", 0, B * F)?,
                    self.view("activation", 0, B * F)?,
                )
                .map_err(err)?
                .submit_for_plan(&self.stream)
                .map_err(err)?;
                self.matvec(
                    &format!("{p}.mlp.down_proj.weight"),
                    H,
                    F,
                    "activation",
                    "projected",
                    B,
                    0,
                )?;
                self.conv(
                    &format!("{p}.mlp_conv.base_kernel"),
                    "projected",
                    "conv_out",
                    1,
                )?;
                self.add("ffn_in", "conv_out", &format!("hidden{}", layer + 1))?;
            }
            self.norm("norm.weight", "hidden5", "final", B, H)?;
            #[cfg(feature = "qwen35-prefill-cublas")]
            let gate_f32 = if let Some(owner) = self
                .weights
                .get(selector_gate::KEY)
                .filter(|_| self.selector_gate_enabled)
            {
                let plan = self
                    .llama_projection_plan
                    .as_ref()
                    .ok_or("F32 gate requires llama projection plan")?;
                let weight = CudaF32View::new(
                    owner.view(0, owner.len_bytes()).map_err(err)?,
                    PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                    256 * H,
                )
                .map_err(err)?;
                plan.submit_dflash_selector_gate_with_accumulation(
                    &self.stream,
                    weight,
                    self.view("final", 0, B * H)?,
                    self.view("selector", 0, B * 256)?,
                    self.weights.inner.selector_gate_fp64,
                )
                .map_err(err)?;
                true
            } else {
                false
            };
            #[cfg(not(feature = "qwen35-prefill-cublas"))]
            let gate_f32 = false;
            // All backbone layers have consumed projected. Keep the Q8 gate in
            // this dead scratch until the shared-head consumer drains its stream.
            if gate_f32 && self.selector_phase_gate {
                self.matvec(
                    "candidate_selector.hidden_projection.weight",
                    256,
                    H,
                    "final",
                    "projected",
                    B,
                    0,
                )?;
            }
            if !gate_f32 {
                self.matvec(
                    "candidate_selector.hidden_projection.weight",
                    256,
                    H,
                    "final",
                    "selector",
                    B,
                    0,
                )?;
            }
            #[cfg(feature = "qwen35-output-head-acceptance")]
            if let Some(candidates) = &self.candidates {
                candidates.submit(self)?;
            }
            Ok(())
        })();
        result
    }

    /// Capture the complete prepared five-layer chain into a CUDA graph.
    ///
    /// This is deliberately opt-in.  Capture executes the same `submit`
    /// schedule once while the stream is in capture mode, so the resulting
    /// graph owns the exact device addresses, launch geometry and absolute
    /// attention-position arguments used by that prepared block.  The block
    /// becomes submitted after a successful capture and must be drained before
    /// replay or destruction.  Rebinding is rejected while a graph exists:
    /// the graph also retains scalar launch arguments such as position and
    /// window, not only the buffer addresses.
    pub fn capture_graph(&mut self) -> Result<u64> {
        if !self.prepared || self.submitted || self.poisoned || self.graph.is_some() {
            return Err("block not ready for graph capture".into());
        }
        self.poisoned = true;
        let graph = OwnedCudaGraphExec::capture(&self.stream, || {
            self.submit_ops()
                .map_err(|_| CudaRuntimeError::invalid_metadata("dflash_q8_graph_capture_submit"))
        })
        .map_err(err)?;
        let nodes = graph.node_count();
        self.graph = Some(graph);
        self.submitted = true;
        self.poisoned = false;
        Ok(nodes)
    }

    /// Launch a previously captured complete chain without rebuilding any
    /// operation descriptors on the host.  This method intentionally exposes
    /// no output or acceptance authority; callers still need an explicit
    /// terminal synchronization boundary and the normal diagnostic readback.
    pub fn replay_graph(&self) -> Result<()> {
        if !self.submitted || self.poisoned {
            return Err("graph replay requires a submitted block".into());
        }
        self.graph
            .as_ref()
            .ok_or_else(|| "no graph captured".to_owned())?
            .launch(&self.stream)
            .map_err(err)
    }

    pub fn graph_node_count(&self) -> Option<u64> {
        self.graph.as_ref().map(OwnedCudaGraphExec::node_count)
    }

    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize().map_err(err)
    }

    /// Diagnostic only: the draft's private stream for opt-in event billing.
    pub(crate) fn diagnostic_stream(&self) -> &OwnedCudaStream {
        &self.stream
    }

    /// Diagnostic only: read back after the complete device chain, not per op.
    pub fn readback_diagnostics(&self) -> Result<DFlashQ8CudaDiagnosticOutput> {
        if !self.submitted || self.poisoned {
            return Err("draft output unavailable".into());
        }
        let mut layers = Vec::new();
        for layer in 1..=5 {
            layers.push(self.read(&format!("hidden{layer}"), B * H)?);
        }
        Ok(DFlashQ8CudaDiagnosticOutput {
            hidden: self.read("final", B * H)?,
            selector_hidden: self.read("selector", B * 256)?,
            layer_hidden: layers,
            d2h_bytes: (6 * B * H + B * 256) * 4,
        })
    }

    /// Explicit opt-in audit, never a steady product fast path. Compare
    /// bounded device fingerprints of all ten K/V planes, five hidden layers,
    /// final norm and selector against full recomputation on the same seed.
    /// Fingerprints are diagnostic, not a cryptographic tensor equality proof.
    #[cfg(feature = "qwen35-draft-target-verify")]
    pub(crate) fn audit_full_context_recompute(&mut self) -> Result<[u8; 32]> {
        if !self.submitted || self.poisoned {
            return Err("cannot audit an unavailable draft".into());
        }
        let before = self.compact_context_diagnostic()?;
        self.prepared = false;
        self.submitted = false;
        self.context_first_row = 0;
        self.prepare_context()?;
        self.submit()?;
        let after = self.compact_context_diagnostic()?;
        if before != after {
            self.poisoned = true;
            return Err("incremental/full context diagnostic fingerprint mismatch".into());
        }
        Ok(before)
    }

    #[cfg(feature = "qwen35-draft-target-verify")]
    fn compact_context_diagnostic(&self) -> Result<[u8; 32]> {
        use augur_backend::{submit_page_digest_into_view_v1, submit_page_digest_v1};
        let mut names = vec!["final".to_owned(), "selector".to_owned()];
        for layer in 0..5 {
            names.extend([
                format!("k{layer}"),
                format!("v{layer}"),
                format!("hidden{}", layer + 1),
            ]);
        }
        for (index, name) in names.iter().enumerate() {
            submit_page_digest_into_view_v1(
                &self.stream,
                &self.buffers[name],
                self.buffers["audit_parts"]
                    .view(index * 32, 32)
                    .map_err(err)?,
            )
            .map_err(err)?;
        }
        submit_page_digest_v1(
            &self.stream,
            &self.buffers["audit_parts"],
            &self.buffers["audit_digest"],
        )
        .map_err(err)?;
        let mut bytes = [0u8; 32];
        self.buffers["audit_digest"]
            .copy_to_host_sync(&self.stream, &mut bytes)
            .map_err(err)?;
        Ok(bytes)
    }

    fn read(&self, name: &str, count: usize) -> Result<Vec<f32>> {
        let mut bytes = vec![0; count * 4];
        self.buffers[name]
            .copy_to_host_sync(&self.stream, &mut bytes)
            .map_err(err)?;
        let output = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        if output.iter().any(|v| !v.is_finite()) {
            return Err(format!("non-finite device output {name}"));
        }
        Ok(output)
    }
    fn view(&self, name: &str, offset: usize, count: usize) -> Result<CudaF32View> {
        CudaF32View::new(
            self.buffers
                .get(name)
                .ok_or_else(|| format!("missing buffer {name}"))?
                .view(offset * 4, count * 4)
                .map_err(err)?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            count,
        )
        .map_err(err)
    }
    fn weight_f32(&self, name: &str, count: usize) -> Result<CudaF32View> {
        CudaF32View::new(
            self.weights
                .get(name)
                .ok_or_else(|| format!("missing weight {name}"))?
                .view(0, count * 4)
                .map_err(err)?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            count,
        )
        .map_err(err)
    }
    fn matvec(
        &self,
        name: &str,
        rows: usize,
        cols: usize,
        input: &str,
        output: &str,
        batch: usize,
        out_offset: usize,
    ) -> Result<()> {
        #[cfg(feature = "qwen35-prefill-cublas")]
        if let Some(plan) = self.llama_projection_plan.as_ref() {
            if out_offset % rows != 0 {
                return Err("unaligned draft output row".into());
            }
            if self.q8_b8_direct && batch == B {
                let owner = self
                    .weights
                    .get(name)
                    .ok_or_else(|| format!("missing weight {name}"))?;
                let weight = CudaQ8_0WeightViewV1::new(
                    owner.view(0, owner.len_bytes()).map_err(err)?,
                    PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_0_CUDA_V1,
                    rows,
                    cols,
                )
                .map_err(err)?;
                return plan
                    .submit_dflash_q8_b8_direct(
                        &self.stream,
                        weight,
                        self.view(input, 0, B * cols)?,
                        self.view(output, out_offset, B * rows)?,
                    )
                    .map_err(err);
            }
            return self.project_llama(
                name,
                rows,
                cols,
                input,
                output,
                0,
                batch,
                out_offset / rows,
            );
        }
        if self.block_rows_launch {
            use augur_backend::cuda_dflash_context_rows::CudaDflashContextRowsV1 as Rows;
            let owner = self
                .weights
                .get(name)
                .ok_or_else(|| format!("missing weight {name}"))?;
            let weight = CudaQ8_0WeightViewV1::new(
                owner.view(0, owner.len_bytes()).map_err(err)?,
                PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_0_CUDA_V1,
                rows,
                cols,
            )
            .map_err(err)?;
            let specialized =
                self.q8_b8_specialized && Rows::specialized_b8_supported(batch, rows, cols);
            let projection = if specialized {
                Rows::projection_b8_specialized
            } else if self.q8_tile4 {
                Rows::projection_tile4
            } else {
                Rows::projection
            };
            projection(
                &self.stream,
                weight,
                self.view(input, 0, batch * cols)?,
                self.view(output, out_offset, batch * rows)?,
                batch,
            )
            .map_err(err)?
            .submit_for_plan(&self.stream)
            .map_err(err)?;
            if specialized {
                self.q8_b8_specialized_submissions
                    .set(self.q8_b8_specialized_submissions.get() + 1);
            }
            return Ok(());
        }
        self.matvec_range(name, rows, cols, input, output, batch, 0, out_offset)
    }
    fn matvec_range(
        &self,
        name: &str,
        rows: usize,
        cols: usize,
        input: &str,
        output: &str,
        batch: usize,
        input_offset: usize,
        out_offset: usize,
    ) -> Result<()> {
        let weight = self
            .weights
            .get(name)
            .ok_or_else(|| format!("missing weight {name}"))?;
        let view = CudaQ8_0WeightViewV1::new(
            weight.view(0, weight.len_bytes()).map_err(err)?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_0_CUDA_V1,
            rows,
            cols,
        )
        .map_err(err)?;
        for item in 0..batch {
            CudaQ8_0F32MatvecArgsV1::bind(
                &self.stream,
                view,
                self.view(input, input_offset + item * cols, cols)?,
                self.view(output, out_offset + item * rows, rows)?,
            )
            .map_err(err)?
            .submit_for_plan(&self.stream)
            .map_err(err)?;
        }
        Ok(())
    }
    fn norm(
        &self,
        weight: &str,
        input: &str,
        output: &str,
        batch: usize,
        width: usize,
    ) -> Result<()> {
        if self.block_rows_launch {
            use augur_backend::cuda_dflash_context_rows::CudaDflashContextRowsV1 as Rows;
            return Rows::norm(
                &self.stream,
                self.view(input, 0, batch * width)?,
                self.weight_f32(weight, width)?,
                self.view(output, 0, batch * width)?,
                batch,
                1e-6,
            )
            .map_err(err)?
            .submit_for_plan(&self.stream)
            .map_err(err);
        }
        self.norm_range(weight, input, output, 0, batch, width)
    }

    #[allow(clippy::too_many_arguments)]
    fn block_head_norm_rope(
        &self,
        weight: &str,
        input: &str,
        norm: &str,
        output: &str,
        batch: usize,
        heads: usize,
        position: usize,
        offset: usize,
    ) -> Result<()> {
        if !self.block_rows_launch {
            return self
                .head_norm_rope(weight, input, norm, output, batch, heads, position, offset);
        }
        use augur_backend::cuda_dflash_context_rows::CudaDflashContextRowsV1 as Rows;
        let elements = batch * heads * 128;
        Rows::norm(
            &self.stream,
            self.view(input, offset, elements)?,
            self.weight_f32(weight, 128)?,
            self.view(norm, offset, elements)?,
            batch * heads,
            1e-6,
        )
        .map_err(err)?
        .submit_for_plan(&self.stream)
        .map_err(err)?;
        Rows::rope(
            &self.stream,
            self.view(norm, offset, elements)?,
            self.view(output, offset, elements)?,
            batch,
            heads,
            position,
            DFLASH_ROPE_THETA,
        )
        .map_err(err)?
        .submit_for_plan(&self.stream)
        .map_err(err)
    }
    fn norm_range(
        &self,
        weight: &str,
        input: &str,
        output: &str,
        first: usize,
        batch: usize,
        width: usize,
    ) -> Result<()> {
        for row in first..first + batch {
            CudaRmsnormF32ArgsV1::bind(
                &self.stream,
                self.view(input, row * width, width)?,
                self.weight_f32(weight, width)?,
                self.view(output, row * width, width)?,
                1e-6,
            )
            .map_err(err)?
            .submit_for_plan(&self.stream)
            .map_err(err)?;
        }
        Ok(())
    }
    fn head_norm_rope(
        &self,
        weight: &str,
        input: &str,
        norm: &str,
        output: &str,
        batch: usize,
        heads: usize,
        position: usize,
        offset: usize,
    ) -> Result<()> {
        for row in 0..batch {
            for head in 0..heads {
                let at = offset + (row * heads + head) * 128;
                CudaRmsnormF32ArgsV1::bind(
                    &self.stream,
                    self.view(input, at, 128)?,
                    self.weight_f32(weight, 128)?,
                    self.view(norm, at, 128)?,
                    1e-6,
                )
                .map_err(err)?
                .submit_for_plan(&self.stream)
                .map_err(err)?;
                CudaDflashRopeF32ArgsV1::bind(
                    &self.stream,
                    self.view(norm, at, 128)?,
                    self.view(output, at, 128)?,
                    128,
                    128,
                    position + row,
                    DFLASH_ROPE_THETA,
                )
                .map_err(err)?
                .submit_for_plan(&self.stream)
                .map_err(err)?;
            }
        }
        Ok(())
    }
    fn conv(&self, base: &str, input: &str, output: &str, side: usize) -> Result<()> {
        if self.block_rows_launch {
            return augur_backend::CudaDflashGroupedConvHistoryRowsF32ArgsV2::bind(
                &self.stream,
                self.view(input, 0, B * H)?,
                self.view("zero", 0, H)?,
                self.view("dynamic", 0, B * DYN)?,
                self.weight_f32(base, 4 * H)?,
                self.view(output, 0, B * H)?,
                B,
                H / 16,
                16,
                side,
            )
            .map_err(err)?
            .submit_for_plan(&self.stream)
            .map_err(err);
        }
        for row in 0..B {
            let history = if row == 0 {
                self.view("zero", 0, H)?
            } else {
                self.view(input, (row - 1) * H, H)?
            };
            CudaDflashGroupedConvHistoryF32ArgsV1::bind(
                &self.stream,
                self.view(input, row * H, H)?,
                history,
                self.view("dynamic", row * DYN, DYN)?,
                self.weight_f32(base, 4 * H)?,
                self.view(output, row * H, H)?,
                H / 16,
                16,
                side,
            )
            .map_err(err)?
            .submit_for_plan(&self.stream)
            .map_err(err)?;
        }
        Ok(())
    }
    fn add(&self, a: &str, b: &str, out: &str) -> Result<()> {
        CudaAddF32ArgsV1::bind(
            &self.stream,
            self.view(a, 0, B * H)?,
            self.view(b, 0, B * H)?,
            self.view(out, 0, B * H)?,
        )
        .map_err(err)?
        .submit_for_plan(&self.stream)
        .map_err(err)
    }
}

impl Drop for DFlashQ8CudaBlock {
    fn drop(&mut self) {
        // Includes partial-submit failures. Do not release owners while queued
        // work can reference their views. No work uses another stream.
        let _ = self.stream.synchronize();
    }
}

fn validate_device_handoff(
    config: &DFlashConfig,
    seed: &DFlashDraftBlockInput,
    capture: &DFlashTargetFeatureCaptureReceipt,
    features: CudaBufferView,
    noise: &[f32],
    start: usize,
    window: usize,
    device: i32,
    source_device: i32,
) -> Result<usize> {
    seed.validate_initial_seed(config).map_err(err)?;
    capture.validate(config).map_err(err)?;
    let context = capture.context_len;
    let bytes = context
        .checked_mul(5 * H * 4)
        .ok_or_else(|| "device feature extent overflow".to_owned())?;
    if features.len_bytes() != bytes
        || capture.context_start_position != start
        || capture.feature_elements != context * 5 * H
        || capture.feature_sha256 != seed.target_feature_sha256
        || noise.len() != B * H
        || noise.iter().any(|v| !v.is_finite())
        || !(1..=2048).contains(&context)
        || !(1..=2048).contains(&window)
        || features.device_ordinal() != device
        || source_device != device
    {
        return Err("device feature handoff contract mismatch".into());
    }
    Ok(context)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn immutable_weight_lease_retains_payload_until_last_request_releases_it() {
        // CPU-only ownership test: empty storage is constructed only inside
        // this module. Production handles can only be created by upload().
        let config = DFlashConfig {
            architecture: "DFlash2DraftModel".into(),
            model_type: "qwen3".into(),
            dtype: "float16".into(),
            hidden_size: H,
            intermediate_size: F,
            num_hidden_layers: 5,
            num_target_layers: 64,
            attention_heads: 32,
            kv_heads: 8,
            head_dim: 128,
            max_position_embeddings: 262144,
            vocab_size: 248320,
            block_size: B,
            conv_group_size: 16,
            conv_kernel_size: 2,
            selector_rank: 256,
            selector_top_k: 16,
            target_layer_ids: crate::DFLASH_TARGET_CAPTURE_LAYER_IDS.to_vec(),
        };
        let weights = DFlashQ8CudaWeights {
            inner: Rc::new(DFlashQ8CudaWeightPayload {
                owners: BTreeMap::new(),
                config: config.clone(),
                device: 0,
                file_sha256: "a".repeat(64),
                file_bytes: 123,
                selector_gate_f32_sha256: None,
                selector_gate_fp64: false,
                selector_codebooks_bf16_sha256: None,
            }),
        };
        let weak = Rc::downgrade(&weights.inner);
        let request1 = weights.clone();
        let request2 = weights.clone();
        assert!(Rc::ptr_eq(&request1.inner, &request2.inner));
        assert_eq!(request1.config(), &config);
        assert_eq!(request2.file_sha256(), "a".repeat(64));
        assert_eq!(request2.file_bytes(), 123);
        assert_eq!(request2.device_ordinal(), 0);
        drop(weights);
        drop(request1);
        assert!(weak.upgrade().is_some());
        drop(request2);
        assert!(weak.upgrade().is_none());
    }

    fn cached_capture(
        start: usize,
        generation: u64,
        context: usize,
    ) -> DFlashTargetFeatureCaptureReceipt {
        DFlashTargetFeatureCaptureReceipt {
            schema: crate::DFLASH_TARGET_CAPTURE_SCHEMA.into(),
            request_generation: generation,
            context_start_position: start,
            context_len: context,
            target_layer_ids: crate::DFLASH_TARGET_CAPTURE_LAYER_IDS.to_vec(),
            target_hidden_size: H,
            feature_width: 5 * H,
            feature_elements: context * 5 * H,
            source_target_model_sha256: [1; 32],
            feature_sha256: [2; 32],
        }
    }

    #[test]
    fn context_reuse_preserves_absolute_rope_positions_and_only_adds_committed_rows() {
        for context in [1, 8, 17, 2048] {
            for committed in 1..=B {
                let prior = cached_capture(196000, 5, context);
                let next = cached_capture(196000 + committed, 6, context);
                let retained = context_reuse_prefix(&prior, &next, committed).unwrap();
                assert_eq!(context - retained, committed.min(context));
                for row in 0..retained {
                    assert_eq!(
                        prior.context_start_position + committed + row,
                        next.context_start_position + row
                    );
                }
                let mut positions = (prior.context_start_position
                    ..prior.context_start_position + context)
                    .collect::<Vec<_>>();
                if retained > 0 {
                    let scratch = positions[committed..].to_vec();
                    positions[..retained].copy_from_slice(&scratch);
                }
                for row in retained..context {
                    positions[row] = next.context_start_position + row;
                }
                assert_eq!(
                    positions,
                    (next.context_start_position..next.context_start_position + context)
                        .collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn context_reuse_rejects_stale_foreign_unbounded_or_skipped_history() {
        let prior = cached_capture(100, 7, 2048);
        let next = cached_capture(102, 8, 2048);
        for mutation in 0..6 {
            let mut bad = next.clone();
            match mutation {
                0 => bad.request_generation = 7,
                1 => bad.request_generation = 9,
                2 => bad.source_target_model_sha256 = [3; 32],
                3 => bad.context_start_position = 103,
                4 => bad.context_len = 8,
                _ => bad.target_layer_ids[0] = 6,
            }
            assert!(context_reuse_prefix(&prior, &bad, 2).is_err());
        }
        assert!(context_reuse_prefix(&prior, &next, 0).is_err());
        assert!(context_reuse_prefix(&prior, &next, 9).is_err());
        let mut overflow = prior.clone();
        overflow.request_generation = u64::MAX;
        assert!(context_reuse_prefix(&overflow, &next, 2).is_err());
        overflow = prior.clone();
        overflow.context_start_position = usize::MAX;
        assert!(context_reuse_prefix(&overflow, &next, 2).is_err());
    }

    #[test]
    fn growing_rebind_matches_receipts_for_offset_growth_and_rollover() {
        for start in [0, 6144, 194560] {
            for rows in [3, 8, 17, 2047, 2048] {
                let mut history = GrowingHistory::from_extent(
                    crate::GrowingHistoryConfig::default_2048(),
                    start,
                    rows,
                )
                .unwrap();
                let mut prior = cached_capture(start, 9, rows);
                for committed in [1, 7, 3, 8] {
                    let plan = history.append(committed).unwrap();
                    let next = cached_capture(
                        history.start_position(),
                        prior.request_generation + 1,
                        history.active_rows(),
                    );
                    validate_growing_rebind(&prior, &next, plan, history, 2048, committed).unwrap();
                    prior = next;
                }
            }
        }
    }

    #[test]
    fn growing_rebind_accepts_one_plan_for_multiple_suffix_blocks() {
        for (start, rows, commits) in [
            (0, 3, [8usize, 1, 7]),
            (6144, 2047, [8usize, 8, 1]),
            (194560, 2048, [8usize, 8, 8]),
        ] {
            let mut history = GrowingHistory::from_extent(
                crate::GrowingHistoryConfig::default_2048(),
                start,
                rows,
            )
            .unwrap();
            let old = history;
            let prior = cached_capture(old.start_position(), 9, old.active_rows());
            let expected_append = commits.into_iter().sum::<usize>();
            for committed in commits {
                history.append(committed).unwrap();
            }
            let plan = GrowingHistoryPlan::between(old, history).unwrap();
            let next = cached_capture(
                history.start_position(),
                prior.request_generation + 1,
                history.active_rows(),
            );
            assert_eq!(plan.append_rows, expected_append);
            validate_growing_rebind(&prior, &next, plan, history, 2048, plan.append_rows).unwrap();
        }
    }

    #[test]
    fn anchor_only_fresh_draft_does_not_replay_input_growth_plan() {
        for rows in [8, 2048] {
            for input_tail in [1, 8] {
                let mut history = GrowingHistory::from_extent(
                    crate::GrowingHistoryConfig::default_2048(),
                    6144,
                    rows,
                )
                .unwrap();
                let applied_input_plan = history.append(input_tail).unwrap();
                let fresh = cached_capture(history.start_position(), 1, history.active_rows());
                // A newly rebuilt draft already covers this input. Replaying
                // its old append plan must remain rejected, even if all bytes
                // and model identities match and only one anchor was emitted.
                assert!(validate_growing_rebind(
                    &fresh,
                    &fresh,
                    applied_input_plan,
                    history,
                    2048,
                    input_tail
                )
                .is_err());
                assert!(validate_growing_refresh_phase(
                    false,
                    false,
                    false,
                    false,
                    0,
                    history.active_rows()
                )
                .is_ok());
                let draft_base = history;
                // The next real user suffix advances the target, not a fake
                // seed submit. A full refresh then admits the untouched draft.
                for count in [8, 1] {
                    history.append(count).unwrap();
                }
                let next = cached_capture(history.start_position(), 2, history.active_rows());
                validate_growing_refresh(&fresh, &next, history, 2048).unwrap();
                assert!(validate_growing_refresh(&fresh, &fresh, history, 2048).is_err());
                // More anchor-only turns keep the original draft base. Their
                // last individual append plan cannot replace the aggregate.
                for generation in [3, 4] {
                    let tail = history.append(3).unwrap();
                    let next =
                        cached_capture(history.start_position(), generation, history.active_rows());
                    assert!(validate_growing_rebind(
                        &fresh,
                        &next,
                        tail,
                        history,
                        2048,
                        tail.append_rows
                    )
                    .is_err());
                    let aggregate = GrowingHistoryPlan::between(draft_base, history).unwrap();
                    validate_growing_rebind(
                        &fresh,
                        &next,
                        aggregate,
                        history,
                        2048,
                        aggregate.append_rows,
                    )
                    .unwrap();
                    assert_eq!(
                        growing_projection_start(
                            false,
                            false,
                            0,
                            draft_base.active_rows(),
                            aggregate.evicted_rows,
                            aggregate.retained_rows
                        )
                        .unwrap(),
                        0
                    );
                }
            }
        }
    }

    #[test]
    fn growing_full_refresh_accepts_completed_and_unprepared_rebind_not_partial_submit() {
        for graph in [false, true] {
            for prepared in [false, true] {
                for submitted in [false, true] {
                    for poisoned in [false, true] {
                        for first_row in [0, 8, 2048, 2049] {
                            let expected =
                                !graph && !poisoned && prepared == submitted && first_row <= 2048;
                            assert_eq!(
                                validate_growing_refresh_phase(
                                    graph, prepared, submitted, poisoned, first_row, 2048
                                )
                                .is_ok(),
                                expected
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn long_suffix_full_refresh_accepts_pending_incremental_tail_without_projecting_it() {
        // Real HTTP 8K + 2048 suffix (+ pending anchor): append first aligns
        // the prior two-token commit, leaving 2046 cached rows and a dirty tail.
        // The subsequent >window append must refresh all 2048 rows, not reject.
        assert!(validate_growing_refresh_phase(false, false, false, false, 2046, 2048).is_ok());
        for (first, rows) in [(0, 0), (0, 2049), (9, 8), (usize::MAX, 2048)] {
            assert!(
                validate_growing_refresh_phase(false, false, false, false, first, rows).is_err()
            );
        }
        assert!(validate_growing_refresh_phase(false, true, false, false, 2046, 2048).is_err());
        assert!(validate_growing_refresh_phase(false, false, true, false, 2046, 2048).is_err());
    }

    #[test]
    fn chained_rebind_projects_prior_dirty_tail_together_with_new_rows() {
        // Completed 2048 rows -> commit2 -> input63, without prepare between.
        let first = growing_projection_start(true, true, 2040, 2048, 2, 2046).unwrap();
        assert_eq!(first, 2046);
        let second = growing_projection_start(false, false, first, 2048, 63, 1985).unwrap();
        assert_eq!(second, 1983);
        assert_eq!(2048 - second, 65);
        for old_rows in [1, 8, 31, 2048] {
            for first in 0..=old_rows {
                for evicted in [0, old_rows / 2, old_rows] {
                    let retained = old_rows - evicted;
                    let valid_rows = (0..old_rows)
                        .map(|i| i < first)
                        .skip(evicted)
                        .take_while(|valid| *valid)
                        .count();
                    assert_eq!(
                        growing_projection_start(false, false, first, old_rows, evicted, retained)
                            .unwrap(),
                        valid_rows
                    );
                    assert_eq!(
                        growing_projection_start(true, true, first, old_rows, evicted, retained)
                            .unwrap(),
                        retained
                    );
                }
            }
        }
        for (p, s, first, rows, evict, retain) in [
            (true, false, 0, 8, 1, 7),
            (false, true, 0, 8, 1, 7),
            (false, false, 9, 8, 1, 7),
            (false, false, 0, 8, 1, 8),
            (false, false, 0, 0, 0, 0),
            (false, false, 0, 2049, 1, 2048),
        ] {
            assert!(growing_projection_start(p, s, first, rows, evict, retain).is_err());
        }
    }

    #[test]
    fn growing_full_refresh_binds_all_input_rows_not_just_last_append_plan() {
        for rows in [1, 7, 8, 31, 2047, 2048] {
            let prior = cached_capture(0, 9, rows);
            let mut history =
                GrowingHistory::from_extent(crate::GrowingHistoryConfig::default_2048(), 0, rows)
                    .unwrap();
            for _ in 0..2051 {
                history.append(1).unwrap();
            }
            let next = cached_capture(history.start_position(), 11, history.active_rows());
            validate_growing_refresh(&prior, &next, history, 2048).unwrap();
            for mutation in 0..6 {
                let mut bad = next.clone();
                match mutation {
                    0 => bad.context_len -= 1,
                    1 => bad.context_start_position += 1,
                    2 => bad.request_generation = 9,
                    3 => bad.source_target_model_sha256 = [3; 32],
                    4 => bad.target_layer_ids.clear(),
                    _ => bad.target_hidden_size += 1,
                }
                assert!(validate_growing_refresh(&prior, &bad, history, 2048).is_err());
            }
            assert!(validate_growing_refresh(&prior, &next, history, 1024).is_err());
        }
    }

    #[test]
    fn growing_rebind_rejects_corrupt_plan_extent_generation_and_identity() {
        let mut history =
            GrowingHistory::from_extent(crate::GrowingHistoryConfig::default_2048(), 6144, 2047)
                .unwrap();
        let prior = cached_capture(6144, 9, 2047);
        let plan = history.append(3).unwrap();
        let next = cached_capture(history.start_position(), 10, history.active_rows());
        for mutation in 0..10 {
            let mut bad_plan = plan;
            let mut bad_next = next.clone();
            match mutation {
                0 => bad_plan.append_offset += 1,
                1 => bad_plan.append_rows += 1,
                2 => bad_plan.retained_rows -= 1,
                3 => bad_plan.evicted_rows += 1,
                4 => bad_plan.generation_before += 1,
                5 => bad_plan.generation_after += 1,
                6 => bad_next.request_generation -= 1,
                7 => bad_next.source_target_model_sha256 = [3; 32],
                8 => bad_next.context_start_position += 1,
                _ => bad_next.context_len -= 1,
            }
            assert!(
                validate_growing_rebind(&prior, &bad_next, bad_plan, history, 2048, 3).is_err()
            );
        }
        assert!(validate_growing_rebind(&prior, &next, plan, history, 1024, 3).is_err());
        assert!(validate_growing_rebind(&prior, &next, plan, history, 2048, 0).is_err());
        assert!(validate_growing_rebind(&prior, &next, plan, history, 2048, 4).is_err());
    }

    #[test]
    #[ignore = "requires an SM70 CUDA device; explicit numerical diagnostic"]
    fn absolute_rope_matches_cpu_phase_contract_on_sm70() {
        let stream = OwnedCudaStream::create(0).unwrap();
        let input = (0..128)
            .map(|i| ((i * 17 % 101) as f32 - 50.0) / 17.0)
            .collect::<Vec<_>>();
        let a = OwnedCudaBuffer::allocate(0, 512).unwrap();
        let b = OwnedCudaBuffer::allocate(0, 512).unwrap();
        a.copy_from_host_sync(
            &stream,
            &input
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let view = |owner: &OwnedCudaBuffer| {
            CudaF32View::new(
                owner.view(0, 512).unwrap(),
                PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                128,
            )
            .unwrap()
        };
        let mut worst = 0.0f32;
        for position in [0, 1, 4096, 4097, 5000, 6144, 196600, 262136] {
            let mut expected = input.clone();
            crate::apply_dflash_neox_rope(&mut expected, position).unwrap();
            CudaDflashRopeF32ArgsV1::bind(
                &stream,
                view(&a),
                view(&b),
                128,
                128,
                position,
                DFLASH_ROPE_THETA,
            )
            .unwrap()
            .submit_for_plan(&stream)
            .unwrap();
            let mut bytes = vec![0; 512];
            b.copy_to_host_sync(&stream, &mut bytes).unwrap();
            let actual = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>();
            let max = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            println!("position={position} max_abs={max:e}");
            worst = worst.max(max);
        }
        assert!(worst <= 2e-6, "CPU/CUDA phase contract drift {worst:e}");
    }
}
