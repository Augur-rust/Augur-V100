//! Same-owner real target verification of a GPU Q8 draft. Diagnostic transaction
//! driver: target conditioning, physical rollback, and accepted-prefix audit.
use super::*;
use crate::{
    DFlashDraftBlockInput, DFlashQ8Backbone, DFlashQ8CudaBlock, DFlashQ8CudaWeights,
    DFlashTargetFeatureCaptureReceipt, GrowingHistory, GrowingHistoryConfig, GrowingHistoryPlan,
    DEFAULT_GROWING_HISTORY_CAPACITY, DFLASH_TARGET_CAPTURE_LAYER_IDS,
    DFLASH_TARGET_CAPTURE_SCHEMA,
};
use augur_backend::{
    parse_cuda_qwen35_dflash_acceptance_receipt_v1, CudaQwen35DflashAcceptanceArgsV1,
    CudaQwen35DflashAcceptanceViewsV1, CudaQwen35DflashQ8SelectorArgsV1,
    CudaQwen35DflashQ8SelectorViewsV1, CudaQwen35VocabTop1ArgsV1, CudaQwen35VocabTopkArgsV1,
    QWEN35_DFLASH_ACCEPTANCE_RECEIPT_BYTES,
};
use augur_model_qwen35::DFlashConfig;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
const H: usize = 5120;
const V: usize = 248320;
const B: usize = 8;
type Result<T> = std::result::Result<T, Qwen35MixedPrefixCudaError>;
mod batched_verify;
#[cfg(all(test, feature = "qwen35-session-continuation"))]
mod qpn_batch_smoke;
mod product_stage_profile;
#[cfg(feature = "qwen35-session-continuation")]
mod committed_session;
#[cfg(feature = "qwen35-session-continuation")]
mod prefill_scalar_audit;
#[cfg(feature = "qwen35-session-continuation")]
mod prefill_tiled_audit;

#[cfg(feature = "qwen35-raw-committed-state")]
mod raw_state;
mod reference_boundary;
#[cfg(all(
    feature = "qwen35-session-continuation",
    feature = "qwen35-draft-private-state"
))]
mod session_private_audit;
mod target_only_reference;
mod fixed_history_logits;
mod top16_validation;
mod trajectory;
use batched_verify::{BatchedVerifyAudit, BatchedVerifyExecution};

fn small_attention_row_contract(tile8: bool, vector: bool, serial: bool, gqa8: bool) -> Result<&'static str> {
    if usize::from(tile8) + usize::from(vector) + usize::from(serial) + usize::from(gqa8) > 1 {
        return Err(invalid("conflicting small attention arithmetic contracts"));
    }
    Ok(if gqa8 { "QWEN35-SMALL-ATTENTION-GQA8-EXACT-DECODE-COHORT-V1" }
        else if tile8 { "QWEN35-SMALL-ATTENTION-FIXED-TILE8-PER-QUERY256-V1" }
        else if vector { "QWEN35-VERIFY-ATTENTION-PARALLEL-B1-VEC-PER-QUERY256-V1" }
        else if serial { "QWEN35-VERIFY-ATTENTION-SERIAL-B1-DIAGNOSTIC-V1" }
        else { "legacy_mixed_vec_tile_v1" })
}

fn product_stage_switch(value: Option<&std::ffi::OsStr>, feature: bool) -> Result<bool> {
    match value {
        None => Ok(false),
        Some(v) if feature && v == "1" => Ok(true),
        _ => Err(invalid(
            "product stage profile requires opt-in feature and switch exactly 1",
        )),
    }
}

fn reference_canonical_prefix_switch(
    value: Option<&std::ffi::OsStr>,
    compiled: bool,
    diagnostic_scope: bool,
) -> Result<bool> {
    match value {
        None => Ok(false),
        Some(v) if v == "1" && compiled && diagnostic_scope => Ok(true),
        _ => Err(invalid("canonical reference prefix requires explicit complete-residency target-only boundary diagnostic")),
    }
}

fn growing_history_switch(value: Option<&std::ffi::OsStr>) -> Result<bool> {
    match value {
        None => Ok(false),
        Some(v) if v == "1" => Ok(true),
        _ => Err(invalid(
            "growing history requires AUGUR_QWEN35_GROWING_HISTORY=1",
        )),
    }
}

fn fast_feature_identity_switch(value: Option<&std::ffi::OsStr>) -> Result<bool> {
    match value {
        None => Ok(false),
        Some(v) if v == "1" => Ok(true),
        _ => Err(invalid(
            "fast feature identity requires AUGUR_QWEN35_FAST_FEATURE_IDENTITY=1",
        )),
    }
}

fn submission_batch_switch(value: Option<&std::ffi::OsStr>, eligible: bool) -> Result<bool> {
    match value {
        None => Ok(false),
        Some(v) if v == "1" && eligible => Ok(true),
        _ => Err(invalid("submission batching requires explicit exact-input product recipe")),
    }
}

fn replay_head_elision_switch(value: Option<&std::ffi::OsStr>) -> Result<bool> {
    match value {
        None => Ok(false),
        Some(v) if v == "1" => Ok(true),
        _ => Err(invalid(
            "replay head elision requires AUGUR_QWEN35_REPLAY_HEAD_ELISION=1",
        )),
    }
}

fn fast_feature_identity(
    identity: [u8; 32],
    generation: u64,
    cursor: usize,
    anchor: u32,
    context_start: usize,
    context_rows: usize,
    capacity: usize,
) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"QWEN35-FEATURE-IDENTITY-FAST-V1\0");
    digest.update(identity);
    digest.update(generation.to_le_bytes());
    digest.update((cursor as u64).to_le_bytes());
    digest.update(anchor.to_le_bytes());
    digest.update((context_start as u64).to_le_bytes());
    digest.update((context_rows as u64).to_le_bytes());
    digest.update((capacity as u64).to_le_bytes());
    digest.finalize().into()
}

fn validate_growing_history_scope(
    enabled: bool,
    product: bool,
    bounded: bool,
    incremental: bool,
    fixed_extent_consumer: bool,
) -> Result<()> {
    if enabled && (!product || !bounded || !incremental || fixed_extent_consumer) {
        return Err(invalid(
            "growing history requires bounded incremental product mode without fixed-extent session/audit consumers",
        ));
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct PublicationBudget<'a> {
    remaining: usize,
    eos: &'a [u32],
}

fn clip_publication(
    tokens: &mut Vec<u32>,
    budget: Option<PublicationBudget<'_>>,
) -> Result<Option<&'static str>> {
    if tokens.is_empty() || tokens.iter().any(|&id| id as usize >= V) {
        return Err(invalid("empty/invalid token publication"));
    }
    let Some(budget) = budget else {
        return Ok(None);
    };
    if budget.remaining == 0 || budget.eos.is_empty() {
        return Err(invalid("exhausted/missing request publication budget"));
    }
    tokens.truncate(budget.remaining);
    if let Some(index) = tokens.iter().position(|id| budget.eos.contains(id)) {
        tokens.truncate(index + 1);
        return Ok(Some("eos"));
    }
    Ok((tokens.len() == budget.remaining).then_some("max_new_tokens"))
}

fn prompt_capture_extent(prompt_len: usize, requested_rows: usize) -> Result<(usize, usize)> {
    if prompt_len == 0 || !(1..=2048).contains(&requested_rows) {
        return Err(invalid("request prompt capture extent"));
    }
    let rows = requested_rows.min(prompt_len);
    Ok((prompt_len - rows, rows))
}

fn validate_shared_prefill_extent(enabled: bool, prefix_tokens: usize) -> Result<()> {
    if enabled && prefix_tokens == 0 {
        return Err(invalid(
            "shared-softmax prefill requires an actual batched prefix",
        ));
    }
    Ok(())
}

fn promote_verified_block(
    enabled: bool,
    accepted: usize,
    committed: usize,
    stop: Option<&str>,
) -> bool {
    enabled && accepted == B - 1 && committed == B && stop.is_none()
}

struct Driver {
    #[cfg(feature = "qwen35-session-continuation")]
    session_cancellation: Option<crate::session_cancellation::SessionCancellationToken>,
    stream: Rc<OwnedCudaStream>,
    plans: Vec<DevicePlan>,
    head: DeviceOutputHeadPlan,
    embedding: Weight,
    noise_ids: OwnedCudaBuffer,
    noise_device: OwnedCudaBuffer,
    token: OwnedCudaBuffer,
    features: OwnedCudaBuffer,
    feature_shift_scratch: OwnedCudaBuffer,
    feature_shift_submissions: u64,
    verified_features: Option<OwnedCudaBuffer>,
    batch_verify_audit: Option<BatchedVerifyAudit>,
    batch_verify_execution: Option<BatchedVerifyExecution>,
    round_profile: Option<product_stage_profile::Profile>,
    #[cfg(feature = "qwen35-session-continuation")]
    batched_suffix: bool,
    feature_digest: OwnedCudaBuffer,
    candidate_ids: OwnedCudaBuffer,
    unary: OwnedCudaBuffer,
    transitions: OwnedCudaBuffer,
    path: OwnedCudaBuffer,
    verify_ids: OwnedCudaBuffer,
    baseline_ids: OwnedCudaBuffer,
    acceptance: OwnedCudaBuffer,
    backups: Vec<OwnedCudaBuffer>,
    digest_parts: OwnedCudaBuffer,
    digest: OwnedCudaBuffer,
    producer_event: OwnedCudaEvent,
    target_handoff_event: OwnedCudaEvent,
    cursor: usize,
    capacity: usize,
    context_rows: usize,
    /// Optional request-local logical extent over the fixed physical feature
    /// allocation.  This is opt-in; `None` preserves the legacy fixed window.
    growing_history: Option<GrowingHistory>,
    last_growth_plan: Option<GrowingHistoryPlan>,
    anchor: u32,
    generation: u64,
    identity: [u8; 32],
    #[cfg(feature = "qwen35-session-checkpoint")]
    checkpoint_model_config_sha256: String,
    #[cfg(feature = "qwen35-session-checkpoint")]
    checkpoint_tokenizer_sha256: String,
    #[cfg(feature = "qwen35-session-checkpoint")]
    checkpoint_generation_config_sha256: Option<String>,
    #[cfg(feature = "qwen35-session-checkpoint")]
    checkpoint_reuse_batched_plan: bool,
    #[cfg(feature = "qwen35-session-checkpoint")]
    checkpoint_share_target_rope: bool,
    feature_sha: [u8; 32],
    fast_feature_identity: bool,
    // Only unaudited receipt-driven replay: next anchor comes from verify_ids.
    replay_head_elision: bool,
    parallel_receipt_digest: bool,
    exact_projection_tiles: bool,
    projection_row_reuse: bool,
    ffn_down_row_reuse: bool,
    preserved_silu_q8: bool,
    canonical_prefill_rows: usize,
    tensor_ffn_down_rows: Option<usize>,
    tensor_ffn_down_compensated: bool,
    integer_mma_ffn: Option<usize>,
    integer_mma_input: bool,
    llama_f16_input: bool,
    llama_original_input: bool,
    llama_full_original: bool,
    llama_rotated_kv: bool,
    planar_rotated_kv: bool,
    llama_bf16_scalars: bool,
    llama_history_full: bool,
    llama_global_chunks: bool,
    llama_exact_input_rows: bool,
    prefill_batched_head_rope: bool,
    batched_feature_copies: bool,
    history_batched_feature_copies: bool,
    history_feature_copy_enqueues: std::cell::Cell<u64>,
    batched_checkpoint_copies: bool,
    batched_checkpoint_save_only: bool,
    batched_replay_copies: bool,
    shared_replay_fold_scalars: bool,
    replay_checkpoint_read: bool,
    replay_prefix_family: bool,
    shared_verify_gdn_scalars: bool,
    dflash_history_projection_reuse: bool,
    dflash_b8_specialized: bool,
    dflash_history_weight_panels: bool,
    dflash_history_panel_token8: bool,
    dflash_context_key_norm_rope: bool,
    batched_output_head_prepare: bool,
    head_prepare_stores: bool,
    batched_target_top1: bool,
    verify_rope_reuse: bool,
    output_head_draft_prepare_batches: std::cell::Cell<u64>,
    output_head_target_prepare_batches: std::cell::Cell<u64>,
    target_top1_batches: std::cell::Cell<u64>,
    shared_replay_fold_layers: std::cell::Cell<u64>,
    checkpoint_read_fold_layers: std::cell::Cell<u64>,
    checkpoint_copy_batch_enqueues: std::cell::Cell<u64>,
    checkpoint_copy_logical_ranges: std::cell::Cell<u64>,
    checkpoint_copy_logical_bytes: std::cell::Cell<u64>,
    replay_copy_batch_enqueues: std::cell::Cell<u64>,
    batched_embedding_gather: bool,
    feature_capture_enqueues: std::cell::Cell<u64>,
    feature_append_enqueues: std::cell::Cell<u64>,
    embedding_gather_enqueues: std::cell::Cell<u64>,
    prefill_batched_head_rope_launches: std::cell::Cell<u64>,
    prefill_batched_head_rope_layer_rows: std::cell::Cell<u64>,
    prefix_boundary_audit: Option<batched_verify::prefill_input_integer::PrefixBoundaryAudit>,
    wide_history_capture: bool,
    trajectory_capture: bool,
    exact_output_head_batch: bool,
    draft_top16_batch: bool,
    reference_boundary: Option<reference_boundary::BoundaryTrace>,
    #[cfg(feature = "qwen35-raw-committed-state")]
    raw_state: Option<raw_state::Capture>,
    
    poisoned: bool,
}
impl Drop for Driver {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
    }
}
fn view(b: &OwnedCudaBuffer) -> Result<CudaBufferView> {
    Ok(b.view(0, b.len_bytes())?)
}
fn copy(
    stream: &OwnedCudaStream,
    destination: &OwnedCudaBuffer,
    offset: usize,
    source: CudaBufferView,
) -> Result<()> {
    // SAFETY: callers below retain both owners within Driver/the borrowed draft,
    // use this single stream (after the explicit producer event dependency), and
    // Driver drains even on partial-submit errors before releasing any buffers.
    unsafe {
        destination.copy_from_device_async_unchecked(stream, offset, source)?;
    }
    Ok(())
}
fn words(
    stream: &OwnedCudaStream,
    b: &OwnedCudaBuffer,
    offset: usize,
    count: usize,
) -> Result<Vec<u32>> {
    let mut bytes = vec![0u8; count * 4];
    // Use a bounded owner copy through the existing submission-proof helper.
    let receipt = stream.submission_receipt()?;
    let proof =
        b.copy_range_to_host_sync_with_submission_proof(stream, offset, &mut bytes, receipt)?;
    if !proof.covers_submission(receipt) || !proof.belongs_to_stream(stream)? {
        return Err(invalid("word readback owner epoch"));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}
fn state_range(full: bool, total: usize, cursor: usize, capacity: usize) -> Result<(usize, usize)> {
    if !full {
        if total == 0 {
            return Err(invalid("empty recurrent state"));
        }
        return Ok((0, total));
    }
    if capacity == 0 || total % capacity != 0 || cursor.checked_add(B).is_none_or(|e| e > capacity)
    {
        return Err(invalid("draft target KV transaction extent"));
    }
    let row = total / capacity;
    if row != QWEN35_KV8_CACHE_ROW_BYTES {
        return Err(invalid("draft target requires exact full KV8 row layout"));
    }
    Ok((cursor * row, B * row))
}

#[derive(Clone, Copy, Debug)]
struct FeatureAdvance {
    retained: usize,
    skip_committed: usize,
    growing_plan: Option<GrowingHistoryPlan>,
}

impl FeatureAdvance {
    fn capture_range(self, committed: usize) -> Result<Option<(usize, usize, usize)>> {
        if !(1..=B).contains(&committed) || self.skip_committed > committed
            || self.growing_plan.is_some_and(|p| p.append_rows != committed) {
            return Err(invalid("committed feature contiguous range"));
        }
        let first = self.skip_committed;
        let count = committed - first;
        if count == 0 { return Ok(None); }
        let destination = self.capture_row(first)
            .ok_or_else(|| invalid("committed feature range start"))?;
        if self.capture_row(committed - 1) != destination.checked_add(count - 1) {
            return Err(invalid("committed feature range is not contiguous"));
        }
        Ok(Some((first, destination, count)))
    }

    fn new(context: usize, committed: usize) -> Result<Self> {
        if !(1..=2048).contains(&context) || !(1..=B).contains(&committed) {
            return Err(invalid("committed feature window extent"));
        }
        Ok(Self {
            retained: context.saturating_sub(committed),
            skip_committed: committed.saturating_sub(context),
            growing_plan: None,
        })
    }

    fn capture_row(self, committed_row: usize) -> Option<usize> {
        if let Some(plan) = self.growing_plan {
            return if committed_row < plan.append_rows {
                plan.append_offset.checked_add(committed_row)
            } else {
                None
            };
        }
        committed_row
            .checked_sub(self.skip_committed)
            .map(|row| self.retained + row)
    }
}

impl Driver {
    fn check_session_cancellation(&self, boundary: &'static str) -> Result<()> {
        #[cfg(feature = "qwen35-session-continuation")]
        if let Some(token) = &self.session_cancellation {
            token
                .check(boundary)
                .map_err(|error| invalid(error.to_string()))?;
        }
        #[cfg(not(feature = "qwen35-session-continuation"))]
        let _ = boundary;
        Ok(())
    }

    fn active_context_rows(&self) -> usize {
        self.growing_history
            .map_or(self.context_rows, GrowingHistory::active_rows)
    }

    fn active_context_start(&self) -> usize {
        self.growing_history.map_or(
            self.cursor.saturating_sub(self.context_rows),
            GrowingHistory::start_position,
        )
    }

    fn active_feature_view(&self) -> Result<CudaBufferView> {
        let bytes = self
            .active_context_rows()
            .checked_mul(5 * H * 4)
            .ok_or_else(|| invalid("active feature extent overflow"))?;
        self.features.view(0, bytes).map_err(Into::into)
    }

    fn promote_verified_features(&mut self, committed: usize) -> Result<()> {
        if self.verified_features.is_none() {
            return Err(invalid("missing verified feature staging"));
        }
        let plan = self.advance_feature_window(committed)?;
        let staging = self
            .verified_features
            .as_ref()
            .ok_or_else(|| invalid("missing verified feature staging"))?;
        let row_bytes = 5 * H * 4;
        if self.batched_feature_copies {
            if let Some((first, destination, count)) = plan.capture_range(committed)? {
                copy(&self.stream, &self.features, destination * row_bytes,
                    staging.view(first * row_bytes, count * row_bytes)?)?;
                self.feature_append_enqueues.set(self.feature_append_enqueues.get() + 1);
            }
            return Ok(());
        }
        for row in 0..committed {
            if let Some(destination) = plan.capture_row(row) {
                copy(
                    &self.stream,
                    &self.features,
                    destination * row_bytes,
                    staging.view(row * row_bytes, row_bytes)?,
                )?;
                self.feature_append_enqueues.set(self.feature_append_enqueues.get() + 1);
            }
        }
        Ok(())
    }

    fn forward_verification(&self, row: usize) -> Result<()> {
        self.forward_capturing(
            self.cursor + row,
            self.verified_features.as_ref().map(|owner| (owner, row)),
            Some(row),
        )
    }

    fn advance_feature_window(&mut self, committed: usize) -> Result<FeatureAdvance> {
        if let Some(mut history) = self.growing_history {
            if history.cursor() != self.cursor {
                return Err(invalid("growing history cursor differs from target owner"));
            }
            let plan = history
                .append(committed)
                .map_err(|error| invalid(error.to_string()))?;
            let row_bytes = 5 * H * 4;
            // A growing window with no eviction retains its prefix in place.
            let shift_rows = if plan.evicted_rows == 0 {
                0
            } else {
                plan.retained_rows
            };
            if shift_rows > 0 {
                // SAFETY: target features are exclusively stream-ordered and
                // retained by this driver through its terminal drain.
                unsafe { self.features.shift_rows_u32_async_unchecked(
                    &self.stream, row_bytes, shift_rows, plan.evicted_rows,
                )?; }
                self.feature_shift_submissions += 1;
            }
            self.growing_history = Some(history);
            self.last_growth_plan = Some(plan);
            return Ok(FeatureAdvance {
                retained: plan.retained_rows,
                skip_committed: 0,
                growing_plan: Some(plan),
            });
        }
        let plan = FeatureAdvance::new(self.context_rows, committed)?;
        let row_bytes = 5 * H * 4;
        if plan.retained > 0 && committed > 0 {
            // SAFETY: one thread owns each column and walks forward, so no
            // overlapping memcpy or cross-thread dependency is introduced.
            unsafe { self.features.shift_rows_u32_async_unchecked(
                &self.stream, row_bytes, plan.retained, committed,
            )?; }
            self.feature_shift_submissions += 1;
        }
        Ok(plan)
    }

    fn bind_gather(&self, ids: CudaBufferView) -> Result<CudaBf16TokenRowGatherArgsV1> {
        Ok(CudaBf16TokenRowGatherArgsV1::bind(
            &self.stream,
            CudaBf16WeightViewV1::new(
                view(&self.embedding)?,
                PhysicalLayoutId::DEVICE_CONTIGUOUS_BF16_CUDA_V1,
                V,
                H,
            )?,
            ids,
            self.plans[0].input_f32_view()?,
        )?)
    }
    fn prepare_noise_device(&self, source: CudaBufferView) -> Result<CudaBufferView> {
        let weight = CudaBf16WeightViewV1::new(
            view(&self.embedding)?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_BF16_CUDA_V1,
            V,
            H,
        )?;
        for row in 0..B {
            CudaBf16TokenRowGatherArgsV1::bind(
                &self.stream,
                weight,
                if row == 0 {
                    source
                } else {
                    self.noise_ids.view(row * 4, 4)?
                },
                CudaF32View::new(
                    self.noise_device.view(row * H * 4, H * 4)?,
                    PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                    H,
                )?,
            )?
            .submit_for_plan(&self.stream)?;
        }
        Ok(view(&self.noise_device)?)
    }
    fn forward(&self, position: usize, capture_row: Option<usize>) -> Result<()> {
        self.check_session_cancellation("serial backbone submit")?;
        self.forward_capturing(position, capture_row.map(|row| (&self.features, row)), None)
    }

    fn forward_capturing(
        &self,
        position: usize,
        capture: Option<(&OwnedCudaBuffer, usize)>,
        audit_row: Option<usize>,
    ) -> Result<()> {
        self.forward_backbone_capturing(position, capture, audit_row)?;
        self.head.submit(&self.stream)?;
        Ok(())
    }

    fn replay_committed(&self, position: usize, capture_row: Option<usize>) -> Result<()> {
        if !self.replay_head_elision {
            return self.forward(position, capture_row);
        }
        self.check_session_cancellation("committed replay backbone submit")?;
        // All 64 layers, recurrent/KV writes and feature capture remain exact.
        // Neither logits nor head scratch are recurrent state. The accepted
        // target argmax is already in verify_ids; no top1 consumes this replay.
        // Audits/reference/restore never enable this policy.
        self.forward_backbone_capturing(
            position,
            capture_row.map(|row| (&self.features, row)),
            None,
        )
    }

    fn forward_backbone_capturing(
        &self,
        position: usize,
        capture: Option<(&OwnedCudaBuffer, usize)>,
        audit_row: Option<usize>,
    ) -> Result<()> {
        for layer in 0..64 {
            if layer > 0 {
                self.plans[layer].copy_input_from_device(
                    &self.stream,
                    self.plans[layer - 1].final_hidden_view()?,
                )?;
            }
            self.plans[layer].submit_resident(&self.stream, position)?;
            #[cfg(feature = "qwen35-raw-committed-state")]
            if let Some(capture) = &self.raw_state {
                capture.stage_hidden(
                    self,
                    layer,
                    position,
                    self.plans[layer].final_hidden_view()?,
                )?;
            }
            if let (Some(audit), Some(row)) = (&self.batch_verify_audit, audit_row) {
                audit.capture_serial_layer(self, layer, row)?;
            }
            if let (Some((destination, row)), Some(slot)) = (
                capture,
                DFLASH_TARGET_CAPTURE_LAYER_IDS
                    .iter()
                    .position(|&i| i == layer),
            ) {
                copy(
                    &self.stream,
                    destination,
                    (row * 5 + slot) * H * 4,
                    self.plans[layer].final_hidden_view()?,
                )?;
            }
        }
        Ok(())
    }
    fn top1(&self, destination: CudaBufferView) -> Result<()> {
        CudaQwen35VocabTop1ArgsV1::bind(
            &self.stream,
            plan_f32_view(&self.head.logits, V)?,
            destination,
            1,
        )?
        .submit(&self.stream)?;
        Ok(())
    }
    fn output_head_batching_report(&self) -> Value {
        json!({
            "prepare_requested":self.batched_output_head_prepare,
            "scratch_producer_stores_requested":self.head_prepare_stores,
            "avoided_scratch_copy_calls":if self.head_prepare_stores {
                self.output_head_draft_prepare_batches.get()*2 + self.output_head_target_prepare_batches.get()*3
            } else { 0 },
            "target_top1_requested":self.batched_target_top1,
            "draft_prepare_batches":self.output_head_draft_prepare_batches.get(),
            "target_prepare_batches":self.output_head_target_prepare_batches.get(),
            "target_top1_batches":self.target_top1_batches.get(),
            "extra_gpu_workspace_bytes":0,
            "last_row_scratch_preserved":true,
            "scope":"owner_lifetime_successful_host_batches_not_graph_replay_node_counts"
        })
    }
    fn checkpoint(&self, restore: bool) -> Result<()> {
        if self.batched_checkpoint_copies || (!restore && self.batched_checkpoint_save_only) {
            return self.checkpoint_copy_batch(restore, 0);
        }
        for (layer, plan) in self.plans.iter().enumerate() {
            for (plane, state) in plan.terminal_state_buffers().into_iter().enumerate() {
                let full = matches!(plan, DevicePlan::Full(_));
                let (offset, len) =
                    state_range(full, state.len_bytes(), self.cursor, self.capacity)?;
                let backup = &self.backups[layer * 2 + plane];
                if backup.len_bytes() != len {
                    return Err(invalid("checkpoint owner extent drift"));
                }
                if restore {
                    copy(&self.stream, state, offset, view(backup)?)?;
                } else {
                    copy(&self.stream, backup, 0, state.view(offset, len)?)?;
                }
            }
        }
        Ok(())
    }
    fn restore_for_recorded_prefix(&self, committed: usize) -> Result<()> {
        self.restore_recorded_prefix_impl(committed, false)
    }
    fn restore_recorded_prefix_impl(&self, committed: usize, direct_checkpoint: bool) -> Result<()> {
        if !(1..=B).contains(&committed) {
            return Err(invalid("recorded prefix extent"));
        }
        if self.batched_checkpoint_copies {
            if direct_checkpoint {
                return Err(invalid("direct checkpoint fold excludes batched restore"));
            }
            return self.checkpoint_copy_batch(true, committed);
        }
        for (layer, plan) in self.plans.iter().enumerate() {
            for (plane, state) in plan.terminal_state_buffers().into_iter().enumerate() {
                let full = matches!(plan, DevicePlan::Full(_));
                let (offset, len) =
                    state_range(full, state.len_bytes(), self.cursor, self.capacity)?;
                let backup = &self.backups[layer * 2 + plane];
                if backup.len_bytes() != len {
                    return Err(invalid("fold checkpoint extent drift"));
                }
                // The selected fold reads this immutable linear backup directly.
                // Full KV rejected-tail restoration is unchanged.
                if direct_checkpoint && !full { continue; }
                // Full KV accepted rows already contain canonical verified bytes.
                // Restore only the rejected tail, including physically unused bytes.
                // Linear history/state must instead restart at the checkpoint.
                let skip = if full { committed * (len / B) } else { 0 };
                if skip < len {
                    copy(
                        &self.stream,
                        state,
                        offset + skip,
                        backup.view(skip, len - skip)?,
                    )?;
                }
            }
        }
        Ok(())
    }
    fn checkpoint_copy_batch(&self, restore: bool, committed: usize) -> Result<()> {
        let first = self.backups.first().ok_or_else(|| invalid("checkpoint backup missing"))?;
        let mut copies = [(first, 0, first.view(0, 16)?); 128];
        let mut count = 0;
        let mut logical_bytes = 0_u64;
        for (layer, plan) in self.plans.iter().enumerate() {
            for (plane, state) in plan.terminal_state_buffers().into_iter().enumerate() {
                let full = matches!(plan, DevicePlan::Full(_));
                let (offset, len) = state_range(full, state.len_bytes(), self.cursor, self.capacity)?;
                let backup = self.backups.get(layer * 2 + plane)
                    .ok_or_else(|| invalid("checkpoint backup extent"))?;
                if backup.len_bytes() != len || count >= copies.len() || committed > B {
                    return Err(invalid("checkpoint batch owner/extent drift"));
                }
                let skip = if restore && full { committed * (len / B) } else { 0 };
                if skip == len { continue; }
                copies[count] = if restore {
                    (state, offset + skip, backup.view(skip, len - skip)?)
                } else {
                    (backup, 0, state.view(offset, len)?)
                };
                logical_bytes += copies[count].2.len_bytes() as u64;
                count += 1;
            }
        }
        // SAFETY: Driver retains every plan/backup and drains on error/drop.
        // Ranges use the same transaction geometry as the scalar copies;
        // native admission proves that all copies are independent before work.
        unsafe { OwnedCudaBuffer::copy_independent_batch_async_unchecked(&self.stream, &copies[..count])?; }
        self.checkpoint_copy_batch_enqueues.set(self.checkpoint_copy_batch_enqueues.get() + 1);
        self.checkpoint_copy_logical_ranges.set(self.checkpoint_copy_logical_ranges.get() + count as u64);
        self.checkpoint_copy_logical_bytes.set(self.checkpoint_copy_logical_bytes.get() + logical_bytes);
        Ok(())
    }

    fn receipt_digest(&self, input: &OwnedCudaBuffer, output: CudaBufferView) -> Result<()> {
        if self.parallel_receipt_digest {
            augur_backend::submit_page_digest_parallel_into_view_v1(&self.stream, input, output)?;
        } else {
            submit_page_digest_into_view_v1(&self.stream, input, output)?;
        }
        Ok(())
    }
    fn state_digest(&self) -> Result<String> {
        for (layer, p) in self.plans.iter().enumerate() {
            for (plane, s) in p.terminal_state_buffers().into_iter().enumerate() {
                self.receipt_digest(
                    s,
                    self.digest_parts.view(
                        (layer * 2 + plane) * QWEN35_KV8_PAGE_DIGEST_BYTES,
                        QWEN35_KV8_PAGE_DIGEST_BYTES,
                    )?,
                )?;
            }
        }
        self.receipt_digest(&self.digest_parts, view(&self.digest)?)?;
        let mut bytes = vec![0u8; QWEN35_KV8_PAGE_DIGEST_BYTES];
        self.digest.copy_to_host_sync(&self.stream, &mut bytes)?;
        Ok(sha256_hex(&bytes))
    }
    fn full_context_feature_digest(&mut self) -> Result<usize> {
        if self.active_context_rows() == 0
            || (self.active_context_rows() < B && self.growing_history.is_none())
        {
            return Err(invalid("rolling context must retain a full draft block"));
        }
        // The rolling ring is already complete on the target/DFlash device.
        // Produce only a bounded device receipt instead of copying the newly
        // appended feature rows to the CPU. This is a deterministic device
        // fingerprint, not a cryptographic hash of the feature payload. The
        // optional report SHA wrapper does not strengthen that device digest.
        let feature_view = self.active_feature_view()?;
        if self.parallel_receipt_digest {
            augur_backend::submit_page_digest_parallel_input_view_into_view_v1(
                &self.stream,
                feature_view,
                view(&self.feature_digest)?,
            )?;
        } else {
            augur_backend::submit_page_digest_input_view_into_view_v1(
                &self.stream,
                feature_view,
                view(&self.feature_digest)?,
            )?;
        }
        let receipt = self.stream.submission_receipt()?;
        let mut bytes = [0u8; QWEN35_KV8_PAGE_DIGEST_BYTES];
        let proof = self
            .feature_digest
            .copy_to_host_sync_with_submission_proof(&self.stream, &mut bytes, receipt)?;
        if !proof.covers_submission(receipt) || !proof.belongs_to_stream(&self.stream)? {
            return Err(invalid("rolling feature fingerprint owner epoch"));
        }
        self.feature_sha = bytes;
        Ok(QWEN35_KV8_PAGE_DIGEST_BYTES)
    }

    fn fast_context_feature_identity(&mut self) -> Result<usize> {
        if self.active_context_rows() == 0
            || (self.active_context_rows() < B && self.growing_history.is_none())
        {
            return Err(invalid("rolling context must retain a full draft block"));
        }
        // Ordinary continuation only needs a stable, owner-local binding for
        // the target feature handoff.  The full device digest remains the
        // explicit audit/checkpoint contract; this path avoids rescanning up
        // to 200 MiB of feature rows at every rolling rebind boundary.
        self.feature_sha = fast_feature_identity(
            self.identity,
            self.generation,
            self.cursor,
            self.anchor,
            self.active_context_start(),
            self.active_context_rows(),
            self.capacity,
        );
        Ok(0)
    }

    fn next_context_feature_digest(&mut self) -> Result<usize> {
        if self.fast_feature_identity {
            self.fast_context_feature_identity()
        } else {
            self.full_context_feature_digest()
        }
    }
    fn verify(
        &mut self,
        draft: &DFlashQ8CudaBlock,
        budget: Option<PublicationBudget<'_>>,
    ) -> Result<Value> {
        if self.poisoned
            || draft.submitted_identity().map_err(invalid)?
                != (self.generation, self.cursor, self.anchor, self.feature_sha)
        {
            return Err(invalid(
                "draft does not bind this live target cursor/anchor/features",
            ));
        }
        self.poisoned = true;
        let result = self.verify_inner(draft, budget);
        if result.is_ok() {
            self.poisoned = false;
        }
        result
    }

    fn score_draft_head(&self, draft: &DFlashQ8CudaBlock) -> Result<()> {
        if let Some(executor) = &self.batch_verify_execution {
            if executor.submit_draft_head(self, draft)? {
                return Ok(());
            }
        }
        // Draft hidden already passed DFlash final norm; never apply target norm.
        for row in 0..B {
            copy(
                &self.stream,
                &self.head.normalized_hidden,
                0,
                draft
                    .final_hidden_row_device(row)
                    .map_err(invalid)?
                    .buffer_view(),
            )?;
            self.head.quantize_op.submit_for_plan(&self.stream)?;
            self.head.projection_op.submit_for_plan(&self.stream)?;
            CudaQwen35VocabTopkArgsV1::bind(
                &self.stream,
                plan_f32_view(&self.head.logits, V)?,
                self.candidate_ids.view(row * 64, 64)?,
                self.unary.view(row * 64, 64)?,
                1,
                16,
            )?
            .submit(&self.stream)?;
        }
        Ok(())
    }

    /// Receipt-driven product verify.  The GPU still performs candidate
    /// top-k, selector routing, target verification, acceptance, rollback and
    /// accepted-prefix replay, but the host does not read the full draft path,
    /// target ids, baseline ids, or state digests. Only the 64-byte acceptance
    /// receipt and the accepted token prefix (including the bonus when all
    /// seven drafts are accepted; at most 32 bytes) cross the boundary here.
    fn mark_round_profile(&self, index: usize) -> Result<()> {
        if let Some(profile) = &self.round_profile {
            profile.mark(1, index, &self.stream)?;
        }
        Ok(())
    }

    fn verify_product(
        &mut self,
        draft: &DFlashQ8CudaBlock,
        budget: Option<PublicationBudget<'_>>,
    ) -> Result<Value> {
        if self.poisoned
            || draft.submitted_identity().map_err(invalid)?
                != (self.generation, self.cursor, self.anchor, self.feature_sha)
        {
            return Err(invalid(
                "draft does not bind this live target cursor/anchor/features",
            ));
        }
        self.poisoned = true;
        let result = (|| {
            let mut before = cuda_host_boundary_snapshot();
            let alloc_before = cuda_device_allocation_snapshot();
            let start = Instant::now();
            self.mark_round_profile(0)?;
            
            self.checkpoint(false)?;
            self.mark_round_profile(1)?;
            
            let checkpoint_save_enqueue_ms = start.elapsed().as_secs_f64() * 1000.0;
            let producer_dependency_start = Instant::now();
            let producer_receipt = draft.stream().submission_receipt()?;
            self.producer_event
                .record_for_submission(draft.stream(), producer_receipt)?;
            self.stream.wait_event(&self.producer_event)?;
            self.mark_round_profile(2)?;
            
            let producer_dependency_enqueue_ms =
                producer_dependency_start.elapsed().as_secs_f64() * 1000.0;
            let target_verify_start = Instant::now();
            self.score_draft_head(draft)?;
            self.mark_round_profile(3)?;
            let (gates, predecessor, successor) =
                draft.selector_inputs_device().map_err(invalid)?;
            CudaQwen35DflashQ8SelectorArgsV1::bind(
                &self.stream,
                CudaQwen35DflashQ8SelectorViewsV1 {
                    predecessor,
                    successor,
                    gates,
                    candidate_ids: view(&self.candidate_ids)?,
                    unary_scores: view(&self.unary)?,
                    transition_scores: view(&self.transitions)?,
                    path_ids: view(&self.path)?,
                },
                self.anchor,
            )?
            .submit(&self.stream)?;
            
            self.mark_round_profile(4)?;
            let shadow_probe = if let Some(audit) = &self.batch_verify_audit {
                let probe = audit.probe(self)?;
                before = cuda_host_boundary_snapshot();
                audit.begin_serial(self)?;
                Some(probe)
            } else {
                None
            };
            let mut batch_execution = if let Some(executor) = &self.batch_verify_execution {
                Some(executor.submit(self)?)
            } else {
                for row in 0..B {
                    self.bind_gather(self.path.view(row * 4, 4)?)?
                        .submit_for_plan(&self.stream)?;
                    self.forward_verification(row)?;
                    if let Some(audit) = &self.batch_verify_audit {
                        audit.capture_serial(self, row)?;
                    }
                    self.top1(self.verify_ids.view(row * 4, 4)?)?;
                }
                None
            };
            if let Some(audit) = &self.batch_verify_audit {
                audit.end_serial(self)?;
            }
            self.mark_round_profile(5)?;
            CudaQwen35DflashAcceptanceArgsV1::bind(
                &self.stream,
                CudaQwen35DflashAcceptanceViewsV1 {
                    draft_ids: self.path.view(4, 7 * 4)?,
                    target_ids: self.verify_ids.view(0, 7 * 4)?,
                    receipt: view(&self.acceptance)?,
                },
                7,
                self.generation,
                self.identity,
            )?
            .submit(&self.stream)?;
            
            self.mark_round_profile(6)?;
            let target_verify_enqueue_ms = target_verify_start.elapsed().as_secs_f64() * 1000.0;
            let submit_ms = start.elapsed().as_secs_f64() * 1000.0;
            let submit_boundaries = boundary_delta(before, cuda_host_boundary_snapshot())?;
            let alloc_after = cuda_device_allocation_snapshot();
            let acceptance_receipt_start = Instant::now();
            let mut receipt = vec![0u8; QWEN35_DFLASH_ACCEPTANCE_RECEIPT_BYTES];
            self.acceptance
                .copy_to_host_sync(&self.stream, &mut receipt)?;
            let acceptance_receipt_readback_ms =
                acceptance_receipt_start.elapsed().as_secs_f64() * 1000.0;
            let receipt_ms = start.elapsed().as_secs_f64() * 1000.0;
            // Explicit diagnostic observation belongs AFTER the enqueue-only
            // boundary and acceptance receipt. Never exempt its D2H from the
            // submission contract by weakening that contract's checks.
            let verify_logits_observed = self.batch_verify_execution.as_ref()
                .map_or(Ok(false), |executor| executor.observe_verify_logits(self))?;
            if verify_logits_observed {
                if let Some(report) = &mut batch_execution {
                    report["verify_logits_diagnostic"] = json!({"observed":true,
                        "host_serialized":true,"performance_evidence":false});
                }
            }
            let trajectory_observation = if self.trajectory_capture {
                Some(trajectory::observe_verification(self)?)
            } else { None };
            let round_profile_report = self.round_profile.as_ref()
                .map(|profile| profile.finish(self, draft)).transpose()?;
            if let (Some(executor), Some(report)) =
                (&self.batch_verify_execution, &mut batch_execution)
            {
                executor.finish_profile(self, report)?;
            }
            let decision = parse_cuda_qwen35_dflash_acceptance_receipt_v1(
                &receipt,
                self.generation,
                self.identity,
            )?;
            let accepted = decision.accepted_tokens as usize;
            let token_publication_start = Instant::now();
            let publication_before = cuda_host_boundary_snapshot();
            let mut emitted_token_ids = if accepted > 0 {
                words(&self.stream, &self.path, 4, accepted)?
            } else {
                Vec::new()
            };
            let bonus = if accepted == 7 {
                words(&self.stream, &self.verify_ids, 7 * 4, 1)?[0]
            } else {
                decision.replacement_token
            };
            if bonus as usize >= V {
                return Err(invalid("invalid bonus token"));
            }
            emitted_token_ids.push(bonus);
            let stop_reason = clip_publication(&mut emitted_token_ids, budget)?;
            let committed = emitted_token_ids.len();
            let next_anchor = *emitted_token_ids
                .last()
                .ok_or_else(|| invalid("missing published anchor"))?;
            let promote = promote_verified_block(
                self.verified_features.is_some(),
                accepted,
                committed,
                stop_reason,
            );
            // The replacement token for accepted < 7 is already carried by
            // the 64-byte acceptance receipt.  Only the accepted prefix is
            // copied from `path`; a full seven-token acceptance additionally
            // copies the target bonus token from `verify_ids`.
            let published_token_d2h_words = accepted
                .checked_add(usize::from(accepted == 7))
                .ok_or_else(|| invalid("published token word overflow"))?;
            let published_token_d2h_bytes: u64 = (published_token_d2h_words as u64)
                .checked_mul(core::mem::size_of::<u32>() as u64)
                .ok_or_else(|| invalid("published token byte overflow"))?;
            let expected_publication_copies: u64 = if accepted == 0 {
                0
            } else {
                (1 + usize::from(accepted == 7)) as u64
            };
            let publication_boundaries =
                boundary_delta(publication_before, cuda_host_boundary_snapshot())?;
            if publication_boundaries.synchronous_d2h_bytes != published_token_d2h_bytes
                || publication_boundaries.synchronous_d2h_copies != expected_publication_copies
                || publication_boundaries.synchronous_h2d_bytes != 0
                || publication_boundaries.synchronous_h2d_copies != 0
                || publication_boundaries.host_waits != expected_publication_copies
            {
                return Err(invalid("unexpected token publication boundary"));
            }
            let token_publication_readback_ms =
                token_publication_start.elapsed().as_secs_f64() * 1000.0;
            let shadow_report = match (&self.batch_verify_audit, shadow_probe) {
                (Some(audit), Some(probe)) => Some(audit.finish(self, probe)?),
                (None, None) => None,
                _ => return Err(invalid("shadow batch audit pairing")),
            };

            // Restore the pre-verify target state before committing the
            // receipt-selected prefix.  The product path intentionally does
            // not run the normal-greedy control sequence here: that control
            // is a semantic audit concern and remains in `verify_inner`.
            // Running it in the receipt-driven product path would execute up
            // to eight extra full target forwards per transaction while the
            // report explicitly says `host_audit_skipped=true` and keeps all
            // semantic/product claims closed.
            let rollback_replay_start = Instant::now();
            
            let folded = if !promote {
                self.batch_verify_execution
                    .as_ref()
                    .map(|executor| executor.fold_verified_prefix(self, committed))
                    .transpose()?
                    .unwrap_or(false)
            } else {
                false
            };
            let fold_audit = if folded {
                self.batch_verify_execution
                    .as_ref()
                    .ok_or_else(|| invalid("ReplaySSM executor missing"))?
                    .audit_folded_prefix(self, committed)?
            } else {
                None
            };
            if promote || folded {
                // Full promotion or prefix state fold reached the exact input
                // boundary; matching features come from target verification.
                self.promote_verified_features(committed)?;
            } else {
                self.checkpoint(true)?;
                let feature_advance = self.advance_feature_window(committed)?;
                for row in 0..committed {
                    self.bind_gather(self.path.view(row * 4, 4)?)?
                        .submit_for_plan(&self.stream)?;
                    self.replay_committed(self.cursor + row, feature_advance.capture_row(row))?;
                }
            }
            // The accepted-prefix target argmax is exactly the next anchor.
            // Do not read the audit-only baseline buffer on the product path.
            copy(
                &self.stream,
                &self.token,
                0,
                self.verify_ids.view((committed - 1) * 4, 4)?,
            )?;
            let from = self.cursor;
            self.cursor += committed;
            self.anchor = next_anchor;
            let generation = self.generation;
            self.generation = self
                .generation
                .checked_add(1)
                .ok_or_else(|| invalid("draft target generation overflow"))?;
            let boundary_ok = submit_boundaries.synchronous_h2d_bytes == 0
                && submit_boundaries.synchronous_d2h_bytes == 0
                && submit_boundaries.host_waits == 0
                && alloc_after.successful_allocations == alloc_before.successful_allocations
                && alloc_after.successful_frees == alloc_before.successful_frees;
            if !boundary_ok {
                return Err(invalid(
                    "unexpected target verify submission host work/allocation",
                ));
            }
            
            let rollback_replay_enqueue_ms = rollback_replay_start.elapsed().as_secs_f64() * 1000.0;
            let product_ms = start.elapsed().as_secs_f64() * 1000.0;
            let mut report = json!({
                "pass": true,
                "generation_before": generation,
                "generation_after": self.generation,
                "cursor_before": from,
                "cursor_after": self.cursor,
                "accepted_drafts": accepted,
                "emitted_token_count": committed,
                "committed_token_count": committed,
                "next_anchor": next_anchor,
                "publication_stop_reason": stop_reason,
                "verified_state_promotion_eligible": promote,
                "verified_state_promoted": promote,
                "batched_verification_shadow":shadow_report,
                "batched_verification_execution":batch_execution,
                "verify_submission_scope":if self.batch_verify_execution.is_some(){"draft_head_selector_and_canonical_batch"}else if self.batch_verify_audit.is_some() { "canonical_serial_after_shadow_restore" } else { "draft_head_selector_and_canonical_serial" },
                "commit_strategy": if promote { "retain_verified_full_block" } else if folded { "restore_fold_recorded_input_prefix" } else { "restore_replay_committed_prefix" },
                "replacement_or_bonus_token": bonus,
                "emitted_token_ids": emitted_token_ids,
                "published_token_d2h_bytes": published_token_d2h_bytes,
                "product_receipt_driven": true,
                "host_audit_skipped": true,
                "verify_submit_boundaries": submit_boundaries,
                "verify_submit_allocation_stable": true,
                "verify_host_submit_ms": submit_ms,
                "verify_submit_to_acceptance_receipt_ms": receipt_ms,
                "product_verify_wall_ms": product_ms,
                "transaction_timing_scope": "host_enqueue_and_receipt_excludes_commit_tail",
                "target_forward_counts": {"verify": B, "audit": if fold_audit.is_some() { committed } else { 0 }, "commit_replay": if promote || folded { 0 } else { committed }},
                "replay_ssm_folded_inputs": if folded { committed } else { 0 },
                "replay_ssm_audit": fold_audit,
                "replay_output_head_elision": {
                    "enabled": self.replay_head_elision,
                    "skipped_heads": if self.replay_head_elision && !promote && !folded { committed } else { 0 },
                    "anchor_source": "verified_target_ids",
                    "backbone_and_feature_capture_preserved": !folded,
                },
                // These are host-observed enqueue/readback intervals. They
                // deliberately do not claim isolated GPU kernel time: the
                // stream is asynchronous except at the bounded receipt and
                // token publication readbacks above.
                "phase_wall_ms": {
                    "checkpoint_save_enqueue_ms": checkpoint_save_enqueue_ms,
                    "producer_dependency_enqueue_ms": producer_dependency_enqueue_ms,
                    "target_verify_enqueue_ms": target_verify_enqueue_ms,
                    "acceptance_receipt_readback_ms": acceptance_receipt_readback_ms,
                    "token_publication_readback_ms": token_publication_readback_ms,
                    "rollback_replay_enqueue_ms": rollback_replay_enqueue_ms,
                },
                "token_publication_boundaries": publication_boundaries,
                "acceptance_receipt_bytes": QWEN35_DFLASH_ACCEPTANCE_RECEIPT_BYTES,
                "bonus_token_d2h_bytes": if accepted == 7 { 4 } else { 0 },
                "audit_ids_d2h_bytes": 0,
                "rollback_policy": if promote { "retain_verified_full_block_device_only" } else if folded { "restore_rejected_kv_tail_and_fold_recorded_gdn_prefix_device_only" } else { "checkpoint_restore_then_replay_accepted_inputs_device_only" },
                "semantic_control_scope": "GPU receipt-driven rollback/replay; full CPU token audit deferred",
                "product_ready": false,
                "rolling_decode": true,
                "visible_tps": false,
            });
            if let Some(profile) = round_profile_report {
                report["round_stage_profile"] = profile;
                report["event_instrumented_diagnostic_only"] = json!(true);
                report["performance_eligible"] = json!(false);
            }
            if let Some(observation) = trajectory_observation {
                report["trajectory_verification"] = observation;
                report["performance_eligible"] = json!(false);
            }
            Ok(report)
        })();
        if result.is_ok() {
            self.poisoned = false;
        }
        result
    }

    fn verify_inner(
        &mut self,
        draft: &DFlashQ8CudaBlock,
        budget: Option<PublicationBudget<'_>>,
    ) -> Result<Value> {
        let base_digest = self.state_digest()?;
        let before = cuda_host_boundary_snapshot();
        let alloc_before = cuda_device_allocation_snapshot();
        let start = Instant::now();
        self.checkpoint(false)?;
        let producer_receipt = draft.stream().submission_receipt()?;
        self.producer_event
            .record_for_submission(draft.stream(), producer_receipt)?;
        self.stream.wait_event(&self.producer_event)?;
        self.score_draft_head(draft)?;
        let (gates, predecessor, successor) = draft.selector_inputs_device().map_err(invalid)?;
        CudaQwen35DflashQ8SelectorArgsV1::bind(
            &self.stream,
            CudaQwen35DflashQ8SelectorViewsV1 {
                predecessor,
                successor,
                gates,
                candidate_ids: view(&self.candidate_ids)?,
                unary_scores: view(&self.unary)?,
                transition_scores: view(&self.transitions)?,
                path_ids: view(&self.path)?,
            },
            self.anchor,
        )?
        .submit(&self.stream)?;
        // Each target input is the actual GPU draft token, not target argmax or
        // a teacher-forced reference. Its logits predict the following position.
        for row in 0..B {
            self.bind_gather(self.path.view(row * 4, 4)?)?
                .submit_for_plan(&self.stream)?;
            self.forward_verification(row)?;
            self.top1(self.verify_ids.view(row * 4, 4)?)?;
        }
        CudaQwen35DflashAcceptanceArgsV1::bind(
            &self.stream,
            CudaQwen35DflashAcceptanceViewsV1 {
                draft_ids: self.path.view(4, 7 * 4)?,
                target_ids: self.verify_ids.view(0, 7 * 4)?,
                receipt: view(&self.acceptance)?,
            },
            7,
            self.generation,
            self.identity,
        )?
        .submit(&self.stream)?;
        let submit_ms = start.elapsed().as_secs_f64() * 1000.0;
        let submit_boundaries = boundary_delta(before, cuda_host_boundary_snapshot())?;
        let alloc_after = cuda_device_allocation_snapshot();
        let mut receipt = vec![0u8; QWEN35_DFLASH_ACCEPTANCE_RECEIPT_BYTES];
        self.acceptance
            .copy_to_host_sync(&self.stream, &mut receipt)?;
        let verify_ms = start.elapsed().as_secs_f64() * 1000.0;
        let decision = parse_cuda_qwen35_dflash_acceptance_receipt_v1(
            &receipt,
            self.generation,
            self.identity,
        )?;
        let accepted = decision.accepted_tokens as usize;
        let bonus = if accepted == 7 {
            words(&self.stream, &self.verify_ids, 7 * 4, 1)?[0]
        } else {
            decision.replacement_token
        };
        if bonus as usize >= V {
            return Err(invalid("invalid bonus token"));
        }
        // Diagnostic audit: full draft ids stay off the normal publication path.
        let path = words(&self.stream, &self.path, 0, B)?;
        let target = words(&self.stream, &self.verify_ids, 0, B)?;
        let cpu_accepted = path[1..]
            .iter()
            .zip(&target)
            .take_while(|(a, b)| a == b)
            .count();
        if cpu_accepted != accepted {
            return Err(invalid(
                "GPU acceptance count disagrees with audited prefix",
            ));
        }
        let mut emitted = path[1..1 + accepted]
            .iter()
            .copied()
            .chain([bonus])
            .collect::<Vec<_>>();
        let stop_reason = clip_publication(&mut emitted, budget)?;
        let committed = emitted.len();
        let next_anchor = *emitted
            .last()
            .ok_or_else(|| invalid("missing audit anchor"))?;
        let promote = promote_verified_block(
            self.verified_features.is_some(),
            accepted,
            committed,
            stop_reason,
        );
        // Audit captures the actual verified state before restoring it, then
        // keeps the independent greedy + replay oracle. No second state owner
        // is allocated merely to retain the optimized path through an audit.
        let verified_digest = if promote {
            Some(self.state_digest()?)
        } else {
            None
        };
        self.checkpoint(true)?;
        let restored_digest = self.state_digest()?;
        if restored_digest != base_digest {
            return Err(invalid("physical target rollback did not restore base"));
        }
        // Same-owner normal greedy continuation after restoring all semantic
        // state. This is the semantic control, never a source for draft inputs.
        copy(&self.stream, &self.token, 0, self.path.view(0, 4)?)?;
        let mut expected_commit_digest = String::new();
        for row in 0..B {
            self.bind_gather(view(&self.token)?)?
                .submit_for_plan(&self.stream)?;
            // Audit-only future tokens must never enter the committed feature window.
            self.forward(self.cursor + row, None)?;
            self.top1(self.baseline_ids.view(row * 4, 4)?)?;
            copy(
                &self.stream,
                &self.token,
                0,
                self.baseline_ids.view(row * 4, 4)?,
            )?;
            if row + 1 == committed {
                expected_commit_digest = self.state_digest()?;
            }
        }
        let baseline = words(&self.stream, &self.baseline_ids, 0, B)?;
        if emitted != baseline[..committed] {
            return Err(invalid(
                "accepted draft output differs from real normal greedy continuation",
            ));
        }
        if verified_digest
            .as_ref()
            .is_some_and(|digest| digest != &expected_commit_digest)
        {
            return Err(invalid(
                "full verified state differs from greedy committed state",
            ));
        }
        self.checkpoint(true)?;
        let feature_advance = self.advance_feature_window(committed)?;
        // Restore+replay is intentionally first correctness implementation.
        // It avoids retaining eight copies of the DeltaNet recurrent state.
        for row in 0..committed {
            self.bind_gather(self.path.view(row * 4, 4)?)?
                .submit_for_plan(&self.stream)?;
            self.forward(self.cursor + row, feature_advance.capture_row(row))?;
        }
        copy(
            &self.stream,
            &self.token,
            0,
            self.verify_ids.view((committed - 1) * 4, 4)?,
        )?;
        let committed_digest = self.state_digest()?;
        if committed_digest != expected_commit_digest {
            return Err(invalid(
                "accepted-prefix state differs from normal greedy state",
            ));
        }
        let from = self.cursor;
        self.cursor += committed;
        self.anchor = next_anchor;
        let generation = self.generation;
        self.generation = self
            .generation
            .checked_add(1)
            .ok_or_else(|| invalid("draft target generation overflow"))?;
        let boundary_ok = submit_boundaries.synchronous_h2d_bytes == 0
            && submit_boundaries.synchronous_d2h_bytes == 0
            && submit_boundaries.host_waits == 0
            && alloc_after.successful_allocations == alloc_before.successful_allocations
            && alloc_after.successful_frees == alloc_before.successful_frees;
        if !boundary_ok {
            return Err(invalid(
                "unexpected target verify submission host work/allocation",
            ));
        }
        Ok(
            json!({"pass":true,"generation_before":generation,"generation_after":self.generation,
            "cursor_before":from,"cursor_after":self.cursor,"accepted_drafts":accepted,
            "committed_token_count":committed,"next_anchor":next_anchor,"publication_stop_reason":stop_reason,
            "verified_state_promotion_eligible":promote,"verified_state_promoted":false,
            "verified_state_digest":verified_digest,"commit_strategy":"audit_reference_replay",
            "replacement_or_bonus_token":bonus,"emitted_tokens":emitted,"gpu_draft_path":path,"draft_conditioned_target_argmax":target,
            "normal_greedy_control":baseline,"base_state_digest":base_digest,"restored_state_digest":restored_digest,
            "committed_state_digest":committed_digest,"normal_greedy_committed_state_digest":expected_commit_digest,
            "shared_head":"target_resident_GGUF_Q8_0_weight_Q8_1_activation_no_second_norm_no_duplicate_head",
            "verify_submit_boundaries":submit_boundaries,"verify_submit_allocation_stable":true,
            "verify_host_submit_ms":submit_ms,"verify_submit_to_acceptance_receipt_ms":verify_ms,
            "acceptance_receipt_bytes":QWEN35_DFLASH_ACCEPTANCE_RECEIPT_BYTES,
            "state_digest_readback_bytes_per_probe":QWEN35_KV8_PAGE_DIGEST_BYTES,
            "audit_ids_d2h_bytes":3*B*4,"rollback_policy":"checkpoint_restore_then_replay_accepted_inputs",
            "checkpoint_bytes":self.backups.iter().map(|b|b.len_bytes()).sum::<usize>(),
            "semantic_control_scope":"one_draft_block_matches_same_owner_greedy_emission_and_committed_state",
            "product_ready":false,"rolling_decode":false,"visible_tps":false}),
        )
    }
}

/// Real target/Q8/selector/verify/physical rollback transaction diagnostic.
/// No full target logits or semantic-state payloads are copied to the CPU.
pub fn run_qwen35_q8_draft_target_verify_cuda(
    config: &Qwen35AutoregressiveCudaConfig,
    q8_path: &Path,
    draft_config_path: &Path,
    context_rows: usize,
) -> Result<Value> {
    run_qwen35_q8_draft_target_verify_rolling_cuda(
        config,
        q8_path,
        draft_config_path,
        context_rows,
        1,
    )
}

/// Same-owner rolling diagnostic. `rolling_blocks > 1` reuses one resident
/// Q8 DFlash owner, advances the target cursor/generation, and rebinds only the
/// sliding feature ring plus the next anchor. Rebinding is valid after either
/// a full draft acceptance or a partial rejection: `verify()` has already
/// restored and replayed the accepted prefix plus replacement token. It
/// remains non-product because feature snapshots are still host-audited
/// between blocks.
pub fn run_qwen35_q8_draft_target_verify_rolling_cuda(
    config: &Qwen35AutoregressiveCudaConfig,
    q8_path: &Path,
    draft_config_path: &Path,
    context_rows: usize,
    rolling_blocks: usize,
) -> Result<Value> {
    run_qwen35_q8_draft_target_verify_rolling_cuda_with_vision(
        config,
        q8_path,
        draft_config_path,
        context_rows,
        rolling_blocks,
        None,
    )
}

/// Same-owner rolling diagnostic with an optional physical Vision owner.
/// The Vision payload is loaded and kept resident for the complete target /
/// DFlash transaction, but no Vision forward is executed by this runner.
pub fn run_qwen35_q8_draft_target_verify_rolling_cuda_with_vision(
    config: &Qwen35AutoregressiveCudaConfig,
    q8_path: &Path,
    draft_config_path: &Path,
    context_rows: usize,
    rolling_blocks: usize,
    vision_safetensors_path: Option<&Path>,
) -> Result<Value> {
    run_qwen35_q8_draft_target_verify_rolling_cuda_with_vision_mode(
        config,
        q8_path,
        draft_config_path,
        context_rows,
        rolling_blocks,
        vision_safetensors_path,
        false,
        false,
        false,
        None,
        None,
    )
}

/// Product-shaped rolling entry point.  It uses the same target-conditioned
/// GPU verify owner as the diagnostic path but publishes only compact receipt
/// data to the host; semantic parity and visible-TPS claims remain closed.
pub fn run_qwen35_q8_draft_target_product_rolling_cuda_with_vision(
    config: &Qwen35AutoregressiveCudaConfig,
    q8_path: &Path,
    draft_config_path: &Path,
    context_rows: usize,
    rolling_blocks: usize,
    vision_safetensors_path: Option<&Path>,
) -> Result<Value> {
    run_qwen35_q8_draft_target_verify_rolling_cuda_with_vision_mode(
        config,
        q8_path,
        draft_config_path,
        context_rows,
        rolling_blocks,
        vision_safetensors_path,
        true,
        false,
        false,
        None,
        None,
    )
}

#[derive(Clone, Debug)]
pub struct Qwen35DraftTargetRequestOptions {
    pub generation_config_path: PathBuf,
    pub max_new_tokens: usize,
    /// Explicit diagnostic profile; legacy output/round caps remain unchanged.
    pub long_generation_budget: bool,
    pub retain_full_accept_state: bool,
    pub audit_batched_target_verify: bool,
    pub profile_batched_target_verify: bool,
    pub native_quantized_batch_ffn: bool,
    pub audit_batched_numerics: bool,
    pub canonical_batched_gdn_scalars: bool,
    pub native_quantized_batch_attention: bool,
    pub canonical_batched_norm: bool,
    pub canonical_batched_full_core: bool,
    pub batched_target_verify: bool,
    pub audit_committed_state: bool,
    pub reuse_batched_plan: bool,
    pub share_target_rope: bool,
    pub prefill_shared_softmax: bool,
    /// Default-off bounded FP64 score GEMM; never changes draft/decode math.
    pub prefill_dgemm_scores: bool,
    pub batched_prompt_capture: bool,
    pub profile_prefill_sparse: bool,
    /// Bound verification lookahead by the request's publication limit.
    /// This does not change the independent rolling-round budget.
    pub request_position_budget: bool,
    pub batched_draft_context: bool,
    pub batched_draft_block: bool,
    pub draft_q8_tile4: bool,
    /// Same V1 byte fingerprints, bounded multi-CTA implementation; opt-in.
    pub parallel_receipt_digest: bool,
    /// Default-off exact scheduling, canonical batched backbone/capture only.
    pub exact_projection_tiles: bool,
    /// Default-off packed projection of all eight draft/target output-head rows.
    pub exact_output_head_batch: bool,
    /// Diagnostic oracle: same conditioning, independent serial target decode.
    pub target_only_reference: bool,
    /// Diagnostic-only logits at output indices 0 and 27; never a timed path.
    pub audit_reference_logits: bool,
    /// Reference-only prefix execution ablation; the captured tail is unchanged.
    pub reference_serial_prefix: bool,
    /// Reference-only final prompt boundaries; contains synchronous audit reads.
    pub reference_boundary_directory: Option<PathBuf>,
    /// Explicit acknowledgement of the isolated ties-away activation build.
    pub diagnostic_q8_away: bool,
    /// Separate numerical identity; never reuse adjacent-RoPE KV evidence.
    pub diagnostic_split_rope: bool,
    /// Versioned target layout for normal matched target/DFlash requests.
    pub target_split_rope: bool,
}

#[derive(Debug)]
struct RollingPositionBudget {
    policy: &'static str,
    reserved_positions: usize,
    end: usize,
}

#[derive(Clone, Copy)]
enum RollingHistoryBudget {
    Fixed,
    GrowingInitialized { active_rows: usize },
}

fn rolling_position_budget(
    cursor: usize,
    rolling_blocks: usize,
    max_new_tokens: Option<usize>,
    capacity: usize,
    bounded: bool,
) -> Result<RollingPositionBudget> {
    rolling_position_budget_with_profile(
        cursor,
        rolling_blocks,
        max_new_tokens,
        capacity,
        bounded,
        false,
    )
}

fn rolling_position_budget_with_profile(
    cursor: usize,
    rolling_blocks: usize,
    max_new_tokens: Option<usize>,
    capacity: usize,
    bounded: bool,
    long_generation: bool,
) -> Result<RollingPositionBudget> {
    rolling_position_budget_with_history(
        cursor,
        rolling_blocks,
        max_new_tokens,
        capacity,
        bounded,
        long_generation,
        RollingHistoryBudget::Fixed,
    )
}

fn rolling_position_budget_with_history(
    cursor: usize,
    rolling_blocks: usize,
    max_new_tokens: Option<usize>,
    capacity: usize,
    bounded: bool,
    long_generation: bool,
    history: RollingHistoryBudget,
) -> Result<RollingPositionBudget> {
    let initialized_growing = match history {
        RollingHistoryBudget::Fixed => false,
        RollingHistoryBudget::GrowingInitialized { active_rows } => {
            // Physical capacity is not initialized input. Bind admission to the
            // actual cursor/window; never pretend a short input has 2048 rows.
            if !long_generation || cursor == 0 || active_rows != cursor.min(2048) {
                return Err(invalid("growing budget requires exact initialized history"));
            }
            true
        }
    };
    let (round_limit, output_limit) = if long_generation {
        (1024, 1024)
    } else {
        (64, 512)
    };
    if !(1..=round_limit).contains(&rolling_blocks)
        || max_new_tokens.is_some_and(|n| !(1..=output_limit).contains(&n))
    {
        return Err(invalid("invalid rolling position budget"));
    }
    if long_generation
        && (!bounded
            || capacity != 196608
            || (!initialized_growing && cursor < 2048)
            || max_new_tokens.is_none_or(|limit| rolling_blocks < limit))
    {
        return Err(invalid(
            "long generation requires bounded 192K capacity and worst-case round coverage",
        ));
    }
    let block_positions = rolling_blocks
        .checked_mul(B)
        .ok_or_else(|| invalid("block overflow"))?;
    let reserved_positions = if bounded {
        let limit =
            max_new_tokens.ok_or_else(|| invalid("request position budget requires a request"))?;
        // The initial anchor is already published. Before any nonterminal
        // verification there are at most limit-1 published tokens, hence at
        // most limit-2 committed inputs. Still reserve all B verification rows.
        // Even a first-anchor stop constructs a B-row workspace, so keep B.
        block_positions.min(B.max(limit + B - 2))
    } else {
        block_positions
    };
    let end = cursor
        .checked_add(reserved_positions)
        .filter(|&end| end <= capacity && capacity <= 262144)
        .ok_or_else(|| invalid("rolling position budget exceeds capacity"))?;
    Ok(RollingPositionBudget {
        policy: if initialized_growing {
            "growing_initialized_max_new_tokens_full_verify_v1"
        } else if bounded {
            "max_new_tokens_full_verify_v1"
        } else {
            "rolling_blocks_x8_v1"
        },
        reserved_positions,
        end,
    })
}

fn validate_batch_execution_options(
    product: bool,
    request: &Qwen35DraftTargetRequestOptions,
) -> Result<()> {
    if request.long_generation_budget
        && (!product
            || !request.batched_target_verify
            || !request.retain_full_accept_state
            || !request.batched_prompt_capture
            || !request.request_position_budget
            || !request.share_target_rope
            || !request.target_split_rope
            || !request.prefill_dgemm_scores
            || request.target_only_reference
            || request.profile_prefill_sparse)
    {
        return Err(invalid("long generation requires canonical split/shared RoPE, DGEMM prefix, full-accept state and bounded request"));
    }
    if request.prefill_dgemm_scores
        && (!product
            || !request.prefill_shared_softmax
            || request.reference_serial_prefix
            || (request.target_only_reference && !raw_capture_enabled())
            || request.profile_prefill_sparse
            || !cfg!(feature = "qwen35-kv8-chunk-prefill"))
    {
        return Err(invalid("DGEMM scores require product-shaped shared batched prefix without reference/profile modes"));
    }
    if request.reference_boundary_directory.is_some()
        && (!request.target_only_reference
            || !request.batched_prompt_capture
            || !request.audit_reference_logits)
    {
        return Err(invalid(
            "boundary trace requires batched target-only reference and logits audit",
        ));
    }
    if request.reference_serial_prefix
        && (!request.target_only_reference || request.prefill_shared_softmax)
    {
        return Err(invalid(
            "serial prefix requires target-only reference without batched-prefix tuning",
        ));
    }
    if request.audit_reference_logits && !request.target_only_reference {
        return Err(invalid(
            "reference logits audit requires target-only reference",
        ));
    }
    if request.target_only_reference
        && (!product
            || !request.audit_committed_state
            || request.profile_prefill_sparse
            || request.draft_q8_tile4
            || request.batched_draft_context
            || request.batched_draft_block
            || request.audit_batched_target_verify
            || request.reuse_batched_plan)
    {
        return Err(invalid("target-only reference requires state-audited product-shaped conditioning and no draft/shadow/profile switches"));
    }
    if request.exact_projection_tiles
        && (!product || !request.batched_target_verify || request.target_only_reference)
    {
        return Err(invalid("exact projection tiles require canonical batched target execution, not reference/shadow mode"));
    }
    if request.exact_output_head_batch
        && (!product || !request.batched_target_verify || request.target_only_reference)
    {
        return Err(invalid("exact output head batching requires canonical batched target execution, not reference/shadow mode"));
    }
    if request.draft_q8_tile4 && (!request.batched_draft_context || !request.batched_draft_block) {
        return Err(invalid(
            "Q8 tile4 requires both context and block row batching",
        ));
    }
    if request.profile_prefill_sparse
        && (!product
            || !request.audit_committed_state
            || false)
    {
        return Err(invalid("sparse prefill profile requires committed-state audit and unfused product-shaped request"));
    }
    if request.batched_prompt_capture
        && (!request.batched_target_verify || !request.retain_full_accept_state)
    {
        return Err(invalid(
            "batch prompt capture requires actual batch verification and feature staging",
        ));
    }
    if request.reuse_batched_plan && !request.batched_target_verify {
        return Err(invalid("batch plan reuse requires actual batch execution"));
    }
    if request.audit_committed_state && !product {
        return Err(invalid(
            "committed-state audit requires product-shaped request",
        ));
    }
    if request.batched_target_verify
        && request.profile_batched_target_verify
        && !request.audit_committed_state
    {
        return Err(invalid(
            "canonical target profiling requires committed-state audit",
        ));
    }
    if request.batched_target_verify
        && (!product
            || request.audit_batched_target_verify
            || request.audit_batched_numerics
            || request.native_quantized_batch_ffn
            || request.canonical_batched_gdn_scalars
            || request.native_quantized_batch_attention
            || request.canonical_batched_norm
            || request.canonical_batched_full_core)
    {
        return Err(invalid("actual batch verifier requires product-shaped request and excludes all shadow switches"));
    }
    Ok(())
}

fn validate_diagnostic_q8_policy(
    request: Option<&Qwen35DraftTargetRequestOptions>,
    compiled_away: bool,
) -> Result<()> {
    let acknowledged = request.is_some_and(|r| r.diagnostic_q8_away);
    if acknowledged != compiled_away {
        return Err(invalid(
            "diagnostic Q8 policy requires matching build and explicit acknowledgement",
        ));
    }
    if compiled_away
        && request.is_none_or(|r| {
            !r.target_only_reference || !r.audit_reference_logits || !r.audit_committed_state
        })
    {
        return Err(invalid(
            "diagnostic Q8 policy requires logits/state-audited target-only reference",
        ));
    }
    if compiled_away && false {
        return Err(invalid(
            "diagnostic Q8 policy excludes separately fused quantization",
        ));
    }
    Ok(())
}

fn validate_diagnostic_split_rope_policy(
    request: Option<&Qwen35DraftTargetRequestOptions>,
    compiled: bool,
) -> Result<()> {
    if request.is_some_and(|r| r.diagnostic_split_rope) != compiled {
        return Err(invalid(
            "split-half RoPE requires matching build and explicit acknowledgement",
        ));
    }
    if compiled
        && (request.is_none_or(|r| {
            !r.target_only_reference
                || !r.audit_reference_logits
                || !r.audit_committed_state
                || r.diagnostic_q8_away
        }) || augur_backend::cuda_q8_0::Q8_1_DIAGNOSTIC_TIES_AWAY)
    {
        return Err(invalid(
            "split-half RoPE requires audited target-only reference and unchanged Q8 policy",
        ));
    }
    Ok(())
}

fn validate_target_split_rope_policy(
    request: Option<&Qwen35DraftTargetRequestOptions>,
    gpu_split: bool,
    cpu_split: bool,
) -> Result<()> {
    if gpu_split != cpu_split || request.is_some_and(|r| r.target_split_rope) != gpu_split {
        return Err(invalid(
            "target split-half RoPE requires matching CPU/CUDA build and explicit request identity",
        ));
    }
    if gpu_split
        && (augur_backend::cuda_q8_0::QWEN35_DIAGNOSTIC_SPLIT_ROPE
            || augur_backend::cuda_q8_0::Q8_1_DIAGNOSTIC_TIES_AWAY
            || request.is_none_or(|r| r.diagnostic_q8_away || r.diagnostic_split_rope))
    {
        return Err(invalid(
            "target split-half numerical contract excludes diagnostic ablations",
        ));
    }
    Ok(())
}

#[derive(Clone, Debug, Default)]
pub struct Qwen35DraftTargetRollingOptions {
    pub product_mode: bool,
    pub incremental_draft_context: bool,
    pub validate_draft_context_reuse: bool,
    pub request: Option<Qwen35DraftTargetRequestOptions>,
}

/// A token prefix that has crossed the physical verify/rollback/commit
/// boundary.  A product sink may publish these ids only after this callback
/// returns; provisional draft ids are never exposed through this type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen35VerifiedCommit {
    /// The first target-produced anchor is request-visible but does not
    /// advance the rolling recurrent/KV cursor. A one-token request may end
    /// at this anchor without executing a speculative verify block.
    pub initial_anchor: bool,
    pub block_index: usize,
    pub generation_before: usize,
    pub generation_after: usize,
    pub cursor_before: usize,
    pub cursor_after: usize,
    pub token_ids: Vec<u32>,
    pub stop_reason: Option<String>,
}

/// Callback used by the product-shaped runner to hand a verified anchor or
/// rolling commit to a streaming transport. Returning an error aborts before
/// the next rolling rebind, so a disconnected client cannot leave the
/// producer advancing after the consumer has stopped accepting bytes.
pub type Qwen35VerifiedCommitSink<'a> =
    dyn Fn(&Qwen35VerifiedCommit) -> std::result::Result<(), Qwen35MixedPrefixCudaError> + 'a;

impl Qwen35VerifiedCommit {
    fn validate_contract(&self) -> Result<()> {
        if self.token_ids.is_empty() {
            return Err(invalid("verified commit has no publishable token"));
        }
        if self.initial_anchor {
            if self.block_index != 0
                || self.token_ids.len() != 1
                || self.generation_after != self.generation_before
                || self.cursor_after != self.cursor_before
                || !matches!(
                    self.stop_reason.as_deref(),
                    None | Some("eos") | Some("max_new_tokens")
                )
            {
                return Err(invalid("invalid initial verified anchor contract"));
            }
            if self.token_ids.iter().any(|&token| token as usize >= V) {
                return Err(invalid("verified anchor contains an out-of-range token"));
            }
            return Ok(());
        }
        if self.generation_after != self.generation_before.saturating_add(1) {
            return Err(invalid(
                "verified commit generation is not a single-step advance",
            ));
        }
        let expected_cursor = self
            .cursor_before
            .checked_add(self.token_ids.len())
            .ok_or_else(|| invalid("verified commit cursor overflow"))?;
        if self.cursor_after != expected_cursor {
            return Err(invalid("verified commit cursor/token extent mismatch"));
        }
        if !matches!(
            self.stop_reason.as_deref(),
            None | Some("eos") | Some("max_new_tokens")
        ) {
            return Err(invalid("verified commit has unknown stop reason"));
        }
        if self.token_ids.iter().any(|&token| token as usize >= V) {
            return Err(invalid("verified commit contains an out-of-range token"));
        }
        Ok(())
    }
}

fn dispatch_verified_commit(
    sink: &Qwen35VerifiedCommitSink<'_>,
    commit: Qwen35VerifiedCommit,
) -> Result<()> {
    commit.validate_contract()?;
    sink(&commit)
}

pub fn run_qwen35_q8_draft_target_rolling_cuda_with_options(
    config: &Qwen35AutoregressiveCudaConfig,
    q8_path: &Path,
    draft_config_path: &Path,
    context_rows: usize,
    rolling_blocks: usize,
    vision_safetensors_path: Option<&Path>,
    options: Qwen35DraftTargetRollingOptions,
) -> Result<Value> {
    run_qwen35_q8_draft_target_verify_rolling_cuda_with_vision_mode(
        config,
        q8_path,
        draft_config_path,
        context_rows,
        rolling_blocks,
        vision_safetensors_path,
        options.product_mode,
        options.incremental_draft_context,
        options.validate_draft_context_reuse,
        options.request.as_ref(),
        None,
    )
}

/// Product-shaped rolling execution with an optional verified-commit sink.
/// The sink runs after each GPU commit and before the next feature rebind.
/// This is the first native boundary at which a transport may safely publish
/// text; it deliberately remains opt-in and does not widen any qualification
/// or TPS claims.
pub fn run_qwen35_q8_draft_target_rolling_cuda_with_options_and_sink(
    config: &Qwen35AutoregressiveCudaConfig,
    q8_path: &Path,
    draft_config_path: &Path,
    context_rows: usize,
    rolling_blocks: usize,
    vision_safetensors_path: Option<&Path>,
    options: Qwen35DraftTargetRollingOptions,
    sink: Option<&Qwen35VerifiedCommitSink<'_>>,
) -> Result<Value> {
    if sink.is_some() {
        if !options.product_mode {
            return Err(invalid(
                "verified commit sink requires product-shaped execution",
            ));
        }
        if options.request.is_none() {
            return Err(invalid(
                "verified commit sink requires request publication contract",
            ));
        }
    }
    run_qwen35_q8_draft_target_verify_rolling_cuda_with_vision_mode(
        config,
        q8_path,
        draft_config_path,
        context_rows,
        rolling_blocks,
        vision_safetensors_path,
        options.product_mode,
        options.incremental_draft_context,
        options.validate_draft_context_reuse,
        options.request.as_ref(),
        sink,
    )
}

fn run_qwen35_q8_draft_target_verify_rolling_cuda_with_vision_mode(
    config: &Qwen35AutoregressiveCudaConfig,
    q8_path: &Path,
    draft_config_path: &Path,
    context_rows: usize,
    rolling_blocks: usize,
    vision_safetensors_path: Option<&Path>,
    product_mode: bool,
    incremental_draft_context: bool,
    validate_draft_context_reuse: bool,
    request: Option<&Qwen35DraftTargetRequestOptions>,
    sink: Option<&Qwen35VerifiedCommitSink<'_>>,
) -> Result<Value> {
    run_rolling_with_resident_draft(
        config,
        q8_path,
        draft_config_path,
        context_rows,
        rolling_blocks,
        vision_safetensors_path,
        product_mode,
        incremental_draft_context,
        validate_draft_context_reuse,
        request,
        sink,
        None,
        None,
        None,
        #[cfg(feature = "qwen35-session-continuation")]
        None,
        #[cfg(feature = "qwen35-session-continuation")]
        None,
        #[cfg(feature = "qwen35-session-continuation")]
        None,
        #[cfg(feature = "qwen35-session-continuation")]
        false,
        #[cfg(feature = "qwen35-session-continuation")]
        false,
        #[cfg(feature = "qwen35-session-continuation")]
        false,
        #[cfg(feature = "qwen35-session-continuation")]
        None,
        #[cfg(feature = "qwen35-session-continuation")]
        None,
    )
}

/// Retain immutable Q8 weights, plus target payloads with `open_with_target`.
/// This bounded diagnostic owner is NOT a qualified resident model server.
/// Each call creates fresh target/DFlash KV, GDN, workspaces, streams and events.
pub struct Qwen35ResidentDraftCuda {
    // Drop the complete mutable session before the immutable weight owners.
    #[cfg(feature = "qwen35-session-continuation")]
    session: Option<committed_session::CommittedSession>,
    #[cfg(all(feature = "qwen35-session-continuation", feature = "qwen35-session-checkpoint", windows))]
    input_prefix: Option<committed_session::InputPrefixOwner>,
    #[cfg(feature = "qwen35-session-continuation")]
    session_cancellation: Option<crate::session_cancellation::SessionCancellationToken>,
    weights: DFlashQ8CudaWeights,
    target: Option<target_weights::TargetWeights>,
    vision: Option<ResidentAuxiliaryOwners>,
    vision_path: Option<PathBuf>,
    q8_path: PathBuf,
    draft_config_path: PathBuf,
    poisoned: bool,
    draft_startup_timing: Value,
}

impl Qwen35ResidentDraftCuda {
    #[cfg(feature = "qwen35-session-continuation")]
    fn has_retained_input_prefix(&self) -> bool {
        #[cfg(all(feature = "qwen35-session-checkpoint", windows))]
        { self.input_prefix.is_some() }
        #[cfg(not(all(feature = "qwen35-session-checkpoint", windows)))]
        { false }
    }

    pub fn open(q8_path: &Path, draft_config_path: &Path, device: i32) -> Result<Self> {
        let (backbone, cpu_load_ms) = Self::load_cpu_backbone(q8_path, draft_config_path)?;
        Self::upload_cpu_backbone(q8_path, draft_config_path, device, backbone, cpu_load_ms, false, 0.0)
    }

    fn load_cpu_backbone(q8_path: &Path, draft_config_path: &Path) -> Result<(DFlashQ8Backbone, f64)> {
        let startup = Instant::now();
        let draft_config = DFlashConfig::parse_f16_opt_in(&read_stable_regular_file(
            draft_config_path,
            "resident DFlash config",
        )?)?;
        let backbone = if target_weights::pinned_startup_enabled()? {
            DFlashQ8Backbone::load_pinned(q8_path, draft_config)
        } else { DFlashQ8Backbone::load(q8_path, draft_config) }.map_err(|e| invalid(e.to_string()))?;
        let cpu_load_ms = startup.elapsed().as_secs_f64() * 1000.0;
        Ok((backbone, cpu_load_ms))
    }

    fn upload_cpu_backbone(q8_path: &Path, draft_config_path: &Path, device: i32,
        backbone: DFlashQ8Backbone, cpu_load_ms: f64, prefetched: bool, join_wait_ms: f64) -> Result<Self> {
        let upload = Instant::now();
        let weights = DFlashQ8CudaWeights::upload(&backbone, device).map_err(invalid)?;
        let upload_ms = upload.elapsed().as_secs_f64() * 1000.0;
        Ok(Self {
            #[cfg(feature = "qwen35-session-continuation")]
            session: None,
            #[cfg(all(feature = "qwen35-session-continuation", feature = "qwen35-session-checkpoint", windows))]
            input_prefix: None,
            #[cfg(feature = "qwen35-session-continuation")]
            session_cancellation: None,
            weights,
            target: None,
            vision: None,
            vision_path: None,
            q8_path: q8_path.to_owned(),
            draft_config_path: draft_config_path.to_owned(),
            poisoned: false,
            draft_startup_timing: json!({"clock":"host_monotonic_wall","gpu_device_time_measured":false,
                "scope":"draft_cpu_and_upload_walls_nested_with_owner_and_target_when_prefetched",
                "cpu_load_prefetched_with_target":prefetched,"authentication_join_wait_ms":join_wait_ms,
                "cpu_plus_upload_ms_not_additive_with_owner":cpu_load_ms+upload_ms,
                "post_target_critical_path_ms":if prefetched {join_wait_ms+upload_ms} else {cpu_load_ms+upload_ms},
                "parts_ms":{"cpu_config_manifest_auth_and_payload_read":cpu_load_ms,"weight_upload":upload_ms}}),
        })
    }

    pub fn resident_weight_bytes(&self) -> usize {
        self.weights.resident_bytes()
    }

    pub fn startup_report(&self) -> Value {
        json!({"schema":"qwen35-resident-startup-host-timing-v1","gpu_device_time_measured":false,
            "target":self.target.as_ref().map(|t|&t.startup_timing),"draft":self.draft_startup_timing,
            "vision_load_and_upload_ms":self.vision.as_ref().map(|v|v.report().upload_wall_ms),
            "scope":"nested_host_walls_not_additive_with_owner_open_or_HTTP_startup"})
    }

    /// Install one monotonic notification before admitting any session. This
    /// token grants no authority to free device memory or reuse cancelled state.
    #[cfg(feature = "qwen35-session-continuation")]
    pub fn install_session_cancellation(
        &mut self,
        token: crate::session_cancellation::SessionCancellationToken,
    ) -> Result<()> {
        if self.poisoned || self.session.is_some() || self.has_retained_input_prefix()
            || self.session_cancellation.is_some() {
            return Err(invalid(
                "session cancellation must be installed once before admission",
            ));
        }
        self.session_cancellation = Some(token);
        Ok(())
    }

    /// Opt-in extraction of immutable target payloads as well as Q8 weights.
    /// Request plans, KV/GDN, streams, workspaces and receipts remain fresh.
    pub fn open_with_target(
        config: &Qwen35AutoregressiveCudaConfig,
        q8_path: &Path,
        draft_config_path: &Path,
    ) -> Result<Self> {
        let prefetch = target_weights::startup_component_enabled("AUGUR_QWEN35_PREFETCH_DRAFT_STARTUP",
            target_weights::pinned_startup_enabled()?)?;
        let (target, mut owner) = if prefetch {
            // Only immutable CPU material crosses threads. Target CUDA owner and
            // draft uploads remain on this caller thread. Join even on failure.
            std::thread::scope(|scope| -> Result<_> {
                let worker = scope.spawn(|| Self::load_cpu_backbone(q8_path, draft_config_path));
                let target = target_weights::TargetWeights::open(config);
                let wait = Instant::now();
                let prepared = worker.join().map_err(|_| invalid("DFlash CPU prefetch worker panicked"))?;
                let wait_ms = wait.elapsed().as_secs_f64()*1000.0;
                let target = target?;
                let (backbone, cpu_load_ms) = prepared?;
                let owner = Self::upload_cpu_backbone(q8_path, draft_config_path, config.device_ordinal,
                    backbone, cpu_load_ms, true, wait_ms)?;
                Ok((target, owner))
            })?
        } else {
            let target = target_weights::TargetWeights::open(config)?;
            let owner = Self::open(q8_path, draft_config_path, config.device_ordinal)?;
            (target, owner)
        };
        owner.target = Some(target);
        Ok(owner)
    }

    /// Research-only complete immutable residency. No request state is reused.
    /// Uploads finish before this owner can serve its first request. The caller
    /// must account for these weights during prefill, not only during decode.
    #[cfg(feature = "qwen35-resident-text-service")]
    pub fn open_with_target_and_vision(
        config: &Qwen35AutoregressiveCudaConfig,
        q8_path: &Path,
        draft_config_path: &Path,
    ) -> Result<Self> {
        let vision_path = config
            .vision_safetensors_path
            .as_deref()
            .ok_or_else(|| invalid("complete resident service requires Vision weights"))?;
        if config.device_ordinal != 0 || config.capacity != 196608 {
            return Err(invalid(
                "complete resident service requires device0/full196608 KV8",
            ));
        }
        let mut owner = Self::open_with_target(config, q8_path, draft_config_path)?;
        let stream = OwnedCudaStream::create(config.device_ordinal)?;
        owner.vision =
            ResidentAuxiliaryOwners::load(None, Some(vision_path), config.device_ordinal, &stream)
                .map_err(invalid)?;
        owner.vision_path = Some(vision_path.to_owned());
        Ok(owner)
    }

    pub fn run_request(
        &mut self,
        config: &Qwen35AutoregressiveCudaConfig,
        context_rows: usize,
        rolling_blocks: usize,
        vision_safetensors_path: Option<&Path>,
        options: Qwen35DraftTargetRollingOptions,
        sink: Option<&Qwen35VerifiedCommitSink<'_>>,
    ) -> Result<Value> {
        #[cfg(feature = "qwen35-session-continuation")]
        if self.session.is_some() || self.has_retained_input_prefix() {
            return Err(invalid("fresh request cannot replace a retained session"));
        }
        if self.poisoned {
            return Err(invalid(
                "resident draft owner poisoned by prior request failure",
            ));
        }
        validate_resident_vision_selection(
            self.vision_path.as_deref(),
            vision_safetensors_path,
            config.vision_safetensors_path.as_deref(),
        )?;
        if !options.product_mode
            || options.request.is_none()
            || options
                .request
                .as_ref()
                .is_some_and(|r| r.target_only_reference)
            || config.device_ordinal != self.weights.device_ordinal()
        {
            return Err(invalid(
                "resident draft requires a same-device product-shaped draft request",
            ));
        }
        let before = cuda_device_allocation_snapshot();
        let request_wall = Instant::now();
        // An execution error may be a CUDA context error, not just a sink
        // failure. Conservatively require a fresh outer owner after any error.
        self.poisoned = true;
        let mut report = run_rolling_with_resident_draft(
            config,
            &self.q8_path,
            &self.draft_config_path,
            context_rows,
            rolling_blocks,
            vision_safetensors_path,
            options.product_mode,
            options.incremental_draft_context,
            options.validate_draft_context_reuse,
            options.request.as_ref(),
            sink,
            Some(&self.weights),
            self.target.as_ref(),
            self.vision.as_ref(),
            #[cfg(feature = "qwen35-session-continuation")]
            None,
            #[cfg(feature = "qwen35-session-continuation")]
            None,
            #[cfg(feature = "qwen35-session-continuation")]
            None,
            #[cfg(feature = "qwen35-session-continuation")]
            false,
            #[cfg(feature = "qwen35-session-continuation")]
            false,
            #[cfg(feature = "qwen35-session-continuation")]
            false,
            #[cfg(feature = "qwen35-session-continuation")]
            self.session_cancellation.as_ref(),
            #[cfg(feature = "qwen35-session-continuation")]
            None,
        )?;
        if let Some(vision) = &self.vision {
            report["resident_vision"] = json!({
                "schema":"qwen35-immutable-resident-vision-v1",
                "weight_bytes":vision.owner_bytes(),
                "weight_allocations":vision.owner_count(),
                "source_identity_policy":"load_time_authenticated_bytes_pinned_until_owner_drop",
                "weight_upload_policy":"reuse_load_time_owner_no_request_upload",
                "vision_resident_report_timing_scope":"owner_open_once_not_request_upload",
                "request_weight_h2d_bytes":0,
                "vision_forward_executed":false,
                "product_ready":false
            });
        }
        report["resident_draft"] = json!({
            "schema":"qwen35-immutable-resident-draft-v1",
            "weight_bytes":self.weights.resident_bytes(),
            "weight_sha256":self.weights.file_sha256(),
            "source_identity_policy":"load_time_authenticated_bytes_pinned_until_owner_drop",
            "weight_upload_policy":"reuse_load_time_owner_no_request_upload",
            "fresh_request_state":true,"target_weights_reused":self.target.is_some(),
            "baseline_live_bytes":before.live_bytes,
            "baseline_live_allocations":before.live_allocations,
            "request_execution_wall_ms":request_wall.elapsed().as_secs_f64()*1000.0,
            "timing_scope":if self.target.is_some() {"run_request_excluding_resident_weights_open_and_transport"} else {"run_request_including_target_reload_excluding_resident_draft_open_and_transport"},
            "resident_model_server":false,"product_ready":false
        });
        if let Some(target) = &self.target {
            report["resident_target"] = target.resident_report();
        }
        report["cleanup"]["scope"] = json!("request_private_owners_returned_to_resident_baseline");
        self.poisoned = false;
        Ok(report)
    }
}

fn validate_resident_vision_selection(
    retained: Option<&Path>,
    request: Option<&Path>,
    configured: Option<&Path>,
) -> Result<()> {
    if retained.is_some() && (request != retained || configured != retained) {
        return Err(invalid(
            "resident Vision selection changed before request submission",
        ));
    }
    Ok(())
}

fn run_rolling_with_resident_draft(
    config: &Qwen35AutoregressiveCudaConfig,
    q8_path: &Path,
    draft_config_path: &Path,
    context_rows: usize,
    rolling_blocks: usize,
    vision_safetensors_path: Option<&Path>,
    product_mode: bool,
    incremental_draft_context: bool,
    validate_draft_context_reuse: bool,
    request: Option<&Qwen35DraftTargetRequestOptions>,
    sink: Option<&Qwen35VerifiedCommitSink<'_>>,
    shared_draft_weights: Option<&DFlashQ8CudaWeights>,
    shared_target_weights: Option<&target_weights::TargetWeights>,
    shared_vision: Option<&ResidentAuxiliaryOwners>,
    #[cfg(feature = "qwen35-session-continuation")] session_output: Option<committed_session::RetainedOutput<'_>>,
    #[cfg(feature = "qwen35-session-continuation")] native_reference: Option<
        committed_session::FreshReference<'_>,
    >,
    #[cfg(feature = "qwen35-session-continuation")] native_transition_directory: Option<&Path>,
    #[cfg(feature = "qwen35-session-continuation")] native_prefill_scalar_audit: bool,
    #[cfg(feature = "qwen35-session-continuation")] native_prefill_full_canonical_audit: bool,
    #[cfg(feature = "qwen35-session-continuation")] native_growing_session: bool,
    #[cfg(feature = "qwen35-session-continuation")] native_cancellation: Option<
        &crate::session_cancellation::SessionCancellationToken,
    >,
    #[cfg(feature = "qwen35-session-continuation")] resident_output_ceiling: Option<usize>,
) -> Result<Value> {
    #[cfg(all(feature = "qwen35-session-continuation", feature = "qwen35-session-checkpoint", windows))]
    let mut session_output = session_output;
    #[cfg(feature = "qwen35-session-continuation")]
    let input_only = session_output.as_ref().is_some_and(|output| output.input_tokens().is_some());
    #[cfg(not(feature = "qwen35-session-continuation"))]
    let input_only = false;
    #[cfg(feature = "qwen35-session-continuation")]
    if input_only && (sink.is_some() || resident_output_ceiling.is_some()
        || native_reference.is_some() || native_transition_directory.is_some()
        || native_prefill_scalar_audit || !native_prefill_full_canonical_audit
        || !native_growing_session) {
        return Err(invalid("input-only prefix cannot publish or impersonate a generated session"));
    }
    #[cfg(feature = "qwen35-session-continuation")]
    if let Some(ceiling) = resident_output_ceiling {
        crate::resident_session_profile::ResidentSessionProfile::Resident
            .output_budget(Some(ceiling), true)
            .map_err(invalid)?;
        if !native_growing_session
            || !native_prefill_full_canonical_audit
            || session_output.is_none()
            || native_reference.is_some()
            || sink.is_none()
        {
            return Err(invalid(
                "resident output ceiling requires canonical initialized retained session",
            ));
        }
    }
    #[cfg(feature = "qwen35-session-continuation")]
    if let Some(token) = native_cancellation {
        token
            .check("request before submit")
            .map_err(|error| invalid(error.to_string()))?;
    }
    #[cfg(not(feature = "qwen35-session-continuation"))]
    let native_growing_session = false;
    validate_diagnostic_q8_policy(request, augur_backend::cuda_q8_0::Q8_1_DIAGNOSTIC_TIES_AWAY)?;
    let projection_corpus_rows = reference_boundary::requested_projection_rows()?;
    if projection_corpus_rows != 0
        && (context_rows != 2048
            || config.capacity != 196608
            || vision_safetensors_path.is_none()
            || !product_mode
            || !request.is_some_and(|r| {
                r.reference_boundary_directory.is_some()
                    && r.target_only_reference
                    && r.batched_prompt_capture
                    && r.audit_reference_logits
                    && r.audit_committed_state
                    && r.target_split_rope
            }))
    {
        return Err(invalid("projection corpus requires explicit canonical reference and complete resident configuration"));
    }
    let reference_canonical_prefix = reference_canonical_prefix_switch(
        std::env::var_os("AUGUR_QWEN35_REFERENCE_CANONICAL_PREFIX").as_deref(),
        cfg!(feature = "qwen35-session-continuation"),
        context_rows == 2048
            && config.capacity == 196608
            && product_mode
            && vision_safetensors_path.is_some()
            && request.is_some_and(|r| {
                r.target_only_reference
                    && r.max_new_tokens == 1
                    && r.reference_boundary_directory.is_some()
                    && r.audit_reference_logits
                    && r.audit_committed_state
                    && r.batched_prompt_capture
                    && r.target_split_rope
                    && !r.reference_serial_prefix
            }),
    )?;
    #[cfg(feature = "qwen35-session-continuation")]
    let canonical_product_prefix = native_prefill_full_canonical_audit;
    #[cfg(not(feature = "qwen35-session-continuation"))]
    let canonical_product_prefix = false;
    // Select the complete opt-in recipe or preserve all explicit experiments.
    // No partial fallback may silently turn a B8 control into a mixed recipe.
    let prefill_switches = batched_verify::prefill_policy::Switches::from_env().resolve(
        batched_verify::prefill_policy::Scope {
            canonical_session: canonical_product_prefix && native_growing_session && product_mode,
            resident: shared_target_weights.is_some() && shared_draft_weights.is_some(),
            exact_batched_target: request.is_some_and(|r| {
                r.batched_target_verify && r.exact_projection_tiles && !r.target_only_reference
            }),
            vision: shared_vision.is_some() && vision_safetensors_path.is_some(),
            capacity: config.capacity,
            history_rows: context_rows,
        },
    )?;
    // One immutable schedule per Driver. Admission and submission must not
    // independently read a mutable process environment.
    let canonical_prefill_rows_switch = prefill_switches.rows.as_deref();
    let canonical_prefill_rows =
        batched_verify::prefill_width::switch(canonical_prefill_rows_switch.as_deref())?;
    let tensor_ffn_down_rows = batched_verify::prefill_ffn_down::requested(
        prefill_switches.tensor_down.as_deref(),
        canonical_prefill_rows_switch.as_deref(),
    )?;
    let tensor_ffn_down_compensated = batched_verify::prefill_ffn_down::requested_compensated(
        prefill_switches.compensated.as_deref(),
        tensor_ffn_down_rows,
    )?;
    let integer_mma_ffn = batched_verify::prefill_integer_mma::requested(
        prefill_switches.ffn.as_deref(),
        canonical_prefill_rows_switch.as_deref(),
        canonical_product_prefix || reference_canonical_prefix,
        tensor_ffn_down_rows.is_some() || tensor_ffn_down_compensated,
    )?;
    if tensor_ffn_down_rows.is_some() && !canonical_product_prefix && !reference_canonical_prefix {
        return Err(invalid(
            "FFN-down Tensor experiment requires explicit canonical cold prefix",
        ));
    }
    let integer_mma_input =
        batched_verify::prefill_input_integer::requested(prefill_switches.input.as_deref(), integer_mma_ffn)?;
    if integer_mma_input && integer_mma_ffn != Some(512) {
        return Err(invalid("input integer requires B512 FFN"));
    }
    let llama_f16_input = prefill_switches.llama_f16_input;
    let llama_original_input = prefill_switches.llama_original_input;
    let llama_full_original = prefill_switches.llama_full_original;
    let llama_rotated_kv = prefill_switches.llama_rotated_kv;
    let llama_bf16_scalars = prefill_switches.llama_bf16_scalars;
    let llama_history_full = prefill_switches.llama_history_full;
    let llama_global_chunks = prefill_switches.llama_global_chunks;
    let llama_exact_input_rows = prefill_switches.llama_exact_input_rows;
    let batched_feature_copies = submission_batch_switch(
        std::env::var_os("AUGUR_QWEN35_BATCHED_FEATURE_COPIES").as_deref(),
        llama_exact_input_rows && canonical_product_prefix)?;
    let batched_embedding_gather = submission_batch_switch(
        std::env::var_os("AUGUR_QWEN35_BATCHED_EMBEDDING_GATHER").as_deref(),
        llama_exact_input_rows && canonical_product_prefix)?;
    let history_batched_feature_copies = submission_batch_switch(
        std::env::var_os("AUGUR_QWEN35_HISTORY_BATCHED_FEATURE_COPIES").as_deref(),
        llama_exact_input_rows && canonical_product_prefix)?;
    let batched_checkpoint_copies = submission_batch_switch(
        std::env::var_os("AUGUR_QWEN35_CHECKPOINT_COPY_BATCH").as_deref(),
        llama_exact_input_rows && canonical_product_prefix)?;
    let batched_replay_copies = submission_batch_switch(
        std::env::var_os("AUGUR_QWEN35_REPLAY_COPY_BATCH").as_deref(),
        llama_exact_input_rows && canonical_product_prefix)?;
    let shared_replay_fold_scalars = submission_batch_switch(
        std::env::var_os("AUGUR_QWEN35_REPLAY_SHARED_SCALARS").as_deref(),
        llama_exact_input_rows && canonical_product_prefix
            && std::env::var("AUGUR_QWEN35_REPLAY_SSM").as_deref() == Ok("1"))?;
    let shared_verify_gdn_scalars = submission_batch_switch(
        std::env::var_os("AUGUR_QWEN35_VERIFY_GDN_SHARED_SCALARS").as_deref(),
        llama_exact_input_rows && canonical_product_prefix)?;
    let replay_checkpoint_read = submission_batch_switch(
        std::env::var_os("AUGUR_QWEN35_REPLAY_CHECKPOINT_READ").as_deref(),
        shared_replay_fold_scalars && !batched_checkpoint_copies)?;
    // Save is an independent copy batch; restore/fold keeps the retained
    // direct-checkpoint path. Never re-enable the rejected restore-copy bundle.
    let batched_checkpoint_save_only = submission_batch_switch(
        std::env::var_os("AUGUR_QWEN35_CHECKPOINT_SAVE_BATCH").as_deref(),
        replay_checkpoint_read && !batched_checkpoint_copies)?;
    let replay_prefix_family = submission_batch_switch(
        std::env::var_os("AUGUR_QWEN35_REPLAY_PREFIX_FAMILY").as_deref(),
        replay_checkpoint_read)?;
    let dflash_history_projection_reuse = submission_batch_switch(
        std::env::var_os("AUGUR_QWEN35_DFLASH_HISTORY_ROW_REUSE").as_deref(),
        llama_exact_input_rows && canonical_product_prefix
            && request.is_some_and(|r| r.batched_draft_context && r.draft_q8_tile4))?;
    let dflash_b8_specialized = submission_batch_switch(
        std::env::var_os("AUGUR_QWEN35_DFLASH_B8_SPECIALIZED").as_deref(),
        llama_exact_input_rows && canonical_product_prefix
            && request.is_some_and(|r| r.batched_draft_context && r.batched_draft_block && r.draft_q8_tile4))?;
    let dflash_history_weight_panels = submission_batch_switch(
        std::env::var_os("AUGUR_QWEN35_DFLASH_HISTORY_WEIGHT_PANELS").as_deref(),
        dflash_history_projection_reuse)?;
    let dflash_history_panel_token8 = submission_batch_switch(
        std::env::var_os("AUGUR_QWEN35_DFLASH_HISTORY_PANEL_TOKEN8").as_deref(),
        dflash_history_weight_panels,
    )?;
    let dflash_context_key_norm_rope = submission_batch_switch(
        std::env::var_os("AUGUR_QWEN35_DFLASH_CONTEXT_NORM_ROPE").as_deref(),
        llama_exact_input_rows && canonical_product_prefix
            && request.is_some_and(|r| r.batched_draft_context))?;
    let output_head_batch_eligible = llama_exact_input_rows && canonical_product_prefix
        && request.is_some_and(|r| r.exact_output_head_batch && r.batched_target_verify);
    let batched_output_head_prepare = submission_batch_switch(
        std::env::var_os("AUGUR_QWEN35_OUTPUT_HEAD_PREPARE_BATCH").as_deref(),
        output_head_batch_eligible)?;
    let head_prepare_stores = submission_batch_switch(
        std::env::var_os("AUGUR_QWEN35_HEAD_PREPARE_STORES_B8").as_deref(),
        batched_output_head_prepare && output_head_batch_eligible)?;
    let batched_target_top1 = submission_batch_switch(
        std::env::var_os("AUGUR_QWEN35_TARGET_TOP1_BATCH").as_deref(),
        output_head_batch_eligible)?;
    let verify_rope_reuse = submission_batch_switch(
        std::env::var_os("AUGUR_QWEN35_VERIFY_ROPE_REUSE").as_deref(),
        llama_exact_input_rows && canonical_product_prefix
            && request.is_some_and(|r| r.batched_target_verify && r.share_target_rope))?;
    let prefill_batched_head_rope = batched_verify::prefill_policy::batched_head_rope_switch(
        std::env::var_os("AUGUR_QWEN35_PREFILL_BATCHED_HEAD_ROPE").as_deref(),
        cfg!(feature = "qwen35-fused-head-norm-rope"),
        llama_exact_input_rows && llama_rotated_kv && canonical_product_prefix
            && canonical_prefill_rows == 512
            && std::env::var("AUGUR_QWEN35_SHARED_KV_QUERIES").as_deref() == Ok("1"),
    )?;
    #[cfg(feature = "qwen35-session-continuation")]
    if prefill_batched_head_rope && (input_only || native_reference.is_some()
        || native_transition_directory.is_some() || native_prefill_scalar_audit) {
        return Err(invalid("batched head RoPE excludes persistent and reference audit owners"));
    }
    if (llama_f16_input || llama_original_input || llama_full_original) && !integer_mma_input
        || usize::from(llama_f16_input) + usize::from(llama_original_input) + usize::from(llama_full_original) > 1 {
        return Err(invalid("llama input requires the admitted B512 input recipe"));
    }
    let projection_row_reuse = batched_verify::projection_row_reuse_switch(
        prefill_switches.row_reuse.as_deref(),
        request.is_some_and(|r| r.exact_projection_tiles),
        product_mode
            && request.is_some_and(|r| r.batched_target_verify && !r.target_only_reference),
    )?;
    let ffn_down_row_reuse = submission_batch_switch(
        std::env::var_os("AUGUR_QWEN35_FFN_DOWN_ROW_REUSE").as_deref(),
        projection_row_reuse,
    )?;
    let preserved_silu_q8 = submission_batch_switch(
        std::env::var_os("AUGUR_QWEN35_SILU_Q8_PRESERVED_B8").as_deref(),
        projection_row_reuse && llama_exact_input_rows && canonical_product_prefix,
    )?;
    let stage_profile_requested = product_stage_switch(
        std::env::var_os("AUGUR_QWEN35_PRODUCT_STAGE_PROFILE").as_deref(),
        false,
    )?;
    let draft_top16_batch = batched_verify::draft_top16_batch_switch(
        std::env::var_os("AUGUR_QWEN35_DRAFT_TOP16_BATCH").as_deref(),
        cfg!(feature = "qwen35-draft-top16-batch"),
    )?;
    let top16_validation = top16_validation::parse(
        std::env::var_os("AUGUR_QWEN35_TOP16_VALIDATION_MODE").as_deref(),
        cfg!(feature = "qwen35-top16-state-client"),
    )?;
    let legacy_growing_switch =
        growing_history_switch(std::env::var_os("AUGUR_QWEN35_GROWING_HISTORY").as_deref())?;
    let fast_feature_identity_requested = fast_feature_identity_switch(
        std::env::var_os("AUGUR_QWEN35_FAST_FEATURE_IDENTITY").as_deref(),
    )?;
    let growing_history_requested = legacy_growing_switch || native_growing_session;
    #[cfg(feature = "qwen35-session-continuation")]
    let fast_session_path_requested = committed_session::fast_session_path_switch(
        std::env::var_os("AUGUR_QWEN35_FAST_SESSION_PATH").as_deref(),
    )?;
    #[cfg(feature = "qwen35-session-continuation")]
    let fast_checks_allowed = committed_session::fast_session_path_active(
        true,
        resident_output_ceiling.is_some() || input_only,
        validate_draft_context_reuse
            || native_reference.is_some()
            || native_transition_directory.is_some()
            || native_prefill_scalar_audit
            || request.is_some_and(|r| {
                r.audit_committed_state
                    || r.audit_batched_target_verify
                    || r.audit_batched_numerics
                    || r.audit_reference_logits
                    || r.target_only_reference
            }),
    );
    #[cfg(not(feature = "qwen35-session-continuation"))]
    let fast_session_path_requested = false;
    #[cfg(not(feature = "qwen35-session-continuation"))]
    let fast_checks_allowed = false;
    let fast_session_path = fast_session_path_requested && fast_checks_allowed;
    let fast_feature_identity = fast_feature_identity_requested && fast_checks_allowed;
    let replay_head_elision =
        replay_head_elision_switch(std::env::var_os("AUGUR_QWEN35_REPLAY_HEAD_ELISION").as_deref())?;
    if replay_head_elision && (!fast_checks_allowed || stage_profile_requested) {
        return Err(invalid(
            "replay head elision requires unaudited live resident execution",
        ));
    }
    let verify_linear_graphs = batched_verify::linear_graphs::switch(
        std::env::var_os("AUGUR_QWEN35_VERIFY_LINEAR_GRAPHS").as_deref(),
    )?;
    if verify_linear_graphs && (!fast_checks_allowed || stage_profile_requested) {
        return Err(invalid(
            "verify linear graphs require unaudited live resident execution",
        ));
    }
    if verify_linear_graphs && (preserved_silu_q8 || head_prepare_stores) {
        return Err(invalid("B8 boundary fusions are eager-only candidates"));
    }
    let replay_ssm_requested =
        batched_verify::replay_ssm::switch(std::env::var_os("AUGUR_QWEN35_REPLAY_SSM").as_deref())?;
    let replay_ssm_audit = batched_verify::replay_ssm::switch(
        std::env::var_os("AUGUR_QWEN35_REPLAY_SSM_AUDIT").as_deref(),
    )?;
    if replay_ssm_audit && !replay_ssm_requested {
        return Err(invalid("ReplaySSM audit requires enabled record/fold path"));
    }
    if std::env::var_os("AUGUR_QWEN35_REPLAY_NUMERIC_DIR").is_some() && !replay_ssm_audit {
        return Err(invalid("replay numeric observer requires explicit ReplaySSM audit"));
    }
    if std::env::var_os("AUGUR_QWEN35_VERIFY_ATTENTION_SERIAL_ROWS").is_some()
        && (!replay_ssm_audit || std::env::var_os("AUGUR_QWEN35_REPLAY_NUMERIC_DIR").is_none()
            || std::env::var("AUGUR_QWEN35_SHARED_KV_QUERIES").as_deref() != Ok("1")) {
        return Err(invalid("serial verify attention requires explicit numeric observer and shared-query owner"));
    }
    if replay_ssm_requested
        && (!fast_checks_allowed || !request.is_some_and(|r| r.batched_target_verify))
    {
        return Err(invalid(
            "ReplaySSM requires unaudited canonical B8 execution",
        ));
    }
    if std::env::var_os("AUGUR_QWEN35_VERIFY_ATTENTION_VECTOR_ROWS").is_some()
        && (std::env::var("AUGUR_QWEN35_VERIFY_ATTENTION_VECTOR_ROWS").as_deref() != Ok("1")
            || std::env::var("AUGUR_QWEN35_SHARED_KV_QUERIES").as_deref() != Ok("1")
            || std::env::var_os("AUGUR_QWEN35_VERIFY_ATTENTION_SERIAL_ROWS").is_some()
            || !request.is_some_and(|r| r.batched_target_verify)) {
        return Err(invalid("vector verify attention requires explicit shared B8 execution"));
    }
    if std::env::var_os("AUGUR_QWEN35_SMALL_ATTENTION_TILE8_ROWS").is_some()
        && (std::env::var("AUGUR_QWEN35_SMALL_ATTENTION_TILE8_ROWS").as_deref() != Ok("1")
            || std::env::var("AUGUR_QWEN35_SHARED_KV_QUERIES").as_deref() != Ok("1")
            || std::env::var_os("AUGUR_QWEN35_VERIFY_ATTENTION_VECTOR_ROWS").is_some()
            || std::env::var_os("AUGUR_QWEN35_VERIFY_ATTENTION_SERIAL_ROWS").is_some()) {
        return Err(invalid("tile8 row contract requires shared owner and excludes other row overrides"));
    }
    if std::env::var_os("AUGUR_QWEN35_SMALL_ATTENTION_GQA8_MMA").is_some()
        && (std::env::var("AUGUR_QWEN35_SMALL_ATTENTION_GQA8_MMA").as_deref() != Ok("1")
            || !llama_rotated_kv || !llama_exact_input_rows
            || std::env::var("AUGUR_QWEN35_SHARED_KV_QUERIES").as_deref() != Ok("1")
            || ["AUGUR_QWEN35_SMALL_ATTENTION_TILE8_ROWS", "AUGUR_QWEN35_VERIFY_ATTENTION_VECTOR_ROWS",
                "AUGUR_QWEN35_VERIFY_ATTENTION_SERIAL_ROWS"].iter().any(|k| std::env::var_os(k).is_some())) {
        return Err(invalid("GQA8 MMA requires exact rotated shared owner without row overrides"));
    }
    let planar_rotated_kv = RotatedFullAttention::planar_switch(
        std::env::var_os("AUGUR_QWEN35_ROTATED_KV_PLANAR").as_deref(),
        std::env::var("AUGUR_QWEN35_SMALL_ATTENTION_GQA8_MMA").as_deref() == Ok("1"))?;
    let rotated_kv_identity = if planar_rotated_kv {
        augur_backend::cuda_qwen35_rotated_kv::QWEN35_ROTATED_KV_PLANAR_IDENTITY
    } else { augur_backend::cuda_qwen35_rotated_kv::QWEN35_ROTATED_KV_IDENTITY };
    #[cfg(feature = "qwen35-session-continuation")]
    let batched_suffix =
        committed_session::suffix::switch(std::env::var_os("AUGUR_QWEN35_BATCHED_SUFFIX").as_deref())?;
    #[cfg(feature = "qwen35-session-continuation")]
    let reuse_batched_plan = committed_session::suffix::retain_plan(
        batched_suffix,
        request.is_some_and(|r| r.reuse_batched_plan) || verify_linear_graphs,
    );
    #[cfg(not(feature = "qwen35-session-continuation"))]
    let reuse_batched_plan = request.is_some_and(|r| r.reuse_batched_plan);
    #[cfg(feature = "qwen35-session-continuation")]
    committed_session::suffix::validate_scope(
        batched_suffix,
        native_growing_session,
        request.is_some_and(|r| r.batched_target_verify),
        reuse_batched_plan,
        stage_profile_requested
            || request
                .is_some_and(|r| r.profile_batched_target_verify || r.audit_batched_target_verify),
    )?;
    #[cfg(not(feature = "qwen35-session-continuation"))]
    if std::env::var_os("AUGUR_QWEN35_BATCHED_SUFFIX").is_some() {
        return Err(invalid(
            "batched suffix requires session continuation feature",
        ));
    }
    if growing_history_requested && context_rows > DEFAULT_GROWING_HISTORY_CAPACITY {
        return Err(invalid(
            "growing history context exceeds physical feature capacity",
        ));
    }
    #[cfg(feature = "qwen35-session-continuation")]
    let has_session_consumer = session_output.is_some()
        || native_reference.is_some()
        || native_transition_directory.is_some()
        || native_prefill_scalar_audit
        || native_prefill_full_canonical_audit;
    #[cfg(not(feature = "qwen35-session-continuation"))]
    let has_session_consumer = false;
    #[cfg(feature = "qwen35-session-continuation")]
    if native_growing_session
        && (legacy_growing_switch
            || !native_prefill_full_canonical_audit
            || native_prefill_scalar_audit
            || native_transition_directory.is_some()
            || session_output.is_none()
            || top16_validation != Some(top16_validation::Mode::Resident)
            || shared_target_weights.is_none()
            || shared_draft_weights.is_none()
            || shared_vision.is_none()
            || native_reference
                .as_ref()
                .is_some_and(|r| !r.profile.is_growing()))
    {
        return Err(invalid(
            "explicit native growing session scope before allocation",
        ));
    }
    validate_growing_history_scope(
        growing_history_requested,
        product_mode,
        request.is_some(),
        incremental_draft_context,
        (has_session_consumer && !native_growing_session)
            || validate_draft_context_reuse
            || (top16_validation.is_some() && !native_growing_session)
            || stage_profile_requested
            || std::env::var_os("AUGUR_QWEN35_RAW_STATE_DIR").is_some()
            || std::env::var_os("AUGUR_QWEN35_DRAFT_PRIVATE_DIR").is_some(),
    )?;
    #[cfg(feature = "qwen35-session-continuation")]
    if (native_prefill_scalar_audit || native_prefill_full_canonical_audit)
        && (native_transition_directory.is_some()
            || (native_prefill_scalar_audit && native_prefill_full_canonical_audit)
            || session_output.is_none()
            || top16_validation != Some(top16_validation::Mode::Resident)
            || shared_target_weights.is_none()
            || shared_draft_weights.is_none()
            || shared_vision.is_none()
            || (draft_top16_batch && !top16_validation::live_prefill_batch_scope(
                native_growing_session,
                native_prefill_full_canonical_audit,
                native_prefill_scalar_audit,
                resident_output_ceiling,
            ))
            || stage_profile_requested)
    {
        return Err(invalid(
            "native scalar audit requires exclusive retained/reference gate",
        ));
    }
    #[cfg(feature = "qwen35-session-continuation")]
    if native_transition_directory.is_some()
        && (native_reference.is_some()
            || session_output.is_none()
            || top16_validation != Some(top16_validation::Mode::Resident)
            || shared_target_weights.is_none()
            || shared_draft_weights.is_none()
            || shared_vision.is_none()
            || draft_top16_batch
            || stage_profile_requested)
    {
        return Err(invalid(
            "native transition audit admission before allocation",
        ));
    }
    #[cfg(feature = "qwen35-session-continuation")]
    if let Some(reference) = native_reference.as_ref() {
        reference.validate()?;
        if reference.profile.is_growing() != native_growing_session {
            return Err(invalid("growing exact-reference profile mismatch"));
        }
        if top16_validation != Some(top16_validation::Mode::Resident)
            || session_output.is_none()
            || shared_target_weights.is_none()
            || shared_draft_weights.is_none()
            || shared_vision.is_none()
            || draft_top16_batch
            || stage_profile_requested
        {
            return Err(invalid(
                "native exact-token reference admission before allocation",
            ));
        }
    }
    if let Some(mode) = top16_validation {
        let r = request.ok_or_else(|| invalid("top16 validation requires a bounded request"))?;
        let admission = top16_validation::Admission {
            frozen_request_shape: product_mode
                && incremental_draft_context
                && !validate_draft_context_reuse
                && context_rows == 2048
                && config.capacity == 196608
                && rolling_blocks == 1024
                && r.long_generation_budget
                && r.max_new_tokens == 1024
                && r.batched_target_verify
                && r.retain_full_accept_state
                && r.exact_projection_tiles
                && r.exact_output_head_batch
                && r.batched_draft_context
                && r.batched_draft_block
                && r.draft_q8_tile4
                && r.batched_prompt_capture
                && r.prefill_dgemm_scores
                && r.prefill_shared_softmax
                && r.share_target_rope
                && r.target_split_rope
                && r.request_position_budget
                && vision_safetensors_path.is_some(),
            other_diagnostic: r.target_only_reference
                || r.audit_batched_target_verify
                || r.audit_batched_numerics
                || r.profile_batched_target_verify
                || r.profile_prefill_sparse
                || r.reference_boundary_directory.is_some()
                || r.reference_serial_prefix
                || r.audit_reference_logits
                || r.diagnostic_q8_away
                || r.diagnostic_split_rope,
            stage_profile: stage_profile_requested,
            shared_weights: shared_target_weights.is_some() || shared_draft_weights.is_some(),
            target_capture: std::env::var_os("AUGUR_QWEN35_RAW_STATE_DIR").is_some(),
            private_capture: std::env::var_os("AUGUR_QWEN35_DRAFT_PRIVATE_DIR").is_some(),
            state_audit: r.audit_committed_state,
            sink: sink.is_some(),
        };
        if input_only {
            mode.validate_input_prefix(admission)?;
        } else {
            mode.validate(admission)?;
        }
    }
    if stage_profile_requested || (draft_top16_batch && top16_validation.is_none()) {
        let r = request.ok_or_else(|| invalid("stage profile requires a bounded request"))?;
        if !product_mode
            || !incremental_draft_context
            || validate_draft_context_reuse
            || context_rows != 2048
            || config.capacity != 196608
            || rolling_blocks != 1024
            || !r.long_generation_budget
            || r.max_new_tokens != 1024
            || !r.audit_committed_state
            || !r.batched_target_verify
            || !r.retain_full_accept_state
            || !r.exact_projection_tiles
            || !r.exact_output_head_batch
            || !r.batched_draft_context
            || !r.batched_draft_block
            || !r.draft_q8_tile4
            || r.target_only_reference
            || r.audit_batched_target_verify
            || r.audit_batched_numerics
            || r.profile_batched_target_verify
            || r.profile_prefill_sparse
            || r.reference_boundary_directory.is_some()
            || r.reference_serial_prefix
            || sink.is_some()
            || shared_target_weights.is_some()
            || shared_draft_weights.is_some()
            || vision_safetensors_path.is_none()
            || std::env::var_os("AUGUR_QWEN35_RAW_STATE_DIR").is_some()
            || std::env::var_os("AUGUR_QWEN35_DRAFT_PRIVATE_DIR").is_some()
        {
            return Err(invalid(
                "stage profile / top16 batch frozen diagnostic axes",
            ));
        }
    }
    #[cfg(feature = "qwen35-draft-private-state")]
    let mut private_audit = {
        use crate::dflash_q8_backbone_cuda::private_state::Audit;
        if std::env::var_os("AUGUR_QWEN35_DRAFT_PRIVATE_DIR").is_some() {
            let r = request.ok_or_else(|| invalid("private draft capture requires request"))?;
            if config.capacity != 196608
                || context_rows != 2048
                || !crate::dflash_q8_backbone_cuda::private_state::capture_budget_matches(
                    rolling_blocks,
                    r.max_new_tokens,
                    r.long_generation_budget,
                )
                || r.target_only_reference
                || !product_mode
                || !incremental_draft_context
                || validate_draft_context_reuse
                || vision_safetensors_path.is_none()
                || shared_draft_weights.is_some()
                || shared_target_weights.is_some()
                || sink.is_some()
                || !r.batched_draft_context
                || !r.batched_draft_block
                || !r.draft_q8_tile4
                || !r.retain_full_accept_state
                || !r.audit_committed_state
                || !r.batched_target_verify
                || !r.batched_prompt_capture
                || !r.prefill_dgemm_scores
                || !r.prefill_shared_softmax
                || !r.share_target_rope
                || !r.target_split_rope
                || !r.request_position_budget
                || r.profile_batched_target_verify
                || r.profile_prefill_sparse
                || r.audit_batched_target_verify
                || r.reference_serial_prefix
                || r.diagnostic_q8_away
                || r.diagnostic_split_rope
                || (!cfg!(feature = "qwen35-optimized-state-bridge")
                    && (r.exact_projection_tiles || r.exact_output_head_batch))
                || (cfg!(feature = "qwen35-optimized-state-bridge")
                    && (std::env::var_os("AUGUR_QWEN35_RAW_STATE_DIR").is_none()
                        || r.exact_projection_tiles != r.exact_output_head_batch))
            {
                return Err(invalid("private draft capture frozen axes"));
            }
            Some(Audit::from_env(&config.source_commit).map_err(invalid)?)
        } else {
            None
        }
    };

    #[cfg(feature = "qwen35-raw-committed-state")]
    let raw_options = raw_state::Options::from_env(
        config,
        request,
        context_rows,
        rolling_blocks,
        vision_safetensors_path,
        shared_draft_weights.is_some() || shared_target_weights.is_some(),
        sink.is_some(),
    )?;
    validate_diagnostic_split_rope_policy(
        request,
        augur_backend::cuda_q8_0::QWEN35_DIAGNOSTIC_SPLIT_ROPE,
    )?;
    validate_target_split_rope_policy(
        request,
        augur_backend::cuda_q8_0::QWEN35_TARGET_SPLIT_ROPE,
        augur_model_qwen35::TARGET_ROPE_SPLIT_HALF,
    )?;
    if config.device_ordinal != 0
        || !(1..=2048).contains(&context_rows)
        || config.prompt_repeat != 0
        || !(1..=if request.is_some_and(|r| r.long_generation_budget) {
            1024
        } else if request.is_some() {
            64
        } else {
            4
        })
            .contains(&rolling_blocks)
        || (request.is_none() && rolling_blocks > 1 && context_rows < B)
        || (validate_draft_context_reuse && (!incremental_draft_context || rolling_blocks < 2))
    {
        return Err(invalid("draft target diagnostic device/history/repeat"));
    }
    let long_generation = request.is_some_and(|r| r.long_generation_budget);
    if long_generation
        && (context_rows != 2048
            || config.capacity != 196608
            || vision_safetensors_path.is_none()
            || !incremental_draft_context)
    {
        return Err(invalid("long generation freezes full2048 history, 192K KV, Vision weights and incremental draft context"));
    }
    let config_bytes = read_stable_regular_file(&config.config_path, "target config")?;
    if let Some(weights) = shared_target_weights {
        weights.validate_request(config, &config_bytes)?;
    }
    if let Some(r) = request {
        validate_batch_execution_options(product_mode, r)?;
        if r.canonical_batched_full_core && !r.canonical_batched_norm {
            return Err(invalid("canonical full core requires canonical norm audit"));
        }
        if r.canonical_batched_norm && !r.native_quantized_batch_attention {
            return Err(invalid(
                "canonical norm requires native attention numerical audit",
            ));
        }
        batched_verify::validate_attention_ablation(
            r.native_quantized_batch_attention,
            r.canonical_batched_gdn_scalars,
            r.native_quantized_batch_ffn,
        )?;
        batched_verify::validate_gdn_ablation(
            r.canonical_batched_gdn_scalars,
            r.audit_batched_target_verify,
            r.audit_batched_numerics,
            r.profile_batched_target_verify,
        )?;
    }
    if request.is_some_and(|r| r.audit_batched_numerics && !r.audit_batched_target_verify) {
        return Err(invalid("batch numerical audit requires shadow audit"));
    }
    if request.is_some_and(|r| {
        r.native_quantized_batch_ffn
            && (!r.audit_batched_target_verify || r.profile_batched_target_verify)
    }) {
        return Err(invalid(
            "native quantized FFN requires shadow audit without legacy stage profile",
        ));
    }
    if request.is_some_and(|r| {
        r.profile_batched_target_verify
            && !r.audit_batched_target_verify
            && !r.batched_target_verify
    }) {
        return Err(invalid(
            "batched stage profile requires shadow or state-audited canonical execution",
        ));
    }
    if request.is_some_and(|r| r.audit_batched_target_verify) && !product_mode {
        return Err(invalid(
            "shadow batched audit requires canonical product-shaped mode",
        ));
    }
    let model = Qwen35Config::parse(&config_bytes)?;
    model.validate_exact_profile()?;
    let (eos_ids, generation_config_sha256, max_new_tokens) = if let Some(request) = request {
        if !(1..=if long_generation { 1024 } else { 512 }).contains(&request.max_new_tokens) {
            return Err(invalid(
                "request max_new_tokens outside selected generation profile",
            ));
        }
        let bytes = read_stable_regular_file(&request.generation_config_path, "generation config")?;
        let value: Value = serde_json::from_slice(&bytes).map_err(|e| invalid(e.to_string()))?;
        let values = match &value["eos_token_id"] {
            Value::Array(values) => values.clone(),
            value @ Value::Number(_) => vec![value.clone()],
            _ => return Err(invalid("generation config EOS ids missing")),
        };
        let mut ids = Vec::new();
        for value in values {
            let id = value
                .as_u64()
                .filter(|&id| id < V as u64)
                .ok_or_else(|| invalid("generation config invalid EOS id"))?
                as u32;
            if ids.contains(&id) {
                return Err(invalid("duplicate generation EOS id"));
            }
            ids.push(id);
        }
        if ids.is_empty() || ids.len() > 8 || !ids.contains(&model.eos_token_id) {
            return Err(invalid("generation EOS contract must include target EOS"));
        }
        (ids, Some(sha256_hex(&bytes)), request.max_new_tokens)
    } else {
        (Vec::new(), None, 0)
    };
    #[cfg(feature = "qwen35-session-continuation")]
    let max_new_tokens = native_reference
        .as_ref()
        .map_or(max_new_tokens, |r| r.max_new_tokens);
    #[cfg(feature = "qwen35-session-continuation")]
    let max_new_tokens = resident_output_ceiling.unwrap_or(max_new_tokens);
    let tokenizer_bytes = read_stable_regular_file(&config.tokenizer_path, "target tokenizer")?;
    let tokenizer_digest_reuse = submission_batch_switch(
        std::env::var_os("AUGUR_QWEN35_REQUEST_TOKENIZER_DIGEST_REUSE").as_deref(),
        shared_target_weights.is_some() && canonical_product_prefix && product_mode
            && request.is_some_and(|r| !r.target_only_reference))?;
    let (tokenizer, checked_tokenizer_digest) = if tokenizer_digest_reuse {
        let (parsed, digest) = shared_target_weights
            .ok_or_else(|| invalid("tokenizer digest reuse missing owner"))?
            .request_tokenizer_with_digest(&tokenizer_bytes)?;
        (parsed, Some(digest))
    } else {
        (match shared_target_weights {
            Some(owner) => owner.request_tokenizer(&tokenizer_bytes)?,
            None => Arc::new(Tokenizer::from_bytes(&tokenizer_bytes).map_err(|e| invalid(e.to_string()))?),
        }, None)
    };
    // tokenizer_bytes is immutable for this call. Reuse only its own verified
    // lease digest, never a path/mtime identity or a digest from another request.
    let tokenizer_digest_consumers = std::cell::Cell::new(0_u32);
    let tokenizer_digest = || {
        if let Some(digest) = &checked_tokenizer_digest {
            tokenizer_digest_consumers.set(tokenizer_digest_consumers.get() + 1);
            digest.clone()
        } else { sha256_hex(&tokenizer_bytes) }
    };
    let tokenize = || -> Result<Vec<usize>> {
        Ok(tokenizer
            .encode(config.prompt_text.as_str(), false)
            .map_err(|e| invalid(e.to_string()))?
            .get_ids()
            .iter()
            .map(|&x| x as usize)
            .collect())
    };
    #[cfg(feature = "qwen35-session-continuation")]
    let prompt = if let Some(tokens) = session_output.as_ref().and_then(|output| output.input_tokens()) {
        tokens.iter().map(|&token| token as usize).collect()
    } else {
        match native_reference.as_ref() {
            Some(r) => r.token_ids.iter().map(|&t| t as usize).collect(),
            None => tokenize()?,
        }
    };
    #[cfg(not(feature = "qwen35-session-continuation"))]
    let prompt = tokenize()?;
    #[cfg(feature = "qwen35-session-continuation")]
    if native_transition_directory.is_some() && prompt.len() != 8192 {
        return Err(invalid("transition audit requires frozen 8192-token input"));
    }
    let (capture_start, context_rows) = if request.is_some() {
        if let Some(mode) = top16_validation {
            #[cfg(feature = "qwen35-session-continuation")]
            if native_growing_session {
                // Public/service profile admission is narrower. Shared initialized
                // history supports the full arena; the checked rolling budget
                // below still reserves every verification row before submission.
                if !(1..=config.capacity).contains(&prompt.len()) {
                    return Err(invalid("bounded native growing prompt extent"));
                }
            } else if native_reference.is_none() {
                mode.validate_prompt(prompt.len())?;
            }
            #[cfg(not(feature = "qwen35-session-continuation"))]
            mode.validate_prompt(prompt.len())?;
        }
        if (stage_profile_requested || (draft_top16_batch && top16_validation.is_none()))
            && prompt.len() != 8192
        {
            return Err(invalid(
                "product stage profile requires exactly 8192 input tokens",
            ));
        }
        prompt_capture_extent(prompt.len(), context_rows)?
    } else {
        (prompt.len().saturating_sub(1), context_rows)
    };
    // The profile's history capacity is not the initialized feature extent.
    // Resolve only after prompt_capture_extent shrinks short/growing inputs;
    // freeze the result before allocation and never reread env during submit.
    let wide_history_capture = batched_verify::history_capture::requested(
        prefill_switches.history.as_deref(),
        integer_mma_ffn,
        integer_mma_input,
        context_rows,
    )?;
    if request.is_some_and(|r| r.reference_boundary_directory.is_some())
        && (context_rows < B || !context_rows.is_multiple_of(B))
    {
        return Err(invalid(
            "boundary trace requires whole eight-row prompt blocks",
        ));
    }
    let prefill_shared_softmax = request.is_some_and(|r| r.prefill_shared_softmax);
    let prefill_dgemm_scores = request.is_some_and(|r| r.prefill_dgemm_scores);
    let reference_serial_prefix = request.is_some_and(|r| r.reference_serial_prefix);
    if reference_serial_prefix && capture_start == 0 {
        return Err(invalid(
            "serial prefix diagnostic requires a nonempty prefix",
        ));
    }
    // The growing canonical path owns its zero-prefix/scalar bootstrap; it does
    // not select the shared-softmax prefix implementation for an empty prefix.
    validate_shared_prefill_extent(
        prefill_shared_softmax && !native_growing_session,
        capture_start,
    )?;
    let profile_prefill_sparse = request.is_some_and(|r| r.profile_prefill_sparse);
    if profile_prefill_sparse && capture_start == 0 {
        return Err(invalid("sparse prefill profile requires a real prefix"));
    }
    let batched_prompt_capture = request.is_some_and(|r| r.batched_prompt_capture);
    if batched_prompt_capture {
        batched_verify::validate_prompt_capture_extent(
            prompt.len(),
            context_rows,
            config.capacity,
            native_growing_session,
        )?;
    }
    let cursor = capture_start
        .checked_add(context_rows)
        .ok_or_else(|| invalid("conditioning cursor overflow"))?;
    let growing_history = if growing_history_requested {
        if context_rows < B && !native_growing_session {
            return Err(invalid(
                "growing history below eight real rows requires an unimplemented scalar bootstrap",
            ));
        }
        let history = GrowingHistory::from_extent(
            GrowingHistoryConfig::default_2048(),
            capture_start,
            context_rows,
        )
        .map_err(|error| invalid(error.to_string()))?;
        Some(history)
    } else {
        None
    };
    let request_position_budget = request.is_some_and(|r| r.request_position_budget);
    let position_budget = if native_growing_session {
        let output_budget = request.map(|r| r.max_new_tokens);
        #[cfg(feature = "qwen35-session-continuation")]
        let output_budget = native_reference
            .as_ref()
            .filter(|r| r.profile == committed_session::NativeAuditProfile::ResidentLong128)
            .map_or(output_budget, |r| Some(r.max_new_tokens));
        rolling_position_budget_with_history(
            cursor,
            rolling_blocks,
            output_budget,
            config.capacity,
            request_position_budget,
            long_generation,
            RollingHistoryBudget::GrowingInitialized {
                active_rows: context_rows,
            },
        )?
    } else if long_generation {
        rolling_position_budget_with_profile(
            cursor,
            rolling_blocks,
            request.map(|r| r.max_new_tokens),
            config.capacity,
            request_position_budget,
            true,
        )?
    } else {
        rolling_position_budget(
            cursor,
            rolling_blocks,
            request.map(|r| r.max_new_tokens),
            config.capacity,
            request_position_budget,
        )?
    };
    let end = position_budget.end;
    // Full-capacity tables are required by an owner that outlives one turn.
    // Per-turn verification admission still uses the bounded `end` above.
    let plan_end = {
        #[cfg(feature = "qwen35-session-continuation")]
        if session_output.is_some() {
            config.capacity
        } else {
            end
        }
        #[cfg(not(feature = "qwen35-session-continuation"))]
        {
            end
        }
    };
    if request.is_some_and(|r| r.target_only_reference) {
        if validate_draft_context_reuse
            || (vision_safetensors_path.is_some()
                && !raw_capture_enabled()
                && projection_corpus_rows == 0)
        {
            return Err(invalid("target-only semantic reference does not run draft reuse audit or load auxiliary weights"));
        }
        target_only_reference::validate_extent(cursor, end, max_new_tokens)?;
    }
    if prompt.is_empty()
        || prompt.iter().any(|&x| x >= V)
        || end > config.capacity
        || config.capacity > 262144
    {
        return Err(invalid("draft target capacity/prompt"));
    }
    let private_manifest = if shared_target_weights.is_none() {
        Some(GgufManifest::inspect(
            &config.gguf_path,
            GgufAuditOptions::default(),
        )?)
    } else {
        None
    };
    let manifest = shared_target_weights
        .map(|w| &w.manifest)
        .or(private_manifest.as_ref())
        .ok_or_else(|| invalid("target manifest owner missing"))?;
    Qwen35TensorSchema::validate_manifest_allowing_mtp(&model, &manifest)?;
    if cuda_device_properties(0)?.compute_capability != augur_backend::CudaComputeCapability::SM70 {
        return Err(invalid("SM70 required"));
    }
    let mut identity = Sha256::new();
    identity.update(b"QWEN35-DRAFT-TARGET-LIVE-V1\0");
    let attention_row_contract = small_attention_row_contract(
        std::env::var_os("AUGUR_QWEN35_SMALL_ATTENTION_TILE8_ROWS").is_some(),
        std::env::var_os("AUGUR_QWEN35_VERIFY_ATTENTION_VECTOR_ROWS").is_some(),
        std::env::var_os("AUGUR_QWEN35_VERIFY_ATTENTION_SERIAL_ROWS").is_some(),
        std::env::var_os("AUGUR_QWEN35_SMALL_ATTENTION_GQA8_MMA").is_some())?;
    if attention_row_contract != "legacy_mixed_vec_tile_v1" {
        if !llama_rotated_kv { return Err(invalid("small attention contract requires rotated owner")); }
        identity.update(attention_row_contract.as_bytes());
        identity.update(b"\0");
    }
    if growing_history_requested {
        identity.update(b"QWEN35-GROWING-HISTORY-2048-ACTIVE-EXTENT-V1\0");
    }
    if augur_backend::cuda_q8_0::QWEN35_TARGET_SPLIT_ROPE {
        identity.update(b"QWEN35-TARGET-SPLIT-HALF-ROPE-CONTRACT-V2\0");
    }
    if augur_backend::cuda_q8_0::QWEN35_DIAGNOSTIC_SPLIT_ROPE {
        identity.update(b"DIAGNOSTIC-TARGET-SPLIT-HALF-ROPE-V1\0");
    }
    if augur_backend::cuda_q8_0::Q8_1_DIAGNOSTIC_TIES_AWAY {
        identity.update(b"DIAGNOSTIC-Q8-DIVISION-TIES-AWAY-V1\0");
    }
    if reference_canonical_prefix {
        identity.update(b"DIAGNOSTIC-FULL-CANONICAL-REFERENCE-PREFIX-V1\0");
    }
    if let Some(rows) = tensor_ffn_down_rows {
        identity.update(b"QWEN35-PREFILL-FFN-DOWN-TENSOR-TILE896-Q8-F16-F32-V1\0");
        identity.update((rows as u64).to_le_bytes());
        if tensor_ffn_down_compensated {
            identity.update(b"QWEN35-PREFILL-DOWN-NVFP4-B128-FACTORED-HILO4096-V1\0");
        }
    }
    if let Some(rows) = integer_mma_ffn {
        if rows == 128 {
            identity.update(b"QWEN35-PREFILL-NVFP4-B128-INTEGER-MMA-CANONICAL-TILE384-V1\0");
        } else {
            identity.update(b"QWEN35-PREFILL-NVFP4-Q8-WIDE-INTEGER-MMA-CANONICAL-TILE384-V1\0");
            identity.update((rows as u64).to_le_bytes());
        }
    }
    if integer_mma_input {
        identity.update(b"QWEN35-PREFILL-Q8-INPUT-B512-INTEGER-MMA-CANONICAL-TILE384-V1\0");
    }
    if llama_f16_input {
        identity.update(b"QWEN35-PREFIX-ONLY-Q8-F16-F32-LLAMA-COALESCED-V1\0");
    }
    if llama_original_input {
        identity.update(b"QWEN35-PREFIX-ONLY-ORIGINAL-F32-F16-F32-LLAMA-V1\0");
    }
    if llama_full_original {
        identity.update(b"QWEN35-PREFIX-ONLY-LLAMA-FULL-ORIGINAL-POSTSCALE-SWIGLU-V1\0");
    }
    if llama_rotated_kv {
        identity.update(rotated_kv_identity.as_bytes());
        identity.update(b"\0PREFIX512-SCALAR-VERIFY8-ROTATED-CACHE-V1\0");
    }
    if llama_bf16_scalars {
        identity.update(b"QWEN35-PREFIX512-ALPHA-BETA-BF16-RNE-CUBLAS-F32-V1\0");
    }
    if llama_history_full {
        identity.update(b"QWEN35-INITIALIZED2048-LLAMA-ORIGINAL400-BF1696-V1\0");
    }
    if llama_global_chunks {
        identity.update(b"QWEN35-GLOBAL512-BOUNDARY-AUDIT-NATIVE-SCALAR-REMAINDER01-V1\0");
    }
    if llama_exact_input_rows {
        identity.update(b"QWEN35-EXACT-INPUT-GLOBAL512-REAL-TAIL1TO511-HISTORY-INTERSECTION-V1\0");
    }
    if prefill_batched_head_rope {
        identity.update(b"QWEN35-INPUT-ONLY-BATCHED-HEAD-NORM-ROPE-EXACT-V1\0");
    }
    if wide_history_capture {
        identity.update(b"QWEN35-INITIALIZED-HISTORY2048-B512-DIRECT-FEATURES-EXACT-V1\0");
    }
    identity.update(
        manifest
            .file_sha256
            .as_deref()
            .ok_or_else(|| invalid("GGUF hash missing"))?
            .as_bytes(),
    );
    identity.update(&config_bytes);
    identity.update(&tokenizer_bytes);
    for &t in &prompt {
        identity.update((t as u32).to_le_bytes());
    }
    if request.is_some() {
        identity.update(b"PROMPT-PREFILL-REQUEST-V1\0");
        if long_generation {
            identity.update(b"LONG-GENERATION-BUDGET-1024-V1\0");
            identity.update((rolling_blocks as u64).to_le_bytes());
        }
        identity.update((context_rows as u64).to_le_bytes());
        identity.update((config.capacity as u64).to_le_bytes());
        identity.update((max_new_tokens as u64).to_le_bytes());
        identity.update(
            generation_config_sha256
                .as_ref()
                .ok_or_else(|| invalid("request generation identity"))?
                .as_bytes(),
        );
    }
    #[cfg(feature = "qwen35-session-continuation")]
    if native_prefill_scalar_audit {
        identity.update(b"QWEN35-NATIVE-PREFILL-CANONICAL-GDN-SCALARS-AUDIT-V1\0");
    }
    #[cfg(feature = "qwen35-session-continuation")]
    if native_prefill_full_canonical_audit {
        identity.update(b"QWEN35-NATIVE-FULL-CANONICAL-PREFILL-AUDIT-V1\0");
    }
    let qpn_requested = target_weights::qpn_ffn_switch(std::env::var_os("AUGUR_QWEN35_QPN_FFN").as_deref())?;
    target_weights::require_ffn_owner_profile(qpn_requested, shared_target_weights.map(|w| w.qpn_ffn_enabled()))?;
    if qpn_requested { identity.update(b"QWEN35-QPN-ORIGINAL-F16-FFN-V1\0"); }
    let identity: [u8; 32] = identity.finalize().into();
    let before = cuda_device_allocation_snapshot();
    let setup = Instant::now();
    let stream = Rc::new(OwnedCudaStream::create(0)?);
    let qpn_workspace_start = Instant::now();
    let qpn_session = qpn_requested.then(|| target_weights::QpnScalarSession::new(Rc::clone(&stream))).transpose()?;
    let qpn_workspace_setup_ms = qpn_workspace_start.elapsed().as_secs_f64() * 1000.0;
    let mut reader = if shared_target_weights.is_none() {
        Some(manifest.open_tensor_reader(&config.gguf_path)?)
    } else {
        None
    };
    let embedding_descriptor = manifest
        .tensor("token_embd.weight")
        .ok_or_else(|| invalid("embedding"))?
        .clone();
    let first = if let Some(weights) = shared_target_weights {
        weights.embedding_row(prompt[0])?
    } else {
        linear::read_embedding_row(
            reader
                .as_mut()
                .ok_or_else(|| invalid("missing cold reader"))?,
            &embedding_descriptor,
            prompt[0],
        )?
    };
    let share_target_rope = request.is_some_and(|r| r.share_target_rope);
    let shared_rope = if share_target_rope {
        Some(Arc::new(FullRopeTables::new(
            &stream,
            model.rope_theta as f32,
            plan_end,
            full::ROTARY_DIM,
        )?))
    } else {
        None
    };
    let layer_plan_start = Instant::now();
    let mut plans = Vec::<DevicePlan>::with_capacity(64);
    for layer in 0..64 {
        #[cfg(feature = "qwen35-session-continuation")]
        if let Some(token) = native_cancellation {
            if let Err(error) = token.check("session plan allocation boundary") {
                // Driver does not own these partial plans yet. Drain explicitly
                // before local plan/rope owners unwind on cancellation.
                stream.synchronize()?;
                return Err(invalid(error.to_string()));
            }
        }
        let source = plans
            .last()
            .map(DevicePlan::final_hidden_f32_view)
            .transpose()?;
        let plan = if let Some(weights) = shared_target_weights {
            match &weights.layers[layer] {
                target_weights::LayerWeights::Linear(w) => {
                    DevicePlan::Linear(bind_linear_plan(&stream, w, &first, None, true, source, qpn_session.as_ref())?)
                }
                target_weights::LayerWeights::Full(w) => DevicePlan::Full(bind_full_plan(
                    &stream,
                    w,
                    &model,
                    config.capacity,
                    plan_end,
                    source,
                    shared_rope.clone(),
                    qpn_session.as_ref(),
                )?),
            }
        } else {
            let materials = load_layer_material_range(
                reader
                    .as_mut()
                    .ok_or_else(|| invalid("missing cold reader"))?,
                manifest,
                &model,
                layer..layer + 1,
            )?;
            match &materials[0] {
                LayerMaterial::Linear(m) => DevicePlan::Linear(build_linear_plan(
                    &stream,
                    m,
                    &first,
                    &project_linear(m, &first, model.rms_norm_eps as f32)?,
                    true,
                    source,
                )?),
                LayerMaterial::Full(m) => DevicePlan::Full(build_full_plan_with_rope(
                    &stream,
                    m,
                    &model,
                    config.capacity,
                    plan_end,
                    source,
                    shared_rope.clone(),
                )?),
            }
        };
        plans.push(plan);
    }
    // Layer leases now own every table through stream terminal completion.
    drop(shared_rope);
    // Each bound QPN program retains the plan, stream and shared workspace.
    // Release the construction lease here so request cleanup (or retained
    // session ownership) observes only the driver-owned execution resources.
    drop(qpn_session);
    let scalar_layer_plan_binding_ms = layer_plan_start.elapsed().as_secs_f64() * 1000.0;
    if llama_rotated_kv {
        let arena = RotatedFullAttention::new(&stream, config.capacity, llama_exact_input_rows)?;
        for plan in &mut plans {
            if let DevicePlan::Full(plan) = plan {
                plan.rotated_attention = Some(arena.clone());
            }
        }
    }
    let mut unique_rope = std::collections::HashSet::new();
    let mut physical_rope_bytes = 0usize;
    let mut logical_rope_bytes = 0usize;
    let mut full_layers = 0usize;
    for plan in &plans {
        if let DevicePlan::Full(plan) = plan {
            full_layers += 1;
            logical_rope_bytes += plan.rope_tables.len_bytes();
            if unique_rope.insert(Arc::as_ptr(&plan.rope_tables) as usize) {
                physical_rope_bytes += plan.rope_tables.len_bytes();
            }
        }
    }
    let mut target_rope_layout = json!({"schema":"qwen35-target-rope-layout-v1",
        "shared":share_target_rope,"positions":plan_end,"rotary_dim":full::ROTARY_DIM,
        "full_layers":full_layers,"physical_tables":unique_rope.len(),
        "physical_device_bytes":physical_rope_bytes,"logical_layer_bytes":logical_rope_bytes,
        "avoided_duplicate_device_bytes":logical_rope_bytes-physical_rope_bytes,
        "legacy_per_head_rope_ops_built":if cfg!(feature="qwen35-fused-head-norm-rope"){0}else{plan_end*full_layers*28},
        "math_policy":"existing_native_fp32_position_table_v1","occupied_context_qualification":false});
    if augur_backend::cuda_q8_0::QWEN35_TARGET_SPLIT_ROPE {
        target_rope_layout["schema"] = json!("qwen35-target-rope-layout-v2");
        target_rope_layout["coordinate_pairing"] = json!("rotary_prefix_split_half_v1");
        target_rope_layout["cpu_reference_pairing"] = json!("rotary_prefix_split_half_v1");
        target_rope_layout["head_dim"] = json!(256);
        target_rope_layout["fresh_state_required"] = json!(true);
    }
    let head = if let Some(weights) = shared_target_weights {
        bind_device_output_head_plan(
            &stream,
            &weights.head,
            plans
                .last()
                .ok_or_else(|| invalid("no plans"))?
                .final_hidden_view()?,
            model.rms_norm_eps as f32,
        )?
    } else {
        let head_material = load_output_head_material(
            reader
                .as_mut()
                .ok_or_else(|| invalid("missing cold reader"))?,
            manifest,
            &model,
        )?;
        build_device_output_head_plan(
            &stream,
            &head_material,
            plans
                .last()
                .ok_or_else(|| invalid("no plans"))?
                .final_hidden_view()?,
            H,
            V,
            model.rms_norm_eps as f32,
        )?
    };
    let alloc = |n| OwnedCudaBuffer::allocate(0, n).map_err(Qwen35MixedPrefixCudaError::from);
    let embedding = if let Some(weights) = shared_target_weights {
        weights.embedding.clone()
    } else {
        let embedding_bytes = reader
            .as_mut()
            .ok_or_else(|| invalid("missing cold reader"))?
            .read_tensor_bytes("token_embd.weight")?;
        if embedding_descriptor.ggml_type != GGML_TYPE_BF16 || embedding_bytes.len() != V * H * 2 {
            return Err(invalid("BF16 embedding geometry"));
        }
        let buffer = alloc(embedding_bytes.len())?;
        buffer.copy_from_host_sync(&stream, &embedding_bytes)?;
        std::rc::Rc::new(buffer)
    };
    let mut read_embedding = |token| -> Result<Vec<f32>> {
        if let Some(weights) = shared_target_weights {
            weights.embedding_row(token)
        } else {
            Ok(linear::read_embedding_row(
                reader
                    .as_mut()
                    .ok_or_else(|| invalid("missing cold reader"))?,
                &embedding_descriptor,
                token,
            )?)
        }
    };
    let mut backups = Vec::with_capacity(128);
    for p in &plans {
        for s in p.terminal_state_buffers() {
            backups.push(alloc(
                state_range(
                    matches!(p, DevicePlan::Full(_)),
                    s.len_bytes(),
                    cursor,
                    config.capacity,
                )?
                .1,
            )?);
        }
    }
    let round_profile_requested = product_stage_profile::requested(
        std::env::var_os("AUGUR_QWEN35_VERIFY_ROUND_PROFILE").as_deref(),
        product_mode && request.is_some_and(|r| r.batched_target_verify && !r.target_only_reference),
        max_new_tokens,
        verify_linear_graphs || std::env::var_os("AUGUR_QWEN35_VERIFY_LAYER_PROFILE").is_some(),
    )?;
    let initial_history_overlap_requested = submission_batch_switch(
        std::env::var_os("AUGUR_QWEN35_DFLASH_INITIAL_HISTORY_OVERLAP").as_deref(),
        product_mode && dflash_history_weight_panels && reuse_batched_plan
            && !round_profile_requested && !verify_linear_graphs
            && std::env::var_os("AUGUR_QWEN35_VERIFY_LAYER_PROFILE").is_none()
            && request.is_some_and(|r| r.batched_target_verify && !r.target_only_reference
                && !r.audit_batched_target_verify && !r.profile_batched_target_verify),
    )?;
    let mut d = Driver {
        #[cfg(feature = "qwen35-session-continuation")]
        session_cancellation: native_cancellation.cloned(),
        exact_projection_tiles: request.is_some_and(|r| r.exact_projection_tiles),
        projection_row_reuse,
        ffn_down_row_reuse,
        preserved_silu_q8,
        exact_output_head_batch: request.is_some_and(|r| r.exact_output_head_batch),
        draft_top16_batch,
        plans,
        head,
        embedding,
        noise_ids: alloc(B * 4)?,
        noise_device: alloc(B * H * 4)?,
        token: alloc(4)?,
        features: alloc(
            growing_history.map_or(context_rows, GrowingHistory::capacity_rows) * 5 * H * 4,
        )?,
        feature_shift_scratch: alloc(B * 5 * H * 4)?,
        feature_shift_submissions: 0,
        verified_features: if request.is_some_and(|r| r.retain_full_accept_state) {
            Some(alloc(B * 5 * H * 4)?)
        } else {
            None
        },
        batch_verify_audit: None,
        batch_verify_execution: None,
        round_profile: round_profile_requested.then(product_stage_profile::Profile::new).transpose()?,
        #[cfg(feature = "qwen35-session-continuation")]
        batched_suffix,
        feature_digest: alloc(QWEN35_KV8_PAGE_DIGEST_BYTES)?,
        candidate_ids: alloc(B * 16 * 4)?,
        unary: alloc(B * 16 * 4)?,
        transitions: alloc(B * 16 * 4)?,
        path: alloc(B * 4)?,
        verify_ids: alloc(B * 4)?,
        baseline_ids: alloc(B * 4)?,
        acceptance: alloc(QWEN35_DFLASH_ACCEPTANCE_RECEIPT_BYTES)?,
        backups,
        digest_parts: alloc(128 * QWEN35_KV8_PAGE_DIGEST_BYTES)?,
        digest: alloc(QWEN35_KV8_PAGE_DIGEST_BYTES)?,
        producer_event: OwnedCudaEvent::create(0)?,
        target_handoff_event: OwnedCudaEvent::create(0)?,
        stream,
        cursor,
        capacity: config.capacity,
        context_rows,
        growing_history,
        last_growth_plan: None,
        anchor: 0,
        generation: 1,
        identity,
        #[cfg(feature = "qwen35-session-checkpoint")]
        checkpoint_model_config_sha256: sha256_hex(&config_bytes),
        #[cfg(feature = "qwen35-session-checkpoint")]
        checkpoint_tokenizer_sha256: tokenizer_digest(),
        #[cfg(feature = "qwen35-session-checkpoint")]
        checkpoint_generation_config_sha256: generation_config_sha256.clone(),
        #[cfg(feature = "qwen35-session-checkpoint")]
        checkpoint_reuse_batched_plan: reuse_batched_plan,
        #[cfg(feature = "qwen35-session-checkpoint")]
        checkpoint_share_target_rope: share_target_rope,
        feature_sha: [0; 32],
        fast_feature_identity,
        replay_head_elision,
        parallel_receipt_digest: request.is_some_and(|r| r.parallel_receipt_digest),
        reference_boundary: request
            .and_then(|r| r.reference_boundary_directory.as_deref())
            .map(|directory| {
                reference_boundary::BoundaryTrace::new(
                    directory,
                    prompt.len() - 1,
                    reference_canonical_prefix,
                )
            })
            .transpose()?,
        canonical_prefill_rows,
        tensor_ffn_down_rows,
        tensor_ffn_down_compensated,
        integer_mma_ffn,
        integer_mma_input,
        llama_f16_input,
        llama_original_input,
        llama_full_original,
        llama_rotated_kv,
        planar_rotated_kv,
        llama_bf16_scalars,
        llama_history_full,
        llama_global_chunks,
        llama_exact_input_rows,
        prefill_batched_head_rope,
        batched_feature_copies,
        history_batched_feature_copies,
        history_feature_copy_enqueues: std::cell::Cell::new(0),
        batched_checkpoint_copies,
        batched_checkpoint_save_only,
        batched_replay_copies,
        shared_replay_fold_scalars,
        replay_checkpoint_read,
        replay_prefix_family,
        shared_verify_gdn_scalars,
        dflash_history_projection_reuse,
        dflash_b8_specialized,
        dflash_history_weight_panels,
        dflash_history_panel_token8,
        dflash_context_key_norm_rope,
        batched_output_head_prepare,
        head_prepare_stores,
        batched_target_top1,
        verify_rope_reuse,
        output_head_draft_prepare_batches: std::cell::Cell::new(0),
        output_head_target_prepare_batches: std::cell::Cell::new(0),
        target_top1_batches: std::cell::Cell::new(0),
        shared_replay_fold_layers: std::cell::Cell::new(0),
        checkpoint_read_fold_layers: std::cell::Cell::new(0),
        checkpoint_copy_batch_enqueues: std::cell::Cell::new(0),
        checkpoint_copy_logical_ranges: std::cell::Cell::new(0),
        checkpoint_copy_logical_bytes: std::cell::Cell::new(0),
        replay_copy_batch_enqueues: std::cell::Cell::new(0),
        batched_embedding_gather,
        feature_capture_enqueues: std::cell::Cell::new(0),
        feature_append_enqueues: std::cell::Cell::new(0),
        embedding_gather_enqueues: std::cell::Cell::new(0),
        prefill_batched_head_rope_launches: std::cell::Cell::new(0),
        prefill_batched_head_rope_layer_rows: std::cell::Cell::new(0),
        prefix_boundary_audit: batched_verify::prefill_input_integer::PrefixBoundaryAudit::from_env(
            prompt.len(), context_rows, canonical_product_prefix && canonical_prefill_rows == 512)?,
        wide_history_capture,
        trajectory_capture: false,
        poisoned: false,
        
        #[cfg(feature = "qwen35-raw-committed-state")]
        raw_state: raw_options
            .map(|options| {
                raw_state::Capture::new(
                    options,
                    config,
                    manifest
                        .file_sha256
                        .as_deref()
                        .ok_or_else(|| invalid("raw model hash"))?,
                    &config_bytes,
                    &tokenizer_bytes,
                    &prompt,
                )
            })
            .transpose()?,
    };
    let mask_ids = vec![crate::DFLASH_SEED_MASK_TOKEN; B]
        .iter()
        .flat_map(|id| id.to_le_bytes())
        .collect::<Vec<_>>();
    d.noise_ids.copy_from_host_sync(&d.stream, &mask_ids)?;
    reset_plans(&mut d.plans, &d.stream)?;
    // The prefill runner leaves its last supplied token unconsumed. Include
    // the first captured tail token as that sentinel, then teacher-force the
    // real prompt tail while capturing its five selected target layers.
    let prefill_prompt = if request.is_some() {
        &prompt[..capture_start + 1]
    } else {
        &prompt
    };
    let full_canonical_prefill = canonical_product_prefix || reference_canonical_prefix;
    let conditioning_start = Instant::now();
    let mut mode = if full_canonical_prefill {
        AutoregressivePrefillMode::Sequential
    } else {
        build_prefill_mode_from_embeddings(
            !reference_serial_prefix,
            &d.stream,
            Some(Rc::clone(&d.stream)),
            &d.plans,
            &mut read_embedding,
            prefill_prompt,
            &model,
            config.capacity,
            512,
            false,
            false,
            prefill_shared_softmax,
            false,
            false,
            false,
            false,
            2048,
        )?
    };
    #[cfg(feature = "qwen35-session-continuation")]
    let native_prefill_scalar_receipt = if native_prefill_scalar_audit {
        Some(prefill_scalar_audit::install(&d, &mut mode)?)
    } else {
        None
    };
    let start_event = OwnedCudaEvent::create(0)?;
    let stop_event = OwnedCudaEvent::create(0)?;
    let terminal = OwnedCudaEvent::create(0)?;
    let prefill_dgemm_receipt = if prefill_dgemm_scores && !full_canonical_prefill {
        #[cfg(feature = "qwen35-kv8-chunk-prefill")]
        {
            Some(prefill_dgemm_scores::install(
                &mut mode,
                &d.stream,
                config.capacity,
            )?)
        }
        #[cfg(not(feature = "qwen35-kv8-chunk-prefill"))]
        {
            return Err(invalid("DGEMM scores require chunk feature"));
        }
    } else {
        None::<Value>
    };
    if profile_prefill_sparse {
        let AutoregressivePrefillMode::Batched(prefill) = &mut mode else {
            return Err(invalid("sparse prefill profile requires batched prefix"));
        };
        prefill.schedule.sparse_profile = Some(sparse_prefill_profile::SparsePrefillProfile::new(
            &d.stream,
            &prefill.schedule,
        )?);
    }
    #[cfg(feature = "qwen35-session-continuation")]
    let mut native_prefill_full_canonical_receipt = None::<Value>;
    #[cfg(not(feature = "qwen35-session-continuation"))]
    let native_prefill_full_canonical_receipt = None::<Value>;
    let prefill_result = if full_canonical_prefill {
        #[cfg(feature = "qwen35-session-continuation")]
        {
            batched_verify::prefill_full_canonical_audit(
                &mut d,
                &prompt,
                model.rms_norm_eps as f32,
                model.rope_theta as f32,
            )
            .map(|(run, receipt)| {
                native_prefill_full_canonical_receipt = Some(receipt);
                run
            })
        }
        #[cfg(not(feature = "qwen35-session-continuation"))]
        {
            Err(invalid("canonical prefix diagnostic not compiled"))
        }
    } else {
        run_prefill_from_embeddings(
            &mut d.plans,
            &mode,
            &mut read_embedding,
            prefill_prompt,
            &d.stream,
            &start_event,
            &stop_event,
            &terminal,
        )
    };
    // mode owns workspace/events and will drop before Driver on this scope's
    // error path. Drain partial submissions before those owners are released.
    if prefill_result.is_err() {
        d.poisoned = true;
        let _ = d.stream.synchronize();
    }
    let mut prefill = prefill_result?;
    #[cfg(feature = "qwen35-session-continuation")]
    if let Some(directory) = native_transition_directory {
        if capture_start != 6144 || !batched_prompt_capture || !prefill_dgemm_scores {
            return Err(invalid("transition audit prefix geometry/policy"));
        }
        let mut input_hidden = Vec::with_capacity(512 * H);
        for &token in &prompt[capture_start..capture_start + 512] {
            input_hidden.extend(read_embedding(token)?);
        }
        d.cursor = capture_start;
        let audit = batched_verify::audit_prefill_transition(
            &mut d,
            mode,
            &input_hidden,
            &prompt[capture_start..capture_start + 512],
            model.rms_norm_eps as f32,
            model.rope_theta as f32,
            directory,
        )?;
        drop(d);
        let after = cuda_device_allocation_snapshot();
        let clean = after.live_bytes == before.live_bytes
            && after.live_allocations == before.live_allocations;
        if !clean {
            return Err(invalid("transition audit private cleanup"));
        }
        return Ok(json!({"schema":"qwen35-prefill-transition-native-v1",
            "status":"PASS_TRANSITION_CAPTURE_ONLY","audit":audit,
            "cleanup":{"pass":clean,"live_bytes":after.live_bytes},
            "generated_tokens":0,"session_retained":false,"product_ready":false,
            "same_session_qualification":false,"http_continuation":false}));
    }
    let sparse_prefill_profile_report = if profile_prefill_sparse {
        let AutoregressivePrefillMode::Batched(prefill) = &mode else {
            return Err(invalid("sparse prefill report owner"));
        };
        Some(
            prefill
                .schedule
                .sparse_profile
                .as_ref()
                .ok_or_else(|| invalid("sparse prefill report missing"))?
                .report()?,
        )
    } else {
        None
    };
    mode.release_workspace_with_report(&mut prefill)?;
    let prefill_ms = prefill.host_wall_ms;
    let capture_fc_requested=submission_batch_switch(
        std::env::var_os("AUGUR_QWEN35_CAPTURE_FC_PIPELINE").as_deref(),
        !input_only && initial_history_overlap_requested && dflash_history_panel_token8
            && batched_prompt_capture && d.llama_exact_input_rows && d.growing_history.is_some()
            && shared_draft_weights.is_some(),
    )?;
    let mut capture_fc=if capture_fc_requested && d.context_rows==2048 && max_new_tokens>1 {
        Some(crate::dflash_q8_backbone_cuda::capture_fc::CaptureFc::new(
            shared_draft_weights.ok_or_else(||invalid("capture FC resident weights"))?,
            &d.stream,d.active_feature_view()?,prompt.len()-d.context_rows,d.identity,d.generation,
        ).map_err(invalid)?)
    } else {None};
    let capture_fc_active=capture_fc.is_some();
    let prompt_capture_report = if batched_prompt_capture {
        Some(batched_verify::capture_prompt(
            &mut d,
            &prompt,
            model.rms_norm_eps as f32,
            model.rope_theta as f32,
            capture_fc.as_mut(),
        )?)
    } else {
        d.token
            .copy_from_host_sync(&d.stream, &(*prompt.last().unwrap() as u32).to_le_bytes())?;
        for row in 0..context_rows {
            if request.is_some() {
                d.token.copy_from_host_sync(
                    &d.stream,
                    &(prompt[capture_start + row] as u32).to_le_bytes(),
                )?;
            }
            d.bind_gather(view(&d.token)?)?.submit_for_plan(&d.stream)?;
            d.forward(capture_start + row, Some(row))?;
            if request.is_none() || row + 1 == context_rows {
                d.top1(view(&d.token)?)?;
            }
        }
        d.stream.synchronize()?;
        None
    };
    let prefetched_fc=capture_fc.take().map(|fc|fc.finish(&d.stream)).transpose().map_err(invalid)?;
    d.anchor = words(&d.stream, &d.token, 0, 1)?[0];
    if d.anchor as usize >= V {
        return Err(invalid("invalid live target anchor"));
    }
    let conditioning_wall_ms = conditioning_start.elapsed().as_secs_f64() * 1000.0;
    // Stop before creating any generated-token ledger or calling a publication
    // sink. No draft execution is needed to persist a consumed-input boundary.
    #[cfg(all(feature = "qwen35-session-continuation", feature = "qwen35-session-checkpoint", windows))]
    if input_only {
        let Some(committed_session::RetainedOutput::InputPrefix { tokens, bindings, destination })
            = session_output.take() else { return Err(invalid("input-prefix consumer disappeared")); };
        d.next_context_feature_digest()?;
        if request.is_some_and(|r| r.batched_target_verify) {
            d.batch_verify_execution = Some(BatchedVerifyExecution::new(
                &d, plan_end, model.rms_norm_eps as f32, model.rope_theta as f32,
                reuse_batched_plan, false, verify_linear_graphs,
            )?);
        }
        let retained = committed_session::InputPrefixOwner::seal(
            d, tokens, bindings, request.ok_or_else(|| invalid("input-prefix request policy missing"))?,
        )?;
        let boundary = retained.boundary.clone();
        *destination = Some(retained);
        return Ok(json!({
            "schema":"qwen35-native-input-prefix-prepare-v1",
            "input_tokens":tokens.len(),"generated_tokens":0,"published_tokens":0,
            "prefill_prefix_tokens":capture_start,"prefill_ms":prefill_ms,
            "conditioning_wall_ms":conditioning_wall_ms,"prompt_capture":prompt_capture_report,
            "prefill_workspace_released":prefill.batched_report.as_ref().is_none_or(|r| r.workspace_released_before_auxiliary_and_decode),
            "prefill_workspace":prefill.batched_report,"boundary":boundary,"draft_executed":false,
            "input_prefix_retained":true,"session_retained":false,
            "product_ready":false,"qualified":false,"semantic_qualification":false,"visible_tps":false
        }));
    }
    let initial_anchor = d.anchor;
    let native_fixed_history_logits = fixed_history_logits::capture(&d, &prompt)?;
    // Exact full-history boundary BEFORE generated input; native diagnostic only.
    #[cfg(feature = "qwen35-session-continuation")]
    let native_reference_initial_state = if let Some(r) = native_reference.as_ref() {
        let mut audit = committed_session::audit_target_state_to(&mut d, r.audit_directory)?;
        audit["consumed_input_digest"] = json!(committed_session::token_hash(r.token_ids));
        Some(audit)
    } else {
        None
    };
    let mut request_tokens = if request.is_some() {
        vec![initial_anchor]
    } else {
        Vec::new()
    };
    let mut request_stop = if request.is_some() {
        clip_publication(
            &mut request_tokens,
            Some(PublicationBudget {
                remaining: max_new_tokens,
                eos: &eos_ids,
            }),
        )?
    } else {
        None
    };
    let initial_feature_d2h_bytes = d.next_context_feature_digest()?;
    let initial_feature_fingerprint = sha256_hex(&d.feature_sha);
    if let Some(sink) = sink {
        let initial_anchor = Qwen35VerifiedCommit {
            initial_anchor: true,
            block_index: 0,
            generation_before: d.generation as usize,
            generation_after: d.generation as usize,
            cursor_before: d.cursor,
            cursor_after: d.cursor,
            token_ids: vec![d.anchor],
            stop_reason: request_stop
                .map(str::to_owned)
                .or_else(|| eos_ids.contains(&d.anchor).then(|| "eos".to_owned())),
        };
        dispatch_verified_commit(sink, initial_anchor)?;
    }
    let mut initial_decode_setup = product_stage_profile::InitialSetup::new();
    // Outside conditioning/rolling timing. Same noncryptographic device-state
    // fingerprint as the existing post-commit audit, not an elementwise proof.
    let initial_conditioning_state = if request.is_some_and(|r| r.audit_committed_state) {
        Some(d.state_digest()?)
    } else {
        None
    };
    let matched_raw_reference = (raw_capture_enabled() || projection_corpus_rows != 0)
        && request.is_some_and(|r| r.target_only_reference);
    let reference_metadata = if request.is_some_and(|r| r.target_only_reference) {
        let metadata = json!({
            "diagnostic_numerical_policy":if augur_backend::cuda_q8_0::Q8_1_DIAGNOSTIC_TIES_AWAY {
                Some(json!({"schema":"qwen35-diagnostic-q8-policy-v1",
                    "activation_quantization":"q8_1_division_ties_away_v1",
                    "explicit_acknowledgement":true,"default_policy_changed":false,
                    "scope":"all_shared_q8_1_activation_calls_in_this_reference_binary",
                    "kv_quantization_changed":false,"product_qualification":false}))
                } else if augur_backend::cuda_q8_0::QWEN35_DIAGNOSTIC_SPLIT_ROPE {
                    Some(json!({"schema":"qwen35-diagnostic-split-rope-policy-v1",
                        "target_rotary_pairing":"split_half_v1","explicit_acknowledgement":true,
                        "default_policy_changed":false,"activation_quantization_changed":false,
                        "scope":"target_full_attention_prefix_and_decode_in_this_reference_binary",
                        "product_qualification":false}))
                } else {None},
            "reference_boundary_trace":d.reference_boundary.as_ref().map(|trace|trace.finish(&d)).transpose()?,
            "native_fixed_history_logits":native_fixed_history_logits,
            "reference_canonical_prefix_requested":reference_canonical_prefix,
            "tensor_ffn_down_requested_rows":d.tensor_ffn_down_rows,
            "tensor_ffn_down_compensated_requested":d.tensor_ffn_down_compensated,
            "integer_mma_ffn_requested":d.integer_mma_ffn.is_some(),
            "integer_mma_input_requested":d.integer_mma_input,
            "llama_f16_input_prefix_requested":d.llama_f16_input,
            "llama_original_input_prefix_requested":d.llama_original_input,
            "llama_full_original_prefix_requested":d.llama_full_original,
            "llama_rotated_kv_requested":d.llama_rotated_kv,
            "small_attention_row_contract":attention_row_contract,
            "llama_bf16_scalars_prefix_requested":d.llama_bf16_scalars,
            "llama_full_initialized_history_requested":d.llama_history_full,
            "llama_global_boundary_audit_requested":d.llama_global_chunks,
            "llama_exact_input_rows_requested":d.llama_exact_input_rows,
            "full_attention_kv_basis":if d.llama_rotated_kv { rotated_kv_identity } else { "qwen35-q8-unrotated-v3-v4" },
            "wide_history_capture_requested":d.wide_history_capture,
            "native_prefill_full_canonical_audit":native_prefill_full_canonical_receipt,
            "source_commit":config.source_commit,"source_dirty":crate::build_identity::source_dirty_for(&config.source_commit),
            "binary_sha256":sha256_hex(&std::fs::read(std::env::current_exe().map_err(Qwen35Error::from)?).map_err(Qwen35Error::from)?),
            "target_gguf_sha256":manifest.file_sha256,"request_identity":sha256_hex(&identity),
            "capacity":config.capacity,"context_rows":context_rows,"prompt_tokens":prompt.len(),
            "reserved_end_position":end,"rolling_blocks":rolling_blocks,
            "parallel_receipt_digest_requested":d.parallel_receipt_digest,
            "receipt_digest_policy":if d.parallel_receipt_digest {"parallel_exact_v1"} else {"single_cta_v1"},
            "request_position_budget_requested":request_position_budget,
            "position_budget":{"schema":"qwen35-rolling-position-budget-v1",
                "policy":position_budget.policy,"verification_width":B,
                "reserved_positions":position_budget.reserved_positions,"end_position_exclusive":end,
                "full_block_lookahead_preserved":true,"generation_completion_guaranteed":false},
            "prefill_shared_softmax_requested":prefill_shared_softmax,
            "reference_serial_prefix_requested":reference_serial_prefix,
            "reference_prefix_policy":if capture_start==0 {"empty_prefix_v1"}
                else if reference_serial_prefix {"sequential_target_prefix_v1"} else {"batched_prefix_v1"},
            "batched_prompt_capture_requested":batched_prompt_capture,
            "target_rope_layout":target_rope_layout,
            "target_split_rope_requested":request.is_some_and(|r| r.target_split_rope),
            "request_output":{
                "prompt_token_ids":prompt,"prompt_text_sha256":sha256_hex(config.prompt_text.as_bytes()),
                "config_sha256":sha256_hex(&config_bytes),"tokenizer_sha256":tokenizer_digest(),
                "feature_start":capture_start,"feature_end":cursor,"prefill_prefix_tokens":capture_start,
                "conditioning_wall_ms":conditioning_wall_ms,"prefill_ms":prefill_ms,
                "prompt_capture":prompt_capture_report,
                "initial_feature_fingerprint":initial_feature_fingerprint,
                "initial_conditioning_state_fingerprint":initial_conditioning_state,
                "initial_conditioning_state_audit_d2h_bytes":32,
                "prefill_workspace_released":prefill.batched_report.as_ref().is_none_or(|r| r.workspace_released_before_auxiliary_and_decode),
                "prefill_workspace":prefill.batched_report,"generated_warmup_tokens":0,
                "initial_anchor":initial_anchor,"initial_anchor_d2h_bytes":4,
                "generation_config_sha256":generation_config_sha256,"eos_token_ids":eos_ids,
                "max_new_tokens":max_new_tokens
            }
        });
        if !matched_raw_reference {
            return target_only_reference::run(
                d,
                metadata,
                &tokenizer,
                &eos_ids,
                max_new_tokens,
                end,
                before,
                request.is_some_and(|r| r.audit_reference_logits),
                None,
            );
        }
        Some(metadata)
    } else {
        None
    };
    initial_decode_setup.mark("post_anchor_audit_ms")?;
    let draft_config = match shared_draft_weights {
        Some(weights) => weights.config().clone(),
        None => DFlashConfig::parse_f16_opt_in(&read_stable_regular_file(
            draft_config_path,
            "DFlash config",
        )?)?,
    };
    let mask_token_id = crate::DFLASH_SEED_MASK_TOKEN;
    let mut tokens = vec![mask_token_id; B];
    tokens[0] = d.anchor;
    let seed = DFlashDraftBlockInput {
        request_generation: d.generation,
        absolute_position_start: cursor,
        anchor_token: d.anchor,
        block_tokens: tokens,
        mask_token_id,
        target_feature_sha256: d.feature_sha,
    };
    let capture = DFlashTargetFeatureCaptureReceipt {
        schema: DFLASH_TARGET_CAPTURE_SCHEMA.into(),
        request_generation: d.generation,
        context_start_position: cursor - context_rows,
        context_len: context_rows,
        target_layer_ids: DFLASH_TARGET_CAPTURE_LAYER_IDS.to_vec(),
        target_hidden_size: H,
        feature_width: 5 * H,
        feature_elements: context_rows * 5 * H,
        source_target_model_sha256: identity,
        feature_sha256: d.feature_sha,
    };
    initial_decode_setup.mark("draft_metadata_ms")?;
    let noise = crate::embed_dflash_seed(&draft_config, &seed, |t| {
        read_embedding(t as usize).map_err(|e| e.to_string())
    })
    .map_err(|e| invalid(e.to_string()))?;
    initial_decode_setup.mark("noise_embedding_ms")?;
    let private_draft_weights = if shared_draft_weights.is_none() {
        let weights = DFlashQ8Backbone::load(q8_path, draft_config.clone())
            .map_err(|e| invalid(e.to_string()))?;
        Some(DFlashQ8CudaWeights::upload(&weights, config.device_ordinal).map_err(invalid)?)
    } else {
        None
    };
    let weights = shared_draft_weights
        .or(private_draft_weights.as_ref())
        .ok_or_else(|| invalid("missing immutable draft owner"))?;
    let q8_gguf_sha256 = weights.file_sha256().to_owned();
    initial_decode_setup.mark("draft_weight_lease_ms")?;
    let mut draft = if let Some(history) = d.growing_history {
        if let Some(fc)=prefetched_fc {
            DFlashQ8CudaBlock::from_resident_growing_device_features_with_fc(
                weights,&seed,&capture,d.active_feature_view()?,&noise,history,2048,&d.stream,fc,
            )
        } else { DFlashQ8CudaBlock::from_resident_growing_device_features(
            weights,
            &seed,
            &capture,
            d.active_feature_view()?,
            &noise,
            history,
            2048,
            &d.stream,
        ) }
    } else {
        DFlashQ8CudaBlock::from_resident_device_features(
            weights,
            &seed,
            &capture,
            d.features.view(0, d.features.len_bytes())?,
            &noise,
            cursor - context_rows,
            2048,
            &d.stream,
        )
    }
    .map_err(invalid)?;
    initial_decode_setup.mark("draft_device_owner_ms")?;
    let batched_draft_context = request.is_some_and(|r| r.batched_draft_context);
    if batched_draft_context {
        draft.enable_context_rows_launch().map_err(invalid)?;
    }
    let batched_draft_block = request.is_some_and(|r| r.batched_draft_block);
    if batched_draft_block {
        draft.enable_block_rows_launch().map_err(invalid)?;
    }
    let draft_q8_tile4 = request.is_some_and(|r| r.draft_q8_tile4);
    if draft_q8_tile4 {
        draft.enable_q8_tile4().map_err(invalid)?;
    }
    draft.enable_q8_b8_specialized(d.dflash_b8_specialized).map_err(invalid)?;
    draft.enable_history_prepare(d.dflash_history_projection_reuse, d.dflash_context_key_norm_rope)
        .map_err(invalid)?;
    draft.enable_history_weight_panels(d.dflash_history_weight_panels, d.dflash_history_panel_token8).map_err(invalid)?;
    initial_decode_setup.mark("draft_policy_ms")?;
    let dflash_resident = json!({
        "path": q8_path,
        "file_bytes": weights.file_bytes(),
        "resident_device_bytes": draft.resident_weight_bytes(),
        "payload_sha256": weights.file_sha256(),
        "device_resident": draft.resident_weight_bytes() > 0,
    });
    let vision_owner = if shared_vision.is_none() {
        ResidentAuxiliaryOwners::load(
            None,
            vision_safetensors_path,
            config.device_ordinal,
            &d.stream,
        )
        .map_err(invalid)?
    } else {
        None
    };
    let vision_resident = shared_vision
        .or(vision_owner.as_ref())
        .map(ResidentAuxiliaryOwners::report);
    #[cfg(feature = "qwen35-raw-committed-state")]
    if let Some(capture) = &d.raw_state {
        capture.initial(
            &d,
            &draft,
            &dflash_resident,
            &serde_json::to_value(&vision_resident).map_err(|e| invalid(e.to_string()))?,
        )?;
    }
    if matched_raw_reference {
        let mut metadata = reference_metadata.ok_or_else(|| invalid("raw reference metadata"))?;
        metadata["dflash_resident"] = dflash_resident;
        metadata["vision_resident"] =
            serde_json::to_value(&vision_resident).map_err(|e| invalid(e.to_string()))?;
        return target_only_reference::run(
            d,
            metadata,
            &tokenizer,
            &eos_ids,
            max_new_tokens,
            end,
            before,
            request.is_some_and(|r| r.audit_reference_logits),
            Some((draft, private_draft_weights, vision_owner)),
        );
    }
    initial_decode_setup.mark("auxiliary_audit_ms")?;
    if request.is_some_and(|r| r.audit_batched_target_verify) {
        d.batch_verify_audit = Some(BatchedVerifyAudit::new(
            &d,
            end,
            model.rms_norm_eps as f32,
            model.rope_theta as f32,
            request.is_some_and(|r| r.profile_batched_target_verify),
            request.is_some_and(|r| r.native_quantized_batch_ffn),
            request.is_some_and(|r| r.audit_batched_numerics),
            request.is_some_and(|r| r.canonical_batched_gdn_scalars),
            request.is_some_and(|r| r.native_quantized_batch_attention),
            request.is_some_and(|r| r.canonical_batched_norm),
            request.is_some_and(|r| r.canonical_batched_full_core),
        )?);
    }
    initial_decode_setup.mark("target_audit_ms")?;
    let initial_history_overlap_active = initial_history_overlap_requested && request_stop.is_none();
    if request.is_some_and(|r| r.batched_target_verify) {
        d.batch_verify_execution = Some(if initial_history_overlap_active {
            BatchedVerifyExecution::new_with_initial_history(
                &d, plan_end, model.rms_norm_eps as f32, model.rope_theta as f32, &mut draft,
            )?
        } else { BatchedVerifyExecution::new(
            &d,
            plan_end,
            model.rms_norm_eps as f32,
            model.rope_theta as f32,
            reuse_batched_plan,
            request.is_some_and(|r| r.profile_batched_target_verify),
            verify_linear_graphs,
        )? });
    }
    initial_decode_setup.mark("target_execution_ms")?;
    let mut trajectory_capture = if std::env::var_os("AUGUR_QWEN35_DFLASH_TRAJECTORY_DIR").is_some() {
        let prompt_ids = prompt.iter().map(|&id|u32::try_from(id).map_err(|_|invalid("trajectory prompt token range")))
            .collect::<Result<Vec<_>>>()?;
        trajectory::Capture::from_env(&d,&draft,&prompt_ids,product_mode)?
    } else { None };
    d.trajectory_capture = trajectory_capture.is_some();
    let mut transactions = Vec::with_capacity(rolling_blocks);
    let mut rolling_feature_d2h_bytes = 0usize;
    let mut rolling_rebind_boundaries = Vec::with_capacity(rolling_blocks.saturating_sub(1));
    let mut draft_context_preparations = Vec::with_capacity(rolling_blocks);
    #[cfg(feature = "qwen35-session-continuation")]
    let mut native_reference_private_capture = None::<Value>;
    #[cfg(all(
        feature = "qwen35-session-continuation",
        feature = "qwen35-draft-private-state"
    ))]
    let mut native_private_series = native_reference
        .as_ref()
        .and_then(|r| r.draft_audit_directory)
        .map(|p| session_private_audit::SessionPrivateAudit::new(p, "session-fresh-recompute"))
        .transpose()?;
    initial_decode_setup.mark("rolling_state_ms")?;
    let initial_decode_setup = initial_decode_setup.report()?;
    let rolling_start = Instant::now();
    for block_index in 0..rolling_blocks {
        if request_stop.is_some() {
            break;
        }
        if d.cursor.checked_add(B).is_none_or(|next| next > end) {
            return Err(invalid(
                "full verification exceeds admitted position budget",
            ));
        }
        let rows = draft.context_projection_rows();
        if let Some(profile) = &d.round_profile {
            profile.begin(&d, &draft)?;
            profile.mark(0, 0, draft.stream())?;
        }
        
        let prepare_start = Instant::now();
        if !(initial_history_overlap_active && block_index == 0) {
            draft.prepare_context().map_err(invalid)?;
        }
        if let Some(profile) = &d.round_profile {
            profile.mark(0, 1, draft.stream())?;
        }
        
        let prepare_enqueue_ms = prepare_start.elapsed().as_secs_f64() * 1000.0;
        let submit_start = Instant::now();
        draft.submit().map_err(invalid)?;
        if let Some(profile) = &d.round_profile {
            profile.mark(0, 2, draft.stream())?;
        }
        let draft_enqueue_ms = submit_start.elapsed().as_secs_f64() * 1000.0;
        if let Some(capture) = &mut trajectory_capture { capture.begin(&d,&draft)?; }
        #[cfg(all(
            feature = "qwen35-session-continuation",
            feature = "qwen35-draft-private-state"
        ))]
        if let Some(audit) = &mut native_private_series {
            let reference = native_reference
                .as_ref()
                .ok_or_else(|| invalid("private reference absent"))?;
            audit.capture_selected(
                &draft,
                &d,
                block_index,
                transactions.last(),
                reference.token_ids,
                &request_tokens,
            )?;
            native_reference_private_capture = audit.first_binding();
        }
        
        #[cfg(feature = "qwen35-optimized-state-bridge")]
        if let Some(capture) = &d.raw_state {
            capture.bridge_before_verify(&d, &draft, block_index, transactions.last())?;
        }
        #[cfg(feature = "qwen35-draft-private-state")]
        if let Some(audit) = &mut private_audit {
            audit
                .capture_selected(
                    &draft,
                    block_index,
                    d.cursor,
                    d.generation,
                    d.anchor,
                    transactions.last(),
                    &dflash_resident,
                    &serde_json::to_value(&vision_resident).map_err(|e| invalid(e.to_string()))?,
                )
                .map_err(invalid)?;
        }
        let audit_start = Instant::now();
        let audit_fingerprint = if validate_draft_context_reuse {
            Some(sha256_hex(
                &draft.audit_full_context_recompute().map_err(invalid)?,
            ))
        } else {
            None
        };
        let audit_wall_ms =
            validate_draft_context_reuse.then(|| audit_start.elapsed().as_secs_f64() * 1000.0);
        draft_context_preparations.push(json!({
            "new_rows": rows, "reused_rows": d.active_context_rows().saturating_sub(rows),
            "primary_prepare_kernel_launches":if batched_draft_context {22}else{92*rows},
            // Only Q8 projection/RMSNorm/RoPE, not conv/attention/selector-head tail.
            "primary_block_row_kernel_launches":if batched_draft_block {77}else{3656},
            "context_start": d.active_context_start(), "active_context_rows": d.active_context_rows(),
            "physical_feature_capacity_rows": d.growing_history.map_or(context_rows, GrowingHistory::capacity_rows),
            "generation": d.generation,
            "prepare_enqueue_ms": prepare_enqueue_ms, "draft_enqueue_ms": draft_enqueue_ms,
            "audit_wall_ms": audit_wall_ms, "audit_fingerprint": audit_fingerprint,
            "audit_d2h_bytes": if validate_draft_context_reuse { 64 } else { 0 },
            "audit_scheme": "17_plane_device_fingerprint_v1_noncryptographic",
        }));
        let budget = request.map(|_| PublicationBudget {
            remaining: max_new_tokens - request_tokens.len(),
            eos: &eos_ids,
        });
        let mut report = if product_mode {
            d.verify_product(&draft, budget)?
        } else {
            d.verify(&draft, budget)?
        };
        if let Some(capture) = &mut trajectory_capture { capture.commit(&d,&report)?; }
        #[cfg(feature = "qwen35-session-continuation")]
        if session_output.is_some() {
            let audited = committed_session::check_committed_path(
                &d,
                &mut report,
                *request_tokens
                    .last()
                    .ok_or_else(|| invalid("session previous anchor"))?,
                fast_session_path,
            )?;
            report["session_input_path_audited"] = json!(audited);
        }
        if request.is_some_and(|r| r.audit_committed_state) {
            let audit_start = Instant::now();
            report["committed_state_fingerprint"] = json!(d.state_digest()?);
            report["committed_state_audit_d2h_bytes"] = json!(QWEN35_KV8_PAGE_DIGEST_BYTES);
            report["committed_state_audit_wall_ms"] =
                json!(audit_start.elapsed().as_secs_f64() * 1000.0);
        }
        
        #[cfg(feature = "qwen35-raw-committed-state")]
        if let Some(capture) = &d.raw_state {
            capture.candidate_commit(
                &d,
                &draft,
                &report,
                block_index + 1 == rolling_blocks
                    || report["publication_stop_reason"].as_str().is_some(),
            )?;
        }
        if request.is_some() {
            let key = if product_mode {
                "emitted_token_ids"
            } else {
                "emitted_tokens"
            };
            for token in report[key]
                .as_array()
                .ok_or_else(|| invalid("request tokens missing"))?
            {
                request_tokens.push(
                    token
                        .as_u64()
                        .ok_or_else(|| invalid("request token invalid"))?
                        as u32,
                );
            }
            request_stop = match report["publication_stop_reason"].as_str() {
                Some("eos") => Some("eos"),
                Some("max_new_tokens") => Some("max_new_tokens"),
                None => None,
                _ => return Err(invalid("unknown publication stop")),
            };
            if let Some(sink) = sink {
                let token_ids = report[key]
                    .as_array()
                    .ok_or_else(|| invalid("request tokens missing for commit sink"))?
                    .iter()
                    .map(|token| {
                        token
                            .as_u64()
                            .ok_or_else(|| invalid("request token invalid for commit sink"))
                            .map(|token| token as u32)
                    })
                    .collect::<Result<Vec<_>>>()?;
                let stop_reason = report["publication_stop_reason"]
                    .as_str()
                    .map(str::to_owned);
                let commit = Qwen35VerifiedCommit {
                    initial_anchor: false,
                    block_index,
                    generation_before: report["generation_before"]
                        .as_u64()
                        .ok_or_else(|| invalid("commit generation_before missing"))?
                        as usize,
                    generation_after: report["generation_after"]
                        .as_u64()
                        .ok_or_else(|| invalid("commit generation_after missing"))?
                        as usize,
                    cursor_before: report["cursor_before"]
                        .as_u64()
                        .ok_or_else(|| invalid("commit cursor_before missing"))?
                        as usize,
                    cursor_after: report["cursor_after"]
                        .as_u64()
                        .ok_or_else(|| invalid("commit cursor_after missing"))?
                        as usize,
                    token_ids,
                    stop_reason,
                };
                dispatch_verified_commit(sink, commit)?;
            }
        }
        transactions.push(report);
        if block_index + 1 == rolling_blocks || request_stop.is_some() {
            break;
        }
        let rebind_before = cuda_host_boundary_snapshot();
        let growth_plan = d.last_growth_plan.take();
        let tail_feature_bytes = d.next_context_feature_digest()?;
        transactions
            .last_mut()
            .ok_or_else(|| invalid("no feature transaction"))?["committed_feature_fingerprint"] =
            json!(sha256_hex(&d.feature_sha));
        rolling_feature_d2h_bytes = rolling_feature_d2h_bytes
            .checked_add(tail_feature_bytes)
            .ok_or_else(|| invalid("rolling feature byte overflow"))?;
        let mut next_tokens = vec![mask_token_id; B];
        next_tokens[0] = d.anchor;
        let next_seed = DFlashDraftBlockInput {
            request_generation: d.generation,
            absolute_position_start: d.cursor,
            anchor_token: d.anchor,
            block_tokens: next_tokens,
            mask_token_id,
            target_feature_sha256: d.feature_sha,
        };
        let next_capture = DFlashTargetFeatureCaptureReceipt {
            schema: DFLASH_TARGET_CAPTURE_SCHEMA.into(),
            request_generation: d.generation,
            context_start_position: d.active_context_start(),
            context_len: d.active_context_rows(),
            target_layer_ids: DFLASH_TARGET_CAPTURE_LAYER_IDS.to_vec(),
            target_hidden_size: H,
            feature_width: 5 * H,
            feature_elements: d.active_context_rows() * 5 * H,
            source_target_model_sha256: identity,
            feature_sha256: d.feature_sha,
        };
        let next_noise = d.prepare_noise_device(view(&d.token)?)?;
        // The feature digest drains earlier target work, not the noise gather
        // just submitted above. Bind the actual final producer epoch explicitly.
        d.target_handoff_event
            .record_for_submission(&d.stream, d.stream.submission_receipt()?)?;
        draft.stream().wait_event(&d.target_handoff_event)?;
        if let Some(plan) = growth_plan {
            let history = d
                .growing_history
                .ok_or_else(|| invalid("missing growing history after plan"))?;
            draft
                .rebind_growing_committed_device_features_with_noise(
                    &next_seed,
                    &next_capture,
                    d.active_feature_view()?,
                    next_noise,
                    plan,
                    history,
                )
                .map_err(invalid)?;
        } else if incremental_draft_context {
            let committed = transactions
                .last()
                .ok_or_else(|| invalid("missing transaction"))?["committed_token_count"]
                .as_u64()
                .ok_or_else(|| invalid("missing committed count"))?
                as usize;
            draft
                .rebind_committed_device_features_with_noise(
                    &next_seed,
                    &next_capture,
                    d.features.view(0, d.features.len_bytes())?,
                    next_noise,
                    committed,
                )
                .map_err(invalid)?;
        } else {
            draft
                .rebind_device_features_with_noise(
                    &next_seed,
                    &next_capture,
                    d.features.view(0, d.features.len_bytes())?,
                    next_noise,
                    d.cursor - context_rows,
                    2048,
                )
                .map_err(invalid)?;
        }
        rolling_rebind_boundaries.push(boundary_delta(
            rebind_before,
            cuda_host_boundary_snapshot(),
        )?);
    }
    let terminal_before = cuda_host_boundary_snapshot();
    d.stream.synchronize()?;
    let rolling_wall_ms = rolling_start.elapsed().as_secs_f64() * 1000.0;
    let rolling_terminal_boundaries =
        boundary_delta(terminal_before, cuda_host_boundary_snapshot())?;
    let setup_ms = setup.elapsed().as_secs_f64() * 1000.0;
    let checkpoint_bytes = d.backups.iter().map(|b| b.len_bytes()).sum::<usize>();
    let report = transactions.first().cloned().unwrap_or(Value::Null);
    let request_output = if request.is_some() {
        let visible_end = request_tokens
            .iter()
            .position(|id| eos_ids.contains(id))
            .unwrap_or(request_tokens.len());
        let visible_tokens = &request_tokens[..visible_end];
        Some(json!({
            "contract":"prompt_prefill_request_v1", "generated_warmup_tokens":0,
            "prompt_token_ids":prompt,
            "prompt_text_sha256":sha256_hex(config.prompt_text.as_bytes()),
            "config_sha256":sha256_hex(&config_bytes),
            "tokenizer_sha256":tokenizer_digest(),
            "feature_start":capture_start, "feature_end":cursor,
            "prefill_prefix_tokens":capture_start, "conditioning_wall_ms":conditioning_wall_ms,
            "prompt_capture":prompt_capture_report,
            "sparse_prefill_profile":sparse_prefill_profile_report,
            "initial_feature_fingerprint":initial_feature_fingerprint,
            "initial_conditioning_state_fingerprint":initial_conditioning_state,
            "initial_conditioning_state_audit_d2h_bytes":if initial_conditioning_state.is_some(){32}else{0},
            "prefill_workspace_released":prefill.batched_report.as_ref().is_none_or(|r| r.workspace_released_before_auxiliary_and_decode),
            "prefill_workspace":prefill.batched_report,
            "initial_anchor":initial_anchor, "initial_anchor_d2h_bytes":4,
            "generation_config_sha256":generation_config_sha256, "eos_token_ids":eos_ids,
            "max_new_tokens":max_new_tokens, "token_ids":request_tokens,
            "visible_token_ids":visible_tokens,
            "decoded_text":tokenizer.decode(visible_tokens, false).map_err(|e| invalid(e.to_string()))?,
            "stop_reason":request_stop.unwrap_or("block_budget"),
            "completed_blocks":transactions.len(),
            "first_token_in_decode_wall":false,
            "request_user_wall_qualified":false,
        }))
    } else {
        None
    };
    let final_cursor = d.cursor;
    let trajectory_report = trajectory_capture.map(|capture|capture.finish()).transpose()?;
    d.trajectory_capture = false;
    let prefill_batched_head_rope_launches = d.prefill_batched_head_rope_launches.get();
    let prefill_batched_head_rope_layer_rows = d.prefill_batched_head_rope_layer_rows.get();
    let terminal_growing_history = d.growing_history;
    if terminal_growing_history.is_some_and(|history| history.cursor() != final_cursor) {
        return Err(invalid("terminal growing history cursor mismatch"));
    }
    let output_head_projection_policy = d
        .batch_verify_execution
        .as_ref()
        .map_or("canonical_serial_v1", |executor| {
            executor.output_head_policy()
        });
    let output_head_weight_layout = d.head.output_weight_layout.0;
    let output_head_weight_physical_sha256 = d.head.output_weight_physical_sha256.clone();
    let output_head_weight_relayout_ms = d.head.output_weight_relayout_ms;
    let output_head_workspace_bytes = d
        .batch_verify_execution
        .as_ref()
        .map_or(0, |executor| executor.output_head_workspace_bytes());
    #[cfg(feature = "qwen35-raw-committed-state")]
    let raw_report = d
        .raw_state
        .as_ref()
        .map(|capture| capture.finish(&d))
        .transpose()?;
    #[cfg(feature = "qwen35-draft-private-state")]
    let private_report = private_audit
        .map(|audit| audit.finish())
        .transpose()
        .map_err(invalid)?;
    let feature_shift_counts = (d.feature_shift_submissions, draft.feature_shift_submissions());
    let submission_batch_counts = (d.feature_capture_enqueues.get(),
        d.feature_append_enqueues.get(), d.embedding_gather_enqueues.get(),
        d.history_feature_copy_enqueues.get());
    let feature_shift_scratch_bytes = d.feature_shift_scratch.len_bytes();
    let state_copy_batch_counts = (d.checkpoint_copy_batch_enqueues.get(),
        d.checkpoint_copy_logical_ranges.get(), d.replay_copy_batch_enqueues.get(),
        d.checkpoint_copy_logical_bytes.get());
    let shared_replay_fold_layers = d.shared_replay_fold_layers.get();
    let checkpoint_read_fold_layers = d.checkpoint_read_fold_layers.get();
    let dflash_history_prepare = draft.history_prepare_report();
    let output_head_batching = d.output_head_batching_report();
    #[cfg(feature = "qwen35-session-continuation")]
    let session_checkpoint = if let Some(committed_session::RetainedOutput::Generated(session_id, destination)) = session_output {
        let output = request_output
            .as_ref()
            .ok_or_else(|| invalid("session output missing"))?;
        let retained = committed_session::CommittedSession::seal_initial(
            d,
            draft,
            session_id,
            output,
            &transactions,
            resident_output_ceiling.is_some(),
            fast_session_path,
        )?;
        let checkpoint = retained.checkpoint.clone();
        *destination = Some(retained);
        Some(checkpoint)
    } else {
        drop(draft);
        drop(d);
        None
    };
    #[cfg(not(feature = "qwen35-session-continuation"))]
    {
        drop(draft);
        drop(d);
    }
    drop(private_draft_weights);
    drop(vision_owner);
    let after = cuda_device_allocation_snapshot();
    let clean =
        after.live_bytes == before.live_bytes && after.live_allocations == before.live_allocations;
    let retained_session = {
        #[cfg(feature = "qwen35-session-continuation")]
        {
            session_checkpoint.is_some()
        }
        #[cfg(not(feature = "qwen35-session-continuation"))]
        {
            false
        }
    };
    if !clean && !retained_session {
        return Err(invalid(format!(
            "draft-target cleanup did not close: live_bytes {} -> {}, live_allocations {} -> {}",
            before.live_bytes, after.live_bytes, before.live_allocations, after.live_allocations
        )));
    }
    let mut result = json!({"schema":if request.is_some() { "qwen35-q8-dflash-request-rolling-v1" } else if product_mode { "qwen35-q8-dflash-product-rolling-v1" } else { "qwen35-q8-live-draft-target-verify-rollback-v1" },"status":if request.is_some() { "PASS_DIAGNOSTIC_Q8_DFLASH_REQUEST_ROLLING" } else if product_mode { "PASS_DIAGNOSTIC_Q8_DFLASH_PRODUCT_ROLLING" } else { "PASS_COMPONENT_Q8_LIVE_TARGET_VERIFY_ROLLBACK" },
        "request_output":request_output,
        "initial_decode_setup":initial_decode_setup,
        "request_plan_setup":{"schema":"qwen35-request-plan-setup-host-v1",
            "qpn_workspace_setup_ms":qpn_workspace_setup_ms,
            "scalar_layer_plan_binding_ms":scalar_layer_plan_binding_ms,
            "scope":"disjoint_host_subintervals_within_setup_including_draft_submit_ms",
            "includes_gpu_runtime_measurement":false},
        "initial_history_overlap_requested":initial_history_overlap_requested,
        "initial_history_overlap_active":initial_history_overlap_active,
        "capture_fc_requested":capture_fc_requested,"capture_fc_active":capture_fc_active,
        "verified_state_reuse_policy":if request.is_some_and(|r| r.retain_full_accept_state) { "full_accept_unclipped_v1" } else { "disabled" },
        "verified_feature_staging_bytes":if request.is_some_and(|r| r.retain_full_accept_state) { B*5*H*4 } else { 0 },
        "batched_target_verification_audit":request.is_some_and(|r| r.audit_batched_target_verify),
        "batched_target_publication_enabled":request.is_some_and(|r| r.batched_target_verify),
        "fast_session_path_requested":fast_session_path_requested,
        "fast_session_path_active":fast_session_path,
        "fast_feature_identity_requested":fast_feature_identity_requested,
        "fast_feature_identity_active":fast_feature_identity,
        "verify_linear_graphs_requested":verify_linear_graphs,
        "replay_head_elision_requested":replay_head_elision,
        "committed_state_audit_enabled":request.is_some_and(|r| r.audit_committed_state),
        "canonical_target_profile_requested":request.is_some_and(|r| r.batched_target_verify && r.profile_batched_target_verify),
        "batched_plan_reuse_enabled":batched_plan_reuse_report_value(reuse_batched_plan),
        "target_rope_layout":target_rope_layout,
        "target_split_rope_requested":request.is_some_and(|r| r.target_split_rope),
        "request_position_budget_requested":request_position_budget,
        "position_budget":{"schema":"qwen35-rolling-position-budget-v1",
            "policy":position_budget.policy,"verification_width":B,
            "reserved_positions":position_budget.reserved_positions,
            "end_position_exclusive":end,"full_block_lookahead_preserved":true,
            "generation_completion_guaranteed":false},
        "prefill_shared_softmax_requested":prefill_shared_softmax,
        "prefill_dgemm_scores_requested":prefill_dgemm_scores,
        "prefill_dgemm_scores":prefill_dgemm_receipt,
        "batched_prompt_capture_requested":batched_prompt_capture,
        "profile_prefill_sparse_requested":profile_prefill_sparse,
        "target_verification_policy":if request.is_some_and(|r| r.batched_target_verify){"canonical_batch_v1"}else{"canonical_serial_v1"},
        "source_commit":config.source_commit,"source_dirty":crate::build_identity::source_dirty_for(&config.source_commit),"binary_sha256":sha256_hex(&std::fs::read(std::env::current_exe().map_err(Qwen35Error::from)?).map_err(Qwen35Error::from)?),
        "target_gguf_sha256":manifest.file_sha256,"q8_gguf_sha256":q8_gguf_sha256,
        "prompt_tokens":prompt.len(),"capacity":config.capacity,"context_rows":context_rows,"full_draft_history":context_rows==2048,
        "initial_feature_transfer_scope":"same_live_target_setup_capture_only; rolling_device_feature_and_noise_handoff_enabled",
        "initial_feature_d2h_bytes":initial_feature_d2h_bytes,
        "initial_feature_fingerprint_scheme":if fast_feature_identity {"live_owner_metadata_identity_v1_not_payload_digest"} else {"qwen35_kv8_page_digest_v1_device_receipt_non_sha256"},
        "rolling_feature_d2h_bytes":rolling_feature_d2h_bytes,
        "rolling_feature_fingerprint_scheme":if fast_feature_identity {"live_owner_metadata_identity_v1_not_payload_digest"} else {"qwen35_kv8_page_digest_v1_device_receipt_non_sha256"},
        "rolling_blocks":rolling_blocks,"product_mode":product_mode,"rolling_rebind_boundaries":rolling_rebind_boundaries,
        "rolling_contract":"committed_feature_window_v2",
        "rolling_wall_scope":"first_draft_prepare_to_final_target_completion",
        "rolling_wall_ms":rolling_wall_ms,"rolling_terminal_boundaries":rolling_terminal_boundaries,
        "feature_shift_scratch_bytes":feature_shift_scratch_bytes,
        "feature_shift_policy":"in_place_u32_column_tile16_v1",
        "feature_shift_target_submissions":feature_shift_counts.0,
        "feature_shift_draft_submissions":feature_shift_counts.1,
        "draft_context_policy":if incremental_draft_context { "incremental_committed_v1" } else { "full_recompute" },
        "batched_draft_context_requested":batched_draft_context,
        "draft_context_launch_policy":if batched_draft_context {"independent_rows_same_arithmetic_v1"} else {"per_row_scalar_v1"},
        "batched_draft_block_requested":batched_draft_block,
        "draft_block_launch_policy":if batched_draft_block {"independent_rows_same_arithmetic_v1"} else {"per_row_scalar_v1"},
        "draft_q8_tile4_requested":draft_q8_tile4,
        "parallel_receipt_digest_requested":request.is_some_and(|r| r.parallel_receipt_digest),
        "exact_projection_tiles_requested":request.is_some_and(|r| r.exact_projection_tiles),
        "projection_row_reuse_requested":projection_row_reuse,
        "ffn_down_row_reuse_requested":ffn_down_row_reuse,
        "preserved_silu_q8_requested":preserved_silu_q8,
        "exact_output_head_batch_requested":request.is_some_and(|r| r.exact_output_head_batch),
        "output_head_projection_policy":output_head_projection_policy,
        "output_head_workspace_bytes":output_head_workspace_bytes,
        "output_head_weight_layout":output_head_weight_layout,
        "output_head_weight_physical_sha256":output_head_weight_physical_sha256,
        "output_head_weight_relayout_ms":output_head_weight_relayout_ms,
        "target_projection_tile_policy":if projection_row_reuse {"canonical_full8_input_gate_up_row2_exact_fallback_v1"} else if request.is_some_and(|r| r.exact_projection_tiles) {"canonical_backbone_prompt_full8_full4_v1"} else {"canonical_batch8_v1"},
        "receipt_digest_policy":if request.is_some_and(|r| r.parallel_receipt_digest) {"parallel_exact_v1"} else {"single_cta_v1"},
        "draft_q8_projection_policy":if draft_q8_tile4 {"four_token_weight_reuse_fp64_v1"} else {"independent_tokens_fp64_v1"},
        "draft_context_recompute_audit":validate_draft_context_reuse,
        "draft_context_preparations":draft_context_preparations,
        "request_identity":sha256_hex(&identity),
        "llama_rotated_kv_requested":llama_rotated_kv,
        "small_attention_row_contract":attention_row_contract,
        "llama_bf16_scalars_prefix_requested":llama_bf16_scalars,
        "llama_full_initialized_history_requested":llama_history_full,
        "llama_global_boundary_audit_requested":llama_global_chunks,
        "llama_exact_input_rows_requested":llama_exact_input_rows,
        "prefill_batched_head_rope_requested":prefill_batched_head_rope,
        "submission_batching":{
            "feature_copies_requested":batched_feature_copies,
            "history_feature_copies_requested":history_batched_feature_copies,
            "history_feature_copy_enqueues":submission_batch_counts.3,
            "checkpoint_copies_requested":batched_checkpoint_copies,
            "checkpoint_save_only_requested":batched_checkpoint_save_only,
            "replay_copies_requested":batched_replay_copies,
            "checkpoint_copy_batch_enqueues":state_copy_batch_counts.0,
            "checkpoint_copy_logical_ranges":state_copy_batch_counts.1,
            "checkpoint_copy_logical_bytes":state_copy_batch_counts.3,
            "replay_copy_batch_enqueues":state_copy_batch_counts.2,
            "shared_replay_fold_scalars_requested":shared_replay_fold_scalars,
            "shared_verify_gdn_scalars_requested":shared_verify_gdn_scalars,
            "shared_replay_fold_layers":shared_replay_fold_layers,
            "replay_checkpoint_read_requested":replay_checkpoint_read,
            "replay_prefix_family_requested":replay_prefix_family,
            "checkpoint_read_fold_layers":checkpoint_read_fold_layers,
            "checkpoint_read_avoided_restore_copies":checkpoint_read_fold_layers * 2,
            "checkpoint_read_avoided_restore_bytes":checkpoint_read_fold_layers
                * ((linear::CONV_HISTORY_ELEMENTS + linear::STATE_ELEMENTS) * 4) as u64,
            "checkpoint_read_byte_scope":"logical_restore_payload_only_excludes_checkpoint_save_and_kernel_reads",
            "dflash_history_prepare":dflash_history_prepare,
            "output_head_batching":output_head_batching,
            "replay_copy_logical_bytes":state_copy_batch_counts.2 * (B * (linear::QKV_ELEMENTS + 2 * linear::VALUE_HEADS) * 4) as u64,
            "embedding_gather_requested":batched_embedding_gather,
            "feature_capture_enqueues":submission_batch_counts.0,
            "feature_append_enqueues":submission_batch_counts.1,
            "embedding_gather_enqueues":submission_batch_counts.2,
            "scope":"owner_lifetime_successful_host_enqueues_not_physical_memory_transactions"
        },
        "prefill_head_rope_submission":{
            "policy":if prefill_batched_head_rope {"input_batch_exact_head_reduction_v1"} else {"scalar_token_v1"},
            "batched_launches":prefill_batched_head_rope_launches,
            "batched_layer_rows":prefill_batched_head_rope_layer_rows,
            "decode_verify_changed":false
        },
        "full_attention_kv_basis":if llama_rotated_kv { rotated_kv_identity } else { "qwen35-q8-unrotated-v3-v4" },
        "dflash_resident":dflash_resident,
        "vision_resident":vision_resident,
        "final_cursor":final_cursor,"prefill_ms":prefill_ms,"setup_including_draft_submit_ms":setup_ms,
        "checkpoint_bytes":checkpoint_bytes,"transaction":report,"transactions":transactions,"cleanup":{"pass":clean,"live_bytes":after.live_bytes,
            "allocations":after.successful_allocations-before.successful_allocations,"frees":after.successful_frees-before.successful_frees,"peak_live_bytes":after.peak_live_bytes},
         "claims":{"real_gpu_draft_conditioned_target_verify":true,"physical_recurrent_and_kv_rollback":true,
             "one_block_greedy_emission_and_state_match":!product_mode,"rolling_owner_rebind":rolling_blocks>1,
             "rolling_decode":product_mode,"product_ready":false,"visible_tps":false}});
    // The persistent HTTP service enters this common runner directly; it does
    // not pass through run_request's non-retained owner report wrapper.
    if let Some(target) = shared_target_weights {
        result["resident_target"] = target.resident_report();
    }
    if tokenizer_digest_reuse {
        result["request_tokenizer_digest_reuse"] = json!({
            "policy":"same_immutable_request_bytes_verified_lease_sha256_v1",
            "source_bytes":tokenizer_bytes.len(),"per_request_content_check_retained":true,
            "stable_file_read_retained":true,"tokenizer_sha256":checked_tokenizer_digest,
            "reused_digest_consumers":tokenizer_digest_consumers.get(),
            "avoided_hash_input_bytes":tokenizer_bytes.len() as u64 * u64::from(tokenizer_digest_consumers.get()),
            "cross_request_digest_cache":false,"numeric_or_token_policy_changed":false
        });
    }
    #[cfg(feature = "qwen35-draft-private-state")]
    if let Some(report) = private_report {
        result["draft_private_raw_state"] = report;
    }
    #[cfg(feature = "qwen35-session-continuation")]
    if let Some(checkpoint) = session_checkpoint {
        result["session_checkpoint"] = checkpoint;
        result["cleanup"]["scope"] = json!("request_state_retained_by_unique_session_owner");
        result["cleanup"]["pass"] = json!(false);
        result["session_state_retained"] = json!(true);
    }
    #[cfg(feature = "qwen35-session-continuation")]
    if let Some(capture) = native_reference_private_capture {
        result["native_session_private_capture"] = capture;
    }
    #[cfg(all(
        feature = "qwen35-session-continuation",
        feature = "qwen35-draft-private-state"
    ))]
    if let Some(audit) = native_private_series {
        result["native_session_private_series"] = audit.report();
    }
    #[cfg(feature = "qwen35-session-continuation")]
    if let Some(audit) = native_reference_initial_state {
        result["native_reference_initial_state"] = audit;
        result["native_exact_token_reference"] = json!(true);
        result["native_session_audit_profile"] = json!(native_reference
            .as_ref()
            .ok_or_else(|| invalid("reference profile"))?
            .profile
            .name());
        result["request_output"]["prompt_text_sha256"] = Value::Null;
        result["request_output"]["input_policy"] = json!("exact_u32_history_no_decode_retokenize");
    }
    #[cfg(feature = "qwen35-session-continuation")]
    if let Some(receipt) = native_prefill_scalar_receipt {
        result["native_prefill_scalar_audit"] = receipt;
    }
    if let Some(receipt) = native_prefill_full_canonical_receipt {
        result["native_prefill_full_canonical_audit"] = receipt;
    }
    if let Some(receipt) = native_fixed_history_logits {
        result["native_fixed_history_logits"] = receipt;
        result["performance_evidence"] = json!(false);
    }
    if native_growing_session {
        result["native_growing_session_audit"] = json!({
            "schema":"qwen35-native-growing-session-v1",
            "physical_feature_capacity_rows":2048,
            "initial_active_rows":context_rows,"padding_tokens":0,
            "product_qualified":false
        });
    }
    if let Some(mode) = top16_validation {
        result["top16_validation"] = mode.receipt(draft_top16_batch);
    }
    if let Some(receipt) = trajectory_report {
        result["dflash_live_trajectory"] = receipt;
        result["performance_evidence"] = json!(false);
    }
    if long_generation {
        result["long_generation_budget"] = json!({
            "schema":"qwen35-long-generation-budget-v1", "policy":"long_generation_1024_v1",
            "max_new_tokens_limit":1024, "max_rolling_rounds":1024,
            "round_budget_covers_zero_acceptance":true,
            "numerical_policy_changed":false, "product_qualification":false
        });
    }
    if let Some(history) = terminal_growing_history {
        result["growing_history"] = json!({
            "schema": crate::GROWING_HISTORY_SCHEMA,
            "physical_capacity_rows": history.capacity_rows(),
            "active_rows": history.active_rows(),
            "start_position": history.start_position(),
            "cursor": history.cursor(),
            "generation": history.generation(),
            "default_path_changed": false,
            "gpu_semantics_qualified": false,
            "http_continuation_qualified": false,
            "product_qualified": false
        });
    }
    #[cfg(feature = "qwen35-raw-committed-state")]
    if let Some(raw) = raw_report {
        result["raw_committed_state"] = raw;
    }
    Ok(result)
}

fn raw_capture_enabled() -> bool {
    cfg!(feature = "qwen35-raw-committed-state")
        && std::env::var_os("AUGUR_QWEN35_RAW_STATE_DIR").is_some()
}

fn batched_plan_reuse_report_value(effective_reuse: bool) -> Value {
    json!(effective_reuse)
}

#[cfg(test)]
mod tests {
    #[test]
    fn small_attention_contracts_have_distinct_identity_tags() {
        use super::small_attention_row_contract;
        let tags=[(false,false,false,false),(true,false,false,false),(false,true,false,false),
            (false,false,true,false),(false,false,false,true)]
            .map(|(t,v,s,g)| small_attention_row_contract(t,v,s,g).unwrap());
        for (i,tag) in tags.iter().enumerate() {
            for other in &tags[..i] { assert_ne!(tag,other); }
        }
        for (t,v,s,g) in [(true,true,false,false),(true,false,true,false),(false,true,true,false),
            (true,true,true,false),(true,false,false,true),(false,true,false,true),
            (false,false,true,true)] {
            assert!(small_attention_row_contract(t,v,s,g).is_err());
        }
        assert_eq!(tags[0],"legacy_mixed_vec_tile_v1");
    }
    use super::*;
    #[test]
    fn canonical_reference_prefix_requires_explicit_complete_diagnostic_scope() {
        assert!(!reference_canonical_prefix_switch(None, false, false).unwrap());
        assert!(reference_canonical_prefix_switch(Some("1".as_ref()), true, true).unwrap());
        for (compiled, scope) in [(false, false), (false, true), (true, false)] {
            assert!(
                reference_canonical_prefix_switch(Some("1".as_ref()), compiled, scope).is_err()
            );
        }
        for value in ["", "0", "true", " 1", "1 "] {
            assert!(reference_canonical_prefix_switch(Some(value.as_ref()), true, true).is_err());
        }
    }
    #[test]
    fn replay_head_elision_is_strictly_opt_in() {
        assert!(!replay_head_elision_switch(None).unwrap());
        assert!(replay_head_elision_switch(Some("1".as_ref())).unwrap());
        for value in ["", "0", "true", " 1", "1 "] {
            assert!(replay_head_elision_switch(Some(value.as_ref())).is_err());
        }
    }

    #[test]
    fn fast_feature_identity_binds_request_generation_anchor_and_extent() {
        use std::ffi::OsStr;

        assert!(!fast_feature_identity_switch(None).unwrap());
        assert!(fast_feature_identity_switch(Some(OsStr::new("1"))).unwrap());
        for value in ["", "0", "true", " 1", "1 "] {
            assert!(fast_feature_identity_switch(Some(OsStr::new(value))).is_err());
        }

        let baseline = fast_feature_identity([7; 32], 4, 4096, 123, 2048, 2048, 196608);
        for changed in [
            fast_feature_identity([8; 32], 4, 4096, 123, 2048, 2048, 196608),
            fast_feature_identity([7; 32], 5, 4096, 123, 2048, 2048, 196608),
            fast_feature_identity([7; 32], 4, 4104, 123, 2048, 2048, 196608),
            fast_feature_identity([7; 32], 4, 4096, 124, 2048, 2048, 196608),
            fast_feature_identity([7; 32], 4, 4096, 123, 2056, 2048, 196608),
            fast_feature_identity([7; 32], 4, 4096, 123, 2048, 2047, 196608),
            fast_feature_identity([7; 32], 4, 4096, 123, 2048, 2048, 262144),
        ] {
            assert_ne!(baseline, changed);
        }
        assert_eq!(
            baseline,
            fast_feature_identity([7; 32], 4, 4096, 123, 2048, 2048, 196608)
        );
    }

    #[test]
    fn growing_history_opt_in_rejects_unadapted_consumers() {
        use std::ffi::OsStr;
        assert!(!growing_history_switch(None).unwrap());
        assert!(growing_history_switch(Some(OsStr::new("1"))).unwrap());
        for value in ["", "0", "true", " 1", "1 "] {
            assert!(growing_history_switch(Some(OsStr::new(value))).is_err());
        }
        for bits in 0..16 {
            let product = bits & 1 != 0;
            let bounded = bits & 2 != 0;
            let incremental = bits & 4 != 0;
            let fixed = bits & 8 != 0;
            assert!(
                validate_growing_history_scope(false, product, bounded, incremental, fixed).is_ok()
            );
            assert_eq!(
                validate_growing_history_scope(true, product, bounded, incremental, fixed).is_ok(),
                product && bounded && incremental && !fixed,
            );
        }
    }

    #[test]
    fn growing_feature_capture_rows_match_exact_append_suffix() {
        for context in [8, 17, 2047, 2048] {
            for committed in 1..=B {
                let mut h = GrowingHistory::from_extent(
                    GrowingHistoryConfig::default_2048(),
                    6144,
                    context,
                )
                .unwrap();
                let plan = h.append(committed).unwrap();
                let advance = FeatureAdvance {
                    retained: plan.retained_rows,
                    skip_committed: 0,
                    growing_plan: Some(plan),
                };
                for row in 0..committed {
                    assert_eq!(advance.capture_row(row), Some(plan.append_offset + row));
                }
                assert_eq!(advance.capture_row(committed), None);
                assert_eq!(advance.capture_row(usize::MAX), None);
            }
        }
    }
    #[test]
    fn submission_batch_defaults_and_exact_feature_ranges() {
        use std::ffi::OsStr;
        for eligible in [false, true] {
            assert!(!submission_batch_switch(None, eligible).unwrap());
            assert_eq!(submission_batch_switch(Some(OsStr::new("1")), eligible).is_ok(), eligible);
            for bad in ["", "0", "true", "01", "1 "] {
                assert!(submission_batch_switch(Some(OsStr::new(bad)), eligible).is_err());
            }
        }
        for context in 1..=2048 {
            for committed in 1..=B {
                let mut history = GrowingHistory::from_extent(
                    GrowingHistoryConfig::default_2048(), 4096, context).unwrap();
                let g = history.append(committed).unwrap();
                for plan in [FeatureAdvance::new(context, committed).unwrap(), FeatureAdvance {
                    retained: g.retained_rows, skip_committed: 0, growing_plan: Some(g),
                }] {
                    let old: Vec<_> = (0..committed).filter_map(|r| plan.capture_row(r).map(|d| (r,d))).collect();
                    let new: Vec<_> = plan.capture_range(committed).unwrap().into_iter()
                        .flat_map(|(s,d,n)| (0..n).map(move |i| (s+i,d+i))).collect();
                    assert_eq!(new, old, "context={context}, committed={committed}");
                    assert!(plan.capture_range(0).is_err());
                    assert!(plan.capture_range(B+1).is_err());
                }
            }
        }
    }
    #[test]
    fn resident_vision_selection_is_pinned_before_request_mutation() {
        let a = Some(Path::new("vision-a.safetensors"));
        let b = Some(Path::new("vision-b.safetensors"));
        assert!(validate_resident_vision_selection(a, a, a).is_ok());
        for (request, configured) in [(None, a), (a, None), (b, a), (a, b), (b, b)] {
            assert!(validate_resident_vision_selection(a, request, configured).is_err());
        }
        assert!(validate_resident_vision_selection(None, a, a).is_ok());
        assert!(validate_resident_vision_selection(None, None, None).is_ok());
    }

    #[cfg(feature = "qwen35-session-continuation")]
    #[test]
    fn batched_plan_reuse_report_preserves_effective_suffix_policy() {
        for (suffix_enabled, requested, expected) in [
            (false, false, false),
            (false, true, true),
            (true, false, true),
            (true, true, true),
        ] {
            let effective = committed_session::suffix::retain_plan(suffix_enabled, requested);
            assert_eq!(batched_plan_reuse_report_value(effective), json!(expected));
        }
    }

    pub(super) fn batch_request_options() -> Qwen35DraftTargetRequestOptions {
        Qwen35DraftTargetRequestOptions {
            generation_config_path: PathBuf::new(),
            max_new_tokens: 8,
            long_generation_budget: false,
            retain_full_accept_state: true,
            audit_batched_target_verify: false,
            profile_batched_target_verify: false,
            native_quantized_batch_ffn: false,
            audit_batched_numerics: false,
            canonical_batched_gdn_scalars: false,
            native_quantized_batch_attention: false,
            canonical_batched_norm: false,
            canonical_batched_full_core: false,
            batched_target_verify: true,
            audit_committed_state: false,
            reuse_batched_plan: false,
            share_target_rope: false,
            prefill_shared_softmax: false,
            prefill_dgemm_scores: false,
            batched_prompt_capture: false,
            request_position_budget: false,
            batched_draft_context: false,
            batched_draft_block: false,
            draft_q8_tile4: false,
            parallel_receipt_digest: false,
            exact_projection_tiles: false,
            exact_output_head_batch: false,
            target_only_reference: false,
            audit_reference_logits: false,
            reference_serial_prefix: false,
            reference_boundary_directory: None,
            diagnostic_q8_away: false,
            diagnostic_split_rope: false,
            target_split_rope: false,
            profile_prefill_sparse: false,
        }
    }

    #[test]
    fn diagnostic_q8_build_requires_reference_acknowledgement() {
        let mut r = batch_request_options();
        assert!(validate_diagnostic_q8_policy(Some(&r), false).is_ok());
        assert!(validate_diagnostic_q8_policy(None, true).is_err());
        assert!(validate_diagnostic_q8_policy(Some(&r), true).is_err());
        r.diagnostic_q8_away = true;
        assert!(validate_diagnostic_q8_policy(Some(&r), false).is_err());
        for flags in 0..8 {
            r.target_only_reference = flags & 1 != 0;
            r.audit_reference_logits = flags & 2 != 0;
            r.audit_committed_state = flags & 4 != 0;
            assert_eq!(
                validate_diagnostic_q8_policy(Some(&r), true).is_ok(),
                flags == 7 && !cfg!(feature = "qwen35-prefill-fused-silu-q8")
            );
        }
    }

    #[test]
    fn dgemm_prefill_requires_explicit_shared_product_mode() {
        let mut r = batch_request_options();
        assert!(!r.prefill_dgemm_scores);
        r.prefill_dgemm_scores = true;
        assert!(validate_batch_execution_options(true, &r).is_err());
        r.prefill_shared_softmax = true;
        assert!(validate_batch_execution_options(false, &r).is_err());
        assert_eq!(
            validate_batch_execution_options(true, &r).is_ok(),
            cfg!(feature = "qwen35-kv8-chunk-prefill")
        );
        r.profile_prefill_sparse = true;
        assert!(validate_batch_execution_options(true, &r).is_err());
        r.profile_prefill_sparse = false;
        r.target_only_reference = true;
        assert!(validate_batch_execution_options(true, &r).is_err());
    }

    #[test]
    fn split_rope_build_requires_separate_audited_reference_identity() {
        let mut r = batch_request_options();
        assert!(validate_diagnostic_split_rope_policy(None, true).is_err());
        assert!(validate_diagnostic_split_rope_policy(Some(&r), false).is_ok());
        assert!(validate_diagnostic_split_rope_policy(Some(&r), true).is_err());
        r.diagnostic_split_rope = true;
        assert!(validate_diagnostic_split_rope_policy(Some(&r), false).is_err());
        for flags in 0..8 {
            r.target_only_reference = flags & 1 != 0;
            r.audit_reference_logits = flags & 2 != 0;
            r.audit_committed_state = flags & 4 != 0;
            assert_eq!(
                validate_diagnostic_split_rope_policy(Some(&r), true).is_ok(),
                flags == 7 && !augur_backend::cuda_q8_0::Q8_1_DIAGNOSTIC_TIES_AWAY
            );
        }
        r.diagnostic_q8_away = true;
        assert!(validate_diagnostic_split_rope_policy(Some(&r), true).is_err());
    }

    #[test]
    fn target_rope_contract_rejects_cpu_cuda_or_request_drift() {
        let mut r = batch_request_options();
        assert!(validate_target_split_rope_policy(Some(&r), false, false).is_ok());
        assert!(validate_target_split_rope_policy(None, true, true).is_err());
        assert!(validate_target_split_rope_policy(Some(&r), true, true).is_err());
        r.target_split_rope = true;
        for (gpu, cpu) in [(false, false), (true, false), (false, true)] {
            assert!(validate_target_split_rope_policy(Some(&r), gpu, cpu).is_err());
        }
        assert_eq!(
            validate_target_split_rope_policy(Some(&r), true, true).is_ok(),
            !augur_backend::cuda_q8_0::QWEN35_DIAGNOSTIC_SPLIT_ROPE
                && !augur_backend::cuda_q8_0::Q8_1_DIAGNOSTIC_TIES_AWAY
        );
        r.diagnostic_split_rope = true;
        assert!(validate_target_split_rope_policy(Some(&r), true, true).is_err());
    }

    #[test]
    fn parallel_receipt_is_opt_in_and_independent_of_draft_or_reference_policies() {
        let mut request = batch_request_options();
        assert!(!request.parallel_receipt_digest);
        request.parallel_receipt_digest = true;
        assert!(validate_batch_execution_options(true, &request).is_ok());
        request.target_only_reference = true;
        request.audit_committed_state = true;
        assert!(validate_batch_execution_options(true, &request).is_ok());
    }
    #[test]
    fn exact_projection_tiles_are_opt_in_and_require_actual_batch_execution() {
        let mut request = batch_request_options();
        assert!(!request.exact_projection_tiles);
        request.exact_projection_tiles = true;
        assert!(validate_batch_execution_options(true, &request).is_ok());
        assert!(validate_batch_execution_options(false, &request).is_err());
        request.batched_target_verify = false;
        assert!(validate_batch_execution_options(true, &request).is_err());
        request.batched_target_verify = true;
        request.target_only_reference = true;
        assert!(validate_batch_execution_options(true, &request).is_err());
    }

    #[test]
    fn exact_output_head_is_independent_opt_in_and_needs_batch_execution() {
        let mut request = batch_request_options();
        assert!(!request.exact_output_head_batch);
        request.exact_output_head_batch = true;
        assert!(!request.exact_projection_tiles);
        assert!(validate_batch_execution_options(true, &request).is_ok());
        assert!(validate_batch_execution_options(false, &request).is_err());
        request.batched_target_verify = false;
        assert!(validate_batch_execution_options(true, &request).is_err());
        request.batched_target_verify = true;
        request.target_only_reference = true;
        assert!(validate_batch_execution_options(true, &request).is_err());
    }

    #[test]
    fn q8_tile4_requires_both_launch_policies_and_is_opt_in() {
        assert!(!batch_request_options().draft_q8_tile4);
        for tile4 in [false, true] {
            for context in [false, true] {
                for block in [false, true] {
                    let mut request = batch_request_options();
                    request.draft_q8_tile4 = tile4;
                    request.batched_draft_context = context;
                    request.batched_draft_block = block;
                    assert_eq!(
                        validate_batch_execution_options(true, &request).is_ok(),
                        !tile4 || (context && block)
                    );
                }
            }
        }
    }

    #[test]
    fn target_only_reference_requires_state_audit_and_rejects_draft_controls() {
        let mut request = batch_request_options();
        request.target_only_reference = true;
        assert!(validate_batch_execution_options(true, &request).is_err());
        request.audit_committed_state = true;
        assert!(validate_batch_execution_options(true, &request).is_ok());
        assert!(validate_batch_execution_options(false, &request).is_err());
        request.batched_draft_block = true;
        assert!(validate_batch_execution_options(true, &request).is_err());
        request.batched_draft_block = false;
        request.batched_draft_context = true;
        assert!(validate_batch_execution_options(true, &request).is_err());
        request.batched_draft_context = false;
        request.reuse_batched_plan = true;
        assert!(validate_batch_execution_options(true, &request).is_err());
    }

    #[test]
    fn reference_logits_audit_is_opt_in_and_requires_no_draft_state_reference() {
        let mut request = batch_request_options();
        assert!(!request.audit_reference_logits);
        request.audit_reference_logits = true;
        assert!(validate_batch_execution_options(true, &request).is_err());
        request.target_only_reference = true;
        assert!(validate_batch_execution_options(true, &request).is_err());
        request.audit_committed_state = true;
        assert!(validate_batch_execution_options(true, &request).is_ok());
        request.batched_draft_block = true;
        assert!(validate_batch_execution_options(true, &request).is_err());
    }

    #[test]
    fn boundary_trace_requires_explicit_audited_batched_target_reference() {
        let mut request = batch_request_options();
        assert!(request.reference_boundary_directory.is_none());
        request.reference_boundary_directory = Some(PathBuf::from("fresh-boundary-diagnostic"));
        request.target_only_reference = true;
        request.audit_reference_logits = true;
        request.audit_committed_state = true;
        request.batched_prompt_capture = true;
        assert!(validate_batch_execution_options(true, &request).is_ok());
        assert!(validate_batch_execution_options(false, &request).is_err());
        for axis in 0..6 {
            let mut bad = request.clone();
            match axis {
                0 => bad.target_only_reference = false,
                1 => bad.audit_reference_logits = false,
                2 => bad.audit_committed_state = false,
                3 => bad.batched_prompt_capture = false,
                4 => bad.batched_target_verify = false,
                _ => bad.retain_full_accept_state = false,
            }
            assert!(
                validate_batch_execution_options(true, &bad).is_err(),
                "axis {axis}"
            );
        }
    }

    #[test]
    fn sequential_prefix_is_opt_in_reference_only_and_rejects_batched_tuning() {
        let mut request = batch_request_options();
        assert!(!request.reference_serial_prefix);
        request.reference_serial_prefix = true;
        assert!(validate_batch_execution_options(true, &request).is_err());
        request.target_only_reference = true;
        request.audit_committed_state = true;
        assert!(validate_batch_execution_options(true, &request).is_ok());
        request.prefill_shared_softmax = true;
        assert!(validate_batch_execution_options(true, &request).is_err());
    }

    #[test]
    fn request_position_budget_preserves_legacy_and_admits_near_capacity() {
        assert!(rolling_position_budget(196352, 64, Some(128), 196608, false).is_err());
        let bounded = rolling_position_budget(196352, 64, Some(128), 196608, true).unwrap();
        assert_eq!(bounded.policy, "max_new_tokens_full_verify_v1");
        assert_eq!(bounded.reserved_positions, 134);
        assert_eq!(bounded.end, 196486);
        assert_eq!(
            rolling_position_budget(196352, 64, Some(250), 196608, true)
                .unwrap()
                .end,
            196608
        );
        assert!(rolling_position_budget(196352, 64, Some(251), 196608, true).is_err());
        let legacy = rolling_position_budget(123, 32, Some(10), 196608, false).unwrap();
        assert_eq!(legacy.policy, "rolling_blocks_x8_v1");
        assert_eq!(legacy.end, 123 + 32 * B);
        assert_eq!(
            rolling_position_budget(123, 32, None, 196608, false)
                .unwrap()
                .end,
            legacy.end
        );
    }

    #[test]
    fn long_generation_budget_preserves_legacy_caps_and_all_verify_rows() {
        assert!(rolling_position_budget(194560, 1024, Some(1024), 196608, true).is_err());
        let value =
            rolling_position_budget_with_profile(194560, 1024, Some(1024), 196608, true, true)
                .unwrap();
        assert_eq!(value.reserved_positions, 1030);
        assert_eq!(value.end, 195590);
        assert_eq!(196608 - value.end, 1018);
        for (cursor, rounds, limit, capacity, bounded) in [
            (196352, 1024, Some(1024), 196608, true),
            (195584, 1024, Some(1024), 196608, true),
            (194560, 128, Some(1024), 196608, true),
            (194560, 1025, Some(1024), 196608, true),
            (194560, 1024, Some(1025), 196608, true),
            (194560, 1024, None, 196608, true),
            (194560, 1024, Some(1024), 262144, true),
            (194560, 1024, Some(1024), 196608, false),
        ] {
            assert!(rolling_position_budget_with_profile(
                cursor, rounds, limit, capacity, bounded, true
            )
            .is_err());
        }
        assert_eq!(
            rolling_position_budget_with_profile(195578, 1024, Some(1024), 196608, true, true)
                .unwrap()
                .end,
            196608
        );
        assert_eq!(
            rolling_position_budget_with_profile(196600, 1, Some(1), 196608, true, true)
                .unwrap()
                .reserved_positions,
            B
        );
    }

    #[test]
    fn long_generation_budget_covers_every_reachable_nonterminal_cursor() {
        for limit in 1..=1024 {
            let budget = rolling_position_budget_with_profile(
                194560,
                limit,
                Some(limit),
                196608,
                true,
                true,
            )
            .unwrap();
            for completed_rounds in 0..limit {
                let least_published = 1 + completed_rounds;
                let most_published = (1 + B * completed_rounds).min(limit - 1);
                if least_published <= most_published {
                    assert!(194560 + most_published - 1 + B <= budget.end);
                }
            }
        }
    }

    #[test]
    fn growing_budget_admits_actual_initialized_cursors_without_changing_fixed_profile() {
        for cursor in [1, 7, 31, 2047, 2048, 2049, 16384] {
            let budget = rolling_position_budget_with_history(
                cursor,
                1024,
                Some(1024),
                196608,
                true,
                true,
                RollingHistoryBudget::GrowingInitialized {
                    active_rows: cursor.min(2048),
                },
            )
            .unwrap();
            assert_eq!(
                budget.policy,
                "growing_initialized_max_new_tokens_full_verify_v1"
            );
            assert_eq!(budget.reserved_positions, 1030);
            assert_eq!(budget.end, cursor + 1030);
            assert_eq!(
                rolling_position_budget_with_profile(cursor, 1024, Some(1024), 196608, true, true,)
                    .is_ok(),
                cursor >= 2048
            );
        }
    }

    #[test]
    fn growing_budget_requires_real_initialized_window_and_frozen_request_bounds() {
        for (cursor, active_rows) in [
            (0, 0),
            (1, 0),
            (7, 2048),
            (31, 32),
            (2047, 2048),
            (2048, 2047),
            (2049, 2049),
        ] {
            assert!(rolling_position_budget_with_history(
                cursor,
                1024,
                Some(1024),
                196608,
                true,
                true,
                RollingHistoryBudget::GrowingInitialized { active_rows },
            )
            .is_err());
        }
        for (rounds, limit, capacity, bounded, long_generation) in [
            (1023, Some(1024), 196608, true, true),
            (0, Some(1), 196608, true, true),
            (1025, Some(1024), 196608, true, true),
            (1024, Some(0), 196608, true, true),
            (1024, Some(1025), 196608, true, true),
            (1024, None, 196608, true, true),
            (1024, Some(1024), 262144, true, true),
            (1024, Some(1024), 196608, false, true),
            (32, Some(32), 196608, true, false),
        ] {
            assert!(rolling_position_budget_with_history(
                31,
                rounds,
                limit,
                capacity,
                bounded,
                long_generation,
                RollingHistoryBudget::GrowingInitialized { active_rows: 31 },
            )
            .is_err());
        }
        for cursor in [195579, 196608, usize::MAX] {
            assert!(rolling_position_budget_with_history(
                cursor,
                1024,
                Some(1024),
                196608,
                true,
                true,
                RollingHistoryBudget::GrowingInitialized { active_rows: 2048 },
            )
            .is_err());
        }
        assert_eq!(
            rolling_position_budget_with_history(
                195578,
                1024,
                Some(1024),
                196608,
                true,
                true,
                RollingHistoryBudget::GrowingInitialized { active_rows: 2048 },
            )
            .unwrap()
            .end,
            196608
        );
    }

    #[test]
    fn growing_budget_covers_zero_acceptance_and_all_full_verification_rows() {
        for cursor in [1, 7, 31, 2047, 2048, 2049] {
            for limit in 1..=1024 {
                let budget = rolling_position_budget_with_history(
                    cursor,
                    limit,
                    Some(limit),
                    196608,
                    true,
                    true,
                    RollingHistoryBudget::GrowingInitialized {
                        active_rows: cursor.min(2048),
                    },
                )
                .unwrap();
                assert!(cursor + B <= budget.end);
                for completed_rounds in 0..limit {
                    let least_published = 1 + completed_rounds;
                    let most_published = (1 + B * completed_rounds).min(limit - 1);
                    if least_published <= most_published {
                        assert!(cursor + most_published - 1 + B <= budget.end);
                    }
                }
            }
        }
    }

    #[test]
    #[cfg(feature = "qwen35-kv8-chunk-prefill")]
    fn long_generation_requires_the_frozen_canonical_request_profile() {
        let mut request = batch_request_options();
        request.long_generation_budget = true;
        request.max_new_tokens = 1024;
        assert!(validate_batch_execution_options(true, &request).is_err());
        request.batched_prompt_capture = true;
        request.request_position_budget = true;
        request.share_target_rope = true;
        request.target_split_rope = true;
        request.prefill_shared_softmax = true;
        request.prefill_dgemm_scores = true;
        assert!(validate_batch_execution_options(true, &request).is_ok());
        assert!(validate_batch_execution_options(false, &request).is_err());
        request.request_position_budget = false;
        assert!(validate_batch_execution_options(true, &request).is_err());
        request.request_position_budget = true;
        request.target_only_reference = true;
        assert!(validate_batch_execution_options(true, &request).is_err());
    }

    #[test]
    fn request_position_budget_covers_every_reachable_nonterminal_cursor() {
        // After r rounds, 1+r..=1+8*r tokens can have been published.
        // Intersect with the nonterminal limit. This covers every acceptance
        // path, including zero acceptance, without relying on model behavior.
        for limit in 1..=512 {
            for rounds in 1..=64 {
                let bound =
                    rolling_position_budget(196096, rounds, Some(limit), 196608, true).unwrap();
                assert!(bound.reserved_positions >= B);
                for r in 0..rounds {
                    let least_published = 1 + r;
                    let most_published = (1 + B * r).min(limit - 1);
                    if least_published <= most_published {
                        let largest_cursor = 196096 + most_published - 1;
                        assert!(largest_cursor + B <= bound.end);
                        assert!(largest_cursor + B <= 196608);
                    }
                }
            }
        }
    }

    #[test]
    fn request_position_budget_keeps_initial_stop_workspace_and_round_limit() {
        for limit in 1..=2 {
            assert_eq!(
                rolling_position_budget(50, 64, Some(limit), 58, true)
                    .unwrap()
                    .end,
                58
            );
            assert!(rolling_position_budget(50, 64, Some(limit), 57, true).is_err());
        }
        assert_eq!(
            rolling_position_budget(50, 2, Some(128), 66, true)
                .unwrap()
                .reserved_positions,
            16
        );
        assert!(rolling_position_budget(0, 1, None, 262144, true).is_err());
        for limit in [0, 513, usize::MAX] {
            assert!(rolling_position_budget(0, 1, Some(limit), 262144, true).is_err());
        }
        for rounds in [0, 65, usize::MAX] {
            assert!(rolling_position_budget(0, rounds, Some(128), 262144, true).is_err());
        }
        assert!(rolling_position_budget(usize::MAX, 1, Some(128), usize::MAX, true).is_err());
    }

    #[test]
    fn sparse_prefill_profile_requires_explicit_state_audit_and_unfused_path() {
        let mut request = batch_request_options();
        request.profile_prefill_sparse = true;
        assert!(validate_batch_execution_options(true, &request).is_err());
        request.audit_committed_state = true;
        assert_eq!(
            validate_batch_execution_options(true, &request).is_ok(),
            !cfg!(feature = "qwen35-prefill-fused-silu-q8")
        );
        assert!(validate_batch_execution_options(false, &request).is_err());
    }

    #[test]
    fn batched_prompt_capture_requires_executing_core_and_feature_staging() {
        let mut request = batch_request_options();
        request.batched_prompt_capture = true;
        assert!(validate_batch_execution_options(true, &request).is_ok());
        request.retain_full_accept_state = false;
        assert!(validate_batch_execution_options(true, &request).is_err());
        request.retain_full_accept_state = true;
        request.batched_target_verify = false;
        assert!(validate_batch_execution_options(true, &request).is_err());
    }

    #[test]
    fn actual_batch_execution_excludes_every_shadow_option() {
        assert!(validate_batch_execution_options(true, &batch_request_options()).is_ok());
        assert!(validate_batch_execution_options(false, &batch_request_options()).is_err());
        let mutations: [fn(&mut Qwen35DraftTargetRequestOptions); 8] = [
            |r| r.audit_batched_target_verify = true,
            |r| r.profile_batched_target_verify = true,
            |r| r.native_quantized_batch_ffn = true,
            |r| r.audit_batched_numerics = true,
            |r| r.canonical_batched_gdn_scalars = true,
            |r| r.native_quantized_batch_attention = true,
            |r| r.canonical_batched_norm = true,
            |r| r.canonical_batched_full_core = true,
        ];
        for mutation in mutations {
            let mut request = batch_request_options();
            mutation(&mut request);
            assert!(validate_batch_execution_options(true, &request).is_err());
        }
    }

    #[test]
    fn canonical_target_profile_requires_state_audit_and_preserves_shadow_exclusion() {
        let mut request = batch_request_options();
        request.profile_batched_target_verify = true;
        assert!(validate_batch_execution_options(true, &request).is_err());
        request.audit_committed_state = true;
        assert!(validate_batch_execution_options(true, &request).is_ok());
        assert!(validate_batch_execution_options(false, &request).is_err());
        let mutations: [fn(&mut Qwen35DraftTargetRequestOptions); 7] = [
            |r| r.audit_batched_target_verify = true,
            |r| r.audit_batched_numerics = true,
            |r| r.native_quantized_batch_ffn = true,
            |r| r.canonical_batched_gdn_scalars = true,
            |r| r.native_quantized_batch_attention = true,
            |r| r.canonical_batched_norm = true,
            |r| r.canonical_batched_full_core = true,
        ];
        for mutation in mutations {
            let mut invalid_request = request.clone();
            mutation(&mut invalid_request);
            assert!(validate_batch_execution_options(true, &invalid_request).is_err());
        }
    }

    #[test]
    fn shared_prefill_cannot_claim_a_zero_prefix_fast_path() {
        assert!(validate_shared_prefill_extent(false, 0).is_ok());
        assert!(validate_shared_prefill_extent(false, 30720).is_ok());
        assert!(validate_shared_prefill_extent(true, 1).is_ok());
        assert!(validate_shared_prefill_extent(true, 30720).is_ok());
        assert!(validate_shared_prefill_extent(true, 0).is_err());
    }

    #[test]
    fn plan_reuse_cannot_attach_to_serial_or_shadow_verification() {
        let mut request = batch_request_options();
        request.reuse_batched_plan = true;
        assert!(validate_batch_execution_options(true, &request).is_ok());
        request.batched_target_verify = false;
        assert!(validate_batch_execution_options(true, &request).is_err());
        request.audit_batched_target_verify = true;
        assert!(validate_batch_execution_options(true, &request).is_err());
    }

    #[test]
    fn committed_state_audit_allows_paired_serial_and_batch_but_not_full_audit_mode() {
        for batch in [false, true] {
            let mut request = batch_request_options();
            request.batched_target_verify = batch;
            request.audit_committed_state = true;
            assert!(validate_batch_execution_options(true, &request).is_ok());
            assert!(validate_batch_execution_options(false, &request).is_err());
        }
    }

    #[test]
    fn verified_state_promotion_requires_opt_in_full_accept_and_no_stop() {
        for enabled in [false, true] {
            for accepted in 0..B {
                for committed in 1..=accepted + 1 {
                    for stop in [None, Some("eos"), Some("max_new_tokens")] {
                        assert_eq!(
                            promote_verified_block(enabled, accepted, committed, stop),
                            enabled && accepted == 7 && committed == 8 && stop.is_none()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn verified_feature_promotion_keeps_only_the_committed_window_tail() {
        for context in [1, 3, 8, 45, 2048] {
            let before = (100..100 + context).collect::<Vec<_>>();
            let verified = (100 + context..100 + context + B).collect::<Vec<_>>();
            let plan = FeatureAdvance::new(context, B).unwrap();
            let mut after = before.iter().skip(B).copied().collect::<Vec<_>>();
            after.resize(context, usize::MAX);
            for (row, &value) in verified.iter().enumerate() {
                if let Some(destination) = plan.capture_row(row) {
                    after[destination] = value;
                }
            }
            assert_eq!(after, (108..108 + context).collect::<Vec<_>>());
        }
    }
    #[test]
    fn request_capture_consumes_prompt_tail_without_generated_warmup_or_padding() {
        for len in [1, 7, 8, 19, 2048, 4097, 196000] {
            for requested in [1, 8, 2048] {
                let (start, rows) = prompt_capture_extent(len, requested).unwrap();
                assert_eq!(start + rows, len);
                assert_eq!(rows, requested.min(len));
                // Prefill's sentinel is consumed exactly once by captured tail.
                assert_eq!(
                    (0..start).chain(start..start + rows).collect::<Vec<_>>(),
                    (0..len).collect::<Vec<_>>()
                );
            }
        }
        for (len, rows) in [(0, 8), (8, 0), (8, 2049)] {
            assert!(prompt_capture_extent(len, rows).is_err());
        }
    }

    #[test]
    fn request_publication_stops_at_first_eos_and_commits_only_retained_inputs() {
        for eos_index in 0..B {
            let mut tokens = (1..=B as u32).collect::<Vec<_>>();
            tokens[eos_index] = 248046;
            let stop = clip_publication(
                &mut tokens,
                Some(PublicationBudget {
                    remaining: B,
                    eos: &[248046, 248044],
                }),
            )
            .unwrap();
            assert_eq!(stop, Some("eos"));
            assert_eq!(tokens.len(), eos_index + 1);
            assert_eq!(tokens.last(), Some(&248046));
            // The EOS itself is the next, unconsumed anchor, not a replay input.
            let replay_input_rows = 0..tokens.len();
            assert_eq!(replay_input_rows.last(), Some(eos_index));
        }
    }

    #[test]
    fn request_length_boundary_wins_before_later_eos_and_first_anchor_is_counted() {
        let mut tokens = vec![10, 11, 248046, 12];
        assert_eq!(
            clip_publication(
                &mut tokens,
                Some(PublicationBudget {
                    remaining: 2,
                    eos: &[248046]
                })
            )
            .unwrap(),
            Some("max_new_tokens")
        );
        assert_eq!(tokens, vec![10, 11]);
        let mut first = vec![10];
        assert_eq!(
            clip_publication(
                &mut first,
                Some(PublicationBudget {
                    remaining: 1,
                    eos: &[248046]
                })
            )
            .unwrap(),
            Some("max_new_tokens")
        );
        let mut terminal = vec![248046];
        assert_eq!(
            clip_publication(
                &mut terminal,
                Some(PublicationBudget {
                    remaining: 1,
                    eos: &[248046]
                })
            )
            .unwrap(),
            Some("eos")
        );
        assert!(clip_publication(
            &mut first,
            Some(PublicationBudget {
                remaining: 0,
                eos: &[248046]
            })
        )
        .is_err());
        assert!(clip_publication(&mut Vec::new(), None).is_err());
    }

    #[test]
    fn legacy_publication_remains_unclipped_without_request_contract() {
        let mut tokens = vec![1, 248046, 2];
        assert_eq!(clip_publication(&mut tokens, None).unwrap(), None);
        assert_eq!(tokens, vec![1, 248046, 2]);
    }
    #[test]
    fn committed_feature_window_keeps_only_chronological_committed_rows() {
        for context in [1, 7, 8, 9, 17, 2048] {
            let mut window = (0..context).collect::<Vec<_>>();
            let mut cursor = context;
            for committed in (1..=8).cycle().take(64) {
                let plan = FeatureAdvance::new(context, committed).unwrap();
                for destination in (0..plan.retained).step_by(B) {
                    let rows = B.min(plan.retained - destination);
                    let scratch =
                        window[destination + committed..destination + committed + rows].to_vec();
                    window[destination..destination + rows].copy_from_slice(&scratch);
                }
                for row in 0..committed {
                    if let Some(destination) = plan.capture_row(row) {
                        window[destination] = cursor + row;
                    }
                }
                cursor += committed;
                assert_eq!(window, (cursor - context..cursor).collect::<Vec<_>>());
            }
        }
    }

    #[test]
    fn feature_window_rejects_empty_or_unbounded_advance() {
        for (context, committed) in [(0, 1), (2049, 1), (8, 0), (8, 9), (usize::MAX, 1)] {
            assert!(FeatureAdvance::new(context, committed).is_err());
        }
    }

    #[test]
    fn verified_commit_contract_accepts_only_single_step_committed_prefixes() {
        let valid = Qwen35VerifiedCommit {
            initial_anchor: false,
            block_index: 3,
            generation_before: 9,
            generation_after: 10,
            cursor_before: 128,
            cursor_after: 131,
            token_ids: vec![10, 11, 12],
            stop_reason: None,
        };
        assert!(valid.validate_contract().is_ok());

        for (generation_after, cursor_after, token_ids, stop_reason) in [
            (11, 131, vec![10, 11, 12], None),
            (10, 132, vec![10, 11, 12], None),
            (10, 131, vec![10, 11, 248_320], None),
            (10, 131, vec![10, 11, 12], Some("unknown")),
            (10, 131, Vec::new(), None),
        ] {
            let invalid_commit = Qwen35VerifiedCommit {
                initial_anchor: false,
                block_index: 3,
                generation_before: 9,
                generation_after,
                cursor_before: 128,
                cursor_after,
                token_ids,
                stop_reason: stop_reason.map(str::to_owned),
            };
            assert!(invalid_commit.validate_contract().is_err());
        }
    }

    #[test]
    fn verified_commit_contract_accepts_initial_anchor_without_cursor_advance() {
        let valid = Qwen35VerifiedCommit {
            initial_anchor: true,
            block_index: 0,
            generation_before: 0,
            generation_after: 0,
            cursor_before: 13,
            cursor_after: 13,
            token_ids: vec![271],
            stop_reason: None,
        };
        assert!(valid.validate_contract().is_ok());

        for (block_index, generation_after, cursor_after, token_ids, stop_reason) in [
            (1, 0, 13, vec![271], None),
            (0, 1, 13, vec![271], None),
            (0, 0, 14, vec![271], None),
            (0, 0, 13, vec![271, 272], None),
            (0, 0, 13, vec![271], Some("unknown")),
        ] {
            let invalid_anchor = Qwen35VerifiedCommit {
                initial_anchor: true,
                block_index,
                generation_before: 0,
                generation_after,
                cursor_before: 13,
                cursor_after,
                token_ids,
                stop_reason: stop_reason.map(str::to_owned),
            };
            assert!(invalid_anchor.validate_contract().is_err());
        }
    }

    #[test]
    fn verified_initial_anchor_can_exhaust_one_token_budget_without_cursor_advance() {
        for (anchor, limit, expected_stop) in [
            (271, 1, Some("max_new_tokens")),
            (271, 2, None),
            (248_046, 1, Some("eos")),
        ] {
            let mut tokens = vec![anchor];
            let stop = clip_publication(
                &mut tokens,
                Some(PublicationBudget {
                    remaining: limit,
                    eos: &[248_046],
                }),
            )
            .unwrap();
            assert_eq!(stop, expected_stop);
            let commit = Qwen35VerifiedCommit {
                initial_anchor: true,
                block_index: 0,
                generation_before: 7,
                generation_after: 7,
                cursor_before: 2600,
                cursor_after: 2600,
                token_ids: tokens,
                stop_reason: stop.map(str::to_owned),
            };
            let received = std::cell::RefCell::new(Vec::new());
            dispatch_verified_commit(
                &|value| {
                    received.borrow_mut().push(value.clone());
                    Ok(())
                },
                commit.clone(),
            )
            .unwrap();
            assert_eq!(*received.borrow(), [commit]);
        }
    }

    #[test]
    fn verified_commit_sink_error_stops_dispatch_without_mutating_commit() {
        let commit = Qwen35VerifiedCommit {
            initial_anchor: false,
            block_index: 0,
            generation_before: 0,
            generation_after: 1,
            cursor_before: 0,
            cursor_after: 1,
            token_ids: vec![42],
            stop_reason: Some("max_new_tokens".to_owned()),
        };
        let seen = std::cell::Cell::new(0usize);
        let sink: &Qwen35VerifiedCommitSink<'_> = &|received| {
            seen.set(seen.get() + received.token_ids.len());
            Err(invalid("consumer cancelled"))
        };
        assert!(dispatch_verified_commit(sink, commit).is_err());
        assert_eq!(seen.get(), 1);
    }

    #[test]
    fn snapshot_is_full_recurrent_but_only_speculative_kv_tail() {
        assert_eq!(state_range(false, 1234, 100, 4096).unwrap(), (0, 1234));
        assert_eq!(
            state_range(true, 4096 * QWEN35_KV8_CACHE_ROW_BYTES, 4088, 4096).unwrap(),
            (
                4088 * QWEN35_KV8_CACHE_ROW_BYTES,
                8 * QWEN35_KV8_CACHE_ROW_BYTES
            )
        );
        for (total, cursor, cap) in [
            (0, 0, 0),
            (4096 * QWEN35_KV8_CACHE_ROW_BYTES, 4089, 4096),
            (4096 * 1024, 0, 4096),
            (4096 * QWEN35_KV8_CACHE_ROW_BYTES, usize::MAX, 4096),
        ] {
            assert!(state_range(true, total, cursor, cap).is_err());
        }
    }
}
