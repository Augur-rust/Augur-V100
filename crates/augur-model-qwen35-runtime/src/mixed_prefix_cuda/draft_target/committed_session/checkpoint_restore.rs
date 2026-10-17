//! Opt-in fresh native destination. No prefill, prior CUDA owner, or saved
//! executable is consumed. HTTP integration and numerical qualification are
//! separate gates; every result retains a closed product claim ceiling.

use super::*;
use crate::session_checkpoint::filesystem::LocalCheckpointReader;
use crate::session_checkpoint::metadata::{
    FeatureExtent, HistoryMode, ModelBindings, NativeMathPolicy, NativeRestoreAdmission,
    SessionDescriptor, CAPACITY, HISTORY,
};
use crate::session_checkpoint::{self as transfer, RestoreTarget, TransferPlan};
use std::io;

fn io_error(error: impl std::fmt::Display) -> io::Error {
    io::Error::other(error.to_string())
}

/// Typed metadata is validated by the caller; this construction view carries
/// no generated ledger for an input prefix. None means an inert token slot,
/// not a restored/publishable prediction.
pub(super) struct FreshTarget<'a> {
    pub consumed: &'a [u32],
    pub features: &'a FeatureExtent,
    pub math: &'a NativeMathPolicy,
    pub bindings: &'a ModelBindings,
    pub pending: Option<u32>,
    pub generation: u64,
    pub metadata: &'a [u8],
}

pub(super) fn fresh_driver(
    weights: &target_weights::TargetWeights,
    model: &Qwen35Config,
    saved: FreshTarget<'_>,
    new_session_id: &str,
) -> Result<Driver> {
    let cursor = saved.consumed.len();
    let math = saved.math;
    // Format2 is currently admitted only for live-owner prefill/continuation.
    // Never reinterpret saved planar bytes through the legacy restore reader.
    if math.full_attention_kv_basis.is_planar()
        || std::env::var_os("AUGUR_QWEN35_ROTATED_KV_PLANAR").is_some() {
        return Err(invalid("planar KV persistent restore is not admitted"));
    }
    if weights.qpn_ffn_enabled() {
        return Err(invalid("persistent checkpoint QPN numerical profile not yet admitted"));
    }
    let stream = Rc::new(OwnedCudaStream::create(0)?);
    let first = weights.embedding_row(saved.consumed[0] as usize)?;
    let shared_rope = if math.share_target_rope {
        Some(Arc::new(FullRopeTables::new(
            &stream,
            model.rope_theta as f32,
            CAPACITY,
            full::ROTARY_DIM,
        )?))
    } else {
        None
    };
    let mut plans = Vec::<DevicePlan>::with_capacity(64);
    for weight in &weights.layers {
        let source = plans
            .last()
            .map(DevicePlan::final_hidden_f32_view)
            .transpose()?;
        plans.push(match weight {
            target_weights::LayerWeights::Linear(weight) => DevicePlan::Linear(bind_linear_plan(
                &stream, weight, &first, None, true, source, None,
            )?),
            target_weights::LayerWeights::Full(weight) => DevicePlan::Full(bind_full_plan(
                &stream,
                weight,
                model,
                CAPACITY,
                CAPACITY,
                source,
                shared_rope.clone(),
                None,
            )?),
        });
    }
    drop(shared_rope);
    if math.full_attention_kv_basis.is_rotated() {
        let arena = RotatedFullAttention::new(&stream, CAPACITY, false)?;
        for plan in &mut plans {
            if let DevicePlan::Full(plan) = plan {
                plan.rotated_attention = Some(arena.clone());
            }
        }
    }
    let head = bind_device_output_head_plan(
        &stream,
        &weights.head,
        plans
            .last()
            .ok_or_else(|| invalid("restore requires target plans"))?
            .final_hidden_view()?,
        model.rms_norm_eps as f32,
    )?;
    let alloc = |n| OwnedCudaBuffer::allocate(0, n).map_err(Qwen35MixedPrefixCudaError::from);
    let mut backups = Vec::with_capacity(128);
    for plan in &plans {
        for buffer in plan.terminal_state_buffers() {
            backups.push(alloc(
                state_range(
                    matches!(plan, DevicePlan::Full(_)),
                    buffer.len_bytes(),
                    cursor,
                    CAPACITY,
                )?
                .1,
            )?);
        }
    }
    let growing_history = match saved.features.mode {
        HistoryMode::Fixed2048 => None,
        HistoryMode::Growing2048 { generation } => Some(
            GrowingHistory::from_committed_extent(
                GrowingHistoryConfig::default_2048(),
                saved.features.start,
                saved.features.rows,
                generation,
            )
            .map_err(|e| invalid(e.to_string()))?,
        ),
    };
    // Retain the old digest as provenance in the descriptor only. The live
    // request identity binds the explicit new handle and the admitted snapshot.
    let mut hash = Sha256::new();
    hash.update(b"QWEN35-NATIVE-SSD-FRESH-REQUEST-V1\0");
    hash.update(saved.metadata);
    hash.update((new_session_id.len() as u64).to_le_bytes());
    hash.update(new_session_id.as_bytes());
    let identity = hash.finalize().into();
    let mut d = Driver {
        trajectory_capture: false,
        session_cancellation: None,
        stream,
        plans,
        head,
        embedding: weights.embedding.clone(),
        noise_ids: alloc(B * 4)?,
        noise_device: alloc(B * H * 4)?,
        token: alloc(4)?,
        features: alloc(HISTORY * 5 * H * 4)?,
        feature_shift_scratch: alloc(B * 5 * H * 4)?,
        feature_shift_submissions: 0,
        verified_features: if math.retain_full_accept_state {
            Some(alloc(B * 5 * H * 4)?)
        } else {
            None
        },
        batch_verify_audit: None,
        batch_verify_execution: None,
        round_profile: None,
        batched_suffix: math.batched_suffix,
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
        cursor,
        capacity: CAPACITY,
        context_rows: saved.features.rows,
        growing_history,
        last_growth_plan: None,
        anchor: saved.pending.unwrap_or(0),
        generation: saved.generation,
        identity,
        checkpoint_model_config_sha256: saved.bindings.model_config_sha256.clone(),
        checkpoint_tokenizer_sha256: saved.bindings.tokenizer_sha256.clone(),
        checkpoint_generation_config_sha256: Some(saved.bindings.generation_config_sha256.clone()),
        checkpoint_reuse_batched_plan: math.reuse_batched_plan,
        checkpoint_share_target_rope: math.share_target_rope,
        feature_sha: [0; 32],
        fast_feature_identity: false,
        replay_head_elision: false,
        parallel_receipt_digest: math.parallel_receipt_digest,
        exact_projection_tiles: math.exact_projection_tiles,
        // Schedule is a fresh launch choice, never checkpoint CUDA authority.
        projection_row_reuse: false,
        ffn_down_row_reuse: false,
        preserved_silu_q8: false,
        // Restore resumes committed state; it never reruns cold prefill.
        canonical_prefill_rows: B,
        tensor_ffn_down_rows: None,
        tensor_ffn_down_compensated: false,
        integer_mma_ffn: None,
        integer_mma_input: false,
        llama_f16_input: false,
        llama_original_input: false,
        llama_full_original: false,
        llama_bf16_scalars: false,
        llama_history_full: false,
        llama_global_chunks: false,
        llama_exact_input_rows: false,
        prefill_batched_head_rope: false,
        batched_feature_copies: false,
        history_batched_feature_copies: false,
        history_feature_copy_enqueues: std::cell::Cell::new(0),
        batched_checkpoint_copies: false,
        batched_checkpoint_save_only: false,
        batched_replay_copies: false,
        shared_replay_fold_scalars: false,
        replay_checkpoint_read: false,
        replay_prefix_family: false,
        shared_verify_gdn_scalars: false,
        dflash_history_projection_reuse: false,
        dflash_b8_specialized: false,
        dflash_history_weight_panels: false,
        dflash_history_panel_token8: false,
        dflash_context_key_norm_rope: false,
        batched_output_head_prepare: false,
        head_prepare_stores: false,
        batched_target_top1: false,
        verify_rope_reuse: false,
        output_head_draft_prepare_batches: std::cell::Cell::new(0),
        output_head_target_prepare_batches: std::cell::Cell::new(0),
        target_top1_batches: std::cell::Cell::new(0),
        shared_replay_fold_layers: std::cell::Cell::new(0),
        checkpoint_read_fold_layers: std::cell::Cell::new(0),
        checkpoint_copy_batch_enqueues: std::cell::Cell::new(0),
        checkpoint_copy_logical_ranges: std::cell::Cell::new(0),
        checkpoint_copy_logical_bytes: std::cell::Cell::new(0),
        replay_copy_batch_enqueues: std::cell::Cell::new(0),
        batched_embedding_gather: false,
        feature_capture_enqueues: std::cell::Cell::new(0),
        feature_append_enqueues: std::cell::Cell::new(0),
        embedding_gather_enqueues: std::cell::Cell::new(0),
        prefill_batched_head_rope_launches: std::cell::Cell::new(0),
        prefill_batched_head_rope_layer_rows: std::cell::Cell::new(0),
        llama_rotated_kv: math.full_attention_kv_basis.is_rotated(),
        planar_rotated_kv: false,
        prefix_boundary_audit: None,
        wide_history_capture: false,
        exact_output_head_batch: math.exact_output_head_batch,
        draft_top16_batch: math.draft_top16_batch,
        reference_boundary: None,
        poisoned: false,
        #[cfg(feature = "qwen35-raw-committed-state")]
        raw_state: None,
    };
    let mask: Vec<_> = (0..B)
        .flat_map(|_| crate::DFLASH_SEED_MASK_TOKEN.to_le_bytes())
        .collect();
    d.noise_ids.copy_from_host_sync(&d.stream, &mask)?;
    reset_plans(&mut d.plans, &d.stream)?;
    // decode_audit rebinds the draft using this device token slot. It must be
    // initialized even when restoring after suffix commit (no subsequent append).
    d.token
        .copy_from_host_sync(&d.stream, &d.anchor.to_le_bytes())?;
    d.stream.synchronize()?;
    Ok(d)
}

struct Destination {
    driver: Option<Driver>,
    expected_cursor: usize,
    expected_pending: Option<u32>,
    metadata: Vec<u8>,
    plan: TransferPlan,
    component: usize,
    offset: u64,
    h2d_bytes: u64,
    active: bool,
    finished: bool,
}

impl RestoreTarget for Destination {
    fn begin(&mut self, expected: &TransferPlan) -> io::Result<()> {
        if self.active || self.finished || expected != &self.plan {
            return Err(io_error("restore destination admission changed"));
        }
        let d = self
            .driver
            .as_ref()
            .ok_or_else(|| io_error("restore destination discarded"))?;
        if d.poisoned || d.plans.len() != 64 || self.plan.components.len() != 130 {
            return Err(io_error("restore destination topology"));
        }
        for (layer, plan) in d.plans.iter().enumerate() {
            if matches!(plan, DevicePlan::Full(_)) != (layer % 4 == 3) {
                return Err(io_error("restore layer kind changed"));
            }
            let buffers = plan.terminal_state_buffers();
            if buffers.len() != 2 {
                return Err(io_error("restore plane count"));
            }
            for (index, buffer) in buffers.iter().enumerate() {
                let expected = self.plan.components[1 + layer * 2 + index].bytes;
                let physical = if layer % 4 == 3 {
                    CAPACITY as u64 * crate::session_checkpoint::metadata::KV_ROW_BYTES
                } else {
                    expected
                };
                if buffer.len_bytes() as u64 != physical || expected > physical {
                    return Err(io_error("restore plane physical extent"));
                }
            }
        }
        if d.features.len_bytes() != HISTORY * 5 * H * 4 {
            return Err(io_error("restore physical feature ring"));
        }
        self.active = true;
        Ok(())
    }

    fn write_exact(&mut self, component: usize, offset: u64, bytes: &[u8]) -> io::Result<()> {
        if !self.active
            || self.finished
            || component != self.component
            || offset != self.offset
            || bytes.is_empty()
        {
            return Err(io_error("restore write is not the next initialized extent"));
        }
        let spec = self
            .plan
            .components
            .get(component)
            .ok_or_else(|| io_error("restore component"))?;
        let end = offset
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| io_error("restore offset overflow"))?;
        if end > spec.bytes {
            return Err(io_error("restore write exceeds initialized extent"));
        }
        let start = usize::try_from(offset).map_err(io_error)?;
        if component == 0 {
            if self.metadata[start..start + bytes.len()] != *bytes {
                return Err(io_error("restore descriptor drift after admission"));
            }
        } else {
            let d = self
                .driver
                .as_ref()
                .ok_or_else(|| io_error("restore destination discarded"))?;
            d.check_session_cancellation("checkpoint restore chunk")
                .map_err(io_error)?;
            let buffer = if component == 129 {
                &d.features
            } else {
                let index = component - 1;
                d.plans[index / 2].terminal_state_buffers()[index % 2]
            };
            buffer
                .copy_range_from_host_sync(&d.stream, start, bytes)
                .map_err(io_error)?;
            self.h2d_bytes += bytes.len() as u64;
        }
        self.offset = end;
        if end == spec.bytes {
            self.component += 1;
            self.offset = 0;
        }
        Ok(())
    }

    fn finish_unpublished(&mut self) -> io::Result<()> {
        if !self.active
            || self.finished
            || self.component != self.plan.components.len()
            || self.offset != 0
        {
            return Err(io_error("restore incomplete destination"));
        }
        let d = self
            .driver
            .as_ref()
            .ok_or_else(|| io_error("restore destination discarded"))?;
        d.stream.synchronize().map_err(io_error)?;
        d.check_session_cancellation("checkpoint restore finish")
            .map_err(io_error)?;
        if d.cursor != self.expected_cursor
            || self
                .expected_pending
                .is_some_and(|pending| d.anchor != pending)
        {
            return Err(io_error("restore cursor or pending token drift"));
        }
        self.finished = true;
        Ok(())
    }

    fn abort(&mut self) -> io::Result<()> {
        self.finished = false;
        if let Some(mut driver) = self.driver.take() {
            driver.poisoned = true;
            let result = driver.stream.synchronize().map_err(io_error);
            drop(driver);
            result
        } else {
            Ok(())
        }
    }
}

pub(super) struct RestoredTarget {
    pub driver: Driver,
    pub receipt: transfer::TransferReceipt,
    pub h2d_bytes: u64,
    pub wall_ms: f64,
}

/// Shared bounded writes, completion and rollback for distinct typed roots.
pub(super) fn restore_target_payload(
    driver: Driver,
    metadata: Vec<u8>,
    plan: TransferPlan,
    mut reader: LocalCheckpointReader,
    expected_manifest_sha256: &str,
    expected_pending: Option<u32>,
) -> Result<RestoredTarget> {
    let manifest_bytes = reader.manifest_bytes().to_vec();
    let mut destination = Destination {
        expected_cursor: driver.cursor,
        expected_pending,
        driver: Some(driver),
        metadata,
        plan: plan.clone(),
        component: 0,
        offset: 0,
        h2d_bytes: 0,
        active: false,
        finished: false,
    };
    let started = Instant::now();
    let receipt = transfer::restore(
        &mut reader,
        &mut destination,
        &manifest_bytes,
        expected_manifest_sha256,
        &plan,
    )
    .map_err(|e| invalid(e.to_string()))?;
    if !destination.finished {
        return Err(invalid("restored native destination not sealed"));
    }
    Ok(RestoredTarget {
        driver: destination
            .driver
            .take()
            .ok_or_else(|| invalid("restored driver absent"))?,
        receipt,
        h2d_bytes: destination.h2d_bytes,
        wall_ms: started.elapsed().as_secs_f64() * 1000.0,
    })
}

pub(super) fn rebuild_draft(
    d: &mut Driver,
    weights: &DFlashQ8CudaWeights,
    target: &target_weights::TargetWeights,
    math: &NativeMathPolicy,
    submit_seed: bool,
) -> Result<DFlashQ8CudaBlock> {
    d.full_context_feature_digest()?;
    let mask_token_id = crate::DFLASH_SEED_MASK_TOKEN;
    let mut tokens = vec![mask_token_id; B];
    tokens[0] = d.anchor;
    let seed = DFlashDraftBlockInput {
        request_generation: d.generation,
        absolute_position_start: d.cursor,
        anchor_token: d.anchor,
        block_tokens: tokens,
        mask_token_id,
        target_feature_sha256: d.feature_sha,
    };
    let capture = DFlashTargetFeatureCaptureReceipt {
        schema: DFLASH_TARGET_CAPTURE_SCHEMA.into(),
        request_generation: d.generation,
        context_start_position: d.active_context_start(),
        context_len: d.active_context_rows(),
        target_layer_ids: DFLASH_TARGET_CAPTURE_LAYER_IDS.to_vec(),
        target_hidden_size: H,
        feature_width: 5 * H,
        feature_elements: d.active_context_rows() * 5 * H,
        source_target_model_sha256: d.identity,
        feature_sha256: d.feature_sha,
    };
    let noise = crate::embed_dflash_seed(weights.config(), &seed, |token| {
        target
            .embedding_row(token as usize)
            .map_err(|e| e.to_string())
    })
    .map_err(|e| invalid(e.to_string()))?;
    let mut draft = match d.growing_history {
        Some(history) => DFlashQ8CudaBlock::from_resident_growing_device_features(
            weights,
            &seed,
            &capture,
            d.active_feature_view()?,
            &noise,
            history,
            HISTORY,
            &d.stream,
        ),
        None => DFlashQ8CudaBlock::from_resident_device_features(
            weights,
            &seed,
            &capture,
            d.active_feature_view()?,
            &noise,
            d.active_context_start(),
            HISTORY,
            &d.stream,
        ),
    }
    .map_err(|e| invalid(e.to_string()))?;
    if math.batched_draft_context {
        draft
            .enable_context_rows_launch()
            .map_err(|e| invalid(e.to_string()))?;
    }
    if math.batched_draft_block {
        draft
            .enable_block_rows_launch()
            .map_err(|e| invalid(e.to_string()))?;
    }
    if math.draft_q8_tile4 {
        draft
            .enable_q8_tile4()
            .map_err(|e| invalid(e.to_string()))?;
    }
    draft.enable_history_prepare(d.dflash_history_projection_reuse, d.dflash_context_key_norm_rope)
        .map_err(|e| invalid(e.to_string()))?;
    // A newly allocated draft is not a completed resident block. Session decode
    // deliberately rebinds only an actually submitted predecessor, including
    // after a teacher-forced suffix. Rebuild and drain this private block so
    // that the existing reuse guard remains valid; do not forge `submitted` or
    // publish these provisional outputs into the target/HTTP token ledger.
    if submit_seed {
        draft
            .prepare_context()
            .map_err(|e| invalid(e.to_string()))?;
        draft.submit().map_err(|e| invalid(e.to_string()))?;
        draft.synchronize().map_err(|e| invalid(e.to_string()))?;
    }
    Ok(draft)
}

impl Qwen35ResidentDraftCuda {
    /// Construct a fresh native session from a fully authenticated checkpoint.
    /// Returns an audit handle only; real suffix/decode and HTTP exact-history
    /// recovery still require independent numerical and product qualification.
    pub fn restore_native_session_checkpoint(
        &mut self,
        config: &Qwen35AutoregressiveCudaConfig,
        generation_config_path: &Path,
        directory: &Path,
        expected_manifest_sha256: &str,
        admission: &NativeRestoreAdmission,
    ) -> Result<Value> {
        if self.session.is_some() || self.has_retained_input_prefix() || self.poisoned {
            return Err(invalid(
                "restore requires healthy owner without a live session",
            ));
        }
        self.validate_checkpoint_bindings(&admission.bindings)?;
        let target = self
            .target
            .as_ref()
            .ok_or_else(|| invalid("restore target owner missing"))?;
        if config.device_ordinal != 0
            || target.device != 0
            || self.weights.device_ordinal() != 0
            || config.capacity != CAPACITY
            || target.gguf_path != config.gguf_path
            || self.vision_path.as_ref() != config.vision_safetensors_path.as_ref()
            || target.layers.len() != 64
        {
            return Err(invalid("restore live device/model/residency configuration"));
        }
        let config_bytes = read_stable_regular_file(&config.config_path, "restore model config")?;
        let tokenizer_bytes =
            read_stable_regular_file(&config.tokenizer_path, "restore tokenizer")?;
        let generation_bytes =
            read_stable_regular_file(generation_config_path, "restore generation config")?;
        if sha256_hex(&config_bytes) != admission.bindings.model_config_sha256
            || sha256_hex(&tokenizer_bytes) != admission.bindings.tokenizer_sha256
            || sha256_hex(&generation_bytes) != admission.bindings.generation_config_sha256
        {
            return Err(invalid(
                "restore live input files differ from admitted identities",
            ));
        }
        let model = Qwen35Config::parse(&config_bytes)?;
        model.validate_exact_profile()?;
        let metadata =
            LocalCheckpointReader::read_session_descriptor(directory, expected_manifest_sha256)
                .map_err(|e| invalid(e.to_string()))?;
        let descriptor = SessionDescriptor::decode(&metadata, &admission.bindings, &admission.math)
            .map_err(|e| invalid(e.to_string()))?;
        admission
            .validate_descriptor(&descriptor)
            .map_err(|e| invalid(e.to_string()))?;
        if descriptor
            .ledger
            .consumed
            .len()
            .checked_add(B)
            .is_none_or(|n| n > CAPACITY)
        {
            return Err(invalid(
                "restore has no capacity for a native continuation transaction",
            ));
        }
        if descriptor.math.draft_top16_batch && !cfg!(feature = "qwen35-draft-top16-batch") {
            return Err(invalid("restore requires compiled draft top16 policy"));
        }
        let generation: Value =
            serde_json::from_slice(&generation_bytes).map_err(|e| invalid(e.to_string()))?;
        let eos: Vec<u32> = match &generation["eos_token_id"] {
            Value::Array(_) => serde_json::from_value(generation["eos_token_id"].clone()),
            Value::Number(_) => {
                serde_json::from_value(generation["eos_token_id"].clone()).map(|id| vec![id])
            }
            _ => return Err(invalid("restore generation EOS field")),
        }
        .map_err(|e| invalid(e.to_string()))?;
        if eos != descriptor.eos_token_ids || !eos.contains(&model.eos_token_id) || eos.len() > 8 {
            return Err(invalid("restore generation EOS differs from snapshot"));
        }
        let plan = descriptor
            .transfer_plan()
            .map_err(|e| invalid(e.to_string()))?;
        let reader = LocalCheckpointReader::open(directory, expected_manifest_sha256, &plan)
            .map_err(|e| invalid(e.to_string()))?;
        // All static/file admission above leaves a healthy owner untouched.
        // After allocation or submission failure, do not retry within this owner.
        self.poisoned = true;
        let before = cuda_device_allocation_snapshot();
        let started = Instant::now();
        let operation = (|| -> Result<(CommittedSession, Value)> {
            let mut driver = fresh_driver(
                target,
                &model,
                FreshTarget {
                    consumed: &descriptor.ledger.consumed,
                    features: &descriptor.features,
                    math: &descriptor.math,
                    bindings: &descriptor.bindings,
                    pending: Some(descriptor.ledger.pending),
                    generation: descriptor.ledger.generation,
                    metadata: &metadata,
                },
                &admission.new_session_id,
            )?;
            driver.session_cancellation = self.session_cancellation.clone();
            let restored = restore_target_payload(
                driver,
                metadata,
                plan.clone(),
                reader,
                expected_manifest_sha256,
                Some(descriptor.ledger.pending),
            )?;
            let receipt = restored.receipt;
            let transfer_wall_ms = restored.wall_ms;
            let h2d_bytes = restored.h2d_bytes;
            let mut driver = restored.driver;
            let rebuild_started = Instant::now();
            let draft = rebuild_draft(&mut driver, &self.weights, target, &descriptor.math, true)?;
            let draft_rebuild_wall_ms = rebuild_started.elapsed().as_secs_f64() * 1000.0;
            if descriptor.math.batched_target_verify {
                driver.batch_verify_execution = Some(BatchedVerifyExecution::new(
                    &driver,
                    CAPACITY,
                    model.rms_norm_eps as f32,
                    model.rope_theta as f32,
                    descriptor.math.reuse_batched_plan,
                    false,
                    false, // Rebuild without restoring graph execution policy.
                )?);
            }
            let saved = &descriptor.ledger;
            let ledger = Ledger {
                session_id: admission.new_session_id.clone(),
                consumed: saved.consumed.clone(),
                generated: saved.generated.clone(),
                suffix_ranges: saved
                    .user_suffix_ranges
                    .iter()
                    .map(|r| (r.start, r.end))
                    .collect(),
                pending: saved.pending,
                pending_published: saved.pending_published,
                generation: saved.generation,
                revision: saved.revision,
                capacity: saved.capacity,
            };
            let mut session = CommittedSession {
                draft,
                driver,
                ledger,
                eos: descriptor.eos_token_ids.clone(),
                draft_growth_base: None,
                checkpoint: Value::Null,
                live_boundaries: false, // Full restore verification precedes live continuation.
                // Authenticate complete restored bytes first. The controller may
                // then enter live continuation using its current explicit launch
                // policy; never deserialize a prior owner's performance authority.
                fast_session_path: false,
            };
            let seal_started = Instant::now();
            session.checkpoint = session.seal()?;
            if words(&session.driver.stream, &session.driver.token, 0, 1)?
                != vec![descriptor.ledger.pending]
            {
                return Err(invalid(
                    "restored device pending-token slot differs from ledger",
                ));
            }
            let seal_wall_ms = seal_started.elapsed().as_secs_f64() * 1000.0;
            let snapshot = cuda_device_allocation_snapshot();
            let report = json!({
                "schema":"qwen35-native-session-checkpoint-restore-v1",
                "source_session_id":saved.origin_session_id,"session_id":admission.new_session_id,
                "manifest_sha256":receipt.manifest_sha256,"transfer_binding_sha256":plan.binding_sha256,
                "components":receipt.components,"chunks":receipt.chunks,"payload_bytes":receipt.payload_bytes,
                "staging_bytes":receipt.staging_bytes,"checkpoint_h2d_bytes":h2d_bytes,
                "restored_prefix_tokens":session.driver.cursor,"prefix_recompute_tokens":0,
                "fresh_native_session_owner":true,"old_cuda_authority_restored":false,
                "device_pending_token_verified":true,
                "draft_restore_recipe":"fresh_canonical_q8_from_target_features_v1",
                "draft_rebuild_submitted_and_drained":true,
                "draft_rebuild_outputs_published":false,
                "draft_rebuild_qualified":false,"real_suffix_executed":false,
                "http_session_restored":false,"complete_session_restore_qualified":false,
                "transfer_wall_ms":transfer_wall_ms,"draft_rebuild_wall_ms":draft_rebuild_wall_ms,
                "seal_audit_wall_ms":seal_wall_ms,"restore_wall_ms":started.elapsed().as_secs_f64()*1000.0,
                "allocation_before":allocation_snapshot_report(before),
                "allocation_after":allocation_snapshot_report(snapshot),
                "checkpoint":session.checkpoint,"product_ready":false,"qualified":false,
                "semantic_qualification":false,"visible_tps":false,
            });
            Ok((session, report))
        })();
        let (session, report) = operation?;
        // Only a completed copy, rebuilt draft and sealed native state become
        // reachable. Errors above leave no partial session in the outer owner.
        self.session = Some(session);
        self.poisoned = false;
        Ok(report)
    }
}
