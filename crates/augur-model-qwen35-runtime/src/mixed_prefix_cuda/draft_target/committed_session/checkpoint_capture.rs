//! Opt-in native complete-target capture. No GPU restore/HTTP qualification yet.
//! A borrowed mutable session retains both streams and all CUDA buffers through
//! synchronous bounded reads. Logical ledgers come from the owner, never a file.

use super::*;
use crate::session_checkpoint::filesystem::LocalCheckpointWriter;
use crate::session_checkpoint::metadata::{
    DraftRestoreRecipe, FeatureExtent, HistoryMode, ModelBindings, NativeLedger, NativeMathPolicy,
    SamplingPolicy, SessionDescriptor, UserSuffixRange, SESSION_SCHEMA,
};
use crate::session_checkpoint::{self as transfer, CaptureSource, TransferPlan};
use std::io;

fn io_error(error: impl std::fmt::Display) -> io::Error {
    io::Error::other(error.to_string())
}

fn invalid(error: impl std::fmt::Display) -> Qwen35MixedPrefixCudaError {
    super::invalid(error.to_string())
}

impl CommittedSession {
    fn checkpoint_descriptor(&self, bindings: &ModelBindings) -> Result<SessionDescriptor> {
        let d = &self.driver;
        if d.poisoned
            || d.cursor != self.ledger.consumed.len()
            || d.anchor != self.ledger.pending
            || d.generation != self.ledger.generation
            || d.plans.len() != 64
            || d.capacity != 196608
            || d.checkpoint_model_config_sha256 != bindings.model_config_sha256
            || d.checkpoint_tokenizer_sha256 != bindings.tokenizer_sha256
            || d.checkpoint_generation_config_sha256.as_ref()
                != Some(&bindings.generation_config_sha256)
        {
            return Err(invalid(
                "native checkpoint live owner/input contract mismatch",
            ));
        }
        let mode = if let Some(history) = d.growing_history {
            if history.cursor() != d.cursor
                || history.active_rows() != d.cursor.min(2048)
                || history.capacity_rows() != 2048
                || history.start_position() != d.active_context_start()
            {
                return Err(invalid(
                    "native checkpoint growing logical/physical extent mismatch",
                ));
            }
            HistoryMode::Growing2048 {
                generation: history.generation(),
            }
        } else {
            HistoryMode::Fixed2048
        };
        let (batched_draft_context, batched_draft_block, draft_q8_tile4) =
            self.draft.checkpoint_launch_policy().map_err(invalid)?;
        let descriptor = SessionDescriptor {
            schema: SESSION_SCHEMA.into(),
            bindings: bindings.clone(),
            ledger: NativeLedger {
                origin_session_id: self.ledger.session_id.clone(),
                consumed: self.ledger.consumed.clone(),
                generated: self.ledger.generated.clone(),
                user_suffix_ranges: self
                    .ledger
                    .suffix_ranges
                    .iter()
                    .map(|&(start, end)| UserSuffixRange { start, end })
                    .collect(),
                pending: self.ledger.pending,
                pending_published: self.ledger.pending_published,
                generation: self.ledger.generation,
                revision: self.ledger.revision,
                capacity: self.ledger.capacity,
            },
            features: FeatureExtent {
                start: d.active_context_start(),
                rows: d.active_context_rows(),
                physical_rows: 2048,
                mode,
            },
            math: NativeMathPolicy {
                full_attention_kv_basis: crate::session_checkpoint::metadata::FullAttentionKvBasis::from_layout(d.llama_rotated_kv, d.planar_rotated_kv),
                exact_projection_tiles: d.exact_projection_tiles,
                exact_output_head_batch: d.exact_output_head_batch,
                draft_top16_batch: d.draft_top16_batch,
                parallel_receipt_digest: d.parallel_receipt_digest,
                batched_target_verify: d.batch_verify_execution.is_some(),
                reuse_batched_plan: d.checkpoint_reuse_batched_plan,
                batched_suffix: d.batched_suffix,
                retain_full_accept_state: d.verified_features.is_some(),
                share_target_rope: d.checkpoint_share_target_rope,
                batched_draft_context,
                batched_draft_block,
                draft_q8_tile4,
            },
            draft_restore: DraftRestoreRecipe::FreshCanonicalQ8FromTargetFeaturesV1,
            sampling: SamplingPolicy::GreedyArgmaxV1,
            eos_token_ids: self.eos.clone(),
            native_request_identity: d.identity.iter().map(|b| format!("{b:02x}")).collect(),
        };
        descriptor.validate().map_err(io_error).map_err(invalid)?;
        let plan = descriptor
            .transfer_plan()
            .map_err(io_error)
            .map_err(invalid)?;
        for (layer, device_plan) in d.plans.iter().enumerate() {
            let full = matches!(device_plan, DevicePlan::Full(_));
            if full != (layer % 4 == 3) {
                return Err(invalid("checkpoint target topology drift"));
            }
            for (plane, buffer) in device_plan.terminal_state_buffers().iter().enumerate() {
                let declared = plan.components[1 + layer * 2 + plane].bytes as usize;
                let physical = if full { d.capacity * 1088 } else { declared };
                if buffer.len_bytes() != physical
                    || buffer.device_ordinal() != d.stream.device_ordinal()
                {
                    return Err(invalid("checkpoint target physical plane geometry drift"));
                }
            }
        }
        if d.features.len_bytes() != 2048 * 5 * H * 4
            || d.active_feature_view()?.len_bytes() as u64 != plan.components[129].bytes
        {
            return Err(invalid("checkpoint feature initialized extent drift"));
        }
        Ok(descriptor)
    }
}

enum CaptureOwner<'a> {
    Session(&'a mut CommittedSession),
    InputPrefix(&'a mut InputPrefixOwner),
}

impl CaptureOwner<'_> {
    fn driver(&self) -> &Driver {
        match self {
            Self::Session(session) => &session.driver,
            Self::InputPrefix(prefix) => &prefix.driver,
        }
    }

    fn poison(&mut self) {
        match self {
            Self::Session(session) => session.driver.poisoned = true,
            Self::InputPrefix(prefix) => prefix.driver.poisoned = true,
        }
    }

    fn synchronize(&self) -> io::Result<()> {
        let draft = match self {
            Self::Session(session) => session.draft.synchronize().map_err(io_error),
            Self::InputPrefix(_) => Ok(()),
        };
        let target = self.driver().stream.synchronize().map_err(io_error);
        draft.and(target)
    }

    fn metadata(&self, bindings: &ModelBindings) -> io::Result<Vec<u8>> {
        match self {
            Self::Session(session) => session
                .checkpoint_descriptor(bindings)
                .map_err(io_error)?
                .encode(),
            Self::InputPrefix(prefix) => prefix
                .checkpoint_descriptor(bindings)
                .map_err(io_error)?
                .encode(),
        }
    }
}

struct Source<'a> {
    owner: CaptureOwner<'a>,
    bindings: ModelBindings,
    metadata: Vec<u8>,
    plan: TransferPlan,
    proof: Option<augur_backend::CudaStreamSubmissionReceipt>,
    active: bool,
    drained: bool,
    read_bytes: u64,
}

impl Drop for Source<'_> {
    fn drop(&mut self) {
        if self.active && !self.drained {
            self.owner.poison();
            let _ = self.owner.synchronize();
        }
    }
}

impl CaptureSource for Source<'_> {
    fn begin(&mut self) -> io::Result<TransferPlan> {
        if self.active {
            return Err(io_error("capture cannot begin twice"));
        }
        self.active = true;
        self.owner
            .driver()
            .check_session_cancellation("checkpoint capture begin")
            .map_err(io_error)?;
        self.owner.synchronize()?;
        if self.owner.metadata(&self.bindings)? != self.metadata {
            return Err(io_error("capture source descriptor changed before begin"));
        }
        self.proof = Some(
            self.owner
                .driver()
                .stream
                .submission_receipt()
                .map_err(io_error)?,
        );
        Ok(self.plan.clone())
    }

    fn read_exact(&mut self, component: usize, offset: u64, output: &mut [u8]) -> io::Result<()> {
        if !self.active || self.drained || output.is_empty() {
            return Err(io_error("inactive capture read"));
        }
        self.owner
            .driver()
            .check_session_cancellation("checkpoint capture chunk")
            .map_err(io_error)?;
        let spec = self
            .plan
            .components
            .get(component)
            .ok_or_else(|| io_error("capture component"))?;
        if offset
            .checked_add(output.len() as u64)
            .is_none_or(|end| end > spec.bytes)
        {
            return Err(io_error("capture initialized range exceeded"));
        }
        if component == 0 {
            let start = offset as usize;
            output.copy_from_slice(&self.metadata[start..start + output.len()]);
        } else {
            let d = self.owner.driver();
            let buffer = if component == 129 {
                &d.features
            } else {
                let index = component - 1;
                d.plans[index / 2].terminal_state_buffers()[index % 2]
            };
            let proof = self
                .proof
                .ok_or_else(|| io_error("capture completion proof absent"))?;
            let completion = buffer
                .copy_range_to_host_sync_with_submission_proof(
                    &d.stream,
                    offset as usize,
                    output,
                    proof,
                )
                .map_err(io_error)?;
            if !completion.covers_submission(proof)
                || !completion.belongs_to_stream(&d.stream).map_err(io_error)?
            {
                return Err(io_error("capture read completion owner/epoch mismatch"));
            }
        }
        self.read_bytes += output.len() as u64;
        Ok(())
    }

    fn finish(&mut self) -> io::Result<TransferPlan> {
        self.owner
            .driver()
            .check_session_cancellation("checkpoint capture finish")
            .map_err(io_error)?;
        self.owner.synchronize()?;
        if self.owner.metadata(&self.bindings)? != self.metadata
            || self.read_bytes != self.plan.validate()?
        {
            return Err(io_error(
                "capture source boundary or complete byte count drift",
            ));
        }
        self.drained = true;
        Ok(self.plan.clone())
    }

    fn abort(&mut self) -> io::Result<()> {
        self.owner.poison();
        let result = self.owner.synchronize();
        self.drained = result.is_ok();
        result
    }
}

impl Qwen35ResidentDraftCuda {
    /// Shared save/restore admission against live immutable owners. Template,
    /// namespace and policy hashes also require independent controller binding.
    pub(super) fn validate_checkpoint_bindings(&self, bindings: &ModelBindings) -> Result<()> {
        if self.poisoned {
            return Err(invalid("checkpoint requires healthy native owner"));
        }
        bindings.validate().map_err(invalid)?;
        let compiled = crate::build_identity::identity();
        if compiled["source_dirty"] != false
            || bindings.frozen_build_sha256
                != transfer::sha256(&serde_json::to_vec(&compiled).map_err(invalid)?)
        {
            return Err(invalid(
                "checkpoint requires exact clean compiled build identity",
            ));
        }
        let target = self
            .target
            .as_ref()
            .ok_or_else(|| invalid("checkpoint requires resident target"))?;
        let vision = self
            .vision
            .as_ref()
            .and_then(|v| v.report().vision)
            .ok_or_else(|| invalid("checkpoint requires resident Vision"))?;
        if target.manifest.file_sha256.as_ref() != Some(&bindings.model_sha256)
            || target.config_sha256 != bindings.model_config_sha256
            || self.weights.file_sha256() != bindings.q8_draft_sha256
            || vision.payload_sha256 != bindings.vision_payload_sha256
            || !vision.device_resident
            || sha256_hex(&read_stable_regular_file(
                &self.draft_config_path,
                "checkpoint draft config",
            )?) != bindings.draft_config_sha256
        {
            return Err(invalid(
                "checkpoint differs from resident model/draft/vision/config",
            ));
        }
        Ok(())
    }

    /// CPU-only admission inventory from the live owner, never client history.
    /// This does not capture state or create a CUDA completion capability.
    pub fn describe_native_session_checkpoint(
        &self,
        expected: &Value,
        bindings: &ModelBindings,
    ) -> Result<SessionDescriptor> {
        self.validate_checkpoint_bindings(bindings)?;
        let session = self
            .session
            .as_ref()
            .ok_or_else(|| invalid("no retained native session"))?;
        if expected != &session.checkpoint {
            return Err(invalid("stale checkpoint descriptor admission"));
        }
        session.checkpoint_descriptor(bindings)
    }

    /// Opt-in native capture, not HTTP save/restore qualification.
    pub fn save_native_session_checkpoint(
        &mut self,
        expected: &Value,
        bindings: ModelBindings,
        directory: &Path,
    ) -> Result<Value> {
        self.validate_checkpoint_bindings(&bindings)?;
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| invalid("no retained native session"))?;
        if expected != &session.checkpoint {
            return Err(invalid("stale checkpoint save admission"));
        }
        let descriptor = session.checkpoint_descriptor(&bindings)?;
        let metadata = descriptor.encode().map_err(invalid)?;
        let plan = descriptor.transfer_plan().map_err(invalid)?;
        // File path/quota failure before capture does not touch CUDA/session state.
        let mut writer = LocalCheckpointWriter::create(directory, &plan).map_err(invalid)?;
        let mut source = Source {
            owner: CaptureOwner::Session(session),
            bindings,
            metadata,
            plan: plan.clone(),
            proof: None,
            active: false,
            drained: false,
            read_bytes: 0,
        };
        let started = Instant::now();
        let result = transfer::save(&mut source, &mut writer, &plan, transfer::MAX_CHUNK_BYTES);
        let bytes = source.read_bytes;
        drop(source);
        match result {
            Ok(receipt) => Ok(json!({
                "schema":"qwen35-native-session-checkpoint-capture-v1",
                "manifest_sha256":receipt.manifest_sha256,"transfer_binding_sha256":plan.binding_sha256,
                "components":receipt.components,"chunks":receipt.chunks,"payload_bytes":receipt.payload_bytes,
                "staging_bytes":receipt.staging_bytes,"capture_total_read_bytes":bytes,
                "wall_ms":started.elapsed().as_secs_f64()*1000.0,
                "draft_restore_recipe":"fresh_canonical_q8_from_target_features_v1",
                "draft_rebuild_qualified":false,"http_session_state_saved":false,
                "complete_session_restore_qualified":false,"product_ready":false,
                "qualified":false,"semantic_qualification":false,"visible_tps":false
            })),
            Err(error) => {
                self.poisoned = true;
                Err(invalid(error.to_string()))
            }
        }
    }

    /// Persist an actual input-only owner through the same bounded stable
    /// transfer implementation. This does not make a generated session/hit.
    pub fn save_native_input_prefix(
        &mut self,
        expected: &Value,
        bindings: ModelBindings,
        directory: &Path,
    ) -> Result<Value> {
        let descriptor = self.describe_native_input_prefix(expected, &bindings)?;
        let metadata = descriptor.encode().map_err(invalid)?;
        let plan = descriptor.transfer_plan().map_err(invalid)?;
        let mut writer = LocalCheckpointWriter::create(directory, &plan).map_err(invalid)?;
        let prefix = self
            .input_prefix
            .as_mut()
            .ok_or_else(|| invalid("input-prefix absent"))?;
        let mut source = Source {
            owner: CaptureOwner::InputPrefix(prefix),
            bindings,
            metadata,
            plan: plan.clone(),
            proof: None,
            active: false,
            drained: false,
            read_bytes: 0,
        };
        let started = Instant::now();
        let result = transfer::save(&mut source, &mut writer, &plan, transfer::MAX_CHUNK_BYTES);
        let bytes = source.read_bytes;
        drop(source);
        match result {
            Ok(receipt) => Ok(json!({
                "schema":"qwen35-native-input-prefix-capture-v1",
                "manifest_sha256":receipt.manifest_sha256,"transfer_binding_sha256":plan.binding_sha256,
                "components":receipt.components,"chunks":receipt.chunks,"payload_bytes":receipt.payload_bytes,
                "staging_bytes":receipt.staging_bytes,"capture_total_read_bytes":bytes,
                "wall_ms":started.elapsed().as_secs_f64()*1000.0,
                "input_tokens":descriptor.consumed_input.len(),"generated_tokens":0,"published_tokens":0,
                "draft_executed":false,"http_session_state_saved":false,"cache_hit_qualified":false,
                "product_ready":false,"qualified":false,"semantic_qualification":false,"visible_tps":false
            })),
            Err(error) => {
                self.poisoned = true;
                Err(invalid(error.to_string()))
            }
        }
    }
}
