//! Explicit HTTP persistence controller. File/CPU admission precedes CUDA;
//! ambiguous capture/restore errors escape to the service fail-closed shutdown.
use super::*;
use augur_model_qwen35_runtime::session_checkpoint::{
    filesystem::http::{HttpCheckpointStore, PinnedHttpCheckpointProfile},
    http::{
        HttpSessionCheckpoint, HttpVisibleLedger, PublishedTurn, StopReason, TargetStateSeal,
        HTTP_SCHEMA,
    },
    metadata::NativeRestoreAdmission,
};

pub(super) struct Persistence {
    profile: PinnedHttpCheckpointProfile,
    store: HttpCheckpointStore,
    profile_sha: String,
    owner_id: String,
    epoch: u64,
}

pub(super) struct LiveSession<'a> {
    pub owner: &'a mut Qwen35ResidentDraftCuda,
    pub config: &'a Qwen35AutoregressiveCudaConfig,
    pub generation: &'a std::path::Path,
    pub tokenizer: &'a Tokenizer,
    pub active: &'a mut Option<Active>,
    pub session_ids: &'a mut BTreeSet<String>,
    pub profile: Profile,
}

impl Persistence {
    pub fn open(
        path: &std::path::Path,
        sha: &str,
        evidence: &std::path::Path,
        tokenizer_sha: &str,
    ) -> Result<Self, Box<dyn Error>> {
        let profile = PinnedHttpCheckpointProfile::open(path, sha)?;
        if profile.profile.bindings.tokenizer_sha256 != tokenizer_sha {
            return Err("checkpoint profile differs from live HTTP tokenizer".into());
        }
        let build = augur_model_qwen35_runtime::build_identity::identity();
        if build["source_dirty"] != false
            || digest(&serde_json::to_vec(&build)?) != profile.profile.bindings.frozen_build_sha256
        {
            return Err("checkpoint profile requires exact clean live build".into());
        }
        let store = HttpCheckpointStore::open(&profile.profile.store_root, profile.profile.budget)?;
        let epoch = u64::try_from(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos(),
        )?;
        let owner_id = digest(&serde_json::to_vec(&json!([
            sha,
            evidence,
            std::process::id(),
            epoch
        ]))?);
        Ok(Self {
            profile,
            store,
            profile_sha: sha.into(),
            owner_id,
            epoch,
        })
    }

    pub fn ready(&self) -> Value {
        json!({"schema":"qwen35-http-checkpoint-service-v1","enabled":true,
            "profile_sha256":self.profile_sha,"owner_id":self.owner_id,"owner_epoch":self.epoch,
            "restore_requires_new_owner":true,"client_delivery_acknowledged":false,"qualified":false})
    }

    pub fn execute(
        &mut self,
        command: &SessionCommand,
        live: LiveSession<'_>,
    ) -> Result<Option<Value>, Box<dyn Error>> {
        let LiveSession {
            owner,
            config,
            generation,
            tokenizer,
            active,
            session_ids,
            profile,
        } = live;
        let max_history = profile.maximum_history_tokens();
        let growing = profile.uses_initialized_history();
        match command {
            SessionCommand::SessionSave {
                request_id,
                session_id,
                revision,
                checkpoint_id,
            } => {
                let start = Instant::now();
                let Some(a) = active.as_ref() else {
                    reject("save requires active session")?;
                    return Ok(None);
                };
                if !active_matches(a, session_id, *revision) {
                    reject("save session/revision mismatch")?;
                    return Ok(None);
                }
                let descriptor = owner.describe_native_session_checkpoint(
                    &a.checkpoint,
                    &self.profile.profile.bindings,
                )?;
                if descriptor.math != self.profile.profile.math
                    || descriptor.ledger.consumed != a.consumed
                {
                    return Err("save differs from startup policy or authoritative history".into());
                }
                // A live boundary is a same-owner capability, not a saved-byte
                // fingerprint. Acquire the latter only for this explicit save.
                let audited_checkpoint = owner.audit_native_session_checkpoint(&a.checkpoint)?;
                let target_seal = TargetStateSeal::from_checkpoint(&audited_checkpoint, &descriptor)?;
                let ledger = &descriptor.ledger;
                let mut envelope = HttpSessionCheckpoint {
                    schema: HTTP_SCHEMA.into(),
                    checkpoint_id: checkpoint_id.clone(),
                    origin_owner_id: self.owner_id.clone(),
                    origin_owner_epoch: self.epoch,
                    native_manifest_sha256: digest(b"unpublished"),
                    native_transfer_binding_sha256: descriptor.transfer_plan()?.binding_sha256,
                    native_descriptor_sha256: digest(&descriptor.encode()?),
                    target_seal,
                    bindings: descriptor.bindings.clone(),
                    math: descriptor.math.clone(),
                    eos_token_ids: descriptor.eos_token_ids.clone(),
                    ledger: HttpVisibleLedger {
                        origin_session_id: a.id.clone(),
                        revision: ledger.revision,
                        generation: ledger.generation,
                        consumed_token_count: a.consumed.len(),
                        consumed_input_digest: a.checkpoint["consumed_input_digest"]
                            .as_str()
                            .ok_or("consumed digest")?
                            .into(),
                        generated_token_count: ledger.generated.len(),
                        pending_token: ledger.pending,
                        pending_token_published: ledger.pending_published,
                        turns: a.turns.clone(),
                    },
                    capacity: 196608,
                    context_rows: 2048,
                };
                envelope.validate_native(&descriptor, &envelope.native_manifest_sha256)?;
                envelope.validate_text(|ids| {
                    tokenizer
                        .decode(ids, true)
                        .map_err(|e| io::Error::other(e.to_string()))
                })?;
                let slot = match self
                    .store
                    .reserve(checkpoint_id, descriptor.transfer_plan()?.validate()?)
                {
                    Ok(s) => s,
                    Err(e) => {
                        // The read-only seal above succeeded, but no capture
                        // or saved checkpoint exists. Bind the unchanged live
                        // boundary so a client can keep chatting or explicitly
                        // retry after fixing quota/disk space. Generic/unbound
                        // rejections remain insufficient to retain a session.
                        return Ok(Some(json!({
                            "schema":"qwen35-http-checkpoint-admission-rejection-v1",
                            "event":"rejected","operation":"save","before_submit":true,
                            "capture_started":false,"session_preserved":true,
                            "request_id":request_id,"session_id":session_id,"revision":revision,
                            "checkpoint_id":checkpoint_id,"profile_sha256":self.profile_sha,
                            "owner_id":self.owner_id,"owner_epoch":self.epoch,
                            "session":metadata(a),"detail":format!("checkpoint admission: {e}"),
                            "qualified":false,"product_ready":false,
                            "semantic_qualification":false,"visible_tps":false
                        })));
                    }
                };
                let capture = owner.save_native_session_checkpoint(
                    &a.checkpoint,
                    descriptor.bindings.clone(),
                    &slot.native_path(),
                )?;
                if owner.describe_native_session_checkpoint(&a.checkpoint, &descriptor.bindings)?
                    != descriptor
                {
                    return Err("capture changed live session boundary".into());
                }
                envelope.native_manifest_sha256 = capture["manifest_sha256"]
                    .as_str()
                    .ok_or("capture manifest hash")?
                    .into();
                let http_sha = slot.publish(&envelope)?;
                Ok(Some(
                    json!({"schema":"qwen35-http-checkpoint-terminal-v1","event":"session_saved",
                    "request_id":request_id,"session_id":session_id,"revision":revision,
                    "owner_id":self.owner_id,"owner_epoch":self.epoch,"profile_sha256":self.profile_sha,
                    "checkpoint_id":checkpoint_id,"http_manifest_sha256":http_sha,
                    "native_manifest_sha256":envelope.native_manifest_sha256,"native_capture":capture,
                    "session":metadata(a),"save_wall_ms":start.elapsed().as_secs_f64()*1000.,
                    "qualified":false,"product_ready":false,"semantic_qualification":false,"visible_tps":false}),
                ))
            }
            SessionCommand::SessionRestore {
                request_id,
                session_id,
                checkpoint_id,
                http_manifest_sha256,
            } => {
                if active.is_some() || session_ids.contains(session_id) {
                    reject("restore requires unused destination session")?;
                    return Ok(None);
                }
                let start = Instant::now();
                // All CPU validation failures here are before native mutation.
                let admitted = (|| -> Result<_, Box<dyn Error>> {
                    let read = self.store.read(
                        checkpoint_id,
                        http_manifest_sha256,
                        &self.profile.profile.bindings,
                        &self.profile.profile.math,
                    )?;
                    read.envelope.validate_restore(
                        &self.owner_id,
                        self.epoch,
                        &NativeRestoreAdmission {
                            new_session_id: session_id.clone(),
                            bindings: self.profile.profile.bindings.clone(),
                            math: self.profile.profile.math.clone(),
                        },
                        &read.descriptor,
                        &read.envelope.native_manifest_sha256,
                    )?;
                    read.envelope.validate_text(|ids| {
                        tokenizer
                            .decode(ids, true)
                            .map_err(|e| io::Error::other(e.to_string()))
                    })?;
                    let original = read.descriptor.ledger.validate()?;
                    let source_growing=matches!(read.descriptor.features.mode,
                        augur_model_qwen35_runtime::session_checkpoint::metadata::HistoryMode::Growing2048 {..});
                    if !profile.admits_initial(original)
                        || source_growing != growing
                    {
                        return Err(
                            "saved session differs from selected initial-context/history profile"
                                .into(),
                        );
                    }
                    if read.descriptor.ledger.consumed.len() + 134 > max_history {
                        return Err("restored history exceeds selected service profile".into());
                    }
                    Ok(read)
                })();
                let read = match admitted {
                    Ok(r) => r,
                    Err(e) => {
                        reject(&format!("checkpoint restore admission: {e}"))?;
                        return Ok(None);
                    }
                };
                let report = owner.restore_native_session_checkpoint(
                    config,
                    generation,
                    &read.native_path(),
                    &read.envelope.native_manifest_sha256,
                    &NativeRestoreAdmission {
                        new_session_id: session_id.clone(),
                        bindings: self.profile.profile.bindings.clone(),
                        math: self.profile.profile.math.clone(),
                    },
                )?;
                read.envelope.target_seal.validate_restored(
                    &report["checkpoint"],
                    &read.descriptor,
                    session_id,
                )?;
                if report["device_pending_token_verified"] != true
                    || report["draft_rebuild_submitted_and_drained"] != true
                    || report["draft_rebuild_outputs_published"] != false
                    || report["prefix_recompute_tokens"] != 0
                {
                    return Err("native restore completion contract".into());
                }
                read.verify_namespace()?;
                let continuation_checkpoint = if profile == Profile::Resident {
                    owner.enable_resident_live_boundary(&report["checkpoint"])?
                } else { report["checkpoint"].clone() };
                let restored = Active {
                    id: session_id.clone(),
                    checkpoint: continuation_checkpoint,
                    consumed: read.descriptor.ledger.consumed.clone(),
                    turns: read.envelope.ledger.turns.clone(),
                };
                let result = json!({"schema":"qwen35-http-checkpoint-terminal-v1","event":"session_restored",
                    "request_id":request_id,"session_id":session_id,"revision":restored.checkpoint["revision"],
                    "owner_id":self.owner_id,"owner_epoch":self.epoch,"profile_sha256":self.profile_sha,
                    "checkpoint_id":checkpoint_id,"http_manifest_sha256":http_manifest_sha256,
                    "native_manifest_sha256":read.envelope.native_manifest_sha256,"native_restore":report,
                    "session":metadata(&restored),"source_owner_id":read.envelope.origin_owner_id,
                    "source_session_id":read.descriptor.ledger.origin_session_id,
                    "restored_prefix_tokens":restored.consumed.len(),"prefix_recompute_tokens":0,
                    "target_seal_verified":true,"client_delivery_acknowledged":false,
                    "restore_wall_ms_including_file_validation":start.elapsed().as_secs_f64()*1000.,
                    "qualified":false,"product_ready":false,"semantic_qualification":false,"visible_tps":false});
                session_ids.insert(session_id.clone());
                *active = Some(restored);
                Ok(Some(result))
            }
            _ => Err("not a checkpoint command".into()),
        }
    }
}

pub(super) fn published_turn<W: Write>(
    p: &Persistence,
    s: &CommittedStream<W>,
    tokens: &[u32],
    stop: &str,
    report_sha: &str,
    seal: &Value,
) -> Result<PublishedTurn, Box<dyn Error>> {
    Ok(PublishedTurn {
        owner_id: p.owner_id.clone(),
        request_id: s.request_id.clone(),
        request_sha256: s.request_sha256.clone(),
        native_report_sha256: report_sha.into(),
        native_revision: seal["revision"].as_u64().ok_or("published revision")?,
        terminal_sequence: s.sequence,
        token_ids: tokens.into(),
        visible_token_ids: s.visible_token_ids.clone(),
        text: s.text.clone(),
        text_sha256: digest(s.text.as_bytes()),
        stop_reason: match stop {
            EOS_STOP => StopReason::Eos,
            MAX_NEW_STOP => StopReason::MaxNewTokens,
            _ => return Err("published stop reason".into()),
        },
    })
}
