//! Native-only same-owner append/decode audit. No HTTP/product admission.
//! A visible final token is a pending input, not an already consumed KV row.
use super::*;
pub(super) mod suffix;
#[cfg(all(feature = "qwen35-session-checkpoint", windows))]
mod checkpoint_capture;
#[cfg(all(feature = "qwen35-session-checkpoint", windows))]
mod checkpoint_restore;
#[cfg(all(feature = "qwen35-session-checkpoint", windows))]
mod input_prefix;

/// Input checkpoints and generated sessions are different retained consumers.
/// In particular, an input checkpoint cannot manufacture a generated ledger.
pub(super) enum RetainedOutput<'a> {
    Generated(&'a str, &'a mut Option<CommittedSession>),
    #[cfg(all(feature = "qwen35-session-checkpoint", windows))]
    InputPrefix {
        tokens: &'a [u32],
        bindings: &'a crate::session_checkpoint::metadata::ModelBindings,
        destination: &'a mut Option<input_prefix::InputPrefixOwner>,
    },
}

impl RetainedOutput<'_> {
    pub(super) fn input_tokens(&self) -> Option<&[u32]> {
        match self {
            Self::Generated(..) => None,
            #[cfg(all(feature = "qwen35-session-checkpoint", windows))]
            Self::InputPrefix { tokens, .. } => Some(tokens),
        }
    }
}

#[cfg(all(feature = "qwen35-session-checkpoint", windows))]
pub(super) use input_prefix::InputPrefixOwner;

fn validate_committed_path(previous_anchor: u32, emitted: &[u32], actual: &[u32]) -> Result<()> {
    if emitted.is_empty()
        || emitted.len() > B
        || previous_anchor as usize >= V
        || emitted.iter().any(|&t| t as usize >= V)
        || actual.len() != emitted.len()
        || actual.first() != Some(&previous_anchor)
        || actual[1..] != emitted[..emitted.len() - 1]
    {
        return Err(invalid("session device consumed-input path mismatch"));
    }
    Ok(())
}

pub(super) fn fast_session_path_switch(value: Option<&std::ffi::OsStr>) -> Result<bool> {
    match value {
        None => Ok(false),
        Some(v) if v == "1" => Ok(true),
        _ => Err(invalid(
            "fast session path requires AUGUR_QWEN35_FAST_SESSION_PATH=1",
        )),
    }
}

pub(super) fn fast_session_path_active(
    requested: bool,
    live_boundary: bool,
    explicit_audit: bool,
) -> bool {
    requested && live_boundary && !explicit_audit
}

#[derive(Debug, Eq, PartialEq)]
struct LiveContinuationPolicy {
    fast_session_path: bool,
    fast_feature_identity: bool,
    replay_head_elision: bool,
}

fn rebind_live_projection_executor<T>(
    mut executor: Option<T>,
    required: bool,
    policy_changed: bool,
    rebind: impl FnOnce(&mut T) -> Result<()>,
) -> Result<Option<T>> {
    if required && executor.is_none() {
        return Err(invalid("live projection policy requires canonical executor"));
    }
    if policy_changed {
        if let Some(executor) = &mut executor {
            rebind(executor)?;
        }
    }
    Ok(executor)
}

impl LiveContinuationPolicy {
    fn from_switches(
        session: Option<&std::ffi::OsStr>,
        features: Option<&std::ffi::OsStr>,
        replay_head: Option<&std::ffi::OsStr>,
    ) -> Result<Self> {
        Ok(Self {
            fast_session_path: fast_session_path_switch(session)?,
            fast_feature_identity: fast_feature_identity_switch(features)?,
            replay_head_elision: replay_head_elision_switch(replay_head)?,
        })
    }
}

fn validate_committed_transaction(
    cursor: usize,
    generation: u64,
    anchor: u32,
    transaction: &Value,
) -> Result<Vec<u32>> {
    let emitted: Vec<u32> = serde_json::from_value(transaction["emitted_token_ids"].clone())
        .map_err(|e| invalid(e.to_string()))?;
    if emitted.is_empty()
        || emitted.len() > B
        || emitted.iter().any(|&token| token as usize >= V)
        || transaction["committed_token_count"].as_u64() != Some(emitted.len() as u64)
        || transaction["cursor_after"].as_u64() != Some(cursor as u64)
        || transaction["generation_after"].as_u64() != Some(generation)
        || transaction["cursor_before"].as_u64()
            .and_then(|before| before.checked_add(emitted.len() as u64))
            != Some(cursor as u64)
        || transaction["generation_before"].as_u64().and_then(|before| before.checked_add(1))
            != Some(generation)
        || transaction["next_anchor"].as_u64() != Some(u64::from(anchor))
        || emitted.last().copied() != Some(anchor)
    {
        return Err(invalid("session device transaction extent"));
    }
    Ok(emitted)
}

pub(super) fn check_committed_path(
    d: &Driver,
    transaction: &mut Value,
    previous_anchor: u32,
    allow_fast_path: bool,
) -> Result<bool> {
    if allow_fast_path {
        validate_committed_transaction(d.cursor, d.generation, d.anchor, transaction)?;
        if previous_anchor as usize >= V {
            return Err(invalid("session previous anchor range"));
        }
        // The removed path readback also drained the commit/replay tail. Keep
        // that completion guarantee before publishing tokens or host ledger:
        // the earlier acceptance receipt covers verification, not replay.
        let receipt = d.stream.submission_receipt()?;
        let proof = d.stream.synchronize_submission(receipt)?;
        if !proof.covers_submission(receipt) || !proof.belongs_to_stream(&d.stream)? {
            return Err(invalid("session commit completion owner epoch"));
        }
        transaction["session_input_path_audited"] = json!(false);
        transaction["session_input_path_policy"] = json!("metadata_only_fast_v1");
        transaction["session_input_path_d2h_bytes"] = json!(0);
        transaction["session_commit_completion_verified"] = json!(true);
        return Ok(false);
    }
    audit_committed_path(d, transaction, previous_anchor)?;
    transaction["session_input_path_audited"] = json!(true);
    transaction["session_input_path_policy"] = json!("device_readback_v1");
    transaction["session_input_path_d2h_bytes"] =
        json!(transaction["committed_token_count"].as_u64().unwrap() * 4);
    transaction["session_commit_completion_verified"] = json!(true);
    Ok(true)
}

pub(super) fn audit_committed_path(
    d: &Driver,
    transaction: &Value,
    previous_anchor: u32,
) -> Result<()> {
    let emitted = validate_committed_transaction(
        d.cursor, d.generation, d.anchor, transaction,
    )?;
    validate_committed_path(
        previous_anchor,
        &emitted,
        &words(&d.stream, &d.path, 0, emitted.len())?,
    )
}

fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

pub(super) fn token_hash(tokens: &[u32]) -> String {
    let mut h = Sha256::new();
    h.update(b"QWEN35-SESSION-CONSUMED-U32LE-V1\0");
    for token in tokens {
        h.update(token.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Ledger {
    session_id: String,
    consumed: Vec<u32>,
    // These are model outputs, never teacher-forced suffix inputs.
    generated: Vec<u32>,
    suffix_ranges: Vec<(usize, usize)>,
    pending: u32,
    pending_published: bool,
    generation: u64,
    revision: u64,
    capacity: usize,
}

impl Ledger {
    fn initial(
        id: &str,
        prompt: &[u32],
        output: &[u32],
        cursor: usize,
        generation: u64,
        capacity: usize,
    ) -> Result<Self> {
        if !valid_id(id)
            || prompt.is_empty()
            || output.is_empty()
            || generation == 0
            || prompt.iter().chain(output).any(|&t| t as usize >= V)
            || prompt.len().checked_add(output.len() - 1) != Some(cursor)
            || cursor.checked_add(B).is_none_or(|n| n > capacity)
        {
            return Err(invalid("session initial input/output ledger mismatch"));
        }
        let mut consumed = Vec::with_capacity(capacity);
        consumed.extend_from_slice(prompt);
        consumed.extend_from_slice(&output[..output.len() - 1]);
        Ok(Self {
            session_id: id.into(),
            consumed,
            generated: output.to_vec(),
            suffix_ranges: Vec::new(),
            pending: *output.last().unwrap(),
            pending_published: true,
            generation,
            revision: 1,
            capacity,
        })
    }

    fn preflight(&self, exact_history: &[u32], suffix: &[u32]) -> Result<Vec<u32>> {
        if exact_history != self.consumed
            || suffix.is_empty()
            || suffix.len() > 2048
            || suffix.iter().any(|&t| t as usize >= V)
        {
            return Err(invalid(
                "session exact history/suffix mismatch before submit",
            ));
        }
        let count = suffix
            .len()
            .checked_add(usize::from(self.pending_published))
            .ok_or_else(|| invalid("session suffix overflow"))?;
        if self
            .consumed
            .len()
            .checked_add(count)
            .and_then(|n| n.checked_add(B))
            .is_none_or(|n| n > self.capacity)
            || self.generation == u64::MAX
            || self.revision == u64::MAX
        {
            return Err(invalid(
                "session capacity/generation admission before submit",
            ));
        }
        let mut inputs = Vec::with_capacity(count);
        if self.pending_published {
            inputs.push(self.pending);
        }
        inputs.extend_from_slice(suffix);
        Ok(inputs)
    }

    fn publish_append(
        &mut self,
        inputs: &[u32],
        cursor: usize,
        generation: u64,
        anchor: u32,
    ) -> Result<()> {
        if inputs.is_empty()
            || inputs.iter().any(|&t| t as usize >= V)
            || anchor as usize >= V
            || self.consumed.len().checked_add(inputs.len()) != Some(cursor)
            || self.generation.checked_add(1) != Some(generation)
            || (self.pending_published && inputs.first() != Some(&self.pending))
        {
            return Err(invalid("session append publication drift"));
        }
        let suffix_start = self.consumed.len() + usize::from(self.pending_published);
        self.consumed.extend_from_slice(inputs);
        self.suffix_ranges.push((suffix_start, cursor));
        self.pending = anchor;
        // This is a fresh target prediction, NOT a published model output.
        self.pending_published = false;
        self.generation = generation;
        self.revision = self
            .revision
            .checked_add(1)
            .ok_or_else(|| invalid("session revision"))?;
        Ok(())
    }

    fn commit_generated(&mut self, emitted: &[u32], cursor: usize, generation: u64) -> Result<()> {
        if !self.pending_published
            || emitted.is_empty()
            || emitted.len() > B
            || emitted.iter().any(|&id| id as usize >= V)
            || self.consumed.len().checked_add(emitted.len()) != Some(cursor)
            || self.generation.checked_add(1) != Some(generation)
        {
            return Err(invalid("session generated commit ledger mismatch"));
        }
        self.consumed.push(self.pending);
        self.consumed
            .extend_from_slice(&emitted[..emitted.len() - 1]);
        self.generated.extend_from_slice(emitted);
        self.pending = *emitted.last().unwrap();
        self.generation = generation;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum NativeAuditProfile {
    ResidentLong128,
    ResidentStream128,
    ResidentStreamBudget,
    Growing32,
    GrowingLong128,
    GrowingStream128,
    Short32,
    Long128,
    Stream128,
    Context32Long128,
    Context32StreamLong128,
    Context64Long128,
    Context64StreamLong128,
    Near192Long128,
    Near192StreamLong128,
}
impl NativeAuditProfile {
    fn validate(self, max_tokens: usize, private_capture: bool) -> Result<()> {
        let valid = match self {
            Self::ResidentLong128 | Self::ResidentStream128 => max_tokens == 128 && !private_capture,
            Self::ResidentStreamBudget => (1..=crate::resident_session_profile::RESIDENT_CONTINUATION_MAX_OUTPUT)
                .contains(&max_tokens) && !private_capture,
            Self::Growing32 => (2..=32).contains(&max_tokens) && !private_capture,
            Self::GrowingLong128 | Self::GrowingStream128 => max_tokens == 128 && !private_capture,
            Self::Short32 => (2..=32).contains(&max_tokens),
            Self::Stream128 => max_tokens == 128 && !private_capture,
            Self::Context32Long128 => {
                max_tokens == 128
                    && private_capture
                    && cfg!(feature = "qwen35-session-context32-audit")
            }
            Self::Context32StreamLong128 => {
                max_tokens == 128
                    && !private_capture
                    && cfg!(feature = "qwen35-session-context32-audit")
            }
            Self::Context64Long128 => {
                max_tokens == 128
                    && private_capture
                    && cfg!(feature = "qwen35-session-context64-audit")
            }
            Self::Context64StreamLong128 => {
                max_tokens == 128
                    && !private_capture
                    && cfg!(feature = "qwen35-session-context64-audit")
            }
            Self::Near192Long128 => {
                max_tokens == 128
                    && private_capture
                    && cfg!(feature = "qwen35-session-near192-audit")
            }
            Self::Near192StreamLong128 => {
                max_tokens == 128
                    && !private_capture
                    && cfg!(feature = "qwen35-session-near192-audit")
            }
            Self::Long128 => {
                max_tokens == 128 && private_capture && cfg!(feature = "qwen35-draft-private-state")
            }
        };
        if !valid {
            return Err(invalid("explicit native audit profile token/capture bound"));
        }
        Ok(())
    }
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::ResidentLong128 => "resident_text_reference128_v1",
            Self::ResidentStream128 => "resident_text_stream128_v1",
            Self::ResidentStreamBudget => "resident_text_stream_budget_v1",
            Self::Growing32 => "native_growing32_v1",
            Self::GrowingLong128 => "native_growing_long128_v1",
            Self::GrowingStream128 => "native_growing_stream128_v1",
            Self::Short32 => "native_short32_v1",
            Self::Long128 => "native_long128_v1",
            Self::Stream128 => "native_stream128_v1",
            Self::Context32Long128 | Self::Context32StreamLong128 => "native_context32_long128_v1",
            Self::Context64Long128 | Self::Context64StreamLong128 => "native_context64_long128_v1",
            Self::Near192Long128 | Self::Near192StreamLong128 => "native_near192_long128_v1",
        }
    }
    pub(super) fn is_growing(self) -> bool {
        matches!(self, Self::ResidentLong128 | Self::ResidentStream128 | Self::ResidentStreamBudget
            | Self::Growing32 | Self::GrowingLong128 | Self::GrowingStream128)
    }
    fn requires_sink(self) -> bool {
        matches!(
            self,
            Self::Stream128
                | Self::ResidentStream128
                | Self::ResidentStreamBudget
                | Self::GrowingStream128
                | Self::Context32StreamLong128
                | Self::Context64StreamLong128
                | Self::Near192StreamLong128
        )
    }
    fn validate_consumer(self, growing_history: bool, sink: bool) -> Result<()> {
        if self.is_growing() != growing_history || self.requires_sink() != sink {
            return Err(invalid("explicit initialized-history and sink-bound native profile required"));
        }
        Ok(())
    }
}

pub(super) struct FreshReference<'a> {
    pub token_ids: &'a [u32],
    pub max_new_tokens: usize,
    pub audit_directory: Option<&'a Path>,
    pub draft_audit_directory: Option<&'a Path>,
    pub profile: NativeAuditProfile,
}
impl FreshReference<'_> {
    pub(super) fn validate(&self) -> Result<()> {
        self.profile
            .validate(self.max_new_tokens, self.draft_audit_directory.is_some())?;
        // One 8K turn, <=1024 outputs + <=2048 new input tokens. Not HTTP admission.
        let admitted_extent = match self.profile {
            NativeAuditProfile::ResidentLong128 => (1..=crate::resident_session_profile::CAPACITY
                - crate::resident_session_profile::CONTINUATION_RESERVE).contains(&self.token_ids.len()),
            NativeAuditProfile::Growing32 | NativeAuditProfile::GrowingLong128
                | NativeAuditProfile::GrowingStream128 => (1..=16384).contains(&self.token_ids.len()),
            NativeAuditProfile::Context32Long128 => {
                cfg!(feature = "qwen35-session-context32-audit")
                    && (32769..=35840).contains(&self.token_ids.len())
            }
            NativeAuditProfile::Context64Long128 => {
                cfg!(feature = "qwen35-session-context64-audit")
                    && (65537..=69632).contains(&self.token_ids.len())
            }
            NativeAuditProfile::Near192Long128 => {
                // 128 outputs reserve128+B-2 input positions, not just128.
                cfg!(feature = "qwen35-session-near192-audit")
                    && (194561..=196474).contains(&self.token_ids.len())
            }
            _ => (8193..=11264).contains(&self.token_ids.len()),
        };
        if self.profile.requires_sink()
            || !admitted_extent
            || self.token_ids.iter().any(|&t| t as usize >= V)
        {
            return Err(invalid("bounded native exact-history reference"));
        }
        Ok(())
    }
}

pub(super) fn audit_target_state(d: &mut Driver) -> Result<Value> {
    audit_target_state_to(d, None)
}

pub(super) fn audit_target_state_to(d: &mut Driver, directory: Option<&Path>) -> Result<Value> {
    use std::io::Write;
    if let Some(path) = directory {
        // Exclusive evidence only; a partial dump is never a checkpoint.
        std::fs::create_dir(path).map_err(|e| invalid(e.to_string()))?;
    }
    let open = |name: &str| -> Result<Option<std::fs::File>> {
        directory
            .map(|path| {
                std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(path.join(name))
                    .map_err(|e| invalid(e.to_string()))
            })
            .transpose()
    };
    if d.poisoned || d.cursor < d.context_rows || d.cursor > d.capacity {
        return Err(invalid("target state audit owner/extent"));
    }
    if let Some(history) = d.growing_history {
        if history.cursor() != d.cursor || history.capacity_rows() != 2048
            || history.active_rows() != d.cursor.min(2048)
            || d.features.len_bytes() != history.capacity_rows() * 5 * H * 4 {
            return Err(invalid("growing state audit initialized/physical extent"));
        }
    }
    d.full_context_feature_digest()?;
    d.stream.synchronize()?;
    let proof = d.stream.submission_receipt()?;
    // Deliberately diagnostic: SHA every initialized committed KV/GDN/conv
    // byte, plus the feature ring. Never hash unwritten capacity/tail rows.
    // Same full-byte/hash contract; three bounded host buffers overlap D2H,
    // per-plane hashing and global hashing without moving CUDA onto a worker.
    // This work is still in user request wall time, not kernel decode TPS.
    let chunk_bytes = 1024 * 1024;
    let mut domain = b"QWEN35-SESSION-TARGET-STATE-V1\0".to_vec();
    domain.extend_from_slice(&(d.cursor as u64).to_le_bytes());
    let mut bytes = 0usize;
    let mut planes = 0usize;
    let mut plane_hashes = Vec::new();
    let mut buffers = Vec::new();
    let mut extents = Vec::new();
    for (layer, plan) in d.plans.iter().enumerate() {
        for (plane, buffer) in plan.terminal_state_buffers().iter().enumerate() {
            let len = if matches!(plan, DevicePlan::Full(_)) {
                if buffer.len_bytes() != d.capacity * QWEN35_KV8_CACHE_ROW_BYTES {
                    return Err(invalid("session committed KV geometry"));
                }
                d.cursor * QWEN35_KV8_CACHE_ROW_BYTES
            } else {
                buffer.len_bytes()
            };
            extents.push(crate::session_seal_hash::Plane {
                header: [(layer as u64).to_le_bytes(), (plane as u64).to_le_bytes(),
                    (len as u64).to_le_bytes()].concat(),
                bytes: len,
            });
            buffers.push(*buffer);
            let file_name = format!("layer-{layer:02}-plane-{plane}.bin");
            plane_hashes.push(json!({"layer":layer,"plane":plane,"bytes":len,
                    "file_name":file_name,"dtype":if matches!(plan,DevicePlan::Full(_)){"q8_0_34bytes"}else{"f32_le"},
                    "kind":if matches!(plan, DevicePlan::Full(_)) {"full_kv8"} else {"gdn_conv"}}));
            planes += 1;
        }
    }
    let mut file = None;
    let hashed = crate::session_seal_hash::run(&domain, &extents, chunk_bytes, |index, offset, staging| {
        if offset == 0 {
            file = open(plane_hashes[index]["file_name"].as_str()
                .ok_or_else(|| "session state plane filename".to_owned())?)
                .map_err(|e| e.to_string())?;
        }
        let read = buffers[index].copy_range_to_host_sync_with_submission_proof(
            &d.stream, offset, staging, proof).map_err(|e| e.to_string())?;
        if !read.covers_submission(proof) || !read.belongs_to_stream(&d.stream).map_err(|e| e.to_string())? {
            return Err("session target state completion proof".to_owned());
        }
        if let Some(f) = file.as_mut() {
            f.write_all(staging).map_err(|e| e.to_string())?;
            if offset + staging.len() == extents[index].bytes {
                f.sync_all().map_err(|e| e.to_string())?;
            }
        }
        bytes += staging.len();
        Ok(())
    }).map_err(invalid)?;
    let hex = |value: &[u8]| value.iter().map(|b| format!("{b:02x}")).collect::<String>();
    let target_sha = hex(&hashed.global);
    for (plane, hash) in plane_hashes.iter_mut().zip(&hashed.planes) {
        plane["sha256"] = json!(hex(hash));
    }
    drop(file);
    let mut staging = vec![0u8; chunk_bytes];
    let mut feature = Sha256::new();
    let mut feature_file = open("features.bin")?;
    let initialized_feature_bytes = d.active_feature_view()?.len_bytes();
    for offset in (0..initialized_feature_bytes).step_by(staging.len()) {
        let count = staging.len().min(initialized_feature_bytes - offset);
        let read = d.features.copy_range_to_host_sync_with_submission_proof(
            &d.stream,
            offset,
            &mut staging[..count],
            proof,
        )?;
        if !read.covers_submission(proof) || !read.belongs_to_stream(&d.stream)? {
            return Err(invalid("session feature completion proof"));
        }
        feature.update(&staging[..count]);
        if let Some(f) = feature_file.as_mut() {
            f.write_all(&staging[..count])
                .map_err(|e| invalid(e.to_string()))?;
        }
        bytes += count;
    }
    if let Some(f) = feature_file.as_mut() {
        f.sync_all().map_err(|e| invalid(e.to_string()))?;
    }
    let mut report = json!({"cursor":d.cursor,"capacity":d.capacity,"generation":d.generation,
        "pending_token":d.anchor,"target_committed_state_sha256":target_sha,
        "feature_bytes_sha256":format!("{:x}",feature.finalize()),
        "target_planes":planes,"state_audit_d2h_bytes":bytes,
        "feature_bytes":initialized_feature_bytes,"state_plane_hashes":plane_hashes,
        "target_completion":format!("{proof:?}")});
    if let Some(history) = d.growing_history {
        report["initialized_history"] = json!({"schema":"qwen35-initialized-session-history-v1",
            "start_position":history.start_position(),"active_rows":history.active_rows(),
            "physical_capacity_rows":history.capacity_rows(),"uninitialized_tail_hashed":false});
    }
    if let Some(mut manifest) = open("manifest.json")? {
        manifest
            .write_all(&serde_json::to_vec_pretty(&report).map_err(|e| invalid(e.to_string()))?)
            .map_err(|e| invalid(e.to_string()))?;
        manifest.sync_all().map_err(|e| invalid(e.to_string()))?;
    }
    Ok(report)
}

fn initial_transaction_path_valid(
    prompt_len: usize,
    output_tokens: &[u32],
    cursor: usize,
    generation: u64,
    transactions: &[Value],
    output: &Value,
    allow_fast_session_path: bool,
) -> bool {
    if transactions.is_empty() {
        // An anchor-only terminal has no verify/rollback transaction to audit.
        // It must retain the exact prefill cursor and fresh driver generation;
        // the published anchor remains a pending input, not an invented KV row.
        output_tokens.len() == 1
            && cursor == prompt_len
            && generation == 1
            && match output["stop_reason"].as_str() {
                Some("max_new_tokens") => output["max_new_tokens"].as_u64() == Some(1),
                Some("eos") => output["eos_token_ids"].as_array().is_some_and(|eos| {
                    eos.iter().any(|token| token.as_u64() == Some(u64::from(output_tokens[0])))
                }),
                _ => false,
            }
    } else {
        transactions.iter().all(|t| {
            t["session_input_path_audited"] == true
                || (allow_fast_session_path
                    && t["session_input_path_audited"] == false
                    && t["session_input_path_policy"] == "metadata_only_fast_v1")
        })
    }
}

fn rebind_driver_draft(
    d: &mut Driver, draft: &mut DFlashQ8CudaBlock,
    committed: Option<usize>, growing_base: Option<GrowingHistory>,
) -> Result<()> {
    d.next_context_feature_digest()?;
    let mut tokens = vec![crate::DFLASH_SEED_MASK_TOKEN; B];
    tokens[0] = d.anchor;
    let seed = DFlashDraftBlockInput {
        request_generation: d.generation,
        absolute_position_start: d.cursor,
        anchor_token: d.anchor,
        block_tokens: tokens,
        mask_token_id: crate::DFLASH_SEED_MASK_TOKEN,
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
    let noise = d.prepare_noise_device(view(&d.token)?)?;
    d.target_handoff_event
        .record_for_submission(&d.stream, d.stream.submission_receipt()?)?;
    draft.stream().wait_event(&d.target_handoff_event)?;
    if let Some(history) = d.growing_history {
        let features = d.active_feature_view()?;
        return if let Some(count) = committed {
            let plan = d.last_growth_plan.ok_or_else(|| invalid("missing session growth plan"))?;
            if plan.append_rows != count {
                return Err(invalid("session commit/growth plan mismatch"));
            }
            draft.rebind_growing_committed_device_features_with_noise(
                &seed, &capture, features, noise, plan, history).map_err(invalid)
        } else if let Some(base) = growing_base {
            match GrowingHistoryPlan::between(base, history) {
                Ok(plan) => draft
                    .rebind_growing_committed_device_features_with_noise(
                        &seed, &capture, features, noise, plan, history,
                    )
                    .map_err(invalid),
                // A long sequence of input-only appends can exceed one
                // physical ring. In that case the old rows are no longer
                // available for a bounded proof, so retain the exact full
                // refresh path rather than guessing a partial overlap.
                Err(_) => draft
                    .refresh_growing_device_features_with_noise(
                        &seed, &capture, features, noise, history,
                    )
                    .map_err(invalid),
            }
        } else {
            draft.refresh_growing_device_features_with_noise(
                &seed, &capture, features, noise, history).map_err(invalid)
        };
    }
    if let Some(count) = committed {
        draft
            .rebind_committed_device_features_with_noise(
                &seed,
                &capture,
                view(&d.features)?,
                noise,
                count,
            )
            .map_err(invalid)
    } else {
        // Across a user suffix, regenerate the draft's derived projections
        // from the retained target feature ring, without reloading weights.
        draft
            .rebind_device_features_with_noise(
                &seed,
                &capture,
                view(&d.features)?,
                noise,
                d.cursor - d.context_rows,
                2048,
            )
            .map_err(invalid)
    }
}

pub(super) struct CommittedSession {
    // Both streams are drained in Drop before any pinned storage is released.
    draft: DFlashQ8CudaBlock,
    driver: Driver,
    ledger: Ledger,
    eos: Vec<u32>,
    // Keeps the draft's known base extent while one or more user suffixes
    // advance the target ring. Continuation can then import just the final
    // retained/new rows once, instead of rebuilding all 2048 projections.
    draft_growth_base: Option<GrowingHistory>,
    // Live continuation authenticates this in-memory owner/ledger, not saved payload bytes.
    live_boundaries: bool,
    fast_session_path: bool,
    pub(super) checkpoint: Value,
}

impl Drop for CommittedSession {
    fn drop(&mut self) {
        let _ = self.draft.synchronize();
        let _ = self.driver.stream.synchronize();
    }
}

impl CommittedSession {
    pub(super) fn seal_initial(
        driver: Driver,
        draft: DFlashQ8CudaBlock,
        id: &str,
        output: &Value,
        transactions: &[Value],
        live_boundaries: bool,
        fast_session_path: bool,
    ) -> Result<Self> {
        let prompt: Vec<u32> = serde_json::from_value(output["prompt_token_ids"].clone())
            .map_err(|e| invalid(e.to_string()))?;
        let initial_anchor = output["initial_anchor"]
            .as_u64()
            .ok_or_else(|| invalid("session initial anchor missing"))?
            as u32;
        let all_tokens: Vec<u32> = serde_json::from_value(output["token_ids"].clone())
            .map_err(|e| invalid(e.to_string()))?;
        if !matches!(
            output["stop_reason"].as_str(),
            Some("eos" | "max_new_tokens")
        ) || driver.poisoned
            || !initial_transaction_path_valid(
                prompt.len(), &all_tokens, driver.cursor, driver.generation, transactions, output,
                live_boundaries && fast_session_path,
            )
            || all_tokens.first().copied() != Some(initial_anchor)
            || driver.anchor
                != *all_tokens
                    .last()
                    .ok_or_else(|| invalid("session no output"))?
        {
            return Err(invalid("session requires healthy complete generated turn"));
        }
        let ledger = Ledger::initial(
            id,
            &prompt,
            &all_tokens,
            driver.cursor,
            driver.generation,
            driver.capacity,
        )?;
        let mut session = Self {
            driver,
            draft,
            ledger,
            eos: serde_json::from_value(output["eos_token_ids"].clone())
                .map_err(|e| invalid(e.to_string()))?,
            draft_growth_base: None,
            checkpoint: Value::Null,
            live_boundaries,
            fast_session_path: live_boundaries && fast_session_path,
        };
        session.checkpoint = session.seal_for_continuation()?;
        Ok(session)
    }

    fn seal_policy(&self) -> &'static str {
        if self.live_boundaries { "live_owner_no_payload_readback_v1" }
        else { "full_bytes_pipelined_dual_sha256_v1" }
    }

    fn seal_for_continuation(&mut self) -> Result<Value> {
        if self.live_boundaries { self.seal_live_boundary() } else { self.seal() }
    }

    fn seal_live_boundary(&mut self) -> Result<Value> {
        let d = &self.driver;
        if d.poisoned || d.cursor == 0 || d.cursor > d.capacity || d.plans.len() != 64
            || d.cursor != self.ledger.consumed.len()
            || d.generation != self.ledger.generation || d.anchor != self.ledger.pending {
            return Err(invalid("live boundary owner/ledger mismatch"));
        }
        let history = d.growing_history.ok_or_else(|| invalid("live boundary requires initialized history"))?;
        if history.cursor() != d.cursor || history.capacity_rows() != 2048
            || history.active_rows() != d.cursor.min(2048)
            || d.features.len_bytes() != 2048 * 5 * H * 4 {
            return Err(invalid("live boundary initialized history extent"));
        }
        // Completion/poison/ownership stay mandatory. No checkpoint hash can replace them.
        self.draft.synchronize().map_err(invalid)?;
        d.stream.synchronize()?;
        let owner_epoch = d.stream.submission_receipt()?;
        let mut boundary = json!({
            "schema":"qwen35-native-live-session-boundary-v1",
            "session_id":self.ledger.session_id,"revision":self.ledger.revision,
            "generation":d.generation,"cursor":d.cursor,"capacity":d.capacity,
            "request_identity":sha256_hex(&d.identity),
            "consumed_input_digest":token_hash(&self.ledger.consumed),
            "pending_token":d.anchor,"pending_token_published":self.ledger.pending_published,
            "generated_token_count":self.ledger.generated.len(),"user_suffix_ranges":self.ledger.suffix_ranges,
            "live_owner_submission":format!("{owner_epoch:?}"),"target_stream_drained":true,
            "draft_stream_drained":true,"draft_state_retained":true,"draft_rebind_required":true,
            "state_seal_policy":"live_owner_no_payload_readback_v1",
            "state_audit_d2h_bytes":0,"payload_checksum_verified":false,"persistable_state_seal":false,
            "initialized_history":{"schema":"qwen35-initialized-session-history-v1",
                "start_position":history.start_position(),"active_rows":history.active_rows(),
                "physical_capacity_rows":2048,"uninitialized_tail_hashed":false},
            "product_ready":false
        });
        boundary["checkpoint_sha256"] = json!(sha256_hex(
            &serde_json::to_vec(&boundary).map_err(|e| invalid(e.to_string()))?));
        Ok(boundary)
    }

    // An explicit payload witness for save/restore and numerical audits, never
    // implied by the live boundary's metadata digest.
    fn seal(&mut self) -> Result<Value> {
        let d = &mut self.driver;
        if d.poisoned
            || d.cursor != self.ledger.consumed.len()
            || d.generation != self.ledger.generation
            || d.anchor != self.ledger.pending
        {
            return Err(invalid("session seal owner/ledger mismatch"));
        }
        self.draft.synchronize().map_err(invalid)?;
        let audit = audit_target_state(d)?;
        let mut checkpoint = json!({
            "schema":"qwen35-native-session-target-checkpoint-v1",
            "session_id":self.ledger.session_id,"revision":self.ledger.revision,
            "generation":d.generation,"cursor":d.cursor,"capacity":d.capacity,
            "request_identity":sha256_hex(&d.identity),
            "consumed_input_digest":token_hash(&self.ledger.consumed),
            "pending_token":d.anchor,"pending_token_published":self.ledger.pending_published,
            "generated_token_count":self.ledger.generated.len(),
            "user_suffix_ranges":self.ledger.suffix_ranges,
            "target_committed_state_sha256":audit["target_committed_state_sha256"],
            "feature_bytes_sha256":audit["feature_bytes_sha256"],
            "target_planes":audit["target_planes"],"state_audit_d2h_bytes":audit["state_audit_d2h_bytes"],
            "state_plane_hashes":audit["state_plane_hashes"],
            "target_completion":audit["target_completion"],"draft_stream_drained":true,
            "draft_state_retained":true,"draft_rebind_required":true,
            "draft_state_equivalence_qualified":false,
            "http_continuation_enabled":false,"product_ready":false
        });
        if self.driver.growing_history.is_some() {
            checkpoint["schema"] = json!("qwen35-native-growing-session-target-checkpoint-v1");
            checkpoint["initialized_history"] = audit["initialized_history"].clone();
        }
        checkpoint["checkpoint_sha256"] = json!(sha256_hex(
            &serde_json::to_vec(&checkpoint).map_err(|e| invalid(e.to_string()))?
        ));
        Ok(checkpoint)
    }

    fn append(&mut self, expected: &Value, exact_history: &[u32], suffix: &[u32]) -> Result<Value> {
        if self.driver.poisoned || expected != &self.checkpoint {
            return Err(invalid(
                "session stale/foreign/corrupt checkpoint before submit",
            ));
        }
        let inputs = self.ledger.preflight(exact_history, suffix)?;
        let plan = suffix::Plan::new(self.driver.batched_suffix, self.driver.cursor,
            inputs.len(), self.driver.capacity)?;
        if plan.batch_rows > 0 {
            self.driver.batch_verify_execution.as_ref()
                .ok_or_else(|| invalid("suffix canonical executor missing before submit"))?
                .validate_suffix(&self.driver, plan.batch_rows)?;
        }
        // Input-ledger storage is reserved before submit. A later report/audit
        // failure must still poison: this is not an allocation-free transaction.
        self.ledger
            .suffix_ranges
            .try_reserve(1)
            .map_err(|e| invalid(e.to_string()))?;
        let start = Instant::now();
        self.driver.poisoned = true;
        let result = (|| {
            if self.draft_growth_base.is_none() {
                if let Some(history) = self.driver.growing_history {
                    if let Some(plan) = self.driver.last_growth_plan {
                        // A successful decode can leave one just-committed target
                        // block ahead of the draft cache. Align that bounded tail
                        // before treating the current target ring as our base.
                        self.rebind_draft(Some(plan.append_rows), None)?;
                        // The rebind reads target feature/noise buffers on the
                        // draft stream. Order their subsequent target overwrite
                        // after that read, without adding a host synchronization.
                        let receipt = self.draft.stream().submission_receipt()?;
                        self.driver.producer_event
                            .record_for_submission(self.draft.stream(), receipt)?;
                        self.driver.stream.wait_event(&self.driver.producer_event)?;
                        self.driver.last_growth_plan = None;
                    }
                    // A fresh/restored draft can already match the target and
                    // have no pending generated block. Keep that exact base
                    // too: repeated anchor-only turns never rebind the draft,
                    // so the next append's last input plan is not its base.
                    self.draft_growth_base = Some(history);
                }
            }
            let d = &mut self.driver;
            for block in inputs[..plan.batch_rows].chunks_exact(B) {
                let block: &[u32; B] = block.try_into().map_err(|_| invalid("suffix B8 extent"))?;
                d.batch_verify_execution.as_ref()
                    .ok_or_else(|| invalid("suffix executor ownership"))?.submit_suffix(d, block)?;
                d.promote_verified_features(B)?;
                d.cursor += B;
            }
            for &token in &inputs[plan.batch_rows..] {
                let advance = d.advance_feature_window(1)?;
                d.token
                    .copy_from_host_sync(&d.stream, &token.to_le_bytes())?;
                d.bind_gather(view(&d.token)?)?.submit_for_plan(&d.stream)?;
                d.forward(d.cursor, advance.capture_row(0))?;
                d.cursor += 1;
            }
            if d.cursor != plan.end { return Err(invalid("suffix final cursor mismatch")); }
            if plan.tail_rows == 0 { d.head.submit(&d.stream)?; }
            d.top1(view(&d.token)?)?;
            d.anchor = words(&d.stream, &d.token, 0, 1)?[0];
            d.generation += 1; // preflight proved no overflow
            d.stream.synchronize()?;
            self.ledger
                .publish_append(&inputs, d.cursor, d.generation, d.anchor)?;
            d.poisoned = false;
            let append_wall_ms = start.elapsed().as_secs_f64() * 1000.0;
            let seal_start = Instant::now();
            let checkpoint = self.seal_for_continuation()?;
            let seal_wall_ms = seal_start.elapsed().as_secs_f64() * 1000.0;
            self.checkpoint = checkpoint.clone();
            let mut report = json!({"schema":"qwen35-native-session-append-v1",
                "status":"PASS_NATIVE_TARGET_SUFFIX_APPEND_ONLY",
                "input_tokens_executed":inputs.len(),"user_suffix_tokens":suffix.len(),
                "pending_published_token_consumed":inputs.len()-suffix.len(),
                "prefix_tokens_recomputed":0,"model_weights_reloaded":false,
                "append_wall_ms_excluding_seal_audit":append_wall_ms,
                "state_seal_wall_ms":seal_wall_ms,"state_seal_policy":self.seal_policy(),
                "new_visible_tokens":[],"checkpoint":checkpoint,
                "decode_continuation_executed":false,"product_ready":false});
            if self.driver.batched_suffix {
                report["known_input_execution"] = json!({
                    "schema":"qwen35-canonical-batched-suffix-v1",
                    "policy":"retained_canonical_b8_real_input_scalar_tail_v1",
                    "batch_width":B,"batch_count":plan.batch_rows/B,
                    "serial_tail_rows":plan.tail_rows,"padding_tokens":0,
                    "generated_warmup_tokens":0,"speculative_acceptance_executed":false,
                    "output_head_calls":plan.tail_rows.max(1),
                    "additional_device_workspace_bytes":0,"publication_enabled":false});
            }
            Ok(report)
        })();
        if result.is_err() {
            self.driver.poisoned = true;
            let _ = self.driver.stream.synchronize();
            let _ = self.draft.synchronize();
        }
        result
    }

    fn rebind_draft(
        &mut self,
        committed: Option<usize>,
        growing_base: Option<GrowingHistory>,
    ) -> Result<()> {
        rebind_driver_draft(&mut self.driver, &mut self.draft, committed, growing_base)
    }

    fn decode_audit(
        &mut self,
        expected: &Value,
        max_tokens: usize,
        private_directory: Option<&Path>,
        profile: NativeAuditProfile,
        sink: Option<&Qwen35VerifiedCommitSink<'_>>,
    ) -> Result<Value> {
        profile.validate(max_tokens, private_directory.is_some())?;
        profile.validate_consumer(self.driver.growing_history.is_some(), sink.is_some())?;
        if self.live_boundaries && !matches!(profile,
            NativeAuditProfile::ResidentStream128 | NativeAuditProfile::ResidentStreamBudget) {
            return Err(invalid("live boundary cannot impersonate a full-state audit profile"));
        }
        if matches!(
            profile,
            NativeAuditProfile::Context32Long128 | NativeAuditProfile::Context32StreamLong128
        )
            && !(32769..=35840).contains(&self.driver.cursor)
        {
            return Err(invalid("context32 continuation cursor gate before submit"));
        }
        if matches!(
            profile,
            NativeAuditProfile::Context64Long128 | NativeAuditProfile::Context64StreamLong128
        )
            && !(65537..=69632).contains(&self.driver.cursor)
        {
            return Err(invalid("context64 continuation cursor gate before submit"));
        }
        if matches!(
            profile,
            NativeAuditProfile::Near192Long128 | NativeAuditProfile::Near192StreamLong128
        )
            && !(194561..=196474).contains(&self.driver.cursor)
        {
            return Err(invalid("near192 continuation cursor gate before submit"));
        }
        if private_directory.is_some_and(|p| !p.is_absolute() || p.exists())
            || (private_directory.is_some() && !cfg!(feature = "qwen35-draft-private-state"))
        {
            return Err(invalid("native private diagnostic path/compiled gate"));
        }
        if self.driver.poisoned
            || expected != &self.checkpoint
            || self.ledger.pending_published
            || self.eos.is_empty()
            || self
                .driver
                .cursor
                .checked_add(max_tokens + B - 2)
                .is_none_or(|n| n > self.driver.capacity)
            || self
                .driver
                .generation
                .checked_add(max_tokens as u64)
                .is_none()
            || self.ledger.revision == u64::MAX
        {
            return Err(invalid("native decode audit admission before submit"));
        }
        self.ledger
            .generated
            .try_reserve(max_tokens)
            .map_err(|e| invalid(e.to_string()))?;
        let start = Instant::now();
        #[cfg(feature = "qwen35-draft-private-state")]
        let mut private_series = private_directory
            .map(|p| session_private_audit::SessionPrivateAudit::new(p, "session-retained-rebind"))
            .transpose()?;
        let mut trajectory_capture = trajectory::Capture::from_env(&self.driver,&self.draft,&self.ledger.consumed,true)?;
        self.driver.trajectory_capture = trajectory_capture.is_some();
        self.driver.poisoned = true;
        let result = (|| {
            let initial = self.driver.anchor;
            let cursor_start = self.driver.cursor;
            let mut tokens = vec![initial];
            let mut stop = clip_publication(
                &mut tokens,
                Some(PublicationBudget {
                    remaining: max_tokens,
                    eos: &self.eos,
                }),
            )?;
            self.ledger.generated.push(initial);
            self.ledger.pending_published = true;
            if let Some(sink) = sink {
                dispatch_verified_commit(
                    sink,
                    Qwen35VerifiedCommit {
                        initial_anchor: true,
                        block_index: 0,
                        generation_before: self.driver.generation as usize,
                        generation_after: self.driver.generation as usize,
                        cursor_before: self.driver.cursor,
                        cursor_after: self.driver.cursor,
                        token_ids: vec![initial],
                        stop_reason: stop.map(str::to_owned),
                    },
                )?;
            }
            let mut transactions = Vec::with_capacity(max_tokens);
            let mut draft_context_reset = false;
            let mut private_capture = None::<Value>;
            if stop.is_none() {
                let growing_base = self.draft_growth_base.take();
                self.rebind_draft(None, growing_base)?;
                draft_context_reset = true;
            }
            for round in 0..max_tokens {
                if stop.is_some() {
                    break;
                }
                self.draft.prepare_context().map_err(invalid)?;
                self.draft.submit().map_err(invalid)?;
                if let Some(capture) = &mut trajectory_capture { capture.begin(&self.driver,&self.draft)?; }
                #[cfg(feature = "qwen35-draft-private-state")]
                if let Some(audit) = &mut private_series {
                    audit.capture_selected(
                        &self.draft,
                        &self.driver,
                        round,
                        transactions.last(),
                        &self.ledger.consumed,
                        &[],
                    )?;
                    private_capture = audit.first_binding();
                }
                self.driver.poisoned = false;
                let previous_anchor = self.driver.anchor;
                let mut tx = self.driver.verify_product(
                    &self.draft,
                    Some(PublicationBudget {
                        remaining: max_tokens - tokens.len(),
                        eos: &self.eos,
                    }),
                )?;
                if let Some(capture) = &mut trajectory_capture { capture.commit(&self.driver,&tx)?; }
                check_committed_path(
                    &self.driver, &mut tx, previous_anchor, self.fast_session_path,
                )?;
                let emitted: Vec<u32> = serde_json::from_value(tx["emitted_token_ids"].clone())
                    .map_err(|e| invalid(e.to_string()))?;
                self.ledger.commit_generated(
                    &emitted,
                    self.driver.cursor,
                    self.driver.generation,
                )?;
                tokens.extend_from_slice(&emitted);
                stop = match tx["publication_stop_reason"].as_str() {
                    Some("eos") => Some("eos"),
                    Some("max_new_tokens") => Some("max_new_tokens"),
                    None => None,
                    _ => return Err(invalid("session decode stop")),
                };
                if let Some(sink) = sink {
                    dispatch_verified_commit(
                        sink,
                        Qwen35VerifiedCommit {
                            initial_anchor: false,
                            block_index: round,
                            generation_before: tx["generation_before"]
                                .as_u64()
                                .ok_or_else(|| invalid("session sink generation"))?
                                as usize,
                            generation_after: self.driver.generation as usize,
                            cursor_before: tx["cursor_before"]
                                .as_u64()
                                .ok_or_else(|| invalid("session sink cursor"))?
                                as usize,
                            cursor_after: self.driver.cursor,
                            token_ids: emitted.clone(),
                            stop_reason: stop.map(str::to_owned),
                        },
                    )?;
                }
                transactions.push(tx);
                if stop.is_some() {
                    break;
                }
                if round + 1 == max_tokens {
                    return Err(invalid("session decode budget exhausted"));
                }
                self.rebind_draft(Some(emitted.len()), None)?;
            }
            let stop = stop.ok_or_else(|| invalid("session decode incomplete"))?;
            self.driver.stream.synchronize()?;
            self.draft.synchronize().map_err(invalid)?;
            self.driver.poisoned = false;
            self.ledger.revision += 1;
            let decode_wall_ms = start.elapsed().as_secs_f64() * 1000.0;
            let seal_start = Instant::now();
            let trajectory_report = trajectory_capture.take().map(|capture|capture.finish()).transpose()?;
            self.driver.trajectory_capture = false;
            self.checkpoint = self.seal_for_continuation()?;
            let seal_wall_ms = seal_start.elapsed().as_secs_f64() * 1000.0;
            let visible_end = tokens
                .iter()
                .position(|t| self.eos.contains(t))
                .unwrap_or(tokens.len());
            #[cfg(feature = "qwen35-draft-private-state")]
            let series_report = private_series.take().map(|a| a.report());
            #[cfg(not(feature = "qwen35-draft-private-state"))]
            let series_report = None::<Value>;
            Ok(json!({"schema":"qwen35-native-session-decode-audit-v1",
                "status":"PASS_NATIVE_SESSION_DECODE_EXECUTED_ONLY",
                "native_session_audit_profile":profile.name(),"max_new_tokens":max_tokens,
                "token_ids":tokens,"visible_token_ids":&tokens[..visible_end],
                "stop_reason":stop,"transactions":transactions,"cursor_start":cursor_start,
                "decode_wall_ms_with_path_audit_excluding_state_seal":decode_wall_ms,
                "decode_wall_ms_excluding_state_seal":decode_wall_ms,
                "fast_session_path_active":self.fast_session_path,
                "fast_feature_identity_active":self.driver.fast_feature_identity,
                "session_input_path_policy":if self.fast_session_path {
                    "metadata_only_fast_v1"
                } else { "device_readback_v1" },
                "commit_completion_required":true,
                "state_seal_wall_ms":seal_wall_ms,"state_seal_policy":self.seal_policy(),
                "draft_context_full_rebind":draft_context_reset,
                "native_session_private_capture":private_capture,
                "native_session_private_series":series_report,
                "dflash_live_trajectory":trajectory_report,
                "draft_weights_reloaded":false,"target_prefix_recomputed":false,
                "dflash_history_prepare":self.draft.history_prepare_report(),
                "output_head_batching":self.driver.output_head_batching_report(),
                "native_verified_sink_used":sink.is_some(),
                "checkpoint":self.checkpoint,"http_publication":false,
                "semantic_qualification":false,"product_ready":false}))
        })();
        if result.is_err() {
            self.driver.poisoned = true;
            let _ = self.driver.stream.synchronize();
            let _ = self.draft.synchronize();
        }
        result
    }
}

#[cfg(feature = "qwen35-session-continuation")]
impl Qwen35ResidentDraftCuda {
    /// Materialize a full payload witness without replacing the live continuation
    /// capability. Explicit save calls this; ordinary text requests do not.
    pub fn audit_native_session_checkpoint(&mut self, expected: &Value) -> Result<Value> {
        if self.poisoned { return Err(invalid("native owner poisoned")); }
        let session = self.session.as_mut().ok_or_else(|| invalid("no retained session"))?;
        if expected != &session.checkpoint { return Err(invalid("stale payload audit boundary")); }
        let result = session.seal();
        if result.is_err() { session.driver.poisoned = true; self.poisoned = true; }
        result
    }

    /// Switch a successfully restored resident owner to live metadata boundaries
    /// only after the caller has checked its separately returned full state seal.
    pub fn enable_resident_live_boundary(&mut self, expected: &Value) -> Result<Value> {
        if self.poisoned { return Err(invalid("native owner poisoned")); }
        // These are fresh launch choices, not restored CUDA authority. Keep
        // the complete payload seal until the controller has authenticated it.
        let policy = LiveContinuationPolicy::from_switches(
            std::env::var_os("AUGUR_QWEN35_FAST_SESSION_PATH").as_deref(),
            std::env::var_os("AUGUR_QWEN35_FAST_FEATURE_IDENTITY").as_deref(),
            std::env::var_os("AUGUR_QWEN35_REPLAY_HEAD_ELISION").as_deref(),
        )?;
        let session = self.session.as_mut().ok_or_else(|| invalid("no retained session"))?;
        if expected != &session.checkpoint { return Err(invalid("stale live-boundary admission")); }
        // Restored state has no cold prefix to replay. Select only the recipe's
        // live B8 row-reuse choice; retain the authenticated restored core.
        let prefill_switches = batched_verify::prefill_policy::Switches::from_env().resolve(
            batched_verify::prefill_policy::Scope {
                canonical_session: true,
                resident: self.target.is_some(),
                exact_batched_target: session.driver.exact_projection_tiles,
                vision: self.vision.is_some(),
                capacity: session.driver.capacity,
                history_rows: session.driver.context_rows,
            },
        )?;
        let row_reuse = batched_verify::projection_row_reuse_switch(
            prefill_switches.row_reuse.as_deref(),
            session.driver.exact_projection_tiles,
            true,
        )?;
        if prefill_switches.llama_rotated_kv != session.driver.llama_rotated_kv {
            return Err(invalid("live continuation cannot change full-attention KV basis"));
        }
        let result = (|| {
            let boundary = session.seal_live_boundary()?;
            let policy_changed = session.driver.projection_row_reuse != row_reuse;
            let executor = session.driver.batch_verify_execution.take();
            session.driver.projection_row_reuse = row_reuse;
            // Keep the restored core/ReplaySSM/head workspace. Only cached
            // operations and graphs belong to the previous projection policy.
            session.driver.batch_verify_execution = rebind_live_projection_executor(
                executor,
                session.driver.batched_suffix || row_reuse,
                policy_changed,
                |executor| executor.rebind_projection_policy(&session.driver),
            )?;
            if policy_changed {
                session.driver.batch_verify_audit = None;
            }
            session.live_boundaries = true;
            session.fast_session_path = policy.fast_session_path;
            session.driver.fast_feature_identity = policy.fast_feature_identity;
            session.driver.replay_head_elision = policy.replay_head_elision;
            session.checkpoint = boundary.clone();
            Ok(boundary)
        })();
        match result {
            Ok(boundary) => Ok(boundary),
            Err(e) => { session.driver.poisoned = true; self.poisoned = true; Err(e) }
        }
    }
    /// Research-only retain gate. Fresh HTTP generation still uses run_request.
    pub fn open_native_session(
        &mut self,
        config: &Qwen35AutoregressiveCudaConfig,
        session_id: &str,
        options: Qwen35DraftTargetRollingOptions,
        sink: &Qwen35VerifiedCommitSink<'_>,
    ) -> Result<Value> {
        self.open_native_session_impl(config, session_id, options, sink, None, None, false, false)
    }

    /// Diagnostic numerical identity; never alters the ordinary native/HTTP path.
    pub fn open_native_session_with_prefill_scalar_audit(
        &mut self,
        config: &Qwen35AutoregressiveCudaConfig,
        session_id: &str,
        options: Qwen35DraftTargetRollingOptions,
        sink: &Qwen35VerifiedCommitSink<'_>,
    ) -> Result<Value> {
        self.open_native_session_impl(config, session_id, options, sink, None, None, true, false)
    }

    /// Native diagnostic only: same canonical B8 math for prefix and capture.
    pub fn open_native_session_with_full_canonical_prefill_audit(
        &mut self,
        config: &Qwen35AutoregressiveCudaConfig,
        session_id: &str,
        options: Qwen35DraftTargetRollingOptions,
        sink: &Qwen35VerifiedCommitSink<'_>,
    ) -> Result<Value> {
        let bytes = read_stable_regular_file(&config.tokenizer_path, "target tokenizer")?;
        let tokenizer = Tokenizer::from_bytes(&bytes).map_err(|e| invalid(e.to_string()))?;
        if tokenizer
            .encode(config.prompt_text.as_str(), false)
            .map_err(|e| invalid(e.to_string()))?
            .len()
            > 16384
        {
            return Err(invalid("legacy canonical session prompt bound"));
        }
        self.open_native_session_impl(config, session_id, options, sink, None, None, false, true)
    }

    /// Separate real32768-input admission; no legacy full-canonical API expansion.
    pub fn open_native_session_context32_audit(
        &mut self,
        config: &Qwen35AutoregressiveCudaConfig,
        session_id: &str,
        options: Qwen35DraftTargetRollingOptions,
        sink: &Qwen35VerifiedCommitSink<'_>,
    ) -> Result<Value> {
        if !cfg!(feature = "qwen35-session-context32-audit") {
            return Err(invalid("context32 session feature not compiled"));
        }
        let bytes = read_stable_regular_file(&config.tokenizer_path, "target tokenizer")?;
        let tokenizer = Tokenizer::from_bytes(&bytes).map_err(|e| invalid(e.to_string()))?;
        if tokenizer
            .encode(config.prompt_text.as_str(), false)
            .map_err(|e| e.to_string())
            .map_err(invalid)?
            .len()
            != 32768
        {
            return Err(invalid("context32 session requires exact32768 prompt"));
        }
        self.open_native_session_impl(config, session_id, options, sink, None, None, false, true)
    }

    /// Separate real65536-input admission; no legacy/full-canonical API expansion.
    pub fn open_native_session_context64_audit(
        &mut self,
        config: &Qwen35AutoregressiveCudaConfig,
        session_id: &str,
        options: Qwen35DraftTargetRollingOptions,
        sink: &Qwen35VerifiedCommitSink<'_>,
    ) -> Result<Value> {
        if !cfg!(feature = "qwen35-session-context64-audit") {
            return Err(invalid("context64 session feature not compiled"));
        }
        let bytes = read_stable_regular_file(&config.tokenizer_path, "target tokenizer")?;
        let tokenizer = Tokenizer::from_bytes(&bytes).map_err(|e| invalid(e.to_string()))?;
        if tokenizer
            .encode(config.prompt_text.as_str(), false)
            .map_err(|e| e.to_string())
            .map_err(invalid)?
            .len()
            != 65536
        {
            return Err(invalid("context64 session requires exact65536 prompt"));
        }
        self.open_native_session_impl(config, session_id, options, sink, None, None, false, true)
    }

    /// Separate near-capacity input; does not expand legacy/32K/64K admission.
    pub fn open_native_session_near192_audit(
        &mut self,
        config: &Qwen35AutoregressiveCudaConfig,
        session_id: &str,
        options: Qwen35DraftTargetRollingOptions,
        sink: &Qwen35VerifiedCommitSink<'_>,
    ) -> Result<Value> {
        if !cfg!(feature = "qwen35-session-near192-audit") {
            return Err(invalid("near192 session feature not compiled"));
        }
        let bytes = read_stable_regular_file(&config.tokenizer_path, "target tokenizer")?;
        let tokenizer = Tokenizer::from_bytes(&bytes).map_err(|e| invalid(e.to_string()))?;
        if tokenizer
            .encode(config.prompt_text.as_str(), false)
            .map_err(|e| invalid(e.to_string()))?
            .len()
            != 194560
        {
            return Err(invalid("near192 session requires exact194560 prompt"));
        }
        self.open_native_session_impl(config, session_id, options, sink, None, None, false, true)
    }

    /// Fresh u32 near-capacity reference. Candidate must already be closed.
    pub fn fresh_recompute_native_session_near192_audit(
        &mut self,
        config: &Qwen35AutoregressiveCudaConfig,
        session_id: &str,
        options: Qwen35DraftTargetRollingOptions,
        exact_history: &[u32],
        draft_directory: &Path,
        sink: &Qwen35VerifiedCommitSink<'_>,
    ) -> Result<Value> {
        let reference = FreshReference {
            token_ids: exact_history,
            max_new_tokens: 128,
            audit_directory: None,
            draft_audit_directory: Some(draft_directory),
            profile: NativeAuditProfile::Near192Long128,
        };
        reference.validate()?;
        if !draft_directory.is_absolute() || draft_directory.exists() {
            return Err(invalid(
                "near192 private audit requires exclusive absolute path",
            ));
        }
        self.open_native_session_impl(
            config,
            session_id,
            options,
            sink,
            Some(reference),
            None,
            false,
            true,
        )
    }

    /// Fresh u32 reference for32K, hashing full target state without raw plane dumps.
    pub fn fresh_recompute_native_session_context32_audit(
        &mut self,
        config: &Qwen35AutoregressiveCudaConfig,
        session_id: &str,
        options: Qwen35DraftTargetRollingOptions,
        exact_history: &[u32],
        draft_directory: &Path,
        sink: &Qwen35VerifiedCommitSink<'_>,
    ) -> Result<Value> {
        let reference = FreshReference {
            token_ids: exact_history,
            max_new_tokens: 128,
            audit_directory: None,
            draft_audit_directory: Some(draft_directory),
            profile: NativeAuditProfile::Context32Long128,
        };
        reference.validate()?;
        if !draft_directory.is_absolute() || draft_directory.exists() {
            return Err(invalid(
                "context32 private audit requires exclusive absolute path",
            ));
        }
        self.open_native_session_impl(
            config,
            session_id,
            options,
            sink,
            Some(reference),
            None,
            false,
            true,
        )
    }

    /// Fresh u32 reference for64K, hashing full target state without raw plane dumps.
    pub fn fresh_recompute_native_session_context64_audit(
        &mut self,
        config: &Qwen35AutoregressiveCudaConfig,
        session_id: &str,
        options: Qwen35DraftTargetRollingOptions,
        exact_history: &[u32],
        draft_directory: &Path,
        sink: &Qwen35VerifiedCommitSink<'_>,
    ) -> Result<Value> {
        let reference = FreshReference {
            token_ids: exact_history,
            max_new_tokens: 128,
            audit_directory: None,
            draft_audit_directory: Some(draft_directory),
            profile: NativeAuditProfile::Context64Long128,
        };
        reference.validate()?;
        if !draft_directory.is_absolute() || draft_directory.exists() {
            return Err(invalid(
                "context64 private audit requires exclusive absolute path",
            ));
        }
        self.open_native_session_impl(
            config,
            session_id,
            options,
            sink,
            Some(reference),
            None,
            false,
            true,
        )
    }

    /// Full fresh prefill from exact u32 history; not a text/HTTP endpoint.
    /// Close the candidate first so two full KV owners never coexist.
    pub fn fresh_recompute_native_session_audit(
        &mut self,
        config: &Qwen35AutoregressiveCudaConfig,
        session_id: &str,
        options: Qwen35DraftTargetRollingOptions,
        exact_history: &[u32],
        max_tokens: usize,
        audit_directory: &Path,
        prefill_scalar_audit: bool,
        full_canonical_prefill_audit: bool,
        draft_audit_directory: Option<&Path>,
        sink: &Qwen35VerifiedCommitSink<'_>,
    ) -> Result<Value> {
        self.fresh_recompute_native_session_impl(
            config,
            session_id,
            options,
            exact_history,
            max_tokens,
            audit_directory,
            prefill_scalar_audit,
            full_canonical_prefill_audit,
            draft_audit_directory,
            sink,
            NativeAuditProfile::Short32,
        )
    }

    /// Explicit128-output full-canonical/private audit. No ordinary API expansion.
    pub fn fresh_recompute_native_session_long_audit(
        &mut self,
        config: &Qwen35AutoregressiveCudaConfig,
        session_id: &str,
        options: Qwen35DraftTargetRollingOptions,
        exact_history: &[u32],
        audit_directory: &Path,
        draft_directory: &Path,
        sink: &Qwen35VerifiedCommitSink<'_>,
    ) -> Result<Value> {
        self.fresh_recompute_native_session_impl(
            config,
            session_id,
            options,
            exact_history,
            128,
            audit_directory,
            false,
            true,
            Some(draft_directory),
            sink,
            NativeAuditProfile::Long128,
        )
    }

    fn fresh_recompute_native_session_impl(
        &mut self,
        config: &Qwen35AutoregressiveCudaConfig,
        session_id: &str,
        options: Qwen35DraftTargetRollingOptions,
        exact_history: &[u32],
        max_tokens: usize,
        audit_directory: &Path,
        prefill_scalar_audit: bool,
        full_canonical_prefill_audit: bool,
        draft_audit_directory: Option<&Path>,
        sink: &Qwen35VerifiedCommitSink<'_>,
        profile: NativeAuditProfile,
    ) -> Result<Value> {
        let reference = FreshReference {
            token_ids: exact_history,
            max_new_tokens: max_tokens,
            audit_directory: Some(audit_directory),
            draft_audit_directory,
            profile,
        };
        reference.validate()?;
        if draft_audit_directory.is_some_and(|p| !p.is_absolute() || p.exists())
            || (draft_audit_directory.is_some()
                && (!full_canonical_prefill_audit || !cfg!(feature = "qwen35-draft-private-state")))
        {
            return Err(invalid(
                "fresh private audit requires full canonical profile/new path",
            ));
        }
        self.open_native_session_impl(
            config,
            session_id,
            options,
            sink,
            Some(reference),
            None,
            prefill_scalar_audit,
            full_canonical_prefill_audit,
        )
    }

    /// Diagnostic-only same-upstream-state prefill/capture comparison; no publication.
    pub fn audit_native_prefill_transition(
        &mut self,
        config: &Qwen35AutoregressiveCudaConfig,
        options: Qwen35DraftTargetRollingOptions,
        directory: &Path,
    ) -> Result<Value> {
        let reject_publication: &Qwen35VerifiedCommitSink<'_> =
            &|_| Err(invalid("transition diagnostic must never publish"));
        self.open_native_session_impl(
            config,
            "transition-audit",
            options,
            reject_publication,
            None,
            Some(directory),
            false,
            false,
        )
    }

    /// Explicit native growing-session diagnostic; no HTTP/default admission.
    pub fn open_native_growing_session_audit(
        &mut self, config: &Qwen35AutoregressiveCudaConfig, session_id: &str,
        options: Qwen35DraftTargetRollingOptions, sink: &Qwen35VerifiedCommitSink<'_>,
    ) -> Result<Value> {
        let bytes = read_stable_regular_file(&config.tokenizer_path, "target tokenizer")?;
        let tokenizer = Tokenizer::from_bytes(&bytes).map_err(|e| invalid(e.to_string()))?;
        let count = tokenizer.encode(config.prompt_text.as_str(), false)
            .map_err(|e| invalid(e.to_string()))?.len();
        if !(1..=16384).contains(&count) {
            return Err(invalid("legacy growing-session prompt bound"));
        }
        self.open_native_session_with_history_impl(config, session_id, options, sink,
            None, None, false, true, true, None)
    }

    /// Variable-length text admission; same initialized-history and canonical
    /// prefill/decode implementation, with the full verification reserve.
    /// This entry point is not itself a product qualification claim.
    pub fn open_resident_text_session(
        &mut self, config: &Qwen35AutoregressiveCudaConfig, session_id: &str,
        options: Qwen35DraftTargetRollingOptions, sink: &Qwen35VerifiedCommitSink<'_>,
    ) -> Result<Value> {
        self.open_resident_text_session_with_budget(config, session_id, options, sink,
            crate::resident_session_profile::INITIAL_OUTPUT_LIMIT)
    }

    /// Per-request publication ceiling; allocation and verification reserves stay fixed.
    pub fn open_resident_text_session_with_budget(
        &mut self, config: &Qwen35AutoregressiveCudaConfig, session_id: &str,
        options: Qwen35DraftTargetRollingOptions, sink: &Qwen35VerifiedCommitSink<'_>,
        output_ceiling: usize,
    ) -> Result<Value> {
        crate::resident_session_profile::ResidentSessionProfile::Resident
            .output_budget(Some(output_ceiling), true).map_err(invalid)?;
        let bytes = read_stable_regular_file(&config.tokenizer_path, "target tokenizer")?;
        let tokenizer = match self.target.as_ref() {
            Some(owner) => owner.request_tokenizer(&bytes)?,
            None => Arc::new(Tokenizer::from_bytes(&bytes).map_err(|e| invalid(e.to_string()))?),
        };
        let count = tokenizer.encode(config.prompt_text.as_str(), false)
            .map_err(|e| invalid(e.to_string()))?.len();
        if !crate::resident_session_profile::ResidentSessionProfile::Resident.admits_initial(count) {
            return Err(invalid("resident initial input/output reserve before submit"));
        }
        self.open_native_session_with_history_impl(config, session_id, options, sink,
            None, None, false, true, true, Some(output_ceiling))
    }

    pub fn decode_resident_text_session_stream(
        &mut self, expected: &Value, sink: &Qwen35VerifiedCommitSink<'_>,
    ) -> Result<Value> {
        self.decode_resident_text_session_stream_with_budget(expected, sink,
            crate::resident_session_profile::CONTINUATION_OUTPUT_LIMIT)
    }

    pub fn decode_resident_text_session_stream_with_budget(
        &mut self, expected: &Value, sink: &Qwen35VerifiedCommitSink<'_>, output_ceiling: usize,
    ) -> Result<Value> {
        crate::resident_session_profile::ResidentSessionProfile::Resident
            .output_budget(Some(output_ceiling), false).map_err(invalid)?;
        if self.poisoned { return Err(invalid("resident owner poisoned")); }
        let session = self.session.as_mut().ok_or_else(|| invalid("no retained resident session"))?;
        let profile = if output_ceiling == 128 { NativeAuditProfile::ResidentStream128 }
            else { NativeAuditProfile::ResidentStreamBudget };
        let result = session.decode_audit(expected, output_ceiling, None, profile, Some(sink));
        if session.driver.poisoned { self.poisoned = true; }
        result
    }

    /// Independent cold recomputation from the exact consumed u32 ledger.
    pub fn fresh_recompute_resident_text_session(
        &mut self, config: &Qwen35AutoregressiveCudaConfig, session_id: &str,
        options: Qwen35DraftTargetRollingOptions, exact_history: &[u32],
        audit_directory: &Path, sink: &Qwen35VerifiedCommitSink<'_>,
    ) -> Result<Value> {
        let reference = FreshReference { token_ids: exact_history, max_new_tokens: 128,
            audit_directory: Some(audit_directory), draft_audit_directory: None,
            profile: NativeAuditProfile::ResidentLong128 };
        reference.validate()?;
        self.open_native_session_with_history_impl(config, session_id, options, sink,
            Some(reference), None, false, true, true, None)
    }

    pub fn fresh_recompute_native_growing_session_audit(
        &mut self, config: &Qwen35AutoregressiveCudaConfig, session_id: &str,
        options: Qwen35DraftTargetRollingOptions, exact_history: &[u32], max_tokens: usize,
        audit_directory: &Path, sink: &Qwen35VerifiedCommitSink<'_>,
    ) -> Result<Value> {
        let reference = FreshReference { token_ids:exact_history, max_new_tokens:max_tokens,
            audit_directory:Some(audit_directory), draft_audit_directory:None,
            profile:NativeAuditProfile::Growing32 };
        reference.validate()?;
        self.open_native_session_with_history_impl(config, session_id, options, sink,
            Some(reference), None, false, true, true, None)
    }

    pub fn decode_native_growing_session_audit(
        &mut self, expected: &Value, max_tokens: usize,
    ) -> Result<Value> {
        if self.poisoned { return Err(invalid("native owner poisoned")); }
        let session = self.session.as_mut().ok_or_else(|| invalid("no retained growing session"))?;
        let result = session.decode_audit(expected, max_tokens, None, NativeAuditProfile::Growing32, None);
        if session.driver.poisoned { self.poisoned = true; }
        result
    }

    /// Separate full128 streaming audit. Does not widen the Growing32 gate or
    /// authorize HTTP/default admission; callback failure poisons the owner.
    pub fn decode_native_growing_session_stream_audit(
        &mut self, expected: &Value, sink: &Qwen35VerifiedCommitSink<'_>,
    ) -> Result<Value> {
        if self.poisoned { return Err(invalid("native owner poisoned")); }
        let session = self.session.as_mut().ok_or_else(|| invalid("no retained growing session"))?;
        let result = session.decode_audit(expected, 128, None, NativeAuditProfile::GrowingStream128, Some(sink));
        if session.driver.poisoned { self.poisoned = true; }
        result
    }

    /// Exact-u32 oracle for the separate full128 initialized-history profile.
    pub fn fresh_recompute_native_growing_session_long_audit(
        &mut self, config: &Qwen35AutoregressiveCudaConfig, session_id: &str,
        options: Qwen35DraftTargetRollingOptions, exact_history: &[u32],
        audit_directory: &Path, sink: &Qwen35VerifiedCommitSink<'_>,
    ) -> Result<Value> {
        let reference = FreshReference { token_ids:exact_history, max_new_tokens:128,
            audit_directory:Some(audit_directory), draft_audit_directory:None,
            profile:NativeAuditProfile::GrowingLong128 };
        reference.validate()?;
        self.open_native_session_with_history_impl(config, session_id, options, sink,
            Some(reference), None, false, true, true, None)
    }

    fn open_native_session_impl(
        &mut self,
        config: &Qwen35AutoregressiveCudaConfig,
        session_id: &str,
        options: Qwen35DraftTargetRollingOptions,
        sink: &Qwen35VerifiedCommitSink<'_>,
        reference: Option<FreshReference<'_>>,
        transition_directory: Option<&Path>,
        prefill_scalar_audit: bool,
        full_canonical_prefill_audit: bool,
    ) -> Result<Value> {
        self.open_native_session_with_history_impl(config, session_id, options, sink, reference,
            transition_directory, prefill_scalar_audit, full_canonical_prefill_audit, false, None)
    }

    fn open_native_session_with_history_impl(
        &mut self,
        config: &Qwen35AutoregressiveCudaConfig,
        session_id: &str,
        options: Qwen35DraftTargetRollingOptions,
        sink: &Qwen35VerifiedCommitSink<'_>,
        reference: Option<FreshReference<'_>>,
        transition_directory: Option<&Path>,
        prefill_scalar_audit: bool,
        full_canonical_prefill_audit: bool,
        growing: bool,
        resident_output_ceiling: Option<usize>,
    ) -> Result<Value> {
        if self.poisoned
            || self.session.is_some()
            || self.has_retained_input_prefix()
            || !valid_id(session_id)
            || self.target.is_none()
            || self.vision.is_none()
            || !options.product_mode
            || !options.incremental_draft_context
            || options.validate_draft_context_reuse
            || options
                .request
                .as_ref()
                .is_none_or(|r| r.target_only_reference || r.audit_committed_state)
            || config.capacity != 196608
            || config.device_ordinal != self.weights.device_ordinal()
            || std::env::var("AUGUR_QWEN35_TOP16_VALIDATION_MODE").as_deref() != Ok("resident")
        {
            return Err(invalid("native session retain admission before submit"));
        }
        validate_resident_vision_selection(
            self.vision_path.as_deref(),
            config.vision_safetensors_path.as_deref(),
            config.vision_safetensors_path.as_deref(),
        )?;
        self.poisoned = true;
        let result = run_rolling_with_resident_draft(
            config,
            &self.q8_path,
            &self.draft_config_path,
            2048,
            1024,
            self.vision_path.as_deref(),
            true,
            true,
            false,
            options.request.as_ref(),
            Some(sink),
            Some(&self.weights),
            self.target.as_ref(),
            self.vision.as_ref(),
            Some(RetainedOutput::Generated(session_id, &mut self.session)),
            reference,
            transition_directory,
            prefill_scalar_audit,
            full_canonical_prefill_audit,
            growing,
            self.session_cancellation.as_ref(),
            resident_output_ceiling,
        );
        if result.is_ok() {
            self.poisoned = false;
        }
        result
    }

    /// Consumes only explicit input tokens, never publishes generated text.
    /// The full exact prior input history prevents digest-only cache admission.
    pub fn append_native_session(
        &mut self,
        expected: &Value,
        exact_history: &[u32],
        suffix: &[u32],
    ) -> Result<Value> {
        if self.poisoned {
            return Err(invalid("native session outer owner poisoned"));
        }
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| invalid("no retained session"))?;
        let result = session.append(expected, exact_history, suffix);
        if session.driver.poisoned {
            self.poisoned = true;
        }
        result
    }

    /// Native numerical diagnosis only. Exports initialized target/feature bytes.
    pub fn capture_native_session_state_audit(
        &mut self,
        expected: &Value,
        directory: &Path,
    ) -> Result<Value> {
        if self.poisoned {
            return Err(invalid("native owner poisoned"));
        }
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| invalid("no retained session"))?;
        if expected != &session.checkpoint {
            return Err(invalid("stale state capture"));
        }
        self.poisoned = true;
        let result = (|| {
            session.draft.synchronize().map_err(invalid)?;
            let audit = audit_target_state_to(&mut session.driver, Some(directory))?;
            for key in [
                "cursor",
                "target_committed_state_sha256",
                "feature_bytes_sha256",
            ] {
                if audit[key] != expected[key] {
                    return Err(invalid("capture/checkpoint drift"));
                }
            }
            Ok(audit)
        })();
        if result.is_ok() {
            self.poisoned = false;
        } else {
            session.driver.poisoned = true;
        }
        result
    }

    pub fn close_native_session(&mut self, expected: &Value) -> Result<Value> {
        if self.poisoned {
            return Err(invalid("poisoned owner requires process restart"));
        }
        let session = self
            .session
            .as_ref()
            .ok_or_else(|| invalid("no retained session"))?;
        if expected != &session.checkpoint {
            return Err(invalid("stale session close"));
        }
        // Any post-admission CUDA failure invalidates the outer owner too.
        self.poisoned = true;
        session.draft.synchronize().map_err(invalid)?;
        session.driver.stream.synchronize()?;
        drop(self.session.take());
        self.poisoned = false;
        Ok(json!({"session_closed":true,"product_ready":false}))
    }

    /// Captures initialized draft state after full rebind and first real block,
    /// before target verification. No HTTP or persistent checkpoint authority.
    pub fn decode_native_session_with_private_audit(
        &mut self,
        expected: &Value,
        max_tokens: usize,
        directory: &Path,
    ) -> Result<Value> {
        if self.poisoned {
            return Err(invalid("native owner poisoned"));
        }
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| invalid("no retained session"))?;
        let result = session.decode_audit(
            expected,
            max_tokens,
            Some(directory),
            NativeAuditProfile::Short32,
            None,
        );
        if session.driver.poisoned {
            self.poisoned = true;
        }
        result
    }

    /// Separate128-token private-state qualification profile, never HTTP/default.
    pub fn decode_native_session_long_private_audit(
        &mut self,
        expected: &Value,
        directory: &Path,
    ) -> Result<Value> {
        if self.poisoned {
            return Err(invalid("native owner poisoned"));
        }
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| invalid("no retained session"))?;
        let result = session.decode_audit(
            expected,
            128,
            Some(directory),
            NativeAuditProfile::Long128,
            None,
        );
        if session.driver.poisoned {
            self.poisoned = true;
        }
        result
    }

    /// Selected private bytes at real32K extent, isolated from8K profiles.
    pub fn decode_native_session_context32_private_audit(
        &mut self,
        expected: &Value,
        directory: &Path,
    ) -> Result<Value> {
        if self.poisoned {
            return Err(invalid("native owner poisoned"));
        }
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| invalid("no retained session"))?;
        let result = session.decode_audit(
            expected,
            128,
            Some(directory),
            NativeAuditProfile::Context32Long128,
            None,
        );
        if session.driver.poisoned {
            self.poisoned = true;
        }
        result
    }

    /// Selected private bytes at real64K extent, isolated from8K/32K profiles.
    pub fn decode_native_session_context64_private_audit(
        &mut self,
        expected: &Value,
        directory: &Path,
    ) -> Result<Value> {
        if self.poisoned {
            return Err(invalid("native owner poisoned"));
        }
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| invalid("no retained session"))?;
        let result = session.decode_audit(
            expected,
            128,
            Some(directory),
            NativeAuditProfile::Context64Long128,
            None,
        );
        if session.driver.poisoned {
            self.poisoned = true;
        }
        result
    }

    /// Selected private bytes after near-capacity append; reserves134 input slots.
    pub fn decode_native_session_near192_private_audit(
        &mut self,
        expected: &Value,
        directory: &Path,
    ) -> Result<Value> {
        if self.poisoned {
            return Err(invalid("native owner poisoned"));
        }
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| invalid("no retained session"))?;
        let result = session.decode_audit(
            expected,
            128,
            Some(directory),
            NativeAuditProfile::Near192Long128,
            None,
        );
        if session.driver.poisoned {
            self.poisoned = true;
        }
        result
    }

    /// Shared callback path for a selected non-private long-context profile.
    /// Any callback failure drains and poisons this owner; only Drop + reopen recovers.
    fn decode_native_session_stream_profile(
        &mut self,
        expected: &Value,
        sink: &Qwen35VerifiedCommitSink<'_>,
        profile: NativeAuditProfile,
    ) -> Result<Value> {
        if self.poisoned {
            return Err(invalid("native owner poisoned"));
        }
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| invalid("no retained session"))?;
        let result = session.decode_audit(
            expected,
            128,
            None,
            profile,
            Some(sink),
        );
        if session.driver.poisoned {
            self.poisoned = true;
        }
        result
    }

    /// Explicit callback qualification path, short8K stream profile.
    pub fn decode_native_session_stream_audit(
        &mut self,
        expected: &Value,
        sink: &Qwen35VerifiedCommitSink<'_>,
    ) -> Result<Value> {
        self.decode_native_session_stream_profile(expected, sink, NativeAuditProfile::Stream128)
    }

    /// Explicit callback qualification path, real32K stream profile.
    pub fn decode_native_session_context32_stream_audit(
        &mut self,
        expected: &Value,
        sink: &Qwen35VerifiedCommitSink<'_>,
    ) -> Result<Value> {
        self.decode_native_session_stream_profile(
            expected,
            sink,
            NativeAuditProfile::Context32StreamLong128,
        )
    }

    /// Explicit callback qualification path, real64K stream profile.
    pub fn decode_native_session_context64_stream_audit(
        &mut self,
        expected: &Value,
        sink: &Qwen35VerifiedCommitSink<'_>,
    ) -> Result<Value> {
        self.decode_native_session_stream_profile(
            expected,
            sink,
            NativeAuditProfile::Context64StreamLong128,
        )
    }

    /// Explicit callback qualification path, near192K stream profile.
    pub fn decode_native_session_near192_stream_audit(
        &mut self,
        expected: &Value,
        sink: &Qwen35VerifiedCommitSink<'_>,
    ) -> Result<Value> {
        self.decode_native_session_stream_profile(
            expected,
            sink,
            NativeAuditProfile::Near192StreamLong128,
        )
    }

    /// Bounded native diagnostic only; returns no transport publication.
    pub fn decode_native_session_audit(
        &mut self,
        expected: &Value,
        max_tokens: usize,
    ) -> Result<Value> {
        if self.poisoned {
            return Err(invalid("native owner poisoned"));
        }
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| invalid("no retained session"))?;
        let result = session.decode_audit(
            expected,
            max_tokens,
            None,
            NativeAuditProfile::Short32,
            None,
        );
        if session.driver.poisoned {
            self.poisoned = true;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resident_budget_stream_extends_only_explicit_resident_requests() {
        for budget in [1, 127, 128, 129, 256, 512, 1024] {
            assert!(NativeAuditProfile::ResidentStreamBudget.validate(budget, false).is_ok());
            assert!(NativeAuditProfile::ResidentStreamBudget.validate(budget, true).is_err());
            assert_eq!(NativeAuditProfile::ResidentStream128.validate(budget, false).is_ok(),budget==128);
        }
        for budget in [0, 1025, usize::MAX] {
            assert!(NativeAuditProfile::ResidentStreamBudget.validate(budget, false).is_err());
        }
    }

    #[test]
    fn growing_reference_is_explicit_and_does_not_expand_legacy_profiles() {
        for n in [1, 7, 8, 31, 2047, 2048, 2049, 8192, 16384] {
            let tokens = vec![17; n];
            let reference = FreshReference { token_ids:&tokens, max_new_tokens:32,
                audit_directory:None, draft_audit_directory:None, profile:NativeAuditProfile::Growing32 };
            reference.validate().unwrap();
            if n <= 8192 {
                let legacy = FreshReference { profile:NativeAuditProfile::Short32, ..reference };
                assert!(legacy.validate().is_err());
            }
        }
        assert!(NativeAuditProfile::Growing32.validate(32,true).is_err());
        assert!(NativeAuditProfile::Growing32.validate(128,false).is_err());
        for n in [0,16385] {
            let tokens = vec![17;n];
            assert!(FreshReference { token_ids:&tokens,max_new_tokens:32,audit_directory:None,
                draft_audit_directory:None,profile:NativeAuditProfile::Growing32 }.validate().is_err());
        }
    }
    #[test]
    fn growing_long_and_stream_profiles_bind_history_sink_and_exact128_budget() {
        for profile in [NativeAuditProfile::GrowingLong128, NativeAuditProfile::GrowingStream128] {
            for limit in [0, 1, 2, 32, 127, 128, 129, 1024] {
                assert_eq!(profile.validate(limit, false).is_ok(), limit == 128);
                assert!(profile.validate(limit, true).is_err());
            }
            for growing in [false, true] {
                for sink in [false, true] {
                    assert_eq!(profile.validate_consumer(growing, sink).is_ok(),
                        growing && sink == (profile == NativeAuditProfile::GrowingStream128));
                }
            }
        }
        assert!(NativeAuditProfile::Stream128.validate_consumer(true, true).is_err());
        assert!(NativeAuditProfile::Growing32.validate_consumer(true, true).is_err());
        assert!(NativeAuditProfile::Growing32.validate(128, false).is_err());
        assert_ne!(NativeAuditProfile::GrowingLong128.name(), NativeAuditProfile::GrowingStream128.name());
    }

    #[test]
    fn fixed_long_stream_profiles_are_sink_bound_and_keep_profile_identity() {
        for (private, stream) in [
            (NativeAuditProfile::Context32Long128, NativeAuditProfile::Context32StreamLong128),
            (NativeAuditProfile::Context64Long128, NativeAuditProfile::Context64StreamLong128),
            (NativeAuditProfile::Near192Long128, NativeAuditProfile::Near192StreamLong128),
        ] {
            assert_eq!(private.name(), stream.name());
            assert!(stream.validate(128, false).is_ok());
            assert!(stream.validate(128, true).is_err());
            assert!(stream.validate_consumer(false, true).is_ok());
            assert!(stream.validate_consumer(false, false).is_err());
            assert!(private.validate_consumer(false, true).is_err());
        }
    }

    #[test]
    fn fixed_long_public_entrypoints_keep_capture_and_sink_profiles() {
        // Profile-only tests cannot detect a correct profile wired to the wrong
        // public entrypoint. Keep this source-level wiring check separate from
        // real CUDA qualification. The six small methods have no brace literals.
        fn method_body<'a>(source: &'a str, name: &str) -> &'a str {
            let marker = format!("pub fn {name}(");
            let start = source.find(&marker).expect("public entrypoint exists");
            let tail = &source[start..];
            let open = tail.find('{').expect("method body begins");
            let mut depth = 0;
            for (offset, ch) in tail[open..].char_indices() {
                match ch {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            return &tail[open..open + offset + 1];
                        }
                    }
                    _ => {}
                }
            }
            panic!("method body must close");
        }
        fn uses_profile(source: &str, name: &str, expected: NativeAuditProfile) -> bool {
            let body = method_body(source, name);
            body.matches("NativeAuditProfile::").count() == 1
                && body.contains(&format!("NativeAuditProfile::{expected:?}"))
        }
        let source = include_str!("committed_session.rs");
        for (private_name, private, stream_name, stream) in [
            (
                "decode_native_session_context32_private_audit",
                NativeAuditProfile::Context32Long128,
                "decode_native_session_context32_stream_audit",
                NativeAuditProfile::Context32StreamLong128,
            ),
            (
                "decode_native_session_context64_private_audit",
                NativeAuditProfile::Context64Long128,
                "decode_native_session_context64_stream_audit",
                NativeAuditProfile::Context64StreamLong128,
            ),
            (
                "decode_native_session_near192_private_audit",
                NativeAuditProfile::Near192Long128,
                "decode_native_session_near192_stream_audit",
                NativeAuditProfile::Near192StreamLong128,
            ),
        ] {
            for (name, expected, swapped) in [
                (private_name, private, stream),
                (stream_name, stream, private),
            ] {
                assert!(uses_profile(source, name, expected), "miswired {name}");
                let body = method_body(source, name);
                let swapped_body = body.replace(
                    &format!("NativeAuditProfile::{expected:?}"),
                    &format!("NativeAuditProfile::{swapped:?}"),
                );
                let mutant = source.replacen(body, &swapped_body, 1);
                assert!(
                    !uses_profile(&mutant, name, expected),
                    "swap escaped {name}"
                );
            }
            assert!(method_body(source, private_name).contains("Some(directory)"));
            assert!(!method_body(source, stream_name).contains("Some(directory)"));
            assert!(
                method_body(source, stream_name).contains("decode_native_session_stream_profile")
            );
        }
    }

    #[test]
    fn growing_long_reference_keeps_exact_input_and_disallows_stream_profile_alias() {
        for n in [0, 1, 7, 31, 2047, 2048, 2049, 16384, 16385] {
            let tokens = vec![17; n];
            let reference = FreshReference { token_ids:&tokens, max_new_tokens:128,
                audit_directory:None, draft_audit_directory:None,
                profile:NativeAuditProfile::GrowingLong128 };
            assert_eq!(reference.validate().is_ok(), (1..=16384).contains(&n));
            let stream = FreshReference { profile:NativeAuditProfile::GrowingStream128, ..reference };
            assert!(stream.validate().is_err());
        }
        assert!(FreshReference { token_ids:&[V as u32], max_new_tokens:128,
            audit_directory:None, draft_audit_directory:None,
            profile:NativeAuditProfile::GrowingLong128 }.validate().is_err());
    }
    #[test]
    fn explicit_long_profile_never_widens_short_or_allows_arbitrary_limits() {
        for n in [0, 1, 33, 64, 127, 128, 129, 1024] {
            assert!(NativeAuditProfile::Short32.validate(n, true).is_err());
            assert_eq!(
                NativeAuditProfile::Long128.validate(n, true).is_ok(),
                n == 128 && cfg!(feature = "qwen35-draft-private-state")
            );
        }
        assert!(NativeAuditProfile::Long128.validate(128, false).is_err());
        assert!(NativeAuditProfile::Short32.validate(32, false).is_ok());
        assert!(NativeAuditProfile::Stream128.validate(128, false).is_ok());
        assert!(NativeAuditProfile::Stream128.validate(128, true).is_err());
        assert!(NativeAuditProfile::Stream128.validate(32, false).is_err());
    }
    #[test]
    fn context32_and_context64_references_are_separate_compiled_bounded_profiles() {
        for n in [
            8192, 8896, 11264, 32768, 32769, 33472, 35840, 35841, 65536, 65537, 67584, 69632, 69633,
        ] {
            let ids = vec![20; n];
            for profile in [
                NativeAuditProfile::Long128,
                NativeAuditProfile::Context32Long128,
                NativeAuditProfile::Context64Long128,
            ] {
                let r = FreshReference {
                    token_ids: &ids,
                    max_new_tokens: 128,
                    audit_directory: None,
                    draft_audit_directory: Some(Path::new("private")),
                    profile,
                };
                let expected = match profile {
                    NativeAuditProfile::Context32Long128 => {
                        cfg!(feature = "qwen35-session-context32-audit")
                            && (32769..=35840).contains(&n)
                    }
                    NativeAuditProfile::Context64Long128 => {
                        cfg!(feature = "qwen35-session-context64-audit")
                            && (65537..=69632).contains(&n)
                    }
                    _ => {
                        cfg!(feature = "qwen35-draft-private-state") && (8193..=11264).contains(&n)
                    }
                };
                assert_eq!(r.validate().is_ok(), expected);
            }
        }
        assert!(NativeAuditProfile::Context32Long128
            .validate(128, false)
            .is_err());
        assert!(NativeAuditProfile::Context32Long128
            .validate(32, true)
            .is_err());
        assert!(NativeAuditProfile::Context64Long128
            .validate(128, false)
            .is_err());
        assert!(NativeAuditProfile::Context64Long128
            .validate(32, true)
            .is_err());
    }

    #[test]
    fn near192_reference_preserves_full_verify_reserve_and_other_profiles() {
        assert_eq!(196608 - (128 + B - 2), 196474);
        for n in [
            8193, 33472, 66240, 194560, 194561, 195264, 196474, 196475, 196480, 196608,
        ] {
            let ids = vec![20; n];
            for profile in [
                NativeAuditProfile::Near192Long128,
                NativeAuditProfile::Context32Long128,
                NativeAuditProfile::Context64Long128,
            ] {
                let r = FreshReference {
                    token_ids: &ids,
                    max_new_tokens: 128,
                    audit_directory: None,
                    draft_audit_directory: Some(Path::new("private")),
                    profile,
                };
                let allowed = match profile {
                    NativeAuditProfile::Near192Long128 => {
                        cfg!(feature = "qwen35-session-near192-audit")
                            && (194561..=196474).contains(&n)
                    }
                    NativeAuditProfile::Context32Long128 => {
                        cfg!(feature = "qwen35-session-context32-audit")
                            && (32769..=35840).contains(&n)
                    }
                    NativeAuditProfile::Context64Long128 => {
                        cfg!(feature = "qwen35-session-context64-audit")
                            && (65537..=69632).contains(&n)
                    }
                    _ => unreachable!(),
                };
                assert_eq!(r.validate().is_ok(), allowed);
            }
        }
        assert!(NativeAuditProfile::Near192Long128
            .validate(128, false)
            .is_err());
        assert!(NativeAuditProfile::Near192Long128
            .validate(32, true)
            .is_err());
    }

    #[test]
    fn exact_reference_bounds_do_not_expand_http_prompt_admission() {
        let mut ids = vec![20; 8832];
        assert!(FreshReference {
            token_ids: &ids,
            max_new_tokens: 32,
            audit_directory: None,
            draft_audit_directory: None,
            profile: NativeAuditProfile::Short32,
        }
        .validate()
        .is_ok());
        for n in [0, 8192, 11265] {
            assert!(FreshReference {
                token_ids: &vec![20; n],
                max_new_tokens: 32,
                audit_directory: None,
                draft_audit_directory: None,
                profile: NativeAuditProfile::Short32,
            }
            .validate()
            .is_err());
        }
        for n in [0, 1, 33, 1024] {
            assert!(FreshReference {
                token_ids: &ids,
                max_new_tokens: n,
                audit_directory: None,
                draft_audit_directory: None,
                profile: NativeAuditProfile::Short32,
            }
            .validate()
            .is_err());
        }
        ids[0] = V as u32;
        assert!(FreshReference {
            token_ids: &ids,
            max_new_tokens: 32,
            audit_directory: None,
            draft_audit_directory: None,
            profile: NativeAuditProfile::Short32,
        }
        .validate()
        .is_err());
        assert!(top16_validation::Mode::Resident
            .validate_prompt(8832)
            .is_err());
    }
    #[test]
    fn fast_session_path_switch_is_exact_and_fail_closed() {
        use std::ffi::OsStr;
        assert!(!fast_session_path_switch(None).unwrap());
        assert!(fast_session_path_switch(Some(OsStr::new("1"))).unwrap());
        for value in ["", "0", "true", " 1", "1 "] {
            assert!(fast_session_path_switch(Some(OsStr::new(value))).is_err());
        }
    }

    #[test]
    fn generation_after_suffix_keeps_teacher_forcing_out_of_model_outputs() {
        let mut l = ledger();
        let inputs = l.preflight(&l.consumed, &[30, 31]).unwrap();
        l.publish_append(&inputs, 8, 4, 40).unwrap();
        assert!(l.commit_generated(&[41], 9, 5).is_err());
        l.generated.push(40);
        l.pending_published = true;
        l.commit_generated(&[41, 42], 10, 5).unwrap();
        assert_eq!(l.consumed, [10, 11, 19, 20, 21, 22, 30, 31, 40, 41]);
        assert_eq!(l.generated, [19, 20, 21, 22, 40, 41, 42]);
        assert_eq!(l.pending, 42);
    }
    #[test]
    fn fast_session_path_only_activates_for_unaudited_live_boundaries() {
        for requested in [false, true] {
            for live in [false, true] {
                for audit in [false, true] {
                    assert_eq!(
                        fast_session_path_active(requested, live, audit),
                        requested && live && !audit,
                    );
                }
            }
        }
    }
    #[test]
    fn restored_live_policy_uses_only_current_explicit_launch_switches() {
        use std::ffi::OsStr;
        let enabled = Some(OsStr::new("1"));
        for session in [None, enabled] {
            for features in [None, enabled] {
                for replay_head in [None, enabled] {
                    let policy = LiveContinuationPolicy::from_switches(session, features, replay_head)
                        .unwrap();
                    assert_eq!(policy.fast_session_path, session.is_some());
                    assert_eq!(policy.fast_feature_identity, features.is_some());
                    assert_eq!(policy.replay_head_elision, replay_head.is_some());
                }
            }
        }
        for value in ["", "0", "true", " 1", "1 "] {
            let bad = Some(OsStr::new(value));
            assert!(LiveContinuationPolicy::from_switches(bad, None, None).is_err());
            assert!(LiveContinuationPolicy::from_switches(None, bad, None).is_err());
            assert!(LiveContinuationPolicy::from_switches(None, None, bad).is_err());
        }
    }
    #[test]
    fn live_projection_executor_is_retained_across_both_policy_directions() {
        for current in [false, true] {
            for requested in [false, true] {
                let mut calls = 0;
                let executor = rebind_live_projection_executor(
                    Some(23u32),
                    true,
                    current != requested,
                    |executor| {
                        assert_eq!(*executor, 23);
                        calls += 1;
                        Ok(())
                    },
                )
                .unwrap();
                assert_eq!(executor, Some(23));
                assert_eq!(calls, usize::from(current != requested));
            }
        }
    }
    #[test]
    fn live_projection_executor_rejects_missing_required_core() {
        for changed in [false, true] {
            assert!(rebind_live_projection_executor::<u32>(None, true, changed, |_| {
                panic!("missing core must not be rebound")
            })
            .is_err());
            assert!(rebind_live_projection_executor::<u32>(None, false, changed, |_| {
                panic!("scalar owner has no canonical core")
            })
            .unwrap()
            .is_none());
        }
    }
    #[test]
    fn live_projection_executor_propagates_rebind_failure_before_publication() {
        assert!(rebind_live_projection_executor(Some(23u32), true, true, |_| {
            Err(invalid("mock projection rebind failure"))
        })
        .is_err());
    }
    #[test]
    fn session_transaction_retains_extent_generation_and_anchor_checks() {
        let tx = json!({
            "emitted_token_ids":[2,3], "committed_token_count":2,
            "cursor_before":10, "cursor_after":12,
            "generation_before":4, "generation_after":5, "next_anchor":3
        });
        assert_eq!(validate_committed_transaction(12, 5, 3, &tx).unwrap(), [2, 3]);
        for (key, bad) in [
            ("emitted_token_ids", json!([])),
            ("emitted_token_ids", json!([2, V])),
            ("emitted_token_ids", json!([2, 4])),
            ("emitted_token_ids", json!(vec![3; B + 1])),
            ("committed_token_count", json!(1)),
            ("committed_token_count", json!(true)),
            ("cursor_before", json!(11)),
            ("cursor_before", json!(u64::MAX)),
            ("cursor_after", json!(11)),
            ("generation_before", json!(3)),
            ("generation_before", json!(u64::MAX)),
            ("generation_after", json!(6)),
            ("next_anchor", json!(2)),
        ] {
            let mut mutated = tx.clone();
            mutated[key] = bad;
            assert!(validate_committed_transaction(12, 5, 3, &mutated).is_err(), "{key}");
        }
        for key in tx.as_object().unwrap().keys() {
            let mut missing = tx.clone();
            missing.as_object_mut().unwrap().remove(key);
            assert!(validate_committed_transaction(12, 5, 3, &missing).is_err(), "{key}");
        }
        assert!(validate_committed_transaction(13, 5, 3, &tx).is_err());
        assert!(validate_committed_transaction(12, 6, 3, &tx).is_err());
        assert!(validate_committed_transaction(12, 5, 4, &tx).is_err());
    }
    #[test]
    fn fast_initial_path_requires_explicit_policy_and_audit_false() {
        let output = json!({"stop_reason":"max_new_tokens","max_new_tokens":2});
        for tx in [
            json!({"session_input_path_policy":"metadata_only_fast_v1"}),
            json!({"session_input_path_audited":null,"session_input_path_policy":"metadata_only_fast_v1"}),
            json!({"session_input_path_audited":false}),
            json!({"session_input_path_audited":false,"session_input_path_policy":"unknown"}),
        ] {
            assert!(!initial_transaction_path_valid(2, &[271, 272], 3, 2, &[tx], &output, true));
        }
    }
    #[test]
    fn device_path_requires_anchor_and_excludes_bonus() {
        assert!(validate_committed_path(1, &[2, 3, 4], &[1, 2, 3]).is_ok());
        assert!(validate_committed_path(1, &[2], &[1]).is_ok());
        for actual in [vec![], vec![2, 3, 4], vec![1, 2], vec![1, 2, 3, 4]] {
            assert!(validate_committed_path(1, &[2, 3, 4], &actual).is_err());
        }
    }
    #[test]
    fn initial_output_anchor_is_counted_exactly_once() {
        let l = Ledger::initial("s", &[10, 11], &[19, 20, 21, 22], 5, 3, 32).unwrap();
        assert_eq!(l.generated.len(), 4);
        assert_eq!(l.consumed.iter().filter(|&&t| t == 19).count(), 1);
        assert!(Ledger::initial("s", &[10, 11], &[19, 20, 21, 22], 6, 3, 32).is_err());
    }
    #[test]
    fn initial_anchor_only_terminal_requires_exact_budget_or_eos_and_unadvanced_state() {
        let output = json!({"stop_reason":"max_new_tokens","max_new_tokens":1,"eos_token_ids":[248046]});
        assert!(initial_transaction_path_valid(2, &[271], 2, 1, &[], &output, false));
        for (tokens, cursor, generation) in [
            (vec![], 2, 1), (vec![271, 272], 3, 2), (vec![271], 3, 1),
            (vec![271], 2, 2), (vec![271], 2, 0),
        ] {
            assert!(!initial_transaction_path_valid(2, &tokens, cursor, generation, &[], &output, false));
        }
        for invalid in [
            json!({"stop_reason":"max_new_tokens","max_new_tokens":2}),
            json!({"stop_reason":"max_new_tokens","max_new_tokens":true}),
            json!({"stop_reason":"unknown","max_new_tokens":1}),
            json!({"stop_reason":"eos","eos_token_ids":[248046]}),
            json!({"stop_reason":"eos"}),
        ] {
            assert!(!initial_transaction_path_valid(2, &[271], 2, 1, &[], &invalid, false));
        }
        let eos = json!({"stop_reason":"eos","max_new_tokens":32,"eos_token_ids":[248046]});
        assert!(initial_transaction_path_valid(2, &[248046], 2, 1, &[], &eos, false));
        assert!(!initial_transaction_path_valid(2, &[271, 272], 3, 2,
            &[json!({"session_input_path_audited":false})], &output, false));
        assert!(initial_transaction_path_valid(2, &[271, 272], 3, 2,
            &[json!({"session_input_path_audited":false,
                "session_input_path_policy":"metadata_only_fast_v1"})], &output, true));
        assert!(!initial_transaction_path_valid(2, &[271, 272], 3, 2,
            &[json!({"session_input_path_audited":false,
                "session_input_path_policy":"metadata_only_fast_v1"})], &output, false));
    }
    #[test]
    fn anchor_only_initial_ledger_consumes_pending_token_exactly_once_on_append() {
        let mut l = Ledger::initial("one", &[10, 11], &[271], 2, 1, 32).unwrap();
        assert_eq!(l.consumed, [10, 11]);
        assert_eq!(l.generated, [271]);
        assert_eq!(l.pending, 271);
        let suffix = l.preflight(&l.consumed, &[30]).unwrap();
        assert_eq!(suffix, [271, 30]);
        l.publish_append(&suffix, 4, 2, 40).unwrap();
        assert_eq!(l.consumed, [10, 11, 271, 30]);
        assert_eq!(l.generated, [271]);
        assert!(!l.pending_published);
        assert_eq!(l.preflight(&l.consumed, &[31]).unwrap(), [31]);
    }
    fn ledger() -> Ledger {
        Ledger::initial("s0", &[10, 11], &[19, 20, 21, 22], 5, 3, 32).unwrap()
    }
    #[test]
    fn pending_output_is_not_consumed() {
        let l = ledger();
        assert_eq!(l.consumed, [10, 11, 19, 20, 21]);
        assert_eq!(l.pending, 22);
        assert_eq!(l.generated, [19, 20, 21, 22]);
        assert!(l.pending_published);
    }
    #[test]
    fn suffix_consumes_previous_visible_anchor_once() {
        let mut l = ledger();
        let inputs = l.preflight(&l.consumed, &[30, 31]).unwrap();
        assert_eq!(inputs, [22, 30, 31]);
        l.publish_append(&inputs, 8, 4, 40).unwrap();
        assert_eq!(l.preflight(&l.consumed, &[32]).unwrap(), [32]);
        assert_eq!(l.generated, [19, 20, 21, 22]);
        assert_eq!(l.suffix_ranges, [(6, 8)]);
        assert!(!l.pending_published);
    }
    #[test]
    fn exact_history_not_only_digest() {
        let l = ledger();
        assert!(l.preflight(&[10, 11, 20, 22], &[30]).is_err());
        assert!(l.preflight(&[10, 11, 20], &[30]).is_err());
    }
    #[test]
    fn pre_submit_rejection_preserves_state() {
        let l = ledger();
        let before = l.clone();
        for suffix in [vec![], vec![V as u32], vec![1; 2049]] {
            assert!(l.preflight(&l.consumed, &suffix).is_err());
            assert_eq!(l, before);
        }
    }
    #[test]
    fn capacity_reserves_full_verify_lookahead() {
        let mut l = ledger();
        l.capacity = 15;
        assert!(l.preflight(&l.consumed, &[30]).is_ok());
        assert!(l.preflight(&l.consumed, &[30, 31]).is_err());
    }
    #[test]
    fn generation_and_revision_exhaustion_reject() {
        let mut l = ledger();
        l.generation = u64::MAX;
        assert!(l.preflight(&l.consumed, &[30]).is_err());
        l.generation = 3;
        l.revision = u64::MAX;
        assert!(l.preflight(&l.consumed, &[30]).is_err());
    }
    #[test]
    fn malformed_initial_ledger_rejects() {
        for id in ["", "../s", "s x"] {
            assert!(Ledger::initial(id, &[1], &[3, 2], 2, 1, 32).is_err());
        }
        assert!(Ledger::initial("s", &[1], &[], 1, 1, 32).is_err());
        assert!(Ledger::initial("s", &[1], &[3, 2], 3, 1, 32).is_err());
        assert!(Ledger::initial("s", &[1], &[3, V as u32], 2, 1, 32).is_err());
    }
    #[test]
    fn publication_requires_exact_cursor_generation_and_pending() {
        let l = ledger();
        for (inputs, cursor, generation) in [
            (vec![22, 30], 6, 4),
            (vec![22, 30], 7, 5),
            (vec![23, 30], 7, 4),
        ] {
            assert!(l
                .clone()
                .publish_append(&inputs, cursor, generation, 40)
                .is_err());
        }
    }
    #[test]
    fn token_digest_is_order_and_extent_sensitive() {
        assert_ne!(token_hash(&[1, 2]), token_hash(&[2, 1]));
        assert_ne!(token_hash(&[1]), token_hash(&[1, 0]));
    }

    #[test]
    fn resident_reference_and_stream_require_initialized_history_and_keep_reserve() {
        assert!(NativeAuditProfile::ResidentStream128.validate_consumer(true, true).is_ok());
        assert!(NativeAuditProfile::ResidentStream128.validate_consumer(false, true).is_err());
        assert!(NativeAuditProfile::ResidentStream128.validate_consumer(true, false).is_err());
        assert!(NativeAuditProfile::ResidentStream128.validate(128, true).is_err());
        for n in [0, 1, 16385, 33003, 194561, 196474, 196475, 196608] {
            let ids = vec![1; n];
            let reference = FreshReference { token_ids: &ids, max_new_tokens: 128,
                audit_directory: None, draft_audit_directory: None,
                profile: NativeAuditProfile::ResidentLong128 };
            assert_eq!(reference.validate().is_ok(), (1..=196474).contains(&n));
        }
    }

    #[test]
    fn resident_request_budget_does_not_weaken_legacy_or_pending_token_contracts() {
        let p = NativeAuditProfile::ResidentStreamBudget;
        for n in [1, 2, 7, 8, 9, 127, 128, 129, 256, 512, 1024] {
            assert!(p.validate(n, false).is_ok());
            assert!(p.validate(n, true).is_err());
            assert_eq!(NativeAuditProfile::ResidentStream128.validate(n, false).is_ok(), n == 128);
        }
        for n in [0, 1025, usize::MAX] { assert!(p.validate(n, false).is_err()); }
        assert!(p.validate_consumer(true, true).is_ok());
        assert!(p.validate_consumer(false, true).is_err());
        assert!(p.validate_consumer(true, false).is_err());
        let l = Ledger::initial("one", &[1, 2, 3], &[10], 3, 1, 64).unwrap();
        assert_eq!(l.consumed, [1, 2, 3]);
        assert_eq!(l.pending, 10);
        assert!(l.pending_published);
        assert_eq!(l.preflight(&l.consumed, &[11]).unwrap(), [10, 11]);
    }
}
