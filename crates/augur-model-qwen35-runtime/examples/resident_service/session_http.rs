//! Explicit canonical8K single-session transport experiment. Defaults stay fresh-only.
use super::*;
use augur_model_qwen35_runtime::resident_session_profile::ResidentSessionProfile as Profile;

#[cfg(all(windows, feature = "qwen35-session-checkpoint"))]
#[path = "checkpoint_http.rs"]
mod checkpoint_http;

#[cfg(all(windows, feature = "qwen35-session-checkpoint"))]
#[path = "input_prefix_http.rs"]
mod input_prefix_http;

#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum SessionCommand {
    SessionOpen {
        request_id: String,
        session_id: String,
        prompt: String,
        max_new_tokens: Option<usize>,
    },
    SessionAppend {
        request_id: String,
        session_id: String,
        revision: u64,
        exact_history: Vec<u32>,
        suffix: String,
        max_new_tokens: Option<usize>,
    },
    SessionClose {
        request_id: String,
        session_id: String,
        revision: u64,
    },
    SessionSave {
        request_id: String,
        session_id: String,
        revision: u64,
        checkpoint_id: String,
    },
    SessionRestore {
        request_id: String,
        session_id: String,
        checkpoint_id: String,
        http_manifest_sha256: String,
    },
    Shutdown {},
}
fn valid_id(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 64
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}
#[cfg(test)]
fn parse_profile(
    bytes: &[u8],
    context32: bool,
    context64: bool,
    near192: bool,
) -> Result<SessionCommand, Box<dyn Error>> {
    parse_with_growing(bytes, context32, context64, near192, false)
}
#[cfg(test)]
fn parse_with_growing(bytes: &[u8], context32: bool, context64: bool, near192: bool, growing: bool)
    -> Result<SessionCommand, Box<dyn Error>>
{
    parse_for_profile(bytes, Profile::from_legacy(context32, context64, near192, growing)?)
}
fn parse_for_profile(bytes: &[u8], profile: Profile) -> Result<SessionCommand, Box<dyn Error>> {
    profile.validate_compiled()?;
    let bounds = profile.history_bounds();
    if bytes.len() as u64 > MAX_COMMAND_BYTES {
        return Err("session command extent".into());
    }
    let c: SessionCommand = serde_json::from_slice(bytes)?;
    match &c {
        SessionCommand::SessionOpen {
            request_id,
            session_id,
            prompt,
            max_new_tokens,
        } => {
            profile.output_budget(*max_new_tokens, true)?;
            if !valid_id(request_id)
                || !valid_id(session_id)
                || prompt.is_empty()
                || prompt.len() > 4 * 1024 * 1024
            {
                return Err("invalid session open".into());
            }
        }
        SessionCommand::SessionAppend {
            request_id,
            session_id,
            revision,
            exact_history,
            suffix,
            max_new_tokens,
        } => {
            profile.output_budget(*max_new_tokens, false)?;
            if !valid_id(request_id)
                || !valid_id(session_id)
                || *revision == 0
                || !bounds.contains(&exact_history.len())
                || exact_history.iter().any(|&t| t >= 248320)
                || suffix.is_empty()
                || suffix.len() > 16384
            {
                return Err("invalid bounded session append".into());
            }
        }
        SessionCommand::SessionClose {
            request_id,
            session_id,
            revision,
        } => {
            if !valid_id(request_id) || !valid_id(session_id) || *revision == 0 {
                return Err("invalid session close".into());
            }
        }
        SessionCommand::Shutdown {} => {}
        SessionCommand::SessionSave {
            request_id,
            session_id,
            revision,
            checkpoint_id,
        } => {
            if !cfg!(all(windows, feature = "qwen35-session-checkpoint"))
                || !valid_id(request_id)
                || !valid_id(session_id)
                || *revision == 0
                || !checkpoint_digest(checkpoint_id)
            {
                return Err("invalid or disabled session save".into());
            }
        }
        SessionCommand::SessionRestore {
            request_id,
            session_id,
            checkpoint_id,
            http_manifest_sha256,
        } => {
            if !cfg!(all(windows, feature = "qwen35-session-checkpoint"))
                || !valid_id(request_id)
                || !valid_id(session_id)
                || !checkpoint_digest(checkpoint_id)
                || !checkpoint_digest(http_manifest_sha256)
            {
                return Err("invalid or disabled session restore".into());
            }
        }
    }
    Ok(c)
}
fn checkpoint_digest(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
#[cfg(test)]
fn parse(bytes: &[u8]) -> Result<SessionCommand, Box<dyn Error>> {
    parse_profile(bytes, false, false, false)
}
fn save(path: &std::path::Path, value: &Value) -> Result<String, Box<dyn Error>> {
    let mut bound = value.clone();
    augur_model_qwen35_runtime::build_identity::annotate(&mut bound);
    let raw = serde_json::to_vec_pretty(&bound)?;
    let mut f = OpenOptions::new().create_new(true).write(true).open(path)?;
    f.write_all(&raw)?;
    f.sync_all()?;
    Ok(digest(&raw))
}
// Opt-in diagnostic timing only. Preserve create-new, exact bytes, fsync and SHA.
fn save_profiled(path: &std::path::Path, value: &Value) -> Result<(String, Value), Box<dyn Error>> {
    let start = Instant::now();
    let mut bound = value.clone();
    augur_model_qwen35_runtime::build_identity::annotate(&mut bound);
    let cloned = Instant::now();
    let raw = serde_json::to_vec_pretty(&bound)?;
    let encoded = Instant::now();
    let mut f = OpenOptions::new().create_new(true).write(true).open(path)?;
    f.write_all(&raw)?;
    let written = Instant::now();
    f.sync_all()?;
    let synced = Instant::now();
    let hash = digest(&raw);
    let hashed = Instant::now();
    let bytes = raw.len();
    drop(f); drop(raw); drop(bound);
    let dropped = Instant::now();
    let ms = |a: Instant, b: Instant| b.duration_since(a).as_secs_f64() * 1000.0;
    Ok((hash, json!({"bytes":bytes,"clone_annotate_ms":ms(start,cloned),
        "serialize_ms":ms(cloned,encoded),"open_write_ms":ms(encoded,written),
        "fsync_ms":ms(written,synced),"sha256_ms":ms(synced,hashed),
        "close_drop_ms":ms(hashed,dropped),"enclosing_ms":ms(start,dropped),
        "durability_changed":false})))
}

// The final report is no longer read after publication. Consume it instead of
// cloning the entire tree and encode the same JSON values without indentation.
// This changes receipt bytes (and hence SHA), never the fsync-before-EOF order.
fn save_compact_owned(path: &std::path::Path, mut bound: Value) -> Result<(String, Value), Box<dyn Error>> {
    let start = Instant::now();
    augur_model_qwen35_runtime::build_identity::annotate(&mut bound);
    let annotated = Instant::now();
    let raw = serde_json::to_vec(&bound)?;
    let encoded = Instant::now();
    let mut f = OpenOptions::new().create_new(true).write(true).open(path)?;
    f.write_all(&raw)?;
    let written = Instant::now();
    f.sync_all()?;
    let synced = Instant::now();
    let hash = digest(&raw);
    let hashed = Instant::now();
    let bytes = raw.len();
    drop(f); drop(raw); drop(bound);
    let dropped = Instant::now();
    let ms = |a: Instant, b: Instant| b.duration_since(a).as_secs_f64()*1000.0;
    Ok((hash, json!({"bytes":bytes,"clone_annotate_ms":ms(start,annotated),
        "serialize_ms":ms(annotated,encoded),"open_write_ms":ms(encoded,written),
        "fsync_ms":ms(written,synced),"sha256_ms":ms(synced,hashed),
        "close_drop_ms":ms(hashed,dropped),"enclosing_ms":ms(start,dropped),
        "owned_compact":true,"full_value_clone_performed":false,"durability_changed":false})))
}
fn compact_receipt_switch(value: Option<&std::ffi::OsStr>, resident: bool) -> Result<bool, Box<dyn Error>> {
    match value {
        None => Ok(false),
        Some(v) if v == "1" && resident => Ok(true),
        _ => Err("compact receipt requires resident profile and value1".into()),
    }
}
struct Active {
    id: String,
    checkpoint: Value,
    consumed: Vec<u32>,
    #[cfg(all(windows, feature = "qwen35-session-checkpoint"))]
    turns: Vec<augur_model_qwen35_runtime::session_checkpoint::http::PublishedTurn>,
}
fn active_matches(a: &Active, id: &str, revision: u64) -> bool {
    a.id == id && a.checkpoint["revision"].as_u64() == Some(revision)
}
fn metadata(a: &Active) -> Value {
    let mut value = json!({"session_id":a.id,"revision":a.checkpoint["revision"],
    "consumed_token_count":a.consumed.len(),"consumed_input_digest":a.checkpoint["consumed_input_digest"],
    "pending_token":a.checkpoint["pending_token"],"pending_token_published":a.checkpoint["pending_token_published"],
    "checkpoint_sha256":a.checkpoint["checkpoint_sha256"],"capacity":196608,"product_ready":false});
    if let Some(history) = a.checkpoint.get("initialized_history") {
        value["initialized_history"] = history.clone();
    }
    value
}
#[cfg(test)]
fn initial_extent_admitted(tokens: usize, fixed_tokens: usize, growing: bool) -> bool {
    if growing {
        (1..=8192).contains(&tokens)
    } else {
        tokens == fixed_tokens
    }
}
fn reject(detail: &str) -> Result<(), Box<dyn Error>> {
    write_frame(
        &mut io::stdout(),
        json!({"event":"rejected","before_submit":true,"detail":detail}),
    )?;
    Ok(())
}

fn top16_session_admitted(
    profile: Profile,
    mode: Option<&std::ffi::OsStr>,
    batch: Option<&std::ffi::OsStr>,
    compiled: bool,
) -> bool {
    mode == Some(std::ffi::OsStr::new("resident"))
        && match batch {
            None => true,
            Some(value) => compiled && profile == Profile::Resident && value == "1",
        }
}

fn batched_head_rope_admitted(
    profile: Profile, value: Option<&std::ffi::OsStr>, compiled: bool, persistence: bool,
) -> bool {
    value.is_none() || value.is_some_and(|v| v == "1"
        && profile == Profile::Resident && compiled && !persistence)
}

pub(crate) fn serve(
    mut config: Qwen35AutoregressiveCudaConfig,
    q8: PathBuf,
    draft_config: PathBuf,
    generation: PathBuf,
    tokenizer: Tokenizer,
    tokenizer_sha: String,
    eos: Vec<u32>,
    evidence: PathBuf,
    profile: Profile,
    checkpoint_profile: Option<(PathBuf, String)>,
    cooperative_owner_cancel: bool,
    input_prefix_tokens: Option<usize>,
) -> Result<(), Box<dyn Error>> {
    profile.validate_compiled()?;
    let resident = profile == Profile::Resident;
    let terminal_stage_timing = match env::var_os("AUGUR_QWEN35_TERMINAL_STAGE_TIMING") {
        None => false,
        Some(v) if v == "1" && resident => true,
        Some(_) => return Err("terminal stage timing requires resident profile and value1".into()),
    };
    let compact_receipt = compact_receipt_switch(
        env::var_os("AUGUR_QWEN35_COMPACT_OWNED_RECEIPT").as_deref(), resident)?;
    let growing = profile.uses_initialized_history();
    let context32 = profile == Profile::Context32k;
    let context64 = profile == Profile::Context64k;
    let near192 = profile == Profile::Near192k;
    let initial_tokens = profile.initial_prompt_tokens().unwrap_or(8192);
    let max_history = profile.maximum_history_tokens();
    let schema = profile.schema();
    // The resident service is normally pinned to the canonical per-row head.
    // Permit the batch top-16 kernel only as an explicitly opt-in resident
    // experiment; the driver still applies the full request-shape admission
    // and the default product path remains unchanged.
    if !top16_session_admitted(
        profile,
        env::var_os("AUGUR_QWEN35_TOP16_VALIDATION_MODE").as_deref(),
        env::var_os("AUGUR_QWEN35_DRAFT_TOP16_BATCH").as_deref(),
        cfg!(feature = "qwen35-draft-top16-batch"),
    ) {
        return Err("separate canonical session admission".into());
    }
    if env::var_os("AUGUR_QWEN35_DRAFT_TOP16_BATCH").is_some()
        && (checkpoint_profile.is_some() || input_prefix_tokens.is_some())
    {
        return Err("batched top16 experiment excludes persistence and input-prefix cache".into());
    }
    if !batched_head_rope_admitted(profile,
        env::var_os("AUGUR_QWEN35_PREFILL_BATCHED_HEAD_ROPE").as_deref(),
        cfg!(feature = "qwen35-fused-head-norm-rope"),
        checkpoint_profile.is_some() || input_prefix_tokens.is_some()) {
        return Err("batched head RoPE requires live resident opt-in without persistence/input-prefix cache".into());
    }
    #[cfg(all(windows, feature = "qwen35-session-checkpoint"))]
    let mut persistence = checkpoint_profile
        .as_ref()
        .map(|(p, sha)| checkpoint_http::Persistence::open(p, sha, &evidence, &tokenizer_sha))
        .transpose()?;
    #[cfg(not(all(windows, feature = "qwen35-session-checkpoint")))]
    if checkpoint_profile.is_some() {
        return Err("checkpoint service not compiled".into());
    }
    if input_prefix_tokens.is_some() && (!resident || checkpoint_profile.is_none()) {
        return Err("input-prefix cache requires resident service and pinned checkpoint profile".into());
    }
    #[cfg(all(windows, feature = "qwen35-session-checkpoint"))]
    let mut input_cache = input_prefix_tokens.map(|tokens| {
        let (path, sha) = checkpoint_profile.as_ref().ok_or("input-prefix profile missing")?;
        input_prefix_http::InputCache::open(path, sha, &evidence, &tokenizer_sha, tokens)
    }).transpose()?;
    #[cfg(not(all(windows, feature = "qwen35-session-checkpoint")))]
    if input_prefix_tokens.is_some() { return Err("input-prefix cache not compiled".into()); }
    fs::create_dir(&evidence)?;
    let baseline = augur_backend::cuda_device_allocation_snapshot();
    let started = Instant::now();
    let mut owner =
        Qwen35ResidentDraftCuda::open_with_target_and_vision(&config, &q8, &draft_config)?;
    let input = if cooperative_owner_cancel {
        let token = augur_model_qwen35_runtime::session_cancellation::SessionCancellationToken::new();
        let input = super::cancellable_input::CancellableInput::start(
            io::stdin(), MAX_COMMAND_BYTES, token.clone(),
        )?;
        owner.install_session_cancellation(token)?;
        Some(input)
    } else {
        None
    };
    let retained = augur_backend::cuda_device_allocation_snapshot();
    let mut ready = json!({"schema":schema,"event":"ready","pid":std::process::id(),
        "owner_open_count":1,"owner_open_wall_ms":started.elapsed().as_secs_f64()*1000.,"resident":allocation_json(retained),
        "startup_timing":owner.startup_report(),
        "initial_prompt_tokens":initial_tokens,"maximum_history_tokens":max_history,"capacity":196608,"context_rows":2048,"continuation_limit":128,
        "same_session_continuation":true,"session_qualified":false,"product_ready":false,"vision_forward_executed":false,
        "numerical_identity":"native_full_canonical_prefill_audit_v1","state_seal_audit_in_wall":!resident,
        "state_seal_policy":if resident {"live_owner_no_payload_readback_v1"} else {"full_bytes_pipelined_dual_sha256_v1"}});
    if growing {
        ready["initial_prompt_tokens"] = Value::Null;
        ready["minimum_initial_prompt_tokens"] = json!(1);
        ready["maximum_initial_prompt_tokens"] = json!(profile.maximum_initial_prompt_tokens());
        ready["growing_initialized_history"] = json!(true);
        ready["uninitialized_tail_hashed"] = json!(false);
    }
    augur_model_qwen35_runtime::build_identity::annotate(&mut ready);
    ready["request_output_budget_supported"] = json!(resident);
    ready["continuation_max_output_limit"] = profile.capabilities()["continuation_max_output_limit"].clone();
    ready["cooperative_input_cancellation"] = json!(cooperative_owner_cancel);
    #[cfg(all(windows, feature = "qwen35-session-checkpoint"))]
    if let Some(cache) = &input_cache { ready["input_prefix_cache"] = cache.ready(); }
    #[cfg(all(windows, feature = "qwen35-session-checkpoint"))]
    if let Some(p) = &persistence {
        ready["checkpoint_service"] = p.ready();
    }
    save(&evidence.join("ready.json"), &ready)?;
    write_frame(&mut io::stdout(), ready)?;
    let mut active = None::<Active>;
    let outcome = (|| -> Result<usize, Box<dyn Error>> {
        let stdin = io::stdin();
        // Do not lock stdin on the execution thread when the independent ingress
        // reader owns it: that reader must observe EOF during silent prefill.
        let mut legacy_input = if input.is_none() { Some(stdin.lock()) } else { None };
        let mut ids = BTreeSet::new();
        let mut session_ids = BTreeSet::new();
        loop {
            let raw = if let Some(input) = &input {
                match input.next()? {
                    Some(raw) => raw,
                    None => return Ok(ids.len()),
                }
            } else {
                let mut raw = Vec::new();
                let count = legacy_input.as_mut().ok_or("missing session input")?
                    .by_ref().take(MAX_COMMAND_BYTES + 1).read_until(b'\n', &mut raw)?;
                if count == 0 { return Ok(ids.len()); }
                raw
            };
            if raw.len() as u64 > MAX_COMMAND_BYTES {
                return Err("oversized session command".into());
            }
            let command = match parse_for_profile(&raw, profile) {
                Ok(c) => c,
                Err(e) => {
                    reject(&e.to_string())?;
                    continue;
                }
            };
            if matches!(command, SessionCommand::Shutdown {}) {
                return Ok(ids.len());
            }
            let request_id = match &command {
                SessionCommand::SessionOpen { request_id, .. }
                | SessionCommand::SessionAppend { request_id, .. }
                | SessionCommand::SessionClose { request_id, .. }
                | SessionCommand::SessionSave { request_id, .. }
                | SessionCommand::SessionRestore { request_id, .. } => request_id.clone(),
                _ => unreachable!(),
            };
            if ids.len() >= 64 || ids.contains(&request_id) {
                reject("duplicate request id or service bound")?;
                continue;
            }
            #[cfg(all(windows, feature = "qwen35-session-checkpoint"))]
            if matches!(
                &command,
                SessionCommand::SessionSave { .. } | SessionCommand::SessionRestore { .. }
            ) {
                let Some(p) = persistence.as_mut() else {
                    reject("checkpoint service disabled at startup")?;
                    continue;
                };
                if let Some(result) = p.execute(
                    &command,
                    checkpoint_http::LiveSession {
                        owner: &mut owner,
                        config: &config,
                        generation: &generation,
                        tokenizer: &tokenizer,
                        active: &mut active,
                        session_ids: &mut session_ids,
                        profile,
                    },
                )? {
                    ids.insert(request_id.clone());
                    save(&evidence.join(format!("{request_id}.report.json")), &result)?;
                    write_frame(&mut io::stdout(), result)?;
                }
                continue;
            }
            if let SessionCommand::SessionClose {
                session_id,
                revision,
                ..
            } = &command
            {
                let Some(a) = active.as_ref() else {
                    reject("no active session")?;
                    continue;
                };
                if !active_matches(a, session_id, *revision) {
                    reject("session identity/revision mismatch")?;
                    continue;
                }
                owner.close_native_session(&a.checkpoint)?;
                active = None;
                let after = augur_backend::cuda_device_allocation_snapshot();
                if after.live_bytes != retained.live_bytes
                    || after.live_allocations != retained.live_allocations
                {
                    return Err("session close resident leak".into());
                }
                ids.insert(request_id.clone());
                let result = json!({"event":"session_closed","request_id":request_id,"session_id":session_id,"resident":allocation_json(after),"owner_open_count":1,"product_ready":false});
                save(&evidence.join(format!("{request_id}.report.json")), &result)?;
                write_frame(&mut io::stdout(), result)?;
                continue;
            }
            // Complete all caller-level semantic validation before begin or CUDA mutation.
            let output_ceiling = match &command {
                SessionCommand::SessionOpen { max_new_tokens, .. } => profile.output_budget(*max_new_tokens, true)?,
                SessionCommand::SessionAppend { max_new_tokens, .. } => profile.output_budget(*max_new_tokens, false)?,
                _ => unreachable!(),
            };
            let (session_id, revision, mut exact, suffix) = match &command {
                SessionCommand::SessionOpen {
                    session_id, prompt, ..
                } => {
                    if active.is_some() || session_ids.contains(session_id) {
                        reject("session active or id already consumed")?;
                        continue;
                    }
                    let ids = tokenizer
                        .encode(prompt.as_str(), false)
                        .map_err(|e| e.to_string())?
                        .get_ids()
                        .to_vec();
                    if !profile.admits_initial(ids.len()) {
                        reject("canonical session selected initial-context bounds")?;
                        continue;
                    }
                    config.prompt_text = prompt.clone();
                    session_ids.insert(session_id.clone());
                    (session_id.clone(), None, ids, Vec::new())
                }
                SessionCommand::SessionAppend {
                    session_id,
                    revision,
                    exact_history,
                    suffix,
                    ..
                } => {
                    let Some(a) = active.as_ref() else {
                        reject("no active session")?;
                        continue;
                    };
                    if !active_matches(a, session_id, *revision) || exact_history != &a.consumed {
                        reject("session identity/revision/exact history mismatch")?;
                        continue;
                    }
                    #[cfg(all(windows, feature = "qwen35-session-checkpoint"))]
                    if persistence.is_some()
                        && a.turns.len()
                            >= augur_model_qwen35_runtime::session_checkpoint::http::MAX_TURNS
                    {
                        reject("checkpoint visible ledger turn quota")?;
                        continue;
                    }
                    let suffix = tokenizer
                        .encode(suffix.as_str(), false)
                        .map_err(|e| e.to_string())?
                        .get_ids()
                        .to_vec();
                    let reserve = profile.continuation_reserve(output_ceiling).ok_or("invalid continuation reserve")?;
                    if profile.append_end_with_budget(a.consumed.len(), suffix.len(), output_ceiling).is_none()
                    {
                        if (near192 || resident) && !suffix.is_empty() && suffix.len() <= 2048 {
                            // Bind the actual tokenized overflow to this unmodified owner.
                            write_frame(
                                &mut io::stdout(),
                                json!({
                                    "event":"rejected","before_submit":true,
                                    "detail":"bounded session suffix/position gate",
                                    "request_id":request_id,"command_sha256":digest(&raw),
                                    "checkpoint":metadata(a),"suffix_tokens":suffix.len(),
                                    "reserved_positions":reserve,
                                    "requested_output_ceiling":output_ceiling,
                                    "projected_position_limit":a.consumed.len()+1+suffix.len()+reserve,
                                    "maximum_history_tokens":max_history
                                }),
                            )?;
                        } else {
                            reject("bounded session suffix/position gate")?;
                        }
                        continue;
                    }
                    (
                        session_id.clone(),
                        Some(*revision),
                        exact_history.clone(),
                        suffix,
                    )
                }
                _ => unreachable!(),
            };
            ids.insert(request_id.clone());
            let request_schema = if resident {
                "qwen35-resident-text-session-request-v1"
            } else if growing {
                "qwen35-initialized-growing-session-request-v1"
            } else {
                "qwen35-canonical8k-session-request-v1"
            };
            let identity = json!({"schema":request_schema,"request_id":request_id,"session_id":session_id,
                "revision":revision,"command_sha256":digest(&raw),"numerical_identity":"native_full_canonical_prefill_audit_v1",
                "max_new_tokens":output_ceiling,
                "capacity":196608,"context_rows":2048,"owner_open_count":1,"product_ready":false});
            let file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(evidence.join(format!("{request_id}.jsonl")))?;
            let stream = RefCell::new(CommittedStream::begin(
                Tee {
                    file: BufWriter::new(file),
                    output: io::stdout(),
                },
                tokenizer.clone(),
                request_id.clone(),
                digest(&serde_json::to_vec(&identity)?),
                eos.clone(),
                tokenizer_sha.clone(),
            )?);
            stream.borrow_mut().profile_terminal = terminal_stage_timing;
            let sink: &Qwen35VerifiedCommitSink<'_> = &|c| stream.borrow_mut().push_commit(c);
            let request_started = Instant::now();
            #[cfg(all(windows, feature = "qwen35-session-checkpoint"))]
            let cached = if revision.is_none() {
                input_cache.as_mut().map(|cache| cache.open_turn(&mut owner, &config,
                    &generation, &exact, &session_id, output_ceiling, sink)).transpose()?.flatten()
            } else { None };
            #[cfg(not(all(windows, feature = "qwen35-session-checkpoint")))]
            let cached = None::<Value>;
            let mut report = if let Some(report) = cached {
                report
            } else if revision.is_none() && resident {
                owner.open_resident_text_session_with_budget(&config, &session_id, options(&generation)?, sink, output_ceiling)?
            } else if revision.is_none() && growing {
                owner.open_native_growing_session_audit(
                    &config,
                    &session_id,
                    options(&generation)?,
                    sink,
                )?
            } else if revision.is_none() && near192 {
                owner.open_native_session_near192_audit(
                    &config,
                    &session_id,
                    options(&generation)?,
                    sink,
                )?
            } else if revision.is_none() && context32 {
                owner.open_native_session_context32_audit(
                    &config,
                    &session_id,
                    options(&generation)?,
                    sink,
                )?
            } else if revision.is_none() && context64 {
                owner.open_native_session_context64_audit(
                    &config,
                    &session_id,
                    options(&generation)?,
                    sink,
                )?
            } else if revision.is_none() {
                owner.open_native_session_with_full_canonical_prefill_audit(
                    &config,
                    &session_id,
                    options(&generation)?,
                    sink,
                )?
            } else {
                let a = active.as_ref().ok_or("missing session")?;
                let appended = owner.append_native_session(&a.checkpoint, &exact, &suffix)?;
                exact.push(
                    a.checkpoint["pending_token"]
                        .as_u64()
                        .ok_or("pending token")? as u32,
                );
                exact.extend_from_slice(&suffix);
                save(
                    &evidence.join(format!("{request_id}.append.json")),
                    &appended,
                )?;
                if resident {
                    owner.decode_resident_text_session_stream_with_budget(&appended["checkpoint"], sink, output_ceiling)?
                } else if growing {
                    owner
                        .decode_native_growing_session_stream_audit(&appended["checkpoint"], sink)?
                } else if context32 {
                    owner.decode_native_session_context32_stream_audit(
                        &appended["checkpoint"],
                        sink,
                    )?
                } else if context64 {
                    owner.decode_native_session_context64_stream_audit(
                        &appended["checkpoint"],
                        sink,
                    )?
                } else if near192 {
                    owner
                        .decode_native_session_near192_stream_audit(&appended["checkpoint"], sink)?
                } else {
                    owner.decode_native_session_stream_audit(&appended["checkpoint"], sink)?
                }
            };
            let native_return_timing = terminal_stage_timing.then(|| {
                let s = stream.borrow();
                json!({"native_return_after_last_text_ms":s.last_text_published
                    .map(|t|t.elapsed().as_secs_f64()*1000.0),
                    "native_return_text_sequence":s.sequence.checked_sub(1)})
            });
            let output = if revision.is_none() {
                &report["request_output"]
            } else {
                &report
            };
            let tokens: Vec<u32> = serde_json::from_value(output["token_ids"].clone())?;
            let visible: Vec<u32> = serde_json::from_value(output["visible_token_ids"].clone())?;
            let stop = output["stop_reason"]
                .as_str()
                .ok_or("missing stop")?
                .to_owned();
            exact.extend_from_slice(&tokens[..tokens.len() - 1]);
            let checkpoint = if revision.is_none() {
                report["session_checkpoint"].clone()
            } else {
                report["checkpoint"].clone()
            };
            if checkpoint["cursor"].as_u64() != Some(exact.len() as u64)
                || checkpoint["pending_token"] != json!(tokens.last())
            {
                return Err("service exact ledger conservation".into());
            }
            active = Some(Active {
                id: session_id,
                checkpoint,
                consumed: exact,
                #[cfg(all(windows, feature = "qwen35-session-checkpoint"))]
                turns: active.take().map(|a| a.turns).unwrap_or_default(),
            });
            let a = active.as_ref().unwrap();
            let native_request_wall_ms = request_started.elapsed().as_secs_f64() * 1000.;
            report["session_service"] = json!({"identity":identity,"pid":std::process::id(),"owner_open_count":1,
                "checkpoint":metadata(a),"native_request_wall_ms_with_state_seal":native_request_wall_ms,
                "same_session_continuation":revision.is_some(),"prefix_recomputed":false,"weights_reloaded":false,
                "initial_prefill_executed":revision.is_none() && report["input_prefix_cache"]["hit"] != true,
                "input_prefix_tokens_reused":report["input_prefix_cache"]["prefix_tokens_reused"].as_u64().unwrap_or(0),
                "prior_prefix_tokens_recomputed":0,"context32_audit":context32,"context64_audit":context64,"near192_audit":near192,
                "after_request":allocation_json(augur_backend::cuda_device_allocation_snapshot()),"http_qualified":false,"product_ready":false});
            let mut stream = stream.into_inner();
            stream.prepare_terminal_text(&stop)?;
            if stream.visible_token_ids != visible {
                return Err("session stream token ledger mismatch".into());
            }
            let (report_sha, terminal_timing) = if let Some(mut timing) = native_return_timing {
                timing["before_receipt_after_last_text_ms"] = json!(stream.last_text_published
                    .map(|t| t.elapsed().as_secs_f64()*1000.0));
                timing["before_receipt_text_sequence"] = json!(stream.sequence.checked_sub(1));
                let path = evidence.join(format!("{request_id}.report.json"));
                let (sha, io) = if compact_receipt { save_compact_owned(&path, report)? }
                    else { save_profiled(&path, &report)? };
                timing["receipt"] = io;
                (sha, Some(timing))
            } else {
                let path = evidence.join(format!("{request_id}.report.json"));
                let sha = if compact_receipt { save_compact_owned(&path, report)?.0 }
                    else { save(&path, &report)? };
                (sha, None)
            };
            #[cfg(all(windows, feature = "qwen35-session-checkpoint"))]
            let published = persistence
                .as_ref()
                .map(|p| {
                    checkpoint_http::published_turn(
                        p,
                        &stream,
                        &tokens,
                        &stop,
                        &report_sha,
                        &a.checkpoint,
                    )
                })
                .transpose()?;
            stream.finish_session(&stop, report_sha, metadata(a), terminal_timing)?;
            #[cfg(all(windows, feature = "qwen35-session-checkpoint"))]
            if let Some(turn) = published {
                active
                    .as_mut()
                    .ok_or("publication session missing")?
                    .turns
                    .push(turn);
            }
        }
    })();
    drop(owner);
    let after = augur_backend::cuda_device_allocation_snapshot();
    let clean = after.live_bytes == baseline.live_bytes
        && after.live_allocations == baseline.live_allocations
        && after.successful_allocations - baseline.successful_allocations
            == after.successful_frees - baseline.successful_frees;
    let mut summary = json!({"schema":schema,"event":"service_closed","success":outcome.is_ok()&&clean,
        "cleanup_ok":clean,"after":allocation_json(after),"error":outcome.as_ref().err().map(|e|e.to_string()),
        "owner_open_count":1,"same_session_continuation":true,"session_qualified":false,"product_ready":false});
    augur_model_qwen35_runtime::build_identity::annotate(&mut summary);
    save(&evidence.join("cleanup.json"), &summary)?;
    write_frame(&mut io::stdout(), summary)?;
    outcome?;
    if !clean {
        return Err("session service cleanup".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn compact_owned_receipt_preserves_json_hash_and_create_new() {
        let root = std::env::temp_dir().join(format!("qwen35-compact-receipt-{}-{}",
            std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::create_dir(&root).unwrap();
        let value = json!({"text":"中文 / English\n```rust\nfn main() {}\n```",
            "nested":[1,2,3],"exact_u64":u64::MAX,"signed_zero":-0.0,"product_ready":false});
        let old = root.join("old.json"); let new = root.join("new.json");
        save(&old, &value).unwrap();
        let (sha, timing) = save_compact_owned(&new, value.clone()).unwrap();
        let raw = std::fs::read(&new).unwrap();
        let old_raw = std::fs::read(&old).unwrap();
        assert_eq!(serde_json::from_slice::<Value>(&raw).unwrap(), serde_json::from_slice::<Value>(&old_raw).unwrap());
        assert_eq!(sha, digest(&raw));
        assert!(raw.len() < old_raw.len());
        assert_eq!(timing["bytes"], raw.len());
        assert_eq!(timing["durability_changed"], false);
        assert_eq!(timing["full_value_clone_performed"], false);
        assert!(save_compact_owned(&new, value).is_err());
        assert_eq!(raw, std::fs::read(&new).unwrap());
        std::fs::remove_file(old).unwrap(); std::fs::remove_file(new).unwrap(); std::fs::remove_dir(root).unwrap();
    }
    #[test]
    fn compact_receipt_is_explicit_resident_only() {
        use std::ffi::OsStr;
        for resident in [false, true] {
            assert!(!compact_receipt_switch(None, resident).unwrap());
            assert_eq!(compact_receipt_switch(Some(OsStr::new("1")), resident).is_ok(), resident);
            for v in ["", "0", "true", "yes", " 1"] {
                assert!(compact_receipt_switch(Some(OsStr::new(v)), resident).is_err());
            }
        }
    }
    #[test]
    fn profiled_receipt_keeps_bytes_hash_and_create_new_contract() {
        let root = std::env::temp_dir().join(format!("qwen35-receipt-profile-{}-{}",
            std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::create_dir(&root).unwrap();
        let value = json!({"text":"中文 / English", "nested":[1,2,3],"product_ready":false});
        let a = root.join("plain.json"); let b = root.join("timed.json");
        let plain = save(&a, &value).unwrap();
        let (timed, times) = save_profiled(&b, &value).unwrap();
        assert_eq!(plain, timed);
        let raw = std::fs::read(&a).unwrap();
        assert_eq!(raw, std::fs::read(&b).unwrap());
        assert_eq!(times["bytes"].as_u64().unwrap(), raw.len() as u64);
        assert_eq!(times["durability_changed"], false);
        let keys = ["clone_annotate_ms","serialize_ms","open_write_ms","fsync_ms","sha256_ms","close_drop_ms"];
        let sum: f64 = keys.iter().map(|k| times[k].as_f64().unwrap()).sum();
        assert!((sum-times["enclosing_ms"].as_f64().unwrap()).abs()<1e-6);
        assert!(save_profiled(&b, &value).is_err());
        assert_eq!(raw, std::fs::read(&b).unwrap());
        std::fs::remove_file(a).unwrap(); std::fs::remove_file(b).unwrap(); std::fs::remove_dir(root).unwrap();
    }
    #[test]
    fn head_rope_batch_excludes_persistence_and_dedicated_audit_profiles() {
        use std::ffi::OsStr;
        for profile in [Profile::Resident, Profile::Context32k, Profile::Context64k, Profile::Near192k] {
            for compiled in [false, true] {
                for persistence in [false, true] {
                    assert!(batched_head_rope_admitted(profile, None, compiled, persistence));
                    assert_eq!(batched_head_rope_admitted(profile, Some(OsStr::new("1")), compiled, persistence),
                        profile == Profile::Resident && compiled && !persistence);
                    for value in ["", "0", "true", "1 "] {
                        assert!(!batched_head_rope_admitted(profile, Some(OsStr::new(value)), compiled, persistence));
                    }
                }
            }
        }
    }
    #[test]
    fn resident_top16_batch_is_explicit_feature_bound_and_profile_scoped() {
        use super::*;
        use std::ffi::OsStr;
        let mode = Some(OsStr::new("resident"));
        assert!(top16_session_admitted(Profile::Resident, mode, None, false));
        assert!(top16_session_admitted(Profile::Resident, mode, Some(OsStr::new("1")), true));
        assert!(!top16_session_admitted(Profile::Resident, mode, Some(OsStr::new("1")), false));
        for profile in [Profile::Context32k, Profile::Context64k, Profile::Near192k] {
            assert!(top16_session_admitted(profile, mode, None, false));
            assert!(!top16_session_admitted(profile, mode, Some(OsStr::new("1")), true));
        }
        for value in ["", "0", "true", "1 "] {
            assert!(!top16_session_admitted(Profile::Resident, mode, Some(OsStr::new(value)), true));
        }
        for value in [None, Some(OsStr::new("state")), Some(OsStr::new("client"))] {
            assert!(!top16_session_admitted(Profile::Resident, value, Some(OsStr::new("1")), true));
        }
    }
    use super::*;
    #[test]
    fn resident_command_budget_is_bounded_and_legacy_modes_stay_fixed() {
        for (mut body, maximum) in [
            (json!({"op":"session_open","request_id":"r","session_id":"s","prompt":"hi"}), 1024),
            (json!({"op":"session_append","request_id":"r","session_id":"s","revision":1,"exact_history":[1,2],"suffix":"x"}), 1024),
        ] {
            for n in [1, 2, 7, 8, 9, maximum] {
                body["max_new_tokens"] = json!(n);
                let raw = serde_json::to_vec(&body).unwrap();
                assert!(parse_for_profile(&raw, Profile::Resident).is_ok());
                assert!(parse_for_profile(&raw, Profile::Growing).is_err());
            }
            for value in [json!(0), json!(-1), json!(true), json!(maximum + 1), json!("1")] {
                body["max_new_tokens"] = value;
                assert!(parse_for_profile(&serde_json::to_vec(&body).unwrap(), Profile::Resident).is_err());
            }
        }
    }
    #[test]
    fn session_protocol_strict_and_not_legacy_generate() {
        assert!(
            parse(br#"{"op":"session_open","request_id":"r","session_id":"s","prompt":"x"}"#)
                .is_ok()
        );
        for raw in [
            br#"{"op":"generate","request_id":"r","prompt":"x"}"#.as_slice(),
            br#"{"op":"session_close","request_id":"r","session_id":"s","revision":0}"#,
            br#"{"op":"session_open","request_id":"r","session_id":"s","prompt":"x","capacity":1}"#,
            br#"{"op":"shutdown","capacity":1}"#,
        ] {
            assert!(parse(raw).is_err());
        }
    }
    #[test]
    fn context32_and_context64_transport_extents_are_explicit_and_not_legacy() {
        for n in [
            8192, 16384, 32767, 32768, 33472, 36864, 36865, 65535, 65536, 65537, 69632, 69633,
        ] {
            let raw = serde_json::to_vec(&json!({"op":"session_append","request_id":"r","session_id":"s","revision":1,"exact_history":vec![1;n],"suffix":"x"})).unwrap();
            assert_eq!(
                parse_profile(&raw, true, false, false).is_ok(),
                cfg!(feature = "qwen35-session-context32-audit") && (32768..=36864).contains(&n)
            );
            assert_eq!(
                parse_profile(&raw, false, true, false).is_ok(),
                cfg!(feature = "qwen35-session-context64-audit") && (65536..=69632).contains(&n)
            );
            assert_eq!(
                parse_profile(&raw, false, false, false).is_ok(),
                (8192..=16384).contains(&n)
            );
        }
    }

    #[test]
    fn near192_transport_is_optin_and_legacy_extents_stay_closed() {
        for n in [8192, 32768, 65536, 194559, 194560, 195264, 196608, 196609] {
            let raw=serde_json::to_vec(&json!({"op":"session_append","request_id":"r","session_id":"s","revision":1,"exact_history":vec![1;n],"suffix":"x"})).unwrap();
            assert_eq!(
                parse_profile(&raw, false, false, true).is_ok(),
                cfg!(feature = "qwen35-session-near192-audit") && (194560..=196608).contains(&n)
            );
            assert!(parse_profile(&raw, false, true, true).is_err());
            assert!(parse_profile(&raw, true, false, true).is_err());
        }
    }

    #[test]
    fn session_revision_is_bound_to_exact_owner() {
        let a = Active {
            id: "s".into(),
            checkpoint: json!({"revision":2}),
            consumed: vec![1],
            #[cfg(all(windows, feature = "qwen35-session-checkpoint"))]
            turns: Vec::new(),
        };
        assert!(active_matches(&a, "s", 2));
        assert!(!active_matches(&a, "s", 1));
        assert!(!active_matches(&a, "other", 2));
    }

    #[test]
    fn checkpoint_protocol_rejects_paths_client_history_and_implicit_authority() {
        let enabled = cfg!(all(windows, feature = "qwen35-session-checkpoint"));
        for op in ["session_save", "session_restore"] {
            let mut v =
                json!({"op":op,"request_id":"r","session_id":"s","checkpoint_id":"a".repeat(64)});
            if op == "session_save" {
                v["revision"] = json!(1);
            } else {
                v["http_manifest_sha256"] = json!("b".repeat(64));
            }
            assert_eq!(parse(&serde_json::to_vec(&v).unwrap()).is_ok(), enabled);
            for key in ["path", "exact_history", "product_ready", "owner_id"] {
                let mut bad = v.clone();
                bad[key] = json!(true);
                assert!(parse(&serde_json::to_vec(&bad).unwrap()).is_err());
            }
            v["checkpoint_id"] = json!("../escape");
            assert!(parse(&serde_json::to_vec(&v).unwrap()).is_err());
        }
    }

    #[test]
    fn growing_transport_is_explicit_and_cannot_mix_fixed_profiles() {
        for n in [0, 1, 7, 31, 2047, 2048, 2049, 8192, 8193, 16384, 16385] {
            assert_eq!(
                initial_extent_admitted(n, 8192, true),
                (1..=8192).contains(&n)
            );
            assert_eq!(initial_extent_admitted(n, 8192, false), n == 8192);
            let raw = serde_json::to_vec(&json!({"op":"session_append","request_id":"r",
                "session_id":"s","revision":1,"exact_history":vec![1;n],"suffix":"x"}))
            .unwrap();
            assert_eq!(
                parse_with_growing(&raw, false, false, false, true).is_ok(),
                (1..=16384).contains(&n)
            );
            for (c32, c64, near) in [
                (true, false, false),
                (false, true, false),
                (false, false, true),
            ] {
                assert!(parse_with_growing(&raw, c32, c64, near, true).is_err());
            }
            if n < 8192 {
                assert!(parse(&raw).is_err());
            }
        }
    }

    #[test]
    fn resident_transport_uses_full_history_but_keeps_token_and_identity_validation() {
        for n in [0, 1, 16385, 33003, 194560, 196608, 196609] {
            let mut body = json!({"op":"session_append","request_id":"r","session_id":"s",
                "revision":1,"exact_history":vec![1;n],"suffix":"x"});
            let raw = serde_json::to_vec(&body).unwrap();
            assert_eq!(parse_for_profile(&raw, Profile::Resident).is_ok(), (1..=196608).contains(&n));
            if n > 16384 { assert!(parse_with_growing(&raw, false, false, false, true).is_err()); }
            body["revision"] = json!(0);
            assert!(parse_for_profile(&serde_json::to_vec(&body).unwrap(), Profile::Resident).is_err());
        }
    }
}
