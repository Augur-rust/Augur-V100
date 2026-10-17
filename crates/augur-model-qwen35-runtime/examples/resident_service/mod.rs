//! Opt-in serial lifecycle gate. Fresh requests only until continuation passes.
//! Stdout is a bounded JSONL transport; no product or session-reuse claims.
use super::*;
#[cfg(feature = "qwen35-session-continuation")]
pub(super) mod session_smoke;
#[cfg(feature = "qwen35-session-continuation")]
pub(super) mod growing_session_smoke;
#[cfg(feature = "qwen35-session-continuation")]
pub(super) mod exact_reference;
#[cfg(feature = "qwen35-session-continuation")]
pub(super) mod session_cancel;
#[cfg(feature = "qwen35-session-continuation")]
pub(super) mod session_http;
#[cfg(feature = "qwen35-session-continuation")]
mod cancellable_input;
#[cfg(all(feature = "qwen35-session-continuation", feature = "qwen35-session-checkpoint", windows))]
pub(super) mod input_prefix_probe;
#[cfg(feature = "qwen35-session-continuation")]
pub(super) mod transition_smoke;
use serde::Deserialize;
use std::{
    collections::BTreeSet,
    io::{BufRead, Read},
    time::Instant,
};

const MAX_COMMAND_BYTES: u64 = 8 * 1024 * 1024;
const MAX_REQUESTS: usize = 64;

#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum Command {
    Generate { request_id: String, prompt: String },
    Shutdown {},
}

fn parse_command(bytes: &[u8]) -> Result<Command, Box<dyn Error>> {
    if bytes.len() as u64 > MAX_COMMAND_BYTES {
        return Err("command exceeds bound".into());
    }
    let value: Command = serde_json::from_slice(bytes)?;
    if let Command::Generate { request_id, prompt } = &value {
        if request_id.is_empty()
            || request_id.len() > 64
            || !request_id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
            || prompt.is_empty()
            || prompt.len() > 4 * 1024 * 1024
        {
            return Err("invalid request identifier or prompt extent".into());
        }
    }
    Ok(value)
}

struct Tee {
    file: BufWriter<std::fs::File>,
    output: io::Stdout,
}
impl Write for Tee {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.file.write_all(bytes)?;
        self.output.write_all(bytes)?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()?;
        self.output.flush()
    }
}

fn options(
    generation: &std::path::Path,
) -> Result<Qwen35DraftTargetRollingOptions, Box<dyn Error>> {
    let request = Qwen35DraftTargetRequestOptions {
        generation_config_path: generation.to_owned(),
        max_new_tokens: 1024,
        long_generation_budget: true,
        retain_full_accept_state: false,
        audit_batched_target_verify: false,
        profile_batched_target_verify: false,
        native_quantized_batch_ffn: false,
        audit_batched_numerics: false,
        canonical_batched_gdn_scalars: false,
        native_quantized_batch_attention: false,
        canonical_batched_norm: false,
        canonical_batched_full_core: false,
        batched_target_verify: false,
        audit_committed_state: false,
        reuse_batched_plan: false,
        share_target_rope: true,
        prefill_shared_softmax: false,
        prefill_dgemm_scores: true,
        batched_prompt_capture: false,
        profile_prefill_sparse: false,
        request_position_budget: true,
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
        target_split_rope: true,
    };
    let mut value = Qwen35DraftTargetRollingOptions {
        product_mode: true,
        incremental_draft_context: false,
        validate_draft_context_reuse: false,
        request: Some(request),
    };
    enable_canonical_batch_profile(&mut value)?;
    Ok(value)
}

pub(super) fn serve(
    mut config: Qwen35AutoregressiveCudaConfig,
    q8_path: PathBuf,
    draft_config_path: PathBuf,
    generation_config_path: PathBuf,
    tokenizer: Tokenizer,
    tokenizer_sha256: String,
    eos_token_ids: Vec<u32>,
    generation_config_bytes: Vec<u8>,
    evidence: PathBuf,
) -> Result<(), Box<dyn Error>> {
    if env::var("AUGUR_QWEN35_TOP16_VALIDATION_MODE").as_deref() != Ok("resident") {
        return Err("resident service requires distinct resident validation mode".into());
    }
    // All diagnostic/profile switches are rejected by the runtime admission.
    // Directory creation never overwrites an earlier run's evidence.
    fs::create_dir(&evidence)?;
    let vision_identity = config
        .vision_safetensors_path
        .as_deref()
        .map(inspect_qwen35_vision_payload)
        .transpose()?;
    let baseline = augur_backend::cuda_device_allocation_snapshot();
    let open_started = Instant::now();
    let mut owner = Qwen35ResidentDraftCuda::open_with_target_and_vision(
        &config,
        &q8_path,
        &draft_config_path,
    )?;
    let open_wall_ms = open_started.elapsed().as_secs_f64() * 1000.0;
    let retained = augur_backend::cuda_device_allocation_snapshot();
    let ready = json!({
        "schema":"qwen35-resident-text-service-v1", "event":"ready",
        "pid":std::process::id(), "owner_open_count":1, "owner_open_wall_ms":open_wall_ms,
        "resident":allocation_json(retained), "capacity":196608, "context_rows":2048,
        "max_new_tokens":1024, "max_requests":MAX_REQUESTS,
        "same_session_continuation":false, "vision_forward_executed":false,
        "product_ready":false, "source_identity":config.source_commit
    });
    fs::write(
        evidence.join("ready.json"),
        serde_json::to_vec_pretty(&ready)?,
    )?;
    write_frame(&mut io::stdout(), ready)?;
    let outcome: Result<usize, Box<dyn Error>> = (|| {
        let stdin = io::stdin();
        let mut input = stdin.lock();
        let mut identifiers = BTreeSet::new();
        loop {
            let mut line = Vec::new();
            let count = input
                .by_ref()
                .take(MAX_COMMAND_BYTES + 1)
                .read_until(b'\n', &mut line)?;
            if count == 0 {
                return Ok(identifiers.len());
            }
            if line.len() as u64 > MAX_COMMAND_BYTES {
                return Err("oversized service command".into());
            }
            let command = match parse_command(&line) {
                Ok(value) => value,
                Err(error) => {
                    write_frame(
                        &mut io::stdout(),
                        json!({"event":"rejected","before_submit":true,"detail":error.to_string()}),
                    )?;
                    continue;
                }
            };
            let Command::Generate { request_id, prompt } = command else {
                return Ok(identifiers.len());
            };
            if identifiers.len() >= MAX_REQUESTS || identifiers.contains(&request_id) {
                write_frame(
                    &mut io::stdout(),
                    json!({"event":"rejected","before_submit":true,"detail":"duplicate id or service request bound"}),
                )?;
                continue;
            }
            let prompt_ids = tokenizer
                .encode(prompt.as_str(), false)
                .map_err(|e| e.to_string())?;
            if ![8192, 32768, 65536, 194560].contains(&prompt_ids.len()) {
                write_frame(
                    &mut io::stdout(),
                    json!({"event":"rejected","before_submit":true,"detail":"prompt outside lifecycle gate ladder"}),
                )?;
                continue;
            }
            identifiers.insert(request_id.clone());
            config.prompt_text = prompt;
            let mut identity = json!({"prompt_text":config.prompt_text,"max_new_tokens":1024,
                "generation_config_sha256":digest(&generation_config_bytes),
                "tokenizer_sha256":tokenizer_sha256,"context_rows":2048,"capacity":196608,
                "resident_service":{"schema":"qwen35-resident-text-service-v1","request_id":request_id,
                    "request_index":identifiers.len()-1,"same_session_continuation":false}});
            bind_stream_rope_identity(&mut identity, true, true);
            bind_stream_vision_identity(&mut identity, vision_identity.as_ref());
            identity["vision_residency"]["lifetime"] = json!("resident_service_owner_v1");
            bind_stream_prefill_identity(&mut identity, true, true)?;
            bind_stream_generation_identity(&mut identity, true);
            let request_sha = digest(&serde_json::to_vec(&identity)?);
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
                request_sha,
                eos_token_ids.clone(),
                tokenizer_sha256.clone(),
            )?);
            let sink: &Qwen35VerifiedCommitSink<'_> =
                &|commit| stream.borrow_mut().push_commit(commit);
            let mut report = owner.run_request(
                &config,
                2048,
                1024,
                config.vision_safetensors_path.as_deref(),
                options(&generation_config_path)?,
                Some(sink),
            )?;
            let after = augur_backend::cuda_device_allocation_snapshot();
            if after.live_bytes != retained.live_bytes
                || after.live_allocations != retained.live_allocations
            {
                return Err("request failed to return to immutable resident baseline".into());
            }
            validate_stream_vision_report(&report, vision_identity.as_ref())?;
            validate_stream_prefill_report(&report, true)?;
            validate_stream_generation_report(&report, true)?;
            report["resident_service"] = json!({"request":identity,"pid":std::process::id(),
                "owner_open_count":1,"request_index":identifiers.len()-1,
                "after_request":allocation_json(after),"same_session_continuation":false,"product_ready":false});
            let stop = report["request_output"]["stop_reason"]
                .as_str()
                .ok_or("missing terminal")?;
            let visible: Vec<u32> =
                serde_json::from_value(report["request_output"]["visible_token_ids"].clone())?;
            let mut stream = stream.into_inner();
            stream.prepare_terminal_text(stop)?;
            if stream.visible_token_ids != visible {
                return Err("resident stream/report token mismatch".into());
            }
            let bytes = serde_json::to_vec(&report)?;
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(evidence.join(format!("{request_id}.report.json")))?;
            file.write_all(&bytes)?;
            file.flush()?;
            stream.finish(stop, digest(&bytes))?;
        }
    })();
    drop(owner);
    let after = augur_backend::cuda_device_allocation_snapshot();
    let cleanup_ok = after.live_bytes == baseline.live_bytes
        && after.live_allocations == baseline.live_allocations
        && after
            .successful_allocations
            .checked_sub(baseline.successful_allocations)
            == after
                .successful_frees
                .checked_sub(baseline.successful_frees);
    let summary = json!({"schema":"qwen35-resident-text-service-v1","event":"service_closed",
        "success":outcome.is_ok() && cleanup_ok,"cleanup_ok":cleanup_ok,
        "before":allocation_json(baseline),"after":allocation_json(after),
        "completed_requests":outcome.as_ref().ok(),"error":outcome.as_ref().err().map(|e|e.to_string()),
        "product_ready":false,"same_session_continuation":false});
    fs::write(
        evidence.join("cleanup.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    write_frame(&mut io::stdout(), summary)?;
    outcome?;
    if !cleanup_ok {
        return Err("resident service final cleanup mismatch".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn request_protocol_does_not_implicitly_admit_sessions_or_overrides() {
        assert!(
            parse_command(br#"{"op":"generate","request_id":"turn-1","prompt":"hello"}"#).is_ok()
        );
        for bytes in [
            br#"{"op":"append","request_id":"a","prompt":"x"}"#.as_slice(),
            br#"{"op":"generate","request_id":"a","prompt":"x","capacity":8192}"#,
            br#"{"op":"generate","request_id":"../x","prompt":"x"}"#,
            br#"{"op":"generate","request_id":"a","prompt":""}"#,
            br#"{"op":"generate","request_id":"a","request_id":"b","prompt":"x"}"#,
        ] {
            assert!(parse_command(bytes).is_err());
        }
        assert!(parse_command(br#"{"op":"shutdown"}"#).is_ok());
        assert!(parse_command(br#"{"op":"shutdown","prompt":"x"}"#).is_err());
    }
}
