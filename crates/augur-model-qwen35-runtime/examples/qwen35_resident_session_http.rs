//! Single-owner HTTP adapter for the Qwen35 resident session producer.
//!
//! The native CUDA producer remains authoritative for model execution and
//! session state.  This binary only translates bounded HTTP requests into the
//! producer's strict JSONL protocol and forwards committed frames as HTTP
//! chunked NDJSON.  It is deliberately fail-closed: a client disconnect or a
//! producer framing error poisons and terminates the owner instead of reusing
//! an uncertain CUDA/session state.

use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    env,
    error::Error,
    fs,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::PathBuf,
    process::{Child, ChildStdin, Command, Stdio},
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

const MAX_REQUEST_BYTES: usize = 8 * 1024 * 1024;
const MAX_FRAME_BYTES: usize = 1_048_576;
const MAX_PROMPT_BYTES: usize = 4 * 1024 * 1024;
const MAX_SUFFIX_BYTES: usize = 16 * 1024;
const MAX_CAPACITY: usize = 196_608;
const MAX_CONTEXT_ROWS: usize = 2_048;
const QUALIFIED_HEADER: &str = "false";
static NEXT_OWNER: AtomicU64 = AtomicU64::new(0);

fn reap_owned_child(child: &mut Child) -> std::io::Result<()> {
    if child.try_wait()?.is_none() {
        if let Err(error) = child.kill() {
            // A natural exit can race kill. Never wait forever after a failed
            // kill of a process whose exit has not been confirmed.
            if child.try_wait()?.is_none() {
                return Err(error);
            }
        }
        child.wait()?;
    }
    Ok(())
}

fn reap_or_stop_adapter(child: &mut Child) {
    if let Err(error) = reap_owned_child(child) {
        eprintln!(
            "QWEN35_OWNER_REAP_UNCONFIRMED pid={} error={error}",
            child.id()
        );
        // No replacement CUDA owner may start after unconfirmed containment.
        // The external coordinator must inspect/reap this exact recorded PID.
        std::process::exit(70);
    }
}

#[path = "resident_service/http_transport.rs"]
mod http_transport;
use http_transport::ProducerFrames;

#[path = "resident_service/http_teardown.rs"]
mod http_teardown;
const DISCONNECT_GRACE: Duration = Duration::from_secs(5);

use augur_model_qwen35_runtime::resident_session_profile::ResidentSessionProfile as Profile;

#[derive(Debug)]
struct Options {
    bind: String,
    producer: PathBuf,
    gguf: PathBuf,
    config: PathBuf,
    tokenizer: PathBuf,
    q8_gguf: PathBuf,
    dflash_config: PathBuf,
    generation_config: PathBuf,
    vision_safetensors: PathBuf,
    evidence_root: PathBuf,
    device: i32,
    profile: Profile,
    checkpoint_profile: Option<(PathBuf, String)>,
    input_prefix_cache_tokens: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OpenRequest {
    request_id: String,
    session_id: String,
    prompt: String,
    max_new_tokens: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AppendRequest {
    request_id: String,
    session_id: String,
    revision: u64,
    exact_history: Vec<u32>,
    suffix: String,
    max_new_tokens: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CloseRequest {
    request_id: String,
    session_id: String,
    revision: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SaveRequest {
    request_id: String,
    session_id: String,
    revision: u64,
    checkpoint_id: String,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RestoreRequest {
    request_id: String,
    session_id: String,
    checkpoint_id: String,
    http_manifest_sha256: String,
}

fn checkpoint_digest(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

#[derive(Debug)]
struct ResidentOwner {
    child: Child,
    stdin: Option<ChildStdin>,
    frames: ProducerFrames,
    ready: Value,
}

impl ResidentOwner {
    fn spawn(options: &Options, client: &TcpStream) -> Result<Self, Box<dyn Error>> {
        http_transport::client_connected(client)?;
        if augur_model_qwen35_runtime::build_identity::SOURCE_COMMIT.is_some() {
            let probe = Command::new(&options.producer)
                .arg("--build-identity")
                .output()?;
            if !probe.status.success() || probe.stdout.len() > 65536 {
                return Err("producer build-identity preflight failed".into());
            }
            augur_model_qwen35_runtime::build_identity::validate_peer(
                &serde_json::from_slice(&probe.stdout)?,
                &augur_model_qwen35_runtime::build_identity::identity(),
            )
            .map_err(|e| e.to_owned())?;
        }
        let owner_id = NEXT_OWNER.fetch_add(1, Ordering::Relaxed);
        let evidence = options.evidence_root.join(format!("owner-{owner_id}"));
        let mut command = Command::new(&options.producer);
        if let Some(tokens) = options.input_prefix_cache_tokens {
            command.arg("--input-prefix-cache-tokens").arg(tokens.to_string());
        }
        if let Some((path, sha)) = &options.checkpoint_profile {
            command
                .arg("--checkpoint-profile")
                .arg(path)
                .arg("--checkpoint-profile-sha256")
                .arg(sha);
        }
        command
            .arg("--gguf")
            .arg(&options.gguf)
            .arg("--config")
            .arg(&options.config)
            .arg("--tokenizer")
            .arg(&options.tokenizer)
            .arg("--q8-gguf")
            .arg(&options.q8_gguf)
            .arg("--dflash-config")
            .arg(&options.dflash_config)
            .arg("--request-generation-config")
            .arg(&options.generation_config)
            .arg("--vision-safetensors")
            .arg(&options.vision_safetensors)
            .arg("--stream-output")
            .arg(&evidence)
            .arg(options.profile.producer_flag())
            .arg("--cooperative-owner-cancel")
            .arg("--target-split-rope")
            .arg("--share-target-rope")
            .arg("--canonical-batch-profile")
            .arg("--prefill-dgemm-scores")
            .arg("--long-generation-budget")
            .arg("--max-new-tokens")
            .arg("1024")
            .arg("--rolling-blocks")
            .arg("1024")
            .arg("--context-rows")
            .arg(MAX_CONTEXT_ROWS.to_string())
            .arg("--capacity")
            .arg(MAX_CAPACITY.to_string())
            .arg("--device")
            .arg(options.device.to_string())
            .env("AUGUR_QWEN35_TOP16_VALIDATION_MODE", "resident")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        let mut child = command.spawn()?;
        let stdin = child.stdin.take().ok_or("producer stdin unavailable")?;
        let stdout = child.stdout.take().ok_or("producer stdout unavailable")?;
        let frames = match ProducerFrames::spawn(stdout, MAX_FRAME_BYTES) {
            Ok(frames) => frames,
            Err(error) => {
                reap_or_stop_adapter(&mut child);
                return Err(error.into());
            }
        };
        let mut owner = Self {
            child,
            stdin: Some(stdin),
            frames,
            ready: Value::Null,
        };
        let ready = owner.read_frame_line(client)?;
        if let Err(error) = validate_ready(&ready, options.profile).and_then(|()| {
            validate_checkpoint_ready(
                &ready,
                options
                    .checkpoint_profile
                    .as_ref()
                    .map(|(_, sha)| sha.as_str()),
            )
        }).and_then(|()| validate_input_prefix_ready(&ready, options.input_prefix_cache_tokens)) {
            owner.terminate();
            return Err(error);
        }
        owner.ready = ready;
        Ok(owner)
    }

    fn read_frame_line(&mut self, client: &TcpStream) -> Result<Value, Box<dyn Error>> {
        self.frames.receive(client)
    }

    fn send_command(&mut self, command: &Value) -> Result<(), Box<dyn Error>> {
        let line = serde_json::to_vec(command)?;
        if line.len() > MAX_REQUEST_BYTES {
            return Err("session command exceeds frame bound".into());
        }
        let input = self.stdin.as_mut().ok_or("producer command pipe retired")?;
        input.write_all(&line)?;
        input.write_all(b"\n")?;
        input.flush()?;
        Ok(())
    }

    fn stream_until_boundary(
        &mut self,
        http: &mut TcpStream,
        close_boundary: bool,
        expected_request_id: Option<&str>,
        expected_session_id: Option<&str>,
        expected_checkpoint: Option<&Value>,
    ) -> Result<bool, Box<dyn Error>> {
        // `begin` owns sequence zero; committed text starts at sequence one.
        let mut next_sequence = 1_u64;
        let mut saw_begin = false;
        loop {
            let frame = self.read_frame_line(http)?;
            let event = frame
                .get("event")
                .and_then(Value::as_str)
                .ok_or("producer frame has no event")?;
            for key in ["product_ready", "session_qualified", "visible_tps"] {
                if let Some(value) = frame.get(key) {
                    if value != &Value::Bool(false) {
                        return Err(format!("producer claim {key} was not false").into());
                    }
                }
            }
            validate_frame_identity(&frame, expected_request_id, expected_session_id)?;
            if expected_checkpoint.is_some_and(|c| {
                matches!(c["op"].as_str(), Some("session_save" | "session_restore"))
            }) && !matches!(event, "session_saved" | "session_restored" | "rejected")
            {
                return Err("checkpoint command cannot produce generation or close frames".into());
            }
            match event {
                "session_saved" | "session_restored" => {
                    if saw_begin {
                        return Err("checkpoint terminal after generation begin".into());
                    }
                    validate_checkpoint_terminal(
                        &frame,
                        expected_checkpoint.ok_or("unexpected checkpoint terminal")?,
                        &self.ready,
                    )?;
                    write_chunk(http, &serde_json::to_vec(&frame)?)?;
                    http.flush()?;
                    return Ok(true);
                }
                "begin" => {
                    if saw_begin || frame.get("sequence").and_then(Value::as_u64) != Some(0) {
                        return Err("invalid producer begin frame".into());
                    }
                    saw_begin = true;
                }
                "text" => {
                    if !saw_begin
                        || frame.get("sequence").and_then(Value::as_u64) != Some(next_sequence)
                    {
                        return Err("invalid producer text sequence".into());
                    }
                }
                "terminal" => {
                    if !saw_begin
                        || frame.get("sequence").and_then(Value::as_u64) != Some(next_sequence)
                    {
                        return Err("invalid producer terminal sequence".into());
                    }
                    write_chunk(http, &serde_json::to_vec(&frame)?)?;
                    http.flush()?;
                    return Ok(true);
                }
                "rejected" => {
                    write_chunk(http, &serde_json::to_vec(&frame)?)?;
                    http.flush()?;
                    return Ok(false);
                }
                "session_closed" if close_boundary => {
                    write_chunk(http, &serde_json::to_vec(&frame)?)?;
                    http.flush()?;
                    return Ok(true);
                }
                "service_closed" if close_boundary => {
                    write_chunk(http, &serde_json::to_vec(&frame)?)?;
                    http.flush()?;
                    return Ok(true);
                }
                _ => return Err(format!("unexpected producer event {event}").into()),
            }
            write_chunk(http, &serde_json::to_vec(&frame)?)?;
            http.flush()?;
            if event == "text" {
                next_sequence = next_sequence
                    .checked_add(1)
                    .ok_or("producer sequence overflow")?;
            }
        }
    }

    fn terminate(&mut self) {
        self.frames.close_receiver();
        reap_or_stop_adapter(&mut self.child);
        self.frames.join_after_child_exit();
    }

    fn retire_disconnected(&mut self) -> http_teardown::ProcessRetirement {
        let retirement_started = std::time::Instant::now();
        // Removing the receiver releases a blocked frame sender. Its next frame
        // (or EOF) drops the stdout reader, so a subsequent native commit sees a
        // broken pipe and uses the existing drain/poison/drop path. Closing stdin
        // also releases an owner already waiting for the next command. Neither
        // operation is CUDA completion authority. The explicitly selected
        // cooperative stdin reader separately notifies safe native boundaries;
        // a force fallback is still containment, never native cleanup proof.
        self.stdin.take();
        self.frames.close_receiver();
        let result = match http_teardown::retire_after_disconnect(&mut self.child, DISCONNECT_GRACE)
        {
            Ok(result) => result,
            Err(error) => {
                eprintln!(
                    "QWEN35_OWNER_REAP_UNCONFIRMED pid={} error={error}",
                    self.child.id()
                );
                std::process::exit(70);
            }
        };
        self.frames.join_after_child_exit();
        eprintln!(
            "{}",
            json!({
                "event":"owner_process_retired_after_transport_failure",
                "pid":self.child.id(), "exit_code":result.exit.code(),
                "force_attempted":result.force_attempted,
                "grace_budget_ms":DISCONNECT_GRACE.as_millis(),
            "process_exit_wait_wall_ms":result.elapsed.as_secs_f64()*1000.0,
            "retirement_wall_ms":retirement_started.elapsed().as_secs_f64()*1000.0,
                "cuda_cleanup_inferred_from_exit":false,
                "product_ready":false, "qualified":false
            })
        );
        result
    }
}

fn validate_input_prefix_ready(frame: &Value, expected: Option<usize>) -> Result<(), Box<dyn Error>> {
    match expected {
        None if frame.get("input_prefix_cache").is_none() => Ok(()),
        Some(tokens) if frame["input_prefix_cache"]["enabled"] == true
            && frame["input_prefix_cache"]["prefix_tokens"] == json!(tokens)
            && frame["input_prefix_cache"]["slots"] == 1
            && frame["input_prefix_cache"]["scope"] == "single_owner_exact_input_prefix"
            && frame["input_prefix_cache"]["capture_on_first_eligible_open"] == true
            && frame["input_prefix_cache"]["equal_prompt_is_miss"] == true
            && frame["input_prefix_cache"]["cross_owner_reuse"] == false
            && frame["input_prefix_cache"]["qualified"] == false => Ok(()),
        _ => Err("producer input-prefix cache differs from explicit startup selection".into()),
    }
}

fn validate_ready(frame: &Value, profile: Profile) -> Result<(), Box<dyn Error>> {
    let strict_false = [
        "product_ready",
        "session_qualified",
        "vision_forward_executed",
    ];
    if frame.get("event").and_then(Value::as_str) != Some("ready")
        || frame.get("schema").and_then(Value::as_str) != Some(profile.schema())
        || frame.get("owner_open_count").and_then(Value::as_u64) != Some(1)
        || frame.get("initial_prompt_tokens") != Some(&json!(profile.initial_prompt_tokens()))
        || frame.get("maximum_history_tokens").and_then(Value::as_u64)
            != Some(profile.maximum_history_tokens() as u64)
        || frame.get("capacity").and_then(Value::as_u64) != Some(MAX_CAPACITY as u64)
        || frame.get("context_rows").and_then(Value::as_u64) != Some(MAX_CONTEXT_ROWS as u64)
        || frame.get("cooperative_input_cancellation") != Some(&Value::Bool(true))
        || frame.get("continuation_limit").and_then(Value::as_u64) != Some(128)
        || frame
            .get("same_session_continuation")
            .and_then(Value::as_bool)
            != Some(true)
        || frame.get("numerical_identity").and_then(Value::as_str)
            != Some("native_full_canonical_prefill_audit_v1")
        || frame
            .get("state_seal_audit_in_wall")
            .and_then(Value::as_bool)
            != Some(profile != Profile::Resident)
        || (profile == Profile::Resident
            && frame.get("state_seal_policy").and_then(Value::as_str)
                != Some("live_owner_no_payload_readback_v1"))
    {
        return Err("producer ready frame violated the profile-bound admission".into());
    }
    if profile.uses_initialized_history()
        && (frame.get("minimum_initial_prompt_tokens") != Some(&json!(1))
            || frame.get("maximum_initial_prompt_tokens") != Some(&json!(profile.maximum_initial_prompt_tokens()))
            || frame.get("growing_initialized_history") != Some(&json!(true))
            || frame.get("uninitialized_tail_hashed") != Some(&json!(false)))
    {
        return Err("producer growing history range/initialization contract".into());
    }
    if frame.get("request_output_budget_supported").unwrap_or(&Value::Bool(false))
        != &Value::Bool(profile == Profile::Resident)
    {
        return Err("producer request output budget capability mismatch".into());
    }
    if profile == Profile::Resident && frame.get("continuation_max_output_limit")
        != Some(&profile.capabilities()["continuation_max_output_limit"])
    {
        return Err("producer extended continuation budget capability mismatch".into());
    }
    for key in strict_false {
        if frame.get(key) != Some(&Value::Bool(false)) {
            return Err(format!("producer ready claim {key} was not false").into());
        }
    }
    if let Some(value) = frame.get("visible_tps") {
        if value != &Value::Bool(false) {
            return Err("producer ready claim visible_tps was not false".into());
        }
    }
    Ok(())
}

fn validate_frame_identity(
    frame: &Value,
    expected_request_id: Option<&str>,
    expected_session_id: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    if let Some(expected) = expected_request_id {
        if let Some(actual) = frame.get("request_id").and_then(Value::as_str) {
            if actual != expected {
                return Err("producer request identity drift".into());
            }
        } else if matches!(
            frame.get("event").and_then(Value::as_str),
            Some(
                "begin"
                    | "text"
                    | "terminal"
                    | "session_closed"
                    | "session_saved"
                    | "session_restored"
            )
        ) {
            return Err("producer frame missing request identity".into());
        }
    }
    if let Some(expected) = expected_session_id {
        let actual = frame.get("session_id").and_then(Value::as_str).or_else(|| {
            frame
                .get("session")
                .and_then(|v| v.get("session_id"))?
                .as_str()
        });
        if let Some(actual) = actual {
            if actual != expected {
                return Err("producer session identity drift".into());
            }
        }
    }
    Ok(())
}

fn validate_checkpoint_ready(ready: &Value, expected: Option<&str>) -> Result<(), Box<dyn Error>> {
    let Some(sha) = expected else {
        if ready.get("checkpoint_service").is_some() {
            return Err("unrequested checkpoint service".into());
        }
        return Ok(());
    };
    let p = &ready["checkpoint_service"];
    if !cfg!(all(windows, feature = "qwen35-session-checkpoint"))
        || !checkpoint_digest(sha)
        || p["schema"] != "qwen35-http-checkpoint-service-v1"
        || p["enabled"] != true
        || p["profile_sha256"] != sha
        || !p["owner_id"].as_str().is_some_and(checkpoint_digest)
        || !p["owner_epoch"]
            .as_u64()
            .is_some_and(|n| n > 0 && n < u64::MAX)
        || p["restore_requires_new_owner"] != true
        || p["client_delivery_acknowledged"] != false
        || p["qualified"] != false
    {
        return Err("startup checkpoint profile/owner binding".into());
    }
    Ok(())
}

fn validate_checkpoint_terminal(
    frame: &Value,
    command: &Value,
    ready: &Value,
) -> Result<(), Box<dyn Error>> {
    let restore = command["op"] == "session_restore";
    let event = if restore {
        "session_restored"
    } else {
        "session_saved"
    };
    let p = &ready["checkpoint_service"];
    if !cfg!(all(windows, feature = "qwen35-session-checkpoint"))
        || (!restore && command["op"] != "session_save")
        || frame["schema"] != "qwen35-http-checkpoint-terminal-v1"
        || frame["event"] != event
        || p["schema"] != "qwen35-http-checkpoint-service-v1"
        || p["enabled"] != true
        || !p["owner_id"].as_str().is_some_and(checkpoint_digest)
        || !p["profile_sha256"].as_str().is_some_and(checkpoint_digest)
        || !p["owner_epoch"]
            .as_u64()
            .is_some_and(|n| n > 0 && n < u64::MAX)
        || frame["owner_id"] != p["owner_id"]
        || frame["owner_epoch"] != p["owner_epoch"]
        || frame["profile_sha256"] != p["profile_sha256"]
        || frame["request_id"] != command["request_id"]
        || frame["session_id"] != command["session_id"]
        || frame["checkpoint_id"] != command["checkpoint_id"]
        || !frame["http_manifest_sha256"]
            .as_str()
            .is_some_and(checkpoint_digest)
        || !frame["native_manifest_sha256"]
            .as_str()
            .is_some_and(checkpoint_digest)
        || frame["session"]["session_id"] != command["session_id"]
        || !frame["revision"].as_u64().is_some_and(|n| n > 0)
        || frame["revision"] != frame["session"]["revision"]
    {
        return Err("checkpoint terminal identity/protocol drift".into());
    }
    for key in [
        "qualified",
        "product_ready",
        "semantic_qualification",
        "visible_tps",
    ] {
        if frame[key] != false {
            return Err("checkpoint terminal claim ceiling".into());
        }
    }
    if restore {
        if frame["http_manifest_sha256"] != command["http_manifest_sha256"]
            || frame["target_seal_verified"] != true
            || frame["prefix_recompute_tokens"] != 0
            || frame["client_delivery_acknowledged"] != false
            || !frame["source_owner_id"]
                .as_str()
                .is_some_and(checkpoint_digest)
            || !frame["source_session_id"].as_str().is_some_and(valid_id)
            || !frame["restored_prefix_tokens"]
                .as_u64()
                .is_some_and(|n| n > 0 && n <= MAX_CAPACITY as u64)
            || frame["source_owner_id"] == frame["owner_id"]
            || frame["source_session_id"] == frame["session_id"]
            || frame["restored_prefix_tokens"] != frame["session"]["consumed_token_count"]
        {
            return Err("checkpoint restore terminal state drift".into());
        }
    } else if frame["revision"] != command["revision"] {
        return Err("checkpoint save revision drift".into());
    }
    Ok(())
}

impl Drop for ResidentOwner {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            self.terminate();
        } else {
            self.frames.join_after_child_exit();
        }
    }
}

fn valid_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
}

fn parse_open(body: &[u8]) -> Result<OpenRequest, Box<dyn Error>> {
    if body.len() > MAX_REQUEST_BYTES {
        return Err("request body exceeds bound".into());
    }
    let request: OpenRequest = serde_json::from_slice(body)?;
    Profile::Resident.output_budget(request.max_new_tokens, true)?;
    if !valid_id(&request.request_id)
        || !valid_id(&request.session_id)
        || request.prompt.is_empty()
        || request.prompt.len() > MAX_PROMPT_BYTES
    {
        return Err("invalid session open".into());
    }
    Ok(request)
}

fn parse_append(body: &[u8], profile: Profile) -> Result<AppendRequest, Box<dyn Error>> {
    if body.len() > MAX_REQUEST_BYTES {
        return Err("request body exceeds bound".into());
    }
    let request: AppendRequest = serde_json::from_slice(body)?;
    profile.output_budget(request.max_new_tokens, false)?;
    if !valid_id(&request.request_id)
        || !valid_id(&request.session_id)
        || request.revision == 0
        || !profile
            .history_bounds()
            .contains(&request.exact_history.len())
        || request.exact_history.iter().any(|&token| token >= 248_320)
        || request.suffix.is_empty()
        || request.suffix.len() > MAX_SUFFIX_BYTES
    {
        return Err("invalid bounded session append".into());
    }
    Ok(request)
}

fn parse_close(body: &[u8]) -> Result<CloseRequest, Box<dyn Error>> {
    if body.len() > MAX_REQUEST_BYTES {
        return Err("request body exceeds bound".into());
    }
    let request: CloseRequest = serde_json::from_slice(body)?;
    if !valid_id(&request.request_id) || !valid_id(&request.session_id) || request.revision == 0 {
        return Err("invalid session close".into());
    }
    Ok(request)
}

fn command_for(
    path: &str,
    body: &[u8],
    profile: Profile,
) -> Result<(Value, bool, u16), Box<dyn Error>> {
    match path {
        "/v1/session/save" => {
            if !cfg!(all(windows, feature = "qwen35-session-checkpoint")) || body.len() > 65536 {
                return Err("checkpoint endpoint disabled/oversized".into());
            }
            let r: SaveRequest = serde_json::from_slice(body)?;
            if !valid_id(&r.request_id)
                || !valid_id(&r.session_id)
                || r.revision == 0
                || !checkpoint_digest(&r.checkpoint_id)
            {
                return Err("invalid checkpoint save request".into());
            }
            Ok((
                json!({"op":"session_save","request_id":r.request_id,"session_id":r.session_id,
                "revision":r.revision,"checkpoint_id":r.checkpoint_id}),
                false,
                200,
            ))
        }
        "/v1/session/restore" => {
            if !cfg!(all(windows, feature = "qwen35-session-checkpoint")) || body.len() > 65536 {
                return Err("checkpoint endpoint disabled/oversized".into());
            }
            let r: RestoreRequest = serde_json::from_slice(body)?;
            if !valid_id(&r.request_id)
                || !valid_id(&r.session_id)
                || !checkpoint_digest(&r.checkpoint_id)
                || !checkpoint_digest(&r.http_manifest_sha256)
            {
                return Err("invalid checkpoint restore request".into());
            }
            Ok((
                json!({"op":"session_restore","request_id":r.request_id,"session_id":r.session_id,
                "checkpoint_id":r.checkpoint_id,"http_manifest_sha256":r.http_manifest_sha256}),
                false,
                200,
            ))
        }
        "/v1/session/open" => {
            let request = parse_open(body)?;
            profile.output_budget(request.max_new_tokens, true)?;
            let mut command = json!({"op":"session_open","request_id":request.request_id,"session_id":request.session_id,"prompt":request.prompt});
            if let Some(budget) = request.max_new_tokens {
                command["max_new_tokens"] = json!(budget);
            }
            Ok((
                command,
                false,
                200,
            ))
        }
        "/v1/session/append" => {
            let request = parse_append(body, profile)?;
            let mut command = json!({"op":"session_append","request_id":request.request_id,"session_id":request.session_id,"revision":request.revision,"exact_history":request.exact_history,"suffix":request.suffix});
            if let Some(budget) = request.max_new_tokens {
                command["max_new_tokens"] = json!(budget);
            }
            Ok((
                command,
                false,
                200,
            ))
        }
        "/v1/session/close" => {
            let request = parse_close(body)?;
            Ok((
                json!({"op":"session_close","request_id":request.request_id,"session_id":request.session_id,"revision":request.revision}),
                true,
                200,
            ))
        }
        "/v1/shutdown" => {
            if !body.is_empty() && body != b"{}" {
                return Err("shutdown body must be empty or {}".into());
            }
            Ok((json!({"op":"shutdown"}), true, 200))
        }
        _ => Err("unknown session path".into()),
    }
}

fn read_request(stream: &mut TcpStream) -> Result<(String, String, Vec<u8>), Box<dyn Error>> {
    let mut bytes = Vec::with_capacity(8_192);
    let mut chunk = [0_u8; 8_192];
    let header_end;
    loop {
        let count = stream.read(&mut chunk)?;
        if count == 0 {
            return Err("connection closed before HTTP headers".into());
        }
        bytes.extend_from_slice(&chunk[..count]);
        if bytes.len() > MAX_REQUEST_BYTES {
            return Err("HTTP request exceeds bound".into());
        }
        if let Some(index) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
            header_end = index + 4;
            break;
        }
    }
    let header = std::str::from_utf8(&bytes[..header_end])?;
    let mut fields = header
        .lines()
        .next()
        .ok_or("missing request line")?
        .split_whitespace();
    let method = fields.next().ok_or("missing method")?.to_owned();
    let path = fields.next().ok_or("missing path")?.to_owned();
    let version = fields.next().ok_or("missing HTTP version")?;
    if version != "HTTP/1.1" || fields.next().is_some() {
        return Err("invalid request line".into());
    }
    let mut content_length = None;
    let mut transfer_encoding = false;
    for line in header.lines().skip(1) {
        if let Some(value) = line
            .strip_prefix("Content-Length:")
            .or_else(|| line.strip_prefix("content-length:"))
        {
            content_length = Some(value.trim().parse::<usize>()?);
        }
        if line.to_ascii_lowercase().starts_with("transfer-encoding:") {
            transfer_encoding = true;
        }
    }
    if transfer_encoding {
        return Err("chunked request bodies are not accepted".into());
    }
    let content_length = content_length.unwrap_or(0);
    if header_end
        .checked_add(content_length)
        .ok_or("request extent overflow")?
        > MAX_REQUEST_BYTES
    {
        return Err("HTTP body exceeds bound".into());
    }
    while bytes.len() < header_end + content_length {
        let count = stream.read(&mut chunk)?;
        if count == 0 {
            return Err("connection closed in HTTP body".into());
        }
        bytes.extend_from_slice(&chunk[..count]);
    }
    if bytes.len() != header_end + content_length {
        return Err("one complete HTTP request per connection; pipelining is unsupported".into());
    }
    Ok((
        method,
        path,
        bytes[header_end..header_end + content_length].to_vec(),
    ))
}

fn write_json(stream: &mut TcpStream, status: u16, value: &Value) -> Result<(), Box<dyn Error>> {
    let body = serde_json::to_vec(value)?;
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        409 => "Conflict",
        422 => "Unprocessable Entity",
        503 => "Service Unavailable",
        _ => "Internal Server Error",
    };
    write!(stream, "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nX-Qwen35-Qualified: {QUALIFIED_HEADER}\r\nX-Qwen35-Product-Ready: false\r\nConnection: close\r\n\r\n", body.len())?;
    stream.write_all(&body)?;
    stream.flush()?;
    Ok(())
}

fn start_chunked(stream: &mut TcpStream) -> Result<(), Box<dyn Error>> {
    stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nTransfer-Encoding: chunked\r\nX-Qwen35-Qualified: false\r\nX-Qwen35-Product-Ready: false\r\nConnection: close\r\n\r\n")?;
    stream.flush()?;
    Ok(())
}

fn write_chunk(stream: &mut TcpStream, body: &[u8]) -> Result<(), Box<dyn Error>> {
    // The response advertises NDJSON. Keep exactly one JSON object per
    // newline even if an intermediary coalesces HTTP chunks.
    write!(stream, "{:X}\r\n", body.len() + 1)?;
    stream.write_all(body)?;
    stream.write_all(b"\n\r\n")?;
    Ok(())
}

fn end_chunked(stream: &mut TcpStream) -> Result<(), Box<dyn Error>> {
    stream.write_all(b"0\r\n\r\n")?;
    stream.flush()?;
    Ok(())
}

fn handle_connection(
    mut stream: TcpStream,
    options: &Options,
    owner: &mut Option<ResidentOwner>,
) -> Result<bool, Box<dyn Error>> {
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    stream.set_write_timeout(Some(Duration::from_secs(30)))?;
    let (method, path, body) = read_request(&mut stream)?;
    if method == "GET" && path == "/health" {
        write_json(
            &mut stream,
            200,
            &json!({"status": if owner.is_some() {"ready"} else {"stopped"}, "profile":options.profile.name(), "session_limits":options.profile.capabilities(), "owner_resident":owner.is_some(), "qualified":false, "product_ready":false}),
        )?;
        return Ok(false);
    }
    if method != "POST" {
        write_json(&mut stream, 404, &json!({"status":"not_found"}))?;
        return Ok(false);
    }
    if !matches!(
        path.as_str(),
        "/v1/session/open"
            | "/v1/session/append"
            | "/v1/session/close"
            | "/v1/shutdown"
            | "/v1/session/save"
            | "/v1/session/restore"
    ) {
        write_json(&mut stream, 404, &json!({"status":"not_found"}))?;
        return Ok(false);
    }
    if matches!(path.as_str(), "/v1/session/save" | "/v1/session/restore")
        && options.checkpoint_profile.is_none()
    {
        write_json(
            &mut stream,
            404,
            &json!({"status":"checkpoint_service_disabled"}),
        )?;
        return Ok(false);
    }
    let (command, close_boundary, _) = match command_for(&path, &body, options.profile) {
        Ok(value) => value,
        Err(error) => {
            write_json(
                &mut stream,
                400,
                &json!({"status":"rejected","detail":error.to_string(),"qualified":false,"product_ready":false}),
            )?;
            return Ok(false);
        }
    };
    if owner.is_none() {
        if path == "/v1/shutdown" {
            write_json(
                &mut stream,
                200,
                &json!({"status":"stopped","owner_resident":false,"qualified":false,"product_ready":false}),
            )?;
            return Ok(true);
        }
        *owner = Some(ResidentOwner::spawn(options, &stream)?);
    }
    let expected_request_id = command.get("request_id").and_then(Value::as_str);
    let expected_session_id = command.get("session_id").and_then(Value::as_str);
    let result = (|| -> Result<bool, Box<dyn Error>> {
        let active = owner
            .as_mut()
            .ok_or_else(|| "resident owner missing after spawn".to_owned())?;
        // Header/client failures must poison an existing owner too, rather than
        // escape before the scoped cleanup below and leave stale state reusable.
        http_transport::client_connected(&stream)?;
        start_chunked(&mut stream)?;
        active.send_command(&command)?;
        let complete = active.stream_until_boundary(
            &mut stream,
            close_boundary,
            expected_request_id,
            expected_session_id,
            Some(&command),
        )?;
        end_chunked(&mut stream)?;
        Ok(complete)
    })();
    match result {
        Ok(true) => {
            if path == "/v1/shutdown" {
                owner.take();
                return Ok(true);
            }
            Ok(false)
        }
        Ok(false) => Ok(false),
        Err(error) => {
            if let Some(mut active) = owner.take() {
                active.retire_disconnected();
            }
            Err(error)
        }
    }
}

fn parse_options() -> Result<Options, Box<dyn Error>> {
    let mut bind = "127.0.0.1:8193".to_owned();
    let mut producer = None;
    let mut gguf = None;
    let mut config = None;
    let mut tokenizer = None;
    let mut q8_gguf = None;
    let mut dflash_config = None;
    let mut generation_config = None;
    let mut vision_safetensors = None;
    let mut evidence_root = PathBuf::from("qwen35-http-evidence");
    let mut device = 0;
    let mut profile = Profile::Resident;
    let mut checkpoint_profile = None::<PathBuf>;
    let mut checkpoint_profile_sha = None::<String>;
    let mut input_prefix_cache_tokens = None::<usize>;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = |args: &mut std::iter::Skip<std::env::Args>| {
            args.next()
                .ok_or_else(|| format!("missing value for {flag}"))
        };
        match flag.as_str() {
            "--input-prefix-cache-tokens" => {
                if input_prefix_cache_tokens.is_some() { return Err("duplicate input-prefix cache size".into()); }
                input_prefix_cache_tokens = Some(value(&mut args)?.parse()?);
            }
            "--checkpoint-profile" => {
                if checkpoint_profile.is_some() {
                    return Err("duplicate checkpoint profile".into());
                }
                checkpoint_profile = Some(value(&mut args)?.into());
            }
            "--checkpoint-profile-sha256" => {
                if checkpoint_profile_sha.is_some() {
                    return Err("duplicate checkpoint profile hash".into());
                }
                checkpoint_profile_sha = Some(value(&mut args)?);
            }
            "--bind" => bind = value(&mut args)?,
            "--producer" => producer = Some(PathBuf::from(value(&mut args)?)),
            "--gguf" => gguf = Some(PathBuf::from(value(&mut args)?)),
            "--config" => config = Some(PathBuf::from(value(&mut args)?)),
            "--tokenizer" => tokenizer = Some(PathBuf::from(value(&mut args)?)),
            "--q8-gguf" => q8_gguf = Some(PathBuf::from(value(&mut args)?)),
            "--dflash-config" => dflash_config = Some(PathBuf::from(value(&mut args)?)),
            "--request-generation-config" => {
                generation_config = Some(PathBuf::from(value(&mut args)?))
            }
            "--vision-safetensors" => vision_safetensors = Some(PathBuf::from(value(&mut args)?)),
            "--evidence-root" => evidence_root = PathBuf::from(value(&mut args)?),
            "--device" => device = value(&mut args)?.parse()?,
            "--profile" => profile = Profile::parse(&value(&mut args)?)?,
            "--help" | "-h" => {
                println!("opt-in exact prefix slot: --input-prefix-cache-tokens N (resident profile plus pinned --checkpoint-profile required)");
                println!("usage: qwen35_resident_session_http --producer PATH --gguf PATH --config PATH --tokenizer PATH --q8-gguf PATH --dflash-config PATH --request-generation-config PATH --vision-safetensors PATH [--profile resident|growing|8k|32k|64k|near192k] [--bind HOST:PORT] [--evidence-root DIR] [--device N]");
                return Err("help requested".into());
            }
            _ => return Err(format!("unknown argument {flag}").into()),
        }
    }
    if checkpoint_profile.is_some() != checkpoint_profile_sha.is_some()
        || (checkpoint_profile.is_some()
            && (!cfg!(all(windows, feature = "qwen35-session-checkpoint"))
                || !checkpoint_profile_sha
                    .as_deref()
                    .is_some_and(checkpoint_digest)))
    {
        return Err("checkpoint profile requires compiled feature and exact hash".into());
    }
    if input_prefix_cache_tokens.is_some_and(|n| !(2048..=16384).contains(&n) || n % 8 != 0)
        || (input_prefix_cache_tokens.is_some() && (profile != Profile::Resident || checkpoint_profile.is_none()))
    { return Err("input-prefix cache requires resident profile, checkpoint profile and aligned bounded size".into()); }
    fs::create_dir_all(&evidence_root)?;
    Ok(Options {
        bind,
        producer: producer.ok_or("--producer is required")?,
        gguf: gguf.ok_or("--gguf is required")?,
        config: config.ok_or("--config is required")?,
        tokenizer: tokenizer.ok_or("--tokenizer is required")?,
        q8_gguf: q8_gguf.ok_or("--q8-gguf is required")?,
        dflash_config: dflash_config.ok_or("--dflash-config is required")?,
        generation_config: generation_config.ok_or("--request-generation-config is required")?,
        vision_safetensors: vision_safetensors.ok_or("--vision-safetensors is required")?,
        evidence_root,
        device,
        profile,
        checkpoint_profile: checkpoint_profile.zip(checkpoint_profile_sha),
        input_prefix_cache_tokens,
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    if augur_model_qwen35_runtime::build_identity::print_cli_identity("adapter")? {
        return Ok(());
    }
    let options = parse_options()?;
    let listener = TcpListener::bind(&options.bind)?;
    eprintln!("QWEN35_RESIDENT_SESSION_HTTP_LISTEN={}", options.bind);
    let mut owner = None;
    for connection in listener.incoming() {
        match connection {
            Ok(stream) => match handle_connection(stream, &options, &mut owner) {
                Ok(true) => break,
                Ok(false) => {}
                Err(error) => eprintln!("QWEN35_RESIDENT_SESSION_HTTP_REQUEST_ERROR={error}"),
            },
            Err(error) => eprintln!("QWEN35_RESIDENT_SESSION_HTTP_ACCEPT_ERROR={error}"),
        }
    }
    if let Some(mut active) = owner {
        active.terminate();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn explicit_prefix_cache_ready_cannot_drift_or_enable_silently() {
        let value = serde_json::json!({"input_prefix_cache":{"enabled":true,"prefix_tokens":8192,
            "slots":1,"scope":"single_owner_exact_input_prefix","capture_on_first_eligible_open":true,
            "equal_prompt_is_miss":true,"cross_owner_reuse":false,"qualified":false}});
        assert!(super::validate_input_prefix_ready(&value, Some(8192)).is_ok());
        assert!(super::validate_input_prefix_ready(&value, None).is_err());
        assert!(super::validate_input_prefix_ready(&value, Some(4096)).is_err());
        assert!(super::validate_input_prefix_ready(&serde_json::json!({}), Some(8192)).is_err());
        assert!(super::validate_input_prefix_ready(&serde_json::json!({}), None).is_ok());
        for key in ["enabled","slots","scope","capture_on_first_eligible_open","equal_prompt_is_miss","cross_owner_reuse","qualified"] {
            let mut changed = value.clone();
            changed["input_prefix_cache"][key] = serde_json::Value::Null;
            assert!(super::validate_input_prefix_ready(&changed, Some(8192)).is_err(), "{key}");
        }
    }
    use super::*;

    #[test]
    fn resident_budget_preserved_and_rejected_before_submit_when_invalid() {
        for (path, mut body, maximum) in [
            ("/v1/session/open", json!({"request_id":"r","session_id":"s","prompt":"hi"}), 1024),
            ("/v1/session/append", json!({"request_id":"r","session_id":"s","revision":1,"exact_history":[1,2],"suffix":"x"}), 1024),
        ] {
            assert!(command_for(path, &serde_json::to_vec(&body).unwrap(), Profile::Resident)
                .unwrap().0.get("max_new_tokens").is_none());
            for n in [1, 2, 7, 8, 9, maximum] {
                body["max_new_tokens"] = json!(n);
                let raw = serde_json::to_vec(&body).unwrap();
                assert_eq!(command_for(path, &raw, Profile::Resident).unwrap().0["max_new_tokens"], n);
                assert!(command_for(path, &raw, Profile::Growing).is_err());
            }
            for invalid in [json!(0), json!(-1), json!(true), json!(1.5), json!(maximum + 1), json!("1")] {
                body["max_new_tokens"] = invalid;
                assert!(command_for(path, &serde_json::to_vec(&body).unwrap(), Profile::Resident).is_err());
            }
        }
    }

    #[test]
    fn checkpoint_requests_are_strict_and_default_off() {
        let id = "a".repeat(64);
        let save = json!({"request_id":"save","session_id":"s","revision":1,"checkpoint_id":id});
        let restore = json!({"request_id":"restore","session_id":"new","checkpoint_id":id,"http_manifest_sha256":"b".repeat(64)});
        let enabled = cfg!(all(windows, feature = "qwen35-session-checkpoint"));
        for (path, body) in [("/v1/session/save", save), ("/v1/session/restore", restore)] {
            assert_eq!(
                command_for(
                    path,
                    &serde_json::to_vec(&body).unwrap(),
                    Profile::Canonical8k
                )
                .is_ok(),
                enabled
            );
            for key in ["path", "exact_history", "product_ready", "owner_override"] {
                let mut bad = body.clone();
                bad[key] = json!(true);
                assert!(command_for(
                    path,
                    &serde_json::to_vec(&bad).unwrap(),
                    Profile::Canonical8k
                )
                .is_err());
            }
            let mut bad = body;
            bad["checkpoint_id"] = json!("../other");
            assert!(command_for(
                path,
                &serde_json::to_vec(&bad).unwrap(),
                Profile::Canonical8k
            )
            .is_err());
        }
    }

    #[cfg(all(windows, feature = "qwen35-session-checkpoint"))]
    #[test]
    fn checkpoint_ready_and_terminals_bind_exact_owner_request_operation_and_hash() {
        let (owner, profile, id, hash) = (
            "a".repeat(64),
            "b".repeat(64),
            "c".repeat(64),
            "d".repeat(64),
        );
        let ready = json!({"checkpoint_service":{"schema":"qwen35-http-checkpoint-service-v1",
            "enabled":true,"profile_sha256":profile,"owner_id":owner,"owner_epoch":1,
            "restore_requires_new_owner":true,"client_delivery_acknowledged":false,"qualified":false}});
        validate_checkpoint_ready(&ready, Some(&profile)).unwrap();
        assert!(validate_checkpoint_ready(&ready, None).is_err());
        assert!(validate_checkpoint_ready(&ready, Some(&hash)).is_err());
        let mut command = json!({"op":"session_save","request_id":"r","session_id":"s","revision":3,"checkpoint_id":id});
        let mut frame = json!({"schema":"qwen35-http-checkpoint-terminal-v1","event":"session_saved",
            "request_id":"r","session_id":"s","revision":3,"owner_id":owner,"owner_epoch":1,
            "profile_sha256":profile,"checkpoint_id":id,"http_manifest_sha256":hash,"native_manifest_sha256":hash,
            "session":{"session_id":"s","revision":3,"consumed_token_count":8804},
            "qualified":false,"product_ready":false,"semantic_qualification":false,"visible_tps":false});
        validate_checkpoint_terminal(&frame, &command, &ready).unwrap();
        for key in [
            "schema",
            "event",
            "owner_id",
            "owner_epoch",
            "profile_sha256",
            "request_id",
            "session_id",
            "checkpoint_id",
            "http_manifest_sha256",
            "native_manifest_sha256",
            "revision",
            "session",
        ] {
            let mut bad = frame.clone();
            bad.as_object_mut().unwrap().remove(key);
            assert!(
                validate_checkpoint_terminal(&bad, &command, &ready).is_err(),
                "{key}"
            );
        }
        for key in [
            "qualified",
            "product_ready",
            "semantic_qualification",
            "visible_tps",
        ] {
            let mut bad = frame.clone();
            bad[key] = json!(true);
            assert!(validate_checkpoint_terminal(&bad, &command, &ready).is_err());
        }
        command["op"] = json!("session_restore");
        command["http_manifest_sha256"] = json!(hash);
        frame["event"] = json!("session_restored");
        frame["target_seal_verified"] = json!(true);
        frame["prefix_recompute_tokens"] = json!(0);
        frame["client_delivery_acknowledged"] = json!(false);
        frame["source_owner_id"] = json!("e".repeat(64));
        frame["source_session_id"] = json!("prior");
        frame["restored_prefix_tokens"] = json!(8804);
        validate_checkpoint_terminal(&frame, &command, &ready).unwrap();
        for (key, value) in [
            ("http_manifest_sha256", json!(id)),
            ("target_seal_verified", json!(false)),
            ("prefix_recompute_tokens", json!(1)),
            ("source_owner_id", json!(owner)),
            ("source_session_id", json!("s")),
            ("restored_prefix_tokens", json!(1)),
            ("client_delivery_acknowledged", json!(true)),
        ] {
            let mut bad = frame.clone();
            bad[key] = value;
            assert!(
                validate_checkpoint_terminal(&bad, &command, &ready).is_err(),
                "{key}"
            );
        }
        command["op"] = json!("session_open");
        assert!(validate_checkpoint_terminal(&frame, &command, &ready).is_err());
    }

    #[cfg(windows)]
    #[test]
    fn silent_child_cancel_reaps_owned_process_and_reader() {
        use std::{net::Shutdown, os::windows::process::CommandExt, thread, time::Instant};
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let mut server = listener.accept().unwrap().0;
        // A real, CPU-only child with a silent stdout. No child-of-child shell
        // command and no CUDA/model artifacts. The finite sleep bounds regression.
        let mut child = Command::new("powershell.exe")
            .args([
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Start-Sleep -Seconds 5",
            ])
            .creation_flags(0x08000000)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let stdin = child.stdin.take().unwrap();
        let frames = ProducerFrames::spawn(child.stdout.take().unwrap(), MAX_FRAME_BYTES).unwrap();
        let mut owner = ResidentOwner {
            child,
            stdin: Some(stdin),
            frames,
            ready: Value::Null,
        };
        let closer = thread::spawn(move || {
            thread::sleep(Duration::from_millis(60));
            client.shutdown(Shutdown::Both).unwrap();
        });
        let start = Instant::now();
        let error = owner
            .stream_until_boundary(&mut server, false, Some("r"), Some("s"), None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("client disconnected"), "{error}");
        owner.terminate();
        assert!(owner.child.try_wait().unwrap().is_some());
        assert!(start.elapsed() < Duration::from_secs(2));
        closer.join().unwrap();
        // Drop is idempotent even after explicit termination joined the worker.
        drop(owner);
    }

    #[cfg(windows)]
    fn cpu_only_owner(script: &str) -> ResidentOwner {
        use std::os::windows::process::CommandExt;
        let mut child = Command::new("powershell.exe")
            .args(["-NoProfile", "-NonInteractive", "-Command", script])
            .creation_flags(0x08000000)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let stdin = child.stdin.take();
        let frames = ProducerFrames::spawn(child.stdout.take().unwrap(), MAX_FRAME_BYTES).unwrap();
        ResidentOwner {
            child,
            stdin,
            frames,
            ready: Value::Null,
        }
    }

    #[cfg(windows)]
    #[test]
    fn cooperative_retirement_closes_idle_input_without_kill() {
        let mut owner = cpu_only_owner("$null = [Console]::In.ReadToEnd(); exit 0");
        let result = owner.retire_disconnected();
        assert!(!result.force_attempted);
        assert!(result.exit.success());
        assert!(owner.stdin.is_none());
        assert!(owner.send_command(&json!({"op":"shutdown"})).is_err());
        assert!(owner.child.try_wait().unwrap().is_some());
        drop(owner);
    }

    #[cfg(windows)]
    #[test]
    fn cooperative_retirement_breaks_active_output_without_kill() {
        use std::{os::windows::process::CommandExt, time::SystemTime};
        // Exercise Rust Write/flush, as used by the real producer. Console.Out
        // in a PowerShell mock need not propagate a broken-pipe write as error.
        let directory = env::temp_dir().join(format!(
            "qwen35-native-pipe-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&directory).unwrap();
        let binary = directory.join("writer.exe");
        let source =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/http_pipe_writer.rs");
        let compiled = Command::new(env::var_os("RUSTC").unwrap_or_else(|| "rustc".into()))
            .arg("--edition=2021")
            .arg(source)
            .arg("-o")
            .arg(&binary)
            .creation_flags(0x08000000)
            .output()
            .unwrap();
        assert!(
            compiled.status.success(),
            "{}",
            String::from_utf8_lossy(&compiled.stderr)
        );
        let mut child = Command::new(&binary)
            .creation_flags(0x08000000)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let stdin = child.stdin.take();
        let frames = ProducerFrames::spawn(child.stdout.take().unwrap(), MAX_FRAME_BYTES).unwrap();
        let mut owner = ResidentOwner {
            child,
            stdin,
            frames,
            ready: Value::Null,
        };
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let server = listener.accept().unwrap().0;
        assert_eq!(
            owner.read_frame_line(&server).unwrap()["event"],
            "cpu_only_commit"
        );
        let result = owner.retire_disconnected();
        drop(client);
        fs::remove_file(&binary).unwrap();
        let symbols = binary.with_extension("pdb");
        if symbols.exists() {
            fs::remove_file(symbols).unwrap();
        }
        fs::remove_dir(&directory).unwrap();
        assert!(!result.force_attempted);
        assert_eq!(result.exit.code(), Some(7));
        assert!(owner.child.try_wait().unwrap().is_some());
        drop(owner);
    }

    #[test]
    fn strict_http_requests_reject_unknown_fields_and_bad_ids() {
        assert!(parse_open(br#"{"request_id":"r","session_id":"s","prompt":"x"}"#).is_ok());
        assert!(
            parse_open(br#"{"request_id":"r","session_id":"s","prompt":"x","capacity":1}"#)
                .is_err()
        );
        assert!(parse_open(br#"{"request_id":"r/evil","session_id":"s","prompt":"x"}"#).is_err());
    }

    #[test]
    fn profile_bounds_match_native_session_contract() {
        let body = serde_json::json!({"request_id":"r","session_id":"s","revision":1,"exact_history":vec![1_u32; 194560],"suffix":"x"});
        assert!(parse_append(&serde_json::to_vec(&body).unwrap(), Profile::Near192k).is_ok());
        assert!(parse_append(&serde_json::to_vec(&body).unwrap(), Profile::Context64k).is_err());
    }

    #[test]
    fn shutdown_is_strictly_empty() {
        assert!(command_for("/v1/shutdown", b"{}", Profile::Canonical8k).is_ok());
        assert!(command_for("/v1/shutdown", br#"{"force":true}"#, Profile::Canonical8k).is_err());
    }

    #[test]
    fn claims_are_closed_on_forwarded_frames() {
        for value in [false, true] {
            let frame = json!({"event":"text","product_ready":value});
            assert_eq!(frame["product_ready"].as_bool(), Some(value));
            if value {
                assert_ne!(frame["product_ready"], Value::Bool(false));
            }
        }
    }

    #[test]
    fn ready_frame_is_bound_to_each_profile_contract() {
        for profile in [
            Profile::Canonical8k,
            Profile::Context32k,
            Profile::Context64k,
            Profile::Near192k,
        ] {
            let frame = json!({
                "schema": profile.schema(),
                "event": "ready",
                "owner_open_count": 1,
                "initial_prompt_tokens": profile.initial_prompt_tokens(),
                "maximum_history_tokens": profile.maximum_history_tokens(),
                "capacity": MAX_CAPACITY,
                "context_rows": MAX_CONTEXT_ROWS,
                "continuation_limit": 128,
                "same_session_continuation": true,
                "product_ready": false,
                "session_qualified": false,
                "vision_forward_executed": false,
                "numerical_identity": "native_full_canonical_prefill_audit_v1",
                "state_seal_audit_in_wall": true,
                "cooperative_input_cancellation": true,
            });
            assert!(validate_ready(&frame, profile).is_ok());

            let mut wrong_schema = frame.clone();
            wrong_schema["schema"] = json!(Profile::Canonical8k.schema());
            assert_eq!(
                validate_ready(&wrong_schema, profile).is_ok(),
                profile == Profile::Canonical8k
            );

            let mut wrong_claim = frame.clone();
            wrong_claim["product_ready"] = json!(true);
            assert!(validate_ready(&wrong_claim, profile).is_err());
            for value in [Value::Null, Value::Bool(false), json!(1)] {
                let mut legacy = frame.clone();
                legacy["cooperative_input_cancellation"] = value;
                assert!(validate_ready(&legacy, profile).is_err());
            }
            let mut missing = frame.clone();
            missing.as_object_mut().unwrap().remove("cooperative_input_cancellation");
            assert!(validate_ready(&missing, profile).is_err());
        }
    }

    #[test]
    fn forwarded_frames_keep_request_and_session_identity() {
        let frame = json!({
            "event": "terminal",
            "request_id": "req-1",
            "session": {"session_id": "sess-1"}
        });
        assert!(validate_frame_identity(&frame, Some("req-1"), Some("sess-1")).is_ok());
        assert!(validate_frame_identity(&frame, Some("req-2"), Some("sess-1")).is_err());
        assert!(validate_frame_identity(&frame, Some("req-1"), Some("sess-2")).is_err());
    }

    #[test]
    fn growing_ready_and_history_require_explicit_nonfixed_initialization() {
        let mut frame = json!({"schema":Profile::Growing.schema(),"event":"ready",
            "owner_open_count":1,"initial_prompt_tokens":null,
            "minimum_initial_prompt_tokens":1,"maximum_initial_prompt_tokens":8192,
            "maximum_history_tokens":16384,"capacity":MAX_CAPACITY,"context_rows":MAX_CONTEXT_ROWS,
            "continuation_limit":128,"same_session_continuation":true,
            "growing_initialized_history":true,"uninitialized_tail_hashed":false,
            "product_ready":false,"session_qualified":false,"vision_forward_executed":false,
            "numerical_identity":"native_full_canonical_prefill_audit_v1","state_seal_audit_in_wall":true,
            "cooperative_input_cancellation":true});
        assert!(validate_ready(&frame, Profile::Growing).is_ok());
        let mut resident = frame.clone();
        resident["schema"] = json!(Profile::Resident.schema());
        resident["maximum_initial_prompt_tokens"] = json!(Profile::Resident.maximum_initial_prompt_tokens());
        resident["maximum_history_tokens"] = json!(MAX_CAPACITY);
        resident["request_output_budget_supported"] = json!(true);
        resident["continuation_max_output_limit"] = json!(1024);
        resident["state_seal_audit_in_wall"] = json!(false);
        resident["state_seal_policy"] = json!("live_owner_no_payload_readback_v1");
        assert!(validate_ready(&resident, Profile::Resident).is_ok());
        assert!(validate_ready(&resident, Profile::Growing).is_err());
        for key in ["growing_initialized_history", "uninitialized_tail_hashed", "maximum_initial_prompt_tokens", "request_output_budget_supported", "continuation_max_output_limit", "state_seal_policy", "state_seal_audit_in_wall"] {
            let mut invalid = resident.clone();
            invalid.as_object_mut().unwrap().remove(key);
            assert!(validate_ready(&invalid, Profile::Resident).is_err());
        }
        for value in [json!(false), Value::Null, json!(1), json!("true")] {
            let mut invalid = resident.clone();
            invalid["request_output_budget_supported"] = value;
            assert!(validate_ready(&invalid, Profile::Resident).is_err());
        }
        for (key,value) in [
            ("state_seal_policy",json!("full_bytes_pipelined_dual_sha256_v1")),
            ("state_seal_policy",json!(true)),
            ("state_seal_audit_in_wall",json!(true)),
            ("state_seal_audit_in_wall",json!(0)),
        ] {
            let mut invalid = resident.clone();
            invalid[key] = value;
            assert!(validate_ready(&invalid, Profile::Resident).is_err());
        }
        assert!(validate_ready(&frame, Profile::Canonical8k).is_err());
        for (key, value) in [
            ("initial_prompt_tokens", json!(8192)),
            ("minimum_initial_prompt_tokens", json!(0)),
            ("maximum_initial_prompt_tokens", json!(16384)),
            ("uninitialized_tail_hashed", json!(true)),
            ("growing_initialized_history", json!(false)),
        ] {
            let mut bad = frame.clone();
            bad[key] = value;
            assert!(validate_ready(&bad, Profile::Growing).is_err());
            bad.as_object_mut().unwrap().remove(key);
            assert!(validate_ready(&bad, Profile::Growing).is_err());
        }
        frame
            .as_object_mut()
            .unwrap()
            .remove("initial_prompt_tokens");
        assert!(validate_ready(&frame, Profile::Growing).is_err());
        for n in [0, 1, 7, 31, 2048, 8192, 16384, 16385] {
            let body = json!({"request_id":"r","session_id":"s","revision":1,
                "exact_history":vec![1;n],"suffix":"x"});
            assert_eq!(
                parse_append(&serde_json::to_vec(&body).unwrap(), Profile::Growing).is_ok(),
                (1..=16384).contains(&n)
            );
            assert_eq!(parse_append(&serde_json::to_vec(&body).unwrap(), Profile::Resident).is_ok(), n > 0);
        }
    }
}
