#![cfg(feature = "qwen35-draft-target-verify")]

//! Native committed-token JSONL producer for the opt-in Qwen35 rolling path.
//!
//! This example is deliberately separate from the legacy whole-response HTTP
//! adapter. It publishes only tokens received after target verify/rollback/
//! commit, and it refuses to emit a terminal frame when the bounded rolling
//! budget ended before the request contract reached EOS or max_new_tokens.
//! The stream is diagnostic and unqualified: it does not publish product/TPS
//! or semantic qualification claims.

use augur_model_qwen35_runtime::{
    inspect_qwen35_vision_payload, run_qwen35_q8_draft_target_rolling_cuda_with_options_and_sink,
    Qwen35AutoregressiveCudaConfig, Qwen35DraftTargetRequestOptions,
    Qwen35DraftTargetRollingOptions, Qwen35MixedPrefixCudaError, Qwen35ResidentDraftCuda,
    Qwen35VerifiedCommit, Qwen35VerifiedCommitSink,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    cell::{Cell, RefCell},
    env,
    error::Error,
    fs::{self, OpenOptions},
    io::{self, BufWriter, Write},
    path::PathBuf,
};
use tokenizers::Tokenizer;

// Native experiment admission only; a diagnostic flag must never fall through
// into ordinary CLI/HTTP operation or silently select a different experiment.
fn validate_native_prefill_audit_selection(
    session: bool,
    transition: bool,
    scalar: bool,
    full: bool,
    suffix: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    if (scalar || full) && (!session || transition || (scalar && full)) {
        return Err("prefill audit requires exclusive native-session smoke".into());
    }
    if suffix.is_some_and(|s| {
        !full
            || !matches!(
                s,
                "add-aligned"
                    | "subtract-nonaligned"
                    | "code-output"
                    | "code-bitmix"
                    | "ledger-recall"
                    | "literal-copy"
                    | "ledger-counterfactual-long"
                    | "ledger-counterfactual-canonical-long"
            )
    }) {
        return Err("suffix case requires full canonical audit and a known fixture".into());
    }
    Ok(())
}

fn validate_native_long_audit(
    long: bool,
    session: bool,
    full: bool,
    transition: bool,
    suffix: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    if long && (!session || !full || transition || !cfg!(feature = "qwen35-draft-private-state")) {
        return Err("long128 audit requires explicit full-canonical private native session".into());
    }
    if matches!(
        suffix,
        Some("ledger-counterfactual-long" | "ledger-counterfactual-canonical-long")
    ) && !long
    {
        return Err("long fixture requires separate long128 identity".into());
    }
    Ok(())
}

fn validate_native_exact_reference(
    input: Option<&std::path::Path>,
    expected_sha256: Option<&str>,
    eligible: bool,
) -> Result<(), Box<dyn Error>> {
    match (input, expected_sha256) {
        (None, None) => Ok(()),
        (Some(path), Some(sha))
            if eligible
                && path.is_absolute()
                && sha.len() == 64
                && sha
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) =>
        {
            Ok(())
        }
        _ => Err(
            "exact reference requires a pinned absolute input and exclusive native long128 audit"
                .into(),
        ),
    }
}

// Explicit external-token reference profiles. They do not widen HTTP admission
// or infer an execution profile from an untrusted token-vector length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NativeLongReferenceProfile {
    Context32,
    Context64,
    Near192,
}

impl NativeLongReferenceProfile {
    fn parse(name: &str) -> Result<Self, Box<dyn Error>> {
        match name {
            "context32" => Ok(Self::Context32),
            "context64" => Ok(Self::Context64),
            "near192" => Ok(Self::Near192),
            _ => Err("unknown native exact-reference long profile".into()),
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::Context32 => "context32",
            Self::Context64 => "context64",
            Self::Near192 => "near192",
        }
    }
    fn schema(self) -> &'static str {
        match self {
            Self::Context32 => "qwen35-context32-exact-history-reference-input-v1",
            Self::Context64 => "qwen35-context64-exact-history-reference-input-v1",
            Self::Near192 => "qwen35-near192-exact-history-reference-input-v1",
        }
    }
    fn bounds(self) -> (usize, usize) {
        match self {
            Self::Context32 => (32769, 35840),
            Self::Context64 => (65537, 69632),
            Self::Near192 => (194561, 196474),
        }
    }
    fn bootstrap_tokens(self) -> usize {
        self.bounds().0 - 1
    }
    fn compiled(self) -> bool {
        match self {
            Self::Context32 => cfg!(feature = "qwen35-session-context32-audit"),
            Self::Context64 => cfg!(feature = "qwen35-session-context64-audit"),
            Self::Near192 => cfg!(feature = "qwen35-session-near192-audit"),
        }
    }
    fn validate_selection(self, has_input: bool, growing: bool) -> Result<(), Box<dyn Error>> {
        if !has_input || growing || !self.compiled() {
            return Err("long exact-reference profile needs pinned input, compiled feature and no growing selector".into());
        }
        Ok(())
    }
}

const STREAM_SCHEMA: &str = "qwen35-committed-text-stream-v1";
const EOS_STOP: &str = "eos";
const MAX_NEW_STOP: &str = "max_new_tokens";
const INJECTED_DISCONNECT: &str = "injected resident stream disconnect after committed output";
const DGEMM_PREFILL_SCORE_POLICY: &str = "fp64_dgemm_cast_fp32_scale_then_ordered_fp32_consumer";
const LONG_GENERATION_POLICY: &str = "long_generation_1024_v1";

#[cfg(feature = "qwen35-resident-text-service")]
#[path = "resident_service/mod.rs"]
mod resident_service;

fn validate_stream_generation_budget(
    enabled: bool,
    canonical: bool,
    dgemm: bool,
    history: usize,
    capacity: usize,
    vision: bool,
    bounded: bool,
    limit: usize,
    rounds: usize,
) -> Result<(), String> {
    if enabled
        && (!canonical
            || !dgemm
            || history != 2048
            || capacity != 196608
            || !vision
            || !bounded
            || !(1..=1024).contains(&limit)
            || !(limit..=1024).contains(&rounds))
    {
        return Err("long generation requires frozen canonical DGEMM/Vision/2048/192K profile and worst-case rounds".into());
    }
    if !enabled && (!(1..=512).contains(&limit) || !(1..=64).contains(&rounds)) {
        return Err("legacy stream generation is bounded to512 outputs/64 rounds".into());
    }
    Ok(())
}

fn bind_stream_generation_identity(request: &mut Value, enabled: bool) {
    if enabled {
        request["generation_budget_policy"] = json!(LONG_GENERATION_POLICY);
    }
}

fn validate_stream_generation_report(report: &Value, enabled: bool) -> Result<(), String> {
    let expected = json!({"schema":"qwen35-long-generation-budget-v1",
        "policy":LONG_GENERATION_POLICY, "max_new_tokens_limit":1024, "max_rolling_rounds":1024,
        "round_budget_covers_zero_acceptance":true,
        "numerical_policy_changed":false, "product_qualification":false});
    if (enabled && report.get("long_generation_budget") != Some(&expected))
        || (!enabled && report.get("long_generation_budget").is_some())
    {
        return Err("native long-generation request/report drift".into());
    }
    Ok(())
}

fn bind_stream_prefill_identity(
    request: &mut Value,
    canonical_batch: bool,
    dgemm_scores: bool,
) -> Result<(), Qwen35MixedPrefixCudaError> {
    if dgemm_scores && !canonical_batch {
        return Err(invalid(
            "DGEMM stream prefill requires canonical batch profile",
        ));
    }
    if canonical_batch {
        request["execution_profile"] = json!("canonical_batch_v1");
    }
    // Keep every existing identity unchanged when this explicit opt-in is off.
    if dgemm_scores {
        request["prefill_score_policy"] = json!(DGEMM_PREFILL_SCORE_POLICY);
    }
    Ok(())
}

fn validate_stream_prefill_report(report: &Value, dgemm_scores: bool) -> Result<(), String> {
    if report["prefill_dgemm_scores_requested"].as_bool() != Some(dgemm_scores) {
        return Err("native prefill DGEMM request/report drift".into());
    }
    if dgemm_scores {
        if report["prefill_dgemm_scores"]["score_policy"].as_str()
            != Some(DGEMM_PREFILL_SCORE_POLICY)
        {
            return Err("native prefill DGEMM score policy drift".into());
        }
    } else if !report["prefill_dgemm_scores"].is_null() {
        return Err("unselected native prefill DGEMM receipt".into());
    }
    // Full dispatch/extent/scratch lifecycle validation belongs to the
    // independently hashed request checker, not this transport boundary.
    Ok(())
}

fn validate_stream_rope_contract(
    requested: bool,
    cuda: bool,
    cpu: bool,
) -> Result<(), Qwen35MixedPrefixCudaError> {
    if requested != cuda || cuda != cpu {
        return Err(invalid(
            "stream target RoPE requires matching explicit flag and CPU/CUDA build",
        ));
    }
    Ok(())
}

fn bind_stream_rope_identity(request: &mut Value, split: bool, shared: bool) {
    // Legacy request hashes remain byte-for-byte unchanged. New numerical or
    // table-ownership identities cannot alias them in the transport envelope.
    if split {
        request["target_rope_contract"] = json!("rotary_prefix_split_half_v1");
    }
    if shared {
        request["target_rope_table_ownership"] = json!("request_shared_v1");
    }
}

// Reuse the already implemented canonical batch path as one explicit profile.
// This is request wiring only; kernels, quantization and acceptance are unchanged.
fn enable_canonical_batch_profile(
    options: &mut Qwen35DraftTargetRollingOptions,
) -> Result<(), Qwen35MixedPrefixCudaError> {
    let request = options
        .request
        .as_mut()
        .ok_or_else(|| invalid("batch stream profile requires request"))?;
    if !options.product_mode
        || !request.target_split_rope
        || !request.share_target_rope
        || !request.request_position_budget
        || request.target_only_reference
    {
        return Err(invalid(
            "batch stream profile requires split-half/shared RoPE and bounded request mode",
        ));
    }
    options.incremental_draft_context = true;
    request.retain_full_accept_state = true;
    request.batched_target_verify = true;
    request.prefill_shared_softmax = true;
    request.batched_prompt_capture = true;
    request.batched_draft_context = true;
    request.batched_draft_block = true;
    request.draft_q8_tile4 = true;
    request.parallel_receipt_digest = true;
    request.exact_projection_tiles = true;
    request.exact_output_head_batch = true;
    Ok(())
}

fn bind_stream_vision_identity(request: &mut Value, vision: Option<&(usize, String)>) {
    if let Some((bytes, sha)) = vision {
        request["vision_residency"] = json!({
            "payload_sha256": sha, "resident_device_bytes": bytes,
            "lifetime": "post_prefill_through_request_terminal_v1"
        });
    }
}

fn validate_stream_vision_report(
    report: &Value,
    expected: Option<&(usize, String)>,
) -> Result<(), String> {
    let vision = &report["vision_resident"];
    let Some((bytes, sha)) = expected else {
        return if vision.is_null() {
            Ok(())
        } else {
            Err("unexpected Vision residency".into())
        };
    };
    if vision["vision"]["payload_sha256"].as_str() != Some(sha.as_str())
        || vision["vision"]["resident_device_bytes"].as_u64() != Some(*bytes as u64)
        || vision["vision"]["device_resident"].as_bool() != Some(true)
        || vision["owner_bytes"].as_u64() != Some(*bytes as u64)
        || vision["h2d_bytes"].as_u64() != Some(*bytes as u64)
        || vision["owner_count"].as_u64() != Some(1)
        || !vision["q8_dflash"].is_null()
        || [
            "vision_forward_executed",
            "product_ready",
            "semantic_parity",
            "decode_consumer_executed",
        ]
        .iter()
        .any(|key| vision[key].as_bool() != Some(false))
    {
        return Err("Vision resident payload/extent/claim changed after CPU binding".into());
    }
    Ok(())
}

// Bounded opt-in fault probe: 1 is the initial anchor; 2 reaches a verified
// draft/target commit. It cannot turn a fault stream into successful output.
struct ResidentStreamFault {
    after_commits: usize,
    seen: Cell<usize>,
    triggered: Cell<bool>,
}

impl ResidentStreamFault {
    fn new(after_commits: usize) -> Result<Self, Qwen35MixedPrefixCudaError> {
        if !(1..=2).contains(&after_commits) {
            return Err(invalid("resident fault point must be 1 or 2 commits"));
        }
        Ok(Self {
            after_commits,
            seen: Cell::new(0),
            triggered: Cell::new(false),
        })
    }

    fn after_commit(&self) -> Result<(), Qwen35MixedPrefixCudaError> {
        if self.triggered.get() {
            return Err(invalid("commit reached an already disconnected sink"));
        }
        self.seen.set(self.seen.get() + 1);
        if self.seen.get() == self.after_commits {
            self.triggered.set(true);
            return Err(invalid(INJECTED_DISCONNECT));
        }
        Ok(())
    }
}

fn allocation_json(s: augur_backend::CudaDeviceAllocationSnapshot) -> Value {
    json!({"successful_allocations":s.successful_allocations,
        "successful_frees":s.successful_frees,"live_allocations":s.live_allocations,
        "live_bytes":s.live_bytes,"peak_live_allocations":s.peak_live_allocations,
        "peak_live_bytes":s.peak_live_bytes})
}

fn invalid(detail: impl Into<String>) -> Qwen35MixedPrefixCudaError {
    Qwen35MixedPrefixCudaError::InvalidConfiguration(detail.into())
}

fn digest(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:064x}", hasher.finalize())
}

fn write_frame<W: Write>(writer: &mut W, frame: Value) -> Result<(), Qwen35MixedPrefixCudaError> {
    let bytes = serde_json::to_vec(&frame).map_err(|error| invalid(error.to_string()))?;
    writer
        .write_all(&bytes)
        .and_then(|_| writer.write_all(b"\n"))
        .map_err(|error| invalid(format!("stream write failed: {error}")))?;
    writer
        .flush()
        .map_err(|error| invalid(format!("stream flush failed: {error}")))
}

struct CommittedStream<W: Write> {
    writer: W,
    tokenizer: Tokenizer,
    request_id: String,
    request_sha256: String,
    eos_token_ids: Vec<u32>,
    sequence: u64,
    last_commit: Option<usize>,
    initial_anchor_seen: bool,
    next_transport_commit_index: usize,
    visible_token_ids: Vec<u32>,
    // A verified BPE block may end inside a UTF-8 code point. Keep its
    // exact IDs until a later verified block completes the visible text.
    pending_token_ids: Vec<u32>,
    terminal_text_stop: Option<String>,
    text: String,
    profile_terminal: bool,
    last_text_published: Option<std::time::Instant>,
}

impl<W: Write> CommittedStream<W> {
    fn begin(
        mut writer: W,
        tokenizer: Tokenizer,
        request_id: String,
        request_sha256: String,
        eos_token_ids: Vec<u32>,
        tokenizer_sha256: String,
    ) -> Result<Self, Qwen35MixedPrefixCudaError> {
        write_frame(
            &mut writer,
            json!({
                "schema": STREAM_SCHEMA,
                "event": "begin",
                "request_id": request_id,
                "request_sha256": request_sha256,
                "sequence": 0,
                "tokenizer_sha256": tokenizer_sha256,
                "terminal_text_flush_supported": true,
            }),
        )?;
        Ok(Self {
            writer,
            tokenizer,
            request_id,
            request_sha256,
            eos_token_ids,
            sequence: 1,
            last_commit: None,
            initial_anchor_seen: false,
            next_transport_commit_index: 0,
            visible_token_ids: Vec::new(),
            pending_token_ids: Vec::new(),
            terminal_text_stop: None,
            text: String::new(),
            profile_terminal: false,
            last_text_published: None,
        })
    }

    fn push_commit(
        &mut self,
        commit: &Qwen35VerifiedCommit,
    ) -> Result<(), Qwen35MixedPrefixCudaError> {
        if self.terminal_text_stop.is_some() {
            return Err(invalid("verified commit after text terminal preparation"));
        }
        if commit.initial_anchor {
            if self.initial_anchor_seen || self.last_commit.is_some() {
                return Err(invalid("duplicate or late initial anchor"));
            }
        } else if !self.initial_anchor_seen {
            return Err(invalid("rolling commit arrived before initial anchor"));
        }
        if self
            .last_commit
            .is_some_and(|last| commit.block_index <= last)
        {
            return Err(invalid("stream commit index is not strictly increasing"));
        }
        let mut published = commit.token_ids.clone();
        if commit.stop_reason.as_deref() == Some(EOS_STOP) {
            if published.is_empty() {
                return Err(invalid("EOS commit has no token"));
            }
            let eos = published
                .last()
                .copied()
                .ok_or_else(|| invalid("EOS commit has no terminal token"))?;
            if !self.eos_token_ids.contains(&eos) {
                return Err(invalid("EOS commit does not end in a configured EOS token"));
            }
            // EOS terminates the request but is not visible text. The native
            // callback has already validated that it is the final committed
            // token in this publication block.
            published.pop();
        }
        if published.is_empty() {
            if commit.initial_anchor {
                self.initial_anchor_seen = true;
            } else {
                self.last_commit = Some(commit.block_index);
            }
            return Ok(());
        }
        let mut cumulative = self.visible_token_ids.clone();
        cumulative.extend_from_slice(&self.pending_token_ids);
        cumulative.extend_from_slice(&published);
        let decoded = self
            .tokenizer
            .decode(&cumulative, true)
            .map_err(|error| invalid(format!("tokenizer decode failed: {error}")))?;
        let text = decoded
            .strip_prefix(&self.text)
            .ok_or_else(|| invalid("decoded committed prefix is not append-only"))?
            .to_owned();
        if text.contains('\u{fffd}') {
            // Tokenizer::decode is lossy for a trailing partial byte sequence.
            // Publishing its replacement now would both corrupt the stream
            // and make the next, completed decode cease to be append-only.
            // Delay only publication, never target verification or token order.
            // If final tokens really decode to a replacement (including a
            // deliberate literal U+FFFD), the terminal flush binds that exact
            // full decode. No guessed continuation byte/token is appended.
            self.pending_token_ids.extend_from_slice(&published);
            if commit.initial_anchor {
                self.initial_anchor_seen = true;
            } else {
                self.last_commit = Some(commit.block_index);
            }
            return Ok(());
        }
        if text.is_empty() {
            return Err(invalid("committed token block has empty visible text"));
        }
        let mut transport_tokens = std::mem::take(&mut self.pending_token_ids);
        transport_tokens.extend_from_slice(&published);
        self.publish_text(cumulative, decoded, transport_tokens, None)?;
        if commit.initial_anchor {
            self.initial_anchor_seen = true;
        } else {
            self.last_commit = Some(commit.block_index);
        }
        Ok(())
    }

    fn publish_text(
        &mut self,
        cumulative: Vec<u32>,
        decoded: String,
        transport_tokens: Vec<u32>,
        terminal_flush: Option<&str>,
    ) -> Result<(), Qwen35MixedPrefixCudaError> {
        let text = decoded.strip_prefix(&self.text)
            .ok_or_else(|| invalid("decoded committed prefix is not append-only"))?
            .to_owned();
        if text.is_empty() || transport_tokens.is_empty() {
            return Err(invalid("committed token block has empty visible text"));
        }
        self.visible_token_ids = cumulative;
        self.text = decoded;
        let mut frame = json!({
                "schema": STREAM_SCHEMA,
                "event": "text",
                "request_id": self.request_id,
                "request_sha256": self.request_sha256,
                "sequence": self.sequence,
                "token_ids": transport_tokens,
                "text": text,
                "commit_index": self.next_transport_commit_index,
            });
        if let Some(stop) = terminal_flush {
            frame["terminal_decode_flush"] = json!(stop);
        }
        write_frame(&mut self.writer, frame)?;
        if self.profile_terminal {
            self.last_text_published = Some(std::time::Instant::now());
        }
        self.sequence = self
            .sequence
            .checked_add(1)
            .ok_or_else(|| invalid("stream sequence overflow"))?;
        self.next_transport_commit_index = self
            .next_transport_commit_index
            .checked_add(1)
            .ok_or_else(|| invalid("stream commit index overflow"))?;
        Ok(())
    }

    fn prepare_terminal_text(&mut self, stop: &str) -> Result<(), Qwen35MixedPrefixCudaError> {
        if !matches!(stop, EOS_STOP | MAX_NEW_STOP) {
            return Err(invalid("text flush requires a successful native stop"));
        }
        if let Some(prior) = &self.terminal_text_stop {
            return if prior == stop { Ok(()) } else { Err(invalid("text terminal stop changed")) };
        }
        self.terminal_text_stop = Some(stop.to_owned());
        if self.pending_token_ids.is_empty() {
            return Ok(());
        }
        let mut cumulative = self.visible_token_ids.clone();
        cumulative.extend_from_slice(&self.pending_token_ids);
        let decoded = self.tokenizer.decode(&cumulative, true)
            .map_err(|error| invalid(format!("terminal tokenizer decode failed: {error}")))?;
        let pending = std::mem::take(&mut self.pending_token_ids);
        self.publish_text(cumulative, decoded, pending, Some(stop))
    }

    #[cfg(feature = "qwen35-session-continuation")]
    fn finish_session(
        mut self,
        stop_reason: &str,
        native_report_sha256: String,
        session: Value,
        terminal_stages: Option<Value>,
    ) -> Result<(), Qwen35MixedPrefixCudaError> {
        if !matches!(stop_reason, EOS_STOP | MAX_NEW_STOP) {
            return Err(invalid("session stream terminal stop"));
        }
        self.prepare_terminal_text(stop_reason)?;
        let mut frame = json!({"schema":STREAM_SCHEMA,"event":"terminal","request_id":self.request_id,
            "request_sha256":self.request_sha256,"sequence":self.sequence,"stop_reason":stop_reason,
            "visible_token_ids":self.visible_token_ids,"text_sha256":digest(self.text.as_bytes()),"native_report_sha256":native_report_sha256,
            "session":session,"product_ready":false});
        if let Some(mut timing) = terminal_stages {
            timing["last_text_to_terminal_frame_prepare_ms"] = json!(self.last_text_published
                .map(|t| t.elapsed().as_secs_f64() * 1000.0));
            timing["last_text_sequence"] = json!(self.sequence.checked_sub(1));
            timing["scope"] = json!("producer_host_intervals_not_CUDA_or_network_duration");
            frame["terminal_stage_timing"] = timing;
        }
        write_frame(&mut self.writer, frame)
    }

    fn finish(
        mut self,
        stop_reason: &str,
        native_report_sha256: String,
    ) -> Result<(), Qwen35MixedPrefixCudaError> {
        if !matches!(stop_reason, EOS_STOP | MAX_NEW_STOP) {
            return Err(invalid("native stream ended before request terminal"));
        }
        self.prepare_terminal_text(stop_reason)?;
        write_frame(
            &mut self.writer,
            json!({
                "schema": STREAM_SCHEMA,
                "event": "terminal",
                "request_id": self.request_id,
                "request_sha256": self.request_sha256,
                "sequence": self.sequence,
                "stop_reason": stop_reason,
                "visible_token_ids": self.visible_token_ids,
                "text_sha256": digest(self.text.as_bytes()),
                "native_report_sha256": native_report_sha256,
            }),
        )
    }
}

fn next_value(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<String, Box<dyn Error>> {
    args.next()
        .ok_or_else(|| format!("missing value for {flag}").into())
}

fn open_stream(path: &str) -> Result<Box<dyn Write>, Box<dyn Error>> {
    if path == "-" {
        return Ok(Box::new(io::stdout()));
    }
    let file = OpenOptions::new().write(true).create_new(true).open(path)?;
    Ok(Box::new(BufWriter::new(file)))
}

fn main() -> Result<(), Box<dyn Error>> {
    if augur_model_qwen35_runtime::build_identity::print_cli_identity("producer")? {
        return Ok(());
    }
    let mut args = env::args().skip(1);
    let mut config = Qwen35AutoregressiveCudaConfig::default();
    let mut q8_path = None::<PathBuf>;
    let mut draft_config_path = None::<PathBuf>;
    let mut generation_config_path = None::<PathBuf>;
    let mut tokenizer_path = None::<PathBuf>;
    let mut stream_output = None::<String>;
    let mut prompt_file = None::<PathBuf>;
    let mut resident_prompts = None::<PathBuf>;
    let mut resident_target_weights = false;
    let mut resident_service_stdin = false;
    let mut resident_session_stdin = false;
    let mut cooperative_owner_cancel = false;
    let mut resident_session_growing = false;
    let mut resident_session_dynamic = false;
    let mut checkpoint_profile = None::<PathBuf>;
    let mut checkpoint_profile_sha256 = None::<String>;
    let mut native_input_prefix_probe = None::<PathBuf>;
    let mut input_prefix_cache_tokens = None::<usize>;
    let mut resident_session_selectors = 0usize;
    let mut resident_session_context32 = false;
    let mut resident_session_context64 = false;
    let mut resident_session_near192 = false;
    let mut native_session_smoke = false;
    let mut native_growing_session_smoke = false;
    let mut native_session_context32 = false;
    let mut native_session_context64 = false;
    let mut native_session_near192 = false;
    let mut native_session_long_audit = false;
    let mut native_session_cancel = false;
    let mut native_transition_audit = false;
    let mut native_prefill_scalar_audit = false;
    let mut native_prefill_full_canonical_audit = false;
    let mut native_session_suffix_case = None::<String>;
    let mut native_exact_reference = None::<PathBuf>;
    let mut native_growing_reference = false;
    let mut native_exact_reference_sha256 = None::<String>;
    let mut native_long_reference_profile = None::<NativeLongReferenceProfile>;
    let mut resident_fault_after_commits = None::<usize>;
    let mut target_split_rope = false;
    let mut share_target_rope = false;
    let mut canonical_batch_profile = false;
    let mut prefill_dgemm_scores = false;
    let mut long_generation_budget = false;
    let mut report_output = None::<PathBuf>;
    let mut max_new_tokens = 8usize;
    let mut rolling_blocks = 1usize;
    let mut context_rows = 8usize;
    let mut request_position_budget = true;
    let mut capacity = 4096usize;
    let mut device = 0i32;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--input-prefix-cache-tokens" => {
                if input_prefix_cache_tokens.is_some() { return Err("duplicate input-prefix cache size".into()); }
                input_prefix_cache_tokens = Some(next_value(&mut args, "--input-prefix-cache-tokens")?.parse()?);
            }
            "--native-input-prefix-probe" => {
                if native_input_prefix_probe.is_some() { return Err("duplicate input-prefix probe".into()); }
                native_input_prefix_probe = Some(next_value(&mut args, "--native-input-prefix-probe")?.into());
            }
            "--checkpoint-profile" => {
                if checkpoint_profile.is_some() {
                    return Err("duplicate checkpoint profile".into());
                }
                checkpoint_profile = Some(next_value(&mut args, "--checkpoint-profile")?.into());
            }
            "--checkpoint-profile-sha256" => {
                if checkpoint_profile_sha256.is_some() {
                    return Err("duplicate checkpoint profile hash".into());
                }
                checkpoint_profile_sha256 =
                    Some(next_value(&mut args, "--checkpoint-profile-sha256")?);
            }
            "--resident-target-weights" => resident_target_weights = true,
            "--resident-service-stdin" => resident_service_stdin = true,
            "--resident-session-canonical-stdin" => {
                resident_session_stdin = true;
                resident_session_selectors += 1;
            }
            "--resident-session-growing-stdin" => {
                resident_session_stdin = true;
                resident_session_growing = true;
                resident_session_selectors += 1;
            }
            "--resident-session-stdin" => {
                resident_session_stdin = true;
                resident_session_dynamic = true;
                resident_session_selectors += 1;
            }
            "--resident-session-context32-stdin" => resident_session_context32 = true,
            "--resident-session-context64-stdin" => resident_session_context64 = true,
            "--resident-session-near192-stdin" => resident_session_near192 = true,
            "--cooperative-owner-cancel" => cooperative_owner_cancel = true,
            "--native-session-append-smoke" => native_session_smoke = true,
            "--native-session-growing-smoke" => {
                native_session_smoke = true;
                native_growing_session_smoke = true;
            }
            "--native-session-context32-audit" => native_session_context32 = true,
            "--native-session-context64-audit" => native_session_context64 = true,
            "--native-session-near192-audit" => native_session_near192 = true,
            "--native-session-long128-audit" => native_session_long_audit = true,
            "--native-session-cancel-smoke" => native_session_cancel = true,
            "--native-prefill-transition-audit" => native_transition_audit = true,
            "--native-prefill-canonical-scalars-audit" => native_prefill_scalar_audit = true,
            "--native-prefill-full-canonical-audit" => native_prefill_full_canonical_audit = true,
            "--native-session-suffix-case" => {
                native_session_suffix_case =
                    Some(next_value(&mut args, "--native-session-suffix-case")?);
            }
            "--native-session-growing-exact-reference" => native_growing_reference = true,
            "--native-session-exact-reference-long-profile" => {
                if native_long_reference_profile.is_some() {
                    return Err("duplicate native exact-reference long profile".into());
                }
                native_long_reference_profile = Some(NativeLongReferenceProfile::parse(
                    &next_value(&mut args, "--native-session-exact-reference-long-profile")?,
                )?);
            }
            "--native-session-exact-reference-json" => {
                native_exact_reference =
                    Some(next_value(&mut args, "--native-session-exact-reference-json")?.into());
            }
            "--native-session-exact-reference-sha256" => {
                native_exact_reference_sha256 = Some(next_value(
                    &mut args,
                    "--native-session-exact-reference-sha256",
                )?);
            }
            "--target-split-rope" => target_split_rope = true,
            "--share-target-rope" => share_target_rope = true,
            "--canonical-batch-profile" => canonical_batch_profile = true,
            "--prefill-dgemm-scores" => prefill_dgemm_scores = true,
            "--long-generation-budget" => long_generation_budget = true,
            "--resident-fault-after-commits" => {
                resident_fault_after_commits =
                    Some(next_value(&mut args, "--resident-fault-after-commits")?.parse()?);
            }
            "--gguf" => config.gguf_path = next_value(&mut args, "--gguf")?.into(),
            "--vision-safetensors" => {
                config.vision_safetensors_path =
                    Some(next_value(&mut args, "--vision-safetensors")?.into())
            }
            "--config" => config.config_path = next_value(&mut args, "--config")?.into(),
            "--tokenizer" => tokenizer_path = Some(next_value(&mut args, "--tokenizer")?.into()),
            "--prompt-text" => config.prompt_text = next_value(&mut args, "--prompt-text")?,
            "--prompt-file" => prompt_file = Some(next_value(&mut args, "--prompt-file")?.into()),
            "--report-output" => {
                report_output = Some(next_value(&mut args, "--report-output")?.into())
            }
            "--resident-draft-prompts-json" => {
                resident_prompts =
                    Some(next_value(&mut args, "--resident-draft-prompts-json")?.into())
            }
            "--q8-gguf" => q8_path = Some(next_value(&mut args, "--q8-gguf")?.into()),
            "--dflash-config" => {
                draft_config_path = Some(next_value(&mut args, "--dflash-config")?.into())
            }
            "--request-generation-config" => {
                generation_config_path =
                    Some(next_value(&mut args, "--request-generation-config")?.into())
            }
            "--stream-output" => stream_output = Some(next_value(&mut args, "--stream-output")?),
            "--max-new-tokens" => {
                max_new_tokens = next_value(&mut args, "--max-new-tokens")?.parse()?
            }
            "--rolling-blocks" => {
                rolling_blocks = next_value(&mut args, "--rolling-blocks")?.parse()?
            }
            "--context-rows" => context_rows = next_value(&mut args, "--context-rows")?.parse()?,
            "--capacity" => capacity = next_value(&mut args, "--capacity")?.parse()?,
            "--device" => device = next_value(&mut args, "--device")?.parse()?,
            "--no-request-position-budget" => request_position_budget = false,
            "--help" | "-h" => {
                println!("opt-in resident prefix slot: --input-prefix-cache-tokens N (2048..16384 aligned to 8; requires --checkpoint-profile and its SHA)");
                println!("usage: qwen35_q8_draft_target_stream_cuda --gguf PATH --config PATH --tokenizer PATH --prompt-text TEXT --q8-gguf PATH --dflash-config PATH --request-generation-config PATH --stream-output PATH|- [--max-new-tokens N] [--rolling-blocks N] [--context-rows N] [--capacity N] [--device N]");
                println!("resident audit: --resident-draft-prompts-json PATH [--resident-target-weights]; opt-in disconnect probe additionally requires --resident-target-weights --resident-fault-after-commits 1|2 (first request only, no successful terminal)");
                println!("numerical contract: --target-split-rope requires qwen35-target-split-rope CPU/CUDA build; --share-target-rope shares immutable request tables. Both are externally bound in the stream request identity.");
                println!("--canonical-batch-profile opt-in reuses batch verify/capture/draft/head and exact receipts; requires split-half, shared RoPE and request position budget. No qualification claim.");
                println!("--prefill-dgemm-scores requires canonical batch; explicitly binds FP64 DGEMM score policy in the stream identity. Default off; no qualification claim.");
                println!("--vision-safetensors PATH loads visual weights after prefill workspace release and holds them through generation; payload SHA/extent enter stream identity. No Vision forward claim.");
                return Ok(());
            }
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    // Fail before reading/loading model artifacts, not after resident upload.
    if checkpoint_profile.is_some() != checkpoint_profile_sha256.is_some()
        || (checkpoint_profile.is_some()
            && (!cfg!(all(windows, feature = "qwen35-session-checkpoint"))
                || !(resident_session_stdin
                    || resident_session_context32
                    || resident_session_context64
                    || resident_session_near192
                    || native_input_prefix_probe.is_some())))
    {
        return Err(
            "checkpoint profile requires explicit hash and compiled Windows session service".into(),
        );
    }
    if input_prefix_cache_tokens.is_some_and(|n| !(2048..=16384).contains(&n) || n % 8 != 0)
        || (input_prefix_cache_tokens.is_some() && (
            !cfg!(all(windows, feature = "qwen35-session-checkpoint"))
            || !resident_session_dynamic || checkpoint_profile.is_none()
            || native_input_prefix_probe.is_some()))
    { return Err("input-prefix cache requires resident HTTP profile, pinned checkpoint profile and aligned bounded size".into()); }
    if native_input_prefix_probe.is_some() && (
        !cfg!(all(windows, feature = "qwen35-session-continuation", feature = "qwen35-session-checkpoint"))
        || checkpoint_profile.is_none() || resident_prompts.is_some() || resident_target_weights
        || resident_service_stdin || resident_session_stdin || cooperative_owner_cancel
        || resident_session_context32 || resident_session_context64 || resident_session_near192
        || native_session_smoke || native_growing_session_smoke || native_session_context32
        || native_session_context64 || native_session_near192 || native_session_long_audit
        || native_session_cancel || native_transition_audit || native_prefill_scalar_audit
        || native_prefill_full_canonical_audit || native_session_suffix_case.is_some()
        || native_exact_reference.is_some() || native_growing_reference
        || native_exact_reference_sha256.is_some() || resident_fault_after_commits.is_some()
        || report_output.is_some() || stream_output.as_deref().is_none_or(|s| s == "-")
        || !long_generation_budget || !canonical_batch_profile || !prefill_dgemm_scores
        || !target_split_rope || !share_target_rope || !request_position_budget
        || context_rows != 2048 || capacity != 196608 || max_new_tokens != 1024 || rolling_blocks != 1024
        || config.vision_safetensors_path.is_none()
    ) { return Err("input-prefix probe requires exclusive frozen complete-residency/profile/evidence configuration".into()); }
    validate_stream_rope_contract(
        target_split_rope,
        augur_backend::cuda_q8_0::QWEN35_TARGET_SPLIT_ROPE,
        augur_model_qwen35::TARGET_ROPE_SPLIT_HALF,
    )?;
    if prefill_dgemm_scores && !canonical_batch_profile {
        return Err("--prefill-dgemm-scores requires --canonical-batch-profile".into());
    }
    if canonical_batch_profile
        && (!target_split_rope
            || !share_target_rope
            || !request_position_budget
            || context_rows < 8)
    {
        return Err("batch stream profile requires explicit split/shared RoPE, position budget and at least 8 history rows".into());
    }
    if resident_session_selectors > 1 {
        return Err("canonical and initialized-growing session modes are exclusive".into());
    }
    if cooperative_owner_cancel && !(resident_session_stdin || resident_session_context32
        || resident_session_context64 || resident_session_near192) {
        return Err("cooperative cancellation requires an explicit stdin session service".into());
    }
    if let Some(path) = prompt_file {
        config.prompt_text = fs::read_to_string(path)?;
    }
    let tokenizer_path = tokenizer_path.ok_or("--tokenizer is required")?;
    let q8_path = q8_path.ok_or("--q8-gguf is required")?;
    let draft_config_path = draft_config_path.ok_or("--dflash-config is required")?;
    let generation_config_path =
        generation_config_path.ok_or("--request-generation-config is required")?;
    let stream_output = stream_output.ok_or("--stream-output is required")?;
    if config.gguf_path.as_os_str().is_empty()
        || config.config_path.as_os_str().is_empty()
        || (config.prompt_text.is_empty()
            && resident_prompts.is_none()
            && !resident_service_stdin
            && !resident_session_stdin
            && !resident_session_context32
            && !resident_session_context64
            && !resident_session_near192)
    {
        return Err("--gguf, --config and prompt text are required".into());
    }
    validate_stream_generation_budget(
        long_generation_budget,
        canonical_batch_profile,
        prefill_dgemm_scores,
        context_rows,
        capacity,
        config.vision_safetensors_path.is_some(),
        request_position_budget,
        max_new_tokens,
        rolling_blocks,
    )?;
    if rolling_blocks < max_new_tokens.div_ceil(8) {
        return Err("--rolling-blocks must cover the request publication budget".into());
    }
    config.capacity = capacity;
    config.device_ordinal = device;
    // The native rolling runner reads the tokenizer through the shared
    // autoregressive config. Keep the stream adapter's explicit tokenizer
    // input bound to that field as well; otherwise the runner receives its
    // default empty path and fails with an opaque Windows `os error 3` before
    // model execution starts.
    config.tokenizer_path = tokenizer_path.clone();
    config.source_commit = augur_model_qwen35_runtime::build_identity::source_commit().to_owned();
    let tokenizer_bytes = fs::read(&tokenizer_path)?;
    let tokenizer_sha256 = digest(&tokenizer_bytes);
    let tokenizer = Tokenizer::from_bytes(&tokenizer_bytes)
        .map_err(|error| format!("tokenizer load failed: {error}"))?;
    let generation_config_bytes = fs::read(&generation_config_path)?;
    let generation_config: Value = serde_json::from_slice(&generation_config_bytes)?;
    let eos_token_ids = match generation_config.get("eos_token_id") {
        Some(Value::Number(value)) => vec![value
            .as_u64()
            .ok_or("generation config EOS token is not an integer")?
            .try_into()
            .map_err(|_| "generation config EOS token exceeds u32")?],
        Some(Value::Array(values)) => values
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .ok_or("generation config EOS array contains a non-integer")
                    .and_then(|id| {
                        id.try_into()
                            .map_err(|_| "generation config EOS token exceeds u32")
                    })
            })
            .collect::<Result<Vec<_>, _>>()?,
        _ => return Err("generation config must declare eos_token_id".into()),
    };
    if eos_token_ids.is_empty() || eos_token_ids.len() > 8 {
        return Err("generation config EOS token set is out of bounds".into());
    }
    #[cfg(all(windows, feature = "qwen35-session-continuation", feature = "qwen35-session-checkpoint"))]
    if let Some(specification) = native_input_prefix_probe {
        return resident_service::input_prefix_probe::run(
            config, q8_path, draft_config_path, generation_config_path, tokenizer,
            PathBuf::from(stream_output), specification,
            checkpoint_profile.zip(checkpoint_profile_sha256).ok_or("input-prefix startup profile absent")?,
        );
    }
    if resident_prompts.is_some() && report_output.is_some() {
        return Err(
            "resident draft audit creates per-request reports; omit --report-output".into(),
        );
    }
    if resident_target_weights && resident_prompts.is_none() {
        return Err("--resident-target-weights requires --resident-draft-prompts-json".into());
    }
    let resident_fault = resident_fault_after_commits
        .map(ResidentStreamFault::new)
        .transpose()?;
    if resident_fault.is_some() && (!resident_target_weights || resident_prompts.is_none()) {
        return Err("resident fault probe requires immutable target+Q8 resident mode".into());
    }
    if canonical_batch_profile && resident_fault.is_some() {
        return Err(
            "batch profile disconnect probe requires a separate qualification identity".into(),
        );
    }
    if config.vision_safetensors_path.is_some() && resident_fault.is_some() {
        return Err("Vision residency has no disconnect-probe qualification".into());
    }
    if native_growing_reference
        && (native_exact_reference.is_none() || native_growing_session_smoke)
    {
        return Err(
            "growing exact-reference selector requires the separate exact-u32 long profile".into(),
        );
    }
    if let Some(profile) = native_long_reference_profile {
        profile.validate_selection(native_exact_reference.is_some(), native_growing_reference)?;
    }
    validate_native_exact_reference(
        native_exact_reference.as_deref(),
        native_exact_reference_sha256.as_deref(),
        cfg!(feature = "qwen35-session-continuation")
            && cfg!(feature = "qwen35-draft-private-state")
            && native_session_smoke
            && native_prefill_full_canonical_audit
            && native_session_long_audit
            && !native_prefill_scalar_audit
            && !native_session_cancel
            && !native_transition_audit
            && !native_session_context32
            && !native_session_context64
            && !native_session_near192
            && !resident_service_stdin
            && !resident_session_stdin
            && !resident_session_context32
            && !resident_session_context64
            && !resident_session_near192
            && native_session_suffix_case.is_none(),
    )?;
    if resident_session_near192
        && (!cfg!(feature = "qwen35-session-near192-audit")
            || resident_session_context32
            || resident_session_context64
            || resident_session_stdin
            || resident_service_stdin
            || native_session_smoke
            || native_session_context32
            || native_session_context64
            || native_session_near192
            || native_session_long_audit
            || native_session_cancel
            || native_transition_audit
            || native_prefill_full_canonical_audit
            || native_prefill_scalar_audit
            || native_session_suffix_case.is_some())
    {
        return Err("near192 HTTP requires exclusive compiled session profile".into());
    }
    if resident_session_context32
        && (!cfg!(feature = "qwen35-session-context32-audit")
            || resident_session_context64
            || resident_session_stdin
            || resident_service_stdin
            || native_session_smoke
            || native_session_context32
            || native_session_context64
            || native_session_long_audit
            || native_session_cancel
            || native_transition_audit
            || native_prefill_full_canonical_audit
            || native_prefill_scalar_audit
            || native_session_suffix_case.is_some())
    {
        return Err("context32 HTTP requires exclusive compiled session profile".into());
    }
    if resident_session_context64
        && (!cfg!(feature = "qwen35-session-context64-audit")
            || resident_session_context32
            || resident_session_stdin
            || resident_service_stdin
            || native_session_smoke
            || native_session_context32
            || native_session_context64
            || native_session_long_audit
            || native_session_cancel
            || native_transition_audit
            || native_prefill_full_canonical_audit
            || native_prefill_scalar_audit
            || native_session_suffix_case.is_some())
    {
        return Err("context64 HTTP requires exclusive compiled session profile".into());
    }
    if resident_session_stdin
        && (resident_service_stdin
            || native_session_smoke
            || native_transition_audit
            || native_session_long_audit
            || native_session_cancel
            || native_prefill_full_canonical_audit
            || native_prefill_scalar_audit)
    {
        return Err("session service requires exclusive distinct CLI mode".into());
    }
    if native_session_cancel
        && (!native_session_smoke
            || !native_prefill_full_canonical_audit
            || native_session_long_audit
            || native_session_suffix_case.is_some()
            || native_transition_audit
            || !cfg!(feature = "qwen35-draft-private-state"))
    {
        return Err("cancel gate requires exclusive full canonical private native profile".into());
    }
    validate_native_prefill_audit_selection(
        native_session_smoke,
        native_transition_audit,
        native_prefill_scalar_audit,
        native_prefill_full_canonical_audit,
        native_session_suffix_case.as_deref(),
    )?;
    validate_native_long_audit(
        native_session_long_audit,
        native_session_smoke,
        native_prefill_full_canonical_audit,
        native_transition_audit,
        native_session_suffix_case.as_deref(),
    )?;
    if native_session_context32
        && (!cfg!(feature = "qwen35-session-context32-audit")
            || native_session_context64
            || !native_session_smoke
            || !native_session_long_audit
            || !native_prefill_full_canonical_audit
            || native_transition_audit
            || native_session_cancel
            || resident_session_stdin
            || native_session_suffix_case.as_deref() != Some("ledger-counterfactual-long"))
    {
        return Err(
            "context32 requires exclusive compiled native/full-canonical/long128 ledger profile"
                .into(),
        );
    }
    if native_session_context64
        && (!cfg!(feature = "qwen35-session-context64-audit")
            || native_session_context32
            || !native_session_smoke
            || !native_session_long_audit
            || !native_prefill_full_canonical_audit
            || native_transition_audit
            || native_session_cancel
            || resident_session_stdin
            || native_session_suffix_case.as_deref() != Some("ledger-counterfactual-long"))
    {
        return Err(
            "context64 requires exclusive compiled native/full-canonical/long128 ledger profile"
                .into(),
        );
    }
    if native_session_smoke || native_transition_audit {
        if native_growing_session_smoke
            && (!cfg!(feature = "qwen35-session-continuation")
                || native_transition_audit
                || native_session_context32
                || native_session_context64
                || native_session_near192
                || native_session_long_audit
                || native_session_cancel
                || native_prefill_scalar_audit
                || !native_prefill_full_canonical_audit
                || native_exact_reference.is_some()
                || native_session_suffix_case.is_some())
        {
            return Err(
                "growing native smoke requires exclusive full-canonical short profile".into(),
            );
        }
        if native_session_near192
            && (!cfg!(feature = "qwen35-session-near192-audit")
                || native_session_context32
                || native_session_context64
                || !native_session_smoke
                || !native_session_long_audit
                || !native_prefill_full_canonical_audit
                || native_transition_audit
                || native_session_cancel
                || resident_session_stdin
                || resident_session_context32
                || resident_session_context64
                || native_session_suffix_case.as_deref() != Some("ledger-counterfactual-long"))
        {
            return Err(
                "near192 requires exclusive compiled native/full-canonical/long128 ledger profile"
                    .into(),
            );
        }
        if (native_session_smoke && native_transition_audit)
            || resident_service_stdin
            || resident_prompts.is_some()
            || resident_target_weights
            || resident_fault.is_some()
            || report_output.is_some()
            || stream_output == "-"
            || !long_generation_budget
            || !canonical_batch_profile
            || !prefill_dgemm_scores
            || context_rows != 2048
            || capacity != 196608
            || max_new_tokens != 1024
            || rolling_blocks != 1024
            || config.vision_safetensors_path.is_none()
            || (!(native_growing_session_smoke || native_growing_reference)
                && tokenizer
                    .encode(config.prompt_text.as_str(), false)
                    .map_err(|e| e.to_string())?
                    .len()
                    != if let Some(profile) = native_long_reference_profile {
                        profile.bootstrap_tokens()
                    } else if native_session_context32 {
                        32768
                    } else if native_session_context64 {
                        65536
                    } else if native_session_near192 {
                        194560
                    } else {
                        8192
                    })
            || ((native_growing_session_smoke || native_growing_reference)
                && !(1..=16384).contains(
                    &tokenizer
                        .encode(config.prompt_text.as_str(), false)
                        .map_err(|e| e.to_string())?
                        .len(),
                ))
        {
            return Err(
                "native session smoke requires exclusive frozen 8K/full2048/192K/Vision profile"
                    .into(),
            );
        }
        #[cfg(feature = "qwen35-session-continuation")]
        if native_growing_session_smoke {
            return resident_service::growing_session_smoke::run(
                config,
                q8_path,
                draft_config_path,
                generation_config_path,
                tokenizer,
                PathBuf::from(stream_output),
            );
        }
        #[cfg(feature = "qwen35-session-continuation")]
        if let Some(input) = native_exact_reference.as_deref() {
            return resident_service::exact_reference::run(
                config,
                q8_path,
                draft_config_path,
                generation_config_path,
                PathBuf::from(stream_output),
                input,
                native_exact_reference_sha256
                    .as_deref()
                    .ok_or("missing reference SHA")?,
                native_growing_reference,
                native_long_reference_profile,
            );
        }
        #[cfg(feature = "qwen35-session-continuation")]
        if native_session_cancel {
            return resident_service::session_cancel::run(
                config,
                q8_path,
                draft_config_path,
                generation_config_path,
                tokenizer,
                PathBuf::from(stream_output),
            );
        }
        #[cfg(feature = "qwen35-session-continuation")]
        if native_transition_audit {
            return resident_service::transition_smoke::run(
                config,
                q8_path,
                draft_config_path,
                generation_config_path,
                PathBuf::from(stream_output),
            );
        }
        #[cfg(feature = "qwen35-session-continuation")]
        return resident_service::session_smoke::run(
            config,
            q8_path,
            draft_config_path,
            generation_config_path,
            tokenizer,
            PathBuf::from(stream_output),
            native_prefill_scalar_audit,
            native_prefill_full_canonical_audit,
            native_session_suffix_case
                .as_deref()
                .unwrap_or("add-aligned"),
            native_session_long_audit,
            native_session_context32,
            native_session_context64,
            native_session_near192,
        );
        #[cfg(not(feature = "qwen35-session-continuation"))]
        return Err("native session continuation feature is not compiled".into());
    }
    if native_session_near192 {
        return Err("near192 selector requires native session smoke".into());
    }
    if resident_service_stdin
        || resident_session_stdin
        || resident_session_context32
        || resident_session_context64
        || resident_session_near192
    {
        if resident_prompts.is_some()
            || resident_target_weights
            || resident_fault.is_some()
            || report_output.is_some()
            || stream_output == "-"
            || !long_generation_budget
            || !canonical_batch_profile
            || !prefill_dgemm_scores
            || context_rows != 2048
            || capacity != 196608
            || max_new_tokens != 1024
            || rolling_blocks != 1024
            || config.vision_safetensors_path.is_none()
        {
            return Err("resident service requires exclusive evidence directory and frozen canonical DGEMM/full2048/192K/Vision/1024 profile".into());
        }
        if resident_session_stdin
            || resident_session_context32
            || resident_session_context64
            || resident_session_near192
        {
            #[cfg(feature = "qwen35-session-continuation")]
            return resident_service::session_http::serve(
                config,
                q8_path,
                draft_config_path,
                generation_config_path,
                tokenizer,
                tokenizer_sha256,
                eos_token_ids,
                PathBuf::from(stream_output),
                if resident_session_dynamic {
                    augur_model_qwen35_runtime::resident_session_profile::ResidentSessionProfile::Resident
                } else {
                    augur_model_qwen35_runtime::resident_session_profile::ResidentSessionProfile::from_legacy(
                        resident_session_context32, resident_session_context64,
                        resident_session_near192, resident_session_growing)?
                },
                checkpoint_profile.zip(checkpoint_profile_sha256),
                cooperative_owner_cancel,
                input_prefix_cache_tokens,
            );
            #[cfg(not(feature = "qwen35-session-continuation"))]
            return Err("session continuation not compiled".into());
        }
        #[cfg(feature = "qwen35-resident-text-service")]
        return resident_service::serve(
            config,
            q8_path,
            draft_config_path,
            generation_config_path,
            tokenizer,
            tokenizer_sha256,
            eos_token_ids,
            generation_config_bytes,
            PathBuf::from(stream_output),
        );
        #[cfg(not(feature = "qwen35-resident-text-service"))]
        return Err("resident service feature is not compiled".into());
    }
    // Establish the selected payload identity before any CUDA/model upload.
    // Runtime re-reads the payload after prefill; a changed payload prevents
    // a successful terminal (partial text never proves residency qualification).
    let vision_identity = config
        .vision_safetensors_path
        .as_deref()
        .map(inspect_qwen35_vision_payload)
        .transpose()?;
    let resident_baseline = augur_backend::cuda_device_allocation_snapshot();
    let resident_open_started = std::time::Instant::now();
    let prompts = if let Some(path) = &resident_prompts {
        let prompts: Vec<String> = serde_json::from_slice(&fs::read(path)?)?;
        if !(2..=8).contains(&prompts.len())
            || prompts
                .iter()
                .any(|p| p.is_empty() || p.len() > 4 * 1024 * 1024)
        {
            return Err("resident draft audit requires 2..8 bounded nonempty prompts".into());
        }
        if stream_output == "-" {
            return Err("resident draft audit requires a new --stream-output directory".into());
        }
        fs::create_dir(&stream_output)?;
        prompts
    } else {
        vec![config.prompt_text.clone()]
    };
    let mut resident = if resident_prompts.is_some() {
        Some(if resident_target_weights {
            Qwen35ResidentDraftCuda::open_with_target(&config, &q8_path, &draft_config_path)?
        } else {
            Qwen35ResidentDraftCuda::open(&q8_path, &draft_config_path, device)?
        })
    } else {
        None
    };
    let resident_open_wall_ms = resident_open_started.elapsed().as_secs_f64() * 1000.0;
    for (request_index, prompt) in prompts.into_iter().enumerate() {
        config.prompt_text = prompt;
        let request_stream_output = if resident.is_some() {
            PathBuf::from(&stream_output)
                .join(format!("request-{request_index}.jsonl"))
                .to_str()
                .ok_or("non-UTF8 stream path")?
                .to_owned()
        } else {
            stream_output.clone()
        };
        let mut stream_request = json!({
            "prompt_text": config.prompt_text.clone(),
            "max_new_tokens": max_new_tokens,
            "generation_config_sha256": digest(&generation_config_bytes),
            "tokenizer_sha256": tokenizer_sha256.clone(),
            "context_rows": context_rows,
            "capacity": capacity,
        });
        bind_stream_rope_identity(&mut stream_request, target_split_rope, share_target_rope);
        bind_stream_vision_identity(&mut stream_request, vision_identity.as_ref());
        bind_stream_prefill_identity(
            &mut stream_request,
            canonical_batch_profile,
            prefill_dgemm_scores,
        )?;
        bind_stream_generation_identity(&mut stream_request, long_generation_budget);
        let request_bytes = serde_json::to_vec(&stream_request)?;
        let request_sha256 = digest(&request_bytes);
        let request_id = request_sha256[..16].to_owned();
        let writer = open_stream(&request_stream_output)?;
        let stream = RefCell::new(CommittedStream::begin(
            writer,
            tokenizer.clone(),
            request_id,
            request_sha256,
            eos_token_ids.clone(),
            tokenizer_sha256.clone(),
        )?);
        let fault_callback_boundaries = Cell::new(None);
        let fault_callback_allocations = Cell::new(None);
        let sink: &Qwen35VerifiedCommitSink<'_> = &|commit| {
            stream.borrow_mut().push_commit(commit)?;
            if let Some(fault) = &resident_fault {
                if let Err(error) = fault.after_commit() {
                    // Sample at the fault, not before the whole request, so
                    // successful prefill waits cannot stand in for error drain.
                    fault_callback_boundaries.set(Some(augur_backend::cuda_host_boundary_snapshot()));
                    fault_callback_allocations
                        .set(Some(augur_backend::cuda_device_allocation_snapshot()));
                    return Err(error);
                }
            }
            Ok(())
        };
        let request = Qwen35DraftTargetRequestOptions {
            generation_config_path: generation_config_path.clone(),
            max_new_tokens,
            long_generation_budget,
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
            share_target_rope,
            prefill_shared_softmax: false,
            prefill_dgemm_scores,
            batched_prompt_capture: false,
            profile_prefill_sparse: false,
            request_position_budget,
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
            target_split_rope,
        };
        let mut options = Qwen35DraftTargetRollingOptions {
            product_mode: true,
            incremental_draft_context: false,
            validate_draft_context_reuse: false,
            request: Some(request),
        };
        if canonical_batch_profile {
            enable_canonical_batch_profile(&mut options)?;
        }
        let retry_options = resident_fault.as_ref().map(|_| options.clone());
        let request_baseline = augur_backend::cuda_device_allocation_snapshot();
        let execution = if let Some(resident) = resident.as_mut() {
            resident.run_request(
                &config,
                context_rows,
                rolling_blocks,
                config.vision_safetensors_path.as_deref(),
                options,
                Some(sink),
            )
        } else {
            run_qwen35_q8_draft_target_rolling_cuda_with_options_and_sink(
                &config,
                &q8_path,
                &draft_config_path,
                context_rows,
                rolling_blocks,
                config.vision_safetensors_path.as_deref(),
                options,
                Some(sink),
            )
        };
        let report = match execution {
            Ok(report) if resident_fault.is_none() => report,
            Ok(_) => return Err("resident fault injection was not reached; no fault PASS".into()),
            Err(error) => {
                let Some(fault) = resident_fault.as_ref() else {
                    return Err(format!("native stream execution failed: {error}").into());
                };
                if !fault.triggered.get() || !error.to_string().contains(INJECTED_DISCONNECT) {
                    return Err(format!("unexpected resident fault-probe error: {error}").into());
                }
                let after_error = augur_backend::cuda_device_allocation_snapshot();
                let error_boundaries = augur_backend::cuda_host_boundary_snapshot()
                    .checked_delta_since(
                        fault_callback_boundaries
                            .get()
                            .ok_or("fault boundary not sampled")?,
                    )
                    .ok_or("fault-probe boundary counter regression")?;
                let allocated = after_error
                    .successful_allocations
                    .checked_sub(request_baseline.successful_allocations)
                    .ok_or("fault-probe allocation regression")?;
                let freed = after_error
                    .successful_frees
                    .checked_sub(request_baseline.successful_frees)
                    .ok_or("fault-probe free regression")?;
                if allocated == 0
                    || allocated != freed
                    || after_error.live_bytes != request_baseline.live_bytes
                    || after_error.live_allocations != request_baseline.live_allocations
                    || error_boundaries.stream_synchronizations == 0
                {
                    return Err("resident error did not drain/return to retained baseline".into());
                }
                let retry_boundaries = augur_backend::cuda_host_boundary_snapshot();
                let retry = resident
                    .as_mut()
                    .ok_or("missing fault-probe resident owner")?
                    .run_request(
                        &config,
                        context_rows,
                        rolling_blocks,
                        None,
                        retry_options.ok_or("missing fault-probe retry options")?,
                        Some(sink),
                    );
                let retry_error = retry
                    .err()
                    .ok_or("poisoned resident owner accepted retry")?;
                if !retry_error
                    .to_string()
                    .contains("poisoned by prior request failure")
                    || augur_backend::cuda_device_allocation_snapshot() != after_error
                    || augur_backend::cuda_host_boundary_snapshot() != retry_boundaries
                    || fault.seen.get() != fault.after_commits
                {
                    return Err("poisoned retry was not rejected before work".into());
                }
                let partial = stream.into_inner();
                let visible_ids = partial.visible_token_ids.clone();
                drop(partial); // Deliberately no successful terminal frame.
                let stream_bytes = fs::read(&request_stream_output)?;
                for line in stream_bytes
                    .split(|&b| b == b'\n')
                    .filter(|line| !line.is_empty())
                {
                    let frame: Value = serde_json::from_slice(line)?;
                    if frame["event"] == "terminal" {
                        return Err("fault stream contains a successful terminal".into());
                    }
                }
                drop(resident.take());
                let final_cleanup = augur_backend::cuda_device_allocation_snapshot();
                if final_cleanup.live_bytes != resident_baseline.live_bytes
                    || final_cleanup.live_allocations != resident_baseline.live_allocations
                    || final_cleanup
                        .successful_allocations
                        .checked_sub(resident_baseline.successful_allocations)
                        != final_cleanup
                            .successful_frees
                            .checked_sub(resident_baseline.successful_frees)
                {
                    return Err("fault-probe outer owner cleanup did not close".into());
                }
                let audit = json!({"schema":"qwen35-resident-disconnect-audit-v1",
                    "status":"PASS_DIAGNOSTIC_RESIDENT_DISCONNECT_POISON",
                    "binary_sha256":digest(&fs::read(env::current_exe()?)?),
                    "source_commit":config.source_commit,"source_dirty":augur_model_qwen35_runtime::build_identity::source_dirty_for(&config.source_commit),
                    "target_weights_reused":true,"fault_after_commits":fault.after_commits,
                    "commits_observed":fault.seen.get(),"injected_error":error.to_string(),
                    "retry_error":retry_error.to_string(),"retry_allocation_counters_unchanged":true,
                    "retry_host_boundaries_unchanged":true,"native_terminal_emitted":false,
                    "stream_sha256":digest(&stream_bytes),"partial_visible_token_ids":visible_ids,
                    "prompt_text_sha256":digest(config.prompt_text.as_bytes()),
                    "tokenizer_sha256":tokenizer_sha256,
                    "generation_config_sha256":digest(&generation_config_bytes),
                    "config_sha256":digest(&fs::read(&config.config_path)?),
                    "context_rows":context_rows,"capacity":capacity,"max_new_tokens":max_new_tokens,
                    "target_split_rope_requested":target_split_rope,"share_target_rope_requested":share_target_rope,
                    "request_baseline":allocation_json(request_baseline),
                    "fault_callback_allocations":allocation_json(fault_callback_allocations.get().ok_or("fault allocation not sampled")?),
                    "after_error":allocation_json(after_error),"initial_baseline":allocation_json(resident_baseline),
                    "final_cleanup":allocation_json(final_cleanup),
                    "error_path_stream_synchronizations":error_boundaries.stream_synchronizations,
                    "product_ready":false,"semantic_qualification":false,"product_tps_claim_permitted":false});
                let path = PathBuf::from(&stream_output).join("fault-audit.json");
                let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
                file.write_all(serde_json::to_string_pretty(&audit)?.as_bytes())?;
                file.flush()?;
                println!(
                    "QWEN35_RESIDENT_DISCONNECT=PASS commits={}",
                    fault.after_commits
                );
                return Ok(());
            }
        };
        validate_stream_vision_report(&report, vision_identity.as_ref())?;
        validate_stream_prefill_report(&report, prefill_dgemm_scores)?;
        validate_stream_generation_report(&report, long_generation_budget)?;
        let request_output = report
            .get("request_output")
            .ok_or("native report missing request_output")?;
        let stop_reason = request_output
            .get("stop_reason")
            .and_then(Value::as_str)
            .ok_or("native report missing terminal stop_reason")?;
        if !matches!(stop_reason, EOS_STOP | MAX_NEW_STOP) {
            return Err(
                format!("native stream ended with nonterminal stop_reason={stop_reason}").into(),
            );
        }
        let visible = request_output
            .get("visible_token_ids")
            .and_then(Value::as_array)
            .ok_or("native report missing visible_token_ids")?;
        let visible_ids = visible
            .iter()
            .map(|id| id.as_u64().ok_or("invalid native visible token"))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .map(|id| id as u32)
            .collect::<Vec<_>>();
        let report_bytes = serde_json::to_vec(&report)?;
        let report_sha256 = digest(&report_bytes);
        if let Some(path) = &report_output {
            let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
            file.write_all(&report_bytes)?;
            file.flush()?;
        }
        let mut stream = stream.into_inner();
        stream.prepare_terminal_text(stop_reason)?;
        if stream.visible_token_ids != visible_ids {
            return Err("native stream visible IDs differ from terminal report".into());
        }
        stream.finish(stop_reason, report_sha256)?;
        if resident.is_some() {
            let path =
                PathBuf::from(&stream_output).join(format!("request-{request_index}.report.json"));
            let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
            file.write_all(&report_bytes)?;
            file.flush()?;
            println!(
                "QWEN35_RESIDENT_DRAFT_REQUEST={request_index} terminal={stop_reason} visible={}",
                visible_ids.len()
            );
        } else if stream_output == "-" {
            eprintln!(
                "QWEN35_NATIVE_STREAM_REPORT={}",
                serde_json::to_string(&report)?
            );
        } else {
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
    }
    drop(resident);
    if resident_prompts.is_some() {
        let after = augur_backend::cuda_device_allocation_snapshot();
        if after.live_bytes != resident_baseline.live_bytes
            || after.live_allocations != resident_baseline.live_allocations
        {
            return Err("resident draft final cleanup did not return to baseline".into());
        }
        let summary = json!({"schema":"qwen35-resident-draft-audit-cleanup-v1",
            "target_weights_reused":resident_target_weights,"resident_open_wall_ms":resident_open_wall_ms,
            "live_bytes":after.live_bytes,"live_allocations":after.live_allocations,
            "allocations":after.successful_allocations-resident_baseline.successful_allocations,
            "frees":after.successful_frees-resident_baseline.successful_frees,
            "peak_live_bytes":after.peak_live_bytes,"resident_model_server":false,"product_ready":false});
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(PathBuf::from(&stream_output).join("cleanup.json"))?;
        file.write_all(serde_json::to_string_pretty(&summary)?.as_bytes())?;
        file.flush()?;
    }
    Ok(())
}

#[cfg(test)]
mod fault_tests {
    use super::*;

    fn byte_stream(writer: &mut Vec<u8>) -> CommittedStream<&mut Vec<u8>> {
        let tokenizer = Tokenizer::from_bytes(serde_json::to_vec(&json!({
            "version":"1.0", "truncation":null, "padding":null,
            "added_tokens":[], "normalizer":null, "pre_tokenizer":null,
            "post_processor":null, "decoder":{"type":"ByteLevel",
                "add_prefix_space":true, "trim_offsets":true, "use_regex":true},
            "model":{"type":"WordLevel", "unk_token":"[UNK]", "vocab":{
                "â":0, "Ĥ":1, "¬":2, "A":3, "[EOS]":4, "[UNK]":5,
                "Ġâ":6, "!":7
            }}
        })).unwrap()).unwrap();
        CommittedStream::begin(writer, tokenizer, "r".into(), "a".repeat(64),
            vec![4], "b".repeat(64)).unwrap()
    }

    fn text_commit(anchor: bool, index: usize, tokens: &[u32], stop: Option<&str>) -> Qwen35VerifiedCommit {
        Qwen35VerifiedCommit {
            initial_anchor: anchor, block_index: index, generation_before: 0,
            generation_after: 0, cursor_before: 0, cursor_after: 0,
            token_ids: tokens.to_vec(), stop_reason: stop.map(str::to_owned),
        }
    }

    fn stream_frames(bytes: &[u8]) -> Vec<Value> {
        std::str::from_utf8(bytes).unwrap().lines()
            .map(|line| serde_json::from_str(line).unwrap()).collect()
    }

    #[test]
    fn committed_unicode_buffers_partial_bpe_across_verified_blocks() {
        let mut bytes = Vec::new();
        let mut stream = byte_stream(&mut bytes);
        stream.push_commit(&text_commit(true, 0, &[3], None)).unwrap();
        stream.push_commit(&text_commit(false, 0, &[6], None)).unwrap();
        stream.push_commit(&text_commit(false, 1, &[1], None)).unwrap();
        assert_eq!(stream.visible_token_ids, vec![3]);
        assert_eq!(stream.pending_token_ids, vec![6, 1]);
        assert_eq!(stream.text, "A");
        stream.push_commit(&text_commit(false, 2, &[2, 7, 4], Some(EOS_STOP))).unwrap();
        assert_eq!(stream.visible_token_ids, vec![3, 6, 1, 2, 7]);
        assert!(stream.pending_token_ids.is_empty());
        assert_eq!(stream.text, "A €!");
        stream.finish(EOS_STOP, "c".repeat(64)).unwrap();
        let frames = stream_frames(&bytes);
        assert_eq!(frames.len(), 4);
        assert_eq!(frames[2]["text"], " €!");
        assert_eq!(frames[2]["token_ids"], json!([6, 1, 2, 7]));
        assert_eq!(frames[2]["sequence"], 2);
        assert_eq!(frames[2]["commit_index"], 1);
        assert_eq!(frames[3]["visible_token_ids"], json!([3, 6, 1, 2, 7]));
        assert_eq!(frames[3]["text_sha256"], digest("A €!".as_bytes()));
        assert!(!std::str::from_utf8(&bytes).unwrap().contains('\u{fffd}'));
    }

    #[test]
    fn committed_unicode_can_buffer_initial_anchor_without_reordering() {
        let mut bytes = Vec::new();
        let mut stream = byte_stream(&mut bytes);
        stream.push_commit(&text_commit(true, 0, &[0], None)).unwrap();
        assert!(stream.initial_anchor_seen);
        assert_eq!(stream.sequence, 1);
        stream.push_commit(&text_commit(false, 0, &[1, 2], Some(MAX_NEW_STOP))).unwrap();
        stream.finish(MAX_NEW_STOP, "c".repeat(64)).unwrap();
        let frames = stream_frames(&bytes);
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[1]["text"], "€");
        assert_eq!(frames[1]["token_ids"], json!([0, 1, 2]));
    }

    #[test]
    fn committed_unicode_pending_does_not_allow_duplicate_commits() {
        let mut bytes = Vec::new();
        let mut stream = byte_stream(&mut bytes);
        stream.push_commit(&text_commit(true, 0, &[3], None)).unwrap();
        stream.push_commit(&text_commit(false, 0, &[0], None)).unwrap();
        assert!(stream.push_commit(&text_commit(false, 0, &[1], None)).is_err());
        assert!(stream.push_commit(&text_commit(true, 0, &[1], None)).is_err());
        assert_eq!(stream.pending_token_ids, vec![0]);
    }

    #[test]
    fn committed_unicode_terminal_flush_keeps_exact_ids_and_tokenizer_text() {
        for stop in [EOS_STOP, MAX_NEW_STOP] {
            let mut bytes = Vec::new();
            let mut stream = byte_stream(&mut bytes);
            stream.push_commit(&text_commit(true, 0, &[0], None)).unwrap();
            stream.finish(stop, "c".repeat(64)).unwrap();
            let frames = stream_frames(&bytes);
            assert_eq!(frames.len(), 3);
            assert_eq!(frames[0]["terminal_text_flush_supported"], true);
            assert_eq!(frames[1]["text"], "\u{fffd}");
            assert_eq!(frames[1]["token_ids"], json!([0]));
            assert_eq!(frames[1]["terminal_decode_flush"], stop);
            assert_eq!(frames[2]["visible_token_ids"], json!([0]));
            assert_eq!(frames[2]["stop_reason"], stop);
            assert_eq!(frames[2]["text_sha256"], digest("\u{fffd}".as_bytes()));
        }
        let mut bytes = Vec::new();
        let mut stream = byte_stream(&mut bytes);
        stream.push_commit(&text_commit(true, 0, &[0], None)).unwrap();
        stream.push_commit(&text_commit(false, 0, &[4], Some(EOS_STOP))).unwrap();
        assert_eq!(stream.pending_token_ids, vec![0]);
        stream.finish(EOS_STOP, "c".repeat(64)).unwrap();
        assert_eq!(stream_frames(&bytes)[1]["token_ids"], json!([0]));
    }

    #[test]
    fn committed_unicode_replacement_is_only_published_at_bound_terminal() {
        let mut bytes = Vec::new();
        let mut stream = byte_stream(&mut bytes);
        stream.push_commit(&text_commit(true, 0, &[0, 3], None)).unwrap();
        assert!(stream.visible_token_ids.is_empty());
        assert!(stream.text.is_empty());
        assert!(stream.prepare_terminal_text("cancelled").is_err());
        stream.prepare_terminal_text(MAX_NEW_STOP).unwrap();
        assert_eq!(stream.text, "\u{fffd}A");
        let sequence = stream.sequence;
        stream.prepare_terminal_text(MAX_NEW_STOP).unwrap();
        assert_eq!(stream.sequence, sequence);
        assert!(stream.prepare_terminal_text(EOS_STOP).is_err());
        assert!(stream.push_commit(&text_commit(false, 0, &[3], None)).is_err());
        stream.finish(MAX_NEW_STOP, "c".repeat(64)).unwrap();
        assert_eq!(stream_frames(&bytes)[1]["terminal_decode_flush"], MAX_NEW_STOP);
    }

    #[cfg(feature = "qwen35-session-continuation")]
    #[test]
    fn committed_unicode_session_terminal_flushes_before_receipt() {
        let mut bytes = Vec::new();
        let mut stream = byte_stream(&mut bytes);
        stream.push_commit(&text_commit(true, 0, &[0], None)).unwrap();
        stream.prepare_terminal_text(MAX_NEW_STOP).unwrap();
        assert_eq!(stream.visible_token_ids, vec![0]);
        assert_eq!(stream.text, "\u{fffd}");
        stream.finish_session(MAX_NEW_STOP, "c".repeat(64), json!({}), None).unwrap();
        assert_eq!(stream_frames(&bytes).len(), 3);
    }

    #[cfg(feature = "qwen35-session-continuation")]
    #[test]
    fn committed_terminal_timing_is_opt_in_and_keeps_stream_identity() {
        for profile in [false, true] {
            let mut bytes = Vec::new();
            let mut stream = byte_stream(&mut bytes);
            stream.profile_terminal = profile;
            stream.push_commit(&text_commit(true, 0, &[3], None)).unwrap();
            assert_eq!(stream.last_text_published.is_some(), profile);
            stream.finish_session(MAX_NEW_STOP, "c".repeat(64), json!({}),
                profile.then(|| json!({"native_return_text_sequence":1}))).unwrap();
            let frames = stream_frames(&bytes);
            assert_eq!(frames.len(), 3);
            assert_eq!(frames[1]["token_ids"], json!([3]));
            assert_eq!(frames[2]["visible_token_ids"], json!([3]));
            assert_eq!(frames[2]["native_report_sha256"], "c".repeat(64));
            assert_eq!(frames[2]["product_ready"], false);
            assert_eq!(frames[2].get("terminal_stage_timing").is_some(), profile);
            if profile {
                let t = &frames[2]["terminal_stage_timing"];
                assert_eq!(t["last_text_sequence"], 1);
                assert!(t["last_text_to_terminal_frame_prepare_ms"].as_f64().unwrap() >= 0.0);
            }
        }
    }

    #[test]
    fn long_generation_is_explicit_and_keeps_legacy_caps() {
        let valid = |on, n, rounds| {
            validate_stream_generation_budget(on, true, true, 2048, 196608, true, true, n, rounds)
        };
        assert!(valid(true, 1024, 1024).is_ok());
        assert!(valid(false, 512, 64).is_ok());
        for (on, n, rounds) in [
            (false, 1024, 1024),
            (true, 1024, 128),
            (true, 1025, 1024),
            (true, 1024, 1025),
            (true, 0, 1),
        ] {
            assert!(valid(on, n, rounds).is_err());
        }
        for (canonical, dgemm, history, capacity, vision, bounded) in [
            (false, true, 2048, 196608, true, true),
            (true, false, 2048, 196608, true, true),
            (true, true, 1024, 196608, true, true),
            (true, true, 2048, 262144, true, true),
            (true, true, 2048, 196608, false, true),
            (true, true, 2048, 196608, true, false),
        ] {
            assert!(validate_stream_generation_budget(
                true, canonical, dgemm, history, capacity, vision, bounded, 1024, 1024
            )
            .is_err());
        }
    }

    #[test]
    fn long_generation_identity_and_terminal_receipt_cannot_alias_legacy() {
        let mut request = json!({"prompt_text":"same"});
        let old = request.clone();
        bind_stream_generation_identity(&mut request, false);
        assert_eq!(request, old);
        bind_stream_generation_identity(&mut request, true);
        assert_eq!(request["generation_budget_policy"], LONG_GENERATION_POLICY);
        let mut report = json!({"long_generation_budget":{
            "schema":"qwen35-long-generation-budget-v1", "policy":LONG_GENERATION_POLICY,
            "max_new_tokens_limit":1024, "max_rolling_rounds":1024,
            "round_budget_covers_zero_acceptance":true, "numerical_policy_changed":false,
            "product_qualification":false}});
        assert!(validate_stream_generation_report(&report, true).is_ok());
        assert!(validate_stream_generation_report(&report, false).is_err());
        assert!(validate_stream_generation_report(&json!({}), true).is_err());
        report["long_generation_budget"]["max_new_tokens_limit"] = json!(512);
        assert!(validate_stream_generation_report(&report, true).is_err());
    }

    #[test]
    fn terminal_prefill_policy_must_match_explicit_identity() {
        let retained = json!({"prefill_dgemm_scores_requested":false,"prefill_dgemm_scores":null});
        let dgemm = json!({"prefill_dgemm_scores_requested":true,
            "prefill_dgemm_scores":{"score_policy":DGEMM_PREFILL_SCORE_POLICY}});
        assert!(validate_stream_prefill_report(&retained, false).is_ok());
        assert!(validate_stream_prefill_report(&dgemm, true).is_ok());
        assert!(validate_stream_prefill_report(&retained, true).is_err());
        assert!(validate_stream_prefill_report(&dgemm, false).is_err());
        assert!(validate_stream_prefill_report(&json!({}), false).is_err());
        let mut wrong = dgemm.clone();
        wrong["prefill_dgemm_scores"]["score_policy"] = json!("fp32");
        assert!(validate_stream_prefill_report(&wrong, true).is_err());
    }

    #[test]
    fn dgemm_prefill_identity_is_opt_in_and_cannot_alias_retained() {
        let old = json!({"prompt_text":"same"});
        let mut serial = old.clone();
        bind_stream_prefill_identity(&mut serial, false, false).unwrap();
        assert_eq!(serial, old);
        let mut retained = old.clone();
        bind_stream_prefill_identity(&mut retained, true, false).unwrap();
        assert_eq!(
            retained,
            json!({"prompt_text":"same","execution_profile":"canonical_batch_v1"})
        );
        let mut dgemm = old.clone();
        bind_stream_prefill_identity(&mut dgemm, true, true).unwrap();
        assert_eq!(dgemm["prefill_score_policy"], DGEMM_PREFILL_SCORE_POLICY);
        assert_ne!(
            digest(&serde_json::to_vec(&retained).unwrap()),
            digest(&serde_json::to_vec(&dgemm).unwrap())
        );
        let mut invalid_request = old.clone();
        assert!(bind_stream_prefill_identity(&mut invalid_request, false, true).is_err());
        assert_eq!(invalid_request, old);
    }

    #[test]
    fn vision_identity_is_explicit_and_separate_from_text_only() {
        let mut request = json!({"prompt_text":"same"});
        let old = request.clone();
        bind_stream_vision_identity(&mut request, None);
        assert_eq!(request, old);
        bind_stream_vision_identity(&mut request, Some(&(8, "a".repeat(64))));
        assert_ne!(request, old);
        assert_eq!(request["vision_residency"]["resident_device_bytes"], 8);
        assert_eq!(
            request["vision_residency"]["lifetime"],
            "post_prefill_through_request_terminal_v1"
        );
    }

    #[test]
    fn vision_terminal_requires_actual_selected_payload_and_closed_claims() {
        let identity = (8, "a".repeat(64));
        let report = json!({"vision_resident": {
            "vision":{"payload_sha256":identity.1,"resident_device_bytes":8,"device_resident":true},
            "owner_bytes":8,"h2d_bytes":8,"owner_count":1,"q8_dflash":null,
            "vision_forward_executed":false,"product_ready":false,
            "semantic_parity":false,"decode_consumer_executed":false
        }});
        assert!(validate_stream_vision_report(&report, Some(&identity)).is_ok());
        assert!(validate_stream_vision_report(&report, None).is_err());
        assert!(
            validate_stream_vision_report(&json!({"vision_resident":null}), Some(&identity))
                .is_err()
        );
        assert!(validate_stream_vision_report(&report, Some(&(9, identity.1.clone()))).is_err());
        assert!(validate_stream_vision_report(&report, Some(&(8, "b".repeat(64)))).is_err());
        for key in [
            "vision_forward_executed",
            "product_ready",
            "semantic_parity",
            "decode_consumer_executed",
        ] {
            let mut wrong = report.clone();
            wrong["vision_resident"][key] = json!(true);
            assert!(validate_stream_vision_report(&wrong, Some(&identity)).is_err());
        }
    }

    #[test]
    fn native_prefill_audit_flags_cannot_fall_through_or_mix() {
        assert!(validate_native_prefill_audit_selection(false, false, false, false, None).is_ok());
        for (scalar, full) in [(true, false), (false, true)] {
            assert!(
                validate_native_prefill_audit_selection(true, false, scalar, full, None).is_ok()
            );
            assert!(
                validate_native_prefill_audit_selection(false, false, scalar, full, None).is_err()
            );
            assert!(
                validate_native_prefill_audit_selection(true, true, scalar, full, None).is_err()
            );
        }
        assert!(validate_native_prefill_audit_selection(true, false, true, true, None).is_err());
    }

    #[test]
    fn diagnostic_suffix_selection_is_closed_and_full_only() {
        for case in [
            "add-aligned",
            "subtract-nonaligned",
            "code-output",
            "code-bitmix",
            "ledger-recall",
            "literal-copy",
            "ledger-counterfactual-long",
            "ledger-counterfactual-canonical-long",
        ] {
            assert!(
                validate_native_prefill_audit_selection(true, false, false, true, Some(case))
                    .is_ok()
            );
            assert!(
                validate_native_prefill_audit_selection(true, false, false, false, Some(case))
                    .is_err()
            );
        }
        assert!(
            validate_native_prefill_audit_selection(true, false, false, true, Some("unknown"))
                .is_err()
        );
    }

    #[test]
    fn long_native_fixture_cannot_fall_into_short_or_http() {
        assert!(validate_native_long_audit(
            false,
            true,
            true,
            false,
            Some("ledger-counterfactual-long")
        )
        .is_err());
        for (session, full, transition) in [
            (false, true, false),
            (true, false, false),
            (true, true, true),
        ] {
            assert!(validate_native_long_audit(true, session, full, transition, None).is_err());
        }
        assert_eq!(
            validate_native_long_audit(true, true, true, false, Some("ledger-counterfactual-long"))
                .is_ok(),
            cfg!(feature = "qwen35-draft-private-state")
        );
        assert!(validate_native_long_audit(false, false, false, false, None).is_ok());
    }
    #[test]
    fn canonical_ledger_reference_requires_long_native_profile() {
        let case = Some("ledger-counterfactual-canonical-long");
        assert!(validate_native_long_audit(false, true, true, false, case).is_err());
        assert!(validate_native_prefill_audit_selection(false, false, false, true, case).is_err());
        assert!(validate_native_prefill_audit_selection(true, false, false, false, case).is_err());
        assert_eq!(
            validate_native_long_audit(true, true, true, false, case).is_ok(),
            cfg!(feature = "qwen35-draft-private-state")
        );
    }

    #[test]
    fn canonical_batch_profile_requires_a_real_request() {
        let mut options = Qwen35DraftTargetRollingOptions::default();
        assert!(enable_canonical_batch_profile(&mut options).is_err());
        assert!(!options.incremental_draft_context);
    }

    #[test]
    fn rope_flag_must_match_both_compiled_contracts() {
        for requested in [false, true] {
            for cuda in [false, true] {
                for cpu in [false, true] {
                    assert_eq!(
                        validate_stream_rope_contract(requested, cuda, cpu).is_ok(),
                        requested == cuda && cuda == cpu
                    );
                }
            }
        }
    }

    #[test]
    fn rope_transport_identity_does_not_alias_legacy_or_table_ownership() {
        let legacy = json!({"prompt_text":"same prompt"});
        let mut hashes = std::collections::BTreeSet::new();
        for split in [false, true] {
            for shared in [false, true] {
                let mut value = legacy.clone();
                bind_stream_rope_identity(&mut value, split, shared);
                if !split && !shared {
                    assert_eq!(value, legacy);
                }
                assert_eq!(value.get("target_rope_contract").is_some(), split);
                assert_eq!(value.get("target_rope_table_ownership").is_some(), shared);
                assert!(hashes.insert(digest(&serde_json::to_vec(&value).unwrap())));
            }
        }
    }

    #[test]
    fn anchor_fault_is_exact_and_sticky() {
        let fault = ResidentStreamFault::new(1).unwrap();
        assert!(fault
            .after_commit()
            .unwrap_err()
            .to_string()
            .contains(INJECTED_DISCONNECT));
        assert!(fault.triggered.get());
        assert_eq!(fault.seen.get(), 1);
        assert!(fault.after_commit().is_err());
        assert_eq!(fault.seen.get(), 1);
    }

    #[test]
    fn verified_block_fault_waits_for_second_commit() {
        let fault = ResidentStreamFault::new(2).unwrap();
        fault.after_commit().unwrap();
        assert!(!fault.triggered.get());
        assert!(fault
            .after_commit()
            .unwrap_err()
            .to_string()
            .contains(INJECTED_DISCONNECT));
        assert_eq!(fault.seen.get(), 2);
    }

    #[test]
    fn fault_points_are_bounded() {
        for value in [0, 3, usize::MAX] {
            assert!(ResidentStreamFault::new(value).is_err());
        }
    }
}
