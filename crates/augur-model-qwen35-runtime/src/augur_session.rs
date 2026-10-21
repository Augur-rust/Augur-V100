//! Exact token-session groundwork for Augur, separate from the legacy GGUF owner.
//! Full 256Ki KV8 and the original BF16 head are mandatory. This module does not
//! confer HTTP, long-history, DFlash, Vision or installed-release qualification.
#[cfg(any(test, feature = "qwen35-augur-session"))]
use crate::session_cancellation::SessionCancellationToken;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
#[cfg(any(test, feature = "qwen35-augur-session"))]
use std::{collections::HashSet, time::Instant};
use std::{error::Error, fs::OpenOptions, io::Write, path::Path};

pub mod selector;
mod context;
pub use context::{context_capacity, context_output_limit, fits_context, parse_context_capacity, CONTEXT_CHOICES};

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;
#[cfg(not(feature = "qwen35-augur-http"))]
pub const PROFILE: &str = "quasar_256ki_kv8_bf16_head_greedy_v1";
#[cfg(feature = "qwen35-augur-http")]
pub const PROFILE: &str = "quasar_b4352_rotated_decode_http_v1";
pub const DFLASH_B8_PROFILE:&str="quasar_b4352_dflash_b8_packed_http_v1";
/// Explicit numerical-policy opt-in. Resident-only remains the product default.
pub fn parse_dflash_mode(value:Option<&str>)->Result<bool> {
    match value {None|Some("resident_only")=>Ok(false),Some("batched8")=>Ok(true),
        _=>Err("AUGUR_HTTP_DFLASH must be resident_only or batched8".into())}
}
pub fn requested_dflash_enabled()->Result<bool> {
    let value=std::env::var("AUGUR_HTTP_DFLASH");
    let enabled=match value {Ok(v)=>parse_dflash_mode(Some(&v))?,Err(std::env::VarError::NotPresent)=>false,Err(e)=>return Err(e.into())};
    if enabled && !cfg!(feature="qwen35-augur-http") {return Err("B8 DFlash HTTP not compiled".into());}Ok(enabled)
}
pub fn parse_dflash_plan_cache(value:Option<&str>)->Result<bool> {
    match value {None|Some("recreate")=>Ok(false),Some("park")=>Ok(true),
        _=>Err("AUGUR_HTTP_DFLASH_PLAN must be recreate or park".into())}
}
pub fn resolve_dflash_plan_cache(active:bool,value:Option<&str>)->Result<bool> {
    let enabled=match value {None=>active,Some(v)=>parse_dflash_plan_cache(Some(v))?};
    if enabled && !active {return Err("parked B8 plan requires active batched8 HTTP".into());}Ok(enabled)
}
/// Active B8 reuses quiescent plans by default; recreate is the explicit rollback.
/// Resident-only remains the overall default and owns no B8 plan.
pub fn requested_dflash_plan_cache()->Result<bool> {
    let active=requested_dflash_enabled()?;
    match std::env::var("AUGUR_HTTP_DFLASH_PLAN") {
        Ok(v)=>resolve_dflash_plan_cache(active,Some(&v)),
        Err(std::env::VarError::NotPresent)=>resolve_dflash_plan_cache(active,None),Err(e)=>Err(e.into()),
    }
}
pub fn requested_profile()->Result<&'static str> {requested_dflash_plan_cache()?;Ok(if requested_dflash_enabled()?{DFLASH_B8_PROFILE}else{PROFILE})}
pub const CAPACITY: usize = 262144;
pub const VOCAB: u32 = 248320;
pub const MAX_OUTPUT: usize = 249036;
pub const PREFILL_CHUNK: usize = 128; // divisible by the frozen FFN B8 width
pub const NUMERICAL_POLICY: &str = "checkpoint_quasar_qpn_all_workspace_shape_small_pair_f64_v1";
pub mod images;

pub fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}
pub fn valid_digest(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
pub fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
#[cfg(any(test, feature = "qwen35-augur-session"))]
fn token_digest(ids: &[u32]) -> String {
    let mut h = Sha256::new();
    for id in ids {
        h.update(id.to_le_bytes());
    }
    format!("{h:x}", h = h.finalize())
}
#[cfg(any(test, feature = "qwen35-augur-session"))]
fn valid_tokens(tokens: &[u32]) -> bool {
    tokens.iter().all(|&t| t < VOCAB)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TurnInput {
    Open {
        prompt_token_ids: Vec<u32>,
    },
    OpenImages {
        prompt_token_ids: Vec<u32>,
        images: Vec<images::ImageInput>,
    },
    // History is the exact consumed sequence, excluding the pending token.
    // The owner consumes pending exactly once before appending this suffix.
    Append {
        revision: u64,
        exact_history: Vec<u32>,
        suffix_token_ids: Vec<u32>,
    },
    AppendImages {
        revision: u64,
        exact_history: Vec<u32>,
        suffix_token_ids: Vec<u32>,
        image_input_sha256: String,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TurnRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selector_policy: Option<selector::Policy>,
    pub request_id: String,
    pub session_id: String,
    pub max_new_tokens: usize,
    pub input: TurnInput,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionSnapshot {
    pub profile: String,
    pub manifest_sha256: String,
    pub session_id: String,
    pub revision: u64,
    pub consumed_token_ids: Vec<u32>,
    pub consumed_input_sha256: String,
    pub pending_token: u32,
    pub pending_token_published: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_input_sha256: Option<String>,
}
#[derive(Clone, Debug, Serialize)]
pub struct TurnReceipt {
    pub schema: &'static str,
    pub request_id: String,
    pub previous_receipt_sha256: Option<String>,
    pub session: SessionSnapshot,
    pub output_token_ids: Vec<u32>,
    pub selected_tokens_including_eos: usize,
    pub stop_reason: &'static str,
    pub prefill_wall_s: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_detail: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_detail: Option<serde_json::Value>,
    pub whole_wall_s_before_receipt: f64,
    pub generation_complete_unix_ns: String,
    pub product_ready: bool,
    pub dflash_enabled: bool,
    pub vision_enabled: bool,
}
#[derive(Clone, Debug, Serialize)]
pub struct CommittedTurn {
    pub receipt: TurnReceipt,
    pub receipt_sha256: String,
}
#[derive(Clone, Debug, Serialize)]
pub struct CloseReceipt {
    pub schema: &'static str,
    pub request_id: String,
    pub session_id: String,
    pub revision: u64,
    pub manifest_sha256: String,
    pub last_turn_receipt_sha256: String,
    pub reset_completed: bool,
}
/// Discards an unpublished, typed cooperative-cancellation turn. This is not a
/// commit or rollback to old history: only a fresh session may follow it.
#[cfg(any(feature = "qwen35-augur-http", feature = "qwen35-nerkyor-cancel-recovery"))]
#[derive(Clone, Debug, Serialize)]
pub struct CancellationDiscardReceipt {
    pub schema: &'static str,
    pub request_id: String,
    pub cancelled_request_id: String,
    pub session_id: String,
    pub manifest_sha256: String,
    pub cancellation_boundary: &'static str,
    pub submitted_tokens_before_discard: usize,
    pub submitted_tokens_after_discard: usize,
    pub drain_completed: bool,
    pub reset_completed: bool,
    pub session_discarded: bool,
    pub previous_history_reusable: bool,
    pub product_ready: bool,
    pub dflash_enabled: bool,
}
#[cfg(any(feature = "qwen35-augur-http", feature = "qwen35-nerkyor-cancel-recovery"))]
#[derive(Clone, Debug, Serialize)]
pub struct CommittedCancellationDiscard {
    pub receipt: CancellationDiscardReceipt,
    pub receipt_sha256: String,
}
#[derive(Clone, Debug, Serialize)]
pub struct TokenEvent {
    pub request_id: String,
    pub session_id: String,
    pub sequence: usize,
    pub token_id: u32,
}

/// A terminal may be emitted only AFTER this returns. Partial/colliding receipts
/// are never overwritten or treated as successful commits. This is a durable
/// turn ledger, not a GPU snapshot or an implemented restore API.
pub fn receipt_handle_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| matches!(std::env::var("AUGUR_HTTP_FINISH").as_deref(), Ok("receipt_handle_v2" | "receipt_ref_v3")))
}

pub fn receipt_reference_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("AUGUR_HTTP_FINISH").as_deref() == Ok("receipt_ref_v3"))
}

pub fn reserve_receipt(root: &Path, request_id: &str) -> Result<std::fs::File> {
    if !valid_id(request_id) {
        return Err("invalid receipt request id".into());
    }
    Ok(OpenOptions::new().write(true).create_new(true)
        .open(root.join(format!("{request_id}.json")))?)
}

/// Same create-new and sync-before-terminal contract, with an early bound file.
/// An unused or failed reservation stays as evidence and can never be reused.
pub fn persist_reserved_receipt<T: Serialize>(file: &mut std::fs::File, value: &T) -> Result<String> {
    use std::io::Seek;
    if file.metadata()?.len() != 0 || file.stream_position()? != 0 {
        return Err("receipt reservation already written".into());
    }
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    file.write_all(&bytes)?;
    file.sync_all()?;
    Ok(digest(&bytes))
}

pub fn persist_receipt<T: Serialize>(root: &Path, request_id: &str, value: &T) -> Result<String> {
    if !valid_id(request_id) {
        return Err("invalid receipt request id".into());
    }
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(root.join(format!("{request_id}.json")))?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    Ok(digest(&bytes))
}

#[cfg(any(test, feature = "qwen35-augur-session"))]
pub(crate) struct SpeculativeStep {pub tokens:Vec<u32>,pub consumed:usize}
#[cfg(any(test, feature = "qwen35-augur-session"))]
pub(crate) trait SessionTarget {
    fn profile(&self)->&'static str {PROFILE}
    // Per-owner hooks: default QUASAR behavior is unchanged. Other model
    // adapters must supply their own bounded capacity and receipt namespace.
    fn context_capacity(&self)->Result<usize> {context_capacity()}
    fn context_output_limit(&self)->Result<usize> {
        self.context_capacity()?.checked_mul(95).map(|n|n/100).ok_or_else(||"session output capacity overflow".into())
    }
    fn turn_receipt_schema(&self)->&'static str {"quasar-token-turn-receipt-v1"}
    fn close_receipt_schema(&self)->&'static str {"quasar-session-close-v1"}
    fn cancellation_receipt_schema(&self)->&'static str {"quasar-session-cancellation-discard-v1"}
    fn dflash_enabled(&self)->bool {false}
    fn validate_selector_policy(&self,policy:Option<selector::Policy>,suffix:&[u32],images:bool)->Result<()> {
        selector::validate_request(policy,false,false,suffix,images)
    }
    fn begin_decode(&mut self,_policy:Option<selector::Policy>)->Result<()> {Ok(())}
    /// The anchor has already been published but is not yet consumed.
    /// All returned state is committed internally; the Session alone publishes.
    fn speculative_step(&mut self,_anchor:u32,_remaining:usize,_eos:&[u32],_cancel:&SessionCancellationToken)->Result<Option<SpeculativeStep>> {Ok(None)}

    #[cfg(all(windows,feature="qwen35-augur-http"))]
    fn checkpoint_bindings(&self)->Option<checkpoint::Bindings>{None}
    #[cfg(all(windows,feature="qwen35-augur-http"))]
    fn save_checkpoint(&mut self,_:&checkpoint::Descriptor,_:&str)->Result<serde_json::Value>{Err("checkpoint storage disabled".into())}
    #[cfg(all(windows,feature="qwen35-augur-http"))]
    fn restore_checkpoint(&mut self,_:&checkpoint::RestoreRequest)->Result<(checkpoint::Descriptor,serde_json::Value)>{Err("checkpoint storage disabled".into())}
    fn consumed_tokens(&self) -> usize;
    fn consume_sequence(&mut self, tokens: &[u32]) -> Result<()>;
    fn prefill(
        &mut self,
        tokens: &[u32],
        cancel: &SessionCancellationToken,
    ) -> Result<Option<serde_json::Value>> {
        for chunk in tokens.chunks(PREFILL_CHUNK) {
            cancel.check("before Augur prefill chunk")?;
            self.consume_sequence(chunk)?;
        }
        Ok(None)
    }
    /// Append text to an authenticated image session without re-encoding images.
    /// Owners must retain any multimodal decode restrictions on this path.
    fn prefill_image_history(&mut self, tokens: &[u32], cancel: &SessionCancellationToken)
        -> Result<Option<serde_json::Value>> { self.prefill(tokens, cancel) }
    fn decode_detail(&self) -> Option<serde_json::Value> { None }
    fn supports_images(&self) -> bool { false }
    fn prefill_images(&mut self, _tokens: &[u32], _images: &[augur_model_qwen35::Qwen35PreparedF16Image],
        _cancel: &SessionCancellationToken) -> Result<Option<serde_json::Value>> {
        Err("Vision not compiled for this owner".into())
    }
    fn greedy_next_token(&mut self) -> Result<u32>;
    fn consume_selected(&mut self) -> Result<()>;
    fn drain(&mut self) -> Result<()>;
    fn reset_empty(&mut self) -> Result<()>;
}
#[cfg(any(test, feature = "qwen35-augur-session"))]
#[derive(Clone)]
struct Active {
    snapshot: SessionSnapshot,
    receipt_sha256: String,
}
#[cfg(any(feature = "qwen35-augur-http", feature = "qwen35-nerkyor-cancel-recovery"))]
struct CancelledTurn {
    request_id: String,
    session_id: String,
    boundary: &'static str,
}
#[cfg(any(test, feature = "qwen35-augur-session"))]
pub(crate) struct Session<T> {
    target: T,
    manifest_sha256: String,
    eos: Vec<u32>,
    active: Option<Active>,
    poisoned: bool,
    requests: HashSet<String>,
    sessions: HashSet<String>,
    #[cfg(any(feature = "qwen35-augur-http", feature = "qwen35-nerkyor-cancel-recovery"))]
    cancelled_turn: Option<CancelledTurn>,
}
#[cfg(any(test, feature = "qwen35-augur-session"))]
impl<T: SessionTarget> Session<T> {
    #[cfg(test)]
    pub(crate) fn target_for_test(&mut self)->&mut T {&mut self.target}

    pub(crate) fn new(target: T, manifest_sha256: String, eos: Vec<u32>) -> Result<Self> {
        if !valid_digest(&manifest_sha256)
            || eos.is_empty()
            || !valid_tokens(&eos)
            || eos.iter().collect::<HashSet<_>>().len() != eos.len()
            || target.consumed_tokens() != 0
        {
            return Err("invalid Augur fresh owner identity/EOS/cursor".into());
        }
        Ok(Self {
            target,
            manifest_sha256,
            eos,
            active: None,
            poisoned: false,
            requests: HashSet::new(),
            sessions: HashSet::new(),
            #[cfg(any(feature = "qwen35-augur-http", feature = "qwen35-nerkyor-cancel-recovery"))]
            cancelled_turn: None,
        })
    }
    pub(crate) fn snapshot(&self) -> Option<&SessionSnapshot> {
        if self.poisoned {
            None
        } else {
            self.active.as_ref().map(|a| &a.snapshot)
        }
    }
    pub(crate) fn poisoned(&self) -> bool {
        self.poisoned
    }
    pub(crate) fn submitted_tokens(&self) -> usize {
        self.target.consumed_tokens()
    }
    pub(crate) fn retire(&mut self) -> Result<()> {
        self.poisoned = true;
        self.active = None;
        #[cfg(any(feature = "qwen35-augur-http", feature = "qwen35-nerkyor-cancel-recovery"))]
        { self.cancelled_turn = None; }
        self.target.drain()
    }
    /// Opt-in recovery authority, deliberately separate from a cancellation
    /// flag or an ordinary reset. Unknown errors, failed drains, foreign IDs,
    /// reused IDs, reset/cursor errors and failed persistence never qualify.
    /// Transport still must rotate to a fresh per-turn cancellation token.
    #[cfg(any(feature = "qwen35-augur-http", feature = "qwen35-nerkyor-cancel-recovery"))]
    pub(crate) fn discard_cancelled_turn(
        &mut self,
        request_id: &str,
        cancelled_request_id: &str,
        session_id: &str,
        persist: impl FnOnce(&CancellationDiscardReceipt) -> Result<String>,
    ) -> Result<CommittedCancellationDiscard> {
        let cancelled = self.cancelled_turn.as_ref().ok_or("no recoverable typed Augur cancellation")?;
        if !self.poisoned || self.active.is_some() || !valid_id(request_id)
            || self.requests.contains(request_id) || cancelled.request_id != cancelled_request_id
            || cancelled.session_id != session_id {
            return Err("Augur cancellation discard identity/state mismatch".into());
        }
        let boundary = cancelled.boundary;
        self.requests.insert(request_id.into());
        // Recovery is one-shot. Any failure below leaves this owner retired and
        // cannot be retried with a new discard ID to conceal a failed receipt.
        self.cancelled_turn = None;
        let before = self.target.consumed_tokens();
        let result = (|| -> Result<CommittedCancellationDiscard> {
            self.target.drain()?;
            self.target.reset_empty()?;
            self.target.drain()?;
            self.cursor(0)?;
            let receipt = CancellationDiscardReceipt {
                schema: self.target.cancellation_receipt_schema(),
                request_id: request_id.into(), cancelled_request_id: cancelled_request_id.into(),
                session_id: session_id.into(), manifest_sha256: self.manifest_sha256.clone(),
                cancellation_boundary: boundary, submitted_tokens_before_discard: before,
                submitted_tokens_after_discard: 0, drain_completed: true, reset_completed: true,
                session_discarded: true, previous_history_reusable: false,
                product_ready: false, dflash_enabled: self.target.dflash_enabled(),
            };
            let receipt_sha256 = persist(&receipt)?;
            if !valid_digest(&receipt_sha256) {
                return Err("invalid cancellation discard receipt digest".into());
            }
            Ok(CommittedCancellationDiscard { receipt, receipt_sha256 })
        })();
        match result {
            Ok(receipt) => { self.poisoned = false; Ok(receipt) }
            Err(error) => {
                if let Err(drain) = self.retire() {
                    return Err(format!("{error}; Augur cancellation discard drain failed: {drain}").into());
                }
                Err(error)
            }
        }
    }
    fn cursor(&mut self, expected: usize) -> Result<()> {
        if self.target.consumed_tokens() != expected {
            self.retire()?;
            return Err("Augur exact owner/ledger cursor mismatch".into());
        }
        Ok(())
    }
    /// Preflight is CPU-only and leaves the owner untouched on malformed input.
    /// Once execution starts, ANY failure poisons the owner; no uncommitted
    /// state, partially published turn, stale logits or pending token is reused.
    /// HTTP may explicitly discard a typed cancellation only after a separate
    /// successful drain/reset/durable receipt. Other errors require retirement.
    pub(crate) fn turn(
        &mut self,
        request: &TurnRequest,
        cancel: &SessionCancellationToken,
        emit: impl FnMut(&TokenEvent) -> Result<()>,
        persist: impl FnOnce(&TurnReceipt) -> Result<String>,
    ) -> Result<CommittedTurn> {
        self.turn_with_admission(request, cancel, || Ok(()), emit, persist)
    }
    /// Prepare the receipt only after all request validation, before GPU/state mutation.
    pub(crate) fn turn_with_admission(
        &mut self,
        request: &TurnRequest,
        cancel: &SessionCancellationToken,
        admit: impl FnOnce() -> Result<()>,
        mut emit: impl FnMut(&TokenEvent) -> Result<()>,
        persist: impl FnOnce(&TurnReceipt) -> Result<String>,
    ) -> Result<CommittedTurn> {
        if self.poisoned {
            return Err("Augur owner retired; create a new owner".into());
        }
        if !valid_id(&request.request_id)
            || !valid_id(&request.session_id)
            || self.requests.contains(&request.request_id)
            || !(1..=self.target.context_output_limit()?).contains(&request.max_new_tokens)
        {
            return Err("invalid or reused request identity/output budget".into());
        }
        let mut prepared_images = None;
        let mut image_identity = None;
        let (mut exact, prefix, revision, previous) = match &request.input {
            TurnInput::Open { prompt_token_ids } | TurnInput::OpenImages { prompt_token_ids, .. } => {
                if self.active.is_some()
                    || self.sessions.contains(&request.session_id)
                    || prompt_token_ids.is_empty()
                    || !valid_tokens(prompt_token_ids)
                {
                    return Err("invalid fresh Augur session/prompt".into());
                }
                if let TurnInput::OpenImages { images, .. } = &request.input {
                    if !self.target.supports_images() {return Err("Vision unavailable".into());}
                    let prepared = images::prepare(prompt_token_ids, images)?;
                    image_identity=Some(prepared.identity_sha256.clone());
                    prepared_images=Some(prepared);
                } else if prompt_token_ids.iter().any(|id| matches!(id, 248053..=248057)) {
                    return Err("Vision markers require authenticated images".into());
                }
                (Vec::new(), prompt_token_ids.clone(), 1, None)
            }
            TurnInput::Append {
                revision,
                exact_history,
                suffix_token_ids,
            } | TurnInput::AppendImages { revision, exact_history, suffix_token_ids, .. } => {
                let a = self.active.as_ref().ok_or("no retained Augur session")?;
                if a.snapshot.session_id != request.session_id
                    || a.snapshot.revision != *revision
                    || &a.snapshot.consumed_token_ids != exact_history
                    || suffix_token_ids.is_empty()
                    || !valid_tokens(suffix_token_ids)
                {
                    return Err("append identity/revision/exact history/suffix mismatch".into());
                }
                image_identity=a.snapshot.image_input_sha256.clone();
                let supplied=match &request.input {
                    TurnInput::AppendImages {image_input_sha256,..} => Some(image_input_sha256),_=>None };
                if supplied!=image_identity.as_ref() || suffix_token_ids.iter().any(|id| matches!(id,248053..=248057)) {
                    return Err("append image identity mismatch or new image needs full-history open".into());
                }
                let next_revision = revision.checked_add(1).ok_or("session revision overflow")?;
                let mut prefix = Vec::with_capacity(suffix_token_ids.len() + 1);
                prefix.push(a.snapshot.pending_token);
                prefix.extend_from_slice(suffix_token_ids);
                (
                    exact_history.clone(),
                    prefix,
                    next_revision,
                    Some(a.receipt_sha256.clone()),
                )
            }
        };
        let capacity = self.target.context_capacity()?;
        if !exact.len().checked_add(prefix.len())
            .and_then(|n| n.checked_add(request.max_new_tokens))
            .is_some_and(|n| n <= capacity)
        {
            return Err("selected context budget exceeded; no history truncation".into());
        }
        cancel.check("before Augur turn admission")?;
        self.cursor(exact.len())?;
        admit()?;
        self.requests.insert(request.request_id.clone());
        self.sessions.insert(request.session_id.clone());
        self.target.validate_selector_policy(request.selector_policy,&prefix,image_identity.is_some())?;
        self.active = None;
        self.poisoned = true; // cleared only after a successful durable receipt
        let mut persistence_started = false;
        let result = (|| -> Result<CommittedTurn> {
            let started = Instant::now();
            let prefill_detail = if let Some(images)=&prepared_images {
                self.target.prefill_images(&prefix,&images.images,cancel)?
            } else if image_identity.is_some() {
                self.target.prefill_image_history(&prefix,cancel)?
            } else {self.target.prefill(&prefix,cancel)?};
            exact.extend_from_slice(&prefix);
            self.cursor(exact.len())?;
            self.target.begin_decode(request.selector_policy)?;
            let prefill_wall_s = started.elapsed().as_secs_f64();
            let mut output = Vec::new();
            let (pending, published, stop) = loop {
                cancel.check("before QUASAR target selection")?;
                let id = self.target.greedy_next_token()?; // also drains prior decode
                if id >= VOCAB {
                    return Err("invalid/nonfinite target token".into());
                }
                cancel.check("before Augur token publication")?;
                if self.eos.contains(&id) {
                    break (id, false, "eos");
                }
                emit(&TokenEvent {
                    request_id: request.request_id.clone(),
                    session_id: request.session_id.clone(),
                    sequence: output.len(),
                    token_id: id,
                })?;
                output.push(id);
                if output.len() == request.max_new_tokens {
                    break (id, true, "length");
                }
                cancel.check("before Augur selected-token consume")?;
                let remaining=request.max_new_tokens-output.len()+1;
                if let Some(step)=self.target.speculative_step(id,remaining,&self.eos,cancel)? {
                    let count=step.tokens.len();
                    if count==0 || count>8 || count>remaining || step.tokens[0]!=id
                        || step.tokens.iter().any(|id|*id>=VOCAB || self.eos.contains(id))
                        || step.consumed!=count-usize::from(count==remaining) {
                        return Err("invalid speculative prefix/publication boundary".into());
                    }
                    exact.extend_from_slice(&step.tokens[..step.consumed]);
                    self.cursor(exact.len())?;
                    for &token_id in &step.tokens[1..] {
                        cancel.check("before Augur token publication")?;
                        emit(&TokenEvent {request_id:request.request_id.clone(),session_id:request.session_id.clone(),
                            sequence:output.len(),token_id})?;
                        output.push(token_id);
                    }
                    if output.len()==request.max_new_tokens {break (*output.last().unwrap(),true,"length");}
                    continue;
                }
                self.target.consume_selected()?;
                exact.push(id);
                self.cursor(exact.len())?;
            };
            self.target.drain()?;
            let generation_complete_unix_ns = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?.as_nanos().to_string();
            self.cursor(exact.len())?;
            cancel.check("before Augur durable receipt")?;
            let snapshot = SessionSnapshot {
                profile: self.target.profile().into(),
                manifest_sha256: self.manifest_sha256.clone(),
                session_id: request.session_id.clone(),
                revision,
                consumed_input_sha256: token_digest(&exact),
                consumed_token_ids: exact,
                pending_token: pending,
                pending_token_published: published,
                image_input_sha256: image_identity.clone(),
            };
            let receipt = TurnReceipt {
                schema: self.target.turn_receipt_schema(),
                request_id: request.request_id.clone(),
                previous_receipt_sha256: previous,
                session: snapshot,
                selected_tokens_including_eos: output.len() + usize::from(!published),
                output_token_ids: output,
                stop_reason: stop,
                prefill_wall_s,
                prefill_detail,
                decode_detail: self.target.decode_detail(),
                whole_wall_s_before_receipt: started.elapsed().as_secs_f64(),
                generation_complete_unix_ns,
                product_ready: false,
                dflash_enabled: self.target.dflash_enabled(),
                vision_enabled: image_identity.is_some(),
            };
            persistence_started = true;
            let receipt_sha256 = persist(&receipt)?;
            if !valid_digest(&receipt_sha256) {
                return Err("invalid persisted receipt digest".into());
            }
            // Persistence is the commit point. A later transport failure must
            // retire the owner, not roll back or reuse a published revision.
            Ok(CommittedTurn {
                receipt,
                receipt_sha256,
            })
        })();
        match result {
            Ok(turn) => {
                self.active = Some(Active {
                    snapshot: turn.receipt.session.clone(),
                    receipt_sha256: turn.receipt_sha256.clone(),
                });
                self.poisoned = false;
                Ok(turn)
            }
            Err(error) => {
                if let Err(drain) = self.retire() {
                    return Err(format!("{error}; Augur retirement drain failed: {drain}").into());
                }
                #[cfg(any(feature = "qwen35-augur-http", feature = "qwen35-nerkyor-cancel-recovery"))]
                if !persistence_started && cancel.is_requested() {
                    if let Some(cause) = error.downcast_ref::<crate::session_cancellation::SessionCancellationRequested>() {
                        self.cancelled_turn = Some(CancelledTurn {
                            request_id: request.request_id.clone(), session_id: request.session_id.clone(),
                            boundary: cause.boundary,
                        });
                    }
                }
                Err(error)
            }
        }
    }
    pub(crate) fn close(
        &mut self,
        request_id: &str,
        session_id: &str,
        revision: u64,
        persist: impl FnOnce(&CloseReceipt) -> Result<String>,
    ) -> Result<String> {
        if self.poisoned {
            return Err("Augur owner retired".into());
        }
        let a = self.active.as_ref().ok_or("no Augur session to close")?;
        if !valid_id(request_id)
            || self.requests.contains(request_id)
            || a.snapshot.session_id != session_id
            || a.snapshot.revision != revision
        {
            return Err("invalid close identity/revision".into());
        }
        let receipt = CloseReceipt {
            schema: self.target.close_receipt_schema(),
            request_id: request_id.into(),
            session_id: session_id.into(),
            revision,
            manifest_sha256: self.manifest_sha256.clone(),
            last_turn_receipt_sha256: a.receipt_sha256.clone(),
            reset_completed: true,
        };
        let expected = a.snapshot.consumed_token_ids.len();
        self.cursor(expected)?;
        self.requests.insert(request_id.into());
        self.poisoned = true;
        self.active = None;
        let result = (|| {
            self.target.reset_empty()?;
            self.cursor(0)?;
            let digest = persist(&receipt)?;
            if !valid_digest(&digest) {
                return Err("invalid close receipt digest".into());
            }
            Ok(digest)
        })();
        match result {
            Ok(hash) => {
                self.poisoned = false;
                Ok(hash)
            }
            Err(error) => {
                if let Err(drain) = self.retire() {
                    return Err(
                        format!("{error}; Augur close retirement drain failed: {drain}").into(),
                    );
                }
                Err(error)
            }
        }
    }
}

#[cfg(test)]
mod tests;

pub mod manifest;

#[cfg(any(test, feature = "qwen35-augur-session"))]
pub mod checkpoint;

#[cfg(test)]
mod model_contract_tests;
