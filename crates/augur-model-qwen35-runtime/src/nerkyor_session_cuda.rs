//! Bounded, explicit Nerkyor token-session adapter. Not a public HTTP profile.
//! Own manifest, canonical F32 target/KV and own Q8 DFlash2; no MTP/vision/sampling.
use crate::{
    augur_session::{self as session, *},
    checkpoint_projection_cuda::CheckpointProjectionPolicy,
    checkpoint_target_cuda::{
        nerkyor_dflash_research::{NerkyorBatchedRound, NerkyorDflashResearchDecoder},
        CheckpointCudaTarget,
    },
    nerkyor_session_manifest::{AdmittedNerkyorManifest, DRAFT_FILE},
    session_cancellation::SessionCancellationToken,
    DFlashQ8Backbone, DFlashQ8CudaWeights,
};
use augur_model_qwen35::Qwen35CheckpointProfile;
use serde_json::{json, Value};

pub const PROFILE: &str = "nerkyor_f32kv512_core_b8prefill_dflash_greedy_session_research_v2";
pub const CAPACITY: usize = 512;
pub const NO_DFLASH_PROFILE: &str = "nerkyor_f32kv4096_qpn_reassoc_no_dflash_greedy_session_research_v1";
pub const NO_DFLASH_CAPACITY: usize = 4096;

// DFlash keeps its bounded profile; target-only execution has a separate pinned identity.
#[derive(Clone, Copy)]
enum Context {
    Bounded512,
    NoDflash4096,
    #[cfg(test)]
    Extended4096,
}
impl Context {
    fn capacity(self) -> usize {
        match self {
            Self::Bounded512 => CAPACITY,
            Self::NoDflash4096 => NO_DFLASH_CAPACITY,
            #[cfg(test)]
            Self::Extended4096 => 4096,
        }
    }
    fn no_dflash(self) -> bool { matches!(self, Self::NoDflash4096) }
    fn execution(self) -> &'static str {
        if self.no_dflash() { crate::nerkyor_session_manifest::NO_DFLASH_EXECUTION }
        else { crate::nerkyor_session_manifest::EXECUTION }
    }
    fn profile(self) -> &'static str {
        match self {
            Self::Bounded512 => PROFILE,
            Self::NoDflash4096 => NO_DFLASH_PROFILE,
            #[cfg(test)]
            Self::Extended4096 => "nerkyor_f32kv4096_core_b8prefill_dflash_greedy_session_research_v1",
        }
    }
}

struct Publication {
    tokens: Vec<u32>,
    consume_pending: bool,
}
fn publication(
    r: &NerkyorBatchedRound,
    anchor: u32,
    remaining: usize,
    eos: &[u32],
) -> Result<Publication> {
    let ids = &r.output_tokens;
    if ids.is_empty()
        || ids.len() > 8
        || ids.len() > remaining
        || ids[0] != anchor
        || ids.iter().any(|id| *id >= session::VOCAB)
        || r.end_consumed.checked_sub(r.start_consumed) != Some(ids.len() - 1)
        || ids[..ids.len() - 1].iter().any(|id| eos.contains(id))
    {
        return Err("Nerkyor pending-prefix publication contract".into());
    }
    let last_eos = eos.contains(ids.last().unwrap());
    match r.stop {
        "eos_pending" if last_eos && ids.len() > 1 => Ok(Publication {
            tokens: ids[..ids.len() - 1].to_vec(),
            consume_pending: false,
        }),
        "length_pending" if !last_eos && ids.len() == remaining => Ok(Publication {
            tokens: ids.clone(),
            consume_pending: false,
        }),
        "block_complete_pending" | "rejection_fallback_pending"
            if !last_eos && ids.len() < remaining =>
        {
            Ok(Publication {
                tokens: ids.clone(),
                consume_pending: true,
            })
        }
        _ => Err("Nerkyor stop/budget/EOS publication mismatch".into()),
    }
}
struct Target {
    context: Context,
    // Drop the decoder and its shared KV references BEFORE the target owner.
    decoder: Option<NerkyorDflashResearchDecoder>,
    target: CheckpointCudaTarget,
    weights: Option<DFlashQ8CudaWeights>,
    rounds: usize,
    enabled: bool,
    decode_started: Option<std::time::Instant>,
    decode_steps: usize,
    decode_wall_s: f64,
}
impl session::SessionTarget for Target {
    fn profile(&self) -> &'static str {
        self.context.profile()
    }
    fn context_capacity(&self) -> Result<usize> {
        Ok(self.context.capacity())
    }
    fn turn_receipt_schema(&self) -> &'static str {
        "nerkyor-token-turn-receipt-v1"
    }
    fn close_receipt_schema(&self) -> &'static str {
        "nerkyor-session-close-v1"
    }
    fn cancellation_receipt_schema(&self) -> &'static str {
        "nerkyor-session-cancellation-discard-v1"
    }
    fn dflash_enabled(&self) -> bool {
        self.enabled
    }
    // Default selector validation rejects both old QUASAR gate045 policies.
    fn consumed_tokens(&self) -> usize {
        self.target.consumed_tokens()
    }
    fn consume_sequence(&mut self, ids: &[u32]) -> Result<()> {
        self.target.consume_sequence(ids)
    }
    fn prefill(&mut self, ids: &[u32], cancel: &SessionCancellationToken) -> Result<Option<Value>> {
        if self.context.no_dflash() {
            for chunk in ids.chunks(8) {
                cancel.check("before Nerkyor no-DFlash prefill")?;
                self.target.consume_sequence(chunk)?;
            }
            self.target.drain_session_work()?;
            return Ok(Some(json!({"profile":self.context.profile(),"execution_profile":self.context.execution(),
                "prefill":"qpn_source_reassoc_scalar_v1","tokens":ids.len(),"scalar_tokens":ids.len(),
                "b8_full_blocks":0,"dflash_loaded":false,"product_ready":false})));
        }
        // The test's independent scalar oracle must not accidentally use the
        // optimized prefill it is supposed to verify. No public toggle exists.
        #[cfg(test)]
        if !self.enabled {
            for chunk in ids.chunks(8) {
                cancel.check("before independent scalar oracle prefill")?;
                self.target.consume_sequence(chunk)?;
            }
            return Ok(Some(json!({"profile":self.context.profile(),"prefill":"test_only_canonical_scalar_oracle",
                "tokens":ids.len(),"product_ready":false})));
        }
        let mut detail=self.decoder.as_mut().ok_or("missing Nerkyor draft decoder")?.prefill_core_b8(&mut self.target,ids,cancel)?;
        detail["profile"]=json!(self.context.profile());
        Ok(Some(detail))
    }
    fn begin_decode(&mut self, policy: Option<selector::Policy>) -> Result<()> {
        if policy.is_some() {
            return Err("Nerkyor does not admit QUASAR selector calibration".into());
        }
        self.rounds = 0;
        self.decode_started = None;
        self.decode_steps = 0;
        self.decode_wall_s = 0.0;
        if self.enabled && self.target.consumed_tokens() + 8 <= self.context.capacity() {
            let decoder = self.decoder.as_mut().ok_or("missing Nerkyor draft decoder")?;
            decoder.prepare_batched_verifier(&mut self.target)?;
            decoder.prepare_resident_proposal(&mut self.target,
                self.weights.as_ref().ok_or("missing Nerkyor draft weights")?)?;
        }
        Ok(())
    }
    fn greedy_next_token(&mut self) -> Result<u32> {
        let token = self.target.select_next_token()?;
        if let Some(started) = self.decode_started.take() {
            self.decode_wall_s += started.elapsed().as_secs_f64();
            self.decode_steps += 1;
        }
        Ok(token)
    }
    fn consume_selected(&mut self) -> Result<()> {
        if self.context.no_dflash() {
            self.decode_started = Some(std::time::Instant::now());
        }
        self.target.consume_selected()
    }
    fn drain(&mut self) -> Result<()> {
        if let Some(decoder) = self.decoder.as_mut() { decoder.drain_research_work()?; }
        self.target.drain_session_work()
    }
    fn reset_empty(&mut self) -> Result<()> {
        if let Some(decoder) = self.decoder.as_mut() { decoder.release_batched_verifier()?; }
        self.target.reset_empty()?;
        self.rounds = 0;
        Ok(())
    }
    fn speculative_step(
        &mut self,
        anchor: u32,
        remaining: usize,
        eos: &[u32],
        cancel: &SessionCancellationToken,
    ) -> Result<Option<SpeculativeStep>> {
        if !self.enabled || self.target.consumed_tokens() + 8 > self.context.capacity() {
            return Ok(None);
        }
        cancel.check("before Nerkyor draft proposal")?;
        let decoder = self.decoder.as_mut().ok_or("missing Nerkyor draft decoder")?;
        let proposal = decoder.propose_resident(&mut self.target,
            self.weights.as_ref().ok_or("missing Nerkyor draft weights")?)?;
        if proposal.path()[0] != anchor {
            decoder.discard(proposal)?;
            return Err("Nerkyor session anchor drift".into());
        }
        if cancel.is_requested() {
            decoder.discard(proposal)?;
            cancel.check("before Nerkyor B8 verify")?;
            return Err("Nerkyor cancellation unexpectedly cleared".into());
        }
        let before = self.target.consumed_tokens();
        let r = decoder.verify_batched(&mut self.target, &proposal, remaining, eos)?;
        if r.start_consumed != before || r.end_consumed != self.target.consumed_tokens() {
            return Err("Nerkyor session verifier cursor drift".into());
        }
        let p = publication(&r, anchor, remaining, eos)?;
        if p.consume_pending {
            cancel.check("before Nerkyor pending-tail consume")?;
            self.target.consume_selected()?;
        }
        self.rounds += 1;
        Ok(Some(SpeculativeStep {
            tokens: p.tokens,
            consumed: self.target.consumed_tokens() - before,
        }))
    }
    fn decode_detail(&self) -> Option<Value> {
        if self.context.no_dflash() {
            return Some(json!({"profile":self.context.profile(),"execution_profile":self.context.execution(),
                "dflash_enabled":false,"dflash_loaded":false,"rounds":0,"sampling":"greedy","kv_dtype":"F32",
                "capacity":self.context.capacity(),"selector":"target_greedy","product_ready":false,"mtp_executed":false,
                "decode_steps":self.decode_steps,"decode_wall_s":self.decode_wall_s,
                "decode_tokens_per_second":if self.decode_wall_s>0.0 {Some(self.decode_steps as f64/self.decode_wall_s)} else {None},
                "first_prefill_selection_excluded":true,
                "timing_scope":"sum_consume_selected_through_greedy_readback_excludes_stream_emit_and_receipt"}));
        }
        Some(
            json!({"profile":self.context.profile(),"execution_profile":crate::nerkyor_session_manifest::EXECUTION,
            "dflash_enabled":self.enabled,"rounds":self.rounds,"sampling":"greedy","kv_dtype":"F32",
            "capacity":self.context.capacity(),"selector":"own_q8_default_uncalibrated","product_ready":false,"mtp_executed":false}),
        )
    }
}
/// Exact-history/receipt session groundwork. Only a bounded research profile is
/// exposed here; existing QUASAR launchers/HTTP admission remain unchanged.
pub struct NerkyorResearchCudaSession {
    owner: session::Session<Target>,
    info: Value,
}
impl NerkyorResearchCudaSession {
    pub fn open(admitted: AdmittedNerkyorManifest) -> Result<Self> {
        let context = if admitted.manifest.execution_profile == crate::nerkyor_session_manifest::NO_DFLASH_EXECUTION {
            Context::NoDflash4096
        } else { Context::Bounded512 };
        Self::open_context(admitted, context)
    }
    fn open_context(admitted: AdmittedNerkyorManifest, context: Context) -> Result<Self> {
        if admitted.manifest.execution_profile != context.execution() {
            return Err("Nerkyor session execution differs from pinned manifest".into());
        }
        // Retained bytes, never a hash-then-reopen tokenizer path.
        let tokenizer = tokenizers::Tokenizer::from_bytes(admitted.tokenizer_bytes())
            .map_err(|e| e.to_string())?;
        for (text, id) in [
            ("<|im_end|>", 248046),
            ("<|endoftext|>", 248044),
            ("<|audio_start|>", 248070),
        ] {
            if tokenizer.token_to_id(text) != Some(id) {
                return Err("Nerkyor tokenizer special-token contract".into());
            }
        }
        let weights = if context.no_dflash() { None } else {
            let cpu = DFlashQ8Backbone::load_pinned(
                admitted.draft_root.join(DRAFT_FILE),
                admitted.draft_config.clone(),
            )?;
            if cpu.manifest().file_sha256.as_deref()
                != Some(admitted.manifest.draft_gguf_sha256.as_str())
            {
                return Err("Nerkyor own converted draft digest mismatch".into());
            }
            let weights = DFlashQ8CudaWeights::upload(&cpu, 0)?;
            drop(cpu);
            Some(weights)
        };
        let mut target = CheckpointCudaTarget::open_with_projection_policy(
            &admitted.root,
            Qwen35CheckpointProfile::NerkyorW4a16,
            admitted.manifest.checkpoint.clone(),
            context.capacity(),
            if context.no_dflash() { CheckpointProjectionPolicy::NerkyorQpnSourceReassocResearch }
            else { CheckpointProjectionPolicy::NerkyorBf16GateF32CtaResearch },
        )?;
        let decoder = if let Some(weights) = weights.as_ref() {
            target.enable_nerkyor_core_b8_prefill_research_capture(weights)?;
            Some(NerkyorDflashResearchDecoder::new(&mut target, weights)?)
        } else { None };
        let mut info = target.info().clone();
        info["session_profile"] = json!(context.profile());
        if context.no_dflash() {
            info["nerkyor_target_execution"] = json!(context.execution());
            info["dflash_loaded"] = json!(false);
            info["dflash_enabled"] = json!(false);
        }
        info["product_admitted"] = json!(false);
        info["session_manifest_sha256"] = json!(admitted.sha256);
        let owner = session::Session::new(
            Target {
                context,
                decoder,
                target,
                weights,
                rounds: 0,
                enabled: !context.no_dflash(),
                decode_started: None,
                decode_steps: 0,
                decode_wall_s: 0.0,
            },
            admitted.sha256,
            admitted.eos,
        )?;
        Ok(Self { owner, info })
    }
    pub fn info(&self) -> &Value {
        &self.info
    }
    pub fn snapshot(&self) -> Option<&SessionSnapshot> {
        self.owner.snapshot()
    }
    pub fn profile(&self) -> &str { self.info["session_profile"].as_str().unwrap() }
    pub fn capacity(&self) -> usize { self.info["kv_capacity"].as_u64().unwrap() as usize }
    pub fn dflash_enabled(&self) -> bool { self.profile() != NO_DFLASH_PROFILE }
    pub fn poisoned(&self) -> bool {
        self.owner.poisoned()
    }
    pub fn submitted_tokens(&self) -> usize {
        self.owner.submitted_tokens()
    }
    pub fn retire(&mut self) -> Result<()> {
        self.owner.retire()
    }
    pub fn turn(
        &mut self,
        r: &TurnRequest,
        cancel: &SessionCancellationToken,
        emit: impl FnMut(&TokenEvent) -> Result<()>,
        persist: impl FnOnce(&TurnReceipt) -> Result<String>,
    ) -> Result<CommittedTurn> {
        if r.selector_policy.is_some() {
            return Err("Nerkyor does not admit QUASAR selector calibration".into());
        }
        self.owner.turn(r, cancel, emit, persist)
    }
    pub fn close(
        &mut self,
        request_id: &str,
        session_id: &str,
        revision: u64,
        persist: impl FnOnce(&CloseReceipt) -> Result<String>,
    ) -> Result<String> {
        self.owner.close(request_id, session_id, revision, persist)
    }
    #[cfg(any(feature = "qwen35-augur-http", feature = "qwen35-nerkyor-cancel-recovery"))]
    pub fn discard_cancelled_turn(
        &mut self,
        request_id: &str,
        cancelled_request_id: &str,
        session_id: &str,
        persist: impl FnOnce(&CancellationDiscardReceipt) -> Result<String>,
    ) -> Result<CommittedCancellationDiscard> {
        self.owner
            .discard_cancelled_turn(request_id, cancelled_request_id, session_id, persist)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn context_profiles_do_not_silently_expand_default_admission() {
        assert_eq!(Context::Bounded512.capacity(), 512);
        assert_eq!(Context::Bounded512.profile(), PROFILE);
        assert_eq!(Context::Extended4096.capacity(), 4096);
        assert_ne!(Context::Extended4096.profile(), PROFILE);
        assert_eq!(Context::Extended4096.capacity(),
            augur_backend::QWEN35_FULL_ATTENTION_MAX_CAPACITY);
    }

    fn round(ids: Vec<u32>, stop: &'static str) -> NerkyorBatchedRound {
        NerkyorBatchedRound {
            start_consumed: 20,
            end_consumed: 20 + ids.len() - 1,
            output_tokens: ids,
            stop,
            matched_draft_tokens: 0,
            rejection_offset: None,
            execution_identity_sha256: [0; 32],
            used_batched_target: true,
        }
    }
    #[test]
    fn pending_tail_adapter_keeps_eos_unpublished_and_budget_pending() {
        let eos = [9];
        let p = publication(&round(vec![1, 2, 9], "eos_pending"), 1, 8, &eos).unwrap();
        assert_eq!(p.tokens, [1, 2]);
        assert!(!p.consume_pending);
        let p = publication(&round(vec![1, 2], "length_pending"), 1, 2, &eos).unwrap();
        assert_eq!(p.tokens, [1, 2]);
        assert!(!p.consume_pending);
        for stop in ["block_complete_pending", "rejection_fallback_pending"] {
            let p = publication(&round(vec![1, 2], stop), 1, 8, &eos).unwrap();
            assert_eq!(p.tokens, [1, 2]);
            assert!(p.consume_pending);
        }
        for (ids, stop, budget) in [
            (vec![1, 9], "length_pending", 2),
            (vec![1, 2], "length_pending", 8),
            (vec![1, 2], "block_complete_pending", 2),
            (vec![1, 9, 2], "block_complete_pending", 8),
            (vec![9], "eos_pending", 8),
        ] {
            let r = round(ids, stop);
            assert!(publication(&r, r.output_tokens[0], budget, &eos).is_err());
        }
        let mut bad = round(vec![1, 2], "rejection_fallback_pending");
        bad.end_consumed += 1;
        assert!(publication(&bad, 1, 8, &eos).is_err());
    }
}

#[cfg(test)]
mod gpu_tests;

#[cfg(all(test, feature = "qwen35-nerkyor-cancel-recovery"))]
mod recovery_gpu_tests;

#[cfg(all(test, feature = "qwen35-nerkyor-cancel-recovery"))]
mod context_gpu_tests;

#[cfg(test)]
mod no_dflash_gpu_tests;
