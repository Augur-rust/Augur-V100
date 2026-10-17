//! Explicit request-scoped selector experiment. It never relaxes target verification.
use serde::{Deserialize, Serialize};
use super::Result;

pub const POLICY: &str = "english_thinking_gate045_v1";
pub const GLOBAL_POLICY: &str = "global_gate045_v1";
pub const GATE_SHA256: &str = "0bcb694a4e84aa304d888829742fdef92941771ac5e97917ca5dc4388f6821c7";
pub const CLOSE_THINK: u32 = 248069;
pub const OPEN_THINK_SUFFIX: [u32; 2] = [248068, 198];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Policy {
    #[serde(rename = "english_thinking_gate045_v1")]
    EnglishThinkingGate045V1,
    #[serde(rename = "global_gate045_v1")]
    GlobalGate045V1,
}

pub(crate) fn admit(value: Option<&str>, gate_sha: Option<&str>, precision: &str, codebooks: bool) -> Result<bool> {
    match value {
        None if gate_sha.is_none() && !codebooks => Ok(false),
        Some(POLICY | GLOBAL_POLICY) if gate_sha == Some(GATE_SHA256) && precision == "f32_cublas_pedantic" && !codebooks => Ok(true),
        _ => Err("HTTP English selector requires explicit policy, pinned 0.45 F32 gate, FP32 accumulation and Q8 codebooks".into()),
    }
}

pub(crate) fn validate_request(policy: Option<Policy>, available: bool, global_available: bool, prompt_suffix: &[u32], images: bool) -> Result<()> {
    if let Some(policy)=policy {
        let global=policy==Policy::GlobalGate045V1;
        if !available || images || global!=global_available || (!global && !prompt_suffix.ends_with(&OPEN_THINK_SUFFIX)) {
            return Err("selector request requires matching startup scope, fixed B8, pinned F32 weights and text input".into());
        }
    }
    Ok(())
}

#[derive(Default)]
pub(crate) struct Phase {
    pub eligible: bool,
    pub global: bool,
    pub thinking: bool,
    pub calibrated_rounds: usize,
    pub calibrated_accepted: usize,
    pub calibrated_checked: usize,
    pub q8_rounds: usize,
    pub q8_accepted: usize,
    pub q8_checked: usize,
    pub boundary_rounds: usize,
    pub close_seen: bool,
}
impl Phase {
    pub fn gate_active(&self)->bool {self.global || self.thinking}
    pub fn policy_name(&self)->Option<&'static str> {self.eligible.then_some(if self.global {GLOBAL_POLICY} else {POLICY})}
    pub fn begin(&mut self, policy: Option<Policy>) {
        *self = Self { eligible: policy.is_some(), global: policy==Some(Policy::GlobalGate045V1), thinking: policy==Some(Policy::EnglishThinkingGate045V1), ..Self::default() };
    }
    /// Count only verified positions. A close predicted beyond the first
    /// rejection cannot change either the request phase or these counters.
    pub fn verified_round(&mut self, tokens: &[u32], rejected: bool) {
        debug_assert!(!tokens.is_empty());
        let accepted=tokens.len().saturating_sub(1);
        if self.global {
            self.calibrated_rounds+=1;self.calibrated_accepted+=accepted;
            self.calibrated_checked+=accepted+usize::from(rejected);
            self.close_seen|=tokens.contains(&CLOSE_THINK);return;
        }
        if self.thinking {
            self.calibrated_rounds+=1;
            let close=tokens.iter().position(|&t|t==CLOSE_THINK);
            let thinking_accepted=close.map_or(accepted,|i|i.min(accepted));
            self.calibrated_accepted+=thinking_accepted;
            self.calibrated_checked+=thinking_accepted+usize::from(rejected && close.is_none());
            self.q8_accepted+=accepted-thinking_accepted;
            self.q8_checked+=accepted-thinking_accepted+usize::from(rejected && close.is_some());
            if close.is_some() {self.boundary_rounds+=1;self.selected(CLOSE_THINK);}
        } else {
            self.q8_rounds+=1;self.q8_accepted+=accepted;self.q8_checked+=accepted+usize::from(rejected);
        }
    }
    pub fn selected(&mut self, token: u32) {
        if self.global {self.close_seen|=token==CLOSE_THINK;return;}
        if self.thinking && token == CLOSE_THINK { self.thinking = false; self.close_seen = true; }
    }
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn global_gate_requires_explicit_matching_startup_scope_and_remains_active_after_close() {
        let g=Some(Policy::GlobalGate045V1);
        assert!(admit(Some(GLOBAL_POLICY),Some(GATE_SHA256),"f32_cublas_pedantic",false).unwrap());
        assert!(validate_request(g,true,true,&[42],false).is_ok());
        assert!(validate_request(g,true,false,&OPEN_THINK_SUFFIX,false).is_err());
        assert!(validate_request(g,true,true,&[42],true).is_err());
        assert!(validate_request(Some(Policy::EnglishThinkingGate045V1),true,true,&OPEN_THINK_SUFFIX,false).is_err());
        let mut p=Phase::default();p.begin(g);assert!(p.gate_active());
        p.selected(CLOSE_THINK);assert!(p.gate_active());
        p.verified_round(&[42,CLOSE_THINK,43],true);p.verified_round(&[44,45],false);
        assert_eq!((p.calibrated_rounds,p.calibrated_accepted,p.calibrated_checked),(2,3,4));
        assert_eq!((p.q8_rounds,p.q8_accepted,p.q8_checked,p.boundary_rounds),(0,0,0,0));
        p.begin(None);assert!(!p.gate_active() && !p.global && !p.close_seen);
    }
    #[test] fn pinned_policy_cannot_silently_admit_a_foreign_or_high_precision_gate() {
        assert!(!admit(None,None,"q8_0",false).unwrap());
        assert!(admit(Some(POLICY),Some(GATE_SHA256),"f32_cublas_pedantic",false).unwrap());
        for (policy,sha,precision,books) in [
            (None,Some(GATE_SHA256),"f32_cublas_pedantic",false),
            (Some(POLICY),Some("wrong"),"f32_cublas_pedantic",false),
            (Some(POLICY),Some(GATE_SHA256),"f32_weights_f64_accum",false),
            (Some(POLICY),Some(GATE_SHA256),"f32_cublas_pedantic",true),
        ] { assert!(admit(policy,sha,precision,books).is_err()); }
    }
    #[test] fn phase_is_request_local_and_closing_think_restores_q8() {
        let mut phase=Phase::default();phase.begin(Some(Policy::EnglishThinkingGate045V1));
        phase.selected(17);assert!(phase.thinking);
        phase.selected(CLOSE_THINK);assert!(!phase.thinking && phase.close_seen);
        phase.selected(248068);assert!(!phase.thinking); // literal text cannot reopen the phase.
        phase.begin(None);assert!(!phase.thinking && !phase.eligible && !phase.close_seen);
        phase.begin(Some(Policy::EnglishThinkingGate045V1));assert!(phase.thinking);
    }
    #[test] fn explicit_request_requires_capability_and_canonical_text_thinking_suffix() {
        let policy=Some(Policy::EnglishThinkingGate045V1);
        assert!(validate_request(policy,true,false,&[42,248068,198],false).is_ok());
        for (available,suffix,images) in [(false,&[248068,198][..],false),(true,&[248068][..],false),(true,&[248068,198][..],true)] {
            assert!(validate_request(policy,available,false,suffix,images).is_err());
        }
        assert!(validate_request(None,false,false,&[1],true).is_ok());
        assert!(serde_json::from_str::<Policy>("\"anything\"").is_err());
    }
    #[test] fn verified_phase_ledger_splits_close_and_rejected_suffix() {
        for pos in 1..8 {
            let mut phase=Phase::default();phase.begin(Some(Policy::EnglishThinkingGate045V1));
            let mut tokens=vec![42;8];tokens[pos]=CLOSE_THINK;
            phase.verified_round(&tokens,true);
            assert_eq!((phase.calibrated_accepted,phase.calibrated_checked),(pos,pos));
            assert_eq!((phase.q8_accepted,phase.q8_checked),(7-pos,8-pos));
            assert_eq!(phase.boundary_rounds,1);assert!(!phase.thinking && phase.close_seen);
            phase.verified_round(&[248068,42],false);
            assert_eq!(phase.q8_rounds,1);assert!(!phase.thinking);
        }
        let mut p=Phase::default();p.begin(Some(Policy::EnglishThinkingGate045V1));
        p.verified_round(&[42,43],true); // rejected predicted close was not committed
        assert!(p.thinking);assert_eq!((p.calibrated_accepted,p.calibrated_checked),(1,2));
        assert_eq!(p.boundary_rounds,0);
        p.verified_round(&[CLOSE_THINK,44],false);
        assert_eq!((p.q8_accepted,p.q8_checked),(1,1));assert!(!p.thinking);
    }

}
