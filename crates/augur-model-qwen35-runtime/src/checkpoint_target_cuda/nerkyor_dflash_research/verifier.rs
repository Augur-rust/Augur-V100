//! Sealed B8 greedy verifier; no sampled or product publication authority.
use super::*;
use crate::checkpoint_target_cuda::nerkyor_dflash_research::batched::Batch;
#[derive(Debug)]
pub struct NerkyorBatchedRound {
    pub output_tokens: Vec<u32>,
    pub matched_draft_tokens: usize,
    pub rejection_offset: Option<usize>,
    pub stop: &'static str,
    pub start_consumed: usize,
    pub end_consumed: usize,
    pub execution_identity_sha256: [u8; 32],
    pub used_batched_target: bool,
}
struct Decision {
    output: Vec<u32>,
    matched: usize,
    rejection: Option<usize>,
    stop: &'static str,
}
fn decision(
    anchor: u32,
    path: &[u32],
    next: &[u32],
    budget: usize,
    eos: &[u32],
) -> Result<Decision> {
    if path.len() != B
        || next.len() != B
        || budget == 0
        || eos.is_empty()
        || path[0] != anchor
        || std::iter::once(&anchor)
            .chain(path)
            .chain(next)
            .chain(eos)
            .any(|&v| v as usize >= V)
    {
        return Err("invalid B8 acceptance policy or token rows".into());
    }
    let mut output = Vec::with_capacity(B.min(budget));
    let mut matched = 0;
    let mut rejection = None;
    for offset in 0..B {
        let correct = if offset == 0 {
            anchor
        } else {
            next[offset - 1]
        };
        let proposed = path[offset];
        if offset > 0 {
            if correct == proposed {
                matched += 1;
            } else {
                rejection = Some(offset);
            }
        }
        output.push(correct);
        if let Some(stop) = stop_after(correct, proposed, offset, budget, eos) {
            return Ok(Decision {
                output,
                matched,
                rejection,
                stop,
            });
        }
    }
    Err("B8 decision did not terminate".into())
}
impl NerkyorDflashResearchDecoder {
    /// Retain the same target's B8 execution plan. Call once after prefill, before
    /// proposing. Repeated rounds reuse all allocations; release before reset.
    pub fn prepare_batched_verifier(&mut self, target: &mut CheckpointCudaTarget) -> Result<()> {
        self.validate_target(target)?;
        if !self.binding.policy.is_b8() || self.live.is_some() {
            return Err("B8 preparation requires its sealed identity and no live proposal".into());
        }
        if let Some(b) = &self.batched {
            if !b.quiescent() {
                return Err("nonquiescent B8 preparation".into());
            }
            return Ok(());
        }
        self.batched = Some(Batch::new(target)?);
        Ok(())
    }
    pub fn release_batched_verifier(&mut self) -> Result<()> {
        if self.poisoned
            || self.live.is_some()
            || self.batched.as_ref().is_some_and(|b| !b.quiescent())
        {
            return Err(
                "B8 release requires a healthy owner without a live proposal or transaction".into(),
            );
        }
        self.stream.synchronize()?;
        drop(self.batched.take());
        Ok(())
    }
    /// Output includes the canonical anchor, accepted candidates and, on first
    /// rejection, the correct fallback. The final returned token remains selected
    /// but unconsumed, exactly like the serial interface. No rejected tail is
    /// published in capture/KV/GDN state. Scalar and B8 numerical identities are
    /// deliberately noninterchangeable.
    pub fn verify_batched(
        &mut self,
        target: &mut CheckpointCudaTarget,
        proposal: &NerkyorResearchProposal,
        budget: usize,
        eos: &[u32],
    ) -> Result<NerkyorBatchedRound> {
        self.validate_target(target)?;
        self.validate_proposal(proposal)?;
        if !self.binding.policy.is_b8()
            || self.batched.as_ref().is_none_or(|b| !b.quiescent())
            || budget == 0
            || eos.is_empty()
            || eos.iter().any(|&id| id as usize >= V)
            || proposal.path.len() != B
            || proposal.path.iter().any(|&id| id as usize >= V)
            || target.selected != proposal.path.first().copied()
            || target.consumed != proposal.committed.2
            || target
                .dflash_capture
                .as_ref()
                .ok_or("missing B8 capture")?
                .committed_identity()?
                != proposal.committed
        {
            return Err(
                "unprepared, foreign, stale B8 proposal or invalid EOS/budget/anchor".into(),
            );
        }
        // Every caller-controlled check above precedes retirement or mutation.
        self.poisoned = true;
        self.live = None;
        let result = (|| -> Result<NerkyorBatchedRound> {
            let start_consumed = target.consumed;
            let anchor = proposal.path[0];
            if let Some(stop) = stop_after(anchor, anchor, 0, budget, eos) {
                self.stream.synchronize()?;
                return Ok(NerkyorBatchedRound {
                    output_tokens: vec![anchor],
                    matched_draft_tokens: 0,
                    rejection_offset: None,
                    stop,
                    start_consumed,
                    end_consumed: start_consumed,
                    execution_identity_sha256: self.binding.source_sha256,
                    used_batched_target: false,
                });
            }
            let b = self.batched.as_mut().ok_or("missing B8 verifier")?;
            b.execute(target, &proposal.path)?;
            let next = b.selected_rows()?;
            let d = decision(anchor, &proposal.path, &next, budget, eos)?;
            let committed = d.output.len() - 1;
            b.commit(target, committed)?;
            // Uses the committed logit row, and installs the device token slot
            // used by consume_selected. Never cache only a host-side token ID.
            if target.select_next_token()? != *d.output.last().ok_or("empty B8 output")? {
                return Err("B8 pending token disagrees with committed target logits".into());
            }
            Ok(NerkyorBatchedRound {
                output_tokens: d.output,
                matched_draft_tokens: d.matched,
                rejection_offset: d.rejection,
                stop: d.stop,
                start_consumed,
                end_consumed: target.consumed,
                execution_identity_sha256: self.binding.source_sha256,
                used_batched_target: true,
            })
        })();
        if result.is_ok() {
            self.poisoned = false;
        } else {
            target.poisoned = true;
        }
        result
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    const PATH: [u32; 8] = [10, 11, 12, 13, 14, 15, 16, 17];
    const NEXT: [u32; 8] = [11, 12, 13, 14, 15, 16, 17, 18];
    #[test]
    fn all_budgets_and_full_match_leave_last_pending() {
        for budget in 1..=8 {
            let d = decision(10, &PATH, &NEXT, budget, &[99]).unwrap();
            assert_eq!(d.output, PATH[..budget]);
            assert_eq!(d.matched, budget - 1);
            assert_eq!(d.stop, "length_pending");
            assert_eq!(d.rejection, None);
        }
        let d = decision(10, &PATH, &NEXT, 9, &[99]).unwrap();
        assert_eq!(d.output, PATH);
        assert_eq!(d.stop, "block_complete_pending");
    }
    #[test]
    fn every_rejection_returns_correct_pending_fallback_not_draft() {
        for offset in 1..8 {
            let mut path = PATH;
            path[offset] = 88;
            let d = decision(10, &path, &NEXT, 99, &[99]).unwrap();
            assert_eq!(d.output, PATH[..=offset]);
            assert_eq!(d.matched, offset - 1);
            assert_eq!(d.rejection, Some(offset));
            assert_eq!(d.stop, "rejection_fallback_pending");
        }
    }
    #[test]
    fn eos_budget_and_rejection_priority_matches_serial() {
        for offset in 0..8 {
            let mut path = PATH;
            if offset > 0 {
                path[offset] = 88;
            }
            let d = decision(10, &path, &NEXT, offset + 1, &[PATH[offset]]).unwrap();
            assert_eq!(d.output, PATH[..=offset]);
            assert_eq!(d.stop, "eos_pending");
            let d = decision(10, &path, &NEXT, offset + 1, &[99]).unwrap();
            assert_eq!(d.stop, "length_pending");
        }
    }
    #[test]
    fn invalid_rows_anchor_budget_and_stop_ids_are_closed() {
        assert!(decision(10, &PATH, &NEXT, 0, &[99]).is_err());
        assert!(decision(10, &PATH, &NEXT, 8, &[]).is_err());
        assert!(decision(10, &PATH, &NEXT, 8, &[248320]).is_err());
        assert!(decision(11, &PATH, &NEXT, 8, &[99]).is_err());
        assert!(decision(10, &PATH[..7], &NEXT, 8, &[99]).is_err());
        assert!(decision(10, &PATH, &NEXT[..7], 8, &[99]).is_err());
        let mut n = NEXT;
        n[7] = u32::MAX;
        assert!(decision(10, &PATH, &n, 8, &[99]).is_err());
        let mut p = PATH;
        p[7] = 248320;
        assert!(decision(10, &p, &NEXT, 8, &[99]).is_err());
    }
}
