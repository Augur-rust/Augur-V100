//! Core-arithmetic B8 prefill under its own sealed identity. No QUASAR route.
use super::*;
use crate::session_cancellation::SessionCancellationToken;

fn partition(consumed: usize, count: usize) -> Result<(usize, usize, usize)> {
    if count == 0 { return Err("nonempty known-token prefill required".into()); }
    let initial=usize::from(consumed==0);
    let remaining=count-initial;
    Ok((initial,remaining/B,remaining%B))
}

impl NerkyorDflashResearchDecoder {
    /// Consume only caller-known prompt IDs. Reuses the same fully committed
    /// batch plan for later verification; no speculative token is selected here.
    pub fn prefill_core_b8(&mut self, target: &mut CheckpointCudaTarget, ids: &[u32],
                          cancel: &SessionCancellationToken) -> Result<serde_json::Value> {
        self.validate_target(target)?;
        if self.binding.policy != ResearchPolicy::CoreB8Prefill || self.live.is_some()
            || self.batched.as_ref().is_some_and(|b| !b.quiescent())
            || ids.is_empty() || ids.iter().any(|&t| t as usize >= V)
            || target.consumed.checked_add(ids.len()).is_none_or(|n| n>target.capacity)
            || target.selected.is_some_and(|id| ids.first().copied()!=Some(id)) {
            return Err("invalid/foreign core-B8 prefill state, token sequence or pending anchor".into());
        }
        let (initial,blocks,tail)=partition(target.consumed,ids.len())?;
        cancel.check("before core-B8 prefill mutation")?;
        let result=(|| -> Result<serde_json::Value> {
            let mut at=0;
            if initial==1 {
                target.consume_sequence(&ids[..1])?;
                at=1;
            }
            if blocks>0 {
                cancel.check("before core-B8 prefill plan")?;
                self.prepare_batched_verifier(target)?;
            }
            for _ in 0..blocks {
                cancel.check("before core-B8 known-token block")?;
                let batch=self.batched.as_mut().ok_or("missing core-B8 prefill plan")?;
                batch.execute(target,&ids[at..at+B])?;
                batch.commit(target,B)?;
                at+=B;
            }
            if tail>0 {
                cancel.check("before core-B8 scalar prefill tail")?;
                target.consume_sequence(&ids[at..])?;
            }
            Ok(serde_json::json!({"prefill":"scalar_initial1_full_b8_commit_scalar_tail_v1",
                "execution_profile":identity::CORE_B8_EXECUTION,"tokens":ids.len(),
                "scalar_tokens":initial+tail,"b8_full_blocks":blocks,"product_ready":false,
                "source_identity_sha256":format!("{:x}",Sha256::digest(serde_json::to_vec(
                    &target.info["dflash_feature_capture"]["source_identity_contract"])?))}))
        })();
        if result.is_err() && target.poisoned { self.poisoned=true; }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_partition_includes_first_scalar_and_every_residual_tail() {
        assert!(partition(0,0).is_err());assert!(partition(15,0).is_err());
        for count in 1..=4096 {
            for consumed in [0,1,16,256,512,2048,4088] {
                let (initial,blocks,tail)=partition(consumed,count).unwrap();
                assert_eq!(initial,usize::from(consumed==0));
                assert_eq!(initial+blocks*8+tail,count);assert!(tail<8);
            }
        }
        assert_eq!(partition(0,8).unwrap(),(1,0,7));
        assert_eq!(partition(0,9).unwrap(),(1,1,0));
        assert_eq!(partition(9,8).unwrap(),(0,1,0));
    }
}
