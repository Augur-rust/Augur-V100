//! Same-owner candidate production and explicitly sealed serial/B8 research
//! verifiers. The B8 verifier commits only its accepted prefix through KV/GDN
//! rollback; it is not a persisted product-session publication path. Every
//! returned final token stays selected but unconsumed.
use super::*;
use crate::checkpoint_projection_cuda::{
    CheckpointScalarCudaProjection, CheckpointSharedBf16HeadRows,
};
use augur_backend::{
    CudaF32View, CudaQwen35DflashQ8SelectorArgsV1, CudaQwen35DflashQ8SelectorViewsV1,
    CudaQwen35VocabTopkArgsV1, OwnedCudaEvent,
};
const V: usize = 248320;
const K: usize = 16;

pub struct NerkyorResearchProposal {
    instance: Rc<()>,
    generation: u64,
    committed: (u64, u64, usize),
    binding: ResearchBinding,
    path: Vec<u32>,
}
impl NerkyorResearchProposal {
    /// Diagnostic candidates, not accepted or published output.
    pub fn path(&self) -> &[u32] {
        &self.path
    }
}

pub struct NerkyorHeadAudit {
    pub normalized_hidden: Vec<f32>,
    pub optimized_logits: Vec<f32>,
    pub scalar_logits: Vec<f32>,
    pub optimized_kernel: &'static str,
    pub optimized_gpu_ms: f32,
    pub scalar_gpu_ms: f32,
    pub timing_iterations: usize,
}

#[derive(Debug)]
pub struct NerkyorSerialRound {
    /// Canonical output including the last, selected-but-not-consumed token.
    pub output_tokens: Vec<u32>,
    /// Matches after the separately selected target anchor; not an acceptance rate.
    pub matched_draft_tokens: usize,
    pub rejection_offset: Option<usize>,
    pub stop: &'static str,
    pub start_consumed: usize,
    pub end_consumed: usize,
}

struct Drain(Rc<OwnedCudaStream>);
impl Drop for Drain {
    fn drop(&mut self) {
        let _ = self.0.synchronize();
    }
}

/// One outstanding move-only proposal. No duplicate resident vocabulary weight;
/// both head schedules retain the target's exact original BF16 head owner.
pub struct NerkyorDflashResearchDecoder {
    stream: Rc<OwnedCudaStream>,
    instance: Rc<()>,
    binding: ResearchBinding,
    generation: u64,
    live: Option<u64>,
    poisoned: bool,
    normalized: Rc<OwnedCudaBuffer>,
    logits: Rc<OwnedCudaBuffer>,
    scalar_logits: Rc<OwnedCudaBuffer>,
    optimized: CheckpointSharedBf16HeadRows,
    scalar: Vec<CheckpointScalarCudaProjection>,
    topk: CudaQwen35VocabTopkArgsV1,
    ids: OwnedCudaBuffer,
    scores: OwnedCudaBuffer,
    transitions: OwnedCudaBuffer,
    path: OwnedCudaBuffer,
    #[cfg(feature = "qwen35-prefill-cublas")]
    batched: Option<super::batched::Batch>,
    #[cfg(feature = "qwen35-prefill-cublas")]
    resident: Option<resident::Resident>,
}

fn stop_after(
    correct: u32,
    proposed: u32,
    offset: usize,
    budget: usize,
    eos: &[u32],
) -> Option<&'static str> {
    if eos.contains(&correct) {
        Some("eos_pending")
    } else if offset + 1 == budget {
        Some("length_pending")
    } else if correct != proposed {
        Some("rejection_fallback_pending")
    } else if offset + 1 == B {
        Some("block_complete_pending")
    } else {
        None
    }
}

impl NerkyorDflashResearchDecoder {
    pub fn new(target: &mut CheckpointCudaTarget, weights: &DFlashQ8CudaWeights) -> Result<Self> {
        target.validate_cursor()?;
        let binding = ResearchBinding::read(target)?;
        binding.validate_weights(target, weights)?;
        if binding.policy.is_b8() {
            identity::require_b8_head()?;
        }
        let stream = Rc::clone(&target.stream);
        let normalized = empty(&stream, B * H)?;
        let logits = empty(&stream, B * V)?;
        let scalar_logits = empty(&stream, B * V)?;
        let optimized = target.head.bind_shared_normalized_bf16_rows(
            &stream,
            Rc::clone(&normalized),
            Rc::clone(&logits),
            B,
        )?;
        if binding.policy.is_b8()
            && optimized.kernel_label() != identity::B8_HEAD
        {
            return Err("bound draft head differs from sealed B8 identity".into());
        }
        let scalar = (0..B)
            .map(|row| {
                target.head.bind_shared_normalized_bf16(
                    &stream,
                    Rc::clone(&normalized),
                    row * H * 4,
                    Rc::clone(&scalar_logits),
                    row * V * 4,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let alloc = |bytes| OwnedCudaBuffer::allocate(stream.device_ordinal(), bytes);
        let ids = alloc(B * K * 4)?;
        let scores = alloc(B * K * 4)?;
        let transitions = alloc(B * K * 4)?;
        let path = alloc(B * 4)?;
        let topk = CudaQwen35VocabTopkArgsV1::bind(
            &stream,
            CudaF32View::new(
                logits.view(0, B * V * 4)?,
                PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                B * V,
            )?,
            ids.view(0, B * K * 4)?,
            scores.view(0, B * K * 4)?,
            B,
            K,
        )?;
        Ok(Self {
            stream,
            instance: Rc::new(()),
            binding,
            generation: 0,
            live: None,
            poisoned: false,
            normalized,
            logits,
            scalar_logits,
            optimized,
            scalar,
            topk,
            ids,
            scores,
            transitions,
            path,
            #[cfg(feature = "qwen35-prefill-cublas")]
            batched: None,
            #[cfg(feature = "qwen35-prefill-cublas")]
            resident: None,
        })
    }
    fn validate_head_identity(&self) -> Result<()> {
        if self.binding.policy.is_b8()
            && self.optimized.kernel_label() != identity::B8_HEAD
        {
            return Err("retained draft head differs from sealed B8 identity".into());
        }
        Ok(())
    }
    fn validate_target(&self, target: &mut CheckpointCudaTarget) -> Result<()> {
        self.validate_head_identity()?;
        target.validate_cursor()?;
        #[cfg(feature = "qwen35-augur-http")]
        if self.binding.policy.is_b8() && target.sampler.is_some() {
            return Err("B8 research is greedy-only".into());
        }
        if self.poisoned
            || !Rc::ptr_eq(&self.stream, &target.stream)
            || self.binding != ResearchBinding::read(target)?
        {
            return Err(
                "poisoned decoder, foreign target owner or changed numerical identity".into(),
            );
        }
        Ok(())
    }
    fn validate_proposal(&self, proposal: &NerkyorResearchProposal) -> Result<()> {
        self.validate_head_identity()?;
        if self.poisoned
            || !Rc::ptr_eq(&self.instance, &proposal.instance)
            || self.live != Some(proposal.generation)
            || self.binding != proposal.binding
        {
            return Err("foreign, stale, or retired research proposal".into());
        }
        Ok(())
    }
    pub fn propose(
        &mut self,
        target: &mut CheckpointCudaTarget,
        weights: &DFlashQ8CudaWeights,
    ) -> Result<NerkyorResearchProposal> {
        self.validate_target(target)?;
        if self.live.is_some()
            || self.generation == u64::MAX
            || weights.file_sha256() != self.binding.draft_sha256
        {
            return Err("live proposal, generation overflow, or mismatched draft".into());
        }
        let input = target.capture_nerkyor_dflash_research_input()?;
        let mut block = input.bind_block(target, weights)?;
        self.poisoned = true;
        let result = (|| -> Result<NerkyorResearchProposal> {
            block.prepare_context()?;
            block.submit()?;
            let path = self.project_draft(&block, input.seed.anchor_token)?;
            self.generation += 1;
            Ok(NerkyorResearchProposal {
                instance: Rc::clone(&self.instance),
                generation: self.generation,
                committed: input.committed,
                binding: self.binding.clone(),
                path,
            })
        })();
        if let Ok(proposal) = &result {
            self.live = Some(proposal.generation);
            self.poisoned = false;
        }
        result
    }
    fn project_draft(&self, block: &DFlashQ8CudaBlock, anchor: u32) -> Result<Vec<u32>> {
        self.validate_head_identity()?;
        block.synchronize()?;
        // Must drain before block can drop, even after a partial D2D enqueue.
        let _drain = Drain(Rc::clone(&self.stream));
        unsafe {
            self.normalized.copy_from_device_async_unchecked(
                &self.stream,
                0,
                block.final_hidden_rows_device()?.buffer_view(),
            )?;
        }
        self.optimized.submit()?; // Already draft-normalized: never apply target RMSNorm again.
        self.topk.submit(&self.stream)?;
        let (gates, predecessor, successor) = block.selector_inputs_device()?;
        CudaQwen35DflashQ8SelectorArgsV1::bind(
            &self.stream,
            CudaQwen35DflashQ8SelectorViewsV1 {
                gates,
                predecessor,
                successor,
                candidate_ids: self.ids.view(0, B * K * 4)?,
                unary_scores: self.scores.view(0, B * K * 4)?,
                transition_scores: self.transitions.view(0, B * K * 4)?,
                path_ids: self.path.view(0, B * 4)?,
            },
            anchor,
        )?
        .submit(&self.stream)?;
        let mut bytes = vec![0; B * 4];
        self.path.copy_to_host_sync(&self.stream, &mut bytes)?;
        let path: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        if path[0] != anchor || path.iter().any(|&id| id as usize >= V) {
            return Err("invalid research candidate path".into());
        }
        Ok(path)
    }
    /// Completion only, including after failure. This never clears poison,
    /// retires a proposal, commits state or grants reset/reuse authority.
    pub(crate) fn drain_research_work(&mut self) -> Result<()> {
        #[cfg(feature = "qwen35-prefill-cublas")]
        if let Some(state) = &self.resident {
            state.drain()?;
        }
        self.stream.synchronize()?;
        Ok(())
    }
    pub fn discard(&mut self, proposal: NerkyorResearchProposal) -> Result<()> {
        self.validate_proposal(&proposal)?;
        self.stream.synchronize()?;
        self.live = None;
        Ok(())
    }
    /// Diagnostic numerical/performance experiment, outside generation. Compares
    /// all vocabulary outputs with OriginalF32 scalar products/F64 reductions.
    /// Independent source BF16/F64 samples are checked by the external monitor.
    pub fn audit_head(
        &self,
        proposal: &NerkyorResearchProposal,
        iterations: usize,
    ) -> Result<NerkyorHeadAudit> {
        self.validate_proposal(proposal)?;
        if !(1..=100).contains(&iterations) {
            return Err("head timing iterations must be 1..=100".into());
        }
        let _drain = Drain(Rc::clone(&self.stream));
        for op in &self.scalar {
            op.submit_for_plan()?;
        }
        self.optimized.submit()?;
        self.stream.synchronize()?;
        let start = OwnedCudaEvent::create(self.stream.device_ordinal())?;
        let end = OwnedCudaEvent::create(self.stream.device_ordinal())?;
        let time = |scalar: bool| -> Result<f32> {
            start.record(&self.stream)?;
            for _ in 0..iterations {
                if scalar {
                    for op in &self.scalar {
                        op.submit_for_plan()?;
                    }
                } else {
                    self.optimized.submit()?;
                }
            }
            end.record(&self.stream)?;
            end.synchronize()?;
            Ok(start.elapsed_ms(&end)? / iterations as f32)
        };
        let scalar_gpu_ms = time(true)?;
        let optimized_gpu_ms = time(false)?;
        let normalized_hidden = read_f32(&self.stream, &self.normalized)?;
        let optimized_logits = read_f32(&self.stream, &self.logits)?;
        let scalar_logits = read_f32(&self.stream, &self.scalar_logits)?;
        if normalized_hidden
            .iter()
            .chain(&optimized_logits)
            .chain(&scalar_logits)
            .any(|v| !v.is_finite())
        {
            return Err("nonfinite complete shared head output".into());
        }
        Ok(NerkyorHeadAudit {
            normalized_hidden,
            optimized_logits,
            scalar_logits,
            optimized_kernel: self.optimized.kernel_label(),
            optimized_gpu_ms,
            scalar_gpu_ms,
            timing_iterations: iterations,
        })
    }
    /// Final token is selected but unconsumed, including EOS and a rejection's
    /// correct fallback. Caller must consume it before continuing, or stop.
    /// Invalid inputs are rejected before token selection or target mutation.
    pub fn verify_serial(
        &mut self,
        target: &mut CheckpointCudaTarget,
        proposal: &NerkyorResearchProposal,
        budget: usize,
        eos: &[u32],
    ) -> Result<NerkyorSerialRound> {
        self.verify_serial_observed(target, proposal, budget, eos, |_, _| Ok(()))
    }
    // Same verifier with a read-only diagnostic boundary observer. Production
    // passes a no-op; unit tests read every full logit/capture/gate boundary.
    // Observer readback is never included in speed claims.
    fn verify_serial_observed(
        &mut self,
        target: &mut CheckpointCudaTarget,
        proposal: &NerkyorResearchProposal,
        budget: usize,
        eos: &[u32],
        mut observe: impl FnMut(&mut CheckpointCudaTarget, usize) -> Result<()>,
    ) -> Result<NerkyorSerialRound> {
        self.validate_target(target)?;
        self.validate_proposal(proposal)?;
        if self.binding.policy.is_b8() {
            return Err("sealed B8 proposal must use its B8 verifier".into());
        }
        if budget == 0
            || eos.is_empty()
            || eos.iter().any(|&id| id as usize >= V)
            || target
                .dflash_capture
                .as_ref()
                .ok_or("missing target capture")?
                .committed_identity()?
                != proposal.committed
            || target.consumed != proposal.committed.2
        {
            return Err("stale cursor, zero budget, or invalid EOS policy".into());
        }
        self.poisoned = true;
        self.live = None;
        let result = (|| -> Result<NerkyorSerialRound> {
            let start_consumed = target.consumed;
            let mut output_tokens = Vec::with_capacity(B.min(budget));
            let mut matched_draft_tokens = 0;
            let mut rejection_offset = None;
            for offset in 0..B {
                let correct = target.select_next_token()?;
                let proposed = proposal.path[offset];
                if offset == 0 && correct != proposed {
                    return Err("selected research anchor changed".into());
                }
                if offset > 0 {
                    if correct == proposed {
                        matched_draft_tokens += 1;
                    } else {
                        rejection_offset = Some(offset);
                    }
                }
                output_tokens.push(correct);
                observe(target, offset)?;
                if let Some(stop) = stop_after(correct, proposed, offset, budget, eos) {
                    self.stream.synchronize()?;
                    return Ok(NerkyorSerialRound {
                        output_tokens,
                        matched_draft_tokens,
                        rejection_offset,
                        stop,
                        start_consumed,
                        end_consumed: target.consumed,
                    });
                }
                // Only a canonical selected token, never a rejected candidate.
                target.consume_selected()?;
            }
            Err("research verifier did not terminate".into())
        })();
        if result.is_ok() {
            self.poisoned = false;
        } else {
            target.poisoned = true;
        }
        result
    }
}
impl Drop for NerkyorDflashResearchDecoder {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn serial_policy_eos_budget_rejection_and_full_block_leave_final_pending() {
        assert_eq!(stop_after(9, 9, 0, 8, &[9]), Some("eos_pending"));
        assert_eq!(stop_after(3, 3, 0, 1, &[9]), Some("length_pending"));
        assert_eq!(
            stop_after(3, 4, 2, 8, &[9]),
            Some("rejection_fallback_pending")
        );
        assert_eq!(stop_after(3, 4, 2, 3, &[9]), Some("length_pending"));
        assert_eq!(
            stop_after(3, 3, 7, 99, &[9]),
            Some("block_complete_pending")
        );
        // Budget takes precedence when the last block token also fills it.
        assert_eq!(stop_after(3, 3, 7, 8, &[9]), Some("length_pending"));
        assert_eq!(stop_after(3, 3, 1, 8, &[9]), None);
    }
}

#[cfg(test)]
#[path = "decoder_gpu_tests.rs"]
mod gpu_tests;

#[cfg(feature = "qwen35-prefill-cublas")]
#[path = "verifier.rs"]
mod verifier;
#[cfg(feature = "qwen35-prefill-cublas")]
pub use verifier::NerkyorBatchedRound;

#[cfg(feature = "qwen35-prefill-cublas")]
#[path = "resident.rs"]
mod resident;

#[cfg(feature = "qwen35-prefill-cublas")]
#[path = "prefill.rs"]
mod prefill;
