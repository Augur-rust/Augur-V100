//! Explicit new-model capture/embedding seam, not the QUASAR product bridge.
//! Candidate, serial and explicitly sealed B8 verification remain research-only.
//! No selector calibration or session/HTTP publication authority.
use super::*;
use crate::{
    DFlashDraftBlockInput, DFlashQ8CudaBlock, DFlashQ8CudaWeights,
    DFlashTargetFeatureCaptureReceipt, DFLASH_SEED_MASK_TOKEN,
};
use sha2::{Digest, Sha256};

mod identity;
use identity::{ResearchBinding, ResearchPolicy, validate_draft};

const H: usize = 5120;
const B: usize = 8;

/// Move-only evidence from one target stream, reset epoch and consumed cursor.
/// Public accessors are read-only; caller-authored feature receipts cannot enter
/// this seam. D2D is copied and drained by the block constructor, not borrowed.
pub struct NerkyorDflashResearchInput {
    stream: Rc<OwnedCudaStream>,
    committed: (u64, u64, usize),
    binding: ResearchBinding,
    seed: DFlashDraftBlockInput,
    capture: DFlashTargetFeatureCaptureReceipt,
    features: Vec<f32>,
    noise: Vec<f32>,
}

fn same_owner_revision<T>(
    captured: &Rc<T>,
    current: &Rc<T>,
    captured_revision: (u64, u64, usize),
    current_revision: (u64, u64, usize),
) -> bool {
    Rc::ptr_eq(captured, current) && captured_revision == current_revision
}

impl CheckpointCudaTarget {
    /// Separate opt-in numerical identity. Does not relax production capture or
    /// QUASAR session/QPN guards. Must precede consuming any token.
    pub fn enable_nerkyor_dflash_research_capture(
        &mut self,
        weights: &DFlashQ8CudaWeights,
    ) -> Result<()> {
        self.validate_cursor()?;
        let policy = ResearchPolicy::from_info(&self.info)?;
        validate_draft(self, weights)?;
        self.install_feature_capture(serde_json::to_vec(&policy.contract(
            self._artifact.identity(), weights.file_sha256(),
        ))?)
    }

    /// Fresh-owner opt-in to the sealed B8 numerical family. Scalar prefill and
    /// selected-token continuation remain explicit members of this family.
    pub fn enable_nerkyor_batched_dflash_research_capture(&mut self, weights: &DFlashQ8CudaWeights) -> Result<()> {
        self.validate_cursor()?;
        if ResearchPolicy::from_info(&self.info)? != ResearchPolicy::GateF32Cta {
            return Err("B8 capture requires the explicit gate-F32 projection policy".into());
        }
        identity::require_b8_head()?; validate_draft(self,weights)?;
        #[cfg(feature="qwen35-augur-http")]
        if self.sampler.is_some() { return Err("B8 research is greedy-only".into()); }
        self.install_feature_capture(serde_json::to_vec(&ResearchPolicy::GateF32CtaB8.contract(self._artifact.identity(),weights.file_sha256()))?)?;
        self.info["nerkyor_target_execution"] = serde_json::json!(identity::B8_EXECUTION);
        Ok(())
    }

    /// Independent arithmetic family: core-TU attention for both full prefill
    /// and B8 verification, never an alias for the old scalar-prefill identity.
    pub fn enable_nerkyor_core_b8_prefill_research_capture(&mut self, weights: &DFlashQ8CudaWeights) -> Result<()> {
        self.validate_cursor()?;
        if ResearchPolicy::from_info(&self.info)? != ResearchPolicy::GateF32Cta {
            return Err("core B8 prefill requires a fresh gate-F32 target".into());
        }
        identity::require_b8_head()?; validate_draft(self,weights)?;
        #[cfg(feature="qwen35-augur-http")]
        if self.sampler.is_some() { return Err("core B8 prefill is greedy-only".into()); }
        self.install_feature_capture(serde_json::to_vec(&ResearchPolicy::CoreB8Prefill.contract(self._artifact.identity(),weights.file_sha256()))?)?;
        self.info["nerkyor_target_execution"] = serde_json::json!(identity::CORE_B8_EXECUTION);
        Ok(())
    }

    pub fn capture_nerkyor_dflash_research_input(&mut self) -> Result<NerkyorDflashResearchInput> {
        self.validate_cursor()?;
        let binding = ResearchBinding::read(self)?;
        if self
            .consumed
            .checked_add(B)
            .is_none_or(|end| end > self.capacity)
        {
            return Err("research draft needs eight available absolute positions".into());
        }
        let capture = self
            .dflash_capture
            .as_ref()
            .ok_or("missing research capture")?;
        let committed = capture.committed_identity()?;
        if committed.2 != self.consumed {
            return Err("research capture cursor drift".into());
        }
        let snapshot = capture.snapshot()?;
        let anchor = self.select_next_token()?;
        let mut tokens = vec![DFLASH_SEED_MASK_TOKEN; B];
        tokens[0] = anchor;
        let seed = DFlashDraftBlockInput {
            request_generation: snapshot.receipt.request_generation,
            absolute_position_start: self.consumed,
            anchor_token: anchor,
            block_tokens: tokens,
            mask_token_id: DFLASH_SEED_MASK_TOKEN,
            target_feature_sha256: snapshot.receipt.feature_sha256,
        };
        let mut noise = Vec::with_capacity(B * H);
        // Only the two distinct rows; repeat the actual mask embedding seven
        // times. Never substitute zeros, random activations, or another model.
        for (token, repeats) in [(anchor, 1), (DFLASH_SEED_MASK_TOKEN, B - 1)] {
            let mut bytes = vec![0; H * 2];
            self._artifact.read_tensor_range_into(
                "model.language_model.embed_tokens.weight",
                u64::from(token) * (H * 2) as u64,
                &mut bytes,
            )?;
            let row: Vec<f32> = bytes
                .chunks_exact(2)
                .map(|b| f32::from_bits(u32::from(u16::from_le_bytes(b.try_into().unwrap())) << 16))
                .collect();
            if row.iter().any(|v| !v.is_finite()) {
                return Err("nonfinite target embedding".into());
            }
            for _ in 0..repeats {
                noise.extend_from_slice(&row);
            }
        }
        Ok(NerkyorDflashResearchInput {
            stream: Rc::clone(&self.stream),
            committed,
            binding,
            seed,
            capture: snapshot.receipt,
            features: snapshot
                .little_endian_f32
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
            noise,
        })
    }
}

impl NerkyorDflashResearchInput {
    pub fn seed(&self) -> &DFlashDraftBlockInput {
        &self.seed
    }
    pub fn capture(&self) -> &DFlashTargetFeatureCaptureReceipt {
        &self.capture
    }
    pub fn features(&self) -> &[f32] {
        &self.features
    }
    pub fn noise_embedding(&self) -> &[f32] {
        &self.noise
    }
    pub fn noise_sha256(&self) -> String {
        let mut sha = Sha256::new();
        for v in &self.noise {
            sha.update(v.to_le_bytes());
        }
        format!("{:x}", sha.finalize())
    }
    pub fn bind_block(
        &self,
        target: &mut CheckpointCudaTarget,
        weights: &DFlashQ8CudaWeights,
    ) -> Result<DFlashQ8CudaBlock> {
        target.validate_cursor()?;
        if self.binding != ResearchBinding::read(target)? {
            return Err("research input numerical identity changed".into());
        }
        self.binding.validate_weights(target, weights)?;
        let capture = target
            .dflash_capture
            .as_ref()
            .ok_or("missing research capture")?;
        if !same_owner_revision(
            &self.stream,
            &target.stream,
            self.committed,
            capture.committed_identity()?,
        ) || target.consumed != self.seed.absolute_position_start
        {
            return Err("stale, foreign, or mismatched Nerkyor research capture/draft".into());
        }
        let (snapshot, view) = capture.snapshot_and_device_view()?;
        if snapshot.receipt != self.capture {
            return Err("research capture bytes/identity changed".into());
        }
        self.capture.validate_for_block(
            weights.config(),
            &self.seed,
            self.features.len(),
            self.seed.target_feature_sha256,
        )?;
        Ok(DFlashQ8CudaBlock::from_resident_device_features(
            weights,
            &self.seed,
            &self.capture,
            view,
            &self.noise,
            self.capture.context_start_position,
            2048,
            &target.stream,
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn owner_incarnation_epoch_revision_and_cursor_are_all_required() {
        let a = Rc::new(());
        let same = Rc::clone(&a);
        let foreign = Rc::new(());
        let revision = (1, 2, 22);
        assert!(same_owner_revision(&a, &same, revision, revision));
        assert!(!same_owner_revision(&a, &foreign, revision, revision));
        for changed in [(2, 2, 22), (1, 3, 22), (1, 2, 23)] {
            assert!(!same_owner_revision(&a, &same, revision, changed));
        }
    }
}

mod decoder;
pub use decoder::{
    NerkyorDflashResearchDecoder, NerkyorHeadAudit, NerkyorResearchProposal, NerkyorSerialRound,
};

#[cfg(feature = "qwen35-prefill-cublas")]
pub(crate) mod batched;
#[cfg(feature = "qwen35-prefill-cublas")]
pub use decoder::NerkyorBatchedRound;


#[cfg(test)]
impl CheckpointCudaTarget {
    pub(crate) fn nerkyor_all_state_for_test(&mut self) -> Result<Vec<(String,Vec<f32>)>> {
        let mut rows=vec![("logits".into(),self.logits_snapshot()?)];
        let decode=|raw:Vec<u8>| -> Vec<f32> { raw.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect() };
        let capture=self.dflash_capture.as_ref().ok_or("missing capture")?.snapshot()?;
        rows.push(("capture".into(),decode(capture.little_endian_f32)));
        for (i,layer) in self.layers.iter().enumerate() {
            let buffers=match layer {
                Layer::Linear(l)=>crate::checkpoint_linear_cuda::nerkyor_batch::diagnostic_state(l)?,
                Layer::Full(l)=>crate::checkpoint_full_cuda::nerkyor_batch::diagnostic_state(l)?,
            };
            for (name,raw) in buffers { rows.push((format!("layer-{i}-{name}"),decode(raw))); }
        }
        Ok(rows)
    }
}
