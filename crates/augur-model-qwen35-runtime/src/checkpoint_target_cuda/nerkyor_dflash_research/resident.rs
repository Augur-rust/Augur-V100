//! Transport-only resident proposal lane. Default Q8 arithmetic is unchanged.
//! No feature payload SHA is fabricated: the device lease carries the sealed
//! source/stream/epoch/revision and initialized range instead.
use super::*;
use crate::checkpoint_target_cuda::dflash_capture::OwnerFeatureLease;
use augur_backend::{CudaBf16TokenRowsGatherArgsV1, CudaBf16WeightViewV1};
pub(super) struct Resident {
    draft: DFlashQ8CudaBlock,
    stream: Rc<OwnedCudaStream>,
    ids: OwnedCudaBuffer,
    noise: OwnedCudaBuffer,
    gather: CudaBf16TokenRowsGatherArgsV1,
    unused_seed: bool,
}
impl Resident {
    fn seed(&self, anchor: u32) -> Result<()> {
        let mut ids = [DFLASH_SEED_MASK_TOKEN; B];
        ids[0] = anchor;
        let mut bytes = [0u8; B * 4];
        for (i, id) in ids.into_iter().enumerate() {
            bytes[i * 4..i * 4 + 4].copy_from_slice(&id.to_le_bytes());
        }
        self.ids.copy_from_host_sync(&self.stream, &bytes)?;
        self.gather.submit_for_plan(&self.stream)?;
        Ok(())
    }
    fn new(
        target: &mut CheckpointCudaTarget,
        weights: &DFlashQ8CudaWeights,
        anchor: u32,
        generation: u64,
    ) -> Result<Self> {
        let stream = Rc::clone(&target.stream);
        let _drain = Drain(Rc::clone(&stream));
        let ids = OwnedCudaBuffer::allocate(stream.device_ordinal(), B * 4)?;
        let noise = OwnedCudaBuffer::allocate(stream.device_ordinal(), B * H * 4)?;
        let gather = CudaBf16TokenRowsGatherArgsV1::bind(
            &stream,
            CudaBf16WeightViewV1::new(
                target._embedding.view(0, V * H * 2)?,
                PhysicalLayoutId::DEVICE_CONTIGUOUS_BF16_CUDA_V1,
                V,
                H,
            )?,
            ids.view(0, B * 4)?,
            CudaF32View::new(
                noise.view(0, B * H * 4)?,
                PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                B * H,
            )?,
            B,
        )?;
        let mut tokens = [DFLASH_SEED_MASK_TOKEN; B];
        tokens[0] = anchor;
        let bytes: Vec<u8> = tokens.iter().flat_map(|id| id.to_le_bytes()).collect();
        ids.copy_from_host_sync(&stream, &bytes)?;
        gather.submit_for_plan(&stream)?;
        let lease = target
            .dflash_capture
            .as_mut()
            .ok_or("missing resident feature capture")?
            .device_lease()?;
        // The constructor drains both the producer and its borrowed D2D copies.
        // Do not enable QUASAR row/tile/llama arithmetic switches here.
        let draft = DFlashQ8CudaBlock::from_augur_owner(
            weights,
            &lease,
            noise.view(0, B * H * 4)?,
            anchor,
            generation,
        )?;
        Ok(Self {
            draft,
            stream,
            ids,
            noise,
            gather,
            unused_seed: true,
        })
    }
    fn refresh(
        &mut self,
        lease: &OwnerFeatureLease<'_>,
        anchor: u32,
        generation: u64,
    ) -> Result<()> {
        let noise = self.noise.view(0, B * H * 4)?;
        if self.unused_seed {
            self.draft
                .seed_preallocated_augur_owner(lease, noise, anchor, generation)?;
            self.unused_seed = false;
        } else {
            self.draft
                .refresh_augur_owner(lease, noise, anchor, generation)?;
        }
        Ok(())
    }
    pub(super) fn drain(&self) -> Result<()> {
        self.draft.synchronize()?;
        self.stream.synchronize()?;
        Ok(())
    }
    #[cfg(test)]
    pub(super) fn noise_snapshot(&self) -> Result<Vec<f32>> {
        read_f32(&self.stream, &self.noise)
    }
}
impl Drop for Resident {
    fn drop(&mut self) {
        let _ = self.draft.synchronize();
        let _ = self.stream.synchronize();
    }
}
impl NerkyorDflashResearchDecoder {
    /// Allocate the reusable 2048-row draft owner once, after target prefill.
    /// B8 numerical admission remains separate; this changes transport only.
    pub fn prepare_resident_proposal(
        &mut self,
        target: &mut CheckpointCudaTarget,
        weights: &DFlashQ8CudaWeights,
    ) -> Result<()> {
        self.validate_target(target)?;
        self.binding.validate_weights(target, weights)?;
        if !self.binding.policy.is_b8()
            || self.live.is_some()
            || !target.logits_current
            || target
                .consumed
                .checked_add(B)
                .is_none_or(|end| end > target.capacity)
        {
            return Err(
                "resident proposal requires a healthy sealed B8 target and no live proposal".into(),
            );
        }
        let committed = target
            .dflash_capture
            .as_ref()
            .ok_or("missing resident capture")?
            .committed_identity()?;
        if committed.2 != target.consumed {
            return Err("resident capture cursor drift".into());
        }
        if self.resident.is_some() {
            return Ok(());
        }
        let generation = self
            .generation
            .checked_add(1)
            .ok_or("resident seed generation overflow")?;
        let anchor = target.select_next_token()?;
        self.poisoned = true;
        let state = Resident::new(target, weights, anchor, generation)?;
        self.resident = Some(state);
        self.generation = generation;
        self.poisoned = false;
        Ok(())
    }
    /// Release draft workspace only at a completed/discarded proposal boundary.
    /// The numerical family and generation counter are never reset by release.
    pub fn release_resident_proposal(&mut self) -> Result<()> {
        if self.poisoned || self.live.is_some() {
            return Err("resident release requires a healthy quiescent decoder".into());
        }
        self.stream.synchronize()?;
        drop(self.resident.take());
        Ok(())
    }
    /// Reuse GPU feature/embedding storage and the complete draft allocation.
    /// Host traffic is only seed IDs (H2D) and compact candidate IDs (D2H), plus
    /// the existing 4-byte anchor selection if it was not already cached.
    pub fn propose_resident(
        &mut self,
        target: &mut CheckpointCudaTarget,
        weights: &DFlashQ8CudaWeights,
    ) -> Result<NerkyorResearchProposal> {
        self.validate_target(target)?;
        self.binding.validate_weights(target, weights)?;
        if !self.binding.policy.is_b8()
            || self.live.is_some()
            || self.resident.is_none()
            || self.generation == u64::MAX
            || target
                .consumed
                .checked_add(B)
                .is_none_or(|end| end > target.capacity)
        {
            return Err("unprepared, live, overflow or invalid resident proposal owner".into());
        }
        let committed = target
            .dflash_capture
            .as_ref()
            .ok_or("missing resident capture")?
            .committed_identity()?;
        if committed.2 != target.consumed {
            return Err("resident capture cursor drift".into());
        }
        let anchor = target.select_next_token()?;
        let generation = self.generation + 1;
        self.poisoned = true;
        let mut state = self.resident.take().ok_or("missing resident draft state")?;
        let _drain = Drain(Rc::clone(&self.stream));
        let result = (|| -> Result<NerkyorResearchProposal> {
            state.seed(anchor)?;
            let lease = target.dflash_capture.as_mut().unwrap().device_lease()?;
            let identity = lease.identity();
            if identity.capture_identity() != committed {
                return Err("resident feature lease changed".into());
            }
            state.refresh(&lease, anchor, generation)?;
            drop(lease);
            state.draft.prepare_context()?;
            state.draft.submit()?;
            if state.draft.submitted_augur_identity()? != (generation, anchor, identity) {
                return Err("resident draft provenance drift".into());
            }
            let path = self.project_draft(&state.draft, anchor)?;
            Ok(NerkyorResearchProposal {
                instance: Rc::clone(&self.instance),
                generation,
                committed,
                binding: self.binding.clone(),
                path,
            })
        })();
        self.resident = Some(state);
        if let Ok(p) = &result {
            self.generation = generation;
            self.live = Some(p.generation);
            self.poisoned = false;
        }
        result
    }
}
