//! Resident handoff: sealed producer lease, device BF16 seed gather, compact path.
//! No per-round feature/logit D2H or file I/O. Canonical verifier is still B1.
use super::*;
use crate::checkpoint_target_cuda::dflash_capture::OwnerFeatureIdentity;

/// Opt-in proposal component bill (AUGUR_DFLASH_PROPOSAL_BILL=1). Draft-stream GPU spans and
/// host walls only; the head stream span is bounded by host walls because it starts after a sync.
struct ProposalBill {events:[augur_backend::OwnedCudaEvent;4]}
impl ProposalBill {
    fn enabled()->bool {std::env::var("AUGUR_DFLASH_PROPOSAL_BILL").as_deref()==Ok("1")}
    fn new(device:i32)->Result<Self> {
        Ok(Self{events:[augur_backend::OwnedCudaEvent::create(device)?,augur_backend::OwnedCudaEvent::create(device)?,
            augur_backend::OwnedCudaEvent::create(device)?,augur_backend::OwnedCudaEvent::create(device)?]})
    }
}
#[derive(Clone,Copy,Debug,serde::Serialize)]
pub(crate) struct ProposalBillRow {pub context_gpu_ms:f64,pub draft_layers_gpu_ms:f64,pub host_submit_ms:f64,
    pub host_head_enqueue_ms:f64,pub host_total_through_path_ms:f64,pub host_total_ms:f64}
thread_local! {pub(crate) static PROPOSAL_BILL:std::cell::RefCell<Vec<ProposalBillRow>>=const{std::cell::RefCell::new(Vec::new())};}
pub(super) struct ResidentDraftState {
    // Draft drains before source buffers are released.
    pub(super) draft: Option<DFlashQ8CudaBlock>,
    stream: Rc<OwnedCudaStream>,
    ids: OwnedCudaBuffer,
    noise: OwnedCudaBuffer,
    gathers: Vec<CudaBf16TokenRowGatherArgsV1>,
    needs_first_seed: bool,
}
impl ResidentDraftState {
    fn configure(draft: &mut DFlashQ8CudaBlock, llama_projection: bool) -> Result<()> {
        draft.enable_context_rows_launch()?;
        draft.enable_block_rows_launch()?;
        draft.enable_q8_tile4()?;
        if llama_projection { draft.enable_llama_projection()?; }
        Ok(())
    }
    fn new(target: &CheckpointCudaTarget) -> Result<Self> {
        let stream = Rc::clone(&target.stream);
        let ids = OwnedCudaBuffer::allocate(stream.device_ordinal(), B * 4)?;
        let noise = OwnedCudaBuffer::allocate(stream.device_ordinal(), B * H * 4)?;
        let mut gathers = Vec::with_capacity(B);
        for row in 0..B {
            gathers.push(CudaBf16TokenRowGatherArgsV1::bind(
                &stream,
                CudaBf16WeightViewV1::new(
                    target._embedding.view(0, V * H * 2)?,
                    PhysicalLayoutId::DEVICE_CONTIGUOUS_BF16_CUDA_V1,
                    V,
                    H,
                )?,
                ids.view(row * 4, 4)?,
                CudaF32View::new(
                    noise.view(row * H * 4, H * 4)?,
                    PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                    H,
                )?,
            )?);
        }
        Ok(Self {
            draft: None,
            stream,
            ids,
            noise,
            gathers,
            needs_first_seed: false,
        })
    }
    fn seed(&self, anchor: u32) -> Result<()> {
        let mut tokens = [DFLASH_SEED_MASK_TOKEN; B];
        tokens[0] = anchor;
        let bytes: Vec<u8> = tokens.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.ids.copy_from_host_sync(&self.stream, &bytes)?;
        for gather in &self.gathers {
            gather.submit_for_plan(&self.stream)?;
        }
        Ok(())
    }
}
impl Drop for ResidentDraftState {
    fn drop(&mut self) {
        if let Some(draft) = &self.draft {
            let _ = draft.synchronize();
        }
        let _ = self.stream.synchronize();
    }
}
/// Move-only owner-bound path, not a claimed payload-SHA capture receipt.
/// This grants no Session/HTTP commit or publication authority.
pub struct AugurResidentDraftProposal {
    pub(super) instance: Rc<()>,
    pub(super) generation: u64,
    pub(super) identity: OwnerFeatureIdentity,
    pub(super) stream: Rc<OwnedCudaStream>,
    pub(super) path: Vec<u32>,
    pub(super) probability: Option<crate::augur_speculative_sampling::Proposal>,
}
impl AugurResidentDraftProposal {
    pub fn path(&self) -> &[u32] {
        &self.path
    }
    pub fn initialized_feature_range(&self) -> (usize, usize) {
        (self.identity.start(), self.identity.end())
    }
}
impl AugurDflashBridge {
    /// Test-only residency setup: allocate the real draft block, but do not run
    /// drafting or claim the experimental prefill feature profile is qualified.
    /// from_augur_owner completes its borrowed-feature copies before returning.
    #[cfg(any(test, feature = "qwen35-augur-http"))]
    pub(crate) fn retain_prefill_benchmark_workspace(
        &mut self,
        target: &mut CheckpointCudaTarget,
    ) -> Result<()> {
        self.validate_target(target)?;
        if self.resident.is_some() || self.live_proposal.is_some() || self.generation != 0 {
            return Err("prefill benchmark residency requires a fresh bridge".into());
        }
        let anchor = target.select_next_token()?;
        let mut state = ResidentDraftState::new(target)?;
        state.seed(anchor)?;
        let lease = target.dflash_capture.as_mut().unwrap().device_lease()?;
        let noise = state.noise.view(0, B * H * 4)?;
        state.draft = Some(DFlashQ8CudaBlock::from_augur_owner(
            &self.weights,
            &lease,
            noise,
            anchor,
            1,
        )?);
        // Product startup preallocates this same block. The first real proposal
        // must not silently use slower default policies merely because it exists.
        ResidentDraftState::configure(state.draft.as_mut().unwrap(), self.llama_projection)?;
        state.needs_first_seed = true;
        drop(lease);
        self.stream.synchronize()?;
        self.resident = Some(state);
        // Generation1 identifies the unused startup seed, not a draft proposal.
        // First real proposal uses generation2 and records its real target lease.
        self.generation = 1;
        Ok(())
    }
    pub fn propose_resident(
        &mut self,
        target: &mut CheckpointCudaTarget,
    ) -> Result<AugurResidentDraftProposal> {
        self.validate_target(target)?;
        target.validate_cursor()?;
        if self.live_proposal.is_some()
            || target
                .consumed
                .checked_add(B)
                .is_none_or(|end| end > target.capacity)
        {
            return Err("prior proposal live or resident draft query exceeds capacity".into());
        }
        let capture_id = target
            .dflash_capture
            .as_ref()
            .unwrap()
            .committed_identity()?;
        if capture_id.2 != target.consumed {
            return Err("target/capture cursor drift".into());
        }
        let generation = self
            .generation
            .checked_add(1)
            .ok_or("draft generation overflow")?;
        self.poisoned = true;
        let result = (|| -> Result<AugurResidentDraftProposal> {
            let anchor = target.select_next_token()?;
            if self.resident.is_none() {
                self.resident = Some(ResidentDraftState::new(target)?);
            }
            let state = self.resident.as_mut().unwrap();
            // This retained stream drain also covers a partial seed-gather error.
            let source_stream = Rc::clone(&self.stream);
            let _source_drain = ConsumerDrain(&source_stream);
            state.seed(anchor)?;
            let lease = target.dflash_capture.as_mut().unwrap().device_lease()?;
            let identity = lease.identity();
            let noise = state.noise.view(0, B * H * 4)?;
            if let Some(draft) = &mut state.draft {
                if state.needs_first_seed {
                    draft.seed_preallocated_augur_owner(&lease, noise, anchor, generation)?;
                    state.needs_first_seed = false;
                } else if self.reuse_context {
                    draft.refresh_augur_owner_retaining_context(&lease, noise, anchor, generation)?;
                } else {
                    draft.refresh_augur_owner(&lease, noise, anchor, generation)?;
                }
            } else {
                let mut draft = DFlashQ8CudaBlock::from_augur_owner(
                    &self.weights,
                    &lease,
                    noise,
                    anchor,
                    generation,
                )?;
                ResidentDraftState::configure(&mut draft, self.llama_projection)?;
                state.draft = Some(draft);
            }
            drop(lease); // All borrowed-source copies have completed, even on error.
            let draft = state.draft.as_mut().unwrap();
            #[cfg(feature="qwen35-augur-http")]
            let phase=self.http_english_gate && !self.http_global_gate;
            #[cfg(not(feature="qwen35-augur-http"))]
            let phase=false;
            draft.select_resident_gate(self.selector_gate_enabled,phase)?;
            // Opt-in proposal bill: events only, read after the existing path readback sync.
            let bill=ProposalBill::enabled().then(||ProposalBill::new(draft.diagnostic_stream().device_ordinal())).transpose()?;
            let host_started=std::time::Instant::now();
            if let Some(b)=&bill {b.events[0].record(draft.diagnostic_stream())?;}
            draft.prepare_context()?;
            if let Some(b)=&bill {b.events[1].record(draft.diagnostic_stream())?;}
            draft.submit()?;
            if let Some(b)=&bill {b.events[2].record(draft.diagnostic_stream())?;}
            let host_submitted=host_started.elapsed().as_secs_f64();
            if draft.submitted_augur_identity()? != (generation, anchor, identity) {
                return Err("foreign/stale resident draft hidden".into());
            }
            // submit_hidden begins with the existing draft synchronize, so the head-stream
            // start event follows draft completion; its stop precedes the existing path readback.
            let head_started=std::time::Instant::now();
            self.head.submit_hidden(draft, anchor, target.consumed)?;
            if let Some(b)=&bill {b.events[3].record(&self.stream)?;}
            let host_head_enqueued=head_started.elapsed().as_secs_f64();
            let path = self.head.read_path(anchor)?;
            let host_through_path=host_started.elapsed().as_secs_f64();
            let probability=self.head.read_probability(target.consumed,&path)?;
            if let Some(b)=&bill {
                let ms=|a:usize,z:usize|->Result<f64>{Ok(f64::from(b.events[a].elapsed_ms(&b.events[z])?))};
                let row=ProposalBillRow{context_gpu_ms:ms(0,1)?,draft_layers_gpu_ms:ms(1,2)?,
                    host_submit_ms:1e3*host_submitted,host_head_enqueue_ms:1e3*host_head_enqueued,
                    host_total_through_path_ms:1e3*host_through_path,host_total_ms:1e3*host_started.elapsed().as_secs_f64()};
                PROPOSAL_BILL.with(|cell|cell.borrow_mut().push(row));
            }
            Ok(AugurResidentDraftProposal {
                probability,
                instance: Rc::clone(&self.instance),
                generation,
                identity,
                stream: Rc::clone(&self.stream),
                path,
            })
        })();
        if result.is_ok() {
            self.generation = generation;
            self.live_proposal = Some(generation);
            self.poisoned = false;
        } else {
            target.poisoned = true;
        }
        result
    }
    pub fn discard_resident(&mut self, proposal: AugurResidentDraftProposal) -> Result<()> {
        if !Rc::ptr_eq(&self.instance, &proposal.instance)
            || self.live_proposal != Some(proposal.generation)
            || !Rc::ptr_eq(&self.stream, &proposal.stream)
        {
            return Err("foreign/stale resident discard".into());
        }
        self.live_proposal = None;
        Ok(())
    }
    pub fn verify_resident_serial(
        &mut self,
        target: &mut CheckpointCudaTarget,
        proposal: AugurResidentDraftProposal,
        remaining: usize,
        eos: &[u32],
    ) -> Result<SerialVerifiedRound> {
        if proposal.probability.is_some(){return Err("probability proposal requires batched verification".into());}
        self.verify_bound_path(
            target,
            &proposal.instance,
            proposal.generation,
            proposal.identity.capture_identity(),
            &proposal.stream,
            &proposal.path,
            None,
            remaining,
            eos,
        )
    }
    /// Kernel used for the per-round B8 draft block projections; None before the resident draft exists.
    pub fn draft_block_projection_kernel(&self)->Option<&'static str> {
        self.resident.as_ref().and_then(|s|s.draft.as_ref()).map(|d|d.block_projection_kernel())
    }
    fn has_batched_verifier(&self)->bool {
        #[cfg(feature="qwen35-augur-http")] {self.batched.is_some()}
        #[cfg(not(feature="qwen35-augur-http"))] {false}
    }
    pub fn resident_workspace_info(&self) -> Value {
        let target_width = self.batched.as_ref().map_or(B, |b| b.width);
        json!({"schema":"quasar-q8-resident-owner-workspace-v1",
            "provenance":"sealed_source_owner_epoch_revision_initialized_extent_NOT_payload_SHA256",
            "target_width":target_width,
            "target_rows_executed":target_width,
            "draft_workspace_bytes":self.resident.as_ref().and_then(|s|s.draft.as_ref()).map(|d|d.workspace_bytes()),
            "seed_workspace_bytes":self.resident.as_ref().map_or(0, |_| B*4+B*H*4),
            "feature_payload_D2H_bytes_per_round":0,"full_logits_D2H_bytes_per_round":0,
            "seed_H2D_bytes_per_round":B*4,"draft_path_D2H_bytes_per_round":B*4,
            "probability_prefix_positions":self.head.probability_positions,"proposal_q_workspace_bytes":self.head.proposal_q.as_ref().map_or(0,|b|b.len_bytes()),
            "proposal_q_ids_D2H_bytes_per_round":if self.head.probability_positions>0{1536}else{0},
            "context_projection_policy":if self.reuse_context {"sealed_owner_retained_rows_epoch_reset_full_refresh"} else {"full_refresh_no_incremental_reuse_yet"},
            "target_verifier":if self.has_batched_verifier(){format!("B{target_width}_recorded_projection_prefix_NOT_B1_equivalence")}else{"canonical_serial_B1_NOT_speculative_B8".to_owned()},
            "target_execution_profile":self.target_execution.label(),
            "resident_projection_policy":if self.llama_projection {"llama_q8_mmvq_mmq_f16_shared_workspace"} else {"context_rows_block_rows_q8_tile4"},
            "block_projection_kernel":self.draft_block_projection_kernel(),
            "projection_numerics_changed":self.llama_projection || self.llama_head,
            "selector_gate_precision":self.weights.selector_gate_precision(),
            "selector_gate_f32_sha256":self.weights.selector_gate_f32_sha256(),
            "selector_codebooks_bf16_sha256":self.weights.selector_codebooks_bf16_sha256(),
            "selector_transition_scale":self.head.selector_scale,"selector_path":self.head.selector_path,
            "selector_early_position_scales":(self.head.selector_path=="position").then_some(self.head.selector_early_scales),
            "shared_head_topk":self.head_topk_policy(),
            "shared_head_projection":if self.llama_head {"llama_b8_bf16_f32_accum"} else {"canonical_scalar_bf16"},
            "projection_scratch_bytes":self.resident.as_ref().and_then(|s|s.draft.as_ref()).map_or(0,|d|d.llama_projection_workspace_bytes()),
            "native_HTTP_product_qualified":false,"speedup_claim":false})
    }
}
