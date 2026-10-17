//! Optional resident post-norm head/top16/Q8 selector; still a single draft block.
use super::*;
use crate::{Qwen35RealTargetOutputHeadCudaConfig, Qwen35RealTargetOutputHeadCudaSession};
use augur_backend::{
    CudaBufferView, CudaQwen35DflashQ8SelectorArgsV1, CudaQwen35DflashQ8SelectorViewsV1,
    CudaQwen35VocabTopkArgsV1,
};
const V: usize = 248320;

pub(super) struct ResidentCandidates {
    head: Qwen35RealTargetOutputHeadCudaSession,
    ids: OwnedCudaBuffer,
    scores: OwnedCudaBuffer,
    transitions: OwnedCudaBuffer,
    path: OwnedCudaBuffer,
}

#[derive(Debug)]
pub struct DFlashQ8CudaCandidateDiagnostics {
    pub candidate_ids: Vec<u32>,
    pub unary_scores: Vec<f32>,
    pub transition_scores: Vec<f32>,
    /// Anchor followed by seven draft tokens, NOT target-accepted tokens.
    pub path_ids: Vec<u32>,
    /// Full slab is a terminal audit only. It never travels to CPU during submit.
    pub logits: Vec<f32>,
    pub d2h_bytes: usize,
}

impl DFlashQ8CudaBlock {
    /// Diagnostic-only actual input bytes, including reused rows. This read is
    /// deliberately outside target verification's enqueue-only boundary.
    pub(crate) fn trajectory_input_features(&self) -> Result<(usize, usize, Vec<u8>)> {
        if !self.submitted || self.poisoned { return Err("trajectory draft input unavailable".into()); }
        let bytes = self.context.checked_mul(5*H*4).ok_or("trajectory feature extent")?;
        let mut host = vec![0;bytes];
        self.buffers["features"].view(0,bytes).map_err(err)?
            .copy_to_host_sync_with_submission_proof(&self.stream,&mut host,
                self.stream.submission_receipt().map_err(err)?).map_err(err)?;
        Ok((self.start,self.context,host))
    }
    pub(crate) fn trajectory_artifact_sha256(&self) -> &str { self.weights.file_sha256() }
    /// Test-only selector input: last committed target row after fc + hidden_norm.
    /// Read before verification; the pending anchor is deliberately excluded.
    #[cfg(test)]
    pub(crate) fn selector_consumed_target_tail(&self) -> Result<(usize, usize, u64, Vec<u8>)> {
        self.validate_submitted()?;
        if self.context == 0 { return Err("empty selector target context".into()); }
        let end = self.start.checked_add(self.context).ok_or("target tail extent")?;
        let source = self.view("target", (self.context - 1) * H, H)?.buffer_view();
        let mut bytes = vec![0u8; H * 4];
        source.copy_to_host_sync_with_submission_proof(&self.stream, &mut bytes,
            self.stream.submission_receipt().map_err(err)?).map_err(err)?;
        if bytes.chunks_exact(4).any(|v| !f32::from_le_bytes(v.try_into().unwrap()).is_finite()) {
            return Err("nonfinite consumed target projection".into());
        }
        Ok((self.start, end, self.request_generation, bytes))
    }
    /// Identity of this submitted draft, checked by the same-owner target verifier.
    pub fn submitted_identity(&self) -> Result<(u64, usize, u32, [u8; 32])> {
        if !self.submitted || self.poisoned {
            return Err("draft has not submitted successfully".into());
        }
        Ok((
            self.request_generation,
            self.start + self.context,
            self.anchor_token,
            self.feature_identity.payload_sha256()?,
        ))
    }
    fn validate_submitted(&self) -> Result<()> {
        if !self.submitted || self.poisoned { return Err("draft has not submitted successfully".into()); }
        Ok(())
    }
    /// Already DFlash-normalized; retain this owner through consumer completion.
    pub fn final_hidden_row_device(&self, row: usize) -> Result<CudaF32View> {
        self.validate_submitted()?;
        if row >= B {
            return Err("draft hidden row out of bounds".into());
        }
        self.view("final", row * H, H)
    }
    /// Contiguous, already normalized draft rows. The consumer must retain this
    /// owner and establish the same exact producer-event dependency as row views.
    pub(crate) fn final_hidden_rows_device(&self) -> Result<CudaF32View> {
        self.validate_submitted()?;
        self.view("final", 0, B * H)
    }
    /// Existing F32 gate and Q8 codebooks, for a target-owned shared-head chain.
    /// Caller must establish a producer event dependency and retain this owner.
    pub(crate) fn selector_after_think_gates(&self)->Result<Option<CudaBufferView>> {
        self.validate_submitted()?;
        if self.selector_phase_gate {Ok(Some(self.view("projected",0,B*256)?.buffer_view()))} else {Ok(None)}
    }
    pub(crate) fn selector_uses_original_codebooks(&self)->bool {self.weights.selector_codebooks_bf16_sha256().is_some()}
    pub fn selector_inputs_device(
        &self,
    ) -> Result<(CudaBufferView, CudaBufferView, CudaBufferView)> {
        self.validate_submitted()?;
        if let Some(owner)=self.weights.get(selector_gate::CODEBOOK_KEY) {
            return Ok((self.view("selector",0,B*256)?.buffer_view(),
                owner.view(0,selector_gate::CODEBOOK_BYTES).map_err(err)?,
                owner.view(selector_gate::CODEBOOK_BYTES,selector_gate::CODEBOOK_BYTES).map_err(err)?));
        }
        let view = |name: &str| -> Result<CudaBufferView> {
            let b = self.weights.get(name).ok_or("missing Q8 codebook")?;
            b.view(0, b.len_bytes()).map_err(err)
        };
        Ok((
            self.view("selector", 0, B * 256)?.buffer_view(),
            view("candidate_selector.predecessor_codebook")?,
            view("candidate_selector.successor_codebook")?,
        ))
    }
    /// Allocate an opt-in candidate head before context/block execution. This
    /// diagnostic owner loads one real target head; a full target session must
    /// share its own resident head rather than duplicate that weight allocation.
    pub fn enable_candidate_path(
        &mut self,
        config: &Qwen35RealTargetOutputHeadCudaConfig,
    ) -> Result<()> {
        if self.prepared || self.submitted || self.poisoned || self.candidates.is_some() {
            return Err("candidate owners must be installed once before execution".into());
        }
        let head =
            Qwen35RealTargetOutputHeadCudaSession::open(config, &self.stream, B).map_err(err)?;
        let device = self.stream.device_ordinal();
        self.candidates = Some(ResidentCandidates {
            head,
            ids: OwnedCudaBuffer::allocate(device, B * 16 * 4).map_err(err)?,
            scores: OwnedCudaBuffer::allocate(device, B * 16 * 4).map_err(err)?,
            transitions: OwnedCudaBuffer::allocate(device, B * 16 * 4).map_err(err)?,
            path: OwnedCudaBuffer::allocate(device, B * 4).map_err(err)?,
        });
        Ok(())
    }
    /// All consumers must use this stream (or establish an explicit dependency)
    /// and retain this owner until completion. No cross-stream handoff is implied.
    pub fn stream(&self) -> &OwnedCudaStream {
        &self.stream
    }
    /// Device draft-path handoff. UINT32_MAX denotes bad runtime data; the
    /// downstream verifier must reject it. Merely obtaining a view grants no
    /// acceptance, rollback, cache commit, or token-publication authority.
    pub fn candidate_path_device(&self) -> Result<CudaBufferView> {
        let c = self.ready_candidates()?;
        c.path.view(0, B * 4).map_err(err)
    }
    fn ready_candidates(&self) -> Result<&ResidentCandidates> {
        if !self.submitted || self.poisoned {
            return Err("candidate output unavailable".into());
        }
        self.candidates
            .as_ref()
            .ok_or_else(|| "candidate head not enabled".into())
    }
    pub fn readback_candidate_diagnostics(&self) -> Result<DFlashQ8CudaCandidateDiagnostics> {
        let c = self.ready_candidates()?;
        let read = |o: &OwnedCudaBuffer| -> Result<Vec<u8>> {
            let mut b = vec![0u8; o.len_bytes()];
            o.copy_to_host_sync(&self.stream, &mut b).map_err(err)?;
            Ok(b)
        };
        let ids = |b: Vec<u8>| {
            b.chunks_exact(4)
                .map(|x| u32::from_le_bytes(x.try_into().unwrap()))
                .collect()
        };
        let floats = |b: Vec<u8>| {
            b.chunks_exact(4)
                .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                .collect()
        };
        let mut logits = vec![0u8; B * V * 4];
        c.head
            .copy_logits_slab_to_host(&self.stream, &mut logits)
            .map_err(err)?;
        Ok(DFlashQ8CudaCandidateDiagnostics {
            candidate_ids: ids(read(&c.ids)?),
            unary_scores: floats(read(&c.scores)?),
            transition_scores: floats(read(&c.transitions)?),
            path_ids: ids(read(&c.path)?),
            logits: floats(logits),
            d2h_bytes: B * V * 4 + 3 * B * 16 * 4 + B * 4,
        })
    }
}
impl ResidentCandidates {
    pub(super) fn submit(&self, block: &DFlashQ8CudaBlock) -> Result<()> {
        let stream = &block.stream;
        for row in 0..B {
            // DFlash's final norm has already executed: never target-normalize again.
            self.head
                .submit_normalized_hidden_device(stream, block.view("final", row * H, H)?, row)
                .map_err(err)?;
            CudaQwen35VocabTopkArgsV1::bind(
                stream,
                self.head.logits_view(row).map_err(err)?,
                self.ids.view(row * 16 * 4, 16 * 4).map_err(err)?,
                self.scores.view(row * 16 * 4, 16 * 4).map_err(err)?,
                1,
                16,
            )
            .map_err(err)?
            .submit(stream)
            .map_err(err)?;
        }
        let codebook = |name: &str| -> Result<CudaBufferView> {
            let owner = block
                .weights
                .get(name)
                .ok_or("missing Q8 selector codebook")?;
            owner.view(0, owner.len_bytes()).map_err(err)
        };
        CudaQwen35DflashQ8SelectorArgsV1::bind(
            stream,
            CudaQwen35DflashQ8SelectorViewsV1 {
                predecessor: codebook("candidate_selector.predecessor_codebook")?,
                successor: codebook("candidate_selector.successor_codebook")?,
                candidate_ids: self.ids.view(0, B * 16 * 4).map_err(err)?,
                unary_scores: self.scores.view(0, B * 16 * 4).map_err(err)?,
                gates: block.view("selector", 0, B * 256)?.buffer_view(),
                transition_scores: self.transitions.view(0, B * 16 * 4).map_err(err)?,
                path_ids: self.path.view(0, B * 4).map_err(err)?,
            },
            block.anchor_token,
        )
        .map_err(err)?
        .submit(stream)
        .map_err(err)
    }
}
