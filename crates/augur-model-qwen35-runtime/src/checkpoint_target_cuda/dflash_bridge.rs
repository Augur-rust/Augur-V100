//! Explicit Augur -> Q8 DFlash correctness bridge. Not an HTTP default.
//!
//! Uses authentic captured features, the target's BF16 embeddings and the
//! SAME retained BF16 vocabulary head. Draft final hidden is already normalized;
//! there is no second target RMSNorm and no legacy GGUF head. Serial canonical
//! verification never writes a rejected candidate into target state. This first
//! diagnostic lane does full feature/logit D2H and does NOT claim fast drafting,
//! batched verification, committed session publication or product wall benefit.
use super::*;
use crate::checkpoint_projection_cuda::CheckpointScalarCudaProjection;
use crate::resident_growing_history::{GrowingHistory, GrowingHistoryConfig};
use crate::{
    DFlashDraftBlockInput, DFlashQ8CudaBlock, DFlashQ8CudaWeights,
    DFlashTargetFeatureCaptureReceipt, DFLASH_SEED_MASK_TOKEN,
};
use augur_backend::{
    CudaF32View, CudaQwen35DflashQ8SelectorArgsV1, CudaQwen35DflashQ8SelectorViewsV1,
    CudaQwen35VocabTopkArgsV1,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
const B: usize = 8;
const H: usize = 5120;
const V: usize = 248320;
const K: usize = 16;
#[cfg(feature="qwen35-augur-http")]
mod batched;
#[cfg(feature="qwen35-augur-http")]
pub use batched::BatchedVerifiedRound;
#[cfg(feature="qwen35-augur-http")]
mod rejection_trace;
mod resident;
pub use resident::AugurResidentDraftProposal;
/// Drain and summarize the opt-in proposal component bill recorded on this thread (None if disabled).
pub(crate) fn proposal_bill_summary()->Option<serde_json::Value> {
    let rows=resident::PROPOSAL_BILL.with(|c|std::mem::take(&mut *c.borrow_mut()));
    if rows.is_empty() {return None;}
    let n=rows.len() as f64;let mean=|f:fn(&resident::ProposalBillRow)->f64|rows.iter().map(f).sum::<f64>()/n;
    Some(serde_json::json!({"rounds":rows.len(),
        "context_gpu_ms":mean(|r|r.context_gpu_ms),"draft_layers_gpu_ms":mean(|r|r.draft_layers_gpu_ms),
        "host_submit_ms":mean(|r|r.host_submit_ms),"host_head_enqueue_ms":mean(|r|r.host_head_enqueue_ms),
        "host_total_through_path_ms":mean(|r|r.host_total_through_path_ms),"host_total_ms":mean(|r|r.host_total_ms),
        "diagnostic_timing":true,"HTTP_speedup_claim":false}))
}
mod target_execution;
use target_execution::TargetExecution;


struct SharedHead {
    stream: Rc<OwnedCudaStream>,
    normalized: Rc<OwnedCudaBuffer>,
    logits: Rc<OwnedCudaBuffer>,
    projections: Vec<CheckpointScalarCudaProjection>,
    #[cfg(feature="qwen35-prefill-cublas")]
    batched: Option<crate::checkpoint_projection_cuda::CheckpointSharedBf16HeadRows>,
    topk_batched: Option<CudaQwen35VocabTopkArgsV1>,
    _topk_tile_scratch: Option<OwnedCudaBuffer>,
    ids: OwnedCudaBuffer,
    scores: OwnedCudaBuffer,
    transitions: OwnedCudaBuffer,
    path: OwnedCudaBuffer,
    proposal_q: Option<OwnedCudaBuffer>,
    probability_positions: usize,
    probability_calibration: crate::augur_speculative_sampling::ProposalCalibration,
    selector_scale: f32,
    selector_path: &'static str,
    selector_early_scales: [f32; 3],
    #[cfg(test)]
    fail_after_copy: bool,
}
impl SharedHead {
    fn new(target: &CheckpointCudaTarget) -> Result<Self> {
        let stream = Rc::clone(&target.stream);
        let normalized = empty(&stream, B * H)?;
        let logits = empty(&stream, B * V)?;
        let mut projections = Vec::with_capacity(B);
        for row in 0..B {
            projections.push(target.head.bind_shared_normalized_bf16(
                &stream,
                Rc::clone(&normalized),
                row * H * 4,
                Rc::clone(&logits),
                row * V * 4,
            )?);
        }
        let alloc = |bytes| OwnedCudaBuffer::allocate(stream.device_ordinal(), bytes);
        let ids = alloc(B * K * 4)?;
        let scores = alloc(B * K * 4)?;
        let transitions = alloc(B * K * 4)?;
        let path = alloc(B * 4)?;
        // Startup-fixed scheduling only: original logits, top16 order and scores.
        let mut topk_batched = match std::env::var("AUGUR_DFLASH_TOPK") {
            Err(std::env::VarError::NotPresent) => None,
            Ok(value) if value == "scalar" => None,
            Ok(value) if value == "batch8" => Some(CudaQwen35VocabTopkArgsV1::bind(
                &stream,
                CudaF32View::new(logits.view(0, B * V * 4)?,
                    PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1, B * V)?,
                ids.view(0, B * K * 4)?, scores.view(0, B * K * 4)?, B, K,
            )?),
            _ => return Err("AUGUR_DFLASH_TOPK must be scalar or batch8".into()),
        };
        let tiled=match std::env::var("AUGUR_DFLASH_TOPK_TILED") {
            Err(std::env::VarError::NotPresent)=>false,
            Ok(v) if v=="0"=>false,
            Ok(v) if v=="1"=>true,
            _=>return Err("AUGUR_DFLASH_TOPK_TILED requires 0 or 1".into()),
        };
        let topk_tile_scratch=if tiled {
            let size=B*32*K*8;
            let scratch=alloc(size)?;
            topk_batched=Some(topk_batched.ok_or("tiled draft topk requires batch8")?
                .with_tiled_scratch(scratch.view(0,size)?,32)?);
            Some(scratch)
        } else {None};
        Ok(Self {
            stream,
            normalized,
            logits,
            projections,
            #[cfg(feature="qwen35-prefill-cublas")]
            batched: None,
            topk_batched,
            _topk_tile_scratch:topk_tile_scratch,
            ids,
            scores,
            transitions,
            path,
            proposal_q: None,
            probability_positions: 0,
            probability_calibration: crate::augur_speculative_sampling::ProposalCalibration::DEFAULT,
            selector_scale: 1.0,
            selector_path: "greedy",
            selector_early_scales: [1.0; 3],
            #[cfg(test)]
            fail_after_copy: false,
        })
    }
    fn project(
        &self,
        draft: &DFlashQ8CudaBlock,
        seed: &DFlashDraftBlockInput,
    ) -> Result<HeadDiagnostics> {
        if draft.submitted_identity()?
            != (
                seed.request_generation,
                seed.absolute_position_start,
                seed.anchor_token,
                seed.target_feature_sha256,
            )
        {
            return Err("foreign/stale draft hidden identity".into());
        }
        if self.probability_positions>0 {return Err("probability proposals require resident batched verification".into());}
        self.submit_hidden(draft, seed.anchor_token, seed.absolute_position_start)?;
        self.read_diagnostics(seed.anchor_token)
    }
    fn submit_hidden(&self, draft: &DFlashQ8CudaBlock, anchor: u32, absolute_start: usize) -> Result<()> {
        // Explicit producer completion, then target-stream ordered consumers.
        // The borrowed draft stays alive until the final consumer sync/readback.
        draft.synchronize()?;
        // Drains target consumers before the borrowed draft can be released,
        // including an error after only the first cross-stream D2D enqueue.
        let _consumer_drain = ConsumerDrain(&self.stream);
        let hidden = draft.final_hidden_rows_device()?;
        unsafe {
            self.normalized.copy_from_device_async_unchecked(
                &self.stream,
                0,
                hidden.buffer_view(),
            )?;
        }
        #[cfg(test)]
        if self.fail_after_copy {
            return Err("injected shared-head failure after D2D".into());
        }
        #[allow(unused_mut)]
        let mut scalar = true;
        #[cfg(feature="qwen35-prefill-cublas")]
        if let Some(head) = &self.batched { head.submit()?; scalar = false; }
        for (row, projection) in self.projections.iter().enumerate() {
            if scalar { projection.submit_for_plan()?; } // NO second target norm.
            if self.topk_batched.is_none() {
            CudaQwen35VocabTopkArgsV1::bind(
                &self.stream,
                CudaF32View::new(
                    self.logits.view(row * V * 4, V * 4)?,
                    PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                    V,
                )?,
                self.ids.view(row * K * 4, K * 4)?,
                self.scores.view(row * K * 4, K * 4)?,
                1,
                K,
            )?
            .submit(&self.stream)?;
            }
        }
        if let Some(topk) = &self.topk_batched { topk.submit(&self.stream)?; }
        let (gates, predecessor, successor) = draft.selector_inputs_device()?;
        if self.probability_positions>0 && (self.selector_path!="greedy" || self.selector_scale!=1.0
            || draft.selector_uses_original_codebooks() || draft.selector_after_think_gates()?.is_some()) {
            return Err("probability prefix requires plain Q8 selector, global gate and scale1".into());
        }
        if let Some(after)=draft.selector_after_think_gates()? {
            if self.selector_path!="greedy" || self.selector_scale!=1.0 || draft.selector_uses_original_codebooks() {
                return Err("phase selector requires greedy scale=1 and Q8 codebooks".into());
            }
            augur_backend::CudaQwen35DflashQ8SelectorPhaseArgsV1::bind(&self.stream,
                CudaQwen35DflashQ8SelectorViewsV1{predecessor,successor,gates,
                    candidate_ids:self.ids.view(0,B*K*4)?,unary_scores:self.scores.view(0,B*K*4)?,
                    transition_scores:self.transitions.view(0,B*K*4)?,path_ids:self.path.view(0,B*4)?},
                anchor,after,248069)?.submit(&self.stream)?;
            return Ok(());
        }
        if draft.selector_uses_original_codebooks() {
            if self.selector_path!="greedy" || self.selector_scale!=1.0 {return Err("original codebooks require greedy scale=1".into());}
            augur_backend::CudaQwen35DflashBf16CodebooksArgsV1::bind(&self.stream,
                CudaQwen35DflashQ8SelectorViewsV1{predecessor,successor,gates,
                    candidate_ids:self.ids.view(0,B*K*4)?,unary_scores:self.scores.view(0,B*K*4)?,
                    transition_scores:self.transitions.view(0,B*K*4)?,path_ids:self.path.view(0,B*4)?},anchor)?.submit(&self.stream)?;
            return Ok(());
        }
        let bound = CudaQwen35DflashQ8SelectorArgsV1::bind(
            &self.stream,
            CudaQwen35DflashQ8SelectorViewsV1 {
                predecessor,
                successor,
                candidate_ids: self.ids.view(0, B * K * 4)?,
                unary_scores: self.scores.view(0, B * K * 4)?,
                gates,
                transition_scores: self.transitions.view(0, B * K * 4)?,
                path_ids: self.path.view(0, B * 4)?,
            },
            anchor,
        )?;
        if let Some(q)=&self.proposal_q {
            let c=self.probability_calibration;
            if c==crate::augur_speculative_sampling::ProposalCalibration::DEFAULT {
                bound.submit_probability_prefix(&self.stream,q.view(0,1024)?,absolute_start as u64,self.probability_positions as u32)?;
            } else {
                bound.submit_probability_positions(&self.stream,q.view(0,1024)?,absolute_start as u64,self.probability_positions as u32,c.temperatures,c.pair_scales)?;
            }
        } else if self.selector_path == "dp" {
            bound.submit_with_transition_dp(&self.stream, self.selector_scale)?;
        } else if self.selector_path == "tail075" {
            bound.submit_with_tail_transition_scale(&self.stream, self.selector_scale)?;
        } else if self.selector_path == "rankcomp" {
            bound.submit_with_rank_compensation(&self.stream, self.selector_scale)?;
        } else if self.selector_path == "tailrankcomp" {
            bound.submit_with_tail_rank_compensation(&self.stream, 0.75, self.selector_scale)?;
        } else if self.selector_path == "position" {
            bound.submit_with_early_position_scales(&self.stream, self.selector_early_scales)?;
        } else if self.selector_path == "early" {
            bound.submit_with_early_transition_scale(&self.stream, self.selector_scale)?;
        } else {
            bound.submit_with_transition_scale(&self.stream, self.selector_scale)?;
        }
        Ok(())
    }
    fn read_probability(&self,start:usize,path:&[u32])->Result<Option<crate::augur_speculative_sampling::Proposal>> {
        let Some(q)=&self.proposal_q else{return Ok(None)};
        let mut bytes=vec![0u8;1024];q.copy_to_host_sync(&self.stream,&mut bytes)?;
        let q=bytes.chunks_exact(8).map(|b|f64::from_le_bytes(b.try_into().unwrap())).collect();
        let mut bytes=vec![0u8;512];self.ids.copy_to_host_sync(&self.stream,&mut bytes)?;
        let ids=bytes.chunks_exact(4).map(|b|u32::from_le_bytes(b.try_into().unwrap())).collect();
        let p=crate::augur_speculative_sampling::Proposal{ids,q,start,positions:self.probability_positions,calibration:self.probability_calibration};
        p.validate(path)?;Ok(Some(p))
    }
    fn read_path(&self, anchor: u32) -> Result<Vec<u32>> {
        let mut path_bytes = vec![0; B * 4];
        self.path.copy_to_host_sync(&self.stream, &mut path_bytes)?;
        let path: Vec<u32> = path_bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        if path[0] != anchor || path.iter().any(|id| *id as usize >= V) {
            return Err("invalid/nonfinite Q8 selector path".into());
        }
        Ok(path)
    }
    fn read_diagnostics(&self, anchor: u32) -> Result<HeadDiagnostics> { self.read_diagnostics_mode(anchor,true) }
    fn read_candidate_diagnostics(&self,anchor:u32)->Result<HeadDiagnostics>{self.read_diagnostics_mode(anchor,false)}
    fn read_diagnostics_mode(&self,anchor:u32,full:bool)->Result<HeadDiagnostics>{
        let path = self.read_path(anchor)?;
        let mut id_bytes = vec![0; B * K * 4];
        self.ids.copy_to_host_sync(&self.stream, &mut id_bytes)?;
        let ids = id_bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let logits = if full {read_f32(&self.stream, &self.logits)?}else{Vec::new()};
        let normalized = if full {read_f32(&self.stream, &self.normalized)?}else{Vec::new()};
        let scores = read_f32(&self.stream, &self.scores)?;
        let transitions = read_f32(&self.stream, &self.transitions)?;
        if logits
            .iter()
            .chain(&normalized)
            .chain(&scores)
            .chain(&transitions)
            .any(|v| !v.is_finite())
        {
            return Err("nonfinite complete shared BF16 head diagnostics".into());
        }
        Ok(HeadDiagnostics {
            path,
            candidate_ids: ids,
            unary_scores: scores,
            transition_scores: transitions,
            logits,
            normalized_hidden: normalized,
        })
    }
    fn bytes(&self) -> usize {
        B * (H + V) * 4 + 3 * B * K * 4 + B * 4
            + self.proposal_q.as_ref().map_or(0,|q|q.len_bytes())
    }
}
impl Drop for SharedHead {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
    }
}

#[derive(Debug)]
pub struct HeadDiagnostics {
    pub path: Vec<u32>,
    pub candidate_ids: Vec<u32>,
    pub unary_scores: Vec<f32>,
    pub transition_scores: Vec<f32>,
    pub logits: Vec<f32>,
    pub normalized_hidden: Vec<f32>,
}
/// Opaque, move-only proposal bound to this target incarnation/epoch/cursor.
/// Reading diagnostics does not grant token publication or session commit.
pub struct AugurDraftProposal {
    bridge_instance: Rc<()>,
    bridge_generation: u64,
    capture_identity: (u64, u64, usize),
    target_stream: Rc<OwnedCudaStream>,
    capture: DFlashTargetFeatureCaptureReceipt,
    head: HeadDiagnostics,
    diagnostic: Value,
}
impl AugurDraftProposal {
    pub fn capture(&self) -> &DFlashTargetFeatureCaptureReceipt {
        &self.capture
    }
    pub fn head(&self) -> &HeadDiagnostics {
        &self.head
    }
    pub fn diagnostic(&self) -> &Value {
        &self.diagnostic
    }
}
struct ConsumerDrain<'a>(&'a OwnedCudaStream);
impl Drop for ConsumerDrain<'_> {
    fn drop(&mut self) {
        let _ = self.0.synchronize();
    }
}
#[derive(Debug)]
pub struct SerialVerifiedRound {
    pub canonical_output_tokens: Vec<u32>,
    pub first_rejection: Option<Value>,
    pub stop: &'static str,
    pub start_consumed: usize,
    pub end_consumed: usize,
    pub serial_target_verification_s: f64,
}
/// Shared Q8 weight lease/head. The first bridge is intentionally diagnostic:
/// full genuine feature SHA/readback rather than a fabricated payload hash.
pub struct AugurDflashBridge {
    instance: Rc<()>,
    stream: Rc<OwnedCudaStream>,
    weights: DFlashQ8CudaWeights,
    head: SharedHead,
    generation: u64,
    live_proposal: Option<u64>,
    poisoned: bool,
    resident: Option<resident::ResidentDraftState>,
    target_execution: TargetExecution,
    reuse_context: bool,
    llama_projection: bool,
    llama_head: bool,
    selector_gate_enabled: bool,
    #[cfg(feature="qwen35-augur-http")]
    http_english_gate: bool,
    #[cfg(feature="qwen35-augur-http")]
    http_global_gate: bool,
    #[cfg(feature="qwen35-augur-http")]
    batched: Option<batched::Executor>,
    #[cfg(feature="qwen35-augur-http")]
    rejection_trace: Option<rejection_trace::Trace>,
}
impl AugurDflashBridge {
    /// Actual startup-fixed head schedule; no process-environment reread.
    pub fn head_topk_policy(&self) -> &'static str {
        if self.head.topk_batched.is_some() { "batch8" } else { "scalar" }
    }
    /// Kernel label of the draft-side shared head; the B8 verifier binds the same env choice.
    pub fn shared_head_kernel(&self) -> &'static str {
        #[cfg(feature="qwen35-prefill-cublas")]
        if let Some(h)=&self.head.batched {return h.kernel_label();}
        "bf16_f32_matvec_v1"
    }
    pub fn verify_head_kernel(&self) -> Option<&'static str> {
        self.batched.as_ref().map(|b|b.head_kernel_label())
    }
    pub fn gdn_input_kernel(&self) -> Option<&'static str> {
        self.batched.as_ref().map(|b|b.gdn_input_kernel_label())
    }

    pub fn new(target: &CheckpointCudaTarget, weights: &DFlashQ8CudaWeights) -> Result<Self> {
        if target.poisoned
            || target.dflash_capture.is_none()
            || target.info["profile"] != "quasar_nvfp4"
            || weights.device_ordinal() != target.stream.device_ordinal()
        {
            return Err(
                "draft bridge requires the authenticated capture-enabled Augur owner".into(),
            );
        }
        weights.config().validate_exact_profile()?;
        let target_execution = TargetExecution::for_target(target)?;
        let reuse_context = match std::env::var("AUGUR_DFLASH_CONTEXT").as_deref() {
            Err(std::env::VarError::NotPresent) | Ok("full_refresh") => false,
            Ok("retained") if target_execution == TargetExecution::AugurRotatedQ8 => true,
            _ => return Err("DFlash context policy must be full_refresh, or retained on the current target".into()),
        };
        let llama_projection = match std::env::var("AUGUR_DFLASH_PROJECTION").as_deref() {
            Err(std::env::VarError::NotPresent) | Ok("f32_tile4") => false,
            Ok("llama") if target_execution == TargetExecution::AugurRotatedQ8 => true,
            _ => return Err("DFlash projection must be f32_tile4, or llama on the current target".into()),
        };
        let llama_head = match std::env::var("AUGUR_DFLASH_HEAD").as_deref() {
            Err(std::env::VarError::NotPresent) | Ok("scalar_bf16") => false,
            Ok("llama_b8") if target_execution == TargetExecution::AugurRotatedQ8 => true,
            _ => return Err("DFlash head must be scalar_bf16, or llama_b8 on the current target".into()),
        };
        #[allow(unused_mut)]
        let selector_scale = match std::env::var("AUGUR_DFLASH_SELECTOR_SCALE") {
            Ok(value) => parse_selector_transition_scale(Some(&value))?,
            Err(std::env::VarError::NotPresent) => 1.0,
            Err(_) => return Err("non-Unicode DFlash selector scale".into()),
        };
        let selector_path = match std::env::var("AUGUR_DFLASH_SELECTOR_PATH").as_deref() {
            Err(std::env::VarError::NotPresent) | Ok("greedy") => "greedy",
            Ok("dp") if target_execution == TargetExecution::AugurRotatedQ8 => "dp",
            Ok("tail075") if target_execution == TargetExecution::AugurRotatedQ8 => "tail075",
            Ok("rankcomp") if target_execution == TargetExecution::AugurRotatedQ8 => "rankcomp",
            Ok("tailrankcomp") if target_execution == TargetExecution::AugurRotatedQ8 => "tailrankcomp",
            Ok("early") if target_execution == TargetExecution::AugurRotatedQ8 => "early",
            Ok("position") if target_execution == TargetExecution::AugurRotatedQ8 => "position",
            Ok(_) => return Err("DFlash selector path must be greedy, dp, tail075, rankcomp, tailrankcomp, early, or position on the current target".into()),
            Err(_) => return Err("non-Unicode DFlash selector path".into()),
        };
        let early_scales_text = match std::env::var("AUGUR_DFLASH_SELECTOR_EARLY_SCALES") {
            Ok(value) => Some(value),
            Err(std::env::VarError::NotPresent) => None,
            Err(_) => return Err("non-Unicode DFlash selector early scales".into()),
        };
        let early_scales = selector_policy::parse_early_scales(
            selector_path, selector_scale, early_scales_text.as_deref())?;
        let probability_positions=crate::augur_speculative_sampling::positions_from_env()?;
        if probability_positions>0 && (crate::augur_sampling::Mode::from_env()?!=crate::augur_sampling::Mode::Thinking
            || selector_path!="greedy" || selector_scale!=1.0) {
            return Err("probability prefix requires sampled target and unscaled plain selector".into());
        }
        let mut head = SharedHead::new(target)?;
        head.probability_positions=probability_positions;
        head.probability_calibration=crate::augur_speculative_sampling::ProposalCalibration::from_env(probability_positions)?;
        if probability_positions>0 {head.proposal_q=Some(OwnedCudaBuffer::allocate(target.stream.device_ordinal(),1024)?);}
        head.selector_early_scales = early_scales;
        head.selector_scale = selector_scale;
        head.selector_path = selector_path;
        if llama_head {
            #[cfg(feature="qwen35-prefill-cublas")]
            { head.batched = Some(target.head.bind_shared_normalized_bf16_b8(&head.stream,
                Rc::clone(&head.normalized),Rc::clone(&head.logits))?); }
            #[cfg(not(feature="qwen35-prefill-cublas"))]
            return Err("llama B8 head requires prefill-cublas feature".into());
        }
        Ok(Self {
            instance: Rc::new(()),
            stream: Rc::clone(&target.stream),
            weights: weights.clone(),
            head,
            generation: 0,
            live_proposal: None,
            poisoned: false,
            resident: None,
            target_execution,
            reuse_context,
            llama_projection,
            llama_head,
            selector_gate_enabled: weights.selector_gate_f32_sha256().is_some(),
            #[cfg(feature="qwen35-augur-http")]
            http_english_gate: false,
            #[cfg(feature="qwen35-augur-http")]
            http_global_gate: false,
            #[cfg(feature="qwen35-augur-http")]
            batched: None,
            #[cfg(feature="qwen35-augur-http")]
            rejection_trace: rejection_trace::Trace::from_env()?,
        })
    }
    /// Fixed HTTP B8 policy: never silently inherit experimental env overrides.
    #[cfg(feature="qwen35-augur-http")]
    pub(crate) fn new_http_batched(target:&CheckpointCudaTarget,weights:&DFlashQ8CudaWeights)->Result<Self> {
        for (key,expected) in [("AUGUR_DFLASH_CONTEXT","retained"),("AUGUR_DFLASH_PROJECTION","llama"),
            ("AUGUR_DFLASH_HEAD","llama_b8"),("AUGUR_DFLASH_TARGET_PROJECTION","packed_b8")] {
            match std::env::var(key) {Ok(v) if v!=expected=>return Err(format!("{key} conflicts with HTTP B8 profile").into()),
                Err(std::env::VarError::NotUnicode(_))=>return Err("non-Unicode DFlash policy".into()),_=>{}}
        }
        let policy=std::env::var("AUGUR_DFLASH_ENGLISH_GATE");
        let policy=match &policy {Ok(value)=>Some(value.as_str()),Err(std::env::VarError::NotPresent)=>None,Err(e)=>return Err(e.to_string().into())};
        let policy=policy.or_else(||weights.embedded_selector_gate().then_some(crate::augur_session::selector::GLOBAL_POLICY));
        if weights.embedded_selector_gate() && policy!=Some(crate::augur_session::selector::GLOBAL_POLICY) {
            return Err("embedded FP32 gate0.45 requires global policy".into());
        }
        let english_gate=crate::augur_session::selector::admit(policy,weights.selector_gate_f32_sha256(),
            weights.selector_gate_precision(),weights.selector_codebooks_bf16_sha256().is_some())?;
        let mut b=Self::new(target,weights)?;
        let http_position = match std::env::var("AUGUR_DFLASH_HTTP_POSITION") {
            Ok(value) => Some(value),
            Err(std::env::VarError::NotPresent) => None,
            Err(_) => return Err("non-Unicode HTTP position opt-in".into()),
        };
        selector_policy::admit_http(b.head.selector_path, b.head.selector_scale,
            policy == Some(crate::augur_session::selector::GLOBAL_POLICY), http_position.as_deref())?;
        if b.head.probability_positions>0 && policy!=Some(crate::augur_session::selector::GLOBAL_POLICY) {
            return Err("probability HTTP requires the global FP32 gate policy".into());
        }
        b.reuse_context=true;b.llama_projection=true;b.llama_head=true;
        b.http_english_gate=english_gate;b.http_global_gate=policy==Some(crate::augur_session::selector::GLOBAL_POLICY);b.selector_gate_enabled=false;
        b.head.batched=Some(target.head.bind_shared_normalized_bf16_b8(&b.head.stream,Rc::clone(&b.head.normalized),Rc::clone(&b.head.logits))?);
        Ok(b)
    }
    #[cfg(feature="qwen35-augur-http")]
    pub(crate) fn http_english_gate_available(&self)->bool {self.http_english_gate}
    #[cfg(feature="qwen35-augur-http")]
    pub(crate) fn http_global_gate_available(&self)->bool {self.http_global_gate}
    #[cfg(feature="qwen35-augur-http")]
    pub(crate) fn select_http_english_gate(&mut self,enabled:bool)->Result<()> {
        if self.live_proposal.is_some() || self.poisoned || (enabled && !self.http_english_gate) {
            return Err("English selector switch requires admitted HTTP profile and no live proposal".into());
        }
        self.selector_gate_enabled=enabled;Ok(())
    }
    pub(crate) fn drain(&mut self)->Result<()> {
        if let Some(d)=self.resident.as_ref().and_then(|r|r.draft.as_ref()){d.synchronize()?;}
        self.head.stream.synchronize()?;self.stream.synchronize()?;Ok(())
    }
    pub fn resident_info(&self) -> Value {
        json!({"shared_head":"target_retained_original_BF16_no_second_norm",
            "selector_transition_scale":self.head.selector_scale,"selector_path":self.head.selector_path,
            "selector_early_position_scales":(self.head.selector_path=="position").then_some(self.head.selector_early_scales),
            "duplicate_head_weight_bytes":0,"head_workspace_bytes":self.head.bytes(),
            "q8_draft_sha256":self.weights.file_sha256(),"q8_draft_resident_bytes":self.weights.resident_bytes(),
            "diagnostic_only":true,"feature_payload_hash":"actual_F32_bytes_SHA256_D2H",
            "native_product_enabled":false,"batched_target_verifier":false,
            "target_execution_profile":self.target_execution.label()})
    }
    fn validate_target(&self, target: &CheckpointCudaTarget) -> Result<()> {
        if self.poisoned
            || target.poisoned
            || !Rc::ptr_eq(&self.stream, &target.stream)
            || target.dflash_capture.is_none()
            || TargetExecution::for_target(target)? != self.target_execution
        {
            return Err("poisoned or foreign QUASAR/DFlash owner".into());
        }
        Ok(())
    }
    pub fn propose_diagnostic(
        &mut self,
        target: &mut CheckpointCudaTarget,
    ) -> Result<AugurDraftProposal> {
        self.validate_target(target)?;
        target.validate_cursor()?;
        if self.live_proposal.is_some() {
            return Err("prior proposal must be verified or explicitly discarded".into());
        }
        if target
            .consumed
            .checked_add(B)
            .is_none_or(|end| end > target.capacity)
        {
            return Err("draft requires eight absolute query positions; final tail needs no-DFlash fallback".into());
        }
        let capture_identity = target
            .dflash_capture
            .as_ref()
            .unwrap()
            .committed_identity()?;
        if capture_identity.2 != target.consumed {
            return Err("target/capture cursor drift".into());
        }
        let generation = self
            .generation
            .checked_add(1)
            .ok_or("draft generation overflow")?;
        self.poisoned = true;
        let result = (|| -> Result<AugurDraftProposal> {
            let started = Instant::now();
            let anchor = target.select_next_token()?;
            let (snapshot, feature_view) = target
                .dflash_capture
                .as_ref()
                .unwrap()
                .snapshot_and_device_view()?;
            let feature_readback_sha_s = started.elapsed().as_secs_f64();
            let mut capture = snapshot.receipt;
            // Request/draft generation is independent of capture operations;
            // retain that original epoch/revision/cursor separately above.
            capture.request_generation = generation;
            let mut block_tokens = vec![DFLASH_SEED_MASK_TOKEN; B];
            block_tokens[0] = anchor;
            let seed = DFlashDraftBlockInput {
                request_generation: generation,
                absolute_position_start: target.consumed,
                anchor_token: anchor,
                block_tokens,
                mask_token_id: DFLASH_SEED_MASK_TOKEN,
                target_feature_sha256: capture.feature_sha256,
            };
            capture.validate_for_block(
                self.weights.config(),
                &seed,
                capture.feature_elements,
                capture.feature_sha256,
            )?;
            // Authenticated BF16 target embeddings, not old-target embeddings.
            // This diagnostic first constructor has an explicit host seed seam.
            let mut noise = Vec::with_capacity(B * H);
            for token in &seed.block_tokens {
                let mut bytes = vec![0; H * 2];
                target._artifact.read_tensor_range_into(
                    "model.language_model.embed_tokens.weight",
                    u64::from(*token) * (H * 2) as u64,
                    &mut bytes,
                )?;
                noise.extend(bytes.chunks_exact(2).map(|b| {
                    f32::from_bits(u32::from(u16::from_le_bytes(b.try_into().unwrap())) << 16)
                }));
            }
            let noise_sha: [u8; 32] = Sha256::digest(
                noise
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .into();
            let history = GrowingHistory::from_extent(
                GrowingHistoryConfig::default_2048(),
                capture.context_start_position,
                capture.context_len,
            )?;
            let setup = Instant::now();
            let mut draft = DFlashQ8CudaBlock::from_resident_growing_device_features(
                &self.weights,
                &seed,
                &capture,
                feature_view,
                &noise,
                history,
                2048,
                &self.stream,
            )?;
            draft.enable_context_rows_launch()?;
            draft.enable_block_rows_launch()?;
            draft.enable_q8_tile4()?;
            let draft_workspace = draft.workspace_bytes();
            let block_setup_s = setup.elapsed().as_secs_f64();
            let run = Instant::now();
            draft.prepare_context()?;
            draft.submit()?;
            draft.synchronize()?;
            let draft_prepare_and_submit_s = run.elapsed().as_secs_f64();
            let run = Instant::now();
            let head = self.head.project(&draft, &seed)?;
            let shared_head_and_diagnostics_s = run.elapsed().as_secs_f64();
            let draft_diagnostics = draft.readback_diagnostics()?;
            if draft_diagnostics.hidden != head.normalized_hidden {
                return Err("draft/shared-head normalized input copy mismatch".into());
            }
            let gates = draft_diagnostics.selector_hidden;
            let diagnostic = json!({"schema":"quasar-q8-dflash-diagnostic-proposal-v1",
                "q8_draft_sha256":self.weights.file_sha256(),"capture_owner_epoch":capture_identity.0,
                "capture_operation_generation":capture_identity.1,"absolute_start":seed.absolute_position_start,
                "anchor":anchor,"mask":DFLASH_SEED_MASK_TOKEN,"noise_sha256":noise_sha,
                "physical_draft_history_rows":2048,"active_feature_rows":capture.context_len,
                "draft_workspace_bytes":draft_workspace,"feature_readback_sha_s":feature_readback_sha_s,
                "block_setup_s":block_setup_s,"draft_prepare_and_submit_s":draft_prepare_and_submit_s,
                "shared_head_and_diagnostics_s":shared_head_and_diagnostics_s,
                "diagnostic_proposal_wall_s":started.elapsed().as_secs_f64(),
                "selector_gates":gates,"performance_qualified":false,
                "actual_feature_readback_bytes":snapshot.little_endian_f32.len(),
                "seed_embedding_host_upload_bytes":B*H*4,"duplicate_head_bytes":0});
            Ok(AugurDraftProposal {
                bridge_instance: Rc::clone(&self.instance),
                bridge_generation: generation,
                capture_identity,
                target_stream: Rc::clone(&self.stream),
                capture,
                head,
                diagnostic,
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
    /// Explicitly discard without altering target state; stale proposals cannot
    /// later be submitted because verification consumes this move-only value.
    pub fn discard(&mut self, proposal: AugurDraftProposal) -> Result<()> {
        if !Rc::ptr_eq(&self.instance, &proposal.bridge_instance)
            || self.live_proposal != Some(proposal.bridge_generation)
            || !Rc::ptr_eq(&self.stream, &proposal.target_stream)
        {
            return Err("foreign/stale discard".into());
        }
        self.live_proposal = None;
        Ok(())
    }
    /// Correctness verifier: no speculative GDN/KV writes or rollback. Outputs
    /// are canonical but are NOT a persisted Session/HTTP commit. Caller must
    /// run the existing receipt/commit/cancel publication contract separately.
    pub fn verify_serial(
        &mut self,
        target: &mut CheckpointCudaTarget,
        proposal: AugurDraftProposal,
        remaining_output_budget: usize,
        eos: &[u32],
    ) -> Result<SerialVerifiedRound> {
        self.verify_bound_path(target, &proposal.bridge_instance, proposal.bridge_generation, proposal.capture_identity,
            &proposal.target_stream, &proposal.head.path, Some(&proposal.head.candidate_ids),
            remaining_output_budget, eos)
    }
    fn verify_bound_path(
        &mut self, target: &mut CheckpointCudaTarget, instance: &Rc<()>, generation: u64,
        capture_identity: (u64, u64, usize), stream: &Rc<OwnedCudaStream>,
        path: &[u32], candidate_ids: Option<&[u32]>,
        remaining_output_budget: usize, eos: &[u32],
    ) -> Result<SerialVerifiedRound> {
        self.validate_target(target)?;
        if !Rc::ptr_eq(&self.instance, instance) || path.len() != B || path.iter().any(|id| *id as usize >= V)
            || candidate_ids.is_some_and(|ids| ids.len() != B * K)
            || remaining_output_budget == 0
            || eos.is_empty()
            || eos.iter().any(|id| *id as usize >= V)
            || self.live_proposal != Some(generation)
            || !Rc::ptr_eq(&self.stream, stream)
            || target
                .dflash_capture
                .as_ref()
                .unwrap()
                .committed_identity()?
                != capture_identity
            || target.consumed != capture_identity.2
        {
            return Err("stale proposal, cursor, stop policy or output budget".into());
        }
        self.poisoned = true;
        self.live_proposal = None;
        let started = Instant::now();
        let result = (|| -> Result<SerialVerifiedRound> {
            let start_consumed = target.consumed;
            let mut tokens = Vec::with_capacity(B);
            let mut rejection = None;
            let mut stop = "block_complete";
            for offset in 0..B {
                let correct = target.select_next_token()?;
                if offset == 0 && correct != path[0] {
                    return Err("selected anchor changed".into());
                }
                if eos.contains(&correct) {
                    stop = "eos_pending";
                    break;
                }
                if offset > 0 && correct != path[offset] {
                    let rank = candidate_ids.and_then(|ids|
                        ids[offset * K..(offset + 1) * K].iter().position(|v| *v == correct));
                    rejection = Some(
                        json!({"offset":offset,"absolute_position":start_consumed+offset,
                        "correct_target_token":correct,"proposed_token":path[offset],
                        "correct_top16_rank":rank,
                        "reason":if candidate_ids.is_none(){"not_collected_no_diagnostic_IO"}else if rank.is_none(){"top16_coverage_missing"}else{"selector_path_ranking"},
                        "canonical_labels_after_this_rejection":false}),
                    );
                    stop = "first_rejection_pending";
                    break;
                }
                tokens.push(correct);
                if tokens.len() == remaining_output_budget {
                    stop = "length_pending";
                    break;
                }
                self.target_execution.consume_selected(target)?;
            }
            target.stream.synchronize()?;
            Ok(SerialVerifiedRound {
                canonical_output_tokens: tokens,
                first_rejection: rejection,
                stop,
                start_consumed,
                end_consumed: target.consumed,
                serial_target_verification_s: started.elapsed().as_secs_f64(),
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
impl Drop for AugurDflashBridge {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod acceptance_tests;

#[cfg(all(test, feature = "qwen35-augur-http"))]
mod context_reuse_tests;

mod selector_policy;

// Read once when constructing the owner; never change proposal policy mid-session.
fn parse_selector_transition_scale(value: Option<&str>) -> Result<f32> {
    let scale = value.unwrap_or("1").parse::<f32>().map_err(|_| "invalid DFlash selector scale")?;
    if !scale.is_finite() || !(0.0..=2.0).contains(&scale) {
        return Err("DFlash selector scale must be finite and within [0,2]".into());
    }
    Ok(scale)
}

#[cfg(test)]
#[test]
fn selector_scale_default_and_rejection() {
    assert_eq!(parse_selector_transition_scale(None).unwrap(), 1.0);
    assert_eq!(parse_selector_transition_scale(Some("0.75")).unwrap(), 0.75);
    for value in ["NaN", "inf", "-inf", "-0.1", "2.01", "junk"] {
        assert!(parse_selector_transition_scale(Some(value)).is_err());
    }
}
