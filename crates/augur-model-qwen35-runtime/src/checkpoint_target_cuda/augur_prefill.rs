//! Speed-first Augur adaptation for all64 QUASAR layers; not HTTP admission.
//! Explicit GEMM/GDN superbatch plus literal tail; Flash Attention stays512-tiled. Host timing includes setup/upload/history/head/release.
//! Rotated cache bytes MUST NOT be fed to the old scalar decoder.
use super::*;
use crate::checkpoint_full_cuda::augur_prefill::{self as full, AttentionMode};
use crate::checkpoint_linear_cuda::augur_prefill as gdn;
use augur_backend::{CudaBf16TokenRowsGatherArgsV1, CudaF32View};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
const H: usize = 5120;
#[cfg(feature = "qwen35-augur-vision")]
pub(super) mod image_rows;
pub(crate) const PROFILE: &str = "quasar_superbatch_flash512_f16_f32_v2";
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Plan {
    schema: String,
    numerical_profile: String,
    kv_identity: String,
    checkpoint: serde_json::Value,
    first: usize,
    token_ids_sha256: String,
    widths: Vec<usize>,
}
fn ids_hash(tokens: &[u32]) -> String {
    let mut h = Sha256::new();
    for token in tokens {
        h.update(token.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}
fn validate_widths(widths: &[usize], tokens: usize) -> Result<usize> {
    let mut total = 0usize;
    let mut max = 0;
    if widths.is_empty() {
        return Err("empty global prefill plan".into());
    }
    for &n in widths {
        if !(1..=augur_backend::cuda_qwen35_prefill_cublas::AUGUR_MAX_BATCH).contains(&n) {
            return Err("global prefill exceeds compiled superbatch capacity".into());
        }
        total = total.checked_add(n).ok_or("global plan size overflow")?;
        max = max.max(n);
    }
    if total != tokens {
        return Err("global plan width sum mismatch".into());
    }
    Ok(max)
}
#[derive(Clone, Copy)]
pub(crate) struct Budget {
    pub owner_limit: u64,
    pub safety: u64,
    pub external_reserve: u64,
}
impl Default for Budget {
    fn default() -> Self {
        Self {
            owner_limit: u64::MAX,
            safety: 128 * 1024 * 1024,
            external_reserve: 0,
        }
    }
}
#[derive(Debug, Serialize)]
pub(crate) struct Receipt {
    pub plan: Plan,
    pub numerical_qualified: bool,
    pub http_qualified: bool,
    pub workspace_profile: &'static str,
    pub safety_bytes: u64,
    pub phase_owned_bytes: usize,
    pub resident_padded_gate_bytes: usize,
    pub padded_gates_reused: bool,
    #[cfg(feature = "qwen35-augur-vision")]
    pub vision_memory_budget: serde_json::Value,
    pub actual_free_before: u64,
    pub actual_free_after_release: u64,
    pub setup_seconds: f64,
    pub execute_with_upload_head_capture_seconds: f64,
    pub drain_release_seconds: f64,
    pub complete_wall_seconds: f64,
    pub capture_rows: usize,
    pub head_submissions: usize,
    pub hot_allocations: u64,
    pub hot_frees: u64,
    pub diagnostic_io: bool,
    #[cfg(feature = "qwen35-augur-vision")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_input: Option<serde_json::Value>,
}
#[derive(Clone, Copy)]
pub(crate) struct Progress {
    pub first: usize,
    pub rows: usize,
    pub layer: usize,
}
#[cfg(test)]
pub(super) type Diagnostic<'a> =
    dyn FnMut(Progress, &OwnedCudaStream, &OwnedCudaBuffer, &OwnedCudaBuffer, &gdn::Workspace) -> Result<()> + 'a;
enum Program {
    Gdn(gdn::Program),
    Full(full::Program),
}
struct Chunk {
    first: usize,
    rows: usize,
    gathers: Vec<CudaBf16TokenRowsGatherArgsV1>,
    programs: Vec<Program>,
}
#[cfg(test)]
pub(crate) fn default_required_free_bytes(rows:usize)->Result<u64> {
    default_required_free_bytes_mode(rows, AttentionMode::Legacy)
}
pub(crate) fn default_required_free_bytes_mode(rows:usize, mode:AttentionMode)->Result<u64> {
    let widths=augur_backend::cuda_qwen35_prefill_cublas::augur_prefill_widths(rows)?;
    Phase::required_free_bytes_mode(*widths.iter().max().ok_or("empty prefill memory plan")?,Budget::default(),mode)
}
struct Phase {
    workspace: Rc<full::Workspace>,
    hidden: [Rc<OwnedCudaBuffer>; 2],
    sources: BTreeMap<usize, Rc<gdn::Source>>,
    chunks: Vec<Chunk>,
}
impl Phase {
    #[cfg(test)]
    fn owned_recipe(width: usize) -> Result<usize> {
        Self::owned_recipe_mode(width, AttentionMode::Legacy)
    }
    fn owned_recipe_mode(width: usize, mode: AttentionMode) -> Result<usize> {
        Ok(full::Workspace::owned_recipe_mode(width, 262144, mode)?
            + 2 * width * H * 4
            + 48 * gdn::Source::PADDED_BYTES)
    }
    #[cfg(test)]
    fn owned_recipe_with_sources(width:usize,reused:bool)->Result<usize> {
        Self::owned_recipe_with_sources_mode(width,reused,AttentionMode::Legacy)
    }
    fn owned_recipe_with_sources_mode(width:usize,reused:bool,mode:AttentionMode)->Result<usize> {
        Self::owned_recipe_mode(width,mode)?.checked_sub(if reused {48*gdn::Source::PADDED_BYTES} else {0})
            .ok_or_else(||"padded source accounting overflow".into())
    }
    fn required_free_bytes_mode(width:usize,budget:Budget,mode:AttentionMode)->Result<u64> {
        (Self::owned_recipe_mode(width,mode)? as u64).checked_add(budget.safety)
            .and_then(|n|n.checked_add(budget.external_reserve))
            .and_then(|n|n.checked_add(gdn::Source::PADDED_BYTES as u64))
            .ok_or_else(||"Augur budget overflow".into())
    }
    fn bind(target: &CheckpointCudaTarget, plan: &Plan, budget: Budget) -> Result<Self> {
        let width = *plan.widths.iter().max().ok_or("empty augur plan")?;
        let mode = target.attention_mode();
        let s = &target.stream;
        let bytes = Self::owned_recipe_with_sources_mode(width,target.augur_sources.is_some(),mode)? as u64;
        let resident = cuda_device_allocation_snapshot().live_bytes;
        let free = augur_backend::cuda_memory_info(s.device_ordinal())?.free_bytes;
        let required = Self::required_free_bytes_mode(width,budget,mode)?;
        if required > free
            || resident
                .checked_add(bytes)
                .is_none_or(|n| n > budget.owner_limit)
        {
            return Err("Augur phase memory budget exceeded".into());
        }
        let workspace = full::Workspace::begin(Rc::clone(s), width, target.capacity, mode)?;
        let hidden = [empty(s, width * H)?, empty(s, width * H)?];
        let sources = if let Some(sources)=&target.augur_sources {
            // Explicit model-owned immutable weights; Program::bind checks
            // original weight and state/history owners on every new binding.
            sources.clone()
        } else {
            let mut sources=BTreeMap::new();
            for (index,layer) in target.layers.iter().enumerate() {
                if let Layer::Linear(l)=layer {sources.insert(index,gdn::Source::new(l)?);}
            }
            sources
        };
        if sources.len() != 48 {
            return Err("global GDN inventory changed".into());
        }
        let weight = CudaBf16WeightViewV1::new(
            target._embedding.view(0, target._embedding.len_bytes())?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_BF16_CUDA_V1,
            248320,
            H,
        )?;
        let mut chunks = Vec::new();
        let mut first = 0;
        for &rows in &plan.widths {
            // Keep the existing512-row BF16 gather ABI; no fictitious padding.
            let mut gathers = Vec::new();
            for offset in (0..rows).step_by(512) {
                let n = (rows - offset).min(512);
                gathers.push(CudaBf16TokenRowsGatherArgsV1::bind(
                    s,
                    weight,
                    target.prompt_tokens.view((first + offset) * 4, n * 4)?,
                    CudaF32View::new(
                        hidden[0].view(offset * H * 4, n * H * 4)?,
                        PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                        n * H,
                    )?,
                    n,
                )?);
            }
            let mut programs = Vec::with_capacity(64);
            for (i, layer) in target.layers.iter().enumerate() {
                let input = Rc::clone(&hidden[i % 2]);
                let output = Rc::clone(&hidden[(i + 1) % 2]);
                programs.push(match layer {
                    Layer::Linear(l) => Program::Gdn(gdn::Program::bind(
                        l,
                        Rc::clone(&workspace.common),
                        Rc::clone(&sources[&i]),
                        input,
                        0,
                        output,
                        0,
                        rows,
                        target._artifact.config().rms_norm_eps as f32,
                    )?),
                    Layer::Full(l) => Program::Full(full::Program::bind(
                        l,
                        Rc::clone(&workspace),
                        input,
                        0,
                        output,
                        0,
                        rows,
                        plan.first + first,
                    )?),
                });
            }
            chunks.push(Chunk {
                first,
                rows,
                gathers,
                programs,
            });
            first += rows;
        }
        if augur_backend::cuda_memory_info(s.device_ordinal())?.free_bytes
            < budget
                .safety
                .checked_add(budget.external_reserve)
                .ok_or("global margin overflow")?
        {
            return Err("global phase actual free margin".into());
        }
        Ok(Self {
            workspace,
            hidden,
            sources,
            chunks,
        })
    }
    fn finish(self) -> Result<()> {
        let Self {
            workspace,
            hidden,
            sources,
            chunks,
        } = self;
        workspace.common.stream.synchronize()?;
        drop(chunks);
        drop(sources);
        drop(hidden);
        workspace.finish()?;
        Ok(())
    }
}
impl CheckpointCudaTarget {
    pub(crate) fn attention_mode(&self) -> AttentionMode {
        #[cfg(feature = "qwen35-augur-http")]
        if let Some(decode) = &self.augur_decode { return decode.mode(); }
        AttentionMode::Legacy
    }
    /// Augur-style model-owned immutable source preparation. Never cache a
    /// request's activations, prompt IDs, positions, KV contents or logits.
    /// HTTP installs once before its real startup smoke; non-HTTP callers retain
    /// the original phase-only behavior until they explicitly opt in.
    pub(crate) fn retain_augur_source_weights(&mut self)->Result<serde_json::Value> {
        self.validate_cursor()?;
        if self.consumed!=0 || self.augur_sources.is_some() {
            return Err("padded source install requires a fresh empty owner".into());
        }
        let started=Instant::now();let before=cuda_device_allocation_snapshot();
        let bytes=48*gdn::Source::PADDED_BYTES;
        let free=augur_backend::cuda_memory_info(self.stream.device_ordinal())?.free_bytes;
        if free < bytes as u64 + Budget::default().safety + gdn::Source::PADDED_BYTES as u64 {
            return Err("padded source resident budget exceeded".into());
        }
        let mut sources=BTreeMap::new();
        for (index,layer) in self.layers.iter().enumerate() {
            if let Layer::Linear(l)=layer {sources.insert(index,gdn::Source::new(l)?);}
        }
        if sources.len()!=48 {return Err("padded source layer inventory changed".into());}
        self.stream.synchronize()?;
        let after=cuda_device_allocation_snapshot();
        if after.live_bytes.checked_sub(before.live_bytes)!=Some(bytes as u64) {
            return Err("padded source retained allocation recipe drift".into());
        }
        self.augur_sources=Some(sources);
        let receipt=serde_json::json!({"scope":"same_owner_immutable_original_qpn_gates",
            "owned_bytes":bytes,"weight_count":96,"setup_s":started.elapsed().as_secs_f64(),
            "activation_or_prompt_cache":false,"original_weight_owner_checked_on_each_bind":true});
        self.info["resident_prefill_sources"]=receipt.clone();
        Ok(receipt)
    }

    pub(crate) fn enable_augur_feature_capture(&mut self) -> Result<()> {
        self.validate_cursor()?;
        if !crate::augur_session::CONTEXT_CHOICES.contains(&self.capacity)
            || self.info["profile"] != "quasar_nvfp4"
            || self.info["kv_dtype"] != "Q8_0"
        {
            return Err("Augur capture requires a supported QUASAR KV8 capacity".into());
        }
        let identity = serde_json::to_vec(&serde_json::json!({
            "schema":"quasar-quasar-speed-first-feature-identity-v1",
            "numerical_profile":PROFILE,"kv_identity":self.attention_mode().kv_identity(),
            "authenticated_checkpoint":self._artifact.identity()
        }))?;
        self.install_feature_capture(identity)
    }
    pub(crate) fn plan_augur_sequence(
        &mut self,
        tokens: &[u32],
        widths: Vec<usize>,
    ) -> Result<Plan> {
        self.validate_cursor()?;
        if self.consumed != 0 && (!self.augur_rotated_kv
            || self.info["active_kv_format"] != self.attention_mode().kv_identity()) {
            return Err("cannot reinterpret an existing unrotated KV prefix".into());
        }
        validate_sequence(self.consumed, self.capacity, tokens)?;
        validate_widths(&widths, tokens.len())?;
        if !crate::augur_session::CONTEXT_CHOICES.contains(&self.capacity)
            || self.info["kv_dtype"] != "Q8_0"
            || self.info["policy"]
                != CheckpointProjectionPolicy::AugurQpnAllWorkspaceShapeSmallPairF64.label()
        {
            return Err(
                "Augur loader requires supported KV capacity and original paired-gate weights".into(),
            );
        }
        Ok(Plan {
            schema: "quasar-global-prefill-plan-v1".into(),
            numerical_profile: PROFILE.into(),
            kv_identity: self.attention_mode().kv_identity().into(),
            checkpoint: self.info["source_identity"].clone(),
            first: self.consumed,
            token_ids_sha256: ids_hash(tokens),
            widths,
        })
    }
    pub(crate) fn consume_sequence_augur(
        &mut self,
        tokens: &[u32],
        plan: &Plan,
        budget: Budget,
        mut cancel: impl FnMut(Progress) -> bool,
    ) -> Result<Receipt> {
        self.consume_augur_impl(
            tokens,
            plan,
            budget,
            &mut cancel,
            #[cfg(feature = "qwen35-augur-vision")]
            None,
            #[cfg(test)]
            None,
        )
    }
    #[cfg(feature = "qwen35-augur-vision")]
    pub(super) fn consume_sequence_augur_image_rows(&mut self,tokens:&[u32],plan:&Plan,
        rows:&image_rows::DeviceRows,mut cancel:impl FnMut(Progress)->bool)->Result<Receipt> {
        self.consume_augur_impl(tokens,plan,Budget::default(),&mut cancel,Some(rows),
            #[cfg(test)] None)
    }
    pub(super) fn consume_augur_impl(
        &mut self,
        tokens: &[u32],
        plan: &Plan,
        budget: Budget,
        cancel: &mut dyn FnMut(Progress) -> bool,
        #[cfg(feature = "qwen35-augur-vision")] image_rows: Option<&image_rows::DeviceRows>,
        #[cfg(test)] mut diagnostic: Option<&mut Diagnostic<'_>>,
    ) -> Result<Receipt> {
        let wall = Instant::now();
        #[cfg(feature = "qwen35-augur-vision")]
        if let Some(rows)=image_rows {rows.validate(&self.stream,tokens)?;}
        let expected = self.plan_augur_sequence(tokens, plan.widths.clone())?;
        if *plan != expected {
            return Err("foreign/stale/wrong-token global numerical plan".into());
        }
        let width = validate_widths(&plan.widths, tokens.len())?;
        // Reclaim BEFORE the phase baseline/scratch allocation. Never change
        // B4352, KV capacity or arithmetic to conceal memory pressure.
        #[cfg(feature = "qwen35-augur-vision")]
        let vision_memory_budget = self.reclaim_vision_for_budget(
            Phase::required_free_bytes_mode(width, budget, self.attention_mode())?, "prefill_workspace")?;
        let before = cuda_device_allocation_snapshot();
        let free = augur_backend::cuda_memory_info(self.stream.device_ordinal())?.free_bytes;
        let setup = Instant::now();
        let phase = Phase::bind(self, plan, budget)?;
        let phase_bytes = Phase::owned_recipe_with_sources_mode(width,self.augur_sources.is_some(),self.attention_mode())?;
        let after_setup = cuda_device_allocation_snapshot();
        if after_setup.live_bytes - before.live_bytes != phase_bytes as u64 {
            return Err("global phase allocation recipe drift".into());
        }
        let setup_seconds = setup.elapsed().as_secs_f64();
        self.poisoned = true;
        self.augur_rotated_kv = true;
        self.info["active_kv_format"] = self.attention_mode().kv_identity().into();
        self.logits_current = false;
        self.selected = None;
        let execute = Instant::now();
        #[cfg(test)]
        let diagnostic_io = diagnostic.is_some();
        #[cfg(not(test))]
        let diagnostic_io = false;
        let outcome = (|| -> Result<()> {
            if let Some(c) = &mut self.dflash_capture {
                c.begin(self.consumed, tokens.len())?;
            }
            self.prompt_tokens.copy_from_host_sync(
                &self.stream,
                &tokens
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            for chunk in &phase.chunks {
                for gather in &chunk.gathers {
                    gather.submit_for_plan(&self.stream)?;
                }
                #[cfg(feature = "qwen35-augur-vision")]
                if let Some(rows)=image_rows {rows.submit(&phase.hidden[0],chunk.first,chunk.rows)?;}
                for (index, (program, layer)) in
                    chunk.programs.iter().zip(&mut self.layers).enumerate()
                {
                    let progress = Progress {
                        first: plan.first + chunk.first,
                        rows: chunk.rows,
                        layer: index,
                    };
                    if cancel(progress) {
                        return Err(
                            "augur prefill cancelled; reset and recorded-plan replay required"
                                .into(),
                        );
                    }
                    if layer.tokens() != progress.first {
                        return Err("augur layer frontier drift".into());
                    }
                    // Only the diagnostic callback may read these live planes;
                    // layer input and output are disjoint and remain live here.
                    match (program, layer) {
                        (Program::Gdn(p), Layer::Linear(l)) => p.submit(l)?,
                        (Program::Full(p), Layer::Full(l)) => p.submit(l)?,
                        _ => return Err("global augur layer inventory mismatch".into()),
                    }
                    if let Some(c) = &mut self.dflash_capture {
                        c.layer_window(
                            index,
                            progress.first,
                            chunk.rows,
                            &phase.hidden[(index + 1) % 2],
                        )?;
                    }
                    #[cfg(test)]
                    if let Some(d) = diagnostic.as_deref_mut() {
                        d(
                            progress,
                            &self.stream,
                            &phase.hidden[index % 2],
                            &phase.hidden[(index + 1) % 2],
                            &phase.workspace.common,
                        )?;
                    }
                }
            }
            let last = phase.chunks.last().ok_or("missing global chunk")?;
            // SAFETY: final layer64 lands in hidden0; copy ONLY the last row to
            // the retained scalar owner consumed by the original BF16 head.
            // phase owns the source and self owns the destination on the same
            // stream. Both remain live through the head and synchronize below;
            // the outer path invokes phase.finish before propagating outcome.
            // The copied view itself does not keep phase storage alive. A drain
            // failure is an error, not evidence that the copy has completed.
            unsafe {
                self.hidden[0].copy_from_device_async_unchecked(
                    &self.stream,
                    0,
                    phase.hidden[0].view((last.rows - 1) * H * 4, H * 4)?,
                )?;
            }
            self.head.submit_for_plan()?;
            self.stream.synchronize()?;
            Ok(())
        })();
        let execute_seconds = execute.elapsed().as_secs_f64();
        let hot = cuda_device_allocation_snapshot();
        let drain = Instant::now();
        let release = phase.finish();
        let drain_seconds = drain.elapsed().as_secs_f64();
        // Both success and failed partial submissions release the same phase;
        // nothing commits the target/capture frontier until release succeeds.
        outcome?;
        release?;
        let after = cuda_device_allocation_snapshot();
        if before.live_bytes != after.live_bytes
            || before.live_allocations != after.live_allocations
        {
            return Err("global phase did not return to resident baseline".into());
        }
        if let Some(c) = &mut self.dflash_capture {
            c.commit()?;
        }
        self.consumed += tokens.len();
        self.logits_current = true;
        self.poisoned = false;
        let receipt = Receipt {
            plan: plan.clone(),
            numerical_qualified: false,
            http_qualified: false,
            workspace_profile: gdn::arena::profile(),
            safety_bytes: budget.safety,
            phase_owned_bytes: phase_bytes,
            resident_padded_gate_bytes: if self.augur_sources.is_some() {48*gdn::Source::PADDED_BYTES} else {0},
            padded_gates_reused: self.augur_sources.is_some(),
            #[cfg(feature = "qwen35-augur-vision")]
            vision_memory_budget,
            actual_free_before: free,
            actual_free_after_release: augur_backend::cuda_memory_info(self.stream.device_ordinal())?
                .free_bytes,
            setup_seconds,
            execute_with_upload_head_capture_seconds: execute_seconds,
            drain_release_seconds: drain_seconds,
            complete_wall_seconds: wall.elapsed().as_secs_f64(),
            capture_rows: if self.dflash_capture.is_some() {
                tokens.len().min(2048)
            } else {
                0
            },
            head_submissions: 1,
            hot_allocations: hot.successful_allocations - after_setup.successful_allocations,
            hot_frees: hot.successful_frees - after_setup.successful_frees,
            diagnostic_io,
            #[cfg(feature = "qwen35-augur-vision")]
            image_input: image_rows.map(|rows|rows.receipt()),
        };
        self.info["last_research_augur_prefill"] = serde_json::to_value(&receipt)?;
        Ok(receipt)
    }
}
#[cfg(test)]
mod tests;
