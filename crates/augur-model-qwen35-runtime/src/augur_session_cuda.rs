//! Explicit Augur-only product-owner connection. Text/token protocol first;
//! no DFlash, Vision, HTTP, real256Ki-history or installed-release claim.
use crate::{
    checkpoint_full_cuda::CheckpointKvPolicy,
    checkpoint_projection_cuda::CheckpointProjectionPolicy,
    checkpoint_target_cuda::CheckpointCudaTarget,
    augur_session::{self as session, manifest::AdmittedManifest, *},
    session_cancellation::SessionCancellationToken,
};
use augur_model_qwen35::Qwen35CheckpointProfile;
use serde_json::Value;

impl session::SessionTarget for CheckpointCudaTarget {
    fn supports_images(&self)->bool {cfg!(feature="qwen35-augur-vision")}
    #[cfg(feature="qwen35-augur-vision")]
    fn prefill_images(&mut self,tokens:&[u32],images:&[augur_model_qwen35::Qwen35PreparedF16Image],
        cancel:&SessionCancellationToken)->Result<Option<Value>> {
        Ok(Some(CheckpointCudaTarget::prefill_images(self,tokens,images,||cancel.is_requested())?))
    }
    fn consumed_tokens(&self) -> usize {
        self.consumed_tokens()
    }
    fn consume_sequence(&mut self, ids: &[u32]) -> Result<()> {
        self.consume_sequence(ids)
    }
    #[cfg(feature = "qwen35-augur-http")]
    fn prefill(&mut self, ids: &[u32], cancel: &SessionCancellationToken) -> Result<Option<Value>> {
        use crate::checkpoint_target_cuda::augur_prefill::Budget;
        let widths =
            augur_backend::cuda_qwen35_prefill_cublas::augur_prefill_widths(ids.len())?;
        let plan = self.plan_augur_sequence(ids, widths)?;
        let receipt = self
            .consume_sequence_augur(ids, &plan, Budget::default(), |_| cancel.is_requested())?;
        Ok(Some(serde_json::to_value(receipt)?))
    }
    #[cfg(feature = "qwen35-augur-http")]
    fn decode_detail(&self) -> Option<Value> {
        let mut detail=serde_json::json!({"attention":self.info()["decode_attention"],
            "diagnostic_timing":false,"numerical_qualified":false,"sampling":self.sampling_detail()});
        #[cfg(feature="qwen35-augur-vision")]
        { detail["vision_residency"]=self.vision_residency_info(); }
        Some(detail)
    }
    fn greedy_next_token(&mut self) -> Result<u32> {
        self.select_next_token()
    }
    fn consume_selected(&mut self) -> Result<()> {
        #[cfg(feature = "qwen35-augur-http")]
        {
            self.consume_selected_augur()
        }
        #[cfg(not(feature = "qwen35-augur-http"))]
        {
            self.consume_selected()
        }
    }
    fn drain(&mut self) -> Result<()> {
        self.drain_session_work()
    }
    fn reset_empty(&mut self) -> Result<()> {
        self.reset_empty()
    }
}
#[cfg(feature = "qwen35-augur-http")]
pub(crate) mod augur_http;
#[cfg(feature="qwen35-augur-http")]
mod speculative_http;
#[cfg(feature="qwen35-augur-http")]
pub(crate) type CudaTarget=speculative_http::Target;
#[cfg(not(feature="qwen35-augur-http"))]
type CudaTarget=CheckpointCudaTarget;
pub struct AugurCudaSession {
    owner: session::Session<CudaTarget>,
    info: Value,
}
impl AugurCudaSession {
    pub fn open(admitted: AdmittedManifest) -> Result<Self> {
        let active=session::requested_dflash_enabled()?;
        let park=session::requested_dflash_plan_cache()?;
        #[cfg(feature="qwen35-augur-http")]
        let cost_gate=speculative_http::requested_cost_gate(active)?;
        #[cfg(feature = "qwen35-augur-http")]
        let draft_load = augur_http::DraftLoad::begin()?;
        let mut target = CheckpointCudaTarget::open_with_execution_policy(
            &admitted.root,
            Qwen35CheckpointProfile::QuasarNvfp4,
            admitted.manifest.checkpoint.clone(),
            context_capacity()?,
            CheckpointProjectionPolicy::AugurQpnAllWorkspaceShapeSmallPairF64,
            CheckpointKvPolicy::Q8_0,
        )?;
        target.enable_exact_warp_tree_norm()?;
        target.enable_fused_input_qpn_norm()?;
        target.enable_batched_ffn8()?;
        #[cfg(feature = "qwen35-augur-http")]
        target.configure_sampling()?;
        let info = target.info().clone();
        if info["kv_capacity"] != context_capacity()?
            || info["kv_dtype"] != "Q8_0"
            || info["policy"] != NUMERICAL_POLICY
            || info["qpn_input_staging_schedule"] != "fused_cached_norm_qkv_v1"
            || info["rmsnorm_schedule"] != "same_tree_warp_cached5120_v1"
            || info["ffn_batch_max_rows"] != 8
        {
            return Err("Augur frozen session numerical/capacity contract mismatch".into());
        }
        #[cfg(feature = "qwen35-augur-http")]
        let (draft, mut info) = augur_http::prepare(&mut target, info, draft_load)?;
        #[cfg(feature="qwen35-augur-http")]
        {info["dflash_adaptive_gate"]=serde_json::json!(if !active {"inactive"} else {match cost_gate {speculative_http::CostPolicy::FixedB8=>"fixed_b8_v1",speculative_http::CostPolicy::Legacy=>"legacy_cumulative_checked_acceptance",speculative_http::CostPolicy::Immediate=>"measured_decode_cost_v1",speculative_http::CostPolicy::Deferred=>"deferred_decode_cost_v2"}});}
        #[cfg(feature="qwen35-augur-http")]
        let target=speculative_http::Target::new(target,draft,active,park,cost_gate);
        let owner = session::Session::new(target, admitted.sha256, admitted.eos)?;
        Ok(Self {
            owner,
            info,
        })
    }
    pub fn info(&self) -> &Value {
        &self.info
    }
    pub fn snapshot(&self) -> Option<&SessionSnapshot> {
        self.owner.snapshot()
    }
    pub fn poisoned(&self) -> bool {
        self.owner.poisoned()
    }
    /// Diagnostic cursor only; never a committed-session or recovery authority.
    pub fn submitted_tokens(&self) -> usize {
        self.owner.submitted_tokens()
    }
    pub fn retire(&mut self) -> Result<()> {
        self.owner.retire()
    }
    #[cfg(feature = "qwen35-augur-http")]
    pub fn discard_cancelled_turn(
        &mut self, request_id: &str, cancelled_request_id: &str, session_id: &str,
        persist: impl FnOnce(&CancellationDiscardReceipt) -> Result<String>,
    ) -> Result<CommittedCancellationDiscard> {
        self.owner.discard_cancelled_turn(request_id, cancelled_request_id, session_id, persist)
    }
    pub fn turn(
        &mut self,
        request: &TurnRequest,
        cancel: &SessionCancellationToken,
        emit: impl FnMut(&TokenEvent) -> Result<()>,
        persist: impl FnOnce(&TurnReceipt) -> Result<String>,
    ) -> Result<CommittedTurn> {
        self.owner.turn(request, cancel, emit, persist)
    }
    pub fn turn_with_admission(
        &mut self,
        request: &TurnRequest,
        cancel: &SessionCancellationToken,
        admit: impl FnOnce() -> Result<()>,
        emit: impl FnMut(&TokenEvent) -> Result<()>,
        persist: impl FnOnce(&TurnReceipt) -> Result<String>,
    ) -> Result<CommittedTurn> {
        self.owner.turn_with_admission(request, cancel, admit, emit, persist)
    }
    pub fn close(
        &mut self,
        request_id: &str,
        session_id: &str,
        revision: u64,
        persist: impl FnOnce(&CloseReceipt) -> Result<String>,
    ) -> Result<String> {
        self.owner.close(request_id, session_id, revision, persist)
    }
}
#[cfg(all(test, feature = "qwen35-augur-http"))]
mod cancellation_discard_tests;

#[cfg(all(windows,feature="qwen35-augur-http"))]
mod checkpoint;
#[cfg(all(windows,feature="qwen35-augur-http"))]
impl AugurCudaSession {
    pub fn configure_checkpoint(&mut self,evidence:&std::path::Path)->Result<Value>{self.owner.configure_checkpoint(evidence)}
    pub fn save_checkpoint(&mut self,r:&session::checkpoint::SaveRequest,persist:impl FnOnce(&Value)->Result<String>)->Result<Value>{self.owner.save_checkpoint(r,persist)}
    pub fn restore_checkpoint(&mut self,r:&session::checkpoint::RestoreRequest,persist:impl FnOnce(&Value)->Result<String>)->Result<Value>{self.owner.restore_checkpoint(r,persist)}
}
