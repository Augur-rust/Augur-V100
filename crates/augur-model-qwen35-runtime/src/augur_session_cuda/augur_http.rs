//! Explicit experimental HTTP setup. Resident DFlash is not speculative decoding.
use super::*;
use crate::checkpoint_target_cuda::{dflash_bridge::AugurDflashBridge, augur_prefill::Budget};
use crate::dflash_q8_backbone_cuda::DFlashQ8CudaWeights;
use std::time::Instant;

pub(crate) fn prepare(
    target: &mut CheckpointCudaTarget,
    mut info: Value,
    draft_load: DraftLoad,
) -> Result<(AugurDflashBridge, Value)> {
    let started = Instant::now();
    let resident_sources=target.retain_augur_source_weights()?;
    target.enable_augur_decode()?;
    for field in ["b1_linear_submission", "b1_linear_graph_layers", "b1_linear_graph_nodes",
        "b1_full_submission", "b1_full_graph_layers", "b1_full_graph_nodes", "b1_head_graph_nodes"] {
        if let Some(value) = target.info().get(field) { info[field] = value.clone(); }
    }
    target.enable_augur_feature_capture()?;
    // Real initialized feature source for the resident draft's existing constructor.
    // This startup smoke is reset before ready and is never counted as request work.
    let ids = vec![16; 16];
    let plan = target.plan_augur_sequence(&ids, vec![16])?;
    let startup_prefill =
        target.consume_sequence_augur(&ids, &plan, Budget::default(), |_| false)?;
    if target.consume_selected().is_ok() {
        return Err("legacy decoder accepted rotated KV".into());
    }
    let mut smoke_ids = Vec::new();
    for _ in 0..3 {
        smoke_ids.push(target.select_next_token()?);
        target.consume_selected_augur()?;
    }
    target.logits_snapshot()?; // drains and rejects any nonfinite logit
    if target.consumed_tokens() != 19 {
        return Err("rotated decode smoke frontier".into());
    }
    let smoke_s = started.elapsed().as_secs_f64();
    let wait = Instant::now();
    let parallel_draft = draft_load.parallel();
    let (q8, draft_cpu_s) = draft_load.finish()?;
    let draft_wait_s = wait.elapsed().as_secs_f64();
    let upload = Instant::now();
    let weights = DFlashQ8CudaWeights::upload(&q8, 0)?;
    let draft_upload_s = upload.elapsed().as_secs_f64();
    let bridge = Instant::now();
    let active=session::requested_dflash_enabled()?;
    let mut draft = if active {AugurDflashBridge::new_http_batched(target,&weights)?} else {AugurDflashBridge::new(target, &weights)?};
    draft.retain_prefill_benchmark_workspace(target)?;
    let bridge_s = bridge.elapsed().as_secs_f64();
    info["http_execution"] = serde_json::json!({
        "resident_prefill_sources":resident_sources,
        "startup_detail":{"smoke_and_decode_setup_s":smoke_s,"parallel_draft_cpu":parallel_draft,
            "draft_cpu_load_s":draft_cpu_s,"draft_join_or_serial_load_s":draft_wait_s,
            "draft_upload_s":draft_upload_s,"bridge_workspace_s":bridge_s},
        "profile": session::requested_profile()?, "prefill_profile": crate::checkpoint_target_cuda::augur_prefill::PROFILE,
        "decode_attention":target.info()["decode_attention"],
        "kv_identity":target.attention_mode().kv_identity(),
        "batch":4352,"decode":if active{"retained_packed_b8_gdn_recorded_prefix_commit"}else{"scalar_qpn_gdn_rotated_q8_full_attention_b1"},
        "scalar_fallback":"budget_one_or_physical_kv_tail_less_than_eight",
        "numerical_qualified":false,"http_qualified":false,"dflash_plan_cache":if session::requested_dflash_plan_cache()?{"park"}else{"recreate"},"dflash_active":active,"dflash_execution_policy":if active{session::DFLASH_B8_PROFILE}else{PROFILE},
        "dflash_resident":draft.resident_info(),"dflash_workspace":draft.resident_workspace_info(),
        "startup_smoke_prefill":startup_prefill,"startup_smoke_decode_ids":smoke_ids,
        "startup_smoke_frontier":19
    });
    if active {
        // resident_info also serves the historical serial diagnostic bridge.
        // Do not advertise those diagnostic-only/inactive flags for this HTTP owner.
        info["http_execution"]["dflash_resident"]["diagnostic_only"] = false.into();
        info["http_execution"]["dflash_resident"]["native_product_enabled"] = true.into();
        info["http_execution"]["dflash_resident"]["batched_target_verifier"] = true.into();
        info["http_execution"]["dflash_resident"]["feature_payload_hash"] = "not_on_resident_http_hot_path".into();
        info["http_execution"]["dflash_resident"]["request_plan_setup"] = "lazy_after_first_published_anchor".into();
    }
    let reset = Instant::now();
    target.reset_empty()?;
    if target.consumed_tokens() != 0 {
        return Err("rotated decode startup reset".into());
    }
    info["http_execution"]["startup_detail"]["reset_s"] = reset.elapsed().as_secs_f64().into();
    info["http_execution"]["setup_including_reset_s"] = started.elapsed().as_secs_f64().into();
    #[cfg(feature="qwen35-augur-vision")]
    {
        info["vision_initialization"] = target.initialize_vision()?;
        info["vision_host"] = target.info()["vision_host"].clone();
        info["http_execution"]["setup_including_reset_and_vision_s"] = started.elapsed().as_secs_f64().into();
    }
    Ok((draft, info))
}

/// CPU-only authentication/payload prefetch. The handle is always joined,
/// including target-load failures; CUDA ownership remains on the owner thread.
pub(crate) struct DraftLoad {
    pending: Option<std::thread::JoinHandle<std::result::Result<(crate::DFlashQ8Backbone, f64), String>>>,
}
impl DraftLoad {
    pub(crate) fn begin() -> Result<Self> {
        let parallel = match std::env::var("AUGUR_PARALLEL_DRAFT_LOAD") {
            Err(std::env::VarError::NotPresent) => true,
            Ok(v) if v == "0" => false,
            Ok(v) if v == "1" => true,
            _ => return Err("AUGUR_PARALLEL_DRAFT_LOAD must be 0 or 1".into()),
        };
        let pending = if parallel {
            Some(std::thread::Builder::new().name("augur-draft-auth".into())
                .spawn(|| load_draft().map_err(|e| e.to_string()))?)
        } else { None };
        Ok(Self { pending })
    }
    fn parallel(&self) -> bool { self.pending.is_some() }
    fn finish(mut self) -> Result<(crate::DFlashQ8Backbone, f64)> {
        match self.pending.take() {
            Some(h) => h.join().map_err(|_| "draft authentication worker panicked")?
                .map_err(Into::into),
            None => load_draft(),
        }
    }
}
impl Drop for DraftLoad {
    fn drop(&mut self) { if let Some(h) = self.pending.take() { let _ = h.join(); } }
}
fn load_draft() -> Result<(crate::DFlashQ8Backbone, f64)> {
    let started = Instant::now();
    let cfg = std::fs::read(std::env::var("AUGUR_DRAFT_CONFIG")?)?;
    if session::digest(&cfg) != std::env::var("AUGUR_DRAFT_CONFIG_SHA256")? {
        return Err("HTTP resident draft config SHA mismatch".into());
    }
    let q8 = crate::DFlashQ8Backbone::load_pinned(
        std::env::var("AUGUR_DRAFT_PATH")?,
        augur_model_qwen35::DFlashConfig::parse_f16_opt_in(&cfg)?,
    )?;
    if q8.manifest().file_sha256.as_ref() != Some(&std::env::var("AUGUR_DRAFT_SHA256")?) {
        return Err("HTTP resident draft weights SHA mismatch".into());
    }
    Ok((q8, started.elapsed().as_secs_f64()))
}
