//! No token publication. One exact shared prefix, three isolated numeric arms.
use super::*;
pub(crate) fn run(
    config: Qwen35AutoregressiveCudaConfig,
    q8: PathBuf,
    draft: PathBuf,
    generation: PathBuf,
    evidence: PathBuf,
) -> Result<(), Box<dyn Error>> {
    fs::create_dir(&evidence)?;
    let baseline = augur_backend::cuda_device_allocation_snapshot();
    let mut owner = Qwen35ResidentDraftCuda::open_with_target_and_vision(&config, &q8, &draft)?;
    let result = owner.audit_native_prefill_transition(
        &config,
        options(&generation)?,
        &evidence.join("transition"),
    );
    drop(owner);
    let after = augur_backend::cuda_device_allocation_snapshot();
    let clean = after.live_bytes == baseline.live_bytes
        && after.live_allocations == baseline.live_allocations
        && after.successful_allocations - baseline.successful_allocations
            == after.successful_frees - baseline.successful_frees;
    let report = json!({"schema":"qwen35-native-transition-smoke-v1","success":result.is_ok()&&clean,
        "result":result.as_ref().ok(),"error":result.as_ref().err().map(|e|e.to_string()),
        "cleanup_ok":clean,"after":allocation_json(after),"source_dirty":augur_model_qwen35_runtime::build_identity::source_dirty_for(&config.source_commit),
        "product_ready":false,"same_session_qualification":false});
    let mut f = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(evidence.join("complete.json"))?;
    f.write_all(&serde_json::to_vec_pretty(&report)?)?;
    f.sync_all()?;
    result?;
    if !clean {
        return Err("transition owner cleanup failed".into());
    }
    println!("QWEN35_PREFILL_TRANSITION=PASS_CAPTURE_ONLY");
    Ok(())
}
