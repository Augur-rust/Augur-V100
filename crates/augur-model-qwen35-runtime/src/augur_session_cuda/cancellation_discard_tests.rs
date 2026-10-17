use super::*;
use std::{fs, path::PathBuf, time::Instant};

/// Real admitted target, full KV and resident inactive draft. Not an HTTP test;
/// explicit producer/broker cancellation-token rotation is still required.
#[test]
#[ignore = "exclusive admitted Augur SM70 GPU and fresh evidence directory required"]
fn cancellation_discard_preserves_weights_and_fresh_output_on_real_gpu() -> Result<()> {
    let root = PathBuf::from(std::env::var("AUGUR_MODEL_ROOT")?);
    let manifest_path = PathBuf::from(std::env::var("AUGUR_MANIFEST")?);
    let manifest_hash = std::env::var("AUGUR_MANIFEST_SHA256")?;
    let evidence = PathBuf::from(std::env::var("AUGUR_CANCEL_DISCARD_EVIDENCE")?);
    fs::create_dir(&evidence)?;
    let before = augur_backend::cuda_device_allocation_snapshot();
    if before.live_allocations != 0 || before.live_bytes != 0 {
        return Err("exclusive empty native allocation scope required".into());
    }
    let admitted = crate::augur_session::manifest::load(&root, &manifest_path, &manifest_hash)?;
    let mut owner = AugurCudaSession::open(admitted)?;
    persist_receipt(&evidence, "target-info", owner.info())?;
    let ids: Vec<u32> = serde_json::from_str(&std::env::var("AUGUR_CANCEL_DISCARD_PROMPT_IDS")?)?;
    let make = |rid: &str, sid: &str| TurnRequest {
        selector_policy: None, request_id: rid.into(), session_id: sid.into(), max_new_tokens: 16,
        input: TurnInput::Open { prompt_token_ids: ids.clone() },
    };
    let token = SessionCancellationToken::new();
    let reference = owner.turn(&make("reference", "reference-session"), &token, |_| Ok(()),
        |r| persist_receipt(&evidence, &r.request_id, r))?;
    owner.close("close-reference", "reference-session", 1,
        |r| persist_receipt(&evidence, &r.request_id, r))?;
    let cancel = SessionCancellationToken::new();
    let interrupted = make("interrupted", "interrupted-session");
    let mut emitted = 0;
    let error = owner.turn(&interrupted, &cancel, |_| {
        emitted += 1;
        if emitted == 1 { cancel.request_cancel(); }
        Ok(())
    }, |_| Err("cancelled turn must not persist success".into())).unwrap_err();
    if !error.is::<crate::session_cancellation::SessionCancellationRequested>() || !owner.poisoned()
        || owner.snapshot().is_some() || emitted != 1 {
        return Err("typed GPU cancellation/poison/commit boundary mismatch".into());
    }
    let started = Instant::now();
    let discarded = owner.discard_cancelled_turn("discard", "interrupted", "interrupted-session",
        |r| persist_receipt(&evidence, &r.request_id, r))?;
    let discard_s = started.elapsed().as_secs_f64();
    if owner.poisoned() || owner.snapshot().is_some() || owner.submitted_tokens() != 0 || !cancel.is_requested() {
        return Err("GPU cancellation discard/token contract mismatch".into());
    }
    let started = Instant::now();
    let actual = owner.turn(&make("after-discard", "after-discard-session"),
        &SessionCancellationToken::new(), |_| Ok(()), |r| persist_receipt(&evidence, &r.request_id, r))?;
    let replay_s = started.elapsed().as_secs_f64();
    if actual.receipt.output_token_ids != reference.receipt.output_token_ids
        || actual.receipt.session.consumed_token_ids != reference.receipt.session.consumed_token_ids
        || actual.receipt.session.consumed_input_sha256 != reference.receipt.session.consumed_input_sha256
        || actual.receipt.session.pending_token != reference.receipt.session.pending_token
        || actual.receipt.session.pending_token_published != reference.receipt.session.pending_token_published {
        return Err("GPU post-discard output/history/pending differs from same-owner fresh control".into());
    }
    owner.close("close-actual", "after-discard-session", 1,
        |r| persist_receipt(&evidence, &r.request_id, r))?;
    owner.retire()?;
    drop(owner);
    let allocations = augur_backend::cuda_device_allocation_snapshot();
    if allocations.live_allocations != 0 || allocations.live_bytes != 0 {
        return Err("GPU cancellation discard allocation closure failed".into());
    }
    let complete = serde_json::json!({"gpu_pass":true,"interrupted_emitted_tokens":emitted,
        "discard_wall_s":discard_s,"same_owner_fresh_replay_wall_s":replay_s,
        "reference_output_token_ids":reference.receipt.output_token_ids,
        "actual_output_token_ids":actual.receipt.output_token_ids,
        "discard_receipt_sha256":discarded.receipt_sha256,"live_allocations":0,"live_bytes":0,
        "http_recovery_integrated":false,"http_speedup_verified":false,
        "product_ready":false,"numerical_qualified":false,"dflash_active":false});
    persist_receipt(&evidence, "complete", &complete)?;
    println!("{complete}");
    Ok(())
}
