//! Augur-only token JSONL transport. This is NOT the legacy HTTP protocol.
//! Canonical chat rendering/tokenization belongs to the caller; this stage emits
//! exact token IDs, not decoded text. Optional native checkpoints have an explicit private store.
use augur_model_qwen35_runtime::{
    augur_session::{manifest, *},
    augur_session_cuda::AugurCudaSession,
    session_cancellation::SessionCancellationToken,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs,
    io::{self, Write},
    path::PathBuf,
    time::Instant,
};
#[path = "resident_service/cancellable_input.rs"]
mod cancellable_input;
use cancellable_input::CancellableInput;
#[cfg(feature = "qwen35-augur-http")]
#[path = "augur_service/recovery_input.rs"]
mod recovery_input;

enum Input {
    Legacy(CancellableInput),
    #[cfg(feature = "qwen35-augur-http")]
    Recoverable(recovery_input::Input),
}
impl Input {
    fn next(&self, lifetime: &SessionCancellationToken) -> Result<Option<(Vec<u8>, SessionCancellationToken)>> {
        Ok(match self {
            Self::Legacy(input) => input.next()?.map(|raw| (raw, lifetime.clone())),
            #[cfg(feature = "qwen35-augur-http")]
            Self::Recoverable(input) => input.next()?.map(|work| (work.raw, work.token)),
        })
    }
    fn finish(&self, rid: &str, sid: &str, discarded: bool) -> Result<()> {
        #[cfg(feature = "qwen35-augur-http")]
        if let Self::Recoverable(input) = self { input.finish(rid, sid, discarded)?; }
        let _ = (rid, sid, discarded);
        Ok(())
    }
}

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum Command {
    Turn {
        request: TurnRequest,
    },
    Close {
        request_id: String,
        session_id: String,
        revision: u64,
    },
    #[cfg(all(windows,feature="qwen35-augur-http"))]
    SaveCheckpoint { request: checkpoint::SaveRequest },
    #[cfg(all(windows,feature="qwen35-augur-http"))]
    RestoreCheckpoint { request: checkpoint::RestoreRequest },
    Shutdown {},
}
fn output(value: &Value) -> Result<()> {
    let mut raw = serde_json::to_vec(value)?;
    raw.push(b'\n');
    let mut out = io::stdout().lock();
    out.write_all(&raw)?;
    out.flush()?;
    Ok(())
}
fn serve(
    owner: &mut AugurCudaSession,
    input: &Input,
    cancel: &SessionCancellationToken,
    receipts: &std::path::Path,
) -> Result<&'static str> {
    loop {
        let Some((raw, turn_cancel)) = input.next(cancel)? else {
            return Ok("input_eof");
        };
        let command = match serde_json::from_slice::<Command>(&raw) {
            Ok(command) => command,
            Err(e) => {
                output(&json!({"event":"rejected","reason":e.to_string(),
                "state_unchanged":true,"product_ready":false}))?;
                continue;
            }
        };
        match command {
            Command::Turn { request } => {
                let started = Instant::now();
                let mut receipt_commit_s = 0.0;
                let mut pre_receipt_unix_ns = String::new();
                let reservation = std::cell::RefCell::new(None);
                let result = owner.turn_with_admission(
                    &request,
                    &turn_cancel,
                    || {
                        if receipt_handle_enabled() {
                            let file = reserve_receipt(receipts, &request.request_id)?;
                            output(&json!({"event":"receipt_reserved","request_id":request.request_id,
                                "session_id":request.session_id,"product_ready":false}))?;
                            *reservation.borrow_mut() = Some(file);
                        }
                        Ok(())
                    },
                    |t| output(&json!({"event":"token","token":t,"product_ready":false})),
                    |receipt| {
                        pre_receipt_unix_ns = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)?.as_nanos().to_string();
                        let commit_started = Instant::now();
                        let result = if let Some(file) = reservation.borrow_mut().as_mut() {
                            persist_reserved_receipt(file, receipt)
                        } else {persist_receipt(receipts, &receipt.request_id, receipt)};
                        receipt_commit_s = commit_started.elapsed().as_secs_f64();
                        result
                    },
                );
                match result {
                    Ok(turn) => {
                        input.finish(&request.request_id, &request.session_id, false)?;
                        output(&json!({"event":"terminal","request_id":request.request_id,
                        "session_id":request.session_id,
                        "receipt_transport":if receipt_reference_enabled(){"bound_receipt_v1"}else{"inline_v1"},
                        "committed":if receipt_reference_enabled(){json!({"receipt_sha256":turn.receipt_sha256})}else{serde_json::to_value(&turn)?},
                        "wall_s_through_receipt":started.elapsed().as_secs_f64(),
                        "native_pre_receipt_unix_ns":pre_receipt_unix_ns,
                        "native_receipt_commit_s":receipt_commit_s,
                        "producer_before_terminal_unix_ns":std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)?.as_nanos().to_string(),
                        "product_ready":false,"http_performance_qualified":false}))?
                    }
                    Err(e) => {
                        #[cfg(feature = "qwen35-augur-http")]
                        if let Input::Recoverable(control) = input {
                            if let Some(discard) = control.cancellation(&request.request_id, &request.session_id)? {
                                let record = owner.discard_cancelled_turn(&discard.request_id,
                                    &discard.cancelled_request_id, &discard.session_id,
                                    |r| persist_receipt(receipts, &r.request_id, r))?;
                                input.finish(&request.request_id, &request.session_id, true)?;
                                output(&json!({"event":"cancelled","request_id":request.request_id,
                                    "session_id":request.session_id,"discarded":record,
                                    "owner_retired":false,"product_ready":false}))?;
                                continue;
                            }
                        }
                        if !owner.poisoned() { input.finish(&request.request_id, &request.session_id, false)?; }
                        output(
                            &json!({"event":if owner.poisoned(){"error"}else{"rejected"},
                            "request_id":request.request_id,"session_id":request.session_id,
                            "owner_retired":owner.poisoned(),"reason":e.to_string(),
                            "state_unchanged":!owner.poisoned(),"product_ready":false}),
                        )?;
                        if owner.poisoned() {
                            return Err(e);
                        }
                    }
                }
            }
            Command::Close {
                request_id,
                session_id,
                revision,
            } => {
                match owner.close(&request_id, &session_id, revision, |receipt| {
                    persist_receipt(receipts, &request_id, receipt)
                }) {
                    Ok(hash) => output(&json!({"event":"closed","request_id":request_id,
                        "session_id":session_id,"revision":revision,"receipt_sha256":hash,
                        "product_ready":false}))?,
                    Err(e) => {
                        output(
                            &json!({"event":if owner.poisoned(){"error"}else{"rejected"},
                            "request_id":request_id,"session_id":session_id,
                            "owner_retired":owner.poisoned(),"reason":e.to_string(),"product_ready":false}),
                        )?;
                        if owner.poisoned() {
                            return Err(e);
                        }
                    }
                }
            }
            #[cfg(all(windows,feature="qwen35-augur-http"))]
            Command::SaveCheckpoint { request } => {
                match owner.save_checkpoint(&request,|r|persist_receipt(receipts,&request.request_id,r)) {
                    Ok(saved)=>output(&json!({"event":"checkpoint_saved","request_id":request.request_id,"session_id":request.session_id,
                        "committed":saved,"product_ready":false}))?,
                    Err(e)=>{output(&json!({"event":if owner.poisoned(){"error"}else{"rejected"},"request_id":request.request_id,
                        "session_id":request.session_id,"owner_retired":owner.poisoned(),"reason":e.to_string(),"product_ready":false}))?;
                        if owner.poisoned(){return Err(e);}}
                }
            }
            #[cfg(all(windows,feature="qwen35-augur-http"))]
            Command::RestoreCheckpoint { request } => {
                match owner.restore_checkpoint(&request,|r|persist_receipt(receipts,&request.request_id,r)) {
                    Ok(restored)=>output(&json!({"event":"checkpoint_restored","request_id":request.request_id,"session_id":request.session_id,
                        "committed":restored,"product_ready":false}))?,
                    Err(e)=>{output(&json!({"event":if owner.poisoned(){"error"}else{"rejected"},"request_id":request.request_id,
                        "session_id":request.session_id,"owner_retired":owner.poisoned(),"reason":e.to_string(),"product_ready":false}))?;
                        if owner.poisoned(){return Err(e);}}
                }
            }
            Command::Shutdown {} => return Ok("shutdown"),
        }
    }
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args == ["--build-identity"] {
        return output(&augur_model_qwen35_runtime::build_identity::identity());
    }
    if args == ["--help"] {
        eprintln!("augur_session_jsonl --model-root ROOT --manifest FILE --manifest-sha256 HASH --evidence-root NEW_DIRECTORY\nToken IDs only; caller keeps stdin open until shutdown/terminal. Fixed 262144 KV8, BF16 head, greedy; DFlash/Vision/HTTP/restore unavailable.");
        return Ok(());
    }
    let allowed = [
        "--model-root",
        "--manifest",
        "--manifest-sha256",
        "--evidence-root",
    ];
    let mut options = BTreeMap::new();
    if args.len() != 8 {
        return Err("exactly four explicit options required; see --help".into());
    }
    for pair in args.chunks_exact(2) {
        if !allowed.contains(&pair[0].as_str())
            || options.insert(pair[0].as_str(), pair[1].as_str()).is_some()
        {
            return Err("unknown or duplicate Augur option".into());
        }
    }
    let root = PathBuf::from(options["--model-root"]);
    let path = PathBuf::from(options["--manifest"]);
    let evidence = PathBuf::from(options["--evidence-root"]);
    let admitted = manifest::load(&root, &path, options["--manifest-sha256"])?;
    // Evidence namespace is create-new, never an overwritten prior owner.
    fs::create_dir(&evidence)?;
    let receipts = evidence.join("turns");
    fs::create_dir(&receipts)?;
    let mut manifest_file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(evidence.join("model-manifest.json"))?;
    manifest_file.write_all(admitted.raw())?;
    manifest_file.sync_all()?;
    let manifest_sha256 = admitted.sha256().to_owned();
    let cancel = SessionCancellationToken::new();
    let input_limit=if cfg!(feature="qwen35-augur-vision") {augur_model_qwen35_runtime::augur_session::images::MAX_REQUEST_BYTES as u64} else {16 * 1024 * 1024};
    let recovery = match std::env::var("AUGUR_CANCEL_RECOVERY").as_deref() {
        Ok("discard") if cfg!(feature = "qwen35-augur-http") => true,
        Ok("retire") | Err(std::env::VarError::NotPresent) => false,
        _ => return Err("invalid AUGUR_CANCEL_RECOVERY; expected retire or supported discard".into()),
    };
    #[cfg(feature = "qwen35-augur-http")]
    let input = if recovery { Input::Recoverable(recovery_input::Input::start(io::stdin(), input_limit, cancel.clone())?) }
        else { Input::Legacy(CancellableInput::start(io::stdin(), input_limit, cancel.clone())?) };
    #[cfg(not(feature = "qwen35-augur-http"))]
    let input = Input::Legacy(CancellableInput::start(io::stdin(), input_limit, cancel.clone())?);
    let before = augur_backend::cuda_device_allocation_snapshot();
    let started = Instant::now();
    let mut owner = AugurCudaSession::open(admitted)?;
    #[cfg(all(windows,feature="qwen35-augur-http"))]
    let checkpoint_capabilities=owner.configure_checkpoint(&evidence)?;
    #[cfg(not(all(windows,feature="qwen35-augur-http")))]
    let checkpoint_capabilities=json!({"save_restore_enabled":false,"input_prefix_tokens":null});
    let ready = json!({"event":"ready","schema":"quasar-token-session-jsonl-v1",
        "profile":requested_profile()?,"manifest_sha256":manifest_sha256,"capacity":context_capacity()?,
        "generation":"greedy_lowest_id_tie_nonfinite_fail_closed",
        "lm_head":"original_bf16","kv":"Q8_0","prefill_chunk_tokens":PREFILL_CHUNK,
        "maximum_output_tokens":context_output_limit()?,"cooperative_input_cancellation":true,
        "request_bound_cancellation_discard":recovery,
        "tokenizer_execution":"caller_owned_token_ids","decoded_text_output":false,
        "pending_contract":"consumed_history_excludes_pending_append_consumes_pending_once",
        "persistence":"create_new_fsync_turn_receipts_not_gpu_restore",
        "startup_wall_s":started.elapsed().as_secs_f64(),"target":owner.info(),
        "product_ready":false,"actual_long_history_qualified":false,"http_enabled":false,
        "dflash_enabled":requested_dflash_enabled()?,"vision_enabled":cfg!(feature="qwen35-augur-vision"),"restore_enabled":checkpoint_capabilities["save_restore_enabled"],"checkpoint_capabilities":checkpoint_capabilities});
    let result = (|| {
        cancel.check("before Augur ready publication")?;
        persist_receipt(&evidence, "ready", &ready)?;
        output(&ready)?;
        serve(&mut owner, &input, &cancel, &receipts)
    })();
    let submitted_tokens_before_retirement = owner.submitted_tokens();
    let drain = owner.retire();
    drop(owner); // release all CUDA ownership before a shutdown success frame
    let after = augur_backend::cuda_device_allocation_snapshot();
    let closed =
        before.live_bytes == after.live_bytes && before.live_allocations == after.live_allocations;
    let cleanup = json!({"event":"owner_stopped","reason":result.as_ref().copied().unwrap_or("error"),
        "error":result.as_ref().err().map(ToString::to_string),
        "drain_error":drain.as_ref().err().map(ToString::to_string),
        "live_bytes_before":before.live_bytes,"live_bytes_after":after.live_bytes,
        "live_allocations_before":before.live_allocations,"live_allocations_after":after.live_allocations,
        "allocation_closure":closed,"submitted_tokens_before_retirement":submitted_tokens_before_retirement,
        "product_ready":false});
    persist_receipt(&evidence, "owner-stopped", &cleanup)?;
    output(&cleanup)?;
    drain?;
    result?;
    if !closed {
        return Err("Augur owner allocation closure failed".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn protocol_does_not_accept_legacy_gguf_generation_or_implicit_restore() {
        for raw in [
            r#"{"op":"shutdown","unknown":true}"#,
            r#"{"op":"session_restore","checkpoint_id":"old"}"#,
            r#"{"op":"generate","prompt":"text"}"#,
            r#"{"op":"close","request_id":"r","session_id":"s","revision":1,"capacity":196608}"#,
        ] {
            assert!(
                serde_json::from_str::<Command>(raw).is_err(),
                "accepted: {raw}"
            );
        }
        assert!(serde_json::from_str::<Command>(r#"{"op":"shutdown"}"#).is_ok());
    }
}
