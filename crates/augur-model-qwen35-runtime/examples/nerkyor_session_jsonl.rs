//! Pinned Nerkyor token producer. Explicit bounded research profile, not QUASAR.
//! The caller renders/decodes the model's own template; this process owns CUDA,
//! exact history and durable receipts. EOF cancels and retires this one owner.
use augur_model_qwen35_runtime::{
    augur_session::{self as session, persist_receipt, Result, TurnRequest},
    nerkyor_session_cuda::NerkyorResearchCudaSession,
    nerkyor_session_manifest,
    session_cancellation::SessionCancellationToken,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    time::Instant,
};
#[cfg(not(feature = "qwen35-nerkyor-cancel-recovery"))]
#[path = "resident_service/cancellable_input.rs"]
mod cancellable_input;
#[cfg(not(feature = "qwen35-nerkyor-cancel-recovery"))]
use cancellable_input::CancellableInput;
#[cfg(feature = "qwen35-nerkyor-cancel-recovery")]
#[path = "nerkyor_service/recovery.rs"]
mod recovery;
#[cfg(feature = "qwen35-nerkyor-cancel-recovery")]
#[path = "nerkyor_service/recovery_input.rs"]
mod recovery_input;
#[cfg(feature = "qwen35-nerkyor-cancel-recovery")]
use recovery::serve;
#[cfg(feature = "qwen35-nerkyor-cancel-recovery")]
use recovery_input::Input as CancellableInput;
const INPUT_LIMIT: u64 = 64 * 1024;
#[cfg(not(feature = "qwen35-nerkyor-cancel-recovery"))]
const SCHEMA: &str = "nerkyor-token-session-jsonl-v1";
#[cfg(feature = "qwen35-nerkyor-cancel-recovery")]
const SCHEMA: &str = "nerkyor-token-session-jsonl-v2";

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
    #[cfg(feature = "qwen35-nerkyor-cancel-recovery")]
    CancelTurn {
        request_id: String,
        cancelled_request_id: String,
        session_id: String,
    },
    Shutdown {},
}
fn output(value: &Value) -> Result<()> {
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    let mut stdout = io::stdout().lock();
    stdout.write_all(&bytes)?;
    stdout.flush()?;
    Ok(())
}
fn parse_command(raw: &[u8]) -> Result<Command> {
    augur_model_qwen35::validate_strict_json(raw)?;
    Ok(serde_json::from_slice(raw)?)
}
#[cfg(not(feature = "qwen35-nerkyor-cancel-recovery"))]
fn serve(
    owner: &mut NerkyorResearchCudaSession,
    input: &CancellableInput,
    cancel: &SessionCancellationToken,
    receipts: &Path,
) -> Result<&'static str> {
    while let Some(raw) = input.next()? {
        let command = match parse_command(&raw) {
            Ok(c) => c,
            Err(e) => {
                output(
                    &json!({"event":"rejected","state_unchanged":true,"owner_retired":false,"reason":e.to_string(),"product_ready":false}),
                )?;
                continue;
            }
        };
        match command {
            Command::Turn { request } => {
                let started = Instant::now();
                match owner.turn(
                    &request,
                    cancel,
                    |t| output(&json!({"event":"token","token":t,"product_ready":false})),
                    |r| persist_receipt(receipts, &r.request_id, r),
                ) {
                    Ok(committed) => output(
                        &json!({"event":"terminal","request_id":request.request_id,"session_id":request.session_id,
                        "committed":committed,"wall_s_through_receipt":started.elapsed().as_secs_f64(),"product_ready":false}),
                    )?,
                    Err(e) => {
                        output(
                            &json!({"event":if owner.poisoned(){"error"}else{"rejected"},"request_id":request.request_id,
                            "session_id":request.session_id,"owner_retired":owner.poisoned(),"state_unchanged":!owner.poisoned(),"reason":e.to_string(),"product_ready":false}),
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
                match owner.close(&request_id, &session_id, revision, |r| {
                    persist_receipt(receipts, &request_id, r)
                }) {
                    Ok(sha) => output(
                        &json!({"event":"closed","request_id":request_id,"session_id":session_id,"revision":revision,
                        "receipt_sha256":sha,"product_ready":false}),
                    )?,
                    Err(e) => {
                        output(
                            &json!({"event":if owner.poisoned(){"error"}else{"rejected"},"request_id":request_id,"session_id":session_id,
                            "owner_retired":owner.poisoned(),"state_unchanged":!owner.poisoned(),"reason":e.to_string(),"product_ready":false}),
                        )?;
                        if owner.poisoned() {
                            return Err(e);
                        }
                    }
                }
            }
            Command::Shutdown {} => return Ok("shutdown"),
        }
    }
    Ok("input_eof")
}
fn options(args: &[String]) -> Result<BTreeMap<String, String>> {
    let allowed = [
        "--model-root",
        "--draft-root",
        "--manifest",
        "--manifest-sha256",
        "--evidence-root",
    ];
    if args.len() != allowed.len() * 2 {
        return Err("exactly five explicit options required; see --help".into());
    }
    let mut options = BTreeMap::new();
    for pair in args.chunks_exact(2) {
        if !allowed.contains(&pair[0].as_str())
            || pair[1].is_empty()
            || options.insert(pair[0].clone(), pair[1].clone()).is_some()
        {
            return Err("unknown, duplicate or empty Nerkyor option".into());
        }
    }
    if !session::valid_digest(&options["--manifest-sha256"]) {
        return Err("explicit lowercase manifest SHA256 required".into());
    }
    Ok(options)
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args == ["--build-identity"] {
        return output(&augur_model_qwen35_runtime::build_identity::identity());
    }
    if args == ["--help"] {
        eprintln!("nerkyor_session_jsonl --model-root ROOT --draft-root Q8_DIRECTORY --manifest FILE --manifest-sha256 HASH --evidence-root NEW_DIRECTORY\nPinned manifest selects DFlash2/512 or QPN source-reassoc no-DFlash/4096; F32 activation/KV, greedy. No MTP/vision/restore/sampling. Keep stdin open through terminal/shutdown.");
        return Ok(());
    }
    let opts = options(&args)?;
    // This fixed profile never silently accepts another head schedule.
    match std::env::var("AUGUR_SHARED_HEAD_KERNEL").as_deref() {
        Err(std::env::VarError::NotPresent) => {
            std::env::set_var("AUGUR_SHARED_HEAD_KERNEL", "llama")
        }
        Ok("llama") => (),
        _ => return Err("Nerkyor B8 requires AUGUR_SHARED_HEAD_KERNEL=llama".into()),
    }
    let root = PathBuf::from(&opts["--model-root"]);
    let draft = PathBuf::from(&opts["--draft-root"]);
    let admitted = nerkyor_session_manifest::load(
        &root,
        &draft,
        Path::new(&opts["--manifest"]),
        &opts["--manifest-sha256"],
    )?;
    let manifest_sha = admitted.sha256().to_owned();
    let eos = admitted.eos_token_ids().to_vec();
    let evidence = PathBuf::from(&opts["--evidence-root"]);
    fs::create_dir(&evidence)?;
    let receipts = evidence.join("turns");
    fs::create_dir(&receipts)?;
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(evidence.join("model-manifest.json"))?;
    file.write_all(admitted.canonical_bytes())?;
    file.sync_all()?;
    drop(file);
    let cancel = SessionCancellationToken::new();
    let input = CancellableInput::start(io::stdin(), INPUT_LIMIT, cancel.clone())?;
    if augur_backend::cuda_device_properties(0)?.compute_capability
        != augur_backend::CudaComputeCapability::SM70
    {
        return Err("Nerkyor bounded profile is only qualified on SM70".into());
    }
    let baseline = augur_backend::cuda_device_allocation_snapshot();
    let started = Instant::now();
    let mut owner = NerkyorResearchCudaSession::open(admitted)?;
    let profile = owner.profile().to_owned();
    let capacity = owner.capacity();
    let dflash_enabled = owner.dflash_enabled();
    let ready = json!({"event":"ready","schema":SCHEMA,"profile":profile,"manifest_sha256":manifest_sha,
        "capacity":capacity,"maximum_output_tokens":capacity*95/100,"generation":"greedy_lowest_id_tie_nonfinite_fail_closed",
        "activation_dtype":"F32","kv":"F32","lm_head":"original_bf16","eos_token_ids":eos,
        "cooperative_input_cancellation":true,"request_bound_cancellation_discard":cfg!(feature = "qwen35-nerkyor-cancel-recovery"),"input_frame_limit_bytes":INPUT_LIMIT,
        "pending_contract":"consumed_history_excludes_pending_append_consumes_pending_once",
        "persistence":"create_new_fsync_turn_receipts_not_gpu_restore","tokenizer_execution":"caller_owned_token_ids",
        "decoded_text_output":false,"startup_wall_s":started.elapsed().as_secs_f64(),"target":owner.info(),
        "dflash_enabled":dflash_enabled,"mtp_enabled":false,"vision_enabled":false,"restore_enabled":false,
        "product_ready":false,"actual_long_history_qualified":false,"http_enabled":false});
    #[cfg(feature = "qwen35-nerkyor-cancel-recovery")]
    let ready = {
        let mut value = ready;
        value["request_cancellation_contract"] = json!(recovery::CONTRACT);
        value
    };
    let result = (|| {
        cancel.check("before Nerkyor ready publication")?;
        persist_receipt(&evidence, "ready", &ready)?;
        output(&ready)?;
        serve(&mut owner, &input, &cancel, &receipts)
    })();
    let submitted = owner.submitted_tokens();
    let drain = owner.retire();
    drop(owner);
    let after = augur_backend::cuda_device_allocation_snapshot();
    let closed = baseline.live_bytes == after.live_bytes
        && baseline.live_allocations == after.live_allocations;
    let stopped = json!({"event":"owner_stopped","schema":"nerkyor-owner-stopped-v1","profile":profile,
        "manifest_sha256":manifest_sha,"reason":result.as_ref().copied().unwrap_or("error"),"error":result.as_ref().err().map(ToString::to_string),
        "drain_error":drain.as_ref().err().map(ToString::to_string),"submitted_tokens_before_retirement":submitted,
        "live_bytes_before":baseline.live_bytes,"live_bytes_after":after.live_bytes,"live_allocations_before":baseline.live_allocations,
        "live_allocations_after":after.live_allocations,"allocation_closure":closed,"product_ready":false});
    persist_receipt(&evidence, "owner-stopped", &stopped)?;
    output(&stopped)?;
    drain?;
    result?;
    if !closed {
        return Err("Nerkyor owner allocation closure failed".into());
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_options_and_protocol_are_closed() {
        let args: Vec<String> = [
            "--model-root",
            "root",
            "--draft-root",
            "draft",
            "--manifest",
            "file",
            "--manifest-sha256",
            &"a".repeat(64),
            "--evidence-root",
            "new",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        assert_eq!(options(&args).unwrap().len(), 5);
        assert!(options(&args[..8]).is_err());
        let mut bad = args.clone();
        bad[2] = "--model-root".into();
        assert!(options(&bad).is_err());
        let mut bad = args.clone();
        bad[2] = "--mtp".into();
        assert!(options(&bad).is_err());
        for raw in [
            br#"{"op":"shutdown","seed":0}"#.as_slice(),
            br#"{"op":"shutdown","op":"shutdown"}"#,
            br#"{"op":"save_checkpoint"}"#,
        ] {
            assert!(parse_command(raw).is_err());
        }
        assert!(parse_command(br#"{"op":"shutdown"}"#).is_ok());
    }
}
