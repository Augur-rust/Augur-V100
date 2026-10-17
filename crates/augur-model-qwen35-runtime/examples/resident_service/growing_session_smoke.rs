//! Native initialized-history migration gate. No ordinary HTTP admission yet.
use super::session_smoke::{compare, render_suffix, save};
use super::*;

pub(crate) fn run(
    config: Qwen35AutoregressiveCudaConfig,
    q8: PathBuf,
    draft_config: PathBuf,
    generation: PathBuf,
    tokenizer: Tokenizer,
    evidence: PathBuf,
) -> Result<(), Box<dyn Error>> {
    if evidence.exists() || !evidence.is_absolute() {
        return Err("growing smoke requires fresh absolute evidence".into());
    }
    let prompt = tokenizer
        .encode(config.prompt_text.as_str(), false)
        .map_err(|e| e.to_string())?;
    if !(1..=16384).contains(&prompt.len()) {
        return Err("growing input bound".into());
    }
    fs::create_dir(&evidence)?;
    let baseline = augur_backend::cuda_device_allocation_snapshot();
    let mut owner =
        Qwen35ResidentDraftCuda::open_with_target_and_vision(&config, &q8, &draft_config)?;
    let retained = augur_backend::cuda_device_allocation_snapshot();
    save(&evidence.join("resident.json"), &allocation_json(retained))?;
    let outcome = (|| -> Result<Value, Box<dyn Error>> {
        let sink: &Qwen35VerifiedCommitSink<'_> = &|_| Ok(());
        let opened = owner.open_native_growing_session_audit(
            &config,
            "growing-candidate",
            options(&generation)?,
            sink,
        )?;
        save(&evidence.join("open.json"), &opened)?;
        let generated: Vec<u32> =
            serde_json::from_value(opened["request_output"]["token_ids"].clone())?;
        let mut history: Vec<u32> =
            serde_json::from_value(opened["request_output"]["prompt_token_ids"].clone())?;
        if generated.is_empty()
            || opened["request_output"]["stop_reason"] != "eos"
            || tokenizer.token_to_id("<|im_end|>") != generated.last().copied()
        {
            return Err("growing canonical fixture needs a real pending message-end EOS".into());
        }
        history.extend_from_slice(&generated[..generated.len() - 1]);
        let question = "What is 7 - 2? Reply with exactly one integer, no explanation.";
        let suffix_text = render_suffix(question, true);
        let suffix = tokenizer
            .encode(suffix_text.as_str(), false)
            .map_err(|e| e.to_string())?
            .get_ids()
            .to_vec();
        save(
            &evidence.join("suffix-input.json"),
            &json!({"suffix_text":suffix_text,"suffix_ids":suffix,
            "pending_token_consumed_once":generated.last(),"padding_tokens":0,"product_ready":false}),
        )?;
        let checkpoint = &opened["session_checkpoint"];
        let before = augur_backend::cuda_host_boundary_snapshot();
        let allocations = augur_backend::cuda_device_allocation_snapshot();
        let mut stale = checkpoint.clone();
        stale["initialized_history"]["active_rows"] = json!(0);
        if owner
            .append_native_session(&stale, &history, &suffix)
            .is_ok()
        {
            return Err("corrupt initialized-history checkpoint accepted".into());
        }
        if before != augur_backend::cuda_host_boundary_snapshot()
            || allocations != augur_backend::cuda_device_allocation_snapshot()
        {
            return Err("growing pre-submit reject touched CUDA".into());
        }
        let appended = owner.append_native_session(checkpoint, &history, &suffix)?;
        save(&evidence.join("append.json"), &appended)?;
        history.push(*generated.last().unwrap());
        history.extend_from_slice(&suffix);
        save(
            &evidence.join("exact-history.json"),
            &json!({"token_ids":history,"product_ready":false}),
        )?;
        let captured = owner.capture_native_session_state_audit(
            &appended["checkpoint"],
            &evidence.join("candidate-state"),
        )?;
        save(&evidence.join("candidate-state-audit.json"), &captured)?;
        let decoded = owner.decode_native_growing_session_audit(&appended["checkpoint"], 32)?;
        save(&evidence.join("decode.json"), &decoded)?;
        let closed = owner.close_native_session(&decoded["checkpoint"])?;
        save(&evidence.join("close.json"), &closed)?;
        let now = augur_backend::cuda_device_allocation_snapshot();
        if now.live_bytes != retained.live_bytes
            || now.live_allocations != retained.live_allocations
        {
            return Err("candidate close did not release before fresh KV owner".into());
        }
        let reference = owner.fresh_recompute_native_growing_session_audit(
            &config,
            "growing-reference",
            options(&generation)?,
            &history,
            32,
            &evidence.join("fresh-initial-state"),
            sink,
        )?;
        save(&evidence.join("fresh-reference.json"), &reference)?;
        let reference_closed = owner.close_native_session(&reference["session_checkpoint"])?;
        save(
            &evidence.join("fresh-reference-close.json"),
            &reference_closed,
        )?;
        let mut comparison = compare(&appended["checkpoint"], &decoded, &reference, &history)?;
        let windows_match = [
            &appended["checkpoint"]["initialized_history"],
            &decoded["checkpoint"]["initialized_history"],
        ]
        .into_iter()
        .zip([
            &reference["native_reference_initial_state"]["initialized_history"],
            &reference["session_checkpoint"]["initialized_history"],
        ])
        .all(|(a, b)| {
            !a.is_null()
                && a == b
                && a["physical_capacity_rows"] == 2048
                && a["uninitialized_tail_hashed"] == false
        });
        comparison["initialized_history_exact"] = json!(windows_match);
        let exact = comparison["exact_target_and_output_match"] == true && windows_match;
        comparison["exact_target_and_output_match"] = json!(exact);
        comparison["status"] = json!(if exact {
            "PASS_BOUNDED_GROWING_TARGET_OUTPUT"
        } else {
            "NO_GO_GROWING_EXACT_HISTORY"
        });
        save(&evidence.join("comparison.json"), &comparison)?;
        Ok(
            json!({"schema":"qwen35-native-growing-session-smoke-v1","prompt_tokens":prompt.len(),
            "exact_reference_tokens":history.len(),"suffix_tokens":suffix.len(),
            "initial_history":checkpoint["initialized_history"],"appended_history":appended["checkpoint"]["initialized_history"],
            "decoded_history":decoded["checkpoint"]["initialized_history"],
            "exact_target_and_output_match":exact,"comparison":comparison,"padding_tokens":0,
            "same_immutable_model_owner":true,"physical_target_capacity":196608,
            "pre_submit_rejection_cuda_unchanged":true,"http_continuation":false,"product_ready":false}),
        )
    })();
    drop(owner);
    let after = augur_backend::cuda_device_allocation_snapshot();
    let clean = after.live_bytes == baseline.live_bytes
        && after.live_allocations == baseline.live_allocations
        && after.successful_allocations - baseline.successful_allocations
            == after.successful_frees - baseline.successful_frees;
    let exact = outcome
        .as_ref()
        .is_ok_and(|v| v["exact_target_and_output_match"] == true);
    save(
        &evidence.join("complete.json"),
        &json!({"schema":"qwen35-native-growing-complete-v1",
        "success":outcome.is_ok()&&clean&&exact,"execution_completed":outcome.is_ok(),"cleanup_ok":clean,
        "result":outcome.as_ref().ok(),"error":outcome.as_ref().err().map(|e|e.to_string()),
        "after":allocation_json(after),"product_ready":false,"qualified":false,"visible_tps":false}),
    )?;
    outcome?;
    if !clean || !exact {
        return Err("NO_GO_GROWING_EXACT_HISTORY_OR_CLEANUP".into());
    }
    println!("QWEN35_NATIVE_GROWING_SESSION=PASS_BOUNDED_TARGET_OUTPUT_ONLY");
    Ok(())
}
