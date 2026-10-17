//! Native sink disconnect: stop, poison, drop, then explicit fresh-owner recovery.
use super::*;

fn save(path: &std::path::Path, value: &Value) -> Result<(), Box<dyn Error>> {
    let mut f = OpenOptions::new().write(true).create_new(true).open(path)?;
    f.write_all(&serde_json::to_vec_pretty(value)?)?;
    f.sync_all()?;
    Ok(())
}

struct Disconnect {
    callbacks: usize,
    published: Vec<u32>,
    accepted_frames: Vec<Value>,
    failed_frame: Option<Value>,
}
impl Disconnect {
    fn new() -> Self {
        Self {
            callbacks: 0,
            published: Vec::new(),
            accepted_frames: Vec::new(),
            failed_frame: None,
        }
    }
    fn consume(&mut self, commit: &Qwen35VerifiedCommit) -> Result<(), Qwen35MixedPrefixCudaError> {
        self.callbacks += 1;
        let frame = json!({"initial":commit.initial_anchor,"round":commit.block_index,
            "generation_before":commit.generation_before,"generation_after":commit.generation_after,
            "cursor_before":commit.cursor_before,"cursor_after":commit.cursor_after,
            "token_ids":commit.token_ids,"stop_reason":commit.stop_reason});
        if self.callbacks == 3 {
            self.failed_frame = Some(frame);
            return Err(invalid("native_session_simulated_consumer_disconnect"));
        }
        if self.callbacks > 3 {
            return Err(invalid("callback continued after disconnect"));
        }
        self.published.extend_from_slice(&commit.token_ids);
        self.accepted_frames.push(frame);
        Ok(())
    }
}

pub(crate) fn run(
    config: Qwen35AutoregressiveCudaConfig,
    q8: PathBuf,
    draft_config: PathBuf,
    generation: PathBuf,
    tokenizer: Tokenizer,
    evidence: PathBuf,
) -> Result<(), Box<dyn Error>> {
    fs::create_dir(&evidence)?;
    let baseline = augur_backend::cuda_device_allocation_snapshot();
    let outcome = (|| -> Result<Value, Box<dyn Error>> {
        let mut owner =
            Qwen35ResidentDraftCuda::open_with_target_and_vision(&config, &q8, &draft_config)?;
        let discard: &Qwen35VerifiedCommitSink<'_> = &|_| Ok(());
        let opened = owner.open_native_session_with_full_canonical_prefill_audit(
            &config,
            "cancel-original",
            options(&generation)?,
            discard,
        )?;
        save(&evidence.join("open.json"), &opened)?;
        let old_checkpoint = &opened["session_checkpoint"];
        let mut history: Vec<u32> =
            serde_json::from_value(opened["request_output"]["prompt_token_ids"].clone())?;
        let generated: Vec<u32> =
            serde_json::from_value(opened["request_output"]["token_ids"].clone())?;
        history.extend_from_slice(&generated[..generated.len() - 1]);
        let question="Recompute active ledger rows r05, r18, r31, r04, r17 and r30, in exactly that order. This time net = outgoing - incoming. Use credit for positive, debit for negative, even for zero. Return only compact JSON {\"rows\":[...]} with id, net, direction in each row. No Markdown.";
        let text=format!("\n<|im_start|>user\n{question}\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
        let suffix = tokenizer
            .encode(text, false)
            .map_err(|e| e.to_string())?
            .get_ids()
            .to_vec();
        let appended = owner.append_native_session(old_checkpoint, &history, &suffix)?;
        save(&evidence.join("append.json"), &appended)?;
        history.push(*generated.last().ok_or("missing pending")?);
        history.extend_from_slice(&suffix);
        save(
            &evidence.join("exact-history.json"),
            &json!({"token_ids":history,"decode_retokenize":false}),
        )?;
        let observed = RefCell::new(Disconnect::new());
        let sink: &Qwen35VerifiedCommitSink<'_> = &|commit| observed.borrow_mut().consume(commit);
        let started = Instant::now();
        let cancelled = owner.decode_native_session_stream_audit(&appended["checkpoint"], sink);
        let error = cancelled
            .err()
            .ok_or("disconnect unexpectedly completed")?
            .to_string();
        let fault = observed.into_inner();
        if !error.contains("native_session_simulated_consumer_disconnect")
            || fault.callbacks != 3
            || fault.accepted_frames.len() != 2
            || fault.failed_frame.is_none()
            || fault.published.len() < 2
        {
            return Err("disconnect boundary did not trigger as specified".into());
        }
        let counts = augur_backend::cuda_device_allocation_snapshot();
        let boundaries = augur_backend::cuda_host_boundary_snapshot();
        let probes = json!({
            "decode":owner.decode_native_session_stream_audit(&appended["checkpoint"],discard).err().map(|e|e.to_string()),
            "append":owner.append_native_session(&appended["checkpoint"],&history,&suffix).err().map(|e|e.to_string()),
            "close":owner.close_native_session(&appended["checkpoint"]).err().map(|e|e.to_string()),
            "open":owner.open_native_session_with_full_canonical_prefill_audit(&config,"must-reject",options(&generation)?,discard).err().map(|e|e.to_string())});
        if probes
            .as_object()
            .ok_or("probes")?
            .values()
            .any(Value::is_null)
            || counts != augur_backend::cuda_device_allocation_snapshot()
            || boundaries != augur_backend::cuda_host_boundary_snapshot()
        {
            return Err("poisoned owner performed work or admitted reuse".into());
        }
        let cancellation = json!({"schema":"qwen35-native-session-disconnect-v1","expected_error":error,
            "callbacks":fault.callbacks,"published_token_ids":fault.published,"accepted_frames":fault.accepted_frames,
            "failed_frame_not_published":fault.failed_frame,"terminal_success_emitted":false,
            "poisoned_rejections":probes,"rejections_cuda_unchanged":true,
            "native_call_return_ms_including_drain":started.elapsed().as_secs_f64()*1000.,
            "http_disconnect":false,"product_ready":false});
        save(&evidence.join("cancel.json"), &cancellation)?;
        drop(owner);
        let dropped = augur_backend::cuda_device_allocation_snapshot();
        if dropped.live_bytes != baseline.live_bytes
            || dropped.live_allocations != baseline.live_allocations
            || dropped.successful_allocations - baseline.successful_allocations
                != dropped.successful_frees - baseline.successful_frees
        {
            return Err("cancelled owner drop did not reclaim complete state".into());
        }
        save(
            &evidence.join("after-cancel-drop.json"),
            &allocation_json(dropped),
        )?;
        // No attempt to rehabilitate a poisoned owner. Explicit full history replay.
        let mut fresh =
            Qwen35ResidentDraftCuda::open_with_target_and_vision(&config, &q8, &draft_config)?;
        let retained = augur_backend::cuda_device_allocation_snapshot();
        let recovery_frames = RefCell::new(Vec::<Value>::new());
        let recover_sink: &Qwen35VerifiedCommitSink<'_> = &|c| {
            recovery_frames.borrow_mut().push(json!({"initial":c.initial_anchor,"round":c.block_index,
                "cursor_before":c.cursor_before,"cursor_after":c.cursor_after,"token_ids":c.token_ids,"stop_reason":c.stop_reason}));
            Ok(())
        };
        let recovery = fresh.fresh_recompute_native_session_long_audit(
            &config,
            "cancel-recovery-new-owner",
            options(&generation)?,
            &history,
            &evidence.join("recovery-state"),
            &evidence.join("recovery-private"),
            recover_sink,
        )?;
        save(&evidence.join("recovery.json"), &recovery)?;
        save(
            &evidence.join("recovery-commits.json"),
            &json!(recovery_frames.into_inner()),
        )?;
        let recovered: Vec<u32> =
            serde_json::from_value(recovery["request_output"]["token_ids"].clone())?;
        if !recovered.starts_with(&fault.published)
            || recovery["request_output"]["stop_reason"] != "eos"
        {
            return Err("recovered exact history failed prefix/EOS".into());
        }
        let before = augur_backend::cuda_device_allocation_snapshot();
        let boundary = augur_backend::cuda_host_boundary_snapshot();
        if fresh.close_native_session(&appended["checkpoint"]).is_ok()
            || fresh
                .append_native_session(&appended["checkpoint"], &history, &suffix)
                .is_ok()
            || fresh
                .decode_native_session_stream_audit(&appended["checkpoint"], discard)
                .is_ok()
            || before != augur_backend::cuda_device_allocation_snapshot()
            || boundary != augur_backend::cuda_host_boundary_snapshot()
        {
            return Err("old session capability admitted to new owner".into());
        }
        let recovery_close = fresh.close_native_session(&recovery["session_checkpoint"])?;
        save(&evidence.join("recovery-close.json"), &recovery_close)?;
        let closed = augur_backend::cuda_device_allocation_snapshot();
        if closed.live_bytes != retained.live_bytes
            || closed.live_allocations != retained.live_allocations
        {
            return Err("recovery request cleanup".into());
        }
        drop(fresh);
        Ok(
            json!({"status":"PASS_NATIVE_DISCONNECT_POISON_DROP_REOPEN_ONLY","cancelled_owner_reuse_rejected":true,
            "poisoned_drop_live_zero":true,"new_owner_stale_capabilities_rejected":true,
            "recovery_output_tokens":recovered.len(),"recovery_eos":true,"visible_prefix_preserved":true,
            "weights_reloaded_for_safe_recovery":true,"transparent_resume":false,"http_qualified":false,"product_ready":false}),
        )
    })();
    let after = augur_backend::cuda_device_allocation_snapshot();
    let clean = after.live_bytes == baseline.live_bytes
        && after.live_allocations == baseline.live_allocations
        && after.successful_allocations - baseline.successful_allocations
            == after.successful_frees - baseline.successful_frees;
    save(
        &evidence.join("complete.json"),
        &json!({"schema":"qwen35-native-session-cancel-recovery-v1",
        "success":outcome.is_ok()&&clean,"result":outcome.as_ref().ok(),"error":outcome.as_ref().err().map(|e|e.to_string()),
        "cleanup_ok":clean,"after":allocation_json(after),"source_dirty":augur_model_qwen35_runtime::build_identity::source_dirty_for(&config.source_commit),"product_ready":false}),
    )?;
    outcome?;
    if !clean {
        return Err("cancel gate final cleanup".into());
    }
    println!("QWEN35_NATIVE_SESSION_CANCEL=PASS_BOUNDED_NATIVE_ONLY");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn commit(initial: bool, round: usize, tokens: Vec<u32>) -> Qwen35VerifiedCommit {
        Qwen35VerifiedCommit {
            initial_anchor: initial,
            block_index: round,
            generation_before: 83 + round,
            generation_after: 84 + round,
            cursor_before: 8896 + round,
            cursor_after: 8897 + round,
            token_ids: tokens,
            stop_reason: None,
        }
    }
    #[test]
    fn disconnect_never_adds_failed_frame_to_published_ledger() {
        let mut sink = Disconnect::new();
        sink.consume(&commit(true, 0, vec![10])).unwrap();
        sink.consume(&commit(false, 0, vec![11, 12])).unwrap();
        assert!(sink.consume(&commit(false, 1, vec![13, 14])).is_err());
        assert_eq!(sink.published, vec![10, 11, 12]);
        assert!(sink.consume(&commit(false, 2, vec![15])).is_err());
        assert_eq!(sink.published, vec![10, 11, 12]);
    }
}
