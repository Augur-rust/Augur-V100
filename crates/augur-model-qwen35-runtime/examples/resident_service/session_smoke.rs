//! Native 8K retain/append vs exact-token fresh prefill. No HTTP/TPS qualification.
use super::*;

pub(super) fn save(path: &std::path::Path, value: &Value) -> Result<(), Box<dyn Error>> {
    let mut f = OpenOptions::new().write(true).create_new(true).open(path)?;
    let mut bound = value.clone();
    if bound.is_object() {
        augur_model_qwen35_runtime::build_identity::annotate(&mut bound);
    }
    f.write_all(&serde_json::to_vec_pretty(&bound)?)?;
    f.flush()?;
    Ok(())
}

pub(crate) fn run(
    config: Qwen35AutoregressiveCudaConfig,
    q8: PathBuf,
    draft_config: PathBuf,
    generation: PathBuf,
    tokenizer: Tokenizer,
    evidence: PathBuf,
    prefill_scalar_audit: bool,
    full_canonical_prefill_audit: bool,
    suffix_case: &str,
    long_audit: bool,
    context32: bool,
    context64: bool,
    near192: bool,
) -> Result<(), Box<dyn Error>> {
    if [context32, context64, near192]
        .iter()
        .filter(|&&x| x)
        .count()
        > 1
    {
        return Err("long-context selectors are mutually exclusive".into());
    }
    if context32 && (!cfg!(feature = "qwen35-session-context32-audit") || !long_audit) {
        return Err("context32 requires explicit compiled long128 profile".into());
    }
    if context64 && (!cfg!(feature = "qwen35-session-context64-audit") || !long_audit) {
        return Err("context64 requires explicit compiled long128 profile".into());
    }
    if near192 && (!cfg!(feature = "qwen35-session-near192-audit") || !long_audit) {
        return Err("near192 requires explicit compiled long128 profile".into());
    }
    let prompt_tokens = if context32 {
        32768
    } else if context64 {
        65536
    } else if near192 {
        194560
    } else {
        8192
    };
    if long_audit && !full_canonical_prefill_audit {
        return Err("long audit needs full canonical".into());
    }
    let audit_profile = if context32 {
        "native_context32_long128_v1"
    } else if context64 {
        "native_context64_long128_v1"
    } else if near192 {
        "native_near192_long128_v1"
    } else if long_audit {
        "native_long128_v1"
    } else {
        "native_short32_v1"
    };
    let continuation_budget = if long_audit { 128 } else { 32 };
    let canonical_suffix = suffix_case == "ledger-counterfactual-canonical-long";
    if canonical_suffix && (context32 || context64 || near192) {
        return Err("canonical suffix reference is bounded to the new 8K identity".into());
    }
    let suffix_question = match suffix_case {
        "add-aligned" => "What is 2 + 3? Answer only the integer.",
        "subtract-nonaligned" if full_canonical_prefill_audit =>
            "What is 7 - 2? Reply with exactly one integer, no explanation.",
        "code-output" if full_canonical_prefill_audit =>
            "Write a Python function named square(x) that returns x * x. Then write assert square(7) == 49. Output only the code, without Markdown.",
        "code-bitmix" if full_canonical_prefill_audit =>
            "Write a Python function named qz(v) returning (v ^ 90) + 137. Then add assert qz(23) == 214. Output code only, no Markdown.",
        "ledger-recall" if full_canonical_prefill_audit =>
            "For the active ledger above, return only a Python list of the net values for r05, r18, r31, r04 and r17, in that order. No explanation or Markdown.",
        "literal-copy" if full_canonical_prefill_audit =>
            "Output exactly one Python assignment: r = \"v9Qz_7mRx_3bTa\". Copy every character exactly. No Markdown or explanation.",
        "ledger-counterfactual-long" | "ledger-counterfactual-canonical-long" if full_canonical_prefill_audit && long_audit =>
            "Recompute active ledger rows r05, r18, r31, r04, r17 and r30, in exactly that order. This time net = outgoing - incoming. Use credit for positive, debit for negative, even for zero. Return only compact JSON {\"rows\":[...]} with id, net, direction in each row. No Markdown.",
        _ => return Err("unknown/ineligible native suffix fixture".into()),
    };
    fs::create_dir(&evidence)?;
    let baseline = augur_backend::cuda_device_allocation_snapshot();
    let mut owner =
        Qwen35ResidentDraftCuda::open_with_target_and_vision(&config, &q8, &draft_config)?;
    let retained = augur_backend::cuda_device_allocation_snapshot();
    save(&evidence.join("resident.json"), &allocation_json(retained))?;
    let outcome = (|| -> Result<Value, Box<dyn Error>> {
        let commits = RefCell::new(Vec::<Value>::new());
        let sink: &Qwen35VerifiedCommitSink<'_> = &|commit| {
            commits.borrow_mut().push(json!({"initial":commit.initial_anchor,
                "cursor_before":commit.cursor_before,"cursor_after":commit.cursor_after,
                "generation_before":commit.generation_before,"generation_after":commit.generation_after,
                "tokens":commit.token_ids,"stop_reason":commit.stop_reason}));
            Ok(())
        };
        let opened = if context32 {
            owner.open_native_session_context32_audit(
                &config,
                "native-context32",
                options(&generation)?,
                sink,
            )?
        } else if context64 {
            owner.open_native_session_context64_audit(
                &config,
                "native-context64",
                options(&generation)?,
                sink,
            )?
        } else if near192 {
            owner.open_native_session_near192_audit(
                &config,
                "native-near192",
                options(&generation)?,
                sink,
            )?
        } else if full_canonical_prefill_audit {
            owner.open_native_session_with_full_canonical_prefill_audit(
                &config,
                "native-smoke-full-canonical",
                options(&generation)?,
                sink,
            )?
        } else if prefill_scalar_audit {
            owner.open_native_session_with_prefill_scalar_audit(
                &config,
                "native-smoke-scalar-audit",
                options(&generation)?,
                sink,
            )?
        } else {
            owner.open_native_session(&config, "native-smoke", options(&generation)?, sink)?
        };
        save(&evidence.join("open.json"), &opened)?;
        save(&evidence.join("commits.json"), &json!(commits.into_inner()))?;
        let checkpoint = &opened["session_checkpoint"];
        let mut history: Vec<u32> =
            serde_json::from_value(opened["request_output"]["prompt_token_ids"].clone())?;
        let generated: Vec<u32> =
            serde_json::from_value(opened["request_output"]["token_ids"].clone())?;
        if generated.is_empty() {
            return Err("initial native fixture produced no pending token".into());
        }
        if canonical_suffix && (opened["request_output"]["stop_reason"] != "eos"
            || tokenizer.token_to_id("<|im_end|>") != generated.last().copied()) {
            return Err("canonical ledger reference requires pending message-end EOS".into());
        }
        history.extend_from_slice(&generated[..generated.len() - 1]);
        let suffix_text = render_suffix(suffix_question, canonical_suffix);
        let suffix = tokenizer
            .encode(suffix_text.as_str(), false)
            .map_err(|e| e.to_string())?
            .get_ids()
            .to_vec();
        if canonical_suffix {
            save(&evidence.join("suffix-input.json"), &json!({
                "schema":"qwen35-canonical-ledger-suffix-audit-v1",
                "framing":"canonical_ledger_text_no_thinking_v1",
                "suffix_text":suffix_text,"suffix_ids":suffix,
                "pending_token_consumed_once":generated.last(),
                "template_validation":"external_pinned_full_transcript_required",
                "semantic_qualification":false,"product_ready":false}))?;
        }
        if suffix_case == "subtract-nonaligned" && (8192 + generated.len() + suffix.len()) % 8 == 0
        {
            return Err("nonaligned fixture unexpectedly aligned".into());
        }
        if suffix.is_empty() || suffix.len() > 128 {
            return Err("smoke suffix bound".into());
        }
        let before_reject = augur_backend::cuda_device_allocation_snapshot();
        let boundaries = augur_backend::cuda_host_boundary_snapshot();
        let mut stale = checkpoint.clone();
        stale["generation"] = json!(0);
        if owner
            .append_native_session(&stale, &history, &suffix)
            .is_ok()
        {
            return Err("stale accepted".into());
        }
        let mut wrong = history.clone();
        wrong[0] = (wrong[0] + 1) % 248320;
        if owner
            .append_native_session(checkpoint, &wrong, &suffix)
            .is_ok()
        {
            return Err("foreign history accepted".into());
        }
        if owner
            .append_native_session(checkpoint, &history, &[])
            .is_ok()
        {
            return Err("empty suffix accepted".into());
        }
        if before_reject != augur_backend::cuda_device_allocation_snapshot()
            || boundaries != augur_backend::cuda_host_boundary_snapshot()
        {
            return Err("pre-submit rejection performed CUDA work".into());
        }
        let appended = owner.append_native_session(checkpoint, &history, &suffix)?;
        save(&evidence.join("append.json"), &appended)?;
        history.push(*generated.last().unwrap());
        history.extend_from_slice(&suffix);
        if appended["checkpoint"]["cursor"].as_u64() != Some(history.len() as u64) {
            return Err("append ledger cursor".into());
        }
        if owner
            .append_native_session(checkpoint, &history, &suffix)
            .is_ok()
        {
            return Err("old checkpoint replay accepted".into());
        }
        let captured = if context32 || context64 || near192 {
            // Seal hashes all128 target planes/full2048 features; no second GPU owner.
            appended["checkpoint"].clone()
        } else {
            owner.capture_native_session_state_audit(
                &appended["checkpoint"],
                &evidence.join("candidate-state"),
            )?
        };
        save(&evidence.join("candidate-state-audit.json"), &captured)?;
        let decoded = if context32 {
            owner.decode_native_session_context32_private_audit(
                &appended["checkpoint"],
                &evidence.join("candidate-private"),
            )?
        } else if context64 {
            owner.decode_native_session_context64_private_audit(
                &appended["checkpoint"],
                &evidence.join("candidate-private"),
            )?
        } else if near192 {
            owner.decode_native_session_near192_private_audit(
                &appended["checkpoint"],
                &evidence.join("candidate-private"),
            )?
        } else if long_audit {
            owner.decode_native_session_long_private_audit(
                &appended["checkpoint"],
                &evidence.join("candidate-private"),
            )?
        } else if full_canonical_prefill_audit {
            owner.decode_native_session_with_private_audit(
                &appended["checkpoint"],
                32,
                &evidence.join("candidate-private"),
            )?
        } else {
            owner.decode_native_session_audit(&appended["checkpoint"], 32)?
        };
        save(&evidence.join("decode.json"), &decoded)?;
        let visible: Vec<u32> = serde_json::from_value(decoded["visible_token_ids"].clone())?;
        let text = tokenizer
            .decode(&visible, false)
            .map_err(|e| e.to_string())?;
        save(
            &evidence.join("decoded-text.json"),
            &json!({"text":text,"semantic_qualification":false}),
        )?;
        let close = owner.close_native_session(&decoded["checkpoint"])?;
        save(&evidence.join("close.json"), &close)?;
        let after = augur_backend::cuda_device_allocation_snapshot();
        if after.live_bytes != retained.live_bytes
            || after.live_allocations != retained.live_allocations
        {
            return Err("session-private state failed to return resident baseline".into());
        }
        // No retokenization and no replay of the initial generated turn:
        // prefill all 8192 + generated (INCLUDING final pending EOS) + suffix IDs.
        save(
            &evidence.join("exact-history.json"),
            &json!({"token_ids":history,
            "original_prompt_tokens":prompt_tokens,"generated_tokens_including_eos":generated.len(),
            "suffix_tokens":suffix.len(),"suffix_case":suffix_case,"decode_retokenize":false,
            "native_session_audit_profile":audit_profile,"max_new_tokens":continuation_budget}),
        )?;
        let reference_commits = RefCell::new(Vec::<Value>::new());
        let reference_sink: &Qwen35VerifiedCommitSink<'_> = &|commit| {
            reference_commits.borrow_mut().push(json!({"initial":commit.initial_anchor,
                "cursor_before":commit.cursor_before,"cursor_after":commit.cursor_after,
                "generation_before":commit.generation_before,"generation_after":commit.generation_after,
                "tokens":commit.token_ids,"stop_reason":commit.stop_reason}));
            Ok(())
        };
        let reference = if context32 {
            owner.fresh_recompute_native_session_context32_audit(
                &config,
                "native-context32-reference",
                options(&generation)?,
                &history,
                &evidence.join("reference-private"),
                reference_sink,
            )?
        } else if context64 {
            owner.fresh_recompute_native_session_context64_audit(
                &config,
                "native-context64-reference",
                options(&generation)?,
                &history,
                &evidence.join("reference-private"),
                reference_sink,
            )?
        } else if near192 {
            owner.fresh_recompute_native_session_near192_audit(
                &config,
                "native-near192-reference",
                options(&generation)?,
                &history,
                &evidence.join("reference-private"),
                reference_sink,
            )?
        } else if long_audit {
            owner.fresh_recompute_native_session_long_audit(
                &config,
                "native-fresh-reference-long",
                options(&generation)?,
                &history,
                &evidence.join("reference-state"),
                &evidence.join("reference-private"),
                reference_sink,
            )?
        } else {
            owner.fresh_recompute_native_session_audit(
                &config,
                "native-fresh-reference",
                options(&generation)?,
                &history,
                32,
                &evidence.join("reference-state"),
                prefill_scalar_audit,
                full_canonical_prefill_audit,
                full_canonical_prefill_audit
                    .then_some(evidence.join("reference-private"))
                    .as_deref(),
                reference_sink,
            )?
        };
        save(&evidence.join("fresh-reference.json"), &reference)?;
        save(
            &evidence.join("fresh-reference-commits.json"),
            &json!(reference_commits.into_inner()),
        )?;
        let reference_close = owner.close_native_session(&reference["session_checkpoint"])?;
        save(
            &evidence.join("fresh-reference-close.json"),
            &reference_close,
        )?;
        let after_reference = augur_backend::cuda_device_allocation_snapshot();
        if after_reference.live_bytes != retained.live_bytes
            || after_reference.live_allocations != retained.live_allocations
        {
            return Err("fresh reference failed to return immutable baseline".into());
        }
        if prefill_scalar_audit
            && (opened["native_prefill_scalar_audit"].is_null()
                || reference["native_prefill_scalar_audit"].is_null())
        {
            return Err("scalar audit must be applied to BOTH numerical identities".into());
        }
        if full_canonical_prefill_audit
            && (opened["native_prefill_full_canonical_audit"].is_null()
                || reference["native_prefill_full_canonical_audit"].is_null())
        {
            return Err("full canonical audit must be applied to BOTH numerical identities".into());
        }
        if full_canonical_prefill_audit
            && (decoded["native_session_private_capture"].is_null()
                || reference["native_session_private_capture"].is_null())
        {
            return Err("paired private draft capture missing".into());
        }
        let comparison = compare(&appended["checkpoint"], &decoded, &reference, &history)?;
        save(&evidence.join("comparison.json"), &comparison)?;
        Ok(json!({"status":comparison["status"],
            "prompt_tokens":prompt_tokens,"generated_tokens":generated.len(),"suffix_tokens":suffix.len(),
            "cursor_before":checkpoint["cursor"],"cursor_after":appended["checkpoint"]["cursor"],
            "pre_submit_rejections_cuda_unchanged":true,"fresh_recompute_compared":true,
            "exact_target_and_output_match":comparison["exact_target_and_output_match"],
            "comparison":comparison,
            "prefill_scalar_audit":prefill_scalar_audit,
            "full_canonical_prefill_audit":full_canonical_prefill_audit,"suffix_case":suffix_case,
            "native_session_audit_profile":audit_profile,"max_new_tokens":continuation_budget,
            "draft_decode_continuation":true,"decoded_text":text,"same_session_qualification":false,
            "draft_state_equivalence_qualified":false,"draft_private_pair_captured":full_canonical_prefill_audit,"context32":context32,"context64":context64,"near192":near192,"raw_target_planes_dumped":!(context32 || context64 || near192),
            "http_continuation":false,"product_ready":false}))
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
    let complete = json!({"schema":"qwen35-native-session-fresh-comparison-v1",
        "success":outcome.is_ok()&&clean&&exact,"execution_completed":outcome.is_ok(),
        "result":outcome.as_ref().ok(),
        "error":outcome.as_ref().err().map(|e|e.to_string()),"cleanup_ok":clean,
        "after":allocation_json(after),"product_ready":false,"source_dirty":augur_model_qwen35_runtime::build_identity::source_dirty_for(&config.source_commit)});
    save(&evidence.join("complete.json"), &complete)?;
    outcome?;
    if !clean {
        return Err("session smoke cleanup failed".into());
    }
    if !exact {
        return Err("NO_GO_EXACT_FULL_HISTORY_EQUIVALENCE; see comparison.json".into());
    }
    println!("QWEN35_NATIVE_SESSION_COMPARE=PASS_BOUNDED_TARGET_OUTPUT_ONLY");
    Ok(())
}

// Keep the historical manual fixture byte-for-byte. The separately named
// canonical case removes its extra newline BEFORE the user message terminator.
// The external evidence harness must compare this against the pinned real model
// template. This helper is not a general chat-template implementation.
pub(super) fn render_suffix(question: &str, canonical: bool) -> String {
    let tail = if canonical { "" } else { "\n" };
    format!("\n<|im_start|>user\n{question}{tail}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
}

// Only these semantic state axes are comparable. Owner IDs, absolute generation,
// request identity and accumulated generated-count intentionally differ.
pub(super) fn compare(
    appended: &Value,
    decoded: &Value,
    reference: &Value,
    history: &[u32],
) -> Result<Value, Box<dyn Error>> {
    let start = &reference["native_reference_initial_state"];
    let end = &reference["session_checkpoint"];
    let output = &reference["request_output"];
    let expected_input: Vec<u32> = serde_json::from_value(output["prompt_token_ids"].clone())?;
    if expected_input != history || reference["native_exact_token_reference"] != true {
        return Err("fresh reference did not consume the exact token ledger".into());
    }
    let state_keys = [
        "cursor",
        "capacity",
        "pending_token",
        "consumed_input_digest",
        "target_committed_state_sha256",
        "feature_bytes_sha256",
        "target_planes",
    ];
    let mut checks = serde_json::Map::new();
    for (phase, a, b) in [
        ("after_suffix", appended, start),
        ("after_decode", &decoded["checkpoint"], end),
    ] {
        for key in state_keys {
            if a[key].is_null() || b[key].is_null() {
                return Err(format!("missing {phase} comparison field: {key}").into());
            }
            checks.insert(format!("{phase}.{key}"), json!(a[key] == b[key]));
        }
    }
    for key in ["token_ids", "visible_token_ids", "stop_reason"] {
        if decoded[key].is_null() || output[key].is_null() {
            return Err(format!("missing decode comparison field: {key}").into());
        }
        checks.insert(format!("output.{key}"), json!(decoded[key] == output[key]));
    }
    let candidate_transactions = normalized_transactions(&decoded["transactions"], appended)?;
    let reference_transactions = normalized_transactions(&reference["transactions"], start)?;
    checks.insert(
        "transactions.relative_generation_and_commit_path".into(),
        json!(candidate_transactions == reference_transactions),
    );
    let drift = |a: &Value, b: &Value| -> Result<Vec<Value>, Box<dyn Error>> {
        let aa = a["state_plane_hashes"]
            .as_array()
            .ok_or("missing plane audit")?;
        let bb = b["state_plane_hashes"]
            .as_array()
            .ok_or("missing plane audit")?;
        if aa.len() != 128 || bb.len() != 128 {
            return Err("state plane count".into());
        }
        Ok(aa
            .iter()
            .zip(bb)
            .filter(|(a, b)| a != b)
            .map(|(a, b)| json!({"candidate":a,"reference":b}))
            .collect())
    };
    let initial_drift = drift(appended, start)?;
    let final_drift = drift(&decoded["checkpoint"], end)?;
    checks.insert(
        "after_suffix.per_plane_exact".into(),
        json!(initial_drift.is_empty()),
    );
    checks.insert(
        "after_decode.per_plane_exact".into(),
        json!(final_drift.is_empty()),
    );
    let exact = checks.values().all(|v| v == true);
    Ok(json!({"schema":"qwen35-native-exact-history-comparison-v1",
        "status":if exact {"PASS_BOUNDED_TARGET_OUTPUT_EQUIVALENCE"}
            else {"NO_GO_EXACT_FULL_HISTORY_EQUIVALENCE"},
        "exact_target_and_output_match":exact,"checks":checks,
        "input_tokens":history.len(),"input_policy":"exact_u32_no_retokenization",
        "candidate_transactions":candidate_transactions,"reference_transactions":reference_transactions,
        "after_suffix_plane_mismatches":initial_drift,"after_decode_plane_mismatches":final_drift,
        "candidate_output":decoded["token_ids"],"reference_output":output["token_ids"],
        "numerical_tolerance_relaxed":false,"draft_state_equivalence_qualified":false,
        "http_continuation":false,"product_ready":false}))
}

fn normalized_transactions(value: &Value, start: &Value) -> Result<Vec<Value>, Box<dyn Error>> {
    let epoch = start["generation"]
        .as_u64()
        .ok_or("missing initial generation")?;
    let mut cursor = start["cursor"].as_u64().ok_or("missing initial cursor")?;
    let mut generation = epoch;
    let mut normalized = Vec::new();
    for tx in value.as_array().ok_or("missing transactions")? {
        let before = tx["generation_before"]
            .as_u64()
            .ok_or("missing generation before")?;
        let after = tx["generation_after"]
            .as_u64()
            .ok_or("missing generation after")?;
        let count = tx["committed_token_count"]
            .as_u64()
            .ok_or("missing committed count")?;
        let emitted: Vec<u32> = serde_json::from_value(tx["emitted_token_ids"].clone())?;
        if before != generation
            || after != generation.checked_add(1).ok_or("epoch overflow")?
            || tx["cursor_before"].as_u64() != Some(cursor)
            || tx["cursor_after"].as_u64() != cursor.checked_add(count)
            || count == 0
            || count > 8
            || emitted.len() as u64 != count
            || tx["accepted_drafts"].as_u64().is_none()
        {
            return Err("nonconserving transaction in comparison".into());
        }
        normalized.push(json!({"generation_before_delta":before-epoch,
            "generation_after_delta":after-epoch,"cursor_before":cursor,
            "cursor_after":cursor+count,"committed_token_count":count,
            "accepted_drafts":tx["accepted_drafts"],"emitted_token_ids":emitted,
            "publication_stop_reason":tx["publication_stop_reason"]}));
        cursor += count;
        generation = after;
    }
    Ok(normalized)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn canonical_suffix_does_not_reconsume_pending_or_alias_legacy() {
        let canonical = render_suffix("question", true);
        let legacy = render_suffix("question", false);
        assert_eq!(canonical, "\n<|im_start|>user\nquestion<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
        assert_eq!(legacy, "\n<|im_start|>user\nquestion\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
        assert_ne!(canonical, legacy);
        assert_eq!(canonical.matches("<|im_end|>").count(), 1);
    }
    #[test]
    fn epoch_normalization_preserves_committed_boundaries() {
        let mut tx = json!([{"generation_before":83,"generation_after":84,
            "cursor_before":8832,"cursor_after":8833,"committed_token_count":1,
            "accepted_drafts":1,"emitted_token_ids":[248046],"publication_stop_reason":"eos"}]);
        let a = normalized_transactions(&tx, &json!({"generation":83,"cursor":8832})).unwrap();
        tx[0]["generation_before"] = json!(1);
        tx[0]["generation_after"] = json!(2);
        let b = normalized_transactions(&tx, &json!({"generation":1,"cursor":8832})).unwrap();
        assert_eq!(a, b);
        tx[0]["cursor_after"] = json!(8834);
        assert!(normalized_transactions(&tx, &json!({"generation":1,"cursor":8832})).is_err());
    }
    #[test]
    fn missing_state_fields_cannot_be_equal_by_null() {
        let reference = json!({"native_exact_token_reference":true,
            "request_output":{"prompt_token_ids":[1,2,3]}});
        assert!(compare(&Value::Null, &Value::Null, &reference, &[1, 2, 3]).is_err());
        assert!(compare(&Value::Null, &Value::Null, &reference, &[1, 2, 4]).is_err());
    }
}
