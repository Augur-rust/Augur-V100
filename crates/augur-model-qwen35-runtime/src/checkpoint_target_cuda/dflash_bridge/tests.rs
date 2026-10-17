use super::*;
use crate::DFlashQ8Backbone;
use augur_model_qwen35::DFlashConfig;
use std::{collections::BTreeSet, io::Write};

fn save(name: &str, value: &Value) {
    let root =
        std::path::PathBuf::from(std::env::var_os("AUGUR_BRIDGE_OUTPUT").expect("output"));
    let mut file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(root.join(name))
        .unwrap();
    file.write_all(&serde_json::to_vec_pretty(value).unwrap())
        .unwrap();
    file.sync_all().unwrap();
}
fn f64_metric(actual: &[f32], reference: &[f64]) -> Value {
    assert_eq!(actual.len(), reference.len());
    assert!(actual.iter().all(|v| v.is_finite()) && reference.iter().all(|v| v.is_finite()));
    let error2: f64 = actual
        .iter()
        .zip(reference)
        .map(|(a, b)| (f64::from(*a) - b).powi(2))
        .sum();
    let norm2: f64 = reference.iter().map(|v| v * v).sum();
    let max_abs = actual
        .iter()
        .zip(reference)
        .map(|(a, b)| (f64::from(*a) - b).abs())
        .fold(0., f64::max);
    let max_ref = reference.iter().map(|v| v.abs()).fold(0., f64::max);
    let rel_l2 = (error2 / norm2.max(1e-300)).sqrt();
    let scaled_max = max_abs / (1. + max_ref);
    assert!(
        rel_l2 <= 1e-3 && scaled_max <= 1e-3,
        "relL2={rel_l2} scaledMax={scaled_max}"
    );
    json!({"observations":actual.len(),"rel_l2":rel_l2,"scaled_max":scaled_max})
}
fn head_math(
    target: &mut CheckpointCudaTarget,
    proposal: &AugurDraftProposal,
    q8: &DFlashQ8Backbone,
) -> Value {
    let d = proposal.head();
    let mut actual = Vec::new();
    let mut expected = Vec::new();
    let mut counts = Vec::new();
    for row in 0..B {
        let mut ids: BTreeSet<u32> = d.candidate_ids[row * K..(row + 1) * K]
            .iter()
            .copied()
            .collect();
        ids.extend([0, 1, 2, 248319, 248070]);
        ids.extend((0..32).map(|i| ((i * 7919 + row * 3571) % V) as u32));
        counts.push(ids.len());
        for id in ids {
            let mut wire = vec![0; H * 2];
            target
                ._artifact
                .read_tensor_range_into("lm_head.weight", u64::from(id) * (H * 2) as u64, &mut wire)
                .unwrap();
            let sum: f64 = wire
                .chunks_exact(2)
                .zip(&d.normalized_hidden[row * H..(row + 1) * H])
                .map(|(b, x)| {
                    f64::from(f32::from_bits(
                        u32::from(u16::from_le_bytes(b.try_into().unwrap())) << 16,
                    )) * f64::from(*x)
                })
                .sum();
            expected.push(sum);
            actual.push(d.logits[row * V + id as usize]);
        }
        // Entire-vocabulary top16 CPU order, not just the sampled matrix rows.
        let mut order: Vec<usize> = (0..V).collect();
        let logits = &d.logits[row * V..(row + 1) * V];
        order.sort_unstable_by(|a, b| logits[*b].partial_cmp(&logits[*a]).unwrap().then(a.cmp(b)));
        assert_eq!(
            d.candidate_ids[row * K..(row + 1) * K],
            order[..K].iter().map(|v| *v as u32).collect::<Vec<_>>()
        );
        for i in 0..K {
            assert_eq!(d.unary_scores[row * K + i], logits[order[i]]);
        }
    }
    let head_metric = f64_metric(&actual, &expected);
    let gates: Vec<f32> = proposal.diagnostic()["selector_gates"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let mut selector_actual = Vec::new();
    let mut selector_reference = Vec::new();
    let mut prior = d.path[0];
    let mut margins = Vec::new();
    for row in 1..B {
        let pred = q8.selector_codebook_row(true, prior).unwrap();
        let mut scores = Vec::new();
        for c in 0..K {
            let succ = q8
                .selector_codebook_row(false, d.candidate_ids[row * K + c])
                .unwrap();
            let dot: f64 = (0..256)
                .map(|i| f64::from(pred[i] * gates[row * 256 + i]) * f64::from(succ[i]))
                .sum();
            let score = f64::from(d.unary_scores[row * K + c]) + dot;
            scores.push(score);
            selector_reference.push(score);
            selector_actual.push(d.transition_scores[row * K + c]);
        }
        let mut order: Vec<usize> = (0..K).collect();
        order.sort_unstable_by(|a, b| scores[*b].partial_cmp(&scores[*a]).unwrap().then(a.cmp(b)));
        let expected_id = d.candidate_ids[row * K + order[0]];
        assert_eq!(
            expected_id, d.path[row],
            "same-Q8 selector CPU F64 path at row{row}"
        );
        margins.push(scores[order[0]] - scores[order[1]]);
        prior = expected_id;
    }
    json!({"bf16_head_sampled_math":head_metric,"sampled_vocab_rows_per_draft_row":counts,
        "whole_vocab_top16_cpu_match":true,"same_q8_selector_math":f64_metric(&selector_actual,&selector_reference),
        "same_q8_selector_path_exact":true,"selector_margins":margins,"limits_unchanged":1e-3,
        "draft_normalized_input_only":true,"new_head_quantization":false})
}
fn baseline(
    target: &mut CheckpointCudaTarget,
    prompt: &[u32],
    budget: usize,
    eos: &[u32],
) -> (Vec<u32>, usize, Vec<f32>, f64) {
    target.reset_empty().unwrap();
    target.consume_sequence(prompt).unwrap();
    let started = Instant::now();
    let mut out = Vec::new();
    while out.len() < budget {
        let id = target.greedy_next_token().unwrap();
        if eos.contains(&id) {
            break;
        }
        out.push(id);
        if out.len() == budget {
            break;
        }
        target.consume_selected().unwrap();
    }
    target.stream.synchronize().unwrap();
    let wall = started.elapsed().as_secs_f64();
    (
        out,
        target.consumed,
        target.logits_snapshot().unwrap(),
        wall,
    )
}
#[test]
fn shared_head_workspace_formula_cannot_hide_a_second_weight_copy() {
    let workspace = B * (H + V) * 4 + 3 * B * K * 4 + B * 4;
    assert_eq!(workspace, 8_111_648);
    assert!(workspace < V * H * 2 / 100);
}
#[test]
#[ignore = "exclusive full256Ki Augur + real Q8 DFlash; diagnostic serial bridge, not HTTP speed qualification"]
fn real_augur_q8_shared_bf16_head_and_canonical_serial_bridge() {
    use super::super::transfer_tests::{ids, open};
    let source =
        std::fs::read(std::env::var_os("AUGUR_CHECKPOINT_TARGET_REFERENCE").expect("fixture"))
            .unwrap();
    let fixture: Value = serde_json::from_slice(&source).unwrap();
    let cfg_bytes =
        std::fs::read(std::env::var_os("AUGUR_DRAFT_CONFIG").expect("draft config")).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&cfg_bytes)),
        std::env::var("AUGUR_DRAFT_CONFIG_SHA256").unwrap()
    );
    let config = DFlashConfig::parse_f16_opt_in(&cfg_bytes).unwrap();
    let q8 = DFlashQ8Backbone::load_pinned(
        std::env::var_os("AUGUR_DRAFT_PATH").expect("Q8 path"),
        config,
    )
    .unwrap();
    assert_eq!(
        q8.manifest().file_sha256.as_ref().unwrap(),
        &std::env::var("AUGUR_DRAFT_SHA256").unwrap()
    );
    let before = cuda_device_allocation_snapshot();
    let mut target = open(&fixture);
    assert_eq!(target.capacity, 262144);
    target.enable_dflash_feature_capture().unwrap();
    let target_info = target.info().clone();
    let resident = DFlashQ8CudaWeights::upload(&q8, 0).unwrap();
    // A different CUDA stream incarnation may not bind this target's head,
    // even on the same GPU. Rejection must allocate no duplicate model weight.
    let foreign_stream = Rc::new(OwnedCudaStream::create(0).unwrap());
    let foreign_input = empty(&foreign_stream, H).unwrap();
    let foreign_logits = empty(&foreign_stream, V).unwrap();
    assert!(target
        .head
        .bind_shared_normalized_bf16(&foreign_stream, foreign_input, 0, foreign_logits, 0)
        .is_err());
    drop(foreign_stream);
    let head_before = cuda_device_allocation_snapshot();
    let mut bridge = AugurDflashBridge::new(&target, &resident).unwrap();
    let head_after = cuda_device_allocation_snapshot();
    assert_eq!(
        head_after.live_bytes - head_before.live_bytes,
        bridge.head.bytes() as u64
    );
    assert_eq!(bridge.head.bytes(), 8_111_648);
    let bridge_info = bridge.resident_info();
    let eos = ids(&fixture["eos_token_ids"]);
    let tokenizer = tokenizers::Tokenizer::from_bytes(super::super::tests::bytes(
        &fixture["metadata"]["tokenizer.json"],
    ))
    .unwrap();
    let mut families = Vec::new();
    let mut mathematical = None;
    for request in fixture["generation"].as_array().unwrap() {
        let prompt = ids(&request["token_ids"]);
        assert_eq!(
            tokenizer
                .encode(request["canonical_prompt"].as_str().unwrap(), false)
                .unwrap()
                .get_ids(),
            prompt
        );
        let budget = request["max_new_tokens"].as_u64().unwrap() as usize;
        let (expected, cursor, logits, baseline_wall) =
            baseline(&mut target, &prompt, budget, &eos);
        target.reset_empty().unwrap();
        target.consume_sequence(&prompt).unwrap();
        let started = Instant::now();
        let mut output = Vec::new();
        let mut rounds = Vec::new();
        while output.len() < budget {
            let anchor = target.greedy_next_token().unwrap();
            if eos.contains(&anchor) {
                break;
            }
            let proposal = bridge.propose_diagnostic(&mut target).unwrap();
            assert_eq!(proposal.head().path[0], anchor);
            assert_eq!(
                proposal.capture().context_start_position + proposal.capture().context_len,
                target.consumed
            );
            let diagnostic = proposal.diagnostic().clone();
            let head_path = proposal.head().path.clone();
            let top16 = proposal.head().candidate_ids.clone();
            // First real block gets an independent sampled BF16/selector oracle.
            // This extra audit is NOT a generation-performance observation.
            let mut audit_s = 0.;
            if mathematical.is_none() {
                let audit = Instant::now();
                mathematical = Some(head_math(&mut target, &proposal, &q8));
                audit_s = audit.elapsed().as_secs_f64();
            }
            let result = bridge
                .verify_serial(&mut target, proposal, budget - output.len(), &eos)
                .unwrap();
            assert!(!result.canonical_output_tokens.is_empty() || result.stop == "eos_pending");
            output.extend_from_slice(&result.canonical_output_tokens);
            assert_eq!(
                &output,
                &expected[..output.len()],
                "same-target prefix, family={}",
                request["family"]
            );
            rounds.push(json!({"diagnostic":diagnostic,"proposal_path":head_path,"top16":top16,
                "canonical_prefix":result.canonical_output_tokens,"first_rejection":result.first_rejection,
                "stop":result.stop,"start_consumed":result.start_consumed,"end_consumed":result.end_consumed,
                "serial_target_verification_s":result.serial_target_verification_s,"extra_cpu_math_audit_s":audit_s}));
            if result.stop == "length_pending" || result.stop == "eos_pending" {
                break;
            }
        }
        target.stream.synchronize().unwrap();
        let diagnostic_wall = started.elapsed().as_secs_f64();
        assert_eq!(output, expected);
        assert_eq!(target.consumed, cursor);
        assert_eq!(
            target.logits_snapshot().unwrap(),
            logits,
            "final target logits unchanged"
        );
        let accepted: usize = rounds
            .iter()
            .map(|v| {
                v["canonical_prefix"]
                    .as_array()
                    .unwrap()
                    .len()
                    .saturating_sub(1)
            })
            .sum();
        families.push(json!({"family":request["family"],"input_tokens":prompt.len(),"output_tokens":output,
            "rounds":rounds,"accepted_draft_tokens_excluding_anchor":accepted,
            "mean_content_tokens_per_round":expected.len() as f64/rounds.len() as f64,
            "baseline_native_generation_wall_s":baseline_wall,"diagnostic_generation_wall_s":diagnostic_wall,
            "performance_qualified":false,"wall_includes_d2h_hash_allocation_and_audit":true,
            "baseline_capture_enabled":true,"baseline_q8_draft_weights_resident":true,
            "same_target_output_cursor_and_logits_exact":true}));
    }
    // One actual canonical target history over2048 tokens checks the feature
    // window's nonzero absolute origin. This is NOT long-context qualification.
    let base_prompt = fixture["generation"][0]["canonical_prompt"]
        .as_str()
        .unwrap();
    let mut repeats = 192;
    let (long_text, long_prompt) = loop {
        let filler = "This is neutral context for a position test.\n".repeat(repeats);
        let text = base_prompt.replacen("用中文回答", &(filler + "用中文回答"), 1);
        let ids = tokenizer
            .encode(text.as_str(), false)
            .unwrap()
            .get_ids()
            .to_vec();
        if ids.len() > 2048 {
            break (text, ids);
        }
        repeats += 8;
        assert!(repeats <= 320);
    };
    assert!(long_prompt.len() <= 2304);
    let (expected, long_cursor, long_logits, _) = baseline(&mut target, &long_prompt, 12, &eos);
    target.reset_empty().unwrap();
    target.consume_sequence(&long_prompt).unwrap();
    let proposal = bridge.propose_diagnostic(&mut target).unwrap();
    assert_eq!(proposal.capture.context_len, 2048);
    assert_eq!(
        proposal.capture.context_start_position,
        long_prompt.len() - 2048
    );
    let long_capture = proposal.capture.clone();
    let long_diag = proposal.diagnostic.clone();
    let round = bridge
        .verify_serial(&mut target, proposal, 12, &eos)
        .unwrap();
    let prefix = round.canonical_output_tokens.clone();
    let mut long_output = prefix.clone();
    // Finish only with canonical target selections, matching the independent
    // no-DFlash endpoint including the pending EOS/length token state.
    while long_output.len() < 12 {
        let id = target.greedy_next_token().unwrap();
        if eos.contains(&id) {
            break;
        }
        long_output.push(id);
        if long_output.len() == 12 {
            break;
        }
        target.consume_selected().unwrap();
    }
    assert_eq!(long_output, expected);
    assert_eq!(target.consumed, long_cursor);
    assert_eq!(target.logits_snapshot().unwrap(), long_logits);
    let long_window = json!({"input_tokens":long_prompt.len(),"canonical_prompt_sha256":format!("{:x}",Sha256::digest(long_text.as_bytes())),
        "canonical_token_ids":long_prompt,"capture":long_capture,"diagnostic":long_diag,
        "first_round_canonical_prefix":prefix,"first_rejection":round.first_rejection,
        "same_target_final_ids_cursor_logits":true,"scope":"actual_2K_window_origin_not_256Ki_history"});
    // A reset invalidates an outstanding proposal even if the new prompt length
    // repeats the old cursor. This rejection happens before any target enqueue.
    let prompt = ids(&fixture["generation"][0]["token_ids"]);
    target.reset_empty().unwrap();
    target.consume_sequence(&prompt).unwrap();
    let stale = bridge.propose_diagnostic(&mut target).unwrap();
    target.reset_empty().unwrap();
    target.consume_sequence(&prompt).unwrap();
    let before_cursor = target.consumed;
    assert!(bridge.verify_serial(&mut target, stale, 8, &eos).is_err());
    assert_eq!(target.consumed, before_cursor);
    assert!(!target.poisoned);
    // Drop the bridge with rejected caller misuse, then exercise asynchronous
    // shared-head copy failure containment using a fresh immutable-weight lease.
    drop(bridge);
    let mut bridge = AugurDflashBridge::new(&target, &resident).unwrap();
    bridge.head.fail_after_copy = true;
    assert!(bridge.propose_diagnostic(&mut target).is_err());
    assert!(target.poisoned && bridge.poisoned);
    drop(bridge);
    drop(resident);
    drop(target);
    let after = cuda_device_allocation_snapshot();
    assert_eq!(after.live_bytes, before.live_bytes);
    assert_eq!(after.live_allocations, before.live_allocations);
    save(
        "result.json",
        &json!({"status":"PASS_AUGUR_Q8_SHARED_BF16_DIAGNOSTIC_BRIDGE",
        "target":target_info,"bridge":bridge_info,"independent_math":mathematical,
        "families":families,"actual_2048_window_probe":long_window,
        "peak_live_bytes":after.peak_live_bytes,
        "stale_epoch_rejected_without_target_mutation":true,
        "head_partial_copy_failure_drained_before_draft_drop":true,
        "cleanup_live_bytes_delta":after.live_bytes-before.live_bytes,
        "cleanup_live_allocations_delta":after.live_allocations-before.live_allocations,
        "fixture_sha256":format!("{:x}",Sha256::digest(&source)),
        "actual_long_history":false,"vision_loaded":false,"http_dflash_enabled":false,
        "fast_handoff_and_batched_verification_qualified":false}),
    );
}

#[test]
#[ignore = "exclusive QUASAR/Q8; deliberately corrupted proposal checks rejection isolation, NOT natural acceptance data"]
fn real_augur_q8_injected_first_rejection_keeps_canonical_pending_and_state() {
    use super::super::transfer_tests::{ids, open};
    let fixture_bytes =
        std::fs::read(std::env::var_os("AUGUR_CHECKPOINT_TARGET_REFERENCE").unwrap()).unwrap();
    let fixture: Value = serde_json::from_slice(&fixture_bytes).unwrap();
    let config_bytes = std::fs::read(std::env::var_os("AUGUR_DRAFT_CONFIG").unwrap()).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&config_bytes)),
        std::env::var("AUGUR_DRAFT_CONFIG_SHA256").unwrap()
    );
    let q8 = DFlashQ8Backbone::load_pinned(
        std::env::var_os("AUGUR_DRAFT_PATH").unwrap(),
        DFlashConfig::parse_f16_opt_in(&config_bytes).unwrap(),
    )
    .unwrap();
    assert_eq!(
        q8.manifest().file_sha256.as_ref().unwrap(),
        &std::env::var("AUGUR_DRAFT_SHA256").unwrap()
    );
    let before = cuda_device_allocation_snapshot();
    let mut target = open(&fixture);
    target.enable_dflash_feature_capture().unwrap();
    let weights = DFlashQ8CudaWeights::upload(&q8, 0).unwrap();
    let mut bridge = AugurDflashBridge::new(&target, &weights).unwrap();
    let prompt = ids(&fixture["generation"][0]["token_ids"]);
    let eos = ids(&fixture["eos_token_ids"]);
    let (expected, final_cursor, final_logits, _) = baseline(&mut target, &prompt, 24, &eos);
    assert!(expected.len() >= 2);
    target.reset_empty().unwrap();
    target.consume_sequence(&prompt).unwrap();
    let anchor = target.greedy_next_token().unwrap();
    target.consume_selected().unwrap();
    let correct = target.greedy_next_token().unwrap();
    let reference_logits = target.logits_snapshot().unwrap();
    let reference_cursor = target.consumed;
    let reference_features = target.dflash_feature_snapshot().unwrap().little_endian_f32;
    assert!(!eos.contains(&anchor) && !eos.contains(&correct));
    target.reset_empty().unwrap();
    target.consume_sequence(&prompt).unwrap();
    let mut proposal = bridge.propose_diagnostic(&mut target).unwrap();
    let actual_model_path = proposal.head.path.clone();
    // Mutation is reachable only inside this child test module. Public callers
    // receive read-only diagnostics and cannot forge a sealed proposal path.
    proposal.head.path[1] = eos[0];
    assert_ne!(proposal.head.path[1], correct);
    let round = bridge
        .verify_serial(&mut target, proposal, 24, &eos)
        .unwrap();
    assert_eq!(round.stop, "first_rejection_pending");
    assert_eq!(round.canonical_output_tokens, vec![anchor]);
    assert_eq!(
        round.first_rejection.as_ref().unwrap()["correct_target_token"],
        correct
    );
    assert_eq!(round.first_rejection.as_ref().unwrap()["offset"], 1);
    assert_eq!(target.consumed, reference_cursor);
    assert_eq!(target.greedy_next_token().unwrap(), correct);
    assert_eq!(target.logits_snapshot().unwrap(), reference_logits);
    assert_eq!(
        target.dflash_feature_snapshot().unwrap().little_endian_f32,
        reference_features
    );
    // The corrective canonical token becomes the next anchor exactly once;
    // an incorrect speculative EOS must not terminate or enter target KV/GDN.
    let next = bridge.propose_diagnostic(&mut target).unwrap();
    assert_eq!(next.head.path[0], correct);
    let next_round = bridge.verify_serial(&mut target, next, 23, &eos).unwrap();
    let mut output = round.canonical_output_tokens.clone();
    output.extend_from_slice(&next_round.canonical_output_tokens);
    while output.len() < 24 {
        let id = target.greedy_next_token().unwrap();
        if eos.contains(&id) {
            break;
        }
        output.push(id);
        if output.len() == 24 {
            break;
        }
        target.consume_selected().unwrap();
    }
    assert_eq!(output, expected);
    assert_eq!(target.consumed, final_cursor);
    assert_eq!(target.logits_snapshot().unwrap(), final_logits);
    drop(bridge);
    drop(weights);
    drop(target);
    let after = cuda_device_allocation_snapshot();
    assert_eq!(after.live_bytes, before.live_bytes);
    assert_eq!(after.live_allocations, before.live_allocations);
    save(
        "rejection.json",
        &json!({"status":"PASS_INJECTED_REJECTION_ISOLATION_NOT_NATURAL_ACCEPTANCE",
        "synthetic_injection":true,"include_in_training_or_acceptance_statistics":false,
        "unmodified_model_path":actual_model_path,"injected_candidate_offset":1,"injected_eos_token":eos[0],
        "rejection":round.first_rejection,"canonical_anchor":anchor,"canonical_next_anchor":correct,
        "state_logits_and_features_bitwise_equal_after_rejection":true,
        "no_premature_eos":true,"no_rejected_candidate_written":true,
        "final_output_cursor_logits_same_target":true,
        "cleanup_live_bytes_delta":after.live_bytes-before.live_bytes,
        "cleanup_live_allocations_delta":after.live_allocations-before.live_allocations}),
    );
}

#[test]
#[ignore = "exclusive full256Ki Augur + Q8; resident device handoff admission, not HTTP speed"]
fn real_augur_resident_device_bridge_equals_diagnostic_without_bulk_io() {
    use super::super::transfer_tests::{ids, open};
    use augur_backend::cuda_host_boundary_snapshot;
    let fixture: Value=serde_json::from_slice(&std::fs::read(std::env::var_os("AUGUR_CHECKPOINT_TARGET_REFERENCE").unwrap()).unwrap()).unwrap();
    let cfg=std::fs::read(std::env::var_os("AUGUR_DRAFT_CONFIG").unwrap()).unwrap();
    assert_eq!(format!("{:x}",Sha256::digest(&cfg)),std::env::var("AUGUR_DRAFT_CONFIG_SHA256").unwrap());
    let q8=DFlashQ8Backbone::load_pinned(std::env::var_os("AUGUR_DRAFT_PATH").unwrap(),DFlashConfig::parse_f16_opt_in(&cfg).unwrap()).unwrap();
    assert_eq!(q8.manifest().file_sha256.as_ref().unwrap(),&std::env::var("AUGUR_DRAFT_SHA256").unwrap());
    let before=cuda_device_allocation_snapshot();
    let mut target=open(&fixture);
    assert_eq!(target.capacity,262144);
    target.enable_dflash_feature_capture().unwrap();
    let weights=DFlashQ8CudaWeights::upload(&q8,0).unwrap();
    let mut bridge=AugurDflashBridge::new(&target,&weights).unwrap();
    let eos=ids(&fixture["eos_token_ids"]);
    let mut families=Vec::new();
    let mut rounds_checked=0;
    for request in fixture["generation"].as_array().unwrap() {
        let prompt=ids(&request["token_ids"]);
        let budget=request["max_new_tokens"].as_u64().unwrap() as usize;
        target.reset_empty().unwrap();target.consume_sequence(&prompt).unwrap();
        // Frozen diagnostic math, exact same target input; outside any timing.
        let diagnostic=bridge.propose_diagnostic(&mut target).unwrap();
        let expected_features=target.dflash_feature_snapshot().unwrap().little_endian_f32;
        let reference_head=HeadDiagnostics {path:diagnostic.head.path.clone(),candidate_ids:diagnostic.head.candidate_ids.clone(),
            unary_scores:diagnostic.head.unary_scores.clone(),transition_scores:diagnostic.head.transition_scores.clone(),
            logits:diagnostic.head.logits.clone(),normalized_hidden:diagnostic.head.normalized_hidden.clone()};
        bridge.discard(diagnostic).unwrap();
        let proposal=bridge.propose_resident(&mut target).unwrap();
        assert_eq!(proposal.path(),reference_head.path);
        let draft=bridge.resident.as_ref().unwrap().draft.as_ref().unwrap();
        assert!(draft.submitted_identity().is_err(),"owner identity must not impersonate a payload SHA");
        assert_eq!(draft.trajectory_input_features().unwrap().2,expected_features);
        let actual=bridge.head.read_diagnostics(proposal.path()[0]).unwrap();
        assert_eq!(actual.normalized_hidden,reference_head.normalized_hidden);
        assert_eq!(actual.logits,reference_head.logits);
        assert_eq!(actual.candidate_ids,reference_head.candidate_ids);
        assert_eq!(actual.unary_scores,reference_head.unary_scores);
        assert_eq!(actual.transition_scores,reference_head.transition_scores);
        bridge.discard_resident(proposal).unwrap();
        // Native generation repetitions: no optional diagnostic I/O in interval.
        let mut observations=Vec::new();
        for repeat in 0..3 {
            let (expected,cursor,logits,baseline_wall)=baseline(&mut target,&prompt,budget,&eos);
            target.reset_empty().unwrap();target.consume_sequence(&prompt).unwrap();
            let alloc_before=cuda_device_allocation_snapshot();
            let started=Instant::now();
            let mut output=Vec::new();let mut rounds=Vec::new();
            while output.len()<budget {
                let anchor=target.greedy_next_token().unwrap();
                if eos.contains(&anchor) { break; }
                let boundary=cuda_host_boundary_snapshot();
                let proposal=bridge.propose_resident(&mut target).unwrap();
                let delta=cuda_host_boundary_snapshot().checked_delta_since(boundary).unwrap();
                assert_eq!(delta.synchronous_h2d_bytes,(B*4) as u64);
                assert_eq!(delta.synchronous_d2h_bytes,(B*4) as u64);
                rounds_checked+=1;
                let result=bridge.verify_resident_serial(&mut target,proposal,budget-output.len(),&eos).unwrap();
                assert!(!result.canonical_output_tokens.is_empty() || result.stop=="eos_pending");
                output.extend_from_slice(&result.canonical_output_tokens);
                rounds.push(json!({"canonical_tokens":result.canonical_output_tokens,"first_rejection":result.first_rejection,
                    "stop":result.stop,"proposal_H2D_bytes":delta.synchronous_h2d_bytes,"proposal_D2H_bytes":delta.synchronous_d2h_bytes,
                    "proposal_host_waits":delta.host_waits}));
                if result.stop=="eos_pending" || result.stop=="length_pending" {break;}
            }
            target.stream.synchronize().unwrap();let resident_wall=started.elapsed().as_secs_f64();
            let alloc_after=cuda_device_allocation_snapshot();
            assert_eq!(alloc_after.successful_allocations,alloc_before.successful_allocations);
            assert_eq!(alloc_after.successful_frees,alloc_before.successful_frees);
            assert_eq!(output,expected);assert_eq!(target.consumed,cursor);
            assert_eq!(target.logits_snapshot().unwrap(),logits);
            observations.push(json!({"repeat":repeat,"output_tokens":output,"rounds":rounds,
                "no_draft_generation_s":baseline_wall,"resident_serial_generation_s":resident_wall,
                "GPU_allocations_during_generation":0,"optional_bulk_diagnostic_IO":false,
                "both_lanes_capture_and_Q8_workspace_resident":true}));
        }
        families.push(json!({"family":request["family"],"input_tokens":prompt.len(),"observations":observations,
            "same_input_features_hidden_logits_top16_scores_path_bitwise_equal":true}));
    }
    let info=bridge.resident_workspace_info();
    let peak=cuda_device_allocation_snapshot();
    // A reset invalidates an already-issued move-only resident proposal.
    target.reset_empty().unwrap();target.consume_sequence(&ids(&fixture["generation"][0]["token_ids"])).unwrap();
    let stale=bridge.propose_resident(&mut target).unwrap();
    target.reset_empty().unwrap();target.consume_sequence(&ids(&fixture["generation"][0]["token_ids"])).unwrap();
    assert!(bridge.verify_resident_serial(&mut target,stale,8,&eos).is_err());
    drop(bridge);
    // Failure after borrowed-source enqueue drains, poisons and cannot publish.
    let mut bridge=AugurDflashBridge::new(&target,&weights).unwrap();
    let first=bridge.propose_resident(&mut target).unwrap();bridge.discard_resident(first).unwrap();
    crate::dflash_q8_backbone_cuda::augur_owner::FAIL_AFTER_FEATURE_COPY.with(|flag|flag.set(true));
    assert!(bridge.propose_resident(&mut target).is_err());
    assert!(bridge.poisoned && target.poisoned);
    assert!(bridge.propose_resident(&mut target).is_err());
    drop(bridge);drop(weights);drop(target);
    let after=cuda_device_allocation_snapshot();
    assert_eq!(after.live_bytes,before.live_bytes);assert_eq!(after.live_allocations,before.live_allocations);
    save("resident.json",&json!({"status":"PASS_RESIDENT_OWNER_HANDOFF_NOT_PRODUCT_SPEEDUP",
        "workspace":info,"families":families,"compact_proposal_rounds_checked":rounds_checked,
        "stale_reset_and_partial_copy_failure_fail_closed":true,
        "admission_peak_owned_bytes_including_diagnostic_reference":peak.peak_live_bytes,
        "resident_steady_owned_bytes":peak.live_bytes-before.live_bytes,"vision_resident":false,
        "general_acceptance_qualified":false,"HTTP_qualified":false,"speculative_B8_verifier":false,
        "cleanup_live_bytes_delta":after.live_bytes-before.live_bytes,
        "cleanup_live_allocations_delta":after.live_allocations-before.live_allocations}));
}

#[test]
fn bridge_incarnations_are_distinct_even_for_same_target_and_generation() {
    let a=Rc::new(());let b=Rc::new(());let retained=Rc::clone(&a);
    assert!(!Rc::ptr_eq(&a,&b));assert!(Rc::ptr_eq(&a,&retained));
}
