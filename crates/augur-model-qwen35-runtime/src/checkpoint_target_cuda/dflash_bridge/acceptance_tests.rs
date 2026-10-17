//! New-target-only acceptance probes. No Session/HTTP publication authority.
//! Timed lanes never collect feature/head diagnostics. A separate exact replay
//! records canonical labels only through the first rejection, not beyond it.
use super::*;
use crate::DFlashQ8Backbone;
use augur_backend::cuda_host_boundary_snapshot;
use augur_model_qwen35::DFlashConfig;
use std::io::{BufRead, Write};

#[derive(Clone)]
enum Replay {
    Prompt(Vec<u32>),
    Canonical(Vec<u32>),
}
fn flatten(ops: &[Replay]) -> Vec<u32> {
    ops.iter()
        .flat_map(|op| match op {
            Replay::Prompt(ids) | Replay::Canonical(ids) => ids.iter().copied(),
        })
        .collect()
}
fn consume_prompt(target: &mut CheckpointCudaTarget, ids: &[u32]) {
    match TargetExecution::for_target(target).unwrap() {
        TargetExecution::LegacyQpn => target.consume_sequence(ids).unwrap(),
        TargetExecution::AugurRotatedQ8 => {
            let widths = ids.chunks(4352).map(|chunk| chunk.len()).collect();
            let plan = target.plan_augur_sequence(ids, widths).unwrap();
            let receipt = target.consume_sequence_augur(ids, &plan,
                super::super::augur_prefill::Budget::default(), |_| false).unwrap();
            assert_eq!(receipt.plan, plan);
        }
    }
}
fn consume_selected(target: &mut CheckpointCudaTarget) {
    TargetExecution::for_target(target).unwrap().consume_selected(target).unwrap();
}
fn restore(target: &mut CheckpointCudaTarget, ops: &[Replay]) -> f64 {
    target.reset_empty().unwrap();
    let started = Instant::now();
    for op in ops {
        match op {
            Replay::Prompt(ids) => consume_prompt(target, ids),
            Replay::Canonical(ids) => {
                for &id in ids {
                    assert_eq!(
                        target.greedy_next_token().unwrap(),
                        id,
                        "exact canonical replay must preserve B1/B8 numerical boundaries"
                    );
                    consume_selected(target);
                }
            }
        }
    }
    target.stream.synchronize().unwrap();
    assert_eq!(target.consumed, flatten(ops).len());
    started.elapsed().as_secs_f64()
}
fn root() -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var_os("AUGUR_BRIDGE_OUTPUT").unwrap())
}
fn save_bytes(name: &str, raw: &[u8]) -> String {
    let mut file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(root().join(name))
        .unwrap();
    file.write_all(raw).unwrap();
    file.sync_all().unwrap();
    format!("{:x}", Sha256::digest(raw))
}
fn save(name: &str, value: &Value) -> String {
    save_bytes(name, &serde_json::to_vec_pretty(value).unwrap())
}
fn hash_tokens(ids: &[u32]) -> String {
    let raw: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    format!("{:x}", Sha256::digest(raw))
}
struct Run {
    output: Vec<u32>,
    cursor: usize,
    logits: Vec<f32>,
    pending: u32,
    stop: &'static str,
    wire: Value,
}
fn no_draft(target: &mut CheckpointCudaTarget, budget: usize, eos: &[u32]) -> Run {
    let before = cuda_device_allocation_snapshot();
    let started = Instant::now();
    let mut output = Vec::new();
    let mut selected_at = Vec::new();
    let stop = loop {
        let id = target.greedy_next_token().unwrap();
        if eos.contains(&id) {
            break "eos";
        }
        output.push(id);
        selected_at.push(started.elapsed().as_secs_f64());
        if output.len() == budget {
            break "length";
        }
        consume_selected(target);
    };
    target.stream.synchronize().unwrap();
    let wall = started.elapsed().as_secs_f64();
    let after = cuda_device_allocation_snapshot();
    assert_eq!(after.successful_allocations, before.successful_allocations);
    assert_eq!(after.successful_frees, before.successful_frees);
    let pending = target.greedy_next_token().unwrap();
    Run {
        output,
        cursor: target.consumed,
        logits: target.logits_snapshot().unwrap(),
        pending,
        stop,
        wire: json!({"lane":"no_dflash","generation_wall_s":wall,"token_selection_elapsed_s":selected_at,
            "GPU_allocations":0,"GPU_frees":0,"optional_diagnostic_IO":false}),
    }
}
fn classify_rejection(head: &HeadDiagnostics, offset: usize, correct: u32) -> Value {
    assert!((1..B).contains(&offset));
    assert!((correct as usize) < V);
    let ids = &head.candidate_ids[offset * K..(offset + 1) * K];
    let unary = &head.unary_scores[offset * K..(offset + 1) * K];
    let transition = &head.transition_scores[offset * K..(offset + 1) * K];
    let rank = ids.iter().position(|id| *id == correct);
    let chosen = ids
        .iter()
        .position(|id| *id == head.path[offset])
        .expect("selector path must be in top16");
    let max_unary = unary.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    // Compact adaptation records have no full-vocabulary readback. A covered
    // label has its exact unary score; missing-candidate logits remain unknown.
    let correct_logit=head.logits.get(offset*V+correct as usize).copied().or_else(||rank.map(|i|unary[i]));
    json!({"offset":offset,"correct_target_token":correct,"proposed_token":head.path[offset],
        "correct_top16_rank":rank,"selected_top16_rank":chosen,
        "reason":if rank.is_some(){"selector_path_ranking"}else{"top16_coverage_missing"},
        "candidate_ids":ids,"unary_scores":unary,"selector_scores":transition,
        "correct_draft_head_logit":correct_logit,
        "correct_minus_best_unary":correct_logit.map(|x|x-max_unary),
        "correct_minus_selected_selector":rank.map(|i|transition[i]-transition[chosen]),
        "Q8_vs_FP16_cause_measured":false,"canonical_labels_after_this_rejection":false})
}
fn label_mask(canonical: &[u32], rejection: Option<u32>, eos: Option<u32>) -> Vec<Option<u32>> {
    assert!(canonical.len() <= B && !(rejection.is_some() && eos.is_some()));
    let mut labels: Vec<_> = canonical.iter().copied().map(Some).collect();
    if let Some(id) = rejection.or(eos) {
        assert!(labels.len() < B);
        labels.push(Some(id));
    }
    labels.resize(B, None);
    labels
}
fn resident(
    target: &mut CheckpointCudaTarget,
    bridge: &mut AugurDflashBridge,
    budget: usize,
    eos: &[u32],
    audit: bool,
    name: &str,
) -> Run {
    let before = cuda_device_allocation_snapshot();
    let started = Instant::now();
    let mut output = Vec::new();
    let mut rounds = Vec::new();
    let mut saved_features = false;
    let stop = loop {
        let round_begin = started.elapsed().as_secs_f64();
        let anchor = target.greedy_next_token().unwrap();
        if eos.contains(&anchor) {
            break "eos";
        }
        let boundary = cuda_host_boundary_snapshot();
        let proposal = bridge.propose_resident(target).unwrap();
        let proposal_end = started.elapsed().as_secs_f64();
        let delta = cuda_host_boundary_snapshot()
            .checked_delta_since(boundary)
            .unwrap();
        assert_eq!(delta.synchronous_h2d_bytes, 32);
        assert_eq!(delta.synchronous_d2h_bytes, 32);
        let path = proposal.path().to_vec();
        let extent = proposal.initialized_feature_range();
        let projected_context_rows = bridge.resident.as_ref().unwrap().draft.as_ref().unwrap().context_projection_rows();
        let head = audit.then(|| bridge.head.read_diagnostics(anchor).unwrap());
        if let Some(h) = &head {
            assert_eq!(path, h.path);
        }
        let start = output.len();
        let verified = bridge
            .verify_resident_serial(target, proposal, budget - start, eos)
            .unwrap();
        let round_end = started.elapsed().as_secs_f64();
        assert!(!verified.canonical_output_tokens.is_empty() || verified.stop == "eos_pending");
        let mut rejection = verified.first_rejection.clone();
        let mut collection = Value::Null;
        if let Some(head) = head {
            if let Some(reject) = &verified.first_rejection {
                let offset = reject["offset"].as_u64().unwrap() as usize;
                let correct = reject["correct_target_token"].as_u64().unwrap() as u32;
                rejection = Some(classify_rejection(&head, offset, correct));
            }
            let draft = bridge.resident.as_ref().unwrap().draft.as_ref().unwrap();
            let (feature_start, feature_rows, features) =
                draft.trajectory_input_features().unwrap();
            assert_eq!((feature_start, feature_start + feature_rows), extent);
            let feature_sha = format!("{:x}", Sha256::digest(&features));
            let retained = if rejection.is_some() && !saved_features {
                let filename = format!("{name}-first-rejection-features.f32le");
                assert_eq!(save_bytes(&filename, &features), feature_sha);
                saved_features = true;
                Some(filename)
            } else {
                None
            };
            let rejected_token = rejection
                .as_ref()
                .map(|r| r["correct_target_token"].as_u64().unwrap() as u32);
            let terminal_eos =
                (verified.stop == "eos_pending").then(|| target.greedy_next_token().unwrap());
            let labels = label_mask(
                &verified.canonical_output_tokens,
                rejected_token,
                terminal_eos,
            );
            collection = json!({"actual_feature_payload_sha256":feature_sha,"feature_bytes":features.len(),
                "retained_feature_file":retained,"initialized_feature_range":extent,
                "draft_query_absolute_start":extent.1,"seed_tokens":[anchor,DFLASH_SEED_MASK_TOKEN,DFLASH_SEED_MASK_TOKEN,DFLASH_SEED_MASK_TOKEN,
                    DFLASH_SEED_MASK_TOKEN,DFLASH_SEED_MASK_TOKEN,DFLASH_SEED_MASK_TOKEN,DFLASH_SEED_MASK_TOKEN],
                "proposal_path":path,"candidate_ids":head.candidate_ids,"unary_scores":head.unary_scores,
                "selector_scores":head.transition_scores,"canonical_target_labels":labels,
                "row0_is_anchor_not_acceptance_trial":true,"EOS_comparison_censored":terminal_eos.is_some(),
                "post_first_rejection_labels_masked":true,"old_target_samples":false});
        }
        rounds.push(json!({"output_start":start,"canonical_tokens":verified.canonical_output_tokens,
            "context_rows":extent.1-extent.0,"context_projected_rows":projected_context_rows,
            "context_reused_rows":extent.1-extent.0-projected_context_rows,
            "first_rejection":rejection,"stop":verified.stop,"start_consumed":verified.start_consumed,
            "end_consumed":verified.end_consumed,"wall_s":round_end-round_begin,
            "proposal_and_anchor_s":proposal_end-round_begin,"serial_verification_s":verified.serial_target_verification_s,
            "proposal_H2D_bytes":delta.synchronous_h2d_bytes,"proposal_D2H_bytes":delta.synchronous_d2h_bytes,
            "collection":collection}));
        output.extend_from_slice(&verified.canonical_output_tokens);
        if verified.stop == "eos_pending" {
            break "eos";
        }
        if verified.stop == "length_pending" {
            break "length";
        }
    };
    target.stream.synchronize().unwrap();
    let wall = started.elapsed().as_secs_f64();
    let after = cuda_device_allocation_snapshot();
    assert_eq!(after.successful_allocations, before.successful_allocations);
    assert_eq!(after.successful_frees, before.successful_frees);
    let pending = target.greedy_next_token().unwrap();
    Run {
        output,
        cursor: target.consumed,
        logits: target.logits_snapshot().unwrap(),
        pending,
        stop,
        wire: json!({"lane":if audit{"audit_replay_NOT_timing"}else{"resident"},"generation_wall_s":wall,
            "rounds":rounds,"GPU_allocations":0,"GPU_frees":0,"optional_diagnostic_IO":audit}),
    }
}
fn equal(actual: &Run, expected: &Run) {
    assert_eq!(actual.output, expected.output);
    assert_eq!(actual.cursor, expected.cursor);
    assert_eq!(actual.pending, expected.pending);
    assert_eq!(actual.stop, expected.stop);
    assert_eq!(
        actual.logits, expected.logits,
        "same numerical replay and final target state"
    );
}
fn accept_case(
    target: &mut CheckpointCudaTarget,
    bridge: &mut AugurDflashBridge,
    case: &Value,
    ops: &[Replay],
    eos: &[u32],
    already_initialized: bool,
) -> (Vec<u32>, u32, &'static str, Value) {
    let budget = usize::try_from(case["max_new_tokens"].as_u64().unwrap()).unwrap();
    assert!((1..=1024).contains(&budget));
    let input = flatten(ops);
    assert!(input.len() + budget + B <= 262144);
    let name = case["id"].as_str().unwrap();
    assert!(name
        .bytes()
        .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'));
    // Optional real same-owner continuation is compared with exact-operation
    // replay, NOT flattened B8 prefill of previously generated B1 history.
    let continuation = already_initialized.then(|| no_draft(target, budget, eos));
    let mut reference = continuation;
    let mut timed = Vec::new();
    for repeat in 0..2 {
        for lane in if repeat == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            let setup = restore(target, ops);
            let mut run = if lane {
                resident(target, bridge, budget, eos, false, name)
            } else {
                no_draft(target, budget, eos)
            };
            if let Some(expected) = &reference {
                equal(&run, expected);
            }
            run.wire["repeat"] = json!(repeat);
            run.wire["setup_exact_replay_wall_s"] = json!(setup);
            timed.push(run.wire.clone());
            if reference.is_none() {
                reference = Some(run);
            }
        }
    }
    let reference = reference.unwrap();
    restore(target, ops);
    let audit = resident(target, bridge, budget, eos, true, name);
    equal(&audit, &reference);
    // Collection replay must reproduce every first rejection from timed lanes.
    for run in timed.iter().filter(|r| r["lane"] == "resident") {
        let rows = run["rounds"].as_array().unwrap();
        let audits = audit.wire["rounds"].as_array().unwrap();
        assert_eq!(rows.len(), audits.len());
        for (a, b) in rows.iter().zip(audits) {
            assert_eq!(a["canonical_tokens"], b["canonical_tokens"]);
            assert_eq!(a["stop"], b["stop"]);
            for key in ["offset", "correct_target_token", "proposed_token"] {
                assert_eq!(a["first_rejection"][key], b["first_rejection"][key]);
            }
        }
    }
    let result = json!({"id":name,"family":case["family"],"thinking":case["thinking"],"input_tokens":input.len(),
        "input_token_ids":input,"input_u32le_sha256":hash_tokens(&input),"output_tokens":audit.output,
        "pending_token":audit.pending,"stop":audit.stop,"final_consumed_cursor":audit.cursor,
        "same_target_tokens_cursor_logits_exact":true,"same_owner_continuation_checked":already_initialized,
        "target_execution_profile":bridge.target_execution.label(),
        "exact_replay_operations":ops.iter().map(|op|match op {Replay::Prompt(ids)=>json!({
            "kind":if bridge.target_execution == TargetExecution::AugurRotatedQ8 {"prompt_B4352_literal_tail"} else {"prompt_B8"},"tokens":ids}),
            Replay::Canonical(ids)=>json!({"kind":"canonical_selected_B1","tokens":ids})}).collect::<Vec<_>>(),
        "timed":timed,"audit":audit.wire,"HTTP_or_Session_commit_authority":false});
    (audit.output, audit.pending, audit.stop, result)
}

#[test]
fn labels_stop_at_first_rejection_and_never_treat_anchor_as_a_trial() {
    assert_eq!(
        label_mask(&[7, 8], Some(9), None),
        vec![Some(7), Some(8), Some(9), None, None, None, None, None]
    );
    assert_eq!(
        label_mask(&[7, 8], None, Some(248046)),
        vec![Some(7), Some(8), Some(248046), None, None, None, None, None]
    );
    assert_eq!(
        label_mask(&[7, 8], None, None),
        vec![Some(7), Some(8), None, None, None, None, None, None]
    );
}
#[test]
#[ignore = "exclusive full256Ki QUASAR+Q8: English-thinking-first native acceptance, separate diagnostic replay"]
fn real_english_thinking_first_acceptance_and_separate_wall() {
    run_acceptance(false);
}

#[cfg(feature = "qwen35-augur-http")]
#[test]
#[ignore = "exclusive full256Ki current rotated-Q8 target; native DFlash baseline, not HTTP acceleration"]
fn real_current_rotated_target_acceptance_and_separate_wall() {
    run_acceptance(true);
}

fn run_acceptance(current_target: bool) {
    use super::super::transfer_tests::{ids, open};
    let raw = std::fs::read(std::env::var_os("AUGUR_ACCEPTANCE_INPUT").unwrap()).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&raw)),
        std::env::var("AUGUR_ACCEPTANCE_INPUT_SHA256").unwrap()
    );
    let probes: Value = serde_json::from_slice(&raw).unwrap();
    assert_eq!(probes["schema"], if current_target {
        "quasar-current-rotated-thinking-first-probes-v1"
    } else { "quasar-english-thinking-first-probes-v1" });
    let fixture: Value = serde_json::from_slice(
        &std::fs::read(std::env::var_os("AUGUR_CHECKPOINT_TARGET_REFERENCE").unwrap()).unwrap(),
    )
    .unwrap();
    assert_eq!(
        probes["tokenizer_sha256"],
        fixture["metadata"]["tokenizer.json"]["sha256"]
    );
    assert_eq!(
        probes["template_sha256"],
        fixture["metadata"]["chat_template.jinja"]["sha256"]
    );
    let cfg = std::fs::read(std::env::var_os("AUGUR_DRAFT_CONFIG").unwrap()).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&cfg)),
        std::env::var("AUGUR_DRAFT_CONFIG_SHA256").unwrap()
    );
    let q8 = DFlashQ8Backbone::load_pinned(
        std::env::var_os("AUGUR_DRAFT_PATH").unwrap(),
        DFlashConfig::parse_f16_opt_in(&cfg).unwrap(),
    )
    .unwrap();
    assert_eq!(
        q8.manifest().file_sha256.as_ref().unwrap(),
        &std::env::var("AUGUR_DRAFT_SHA256").unwrap()
    );
    let before = cuda_device_allocation_snapshot();
    let load_start = Instant::now();
    let mut target = open(&fixture);
    assert_eq!(target.capacity, 262144);
    if current_target {
        #[cfg(feature = "qwen35-augur-http")]
        target.enable_augur_decode().unwrap();
        target.enable_augur_feature_capture().unwrap();
        #[cfg(not(feature = "qwen35-augur-http"))]
        panic!("current-target test requires rotated decoder feature");
    } else {
        target.enable_dflash_feature_capture().unwrap();
    }
    let target_attention_mode = target.info()["attention_mode"].clone();
    if current_target {
        assert!(["tile8_b1", "gqa8_planar", "gqa8_c4_planar"].contains(&target_attention_mode.as_str().unwrap()));
        assert_eq!(probes["target_attention_mode"], target_attention_mode, "requested vs actual target attention");
    }
    let weights = DFlashQ8CudaWeights::upload(&q8, 0).unwrap();
    let mut bridge = AugurDflashBridge::new(&target, &weights).unwrap();
    let roots = probes["cases"].as_array().unwrap();
    // Explicit short component screen is not the full acceptance collection.
    if current_target && probes["probe_set"] == "projection_screen" {
        assert_eq!(roots.iter().map(|r|r["id"].as_str().unwrap()).collect::<Vec<_>>(),
            vec!["reason_probability","code_python","tool_add"]);
        for case in &roots[..2] { assert_eq!(case["max_new_tokens"],128); }
    } else {
        assert!(probes["probe_set"].is_null() || probes["probe_set"] == "full");
        assert_eq!(roots.len(), 8);
    }
    let eos = ids(&fixture["eos_token_ids"]);
    restore(&mut target, &[Replay::Prompt(ids(&roots[0]["token_ids"]))]);
    if current_target {
        // Exercise the real product's preallocated-but-previously-inactive path.
        bridge.retain_prefill_benchmark_workspace(&mut target).unwrap();
        assert!(target.consume_selected().is_err(), "legacy decoder must keep rejecting rotated KV");
    }
    let cold_before = cuda_device_allocation_snapshot();
    let cold = bridge.propose_resident(&mut target).unwrap();
    if current_target {
        let after = cuda_device_allocation_snapshot();
        assert_eq!(after.successful_allocations, cold_before.successful_allocations);
        assert_eq!(after.successful_frees, cold_before.successful_frees);
        assert_eq!(bridge.generation, 2, "unused startup seed is not a generation proposal");
    }
    bridge.discard_resident(cold).unwrap();
    let cold_load_and_workspace_s = load_start.elapsed().as_secs_f64();
    let mut results = Vec::new();
    let stdin = std::io::stdin();
    let mut control = stdin.lock();
    for root_case in roots {
        let mut case = root_case.clone();
        let mut ops = vec![Replay::Prompt(ids(&case["token_ids"]))];
        let mut initialized = false;
        loop {
            let (output, pending, stop, result) =
                accept_case(&mut target, &mut bridge, &case, &ops, &eos, initialized);
            let name = case["id"].as_str().unwrap().to_owned();
            let digest = save(&format!("{name}.json"), &result);
            results.push(json!({"id":name,"sha256":digest,"family":case["family"],"thinking":case["thinking"]}));
            println!(
                "AUGUR_ACCEPTANCE_CASE={}",
                json!({"id":name,"sha256":digest})
            );
            std::io::stdout().flush().unwrap();
            let mut line = String::new();
            assert!(
                control.read_line(&mut line).unwrap() > 0,
                "controller disappeared"
            );
            assert!(line.len() < 1024 * 1024);
            let command: Value = serde_json::from_str(&line).unwrap();
            if command["append"] == false {
                break;
            }
            assert_eq!(command["append"], true);
            assert_eq!(stop, "eos");
            assert_eq!(pending, 248046);
            let next = command["case"].clone();
            assert_eq!(next["parent_case"], name);
            let suffix = ids(&next["suffix_token_ids"]);
            assert!(!suffix.is_empty() && suffix.len() < 4096);
            // Consume the real pending EOS and only the newly rendered suffix.
            assert_eq!(target.greedy_next_token().unwrap(), pending);
            consume_selected(&mut target);
            consume_prompt(&mut target, &suffix);
            let mut committed = output;
            committed.push(pending);
            ops.push(Replay::Canonical(committed));
            ops.push(Replay::Prompt(suffix));
            assert_eq!(target.consumed, flatten(&ops).len());
            case = next;
            initialized = true;
        }
    }
    let info = bridge.resident_workspace_info();
    let peak = cuda_device_allocation_snapshot();
    drop(bridge);
    drop(weights);
    drop(target);
    let after = cuda_device_allocation_snapshot();
    assert_eq!(after.live_bytes, before.live_bytes);
    assert_eq!(after.live_allocations, before.live_allocations);
    save(
        "native-complete.json",
        &json!({"status":if current_target {"PASS_CURRENT_ROTATED_TARGET_NATIVE_ACCEPTANCE"} else {"PASS_ENGLISH_THINKING_FIRST_NATIVE_ACCEPTANCE"},"results":results,
        "current_product_target":current_target,"target_attention_mode":target_attention_mode,"prefill_max_batch":if current_target {4352} else {8},
        "cold_load_and_workspace_s":cold_load_and_workspace_s,"workspace":info,"target_capacity":262144,
        "owned_peak_bytes":peak.peak_live_bytes,"steady_owned_bytes":peak.live_bytes-before.live_bytes,
        "vision_loaded":false,"native_HTTP_or_Session_commits":false,"target_verifier":"canonical_B1",
        "cleanup_live_bytes_delta":after.live_bytes-before.live_bytes,"cleanup_live_allocations_delta":after.live_allocations-before.live_allocations}),
    );
}

#[test]
fn rejection_classifier_does_not_blame_selector_for_missing_coverage() {
    let mut head = HeadDiagnostics {
        path: vec![102; B],
        candidate_ids: vec![0; B * K],
        unary_scores: vec![0.; B * K],
        transition_scores: vec![0.; B * K],
        logits: vec![0.; B * V],
        normalized_hidden: Vec::new(),
    };
    for row in 0..B {
        for rank in 0..K {
            head.candidate_ids[row * K + rank] = 100 + rank as u32;
            head.unary_scores[row * K + rank] = 10. - rank as f32;
            head.transition_scores[row * K + rank] = 5. - rank as f32;
        }
    }
    head.transition_scores[K + 2] = 7.;
    head.transition_scores[K + 5] = 6.;
    let ranking = classify_rejection(&head, 1, 105);
    assert_eq!(ranking["reason"], "selector_path_ranking");
    assert_eq!(ranking["correct_top16_rank"], 5);
    assert_eq!(ranking["correct_minus_selected_selector"], -1.);
    let missing = classify_rejection(&head, 1, 200);
    assert_eq!(missing["reason"], "top16_coverage_missing");
    assert!(missing["correct_top16_rank"].is_null());
    assert!(missing["correct_minus_selected_selector"].is_null());
    assert_eq!(missing["Q8_vs_FP16_cause_measured"], false);
}

#[cfg(feature="qwen35-augur-http")]
mod batched;
