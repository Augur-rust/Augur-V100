//! Actual proposals + controlled stop/rejection branches; full committed state.
use super::batched::{compare, snapshot};
use super::*;
use crate::checkpoint_projection_cuda::CheckpointProjectionPolicy;
fn round_record(r: &NerkyorBatchedRound) -> Value {
    json!({"output_tokens":r.output_tokens,"matched_draft_tokens":r.matched_draft_tokens,"rejection_offset":r.rejection_offset,
        "stop":r.stop,"start_consumed":r.start_consumed,"end_consumed":r.end_consumed,
        "execution_identity_sha256":r.execution_identity_sha256,"used_batched_target":r.used_batched_target})
}
fn compare_snapshot(actual: &[(String, Vec<u8>)], expected: &[(String, Vec<u8>)]) -> Value {
    assert_eq!(actual.len(), expected.len());
    let groups: Vec<_> = actual
        .iter()
        .zip(expected)
        .map(|((name, a), (n, b))| {
            assert_eq!(name, n);
            json!({"name":name,"metric":compare(a,b)})
        })
        .collect();
    json!({"pass":groups.iter().all(|g|g["metric"]["pass"]==true),"groups":groups})
}
fn oracle(
    t: &mut CheckpointCudaTarget,
    prompt: &[u32],
    expected: &[u32],
) -> Vec<(String, Vec<u8>)> {
    reset(t, prompt);
    for (i, &id) in expected.iter().enumerate() {
        assert_eq!(
            t.select_next_token().unwrap(),
            id,
            "scalar greedy token {i}"
        );
        if i + 1 < expected.len() {
            t.consume_selected().unwrap();
        }
    }
    snapshot(t)
}
fn oracle_round(
    t: &mut CheckpointCudaTarget,
    prompt: &[u32],
    path: &[u32],
    budget: usize,
    eos: &[u32],
) -> (NerkyorSerialRound, Vec<(String, Vec<u8>)>) {
    reset(t, prompt);
    let start = t.consumed;
    let mut output = Vec::new();
    let mut matched = 0;
    let mut rejected = None;
    for offset in 0..8 {
        let correct = t.select_next_token().unwrap();
        assert!(offset > 0 || correct == path[0]);
        if offset > 0 {
            if correct == path[offset] {
                matched += 1;
            } else {
                rejected = Some(offset);
            }
        }
        output.push(correct);
        let stop = if eos.contains(&correct) {
            Some("eos_pending")
        } else if output.len() == budget {
            Some("length_pending")
        } else if correct != path[offset] {
            Some("rejection_fallback_pending")
        } else if offset == 7 {
            Some("block_complete_pending")
        } else {
            None
        };
        if let Some(stop) = stop {
            return (
                NerkyorSerialRound {
                    output_tokens: output,
                    matched_draft_tokens: matched,
                    rejection_offset: rejected,
                    stop,
                    start_consumed: start,
                    end_consumed: t.consumed,
                },
                snapshot(t),
            );
        }
        t.consume_selected().unwrap();
    }
    panic!("scalar oracle must terminate")
}
fn gold(t: &mut CheckpointCudaTarget, prompt: &[u32], limit: usize, eos: &[u32]) -> Vec<u32> {
    reset(t, prompt);
    let mut out = Vec::new();
    for _ in 0..limit {
        let id = t.select_next_token().unwrap();
        out.push(id);
        if eos.contains(&id) {
            break;
        }
        t.consume_selected().unwrap();
    }
    out
}
fn invalids(
    t: &mut CheckpointCudaTarget,
    d: &mut NerkyorDflashResearchDecoder,
    p: &NerkyorResearchProposal,
    eos: &[u32],
) {
    let before = state(t);
    let cursor = t.consumed;
    let selected = t.selected;
    let alloc = cuda_device_allocation_snapshot().successful_allocations;
    for (b, ids) in [(0, vec![248046]), (8, vec![]), (8, vec![248320])] {
        assert!(d.verify_batched(t, p, b, &ids).is_err());
    }
    assert!(d.verify_serial(t, p, 8, eos).is_err());
    assert!(d.release_batched_verifier().is_err());
    assert!(d.prepare_batched_verifier(t).is_err());
    for which in 0..7 {
        let mut bad = NerkyorResearchProposal {
            instance: Rc::clone(&p.instance),
            generation: p.generation,
            committed: p.committed,
            binding: p.binding.clone(),
            path: p.path.clone(),
        };
        match which {
            0 => bad.instance = Rc::new(()),
            1 => bad.generation += 1,
            2 => bad.committed.0 += 1,
            3 => bad.committed.1 += 1,
            4 => bad.binding.source_sha256[0] ^= 1,
            5 => {
                bad.path.pop();
            }
            _ => bad.path[0] = (bad.path[0] + 1) % 248320,
        }
        assert!(
            d.verify_batched(t, &bad, 8, eos).is_err(),
            "bad proposal {which}"
        );
    }
    let original = t.info.clone();
    let scalar =
        ResearchPolicy::GateF32Cta.contract(t._artifact.identity(), &d.binding.draft_sha256);
    t.info
        .as_object_mut()
        .unwrap()
        .remove("nerkyor_target_execution");
    t.info["dflash_feature_capture"]["source_identity_contract"] = scalar.clone();
    t.info["dflash_feature_capture"]["source_identity_sha256"] = json!(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&scalar).unwrap())
    ));
    assert!(d.verify_batched(t, p, 8, eos).is_err());
    assert!(t.capture_nerkyor_dflash_research_input().is_err());
    t.info = original;
    assert_eq!(cursor, t.consumed);
    assert_eq!(selected, t.selected);
    assert!(!t.poisoned);
    assert_eq!(before, state(t));
    assert_eq!(
        alloc,
        cuda_device_allocation_snapshot().successful_allocations
    );
}
#[allow(clippy::too_many_arguments)]
fn run_case(
    t: &mut CheckpointCudaTarget,
    d: &mut NerkyorDflashResearchDecoder,
    w: &DFlashQ8CudaWeights,
    out: &Path,
    prompt: &[u32],
    name: &str,
    path: Option<Vec<u32>>,
    budget: usize,
    eos: &[u32],
    negatives: bool,
) -> Value {
    reset(t, prompt);
    d.prepare_batched_verifier(t).unwrap();
    let mut proposal = d.propose(t, w).unwrap();
    let controlled = path.is_some();
    if let Some(path) = path {
        proposal.path = path;
    }
    if negatives {
        invalids(t, d, &proposal, eos);
    }
    let initial = state(t);
    let initial_capture = t
        .dflash_capture
        .as_ref()
        .unwrap()
        .committed_identity()
        .unwrap();
    let allocations = cuda_device_allocation_snapshot().successful_allocations;
    let start = Instant::now();
    let r = d.verify_batched(t, &proposal, budget, eos).unwrap();
    let verify_s = start.elapsed().as_secs_f64();
    assert_eq!(
        allocations,
        cuda_device_allocation_snapshot().successful_allocations
    );
    record(out.join(format!("{name}-round.json")), round_record(&r));
    assert_eq!(r.execution_identity_sha256, d.binding.source_sha256);
    assert_eq!(r.end_consumed, r.start_consumed + r.output_tokens.len() - 1);
    assert_eq!(
        t.select_next_token().unwrap(),
        *r.output_tokens.last().unwrap()
    );
    let mut raw = [0u8; 4];
    t.token.copy_to_host_sync(&t.stream, &mut raw).unwrap();
    assert_eq!(u32::from_le_bytes(raw), *r.output_tokens.last().unwrap());
    assert!(d.verify_batched(t, &proposal, budget, eos).is_err());
    if r.end_consumed == r.start_consumed {
        assert!(!r.used_batched_target);
        assert_eq!(initial, state(t));
        assert_eq!(
            initial_capture,
            t.dflash_capture
                .as_ref()
                .unwrap()
                .committed_identity()
                .unwrap()
        );
    }
    let actual = snapshot(t);
    d.release_batched_verifier().unwrap();
    let (reference, expected) = oracle_round(t, prompt, &proposal.path, budget, eos);
    assert_eq!(
        r.output_tokens, reference.output_tokens,
        "independent scalar output/length"
    );
    assert_eq!(r.stop, reference.stop, "independent scalar stop reason");
    assert_eq!(r.matched_draft_tokens, reference.matched_draft_tokens);
    assert_eq!(r.rejection_offset, reference.rejection_offset);
    assert_eq!(r.end_consumed, reference.end_consumed);
    let metrics = compare_snapshot(&actual, &expected);
    if r.end_consumed == r.start_consumed {
        assert!(metrics["groups"]
            .as_array()
            .unwrap()
            .iter()
            .all(|g| g["metric"]["bitwise_equal"] == true));
    }
    let result = json!({"name":name,"controlled_private_path":controlled,"proposal_path":proposal.path,"budget":budget,"eos":eos,
        "round":round_record(&r),"full_committed_state":metrics,"verify_wall_s_diagnostic_only":verify_s,"no_cuda_allocation_in_verify":true});
    record(out.join(format!("{name}-result.json")), result.clone());
    println!("VERIFIER_CASE={name} PASS={}", metrics["pass"]);
    result
}
#[test]
#[ignore = "exclusive SM70 sealed B8 real proposals, all stop branches, multi-round state oracle"]
fn real_sealed_b8_verifier_pending_rejection_eos_budget_and_continuation() {
    let guard = read(std::env::var_os("AUGUR_NERKYOR_GPU_GUARD").unwrap());
    assert_eq!(guard["gpu_compute_apps_empty"], true);
    assert_eq!(guard["queues_paused"], true);
    assert_eq!(guard["benchmark_auto_resume"], false);
    let out = PathBuf::from(std::env::var_os("AUGUR_NERKYOR_GPU_OUTPUT").unwrap());
    fs::create_dir(&out).unwrap();
    let model = PathBuf::from(std::env::var_os("AUGUR_NERKYOR_MODEL").unwrap());
    let draft = PathBuf::from(std::env::var_os("AUGUR_NERKYOR_DFLASH_Q8").unwrap());
    let reference = read(std::env::var_os("AUGUR_NERKYOR_DFLASH_REFERENCE").unwrap());
    let config = fs::read(draft.join("config.json")).unwrap();
    assert_eq!(
        Sha256::digest(&config),
        Sha256::digest(fs::read(model.join("dflash/config.json")).unwrap())
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(&config)),
        reference["config_sha256"].as_str().unwrap()
    );
    let identity: CheckpointExpectedIdentity = serde_json::from_slice(
        &fs::read(std::env::var_os("AUGUR_NERKYOR_IDENTITY").unwrap()).unwrap(),
    )
    .unwrap();
    let tokenizer =
        tokenizers::Tokenizer::from_bytes(&fs::read(model.join("tokenizer.json")).unwrap())
            .unwrap();
    let prompts:Vec<Vec<u32>>=["Count from 1 to 20, separated by spaces. Output only the numbers.",
        "Write a Python function add(a, b) that returns their sum.","用中文简短说明：栈和队列的主要区别是什么？",
        "What is 17 * 23? Output only the answer."].iter().map(|q|tokenizer.encode(format!("<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),false).unwrap().get_ids().to_vec()).collect();
    let baseline = cuda_device_allocation_snapshot();
    let cpu = DFlashQ8Backbone::load_pinned(
        draft.join("qwen35-27b-dflash2-q8_0.gguf"),
        DFlashConfig::parse(&config).unwrap(),
    )
    .unwrap();
    assert_eq!(
        cpu.manifest().file_sha256.as_deref(),
        reference["gguf_sha256"].as_str()
    );
    let weights = DFlashQ8CudaWeights::upload(&cpu, 0).unwrap();
    drop(cpu);
    let mut target = CheckpointCudaTarget::open_with_projection_policy(
        &model,
        Qwen35CheckpointProfile::NerkyorW4a16,
        identity,
        128,
        CheckpointProjectionPolicy::NerkyorBf16GateF32CtaResearch,
    )
    .unwrap();
    let allocation_before_bad_head = cuda_device_allocation_snapshot().successful_allocations;
    std::env::set_var("AUGUR_SHARED_HEAD_KERNEL", "rw4");
    assert!(target
        .enable_nerkyor_batched_dflash_research_capture(&weights)
        .is_err());
    assert!(target.dflash_capture.is_none());
    assert_eq!(
        allocation_before_bad_head,
        cuda_device_allocation_snapshot().successful_allocations
    );
    std::env::set_var("AUGUR_SHARED_HEAD_KERNEL", "llama");
    target
        .enable_nerkyor_batched_dflash_research_capture(&weights)
        .unwrap();
    let mut decoder = NerkyorDflashResearchDecoder::new(&mut target, &weights).unwrap();
    assert_eq!(decoder.binding.policy, ResearchPolicy::GateF32CtaB8);
    record(out.join("target-info.json"), target.info().clone());
    let eos = [248046, 248044];
    let mut results = Vec::new();
    let g = gold(&mut target, &prompts[0], 8, &eos);
    assert_eq!(g.len(), 8);
    for (p, prompt) in prompts.iter().enumerate() {
        results.push(run_case(
            &mut target,
            &mut decoder,
            &weights,
            &out,
            prompt,
            &format!("real-{p}"),
            None,
            8,
            &eos,
            p == 0,
        ));
    }
    for budget in 1..=8 {
        results.push(run_case(
            &mut target,
            &mut decoder,
            &weights,
            &out,
            &prompts[0],
            &format!("budget-{budget}"),
            Some(g.clone()),
            budget,
            &eos,
            false,
        ));
    }
    results.push(run_case(
        &mut target,
        &mut decoder,
        &weights,
        &out,
        &prompts[0],
        "full-block",
        Some(g.clone()),
        9,
        &eos,
        false,
    ));
    for at in 1..8 {
        let mut p = g.clone();
        p[at] = (p[at] + 137) % 248320;
        results.push(run_case(
            &mut target,
            &mut decoder,
            &weights,
            &out,
            &prompts[0],
            &format!("reject-{at}"),
            Some(p),
            9,
            &eos,
            false,
        ));
    }
    for at in [0, 1, 2, 6] {
        results.push(run_case(
            &mut target,
            &mut decoder,
            &weights,
            &out,
            &prompts[0],
            &format!("eos-{at}"),
            Some(g.clone()),
            9,
            &[g[at]],
            false,
        ));
    }
    let mut p = g.clone();
    p[2] = (p[2] + 137) % 248320;
    results.push(run_case(
        &mut target,
        &mut decoder,
        &weights,
        &out,
        &prompts[0],
        "eos-budget-rejection-priority",
        Some(p),
        3,
        &[g[2]],
        false,
    ));
    // Real multi-round generation, retaining the same B8 plan across rounds.
    for p in [0usize, 2] {
        reset(&mut target, &prompts[p]);
        decoder.prepare_batched_verifier(&mut target).unwrap();
        let mut generated = Vec::new();
        let mut rounds = Vec::new();
        while generated.len() < 24 {
            let proposal = decoder.propose(&mut target, &weights).unwrap();
            let allocations = cuda_device_allocation_snapshot().successful_allocations;
            let r = decoder
                .verify_batched(&mut target, &proposal, 24 - generated.len(), &eos)
                .unwrap();
            assert_eq!(
                allocations,
                cuda_device_allocation_snapshot().successful_allocations
            );
            generated.extend_from_slice(&r.output_tokens);
            rounds.push(round_record(&r));
            if eos.contains(generated.last().unwrap()) || generated.len() == 24 {
                break;
            }
            target.consume_selected().unwrap();
        }
        let actual = snapshot(&mut target);
        decoder.release_batched_verifier().unwrap();
        let expected = oracle(&mut target, &prompts[p], &generated);
        let metrics = compare_snapshot(&actual, &expected);
        let result = json!({"name":format!("multi-{p}"),"controlled_private_path":false,"tokens":generated,"rounds":rounds,"full_committed_state":metrics});
        record(out.join(format!("multi-{p}.json")), result.clone());
        results.push(result);
    }
    // Cursor staleness must reject before retirement, then explicit discard/release.
    reset(&mut target, &prompts[0]);
    decoder.prepare_batched_verifier(&mut target).unwrap();
    let stale = decoder.propose(&mut target, &weights).unwrap();
    target.consume_selected().unwrap();
    let before = state(&mut target);
    assert!(decoder
        .verify_batched(&mut target, &stale, 8, &eos)
        .is_err());
    assert_eq!(before, state(&mut target));
    decoder.discard(stale).unwrap();
    decoder.release_batched_verifier().unwrap();
    // Partial submission is unrecoverable without destroying the failed owner.
    reset(&mut target, &prompts[0]);
    decoder.prepare_batched_verifier(&mut target).unwrap();
    let proposal = decoder.propose(&mut target, &weights).unwrap();
    decoder.batched.as_mut().unwrap().inject_failure();
    assert!(decoder
        .verify_batched(&mut target, &proposal, 8, &eos)
        .is_err());
    assert!(target.poisoned && decoder.poisoned);
    assert!(target.consume_selected().is_err());
    assert!(target.logits_snapshot().is_err());
    assert!(decoder.release_batched_verifier().is_err());
    drop(decoder);
    reset(&mut target, &prompts[0]);
    drop(target);
    drop(weights);
    let freed = cuda_device_allocation_snapshot();
    assert_eq!(baseline.live_bytes, freed.live_bytes);
    assert_eq!(baseline.live_allocations, freed.live_allocations);
    let pass = results
        .iter()
        .all(|r| r["full_committed_state"]["pass"] == true);
    record(
        out.join("result.json"),
        json!({"status":if pass{"PASS_SEALED_B8_VERIFIER_AND_PENDING_CONTINUATION"}else{"FAIL_SEALED_B8_NUMERICAL_CONTRACT"},
        "cases":results,"allocation_cleanup":true,"invalid_inputs_rejected_without_mutation":true,"partial_failure_fail_closed":true,
        "product_admitted":false,"speed_qualified":false,"official_bf16_qualified":false,"independent_whole_path_f64_qualified":false}),
    );
    assert!(pass, "B8 verifier fixed numerical contract failed");
}
