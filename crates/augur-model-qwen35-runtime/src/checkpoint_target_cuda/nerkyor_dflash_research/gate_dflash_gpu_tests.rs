//! Real model combination test: same numerical-policy target-only oracle,
//! actual model-local Q8 candidates, and every canonical verifier boundary.
use super::*;
use crate::checkpoint_projection_cuda::CheckpointProjectionPolicy;

fn observe(
    target: &mut CheckpointCudaTarget,
    out: &Path,
    p: usize,
    step: usize,
    expected: &[Value],
    gate_inputs: &mut Vec<u8>,
    gate_outputs: &mut Vec<u8>,
    rows: &mut Vec<Value>,
) {
    assert_eq!(
        state(target),
        expected[step],
        "complete canonical logits/capture at p{p}/s{step}"
    );
    let logits = target.logits_snapshot().unwrap();
    write(
        out.join(format!("verified-p{p}-s{step}.f32le")),
        &bytes(&logits),
    );
    let mut layers = 0;
    for layer in &target.layers {
        if let Layer::Linear(layer) = layer {
            let input = layer.diagnostic_normalized_buffer();
            let mut raw = vec![0; input.len_bytes()];
            input.copy_to_host_sync(&target.stream, &mut raw).unwrap();
            gate_inputs.extend(raw);
            let (a, b) = layer.diagnostic_bf16_gate_outputs();
            for buffer in [a, b] {
                let mut raw = vec![0; buffer.len_bytes()];
                buffer.copy_to_host_sync(&target.stream, &mut raw).unwrap();
                gate_outputs.extend(raw);
            }
            layers += 1;
        }
    }
    assert_eq!(layers, 48);
    rows.push(json!({"prompt_index":p,"step":step,"selected_token":target.select_next_token().unwrap(),
        "consumed":target.consumed_tokens(),"logits_sha256":hash(&logits),"full_logits_capture_bitwise_equal":true}));
}

#[test]
#[ignore = "exclusive SM70 real FP32-gate target + own DFlash2, four short prompts and policy/epoch negatives"]
fn real_fp32_gate_dflash_policy_bound_four_prompts() {
    let guard = read(std::env::var_os("AUGUR_NERKYOR_GPU_GUARD").unwrap());
    assert_eq!(guard["gpu_compute_apps_empty"], true);
    assert_eq!(guard["queues_paused"], true);
    assert_eq!(guard["benchmark_auto_resume"], false);
    assert_eq!(
        cuda_device_properties(0).unwrap().compute_capability,
        CudaComputeCapability::SM70
    );
    for name in [
        "same-build-three-seeds-20261018",
        "priority-gpqa-seed1-20261018",
    ] {
        let v = read(
            PathBuf::from("F:/augur/evidence/benchmarks")
                .join(name)
                .join("status.json"),
        );
        assert_eq!(v["stage"], "paused_by_user");
        assert_eq!(v["automatic_resume"], false);
    }
    let out = PathBuf::from(std::env::var_os("AUGUR_NERKYOR_GPU_OUTPUT").unwrap());
    fs::create_dir(&out).unwrap();
    let model = PathBuf::from(std::env::var_os("AUGUR_NERKYOR_MODEL").unwrap());
    let draft = PathBuf::from(std::env::var_os("AUGUR_NERKYOR_DFLASH_Q8").unwrap());
    let reference = read(std::env::var_os("AUGUR_NERKYOR_DFLASH_REFERENCE").unwrap());
    let config = fs::read(draft.join("config.json")).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&config)),
        reference["config_sha256"].as_str().unwrap()
    );
    assert_eq!(
        Sha256::digest(fs::read(model.join("dflash/config.json")).unwrap()),
        Sha256::digest(&config)
    );
    let identity: CheckpointExpectedIdentity = serde_json::from_slice(
        &fs::read(std::env::var_os("AUGUR_NERKYOR_IDENTITY").unwrap()).unwrap(),
    )
    .unwrap();
    let fixture = read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/nerkyor-tokenizer-reference.json"),
    );
    for (name, sha) in fixture["files"].as_object().unwrap() {
        assert_eq!(
            format!("{:x}", Sha256::digest(fs::read(model.join(name)).unwrap())),
            sha.as_str().unwrap()
        );
    }
    let tokenizer =
        tokenizers::Tokenizer::from_bytes(&fs::read(model.join("tokenizer.json")).unwrap())
            .unwrap();
    let questions = [
        "Count from 1 to 20, separated by spaces. Output only the numbers.",
        "Write a Python function add(a, b) that returns their sum. Output code only.",
        "用中文简短说明：栈和队列的主要区别是什么？",
        "What is 17 * 23? Output only the answer.",
    ];
    let prompts: Vec<_> = questions
        .iter()
        .map(|q| {
            let text = format!(
                "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            );
            let ids = tokenizer
                .encode(text.as_str(), false)
                .unwrap()
                .get_ids()
                .to_vec();
            assert!(ids.len() + 24 < 128);
            (text, ids)
        })
        .collect();
    let started = Instant::now();
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
    assert!(cpu.embedded_selector_gate().is_none());
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
    assert!(target.enable_dflash_feature_capture().is_err());
    target
        .enable_nerkyor_dflash_research_capture(&weights)
        .unwrap();
    let binding = ResearchBinding::read(&target).unwrap();
    assert_eq!(binding.policy, ResearchPolicy::GateF32Cta);
    let mut decoder = NerkyorDflashResearchDecoder::new(&mut target, &weights).unwrap();
    record(out.join("target-info.json"), target.info().clone());
    record(
        out.join("input-contract.json"),
        json!({"prompts":prompts,"max_new_tokens":8,"eos_ids":[248046,248044],
        "semantic_benchmark":false,"target_only_oracle_same_policy":true,"candidate_source":"own DFlash2 Q8, uncalibrated"}),
    );
    let eos = [248046, 248044];
    let mut cases = Vec::new();
    let mut boundaries = Vec::new();
    let mut gate_inputs = Vec::new();
    let mut gate_outputs = Vec::new();
    for (p, (_, ids)) in prompts.iter().enumerate() {
        reset(&mut target, ids);
        let mut gold = Vec::new();
        let mut states = Vec::new();
        for step in 0..8 {
            let token = target.select_next_token().unwrap();
            gold.push(token);
            states.push(state(&mut target));
            write(
                out.join(format!("canonical-p{p}-s{step}.f32le")),
                &bytes(&target.logits_snapshot().unwrap()),
            );
            if eos.contains(&token) || step == 7 {
                break;
            }
            target.consume_selected().unwrap();
        }
        reset(&mut target, ids);
        let before = state(&mut target);
        let actual = decoder.propose(&mut target, &weights).unwrap();
        assert_eq!(
            state(&mut target),
            before,
            "propose cannot mutate target canonical state"
        );
        let path = actual.path.clone();
        assert!(decoder.propose(&mut target, &weights).is_err());
        assert_eq!(state(&mut target), before);
        if p == 0 {
            let input = target.capture_nerkyor_dflash_research_input().unwrap();
            let original_info = target.info.clone();
            let invalid_fields = [
                (
                    "policy",
                    json!(CheckpointProjectionPolicy::OriginalF32.label()),
                ),
                ("kv_dtype", json!("BF16")),
                ("profile", json!("quasar_nvfp4")),
            ];
            for (key, value) in invalid_fields {
                target.info[key] = value;
                let allocations = cuda_device_allocation_snapshot().successful_allocations;
                assert!(decoder.propose(&mut target, &weights).is_err());
                assert!(decoder
                    .verify_serial(&mut target, &actual, 8, &eos)
                    .is_err());
                assert!(NerkyorDflashResearchDecoder::new(&mut target, &weights).is_err());
                assert!(target.capture_nerkyor_dflash_research_input().is_err());
                assert!(input.bind_block(&mut target, &weights).is_err());
                assert_eq!(
                    allocations,
                    cuda_device_allocation_snapshot().successful_allocations
                );
                target.info = original_info.clone();
                assert_eq!(state(&mut target), before);
            }
            // Even a self-consistent replacement JSON/digest cannot replace the
            // installed capture's sealed source identity.
            let forged = ResearchPolicy::OriginalF32
                .contract(target._artifact.identity(), weights.file_sha256());
            target.info["policy"] = json!(CheckpointProjectionPolicy::OriginalF32.label());
            target.info["dflash_feature_capture"]["source_identity_contract"] = forged.clone();
            target.info["dflash_feature_capture"]["source_identity_sha256"] = json!(format!(
                "{:x}",
                Sha256::digest(serde_json::to_vec(&forged).unwrap())
            ));
            assert!(ResearchBinding::read(&target).is_err());
            target.info = original_info;
            for change in 0..3 {
                let mut foreign = NerkyorResearchProposal {
                    instance: Rc::clone(&actual.instance),
                    generation: actual.generation,
                    committed: actual.committed,
                    binding: actual.binding.clone(),
                    path: path.clone(),
                };
                match change {
                    0 => foreign.instance = Rc::new(()),
                    1 => foreign.binding.source_sha256[0] ^= 1,
                    _ => foreign.binding.policy = ResearchPolicy::OriginalF32,
                }
                assert!(decoder
                    .verify_serial(&mut target, &foreign, 8, &eos)
                    .is_err());
                assert_eq!(state(&mut target), before);
            }
            let saved = decoder.binding.clone();
            decoder.binding.source_sha256[0] ^= 1;
            assert!(decoder
                .verify_serial(&mut target, &actual, 8, &eos)
                .is_err());
            decoder.binding = saved;
            for (budget, stop) in [(0, vec![248046]), (8, vec![]), (8, vec![248320])] {
                assert!(decoder
                    .verify_serial(&mut target, &actual, budget, &stop)
                    .is_err());
                assert_eq!(state(&mut target), before);
            }
        }
        let allocations = cuda_device_allocation_snapshot().successful_allocations;
        let round = decoder
            .verify_serial_observed(&mut target, &actual, 8, &eos, |target, step| {
                observe(
                    target,
                    &out,
                    p,
                    step,
                    &states,
                    &mut gate_inputs,
                    &mut gate_outputs,
                    &mut boundaries,
                );
                Ok(())
            })
            .unwrap();
        check_round(&mut target, &round, &gold, &states);
        for step in round.output_tokens.len()..gold.len() {
            target.consume_selected().unwrap();
            assert_eq!(target.select_next_token().unwrap(), gold[step]);
            observe(
                &mut target,
                &out,
                p,
                step,
                &states,
                &mut gate_inputs,
                &mut gate_outputs,
                &mut boundaries,
            );
        }
        assert_eq!(
            allocations,
            cuda_device_allocation_snapshot().successful_allocations
        );
        let final_state = state(&mut target);
        assert_eq!(final_state, *states.last().unwrap());
        // Replay through the ordinary public verifier (no observer), proving
        // reset epoch renewal and diagnostic readback do not alter the round.
        reset(&mut target, ids);
        let replay = decoder.propose(&mut target, &weights).unwrap();
        assert_eq!(replay.path, path);
        let replay_round = decoder
            .verify_serial(&mut target, &replay, 8, &eos)
            .unwrap();
        assert_eq!(round.output_tokens, replay_round.output_tokens);
        assert_eq!(round.stop, replay_round.stop);
        check_round(&mut target, &replay_round, &gold, &states);
        let terminal = state(&mut target);
        assert!(decoder
            .verify_serial(&mut target, &replay, 8, &eos)
            .is_err());
        assert_eq!(state(&mut target), terminal);
        let mut controlled = Vec::new();
        if p == 0 {
            assert_eq!(gold.len(), 8);
            assert!(!gold.iter().any(|t| eos.contains(t)));
            for (name, rejection, budget, stops) in [
                ("all_seven_match", None, 9, eos.to_vec()),
                ("all_seven_match_at_budget", None, 8, eos.to_vec()),
                ("first_candidate_reject", Some(1), 8, eos.to_vec()),
                ("late_candidate_reject", Some(4), 8, eos.to_vec()),
                ("budget_one", None, 1, eos.to_vec()),
                ("budget_three", None, 3, eos.to_vec()),
                ("controlled_anchor_eos", None, 8, vec![gold[0]]),
                ("controlled_third_eos", None, 8, vec![gold[2]]),
            ] {
                reset(&mut target, ids);
                let mut proposal = decoder.propose(&mut target, &weights).unwrap();
                proposal.path = gold.clone();
                if let Some(i) = rejection {
                    proposal.path[i] = (gold[i] + 1) % 248320;
                }
                let r = decoder
                    .verify_serial(&mut target, &proposal, budget, &stops)
                    .unwrap();
                check_round(&mut target, &r, &gold, &states);
                if let Some(i) = rejection {
                    assert_eq!(r.rejection_offset, Some(i));
                    assert_eq!(r.stop, "rejection_fallback_pending");
                }
                if name == "all_seven_match" {
                    assert_eq!(r.matched_draft_tokens, 7);
                    assert_eq!(r.stop, "block_complete_pending");
                }
                if name == "all_seven_match_at_budget" {
                    assert_eq!(r.stop, "length_pending");
                }
                if name.starts_with("controlled_") {
                    assert_eq!(r.stop, "eos_pending");
                }
                controlled.push(round_json(name, &r, true));
            }
            for reset_epoch in [false, true] {
                reset(&mut target, ids);
                let captured = target.capture_nerkyor_dflash_research_input().unwrap();
                let stale = decoder.propose(&mut target, &weights).unwrap();
                if reset_epoch {
                    reset(&mut target, ids);
                } else {
                    target.consume_selected().unwrap();
                }
                let state_before = state(&mut target);
                let allocations = cuda_device_allocation_snapshot().successful_allocations;
                assert!(decoder.verify_serial(&mut target, &stale, 8, &eos).is_err());
                assert!(captured.bind_block(&mut target, &weights).is_err());
                assert_eq!(
                    allocations,
                    cuda_device_allocation_snapshot().successful_allocations
                );
                assert_eq!(state(&mut target), state_before);
                decoder.discard(stale).unwrap();
            }
        }
        let case = json!({"prompt_index":p,"canonical_tokens":gold,"decoded":tokenizer.decode(&gold,false).unwrap(),
            "canonical_states":states,"actual_candidate_path":path,"actual_round":round_json("real_uncalibrated_q8_candidates",&round,false),
            "reset_replay_same_candidates_and_canonical_state":true,"controlled_rounds":controlled});
        record(out.join(format!("case-{p}.json")), case.clone());
        cases.push(case);
    }
    let n = boundaries.len();
    assert_eq!(gate_inputs.len(), n * 48 * 5120 * 4);
    assert_eq!(gate_outputs.len(), n * 48 * 96 * 4);
    write(out.join("combined-gate-inputs.f32le"), &gate_inputs);
    write(out.join("combined-gate-outputs.f32le"), &gate_outputs);
    drop(decoder);
    drop(target);
    drop(weights);
    let freed = cuda_device_allocation_snapshot();
    assert_eq!(baseline.live_bytes, freed.live_bytes);
    assert_eq!(baseline.live_allocations, freed.live_allocations);
    record(
        out.join("result.json"),
        json!({"status":"PASS_POLICY_BOUND_FP32_GATE_DFLASH_SERIAL_FOUR_PROMPTS",
        "elapsed_s":started.elapsed().as_secs_f64(),"policy":"checkpoint_nerkyor_bf16_gate_f32_cta_research_v1",
        "binding_source_sha256":binding.source_sha256,"cases":cases,"boundaries":boundaries,"boundary_count":n,
        "real_candidate_rounds":8,"controlled_private_rounds":8,"gate_input_shape":[n,48,5120],"gate_output_shape":[n,48,96],
        "gate_input_sha256":format!("{:x}",Sha256::digest(&gate_inputs)),"gate_output_sha256":format!("{:x}",Sha256::digest(&gate_outputs)),
        "numerical_identity_negative_cases_passed":true,"epoch_cursor_stale_cases_passed":true,"allocation_cleanup_passed":true,
        "verifier_per_token_gpu_allocation":false,"fast_batched_verifier":false,"mtp_executed":false,"benchmark_resumed":false,
        "general_acceptance_rate_qualified":false,"end_to_end_speed_qualified":false,"public_admission_changed":false}),
    );
}
