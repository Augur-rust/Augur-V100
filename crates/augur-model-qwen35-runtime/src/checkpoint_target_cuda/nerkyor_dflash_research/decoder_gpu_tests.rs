//! Explicit full-model state oracle. Controlled paths below are private unit-test
//! interventions, never a public candidate injection API or acceptance estimate.
use super::*;
use crate::DFlashQ8Backbone;
use augur_backend::{
    cuda_device_allocation_snapshot, cuda_device_properties, CudaComputeCapability,
};
use augur_model_qwen35::{CheckpointExpectedIdentity, DFlashConfig, Qwen35CheckpointProfile};
use serde_json::{json, Value};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};
fn read(path: impl AsRef<Path>) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}
fn write(path: impl AsRef<Path>, bytes: &[u8]) {
    fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .unwrap()
        .write_all(bytes)
        .unwrap();
}
fn record(path: impl AsRef<Path>, v: Value) {
    write(path, &serde_json::to_vec_pretty(&v).unwrap());
}
fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn hash(v: &[f32]) -> String {
    assert!(v.iter().all(|v| v.is_finite()));
    format!("{:x}", Sha256::digest(bytes(v)))
}
fn state(target: &mut CheckpointCudaTarget) -> Value {
    let logits = target.logits_snapshot().unwrap();
    let features = target.capture_nerkyor_dflash_research_input().unwrap();
    json!({"consumed":target.consumed_tokens(),"logits_sha256":hash(&logits),"features_sha256":hash(features.features())})
}
fn reset(target: &mut CheckpointCudaTarget, prompt: &[u32]) {
    target.reset_empty().unwrap();
    target.consume_sequence(prompt).unwrap();
}
fn check_round(
    target: &mut CheckpointCudaTarget,
    round: &NerkyorSerialRound,
    gold: &[u32],
    states: &[Value],
) {
    let n = round.output_tokens.len();
    assert!((1..=8).contains(&n));
    assert_eq!(round.output_tokens, gold[..n]);
    assert_eq!(round.end_consumed, round.start_consumed + n - 1);
    assert_eq!(target.select_next_token().unwrap(), gold[n - 1]);
    assert_eq!(
        state(target),
        states[n - 1],
        "full canonical logits + capture state, including fallback/pending boundary"
    );
}
fn round_json(name: &str, r: &NerkyorSerialRound, controlled: bool) -> Value {
    json!({"name":name,"controlled_private_path":controlled,"output_tokens":r.output_tokens,"matched_draft_tokens":r.matched_draft_tokens,
        "rejection_offset":r.rejection_offset,"stop":r.stop,"start_consumed":r.start_consumed,"end_consumed":r.end_consumed,
        "full_logits_and_capture_state_equal_to_canonical":true})
}
fn head_metrics(a: &[f32], b: &[f32]) -> Value {
    assert_eq!(a.len(), 8 * 248320);
    assert_eq!(a.len(), b.len());
    let (mut e2, mut r2, mut max_error, mut peak) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for (&actual, &reference) in a.iter().zip(b) {
        assert!(actual.is_finite() && reference.is_finite());
        let d = f64::from(actual) - f64::from(reference);
        e2 += d * d;
        r2 += f64::from(reference).powi(2);
        max_error = max_error.max(d.abs());
        peak = peak.max(f64::from(reference).abs());
    }
    let relative = (e2 / r2).sqrt();
    let max_limit = 1e-4 + 1e-5 * peak;
    assert!(
        relative <= 1e-5 && max_error <= max_limit,
        "fixed head numerical policy failed"
    );
    let argmax = |row: &[f32]| {
        row.iter()
            .enumerate()
            .max_by(|(ia, a), (ib, b)| a.total_cmp(b).then_with(|| ib.cmp(ia)))
            .unwrap()
            .0
    };
    let equal = (0..8)
        .filter(|&i| {
            argmax(&a[i * 248320..(i + 1) * 248320]) == argmax(&b[i * 248320..(i + 1) * 248320])
        })
        .count();
    json!({"elements":a.len(),"relative_l2":relative,"relative_l2_limit":1e-5,"max_abs_error":max_error,"reference_peak":peak,
        "max_abs_limit":max_limit,"top1_equal_rows":equal,"all_finite":true})
}
#[test]
#[ignore = "exclusive SM70 + new-model target/draft/tokenizer; full shared head + canonical state oracle"]
fn real_nerkyor_shared_head_and_serial_verifier() {
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
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/nerkyor-tokenizer-reference.json"
    ))
    .unwrap();
    let mut tokenizer = None;
    for (name, sha) in fixture["files"].as_object().unwrap() {
        let bytes = fs::read(model.join(name)).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(&bytes)),
            sha.as_str().unwrap()
        );
        if name == "tokenizer.json" {
            tokenizer = Some(tokenizers::Tokenizer::from_bytes(&bytes).unwrap());
        }
    }
    let tokenizer = tokenizer.unwrap();
    // Own tokenizer, explicit non-thinking chat markers, no private benchmark data.
    let prompt="<|im_start|>user\nCount from 1 to 20, separated by spaces. Output only the numbers.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    let prompt_ids = tokenizer.encode(prompt, false).unwrap().get_ids().to_vec();
    assert!(prompt_ids.len() + 24 < 128);
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
    record(
        out.join("draft-ready.json"),
        json!({"elapsed_s":started.elapsed().as_secs_f64()}),
    );
    let mut target =
        CheckpointCudaTarget::open(&model, Qwen35CheckpointProfile::NerkyorW4a16, identity, 128)
            .unwrap();
    assert!(target.enable_dflash_feature_capture().is_err());
    target
        .enable_nerkyor_dflash_research_capture(&weights)
        .unwrap();
    let mut decoder = NerkyorDflashResearchDecoder::new(&mut target, &weights).unwrap();
    let loaded_s = started.elapsed().as_secs_f64();
    record(
        out.join("target-ready.json"),
        json!({"elapsed_s":loaded_s,"target_info":target.info(),"prompt_ids":prompt_ids}),
    );
    target.consume_sequence(&prompt_ids).unwrap();
    let eos = [248046, 248044];
    let mut gold = Vec::new();
    let mut states = Vec::new();
    for offset in 0..8 {
        let id = target.select_next_token().unwrap();
        assert!(
            !eos.contains(&id),
            "choose a longer nonprivate smoke prompt, never silently ignore an early model EOS"
        );
        gold.push(id);
        states.push(state(&mut target));
        if offset < 7 {
            target.consume_selected().unwrap();
        }
    }
    record(
        out.join("canonical.json"),
        json!({"prompt":prompt,"prompt_ids":prompt_ids,"tokens":gold,"states":states,
        "decoded":tokenizer.decode(&gold,false).unwrap(),"generation_eos":eos}),
    );
    reset(&mut target, &prompt_ids);
    let actual = decoder.propose(&mut target, &weights).unwrap();
    assert!(
        decoder.propose(&mut target, &weights).is_err(),
        "only one live proposal"
    );
    let actual_path = actual.path().to_vec();
    let before_audit = cuda_device_allocation_snapshot();
    let audit = decoder.audit_head(&actual, 15).unwrap();
    assert_eq!(
        before_audit.successful_allocations,
        cuda_device_allocation_snapshot().successful_allocations
    );
    let metrics = head_metrics(&audit.optimized_logits, &audit.scalar_logits);
    write(
        out.join("head-hidden.f32le"),
        &bytes(&audit.normalized_hidden),
    );
    write(
        out.join("head-optimized.f32le"),
        &bytes(&audit.optimized_logits),
    );
    write(out.join("head-scalar.f32le"), &bytes(&audit.scalar_logits));
    record(
        out.join("head-audit.json"),
        json!({"status":"PASS_FULL_SHARED_HEAD_VS_ORIGINAL_SCALAR_NUMERICAL_CONTRACT",
        "metrics":metrics,"optimized_kernel":audit.optimized_kernel,"timing_iterations":audit.timing_iterations,
        "optimized_gpu_ms":audit.optimized_gpu_ms,"scalar_gpu_ms":audit.scalar_gpu_ms,
        "component_speed_ratio":audit.scalar_gpu_ms/audit.optimized_gpu_ms,"end_to_end_speed_qualified":false,
        "hidden_sha256":hash(&audit.normalized_hidden),"optimized_sha256":hash(&audit.optimized_logits),"scalar_sha256":hash(&audit.scalar_logits)}),
    );
    let before_invalid = state(&mut target);
    for (budget, stop_ids) in [(0, vec![248046]), (8, vec![]), (8, vec![248320])] {
        assert!(decoder
            .verify_serial(&mut target, &actual, budget, &stop_ids)
            .is_err());
        assert_eq!(state(&mut target), before_invalid);
    }
    let foreign = NerkyorResearchProposal {
        instance: Rc::new(()),
        generation: actual.generation,
        committed: actual.committed,
        binding: actual.binding.clone(),
        path: actual.path.clone(),
    };
    assert!(decoder
        .verify_serial(&mut target, &foreign, 8, &eos)
        .is_err());
    assert_eq!(state(&mut target), before_invalid);
    let mut results = Vec::new();
    let round = decoder
        .verify_serial(&mut target, &actual, 8, &eos)
        .unwrap();
    check_round(&mut target, &round, &gold, &states);
    results.push(round_json("real_uncalibrated_q8_candidates", &round, false));
    record(out.join("real-round.json"), results.last().unwrap().clone());
    let terminal = state(&mut target);
    assert!(decoder
        .verify_serial(&mut target, &actual, 8, &eos)
        .is_err());
    assert_eq!(state(&mut target), terminal);
    // Complete actual output budget by canonical fallback, without reset. This
    // checks pending fallback carry and final full state against target-only.
    for offset in round.output_tokens.len()..8 {
        target.consume_selected().unwrap();
        assert_eq!(target.select_next_token().unwrap(), gold[offset]);
        assert_eq!(state(&mut target), states[offset]);
    }
    for (name, rejection, budget, stop_ids) in [
        // Keep block completion distinct from a simultaneous budget boundary.
        ("all_seven_match", None, 9, eos.to_vec()),
        ("all_seven_match_at_budget", None, 8, eos.to_vec()),
        ("first_candidate_reject", Some(1), 8, eos.to_vec()),
        ("late_candidate_reject", Some(4), 8, eos.to_vec()),
        ("budget_one", None, 1, eos.to_vec()),
        ("budget_three", None, 3, eos.to_vec()),
        ("controlled_anchor_eos", None, 8, vec![gold[0]]),
        ("controlled_third_eos", None, 8, vec![gold[2]]),
    ] {
        reset(&mut target, &prompt_ids);
        let mut proposal = decoder.propose(&mut target, &weights).unwrap();
        proposal.path = gold.clone();
        if let Some(i) = rejection {
            proposal.path[i] = (gold[i] + 1) % 248320;
        }
        let before = cuda_device_allocation_snapshot();
        let round = decoder
            .verify_serial(&mut target, &proposal, budget, &stop_ids)
            .unwrap();
        assert_eq!(
            before.successful_allocations,
            cuda_device_allocation_snapshot().successful_allocations
        );
        check_round(&mut target, &round, &gold, &states);
        if let Some(i) = rejection {
            assert_eq!(round.rejection_offset, Some(i));
            assert_eq!(round.output_tokens.len(), i + 1);
            assert_eq!(round.stop, "rejection_fallback_pending");
        }
        if name == "all_seven_match" {
            assert_eq!(round.matched_draft_tokens, 7);
            assert_eq!(round.output_tokens.len(), 8);
            assert_eq!(round.stop, "block_complete_pending");
        }
        if name == "all_seven_match_at_budget" {
            assert_eq!(round.matched_draft_tokens, 7);
            assert_eq!(round.output_tokens.len(), 8);
            assert_eq!(round.stop, "length_pending");
        }
        if name.starts_with("controlled_") {
            assert_eq!(round.stop, "eos_pending");
        }
        results.push(round_json(name, &round, true));
        record(
            out.join(format!("controlled-{name}.json")),
            results.last().unwrap().clone(),
        );
    }
    reset(&mut target, &prompt_ids);
    let stale = decoder.propose(&mut target, &weights).unwrap();
    target.consume_selected().unwrap();
    let before = state(&mut target);
    assert!(decoder.verify_serial(&mut target, &stale, 8, &eos).is_err());
    assert_eq!(state(&mut target), before);
    decoder.discard(stale).unwrap();
    reset(&mut target, &prompt_ids);
    let stale = decoder.propose(&mut target, &weights).unwrap();
    reset(&mut target, &prompt_ids);
    let before = state(&mut target);
    assert!(decoder.verify_serial(&mut target, &stale, 8, &eos).is_err());
    assert_eq!(state(&mut target), before);
    decoder.discard(stale).unwrap();
    // A second decoder identity is CPU-proven above, not a second 20GiB GPU owner.
    drop(decoder);
    drop(target);
    drop(weights);
    let freed = cuda_device_allocation_snapshot();
    assert_eq!(freed.live_bytes, baseline.live_bytes);
    assert_eq!(freed.live_allocations, baseline.live_allocations);
    record(
        out.join("result.json"),
        json!({"status":"PASS_REAL_NERKYOR_SHARED_HEAD_AND_CANONICAL_SERIAL_VERIFIER",
        "load_s":loaded_s,"elapsed_s":started.elapsed().as_secs_f64(),"actual_candidate_path":actual_path,"canonical_tokens":gold,"rounds":results,
        "real_candidate_rounds":1,"controlled_private_rounds":8,"canonical_full_logits_capture_equal":true,"fallback_carry_matches_canonical":true,
        "invalid_policy_proposal_cursor_epoch_rejected_before_mutation":true,"shared_original_bf16_head":true,"duplicate_head_weight_uploaded":false,
        "new_head_all_outputs_numerical_contract_passed":true,"allocation_cleanup_passed":true,
        "serial_verifier_never_writes_rejected_tokens":true,"batched_speculative_rollback_tested":false,
        "general_acceptance_rate_qualified":false,"selector_calibrated":false,"source_fp8_equivalence_qualified":false,
        "official_bf16_equivalence_qualified":false,"semantic_benchmark_qualified":false,"speed_qualified":false,
        "public_admission_changed":false,"session_publication_qualified":false,"mtp_uploaded":false,"mtp_executed":false,"benchmark_resumed":false}),
    );
}
#[test]
#[ignore = "exclusive SM70 actual target BF16 gate normalized input capture; diagnostic only"]
fn real_target_bf16_gate_inputs() {
    let guard = read(std::env::var_os("AUGUR_NERKYOR_GPU_GUARD").unwrap());
    assert_eq!(guard["gpu_compute_apps_empty"], true);
    assert_eq!(guard["queues_paused"], true);
    assert_eq!(guard["benchmark_auto_resume"], false);
    assert_eq!(
        cuda_device_properties(0).unwrap().compute_capability,
        CudaComputeCapability::SM70
    );
    let out = PathBuf::from(std::env::var_os("AUGUR_NERKYOR_GPU_OUTPUT").unwrap());
    fs::create_dir(&out).unwrap();
    let model = PathBuf::from(std::env::var_os("AUGUR_NERKYOR_MODEL").unwrap());
    let identity: CheckpointExpectedIdentity = serde_json::from_slice(
        &fs::read(std::env::var_os("AUGUR_NERKYOR_IDENTITY").unwrap()).unwrap(),
    )
    .unwrap();
    let fixture = read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/nerkyor-tokenizer-reference.json"),
    );
    let tokenizer_bytes = fs::read(model.join("tokenizer.json")).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&tokenizer_bytes)),
        fixture["files"]["tokenizer.json"].as_str().unwrap()
    );
    let tokenizer = tokenizers::Tokenizer::from_bytes(&tokenizer_bytes).unwrap();
    let prompt = "<|im_start|>user\nCount from 1 to 20, separated by spaces. Output only the numbers.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    let prompt_ids = tokenizer.encode(prompt, false).unwrap().get_ids().to_vec();
    let baseline = cuda_device_allocation_snapshot();
    let started = Instant::now();
    let mut target =
        CheckpointCudaTarget::open(&model, Qwen35CheckpointProfile::NerkyorW4a16, identity, 128)
            .unwrap();
    assert_eq!(target.info()["policy"], "checkpoint_original_f32_scalar_v1");
    target.consume_sequence(&prompt_ids).unwrap();
    let mut inputs = vec![0.0_f32; 48 * 8 * 5120];
    let mut positions = Vec::new();
    let mut layers = Vec::new();
    for step in 0..8 {
        let selected = target.select_next_token().unwrap();
        assert!(
            ![248046, 248044].contains(&selected),
            "early EOS: do not fabricate further generated positions"
        );
        let before = hash(&target.logits_snapshot().unwrap());
        let consumed = target.consumed_tokens();
        let mut slot = 0;
        for (layer_index, layer) in target.layers.iter().enumerate() {
            if let Layer::Linear(linear) = layer {
                let x = read_f32(&target.stream, linear.diagnostic_normalized_buffer()).unwrap();
                assert_eq!(x.len(), 5120);
                assert!(x.iter().all(|v| v.is_finite()));
                inputs[(slot * 8 + step) * 5120..(slot * 8 + step + 1) * 5120].copy_from_slice(&x);
                if step == 0 {
                    layers.push(layer_index);
                }
                slot += 1;
            }
        }
        assert_eq!(slot, 48);
        assert_eq!(hash(&target.logits_snapshot().unwrap()), before);
        assert_eq!(target.consumed_tokens(), consumed);
        positions.push(json!({"step":step,"consumed":consumed,"selected_pending":selected,"logits_sha256":before}));
        if step < 7 {
            target.consume_selected().unwrap();
        }
    }
    assert_eq!(
        layers,
        (0..64).filter(|l| (l + 1) % 4 != 0).collect::<Vec<_>>()
    );
    write(out.join("target-gate-inputs.f32le"), &bytes(&inputs));
    drop(target);
    let freed = cuda_device_allocation_snapshot();
    assert_eq!(freed.live_bytes, baseline.live_bytes);
    assert_eq!(freed.live_allocations, baseline.live_allocations);
    record(
        out.join("result.json"),
        json!({
            "status":"PASS_REAL_TARGET_BF16_GATE_INPUT_CAPTURE", "elapsed_s":started.elapsed().as_secs_f64(),
            "prompt":prompt, "prompt_ids":prompt_ids, "positions":positions, "layers":layers,
            "shape":[48,8,5120], "input_sha256":hash(&inputs), "finite":true,
            "actual_input_rmsnorm_output_before_a_b_projections":true,
            "readback_preserves_canonical_logits_and_cursor":true, "allocation_cleanup_passed":true,
            "mtp_uploaded":false, "mtp_executed":false, "draft_loaded":false, "runtime_admission_changed":false
        }),
    );
}

/// Target-only differential gate policy probe; deliberately not admitted to the
/// DFlash capture policy or any product/session surface. No timing lane here.
#[test]
#[ignore = "exclusive SM70 two sequential authenticated targets, isolated F32 gate experiment"]
fn real_nerkyor_bf16_gate_f32_cta_full_logits() {
    use crate::checkpoint_projection_cuda::CheckpointProjectionPolicy;
    let guard = read(std::env::var_os("AUGUR_NERKYOR_GPU_GUARD").unwrap());
    assert_eq!(guard["gpu_compute_apps_empty"], true);
    assert_eq!(guard["queues_paused"], true);
    assert_eq!(guard["benchmark_auto_resume"], false);
    assert_eq!(
        cuda_device_properties(0).unwrap().compute_capability,
        CudaComputeCapability::SM70
    );
    let out = PathBuf::from(std::env::var_os("AUGUR_NERKYOR_GPU_OUTPUT").unwrap());
    fs::create_dir(&out).unwrap();
    let model = PathBuf::from(std::env::var_os("AUGUR_NERKYOR_MODEL").unwrap());
    let identity: CheckpointExpectedIdentity = serde_json::from_slice(
        &fs::read(std::env::var_os("AUGUR_NERKYOR_IDENTITY").unwrap()).unwrap(),
    )
    .unwrap();
    let policy = read(std::env::var_os("AUGUR_NERKYOR_GATE_POLICY").unwrap());
    assert_eq!(policy["relative_l2_limit"].as_f64(), Some(1e-5));
    assert_eq!(policy["absolute_floor"].as_f64(), Some(1e-4));
    assert_eq!(
        policy["max_abs_over_reference_peak_limit"].as_f64(),
        Some(1e-5)
    );
    assert_eq!(policy["declared_before_gpu_output"], true);
    let fixture = read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/nerkyor-tokenizer-reference.json"),
    );
    let tokenizer_bytes = fs::read(model.join("tokenizer.json")).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&tokenizer_bytes)),
        fixture["files"]["tokenizer.json"].as_str().unwrap()
    );
    let tokenizer = tokenizers::Tokenizer::from_bytes(&tokenizer_bytes).unwrap();
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
            assert!(!ids.is_empty() && ids.len() + 8 <= 128);
            (text, ids)
        })
        .collect();
    record(
        out.join("input-contract.json"),
        json!({"prompts":prompts,"max_new_tokens":8,"eos_ids":[248046,248044],"tokenizer_sha256":fixture["files"]["tokenizer.json"],"semantic_benchmark":false}),
    );
    let before = cuda_device_allocation_snapshot();
    let started = Instant::now();
    let mut gold: Vec<Vec<(u32, Vec<f32>)>> = Vec::new();
    let mut loads = Vec::new();
    {
        let load = Instant::now();
        let mut target = CheckpointCudaTarget::open(
            &model,
            Qwen35CheckpointProfile::NerkyorW4a16,
            identity.clone(),
            128,
        )
        .unwrap();
        loads.push(load.elapsed().as_secs_f64());
        assert_eq!(target.info()["policy"], "checkpoint_original_f32_scalar_v1");
        record(out.join("original-target-info.json"), target.info().clone());
        for (p, (_, ids)) in prompts.iter().enumerate() {
            target.reset_empty().unwrap();
            target.consume_sequence(ids).unwrap();
            let mut rows = Vec::new();
            for step in 0..8 {
                assert_eq!(target.consumed_tokens(), ids.len() + step);
                let token = target.select_next_token().unwrap();
                let logits = target.logits_snapshot().unwrap();
                write(
                    out.join(format!("original-p{p}-s{step}.f32le")),
                    &bytes(&logits),
                );
                rows.push((token, logits));
                if token == 248046 || token == 248044 {
                    break;
                }
                if step < 7 {
                    target.consume_selected().unwrap();
                }
            }
            record(
                out.join(format!("original-p{p}.json")),
                json!({"tokens":rows.iter().map(|r|r.0).collect::<Vec<_>>(),"boundaries":rows.len()}),
            );
            gold.push(rows);
        }
    }
    let dropped = cuda_device_allocation_snapshot();
    assert_eq!(dropped.live_bytes, before.live_bytes);
    assert_eq!(dropped.live_allocations, before.live_allocations);
    let candidate_policy = CheckpointProjectionPolicy::NerkyorBf16GateF32CtaResearch;
    // Wrong-profile and KV routes fail before artifact opening / GPU allocation.
    assert!(CheckpointCudaTarget::open_with_projection_policy(
        &model,
        Qwen35CheckpointProfile::QuasarNvfp4,
        identity.clone(),
        128,
        candidate_policy
    )
    .is_err());
    let mut gates_inputs = Vec::new();
    let mut gates_outputs = Vec::new();
    let mut cases = Vec::new();
    let mut all_ok = true;
    {
        let load = Instant::now();
        let mut target = CheckpointCudaTarget::open_with_projection_policy(
            &model,
            Qwen35CheckpointProfile::NerkyorW4a16,
            identity,
            128,
            candidate_policy,
        )
        .unwrap();
        loads.push(load.elapsed().as_secs_f64());
        assert_eq!(target.info()["policy"], candidate_policy.label());
        assert_eq!(target.info()["activation_dtype"], "F32");
        assert_eq!(target.info()["kv_dtype"], "F32");
        assert_eq!(target.info()["small_gate_pair_count"], 48);
        assert_eq!(target.info()["nerkyor_gate_f32_cta_research"], true);
        record(
            out.join("candidate-target-info.json"),
            target.info().clone(),
        );
        let after_load = cuda_device_allocation_snapshot();
        for (p, (_, ids)) in prompts.iter().enumerate() {
            target.reset_empty().unwrap();
            target.consume_sequence(ids).unwrap();
            let mut rows = Vec::new();
            let mut last_hash = String::new();
            for (step, (token, reference)) in gold[p].iter().enumerate() {
                assert_eq!(target.consumed_tokens(), ids.len() + step);
                let selected = target.select_next_token().unwrap();
                let logits = target.logits_snapshot().unwrap();
                write(
                    out.join(format!("candidate-p{p}-s{step}.f32le")),
                    &bytes(&logits),
                );
                let mut e2 = 0.0_f64;
                let mut r2 = 0.0_f64;
                let mut peak = 0.0_f64;
                let mut max_abs = 0.0_f64;
                let finite =
                    logits.iter().all(|x| x.is_finite()) && reference.iter().all(|x| x.is_finite());
                for (&a, &b) in logits.iter().zip(reference) {
                    let d = f64::from(a) - f64::from(b);
                    e2 += d * d;
                    r2 += f64::from(b).powi(2);
                    peak = peak.max(f64::from(b).abs());
                    max_abs = max_abs.max(d.abs());
                }
                let relative = (e2 / r2.max(f64::MIN_POSITIVE)).sqrt();
                let absolute_limit = 1e-4 + 1e-5 * peak;
                let passed =
                    finite && relative <= 1e-5 && max_abs <= absolute_limit && selected == *token;
                all_ok &= passed;
                last_hash = hash(&logits);
                let mut layer_count = 0;
                for layer in &target.layers {
                    if let Layer::Linear(layer) = layer {
                        for buffer in [layer.diagnostic_normalized_buffer()] {
                            let mut raw = vec![0u8; buffer.len_bytes()];
                            buffer.copy_to_host_sync(&target.stream, &mut raw).unwrap();
                            gates_inputs.extend(raw);
                        }
                        let (a, b) = layer.diagnostic_bf16_gate_outputs();
                        for buffer in [a, b] {
                            let mut raw = vec![0u8; buffer.len_bytes()];
                            buffer.copy_to_host_sync(&target.stream, &mut raw).unwrap();
                            gates_outputs.extend(raw);
                        }
                        layer_count += 1;
                    }
                }
                assert_eq!(layer_count, 48);
                // Diagnostic readback must not alter logits or the pending selection.
                assert_eq!(hash(&target.logits_snapshot().unwrap()), last_hash);
                assert_eq!(target.select_next_token().unwrap(), selected);
                rows.push(json!({"step":step,"elements":logits.len(),"relative_l2":relative,"max_abs_error":max_abs,"reference_peak":peak,"absolute_limit":absolute_limit,"finite":finite,"reference_token":token,"selected_token":selected,"passed":passed}));
                if step + 1 < gold[p].len() {
                    target.consume(*token, true, &[]).unwrap();
                }
            }
            // Canonical history is forced only to collect diagnostics on failure;
            // equal selections throughout imply the same autoregressive history.
            target.reset_empty().unwrap();
            target.consume_sequence(ids).unwrap();
            for (token, _) in gold[p].iter().take(gold[p].len() - 1) {
                target.consume(*token, true, &[]).unwrap();
            }
            assert_eq!(hash(&target.logits_snapshot().unwrap()), last_hash);
            assert_eq!(target.consumed_tokens(), ids.len() + gold[p].len() - 1);
            let case =
                json!({"prompt_index":p,"boundaries":rows,"reset_replay_bitwise_equal":true});
            record(out.join(format!("candidate-p{p}.json")), case.clone());
            cases.push(case);
        }
        assert_eq!(
            cuda_device_allocation_snapshot().successful_allocations,
            after_load.successful_allocations
        );
    }
    write(out.join("runtime-gate-inputs.f32le"), &gates_inputs);
    write(out.join("runtime-gate-outputs.f32le"), &gates_outputs);
    let freed = cuda_device_allocation_snapshot();
    assert_eq!(freed.live_bytes, before.live_bytes);
    assert_eq!(freed.live_allocations, before.live_allocations);
    let boundaries: usize = gold.iter().map(Vec::len).sum();
    assert_eq!(gates_inputs.len(), boundaries * 48 * 5120 * 4);
    assert_eq!(gates_outputs.len(), boundaries * 48 * 96 * 4);
    record(
        out.join("result.json"),
        json!({"status":if all_ok {"PASS_RESEARCH_GATE_RUNTIME_FULL_LOGITS_AND_TOKENS"} else {"FAIL_RESEARCH_GATE_RUNTIME_NUMERICAL_CONTRACT"},
        "elapsed_s":started.elapsed().as_secs_f64(),"load_s":loads,"policy":candidate_policy.label(),"prompt_count":prompts.len(),"boundaries":boundaries,"cases":cases,
        "gate_input_shape":[boundaries,48,5120],"gate_output_shape":[boundaries,48,96],"gate_input_sha256":format!("{:x}",Sha256::digest(&gates_inputs)),"gate_output_sha256":format!("{:x}",Sha256::digest(&gates_outputs)),
        "full_logits_contract_passed":all_ok,"all_greedy_selections_equal":all_ok,"reset_replay_bitwise_equal":true,"per_token_gpu_allocation":false,"allocation_cleanup_passed":true,
        "official_bf16_equivalence_qualified":false,"semantic_benchmark_qualified":false,"end_to_end_speed_qualified":false,"dflash_loaded":false,"mtp_executed":false,"public_admission_changed":false,"benchmark_resumed":false}),
    );
    assert!(all_ok,"isolated F32 gate runtime failed the predeclared full-logits/token contract; do not admit candidate");
}

#[path = "gate_dflash_gpu_tests.rs"]
mod gate_dflash;

#[cfg(feature = "qwen35-prefill-cublas")]
#[path = "batched_gpu_tests.rs"]
mod batched;

#[cfg(feature = "qwen35-prefill-cublas")]
#[path = "verifier_gpu_tests.rs"]
mod verifier;

#[cfg(feature = "qwen35-prefill-cublas")]
#[path = "resident_gpu_tests.rs"]
mod resident;

#[cfg(feature = "qwen35-prefill-cublas")]
#[path = "prefill_gpu_tests.rs"]
mod prefill;
