//! Model-local candidate teacher forcing: scalar oracle vs independent B8.
//! This is not yet the public speculative acceptance loop or a speed claim.
use super::*;
use crate::checkpoint_projection_cuda::CheckpointProjectionPolicy;
use crate::checkpoint_target_cuda::nerkyor_dflash_research::batched::Batch;
pub(super) fn snapshot(t: &mut CheckpointCudaTarget) -> Vec<(String, Vec<u8>)> {
    let mut rows = vec![
        ("logits".into(), bytes(&t.logits_snapshot().unwrap())),
        (
            "capture".into(),
            t.dflash_capture
                .as_ref()
                .unwrap()
                .snapshot()
                .unwrap()
                .little_endian_f32,
        ),
    ];
    for (i, l) in t.layers.iter().enumerate() {
        let state = match l {
            Layer::Linear(l) => {
                crate::checkpoint_linear_cuda::nerkyor_batch::diagnostic_state(l).unwrap()
            }
            Layer::Full(l) => {
                crate::checkpoint_full_cuda::nerkyor_batch::diagnostic_state(l).unwrap()
            }
        };
        for (name, raw) in state {
            rows.push((format!("layer-{i}-{name}"), raw));
        }
    }
    rows
}
pub(super) fn compare(a: &[u8], b: &[u8]) -> Value {
    assert_eq!(a.len(), b.len());
    let (mut e2, mut r2, mut max, mut peak) = (0f64, 0f64, 0f64, 0f64);
    for (x, y) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        let x = f64::from(f32::from_le_bytes(x.try_into().unwrap()));
        let y = f64::from(f32::from_le_bytes(y.try_into().unwrap()));
        assert!(x.is_finite() && y.is_finite());
        let d = x - y;
        e2 += d * d;
        r2 += y * y;
        max = max.max(d.abs());
        peak = peak.max(y.abs());
    }
    let rel = if r2 == 0. {
        if e2 == 0. {
            0.
        } else {
            f64::INFINITY
        }
    } else {
        (e2 / r2).sqrt()
    };
    json!({"elements":a.len()/4,"bitwise_equal":a==b,"relative_l2":rel,"max_abs":max,"reference_peak":peak,
        "pass":rel<=1e-5 && max<=1e-4+1e-5*peak})
}
#[test]
#[ignore = "exclusive SM70 complete B8 target and all prefix states; research only"]
fn real_gate_f32_b8_target_all_prefixes() {
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
    let tokenizer =
        tokenizers::Tokenizer::from_bytes(&fs::read(model.join("tokenizer.json")).unwrap())
            .unwrap();
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
    target
        .enable_nerkyor_dflash_research_capture(&weights)
        .unwrap();
    let mut decoder = NerkyorDflashResearchDecoder::new(&mut target, &weights).unwrap();
    let mut all = Vec::new();
    let mut last_prompt = Vec::new();
    for (p, q) in [
        "Count from 1 to 20, separated by spaces. Output only the numbers.",
        "用中文简短说明：栈和队列的主要区别是什么？",
    ]
    .into_iter()
    .enumerate()
    {
        let prompt=tokenizer.encode(format!("<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),false).unwrap().get_ids().to_vec();
        reset(&mut target, &prompt);
        let proposal = decoder.propose(&mut target, &weights).unwrap();
        let path = proposal.path().to_vec();
        decoder.discard(proposal).unwrap();
        record(
            out.join(format!("path-{p}.json")),
            json!({"prompt":q,"prompt_ids":prompt,"path":path,"candidate_source":"own DFlash2 Q8 uncalibrated","teacher_forced_not_acceptance":true}),
        );
        let prefixes: Vec<usize> = if p == 0 {
            vec![8, 0, 1, 2, 3, 4, 5, 6, 7]
        } else {
            vec![7]
        };
        for n in prefixes {
            reset(&mut target, &prompt);
            let mut full_logits = Vec::new();
            for &id in &path[..n] {
                target.consume(id, true, &[]).unwrap();
                full_logits.extend(target.logits_snapshot().unwrap());
            }
            let expected = snapshot(&mut target);
            for &id in &path[n..] {
                target.consume(id, true, &[]).unwrap();
                full_logits.extend(target.logits_snapshot().unwrap());
            }
            reset(&mut target, &prompt);
            let before_capture = target
                .dflash_capture
                .as_ref()
                .unwrap()
                .committed_identity()
                .unwrap();
            let mut batch = Batch::new(&mut target).unwrap();
            let before = cuda_device_allocation_snapshot().successful_allocations;
            for bad in [vec![1; 7], vec![248320; 8]] {
                assert!(batch.execute(&mut target, &bad).is_err());
                assert!(!target.poisoned);
            }
            batch.execute(&mut target, &path).unwrap();
            assert!(target.poisoned);
            assert!(target.logits_snapshot().is_err());
            assert!(target.select_next_token().is_err());
            assert!(batch.execute(&mut target, &path).is_err());
            assert!(batch.commit(&mut target, 9).is_err());
            let actual_logits = batch.diagnostic_logits().unwrap();
            let logits_metric = compare(&bytes(&actual_logits), &bytes(&full_logits));
            batch.commit(&mut target, n).unwrap();
            assert_eq!(target.consumed, prompt.len() + n);
            assert!(!target.poisoned);
            assert_eq!(
                before,
                cuda_device_allocation_snapshot().successful_allocations,
                "per-round CUDA allocation"
            );
            if n == 0 {
                assert_eq!(
                    before_capture,
                    target
                        .dflash_capture
                        .as_ref()
                        .unwrap()
                        .committed_identity()
                        .unwrap()
                );
            }
            assert!(batch.commit(&mut target, n).is_err());
            let actual = snapshot(&mut target);
            assert_eq!(actual.len(), expected.len());
            let mut groups = Vec::new();
            for ((name, a), (expected_name, b)) in actual.iter().zip(&expected) {
                assert_eq!(name, expected_name);
                let m = compare(a, b);
                if n == 0 {
                    assert_eq!(
                        a, b,
                        "zero prefix must restore exact physical state: {name}"
                    );
                }
                groups.push(json!({"group":name,"metric":m}));
            }
            let pass =
                logits_metric["pass"] == true && groups.iter().all(|v| v["metric"]["pass"] == true);
            let result = json!({"prompt":p,"commit":n,"batch_logits":logits_metric,"groups":groups,"pass":pass,
                "no_cuda_allocation_in_execute_commit":true,"target_pending_fail_closed":true,"zero_prefix_bitwise":n==0});
            record(out.join(format!("p{p}-prefix-{n}.json")), result.clone());
            all.push(result);
            if n == 8 {
                write(
                    out.join(format!("p{p}-scalar-logits.f32le")),
                    &bytes(&full_logits),
                );
                write(
                    out.join(format!("p{p}-batch-logits.f32le")),
                    &bytes(&actual_logits),
                );
            }
            drop(batch);
            println!("B8_PROMPT={p} PREFIX={n} PASS={pass}");
        }
        last_prompt = prompt;
    }
    reset(&mut target, &last_prompt);
    let mut batch = Batch::new(&mut target).unwrap();
    batch.inject_failure();
    assert!(batch.execute(&mut target, &[1; 8]).is_err());
    assert!(target.poisoned);
    assert!(batch.commit(&mut target, 0).is_err());
    assert!(target.consume(1, true, &[]).is_err());
    assert!(target.capture_nerkyor_dflash_research_input().is_err());
    drop(batch);
    reset(&mut target, &last_prompt);
    drop(decoder);
    drop(target);
    drop(weights);
    let final_alloc = cuda_device_allocation_snapshot();
    assert_eq!(baseline.live_bytes, final_alloc.live_bytes);
    assert_eq!(baseline.live_allocations, final_alloc.live_allocations);
    let pass = all.iter().all(|v| v["pass"] == true);
    record(
        out.join("result.json"),
        json!({"status":if pass{"PASS_B8_TARGET_ALL_PREFIXES"}else{"FAIL_B8_TARGET_NUMERICAL_CONTRACT"},
        "cases":all,"failure_after_gdn_recurrence_fail_closed":true,"allocation_cleanup":true,"fast_speculative_verifier_admitted":false,
        "product_admitted":false,"speed_qualified":false,"independent_f64_qualified":false}),
    );
    assert!(pass, "B8 numerical contract failed; metrics preserved");
}
