//! Resident transport equivalence, actual host counters, lifecycle and warm cost.
//! All readbacks for the full numerical oracle sit outside measured intervals.
use super::batched::{compare, snapshot};
use super::*;
use crate::checkpoint_projection_cuda::CheckpointProjectionPolicy;
use augur_backend::{cuda_host_boundary_snapshot, CudaHostBoundarySnapshot};
fn raw(s: &OwnedCudaStream, b: &OwnedCudaBuffer) -> Vec<u8> {
    let mut v = vec![0; b.len_bytes()];
    b.copy_to_host_sync(s, &mut v).unwrap();
    v
}
fn head_planes(d: &NerkyorDflashResearchDecoder) -> Vec<Vec<u8>> {
    [
        &*d.normalized,
        &*d.logits,
        &d.ids,
        &d.scores,
        &d.transitions,
    ]
    .into_iter()
    .map(|b| raw(&d.stream, b))
    .collect()
}
fn compare_state(a: &[(String, Vec<u8>)], b: &[(String, Vec<u8>)]) -> Value {
    assert_eq!(a.len(), b.len());
    let groups: Vec<_> = a
        .iter()
        .zip(b)
        .map(|((n, a), (m, b))| {
            assert_eq!(n, m);
            json!({"name":n,"metric":compare(a,b)})
        })
        .collect();
    json!({"pass":groups.iter().all(|g|g["metric"]["pass"]==true),"groups":groups})
}
fn assert_same_state(a: &[(String, Vec<u8>)], b: &[(String, Vec<u8>)]) {
    assert_eq!(a.len(), b.len());
    for ((n, a), (m, b)) in a.iter().zip(b) {
        assert_eq!(n, m);
        assert_eq!(Sha256::digest(a), Sha256::digest(b), "{n}");
    }
}
fn counter_json(c: CudaHostBoundarySnapshot) -> Value {
    json!({"h2d_bytes":c.synchronous_h2d_bytes,"d2h_bytes":c.synchronous_d2h_bytes,"h2d_copies":c.synchronous_h2d_copies,
        "d2h_copies":c.synchronous_d2h_copies,"host_waits":c.host_waits,"stream_synchronizations":c.stream_synchronizations})
}
fn negatives(
    t: &mut CheckpointCudaTarget,
    d: &mut NerkyorDflashResearchDecoder,
    w: &DFlashQ8CudaWeights,
) {
    let before = state(t);
    let meta = (t.consumed, t.selected, t.poisoned, d.generation, d.live);
    let foreign = Rc::new(OwnedCudaStream::create(0).unwrap());
    let original = std::mem::replace(&mut t.stream, foreign);
    assert!(d.propose_resident(t, w).is_err());
    t.stream = original;
    // Pin the real bound operation, not just an environment string or metadata.
    std::env::set_var("AUGUR_SHARED_HEAD_KERNEL", "rw4");
    let bad = t
        .head
        .bind_shared_normalized_bf16_rows(
            &t.stream,
            Rc::clone(&d.normalized),
            Rc::clone(&d.logits),
            8,
        )
        .unwrap();
    std::env::set_var("AUGUR_SHARED_HEAD_KERNEL", "llama");
    let good = std::mem::replace(&mut d.optimized, bad);
    let alloc = cuda_device_allocation_snapshot().successful_allocations;
    let host = cuda_host_boundary_snapshot();
    assert!(d.propose(t, w).is_err());
    assert!(d.propose_resident(t, w).is_err());
    assert!(d.prepare_resident_proposal(t, w).is_err());
    let delta = cuda_host_boundary_snapshot()
        .checked_delta_since(host)
        .unwrap();
    assert_eq!(
        (delta.synchronous_h2d_bytes, delta.synchronous_d2h_bytes),
        (0, 0)
    );
    assert_eq!(
        alloc,
        cuda_device_allocation_snapshot().successful_allocations
    );
    d.optimized = good;
    assert_eq!(
        meta,
        (t.consumed, t.selected, t.poisoned, d.generation, d.live)
    );
    assert_eq!(before, state(t));
}
fn oracle(t: &mut CheckpointCudaTarget, prompt: &[u32], ids: &[u32]) -> Vec<(String, Vec<u8>)> {
    reset(t, prompt);
    for (i, &id) in ids.iter().enumerate() {
        assert_eq!(t.select_next_token().unwrap(), id, "scalar token {i}");
        if i + 1 < ids.len() {
            t.consume_selected().unwrap();
        }
    }
    snapshot(t)
}
fn decode(
    t: &mut CheckpointCudaTarget,
    d: &mut NerkyorDflashResearchDecoder,
    w: &DFlashQ8CudaWeights,
    mode: &str,
    eos: &[u32],
    limit: usize,
) -> (Vec<u32>, usize) {
    let mut ids = Vec::new();
    let mut rounds = 0;
    while ids.len() < limit {
        if mode == "scalar" {
            ids.push(t.select_next_token().unwrap());
        } else {
            let p = if mode == "legacy_b8" {
                d.propose(t, w).unwrap()
            } else {
                d.propose_resident(t, w).unwrap()
            };
            let r = d.verify_batched(t, &p, limit - ids.len(), eos).unwrap();
            ids.extend(r.output_tokens);
            rounds += 1;
        }
        if eos.contains(ids.last().unwrap()) || ids.len() == limit {
            break;
        }
        t.consume_selected().unwrap();
    }
    (ids, rounds)
}
#[test]
#[ignore = "exclusive SM70 resident transport equivalence, lifecycle and warm request cost"]
fn real_resident_proposals_bitwise_host_boundary_reset_and_warm_cost() {
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
        "Count from 1 to 20, separated by spaces. Output only the numbers.".to_owned(),
        format!(
            "{}Write a Python function add(a, b) that returns their sum.",
            "Context note: use concise answers. ".repeat(16)
        ),
        "用中文简短说明：栈和队列的主要区别是什么？".into(),
        "What is 17 * 23? Output only the answer.".into(),
    ];
    let prompts:Vec<Vec<u32>>=questions.iter().map(|q|tokenizer.encode(format!("<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),false).unwrap().get_ids().to_vec()).collect();
    assert!(prompts.iter().all(|p| p.len() + 32 < 512));
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
        512,
        CheckpointProjectionPolicy::NerkyorBf16GateF32CtaResearch,
    )
    .unwrap();
    target
        .enable_nerkyor_batched_dflash_research_capture(&weights)
        .unwrap();
    let mut decoder = NerkyorDflashResearchDecoder::new(&mut target, &weights).unwrap();
    let eos = [248046, 248044];
    let mut cases = Vec::new();
    let mut pairs = Vec::new();
    let mut reset_reference = None;
    for (p, prompt) in prompts.iter().enumerate() {
        reset(&mut target, prompt);
        if p == 0 {
            reset_reference = Some(snapshot(&mut target));
            let before = (
                target.consumed,
                target.selected,
                target.poisoned,
                decoder.generation,
            );
            let alloc = cuda_device_allocation_snapshot().successful_allocations;
            assert!(decoder.propose_resident(&mut target, &weights).is_err());
            assert_eq!(
                before,
                (
                    target.consumed,
                    target.selected,
                    target.poisoned,
                    decoder.generation
                )
            );
            assert_eq!(
                alloc,
                cuda_device_allocation_snapshot().successful_allocations
            );
        }
        decoder.prepare_batched_verifier(&mut target).unwrap();
        decoder
            .prepare_resident_proposal(&mut target, &weights)
            .unwrap();
        if p == 0 {
            negatives(&mut target, &mut decoder, &weights);
        }
        let mut generated = Vec::new();
        let mut round = 0;
        while generated.len() < 24 {
            target.select_next_token().unwrap();
            let input = target.capture_nerkyor_dflash_research_input().unwrap();
            let expected_noise = input.noise_embedding().to_vec();
            drop(input);
            let h = cuda_host_boundary_snapshot();
            let tick = Instant::now();
            let legacy = decoder.propose(&mut target, &weights).unwrap();
            let legacy_s = tick.elapsed().as_secs_f64();
            let legacy_host = cuda_host_boundary_snapshot()
                .checked_delta_since(h)
                .unwrap();
            let path = legacy.path.clone();
            let old = head_planes(&decoder);
            decoder.discard(legacy).unwrap();
            let alloc = cuda_device_allocation_snapshot().successful_allocations;
            let h = cuda_host_boundary_snapshot();
            let tick = Instant::now();
            let resident = decoder.propose_resident(&mut target, &weights).unwrap();
            let resident_s = tick.elapsed().as_secs_f64();
            let host = cuda_host_boundary_snapshot()
                .checked_delta_since(h)
                .unwrap();
            assert_eq!(
                alloc,
                cuda_device_allocation_snapshot().successful_allocations
            );
            assert_eq!(
                (host.synchronous_h2d_bytes, host.synchronous_d2h_bytes),
                (32, 32)
            );
            assert_eq!(path, resident.path);
            let new = head_planes(&decoder);
            let plane_equal: Vec<_> = old
                .iter()
                .zip(&new)
                .map(|(a, b)| Sha256::digest(a) == Sha256::digest(b))
                .collect();
            let noise = decoder.resident.as_ref().unwrap().noise_snapshot().unwrap();
            assert_eq!(
                Sha256::digest(bytes(&expected_noise)),
                Sha256::digest(bytes(&noise)),
                "resident embedding gather"
            );
            let pair = json!({"prompt":p,"round":round,"context_len":target.consumed_tokens(),"plane_names":["normalized_hidden","full_logits","topk_ids","topk_scores","transitions"],"bitwise_equal":plane_equal,
                "path_equal":true,"embedding_equal":true,"resident_no_cuda_allocations":true,"resident_host":counter_json(host),"legacy_host":counter_json(legacy_host),"legacy_proposal_s":legacy_s,"resident_proposal_s":resident_s});
            record(out.join(format!("pair-{p}-{round}.json")), pair.clone());
            assert!(
                plane_equal.iter().all(|v| *v),
                "transport changed draft arithmetic"
            );
            pairs.push(pair);
            let before = cuda_host_boundary_snapshot();
            let alloc = cuda_device_allocation_snapshot().successful_allocations;
            assert!(decoder.propose_resident(&mut target, &weights).is_err());
            assert!(decoder.propose(&mut target, &weights).is_err());
            assert!(decoder.release_resident_proposal().is_err());
            assert!(decoder
                .prepare_resident_proposal(&mut target, &weights)
                .is_err());
            let h = cuda_host_boundary_snapshot()
                .checked_delta_since(before)
                .unwrap();
            assert_eq!((h.synchronous_h2d_bytes, h.synchronous_d2h_bytes), (0, 0));
            assert_eq!(
                alloc,
                cuda_device_allocation_snapshot().successful_allocations
            );
            let r = decoder
                .verify_batched(&mut target, &resident, 24 - generated.len(), &eos)
                .unwrap();
            generated.extend(r.output_tokens);
            round += 1;
            if eos.contains(generated.last().unwrap()) || generated.len() == 24 {
                break;
            }
            target.consume_selected().unwrap();
        }
        let actual = snapshot(&mut target);
        decoder.release_batched_verifier().unwrap();
        let expected = oracle(&mut target, prompt, &generated);
        let metrics = compare_state(&actual, &expected);
        let case = json!({"prompt":p,"prompt_tokens":prompt.len(),"output_tokens":generated,"rounds":round,"full_committed_state":metrics});
        record(out.join(format!("case-{p}.json")), case.clone());
        cases.push(case);
    }
    // Order-balanced, same-model warm requests. Load and prefill are separated;
    // setup+decode and decode-only are reported, never an HTTP speed claim.
    let mut timings = Vec::new();
    for p in [0usize, 2] {
        let mut gold_tokens = None;
        for repeat in 0..3 {
            let orders = [
                ["scalar", "legacy_b8", "resident_b8"],
                ["legacy_b8", "resident_b8", "scalar"],
                ["resident_b8", "scalar", "legacy_b8"],
            ];
            for mode in orders[repeat] {
                decoder.release_batched_verifier().unwrap();
                let pre = Instant::now();
                reset(&mut target, &prompts[p]);
                let prefill_s = pre.elapsed().as_secs_f64();
                let setup = Instant::now();
                if mode != "scalar" {
                    decoder.prepare_batched_verifier(&mut target).unwrap();
                    if mode == "resident_b8" {
                        decoder
                            .prepare_resident_proposal(&mut target, &weights)
                            .unwrap();
                    }
                }
                let setup_s = setup.elapsed().as_secs_f64();
                let alloc = cuda_device_allocation_snapshot().successful_allocations;
                let host = cuda_host_boundary_snapshot();
                let tick = Instant::now();
                let (ids, rounds) = decode(&mut target, &mut decoder, &weights, mode, &eos, 24);
                let decode_s = tick.elapsed().as_secs_f64();
                let allocs = cuda_device_allocation_snapshot().successful_allocations - alloc;
                let host = cuda_host_boundary_snapshot()
                    .checked_delta_since(host)
                    .unwrap();
                if mode == "resident_b8" {
                    assert_eq!(allocs, 0);
                }
                if let Some(gold) = &gold_tokens {
                    assert_eq!(&ids, gold);
                } else {
                    gold_tokens = Some(ids.clone());
                }
                let row = json!({"prompt":p,"repeat":repeat,"mode":mode,"tokens":ids.len(),"rounds":rounds,"prefill_s":prefill_s,"setup_s":setup_s,"decode_s":decode_s,"setup_plus_decode_s":setup_s+decode_s,"decode_cuda_allocations":allocs,"host":counter_json(host),"output_tokens":ids});
                record(
                    out.join(format!("cost-{p}-{repeat}-{mode}.json")),
                    row.clone(),
                );
                timings.push(row);
            }
        }
    }
    decoder.release_batched_verifier().unwrap();
    reset(&mut target, &prompts[0]);
    decoder
        .prepare_resident_proposal(&mut target, &weights)
        .unwrap();
    target.select_next_token().unwrap();
    let before = snapshot(&mut target);
    let metadata_before = (
        target.consumed,
        target.selected,
        target.logits_current,
        target
            .dflash_capture
            .as_ref()
            .unwrap()
            .committed_identity()
            .unwrap(),
    );
    crate::dflash_q8_backbone_cuda::augur_owner::FAIL_AFTER_FEATURE_COPY.with(|f| f.set(true));
    assert!(decoder.propose_resident(&mut target, &weights).is_err());
    assert!(decoder.poisoned);
    assert!(!target.poisoned);
    assert!(decoder.release_resident_proposal().is_err());
    assert_same_state(&before, &snapshot(&mut target));
    assert_eq!(
        metadata_before,
        (
            target.consumed,
            target.selected,
            target.logits_current,
            target
                .dflash_capture
                .as_ref()
                .unwrap()
                .committed_identity()
                .unwrap()
        )
    );
    drop(decoder);
    reset(&mut target, &prompts[0]);
    assert_same_state(reset_reference.as_ref().unwrap(), &snapshot(&mut target));
    drop(target);
    drop(weights);
    let freed = cuda_device_allocation_snapshot();
    assert_eq!(baseline.live_bytes, freed.live_bytes);
    assert_eq!(baseline.live_allocations, freed.live_allocations);
    let pass = cases
        .iter()
        .all(|c| c["full_committed_state"]["pass"] == true);
    record(
        out.join("result.json"),
        json!({"status":if pass{"PASS_RESIDENT_TRANSPORT_BITWISE_AND_HOST_BOUNDARY"}else{"FAIL_RESIDENT_STATE_NUMERICAL_CONTRACT"},"cases":cases,"pairs":pairs,"warm_cost":timings,
        "failure_after_borrowed_copy_target_unchanged":true,"reset_full_state_bitwise_recovered":true,"allocation_cleanup":true,"head_bound_operation_negative":true,
        "product_admitted":false,"HTTP_speed_qualified":false,"long_context_qualified":false,"independent_whole_path_f64_qualified":false}),
    );
    assert!(pass, "resident full state numerical contract failed");
}
