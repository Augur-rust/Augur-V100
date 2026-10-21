//! Test-only 4K context qualification; the released constructor stays at 512.
use super::gpu_tests::{close, metric, read, record, state, turn};
use super::*;
use augur_backend::{
    cuda_device_allocation_snapshot, cuda_device_properties, CudaComputeCapability,
};
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf, time::Instant};

struct Reference {
    receipt: TurnReceipt,
    logits: Vec<f32>,
    states: Vec<Value>,
}
fn digest_state(s: &mut NerkyorResearchCudaSession) -> (Vec<f32>, Vec<Value>) {
    let mut all = state(s).into_iter();
    let (name, logits) = all.next().unwrap();
    assert_eq!(name, "logits");
    let states = all
        .map(|(name, values)| {
            assert!(values.iter().all(|v| v.is_finite()));
            let mut h = Sha256::new();
            // Hash every element of full allocated KV/GDN and captured features;
            // retain only hashes so the test does not hold several GB of snapshots.
            for chunk in values.chunks(16_384) {
                let bytes: Vec<u8> = chunk.iter().flat_map(|v| v.to_le_bytes()).collect();
                h.update(&bytes);
            }
            json!({"tensor":name,"elements":values.len(),"sha256":format!("{:x}",h.finalize())})
        })
        .collect();
    (logits, states)
}
fn compare(s: &mut NerkyorResearchCudaSession, r: &TurnReceipt, gold: &Reference) -> Value {
    assert_eq!(r.output_token_ids, gold.receipt.output_token_ids);
    assert_eq!(r.stop_reason, gold.receipt.stop_reason);
    assert_eq!(
        r.session.consumed_token_ids,
        gold.receipt.session.consumed_token_ids
    );
    assert_eq!(r.session.pending_token, gold.receipt.session.pending_token);
    assert_eq!(
        r.session.pending_token_published,
        gold.receipt.session.pending_token_published
    );
    let (logits, states) = digest_state(s);
    let logits_metric = metric(&logits, &gold.logits);
    assert_eq!(logits_metric["pass"], true);
    assert_eq!(states.len(), 129);
    // Deliberately stronger than the numeric contract for non-logit state:
    // mismatch is a failed experiment, not permission to relax tolerances.
    assert_eq!(
        states, gold.states,
        "complete non-logit state byte-digest mismatch"
    );
    json!({"exact_output_stop_history_pending":true,"logits":logits_metric,
        "nonlogit_groups":states.len(),"nonlogit_full_bytes_sha256_equal":true,
        "nonlogit_elements":states.iter().map(|v|v["elements"].as_u64().unwrap()).sum::<u64>(),
        "comparison":"full non-logit byte SHA256 equality; logits F64-accumulated fixed contract"})
}

#[test]
#[ignore = "exclusive SM70; test-only 4096-capacity full-state scalar oracle and recovery"]
fn real_4096_context_append_boundary_scalar_oracle_and_recovery() {
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
    let root = PathBuf::from(std::env::var_os("AUGUR_NERKYOR_MODEL").unwrap());
    let draft = PathBuf::from(std::env::var_os("AUGUR_NERKYOR_DFLASH_Q8").unwrap());
    let manifest = PathBuf::from(std::env::var_os("AUGUR_NERKYOR_SESSION_MANIFEST").unwrap());
    let pin = std::env::var("AUGUR_NERKYOR_SESSION_MANIFEST_SHA256").unwrap();
    let admitted = crate::nerkyor_session_manifest::load(&root, &draft, &manifest, &pin).unwrap();
    let tokenizer = tokenizers::Tokenizer::from_bytes(admitted.tokenizer_bytes()).unwrap();
    let encode = |text: &str| tokenizer.encode(text, false).unwrap().get_ids().to_vec();
    let filler=encode(&"Reference notes: a stack is last in first out; a queue is first in first out. Python: def add(a, b): return a + b. 参考资料：栈后进先出，队列先进先出。\n".repeat(100));
    let header = encode("<|im_start|>user\nRead these reference notes.\n");
    let append_header = encode("<|im_end|>\n<|im_start|>user\nMore reference notes follow.\n");
    let footer=encode("\nNow count from 1 to 1000, separated by spaces. Output only the numbers.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    let known = |length: usize, append: bool| {
        let start = if append { &append_header } else { &header };
        assert!(length > start.len() + footer.len());
        let n = length - start.len() - footer.len();
        assert!(n <= filler.len());
        [start.as_slice(), &filler[..n], footer.as_slice()].concat()
    };
    let short=encode("<|im_start|>user\nWhat is 17 * 23? Output only the answer.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    record(
        out.join("policy.json"),
        &json!({"declared_before_gpu_output":true,
        "capacity":4096,"input_boundaries":[513,1025,2049,4072,4095],"max_new_tokens":[16,16,16,8,1],
        "relative_l2_limit":1e-5,"absolute_floor":1e-4,"max_abs_over_reference_peak_limit":1e-5,
        "nonlogit_gate":"full byte SHA256 equality, stricter than numeric contract",
        "inputs":"self-authored mixed English/Chinese/code, exact token lengths; not official quality or HF template qualification",
        "public_capacity_unchanged":512,"HTTP_qualified":false,"product_ready":false}),
    );
    let baseline = cuda_device_allocation_snapshot();
    let load = Instant::now();
    let mut s = NerkyorResearchCudaSession::open_context(admitted, Context::Extended4096).unwrap();
    let load_s = load.elapsed().as_secs_f64();
    assert_eq!(s.info()["kv_capacity"], 4096);
    record(out.join("target-info.json"), s.info());
    let loaded_bytes = cuda_device_allocation_snapshot().live_bytes - baseline.live_bytes;
    let mut reference = Vec::new();
    let mut comparisons = Vec::new();
    let mut timings = Vec::new();
    for mode in ["scalar", "dflash"] {
        s.owner.target_for_test().enabled = mode == "dflash";
        let mut previous: Option<TurnReceipt> = None;
        for (index, (end, budget)) in [(513, 16), (1025, 16), (2049, 16), (4072, 8), (4095, 1)]
            .into_iter()
            .enumerate()
        {
            let id = format!("{mode}-context-{end}");
            let input = if let Some(r) = &previous {
                let n = end - r.session.consumed_token_ids.len() - 1;
                TurnInput::Append {
                    revision: r.session.revision,
                    exact_history: r.session.consumed_token_ids.clone(),
                    suffix_token_ids: if index == 4 {
                        filler[..n].to_vec()
                    } else {
                        known(n, true)
                    },
                }
            } else {
                TurnInput::Open {
                    prompt_token_ids: known(end, false),
                }
            };
            let request = TurnRequest {
                request_id: id.clone(),
                session_id: format!("{mode}-4k-session"),
                input,
                max_new_tokens: budget,
                selector_policy: None,
            };
            let started = Instant::now();
            let r = turn(&mut s, &request, &out);
            let elapsed = started.elapsed().as_secs_f64();
            assert!(r.output_token_ids.len() <= budget);
            assert!(matches!(r.stop_reason, "length" | "eos"));
            let published = r.stop_reason == "length";
            assert_eq!(r.session.pending_token_published, published);
            assert_eq!(
                r.session.consumed_token_ids.len(),
                end + r.output_token_ids.len() - usize::from(published)
            );
            let rounds = s.owner.target_for_test().rounds;
            if mode == "dflash" && index < 4 {
                assert!(rounds > 0, "long-context DFlash must actually execute");
            }
            timings.push(json!({"mode":mode,"input_cursor":end,"consumed":s.submitted_tokens(),"rounds":rounds,"session_turn_s":elapsed}));
            if mode == "scalar" {
                let (logits, states) = digest_state(&mut s);
                record(
                    out.join(format!("scalar-state-{end}.json")),
                    &json!({"nonlogit":states,"logits_elements":logits.len()}),
                );
                reference.push(Reference {
                    receipt: r.clone(),
                    logits,
                    states,
                });
            } else {
                let result = compare(&mut s, &r, &reference[index]);
                record(out.join(format!("comparison-{end}.json")), &result);
                comparisons.push(result);
            }
            println!("NERKYOR_4K_PROGRESS mode={mode} input_cursor={end} consumed={} turn_s={elapsed:.3}",s.submitted_tokens());
            previous = Some(r);
        }
        assert_eq!(s.submitted_tokens(), 4095);
        let last = previous.unwrap();
        let overflow = TurnRequest {
            request_id: format!("{mode}-over-budget"),
            session_id: last.session.session_id.clone(),
            input: TurnInput::Append {
                revision: last.session.revision,
                exact_history: last.session.consumed_token_ids.clone(),
                suffix_token_ids: vec![1],
            },
            max_new_tokens: 1,
            selector_policy: None,
        };
        let before = cuda_device_allocation_snapshot().successful_allocations;
        assert!(s
            .turn(
                &overflow,
                &SessionCancellationToken::new(),
                |_| panic!("no emission"),
                |_| panic!("no receipt")
            )
            .is_err());
        assert_eq!(
            before,
            cuda_device_allocation_snapshot().successful_allocations
        );
        assert_eq!(s.submitted_tokens(), 4095);
        assert!(!s.poisoned());
        close(&mut s, &last);
    }
    // Repeated reset at the new capacity must clear even unused KV/capture tails.
    s.owner.target_for_test().enabled = false;
    let short_request = |id: &str| TurnRequest {
        request_id: id.into(),
        session_id: format!("session-{id}"),
        input: TurnInput::Open {
            prompt_token_ids: short.clone(),
        },
        max_new_tokens: 8,
        selector_policy: None,
    };
    let gold = turn(&mut s, &short_request("short-reference"), &out);
    let (logits, states) = digest_state(&mut s);
    let short_gold = Reference {
        receipt: gold.clone(),
        logits,
        states,
    };
    close(&mut s, &gold);
    s.owner.target_for_test().enabled = true;
    let cancel = SessionCancellationToken::new();
    let mut emitted = 0;
    let mut persisted = false;
    let cancelled = TurnRequest {
        request_id: "cancel-after-1025".into(),
        session_id: "cancel-long-session".into(),
        input: TurnInput::Open {
            prompt_token_ids: known(1025, false),
        },
        max_new_tokens: 24,
        selector_policy: None,
    };
    let error = s
        .turn(
            &cancelled,
            &cancel,
            |_| {
                emitted += 1;
                if emitted == 2 {
                    cancel.request_cancel();
                }
                Ok(())
            },
            |_| {
                persisted = true;
                Err("no commit".into())
            },
        )
        .unwrap_err();
    assert!(error.is::<crate::session_cancellation::SessionCancellationRequested>());
    assert_eq!(emitted, 2);
    assert!(!persisted);
    assert!(s.poisoned());
    let cancelled_cursor = s.submitted_tokens();
    assert!(cancelled_cursor > 1025);
    s.discard_cancelled_turn(
        "discard-long",
        &cancelled.request_id,
        &cancelled.session_id,
        |r| {
            record(out.join("long-discard.json"), &serde_json::to_value(r)?);
            Ok(session::digest(&serde_json::to_vec(r)?))
        },
    )
    .unwrap();
    assert_eq!(s.submitted_tokens(), 0);
    assert!(!s.poisoned());
    let recovered = turn(&mut s, &short_request("short-recovered"), &out);
    comparisons.push(compare(&mut s, &recovered, &short_gold));
    close(&mut s, &recovered);
    let oversized = TurnRequest {
        request_id: "oversized-fresh".into(),
        session_id: "oversized-fresh-session".into(),
        input: TurnInput::Open {
            prompt_token_ids: vec![1; 4096],
        },
        max_new_tokens: 1,
        selector_policy: None,
    };
    let before = cuda_device_allocation_snapshot().successful_allocations;
    assert!(s
        .turn(
            &oversized,
            &SessionCancellationToken::new(),
            |_| panic!("no emission"),
            |_| panic!("no receipt")
        )
        .is_err());
    assert_eq!(
        before,
        cuda_device_allocation_snapshot().successful_allocations
    );
    assert_eq!(s.submitted_tokens(), 0);
    let resident_bytes = cuda_device_allocation_snapshot().live_bytes - baseline.live_bytes;
    s.retire().unwrap();
    drop(s);
    let freed = cuda_device_allocation_snapshot();
    assert_eq!(baseline.live_bytes, freed.live_bytes);
    assert_eq!(baseline.live_allocations, freed.live_allocations);
    record(
        out.join("result.json"),
        &json!({"status":"PASS_TEST_ONLY_4096_SESSION_FULL_STATE_APPEND_BOUNDARY_RECOVERY",
        "context_profile":Context::Extended4096.profile(),"capacity":4096,"public_default_capacity":CAPACITY,
        "comparisons":comparisons,"timings":timings,"load_s":load_s,"loaded_device_bytes":loaded_bytes,"resident_device_bytes":resident_bytes,
        "same_weight_owner_loads":1,"append_boundaries":[513,1025,2049,4072,4095],"final_consumed":4095,"final_pending_matches_scalar":true,
        "over_budget_append_and_open_no_submission":true,"cancelled_cursor":cancelled_cursor,"same_owner_cancel_reset_replay":true,
        "allocation_cleanup":true,"HTTP_qualified":false,"official_BF16":false,"full_model_FP64":false,"product_ready":false}),
    );
}
