use super::tests::{bytes, hash, projection_policy, top};
use super::*;
use augur_backend::{cuda_host_boundary_snapshot, CudaHostBoundarySnapshot};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

#[test]
fn token_sequence_contract_checks_all_ids_and_capacity_before_enqueue() {
    assert!(validate_sequence(0, 8, &[0, 248319]).is_ok());
    assert!(validate_sequence(7, 8, &[1]).is_ok());
    for tokens in [vec![], vec![1, 248320], vec![u32::MAX]] {
        assert!(validate_sequence(0, 8, &tokens).is_err());
    }
    assert!(validate_sequence(8, 8, &[1]).is_err());
    assert!(validate_sequence(7, 8, &[1, 2]).is_err());
    assert!(validate_sequence(usize::MAX, usize::MAX, &[1]).is_err());
    assert_eq!(top(&[-0.0, 0.0, -1.0]), 0, "signed zero is a numeric tie");
    assert_eq!(top(&[1.0, 2.0, 2.0]), 1);
}

#[test]
fn projection_policy_is_explicit_and_cannot_relabel_augur() {
    assert!(CheckpointProjectionPolicy::QpnFfnF16.uses_qpn_ffn());
    assert!(!CheckpointProjectionPolicy::QpnFfnF16.uses_fp8_warp());
    assert!(CheckpointProjectionPolicy::NvidiaFp8WarpQpnFfnF16.uses_qpn_ffn());
    assert!(CheckpointProjectionPolicy::NvidiaFp8WarpQpnFfnF16.uses_fp8_warp());
    assert!(!CheckpointProjectionPolicy::OriginalF32.uses_qpn_ffn());
    assert!(CheckpointProjectionPolicy::QpnFfnF16PostScale.qpn_postscale());
    assert!(CheckpointProjectionPolicy::NvidiaFp8WarpQpnFfnF16PostScale.uses_fp8_warp());
    assert!(!CheckpointProjectionPolicy::QpnFfnF16.qpn_postscale());
    for layer in 0..65 {
        assert_eq!(
            CheckpointProjectionPolicy::NvidiaFp8WarpQpnFfnF16Tail8.qpn_ffn_layer(layer),
            layer < 56
        );
        assert_eq!(
            CheckpointProjectionPolicy::QpnFfnF16.qpn_ffn_layer(layer),
            layer < 64
        );
    }
    assert_eq!(
        CheckpointProjectionPolicy::default(),
        CheckpointProjectionPolicy::OriginalF32
    );
    assert_ne!(
        CheckpointProjectionPolicy::OriginalF32.label(),
        CheckpointProjectionPolicy::NvidiaFp8WarpF32.label()
    );
    let identity = CheckpointExpectedIdentity {
        config_sha256: "0".repeat(64),
        index_sha256: "0".repeat(64),
        shards: Default::default(),
    };
    // Reject before looking up any artifact or querying CUDA.
    let result = CheckpointCudaTarget::open_with_projection_policy(
        "unopened-not-a-model",
        Qwen35CheckpointProfile::QuasarNvfp4,
        identity,
        512,
        CheckpointProjectionPolicy::NvidiaFp8WarpF32,
    );
    assert!(result.err().unwrap().to_string().contains("NVIDIA-only"));
}

pub(super) fn ids(value: &Value) -> Vec<u32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
        .collect()
}

pub(super) fn open(case: &Value) -> CheckpointCudaTarget {
    let data = std::fs::read(case["audit_path"].as_str().unwrap()).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&data)),
        case["audit_sha256"].as_str().unwrap()
    );
    let audit: Value = serde_json::from_slice(&data).unwrap();
    for record in case["metadata"].as_object().unwrap().values() {
        bytes(record);
    }
    let expected = CheckpointExpectedIdentity {
        config_sha256: audit["metadata"]["config.json"]["sha256"]
            .as_str()
            .unwrap()
            .into(),
        index_sha256: audit["metadata"]["model.safetensors.index.json"]["sha256"]
            .as_str()
            .unwrap()
            .into(),
        shards: audit["shards"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| {
                (
                    s["name"].as_str().unwrap().into(),
                    s["sha256"].as_str().unwrap().into(),
                )
            })
            .collect(),
    };
    let profile = match case["profile"].as_str().unwrap() {
        "quasar" => Qwen35CheckpointProfile::QuasarNvfp4,
        "nvidia" => Qwen35CheckpointProfile::NvidiaMixed,
        _ => panic!("unknown target profile"),
    };
    let kv_policy = match std::env::var("AUGUR_CHECKPOINT_KV_POLICY").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("f32") => CheckpointKvPolicy::F32,
        Ok("q8_0") => CheckpointKvPolicy::Q8_0,
        _ => panic!("invalid checkpoint KV policy"),
    };
    let capacity = std::env::var("AUGUR_CHECKPOINT_KV_CAPACITY")
        .map(|v| v.parse::<usize>().expect("capacity integer"))
        .unwrap_or(512);
    let mut target = CheckpointCudaTarget::open_with_execution_policy(
        audit["model_dir"].as_str().unwrap(),
        profile,
        expected,
        capacity,
        projection_policy(),
        kv_policy,
    )
    .unwrap();
    super::tests::configure_norm(&mut target);
    match std::env::var("AUGUR_CHECKPOINT_FFN_BATCH").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("0") => {}
        Ok("8") => {
            #[cfg(feature = "qwen35-prefill-cublas")]
            target.enable_batched_ffn8().unwrap();
            #[cfg(not(feature = "qwen35-prefill-cublas"))]
            panic!("FFN batch feature absent");
        }
        _ => panic!("invalid explicit FFN batch width"),
    }
    target
}

pub(super) fn boundary(value: CudaHostBoundarySnapshot) -> Value {
    json!({"host_waits":value.host_waits,"stream_synchronizations":value.stream_synchronizations,
        "event_synchronizations":value.event_synchronizations,
        "synchronous_h2d_copies":value.synchronous_h2d_copies,"synchronous_h2d_bytes":value.synchronous_h2d_bytes,
        "synchronous_d2h_copies":value.synchronous_d2h_copies,"synchronous_d2h_bytes":value.synchronous_d2h_bytes})
}

fn request(
    target: &mut CheckpointCudaTarget,
    prompt: &[u32],
    eos: &[u32],
    budget: usize,
    device: bool,
) -> Value {
    target.reset_empty().unwrap();
    let before = cuda_host_boundary_snapshot();
    let started = Instant::now();
    if device {
        target.consume_sequence(prompt).unwrap();
    } else {
        for (i, &id) in prompt.iter().enumerate() {
            target.consume(id, i + 1 == prompt.len(), &[]).unwrap();
        }
    }
    let prefill_s = started.elapsed().as_secs_f64();
    let mut output = Vec::new();
    let mut pending = 0;
    let mut selected_count = 0;
    let mut first_s = 0.0;
    let mut stopped = false;
    for i in 0..budget {
        let id = if device {
            target.greedy_next_token().unwrap()
        } else {
            top(&target.logits_snapshot().unwrap()) as u32
        };
        if i == 0 {
            first_s = started.elapsed().as_secs_f64();
        }
        pending = id;
        selected_count += 1;
        if eos.contains(&id) {
            stopped = true;
            break;
        }
        output.push(id);
        if i + 1 < budget {
            if device {
                target.consume_selected().unwrap();
            } else {
                target.consume(id, true, &[]).unwrap();
            }
        }
    }
    let wall_s = started.elapsed().as_secs_f64();
    let counters = cuda_host_boundary_snapshot()
        .checked_delta_since(before)
        .unwrap();
    assert!(!output.is_empty());
    let consumed = prompt.len() + output.len() - usize::from(!stopped);
    assert_eq!(target.consumed_tokens(), consumed);
    assert_eq!(counters.synchronous_d2h_copies, selected_count);
    assert_eq!(
        counters.synchronous_d2h_bytes,
        selected_count * if device { 4 } else { 248320 * 4 }
    );
    assert_eq!(
        counters.synchronous_h2d_copies,
        if device { 1 } else { consumed as u64 }
    );
    assert_eq!(
        counters.stream_synchronizations,
        if device { 1 } else { consumed as u64 }
    );
    // Full logits are compared AFTER timing; they are not hidden in the fast
    // path's measured transfer totals. Performance remains a scalar component.
    let logits_hash = hash(&target.logits_snapshot().unwrap());
    json!({"path":if device {"sequence_device_top1"} else {"scalar_host_top1"},
        "output_token_ids":output,"pending_token":pending,"stop_reason":if stopped {"eos"} else {"max_new_tokens"},
        "consumed_tokens":consumed,"final_logits_sha256":logits_hash,
        "prefill_wall_s":prefill_s,"first_token_wall_s":first_s,"whole_wall_s":wall_s,
        "first_token_to_finish_s":wall_s-first_s,"boundary":boundary(counters),
        "selected_tokens_including_eos":selected_count,"product_ready":false,"performance_qualified":false})
}

fn equal_requests(left: &Value, right: &Value) {
    for name in [
        "output_token_ids",
        "pending_token",
        "stop_reason",
        "consumed_tokens",
        "final_logits_sha256",
    ] {
        assert_eq!(
            left[name], right[name],
            "target transfer path drift: {name}"
        );
    }
}

#[test]
#[ignore = "exclusive32GiB SM70 and authenticated complete target fixtures"]
fn real_checkpoint_transfer_continuation_greedy_cleanup() {
    let source =
        std::fs::read(std::env::var_os("AUGUR_CHECKPOINT_TARGET_REFERENCE").expect("reference"))
            .unwrap();
    let manifest: Value = serde_json::from_slice(&source).unwrap();
    let cases = manifest
        .get("cases")
        .map(|v| v.as_array().unwrap().clone())
        .unwrap_or_else(|| vec![manifest]);
    assert!((1..=2).contains(&cases.len()));
    let root =
        std::path::PathBuf::from(std::env::var_os("AUGUR_CHECKPOINT_TARGET_OUTPUT").expect("output"));
    std::fs::create_dir(&root).unwrap();
    let baseline = cuda_device_allocation_snapshot();
    let mut reports = Vec::new();
    for case in cases {
        let mut target = open(&case);
        let before = cuda_device_allocation_snapshot();
        let teacher = ids(&case["token_ids"]);
        let eos = ids(&case["eos_token_ids"]);
        assert!(target.greedy_next_token().is_err());
        assert!(target.consume_selected().is_err());
        target.consume_sequence(&teacher).unwrap();
        let teacher_hash = hash(&target.logits_snapshot().unwrap());
        let chosen = target.greedy_next_token().unwrap();
        assert_eq!(chosen, top(&target.logits_snapshot().unwrap()) as u32);
        let boundary_before = cuda_host_boundary_snapshot();
        assert_eq!(chosen, target.greedy_next_token().unwrap());
        assert_eq!(
            boundary_before,
            cuda_host_boundary_snapshot(),
            "cached selection must not resubmit"
        );
        assert!(target.consume_sequence(&[]).is_err());
        assert!(target.consume_sequence(&[1, 248320]).is_err());
        assert!(target
            .consume_sequence(&vec![1; target.capacity + 1])
            .is_err());
        assert_eq!(target.consumed_tokens(), teacher.len());
        assert_eq!(teacher_hash, hash(&target.logits_snapshot().unwrap()));
        assert_eq!(chosen, target.greedy_next_token().unwrap());
        target.consume_selected().unwrap();
        assert!(
            target.consume_selected().is_err(),
            "stale selection must reject"
        );
        let next_hash = hash(&target.logits_snapshot().unwrap());
        target.reset_empty().unwrap();
        assert!(target.consume_selected().is_err());
        for &id in &teacher {
            target.consume(id, true, &[]).unwrap();
        }
        assert_eq!(teacher_hash, hash(&target.logits_snapshot().unwrap()));
        target.consume(chosen, true, &[]).unwrap();
        assert_eq!(next_hash, hash(&target.logits_snapshot().unwrap()));
        // Append/chunk tests exercise persistent recurrent state, full KV and
        // absolute position; no truncated-prefix or independent-session swap.
        for split in 1..teacher.len() {
            target.reset_empty().unwrap();
            target.consume_sequence(&teacher[..split]).unwrap();
            target.consume_sequence(&teacher[split..]).unwrap();
            assert_eq!(teacher_hash, hash(&target.logits_snapshot().unwrap()));
        }
        if target.info()["ffn_batch_max_rows"] == 8 {
            #[cfg(feature = "qwen35-prefill-cublas")]
            {
                assert!(
                    target.enable_batched_ffn8().is_err(),
                    "cannot reconfigure an active owner"
                );
                // Reject every further operation on a partially advanced set
                // of layers. Reset must drain and reconstruct exact empty state.
                for failed_layer in [0, 63] {
                    target.reset_empty().unwrap();
                    target.ffn_batch.as_mut().unwrap().fail_after_layer = Some(failed_layer);
                    assert!(target
                        .consume_sequence(&teacher)
                        .unwrap_err()
                        .to_string()
                        .contains("injected partial"));
                    assert_eq!(target.consumed_tokens(), 0);
                    assert!(target.logits_snapshot().is_err());
                    assert!(target.greedy_next_token().is_err());
                    assert!(target.consume_selected().is_err());
                    assert!(target.consume_sequence(&teacher).is_err());
                    assert!(target.consume(teacher[0], true, &[]).is_err());
                    target.ffn_batch.as_mut().unwrap().fail_after_layer = None;
                    target.reset_empty().unwrap();
                    assert!(
                        target.enable_batched_ffn8().is_err(),
                        "cannot double-allocate a batch arena"
                    );
                    target.consume_sequence(&teacher).unwrap();
                    assert_eq!(teacher_hash, hash(&target.logits_snapshot().unwrap()));
                }
            }
            // Cross an8-row boundary with every remainder; preserve all state
            // when a suffix starts from a nonempty exact history.
            for remainder in 1..=8 {
                let input: Vec<_> = teacher
                    .iter()
                    .copied()
                    .cycle()
                    .take(8 + remainder)
                    .collect();
                target.reset_empty().unwrap();
                for &id in &input {
                    target.consume(id, true, &[]).unwrap();
                }
                let expected = hash(&target.logits_snapshot().unwrap());
                target.reset_empty().unwrap();
                target.consume_sequence(&input).unwrap();
                assert_eq!(
                    expected,
                    hash(&target.logits_snapshot().unwrap()),
                    "batch remainder{remainder}"
                );
                target.reset_empty().unwrap();
                target.consume_sequence(&input[..3]).unwrap();
                target.consume_sequence(&input[3..]).unwrap();
                assert_eq!(
                    expected,
                    hash(&target.logits_snapshot().unwrap()),
                    "nonempty suffix remainder{remainder}"
                );
            }
            target.reset_empty().unwrap();
            target.consume_sequence(&teacher).unwrap();
        }
        // Abandon one asynchronously queued decode, then reset and replay.
        target.greedy_next_token().unwrap();
        target.consume_selected().unwrap();
        target.reset_empty().unwrap();
        target.consume_sequence(&teacher).unwrap();
        assert_eq!(teacher_hash, hash(&target.logits_snapshot().unwrap()));
        // An invalid value ANYWHERE in the vocabulary must poison the owner,
        // not merely lose an argmax contest. Reset is the only recovery path.
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut logits = vec![0_f32; 248320];
            logits[248319] = bad;
            let data: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
            target
                .logits
                .copy_from_host_sync(&target.stream, &data)
                .unwrap();
            assert!(target.greedy_next_token().is_err());
            assert!(target.consume_sequence(&[1]).is_err());
            assert!(target.consume(1, true, &[]).is_err());
            assert!(target.logits_snapshot().is_err());
            target.reset_empty().unwrap();
            target.consume_sequence(&teacher).unwrap();
            assert_eq!(teacher_hash, hash(&target.logits_snapshot().unwrap()));
        }
        let tokenizer =
            tokenizers::Tokenizer::from_bytes(bytes(&case["metadata"]["tokenizer.json"])).unwrap();
        let mut requests = Vec::new();
        for prompt_case in case["generation"].as_array().unwrap() {
            let prompt = ids(&prompt_case["token_ids"]);
            assert_eq!(
                prompt,
                tokenizer
                    .encode(prompt_case["canonical_prompt"].as_str().unwrap(), false)
                    .unwrap()
                    .get_ids()
            );
            let budget = prompt_case["max_new_tokens"].as_u64().unwrap() as usize;
            let family = prompt_case["family"].as_str().unwrap();
            let repetitions = if ["english", "python"].contains(&family) {
                3
            } else {
                1
            };
            let mut first: Option<Value> = None;
            for repetition in 0..repetitions {
                let mut runs = Vec::new();
                let paired_batch =
                    std::env::var("AUGUR_CHECKPOINT_COMPARE_BATCH0").as_deref() == Ok("1");
                let mut modes = if paired_batch {
                    assert_eq!(target.info()["ffn_batch_max_rows"], 8);
                    vec![(false, false), (true, true), (true, false)]
                } else if repetition % 2 == 0 {
                    vec![(false, false), (true, false)]
                } else {
                    vec![(true, false), (false, false)]
                };
                if paired_batch {
                    let count = modes.len();
                    modes.rotate_left(repetition % count);
                }
                // Same loaded model/binary/device-top1 path: park but retain the
                // exact batch arena for the batch0 control; no setup in timing.
                for (device, suspend_batch) in modes {
                    #[cfg(feature = "qwen35-prefill-cublas")]
                    let parked = if suspend_batch {
                        Some(target.ffn_batch.take().expect("batch arena"))
                    } else {
                        None
                    };
                    let mut result = request(&mut target, &prompt, &eos, budget, device);
                    result["executed_ffn_batch_rows"] = json!(if !device || suspend_batch {
                        0
                    } else {
                        target.info()["ffn_batch_max_rows"].as_u64().unwrap_or(0)
                    });
                    #[cfg(feature = "qwen35-prefill-cublas")]
                    if let Some(batch) = parked {
                        assert!(target.ffn_batch.is_none());
                        target.ffn_batch = Some(batch);
                    }
                    if let Some(reference) = &first {
                        equal_requests(reference, &result);
                    } else {
                        first = Some(result.clone());
                    }
                    runs.push(result);
                }
                requests.push(json!({"family":family,"repetition":repetition,"input_tokens":prompt.len(),"runs":runs}));
            }
            // Continue an actually generated prefix, then replay that exact
            // history from empty. This is token-level continuation, not yet a
            // canonical second user turn or HTTP recovery qualification.
            let output = ids(&first.as_ref().unwrap()["output_token_ids"]);
            let continuation_at = (output.len() / 2).max(1);
            target.reset_empty().unwrap();
            target.consume_sequence(&prompt).unwrap();
            target.consume_sequence(&output[..continuation_at]).unwrap();
            let suffix_hash = hash(&target.logits_snapshot().unwrap());
            let mut complete = prompt.clone();
            complete.extend_from_slice(&output[..continuation_at]);
            target.reset_empty().unwrap();
            target.consume_sequence(&complete).unwrap();
            assert_eq!(suffix_hash, hash(&target.logits_snapshot().unwrap()));
            let text = tokenizer.decode(&output, false).unwrap();
            println!(
                "CHECKPOINT_TRANSFER={} {family} exact; {repetitions} timing pairs",
                case["profile"]
            );
            std::fs::write(root.join(format!("{}-{family}.json", case["profile"].as_str().unwrap())),
                serde_json::to_vec_pretty(&json!({"family":family,"text":text,"exact_generated_prefix_continuation":true})).unwrap()).unwrap();
        }
        assert_eq!(
            before.successful_allocations,
            cuda_device_allocation_snapshot().successful_allocations
        );
        reports.push(
            json!({"profile":case["profile"],"info":target.info(),"requests":requests,
            "sequence_scalar_logits_bitwise_equal":true,"device_host_greedy_equal":true,
            "chunked_and_generated_prefix_continuation_bitwise_equal":true,
            "invalid_input_preserves_state":true,"nonfinite_selection_fail_closed":true,
            "reset_after_queued_decode":true,"model_loaded_once":true}),
        );
        drop(target);
        let after = cuda_device_allocation_snapshot();
        assert_eq!(baseline.live_bytes, after.live_bytes);
        assert_eq!(baseline.live_allocations, after.live_allocations);
    }
    let after = cuda_device_allocation_snapshot();
    assert_eq!(
        after.successful_allocations - baseline.successful_allocations,
        after.successful_frees - baseline.successful_frees
    );
    std::fs::write(
        root.join("result.json"),
        serde_json::to_vec_pretty(&json!({
        "status":"PASS_CHECKPOINT_SEQUENCE_DEVICE_GREEDY_CONTINUATION_CLEANUP",
        "reference_sha256":format!("{:x}",Sha256::digest(&source)),"targets":reports,
        "live_bytes_after":after.live_bytes,"live_allocations_after":after.live_allocations,
        "product_ready":false,"semantic_qualified":false,"dflash_enabled":false,
        "vision_enabled":false,"native_w4a4":false,
        "kv_capacity":reports[0]["info"]["kv_capacity"],"kv_dtype":reports[0]["info"]["kv_dtype"]}))
        .unwrap(),
    )
    .unwrap();
}

#[test]
#[ignore = "exclusive CUDA; actual canonical8K context and complete source-bound target"]
fn real_checkpoint_context_exact_replay_cleanup() {
    let source =
        std::fs::read(std::env::var_os("AUGUR_CHECKPOINT_TARGET_REFERENCE").expect("reference"))
            .unwrap();
    let case: Value = serde_json::from_slice(&source).unwrap();
    let root =
        std::path::PathBuf::from(std::env::var_os("AUGUR_CHECKPOINT_TARGET_OUTPUT").expect("output"));
    std::fs::create_dir(&root).unwrap();
    let prompt = ids(&case["context_request"]["token_ids"]);
    assert!((8192..=16384).contains(&prompt.len()));
    let tokenizer =
        tokenizers::Tokenizer::from_bytes(bytes(&case["metadata"]["tokenizer.json"])).unwrap();
    assert_eq!(
        tokenizer
            .encode(
                case["context_request"]["canonical_prompt"]
                    .as_str()
                    .unwrap(),
                false
            )
            .unwrap()
            .get_ids(),
        prompt
    );
    let eos = ids(&case["eos_token_ids"]);
    let budget = case["context_request"]["max_new_tokens"].as_u64().unwrap() as usize;
    assert!((1..=64).contains(&budget));
    let baseline = cuda_device_allocation_snapshot();
    let mut target = open(&case);
    assert!(prompt.len() + budget <= target.capacity);
    let before = cuda_device_allocation_snapshot();
    let load_info = target.info().clone();
    let started = Instant::now();
    let mut checkpoints = Vec::new();
    for (chunk, input) in prompt.chunks(256).enumerate() {
        target.consume_sequence(input).unwrap();
        let point = json!({"chunk":chunk,"consumed_tokens":target.consumed_tokens(),
            "elapsed_s":started.elapsed().as_secs_f64(),"logits_sha256":hash(&target.logits_snapshot().unwrap())});
        checkpoints.push(point);
        std::fs::write(root.join("progress.json"),serde_json::to_vec_pretty(&json!({"status":"RUNNING",
            "phase":"prefill","actual_tokens":target.consumed_tokens(),"elapsed_s":started.elapsed().as_secs_f64()})).unwrap()).unwrap();
        println!(
            "CHECKPOINT_CONTEXT_PREFILL={} elapsed_s={:.3}",
            target.consumed_tokens(),
            started.elapsed().as_secs_f64()
        );
    }
    let prefill_s = started.elapsed().as_secs_f64();
    let prefill_hash = hash(&target.logits_snapshot().unwrap());
    let mut output = Vec::new();
    let mut first_s = None;
    let mut stop = "max_new_tokens";
    for _ in 0..budget {
        let token = target.greedy_next_token().unwrap();
        first_s.get_or_insert_with(|| started.elapsed().as_secs_f64());
        if eos.contains(&token) {
            stop = "eos";
            break;
        }
        output.push(token);
        target.consume_selected().unwrap();
    }
    assert!(!output.is_empty());
    let final_hash = hash(&target.logits_snapshot().unwrap());
    let whole_s = started.elapsed().as_secs_f64();
    let text = tokenizer.decode(&output, false).unwrap();
    let expected = case["context_request"]["expected_text"].as_str().unwrap();
    let request = json!({"input_tokens":prompt.len(),"output_token_ids":output,"text":text,
        "expected_text":expected,"task_answer_matches":text.trim()==expected,"stop_reason":stop,
        "prefill_wall_s":prefill_s,"first_token_wall_s":first_s,"whole_wall_s":whole_s,
        "prefill_timing_includes_periodic_logits_readback_and_progress_io":true,
        "prefill_logits_sha256":prefill_hash,"final_logits_sha256":final_hash,"checkpoints":checkpoints});
    std::fs::write(
        root.join("request.json"),
        serde_json::to_vec_pretty(&request).unwrap(),
    )
    .unwrap();
    target.reset_empty().unwrap();
    let replay_started = Instant::now();
    // Different chunking/nonempty suffixes exercise state, not text recreation.
    for input in prompt.chunks(257) {
        target.consume_sequence(input).unwrap();
        std::fs::write(root.join("progress.json"),serde_json::to_vec_pretty(&json!({"status":"RUNNING",
            "phase":"exact_replay","actual_tokens":target.consumed_tokens(),"elapsed_s":replay_started.elapsed().as_secs_f64()})).unwrap()).unwrap();
    }
    assert_eq!(
        prefill_hash,
        hash(&target.logits_snapshot().unwrap()),
        "actual8K re-chunked history"
    );
    target.consume_sequence(&output).unwrap();
    assert_eq!(
        final_hash,
        hash(&target.logits_snapshot().unwrap()),
        "actual8K plus generated-prefix continuation"
    );
    assert_eq!(target.consumed_tokens(), prompt.len() + output.len());
    assert_eq!(
        before.successful_allocations,
        cuda_device_allocation_snapshot().successful_allocations
    );
    let peak = cuda_device_allocation_snapshot().peak_live_bytes;
    drop(target);
    let after = cuda_device_allocation_snapshot();
    assert_eq!(after.live_bytes, baseline.live_bytes);
    assert_eq!(after.live_allocations, baseline.live_allocations);
    assert_eq!(
        after.successful_allocations - baseline.successful_allocations,
        after.successful_frees - baseline.successful_frees
    );
    std::fs::write(root.join("result.json"),serde_json::to_vec_pretty(&json!({
        "status":"PASS_CHECKPOINT_ACTUAL_CONTEXT_REPLAY_CLEANUP","reference_sha256":format!("{:x}",Sha256::digest(&source)),
        "targets":[{"info":load_info,"request":request}],"exact_generated_prefix_replay":true,
        "actual_prompt_tokens":prompt.len(),"replay_wall_s":replay_started.elapsed().as_secs_f64(),
        "peak_owned_live_bytes":peak,"live_bytes_after":after.live_bytes,"live_allocations_after":after.live_allocations,
        "product_ready":false,"semantic_qualified":false,"http_tested":false,"dflash_enabled":false,"vision_enabled":false})).unwrap()).unwrap();
}

#[test]
fn expanded_qpn_profiles_keep_geometry_identity_and_original_gate() {
    use augur_model_qwen35::{checkpoint_linear_specs, CheckpointLinearStorage};
    for policy in [
        CheckpointProjectionPolicy::AugurQpnAllF16PostScale,
        CheckpointProjectionPolicy::AugurQpnAllResidualF32,
        CheckpointProjectionPolicy::AugurQpnAllRefinedF64,
        CheckpointProjectionPolicy::AugurQpnAllRefinedFastF64,
        CheckpointProjectionPolicy::AugurQpnAllWorkspaceF64,
    ] {
        let specs = checkpoint_linear_specs(Qwen35CheckpointProfile::QuasarNvfp4);
        let count = specs
            .iter()
            .filter(|s| {
                s.storage == CheckpointLinearStorage::Nvfp4
                    && policy.admits_qpn_projection(s.rows, s.cols)
            })
            .count();
        assert_eq!(count, 400);
        assert!(policy.qpn_ffn_layer(0) && policy.qpn_ffn_layer(63));
        assert!(!policy.admits_qpn_projection(48, 5120));
        assert!(!policy.admits_qpn_projection(0, 5120));
        assert!(!policy.admits_qpn_projection(5120, 63));
        let identity = CheckpointExpectedIdentity {
            config_sha256: "0".repeat(64),
            index_sha256: "0".repeat(64),
            shards: Default::default(),
        };
        assert!(CheckpointCudaTarget::open_with_projection_policy(
            "unopened",
            Qwen35CheckpointProfile::NvidiaMixed,
            identity,
            512,
            policy
        )
        .err()
        .unwrap()
        .to_string()
        .contains("Augur-only"));
    }
    assert_eq!(
        CheckpointProjectionPolicy::default(),
        CheckpointProjectionPolicy::OriginalF32
    );
    assert!(!CheckpointProjectionPolicy::AugurQpnAllF16PostScale.qpn_residual());
    assert!(CheckpointProjectionPolicy::AugurQpnAllRefinedF64.qpn_residual());
}
