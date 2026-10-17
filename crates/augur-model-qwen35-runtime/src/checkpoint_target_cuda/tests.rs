use super::*;
use serde_json::Value;
use sha2::{Digest, Sha256};

pub(super) fn bytes(record: &Value) -> Vec<u8> {
    let data = std::fs::read(record["path"].as_str().unwrap()).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&data)),
        record["sha256"].as_str().unwrap()
    );
    data
}
pub(super) fn configure_norm(target: &mut CheckpointCudaTarget) {
    match std::env::var("AUGUR_CHECKPOINT_NORM_POLICY").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("original") => {},
        Ok("warp-cached") | Ok("warp-cached-qpn-input") => {
            target.enable_exact_warp_tree_norm().unwrap();
            assert!(target.enable_exact_warp_tree_norm().is_err());
            if std::env::var("AUGUR_CHECKPOINT_NORM_POLICY").as_deref()==Ok("warp-cached-qpn-input") {
                #[cfg(feature = "qwen35-prefill-cublas")]
                {
                    target.enable_fused_input_qpn_norm().unwrap();
                    assert!(target.enable_fused_input_qpn_norm().is_err());
                }
                #[cfg(not(feature = "qwen35-prefill-cublas"))]
                panic!("fused QPN input requires explicit feature");
            }
        },
        _ => panic!("invalid explicit norm policy"),
    }
}

fn f64_values(record: &Value) -> Vec<f64> {
    let data = bytes(record);
    assert_eq!(
        data.len(),
        record["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .product::<usize>()
            * 8
    );
    data.chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn compare(actual: &[f32], expected: &[f64], label: &str) -> Value {
    assert_eq!(actual.len(), expected.len());
    assert!(actual.iter().all(|v| v.is_finite()) && expected.iter().all(|v| v.is_finite()));
    let (mut error2, mut norm2, mut max_abs, mut max_ref) = (0., 0., 0.0_f64, 0.0_f64);
    let mut worst_index = 0;
    for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let d = f64::from(a) - e;
        error2 += d * d;
        norm2 += e * e;
        if d.abs() > max_abs {
            worst_index = index;
        }
        max_abs = max_abs.max(d.abs());
        max_ref = max_ref.max(e.abs());
    }
    let relative_l2 = (error2 / norm2.max(1e-300)).sqrt();
    let scaled_max = max_abs / (1. + max_ref);
    // Preserve the fixed gate, but collect every finite layer/logits result
    // before failing. Generation is skipped on any math failure; the caller
    // writes a FAIL ledger after owner cleanup and then fails the test.
    let passed = relative_l2 <= 1e-3 && scaled_max <= 1e-3;
    serde_json::json!({"label":label,"relative_l2":relative_l2,"max_abs":max_abs,
        "scaled_max":scaled_max,"elements":actual.len(),"passed":passed,
        "worst_index":worst_index,"worst_actual":actual[worst_index],"worst_reference":expected[worst_index]})
}

#[test]
fn math_ledger_keeps_both_original_limits() {
    assert_eq!(compare(&[1.0], &[1.0], "same")["passed"], true);
    let relative = compare(&[0.0001], &[0.0], "relative failure");
    assert!(relative["scaled_max"].as_f64().unwrap() < 1e-3);
    assert_eq!(relative["passed"], false);
    let mut actual = vec![1.0; 100];
    actual[37] = 1.003;
    let spike = compare(&actual, &vec![1.0; 100], "scaled-max failure");
    assert!(spike["relative_l2"].as_f64().unwrap() < 1e-3);
    assert_eq!(spike["passed"], false);
    assert_eq!(spike["worst_index"], 37);
}
pub(super) fn hash(values: &[f32]) -> String {
    format!(
        "{:x}",
        Sha256::digest(
            values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>()
        )
    )
}
pub(super) fn top(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(ia, a), (ib, b)| a.partial_cmp(b).expect("finite logits").then(ib.cmp(ia)))
        .unwrap()
        .0
}

pub(super) fn projection_policy() -> CheckpointProjectionPolicy {
    match std::env::var("AUGUR_CHECKPOINT_PROJECTION_POLICY").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("original") => {
            CheckpointProjectionPolicy::OriginalF32
        }
        Ok("nvidia-fp8-warp") => CheckpointProjectionPolicy::NvidiaFp8WarpF32,
        Ok("qpn-ffn") => CheckpointProjectionPolicy::QpnFfnF16,
        Ok("qpn-ffn-postscale") => CheckpointProjectionPolicy::QpnFfnF16PostScale,
        Ok("augur-qpn-all-workspace-shape-pair") => CheckpointProjectionPolicy::AugurQpnAllWorkspaceShapeSmallPairF64,
        Ok("augur-qpn-all-workspace-shape-pair-prefetch") => CheckpointProjectionPolicy::AugurQpnAllWorkspaceShapeSmallPairPrefetchF64,
        Ok("augur-qpn-all-workspace-shape-small") => CheckpointProjectionPolicy::AugurQpnAllWorkspaceShapeSmallF64,
        Ok("augur-qpn-all-workspace-shape") => CheckpointProjectionPolicy::AugurQpnAllWorkspaceShapeF64,
        Ok("augur-qpn-all-workspace-cta-fold") => CheckpointProjectionPolicy::AugurQpnAllWorkspaceCtaFoldF64,
        Ok("augur-qpn-all-workspace") => CheckpointProjectionPolicy::AugurQpnAllWorkspaceF64,
        Ok("augur-qpn-all-refined-fast") => CheckpointProjectionPolicy::AugurQpnAllRefinedFastF64,
        Ok("augur-qpn-all-refined") => CheckpointProjectionPolicy::AugurQpnAllRefinedF64,
        Ok("augur-qpn-all-residual") => CheckpointProjectionPolicy::AugurQpnAllResidualF32,
        Ok("augur-qpn-all-postscale") => CheckpointProjectionPolicy::AugurQpnAllF16PostScale,
        Ok("nvidia-fp8-warp-qpn-ffn-postscale") => {
            CheckpointProjectionPolicy::NvidiaFp8WarpQpnFfnF16PostScale
        }
        Ok("nvidia-fp8-warp-qpn-ffn") => CheckpointProjectionPolicy::NvidiaFp8WarpQpnFfnF16,
        Ok("nvidia-fp8-warp-qpn-ffn-tail8") => {
            CheckpointProjectionPolicy::NvidiaFp8WarpQpnFfnF16Tail8
        }
        _ => panic!("invalid explicit checkpoint projection policy"),
    }
}

#[test]
#[ignore = "exclusive32GiB SM70 and independent complete-target CPU reference"]
fn real_complete_checkpoint_targets_math_chat_replay_cleanup() {
    let reference = std::env::var_os("AUGUR_CHECKPOINT_TARGET_REFERENCE").expect("reference");
    let reference_bytes = std::fs::read(reference).unwrap();
    let manifest: Value = serde_json::from_slice(&reference_bytes).unwrap();
    // A completed per-model fixture is independently usable while the other
    // model's CPU reference is still running. Every tensor file is SHA-bound.
    let cases = if let Some(cases) = manifest.get("cases") {
        cases.as_array().unwrap().clone()
    } else {
        vec![manifest]
    };
    assert!((1..=2).contains(&cases.len()));
    let root =
        std::path::PathBuf::from(std::env::var_os("AUGUR_CHECKPOINT_TARGET_OUTPUT").expect("output"));
    std::fs::create_dir(&root).unwrap();
    let baseline = cuda_device_allocation_snapshot();
    let mut reports = Vec::new();
    let mut math_failures = Vec::new();
    let capture_final_branches =
        match std::env::var("AUGUR_CHECKPOINT_CAPTURE_FINAL_BRANCHES").as_deref() {
            Err(std::env::VarError::NotPresent) | Ok("0") => false,
            Ok("1") => true,
            _ => panic!("invalid final-layer branch capture option"),
        };
    for case in cases {
        let audit_bytes = std::fs::read(case["audit_path"].as_str().unwrap()).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(&audit_bytes)),
            case["audit_sha256"].as_str().unwrap()
        );
        let a: Value = serde_json::from_slice(&audit_bytes).unwrap();
        let profile = match case["profile"].as_str().unwrap() {
            "quasar" => Qwen35CheckpointProfile::QuasarNvfp4,
            "nvidia" => Qwen35CheckpointProfile::NvidiaMixed,
            _ => panic!("profile"),
        };
        for record in case["metadata"].as_object().unwrap().values() {
            bytes(record);
        }
        let tokenizer =
            tokenizers::Tokenizer::from_bytes(bytes(&case["metadata"]["tokenizer.json"])).unwrap();
        let expected = CheckpointExpectedIdentity {
            config_sha256: a["metadata"]["config.json"]["sha256"]
                .as_str()
                .unwrap()
                .into(),
            index_sha256: a["metadata"]["model.safetensors.index.json"]["sha256"]
                .as_str()
                .unwrap()
                .into(),
            shards: a["shards"]
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
        let kv_policy = match case["kv_dtype"].as_str() {
            None | Some("F32") => CheckpointKvPolicy::F32,
            Some("Q8_0") => CheckpointKvPolicy::Q8_0,
            _ => panic!("unknown target reference KV policy"),
        };
        let mut target = CheckpointCudaTarget::open_with_execution_policy(
            a["model_dir"].as_str().unwrap(),
            profile,
            expected,
            512,
            projection_policy(),
            kv_policy,
        )
        .unwrap();
        configure_norm(&mut target);
        match std::env::var("AUGUR_CHECKPOINT_FFN_BATCH").as_deref() {
            Err(std::env::VarError::NotPresent) | Ok("0") => {}
            Ok("8") => {
                #[cfg(feature = "qwen35-prefill-cublas")]
                target.enable_batched_ffn8().unwrap();
                #[cfg(not(feature = "qwen35-prefill-cublas"))]
                panic!("FFN batch feature absent");
            }
            _ => panic!("invalid target math batch width"),
        }
        let before = cuda_device_allocation_snapshot();
        assert!(target.logits_snapshot().is_err());
        assert!(target.consume(248320, true, &[]).is_err());
        assert!(target.consume(1, true, &[2, 1]).is_err());
        assert_eq!(target.consumed_tokens(), 0);
        let ids: Vec<u32> = case["token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(ids.len(), 4);
        assert_eq!(case["layers"].as_array().unwrap().len(), 64);
        let refs: Vec<Vec<f64>> = case["layers"]
            .as_array()
            .unwrap()
            .iter()
            .map(f64_values)
            .collect();
        let expected_logits = f64_values(&case["logits"]);
        let capture_layers = (0..64).collect::<Vec<_>>();
        let mut metrics = Vec::new();
        let mut hashes = BTreeMap::new();
        let mut final_branch_observations = Vec::new();
        for pass in 0..2 {
            if pass == 1 {
                target.reset_empty().unwrap();
                assert!(target.logits_snapshot().is_err());
            }
            for (index, &token) in ids.iter().enumerate() {
                let layers = target.consume(token, true, &capture_layers).unwrap();
                assert_eq!(layers.len(), 64);
                assert_eq!(target.consumed_tokens(), index + 1);
                if pass == 0 && capture_final_branches {
                    // Layer63 is the last layer: its ping-pong input/output are
                    // still valid here. Earlier layers' shared hidden buffers
                    // cannot be observed after the entire step.
                    let Layer::Full(layer) = &mut target.layers[63] else {
                        panic!("last target layer must be full attention");
                    };
                    let mut stages = BTreeMap::new();
                    for stage in [
                        "input",
                        "normalized",
                        "q_full",
                        "query",
                        "key",
                        "value",
                        "attention",
                        "gated",
                        "attention_output",
                        "attention_residual",
                        "post_norm",
                        "ffn_output",
                        "output",
                    ] {
                        let values = layer.readback(stage).unwrap();
                        assert!(values.iter().all(|v| v.is_finite()));
                        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                        let path = root.join(format!(
                            "{}-token{index}-layer63-{stage}.f32",
                            case["profile"].as_str().unwrap()
                        ));
                        std::fs::write(&path, &data).unwrap();
                        stages.insert(
                            stage,
                            serde_json::json!({"path":path,
                            "sha256":format!("{:x}",Sha256::digest(&data)),"shape":[values.len()]}),
                        );
                    }
                    final_branch_observations
                        .push(serde_json::json!({"token_index":index,"layer":63,"stages":stages}));
                }
                for (layer, actual) in layers {
                    let label = format!("{profile:?} token{index} layer{layer}");
                    let result = compare(
                        &actual,
                        &refs[layer][index * 5120..(index + 1) * 5120],
                        &label,
                    );
                    if pass == 0 {
                        hashes.insert((index, layer), hash(&actual));
                        metrics.push(result);
                    } else {
                        assert_eq!(hashes[&(index, layer)], hash(&actual));
                    }
                }
                let logits = target.logits_snapshot().unwrap();
                let mut result = compare(
                    &logits,
                    &expected_logits[index * 248320..(index + 1) * 248320],
                    &format!("{profile:?} token{index} logits"),
                );
                let selected = top(&logits);
                let expected_top = case["argmax"][index].as_u64().unwrap() as usize;
                result["argmax"] = serde_json::json!(selected);
                result["reference_argmax"] = serde_json::json!(expected_top);
                result["argmax_equal"] = serde_json::json!(selected == expected_top);
                if pass == 0 {
                    hashes.insert((index, 64), hash(&logits));
                    metrics.push(result);
                } else {
                    assert_eq!(hashes[&(index, 64)], hash(&logits));
                }
            }
        }
        if target.info()["ffn_batch_max_rows"] == 8 {
            // Each prefix verifies an actually batched final logits vector,
            // against the scalar whose64 layer outputs were checked above.
            for end in 1..=ids.len() {
                target.reset_empty().unwrap();
                target.consume_sequence(&ids[..end]).unwrap();
                assert_eq!(
                    hashes[&(end - 1, 64)],
                    hash(&target.logits_snapshot().unwrap())
                );
            }
        }
        let failed: Vec<_> = metrics
            .iter()
            .filter(|m| m["passed"] != true || m.get("argmax_equal") == Some(&Value::Bool(false)))
            .cloned()
            .collect();
        let math_passed = failed.is_empty();
        math_failures.extend(failed);
        let eos: Vec<u32> = case["eos_token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let mut generated = Vec::new();
        for request in case["generation"].as_array().unwrap() {
            if !math_passed {
                break;
            }
            target.reset_empty().unwrap();
            let prompt: Vec<u32> = request["token_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            let encoded = tokenizer
                .encode(request["canonical_prompt"].as_str().unwrap(), false)
                .unwrap();
            assert_eq!(encoded.get_ids(), prompt);
            let budget = request["max_new_tokens"].as_u64().unwrap() as usize;
            assert!(prompt.len() + budget <= 512);
            let started = Instant::now();
            for (i, &id) in prompt.iter().enumerate() {
                target.consume(id, i + 1 == prompt.len(), &[]).unwrap();
            }
            let prefill_s = started.elapsed().as_secs_f64();
            let mut output = Vec::new();
            let mut selected_steps = Vec::new();
            let mut stop = "max_new_tokens";
            let mut pending = 0;
            for i in 0..budget {
                let logits = target.logits_snapshot().unwrap();
                let id = top(&logits) as u32;
                pending = id;
                selected_steps.push(serde_json::json!({"id":id,"logits_sha256":hash(&logits)}));
                if eos.contains(&id) {
                    stop = "eos";
                    break;
                }
                output.push(id);
                if i + 1 < budget {
                    target.consume(id, true, &[]).unwrap();
                }
            }
            assert!(!output.is_empty());
            let text = tokenizer.decode(&output, false).unwrap();
            let expected_consumed = prompt.len() + output.len() - usize::from(stop != "eos");
            assert_eq!(target.consumed_tokens(), expected_consumed);
            let result = serde_json::json!({"family":request["family"],"prompt":request["prompt"],
                "canonical_prompt":request["canonical_prompt"],"input_token_ids":prompt,
                "output_token_ids":output,"text":text,"stop_reason":stop,"pending_token":pending,
                "consumed_tokens":target.consumed_tokens(),"prefill_wall_s":prefill_s,
                "whole_diagnostic_wall_s":started.elapsed().as_secs_f64(),"steps":selected_steps,
                "timing_includes_full_logits_D2H_and_host_argmax":true,"semantic_qualified":false});
            std::fs::write(
                root.join(format!(
                    "{}-{}.json",
                    case["profile"].as_str().unwrap(),
                    request["family"].as_str().unwrap()
                )),
                serde_json::to_vec_pretty(&result).unwrap(),
            )
            .unwrap();
            generated.push(result);
            println!(
                "CHECKPOINT_TARGET_CHAT={profile:?} {} captured",
                request["family"]
            );
        }
        assert_eq!(
            before.successful_allocations,
            cuda_device_allocation_snapshot().successful_allocations
        );
        reports.push(
            serde_json::json!({"info":target.info(),"math":metrics,"generation":generated,
            "math_passed":math_passed,"generation_skipped_math_failure":!math_passed,
            "final_layer_branch_observations":final_branch_observations,
            "exact_reset_replay":true,"model_loaded_once":true}),
        );
        drop(target);
        let after = cuda_device_allocation_snapshot();
        assert_eq!(baseline.live_bytes, after.live_bytes);
        assert_eq!(baseline.live_allocations, after.live_allocations);
        println!("CHECKPOINT_TARGET={profile:?} math_passed={math_passed} REPLAY_CLEANUP PASS");
    }
    let after = cuda_device_allocation_snapshot();
    assert_eq!(
        after.successful_allocations - baseline.successful_allocations,
        after.successful_frees - baseline.successful_frees
    );
    std::fs::write(root.join("result.json"),serde_json::to_vec_pretty(&serde_json::json!({
        "status":if math_failures.is_empty() {"PASS_COMPLETE_CHECKPOINT_TARGET_MATH_CHAT_CAPTURE_REPLAY_CLEANUP"} else {"FAIL_COMPLETE_CHECKPOINT_TARGET_MATH"},
        "math_failures":math_failures,
        "reference_sha256":format!("{:x}",Sha256::digest(&reference_bytes)),"targets":reports,
        "math_relative_l2_limit":1e-3,"math_scaled_max_limit":1e-3,
        "live_bytes_after":after.live_bytes,"live_allocations_after":after.live_allocations,
        "complete64_layer_target_executed":true,"product_ready":false,"semantic_qualified":false,
        "dflash_enabled":false,"vision_enabled":false,"native_w4a4":false,
        "kv_dtype":reports[0]["info"]["kv_dtype"],"kv_capacity":512})).unwrap()).unwrap();
    assert!(
        math_failures.is_empty(),
        "{} fixed-bound math/argmax failures; see result.json",
        math_failures.len()
    );
}

use std::collections::BTreeMap;
