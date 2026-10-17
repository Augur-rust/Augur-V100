use super::*;
use augur_backend::cuda_device_allocation_snapshot;
use augur_model_qwen35::{CheckpointExpectedIdentity, Qwen35CheckpointProfile};
use serde_json::Value;
use sha2::{Digest, Sha256};

fn checked(record: &Value) -> Vec<u8> {
    let b = std::fs::read(record["path"].as_str().unwrap()).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&b)),
        record["sha256"].as_str().unwrap()
    );
    b
}
fn reader(case: &Value) -> CheckpointArtifact {
    let b = std::fs::read(case["audit_path"].as_str().unwrap()).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&b)),
        case["audit_sha256"].as_str().unwrap()
    );
    let a: Value = serde_json::from_slice(&b).unwrap();
    let profile = match case["profile"].as_str().unwrap() {
        "quasar" => Qwen35CheckpointProfile::QuasarNvfp4,
        "nvidia" => Qwen35CheckpointProfile::NvidiaMixed,
        _ => panic!("bad profile"),
    };
    CheckpointArtifact::open(
        a["model_dir"].as_str().unwrap(),
        profile,
        CheckpointExpectedIdentity {
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
        },
    )
    .unwrap()
}
fn references(case: &Value, tokens: usize) -> BTreeMap<String, Vec<f64>> {
    case["stages"]
        .as_object()
        .unwrap()
        .iter()
        .filter(|(name, _)| name.as_str() != "input")
        .map(|(name, r)| {
            let b = checked(r);
            assert_eq!(r["shape"][0].as_u64().unwrap() as usize, tokens);
            assert_eq!(
                b.len(),
                tokens * r["shape"][1].as_u64().unwrap() as usize * 8
            );
            (
                name.clone(),
                b.chunks_exact(8)
                    .map(|x| f64::from_le_bytes(x.try_into().unwrap()))
                    .collect(),
            )
        })
        .collect()
}
fn compare(actual: &[f32], expected: &[f64], label: &str) -> Value {
    assert_eq!(actual.len(), expected.len());
    assert!(actual.iter().all(|v| v.is_finite()) && expected.iter().all(|v| v.is_finite()));
    let (mut error2, mut norm2, mut max_abs, mut max_ref) = (0., 0., 0.0_f64, 0.0_f64);
    for (&a, &r) in actual.iter().zip(expected) {
        let delta = f64::from(a) - r;
        error2 += delta * delta;
        norm2 += r * r;
        max_abs = max_abs.max(delta.abs());
        max_ref = max_ref.max(r.abs());
    }
    let relative_l2 = (error2 / norm2.max(1e-300)).sqrt();
    let scaled_max = max_abs / (1. + max_ref);
    serde_json::json!({"label":label,"elements":actual.len(),"relative_l2":relative_l2,
        "max_abs":max_abs,"scaled_max":scaled_max,
        "passed":relative_l2 <= 2e-4 && scaled_max <= 2e-4})
}

fn reference_kv8_row(actual: &[f32]) -> Vec<u8> {
    assert_eq!(actual.len(), 1024);
    let mut bytes = Vec::with_capacity(1088);
    for block in actual.chunks_exact(32) {
        assert!(block.iter().all(|v| v.is_finite()));
        let maximum = block.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
        let inverse = if maximum != 0.0 { 127.0 / maximum } else { 0.0 };
        let scale = half::f16::from_f32(maximum / 127.0);
        assert!(scale.is_finite() && inverse.is_finite());
        bytes.extend_from_slice(&scale.to_bits().to_le_bytes());
        for &v in block {
            bytes.push((v * inverse).round_ties_even().clamp(-128., 127.) as i8 as u8);
        }
    }
    bytes
}

// Independent F64 attention, conditioned on the ACTUAL GPU projection outputs.
// This isolates the discontinuous quantizer from upstream rounding; it does
// not replace the end-to-end F64 layer comparison or confer model qualification.
fn conditional_kv8_reference(
    layer: &mut CheckpointFullCudaLayer,
    token: usize,
    key_bytes: &mut [u8],
    value_bytes: &mut [u8],
) -> Vec<Value> {
    for (stage, cache, expected) in [
        ("key", "key_cache", &mut *key_bytes),
        ("value", "value_cache", &mut *value_bytes),
    ] {
        let actual = layer.readback(stage).unwrap();
        let encoded = reference_kv8_row(&actual);
        expected[token * 1088..(token + 1) * 1088].copy_from_slice(&encoded);
        let mut bytes = vec![0; expected.len()];
        layer.buffers[cache]
            .copy_to_host_sync(&layer.stream, &mut bytes)
            .unwrap();
        assert_eq!(
            &bytes, expected,
            "same-input cache bytes {stage} token{token}"
        );
    }
    let decode = |bytes: &[u8]| -> Vec<f64> {
        bytes
            .chunks_exact(34)
            .flat_map(|b| {
                let scale = half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f64();
                b[2..].iter().map(move |&x| (x as i8 as f64) * scale)
            })
            .collect()
    };
    let keys = decode(key_bytes);
    let values = decode(value_bytes);
    let query = layer.readback("query").unwrap();
    let qg = layer.readback("q_full").unwrap();
    let mut expected = vec![0_f64; 6144];
    let mut gated = vec![0_f64; 6144];
    for head in 0..24 {
        let scores: Vec<f64> = (0..=token)
            .map(|pos| {
                (0..256)
                    .map(|d| {
                        f64::from(query[head * 256 + d]) * keys[pos * 1024 + head / 6 * 256 + d]
                    })
                    .sum::<f64>()
                    / 16.
            })
            .collect();
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let probabilities: Vec<f64> = scores.iter().map(|s| (s - maximum).exp()).collect();
        let denominator = probabilities.iter().sum::<f64>();
        for d in 0..256 {
            let index = head * 256 + d;
            expected[index] = (0..=token)
                .map(|pos| probabilities[pos] * values[pos * 1024 + head / 6 * 256 + d])
                .sum::<f64>()
                / denominator;
            gated[index] = expected[index] / (1. + (-f64::from(qg[6144 + index])).exp());
        }
    }
    [("attention", expected), ("gated", gated)]
        .into_iter()
        .map(|(stage, expected)| {
            let result = compare(
                &layer.readback(stage).unwrap(),
                &expected,
                &format!("conditional KV8 {stage} token{token}"),
            );
            assert_eq!(
                result["passed"], true,
                "same-input KV8 arithmetic: {result}"
            );
            result
        })
        .collect()
}

#[test]
fn independent_kv8_reference_uses_rne_and_f16_scale() {
    let mut row = vec![0_f32; 1024];
    row[..7].copy_from_slice(&[127., 0.5, -0.5, 1.5, -1.5, 2.5, 3.5]);
    let bytes = reference_kv8_row(&row);
    assert_eq!(&bytes[..2], &half::f16::ONE.to_bits().to_le_bytes());
    assert_eq!(&bytes[2..9], &[127, 0, 0, 2, 254, 2, 4]);
    assert!(bytes[34..].iter().all(|&b| b == 0));
}
fn bits_hash(actual: &[f32]) -> String {
    format!(
        "{:x}",
        Sha256::digest(
            actual
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>()
        )
    )
}

#[test]
#[ignore = "exclusive CUDA and audited full-layer/head CPU references"]
fn real_checkpoint_full_layers_heads_replay_cleanup() {
    let path = std::env::var_os("AUGUR_CHECKPOINT_FULL_REFERENCE").expect("reference");
    let bytes = std::fs::read(path).unwrap();
    let manifest: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        manifest["status"],
        "PASS_CPU_FULL_LAYER_AND_HEAD_REFERENCE_ONLY"
    );
    assert_eq!(manifest["cases"].as_array().unwrap().len(), 4);
    assert_eq!(manifest["heads"].as_array().unwrap().len(), 2);
    let root =
        std::path::PathBuf::from(std::env::var_os("AUGUR_CHECKPOINT_FULL_OUTPUT").expect("output"));
    std::fs::create_dir(&root).unwrap();
    let baseline = cuda_device_allocation_snapshot();
    let mut results = Vec::new();
    let mut conditional = Vec::new();
    for case in manifest["cases"].as_array().unwrap() {
        let mut reader = reader(case);
        let n = case["tokens"].as_u64().unwrap() as usize;
        let capacity = case["capacity"].as_u64().unwrap() as usize;
        assert_eq!(n, 8);
        assert_eq!(capacity, n);
        let layer_index = case["layer"].as_u64().unwrap() as usize;
        let absolute = case["absolute_start"].as_u64().unwrap() as usize;
        let inputs = checked(&case["stages"]["input"]);
        assert_eq!(inputs.len(), n * 5120 * 4);
        let refs = references(case, n);
        let stream = Rc::new(OwnedCudaStream::create(0).unwrap());
        let input = empty(&stream, 5120).unwrap();
        let output = empty(&stream, 5120).unwrap();
        for (bad_layer, bad_capacity, bad_start) in
            [(0, n, 0), (3, 0, 0), (3, 4097, 0), (3, n, 262144)]
        {
            assert!(CheckpointFullCudaLayer::load(
                &mut reader,
                bad_layer,
                bad_capacity,
                bad_start,
                Rc::clone(&stream),
                Rc::clone(&input),
                Rc::clone(&output)
            )
            .is_err());
        }
        assert!(CheckpointFullCudaLayer::load(
            &mut reader,
            layer_index,
            n,
            absolute,
            Rc::clone(&stream),
            Rc::clone(&input),
            Rc::clone(&input)
        )
        .is_err());
        let kv_policy = match case["kv_dtype"].as_str() {
            None | Some("F32") => CheckpointKvPolicy::F32,
            Some("Q8_0") => CheckpointKvPolicy::Q8_0,
            _ => panic!("unknown fixture KV policy"),
        };
        let ffn =
            CheckpointFfnContext::new(Rc::clone(&stream), CheckpointProjectionPolicy::OriginalF32)
                .unwrap();
        let mut layer = CheckpointFullCudaLayer::load_with_cache_context(
            &mut reader,
            layer_index,
            capacity,
            absolute,
            Rc::clone(&stream),
            Rc::clone(&input),
            Rc::clone(&output),
            &ffn,
            kv_policy,
            None,
        )
        .unwrap();
        let weak_in = Rc::downgrade(&input);
        let weak_out = Rc::downgrade(&output);
        drop(output);
        assert!(weak_out.upgrade().is_some());
        let before = cuda_device_allocation_snapshot();
        let mut hashes = BTreeMap::new();
        for pass in 0..2 {
            let mut expected_keys = vec![0_u8; n * 1088];
            let mut expected_values = vec![0_u8; n * 1088];
            if pass == 1 {
                layer.poisoned = true;
                assert!(layer.submit_for_plan().is_err());
                layer.reset_empty().unwrap();
                assert_eq!(layer.submitted_tokens(), 0);
            }
            for token in 0..n {
                input
                    .copy_from_host_sync(&stream, &inputs[token * 5120 * 4..(token + 1) * 5120 * 4])
                    .unwrap();
                layer.submit_for_plan().unwrap();
                assert_eq!(layer.submitted_tokens(), token + 1);
                if kv_policy == CheckpointKvPolicy::Q8_0 {
                    let metrics = conditional_kv8_reference(
                        &mut layer,
                        token,
                        &mut expected_keys,
                        &mut expected_values,
                    );
                    if pass == 0 {
                        conditional.push(serde_json::json!({"profile":case["profile"],
                        "layer":layer_index,"token":token,"actual_input_cache_bytes_exact":true,"metrics":metrics}));
                    }
                }
                for (stage, reference) in &refs {
                    let actual = layer.readback(stage).unwrap();
                    assert_eq!(reference.len(), actual.len() * n);
                    let label = format!(
                        "{} layer{layer_index} abs{} {stage}",
                        case["profile"],
                        absolute + token
                    );
                    let result = compare(
                        &actual,
                        &reference[token * actual.len()..(token + 1) * actual.len()],
                        &label,
                    );
                    let hash = bits_hash(&actual);
                    if pass == 0 {
                        results.push(result);
                        hashes.insert((token, stage.clone()), hash);
                    } else {
                        assert_eq!(hashes[&(token, stage.clone())], hash);
                    }
                }
            }
            assert!(layer.submit_for_plan().is_err());
            assert_eq!(layer.submitted_tokens(), capacity);
            assert_eq!(
                bits_hash(&layer.readback("key_cache").unwrap()),
                hashes[&(n - 1, "key_cache".into())]
            );
        }
        let after = cuda_device_allocation_snapshot();
        assert_eq!(before.successful_allocations, after.successful_allocations);
        layer.synchronize().unwrap();
        drop(input);
        assert!(weak_in.upgrade().is_some());
        drop(layer);
        assert!(weak_in.upgrade().is_none() && weak_out.upgrade().is_none());
        println!(
            "CHECKPOINT_FULL_LAYER={} layer{layer_index} abs{absolute} LEDGER_REPLAY_CLEANUP PASS",
            case["profile"]
        );
    }
    let mut heads = Vec::new();
    for case in manifest["heads"].as_array().unwrap() {
        let mut reader = reader(case);
        let n = case["tokens"].as_u64().unwrap() as usize;
        assert_eq!(n, 2);
        let inputs = checked(&case["stages"]["input"]);
        assert_eq!(inputs.len(), n * 5120 * 4);
        let refs = references(case, n);
        let stream = Rc::new(OwnedCudaStream::create(0).unwrap());
        let input = empty(&stream, 5120).unwrap();
        let logits = empty(&stream, 248320).unwrap();
        let head = CheckpointCudaOutputHead::load(
            &mut reader,
            Rc::clone(&stream),
            Rc::clone(&input),
            Rc::clone(&logits),
        )
        .unwrap();
        let before = cuda_device_allocation_snapshot();
        let mut hashes = Vec::new();
        for pass in 0..2 {
            for token in 0..n {
                input
                    .copy_from_host_sync(&stream, &inputs[token * 5120 * 4..(token + 1) * 5120 * 4])
                    .unwrap();
                head.submit_for_plan().unwrap();
                let normalized = head.normalized().unwrap();
                let mut bytes = vec![0; 248320 * 4];
                logits.copy_to_host_sync(&stream, &mut bytes).unwrap();
                let actual: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                let norm_metrics = compare(
                    &normalized,
                    &refs["normalized"][token * 5120..(token + 1) * 5120],
                    "head norm",
                );
                let mut metrics = compare(
                    &actual,
                    &refs["logits"][token * 248320..(token + 1) * 248320],
                    "head logits full vocabulary",
                );
                let argmax = actual
                    .iter()
                    .enumerate()
                    .max_by(|(ia, a), (ib, b)| a.total_cmp(b).then(ib.cmp(ia)))
                    .unwrap()
                    .0;
                assert_eq!(argmax, case["argmax"][token].as_u64().unwrap() as usize);
                metrics["profile"] = case["profile"].clone();
                metrics["input"] = serde_json::json!(token);
                metrics["argmax"] = serde_json::json!(argmax);
                if pass == 0 {
                    results.push(norm_metrics);
                    heads.push(metrics);
                    hashes.push(bits_hash(&actual));
                } else {
                    assert_eq!(hashes[token], bits_hash(&actual));
                }
            }
        }
        assert_eq!(
            before.successful_allocations,
            cuda_device_allocation_snapshot().successful_allocations
        );
        stream.synchronize().unwrap();
        println!(
            "CHECKPOINT_OUTPUT_HEAD={} complete_vocabulary PASS",
            case["profile"]
        );
    }
    let after = cuda_device_allocation_snapshot();
    assert_eq!(baseline.live_bytes, after.live_bytes);
    assert_eq!(baseline.live_allocations, after.live_allocations);
    assert_eq!(
        after.successful_allocations - baseline.successful_allocations,
        after.successful_frees - baseline.successful_frees
    );
    let failures: Vec<_> = results
        .iter()
        .chain(heads.iter())
        .filter(|m| m["passed"] != true)
        .cloned()
        .collect();
    std::fs::write(
        root.join("result.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
        "status":if failures.is_empty() {"PASS_NEW_CHECKPOINT_FULL_LAYERS_HEADS_REFERENCE_REPLAY_CLEANUP"} else {"FAIL_NEW_CHECKPOINT_FULL_LAYER_MATH"},
        "math_failures":failures,"conditional_actual_input_kv8_reference":conditional,
        "reference_sha256":format!("{:x}",Sha256::digest(&bytes)),"stages":results,"heads":heads,
        "live_bytes_after":after.live_bytes,"live_allocations_after":after.live_allocations,
        "relative_l2_limit":2e-4,"scaled_max_limit":2e-4,
        "kv_dtype":manifest.get("kv_dtype").unwrap_or(&serde_json::json!("F32")),"capacity":8,
        "high_position_case_has_eight_KV_rows_not_128K_history":true,
        "product_ready":false,"whole_target_executed":false,"native_w4a4":false}))
        .unwrap(),
    )
    .unwrap();
    assert!(
        failures.is_empty(),
        "{} fixed-bound full-layer failures; see result.json",
        failures.len()
    );
}
