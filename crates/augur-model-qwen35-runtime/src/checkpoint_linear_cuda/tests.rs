use super::*;
use augur_backend::cuda_device_allocation_snapshot;
use augur_model_qwen35::{CheckpointExpectedIdentity, Qwen35CheckpointProfile};
use sha2::{Digest, Sha256};
use std::path::Path;

fn checked_file(value: &serde_json::Value) -> Vec<u8> {
    let bytes = std::fs::read(value["path"].as_str().unwrap()).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        value["sha256"].as_str().unwrap()
    );
    bytes
}

#[test]
#[ignore = "exclusive CUDA, authenticated new checkpoints and independent CPU layer fixtures"]
fn real_checkpoint_linear_layers_reference_replay_cleanup() {
    let reference = std::env::var_os("AUGUR_CHECKPOINT_LAYER_REFERENCE").expect("reference manifest");
    let manifest_bytes = std::fs::read(reference).unwrap();
    let manifest: serde_json::Value = serde_json::from_slice(&manifest_bytes).unwrap();
    assert_eq!(
        manifest["status"],
        "PASS_CPU_COMPLETE_LINEAR_LAYER_REFERENCE_ONLY"
    );
    let root =
        std::path::PathBuf::from(std::env::var_os("AUGUR_CHECKPOINT_LAYER_OUTPUT").expect("output"));
    std::fs::create_dir(&root).unwrap();
    let baseline = cuda_device_allocation_snapshot();
    let mut results = Vec::new();
    assert_eq!(manifest["cases"].as_array().unwrap().len(), 4);
    for case in manifest["cases"].as_array().unwrap() {
        let profile = match case["profile"].as_str().unwrap() {
            "quasar" => Qwen35CheckpointProfile::QuasarNvfp4,
            "nvidia" => Qwen35CheckpointProfile::NvidiaMixed,
            _ => panic!("unknown profile"),
        };
        let audit_bytes = std::fs::read(case["audit_path"].as_str().unwrap()).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(&audit_bytes)),
            case["audit_sha256"].as_str().unwrap()
        );
        let audit: serde_json::Value = serde_json::from_slice(&audit_bytes).unwrap();
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
        let mut reader = CheckpointArtifact::open(
            Path::new(audit["model_dir"].as_str().unwrap()),
            profile,
            expected,
        )
        .unwrap();
        let tokens = case["tokens"].as_u64().unwrap() as usize;
        assert_eq!(tokens, 4);
        let layer_index = case["layer"].as_u64().unwrap() as usize;
        let input_bytes = checked_file(&case["stages"]["input"]);
        assert_eq!(input_bytes.len(), tokens * 5120 * 4);
        assert!(reader.load_embedding_row(248320).is_err());
        assert!(reader
            .load_dense_execution("model.language_model.embed_tokens.weight")
            .is_err());
        for (i, token) in case["token_ids"].as_array().unwrap().iter().enumerate() {
            let row = reader
                .load_embedding_row(token.as_u64().unwrap() as usize)
                .unwrap();
            let expected: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
            assert_eq!(&input_bytes[i * 5120 * 4..(i + 1) * 5120 * 4], &expected);
        }
        let references: BTreeMap<String, Vec<f64>> = case["stages"]
            .as_object()
            .unwrap()
            .iter()
            .filter(|(name, _)| name.as_str() != "input")
            .map(|(name, value)| {
                let bytes = checked_file(value);
                let shape = value["shape"].as_array().unwrap();
                assert_eq!(shape[0].as_u64().unwrap() as usize, tokens);
                assert_eq!(
                    bytes.len(),
                    tokens * shape[1].as_u64().unwrap() as usize * 8
                );
                (
                    name.clone(),
                    bytes
                        .chunks_exact(8)
                        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
                        .collect(),
                )
            })
            .collect();
        let stream = Rc::new(OwnedCudaStream::create(0).unwrap());
        let input = empty(&stream, 5120).unwrap();
        let output = empty(&stream, 5120).unwrap();
        assert!(CheckpointLinearCudaLayer::load(
            &mut reader,
            3,
            Rc::clone(&stream),
            Rc::clone(&input),
            Rc::clone(&output)
        )
        .is_err());
        assert!(CheckpointLinearCudaLayer::load(
            &mut reader,
            layer_index,
            Rc::clone(&stream),
            Rc::clone(&input),
            Rc::clone(&input)
        )
        .is_err());
        let mut layer = CheckpointLinearCudaLayer::load(
            &mut reader,
            layer_index,
            Rc::clone(&stream),
            Rc::clone(&input),
            Rc::clone(&output),
        )
        .unwrap();
        let weak_input = Rc::downgrade(&input);
        let weak_output = Rc::downgrade(&output);
        drop(output);
        assert!(weak_output.upgrade().is_some());
        let before = cuda_device_allocation_snapshot();
        let mut hashes = BTreeMap::new();
        for pass in 0..2 {
            if pass == 1 {
                layer.poisoned = true;
                assert!(layer.submit_for_plan().is_err());
                layer.reset_empty().unwrap();
                assert_eq!(layer.submitted_tokens(), 0);
            }
            for token in 0..tokens {
                input
                    .copy_from_host_sync(
                        &stream,
                        &input_bytes[token * 5120 * 4..(token + 1) * 5120 * 4],
                    )
                    .unwrap();
                layer.submit_for_plan().unwrap();
                assert_eq!(layer.submitted_tokens(), token + 1);
                for (stage, reference) in &references {
                    let actual = layer.readback(stage).unwrap();
                    assert_eq!(actual.len() * tokens, reference.len());
                    assert!(actual.iter().all(|x| x.is_finite()));
                    let expected = &reference[token * actual.len()..(token + 1) * actual.len()];
                    assert!(expected.iter().all(|x| x.is_finite()));
                    let mut error2 = 0.;
                    let mut norm2 = 0.;
                    let mut max_abs = 0.0_f64;
                    let mut max_ref = 0.0_f64;
                    for (&a, &r) in actual.iter().zip(expected) {
                        let d = f64::from(a) - r;
                        error2 += d * d;
                        norm2 += r * r;
                        max_abs = max_abs.max(d.abs());
                        max_ref = max_ref.max(r.abs());
                    }
                    let relative_l2 = (error2 / norm2.max(1e-300)).sqrt();
                    let scaled_max = max_abs / (1. + max_ref);
                    assert!(relative_l2<=2e-4 && scaled_max<=2e-4,
                        "{:?} layer{layer_index} token{token} {stage}: relative_l2={relative_l2} scaled_max={scaled_max}",profile);
                    let hash = format!(
                        "{:x}",
                        Sha256::digest(
                            actual
                                .iter()
                                .flat_map(|v| v.to_le_bytes())
                                .collect::<Vec<_>>()
                        )
                    );
                    if pass == 0 {
                        hashes.insert((token, stage.clone()), hash);
                        results.push(serde_json::json!({"profile":case["profile"],"layer":layer_index,
                            "token":token,"stage":stage,"elements":actual.len(),"relative_l2":relative_l2,
                            "max_abs":max_abs,"scaled_max":scaled_max}));
                    } else {
                        assert_eq!(
                            hashes[&(token, stage.clone())],
                            hash,
                            "reset/replay changed {stage}"
                        );
                    }
                }
            }
        }
        let after = cuda_device_allocation_snapshot();
        assert_eq!(before.successful_allocations, after.successful_allocations);
        layer.synchronize().unwrap();
        drop(input);
        assert!(weak_input.upgrade().is_some());
        drop(layer);
        assert!(weak_input.upgrade().is_none() && weak_output.upgrade().is_none());
        println!("CHECKPOINT_LINEAR_LAYER={profile:?} layer{layer_index} four_tokens_replay PASS");
    }
    // Check the retained GGUF interpretation on the same native binary.
    let legacy = crate::real_linear_step_cuda::run_qwen35_real_linear_step_cuda(
        &crate::real_linear_step_cuda::Qwen35RealLinearStepCudaConfig {
            gguf_path: "E:/qwen35-27b-mixed-nvfp4-q8.gguf".into(),
            config_path: "F:/qwen3.8 27B/config.json".into(),
            token_id: 9707,
            device_ordinal: 0,
            warmup: 1,
            iterations: 2,
            source_commit: "SOURCE_BYTES_BOUND_BY_OUTER_HARNESS".into(),
        },
    )
    .unwrap();
    assert_eq!(legacy.status, "PASS_COMPONENT");
    std::fs::write(
        root.join("legacy-recurrent.json"),
        serde_json::to_vec_pretty(&legacy).unwrap(),
    )
    .unwrap();
    let after = cuda_device_allocation_snapshot();
    assert_eq!(baseline.live_bytes, after.live_bytes);
    assert_eq!(baseline.live_allocations, after.live_allocations);
    assert_eq!(
        after.successful_allocations - baseline.successful_allocations,
        after.successful_frees - baseline.successful_frees
    );
    std::fs::write(
        root.join("result.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
        "status":"PASS_NEW_CHECKPOINT_LINEAR_LAYERS_REFERENCE_REPLAY_OWNER_CLEANUP",
        "reference_manifest_sha256":format!("{:x}",Sha256::digest(&manifest_bytes)),
        "relative_l2_limit":2e-4,"scaled_max_limit":2e-4,"cases":results,
        "live_bytes_after":after.live_bytes,"live_allocations_after":after.live_allocations,
        "product_ready":false,"semantic_qualified":false,"whole_target_executed":false,
        "native_w4a4":false,"layer50_input":"synthetic_not_upstream_capture"}))
        .unwrap(),
    )
    .unwrap();
}
