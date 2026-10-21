//! CPU-only profile/weight contract. No GPU, tokenizer or speculative admission.
use augur_model_qwen35::{
    checkpoint_linear_specs, CheckpointArtifact, CheckpointExpectedIdentity,
    CheckpointLinearStorage, LoadedCheckpointLinear, Qwen35CheckpointProfile, Qwen35Config,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const CONFIG: &[u8] = include_bytes!("fixtures/nerkyor-w4a16-config.json");
const PROFILE: Qwen35CheckpointProfile = Qwen35CheckpointProfile::NerkyorW4a16;
fn fixture() -> Value {
    serde_json::from_slice(CONFIG).unwrap()
}
fn rejected(v: Value) {
    assert!(Qwen35Config::parse_with_profile(&serde_json::to_vec(&v).unwrap(), PROFILE).is_err());
}

#[test]
fn source_fixture_and_independent_inventory_are_exact() {
    assert_eq!(
        format!("{:x}", Sha256::digest(CONFIG)),
        "e71fd7b2cbade1ebb8be0382c81f7bf69eb1e039e30734914378d2c41217ef62"
    );
    let c = Qwen35Config::parse_with_profile(CONFIG, PROFILE).unwrap();
    assert_eq!(c.quantization.group0_targets.len(), 208);
    assert_eq!(c.quantization.group1_targets.len(), 192);
    assert_eq!(c.quantization.ignored_tensor_count, 260);
    assert_eq!(
        (c.hidden_size, c.num_hidden_layers, c.vocab_size),
        (5120, 64, 248320)
    );
    let specs = checkpoint_linear_specs(PROFILE);
    for s in &specs {
        let expected = if s.name == "lm_head"
            || s.name.ends_with(".in_proj_a")
            || s.name.ends_with(".in_proj_b")
        {
            CheckpointLinearStorage::Bf16
        } else {
            CheckpointLinearStorage::Nvfp4
        };
        assert_eq!(s.storage, expected, "{}", s.name);
    }
    assert_eq!(specs.len(), 497);
    // Positive also with noncanonical ordering: this is a set, not a JSON order contract.
    let mut reordered = fixture();
    reordered["quantization_config"]["ignore"]
        .as_array_mut()
        .unwrap()
        .reverse();
    Qwen35Config::parse_with_profile(&serde_json::to_vec(&reordered).unwrap(), PROFILE).unwrap();
}

#[test]
fn source_cannot_impersonate_legacy_profiles_or_cache_contract() {
    for p in [
        Qwen35CheckpointProfile::OriginalMixed,
        Qwen35CheckpointProfile::QuasarNvfp4,
        Qwen35CheckpointProfile::NvidiaMixed,
    ] {
        assert!(Qwen35Config::parse_with_profile(CONFIG, p).is_err());
    }
    let mut v = fixture();
    v["text_config"]["use_cache"] = json!(true);
    rejected(v);
}

#[test]
fn producer_method_algorithm_and_architecture_drift_reject() {
    for (path, value) in [
        ("/quantization_config/producer/version", json!("0.47.0")),
        ("/quantization_config/producer/name", json!("other")),
        ("/quantization_config/quant_method", json!("modelopt")),
        ("/quantization_config/quant_algo", json!("NVFP4")),
        ("/text_config/hidden_size", json!(4096)),
    ] {
        let mut v = fixture();
        *v.pointer_mut(path).unwrap() = value;
        rejected(v);
    }
}

#[test]
fn activation_config_and_extra_producer_fields_reject() {
    for key in ["config_groups", "input_activations", "activation_scale"] {
        let mut v = fixture();
        v["quantization_config"][key] = json!({});
        rejected(v);
    }
    let mut v = fixture();
    v["quantization_config"]["producer"]["activation_bits"] = json!(4);
    rejected(v);
}

#[test]
fn target_missing_extra_wrong_bits_group_and_extra_calibration_reject() {
    let key = "model.language_model.layers.0.mlp.gate_proj";
    let mut v = fixture();
    v["quantization_config"]["quantized_layers"]
        .as_object_mut()
        .unwrap()
        .remove(key);
    rejected(v);
    let mut v = fixture();
    v["quantization_config"]["quantized_layers"]["lm_head"] =
        json!({"quant_algo":"W4A16_NVFP4","group_size":16});
    rejected(v);
    for entry in [
        json!({"quant_algo":"NVFP4","group_size":16}),
        json!({"quant_algo":"W4A16_NVFP4","group_size":32}),
        json!({"quant_algo":"W4A16_NVFP4","group_size":16,"input_scale":1.0}),
    ] {
        let mut v = fixture();
        v["quantization_config"]["quantized_layers"][key] = entry;
        rejected(v);
    }
    let mut v = fixture();
    let text = serde_json::to_string(&v).unwrap();
    let needle = format!("\"{key}\":{{");
    // Duplicate JSON keys must fail even when both values are identical.
    let dup = text.replacen(
        &needle,
        &format!("\"{key}\":{{\"group_size\":16,\"quant_algo\":\"W4A16_NVFP4\"}},{needle}"),
        1,
    );
    assert_ne!(dup, text);
    assert!(Qwen35Config::parse_with_profile(dup.as_bytes(), PROFILE).is_err());
    v["quantization_config"]["quantized_layers"]
        ["model.language_model.layers.0.linear_attn.in_proj_a"] =
        json!({"quant_algo":"W4A16_NVFP4","group_size":16});
    rejected(v);
}

#[test]
fn excluded_gate_embedding_vision_mtp_and_duplicates_are_not_unchecked() {
    for removed in [
        "mtp",
        "lm_head",
        "model.language_model.embed_tokens",
        "model.language_model.layers.0.linear_attn.in_proj_a",
        "model.visual",
    ] {
        let mut v = fixture();
        v["quantization_config"]["ignore"]
            .as_array_mut()
            .unwrap()
            .retain(|n| n.as_str() != Some(removed));
        rejected(v);
    }
    let mut v = fixture();
    v["quantization_config"]["ignore"]
        .as_array_mut()
        .unwrap()
        .push(json!("mtp"));
    rejected(v);
    let mut v = fixture();
    v["quantization_config"]["ignore"]
        .as_array_mut()
        .unwrap()
        .push(json!("new.layer"));
    rejected(v);
}

#[test]
#[ignore = "requires explicit local identity + independent Python row references; hashes all source shards"]
fn authenticated_real_checkpoint_matches_independent_rows() {
    let root = std::env::var("AUGUR_NERKYOR_MODEL").expect("AUGUR_NERKYOR_MODEL");
    let identity_path = std::env::var("AUGUR_NERKYOR_IDENTITY").expect("AUGUR_NERKYOR_IDENTITY");
    let reference_path = std::env::var("AUGUR_NERKYOR_ROWS").expect("AUGUR_NERKYOR_ROWS");
    let identity: CheckpointExpectedIdentity =
        serde_json::from_slice(&std::fs::read(identity_path).unwrap()).unwrap();
    let refs: Value = serde_json::from_slice(&std::fs::read(reference_path).unwrap()).unwrap();
    assert_eq!(refs["config_sha256"], identity.config_sha256);
    assert_eq!(refs["index_sha256"], identity.index_sha256);
    let started = std::time::Instant::now();
    let mut reader =
        CheckpointArtifact::open_with_shard_workers(&root, PROFILE, identity, 2).unwrap();
    assert_eq!(reader.linears().count(), 497);
    let open_ms = started.elapsed().as_secs_f64() * 1000.0;
    let cases = refs["cases"].as_array().unwrap();
    let expected: std::collections::BTreeSet<_> =
        reader.linears().map(|s| s.name.clone()).collect();
    let actual: std::collections::BTreeSet<_> = cases
        .iter()
        .map(|c| c["name"].as_str().unwrap().to_owned())
        .collect();
    assert_eq!(actual, expected);
    assert_eq!(cases.len(), 497);
    let mut rows = 0;
    let mut values = 0;
    for case in cases {
        let name = case["name"].as_str().unwrap();
        for row in case["rows"].as_array().unwrap() {
            let index = row["row"].as_u64().unwrap() as usize;
            let loaded = reader.load_linear_rows(name, index, 1).unwrap();
            if let LoadedCheckpointLinear::Nvfp4 { global, .. } = &loaded {
                assert!(matches!(
                    global,
                    augur_model_qwen35::CheckpointNvfp4Scale::ModelOptW4A16 { .. }
                ));
            }
            let decoded = loaded.decode_f64().unwrap();
            let mut hash = Sha256::new();
            for x in &decoded {
                assert!(x.is_finite());
                hash.update(x.to_le_bytes());
            }
            assert_eq!(
                format!("{:x}", hash.finalize()),
                row["f64_le_sha256"].as_str().unwrap(),
                "{name} row {index}"
            );
            assert_eq!(decoded.len(), row["values"].as_u64().unwrap() as usize);
            rows += 1;
            values += decoded.len();
        }
        assert!(reader.load_linear_rows(name, 0, 0).is_err());
        assert!(reader.load_linear_rows(name, usize::MAX, 1).is_err());
    }
    assert!(reader.load_linear("mtp.layers.0.self_attn.q_proj").is_err());
    let receipt = json!({"status":"PASS_CPU_LOAD_AND_SAMPLED_WEIGHT_VALUES", "profile":"nerkyor_w4a16",
        "identity":reader.identity(), "authenticated_shards":reader.identity().shards.len(),
        "linears":497, "sampled_rows":rows, "compared_f64_values":values, "hash_and_open_ms":open_ms,
        "elapsed_ms":started.elapsed().as_secs_f64()*1000.0, "gpu_executed":false,
        "mtp_executed":false, "mtp_uploaded":false, "tokenizer_admitted":false, "dflash_executed":false,
        "full_weight_value_oracle":false});
    println!("{}", serde_json::to_string(&receipt).unwrap());
    if let Ok(path) = std::env::var("AUGUR_NERKYOR_RECEIPT") {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .unwrap();
        file.write_all(serde_json::to_string_pretty(&receipt).unwrap().as_bytes())
            .unwrap();
    }
}
