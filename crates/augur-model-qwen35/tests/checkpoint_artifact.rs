use augur_model_qwen35::{
    CheckpointArtifact, CheckpointExpectedIdentity, Qwen35CheckpointProfile, Qwen35Config,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::Path};

#[test]
#[ignore = "CPU actual pinned QUASAR/NVIDIA shards; AUGUR_CHECKPOINT_ARTIFACT_REFERENCES"]
fn actual_sharded_loader_and_negative_config_contracts() {
    let manifest_path =
        std::env::var_os("AUGUR_CHECKPOINT_ARTIFACT_REFERENCES").expect("reference manifest");
    let reference: Value = serde_json::from_slice(&std::fs::read(manifest_path).unwrap()).unwrap();
    let mut checked = 0;
    for (profile, label, audit) in [
        (
            Qwen35CheckpointProfile::QuasarNvfp4,
            "quasar",
            "F:/qwen35-quasar-input-audit-20261008-v2/checkpoint.json",
        ),
        (
            Qwen35CheckpointProfile::NvidiaMixed,
            "nvidia",
            "F:/qwen35-nvidia-input-audit-20261008-v1/checkpoint.json",
        ),
    ] {
        let audit: Value = serde_json::from_slice(&std::fs::read(audit).unwrap()).unwrap();
        assert_eq!(audit["status"], "PASS_INPUT_INTEGRITY");
        let root = Path::new(audit["model_dir"].as_str().unwrap());
        let config = std::fs::read(root.join("config.json")).unwrap();
        assert!(
            Qwen35Config::parse(&config).is_err(),
            "new metadata must not impersonate old mixed config"
        );
        let model = Qwen35Config::parse_with_profile(&config, profile).unwrap();
        assert_eq!(model.quantization.source_profile, profile);
        assert_eq!(model.hidden_size, 5120);
        let wrong = if profile == Qwen35CheckpointProfile::QuasarNvfp4 {
            Qwen35CheckpointProfile::NvidiaMixed
        } else {
            Qwen35CheckpointProfile::QuasarNvfp4
        };
        assert!(Qwen35Config::parse_with_profile(&config, wrong).is_err());
        let value: Value = serde_json::from_slice(&config).unwrap();
        for pointer in [
            "/text_config/hidden_size",
            "/quantization_config/config_groups/group_0/weights/num_bits",
        ] {
            let mut bad = value.clone();
            *bad.pointer_mut(pointer).unwrap() = serde_json::json!(3);
            assert!(
                Qwen35Config::parse_with_profile(&serde_json::to_vec(&bad).unwrap(), profile)
                    .is_err()
            );
        }
        let mut duplicate = value.clone();
        let targets = duplicate["quantization_config"]["config_groups"]["group_0"]["targets"]
            .as_array_mut()
            .unwrap();
        targets.push(targets[0].clone());
        assert!(Qwen35Config::parse_with_profile(
            &serde_json::to_vec(&duplicate).unwrap(),
            profile
        )
        .is_err());
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
                .collect::<BTreeMap<_, _>>(),
        };
        let started = std::time::Instant::now();
        let mut reader = CheckpointArtifact::open(root, profile, expected).unwrap();
        assert_eq!(reader.linears().count(), 497);
        println!(
            "CHECKPOINT_LOADER_READY profile={label} shards={} hash_and_open_ms={}",
            reader.identity().shards.len(),
            started.elapsed().as_secs_f64() * 1000.
        );
        for case in reference["cases"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|v| v["profile"] == label)
        {
            let name = case["name"].as_str().unwrap();
            let path = case["reference"]["path"].as_str().unwrap();
            let bytes = std::fs::read(path).unwrap();
            assert_eq!(
                format!("{:x}", Sha256::digest(&bytes)),
                case["reference"]["sha256"].as_str().unwrap()
            );
            let mut result = Vec::new();
            for row in case["row_indices"].as_array().unwrap() {
                result.extend(
                    reader
                        .load_linear_rows(name, row.as_u64().unwrap() as usize, 1)
                        .unwrap()
                        .decode_f64()
                        .unwrap(),
                );
            }
            assert_eq!(result.len() * 8, bytes.len());
            for (a, b) in result.iter().zip(bytes.chunks_exact(8)) {
                assert_eq!(
                    a.to_bits(),
                    f64::from_le_bytes(b.try_into().unwrap()).to_bits(),
                    "{name}"
                );
            }
            assert!(reader.load_linear_rows(name, 0, 0).is_err());
            assert!(reader.load_linear_rows(name, usize::MAX, 1).is_err());
            checked += result.len();
        }
        assert!(reader.load_linear("unknown.projection").is_err());
        // All sampled calls reuse the same authenticated descriptor owners.
        println!("CHECKPOINT_LOADER_PASS profile={label} linear_inventory=497");
    }
    assert_eq!(checked, 1_531_904);
}
