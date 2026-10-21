//! Explicit CPU artifact checks, separate from GPU ownership and old benchmark queues.
#[path = "common/nerkyor_tokenizer.rs"]
mod tokenizer;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::PathBuf, time::Instant};

fn model() -> PathBuf {
    PathBuf::from(std::env::var_os("AUGUR_NERKYOR_MODEL").expect("explicit model"))
}
fn write_receipt(value: &Value) {
    let path = std::env::var_os("AUGUR_NERKYOR_ARTIFACT_RECEIPT").expect("fresh receipt path");
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap();
    file.write_all(&serde_json::to_vec_pretty(value).unwrap())
        .unwrap();
}

#[test]
fn tokenizer_reference_has_new_identity_and_explicit_hf_backend() {
    let reference = tokenizer::reference();
    assert_eq!(reference["hf_class"], "Qwen2Tokenizer");
    assert_eq!(reference["transformers_version"], "5.12.0");
    assert_eq!(reference["tokenizers_version"], "0.22.2");
    assert_eq!(reference["hf_and_direct_pre_tokenizers_identical"], true);
    assert_eq!(reference["config_regex_not_rewritten"], true);
    assert_eq!(
        reference["files"]["tokenizer.json"],
        "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
    );
    assert_eq!(reference["cases"].as_array().unwrap().len(), 20);
    assert_eq!(reference["chat_cases"].as_array().unwrap().len(), 2);
    assert!(reference["cases"]
        .as_array()
        .unwrap()
        .iter()
        .all(|case| case["equal"] == true));
}

#[test]
#[ignore = "requires exact local new-model tokenizer files; CPU only"]
fn real_nerkyor_tokenizer_matches_offline_hf_reference() {
    let tokenizer = tokenizer::load(model());
    tokenizer::validate_samples(&tokenizer);
    write_receipt(
        &json!({"status":"PASS_RUST_TOKENIZER_VS_PINNED_OFFLINE_HF_SAMPLES",
        "files":tokenizer::reference()["files"],"text_cases":20,"chat_rendered_cases":2,
        "constructs_from_authenticated_bytes":true,"tokenizer_modified":false,
        "rust_jinja_renderer_tested":false,"all_unicode_qualified":false,
        "gpu_executed":false,"mtp_executed":false,"old_tokenizer_reused":false}),
    );
}

#[test]
#[ignore = "requires new converted Q8 artifact and independent NumPy selector references; CPU only"]
fn real_nerkyor_dflash_q8_loads_without_old_selector_calibration() {
    use augur_model_qwen35::DFlashConfig;
    use augur_model_qwen35_runtime::DFlashQ8Backbone;
    let started = Instant::now();
    let root =
        PathBuf::from(std::env::var_os("AUGUR_NERKYOR_DFLASH_Q8").expect("new Q8 conversion"));
    let reference: Value = serde_json::from_slice(
        &fs::read(
            std::env::var_os("AUGUR_NERKYOR_DFLASH_REFERENCE").expect("independent reference"),
        )
        .unwrap(),
    )
    .unwrap();
    let config_bytes = fs::read(root.join("config.json")).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&config_bytes)),
        reference["config_sha256"].as_str().unwrap()
    );
    let config = DFlashConfig::parse(&config_bytes).unwrap();
    config.validate_exact_profile().unwrap();
    let backbone =
        DFlashQ8Backbone::load_pinned(root.join("qwen35-27b-dflash2-q8_0.gguf"), config).unwrap();
    let manifest = backbone.manifest();
    assert_eq!(
        manifest.file_sha256.as_deref(),
        reference["gguf_sha256"].as_str()
    );
    assert_eq!(manifest.tensors.len(), 81);
    assert_eq!(
        manifest.tensors.iter().filter(|t| t.ggml_type == 8).count(),
        49
    );
    assert_eq!(
        manifest.tensors.iter().filter(|t| t.ggml_type == 0).count(),
        32
    );
    assert!(
        backbone.embedded_selector_gate().is_none(),
        "must not admit old gate0.45 profile"
    );
    assert!(backbone.selector_codebook_row(true, 248320).is_err());
    for case in reference["selector_rows"].as_array().unwrap() {
        let predecessor = case["name"] == "selector_predecessor.weight";
        assert!(predecessor || case["name"] == "selector_successor.weight");
        let values = backbone
            .selector_codebook_row(predecessor, case["row"].as_u64().unwrap() as u32)
            .unwrap();
        assert_eq!(values.len(), 256);
        assert!(values.iter().all(|v| v.is_finite()));
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(
            format!("{:x}", Sha256::digest(bytes)),
            case["f32_le_sha256"].as_str().unwrap()
        );
    }
    write_receipt(
        &json!({"status":"PASS_NEW_DFLASH_Q8_PINNED_CPU_LOAD_AND_SELECTOR_ROWS",
        "gguf_sha256":reference["gguf_sha256"],"config_sha256":reference["config_sha256"],
        "tensors":81,"q8_tensors":49,"f32_tensors":32,"selector_rows":6,
        "payload_bytes":backbone.payload_bytes(),"elapsed_s":started.elapsed().as_secs_f64(),
        "old_selector_calibration_reused":false,"new_selector_calibrated":false,
        "source_fp8_vs_q8_equivalence_qualified":false,"gpu_executed":false,
        "draft_block_executed":false,"mtp_executed":false,"acceptance_measured":false}),
    );
}
