//! Contract component tests; these do not build or qualify the CUDA producer.
#[path = "../src/public_source_scope.rs"]
mod scope;

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

fn fixture() -> (Value, Value, Map<String, Value>) {
    let declaration: Value =
        serde_json::from_str(include_str!("../../../qwen35-public-source-scope.json")).unwrap();
    let sources = declaration["files"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| (p.as_str().unwrap().to_owned(), json!("a".repeat(64))))
        .collect();
    let attestation = json!({"schema":scope::ATTESTATION_SCHEMA,"source_commit":"1".repeat(40),
        "source_dirty":false,"profile":"release","opt_level":"3","cuda_arch":"arch=compute_70,code=sm_70",
        "target":"x86_64-pc-windows-msvc","resolved_features":scope::RUNTIME_FEATURES,
        "requested_features":scope::RUNTIME_FEATURES});
    (declaration, attestation, sources)
}

fn remove(declaration: &mut Value, sources: &mut Map<String, Value>, name: &str) {
    sources.remove(name);
    declaration["files"]
        .as_array_mut()
        .unwrap()
        .retain(|value| value != name);
}

#[test]
fn complete_generated_scope_is_valid_metadata_only() {
    let (declaration, attestation, sources) = fixture();
    scope::validate_value(&declaration, &attestation, &sources).unwrap();
    assert!(sources.len() > scope::PACKAGES.len());
}

#[test]
fn private_v1_or_borrowed_upstream_identity_is_not_public_identity() {
    let (declaration, mut attestation, sources) = fixture();
    attestation["schema"] = json!("qwen35-frozen-source-build-input-v1");
    assert!(scope::validate_value(&declaration, &attestation, &sources).is_err());
    attestation["schema"] = json!(scope::ATTESTATION_SCHEMA);
    attestation["source_commit"] = json!(scope::UPSTREAM);
    assert!(scope::validate_value(&declaration, &attestation, &sources).is_err());
}

#[test]
fn dropping_declared_source_or_inserting_undeclared_source_fails() {
    let (declaration, attestation, mut sources) = fixture();
    sources.remove("NOTICE");
    assert!(scope::validate_value(&declaration, &attestation, &sources).is_err());
    sources.insert("NOTICE".to_owned(), json!("a".repeat(64)));
    sources.insert("injected.rs".to_owned(), json!("a".repeat(64)));
    assert!(scope::validate_value(&declaration, &attestation, &sources).is_err());
}

#[test]
fn deleting_required_implementation_from_both_maps_still_fails() {
    for name in [
        "crates/augur-core/src/lib.rs",
        "crates/augur-model-qwen35-runtime/examples/qwen35_resident_session_http.rs",
        "crates/augur-cuda-sys/native/augur_qwen35_linear_step_sm70.cu",
        "crates/augur-cuda-sys/native/augur_qwen35_prefill_llama_mmq_sm70.cu",
        "crates/augur-cuda-sys/native/augur_qwen35_prefill_llama_small_sm70.cuh",
        "crates/augur-cuda-sys/native/llama_attention_vendor/mmq.cuh",
        scope::SCOPE_FILE,
        "scripts/build-public-qwen.py",
        "BUILDING.md",
    ] {
        let (mut declaration, attestation, mut sources) = fixture();
        remove(&mut declaration, &mut sources, name);
        assert!(
            scope::validate_value(&declaration, &attestation, &sources).is_err(),
            "{name}"
        );
    }
}

#[test]
fn private_package_weight_and_case_alias_cannot_enter_declared_scope() {
    for name in [
        "crates/augur-model-deepseek-v4-flash/src/lib.rs",
        "model.gguf",
        "../outside.rs",
        "crates/augur-core/src/LIB.rs",
        "crates/augur-core/src/alias:stream.rs",
        "scripts/private_driver.py",
        "scripts/qwen35_unknown.py",
        "scripts/qwen35_api.py.backup",
        "docs/ACTIVE_EXECUTION.md",
    ] {
        let (mut declaration, attestation, mut sources) = fixture();
        sources.insert(name.to_owned(), json!("a".repeat(64)));
        let mut files: Vec<_> = declaration["files"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_owned())
            .collect();
        files.push(name.to_owned());
        files.sort();
        declaration["files"] = json!(files);
        assert!(
            scope::validate_value(&declaration, &attestation, &sources).is_err(),
            "{name}"
        );
    }
}

#[test]
fn reviewed_api_and_persistence_files_are_valid_scope_metadata() {
    let (mut declaration, attestation, mut sources) = fixture();
    for name in scope::CLIENT_FILES {
        sources.insert((*name).to_owned(), json!("a".repeat(64)));
    }
    declaration["files"] = json!(sources.keys().collect::<Vec<_>>());
    scope::validate_value(&declaration, &attestation, &sources).unwrap();
}

#[test]
fn schema_keys_claim_types_and_build_profile_are_closed() {
    for field in [
        "packages",
        "native_files",
        "product_entrypoints",
        "runtime_features",
    ] {
        let (mut declaration, attestation, sources) = fixture();
        declaration[field] = json!([]);
        assert!(scope::validate_value(&declaration, &attestation, &sources).is_err());
    }
    let (mut declaration, mut attestation, sources) = fixture();
    declaration["claims"]["product_ready"] = json!("false");
    assert!(scope::validate_value(&declaration, &attestation, &sources).is_err());
    declaration["claims"]["product_ready"] = json!(false);
    declaration["unrecognized_authority"] = json!(true);
    assert!(scope::validate_value(&declaration, &attestation, &sources).is_err());
    declaration
        .as_object_mut()
        .unwrap()
        .remove("unrecognized_authority");
    attestation["opt_level"] = json!("0");
    assert!(scope::validate_value(&declaration, &attestation, &sources).is_err());
}

static COUNTER: AtomicU64 = AtomicU64::new(0);
struct TempScope(PathBuf);
impl TempScope {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "augur-public-scope-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::SeqCst)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for TempScope {
    fn drop(&mut self) {
        let _ = fs::remove_file(self.0.join(scope::SCOPE_FILE));
        let _ = fs::remove_dir(&self.0);
    }
}

#[test]
fn actual_scope_file_requires_canonical_bytes_and_external_hash() {
    let (declaration, mut attestation, mut sources) = fixture();
    let temporary = TempScope::new();
    let raw = serde_json::to_vec(&declaration).unwrap();
    let digest = format!("{:x}", Sha256::digest(&raw));
    attestation["public_source_scope_sha256"] = json!(digest);
    sources.insert(scope::SCOPE_FILE.to_owned(), json!(digest));
    fs::write(temporary.0.join(scope::SCOPE_FILE), &raw).unwrap();
    scope::validate(&temporary.0, &attestation, &sources).unwrap();
    attestation["public_source_scope_sha256"] = json!("0".repeat(64));
    assert!(scope::validate(&temporary.0, &attestation, &sources).is_err());
    let mut noncanonical = raw;
    noncanonical.push(b'\n');
    let changed = format!("{:x}", Sha256::digest(&noncanonical));
    attestation["public_source_scope_sha256"] = json!(changed);
    sources.insert(scope::SCOPE_FILE.to_owned(), json!(changed));
    fs::write(temporary.0.join(scope::SCOPE_FILE), noncanonical).unwrap();
    assert!(scope::validate(&temporary.0, &attestation, &sources).is_err());
}
