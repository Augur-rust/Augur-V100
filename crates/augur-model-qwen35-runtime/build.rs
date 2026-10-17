//! Source archive attestation for an opt-in, still-unqualified release build.
#[path = "src/public_source_scope.rs"]
mod public_source_scope;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    env, fs,
    path::{Component, Path},
};

fn regular(path: &Path, directory: bool) {
    let m = fs::symlink_metadata(path).expect("source metadata");
    assert!(!m.file_type().is_symlink(), "symlink in build inputs");
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        assert_eq!(m.file_attributes() & 0x400, 0, "reparse build input");
    }
    assert!(
        if directory { m.is_dir() } else { m.is_file() },
        "regular build input"
    );
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn hex(value: &Value, len: usize) -> &str {
    let s = value.as_str().expect("string digest");
    assert!(
        s.len() == len
            && s.bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)),
        "lowercase hex digest"
    );
    s
}
fn namespace(root: &Path, at: &Path, paths: &mut BTreeSet<String>) {
    regular(at, true);
    for entry in fs::read_dir(at).expect("source directory") {
        let entry = entry.expect("source entry");
        let path = entry.path();
        if entry.file_type().expect("source type").is_dir() {
            namespace(root, &path, paths);
        } else {
            regular(&path, false);
            paths.insert(
                path.strip_prefix(root)
                    .unwrap()
                    .to_str()
                    .expect("UTF8 source path")
                    .replace('\\', "/"),
            );
        }
    }
}
fn main() {
    println!("cargo:rerun-if-env-changed=AUGUR_QWEN35_BUILD_ATTESTATION");
    let out = Path::new(&env::var_os("OUT_DIR").unwrap()).join("qwen35_build_identity.rs");
    if env::var_os("CARGO_FEATURE_QWEN35_FROZEN_BUILD").is_none() {
        fs::write(out, "pub const SOURCE_COMMIT: Option<&str> = None;\npub const IDENTITY_JSON: Option<&str> = None;\n").unwrap();
        return;
    }
    let path = std::path::PathBuf::from(
        env::var_os("AUGUR_QWEN35_BUILD_ATTESTATION")
            .expect("frozen build requires external source attestation"),
    );
    assert!(path.is_absolute(), "absolute build attestation required");
    regular(&path, false);
    assert!(
        fs::metadata(&path).unwrap().len() < 2 * 1024 * 1024,
        "bounded attestation"
    );
    let raw = fs::read(&path).expect("read build attestation");
    let v: Value = serde_json::from_slice(&raw).expect("JSON attestation");
    assert_eq!(
        serde_json::to_vec(&v).unwrap(),
        raw,
        "canonical attestation bytes required"
    );
    assert_eq!(v["schema"], public_source_scope::ATTESTATION_SCHEMA);
    assert_eq!(v["source_dirty"], false);
    assert_eq!(v["profile"], "release");
    assert_eq!(v["opt_level"], "3");
    assert_eq!(v["cuda_arch"], "arch=compute_70,code=sm_70");
    assert_eq!(env::var("PROFILE").unwrap(), "release");
    assert_eq!(env::var("OPT_LEVEL").unwrap(), "3");
    assert_eq!(env::var("DEBUG").unwrap(), "false");
    assert_eq!(
        env::var("AUGUR_CUDA_GENCODE").unwrap(),
        "arch=compute_70,code=sm_70"
    );
    let compiler = std::process::Command::new(env::var_os("RUSTC").unwrap())
        .arg("-Vv")
        .output()
        .expect("compiler identity");
    assert!(compiler.status.success());
    assert_eq!(
        String::from_utf8(compiler.stdout).unwrap().trim(),
        v["rustc"].as_str().unwrap().trim()
    );
    assert_eq!(
        v["target"].as_str(),
        Some(env::var("TARGET").unwrap().as_str())
    );
    let commit = hex(&v["source_commit"], 40);
    hex(&v["source_tree"], 40);
    let expected_map_sha = hex(&v["source_map_sha256"], 64);
    let sources = v["sources"].as_object().expect("source map");
    assert_eq!(
        digest(&serde_json::to_vec(sources).unwrap()),
        expected_map_sha,
        "source map identity"
    );
    let manifest = std::path::PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let root = manifest.parent().unwrap().parent().unwrap();
    public_source_scope::validate(root, &v, sources).expect("complete public source scope");
    let mut expected = BTreeSet::new();
    for (name, hash) in sources {
        let rel = Path::new(name);
        assert!(
            !name.is_empty()
                && !name.contains(['\\', ':'])
                && !rel.is_absolute()
                && rel.components().all(|c| matches!(c, Component::Normal(_))),
            "safe source name"
        );
        let file = root.join(rel);
        regular(&file, false);
        assert_eq!(
            digest(&fs::read(&file).expect("source bytes")),
            hex(hash, 64),
            "source drift: {name}"
        );
        println!("cargo:rerun-if-changed={}", file.display());
        expected.insert(name.clone());
    }
    let mut actual = BTreeSet::new();
    namespace(root, root, &mut actual);
    assert_eq!(actual, expected, "archive source namespace drift");
    println!("cargo:rerun-if-changed={}", root.display());
    println!("cargo:rerun-if-changed={}", path.display());
    let mut features: Vec<String> = env::vars()
        .filter_map(|(k, _)| {
            k.strip_prefix("CARGO_FEATURE_")
                .map(|s| s.to_lowercase().replace('_', "-"))
        })
        .collect();
    features.sort();
    features.dedup();
    assert_eq!(
        json!(features),
        v["resolved_features"],
        "resolved feature drift"
    );
    let identity = json!({"schema":"qwen35-compiled-build-identity-v1",
        "source_commit":commit,"source_tree":v["source_tree"],"source_dirty":false,
        "source_map_sha256":expected_map_sha,"attestation_sha256":digest(&raw),
        "profile":"release","opt_level":"3","target":v["target"],"cuda_arch":v["cuda_arch"],
        "features":features,"rustc":v["rustc"],"build_scope":"verified_git_archive_source_only",
        "product_ready":false,"qualified":false,"visible_tps":false,"semantic_qualification":false});
    fs::write(out,format!("pub const SOURCE_COMMIT: Option<&str> = Some({commit:?});\npub const IDENTITY_JSON: Option<&str> = Some({:?});\n",serde_json::to_string(&identity).unwrap())).unwrap();
}
