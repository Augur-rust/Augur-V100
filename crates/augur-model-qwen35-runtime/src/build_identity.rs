//! Build provenance is NOT semantic, performance or product authority.
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{error::Error, fs};

include!(concat!(env!("OUT_DIR"), "/qwen35_build_identity.rs"));

pub fn source_commit() -> &'static str {
    SOURCE_COMMIT.unwrap_or("QWEN35_NATIVE_STREAM_PRODUCER_UNQUALIFIED")
}

pub fn source_dirty_for(declared: &str) -> bool {
    SOURCE_COMMIT != Some(declared)
}

pub fn identity() -> Value {
    IDENTITY_JSON
        .map(|s| serde_json::from_str(s).expect("build-generated JSON"))
        .unwrap_or_else(|| {
            json!({"schema":"qwen35-compiled-build-identity-v1",
            "source_commit":source_commit(),"source_dirty":true,"product_ready":false,
            "qualified":false,"visible_tps":false,"semantic_qualification":false})
        })
}

/// Preserve legacy research report bytes unless the explicit build gate passed.
pub fn annotate(value: &mut Value) {
    if SOURCE_COMMIT.is_some() {
        value["source_commit"] = json!(source_commit());
        value["source_dirty"] = json!(false);
        value["build_identity"] = identity();
    }
}

/// No CUDA/model input is opened on this probe. Not a release certificate.
pub fn print_cli_identity(role: &str) -> Result<bool, Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 1 || args[0] != "--build-identity" {
        return Ok(false);
    }
    let bytes = fs::read(std::env::current_exe()?)?;
    println!(
        "{}",
        json!({"schema":"qwen35-executable-build-identity-v1", "role":role,
        "binary_sha256":format!("{:x}",Sha256::digest(bytes)),"build_identity":identity(),
        "gpu_started":false,"product_ready":false})
    );
    Ok(true)
}

pub fn validate_peer(actual: &Value, expected: &Value) -> Result<(), &'static str> {
    if actual["schema"] != "qwen35-executable-build-identity-v1"
        || expected["source_dirty"] != false
        || actual["build_identity"] != *expected
        || actual["role"] != "producer"
        || actual["gpu_started"] != false
        || actual["product_ready"] != false
    {
        return Err("frozen adapter/producer compiled identity mismatch");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unbound_commit_never_becomes_clean() {
        assert!(source_dirty_for(""));
        assert!(source_dirty_for("fake"));
        assert_eq!(source_dirty_for(source_commit()), SOURCE_COMMIT.is_none());
    }
    #[test]
    fn compiled_identity_never_opens_claims() {
        for key in [
            "product_ready",
            "qualified",
            "visible_tps",
            "semantic_qualification",
        ] {
            assert_eq!(identity()[key], false);
        }
    }
    #[test]
    fn peers_require_exact_identity_and_role() {
        let expected = json!({"source_dirty":false,"source_commit":"fixture"});
        let peer = json!({"schema":"qwen35-executable-build-identity-v1","role":"producer","build_identity":expected,"gpu_started":false,"product_ready":false});
        assert!(validate_peer(&peer, &expected).is_ok());
        for key in ["role", "gpu_started", "product_ready", "build_identity"] {
            let mut bad = peer.clone();
            bad[key] = json!(true);
            assert!(validate_peer(&bad, &expected).is_err());
        }
        let bad = json!({"source_dirty":true});
        assert!(validate_peer(&peer, &bad).is_err());
    }
    #[test]
    fn research_annotation_preserves_original() {
        if SOURCE_COMMIT.is_none() {
            let mut value =
                json!({"source_commit":"old","source_dirty":true,"product_ready":false});
            let old = value.clone();
            annotate(&mut value);
            assert_eq!(value, old);
        }
    }
}
