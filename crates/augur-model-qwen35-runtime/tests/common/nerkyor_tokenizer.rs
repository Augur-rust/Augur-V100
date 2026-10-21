//! Shared local test harness, not a public session admission path.
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{fs, path::Path};
use tokenizers::Tokenizer;

pub fn reference() -> Value {
    serde_json::from_str(include_str!("../fixtures/nerkyor-tokenizer-reference.json")).unwrap()
}

pub fn load(root: impl AsRef<Path>) -> Tokenizer {
    let reference = reference();
    let mut tokenizer = None;
    for (name, expected) in reference["files"].as_object().unwrap() {
        let bytes = fs::read(root.as_ref().join(name)).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(&bytes)),
            expected.as_str().unwrap(),
            "{name} identity changed"
        );
        if name == "tokenizer.json" {
            // Construct from the exact hashed bytes; do not reopen a mutable path.
            tokenizer = Some(Tokenizer::from_bytes(&bytes).unwrap());
        }
    }
    let tokenizer = tokenizer.expect("own tokenizer.json must be authenticated");
    assert!(tokenizer.get_vocab_size(true) <= 248320);
    tokenizer
}

pub fn validate_samples(tokenizer: &Tokenizer) {
    let reference = reference();
    for case in reference["cases"].as_array().unwrap() {
        let ids = tokenizer
            .encode(case["text"].as_str().unwrap(), false)
            .unwrap();
        let expected: Vec<u32> = case["hf_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(ids.get_ids(), expected, "{}", case["name"]);
        assert!(ids.get_ids().iter().all(|&id| id < 248320));
        assert_eq!(
            tokenizer.decode(ids.get_ids(), false).unwrap(),
            case["hf_decoded"].as_str().unwrap()
        );
    }
    for case in reference["chat_cases"].as_array().unwrap() {
        // Rendered by pinned offline HF in the fixture; this does NOT test a Rust Jinja renderer.
        let ids = tokenizer
            .encode(case["rendered"].as_str().unwrap(), false)
            .unwrap();
        let expected: Vec<u32> = case["hf_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(ids.get_ids(), expected);
    }
}
