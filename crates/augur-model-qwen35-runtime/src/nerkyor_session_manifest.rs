//! Model-specific, caller-pinned metadata for the Nerkyor session adapter.
//! This CPU admission is not product, sampling or long-context qualification.
//! Target shard bytes and draft GGUF bytes are verified again by their loaders.
use crate::augur_session::{digest, valid_digest, Result, VOCAB};
use augur_model_qwen35::{validate_strict_json, CheckpointExpectedIdentity, DFlashConfig};
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

pub const SCHEMA: &str = "nerkyor-session-model-manifest-v1";
pub const EXECUTION: &str = "nerkyor_original_nvfp4_core_f32_kv_b8_prefill_pending_v2";
pub const NO_DFLASH_EXECUTION: &str = "nerkyor_qpn_source_reassoc_f32_kv4096_no_dflash_v1";
pub const DRAFT_FILE: &str = "qwen35-27b-dflash2-q8_0.gguf";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NerkyorManifest {
    pub schema: String,
    pub execution_profile: String,
    pub checkpoint: CheckpointExpectedIdentity,
    pub tokenizer_sha256: String,
    pub tokenizer_config_sha256: String,
    pub generation_config_sha256: String,
    pub chat_template_sha256: String,
    pub draft_config_sha256: String,
    pub draft_gguf_sha256: String,
}
/// Private retained inputs: downstream consumers must not hash then reopen.
pub struct AdmittedNerkyorManifest {
    pub(crate) root: PathBuf,
    pub(crate) draft_root: PathBuf,
    pub(crate) manifest: NerkyorManifest,
    pub(crate) sha256: String,
    pub(crate) eos: Vec<u32>,
    pub(crate) draft_config: DFlashConfig,
    tokenizer: Vec<u8>,
    template: Vec<u8>,
    raw: Vec<u8>,
}
impl AdmittedNerkyorManifest {
    pub fn model_root(&self) -> &Path {
        &self.root
    }
    pub fn draft_root(&self) -> &Path {
        &self.draft_root
    }
    pub fn manifest(&self) -> &NerkyorManifest {
        &self.manifest
    }
    pub fn sha256(&self) -> &str {
        &self.sha256
    }
    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos
    }
    pub fn tokenizer_bytes(&self) -> &[u8] {
        &self.tokenizer
    }
    pub fn chat_template_bytes(&self) -> &[u8] {
        &self.template
    }
    pub fn canonical_bytes(&self) -> &[u8] {
        &self.raw
    }
}
impl NerkyorManifest {
    pub fn validate(&self) -> Result<()> {
        if self.schema != SCHEMA
            || !matches!(self.execution_profile.as_str(), EXECUTION | NO_DFLASH_EXECUTION)
            || [
                &self.checkpoint.config_sha256,
                &self.checkpoint.index_sha256,
                &self.tokenizer_sha256,
                &self.tokenizer_config_sha256,
                &self.generation_config_sha256,
                &self.chat_template_sha256,
                &self.draft_config_sha256,
                &self.draft_gguf_sha256,
            ]
            .iter()
            .any(|s| !valid_digest(s))
            || self.checkpoint.shards.len() != 7
        {
            return Err("invalid Nerkyor manifest or numerical identity".into());
        }
        // MTP and vision are pinned source artifacts, NOT execution authority.
        let names = [
            "text-01.safetensors",
            "text-02.safetensors",
            "text-03.safetensors",
            "text-04.safetensors",
            "text-05.safetensors",
            "mtp-bf16.safetensors",
            "vision-bf16.safetensors",
        ];
        if names.iter().any(|n| {
            !self
                .checkpoint
                .shards
                .get(*n)
                .is_some_and(|h| valid_digest(h))
        }) {
            return Err("Nerkyor manifest requires exactly the seven source shard names".into());
        }
        Ok(())
    }
}
fn bounded(path: &Path, max: u64) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    File::open(path)?.take(max + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > max {
        return Err("Nerkyor metadata extent exceeded".into());
    }
    Ok(bytes)
}
fn pinned(path: &Path, expected: &str, max: u64) -> Result<Vec<u8>> {
    let bytes = bounded(path, max)?;
    if digest(&bytes) != expected {
        return Err(format!("Nerkyor metadata digest mismatch: {}", path.display()).into());
    }
    Ok(bytes)
}
fn metadata_eos(config: &[u8], template: &[u8], generation: &[u8]) -> Result<Vec<u32>> {
    validate_strict_json(config)?;
    validate_strict_json(generation)?;
    let t: serde_json::Value = serde_json::from_slice(config)?;
    let text = std::str::from_utf8(template)?;
    // This export has a separate Jinja file, not QUASAR's embedded template.
    // An embedded declaration, if present, must still agree byte-for-byte.
    if text.is_empty()
        || t.get("chat_template")
            .is_some_and(|v| v.as_str() != Some(text))
    {
        return Err("Nerkyor separate/embedded chat template mismatch".into());
    }
    let g: serde_json::Value = serde_json::from_slice(generation)?;
    let eos: Vec<u32> = if let Some(v) = g["eos_token_id"].as_u64() {
        vec![u32::try_from(v)?]
    } else {
        serde_json::from_value(g["eos_token_id"].clone())?
    };
    if eos.is_empty()
        || eos.iter().any(|v| *v >= VOCAB)
        || eos.iter().collect::<std::collections::HashSet<_>>().len() != eos.len()
    {
        return Err("invalid Nerkyor generation EOS".into());
    }
    // Pin the target's two-stop export, not the single EOS in the draft config.
    if eos != [248046, 248044] {
        return Err("unexpected Nerkyor generation EOS contract".into());
    }
    Ok(eos)
}
/// `expected_sha256` is supplied by the caller, not trusted from the manifest.
/// The session adapter is greedy-only despite source do_sample=true.
pub fn load(
    root: &Path,
    draft_root: &Path,
    path: &Path,
    expected_sha256: &str,
) -> Result<AdmittedNerkyorManifest> {
    if !valid_digest(expected_sha256) {
        return Err("caller-pinned Nerkyor manifest digest required".into());
    }
    let raw = pinned(path, expected_sha256, 1024 * 1024)?;
    validate_strict_json(&raw)?;
    let manifest: NerkyorManifest = serde_json::from_slice(&raw)?;
    manifest.validate()?;
    let tokenizer = pinned(
        &root.join("tokenizer.json"),
        &manifest.tokenizer_sha256,
        64 * 1024 * 1024,
    )?;
    let config = pinned(
        &root.join("tokenizer_config.json"),
        &manifest.tokenizer_config_sha256,
        4 * 1024 * 1024,
    )?;
    let template = pinned(
        &root.join("chat_template.jinja"),
        &manifest.chat_template_sha256,
        4 * 1024 * 1024,
    )?;
    let generation = pinned(
        &root.join("generation_config.json"),
        &manifest.generation_config_sha256,
        1024 * 1024,
    )?;
    let eos = metadata_eos(&config, &template, &generation)?;
    let own_draft = pinned(
        &root.join("dflash/config.json"),
        &manifest.draft_config_sha256,
        1024 * 1024,
    )?;
    let converted = pinned(
        &draft_root.join("config.json"),
        &manifest.draft_config_sha256,
        1024 * 1024,
    )?;
    if own_draft != converted {
        return Err("converted draft config differs from own model".into());
    }
    let draft_config = DFlashConfig::parse(&converted)?;
    draft_config.validate_exact_profile()?;
    Ok(AdmittedNerkyorManifest {
        root: root.to_owned(),
        draft_root: draft_root.to_owned(),
        manifest,
        sha256: expected_sha256.into(),
        eos,
        draft_config,
        tokenizer,
        template,
        raw,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn manifest() -> NerkyorManifest {
        let checkpoint:CheckpointExpectedIdentity=serde_json::from_value(serde_json::json!({
            "config_sha256":"a".repeat(64),"index_sha256":"b".repeat(64),"shards":{
                "text-01.safetensors":"c".repeat(64),"text-02.safetensors":"c".repeat(64),
                "text-03.safetensors":"c".repeat(64),"text-04.safetensors":"c".repeat(64),
                "text-05.safetensors":"c".repeat(64),"mtp-bf16.safetensors":"c".repeat(64),"vision-bf16.safetensors":"c".repeat(64)}})).unwrap();
        NerkyorManifest {
            schema: SCHEMA.into(),
            execution_profile: EXECUTION.into(),
            checkpoint,
            tokenizer_sha256: "d".repeat(64),
            tokenizer_config_sha256: "d".repeat(64),
            generation_config_sha256: "d".repeat(64),
            chat_template_sha256: "d".repeat(64),
            draft_config_sha256: "e".repeat(64),
            draft_gguf_sha256: "f".repeat(64),
        }
    }
    #[test]
    fn own_profile_and_exact_source_names_are_closed() {
        let good = manifest();
        good.validate().unwrap();
        for schema in ["quasar-session-model-manifest-v1", ""] {
            let mut bad = good.clone();
            bad.schema = schema.into();
            assert!(bad.validate().is_err());
        }
        let mut bad = good.clone();
        bad.execution_profile = "quasar_b4352_dflash_b8_packed_http_v1".into();
        assert!(bad.validate().is_err());
        let mut bad = good.clone();
        bad.checkpoint.shards.remove("mtp-bf16.safetensors");
        assert!(bad.validate().is_err());
        let mut bad = good.clone();
        bad.checkpoint
            .shards
            .insert("../text-01.safetensors".into(), "0".repeat(64));
        assert!(bad.validate().is_err());
        let mut bad = good.clone();
        bad.draft_gguf_sha256 = "A".repeat(64);
        assert!(bad.validate().is_err());
        let mut value = serde_json::to_value(good).unwrap();
        value["mtp_enabled"] = serde_json::json!(true);
        assert!(serde_json::from_value::<NerkyorManifest>(value).is_err());
    }
    #[test]
    fn separate_template_and_generation_eos_not_draft_eos() {
        let generation = br#"{"do_sample":true,"eos_token_id":[248046,248044]}"#;
        assert_eq!(
            metadata_eos(b"{}", b"own jinja", generation).unwrap(),
            [248046, 248044]
        );
        assert!(metadata_eos(br#"{"chat_template":"foreign"}"#, b"own jinja", generation).is_err());
        assert!(metadata_eos(br#"{"chat_template":null}"#, b"own jinja", generation).is_err());
        assert!(metadata_eos(b"{}", b"", generation).is_err());
        for bad in [
            r#"{"eos_token_id":248044}"#,
            r#"{"eos_token_id":[248046,248046]}"#,
            r#"{"eos_token_id":[248320]}"#,
            r#"{"eos_token_id":[248046,248044],"eos_token_id":[248046,248044]}"#,
        ] {
            assert!(metadata_eos(b"{}", b"own jinja", bad.as_bytes()).is_err());
        }
    }
}
