//! Caller-pinned model/metadata identity. Loading this is CPU-only; the CUDA
//! artifact loader subsequently verifies all five full shard digests itself.
use super::{digest, valid_digest, Result, VOCAB};
use augur_model_qwen35::CheckpointExpectedIdentity;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AugurManifest {
    pub schema: String,
    pub upstream_repo: String,
    pub upstream_revision: String,
    pub checkpoint: CheckpointExpectedIdentity,
    pub tokenizer_sha256: String,
    pub tokenizer_config_sha256: String,
    pub generation_config_sha256: String,
    pub chat_template_sha256: String,
}
pub struct AdmittedManifest {
    pub(crate) root: PathBuf,
    pub(crate) manifest: AugurManifest,
    pub(crate) sha256: String,
    pub(crate) eos: Vec<u32>,
    raw: Vec<u8>,
}
impl AdmittedManifest {
    pub fn sha256(&self) -> &str {
        &self.sha256
    }
    pub fn manifest(&self) -> &AugurManifest {
        &self.manifest
    }
    pub fn model_root(&self) -> &Path {
        &self.root
    }
    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos
    }
    pub fn raw(&self) -> &[u8] {
        &self.raw
    }
}
fn hash_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
fn bounded(path: &Path, max: u64) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    File::open(path)?.take(max + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > max {
        return Err("Augur metadata extent".into());
    }
    Ok(bytes)
}
impl AugurManifest {
    pub fn validate(&self) -> Result<()> {
        if self.schema != "quasar-session-model-manifest-v1"
            || self.upstream_repo != "QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4"
            || self.upstream_revision.len() != 40
            || !self
                .upstream_revision
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            || [
                &self.checkpoint.config_sha256,
                &self.checkpoint.index_sha256,
                &self.tokenizer_sha256,
                &self.tokenizer_config_sha256,
                &self.generation_config_sha256,
                &self.chat_template_sha256,
            ]
            .iter()
            .any(|s| !valid_digest(s))
            || self.checkpoint.shards.len() != 5
        {
            return Err("invalid explicit QUASAR model manifest".into());
        }
        for i in 1..=5 {
            let name = format!("model-{i:05}-of-00005.safetensors");
            if !self
                .checkpoint
                .shards
                .get(&name)
                .is_some_and(|s| valid_digest(s))
            {
                return Err("Augur manifest requires exactly five named shard digests".into());
            }
        }
        Ok(())
    }
}
/// The caller supplies the expected digest; a file's own declaration is not
/// authentication. EOS comes from the hashed generation config, but sampling
/// remains explicitly GREEDY, not an implementation of its do_sample=true.
pub fn load(root: &Path, path: &Path, expected_sha256: &str) -> Result<AdmittedManifest> {
    if !valid_digest(expected_sha256) {
        return Err("expected manifest digest required".into());
    }
    let raw = bounded(path, 1024 * 1024)?;
    if digest(&raw) != expected_sha256 {
        return Err("Augur manifest digest mismatch".into());
    }
    let manifest: AugurManifest = serde_json::from_slice(&raw)?;
    manifest.validate()?;
    if hash_file(&root.join("tokenizer.json"))? != manifest.tokenizer_sha256 {
        return Err("QUASAR tokenizer digest mismatch".into());
    }
    // Check and parse the SAME retained bytes (no hash/reopen TOCTOU).
    let template = bounded(&root.join("tokenizer_config.json"), 4 * 1024 * 1024)?;
    let jinja = bounded(&root.join("chat_template.jinja"), 4 * 1024 * 1024)?;
    let generation = bounded(&root.join("generation_config.json"), 1024 * 1024)?;
    if digest(&template) != manifest.tokenizer_config_sha256
        || digest(&generation) != manifest.generation_config_sha256
        || digest(&jinja) != manifest.chat_template_sha256
    {
        return Err("QUASAR template/generation identity mismatch".into());
    }
    let t: serde_json::Value = serde_json::from_slice(&template)?;
    if !t["chat_template"]
        .as_str()
        .is_some_and(|s| !s.is_empty() && s.as_bytes() == jinja)
    {
        return Err("Augur canonical chat template missing".into());
    }
    let g: serde_json::Value = serde_json::from_slice(&generation)?;
    let eos: Vec<u32> = if let Some(id) = g["eos_token_id"].as_u64() {
        vec![u32::try_from(id)?]
    } else {
        serde_json::from_value(g["eos_token_id"].clone())?
    };
    if eos.is_empty()
        || eos.iter().any(|&id| id >= VOCAB)
        || eos.iter().collect::<std::collections::HashSet<_>>().len() != eos.len()
    {
        return Err("invalid QUASAR EOS policy".into());
    }
    Ok(AdmittedManifest {
        root: root.to_owned(),
        manifest,
        sha256: expected_sha256.into(),
        eos,
        raw,
    })
}
