use crate::config::{DFlashConfig, Qwen35Config};
use crate::gguf::{GgufAuditOptions, GgufManifest, GgufValue};
use crate::schema::Qwen35TensorSchema;
use crate::tokenizer::TokenizerIdentity;
use crate::Qwen35Error;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fs::{self, File, Metadata, OpenOptions};
use std::io::Read;
use std::path::{Component, Path, PathBuf};

pub const PROFILE_ID: &str = "crystal-quant.qwen3.8-27b.experimental.v1";
pub const GGUF_ARCHITECTURE: &str = "qwen35";

#[derive(Clone, Debug)]
pub struct ArtifactPaths {
    pub config: PathBuf,
    pub target_gguf: PathBuf,
    pub tokenizer_json: PathBuf,
    pub chat_template: PathBuf,
    pub dflash_config: Option<PathBuf>,
    pub draft_gguf: Option<PathBuf>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GgufIdentitySummary {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_count: u64,
    pub alignment: u64,
    pub header_bytes: u64,
    pub data_offset: u64,
    pub file_bytes: u64,
    pub file_sha256: String,
    pub quantization_tensor_counts: std::collections::BTreeMap<String, usize>,
    pub quantization_bytes: std::collections::BTreeMap<String, u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ModelIdentity {
    pub profile_id: String,
    pub config_sha256: String,
    pub target_gguf_sha256: String,
    pub target_gguf_bytes: u64,
    pub tokenizer: TokenizerIdentity,
    pub dflash_config_sha256: Option<String>,
    pub draft_gguf_sha256: Option<String>,
    pub gguf: GgufIdentitySummary,
    pub qwen: Qwen35Config,
    pub dflash: Option<DFlashConfig>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ModelIdentityReceipt {
    pub schema: String,
    pub status: String,
    pub profile_id: String,
    pub config_sha256: String,
    pub target_gguf_sha256: String,
    pub target_gguf_bytes: u64,
    pub tokenizer: TokenizerIdentity,
    pub dflash_config_sha256: Option<String>,
    pub draft_gguf_sha256: Option<String>,
    pub gguf: GgufIdentitySummary,
    pub claims: IdentityClaims,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct IdentityClaims {
    pub cpu_only: bool,
    pub component_only: bool,
    pub cuda_started: bool,
    pub decode_executed: bool,
    pub token_publication_permitted: bool,
    pub product_tps_permitted: bool,
}

impl ModelIdentity {
    pub fn audit(paths: &ArtifactPaths) -> Result<Self, Qwen35Error> {
        let config_bytes = read_stable_bytes(&paths.config)?;
        let qwen = Qwen35Config::parse(&config_bytes)?;
        let gguf = GgufManifest::inspect(&paths.target_gguf, GgufAuditOptions::default())?;
        validate_qwen_gguf(&qwen, &gguf)?;
        Qwen35TensorSchema::validate_manifest(&qwen, &gguf)?;
        let tokenizer = TokenizerIdentity::audit(&paths.tokenizer_json, &paths.chat_template)?;
        if tokenizer.maximum_token_id >= qwen.vocab_size as u32 {
            return Err(Qwen35Error::invalid(
                "tokenizer.json token ids",
                format!(
                    "maximum token id {} exceeds model vocab size {}",
                    tokenizer.maximum_token_id, qwen.vocab_size
                ),
            ));
        }
        let (dflash, dflash_config_sha256) = match &paths.dflash_config {
            Some(path) => {
                let bytes = read_stable_bytes(path)?;
                (Some(DFlashConfig::parse(&bytes)?), Some(sha256_hex(&bytes)))
            }
            None => (None, None),
        };
        let draft_gguf_sha256 = match &paths.draft_gguf {
            Some(path) => {
                let mut file = open_regular_file(path)?;
                Some(sha256_file(&mut file)?)
            }
            None => None,
        };
        let target_gguf_sha256 = gguf
            .file_sha256
            .clone()
            .ok_or_else(|| Qwen35Error::Gguf("GGUF hash was not requested".into()))?;
        Ok(Self {
            profile_id: PROFILE_ID.into(),
            config_sha256: sha256_hex(&config_bytes),
            target_gguf_sha256,
            target_gguf_bytes: gguf.file_size,
            tokenizer,
            dflash_config_sha256,
            draft_gguf_sha256,
            gguf: summarize_gguf(&gguf),
            qwen,
            dflash,
        })
    }

    pub fn receipt(&self) -> ModelIdentityReceipt {
        ModelIdentityReceipt {
            schema: "CQ-QWEN35-MODEL-IDENTITY-RECEIPT-V1".into(),
            status: "DIAGNOSTIC_ONLY_CPU_IDENTITY".into(),
            profile_id: self.profile_id.clone(),
            config_sha256: self.config_sha256.clone(),
            target_gguf_sha256: self.target_gguf_sha256.clone(),
            target_gguf_bytes: self.target_gguf_bytes,
            tokenizer: self.tokenizer.clone(),
            dflash_config_sha256: self.dflash_config_sha256.clone(),
            draft_gguf_sha256: self.draft_gguf_sha256.clone(),
            gguf: self.gguf.clone(),
            claims: IdentityClaims {
                cpu_only: true,
                component_only: true,
                cuda_started: false,
                decode_executed: false,
                token_publication_permitted: false,
                product_tps_permitted: false,
            },
        }
    }
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub(crate) fn sha256_file(file: &mut File) -> Result<String, Qwen35Error> {
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 8 * 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub(crate) fn read_stable_bytes(path: &Path) -> Result<Vec<u8>, Qwen35Error> {
    validate_regular_path(path)?;
    let mut file = open_regular_file(path)?;
    let before = file.metadata()?;
    let length = usize::try_from(before.len()).map_err(|_| Qwen35Error::SizeOverflow)?;
    let mut bytes = Vec::with_capacity(length);
    file.read_to_end(&mut bytes)?;
    let after = file.metadata()?;
    ensure_stable_metadata(&before, after)?;
    if bytes.len() != length {
        return Err(Qwen35Error::Io("file length changed during read".into()));
    }
    Ok(bytes)
}

fn open_regular_file(path: &Path) -> Result<File, Qwen35Error> {
    validate_regular_path(path)?;
    Ok(OpenOptions::new().read(true).open(path)?)
}

fn validate_regular_path(path: &Path) -> Result<(), Qwen35Error> {
    let mut current = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => current.push(prefix.as_os_str()),
            Component::RootDir => current.push(std::path::MAIN_SEPARATOR.to_string()),
            Component::CurDir => {}
            Component::ParentDir => {
                return Err(Qwen35Error::Io(
                    "artifact path may not contain parent components".into(),
                ));
            }
            Component::Normal(name) => {
                current.push(name);
                let metadata = fs::symlink_metadata(&current).map_err(|error| {
                    Qwen35Error::Io(format!(
                        "cannot inspect path component {current:?}: {error}"
                    ))
                })?;
                if metadata.file_type().is_symlink() || is_reparse_point(&metadata) {
                    return Err(Qwen35Error::Io(format!(
                        "reparse/symlink path component is not allowed: {current:?}"
                    )));
                }
            }
        }
    }
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.is_file() || metadata.file_type().is_symlink() || is_reparse_point(&metadata) {
        return Err(Qwen35Error::Io(
            "artifact must be a regular non-reparse file".into(),
        ));
    }
    Ok(())
}

fn is_reparse_point(metadata: &Metadata) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0400;
        metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
    }
    #[cfg(not(windows))]
    {
        let _ = metadata;
        false
    }
}

fn ensure_stable_metadata(before: &Metadata, after: Metadata) -> Result<(), Qwen35Error> {
    if before.len() != after.len() || before.modified().ok() != after.modified().ok() {
        return Err(Qwen35Error::Io(
            "artifact metadata changed during read".into(),
        ));
    }
    Ok(())
}

fn validate_qwen_gguf(config: &Qwen35Config, gguf: &GgufManifest) -> Result<(), Qwen35Error> {
    expect_metadata_string(gguf, "general.architecture", GGUF_ARCHITECTURE)?;
    expect_metadata_string(gguf, "general.type", "model")?;
    expect_metadata_u64(gguf, "qwen35.block_count", config.num_hidden_layers as u64)?;
    expect_metadata_u64(
        gguf,
        "qwen35.context_length",
        config.max_position_embeddings as u64,
    )?;
    expect_metadata_u64(gguf, "qwen35.embedding_length", config.hidden_size as u64)?;
    expect_metadata_u64(
        gguf,
        "qwen35.feed_forward_length",
        config.intermediate_size as u64,
    )?;
    expect_metadata_u64(
        gguf,
        "qwen35.attention.head_count",
        config.num_attention_heads as u64,
    )?;
    expect_metadata_u64(
        gguf,
        "qwen35.attention.head_count_kv",
        config.num_key_value_heads as u64,
    )?;
    expect_metadata_u64(gguf, "qwen35.attention.key_length", config.head_dim as u64)?;
    expect_metadata_u64(
        gguf,
        "qwen35.attention.value_length",
        config.head_dim as u64,
    )?;
    expect_metadata_u64(
        gguf,
        "qwen35.full_attention_interval",
        config.full_attention_interval as u64,
    )?;
    expect_metadata_u64(gguf, "qwen35.ssm.conv_kernel", 4)?;
    expect_metadata_u64(gguf, "qwen35.ssm.state_size", 128)?;
    expect_metadata_u64(gguf, "qwen35.ssm.group_count", 16)?;
    expect_metadata_u64(gguf, "qwen35.ssm.time_step_rank", 48)?;
    expect_metadata_u64(gguf, "qwen35.ssm.inner_size", 6144)?;
    expect_metadata_string(gguf, "tokenizer.ggml.model", "gpt2")?;
    let tokens = gguf
        .metadata("tokenizer.ggml.tokens")
        .and_then(GgufValue::array_len)
        .ok_or_else(|| Qwen35Error::Gguf("tokenizer token array is missing".into()))?;
    if tokens != config.vocab_size {
        return Err(Qwen35Error::invalid(
            "tokenizer.ggml.tokens",
            format!("expected {} entries, got {tokens}", config.vocab_size),
        ));
    }
    let counts = gguf.quantization_counts();
    for (kind, label) in [(8, "Q8_0"), (30, "BF16"), (40, "NVFP4")] {
        if counts.get(&kind).copied().unwrap_or(0) == 0 {
            return Err(Qwen35Error::Gguf(format!(
                "expected at least one {label} tensor"
            )));
        }
    }
    Ok(())
}

fn expect_metadata_string(
    gguf: &GgufManifest,
    key: &str,
    expected: &str,
) -> Result<(), Qwen35Error> {
    let actual = gguf
        .metadata(key)
        .and_then(GgufValue::as_str)
        .ok_or_else(|| Qwen35Error::Gguf(format!("metadata {key:?} is not a string")))?;
    if actual == expected {
        Ok(())
    } else {
        Err(Qwen35Error::Gguf(format!(
            "metadata {key:?}: expected {expected:?}, got {actual:?}"
        )))
    }
}

fn expect_metadata_u64(gguf: &GgufManifest, key: &str, expected: u64) -> Result<(), Qwen35Error> {
    let actual = gguf
        .metadata(key)
        .and_then(GgufValue::as_u64)
        .ok_or_else(|| Qwen35Error::Gguf(format!("metadata {key:?} is not an integer")))?;
    if actual == expected {
        Ok(())
    } else {
        Err(Qwen35Error::Gguf(format!(
            "metadata {key:?}: expected {expected}, got {actual}"
        )))
    }
}

fn summarize_gguf(gguf: &GgufManifest) -> GgufIdentitySummary {
    GgufIdentitySummary {
        version: gguf.version,
        tensor_count: gguf.tensor_count,
        metadata_count: gguf.metadata_count,
        alignment: gguf.alignment,
        header_bytes: gguf.header_bytes,
        data_offset: gguf.data_offset,
        file_bytes: gguf.file_size,
        file_sha256: gguf.file_sha256.clone().unwrap_or_default(),
        quantization_tensor_counts: gguf
            .quantization_counts()
            .into_iter()
            .map(|(kind, count)| (kind.to_string(), count))
            .collect(),
        quantization_bytes: gguf
            .quantization_bytes()
            .into_iter()
            .map(|(kind, bytes)| (kind.to_string(), bytes))
            .collect(),
    }
}
