//! Stable, CUDA-free inspection of a Qwen DFlash safetensors artifact.
//!
//! The inspector binds the complete file SHA, config SHA, header geometry,
//! tensor dtype, and contiguous payload extent before a future H2D/upload path
//! is allowed to call the lifecycle admission API.

use crate::{DFlashArtifactIdentity, DFlashDType};
use augur_model_qwen35::{validate_strict_json, DFlashConfig};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const MAX_HEADER_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DFlashArtifactInspection {
    pub artifact: DFlashArtifactIdentity,
    pub artifact_path: PathBuf,
    pub config_path: PathBuf,
    pub file_bytes: u64,
    pub header_bytes: u64,
    pub tensor_count: usize,
    pub payload_sha256: [u8; 32],
}

/// A validated, pageable-RAM owner for a DFlash artifact.
///
/// The resident copy is ordinary host memory, not a pinned allocation and not
/// a CUDA pointer. The uploader borrows bounded pinned lanes only during H2D.
#[derive(Clone, Debug)]
pub struct DFlashRamResident {
    inspection: DFlashArtifactInspection,
    file_bytes: Arc<[u8]>,
}

impl DFlashRamResident {
    /// Load and revalidate the complete artifact into pageable RAM.
    pub fn load(inspection: &DFlashArtifactInspection) -> Result<Self, DFlashArtifactError> {
        Self::load_with_budget(inspection, u64::MAX)
    }

    /// Load a complete artifact only when its pageable-RAM footprint fits the
    /// caller's explicit host budget. The size check happens before the file is
    /// read so an over-budget request cannot transiently allocate the resident.
    pub fn load_with_budget(
        inspection: &DFlashArtifactInspection,
        max_bytes: u64,
    ) -> Result<Self, DFlashArtifactError> {
        if max_bytes == 0 || inspection.file_bytes > max_bytes {
            return Err(DFlashArtifactError::Invalid(
                "RAM artifact exceeds resident budget".into(),
            ));
        }
        let bytes = read_stable_file(&inspection.artifact_path, "DFlash RAM artifact")?;
        if bytes.len() as u64 != inspection.file_bytes {
            return Err(DFlashArtifactError::Invalid(
                "RAM artifact extent differs from inspection".into(),
            ));
        }
        if sha256(&bytes) != inspection.artifact.artifact_sha256 {
            return Err(DFlashArtifactError::Invalid(
                "RAM artifact SHA differs from inspection".into(),
            ));
        }
        let payload_offset =
            usize::try_from(inspection.header_bytes.checked_add(8).ok_or_else(|| {
                DFlashArtifactError::Invalid("RAM payload offset overflow".into())
            })?)
            .map_err(|_| DFlashArtifactError::Invalid("RAM payload offset overflow".into()))?;
        let payload = bytes.get(payload_offset..).ok_or_else(|| {
            DFlashArtifactError::Invalid("RAM payload offset exceeds artifact".into())
        })?;
        if sha256(payload) != inspection.payload_sha256 {
            return Err(DFlashArtifactError::Invalid(
                "RAM payload SHA differs from inspection".into(),
            ));
        }
        Ok(Self {
            inspection: inspection.clone(),
            file_bytes: Arc::from(bytes),
        })
    }

    pub fn inspection(&self) -> &DFlashArtifactInspection {
        &self.inspection
    }

    pub fn file_bytes(&self) -> &[u8] {
        &self.file_bytes
    }

    pub fn payload_offset(&self) -> usize {
        (self.inspection.header_bytes + 8) as usize
    }

    pub fn payload(&self) -> &[u8] {
        &self.file_bytes[self.payload_offset()..]
    }

    pub fn retained_bytes(&self) -> u64 {
        self.file_bytes.len() as u64
    }
}

#[derive(Debug)]
pub enum DFlashArtifactError {
    Io(String),
    Invalid(String),
    Config(String),
}

impl fmt::Display for DFlashArtifactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "DFlash artifact I/O failed: {error}"),
            Self::Invalid(error) => write!(f, "invalid DFlash safetensors artifact: {error}"),
            Self::Config(error) => write!(f, "invalid DFlash config: {error}"),
        }
    }
}

impl std::error::Error for DFlashArtifactError {}

fn read_stable_file(path: &Path, role: &str) -> Result<Vec<u8>, DFlashArtifactError> {
    let before =
        fs::symlink_metadata(path).map_err(|e| DFlashArtifactError::Io(format!("{role}: {e}")))?;
    if !before.file_type().is_file() {
        return Err(DFlashArtifactError::Invalid(format!(
            "{role} is not a regular file"
        )));
    }
    let bytes = fs::read(path).map_err(|e| DFlashArtifactError::Io(format!("{role}: {e}")))?;
    let after =
        fs::symlink_metadata(path).map_err(|e| DFlashArtifactError::Io(format!("{role}: {e}")))?;
    if !after.file_type().is_file()
        || before.len() != after.len()
        || after.len() != bytes.len() as u64
    {
        return Err(DFlashArtifactError::Invalid(format!(
            "{role} changed while being read"
        )));
    }
    Ok(bytes)
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

/// Inspect a DFlash safetensors artifact and bind it to its config.
pub fn inspect_dflash_artifact(
    artifact_path: impl AsRef<Path>,
    config_path: impl AsRef<Path>,
    allow_f16: bool,
) -> Result<DFlashArtifactInspection, DFlashArtifactError> {
    let artifact_path = artifact_path.as_ref().to_owned();
    let config_path = config_path.as_ref().to_owned();
    let config_bytes = read_stable_file(&config_path, "DFlash config")?;
    let config = if allow_f16 {
        DFlashConfig::parse_f16_opt_in(&config_bytes)
    } else {
        DFlashConfig::parse(&config_bytes)
    }
    .map_err(|e| DFlashArtifactError::Config(e.to_string()))?;
    let bytes = read_stable_file(&artifact_path, "DFlash artifact")?;
    let file_bytes = u64::try_from(bytes.len())
        .map_err(|_| DFlashArtifactError::Invalid("file size overflow".into()))?;
    let prefix: [u8; 8] = bytes
        .get(..8)
        .ok_or_else(|| DFlashArtifactError::Invalid("missing safetensors length prefix".into()))?
        .try_into()
        .map_err(|_| DFlashArtifactError::Invalid("invalid safetensors length prefix".into()))?;
    let header_bytes = u64::from_le_bytes(prefix);
    if header_bytes == 0 || header_bytes > MAX_HEADER_BYTES {
        return Err(DFlashArtifactError::Invalid(format!(
            "header length {header_bytes} exceeds bound"
        )));
    }
    let header_end = 8_u64
        .checked_add(header_bytes)
        .ok_or_else(|| DFlashArtifactError::Invalid("header extent overflow".into()))?;
    let header_end_usize = usize::try_from(header_end).map_err(|_| {
        DFlashArtifactError::Invalid("header extent does not fit host usize".into())
    })?;
    let header = bytes
        .get(8..header_end_usize)
        .ok_or_else(|| DFlashArtifactError::Invalid("truncated safetensors header".into()))?;
    validate_strict_json(header)
        .map_err(|e| DFlashArtifactError::Invalid(format!("strict header JSON: {e}")))?;
    let root: Value = serde_json::from_slice(header)
        .map_err(|e| DFlashArtifactError::Invalid(format!("header JSON: {e}")))?;
    let object = root
        .as_object()
        .ok_or_else(|| DFlashArtifactError::Invalid("header root is not an object".into()))?;
    let payload_bytes = file_bytes
        .checked_sub(header_end)
        .ok_or_else(|| DFlashArtifactError::Invalid("header exceeds file".into()))?;
    let expected_dtype = if config.dtype == "float16" {
        DFlashDType::F16
    } else {
        DFlashDType::Bf16
    };
    let dtype_name = match expected_dtype {
        DFlashDType::F16 => "F16",
        DFlashDType::Bf16 => "BF16",
    };
    let mut ranges = Vec::new();
    for (name, entry) in object {
        if name == "__metadata__" {
            if !entry.is_object() {
                return Err(DFlashArtifactError::Invalid(
                    "__metadata__ is not an object".into(),
                ));
            }
            continue;
        }
        let tensor = entry.as_object().ok_or_else(|| {
            DFlashArtifactError::Invalid(format!("tensor {name} is not an object"))
        })?;
        if tensor.get("dtype").and_then(Value::as_str) != Some(dtype_name) {
            return Err(DFlashArtifactError::Invalid(format!(
                "tensor {name} dtype is not {dtype_name}"
            )));
        }
        let offsets = tensor
            .get("data_offsets")
            .and_then(Value::as_array)
            .filter(|v| v.len() == 2)
            .ok_or_else(|| DFlashArtifactError::Invalid(format!("tensor {name} offsets")))?;
        let start = offsets[0]
            .as_u64()
            .ok_or_else(|| DFlashArtifactError::Invalid(format!("tensor {name} start offset")))?;
        let end = offsets[1]
            .as_u64()
            .ok_or_else(|| DFlashArtifactError::Invalid(format!("tensor {name} end offset")))?;
        if start > end || end > payload_bytes {
            return Err(DFlashArtifactError::Invalid(format!(
                "tensor {name} offsets exceed payload"
            )));
        }
        ranges.push((start, end, name.clone()));
    }
    if ranges.is_empty() {
        return Err(DFlashArtifactError::Invalid(
            "artifact contains no tensors".into(),
        ));
    }
    ranges.sort_by_key(|(start, _, _)| *start);
    let mut cursor = 0_u64;
    for (start, end, name) in &ranges {
        if *start != cursor {
            return Err(DFlashArtifactError::Invalid(format!(
                "tensor {name} leaves a payload gap or overlaps"
            )));
        }
        cursor = *end;
    }
    if cursor != payload_bytes {
        return Err(DFlashArtifactError::Invalid(
            "tensor ranges do not cover the payload".into(),
        ));
    }
    Ok(DFlashArtifactInspection {
        artifact: DFlashArtifactIdentity {
            artifact_sha256: sha256(&bytes),
            config_sha256: sha256(&config_bytes),
            dtype: expected_dtype,
            bytes: payload_bytes,
        },
        artifact_path,
        config_path,
        file_bytes,
        header_bytes,
        tensor_count: ranges.len(),
        payload_sha256: sha256(&bytes[header_end_usize..]),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn ram_resident_revalidates_full_file_and_payload() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "qwen35-dflash-ram-resident-{}-{unique}.bin",
            std::process::id()
        ));
        let header = b"header-for-test";
        let payload = b"payload-for-test";
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(payload);
        std::fs::write(&path, &bytes).expect("write fixture");
        let inspection = DFlashArtifactInspection {
            artifact: DFlashArtifactIdentity {
                artifact_sha256: sha256(&bytes),
                config_sha256: [3; 32],
                dtype: DFlashDType::Bf16,
                bytes: payload.len() as u64,
            },
            artifact_path: path.clone(),
            config_path: path.clone(),
            file_bytes: bytes.len() as u64,
            header_bytes: header.len() as u64,
            tensor_count: 1,
            payload_sha256: sha256(payload),
        };
        let resident = DFlashRamResident::load(&inspection).expect("resident load");
        assert_eq!(resident.file_bytes(), bytes.as_slice());
        assert_eq!(resident.payload(), payload);
        assert_eq!(resident.retained_bytes(), bytes.len() as u64);
        std::fs::remove_file(path).expect("remove fixture");
    }

    #[test]
    fn ram_resident_rejects_extent_or_hash_drift() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "qwen35-dflash-ram-resident-drift-{}-{unique}.bin",
            std::process::id()
        ));
        let bytes = b"stable-file";
        std::fs::write(&path, bytes).expect("write fixture");
        let inspection = DFlashArtifactInspection {
            artifact: DFlashArtifactIdentity {
                artifact_sha256: [7; 32],
                config_sha256: [3; 32],
                dtype: DFlashDType::Bf16,
                bytes: 1,
            },
            artifact_path: path.clone(),
            config_path: path.clone(),
            file_bytes: bytes.len() as u64,
            header_bytes: 0,
            tensor_count: 1,
            payload_sha256: [8; 32],
        };
        let error = DFlashRamResident::load(&inspection).expect_err("drift must fail");
        assert!(error.to_string().contains("SHA"));
        std::fs::remove_file(path).expect("remove fixture");
    }

    #[test]
    fn ram_resident_rejects_budget_before_read() {
        let inspection = DFlashArtifactInspection {
            artifact: DFlashArtifactIdentity {
                artifact_sha256: [1; 32],
                config_sha256: [2; 32],
                dtype: DFlashDType::Bf16,
                bytes: 8,
            },
            artifact_path: std::env::temp_dir().join("qwen35-missing-ram-budget-fixture"),
            config_path: std::env::temp_dir().join("qwen35-missing-ram-budget-config"),
            file_bytes: 8,
            header_bytes: 0,
            tensor_count: 1,
            payload_sha256: [3; 32],
        };
        let error = DFlashRamResident::load_with_budget(&inspection, 7)
            .expect_err("over-budget resident must fail before I/O");
        assert!(error.to_string().contains("budget"));
    }
}
