//! CPU oracle for the real DFlash feature-to-draft-KV boundary.
//!
//! The target feature is projected independently through all five draft
//! layer K/V matrices.  This is intentionally separate from the decoder:
//! draft hidden states still come from target anchor/mask embeddings and the
//! projected K/V must not be called a cache-ready draft state until RoPE and
//! cache insertion are wired on the same execution path.

use augur_model_qwen35::{quantized_matvec, GgufAuditOptions, GgufManifest, GgufTensor};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    fmt,
    path::{Path, PathBuf},
};

pub const DFLASH_FEATURE_KV_SCHEMA: &str = "qwen35-dflash-feature-kv-cpu-oracle-v1";
pub const DFLASH_FEATURE_HIDDEN: usize = 5_120;
pub const DFLASH_TARGET_FEATURE_ELEMENTS: usize = 25_600;
pub const DFLASH_FEATURE_HEADS: usize = 8;
pub const DFLASH_FEATURE_HEAD_DIM: usize = 128;
pub const DFLASH_FEATURE_KV: usize = DFLASH_FEATURE_HEADS * DFLASH_FEATURE_HEAD_DIM;
pub const DFLASH_TARGET_LAYER_IDS: [usize; 5] = [5, 19, 33, 47, 61];
pub const DFLASH_GGUF_TARGET_LAYER_IDS: [usize; 5] = [6, 20, 34, 48, 62];

#[derive(Clone, Debug, Serialize)]
pub struct DFlashFeatureKvOutput {
    pub schema: String,
    pub source_path: PathBuf,
    pub source_sha256: String,
    pub input_sha256: String,
    pub target_layer_ids: [usize; 5],
    pub gguf_target_layer_ids: [usize; 5],
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
    pub k_normed: Vec<Vec<f32>>,
    pub k_norm_epsilon: f32,
    pub rope_required_before_cache: bool,
}

#[derive(Debug)]
pub enum DFlashFeatureKvError {
    Invalid(String),
    Model(augur_model_qwen35::Qwen35Error),
}

impl fmt::Display for DFlashFeatureKvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(s) => write!(f, "invalid DFlash feature KV: {s}"),
            Self::Model(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for DFlashFeatureKvError {}
impl From<augur_model_qwen35::Qwen35Error> for DFlashFeatureKvError {
    fn from(e: augur_model_qwen35::Qwen35Error) -> Self {
        Self::Model(e)
    }
}

/// Execute the complete real Q8 encoder boundary (`fc -> output_norm`) and
/// then produce all five draft K/V projections. This is still a CPU oracle;
/// no decoder token, RoPE cache write, or acceptance decision is performed.
pub fn encode_target_features_kv_q8(
    path: impl AsRef<Path>,
    target_features: &[f32],
    epsilon: f32,
) -> Result<DFlashFeatureKvOutput, DFlashFeatureKvError> {
    if target_features.len() != DFLASH_TARGET_FEATURE_ELEMENTS {
        return Err(DFlashFeatureKvError::Invalid(
            "target feature concat geometry".into(),
        ));
    }
    let path = path.as_ref();
    let manifest = GgufManifest::inspect(
        path,
        GgufAuditOptions {
            hash_file: true,
            ..Default::default()
        },
    )?;
    let fc_name = "fc.weight";
    let norm_name = "enc.output_norm.weight";
    let fc = manifest.tensor(fc_name).ok_or_else(|| missing(fc_name))?;
    if fc.dimensions
        != vec![
            DFLASH_TARGET_FEATURE_ELEMENTS as u64,
            DFLASH_FEATURE_HIDDEN as u64,
        ]
        || fc.ggml_type != 8
    {
        return Err(DFlashFeatureKvError::Invalid(
            "fc.weight must be Q8_0 [25600,5120]".into(),
        ));
    }
    let norm = manifest
        .tensor(norm_name)
        .ok_or_else(|| missing(norm_name))?;
    if norm.dimensions != vec![DFLASH_FEATURE_HIDDEN as u64] || norm.ggml_type != 0 {
        return Err(DFlashFeatureKvError::Invalid(
            "enc.output_norm.weight must be F32 [5120]".into(),
        ));
    }
    let mut reader = manifest.open_tensor_reader(path)?;
    let fc_bytes = reader.read_tensor_bytes(fc_name)?;
    let norm_bytes = reader.read_tensor_bytes(norm_name)?;
    let projected = quantized_matvec(fc, &fc_bytes, target_features)?;
    let mut normalized = vec![0.0_f32; DFLASH_FEATURE_HIDDEN];
    let mut mean = 0.0_f32;
    for value in &projected {
        mean += value * value;
    }
    let scale = (mean / DFLASH_FEATURE_HIDDEN as f32 + epsilon)
        .sqrt()
        .recip();
    for (index, value) in normalized.iter_mut().enumerate() {
        let start = index * 4;
        let weight = f32::from_le_bytes(
            norm_bytes[start..start + 4]
                .try_into()
                .map_err(|_| DFlashFeatureKvError::Invalid("output norm extent".into()))?,
        );
        *value = projected[index] * scale * weight;
    }
    project_feature_kv_q8_from_manifest(path, &manifest, &normalized, epsilon)
}

/// Project one normalized target feature through the five real Q8 draft K/V
/// matrices and apply the per-head K RMSNorm. `feature` is the output of the
/// shared `fc -> hidden_norm` encoder, not a decoder token embedding.
pub fn project_feature_kv_q8(
    path: impl AsRef<Path>,
    feature: &[f32],
    epsilon: f32,
) -> Result<DFlashFeatureKvOutput, DFlashFeatureKvError> {
    if feature.len() != DFLASH_FEATURE_HIDDEN || !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(DFlashFeatureKvError::Invalid(
            "feature geometry or epsilon".into(),
        ));
    }
    let path = path.as_ref().to_owned();
    let manifest = GgufManifest::inspect(
        &path,
        GgufAuditOptions {
            hash_file: true,
            ..Default::default()
        },
    )?;
    project_feature_kv_q8_from_manifest(&path, &manifest, feature, epsilon)
}

fn project_feature_kv_q8_from_manifest(
    path: &Path,
    manifest: &GgufManifest,
    feature: &[f32],
    epsilon: f32,
) -> Result<DFlashFeatureKvOutput, DFlashFeatureKvError> {
    let source_sha256 = manifest
        .file_sha256
        .clone()
        .ok_or_else(|| DFlashFeatureKvError::Invalid("manifest file digest missing".into()))?;
    let input_sha256 = digest_f32(feature);
    let mut k = Vec::with_capacity(5);
    let mut v = Vec::with_capacity(5);
    let mut k_normed = Vec::with_capacity(5);
    for layer in 0..5 {
        let k_name = format!("blk.{layer}.attn_k.weight");
        let v_name = format!("blk.{layer}.attn_v.weight");
        let kn_name = format!("blk.{layer}.attn_k_norm.weight");
        let (k_tensor, v_tensor, kn_tensor) = (
            manifest.tensor(&k_name).ok_or_else(|| missing(&k_name))?,
            manifest.tensor(&v_name).ok_or_else(|| missing(&v_name))?,
            manifest.tensor(&kn_name).ok_or_else(|| missing(&kn_name))?,
        );
        validate_matrix(k_tensor, &k_name)?;
        validate_matrix(v_tensor, &v_name)?;
        if kn_tensor.dimensions != vec![DFLASH_FEATURE_HEAD_DIM as u64] || kn_tensor.ggml_type != 0
        {
            return Err(DFlashFeatureKvError::Invalid(format!("{kn_name} geometry")));
        }
        let mut reader = manifest.open_tensor_reader(path)?;
        let kb = reader.read_tensor_bytes(&k_name)?;
        let vb = reader.read_tensor_bytes(&v_name)?;
        let nb = reader.read_tensor_bytes(&kn_name)?;
        let kp = quantized_matvec(k_tensor, &kb, feature)?;
        let vp = quantized_matvec(v_tensor, &vb, feature)?;
        let mut kn = vec![0.0; DFLASH_FEATURE_KV];
        for head in 0..DFLASH_FEATURE_HEADS {
            let base = head * DFLASH_FEATURE_HEAD_DIM;
            let mut mean = 0.0;
            for x in &kp[base..base + DFLASH_FEATURE_HEAD_DIM] {
                mean += x * x;
            }
            let scale = (mean / DFLASH_FEATURE_HEAD_DIM as f32 + epsilon)
                .sqrt()
                .recip();
            for j in 0..DFLASH_FEATURE_HEAD_DIM {
                let w = f32::from_le_bytes(nb[(j * 4)..(j * 4 + 4)].try_into().unwrap());
                kn[base + j] = kp[base + j] * scale * w;
            }
        }
        k.push(kp);
        v.push(vp);
        k_normed.push(kn);
    }
    Ok(DFlashFeatureKvOutput {
        schema: DFLASH_FEATURE_KV_SCHEMA.into(),
        source_path: path.to_path_buf(),
        source_sha256,
        input_sha256,
        target_layer_ids: DFLASH_TARGET_LAYER_IDS,
        gguf_target_layer_ids: DFLASH_GGUF_TARGET_LAYER_IDS,
        k,
        v,
        k_normed,
        k_norm_epsilon: epsilon,
        rope_required_before_cache: true,
    })
}

fn validate_matrix(t: &GgufTensor, name: &str) -> Result<(), DFlashFeatureKvError> {
    if t.dimensions != vec![DFLASH_FEATURE_HIDDEN as u64, DFLASH_FEATURE_KV as u64]
        || t.ggml_type != 8
    {
        return Err(DFlashFeatureKvError::Invalid(format!(
            "{name} must be Q8_0 [5120,1024]"
        )));
    }
    Ok(())
}
fn missing(name: &str) -> DFlashFeatureKvError {
    DFlashFeatureKvError::Invalid(format!("missing tensor {name}"))
}
fn digest_f32(values: &[f32]) -> String {
    let mut h = Sha256::new();
    for v in values {
        h.update(v.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn layer_mapping_matches_converter_contract() {
        assert_eq!(DFLASH_GGUF_TARGET_LAYER_IDS, [6, 20, 34, 48, 62]);
    }
    #[test]
    fn per_head_k_norm_is_not_decoder_norm() {
        let x = [3.0_f32; DFLASH_FEATURE_HEAD_DIM];
        let rms = (x.iter().map(|v| v * v).sum::<f32>() / 128.0 + 1e-6).sqrt();
        assert!((3.0 / rms - 1.0).abs() < 1e-3);
    }
}
