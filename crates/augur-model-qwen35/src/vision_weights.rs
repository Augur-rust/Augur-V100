//! Exact CPU metadata binding for the local BF16 Vision payload.
//! Does not read tensor payloads, convert BF16, allocate CUDA, or run Vision.

use crate::{validate_strict_json, Qwen35Error, SafetensorsManifest, SafetensorsRecord};
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

const PREFIX: &str = "model.visual.";
const PAYLOAD_BYTES: u64 = 921_460_192;

#[derive(Debug, Clone)]
pub struct Qwen35VisionProfile {
    config_sha256: String,
}

impl Qwen35VisionProfile {
    /// Fixed architecture only. Unknown Vision config keys reject, so a later
    /// architecture change cannot silently inherit this tensor binding.
    pub fn from_config_bytes(bytes: &[u8]) -> Result<Self, Qwen35Error> {
        if bytes.len() > 1024 * 1024 {
            return Err(Qwen35Error::invalid(
                "vision_config",
                "config exceeds 1 MiB",
            ));
        }
        validate_strict_json(bytes)?;
        let root: Value =
            serde_json::from_slice(bytes).map_err(|e| Qwen35Error::Json(e.to_string()))?;
        if root.get("model_type") != Some(&json!("qwen3_5"))
            || root
                .pointer("/text_config/hidden_size")
                .and_then(Value::as_u64)
                != Some(5120)
        {
            return Err(Qwen35Error::invalid(
                "vision_config",
                "target architecture/output dimension mismatch",
            ));
        }
        let vision = root
            .get("vision_config")
            .and_then(Value::as_object)
            .ok_or_else(|| Qwen35Error::invalid("vision_config", "missing object"))?;
        let expected = fixture_config();
        let expected = expected["vision_config"].as_object().unwrap();
        if vision.keys().collect::<BTreeSet<_>>() != expected.keys().collect::<BTreeSet<_>>() {
            return Err(Qwen35Error::invalid(
                "vision_config",
                "unexpected or missing config fields",
            ));
        }
        for (key, value) in expected {
            // Initializer range only affects randomly initialized models, not
            // this pretrained-weight mapping; nevertheless pin this profile.
            // Augur serializes the parent model_type in the nested Vision
            // config. Both fixed aliases use the identical 333-tensor geometry;
            // all dimensions, activations and storage fields remain pinned.
            let known_alias = key == "model_type"
                && matches!(vision.get(key).and_then(Value::as_str), Some("qwen3_5" | "qwen3_5_vision"));
            if !known_alias && vision.get(key) != Some(value) {
                return Err(Qwen35Error::invalid(
                    format!("vision_config.{key}"),
                    "fixed Vision profile mismatch",
                ));
            }
        }
        Ok(Self {
            config_sha256: format!("{:x}", Sha256::digest(bytes)),
        })
    }

    pub fn config_sha256(&self) -> &str {
        &self.config_sha256
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Qwen35VisionTensorBinding {
    pub name: String,
    pub stage: String,
    pub block: Option<usize>,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub file_data_offsets: [u64; 2],
    /// Offset in the existing loader's file-offset-sorted packed selection.
    pub resident_offset: u64,
    pub byte_len: u64,
}

/// Immutable metadata plan; not a payload integrity or CUDA-lifetime proof.
#[derive(Debug, Clone, Serialize)]
pub struct Qwen35VisionWeightMap {
    config_sha256: String,
    bindings: Vec<Qwen35VisionTensorBinding>,
    bytes_by_stage: BTreeMap<String, u64>,
    selected_payload_bytes: u64,
}

impl Qwen35VisionWeightMap {
    pub fn from_manifest(
        profile: &Qwen35VisionProfile,
        manifest: &SafetensorsManifest,
    ) -> Result<Self, Qwen35Error> {
        Self::from_records(profile, manifest.tensors())
    }

    pub fn bindings(&self) -> &[Qwen35VisionTensorBinding] {
        &self.bindings
    }
    pub fn bytes_by_stage(&self) -> &BTreeMap<String, u64> {
        &self.bytes_by_stage
    }
    pub fn selected_payload_bytes(&self) -> u64 {
        self.selected_payload_bytes
    }
    pub fn tensor(&self, name: &str) -> Option<&Qwen35VisionTensorBinding> {
        self.bindings.iter().find(|binding| binding.name == name)
    }

    fn from_records<'a>(
        profile: &Qwen35VisionProfile,
        records: impl Iterator<Item = &'a SafetensorsRecord>,
    ) -> Result<Self, Qwen35Error> {
        let specs = expected_specs();
        let mut actual = BTreeMap::new();
        for record in records.filter(|r| r.name.starts_with(PREFIX)) {
            if actual.insert(record.name.as_str(), record).is_some() {
                return Err(Qwen35Error::invalid("vision_tensor", "duplicate name"));
            }
        }
        if actual.len() != specs.len() {
            return Err(Qwen35Error::invalid(
                "vision_tensor_count",
                "expected exactly 333 Vision tensors",
            ));
        }
        let mut bindings = Vec::with_capacity(specs.len());
        for (name, (shape, stage, block)) in specs {
            let record = actual
                .get(name.as_str())
                .ok_or_else(|| Qwen35Error::MissingField(name.clone()))?;
            if record.dtype != "BF16"
                || record.shape != shape
                || record.byte_len()?
                    != record
                        .element_count()?
                        .checked_mul(2)
                        .ok_or(Qwen35Error::SizeOverflow)?
            {
                return Err(Qwen35Error::invalid(
                    name,
                    "Vision BF16 tensor shape/extent mismatch",
                ));
            }
            bindings.push(Qwen35VisionTensorBinding {
                name,
                stage: stage.into(),
                block,
                dtype: "BF16".into(),
                shape,
                file_data_offsets: record.data_offsets,
                resident_offset: 0,
                byte_len: record.byte_len()?,
            });
        }
        bindings.sort_by_key(|b| b.file_data_offsets[0]);
        for pair in bindings.windows(2) {
            if pair[0].file_data_offsets[1] > pair[1].file_data_offsets[0] {
                return Err(Qwen35Error::invalid(
                    "vision_ranges",
                    "selected tensors overlap",
                ));
            }
        }
        let mut selected_payload_bytes = 0u64;
        let mut bytes_by_stage = BTreeMap::new();
        for binding in &mut bindings {
            binding.resident_offset = selected_payload_bytes;
            selected_payload_bytes = selected_payload_bytes
                .checked_add(binding.byte_len)
                .ok_or(Qwen35Error::SizeOverflow)?;
            *bytes_by_stage.entry(binding.stage.clone()).or_insert(0) += binding.byte_len;
        }
        if selected_payload_bytes != PAYLOAD_BYTES {
            return Err(Qwen35Error::invalid(
                "vision_bytes",
                "fixed payload size mismatch",
            ));
        }
        Ok(Self {
            config_sha256: profile.config_sha256.clone(),
            bindings,
            bytes_by_stage,
            selected_payload_bytes,
        })
    }
}

type Specs = BTreeMap<String, (Vec<u64>, &'static str, Option<usize>)>;

fn expected_specs() -> Specs {
    let mut out = BTreeMap::new();
    let mut add = |suffix: String, shape: Vec<u64>, stage, block| {
        out.insert(format!("{PREFIX}{suffix}"), (shape, stage, block));
    };
    add(
        "patch_embed.proj.weight".into(),
        vec![1152, 3, 2, 16, 16],
        "patch_embed",
        None,
    );
    add(
        "patch_embed.proj.bias".into(),
        vec![1152],
        "patch_embed",
        None,
    );
    add(
        "pos_embed.weight".into(),
        vec![2304, 1152],
        "pos_embed",
        None,
    );
    for layer in 0..27 {
        for (name, rows, cols) in [
            ("attn.qkv", 3456, 1152),
            ("attn.proj", 1152, 1152),
            ("mlp.linear_fc1", 4304, 1152),
            ("mlp.linear_fc2", 1152, 4304),
        ] {
            add(
                format!("blocks.{layer}.{name}.weight"),
                vec![rows, cols],
                "blocks",
                Some(layer),
            );
            add(
                format!("blocks.{layer}.{name}.bias"),
                vec![rows],
                "blocks",
                Some(layer),
            );
        }
        for norm in ["norm1", "norm2"] {
            for kind in ["weight", "bias"] {
                add(
                    format!("blocks.{layer}.{norm}.{kind}"),
                    vec![1152],
                    "blocks",
                    Some(layer),
                );
            }
        }
    }
    for (name, rows, cols) in [("linear_fc1", 4608, 4608), ("linear_fc2", 5120, 4608)] {
        add(
            format!("merger.{name}.weight"),
            vec![rows, cols],
            "merger",
            None,
        );
        add(format!("merger.{name}.bias"), vec![rows], "merger", None);
    }
    for kind in ["weight", "bias"] {
        add(format!("merger.norm.{kind}"), vec![1152], "merger", None);
    }
    out
}

fn fixture_config() -> Value {
    json!({"model_type":"qwen3_5","text_config":{"hidden_size":5120},"vision_config":{
        "deepstack_visual_indexes":[],"depth":27,"dtype":"bfloat16","hidden_act":"gelu_pytorch_tanh",
        "hidden_size":1152,"in_channels":3,"initializer_range":0.02,"intermediate_size":4304,
        "model_type":"qwen3_5_vision","num_heads":16,"num_position_embeddings":2304,
        "out_hidden_size":5120,"patch_size":16,"spatial_merge_size":2,"temporal_patch_size":2}})
}

#[cfg(test)]
mod tests {
    use super::*;
    fn profile() -> Qwen35VisionProfile {
        Qwen35VisionProfile::from_config_bytes(&serde_json::to_vec(&fixture_config()).unwrap())
            .unwrap()
    }
    #[test]
    fn augur_nested_model_type_alias_retains_fixed_architecture() {
        let mut config = fixture_config();
        config["vision_config"]["model_type"] = json!("qwen3_5");
        assert!(Qwen35VisionProfile::from_config_bytes(&serde_json::to_vec(&config).unwrap()).is_ok());
        config["vision_config"]["hidden_size"] = json!(1024);
        assert!(Qwen35VisionProfile::from_config_bytes(&serde_json::to_vec(&config).unwrap()).is_err());
        config = fixture_config();config["vision_config"]["model_type"] = json!("qwen_future");
        assert!(Qwen35VisionProfile::from_config_bytes(&serde_json::to_vec(&config).unwrap()).is_err());
    }
    fn records() -> Vec<SafetensorsRecord> {
        let mut cursor = 0u64;
        expected_specs()
            .into_iter()
            .map(|(name, (shape, _, _))| {
                let n = shape.iter().product::<u64>() * 2;
                let record = SafetensorsRecord {
                    name,
                    dtype: "BF16".into(),
                    shape,
                    data_offsets: [cursor, cursor + n],
                };
                cursor += n;
                record
            })
            .collect()
    }
    #[test]
    fn exact_geometry_closes_storage_ledger_and_physical_order() {
        let mut records = records();
        records.reverse();
        let map = Qwen35VisionWeightMap::from_records(&profile(), records.iter()).unwrap();
        assert_eq!(map.bindings().len(), 333);
        assert_eq!(map.selected_payload_bytes(), PAYLOAD_BYTES);
        assert_eq!(map.bytes_by_stage()["blocks"], 822_933_216);
        assert_eq!(map.bytes_by_stage()["merger"], 89_677_312);
        assert_eq!(map.bytes_by_stage()["patch_embed"], 3_541_248);
        assert_eq!(map.bytes_by_stage()["pos_embed"], 5_308_416);
        for pair in map.bindings().windows(2) {
            assert_eq!(
                pair[0].resident_offset + pair[0].byte_len,
                pair[1].resident_offset
            );
        }
        assert_eq!(
            map.tensor("model.visual.patch_embed.proj.weight")
                .unwrap()
                .shape,
            vec![1152, 3, 2, 16, 16]
        );
    }
    #[test]
    fn config_geometry_activation_and_unknown_fields_reject() {
        for (key, value) in [
            ("depth", json!(26)),
            ("hidden_act", json!("gelu")),
            ("dtype", json!("float16")),
            ("num_heads", json!(8)),
            ("deepstack_visual_indexes", json!([1])),
            ("new_mode", json!(true)),
        ] {
            let mut config = fixture_config();
            config["vision_config"][key] = value;
            assert!(
                Qwen35VisionProfile::from_config_bytes(&serde_json::to_vec(&config).unwrap())
                    .is_err()
            );
        }
        let mut config = fixture_config();
        config["text_config"]["hidden_size"] = json!(4096);
        assert!(
            Qwen35VisionProfile::from_config_bytes(&serde_json::to_vec(&config).unwrap()).is_err()
        );
        assert!(Qwen35VisionProfile::from_config_bytes(
            br#"{"model_type":"qwen3_5","model_type":"qwen3_5"}"#
        )
        .is_err());
    }
    #[test]
    fn missing_extra_duplicate_and_renamed_visual_tensors_reject() {
        let profile = profile();
        let good = records();
        let mut bad = good.clone();
        bad.pop();
        assert!(Qwen35VisionWeightMap::from_records(&profile, bad.iter()).is_err());
        let mut bad = good.clone();
        bad.push(good[0].clone());
        assert!(Qwen35VisionWeightMap::from_records(&profile, bad.iter()).is_err());
        let mut bad = good.clone();
        bad[0].name = "model.visual.unknown.weight".into();
        assert!(Qwen35VisionWeightMap::from_records(&profile, bad.iter()).is_err());
        let mut bad = good.clone();
        let mut extra = good[0].clone();
        extra.name = "model.visual.extra.weight".into();
        bad.push(extra);
        assert!(Qwen35VisionWeightMap::from_records(&profile, bad.iter()).is_err());
    }
    #[test]
    fn dtype_shape_extent_and_overlap_drift_reject() {
        let profile = profile();
        let good = records();
        for mutation in 0..5 {
            let mut bad = good.clone();
            match mutation {
                0 => bad[0].dtype = "F16".into(),
                1 => bad[0].shape.push(1),
                2 => bad[0].data_offsets[1] += 2,
                3 => {
                    let size = bad[1].byte_len().unwrap();
                    bad[1].data_offsets = [bad[0].data_offsets[0], bad[0].data_offsets[0] + size];
                }
                _ => bad[0].data_offsets.swap(0, 1),
            }
            assert!(Qwen35VisionWeightMap::from_records(&profile, bad.iter()).is_err());
        }
    }
    #[test]
    fn nonvisual_namespace_is_not_admitted_as_vision() {
        let mut records = records();
        let mut other = records[0].clone();
        other.name = "model.language_model.embed_tokens.weight".into();
        records.push(other);
        assert_eq!(
            Qwen35VisionWeightMap::from_records(&profile(), records.iter())
                .unwrap()
                .bindings()
                .len(),
            333
        );
    }
}
