//! Exact tensor inventory binding for the Qwen35 DFlash2 draft backbone.
//!
//! The loader and the CUDA producer deliberately use different layers of
//! responsibility. [`DFlashBackboneManifest`] only proves that one
//! identity-bound safetensors resident contains the complete five-layer
//! DFlash tensor set with the geometry described by [`DFlashConfig`].  It does
//! not execute a layer, produce logits, or authorize a decode claim.  Keeping
//! this binding separate gives the future producer a single source of truth
//! for tensor names, shapes, and payload offsets.

use crate::{DFlashDType, DFlashRamResident};
use augur_model_qwen35::DFlashConfig;
use serde_json::Value;
use std::{collections::BTreeMap, fmt};

pub const DFLASH_BACKBONE_MANIFEST_SCHEMA: &str = "qwen35-dflash2-complete-backbone-manifest-v1";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DFlashTensorBinding {
    pub name: String,
    pub shape: Vec<usize>,
    pub offset_bytes: (u64, u64),
    pub dtype: DFlashDType,
}

impl DFlashTensorBinding {
    pub fn byte_len(&self) -> u64 {
        self.offset_bytes.1 - self.offset_bytes.0
    }
}

#[derive(Clone, Debug)]
pub struct DFlashBackboneManifest {
    resident: DFlashRamResident,
    config: DFlashConfig,
    tensors: Vec<DFlashTensorBinding>,
}

impl DFlashBackboneManifest {
    /// Bind all 81 tensors of the exact Qwen35 DFlash2 profile.
    pub fn bind(
        resident: DFlashRamResident,
        config: DFlashConfig,
    ) -> Result<Self, DFlashBackboneError> {
        config
            .validate_exact_profile()
            .map_err(|error| DFlashBackboneError::Invalid(error.to_string()))?;
        let expected_dtype = dtype_for_config(&config)?;
        if resident.inspection().artifact.dtype != expected_dtype {
            return Err(DFlashBackboneError::Invalid(format!(
                "artifact dtype {:?} differs from config dtype {:?}",
                resident.inspection().artifact.dtype,
                expected_dtype
            )));
        }
        let header_len = usize::try_from(resident.inspection().header_bytes)
            .map_err(|_| DFlashBackboneError::Invalid("header extent".into()))?;
        let header_end = 8usize
            .checked_add(header_len)
            .ok_or_else(|| DFlashBackboneError::Invalid("header extent overflow".into()))?;
        let header = resident
            .file_bytes()
            .get(8..header_end)
            .ok_or_else(|| DFlashBackboneError::Invalid("header extent".into()))?;
        let root: Value = serde_json::from_slice(header)
            .map_err(|error| DFlashBackboneError::Invalid(format!("header JSON: {error}")))?;
        let object = root
            .as_object()
            .ok_or_else(|| DFlashBackboneError::Invalid("header root is not an object".into()))?;
        let expected = expected_tensor_specs(&config);
        let actual_names = object
            .keys()
            .filter(|name| name.as_str() != "__metadata__")
            .cloned()
            .collect::<std::collections::BTreeSet<_>>();
        let expected_names = expected
            .keys()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>();
        if actual_names != expected_names {
            let missing = expected_names
                .difference(&actual_names)
                .cloned()
                .collect::<Vec<_>>();
            let unknown = actual_names
                .difference(&expected_names)
                .cloned()
                .collect::<Vec<_>>();
            return Err(DFlashBackboneError::Invalid(format!(
                "tensor set mismatch: missing={missing:?}, unknown={unknown:?}"
            )));
        }

        let payload_len = resident.payload().len() as u64;
        let dtype_name = dtype_name(expected_dtype);
        let mut tensors = Vec::with_capacity(expected.len());
        for (name, expected_shape) in expected {
            let tensor = object
                .get(&name)
                .and_then(Value::as_object)
                .ok_or_else(|| DFlashBackboneError::Invalid(format!("tensor {name} object")))?;
            if tensor.get("dtype").and_then(Value::as_str) != Some(dtype_name) {
                return Err(DFlashBackboneError::Invalid(format!(
                    "tensor {name} dtype differs from {dtype_name}"
                )));
            }
            let shape = tensor
                .get("shape")
                .and_then(Value::as_array)
                .ok_or_else(|| DFlashBackboneError::Invalid(format!("tensor {name} shape")))?
                .iter()
                .map(|value| {
                    value
                        .as_u64()
                        .and_then(|value| usize::try_from(value).ok())
                        .ok_or_else(|| {
                            DFlashBackboneError::Invalid(format!("tensor {name} shape value"))
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            if shape != expected_shape {
                return Err(DFlashBackboneError::Invalid(format!(
                    "tensor {name} shape mismatch: expected {expected_shape:?}, got {shape:?}"
                )));
            }
            let offsets = tensor
                .get("data_offsets")
                .and_then(Value::as_array)
                .filter(|offsets| offsets.len() == 2)
                .ok_or_else(|| DFlashBackboneError::Invalid(format!("tensor {name} offsets")))?;
            let start = offsets[0]
                .as_u64()
                .ok_or_else(|| DFlashBackboneError::Invalid(format!("tensor {name} start")))?;
            let end = offsets[1]
                .as_u64()
                .ok_or_else(|| DFlashBackboneError::Invalid(format!("tensor {name} end")))?;
            let elements = shape
                .iter()
                .try_fold(1_u64, |acc, value| acc.checked_mul(*value as u64))
                .ok_or_else(|| {
                    DFlashBackboneError::Invalid(format!("tensor {name} size overflow"))
                })?;
            let expected_bytes = elements.checked_mul(2).ok_or_else(|| {
                DFlashBackboneError::Invalid(format!("tensor {name} byte overflow"))
            })?;
            if start >= end || end - start != expected_bytes || end > payload_len {
                return Err(DFlashBackboneError::Invalid(format!(
                    "tensor {name} extent mismatch"
                )));
            }
            tensors.push(DFlashTensorBinding {
                name,
                shape,
                offset_bytes: (start, end),
                dtype: expected_dtype,
            });
        }
        tensors.sort_by_key(|tensor| tensor.offset_bytes.0);
        let mut cursor = 0_u64;
        for tensor in &tensors {
            if tensor.offset_bytes.0 != cursor {
                return Err(DFlashBackboneError::Invalid(format!(
                    "tensor {} leaves a payload gap or overlaps",
                    tensor.name
                )));
            }
            cursor = tensor.offset_bytes.1;
        }
        if cursor != payload_len {
            return Err(DFlashBackboneError::Invalid(format!(
                "tensor coverage ends at {cursor}, payload is {payload_len}"
            )));
        }
        Ok(Self {
            resident,
            config,
            tensors,
        })
    }

    pub fn resident(&self) -> &DFlashRamResident {
        &self.resident
    }

    pub fn config(&self) -> &DFlashConfig {
        &self.config
    }

    pub fn tensors(&self) -> &[DFlashTensorBinding] {
        &self.tensors
    }

    pub fn tensor(&self, name: &str) -> Option<&DFlashTensorBinding> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }

    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8], DFlashBackboneError> {
        let tensor = self
            .tensor(name)
            .ok_or_else(|| DFlashBackboneError::Invalid(format!("unknown tensor {name}")))?;
        let (start, end) = tensor.offset_bytes;
        let start = usize::try_from(start)
            .map_err(|_| DFlashBackboneError::Invalid(format!("tensor {name} start extent")))?;
        let end = usize::try_from(end)
            .map_err(|_| DFlashBackboneError::Invalid(format!("tensor {name} end extent")))?;
        self.resident
            .payload()
            .get(start..end)
            .ok_or_else(|| DFlashBackboneError::Invalid(format!("tensor {name} payload extent")))
    }

    pub fn layer_tensors(&self, layer: usize) -> impl Iterator<Item = &DFlashTensorBinding> {
        let prefix = format!("layers.{layer}.");
        self.tensors
            .iter()
            .filter(move |tensor| tensor.name.starts_with(&prefix))
    }
}

#[derive(Debug)]
pub enum DFlashBackboneError {
    Invalid(String),
}

impl fmt::Display for DFlashBackboneError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(error) => write!(formatter, "invalid DFlash backbone manifest: {error}"),
        }
    }
}

impl std::error::Error for DFlashBackboneError {}

fn dtype_for_config(config: &DFlashConfig) -> Result<DFlashDType, DFlashBackboneError> {
    match config.dtype.as_str() {
        "float16" => Ok(DFlashDType::F16),
        "bfloat16" => Ok(DFlashDType::Bf16),
        other => Err(DFlashBackboneError::Invalid(format!(
            "unsupported config dtype {other}"
        ))),
    }
}

fn dtype_name(dtype: DFlashDType) -> &'static str {
    match dtype {
        DFlashDType::F16 => "F16",
        DFlashDType::Bf16 => "BF16",
    }
}

pub(crate) fn expected_tensor_specs(config: &DFlashConfig) -> BTreeMap<String, Vec<usize>> {
    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;
    let attention = config.attention_heads * config.head_dim;
    let kv = config.kv_heads * config.head_dim;
    let dynamic = 2 * config.conv_kernel_size * hidden / config.conv_group_size;
    let mut specs = BTreeMap::new();
    specs.insert(
        "candidate_selector.hidden_projection.weight".into(),
        vec![config.selector_rank, hidden],
    );
    specs.insert(
        "candidate_selector.predecessor_codebook".into(),
        vec![config.vocab_size, config.selector_rank],
    );
    specs.insert(
        "candidate_selector.successor_codebook".into(),
        vec![config.vocab_size, config.selector_rank],
    );
    specs.insert("fc.weight".into(), vec![hidden, 5 * hidden]);
    specs.insert("hidden_norm.weight".into(), vec![hidden]);
    specs.insert("norm.weight".into(), vec![hidden]);
    for layer in 0..config.num_hidden_layers {
        let prefix = format!("layers.{layer}");
        specs.insert(
            format!("{prefix}.attention_conv.base_kernel"),
            vec![2, config.conv_kernel_size, hidden],
        );
        specs.insert(
            format!("{prefix}.attention_conv.kernel_projection.weight"),
            vec![dynamic, hidden],
        );
        specs.insert(format!("{prefix}.input_layernorm.weight"), vec![hidden]);
        specs.insert(
            format!("{prefix}.mlp.down_proj.weight"),
            vec![hidden, intermediate],
        );
        specs.insert(
            format!("{prefix}.mlp.gate_proj.weight"),
            vec![intermediate, hidden],
        );
        specs.insert(
            format!("{prefix}.mlp.up_proj.weight"),
            vec![intermediate, hidden],
        );
        specs.insert(
            format!("{prefix}.mlp_conv.base_kernel"),
            vec![2, config.conv_kernel_size, hidden],
        );
        specs.insert(
            format!("{prefix}.mlp_conv.kernel_projection.weight"),
            vec![dynamic, hidden],
        );
        specs.insert(
            format!("{prefix}.post_attention_layernorm.weight"),
            vec![hidden],
        );
        specs.insert(
            format!("{prefix}.self_attn.k_norm.weight"),
            vec![config.head_dim],
        );
        specs.insert(
            format!("{prefix}.self_attn.k_proj.weight"),
            vec![kv, hidden],
        );
        specs.insert(
            format!("{prefix}.self_attn.o_proj.weight"),
            vec![hidden, attention],
        );
        specs.insert(
            format!("{prefix}.self_attn.q_norm.weight"),
            vec![config.head_dim],
        );
        specs.insert(
            format!("{prefix}.self_attn.q_proj.weight"),
            vec![attention, hidden],
        );
        specs.insert(
            format!("{prefix}.self_attn.v_proj.weight"),
            vec![kv, hidden],
        );
    }
    specs
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn config() -> DFlashConfig {
        DFlashConfig {
            architecture: "DFlash2DraftModel".into(),
            model_type: "qwen3".into(),
            dtype: "float16".into(),
            hidden_size: 5120,
            intermediate_size: 17408,
            num_hidden_layers: 5,
            num_target_layers: 64,
            attention_heads: 32,
            kv_heads: 8,
            head_dim: 128,
            max_position_embeddings: 262_144,
            vocab_size: 248_320,
            block_size: 8,
            conv_group_size: 16,
            conv_kernel_size: 2,
            selector_rank: 256,
            selector_top_k: 16,
            target_layer_ids: vec![5, 19, 33, 47, 61],
        }
    }

    #[test]
    fn exact_profile_has_81_tensors_and_five_layers() {
        let specs = expected_tensor_specs(&config());
        assert_eq!(specs.len(), 81);
        assert_eq!(
            specs
                .keys()
                .filter(|name| name.starts_with("layers.0."))
                .count(),
            15
        );
        assert_eq!(
            specs
                .keys()
                .filter(|name| name.starts_with("layers.4."))
                .count(),
            15
        );
    }

    #[test]
    fn unknown_root_tensor_is_not_part_of_exact_set() {
        let specs = expected_tensor_specs(&config());
        let mut root = serde_json::Map::new();
        root.insert(
            "unknown.weight".into(),
            json!({"dtype":"F16","shape":[1],"data_offsets":[0,2]}),
        );
        let actual = root
            .keys()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>();
        let expected = specs
            .keys()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>();
        assert_ne!(actual, expected);
    }
}
