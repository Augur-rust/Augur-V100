use crate::config::Qwen35Config;
use crate::error::Qwen35Error;
use crate::gguf::{GgufManifest, GgufTensor};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

/// Summary returned after validating the converter's Qwen3.8 tensor ABI.
///
/// The summary is intentionally descriptive only.  It does not imply that a
/// tensor payload was decoded or that a runtime is permitted to execute it.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TensorSchemaSummary {
    pub expected_tensor_count: usize,
    pub actual_tensor_count: usize,
    pub optional_auxiliary_tensor_count: usize,
    pub quantization_counts: BTreeMap<u32, usize>,
    pub linear_attention_layer_count: usize,
    pub full_attention_layer_count: usize,
}

/// Strict tensor-name, shape, and GGML-type contract for the audited Qwen
/// mixed NVFP4/Q8 GGUF conversion.
pub struct Qwen35TensorSchema;

impl Qwen35TensorSchema {
    /// Validate all tensor entries in a parsed GGUF manifest.
    pub fn validate_manifest(
        config: &Qwen35Config,
        manifest: &GgufManifest,
    ) -> Result<TensorSchemaSummary, Qwen35Error> {
        Self::validate_tensors(config, &manifest.tensors)
    }

    /// Validate the target model while allowing the exact 15-tensor Qwen MTP
    /// block emitted by the combined product artifact.  The default strict
    /// validator remains unchanged; callers must opt into this method and the
    /// auxiliary block is accepted only when every name, shape, and GGML type
    /// matches the audited profile.
    pub fn validate_manifest_allowing_mtp(
        config: &Qwen35Config,
        manifest: &GgufManifest,
    ) -> Result<TensorSchemaSummary, Qwen35Error> {
        Self::validate_tensors_allowing_mtp(config, &manifest.tensors)
    }

    /// Validate a tensor list without requiring a complete GGUF object.
    ///
    /// This is useful for unit tests and for loaders that have already
    /// authenticated the GGUF header through another transport.
    pub fn validate_tensors(
        config: &Qwen35Config,
        tensors: &[GgufTensor],
    ) -> Result<TensorSchemaSummary, Qwen35Error> {
        Self::validate_tensors_with_options(config, tensors, false)
    }

    /// Validate a tensor list with the exact optional MTP block enabled.
    pub fn validate_tensors_allowing_mtp(
        config: &Qwen35Config,
        tensors: &[GgufTensor],
    ) -> Result<TensorSchemaSummary, Qwen35Error> {
        Self::validate_tensors_with_options(config, tensors, true)
    }

    fn validate_tensors_with_options(
        config: &Qwen35Config,
        tensors: &[GgufTensor],
        allow_mtp: bool,
    ) -> Result<TensorSchemaSummary, Qwen35Error> {
        config.validate_exact_profile()?;
        if config.quantization.source_profile != crate::Qwen35CheckpointProfile::OriginalMixed {
            return Err(Qwen35Error::invalid("gguf.profile", "new checkpoints require their explicit tensor loader"));
        }
        let expected = expected_specs(config);
        let optional_specs = if allow_mtp {
            mtp_specs(config)
        } else {
            Vec::new()
        };
        if (!allow_mtp && tensors.len() != expected.len())
            || (allow_mtp
                && tensors.len() != expected.len()
                && tensors.len() != expected.len() + optional_specs.len())
        {
            return Err(Qwen35Error::invalid(
                "gguf.tensor_count",
                format!(
                    "expected {} tensors for the audited profile{}, got {}",
                    expected.len(),
                    if allow_mtp {
                        format!(
                            " or {} with the exact optional MTP block",
                            expected.len() + optional_specs.len()
                        )
                    } else {
                        String::new()
                    },
                    tensors.len()
                ),
            ));
        }

        let mut actual_names = BTreeSet::new();
        let mut actual_by_name = BTreeMap::new();
        for tensor in tensors {
            if !actual_names.insert(tensor.name.clone()) {
                return Err(Qwen35Error::Gguf(format!(
                    "duplicate tensor name {:?}",
                    tensor.name
                )));
            }
            actual_by_name.insert(tensor.name.as_str(), tensor);
        }

        for spec in &expected {
            let tensor = actual_by_name.get(spec.name.as_str()).ok_or_else(|| {
                Qwen35Error::Gguf(format!("missing required tensor {:?}", spec.name))
            })?;
            if tensor.dimensions != spec.dimensions {
                return Err(Qwen35Error::invalid(
                    format!("tensor {} dimensions", spec.name),
                    format!(
                        "expected {:?}, got {:?}",
                        spec.dimensions, tensor.dimensions
                    ),
                ));
            }
            if tensor.ggml_type != spec.ggml_type {
                return Err(Qwen35Error::invalid(
                    format!("tensor {} ggml_type", spec.name),
                    format!("expected {}, got {}", spec.ggml_type, tensor.ggml_type),
                ));
            }
        }

        let expected_names: BTreeSet<String> =
            expected.iter().map(|spec| spec.name.clone()).collect();
        let optional_by_name: BTreeMap<&str, &TensorSpec> = optional_specs
            .iter()
            .map(|spec| (spec.name.as_str(), spec))
            .collect();
        let mut optional_auxiliary_tensor_count = 0;
        for name in actual_names.difference(&expected_names) {
            let Some(spec) = optional_by_name.get(name.as_str()) else {
                return Err(Qwen35Error::Gguf(format!(
                    "unexpected tensor {:?} in Qwen profile",
                    name
                )));
            };
            let tensor = actual_by_name
                .get(name.as_str())
                .expect("optional tensor is present in actual names");
            if tensor.dimensions != spec.dimensions {
                return Err(Qwen35Error::invalid(
                    format!("tensor {} dimensions", spec.name),
                    format!(
                        "expected {:?}, got {:?}",
                        spec.dimensions, tensor.dimensions
                    ),
                ));
            }
            if tensor.ggml_type != spec.ggml_type {
                return Err(Qwen35Error::invalid(
                    format!("tensor {} ggml_type", spec.name),
                    format!("expected {}, got {}", spec.ggml_type, tensor.ggml_type),
                ));
            }
            optional_auxiliary_tensor_count += 1;
        }
        if allow_mtp
            && optional_auxiliary_tensor_count != 0
            && optional_auxiliary_tensor_count != optional_specs.len()
        {
            return Err(Qwen35Error::Gguf(format!(
                "optional MTP block is incomplete: expected {}, got {}",
                optional_specs.len(),
                optional_auxiliary_tensor_count
            )));
        }

        let mut quantization_counts = BTreeMap::new();
        for tensor in tensors {
            *quantization_counts.entry(tensor.ggml_type).or_insert(0) += 1;
        }
        Ok(TensorSchemaSummary {
            expected_tensor_count: expected.len(),
            actual_tensor_count: tensors.len(),
            optional_auxiliary_tensor_count,
            quantization_counts,
            linear_attention_layer_count: config.linear_attention_layers.len(),
            full_attention_layer_count: config.full_attention_layers.len(),
        })
    }

    /// Return the authenticated GGML type used by all three FFN weight
    /// tensors in one layer.
    ///
    /// The converted artifact keeps layers `0..56` in NVFP4 and the final
    /// eight layers (`56..64`) in Q8_0.  Loaders must use this helper instead
    /// of assuming that every FFN projection has the group-1 type; the final
    /// Q8_0 layers intentionally have no `.scale` or `.input_scale` tensors.
    pub fn expected_ffn_weight_ggml_type(
        config: &Qwen35Config,
        layer: usize,
    ) -> Result<u32, Qwen35Error> {
        config.validate_exact_profile()?;
        if config.quantization.source_profile != crate::Qwen35CheckpointProfile::OriginalMixed {
            return Err(Qwen35Error::invalid("gguf.profile", "legacy FFN type routing does not apply to new checkpoints"));
        }
        if layer >= config.num_hidden_layers {
            return Err(Qwen35Error::invalid(
                "layer.index",
                format!(
                    "layer {layer} is outside {} layers",
                    config.num_hidden_layers
                ),
            ));
        }
        Ok(ffn_type(layer, config))
    }
}

#[derive(Clone, Debug)]
struct TensorSpec {
    name: String,
    dimensions: Vec<u64>,
    ggml_type: u32,
}

fn expected_specs(config: &Qwen35Config) -> Vec<TensorSpec> {
    let mut specs = Vec::with_capacity(1187);
    specs.push(spec(
        "token_embd.weight",
        [config.hidden_size as u64, config.vocab_size as u64],
        30,
    ));
    specs.push(spec("output_norm.weight", [config.hidden_size as u64], 0));
    specs.push(spec(
        "output.weight",
        [config.hidden_size as u64, config.vocab_size as u64],
        8,
    ));

    for layer in 0..config.num_hidden_layers {
        let prefix = format!("blk.{layer}");
        specs.push(spec(
            format!("{prefix}.ffn_down.weight"),
            [config.intermediate_size as u64, config.hidden_size as u64],
            ffn_type(layer, config),
        ));
        specs.push(spec(
            format!("{prefix}.ffn_gate.weight"),
            [config.hidden_size as u64, config.intermediate_size as u64],
            ffn_type(layer, config),
        ));
        specs.push(spec(
            format!("{prefix}.ffn_up.weight"),
            [config.hidden_size as u64, config.intermediate_size as u64],
            ffn_type(layer, config),
        ));
        if ffn_type(layer, config) == 40 {
            for projection in ["down", "gate", "up"] {
                specs.push(spec(format!("{prefix}.ffn_{projection}.scale"), [1], 0));
                specs.push(spec(
                    format!("{prefix}.ffn_{projection}.input_scale"),
                    [1],
                    0,
                ));
            }
        }

        specs.push(spec(
            format!("{prefix}.attn_norm.weight"),
            [config.hidden_size as u64],
            0,
        ));
        if config.linear_attention_layers.contains(&layer) {
            specs.push(spec(
                format!("{prefix}.ssm_a"),
                [config.linear_num_value_heads as u64],
                0,
            ));
            specs.push(spec(
                format!("{prefix}.ssm_conv1d.weight"),
                [
                    config.linear_conv_kernel_dim as u64,
                    (config.linear_num_value_heads * config.linear_value_head_dim
                        + 2 * config.linear_num_key_heads * config.linear_key_head_dim)
                        as u64,
                ],
                0,
            ));
            specs.push(spec(
                format!("{prefix}.ssm_dt.bias"),
                [config.linear_num_value_heads as u64],
                0,
            ));
            specs.push(spec(
                format!("{prefix}.ssm_alpha.weight"),
                [
                    config.hidden_size as u64,
                    config.linear_num_value_heads as u64,
                ],
                30,
            ));
            specs.push(spec(
                format!("{prefix}.ssm_beta.weight"),
                [
                    config.hidden_size as u64,
                    config.linear_num_value_heads as u64,
                ],
                30,
            ));
            specs.push(spec(
                format!("{prefix}.ssm_norm.weight"),
                [config.linear_value_head_dim as u64],
                0,
            ));
            specs.push(spec(
                format!("{prefix}.post_attention_norm.weight"),
                [config.hidden_size as u64],
                0,
            ));
            specs.push(spec(
                format!("{prefix}.attn_qkv.weight"),
                [config.hidden_size as u64, (2 * config.hidden_size) as u64],
                8,
            ));
            specs.push(spec(
                format!("{prefix}.attn_gate.weight"),
                [
                    config.hidden_size as u64,
                    (config.linear_num_value_heads * config.linear_value_head_dim) as u64,
                ],
                8,
            ));
            specs.push(spec(
                format!("{prefix}.ssm_out.weight"),
                [
                    (config.linear_num_value_heads * config.linear_value_head_dim) as u64,
                    config.hidden_size as u64,
                ],
                8,
            ));
        } else {
            specs.push(spec(
                format!("{prefix}.post_attention_norm.weight"),
                [config.hidden_size as u64],
                0,
            ));
            specs.push(spec(
                format!("{prefix}.attn_k_norm.weight"),
                [config.head_dim as u64],
                0,
            ));
            specs.push(spec(
                format!("{prefix}.attn_q_norm.weight"),
                [config.head_dim as u64],
                0,
            ));
            specs.push(spec(
                format!("{prefix}.attn_k.weight"),
                [
                    config.hidden_size as u64,
                    (config.num_key_value_heads * config.head_dim) as u64,
                ],
                8,
            ));
            specs.push(spec(
                format!("{prefix}.attn_q.weight"),
                [
                    config.hidden_size as u64,
                    (2 * config.num_attention_heads * config.head_dim) as u64,
                ],
                8,
            ));
            specs.push(spec(
                format!("{prefix}.attn_v.weight"),
                [
                    config.hidden_size as u64,
                    (config.num_key_value_heads * config.head_dim) as u64,
                ],
                8,
            ));
            specs.push(spec(
                format!("{prefix}.attn_output.weight"),
                [
                    (config.linear_num_value_heads * config.linear_value_head_dim) as u64,
                    config.hidden_size as u64,
                ],
                8,
            ));
        }
    }
    specs
}

fn mtp_specs(config: &Qwen35Config) -> Vec<TensorSpec> {
    let hidden = config.hidden_size as u64;
    let intermediate = config.intermediate_size as u64;
    let q_width = (config.num_attention_heads * config.head_dim) as u64;
    let kv_width = (config.num_key_value_heads * config.head_dim) as u64;
    vec![
        spec("blk.64.nextn.eh_proj.weight", [2 * hidden, hidden], 30),
        spec("blk.64.attn_norm.weight", [hidden], 0),
        spec("blk.64.ffn_down.weight", [intermediate, hidden], 30),
        spec("blk.64.ffn_gate.weight", [hidden, intermediate], 30),
        spec("blk.64.ffn_up.weight", [hidden, intermediate], 30),
        spec("blk.64.post_attention_norm.weight", [hidden], 0),
        spec("blk.64.attn_k_norm.weight", [config.head_dim as u64], 0),
        spec("blk.64.attn_k.weight", [hidden, kv_width], 30),
        spec("blk.64.attn_output.weight", [q_width, hidden], 30),
        spec("blk.64.attn_q_norm.weight", [config.head_dim as u64], 0),
        spec("blk.64.attn_q.weight", [hidden, 2 * q_width], 30),
        spec("blk.64.attn_v.weight", [hidden, kv_width], 30),
        spec("blk.64.nextn.shared_head_norm.weight", [hidden], 0),
        spec("blk.64.nextn.enorm.weight", [hidden], 0),
        spec("blk.64.nextn.hnorm.weight", [hidden], 0),
    ]
}

fn ffn_type(layer: usize, config: &Qwen35Config) -> u32 {
    // The audited converter leaves the final eight FFN layers in Q8_0 while
    // the first 56 use the NVFP4 block type.
    if layer >= config.num_hidden_layers.saturating_sub(8) {
        8
    } else {
        40
    }
}

fn spec<const N: usize>(
    name: impl Into<String>,
    dimensions: [u64; N],
    ggml_type: u32,
) -> TensorSpec {
    TensorSpec {
        name: name.into(),
        dimensions: dimensions.into(),
        ggml_type,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::QuantizationProfile;

    fn profile() -> Qwen35Config {
        let full_attention_layers =
            vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63];
        Qwen35Config {
            architecture: "Qwen3_5ForConditionalGeneration".into(),
            model_type: "qwen3_5".into(),
            dtype: "bfloat16".into(),
            hidden_size: 5120,
            intermediate_size: 17408,
            num_hidden_layers: 64,
            num_attention_heads: 24,
            num_key_value_heads: 4,
            head_dim: 256,
            max_position_embeddings: 262_144,
            vocab_size: 248_320,
            rms_norm_eps: 1e-6,
            full_attention_interval: 4,
            linear_attention_layers: (0..64)
                .filter(|layer| !full_attention_layers.contains(layer))
                .collect(),
            full_attention_layers,
            linear_conv_kernel_dim: 4,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_num_key_heads: 16,
            linear_num_value_heads: 48,
            rope_theta: 10_000_000.0,
            rope_dimension: 64,
            rope_sections: [11, 11, 10],
            bos_token_id: 248_044,
            eos_token_id: 248_044,
            image_token_id: 248_056,
            vision_start_token_id: 248_053,
            vision_end_token_id: 248_054,
            has_vision_config: true,
            quantization: QuantizationProfile {
                source_profile: crate::Qwen35CheckpointProfile::OriginalMixed,
                group0_num_bits: 8,
                group1_num_bits: 4,
                group1_group_size: 16,
                group0_targets: Vec::new(),
                group1_targets: Vec::new(),
                ignored_tensor_count: 1,
            },
        }
    }

    fn tensors_from_expected(config: &Qwen35Config) -> Vec<GgufTensor> {
        expected_specs(config)
            .into_iter()
            .map(|spec| GgufTensor {
                name: spec.name,
                dimensions: spec.dimensions,
                ggml_type: spec.ggml_type,
                relative_offset: 0,
                absolute_offset: 0,
                byte_len: 0,
            })
            .collect()
    }

    #[test]
    fn audited_profile_has_expected_tensor_geometry() {
        let config = profile();
        let tensors = tensors_from_expected(&config);
        let summary = Qwen35TensorSchema::validate_tensors(&config, &tensors).unwrap();
        assert_eq!(summary.expected_tensor_count, 1187);
        assert_eq!(summary.actual_tensor_count, 1187);
        assert_eq!(summary.quantization_counts.get(&40), Some(&168));
        assert_eq!(summary.quantization_counts.get(&8), Some(&233));
        assert_eq!(summary.quantization_counts.get(&30), Some(&97));
        assert_eq!(summary.quantization_counts.get(&0), Some(&689));
        let q = tensors
            .iter()
            .find(|tensor| tensor.name == "blk.11.attn_q.weight")
            .unwrap();
        assert_eq!(q.dimensions, vec![5120, 12288]);
    }

    #[test]
    fn exact_optional_mtp_block_is_accepted_only_when_opted_in() {
        let config = profile();
        let mut tensors = tensors_from_expected(&config);
        tensors.extend(mtp_specs(&config).into_iter().map(|spec| GgufTensor {
            name: spec.name,
            dimensions: spec.dimensions,
            ggml_type: spec.ggml_type,
            relative_offset: 0,
            absolute_offset: 0,
            byte_len: 0,
        }));
        assert!(Qwen35TensorSchema::validate_tensors(&config, &tensors).is_err());
        let summary = Qwen35TensorSchema::validate_tensors_allowing_mtp(&config, &tensors).unwrap();
        assert_eq!(summary.actual_tensor_count, 1202);
        assert_eq!(summary.optional_auxiliary_tensor_count, 15);
    }

    #[test]
    fn wrong_projection_shape_is_rejected() {
        let config = profile();
        let mut tensors = tensors_from_expected(&config);
        let tensor = tensors
            .iter_mut()
            .find(|tensor| tensor.name == "blk.0.attn_qkv.weight")
            .unwrap();
        tensor.dimensions[1] = 1024;
        assert!(Qwen35TensorSchema::validate_tensors(&config, &tensors).is_err());
    }

    #[test]
    fn missing_or_extra_tensor_is_rejected() {
        let config = profile();
        let mut tensors = tensors_from_expected(&config);
        tensors.pop();
        assert!(Qwen35TensorSchema::validate_tensors(&config, &tensors).is_err());

        let mut tensors = tensors_from_expected(&config);
        tensors[0].name = "unexpected.weight".into();
        assert!(Qwen35TensorSchema::validate_tensors(&config, &tensors).is_err());
    }

    #[test]
    fn ffn_weight_type_tracks_final_q8_layers() {
        let config = profile();
        assert_eq!(
            Qwen35TensorSchema::expected_ffn_weight_ggml_type(&config, 0).unwrap(),
            40
        );
        assert_eq!(
            Qwen35TensorSchema::expected_ffn_weight_ggml_type(&config, 55).unwrap(),
            40
        );
        assert_eq!(
            Qwen35TensorSchema::expected_ffn_weight_ggml_type(&config, 56).unwrap(),
            8
        );
        assert_eq!(
            Qwen35TensorSchema::expected_ffn_weight_ggml_type(&config, 63).unwrap(),
            8
        );
        assert!(Qwen35TensorSchema::expected_ffn_weight_ggml_type(&config, 64).is_err());
    }

    #[test]
    fn checkpoint_profiles_cannot_inherit_the_legacy_q8_tail() {
        for (kind,bits) in [(crate::Qwen35CheckpointProfile::QuasarNvfp4,4),
                            (crate::Qwen35CheckpointProfile::NvidiaMixed,8)] {
            let mut config=profile();
            config.quantization.source_profile=kind;
            config.quantization.group0_num_bits=bits;
            config.validate_exact_profile().unwrap();
            assert!(Qwen35TensorSchema::expected_ffn_weight_ggml_type(&config,63).is_err());
            assert!(Qwen35TensorSchema::validate_tensors(&config,&[]).is_err());
        }
    }
}
