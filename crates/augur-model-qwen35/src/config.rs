use crate::strict_json::{required, StrictValue};
use crate::Qwen35Error;
use serde::Serialize;
use std::collections::BTreeMap;
use crate::Qwen35CheckpointProfile;
#[path = "checkpoint_config.rs"]
mod checkpoint;

const EXPECTED_ARCHITECTURE: &str = "Qwen3_5ForConditionalGeneration";
const EXPECTED_MODEL_TYPE: &str = "qwen3_5";
const EXPECTED_TEXT_MODEL_TYPE: &str = "qwen3_5_text";

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct QuantizationProfile {
    pub source_profile: Qwen35CheckpointProfile,
    pub group0_num_bits: usize,
    pub group1_num_bits: usize,
    pub group1_group_size: usize,
    pub group0_targets: Vec<String>,
    pub group1_targets: Vec<String>,
    pub ignored_tensor_count: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Qwen35Config {
    pub architecture: String,
    pub model_type: String,
    pub dtype: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub full_attention_interval: usize,
    pub full_attention_layers: Vec<usize>,
    pub linear_attention_layers: Vec<usize>,
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub rope_theta: f64,
    pub rope_dimension: usize,
    pub rope_sections: [usize; 3],
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub image_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
    pub has_vision_config: bool,
    pub quantization: QuantizationProfile,
}

impl Qwen35Config {
    pub fn parse(bytes: &[u8]) -> Result<Self, Qwen35Error> {
        Self::parse_with_profile(bytes, Qwen35CheckpointProfile::OriginalMixed)
    }

    pub fn parse_with_profile(bytes: &[u8], profile: Qwen35CheckpointProfile) -> Result<Self, Qwen35Error> {
        let value = StrictValue::parse(bytes)?;
        let root = value.object("$")?;

        expect_string(root, "model_type", EXPECTED_MODEL_TYPE)?;
        expect_string(root, "dtype", "bfloat16")?;
        expect_bool(root, "language_model_only", false)?;
        expect_bool(root, "tie_word_embeddings", false)?;

        let architectures = required(root, "architectures")?.array("architectures")?;
        if architectures.len() != 1
            || architectures[0].string("architectures")? != EXPECTED_ARCHITECTURE
        {
            return Err(Qwen35Error::invalid(
                "architectures",
                "expected Qwen3_5ForConditionalGeneration",
            ));
        }

        let text = required(root, "text_config")?.object("text_config")?;
        expect_string(text, "model_type", EXPECTED_TEXT_MODEL_TYPE)?;
        expect_string(text, "dtype", "bfloat16")?;
        expect_string(text, "hidden_act", "silu")?;
        expect_bool(text, "attention_bias", false)?;
        expect_bool(text, "attn_output_gate", true)?;
        expect_bool(text, "tie_word_embeddings", false)?;
        // The source export disables HF caching; this is metadata validation,
        // not a prohibition on the separately owned runtime KV/GDN cache.
        expect_bool(
            text,
            "use_cache",
            profile != Qwen35CheckpointProfile::NerkyorW4a16,
        )?;
        expect_string(text, "output_gate_type", "swish")?;
        expect_string(text, "mamba_ssm_dtype", "float32")?;
        if !required(root, "vision_config")?
            .object("vision_config")?
            .is_empty()
        {
            // The vision branch is recorded but intentionally not loaded by Q0.
        }

        let layer_types = required(text, "layer_types")?.array("layer_types")?;
        if layer_types.len() != 64 {
            return Err(Qwen35Error::invalid(
                "text_config.layer_types",
                format!("expected 64 entries, got {}", layer_types.len()),
            ));
        }
        let mut full_attention_layers = Vec::new();
        let mut linear_attention_layers = Vec::new();
        for (index, value) in layer_types.iter().enumerate() {
            let kind = value.string("text_config.layer_types")?;
            let expects_full = (index + 1) % 4 == 0;
            match (expects_full, kind) {
                (true, "full_attention") => full_attention_layers.push(index),
                (false, "linear_attention") => linear_attention_layers.push(index),
                (true, _) | (false, _) => {
                    return Err(Qwen35Error::invalid(
                        format!("text_config.layer_types[{index}]"),
                        "does not match the 3-linear/1-full pattern",
                    ));
                }
            }
        }

        let rope = required(text, "rope_parameters")?.object("text_config.rope_parameters")?;
        expect_string(rope, "rope_type", "default")?;
        expect_f64(rope, "rope_theta", 10_000_000.0)?;
        expect_f64(rope, "partial_rotary_factor", 0.25)?;
        let rope_sections = array_usizes(rope, "mrope_section")?;
        if rope_sections != [11, 11, 10] {
            return Err(Qwen35Error::invalid(
                "text_config.rope_parameters.mrope_section",
                "expected [11, 11, 10]",
            ));
        }
        expect_bool(rope, "mrope_interleaved", true)?;

        let quantization = if profile == Qwen35CheckpointProfile::OriginalMixed {
            parse_quantization(root)?
        } else { checkpoint::parse(root, profile)? };
        let config = Self {
            architecture: EXPECTED_ARCHITECTURE.into(),
            model_type: EXPECTED_MODEL_TYPE.into(),
            dtype: "bfloat16".into(),
            hidden_size: get_usize(text, "hidden_size")?,
            intermediate_size: get_usize(text, "intermediate_size")?,
            num_hidden_layers: get_usize(text, "num_hidden_layers")?,
            num_attention_heads: get_usize(text, "num_attention_heads")?,
            num_key_value_heads: get_usize(text, "num_key_value_heads")?,
            head_dim: get_usize(text, "head_dim")?,
            max_position_embeddings: get_usize(text, "max_position_embeddings")?,
            vocab_size: get_usize(text, "vocab_size")?,
            rms_norm_eps: required(text, "rms_norm_eps")?.f64("text_config.rms_norm_eps")?,
            full_attention_interval: get_usize(text, "full_attention_interval")?,
            full_attention_layers,
            linear_attention_layers,
            linear_conv_kernel_dim: get_usize(text, "linear_conv_kernel_dim")?,
            linear_key_head_dim: get_usize(text, "linear_key_head_dim")?,
            linear_value_head_dim: get_usize(text, "linear_value_head_dim")?,
            linear_num_key_heads: get_usize(text, "linear_num_key_heads")?,
            linear_num_value_heads: get_usize(text, "linear_num_value_heads")?,
            rope_theta: required(rope, "rope_theta")?.f64("rope_theta")?,
            rope_dimension: get_usize(text, "head_dim")? / 4,
            rope_sections: [rope_sections[0], rope_sections[1], rope_sections[2]],
            bos_token_id: required(text, "bos_token_id")?.u32("text_config.bos_token_id")?,
            eos_token_id: required(text, "eos_token_id")?.u32("text_config.eos_token_id")?,
            image_token_id: required(root, "image_token_id")?.u32("image_token_id")?,
            vision_start_token_id: required(root, "vision_start_token_id")?
                .u32("vision_start_token_id")?,
            vision_end_token_id: required(root, "vision_end_token_id")?
                .u32("vision_end_token_id")?,
            has_vision_config: true,
            quantization,
        };
        config.validate_exact_profile()?;
        Ok(config)
    }

    pub fn validate_exact_profile(&self) -> Result<(), Qwen35Error> {
        let expected = [
            ("hidden_size", self.hidden_size, 5120),
            ("intermediate_size", self.intermediate_size, 17408),
            ("num_hidden_layers", self.num_hidden_layers, 64),
            ("num_attention_heads", self.num_attention_heads, 24),
            ("num_key_value_heads", self.num_key_value_heads, 4),
            ("head_dim", self.head_dim, 256),
            (
                "max_position_embeddings",
                self.max_position_embeddings,
                262_144,
            ),
            ("vocab_size", self.vocab_size, 248_320),
            ("full_attention_interval", self.full_attention_interval, 4),
            ("linear_conv_kernel_dim", self.linear_conv_kernel_dim, 4),
            ("linear_key_head_dim", self.linear_key_head_dim, 128),
            ("linear_value_head_dim", self.linear_value_head_dim, 128),
            ("linear_num_key_heads", self.linear_num_key_heads, 16),
            ("linear_num_value_heads", self.linear_num_value_heads, 48),
        ];
        for (field, actual, expected) in expected {
            if actual != expected {
                return Err(Qwen35Error::invalid(
                    field,
                    format!("expected {expected}, got {actual}"),
                ));
            }
        }
        let expected_full = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63];
        let expected_linear: Vec<usize> = (0..self.num_hidden_layers)
            .filter(|layer| !expected_full.contains(layer))
            .collect();
        if self.full_attention_layers != expected_full
            || self.linear_attention_layers != expected_linear
        {
            return Err(Qwen35Error::invalid(
                "layer_types",
                "attention layer partition is not the Qwen3.8 profile",
            ));
        }
        if !approx_eq(self.rms_norm_eps, 1e-6)
            || !approx_eq(self.rope_theta, 10_000_000.0)
            || self.rope_dimension != 64
            || self.rope_sections != [11, 11, 10]
        {
            return Err(Qwen35Error::invalid(
                "rope/norm",
                "unexpected numerical profile",
            ));
        }
        if self.bos_token_id != 248_044
            || self.eos_token_id != 248_044
            || self.image_token_id != 248_056
            || self.vision_start_token_id != 248_053
            || self.vision_end_token_id != 248_054
        {
            return Err(Qwen35Error::invalid(
                "token_ids",
                "unexpected Qwen token ids",
            ));
        }
        let attention_bits = if matches!(self.quantization.source_profile, Qwen35CheckpointProfile::QuasarNvfp4 | Qwen35CheckpointProfile::NerkyorW4a16) {4} else {8};
        if self.quantization.group0_num_bits != attention_bits
            || self.quantization.group1_num_bits != 4
            || self.quantization.group1_group_size != 16
        {
            return Err(Qwen35Error::invalid(
                "quantization_config",
                "mixed NVFP4/8-bit profile does not match",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DFlashConfig {
    pub architecture: String,
    pub model_type: String,
    pub dtype: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_target_layers: usize,
    pub attention_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    pub block_size: usize,
    pub conv_group_size: usize,
    pub conv_kernel_size: usize,
    pub selector_rank: usize,
    pub selector_top_k: usize,
    pub target_layer_ids: Vec<usize>,
}

impl DFlashConfig {
    pub fn parse(bytes: &[u8]) -> Result<Self, Qwen35Error> {
        Self::parse_with_dtype_policy(bytes, false)
    }

    /// Parse the converted FP16 profile only when the caller explicitly opts
    /// in. The default `parse` path remains BF16-only so a file conversion
    /// cannot silently change the execution contract.
    pub fn parse_f16_opt_in(bytes: &[u8]) -> Result<Self, Qwen35Error> {
        Self::parse_with_dtype_policy(bytes, true)
    }

    fn parse_with_dtype_policy(bytes: &[u8], allow_f16: bool) -> Result<Self, Qwen35Error> {
        let value = StrictValue::parse(bytes)?;
        let root = value.object("$")?;
        let architectures = required(root, "architectures")?.array("architectures")?;
        if architectures.len() != 1
            || architectures[0].string("architectures")? != "DFlash2DraftModel"
        {
            return Err(Qwen35Error::invalid(
                "architectures",
                "expected DFlash2DraftModel",
            ));
        }
        expect_string(root, "model_type", "qwen3")?;
        let dtype = required(root, "dtype")?.string("dtype")?.to_owned();
        if dtype != "bfloat16" && !(allow_f16 && dtype == "float16") {
            return Err(Qwen35Error::invalid(
                "dtype",
                if allow_f16 {
                    "expected bfloat16 or explicit float16 opt-in"
                } else {
                    "expected bfloat16; use parse_f16_opt_in for float16"
                },
            ));
        }
        expect_string(root, "hidden_act", "silu")?;
        expect_bool(root, "is_causal", false)?;
        expect_bool(root, "use_cache", true)?;
        expect_bool(root, "use_sliding_window", true)?;
        // This exact draft profile uses the checkpoint's NeoX half-split
        // Q/K layout. Missing flags inherit that default; an explicit other
        // layout must not silently reach a hard-coded NeoX executor.
        for key in ["rope_is_neox_style", "is_neox_style"] {
            if root.contains_key(key) {
                expect_bool(root, key, true)?;
            }
        }
        if let Some(rope) = root.get("rope_parameters") {
            let rope = rope.object("rope_parameters")?;
            expect_string(rope, "rope_type", "default")?;
            expect_f64(rope, "rope_theta", 10_000_000.0)?;
            if rope.contains_key("partial_rotary_factor") {
                expect_f64(rope, "partial_rotary_factor", 1.0)?;
            }
        }
        if root.contains_key("rms_norm_eps") {
            expect_f64(root, "rms_norm_eps", 1.0e-6)?;
        }
        let layer_types = required(root, "layer_types")?.array("layer_types")?;
        if layer_types.len() != 5 {
            return Err(Qwen35Error::invalid(
                "layer_types",
                "DFlash draft must have five sliding-attention layers",
            ));
        }
        for value in layer_types {
            if value.string("layer_types")? != "sliding_attention" {
                return Err(Qwen35Error::invalid(
                    "layer_types",
                    "DFlash draft must have five sliding-attention layers",
                ));
            }
        }
        let dflash = required(root, "dflash_config")?.object("dflash_config")?;
        if get_usize(dflash, "mask_token_id")? != 248070 {
            return Err(Qwen35Error::invalid(
                "dflash_config.mask_token_id",
                "expected 248070",
            ));
        }
        let target_layer_ids = array_usizes(dflash, "target_layer_ids")?;
        let config = Self {
            architecture: "DFlash2DraftModel".into(),
            model_type: "qwen3".into(),
            dtype,
            hidden_size: get_usize(root, "hidden_size")?,
            intermediate_size: get_usize(root, "intermediate_size")?,
            num_hidden_layers: get_usize(root, "num_hidden_layers")?,
            num_target_layers: get_usize(root, "num_target_layers")?,
            attention_heads: get_usize(root, "num_attention_heads")?,
            kv_heads: get_usize(root, "num_key_value_heads")?,
            head_dim: get_usize(root, "head_dim")?,
            max_position_embeddings: get_usize(root, "max_position_embeddings")?,
            vocab_size: get_usize(root, "vocab_size")?,
            block_size: get_usize(dflash, "block_size")?,
            conv_group_size: get_usize(dflash, "conv_group_size")?,
            conv_kernel_size: get_usize(dflash, "conv_kernel_size")?,
            selector_rank: get_usize(dflash, "selector_rank")?,
            selector_top_k: get_usize(dflash, "selector_top_k")?,
            target_layer_ids,
        };
        config.validate_exact_profile()?;
        Ok(config)
    }

    pub fn validate_exact_profile(&self) -> Result<(), Qwen35Error> {
        let expected = [
            ("hidden_size", self.hidden_size, 5120),
            ("intermediate_size", self.intermediate_size, 17408),
            ("num_hidden_layers", self.num_hidden_layers, 5),
            ("num_target_layers", self.num_target_layers, 64),
            ("attention_heads", self.attention_heads, 32),
            ("kv_heads", self.kv_heads, 8),
            ("head_dim", self.head_dim, 128),
            (
                "max_position_embeddings",
                self.max_position_embeddings,
                262_144,
            ),
            ("vocab_size", self.vocab_size, 248_320),
            ("block_size", self.block_size, 8),
            ("conv_group_size", self.conv_group_size, 16),
            ("conv_kernel_size", self.conv_kernel_size, 2),
            ("selector_rank", self.selector_rank, 256),
            ("selector_top_k", self.selector_top_k, 16),
        ];
        for (field, actual, expected) in expected {
            if actual != expected {
                return Err(Qwen35Error::invalid(
                    field,
                    format!("expected {expected}, got {actual}"),
                ));
            }
        }
        if self.target_layer_ids != [5, 19, 33, 47, 61] {
            return Err(Qwen35Error::invalid(
                "dflash_config.target_layer_ids",
                "expected [5, 19, 33, 47, 61]",
            ));
        }
        Ok(())
    }
}

fn parse_quantization(
    root: &BTreeMap<String, StrictValue>,
) -> Result<QuantizationProfile, Qwen35Error> {
    let quant = required(root, "quantization_config")?.object("quantization_config")?;
    expect_string(quant, "format", "mixed-precision")?;
    let groups = required(quant, "config_groups")?.object("quantization_config.config_groups")?;
    let group0 = required(groups, "group_0")?.object("config_groups.group_0")?;
    let group1 = required(groups, "group_1")?.object("config_groups.group_1")?;
    expect_string(group0, "format", "float-quantized")?;
    expect_string(group1, "format", "nvfp4-pack-quantized")?;

    let group0_weights = required(group0, "weights")?.object("group_0.weights")?;
    expect_string(group0_weights, "type", "float")?;
    expect_string(group0_weights, "strategy", "channel")?;
    expect_string(group0_weights, "observer", "memoryless_minmax")?;
    expect_bool(group0_weights, "dynamic", false)?;
    expect_bool(group0_weights, "symmetric", true)?;
    let group0_targets = array_strings(group0, "targets")?;
    if group0_targets
        != [
            "re:.*self_attn\\.(q|k|v|o)_proj$",
            "re:.*linear_attn\\.(in_proj_qkv|in_proj_z|out_proj)$",
            "re:.*lm_head",
            "re:.*layers\\.(56|57|58|59|60|61|62|63)\\.mlp\\.(gate|up|down)_proj$",
        ]
    {
        return Err(Qwen35Error::invalid(
            "quantization_config.config_groups.group_0.targets",
            "target regex set differs from the audited Qwen profile",
        ));
    }

    let group1_weights = required(group1, "weights")?.object("group_1.weights")?;
    expect_string(group1_weights, "type", "float")?;
    expect_string(group1_weights, "strategy", "tensor_group")?;
    expect_string(group1_weights, "actorder", "static")?;
    expect_string(group1_weights, "scale_dtype", "torch.float8_e4m3fn")?;
    expect_bool(group1_weights, "dynamic", false)?;
    expect_bool(group1_weights, "symmetric", true)?;
    let group1_targets = array_strings(group1, "targets")?;
    if group1_targets != ["re:.*mlp\\.(gate|up|down)_proj$"] {
        return Err(Qwen35Error::invalid(
            "quantization_config.config_groups.group_1.targets",
            "target regex set differs from the audited Qwen profile",
        ));
    }
    let ignored_tensor_count = required(quant, "ignore")?
        .array("quantization_config.ignore")?
        .len();
    Ok(QuantizationProfile {
        source_profile: Qwen35CheckpointProfile::OriginalMixed,
        group0_num_bits: get_usize(group0_weights, "num_bits")?,
        group1_num_bits: get_usize(group1_weights, "num_bits")?,
        group1_group_size: get_usize(group1_weights, "group_size")?,
        group0_targets,
        group1_targets,
        ignored_tensor_count,
    })
}

fn get_usize(
    object: &BTreeMap<String, StrictValue>,
    field: &'static str,
) -> Result<usize, Qwen35Error> {
    required(object, field)?.usize(field)
}

fn array_usizes(
    object: &BTreeMap<String, StrictValue>,
    field: &'static str,
) -> Result<Vec<usize>, Qwen35Error> {
    required(object, field)?
        .array(field)?
        .iter()
        .map(|value| value.usize(field))
        .collect()
}

fn array_strings(
    object: &BTreeMap<String, StrictValue>,
    field: &'static str,
) -> Result<Vec<String>, Qwen35Error> {
    required(object, field)?
        .array(field)?
        .iter()
        .map(|value| Ok(value.string(field)?.to_owned()))
        .collect()
}

fn expect_string(
    object: &BTreeMap<String, StrictValue>,
    field: &'static str,
    expected: &str,
) -> Result<(), Qwen35Error> {
    let actual = required(object, field)?.string(field)?;
    if actual == expected {
        Ok(())
    } else {
        Err(Qwen35Error::invalid(
            field,
            format!("expected {expected:?}, got {actual:?}"),
        ))
    }
}

fn expect_bool(
    object: &BTreeMap<String, StrictValue>,
    field: &'static str,
    expected: bool,
) -> Result<(), Qwen35Error> {
    let actual = required(object, field)?.bool(field)?;
    if actual == expected {
        Ok(())
    } else {
        Err(Qwen35Error::invalid(
            field,
            format!("expected {expected}, got {actual}"),
        ))
    }
}

fn expect_f64(
    object: &BTreeMap<String, StrictValue>,
    field: &'static str,
    expected: f64,
) -> Result<(), Qwen35Error> {
    let actual = required(object, field)?.f64(field)?;
    if approx_eq(actual, expected) {
        Ok(())
    } else {
        Err(Qwen35Error::invalid(
            field,
            format!("expected {expected}, got {actual}"),
        ))
    }
}

fn approx_eq(left: f64, right: f64) -> bool {
    (left - right).abs() <= right.abs() * 1e-12_f64.max(1e-12)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_config() -> String {
        let layers = (0..64)
            .map(|index| {
                if (index + 1) % 4 == 0 {
                    "full_attention"
                } else {
                    "linear_attention"
                }
            })
            .map(|value| format!("\"{value}\""))
            .collect::<Vec<_>>()
            .join(",");
        let value = serde_json::json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "dtype": "bfloat16",
            "language_model_only": false,
            "tie_word_embeddings": false,
            "image_token_id": 248056,
            "vision_start_token_id": 248053,
            "vision_end_token_id": 248054,
            "vision_config": {},
            "text_config": {
                "model_type": "qwen3_5_text", "dtype": "bfloat16", "hidden_act": "silu",
                "attention_bias": false, "attn_output_gate": true, "tie_word_embeddings": false,
                "use_cache": true, "output_gate_type": "swish", "mamba_ssm_dtype": "float32",
                "layer_types": serde_json::from_str::<serde_json::Value>(&format!("[{layers}]")).unwrap(),
                "rope_parameters": {"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "mrope_section": [11,11,10], "mrope_interleaved": true},
                "hidden_size": 5120, "intermediate_size": 17408, "num_hidden_layers": 64,
                "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256,
                "max_position_embeddings": 262144, "vocab_size": 248320, "rms_norm_eps": 0.000001,
                "full_attention_interval": 4, "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128,
                "linear_value_head_dim": 128, "linear_num_key_heads": 16, "linear_num_value_heads": 48,
                "bos_token_id": 248044, "eos_token_id": 248044
            },
            "quantization_config": {
                "format": "mixed-precision", "ignore": ["vision"],
                "config_groups": {
                    "group_0": {"format": "float-quantized", "targets": [
                        "re:.*self_attn\\.(q|k|v|o)_proj$", "re:.*linear_attn\\.(in_proj_qkv|in_proj_z|out_proj)$",
                        "re:.*lm_head", "re:.*layers\\.(56|57|58|59|60|61|62|63)\\.mlp\\.(gate|up|down)_proj$"
                    ], "weights": {"type": "float", "strategy": "channel", "observer": "memoryless_minmax", "dynamic": false, "symmetric": true, "num_bits": 8}},
                    "group_1": {"format": "nvfp4-pack-quantized", "targets": ["re:.*mlp\\.(gate|up|down)_proj$"],
                        "weights": {"type": "float", "strategy": "tensor_group", "observer": "static_minmax", "actorder": "static", "scale_dtype": "torch.float8_e4m3fn", "dynamic": false, "symmetric": true, "num_bits": 4, "group_size": 16}}
                }
            }
        });
        serde_json::to_string(&value).unwrap()
    }

    fn valid_dflash() -> String {
        r#"{"architectures":["DFlash2DraftModel"],"model_type":"qwen3","dtype":"bfloat16","hidden_act":"silu","is_causal":false,"use_cache":true,"use_sliding_window":true,"layer_types":["sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention"],"hidden_size":5120,"intermediate_size":17408,"num_hidden_layers":5,"num_target_layers":64,"num_attention_heads":32,"num_key_value_heads":8,"head_dim":128,"max_position_embeddings":262144,"vocab_size":248320,"dflash_config":{"mask_token_id":248070,"block_size":8,"conv_group_size":16,"conv_kernel_size":2,"selector_rank":256,"selector_top_k":16,"target_layer_ids":[5,19,33,47,61]}}"#.into()
    }

    #[test]
    fn exact_qwen_profile_parses() {
        let config = Qwen35Config::parse(valid_config().as_bytes()).unwrap();
        assert_eq!(config.full_attention_layers.len(), 16);
        assert_eq!(config.linear_attention_layers.len(), 48);
        assert_eq!(config.quantization.group1_group_size, 16);
    }

    #[test]
    fn duplicate_nested_key_is_rejected() {
        let changed = valid_config().replace(
            "\"rope_type\":\"default\"",
            "\"rope_type\":\"default\",\"rope_type\":\"linear\"",
        );
        assert!(Qwen35Config::parse(changed.as_bytes()).is_err());
    }

    #[test]
    fn layer_pattern_drift_is_rejected() {
        let changed = valid_config().replace(
            "\"linear_attention\",\"linear_attention\",\"linear_attention\",\"full_attention\"",
            "\"full_attention\",\"linear_attention\",\"linear_attention\",\"full_attention\"",
        );
        assert!(Qwen35Config::parse(changed.as_bytes()).is_err());
    }

    #[test]
    fn dflash_profile_parses() {
        let config = DFlashConfig::parse(valid_dflash().as_bytes()).unwrap();
        assert_eq!(config.dtype, "bfloat16");
        assert_eq!(config.target_layer_ids, [5, 19, 33, 47, 61]);
        assert_eq!(config.selector_top_k, 16);
    }

    #[test]
    fn dflash_fp16_requires_explicit_opt_in() {
        let changed = valid_dflash().replace("\"dtype\":\"bfloat16\"", "\"dtype\":\"float16\"");
        assert!(DFlashConfig::parse(changed.as_bytes()).is_err());
        let config = DFlashConfig::parse_f16_opt_in(changed.as_bytes()).unwrap();
        assert_eq!(config.dtype, "float16");
    }

    #[test]
    fn dflash_explicit_rope_layout_and_numeric_profile_drift_are_rejected() {
        for field in [
            r#""rope_is_neox_style":false"#,
            r#""is_neox_style":false"#,
            r#""rope_is_neox_style":1"#,
            r#""rope_parameters":{"rope_type":"linear","rope_theta":10000000}"#,
            r#""rope_parameters":{"rope_type":"default","rope_theta":10000}"#,
            r#""rope_parameters":{"rope_type":"default","rope_theta":10000000,"partial_rotary_factor":0.5}"#,
            r#""rms_norm_eps":0.001"#,
        ] {
            let changed = valid_dflash().replacen('{', &format!("{{{field},"), 1);
            assert!(DFlashConfig::parse(changed.as_bytes()).is_err(), "{field}");
        }
        let explicit = valid_dflash().replacen(
            '{',
            r#"{"rope_is_neox_style":true,"is_neox_style":true,"rms_norm_eps":0.000001,"rope_parameters":{"rope_type":"default","rope_theta":10000000},"#,
            1,
        );
        assert!(DFlashConfig::parse(explicit.as_bytes()).is_ok());
    }

    #[test]
    fn dflash_target_layer_drift_is_rejected() {
        let changed = valid_dflash().replace("[5,19,33,47,61]", "[5,19,33,47,60]");
        assert!(DFlashConfig::parse(changed.as_bytes()).is_err());
    }

    #[test]
    fn dflash_mask_identity_is_not_an_unchecked_metadata_field() {
        for changed in [
            valid_dflash().replace("\"mask_token_id\":248070,", ""),
            valid_dflash().replace("\"mask_token_id\":248070", "\"mask_token_id\":0"),
        ] {
            assert!(DFlashConfig::parse(changed.as_bytes()).is_err());
        }
    }
}
