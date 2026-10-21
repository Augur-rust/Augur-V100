//! Child of config: reuses exact architecture parsing without inventing legacy quant metadata.
use super::*;
use crate::{checkpoint_linear_specs, CheckpointLinearStorage};
use std::collections::BTreeSet;

fn same_names(
    actual: Vec<String>,
    expected: BTreeSet<String>,
    label: &str,
) -> Result<Vec<String>, Qwen35Error> {
    let set: BTreeSet<String> = actual.iter().cloned().collect();
    if set.len() != actual.len() || set != expected {
        return Err(Qwen35Error::invalid(
            label,
            "duplicate, missing or extra target",
        ));
    }
    Ok(set.into_iter().collect())
}

pub(super) fn parse(
    root: &BTreeMap<String, StrictValue>,
    profile: Qwen35CheckpointProfile,
) -> Result<QuantizationProfile, Qwen35Error> {
    let q = required(root, "quantization_config")?.object("quantization_config")?;
    // W4A16 ModelOpt has no activation config_groups. Never translate it to
    // the older W4A4/FP8 contract or invent input_scale metadata.
    if profile == Qwen35CheckpointProfile::NerkyorW4a16 {
        return parse_nerkyor_w4a16(q);
    }
    let groups = required(q, "config_groups")?.object("config_groups")?;
    let specs = checkpoint_linear_specs(profile);
    let (bits, attention, ffn) = match profile {
        Qwen35CheckpointProfile::QuasarNvfp4 => {
            expect_string(q, "quant_method", "compressed-tensors")?;
            expect_string(q, "format", "nvfp4-pack-quantized")?;
            expect_string(q, "quantization_status", "compressed")?;
            if groups.len() != 1 {
                return Err(Qwen35Error::invalid(
                    "config_groups",
                    "expected one Augur group",
                ));
            }
            let g = required(groups, "group_0")?.object("group_0")?;
            expect_string(g, "format", "nvfp4-pack-quantized")?;
            if array_strings(g, "targets")? != ["Linear"] {
                return Err(Qwen35Error::invalid("targets", "expected Linear"));
            }
            for key in ["weights", "input_activations"] {
                let a = required(g, key)?.object(key)?;
                expect_string(a, "type", "float")?;
                expect_string(a, "strategy", "tensor_group")?;
                expect_string(a, "scale_dtype", "torch.float8_e4m3fn")?;
                expect_bool(a, "symmetric", true)?;
                if get_usize(a, "num_bits")? != 4 || get_usize(a, "group_size")? != 16 {
                    return Err(Qwen35Error::invalid(key, "expected NVFP4 K16"));
                }
                if key == "weights" {
                    expect_bool(a, "dynamic", false)?;
                } else {
                    expect_string(a, "dynamic", "local")?;
                }
            }
            same_names(
                array_strings(q, "ignore")?,
                [
                    "lm_head",
                    "re:.*visual.*",
                    "re:.*mtp.*",
                    "re:.*embed_vision.*",
                    "re:.*embed_audio.*",
                    "re:.*vision_embedder.*",
                ]
                .into_iter()
                .map(String::from)
                .collect(),
                "ignore",
            )?;
            let names = specs
                .iter()
                .filter(|s| s.storage == CheckpointLinearStorage::Nvfp4)
                .map(|s| s.name.clone())
                .collect::<Vec<_>>();
            (
                4,
                names
                    .iter()
                    .filter(|s| !s.contains(".mlp."))
                    .cloned()
                    .collect(),
                names.into_iter().filter(|s| s.contains(".mlp.")).collect(),
            )
        }
        Qwen35CheckpointProfile::NvidiaMixed => {
            expect_string(q, "quant_method", "modelopt")?;
            expect_string(q, "quant_algo", "MIXED_PRECISION")?;
            if groups.len() != 2 {
                return Err(Qwen35Error::invalid(
                    "config_groups",
                    "expected two NVIDIA groups",
                ));
            }
            let mut names = Vec::new();
            for (group, bits, storage) in [
                ("group_0", 8, CheckpointLinearStorage::Fp8),
                ("group_1", 4, CheckpointLinearStorage::Nvfp4),
            ] {
                let g = required(groups, group)?.object(group)?;
                for key in ["weights", "input_activations"] {
                    let a = required(g, key)?.object(key)?;
                    expect_string(a, "type", "float")?;
                    expect_bool(a, "dynamic", false)?;
                    if get_usize(a, "num_bits")? != bits
                        || (bits == 4 && get_usize(a, "group_size")? != 16)
                    {
                        return Err(Qwen35Error::invalid(
                            key,
                            "ModelOpt bit/group profile mismatch",
                        ));
                    }
                }
                names.push(same_names(
                    array_strings(g, "targets")?,
                    specs
                        .iter()
                        .filter(|s| s.storage == storage)
                        .map(|s| s.name.clone())
                        .collect(),
                    group,
                )?);
            }
            let layers = required(q, "quantized_layers")?.object("quantized_layers")?;
            let expected: BTreeSet<String> = specs
                .iter()
                .filter(|s| s.storage != CheckpointLinearStorage::Bf16)
                .map(|s| s.name.clone())
                .collect();
            same_names(
                layers.keys().cloned().collect(),
                expected,
                "quantized_layers",
            )?;
            for spec in &specs {
                if spec.storage == CheckpointLinearStorage::Bf16 {
                    continue;
                }
                let v = required(layers, &spec.name)?.object(&spec.name)?;
                expect_string(
                    v,
                    "quant_algo",
                    if spec.storage == CheckpointLinearStorage::Nvfp4 {
                        "NVFP4"
                    } else {
                        "FP8"
                    },
                )?;
                if spec.storage == CheckpointLinearStorage::Nvfp4
                    && get_usize(v, "group_size")? != 16
                {
                    return Err(Qwen35Error::invalid(&spec.name, "NVFP4 group size"));
                }
            }
            same_names(
                array_strings(q, "ignore")?,
                ["mtp*", "mtp.layers.0*"]
                    .into_iter()
                    .map(String::from)
                    .collect(),
                "ignore",
            )?;
            (8, names.remove(0), names.remove(0))
        }
        Qwen35CheckpointProfile::OriginalMixed | Qwen35CheckpointProfile::NerkyorW4a16 => {
            return Err(Qwen35Error::invalid("profile", "wrong parser"))
        }
    };
    Ok(QuantizationProfile {
        source_profile: profile,
        group0_num_bits: bits,
        group1_num_bits: 4,
        group1_group_size: 16,
        group0_targets: attention,
        group1_targets: ffn,
        ignored_tensor_count: array_strings(q, "ignore")?.len(),
    })
}

/// Exact exclusions for this export; vision and MTP remain outside the text
/// execution inventory. The explicit set prevents a small gate being silently
/// quantized or an important projection being silently skipped.
fn nerkyor_w4a16_ignored() -> BTreeSet<String> {
    let mut names: BTreeSet<String> = [
        "lm_head",
        "model.language_model.embed_tokens",
        "model.visual",
        "mtp",
        "model.visual.merger.linear_fc1",
        "model.visual.merger.linear_fc2",
        "model.visual.merger.norm",
        "model.visual.patch_embed.proj",
    ]
    .into_iter()
    .map(String::from)
    .collect();
    for layer in 0..64 {
        if (layer + 1) % 4 != 0 {
            for suffix in ["conv1d", "in_proj_a", "in_proj_b"] {
                names.insert(format!(
                    "model.language_model.layers.{layer}.linear_attn.{suffix}"
                ));
            }
        }
    }
    for layer in 0..27 {
        for suffix in ["attn.proj", "attn.qkv", "mlp.linear_fc1", "mlp.linear_fc2"] {
            names.insert(format!("model.visual.blocks.{layer}.{suffix}"));
        }
    }
    names
}

fn parse_nerkyor_w4a16(
    q: &BTreeMap<String, StrictValue>,
) -> Result<QuantizationProfile, Qwen35Error> {
    same_names(
        q.keys().cloned().collect(),
        [
            "producer",
            "quant_algo",
            "quant_method",
            "quantized_layers",
            "ignore",
        ]
        .into_iter()
        .map(String::from)
        .collect(),
        "quantization_config.keys",
    )?;
    expect_string(q, "quant_method", "modelopt_mixed")?;
    expect_string(q, "quant_algo", "MIXED_PRECISION")?;
    let producer = required(q, "producer")?.object("producer")?;
    same_names(
        producer.keys().cloned().collect(),
        ["name", "version"].into_iter().map(String::from).collect(),
        "producer.keys",
    )?;
    expect_string(producer, "name", "modelopt")?;
    expect_string(producer, "version", "0.46.0")?;
    let specs = checkpoint_linear_specs(Qwen35CheckpointProfile::NerkyorW4a16);
    let layers = required(q, "quantized_layers")?.object("quantized_layers")?;
    same_names(
        layers.keys().cloned().collect(),
        specs
            .iter()
            .filter(|s| s.storage == CheckpointLinearStorage::Nvfp4)
            .map(|s| s.name.clone())
            .collect(),
        "quantized_layers",
    )?;
    for (name, value) in layers {
        let entry = value.object(name)?;
        same_names(
            entry.keys().cloned().collect(),
            ["group_size", "quant_algo"]
                .into_iter()
                .map(String::from)
                .collect(),
            name,
        )?;
        expect_string(entry, "quant_algo", "W4A16_NVFP4")?;
        if get_usize(entry, "group_size")? != 16 {
            return Err(Qwen35Error::invalid(name, "W4A16 NVFP4 requires K16"));
        }
    }
    same_names(
        array_strings(q, "ignore")?,
        nerkyor_w4a16_ignored(),
        "ignore",
    )?;
    Ok(QuantizationProfile {
        source_profile: Qwen35CheckpointProfile::NerkyorW4a16,
        group0_num_bits: 4,
        group1_num_bits: 4,
        group1_group_size: 16,
        group0_targets: specs
            .iter()
            .filter(|s| s.storage == CheckpointLinearStorage::Nvfp4 && !s.name.contains(".mlp."))
            .map(|s| s.name.clone())
            .collect(),
        group1_targets: specs
            .iter()
            .filter(|s| s.storage == CheckpointLinearStorage::Nvfp4 && s.name.contains(".mlp."))
            .map(|s| s.name.clone())
            .collect(),
        ignored_tensor_count: nerkyor_w4a16_ignored().len(),
    })
}
