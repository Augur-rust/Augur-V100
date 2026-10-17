//! Explicit source formats and the complete text-linear inventory.
use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Qwen35CheckpointProfile {
    OriginalMixed,
    QuasarNvfp4,
    NvidiaMixed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointLinearStorage {
    Nvfp4,
    Fp8,
    Bf16,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CheckpointLinearSpec {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub storage: CheckpointLinearStorage,
}

pub fn checkpoint_linear_specs(profile: Qwen35CheckpointProfile) -> Vec<CheckpointLinearSpec> {
    use CheckpointLinearStorage::*;
    if profile == Qwen35CheckpointProfile::OriginalMixed {
        return Vec::new();
    }
    let mut specs = Vec::new();
    for layer in 0..64 {
        let prefix = format!("model.language_model.layers.{layer}");
        let mut add = |suffix: &str, rows, cols, storage| {
            specs.push(CheckpointLinearSpec {
                name: format!("{prefix}.{suffix}"),
                rows,
                cols,
                storage,
            })
        };
        for (suffix, rows, cols) in [
            ("gate_proj", 17408, 5120),
            ("up_proj", 17408, 5120),
            ("down_proj", 5120, 17408),
        ] {
            add(&format!("mlp.{suffix}"), rows, cols, Nvfp4);
        }
        let attention = if profile == Qwen35CheckpointProfile::QuasarNvfp4 {
            Nvfp4
        } else {
            Fp8
        };
        if (layer + 1) % 4 == 0 {
            for (suffix, rows, cols) in [
                ("q_proj", 12288, 5120),
                ("k_proj", 1024, 5120),
                ("v_proj", 1024, 5120),
                ("o_proj", 5120, 6144),
            ] {
                add(&format!("self_attn.{suffix}"), rows, cols, attention);
            }
        } else {
            for (suffix, rows, cols) in [
                ("in_proj_qkv", 10240, 5120),
                ("in_proj_z", 6144, 5120),
                ("out_proj", 5120, 6144),
            ] {
                add(&format!("linear_attn.{suffix}"), rows, cols, attention);
            }
            let gate = if profile == Qwen35CheckpointProfile::QuasarNvfp4 {
                Nvfp4
            } else {
                Bf16
            };
            add("linear_attn.in_proj_a", 48, 5120, gate);
            add("linear_attn.in_proj_b", 48, 5120, gate);
        }
    }
    specs.push(CheckpointLinearSpec {
        name: "lm_head".into(),
        rows: 248320,
        cols: 5120,
        storage: if profile == Qwen35CheckpointProfile::QuasarNvfp4 {
            Bf16
        } else {
            Nvfp4
        },
    });
    specs.sort_by(|a, b| a.name.cmp(&b.name));
    specs
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn inventories_are_complete_and_model_specific() {
        for (profile, counts) in [
            (Qwen35CheckpointProfile::QuasarNvfp4, [496, 0, 1]),
            (Qwen35CheckpointProfile::NvidiaMixed, [193, 208, 96]),
        ] {
            let specs = checkpoint_linear_specs(profile);
            assert_eq!(specs.len(), 497);
            assert!(specs.windows(2).all(|w| w[0].name < w[1].name));
            for (kind, count) in [
                CheckpointLinearStorage::Nvfp4,
                CheckpointLinearStorage::Fp8,
                CheckpointLinearStorage::Bf16,
            ]
            .into_iter()
            .zip(counts)
            {
                assert_eq!(specs.iter().filter(|s| s.storage == kind).count(), count);
            }
            assert_eq!(specs.iter().filter(|s| s.rows == 48).count(), 96);
        }
    }
}
