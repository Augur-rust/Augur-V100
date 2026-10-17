//! Source checkpoint vectors are not already GGUF execution vectors.
//! In particular RMSNorm offsets and GDN log-decays require different maps.
use crate::{CheckpointArtifact, Qwen35Error};
use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum CheckpointDenseTransform {
    Identity,
    AddUnitOffset,
    NegativeExp,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CheckpointDenseSpec {
    pub name: String,
    pub shape: Vec<u64>,
    pub transform: CheckpointDenseTransform,
}

pub fn checkpoint_dense_specs() -> Vec<CheckpointDenseSpec> {
    use CheckpointDenseTransform::*;
    let mut out = Vec::new();
    let mut add = |name: String, shape: &[u64], transform| {
        out.push(CheckpointDenseSpec {
            name,
            shape: shape.to_vec(),
            transform,
        });
    };
    add(
        "model.language_model.embed_tokens.weight".into(),
        &[248320, 5120],
        Identity,
    );
    add(
        "model.language_model.norm.weight".into(),
        &[5120],
        AddUnitOffset,
    );
    for layer in 0..64 {
        let p = format!("model.language_model.layers.{layer}");
        for name in ["input_layernorm", "post_attention_layernorm"] {
            add(format!("{p}.{name}.weight"), &[5120], AddUnitOffset);
        }
        if (layer + 1) % 4 == 0 {
            for name in ["q_norm", "k_norm"] {
                add(
                    format!("{p}.self_attn.{name}.weight"),
                    &[256],
                    AddUnitOffset,
                );
            }
        } else {
            add(format!("{p}.linear_attn.A_log"), &[48], NegativeExp);
            add(format!("{p}.linear_attn.dt_bias"), &[48], Identity);
            add(format!("{p}.linear_attn.norm.weight"), &[128], Identity);
            add(
                format!("{p}.linear_attn.conv1d.weight"),
                &[10240, 1, 4],
                Identity,
            );
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

fn decode(bytes: &[u8], transform: CheckpointDenseTransform) -> Result<Vec<f32>, Qwen35Error> {
    if bytes.len() % 2 != 0 {
        return Err(Qwen35Error::invalid("checkpoint.dense", "unaligned BF16"));
    }
    bytes
        .chunks_exact(2)
        .map(|b| {
            let value = f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16);
            if !value.is_finite() {
                return Err(Qwen35Error::invalid("checkpoint.dense", "nonfinite BF16"));
            }
            let result = match transform {
                CheckpointDenseTransform::Identity => value,
                CheckpointDenseTransform::AddUnitOffset => 1.0 + value,
                CheckpointDenseTransform::NegativeExp => -value.exp(),
            };
            if !result.is_finite()
                || (transform == CheckpointDenseTransform::NegativeExp && result >= 0.0)
            {
                return Err(Qwen35Error::invalid(
                    "checkpoint.dense",
                    "invalid execution transform",
                ));
            }
            Ok(result)
        })
        .collect()
}

impl CheckpointArtifact {
    pub(crate) fn validate_dense(&self) -> Result<(), Qwen35Error> {
        for spec in checkpoint_dense_specs() {
            let t = self.tensor(&spec.name)?;
            if t.dtype != "BF16" || t.shape != spec.shape {
                return Err(Qwen35Error::invalid(
                    &spec.name,
                    "dense dtype/shape mismatch",
                ));
            }
        }
        Ok(())
    }

    /// Load a named execution vector. Embeddings must use the bounded row API.
    pub fn load_dense_execution(&mut self, name: &str) -> Result<Vec<f32>, Qwen35Error> {
        let spec = checkpoint_dense_specs()
            .into_iter()
            .find(|s| s.name == name)
            .ok_or_else(|| Qwen35Error::MissingField(name.into()))?;
        if name == "model.language_model.embed_tokens.weight" {
            return Err(Qwen35Error::invalid(name, "use bounded embedding row API"));
        }
        let t = self.tensor(name)?;
        if t.dtype != "BF16" || t.shape != spec.shape {
            return Err(Qwen35Error::invalid(name, "dense dtype/shape mismatch"));
        }
        let mut bytes =
            vec![0; usize::try_from(t.byte_len()?).map_err(|_| Qwen35Error::SizeOverflow)?];
        self.read_tensor_range_into(name, 0, &mut bytes)?;
        decode(&bytes, spec.transform)
    }

    pub fn load_embedding_row(&mut self, token_id: usize) -> Result<Vec<f32>, Qwen35Error> {
        let name = "model.language_model.embed_tokens.weight";
        let t = self.tensor(name)?;
        if token_id >= 248320 || t.dtype != "BF16" || t.shape != [248320, 5120] {
            return Err(Qwen35Error::invalid(
                name,
                "embedding row out of range or wrong layout",
            ));
        }
        let mut bytes = vec![0; 5120 * 2];
        self.read_tensor_range_into(name, (token_id * 5120 * 2) as u64, &mut bytes)?;
        decode(&bytes, CheckpointDenseTransform::Identity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dense_inventory_and_transforms_are_not_interchangeable() {
        let specs = checkpoint_dense_specs();
        assert_eq!(specs.len(), 354);
        assert!(specs.windows(2).all(|w| w[0].name < w[1].name));
        let bytes = [0x00, 0x00, 0x80, 0x3f, 0x80, 0xbf];
        assert_eq!(
            decode(&bytes, CheckpointDenseTransform::Identity).unwrap(),
            [0.0, 1.0, -1.0]
        );
        assert_eq!(
            decode(&bytes, CheckpointDenseTransform::AddUnitOffset).unwrap(),
            [1.0, 2.0, 0.0]
        );
        let decay = decode(&bytes, CheckpointDenseTransform::NegativeExp).unwrap();
        assert_eq!(decay, [-1.0, -1.0_f32.exp(), -(-1.0_f32).exp()]);
        assert_eq!(
            specs
                .iter()
                .filter(|s| s.transform == CheckpointDenseTransform::NegativeExp)
                .count(),
            48
        );
        assert_eq!(
            specs
                .iter()
                .filter(|s| s.transform == CheckpointDenseTransform::AddUnitOffset)
                .count(),
            161
        );
        for bad in [&[0][..], &[0x80, 0x7f][..], &[0xc0, 0x7f][..]] {
            assert!(decode(bad, CheckpointDenseTransform::Identity).is_err());
        }
        assert!(decode(&[0xc8, 0x42], CheckpointDenseTransform::NegativeExp).is_err());
        assert!(decode(&[0x48, 0xc3], CheckpointDenseTransform::NegativeExp).is_err());
    }
}
