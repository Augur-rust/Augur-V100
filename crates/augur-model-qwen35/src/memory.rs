use crate::config::Qwen35Config;
use crate::Qwen35Error;
use serde::Serialize;

/// Storage formats supported by the first Qwen state-budget model.
///
/// Q8_0 uses the GGML block geometry (32 values plus a two-byte scale).  The
/// row must be block aligned; silently padding a KV row would make the budget
/// disagree with the eventual cache layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum KvDType {
    F32,
    F16,
    BF16,
    Q8_0,
}

impl KvDType {
    pub const Q8_0_BLOCK_ELEMENTS: usize = 32;
    pub const Q8_0_BLOCK_BYTES: usize = 34;

    pub fn bytes_for_row(self, elements: usize) -> Result<u64, Qwen35Error> {
        if elements == 0 {
            return Err(Qwen35Error::invalid("kv_row_elements", "must be nonzero"));
        }
        let elements = u64::try_from(elements).map_err(|_| Qwen35Error::SizeOverflow)?;
        match self {
            Self::F32 => elements.checked_mul(4).ok_or(Qwen35Error::SizeOverflow),
            Self::F16 | Self::BF16 => elements.checked_mul(2).ok_or(Qwen35Error::SizeOverflow),
            Self::Q8_0 => {
                if !(elements as usize).is_multiple_of(Self::Q8_0_BLOCK_ELEMENTS) {
                    return Err(Qwen35Error::invalid(
                        "kv_row_elements",
                        "Q8_0 rows must be divisible by 32",
                    ));
                }
                let blocks = elements / Self::Q8_0_BLOCK_ELEMENTS as u64;
                blocks
                    .checked_mul(Self::Q8_0_BLOCK_BYTES as u64)
                    .ok_or(Qwen35Error::SizeOverflow)
            }
        }
    }
}

/// Exact byte budget for one request's mutable hybrid state.
///
/// The recurrent state is deliberately fixed to F32 for the first semantic
/// baseline.  Only full-attention K/V uses `kv_dtype`; no host persistence or
/// allocator fragmentation is hidden in this value.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Qwen35StateLayout {
    pub context_tokens: usize,
    pub kv_dtype: KvDType,
    pub full_attention_layers: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub kv_row_bytes: u64,
    pub full_attention_kv_bytes: u64,
    pub recurrent_state_bytes: u64,
    pub recurrent_conv_bytes: u64,
    pub recurrent_total_bytes: u64,
    pub total_state_bytes: u64,
}

impl Qwen35StateLayout {
    pub fn for_config(
        config: &Qwen35Config,
        context_tokens: usize,
        kv_dtype: KvDType,
    ) -> Result<Self, Qwen35Error> {
        config.validate_exact_profile()?;
        if context_tokens == 0 || context_tokens > config.max_position_embeddings {
            return Err(Qwen35Error::invalid(
                "context_tokens",
                format!(
                    "must be in 1..={} (got {context_tokens})",
                    config.max_position_embeddings
                ),
            ));
        }
        if config.num_key_value_heads == 0 || config.head_dim == 0 {
            return Err(Qwen35Error::invalid(
                "attention dimensions",
                "KV heads and head dimension must be nonzero",
            ));
        }
        if config.linear_conv_kernel_dim < 2 {
            return Err(Qwen35Error::invalid(
                "linear_conv_kernel_dim",
                "must leave at least one recurrent history slot",
            ));
        }
        let kv_row_bytes = kv_dtype.bytes_for_row(config.head_dim)?;
        let kv_rows = checked_product([
            context_tokens,
            config.full_attention_layers.len(),
            2,
            config.num_key_value_heads,
        ])?;
        let full_attention_kv_bytes = checked_product_u64([
            u64::try_from(kv_rows).map_err(|_| Qwen35Error::SizeOverflow)?,
            kv_row_bytes,
        ])?;

        let value_state_elements = checked_product([
            config.linear_attention_layers.len(),
            config.linear_num_value_heads,
            config.linear_value_head_dim,
            config.linear_value_head_dim,
        ])?;
        let recurrent_state_bytes = checked_product_u64([
            u64::try_from(value_state_elements).map_err(|_| Qwen35Error::SizeOverflow)?,
            4,
        ])?;

        let conv_channels = checked_sum([
            checked_product([config.linear_num_value_heads, config.linear_value_head_dim])?,
            checked_product([2, config.linear_num_key_heads, config.linear_key_head_dim])?,
        ])?;
        let conv_elements = checked_product([
            config.linear_attention_layers.len(),
            config.linear_conv_kernel_dim.saturating_sub(1),
            conv_channels,
        ])?;
        let recurrent_conv_bytes = checked_product_u64([
            u64::try_from(conv_elements).map_err(|_| Qwen35Error::SizeOverflow)?,
            4,
        ])?;
        let recurrent_total_bytes = recurrent_state_bytes
            .checked_add(recurrent_conv_bytes)
            .ok_or(Qwen35Error::SizeOverflow)?;
        let total_state_bytes = full_attention_kv_bytes
            .checked_add(recurrent_total_bytes)
            .ok_or(Qwen35Error::SizeOverflow)?;

        Ok(Self {
            context_tokens,
            kv_dtype,
            full_attention_layers: config.full_attention_layers.len(),
            kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            kv_row_bytes,
            full_attention_kv_bytes,
            recurrent_state_bytes,
            recurrent_conv_bytes,
            recurrent_total_bytes,
            total_state_bytes,
        })
    }
}

/// A pre-allocation admission calculation.  It does not allocate and is
/// intentionally independent from CUDA; the runtime can use it before any
/// device API call.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct MemoryAdmission {
    pub device_budget_bytes: u64,
    pub target_weight_bytes: u64,
    pub draft_weight_bytes: u64,
    pub state_bytes: u64,
    pub workspace_bytes: u64,
    pub safety_margin_bytes: u64,
    pub required_bytes: u64,
}

impl MemoryAdmission {
    pub fn evaluate(
        device_budget_bytes: u64,
        target_weight_bytes: u64,
        draft_weight_bytes: u64,
        state: &Qwen35StateLayout,
        workspace_bytes: u64,
        safety_margin_bytes: u64,
    ) -> Result<Self, Qwen35Error> {
        if device_budget_bytes == 0 {
            return Err(Qwen35Error::invalid(
                "device_budget_bytes",
                "must be nonzero",
            ));
        }
        if target_weight_bytes == 0 {
            return Err(Qwen35Error::invalid(
                "target_weight_bytes",
                "must be nonzero",
            ));
        }
        let required_bytes = checked_sum_u64([
            target_weight_bytes,
            draft_weight_bytes,
            state.total_state_bytes,
            workspace_bytes,
            safety_margin_bytes,
        ])?;
        Ok(Self {
            device_budget_bytes,
            target_weight_bytes,
            draft_weight_bytes,
            state_bytes: state.total_state_bytes,
            workspace_bytes,
            safety_margin_bytes,
            required_bytes,
        })
    }

    pub fn fits(&self) -> bool {
        self.required_bytes <= self.device_budget_bytes
    }

    pub fn require_fit(&self) -> Result<(), Qwen35Error> {
        if self.fits() {
            Ok(())
        } else {
            Err(Qwen35Error::invalid(
                "memory_budget",
                format!(
                    "required {} bytes exceeds device budget {} bytes",
                    self.required_bytes, self.device_budget_bytes
                ),
            ))
        }
    }
}

fn checked_product(values: impl IntoIterator<Item = usize>) -> Result<usize, Qwen35Error> {
    values.into_iter().try_fold(1usize, |accumulator, value| {
        accumulator
            .checked_mul(value)
            .ok_or(Qwen35Error::SizeOverflow)
    })
}

fn checked_sum(values: impl IntoIterator<Item = usize>) -> Result<usize, Qwen35Error> {
    values.into_iter().try_fold(0usize, |accumulator, value| {
        accumulator
            .checked_add(value)
            .ok_or(Qwen35Error::SizeOverflow)
    })
}

fn checked_product_u64(values: impl IntoIterator<Item = u64>) -> Result<u64, Qwen35Error> {
    values.into_iter().try_fold(1u64, |accumulator, value| {
        accumulator
            .checked_mul(value)
            .ok_or(Qwen35Error::SizeOverflow)
    })
}

fn checked_sum_u64(values: impl IntoIterator<Item = u64>) -> Result<u64, Qwen35Error> {
    values.into_iter().try_fold(0u64, |accumulator, value| {
        accumulator
            .checked_add(value)
            .ok_or(Qwen35Error::SizeOverflow)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::QuantizationProfile;

    fn profile() -> Qwen35Config {
        let full_attention_layers =
            vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63];
        let linear_attention_layers: Vec<usize> = (0..64)
            .filter(|layer| !full_attention_layers.contains(layer))
            .collect();
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
            full_attention_layers,
            linear_attention_layers,
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

    #[test]
    fn q8_row_uses_scale_overhead() {
        assert_eq!(KvDType::Q8_0.bytes_for_row(256).unwrap(), 272);
        assert!(KvDType::Q8_0.bytes_for_row(255).is_err());
    }

    #[test]
    fn profile_budget_matches_hybrid_state_geometry() {
        let layout = Qwen35StateLayout::for_config(&profile(), 131_072, KvDType::Q8_0).unwrap();
        assert_eq!(layout.kv_row_bytes, 272);
        assert_eq!(layout.full_attention_kv_bytes, 4_563_402_752);
        assert_eq!(layout.recurrent_state_bytes, 150_994_944);
        assert_eq!(layout.recurrent_conv_bytes, 5_898_240);
        assert_eq!(layout.total_state_bytes, 4_720_295_936);
    }

    #[test]
    fn admission_is_fail_closed_without_allocation() {
        let layout = Qwen35StateLayout::for_config(&profile(), 262_144, KvDType::Q8_0).unwrap();
        let admission = MemoryAdmission::evaluate(
            32 * 1024 * 1024 * 1024,
            22_323_015_808,
            0,
            &layout,
            2 * 1024 * 1024 * 1024,
            2 * 1024 * 1024 * 1024,
        )
        .unwrap();
        assert!(!admission.fits());
        assert!(admission.require_fit().is_err());
    }

    #[test]
    fn context_above_profile_is_rejected() {
        assert!(Qwen35StateLayout::for_config(&profile(), 262_145, KvDType::F16).is_err());
    }
}
