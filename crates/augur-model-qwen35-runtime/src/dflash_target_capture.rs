//! Identity-bound target-feature capture contract for the Qwen35 DFlash lane.
//!
//! The DFlash block producer consumes a concatenation of the target model's
//! hidden rows at layers `[5, 19, 33, 47, 61]`. A raw `[N, 25600]` file is
//! not sufficient evidence for that handoff: it must be tied to one request,
//! one context range, the exact target-layer set, and the bytes consumed by
//! the producer. This module validates that metadata without reading or
//! authorizing model logits.

use crate::DFlashDraftBlockInput;
use augur_model_qwen35::DFlashConfig;
use serde::{Deserialize, Serialize};

pub const DFLASH_TARGET_CAPTURE_SCHEMA: &str = "qwen35-dflash-target-feature-capture-v1";
pub const DFLASH_TARGET_CAPTURE_LAYER_IDS: [usize; 5] = [5, 19, 33, 47, 61];
pub const DFLASH_TARGET_CAPTURE_HIDDEN_SIZE: usize = 5_120;
pub const DFLASH_TARGET_CAPTURE_FEATURE_WIDTH: usize =
    DFLASH_TARGET_CAPTURE_LAYER_IDS.len() * DFLASH_TARGET_CAPTURE_HIDDEN_SIZE;

/// Metadata accompanying the exact F32 target-feature rows consumed by DFlash.
///
/// `source_target_model_sha256` identifies the target artifact/config owner;
/// `feature_sha256` is the digest of the little-endian F32 bytes that the
/// producer will consume. Neither digest authorizes candidate publication.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct DFlashTargetFeatureCaptureReceipt {
    pub schema: String,
    pub request_generation: u64,
    pub context_start_position: usize,
    pub context_len: usize,
    pub target_layer_ids: Vec<usize>,
    pub target_hidden_size: usize,
    pub feature_width: usize,
    pub feature_elements: usize,
    pub source_target_model_sha256: [u8; 32],
    pub feature_sha256: [u8; 32],
}

impl DFlashTargetFeatureCaptureReceipt {
    pub fn validate(&self, config: &DFlashConfig) -> Result<(), DFlashTargetFeatureCaptureError> {
        config
            .validate_exact_profile()
            .map_err(|error| invalid(error.to_string()))?;
        if self.schema != DFLASH_TARGET_CAPTURE_SCHEMA {
            return Err(invalid("target feature capture schema"));
        }
        if self.request_generation == 0 {
            return Err(invalid("target feature capture generation must be nonzero"));
        }
        if self.context_len == 0 {
            return Err(invalid("target feature capture context must be nonempty"));
        }
        if self
            .context_start_position
            .checked_add(self.context_len)
            .is_none_or(|end| end > config.max_position_embeddings)
        {
            return Err(invalid(
                "target feature capture context exceeds position capacity",
            ));
        }
        if self.target_layer_ids != DFLASH_TARGET_CAPTURE_LAYER_IDS {
            return Err(invalid("target layer set must be [5,19,33,47,61]"));
        }
        if self.target_hidden_size != DFLASH_TARGET_CAPTURE_HIDDEN_SIZE {
            return Err(invalid("target hidden size"));
        }
        if self.feature_width != DFLASH_TARGET_CAPTURE_FEATURE_WIDTH {
            return Err(invalid("target feature width"));
        }
        let expected = self
            .context_len
            .checked_mul(self.feature_width)
            .ok_or_else(|| invalid("target feature element overflow"))?;
        if self.feature_elements != expected {
            return Err(invalid("target feature element count"));
        }
        if self.source_target_model_sha256 == [0; 32] {
            return Err(invalid("target model identity digest is missing"));
        }
        if self.feature_sha256 == [0; 32] {
            return Err(invalid("target feature digest is missing"));
        }
        Ok(())
    }

    /// Bind this receipt to the exact seed and feature bytes passed to the
    /// block producer. This is an ownership/geometry check only; it does not
    /// imply target logits, candidate validity, or acceptance.
    pub fn validate_for_block(
        &self,
        config: &DFlashConfig,
        seed: &DFlashDraftBlockInput,
        feature_elements: usize,
        feature_sha256: [u8; 32],
    ) -> Result<(), DFlashTargetFeatureCaptureError> {
        self.validate(config)?;
        seed.validate_initial_seed(config)
            .map_err(|error| invalid(error.to_string()))?;
        if self.request_generation != seed.request_generation {
            return Err(invalid("target feature generation differs from seed"));
        }
        if self.context_start_position.checked_add(self.context_len)
            != Some(seed.absolute_position_start)
        {
            return Err(invalid(
                "target feature context does not end at block start",
            ));
        }
        if feature_elements != self.feature_elements {
            return Err(invalid("target feature bytes do not match receipt extent"));
        }
        if feature_sha256 != self.feature_sha256 || feature_sha256 != seed.target_feature_sha256 {
            return Err(invalid(
                "target feature bytes do not match receipt/seed digest",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DFlashTargetFeatureCaptureError {
    Invalid(String),
}

impl core::fmt::Display for DFlashTargetFeatureCaptureError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Invalid(message) => {
                write!(
                    formatter,
                    "invalid DFlash target feature capture: {message}"
                )
            }
        }
    }
}

impl std::error::Error for DFlashTargetFeatureCaptureError {}

fn invalid(message: impl Into<String>) -> DFlashTargetFeatureCaptureError {
    DFlashTargetFeatureCaptureError::Invalid(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

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
            target_layer_ids: DFLASH_TARGET_CAPTURE_LAYER_IDS.to_vec(),
        }
    }

    fn seed() -> DFlashDraftBlockInput {
        DFlashDraftBlockInput {
            request_generation: 9,
            absolute_position_start: 1008,
            anchor_token: 7,
            block_tokens: vec![7, 248070, 248070, 248070, 248070, 248070, 248070, 248070],
            mask_token_id: 248070,
            target_feature_sha256: [3; 32],
        }
    }

    fn receipt() -> DFlashTargetFeatureCaptureReceipt {
        DFlashTargetFeatureCaptureReceipt {
            schema: DFLASH_TARGET_CAPTURE_SCHEMA.into(),
            request_generation: 9,
            context_start_position: 1000,
            context_len: 8,
            target_layer_ids: DFLASH_TARGET_CAPTURE_LAYER_IDS.to_vec(),
            target_hidden_size: DFLASH_TARGET_CAPTURE_HIDDEN_SIZE,
            feature_width: DFLASH_TARGET_CAPTURE_FEATURE_WIDTH,
            feature_elements: 8 * DFLASH_TARGET_CAPTURE_FEATURE_WIDTH,
            source_target_model_sha256: [4; 32],
            feature_sha256: [3; 32],
        }
    }

    #[test]
    fn receipt_binds_same_owner_block_geometry() {
        let cfg = config();
        let seed = seed();
        let receipt = receipt();
        receipt
            .validate_for_block(&cfg, &seed, receipt.feature_elements, [3; 32])
            .unwrap();
    }

    #[test]
    fn receipt_rejects_layer_generation_extent_and_digest_drift() {
        let cfg = config();
        let seed = seed();
        let mut receipt_value = receipt();
        receipt_value.target_layer_ids.swap(0, 1);
        assert!(receipt_value.validate(&cfg).is_err());
        receipt_value = receipt();
        receipt_value.request_generation += 1;
        assert!(receipt_value
            .validate_for_block(&cfg, &seed, receipt_value.feature_elements, [3; 32])
            .is_err());
        receipt_value = receipt();
        receipt_value.feature_elements -= 1;
        assert!(receipt_value.validate(&cfg).is_err());
        receipt_value = receipt();
        assert!(receipt_value
            .validate_for_block(&cfg, &seed, receipt_value.feature_elements, [8; 32])
            .is_err());
    }
}
