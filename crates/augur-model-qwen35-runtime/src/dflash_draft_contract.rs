//! CPU-only identity and geometry contract for a real DFlash draft block.
//!
//! This module does not run a draft layer or select tokens. It defines the
//! handoff that a future CUDA producer must satisfy before its query can be
//! consumed by cache attention. Keeping it separate prevents the diagnostic
//! repeated-K query from being mistaken for a learned draft query.

use augur_model_qwen35::DFlashConfig;

pub const DFLASH_DRAFT_CONTRACT_SCHEMA: &str = "qwen35-dflash-learned-query-block-contract-v3";
pub const DFLASH_DRAFT_BLOCK_SCHEMA: &str = "qwen35-dflash-learned-query-eight-frame-block-v2";
const QUERY_ELEMENTS: usize = 32 * 128;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DFlashDraftBlockInput {
    pub request_generation: u64,
    pub absolute_position_start: usize,
    pub anchor_token: u32,
    /// Initial block input: the committed anchor followed by seven mask
    /// slots. Candidate tokens are a later, separate publication.
    pub block_tokens: Vec<u32>,
    pub mask_token_id: u32,
    /// Digest of the same-owner target feature capture consumed by `fc`.
    pub target_feature_sha256: [u8; 32],
}

#[derive(Clone, Debug, PartialEq)]
pub struct DFlashLearnedQueryFrame {
    pub request_generation: u64,
    pub block_offset: usize,
    pub absolute_position: usize,
    pub hidden: Vec<f32>,
    pub query: Vec<f32>,
    /// DFlash2's two-tap attention convolution is evaluated inside the fixed
    /// draft block. The producer must explicitly attest that the block-local
    /// left boundary is zero padded; a cache-attention consumer must never
    /// invent cross-request convolution history.
    pub attention_conv_block_local_zero_pad: bool,
    /// Same boundary contract for the MLP convolution path.
    pub mlp_conv_block_local_zero_pad: bool,
    pub state_digest: [u8; 32],
}

/// Atomic eight-frame handoff for one block-diffusion draft.
///
/// A consumer must not treat an individual learned-query frame as a candidate
/// token or cache-ready state.  Keeping the block wrapper explicit forces the
/// producer to present the complete fixed block with one request generation,
/// one anchor/position origin, and contiguous offsets before a later selector
/// or verifier is allowed to inspect it.
#[derive(Clone, Debug, PartialEq)]
pub struct DFlashLearnedQueryBlock {
    pub request_generation: u64,
    pub absolute_position_start: usize,
    pub anchor_token: u32,
    pub frames: Vec<DFlashLearnedQueryFrame>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DFlashDraftContractError {
    Invalid(String),
}

impl core::fmt::Display for DFlashDraftContractError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Invalid(message) => {
                write!(formatter, "invalid DFlash draft contract: {message}")
            }
        }
    }
}

impl std::error::Error for DFlashDraftContractError {}

impl DFlashDraftBlockInput {
    pub fn validate(&self, config: &DFlashConfig) -> Result<(), DFlashDraftContractError> {
        config
            .validate_exact_profile()
            .map_err(|error| invalid(error.to_string()))?;
        if self.request_generation == 0 {
            return Err(invalid("request generation must be nonzero"));
        }
        if self.block_tokens.len() != config.block_size {
            return Err(invalid(format!(
                "block token count {} != configured block size {}",
                self.block_tokens.len(),
                config.block_size
            )));
        }
        let block_end = self
            .absolute_position_start
            .checked_add(self.block_tokens.len())
            .ok_or_else(|| invalid("block position overflow"))?;
        if block_end > config.max_position_embeddings {
            return Err(invalid("block exceeds model position capacity"));
        }
        if self.anchor_token as usize >= config.vocab_size {
            return Err(invalid("anchor token exceeds vocabulary"));
        }
        if self.mask_token_id as usize >= config.vocab_size {
            return Err(invalid("mask token exceeds vocabulary"));
        }
        if self
            .block_tokens
            .iter()
            .any(|&token| token as usize >= config.vocab_size)
        {
            return Err(invalid("block token exceeds vocabulary"));
        }
        if self.target_feature_sha256 == [0; 32] {
            return Err(invalid("target feature digest is missing"));
        }
        Ok(())
    }

    pub fn validate_initial_seed(
        &self,
        config: &DFlashConfig,
    ) -> Result<(), DFlashDraftContractError> {
        self.validate(config)?;
        if self.block_tokens[0] != self.anchor_token {
            return Err(invalid(
                "initial draft block must begin with the anchor token",
            ));
        }
        if self.block_tokens[1..]
            .iter()
            .any(|&token| token != self.mask_token_id)
        {
            return Err(invalid("initial draft block tail must be mask tokens"));
        }
        Ok(())
    }

    /// Compatibility spelling retained for callers from the pre-v3 contract.
    /// The validated geometry is now `[anchor, mask x 7]`, not eight masks.
    #[deprecated(note = "use validate_initial_seed; v3 seed includes anchor at offset 0")]
    pub fn validate_initial_masked(
        &self,
        config: &DFlashConfig,
    ) -> Result<(), DFlashDraftContractError> {
        self.validate_initial_seed(config)
    }
}

impl DFlashLearnedQueryFrame {
    pub fn validate(
        &self,
        config: &DFlashConfig,
        block: &DFlashDraftBlockInput,
    ) -> Result<(), DFlashDraftContractError> {
        block.validate(config)?;
        if self.request_generation != block.request_generation {
            return Err(invalid("query generation differs from block generation"));
        }
        if self.block_offset >= block.block_tokens.len() {
            return Err(invalid("query block offset outside draft block"));
        }
        let expected_position = block
            .absolute_position_start
            .checked_add(self.block_offset)
            .ok_or_else(|| invalid("query position overflow"))?;
        if self.absolute_position != expected_position {
            return Err(invalid("query absolute position differs from block offset"));
        }
        if self.hidden.len() != config.hidden_size || self.query.len() != QUERY_ELEMENTS {
            return Err(invalid("learned query hidden/query geometry"));
        }
        if self
            .hidden
            .iter()
            .chain(&self.query)
            .any(|value| !value.is_finite())
        {
            return Err(invalid(
                "learned query hidden/query contains non-finite value",
            ));
        }
        if !self.attention_conv_block_local_zero_pad {
            return Err(invalid(
                "attention convolution must use block-local zero padding",
            ));
        }
        if !self.mlp_conv_block_local_zero_pad {
            return Err(invalid("MLP convolution must use block-local zero padding"));
        }
        if self.state_digest == [0; 32] {
            return Err(invalid("learned query state digest is missing"));
        }
        Ok(())
    }
}

impl DFlashLearnedQueryBlock {
    pub fn validate(
        &self,
        config: &DFlashConfig,
        seed: &DFlashDraftBlockInput,
    ) -> Result<(), DFlashDraftContractError> {
        seed.validate_initial_seed(config)?;
        if self.request_generation != seed.request_generation {
            return Err(invalid("query block generation differs from seed"));
        }
        if self.absolute_position_start != seed.absolute_position_start {
            return Err(invalid("query block position differs from seed"));
        }
        if self.anchor_token != seed.anchor_token {
            return Err(invalid("query block anchor differs from seed"));
        }
        if self.frames.len() != config.block_size {
            return Err(invalid(format!(
                "query frame count {} != configured block size {}",
                self.frames.len(),
                config.block_size
            )));
        }
        for (expected_offset, frame) in self.frames.iter().enumerate() {
            if frame.block_offset != expected_offset {
                return Err(invalid("query block offsets are not contiguous"));
            }
            frame.validate(config, seed)?;
        }
        Ok(())
    }
}

fn invalid(message: impl Into<String>) -> DFlashDraftContractError {
    DFlashDraftContractError::Invalid(message.into())
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
            target_layer_ids: vec![5, 19, 33, 47, 61],
        }
    }

    fn block() -> DFlashDraftBlockInput {
        DFlashDraftBlockInput {
            request_generation: 9,
            absolute_position_start: 196_600,
            anchor_token: 7,
            block_tokens: vec![
                7, 248_070, 248_070, 248_070, 248_070, 248_070, 248_070, 248_070,
            ],
            mask_token_id: 248_070,
            target_feature_sha256: [3; 32],
        }
    }

    fn frame() -> DFlashLearnedQueryFrame {
        DFlashLearnedQueryFrame {
            request_generation: 9,
            block_offset: 3,
            absolute_position: 196_603,
            hidden: vec![0.25; 5120],
            query: vec![0.5; QUERY_ELEMENTS],
            attention_conv_block_local_zero_pad: true,
            mlp_conv_block_local_zero_pad: true,
            state_digest: [4; 32],
        }
    }

    fn query_block() -> DFlashLearnedQueryBlock {
        let mut frames = Vec::with_capacity(8);
        for offset in 0..8 {
            frames.push(DFlashLearnedQueryFrame {
                request_generation: 9,
                block_offset: offset,
                absolute_position: 196_600 + offset,
                hidden: vec![0.25; 5120],
                query: vec![0.5; QUERY_ELEMENTS],
                attention_conv_block_local_zero_pad: true,
                mlp_conv_block_local_zero_pad: true,
                state_digest: [offset as u8 + 4; 32],
            });
        }
        DFlashLearnedQueryBlock {
            request_generation: 9,
            absolute_position_start: 196_600,
            anchor_token: 7,
            frames,
        }
    }

    #[test]
    fn initial_mask_block_and_frame_validate() {
        let cfg = config();
        let b = block();
        b.validate_initial_seed(&cfg).unwrap();
        frame().validate(&cfg, &b).unwrap();
    }

    #[test]
    fn contract_rejects_partial_mask_and_identity_drift() {
        let cfg = config();
        let mut b = block();
        b.block_tokens[4] = 1;
        assert!(b.validate_initial_seed(&cfg).is_err());
        b = block();
        b.block_tokens[0] = b.mask_token_id;
        assert!(b.validate_initial_seed(&cfg).is_err());
        b = block();
        b.target_feature_sha256 = [0; 32];
        assert!(b.validate(&cfg).is_err());
        let mut f = frame();
        f.request_generation = 10;
        assert!(f.validate(&cfg, &b).is_err());
        f = frame();
        f.absolute_position += 1;
        assert!(f.validate(&cfg, &b).is_err());
    }

    #[test]
    fn contract_rejects_bad_query_and_conv_boundary() {
        let cfg = config();
        let b = block();
        let mut f = frame();
        f.query.pop();
        assert!(f.validate(&cfg, &b).is_err());
        f = frame();
        f.attention_conv_block_local_zero_pad = false;
        assert!(f.validate(&cfg, &b).is_err());
        f = frame();
        f.mlp_conv_block_local_zero_pad = false;
        assert!(f.validate(&cfg, &b).is_err());
    }

    #[test]
    fn contract_rejects_position_and_vocab_overflow() {
        let cfg = config();
        let mut b = block();
        b.absolute_position_start = cfg.max_position_embeddings - 4;
        assert!(b.validate(&cfg).is_err());
        b = block();
        b.anchor_token = cfg.vocab_size as u32;
        assert!(b.validate(&cfg).is_err());
        b = block();
        b.request_generation = 0;
        assert!(b.validate(&cfg).is_err());
    }

    #[test]
    fn query_block_requires_complete_contiguous_same_owner_frames() {
        let cfg = config();
        let b = block();
        query_block().validate(&cfg, &b).unwrap();

        let mut bad = query_block();
        bad.frames.pop();
        assert!(bad.validate(&cfg, &b).is_err());

        let mut bad = query_block();
        bad.frames.swap(2, 3);
        assert!(bad.validate(&cfg, &b).is_err());

        let mut bad = query_block();
        bad.frames[4].request_generation += 1;
        assert!(bad.validate(&cfg, &b).is_err());

        let mut bad = query_block();
        bad.absolute_position_start += 1;
        assert!(bad.validate(&cfg, &b).is_err());
    }
}
