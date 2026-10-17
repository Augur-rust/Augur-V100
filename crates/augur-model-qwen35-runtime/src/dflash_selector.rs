//! DFlash2 candidate-lattice selector oracle.
//!
//! The draft network produces, for every block position, a top-k candidate
//! set, unary target-logit scores, and a rank-sized selector gate. The
//! selector combines those values with the predecessor/successor codebooks
//! and walks one coherent path through the block. This module is deliberately
//! CUDA-free: it is the semantic reference used by the later hot-window GPU
//! consumer and never substitutes an oracle output for a producer output.

use std::fmt;

use crate::dflash_artifact::DFlashRamResident;
use crate::dflash_lifecycle::DFlashDType;
use serde_json::Value;

pub const DFLASH_SELECTOR_SCHEMA: &str = "qwen35-dflash2-selector-oracle-v1";

const PREDECESSOR_CODEBOOK: &str = "candidate_selector.predecessor_codebook";
const SUCCESSOR_CODEBOOK: &str = "candidate_selector.successor_codebook";
const HIDDEN_PROJECTION: &str = "candidate_selector.hidden_projection.weight";

#[derive(Clone, Debug)]
pub struct DFlashSelectorArtifact {
    resident: DFlashRamResident,
    dtype: DFlashDType,
    vocab_size: usize,
    selector_rank: usize,
    hidden_size: usize,
    predecessor_range: (usize, usize),
    successor_range: (usize, usize),
    hidden_projection_range: (usize, usize),
}

impl DFlashSelectorArtifact {
    /// Bind the three selector tensors to one already identity-validated RAM
    /// owner. The returned handle is cheap to clone because the resident's
    /// bytes are reference counted; it does not make a second 3.85 GB copy.
    pub fn bind(
        resident: DFlashRamResident,
        vocab_size: usize,
        selector_rank: usize,
        hidden_size: usize,
    ) -> Result<Self, DFlashSelectorError> {
        if resident.inspection().artifact.dtype != DFlashDType::Bf16
            || vocab_size == 0
            || selector_rank == 0
            || hidden_size == 0
        {
            return Err(DFlashSelectorError::Invalid(
                "selector artifact requires BF16 and nonzero geometry".into(),
            ));
        }
        let header_len = usize::try_from(resident.inspection().header_bytes)
            .map_err(|_| DFlashSelectorError::Invalid("selector header extent".into()))?;
        let header_end = 8usize
            .checked_add(header_len)
            .ok_or_else(|| DFlashSelectorError::Invalid("selector header overflow".into()))?;
        let header = resident
            .file_bytes()
            .get(8..header_end)
            .ok_or_else(|| DFlashSelectorError::Invalid("selector header extent".into()))?;
        let root: Value = serde_json::from_slice(header).map_err(|error| {
            DFlashSelectorError::Invalid(format!("selector header JSON: {error}"))
        })?;
        let predecessor_range = tensor_range(
            &root,
            PREDECESSOR_CODEBOOK,
            &[vocab_size, selector_rank],
            resident.payload().len(),
            DFlashDType::Bf16,
        )?;
        let successor_range = tensor_range(
            &root,
            SUCCESSOR_CODEBOOK,
            &[vocab_size, selector_rank],
            resident.payload().len(),
            DFlashDType::Bf16,
        )?;
        let hidden_projection_range = tensor_range(
            &root,
            HIDDEN_PROJECTION,
            &[selector_rank, hidden_size],
            resident.payload().len(),
            DFlashDType::Bf16,
        )?;
        Ok(Self {
            resident,
            dtype: DFlashDType::Bf16,
            vocab_size,
            selector_rank,
            hidden_size,
            predecessor_range,
            successor_range,
            hidden_projection_range,
        })
    }

    /// Bind selector tensors from the identity-validated F16 artifact.  The
    /// CPU selector remains numerically exact to the F16 wire values, while
    /// bounded GPU consumers can request a separately converted BF16 tile.
    pub fn bind_f16(
        resident: DFlashRamResident,
        vocab_size: usize,
        selector_rank: usize,
        hidden_size: usize,
    ) -> Result<Self, DFlashSelectorError> {
        if resident.inspection().artifact.dtype != DFlashDType::F16
            || vocab_size == 0
            || selector_rank == 0
            || hidden_size == 0
        {
            return Err(DFlashSelectorError::Invalid(
                "selector F16 artifact requires F16 and nonzero geometry".into(),
            ));
        }
        let header_len = usize::try_from(resident.inspection().header_bytes)
            .map_err(|_| DFlashSelectorError::Invalid("selector header extent".into()))?;
        let header_end = 8usize
            .checked_add(header_len)
            .ok_or_else(|| DFlashSelectorError::Invalid("selector header overflow".into()))?;
        let header = resident
            .file_bytes()
            .get(8..header_end)
            .ok_or_else(|| DFlashSelectorError::Invalid("selector header extent".into()))?;
        let root: Value = serde_json::from_slice(header).map_err(|error| {
            DFlashSelectorError::Invalid(format!("selector header JSON: {error}"))
        })?;
        let predecessor_range = tensor_range(
            &root,
            PREDECESSOR_CODEBOOK,
            &[vocab_size, selector_rank],
            resident.payload().len(),
            DFlashDType::F16,
        )?;
        let successor_range = tensor_range(
            &root,
            SUCCESSOR_CODEBOOK,
            &[vocab_size, selector_rank],
            resident.payload().len(),
            DFlashDType::F16,
        )?;
        let hidden_projection_range = tensor_range(
            &root,
            HIDDEN_PROJECTION,
            &[selector_rank, hidden_size],
            resident.payload().len(),
            DFlashDType::F16,
        )?;
        Ok(Self {
            resident,
            dtype: DFlashDType::F16,
            vocab_size,
            selector_rank,
            hidden_size,
            predecessor_range,
            successor_range,
            hidden_projection_range,
        })
    }

    pub fn resident(&self) -> &DFlashRamResident {
        &self.resident
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn selector_rank(&self) -> usize {
        self.selector_rank
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn predecessor_row_f32(&self, token: u32) -> Result<Vec<f32>, DFlashSelectorError> {
        self.read_row(self.predecessor_range, token)
    }

    pub fn successor_row_f32(&self, token: u32) -> Result<Vec<f32>, DFlashSelectorError> {
        self.read_row(self.successor_range, token)
    }

    /// Borrow a contiguous successor-codebook tile in its original BF16
    /// representation. The hot-window uploader uses this method so the
    /// bytes copied to HBM are exactly the identity-bound artifact bytes.
    pub fn successor_tile_bf16(
        &self,
        first_token: u32,
        token_count: usize,
    ) -> Result<&[u8], DFlashSelectorError> {
        let first = usize::try_from(first_token)
            .map_err(|_| DFlashSelectorError::Invalid("selector token".into()))?;
        let end_token = first
            .checked_add(token_count)
            .ok_or_else(|| DFlashSelectorError::Invalid("selector tile token overflow".into()))?;
        if token_count == 0 || end_token > self.vocab_size {
            return Err(DFlashSelectorError::Invalid(
                "selector tile exceeds vocabulary".into(),
            ));
        }
        let row_bytes = self
            .selector_rank
            .checked_mul(2)
            .ok_or_else(|| DFlashSelectorError::Invalid("codebook row overflow".into()))?;
        let start = first
            .checked_mul(row_bytes)
            .ok_or_else(|| DFlashSelectorError::Invalid("selector tile offset".into()))?;
        let bytes = token_count
            .checked_mul(row_bytes)
            .ok_or_else(|| DFlashSelectorError::Invalid("selector tile bytes overflow".into()))?;
        let source = self.payload_slice(self.successor_range)?;
        source
            .get(start..start + bytes)
            .ok_or_else(|| DFlashSelectorError::Invalid("selector tile extent".into()))
    }

    /// Compute the selector gate from the real hidden-projection tensor. The
    /// caller supplies the DFlash backbone's hidden vector; this method does
    /// not invent or replace that vector.
    pub fn hidden_projection_gate_f32(
        &self,
        hidden: &[f32],
    ) -> Result<Vec<f32>, DFlashSelectorError> {
        if hidden.len() != self.hidden_size {
            return Err(DFlashSelectorError::Invalid(
                "hidden vector does not match selector projection".into(),
            ));
        }
        let source = self.payload_slice(self.hidden_projection_range)?;
        let row_bytes = self
            .hidden_size
            .checked_mul(2)
            .ok_or_else(|| DFlashSelectorError::Invalid("hidden projection row overflow".into()))?;
        let mut output = Vec::with_capacity(self.selector_rank);
        for row in 0..self.selector_rank {
            let start = row
                .checked_mul(row_bytes)
                .ok_or_else(|| DFlashSelectorError::Invalid("hidden projection offset".into()))?;
            let bytes = source.get(start..start + row_bytes).ok_or_else(|| {
                DFlashSelectorError::Invalid("hidden projection row extent".into())
            })?;
            let mut sum = 0.0_f64;
            for (index, word) in bytes.chunks_exact(2).enumerate() {
                let value = self.decode_scalar(word)?;
                sum += f64::from(value) * f64::from(hidden[index]);
            }
            let value = sum as f32;
            if !value.is_finite() {
                return Err(DFlashSelectorError::Invalid(
                    "hidden projection output is non-finite".into(),
                ));
            }
            output.push(value);
        }
        Ok(output)
    }

    /// Borrow the selector hidden-projection matrix in its original BF16 wire
    /// layout. A hot-window adapter can consume this bounded real tensor
    /// without uploading the complete draft model.
    pub fn hidden_projection_tile_bf16(&self) -> Result<&[u8], DFlashSelectorError> {
        if self.dtype != DFlashDType::Bf16 {
            return Err(DFlashSelectorError::Invalid(
                "F16 selector requires owned BF16 conversion".into(),
            ));
        }
        self.payload_slice(self.hidden_projection_range)
    }

    /// Return a bounded selector tile in BF16 wire format. For BF16 this is a
    /// copy of the original bytes; for F16 each element is converted with the
    /// canonical round-to-nearest-even BF16 conversion. The full artifact is
    /// never duplicated or converted.
    pub fn hidden_projection_tile_bf16_owned(&self) -> Result<Vec<u8>, DFlashSelectorError> {
        let source = self.payload_slice(self.hidden_projection_range)?;
        if self.dtype == DFlashDType::Bf16 {
            return Ok(source.to_vec());
        }
        let mut output = Vec::with_capacity(source.len());
        for word in source.chunks_exact(2) {
            let value = half::f16::from_bits(u16::from_le_bytes([word[0], word[1]])).to_f32();
            if !value.is_finite() {
                return Err(DFlashSelectorError::Invalid(
                    "F16 selector contains non-finite value".into(),
                ));
            }
            let bits = value.to_bits();
            let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
            output.extend_from_slice(&((rounded >> 16) as u16).to_le_bytes());
        }
        Ok(output)
    }

    fn read_row(&self, range: (usize, usize), token: u32) -> Result<Vec<f32>, DFlashSelectorError> {
        let token = usize::try_from(token)
            .map_err(|_| DFlashSelectorError::Invalid("selector token".into()))?;
        if token >= self.vocab_size {
            return Err(DFlashSelectorError::Invalid(
                "selector token exceeds vocabulary".into(),
            ));
        }
        let row_bytes = self
            .selector_rank
            .checked_mul(2)
            .ok_or_else(|| DFlashSelectorError::Invalid("codebook row overflow".into()))?;
        let start = token
            .checked_mul(row_bytes)
            .ok_or_else(|| DFlashSelectorError::Invalid("codebook row offset".into()))?;
        let source = self.payload_slice(range)?;
        let bytes = source
            .get(start..start + row_bytes)
            .ok_or_else(|| DFlashSelectorError::Invalid("codebook row extent".into()))?;
        bytes
            .chunks_exact(2)
            .map(|word| self.decode_scalar(word))
            .collect()
    }

    fn payload_slice(&self, range: (usize, usize)) -> Result<&[u8], DFlashSelectorError> {
        self.resident
            .payload()
            .get(range.0..range.1)
            .ok_or_else(|| DFlashSelectorError::Invalid("selector tensor payload extent".into()))
    }

    fn decode_scalar(&self, word: &[u8]) -> Result<f32, DFlashSelectorError> {
        if word.len() != 2 {
            return Err(DFlashSelectorError::Invalid(
                "selector scalar extent".into(),
            ));
        }
        let bits = u16::from_le_bytes([word[0], word[1]]);
        let value = match self.dtype {
            DFlashDType::Bf16 => f32::from_bits(u32::from(bits) << 16),
            DFlashDType::F16 => half::f16::from_bits(bits).to_f32(),
        };
        if value.is_finite() {
            Ok(value)
        } else {
            Err(DFlashSelectorError::Invalid(
                "selector scalar is non-finite".into(),
            ))
        }
    }
}

fn tensor_range(
    root: &Value,
    name: &str,
    expected_shape: &[usize],
    payload_len: usize,
    dtype: DFlashDType,
) -> Result<(usize, usize), DFlashSelectorError> {
    let tensor = root
        .get(name)
        .ok_or_else(|| DFlashSelectorError::Invalid(format!("missing selector tensor {name}")))?;
    let expected_dtype = match dtype {
        DFlashDType::Bf16 => "BF16",
        DFlashDType::F16 => "F16",
    };
    if tensor.get("dtype").and_then(Value::as_str) != Some(expected_dtype) {
        return Err(DFlashSelectorError::Invalid(format!(
            "selector tensor {name} is not {expected_dtype}"
        )));
    }
    let shape = tensor
        .get("shape")
        .and_then(Value::as_array)
        .ok_or_else(|| DFlashSelectorError::Invalid(format!("selector tensor {name} shape")))?;
    let actual_shape: Vec<usize> = shape
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| {
                    DFlashSelectorError::Invalid(format!("selector tensor {name} shape"))
                })
        })
        .collect::<Result<_, _>>()?;
    if actual_shape != expected_shape {
        return Err(DFlashSelectorError::Invalid(format!(
            "selector tensor {name} shape mismatch"
        )));
    }
    let offsets = tensor
        .get("data_offsets")
        .and_then(Value::as_array)
        .ok_or_else(|| DFlashSelectorError::Invalid(format!("selector tensor {name} offsets")))?;
    let start = offsets
        .first()
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| DFlashSelectorError::Invalid(format!("selector tensor {name} start")))?;
    let end = offsets
        .get(1)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| DFlashSelectorError::Invalid(format!("selector tensor {name} end")))?;
    let elements = expected_shape.iter().try_fold(1usize, |acc, value| {
        acc.checked_mul(*value)
            .ok_or_else(|| DFlashSelectorError::Invalid("selector tensor size overflow".into()))
    })?;
    let expected_bytes = elements
        .checked_mul(2)
        .ok_or_else(|| DFlashSelectorError::Invalid("selector tensor byte overflow".into()))?;
    if start >= end || end - start != expected_bytes || end > payload_len {
        return Err(DFlashSelectorError::Invalid(format!(
            "selector tensor {name} extent mismatch"
        )));
    }
    Ok((start, end))
}

#[derive(Clone, Debug, PartialEq)]
pub struct DFlashSelectorBlock {
    /// The committed token immediately before the draft block.
    pub anchor_token: u32,
    /// Candidate token ids for each draft position.
    pub candidate_tokens: Vec<Vec<u32>>,
    /// Target-model unary logits for each candidate at each draft position.
    pub unary_scores: Vec<Vec<f32>>,
    /// Selector hidden projection (the gate) for each draft position.
    pub gates: Vec<Vec<f32>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DFlashSelectorPath {
    pub tokens: Vec<u32>,
    pub selected_candidate_indices: Vec<usize>,
    pub selected_scores: Vec<f32>,
    pub transition_scores: Vec<Vec<f32>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DFlashSelectorError {
    Invalid(String),
}

impl fmt::Display for DFlashSelectorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => write!(f, "invalid DFlash selector input: {message}"),
        }
    }
}

impl std::error::Error for DFlashSelectorError {}

/// Walk one DFlash2 candidate lattice using the exact CPU-side recurrence.
///
/// `predecessor_codebook` and `successor_codebook` are row-major
/// `[vocab_size, selector_rank]` BF16-to-FP32-converted values. The first
/// position uses the anchor as its predecessor; later positions use the
/// previously selected candidate. Accumulation is in f64 and the published
/// score is cast to f32.
pub fn select_dflash2_path(
    block: &DFlashSelectorBlock,
    predecessor_codebook: &[f32],
    successor_codebook: &[f32],
    vocab_size: usize,
    selector_rank: usize,
    selector_top_k: usize,
) -> Result<DFlashSelectorPath, DFlashSelectorError> {
    if vocab_size == 0 || selector_rank == 0 || selector_top_k == 0 {
        return Err(DFlashSelectorError::Invalid(
            "vocab, rank, and top-k must be nonzero".into(),
        ));
    }
    let codebook_len = vocab_size
        .checked_mul(selector_rank)
        .ok_or_else(|| DFlashSelectorError::Invalid("codebook extent overflow".into()))?;
    if predecessor_codebook.len() != codebook_len || successor_codebook.len() != codebook_len {
        return Err(DFlashSelectorError::Invalid(
            "codebook extent does not match vocab*rank".into(),
        ));
    }
    if block.candidate_tokens.len() != block.unary_scores.len()
        || block.candidate_tokens.len() != block.gates.len()
    {
        return Err(DFlashSelectorError::Invalid(
            "candidate, unary, and gate position counts differ".into(),
        ));
    }
    if block.candidate_tokens.is_empty() {
        return Err(DFlashSelectorError::Invalid(
            "selector block must contain at least one position".into(),
        ));
    }
    for position in 0..block.candidate_tokens.len() {
        let candidates = &block.candidate_tokens[position];
        let unary = &block.unary_scores[position];
        let gate = &block.gates[position];
        if candidates.len() != selector_top_k || unary.len() != selector_top_k {
            return Err(DFlashSelectorError::Invalid(format!(
                "position {position} must have exactly {selector_top_k} candidates and unary scores"
            )));
        }
        if gate.len() != selector_rank {
            return Err(DFlashSelectorError::Invalid(format!(
                "position {position} gate length differs from selector rank"
            )));
        }
        for (candidate, score) in candidates.iter().zip(unary) {
            if *candidate as usize >= vocab_size {
                return Err(DFlashSelectorError::Invalid(format!(
                    "position {position} candidate {candidate} exceeds vocabulary"
                )));
            }
            if !score.is_finite() {
                return Err(DFlashSelectorError::Invalid(
                    "unary scores must be finite".into(),
                ));
            }
        }
        if gate.iter().any(|value| !value.is_finite()) {
            return Err(DFlashSelectorError::Invalid(
                "selector gates must be finite".into(),
            ));
        }
    }
    if block.anchor_token as usize >= vocab_size {
        return Err(DFlashSelectorError::Invalid(
            "anchor exceeds vocabulary".into(),
        ));
    }

    let mut predecessor_id = block.anchor_token;
    let mut tokens = Vec::with_capacity(block.candidate_tokens.len());
    let mut selected_candidate_indices = Vec::with_capacity(block.candidate_tokens.len());
    let mut selected_scores = Vec::with_capacity(block.candidate_tokens.len());
    let mut transition_scores = Vec::with_capacity(block.candidate_tokens.len());

    for position in 0..block.candidate_tokens.len() {
        let candidates = &block.candidate_tokens[position];
        let unary = &block.unary_scores[position];
        let gate = &block.gates[position];
        let predecessor_row = row(predecessor_codebook, predecessor_id as usize, selector_rank);
        let conditioned: Vec<f32> = predecessor_row
            .iter()
            .zip(gate)
            .map(|(value, multiplier)| value * multiplier)
            .collect();
        let mut scores = Vec::with_capacity(selector_top_k);
        for (candidate, unary_score) in candidates.iter().zip(unary) {
            let successor_row = row(successor_codebook, *candidate as usize, selector_rank);
            let mut dot = 0.0_f64;
            for (left, right) in conditioned.iter().zip(successor_row) {
                dot += f64::from(*left) * f64::from(*right);
            }
            let score = (f64::from(*unary_score) + dot) as f32;
            if !score.is_finite() {
                return Err(DFlashSelectorError::Invalid(
                    "selector score is not finite".into(),
                ));
            }
            scores.push(score);
        }
        let selected = scores
            .iter()
            .enumerate()
            .max_by(|(left_index, left), (right_index, right)| {
                left.partial_cmp(right)
                    .expect("finite selector scores")
                    .then_with(|| right_index.cmp(left_index))
            })
            .map(|(index, _)| index)
            .ok_or_else(|| DFlashSelectorError::Invalid("empty score row".into()))?;
        predecessor_id = candidates[selected];
        tokens.push(predecessor_id);
        selected_candidate_indices.push(selected);
        selected_scores.push(scores[selected]);
        transition_scores.push(scores);
    }

    Ok(DFlashSelectorPath {
        tokens,
        selected_candidate_indices,
        selected_scores,
        transition_scores,
    })
}

fn row(codebook: &[f32], token: usize, rank: usize) -> &[f32] {
    let start = token * rank;
    &codebook[start..start + rank]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn walks_anchor_then_previous_candidate() {
        let block = DFlashSelectorBlock {
            anchor_token: 0,
            candidate_tokens: vec![vec![1, 2], vec![3, 4]],
            unary_scores: vec![vec![0.0, 0.0], vec![0.0, 0.0]],
            gates: vec![vec![1.0, 0.0], vec![1.0, 0.0]],
        };
        let predecessor = vec![
            1.0, 2.0, // anchor 0
            0.0, 0.0, // token 1
            0.0, 0.0, // token 2
            0.0, 0.0, // token 3
            0.0, 0.0, // token 4
        ];
        let successor = vec![
            0.0, 0.0, // token 0
            0.0, 0.0, // token 1
            2.0, 0.0, // token 2
            0.0, 0.0, // token 3
            0.0, 0.0, // token 4
        ];
        let result = select_dflash2_path(&block, &predecessor, &successor, 5, 2, 2).unwrap();
        assert_eq!(result.tokens, vec![2, 3]);
        assert_eq!(result.selected_candidate_indices, vec![1, 0]);
        assert_eq!(result.transition_scores[0], vec![0.0, 2.0]);
        assert_eq!(result.transition_scores[1], vec![0.0, 0.0]);
    }

    #[test]
    fn rejects_wrong_shape_and_nonfinite_inputs() {
        let block = DFlashSelectorBlock {
            anchor_token: 0,
            candidate_tokens: vec![vec![1]],
            unary_scores: vec![vec![f32::NAN]],
            gates: vec![vec![1.0]],
        };
        let error = select_dflash2_path(&block, &[1.0, 0.0], &[1.0, 0.0], 2, 1, 1).unwrap_err();
        assert!(error.to_string().contains("unary scores"));
    }

    #[test]
    fn numeric_signed_zero_tie_keeps_first_candidate() {
        let block = DFlashSelectorBlock {
            anchor_token: 0,
            candidate_tokens: vec![vec![1, 2]],
            unary_scores: vec![vec![0.0, 0.0]],
            gates: vec![vec![1.0e-30]],
        };
        let path =
            select_dflash2_path(&block, &[1.0, 0.0, 0.0], &[0.0, -1.0e-20, 1.0e-20], 3, 1, 2)
                .unwrap();
        assert_eq!(path.transition_scores[0][0].to_bits(), (-0.0_f32).to_bits());
        assert_eq!(path.tokens, [1]);
    }
}
