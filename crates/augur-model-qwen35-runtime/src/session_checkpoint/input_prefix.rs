//! A genuine consumed-input boundary, distinct from a generated conversation.
//!
//! There is deliberately no generated-token ledger, EOS, pending output, old
//! logits or publication flag here. A native owner must independently capture
//! and seal all target planes and initialized features before this metadata is
//! usable. Descriptor/lookup tests do not establish a CUDA hit or semantic reuse.

use super::metadata::{
    target_components, DraftRestoreRecipe, FeatureExtent, HistoryMode, ModelBindings,
    NativeMathPolicy, SamplingPolicy, CAPACITY, HISTORY, MAX_DESCRIPTOR_BYTES, VOCAB,
};
use super::{invalid, require, sha256, valid_digest, TransferPlan};
use crate::{GrowingHistory, GrowingHistoryConfig};
use serde::{Deserialize, Serialize};
use std::io;

pub const INPUT_PREFIX_SCHEMA: &str = "cq-qwen35-input-prefix-descriptor-v1";

/// Caller-supplied live admission, not copied from a checkpoint. Borrowing the
/// exact request prevents a change between lookup and native restore.
pub struct InputPrefixRestoreAdmission<'a> {
    pub bindings: &'a ModelBindings,
    pub math: &'a NativeMathPolicy,
    pub new_request_id: &'a str,
    pub request_input: &'a [u32],
    pub output_reserve: usize,
}

/// New first-turn publication admission. This is not a saved session ledger.
pub struct InputPrefixGenerationAdmission<'a> {
    pub bindings: &'a ModelBindings,
    pub session_id: &'a str,
    pub max_new_tokens: usize,
}

impl InputPrefixGenerationAdmission<'_> {
    pub fn validate(&self, descriptor: &InputPrefixDescriptor) -> io::Result<()> {
        descriptor.validate()?;
        require(
            self.bindings == &descriptor.bindings,
            "input-prefix generation bindings",
        )?;
        require(
            !self.session_id.is_empty()
                && self.session_id.len() <= 64
                && self
                    .session_id
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b)),
            "input-prefix generated session label",
        )?;
        let limit = crate::resident_session_profile::ResidentSessionProfile::Resident
            .output_budget(Some(self.max_new_tokens), true)
            .map_err(invalid)?;
        require(
            descriptor
                .consumed_input
                .len()
                .checked_add(limit)
                .and_then(|end| end.checked_add(crate::resident_session_profile::VERIFY_WIDTH - 2))
                .is_some_and(|end| end <= CAPACITY),
            "input-prefix generation verification reserve",
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputPrefixDescriptor {
    pub schema: String,
    pub bindings: ModelBindings,
    /// Exact already-consumed input IDs. Never re-tokenized assistant text.
    pub consumed_input: Vec<u32>,
    pub features: FeatureExtent,
    pub math: NativeMathPolicy,
    pub draft_restore: DraftRestoreRecipe,
    pub sampling: SamplingPolicy,
    /// Provenance only, not a capability that revives the original owner.
    pub origin_request_sha256: String,
}

impl InputPrefixDescriptor {
    pub fn validate(&self) -> io::Result<()> {
        require(self.schema == INPUT_PREFIX_SCHEMA, "input-prefix schema")?;
        self.bindings.validate()?;
        self.math.validate()?;
        let cursor = self.consumed_input.len();
        require(
            (1..=CAPACITY).contains(&cursor) && self.consumed_input.iter().all(|&id| id < VOCAB),
            "input-prefix token extent or vocabulary",
        )?;
        require(
            valid_digest(&self.origin_request_sha256),
            "input-prefix request provenance",
        )?;
        require(
            self.features.physical_rows == HISTORY && self.features.rows == cursor.min(HISTORY),
            "input-prefix initialized feature extent",
        )?;
        let generation = match &self.features.mode {
            HistoryMode::Growing2048 { generation } => *generation,
            HistoryMode::Fixed2048 => return Err(invalid("input-prefix requires growing history")),
        };
        // Reuse the existing ring/counter admission, not a second arithmetic
        // authority for the initialized target-feature window.
        let history = GrowingHistory::from_committed_extent(
            GrowingHistoryConfig::default_2048(),
            self.features.start,
            self.features.rows,
            generation,
        )
        .map_err(io::Error::other)?;
        require(
            history.cursor() == cursor,
            "input-prefix feature cursor differs from input",
        )
    }

    pub fn encode(&self) -> io::Result<Vec<u8>> {
        self.validate()?;
        let bytes = serde_json::to_vec(self).map_err(io::Error::other)?;
        require(
            bytes.len() <= MAX_DESCRIPTOR_BYTES,
            "input-prefix descriptor too large",
        )?;
        Ok(bytes)
    }

    pub fn decode(
        bytes: &[u8],
        expected_bindings: &ModelBindings,
        expected_math: &NativeMathPolicy,
    ) -> io::Result<Self> {
        require(
            bytes.len() <= MAX_DESCRIPTOR_BYTES,
            "input-prefix descriptor too large",
        )?;
        expected_bindings.validate()?;
        expected_math.validate()?;
        let descriptor: Self = serde_json::from_slice(bytes).map_err(io::Error::other)?;
        descriptor.validate()?;
        require(
            &descriptor.bindings == expected_bindings && &descriptor.math == expected_math,
            "input-prefix differs from live bindings/math",
        )?;
        require(
            descriptor.encode()? == bytes,
            "input-prefix canonical bytes",
        )?;
        Ok(descriptor)
    }

    /// Same full KV/GDN/features inventory as session persistence, with a
    /// distinct descriptor role and binding domain. It grants no owner rights.
    pub fn transfer_plan(&self) -> io::Result<TransferPlan> {
        let bytes = self.encode()?;
        let components = target_components(
            "input_prefix.metadata",
            bytes.len(),
            self.consumed_input.len(),
            self.features.rows,
        )?;
        let mut binding = b"CQ-QWEN35-INPUT-PREFIX-DESCRIPTOR-V1\0".to_vec();
        binding.extend_from_slice(&bytes);
        let plan = TransferPlan {
            binding_sha256: sha256(&binding),
            components,
        };
        plan.validate()?;
        Ok(plan)
    }

    /// Verify an entire saved checkpoint against the requested canonical input.
    /// Matching part of a checkpoint never allows truncating its GDN state.
    /// The borrowed request prevents a caller mutating tokens after planning.
    pub fn lookup_suffix<'a>(
        &'a self,
        expected_bindings: &ModelBindings,
        expected_math: &NativeMathPolicy,
        request_input: &'a [u32],
        output_reserve: usize,
    ) -> io::Result<Option<InputPrefixSuffix<'a>>> {
        self.validate()?;
        require(
            &self.bindings == expected_bindings && &self.math == expected_math,
            "input-prefix lookup identity mismatch",
        )?;
        require(
            !request_input.is_empty()
                && output_reserve > 0
                && request_input
                    .len()
                    .checked_add(output_reserve)
                    .is_some_and(|n| n <= CAPACITY)
                && request_input.iter().all(|&id| id < VOCAB),
            "input-prefix request/reserve exceeds capacity or vocabulary",
        )?;
        // Different prompts and a checkpoint beyond the common prefix are
        // ordinary misses. An equal/no-suffix prompt also misses this first
        // input-only path: it must not replay old logits to manufacture a hit.
        if request_input.len() <= self.consumed_input.len()
            || !request_input.starts_with(&self.consumed_input)
        {
            return Ok(None);
        }
        Ok(Some(InputPrefixSuffix {
            descriptor: self,
            request_input,
            output_reserve,
        }))
    }
}

/// Validated CPU input plan only. Installing it still requires native payload
/// verification, fresh owner/stream authority and an actual suffix submission.
#[derive(Debug)]
pub struct InputPrefixSuffix<'a> {
    descriptor: &'a InputPrefixDescriptor,
    request_input: &'a [u32],
    output_reserve: usize,
}

impl InputPrefixSuffix<'_> {
    pub fn restored_tokens(&self) -> usize {
        self.descriptor.consumed_input.len()
    }
    pub fn suffix(&self) -> &[u32] {
        &self.request_input[self.restored_tokens()..]
    }
    pub fn final_input_tokens(&self) -> usize {
        self.request_input.len()
    }
    pub fn output_reserve(&self) -> usize {
        self.output_reserve
    }
}

#[cfg(test)]
#[path = "input_prefix_tests.rs"]
mod tests;
