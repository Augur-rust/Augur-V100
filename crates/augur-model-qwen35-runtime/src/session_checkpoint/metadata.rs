//! Exact native-session descriptor. This is data, not an owner capability.
//!
//! It describes the initialized target payload and the explicit canonical draft
//! rebuild recipe. A decoder must independently match live artifact/build policy,
//! create fresh handles/streams/epochs and verify the real rebuild/continuation.
//! HTTP request state and publication authority are not reconstructed here.

use super::{invalid, require, sha256, valid_digest, Component, TransferPlan};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::io;

pub const SESSION_SCHEMA: &str = "cq-qwen35-native-session-ssd-descriptor-v1";
pub const MAX_DESCRIPTOR_BYTES: usize = 4 * 1024 * 1024;
pub const CAPACITY: usize = 196608;
pub const HISTORY: usize = 2048;
pub const VOCAB: u32 = 248320;
pub const KV_ROW_BYTES: u64 = 1088;
pub const FEATURE_ROW_BYTES: u64 = 5 * 5120 * 4;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelBindings {
    pub model_sha256: String,
    pub q8_draft_sha256: String,
    pub vision_sha256: String,
    /// Exact resident Vision payload digest, distinct from the source container.
    pub vision_payload_sha256: String,
    pub model_config_sha256: String,
    pub draft_config_sha256: String,
    pub tokenizer_sha256: String,
    pub chat_template_sha256: String,
    pub generation_config_sha256: String,
    /// SHA256 of canonical serialized build_identity::identity(), not a filename.
    pub frozen_build_sha256: String,
    pub numerical_contract_sha256: String,
    pub state_layout_sha256: String,
    pub namespace_sha256: String,
}

impl ModelBindings {
    pub fn validate(&self) -> io::Result<()> {
        for digest in [
            &self.model_sha256,
            &self.q8_draft_sha256,
            &self.vision_sha256,
            &self.vision_payload_sha256,
            &self.model_config_sha256,
            &self.draft_config_sha256,
            &self.tokenizer_sha256,
            &self.chat_template_sha256,
            &self.generation_config_sha256,
            &self.frozen_build_sha256,
            &self.numerical_contract_sha256,
            &self.state_layout_sha256,
            &self.namespace_sha256,
        ] {
            require(
                valid_digest(digest),
                "missing/noncanonical model or session binding",
            )?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UserSuffixRange {
    pub start: usize,
    pub end: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeLedger {
    /// Provenance only. Restore must mint a different live session handle.
    pub origin_session_id: String,
    pub consumed: Vec<u32>,
    /// Native committed model outputs, including EOS where applicable; these are
    /// NOT the rendered HTTP visible-text ledger and never teacher-forced input.
    pub generated: Vec<u32>,
    pub user_suffix_ranges: Vec<UserSuffixRange>,
    pub pending: u32,
    pub pending_published: bool,
    /// Logical model transaction counter, never a CUDA submission epoch/handle.
    pub generation: u64,
    pub revision: u64,
    pub capacity: usize,
}

impl NativeLedger {
    /// Reconstruct the ledger partition and compare every consumed model output.
    /// This rejects the common error of treating a visible pending token as an
    /// already committed KV row, or treating a teacher-forced suffix as output.
    /// Returns the original prompt length from that exact partition, never padding.
    pub fn validate(&self) -> io::Result<usize> {
        require(
            !self.origin_session_id.is_empty()
                && self.origin_session_id.len() <= 64
                && self
                    .origin_session_id
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b)),
            "origin session label",
        )?;
        require(
            self.capacity == CAPACITY
                && !self.consumed.is_empty()
                && self.consumed.len() <= CAPACITY
                && !self.generated.is_empty()
                && self.generated.len() <= CAPACITY + 1,
            "native ledger extent",
        )?;
        require(
            (1..u64::MAX).contains(&self.generation) && (1..u64::MAX).contains(&self.revision),
            "logical generation/revision exhausted or absent",
        )?;
        require(
            self.pending < VOCAB
                && self
                    .consumed
                    .iter()
                    .chain(&self.generated)
                    .all(|&token| token < VOCAB),
            "native ledger token outside vocabulary",
        )?;
        require(
            self.user_suffix_ranges.len() <= CAPACITY
                && (self.user_suffix_ranges.len() as u64) < self.revision,
            "suffix/revision bounds",
        )?;
        if self.pending_published {
            require(
                self.generated.last() == Some(&self.pending),
                "published pending output drift",
            )?;
        } else {
            require(
                self.user_suffix_ranges
                    .last()
                    .is_some_and(|r| r.end == self.consumed.len()),
                "unpublished prediction must follow the committed user suffix",
            )?;
        }

        let mut user_bytes = 0usize;
        let mut prior_end = 0usize;
        for range in &self.user_suffix_ranges {
            require(
                range.start >= prior_end
                    && range.start < range.end
                    && range.end <= self.consumed.len()
                    && range.end - range.start <= HISTORY,
                "user suffix ranges overlap, overflow or exceed native append bound",
            )?;
            user_bytes = user_bytes
                .checked_add(range.end - range.start)
                .ok_or_else(|| invalid("suffix token count overflow"))?;
            prior_end = range.end;
        }
        let generated_consumed = self.generated.len() - usize::from(self.pending_published);
        let prompt = self
            .consumed
            .len()
            .checked_sub(generated_consumed)
            .and_then(|n| n.checked_sub(user_bytes))
            .ok_or_else(|| invalid("ledger partition overflow"))?;
        require(prompt > 0, "missing original prompt")?;
        let mut cursor = prompt;
        let mut output = 0usize;
        for range in &self.user_suffix_ranges {
            require(
                range.start >= cursor,
                "user suffix overlaps original prompt",
            )?;
            let next = output
                .checked_add(range.start - cursor)
                .ok_or_else(|| invalid("output partition overflow"))?;
            require(
                next <= generated_consumed
                    && self.consumed[cursor..range.start] == self.generated[output..next],
                "model output path differs from consumed KV history",
            )?;
            output = next;
            cursor = range.end;
        }
        require(
            self.consumed.len() - cursor == generated_consumed - output
                && self.consumed[cursor..] == self.generated[output..generated_consumed],
            "final model output path differs from consumed KV history",
        )?;
        Ok(prompt)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum HistoryMode {
    Fixed2048,
    Growing2048 { generation: u64 },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeatureExtent {
    pub start: usize,
    pub rows: usize,
    pub physical_rows: usize,
    pub mode: HistoryMode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DraftRestoreRecipe {
    /// No old graph, cached candidate, scratch bytes or stream authority is reused.
    /// A real new canonical Q8 block must rebuild from the exact saved target ring.
    /// This recipe's numerical equivalence is NOT established by descriptor tests.
    FreshCanonicalQ8FromTargetFeaturesV1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplingPolicy {
    /// Stochastic sampling would require a separately versioned RNG-state contract.
    GreedyArgmaxV1,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeMathPolicy {
    /// Representation identity, independent of payload dimensions. Missing in
    /// old descriptors means unrotated; new rotated bytes may never use it.
    #[serde(default, skip_serializing_if = "FullAttentionKvBasis::is_unrotated")]
    pub full_attention_kv_basis: FullAttentionKvBasis,
    pub exact_projection_tiles: bool,
    pub exact_output_head_batch: bool,
    pub draft_top16_batch: bool,
    pub parallel_receipt_digest: bool,
    pub batched_target_verify: bool,
    pub reuse_batched_plan: bool,
    pub batched_suffix: bool,
    pub retain_full_accept_state: bool,
    pub share_target_rope: bool,
    pub batched_draft_context: bool,
    pub batched_draft_block: bool,
    pub draft_q8_tile4: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FullAttentionKvBasis {
    #[default]
    UnrotatedQ8V3V4,
    Hadamard256KeyHadamard64ValueV1,
    Hadamard256KeyHadamard64ValuePlanarV2,
}

impl FullAttentionKvBasis {
    pub fn from_rotated(rotated: bool) -> Self {
        if rotated { Self::Hadamard256KeyHadamard64ValueV1 } else { Self::UnrotatedQ8V3V4 }
    }
    pub fn from_layout(rotated: bool, planar: bool) -> Self {
        if rotated && planar { Self::Hadamard256KeyHadamard64ValuePlanarV2 }
        else { Self::from_rotated(rotated) }
    }
    pub fn is_planar(&self) -> bool { matches!(self, Self::Hadamard256KeyHadamard64ValuePlanarV2) }
    pub fn is_unrotated(&self) -> bool { matches!(self, Self::UnrotatedQ8V3V4) }
    pub fn is_rotated(&self) -> bool { !self.is_unrotated() }
}

impl NativeMathPolicy {
    pub fn validate(&self) -> io::Result<()> {
        require(
            !self.reuse_batched_plan || self.batched_target_verify,
            "retained verify plan requires actual batched execution",
        )?;
        require(
            !self.batched_suffix || (self.batched_target_verify && self.reuse_batched_plan),
            "batched suffix requires a retained canonical executor",
        )?;
        require(
            !(self.exact_projection_tiles || self.exact_output_head_batch)
                || self.batched_target_verify,
            "exact scheduling requires actual batched target",
        )?;
        require(
            !self.draft_q8_tile4 || (self.batched_draft_context && self.batched_draft_block),
            "draft tile4 requires both row-launch policies",
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionDescriptor {
    pub schema: String,
    pub bindings: ModelBindings,
    pub ledger: NativeLedger,
    pub features: FeatureExtent,
    pub math: NativeMathPolicy,
    pub draft_restore: DraftRestoreRecipe,
    pub sampling: SamplingPolicy,
    pub eos_token_ids: Vec<u32>,
    /// Original native request digest, which also binds its initial prompt and
    /// profile. Restoring it does not revive the old request or CUDA owner.
    pub native_request_identity: String,
}

/// Live controller admission; never constructed by trusting a checkpoint alone.
/// The caller binds independently selected artifacts, math policy and a new handle.
#[derive(Clone, Debug)]
pub struct NativeRestoreAdmission {
    pub new_session_id: String,
    pub bindings: ModelBindings,
    pub math: NativeMathPolicy,
}

impl NativeRestoreAdmission {
    pub fn validate_descriptor(&self, descriptor: &SessionDescriptor) -> io::Result<()> {
        descriptor.validate()?;
        self.bindings.validate()?;
        self.math.validate()?;
        require(
            !self.new_session_id.is_empty()
                && self.new_session_id.len() <= 64
                && self
                    .new_session_id
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b))
                && self.new_session_id != descriptor.ledger.origin_session_id,
            "restore requires a distinct valid live session handle",
        )?;
        require(
            self.bindings == descriptor.bindings && self.math == descriptor.math,
            "restore admission differs from live bindings/math",
        )
    }
}

impl SessionDescriptor {
    pub fn validate(&self) -> io::Result<()> {
        require(
            self.schema == SESSION_SCHEMA,
            "native session descriptor schema",
        )?;
        self.bindings.validate()?;
        self.math.validate()?;
        self.ledger.validate()?;
        let cursor = self.ledger.consumed.len();
        require(
            self.features.physical_rows == HISTORY
                && self.features.rows == cursor.min(HISTORY)
                && self.features.start == cursor - self.features.rows,
            "initialized feature extent cannot include fake/future rows",
        )?;
        match self.features.mode {
            HistoryMode::Fixed2048 => {
                require(cursor >= HISTORY, "fixed history not initialized")?;
                require(
                    !self.math.batched_suffix,
                    "batched suffix requires growing history",
                )?;
            }
            HistoryMode::Growing2048 { generation } => {
                require(generation < u64::MAX, "history generation exhausted")?
            }
        }
        require(
            valid_digest(&self.native_request_identity),
            "native request identity",
        )?;
        let ids: BTreeSet<_> = self.eos_token_ids.iter().copied().collect();
        require(
            !ids.is_empty()
                && ids.len() <= 32
                && ids.len() == self.eos_token_ids.len()
                && ids.iter().all(|&id| id < VOCAB),
            "EOS identity or vocabulary",
        )
    }

    pub fn encode(&self) -> io::Result<Vec<u8>> {
        self.validate()?;
        let bytes = serde_json::to_vec(self).map_err(io::Error::other)?;
        require(bytes.len() <= MAX_DESCRIPTOR_BYTES, "descriptor byte quota")?;
        Ok(bytes)
    }

    pub fn decode(
        bytes: &[u8],
        expected_bindings: &ModelBindings,
        expected_math: &NativeMathPolicy,
    ) -> io::Result<Self> {
        require(
            !bytes.is_empty() && bytes.len() <= MAX_DESCRIPTOR_BYTES,
            "descriptor byte quota",
        )?;
        expected_bindings.validate()?;
        let descriptor: Self = serde_json::from_slice(bytes).map_err(io::Error::other)?;
        descriptor.validate()?;
        require(
            &descriptor.bindings == expected_bindings && &descriptor.math == expected_math,
            "live artifacts/build/namespace/numerical policy differ from checkpoint",
        )?;
        require(descriptor.encode()? == bytes, "descriptor canonical bytes")?;
        Ok(descriptor)
    }

    /// Complete initialized TARGET inventory plus descriptor. Draft-private bytes
    /// are deliberately absent only because the explicit fresh-rebuild recipe is
    /// mandatory; this method does not qualify that recipe or publish a session.
    pub fn transfer_plan(&self) -> io::Result<TransferPlan> {
        let descriptor = self.encode()?;
        let components = target_components(
            "session.metadata",
            descriptor.len(),
            self.ledger.consumed.len(),
            self.features.rows,
        )?;
        let mut binding = b"CQ-QWEN35-NATIVE-SESSION-DESCRIPTOR-V1\0".to_vec();
        binding.extend_from_slice(&descriptor);
        let plan = TransferPlan {
            binding_sha256: sha256(&binding),
            components,
        };
        plan.validate()?;
        Ok(plan)
    }
}

/// Shared physical target inventory. This does not reuse a generated ledger or
/// imply that a prefix descriptor has been captured by a native owner.
pub(super) fn target_components(
    metadata_name: &str,
    metadata_bytes: usize,
    cursor: usize,
    feature_rows: usize,
) -> io::Result<Vec<Component>> {
    require(
        (1..=CAPACITY).contains(&cursor) && feature_rows == cursor.min(HISTORY),
        "target inventory requires actual initialized extent",
    )?;
    let mut components = Vec::with_capacity(130);
    components.push(Component {
        name: metadata_name.into(),
        bytes: metadata_bytes as u64,
    });
    for layer in 0..64 {
        let planes = if layer % 4 == 3 {
            [
                ("key", cursor as u64 * KV_ROW_BYTES),
                ("value", cursor as u64 * KV_ROW_BYTES),
            ]
        } else {
            [("conv_history", 30720 * 4), ("recurrent", 786432 * 4)]
        };
        for (name, bytes) in planes {
            components.push(Component {
                name: format!("target.layer{layer:02}.{name}"),
                bytes,
            });
        }
    }
    components.push(Component {
        name: "target.features".into(),
        bytes: feature_rows as u64 * FEATURE_ROW_BYTES,
    });
    Ok(components)
}

#[cfg(test)]
#[path = "metadata_tests.rs"]
mod tests;
