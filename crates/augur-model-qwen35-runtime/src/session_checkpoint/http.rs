//! HTTP publication metadata, not a CUDA capability or a client delivery ACK.
//!
//! Checkpoints are explicit completed-turn boundaries. Native consumed/generated
//! histories remain authoritative. The outer record binds every emitted turn to
//! that descriptor and requires a new owner/session at restore. Disk publication,
//! tokenizer validation and real HTTP recovery are separate obligations.

use super::metadata::{
    ModelBindings, NativeMathPolicy, NativeRestoreAdmission, SessionDescriptor, CAPACITY, HISTORY,
    VOCAB,
};
use super::{invalid, require, sha256, valid_digest};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::io;

#[path = "http_seal.rs"]
mod seal;
pub use seal::{StatePlaneSeal, TargetStateSeal};

pub const HTTP_SCHEMA: &str = "cq-qwen35-http-session-checkpoint-v1";
pub const MAX_HTTP_METADATA_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_TURNS: usize = 64;
pub const MAX_VISIBLE_TEXT_BYTES: usize = 8 * 1024 * 1024;

fn valid_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 64
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-_".contains(&b))
}

fn input_digest(tokens: &[u32]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(b"QWEN35-SESSION-CONSUMED-U32LE-V1\0");
    for token in tokens {
        h.update(token.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    Eos,
    MaxNewTokens,
}

/// Producer-emitted terminal state. A successful socket write does not prove
/// the client received it, so this record never claims an application-level ACK.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublishedTurn {
    pub owner_id: String,
    pub request_id: String,
    pub request_sha256: String,
    pub native_report_sha256: String,
    pub native_revision: u64,
    pub terminal_sequence: u64,
    pub token_ids: Vec<u32>,
    pub visible_token_ids: Vec<u32>,
    pub text: String,
    pub text_sha256: String,
    pub stop_reason: StopReason,
}

impl PublishedTurn {
    fn validate(&self, eos: &[u32]) -> io::Result<()> {
        require(
            valid_id(&self.owner_id) && valid_id(&self.request_id),
            "HTTP turn owner/request ID",
        )?;
        require(
            (1..u64::MAX).contains(&self.native_revision),
            "HTTP turn revision",
        )?;
        for digest in [
            &self.request_sha256,
            &self.native_report_sha256,
            &self.text_sha256,
        ] {
            require(valid_digest(digest), "HTTP turn digest")?;
        }
        require(
            !self.token_ids.is_empty()
                && self.token_ids.len() <= CAPACITY + 1
                && self.token_ids.iter().all(|&t| t < VOCAB),
            "HTTP turn token extent/vocabulary",
        )?;
        let visible = match self.stop_reason {
            StopReason::Eos => {
                require(
                    eos.contains(self.token_ids.last().unwrap()),
                    "HTTP EOS terminal token",
                )?;
                &self.token_ids[..self.token_ids.len() - 1]
            }
            StopReason::MaxNewTokens => &self.token_ids[..],
        };
        require(
            !visible.iter().any(|t| eos.contains(t)) && visible == self.visible_token_ids,
            "HTTP visible tokens differ from native publication",
        )?;
        require(
            self.terminal_sequence > 0
                && self.terminal_sequence <= self.visible_token_ids.len() as u64 + 1
                && (self.visible_token_ids.is_empty() == (self.terminal_sequence == 1)),
            "HTTP terminal/text frame sequence",
        )?;
        require(
            self.text.len() <= MAX_VISIBLE_TEXT_BYTES
                && (self.visible_token_ids.is_empty() == self.text.is_empty())
                && sha256(self.text.as_bytes()) == self.text_sha256,
            "HTTP text extent/digest",
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HttpVisibleLedger {
    pub origin_session_id: String,
    pub revision: u64,
    pub generation: u64,
    pub consumed_token_count: usize,
    pub consumed_input_digest: String,
    pub generated_token_count: usize,
    pub pending_token: u32,
    pub pending_token_published: bool,
    pub turns: Vec<PublishedTurn>,
}

impl HttpVisibleLedger {
    fn validate(&self, eos: &[u32]) -> io::Result<()> {
        require(valid_id(&self.origin_session_id), "HTTP origin session ID")?;
        require(
            (1..u64::MAX).contains(&self.revision) && (1..u64::MAX).contains(&self.generation),
            "HTTP revision/generation",
        )?;
        require(
            (1..=CAPACITY).contains(&self.consumed_token_count)
                && (1..=CAPACITY + 1).contains(&self.generated_token_count),
            "HTTP ledger extent",
        )?;
        require(
            self.pending_token < VOCAB && self.pending_token_published,
            "HTTP checkpoint requires a completed-turn pending boundary",
        )?;
        require(
            valid_digest(&self.consumed_input_digest),
            "HTTP consumed digest",
        )?;
        require(
            !self.turns.is_empty() && self.turns.len() <= MAX_TURNS,
            "HTTP turn quota",
        )?;
        let mut ids = BTreeSet::new();
        let (mut generated, mut text_bytes, mut revision) = (0usize, 0usize, 0u64);
        for turn in &self.turns {
            turn.validate(eos)?;
            require(
                ids.insert((&turn.owner_id, &turn.request_id)),
                "duplicate HTTP owner/request pair",
            )?;
            require(turn.native_revision > revision, "HTTP turn revision order")?;
            revision = turn.native_revision;
            generated = generated
                .checked_add(turn.token_ids.len())
                .ok_or_else(|| invalid("HTTP token overflow"))?;
            text_bytes = text_bytes
                .checked_add(turn.text.len())
                .ok_or_else(|| invalid("HTTP text overflow"))?;
            require(
                generated <= CAPACITY + 1 && text_bytes <= MAX_VISIBLE_TEXT_BYTES,
                "HTTP aggregate quota",
            )?;
        }
        require(
            generated == self.generated_token_count
                && revision == self.revision
                && self.turns.last().and_then(|t| t.token_ids.last()) == Some(&self.pending_token),
            "HTTP generated/revision/pending conservation",
        )
    }
}

/// Data-only, externally hash-bound envelope. SHA addresses are opaque names,
/// never paths. Native bytes are sealed first and this outer record last.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HttpSessionCheckpoint {
    pub schema: String,
    pub checkpoint_id: String,
    pub origin_owner_id: String,
    pub origin_owner_epoch: u64,
    pub native_manifest_sha256: String,
    pub native_transfer_binding_sha256: String,
    pub native_descriptor_sha256: String,
    pub target_seal: TargetStateSeal,
    pub bindings: ModelBindings,
    pub math: NativeMathPolicy,
    pub eos_token_ids: Vec<u32>,
    pub ledger: HttpVisibleLedger,
    pub capacity: usize,
    pub context_rows: usize,
}

impl HttpSessionCheckpoint {
    pub fn validate(&self) -> io::Result<()> {
        require(self.schema == HTTP_SCHEMA, "HTTP checkpoint schema")?;
        require(
            valid_digest(&self.checkpoint_id)
                && valid_id(&self.origin_owner_id)
                && (1..u64::MAX).contains(&self.origin_owner_epoch),
            "HTTP publication identity",
        )?;
        for d in [
            &self.native_manifest_sha256,
            &self.native_transfer_binding_sha256,
            &self.native_descriptor_sha256,
        ] {
            require(valid_digest(d), "HTTP native binding digest")?;
        }
        self.bindings.validate()?;
        self.math.validate()?;
        let eos: BTreeSet<_> = self.eos_token_ids.iter().copied().collect();
        require(
            !eos.is_empty()
                && eos.len() == self.eos_token_ids.len()
                && eos.len() <= 32
                && eos.iter().all(|&t| t < VOCAB),
            "HTTP EOS identity",
        )?;
        self.ledger.validate(&self.eos_token_ids)?;
        self.target_seal
            .validate(self.ledger.consumed_token_count)?;
        require(
            self.capacity == CAPACITY && self.context_rows == HISTORY,
            "HTTP capacity/layout",
        )
    }

    pub fn encode(&self) -> io::Result<Vec<u8>> {
        self.validate()?;
        let bytes = serde_json::to_vec(self).map_err(io::Error::other)?;
        require(
            bytes.len() <= MAX_HTTP_METADATA_BYTES,
            "HTTP metadata byte quota",
        )?;
        Ok(bytes)
    }

    pub fn decode(bytes: &[u8], expected_sha256: &str) -> io::Result<Self> {
        require(
            !bytes.is_empty() && bytes.len() <= MAX_HTTP_METADATA_BYTES,
            "HTTP metadata byte quota",
        )?;
        require(
            valid_digest(expected_sha256) && sha256(bytes) == expected_sha256,
            "external HTTP metadata SHA",
        )?;
        let value: Self = serde_json::from_slice(bytes).map_err(io::Error::other)?;
        require(value.encode()? == bytes, "HTTP metadata canonical bytes")?;
        Ok(value)
    }

    /// Cross-check with the separately authenticated native descriptor, never
    /// a caller's token history. This does not publish a restored CUDA session.
    pub fn validate_native(
        &self,
        descriptor: &SessionDescriptor,
        manifest_sha256: &str,
    ) -> io::Result<()> {
        self.validate()?;
        descriptor.validate()?;
        self.target_seal.validate_descriptor(descriptor)?;
        require(
            self.native_manifest_sha256 == manifest_sha256
                && self.native_descriptor_sha256 == sha256(&descriptor.encode()?)
                && self.native_transfer_binding_sha256
                    == descriptor.transfer_plan()?.binding_sha256,
            "HTTP/native content bindings",
        )?;
        require(
            self.bindings == descriptor.bindings
                && self.math == descriptor.math
                && self.eos_token_ids == descriptor.eos_token_ids,
            "HTTP/native model/math/EOS identity",
        )?;
        let n = &descriptor.ledger;
        let h = &self.ledger;
        require(
            h.origin_session_id == n.origin_session_id
                && h.revision == n.revision
                && h.generation == n.generation
                && h.consumed_token_count == n.consumed.len()
                && h.consumed_input_digest == input_digest(&n.consumed)
                && h.generated_token_count == n.generated.len()
                && h.pending_token == n.pending
                && h.pending_token_published == n.pending_published,
            "HTTP/native exact ledger boundary",
        )?;
        require(
            h.turns
                .iter()
                .flat_map(|t| &t.token_ids)
                .eq(n.generated.iter()),
            "HTTP emitted turns differ from native generated tokens",
        )?;
        // The serial service permits one initial generation, then exactly one
        // suffix/decode per turn. A save is never taken mid-suffix or mid-stream.
        require(
            h.turns.len() == n.user_suffix_ranges.len() + 1
                && h.turns
                    .iter()
                    .enumerate()
                    .all(|(i, t)| t.native_revision == 2 * i as u64 + 1),
            "HTTP completed-turn/native suffix partition",
        )
    }

    pub fn validate_text<F>(&self, mut decode: F) -> io::Result<()>
    where
        F: FnMut(&[u32]) -> io::Result<String>,
    {
        self.validate()?;
        for turn in &self.ledger.turns {
            require(
                decode(&turn.visible_token_ids)? == turn.text,
                "HTTP tokenizer/text mismatch",
            )?;
        }
        Ok(())
    }

    pub fn validate_restore(
        &self,
        new_owner_id: &str,
        new_owner_epoch: u64,
        admission: &NativeRestoreAdmission,
        descriptor: &SessionDescriptor,
        expected_native_manifest_sha256: &str,
    ) -> io::Result<()> {
        self.validate_native(descriptor, expected_native_manifest_sha256)?;
        admission.validate_descriptor(descriptor)?;
        let new_session_id = admission.new_session_id.as_str();
        require(
            valid_id(new_owner_id)
                && valid_id(new_session_id)
                && (1..u64::MAX).contains(&new_owner_epoch),
            "new HTTP owner/session identity",
        )?;
        require(
            new_owner_id != self.origin_owner_id && new_session_id != self.ledger.origin_session_id,
            "restore cannot revive old owner/session",
        )?;
        require(
            self.ledger.turns.iter().all(|t| t.owner_id != new_owner_id),
            "restore cannot reuse ancestor owner",
        )?;
        require(
            self.bindings == admission.bindings && self.math == admission.math,
            "HTTP restore differs from live model/build/math/namespace",
        )
    }
}

#[cfg(test)]
#[path = "http_tests.rs"]
pub(super) mod tests;
