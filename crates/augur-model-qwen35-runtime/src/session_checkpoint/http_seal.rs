//! Data-only native state fingerprints. Never deserialize CUDA capabilities.
use super::{input_digest, invalid, require, sha256, valid_digest};
use crate::session_checkpoint::metadata::{
    HistoryMode, SessionDescriptor, CAPACITY, FEATURE_ROW_BYTES, HISTORY, KV_ROW_BYTES,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StatePlaneSeal {
    pub layer: usize,
    pub plane: usize,
    pub bytes: u64,
    pub file_name: String,
    pub dtype: String,
    pub kind: String,
    pub sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetStateSeal {
    pub target_committed_state_sha256: String,
    pub feature_bytes_sha256: String,
    pub state_plane_hashes: Vec<StatePlaneSeal>,
}

impl TargetStateSeal {
    pub fn validate(&self, cursor: usize) -> io::Result<()> {
        require((1..=CAPACITY).contains(&cursor), "seal cursor extent")?;
        require(
            valid_digest(&self.target_committed_state_sha256)
                && valid_digest(&self.feature_bytes_sha256),
            "seal digest",
        )?;
        require(
            self.state_plane_hashes.len() == 128,
            "complete 64-layer state seal",
        )?;
        for (i, plane) in self.state_plane_hashes.iter().enumerate() {
            let layer = i / 2;
            let full = layer % 4 == 3;
            let bytes = if full {
                cursor as u64 * KV_ROW_BYTES
            } else if i % 2 == 0 {
                122880
            } else {
                3145728
            };
            require(
                plane.layer == layer
                    && plane.plane == i % 2
                    && plane.bytes == bytes
                    && plane.file_name == format!("layer-{layer:02}-plane-{}.bin", i % 2)
                    && plane.kind == if full { "full_kv8" } else { "gdn_conv" }
                    && plane.dtype == if full { "q8_0_34bytes" } else { "f32_le" }
                    && valid_digest(&plane.sha256),
                "state seal plane geometry/order/digest",
            )?;
        }
        Ok(())
    }

    pub fn validate_descriptor(&self, descriptor: &SessionDescriptor) -> io::Result<()> {
        descriptor.validate()?;
        self.validate(descriptor.ledger.consumed.len())?;
        let plan = descriptor.transfer_plan()?;
        require(
            self.state_plane_hashes
                .iter()
                .zip(&plan.components[1..129])
                .all(|(p, c)| p.bytes == c.bytes),
            "state seal native transfer extent",
        )
    }

    /// Extract only typed math-state fields from a producer seal. The original
    /// report hash/ledger/initialization must agree before accepting the fields.
    /// This method alone does not verify payload bytes or confer runtime authority.
    pub fn from_checkpoint(checkpoint: &Value, descriptor: &SessionDescriptor) -> io::Result<Self> {
        descriptor.validate()?;
        let mut unsigned = checkpoint.clone();
        let supplied = unsigned
            .as_object_mut()
            .ok_or_else(|| invalid("native seal object"))?
            .remove("checkpoint_sha256")
            .ok_or_else(|| invalid("native seal hash missing"))?;
        require(
            supplied
                == json!(sha256(
                    &serde_json::to_vec(&unsigned).map_err(io::Error::other)?
                )),
            "native checkpoint content hash",
        )?;
        let ledger = &descriptor.ledger;
        let suffixes: Vec<_> = ledger
            .user_suffix_ranges
            .iter()
            .map(|r| (r.start, r.end))
            .collect();
        require(
            checkpoint["session_id"] == json!(ledger.origin_session_id)
                && checkpoint["revision"] == json!(ledger.revision)
                && checkpoint["generation"] == json!(ledger.generation)
                && checkpoint["cursor"] == json!(ledger.consumed.len())
                && checkpoint["capacity"] == json!(CAPACITY)
                && checkpoint["consumed_input_digest"] == json!(input_digest(&ledger.consumed))
                && checkpoint["pending_token"] == json!(ledger.pending)
                && checkpoint["pending_token_published"] == json!(ledger.pending_published)
                && checkpoint["generated_token_count"] == json!(ledger.generated.len())
                && checkpoint["user_suffix_ranges"] == json!(suffixes)
                && checkpoint["target_planes"] == json!(128)
                && checkpoint["draft_stream_drained"] == json!(true)
                && checkpoint["draft_rebind_required"] == json!(true)
                && checkpoint["product_ready"] == json!(false),
            "native checkpoint descriptor/ledger boundary",
        )?;
        let schema = match descriptor.features.mode {
            HistoryMode::Growing2048 { .. } => {
                require(
                    checkpoint["initialized_history"]
                        == json!({
                            "schema":"qwen35-initialized-session-history-v1",
                            "start_position":descriptor.features.start,
                            "active_rows":descriptor.features.rows,
                            "physical_capacity_rows":HISTORY,"uninitialized_tail_hashed":false
                        }),
                    "native seal initialized feature extent",
                )?;
                "qwen35-native-growing-session-target-checkpoint-v1"
            }
            HistoryMode::Fixed2048 => {
                require(
                    checkpoint.get("initialized_history").is_none(),
                    "unexpected growing seal",
                )?;
                "qwen35-native-session-target-checkpoint-v1"
            }
        };
        require(
            checkpoint["schema"] == json!(schema),
            "native target seal schema",
        )?;
        let result: Self = serde_json::from_value(json!({
            "target_committed_state_sha256":checkpoint["target_committed_state_sha256"],
            "feature_bytes_sha256":checkpoint["feature_bytes_sha256"],
            "state_plane_hashes":checkpoint["state_plane_hashes"]
        }))
        .map_err(io::Error::other)?;
        result.validate_descriptor(descriptor)?;
        let audit_bytes: u64 = result
            .state_plane_hashes
            .iter()
            .map(|p| p.bytes)
            .sum::<u64>()
            + descriptor.features.rows as u64 * FEATURE_ROW_BYTES;
        require(
            checkpoint["state_audit_d2h_bytes"] == json!(audit_bytes),
            "seal audit byte conservation",
        )?;
        Ok(result)
    }

    /// Called with a newly computed GPU checkpoint, never the saved report.
    /// Session provenance changes; mathematical state/ledger must not change.
    pub fn validate_restored(
        &self,
        checkpoint: &Value,
        saved: &SessionDescriptor,
        new_session_id: &str,
    ) -> io::Result<()> {
        require(
            super::valid_id(new_session_id) && new_session_id != saved.ledger.origin_session_id,
            "restored seal requires new session",
        )?;
        let mut destination = saved.clone();
        destination.ledger.origin_session_id = new_session_id.into();
        require(
            Self::from_checkpoint(checkpoint, &destination)? == *self,
            "restored GPU state differs from saved target seal",
        )
    }
}
