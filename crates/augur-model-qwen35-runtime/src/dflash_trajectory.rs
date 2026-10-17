//! Diagnostic-only, reconstructible target-feature journal. No model mutation.
//! The live draft input is read independently each round, but only its base and
//! new committed suffix are persisted. Retained bytes must match exactly.
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Component, Path, PathBuf},
};

type Result<T> = std::result::Result<T, String>;
fn require(ok: bool, message: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(message.into())
    }
}
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub(crate) fn safe_existing_directory(path: &Path) -> Result<()> {
    require(
        path.is_absolute() && !path.components().any(|c| matches!(c, Component::ParentDir)),
        "trajectory requires absolute nontraversing directory",
    )?;
    for ancestor in path.ancestors() {
        let metadata = fs::symlink_metadata(ancestor).map_err(|e| e.to_string())?;
        require(
            metadata.is_dir() && !metadata.file_type().is_symlink(),
            "trajectory directory/symlink",
        )?;
        #[cfg(windows)]
        {
            use std::os::windows::fs::MetadataExt;
            require(
                metadata.file_attributes() & 0x400 == 0,
                "trajectory reparse directory",
            )?;
        }
    }
    Ok(())
}

#[derive(Clone)]
struct Input {
    start: usize,
    cursor: usize,
    generation: u64,
    anchor: u32,
    tokens: Vec<u32>,
    features: Vec<u8>,
}
struct Commit {
    cursor: usize,
    generation: u64,
    anchor: u32,
    consumed: Vec<u32>,
}

pub(crate) struct Journal {
    directory: PathBuf,
    metadata: Value,
    row_bytes: usize,
    capacity: usize,
    vocab: usize,
    data: File,
    data_hash: Sha256,
    data_bytes: usize,
    input: Option<Input>,
    pending: Option<Commit>,
    open_round: Option<Value>,
    rounds: Vec<Value>,
}

impl Journal {
    pub(crate) fn create(
        directory: &Path,
        metadata: Value,
        width: usize,
        capacity: usize,
        vocab: usize,
    ) -> Result<Self> {
        require(
            width > 0 && capacity > 0 && vocab > 1,
            "trajectory geometry",
        )?;
        safe_existing_directory(directory.parent().ok_or("trajectory parent")?)?;
        require(!directory.exists(), "trajectory output already exists")?;
        let row_bytes = width.checked_mul(4).ok_or("trajectory row overflow")?;
        fs::create_dir(directory).map_err(|e| e.to_string())?;
        let data = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(directory.join("data.bin"))
            .map_err(|e| e.to_string())?;
        Ok(Self {
            directory: directory.into(),
            metadata,
            row_bytes,
            capacity,
            vocab,
            data,
            data_hash: Sha256::new(),
            data_bytes: 0,
            input: None,
            pending: None,
            open_round: None,
            rounds: Vec::new(),
        })
    }

    pub(crate) fn blob(&mut self, bytes: &[u8], role: &str) -> Result<Value> {
        let offset = self.data_bytes;
        self.data_bytes = offset
            .checked_add(bytes.len())
            .ok_or("trajectory file overflow")?;
        self.data.write_all(bytes).map_err(|e| e.to_string())?;
        self.data_hash.update(bytes);
        Ok(
            json!({"file":"data.bin","offset":offset,"bytes":bytes.len(),
            "sha256":sha(bytes),"role":role}),
        )
    }

    pub(crate) fn begin(
        &mut self,
        start: usize,
        generation: u64,
        anchor: u32,
        tokens: &[u32],
        actual_features: Vec<u8>,
    ) -> Result<()> {
        require(self.open_round.is_none(), "trajectory round already open")?;
        require(
            !tokens.is_empty()
                && tokens.len() <= self.capacity
                && generation > 0
                && (anchor as usize) < self.vocab
                && tokens.iter().all(|&t| (t as usize) < self.vocab),
            "trajectory input token/extent",
        )?;
        let cursor = start
            .checked_add(tokens.len())
            .ok_or("trajectory cursor overflow")?;
        require(
            tokens.len().checked_mul(self.row_bytes) == Some(actual_features.len()),
            "trajectory actual feature geometry",
        )?;
        let (retained, new_rows, kind) = if let Some(previous) = &self.input {
            let commit = self
                .pending
                .as_ref()
                .ok_or("trajectory missing previous commit")?;
            require(
                cursor == commit.cursor
                    && generation == commit.generation
                    && anchor == commit.anchor,
                "trajectory input commit position/generation/anchor drift",
            )?;
            require(
                start >= previous.start && start <= previous.cursor,
                "trajectory retained position drift",
            )?;
            let evicted = start - previous.start;
            let retained = previous
                .tokens
                .len()
                .checked_sub(evicted)
                .ok_or("trajectory eviction extent")?;
            let mut expected = previous.tokens[evicted..].to_vec();
            expected.extend_from_slice(&commit.consumed);
            require(
                expected == tokens,
                "trajectory consumed token history drift",
            )?;
            require(
                previous.features[evicted * self.row_bytes..]
                    == actual_features[..retained * self.row_bytes],
                "trajectory retained hidden bytes drift",
            )?;
            (retained, commit.consumed.len(), "committed_delta")
        } else {
            (0, tokens.len(), "initial_base")
        };
        require(
            retained + new_rows == tokens.len(),
            "trajectory base/delta extent",
        )?;
        let suffix = &actual_features[retained * self.row_bytes..];
        require(
            suffix
                .chunks_exact(4)
                .all(|x| f32::from_le_bytes(x.try_into().unwrap()).is_finite()),
            "trajectory nonfinite feature",
        )?;
        let full_hash = sha(&actual_features);
        let blob = self.blob(suffix, kind)?;
        self.open_round = Some(json!({"round_index":self.rounds.len(),"input":{
            "context_start":start,"context_rows":tokens.len(),"cursor":cursor,
            "generation":generation,"anchor":anchor,"token_ids":tokens,
            "retained_rows":retained,"new_rows":new_rows,"features":blob,
            "actual_draft_input_sha256":full_hash,"retained_bytes_checked_against_live_draft":true}}));
        self.input = Some(Input {
            start,
            cursor,
            generation,
            anchor,
            tokens: tokens.into(),
            features: actual_features,
        });
        self.pending = None;
        Ok(())
    }

    pub(crate) fn commit(
        &mut self,
        emitted: &[u32],
        consumed: &[u32],
        cursor: usize,
        generation: u64,
        anchor: u32,
        observation: Value,
    ) -> Result<()> {
        let input = self
            .input
            .as_ref()
            .ok_or("trajectory commit missing input")?;
        require(
            self.open_round.is_some() && self.pending.is_none(),
            "trajectory commit phase",
        )?;
        require(
            (1..=8).contains(&emitted.len())
                && emitted.len() == consumed.len()
                && emitted.iter().all(|&t| (t as usize) < self.vocab),
            "trajectory commit token extent",
        )?;
        let mut expected = vec![input.anchor];
        expected.extend_from_slice(&emitted[..emitted.len() - 1]);
        require(
            consumed == expected && emitted.last() == Some(&anchor),
            "trajectory pending anchor/consumed path mismatch",
        )?;
        require(
            input.cursor.checked_add(consumed.len()) == Some(cursor)
                && input.generation.checked_add(1) == Some(generation),
            "trajectory commit cursor/generation",
        )?;
        let mut round = self.open_round.take().ok_or("trajectory round absent")?;
        round["commit"] = json!({"consumed_input_token_ids":consumed,"emitted_token_ids":emitted,
            "cursor_after":cursor,"generation_after":generation,"next_pending_anchor":anchor,
            "pending_anchor_has_no_fabricated_hidden":true});
        round["observation"] = observation;
        self.rounds.push(round);
        self.pending = Some(Commit {
            cursor,
            generation,
            anchor,
            consumed: consumed.into(),
        });
        Ok(())
    }

    pub(crate) fn attach_f32(&mut self, role: &str, bytes: &[u8]) -> Result<()> {
        require(
            self.open_round.is_some()
                && !bytes.is_empty()
                && bytes.len().is_multiple_of(4)
                && bytes
                    .chunks_exact(4)
                    .all(|x| f32::from_le_bytes(x.try_into().unwrap()).is_finite()),
            "trajectory auxiliary live f32 extent/nonfinite",
        )?;
        let blob = self.blob(bytes, role)?;
        self.open_round.as_mut().ok_or("trajectory round missing")?[role] = blob;
        Ok(())
    }

    pub(crate) fn finish(self) -> Result<Value> {
        require(
            self.open_round.is_none() && (self.rounds.is_empty() || self.pending.is_some()),
            "trajectory incomplete committed round",
        )?;
        self.data.sync_all().map_err(|e| e.to_string())?;
        let report = json!({"schema":"qwen35-dflash-live-trajectory-v1","status":"COMPLETE_DIAGNOSTIC_CAPTURE",
            "metadata":self.metadata,"feature_width":self.row_bytes/4,"feature_dtype":"f32le",
            "history_capacity":self.capacity,"rounds":self.rounds,"data_bytes":self.data_bytes,
            "data_sha256":format!("{:x}",self.data_hash.finalize()),
            "label_policy":"accepted_prefix_plus_first_rejection_valid; later_draft_conditioned_labels_masked",
            "default_path_changed":false,"diagnostic_host_serialized":true,
            "performance_evidence":false,"product_ready":false,"qualified":false,
            "semantic_qualification":false,"visible_tps":false});
        let bytes = serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?;
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(self.directory.join("complete.json"))
            .map_err(|e| e.to_string())?;
        file.write_all(&bytes).map_err(|e| e.to_string())?;
        file.sync_all().map_err(|e| e.to_string())?;
        Ok(
            json!({"directory":self.directory,"complete_sha256":sha(&bytes),"rounds":report["rounds"].as_array().unwrap().len(),
            "data_bytes":self.data_bytes,"performance_evidence":false,"qualified":false}),
        )
    }
}

/// Slot zero is the pending anchor, not a predicted token. Target row i labels
/// draft slot i+1. Rows AFTER the first rejected slot have a noncanonical prefix.
pub(crate) fn classify(
    accepted: usize,
    candidate_ids: &[u32],
    path: &[u32],
    target: &[u32],
) -> Result<Value> {
    require(
        accepted <= 7 && candidate_ids.len() == 8 * 16 && path.len() == 8 && target.len() == 8,
        "trajectory candidate/label extent",
    )?;
    require(
        path[1..1 + accepted] == target[..accepted],
        "trajectory accepted prefix mismatch",
    )?;
    if accepted < 7 {
        require(
            path[accepted + 1] != target[accepted],
            "trajectory first rejection mismatch",
        )?;
    }
    let labels = (1..8)
        .map(|slot| {
            let teacher = target[slot - 1];
            let rank = candidate_ids[slot * 16..(slot + 1) * 16]
                .iter()
                .position(|&t| t == teacher);
            let valid = slot <= accepted + 1;
            let error = if !valid {
                "masked_noncanonical_draft_prefix"
            } else if path[slot] == teacher {
                "accepted"
            } else if rank.is_none() {
                "unary_top16_coverage_miss"
            } else {
                "selector_ranking_error"
            };
            json!({"draft_slot":slot,"target_row":slot-1,"teacher_token":teacher,
            "teacher_prefix_valid":valid,"teacher_candidate_rank_zero_based":rank,"error":error})
        })
        .collect::<Vec<_>>();
    Ok(
        json!({"accepted_drafts":accepted,"first_rejected_draft_slot":(accepted<7).then_some(accepted+1),
        "valid_target_prefix_rows":accepted+1,"labels":labels}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    fn fixture() -> (PathBuf, Journal) {
        let dir = std::env::temp_dir().join(format!(
            "augur-trajectory-{}-{}",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        let journal = Journal::create(&dir, json!({"fixture":true}), 2, 4, 100).unwrap();
        (dir, journal)
    }
    fn bytes(rows: &[u32]) -> Vec<u8> {
        rows.iter()
            .flat_map(|&x| [(x as f32).to_le_bytes(), (x as f32 + 0.5).to_le_bytes()].concat())
            .collect()
    }
    #[test]
    fn base_delta_reconstructs_growth_and_eviction_without_duplicate_payload() {
        let (dir, mut j) = fixture();
        j.begin(10, 1, 3, &[1, 2], bytes(&[1, 2])).unwrap();
        j.commit(&[4, 5], &[3, 4], 14, 2, 5, json!({})).unwrap();
        j.begin(10, 2, 5, &[1, 2, 3, 4], bytes(&[1, 2, 3, 4]))
            .unwrap();
        j.commit(&[6], &[5], 15, 3, 6, json!({})).unwrap();
        j.begin(11, 3, 6, &[2, 3, 4, 5], bytes(&[2, 3, 4, 5]))
            .unwrap();
        j.commit(&[7], &[6], 16, 4, 7, json!({})).unwrap();
        let receipt = j.finish().unwrap();
        assert_eq!(receipt["data_bytes"], 5 * 8);
        assert_eq!(
            fs::read(dir.join("data.bin")).unwrap(),
            bytes(&[1, 2, 3, 4, 5])
        );
        let complete: Value =
            serde_json::from_slice(&fs::read(dir.join("complete.json")).unwrap()).unwrap();
        assert_eq!(complete["rounds"][2]["input"]["retained_rows"], 3);
        assert_eq!(
            complete["rounds"][2]["input"]["actual_draft_input_sha256"],
            sha(&bytes(&[2, 3, 4, 5]))
        );
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn rejects_retained_hidden_token_position_generation_and_anchor_drift() {
        for axis in 0..5 {
            let (dir, mut j) = fixture();
            j.begin(10, 1, 3, &[1, 2], bytes(&[1, 2])).unwrap();
            j.commit(&[4], &[3], 13, 2, 4, json!({})).unwrap();
            let mut features = bytes(&[1, 2, 3]);
            if axis == 0 {
                features[0] ^= 1;
            }
            let tokens = if axis == 1 {
                vec![1, 9, 3]
            } else {
                vec![1, 2, 3]
            };
            assert!(j
                .begin(
                    if axis == 2 { 11 } else { 10 },
                    if axis == 3 { 3 } else { 2 },
                    if axis == 4 { 5 } else { 4 },
                    &tokens,
                    features
                )
                .is_err());
            drop(j);
            assert!(!dir.join("complete.json").exists());
            fs::remove_dir_all(dir).unwrap();
        }
    }
    #[test]
    fn commit_consumes_old_anchor_and_not_new_pending_anchor_or_rejected_suffix() {
        let (dir, mut j) = fixture();
        j.begin(0, 1, 3, &[1, 2], bytes(&[1, 2])).unwrap();
        assert!(j.commit(&[4, 5], &[4, 5], 4, 2, 5, json!({})).is_err());
        assert!(j.commit(&[4, 5], &[3, 9], 4, 2, 5, json!({})).is_err());
        j.commit(&[4, 5], &[3, 4], 4, 2, 5, json!({})).unwrap();
        drop(j);
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn incomplete_round_or_nonfinite_input_never_writes_completion() {
        let (dir, mut j) = fixture();
        assert!(j.begin(0, 1, 3, &[1], vec![0xff; 8]).is_err());
        j.begin(0, 1, 3, &[1], bytes(&[1])).unwrap();
        assert!(j.finish().is_err());
        assert!(!dir.join("complete.json").exists());
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn auxiliary_features_are_bound_and_nonfinite_values_are_rejected() {
        let (dir, mut j) = fixture();
        assert!(j.attach_f32("selector_gate_f32le", &bytes(&[1])).is_err());
        j.begin(0, 1, 3, &[1], bytes(&[1])).unwrap();
        assert!(j.attach_f32("selector_gate_f32le", &[0xff; 4]).is_err());
        j.attach_f32("selector_gate_f32le", &bytes(&[2])).unwrap();
        j.commit(&[4], &[3], 2, 2, 4, json!({})).unwrap();
        j.finish().unwrap();
        let report: Value =
            serde_json::from_slice(&fs::read(dir.join("complete.json")).unwrap()).unwrap();
        assert_eq!(
            report["rounds"][0]["selector_gate_f32le"]["sha256"],
            sha(&bytes(&[2]))
        );
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn all_acceptance_lengths_mask_only_post_rejection_labels() {
        for accepted in 0..=7 {
            let mut path = vec![10; 8];
            let mut targets = vec![10; 8];
            if accepted < 7 {
                path[accepted + 1] = 20;
                targets[accepted] = 30;
            }
            let mut candidates = vec![10; 128];
            let labels = classify(accepted, &candidates, &path, &targets).unwrap();
            if accepted < 7 {
                assert_eq!(
                    labels["labels"][accepted]["error"],
                    "unary_top16_coverage_miss"
                );
                candidates[(accepted + 1) * 16] = 30;
                assert_eq!(
                    classify(accepted, &candidates, &path, &targets).unwrap()["labels"][accepted]
                        ["error"],
                    "selector_ranking_error"
                );
            }
            for slot in 1..8 {
                assert_eq!(
                    labels["labels"][slot - 1]["teacher_prefix_valid"],
                    slot <= accepted + 1
                );
            }
        }
        assert!(classify(8, &[0; 128], &[0; 8], &[0; 8]).is_err());
    }
}
