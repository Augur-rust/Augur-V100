//! Diagnostic-only live-range byte evidence and separate-owner fresh recomputation.
//! Does not mutate the active draft, target state, token choice or acceptance rule.
use super::*;
use augur_backend::{cuda_device_allocation_snapshot, cuda_memory_info};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Component, Path, PathBuf};

const CHUNK: usize = 8 * 1024 * 1024;
const WORKSPACE: usize = 399_708_736;
const MIN_FREE_RESERVE: u64 = 256 * 1024 * 1024;

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
fn safe_path(path: &Path) -> Result<()> {
    require(
        path.is_absolute() && !path.components().any(|p| matches!(p, Component::ParentDir)),
        "absolute nontraversing path",
    )?;
    for parent in path.ancestors() {
        let metadata = fs::symlink_metadata(parent).map_err(err)?;
        require(
            !metadata.file_type().is_symlink(),
            "private capture symlink",
        )?;
        #[cfg(windows)]
        {
            use std::os::windows::fs::MetadataExt;
            require(
                metadata.file_attributes() & 0x400 == 0,
                "private capture reparse point",
            )?;
        }
    }
    Ok(())
}
fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    safe_path(path.parent().ok_or("capture parent")?)?;
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .map_err(err)?;
    file.write_all(bytes).map_err(err)?;
    file.sync_all().map_err(err)
}
fn write_json(path: &Path, value: &Value) -> Result<()> {
    write_new(path, &serde_json::to_vec_pretty(value).map_err(err)?)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ByteRange {
    offset: usize,
    bytes: usize,
    role: String,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Plane {
    name: String,
    physical_bytes: usize,
    ranges: Vec<ByteRange>,
    omitted: Vec<ByteRange>,
}

fn geometry() -> BTreeMap<String, usize> {
    let mut all = BTreeMap::new();
    for (name, count) in [
        ("features", 2048 * 5 * H),
        ("feature_shift_scratch", B * 5 * H),
        ("target_fc", 2048 * H),
        ("target", 2048 * H),
        ("kv_raw", (2048 + B) * KV),
        ("kv_norm", (2048 + B) * KV),
        ("norm", B * H),
        ("dynamic", B * DYN),
        ("conv", B * H),
        ("q_raw", B * Q),
        ("q_norm", B * Q),
        ("q_rope", B * Q),
        ("attention", B * Q),
        ("projected", B * H),
        ("conv_out", B * H),
        ("ffn_in", B * H),
        ("gate", B * F),
        ("up", B * F),
        ("activation", B * F),
        ("final", B * H),
        ("selector", B * 256),
        ("zero", H),
    ] {
        all.insert(name.into(), count * 4);
    }
    for layer in 0..=5 {
        all.insert(format!("hidden{layer}"), B * H * 4);
        if layer < 5 {
            for kind in ['k', 'v'] {
                all.insert(format!("{kind}{layer}"), (2048 + B) * KV * 4);
            }
        }
    }
    all.insert("audit_parts".into(), 17 * 32);
    all.insert("audit_digest".into(), 32);
    all
}

fn inventory(first: usize) -> Result<Vec<Plane>> {
    require(
        first == 0 || (2040..2048).contains(&first),
        "valid full-history projected suffix",
    )?;
    let mut result = Vec::new();
    for (name, bytes) in geometry() {
        let mut ranges = Vec::new();
        let mut omitted = Vec::new();
        if name == "feature_shift_scratch" || name.starts_with("audit_") {
            omitted.push(ByteRange {
                offset: 0,
                bytes,
                role: if name.starts_with("audit_") {
                    "audit_not_executed_uninitialized"
                } else {
                    "copy_scratch_not_live_and_fresh_uninitialized"
                }
                .into(),
            });
        } else {
            let width = if name == "target_fc" || name == "target" {
                Some(H)
            } else if name == "kv_raw" || name == "kv_norm" {
                Some(KV)
            } else {
                None
            };
            let offset = width.map_or(0, |w| first * w * 4);
            if offset > 0 {
                omitted.push(ByteRange {
                    offset: 0,
                    bytes: offset,
                    role: "retained_projection_scratch_not_current_or_read_before_rewrite".into(),
                });
            }
            ranges.push(ByteRange {
                offset,
                bytes: bytes - offset,
                role: if width.is_some() {
                    "fresh_projected_suffix_and_submitted_noise_if_present"
                } else {
                    "initialized_current_input_or_submitted_output"
                }
                .into(),
            });
        }
        result.push(Plane {
            name,
            physical_bytes: bytes,
            ranges,
            omitted,
        });
    }
    require(
        result.len() == 40 && result.iter().map(|p| p.physical_bytes).sum::<usize>() == WORKSPACE,
        "40-buffer workspace inventory",
    )?;
    Ok(result)
}

fn classify(block: usize, first: usize) -> Result<Option<&'static str>> {
    require(
        first == 0 || (2040..2048).contains(&first),
        "projected first-row policy",
    )?;
    if block == 0 {
        require(first == 0, "initial context may not be reused")?;
        return Ok(Some("initial"));
    }
    Ok(if first == 2040 {
        Some("after_full_commit")
    } else if first > 0 {
        Some("after_partial_commit")
    } else {
        None
    })
}

/// Three snapshots maximum: initial, first full and first natural partial
/// commit after at least512 committed inputs. This does not force acceptance.
#[cfg(feature = "qwen35-draft-private-long")]
fn classify_long(block: usize, cursor: usize, first: usize) -> Result<Option<&'static str>> {
    require(
        (8192..=9215).contains(&cursor) && block < 1024,
        "private long position bounds",
    )?;
    let phase = classify(block, first)?;
    if block == 0 {
        require(cursor == 8192, "private long exact8192 prompt")?;
        return Ok(Some("initial"));
    }
    require(
        cursor > 8192 && cursor - 8192 >= block,
        "private long monotonic cursor",
    )?;
    if cursor - 8192 < 512 {
        return Ok(None);
    }
    Ok(match phase {
        Some("after_full_commit") => Some("late_full_after512"),
        Some("after_partial_commit") => Some("late_partial_after512"),
        _ => None,
    })
}

pub(crate) fn capture_budget_matches(rounds: usize, output: usize, long: bool) -> bool {
    if cfg!(feature = "qwen35-draft-private-long") {
        rounds == 1024 && output == 1024 && long
    } else {
        rounds == 32 && output == 32 && !long
    }
}

pub(crate) struct Audit {
    directory: PathBuf,
    seen: BTreeSet<String>,
    records: Vec<Value>,
    source_commit: String,
}
impl Audit {
    pub(crate) fn from_env(source_commit: &str) -> Result<Self> {
        require(
            std::env::var_os("AUGUR_QWEN35_RAW_STATE_DIR").is_none()
                || cfg!(feature = "qwen35-optimized-state-bridge"),
            "private capture cannot run full target raw concurrently",
        )?;
        let directory = PathBuf::from(
            std::env::var_os("AUGUR_QWEN35_DRAFT_PRIVATE_DIR").ok_or("private capture directory")?,
        );
        require(
            directory.is_absolute()
                && !directory.exists()
                && !directory
                    .components()
                    .any(|c| matches!(c, Component::ParentDir)),
            "new absolute private output directory",
        )?;
        safe_path(directory.parent().ok_or("private parent")?)?;
        fs::create_dir(&directory).map_err(err)?;
        write_json(
            &directory.join("preflight.json"),
            &json!({"schema":"qwen35-draft-private-raw-preflight-v1",
            "max_boundaries":3,"long_generation_sparse_after512":cfg!(feature="qwen35-draft-private-long"),"max_roles_per_boundary":if cfg!(feature="qwen35-draft-private-capture-only"){2}else{3},"max_raw_bytes_per_snapshot":WORKSPACE,
            "fresh_recompute_deferred_until_request_exit":cfg!(feature="qwen35-draft-private-capture-only"),
            "host_chunk_bytes":CHUNK,"source_commit":source_commit,"source_dirty":crate::build_identity::source_dirty_for(source_commit),
            "requires_external_disk_preflight":true,"no_default_change":true,
            "complete_model_state_qualified":false,"product_tps_permitted":false}),
        )?;
        Ok(Self {
            directory,
            seen: BTreeSet::new(),
            records: Vec::new(),
            source_commit: source_commit.into(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn capture_selected(
        &mut self,
        draft: &DFlashQ8CudaBlock,
        block: usize,
        cursor: usize,
        generation: u64,
        anchor: u32,
        previous: Option<&Value>,
        dflash: &Value,
        vision: &Value,
    ) -> Result<()> {
        #[cfg(feature = "qwen35-draft-private-long")]
        let selected = classify_long(block, cursor, draft.context_first_row)?;
        #[cfg(not(feature = "qwen35-draft-private-long"))]
        let selected = classify(block, draft.context_first_row)?;
        let Some(label) = selected else {
            return Ok(());
        };
        if self.seen.contains(label) {
            return Ok(());
        }
        draft.validate_private_phase()?;
        require(
            cursor == draft.start + draft.context
                && generation == draft.request_generation
                && anchor == draft.anchor_token,
            "same target/draft pre-verification boundary",
        )?;
        require(
            dflash["resident_device_bytes"].as_u64().unwrap_or(0) > 0
                && vision["vision"]["resident_device_bytes"]
                    .as_u64()
                    .unwrap_or(0)
                    > 0,
            "actual full auxiliary residency",
        )?;
        if label != "initial" {
            let tx = previous.ok_or("prior commit required for reuse")?;
            require(
                tx["pass"] == true
                    && tx["cursor_after"].as_u64() == Some(cursor as u64)
                    && tx["generation_after"].as_u64() == Some(generation)
                    && tx["committed_token_count"].as_u64()
                        == Some((draft.context - draft.context_first_row) as u64),
                "reuse tied to actual previous commit",
            )?;
        }
        require(self.records.len() < 3, "bounded private snapshots")?;
        let directory = self.directory.join(format!("block{block:03}-{label}"));
        fs::create_dir(&directory).map_err(err)?;
        let layout = inventory(draft.context_first_row)?;
        let baseline =
            draft.capture_private(&directory.join("active-before"), &layout, "active-before")?;
        let memory_before = cuda_memory_info(draft.weights.device_ordinal()).map_err(err)?;
        #[cfg(not(feature = "qwen35-draft-private-capture-only"))]
        let record = {
            require(
                memory_before.free_bytes >= WORKSPACE as u64 + MIN_FREE_RESERVE,
                "extra fresh draft workspace memory gate",
            )?;
            let before = cuda_device_allocation_snapshot();
            let mut fresh = draft.fresh_private_reference()?;
            require(
                fresh.workspace_bytes() == WORKSPACE,
                "fresh workspace extent",
            )?;
            fresh.prepare_context()?;
            fresh.submit()?;
            let reference =
                fresh.capture_private(&directory.join("fresh"), &layout, "fresh-same-input")?;
            let memory_with_fresh =
                cuda_memory_info(draft.weights.device_ordinal()).map_err(err)?;
            require(
                memory_with_fresh.free_bytes >= MIN_FREE_RESERVE,
                "fresh live memory reserve",
            )?;
            drop(fresh);
            let after = cuda_device_allocation_snapshot();
            require(
                after.live_bytes == before.live_bytes
                    && after.live_allocations == before.live_allocations,
                "fresh owner cleanup",
            )?;
            let unchanged =
                draft.capture_private(&directory.join("active-after"), &layout, "active-after")?;
            let record = json!({"schema":"qwen35-draft-private-pair-v1","label":label,"block":block,
            "target_cursor":cursor,"target_generation":generation,"anchor":anchor,
            "previous_transaction":previous,"directory":directory,"layout_first_row":draft.context_first_row,
            "active_before":baseline,"fresh":reference,"active_after":unchanged,
            "dflash":dflash,"vision":vision,"memory_before_free_bytes":memory_before.free_bytes,
            "memory_with_fresh_free_bytes":memory_with_fresh.free_bytes,"extra_workspace_bytes":WORKSPACE,
            "fresh_allocations":after.successful_allocations-before.successful_allocations,
            "fresh_frees":after.successful_frees-before.successful_frees,"fresh_cleanup_live_delta":0,
            "canonical_private_bytes_equivalent":false,"complete_model_state_qualified":false,"product_tps_permitted":false});
            record
        };
        #[cfg(feature = "qwen35-draft-private-capture-only")]
        let record = {
            // No second owner while full target/KV/auxiliaries are resident.
            // Same active stream is drained and captured twice without mutation.
            let unchanged =
                draft.capture_private(&directory.join("active-after"), &layout, "active-after")?;
            json!({"schema":if cfg!(feature="qwen35-draft-private-long"){"qwen35-draft-private-long-pair-v1"}else{"qwen35-draft-private-archived-pair-v1"},"label":label,"block":block,
                "target_cursor":cursor,"target_generation":generation,"anchor":anchor,
                "previous_transaction":previous,"directory":directory,"layout_first_row":draft.context_first_row,
                "active_before":baseline,"active_after":unchanged,"dflash":dflash,"vision":vision,
                "memory_free_bytes_at_capture":memory_before.free_bytes,"extra_workspace_bytes":0,
                "fresh_recompute_deferred_until_request_exit":true,
                "canonical_private_bytes_equivalent":false,"complete_model_state_qualified":false,
                "product_tps_permitted":false})
        };
        write_json(&directory.join("pair.json"), &record)?;
        self.records.push(record);
        self.seen.insert(label.into());
        Ok(())
    }

    pub(crate) fn finish(self) -> Result<Value> {
        let result = json!({"schema":if cfg!(feature="qwen35-draft-private-long"){"qwen35-draft-private-long-runtime-v1"}else if cfg!(feature="qwen35-draft-private-capture-only") {"qwen35-draft-private-archived-runtime-v1"}else{"qwen35-draft-private-runtime-v1"},"source_commit":self.source_commit,
            "source_dirty":crate::build_identity::source_dirty_for(&self.source_commit),"records":self.records,"no_arithmetic_or_acceptance_changes":true,
            "canonical_private_bytes_equivalent":false,"complete_model_state_qualified":false,"product_tps_permitted":false});
        write_json(&self.directory.join("complete.json"), &result)?;
        Ok(result)
    }
}

impl DFlashQ8CudaBlock {
    /// Native session discriminator: full rebind or natural committed-row reuse.
    /// Reuses audited initialized-range inventory; never allocates a second GPU owner.
    #[cfg(feature = "qwen35-session-continuation")]
    pub(crate) fn capture_native_session_private_audit(
        &self,
        directory: &Path,
        role: &str,
    ) -> Result<Value> {
        // The inventory explicitly omits non-live projection prefixes after reuse.
        require(
            matches!(role, "session-retained-rebind" | "session-fresh-recompute"),
            "session private capture role",
        )?;
        self.capture_private(directory, &inventory(self.context_first_row)?, role)
    }

    fn validate_private_phase(&self) -> Result<()> {
        require(
            self.context == 2048
                && self.window == 2048
                && self.prepared
                && self.submitted
                && !self.poisoned
                && self.graph.is_none()
                && self.candidates.is_none()
                && self.context_rows_launch
                && self.block_rows_launch
                && self.q8_tile4,
            "private snapshot requires canonical submitted full2048 draft",
        )?;
        require(
            self.buffers.keys().eq(geometry().keys()),
            "exact draft owner namespace",
        )?;
        for (name, size) in geometry() {
            require(
                self.buffers[&name].len_bytes() == size,
                "actual draft physical extent",
            )?;
        }
        let capture = self
            .cached_capture
            .as_ref()
            .ok_or("real target capture receipt missing")?;
        let seed = self.private_seed()?;
        seed.validate(&self.config).map_err(err)?;
        capture
            .validate_for_block(
                &self.config,
                &seed,
                self.context * 5 * H,
                self.feature_identity.payload_sha256()?,
            )
            .map_err(err)?;
        require(
            capture.context_start_position == self.start && capture.context_len == self.context,
            "private positions",
        )?;
        Ok(())
    }
    fn private_seed(&self) -> Result<DFlashDraftBlockInput> {
        let mut tokens = vec![crate::DFLASH_SEED_MASK_TOKEN; B];
        tokens[0] = self.anchor_token;
        Ok(DFlashDraftBlockInput {
            request_generation: self.request_generation,
            absolute_position_start: self.start + self.context,
            anchor_token: self.anchor_token,
            block_tokens: tokens,
            mask_token_id: crate::DFLASH_SEED_MASK_TOKEN,
            target_feature_sha256: self.feature_identity.payload_sha256()?,
        })
    }
    fn fresh_private_reference(&self) -> Result<Self> {
        self.validate_private_phase()?;
        let submission = self.stream.submission_receipt().map_err(err)?;
        let mut raw = vec![0u8; B * H * 4];
        let proof = self.buffers["hidden0"]
            .copy_range_to_host_sync_with_submission_proof(&self.stream, 0, &mut raw, submission)
            .map_err(err)?;
        require(
            proof.covers_submission(submission)
                && proof.belongs_to_stream(&self.stream).map_err(err)?,
            "actual draft seed owner proof",
        )?;
        let noise = raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        require(
            noise.iter().all(|v| v.is_finite()),
            "finite actual noise embedding",
        )?;
        let mut fresh = Self::from_resident_device_features(
            &self.weights,
            &self.private_seed()?,
            self.cached_capture.as_ref().ok_or("capture")?,
            self.buffers["features"]
                .view(0, self.context * 5 * H * 4)
                .map_err(err)?,
            &noise,
            self.start,
            self.window,
            &self.stream,
        )?;
        fresh.enable_context_rows_launch()?;
        fresh.enable_block_rows_launch()?;
        fresh.enable_q8_tile4()?;
        Ok(fresh)
    }
    fn capture_private(&self, directory: &Path, layout: &[Plane], role: &str) -> Result<Value> {
        self.validate_private_phase()?;
        let first = layout
            .iter()
            .find(|p| p.name == "target")
            .and_then(|p| p.ranges.first())
            .ok_or("layout target")?
            .offset
            / (H * 4);
        require(
            layout == inventory(first)?,
            "explicit allowed range inventory",
        )?;
        if role != "fresh-same-input" {
            require(first == self.context_first_row, "active layout phase")?;
        }
        let submission = self.stream.submission_receipt().map_err(err)?;
        let proof = self
            .stream
            .synchronize_submission(submission)
            .map_err(err)?;
        require(
            proof.covers_submission(submission)
                && proof.belongs_to_stream(&self.stream).map_err(err)?,
            "private quiescent proof",
        )?;
        safe_path(directory.parent().ok_or("capture parent")?)?;
        fs::create_dir(directory).map_err(err)?;
        let mut host = vec![0u8; CHUNK];
        let mut chunks = Vec::new();
        let mut total = 0usize;
        for (index, plane) in layout.iter().enumerate() {
            for (range_index, range) in plane.ranges.iter().enumerate() {
                let mut consumed = 0;
                while consumed < range.bytes {
                    let len = CHUNK.min(range.bytes - consumed);
                    let offset = range.offset + consumed;
                    let proof = self.buffers[&plane.name]
                        .copy_range_to_host_sync_with_submission_proof(
                            &self.stream,
                            offset,
                            &mut host[..len],
                            submission,
                        )
                        .map_err(err)?;
                    require(
                        proof.covers_submission(submission)
                            && proof.belongs_to_stream(&self.stream).map_err(err)?,
                        "private readback owner proof",
                    )?;
                    let file = format!("chunk{:06}.bin", chunks.len());
                    write_new(&directory.join(&file), &host[..len])?;
                    chunks.push(json!({"file":file,"plane":index,"range":range_index,"offset":offset,"bytes":len,
                        "sha256":sha(&host[..len])}));
                    total += len;
                    consumed += len;
                }
            }
        }
        self.stream.synchronize().map_err(err)?;
        self.validate_private_phase()?;
        require(total <= WORKSPACE, "bounded snapshot byte total")?;
        let receipt = json!({"schema":"qwen35-draft-private-live-bytes-v1","role":role,
            "owner":format!("{submission:?}"),"context":self.context,"start":self.start,"window":self.window,
            "generation":self.request_generation,"anchor":self.anchor_token,
            "context_first_row":self.context_first_row,"layout_first_row":first,
            "prepared":self.prepared,"submitted":self.submitted,"poisoned":self.poisoned,
            "cached_capture":self.cached_capture,"feature_receipt_digest_semantics":"existing_noncryptographic_device_fingerprint_not_raw_sha",
            "q8_weight_sha256":self.weights.file_sha256(),"q8_weight_bytes":self.weights.resident_bytes(),
            "context_rows_launch":self.context_rows_launch,"block_rows_launch":self.block_rows_launch,"q8_tile4":self.q8_tile4,
            "planes":layout,"chunks":chunks,"captured_bytes":total,"workspace_bytes":WORKSPACE,
            "host_chunk_bytes":CHUNK,"drain_completed":true,"all_physical_bytes_captured":false,
            "canonical_private_bytes_equivalent":false,"complete_model_state_qualified":false,"product_tps_permitted":false});
        write_json(&directory.join("complete.json"), &receipt)?;
        let expected = std::iter::once("complete.json".to_owned())
            .chain(
                chunks
                    .iter()
                    .map(|c| c["file"].as_str().unwrap().to_owned()),
            )
            .collect::<BTreeSet<_>>();
        let actual = fs::read_dir(directory)
            .map_err(err)?
            .map(|e| {
                e.map_err(err)
                    .map(|e| e.file_name().to_string_lossy().into_owned())
            })
            .collect::<Result<BTreeSet<_>>>()?;
        require(actual == expected, "private capture namespace")?;
        Ok(
            json!({"directory":directory,"receipt_sha256":sha(&fs::read(directory.join("complete.json")).map_err(err)?),
            "captured_bytes":total,"owner":format!("{submission:?}")}),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all40physical_owners_accounted() {
        let x = geometry();
        assert_eq!(x.len(), 40);
        assert_eq!(x.values().sum::<usize>(), WORKSPACE);
    }
    #[test]
    fn initial_has_37_initialized_read_ranges() {
        let x = inventory(0).unwrap();
        assert_eq!(x.iter().filter(|p| !p.ranges.is_empty()).count(), 37);
    }
    #[test]
    fn audit_and_shift_never_read() {
        for first in [0, 2040, 2043, 2047] {
            let x = inventory(first).unwrap();
            for p in x {
                if p.name.starts_with("audit_") || p.name == "feature_shift_scratch" {
                    assert!(p.ranges.is_empty());
                    assert_eq!(p.omitted[0].bytes, p.physical_bytes);
                }
            }
        }
    }
    #[test]
    fn full_byte_partition() {
        for first in [0, 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047] {
            for p in inventory(first).unwrap() {
                let mut r = p.ranges.iter().chain(&p.omitted).collect::<Vec<_>>();
                r.sort_by_key(|r| r.offset);
                let mut end = 0;
                for part in r {
                    assert_eq!(part.offset, end);
                    assert!(part.bytes > 0);
                    end += part.bytes;
                }
                assert_eq!(end, p.physical_bytes);
            }
        }
    }
    #[test]
    fn stale_projection_prefix_not_read() {
        let x = inventory(2043).unwrap();
        for (name, w) in [
            ("target", H),
            ("target_fc", H),
            ("kv_raw", KV),
            ("kv_norm", KV),
        ] {
            let p = x.iter().find(|p| p.name == name).unwrap();
            assert_eq!(p.ranges[0].offset, 2043 * w * 4);
            assert_eq!(p.omitted[0].bytes, 2043 * w * 4);
        }
    }
    #[test]
    fn noise_kv_suffix_included() {
        let x = inventory(2043).unwrap();
        let p = x.iter().find(|p| p.name == "kv_raw").unwrap();
        assert_eq!(p.ranges[0].bytes, (5 + B) * KV * 4);
    }
    #[test]
    fn all_context_kv_always_compared() {
        for first in [0, 2040, 2043] {
            let x = inventory(first).unwrap();
            for layer in 0..5 {
                for kind in ['k', 'v'] {
                    let p = x
                        .iter()
                        .find(|p| p.name == format!("{kind}{layer}"))
                        .unwrap();
                    assert_eq!(p.ranges[0].bytes, (2048 + B) * KV * 4);
                    assert_eq!(p.ranges[0].offset, 0);
                }
            }
        }
    }
    #[test]
    fn input_seed_and_constants_retained() {
        let x = inventory(2043).unwrap();
        for name in ["features", "hidden0", "zero"] {
            let p = x.iter().find(|p| p.name == name).unwrap();
            assert_eq!(p.ranges[0].bytes, p.physical_bytes);
        }
    }
    #[test]
    fn no_ambiguous_partial_geometry() {
        for n in [1, 8, 1024, 2039, 2048, usize::MAX] {
            assert!(inventory(n).is_err());
        }
    }
    #[test]
    fn classify_actual_previous_commit() {
        assert_eq!(classify(0, 0).unwrap(), Some("initial"));
        assert_eq!(classify(1, 2043).unwrap(), Some("after_partial_commit"));
        assert_eq!(classify(3, 2040).unwrap(), Some("after_full_commit"));
    }
    #[test]
    fn no_reuse_label_for_fresh() {
        assert_eq!(classify(1, 0).unwrap(), None);
        assert!(classify(0, 2043).is_err());
    }
    #[test]
    fn selected_pair_host_bound() {
        assert_eq!(CHUNK, 8388608);
        assert!(9 * WORKSPACE < 4_000_000_000);
    }
}

// Standalone replay is deliberately isolated from Audit::capture_selected and
// cannot release, shrink or rewrite a live request owner to make room.






#[cfg(all(test, feature = "qwen35-draft-private-long"))]
mod long_private_tests {
    use super::*;
    #[test]
    fn initial_exact_prompt() {
        assert_eq!(classify_long(0, 8192, 0).unwrap(), Some("initial"));
        assert!(classify_long(0, 8193, 0).is_err());
    }
    #[test]
    fn no_early_tail_sample() {
        for cursor in [8193, 8200, 8703] {
            assert_eq!(classify_long(1, cursor, 2040).unwrap(), None);
            assert_eq!(classify_long(1, cursor, 2043).unwrap(), None);
        }
    }
    #[test]
    fn exact_threshold_and_later_full() {
        for cursor in [8704, 8715, 9200] {
            assert_eq!(
                classify_long(70, cursor, 2040).unwrap(),
                Some("late_full_after512")
            );
        }
    }
    #[test]
    fn threshold_partial_and_all_reject_lengths() {
        for first in 2041..2048 {
            assert_eq!(
                classify_long(70, 8722, first).unwrap(),
                Some("late_partial_after512")
            );
        }
    }
    #[test]
    fn fresh_rebuild_not_called_reuse() {
        assert_eq!(classify_long(70, 8722, 0).unwrap(), None);
    }
    #[test]
    fn invalid_cursor_round_or_geometry() {
        for (block, cursor, first) in [
            (1, 8192, 2040),
            (900, 8704, 2040),
            (1024, 9215, 2040),
            (1, 9216, 2040),
            (1, 8704, 1),
        ] {
            assert!(classify_long(block, cursor, first).is_err());
        }
    }
    #[test]
    fn exact_optin_budget_only() {
        assert!(capture_budget_matches(1024, 1024, true));
        for (rounds, output, long) in [
            (32, 32, false),
            (1023, 1024, true),
            (1024, 1023, true),
            (1024, 1024, false),
        ] {
            assert!(!capture_budget_matches(rounds, output, long));
        }
    }
    #[test]
    fn bounded_extra_storage_no_new_owner() {
        assert!(cfg!(feature = "qwen35-draft-private-capture-only"));
        assert!(6 * WORKSPACE < 2_400_000_000);
    }
}
