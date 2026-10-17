//! Opt-in Augur post-layer feature capture. No candidate/acceptance authority.
//!
//! Capture copies are stream-ordered, bitwise F32 D2D after the complete layer
//! residual. The fixed 2048-row ring never includes a selected-but-unconsumed
//! token. A separate contiguous scratch is populated only at handoff/readback;
//! it never exposes uninitialized capacity. Reset invalidates metadata, not KV
//! capacity. Diagnostic SHA256 is over actual chronological little-endian bytes.
use super::*;
use crate::{
    DFlashTargetFeatureCaptureReceipt, DFLASH_TARGET_CAPTURE_FEATURE_WIDTH,
    DFLASH_TARGET_CAPTURE_HIDDEN_SIZE, DFLASH_TARGET_CAPTURE_LAYER_IDS,
    DFLASH_TARGET_CAPTURE_SCHEMA,
};
use sha2::{Digest, Sha256};

pub const HISTORY_ROWS: usize = 2048;
const H: usize = DFLASH_TARGET_CAPTURE_HIDDEN_SIZE;
const WIDTH: usize = DFLASH_TARGET_CAPTURE_FEATURE_WIDTH;
const ROW_BYTES: usize = WIDTH * 4;
const LAYER_BYTES: usize = H * 4;
const ALL_LAYERS: u8 = (1 << DFLASH_TARGET_CAPTURE_LAYER_IDS.len()) - 1;

/// Diagnostic bytes, not a product transfer and not acceptance evidence.
#[derive(Debug)]
pub struct FeatureSnapshot {
    pub receipt: DFlashTargetFeatureCaptureReceipt,
    pub owner_epoch: u64,
    pub little_endian_f32: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Segment {
    source_row: usize,
    destination_row: usize,
    rows: usize,
}
fn ring_segments(first: usize, rows: usize) -> Result<Vec<Segment>> {
    if rows == 0 || rows > HISTORY_ROWS || first.checked_add(rows).is_none() {
        return Err("invalid capture row range".into());
    }
    let slot = first % HISTORY_ROWS;
    let first_rows = rows.min(HISTORY_ROWS - slot);
    let mut result = vec![Segment {
        source_row: 0,
        destination_row: slot,
        rows: first_rows,
    }];
    if first_rows < rows {
        result.push(Segment {
            source_row: first_rows,
            destination_row: 0,
            rows: rows - first_rows,
        });
    }
    Ok(result)
}

// This metadata is a strict initialized-row ledger, not guessed window length.
// Missing/out-of-order layer copies cannot publish even if GPU bytes are stale.
struct Ledger {
    capacity: usize,
    end: usize,
    generation: u64,
    epoch: u64,
    pending: Option<(usize, usize)>,
    positions: Vec<Option<usize>>,
    masks: Vec<u8>,
}
impl Ledger {
    fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 || capacity > 262144 {
            return Err("capture capacity".into());
        }
        Ok(Self {
            capacity,
            end: 0,
            generation: 0,
            epoch: 1,
            pending: None,
            positions: vec![None; HISTORY_ROWS],
            masks: vec![0; HISTORY_ROWS],
        })
    }
    fn begin(&mut self, first: usize, rows: usize) -> Result<()> {
        let end = first.checked_add(rows).ok_or("capture position overflow")?;
        if self.pending.is_some()
            || rows == 0
            || first != self.end
            || end > self.capacity
            || self.generation == u64::MAX
        {
            return Err("capture transaction cursor/capacity/generation".into());
        }
        self.pending = Some((first, end));
        Ok(())
    }
    fn check_layer(&self, layer_slot: usize, first: usize, rows: usize) -> Result<()> {
        let (begin, end) = self
            .pending
            .ok_or("capture requires an active target operation")?;
        if layer_slot >= DFLASH_TARGET_CAPTURE_LAYER_IDS.len()
            || rows == 0
            || rows > HISTORY_ROWS
            || first < begin
            || first.checked_add(rows).is_none_or(|last| last > end)
        {
            return Err("capture layer/range outside active operation".into());
        }
        for pos in first..first + rows {
            let index = pos % HISTORY_ROWS;
            if layer_slot == 0 {
                if self.positions[index].is_some_and(|old| old >= pos) {
                    return Err("duplicate or backwards feature row".into());
                }
            } else if self.positions[index] != Some(pos)
                || self.masks[index] != (1 << layer_slot) - 1
            {
                return Err("capture layers must be complete and in exact order".into());
            }
        }
        Ok(())
    }
    fn mark_layer(&mut self, layer_slot: usize, first: usize, rows: usize) {
        for pos in first..first + rows {
            let index = pos % HISTORY_ROWS;
            self.positions[index] = Some(pos);
            self.masks[index] = (1 << (layer_slot + 1)) - 1;
        }
    }
    fn commit(&mut self) -> Result<()> {
        let (first, end) = self.pending.ok_or("no capture operation to commit")?;
        for pos in first.max(end.saturating_sub(HISTORY_ROWS))..end {
            let index = pos % HISTORY_ROWS;
            if self.positions[index] != Some(pos) || self.masks[index] != ALL_LAYERS {
                return Err("incomplete post-layer feature capture".into());
            }
        }
        self.generation = self
            .generation
            .checked_add(1)
            .ok_or("capture generation overflow")?;
        self.end = end;
        self.pending = None;
        Ok(())
    }
    fn range(&self) -> Result<(usize, usize)> {
        if self.pending.is_some() || self.end == 0 {
            return Err("no complete feature window".into());
        }
        let rows = self.end.min(HISTORY_ROWS);
        let first = self.end - rows;
        for pos in first..self.end {
            let index = pos % HISTORY_ROWS;
            if self.positions[index] != Some(pos) || self.masks[index] != ALL_LAYERS {
                return Err("uninitialized or stale feature window row".into());
            }
        }
        Ok((first, rows))
    }
    fn reset(&mut self) -> Result<()> {
        let epoch = self.epoch.checked_add(1).ok_or("capture epoch overflow")?;
        let generation = self
            .generation
            .checked_add(1)
            .ok_or("capture generation overflow")?;
        self.epoch = epoch;
        self.generation = generation;
        self.end = 0;
        self.pending = None;
        self.positions.fill(None);
        self.masks.fill(0);
        Ok(())
    }
}

/// Trusted same-process producer identity, NOT a digest of feature bytes.
/// Private fields prevent callers from constructing a payload/owner substitute.
#[cfg(feature = "qwen35-augur-dflash-bridge")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OwnerFeatureIdentity {
    source: [u8; 32],
    owner: usize,
    epoch: u64,
    revision: u64,
    start: usize,
    rows: usize,
}
#[cfg(feature = "qwen35-augur-dflash-bridge")]
impl OwnerFeatureIdentity {
    pub(crate) fn start(self) -> usize { self.start }
    pub(crate) fn rows(self) -> usize { self.rows }
    pub(crate) fn end(self) -> usize { self.start + self.rows }
    pub(crate) fn capture_identity(self) -> (u64, u64, usize) {
        (self.epoch, self.revision, self.end())
    }
    pub(crate) fn validate_successor(self, next: Self) -> Result<()> {
        if self.source != next.source || self.owner != next.owner
            || next.epoch < self.epoch || next.revision < self.revision
            || (next.epoch == self.epoch && next.end() < self.end())
            || (next.epoch == self.epoch && next.revision == self.revision && next != self)
            || (next.epoch > self.epoch && next.revision <= self.revision)
        { return Err("stale/foreign owner feature lease".into()); }
        Ok(())
    }
    /// Retained context rows are valid only within the same sealed owner epoch.
    /// Returned rows describe old-buffer eviction then new-buffer prefix reuse;
    /// a reset or disjoint window requires full projection, never guessed reuse.
    pub(crate) fn retained_context_rows(self, next: Self) -> Result<(usize, usize)> {
        self.validate_successor(next)?;
        if self.epoch != next.epoch || next.start >= self.end() {
            return Ok((0, 0));
        }
        if next.start < self.start {
            return Err("owner context window moved backwards".into());
        }
        let evicted = next.start - self.start;
        let retained = self.end() - next.start;
        if retained > next.rows || evicted + retained != self.rows {
            return Err("invalid retained owner context extent".into());
        }
        Ok((retained, evicted))
    }
}
/// The mutable capture borrow prevents reset/overwrite while a consumer copies.
/// Cross-stream users must complete their copy before releasing this lease.
#[cfg(feature = "qwen35-augur-dflash-bridge")]
pub(crate) struct OwnerFeatureLease<'a> {
    capture: &'a mut Capture,
    identity: OwnerFeatureIdentity,
}
#[cfg(feature = "qwen35-augur-dflash-bridge")]
impl OwnerFeatureLease<'_> {
    pub(crate) fn identity(&self) -> OwnerFeatureIdentity { self.identity }
    pub(crate) fn stream(&self) -> &OwnedCudaStream { &self.capture.stream }
    pub(crate) fn view(&self) -> Result<augur_backend::CudaBufferView> {
        Ok(self.capture.chronological.view(0, self.identity.rows * ROW_BYTES)?)
    }
}

pub(super) struct Capture {
    stream: Rc<OwnedCudaStream>,
    ring: OwnedCudaBuffer,
    chronological: OwnedCudaBuffer,
    source_identity: [u8; 32],
    ledger: Ledger,
}
impl Capture {
    pub(super) fn new(stream: Rc<OwnedCudaStream>, capacity: usize, identity: [u8; 32]) -> Result<Self> {
        if identity == [0; 32] {
            return Err("missing target source identity".into());
        }
        let ledger = Ledger::new(capacity)?;
        let ring = OwnedCudaBuffer::allocate(stream.device_ordinal(), HISTORY_ROWS * ROW_BYTES)?;
        let chronological =
            OwnedCudaBuffer::allocate(stream.device_ordinal(), HISTORY_ROWS * ROW_BYTES)?;
        Ok(Self {
            stream,
            ring,
            chronological,
            source_identity: identity,
            ledger,
        })
    }
    pub(super) fn begin(&mut self, first: usize, rows: usize) -> Result<()> {
        self.ledger.begin(first, rows)
    }
    pub(super) fn layer(
        &mut self,
        layer: usize,
        first: usize,
        rows: usize,
        source: &OwnedCudaBuffer,
    ) -> Result<()> {
        let Ok(slot) = DFLASH_TARGET_CAPTURE_LAYER_IDS.binary_search(&layer) else {
            return Ok(());
        };
        self.ledger.check_layer(slot, first, rows)?;
        // Validate the entire source before starting either half of a wrap copy.
        source.view(0, rows * LAYER_BYTES)?;
        for segment in ring_segments(first, rows)? {
            // SAFETY: source belongs to the parent target and remains retained
            // through its Drop drain. Ring and parent use the exact same stream.
            // No source overwrite can pass these post-residual D2D enqueues.
            unsafe {
                self.ring.copy_from_device_2d_async_unchecked(
                    &self.stream,
                    segment.destination_row * ROW_BYTES + slot * LAYER_BYTES,
                    ROW_BYTES,
                    source.view(segment.source_row * LAYER_BYTES, segment.rows * LAYER_BYTES)?,
                    LAYER_BYTES,
                    LAYER_BYTES,
                    segment.rows,
                )?;
            }
        }
        self.ledger.mark_layer(slot, first, rows);
        Ok(())
    }
    /// Global-wide phase only: copy the intersection with the FINAL2048 rows,
    /// including an offset inside a chunk. Earlier rows never touch the ring.
    pub(super) fn layer_window(&mut self,layer:usize,first:usize,rows:usize,
        source:&OwnedCudaBuffer)->Result<()> {
        let Some((_,end))=self.ledger.pending else {return Err("capture operation absent".into())};
        let last=first.checked_add(rows).ok_or("capture range overflow")?;
        if last>end {return Err("capture chunk exceeds operation".into());}
        let begin=first.max(end.saturating_sub(HISTORY_ROWS));
        if begin>=last {return Ok(());}
        let Ok(slot)=DFLASH_TARGET_CAPTURE_LAYER_IDS.binary_search(&layer) else {return Ok(())};
        let count=last-begin;let offset=begin-first;
        self.ledger.check_layer(slot,begin,count)?;
        source.view(0,rows*LAYER_BYTES)?;
        for segment in ring_segments(begin,count)? {
            // SAFETY: same serial stream and retained parent/phase source;
            // offset is the validated final-window intersection, not row0.
            unsafe {self.ring.copy_from_device_2d_async_unchecked(&self.stream,
                segment.destination_row*ROW_BYTES+slot*LAYER_BYTES,ROW_BYTES,
                source.view((offset+segment.source_row)*LAYER_BYTES,segment.rows*LAYER_BYTES)?,
                LAYER_BYTES,LAYER_BYTES,segment.rows)?;}
        }
        self.ledger.mark_layer(slot,begin,count);Ok(())
    }
    pub(super) fn commit(&mut self) -> Result<()> {
        self.ledger.commit()
    }
    pub(super) fn reset(&mut self) -> Result<()> {
        self.ledger.reset()
    }
    /// Diagnostic bridge only: genuine payload SHA plus its already-copied
    /// chronological device range. Caller must retain this target until the
    /// draft constructor's synchronous D2D finishes. Never an owner-token hash
    /// masquerading as a digest of feature bytes.
    #[cfg(feature = "qwen35-augur-dflash-bridge")]
    pub(super) fn snapshot_and_device_view(
        &self,
    ) -> Result<(FeatureSnapshot, augur_backend::CudaBufferView)> {
        let snapshot = self.snapshot()?;
        let view = self
            .chronological
            .view(0, snapshot.little_endian_f32.len())?;
        Ok((snapshot, view))
    }
    #[cfg(feature = "qwen35-augur-dflash-bridge")]
    pub(super) fn committed_identity(&self) -> Result<(u64, u64, usize)> {
        self.ledger.range()?;
        Ok((self.ledger.epoch, self.ledger.generation, self.ledger.end))
    }
    #[cfg(feature = "qwen35-augur-dflash-bridge")]
    pub(super) fn device_lease(&mut self) -> Result<OwnerFeatureLease<'_>> {
        let (start, rows) = self.copy_chronological()?;
        let identity = OwnerFeatureIdentity {
            source: self.source_identity, owner: self.stream.owner_instance_key()?,
            epoch: self.ledger.epoch, revision: self.ledger.generation, start, rows,
        };
        Ok(OwnerFeatureLease { capture: self, identity })
    }
    fn copy_chronological(&self) -> Result<(usize, usize)> {
        let (first, rows) = self.ledger.range()?;
        // Convert ring slots to chronological rows with at most two D2D copies.
        // No bytes outside the initialized extent are read, hashed or returned.
        for segment in ring_segments(first, rows)? {
            // SAFETY: both allocations and this exact stream are self-owned;
            // synchronous readback below and Drop also drain on partial errors.
            unsafe {
                self.chronological.copy_from_device_async_unchecked(
                    &self.stream,
                    segment.source_row * ROW_BYTES,
                    self.ring.view(
                        segment.destination_row * ROW_BYTES,
                        segment.rows * ROW_BYTES,
                    )?,
                )?;
            }
        }
        Ok((first, rows))
    }
    pub(super) fn snapshot(&self) -> Result<FeatureSnapshot> {
        let (first, rows) = self.copy_chronological()?;
        let mut bytes = vec![0; rows * ROW_BYTES];
        self.chronological
            .view(0, bytes.len())?
            .copy_to_host_sync(&self.stream, &mut bytes)?;
        if bytes
            .chunks_exact(4)
            .any(|b| !f32::from_le_bytes(b.try_into().unwrap()).is_finite())
        {
            return Err("nonfinite captured target feature".into());
        }
        Ok(FeatureSnapshot {
            receipt: DFlashTargetFeatureCaptureReceipt {
                schema: DFLASH_TARGET_CAPTURE_SCHEMA.into(),
                request_generation: self.ledger.generation,
                context_start_position: first,
                context_len: rows,
                target_layer_ids: DFLASH_TARGET_CAPTURE_LAYER_IDS.to_vec(),
                target_hidden_size: H,
                feature_width: WIDTH,
                feature_elements: rows * WIDTH,
                source_target_model_sha256: self.source_identity,
                feature_sha256: Sha256::digest(&bytes).into(),
            },
            owner_epoch: self.ledger.epoch,
            little_endian_f32: bytes,
        })
    }
}
impl Drop for Capture {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
    }
}

impl CheckpointCudaTarget {
    /// Install once on an empty explicit Augur owner. This enables capture,
    /// NOT DFlash generation, speculative writes, commit or token publication.
    pub fn enable_dflash_feature_capture(&mut self) -> Result<()> {
        self.validate_cursor()?;
        if self.consumed != 0
            || self.dflash_capture.is_some()
            || self.info["profile"] != "quasar_nvfp4"
            || self.info["policy"] != "checkpoint_quasar_qpn_all_workspace_shape_small_pair_f64_v1"
            || self.info["kv_dtype"] != "Q8_0"
            || self.info["rmsnorm_schedule"] != "same_tree_warp_cached5120_v1"
            || self.info["qpn_input_staging_schedule"] != "fused_cached_norm_qkv_v1"
            || self.ffn_batch.is_none()
        {
            return Err(
                "capture requires an empty explicit Augur production numerical profile".into(),
            );
        }
        let canonical_identity = serde_json::to_vec(&serde_json::json!({
            "schema":"quasar-dflash-source-identity-v1", "profile":"quasar_nvfp4",
            "authenticated_checkpoint":self._artifact.identity()
        }))?;
        self.install_feature_capture(canonical_identity)
    }
    /// Shared storage/ledger only; callers provide their own numerical identity.
    pub(super) fn install_feature_capture(&mut self, canonical_identity: Vec<u8>) -> Result<()> {
        if self.consumed != 0 || self.dflash_capture.is_some() || self.poisoned {
            return Err("feature capture requires an empty healthy owner".into());
        }
        let before = cuda_device_allocation_snapshot();
        let capture = Capture::new(
            Rc::clone(&self.stream),
            self.capacity,
            Sha256::digest(&canonical_identity).into(),
        )?;
        let after = cuda_device_allocation_snapshot();
        let added = after.live_bytes - before.live_bytes;
        self.info["dflash_feature_capture"] = serde_json::json!({
            "schema":"quasar-device-post-layer-ring-v1", "enabled":true,
            "layers_zero_based":DFLASH_TARGET_CAPTURE_LAYER_IDS,
            "capture_point":"post_attention_plus_FFN_residual_before_next_layer",
            "layout":"chronological_row_then_layer_then_hidden_F32_le",
            "history_capacity_rows":HISTORY_ROWS, "ring_bytes":HISTORY_ROWS * ROW_BYTES,
            "chronological_workspace_bytes":HISTORY_ROWS * ROW_BYTES,
            "added_live_bytes":added, "source_identity_sha256":format!("{:x}",Sha256::digest(&canonical_identity)),
            "source_identity_contract":serde_json::from_slice::<serde_json::Value>(&canonical_identity)?,
            "hot_capture_host_transfers":false, "diagnostic_readback_is_timing":false,
            "selected_pending_token_included":false, "drafting_enabled":false
        });
        self.info["owned_live_bytes_including_KV_state_and_scratch"] = serde_json::json!(
            self.info["owned_live_bytes_including_KV_state_and_scratch"]
                .as_u64()
                .ok_or("memory receipt")?
                + added
        );
        self.info["owned_live_allocations"] = serde_json::json!(
            self.info["owned_live_allocations"]
                .as_u64()
                .ok_or("allocation receipt")?
                + after.live_allocations
                - before.live_allocations
        );
        self.dflash_capture = Some(capture);
        Ok(())
    }
    /// Explicit terminal diagnostic. Never call inside a timed request. The
    /// returned bytes and SHA256 belong to this model, operation and reset epoch.
    pub fn dflash_feature_snapshot(&mut self) -> Result<FeatureSnapshot> {
        self.validate_cursor()?;
        let capture = self
            .dflash_capture
            .as_ref()
            .ok_or("DFlash capture is not enabled")?;
        capture.ledger.range()?; // Empty is a precondition error, not device poison.
        if capture.ledger.end != self.consumed {
            self.poisoned = true;
            return Err("target/feature cursor drift".into());
        }
        let result = capture.snapshot();
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }
}

#[cfg(test)]
mod tests;

#[cfg(all(test, feature = "qwen35-augur-dflash-bridge"))]
mod resident_tests;

#[cfg(all(windows, feature="qwen35-augur-http"))]
impl Capture {
    pub(super) fn checkpoint_ring(&self,consumed:usize)->Result<(&OwnedCudaBuffer,usize)> {
        if consumed==0 || consumed>=self.ledger.capacity {return Err("feature checkpoint extent".into());}
        Ok((&self.ring,consumed.min(HISTORY_ROWS)*ROW_BYTES))
    }
    pub(super) fn checkpoint_check(&self,consumed:usize)->Result<()> {
        let (first,rows)=self.ledger.range()?;
        if first+rows!=consumed {return Err("feature checkpoint cursor".into());}Ok(())
    }
    pub(super) fn checkpoint_resume(&mut self,consumed:usize)->Result<()> {
        if self.ledger.pending.is_some() || self.ledger.end!=0 || consumed==0 || consumed>=self.ledger.capacity {
            return Err("feature checkpoint requires a reset/new local epoch".into());}
        // Old owner epochs, pointers, proposal generations and GPU handles never
        // cross a checkpoint. Rebuild the initialized-window ledger in this owner.
        let generation=self.ledger.generation.checked_add(1).ok_or("feature generation overflow")?;
        self.ledger.positions.fill(None);self.ledger.masks.fill(0);
        for pos in consumed.saturating_sub(HISTORY_ROWS)..consumed {
            self.ledger.positions[pos%HISTORY_ROWS]=Some(pos);self.ledger.masks[pos%HISTORY_ROWS]=ALL_LAYERS;
        }
        self.ledger.end=consumed;self.ledger.generation=generation;self.ledger.range()?;Ok(())
    }
}
