//! Bounded checkpoint transfer protocol, not a model-state or publication authority.
//!
//! A runtime adapter must derive the expected plan from a complete, quiescent
//! session (including draft state and ledgers). The storage adapter must supply
//! immutable, path-safe, stable reads and manifest-last publication. The local
//! Windows filesystem backend and opt-in native Qwen/CUDA adapter are separate
//! layers. Their GPU semantic qualification is independent. A transfer receipt does not
//! publish an HTTP session or qualify semantic continuation or power-loss safety.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt;
use std::io::{self, Read};

#[path = "session_checkpoint/parallel_restore.rs"]
mod parallel_restore;
pub use parallel_restore::restore_parallel_verified;
#[path = "session_checkpoint/pipelined_restore.rs"]
mod pipelined_restore;
pub use pipelined_restore::restore_pipelined_verified;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ParallelRestoreSchedule {
    #[default]
    Batch,
    Pipeline,
}
impl ParallelRestoreSchedule {
    pub fn label(self) -> &'static str {
        match self { Self::Batch => "batch", Self::Pipeline => "pipeline" }
    }
    pub fn parse(value: &str) -> io::Result<Self> {
        match value {
            "batch" => Ok(Self::Batch),
            "pipeline" => Ok(Self::Pipeline),
            _ => Err(io::Error::other("restore schedule must be batch or pipeline")),
        }
    }
}


pub const SCHEMA: &str = "cq-qwen35-session-checkpoint-transfer-v1";
pub const MAX_CHUNK_BYTES: usize = 8 * 1024 * 1024;
pub const MAX_MANIFEST_BYTES: usize = 4 * 1024 * 1024;
pub const MAX_COMPONENTS: usize = 512;
pub const MAX_CHUNKS: usize = 16384;
pub const MAX_PAYLOAD_BYTES: u64 = 16 * 1024 * 1024 * 1024;

#[path = "session_checkpoint/metadata.rs"]
pub mod metadata;

/// Input-only checkpoint descriptors. They do not publish a generated session
/// or grant CUDA restore authority; native capture/restore is a separate gate.
#[path = "session_checkpoint/input_prefix.rs"]
pub mod input_prefix;

#[path = "session_checkpoint/http.rs"]
pub mod http;

/// Windows local-filesystem backend. Other platforms need their own handle-
/// relative/path-stability implementation; they are not silently treated as safe.
#[cfg(windows)]
#[path = "session_checkpoint/filesystem.rs"]
pub mod filesystem;

#[cfg(test)]
#[path = "session_checkpoint/tests.rs"]
mod tests;

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn require(condition: bool, message: &str) -> io::Result<()> {
    if condition {
        Ok(())
    } else {
        Err(invalid(message))
    }
}

pub fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        && value.bytes().any(|b| b != b'0')
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Component {
    /// Logical role only, never a filesystem path.
    pub name: String,
    /// Exact initialized bytes, not the allocation's unused capacity.
    pub bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransferPlan {
    /// Hash of the independently expected complete session contract: artifacts,
    /// build/layout/numerics, namespace, cursor/ledgers, draft state and positions.
    /// This is not inferred from the untrusted manifest during restore.
    pub binding_sha256: String,
    /// Exact ordered inventory supplied by the runtime adapter.
    pub components: Vec<Component>,
}

impl TransferPlan {
    pub fn validate(&self) -> io::Result<u64> {
        require(
            valid_digest(&self.binding_sha256),
            "invalid expected binding",
        )?;
        require(
            !self.components.is_empty() && self.components.len() <= MAX_COMPONENTS,
            "component count bound",
        )?;
        let mut names = BTreeSet::new();
        let mut total = 0u64;
        for component in &self.components {
            let name = &component.name;
            require(
                !name.is_empty()
                    && name.len() <= 96
                    && name.bytes().all(|b| {
                        b.is_ascii_lowercase() || b.is_ascii_digit() || b"._-".contains(&b)
                    })
                    && !name.contains("..")
                    && names.insert(name),
                "component role or duplicate",
            )?;
            require(component.bytes > 0, "empty component")?;
            total = total
                .checked_add(component.bytes)
                .ok_or_else(|| invalid("payload overflow"))?;
            require(total <= MAX_PAYLOAD_BYTES, "payload quota")?;
        }
        Ok(total)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Chunk {
    pub component: usize,
    pub offset: u64,
    pub bytes: usize,
    pub sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema: String,
    pub plan: TransferPlan,
    pub chunk_bytes: usize,
    pub chunks: Vec<Chunk>,
}

fn chunk_count(plan: &TransferPlan, chunk_bytes: usize) -> io::Result<usize> {
    plan.validate()?;
    require(
        (1..=MAX_CHUNK_BYTES).contains(&chunk_bytes),
        "chunk staging bound",
    )?;
    let mut total = 0u64;
    for component in &plan.components {
        total = total
            .checked_add(component.bytes.div_ceil(chunk_bytes as u64))
            .ok_or_else(|| invalid("chunk count overflow"))?;
        require(total <= MAX_CHUNKS as u64, "chunk count quota")?;
    }
    Ok(total as usize)
}

impl Manifest {
    pub fn validate(&self, expected: &TransferPlan) -> io::Result<u64> {
        let total = expected.validate()?;
        require(self.schema == SCHEMA, "checkpoint transfer schema")?;
        require(
            &self.plan == expected,
            "independently expected session plan mismatch",
        )?;
        require(
            self.chunks.len() == chunk_count(expected, self.chunk_bytes)?,
            "exact chunk count",
        )?;
        let mut index = 0;
        for (component, spec) in expected.components.iter().enumerate() {
            let mut offset = 0;
            while offset < spec.bytes {
                let bytes = (spec.bytes - offset).min(self.chunk_bytes as u64) as usize;
                let chunk = &self.chunks[index];
                require(
                    chunk.component == component
                        && chunk.offset == offset
                        && chunk.bytes == bytes
                        && valid_digest(&chunk.sha256),
                    "ordered contiguous component extent or digest",
                )?;
                offset += bytes as u64;
                index += 1;
            }
        }
        Ok(total)
    }

    /// Strict, bounded and externally hash-bound. Unknown/duplicate typed fields
    /// and noncanonical JSON are rejected before any destination mutation.
    pub fn decode(
        bytes: &[u8],
        expected_sha256: &str,
        expected: &TransferPlan,
    ) -> io::Result<Self> {
        expected.validate()?;
        let manifest = Self::inspect_content_bound(bytes, expected_sha256)?;
        manifest.validate(expected)?;
        Ok(manifest)
    }

    /// Bootstrap inspection only. It verifies storage structure/content, NOT
    /// an independently expected live session. Never pass its own plan back as
    /// a substitute for runtime-derived restore admission.
    pub(crate) fn inspect_content_bound(bytes: &[u8], expected_sha256: &str) -> io::Result<Self> {
        require(
            !bytes.is_empty() && bytes.len() <= MAX_MANIFEST_BYTES,
            "manifest byte bound",
        )?;
        require(
            valid_digest(expected_sha256) && sha256(bytes) == expected_sha256,
            "external manifest hash mismatch",
        )?;
        let manifest: Self = serde_json::from_slice(bytes).map_err(io::Error::other)?;
        manifest.validate(&manifest.plan)?;
        require(
            serde_json::to_vec(&manifest).map_err(io::Error::other)? == bytes,
            "manifest requires canonical serializer bytes",
        )?;
        Ok(manifest)
    }

    pub fn encode(&self) -> io::Result<Vec<u8>> {
        self.validate(&self.plan)?;
        let bytes = serde_json::to_vec(self).map_err(io::Error::other)?;
        require(bytes.len() <= MAX_MANIFEST_BYTES, "manifest byte bound")?;
        Ok(bytes)
    }
}

/// Role and absolute component offset are part of the chunk content address.
/// Identical payload bytes in another role cannot be silently substituted.
fn chunk_digest(component: &Component, offset: u64, bytes: &[u8]) -> String {
    let mut hash = Sha256::new();
    hash.update(b"CQ-QWEN35-SESSION-CHUNK-V1\0");
    hash.update((component.name.len() as u64).to_le_bytes());
    hash.update(component.name.as_bytes());
    hash.update(component.bytes.to_le_bytes());
    hash.update(offset.to_le_bytes());
    hash.update((bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
    format!("{:x}", hash.finalize())
}

pub trait CaptureSource {
    /// Acquire exclusive state ownership and quiesce; retain all buffers until
    /// finish/abort drains every submitted copy. Return the actual boundary.
    /// Implementations also require RAII draining on unwind; a callback error
    /// path alone is not a complete GPU buffer-lifetime guarantee.
    fn begin(&mut self) -> io::Result<TransferPlan>;
    /// Complete every access to the borrowed staging slice before returning;
    /// a queued D2H whose destination still aliases it is not sufficient.
    fn read_exact(&mut self, component: usize, offset: u64, output: &mut [u8]) -> io::Result<()>;
    /// Drain copies and revalidate that no state/identity changed during capture.
    fn finish(&mut self) -> io::Result<TransferPlan>;
    fn abort(&mut self) -> io::Result<()>;
}

pub trait CheckpointWriter {
    /// Immutable/no-clobber chunk write. A deduplicated existing chunk must be
    /// checked for exact content, not accepted merely because its name exists.
    fn write_chunk(&mut self, digest: &str, bytes: &[u8]) -> io::Result<()>;
    /// Called exactly once, after capture is drained and all chunks succeeded.
    /// The concrete store is responsible for durability and atomic publication.
    fn seal_manifest(&mut self, bytes: &[u8], digest: &str) -> io::Result<()>;
    /// Leave no visible partial checkpoint; never delete a prior sealed snapshot.
    fn abort(&mut self) -> io::Result<()>;
}

pub trait CheckpointReader {
    type ChunkReader: Read;
    /// Must open the immutable, path-safe, stable object identified by digest.
    /// The transfer layer also checks exact length, EOF and content before use.
    fn open_chunk(&mut self, digest: &str) -> io::Result<Self::ChunkReader>;
    /// Recheck the exact namespace and any held storage identity before the
    /// destination can complete. Late insertion/replacement must not be ignored.
    fn finish_read(&mut self) -> io::Result<()>;
}

pub trait RestoreTarget {
    /// Destination must be fresh and unpublished. Even a partially failing begin
    /// is followed by abort. No old stream/event/pointer/handle is resurrected.
    fn begin(&mut self, expected: &TransferPlan) -> io::Result<()>;
    /// Complete every access to these borrowed bytes before returning. They are
    /// reused immediately; do not retain a pending H2D from this staging slice.
    fn write_exact(&mut self, component: usize, offset: u64, bytes: &[u8]) -> io::Result<()>;
    /// Drain and validate restored state, but DO NOT publish a session handle.
    /// Runtime semantics and real suffix execution remain separate gates.
    fn finish_unpublished(&mut self) -> io::Result<()>;
    /// Drain and irreversibly discard/poison all partially restored state.
    fn abort(&mut self) -> io::Result<()>;
}

#[derive(Debug)]
pub struct TransferFailure {
    pub phase: &'static str,
    pub cause: io::Error,
    /// Cleanup failures are preserved; never replace them with a false clean flag.
    pub abort_errors: Vec<String>,
}

impl fmt::Display for TransferFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.phase, self.cause)?;
        for error in &self.abort_errors {
            write!(f, "; abort: {error}")?;
        }
        Ok(())
    }
}

impl std::error::Error for TransferFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct TransferReceipt {
    pub manifest_sha256: String,
    pub components: usize,
    pub chunks: usize,
    pub payload_bytes: u64,
    pub staging_bytes: usize,
}

fn failure(phase: &'static str, cause: io::Error) -> TransferFailure {
    TransferFailure {
        phase,
        cause,
        abort_errors: Vec::new(),
    }
}

fn receipt(manifest: &Manifest, digest: String) -> io::Result<TransferReceipt> {
    Ok(TransferReceipt {
        manifest_sha256: digest,
        components: manifest.plan.components.len(),
        chunks: manifest.chunks.len(),
        payload_bytes: manifest.validate(&manifest.plan)?,
        staging_bytes: manifest.chunk_bytes,
    })
}

pub fn save<S: CaptureSource, W: CheckpointWriter>(
    source: &mut S,
    store: &mut W,
    expected: &TransferPlan,
    chunk_bytes: usize,
) -> Result<TransferReceipt, TransferFailure> {
    // No source or store side effects until all known quotas are checked.
    let count = chunk_count(expected, chunk_bytes).map_err(|e| failure("save preflight", e))?;
    let mut staging = vec![0; chunk_bytes];
    let mut phase = "capture begin";
    let operation = (|| -> io::Result<TransferReceipt> {
        require(source.begin()? == *expected, "capture begin boundary drift")?;
        let mut manifest = Manifest {
            schema: SCHEMA.into(),
            plan: expected.clone(),
            chunk_bytes,
            chunks: Vec::with_capacity(count),
        };
        for (component, spec) in expected.components.iter().enumerate() {
            let mut offset = 0u64;
            while offset < spec.bytes {
                let bytes = (spec.bytes - offset).min(chunk_bytes as u64) as usize;
                phase = "capture read";
                source.read_exact(component, offset, &mut staging[..bytes])?;
                let digest = chunk_digest(spec, offset, &staging[..bytes]);
                phase = "chunk write";
                store.write_chunk(&digest, &staging[..bytes])?;
                manifest.chunks.push(Chunk {
                    component,
                    offset,
                    bytes,
                    sha256: digest,
                });
                offset += bytes as u64;
            }
        }
        phase = "capture finish";
        require(
            source.finish()? == *expected,
            "capture finish boundary drift",
        )?;
        phase = "manifest encode";
        let bytes = manifest.encode()?;
        let digest = sha256(&bytes);
        let result = receipt(&manifest, digest.clone())?;
        phase = "manifest seal";
        store.seal_manifest(&bytes, &digest)?;
        Ok(result)
    })();
    operation.map_err(|cause| {
        let mut error = failure(phase, cause);
        if let Err(e) = source.abort() {
            error.abort_errors.push(format!("capture: {e}"));
        }
        if let Err(e) = store.abort() {
            error.abort_errors.push(format!("store: {e}"));
        }
        error
    })
}

/// Independent diagnostic lane. Timers are compile-time disabled in restore().
#[derive(Debug, Default, Serialize)]
pub struct RestoreTiming {
    pub preflight_s: f64,
    pub staging_allocation_s: f64,
    pub destination_begin_s: f64,
    pub chunk_read_s: f64,
    pub chunk_verify_s: f64,
    pub destination_write_s: f64,
    pub namespace_verification_s: f64,
    pub destination_finish_s: f64,
    pub complete_s: f64,
}
fn restore_tick<const TIMED: bool>() -> Option<std::time::Instant> {
    if TIMED { Some(std::time::Instant::now()) } else { None }
}
fn restore_elapsed(tick: Option<std::time::Instant>) -> f64 {
    tick.map_or(0.0, |t| t.elapsed().as_secs_f64())
}
pub fn restore<R: CheckpointReader, T: RestoreTarget>(
    store: &mut R, target: &mut T, manifest_bytes: &[u8],
    expected_manifest_sha256: &str, expected: &TransferPlan,
) -> Result<TransferReceipt, TransferFailure> {
    restore_impl::<false, R, T>(store, target, manifest_bytes, expected_manifest_sha256, expected)
        .map(|(receipt, _)| receipt)
}
pub fn restore_with_timing<R: CheckpointReader, T: RestoreTarget>(
    store: &mut R, target: &mut T, manifest_bytes: &[u8],
    expected_manifest_sha256: &str, expected: &TransferPlan,
) -> Result<(TransferReceipt, RestoreTiming), TransferFailure> {
    restore_impl::<true, R, T>(store, target, manifest_bytes, expected_manifest_sha256, expected)
}
fn restore_impl<const TIMED: bool, R: CheckpointReader, T: RestoreTarget>(
    store: &mut R, target: &mut T, manifest_bytes: &[u8],
    expected_manifest_sha256: &str, expected: &TransferPlan,
) -> Result<(TransferReceipt, RestoreTiming), TransferFailure> {
    let started = restore_tick::<TIMED>();
    let tick = restore_tick::<TIMED>();
    let manifest = Manifest::decode(manifest_bytes, expected_manifest_sha256, expected)
        .map_err(|e| failure("restore preflight", e))?;
    let result = receipt(&manifest, expected_manifest_sha256.into())
        .map_err(|e| failure("restore preflight", e))?;
    let mut timing = RestoreTiming { preflight_s: restore_elapsed(tick), ..RestoreTiming::default() };
    let tick = restore_tick::<TIMED>();
    let mut staging = vec![0; manifest.chunk_bytes];
    timing.staging_allocation_s = restore_elapsed(tick);
    let mut phase = "restore begin";
    let operation = (|| -> io::Result<()> {
        let tick = restore_tick::<TIMED>();
        target.begin(expected)?;
        timing.destination_begin_s = restore_elapsed(tick);
        for chunk in &manifest.chunks {
            phase = "chunk read";
            let tick = restore_tick::<TIMED>();
            let mut reader = store.open_chunk(&chunk.sha256)?;
            reader.read_exact(&mut staging[..chunk.bytes])?;
            let mut tail = [0];
            require(reader.read(&mut tail)? == 0, "chunk trailing bytes")?;
            timing.chunk_read_s += restore_elapsed(tick);
            phase = "chunk verify";
            let tick = restore_tick::<TIMED>();
            require(chunk_digest(&expected.components[chunk.component], chunk.offset,
                &staging[..chunk.bytes]) == chunk.sha256, "chunk content mismatch")?;
            timing.chunk_verify_s += restore_elapsed(tick);
            phase = "restore write";
            let tick = restore_tick::<TIMED>();
            target.write_exact(chunk.component, chunk.offset, &staging[..chunk.bytes])?;
            timing.destination_write_s += restore_elapsed(tick);
        }
        phase = "store final verification";
        let tick = restore_tick::<TIMED>();
        store.finish_read()?;
        timing.namespace_verification_s = restore_elapsed(tick);
        phase = "restore finish unpublished";
        let tick = restore_tick::<TIMED>();
        target.finish_unpublished()?;
        timing.destination_finish_s = restore_elapsed(tick);
        Ok(())
    })();
    match operation {
        Ok(()) => { timing.complete_s = restore_elapsed(started); Ok((result, timing)) },
        Err(cause) => {
            let mut error = failure(phase, cause);
            if let Err(e) = target.abort() { error.abort_errors.push(format!("destination: {e}")); }
            Err(error)
        }
    }
}
