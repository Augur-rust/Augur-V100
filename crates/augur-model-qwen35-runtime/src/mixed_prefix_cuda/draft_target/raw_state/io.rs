//! Isolated read-only raw-state capture machinery; not a product qualification.
//!
//! The runtime adapter must own an exclusive transaction and retain every buffer
//! through the final drain. This module neither restores nor advances model state.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs::{self, OpenOptions};
use std::io::{self, Read, Write};
use std::path::Path;

// Derived from the frozen standalone component; runtime adapter lives in the
// parent module. Original experiment source/binary remain unchanged.

pub const MAX_CHUNK: usize = 8 * 1024 * 1024;
pub const CAPACITY: usize = 196608;
pub const HISTORY: usize = 2048;

#[cfg(test)]
#[path = "io_tests.rs"]
mod tests;

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn require(value: bool, message: &str) -> io::Result<()> {
    if value {
        Ok(())
    } else {
        Err(invalid(message))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Boundary {
    pub model: String,
    pub config: String,
    pub numerical_identity: String,
    pub prompt_ids: String,
    pub committed_input_ids: String,
    pub cursor: usize,
    pub anchor: u32,
    pub context_start: usize,
    pub context_rows: usize,
}

impl Boundary {
    pub fn validate(&self) -> io::Result<()> {
        for hash in [
            &self.model,
            &self.config,
            &self.numerical_identity,
            &self.prompt_ids,
            &self.committed_input_ids,
        ] {
            require(
                hash.len() == 64
                    && hash
                        .bytes()
                        .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)),
                "noncanonical identity digest",
            )?;
        }
        require((HISTORY..=CAPACITY).contains(&self.cursor), "cursor")?;
        require(self.anchor < 248320, "anchor vocabulary")?;
        require(
            self.context_rows == HISTORY && self.context_start == self.cursor - HISTORY,
            "full absolute feature history",
        )
    }

    pub fn matches(&self, other: &Self) -> io::Result<()> {
        self.validate()?;
        other.validate()?;
        require(self == other, "different committed input boundary")
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Scope {
    TargetState,
    Features,
    Anchor,
    Hidden,
    Logits,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Plane {
    pub name: String,
    pub scope: Scope,
    pub physical_bytes: usize,
    pub committed_bytes: usize,
}

fn plane(name: String, scope: Scope, physical: usize, committed: usize) -> Plane {
    Plane {
        name,
        scope,
        physical_bytes: physical,
        committed_bytes: committed,
    }
}

/// Required target inventory. Future KV capacity is captured but never compared
/// as logical committed state. Draft-private validity/positions are a separate gate.
pub fn target_inventory(cursor: usize) -> io::Result<Vec<Plane>> {
    require((HISTORY..=CAPACITY).contains(&cursor), "inventory cursor")?;
    let mut planes = Vec::new();
    for layer in 0..64 {
        if layer % 4 == 3 {
            for kind in ["key", "value"] {
                planes.push(plane(
                    format!("layer{layer:02}.{kind}"),
                    Scope::TargetState,
                    CAPACITY * 1088,
                    cursor * 1088,
                ));
            }
        } else {
            for (kind, count) in [("conv_history", 30720), ("recurrent", 786432)] {
                planes.push(plane(
                    format!("layer{layer:02}.{kind}"),
                    Scope::TargetState,
                    count * 4,
                    count * 4,
                ));
            }
        }
    }
    planes.push(plane(
        "features".into(),
        Scope::Features,
        HISTORY * 5 * 5120 * 4,
        HISTORY * 5 * 5120 * 4,
    ));
    planes.push(plane("anchor".into(), Scope::Anchor, 4, 4));
    for layer in 0..64 {
        planes.push(plane(
            format!("layer{layer:02}.hidden"),
            Scope::Hidden,
            5120 * 4,
            5120 * 4,
        ));
    }
    planes.push(plane(
        "logits".into(),
        Scope::Logits,
        248320 * 4,
        248320 * 4,
    ));
    Ok(planes)
}

/// Export only initialized logical ranges while leaving full device allocations unchanged.
/// Receipt target_inventory_complete stays false: future capacity was NOT exported.
pub fn committed_inventory(cursor: usize) -> io::Result<Vec<Plane>> {
    let mut result=target_inventory(cursor)?;
    for p in &mut result {p.physical_bytes=p.committed_bytes;}
    Ok(result)
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Snapshot {
    pub boundary: Boundary,
    // Per-token reference and per-block candidate generation need not be equal.
    pub owner: String,
    pub generation: u64,
    pub backend: String,
    pub planes: Vec<Plane>,
}

impl Snapshot {
    fn validate(&self) -> io::Result<()> {
        self.boundary.validate()?;
        require(
            !self.owner.is_empty() && !self.backend.is_empty(),
            "owner/backend absent",
        )?;
        require(
            !self.planes.is_empty() && self.planes.len() <= 512,
            "plane count",
        )?;
        let mut names = BTreeSet::new();
        for p in &self.planes {
            require(
                !p.name.is_empty()
                    && p.name.len() < 80
                    && p.name.bytes().all(|c| {
                        c.is_ascii_lowercase() || c.is_ascii_digit() || b"._-".contains(&c)
                    })
                    && !p.name.contains("..")
                    && names.insert(&p.name),
                "plane name/duplicate",
            )?;
            require(
                p.physical_bytes > 0
                    && p.physical_bytes % 4 == 0
                    && p.committed_bytes > 0
                    && p.committed_bytes <= p.physical_bytes
                    && p.committed_bytes % 4 == 0,
                "plane extent",
            )?;
        }
        Ok(())
    }
}

/// Implementors must synchronize all producers, retain exclusive owner lifetime,
/// and validate the real stream/submission proof on each completed read.
/// CPU fixtures explicitly identify themselves and never provide CUDA authority.
pub trait ReadbackSource {
    fn quiesce(&mut self) -> io::Result<()>;
    fn snapshot(&self) -> io::Result<Snapshot>;
    fn read_completed(&mut self, plane: usize, offset: usize, out: &mut [u8]) -> io::Result<()>;
    fn drain(&mut self) -> io::Result<()>;
}

#[derive(Clone, Debug)]
pub struct Budget {
    pub chunk_bytes: usize,
    pub max_total_bytes: u64,
    pub max_chunks: usize,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Chunk {
    pub file: String,
    pub plane: usize,
    pub offset: usize,
    pub bytes: usize,
    pub region: String,
    pub sha256: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureReceipt {
    pub schema: String,
    pub status: String,
    pub snapshot: Snapshot,
    pub chunks: Vec<Chunk>,
    pub total_bytes: u64,
    pub host_staging_bytes: usize,
    pub drain_completed: bool,
    pub target_inventory_complete: bool,
    pub draft_private_state_qualified: bool,
    pub model_state_equivalence_qualified: bool,
    pub product_tps_permitted: bool,
}

fn is_reparse(meta: &fs::Metadata) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        meta.file_attributes() & 0x400 != 0
    }
    #[cfg(not(windows))]
    {
        meta.file_type().is_symlink()
    }
}

fn safe_path(path: &Path) -> io::Result<()> {
    require(
        path.is_absolute()
            && !path
                .components()
                .any(|c| matches!(c, std::path::Component::ParentDir)),
        "absolute nontraversing path required",
    )?;
    for ancestor in path.ancestors() {
        let meta = fs::symlink_metadata(ancestor)?;
        require(
            !meta.file_type().is_symlink() && !is_reparse(&meta),
            "reparse/symlink path",
        )?;
    }
    Ok(())
}

fn namespace(path: &Path, expected: &BTreeSet<String>) -> io::Result<()> {
    safe_path(path)?;
    let mut actual = BTreeSet::new();
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let meta = fs::symlink_metadata(entry.path())?;
        require(
            meta.is_file() && !meta.file_type().is_symlink() && !is_reparse(&meta),
            "nonregular capture file",
        )?;
        actual.insert(
            entry
                .file_name()
                .into_string()
                .map_err(|_| invalid("filename encoding"))?,
        );
    }
    require(&actual == expected, "capture namespace drift")
}

fn write_new(path: &Path, bytes: &[u8]) -> io::Result<()> {
    safe_path(path.parent().ok_or_else(|| invalid("output parent"))?)?;
    let mut file = OpenOptions::new().create_new(true).write(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    safe_path(path)
}

/// Independent readback validation. This checks stored bytes/namespace and
/// coverage; it does not manufacture model equivalence or CUDA authority.
pub fn verify_capture(out: &Path, budget: &Budget) -> io::Result<CaptureReceipt> {
    require(
        budget.chunk_bytes > 0 && budget.chunk_bytes <= MAX_CHUNK,
        "verification staging bound",
    )?;
    safe_path(out)?;
    let manifest_path = out.join("complete.json");
    safe_path(&manifest_path)?;
    let size = fs::metadata(&manifest_path)?.len();
    require(size <= MAX_CHUNK as u64, "manifest bound")?;
    let raw = fs::read(&manifest_path)?;
    let receipt: CaptureReceipt = serde_json::from_slice(&raw).map_err(io::Error::other)?;
    receipt.snapshot.validate()?;
    require(
        receipt.schema == "QWEN35-BOUNDED-RAW-STATE-CAPTURE-V1"
            && receipt.status == "CAPTURE_ONLY_NOT_STATE_EQUIVALENCE"
            && receipt.drain_completed
            && !receipt.draft_private_state_qualified
            && !receipt.model_state_equivalence_qualified
            && !receipt.product_tps_permitted,
        "capture claim ceiling",
    )?;
    require(
        receipt.host_staging_bytes > 0
            && receipt.host_staging_bytes <= budget.chunk_bytes
            && receipt.host_staging_bytes.is_multiple_of(4)
            && receipt.chunks.len() <= budget.max_chunks,
        "capture limits",
    )?;
    require(
        receipt.target_inventory_complete
            == (receipt.snapshot.planes == target_inventory(receipt.snapshot.boundary.cursor)?),
        "target coverage claim mismatch",
    )?;
    let mut expected = BTreeSet::from(["complete.json".to_string()]);
    for (index, chunk) in receipt.chunks.iter().enumerate() {
        require(
            chunk.file == format!("chunk{index:08}.bin"),
            "chunk filename/order",
        )?;
        expected.insert(chunk.file.clone());
    }
    namespace(out, &expected)?;
    let mut scratch = vec![0_u8; receipt.host_staging_bytes];
    let mut index = 0_usize;
    let mut total = 0_u64;
    for (plane_index, p) in receipt.snapshot.planes.iter().enumerate() {
        for (begin, end, region) in [
            (0, p.committed_bytes, "committed"),
            (p.committed_bytes, p.physical_bytes, "future_capacity"),
        ] {
            let mut offset = begin;
            while offset < end {
                let count = (end - offset).min(scratch.len());
                let c = receipt
                    .chunks
                    .get(index)
                    .ok_or_else(|| invalid("missing chunk"))?;
                require(
                    c.plane == plane_index
                        && c.offset == offset
                        && c.bytes == count
                        && c.region == region,
                    "chunk gap/overlap/region/extent",
                )?;
                let path = out.join(&c.file);
                safe_path(&path)?;
                let mut file = fs::File::open(&path)?;
                let before = file.metadata()?;
                require(
                    before.is_file() && before.len() == count as u64,
                    "chunk file extent",
                )?;
                file.read_exact(&mut scratch[..count])?;
                require(file.read(&mut [0_u8; 1])? == 0, "chunk grew while reading")?;
                let after = file.metadata()?;
                require(
                    before.len() == after.len() && before.modified()? == after.modified()?,
                    "chunk descriptor drift",
                )?;
                require(
                    format!("{:x}", Sha256::digest(&scratch[..count])) == c.sha256,
                    "chunk content hash",
                )?;
                total = total
                    .checked_add(count as u64)
                    .ok_or_else(|| invalid("verified byte overflow"))?;
                require(total <= budget.max_total_bytes, "verified byte budget")?;
                offset += count;
                index += 1;
            }
        }
    }
    require(
        index == receipt.chunks.len() && total == receipt.total_bytes,
        "total coverage mismatch",
    )?;
    namespace(out, &expected)?;
    require(
        fs::read(&manifest_path)? == raw,
        "manifest changed during verification",
    )?;
    Ok(receipt)
}

fn capture_inner(
    source: &mut impl ReadbackSource,
    out: &Path,
    budget: &Budget,
) -> io::Result<CaptureReceipt> {
    require(
        budget.chunk_bytes > 0
            && budget.chunk_bytes <= MAX_CHUNK
            && budget.chunk_bytes.is_multiple_of(4),
        "bounded aligned chunk",
    )?;
    source.quiesce()?;
    let snapshot = source.snapshot()?;
    snapshot.validate()?;
    let mut total = 0_u64;
    let mut count = 0_usize;
    for p in &snapshot.planes {
        total = total
            .checked_add(p.physical_bytes as u64)
            .ok_or_else(|| invalid("total overflow"))?;
        for size in [p.committed_bytes, p.physical_bytes - p.committed_bytes] {
            count = count
                .checked_add(size.div_ceil(budget.chunk_bytes))
                .ok_or_else(|| invalid("chunk overflow"))?;
        }
    }
    require(
        total <= budget.max_total_bytes && count <= budget.max_chunks,
        "capture budget exceeded",
    )?;
    require(count <= 1_000_000, "metadata count bound")?;
    safe_path(out.parent().ok_or_else(|| invalid("capture parent"))?)?;
    fs::create_dir(out)?; // Existing output is never reused, even after failure.
    safe_path(out)?;
    let mut expected = BTreeSet::new();
    namespace(out, &expected)?;
    let mut staging = vec![0_u8; budget.chunk_bytes];
    let mut chunks = Vec::with_capacity(count);
    for (index, p) in snapshot.planes.iter().enumerate() {
        for (begin, end, region) in [
            (0, p.committed_bytes, "committed"),
            (p.committed_bytes, p.physical_bytes, "future_capacity"),
        ] {
            let mut offset = begin;
            while offset < end {
                require(
                    source.snapshot()? == snapshot,
                    "source identity/extent drift before read",
                )?;
                let bytes = (end - offset).min(staging.len());
                source.read_completed(index, offset, &mut staging[..bytes])?;
                require(
                    source.snapshot()? == snapshot,
                    "source identity/extent drift after read",
                )?;
                let file = format!("chunk{:08}.bin", chunks.len());
                write_new(&out.join(&file), &staging[..bytes])?;
                expected.insert(file.clone());
                chunks.push(Chunk {
                    file,
                    plane: index,
                    offset,
                    bytes,
                    region: region.into(),
                    sha256: format!("{:x}", Sha256::digest(&staging[..bytes])),
                });
                offset += bytes;
            }
        }
    }
    source.drain()?;
    require(source.snapshot()? == snapshot, "source terminal drift")?;
    namespace(out, &expected)?;
    let complete = snapshot.planes == target_inventory(snapshot.boundary.cursor)?;
    Ok(CaptureReceipt {
        schema: "QWEN35-BOUNDED-RAW-STATE-CAPTURE-V1".into(),
        status: "CAPTURE_ONLY_NOT_STATE_EQUIVALENCE".into(),
        snapshot,
        chunks,
        total_bytes: total,
        host_staging_bytes: staging.len(),
        drain_completed: true,
        target_inventory_complete: complete,
        draft_private_state_qualified: false,
        model_state_equivalence_qualified: false,
        product_tps_permitted: false,
    })
}

/// The final receipt is published only after every chunk and successful drain.
/// Failure preserves partial chunks without a completion receipt. Callers must
/// poison their diagnostic transaction on error; a failed drain is not hidden.
pub fn capture(
    source: &mut impl ReadbackSource,
    out: &Path,
    budget: &Budget,
) -> io::Result<CaptureReceipt> {
    let result = capture_inner(source, out, budget);
    match result {
        Ok(receipt) => {
            let raw = serde_json::to_vec_pretty(&receipt).map_err(io::Error::other)?;
            let pending = out.join("receipt.pending");
            write_new(&pending, &raw)?;
            let mut expected: BTreeSet<_> = receipt.chunks.iter().map(|c| c.file.clone()).collect();
            expected.insert("receipt.pending".into());
            namespace(out, &expected)?;
            // Linking a fully flushed file publishes atomically without
            // replacing an existing destination. Unsupported filesystems fail.
            fs::hard_link(&pending, out.join("complete.json"))?;
            fs::remove_file(pending)?;
            Ok(receipt)
        }
        Err(error) => match source.drain() {
            Ok(()) => Err(error),
            Err(drain) => Err(io::Error::other(format!(
                "capture failed ({error}); drain failed ({drain})"
            ))),
        },
    }
}
