//! Exclusive local HTTP checkpoint namespace with finite, failure-retaining quotas.
//! Native payload uses its own immutable subdirectory. HTTP metadata is the last
//! publication and never makes an incomplete native checkpoint restorable.

use super::*;
use crate::session_checkpoint::http::{HttpSessionCheckpoint, MAX_HTTP_METADATA_BYTES};
use crate::session_checkpoint::metadata::{ModelBindings, NativeMathPolicy, SessionDescriptor};

#[path = "available_space.rs"]
mod available_space;

const RESERVATION: &str = "reservation.json";
const OUTER_BLOB: &str = "http.blob";
const OUTER_SEAL: &str = "http.json";
const LOCK: &str = "owner.lock";
const SCHEMA: &str = "cq-qwen35-http-slot-reservation-v1";
const OVERHEAD: u64 = 2 * MAX_HTTP_METADATA_BYTES as u64 + 2 * MAX_MANIFEST_BYTES as u64 + 4096;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoreBudget {
    pub max_checkpoints: usize,
    pub max_reserved_bytes: u64,
}

/// An externally hash-bound, startup-only policy. HTTP requests cannot change
/// the storage root, namespace, live artifacts or numerical policy.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HttpCheckpointProfile {
    pub schema: String,
    pub store_root: PathBuf,
    pub chat_template: PathBuf,
    pub budget: StoreBudget,
    pub bindings: ModelBindings,
    pub math: NativeMathPolicy,
}

pub struct PinnedHttpCheckpointProfile {
    pub profile: HttpCheckpointProfile,
    _directories: [DirectoryLease; 2],
    _files: [File; 2],
}

impl PinnedHttpCheckpointProfile {
    pub fn open(path: &Path, expected_sha256: &str) -> io::Result<Self> {
        require(
            path.is_absolute() && valid_digest(expected_sha256),
            "checkpoint profile path/hash",
        )?;
        let directory =
            DirectoryLease::acquire(path.parent().ok_or_else(|| invalid("profile parent"))?)?;
        let name = path
            .file_name()
            .and_then(|v| v.to_str())
            .ok_or_else(|| invalid("profile name"))?;
        require(
            !name.contains(['\0', ':']) && !name.ends_with([' ', '.']),
            "profile leaf alias",
        )?;
        let mut file = directory.read_file(name, 65536)?;
        let raw = bounded_bytes(&mut file, 65536)?;
        require(
            sha256(&raw) == expected_sha256,
            "external checkpoint profile hash",
        )?;
        let profile: HttpCheckpointProfile =
            serde_json::from_slice(&raw).map_err(io::Error::other)?;
        require(
            profile.schema == "cq-qwen35-http-checkpoint-profile-v1"
                && profile.store_root.is_absolute()
                && profile.chat_template.is_absolute(),
            "checkpoint startup profile",
        )?;
        profile.budget.validate()?;
        profile.bindings.validate()?;
        profile.math.validate()?;
        require(
            sha256(&serde_json::to_vec(&profile.math).map_err(io::Error::other)?)
                == profile.bindings.numerical_contract_sha256,
            "checkpoint numerical profile digest",
        )?;
        let template_directory = DirectoryLease::acquire(
            profile
                .chat_template
                .parent()
                .ok_or_else(|| invalid("template parent"))?,
        )?;
        let template_name = profile
            .chat_template
            .file_name()
            .and_then(|v| v.to_str())
            .ok_or_else(|| invalid("template name"))?;
        require(
            !template_name.contains(['\0', ':']) && !template_name.ends_with([' ', '.']),
            "template leaf alias",
        )?;
        let mut template = template_directory.read_file(template_name, 1024 * 1024)?;
        require(
            sha256(&bounded_bytes(&mut template, 1024 * 1024)?)
                == profile.bindings.chat_template_sha256,
            "checkpoint canonical template binding",
        )?;
        Ok(Self {
            profile,
            _directories: [directory, template_directory],
            _files: [file, template],
        })
    }
}

impl StoreBudget {
    fn validate(self) -> io::Result<()> {
        require(
            (1..=64).contains(&self.max_checkpoints)
                && self.max_reserved_bytes > OVERHEAD
                && self.max_reserved_bytes <= 64 * (MAX_PAYLOAD_BYTES + OVERHEAD),
            "finite HTTP checkpoint store budget",
        )
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Reservation {
    schema: String,
    checkpoint_id: String,
    native_payload_bytes: u64,
    reserved_bytes: u64,
}

impl Reservation {
    fn bytes(&self) -> io::Result<Vec<u8>> {
        require(
            self.schema == SCHEMA
                && valid_digest(&self.checkpoint_id)
                && (1..=MAX_PAYLOAD_BYTES).contains(&self.native_payload_bytes)
                && self.reserved_bytes == self.native_payload_bytes + OVERHEAD,
            "HTTP slot reservation contract",
        )?;
        serde_json::to_vec(self).map_err(io::Error::other)
    }
    fn read(dir: &DirectoryLease, id: &str) -> io::Result<Self> {
        let raw = bounded_bytes(&mut dir.read_file(RESERVATION, 4096)?, 4096)?;
        let r: Self = serde_json::from_slice(&raw).map_err(io::Error::other)?;
        require(
            r.checkpoint_id == id && r.bytes()? == raw,
            "HTTP reservation identity/canonical bytes",
        )?;
        Ok(r)
    }
}

struct StoreLease {
    directory: DirectoryLease,
    _lock: File,
}

pub struct HttpCheckpointStore {
    lease: Arc<StoreLease>,
    budget: StoreBudget,
}

impl HttpCheckpointStore {
    /// The caller selects an existing local store root, never an HTTP path.
    /// The persistent lock file is reopened without truncation after process exit.
    pub fn open(root: &Path, budget: StoreBudget) -> io::Result<Self> {
        budget.validate()?;
        let directory = DirectoryLease::acquire(root)?;
        let path = root.join(LOCK);
        if path.try_exists()? {
            plain(&fs::symlink_metadata(&path)?, false)?;
        }
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .share_mode(0)
            .custom_flags(OPEN_REPARSE)
            .open(path)?;
        plain(&lock.metadata()?, false)?;
        require(
            lock.metadata()?.len() == 0,
            "unexpected owner lock contents",
        )?;
        let store = Self {
            lease: Arc::new(StoreLease {
                directory,
                _lock: lock,
            }),
            budget,
        };
        store.usage()?;
        Ok(store)
    }

    fn usage(&self) -> io::Result<(usize, u64)> {
        let (mut count, mut bytes) = (0usize, 0u64);
        for entry in fs::read_dir(&self.lease.directory.root)? {
            let entry = entry?;
            let id = entry
                .file_name()
                .into_string()
                .map_err(|_| invalid("HTTP store name encoding"))?;
            if id == LOCK {
                plain(&fs::symlink_metadata(entry.path())?, false)?;
                continue;
            }
            require(valid_digest(&id), "HTTP store has unexpected entry")?;
            plain(&fs::symlink_metadata(entry.path())?, true)?;
            count += 1;
            require(
                count <= self.budget.max_checkpoints,
                "HTTP checkpoint count quota",
            )?;
            let dir = DirectoryLease::acquire(&entry.path())?;
            // An interrupted reservation is retained and blocks admission until
            // explicitly inspected. Never silently ignore or auto-delete it.
            let r = Reservation::read(&dir, &id)?;
            bytes = bytes
                .checked_add(r.reserved_bytes)
                .ok_or_else(|| invalid("store byte overflow"))?;
            require(
                bytes <= self.budget.max_reserved_bytes,
                "HTTP checkpoint byte quota",
            )?;
        }
        Ok((count, bytes))
    }

    /// Nonmutating quota check for optional automatic cache captures. Disk and
    /// publication errors remain failures; a full cache need not fail inference.
    pub fn has_capacity_for(&self,native_payload_bytes:u64)->io::Result<bool> {
        require((1..=MAX_PAYLOAD_BYTES).contains(&native_payload_bytes),"native payload quota extent")?;
        let reserved=native_payload_bytes.checked_add(OVERHEAD).ok_or_else(||invalid("reservation overflow"))?;
        let (count,used)=self.usage()?;
        Ok(count<self.budget.max_checkpoints && used.checked_add(reserved).is_some_and(|n|n<=self.budget.max_reserved_bytes))
    }

    pub fn reserve(
        &mut self,
        id: &str,
        native_payload_bytes: u64,
    ) -> io::Result<HttpCheckpointSlot> {
        self.reserve_with_available(id, native_payload_bytes, available_space::available_bytes)
    }

    fn reserve_with_available(
        &mut self,
        id: &str,
        native_payload_bytes: u64,
        available: impl FnOnce(&Path) -> io::Result<u64>,
    ) -> io::Result<HttpCheckpointSlot> {
        require(valid_digest(id), "opaque checkpoint ID, not a path")?;
        let reservation = Reservation {
            schema: SCHEMA.into(),
            checkpoint_id: id.into(),
            native_payload_bytes,
            reserved_bytes: native_payload_bytes
                .checked_add(OVERHEAD)
                .ok_or_else(|| invalid("reservation overflow"))?,
        };
        let raw = reservation.bytes()?;
        let (count, used) = self.usage()?;
        require(
            count < self.budget.max_checkpoints
                && used
                    .checked_add(reservation.reserved_bytes)
                    .is_some_and(|n| n <= self.budget.max_reserved_bytes),
            "HTTP checkpoint backpressure before capture",
        )?;
        // Reject before any slot, payload, or CUDA capture mutation. Another
        // process can still consume disk later; retain the existing writer's
        // manifest-last/failed-slot semantics for that race.
        available_space::require_capacity(
            reservation.reserved_bytes,
            available(&self.lease.directory.root)?,
        )?;
        let root = self.lease.directory.root.join(id);
        fs::create_dir(&root)?;
        let directory = Arc::new(DirectoryLease::acquire(&root)?);
        directory.create_file(RESERVATION, &raw)?;
        Ok(HttpCheckpointSlot {
            directory,
            _store: self.lease.clone(),
            reservation,
        })
    }

    /// Keep returned lease alive through native restore and final publication.
    pub fn read(
        &self,
        id: &str,
        expected_sha256: &str,
        bindings: &ModelBindings,
        math: &NativeMathPolicy,
    ) -> io::Result<HttpCheckpointRead> {
        require(valid_digest(id), "opaque checkpoint ID, not a path")?;
        self.usage()?;
        let directory = Arc::new(DirectoryLease::acquire(
            &self.lease.directory.root.join(id),
        )?);
        let reservation = Reservation::read(&directory, id)?;
        let mut seal = directory.read_file(OUTER_SEAL, MAX_HTTP_METADATA_BYTES as u64)?;
        let raw = bounded_bytes(&mut seal, MAX_HTTP_METADATA_BYTES)?;
        let envelope = HttpSessionCheckpoint::decode(&raw, expected_sha256)?;
        let mut blob = directory.read_file(OUTER_BLOB, MAX_HTTP_METADATA_BYTES as u64)?;
        require(
            bounded_bytes(&mut blob, MAX_HTTP_METADATA_BYTES)? == raw,
            "HTTP publication alias bytes",
        )?;
        require(
            envelope.checkpoint_id == id
                && &envelope.bindings == bindings
                && &envelope.math == math,
            "HTTP outer/live identity",
        )?;
        let native_path = directory.root.join("native");
        let native_directory = DirectoryLease::acquire(&native_path)?;
        let descriptor_bytes = LocalCheckpointReader::read_session_descriptor(
            &native_path,
            &envelope.native_manifest_sha256,
        )?;
        let descriptor = SessionDescriptor::decode(&descriptor_bytes, bindings, math)?;
        envelope.validate_native(&descriptor, &envelope.native_manifest_sha256)?;
        require(
            descriptor.transfer_plan()?.validate()? == reservation.native_payload_bytes,
            "native reservation extent",
        )?;
        outer_namespace(&directory, true)?;
        Ok(HttpCheckpointRead {
            directory,
            _store: self.lease.clone(),
            _native_directory: native_directory,
            _files: [seal, blob],
            envelope,
            descriptor,
        })
    }
}

pub struct HttpCheckpointSlot {
    directory: Arc<DirectoryLease>,
    _store: Arc<StoreLease>,
    reservation: Reservation,
}

impl HttpCheckpointSlot {
    pub fn native_path(&self) -> PathBuf {
        self.directory.root.join("native")
    }

    /// Called only after native capture returned its authenticated completion.
    /// Native payload will be checked again by the restore owner before publish.
    pub fn publish(self, envelope: &HttpSessionCheckpoint) -> io::Result<String> {
        require(
            envelope.checkpoint_id == self.reservation.checkpoint_id,
            "HTTP reserved checkpoint ID",
        )?;
        let path = self.native_path();
        let _native = DirectoryLease::acquire(&path)?;
        let raw = LocalCheckpointReader::read_session_descriptor(
            &path,
            &envelope.native_manifest_sha256,
        )?;
        let descriptor = SessionDescriptor::decode(&raw, &envelope.bindings, &envelope.math)?;
        envelope.validate_native(&descriptor, &envelope.native_manifest_sha256)?;
        require(
            descriptor.transfer_plan()?.validate()? == self.reservation.native_payload_bytes,
            "HTTP saved reservation extent",
        )?;
        verify_native_payload(
            &path,
            &envelope.native_manifest_sha256,
            &descriptor.transfer_plan()?,
            &envelope.target_seal,
            descriptor.ledger.consumed.len(),
        )?;
        outer_namespace(&self.directory, false)?;
        let raw = envelope.encode()?;
        let digest = sha256(&raw);
        self.directory.create_file(OUTER_BLOB, &raw)?;
        let mut blob = self
            .directory
            .read_file(OUTER_BLOB, MAX_HTTP_METADATA_BYTES as u64)?;
        require(
            bounded_bytes(&mut blob, MAX_HTTP_METADATA_BYTES)? == raw,
            "HTTP blob stable bytes",
        )?;
        fs::hard_link(
            self.directory.root.join(OUTER_BLOB),
            self.directory.root.join(OUTER_SEAL),
        )?;
        let mut seal = self
            .directory
            .read_file(OUTER_SEAL, MAX_HTTP_METADATA_BYTES as u64)?;
        require(
            bounded_bytes(&mut seal, MAX_HTTP_METADATA_BYTES)? == raw,
            "HTTP published bytes",
        )?;
        outer_namespace(&self.directory, true)?;
        Ok(digest)
    }
}

pub struct HttpCheckpointRead {
    directory: Arc<DirectoryLease>,
    _store: Arc<StoreLease>,
    _native_directory: DirectoryLease,
    _files: [File; 2],
    pub envelope: HttpSessionCheckpoint,
    pub descriptor: SessionDescriptor,
}

impl HttpCheckpointRead {
    pub fn native_path(&self) -> PathBuf {
        self.directory.root.join("native")
    }
    pub fn verify_namespace(&self) -> io::Result<()> {
        outer_namespace(&self.directory, true)
    }
}

fn outer_namespace(directory: &DirectoryLease, sealed: bool) -> io::Result<()> {
    let expected: BTreeSet<&str> = if sealed {
        [RESERVATION, "native", OUTER_BLOB, OUTER_SEAL]
            .into_iter()
            .collect()
    } else {
        [RESERVATION, "native"].into_iter().collect()
    };
    let mut actual = BTreeSet::new();
    for entry in fs::read_dir(&directory.root)? {
        let entry = entry?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| invalid("HTTP outer filename"))?;
        require(
            expected.contains(name.as_str()),
            "unexpected HTTP checkpoint member",
        )?;
        plain(&fs::symlink_metadata(entry.path())?, name == "native")?;
        actual.insert(name);
    }
    require(
        actual.iter().map(String::as_str).collect::<BTreeSet<_>>() == expected,
        "incomplete HTTP checkpoint",
    )
}

fn verify_native_payload(
    root: &Path,
    manifest_sha: &str,
    plan: &TransferPlan,
    seal: &crate::session_checkpoint::http::TargetStateSeal,
    cursor: usize,
) -> io::Result<()> {
    use sha2::{Digest, Sha256};
    let mut reader = LocalCheckpointReader::open(root, manifest_sha, plan)?;
    let manifest = Manifest::decode(reader.manifest_bytes(), manifest_sha, plan)?;
    let mut target = Sha256::new();
    target.update(b"QWEN35-SESSION-TARGET-STATE-V1\0");
    target.update((cursor as u64).to_le_bytes());
    let mut planes = vec![Sha256::new(); 128];
    let mut feature = Sha256::new();
    let mut staging = vec![0; manifest.chunk_bytes];
    for chunk in &manifest.chunks {
        let mut input = reader.open_chunk(&chunk.sha256)?;
        let bytes = &mut staging[..chunk.bytes];
        input.read_exact(bytes)?;
        let mut tail = [0];
        require(
            input.read(&mut tail)? == 0
                && chunk_digest(&plan.components[chunk.component], chunk.offset, bytes)
                    == chunk.sha256,
            "HTTP native payload content/extent before publication",
        )?;
        if (1..129).contains(&chunk.component) {
            let i = chunk.component - 1;
            if chunk.offset == 0 {
                target.update((i as u64 / 2).to_le_bytes());
                target.update((i as u64 % 2).to_le_bytes());
                target.update(plan.components[chunk.component].bytes.to_le_bytes());
            }
            target.update(&*bytes);
            planes[i].update(&*bytes);
        } else if chunk.component == 129 {
            feature.update(&*bytes);
        }
    }
    reader.finish_read()?;
    require(
        format!("{:x}", target.finalize()) == seal.target_committed_state_sha256
            && format!("{:x}", feature.finalize()) == seal.feature_bytes_sha256
            && planes
                .into_iter()
                .zip(&seal.state_plane_hashes)
                .all(|(hash, plane)| format!("{:x}", hash.finalize()) == plane.sha256),
        "HTTP seal differs from native payload bytes",
    )
}

#[cfg(test)]
#[path = "http_filesystem_tests.rs"]
mod tests;

#[path = "augur_store.rs"]
pub mod augur;
