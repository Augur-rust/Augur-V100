//! Local Windows immutable checkpoint files, with manifest-last publication.
//!
//! Directory handles deny deletion/rename for every ancestor. File handles deny
//! concurrent writers/deletion; OPEN_REPARSE_POINT plus handle metadata rejects
//! reparse redirection. Reads are bounded and content/namespace checked again at
//! completion. This is a process-consistency backend, NOT a power-loss claim.
//! No recursive cleanup, overwrite, eviction, global dedup or CUDA is performed.

use super::*;
use std::fs::{self, File, Metadata, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::os::windows::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Component as PathComponent, Path, PathBuf, Prefix};
use std::sync::Arc;

#[path = "http_filesystem.rs"]
pub mod http;

const SHARE_READ: u32 = 1;
const SHARE_WRITE: u32 = 2;
const OPEN_REPARSE: u32 = 0x00200000;
const BACKUP_SEMANTICS: u32 = 0x02000000;
const REPARSE_ATTRIBUTE: u32 = 0x400;
const MANIFEST_BLOB: &str = "manifest.blob";
const MANIFEST_SEAL: &str = "manifest.json";

fn plain(metadata: &Metadata, directory: bool) -> io::Result<()> {
    require(
        !metadata.file_type().is_symlink()
            && metadata.file_attributes() & REPARSE_ATTRIBUTE == 0
            && if directory {
                metadata.is_dir()
            } else {
                metadata.is_file()
            },
        "checkpoint path must be a non-reparse regular file/directory",
    )
}

struct DirectoryLease {
    root: PathBuf,
    // These handles are the lifetime guard, not redundant cached metadata.
    _ancestors: Vec<File>,
}

impl DirectoryLease {
    fn acquire(root: &Path) -> io::Result<Self> {
        require(root.is_absolute(), "checkpoint root must be absolute")?;
        let mut path = PathBuf::new();
        let mut handles = Vec::new();
        for component in root.components() {
            match component {
                PathComponent::Prefix(prefix) => {
                    require(
                        matches!(prefix.kind(), Prefix::Disk(_) | Prefix::VerbatimDisk(_)),
                        "checkpoint requires a local disk path",
                    )?;
                    path.push(prefix.as_os_str());
                    continue;
                }
                PathComponent::RootDir => path.push(component.as_os_str()),
                PathComponent::Normal(name) => {
                    let text = name
                        .to_str()
                        .ok_or_else(|| invalid("invalid path encoding"))?;
                    require(
                        !text.contains(['\0', ':']) && !text.ends_with([' ', '.']),
                        "checkpoint path alias/stream",
                    )?;
                    path.push(name);
                }
                _ => return Err(invalid("checkpoint path traversal")),
            }
            plain(&fs::symlink_metadata(&path)?, true)?;
            let handle = OpenOptions::new()
                .read(true)
                .share_mode(SHARE_READ | SHARE_WRITE)
                .custom_flags(OPEN_REPARSE | BACKUP_SEMANTICS)
                .open(&path)?;
            plain(&handle.metadata()?, true)?;
            // Parent handles already held prevent component rename during descent.
            plain(&fs::symlink_metadata(&path)?, true)?;
            handles.push(handle);
        }
        require(!handles.is_empty(), "checkpoint has no directory")?;
        Ok(Self {
            root: root.into(),
            _ancestors: handles,
        })
    }

    fn read_file(&self, name: &str, bound: u64) -> io::Result<File> {
        let path = self.root.join(name);
        plain(&fs::symlink_metadata(&path)?, false)?;
        let file = OpenOptions::new()
            .read(true)
            .share_mode(SHARE_READ)
            .custom_flags(OPEN_REPARSE)
            .open(&path)?;
        let metadata = file.metadata()?;
        plain(&metadata, false)?;
        require(
            metadata.len() > 0 && metadata.len() <= bound,
            "bounded checkpoint file length",
        )?;
        Ok(file)
    }

    fn create_file(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .share_mode(SHARE_READ)
            .custom_flags(OPEN_REPARSE)
            .open(self.root.join(name))?;
        plain(&file.metadata()?, false)?;
        file.write_all(bytes)?;
        file.sync_all()
    }

    fn namespace(&self, expected: &BTreeSet<String>) -> io::Result<()> {
        let mut actual = BTreeSet::new();
        for entry in fs::read_dir(&self.root)? {
            let entry = entry?;
            require(actual.len() < MAX_CHUNKS + 2, "checkpoint namespace quota")?;
            let name = entry
                .file_name()
                .into_string()
                .map_err(|_| invalid("namespace encoding"))?;
            require(expected.contains(&name), "unexpected checkpoint entry")?;
            plain(&fs::symlink_metadata(entry.path())?, false)?;
            require(actual.insert(name), "duplicate checkpoint entry")?;
        }
        require(&actual == expected, "incomplete checkpoint namespace")
    }
}

fn chunk_name(digest: &str) -> io::Result<String> {
    require(valid_digest(digest), "invalid chunk filename digest")?;
    Ok(format!("{digest}.chunk"))
}

fn namespace(manifest: &Manifest, sealed: bool) -> io::Result<BTreeSet<String>> {
    let mut names = BTreeSet::new();
    for chunk in &manifest.chunks {
        require(
            names.insert(chunk_name(&chunk.sha256)?),
            "duplicate role/offset content address",
        )?;
    }
    if sealed {
        names.insert(MANIFEST_BLOB.into());
        names.insert(MANIFEST_SEAL.into());
    }
    Ok(names)
}

fn bounded_bytes(file: &mut File, limit: usize) -> io::Result<Vec<u8>> {
    let len = file.metadata()?.len();
    require(len > 0 && len <= limit as u64, "metadata length quota")?;
    let mut bytes = vec![0; len as usize];
    file.read_exact(&mut bytes)?;
    let mut tail = [0];
    require(file.read(&mut tail)? == 0, "checkpoint file grew")?;
    Ok(bytes)
}

pub struct LocalCheckpointWriter {
    directory: Arc<DirectoryLease>,
    expected: TransferPlan,
    written: BTreeSet<String>,
    payload_bytes: u64,
    aborted: bool,
    sealed: bool,
}

impl LocalCheckpointWriter {
    /// Creates exactly one fresh leaf directory; never recursively creates an
    /// unchecked ancestor and never reuses or overwrites an old checkpoint root.
    pub fn create(root: &Path, expected: &TransferPlan) -> io::Result<Self> {
        expected.validate()?;
        let parent = root
            .parent()
            .ok_or_else(|| invalid("checkpoint parent missing"))?;
        let _parents = DirectoryLease::acquire(parent)?;
        // Validate the not-yet-existing leaf without resolving it through aliases.
        let leaf = root
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or_else(|| invalid("checkpoint leaf"))?;
        require(
            !leaf.contains(['\0', ':']) && !leaf.ends_with([' ', '.']),
            "checkpoint leaf alias",
        )?;
        fs::create_dir(root)?;
        let directory = Arc::new(DirectoryLease::acquire(root)?);
        directory.namespace(&BTreeSet::new())?;
        Ok(Self {
            directory,
            expected: expected.clone(),
            written: BTreeSet::new(),
            payload_bytes: 0,
            aborted: false,
            sealed: false,
        })
    }
}

impl CheckpointWriter for LocalCheckpointWriter {
    fn write_chunk(&mut self, digest: &str, bytes: &[u8]) -> io::Result<()> {
        require(!self.aborted && !self.sealed, "writer is no longer pending")?;
        require(
            !bytes.is_empty() && bytes.len() <= MAX_CHUNK_BYTES,
            "chunk write bound",
        )?;
        let name = chunk_name(digest)?;
        require(
            self.written.len() < MAX_CHUNKS && !self.written.contains(&name),
            "duplicate chunk write",
        )?;
        let total = self
            .payload_bytes
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| invalid("payload overflow"))?;
        require(
            total <= self.expected.validate()?,
            "checkpoint payload quota",
        )?;
        self.directory.create_file(&name, bytes)?;
        self.written.insert(name);
        self.payload_bytes = total;
        Ok(())
    }

    fn seal_manifest(&mut self, bytes: &[u8], digest: &str) -> io::Result<()> {
        require(!self.aborted && !self.sealed, "writer is no longer pending")?;
        let manifest = Manifest::decode(bytes, digest, &self.expected)?;
        let expected_namespace = namespace(&manifest, false)?;
        require(
            self.written == expected_namespace && self.payload_bytes == self.expected.validate()?,
            "written payload set or bytes",
        )?;
        self.directory.namespace(&expected_namespace)?;
        // Keep verified read handles until publication so files cannot be replaced
        // or opened for mutation between verification and the seal hard link.
        let mut pinned = Vec::with_capacity(manifest.chunks.len());
        let mut staging = vec![0; manifest.chunk_bytes];
        for chunk in &manifest.chunks {
            let mut file = self
                .directory
                .read_file(&chunk_name(&chunk.sha256)?, chunk.bytes as u64)?;
            require(
                file.metadata()?.len() == chunk.bytes as u64,
                "seal exact chunk length",
            )?;
            file.read_exact(&mut staging[..chunk.bytes])?;
            require(
                chunk_digest(
                    &self.expected.components[chunk.component],
                    chunk.offset,
                    &staging[..chunk.bytes],
                ) == chunk.sha256,
                "stored chunk drift before seal",
            )?;
            pinned.push(file);
        }
        self.directory.create_file(MANIFEST_BLOB, bytes)?;
        let mut blob = self
            .directory
            .read_file(MANIFEST_BLOB, MAX_MANIFEST_BYTES as u64)?;
        require(
            bounded_bytes(&mut blob, MAX_MANIFEST_BYTES)? == bytes,
            "manifest blob drift",
        )?;
        let mut prepublish = expected_namespace;
        prepublish.insert(MANIFEST_BLOB.into());
        self.directory.namespace(&prepublish)?;
        // No-clobber atomic name publication on this local filesystem. The blob is
        // retained deliberately; both names are in the exact sealed namespace.
        fs::hard_link(
            self.directory.root.join(MANIFEST_BLOB),
            self.directory.root.join(MANIFEST_SEAL),
        )?;
        self.directory.namespace(&namespace(&manifest, true)?)?;
        self.sealed = true;
        Ok(())
    }

    fn abort(&mut self) -> io::Result<()> {
        self.aborted = true;
        // Preserve incomplete files for diagnosis; no manifest is created here.
        // A post-publication error is never reported as rollback of a valid seal.
        if self.directory.root.join(MANIFEST_SEAL).try_exists()? {
            return Err(invalid(
                "seal exists after failed operation; retain and independently inspect",
            ));
        }
        Ok(())
    }
}

pub struct LocalCheckpointReader {
    directory: Arc<DirectoryLease>,
    manifest: Manifest,
    manifest_bytes: Vec<u8>,
    expected_namespace: BTreeSet<String>,
    // Both manifest names remain pinned against writes or replacement.
    _manifest_files: [File; 2],
}

impl LocalCheckpointReader {
    /// Read only the bounded, hash-verified descriptor for bootstrap. These bytes
    /// still need SessionDescriptor::decode against live bindings/math and a new
    /// transfer_plan(); open/restore must then use that independently derived
    /// plan. This method never creates or mutates a CUDA/session destination.
    pub fn read_session_descriptor(
        root: &Path,
        expected_manifest_sha256: &str,
    ) -> io::Result<Vec<u8>> {
        Self::read_descriptor(root, expected_manifest_sha256, "session.metadata")
    }

    /// Independent input-only bootstrap. A generated-session reader must not
    /// consume this role, and an input-prefix reader must not revive a session.
    /// Payload authentication and typed/live binding admission remain required.
    pub fn read_input_prefix_descriptor(
        root: &Path,
        expected_manifest_sha256: &str,
    ) -> io::Result<Vec<u8>> {
        Self::read_descriptor(root, expected_manifest_sha256, "input_prefix.metadata")
    }

    fn read_descriptor(
        root: &Path,
        expected_manifest_sha256: &str,
        role: &str,
    ) -> io::Result<Vec<u8>> {
        let directory = DirectoryLease::acquire(root)?;
        let mut seal = directory.read_file(MANIFEST_SEAL, MAX_MANIFEST_BYTES as u64)?;
        let manifest_bytes = bounded_bytes(&mut seal, MAX_MANIFEST_BYTES)?;
        let manifest = Manifest::inspect_content_bound(&manifest_bytes, expected_manifest_sha256)?;
        let mut blob = directory.read_file(MANIFEST_BLOB, MAX_MANIFEST_BYTES as u64)?;
        require(
            bounded_bytes(&mut blob, MAX_MANIFEST_BYTES)? == manifest_bytes,
            "manifest alias drift",
        )?;
        let expected_namespace = namespace(&manifest, true)?;
        directory.namespace(&expected_namespace)?;
        let first = &manifest.plan.components[0];
        require(
            first.name == role && first.bytes <= super::metadata::MAX_DESCRIPTOR_BYTES as u64,
            "bootstrap descriptor role/extent",
        )?;
        let mut bytes = vec![0; first.bytes as usize];
        for chunk in manifest.chunks.iter().take_while(|c| c.component == 0) {
            let start = chunk.offset as usize;
            let output = &mut bytes[start..start + chunk.bytes];
            let mut file = directory.read_file(&chunk_name(&chunk.sha256)?, chunk.bytes as u64)?;
            require(
                file.metadata()?.len() == chunk.bytes as u64,
                "descriptor chunk extent",
            )?;
            file.read_exact(output)?;
            require(
                chunk_digest(first, chunk.offset, output) == chunk.sha256,
                "descriptor chunk content drift",
            )?;
        }
        directory.namespace(&expected_namespace)?;
        Ok(bytes)
    }

    pub fn open(
        root: &Path,
        expected_manifest_sha256: &str,
        expected: &TransferPlan,
    ) -> io::Result<Self> {
        expected.validate()?;
        let directory = Arc::new(DirectoryLease::acquire(root)?);
        let mut seal = directory.read_file(MANIFEST_SEAL, MAX_MANIFEST_BYTES as u64)?;
        let bytes = bounded_bytes(&mut seal, MAX_MANIFEST_BYTES)?;
        let manifest = Manifest::decode(&bytes, expected_manifest_sha256, expected)?;
        let mut blob = directory.read_file(MANIFEST_BLOB, MAX_MANIFEST_BYTES as u64)?;
        require(
            bounded_bytes(&mut blob, MAX_MANIFEST_BYTES)? == bytes,
            "manifest alias bytes drift",
        )?;
        let expected_namespace = namespace(&manifest, true)?;
        directory.namespace(&expected_namespace)?;
        Ok(Self {
            directory,
            manifest,
            manifest_bytes: bytes,
            expected_namespace,
            _manifest_files: [seal, blob],
        })
    }

    pub fn manifest_bytes(&self) -> &[u8] {
        &self.manifest_bytes
    }
}

pub struct LockedChunkReader {
    file: File,
    _directory: Arc<DirectoryLease>,
}

impl Read for LockedChunkReader {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        self.file.read(bytes)
    }
}

impl CheckpointReader for LocalCheckpointReader {
    type ChunkReader = LockedChunkReader;

    fn open_chunk(&mut self, digest: &str) -> io::Result<Self::ChunkReader> {
        let chunk = self
            .manifest
            .chunks
            .iter()
            .find(|c| c.sha256 == digest)
            .ok_or_else(|| invalid("chunk outside sealed manifest"))?;
        let file = self
            .directory
            .read_file(&chunk_name(digest)?, chunk.bytes as u64)?;
        require(
            file.metadata()?.len() == chunk.bytes as u64,
            "stored chunk exact extent",
        )?;
        Ok(LockedChunkReader {
            file,
            _directory: self.directory.clone(),
        })
    }

    fn finish_read(&mut self) -> io::Result<()> {
        self.directory.namespace(&self.expected_namespace)?;
        // Re-read the pinned manifests, not newly resolved path objects.
        for file in &mut self._manifest_files {
            file.seek(SeekFrom::Start(0))?;
            require(
                bounded_bytes(file, MAX_MANIFEST_BYTES)? == self.manifest_bytes,
                "manifest drift during restore",
            )?;
        }
        Ok(())
    }
}
