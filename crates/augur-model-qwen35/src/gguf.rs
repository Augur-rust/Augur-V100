use crate::error::Qwen35Error;
use crate::identity::sha256_file;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, Metadata, OpenOptions};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Component, Path, PathBuf};
use std::time::SystemTime;

pub const GGUF_MAGIC: u32 = 0x4655_4747;
pub const GGUF_DEFAULT_ALIGNMENT: u64 = 32;
pub const GGUF_VERSION_2: u32 = 2;
pub const GGUF_VERSION_3: u32 = 3;

const MAX_TENSORS: u64 = 2_000_000;
const MAX_METADATA_KEYS: u64 = 2_000_000;
const MAX_STRING_BYTES: u64 = 1 << 20;
const MAX_ARRAY_ELEMENTS: u64 = 1 << 20;

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", content = "value")]
pub enum GgufValue {
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U64(value) => Some(*value),
            Self::I64(value) if *value >= 0 => Some(*value as u64),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::F64(value) => Some(*value),
            Self::U64(value) => Some(*value as f64),
            Self::I64(value) => Some(*value as f64),
            _ => None,
        }
    }

    pub fn array_len(&self) -> Option<usize> {
        match self {
            Self::Array(values) => Some(values.len()),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GgufTensor {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub ggml_type: u32,
    pub relative_offset: u64,
    pub absolute_offset: u64,
    pub byte_len: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GgufAuditOptions {
    pub max_header_bytes: u64,
    pub hash_file: bool,
}

impl Default for GgufAuditOptions {
    fn default() -> Self {
        Self {
            max_header_bytes: 128 * 1024 * 1024,
            hash_file: true,
        }
    }
}

/// Stable identity captured from the descriptor used to inspect a GGUF file.
#[derive(Clone, Debug, PartialEq, Eq)]
struct GgufFileIdentity {
    file_size: u64,
    modified: Option<SystemTime>,
    created: Option<SystemTime>,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(windows)]
    file_attributes: u32,
    #[cfg(windows)]
    creation_time: u64,
    #[cfg(windows)]
    last_write_time: u64,
}

impl GgufFileIdentity {
    fn from_metadata(metadata: &Metadata) -> Self {
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt;
        #[cfg(windows)]
        use std::os::windows::fs::MetadataExt;

        Self {
            file_size: metadata.len(),
            modified: metadata.modified().ok(),
            created: metadata.created().ok(),
            #[cfg(unix)]
            device: metadata.dev(),
            #[cfg(unix)]
            inode: metadata.ino(),
            #[cfg(windows)]
            file_attributes: metadata.file_attributes(),
            #[cfg(windows)]
            creation_time: metadata.creation_time(),
            #[cfg(windows)]
            last_write_time: metadata.last_write_time(),
        }
    }

    fn matches(&self, metadata: &Metadata) -> bool {
        self == &Self::from_metadata(metadata)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GgufManifest {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_count: u64,
    pub alignment: u64,
    pub header_bytes: u64,
    pub data_offset: u64,
    pub file_size: u64,
    pub file_sha256: Option<String>,
    pub tensors: Vec<GgufTensor>,
    metadata: BTreeMap<String, GgufValue>,
    file_identity: GgufFileIdentity,
    inspected_file_sha256: Option<String>,
}

/// A descriptor-bound reader for raw GGUF tensor payloads.
pub struct GgufTensorReader<'a> {
    manifest: &'a GgufManifest,
    file: File,
    path: PathBuf,
    identity: GgufFileIdentity,
}

/// One authenticated Windows descriptor retained through payload loading.
/// The descriptor denies new writers/deletion; snapshots cannot alter its manifest.
pub struct GgufPinnedArtifact {
    manifest: GgufManifest,
    file: File,
    path: PathBuf,
}

/// Concurrent walls are nested, not additive. No authenticated artifact or
/// loaded value is returned until the complete hash and identity checks finish.
#[derive(Clone, Debug, Serialize)]
pub struct GgufLoadOverlapTiming {
    pub full_file_hash_ms: f64,
    pub payload_load_ms: f64,
    pub authentication_join_wait_ms: f64,
    pub enclosing_ms: f64,
}

#[cfg(windows)]
fn independent_read_descriptor(file: &File) -> Result<File, Qwen35Error> {
    use std::os::windows::io::{AsRawHandle, FromRawHandle, RawHandle};
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn ReOpenFile(file: RawHandle, access: u32, sharing: u32, flags: u32) -> RawHandle;
    }
    // ReOpenFile refers to the already locked file object, not a re-resolved
    // pathname. Unlike DuplicateHandle/try_clone, its seek cursor is independent.
    // SAFETY: file is live; successful ReOpenFile transfers one owned handle.
    let handle = unsafe { ReOpenFile(file.as_raw_handle(), 0x80000000, 1, 0x08000000) };
    if handle == (-1isize) as RawHandle {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(unsafe { File::from_raw_handle(handle) })
}

impl GgufPinnedArtifact {
    /// Load immutable payloads while a second, independently positioned locked
    /// descriptor authenticates ALL file bytes. The callback must only prepare
    /// state: do not execute inference or publish readiness inside it. The
    /// callback sees an explicitly unhashed manifest; only the returned artifact
    /// is authenticated. Failure drops its result and joins the hash worker.
    pub fn load_overlapped<T, E: From<Qwen35Error>>(
        path: impl AsRef<Path>, options: GgufAuditOptions,
        load: impl FnOnce(&GgufManifest, &mut GgufTensorReader<'_>) -> Result<T, E>,
    ) -> Result<(T, Self, GgufLoadOverlapTiming), E> {
        if !options.hash_file {
            return Err(Qwen35Error::Gguf("overlapped loading requires a full content hash".into()).into());
        }
        let path = path.as_ref();
        validate_regular_path(path).map_err(E::from)?;
        #[cfg(not(windows))]
        { let _ = load; Err(Qwen35Error::Gguf("overlapped loading requires Windows sharing locks".into()).into()) }
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            let started = std::time::Instant::now();
            let file = OpenOptions::new().read(true).share_mode(1).open(path)
                .map_err(Qwen35Error::from).map_err(E::from)?;
            let (mut manifest, file) = GgufManifest::inspect_open_file(path,
                GgufAuditOptions { hash_file: false, ..options }, file).map_err(E::from)?;
            let mut hash_file = independent_read_descriptor(&file).map_err(E::from)?;
            let identity = manifest.file_identity.clone();
            let (value, digest, hash_ms, load_ms, join_ms) = std::thread::scope(|scope| {
                let worker = scope.spawn(move || -> Result<_, Qwen35Error> {
                    let hash_started = std::time::Instant::now();
                    let before = hash_file.metadata()?;
                    if !identity.matches(&before) { return Err(Qwen35Error::Gguf("hash descriptor identity drift".into())); }
                    hash_file.seek(SeekFrom::Start(0))?;
                    let digest = sha256_file(&mut hash_file)?;
                    let after = hash_file.metadata()?;
                    if !identity.matches(&after) || hash_file.stream_position()? != identity.file_size {
                        return Err(Qwen35Error::Gguf("full hash extent or identity drift".into()));
                    }
                    Ok((digest, hash_started.elapsed().as_secs_f64() * 1000.0))
                });
                let load_started = std::time::Instant::now();
                let mut reader = GgufTensorReader { manifest: &manifest,
                    file: file.try_clone().map_err(Qwen35Error::from).map_err(E::from)?,
                    path: path.to_path_buf(), identity: manifest.file_identity.clone() };
                let loaded = load(&manifest, &mut reader);
                drop(reader);
                let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
                let join_started = std::time::Instant::now();
                let hash_result = worker.join().map_err(|_| E::from(Qwen35Error::Gguf("full hash worker panicked".into())))?;
                let (digest, hash_ms) = hash_result.map_err(E::from)?;
                Ok::<_, E>((loaded?, digest, hash_ms, load_ms, join_started.elapsed().as_secs_f64()*1000.0))
            })?;
            let after = file.metadata().map_err(Qwen35Error::from).map_err(E::from)?;
            if !manifest.file_identity.matches(&after) {
                return Err(Qwen35Error::Gguf("loaded artifact identity drift".into()).into());
            }
            ensure_path_matches_identity(path, &manifest.file_identity).map_err(E::from)?;
            manifest.file_sha256 = Some(digest.clone());
            manifest.inspected_file_sha256 = Some(digest);
            let timing = GgufLoadOverlapTiming { full_file_hash_ms: hash_ms, payload_load_ms: load_ms,
                authentication_join_wait_ms: join_ms, enclosing_ms: started.elapsed().as_secs_f64()*1000.0 };
            Ok((value, Self { manifest, file, path: path.to_path_buf() }, timing))
        }
    }

    pub fn open(path: impl AsRef<Path>, options: GgufAuditOptions) -> Result<Self, Qwen35Error> {
        if !options.hash_file {
            return Err(Qwen35Error::Gguf("pinned loading requires a full content hash".into()));
        }
        let path = path.as_ref();
        validate_regular_path(path)?;
        #[cfg(not(windows))]
        { Err(Qwen35Error::Gguf("pinned single-pass loading requires Windows sharing locks".into())) }
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            const FILE_SHARE_READ: u32 = 1;
            let file = OpenOptions::new().read(true).share_mode(FILE_SHARE_READ).open(path)?;
            let (manifest, file) = GgufManifest::inspect_open_file(path, options, file)?;
            Ok(Self { manifest, file, path: path.to_path_buf() })
        }
    }

    pub fn manifest(&self) -> &GgufManifest {
        &self.manifest
    }

    pub fn tensor_reader(&mut self) -> Result<GgufTensorReader<'_>, Qwen35Error> {
        let before = self.file.metadata()?;
        if !self.manifest.file_identity.matches(&before) || before.len() != self.manifest.file_size {
            return Err(Qwen35Error::Gguf("pinned artifact identity drift".into()));
        }
        ensure_path_matches_identity(&self.path, &self.manifest.file_identity)?;
        validate_manifest_ranges(&self.manifest, before.len())?;
        Ok(GgufTensorReader {
            manifest: &self.manifest,
            file: self.file.try_clone()?,
            path: self.path.clone(),
            identity: self.manifest.file_identity.clone(),
        })
    }
}

impl GgufManifest {
    pub fn inspect(path: impl AsRef<Path>, options: GgufAuditOptions) -> Result<Self, Qwen35Error> {
        let path = path.as_ref();
        validate_regular_path(path)?;
        let file = OpenOptions::new().read(true).open(path)?;
        Self::inspect_open_file(path, options, file).map(|(manifest, _)| manifest)
    }

    fn inspect_open_file(path: &Path, options: GgufAuditOptions, file: File) -> Result<(Self, File), Qwen35Error> {
        let before = file.metadata()?;
        if !before.is_file() {
            return Err(Qwen35Error::Gguf("artifact is not a regular file".into()));
        }
        let file_identity = GgufFileIdentity::from_metadata(&before);
        ensure_path_matches_identity(path, &file_identity)?;
        let file_size = before.len();
        if file_size < 20 {
            return Err(Qwen35Error::Gguf(
                "file is shorter than the GGUF header".into(),
            ));
        }
        if options.max_header_bytes < 20 {
            return Err(Qwen35Error::Gguf(
                "max_header_bytes must leave room for the fixed header".into(),
            ));
        }
        let mut reader = Reader {
            file,
            position: 0,
            file_size,
            max_position: options.max_header_bytes.min(file_size),
        };
        let magic = reader.read_u32()?;
        if magic != GGUF_MAGIC {
            return Err(Qwen35Error::Gguf(format!(
                "bad magic 0x{magic:08x}, expected 0x{GGUF_MAGIC:08x}"
            )));
        }
        let version = reader.read_u32()?;
        if version != GGUF_VERSION_2 && version != GGUF_VERSION_3 {
            return Err(Qwen35Error::Gguf(format!(
                "unsupported GGUF version {version}"
            )));
        }
        let tensor_count = reader.read_u64()?;
        let metadata_count = reader.read_u64()?;
        if tensor_count > MAX_TENSORS {
            return Err(Qwen35Error::Gguf(
                "tensor count exceeds bounded limit".into(),
            ));
        }
        if metadata_count > MAX_METADATA_KEYS {
            return Err(Qwen35Error::Gguf(
                "metadata count exceeds bounded limit".into(),
            ));
        }

        let mut metadata = BTreeMap::new();
        for _ in 0..metadata_count {
            let key = reader.read_string()?;
            if metadata.contains_key(&key) {
                return Err(Qwen35Error::Gguf(format!("duplicate metadata key {key:?}")));
            }
            let value = reader.read_value(0)?;
            metadata.insert(key, value);
        }

        let mut tensors = Vec::with_capacity(tensor_count as usize);
        let mut names = BTreeSet::new();
        for _ in 0..tensor_count {
            let name = reader.read_string()?;
            if !names.insert(name.clone()) {
                return Err(Qwen35Error::Gguf(format!("duplicate tensor name {name:?}")));
            }
            let dimensions_count = reader.read_u32()?;
            if dimensions_count == 0 || dimensions_count > 4 {
                return Err(Qwen35Error::Gguf(format!(
                    "tensor {name:?} has invalid dimension count {dimensions_count}"
                )));
            }
            let mut dimensions = Vec::with_capacity(dimensions_count as usize);
            for _ in 0..dimensions_count {
                let dimension = reader.read_u64()?;
                if dimension == 0 {
                    return Err(Qwen35Error::Gguf(format!(
                        "tensor {name:?} has a zero dimension"
                    )));
                }
                dimensions.push(dimension);
            }
            let ggml_type = reader.read_u32()?;
            let relative_offset = reader.read_u64()?;
            let byte_len = tensor_byte_len(&dimensions, ggml_type).ok_or_else(|| {
                Qwen35Error::Gguf(format!(
                    "tensor {name:?} uses unsupported or invalid GGML type {ggml_type}"
                ))
            })?;
            tensors.push(GgufTensor {
                name,
                dimensions,
                ggml_type,
                relative_offset,
                absolute_offset: 0,
                byte_len,
            });
        }

        let header_bytes = reader.position;
        let alignment = metadata
            .get("general.alignment")
            .and_then(GgufValue::as_u64)
            .unwrap_or(GGUF_DEFAULT_ALIGNMENT);
        if alignment == 0 || alignment > (1 << 20) || alignment & (alignment - 1) != 0 {
            return Err(Qwen35Error::Gguf(format!(
                "invalid alignment {alignment}, expected a bounded power of two"
            )));
        }
        let data_offset = align_up(header_bytes, alignment)?;
        if data_offset > file_size {
            return Err(Qwen35Error::Gguf(
                "aligned tensor data offset exceeds file size".into(),
            ));
        }
        let payload_size = file_size - data_offset;
        let mut ranges = Vec::with_capacity(tensors.len());
        for tensor in &mut tensors {
            if tensor.relative_offset % alignment != 0 {
                return Err(Qwen35Error::Gguf(format!(
                    "tensor {:?} offset {} is not aligned to {}",
                    tensor.name, tensor.relative_offset, alignment
                )));
            }
            let end = tensor
                .relative_offset
                .checked_add(tensor.byte_len)
                .ok_or(Qwen35Error::SizeOverflow)?;
            if end > payload_size {
                return Err(Qwen35Error::Gguf(format!(
                    "tensor {:?} range [{}, {}) exceeds payload {}",
                    tensor.name, tensor.relative_offset, end, payload_size
                )));
            }
            tensor.absolute_offset = data_offset
                .checked_add(tensor.relative_offset)
                .ok_or(Qwen35Error::SizeOverflow)?;
            ranges.push((tensor.absolute_offset, end, tensor.name.clone()));
        }
        ranges.sort_unstable_by_key(|range| range.0);
        for pair in ranges.windows(2) {
            if pair[0].1 > pair[1].0 {
                return Err(Qwen35Error::Gguf(format!(
                    "tensor ranges overlap: {:?} and {:?}",
                    pair[0].2, pair[1].2
                )));
            }
        }

        let file_sha256 = if options.hash_file {
            let mut hashing_file = reader.file.try_clone()?;
            hashing_file.seek(SeekFrom::Start(0))?;
            Some(sha256_file(&mut hashing_file)?)
        } else {
            None
        };
        let after = reader.file.metadata()?;
        ensure_stable_metadata(&before, &after)?;
        ensure_path_matches_identity(path, &file_identity)?;
        let header_after = reader.position;
        if header_after != header_bytes {
            return Err(Qwen35Error::Gguf(
                "reader position changed unexpectedly".into(),
            ));
        }
        Ok((Self {
            version,
            tensor_count,
            metadata_count,
            alignment,
            header_bytes,
            data_offset,
            file_size,
            file_sha256: file_sha256.clone(),
            tensors,
            metadata,
            file_identity,
            inspected_file_sha256: file_sha256.clone(),
        }, reader.file))
    }

    /// Return a tensor descriptor by its exact GGUF name.
    pub fn tensor(&self, name: &str) -> Option<&GgufTensor> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }

    /// Open a descriptor-bound payload reader for this inspected manifest.
    pub fn open_tensor_reader(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<GgufTensorReader<'_>, Qwen35Error> {
        let path = path.as_ref();
        validate_regular_path(path)?;
        let file = OpenOptions::new().read(true).open(path)?;
        let before = file.metadata()?;
        if !before.is_file() {
            return Err(Qwen35Error::Gguf("artifact is not a regular file".into()));
        }
        let identity = GgufFileIdentity::from_metadata(&before);
        if identity != self.file_identity {
            return Err(Qwen35Error::Gguf(
                "opened artifact identity differs from inspected manifest".into(),
            ));
        }
        ensure_path_matches_identity(path, &identity)?;
        if before.len() != self.file_size {
            return Err(Qwen35Error::Gguf(
                "opened artifact size differs from inspected manifest".into(),
            ));
        }
        validate_manifest_ranges(self, before.len())?;
        if self.file_sha256 != self.inspected_file_sha256 {
            return Err(Qwen35Error::Gguf(
                "manifest content digest was modified after inspection".into(),
            ));
        }
        if let Some(expected) = &self.inspected_file_sha256 {
            let mut hashing_file = file.try_clone()?;
            hashing_file.seek(SeekFrom::Start(0))?;
            let actual = sha256_file(&mut hashing_file)?;
            if &actual != expected {
                return Err(Qwen35Error::Gguf(
                    "opened artifact content differs from inspected manifest".into(),
                ));
            }
        }
        let after = file.metadata()?;
        ensure_stable_metadata(&before, &after)?;
        ensure_path_matches_identity(path, &identity)?;
        Ok(GgufTensorReader {
            manifest: self,
            file,
            path: path.to_path_buf(),
            identity,
        })
    }

    /// Read one tensor's exact raw payload bytes after rebinding the file.
    pub fn read_tensor_bytes(
        &self,
        path: impl AsRef<Path>,
        name: &str,
    ) -> Result<Vec<u8>, Qwen35Error> {
        let mut reader = self.open_tensor_reader(path)?;
        reader.read_tensor_bytes(name)
    }

    /// Alias for [`GgufManifest::read_tensor_bytes`].
    pub fn read_tensor(&self, path: impl AsRef<Path>, name: &str) -> Result<Vec<u8>, Qwen35Error> {
        self.read_tensor_bytes(path, name)
    }

    /// Open an inspected GGUF and read a bounded slice from one tensor.
    pub fn read_tensor_slice(
        &self,
        path: impl AsRef<Path>,
        name: &str,
        byte_offset: u64,
        byte_len: u64,
    ) -> Result<Vec<u8>, Qwen35Error> {
        let mut reader = self.open_tensor_reader(path)?;
        reader.read_tensor_slice(name, byte_offset, byte_len)
    }

    pub fn metadata(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }

    pub fn metadata_keys(&self) -> impl Iterator<Item = &str> {
        self.metadata.keys().map(String::as_str)
    }

    pub fn quantization_counts(&self) -> BTreeMap<u32, usize> {
        let mut counts = BTreeMap::new();
        for tensor in &self.tensors {
            *counts.entry(tensor.ggml_type).or_insert(0) += 1;
        }
        counts
    }

    pub fn quantization_bytes(&self) -> BTreeMap<u32, u64> {
        let mut bytes = BTreeMap::new();
        for tensor in &self.tensors {
            *bytes.entry(tensor.ggml_type).or_insert(0) += tensor.byte_len;
        }
        bytes
    }
}

impl<'a> GgufTensorReader<'a> {
    /// Return a tensor descriptor by exact name from the bound manifest.
    pub fn tensor(&self, name: &str) -> Option<&GgufTensor> {
        self.manifest.tensor(name)
    }

    /// Read one tensor into a caller-owned buffer.
    pub fn read_tensor_into(
        &mut self,
        name: &str,
        destination: &mut [u8],
    ) -> Result<(), Qwen35Error> {
        let (absolute_offset, byte_len) = {
            let tensor = self
                .manifest
                .tensor(name)
                .ok_or_else(|| Qwen35Error::Gguf(format!("unknown tensor {name:?}")))?;
            (tensor.absolute_offset, tensor.byte_len)
        };
        let expected_len = usize::try_from(byte_len).map_err(|_| Qwen35Error::SizeOverflow)?;
        if destination.len() != expected_len {
            return Err(Qwen35Error::invalid(
                format!("tensor {name} destination length"),
                format!("expected {expected_len}, got {}", destination.len()),
            ));
        }
        self.read_range(name, absolute_offset, byte_len, destination)
    }

    /// Read one tensor's exact raw payload bytes.
    pub fn read_tensor_bytes(&mut self, name: &str) -> Result<Vec<u8>, Qwen35Error> {
        let byte_len = self
            .manifest
            .tensor(name)
            .ok_or_else(|| Qwen35Error::Gguf(format!("unknown tensor {name:?}")))?
            .byte_len;
        let length = usize::try_from(byte_len).map_err(|_| Qwen35Error::SizeOverflow)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length)
            .map_err(|_| Qwen35Error::Gguf("tensor payload allocation failed".into()))?;
        bytes.resize(length, 0);
        self.read_tensor_into(name, &mut bytes)?;
        Ok(bytes)
    }

    /// Read a bounded byte slice from one tensor's payload.
    ///
    /// The slice is checked against the descriptor before any allocation or
    /// seek.  The same artifact identity, range, and TOCTOU checks used by a
    /// full tensor read are then applied by [`Self::read_range`].  Callers
    /// that decode typed values must provide element-aligned offsets and
    /// lengths for the selected GGML type.
    pub fn read_tensor_slice(
        &mut self,
        name: &str,
        byte_offset: u64,
        byte_len: u64,
    ) -> Result<Vec<u8>, Qwen35Error> {
        if byte_len == 0 {
            return Err(Qwen35Error::invalid(
                format!("tensor {name} slice"),
                "slice length must be nonzero",
            ));
        }
        let (absolute_offset, tensor_byte_len) = {
            let tensor = self
                .manifest
                .tensor(name)
                .ok_or_else(|| Qwen35Error::Gguf(format!("unknown tensor {name:?}")))?;
            (tensor.absolute_offset, tensor.byte_len)
        };
        let slice_end = byte_offset
            .checked_add(byte_len)
            .ok_or(Qwen35Error::SizeOverflow)?;
        if slice_end > tensor_byte_len {
            return Err(Qwen35Error::invalid(
                format!("tensor {name} slice"),
                format!(
                    "range [{byte_offset}, {slice_end}) exceeds tensor payload {tensor_byte_len}"
                ),
            ));
        }
        let absolute_slice_offset = absolute_offset
            .checked_add(byte_offset)
            .ok_or(Qwen35Error::SizeOverflow)?;
        let length = usize::try_from(byte_len).map_err(|_| Qwen35Error::SizeOverflow)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length)
            .map_err(|_| Qwen35Error::Gguf("tensor slice allocation failed".into()))?;
        bytes.resize(length, 0);
        self.read_range(name, absolute_slice_offset, byte_len, &mut bytes)?;
        Ok(bytes)
    }

    /// Alias for [`GgufTensorReader::read_tensor_bytes`].
    pub fn read_tensor(&mut self, name: &str) -> Result<Vec<u8>, Qwen35Error> {
        self.read_tensor_bytes(name)
    }

    fn read_range(
        &mut self,
        name: &str,
        absolute_offset: u64,
        byte_len: u64,
        destination: &mut [u8],
    ) -> Result<(), Qwen35Error> {
        let before = self.file.metadata()?;
        if !self.identity.matches(&before) || before.len() != self.manifest.file_size {
            return Err(Qwen35Error::Gguf(
                "bound artifact changed before tensor read".into(),
            ));
        }
        ensure_path_matches_identity(&self.path, &self.identity)?;
        validate_manifest_ranges(self.manifest, before.len())?;
        let expected_end = absolute_offset
            .checked_add(byte_len)
            .ok_or(Qwen35Error::SizeOverflow)?;
        if expected_end > before.len() {
            return Err(Qwen35Error::Gguf(format!(
                "tensor {name:?} range exceeds bound artifact"
            )));
        }
        self.file.seek(SeekFrom::Start(absolute_offset))?;
        self.file.read_exact(destination)?;
        let after = self.file.metadata()?;
        ensure_stable_metadata(&before, &after)?;
        if !self.identity.matches(&after) {
            return Err(Qwen35Error::Gguf(
                "bound artifact changed during tensor read".into(),
            ));
        }
        ensure_path_matches_identity(&self.path, &self.identity)?;
        Ok(())
    }
}

fn validate_manifest_ranges(manifest: &GgufManifest, file_size: u64) -> Result<(), Qwen35Error> {
    if manifest.version != GGUF_VERSION_2 && manifest.version != GGUF_VERSION_3 {
        return Err(Qwen35Error::Gguf(
            "manifest GGUF version is unsupported".into(),
        ));
    }
    if manifest.file_size != file_size {
        return Err(Qwen35Error::Gguf(
            "manifest file size differs from opened artifact".into(),
        ));
    }
    let tensor_count =
        u64::try_from(manifest.tensors.len()).map_err(|_| Qwen35Error::SizeOverflow)?;
    if manifest.tensor_count != tensor_count {
        return Err(Qwen35Error::Gguf(
            "manifest tensor count differs from tensor descriptors".into(),
        ));
    }
    if manifest.alignment == 0
        || manifest.alignment > (1 << 20)
        || manifest.alignment & (manifest.alignment - 1) != 0
    {
        return Err(Qwen35Error::Gguf("manifest alignment is invalid".into()));
    }
    let expected_data_offset = align_up(manifest.header_bytes, manifest.alignment)?;
    if manifest.data_offset != expected_data_offset {
        return Err(Qwen35Error::Gguf(
            "manifest data offset is inconsistent with its header".into(),
        ));
    }
    if manifest.data_offset > file_size {
        return Err(Qwen35Error::Gguf(
            "manifest data offset exceeds opened artifact".into(),
        ));
    }
    let payload_size = file_size - manifest.data_offset;
    let mut ranges = Vec::with_capacity(manifest.tensors.len());
    let mut names = BTreeSet::new();
    for tensor in &manifest.tensors {
        if tensor.dimensions.is_empty() || tensor.dimensions.len() > 4 {
            return Err(Qwen35Error::Gguf(format!(
                "tensor {:?} has invalid dimension count",
                tensor.name
            )));
        }
        if tensor.dimensions.contains(&0) {
            return Err(Qwen35Error::Gguf(format!(
                "tensor {:?} has a zero dimension",
                tensor.name
            )));
        }
        if !names.insert(tensor.name.as_str()) {
            return Err(Qwen35Error::Gguf(format!(
                "duplicate tensor name {:?}",
                tensor.name
            )));
        }
        let expected_len =
            tensor_byte_len(&tensor.dimensions, tensor.ggml_type).ok_or_else(|| {
                Qwen35Error::Gguf(format!(
                    "tensor {:?} uses unsupported or invalid GGML type {}",
                    tensor.name, tensor.ggml_type
                ))
            })?;
        if tensor.byte_len != expected_len {
            return Err(Qwen35Error::Gguf(format!(
                "tensor {:?} byte length does not match its geometry",
                tensor.name
            )));
        }
        if tensor.relative_offset % manifest.alignment != 0 {
            return Err(Qwen35Error::Gguf(format!(
                "tensor {:?} offset is not aligned",
                tensor.name
            )));
        }
        let relative_end = tensor
            .relative_offset
            .checked_add(tensor.byte_len)
            .ok_or(Qwen35Error::SizeOverflow)?;
        if relative_end > payload_size {
            return Err(Qwen35Error::Gguf(format!(
                "tensor {:?} range exceeds payload",
                tensor.name
            )));
        }
        let expected_absolute = manifest
            .data_offset
            .checked_add(tensor.relative_offset)
            .ok_or(Qwen35Error::SizeOverflow)?;
        if tensor.absolute_offset != expected_absolute {
            return Err(Qwen35Error::Gguf(format!(
                "tensor {:?} absolute offset is inconsistent",
                tensor.name
            )));
        }
        let absolute_end = expected_absolute
            .checked_add(tensor.byte_len)
            .ok_or(Qwen35Error::SizeOverflow)?;
        if absolute_end > file_size {
            return Err(Qwen35Error::Gguf(format!(
                "tensor {:?} absolute range exceeds artifact",
                tensor.name
            )));
        }
        ranges.push((expected_absolute, absolute_end, tensor.name.as_str()));
    }
    ranges.sort_unstable_by_key(|range| range.0);
    for pair in ranges.windows(2) {
        if pair[0].1 > pair[1].0 {
            return Err(Qwen35Error::Gguf(format!(
                "tensor ranges overlap: {:?} and {:?}",
                pair[0].2, pair[1].2
            )));
        }
    }
    Ok(())
}

struct Reader {
    file: File,
    position: u64,
    file_size: u64,
    max_position: u64,
}

impl Reader {
    fn ensure(&self, amount: u64) -> Result<(), Qwen35Error> {
        let end = self
            .position
            .checked_add(amount)
            .ok_or(Qwen35Error::SizeOverflow)?;
        if end > self.file_size || end > self.max_position {
            return Err(Qwen35Error::Gguf(
                "GGUF header exceeds file or configured bound".into(),
            ));
        }
        Ok(())
    }

    fn read_exact<const N: usize>(&mut self) -> Result<[u8; N], Qwen35Error> {
        self.ensure(N as u64)?;
        let mut bytes = [0u8; N];
        self.file.read_exact(&mut bytes)?;
        self.position += N as u64;
        Ok(bytes)
    }

    fn read_u32(&mut self) -> Result<u32, Qwen35Error> {
        Ok(u32::from_le_bytes(self.read_exact()?))
    }

    fn read_u64(&mut self) -> Result<u64, Qwen35Error> {
        Ok(u64::from_le_bytes(self.read_exact()?))
    }

    fn read_i8(&mut self) -> Result<i8, Qwen35Error> {
        Ok(i8::from_le_bytes(self.read_exact()?))
    }

    fn read_i16(&mut self) -> Result<i16, Qwen35Error> {
        Ok(i16::from_le_bytes(self.read_exact()?))
    }

    fn read_i32(&mut self) -> Result<i32, Qwen35Error> {
        Ok(i32::from_le_bytes(self.read_exact()?))
    }

    fn read_i64(&mut self) -> Result<i64, Qwen35Error> {
        Ok(i64::from_le_bytes(self.read_exact()?))
    }

    fn read_f32(&mut self) -> Result<f32, Qwen35Error> {
        Ok(f32::from_le_bytes(self.read_exact()?))
    }

    fn read_f64(&mut self) -> Result<f64, Qwen35Error> {
        Ok(f64::from_le_bytes(self.read_exact()?))
    }

    fn read_string(&mut self) -> Result<String, Qwen35Error> {
        let length = self.read_u64()?;
        if length > MAX_STRING_BYTES {
            return Err(Qwen35Error::Gguf(format!(
                "string length {length} exceeds bounded limit"
            )));
        }
        self.ensure(length)?;
        let mut bytes = vec![0u8; usize::try_from(length).map_err(|_| Qwen35Error::SizeOverflow)?];
        self.file.read_exact(&mut bytes)?;
        self.position += length;
        String::from_utf8(bytes)
            .map_err(|_| Qwen35Error::Gguf("metadata string is not valid UTF-8".into()))
    }

    fn read_value(&mut self, depth: usize) -> Result<GgufValue, Qwen35Error> {
        if depth > 16 {
            return Err(Qwen35Error::Gguf(
                "metadata array nesting is too deep".into(),
            ));
        }
        match self.read_u32()? {
            0 => Ok(GgufValue::U64(self.read_exact::<1>()?[0] as u64)),
            1 => Ok(GgufValue::I64(self.read_i8()? as i64)),
            2 => Ok(GgufValue::U64(u16::from_le_bytes(self.read_exact()?) as u64)),
            3 => Ok(GgufValue::I64(self.read_i16()? as i64)),
            4 => Ok(GgufValue::U64(self.read_u32()? as u64)),
            5 => Ok(GgufValue::I64(self.read_i32()? as i64)),
            6 => {
                let value = self.read_f32()? as f64;
                if value.is_finite() {
                    Ok(GgufValue::F64(value))
                } else {
                    Err(Qwen35Error::Gguf("non-finite FLOAT32 metadata".into()))
                }
            }
            7 => {
                let value = self.read_exact::<1>()?[0];
                match value {
                    0 => Ok(GgufValue::Bool(false)),
                    1 => Ok(GgufValue::Bool(true)),
                    _ => Err(Qwen35Error::Gguf("boolean metadata is not 0 or 1".into())),
                }
            }
            8 => Ok(GgufValue::String(self.read_string()?)),
            9 => {
                let element_type = self.read_u32()?;
                let count = self.read_u64()?;
                if count > MAX_ARRAY_ELEMENTS {
                    return Err(Qwen35Error::Gguf(format!(
                        "metadata array length {count} exceeds bounded limit"
                    )));
                }
                let mut values = Vec::with_capacity(
                    usize::try_from(count).map_err(|_| Qwen35Error::SizeOverflow)?,
                );
                for _ in 0..count {
                    values.push(self.read_value_with_type(element_type, depth + 1)?);
                }
                Ok(GgufValue::Array(values))
            }
            10 => Ok(GgufValue::U64(self.read_u64()?)),
            11 => Ok(GgufValue::I64(self.read_i64()?)),
            12 => {
                let value = self.read_f64()?;
                if value.is_finite() {
                    Ok(GgufValue::F64(value))
                } else {
                    Err(Qwen35Error::Gguf("non-finite FLOAT64 metadata".into()))
                }
            }
            kind => Err(Qwen35Error::Gguf(format!(
                "unknown GGUF metadata value type {kind}"
            ))),
        }
    }

    fn read_value_with_type(&mut self, kind: u32, depth: usize) -> Result<GgufValue, Qwen35Error> {
        if depth > 16 {
            return Err(Qwen35Error::Gguf(
                "metadata array nesting is too deep".into(),
            ));
        }
        match kind {
            0 => Ok(GgufValue::U64(self.read_exact::<1>()?[0] as u64)),
            1 => Ok(GgufValue::I64(self.read_i8()? as i64)),
            2 => Ok(GgufValue::U64(u16::from_le_bytes(self.read_exact()?) as u64)),
            3 => Ok(GgufValue::I64(self.read_i16()? as i64)),
            4 => Ok(GgufValue::U64(self.read_u32()? as u64)),
            5 => Ok(GgufValue::I64(self.read_i32()? as i64)),
            6 => {
                let value = self.read_f32()? as f64;
                if value.is_finite() {
                    Ok(GgufValue::F64(value))
                } else {
                    Err(Qwen35Error::Gguf("non-finite FLOAT32 metadata".into()))
                }
            }
            7 => {
                let value = self.read_exact::<1>()?[0];
                match value {
                    0 => Ok(GgufValue::Bool(false)),
                    1 => Ok(GgufValue::Bool(true)),
                    _ => Err(Qwen35Error::Gguf("boolean metadata is not 0 or 1".into())),
                }
            }
            8 => Ok(GgufValue::String(self.read_string()?)),
            9 => {
                let nested_kind = self.read_u32()?;
                let count = self.read_u64()?;
                if count > MAX_ARRAY_ELEMENTS {
                    return Err(Qwen35Error::Gguf(format!(
                        "metadata array length {count} exceeds bounded limit"
                    )));
                }
                let mut values = Vec::with_capacity(
                    usize::try_from(count).map_err(|_| Qwen35Error::SizeOverflow)?,
                );
                for _ in 0..count {
                    values.push(self.read_value_with_type(nested_kind, depth + 1)?);
                }
                Ok(GgufValue::Array(values))
            }
            10 => Ok(GgufValue::U64(self.read_u64()?)),
            11 => Ok(GgufValue::I64(self.read_i64()?)),
            12 => {
                let value = self.read_f64()?;
                if value.is_finite() {
                    Ok(GgufValue::F64(value))
                } else {
                    Err(Qwen35Error::Gguf("non-finite FLOAT64 metadata".into()))
                }
            }
            kind => Err(Qwen35Error::Gguf(format!(
                "unknown GGUF metadata array element type {kind}"
            ))),
        }
    }
}

fn align_up(value: u64, alignment: u64) -> Result<u64, Qwen35Error> {
    let mask = alignment - 1;
    value
        .checked_add(mask)
        .map(|aligned| aligned & !mask)
        .ok_or(Qwen35Error::SizeOverflow)
}

fn tensor_byte_len(dimensions: &[u64], ggml_type: u32) -> Option<u64> {
    let elements = dimensions
        .iter()
        .try_fold(1u64, |product, dimension| product.checked_mul(*dimension))?;
    let (block_elements, block_bytes) = match ggml_type {
        0 => (1, 4),   // F32
        1 => (1, 2),   // F16
        2 => (32, 18), // Q4_0
        3 => (32, 20), // Q4_1
        6 => (32, 22), // Q5_0
        7 => (32, 24), // Q5_1
        8 => (32, 34), // Q8_0
        9 => (32, 36), // Q8_1
        10 => (256, 84),
        11 => (256, 110),
        12 => (256, 144),
        13 => (256, 176),
        14 => (256, 210),
        15 => (256, 292),
        16 => (256, 66),
        17 => (256, 74),
        18 => (256, 98),
        19 => (256, 50),
        20 => (32, 18),
        21 => (256, 110),
        22 => (256, 82),
        23 => (256, 136),
        24 => (1, 1), // I8
        25 => (1, 2), // I16
        26 => (1, 4), // I32
        27 => (1, 8), // I64
        28 => (1, 8), // F64
        29 => (256, 56),
        30 => (1, 2), // BF16
        34 => (256, 84),
        35 => (256, 66),
        39 => (32, 17), // MXFP4
        40 => (64, 36), // NVFP4
        41 => (128, 18),
        42 => (256, 66),
        _ => return None,
    };
    if elements % block_elements != 0 {
        return None;
    }
    elements
        .checked_div(block_elements)
        .and_then(|blocks| blocks.checked_mul(block_bytes))
}

fn validate_regular_path(path: &Path) -> Result<(), Qwen35Error> {
    let mut current = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => current.push(prefix.as_os_str()),
            Component::RootDir => current.push(std::path::MAIN_SEPARATOR.to_string()),
            Component::CurDir => {}
            Component::ParentDir => {
                return Err(Qwen35Error::Gguf(
                    "artifact path may not contain parent components".into(),
                ));
            }
            Component::Normal(name) => {
                current.push(name);
                let metadata = fs::symlink_metadata(&current).map_err(|error| {
                    Qwen35Error::Gguf(format!(
                        "cannot inspect path component {current:?}: {error}"
                    ))
                })?;
                if metadata.file_type().is_symlink() || is_reparse_point(&metadata) {
                    return Err(Qwen35Error::Gguf(format!(
                        "reparse/symlink path component is not allowed: {current:?}"
                    )));
                }
            }
        }
    }
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.is_file() || metadata.file_type().is_symlink() || is_reparse_point(&metadata) {
        return Err(Qwen35Error::Gguf(
            "artifact must be a regular non-reparse file".into(),
        ));
    }
    Ok(())
}

fn ensure_path_matches_identity(
    path: &Path,
    expected: &GgufFileIdentity,
) -> Result<(), Qwen35Error> {
    validate_regular_path(path)?;
    let metadata = fs::symlink_metadata(path)?;
    if !expected.matches(&metadata) {
        return Err(Qwen35Error::Gguf(
            "artifact path identity differs from opened descriptor".into(),
        ));
    }
    Ok(())
}

fn is_reparse_point(metadata: &Metadata) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0400;
        metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
    }
    #[cfg(not(windows))]
    {
        let _ = metadata;
        false
    }
}

fn ensure_stable_metadata(before: &Metadata, after: &Metadata) -> Result<(), Qwen35Error> {
    if !GgufFileIdentity::from_metadata(before).matches(after) {
        return Err(Qwen35Error::Gguf(
            "artifact identity or metadata changed during read".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn put_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    fn put_metadata_string(bytes: &mut Vec<u8>, key: &str, value: &str) {
        put_string(bytes, key);
        bytes.extend_from_slice(&8u32.to_le_bytes());
        put_string(bytes, value);
    }

    fn put_metadata_u32(bytes: &mut Vec<u8>, key: &str, value: u32) {
        put_string(bytes, key);
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn put_metadata_string_array(bytes: &mut Vec<u8>, key: &str, values: &[&str]) {
        put_string(bytes, key);
        bytes.extend_from_slice(&9u32.to_le_bytes());
        bytes.extend_from_slice(&8u32.to_le_bytes());
        bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            put_string(bytes, value);
        }
    }

    fn put_tensor(bytes: &mut Vec<u8>, name: &str, dimensions: &[u64], kind: u32, offset: u64) {
        put_string(bytes, name);
        bytes.extend_from_slice(&(dimensions.len() as u32).to_le_bytes());
        for dimension in dimensions {
            bytes.extend_from_slice(&dimension.to_le_bytes());
        }
        bytes.extend_from_slice(&kind.to_le_bytes());
        bytes.extend_from_slice(&offset.to_le_bytes());
    }

    fn synthetic_file(
        metadata_count: u64,
        duplicate: bool,
        bad_range: bool,
        second_kind: u32,
    ) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&GGUF_VERSION_3.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes());
        bytes.extend_from_slice(&metadata_count.to_le_bytes());
        put_metadata_string(&mut bytes, "general.architecture", "qwen35");
        put_metadata_u32(&mut bytes, "general.alignment", 32);
        put_metadata_string_array(&mut bytes, "tokenizer.ggml.tokens", &["a", "b"]);
        if duplicate {
            put_metadata_string(&mut bytes, "general.architecture", "qwen35");
        }
        let second_offset = if bad_range { 32_000 } else { 64 };
        put_tensor(&mut bytes, "nvfp4", &[64], 40, 0);
        put_tensor(&mut bytes, "second", &[1], second_kind, second_offset);
        let data_offset = align_up(bytes.len() as u64, 32).unwrap();
        bytes.resize(data_offset as usize, 0);
        let second_bytes = tensor_byte_len(&[1], second_kind).unwrap_or(4);
        let payload_end = if bad_range {
            data_offset as usize
        } else {
            data_offset as usize + second_offset as usize + second_bytes as usize
        };
        bytes.resize(payload_end, 0);
        if !bad_range {
            for (index, byte) in bytes[data_offset as usize..].iter_mut().enumerate() {
                *byte = (index as u8).wrapping_mul(17).wrapping_add(3);
            }
        }
        bytes
    }

    fn temp_path(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "augur-qwen35-{label}-{}-{}.gguf",
            std::process::id(),
            TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[test]
    fn parses_header_metadata_and_tensor_ranges() {
        let path = temp_path("positive");
        fs::write(&path, synthetic_file(3, false, false, 0)).unwrap();
        let manifest = GgufManifest::inspect(
            &path,
            GgufAuditOptions {
                max_header_bytes: 1 << 20,
                hash_file: false,
            },
        )
        .unwrap();
        assert_eq!(manifest.version, 3);
        assert_eq!(manifest.tensor_count, 2);
        assert_eq!(manifest.metadata_count, 3);
        assert_eq!(
            manifest
                .metadata("general.architecture")
                .and_then(GgufValue::as_str),
            Some("qwen35")
        );
        assert_eq!(
            manifest
                .metadata("tokenizer.ggml.tokens")
                .and_then(GgufValue::array_len),
            Some(2)
        );
        assert_eq!(manifest.quantization_counts().get(&40), Some(&1));
        assert_eq!(manifest.tensors[0].byte_len, 36);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn duplicate_metadata_key_is_rejected() {
        let path = temp_path("duplicate");
        fs::write(&path, synthetic_file(4, true, false, 0)).unwrap();
        let result = GgufManifest::inspect(
            &path,
            GgufAuditOptions {
                hash_file: false,
                ..Default::default()
            },
        );
        assert!(result.is_err());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn tensor_range_overflow_is_rejected() {
        let path = temp_path("range");
        fs::write(&path, synthetic_file(3, false, true, 0)).unwrap();
        let result = GgufManifest::inspect(
            &path,
            GgufAuditOptions {
                hash_file: false,
                ..Default::default()
            },
        );
        assert!(result.is_err());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn unsupported_tensor_type_is_rejected() {
        let bytes = synthetic_file(3, false, false, 999);
        let path = temp_path("type");
        fs::write(&path, bytes).unwrap();
        let result = GgufManifest::inspect(
            &path,
            GgufAuditOptions {
                hash_file: false,
                ..Default::default()
            },
        );
        assert!(result.is_err());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn descriptor_bound_reader_returns_exact_payload_and_supports_reuse() {
        let path = temp_path("reader-positive");
        fs::write(&path, synthetic_file(3, false, false, 0)).unwrap();
        let manifest = GgufManifest::inspect(
            &path,
            GgufAuditOptions {
                hash_file: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(manifest.tensor("nvfp4").unwrap().byte_len, 36);
        let mut reader = manifest.open_tensor_reader(&path).unwrap();
        let first = reader.read_tensor_bytes("nvfp4").unwrap();
        assert_eq!(first.len(), 36);
        assert_eq!(&first[..4], &[3, 20, 37, 54]);
        let mut second = vec![0u8; 4];
        reader.read_tensor_into("second", &mut second).unwrap();
        assert_eq!(second, vec![67, 84, 101, 118]);
        assert!(reader.read_tensor_bytes("missing").is_err());
        fs::remove_file(path).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn pinned_reader_matches_legacy_and_keeps_full_digest() {
        let path = temp_path("reader-pinned-exact");
        fs::write(&path, synthetic_file(3, false, false, 0)).unwrap();
        let legacy = GgufManifest::inspect(&path, GgufAuditOptions::default()).unwrap();
        let mut artifact = GgufPinnedArtifact::open(&path, GgufAuditOptions::default()).unwrap();
        assert_eq!(artifact.manifest(), &legacy);
        let expected = legacy.read_tensor_bytes(&path, "nvfp4").unwrap();
        let mut reader = artifact.tensor_reader().unwrap();
        assert_eq!(reader.read_tensor_bytes("nvfp4").unwrap(), expected);
        assert!(reader.read_tensor_slice("second", 3, 2).is_err());
        drop(reader);
        drop(artifact);
        assert!(GgufPinnedArtifact::open(&path, GgufAuditOptions { hash_file: false, ..Default::default() }).is_err());
        fs::remove_file(path).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn pinned_reader_prevents_writer_and_path_replacement_until_drop() {
        let path = temp_path("reader-pinned-writer");
        let replaced = temp_path("reader-pinned-replacement");
        let bytes = synthetic_file(3, false, false, 0);
        fs::write(&path, &bytes).unwrap();
        let mut artifact = GgufPinnedArtifact::open(&path, GgufAuditOptions::default()).unwrap();
        assert!(OpenOptions::new().write(true).open(&path).is_err());
        assert!(fs::rename(&path, &replaced).is_err());
        let mut fake = artifact.manifest().clone();
        fake.file_sha256 = Some("0".repeat(64));
        assert_ne!(fake.file_sha256, artifact.manifest().file_sha256);
        assert_eq!(artifact.tensor_reader().unwrap().read_tensor_bytes("second").unwrap(), [67, 84, 101, 118]);
        drop(artifact);
        fs::write(&path, &bytes).unwrap();
        fs::remove_file(path).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn overlapped_full_hash_and_tensor_reads_use_independent_cursors() {
        let path = temp_path("overlap-cursors");
        let mut bytes = synthetic_file(3, false, false, 0);
        // Tail bytes are not tensors but MUST still participate in the full hash.
        bytes.extend((0..17*1024*1024).map(|i| (i % 251) as u8));
        fs::write(&path, &bytes).unwrap();
        let expected = crate::identity::sha256_hex(&bytes);
        let (loaded, mut artifact, timing) = GgufPinnedArtifact::load_overlapped(
            &path, GgufAuditOptions::default(), |manifest, reader| -> Result<_, Qwen35Error> {
                assert!(manifest.file_sha256.is_none());
                assert!(OpenOptions::new().write(true).open(&path).is_err());
                for _ in 0..256 {
                    assert_eq!(reader.read_tensor_bytes("second")?, [67, 84, 101, 118]);
                    assert_eq!(&reader.read_tensor_bytes("nvfp4")?[..4], &[3, 20, 37, 54]);
                }
                // Rc demonstrates that only hashing, never a CUDA owner/value,
                // crosses threads; the payload callback stays on its caller.
                Ok(std::rc::Rc::new(reader.read_tensor_bytes("second")?))
            }).unwrap();
        assert_eq!(loaded.as_slice(), [67, 84, 101, 118]);
        assert_eq!(artifact.manifest().file_sha256.as_deref(), Some(expected.as_str()));
        assert!(timing.enclosing_ms >= timing.payload_load_ms);
        assert!(timing.enclosing_ms >= timing.full_file_hash_ms);
        assert_eq!(artifact.tensor_reader().unwrap().read_tensor_bytes("second").unwrap(), *loaded);
        drop(artifact);
        fs::remove_file(path).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn overlapped_callback_failure_joins_worker_and_releases_locks() {
        let path = temp_path("overlap-failure");
        let bytes = synthetic_file(3, false, false, 0);
        fs::write(&path, &bytes).unwrap();
        let result = GgufPinnedArtifact::load_overlapped(&path, GgufAuditOptions::default(),
            |_, _| -> Result<(), Qwen35Error> { Err(Qwen35Error::Gguf("payload preparation failed".into())) });
        assert!(result.is_err());
        fs::write(&path, &bytes).unwrap();
        fs::remove_file(path).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn overlapped_callback_panic_joins_worker_and_releases_locks() {
        let path = temp_path("overlap-panic");
        let bytes = synthetic_file(3, false, false, 0);
        fs::write(&path, &bytes).unwrap();
        let result = std::panic::catch_unwind(|| {
            GgufPinnedArtifact::load_overlapped(&path, GgufAuditOptions::default(),
                |_, _| -> Result<(), Qwen35Error> { panic!("payload preparation panicked") })
        });
        assert!(result.is_err());
        fs::write(&path, &bytes).unwrap();
        fs::remove_file(path).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn overlapped_loading_rejects_unhashed_options_before_callback() {
        let path = temp_path("overlap-nohash");
        fs::write(&path, synthetic_file(3, false, false, 0)).unwrap();
        let result = GgufPinnedArtifact::load_overlapped(&path,
            GgufAuditOptions { hash_file: false, ..Default::default() },
            |_, _| -> Result<(), Qwen35Error> { panic!("must not load without authentication") });
        assert!(result.is_err());
        fs::remove_file(path).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn overlapped_loading_rejects_invalid_header_before_callback() {
        let path = temp_path("overlap-badheader");
        fs::write(&path, b"not a valid GGUF file header").unwrap();
        let result = GgufPinnedArtifact::load_overlapped(&path, GgufAuditOptions::default(),
            |_, _| -> Result<(), Qwen35Error> { panic!("must validate header before loading") });
        assert!(result.is_err());
        fs::write(&path, b"locks released").unwrap();
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn descriptor_bound_reader_reads_only_in_tensor_slice() {
        let path = temp_path("reader-slice");
        fs::write(&path, synthetic_file(3, false, false, 0)).unwrap();
        let manifest = GgufManifest::inspect(&path, GgufAuditOptions::default()).unwrap();
        let mut reader = manifest.open_tensor_reader(&path).unwrap();
        assert_eq!(
            reader.read_tensor_slice("second", 1, 2).unwrap(),
            vec![84, 101]
        );
        assert!(reader.read_tensor_slice("second", 0, 0).is_err());
        assert!(reader.read_tensor_slice("second", 3, 2).is_err());
        assert!(reader.read_tensor_slice("second", u64::MAX, 1).is_err());
        assert!(reader.read_tensor_slice("missing", 0, 1).is_err());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn descriptor_bound_reader_decodes_a_tensor_through_the_quant_layer() {
        let path = temp_path("reader-dequant");
        fs::write(&path, synthetic_file(3, false, false, 0)).unwrap();
        let manifest = GgufManifest::inspect(&path, GgufAuditOptions::default()).unwrap();
        let mut reader = manifest.open_tensor_reader(&path).unwrap();
        let values = reader.read_tensor_f32("nvfp4").unwrap();
        assert_eq!(values.len(), 64);
        assert!(values.iter().all(|value| value.is_finite()));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn reader_rejects_destination_size_and_manifest_range_drift() {
        let path = temp_path("reader-range");
        fs::write(&path, synthetic_file(3, false, false, 0)).unwrap();
        let manifest = GgufManifest::inspect(
            &path,
            GgufAuditOptions {
                hash_file: false,
                ..Default::default()
            },
        )
        .unwrap();
        let mut reader = manifest.open_tensor_reader(&path).unwrap();
        assert!(reader.read_tensor_into("nvfp4", &mut [0u8; 1]).is_err());

        let mut drifted = manifest.clone();
        drifted.tensors[0].absolute_offset += drifted.alignment;
        assert!(drifted.read_tensor_bytes(&path, "nvfp4").is_err());
        let mut digest_drifted = manifest.clone();
        digest_drifted.file_sha256 = Some("0".repeat(64));
        assert!(digest_drifted.read_tensor_bytes(&path, "nvfp4").is_err());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn reader_rejects_replaced_artifact_even_when_name_and_size_match() {
        let path = temp_path("reader-replaced");
        let original = synthetic_file(3, false, false, 0);
        fs::write(&path, &original).unwrap();
        let manifest = GgufManifest::inspect(&path, GgufAuditOptions::default()).unwrap();
        let mut replacement = original.clone();
        let last = replacement.last_mut().unwrap();
        *last ^= 0x5a;
        fs::write(&path, replacement).unwrap();
        assert!(manifest.open_tensor_reader(&path).is_err());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn reader_rejects_parent_components() {
        let path = temp_path("reader-parent");
        fs::write(&path, synthetic_file(3, false, false, 0)).unwrap();
        let manifest = GgufManifest::inspect(&path, GgufAuditOptions::default()).unwrap();
        let parent_path = path
            .parent()
            .unwrap()
            .join("..")
            .join(path.file_name().unwrap());
        assert!(manifest.open_tensor_reader(parent_path).is_err());
        fs::remove_file(path).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn reader_rejects_nested_symlink_component() {
        use std::os::unix::fs::symlink;

        let real_dir = std::env::temp_dir().join(format!(
            "augur-qwen35-reader-real-{}-{}",
            std::process::id(),
            TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let alias_dir = std::env::temp_dir().join(format!(
            "augur-qwen35-reader-alias-{}-{}",
            std::process::id(),
            TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&real_dir).unwrap();
        let real_file = real_dir.join("model.gguf");
        fs::write(&real_file, synthetic_file(3, false, false, 0)).unwrap();
        symlink(&real_dir, &alias_dir).unwrap();
        assert!(
            GgufManifest::inspect(alias_dir.join("model.gguf"), GgufAuditOptions::default())
                .is_err()
        );
        fs::remove_file(real_file).unwrap();
        fs::remove_dir(alias_dir).unwrap();
        fs::remove_dir(real_dir).unwrap();
    }
}
