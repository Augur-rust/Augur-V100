//! Strict, descriptor-bound reader for the Hugging Face safetensors format.
//!
//! This reader authenticates the header, checks every data range, and exposes
//! bounded byte reads for callers that own tensor semantics. The file is
//! re-opened for each read and its metadata is checked before and after.

use crate::{strict_json::StrictValue, Qwen35Error};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File, Metadata, OpenOptions};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Component, Path, PathBuf};

const MAX_HEADER_BYTES: u64 = 128 * 1024 * 1024;
const MAX_RANK: usize = 8;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SafetensorsRecord {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub data_offsets: [u64; 2],
}

impl SafetensorsRecord {
    pub fn element_count(&self) -> Result<u64, Qwen35Error> {
        self.shape.iter().try_fold(1_u64, |acc, value| {
            acc.checked_mul(*value).ok_or(Qwen35Error::SizeOverflow)
        })
    }

    pub fn byte_len(&self) -> Result<u64, Qwen35Error> {
        self.data_offsets[1]
            .checked_sub(self.data_offsets[0])
            .ok_or_else(|| Qwen35Error::invalid(self.name.clone(), "data offsets are reversed"))
    }
}

#[derive(Clone, Debug)]
struct FileIdentity {
    len: u64,
    modified: Option<std::time::SystemTime>,
    created: Option<std::time::SystemTime>,
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

impl FileIdentity {
    fn from_metadata(metadata: &Metadata) -> Self {
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt;
        #[cfg(windows)]
        use std::os::windows::fs::MetadataExt;

        Self {
            len: metadata.len(),
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

impl PartialEq for FileIdentity {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len
            && self.modified == other.modified
            && self.created == other.created
            && self.cfg_eq(other)
    }
}

impl Eq for FileIdentity {}

#[cfg(unix)]
impl FileIdentity {
    fn cfg_eq(&self, other: &Self) -> bool {
        self.device == other.device && self.inode == other.inode
    }
}

#[cfg(windows)]
impl FileIdentity {
    fn cfg_eq(&self, other: &Self) -> bool {
        self.file_attributes == other.file_attributes
            && self.creation_time == other.creation_time
            && self.last_write_time == other.last_write_time
    }
}

#[cfg(not(any(unix, windows)))]
impl FileIdentity {
    fn cfg_eq(&self, _other: &Self) -> bool {
        true
    }
}

#[derive(Clone, Debug)]
pub struct SafetensorsManifest {
    path: PathBuf,
    pub file_size: u64,
    pub header_bytes: u64,
    pub data_offset: u64,
    pub file_sha256: Option<String>,
    records: BTreeMap<String, SafetensorsRecord>,
    identity: FileIdentity,
}

impl SafetensorsManifest {
    pub fn inspect(path: impl AsRef<Path>, hash_file: bool) -> Result<Self, Qwen35Error> {
        let path = path.as_ref();
        validate_regular_path(path)?;
        let mut file = OpenOptions::new().read(true).open(path)?;
        Self::inspect_file(path, &mut file, hash_file)
    }

    fn inspect_file(path: &Path, file: &mut File, hash_file: bool) -> Result<Self, Qwen35Error> {
        file.seek(SeekFrom::Start(0))?;
        let before = file.metadata()?;
        if !before.is_file() {
            return Err(Qwen35Error::Io(
                "safetensors artifact is not a regular file".into(),
            ));
        }
        let identity = FileIdentity::from_metadata(&before);
        ensure_path_identity(path, &identity)?;
        if before.len() < 8 {
            return Err(Qwen35Error::Gguf(
                "safetensors file is shorter than header length".into(),
            ));
        }
        let mut length_bytes = [0_u8; 8];
        file.read_exact(&mut length_bytes)?;
        let header_bytes = u64::from_le_bytes(length_bytes);
        if header_bytes == 0 || header_bytes > MAX_HEADER_BYTES {
            return Err(Qwen35Error::invalid(
                "safetensors.header_length",
                format!("header length {header_bytes} exceeds bounded limit"),
            ));
        }
        let data_offset = 8_u64
            .checked_add(header_bytes)
            .ok_or(Qwen35Error::SizeOverflow)?;
        if data_offset > before.len() {
            return Err(Qwen35Error::Gguf("safetensors header exceeds file".into()));
        }
        let header_len: usize = header_bytes
            .try_into()
            .map_err(|_| Qwen35Error::SizeOverflow)?;
        let mut header = vec![0_u8; header_len];
        file.read_exact(&mut header)?;
        let _ = StrictValue::parse(&header)?;
        let value: Value = serde_json::from_slice(&header)
            .map_err(|error| Qwen35Error::Json(error.to_string()))?;
        let object = value
            .as_object()
            .ok_or_else(|| Qwen35Error::invalid("safetensors.header", "expected object"))?;

        let mut records = BTreeMap::new();
        for (name, descriptor) in object {
            if name == "__metadata__" {
                if !descriptor.is_object() {
                    return Err(Qwen35Error::invalid(
                        "safetensors.__metadata__",
                        "expected object",
                    ));
                }
                continue;
            }
            let descriptor = descriptor.as_object().ok_or_else(|| {
                Qwen35Error::invalid(
                    format!("safetensors.tensor.{name}"),
                    "expected descriptor object",
                )
            })?;
            let dtype = descriptor
                .get("dtype")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    Qwen35Error::MissingField(format!("safetensors.tensor.{name}.dtype"))
                })?
                .to_owned();
            let shape_value = descriptor
                .get("shape")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    Qwen35Error::MissingField(format!("safetensors.tensor.{name}.shape"))
                })?;
            if shape_value.len() > MAX_RANK {
                return Err(Qwen35Error::invalid(
                    format!("safetensors.tensor.{name}.shape"),
                    "rank exceeds bounded limit",
                ));
            }
            let mut shape = Vec::with_capacity(shape_value.len());
            for (index, value) in shape_value.iter().enumerate() {
                shape.push(value.as_u64().ok_or_else(|| {
                    Qwen35Error::invalid(
                        format!("safetensors.tensor.{name}.shape[{index}]"),
                        "expected nonnegative integer",
                    )
                })?);
            }
            let offsets = descriptor
                .get("data_offsets")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    Qwen35Error::MissingField(format!("safetensors.tensor.{name}.data_offsets"))
                })?;
            if offsets.len() != 2 {
                return Err(Qwen35Error::invalid(
                    format!("safetensors.tensor.{name}.data_offsets"),
                    "expected exactly two offsets",
                ));
            }
            let start = offsets[0].as_u64().ok_or_else(|| {
                Qwen35Error::invalid(
                    format!("safetensors.tensor.{name}.data_offsets[0]"),
                    "expected nonnegative integer",
                )
            })?;
            let end = offsets[1].as_u64().ok_or_else(|| {
                Qwen35Error::invalid(
                    format!("safetensors.tensor.{name}.data_offsets[1]"),
                    "expected nonnegative integer",
                )
            })?;
            if end < start {
                return Err(Qwen35Error::invalid(
                    format!("safetensors.tensor.{name}.data_offsets"),
                    "end precedes start",
                ));
            }
            if data_offset
                .checked_add(end)
                .ok_or(Qwen35Error::SizeOverflow)?
                > before.len()
            {
                return Err(Qwen35Error::invalid(
                    format!("safetensors.tensor.{name}.data_offsets"),
                    "range exceeds file",
                ));
            }
            let record = SafetensorsRecord {
                name: name.clone(),
                dtype,
                shape,
                data_offsets: [start, end],
            };
            let expected_bytes = expected_dtype_bytes(&record)?;
            if expected_bytes != record.byte_len()? {
                return Err(Qwen35Error::invalid(
                    format!("safetensors.tensor.{name}"),
                    format!(
                        "shape/dtype imply {expected_bytes} bytes, range has {}",
                        record.byte_len()?
                    ),
                ));
            }
            if records.insert(name.clone(), record).is_some() {
                return Err(Qwen35Error::Gguf(format!(
                    "duplicate safetensors tensor {name:?}"
                )));
            }
        }
        validate_non_overlapping_ranges(&records)?;
        ensure_stable_identity(&identity, &file.metadata()?)?;
        let file_sha256 = if hash_file {
            Some(sha256_descriptor(file)?)
        } else {
            None
        };
        ensure_stable_identity(&identity, &fs::symlink_metadata(path)?)?;
        Ok(Self {
            path: path.to_owned(),
            file_size: before.len(),
            header_bytes,
            data_offset,
            file_sha256,
            records,
            identity,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn tensor(&self, name: &str) -> Option<&SafetensorsRecord> {
        self.records.get(name)
    }

    pub fn tensors(&self) -> impl Iterator<Item = &SafetensorsRecord> {
        self.records.values()
    }

    pub fn read_tensor(&self, name: &str) -> Result<Vec<u8>, Qwen35Error> {
        let record = self
            .tensor(name)
            .ok_or_else(|| Qwen35Error::MissingField(format!("safetensors.tensor.{name}")))?;
        let length: usize = record
            .byte_len()?
            .try_into()
            .map_err(|_| Qwen35Error::SizeOverflow)?;
        let mut file = OpenOptions::new().read(true).open(&self.path)?;
        ensure_stable_identity(&self.identity, &file.metadata()?)?;
        let absolute = self
            .data_offset
            .checked_add(record.data_offsets[0])
            .ok_or(Qwen35Error::SizeOverflow)?;
        file.seek(SeekFrom::Start(absolute))?;
        let mut bytes = vec![0_u8; length];
        file.read_exact(&mut bytes)?;
        ensure_stable_identity(&self.identity, &file.metadata()?)?;
        Ok(bytes)
    }
}

fn expected_dtype_bytes(record: &SafetensorsRecord) -> Result<u64, Qwen35Error> {
    let elements = record.element_count()?;
    let bytes_per_element = match record.dtype.as_str() {
        "BF16" | "F16" | "I16" | "U16" => 2,
        "F32" | "I32" | "U32" => 4,
        "F64" | "I64" | "U64" => 8,
        "BOOL" | "I8" | "U8" | "F8_E4M3" => 1,
        other => {
            return Err(Qwen35Error::invalid(
                format!("safetensors.tensor.{}.dtype", record.name),
                format!("unsupported dtype {other:?}"),
            ));
        }
    };
    elements
        .checked_mul(bytes_per_element)
        .ok_or(Qwen35Error::SizeOverflow)
}

fn validate_non_overlapping_ranges(
    records: &BTreeMap<String, SafetensorsRecord>,
) -> Result<(), Qwen35Error> {
    let mut ranges: Vec<(&str, [u64; 2])> = records
        .values()
        .map(|record| (record.name.as_str(), record.data_offsets))
        .collect();
    ranges.sort_by_key(|(_, offsets)| offsets[0]);
    for pair in ranges.windows(2) {
        if pair[0].1[1] > pair[1].1[0] {
            return Err(Qwen35Error::Gguf(format!(
                "overlapping safetensors ranges {:?} and {:?}",
                pair[0].0, pair[1].0
            )));
        }
    }
    Ok(())
}

fn sha256_descriptor(file: &mut File) -> Result<String, Qwen35Error> {
    file.seek(SeekFrom::Start(0))?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

/// Startup-owned descriptor. Windows denies writes and deletion until this
/// object drops. Header inspection, full hash and later tensor reads use the
/// same file; no per-projection reopen or repeated full-shard hash.
pub struct SafetensorsPinnedShard {
    manifest: SafetensorsManifest,
    file: File,
}

impl SafetensorsPinnedShard {
    pub fn open(path: impl AsRef<Path>, expected_sha256: &str) -> Result<Self, Qwen35Error> {
        let path = path.as_ref();
        if expected_sha256.len()!=64 || !expected_sha256.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
            return Err(Qwen35Error::invalid("safetensors.sha256","expected lowercase SHA256"));
        }
        validate_regular_path(path)?;
        let mut options=OpenOptions::new();
        options.read(true);
        #[cfg(windows)] {
            use std::os::windows::fs::OpenOptionsExt;
            options.share_mode(1); // FILE_SHARE_READ only.
        }
        let mut file=options.open(path)?;
        let manifest=SafetensorsManifest::inspect_file(path,&mut file,true)?;
        if manifest.file_sha256.as_deref()!=Some(expected_sha256) {
            return Err(Qwen35Error::invalid("safetensors.sha256","pinned content differs from expected identity"));
        }
        Ok(Self { manifest, file })
    }

    pub fn manifest(&self) -> &SafetensorsManifest { &self.manifest }

    pub fn read_range_into(&mut self, name: &str, relative_offset: u64, output: &mut [u8]) -> Result<(), Qwen35Error> {
        let record=self.manifest.tensor(name).ok_or_else(|| Qwen35Error::MissingField(name.into()))?;
        let end=relative_offset.checked_add(output.len() as u64).ok_or(Qwen35Error::SizeOverflow)?;
        if end>record.byte_len()? { return Err(Qwen35Error::invalid(name,"tensor read exceeds extent")); }
        ensure_stable_identity(&self.manifest.identity,&self.file.metadata()?)?;
        let offset=self.manifest.data_offset.checked_add(record.data_offsets[0])
            .and_then(|v|v.checked_add(relative_offset)).ok_or(Qwen35Error::SizeOverflow)?;
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.read_exact(output)?;
        ensure_stable_identity(&self.manifest.identity,&self.file.metadata()?)?;
        ensure_path_identity(self.manifest.path(),&self.manifest.identity)?;
        Ok(())
    }
}

fn validate_regular_path(path: &Path) -> Result<(), Qwen35Error> {
    if path.as_os_str().is_empty() {
        return Err(Qwen35Error::Io("empty safetensors path".into()));
    }
    let mut current = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => current.push(prefix.as_os_str()),
            Component::RootDir => current.push(Path::new(std::path::MAIN_SEPARATOR_STR)),
            Component::CurDir => {}
            Component::ParentDir => {
                return Err(Qwen35Error::Io(
                    "parent path components are not allowed".into(),
                ))
            }
            Component::Normal(name) => {
                current.push(name);
                let metadata = fs::symlink_metadata(&current)?;
                if metadata.file_type().is_symlink() || is_reparse_point(&metadata) {
                    return Err(Qwen35Error::Io(format!(
                        "reparse/symlink path component is not allowed: {current:?}"
                    )));
                }
            }
        }
    }
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.is_file() || metadata.file_type().is_symlink() || is_reparse_point(&metadata) {
        return Err(Qwen35Error::Io(
            "safetensors path must be a regular non-reparse file".into(),
        ));
    }
    Ok(())
}

fn ensure_path_identity(path: &Path, identity: &FileIdentity) -> Result<(), Qwen35Error> {
    ensure_stable_identity(identity, &fs::symlink_metadata(path)?)
}

fn ensure_stable_identity(identity: &FileIdentity, metadata: &Metadata) -> Result<(), Qwen35Error> {
    if !identity.matches(metadata) {
        return Err(Qwen35Error::Io(
            "safetensors artifact changed during read".into(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_fixture(path: &Path, header: &str, payload: &[u8]) {
        let mut file = File::create(path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header.as_bytes()).unwrap();
        file.write_all(payload).unwrap();
    }

    #[test]
    fn parses_and_reads_bf16_and_f8_records() {
        let root = std::env::temp_dir().join(format!("augur-qwen-safetensors-{}", std::process::id()));
        let _ = fs::create_dir_all(&root);
        let path = root.join("fixture.safetensors");
        let header = r#"{"a":{"dtype":"F8_E4M3","shape":[4],"data_offsets":[0,4]},"b":{"dtype":"BF16","shape":[2],"data_offsets":[4,8]},"__metadata__":{"format":"pt"}}"#;
        write_fixture(&path, header, &[1, 2, 3, 4, 0, 0, 0, 0]);
        let manifest = SafetensorsManifest::inspect(&path, true).unwrap();
        assert_eq!(manifest.tensor("a").unwrap().byte_len().unwrap(), 4);
        assert_eq!(manifest.read_tensor("b").unwrap(), vec![0, 0, 0, 0]);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn rejects_overlapping_ranges() {
        let root = std::env::temp_dir().join(format!(
            "augur-qwen-safetensors-overlap-{}",
            std::process::id()
        ));
        let _ = fs::create_dir_all(&root);
        let path = root.join("fixture.safetensors");
        let header = r#"{"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"b":{"dtype":"U8","shape":[4],"data_offsets":[3,7]}}"#;
        write_fixture(&path, header, &[0; 7]);
        assert!(SafetensorsManifest::inspect(&path, false).is_err());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn pinned_descriptor_binds_hash_and_bounds_and_releases_lock() {
        let root=std::env::temp_dir().join(format!("augur-qwen-pinned-shard-{}",std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let path=root.join("weights.safetensors");
        write_fixture(&path,r#"{"x":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}"#,&[2,3,5,7]);
        let hash=format!("{:x}",Sha256::digest(fs::read(&path).unwrap()));
        assert!(SafetensorsPinnedShard::open(&path,&"0".repeat(64)).is_err());
        assert!(SafetensorsPinnedShard::open(&path,"not-a-hash").is_err());
        let mut pinned=SafetensorsPinnedShard::open(&path,&hash).unwrap();
        let mut output=[0;2];
        pinned.read_range_into("x",1,&mut output).unwrap();
        assert_eq!(output,[3,5]);
        assert!(pinned.read_range_into("x",3,&mut output).is_err());
        assert!(pinned.read_range_into("x",u64::MAX,&mut output).is_err());
        assert!(pinned.read_range_into("other",0,&mut output).is_err());
        #[cfg(windows)] {
            assert!(OpenOptions::new().write(true).open(&path).is_err());
            assert!(fs::rename(&path,root.join("replacement")).is_err());
        }
        drop(pinned);
        assert!(OpenOptions::new().write(true).open(&path).is_ok());
        fs::remove_dir_all(root).unwrap();
    }
}
