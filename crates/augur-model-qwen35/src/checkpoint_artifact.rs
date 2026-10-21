//! Authenticated sharded text target. The loader never rewrites the model and
//! retains scale bytes separately from code bytes. CUDA execution is a caller.
use crate::{
    checkpoint_linear_specs, validate_strict_json, CheckpointFp8Matrix, CheckpointLinearSpec,
    CheckpointLinearStorage, CheckpointNvfp4Matrix, CheckpointNvfp4Scale, Qwen35CheckpointProfile,
    Qwen35Config, Qwen35Error, SafetensorsPinnedShard, SafetensorsRecord,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Component, Path},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointExpectedIdentity {
    pub config_sha256: String,
    pub index_sha256: String,
    pub shards: BTreeMap<String, String>,
}

pub struct CheckpointArtifact {
    config_bytes: Vec<u8>,
    config: Qwen35Config,
    profile: Qwen35CheckpointProfile,
    identity: CheckpointExpectedIdentity,
    locations: BTreeMap<String, String>,
    shards: BTreeMap<String, SafetensorsPinnedShard>,
    linears: BTreeMap<String, CheckpointLinearSpec>,
}

pub enum LoadedCheckpointLinear {
    Nvfp4 {
        spec: CheckpointLinearSpec,
        packed: Vec<u8>,
        scales: Vec<u8>,
        global: CheckpointNvfp4Scale,
    },
    Fp8 {
        spec: CheckpointLinearSpec,
        weights: Vec<u8>,
        weight_scale: f32,
        input_scale: f32,
    },
    Bf16 {
        spec: CheckpointLinearSpec,
        weights: Vec<u8>,
    },
}
impl LoadedCheckpointLinear {
    pub fn spec(&self) -> &CheckpointLinearSpec {
        match self {
            Self::Nvfp4 { spec, .. } | Self::Fp8 { spec, .. } | Self::Bf16 { spec, .. } => spec,
        }
    }
    pub fn decode_f64(&self) -> Result<Vec<f64>, Qwen35Error> {
        let spec = self.spec();
        let mut values = vec![0.0; spec.rows * spec.cols];
        match self {
            Self::Nvfp4 {
                packed,
                scales,
                global,
                ..
            } => CheckpointNvfp4Matrix::new(spec.rows, spec.cols, packed, scales, *global)?
                .decode_f64_into(&mut values)?,
            Self::Fp8 {
                weights,
                weight_scale,
                input_scale,
                ..
            } => CheckpointFp8Matrix::new(
                spec.rows,
                spec.cols,
                weights,
                *weight_scale,
                *input_scale,
            )?
            .decode_f64_into(&mut values)?,
            Self::Bf16 { weights, .. } => {
                for (out, bytes) in values.iter_mut().zip(weights.chunks_exact(2)) {
                    let v = f32::from_bits(
                        u32::from(u16::from_le_bytes(bytes.try_into().unwrap())) << 16,
                    );
                    if !v.is_finite() {
                        return Err(Qwen35Error::invalid(&spec.name, "nonfinite BF16"));
                    }
                    *out = f64::from(v);
                }
            }
        }
        Ok(values)
    }
}

fn checked_metadata(root: &Path, name: &str, expected: &str) -> Result<Vec<u8>, Qwen35Error> {
    let path = root.join(name);
    let meta = fs::symlink_metadata(&path)?;
    if !meta.is_file() || meta.file_type().is_symlink() || meta.len() > 32 * 1024 * 1024 {
        return Err(Qwen35Error::invalid(name, "invalid bounded metadata file"));
    }
    let bytes = fs::read(path)?;
    if format!("{:x}", Sha256::digest(&bytes)) != expected {
        return Err(Qwen35Error::invalid(name, "expected hash mismatch"));
    }
    validate_strict_json(&bytes)?;
    Ok(bytes)
}

fn shard_name(name: &str) -> Result<(), Qwen35Error> {
    let mut parts = Path::new(name).components();
    if !matches!(parts.next(), Some(Component::Normal(_)))
        || parts.next().is_some()
        || !name.ends_with(".safetensors")
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
    {
        return Err(Qwen35Error::invalid(
            "checkpoint.shard",
            "requires a simple safetensors basename",
        ));
    }
    Ok(())
}

impl CheckpointArtifact {
    pub fn open(
        root: impl AsRef<Path>,
        profile: Qwen35CheckpointProfile,
        identity: CheckpointExpectedIdentity,
    ) -> Result<Self, Qwen35Error> {
        Self::open_with_shard_workers(root, profile, identity, 1)
    }

    /// Authenticate independent shards concurrently, retaining the same pinned
    /// descriptors for tensor reads. No hash cache or unauthenticated fast path.
    pub fn open_with_shard_workers(
        root: impl AsRef<Path>, profile: Qwen35CheckpointProfile,
        identity: CheckpointExpectedIdentity, workers: usize,
    ) -> Result<Self, Qwen35Error> {
        if !(1..=8).contains(&workers) {
            return Err(Qwen35Error::invalid("checkpoint.shard_workers", "expected 1..8"));
        }
        if profile == Qwen35CheckpointProfile::OriginalMixed {
            return Err(Qwen35Error::invalid(
                "profile",
                "original mixed uses its existing loader",
            ));
        }
        let root = root.as_ref();
        let config_bytes = checked_metadata(root, "config.json", &identity.config_sha256)?;
        let config = Qwen35Config::parse_with_profile(
            &config_bytes,
            profile,
        )?;
        let index: serde_json::Value = serde_json::from_slice(&checked_metadata(
            root,
            "model.safetensors.index.json",
            &identity.index_sha256,
        )?)
        .map_err(|e| Qwen35Error::Json(e.to_string()))?;
        let map = index["weight_map"]
            .as_object()
            .ok_or_else(|| Qwen35Error::invalid("weight_map", "expected tensor index"))?;
        let mut locations = BTreeMap::new();
        for (name, value) in map {
            let shard = value
                .as_str()
                .ok_or_else(|| Qwen35Error::invalid(name, "non-string shard"))?;
            shard_name(shard)?;
            locations.insert(name.clone(), shard.to_owned());
        }
        let names: BTreeSet<String> = locations.values().cloned().collect();
        if names.is_empty()
            || names != identity.shards.keys().cloned().collect()
            || names
                .iter()
                .map(|s| s.to_ascii_lowercase())
                .collect::<BTreeSet<_>>()
                .len()
                != names.len()
        {
            return Err(Qwen35Error::invalid(
                "checkpoint.shards",
                "hash/index set or case identity mismatch",
            ));
        }
        let shards = open_pinned_shards(root, &identity.shards, workers)?;
        let mut observed = BTreeMap::new();
        for (name, shard) in &shards {
            for record in shard.manifest().tensors() {
                if observed.insert(record.name.clone(), name.clone()).is_some() {
                    return Err(Qwen35Error::invalid(
                        &record.name,
                        "duplicate tensor across shards",
                    ));
                }
            }
        }
        if observed != locations {
            return Err(Qwen35Error::invalid(
                "checkpoint.index",
                "header/index assignment mismatch",
            ));
        }
        let linears = checkpoint_linear_specs(profile)
            .into_iter()
            .map(|s| (s.name.clone(), s))
            .collect();
        let artifact = Self {
            config_bytes,
            config,
            profile,
            identity,
            locations,
            shards,
            linears,
        };
        artifact.validate_linears()?;
        artifact.validate_dense()?;
        Ok(artifact)
    }

    pub fn authenticated_config_bytes(&self) -> &[u8] { &self.config_bytes }
    /// The fixed333 Vision tensors must share one pinned, already-hashed shard.
    /// The local QUASAR artifact satisfies this; do not reopen a second model.
    pub fn vision_weight_map(&self) -> Result<crate::Qwen35VisionWeightMap, Qwen35Error> {
        let profile = crate::Qwen35VisionProfile::from_config_bytes(&self.config_bytes)?;
        let names: BTreeSet<_> = self.locations.iter().filter(|(n,_)| n.starts_with("model.visual."))
            .map(|(_, shard)| shard).collect();
        if names.len() != 1 { return Err(Qwen35Error::invalid("vision.shards", "expected one pinned Vision shard")); }
        crate::Qwen35VisionWeightMap::from_manifest(&profile, self.shards[*names.first().unwrap()].manifest())
    }
    pub fn config(&self) -> &Qwen35Config {
        &self.config
    }
    pub fn identity(&self) -> &CheckpointExpectedIdentity {
        &self.identity
    }
    pub fn linears(&self) -> impl Iterator<Item = &CheckpointLinearSpec> {
        self.linears.values()
    }
    pub fn tensor(&self, name: &str) -> Result<&SafetensorsRecord, Qwen35Error> {
        let file = self
            .locations
            .get(name)
            .ok_or_else(|| Qwen35Error::MissingField(name.into()))?;
        self.shards[file]
            .manifest()
            .tensor(name)
            .ok_or_else(|| Qwen35Error::MissingField(name.into()))
    }
    fn require_tensor(&self, name: &str, dtype: &str, shape: &[u64]) -> Result<(), Qwen35Error> {
        let t = self.tensor(name)?;
        if t.dtype != dtype || t.shape != shape {
            return Err(Qwen35Error::invalid(
                name,
                "checkpoint dtype/shape mismatch",
            ));
        }
        Ok(())
    }
    fn validate_linears(&self) -> Result<(), Qwen35Error> {
        let mut expected_scaled = BTreeSet::new();
        for s in self.linears.values() {
            let suffix = |v: &str| format!("{}.{}", s.name, v);
            let (n, k) = (s.rows as u64, s.cols as u64);
            match s.storage {
                CheckpointLinearStorage::Nvfp4 => {
                    let augur = self.profile == Qwen35CheckpointProfile::QuasarNvfp4;
                    self.require_tensor(
                        &suffix(if augur { "weight_packed" } else { "weight" }),
                        "U8",
                        &[n, k / 2],
                    )?;
                    self.require_tensor(&suffix("weight_scale"), "F8_E4M3", &[n, k / 16])?;
                    let shape = if augur { &[1_u64][..] } else { &[][..] };
                    self.require_tensor(
                        &suffix(if augur {
                            "weight_global_scale"
                        } else {
                            "weight_scale_2"
                        }),
                        "F32",
                        shape,
                    )?;
                    if self.profile != Qwen35CheckpointProfile::NerkyorW4a16 {
                        self.require_tensor(
                            &suffix(if augur { "input_global_scale" } else { "input_scale" }),
                            "F32",
                            shape,
                        )?;
                    }
                    expected_scaled.insert(suffix("weight_scale"));
                }
                CheckpointLinearStorage::Fp8 => {
                    self.require_tensor(&suffix("weight"), "F8_E4M3", &[n, k])?;
                    self.require_tensor(&suffix("weight_scale"), "F32", &[])?;
                    self.require_tensor(&suffix("input_scale"), "F32", &[])?;
                    expected_scaled.insert(suffix("weight_scale"));
                }
                CheckpointLinearStorage::Bf16 => {
                    self.require_tensor(&suffix("weight"), "BF16", &[n, k])?
                }
            }
        }
        let actual: BTreeSet<String> = self
            .locations
            .keys()
            .filter(|n| n.ends_with(".weight_scale"))
            .cloned()
            .collect();
        if actual != expected_scaled {
            return Err(Qwen35Error::invalid(
                "checkpoint.scales",
                "unexpected quantized linear namespace",
            ));
        }
        if self.profile == Qwen35CheckpointProfile::NerkyorW4a16 {
            let expected: BTreeSet<_> = expected_scaled.iter().map(|n| format!("{n}_2")).collect();
            let actual: BTreeSet<_> = self.locations.keys()
                .filter(|n| n.ends_with(".weight_scale_2")).cloned().collect();
            if actual != expected || self.locations.keys().any(|n| {
                [".input_scale", ".input_scale_2", ".input_global_scale", ".activation_scale",
                 ".activation_scale_2", ".weight_global_scale", ".weight_packed"]
                    .iter().any(|suffix| n.ends_with(suffix))
            }) {
                return Err(Qwen35Error::invalid("checkpoint.scales",
                    "W4A16 requires weight-only scale namespaces, no activation calibration"));
            }
        }
        Ok(())
    }
    pub fn read_tensor_range_into(
        &mut self,
        name: &str,
        offset: u64,
        output: &mut [u8],
    ) -> Result<(), Qwen35Error> {
        let file = self
            .locations
            .get(name)
            .ok_or_else(|| Qwen35Error::MissingField(name.into()))?;
        self.shards
            .get_mut(file)
            .unwrap()
            .read_range_into(name, offset, output)
    }
    fn scalar(&mut self, name: &str) -> Result<f32, Qwen35Error> {
        if self.tensor(name)?.byte_len()? != 4 {
            return Err(Qwen35Error::invalid(name, "not one F32 scalar"));
        }
        let mut bytes = [0; 4];
        self.read_tensor_range_into(name, 0, &mut bytes)?;
        let v = f32::from_le_bytes(bytes);
        if !v.is_finite() || v <= 0.0 {
            return Err(Qwen35Error::invalid(name, "nonpositive/nonfinite scalar"));
        }
        Ok(v)
    }
    fn rows(&mut self, name: &str, start: usize, count: usize) -> Result<Vec<u8>, Qwen35Error> {
        let t = self.tensor(name)?;
        let total = *t
            .shape
            .first()
            .ok_or_else(|| Qwen35Error::invalid(name, "scalar has no rows"))?;
        if count == 0
            || (start as u64)
                .checked_add(count as u64)
                .ok_or(Qwen35Error::SizeOverflow)?
                > total
            || total == 0
        {
            return Err(Qwen35Error::invalid(name, "row range exceeds tensor"));
        }
        let stride = t.byte_len()? / total;
        let length = stride
            .checked_mul(count as u64)
            .ok_or(Qwen35Error::SizeOverflow)?;
        let mut data = vec![0; usize::try_from(length).map_err(|_| Qwen35Error::SizeOverflow)?];
        self.read_tensor_range_into(name, start as u64 * stride, &mut data)?;
        Ok(data)
    }
    pub fn load_linear(&mut self, name: &str) -> Result<LoadedCheckpointLinear, Qwen35Error> {
        let rows = self
            .linears
            .get(name)
            .ok_or_else(|| Qwen35Error::MissingField(name.into()))?
            .rows;
        self.load_linear_rows(name, 0, rows)
    }
    pub fn load_linear_rows(
        &mut self,
        name: &str,
        start: usize,
        count: usize,
    ) -> Result<LoadedCheckpointLinear, Qwen35Error> {
        let mut spec = self
            .linears
            .get(name)
            .ok_or_else(|| Qwen35Error::MissingField(name.into()))?
            .clone();
        let suffix = |v: &str| format!("{name}.{v}");
        spec.rows = count;
        Ok(match spec.storage {
            CheckpointLinearStorage::Nvfp4 => {
                let q = self.profile == Qwen35CheckpointProfile::QuasarNvfp4;
                let packed = self.rows(
                    &suffix(if q { "weight_packed" } else { "weight" }),
                    start,
                    count,
                )?;
                let scales = self.rows(&suffix("weight_scale"), start, count)?;
                let w = self.scalar(&suffix(if q {
                    "weight_global_scale"
                } else {
                    "weight_scale_2"
                }))?;
                let global = if self.profile == Qwen35CheckpointProfile::NerkyorW4a16 {
                    CheckpointNvfp4Scale::ModelOptW4A16 { weight_scale_2: w }
                } else if q {
                    let x = self.scalar(&suffix("input_global_scale"))?;
                    CheckpointNvfp4Scale::CompressedTensors {
                        weight_global_scale: w,
                        input_global_scale: x,
                    }
                } else {
                    CheckpointNvfp4Scale::ModelOpt {
                        weight_scale_2: w,
                        input_scale: self.scalar(&suffix("input_scale"))?,
                    }
                };
                CheckpointNvfp4Matrix::new(count, spec.cols, &packed, &scales, global)?;
                LoadedCheckpointLinear::Nvfp4 {
                    spec,
                    packed,
                    scales,
                    global,
                }
            }
            CheckpointLinearStorage::Fp8 => {
                let weights = self.rows(&suffix("weight"), start, count)?;
                let weight_scale = self.scalar(&suffix("weight_scale"))?;
                let input_scale = self.scalar(&suffix("input_scale"))?;
                CheckpointFp8Matrix::new(count, spec.cols, &weights, weight_scale, input_scale)?;
                LoadedCheckpointLinear::Fp8 {
                    spec,
                    weights,
                    weight_scale,
                    input_scale,
                }
            }
            CheckpointLinearStorage::Bf16 => {
                let weights = self.rows(&suffix("weight"), start, count)?;
                LoadedCheckpointLinear::Bf16 { spec, weights }
            }
        })
    }
}

fn open_pinned_shards(
    root: &Path, expected: &BTreeMap<String, String>, workers: usize,
) -> Result<BTreeMap<String, SafetensorsPinnedShard>, Qwen35Error> {
    if !(1..=8).contains(&workers) {
        return Err(Qwen35Error::invalid("checkpoint.shard_workers", "expected 1..8"));
    }
    for name in expected.keys() { shard_name(name)?; }
    if workers == 1 {
        return expected.iter().map(|(name, hash)| {
            Ok((name.clone(), SafetensorsPinnedShard::open(root.join(name), hash)?))
        }).collect();
    }
    let jobs: Vec<_> = expected.iter().collect();
    let mut shards = BTreeMap::new();
    for batch in jobs.chunks(workers) {
        // Join every worker before returning any error: partially opened
        // descriptors are dropped, no detached work can outlive a failed load.
        let results = std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for &(name, hash) in batch {
                handles.push(std::thread::Builder::new()
                    .name("augur-shard-auth".into())
                    .spawn_scoped(scope, move || {
                        Ok((name.clone(), SafetensorsPinnedShard::open(root.join(name), hash)?))
                    }));
            }
            handles.into_iter().map(|h| match h {
                Ok(h) => h.join().unwrap_or_else(|_| Err(Qwen35Error::invalid(
                    "checkpoint.shard_workers", "authentication worker panicked"))),
                Err(e) => Err(e.into()),
            }).collect::<Vec<Result<_, Qwen35Error>>>()
        });
        for result in results {
            let (name, shard) = result?;
            shards.insert(name, shard);
        }
    }
    Ok(shards)
}

#[cfg(test)]
mod parallel_shard_tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    struct Fixture(std::path::PathBuf);
    impl Drop for Fixture { fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); } }
    fn fixture() -> (Fixture, BTreeMap<String,String>) {
        let path = std::env::temp_dir().join(format!("augur-shard-auth-{}-{}",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)));
        fs::create_dir(&path).unwrap();
        let mut hashes=BTreeMap::new();
        for i in 0..5 {
            let name=format!("shard-{i}.safetensors");
            let header=format!("{{\"tensor{i}\":{{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[0,4]}}}}");
            let mut bytes=(header.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(header.as_bytes());bytes.extend_from_slice(&[i as u8,0,1,0]);
            fs::write(path.join(&name),&bytes).unwrap();hashes.insert(name,format!("{:x}",Sha256::digest(&bytes)));
        }
        (Fixture(path),hashes)
    }
    #[test]
    fn parallel_pinned_shards_match_serial_and_retained_reads() {
        let (f, hashes)=fixture();
        for workers in [1,2,5,8] {
            let mut shards=open_pinned_shards(&f.0,&hashes,workers).unwrap();
            assert_eq!(shards.len(),5);
            for (i, (_,shard)) in shards.iter_mut().enumerate() {
                let mut bytes=[0;4];shard.read_range_into(&format!("tensor{i}"),0,&mut bytes).unwrap();
                assert_eq!(bytes,[i as u8,0,1,0]);
            }
        }
    }
    #[test]
    fn parallel_auth_fails_closed_and_releases_all_handles() {
        let (f, mut hashes)=fixture();
        let name="shard-2.safetensors";let original=hashes[name].clone();hashes.insert(name.into(),"0".repeat(64));
        for workers in [1,2,5,8] {
            assert!(open_pinned_shards(&f.0,&hashes,workers).is_err());
            // Windows share-read handles forbid this write if a worker leaked.
            let path=f.0.join("shard-0.safetensors");let bytes=fs::read(&path).unwrap();fs::write(&path,&bytes).unwrap();
        }
        hashes.insert(name.into(),original);
        assert!(open_pinned_shards(&f.0,&hashes,5).is_ok());
        for workers in [0,9,usize::MAX] {assert!(open_pinned_shards(&f.0,&hashes,workers).is_err());}
        hashes.insert("../escape.safetensors".into(),"0".repeat(64));
        assert!(open_pinned_shards(&f.0,&hashes,5).is_err());
    }
}

#[cfg(test)]
#[path = "checkpoint_artifact_nerkyor_tests.rs"]
mod nerkyor_tests;
