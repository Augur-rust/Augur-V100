//! CPU-only conversion of floating-point GGUF tensors to Q8_0.
//! Generic GGUF conversion preserves metadata/name/shape. The separate dflash
//! module maps the admitted original safetensors profile and target tokenizer.
use augur_core::F16Bits;
use augur_formats::quantize_q8_0_block_ref;
use augur_model_qwen35::{GgufAuditOptions, GgufManifest, GgufPinnedArtifact};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const MAX_HEADER: u64 = 128 * 1024 * 1024;
pub const DEFAULT_CHUNK_BYTES: usize = 1024 * 1024;
static TEMP_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Serialize)]
pub struct TensorPlan {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub input_type: u32,
    pub output_type: u32,
    pub input_bytes: u64,
    pub output_bytes: u64,
    pub output_offset: u64,
    pub action: &'static str,
}

#[derive(Debug, Serialize)]
pub struct Plan {
    pub schema: &'static str,
    pub input_sha256: String,
    pub input_bytes: u64,
    pub output_bytes: u64,
    pub required_additional_free_bytes: u64,
    pub alignment: u64,
    pub data_offset: u64,
    pub chunk_bytes: usize,
    pub tensors: Vec<TensorPlan>,
    #[serde(skip)]
    header: Vec<u8>,
}

fn align(n: u64, alignment: u64) -> Result<u64> {
    Ok(n.checked_add(alignment - 1)
        .ok_or("aligned offset overflow")?
        & !(alignment - 1))
}

fn string(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}

struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}
impl<'a> Cursor<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        let end = self
            .at
            .checked_add(count)
            .ok_or("metadata length overflow")?;
        let value = self.bytes.get(self.at..end).ok_or("truncated metadata")?;
        self.at = end;
        Ok(value)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }
    fn text(&mut self) -> Result<&'a str> {
        let n = usize::try_from(self.u64()?)?;
        Ok(std::str::from_utf8(self.take(n)?)?)
    }
    fn skip(&mut self, kind: u32, depth: usize) -> Result<()> {
        if depth > 16 {
            return Err("metadata nesting exceeds limit".into());
        }
        match kind {
            0 | 1 | 7 => {
                self.take(1)?;
            }
            2 | 3 => {
                self.take(2)?;
            }
            4 | 5 | 6 => {
                self.take(4)?;
            }
            10 | 11 | 12 => {
                self.take(8)?;
            }
            8 => {
                self.text()?;
            }
            9 => {
                let element = self.u32()?;
                let count = self.u64()?;
                if count > 1 << 20 {
                    return Err("metadata array exceeds limit".into());
                }
                for _ in 0..count {
                    self.skip(element, depth + 1)?;
                }
            }
            _ => return Err("unsupported metadata type".into()),
        }
        Ok(())
    }
}

fn metadata_header(input: &Path, manifest: &GgufManifest) -> Result<Vec<u8>> {
    let descriptor_bytes = manifest
        .tensors
        .iter()
        .try_fold(0_u64, |sum, t| {
            sum.checked_add(24 + t.name.len() as u64 + 8 * t.dimensions.len() as u64)
        })
        .ok_or("tensor descriptor length overflow")?;
    let prefix_bytes = manifest
        .header_bytes
        .checked_sub(descriptor_bytes)
        .ok_or("invalid header length")?;
    if !(24..=MAX_HEADER).contains(&prefix_bytes) {
        return Err("header outside bounds".into());
    }
    let mut prefix = vec![0; prefix_bytes as usize];
    File::open(input)?.read_exact(&mut prefix)?;
    let mut cursor = Cursor {
        bytes: &prefix,
        at: 24,
    };
    let mut out = prefix[..24].to_vec();
    let mut count = 0_u64;
    for _ in 0..manifest.metadata_count {
        let start = cursor.at;
        let key = cursor.text()?;
        let kind = cursor.u32()?;
        cursor.skip(kind, 0)?;
        if key != "general.file_type" && key != "general.quantization_version" {
            out.extend_from_slice(&prefix[start..cursor.at]);
            count += 1;
        }
    }
    if cursor.at != prefix.len() {
        return Err("metadata/tensor boundary mismatch".into());
    }
    for (key, value) in [
        ("general.file_type", 7_u32),
        ("general.quantization_version", 2_u32),
    ] {
        string(&mut out, key);
        out.extend_from_slice(&4_u32.to_le_bytes()); // uint32 metadata
        out.extend_from_slice(&value.to_le_bytes());
        count += 1;
    }
    out[16..24].copy_from_slice(&count.to_le_bytes());
    Ok(out)
}

pub fn plan(input: &Path, artifact: &GgufPinnedArtifact, chunk_bytes: usize) -> Result<Plan> {
    if !(4096..=8 * 1024 * 1024).contains(&chunk_bytes) || chunk_bytes % 128 != 0 {
        return Err("chunk bytes must be a multiple of 128 in [4096, 8388608]".into());
    }
    let m = artifact.manifest();
    if m.tensors.is_empty() {
        return Err("input contains no tensors".into());
    }
    let mut header = metadata_header(input, m)?;
    let mut tensors = Vec::with_capacity(m.tensors.len());
    let mut offset = 0_u64;
    for t in &m.tensors {
        if !matches!(t.ggml_type, 0 | 1 | 30) {
            return Err(format!(
                "{}: expected float F32/F16/BF16 input, got type {}",
                t.name, t.ggml_type
            )
            .into());
        }
        let dflash = m.metadata("general.architecture").and_then(|v| v.as_str()) == Some("dflash");
        let dflash_float_role = dflash && (t.dimensions.len() != 2
            || !t.name.ends_with(".weight") || t.name.ends_with("_norm.weight"));
        if dflash_float_role && t.ggml_type != 0 {
            return Err("mapped DFlash norms/convolution bases must already be F32; use dflash-convert for raw HF inputs".into());
        }
        let quantize = !dflash_float_role && t.dimensions.len() >= 2 && t.dimensions[0] % 32 == 0;
        let output_type = if quantize { 8 } else { t.ggml_type };
        let bytes_per_element = if t.ggml_type == 0 { 4 } else { 2 };
        let output_bytes = if quantize {
            (t.byte_len / bytes_per_element / 32)
                .checked_mul(34)
                .ok_or("tensor size overflow")?
        } else {
            t.byte_len
        };
        offset = align(offset, m.alignment)?;
        string(&mut header, &t.name);
        header.extend_from_slice(&(t.dimensions.len() as u32).to_le_bytes());
        for dimension in &t.dimensions {
            header.extend_from_slice(&dimension.to_le_bytes());
        }
        header.extend_from_slice(&output_type.to_le_bytes());
        header.extend_from_slice(&offset.to_le_bytes());
        tensors.push(TensorPlan {
            name: t.name.clone(),
            dimensions: t.dimensions.clone(),
            input_type: t.ggml_type,
            output_type,
            input_bytes: t.byte_len,
            output_bytes,
            output_offset: offset,
            action: if quantize {
                "quantize_q8_0"
            } else {
                "preserve_float"
            },
        });
        offset = offset
            .checked_add(output_bytes)
            .ok_or("output size overflow")?;
    }
    if tensors.iter().all(|t| t.output_type != 8) {
        return Err("no Q8_0-compatible matrix in input".into());
    }
    if header.len() as u64 > MAX_HEADER {
        return Err("output header exceeds limit".into());
    }
    let data_offset = align(header.len() as u64, m.alignment)?;
    header.resize(usize::try_from(data_offset)?, 0);
    let output_bytes = data_offset
        .checked_add(offset)
        .ok_or("output size overflow")?;
    Ok(Plan {
        schema: "quasar-rust-q8-gguf-plan-v1",
        input_sha256: m
            .file_sha256
            .clone()
            .ok_or("missing authenticated input hash")?,
        input_bytes: m.file_size,
        output_bytes,
        required_additional_free_bytes: output_bytes
            .checked_add(MAX_HEADER)
            .ok_or("space overflow")?,
        alignment: m.alignment,
        data_offset,
        chunk_bytes,
        tensors,
        header,
    })
}

fn decode(bytes: &[u8], kind: u32) -> Result<f32> {
    let value = match kind {
        0 => f32::from_le_bytes(bytes.try_into()?),
        1 => F16Bits::from_bits(u16::from_le_bytes(bytes.try_into()?)).to_f32(),
        30 => f32::from_bits(u32::from(u16::from_le_bytes(bytes.try_into()?)) << 16),
        _ => return Err("unsupported floating type".into()),
    };
    if !value.is_finite() {
        return Err("nonfinite input value".into());
    }
    Ok(value)
}

#[cfg(windows)]
fn free_bytes(path: &Path) -> Result<u64> {
    use std::os::windows::ffi::OsStrExt;
    #[link(name = "kernel32")]
    extern "system" {
        fn GetDiskFreeSpaceExW(
            path: *const u16,
            available: *mut u64,
            total: *mut u64,
            free: *mut u64,
        ) -> i32;
    }
    let wide: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
    let mut available = 0_u64;
    // SAFETY: wide owns terminated UTF-16 storage for this synchronous call;
    // available is live, aligned u64 output storage. The two null out-pointers
    // are optional outputs in GetDiskFreeSpaceExW. No pointer escapes the call.
    // This is a space observation, not a reservation against concurrent writes.
    if unsafe {
        GetDiskFreeSpaceExW(
            wide.as_ptr(),
            &mut available,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    } == 0
    {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(available)
}
#[cfg(not(windows))]
fn free_bytes(_path: &Path) -> Result<u64> {
    Err("this release requires Windows disk-space preflight and pinned file locks".into())
}

struct TemporaryOutput {
    path: PathBuf,
    published: bool,
}
impl Drop for TemporaryOutput {
    fn drop(&mut self) {
        if !self.published {
            // Only files created by this converter are removed. No recursive delete.
            let _ = fs::remove_file(self.path.join("model-q8_0.gguf"));
            let _ = fs::remove_file(self.path.join("receipt.json"));
            let _ = fs::remove_dir(&self.path);
        }
    }
}

fn file_sha256(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 128 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

pub fn convert(input: &Path, destination: &Path, chunk_bytes: usize) -> Result<serde_json::Value> {
    let started = std::time::Instant::now();
    if destination.try_exists()? {
        return Err("destination already exists; refusing overwrite".into());
    }
    let parent = destination
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let parent = parent.canonicalize()?;
    let final_name = destination
        .file_name()
        .ok_or("destination needs a directory name")?;
    let destination = parent.join(final_name);
    let mut artifact = GgufPinnedArtifact::open(input, GgufAuditOptions::default())?;
    let p = plan(input, &artifact, chunk_bytes)?;
    let available = free_bytes(&parent)?;
    if available < p.required_additional_free_bytes {
        return Err("insufficient additional free disk space".into());
    }
    let path = parent.join(format!(
        ".augur-q8-partial-{}-{}",
        std::process::id(),
        TEMP_ID.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&path)?;
    let mut temporary = TemporaryOutput {
        path,
        published: false,
    };
    let output = temporary.path.join("model-q8_0.gguf");
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output)?;
    file.write_all(&p.header)?;
    let mut written = p.data_offset;
    let mut reader = artifact.tensor_reader()?;
    let mut errors = Vec::new();
    for t in &p.tensors {
        let mut max_error = 0.0_f64;
        let mut squared_error = 0.0_f64;
        let mut squared_reference = 0.0_f64;
        let mut count_values = 0_u64;
        let target = p.data_offset + t.output_offset;
        let gap = target
            .checked_sub(written)
            .ok_or("tensor offset regressed")?;
        file.write_all(&vec![0; usize::try_from(gap)?])?;
        written = target;
        let width = if t.input_type == 0 { 4 } else { 2 };
        let mut offset = 0_u64;
        while offset < t.input_bytes {
            let count = (t.input_bytes - offset).min(chunk_bytes as u64);
            let raw = reader.read_tensor_slice(&t.name, offset, count)?;
            if t.output_type == 8 {
                let mut packed = Vec::with_capacity(raw.len() / width / 32 * 34);
                for (block_index, block) in raw.chunks_exact(32 * width).enumerate() {
                    let mut values = [0.0_f32; 32];
                    for (value, bytes) in values.iter_mut().zip(block.chunks_exact(width)) {
                        *value = decode(bytes, t.input_type)?;
                    }
                    let q = quantize_q8_0_block_ref(&values).map_err(|e| {
                        format!(
                            "{} at byte {}: {:?}",
                            t.name,
                            offset + (block_index * 32 * width) as u64,
                            e
                        )
                    })?;
                    let scale = q.d.to_f32();
                    for (value, quant) in values.iter().zip(q.qs.iter()) {
                        let error = f64::from(scale * f32::from(*quant)) - f64::from(*value);
                        max_error = max_error.max(error.abs());
                        squared_error += error * error;
                        squared_reference += f64::from(*value).powi(2);
                        count_values += 1;
                    }
                    packed.extend_from_slice(&q.to_le_bytes());
                }
                file.write_all(&packed)?;
                written += packed.len() as u64;
            } else {
                for bytes in raw.chunks_exact(width) {
                    decode(bytes, t.input_type)?;
                }
                file.write_all(&raw)?;
                written += raw.len() as u64;
            }
            offset += count;
        }
        if written != target + t.output_bytes {
            return Err("tensor payload length mismatch".into());
        }
        if count_values > 0 {
            errors.push(serde_json::json!({"tensor": t.name, "elements": count_values,
                "max_abs_error": max_error, "rmse": (squared_error / count_values as f64).sqrt(),
                "relative_l2": if squared_reference > 0.0 { (squared_error / squared_reference).sqrt() } else { 0.0 }}));
        }
    }
    if written != p.output_bytes {
        return Err("total output length mismatch".into());
    }
    file.sync_all()?;
    drop(file);
    let checked = GgufManifest::inspect(&output, GgufAuditOptions::default())?;
    if checked.file_size != p.output_bytes || checked.tensors.len() != p.tensors.len() {
        return Err("output manifest mismatch".into());
    }
    for (actual, expected) in checked.tensors.iter().zip(&p.tensors) {
        if actual.name != expected.name
            || actual.dimensions != expected.dimensions
            || actual.ggml_type != expected.output_type
            || actual.byte_len != expected.output_bytes
            || actual.relative_offset != expected.output_offset
        {
            return Err("output tensor audit mismatch".into());
        }
    }
    let receipt = serde_json::json!({
        "schema": "quasar-rust-q8-gguf-receipt-v1", "status": "PASS_FORMAT_AND_CONVERSION",
        "input": input.canonicalize()?, "input_sha256": p.input_sha256,
        "output": destination.join("model-q8_0.gguf"), "output_sha256": checked.file_sha256,
        "plan": p, "native_rust_quantization": true, "external_quantizer_executed": false,
        "model_semantics_qualified": false, "dflash_acceptance_measured": false,
        "policy": "rank>=2 float matrices with ne0 divisible by32; mapped DFlash F32 norms/convolution bases preserved",
        "supported_input": "already mapped little-endian GGUF v2/v3 F32/F16/BF16",
        "additional_free_bytes_at_start": available,
        "quantization_error": errors, "conversion_wall_seconds": started.elapsed().as_secs_f64(),
        "converter_version": env!("CARGO_PKG_VERSION"),
        "converter_executable_sha256": file_sha256(&std::env::current_exe()?)?,
    });
    let mut receipt_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(temporary.path.join("receipt.json"))?;
    receipt_file.write_all(&serde_json::to_vec_pretty(&receipt)?)?;
    receipt_file.sync_all()?;
    drop(receipt_file);
    if destination.try_exists()? {
        return Err("destination appeared during conversion".into());
    }
    fs::rename(&temporary.path, &destination)?;
    temporary.published = true;
    Ok(receipt)
}

#[cfg(test)]
mod tests;

/// Original DFlash safetensors conversion with target tokenizer metadata.
pub mod dflash;
