//! File-driven offline replay through the existing B8 FP32 gate and Q8 selector.
//! It emits raw projections/scores/paths, never target labels or acceptance/TPS.
use augur_backend::{
    CudaF32View, CudaQwen35DflashQ8SelectorArgsV1, CudaQwen35DflashQ8SelectorViewsV1,
    CudaQwen35PrefillCublasPlanV1, OwnedCudaBuffer, OwnedCudaStream,
    QWEN35_DFLASH_Q8_SELECTOR_BYTES,
};
use augur_core::PhysicalLayoutId;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    error::Error,
    fs::{self, File, OpenOptions},
    io::{BufReader, BufWriter, Read, Write},
    path::{Component, Path, PathBuf},
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const HIDDEN: usize = 8 * 5120 * 4;
const GATES: usize = 8 * 256 * 4;
const CANDIDATES: usize = 8 * 16 * 4;
const GUARD: usize = 256;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Blob {
    file: String,
    sha256: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema: String,
    rounds: usize,
    gate: Blob,
    predecessor_q8: Blob,
    successor_q8: Blob,
    hidden: Blob,
    candidate_ids: Blob,
    unary_scores: Blob,
    anchors: Blob,
}

fn checked_path(root: &Path, blob: &Blob, expected: usize) -> Result<PathBuf> {
    let relative = Path::new(&blob.file);
    if relative.components().count() != 1
        || !matches!(relative.components().next(), Some(Component::Normal(_)))
    {
        return Err("payload must be a filename within the manifest directory".into());
    }
    let path = root.join(relative);
    if fs::metadata(&path)?.len() != expected as u64 {
        return Err(format!("wrong byte count: {}", blob.file).into());
    }
    let mut file = File::open(&path)?;
    let mut hash = Sha256::new();
    let mut block = [0u8; 65536];
    loop {
        let n = file.read(&mut block)?;
        if n == 0 {
            break;
        }
        hash.update(&block[..n]);
    }
    if format!("{:x}", hash.finalize()) != blob.sha256 {
        return Err(format!("payload SHA mismatch: {}", blob.file).into());
    }
    Ok(path)
}
fn f32_finite(bytes: &[u8]) -> bool {
    bytes.len() % 4 == 0
        && bytes
            .chunks_exact(4)
            .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite())
}
fn validate_round(ids: &[u8], unary: &[u8], hidden: &[u8], anchor: u32) -> Result<()> {
    if ids.len() != CANDIDATES
        || unary.len() != CANDIDATES
        || hidden.len() != HIDDEN
        || anchor >= 248320
        || !f32_finite(unary)
        || !f32_finite(hidden)
    {
        return Err("invalid replay round shape/value".into());
    }
    for row in 0..8 {
        let mut seen = Vec::new();
        let mut previous = f32::INFINITY;
        for k in 0..16 {
            let at = (row * 16 + k) * 4;
            let token = u32::from_le_bytes(ids[at..at + 4].try_into().unwrap());
            let score = f32::from_le_bytes(unary[at..at + 4].try_into().unwrap());
            if token >= 248320 || seen.contains(&token) || score > previous {
                return Err("invalid or unsorted candidate row".into());
            }
            // Native top-k uses token id as the secondary key for equal scores.
            if score == previous && seen.last().is_some_and(|old| *old >= token) {
                return Err("candidate tie order differs from native top-k".into());
            }
            seen.push(token);
            previous = score;
        }
    }
    Ok(())
}
struct Guarded {
    owner: OwnedCudaBuffer,
    bytes: usize,
}
impl Guarded {
    fn new(stream: &OwnedCudaStream, bytes: usize) -> Result<Self> {
        let owner = OwnedCudaBuffer::allocate(0, bytes + 2 * GUARD)?;
        owner.copy_from_host_sync(stream, &vec![0xa5; bytes + 2 * GUARD])?;
        Ok(Self { owner, bytes })
    }
    fn upload(&self, stream: &OwnedCudaStream, payload: &[u8]) -> Result<()> {
        if payload.len() != self.bytes {
            return Err("upload extent".into());
        }
        let mut guarded = vec![0xa5; self.bytes + 2 * GUARD];
        guarded[GUARD..GUARD + self.bytes].copy_from_slice(payload);
        self.owner.copy_from_host_sync(stream, &guarded)?;
        Ok(())
    }
    fn view(&self) -> Result<augur_backend::CudaBufferView> {
        Ok(self.owner.view(GUARD, self.bytes)?)
    }
    fn f32(&self) -> Result<CudaF32View> {
        Ok(CudaF32View::new(
            self.view()?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            self.bytes / 4,
        )?)
    }
    fn download(&self, stream: &OwnedCudaStream) -> Result<Vec<u8>> {
        let mut bytes = vec![0; self.bytes + 2 * GUARD];
        self.owner.copy_to_host_sync(stream, &mut bytes)?;
        if bytes[..GUARD]
            .iter()
            .chain(bytes[GUARD + self.bytes..].iter())
            .any(|b| *b != 0xa5)
        {
            return Err("native replay clobbered output guard".into());
        }
        Ok(bytes[GUARD..GUARD + self.bytes].to_vec())
    }
}
fn output(root: &Path, name: &str) -> Result<BufWriter<File>> {
    Ok(BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(root.join(name))?,
    ))
}
fn execute(manifest_path: &Path, out: &Path) -> Result<()> {
    let manifest_bytes = fs::read(manifest_path)?;
    let m: Manifest = serde_json::from_slice(&manifest_bytes)?;
    if m.schema != "quasar-native-selector-replay-v1" || m.rounds == 0 || m.rounds > 65536 {
        return Err("unsupported replay schema or bounded round count".into());
    }
    let root = manifest_path.parent().ok_or("manifest parent")?;
    let gate = checked_path(root, &m.gate, 256 * 5120 * 4)?;
    let pre = checked_path(root, &m.predecessor_q8, QWEN35_DFLASH_Q8_SELECTOR_BYTES[0])?;
    let post = checked_path(root, &m.successor_q8, QWEN35_DFLASH_Q8_SELECTOR_BYTES[1])?;
    let hidden = checked_path(root, &m.hidden, m.rounds * HIDDEN)?;
    let ids = checked_path(root, &m.candidate_ids, m.rounds * CANDIDATES)?;
    let unary = checked_path(root, &m.unary_scores, m.rounds * CANDIDATES)?;
    let anchors = checked_path(root, &m.anchors, m.rounds * 4)?;
    let gate_bytes = fs::read(&gate)?;
    if !f32_finite(&gate_bytes) {
        return Err("nonfinite FP32 gate".into());
    }
    let mut h = BufReader::new(File::open(hidden)?);
    let mut i = BufReader::new(File::open(ids)?);
    let mut u = BufReader::new(File::open(unary)?);
    let mut a = BufReader::new(File::open(anchors)?);
    fs::create_dir(out)?;
    let mut gout = output(out, "gates.f32le")?;
    let mut sout = output(out, "scores.f32le")?;
    let mut pout = output(out, "paths.u32le")?;
    let allocation_before = augur_backend::cuda_device_allocation_snapshot();
    {
        let stream = OwnedCudaStream::create(0)?;
        let plan = CudaQwen35PrefillCublasPlanV1::new(&stream, 256 * 5120, 5120, 8)?;
        let w = Guarded::new(&stream, gate_bytes.len())?;
        w.upload(&stream, &gate_bytes)?;
        let p = Guarded::new(&stream, QWEN35_DFLASH_Q8_SELECTOR_BYTES[0])?;
        p.upload(&stream, &fs::read(pre)?)?;
        let s = Guarded::new(&stream, QWEN35_DFLASH_Q8_SELECTOR_BYTES[1])?;
        s.upload(&stream, &fs::read(post)?)?;
        let x = Guarded::new(&stream, HIDDEN)?;
        let c = Guarded::new(&stream, CANDIDATES)?;
        let bias = Guarded::new(&stream, CANDIDATES)?;
        let g = Guarded::new(&stream, GATES)?;
        let scores = Guarded::new(&stream, CANDIDATES)?;
        let paths = Guarded::new(&stream, 8 * 4)?;
        let mut hb = vec![0; HIDDEN];
        let mut ib = vec![0; CANDIDATES];
        let mut ub = vec![0; CANDIDATES];
        let mut ab = [0u8; 4];
        for round in 0..m.rounds {
            h.read_exact(&mut hb)?;
            i.read_exact(&mut ib)?;
            u.read_exact(&mut ub)?;
            a.read_exact(&mut ab)?;
            let anchor = u32::from_le_bytes(ab);
            validate_round(&ib, &ub, &hb, anchor)?;
            x.upload(&stream, &hb)?;
            c.upload(&stream, &ib)?;
            bias.upload(&stream, &ub)?;
            plan.submit_dflash_selector_gate_f32(&stream, w.f32()?, x.f32()?, g.f32()?)?;
            let selector = CudaQwen35DflashQ8SelectorArgsV1::bind(
                &stream,
                CudaQwen35DflashQ8SelectorViewsV1 {
                    predecessor: p.view()?,
                    successor: s.view()?,
                    candidate_ids: c.view()?,
                    unary_scores: bias.view()?,
                    gates: g.view()?,
                    transition_scores: scores.view()?,
                    path_ids: paths.view()?,
                },
                anchor,
            )?;
            selector.submit(&stream)?;
            stream.synchronize()?;
            let gb = g.download(&stream)?;
            let sb = scores.download(&stream)?;
            let pb = paths.download(&stream)?;
            if !f32_finite(&gb)
                || !f32_finite(&sb)
                || pb
                    .chunks_exact(4)
                    .any(|b| u32::from_le_bytes(b.try_into().unwrap()) >= 248320)
            {
                return Err(
                    format!("native nonfinite/invalid selector result at round {round}").into(),
                );
            }
            gout.write_all(&gb)?;
            sout.write_all(&sb)?;
            pout.write_all(&pb)?;
        }
    }
    gout.flush()?;
    sout.flush()?;
    pout.flush()?;
    let allocation_after = augur_backend::cuda_device_allocation_snapshot();
    if allocation_before.live_bytes != allocation_after.live_bytes
        || allocation_before.live_allocations != allocation_after.live_allocations
    {
        return Err("native replay allocation did not return to baseline".into());
    }
    let receipt = serde_json::json!({"status":"NATIVE_SELECTOR_REPLAY_COMPLETE","rounds":m.rounds,
        "manifest_sha256":format!("{:x}",Sha256::digest(&manifest_bytes)),"gate_sha256":m.gate.sha256,
        "projection":"existing B8 cublasGemmEx CUBLAS_COMPUTE_32F_PEDANTIC",
        "selector":"existing Q8 greedy; transition scale 1; no phase switch",
        "scope":"raw paths only; known-label prefix and thinking-phase masks must be audited by caller",
        "allocation_returned_to_baseline":true,"GPU_used":true,"target_loaded":false,"acceptance_measured":false,"HTTP_TPS_measured":false});
    let mut complete = output(out, "complete.json")?;
    serde_json::to_writer_pretty(&mut complete, &receipt)?;
    complete.flush()?;
    println!("{}", receipt);
    Ok(())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: augur_selector_replay MANIFEST.json NEW_OUTPUT_DIRECTORY".into());
    }
    execute(Path::new(&args[0]), Path::new(&args[1]))
}
#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let ids: Vec<_> = (0..8)
            .flat_map(|_| (0u32..16).flat_map(u32::to_le_bytes))
            .collect();
        let unary: Vec<_> = (0..8)
            .flat_map(|_| (0..16).flat_map(|i| (16.0 - i as f32).to_le_bytes()))
            .collect();
        (ids, unary, vec![0; HIDDEN])
    }
    #[test]
    fn finite_sorted_payload() {
        let (i, u, h) = fixture();
        assert!(validate_round(&i, &u, &h, 200000).is_ok());
    }
    #[test]
    fn rejects_bad_ids_and_nonfinite_hidden() {
        let (mut i, u, mut h) = fixture();
        i[..4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(validate_round(&i, &u, &h, 0).is_err());
        let (i, _, _) = fixture();
        h[..4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(validate_round(&i, &u, &h, 0).is_err());
    }
    #[test]
    fn rejects_out_of_order_and_duplicates() {
        let (mut i, mut u, h) = fixture();
        i[4..8].copy_from_slice(&0u32.to_le_bytes());
        assert!(validate_round(&i, &u, &h, 0).is_err());
        let (i, _, _) = fixture();
        u[4..8].copy_from_slice(&100f32.to_le_bytes());
        assert!(validate_round(&i, &u, &h, 0).is_err());
    }
}
