//! CPU-only real-file ABBA for hash/reader setup, not GPU startup qualification.
use augur_model_qwen35::{GgufAuditOptions, GgufManifest, GgufPinnedArtifact};
use serde_json::json;
use std::{error::Error, path::PathBuf, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 2 { return Err("usage: pinned_startup_bill GGUF FRESH_ROOT".into()); }
    let model = PathBuf::from(&args[0]);
    let root = PathBuf::from(&args[1]);
    if !root.is_absolute() || root.exists() { return Err("fresh absolute root required".into()); }
    std::fs::create_dir(&root)?;
    let mut samples = Vec::new();
    let mut expected = None;
    let mut expected_probe = None;
    for mode in ["legacy", "pinned", "pinned", "legacy"] {
        let started = Instant::now();
        let (digest, probe, inspected_ms, setup_ms) = if mode == "legacy" {
            let manifest = GgufManifest::inspect(&model, GgufAuditOptions::default())?;
            let inspected_ms = started.elapsed().as_secs_f64() * 1000.0;
            let mut reader = manifest.open_tensor_reader(&model)?;
            let setup_ms = started.elapsed().as_secs_f64() * 1000.0;
            let probe = reader.read_tensor_slice("token_embd.weight", 0, 32)?;
            (manifest.file_sha256.clone(), probe, inspected_ms, setup_ms)
        } else {
            let mut artifact = GgufPinnedArtifact::open(&model, GgufAuditOptions::default())?;
            let inspected_ms = started.elapsed().as_secs_f64() * 1000.0;
            let digest = artifact.manifest().file_sha256.clone();
            let mut reader = artifact.tensor_reader()?;
            let setup_ms = started.elapsed().as_secs_f64() * 1000.0;
            let probe = reader.read_tensor_slice("token_embd.weight", 0, 32)?;
            (digest, probe, inspected_ms, setup_ms)
        };
        if digest.is_none() { return Err("missing full digest".into()); }
        if let Some(ref e) = expected { if e != &digest { return Err("model digest drift".into()); } }
        else { expected = Some(digest.clone()); }
        if let Some(ref e) = expected_probe { if e != &probe { return Err("payload drift".into()); } }
        else { expected_probe = Some(probe.clone()); }
        samples.push(json!({"mode":mode,"full_file_sha256":digest,"inspect_ms":inspected_ms,
            "reader_rebind_ms":setup_ms-inspected_ms,"setup_ms":setup_ms,"wall_with_probe_ms":started.elapsed().as_secs_f64()*1000.0,
            "whole_file_hash_passes":if mode=="legacy" {2}else{1},"payload_probe_bytes":probe}));
        println!("{mode}: {setup_ms:.3} ms");
    }
    let report = json!({"status":"PASS_REAL_GGUF_READER_SETUP_ABBA_CPU_COMPONENT_ONLY", "model":model,
        "samples":samples,"cold_os_cache_controlled":false,"full_model_upload_executed":false,
        "gpu_started":false,"product_ready":false,"qualified":false,"visible_tps":false,
        "limits":["Only authenticated hash/reader setup and a32-byte payload probe; no target/DFlash initialization or H2D.",
            "OS caching is uncontrolled. These timings do not predict complete service startup savings.",
            "No claim that startup now matches or beats llama."]});
    std::fs::write(root.join("bill.json"), serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}
