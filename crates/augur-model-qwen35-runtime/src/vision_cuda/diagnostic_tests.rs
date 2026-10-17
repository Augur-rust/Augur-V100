use super::*;
use augur_backend::cuda_device_allocation_snapshot;
use augur_model_qwen35::{prepare_qwen35_aligned_rgb8_f16, Qwen35ImageGrid};
use serde_json::{json, Value};
use std::{fs, io::Write, path::PathBuf};

fn save(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = fs::OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    Ok(())
}

#[test]
#[ignore = "real authenticated standalone Vision tensor trace; not performance evidence"]
fn augur_encoder_trace() -> Result<()> {
    let fixture = PathBuf::from(std::env::var_os("AUGUR_VISION_FIXTURE").ok_or("fixture")?);
    let out = PathBuf::from(std::env::var_os("AUGUR_VISION_OUTPUT").ok_or("output")?);
    fs::create_dir(&out)?;
    let value: Value = serde_json::from_slice(&fs::read(&fixture)?)?;
    let root = PathBuf::from(value["model_root"].as_str().ok_or("root")?);
    let admitted = crate::augur_session::manifest::load(&root,
        Path::new(value["manifest"].as_str().ok_or("manifest")?),
        value["manifest_sha256"].as_str().ok_or("manifest hash")?).map_err(|e| e.to_string())?;
    let mut reader = augur_model_qwen35::CheckpointArtifact::open_with_shard_workers(
        &root, augur_model_qwen35::Qwen35CheckpointProfile::QuasarNvfp4,
        admitted.manifest().checkpoint.clone(), 5)?;
    let host = Qwen35VisionHostWeights::from_checkpoint(&mut reader)?;
    let sample = value["cases"].as_array().ok_or("cases")?.iter()
        .find(|v| v["name"] == "digits").ok_or("digits case")?;
    let image = &sample["images"][0];
    let rgb = fs::read(image["rgb_path"].as_str().ok_or("rgb")?)?;
    assert_eq!(format!("{:x}", Sha256::digest(&rgb)), image["sha256"].as_str().unwrap());
    let grid = Qwen35ImageGrid::from_aligned_pixels(image["height"].as_u64().unwrap() as usize,
        image["width"].as_u64().unwrap() as usize)?;
    let prepared = prepare_qwen35_aligned_rgb8_f16(&rgb, grid, 32 * 1024 * 1024)?;
    save(&out.join("pixels.f16"), &prepared.pixel_values_f16_le)?;
    let geometry = Qwen35VisionGeometry::new(grid, 4 * 1024 * 1024)?;
    save(&out.join("geometry.json"), &serde_json::to_vec(&geometry)?)?;
    let before = cuda_device_allocation_snapshot();
    let mut encoder = host.upload(0)?;
    let result = encoder.encode_traced(&prepared, || false, |stage, stream, view| {
        let mut bytes = vec![0; view.len_bytes()];
        view.copy_to_host_sync(stream, &mut bytes)?;
        save(&out.join(format!("{stage}.f32")), &bytes)
    })?;
    encoder.close()?;
    let after = cuda_device_allocation_snapshot();
    assert_eq!(before.live_bytes, after.live_bytes);
    assert_eq!(before.live_allocations, after.live_allocations);
    save(&out.join("complete.json"), &serde_json::to_vec_pretty(&json!({
        "status":"DIAGNOSTIC_TENSORS_ONLY", "host":host.info(), "result":result,
        "fixture_sha256":format!("{:x}", Sha256::digest(fs::read(fixture)?)),
        "allocation_closure":true, "performance_qualified":false, "product_ready":false
    }))?)?;
    Ok(())
}
