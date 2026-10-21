//! Tiny validator fixtures, not a shortcut around full model authentication.
use super::*;
use serde_json::json;
use std::sync::atomic::{AtomicU64, Ordering};
const NAME: &str = "model.language_model.layers.0.mlp.gate_proj";
const GATE: &str = "model.language_model.layers.0.linear_attn.in_proj_a";
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Cleanup(std::path::PathBuf);
impl Drop for Cleanup {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
type Tensor = (String, &'static str, Vec<u64>, Vec<u8>);
fn tensors(w: f32, scale: u8) -> Vec<Tensor> {
    vec![
        (format!("{NAME}.weight"), "U8", vec![2, 8], vec![0x72; 16]),
        (
            format!("{NAME}.weight_scale"),
            "F8_E4M3",
            vec![2, 1],
            vec![0x38, scale],
        ),
        (
            format!("{NAME}.weight_scale_2"),
            "F32",
            vec![],
            w.to_le_bytes().to_vec(),
        ),
        (format!("{GATE}.weight"), "BF16", vec![2, 16], vec![0; 64]),
    ]
}
fn fixture(tensors: Vec<Tensor>) -> (CheckpointArtifact, Cleanup) {
    let root = std::env::temp_dir().join(format!(
        "augur-nerkyor-tiny-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&root).unwrap();
    let cleanup = Cleanup(root.clone());
    let mut header = serde_json::Map::new();
    let mut payload = Vec::new();
    let mut locations = BTreeMap::new();
    for (name, dtype, shape, bytes) in tensors {
        let start = payload.len();
        payload.extend_from_slice(&bytes);
        header.insert(
            name.clone(),
            json!({"dtype":dtype,"shape":shape,"data_offsets":[start,payload.len()]}),
        );
        locations.insert(name, "tiny.safetensors".to_owned());
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(&header);
    file.extend_from_slice(&payload);
    fs::write(root.join("tiny.safetensors"), &file).unwrap();
    let hash = format!("{:x}", Sha256::digest(&file));
    let shard = SafetensorsPinnedShard::open(root.join("tiny.safetensors"), &hash).unwrap();
    let config_bytes = include_bytes!("../tests/fixtures/nerkyor-w4a16-config.json").to_vec();
    let config =
        Qwen35Config::parse_with_profile(&config_bytes, Qwen35CheckpointProfile::NerkyorW4a16)
            .unwrap();
    let linears = [
        (NAME, CheckpointLinearStorage::Nvfp4),
        (GATE, CheckpointLinearStorage::Bf16),
    ]
    .into_iter()
    .map(|(name, storage)| {
        (
            name.into(),
            CheckpointLinearSpec {
                name: name.into(),
                rows: 2,
                cols: 16,
                storage,
            },
        )
    })
    .collect();
    (
        CheckpointArtifact {
            config_bytes,
            config,
            profile: Qwen35CheckpointProfile::NerkyorW4a16,
            identity: CheckpointExpectedIdentity {
                config_sha256: String::new(),
                index_sha256: String::new(),
                shards: [("tiny.safetensors".into(), hash)].into_iter().collect(),
            },
            locations,
            shards: [("tiny.safetensors".into(), shard)].into_iter().collect(),
            linears,
        },
        cleanup,
    )
}

#[test]
fn weight_only_loading_preserves_scales_and_row_ranges_without_input_scale() {
    let (mut a, _cleanup) = fixture(tensors(2.0, 0x40));
    a.validate_linears().unwrap();
    let loaded = a.load_linear_rows(NAME, 1, 1).unwrap();
    let values = loaded.decode_f64().unwrap();
    assert_eq!(values, [4.0, 24.0].repeat(8));
    if let LoadedCheckpointLinear::Nvfp4 {
        global,
        packed,
        scales,
        ..
    } = loaded
    {
        assert_eq!(
            global,
            CheckpointNvfp4Scale::ModelOptW4A16 {
                weight_scale_2: 2.0
            }
        );
        assert_eq!(packed, vec![0x72; 8]);
        assert_eq!(scales, [0x40]);
    } else {
        panic!("wrong storage");
    }
    assert_eq!(
        a.load_linear(GATE).unwrap().decode_f64().unwrap(),
        [0.0; 32]
    );
    for (start, count) in [(0, 0), (1, 2), (2, 1), (usize::MAX, 1)] {
        assert!(a.load_linear_rows(NAME, start, count).is_err());
    }
}

#[test]
fn invented_activation_calibration_and_foreign_scale_namespaces_reject() {
    for suffix in [
        "input_scale",
        "input_scale_2",
        "input_global_scale",
        "activation_scale",
        "activation_scale_2",
        "weight_global_scale",
    ] {
        let mut t = tensors(2.0, 0x40);
        t.push((
            format!("{NAME}.{suffix}"),
            "F32",
            vec![],
            1f32.to_le_bytes().to_vec(),
        ));
        let (a, _cleanup) = fixture(t);
        assert!(a.validate_linears().is_err(), "{suffix}");
    }
    let mut t = tensors(2.0, 0x40);
    t.push((
        format!("{GATE}.weight_scale_2"),
        "F32",
        vec![],
        1f32.to_le_bytes().to_vec(),
    ));
    let (a, _cleanup) = fixture(t);
    assert!(a.validate_linears().is_err());
}

#[test]
fn missing_scalar_wrong_scalar_shape_and_bf16_gate_dtype_reject() {
    let mut t = tensors(2.0, 0x40);
    t.remove(2);
    let (a, _cleanup) = fixture(t);
    assert!(a.validate_linears().is_err());
    let mut t = tensors(2.0, 0x40);
    t[2].2 = vec![1];
    let (a, _cleanup) = fixture(t);
    assert!(a.validate_linears().is_err());
    let mut t = tensors(2.0, 0x40);
    t[3].1 = "F16";
    let (a, _cleanup) = fixture(t);
    assert!(a.validate_linears().is_err());
}

#[test]
fn loaded_nonfinite_global_negative_or_nan_block_scales_fail_closed() {
    for w in [0.0, -1.0, f32::INFINITY, f32::NAN] {
        let (mut a, _cleanup) = fixture(tensors(w, 0x40));
        a.validate_linears().unwrap();
        assert!(a.load_linear(NAME).is_err());
    }
    for s in [127, 128, 255] {
        let (mut a, _cleanup) = fixture(tensors(2.0, s));
        a.validate_linears().unwrap();
        assert!(a.load_linear(NAME).is_err());
    }
}
