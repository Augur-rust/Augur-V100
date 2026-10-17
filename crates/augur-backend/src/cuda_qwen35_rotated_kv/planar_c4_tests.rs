//! Opt-in C4 owner: dense planar Q8 history, causal B1..B8 replay and admission.
//! Synthetic structural checks only; trained-model output/TPS need separate runs.
use super::*;
use crate::{cuda_device_allocation_snapshot, OwnedCudaBuffer};

const CAP: usize = 262144;
const CACHE_ROW: usize = 1088;
const Q_ROW: usize = 6144 * 4;
const KV_ROW: usize = 1024 * 4;

fn upload(stream: &OwnedCudaStream, n: usize, salt: usize) -> OwnedCudaBuffer {
    let raw: Vec<u8> = (0..n)
        .flat_map(|i| (((i * 37 + salt) % 509) as f32 / 177.0 - 1.4).to_le_bytes())
        .collect();
    let mut b = OwnedCudaBuffer::allocate(0, raw.len()).unwrap();
    b.copy_from_host_sync(stream, &raw).unwrap();
    b
}
fn read(stream: &OwnedCudaStream, b: &OwnedCudaBuffer, offset: usize, n: usize) -> Vec<u8> {
    let mut raw = vec![0; n];
    b.view(offset, n)
        .unwrap()
        .copy_to_host_sync(stream, &mut raw)
        .unwrap();
    raw
}
fn difference(a: &[u8], b: &[u8]) -> (f32, f64) {
    assert_eq!(a.len(), b.len());
    let (mut max, mut error, mut norm) = (0f32, 0f64, 0f64);
    for (x, y) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        let x = f32::from_le_bytes(x.try_into().unwrap());
        let y = f32::from_le_bytes(y.try_into().unwrap());
        assert!(x.is_finite() && y.is_finite());
        let delta = x as f64 - y as f64;
        max = max.max((x - y).abs());
        error += delta * delta;
        norm += (x as f64) * (x as f64);
        // Existing planar consumer bound; C4 does not relax it.
        assert!(
            (x - y).abs() <= 0.0001 + 0.002 * x.abs(),
            "C8/C4 {x} vs {y}"
        );
    }
    (max, (error / norm.max(f64::MIN_POSITIVE)).sqrt())
}

#[test]
#[ignore = "exclusive SM70/72SM: planar C4 dense B1..B8, full256k, stream/extent/codec and allocation"]
fn augur_planar_gqa8_c4_owner_boundaries_and_replay() {
    run_boundaries_and_replay(None);
}

#[test]
#[ignore = "exclusive GPU diagnostic; records cross-mode errors without admitting C4"]
fn augur_planar_gqa8_c4_dense_diagnostic() {
    let root = std::path::PathBuf::from(
        std::env::var("AUGUR_C4_DIAGNOSTIC_DIR").expect("diagnostic directory required"),
    );
    std::fs::create_dir(&root).unwrap();
    run_boundaries_and_replay(Some(&root));
}

fn run_boundaries_and_replay(diagnostic: Option<&std::path::Path>) {
    let before = cuda_device_allocation_snapshot();
    {
        let s = OwnedCudaStream::create(0).unwrap();
        let other = OwnedCudaStream::create(0).unwrap();
        let q = upload(&s, 8 * 6144, 101);
        let gate = upload(&s, 8 * 6144, 109);
        let k = upload(&s, 8 * 1024, 117);
        let v = upload(&s, 8 * 1024, 123);
        let mut caches: Vec<_> = (0..4)
            .map(|_| OwnedCudaBuffer::allocate(0, CAP * CACHE_ROW).unwrap())
            .collect();
        for pair in 0..2 {
            // Identical dense finite planar Q8 K/V histories for the two owners.
            let mut seed = vec![0; 256 * CACHE_ROW];
            for (i, head) in seed.chunks_exact_mut(272).enumerate() {
                for j in 0..256 {
                    head[j] = (((i * 97 + j * 37 + pair * 13) % 255) as i16 - 127) as i8 as u8;
                }
                for scale in head[256..].chunks_exact_mut(2) {
                    scale.copy_from_slice(&0x2000u16.to_le_bytes());
                }
            }
            let raw = seed.repeat(CAP / 256);
            caches[pair].copy_from_host_sync(&s, &raw).unwrap();
            caches[2 + pair].copy_from_host_sync(&s, &raw).unwrap();
        }
        let outputs: Vec<_> = (0..4)
            .map(|_| OwnedCudaBuffer::allocate(0, 8 * Q_ROW).unwrap())
            .collect();
        let c8 = CudaQwen35RotatedKvGqa8PlanV1::new_planar(&s).unwrap();
        let c4 = CudaQwen35RotatedKvGqa8PlanV1::new_planar_c4(&s).unwrap();
        let scratch_bytes = CudaQwen35RotatedKvGqa8PlanV1::workspace_bytes_planar(CAP, 8).unwrap();
        let scratch: Vec<_> = (0..2)
            .map(|_| OwnedCudaBuffer::allocate(0, scratch_bytes).unwrap())
            .collect();
        let views = |row: usize, rows: usize, owner: usize| Qwen35RotatedKvViews {
            query: q.view(row * Q_ROW, rows * Q_ROW).unwrap(),
            gate: gate.view(row * Q_ROW, rows * Q_ROW).unwrap(),
            key: k.view(row * KV_ROW, rows * KV_ROW).unwrap(),
            value: v.view(row * KV_ROW, rows * KV_ROW).unwrap(),
            key_cache: caches[owner * 2].view(0, CAP * CACHE_ROW).unwrap(),
            value_cache: caches[owner * 2 + 1].view(0, CAP * CACHE_ROW).unwrap(),
            attention: outputs[owner * 2].view(row * Q_ROW, rows * Q_ROW).unwrap(),
            gated: outputs[owner * 2 + 1]
                .view(row * Q_ROW, rows * Q_ROW)
                .unwrap(),
            workspace: scratch[owner]
                .view(
                    0,
                    CudaQwen35RotatedKvGqa8PlanV1::workspace_bytes_planar(CAP, rows).unwrap(),
                )
                .unwrap(),
        };
        for position in [
            0, 254, 255, 256, 511, 32760, 65535, 131071, 131072, 131073, 196607, 196608, 245760,
            262136, 262140, 262143,
        ] {
            for rows in 1..=8.min(CAP - position) {
                let allocated = cuda_device_allocation_snapshot();
                c8.submit(&s, views(0, rows, 0), CAP, position, rows)
                    .unwrap();
                c4.submit(&s, views(0, rows, 1), CAP, position, rows)
                    .unwrap();
                let expected_attention = read(&s, &outputs[2], 0, rows * Q_ROW);
                let expected_gated = read(&s, &outputs[3], 0, rows * Q_ROW);
                let baseline_gated = read(&s, &outputs[1], 0, rows * Q_ROW);
                let (max, relative_l2) = if let Some(root) = diagnostic {
                    if position == 32760 && rows == 1 {
                        let files = [
                            ("query_rotated.f32", read(&s, &scratch[1], 0, rows * Q_ROW)),
                            ("gate.f32", read(&s, &gate, 0, rows * Q_ROW)),
                            (
                                "key.planar",
                                read(&s, &caches[2], 0, (position + rows) * CACHE_ROW),
                            ),
                            (
                                "value.planar",
                                read(&s, &caches[3], 0, (position + rows) * CACHE_ROW),
                            ),
                            ("c8_attention.f32", read(&s, &outputs[0], 0, rows * Q_ROW)),
                            ("c4_attention.f32", expected_attention.clone()),
                            ("c8_gated.f32", baseline_gated.clone()),
                            ("c4_gated.f32", expected_gated.clone()),
                        ];
                        use std::io::Write;
                        for (name, raw) in files {
                            std::fs::OpenOptions::new()
                                .write(true)
                                .create_new(true)
                                .open(root.join(name))
                                .unwrap()
                                .write_all(&raw)
                                .unwrap();
                        }
                        std::fs::write(root.join("geometry.json"),
                            format!("{{\"position\":{position},\"rows\":{rows},\"capacity\":{CAP},\"query_heads\":24,\"kv_heads\":4,\"head_dim\":256,\"strict_admission\":false}}"))
                            .unwrap();
                    }
                    diagnostic_difference(&baseline_gated, &expected_gated, position, rows)
                } else {
                    difference(&baseline_gated, &expected_gated)
                };
                let end = (position + rows).div_ceil(256) * 256;
                let start = position.saturating_sub(1);
                let expected_cache: Vec<_> = (0..2)
                    .map(|pair| {
                        let bytes = read(
                            &s,
                            &caches[2 + pair],
                            start * CACHE_ROW,
                            (end - start) * CACHE_ROW,
                        );
                        assert_eq!(
                            bytes,
                            read(
                                &s,
                                &caches[pair],
                                start * CACHE_ROW,
                                (end - start) * CACHE_ROW
                            ),
                            "codec {position}/{rows}/{pair}"
                        );
                        let padding = (position + rows - start) * CACHE_ROW;
                        assert!(
                            bytes[padding..].iter().all(|&b| b == 0),
                            "padding {position}/{rows}"
                        );
                        bytes
                    })
                    .collect();
                c4.submit(&s, views(0, rows, 1), CAP, position, rows)
                    .unwrap();
                assert_eq!(
                    expected_attention,
                    read(&s, &outputs[2], 0, rows * Q_ROW),
                    "repeat attention {position}/{rows}"
                );
                assert_eq!(
                    expected_gated,
                    read(&s, &outputs[3], 0, rows * Q_ROW),
                    "repeat gate {position}/{rows}"
                );
                for row in 0..rows {
                    c4.submit(&s, views(row, 1, 1), CAP, position + row, 1)
                        .unwrap();
                }
                assert_eq!(
                    expected_attention,
                    read(&s, &outputs[2], 0, rows * Q_ROW),
                    "B1 attention {position}/{rows}"
                );
                assert_eq!(
                    expected_gated,
                    read(&s, &outputs[3], 0, rows * Q_ROW),
                    "B1 gate {position}/{rows}"
                );
                for pair in 0..2 {
                    assert_eq!(
                        expected_cache[pair],
                        read(
                            &s,
                            &caches[2 + pair],
                            start * CACHE_ROW,
                            (end - start) * CACHE_ROW
                        ),
                        "B1 cache {position}/{rows}/{pair}"
                    );
                }
                let after = cuda_device_allocation_snapshot();
                assert_eq!(
                    allocated.successful_allocations,
                    after.successful_allocations
                );
                assert_eq!(allocated.successful_frees, after.successful_frees);
                eprintln!(
                    "PLANAR_GQA8_C4 position={position} rows={rows} max_abs={max} relative_l2={relative_l2} repeat_exact=true b1_rows_exact=true codec_exact=true padding_zero=true submit_allocations=0"
                );
            }
        }
        let saved: Vec<_> = outputs.iter().map(|b| read(&s, b, 0, 8 * Q_ROW)).collect();
        let saved_cache: Vec<_> = caches[2..]
            .iter()
            .map(|b| read(&s, b, 0, 256 * CACHE_ROW))
            .collect();
        // Use valid views while passing invalid geometry, to test rejection rather than view panics.
        assert!(c4.submit(&s, views(0, 8, 1), CAP, CAP - 7, 8).is_err());
        assert!(c4.submit(&s, views(0, 8, 1), CAP, 0, 9).is_err());
        assert!(c4.submit(&s, views(0, 1, 1), CAP, 0, 0).is_err());
        assert!(c4.submit(&other, views(0, 1, 1), CAP, 0, 1).is_err());
        let mut invalid = views(0, 1, 1);
        invalid.workspace = scratch[1].view(0, scratch_bytes).unwrap();
        assert!(c4.submit(&s, invalid, CAP, 0, 1).is_err());
        invalid = views(0, 1, 1);
        invalid.query = q.view(0, Q_ROW - 4).unwrap();
        assert!(c4.submit(&s, invalid, CAP, 0, 1).is_err());
        invalid = views(0, 1, 1);
        invalid.gated = invalid.attention;
        assert!(c4.submit(&s, invalid, CAP, 0, 1).is_err());
        invalid = views(0, 1, 1);
        invalid.key_cache = caches[2].view(0, CAP * CACHE_ROW - CACHE_ROW).unwrap();
        assert!(c4.submit(&s, invalid, CAP, 0, 1).is_err());
        for (b, saved) in outputs.iter().zip(&saved) {
            assert_eq!(*saved, read(&s, b, 0, 8 * Q_ROW));
        }
        for (b, saved) in caches[2..].iter().zip(&saved_cache) {
            assert_eq!(*saved, read(&s, b, 0, 256 * CACHE_ROW));
        }
        s.synchronize().unwrap();
        other.synchronize().unwrap();
    }
    let after = cuda_device_allocation_snapshot();
    assert_eq!(before.live_bytes, after.live_bytes);
    assert_eq!(before.live_allocations, after.live_allocations);
}

// Diagnostic only: the separate strict test above retains its original bound.
fn diagnostic_difference(a: &[u8], b: &[u8], position: usize, rows: usize) -> (f32, f64) {
    assert_eq!(a.len(), b.len());
    let (mut max, mut error, mut norm, mut violations) = (0f32, 0f64, 0f64, 0usize);
    for (x, y) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        let x = f32::from_le_bytes(x.try_into().unwrap());
        let y = f32::from_le_bytes(y.try_into().unwrap());
        assert!(x.is_finite() && y.is_finite());
        let delta = x as f64 - y as f64;
        max = max.max((x - y).abs());
        error += delta * delta;
        norm += (x as f64) * (x as f64);
        violations += usize::from((x - y).abs() > 0.0001 + 0.002 * x.abs());
    }
    eprintln!("C4_DIAGNOSTIC position={position} rows={rows} strict_bound_violations={violations} elements={} admitted=false", a.len() / 4);
    (max, (error / norm.max(f64::MIN_POSITIVE)).sqrt())
}
