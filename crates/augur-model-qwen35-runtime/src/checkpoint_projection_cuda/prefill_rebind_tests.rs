//! Isolated QKV/Z projection gate, NOT a complete prefill implementation.
use super::*;
use crate::checkpoint_ffn_cuda::CheckpointFfnContext;
use augur_backend::cuda_device_allocation_snapshot;
use augur_backend::cuda_qwen35_prefill_cublas::{
    CudaQwen35PrefillCublasPlanV1, CudaQwen35QpnRetainedProjectionV1,
};
use augur_model_qwen35::{CheckpointArtifact, CheckpointExpectedIdentity, Qwen35CheckpointProfile};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{path::PathBuf, time::Instant};
const K: usize = 5120;
const B: usize = 8;
const N: [usize; 2] = [10240, 6144];
type Pair = [CudaQwen35QpnRetainedProjectionV1; 2];
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn checked_file(v: &Value) -> Vec<u8> {
    let b = std::fs::read(v["path"].as_str().unwrap()).unwrap();
    assert_eq!(hash(&b), v["sha256"].as_str().unwrap());
    b
}
fn read(stream: &OwnedCudaStream, b: &OwnedCudaBuffer) -> Vec<u8> {
    let mut v = vec![0; b.len_bytes()];
    b.copy_to_host_sync(stream, &mut v).unwrap();
    v
}
fn buffer(stream: &OwnedCudaStream, n: usize) -> Rc<OwnedCudaBuffer> {
    Rc::new(OwnedCudaBuffer::allocate(stream.device_ordinal(), n).unwrap())
}
fn submit(pair: &Pair) {
    pair[0].submit_shared_input_pair(&pair[1]).unwrap();
}
fn math(actual: &[u8], reference: &[f64]) -> Value {
    assert_eq!(actual.len(), reference.len() * 4);
    let (mut d2, mut r2, mut worst) = (0.0, 0.0, 0.0_f64);
    for (a, &r) in actual.chunks_exact(4).zip(reference) {
        let a = f32::from_le_bytes(a.try_into().unwrap()) as f64;
        assert!(a.is_finite() && r.is_finite());
        let d = (a - r).abs();
        d2 += d * d;
        r2 += r * r;
        worst = worst.max(d / (1e-3 + 1e-3 * r.abs()));
    }
    let rel = (d2 / r2.max(1e-300)).sqrt();
    assert!(
        rel <= 1e-3 && worst <= 1.0,
        "F64 projection gate rel={rel} worst={worst}"
    );
    json!({"relative_l2":rel,"worst_abs_plus_rel_budget_ratio":worst,"rtol":1e-3,"atol":1e-3})
}
fn elapsed_us(stream: &OwnedCudaStream, mut f: impl FnMut()) -> f64 {
    stream.synchronize().unwrap();
    let start = Instant::now();
    for _ in 0..40 {
        f();
    }
    stream.synchronize().unwrap();
    start.elapsed().as_secs_f64() * 1e6 / 40.0
}
fn median(v: &[f64]) -> f64 {
    let mut v = v.to_vec();
    v.sort_by(f64::total_cmp);
    (v[1] + v[2]) / 2.0
}
#[test]
#[ignore = "exclusive SM70; authenticated QUASAR weights and F64 projection references"]
fn real_augur_gdn_qpn_prefill_rebind_component() {
    let path = PathBuf::from(std::env::var_os("AUGUR_QPN_PREFIX_REFERENCE").expect("reference"));
    let bytes = std::fs::read(&path).unwrap();
    let fixture: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        fixture["status"],
        "PASS_CPU_COMPLETE_LINEAR_LAYER_REFERENCE_ONLY"
    );
    let output = PathBuf::from(std::env::var_os("AUGUR_QPN_PREFIX_OUTPUT").expect("output"));
    let baseline = cuda_device_allocation_snapshot();
    let mut results = Vec::new();
    for case in fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["profile"] == "quasar")
    {
        let audit_bytes = std::fs::read(case["audit_path"].as_str().unwrap()).unwrap();
        assert_eq!(hash(&audit_bytes), case["audit_sha256"].as_str().unwrap());
        let a: Value = serde_json::from_slice(&audit_bytes).unwrap();
        let expected = CheckpointExpectedIdentity {
            config_sha256: a["metadata"]["config.json"]["sha256"]
                .as_str()
                .unwrap()
                .into(),
            index_sha256: a["metadata"]["model.safetensors.index.json"]["sha256"]
                .as_str()
                .unwrap()
                .into(),
            shards: a["shards"]
                .as_array()
                .unwrap()
                .iter()
                .map(|s| {
                    (
                        s["name"].as_str().unwrap().into(),
                        s["sha256"].as_str().unwrap().into(),
                    )
                })
                .collect(),
        };
        let mut reader = CheckpointArtifact::open(
            a["model_dir"].as_str().unwrap(),
            Qwen35CheckpointProfile::QuasarNvfp4,
            expected,
        )
        .unwrap();
        let layer = case["layer"].as_u64().unwrap();
        assert_eq!(case["tokens"], 4);
        // These are the existing independent CPU layer-fixture normalized
        // embedding inputs, not downstream activations from a full target.
        // Widths5..8 explicitly repeat the four fixture rows; no hidden claim
        // that this is an eight-token real target prefill/semantic test.
        let norm = checked_file(&case["stages"]["normalized"]);
        let values: Vec<f32> = norm
            .chunks_exact(8)
            .map(|v| f64::from_le_bytes(v.try_into().unwrap()) as f32)
            .collect();
        assert_eq!(values.len(), 4 * K);
        let repeated: Vec<u8> = (0..B)
            .flat_map(|r| {
                values[(r % 4) * K..(r % 4 + 1) * K]
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
            })
            .collect();
        let reversed: Vec<u8> = (0..B)
            .rev()
            .flat_map(|r| {
                values[(r % 4) * K..(r % 4 + 1) * K]
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
            })
            .collect();
        let stream = Rc::new(OwnedCudaStream::create(0).unwrap());
        let context = CheckpointFfnContext::new(
            Rc::clone(&stream),
            CheckpointProjectionPolicy::AugurQpnAllWorkspaceShapeSmallPairF64,
        )
        .unwrap();
        let x = buffer(&stream, B * K * 4);
        let y: [Rc<OwnedCudaBuffer>; 2] = std::array::from_fn(|i| buffer(&stream, B * N[i] * 4));
        let mut sources = Vec::new();
        for (i, suffix) in ["in_proj_qkv", "in_proj_z"].iter().enumerate() {
            let weight = context
                .upload_projection(
                    Rc::clone(&stream),
                    reader
                        .load_linear(&format!(
                            "model.language_model.layers.{layer}.linear_attn.{suffix}"
                        ))
                        .unwrap(),
                )
                .unwrap();
            sources.push(
                weight
                    .bind(Rc::clone(&x), 0, Rc::clone(&y[i]), 0)
                    .unwrap()
                    .retained_qpn()
                    .unwrap(),
            );
        }
        let before_arena = cuda_device_allocation_snapshot();
        // Bounded F64 partials and hi/lo stage, not a decoded weight copy.
        let plan = Rc::new(
            CudaQwen35PrefillCublasPlanV1::new(&stream, N[0] * B * 16 * 8, K * 2, B).unwrap(),
        );
        let arena_bytes = cuda_device_allocation_snapshot().live_bytes - before_arena.live_bytes;
        let rebind = |tokens: usize, row: usize| -> Pair {
            std::array::from_fn(|i| {
                sources[i]
                    .rebind_shared_weight(
                        Rc::clone(&plan),
                        Rc::clone(&stream),
                        Rc::clone(&x),
                        row * K * 4,
                        Rc::clone(&y[i]),
                        row * N[i] * 4,
                        tokens,
                    )
                    .unwrap()
            })
        };
        let prebind = cuda_device_allocation_snapshot();
        let scalar: Vec<Pair> = (0..B).map(|r| rebind(1, r)).collect();
        let widths: Vec<Pair> = (1..=B).map(|b| rebind(b, 0)).collect();
        assert_eq!(
            prebind.successful_allocations,
            cuda_device_allocation_snapshot().successful_allocations
        );
        for bad in [0, 9, usize::MAX] {
            assert!(sources[0]
                .rebind_shared_weight(
                    Rc::clone(&plan),
                    Rc::clone(&stream),
                    Rc::clone(&x),
                    0,
                    Rc::clone(&y[0]),
                    0,
                    bad
                )
                .is_err());
        }
        assert!(sources[0]
            .rebind_shared_weight(
                Rc::clone(&plan),
                Rc::clone(&stream),
                Rc::clone(&x),
                1,
                Rc::clone(&y[0]),
                0,
                1
            )
            .is_err());
        assert!(sources[0]
            .rebind_shared_weight(
                Rc::clone(&plan),
                Rc::clone(&stream),
                Rc::clone(&y[0]),
                0,
                Rc::clone(&y[0]),
                0,
                1
            )
            .is_err());
        let foreign = Rc::new(OwnedCudaStream::create(0).unwrap());
        assert!(sources[0]
            .rebind_shared_weight(
                Rc::clone(&plan),
                foreign,
                Rc::clone(&x),
                0,
                Rc::clone(&y[0]),
                0,
                1
            )
            .is_err());
        let warm = cuda_device_allocation_snapshot();
        let mut math_results = Vec::new();
        let mut expected_reversed = Vec::new();
        for (pass, input) in [&repeated, &reversed].into_iter().enumerate() {
            x.copy_from_host_sync(&stream, input).unwrap();
            for pair in &scalar {
                submit(pair);
            }
            let reference: Vec<_> = y.iter().map(|b| read(&stream, b)).collect();
            if pass == 0 {
                for (i, name) in ["qkv", "z"].iter().enumerate() {
                    let f = checked_file(&case["stages"][name]);
                    let oracle: Vec<_> = f
                        .chunks_exact(8)
                        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
                        .collect();
                    math_results.push(
                        json!({"projection":name,"metric":math(&reference[i][..4*N[i]*4],&oracle)}),
                    );
                }
            }
            for (index, pair) in widths.iter().enumerate() {
                for b in &y {
                    b.copy_from_host_sync(&stream, &vec![0x7f; b.len_bytes()])
                        .unwrap();
                }
                submit(pair);
                for i in 0..2 {
                    let got = read(&stream, &y[i]);
                    assert_eq!(
                        got[..(index + 1) * N[i] * 4],
                        reference[i][..(index + 1) * N[i] * 4],
                        "layer{layer} projection{i} B{} pass{pass}",
                        index + 1
                    );
                }
            }
            expected_reversed = reference;
        }
        let mut timing = Vec::new();
        for width in 1..=B {
            for _ in 0..4 {
                for pair in &scalar[..width] {
                    submit(pair);
                }
                submit(&widths[width - 1]);
            }
            let mut a = Vec::new();
            let mut b = Vec::new();
            // AB/BA/BA/AB, no diagnostic readback inside timing boundaries.
            for reverse in [false, true, true, false] {
                let mut run = |batch: bool| {
                    let us = elapsed_us(&stream, || {
                        if batch {
                            submit(&widths[width - 1]);
                        } else {
                            for pair in &scalar[..width] {
                                submit(pair);
                            }
                        }
                    });
                    if batch {
                        b.push(us);
                    } else {
                        a.push(us);
                    }
                };
                run(reverse);
                run(!reverse);
            }
            timing.push(json!({"width":width,"scalar_pairs_us":a,"batch_pair_us":b,
                "scalar_median_us":median(&a),"batch_median_us":median(&b),"component_speedup":median(&a)/median(&b)}));
        }
        stream.synchronize().unwrap();
        let after = cuda_device_allocation_snapshot();
        assert_eq!(warm.successful_allocations, after.successful_allocations);
        assert_eq!(warm.successful_frees, after.successful_frees);
        // Neither the original scalar context nor local plan owners are needed
        // after construction: new bindings retain exact weight/plan/IO owners.
        drop(sources);
        drop(context);
        drop(plan);
        drop(scalar);
        let weak_x = Rc::downgrade(&x);
        let weak_y: Vec<_> = y.iter().map(Rc::downgrade).collect();
        drop(x);
        drop(y);
        assert!(weak_x.upgrade().is_some());
        submit(&widths[B - 1]);
        for i in 0..2 {
            assert_eq!(
                read(&stream, &weak_y[i].upgrade().unwrap()),
                expected_reversed[i]
            );
        }
        drop(widths);
        assert!(weak_x.upgrade().is_none() && weak_y.iter().all(|w| w.upgrade().is_none()));
        drop(stream);
        drop(reader);
        let after_drop = cuda_device_allocation_snapshot();
        assert_eq!(after_drop.live_bytes, baseline.live_bytes);
        assert_eq!(after_drop.live_allocations, baseline.live_allocations);
        results.push(json!({"layer":layer,"input_f64_fixture_sha256":hash(&norm),
            "repeated_F32_input_sha256":hash(&repeated),"reversed_F32_input_sha256":hash(&reversed),
            "input_origin":"independent_layer_fixture_embedding_norm_NOT_full_target_activation",
            "widths5_to8":"repeat_fixture_rows_not_new_target_history",
            "shared_arena_bytes":arena_bytes,"io_bytes":B*(K+N[0]+N[1])*4,
            "bitwise_all_widths_and_refreshed_input":true,"math":math_results,"timing":timing,
            "numerical_profile":8,"split_k":16,"hot_allocations":0,"hot_frees":0,
            "reject_foreign_stream_alias_misalignment_invalid_width":true,
            "retained_after_source_drop":true,"live_bytes_delta":0,"live_allocations_delta":0}));
        println!("AUGUR_QPN_PREFIX_COMPONENT layer{layer} B1..8 exact PASS");
    }
    assert_eq!(results.len(), 2);
    let after = cuda_device_allocation_snapshot();
    assert_eq!(
        after.successful_allocations - baseline.successful_allocations,
        after.successful_frees - baseline.successful_frees
    );
    std::fs::write(
        output.join("component.json"),
        serde_json::to_vec_pretty(&json!({
        "status":"PASS_AUGUR_QPN_SHARED_WEIGHT_PREFILL_PROJECTION_COMPONENT_ONLY",
        "reference_sha256":hash(&bytes),"cases":results,
        "complete_prefill_implemented":false,"production_default_changed":false,
        "dynamic_thresholds_measured":false,"HTTP_qualified":false,
        "cleanup_live_bytes_delta":0,"cleanup_live_allocations_delta":0}))
        .unwrap(),
    )
    .unwrap();
}
