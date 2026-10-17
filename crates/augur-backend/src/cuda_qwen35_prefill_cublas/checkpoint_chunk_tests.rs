use super::*;
use crate::{CudaQwen35LinearProjectionPostprocessArgsV1, CudaQwen35LinearStepArgsV1};
use augur_core::PhysicalLayoutId as Layout;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{io::Write, path::Path, time::Instant};

#[test]
fn source_chunk_rejects_null_before_cuda() {
    let r = unsafe {
        augur_cuda_sys::augur_cuda_qwen35_checkpoint_prefill_linear_chunk_prevalidated_submit_v1(
            0,
            std::ptr::null_mut(),
            std::ptr::null(),
        )
    };
    assert_eq!(r.status, augur_cuda_sys::AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
}
struct Guard {
    b: OwnedCudaBuffer,
    n: usize,
}
impl Guard {
    fn new(s: &OwnedCudaStream, data: &[u8]) -> Self {
        let b = OwnedCudaBuffer::allocate(s.device_ordinal(), data.len() + 512).unwrap();
        let g = Self { b, n: data.len() };
        g.write(s, data);
        g
    }
    fn write(&self, s: &OwnedCudaStream, data: &[u8]) {
        assert_eq!(data.len(), self.n);
        let mut bytes = vec![0xa5; self.n + 512];
        bytes[256..256 + self.n].copy_from_slice(data);
        self.b.copy_from_host_sync(s, &bytes).unwrap();
    }
    fn v(&self, first: usize, n: usize) -> CudaF32View {
        CudaF32View::new(
            self.b.view(256 + first * 4, n * 4).unwrap(),
            Layout::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            n,
        )
        .unwrap()
    }
    fn read(&self, s: &OwnedCudaStream) -> Vec<u8> {
        let mut bytes = vec![0; self.n + 512];
        self.b.copy_to_host_sync(s, &mut bytes).unwrap();
        assert!(bytes[..256]
            .iter()
            .chain(&bytes[256 + self.n..])
            .all(|&v| v == 0xa5));
        bytes[256..256 + self.n].to_vec()
    }
}
fn hash(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn file(v: &Value) -> Vec<u8> {
    let b = std::fs::read(v["path"].as_str().unwrap()).unwrap();
    assert_eq!(hash(&b), v["sha256"].as_str().unwrap());
    b
}
fn write(root: &Path, name: &str, b: &[u8]) -> Value {
    let p = root.join(name);
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&p)
        .unwrap();
    f.write_all(b).unwrap();
    f.sync_all().unwrap();
    json!({"path":p,"sha256":hash(b)})
}
fn metric(got: &[u8], r: &[u8]) -> Value {
    assert_eq!(got.len() * 2, r.len());
    let (mut d2, mut r2, mut worst, mut max_abs) = (0f64, 0f64, 0f64, 0f64);
    let mut finite = true;
    for (a, b) in got.chunks_exact(4).zip(r.chunks_exact(8)) {
        let a = f32::from_le_bytes(a.try_into().unwrap()) as f64;
        let b = f64::from_le_bytes(b.try_into().unwrap());
        finite &= a.is_finite() && b.is_finite();
        let d = (a - b).abs();
        d2 += d * d;
        r2 += b * b;
        worst = worst.max(d / (1e-3 + 1e-3 * b.abs()));
        max_abs = max_abs.max(d);
    }
    let l2 = (d2 / r2.max(1e-300)).sqrt();
    json!({"passed":finite&&l2<=1e-3&&worst<=1.,"finite":finite,"relative_l2":l2,"worst_abs_plus_rel_budget_ratio":worst,"max_absolute":max_abs,"atol":1e-3,"rtol":1e-3})
}
fn median(v: &[f64]) -> f64 {
    let mut x = v.to_vec();
    x.sort_by(f64::total_cmp);
    (x[1] + x[2]) * 0.5
}

#[test]
#[ignore = "exclusive SM70; actual Augur projection-input fixture and independent F64 recurrence"]
fn actual_checkpoint_chunk_math_state_schedule_and_wall() {
    let manifest =
        std::fs::read(std::env::var_os("AUGUR_CHECKPOINT_CHUNK_FIXTURE").expect("fixture")).unwrap();
    let m: Value = serde_json::from_slice(&manifest).unwrap();
    assert_eq!(m["schema"], "quasar-source-gdn-chunk-fixture-v1");
    let root =
        std::path::PathBuf::from(std::env::var_os("AUGUR_CHECKPOINT_CHUNK_OUTPUT").expect("output"));
    let before = crate::cuda_device_allocation_snapshot();
    let mut cases = Vec::new();
    let mut all = true;
    {
        let s = OwnedCudaStream::create(0).unwrap();
        let foreign = OwnedCudaStream::create(0).unwrap();
        for case in m["cases"].as_array().unwrap() {
            let id = case["id"].as_str().unwrap();
            let tokens = case["tokens"].as_u64().unwrap() as usize;
            let eps = case["epsilon"].as_f64().unwrap() as f32;
            let names = [
                "qkv",
                "z",
                "conv",
                "history",
                "alpha",
                "beta_raw",
                "bias",
                "ssm_a",
                "state",
                "norm",
                "activated",
                "recurrent",
                "output",
            ];
            let initial: Vec<Vec<u8>> = names.iter().map(|n| file(&case["inputs"][n])).collect();
            let b: Vec<_> = initial.iter().map(|v| Guard::new(&s, v)).collect();
            let views = |first: usize, width: usize| CudaQwen35PrefillLinearChunkViewsV1 {
                qkv: b[0].v(first * 10240, width * 10240),
                z: b[1].v(first * 6144, width * 6144),
                conv_weights: b[2].v(0, 40960),
                conv_history: b[3].v(0, 30720),
                alpha_raw: b[4].v(first * 48, width * 48),
                beta_raw: b[5].v(first * 48, width * 48),
                dt_bias: b[6].v(0, 48),
                ssm_a: b[7].v(0, 48),
                state: b[8].v(0, 48 * 128 * 128),
                norm_weight: b[9].v(0, 128),
                activated: b[10].v(first * 10240, width * 10240),
                recurrent: b[11].v(first * 6144, width * 6144),
                output: b[12].v(first * 6144, width * 6144),
            };
            let schedules: Vec<Vec<usize>> = if tokens == 513 {
                vec![
                    vec![512, 1],
                    vec![256, 256, 1],
                    vec![128, 128, 128, 128, 1],
                    vec![1, 7, 8, 9, 127, 129, 232],
                ]
            } else {
                vec![vec![tokens], vec![1; tokens]]
            };
            let programs: Vec<Vec<_>> = schedules
                .iter()
                .map(|parts| {
                    let mut first = 0;
                    parts
                        .iter()
                        .map(|&n| {
                            let p =
                                CudaQwen35PrefillLinearChunkV1::bind(&s, views(first, n), n, eps)
                                    .unwrap()
                                    .for_checkpoint_source()
                                    .unwrap();
                            first += n;
                            p
                        })
                        .collect()
                })
                .collect();
            assert!(programs[0][0].submit(&foreign).is_err());
            assert!(CudaQwen35PrefillLinearChunkV1::bind(&s, views(0, 1), 0, eps).is_err());
            let d = Guard::new(&s, &vec![0; 48 * 4]);
            let beta = Guard::new(&s, &vec![0; 48 * 4]);
            let scalar: Vec<_> = (0..tokens)
                .map(|at| {
                    let v = views(at, 1);
                    let gate = CudaQwen35LinearProjectionPostprocessArgsV1::bind(
                        &s,
                        v.alpha_raw,
                        v.beta_raw,
                        v.dt_bias,
                        v.ssm_a,
                        d.v(0, 48),
                        beta.v(0, 48),
                    )
                    .unwrap();
                    let step = CudaQwen35LinearStepArgsV1::bind(
                        &s,
                        v.qkv,
                        v.z,
                        v.conv_weights,
                        v.conv_history,
                        d.v(0, 48),
                        beta.v(0, 48),
                        v.state,
                        v.norm_weight,
                        v.activated,
                        v.recurrent,
                        v.output,
                        eps,
                    )
                    .unwrap()
                    .for_checkpoint_source();
                    (gate, step)
                })
                .collect();
            let reset = || {
                for i in [3, 8, 10, 11, 12] {
                    b[i].write(&s, &initial[i]);
                }
            };
            let outputs = [
                (3, "history"),
                (8, "state"),
                (10, "activated"),
                (11, "recurrent"),
                (12, "output"),
            ];
            let references: Vec<_> = outputs
                .iter()
                .map(|(_, n)| file(&case["reference"][n]))
                .collect();
            reset();
            for (g, p) in &scalar {
                g.submit_for_plan(&s).unwrap();
                p.submit_for_plan(&s).unwrap();
            }
            let old_metrics: Vec<_> = outputs
                .iter()
                .zip(&references)
                .map(|((i, n), r)| json!({"stage":n,"metric":metric(&b[*i].read(&s),r)}))
                .collect();
            all &= old_metrics.iter().all(|m| m["metric"]["passed"] == true);
            let mut schedule_reports = Vec::new();
            let mut first_result = None;
            for (schedule, parts) in schedules.iter().zip(&programs) {
                reset();
                let warm = crate::cuda_device_allocation_snapshot();
                for p in parts {
                    p.submit(&s).unwrap();
                }
                s.synchronize().unwrap();
                let hot = crate::cuda_device_allocation_snapshot();
                assert_eq!(warm.successful_allocations, hot.successful_allocations);
                assert_eq!(warm.successful_frees, hot.successful_frees);
                let result: Vec<_> = outputs.iter().map(|(i, _)| b[*i].read(&s)).collect();
                let equal = first_result
                    .as_ref()
                    .is_none_or(|first: &Vec<Vec<u8>>| *first == result);
                all &= equal;
                let metrics: Vec<_> = outputs
                    .iter()
                    .zip(result.iter().zip(&references))
                    .map(|((_, name), (v, r))| json!({"stage":name,"metric":metric(v,r)}))
                    .collect();
                all &= metrics.iter().all(|m| m["metric"]["passed"] == true);
                if first_result.is_none() {
                    let mut files = serde_json::Map::new();
                    for ((_, n), bytes) in outputs.iter().zip(&result) {
                        files.insert((*n).into(), write(&root, &format!("{id}-{n}.f32"), bytes));
                    }
                    write(
                        &root,
                        &format!("{id}-output-files.json"),
                        &serde_json::to_vec_pretty(&files).unwrap(),
                    );
                    first_result = Some(result);
                }
                schedule_reports.push(json!({"schedule":schedule,"math":metrics,"same_input_split_bitwise":equal,"hot_allocations":0,"hot_frees":0}));
            }
            // Negative control: the interleaved legacy profile must not masquerade
            // as checkpoint source math. It consumes identical explicit inputs.
            reset();
            let mut first = 0;
            for &width in &schedules[0] {
                CudaQwen35PrefillLinearChunkV1::bind(&s, views(first, width), width, eps)
                    .unwrap()
                    .submit_shared_scalars(&s)
                    .unwrap();
                first += width;
            }
            let legacy_wrong = metric(&b[12].read(&s), &references[4]);
            assert_eq!(
                legacy_wrong["passed"], false,
                "fixture must catch wrong head/normalization profile"
            );
            let mut a = Vec::new();
            let mut z = Vec::new();
            if tokens == 513 {
                for reverse in [false, true, true, false] {
                    for wide in [reverse, !reverse] {
                        let mut us = 0.;
                        for _ in 0..6 {
                            reset();
                            s.synchronize().unwrap();
                            let start = Instant::now();
                            if wide {
                                programs[0][0].submit(&s).unwrap();
                            } else {
                                for (g, p) in &scalar[..512] {
                                    g.submit_for_plan(&s).unwrap();
                                    p.submit_for_plan(&s).unwrap();
                                }
                            }
                            s.synchronize().unwrap();
                            us += start.elapsed().as_secs_f64() * 1e6;
                        }
                        if wide {
                            z.push(us / 6.);
                        } else {
                            a.push(us / 6.);
                        }
                    }
                }
            }
            for i in [0, 1, 2, 4, 5, 6, 7, 9] {
                assert_eq!(b[i].read(&s), initial[i], "immutable input modified");
            }
            cases.push(json!({"id":id,"tokens":tokens,"scalar_math":old_metrics,"schedule_results":schedule_reports,"legacy_wrong_profile":legacy_wrong,
            "B512_timing":if a.is_empty(){Value::Null}else{json!({"scalar_us":a,"wide_us":z,"scalar_median_us":median(&a),"wide_median_us":median(&z),"ratio":median(&a)/median(&z),"reset_and_diagnostic_IO_in_timing":false})}}));
            println!("SOURCE_GDN {id} math/schedule pass so far: {all}");
        }
        s.synchronize().unwrap();
    }
    let after = crate::cuda_device_allocation_snapshot();
    assert_eq!(before.live_bytes, after.live_bytes);
    assert_eq!(before.live_allocations, after.live_allocations);
    let report = json!({"schema":"quasar-source-gdn-chunk-component-v1","status":if all{"PASS_COMPONENT_ONLY"}else{"FAIL_MATH_OR_SPLIT"},"fixture_sha256":hash(&manifest),
        "profile":CudaQwen35CheckpointPrefillLinearChunkV1::PROFILE_LABEL,"cases":cases,"cleanup_live_bytes_delta":after.live_bytes-before.live_bytes,"cleanup_live_allocations_delta":after.live_allocations-before.live_allocations,
        "projection_source":"CPU F64 source projections rounded to F32, not GPU QPN output","full_layer_qualified":false,"HTTP_qualified":false});
    write(
        &root,
        "component.json",
        &serde_json::to_vec_pretty(&report).unwrap(),
    );
    assert!(all, "preserve evidence; do not relax thresholds");
}
