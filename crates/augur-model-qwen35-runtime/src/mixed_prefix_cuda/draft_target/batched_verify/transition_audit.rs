//! Same live prefix, same tokens. Diagnostic only; never publishes or retains a session.
use super::*;
use std::{
    fs::{self, OpenOptions},
    io::Write,
};

const POSITION: usize = 6144;
const ROWS: usize = 512;
const AUDIT_ROWS: usize = B;

fn dump(directory: &Path, name: &str, bytes: &[u8]) -> Result<Value> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(directory.join(name))
        .map_err(|e| invalid(e.to_string()))?;
    file.write_all(bytes).map_err(|e| invalid(e.to_string()))?;
    file.sync_all().map_err(|e| invalid(e.to_string()))?;
    Ok(json!({"file":name,"bytes":bytes.len(),"sha256":sha256_hex(bytes)}))
}

fn read(d: &Driver, buffer: &OwnedCudaBuffer, offset: usize, len: usize) -> Result<Vec<u8>> {
    d.stream.synchronize()?;
    let receipt = d.stream.submission_receipt()?;
    let mut bytes = vec![0; len];
    let proof = buffer
        .copy_range_to_host_sync_with_submission_proof(&d.stream, offset, &mut bytes, receipt)?;
    if !proof.covers_submission(receipt) || !proof.belongs_to_stream(&d.stream)? {
        return Err(invalid("transition read owner/submission"));
    }
    Ok(bytes)
}

struct PrefixState {
    linear: Vec<(usize, [Vec<u8>; 2])>,
    manifest: Value,
}
impl PrefixState {
    fn capture(d: &Driver) -> Result<Self> {
        let mut linear = Vec::new();
        let mut records = Vec::new();
        for (layer, plan) in d.plans.iter().enumerate() {
            let mut data = Vec::new();
            for (plane, state) in plan.terminal_state_buffers().iter().enumerate() {
                let len = if matches!(plan, DevicePlan::Full(_)) {
                    POSITION * QWEN35_KV8_CACHE_ROW_BYTES
                } else {
                    state.len_bytes()
                };
                let bytes = read(d, state, 0, len)?;
                records.push(json!({"layer":layer,"plane":plane,"bytes":len,
                    "sha256":sha256_hex(&bytes)}));
                if matches!(plan, DevicePlan::Linear(_)) {
                    data.push(bytes);
                }
            }
            if matches!(plan, DevicePlan::Linear(_)) {
                let pair: [Vec<u8>; 2] = data
                    .try_into()
                    .map_err(|_| invalid("linear plane geometry"))?;
                linear.push((layer, pair));
            }
        }
        Ok(Self {
            linear,
            manifest: json!(records),
        })
    }
    fn restore_and_verify(&self, d: &Driver) -> Result<()> {
        d.stream.synchronize()?;
        for (layer, pair) in &self.linear {
            for (state, bytes) in d.plans[*layer].terminal_state_buffers().iter().zip(pair) {
                state.copy_from_host_sync(&d.stream, bytes)?;
            }
        }
        // KV prefix is immutable under these append-only operations. Hash all
        // initialized prefix bytes again; future rows are overwritten per arm.
        if Self::capture(d)?.manifest != self.manifest {
            return Err(invalid("transition restored prefix is not byte-exact"));
        }
        Ok(())
    }
}

fn terminal_manifest(d: &Driver, end: usize) -> Result<Value> {
    if end != POSITION + ROWS { return Err(invalid("transition terminal extent")); }
    let mut records = Vec::new();
    for (layer, plan) in d.plans.iter().enumerate() {
        for (plane, buffer) in plan.terminal_state_buffers().iter().enumerate() {
            let len = if matches!(plan, DevicePlan::Full(_)) {
                end * QWEN35_KV8_CACHE_ROW_BYTES
            } else { buffer.len_bytes() };
            let bytes = read(d, buffer, 0, len)?;
            records.push(json!({"layer":layer,"plane":plane,"bytes":len,"sha256":sha256_hex(&bytes)}));
        }
    }
    Ok(json!(records))
}

// Complete512 real rows through the existing B8 schedule, then compare all
// initialized terminal planes with the layer-major512 diagnostic (no padding).
fn canonical_sequence_terminal(
    d: &mut Driver, ids: &[usize], epsilon: f32, theta: f32,
) -> Result<Value> {
    d.cursor = POSITION;
    let core = BatchedVerifyAudit::allocate(d, POSITION+ROWS, epsilon, theta,
        false, true, false, true, true, true, true)?;
    let result = (|| {
        let operations = core.bind_operations(d)?;
        for start in (0..ROWS).step_by(B) {
            d.cursor = POSITION + start;
            let bytes = ids[start..start+B].iter()
                .flat_map(|&id| (id as u32).to_le_bytes()).collect::<Vec<_>>();
            d.path.copy_from_host_sync(&d.stream, &bytes)?;
            core.submit_backbone(d, &operations, false)?;
        }
        terminal_manifest(d, POSITION+ROWS)
    })();
    let drain = d.stream.synchronize();
    if result.is_err() || drain.is_err() { d.poisoned = true; }
    drain?;
    result
}

fn layer_capture(
    d: &Driver,
    w: &PrefillWorkspace,
    layer: usize,
    directory: &Path,
    arm: &str,
    artifacts: &mut Vec<Value>,
    inputs: Option<&mut Vec<Vec<u8>>>,
) -> Result<()> {
    let input = if layer.is_multiple_of(2) {
        &w.hidden_a
    } else {
        &w.hidden_b
    };
    let output = if layer.is_multiple_of(2) {
        &w.hidden_b
    } else {
        &w.hidden_a
    };
    let bytes = read(d, input, 0, AUDIT_ROWS * H * 4)?;
    if let Some(inputs) = inputs {
        inputs.push(bytes.clone());
    }
    let mut emit = |stage: &str, buffer: &OwnedCudaBuffer, width: usize| -> Result<()> {
        let bytes = read(d, buffer, 0, AUDIT_ROWS * width * 4)?;
        let name = format!("{arm}-layer{layer:02}-{stage}.bin");
        let mut value = dump(directory, &name, &bytes)?;
        value["arm"] = json!(arm);
        value["layer"] = json!(layer);
        value["stage"] = json!(stage);
        value["rows"] = json!(AUDIT_ROWS);
        value["width"] = json!(width);
        value["dtype"] = json!("f32_le");
        artifacts.push(value);
        Ok(())
    };
    emit("input", input, H)?;
    emit("attn_norm", &w.normalized, H)?;
    emit("attention_residual", &w.attention_residual, H)?;
    emit("post_norm", &w.post_norm, H)?;
    emit("ffn_output", &w.ffn_output, H)?;
    emit("hidden_output", output, H)?;
    if matches!(d.plans[layer], DevicePlan::Linear(_)) {
        for (name, buffer, width) in [
            ("qkv", &w.qkv, linear::QKV_ELEMENTS),
            ("z", &w.z, linear::RECURRENT_ELEMENTS),
            ("alpha_raw", &w.alpha_raw, linear::VALUE_HEADS),
            ("beta_raw", &w.beta_raw, linear::VALUE_HEADS),
            ("recurrent", &w.recurrent, linear::RECURRENT_ELEMENTS),
        ] {
            emit(name, buffer, width)?;
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Policy {
    native_ffn: bool,
    canonical_scalars: bool,
    native_attention_projection: bool,
    canonical_norm: bool,
    canonical_full_core: bool,
}
const CANONICAL: Policy = Policy {
    native_ffn: true,
    canonical_scalars: true,
    native_attention_projection: true,
    canonical_norm: true,
    canonical_full_core: true,
};
const PREFILL8: Policy = Policy {
    native_ffn: false,
    canonical_scalars: false,
    native_attention_projection: false,
    canonical_norm: false,
    canonical_full_core: false,
};
fn ablations() -> [(&'static str, Policy); 6] {
    [
        ("prefill8_same_input", PREFILL8),
        (
            "canonical8_cublas_projection_same_input",
            Policy {
                native_ffn: false,
                native_attention_projection: false,
                ..CANONICAL
            },
        ),
        (
            "canonical8_cublas_scalars_same_input",
            Policy {
                canonical_scalars: false,
                ..CANONICAL
            },
        ),
        (
            "canonical8_prefill_norm_same_input",
            Policy {
                canonical_norm: false,
                ..CANONICAL
            },
        ),
        (
            "canonical8_prefill_attention_same_input",
            Policy {
                canonical_full_core: false,
                ..CANONICAL
            },
        ),
        (
            "prefill8_canonical_scalars_same_input",
            Policy {
                canonical_scalars: true,
                ..PREFILL8
            },
        ),
    ]
}

fn canonical_arm(
    d: &mut Driver,
    ids: &[usize],
    epsilon: f32,
    theta: f32,
    directory: &Path,
    arm: &str,
    policy: Policy,
    injected: Option<&[Vec<u8>]>,
    artifacts: &mut Vec<Value>,
) -> Result<()> {
    d.cursor = POSITION;
    let core = BatchedVerifyAudit::allocate(
        d,
        d.capacity,
        epsilon,
        theta,
        false,
        policy.native_ffn,
        false,
        policy.canonical_scalars,
        policy.native_attention_projection,
        policy.canonical_norm,
        policy.canonical_full_core,
    )?;
    let result: Result<()> = (|| {
        let w = &core.workspace;
        let mut operations = core.bind_operations(d)?;
        for op in &mut operations {
            if let PrefillLayerOps::Full(full) = op {
                full.simt_shared_softmax = true;
            }
        }
        for (row, &id) in ids[..AUDIT_ROWS].iter().enumerate() {
            d.token
                .copy_from_host_sync(&d.stream, &(id as u32).to_le_bytes())?;
            CudaBf16TokenRowGatherArgsV1::bind(
                &d.stream,
                CudaBf16WeightViewV1::new(
                    view(&d.embedding)?,
                    PhysicalLayoutId::DEVICE_CONTIGUOUS_BF16_CUDA_V1,
                    V,
                    H,
                )?,
                view(&d.token)?,
                CudaF32View::new(
                    w.hidden_a.view(row * H * 4, H * 4)?,
                    PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                    H,
                )?,
            )?
            .submit_for_plan(&d.stream)?;
        }
        for (layer, operation) in operations.iter().enumerate() {
            if let Some(inputs) = injected {
                let input = if layer.is_multiple_of(2) {
                    &w.hidden_a
                } else {
                    &w.hidden_b
                };
                input.copy_range_from_host_sync(&d.stream, 0, &inputs[layer])?;
            }
            match (operation, &d.plans[layer]) {
                (PrefillLayerOps::Full(p), DevicePlan::Full(plan))
                    if policy.canonical_full_core =>
                {
                    full_attention::submit(d, w, p, plan)?
                }
                _ => operation.submit(&d.stream, &w.cublas)?,
            }
            layer_capture(d, w, layer, directory, arm, artifacts, None)?;
        }
        d.stream.synchronize()?;
        Ok(())
    })();
    // Do not release workspace on a partially submitted stream before draining.
    let drain = d.stream.synchronize();
    if result.is_err() || drain.is_err() {
        d.poisoned = true;
    }
    result?;
    drain?;
    Ok(())
}

fn install_scalar_rows(
    d: &Driver,
    w: &PrefillWorkspace,
    ops: &mut [PrefillLayerOps],
) -> Result<()> {
    for (op, plan) in ops.iter_mut().zip(&d.plans) {
        if let (PrefillLayerOps::Linear(op), DevicePlan::Linear(plan)) = (op, plan) {
            let projection = plan
                .projection
                .as_ref()
                .ok_or_else(|| invalid("scalar weights absent"))?;
            let bind = |row: usize,
                        weight: &OwnedCudaBuffer,
                        output: &OwnedCudaBuffer|
             -> Result<CudaBf16F32MatvecArgsV1> {
                Ok(CudaBf16F32MatvecArgsV1::bind(
                    &d.stream,
                    CudaBf16WeightViewV1::new(
                        view(weight)?,
                        PhysicalLayoutId::DEVICE_CONTIGUOUS_BF16_CUDA_V1,
                        linear::VALUE_HEADS,
                        H,
                    )?,
                    CudaF32View::new(
                        w.normalized.view(row * H * 4, H * 4)?,
                        PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                        H,
                    )?,
                    CudaF32View::new(
                        output.view(row * linear::VALUE_HEADS * 4, linear::VALUE_HEADS * 4)?,
                        PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                        linear::VALUE_HEADS,
                    )?,
                )?)
            };
            op.canonical_gdn_scalars = Some(
                (0..ROWS)
                    .map(|row| {
                        Ok([
                            bind(row, &projection.alpha_weight, &w.alpha_raw)?,
                            bind(row, &projection.beta_weight, &w.beta_raw)?,
                        ])
                    })
                    .collect::<Result<Vec<_>>>()?,
            );
        }
    }
    Ok(())
}

pub(in super::super) fn run(
    d: &mut Driver,
    mode: AutoregressivePrefillMode,
    input_hidden: &[f32],
    ids: &[usize],
    epsilon: f32,
    theta: f32,
    directory: &Path,
) -> Result<Value> {
    if d.cursor != POSITION
        || d.context_rows != 2048
        || d.capacity != 196608
        || input_hidden.len() != ROWS * H
        || ids.len() != ROWS
        || ids.iter().any(|&t| t >= V)
        || directory.exists()
    {
        return Err(invalid("fixed transition diagnostic geometry"));
    }
    fs::create_dir(directory).map_err(|e| invalid(e.to_string()))?;
    let AutoregressivePrefillMode::Batched(prefill) = mode else {
        return Err(invalid(
            "transition audit requires actual batched prefix owner",
        ));
    };
    let result = (|| {
        if prefill.workspace.max_tokens != ROWS
            || !prefill.simt_shared_softmax
            || prefill.tensor_core_prefix_attention
            || prefill.direct_activation
        {
            return Err(invalid("transition prefix execution policy"));
        }
        let prefix = PrefixState::capture(d)?;
        dump(
            directory,
            "prefix-state.json",
            &serde_json::to_vec_pretty(&prefix.manifest).map_err(|e| invalid(e.to_string()))?,
        )?;
        let w = &prefill.workspace;
        // The original workspace initialized RoPE only through the consumed
        // prefix. Extending its schedule MUST initialize the new positions,
        // not merely rely on the larger reserved capacity. Use the unchanged
        // native table generator before any continuation operation is bound.
        w.upload_rope_tables(&d.stream, POSITION + ROWS, 0, theta)?;
        let scratch = w
            .dgemm_score_scratch
            .as_ref()
            .ok_or_else(|| invalid("transition DGEMM scratch absent"))?;
        let mut operations = Vec::with_capacity(64);
        for (layer, plan) in d.plans.iter().enumerate() {
            let op = match plan {
                DevicePlan::Linear(p) => {
                    PrefillLayerOps::Linear(w.build_linear_ops(&d.stream, p, layer, ROWS, epsilon)?)
                }
                DevicePlan::Full(p) => {
                    let mut op = w.build_full_ops(&d.stream, p, layer, ROWS, POSITION, epsilon)?;
                    op.simt_shared_softmax = true;
                    let n = CudaQwen35PrefillKv8ChunkAttentionV1::dgemm_score_elements(
                        ROWS, d.capacity,
                    )?;
                    op.dgemm_attention = Some(op.attention.with_dgemm_scores(
                        &d.stream,
                        &w.cublas,
                        CudaF32View::new(
                            scratch.view(0, n * 4)?,
                            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                            n,
                        )?,
                    )?);
                    PrefillLayerOps::Full(op)
                }
            };
            operations.push(op);
        }
        let mut artifacts = Vec::new();
        let mut inputs = Vec::with_capacity(64);
        let mut full_inputs = Vec::with_capacity(64);
        let mut timings = Vec::new();
        let start = OwnedCudaEvent::create(0)?;
        let stop = OwnedCudaEvent::create(0)?;
        w.hidden_a
            .copy_from_host_sync(&d.stream, &f32_bytes(input_hidden))?;
        for (layer, op) in operations.iter().enumerate() {
            let input = if layer.is_multiple_of(2) {
                &w.hidden_a
            } else {
                &w.hidden_b
            };
            full_inputs.push(read(d, input, 0, ROWS * H * 4)?);
            start.record_for_submission(&d.stream, d.stream.submission_receipt()?)?;
            let host = Instant::now();
            op.submit(&d.stream, &w.cublas)?;
            stop.record_for_submission(&d.stream, d.stream.submission_receipt()?)?;
            d.stream.synchronize()?;
            timings.push(json!({"arm":"prefill512","layer":layer,
                "layer_sync_wall_ms":host.elapsed().as_secs_f64()*1000.0,"cuda_event_ms":start.elapsed_ms(&stop)?}));
            layer_capture(
                d,
                w,
                layer,
                directory,
                "prefill512",
                &mut artifacts,
                Some(&mut inputs),
            )?;
        }
        prefix.restore_and_verify(d)?;
        // Research candidate: keep 512-row prefill operations, replacing ONLY
        // alpha/beta with the existing BF16-weight/F32-input canonical matvec.
        // These row launches measure accuracy/cost, not a proposed fast kernel.
        install_scalar_rows(d, w, &mut operations)?;
        for arm in [
            "prefill512_canonical_scalars_chain",
            "prefill512_canonical_scalars_same_input",
        ] {
            w.hidden_a
                .copy_from_host_sync(&d.stream, &f32_bytes(input_hidden))?;
            for (layer, op) in operations.iter().enumerate() {
                if arm.ends_with("same_input") {
                    let input = if layer.is_multiple_of(2) {
                        &w.hidden_a
                    } else {
                        &w.hidden_b
                    };
                    input.copy_from_host_sync(&d.stream, &full_inputs[layer])?;
                }
                start.record_for_submission(&d.stream, d.stream.submission_receipt()?)?;
                let host = Instant::now();
                op.submit(&d.stream, &w.cublas)?;
                stop.record_for_submission(&d.stream, d.stream.submission_receipt()?)?;
                d.stream.synchronize()?;
                timings.push(json!({"arm":arm,"layer":layer,
                    "layer_sync_wall_ms":host.elapsed().as_secs_f64()*1000.0,"cuda_event_ms":start.elapsed_ms(&stop)?}));
                layer_capture(d, w, layer, directory, arm, &mut artifacts, None)?;
            }
            prefix.restore_and_verify(d)?;
        }
        prefill_tiled_audit::install_projections_and_norms(&mut operations)?;
        let mut tiled_terminal = None;
        let mut grid_terminal = None;
        for arm in ["prefill512_full_canonical_chain", "prefill512_full_canonical_same_input",
                    "prefill512_grid_canonical_chain", "prefill512_grid_canonical_same_input"] {
            if arm.starts_with("prefill512_grid") { prefill_tiled_audit::install_grid(&mut operations)?; }
            d.cursor = POSITION;
            w.hidden_a.copy_from_host_sync(&d.stream, &f32_bytes(input_hidden))?;
            for (layer, op) in operations.iter().enumerate() {
                if arm.ends_with("same_input") {
                    let input = if layer.is_multiple_of(2) { &w.hidden_a } else { &w.hidden_b };
                    input.copy_from_host_sync(&d.stream, &full_inputs[layer])?;
                }
                start.record_for_submission(&d.stream, d.stream.submission_receipt()?)?;
                let host = Instant::now();
                prefill_tiled_audit::submit_layer(d, w, op, &d.plans[layer], ROWS)?;
                stop.record_for_submission(&d.stream, d.stream.submission_receipt()?)?;
                d.stream.synchronize()?;
                timings.push(json!({"arm":arm,"layer":layer,
                    "layer_sync_wall_ms":host.elapsed().as_secs_f64()*1000.0,"cuda_event_ms":start.elapsed_ms(&stop)?}));
                layer_capture(d, w, layer, directory, arm, &mut artifacts, None)?;
            }
            if arm.ends_with("_chain") {
                let snapshot = terminal_manifest(d,POSITION+ROWS)?;
                if arm.starts_with("prefill512_grid") { grid_terminal = Some(snapshot); }
                else { tiled_terminal = Some(snapshot); }
            }
            prefix.restore_and_verify(d)?;
        }
        Ok((prefix, inputs, artifacts, timings, tiled_terminal.ok_or_else(|| invalid("missing tiled terminal"))?,
            grid_terminal.ok_or_else(|| invalid("missing grid terminal"))?))
    })();
    let drain = d.stream.synchronize();
    if result.is_err() || drain.is_err() {
        d.poisoned = true;
    }
    let (prefix, inputs, mut artifacts, timings, tiled_terminal, grid_terminal) = result?;
    drain?;
    // Free large prefill workspace before allocating canonical capture scratch.
    drop(prefill);
    canonical_arm(
        d,
        ids,
        epsilon,
        theta,
        directory,
        "canonical8_chain",
        CANONICAL,
        None,
        &mut artifacts,
    )?;
    prefix.restore_and_verify(d)?;
    canonical_arm(
        d,
        ids,
        epsilon,
        theta,
        directory,
        "canonical8_same_input",
        CANONICAL,
        Some(&inputs),
        &mut artifacts,
    )?;
    prefix.restore_and_verify(d)?;
    let mut policies = Vec::new();
    for (name, policy) in ablations() {
        canonical_arm(
            d,
            ids,
            epsilon,
            theta,
            directory,
            name,
            policy,
            Some(&inputs),
            &mut artifacts,
        )?;
        prefix.restore_and_verify(d)?;
        policies.push(json!({"arm":name,"native_ffn":policy.native_ffn,
            "canonical_scalars":policy.canonical_scalars,
            "native_attention_projection":policy.native_attention_projection,
            "canonical_norm":policy.canonical_norm,"canonical_full_core":policy.canonical_full_core}));
    }
    let canonical_terminal = canonical_sequence_terminal(d, ids, epsilon, theta)?;
    prefix.restore_and_verify(d)?;
    let terminal_exact = tiled_terminal == canonical_terminal;
    let report = json!({"schema":"qwen35-prefill-capture-transition-audit-v5",
        "position":POSITION,"prefill_rows":ROWS,"canonical_rows":B,"audited_rows":AUDIT_ROWS,
        "exact_prefix_restored_between_arms":true,"prefix_planes":128,
        "prefill_rope_initialized_end":POSITION+ROWS,
        "canonical_same_input_policy":"inject_prefill_first8_hidden_at_each_layer",
        "ablation_policies":policies,"diagnostic_arithmetic_ablation":true,
        "large_prefill_scalar_arms":["prefill512_canonical_scalars_chain","prefill512_canonical_scalars_same_input"],
        "large_full_canonical_arms":["prefill512_full_canonical_chain","prefill512_full_canonical_same_input"],
        "large_grid_canonical_arms":["prefill512_grid_canonical_chain","prefill512_grid_canonical_same_input"],
        "grid_canonical_terminal":{"end":POSITION+ROWS,"planes":128,
            "exact":grid_terminal==canonical_terminal,"layer_major512":grid_terminal,"canonical8x64":canonical_terminal},
        "full_canonical_terminal":{"end":POSITION+ROWS,"planes":128,
            "exact":terminal_exact,"layer_major512":tiled_terminal,"canonical8x64":canonical_terminal},
        "scalar_candidate_policy":"existing_BF16_F32_canonical_matvec_512_rows_no_Q8_activation",
        "layer_timings":timings,"timing_scope":"diagnostic_per_layer_sync_excludes_capture_no_TPS_claim",
        "default_numerical_policy_changed":false,
        "artifacts":artifacts,"generation_executed":false,"numerical_policy_changed":false,
        "semantic_qualification":false,"http_continuation":false,"product_ready":false});
    dump(
        directory,
        "manifest.json",
        &serde_json::to_vec_pretty(&report).map_err(|e| invalid(e.to_string()))?,
    )?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ablation_policies_are_explicit_unique_and_no_default_changes() {
        let p = ablations();
        let names = p
            .iter()
            .map(|(n, _)| *n)
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(names.len(), 6);
        assert_eq!(p[0].1, PREFILL8);
        assert!(!p[1].1.native_ffn && !p[1].1.native_attention_projection);
        assert!(p[1].1.canonical_scalars && p[1].1.canonical_norm && p[1].1.canonical_full_core);
        assert_eq!(
            p[2].1,
            Policy {
                canonical_scalars: false,
                ..CANONICAL
            }
        );
        assert_eq!(
            p[3].1,
            Policy {
                canonical_norm: false,
                ..CANONICAL
            }
        );
        assert_eq!(
            p[4].1,
            Policy {
                canonical_full_core: false,
                ..CANONICAL
            }
        );
        assert_eq!(
            p[5].1,
            Policy {
                canonical_scalars: true,
                ..PREFILL8
            }
        );
    }
}
