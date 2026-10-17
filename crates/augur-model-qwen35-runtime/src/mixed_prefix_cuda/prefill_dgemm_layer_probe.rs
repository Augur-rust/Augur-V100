//! Same-binary, real-weight layer-three oracle for FP64 DGEMM attention.
//! This is not a whole-model semantic or request performance qualification.
use super::*;
use serde_json::{json, Value};

#[derive(Clone, Debug)]
pub struct Qwen35PrefillDgemmLayerProbeConfig {
    pub gguf_path: PathBuf,
    pub config_path: PathBuf,
    pub tokens: usize,
    pub source_commit: String,
}

fn geometry(tokens: usize) -> Result<(usize, usize), Qwen35MixedPrefixCudaError> {
    if tokens == 0 || tokens > 32_768 {
        return Err(invalid("DGEMM layer probe tokens must be 1..32768"));
    }
    Ok((tokens.min(512), tokens.div_ceil(512) * 512))
}

fn compare(name: &str, actual: &[u8], expected: &[u8]) -> Value {
    let mismatches = actual.iter().zip(expected).filter(|(a, b)| a != b).count();
    json!({"name":name,"bytes":actual.len(),"reference_bytes":expected.len(),
        "byte_mismatches":mismatches,"actual_sha256":sha256_hex(actual),
        "reference_sha256":sha256_hex(expected),
        "exact":actual.len()==expected.len() && mismatches==0})
}

fn read_view(
    stream: &OwnedCudaStream,
    view: CudaBufferView,
    receipt: CudaStreamSubmissionReceipt,
) -> Result<Vec<u8>, Qwen35MixedPrefixCudaError> {
    let mut bytes = vec![0; view.len_bytes()];
    view.copy_to_host_sync_with_submission_proof(stream, &mut bytes, receipt)?;
    Ok(bytes)
}

pub fn run_qwen35_prefill_dgemm_layer_probe(
    config: &Qwen35PrefillDgemmLayerProbeConfig,
) -> Result<Value, Qwen35MixedPrefixCudaError> {
    let (chunk, capacity) = geometry(config.tokens)?;
    if config.source_commit.len() != 40
        || !config.source_commit.bytes().all(|b| b.is_ascii_hexdigit())
    {
        return Err(invalid("DGEMM layer probe requires a full source SHA"));
    }
    let config_bytes = read_stable_regular_file(&config.config_path, "Qwen config")?;
    let model = Qwen35Config::parse(&config_bytes)?;
    model.validate_exact_profile()?;
    let manifest = GgufManifest::inspect(&config.gguf_path, GgufAuditOptions::default())?;
    Qwen35TensorSchema::validate_manifest_allowing_mtp(&model, &manifest)?;
    let properties = cuda_device_properties(0)?;
    if properties.compute_capability.major != 7 || properties.compute_capability.minor != 0 {
        return Err(invalid("DGEMM layer probe requires SM70"));
    }
    let binary = std::env::current_exe().map_err(Qwen35Error::from)?;
    let binary_hash = sha256_hex(&std::fs::read(&binary).map_err(Qwen35Error::from)?);
    let mut reader = manifest.open_tensor_reader(&config.gguf_path)?;
    let embedding = manifest
        .tensor(linear::EMBEDDING)
        .ok_or_else(|| invalid("DGEMM layer probe embedding"))?
        .clone();
    let token_ids: Vec<_> = (0..config.tokens)
        .map(|i| 1 + (i * 7919) % (model.vocab_size - 1))
        .collect();
    let mut hidden = Vec::with_capacity(config.tokens * linear::HIDDEN);
    for id in &token_ids {
        hidden.extend(linear::read_embedding_row(&mut reader, &embedding, *id)?);
    }
    let allocation_start = cuda_device_allocation_snapshot();
    let mut stream = OwnedCudaStream::create(0)?;
    // Owners are outside the fallible closure: drain precedes their destruction
    // on both success and partially submitted error paths.
    let mut owner: Option<FullPlan> = None;
    let mut workspace_owner: Option<PrefillWorkspace> = None;
    let mut scratch_owner: Option<OwnedCudaBuffer> = None;
    let mut attention_capture: Option<OwnedCudaBuffer> = None;
    let result = (|| -> Result<Value, Qwen35MixedPrefixCudaError> {
        let materials = load_layer_materials(&mut reader, &manifest, &model, 4)?;
        let material = match &materials[3] {
            LayerMaterial::Full(v) => v,
            _ => return Err(invalid("DGEMM layer probe full layer three")),
        };
        owner = Some(build_full_plan(
            &stream,
            material,
            &model,
            capacity,
            config.tokens,
            None,
        )?);
        drop(materials);
        workspace_owner = Some(PrefillWorkspace::new_with_rope_capacity(
            &stream, chunk, capacity,
        )?);
        let workspace = workspace_owner.as_ref().unwrap();
        workspace.upload_rope_tables(&stream, config.tokens, 0, model.rope_theta as f32)?;
        let scratch_elements =
            CudaQwen35PrefillKv8ChunkAttentionV1::dgemm_score_elements(chunk, capacity)?;
        let scratch_bytes = scratch_elements * 4;
        scratch_owner = Some(OwnedCudaBuffer::allocate(0, scratch_bytes + 512)?);
        let scratch = scratch_owner.as_ref().unwrap();
        attention_capture = Some(OwnedCudaBuffer::allocate(
            0,
            chunk * full::QUERY_VALUES * 4,
        )?);
        let saved_attention = attention_capture.as_ref().unwrap();
        scratch.copy_from_host_sync(&stream, &vec![0xa5; scratch_bytes + 512])?;
        let plan = owner.as_mut().unwrap();
        let mut schedule = Vec::new();
        for position in (0..config.tokens).step_by(chunk) {
            let rows = chunk.min(config.tokens - position);
            let mut op = workspace.build_full_ops(
                &stream,
                plan,
                3,
                rows,
                position,
                model.rms_norm_eps as f32,
            )?;
            op.simt_shared_softmax = true;
            let elements =
                CudaQwen35PrefillKv8ChunkAttentionV1::dgemm_score_elements(rows, capacity)?;
            let scores = CudaF32View::new(
                scratch.view(256, elements * 4)?,
                PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                elements,
            )?;
            let candidate = op
                .attention
                .with_dgemm_scores(&stream, &workspace.cublas, scores)?;
            schedule.push((position, rows, op, candidate));
        }
        let start = OwnedCudaEvent::create(0)?;
        let stop = OwnedCudaEvent::create(0)?;
        let mut reference_rows = Vec::new();
        let mut reference_cache: Vec<Vec<u8>> = Vec::new();
        let mut proofs = Vec::new();
        let mut timings = Vec::new();
        let mut completion_proven = true;
        let mut no_execution_allocations = true;
        // Independent reset, identical embedding stream, identical projections
        // and FFN. Only the attention implementation differs.
        for candidate in [false, true] {
            plan.reset(&stream)?;
            for (position, rows, op, alternative) in &schedule {
                let input = f32_bytes(
                    &hidden[position * linear::HIDDEN..(position + rows) * linear::HIDDEN],
                );
                workspace.hidden_b.copy_from_host_sync(&stream, &input)?;
                let alloc_before = cuda_device_allocation_snapshot();
                let boundary_before = cuda_host_boundary_snapshot();
                start.record_for_submission(&stream, stream.submission_receipt()?)?;
                let wall = Instant::now();
                op.submit_projection_prefix(&stream, &workspace.cublas)?;
                if candidate {
                    alternative.submit_tiled_dgemm(&stream, &workspace.cublas)?;
                } else {
                    op.attention.submit_shared_softmax(&stream)?;
                }
                // SAFETY: same-stream copy precedes FFN reuse of ffn_gate;
                // both owners remain alive through terminal proof or error drain.
                unsafe {
                    saved_attention.copy_from_device_async_unchecked(
                        &stream,
                        0,
                        workspace.ffn_gate.view(0, rows * full::QUERY_VALUES * 4)?,
                    )?;
                }
                op.submit_post_attention(&stream, &workspace.cublas)?;
                let receipt = stream.submission_receipt()?;
                stop.record_for_submission(&stream, receipt)?;
                let mut polls = 0_u64;
                loop {
                    if let Some(completion) = stream.query_event_completion(&stop, receipt)? {
                        completion_proven &= completion.covers_submission(receipt)
                            && completion.belongs_to_stream(&stream)?;
                        break;
                    }
                    polls += 1;
                    if polls >= 50_000_000 {
                        return Err(Qwen35MixedPrefixCudaError::CompletionPollLimit);
                    }
                    bounded_completion_poll_backoff(polls);
                }
                let wall_ms = wall.elapsed().as_secs_f64() * 1000.0;
                let allocation =
                    allocation_interval_report(alloc_before, cuda_device_allocation_snapshot())?;
                no_execution_allocations &= allocation.pass;
                timings.push(json!({"candidate":candidate,"position":position,"rows":rows,
                    "cuda_ms":start.elapsed_ms(&stop)?,"host_wall_ms":wall_ms,
                    "allocation":allocation,"host_boundaries":boundary_delta(boundary_before, cuda_host_boundary_snapshot())?}));
                let buffers = [
                    (
                        "attention",
                        saved_attention.view(0, rows * full::QUERY_VALUES * 4)?,
                    ),
                    (
                        "final_hidden",
                        workspace.hidden_a.view(0, rows * linear::HIDDEN * 4)?,
                    ),
                ];
                let mut outputs = Vec::new();
                for (name, view) in buffers {
                    let bytes = read_view(&stream, view, receipt)?;
                    if !f32_vec(&bytes).iter().all(|x| x.is_finite()) {
                        return Err(invalid("nonfinite real layer output"));
                    }
                    if candidate {
                        let reference: &Vec<Vec<u8>> = &reference_rows[position / chunk];
                        proofs.push(compare(
                            &format!("position{position}.{name}"),
                            &bytes,
                            &reference[outputs.len()],
                        ));
                    }
                    outputs.push(bytes);
                }
                if !candidate {
                    reference_rows.push(outputs);
                }
            }
            let receipt = stream.submission_receipt()?;
            stream.synchronize_submission(receipt)?;
            for (i, (name, buffer)) in [
                ("key_cache", &plan.key_cache),
                ("value_cache", &plan.value_cache),
            ]
            .into_iter()
            .enumerate()
            {
                let bytes = read_view(&stream, buffer.view(0, buffer.len_bytes())?, receipt)?;
                if candidate {
                    proofs.push(compare(name, &bytes, &reference_cache[i]));
                } else {
                    reference_cache.push(bytes);
                }
            }
        }
        let mut guards = true;
        for offset in [0, scratch_bytes + 256] {
            let mut bytes = [0; 256];
            scratch
                .view(offset, 256)?
                .copy_to_host_sync(&stream, &mut bytes)?;
            guards &= bytes.iter().all(|b| *b == 0xa5);
        }
        let exact = proofs.iter().all(|p| p["exact"] == true);
        Ok(json!({"schema":"qwen35-prefill-dgemm-real-layer-oracle-v1",
            "source_commit":config.source_commit,"binary_sha256":binary_hash,
            "gguf_sha256":manifest.file_sha256,"config_sha256":sha256_hex(&config_bytes),
            "layer":3,"tokens":config.tokens,"chunk":chunk,"capacity":capacity,
            "input":"real_embedding_rows_deterministic_vocabulary_ids_not_whole_model_hidden",
            "token_ids":token_ids,"input_hidden_sha256":sha256_hex(&f32_bytes(&hidden)),
            "scratch_bytes":scratch_bytes,"scratch_guards_preserved":guards,
            "exact":exact,"owner_epoch_proven":completion_proven,
            "execution_allocation_unchanged":no_execution_allocations,"proofs":proofs,"timings":timings,
            "pass":exact && completion_proven && guards && no_execution_allocations,
            "timing_scope":"single_pair_layer_with_attention_D2D_capture_not_performance_qualification",
            "model_semantics_qualified":false,"product_ready":false,"performance_claim_permitted":false}))
    })();
    let drain = stream.synchronize();
    drop(workspace_owner);
    drop(scratch_owner);
    drop(attention_capture);
    drop(owner);
    let close = stream.close();
    drain?;
    close?;
    let cleanup = allocation_delta(allocation_start, cuda_device_allocation_snapshot())?;
    let zero = cleanup.live_allocations == allocation_start.live_allocations
        && cleanup.live_bytes == allocation_start.live_bytes
        && cleanup.successful_allocations == cleanup.successful_frees;
    let mut report = result?;
    report["cleanup"] =
        serde_json::to_value(cleanup).map_err(|_| invalid("probe cleanup serialization"))?;
    report["cleanup_zero"] = json!(zero);
    report["pass"] = json!(report["pass"] == true && zero);
    report["status"] = json!(if report["pass"] == true {
        "PASS_COMPONENT_REAL_LAYER_BYTE_EXACT"
    } else {
        "NO_GO_COMPONENT_REAL_LAYER"
    });
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_geometry_covers_key_and_row_tail() {
        assert!(geometry(0).is_err());
        assert!(geometry(32769).is_err());
        assert_eq!(geometry(1).unwrap(), (1, 512));
        assert_eq!(geometry(8209).unwrap(), (512, 8704));
        assert_eq!(geometry(32768).unwrap(), (512, 32768));
    }
    #[test]
    fn exact_comparison_rejects_drift_and_different_extent() {
        assert_eq!(compare("x", &[1, 2], &[1, 2])["exact"], true);
        assert_eq!(compare("x", &[1, 3], &[1, 2])["exact"], false);
        assert_eq!(compare("x", &[1], &[1, 2])["exact"], false);
        assert_eq!(compare("x", &[1, 2, 3], &[1, 2])["exact"], false);
    }
}
