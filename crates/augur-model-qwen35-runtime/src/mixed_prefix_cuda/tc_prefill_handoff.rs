//! Independent retained reference, paired teacher forcing, and candidate-owned
//! device handoff. This module is not a default decoder or a publication owner.
use super::*;

#[derive(Clone, Debug)]
pub struct Qwen35TcPrefillHandoffConfig {
    pub base: Qwen35AutoregressiveCudaConfig,
    pub candidate_chunk_tokens: usize,
    pub direct_activation: bool,
    pub sequential_reference: bool,
    pub canonical_seed_tokens: usize,
    pub simt_shared_softmax: bool,
    pub simt_shared_gdn_scalars: bool,
    pub simt_unit_previous_scale: bool,
    pub quad_weight_decode: bool,
    pub wide_prefill_microbatch: bool,
    pub wide_prefill_microbatch_limit: usize,
    /// Research-only precision guard: cap optimized SIMT prefixes above 512
    /// tokens to the known-safe 512-token candidate chunk.
    pub adaptive_safe_width: bool,
}

impl Default for Qwen35TcPrefillHandoffConfig {
    fn default() -> Self {
        Self {
            base: Qwen35AutoregressiveCudaConfig {
                generation_tokens: 8,
                warmup: 0,
                ..Default::default()
            },
            candidate_chunk_tokens: 1024,
            direct_activation: false,
            sequential_reference: false,
            canonical_seed_tokens: 0,
            simt_shared_softmax: false,
            simt_shared_gdn_scalars: false,
            simt_unit_previous_scale: false,
            quad_weight_decode: false,
            wide_prefill_microbatch: false,
            wide_prefill_microbatch_limit: 2048,
            adaptive_safe_width: false,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct TcHandoffTokenComparison {
    pub position: usize,
    pub teacher_forced_input_token_id: usize,
    pub reference_argmax: usize,
    pub candidate_argmax: usize,
    pub logits: TrajectorySampleReport,
    pub hidden: TrajectorySampleReport,
    pub reference_top1_margin: f64,
    pub max_abs_logits_error: f64,
    pub argmax_stability_certified: bool,
    pub pass: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct TcHandoffResources {
    pub setup: BoundaryReport,
    pub reference_free: BoundaryReport,
    pub candidate_free: BoundaryReport,
    pub reference_teacher_forcing: BoundaryReport,
    pub candidate_teacher_forcing: BoundaryReport,
    pub candidate_device_handoff: BoundaryReport,
    pub cleanup: BoundaryReport,
    pub total: BoundaryReport,
    pub boundary_ledger_closed: bool,
    pub resident_allocations: AllocationReport,
    pub handoff_allocation_interval: AllocationIntervalReport,
    pub allocation_cleanup: AllocationCleanupReport,
}

#[derive(Clone, Debug, Serialize)]
pub struct Qwen35TcPrefillHandoffReport {
    pub schema: String,
    pub status: String,
    pub component_only: bool,
    pub source_commit: String,
    pub binary_path: String,
    pub binary_sha256: String,
    pub device: DeviceReport,
    pub gguf_path: String,
    pub gguf_file_sha256: Option<String>,
    pub config_sha256: String,
    pub tokenizer_sha256: String,
    pub prompt_text_sha256: String,
    pub prompt_token_ids: Vec<usize>,
    pub capacity: usize,
    pub final_position_extent: usize,
    pub reference_policy: String,
    pub candidate_chunk_tokens: usize,
    pub requested_candidate_chunk_tokens: usize,
    pub adaptive_safe_width: bool,
    pub adaptive_safe_width_applied: bool,
    pub candidate_tensor_core_prefill: bool,
    pub canonical_seed_tokens_requested: usize,
    pub simt_seed_rows_per_full_layer: usize,
    pub retained_tc_output_rows_per_full_layer: usize,
    pub seed_uses_candidate_query_and_kv_only: bool,
    pub direct_activation_policy: Option<PrefillActivationPolicyComparisonReport>,
    pub reference: AutoregressiveSequenceReport,
    pub candidate: AutoregressiveSequenceReport,
    pub reference_prefill_ms: f64,
    pub candidate_prefill_ms: f64,
    pub reference_generated_token_ids: Vec<usize>,
    pub candidate_generated_token_ids: Vec<usize>,
    pub reference_generated_text: String,
    pub candidate_generated_text: String,
    pub free_running_token_ids_equal: bool,
    pub teacher_forced_input_token_ids: Vec<usize>,
    pub teacher_forced_outputs_are_not_visible_generation: bool,
    pub reference_teacher_forced_replay_exact: bool,
    pub teacher_forced_comparison: Vec<TcHandoffTokenComparison>,
    pub terminal_state_comparison: Vec<TrajectorySampleReport>,
    pub terminal_state_aggregate: TrajectoryAggregateReport,
    pub numerical_contract: PrefillChunkedNumericalContract,
    pub cross_policy_numerical_pass: bool,
    pub device_handoff: DeviceTokenHandoffReport,
    pub candidate_resident_prefill_plus_generation_ms: f64,
    pub setup_and_teacher_forcing_excluded_from_resident_timing: bool,
    pub resources: TcHandoffResources,
    pub all_component_gates_pass: bool,
    pub external_llama_parity_established: bool,
    pub semantic_parity_established: bool,
    pub product_tps_permitted: bool,
    pub default_cli_enabled: bool,
}

fn validate_config(
    config: &Qwen35TcPrefillHandoffConfig,
    candidate_tensor_core: bool,
) -> Result<(), Qwen35MixedPrefixCudaError> {
    let b = &config.base;
    let valid_chunk = if candidate_tensor_core {
        [512, 1024].contains(&config.candidate_chunk_tokens)
    } else if config.wide_prefill_microbatch {
        config.candidate_chunk_tokens > 0
            && config.candidate_chunk_tokens <= config.wide_prefill_microbatch_limit
    } else {
        config.candidate_chunk_tokens == 512
    };
    let valid_seed = if candidate_tensor_core {
        [0, 32, 64, 128].contains(&config.canonical_seed_tokens)
    } else {
        config.canonical_seed_tokens == 0
    };
    if candidate_tensor_core
        && (config.simt_shared_softmax
            || config.simt_shared_gdn_scalars
            || config.simt_unit_previous_scale
            || config.quad_weight_decode
            || config.wide_prefill_microbatch)
    {
        return Err(invalid("TC handoff cannot enable SIMT candidate tuning"));
    }
    if config.simt_unit_previous_scale && !config.simt_shared_softmax {
        return Err(invalid("unit previous scale requires shared SIMT softmax"));
    }
    if config.wide_prefill_microbatch
        && (!false
            || !matches!(config.wide_prefill_microbatch_limit, 2048 | 4096))
    {
        return Err(invalid("wide SIMT handoff feature/limit"));
    }
    if !config.wide_prefill_microbatch && config.wide_prefill_microbatch_limit != 2048 {
        return Err(invalid("wide SIMT limit requires explicit opt-in"));
    }
    if !valid_chunk
        || !valid_seed
        || b.capacity == 0
        || b.capacity > 4096
        || b.device_ordinal < 0
        || b.generation_tokens == 0
        || b.generation_tokens > 32
        || b.warmup != 0
        || b.prompt_text.is_empty()
        || b.prompt_text.len() > 262_144
        || b.prompt_repeat > 4096
        || b.profile_device_handoff_decode
        || b.profile_device_handoff_layer_timing
        || b.prefill_shared_softmax
    {
        return Err(invalid(
            "prefill handoff bounded config; warmup/profile modes are not admitted",
        ));
    }
    let repeated_bytes = b
        .prompt_text
        .len()
        .checked_add(1)
        .and_then(|n| n.checked_mul(b.prompt_repeat.max(1)))
        .ok_or_else(|| invalid("prefill handoff prompt byte overflow"))?;
    if repeated_bytes > 1_048_576 {
        return Err(invalid("prefill handoff repeated prompt byte limit"));
    }
    Ok(())
}

fn phase<T>(
    run: impl FnOnce() -> Result<T, Qwen35MixedPrefixCudaError>,
) -> Result<(T, BoundaryReport), Qwen35MixedPrefixCudaError> {
    let before = cuda_host_boundary_snapshot();
    let value = run()?;
    Ok((
        value,
        boundary_delta(before, cuda_host_boundary_snapshot())?,
    ))
}

fn seed_geometry(
    prefix: usize,
    requested: usize,
) -> Result<(usize, usize), Qwen35MixedPrefixCudaError> {
    if prefix == 0 || ![0, 32, 64, 128].contains(&requested) {
        return Err(invalid("TC handoff canonical seed geometry"));
    }
    let seed = prefix.min(requested);
    Ok((seed, prefix - seed))
}



struct ForcedAudit {
    logits: Vec<Vec<f32>>,
    hidden: Vec<Vec<f32>>,
    states: Vec<(String, Vec<f32>)>,
    proof: bool,
}

fn validate_forced_inputs(
    prompt: &[usize],
    inputs: &[usize],
    vocab: usize,
) -> Result<(), Qwen35MixedPrefixCudaError> {
    if prompt.len() < 2
        || inputs.is_empty()
        || inputs.len() > 32
        || prompt.last() != inputs.first()
        || inputs.iter().any(|id| *id >= vocab)
    {
        return Err(invalid("TC handoff teacher-forced input identity/extent"));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn forced_audit(
    plans: &mut [DevicePlan],
    output: &DeviceOutputHeadPlan,
    reader: &mut GgufTensorReader,
    embedding: &GgufTensor,
    prompt: &[usize],
    inputs: &[usize],
    stream: &OwnedCudaStream,
    mode: &AutoregressivePrefillMode,
    vocab: usize,
) -> Result<ForcedAudit, Qwen35MixedPrefixCudaError> {
    validate_forced_inputs(prompt, inputs, vocab)?;
    reset_plans(plans, stream)?;
    let start = OwnedCudaEvent::create(stream.device_ordinal())?;
    let stop = OwnedCudaEvent::create(stream.device_ordinal())?;
    let terminal = OwnedCudaEvent::create(stream.device_ordinal())?;
    let prefill = run_autoregressive_prefill(
        plans, mode, reader, embedding, prompt, stream, &start, &stop, &terminal,
    )?;
    let mut audit = ForcedAudit {
        logits: Vec::new(),
        hidden: Vec::new(),
        states: Vec::new(),
        proof: prefill.proof,
    };
    for (index, &token) in inputs.iter().enumerate() {
        let hidden = linear::read_embedding_row(reader, embedding, token)?;
        let forward = submit_autoregressive_token(
            plans,
            Some(output),
            &hidden,
            stream,
            prompt.len() - 1 + index,
            &start,
            &stop,
            &terminal,
        )?;
        audit.proof &= forward.proof;
        audit
            .logits
            .push(read_f32_buffer(stream, &output.logits, forward.receipt)?);
        audit.hidden.push(read_f32_buffer(
            stream,
            plans
                .last()
                .ok_or_else(|| invalid("missing terminal layer"))?
                .final_hidden_buffer(),
            forward.receipt,
        )?);
        if index + 1 == inputs.len() {
            audit.states = read_prefill_backbone_states(plans, stream, forward.receipt)?;
        }
    }
    Ok(audit)
}

fn compare_token(
    position: usize,
    input: usize,
    logits: &[f32],
    expected: &[f32],
    hidden: &[f32],
    expected_hidden: &[f32],
    candidate_tensor_core: bool,
) -> Result<TcHandoffTokenComparison, Qwen35MixedPrefixCudaError> {
    if logits.is_empty()
        || logits.len() != expected.len()
        || hidden.is_empty()
        || hidden.len() != expected_hidden.len()
    {
        return Err(invalid("prefill handoff comparison extent"));
    }
    let candidate_argmax =
        greedy_argmax(logits, logits.len()).map_err(|e| invalid(e.to_string()))? as usize;
    let reference_argmax =
        greedy_argmax(expected, expected.len()).map_err(|e| invalid(e.to_string()))? as usize;
    let margin = top1_margin_for_argmax(expected, reference_argmax);
    let max_abs = logits
        .iter()
        .zip(expected)
        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
        .fold(0.0_f64, f64::max);
    let logits = trajectory_sample("teacher_forced_logits", logits, expected);
    let hidden = trajectory_sample("teacher_forced_hidden", hidden, expected_hidden);
    let contract = PrefillChunkedNumericalContract::for_attention_path(candidate_tensor_core);
    let certified = margin.is_finite() && max_abs.is_finite() && margin > 2.0 * max_abs;
    let pass = candidate_argmax == reference_argmax
        && certified
        && contract.logits_pass(&logits)
        && prefill_trajectory_pass(
            &hidden,
            contract.final_hidden_relative_l2_max,
            contract.final_hidden_cosine_min,
        );
    Ok(TcHandoffTokenComparison {
        position,
        teacher_forced_input_token_id: input,
        reference_argmax,
        candidate_argmax,
        logits,
        hidden,
        reference_top1_margin: margin,
        max_abs_logits_error: max_abs,
        argmax_stability_certified: certified,
        pass,
    })
}

pub fn run_qwen35_tc_prefill_handoff_cuda(
    config: &Qwen35TcPrefillHandoffConfig,
) -> Result<Qwen35TcPrefillHandoffReport, Qwen35MixedPrefixCudaError> {
    run_prefill_handoff_with_policy(config, true)
}







fn run_prefill_handoff_with_policy(
    config: &Qwen35TcPrefillHandoffConfig,
    candidate_tensor_core: bool,
) -> Result<Qwen35TcPrefillHandoffReport, Qwen35MixedPrefixCudaError> {
    validate_config(config, candidate_tensor_core)?;
    let b = &config.base;
    let binary_path = std::env::current_exe().map_err(Qwen35Error::from)?;
    let binary_sha256 = sha256_hex(&std::fs::read(&binary_path).map_err(Qwen35Error::from)?);
    let config_bytes = read_stable_regular_file(&b.config_path, "Qwen config")?;
    let config_sha256 = sha256_hex(&config_bytes);
    let model = Qwen35Config::parse(&config_bytes)?;
    model.validate_exact_profile()?;
    let tokenizer_bytes = read_stable_regular_file(&b.tokenizer_path, "Qwen tokenizer")?;
    let tokenizer = Tokenizer::from_bytes(&tokenizer_bytes).map_err(|e| invalid(e.to_string()))?;
    if tokenizer.get_vocab_size(false) > model.vocab_size {
        return Err(invalid("tokenizer/model vocabulary"));
    }
    let text = if b.prompt_repeat == 0 {
        b.prompt_text.clone()
    } else {
        format!("{} ", b.prompt_text)
            .repeat(b.prompt_repeat)
            .trim_end()
            .to_owned()
    };
    let prompt_text_sha256 = sha256_hex(text.as_bytes());
    let prompt: Vec<usize> = tokenizer
        .encode(text.as_str(), false)
        .map_err(|e| invalid(e.to_string()))?
        .get_ids()
        .iter()
        .map(|x| *x as usize)
        .collect();
    let extent = prompt
        .len()
        .checked_sub(1)
        .and_then(|n| n.checked_add(b.generation_tokens))
        .ok_or_else(|| invalid("prompt/suffix overflow"))?;
    if prompt.len() < 2
        || prompt.iter().any(|id| *id >= model.vocab_size)
        || extent > b.capacity
        || extent > model.max_position_embeddings
    {
        return Err(invalid(
            "TC handoff prompt needs >=2 tokens within admitted capacity",
        ));
    }
    let requested_candidate_chunk_tokens = config.candidate_chunk_tokens;
    let adaptive_safe_width_applied = !candidate_tensor_core
        && config.adaptive_safe_width
        && prompt.len() > 512
        && config.candidate_chunk_tokens > 512;
    let effective_candidate_chunk_tokens = if adaptive_safe_width_applied {
        512
    } else {
        config.candidate_chunk_tokens
    };
    let manifest = GgufManifest::inspect(&b.gguf_path, GgufAuditOptions::default())?;
    let schema = Qwen35TensorSchema::validate_manifest_allowing_mtp(&model, &manifest)?;
    if schema.actual_tensor_count < 1187 || model.num_hidden_layers != FULL_TARGET_LAYER_COUNT {
        return Err(invalid("TC handoff model profile"));
    }
    let properties = cuda_device_properties(b.device_ordinal)?;
    if properties.compute_capability.major != 7 || properties.compute_capability.minor != 0 {
        return Err(invalid("TC handoff requires SM70"));
    }
    let device = DeviceReport {
        ordinal: properties.ordinal,
        name: properties.name,
        compute_capability: [7, 0],
        total_global_memory_bytes: properties.total_global_memory_bytes,
    };
    let mut reader = manifest.open_tensor_reader(&b.gguf_path)?;
    let embedding = manifest
        .tensor("token_embd.weight")
        .ok_or_else(|| invalid("embedding descriptor"))?
        .clone();
    let first_hidden = linear::read_embedding_row(&mut reader, &embedding, prompt[0])?;
    let allocation_start = cuda_device_allocation_snapshot();
    let boundary_start = cuda_host_boundary_snapshot();
    let stream = OwnedCudaStream::create(b.device_ordinal)?;
    let materials = load_layer_materials(&mut reader, &manifest, &model, FULL_TARGET_LAYER_COUNT)?;
    let head_material = load_output_head_material(&mut reader, &manifest, &model)?;
    let mut plans: Vec<DevicePlan> = Vec::with_capacity(FULL_TARGET_LAYER_COUNT);
    for material in &materials {
        let input = plans
            .last()
            .map(DevicePlan::final_hidden_f32_view)
            .transpose()?;
        match material {
            LayerMaterial::Linear(material) => {
                let projection =
                    project_linear(material, &first_hidden, model.rms_norm_eps as f32)?;
                plans.push(DevicePlan::Linear(build_linear_plan(
                    &stream,
                    material,
                    &first_hidden,
                    &projection,
                    true,
                    input,
                )?));
            }
            LayerMaterial::Full(material) => plans.push(DevicePlan::Full(build_full_plan(
                &stream, material, &model, b.capacity, extent, input,
            )?)),
        }
    }
    let output = build_device_output_head_plan(
        &stream,
        &head_material,
        plans.last().unwrap().final_hidden_view()?,
        model.hidden_size,
        model.vocab_size,
        model.rms_norm_eps as f32,
    )?;
    drop(head_material);
    drop(materials);
    let handoff = build_device_token_handoff_plan(
        &stream,
        &mut reader,
        &embedding,
        &output,
        model.hidden_size,
        model.vocab_size,
        b.generation_tokens,
        &b.source_commit,
        &binary_sha256,
        &config_sha256,
        manifest.file_sha256.as_deref(),
        &prompt_text_sha256,
        b.capacity,
    )?;
    let mut reference_mode = build_autoregressive_prefill_mode(
        !config.sequential_reference,
        &stream,
        &plans,
        &mut reader,
        &embedding,
        &prompt,
        &model,
        b.capacity,
        false,
    )?;
    let mut candidate_mode = build_autoregressive_prefill_mode_for_policy(
        true,
        &stream,
        &plans,
        &mut reader,
        &embedding,
        &prompt,
        &model,
        b.capacity,
        effective_candidate_chunk_tokens,
        candidate_tensor_core,
        config.direct_activation,
        config.simt_shared_softmax,
        config.simt_shared_gdn_scalars,
        config.simt_unit_previous_scale,
        config.quad_weight_decode,
        config.wide_prefill_microbatch,
        config.wide_prefill_microbatch_limit,
    )?;
    
    let (seed_rows, tc_rows) = if candidate_tensor_core {
        seed_geometry(prompt.len() - 1, config.canonical_seed_tokens)?
    } else {
        (0, prompt.len() - 1)
    };
    let mut setup = boundary_delta(boundary_start, cuda_host_boundary_snapshot())?;
    let resident_allocations =
        allocation_delta(allocation_start, cuda_device_allocation_snapshot())?;
    let (reference, reference_free) = phase(|| {
        reset_plans(&mut plans, &stream)?;
        run_autoregressive_sequence(
            &mut plans,
            &output,
            &mut reader,
            &embedding,
            &prompt,
            b.generation_tokens,
            &stream,
            true,
            &mut reference_mode,
            None,
            None,
            false,
            false,
        )
    })?;
    let (candidate, candidate_free) = phase(|| {
        reset_plans(&mut plans, &stream)?;
        run_autoregressive_sequence(
            &mut plans,
            &output,
            &mut reader,
            &embedding,
            &prompt,
            b.generation_tokens,
            &stream,
            true,
            &mut candidate_mode,
            None,
            None,
            false,
            false,
        )
    })?;
    // `run_autoregressive_sequence` deliberately releases its prefill mode
    // workspace before decode.  The teacher-forcing audit must therefore use
    // fresh modes with the same policy; reusing the released modes silently
    // changes the reference/candidate path to sequential prefill and breaks
    // the free-running replay identity.
    let audit_reference_mode_before = cuda_host_boundary_snapshot();
    let reference_audit_mode = build_autoregressive_prefill_mode(
        !config.sequential_reference,
        &stream,
        &plans,
        &mut reader,
        &embedding,
        &prompt,
        &model,
        b.capacity,
        false,
    )?;
    let mut candidate_audit_mode = build_autoregressive_prefill_mode_for_policy(
        true,
        &stream,
        &plans,
        &mut reader,
        &embedding,
        &prompt,
        &model,
        b.capacity,
        effective_candidate_chunk_tokens,
        candidate_tensor_core,
        config.direct_activation,
        config.simt_shared_softmax,
        config.simt_shared_gdn_scalars,
        config.simt_unit_previous_scale,
        config.quad_weight_decode,
        config.wide_prefill_microbatch,
        config.wide_prefill_microbatch_limit,
    )?;
    
    setup = add_boundaries(
        setup,
        boundary_delta(audit_reference_mode_before, cuda_host_boundary_snapshot())?,
    );

    let inputs = reference
        .tokens
        .iter()
        .map(|t| t.input_token_id)
        .collect::<Vec<_>>();
    let (reference_audit, reference_teacher_forcing) = phase(|| {
        forced_audit(
            &mut plans,
            &output,
            &mut reader,
            &embedding,
            &prompt,
            &inputs,
            &stream,
            &reference_audit_mode,
            model.vocab_size,
        )
    })?;
    let (candidate_audit, candidate_teacher_forcing) = phase(|| {
        forced_audit(
            &mut plans,
            &output,
            &mut reader,
            &embedding,
            &prompt,
            &inputs,
            &stream,
            &candidate_audit_mode,
            model.vocab_size,
        )
    })?;
    let reference_exact = reference
        .tokens
        .iter()
        .zip(&reference_audit.logits)
        .all(|(t, values)| t.logits_sha256 == sha256_f32_bits(values));
    let comparisons = inputs
        .iter()
        .enumerate()
        .map(|(i, &input)| {
            compare_token(
                prompt.len() - 1 + i,
                input,
                &candidate_audit.logits[i],
                &reference_audit.logits[i],
                &candidate_audit.hidden[i],
                &reference_audit.hidden[i],
                candidate_tensor_core,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    if reference_audit.states.len() != 128
        || candidate_audit.states.len() != 128
        || reference_audit
            .states
            .iter()
            .zip(&candidate_audit.states)
            .any(|(r, c)| r.0 != c.0)
    {
        return Err(invalid("TC handoff state inventory"));
    }
    let states = candidate_audit
        .states
        .iter()
        .zip(&reference_audit.states)
        .map(|(a, e)| trajectory_sample(&a.0, &a.1, &e.1))
        .collect::<Vec<_>>();
    let state_aggregate = aggregate_trajectory(states.iter());
    let contract = PrefillChunkedNumericalContract::for_attention_path(candidate_tensor_core);
    let numerical_pass = reference_exact
        && reference_audit.proof
        && candidate_audit.proof
        && comparisons.iter().all(|x| x.pass)
        && states.iter().all(|x| {
            prefill_trajectory_pass(x, contract.state_relative_l2_max, contract.state_cosine_min)
        });
    drop(reference_audit);
    drop(candidate_audit);
    drop(reference_audit_mode);
    drop(candidate_audit_mode);

    // Build one more candidate mode for the measured device handoff.  The
    // free-running candidate has already released its workspace, so the
    // handoff must not accidentally fall back to sequential prefill.
    let handoff_mode_before = cuda_host_boundary_snapshot();
    let mut handoff_mode = build_autoregressive_prefill_mode_for_policy(
        true,
        &stream,
        &plans,
        &mut reader,
        &embedding,
        &prompt,
        &model,
        b.capacity,
        effective_candidate_chunk_tokens,
        candidate_tensor_core,
        config.direct_activation,
        config.simt_shared_softmax,
        config.simt_shared_gdn_scalars,
        config.simt_unit_previous_scale,
        config.quad_weight_decode,
        config.wide_prefill_microbatch,
        config.wide_prefill_microbatch_limit,
    )?;
    
    setup = add_boundaries(
        setup,
        boundary_delta(handoff_mode_before, cuda_host_boundary_snapshot())?,
    );
    let handoff_alloc_start = cuda_device_allocation_snapshot();
    let (handoff_sequence, candidate_device_handoff) = phase(|| {
        reset_plans(&mut plans, &stream)?;
        run_device_token_handoff_sequence(
            &mut plans,
            &output,
            &handoff,
            &mut reader,
            &embedding,
            &prompt,
            b.generation_tokens,
            b.capacity,
            &stream,
            1,
            &mut handoff_mode,
            false,
            false,
            false,
            None,
            None,
            false,
            false,
        )
    })?;
    let handoff_allocation_interval = allocation_lifecycle_release_interval_report(
        handoff_alloc_start,
        cuda_device_allocation_snapshot(),
    )?;
    let handoff_report = device_token_handoff_report(&handoff, &handoff_sequence, &candidate)?;
    let cleanup_start = cuda_host_boundary_snapshot();
    // Every phase has an owner/epoch terminal completion. The explicit close
    // boundary is outside measured handoff and releases all plan/view owners.
    stream.synchronize()?;
    drop(handoff_mode);
    drop(candidate_mode);
    drop(reference_mode);
    drop(handoff);
    drop(output);
    drop(plans);
    drop(stream);
    let cleanup = boundary_delta(cleanup_start, cuda_host_boundary_snapshot())?;
    let allocation_cleanup =
        allocation_cleanup_report(allocation_start, cuda_device_allocation_snapshot())?;
    let total = boundary_delta(boundary_start, cuda_host_boundary_snapshot())?;
    let accounted = [
        setup.clone(),
        reference_free.clone(),
        candidate_free.clone(),
        reference_teacher_forcing.clone(),
        candidate_teacher_forcing.clone(),
        candidate_device_handoff.clone(),
        cleanup.clone(),
    ]
    .into_iter()
    .fold(BoundaryReport::default(), add_boundaries);
    let closed = total == accounted;
    let free_equal = reference.generated_token_ids == candidate.generated_token_ids;
    let pass = numerical_pass
        && free_equal
        && reference.proof
        && candidate.proof
        && handoff_report.complete_numerical_contract
        && handoff_report.measured_boundary_contract_closed
        && handoff_allocation_interval.pass
        && closed
        && allocation_cleanup.pass;
    let decode_text = |ids: &[usize]| {
        tokenizer
            .decode(&ids.iter().map(|id| *id as u32).collect::<Vec<_>>(), false)
            .map_err(|e| invalid(e.to_string()))
    };
    let policy_config = Qwen35PrefillChunkedCudaConfig {
        direct_post_attention_activation: config.direct_activation,
        ..Default::default()
    };
    Ok(Qwen35TcPrefillHandoffReport {
        schema: if candidate_tensor_core {
            "crystal-quant.qwen35.tc-prefill-kv8-decode-handoff.component.v2"
        } else {
            "crystal-quant.qwen35.simt-prefill-kv8-decode-handoff.component.v1"
        }
        .to_owned(),
        status: if pass {
            if candidate_tensor_core {
                "PASS_COMPONENT_TC_PREFILL_HANDOFF"
            } else {
                "PASS_COMPONENT_SIMT_PREFILL_HANDOFF"
            }
        } else {
            if candidate_tensor_core {
                "FAIL_COMPONENT_TC_PREFILL_HANDOFF"
            } else {
                "FAIL_COMPONENT_SIMT_PREFILL_HANDOFF"
            }
        }
        .to_owned(),
        component_only: true,
        source_commit: b.source_commit.clone(),
        binary_path: binary_path.display().to_string(),
        binary_sha256,
        device,
        gguf_path: b.gguf_path.display().to_string(),
        gguf_file_sha256: manifest.file_sha256.clone(),
        config_sha256,
        tokenizer_sha256: sha256_hex(&tokenizer_bytes),
        prompt_text_sha256,
        prompt_token_ids: prompt,
        capacity: b.capacity,
        final_position_extent: extent,
        reference_policy: if config.sequential_reference {
            "independent_sequential_retained"
        } else {
            "independent_SIMT512_retained"
        }
        .to_owned(),
        candidate_chunk_tokens: effective_candidate_chunk_tokens,
        requested_candidate_chunk_tokens,
        adaptive_safe_width: config.adaptive_safe_width,
        adaptive_safe_width_applied,
        candidate_tensor_core_prefill: tc_rows > 0,
        canonical_seed_tokens_requested: config.canonical_seed_tokens,
        simt_seed_rows_per_full_layer: seed_rows,
        retained_tc_output_rows_per_full_layer: tc_rows,
        seed_uses_candidate_query_and_kv_only: candidate_tensor_core,
        direct_activation_policy: prefill_activation_policy_report(
            &policy_config,
            numerical_pass,
            free_equal,
        ),
        reference: autoregressive_sequence_report(&reference),
        candidate: autoregressive_sequence_report(&candidate),
        reference_prefill_ms: reference.prefill_host_wall_ms,
        candidate_prefill_ms: candidate.prefill_host_wall_ms,
        reference_generated_text: decode_text(&reference.generated_token_ids)?,
        candidate_generated_text: decode_text(&candidate.generated_token_ids)?,
        reference_generated_token_ids: reference.generated_token_ids,
        candidate_generated_token_ids: candidate.generated_token_ids,
        free_running_token_ids_equal: free_equal,
        teacher_forced_input_token_ids: inputs,
        teacher_forced_outputs_are_not_visible_generation: true,
        reference_teacher_forced_replay_exact: reference_exact,
        teacher_forced_comparison: comparisons,
        terminal_state_comparison: states,
        terminal_state_aggregate: state_aggregate,
        numerical_contract: contract,
        cross_policy_numerical_pass: numerical_pass,
        candidate_resident_prefill_plus_generation_ms: handoff_report.prefill_host_wall_ms
            + handoff_report.aggregate_decode_host_wall_ms,
        setup_and_teacher_forcing_excluded_from_resident_timing: true,
        device_handoff: handoff_report,
        resources: TcHandoffResources {
            setup,
            reference_free,
            candidate_free,
            reference_teacher_forcing,
            candidate_teacher_forcing,
            candidate_device_handoff,
            cleanup,
            total,
            boundary_ledger_closed: closed,
            resident_allocations,
            handoff_allocation_interval,
            allocation_cleanup,
        },
        all_component_gates_pass: pass,
        external_llama_parity_established: false,
        semantic_parity_established: false,
        product_tps_permitted: false,
        default_cli_enabled: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn seed_geometry_never_promotes_a_full_fallback_as_tc() {
        assert_eq!(seed_geometry(35, 64).unwrap(), (35, 0));
        assert_eq!(seed_geometry(128, 64).unwrap(), (64, 64));
        assert_eq!(seed_geometry(3584, 64).unwrap(), (64, 3520));
        assert_eq!(seed_geometry(35, 0).unwrap(), (0, 35));
        assert!(seed_geometry(0, 64).is_err());
        assert!(seed_geometry(512, 129).is_err());
    }
    #[test]
    fn config_rejects_oversized_and_unmeasured_modes() {
        let mut c = Qwen35TcPrefillHandoffConfig::default();
        c.base.prompt_text = "test".to_owned();
        assert!(validate_config(&c, true).is_ok());
        for width in [0, 256, 1025, usize::MAX] {
            c.candidate_chunk_tokens = width;
            assert!(validate_config(&c, true).is_err());
        }
        c.candidate_chunk_tokens = 1024;
        c.base.capacity = 4097;
        assert!(validate_config(&c, true).is_err());
        c.base.capacity = 4096;
        c.base.generation_tokens = 33;
        assert!(validate_config(&c, true).is_err());
        c.base.generation_tokens = 8;
        c.base.warmup = 1;
        assert!(validate_config(&c, true).is_err());
    }
    #[test]
    fn teacher_forcing_binds_last_prompt_input_and_vocab() {
        assert!(validate_forced_inputs(&[1, 2], &[2, 3], 4).is_ok());
        for inputs in [vec![], vec![1], vec![2, 4], vec![2; 33]] {
            assert!(validate_forced_inputs(&[1, 2], &inputs, 4).is_err());
        }
        assert!(validate_forced_inputs(&[2], &[2], 4).is_err());
    }
    #[test]
    fn comparison_rejects_margin_failure_even_when_argmax_matches() {
        let exact = compare_token(2, 1, &[2.0, 1.0], &[2.0, 1.0], &[1.0], &[1.0], true).unwrap();
        assert!(exact.pass);
        let drift = compare_token(2, 1, &[1.1, 1.0], &[1.01, 1.0], &[1.0], &[1.0], true).unwrap();
        assert_eq!(drift.reference_argmax, drift.candidate_argmax);
        assert!(!drift.pass);
        assert!(!drift.argmax_stability_certified);
    }
    #[test]
    fn comparison_rejects_nonfinite_extent_and_hidden_drift() {
        assert!(compare_token(0, 0, &[], &[1.0], &[1.0], &[1.0], true).is_err());
        assert!(compare_token(0, 0, &[f32::NAN, 1.0], &[2.0, 1.0], &[1.0], &[1.0], true).is_err());
        assert!(
            !compare_token(0, 0, &[2.0, 1.0], &[2.0, 1.0], &[1.1], &[1.0], true)
                .unwrap()
                .pass
        );
    }
}
