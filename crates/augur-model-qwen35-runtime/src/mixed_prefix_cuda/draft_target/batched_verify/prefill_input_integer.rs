//! Default-off B512 input-only companion to the exact integer FFN candidate.
//! No change to B8 history/verify/tails or norms. BF16 alpha/beta is changed only
//! by the separately named B512 llama-BF16 policy, never by the integer recipe.
use super::*;

pub(in super::super) fn requested(
    value: Option<&std::ffi::OsStr>,
    integer_ffn_rows: Option<usize>,
) -> Result<bool> {
    match value {
        None => Ok(false),
        Some(v) if v == "1" && integer_ffn_rows == Some(512) => Ok(true),
        _ => Err(invalid(
            "AUGUR_QWEN35_PREFILL_INPUT_INTEGER requires exactly 1 and B512 integer FFN",
        )),
    }
}

pub(super) const SCRATCH_BYTES: usize = 11_468_800;
pub(super) const ARITHMETIC_POLICY: &str =
    "canonical_nvfp4_q8_ffn_and_q8_input_b512_integer_mma_tile384_v1";
pub(super) const LLAMA_ARITHMETIC_POLICY: &str =
    "canonical_except_prefix_q8_input_b512_llama_f16_f32_v1";
pub(super) const ORIGINAL_ARITHMETIC_POLICY: &str =
    "canonical_except_prefix_original_f32_input_b512_llama_f16_f32_v1";
pub(super) const FULL_ORIGINAL_ARITHMETIC_POLICY: &str =
    "canonical_except_prefix_original_f32_all_projection_b512_llama_postscale_swiglu_v1";

fn relocated_attention_output(full_original: bool, layer: usize, rows: usize) -> bool {
    full_original && rows == 512 && layer < 64 && layer % 4 != 3
}

/// Optional host-serialized boundary witness, never performance evidence.
/// Samples three real rows of the last full B512 prefix block in all64 layers.
pub(in super::super) struct PrefixBoundaryAudit {
    directory: PathBuf,
    batch_start: usize,
    inventory: std::cell::RefCell<Vec<Value>>,
    operand_inventory: std::cell::RefCell<Vec<Value>>,
}

fn original_operand_layer(layer: usize) -> bool {
    matches!(layer, 0 | 31 | 50 | 51 | 63)
}

fn audit_batch_start(prompt: usize, history: usize, admitted: bool) -> Result<usize> {
    let prefix = prompt.checked_sub(history).ok_or_else(|| invalid("prefix witness extent"))?;
    if !admitted || prompt > 16384 || prefix < 512 {
        return Err(invalid("prefix witness requires admitted B512 prefix and <=16Ki real prompt"));
    }
    Ok(prefix / 512 * 512 - 512)
}

impl PrefixBoundaryAudit {
    pub(in super::super) fn from_env(prompt: usize, history: usize, admitted: bool) -> Result<Option<Self>> {
        let Some(value) = std::env::var_os("AUGUR_QWEN35_PREFILL_BOUNDARY_DIR") else { return Ok(None); };
        // The resident owner performs a short warmup before the measured cold
        // request. It has no wide prefix and must not consume the fresh root.
        if admitted && prompt.checked_sub(history).is_some_and(|prefix| prefix < 512) {
            return Ok(None);
        }
        let batch_start = audit_batch_start(prompt, history, admitted)?;
        let directory = PathBuf::from(value);
        if !directory.is_absolute() {
            return Err(invalid("prefix witness requires an absolute fresh directory"));
        }
        std::fs::create_dir(&directory).map_err(Qwen35Error::from)?;
        Ok(Some(Self {directory, batch_start, inventory: std::cell::RefCell::new(Vec::new()),
            operand_inventory: std::cell::RefCell::new(Vec::new())}))
    }

    pub(super) fn capture(&self, d: &Driver, w: &PrefillWorkspace, layer: usize, rows: usize) -> Result<()> {
        if rows != 512 || d.cursor != self.batch_start { return Ok(()); }
        const NAMES: [&str; 7] = ["input_hidden", "attn_norm", "attention_output",
            "attn_residual", "attn_post_norm", "ffn_out", "l_out"];
        if layer >= 64 || self.inventory.borrow().len() != layer * NAMES.len() {
            return Err(invalid("prefix witness layer order or duplicate"));
        }
        let projected_attention = if relocated_attention_output(d.llama_full_original, layer, rows) {
            view(&w.qkv)?
        } else { view(&w.attention_output)? };
        let sources = [w.hidden_input(layer, rows)?.buffer_view(), view(&w.normalized)?,
            projected_attention, view(&w.attention_residual)?, view(&w.post_norm)?,
            view(&w.ffn_output)?, w.hidden_output(layer, rows)?.buffer_view()];
        let receipt = d.stream.submission_receipt()?;
        for (name, source) in NAMES.into_iter().zip(sources) {
            let mut values = Vec::with_capacity(3 * H);
            for row in [0, 255, 511] {
                values.extend(read_f32_view(&d.stream, source.subview(row * H * 4, H * 4)?, receipt)?);
            }
            if values.len() != 3 * H || values.iter().any(|v| !v.is_finite()) {
                return Err(invalid("prefix witness nonfinite or incomplete"));
            }
            let bytes = f32_bytes(&values);
            let filename = format!("layer{layer}-{name}.f32le.bin");
            let mut file = std::fs::OpenOptions::new().write(true).create_new(true)
                .open(self.directory.join(&filename)).map_err(Qwen35Error::from)?;
            std::io::Write::write_all(&mut file, &bytes).map_err(Qwen35Error::from)?;
            self.inventory.borrow_mut().push(json!({"layer":layer,"name":name,"file":filename,
                "rows":[0,255,511],"width":H,"bytes":bytes.len(),"sha256":sha256_hex(&bytes)}));
        }
        // The same completed layer still owns both the original norm output and
        // its packed projection input. Capture full B512 rows, without replay.
        if original_operand_layer(layer) {
            let inputs = [
                ("attn_norm_f32", view(&w.normalized)?, H * 4, "f32_le"),
                ("attn_norm_q8_1", view(&w.normalized_q8_1)?,
                    H / Q8_1_ACTIVATION_BLOCK_ELEMENTS * Q8_1_ACTIVATION_BLOCK_BYTES, "q8_1_32x36"),
            ];
            for (name, source, stride, format) in inputs {
                let mut bytes = vec![0_u8; rows * stride];
                let extent = bytes.len();
                let proof = source.subview(0, extent)?
                    .copy_to_host_sync_with_submission_proof(&d.stream, &mut bytes, receipt)?;
                if !proof.covers_submission(receipt) || !proof.belongs_to_stream(&d.stream)? {
                    return Err(invalid("original projection operands owner/epoch proof"));
                }
                if format == "f32_le" && bytes.chunks_exact(4)
                    .any(|b| !f32::from_le_bytes(b.try_into().unwrap()).is_finite()) {
                    return Err(invalid("original projection operands nonfinite"));
                }
                let filename = format!("operand-layer{layer}-{name}.bin");
                let mut file = std::fs::OpenOptions::new().write(true).create_new(true)
                    .open(self.directory.join(&filename)).map_err(Qwen35Error::from)?;
                std::io::Write::write_all(&mut file, &bytes).map_err(Qwen35Error::from)?;
                self.operand_inventory.borrow_mut().push(json!({"layer":layer,"name":name,
                    "file":filename,"format":format,"rows":rows,"width":H,
                    "start_position":self.batch_start,"row_bytes":stride,"bytes":extent,
                    "sha256":sha256_hex(&bytes),"same_layer_submission":true}));
            }
        }
        Ok(())
    }

    pub(super) fn finish(&self) -> Result<Value> {
        if self.inventory.borrow().len() != 64 * 7 || self.operand_inventory.borrow().len() != 10 {
            return Err(invalid("prefix witness incomplete layer coverage"));
        }
        Ok(json!({"schema":"qwen35-prefix-layer-boundary-witness-v1",
            "directory":self.directory,"batch_start":self.batch_start,"batch_rows":512,
            "inventory":*self.inventory.borrow(),"all64_layers":true,"all_tokens":false,
            "original_projection_operands":{"schema":"qwen35-same-submission-original-operands-v1",
                "layers":[0,31,50,51,63],"full_batch_rows":512,
                "inventory":*self.operand_inventory.borrow(),"reconstructed_inputs":false},
            "host_serialized":true,"performance_evidence":false,"product_qualification":false}))
    }
}

pub(super) fn llama_selected_count(operations: &[PrefillLayerOps]) -> usize {
    operations.iter().map(|op| match op {
        PrefillLayerOps::Linear(p) => [&p.qkv, &p.z].into_iter()
            .filter(|p| p.llama_f16_input_enabled()).count(),
        PrefillLayerOps::Full(p) => [&p.q, &p.k, &p.v].into_iter()
            .filter(|p| p.llama_f16_input_enabled()).count(),
    }).sum()
}

pub(super) fn original_selected_count(operations: &[PrefillLayerOps]) -> usize {
    operations.iter().map(|op| match op {
        PrefillLayerOps::Linear(p) => [&p.qkv, &p.z].into_iter()
            .filter(|p| p.llama_original_input_enabled()).count(),
        PrefillLayerOps::Full(p) => [&p.q, &p.k, &p.v].into_iter()
            .filter(|p| p.llama_original_input_enabled()).count(),
    }).sum()
}

// Prefix-only override. The original normalized buffer already exists and is
// produced on this same stream. History, B8 tails, verify and decode never call
// this binder; no extra resident copy or private cuBLAS workspace is allocated.
pub(super) fn install_original(
    operations: &mut [PrefillLayerOps], rows: usize,
    core: &BatchedVerifyAudit, d: &Driver,
) -> Result<()> {
    if rows != 512 || operations.len() != 64 || selected_count(operations) != 144 {
        return Err(invalid("original input requires all144 admitted B512 projections"));
    }
    let input = CudaF32View::new(core.workspace.normalized.view(0, rows * H * 4)?,
        PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1, rows * H)?;
    let bind = |p: &mut CudaQwen35PrefillCublasProjectionV1| -> Result<()> {
        Ok(core.workspace.cublas.bind_original_input_audit(&d.stream, p, input)?)
    };
    for (index, op) in operations.iter_mut().enumerate() {
        match op {
            PrefillLayerOps::Linear(p) if index % 4 != 3 => {bind(&mut p.qkv)?; bind(&mut p.z)?;}
            PrefillLayerOps::Full(p) if index % 4 == 3 => {
                bind(&mut p.q)?; bind(&mut p.k)?; bind(&mut p.v)?;
            }
            _ => return Err(invalid("original input layer topology drift")),
        }
    }
    if original_selected_count(operations) != 144 || selected_count(operations) != 0
        || llama_selected_count(operations) != 0 {
        return Err(invalid("original input selection count drift"));
    }
    Ok(())
}

pub(super) fn full_original_selected_count(operations: &[PrefillLayerOps]) -> usize {
    operations.iter().flat_map(|p| p.legacy_projections()).filter(|p| p.llama_full_original_enabled()).count()
}

pub(super) fn qpn_projection_count(operations: &[PrefillLayerOps]) -> usize {
    3 * operations.iter().filter(|op| op.ffn().qpn_rows().is_some()).count()
}

pub(super) fn original_operand_count(operations: &[PrefillLayerOps]) -> usize {
    full_original_selected_count(operations) + qpn_projection_count(operations)
}

pub(super) fn validate_original_inventory(legacy: usize, qpn: usize, bf16: usize) -> Result<()> {
    if qpn > 192 || !qpn.is_multiple_of(3) || legacy.checked_add(qpn) != Some(400) || bf16 != 96 {
        return Err(invalid("original input legacy/QPN/BF16 inventory mismatch"));
    }
    Ok(())
}

pub(super) fn bf16_scalar_selected_count(operations: &[PrefillLayerOps]) -> usize {
    operations.iter().map(|op| match op {
        PrefillLayerOps::Linear(p) => [&p.alpha, &p.beta].into_iter()
            .filter(|p| p.llama_full_original_enabled()).count(),
        PrefillLayerOps::Full(_) => 0,
    }).sum()
}

/// B512 prefix recipe, also reused by the explicit initialized-history recipe.
/// Legacy history/B8 tails/verify/decode retain their original scalar math.
pub(super) fn install_bf16_scalars(
    operations: &mut [PrefillLayerOps], rows: usize,
    core: &BatchedVerifyAudit, d: &Driver,
) -> Result<()> {
    if rows != 512 || operations.len() != 64 || full_original_selected_count(operations) != 400
        || bf16_scalar_selected_count(operations) != 0 {
        return Err(invalid("BF16 scalar recipe requires all400 original B512 projections"));
    }
    let normalized = prefill_f32_view(&core.workspace.normalized, rows * H)?;
    for (index, op) in operations.iter_mut().enumerate() {
        match op {
            PrefillLayerOps::Linear(p) if index % 4 != 3 => {
                core.workspace.cublas.bind_llama_full_original_audit(&d.stream, &mut p.alpha, normalized, None)?;
                core.workspace.cublas.bind_llama_full_original_audit(&d.stream, &mut p.beta, normalized, None)?;
            }
            PrefillLayerOps::Full(_) if index % 4 == 3 => {},
            _ => return Err(invalid("BF16 scalar layer topology drift")),
        }
    }
    if bf16_scalar_selected_count(operations) != 96 {
        return Err(invalid("BF16 scalar projection count drift"));
    }
    Ok(())
}

/// All400 original-operand projections for admitted wide prefix or explicit
/// initialized-history. Reuse scratch; never installed by B8 verify/decode.
pub(super) fn validate_exact_input_bindings(operations: &[PrefillLayerOps], rows: usize) -> Result<()> {
    prefill_width::validate_exact_input(rows)?;
    if operations.len() != 64 || llama_norm_selected_count(operations) != 128 {
        return Err(invalid("exact input layer/norm count"));
    }
    let mut count = 0;
    for (layer, op) in operations.iter().enumerate() {
        let projections: Vec<_> = match op {
            PrefillLayerOps::Linear(p) if layer % 4 != 3 => vec![&p.qkv, &p.z, &p.alpha,
                &p.beta, &p.output],
            PrefillLayerOps::Full(p) if layer % 4 == 3 => vec![&p.q, &p.k, &p.v,
                &p.output],
            _ => return Err(invalid("exact input layer topology")),
        };
        for p in projections {
            if !p.llama_exact_input_rows_enabled() || p.raw().tokens as usize != rows {
                return Err(invalid("exact input projection mode/extent drift"));
            }
            count += 1;
        }
        op.ffn().validate_original_input(rows, true)?;
        if let Some(ffn) = op.ffn().legacy_opt() {
            for p in [&ffn.ffn_gate, &ffn.ffn_up, &ffn.ffn_down] {
                if !p.llama_exact_input_rows_enabled() || p.raw().tokens as usize != rows {
                    return Err(invalid("exact legacy FFN input mode/extent drift"));
                }
            }
        }
        count += 3;
    }
    if count != 496 { return Err(invalid("exact input projection count")); }
    validate_original_inventory(full_original_selected_count(operations),
        qpn_projection_count(operations), bf16_scalar_selected_count(operations))
}

// Direct original-input binding. Arbitrary realM must never first select an
// unsupported integer kernel, or inherit B8 verifier projection policies.
pub(super) fn install_exact_input(operations: &mut [PrefillLayerOps], rows: usize,
    core: &BatchedVerifyAudit, d: &Driver) -> Result<()> {
    prefill_width::validate_exact_input(rows)?;
    if !core.exact_input_owner || !d.llama_exact_input_rows || operations.len() != 64 {
        return Err(invalid("exact input binding owner/recipe"));
    }
    install_original_input_bindings(operations, rows, core, d)
}

// Prefix keeps its existing owner, row schedule and workspace lifecycle. Only
// replace the obsolete grid/integer intermediate bindings with the final ABI.
pub(super) fn install_row_input_prefix(operations: &mut [PrefillLayerOps], rows: usize,
    core: &BatchedVerifyAudit, d: &Driver) -> Result<()> {
    if core.exact_input_owner || rows != 512 || operations.len() != 64
        || !d.llama_exact_input_rows || !d.llama_full_original || !d.llama_bf16_scalars
        || !d.llama_rotated_kv || !d.llama_history_full {
        return Err(invalid("row input prefix binding owner/recipe"));
    }
    install_original_input_bindings(operations, rows, core, d)
}

fn install_original_input_bindings(operations: &mut [PrefillLayerOps], rows: usize,
    core: &BatchedVerifyAudit, d: &Driver) -> Result<()> {
    let w = &core.workspace;
    let normalized = prefill_f32_view(&w.normalized, rows * H)?;
    let post_norm = prefill_f32_view(&w.post_norm, rows * H)?;
    let activation = prefill_f32_view(&w.ffn_activation, rows * linear::INTERMEDIATE)?;
    let bind = |p: &mut CudaQwen35PrefillCublasProjectionV1, input, output| -> Result<()> {
        Ok(w.cublas.bind_llama_exact_input_rows_audit(&d.stream, p, input, output)?)
    };
    let bind_ffn = |ffn: &mut PrefillFfn| -> Result<()> {
        if let Some(p) = ffn.legacy_mut_opt() {
            bind(&mut p.ffn_gate, post_norm, None)?;
            bind(&mut p.ffn_up, post_norm, None)?;
            bind(&mut p.ffn_down, activation, None)?;
            p.silu_mul.preflight_llama_input_rows(&d.stream)?;
        }
        ffn.validate_original_input(rows, true)
    };
    for (layer, op) in operations.iter_mut().enumerate() {
        match op {
            PrefillLayerOps::Linear(p) if layer % 4 != 3 => {
                p.attn_norm.enable_llama_exact_input_rows_audit()?;
                p.post_norm.enable_llama_exact_input_rows_audit()?;
                bind(&mut p.qkv, normalized, None)?;
                bind(&mut p.z, normalized, None)?;
                bind(&mut p.alpha, normalized, None)?;
                bind(&mut p.beta, normalized, None)?;
                let original = prefill_f32_view(&w.attention_output, rows * linear::RECURRENT_ELEMENTS)?;
                let projected = prefill_f32_view(&w.qkv, rows * H)?;
                bind(&mut p.output, original, Some(projected))?;
                p.attention_residual = CudaAddF32ArgsV1::bind(&d.stream, projected,
                    w.hidden_input(layer, rows)?, prefill_f32_view(&w.attention_residual, rows * H)?)?;
                bind_ffn(&mut p.ffn)?;
            }
            PrefillLayerOps::Full(p) if layer % 4 == 3 => {
                p.attn_norm.enable_llama_exact_input_rows_audit()?;
                p.post_norm.enable_llama_exact_input_rows_audit()?;
                bind(&mut p.q, normalized, None)?;
                bind(&mut p.k, normalized, None)?;
                bind(&mut p.v, normalized, None)?;
                bind(&mut p.output, prefill_f32_view(&w.ffn_up, rows * full::QUERY_VALUES)?, None)?;
                bind_ffn(&mut p.ffn)?;
            }
            _ => return Err(invalid("exact input layer topology")),
        }
    }
    validate_exact_input_bindings(operations, rows)
}

/// Legacy B512 original-operand installer; integer prerequisite is unchanged.
pub(super) fn install_full_original(
    operations: &mut [PrefillLayerOps], rows: usize,
    core: &BatchedVerifyAudit, d: &Driver,
) -> Result<()> {
    if rows != 512 || operations.len() != 64 || selected_count(operations) != 144
        || prefill_integer_mma::selected_count(operations) != 192 {
        return Err(invalid("full original requires admitted144 input and192 FFN projections"));
    }
    let w = &core.workspace;
    let normalized = prefill_f32_view(&w.normalized, rows * H)?;
    let post_norm = prefill_f32_view(&w.post_norm, rows * H)?;
    let activation = prefill_f32_view(&w.ffn_activation, rows * linear::INTERMEDIATE)?;
    let bind = |p: &mut CudaQwen35PrefillCublasProjectionV1, input, output| -> Result<()> {
        Ok(w.cublas.bind_llama_full_original_audit(&d.stream, p, input, output)?)
    };
    for (layer, op) in operations.iter_mut().enumerate() {
        match op {
            PrefillLayerOps::Linear(p) if layer % 4 != 3 => {
                bind(&mut p.qkv, normalized, None)?;
                bind(&mut p.z, normalized, None)?;
                // Recurrence consumes qkv before producing gated attention.
                // That dead qkv range holds the projected result until the next
                // layer; FFN never touches it. Original attention stays disjoint.
                let original = prefill_f32_view(&w.attention_output, rows * linear::RECURRENT_ELEMENTS)?;
                let projected = prefill_f32_view(&w.qkv, rows * H)?;
                bind(&mut p.output, original, Some(projected))?;
                p.attention_residual = CudaAddF32ArgsV1::bind(&d.stream, projected,
                    w.hidden_input(layer, rows)?, prefill_f32_view(&w.attention_residual, rows * H)?)?;
                bind(&mut p.ffn.legacy_mut()?.ffn_gate, post_norm, None)?;
                bind(&mut p.ffn.legacy_mut()?.ffn_up, post_norm, None)?;
                bind(&mut p.ffn.legacy_mut()?.ffn_down, activation, None)?;
                p.ffn.legacy()?.silu_mul.preflight_llama_prefix(&d.stream)?;
            }
            PrefillLayerOps::Full(p) if layer % 4 == 3 => {
                bind(&mut p.q, normalized, None)?;
                bind(&mut p.k, normalized, None)?;
                bind(&mut p.v, normalized, None)?;
                bind(&mut p.output, prefill_f32_view(&w.ffn_up, rows * full::QUERY_VALUES)?, None)?;
                bind(&mut p.ffn.legacy_mut()?.ffn_gate, post_norm, None)?;
                bind(&mut p.ffn.legacy_mut()?.ffn_up, post_norm, None)?;
                bind(&mut p.ffn.legacy_mut()?.ffn_down, activation, None)?;
                p.ffn.legacy()?.silu_mul.preflight_llama_prefix(&d.stream)?;
            }
            _ => return Err(invalid("full original layer topology drift")),
        }
    }
    if d.llama_exact_input_rows {
        for op in operations.iter_mut() {
            let norms = match op {
                PrefillLayerOps::Linear(p) => [&mut p.attn_norm, &mut p.post_norm],
                PrefillLayerOps::Full(p) => [&mut p.attn_norm, &mut p.post_norm],
            };
            for norm in norms { norm.enable_llama_exact_input_rows_audit()?; }
        }
    }
    if llama_norm_selected_count(operations) != if d.llama_exact_input_rows {128} else {0} {
        return Err(invalid("prefix exact input norm selection"));
    }
    if full_original_selected_count(operations) != 400 || selected_count(operations) != 0
        || prefill_integer_mma::selected_count(operations) != 0
        || original_selected_count(operations) != 0 || llama_selected_count(operations) != 0 {
        return Err(invalid("full original projection selection count drift"));
    }
    Ok(())
}

pub(super) fn llama_norm_selected_count(operations: &[PrefillLayerOps]) -> usize {
    operations.iter().map(|op| {
        let norms = match op {
            PrefillLayerOps::Linear(p) => [&p.attn_norm, &p.post_norm],
            PrefillLayerOps::Full(p) => [&p.attn_norm, &p.post_norm],
        };
        norms.into_iter().filter(|norm| norm.llama_exact_input_rows_enabled()).count()
    }).sum()
}

// Only prompt_capture calls this AFTER ordinary integer binding. The history
// capture and decode binders deliberately never see this numerical override.
pub(super) fn install_llama(operations: &mut [PrefillLayerOps], rows: usize) -> Result<()> {
    if rows != 512 || operations.len() != 64 || selected_count(operations) != 144 {
        return Err(invalid("llama input requires all144 admitted B512 input projections"));
    }
    for (index, op) in operations.iter_mut().enumerate() {
        match op {
            PrefillLayerOps::Linear(p) if index % 4 != 3 => {
                p.qkv.enable_llama_f16_input_audit()?;
                p.z.enable_llama_f16_input_audit()?;
            }
            PrefillLayerOps::Full(p) if index % 4 == 3 => {
                p.q.enable_llama_f16_input_audit()?;
                p.k.enable_llama_f16_input_audit()?;
                p.v.enable_llama_f16_input_audit()?;
            }
            _ => return Err(invalid("llama input layer topology drift")),
        }
    }
    if llama_selected_count(operations) != 144 || selected_count(operations) != 0 {
        return Err(invalid("llama input selection count drift"));
    }
    Ok(())
}

pub(super) fn selected_count(operations: &[PrefillLayerOps]) -> usize {
    operations
        .iter()
        .map(|op| match op {
            PrefillLayerOps::Linear(p) => [&p.qkv, &p.z]
                .into_iter()
                .filter(|v| v.integer_mma_input_enabled())
                .count(),
            PrefillLayerOps::Full(p) => [&p.q, &p.k, &p.v]
                .into_iter()
                .filter(|v| v.integer_mma_input_enabled())
                .count(),
            
        })
        .sum()
}

pub(super) fn install(operations: &mut [PrefillLayerOps], rows: usize) -> Result<()> {
    if operations.len() != 64 || rows != 512 {
        return Err(invalid("input integer MMA needs all64 B512 layers"));
    }
    let enable = |p: &mut CudaQwen35PrefillCublasProjectionV1, output_rows: u32| -> Result<()> {
        let raw = p.raw();
        if raw.tokens != 512
            || raw.rows != output_rows
            || raw.cols != 5120
            || p.integer_mma_input_scratch_bytes() != SCRATCH_BYTES as u64
        {
            return Err(invalid("input integer projection geometry drift"));
        }
        p.enable_integer_mma_input_audit()?;
        Ok(())
    };
    for (index, op) in operations.iter_mut().enumerate() {
        match op {
            PrefillLayerOps::Linear(p) if index % 4 != 3 => {
                enable(&mut p.qkv, 10240)?;
                enable(&mut p.z, 6144)?;
            }
            PrefillLayerOps::Full(p) if index % 4 == 3 => {
                enable(&mut p.q, 12288)?;
                enable(&mut p.k, 1024)?;
                enable(&mut p.v, 1024)?;
            }
            _ => return Err(invalid("input integer layer topology drift")),
        }
    }
    if selected_count(operations) != 144 {
        return Err(invalid("input integer requires exactly144 projections"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn full_original_capture_location_excludes_history_and_full_attention() {
        assert_eq!((0..64).filter(|&l| relocated_attention_output(true,l,512)).count(),48);
        for rows in [1,8,128,256,511,513] {
            assert!(!(0..64).any(|l| relocated_attention_output(true,l,rows)));
        }
        assert!(!(0..64).any(|l| relocated_attention_output(false,l,512)));
        assert!(!relocated_attention_output(true,64,512));
        assert_eq!(48*6+16*7,400);
        assert_eq!(512*linear::RECURRENT_ELEMENTS*4,12_582_912);
    }
    use super::*;
    use std::ffi::OsStr;
    #[test]
    fn input_integer_policy_is_independent_default_off_and_b512_only() {
        assert!(!requested(None, None).unwrap());
        assert!(!requested(None, Some(512)).unwrap());
        assert!(requested(Some(OsStr::new("1")), Some(512)).unwrap());
        for rows in [None, Some(8), Some(128), Some(256), Some(1024)] {
            assert!(requested(Some(OsStr::new("1")), rows).is_err());
        }
        for value in ["", "0", "true", "on", " 1", "1 "] {
            assert!(requested(Some(OsStr::new(value)), Some(512)).is_err());
        }
    }
    #[test]
    fn input_integer_rejects_incomplete_layers() {
        assert_eq!(selected_count(&[]), 0);
        assert_eq!(llama_selected_count(&[]), 0);
        for rows in [8, 128, 256, 512] {
            assert!(install(&mut [], rows).is_err());
            assert!(install_llama(&mut [], rows).is_err());
        }
    }
    #[test]
    fn prefix_boundary_probe_is_bounded_and_cannot_be_decode_or_history() {
        assert_eq!(audit_batch_start(8192, 2048, true).unwrap(), 5632);
        assert_eq!(audit_batch_start(8477, 2048, true).unwrap(), 5632);
        for (prompt, history, enabled) in [(2048,2048,true),(2559,2048,true),
            (32768,2048,true),(8192,2048,false),(1,2048,true)] {
            assert!(audit_batch_start(prompt, history, enabled).is_err());
        }
    }
    #[test]
    fn original_operand_capture_covers_five_real_layers_and_full_extents() {
        assert_eq!((0..64).filter(|&layer| original_operand_layer(layer)).collect::<Vec<_>>(),
            vec![0, 31, 50, 51, 63]);
        assert!(!original_operand_layer(64));
        assert_eq!(512 * H * 4, 10_485_760);
        assert_eq!(512 * H / Q8_1_ACTIVATION_BLOCK_ELEMENTS * Q8_1_ACTIVATION_BLOCK_BYTES,
            2_949_120);
    }
}
