//! Reference-only final-prompt-row capture. No arithmetic or default-path change.
//! Host readback deliberately serializes this diagnostic; never time it as decode.
use super::*;
use std::cell::{Cell, RefCell};
use std::io::Write;

const NAMES: [&str; 7] = [
    "input_hidden",
    "attn_norm",
    "attention_output",
    "attn_residual",
    "attn_post_norm",
    "ffn_out",
    "l_out",
];

pub(super) struct BoundaryTrace {
    directory: PathBuf,
    position: usize,
    inventory: RefCell<Vec<Value>>,
    linear_inventory: RefCell<Vec<Value>>,
    qkv_window: RefCell<Option<Value>>,
    projection_inputs: RefCell<Vec<Value>>,
    full_inventory: RefCell<Vec<Value>>,
    projection_window: Option<ProjectionWindow>,
    numerical_tail_states: Option<RefCell<Vec<Value>>>,
    history_feature_dump: bool,
    history_capture_width: Cell<usize>,
}

// Bounded, host-serialized component corpus. It is only constructed inside
// the already explicit target-only reference boundary diagnostic.
const PROJECTION_LAYERS: [usize; 5] = [0, 31, 50, 51, 63];
const PROJECTION_PLANES: usize = 9;

struct ProjectionWindow {
    start: usize,
    end: usize,
    inventory: RefCell<Vec<Value>>,
}

fn projection_rows(value: Option<&std::ffi::OsStr>) -> Result<usize> {
    match value.and_then(std::ffi::OsStr::to_str) {
        None if value.is_none() => Ok(0),
        Some("64") => Ok(64),
        Some("128") => Ok(128),
        Some("256") => Ok(256),
        Some("512") => Ok(512),
        _ => Err(invalid(
            "reference projection rows must be exactly 64, 128, 256 or 512",
        )),
    }
}

fn projection_span(start: usize, end: usize, batch_start: usize) -> Option<(usize, usize)> {
    let batch_end = batch_start.checked_add(B)?;
    let first = start.max(batch_start);
    let last = end.min(batch_end);
    (first < last).then(|| (first - batch_start, last - first))
}

pub(super) fn requested_projection_rows() -> Result<usize> {
    projection_rows(std::env::var_os("AUGUR_QWEN35_REFERENCE_PROJECTION_ROWS").as_deref())
}

fn full_cache_bytes(position: usize) -> Result<usize> {
    position
        .checked_add(1)
        .filter(|&n| n <= 8192)
        .and_then(|n| n.checked_mul(QWEN35_KV8_CACHE_ROW_BYTES))
        .ok_or_else(|| invalid("full-attention diagnostic is bounded to 8192 occupied positions"))
}

fn selected_row(start: usize, position: usize) -> Option<usize> {
    position.checked_sub(start).filter(|&row| row < B)
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(Qwen35Error::from)?;
    file.write_all(bytes).map_err(Qwen35Error::from)?;
    file.flush().map_err(Qwen35Error::from)?;
    Ok(())
}

impl BoundaryTrace {
    /// A wider cold-prefix may run only before every requested B8 capture row.
    /// It never invokes the eight-row readback ABI or changes the window size.
    pub(super) fn prefix_disjoint(&self, end_exclusive: usize) -> bool {
        self.position >= end_exclusive
            && self
                .projection_window
                .as_ref()
                .is_none_or(|w| w.start >= end_exclusive)
    }

    pub(super) fn new(directory: &Path, position: usize, numerical_tail: bool) -> Result<Self> {
        let rows = requested_projection_rows()?;
        let mut trace = Self::new_with_projection_rows(directory, position, rows)?;
        trace.numerical_tail_states = numerical_tail.then(|| RefCell::new(Vec::new()));
        trace.history_feature_dump = match std::env::var_os("AUGUR_QWEN35_REFERENCE_HISTORY_FEATURES").as_deref() {
            None => false,
            Some(v) if v == "1" && numerical_tail => true,
            _ => return Err(invalid("history feature witness requires exactly1 and numerical-tail audit")),
        };
        Ok(trace)
    }

    fn new_with_projection_rows(directory: &Path, position: usize, rows: usize) -> Result<Self> {
        full_cache_bytes(position)?;
        let end = position + 1;
        if ![0, 64, 128, 256, 512].contains(&rows) || rows > end {
            return Err(invalid("reference projection window exceeds real prompt"));
        }
        // A fresh directory is required. Do not reuse or overwrite frozen dumps.
        std::fs::create_dir(directory).map_err(Qwen35Error::from)?;
        Ok(Self {
            directory: directory.to_path_buf(),
            position,
            inventory: RefCell::new(Vec::new()),
            linear_inventory: RefCell::new(Vec::new()),
            qkv_window: RefCell::new(None),
            projection_inputs: RefCell::new(Vec::new()),
            full_inventory: RefCell::new(Vec::new()),
            numerical_tail_states: None,
            history_feature_dump: false,
            history_capture_width: Cell::new(B),
            projection_window: (rows != 0).then(|| ProjectionWindow {
                start: end - rows,
                end,
                inventory: RefCell::new(Vec::new()),
            }),
        })
    }

    pub(super) fn capture_batch(
        &self,
        d: &Driver,
        w: &PrefillWorkspace,
        layer: usize,
    ) -> Result<()> {
        self.capture_batch_rows(d, w, layer, B)
    }

    pub(super) fn capture_initialized_history(
        &self, d: &Driver, w: &PrefillWorkspace, layer: usize, rows: usize,
    ) -> Result<()> {
        if !self.history_feature_dump || !d.wide_history_capture || rows != 512 {
            return Err(invalid("wide history trace requires explicit full feature witness"));
        }
        self.history_capture_width.set(rows);
        self.capture_batch_rows(d, w, layer, rows)
    }

    fn capture_batch_rows(&self, d: &Driver, w: &PrefillWorkspace, layer: usize, batch_rows: usize) -> Result<()> {
        if let Some(window) = &self.projection_window {
            if batch_rows == B {
                window.capture(d, w, layer, &self.directory)?;
            } else {
                window.capture_wide_history(d, w, layer, batch_rows, &self.directory)?;
            }
        }
        let Some(row) = self.position.checked_sub(d.cursor).filter(|&row| row < batch_rows) else {
            return Ok(());
        };
        if layer >= 64 || self.inventory.borrow().len() != layer * NAMES.len() {
            return Err(invalid("reference boundary layer order or duplicate"));
        }
        let sources = [
            w.hidden_input(layer, batch_rows)?.buffer_view(),
            view(&w.normalized)?,
            view(&w.attention_output)?,
            view(&w.attention_residual)?,
            view(&w.post_norm)?,
            view(&w.ffn_output)?,
            w.hidden_output(layer, batch_rows)?.buffer_view(),
        ];
        let receipt = d.stream.submission_receipt()?;
        if let Some(inventory) = &self.numerical_tail_states {
            if layer == 50 || layer == 51 {
                if row != batch_rows - 1 {
                    return Err(invalid(
                        "numerical tail state requires last initialized B8 row",
                    ));
                }
                let sources = match (layer, &d.plans[layer]) {
                    (50, DevicePlan::Linear(plan)) => vec![
                        (
                            "recurrent_state",
                            plan.state.view(0, plan.state.len_bytes())?,
                            "f32_le",
                            48,
                            16384,
                        ),
                        (
                            "conv_history",
                            plan.history.view(0, plan.history.len_bytes())?,
                            "f32_le",
                            1,
                            plan.history.len_bytes() / 4,
                        ),
                    ],
                    (51, DevicePlan::Full(plan)) => {
                        let bytes = full_cache_bytes(self.position)?;
                        vec![
                            (
                                "key_cache",
                                plan.key_cache.view(0, bytes)?,
                                "q8_0_32x34",
                                self.position + 1,
                                full::KV_VALUES,
                            ),
                            (
                                "value_cache",
                                plan.value_cache.view(0, bytes)?,
                                "q8_0_32x34",
                                self.position + 1,
                                full::KV_VALUES,
                            ),
                        ]
                    }
                    _ => return Err(invalid("numerical tail model topology")),
                };
                for (name, source, format, rows, width) in sources {
                    let mut bytes = vec![0; source.len_bytes()];
                    let proof = source
                        .copy_to_host_sync_with_submission_proof(&d.stream, &mut bytes, receipt)?;
                    if !proof.covers_submission(receipt) || !proof.belongs_to_stream(&d.stream)? {
                        return Err(invalid("numerical tail state owner/epoch"));
                    }
                    if format == "f32_le"
                        && bytes
                            .chunks_exact(4)
                            .any(|word| !f32::from_le_bytes(word.try_into().unwrap()).is_finite())
                    {
                        return Err(invalid("numerical tail nonfinite state"));
                    }
                    let filename = format!("numeric-tail-layer{layer}-{name}.bin");
                    write_new(&self.directory.join(&filename), &bytes)?;
                    inventory
                        .borrow_mut()
                        .push(json!({"file":filename,"name":name,
                        "layer":layer,"format":format,"rows":rows,"width":width,
                        "position":self.position,"bytes":bytes.len(),"sha256":sha256_hex(&bytes),
                        "initialized_committed_only":true,"capacity_readback":false}));
                }
            }
        }
        for (name, source) in NAMES.into_iter().zip(sources) {
            let values = read_f32_view(&d.stream, source.subview(row * H * 4, H * 4)?, receipt)?;
            if values.len() != H || values.iter().any(|v| !v.is_finite()) {
                return Err(invalid("reference boundary nonfinite/extent"));
            }
            let bytes = f32_bytes(&values);
            let filename = format!("{name}-{layer}.f32le.bin");
            write_new(&self.directory.join(&filename), &bytes)?;
            self.inventory
                .borrow_mut()
                .push(json!({"layer":layer,"name":name,
                "file":filename,"position":self.position,"row":row,"batch_start":d.cursor,
                "elements":H,"bytes":bytes.len(),"sha256":sha256_hex(&bytes),"all_finite":true}));
        }
        if layer == 0 {
            // Retain enough actual projected rows for a history-independent
            // four-tap convolution replay. Short final batches stay explicit.
            let rows = (row + 1).min(linear::KERNEL_SIZE);
            let first = row + 1 - rows;
            let width = linear::QKV_ELEMENTS;
            let values = read_f32_view(
                &d.stream,
                w.qkv.view(first * width * 4, rows * width * 4)?,
                receipt,
            )?;
            if values.len() != rows * width || values.iter().any(|v| !v.is_finite()) {
                return Err(invalid("linear QKV window nonfinite/extent"));
            }
            let bytes = f32_bytes(&values);
            let filename = "linear0-qkv-window.f32le.bin";
            write_new(&self.directory.join(filename), &bytes)?;
            *self.qkv_window.borrow_mut() = Some(json!({"file":filename,"layer":0,
                "start_position":d.cursor+first,"end_position_exclusive":self.position+1,
                "rows":rows,"width":width,"bytes":bytes.len(),"sha256":sha256_hex(&bytes),
                "all_finite":true,"history_independent_final_convolution":rows==linear::KERNEL_SIZE}));
            // These input buffers remain live and unmodified through the FFN.
            for (name, source) in [
                ("input_hidden", w.hidden_input(0, batch_rows)?.buffer_view()),
                ("attn_norm", view(&w.normalized)?),
            ] {
                let values = read_f32_view(
                    &d.stream,
                    source.subview(first * H * 4, rows * H * 4)?,
                    receipt,
                )?;
                if values.len() != rows * H || values.iter().any(|v| !v.is_finite()) {
                    return Err(invalid("projection input window nonfinite/extent"));
                }
                let bytes = f32_bytes(&values);
                let filename = format!("linear0-{name}-window.f32le.bin");
                write_new(&self.directory.join(&filename), &bytes)?;
                self.projection_inputs
                    .borrow_mut()
                    .push(json!({"name":name,"file":filename,
                    "format":"f32_le","rows":rows,"width":H,"start_position":d.cursor+first,
                    "end_position_exclusive":self.position+1,"bytes":bytes.len(),
                    "sha256":sha256_hex(&bytes),"all_finite":true}));
            }
            let stride = H / Q8_1_ACTIVATION_BLOCK_ELEMENTS * Q8_1_ACTIVATION_BLOCK_BYTES;
            let source = w.normalized_q8_1.view(first * stride, rows * stride)?;
            let mut bytes = vec![0_u8; rows * stride];
            let proof =
                source.copy_to_host_sync_with_submission_proof(&d.stream, &mut bytes, receipt)?;
            if !proof.covers_submission(receipt) || !proof.belongs_to_stream(&d.stream)? {
                return Err(invalid("Q8 input readback owner epoch proof"));
            }
            let filename = "linear0-attn_norm-window.q8_1.bin";
            write_new(&self.directory.join(filename), &bytes)?;
            self.projection_inputs.borrow_mut().push(json!({"name":"attn_norm_q8_1","file":filename,
                "format":"q8_1_32x36","rows":rows,"width":H,"start_position":d.cursor+first,
                "end_position_exclusive":self.position+1,"bytes":bytes.len(),"sha256":sha256_hex(&bytes)}));
            // These buffers are not reused by the output projection or FFN.
            // attention_output is intentionally excluded: its gated-width
            // storage has already been overwritten by the hidden-width output.
            for (name, source, width) in [
                ("linear_attn_qkv_mixed", &w.qkv, linear::QKV_ELEMENTS),
                ("z", &w.z, linear::RECURRENT_ELEMENTS),
                ("alpha", &w.alpha_raw, linear::VALUE_HEADS),
                ("beta", &w.beta_raw, linear::VALUE_HEADS),
                ("conv_output_silu", &w.activated, linear::QKV_ELEMENTS),
                ("attn_output", &w.recurrent, linear::RECURRENT_ELEMENTS),
            ] {
                let values =
                    read_f32_view(&d.stream, source.view(row * width * 4, width * 4)?, receipt)?;
                if values.len() != width || values.iter().any(|v| !v.is_finite()) {
                    return Err(invalid("linear boundary nonfinite/extent"));
                }
                let bytes = f32_bytes(&values);
                let filename = format!("linear0-{name}.f32le.bin");
                write_new(&self.directory.join(&filename), &bytes)?;
                self.linear_inventory.borrow_mut().push(json!({"layer":0,"name":name,
                    "file":filename,"position":self.position,"row":row,"batch_start":d.cursor,
                    "elements":width,"bytes":bytes.len(),"sha256":sha256_hex(&bytes),"all_finite":true}));
            }
        }
        if layer == 3 {
            let DevicePlan::Full(plan) = &d.plans[layer] else {
                return Err(invalid("first full-attention layer kind"));
            };
            // The canonical per-row full core leaves these buffers at the
            // last submitted prompt position. Never infer a row from a cache
            // capacity or read uninitialized future rows.
            if row != batch_rows - 1 || !self.full_inventory.borrow().is_empty() {
                return Err(invalid(
                    "full-attention capture requires final canonical batch row",
                ));
            }
            for (name, source, width) in [
                ("query_gate", &plan.q_full, full::QUERY_GATE_VALUES),
                ("key_raw", &plan.k_raw, full::KV_VALUES),
                ("value", &plan.value, full::KV_VALUES),
                ("query_normed", &plan.q_normed, full::QUERY_VALUES),
                ("key_normed", &plan.k_normed, full::KV_VALUES),
                ("query", &plan.query, full::QUERY_VALUES),
                ("key", &plan.key, full::KV_VALUES),
                ("attention", &plan.attention, full::QUERY_VALUES),
                ("gated", &plan.gated, full::QUERY_VALUES),
            ] {
                let values = read_f32_view(&d.stream, source.view(0, width * 4)?, receipt)?;
                if values.len() != width || values.iter().any(|v| !v.is_finite()) {
                    return Err(invalid("full-attention capture nonfinite/extent"));
                }
                let bytes = f32_bytes(&values);
                let filename = format!("full3-{name}.f32le.bin");
                write_new(&self.directory.join(&filename), &bytes)?;
                self.full_inventory.borrow_mut().push(json!({"name":name,"layer":3,
                    "position":self.position,"format":"f32_le","elements":width,
                    "file":filename,"bytes":bytes.len(),"sha256":sha256_hex(&bytes),"all_finite":true}));
            }
            let cache_bytes = full_cache_bytes(self.position)?;
            for (name, source) in [
                ("key_cache", &plan.key_cache),
                ("value_cache", &plan.value_cache),
            ] {
                let mut bytes = vec![0u8; cache_bytes];
                let proof = source
                    .view(0, cache_bytes)?
                    .copy_to_host_sync_with_submission_proof(&d.stream, &mut bytes, receipt)?;
                if !proof.covers_submission(receipt) || !proof.belongs_to_stream(&d.stream)? {
                    return Err(invalid("full-attention cache readback owner epoch proof"));
                }
                let filename = format!("full3-{name}.q8_0.bin");
                write_new(&self.directory.join(&filename), &bytes)?;
                self.full_inventory
                    .borrow_mut()
                    .push(json!({"name":name,"layer":3,
                    "position":self.position,"format":"q8_0_32x34","rows":self.position+1,
                    "width":full::KV_VALUES,"row_bytes":QWEN35_KV8_CACHE_ROW_BYTES,
                    "file":filename,"bytes":bytes.len(),"sha256":sha256_hex(&bytes)}));
            }
        }
        Ok(())
    }

    pub(super) fn finish(&self, d: &Driver) -> Result<Value> {
        if self.inventory.borrow().len() != 64 * NAMES.len()
            || self.linear_inventory.borrow().len() != 6
            || self.qkv_window.borrow().is_none()
            || self.projection_inputs.borrow().len() != 3
            || self.full_inventory.borrow().len() != 11
            || self
                .numerical_tail_states
                .as_ref()
                .is_some_and(|items| items.borrow().len() != 4)
            || d.cursor != self.position + 1
        {
            return Err(invalid("reference boundary incomplete prompt capture"));
        }
        let logits = read_f32_view(
            &d.stream,
            view(&d.head.logits)?,
            d.stream.submission_receipt()?,
        )?;
        if logits.len() != V || logits.iter().any(|v| !v.is_finite()) {
            return Err(invalid("reference boundary logit extent/nonfinite"));
        }
        let bytes = f32_bytes(&logits);
        write_new(&self.directory.join("logits.f32le.bin"), &bytes)?;
        let linear_bytes =
            (2 * linear::QKV_ELEMENTS + 2 * linear::RECURRENT_ELEMENTS + 2 * linear::VALUE_HEADS)
                * 4;
        let window_bytes = self.qkv_window.borrow().as_ref().unwrap()["bytes"]
            .as_u64()
            .ok_or_else(|| invalid("QKV window byte count"))? as usize;
        let input_bytes =
            self.projection_inputs
                .borrow()
                .iter()
                .try_fold(0usize, |sum, item| {
                    let bytes = item["bytes"]
                        .as_u64()
                        .ok_or_else(|| invalid("projection input byte count"))?
                        as usize;
                    sum.checked_add(bytes)
                        .ok_or_else(|| invalid("projection input byte overflow"))
                })?;
        let full_bytes: usize = self
            .full_inventory
            .borrow()
            .iter()
            .map(|item| item["bytes"].as_u64().unwrap() as usize)
            .sum();
        let projection_window = self
            .projection_window
            .as_ref()
            .map(|window| window.finish())
            .transpose()?;
        let projection_bytes = projection_window
            .as_ref()
            .map(|window| window["capture_d2h_bytes"].as_u64().unwrap())
            .unwrap_or(0);
        let numerical_tail_bytes = self
            .numerical_tail_states
            .as_ref()
            .map(|items| {
                items
                    .borrow()
                    .iter()
                    .map(|item| item["bytes"].as_u64().unwrap())
                    .sum::<u64>()
            })
            .unwrap_or(0);
        let mut report = json!({"schema":"qwen35-reference-final-prompt-boundary-v5",
            "directory":self.directory,"position":self.position,"tensor_count":64*NAMES.len(),
            "tensors":*self.inventory.borrow(),"logits_file":"logits.f32le.bin",
            "linear_layer0_tensors":*self.linear_inventory.borrow(),
            "linear_layer0_qkv_window":*self.qkv_window.borrow(),
            "linear_layer0_projection_inputs":*self.projection_inputs.borrow(),
            "full_layer3_tensors":*self.full_inventory.borrow(),
            "projection_window":projection_window,
            "logits_sha256":sha256_hex(&bytes),"logits_bytes":bytes.len(),
            "capture_d2h_bytes":(64*NAMES.len()*H*4+bytes.len()+linear_bytes+window_bytes+input_bytes+full_bytes) as u64 + projection_bytes + numerical_tail_bytes,
            "scope":"last_prompt_row_canonical_batch_after_each_whole_layer",
            "host_serialized_diagnostic":true,"product_qualification":false,
            "semantic_parity_proven":false,"performance_claim_permitted":false});
        if let Some(items) = &self.numerical_tail_states {
            report["schema"] = json!("qwen35-reference-final-prompt-boundary-v6");
            report["numerical_tail_states"] = json!(*items.borrow());
            report["numerical_tail_state_bytes"] = json!(numerical_tail_bytes);
        }
        if self.history_feature_dump {
            if d.context_rows != 2048 || d.features.len_bytes() != 2048 * 5 * H * 4 {
                return Err(invalid("full initialized feature witness extent"));
            }
            let receipt = d.stream.submission_receipt()?;
            let mut features = vec![0; d.features.len_bytes()];
            let proof = view(&d.features)?.copy_to_host_sync_with_submission_proof(&d.stream, &mut features, receipt)?;
            if !proof.covers_submission(receipt) || !proof.belongs_to_stream(&d.stream)?
                || features.chunks_exact(4).any(|b| !f32::from_le_bytes(b.try_into().unwrap()).is_finite()) {
                return Err(invalid("initialized feature readback proof/nonfinite"));
            }
            let filename = "initialized-history-features.f32le.bin";
            write_new(&self.directory.join(filename), &features)?;
            report["schema"] = json!("qwen35-reference-final-prompt-boundary-v7");
            report["history_capture_batch_width"] = json!(self.history_capture_width.get());
            report["initialized_history_features"] = json!({"file":filename,"name":"initialized_history_features",
                "format":"f32_le","rows":2048,"width":5*H,"bytes":features.len(),
                "start_position":d.cursor-2048,"end_position_exclusive":d.cursor,
                "sha256":sha256_hex(&features),"all_finite":true});
            report["capture_d2h_bytes"] = json!(report["capture_d2h_bytes"].as_u64().unwrap()+features.len() as u64);
        }
        write_new(
            &self.directory.join("manifest.json"),
            &serde_json::to_vec_pretty(&report)
                .map_err(|_| invalid("boundary manifest serialization"))?,
        )?;
        Ok(report)
    }
}

impl ProjectionWindow {
    fn capture(
        &self,
        d: &Driver,
        w: &PrefillWorkspace,
        layer: usize,
        directory: &Path,
    ) -> Result<()> {
        let Some(layer_index) = PROJECTION_LAYERS.iter().position(|&item| item == layer) else {
            return Ok(());
        };
        let Some((first, rows)) = projection_span(self.start, self.end, d.cursor) else {
            return Ok(());
        };
        // The reference capture is made of complete B8 batches ending at the
        // last real prompt row. Reject partial/repeated/reordered captures.
        if first != 0
            || rows != B
            || d.cursor < self.start
            || (d.cursor - self.start) % B != 0
            || self.inventory.borrow().len()
                != ((d.cursor - self.start) / B * PROJECTION_LAYERS.len() + layer_index)
                    * PROJECTION_PLANES
        {
            return Err(invalid("projection corpus batch/layer order"));
        }
        self.capture_span(d, w, layer, d.cursor, 0, B, B, directory)
    }

    fn capture_wide_history(&self, d: &Driver, w: &PrefillWorkspace, layer: usize, batch_rows: usize, directory: &Path) -> Result<()> {
        let Some(layer_index) = PROJECTION_LAYERS.iter().position(|&item| item == layer) else { return Ok(()); };
        let end = d.cursor.checked_add(batch_rows).ok_or_else(|| invalid("wide history trace overflow"))?;
        if end <= self.start || d.cursor >= self.end { return Ok(()); }
        // The registered <=512-row tail window must fit the final real block.
        // Keep exactly the same B8 slices/filenames, now read layer-major.
        if batch_rows != 512 || self.start < d.cursor || self.end != end
            || (self.start-d.cursor)%B != 0 || (self.end-self.start)%B != 0
            || self.inventory.borrow().len() != layer_index*((self.end-self.start)/B)*PROJECTION_PLANES {
            return Err(invalid("wide history projection window/order"));
        }
        for position in (self.start..self.end).step_by(B) {
            self.capture_span(d,w,layer,position,position-d.cursor,B,batch_rows,directory)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn capture_span(&self, d: &Driver, w: &PrefillWorkspace, layer: usize, position: usize,
                    first: usize, rows: usize, batch_rows: usize, directory: &Path) -> Result<()> {
        let q8_stride =
            |width: usize| width / Q8_1_ACTIVATION_BLOCK_ELEMENTS * Q8_1_ACTIVATION_BLOCK_BYTES;
        let sources = [
            (
                "attn_norm_q8_1",
                view(&w.normalized_q8_1)?,
                H,
                q8_stride(H),
                "q8_1_32x36",
            ),
            (
                "ffn_input_q8_1",
                view(&w.ffn_input_q8_1)?,
                H,
                q8_stride(H),
                "q8_1_32x36",
            ),
            (
                "ffn_down_q8_1",
                view(&w.ffn_down_q8_1)?,
                linear::INTERMEDIATE,
                q8_stride(linear::INTERMEDIATE),
                "q8_1_32x36",
            ),
            (
                "attn_residual",
                view(&w.attention_residual)?,
                H,
                H * 4,
                "f32_le",
            ),
            ("post_norm", view(&w.post_norm)?, H, H * 4, "f32_le"),
            (
                "ffn_gate",
                view(&w.ffn_gate)?,
                linear::INTERMEDIATE,
                linear::INTERMEDIATE * 4,
                "f32_le",
            ),
            (
                "ffn_up",
                view(&w.ffn_up)?,
                linear::INTERMEDIATE,
                linear::INTERMEDIATE * 4,
                "f32_le",
            ),
            ("ffn_output", view(&w.ffn_output)?, H, H * 4, "f32_le"),
            (
                "layer_output",
                w.hidden_output(layer, batch_rows)?.buffer_view(),
                H,
                H * 4,
                "f32_le",
            ),
        ];
        let receipt = d.stream.submission_receipt()?;
        for (name, source, width, stride, format) in sources {
            let mut bytes = vec![0u8; rows * stride];
            let proof = source
                .subview(first * stride, bytes.len())?
                .copy_to_host_sync_with_submission_proof(&d.stream, &mut bytes, receipt)?;
            if !proof.covers_submission(receipt) || !proof.belongs_to_stream(&d.stream)? {
                return Err(invalid("projection corpus owner/epoch proof"));
            }
            if format == "f32_le"
                && bytes
                    .chunks_exact(4)
                    .any(|word| !f32::from_le_bytes(word.try_into().unwrap()).is_finite())
            {
                return Err(invalid("projection corpus nonfinite F32"));
            }
            let filename = format!("projection-{position}-layer{layer}-{name}.bin");
            write_new(&directory.join(&filename), &bytes)?;
            self.inventory
                .borrow_mut()
                .push(json!({"file":filename,"name":name,"layer":layer,
                "start_position":position,"end_position_exclusive":position+rows,"rows":rows,
                "width":width,"row_bytes":stride,"format":format,"bytes":bytes.len(),
                "sha256":sha256_hex(&bytes),"actual_producer_input":true}));
        }
        Ok(())
    }

    fn finish(&self) -> Result<Value> {
        let expected = (self.end - self.start) / B * PROJECTION_LAYERS.len() * PROJECTION_PLANES;
        if self.inventory.borrow().len() != expected {
            return Err(invalid("projection corpus incomplete"));
        }
        let total: u64 = self
            .inventory
            .borrow()
            .iter()
            .map(|item| item["bytes"].as_u64().unwrap())
            .sum();
        Ok(
            json!({"schema":"qwen35-reference-projection-window-v1","start_position":self.start,
            "end_position_exclusive":self.end,"rows":self.end-self.start,"layers":PROJECTION_LAYERS,
            "tensors":*self.inventory.borrow(),"capture_d2h_bytes":total,"host_serialized_diagnostic":true,
            "repeated_or_synthetic_rows":false,"product_qualification":false,"performance_claim_permitted":false}),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn boundary_directory_and_files_never_overwrite_evidence() {
        let name = format!(
            "qwen35-boundary-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let directory = std::env::temp_dir().join(name);
        let trace = BoundaryTrace::new_with_projection_rows(&directory, 7, 0).unwrap();
        assert!(trace.prefix_disjoint(7));
        assert!(!trace.prefix_disjoint(8));
        assert!(BoundaryTrace::new_with_projection_rows(&directory, 7, 0).is_err());
        let path = directory.join("owned-test.bin");
        write_new(&path, b"frozen").unwrap();
        assert!(write_new(&path, b"replacement").is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"frozen");
        drop(trace);
        std::fs::remove_file(path).unwrap();
        std::fs::remove_dir(directory).unwrap();
    }

    #[test]
    fn boundary_selects_only_actual_prompt_row() {
        assert_eq!(selected_row(5631, 5638), Some(7));
        assert_eq!(selected_row(5623, 5638), None);
        assert_eq!(selected_row(5639, 5638), None);
        assert_eq!(selected_row(usize::MAX, 0), None);
        assert_eq!(selected_row(0, 0), Some(0));
        assert_eq!(NAMES.len() * 64 * H * 4, 9_175_040);
        assert_eq!(full_cache_bytes(5638).unwrap(), 5639 * 1088);
        assert!(full_cache_bytes(8192).is_err());
        assert!(full_cache_bytes(usize::MAX).is_err());
    }

    #[test]
    fn projection_window_requires_bounded_explicit_rows() {
        assert_eq!(projection_rows(None).unwrap(), 0);
        for rows in ["64", "128", "256", "512"] {
            assert!(projection_rows(Some(std::ffi::OsStr::new(rows))).is_ok());
        }
        for rows in [
            "", "0", "8", "255", "513", "1024", "0512", "512 ", "64 ", "-64",
        ] {
            assert!(projection_rows(Some(std::ffi::OsStr::new(rows))).is_err());
        }
        assert_eq!(projection_span(3968, 4096, 3968), Some((0, 8)));
        assert_eq!(projection_span(3968, 4096, 4088), Some((0, 8)));
        assert_eq!(projection_span(3968, 4096, 4096), None);
        assert_eq!(projection_span(3968, 4096, 3960), None);
        assert_eq!(projection_span(3968, 4096, usize::MAX), None);
        assert_eq!(projection_span(3584, 4096, 3584), Some((0, 8)));
        assert_eq!(projection_span(3584, 4096, 3576), None);
        assert_eq!(projection_span(3584, 4096, 4088), Some((0, 8)));
        assert_eq!(projection_span(3584, 4096, 4096), None);
        let window = ProjectionWindow {
            start: 3968,
            end: 4096,
            inventory: RefCell::new(Vec::new()),
        };
        assert!(window.finish().is_err());
    }
}
