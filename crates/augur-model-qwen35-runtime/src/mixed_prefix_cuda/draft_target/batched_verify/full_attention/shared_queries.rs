//! One reusable eight-row staging allocation set for all full-attention layers.
use super::*;
use augur_backend::{CudaQwen35Kv8BatchedAttentionArgsV4, CudaQwen35Kv8BatchedAttentionViewsV4};
const Q: usize = full::QUERY_VALUES;
const K: usize = full::KV_VALUES;
const W: usize = augur_backend::QWEN35_FULL_ATTENTION_WORKSPACE_ELEMENTS_V2;

pub(in super::super) struct SharedQueries {
    batched_stage: bool,
    batched_stage_submitted: std::cell::Cell<usize>,
    diagnostic_serial_rows: bool,
    vector_rows: bool,
    serial_rows_submitted: std::cell::Cell<usize>,
    vector_rows_submitted: std::cell::Cell<usize>,
    query: OwnedCudaBuffer,
    gate: OwnedCudaBuffer,
    key: OwnedCudaBuffer,
    value: OwnedCudaBuffer,
    attention: OwnedCudaBuffer,
    gated: OwnedCudaBuffer,
    workspace: OwnedCudaBuffer,
}

impl SharedQueries {
    pub(in super::super) const BYTES: usize = B * (4 * Q + 2 * K + W) * 4;
    pub(in super::super) const STAGING_COPY_BYTES_PER_LAYER: usize = B * (2 * Q + 2 * K) * 4;

    pub(in super::super) fn new(d: &Driver) -> Result<Self> {
        let batched_stage = batched_stage_switch(
            std::env::var_os("AUGUR_QWEN35_VERIFY_BATCHED_HEAD_ROPE").as_deref(),
            d.prefill_batched_head_rope && d.llama_exact_input_rows && d.llama_rotated_kv)?;
        if std::env::var_os("AUGUR_QWEN35_SMALL_ATTENTION_TILE8_ROWS").is_some() && !d.llama_rotated_kv {
            return Err(invalid("tile8 row contract requires a rotated shared-query owner"));
        }
        let diagnostic_serial_rows = serial_rows_switch(
            std::env::var_os("AUGUR_QWEN35_VERIFY_ATTENTION_SERIAL_ROWS").as_deref(),
            std::env::var_os("AUGUR_QWEN35_REPLAY_NUMERIC_DIR").is_some()
                && std::env::var("AUGUR_QWEN35_REPLAY_SSM_AUDIT").as_deref() == Ok("1")
                && d.llama_rotated_kv)?;
        let vector_rows = vector_rows_switch(
            std::env::var_os("AUGUR_QWEN35_VERIFY_ATTENTION_VECTOR_ROWS").as_deref(),
            d.llama_rotated_kv && !diagnostic_serial_rows)?;
        let alloc = |n| OwnedCudaBuffer::allocate(d.stream.device_ordinal(), B * n * 4);
        Ok(Self {
            batched_stage,
            batched_stage_submitted: std::cell::Cell::new(0),
            diagnostic_serial_rows,
            vector_rows,
            serial_rows_submitted: std::cell::Cell::new(0),
            vector_rows_submitted: std::cell::Cell::new(0),
            query: alloc(Q)?,
            gate: alloc(Q)?,
            key: alloc(K)?,
            value: alloc(K)?,
            attention: alloc(Q)?,
            gated: alloc(Q)?,
            workspace: alloc(W)?,
        })
    }

    pub(in super::super) fn serial_rows_submitted(&self) -> usize {
        self.serial_rows_submitted.get()
    }

    pub(in super::super) fn vector_rows_submitted(&self) -> usize {
        self.vector_rows_submitted.get()
    }

    pub(in super::super) fn batched_stage_submitted(&self) -> usize {
        self.batched_stage_submitted.get()
    }

    pub(in super::super) fn staging_copy_bytes_per_layer(&self, rotated: bool) -> usize {
        staging_copy_bytes(self.batched_stage, rotated)
    }

    // Consume the already contiguous projection rows directly. The same
    // head-owned reduction is used by the scalar oracle; no attention, Q8
    // quantization, rollback state or arithmetic policy changes here.
    pub(super) fn stage_batched(
        &self, d: &Driver, w: &PrefillWorkspace, plan: &FullPlan,
        rows: usize, original_f32: bool,
    ) -> Result<bool> {
        use augur_backend::{CudaQwen35BatchedHeadNormRopeArgsV1, CudaQwen35BatchedHeadNormRopeViewsV1};
        if !self.batched_stage { return Ok(false); }
        validate_batched_stage(rows, original_f32)?;
        if !d.prefill_batched_head_rope || !d.llama_exact_input_rows || !d.llama_rotated_kv
            || plan.rotated_attention.is_none() || rows > w.max_tokens
            || d.cursor.checked_add(rows).is_none_or(|end| end > d.capacity
                || end > plan.fused_head_norm_rope_ops.len()) {
            return Err(invalid("batched verifier head RoPE owner extent"));
        }
        let reference = plan.fused_head_norm_rope_ops[d.cursor].as_raw();
        let half = augur_backend::QWEN35_FULL_ATTENTION_ROTARY_DIM / 2;
        let next = self.batched_stage_submitted.get().checked_add(1)
            .ok_or_else(|| invalid("batched verifier head RoPE counter overflow"))?;
        CudaQwen35BatchedHeadNormRopeArgsV1::bind(&d.stream,
            CudaQwen35BatchedHeadNormRopeViewsV1 {
                q_gate: plan_f32_subview(&w.qkv, 0, rows * 2 * Q)?,
                q_weight: plan_f32_view(&plan.q_norm_weight, Q)?,
                query: plan_f32_subview(&self.query, 0, rows * Q)?,
                key_input: plan_f32_subview(&w.z, 0, rows * K)?,
                k_weight: plan_f32_view(&plan.k_norm_weight, K)?,
                key: plan_f32_subview(&self.key, 0, rows * K)?,
                rope_cos: plan_f32_subview(&plan.rope_tables.cos, d.cursor * half, rows * half)?,
                rope_sin: plan_f32_subview(&plan.rope_tables.sin, d.cursor * half, rows * half)?,
                gate: plan_f32_subview(&self.gate, 0, rows * Q)?,
                value_input: plan_f32_subview(&w.recurrent, 0, rows * K)?,
                value: plan_f32_subview(&self.value, 0, rows * K)?,
            }, rows, f32::from_bits(reference.epsilon_bits))?.submit_for_plan(&d.stream)?;
        self.batched_stage_submitted.set(next);
        Ok(true)
    }

    pub(super) fn capture(&self, d: &Driver, plan: &FullPlan, row: usize) -> Result<()> {
        if row >= B {
            return Err(invalid("shared KV query row"));
        }
        copy(
            &d.stream,
            &self.query,
            row * Q * 4,
            plan.query.view(0, Q * 4)?,
        )?;
        copy(
            &d.stream,
            &self.gate,
            row * Q * 4,
            plan.q_full.view(Q * 4, Q * 4)?,
        )?;
        copy(&d.stream, &self.key, row * K * 4, plan.key.view(0, K * 4)?)?;
        copy(
            &d.stream,
            &self.value,
            row * K * 4,
            plan.value.view(0, K * 4)?,
        )?;
        Ok(())
    }

    pub(super) fn submit(
        &self,
        d: &Driver,
        w: &PrefillWorkspace,
        plan: &FullPlan,
        rows: usize,
    ) -> Result<()> {
        if rows == 0 || rows > B {
            return Err(invalid("shared KV query extent"));
        }
        if let Some(rotated) = &plan.rotated_attention {
            if self.diagnostic_serial_rows {
                for row in 0..rows {
                    rotated.submit(&d.stream, plan, d.cursor + row, 1,
                        self.query.view(row * Q * 4, Q * 4)?, self.gate.view(row * Q * 4, Q * 4)?,
                        self.key.view(row * K * 4, K * 4)?, self.value.view(row * K * 4, K * 4)?,
                        self.attention.view(row * Q * 4, Q * 4)?, self.gated.view(row * Q * 4, Q * 4)?)?;
                    self.serial_rows_submitted.set(self.serial_rows_submitted.get().checked_add(1)
                        .ok_or_else(|| invalid("diagnostic attention row counter overflow"))?);
                }
            } else if self.vector_rows {
                rotated.submit_vector_rows(&d.stream, plan, d.cursor, rows,
                    self.query.view(0, rows * Q * 4)?, self.gate.view(0, rows * Q * 4)?,
                    self.key.view(0, rows * K * 4)?, self.value.view(0, rows * K * 4)?,
                    self.attention.view(0, rows * Q * 4)?, self.gated.view(0, rows * Q * 4)?)?;
                self.vector_rows_submitted.set(self.vector_rows_submitted.get().checked_add(rows)
                    .ok_or_else(|| invalid("vector attention row counter overflow"))?);
            } else {
                rotated.submit(&d.stream, plan, d.cursor, rows,
                    self.query.view(0, rows * Q * 4)?, self.gate.view(0, rows * Q * 4)?,
                    self.key.view(0, rows * K * 4)?, self.value.view(0, rows * K * 4)?,
                    self.attention.view(0, rows * Q * 4)?, self.gated.view(0, rows * Q * 4)?)?;
            }
            CudaQ8_1ActivationQuantizeArgsV1::bind(&d.stream,
                plan_f32_subview(&self.gated, 0, rows * Q)?,
                CudaQ8_1ActivationViewV1::new(w.ffn_down_q8_1.view(0,
                    rows * QWEN35_FULL_ATTENTION_FUSED_Q8_BYTES_V2)?,
                    PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_1_ACTIVATION_CUDA_V1, rows * Q)?)?
                .submit_for_plan(&d.stream)?;
            return Ok(());
        }
        let views = CudaQwen35Kv8BatchedAttentionViewsV4 {
            query: plan_f32_subview(&self.query, 0, rows * Q)?,
            gate: plan_f32_subview(&self.gate, 0, rows * Q)?,
            key: plan_f32_subview(&self.key, 0, rows * K)?,
            value: plan_f32_subview(&self.value, 0, rows * K)?,
            key_cache: plan.key_cache.view(0, plan.key_cache.len_bytes())?,
            value_cache: plan.value_cache.view(0, plan.value_cache.len_bytes())?,
            attention: plan_f32_subview(&self.attention, 0, rows * Q)?,
            gated: plan_f32_subview(&self.gated, 0, rows * Q)?,
            workspace: plan_f32_subview(&self.workspace, 0, rows * W)?,
        };
        CudaQwen35Kv8BatchedAttentionArgsV4::bind(&d.stream, views, d.capacity, d.cursor, rows)?
            .submit_for_plan(&d.stream)?;
        for row in 0..rows {
            copy(
                &d.stream,
                &w.ffn_down_q8_1,
                row * QWEN35_FULL_ATTENTION_FUSED_Q8_BYTES_V2,
                self.workspace.view(
                    (row * W + QWEN35_FULL_ATTENTION_PARTIAL_ELEMENTS_V2) * 4,
                    QWEN35_FULL_ATTENTION_FUSED_Q8_BYTES_V2,
                )?,
            )?;
        }
        Ok(())
    }
}

fn batched_stage_switch(value: Option<&std::ffi::OsStr>, recipe: bool) -> Result<bool> {
    match value {
        None => Ok(false),
        Some(v) if v == "1" && recipe => Ok(true),
        _ => Err(invalid("batched verifier head RoPE requires explicit exact rotated batched-head recipe")),
    }
}

fn validate_batched_stage(rows: usize, original_f32: bool) -> Result<()> {
    if rows != B || original_f32 {
        return Err(invalid("batched verifier head RoPE requires native B8 projections"));
    }
    Ok(())
}

fn staging_copy_bytes(batched: bool, rotated: bool) -> usize {
    let inputs = if batched { 0 } else {
        B * (2 * Q + 2 * K) * 4 + SharedQueries::STAGING_COPY_BYTES_PER_LAYER
    };
    // Rotated attention quantizes directly into the batch output arena.
    inputs + if rotated { 0 } else { B * QWEN35_FULL_ATTENTION_FUSED_Q8_BYTES_V2 }
}

fn serial_rows_switch(value: Option<&std::ffi::OsStr>, diagnostic: bool) -> Result<bool> {
    match value {
        None => Ok(false),
        Some(v) if v == "1" && diagnostic => Ok(true),
        _ => Err(invalid("serial verify attention requires explicit replay numeric diagnostic")),
    }
}

fn vector_rows_switch(value: Option<&std::ffi::OsStr>, eligible: bool) -> Result<bool> {
    match value {
        None => Ok(false),
        Some(v) if v == "1" && eligible => Ok(true),
        _ => Err(invalid("vector verify attention requires rotated KV and excludes serial ablation")),
    }
}

#[cfg(test)]
mod serial_row_tests {
    use super::*;
    #[test]
    fn batched_verify_stage_requires_explicit_complete_recipe() {
        use std::ffi::OsStr;
        for recipe in [false, true] {
            assert!(!batched_stage_switch(None, recipe).unwrap());
            assert_eq!(batched_stage_switch(Some(OsStr::new("1")), recipe).is_ok(), recipe);
            for bad in ["", "0", "true", "01", "1 "] {
                assert!(batched_stage_switch(Some(OsStr::new(bad)), recipe).is_err());
            }
        }
        assert!(validate_batched_stage(B, false).is_ok());
        for rows in [0, 1, 7, 9, 512, usize::MAX] {
            assert!(validate_batched_stage(rows, false).is_err());
        }
        assert!(validate_batched_stage(B, true).is_err());
    }
    #[test]
    fn staging_bill_counts_copies_not_replacement_kernel_traffic() {
        assert_eq!(staging_copy_bytes(true, true), 0);
        assert_eq!(staging_copy_bytes(false, true), 2 * B * (2 * Q + 2 * K) * 4);
        assert_eq!(staging_copy_bytes(false, false) - staging_copy_bytes(false, true),
            B * QWEN35_FULL_ATTENTION_FUSED_Q8_BYTES_V2);
    }
    #[test]
    fn vector_attention_requires_explicit_compatible_owner() {
        use std::ffi::OsStr;
        assert!(!vector_rows_switch(None, false).unwrap());
        assert!(vector_rows_switch(Some(OsStr::new("1")), true).unwrap());
        assert!(vector_rows_switch(Some(OsStr::new("1")), false).is_err());
        for value in ["0", "true", "", "1 "] {
            assert!(vector_rows_switch(Some(OsStr::new(value)), true).is_err());
        }
    }
    #[test]
    fn serial_attention_is_only_an_explicit_diagnostic() {
        use std::ffi::OsStr;
        assert!(!serial_rows_switch(None,false).unwrap());
        assert!(serial_rows_switch(Some(OsStr::new("1")),true).unwrap());
        assert!(serial_rows_switch(Some(OsStr::new("1")),false).is_err());
        for value in ["0","true","","1 "] { assert!(serial_rows_switch(Some(OsStr::new(value)),true).is_err()); }
    }
}
