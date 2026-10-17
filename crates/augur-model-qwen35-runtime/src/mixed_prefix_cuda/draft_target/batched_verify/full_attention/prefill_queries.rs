//! Input-only staging: original B8 or explicit, bounded canonical score tiles.
use super::*;
use augur_backend::{CudaQwen35CanonicalPrefillV1, Qwen35CanonicalPrefillScorePolicy};
use augur_backend::{CudaQwen35Kv8BatchedAttentionArgsV4, CudaQwen35Kv8BatchedAttentionViewsV4};
const Q: usize = full::QUERY_VALUES;
const K: usize = full::KV_VALUES;
const W: usize = augur_backend::QWEN35_FULL_ATTENTION_WORKSPACE_ELEMENTS_V2;

pub(in super::super) struct PrefillQueries {
    rotated: bool,
    rows: usize,
    query: OwnedCudaBuffer,
    gate: OwnedCudaBuffer,
    key: OwnedCudaBuffer,
    value: OwnedCudaBuffer,
    attention: OwnedCudaBuffer,
    gated: OwnedCudaBuffer,
    workspace: OwnedCudaBuffer,
    score_workspace: Option<OwnedCudaBuffer>,
    score_policy: Option<Qwen35CanonicalPrefillScorePolicy>,
    group_rows: usize,
}

fn group_range(rows: usize, start: usize) -> Result<std::ops::Range<usize>> {
    if rows == B {
        return Err(invalid("wide prefix owner requires more than B8 rows"));
    }
    group_range_with_width(rows, start, B)
}

fn group_range_with_width(
    rows: usize,
    start: usize,
    width: usize,
) -> Result<std::ops::Range<usize>> {
    prefill_width::validate(rows)?;
    if !matches!(width, 8 | 128)
        || width > rows
        || !rows.is_multiple_of(width)
        || start >= rows
        || !start.is_multiple_of(width)
        || start.checked_add(width).is_none_or(|end| end > rows)
    {
        return Err(invalid("wide prefix shared-query group extent"));
    }
    Ok(start..start + width)
}

fn score_selection(
    requested: Option<Qwen35CanonicalPrefillScorePolicy>,
    rows: usize,
) -> (Option<Qwen35CanonicalPrefillScorePolicy>, usize) {
    let selected = requested.filter(|_| rows >= CudaQwen35CanonicalPrefillV1::MAX_ROWS);
    let width = if selected.is_some() {
        CudaQwen35CanonicalPrefillV1::MAX_ROWS
    } else {
        B
    };
    (selected, width)
}

fn original_gated_range(rows: usize, start: usize, count: usize) -> Result<std::ops::Range<usize>> {
    if rows != 512 || count == 0 || start.checked_add(count).is_none_or(|end| end > rows) {
        return Err(invalid("original gated destination is bounded B512 prefix only"));
    }
    Ok(start * Q..(start + count) * Q)
}

fn rotated_gated_layout(rows: usize, owner_rows: usize, original_f32: bool) -> Result<(std::ops::Range<usize>, bool)> {
    if !matches!(rows, 1..=8 | 512) || rows > owner_rows || (original_f32 && rows != 512) {
        return Err(invalid("rotated wide owner requires B512 or B1..8"));
    }
    Ok((0..rows * Q, !original_f32))
}

fn exact_input_gated_layout(rows: usize, owner_rows: usize) -> Result<std::ops::Range<usize>> {
    prefill_width::validate_exact_input(rows)?;
    if rows > owner_rows { return Err(invalid("exact input attention owner extent")); }
    Ok(0..rows * Q)
}

impl PrefillQueries {
    pub(in super::super) fn request_policy(
        value: Option<&std::ffi::OsStr>,
    ) -> Result<Option<Qwen35CanonicalPrefillScorePolicy>> {
        match value.and_then(std::ffi::OsStr::to_str) {
            None if value.is_none() => Ok(None),
            Some("shared-b8") => Ok(None),
            Some("warp") => Ok(Some(Qwen35CanonicalPrefillScorePolicy::CanonicalWarp)),
            Some("dgemm") => Ok(Some(Qwen35CanonicalPrefillScorePolicy::Dgemm)),
            _ => Err(invalid("invalid AUGUR_QWEN35_CANONICAL_PREFILL_SCORE_MODE")),
        }
    }

    pub(in super::super) fn score_mode_name(&self) -> &'static str {
        if self.rotated { return "llama-rotated-b512-small-tail-v1"; }
        match self.score_policy {
            None => "shared-b8",
            Some(Qwen35CanonicalPrefillScorePolicy::CanonicalWarp) => {
                "canonical-warp-staged-scores"
            }
            Some(Qwen35CanonicalPrefillScorePolicy::Dgemm) => "fp64-dgemm-staged-scores",
        }
    }

    pub(in super::super) fn new(d: &Driver, rows: usize) -> Result<Self> {
        group_range(rows, 0)?;
        let requested = Self::request_policy(
            std::env::var_os("AUGUR_QWEN35_CANONICAL_PREFILL_SCORE_MODE").as_deref(),
        )?;
        // Shape policy: real small prefix batches retain the original B8 path.
        // The independent verify core never constructs this wide-prefix owner.
        let (score_policy, group_rows) = score_selection(requested, rows);
        let input =
            |width: usize| OwnedCudaBuffer::allocate(d.stream.device_ordinal(), rows * width * 4);
        // These outputs are consumed and staged before the next query
        // group. Their lifetime, not the projection width, sets scratch size.
        let scratch = |width: usize| {
            OwnedCudaBuffer::allocate(d.stream.device_ordinal(), group_rows * width * 4)
        };
        Ok(Self {
            rotated: d.llama_rotated_kv,
            rows,
            query: input(Q)?,
            gate: input(Q)?,
            key: input(K)?,
            value: input(K)?,
            attention: if d.llama_rotated_kv { input(Q)? } else { scratch(Q)? },
            gated: scratch(Q)?,
            workspace: scratch(W)?,
            score_workspace: score_policy
                .map(|_| -> Result<OwnedCudaBuffer> {
                    let elements =
                        CudaQwen35CanonicalPrefillV1::score_elements(group_rows, d.capacity)?;
                    Ok(OwnedCudaBuffer::allocate(
                        d.stream.device_ordinal(),
                        elements * 4,
                    )?)
                })
                .transpose()?,
            score_policy,
            group_rows,
        })
    }

    pub(super) fn capture(&self, d: &Driver, plan: &FullPlan, row: usize) -> Result<()> {
        if row >= self.rows {
            return Err(invalid("wide prefix query row"));
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

    pub(super) fn stage_batched(&self, d: &Driver, w: &PrefillWorkspace, plan: &FullPlan, rows: usize) -> Result<()> {
        use augur_backend::{CudaQwen35BatchedHeadNormRopeArgsV1, CudaQwen35BatchedHeadNormRopeViewsV1};
        if !d.prefill_batched_head_rope || !d.llama_exact_input_rows || !self.rotated
            || rows == 0 || rows > self.rows || rows > w.max_tokens
            || d.cursor.checked_add(rows).is_none_or(|end| end > d.capacity) {
            return Err(invalid("batched head RoPE input-only owner extent"));
        }
        let reference = plan.fused_head_norm_rope_ops.get(d.cursor)
            .ok_or_else(|| invalid("batched head RoPE scalar contract"))?.as_raw();
        let half = augur_backend::QWEN35_FULL_ATTENTION_ROTARY_DIM / 2;
        CudaQwen35BatchedHeadNormRopeArgsV1::bind(&d.stream,
            CudaQwen35BatchedHeadNormRopeViewsV1 {
                q_gate:plan_f32_subview(&w.qkv,0,rows*2*Q)?,
                q_weight:plan_f32_view(&plan.q_norm_weight,Q)?,
                query:plan_f32_subview(&self.query,0,rows*Q)?,
                key_input:plan_f32_subview(&w.z,0,rows*K)?,
                k_weight:plan_f32_view(&plan.k_norm_weight,K)?,
                key:plan_f32_subview(&self.key,0,rows*K)?,
                rope_cos:plan_f32_subview(&plan.rope_tables.cos,d.cursor*half,rows*half)?,
                rope_sin:plan_f32_subview(&plan.rope_tables.sin,d.cursor*half,rows*half)?,
                gate:plan_f32_subview(&self.gate,0,rows*Q)?,
                value_input:plan_f32_subview(&w.recurrent,0,rows*K)?,
                value:plan_f32_subview(&self.value,0,rows*K)?,
            },rows,f32::from_bits(reference.epsilon_bits))?.submit_for_plan(&d.stream)?;
        d.prefill_batched_head_rope_launches.set(d.prefill_batched_head_rope_launches.get() + 1);
        d.prefill_batched_head_rope_layer_rows.set(d.prefill_batched_head_rope_layer_rows.get() + rows as u64);
        Ok(())
    }

    pub(super) fn submit(
        &self,
        d: &Driver,
        w: &PrefillWorkspace,
        plan: &FullPlan,
        rows: usize,
        original_f32: bool,
    ) -> Result<()> {
        let exact_input = d.llama_exact_input_rows && original_f32;
        if (!exact_input && rows != self.rows && rows != B)
            || rows > self.rows
            || rows > w.max_tokens
            || d.cursor
                .checked_add(rows)
                .is_none_or(|end| end > d.capacity)
        {
            return Err(invalid("wide prefix shared-query submission extent"));
        }
        if let Some(rotated) = &plan.rotated_attention {
            if self.score_policy.is_some() {
                return Err(invalid("rotated attention cannot mix canonical score policy"));
            }
            // Initialized history also runs B512 but intentionally retains
            // integer projections. Its gated F32 output must use the full dead
            // FFN-up arena, not the eight-row legacy gate scratch. Quantize it
            // before the output projection when that projection consumes Q8_1.
            let (range, needs_quantize) = if exact_input {
                (exact_input_gated_layout(rows, self.rows)?, false)
            } else { rotated_gated_layout(rows, self.rows, original_f32)? };
            let gated = w.ffn_up.view(range.start * 4, range.len() * 4)?;
            let submit = if exact_input { RotatedFullAttention::submit_exact_input }
                else { RotatedFullAttention::submit };
            submit(rotated, &d.stream, plan, d.cursor, rows,
                self.query.view(0, rows * Q * 4)?, self.gate.view(0, rows * Q * 4)?,
                self.key.view(0, rows * K * 4)?, self.value.view(0, rows * K * 4)?,
                self.attention.view(0, rows * Q * 4)?, gated)?;
            if needs_quantize {
                CudaQ8_1ActivationQuantizeArgsV1::bind(&d.stream,
                    plan_f32_subview(&w.ffn_up, range.start, range.len())?,
                    CudaQ8_1ActivationViewV1::new(w.ffn_down_q8_1.view(0,
                        rows * QWEN35_FULL_ATTENTION_FUSED_Q8_BYTES_V2)?,
                        PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_1_ACTIVATION_CUDA_V1, rows * Q)?)?
                    .submit_for_plan(&d.stream)?;
            }
            return Ok(());
        }
        if exact_input { return Err(invalid("exact input cannot fall back to unrotated attention")); }
        // The final B8 group uses exact subviews of this same wide owner.
        // It never invokes the 128-query score policy or adds a second arena.
        let (score_policy, group_rows) = if rows == self.rows {
            (self.score_policy, self.group_rows)
        } else {
            score_selection(self.score_policy, rows)
        };
        for start in (0..rows).step_by(group_rows) {
            let group = group_range_with_width(rows, start, group_rows)?;
            let n = group.len();
            let views = CudaQwen35Kv8BatchedAttentionViewsV4 {
                query: plan_f32_subview(&self.query, start * Q, n * Q)?,
                gate: plan_f32_subview(&self.gate, start * Q, n * Q)?,
                key: plan_f32_subview(&self.key, start * K, n * K)?,
                value: plan_f32_subview(&self.value, start * K, n * K)?,
                key_cache: plan.key_cache.view(0, plan.key_cache.len_bytes())?,
                value_cache: plan.value_cache.view(0, plan.value_cache.len_bytes())?,
                attention: plan_f32_subview(&self.attention, 0, n * Q)?,
                // Preserve real F32 gated output for direct projection. The
                // standalone PrefillFullOps gated destination is not executed
                // by this canonical attention branch. Write the final batch
                // destination directly, before bounded scratch gets reused.
                gated: if original_f32 {
                    let range = original_gated_range(rows, start, n)?;
                    plan_f32_subview(&w.ffn_up, range.start, range.len())?
                } else { plan_f32_subview(&self.gated, 0, n * Q)? },
                workspace: plan_f32_subview(&self.workspace, 0, n * W)?,
            };
            match (&self.score_workspace, score_policy) {
                (_, None) => CudaQwen35Kv8BatchedAttentionArgsV4::bind(
                    &d.stream,
                    views,
                    d.capacity,
                    d.cursor + start,
                    n,
                )?
                .submit_for_plan(&d.stream)?,
                (Some(scores), Some(policy)) => CudaQwen35CanonicalPrefillV1::bind(
                    &d.stream,
                    &w.cublas,
                    views,
                    d.capacity,
                    d.cursor + start,
                    n,
                    scores.view(0, scores.len_bytes())?,
                    policy,
                )?
                .submit(&d.stream, &w.cublas)?,
                _ => return Err(invalid("canonical prefix score workspace policy drift")),
            }
            if original_f32 { continue; }
            // Same-stream copies complete before the next group overwrites
            // scratch. Only this staged Q8 result is used by output projection.
            for row in 0..n {
                copy(
                    &d.stream,
                    &w.ffn_down_q8_1,
                    (start + row) * QWEN35_FULL_ATTENTION_FUSED_Q8_BYTES_V2,
                    self.workspace.view(
                        (row * W + QWEN35_FULL_ATTENTION_PARTIAL_ELEMENTS_V2) * 4,
                        QWEN35_FULL_ATTENTION_FUSED_Q8_BYTES_V2,
                    )?,
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_input_rotated_runtime_admission_does_not_expand_decode_or_verify() {
        for rows in 1..=512 {
            assert!(RotatedFullAttention::validate_rows(rows, true).is_ok());
            assert_eq!(RotatedFullAttention::validate_rows(rows, false).is_ok(),
                matches!(rows, 1..=8 | 512));
        }
        for rows in [0, 513, usize::MAX] {
            assert!(RotatedFullAttention::validate_rows(rows, true).is_err());
            assert!(RotatedFullAttention::validate_rows(rows, false).is_err());
        }
    }

    #[test]
    fn exact_input_rotated_layout_is_independent_of_legacy_b8_admission() {
        for rows in 1..=512 {
            assert_eq!(exact_input_gated_layout(rows,512).unwrap(),0..rows*Q);
        }
        for (rows,owner) in [(0,512),(513,512),(8,7),(512,511)] {
            assert!(exact_input_gated_layout(rows,owner).is_err());
        }
        assert!(rotated_gated_layout(8,512,true).is_err());
        assert!(rotated_gated_layout(9,512,false).is_err());
    }

    #[test]
    fn rotated_prefix_and_initialized_history_share_full_gated_extent() {
        for rows in [1, 7, 8, 512] {
            assert_eq!(rotated_gated_layout(rows, 512, false).unwrap(), (0..rows * Q, true));
        }
        // Original-F32 prefix and integer-projection history use identical
        // storage; only the downstream Q8_1 quantization policy differs.
        for original_f32 in [false, true] {
            let (range, quantize) = rotated_gated_layout(512, 512, original_f32).unwrap();
            assert_eq!(range.len(), 512 * Q);
            assert_eq!(quantize, !original_f32);
        }
        for (rows, owner) in [(0, 512), (9, 512), (511, 512), (513, 512), (512, 8)] {
            assert!(rotated_gated_layout(rows, owner, false).is_err());
        }
        assert!(rotated_gated_layout(8, 512, true).is_err());
    }

    #[test]
    fn original_gated_output_covers_all_rows_without_history_or_overlap() {
        for width in [8,128] {
            let ranges=(0..512).step_by(width).map(|start|
                original_gated_range(512,start,width).unwrap()).collect::<Vec<_>>();
            assert_eq!(ranges[0].start,0);
            assert_eq!(ranges.last().unwrap().end,512*Q);
            assert!(ranges.windows(2).all(|p| p[0].end==p[1].start));
        }
        for (rows,start,count) in [(8,0,8),(256,0,8),(512,512,1),(512,511,2),
            (512,0,0),(512,usize::MAX,8)] {
            assert!(original_gated_range(rows,start,count).is_err());
        }
    }

    #[test]
    fn wider_prefix_still_uses_only_real_b8_query_groups() {
        for rows in [16, 32, 64, 128, 256, 512] {
            let positions = (0..rows)
                .step_by(B)
                .flat_map(|start| group_range(rows, start).unwrap())
                .collect::<Vec<_>>();
            assert_eq!(positions, (0..rows).collect::<Vec<_>>());
            for start in (0..rows).step_by(B) {
                assert_eq!(group_range(rows, start).unwrap().len(), B);
            }
            for start in [1, 7, rows, rows + 8, usize::MAX] {
                assert!(group_range(rows, start).is_err());
            }
        }
        for rows in [0, 8, 24, 129, 768, usize::MAX] {
            assert!(group_range(rows, 0).is_err());
        }
    }

    #[test]
    fn every_group_stages_complete_outputs_before_bounded_scratch_reuse() {
        let scratch_bytes = B * W * 4;
        for rows in [16, 32, 64, 128, 256, 512] {
            for start in (0..rows).step_by(B) {
                let group = group_range(rows, start).unwrap();
                for row in 0..group.len() {
                    let source = (row * W + QWEN35_FULL_ATTENTION_PARTIAL_ELEMENTS_V2) * 4;
                    assert!(source + QWEN35_FULL_ATTENTION_FUSED_Q8_BYTES_V2 <= scratch_bytes);
                    assert_eq!(start + row, group.start + row);
                    assert!(start + row < rows);
                }
            }
        }
    }

    #[test]
    fn staged_score_mode_is_explicit_and_small_batches_keep_b8() {
        use std::ffi::OsStr;
        assert_eq!(PrefillQueries::request_policy(None).unwrap(), None);
        assert_eq!(
            PrefillQueries::request_policy(Some(OsStr::new("shared-b8"))).unwrap(),
            None
        );
        for mode in ["warp", "dgemm"] {
            let requested = PrefillQueries::request_policy(Some(OsStr::new(mode))).unwrap();
            assert!(requested.is_some());
            for rows in [8, 16, 32, 64] {
                assert_eq!(score_selection(requested, rows), (None, B));
            }
            for rows in [128, 256, 512] {
                assert_eq!(score_selection(requested, rows), (requested, 128));
            }
        }
        for mode in ["", "1", "true", "DGEMM", "dgemm ", "legacy"] {
            assert!(PrefillQueries::request_policy(Some(OsStr::new(mode))).is_err());
        }
        for rows in [128, 256, 512] {
            let positions = (0..rows)
                .step_by(128)
                .flat_map(|start| group_range_with_width(rows, start, 128).unwrap())
                .collect::<Vec<_>>();
            assert_eq!(positions, (0..rows).collect::<Vec<_>>());
        }
        for rows in [8, 16, 32, 64] {
            assert!(group_range_with_width(rows, 0, 128).is_err());
        }
        for start in [1, 127, 129, 512, usize::MAX] {
            assert!(group_range_with_width(512, start, 128).is_err());
        }
    }

    #[test]
    fn final_b8_uses_exact_subviews_and_never_the_wide_score_policy() {
        assert_eq!(group_range_with_width(B, 0, B).unwrap(), 0..B);
        for requested in [
            None,
            Some(Qwen35CanonicalPrefillScorePolicy::CanonicalWarp),
            Some(Qwen35CanonicalPrefillScorePolicy::Dgemm),
        ] {
            assert_eq!(score_selection(requested, B), (None, B));
        }
        for width in [8, 128] {
            assert!(group_range_with_width(B, B, width).is_err());
            assert!(group_range_with_width(B - 1, 0, width).is_err());
        }
    }
}
