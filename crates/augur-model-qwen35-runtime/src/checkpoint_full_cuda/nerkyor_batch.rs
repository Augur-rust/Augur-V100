//! Research B8 full attention over the same unrotated F32 KV allocation.
//! Tentative writes are bounded to eight rows; unaccepted rows are restored.
use super::*;
use crate::checkpoint_projection_cuda::CheckpointExactBatchProjection;
use crate::checkpoint_target_cuda::nerkyor_dflash_research::batched::{
    copy, norms, slice, Ffn, Workspace, H, WIDTH,
};
use augur_backend::cuda_qwen35_prefill_cublas::{
    CudaQwen35PrefillFullAttentionV1, CudaQwen35PrefillFullAttentionViewsV1,
};
pub(crate) struct Program {
    w: Rc<Workspace>,
    _input: Buffer,
    _output: Buffer,
    vectors: BTreeMap<&'static str, Buffer>,
    cache: [Buffer; 2],
    backup: [Buffer; 2],
    input_norm: Vec<CudaRmsnormF32ArgsV1>,
    post_norm: Vec<CudaRmsnormF32ArgsV1>,
    q: CheckpointExactBatchProjection,
    k: CheckpointExactBatchProjection,
    v: CheckpointExactBatchProjection,
    out: CheckpointExactBatchProjection,
    residual: CudaAddF32ArgsV1,
    ffn: Ffn,
    final_add: CudaAddF32ArgsV1,
    capacity: usize,
    epsilon: f32,
    pending: Option<usize>,
    core_kv: Option<[Buffer; 2]>,
}
impl Program {
    pub(crate) fn bind(
        layer: &CheckpointFullCudaLayer,
        w: Rc<Workspace>,
        input: Buffer,
        output: Buffer,
        epsilon: f32,
    ) -> Result<Self> {
        let [Operation::Norm(_), Operation::Projection(q), Operation::Projection(k), Operation::Projection(v)] =
            layer.input_ops.as_slice()
        else {
            return Err("B8 full input inventory".into());
        };
        let [Operation::Projection(out), Operation::Add(_), Operation::Norm(_), Operation::Projection(gate), Operation::Projection(up), Operation::Silu(_), Operation::Projection(down), Operation::Add(_)] =
            layer.output_ops.as_slice()
        else {
            return Err("B8 full output inventory".into());
        };
        if layer.poisoned
            || layer.steps.len() != layer.capacity
            || !Rc::ptr_eq(&w.stream, &layer.stream)
            || epsilon.to_bits() != (layer.source_epsilon as f32).to_bits()
        {
            return Err("B8 full requires same-owner original F32 KV and epsilon".into());
        }
        let s = &w.stream;
        let p = &w.planes;
        let cache = [
            Rc::clone(&layer.buffers["key_cache"]),
            Rc::clone(&layer.buffers["value_cache"]),
        ];
        if cache
            .iter()
            .any(|b| b.len_bytes() != layer.capacity * 1024 * 4)
        {
            return Err("B8 full F32 cache extent".into());
        }
        let backup = [empty(s, WIDTH * 1024)?, empty(s, WIDTH * 1024)?];
        let input_norm = norms(
            s,
            &input,
            &layer.buffers["input_norm_weight"],
            &p["normalized"],
            epsilon,
        )?;
        let post_norm = norms(
            s,
            &p["residual"],
            &layer.buffers["post_norm_weight"],
            &p["post_norm"],
            epsilon,
        )?;
        let q = q.rebind_exact_batch(
            s,
            Rc::clone(&p["normalized"]),
            Rc::clone(&p["q_full"]),
            WIDTH,
        )?;
        let k = k.rebind_exact_batch(
            s,
            Rc::clone(&p["normalized"]),
            Rc::clone(&p["k_raw"]),
            WIDTH,
        )?;
        let v = v.rebind_exact_batch(
            s,
            Rc::clone(&p["normalized"]),
            Rc::clone(&p["value"]),
            WIDTH,
        )?;
        let out =
            out.rebind_exact_batch(s, Rc::clone(&p["gated"]), Rc::clone(&p["attention"]), WIDTH)?;
        let residual = CudaAddF32ArgsV1::bind(
            s,
            slice(&input, 0, WIDTH * H)?,
            slice(&p["attention"], 0, WIDTH * H)?,
            slice(&p["residual"], 0, WIDTH * H)?,
        )?;
        let ffn = Ffn::bind(Rc::clone(&w), [gate, up, down])?;
        let final_add = CudaAddF32ArgsV1::bind(
            s,
            slice(&p["residual"], 0, WIDTH * H)?,
            slice(&p["ffn"], 0, WIDTH * H)?,
            slice(&output, 0, WIDTH * H)?,
        )?;
        let vectors = [
            "input_norm_weight",
            "post_norm_weight",
            "q_norm_weight",
            "k_norm_weight",
            "rope_cos",
            "rope_sin",
        ]
        .into_iter()
        .map(|n| (n, Rc::clone(&layer.buffers[n])))
        .collect();
        Ok(Self {
            w,
            _input: input,
            _output: output,
            vectors,
            cache,
            backup,
            input_norm,
            post_norm,
            q,
            k,
            v,
            out,
            residual,
            ffn,
            final_add,
            capacity: layer.capacity,
            epsilon,
            pending: None,
            core_kv: None,
        })
    }
    fn same_owner(&self, l: &CheckpointFullCudaLayer) -> bool {
        Rc::ptr_eq(&self.w.stream, &l.stream)
            && l.capacity == self.capacity
            && Rc::ptr_eq(&self.cache[0], &l.buffers["key_cache"])
            && Rc::ptr_eq(&self.cache[1], &l.buffers["value_cache"])
    }
    pub(crate) fn validate(&self, l: &CheckpointFullCudaLayer, cursor: usize) -> Result<()> {
        if !self.same_owner(l)
            || l.poisoned
            || l.tokens != cursor
            || self.pending.is_some()
            || cursor.checked_add(WIDTH).is_none_or(|v| v > self.capacity)
        {
            return Err("stale/foreign B8 full transaction".into());
        }
        Ok(())
    }
    pub(crate) fn execute(&mut self, l: &mut CheckpointFullCudaLayer, cursor: usize) -> Result<()> {
        self.validate(l, cursor)?;
        let s = &self.w.stream;
        let p = &self.w.planes;
        // Preflight before any mutation, using actual retained RoPE/KV ranges.
        let attention = CudaQwen35PrefillFullAttentionV1::bind(
            s,
            CudaQwen35PrefillFullAttentionViewsV1 {
                q_full: view(&p["q_full"])?,
                k_raw: view(&p["k_raw"])?,
                value: view(&p["value"])?,
                q_norm_weight: slice(&self.vectors["q_norm_weight"], 0, 256)?,
                k_norm_weight: slice(&self.vectors["k_norm_weight"], 0, 256)?,
                rope_cos: slice(&self.vectors["rope_cos"], cursor * 32, WIDTH * 32)?,
                rope_sin: slice(&self.vectors["rope_sin"], cursor * 32, WIDTH * 32)?,
                key_cache: view(&self.cache[0])?,
                value_cache: view(&self.cache[1])?,
                query: view(&p["query"])?,
                gate: view(&p["q_gate"])?,
                attention: view(&p["full_attention"])?,
                gated: view(&p["gated"])?,
            },
            WIDTH,
            cursor,
            self.capacity,
            self.epsilon,
        )?
        .for_nerkyor_canonical_f32_kv()?;
        let core = if self.core_kv.is_some() { Some(self.bind_core_attention(cursor)?) } else { None };
        l.poisoned = true;
        self.pending = Some(cursor);
        for (to, from) in self.backup.iter().zip(&self.cache) {
            copy(s, to, 0, from.view(cursor * 1024 * 4, WIDTH * 1024 * 4)?)?;
        }
        for op in &self.input_norm {
            op.submit_for_plan(s)?;
        }
        self.q.submit()?;
        self.k.submit()?;
        self.v.submit()?;
        l.kv_dirty.mark(cursor, WIDTH, 1)?;
        if let Some((heads, rows)) = core {
            heads.submit_for_plan(s)?;
            for row in rows { row.submit_for_plan(s)?; }
        } else {
            attention.submit(s)?;
        }
        self.out.submit()?;
        self.residual.submit_for_plan(s)?;
        for op in &self.post_norm {
            op.submit_for_plan(s)?;
        }
        self.ffn.submit()?;
        self.final_add.submit_for_plan(s)?;
        l.tokens = cursor + WIDTH;
        l.poisoned = false;
        Ok(())
    }
    pub(crate) fn validate_commit(
        &self,
        l: &CheckpointFullCudaLayer,
        cursor: usize,
        n: usize,
    ) -> Result<()> {
        if n > WIDTH
            || !self.same_owner(l)
            || l.poisoned
            || self.pending != Some(cursor)
            || l.tokens != cursor + WIDTH
        {
            return Err("invalid B8 full prefix commit".into());
        }
        Ok(())
    }
    pub(crate) fn commit(
        &mut self,
        l: &mut CheckpointFullCudaLayer,
        cursor: usize,
        n: usize,
    ) -> Result<()> {
        self.validate_commit(l, cursor, n)?;
        l.poisoned = true;
        if n < WIDTH {
            for (to, from) in self.cache.iter().zip(&self.backup) {
                copy(
                    &self.w.stream,
                    to,
                    (cursor + n) * 1024 * 4,
                    from.view(n * 1024 * 4, (WIDTH - n) * 1024 * 4)?,
                )?;
            }
        }
        l.tokens = cursor + n;
        l.poisoned = false;
        self.pending = None;
        Ok(())
    }
}
impl Drop for Program {
    fn drop(&mut self) {
        let _ = self.w.stream.synchronize();
    }
}

#[cfg(test)]
pub(crate) fn diagnostic_state(
    layer: &CheckpointFullCudaLayer,
) -> Result<Vec<(&'static str, Vec<u8>)>> {
    if layer.poisoned {
        return Err("poisoned KV diagnostic".into());
    }
    // The hidden path is captured alongside the caches so a KV discrepancy can
    // be attributed rather than only observed. `normalized` is this layer's own
    // RMSNorm output -- the operand all three projections actually read -- and
    // is private per layer, so it snapshots the layer input at the current
    // token. `k_raw`/`value` are the QPN results before norm and RoPE and
    // `key` is the post-RoPE row that is stored. The layer's `input`/`output`
    // buffers are deliberately NOT captured: they alias the target's global
    // two-buffer hidden ping-pong (`hidden[index % 2]`), so at snapshot time
    // every layer would report the same final-token contents.
    ["normalized", "k_raw", "value", "key", "key_cache", "value_cache"]
    .into_iter()
    .map(|n| {
        let b = &layer.buffers[n];
        let mut raw = vec![0; b.len_bytes()];
        b.copy_to_host_sync(&layer.stream, &mut raw)?;
        Ok((n, raw))
    })
    .collect()
}


// Independently sealed core-B8 family: use the same compiled core-TU QK norm,
// RoPE and B1 attention as scalar target execution. NVFP4 projections/FFN and
// GDN remain batched. Enabled only by the closed ResearchPolicy binding.
impl Program {
    pub(crate) fn enable_core_attention(&mut self) -> Result<()> {
        if self.pending.is_some() || self.core_kv.is_some() {
            return Err("core attention candidate requires fresh B8 resources".into());
        }
        self.core_kv = Some([empty(&self.w.stream, WIDTH * 1024)?, empty(&self.w.stream, WIDTH * 1024)?]);
        Ok(())
    }
    fn bind_core_attention(&self, cursor: usize) -> Result<(
        augur_backend::CudaQwen35BatchedHeadNormRopeArgsV1,
        Vec<CudaQwen35FullAttentionArgsV1>,
    )> {
        let s=&self.w.stream;let p=&self.w.planes;
        let [key,value]=self.core_kv.as_ref().ok_or("missing core candidate KV rows")?;
        let heads=augur_backend::CudaQwen35BatchedHeadNormRopeArgsV1::bind(s,
            augur_backend::CudaQwen35BatchedHeadNormRopeViewsV1 {
                q_gate:view(&p["q_full"])?,q_weight:view(&self.vectors["q_norm_weight"])?,
                query:view(&p["query"])?,key_input:view(&p["k_raw"])?,
                k_weight:view(&self.vectors["k_norm_weight"])?,key:view(key)?,
                rope_cos:slice(&self.vectors["rope_cos"],cursor*32,WIDTH*32)?,
                rope_sin:slice(&self.vectors["rope_sin"],cursor*32,WIDTH*32)?,
                gate:view(&p["q_gate"])?,value_input:view(&p["value"])?,value:view(value)?,
            },WIDTH,self.epsilon)?;
        let mut rows=Vec::with_capacity(WIDTH);
        for row in 0..WIDTH {
            rows.push(CudaQwen35FullAttentionArgsV1::bind(s,CudaQwen35FullAttentionViewsV1 {
                query:slice(&p["query"],row*6144,6144)?,gate:slice(&p["q_gate"],row*6144,6144)?,
                key:slice(key,row*1024,1024)?,value:slice(value,row*1024,1024)?,
                key_cache:view(&self.cache[0])?,value_cache:view(&self.cache[1])?,
                attention:slice(&p["full_attention"],row*6144,6144)?,
                gated:slice(&p["gated"],row*6144,6144)?,
            },self.capacity,cursor+row)?);
        }
        Ok((heads,rows))
    }
}
