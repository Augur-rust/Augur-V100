//! Augur full attention using the rotated-Q8 Flash Attention path.
//! The retained cache allocation is reused, never duplicated. The enclosing
//! target forbids ordinary (unrotated-cache) decode until reset_empty.
use super::*;
use crate::checkpoint_linear_cuda::augur_prefill::{arena::Plane, Ffn, Workspace as Common};
use augur_backend::cuda_qwen35_prefill_cublas::{
    CudaQwen35PrefillRmsnormV1, CudaQwen35AugurProjectionV1 as Projection,
};
use augur_backend::cuda_qwen35_rotated_kv::{
    CudaQwen35RotatedKvInputPlanV1 as Input, CudaQwen35RotatedKvPlanV1 as Prefix,
    CudaQwen35RotatedKvGqa8PlanV1 as Gqa8, Qwen35RotatedKvViews,
};
use augur_backend::cuda_qwen35_rotated_kv::{QWEN35_ROTATED_KV_IDENTITY, QWEN35_ROTATED_KV_PLANAR_IDENTITY};
/// One cache format and attention arithmetic for the complete owner lifetime.
/// The format2 path is opt-in until independent QUASAR model qualification.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AttentionMode { Legacy, Gqa8Planar, Gqa8C4Planar }
impl AttentionMode {
    pub(crate) fn kv_identity(self) -> &'static str {
        match self { Self::Legacy => QWEN35_ROTATED_KV_IDENTITY,
            Self::Gqa8Planar | Self::Gqa8C4Planar => QWEN35_ROTATED_KV_PLANAR_IDENTITY }
    }
    pub(crate) fn label(self) -> &'static str {
        match self { Self::Legacy => "tile8_b1", Self::Gqa8Planar => "gqa8_planar", Self::Gqa8C4Planar => "gqa8_c4_planar" }
    }
}


use augur_backend::{CudaQwen35BatchedHeadNormRopeArgsV1, CudaQwen35BatchedHeadNormRopeViewsV1};
const H: usize = 5120;
pub(crate) mod dflash;
mod cache_lease;
use cache_lease::CacheLease;
enum ShortAttention {
    Legacy(Prefix),
    Gqa8Planar(Gqa8),
}
pub(crate) struct Workspace {
    pub(crate) common: Rc<Common>,
    scratch: OwnedCudaBuffer,
    prefix: ShortAttention,
    input: Input,
    capacity: usize,
    pub(crate) mode: AttentionMode,
}
impl Workspace {
    fn scratch_bytes(tokens: usize, capacity: usize, mode: AttentionMode) -> Result<usize> {
        let tokens = tokens.min(512);
        Ok(if tokens >= 9 {
            Input::workspace_bytes(capacity, tokens)?
        } else if matches!(mode, AttentionMode::Gqa8Planar | AttentionMode::Gqa8C4Planar) {
            Gqa8::workspace_bytes_planar(capacity, tokens)?
        } else {
            Prefix::workspace_bytes(tokens)?
        })
    }
    fn reserved_scratch_bytes(tokens: usize, capacity: usize, mode: AttentionMode) -> Result<usize> {
        let mut bytes = Self::scratch_bytes(tokens, capacity, mode)?;
        // A shared phase can bind smaller chunks. Split3 metadata makes the
        // exact extent non-monotonic at 128/256 rows; reserve every live tile.
        for rows in 1..tokens.min(512) {
            bytes = bytes.max(Self::scratch_bytes(rows, capacity, mode)?);
        }
        Ok(bytes)
    }
    pub(crate) fn owned_recipe(tokens: usize, capacity: usize) -> Result<usize> {
        Self::owned_recipe_mode(tokens, capacity, AttentionMode::Legacy)
    }
    pub(crate) fn owned_recipe_mode(tokens: usize, capacity: usize, mode: AttentionMode) -> Result<usize> {
        Ok(Common::owned_recipe(tokens)? + Self::reserved_scratch_bytes(tokens, capacity, mode)?)
    }
    pub(crate) fn begin(stream: Rc<OwnedCudaStream>, tokens: usize, capacity: usize, mode: AttentionMode) -> Result<Rc<Self>> {
        Self::begin_kind(stream, tokens, capacity, false, mode)
    }
    pub(crate) fn begin_packed_b8(stream: Rc<OwnedCudaStream>, capacity: usize, mode: AttentionMode) -> Result<Rc<Self>> {
        Self::begin_kind(stream, 8, capacity, true, mode)
    }
    fn begin_kind(stream: Rc<OwnedCudaStream>, tokens: usize, capacity: usize,
                  packed_b8: bool, mode: AttentionMode) -> Result<Rc<Self>> {
        if !crate::augur_session::CONTEXT_CHOICES.contains(&capacity) {
            return Err("Augur requires a supported physical KV capacity".into());
        }
        let prefix = match mode {
            AttentionMode::Legacy => ShortAttention::Legacy(Prefix::new(&stream)?),
            AttentionMode::Gqa8Planar => ShortAttention::Gqa8Planar(Gqa8::new_planar(&stream)?),
            AttentionMode::Gqa8C4Planar => ShortAttention::Gqa8Planar(Gqa8::new_planar_c4(&stream)?),
        };
        let input = match mode {
            AttentionMode::Legacy => Input::new(&stream)?,
            AttentionMode::Gqa8Planar | AttentionMode::Gqa8C4Planar => Input::new_planar(&stream)?,
        };
        let common = if packed_b8 { Common::begin_packed_b8(Rc::clone(&stream))? }
                     else { Common::begin(Rc::clone(&stream), tokens)? };
        let scratch = OwnedCudaBuffer::allocate(stream.device_ordinal(),
            Self::reserved_scratch_bytes(tokens, capacity, mode)?)?;
        Ok(Rc::new(Self { common, scratch, prefix, input, capacity, mode }))
    }
    pub(crate) fn finish(self: Rc<Self>) -> Result<()> {
        let owner = Rc::try_unwrap(self).map_err(|_| "retained Augur full workspace consumer")?;
        owner.common.stream.synchronize()?;
        if Rc::strong_count(&owner.common) != 1 {
            return Err("retained Augur common workspace consumer".into());
        }
        drop(owner);
        Ok(())
    }
}
fn prefix(b: &OwnedCudaBuffer, offset: usize, n: usize) -> Result<CudaF32View> {
    Ok(CudaF32View::new(
        b.view(offset, n * 4)?,
        PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
        n,
    )?)
}
struct AttentionTile {
    heads: CudaQwen35BatchedHeadNormRopeArgsV1,
    attention: Qwen35RotatedKvViews,
    rows: usize,
    position: usize,
}
pub(crate) struct Program {
    workspace: Rc<Workspace>,
    _input: Buffer,
    _output: Buffer,
    _vectors: Vec<Buffer>,
    cache: CacheLease,
    input_norm: CudaQwen35PrefillRmsnormV1,
    post_norm: CudaQwen35PrefillRmsnormV1,
    q: Projection,
    k: Projection,
    v: Projection,
    out: Projection,
    tiles: Vec<AttentionTile>,
    residual: CudaAddF32ArgsV1,
    ffn: Ffn,
    final_add: CudaAddF32ArgsV1,
    tokens: usize,
    position: usize,
}
impl Program {
    pub(crate) fn use_direct_b8(&mut self)->Result<()> {
        for p in [&mut self.q,&mut self.k,&mut self.v,&mut self.out] {p.use_direct_b8()?;}
        self.ffn.use_direct_b8()
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn bind(
        layer: &CheckpointFullCudaLayer,
        workspace: Rc<Workspace>,
        input: Buffer,
        input_offset: usize,
        output: Buffer,
        output_offset: usize,
        tokens: usize,
        position: usize,
    ) -> Result<Self> {
        let w = &workspace;
        let c = &w.common;
        let s = &c.stream;
        if layer.poisoned
            || !Rc::ptr_eq(s, &layer.stream)
            || Rc::ptr_eq(&input, &output)
            || !(1..=c.max_tokens).contains(&tokens)
            || layer.kv8.is_none()
            || layer.capacity != w.capacity
            || position
                .checked_add(tokens)
                .is_none_or(|n| n > layer.capacity)
        {
            return Err("Augur full owner/width/frontier mismatch".into());
        }
        let [Operation::Norm(_), Operation::Projection(q), Operation::Projection(k), Operation::Projection(v)] =
            layer.input_ops.as_slice()
        else {
            return Err("Augur full input inventory".into());
        };
        let [Operation::Projection(out), Operation::Add(_), Operation::Norm(_), Operation::QpnFfn(ffn), Operation::Add(_)] =
            layer.output_ops.as_slice()
        else {
            return Err("Augur full FFN inventory".into());
        };
        let b = &c.planes;
        let cv = |name: &str| b[name].view(0, tokens * H);
        let input_norm = CudaQwen35PrefillRmsnormV1::bind(
            s,
            prefix(&input, input_offset, tokens * H)?,
            view(&layer.buffers["input_norm_weight"])?,
            cv("normalized")?,
            tokens,
            H,
            layer.epsilon,
        )?;
        let post_norm = CudaQwen35PrefillRmsnormV1::bind(
            s,
            cv("attention_residual")?,
            view(&layer.buffers["post_norm_weight"])?,
            cv("post_norm")?,
            tokens,
            H,
            layer.epsilon,
        )?;
        let projection = |source: &CheckpointScalarCudaProjection,
                          from: &Plane,
                          to: &Plane|
         -> Result<Projection> {
            Ok(Projection::bind(
                Rc::clone(&c.phase),
                Rc::clone(s),
                source.retained_qpn()?.original_weight(),
                Rc::clone(&from.owner),
                from.offset,
                Rc::clone(&to.owner),
                to.offset,
                tokens,
            )?)
        };
        let q = projection(q, &b["normalized"], &c.full_planes["q_full"])?;
        let k = projection(k, &b["normalized"], &c.full_planes["k_raw"])?;
        let v = projection(v, &b["normalized"], &c.full_planes["value"])?;
        // Loader separates Q/gate rows and repeats the source256-head norm weights.
        // This module is compiled only with Augur's split-half RoPE feature.
        let cache = [
            Rc::clone(&layer.buffers["key_cache"]),
            Rc::clone(&layer.buffers["value_cache"]),
        ];
        let mut tiles = Vec::new();
        for offset in (0..tokens).step_by(512) {
            let rows = (tokens - offset).min(512);
            let av = |name: &str, cols| c.full_planes[name].view(offset * cols * 4, rows * cols);
            let heads = CudaQwen35BatchedHeadNormRopeArgsV1::bind(
                s,
                CudaQwen35BatchedHeadNormRopeViewsV1 {
                    q_gate: av("q_full", 12288)?,
                    q_weight: view(&layer.buffers["q_norm_weight"])?,
                    query: av("query", 6144)?,
                    key_input: av("k_raw", 1024)?,
                    k_weight: view(&layer.buffers["k_norm_weight"])?,
                    key: av("key_work", 1024)?,
                    rope_cos: subview(
                        &layer.buffers["rope_cos"],
                        (position + offset) * 32,
                        rows * 32,
                    )?,
                    rope_sin: subview(
                        &layer.buffers["rope_sin"],
                        (position + offset) * 32,
                        rows * 32,
                    )?,
                    gate: av("q_gate", 6144)?,
                    value_input: av("value", 1024)?,
                    value: av("value_work", 1024)?,
                },
                rows,
                layer.epsilon,
            )?;
            let attention = Qwen35RotatedKvViews {
                query: av("query", 6144)?.buffer_view(),
                gate: av("q_gate", 6144)?.buffer_view(),
                key: av("key_work", 1024)?.buffer_view(),
                value: av("value_work", 1024)?.buffer_view(),
                key_cache: cache[0].view(0, layer.capacity * 1088)?,
                value_cache: cache[1].view(0, layer.capacity * 1088)?,
                attention: av("attention", 6144)?.buffer_view(),
                gated: av("gated", 6144)?.buffer_view(),
                workspace: w
                    .scratch
                    .view(0, Workspace::scratch_bytes(rows, layer.capacity, w.mode)?)?,
            };
            tiles.push(AttentionTile {
                heads,
                attention,
                rows,
                position: position + offset,
            });
        }
        let out = projection(out, &c.full_planes["gated"], &b["attention"])?;
        let residual = CudaAddF32ArgsV1::bind(
            s,
            prefix(&input, input_offset, tokens * H)?,
            cv("attention")?,
            cv("attention_residual")?,
        )?;
        let ffn = Ffn::bind(
            Rc::clone(c),
            ffn.retained().original_weights(),
            Rc::clone(&b["post_norm"].owner),
            b["post_norm"].offset,
            Rc::clone(&b["ffn_output"].owner),
            b["ffn_output"].offset,
            tokens,
        )?;
        let final_add = CudaAddF32ArgsV1::bind(
            s,
            cv("attention_residual")?,
            cv("ffn_output")?,
            prefix(&output, output_offset, tokens * H)?,
        )?;
        let vectors = [
            "input_norm_weight",
            "post_norm_weight",
            "q_norm_weight",
            "k_norm_weight",
            "rope_cos",
            "rope_sin",
        ]
        .map(|n| Rc::clone(&layer.buffers[n]))
        .to_vec();
        Ok(Self {
            workspace,
            _input: input,
            _output: output,
            _vectors: vectors,
            cache: CacheLease::new(cache)?,
            input_norm,
            post_norm,
            q,
            k,
            v,
            out,
            tiles,
            residual,
            ffn,
            final_add,
            tokens,
            position,
        })
    }
    pub(crate) fn submit(&self, layer: &mut CheckpointFullCudaLayer) -> Result<()> {
        let w = &self.workspace;
        let s = &w.common.stream;
        if layer.poisoned
            || !Rc::ptr_eq(s, &layer.stream)
            || layer.tokens != self.position
            || !self.cache.matches(layer)
        {
            return Err("foreign/stale Augur full program".into());
        }
        // Conservatively include physical padding, including short/B8 tiles.
        // A later B8 commit may rewind tokens, but never this write high-water.
        layer.kv_dirty.mark(self.position, self.tokens, 256)?;
        layer.poisoned = true;
        self.input_norm.submit(s)?;
        Projection::submit_same_input([&self.q, &self.k, &self.v])?;
        for tile in &self.tiles {
            tile.heads.submit_for_plan(s)?;
            if tile.rows >= 9 {
                w.input
                    .submit(s, tile.attention, w.capacity, tile.position, tile.rows)?;
            } else {
                match &w.prefix {
                    ShortAttention::Legacy(p) => p.submit(s, tile.attention, w.capacity, tile.position, tile.rows)?,
                    ShortAttention::Gqa8Planar(p) => p.submit(s, tile.attention, w.capacity, tile.position, tile.rows)?,
                }
            }
        }
        self.out.submit()?;
        self.residual.submit_for_plan(s)?;
        self.post_norm.submit(s)?;
        self.ffn.submit()?;
        self.final_add.submit_for_plan(s)?;
        layer.tokens += self.tokens;
        layer.poisoned = false;
        Ok(())
    }
}
impl Drop for Program {
    fn drop(&mut self) {
        let _ = self.workspace.common.stream.synchronize();
    }
}

/// Only committed bytes; diagnostic readback stays outside performance timing.
#[cfg(test)]
pub(crate) fn diagnostic_committed_bytes(
    layer: &CheckpointFullCudaLayer,
) -> Result<Vec<(&'static str, Vec<u8>)>> {
    if layer.poisoned || layer.tokens == 0 {
        return Err("uncommitted layer diagnostic".into());
    }
    let mut result = Vec::new();
    for name in ["key_cache", "value_cache"] {
        let buffer = &layer.buffers[name];
        let mut bytes = vec![0; layer.tokens * 1088];
        buffer
            .view(0, bytes.len())?
            .copy_to_host_sync(&layer.stream, &mut bytes)?;
        result.push((name, bytes));
    }
    Ok(result)
}

#[cfg(test)]
mod workspace_capacity_tests {
    use super::*;

    #[test]
    fn shared_scratch_reserves_smaller_partitioned_tiles() {
        for mode in [AttentionMode::Legacy, AttentionMode::Gqa8Planar, AttentionMode::Gqa8C4Planar] {
            // 129/257-row phases may include smaller 128/256-row chunks.
            assert_eq!(Workspace::reserved_scratch_bytes(129, 262144, mode).unwrap(), 13_778_944);
            assert_eq!(Workspace::reserved_scratch_bytes(257, 262144, mode).unwrap(), 27_557_888);
            assert_eq!(Workspace::reserved_scratch_bytes(4352, 262144, mode).unwrap(), 55_115_776);
            // Consumer views keep their exact ABI extent, even with a larger reservation.
            assert!(Workspace::scratch_bytes(129, 262144, mode).unwrap() < 13_778_944);
            assert!(Workspace::scratch_bytes(257, 262144, mode).unwrap() < 27_557_888);
        }
    }
}
