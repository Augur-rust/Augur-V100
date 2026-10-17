//! B1 continuation of the rotated-Q8 prefill cache. No legacy KV reinterpretation.
use super::*;
use augur_backend::cuda_qwen35_rotated_kv::{CudaQwen35RotatedKvPlanV1, CudaQwen35RotatedKvSmallPlanV1,
    CudaQwen35RotatedKvGqa8PlanV1, Qwen35RotatedKvViews};

use super::augur_prefill::AttentionMode;

pub(crate) struct Decode {
    stream: Rc<OwnedCudaStream>,
    plan: Plan,
    capacity: usize,
    scratch: OwnedCudaBuffer,
}
enum Plan {
    Flash32(CudaQwen35RotatedKvPlanV1),
    Tile8B1(CudaQwen35RotatedKvSmallPlanV1),
    Gqa8Planar(CudaQwen35RotatedKvGqa8PlanV1),
    Gqa8C4Planar(CudaQwen35RotatedKvGqa8PlanV1),
}
impl Decode {
    pub(crate) fn new(stream: Rc<OwnedCudaStream>, capacity: usize) -> Result<Self> {
        // Fixed for the complete owner lifetime; never switch arithmetic mid-session.
        let mode = match std::env::var("AUGUR_HTTP_DECODE_ATTENTION") {
            Ok(value) => value,
            Err(std::env::VarError::NotPresent) => "gqa8_planar".into(),
            Err(e) => return Err(e.into()),
        };
        let (plan, bytes) = match mode.as_str() {
            "flash32" => (Plan::Flash32(CudaQwen35RotatedKvPlanV1::new(&stream)?),
                CudaQwen35RotatedKvPlanV1::workspace_bytes(1)?),
            "tile8_b1" => (Plan::Tile8B1(CudaQwen35RotatedKvSmallPlanV1::new(&stream)?),
                CudaQwen35RotatedKvSmallPlanV1::workspace_bytes(capacity, 1)?),
            "gqa8_planar" => (Plan::Gqa8Planar(CudaQwen35RotatedKvGqa8PlanV1::new_planar(&stream)?),
                CudaQwen35RotatedKvGqa8PlanV1::workspace_bytes_planar(capacity, 1)?),
            "gqa8_c4_planar" => (Plan::Gqa8C4Planar(CudaQwen35RotatedKvGqa8PlanV1::new_planar_c4(&stream)?),
                CudaQwen35RotatedKvGqa8PlanV1::workspace_bytes_planar(capacity, 1)?),
            _ => return Err("unknown AUGUR_HTTP_DECODE_ATTENTION (flash32|tile8_b1|gqa8_planar|gqa8_c4_planar)".into()),
        };
        let scratch = OwnedCudaBuffer::allocate(stream.device_ordinal(), bytes)?;
        Ok(Self {
            stream,
            plan,
            capacity,
            scratch,
        })
    }
    pub(crate) fn label(&self) -> &'static str {
        match self.plan {
            Plan::Flash32(_) => "rotated_q8_flash32_b1_v1",
            Plan::Tile8B1(_) => "rotated_q8_small_tile8_b1_full262144_v1",
            Plan::Gqa8Planar(_) => "rotated_q8_planar_gqa8_shared_q_early_kv_full262144_v2",
            Plan::Gqa8C4Planar(_) => "rotated_q8_planar_gqa8_c4_shared_q_early_kv_full262144_v1",
        }
    }
    pub(crate) fn mode(&self) -> AttentionMode {
        match self.plan { Plan::Gqa8Planar(_) => AttentionMode::Gqa8Planar,
            Plan::Gqa8C4Planar(_) => AttentionMode::Gqa8C4Planar, _ => AttentionMode::Legacy }
    }
    fn submit(&self, views: Qwen35RotatedKvViews, position: usize) -> Result<()> {
        match &self.plan {
            Plan::Flash32(p) => p.submit(&self.stream, views, self.capacity, position, 1)?,
            Plan::Tile8B1(p) => p.submit_tile8_rows(&self.stream, views, self.capacity, position, 1)?,
            Plan::Gqa8Planar(p) | Plan::Gqa8C4Planar(p) => p.submit(&self.stream, views, self.capacity, position, 1)?,
        }
        Ok(())
    }
}
impl CheckpointFullCudaLayer {
    pub(crate) fn submit_augur_decode(&mut self, decode: &Decode) -> Result<()> {
        if self.poisoned
            || self.tokens >= self.capacity
            || self.capacity != decode.capacity
            || self.kv8.is_none()
            || !Rc::ptr_eq(&self.stream, &decode.stream)
        {
            return Err("invalid rotated decode owner/frontier/stream".into());
        }
        // Match the actual pack padding; Flash32 also permits capacities that
        // are multiples of 32 but not 256. Record before any possible write.
        let alignment = match &decode.plan {
            Plan::Flash32(_) => 32,
            Plan::Tile8B1(_) | Plan::Gqa8Planar(_) | Plan::Gqa8C4Planar(_) => 256,
        };
        self.kv_dirty.mark(self.tokens, 1, alignment)?;
        self.poisoned = true;
        self.submit_b1_input()?;
        let mut heads = self.heads_template;
        heads.rope_cos = subview(&self.buffers["rope_cos"], self.tokens * 32, 32)?;
        heads.rope_sin = subview(&self.buffers["rope_sin"], self.tokens * 32, 32)?;
        CudaQwen35FusedHeadNormRopeArgsV1::bind(&self.stream, heads, self.epsilon)?
            .submit_for_plan(&self.stream)?;
        let v = |name: &str| self.buffers[name].view(0, self.buffers[name].len_bytes());
        decode.submit(
            Qwen35RotatedKvViews {
                query: v("query")?,
                gate: self.buffers["q_full"].view(6144 * 4, 6144 * 4)?,
                key: v("key")?,
                value: v("value")?,
                key_cache: v("key_cache")?,
                value_cache: v("value_cache")?,
                attention: v("attention")?,
                gated: v("gated")?,
                workspace: decode.scratch.view(0, decode.scratch.len_bytes())?,
            },
            self.tokens,
        )?;
        self.submit_b1_tail()?;
        self.tokens += 1;
        self.poisoned = false;
        Ok(())
    }
}
impl Drop for Decode {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
    }
}
