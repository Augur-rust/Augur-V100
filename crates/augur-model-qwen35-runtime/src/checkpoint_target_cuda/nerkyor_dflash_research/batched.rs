//! Independent B8 research target. Original NVFP4 / F32 activation and KV.
//! No QUASAR QPN, FP16 staging, rotated KV, selector, or product admission.
use super::*;
use crate::checkpoint_projection_cuda::{
    CheckpointExactBatchProjection, CheckpointScalarCudaProjection,
};
use augur_backend::{CudaAddF32ArgsV1, CudaF32View, CudaRmsnormF32ArgsV1, CudaSiluMulF32ArgsV1};
use std::collections::BTreeMap;
pub(crate) type Buffer = Rc<OwnedCudaBuffer>;
pub(crate) const WIDTH: usize = 8;
pub(crate) const H: usize = 5120;
pub(crate) const INTERMEDIATE: usize = 17408;
pub(crate) fn slice(b: &OwnedCudaBuffer, at: usize, n: usize) -> Result<CudaF32View> {
    Ok(CudaF32View::new(
        b.view(at * 4, n * 4)?,
        PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
        n,
    )?)
}
pub(crate) fn copy(
    s: &OwnedCudaStream,
    to: &OwnedCudaBuffer,
    offset: usize,
    from: augur_backend::CudaBufferView,
) -> Result<()> {
    // SAFETY: every descriptor is checked against a retained owner; callers
    // exclusively borrow the batch and target and use their single stream.
    unsafe {
        to.copy_from_device_async_unchecked(s, offset, from)?;
    }
    Ok(())
}
pub(crate) struct Workspace {
    pub(crate) stream: Rc<OwnedCudaStream>,
    pub(crate) planes: BTreeMap<&'static str, Buffer>,
}
impl Workspace {
    fn new(stream: Rc<OwnedCudaStream>) -> Result<Rc<Self>> {
        let mut planes = BTreeMap::new();
        for (name, cols) in [
            ("normalized", H),
            ("attention", H),
            ("residual", H),
            ("post_norm", H),
            ("gate", INTERMEDIATE),
            ("up", INTERMEDIATE),
            ("swiglu", INTERMEDIATE),
            ("ffn", H),
            ("activated", 10240),
            ("recurrent", 6144),
            ("linear_output", 6144),
            ("q_full", 12288),
            ("k_raw", 1024),
            ("value", 1024),
            ("query", 6144),
            ("q_gate", 6144),
            ("full_attention", 6144),
            ("gated", 6144),
        ] {
            planes.insert(name, empty(&stream, WIDTH * cols)?);
        }
        Ok(Rc::new(Self { stream, planes }))
    }
}
pub(crate) fn norms(
    s: &OwnedCudaStream,
    input: &OwnedCudaBuffer,
    weight: &OwnedCudaBuffer,
    output: &OwnedCudaBuffer,
    epsilon: f32,
) -> Result<Vec<CudaRmsnormF32ArgsV1>> {
    (0..WIDTH)
        .map(|r| {
            Ok(CudaRmsnormF32ArgsV1::bind(
                s,
                slice(input, r * H, H)?,
                slice(weight, 0, H)?,
                slice(output, r * H, H)?,
                epsilon,
            )?)
        })
        .collect()
}
pub(crate) struct Ffn {
    gate: CheckpointExactBatchProjection,
    up: CheckpointExactBatchProjection,
    down: CheckpointExactBatchProjection,
    silu: CudaSiluMulF32ArgsV1,
    _w: Rc<Workspace>,
}
impl Ffn {
    pub(crate) fn bind(
        w: Rc<Workspace>,
        source: [&CheckpointScalarCudaProjection; 3],
    ) -> Result<Self> {
        let p = &w.planes;
        let s = &w.stream;
        let gate = source[0].rebind_exact_batch(
            s,
            Rc::clone(&p["post_norm"]),
            Rc::clone(&p["gate"]),
            WIDTH,
        )?;
        let up = source[1].rebind_exact_batch(
            s,
            Rc::clone(&p["post_norm"]),
            Rc::clone(&p["up"]),
            WIDTH,
        )?;
        let down = source[2].rebind_exact_batch(
            s,
            Rc::clone(&p["swiglu"]),
            Rc::clone(&p["ffn"]),
            WIDTH,
        )?;
        let silu = CudaSiluMulF32ArgsV1::bind(
            s,
            slice(&p["gate"], 0, WIDTH * INTERMEDIATE)?,
            slice(&p["up"], 0, WIDTH * INTERMEDIATE)?,
            slice(&p["swiglu"], 0, WIDTH * INTERMEDIATE)?,
        )?;
        Ok(Self {
            gate,
            up,
            down,
            silu,
            _w: w,
        })
    }
    pub(crate) fn submit(&self) -> Result<()> {
        self.gate.submit()?;
        self.up.submit()?;
        self.silu.submit_for_plan(&self._w.stream)?;
        self.down.submit()?;
        Ok(())
    }
}
use crate::checkpoint_full_cuda::nerkyor_batch as full;
use crate::checkpoint_linear_cuda::nerkyor_batch as linear;
enum Program {
    Linear(linear::Program),
    Full(full::Program),
}
/// Not public: numerical qualification precedes proposal/decoder admission.
/// A pending execution owns all tentative state; only commit publishes a prefix.
pub(crate) struct Batch {
    stream: Rc<OwnedCudaStream>,
    binding: ResearchBinding,
    _w: Rc<Workspace>,
    hidden: [Buffer; 2],
    features: Vec<Buffer>,
    normalized: Buffer,
    logits: Buffer,
    norm: Vec<CudaRmsnormF32ArgsV1>,
    head: crate::checkpoint_projection_cuda::CheckpointSharedBf16HeadRows,
    token_ids: OwnedCudaBuffer,
    top_ids: OwnedCudaBuffer,
    top1: CudaQwen35VocabTop1ArgsV1,
    gather: augur_backend::CudaBf16TokenRowsGatherArgsV1,
    programs: Vec<Program>,
    pending: Option<(usize, (u64, u64, usize), Option<u32>)>,
    poisoned: bool,
}
impl Batch {
    pub(crate) fn new(t: &mut CheckpointCudaTarget) -> Result<Self> {
        t.validate_cursor()?;
        let binding = ResearchBinding::read(t)?;
        if !matches!(
            binding.policy,
            ResearchPolicy::GateF32Cta | ResearchPolicy::GateF32CtaB8 | ResearchPolicy::CoreB8Prefill
        ) || !t.logits_current
        {
            return Err("B8 research requires a committed gate-F32 target".into());
        }
        if binding.policy.is_b8() {
            identity::require_b8_head()?;
        }
        t.dflash_capture
            .as_ref()
            .ok_or("missing capture")?
            .committed_identity()?;
        let s = Rc::clone(&t.stream);
        let w = Workspace::new(Rc::clone(&s))?;
        let hidden = [empty(&s, WIDTH * H)?, empty(&s, WIDTH * H)?];
        let mut programs = Vec::with_capacity(t.layers.len());
        let epsilon = t._artifact.config().rms_norm_eps as f32;
        for (i, l) in t.layers.iter().enumerate() {
            let x = Rc::clone(&hidden[i % 2]);
            let y = Rc::clone(&hidden[(i + 1) % 2]);
            programs.push(match l {
                Layer::Linear(l) => {
                    Program::Linear(linear::Program::bind(l, Rc::clone(&w), x, y, epsilon)?)
                }
                Layer::Full(l) => {
                    Program::Full(full::Program::bind(l, Rc::clone(&w), x, y, epsilon)?)
                }
            });
        }
        if binding.policy == ResearchPolicy::CoreB8Prefill {
            for program in &mut programs {
                if let Program::Full(p) = program { p.enable_core_attention()?; }
            }
        }
        let normalized = empty(&s, WIDTH * H)?;
        let logits = empty(&s, WIDTH * 248320)?;
        let norm = t
            .head
            .bind_research_rows_norm(&s, &hidden[0], &normalized, epsilon)?;
        let head = t.head.bind_shared_normalized_bf16_rows(
            &s,
            Rc::clone(&normalized),
            Rc::clone(&logits),
            WIDTH,
        )?;
        if binding.policy.is_b8()
            && head.kernel_label() != identity::B8_HEAD
        {
            return Err("B8 head identity mismatch".into());
        }
        let top_ids = OwnedCudaBuffer::allocate(s.device_ordinal(), WIDTH * 4)?;
        let top1 = CudaQwen35VocabTop1ArgsV1::bind(
            &s,
            view(&logits)?,
            top_ids.view(0, WIDTH * 4)?,
            WIDTH,
        )?;
        let token_ids = OwnedCudaBuffer::allocate(s.device_ordinal(), WIDTH * 4)?;
        let gather = augur_backend::CudaBf16TokenRowsGatherArgsV1::bind(
            &s,
            CudaBf16WeightViewV1::new(
                t._embedding.view(0, t._embedding.len_bytes())?,
                PhysicalLayoutId::DEVICE_CONTIGUOUS_BF16_CUDA_V1,
                248320,
                H,
            )?,
            token_ids.view(0, WIDTH * 4)?,
            slice(&hidden[0], 0, WIDTH * H)?,
            WIDTH,
        )?;
        let features = crate::DFLASH_TARGET_CAPTURE_LAYER_IDS
            .iter()
            .map(|_| empty(&s, WIDTH * H))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            stream: s,
            binding,
            _w: w,
            hidden,
            features,
            normalized,
            logits,
            norm,
            head,
            token_ids,
            top_ids,
            top1,
            gather,
            programs,
            pending: None,
            poisoned: false,
        })
    }
    fn validate(&self, t: &mut CheckpointCudaTarget) -> Result<()> {
        t.validate_cursor()?;
        if self.poisoned
            || self.pending.is_some()
            || !Rc::ptr_eq(&self.stream, &t.stream)
            || self.binding != ResearchBinding::read(t)?
            || !t.logits_current
        {
            return Err("foreign, changed, poisoned or pending B8 target".into());
        }
        Ok(())
    }
    pub(crate) fn execute(&mut self, t: &mut CheckpointCudaTarget, path: &[u32]) -> Result<()> {
        self.validate(t)?;
        if path.len() != WIDTH
            || path.iter().any(|&v| v >= 248320)
            || t.selected
                .is_some_and(|id| path.first().copied() != Some(id))
            || t.consumed.checked_add(WIDTH).is_none_or(|v| v > t.capacity)
        {
            return Err("invalid B8 token or capacity".into());
        }
        let identity = t
            .dflash_capture
            .as_ref()
            .ok_or("missing capture")?
            .committed_identity()?;
        let cursor = t.consumed;
        for (p, l) in self.programs.iter().zip(&t.layers) {
            match (p, l) {
                (Program::Linear(p), Layer::Linear(l)) => p.validate(l, cursor)?,
                (Program::Full(p), Layer::Full(l)) => p.validate(l, cursor)?,
                _ => return Err("B8 layer inventory".into()),
            }
        }
        self.poisoned = true;
        t.poisoned = true;
        self.pending = Some((cursor, identity, t.selected));
        let result = (|| -> Result<()> {
            let bytes: Vec<u8> = path.iter().flat_map(|v| v.to_le_bytes()).collect();
            self.token_ids.copy_from_host_sync(&self.stream, &bytes)?;
            self.gather.submit_for_plan(&self.stream)?;
            for (i, (p, l)) in self.programs.iter_mut().zip(&mut t.layers).enumerate() {
                match (p, l) {
                    (Program::Linear(p), Layer::Linear(l)) => p.execute(l, cursor)?,
                    (Program::Full(p), Layer::Full(l)) => p.execute(l, cursor)?,
                    _ => return Err("B8 layer inventory".into()),
                }
                if let Some(slot) = crate::DFLASH_TARGET_CAPTURE_LAYER_IDS
                    .iter()
                    .position(|&id| id == i)
                {
                    copy(
                        &self.stream,
                        &self.features[slot],
                        0,
                        self.hidden[(i + 1) % 2].view(0, WIDTH * H * 4)?,
                    )?;
                }
            }
            for op in &self.norm {
                op.submit_for_plan(&self.stream)?;
            }
            self.head.submit()?;
            self.top1.submit(&self.stream)?;
            self.stream.synchronize()?;
            Ok(())
        })();
        // The target intentionally stays poisoned until successful prefix commit.
        if result.is_ok() {
            self.poisoned = false;
        }
        result
    }
    pub(crate) fn quiescent(&self) -> bool {
        !self.poisoned && self.pending.is_none()
    }
    pub(crate) fn selected_rows(&self) -> Result<[u32; WIDTH]> {
        if self.poisoned || self.pending.is_none() {
            return Err("B8 IDs require a completed pending execution".into());
        }
        let mut raw = [0u8; WIDTH * 4];
        self.top_ids.copy_to_host_sync(&self.stream, &mut raw)?;
        let ids =
            std::array::from_fn(|i| u32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap()));
        if ids.iter().any(|&id| id >= 248320) {
            return Err("nonfinite or invalid B8 target selection".into());
        }
        Ok(ids)
    }
    pub(crate) fn commit(&mut self, t: &mut CheckpointCudaTarget, n: usize) -> Result<()> {
        let (cursor, identity, selected) = self.pending.ok_or("missing B8 transaction")?;
        if self.poisoned
            || n > WIDTH
            || !Rc::ptr_eq(&self.stream, &t.stream)
            || !t.poisoned
            || t.consumed != cursor
            || t.selected != selected
            || self.binding != ResearchBinding::read(t)?
            || t.dflash_capture
                .as_ref()
                .ok_or("missing capture")?
                .committed_identity()?
                != identity
        {
            return Err("invalid/stale B8 prefix commit".into());
        }
        for (p, l) in self.programs.iter().zip(&t.layers) {
            match (p, l) {
                (Program::Linear(p), Layer::Linear(l)) => p.validate_commit(l, cursor, n)?,
                (Program::Full(p), Layer::Full(l)) => p.validate_commit(l, cursor, n)?,
                _ => return Err("B8 commit inventory".into()),
            }
        }
        self.poisoned = true;
        for (p, l) in self.programs.iter_mut().zip(&mut t.layers) {
            match (p, l) {
                (Program::Linear(p), Layer::Linear(l)) => p.commit(l, cursor, n)?,
                (Program::Full(p), Layer::Full(l)) => p.commit(l, cursor, n)?,
                _ => return Err("B8 commit inventory".into()),
            }
        }
        if n > 0 {
            let c = t.dflash_capture.as_mut().ok_or("missing capture")?;
            c.begin(cursor, n)?;
            for (slot, &layer) in crate::DFLASH_TARGET_CAPTURE_LAYER_IDS.iter().enumerate() {
                c.layer_window(layer, cursor, n, &self.features[slot])?;
            }
            copy(
                &self.stream,
                &t.hidden[0],
                0,
                self.hidden[0].view((n - 1) * H * 4, H * 4)?,
            )?;
            copy(
                &self.stream,
                &t.logits,
                0,
                self.logits.view((n - 1) * 248320 * 4, 248320 * 4)?,
            )?;
            self.stream.synchronize()?;
            c.commit()?;
        } else {
            self.stream.synchronize()?;
        }
        t.consumed = cursor + n;
        t.selected = if n == 0 { selected } else { None };
        t.logits_current = true;
        t.poisoned = false;
        self.pending = None;
        self.poisoned = false;
        Ok(())
    }
}
impl Drop for Batch {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
    }
}

#[cfg(test)]
impl Batch {
    pub(crate) fn enable_core_attention_candidate(&mut self) -> Result<()> {
        if !self.quiescent() { return Err("core candidate requires quiescent B8".into()); }
        for program in &mut self.programs {
            if let Program::Full(p) = program { p.enable_core_attention()?; }
        }
        Ok(())
    }
    pub(crate) fn diagnostic_logits(&self) -> Result<Vec<f32>> {
        read_f32(&self.stream, &self.logits)
    }
    pub(crate) fn inject_failure(&mut self) {
        if let Program::Linear(p) = &mut self.programs[0] {
            p.fail_after_recurrence = true;
        } else {
            panic!("expected GDN first layer");
        }
    }
}
