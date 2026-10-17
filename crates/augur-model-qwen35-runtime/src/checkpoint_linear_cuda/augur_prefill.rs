//! Speed-first Augur adapter: QUASAR Tensor Core projections and shared F32 GDN.
use super::*;
use crate::checkpoint_projection_cuda::CheckpointWideSmallGateSource;
use augur_backend::cuda_qwen35_prefill_cublas::{
    CudaQwen35CheckpointPrefillLinearChunkV1, CudaQwen35PrefillCublasPlanV1,
    CudaQwen35PrefillLinearChunkV1, CudaQwen35PrefillLinearChunkViewsV1,
    CudaQwen35PrefillRmsnormV1, CudaQwen35AugurProjectionV1 as Projection,
    CudaQwen35AugurNormGroupB8V1 as NormGroup, NormGroupMemberV1,
};
use augur_backend::cuda_qwen35_qpn::OwnedCudaQwen35QpnWeightV1;
const H: usize = 5120;
const I: usize = 17408;
pub(crate) mod arena;
pub(crate) mod dflash;
mod recorded_graph;
use arena::Plane;
pub(crate) struct Workspace {
    pub(crate) stream: Rc<OwnedCudaStream>,
    pub(crate) phase: Rc<CudaQwen35PrefillCublasPlanV1>,
    pub(crate) planes: BTreeMap<&'static str, Plane>,
    pub(crate) full_planes: BTreeMap<&'static str, Plane>,
    pub(crate) ffn_planes: [Plane; 3],
    pub(crate) max_tokens: usize,
}
impl Workspace {
    pub(crate) fn owned_recipe(tokens: usize) -> Result<usize> {
        if !(1..=augur_backend::cuda_qwen35_prefill_cublas::AUGUR_MAX_BATCH).contains(&tokens)
        {
            return Err("Augur width exceeds compiled superbatch capacity".into());
        }
        Ok(I * H * 2 + I * tokens * 2 + arena::Layout::active(tokens)?.bytes)
    }
    pub(crate) fn begin(stream: Rc<OwnedCudaStream>, tokens: usize) -> Result<Rc<Self>> {
        Self::begin_kind(stream,tokens,false)
    }
    pub(crate) fn begin_packed_b8(stream:Rc<OwnedCudaStream>)->Result<Rc<Self>> {
        Self::begin_kind(stream,8,true)
    }
    fn begin_kind(stream:Rc<OwnedCudaStream>,tokens:usize,packed_b8:bool)->Result<Rc<Self>> {
        Self::owned_recipe(tokens)?;
        let phase=Rc::new(if packed_b8 {CudaQwen35PrefillCublasPlanV1::new_direct_b8(&stream,I)?}
            else {CudaQwen35PrefillCublasPlanV1::new(&stream,I*H,I,tokens)?});
        let arena = arena::bind(&stream, arena::Layout::active(tokens)?)?;
        Ok(Rc::new(Self {
            stream,
            phase,
            planes: arena.common_gdn,
            full_planes: arena.full,
            ffn_planes: arena.ffn,
            max_tokens: tokens,
        }))
    }
}
/// Three contractions share the Augur plan and F32 SwiGLU planes.
pub(crate) struct Ffn {
    gate: Projection,
    up: Projection,
    down: Projection,
    silu: CudaSiluMulF32ArgsV1,
    workspace: Rc<Workspace>,
}
impl Ffn {
    pub(crate) fn bind(
        w: Rc<Workspace>,
        weights: [Rc<OwnedCudaQwen35QpnWeightV1>; 3],
        input: Buffer,
        input_offset: usize,
        output: Buffer,
        output_offset: usize,
        tokens: usize,
    ) -> Result<Self> {
        let [gate, up, down] = weights;
        let bind = |weight, x, xo, y, yo| {
            Projection::bind(
                Rc::clone(&w.phase),
                Rc::clone(&w.stream),
                weight,
                x,
                xo,
                y,
                yo,
                tokens,
            )
        };
        let gate = bind(
            gate,
            Rc::clone(&input),
            input_offset,
            Rc::clone(&w.ffn_planes[0].owner),
            w.ffn_planes[0].offset,
        )?;
        let up = bind(
            up,
            input,
            input_offset,
            Rc::clone(&w.ffn_planes[1].owner),
            w.ffn_planes[1].offset,
        )?;
        let silu = if Rc::ptr_eq(&w.ffn_planes[0].owner, &w.ffn_planes[2].owner)
            && w.ffn_planes[0].offset == w.ffn_planes[2].offset
        {
            CudaSiluMulF32ArgsV1::bind_in_place_gate(
                &w.stream,
                w.ffn_planes[0].view(0, tokens * I)?,
                w.ffn_planes[1].view(0, tokens * I)?,
            )?
        } else {
            CudaSiluMulF32ArgsV1::bind(
                &w.stream,
                w.ffn_planes[0].view(0, tokens * I)?,
                w.ffn_planes[1].view(0, tokens * I)?,
                w.ffn_planes[2].view(0, tokens * I)?,
            )?
        };
        let down = bind(
            down,
            Rc::clone(&w.ffn_planes[2].owner),
            w.ffn_planes[2].offset,
            output,
            output_offset,
        )?;
        Ok(Self {
            gate,
            up,
            down,
            silu,
            workspace: w,
        })
    }
    pub(crate) fn use_direct_b8(&mut self)->Result<()> {
        self.gate.use_direct_b8()?;self.up.use_direct_b8()?;self.down.use_direct_b8()?;Ok(())
    }
    pub(crate) fn gate_up(&self) -> [&Projection; 2] { [&self.gate, &self.up] }
    pub(crate) fn submit_gate_up(&self) -> Result<()> {
        Ok(Projection::submit_same_input([&self.gate, &self.up])?)
    }
    pub(crate) fn submit_activation_down(&self) -> Result<()> {
        Ok(self.down.submit_with_silu(&self.silu)?)
    }
    pub(crate) fn submit(&self) -> Result<()> {
        self.submit_gate_up()?;
        self.submit_activation_down()?;
        Ok(())
    }
}
/// Own one pair of temporary padded COMPRESSED weights per layer per phase,
/// shared by all width bindings. Not 512 copies and never expanded permanently.
pub(crate) struct Source {
    small: CheckpointWideSmallGateSource,
    pub(crate) state: Buffer,
    pub(crate) history: Buffer,
}
impl Source {
    pub(crate) fn new(layer: &CheckpointLinearCudaLayer) -> Result<Rc<Self>> {
        let [Operation::Norm(_), Operation::Projection(_), Operation::Projection(_), Operation::SmallGatePair(pair), Operation::Gates(_), Operation::Recurrent(_), Operation::Projection(_), Operation::Add(_), Operation::Norm(_), Operation::QpnFfn(_), Operation::Add(_)] =
            layer.operations.as_slice()
        else {
            return Err("QUASAR GDN requires exact Augur profile8 paired-gate inventory".into());
        };
        if layer.poisoned {
            return Err("poisoned QUASAR source".into());
        }
        Ok(Rc::new(Self {
            small: pair.wide_phase_source()?,
            state: Rc::clone(&layer.buffers["state"]),
            history: Rc::clone(&layer.buffers["history"]),
        }))
    }
    #[cfg(test)]
    pub(crate) fn diagnostic_padded_identity(&self) -> Result<serde_json::Value> {
        self.small.diagnostic_identity()
    }
    pub(crate) const PADDED_BYTES: usize = CheckpointWideSmallGateSource::PADDED_BYTES;
}
fn prefix(b: &OwnedCudaBuffer, offset: usize, n: usize) -> Result<CudaF32View> {
    Ok(CudaF32View::new(
        b.view(offset, n * 4)?,
        PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
        n,
    )?)
}
#[derive(Debug,Clone,Copy)]
pub(crate) struct DflashGpuBill {
    pub(crate) input_projection_s:f64,
    pub(crate) recurrence_s:f64,
    pub(crate) output_ffn_s:f64,
    pub(crate) output_projection_s:f64,
    pub(crate) post_attention_s:f64,
    pub(crate) ffn_gate_up_s:f64,
    pub(crate) ffn_activation_down_s:f64,
    pub(crate) final_add_s:f64,
}
struct DflashGpuEventPair { start:augur_backend::OwnedCudaEvent, stop:augur_backend::OwnedCudaEvent }
impl DflashGpuEventPair {
    fn new(device:i32)->Result<Self>{Ok(Self{start:augur_backend::OwnedCudaEvent::create(device)?,stop:augur_backend::OwnedCudaEvent::create(device)?})}
    fn elapsed_s(&self)->Result<f64>{Ok(f64::from(self.start.elapsed_ms(&self.stop)?)/1000.0)}
}
struct DflashGpuTimer {
    input_projection:DflashGpuEventPair, recurrence:DflashGpuEventPair, output_ffn:DflashGpuEventPair,
    output_projection:DflashGpuEventPair, post_attention:DflashGpuEventPair,
    ffn_gate_up:DflashGpuEventPair, ffn_activation_down:DflashGpuEventPair, final_add:DflashGpuEventPair,
}
impl DflashGpuTimer {
    fn new(stream:&OwnedCudaStream)->Result<Self>{
        let d=stream.device_ordinal();
        Ok(Self{input_projection:DflashGpuEventPair::new(d)?,recurrence:DflashGpuEventPair::new(d)?,
            output_ffn:DflashGpuEventPair::new(d)?,output_projection:DflashGpuEventPair::new(d)?,
            post_attention:DflashGpuEventPair::new(d)?,ffn_gate_up:DflashGpuEventPair::new(d)?,
            ffn_activation_down:DflashGpuEventPair::new(d)?,final_add:DflashGpuEventPair::new(d)?})
    }
    fn read(&self)->Result<DflashGpuBill>{Ok(DflashGpuBill{
        input_projection_s:self.input_projection.elapsed_s()?,recurrence_s:self.recurrence.elapsed_s()?,
        output_ffn_s:self.output_ffn.elapsed_s()?,output_projection_s:self.output_projection.elapsed_s()?,
        post_attention_s:self.post_attention.elapsed_s()?,ffn_gate_up_s:self.ffn_gate_up.elapsed_s()?,
        ffn_activation_down_s:self.ffn_activation_down.elapsed_s()?,final_add_s:self.final_add.elapsed_s()?})}
}
pub(crate) struct Program {
    recorded_graph: Option<recorded_graph::RecordedGraph>,
    workspace: Rc<Workspace>,
    source: Rc<Source>,
    _vectors: Vec<Buffer>,
    _input: Buffer,
    _output: Buffer,
    input_norm: CudaQwen35PrefillRmsnormV1,
    post_norm: CudaQwen35PrefillRmsnormV1,
    qkv: Projection,
    z: Projection,
    alpha: Projection,
    beta: Projection,
    gdn: CudaQwen35CheckpointPrefillLinearChunkV1,
    out: Projection,
    residual: CudaAddF32ArgsV1,
    ffn: Ffn,
    final_add: CudaAddF32ArgsV1,
    pub(crate) tokens: usize,
    dflash_gpu_timer: Option<DflashGpuTimer>,
    /// Opt-in fused direct-B8 glue (AUGUR_GDN_INPUT_KERNEL=fused); see use_fused_b8_glue.
    fused: Option<FusedGlue>,
    #[cfg(test)]
    pub(crate) fail_after_recurrence: bool,
}
/// Two closed native programs that replace the GDN input and post-attention glue.
/// Declared before nothing it borrows: both hold only raw descriptors over owners
/// that this Program and its Workspace retain until drop (after stream sync).
struct FusedGlue { input: NormGroup, input_recorded: Option<NormGroup>, post: NormGroup }
impl Program {
    pub(crate) fn use_direct_b8(&mut self)->Result<()> {
        if self.recorded_graph.is_some() {return Err("captured GDN projection is immutable".into());}
        for p in [&mut self.qkv,&mut self.z,&mut self.alpha,&mut self.beta,&mut self.out] {p.use_direct_b8()?;}
        self.ffn.use_direct_b8()
    }
    pub(crate) const PROFILE_LABEL: &'static str = "quasar_gdn_contiguous3_f32_v1";
    /// Opt-in fused direct-B8 glue. Same arithmetic, fewer launches:
    ///  input: norm -> FP16 stage -> {qkv,z,alpha,beta} in one grid; alpha/beta write
    ///         their 48 live columns straight into alpha/beta_raw (no strided copy) and,
    ///         when recording, the identical words into the replay records.
    ///  post:  residual add -> norm -> FP16 stage -> {gate,up} in one grid.
    /// Requires direct B8 (use_direct_b8) and no captured graph.
    pub(crate) fn use_fused_b8_glue(&mut self,replay:Option<&dflash::Replay>)->Result<()> {
        if self.recorded_graph.is_some() || self.tokens!=8 {return Err("fused GDN glue requires uncaptured direct B8".into());}
        let p=&self.workspace.planes;
        let trim=|name:&str|->Result<NormGroupMemberV1>{Ok(NormGroupMemberV1{out_cols:48,output:Some(p[name].view(0,8*48)?.buffer_view()),record:None})};
        let plain=|rows|NormGroupMemberV1{out_cols:rows,output:None,record:None};
        let members=|records:Option<&[Buffer;4]>|->Result<[(&Projection,NormGroupMemberV1);4]>{
            let mut m=[(&self.qkv,plain(10240)),(&self.z,plain(6144)),(&self.alpha,trim("alpha")?),(&self.beta,trim("beta_raw")?)];
            if let Some(r)=records {
                for (i,cols) in [10240,6144,48,48].into_iter().enumerate() {m[i].1.record=Some(r[i].view(0,8*cols*4)?);}
            }
            Ok(m)
        };
        let input=NormGroup::bind(&self.input_norm,&members(None)?,None)?;
        let input_recorded=match replay {
            Some(r)=>{let records=r.graph_records(&self.workspace)?;Some(NormGroup::bind(&self.input_norm,&members(Some(&records))?,None)?)},
            None=>None,
        };
        let [gate,up]=self.ffn.gate_up();
        let post=NormGroup::bind(&self.post_norm,&[(gate,plain(I)),(up,plain(I))],Some(&self.residual))?;
        self.fused=Some(FusedGlue{input,input_recorded,post});
        Ok(())
    }
    pub(crate) fn fused_b8_glue(&self)->bool {self.fused.is_some()}
    pub(crate) fn enable_dflash_gpu_bill(&mut self)->Result<()> {
        if self.recorded_graph.is_some() {return Ok(());}
        if self.dflash_gpu_timer.is_none() {self.dflash_gpu_timer=Some(DflashGpuTimer::new(&self.workspace.stream)?);}
        Ok(())
    }
    pub(crate) fn dflash_gpu_bill(&self)->Result<Option<DflashGpuBill>> {
        self.dflash_gpu_timer.as_ref().map(DflashGpuTimer::read).transpose()
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn bind(
        layer: &CheckpointLinearCudaLayer,
        workspace: Rc<Workspace>,
        source: Rc<Source>,
        input: Buffer,
        input_offset: usize,
        output: Buffer,
        output_offset: usize,
        tokens: usize,
        epsilon: f32,
    ) -> Result<Self> {
        if layer.poisoned
            || Rc::ptr_eq(&input, &output)
            || !(1..=workspace.max_tokens).contains(&tokens)
            || !Rc::ptr_eq(&layer.stream, &workspace.stream)
            || !Rc::ptr_eq(&source.state, &layer.buffers["state"])
            || !Rc::ptr_eq(&source.history, &layer.buffers["history"])
        {
            return Err("QUASAR layer owner/width mismatch".into());
        }
        let [Operation::Norm(_), Operation::Projection(qkv), Operation::Projection(z), Operation::SmallGatePair(pair), Operation::Gates(_), Operation::Recurrent(_), Operation::Projection(out), Operation::Add(_), Operation::Norm(_), Operation::QpnFfn(ffn), Operation::Add(_)] =
            layer.operations.as_slice()
        else {
            return Err("QUASAR layer operation inventory changed".into());
        };
        if !source.small.same_owner(pair) {
            return Err("foreign padded gate source".into());
        }
        let w = &workspace;
        let s = &w.stream;
        let b = &w.planes;
        let v = |name: &str, cols: usize| b[name].view(0, tokens * cols);
        let scalar = |name: &str| view(&layer.buffers[name]);
        let input_norm = CudaQwen35PrefillRmsnormV1::bind(
            s,
            prefix(&input, input_offset, tokens * H)?,
            scalar("input_norm_weight")?,
            v("normalized", H)?,
            tokens,
            H,
            epsilon,
        )?;
        let post_norm = CudaQwen35PrefillRmsnormV1::bind(
            s,
            v("attention_residual", H)?,
            scalar("post_norm_weight")?,
            v("post_norm", H)?,
            tokens,
            H,
            epsilon,
        )?;
        let qkv = Projection::bind(
            Rc::clone(&w.phase),
            Rc::clone(s),
            qkv.retained_qpn()?.original_weight(),
            Rc::clone(&b["normalized"].owner),
            b["normalized"].offset,
            Rc::clone(&b["qkv"].owner),
            b["qkv"].offset,
            tokens,
        )?;
        let z = Projection::bind(
            Rc::clone(&w.phase),
            Rc::clone(s),
            z.retained_qpn()?.original_weight(),
            Rc::clone(&b["normalized"].owner),
            b["normalized"].offset,
            Rc::clone(&b["z"].owner),
            b["z"].offset,
            tokens,
        )?;
        let alpha = Projection::bind(
            Rc::clone(&w.phase),
            Rc::clone(s),
            Rc::clone(&source.small.padded[0]),
            Rc::clone(&b["normalized"].owner),
            b["normalized"].offset,
            Rc::clone(&b["alpha64"].owner),
            b["alpha64"].offset,
            tokens,
        )?;
        let beta = Projection::bind(
            Rc::clone(&w.phase),
            Rc::clone(s),
            Rc::clone(&source.small.padded[1]),
            Rc::clone(&b["normalized"].owner),
            b["normalized"].offset,
            Rc::clone(&b["beta64"].owner),
            b["beta64"].offset,
            tokens,
        )?;
        let gdn = CudaQwen35PrefillLinearChunkV1::bind(
            s,
            CudaQwen35PrefillLinearChunkViewsV1 {
                qkv: v("qkv", 10240)?,
                z: v("z", 6144)?,
                conv_weights: scalar("conv")?,
                conv_history: scalar("history")?,
                alpha_raw: v("alpha", 48)?,
                beta_raw: v("beta_raw", 48)?,
                dt_bias: scalar("dt_bias")?,
                ssm_a: scalar("ssm_a")?,
                state: scalar("state")?,
                norm_weight: scalar("norm")?,
                activated: v("activated", 10240)?,
                recurrent: v("recurrent", 6144)?,
                output: v("linear_output", 6144)?,
            },
            tokens,
            epsilon,
        )?
        .for_checkpoint_source()?;
        let out = Projection::bind(
            Rc::clone(&w.phase),
            Rc::clone(s),
            out.retained_qpn()?.original_weight(),
            Rc::clone(&b["linear_output"].owner),
            b["linear_output"].offset,
            Rc::clone(&b["attention"].owner),
            b["attention"].offset,
            tokens,
        )?;
        let residual = CudaAddF32ArgsV1::bind(
            s,
            prefix(&input, input_offset, tokens * H)?,
            v("attention", H)?,
            v("attention_residual", H)?,
        )?;
        let ffn = Ffn::bind(
            Rc::clone(w),
            ffn.retained().original_weights(),
            Rc::clone(&b["post_norm"].owner),
            b["post_norm"].offset,
            Rc::clone(&b["ffn_output"].owner),
            b["ffn_output"].offset,
            tokens,
        )?;
        let final_add = CudaAddF32ArgsV1::bind(
            s,
            v("attention_residual", H)?,
            v("ffn_output", H)?,
            prefix(&output, output_offset, tokens * H)?,
        )?;
        let vectors = [
            "input_norm_weight",
            "post_norm_weight",
            "conv",
            "dt_bias",
            "ssm_a",
            "norm",
        ]
        .map(|n| Rc::clone(&layer.buffers[n]))
        .to_vec();
        Ok(Self {
            workspace,
            source,
            _vectors: vectors,
            _input: input,
            _output: output,
            recorded_graph: None,
            input_norm,
            post_norm,
            qkv,
            z,
            alpha,
            beta,
            gdn,
            out,
            residual,
            ffn,
            final_add,
            tokens,
            dflash_gpu_timer: None,
            fused: None,
            #[cfg(test)]
            fail_after_recurrence: false,
        })
    }
    /// All layer state mutation is one poisoned transaction. No independent
    /// consumer can skip a producer or use stale stage data at the same address.
    pub(crate) fn submit(&self, layer: &mut CheckpointLinearCudaLayer) -> Result<()> {
        self.submit_impl(layer,None)
    }
    pub(crate) fn submit_recorded(&self,layer:&mut CheckpointLinearCudaLayer,replay:&mut dflash::Replay)->Result<()> {
        self.submit_impl(layer,Some(replay))
    }
    fn submit_impl(&self, layer: &mut CheckpointLinearCudaLayer, mut replay:Option<&mut dflash::Replay>) -> Result<()> {
        let w = &self.workspace;
        let s = &w.stream;
        if layer.poisoned
            || !Rc::ptr_eq(s, &layer.stream)
            || !Rc::ptr_eq(&self.source.state, &layer.buffers["state"])
            || !Rc::ptr_eq(&self.source.history, &layer.buffers["history"])
        {
            return Err("poisoned/foreign QUASAR layer submit".into());
        }
        let next = layer
            .tokens
            .checked_add(self.tokens)
            .ok_or("Augur token overflow")?;
        if let Some(replay)=replay.as_deref() {
            replay.validate_record_width(self.tokens)?;
            if let Some(graph)=&self.recorded_graph {replay.validate_graph_records(&graph.records)?;}
        }
        layer.poisoned = true;
        if let Some(replay)=replay.as_deref_mut() {replay.mark_recorded_width(self.tokens)?;}
        let graph=self.recorded_graph.as_ref().filter(|_|replay.is_some());
        #[cfg(test)]
        let graph=graph.filter(|_|!self.fail_after_recurrence);
        if let Some(graph)=graph {graph.exec.launch(s)?;}
        else {self.submit_operations(replay.as_deref())?;}
        layer.tokens = next;
        layer.poisoned = false;
        Ok(())
    }
    fn submit_operations(&self,replay:Option<&dflash::Replay>)->Result<()> {
        let w=&self.workspace;
        let s=&w.stream;
        if let Some(timer)=&self.dflash_gpu_timer {timer.input_projection.start.record(s)?;}
        if let Some(f)=&self.fused {
            // Records are written by the grouped stores themselves, before recurrence/FFN
            // reuse the aliased arena planes, exactly where the copies used to run.
            match (replay,&f.input_recorded) {
                (Some(_),Some(recorded))=>recorded.submit()?,
                (Some(_),None)=>return Err("fused GDN glue bound without replay records".into()),
                (None,_)=>f.input.submit()?,
            }
        } else {
        self.input_norm.submit(s)?;
        Projection::submit_same_input([&self.qkv, &self.z, &self.alpha, &self.beta])?;
        // SAFETY: all source/destination owners retained by this Program and
        // Workspace; same serial stream; exact64->48 row footprints checked.
        for (from, to) in [("alpha64", "alpha"), ("beta64", "beta_raw")] {
            unsafe {
                w.planes[to].owner.copy_from_device_2d_async_unchecked(
                    s,
                    w.planes[to].offset,
                    48 * 4,
                    w.planes[from].view(0, self.tokens * 64)?.buffer_view(),
                    64 * 4,
                    48 * 4,
                    self.tokens,
                )?;
            }
        }
        // The shared arena may alias projection planes with later FFN scratch.
        // Record before recurrence/FFN, never after the entire layer returns.
        if let Some(replay)=replay {replay.submit_record_copies(self.tokens)?;}
        }
        if let Some(timer)=&self.dflash_gpu_timer {timer.input_projection.stop.record(s)?;timer.recurrence.start.record(s)?;}
        self.gdn.submit(s)?;
        #[cfg(test)]
        if self.fail_after_recurrence {
            return Err("injected QUASAR GDN partial failure".into());
        }
        if let Some(timer)=&self.dflash_gpu_timer {
            timer.recurrence.stop.record(s)?;
            timer.output_ffn.start.record(s)?;
            timer.output_projection.start.record(s)?;
        }
        self.out.submit()?;
        if let Some(timer)=&self.dflash_gpu_timer {
            timer.output_projection.stop.record(s)?;
            timer.post_attention.start.record(s)?;
        }
        if let Some(f)=&self.fused {
            // residual + norm + stage + gate/up: the bill's post_attention span is empty.
            if let Some(timer)=&self.dflash_gpu_timer {
                timer.post_attention.stop.record(s)?;
                timer.ffn_gate_up.start.record(s)?;
            }
            f.post.submit()?;
        } else {
        self.residual.submit_for_plan(s)?;
        self.post_norm.submit(s)?;
        if let Some(timer)=&self.dflash_gpu_timer {
            timer.post_attention.stop.record(s)?;
            timer.ffn_gate_up.start.record(s)?;
        }
        self.ffn.submit_gate_up()?;
        }
        if let Some(timer)=&self.dflash_gpu_timer {
            timer.ffn_gate_up.stop.record(s)?;
            timer.ffn_activation_down.start.record(s)?;
        }
        self.ffn.submit_activation_down()?;
        if let Some(timer)=&self.dflash_gpu_timer {
            timer.ffn_activation_down.stop.record(s)?;
            timer.final_add.start.record(s)?;
        }
        self.final_add.submit_for_plan(s)?;
        if let Some(timer)=&self.dflash_gpu_timer {
            timer.final_add.stop.record(s)?;
            timer.output_ffn.stop.record(s)?;
        }
        Ok(())
    }

}
impl Drop for Program {
    fn drop(&mut self) {
        let _ = self.workspace.stream.synchronize();
    }
}

/// Only committed bytes; diagnostic readback stays outside performance timing.
#[cfg(test)]
pub(crate) fn diagnostic_committed_bytes(
    layer: &CheckpointLinearCudaLayer,
) -> Result<Vec<(&'static str, Vec<u8>)>> {
    if layer.poisoned || layer.tokens == 0 {
        return Err("uncommitted layer diagnostic".into());
    }
    let mut result = Vec::new();
    for name in ["state", "history"] {
        let buffer = &layer.buffers[name];
        let mut bytes = vec![0; buffer.len_bytes()];
        buffer
            .view(0, bytes.len())?
            .copy_to_host_sync(&layer.stream, &mut bytes)?;
        result.push((name, bytes));
    }
    Ok(result)
}

