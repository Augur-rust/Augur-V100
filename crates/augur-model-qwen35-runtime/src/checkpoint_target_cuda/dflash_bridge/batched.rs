//! Rust/QUASAR recorded-prefix target verification for Augur.
//!
//! The original owner is B8. Widths B2/B3/B4 are opt-in diagnostic target
//! widths: they reuse the authenticated B8 draft path but execute only the
//! selected target rows. They never change the target model, cache identity,
//! or commit authority. B8 remains the product-compatible default.
use super::*;
#[cfg(test)]
pub(super) mod supervision;
#[cfg(test)]
mod complete_teacher;
use crate::checkpoint_full_cuda::augur_prefill as full;
use crate::checkpoint_linear_cuda::augur_prefill as gdn;
use augur_backend::{CudaBf16TokenRowsGatherArgsV1,CudaQwen35VocabTop1ArgsV1};
use augur_backend::cuda_qwen35_prefill_cublas::CudaQwen35PrefillRmsnormV1;
pub(super) const PROFILE:&str="quasar_recorded_prefix_width_v1";
const MAX_B:usize=B;
type Buffer=Rc<OwnedCudaBuffer>;
enum Program {Gdn(gdn::Program,gdn::dflash::Replay),Full(full::Program,full::dflash::Checkpoint)}
pub(super) struct Executor {
    pub(super) width:usize,
    pub(super) projection_policy:&'static str,
    stream:Rc<OwnedCudaStream>,_workspace:Rc<full::Workspace>,hidden:[Buffer;2],
    features:Vec<Buffer>,normalized:Buffer,head_normalized:Buffer,logits:Buffer,ids:OwnedCudaBuffer,
    /// Optional page-locked D2H staging for the final width token IDs.
    /// Disabled by default; enabled only for an A/B candidate.
    ids_pinned_host:Option<augur_backend::OwnedPinnedD2hBuffer>,
    gather:CudaBf16TokenRowsGatherArgsV1,norm:CudaQwen35PrefillRmsnormV1,
    head:crate::checkpoint_projection_cuda::CheckpointSharedBf16HeadRows,
    top1:CudaQwen35VocabTop1ArgsV1,programs:Vec<Program>,
    gpu_timer:Option<GpuComponentTimer>,gpu_last_bill:Option<GpuComponentBill>,
    #[cfg(feature="qwen35-augur-http")]
    sampler:Option<crate::checkpoint_target_cuda::sampling::Sampler>,
    #[cfg(feature="qwen35-augur-http")]
    sampled_rows:Vec<crate::augur_sampling::Selection>,
    /// Block verification commit boundary for the pending round (None = token verification).
    #[cfg(feature="qwen35-augur-http")]
    block_tau:Option<usize>,
    last_ids_d2h_s:f64,
    pending:Option<usize>,parked:bool,owned_bytes:u64,
}
#[derive(Debug,Clone,Copy)]
pub struct GpuComponentBill {
    pub gather_s:f64,
    pub gdn_layers_s:f64,
    pub gdn_input_projection_s:f64,
    pub gdn_recurrence_s:f64,
    pub gdn_output_ffn_s:f64,
    pub gdn_output_projection_s:f64,
    pub gdn_post_attention_s:f64,
    pub gdn_ffn_gate_up_s:f64,
    pub gdn_ffn_activation_down_s:f64,
    pub gdn_final_add_s:f64,
    pub full_attention_layers_s:f64,
    pub target_norm_s:f64,
    pub output_head_s:f64,
    pub top1_s:f64,
    /// Host wall for the final width*4-byte ID copy plus its stream wait.
    /// This is diagnostic only; it is not the full target execute wall.
    pub ids_d2h_s:f64,
    pub gdn_layer_count:usize,
    pub full_layer_count:usize,
    pub event_pairs:usize,
    pub gdn_stage_event_pairs:usize,
}
struct GpuEventPair { start:augur_backend::OwnedCudaEvent, stop:augur_backend::OwnedCudaEvent }
impl GpuEventPair {
    fn new(device:i32)->Result<Self>{Ok(Self{start:augur_backend::OwnedCudaEvent::create(device)?,stop:augur_backend::OwnedCudaEvent::create(device)?})}
    fn elapsed_s(&self)->Result<f64>{Ok(f64::from(self.start.elapsed_ms(&self.stop)?)/1000.0)}
}
struct GpuComponentTimer {
    gather:GpuEventPair,
    layers:Vec<GpuEventPair>,
    gdn:Vec<bool>,
    target_norm:GpuEventPair,
    output_head:GpuEventPair,
    top1:GpuEventPair,
}
impl GpuComponentTimer {
    fn enabled()->bool {std::env::var("AUGUR_DFLASH_GPU_BILL").as_deref()==Ok("1")}
    fn new(stream:&OwnedCudaStream,programs:&[Program])->Result<Self>{
        let device=stream.device_ordinal();
        Ok(Self{
            gather:GpuEventPair::new(device)?,
            layers:programs.iter().map(|_|GpuEventPair::new(device)).collect::<Result<Vec<_>>>()?,
            gdn:programs.iter().map(|p|matches!(p,Program::Gdn(_, _))).collect(),
            target_norm:GpuEventPair::new(device)?,
            output_head:GpuEventPair::new(device)?,
            top1:GpuEventPair::new(device)?,
        })
    }
    fn read(&self,programs:&[Program],ids_d2h_s:f64)->Result<GpuComponentBill>{
        let mut gdn_layers_s=0.0;let mut full_attention_layers_s=0.0;
        let mut gdn_input_projection_s=0.0;let mut gdn_recurrence_s=0.0;let mut gdn_output_ffn_s=0.0;
        let mut gdn_output_projection_s=0.0;let mut gdn_post_attention_s=0.0;
        let mut gdn_ffn_gate_up_s=0.0;let mut gdn_ffn_activation_down_s=0.0;let mut gdn_final_add_s=0.0;
        let mut gdn_layer_count=0;let mut full_layer_count=0;

        for (event,is_gdn) in self.layers.iter().zip(&self.gdn) {
            let seconds=event.elapsed_s()?;
            if *is_gdn {gdn_layers_s+=seconds;gdn_layer_count+=1;}
            else {full_attention_layers_s+=seconds;full_layer_count+=1;}
        }
        for program in programs {if let Program::Gdn(program,_) = program {if let Some(bill)=program.dflash_gpu_bill()? {
            gdn_input_projection_s+=bill.input_projection_s;gdn_recurrence_s+=bill.recurrence_s;gdn_output_ffn_s+=bill.output_ffn_s;
            gdn_output_projection_s+=bill.output_projection_s;gdn_post_attention_s+=bill.post_attention_s;
            gdn_ffn_gate_up_s+=bill.ffn_gate_up_s;gdn_ffn_activation_down_s+=bill.ffn_activation_down_s;
            gdn_final_add_s+=bill.final_add_s;
        }}}
        Ok(GpuComponentBill{
            gather_s:self.gather.elapsed_s()?,gdn_layers_s,gdn_input_projection_s,gdn_recurrence_s,gdn_output_ffn_s,
            gdn_output_projection_s,gdn_post_attention_s,gdn_ffn_gate_up_s,gdn_ffn_activation_down_s,gdn_final_add_s,
            full_attention_layers_s,target_norm_s:self.target_norm.elapsed_s()?,output_head_s:self.output_head.elapsed_s()?,
            top1_s:self.top1.elapsed_s()?,ids_d2h_s,gdn_layer_count,full_layer_count,
            event_pairs:self.layers.len()+4,gdn_stage_event_pairs:gdn_layer_count*8,
        })
    }
}
#[derive(Debug)]
pub struct BatchedVerifiedRound {
    pub output_tokens:Vec<u32>,pub first_rejection:Option<Value>,pub stop:&'static str,
    pub start_consumed:usize,pub end_consumed:usize,pub batch_verify_commit_s:f64,
    pub target_width:usize,
    /// Opt-in component wall clocks at the existing synchronized boundaries.
    pub verify_execute_s:Option<f64>,pub verify_decision_s:Option<f64>,pub verify_commit_s:Option<f64>,pub ids_d2h_s:Option<f64>,
    pub verify_gpu_component_bill:Option<GpuComponentBill>,
}
fn copy(s:&OwnedCudaStream,to:&OwnedCudaBuffer,offset:usize,from:augur_backend::CudaBufferView)->Result<()> {
    // SAFETY: caller keeps all source and destination owners, on this owner stream.
    unsafe{to.copy_from_device_async_unchecked(s,offset,from)?;} Ok(())
}
fn decision(anchor:u32,path:&[u32],ids:&[u32],remaining:usize,eos:&[u32])
    ->Result<(Vec<u32>,usize,Option<Value>,&'static str)> {
    if path.is_empty() || path.len()>MAX_B || path[0]!=anchor || remaining==0 || eos.is_empty() || ids.len()!=path.len() {return Err("width decision admission".into());}
    let width=path.len();
    let mut output=Vec::with_capacity(width);
    for i in 0..width {
        let correct=if i==0 {anchor}else{ids[i-1]};
        if eos.contains(&correct) {return Ok((output,i,None,"eos_pending"));}
        if i>0 && path[i]!=correct {return Ok((output,i,Some(json!({"offset":i,
            "correct_target_token":correct,"proposed_token":path[i],"correct_top16_rank":null,
            "canonical_labels_after_this_rejection":false,"reason":"width_target_first_rejection"})),"first_rejection_pending"));}
        output.push(correct);
        if output.len()==remaining {return Ok((output,i,None,"length_pending"));}
    }
    Ok((output,width,None,"block_complete"))
}
/// Same publication/EOS/budget rules as `decision`, but the boundary is the block verifier's tau:
/// drafts 1..=tau are accepted and ids[tau] is the pending residual/bonus token, even when it
/// equals path[tau+1] (block residuals need not exclude the rejected draft).
fn block_decision(anchor:u32,path:&[u32],ids:&[u32],tau:usize,remaining:usize,eos:&[u32])
    ->Result<(Vec<u32>,usize,Option<Value>,&'static str)> {
    if path.len()!=MAX_B || ids.len()!=MAX_B || tau>=MAX_B || path[0]!=anchor || remaining==0 || eos.is_empty() {return Err("block decision admission".into());}
    for i in 0..tau {if ids[i]!=path[i+1] {return Err("block accepted prefix drift".into());}}
    let mut output=Vec::with_capacity(MAX_B);
    for i in 0..MAX_B {
        let correct=if i==0 {anchor}else{ids[i-1]};
        if eos.contains(&correct) {return Ok((output,i,None,"eos_pending"));}
        if i>tau {return Ok((output,i,Some(json!({"offset":i,"correct_target_token":correct,"proposed_token":path[i],
            "correct_top16_rank":null,"canonical_labels_after_this_rejection":false,"reason":"block_verification_boundary"})),"first_rejection_pending"));}
        output.push(correct);
        if output.len()==remaining {return Ok((output,i,None,"length_pending"));}
    }
    Ok((output,MAX_B,None,"block_complete"))
}
fn requested_width()->Result<usize> {
    match std::env::var("AUGUR_DFLASH_TARGET_WIDTH") {
        Err(std::env::VarError::NotPresent)=>Ok(MAX_B),
        Ok(v)=>v.parse::<usize>().map_err(|_|"invalid AUGUR_DFLASH_TARGET_WIDTH".into()).and_then(|n|
            if (2..=MAX_B).contains(&n) {Ok(n)} else {Err("target width must be in 2..=8".into())}),
        Err(e)=>Err(format!("target width env: {e}").into()),
    }
}
impl Executor {
    pub(super) fn head_kernel_label(&self)->&'static str {self.head.kernel_label()}
    pub(super) fn gdn_input_kernel_label(&self)->&'static str {
        let fused=self.programs.iter().any(|p|matches!(p,Program::Gdn(p,_) if p.fused_b8_glue()));
        if fused {"augur_qpn_norm_group_b8_v1"} else {"separate_norm_stage_project_copy_v1"}
    }
    fn new(t:&CheckpointCudaTarget)->Result<Self> {
        let width=requested_width()?;
        if t.poisoned || !t.augur_rotated_kv || !crate::augur_session::CONTEXT_CHOICES.contains(&t.capacity) || t.consumed+width>t.capacity || t.layers.len()!=64 {
            return Err("width verifier requires current rotated Augur owner with supported capacity".into());}
        let requested=match std::env::var("AUGUR_DFLASH_TARGET_PROJECTION").as_deref() {
            Err(std::env::VarError::NotPresent)|Ok("full_f16")=>"full_f16",
            Ok("packed_b8")=>"packed_b8",
            _=>return Err("target projection must be full_f16 or packed_b8".into()),
        };
        let projection_policy=if width==MAX_B {requested} else {"full_f16"};
        Self::new_with_projection_width(t,projection_policy,width)
    }
    fn new_with_projection(t:&CheckpointCudaTarget,projection_policy:&'static str)->Result<Self> {
        Self::new_with_projection_width(t,projection_policy,MAX_B)
    }
    fn new_with_projection_width(t:&CheckpointCudaTarget,projection_policy:&'static str,width:usize)->Result<Self> {
        if !(2..=MAX_B).contains(&width) || t.poisoned || !t.augur_rotated_kv || !crate::augur_session::CONTEXT_CHOICES.contains(&t.capacity) || t.consumed+width>t.capacity || t.layers.len()!=64 {
            return Err("width target admission".into());}
        let before=cuda_device_allocation_snapshot().live_bytes;
        let s=&t.stream;let w=if projection_policy=="packed_b8" {full::Workspace::begin_packed_b8(Rc::clone(s),t.capacity,t.attention_mode()).map_err(|e|format!("workspace packed: {e}"))?} else {full::Workspace::begin(Rc::clone(s),width,t.capacity,t.attention_mode()).map_err(|e|format!("workspace width {width}: {e}"))?};
        let hidden=[empty(s,width*H).map_err(|e|format!("hidden0: {e}"))?,empty(s,width*H).map_err(|e|format!("hidden1: {e}"))?];
        let features=(0..5).map(|_|empty(s,width*H)).collect::<Result<Vec<_>>>().map_err(|e|format!("features: {e}"))?;
        let normalized=empty(s,width*H).map_err(|e|format!("normalized: {e}"))?;
        let head_width=if width==MAX_B {width} else {MAX_B};
        let head_normalized=if head_width==width {Rc::clone(&normalized)} else {empty(s,head_width*H).map_err(|e|format!("head normalized: {e}"))?};
        let logits=empty(s,head_width*V).map_err(|e|format!("logits: {e}"))?;
        let ids=OwnedCudaBuffer::allocate(s.device_ordinal(),width*4).map_err(|e|format!("ids: {e}"))?;
        let ids_pinned_host=if std::env::var("AUGUR_DFLASH_PINNED_D2H").as_deref()==Ok("1") {
            Some(augur_backend::OwnedPinnedD2hBuffer::allocate(s.device_ordinal(),width*4)
                .map_err(|e|format!("pinned ids d2h: {e}"))?)
        } else {None};
        let weight=CudaBf16WeightViewV1::new(t._embedding.view(0,t._embedding.len_bytes())?,PhysicalLayoutId::DEVICE_CONTIGUOUS_BF16_CUDA_V1,V,H)?;
        let gather=CudaBf16TokenRowsGatherArgsV1::bind(s,weight,t.prompt_tokens.view(0,width*4)?,
            CudaF32View::new(hidden[0].view(0,width*H*4)?,PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,width*H)?,width).map_err(|e|format!("gather: {e}"))?;
        let epsilon=t._artifact.config().rms_norm_eps as f32;
        let norm=t.head.bind_target_norm_rows(&hidden[0],&normalized,epsilon,width).map_err(|e|format!("target norm: {e}"))?;
        let head=t.head.bind_shared_normalized_bf16_rows(s,Rc::clone(&head_normalized),Rc::clone(&logits),head_width).map_err(|e|format!("head: {e}"))?;
        let top1=CudaQwen35VocabTop1ArgsV1::bind(s,CudaF32View::new(logits.view(0,width*V*4)?,PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,width*V)?,ids.view(0,width*4)?,width)?;
        #[cfg(feature="qwen35-augur-http")]
        let sampler=if t.sampler.is_some(){Some(crate::checkpoint_target_cuda::sampling::Sampler::new(Rc::clone(s),Rc::clone(&logits),width)?)}else{None};
        let mut programs=Vec::with_capacity(64);
        for (i,l) in t.layers.iter().enumerate() {
            let input=Rc::clone(&hidden[i%2]);let output=Rc::clone(&hidden[(i+1)%2]);
            programs.push(match l {
                Layer::Linear(l)=>Program::Gdn(gdn::Program::bind(l,Rc::clone(&w.common),gdn::Source::new(l).map_err(|e|format!("gdn source layer {i}: {e}"))?,input,0,output,0,width,epsilon).map_err(|e|format!("gdn bind layer {i}: {e}"))?,
                    gdn::dflash::Replay::new(l,Rc::clone(&w.common),epsilon).map_err(|e|format!("gdn replay layer {i}: {e}"))?),
                Layer::Full(l)=>Program::Full(full::Program::bind(l,Rc::clone(&w),input,0,output,0,width,t.consumed).map_err(|e|format!("full bind layer {i}: {e}"))?,full::dflash::Checkpoint::new(l).map_err(|e|format!("full checkpoint layer {i}: {e}"))?),
            });
        }
        if projection_policy=="packed_b8" {for p in &mut programs {match p {
            Program::Gdn(p,_)=>p.use_direct_b8()?,Program::Full(p,_)=>p.use_direct_b8()?,
        }} }
        // GDN input/post-attention glue: separate launches (default) or the fused
        // direct-B8 programs. Same arithmetic; incompatible with the captured B8 graph.
        let fused_glue=match std::env::var("AUGUR_GDN_INPUT_KERNEL").as_deref() {
            Err(std::env::VarError::NotPresent)|Ok("separate")=>false,
            Ok("fused") if width==MAX_B && projection_policy=="packed_b8"
                && std::env::var("AUGUR_B8_GDN_GRAPH").as_deref()!=Ok("1")=>true,
            _=>return Err("AUGUR_GDN_INPUT_KERNEL must be separate, or fused with packed B8 and no GDN graph".into()),
        };
        if fused_glue {for p in &mut programs {if let Program::Gdn(p,r)=p {p.use_fused_b8_glue(Some(r))?;}}}
        match std::env::var("AUGUR_B8_GDN_GRAPH").as_deref() {
            Err(std::env::VarError::NotPresent)|Ok("0")=>{},
            Ok("1") if width==MAX_B && projection_policy=="packed_b8"=>{
                for p in &mut programs {if let Program::Gdn(p,r)=p {p.enable_recorded_b8_graph(r)?;}}
            },
            _=>return Err("AUGUR_B8_GDN_GRAPH requires 0 or packed B8 with 1".into()),
        }
        let gpu_bill_enabled=GpuComponentTimer::enabled();
        if gpu_bill_enabled && std::env::var("AUGUR_B8_GDN_GRAPH").as_deref()!=Ok("1") {for program in &mut programs {if let Program::Gdn(program,_) = program {program.enable_dflash_gpu_bill()?;}}}
        let gpu_timer=if gpu_bill_enabled{Some(GpuComponentTimer::new(s,&programs)?)}else{None};
        Ok(Self{
            #[cfg(feature="qwen35-augur-http")] sampler,
            #[cfg(feature="qwen35-augur-http")] sampled_rows:Vec::new(),
            #[cfg(feature="qwen35-augur-http")] block_tau:None,
            width,projection_policy,stream:Rc::clone(s),_workspace:w,hidden,features,normalized,head_normalized,logits,ids,ids_pinned_host,gather,norm,head,top1,programs,gpu_timer,gpu_last_bill:None,last_ids_d2h_s:0.0,pending:None,parked:false,owned_bytes:cuda_device_allocation_snapshot().live_bytes-before})
    }
    fn park(&mut self)->Result<()> {
        if self.pending.is_some() || self.programs.iter().any(|p|matches!(p,Program::Gdn(_,r) if !r.quiescent())) {
            return Err("width park requires completed GDN/KV transaction".into());
        }
        self.stream.synchronize()?;
        for p in &mut self.programs {if let Program::Full(p,c)=p {c.park()?;p.park_cache()?;}}
        self.parked=true;Ok(())
    }
    fn resume(&mut self,t:&CheckpointCudaTarget)->Result<()> {
        if !self.parked || self.pending.is_some() || t.poisoned || !t.augur_rotated_kv
            || !Rc::ptr_eq(&self.stream,&t.stream) || t.layers.len()!=self.programs.len()
            || self._workspace.mode != t.attention_mode()
            || t.info["active_kv_format"] != t.attention_mode().kv_identity() {
            return Err("foreign/stale width request plan resume".into());
        }
        for (p,l) in self.programs.iter_mut().zip(&t.layers) {match (p,l) {
            (Program::Full(p,c),Layer::Full(l))=>{c.resume(l)?;p.resume_cache(l)?;},
            (Program::Gdn(_,r),Layer::Linear(_)) if r.quiescent()=>{},
            _=>return Err("width resume layer inventory/transaction mismatch".into()),
        }}
        self.parked=false;Ok(())
    }
    #[cfg(test)]
    fn execute(&mut self,t:&mut CheckpointCudaTarget,path:&[u32])->Result<Vec<u32>> {self.execute_probability(t,path,None)}
    fn execute_probability(&mut self,t:&mut CheckpointCudaTarget,path:&[u32],probability:Option<&crate::augur_speculative_sampling::Proposal>)->Result<Vec<u32>> {
        let width=self.width;
        if self.parked || self.pending.is_some() || t.poisoned || !Rc::ptr_eq(&self.stream,&t.stream)
            || !t.augur_rotated_kv || self._workspace.mode != t.attention_mode()
            || t.info["active_kv_format"] != t.attention_mode().kv_identity() || path.len()!=width
            || path.iter().any(|&x|x as usize>=V) || t.consumed+width>t.capacity {return Err("width owner/path admission".into());}
        let cursor=t.consumed;t.poisoned=true;self.pending=Some(cursor);self.gpu_last_bill=None;self.last_ids_d2h_s=0.0;
        // Open every rollback transaction first, but defer the backup copies.
        // The old path submitted two D2D memcpys per layer (96 operations per
        // round). The ranges are independent, so aggregate all 64 layer pairs
        // into one bounded copy launch without changing the rollback bytes.
        for (p,l) in self.programs.iter_mut().zip(&t.layers) {match (p,l) {
            (Program::Gdn(_,r),Layer::Linear(l))=>r.begin_width_no_copy(l,cursor,width)?,
            (Program::Full(p,r),Layer::Full(l))=>{r.begin_width_no_copy(l,cursor,width)?;p.rebind_position(l,cursor,width)?;},
            _=>return Err("width layer inventory".into()),
        }}
        let mut rollback_copies=Vec::with_capacity(self.programs.len()*2);
        for p in &self.programs {match p {
            Program::Gdn(_,r)=>rollback_copies.extend(r.backup_copies()?),
            Program::Full(_,r)=>rollback_copies.extend(r.backup_copies(cursor,width)?),
        }}
        if rollback_copies.len()!=128 {return Err("width rollback copy inventory".into());}
        // SAFETY: every source/destination owner is retained by self/programs,
        // all ranges are disjoint, and the stream is drained by commit/drop.
        unsafe{augur_backend::OwnedCudaBuffer::copy_independent_batch_async_unchecked(&t.stream,&rollback_copies)?;}
        let bytes:Vec<_>=path.iter().flat_map(|id|id.to_le_bytes()).collect();
        t.prompt_tokens.copy_from_host_sync(&t.stream,&bytes)?;
        if let Some(timer)=&self.gpu_timer {timer.gather.start.record(&t.stream)?;}
        self.gather.submit_for_plan(&t.stream)?;
        if let Some(timer)=&self.gpu_timer {timer.gather.stop.record(&t.stream)?;}
        for (i,(p,l)) in self.programs.iter_mut().zip(&mut t.layers).enumerate() {
            if let Some(timer)=&self.gpu_timer {timer.layers[i].start.record(&t.stream)?;}
            match (p,l) {
                (Program::Gdn(p,r),Layer::Linear(l))=>p.submit_recorded(l,r)?,
                (Program::Full(p,_),Layer::Full(l))=>p.submit(l)?,
                _=>return Err("width layer inventory changed".into()),
            }
            if let Some(timer)=&self.gpu_timer {timer.layers[i].stop.record(&t.stream)?;}
            if let Some(slot)=crate::DFLASH_TARGET_CAPTURE_LAYER_IDS.iter().position(|&id|id==i) {
                copy(&t.stream,&self.features[slot],0,self.hidden[(i+1)%2].view(0,width*H*4)?)?;
            }
        }
        if let Some(timer)=&self.gpu_timer {timer.target_norm.start.record(&t.stream)?;}
        self.norm.submit(&t.stream)?;
        if width!=MAX_B { copy(&t.stream,&self.head_normalized,0,self.normalized.view(0,width*H*4)?)?; }
        if let Some(timer)=&self.gpu_timer {timer.target_norm.stop.record(&t.stream)?;timer.output_head.start.record(&t.stream)?;}
        self.head.submit()?;
        if let Some(timer)=&self.gpu_timer {timer.output_head.stop.record(&t.stream)?;timer.top1.start.record(&t.stream)?;}
        #[cfg(feature="qwen35-augur-http")]
        if let Some(sampler)=&self.sampler {sampler.submit()?;}else{self.top1.submit(&t.stream)?;}
        #[cfg(not(feature="qwen35-augur-http"))]
        self.top1.submit(&t.stream)?;
        if let Some(timer)=&self.gpu_timer {timer.top1.stop.record(&t.stream)?;}
        let ids_d2h_started=Instant::now();
        #[cfg(feature="qwen35-augur-http")]
        if let Some(sampler)=&self.sampler {
            self.block_tau=None;
            self.sampled_rows=match probability {
                Some(q) if crate::augur_speculative_sampling::block_from_env()? =>{let (tau,rows)=sampler.read_block_verified(cursor+1,path,q)?;self.block_tau=Some(tau);rows},
                Some(q)=>sampler.read_verified(cursor+1,path,q)?,
                None=>sampler.read(cursor+1)?,
            };
            self.last_ids_d2h_s=ids_d2h_started.elapsed().as_secs_f64();
            self.gpu_last_bill=self.gpu_timer.as_ref().map(|timer|timer.read(&self.programs,self.last_ids_d2h_s)).transpose()?;
            return Ok(self.sampled_rows.iter().map(|s|s.id).collect());
        }
        let ids=if let Some(host)=self.ids_pinned_host.as_mut() {
            // The pinned candidate submits the same exact device range, then waits
            // on its completion event. It does not add a stream-wide synchronize.
            let source=self.ids.view(0,width*4)?;
            // SAFETY: source is the checked width*4 range of self.ids, owned
            // by this exclusively borrowed executor. Top1 and D2H are ordered
            // on t.stream; wait_and_read below fences the copy before decoding
            // host bytes. No source mutation/free occurs in this interval.
            // On error, executor/host owners remain responsible for draining;
            // a failed wait is not permission to read or recycle the storage.
            unsafe { host.submit_from_view(&t.stream,source)?; }
            let raw=host.wait_and_read()?;
            let mut ids=Vec::with_capacity(width);
            for b in raw.chunks_exact(4){ids.push(u32::from_le_bytes(b.try_into().unwrap()));}
            ids
        } else {
            let mut raw=vec![0u8;width*4];
            self.ids.copy_to_host_sync(&t.stream,&mut raw)?;
            let mut ids=Vec::with_capacity(width);
            for b in raw.chunks_exact(4){ids.push(u32::from_le_bytes(b.try_into().unwrap()));}
            ids
        };
        self.last_ids_d2h_s=ids_d2h_started.elapsed().as_secs_f64();
        if ids.iter().any(|&id|id as usize>=V) {return Err("width target id out of vocabulary".into());}
        self.gpu_last_bill=self.gpu_timer.as_ref().map(|timer|timer.read(&self.programs,self.last_ids_d2h_s)).transpose()?;
        Ok(ids)
    }
    fn commit(&mut self,t:&mut CheckpointCudaTarget,n:usize)->Result<()> {
        let width=self.width;let cursor=self.pending.ok_or("missing width transaction")?;
        if n>width || t.consumed!=cursor || !t.poisoned {return Err("stale width commit".into());}
        for (p,l) in self.programs.iter_mut().zip(&mut t.layers) {match (p,l) {
            (Program::Gdn(_,r),Layer::Linear(l))=>r.commit_width(l,width,n)?,
            (Program::Full(_,r),Layer::Full(l))=>r.commit_width(l,width,n)?,
            _=>return Err("width commit layer inventory".into()),
        }}
        if n>0 {
            let c=t.dflash_capture.as_mut().ok_or("width capture missing")?;c.begin(cursor,n)?;
            for (slot,&layer) in crate::DFLASH_TARGET_CAPTURE_LAYER_IDS.iter().enumerate(){c.layer_window(layer,cursor,n,&self.features[slot])?;}
            copy(&t.stream,&t.hidden[0],0,self.hidden[0].view((n-1)*H*4,H*4)?)?;
            copy(&t.stream,&t.logits,0,self.logits.view((n-1)*V*4,V*4)?)?;
            t.stream.synchronize()?;c.commit()?;
        } else {t.stream.synchronize()?;}
        t.consumed=cursor+n;t.selected=None;t.logits_current=true;t.poisoned=false;self.pending=None;Ok(())
    }
}
impl Drop for Executor {fn drop(&mut self){let _=self.stream.synchronize();}}
impl AugurDflashBridge {
    /// End the retained request plan before the target's exclusive empty reset.
    pub fn release_batched_verifier(&mut self)->Result<()> {
        if self.poisoned || self.live_proposal.is_some() || self.batched.as_ref().is_some_and(|b|b.pending.is_some()) {
            return Err("width release requires a quiescent committed owner".into());
        }
        self.stream.synchronize()?;drop(self.batched.take());Ok(())
    }
    pub(crate) fn park_batched_verifier(&mut self)->Result<()> {
        if self.poisoned || self.live_proposal.is_some(){return Err("cannot park outstanding draft proposal".into());}
        if let Some(b)=self.batched.as_mut(){b.park()?;}Ok(())
    }
    pub(crate) fn batched_gdn_graph_nodes(&self)->u64 {
        self.batched.as_ref().map_or(0,|b|b.programs.iter().map(|p|match p {Program::Gdn(p,_)=>p.recorded_b8_graph_nodes(),_=>0}).sum())
    }
    pub(crate) fn batched_cached_owned_bytes(&self)->u64 {self.batched.as_ref().map_or(0,|b|b.owned_bytes)}
    pub fn prepare_batched_verifier(&mut self,t:&CheckpointCudaTarget)->Result<()> {
        self.validate_target(t)?;
        if self.target_execution!=TargetExecution::AugurRotatedQ8 || self.live_proposal.is_some() {return Err("width setup requires quiescent current target".into());}
        if self.batched.is_none(){self.batched=Some(Executor::new(t)?);}Ok(())
    }
    pub(crate) fn prepare_http_batched_verifier(&mut self,t:&CheckpointCudaTarget)->Result<&'static str> {
        self.validate_target(t)?;
        if self.live_proposal.is_some(){return Err("width HTTP setup with outstanding proposal".into());}
        if let Some(b)=self.batched.as_mut() {
            if b.width==MAX_B && b.projection_policy!="packed_b8" {return Err("HTTP target projection policy drift".into());}
            if b.parked {b.resume(t)?;Ok("resumed")}else{Ok("retained")}
        } else {let width=requested_width()?;self.batched=Some(Executor::new_with_projection_width(t,if width==MAX_B {"packed_b8"} else {"full_f16"},width)?);Ok("new")}
    }
    pub fn verify_resident_batched(&mut self,t:&mut CheckpointCudaTarget,p:AugurResidentDraftProposal,remaining:usize,eos:&[u32])->Result<BatchedVerifiedRound> {
        self.validate_target(t)?;
        if !Rc::ptr_eq(&self.instance,&p.instance) || self.live_proposal!=Some(p.generation)
            || !Rc::ptr_eq(&self.stream,&p.stream) || t.dflash_capture.as_ref().unwrap().committed_identity()?!=p.identity.capture_identity()
            || self.batched.is_none() || remaining==0 || eos.is_empty() || eos.iter().any(|&x|x as usize>=V) {return Err("stale width proposal/admission".into());}
        let anchor=t.select_next_token()?;let width=self.batched.as_ref().unwrap().width;
        if p.path.len()<width || p.path[0]!=anchor {return Err("width anchor/path changed".into());}
        self.poisoned=true;self.live_proposal=None;let start_consumed=t.consumed;let started=Instant::now();
        let result=(|| {
            static BILL:std::sync::OnceLock<bool>=std::sync::OnceLock::new();
            let bill=*BILL.get_or_init(||std::env::var("AUGUR_DFLASH_VERIFY_BILL").as_deref()==Ok("1"));
            let b=self.batched.as_mut().unwrap();let path=p.path[..b.width].to_vec();
            let execute_started=bill.then(Instant::now);
            let ids=b.execute_probability(t,&path,p.probability.as_ref())?;
            let verify_execute_s=execute_started.map(|s|s.elapsed().as_secs_f64());
            let decision_started=bill.then(Instant::now);
            #[cfg(feature="qwen35-augur-http")]
            let tau=b.block_tau;
            #[cfg(not(feature="qwen35-augur-http"))]
            let tau=None;
            let (output_tokens,n,mut rejection,stop)=match tau {Some(tau)=>block_decision(anchor,&path,&ids,tau,remaining,eos)?,None=>decision(anchor,&path,&ids,remaining,eos)?};
            if let Some(r)=&mut rejection {r["absolute_position"]=json!(start_consumed+n);}
            let verify_decision_s=decision_started.map(|s|s.elapsed().as_secs_f64());
            let commit_started=bill.then(Instant::now);
            b.commit(t,n)?;
            #[cfg(feature="qwen35-augur-http")]
            if b.sampler.is_some() && n>0 {
                // This is the already-sampled next token at the committed boundary.
                // Never resample a rejection, including when it becomes next round's anchor.
                for (row,selection) in b.sampled_rows[..n].iter().enumerate(){
                    t.sampling_trace.record(start_consumed+1+row,selection.id,Some(selection.top1))?;
                }
                t.cache_sampled_selection(b.sampled_rows[n-1])?;
            }
            let verify_commit_s=commit_started.map(|s|s.elapsed().as_secs_f64());
            if let Some(trace)=&mut self.rejection_trace {
                let sampler=b.sampler.as_ref().ok_or("rejection trace requires target sampling")?;
                let rows=sampler.rejection_audit_rows(start_consumed+1)?;
                for (row,id) in rows.iter().zip(&ids) {
                    if p.probability.is_none() && row["selected"].as_u64()!=Some(u64::from(*id)) {return Err("audit target selection drift".into());}
                }
                let draft=self.head.read_candidate_diagnostics(anchor)?;
                if draft.path[..b.width]!=path {return Err("audit draft path drift".into());}
                trace.record(json!({"generation":p.generation,"start_consumed":start_consumed,"end_consumed":t.consumed,
                    "path":path,"output_tokens":output_tokens,"first_rejection":rejection,"stop":stop,
                    "remaining":remaining,"target_rows":rows,"draft_candidates":draft.candidate_ids,
                    "verified_ids":ids,"proposal_q":p.probability.as_ref().map(|q|&q.q),
                    "probability_positions":p.probability.as_ref().map(|q|q.positions),
                    "proposal_calibration":p.probability.as_ref().map(|q|q.calibration),
                    "draft_unary":draft.unary_scores,"draft_transitions":draft.transition_scores,
                    "selector_path":self.head.selector_path,"selector_scale":self.head.selector_scale,
                    "selector_early_scales":self.head.selector_early_scales,
                    "gate_enabled":self.selector_gate_enabled,
                    "canonical_draft_slots":rejection_trace::checked_slots(output_tokens.len(),rejection.as_ref(),b.width)?}))?;
            }
            let verify_gpu_component_bill=b.gpu_last_bill;
            Ok(BatchedVerifiedRound{output_tokens,first_rejection:rejection,stop,start_consumed,end_consumed:t.consumed,
                batch_verify_commit_s:started.elapsed().as_secs_f64(),target_width:b.width,
                verify_execute_s,verify_decision_s,verify_commit_s,ids_d2h_s:Some(b.last_ids_d2h_s),verify_gpu_component_bill})
        })();
        if result.is_ok(){self.poisoned=false;}else{t.poisoned=true;}result
    }
    #[cfg(test)]
    pub(crate) fn audit_parked_no_cache_lease(&mut self,t:&mut CheckpointCudaTarget)->Result<Value> {
        let b=self.batched.as_mut().ok_or("missing cached width plan")?;
        if !b.parked {return Err("expected parked width plan".into());}
        for layer in &t.layers {if let Layer::Full(l)=layer {l.audit_exclusive_cache_lease()?;}}
        let state=(t.poisoned,t.consumed,t.selected,t.logits_current);let zeros=vec![0;b.width];
        if b.execute_probability(t,&zeros,None).is_ok() || state!=(t.poisoned,t.consumed,t.selected,t.logits_current) {return Err("parked plan was executable or mutated target".into());}
        Ok(json!({"parked":true,"KV_strong_count":1,"KV_weak_count":0,"submit_rejected_without_target_mutation":true,"retained_owned_bytes":b.owned_bytes,"target_width":b.width}))
    }
    #[cfg(test)]
    pub(super) fn audit_batched_zero_rollback(&mut self,t:&mut CheckpointCudaTarget,path:&[u32])->Result<()> {
        let identity=t.dflash_capture.as_ref().unwrap().committed_identity()?;let cursor=t.consumed;let logits=t.logits_snapshot()?;
        let width=self.batched.as_ref().ok_or("missing width executor")?.width;self.replay_batched_inputs(t,&path[..width],0)?;
        if identity!=t.dflash_capture.as_ref().unwrap().committed_identity()? || t.consumed!=cursor || logits!=t.logits_snapshot()? {return Err("zero-prefix changed public target/capture boundary".into());}
        for p in &self.batched.as_ref().unwrap().programs {match p {Program::Gdn(_,r)=>r.audit_zero_restored()?,Program::Full(_,r)=>r.audit_zero_restored(cursor)?}}
        Ok(())
    }
    #[cfg(test)]
    pub(super) fn replay_batched_inputs(&mut self,t:&mut CheckpointCudaTarget,path:&[u32],n:usize)->Result<()> {
        self.validate_target(t)?;if self.live_proposal.is_some(){return Err("replay with outstanding proposal".into());}
        let b=self.batched.as_mut().ok_or("width replay owner absent")?;
        if path.len()<b.width { return Err("width replay path shorter than target width".into()); }
        b.execute_probability(t,&path[..b.width],None)?;b.commit(t,n)
    }
}
#[cfg(all(test, windows, feature = "qwen35-augur-http"))]
mod planar_tests;

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn sampled_rejection_commits_only_prefix_and_keeps_the_rejection_draw() {
        let path=[10,11,12,13,14,15,16,17];
        for rejected in 1..8 {
            let mut sampled=[11,12,13,14,15,16,17,18];sampled[rejected-1]=100+rejected as u32;
            let (out,n,r,stop)=decision(10,&path,&sampled,99,&[999]).unwrap();
            assert_eq!(n,rejected);assert_eq!(out,path[..rejected]);
            assert_eq!(sampled[n-1],100+rejected as u32);
            assert_eq!(r.unwrap()["offset"],rejected);assert_eq!(stop,"first_rejection_pending");
        }
    }
    #[test] fn closing_think_is_a_pending_boundary_and_does_not_publish_answer_tail() {
        let close=248069;let path=[10,20,close,40,50,60,70,80];let ids=[20,close,40,50,60,70,80,90];
        let (out,n,reject,stop)=decision(10,&path,&ids,8,&[99,close]).unwrap();
        assert_eq!(out,vec![10,20]);assert_eq!(n,2);assert!(reject.is_none());assert_eq!(stop,"eos_pending");
        // At the next iteration the marker is an ordinary published anchor, using Q8.
        let path=[close,40,50,60,70,80,90,100];let ids=[40,50,60,70,80,90,100,110];
        let (out,n,reject,stop)=decision(close,&path,&ids,9,&[99]).unwrap();
        assert_eq!(out,path);assert_eq!(n,8);assert!(reject.is_none());assert_eq!(stop,"block_complete");
    }
    #[test] fn prefix_decision_preserves_pending_eos_budget_and_rejects_tail() {
        let path=[10,11,12,13,14,15,16,17];let ids=[11,12,13,14,15,16,17,18];
        for budget in 1..=8 {let (out,n,r,stop)=decision(10,&path,&ids,budget,&[99]).unwrap();assert_eq!(out,path[..budget]);assert_eq!(n,budget-1);assert!(r.is_none());assert_eq!(stop,"length_pending");}
        let (out,n,_,s)=decision(10,&path,&ids,99,&[99]).unwrap();assert_eq!(out,path);assert_eq!(n,8);assert_eq!(s,"block_complete");
        for at in 1..8 {let mut wrong=ids;wrong[at-1]=88;let(out,n,r,_)=decision(10,&path,&wrong,99,&[99]).unwrap();assert_eq!(out,path[..at]);assert_eq!(n,at);assert_eq!(r.unwrap()["offset"],at);}
        let mut terminal=ids;terminal[2]=99;let(out,n,r,s)=decision(10,&path,&terminal,99,&[99]).unwrap();assert_eq!(out,path[..3]);assert_eq!(n,3);assert!(r.is_none());assert_eq!(s,"eos_pending");
    }
    #[test] fn prefix_decision_supports_b2_b3_b4() {
        for width in [2usize,3,4] { let path:Vec<u32>=(10..10+width as u32).collect(); let ids:Vec<u32>=(11..11+width as u32).collect(); let (out,n,r,s)=decision(10,&path,&ids,99,&[99]).unwrap(); assert_eq!(out,path); assert_eq!(n,width); assert!(r.is_none()); assert_eq!(s,"block_complete"); }
    }
}
