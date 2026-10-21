//! Retained top-20 device buffers. Only 160 bytes per row cross to the host.
use super::*;
use augur_backend::{CudaF32View,CudaQwen35VocabTopkArgsV1};
use crate::augur_sampling::{self,Selection,TOP_K};
pub(super) struct Sampler {
    stream:Rc<OwnedCudaStream>,_logits:Rc<OwnedCudaBuffer>,output:OwnedCudaBuffer,
    args:CudaQwen35VocabTopkArgsV1,rows:usize,_tile_scratch:Option<OwnedCudaBuffer>,
}
impl Sampler {
    pub(super) fn new(stream:Rc<OwnedCudaStream>,logits:Rc<OwnedCudaBuffer>,rows:usize)->Result<Self> {
        let bytes=rows*TOP_K*4;let output=OwnedCudaBuffer::allocate(stream.device_ordinal(),bytes*2)?;
        let mut args=CudaQwen35VocabTopkArgsV1::bind(&stream,
            CudaF32View::new(logits.view(0,rows*248320*4)?,PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,rows*248320)?,
            output.view(0,bytes)?,output.view(bytes,bytes)?,rows,TOP_K)?;
        let tiled=match std::env::var("AUGUR_SAMPLING_TOPK") {
            Err(std::env::VarError::NotPresent)=>false,
            Ok(v) if v=="reference"=>false,
            Ok(v) if v=="tiled"=>true,
            _=>return Err("AUGUR_SAMPLING_TOPK requires reference or tiled".into()),
        };
        let tile_scratch=if tiled {
            let parts=if rows==1 {64}else{32};
            let size=rows*parts*TOP_K*8;
            let scratch=OwnedCudaBuffer::allocate(stream.device_ordinal(),size)?;
            args=args.with_tiled_scratch(scratch.view(0,size)?,parts)?;Some(scratch)
        } else {None};
        Ok(Self{stream,_logits:logits,output,args,rows,_tile_scratch:tile_scratch})
    }
    fn topk_schedule(&self)->&'static str {if self._tile_scratch.is_some(){"tiled_exact_v1"}else{"single_cta_reference_v1"}}
    pub(super) fn submit(&self)->Result<()> {self.args.submit(&self.stream)?;Ok(())}
    /// Opt-in diagnostic readback only; never participates in token selection.
    pub(super) fn rejection_audit_rows(&self,first_position:usize)->Result<Vec<serde_json::Value>> {
        let bytes=self.rows*TOP_K*4;let mut raw=vec![0u8;bytes*2];
        self.output.copy_to_host_sync(&self.stream,&mut raw)?;
        let ids:Vec<_>=raw[..bytes].chunks_exact(4).map(|b|u32::from_le_bytes(b.try_into().unwrap())).collect();
        let logits:Vec<_>=raw[bytes..].chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();
        (0..self.rows).map(|r| {
            let position=first_position.checked_add(r).ok_or("audit position overflow")?;
            let row_ids=&ids[r*TOP_K..(r+1)*TOP_K];let row_logits=&logits[r*TOP_K..(r+1)*TOP_K];
            let u=augur_sampling::process_uniform(0,position)?;
            let selected=augur_sampling::select(row_ids,row_logits,u)?;
            Ok(serde_json::json!({"absolute_position":position,"ids":row_ids,"logits":row_logits,
                "uniform":u,"selected":selected.id,"top1":selected.top1}))
        }).collect()
    }
    pub(super) fn read_verified(&self,first_position:usize,path:&[u32],q:&crate::augur_speculative_sampling::Proposal)->Result<Vec<Selection>> {
        if path.len()!=self.rows || q.start.checked_add(1)!=Some(first_position) {return Err("foreign probability path/cursor".into());}
        let bytes=self.rows*TOP_K*4;let mut raw=vec![0u8;bytes*2];self.output.copy_to_host_sync(&self.stream,&mut raw)?;
        let ids:Vec<_>=raw[..bytes].chunks_exact(4).map(|b|u32::from_le_bytes(b.try_into().unwrap())).collect();
        let logits:Vec<_>=raw[bytes..].chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();
        (0..self.rows).map(|r| {
            let position=first_position.checked_add(r).ok_or("probability position overflow")?;
            let row_ids=&ids[r*TOP_K..(r+1)*TOP_K];let row_logits=&logits[r*TOP_K..(r+1)*TOP_K];
            Ok(if r+1<self.rows {crate::augur_speculative_sampling::verify(row_ids,row_logits,q,r+1,path[r+1],position)?}
                else {augur_sampling::select(row_ids,row_logits,augur_sampling::process_uniform(crate::augur_speculative_sampling::BONUS,position)?)?})
        }).collect()
    }
    /// Block verification over the same retained rows; returns accepted draft count and selections.
    pub(super) fn read_block_verified(&self,first_position:usize,path:&[u32],q:&crate::augur_speculative_sampling::Proposal)->Result<(usize,Vec<Selection>)> {
        if path.len()!=self.rows || q.start.checked_add(1)!=Some(first_position) {return Err("foreign probability path/cursor".into());}
        let bytes=self.rows*TOP_K*4;let mut raw=vec![0u8;bytes*2];self.output.copy_to_host_sync(&self.stream,&mut raw)?;
        let ids:Vec<_>=raw[..bytes].chunks_exact(4).map(|b|u32::from_le_bytes(b.try_into().unwrap())).collect();
        let logits:Vec<_>=raw[bytes..].chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();
        let rows:Vec<(&[u32],&[f32])>=(0..self.rows).map(|r|(&ids[r*TOP_K..(r+1)*TOP_K],&logits[r*TOP_K..(r+1)*TOP_K])).collect();
        Ok(crate::augur_speculative_sampling::verify_block(&rows,q,path)?)
    }
    pub(super) fn read(&self,first_position:usize)->Result<Vec<Selection>> {
        let bytes=self.rows*TOP_K*4;let mut raw=vec![0u8;bytes*2];self.output.copy_to_host_sync(&self.stream,&mut raw)?;
        let ids:Vec<_>=raw[..bytes].chunks_exact(4).map(|b|u32::from_le_bytes(b.try_into().unwrap())).collect();
        let logits:Vec<_>=raw[bytes..].chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();
        (0..self.rows).map(|r|Ok(augur_sampling::select(&ids[r*TOP_K..(r+1)*TOP_K],&logits[r*TOP_K..(r+1)*TOP_K],
            augur_sampling::process_uniform(0,first_position.checked_add(r).ok_or("sampling position overflow")?)?)?)).collect()
    }
}
impl Drop for Sampler {fn drop(&mut self){let _=self.stream.synchronize();}}
impl CheckpointCudaTarget {
    pub(crate) fn configure_sampling(&mut self)->Result<()> {
        if self.consumed!=0 || self.selected.is_some(){return Err("sampling must be configured before prefill".into());}
        let mode=augur_sampling::Mode::from_env()?;
        let sampling=augur_sampling::effective_description()?;
        if mode==augur_sampling::Mode::Thinking {self.sampler=Some(Sampler::new(Rc::clone(&self.stream),Rc::clone(&self.logits),1)?);}
        self.info["sampling"]=sampling;
        if let Some(sampler)=&self.sampler {self.info["sampling"]["topk_schedule"]=serde_json::json!(sampler.topk_schedule());}
        Ok(())
    }
    pub(super) fn cache_sampled_selection(&mut self,selection:Selection)->Result<()> {
        self.token.copy_from_host_sync(&self.stream,&selection.id.to_le_bytes())?;
        self.selected=Some(selection.id);
        self.sampling_trace.record(self.consumed,selection.id,Some(selection.top1))?;Ok(())
    }
    pub(crate) fn begin_sampling_trace(&mut self) {self.sampling_trace=Default::default();}
    pub(crate) fn sampling_detail(&self)->serde_json::Value {
        let mut v=self.info.get("sampling").cloned().unwrap_or_else(||augur_sampling::Mode::Greedy.description());
        if self.sampler.is_some(){v["trajectory"]=self.sampling_trace.detail();}v
    }
}
