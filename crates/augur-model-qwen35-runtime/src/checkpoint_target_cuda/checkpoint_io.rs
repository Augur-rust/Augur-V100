//! Bounded native state transfer, adapted from the Rust/GGUF committed-session
//! checkpoint. No replay, second KV owner, giant host snapshot, or handle restore.
use super::*;
use crate::augur_session::checkpoint::Descriptor;
use crate::session_checkpoint::{self as transfer,CaptureSource,RestoreTarget,TransferPlan,Component,Manifest,TransferReceipt};
use crate::session_checkpoint::filesystem::{LocalCheckpointWriter,LocalCheckpointReader};
use std::io;
fn ioerr(e:impl std::fmt::Display)->io::Error {io::Error::other(e.to_string())}
struct Span<'a>{name:String,buffer:&'a OwnedCudaBuffer,bytes:usize}
struct Io<'a>{stream:&'a OwnedCudaStream,spans:Vec<Span<'a>>,metadata:Vec<u8>,plan:TransferPlan}
impl Drop for Io<'_>{fn drop(&mut self){let _=self.stream.synchronize();}}
impl Io<'_>{
    fn range(&self,component:usize,offset:u64,n:usize)->io::Result<usize>{
        let start=usize::try_from(offset).map_err(ioerr)?;
        let spec=self.plan.components.get(component).ok_or_else(||ioerr("checkpoint component"))?;
        if n==0 || start.checked_add(n).is_none_or(|end|end as u64>spec.bytes) {return Err(ioerr("checkpoint transfer extent"));}Ok(start)
    }
}
impl CaptureSource for Io<'_>{
    fn begin(&mut self)->io::Result<TransferPlan>{self.stream.synchronize().map_err(ioerr)?;Ok(self.plan.clone())}
    fn read_exact(&mut self,c:usize,offset:u64,out:&mut[u8])->io::Result<()> {
        let start=self.range(c,offset,out.len())?;
        if c==0 {out.copy_from_slice(&self.metadata[start..start+out.len()]);}
        else {self.spans[c-1].buffer.view(start,out.len()).map_err(ioerr)?.copy_to_host_sync(self.stream,out).map_err(ioerr)?;}Ok(())
    }
    fn finish(&mut self)->io::Result<TransferPlan>{self.stream.synchronize().map_err(ioerr)?;Ok(self.plan.clone())}
    fn abort(&mut self)->io::Result<()>{self.stream.synchronize().map_err(ioerr)}
}
impl RestoreTarget for Io<'_>{
    fn begin(&mut self,p:&TransferPlan)->io::Result<()> {
        if *p!=self.plan {return Err(ioerr("restore independently derived plan"));}self.stream.synchronize().map_err(ioerr)
    }
    fn write_exact(&mut self,c:usize,offset:u64,bytes:&[u8])->io::Result<()> {
        let start=self.range(c,offset,bytes.len())?;
        if c==0 {if self.metadata[start..start+bytes.len()]!=*bytes {return Err(ioerr("restored boundary descriptor drift"));}}
        else {self.spans[c-1].buffer.copy_range_from_host_sync(self.stream,start,bytes).map_err(ioerr)?;}Ok(())
    }
    fn finish_unpublished(&mut self)->io::Result<()>{self.stream.synchronize().map_err(ioerr)}
    fn abort(&mut self)->io::Result<()>{self.stream.synchronize().map_err(ioerr)}
}
impl CheckpointCudaTarget {
    /// Independent full-state readback for ignored GPU acceptance only; never
    /// part of production lookup, prefill, capture or restore timing.
    #[cfg(test)]
    pub(crate) fn checkpoint_test_state_sha256(&mut self,d:&Descriptor)->Result<String> {
        use sha2::{Digest,Sha256};
        self.validate_cursor()?;self.drain_session_work()?;
        if self.consumed!=d.tokens().len() || !self.logits_current {return Err("state audit boundary".into());}
        let mut source=self.checkpoint_io(d)?;CaptureSource::begin(&mut source)?;
        let mut hash=Sha256::new();let mut staging=vec![0;transfer::MAX_CHUNK_BYTES];
        for component in 0..source.plan.components.len() {
            let spec=source.plan.components[component].clone();
            hash.update(spec.name.as_bytes());hash.update(spec.bytes.to_le_bytes());
            let mut offset=0;
            while offset<spec.bytes {
                let n=(spec.bytes-offset).min(staging.len() as u64) as usize;
                source.read_exact(component,offset,&mut staging[..n])?;
                hash.update(&staging[..n]);offset+=n as u64;
            }
        }
        source.finish()?;Ok(format!("{:x}",hash.finalize()))
    }

    #[cfg(test)]
    pub(crate) fn checkpoint_test_zero_kv_rows(&mut self,rows:usize)->Result<usize> {
        if rows==0 || rows>self.capacity || self.consumed!=0 {return Err("zero KV audit extent/cursor".into());}
        self.stream.synchronize()?;
        let bytes=rows.div_ceil(256)*256*1088;let mut checked=0;
        let mut staging=vec![0u8;transfer::MAX_CHUNK_BYTES];
        for layer in &self.layers {if let Layer::Full(full)=layer {
            for (role,buffer) in full.checkpoint_buffers() {
                let mut offset=0;
                while offset<bytes {
                    let n=(bytes-offset).min(staging.len());
                    buffer.view(offset,n)?.copy_to_host_sync(&self.stream,&mut staging[..n])?;
                    if staging[..n].iter().any(|&b|b!=0) {return Err(format!("failed restore left nonzero {role} KV at byte {offset}").into());}
                    checked+=n;offset+=n;
                }
            }
        }}
        Ok(checked)
    }

    fn checkpoint_io(&self,d:&Descriptor)->Result<Io<'_>> {
        let n=d.tokens().len();
        if !crate::augur_session::CONTEXT_CHOICES.contains(&self.capacity) || n==0 || n>=self.capacity || self.layers.len()!=64
            || self.augur_decode.is_none()
            || d.bindings.decode_attention != self.attention_mode().label() {return Err("unsupported QUASAR checkpoint owner".into());}
        #[cfg(feature="qwen35-augur-vision")]
        if self.image_rope_active {return Err("Vision checkpoints are not admitted; text cache never aliases images".into());}
        let physical=n.div_ceil(256)*256;
        let mut spans=Vec::new();
        for (index,layer) in self.layers.iter().enumerate() {
            match layer {
                Layer::Linear(l)=>{for (role,buffer) in l.checkpoint_buffers(){spans.push(Span{name:format!("layer{index:02}.{role}"),bytes:buffer.len_bytes(),buffer});}},
                Layer::Full(l)=>{for (role,buffer) in l.checkpoint_buffers(){
                    if buffer.len_bytes()!=self.capacity*1088 {return Err("checkpoint packed KV layout".into());}
                    spans.push(Span{name:format!("layer{index:02}.{role}"),bytes:physical*1088,buffer});}},
            }
        }
        let capture=self.dflash_capture.as_ref().ok_or("checkpoint requires initialized DFlash feature capture")?;
        let (buffer,bytes)=capture.checkpoint_ring(n)?;spans.push(Span{name:"target.feature_ring".into(),buffer,bytes});
        // Input-only snapshots also resume an equal-length prompt without replay.
        spans.push(Span{name:"target.logits".into(),buffer:&self.logits,bytes:self.logits.len_bytes()});
        let plan=d.plan(spans.iter().map(|s|Component{name:s.name.clone(),bytes:s.bytes as u64}).collect())?;
        Ok(Io{stream:&self.stream,spans,metadata:d.encode()?,plan})
    }
    pub(crate) fn checkpoint_plan(&self,d:&Descriptor)->Result<TransferPlan>{Ok(self.checkpoint_io(d)?.plan.clone())}
    pub(crate) fn checkpoint_save(&mut self,d:&Descriptor,path:&Path)->Result<TransferReceipt> {
        self.validate_cursor()?;self.drain_session_work()?;
        if self.poisoned || self.consumed!=d.tokens().len() || !self.augur_rotated_kv || !self.logits_current {
            return Err("checkpoint capture needs a complete rotated target boundary".into());}
        self.dflash_capture.as_ref().ok_or("missing capture")?.checkpoint_check(self.consumed)?;
        if let Some(pending)=d.pending(){if self.select_next_token()?!=pending {return Err("checkpoint pending token mismatch".into());}}
        let mut source=self.checkpoint_io(d)?;let plan=source.plan.clone();
        let mut writer=LocalCheckpointWriter::create(path,&plan)?;
        Ok(transfer::save(&mut source,&mut writer,&plan,transfer::MAX_CHUNK_BYTES)?)
    }
    pub(crate) fn checkpoint_restore(&mut self,d:&Descriptor,path:&Path,manifest_sha:&str)->Result<TransferReceipt> {
        self.checkpoint_restore_inner::<false>(d,path,manifest_sha,1,transfer::ParallelRestoreSchedule::Batch).map(|(r,_)|r)
    }
    pub(crate) fn checkpoint_restore_with_timing(&mut self,d:&Descriptor,path:&Path,manifest_sha:&str)->Result<(TransferReceipt,serde_json::Value)> {
        let (receipt,detail)=self.checkpoint_restore_inner::<true>(d,path,manifest_sha,1,transfer::ParallelRestoreSchedule::Batch)?;
        Ok((receipt,detail.ok_or("missing restore timing")?))
    }
    pub(crate) fn checkpoint_restore_parallel(&mut self,d:&Descriptor,path:&Path,manifest_sha:&str,workers:usize)->Result<TransferReceipt> {
        if !(2..=8).contains(&workers){return Err("invalid restore verifier worker count".into());}
        self.checkpoint_restore_inner::<false>(d,path,manifest_sha,workers,transfer::ParallelRestoreSchedule::Batch).map(|(r,_)|r)
    }
    pub(crate) fn checkpoint_restore_parallel_scheduled(&mut self,d:&Descriptor,path:&Path,manifest_sha:&str,
        workers:usize,schedule:transfer::ParallelRestoreSchedule)->Result<TransferReceipt> {
        if !(2..=8).contains(&workers) {return Err("invalid restore verifier worker count".into());}
        self.checkpoint_restore_inner::<false>(d,path,manifest_sha,workers,schedule).map(|(r,_)|r)
    }
    fn checkpoint_restore_inner<const TIMED:bool>(&mut self,d:&Descriptor,path:&Path,manifest_sha:&str,workers:usize,schedule:transfer::ParallelRestoreSchedule)->Result<(TransferReceipt,Option<serde_json::Value>)> {
        let tick=||if TIMED {Some(std::time::Instant::now())} else {None};
        let elapsed=|t:Option<std::time::Instant>|t.map_or(0.0,|v|v.elapsed().as_secs_f64());
        let started=tick();let phase=tick();
        if self.poisoned || self.consumed!=0 || self.layers.iter().any(|l|l.tokens()!=0) {
            return Err("checkpoint restore requires an unpublished empty target".into());}
        let plan=self.checkpoint_plan(d)?;let plan_s=elapsed(phase);let phase=tick();
        // Read and validate the independently derived manifest before mutation.
        let mut reader=LocalCheckpointReader::open(path,manifest_sha,&plan)?;
        let manifest_bytes=reader.manifest_bytes().to_vec();
        Manifest::decode(&manifest_bytes,manifest_sha,&plan)?;
        let reader_setup_s=elapsed(phase);let phase=tick();
        self.drain_session_work()?;self.poisoned=true;
        for layer in &mut self.layers {if let Layer::Full(full)=layer {full.checkpoint_prepare_restore(d.tokens().len())?;}}
        let drain_s=elapsed(phase);let phase=tick();
        let (restored,transfer_timing)={let mut destination=self.checkpoint_io(d)?;
            if TIMED {let (r,t)=transfer::restore_with_timing(&mut reader,&mut destination,&manifest_bytes,manifest_sha,&plan)?;(r,Some(t))}
            else if workers>1 {
                let r=match schedule {
                    transfer::ParallelRestoreSchedule::Batch => transfer::restore_parallel_verified(&mut reader,&mut destination,&manifest_bytes,manifest_sha,&plan,workers)?,
                    transfer::ParallelRestoreSchedule::Pipeline => transfer::restore_pipelined_verified(&mut reader,&mut destination,&manifest_bytes,manifest_sha,&plan,workers)?,
                };(r,None)
            }
            else {(transfer::restore(&mut reader,&mut destination,&manifest_bytes,manifest_sha,&plan)?,None)}};
        let transfer_call_s=elapsed(phase);let phase=tick();
        let n=d.tokens().len();
        for layer in &mut self.layers {match layer {Layer::Linear(l)=>l.checkpoint_resume(n)?,Layer::Full(l)=>l.checkpoint_resume(n)?}}
        self.dflash_capture.as_mut().ok_or("missing capture")?.checkpoint_resume(n)?;
        self.consumed=n;self.augur_rotated_kv=true;
        self.info["active_kv_format"]=self.attention_mode().kv_identity().into();
        self.selected=None;self.logits_current=true;self.poisoned=false;
        self.validate_cursor()?;
        if let Some(expected)=d.pending() {
            let result=self.select_next_token();
            if result.as_ref().ok()!=Some(&expected) {self.poisoned=true;return Err("restored checkpoint pending/logits validation".into());}
        }
        let detail=if TIMED {Some(serde_json::json!({"diagnostic_only":true,"plan_s":plan_s,"reader_setup_s":reader_setup_s,
            "drain_s":drain_s,"transfer_call_s":transfer_call_s,"resume_s":elapsed(phase),"complete_s":elapsed(started),
            "payload_bytes":restored.payload_bytes,"chunks":restored.chunks,"staging_bytes":restored.staging_bytes,
            "transfer":transfer_timing,"write_scope":"metadata comparison plus synchronous host-to-device copies"}))}else{None};
        Ok((restored,detail))
    }
}
