//! Adapt the Rust/QUASAR recorded-projection prefix replay to QUASAR source GDN.
//! Never use the legacy interleaved-head checkpoint-fold ABI for this target.
use super::*;
use augur_backend::CudaBufferView;
const MAX_B: usize = 8;
pub(crate) struct Replay {
    workspace: Rc<Workspace>,
    state: [Buffer; 2],
    backup: [Buffer; 2],
    records: [Buffer; 4],
    prefixes: Vec<CudaQwen35CheckpointPrefillLinearChunkV1>,
    cursor: Option<usize>,
    recorded: bool,
    recorded_width: usize,
}
fn copy(s:&OwnedCudaStream,to:&OwnedCudaBuffer,from:augur_backend::CudaBufferView)->Result<()> {
    // SAFETY: all owners retained in Replay/Workspace; one ordered owner stream.
    unsafe { to.copy_from_device_async_unchecked(s,0,from)?; } Ok(())
}
impl Replay {
    pub(crate) fn quiescent(&self)->bool {self.cursor.is_none() && !self.recorded}

    pub(crate) fn new(layer:&CheckpointLinearCudaLayer,w:Rc<Workspace>,epsilon:f32)->Result<Self> {
        let state=[Rc::clone(&layer.buffers["history"]),Rc::clone(&layer.buffers["state"])];
        let backup=[empty(&w.stream,state[0].len_bytes()/4)?,empty(&w.stream,state[1].len_bytes()/4)?];
        let records=[empty(&w.stream,MAX_B*10240)?,empty(&w.stream,MAX_B*6144)?,empty(&w.stream,MAX_B*48)?,empty(&w.stream,MAX_B*48)?];
        let mut prefixes=Vec::with_capacity(w.max_tokens);
        for n in 1..=w.max_tokens {
            let v=|name:&str,cols|w.planes[name].view(0,n*cols);
            let scalar=|name:&str|view(&layer.buffers[name]);
            prefixes.push(CudaQwen35PrefillLinearChunkV1::bind(&w.stream,CudaQwen35PrefillLinearChunkViewsV1 {
                qkv:prefix(&records[0],0,n*10240)?,z:prefix(&records[1],0,n*6144)?,
                conv_weights:scalar("conv")?,conv_history:view(&state[0])?,
                alpha_raw:prefix(&records[2],0,n*48)?,beta_raw:prefix(&records[3],0,n*48)?,
                dt_bias:scalar("dt_bias")?,ssm_a:scalar("ssm_a")?,state:view(&state[1])?,
                norm_weight:scalar("norm")?,activated:v("activated",10240)?,
                recurrent:v("recurrent",6144)?,output:v("linear_output",6144)?,
            },n,epsilon)?.for_checkpoint_source()?);
        }
        Ok(Self{workspace:w,state,backup,records,prefixes,cursor:None,recorded:false,recorded_width:0})
    }
    pub(crate) fn begin(&mut self,layer:&CheckpointLinearCudaLayer,cursor:usize)->Result<()> {
        self.begin_width(layer,cursor,MAX_B)
    }
    pub(crate) fn begin_width(&mut self,layer:&CheckpointLinearCudaLayer,cursor:usize,width:usize)->Result<()> {
        self.begin_width_no_copy(layer,cursor,width)?;
        for (to,from) in self.backup.iter().zip(&self.state) {copy(&self.workspace.stream,to,from.view(0,from.len_bytes())?)?;}
        Ok(())
    }
    /// Opens the rollback transaction without submitting the two backup copies.
    /// The target executor uses this form to aggregate all GDN/KV backups into
    /// one bounded independent-copy launch; standalone callers retain the
    /// original begin_width behavior above.
    pub(crate) fn begin_width_no_copy(&mut self,layer:&CheckpointLinearCudaLayer,cursor:usize,width:usize)->Result<()> {
        if !(1..=MAX_B).contains(&width) || self.cursor.is_some() || layer.poisoned || layer.tokens!=cursor
            || !Rc::ptr_eq(&self.state[1],&layer.buffers["state"]) {return Err("stale GDN replay checkpoint".into());}
        self.cursor=Some(cursor);self.recorded=false;self.recorded_width=width;Ok(())
    }
    pub(crate) fn backup_copies(&self)->Result<[( &OwnedCudaBuffer, usize, CudaBufferView); 2]> {
        Ok([
            (&self.backup[0],0,self.state[0].view(0,self.state[0].len_bytes())?),
            (&self.backup[1],0,self.state[1].view(0,self.state[1].len_bytes())?),
        ])
    }
    pub(crate) fn record(&mut self)->Result<()> {
        let width=if self.recorded_width==0 {MAX_B} else {self.recorded_width};
        self.record_width(width)
    }
    pub(crate) fn record_width(&mut self,width:usize)->Result<()> {
        self.validate_record_width(width)?;
        self.submit_record_copies(width)?;
        self.mark_recorded_width(width)
    }
    pub(super) fn validate_record_width(&self,width:usize)->Result<()> {
        if !(1..=MAX_B).contains(&width) || self.cursor.is_none() || self.recorded || self.recorded_width!=width {return Err("duplicate/unbound GDN replay record".into());}
        Ok(())
    }
    pub(super) fn mark_recorded_width(&mut self,width:usize)->Result<()> {
        self.validate_record_width(width)?;
        self.recorded_width=width;self.recorded=true;Ok(())
    }
    pub(super) fn graph_records(&self,w:&Rc<Workspace>)->Result<[Buffer;4]> {
        if !self.quiescent() || !Rc::ptr_eq(&self.workspace,w) {return Err("foreign/nonquiescent GDN graph records".into());}
        Ok(self.records.each_ref().map(Rc::clone))
    }
    pub(super) fn validate_graph_records(&self,records:&[Buffer;4])->Result<()> {
        if self.records.iter().zip(records).any(|(a,b)|!Rc::ptr_eq(a,b)) {return Err("foreign GDN graph replay records".into());}
        Ok(())
    }
    pub(super) fn submit_record_copies(&self,width:usize)->Result<()> {
        if !(1..=MAX_B).contains(&width) {return Err("GDN graph record width".into());}
        for ((to,name),cols) in self.records.iter().zip(["qkv","z","alpha","beta_raw"]).zip([10240,6144,48,48]) {
            let from=&self.workspace.planes[name];
            copy(&self.workspace.stream,to,from.owner.view(from.offset,width*cols*4)?)?;
        }
        Ok(())
    }
    #[cfg(test)]
    pub(crate) fn audit_zero_restored(&self)->Result<()> {
        for (state,backup) in self.state.iter().zip(&self.backup) {
            let mut a=vec![0;state.len_bytes()];let mut b=vec![0;backup.len_bytes()];
            state.copy_to_host_sync(&self.workspace.stream,&mut a)?;backup.copy_to_host_sync(&self.workspace.stream,&mut b)?;
            if a!=b {return Err("GDN zero-prefix state/history changed".into());}
        }Ok(())
    }
    pub(crate) fn commit(&mut self,layer:&mut CheckpointLinearCudaLayer,n:usize)->Result<()> {
        let width=if self.recorded_width==0 {MAX_B} else {self.recorded_width};
        self.commit_width(layer,width,n)
    }
    pub(crate) fn commit_width(&mut self,layer:&mut CheckpointLinearCudaLayer,width:usize,n:usize)->Result<()> {
        let cursor=self.cursor.ok_or("missing GDN checkpoint")?;
        if !(1..=MAX_B).contains(&width) || !self.recorded || self.recorded_width!=width || n>width
            || layer.poisoned || layer.tokens!=cursor+width {return Err("invalid GDN prefix commit".into());}
        if n<width {
            for (to,from) in self.state.iter().zip(&self.backup) {copy(&self.workspace.stream,to,from.view(0,from.len_bytes())?)?;}
            if n>0 {self.prefixes[n-1].submit(&self.workspace.stream)?;}
        }
        layer.tokens=cursor+n;self.cursor=None;self.recorded=false;self.recorded_width=0;Ok(())
    }
}
impl Drop for Replay {fn drop(&mut self){let _=self.workspace.stream.synchronize();}}
