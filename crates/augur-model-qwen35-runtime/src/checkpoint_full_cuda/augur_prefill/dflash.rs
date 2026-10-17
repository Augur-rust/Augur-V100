//! Retained B8 position rebinding and physical rejected-tail restoration.
use super::*;
use augur_backend::CudaBufferView;
const MAX_B:usize=8;
const ROW:usize=1088;
pub(crate) struct Checkpoint {stream:Rc<OwnedCudaStream>,cache:CacheLease,backup:[Buffer;2],cursor:Option<usize>,width:usize}
impl Checkpoint {
    pub(crate) fn new(layer:&CheckpointFullCudaLayer)->Result<Self> {
        Ok(Self{stream:Rc::clone(&layer.stream),cache:CacheLease::new([Rc::clone(&layer.buffers["key_cache"]),Rc::clone(&layer.buffers["value_cache"])])?,
            backup:[empty(&layer.stream,MAX_B*ROW/4)?,empty(&layer.stream,MAX_B*ROW/4)?],cursor:None,width:0})
    }
    pub(crate) fn park(&mut self)->Result<()> {
        if self.cursor.is_some(){return Err("cannot park pending KV checkpoint".into());}
        self.cache.park();Ok(())
    }
    pub(crate) fn resume(&mut self,layer:&CheckpointFullCudaLayer)->Result<()> {
        if self.cursor.is_some() || !Rc::ptr_eq(&self.stream,&layer.stream){return Err("stale KV reattach".into());}
        self.cache.resume(layer)
    }
    pub(crate) fn begin(&mut self,layer:&CheckpointFullCudaLayer,cursor:usize)->Result<()> {
        self.begin_width(layer,cursor,MAX_B)
    }
    pub(crate) fn begin_width(&mut self,layer:&CheckpointFullCudaLayer,cursor:usize,width:usize)->Result<()> {
        self.begin_width_no_copy(layer,cursor,width)?;
        for (to,from) in self.backup.iter().zip(self.cache.get()?) {
            // SAFETY: fixed row-major rotated Q8 layout; owners retained, same stream.
            unsafe{to.copy_from_device_async_unchecked(&self.stream,0,from.view(cursor*ROW,width*ROW)?)?;}
        }
        Ok(())
    }
    /// Opens the rollback transaction without submitting the two backup copies.
    /// The target executor aggregates these ranges with all other layer backups.
    pub(crate) fn begin_width_no_copy(&mut self,layer:&CheckpointFullCudaLayer,cursor:usize,width:usize)->Result<()> {
        if !(1..=MAX_B).contains(&width) || self.cursor.is_some() || layer.poisoned || layer.tokens!=cursor || cursor.checked_add(width).is_none_or(|n|n>layer.capacity)
            || !self.cache.matches(layer) {return Err("stale KV checkpoint".into());}
        self.cursor=Some(cursor);self.width=width;Ok(())
    }
    pub(crate) fn backup_copies(&self,cursor:usize,width:usize)->Result<[( &OwnedCudaBuffer, usize, CudaBufferView); 2]> {
        let caches=self.cache.get()?;
        Ok([
            (&self.backup[0],0,caches[0].view(cursor*ROW,width*ROW)?),
            (&self.backup[1],0,caches[1].view(cursor*ROW,width*ROW)?),
        ])
    }
    #[cfg(test)]
    pub(crate) fn audit_zero_restored(&self,cursor:usize)->Result<()> {
        for (cache,backup) in self.cache.get()?.iter().zip(&self.backup) {
            let mut a=vec![0;MAX_B*ROW];let mut b=vec![0;MAX_B*ROW];
            cache.view(cursor*ROW,MAX_B*ROW)?.copy_to_host_sync(&self.stream,&mut a)?;backup.copy_to_host_sync(&self.stream,&mut b)?;
            if a!=b {return Err("KV rejected tail not physically restored".into());}
        }Ok(())
    }
    pub(crate) fn commit(&mut self,layer:&mut CheckpointFullCudaLayer,n:usize)->Result<()> {
        let width=if self.width==0 {MAX_B} else {self.width};
        self.commit_width(layer,width,n)
    }
    pub(crate) fn commit_width(&mut self,layer:&mut CheckpointFullCudaLayer,width:usize,n:usize)->Result<()> {
        let cursor=self.cursor.ok_or("missing KV checkpoint")?;
        if !(1..=MAX_B).contains(&width) || self.width!=width || n>width || layer.poisoned || layer.tokens!=cursor+width {return Err("invalid KV prefix commit".into());}
        if n<width {for (to,from) in self.cache.get()?.iter().zip(&self.backup) {
            // SAFETY: restore only rejected physical rows, preserving accepted bytes.
            unsafe{to.copy_from_device_async_unchecked(&self.stream,(cursor+n)*ROW,from.view(n*ROW,(width-n)*ROW)?)?;}
        }}
        layer.tokens=cursor+n;self.cursor=None;self.width=0;Ok(())
    }
}
impl Drop for Checkpoint {fn drop(&mut self){let _=self.stream.synchronize();}}
impl Program {
    pub(crate) fn park_cache(&mut self)->Result<()> {
        if !(1..=MAX_B).contains(&self.tokens) || self.tiles.len()!=1 {return Err("only retained cache may park".into());}
        self.cache.park();Ok(())
    }
    pub(crate) fn resume_cache(&mut self,layer:&CheckpointFullCudaLayer)->Result<()> {
        if !(1..=MAX_B).contains(&self.tokens) || self.tiles.len()!=1 || !Rc::ptr_eq(&self.workspace.common.stream,&layer.stream) {
            return Err("foreign cache resume".into());
        }
        self.cache.resume(layer)
    }
    pub(crate) fn park_b8_cache(&mut self)->Result<()> { self.park_cache() }
    pub(crate) fn resume_b8_cache(&mut self,layer:&CheckpointFullCudaLayer)->Result<()> { self.resume_cache(layer) }

    pub(crate) fn rebind_position(&mut self,layer:&CheckpointFullCudaLayer,position:usize,width:usize)->Result<()> {
        if !(1..=MAX_B).contains(&width) || self.tokens!=width || self.tiles.len()!=1 || layer.poisoned || layer.tokens!=position
            || position.checked_add(width).is_none_or(|n|n>layer.capacity)
            || !self.cache.matches(layer) {return Err("invalid retained position".into());}
        let c=&self.workspace.common;
        let av=|name:&str,cols|c.full_planes[name].view(0,width*cols);
        self.tiles[0].heads=CudaQwen35BatchedHeadNormRopeArgsV1::bind(&c.stream,CudaQwen35BatchedHeadNormRopeViewsV1 {
            q_gate:av("q_full",12288)?,q_weight:view(&layer.buffers["q_norm_weight"])?,query:av("query",6144)?,
            key_input:av("k_raw",1024)?,k_weight:view(&layer.buffers["k_norm_weight"])?,key:av("key_work",1024)?,
            rope_cos:subview(&layer.buffers["rope_cos"],position*32,width*32)?,
            rope_sin:subview(&layer.buffers["rope_sin"],position*32,width*32)?,gate:av("q_gate",6144)?,
            value_input:av("value",1024)?,value:av("value_work",1024)?,
        },width,layer.epsilon)?;
        self.tiles[0].position=position;self.position=position;Ok(())
    }
    pub(crate) fn rebind_b8_position(&mut self,layer:&CheckpointFullCudaLayer,position:usize)->Result<()> {
        self.rebind_position(layer,position,8)
    }
}

#[cfg(test)]
impl CheckpointFullCudaLayer {
    pub(crate) fn audit_exclusive_cache_lease(&self)->Result<()> {
        for name in ["key_cache","value_cache"] {
            if Rc::strong_count(&self.buffers[name])!=1 || Rc::weak_count(&self.buffers[name])!=0 {
                return Err("park retained a strong/weak physical KV lease".into());
            }
        }
        Ok(())
    }
}
