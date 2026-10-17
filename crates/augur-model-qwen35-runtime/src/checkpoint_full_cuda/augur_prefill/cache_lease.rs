//! Parkable strong KV leases. A parked descriptor is inert, not executable.
//! CudaBufferView equality includes allocation owner generation, so an allocator
//! reusing an address cannot re-admit a foreign cache. No Weak reference remains
//! while parked: the original Rc::get_mut reset guard must still succeed.
use super::*;
use augur_backend::CudaBufferView;
pub(super) struct CacheLease {
    identity: [CudaBufferView; 2],
    owners: Option<[Buffer; 2]>,
}
impl CacheLease {
    fn views(owners: &[Buffer; 2]) -> Result<[CudaBufferView; 2]> {
        Ok([owners[0].view(0, owners[0].len_bytes())?, owners[1].view(0, owners[1].len_bytes())?])
    }
    pub(super) fn new(owners: [Buffer; 2]) -> Result<Self> {
        Ok(Self { identity: Self::views(&owners)?, owners: Some(owners) })
    }
    pub(super) fn get(&self) -> Result<&[Buffer; 2]> {
        self.owners.as_ref().ok_or_else(|| "parked full-attention cache is not executable".into())
    }
    pub(super) fn matches(&self, layer: &CheckpointFullCudaLayer) -> bool {
        self.owners.as_ref().is_some_and(|owners|
            Rc::ptr_eq(&owners[0], &layer.buffers["key_cache"])
            && Rc::ptr_eq(&owners[1], &layer.buffers["value_cache"]))
    }
    pub(super) fn park(&mut self) { self.owners = None; }
    pub(super) fn resume(&mut self, layer: &CheckpointFullCudaLayer) -> Result<()> {
        let owners = [Rc::clone(&layer.buffers["key_cache"]), Rc::clone(&layer.buffers["value_cache"])];
        if Self::views(&owners)? != self.identity {
            return Err("foreign allocation generation on B8 KV reattach".into());
        }
        self.owners = Some(owners);
        Ok(())
    }
}
