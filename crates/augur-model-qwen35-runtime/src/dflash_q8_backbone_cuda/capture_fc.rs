//! Private capture-ready FC. Target feature rows remain owned by the caller
//! until this owner drains; completed output/scratch move into the draft owner.
use super::*;
use augur_backend::{cuda_dflash_context_rows::CudaDflashContextRowsV1 as Rows, CudaStreamSubmissionReceipt, OwnedCudaEvent};

#[derive(Clone, Copy, Debug)]
struct Extent { start:usize, rows:usize, ready:usize, identity:[u8;32], generation:u64 }
impl Extent {
    fn accept(&self, start:usize, rows:usize, identity:[u8;32], generation:u64) -> Result<usize> {
        if identity!=self.identity || generation!=self.generation || start!=self.start+self.ready
            || !(1..=512).contains(&rows) || self.ready.checked_add(rows).is_none_or(|n|n>self.rows) {
            return Err("capture FC stale/noncontiguous/out-of-range readiness".into());
        }
        Ok(self.ready+rows)
    }
}

pub(crate) struct CaptureFc {
    stream:OwnedCudaStream,
    weights:DFlashQ8CudaWeights,
    output:Option<OwnedCudaBuffer>,
    panel:Option<OwnedCudaBuffer>,
    events:Vec<OwnedCudaEvent>,
    receipts:Vec<CudaStreamSubmissionReceipt>,
    source_owner:usize,
    features:CudaBufferView,
    extent:Extent,
    slices:Vec<usize>,
    poisoned:bool,
}

pub(crate) struct PreparedFc {
    pub(super) output:OwnedCudaBuffer,
    pub(super) panel:OwnedCudaBuffer,
    pub(super) report:serde_json::Value,
    weights:DFlashQ8CudaWeights,
    source_owner:usize,
    extent:Extent,
}
impl PreparedFc {
    pub(super) fn validate(&self, weights:&DFlashQ8CudaWeights,
        capture:&DFlashTargetFeatureCaptureReceipt, source:&OwnedCudaStream) -> Result<()> {
        if !Rc::ptr_eq(&self.weights.inner,&weights.inner)
            || self.source_owner!=source.owner_instance_key().map_err(err)?
            || self.extent.identity!=capture.source_target_model_sha256
            || self.extent.generation!=capture.request_generation
            || self.extent.start!=capture.context_start_position
            || self.extent.rows!=capture.context_len || self.extent.ready!=self.extent.rows
            || self.output.len_bytes()!=self.extent.rows*H*4 || self.panel.len_bytes()!=128*25600*4 {
            return Err("prepared FC owner/model/generation/extent mismatch".into());
        }
        Ok(())
    }
}

impl CaptureFc {
    pub(crate) fn new(weights:&DFlashQ8CudaWeights, source:&OwnedCudaStream,
        features:CudaBufferView, start:usize, identity:[u8;32], generation:u64) -> Result<Self> {
        if identity==[0;32] || generation==0 || features.len_bytes()!=2048*5*H*4
            || features.device_ordinal()!=weights.device_ordinal()
            || source.device_ordinal()!=weights.device_ordinal()
            || start.checked_add(2048).is_none_or(|n|n>262144) {
            return Err("capture FC admission extent/identity".into());
        }
        let device=weights.device_ordinal();
        let stream=OwnedCudaStream::create(device).map_err(err)?;
        let output=OwnedCudaBuffer::allocate(device,2048*H*4).map_err(err)?;
        let panel=OwnedCudaBuffer::allocate(device,128*25600*4).map_err(err)?;
        let events=(0..5).map(|_|OwnedCudaEvent::create(device).map_err(err)).collect::<Result<Vec<_>>>()?;
        Ok(Self { stream,weights:weights.clone(),output:Some(output),panel:Some(panel),events,
            receipts:Vec::with_capacity(5),source_owner:source.owner_instance_key().map_err(err)?,features,
            extent:Extent{start,rows:2048,ready:0,identity,generation},slices:Vec::with_capacity(5),poisoned:false })
    }

    pub(crate) fn submit_ready(&mut self, source:&OwnedCudaStream,
        start:usize, rows:usize, identity:[u8;32], generation:u64) -> Result<()> {
        if self.poisoned || self.source_owner!=source.owner_instance_key().map_err(err)?
            || self.slices.len()>=self.events.len() { return Err("capture FC poisoned/foreign owner".into()); }
        let next=self.extent.accept(start,rows,identity,generation)?;
        let f32_view=|v:CudaBufferView,n|CudaF32View::new(v,PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,n).map_err(err);
        let weight=self.weights.get("fc.weight").ok_or("capture FC weight absent")?;
        let weight=CudaQ8_0WeightViewV1::new(weight.view(0,weight.len_bytes()).map_err(err)?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_0_CUDA_V1,H,5*H).map_err(err)?;
        let input=f32_view(self.features.subview(self.extent.ready*5*H*4,rows*5*H*4).map_err(err)?,rows*5*H)?;
        let output=f32_view(self.output.as_ref().ok_or("capture FC output consumed")?
            .view(self.extent.ready*H*4,rows*H*4).map_err(err)?,rows*H)?;
        let panel=f32_view(self.panel.as_ref().ok_or("capture FC panel consumed")?
            .view(0,128*25600*4).map_err(err)?,128*25600)?;
        let bound=Rows::projection_history_fc_slice(&self.stream,weight,input,output,panel,rows).map_err(err)?;
        self.poisoned=true;
        let receipt=source.submission_receipt().map_err(err)?;
        let event=&self.events[self.slices.len()];
        event.record_for_submission(source,receipt).map_err(err)?;
        self.stream.wait_event(event).map_err(err)?;
        bound.submit_for_plan(&self.stream).map_err(err)?;
        self.receipts.push(receipt);self.slices.push(rows);self.extent.ready=next;self.poisoned=false;
        Ok(())
    }

    pub(crate) fn finish(mut self, source:&OwnedCudaStream) -> Result<PreparedFc> {
        if self.poisoned || self.extent.ready!=self.extent.rows
            || self.source_owner!=source.owner_instance_key().map_err(err)? {
            return Err("capture FC incomplete/foreign finish".into());
        }
        self.poisoned=true;
        self.stream.synchronize().map_err(err)?;
        for (event,&receipt) in self.events.iter().zip(&self.receipts) {
            if source.query_event_completion(event,receipt).map_err(err)?.is_none() {
                return Err("capture FC producer event/epoch not complete".into());
            }
        }
        let report=serde_json::json!({
            "schema":"qwen35-capture-ready-fc-v1","rows":self.extent.rows,
            "feature_start":self.extent.start,"generation":self.extent.generation,
            "slice_rows":self.slices,"source_event_epochs":self.receipts.iter().map(|r|r.submission_epoch()).collect::<Vec<_>>(),
            "source_event_complete":true,"consumer_stream_drained":true,
            "weight_sha256":self.weights.file_sha256(),"fc_output_bytes":2048*H*4,
            "panel_bytes":128*25600*4,"capture_extra_bytes":2048*H*4+128*25600*4,
            "ownership_transfer":"completed_FC_and_private_panel_move_into_draft_no_duplicate",
            "fc_slice_submissions":self.slices.len(),"fc_kernel_launches":80*self.slices.len(),
            "arithmetic_changed":false,"product_ready":false
        });
        Ok(PreparedFc { output:self.output.take().ok_or("FC already consumed")?,
            panel:self.panel.take().ok_or("panel already consumed")?,report,
            weights:self.weights.clone(),source_owner:self.source_owner,extent:self.extent })
    }
}
impl Drop for CaptureFc {
    fn drop(&mut self) { let _=self.stream.synchronize(); }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_capture_intersections_are_contiguous_not_padded() {
        for slices in [vec![512;4],vec![227,512,512,512,285],vec![1,511,512,512,512]] {
            let mut e=Extent{start:6429,rows:2048,ready:0,identity:[7;32],generation:4};
            for rows in slices { e.ready=e.accept(e.start+e.ready,rows,[7;32],4).unwrap(); }
            assert_eq!(e.ready,2048);assert!(e.accept(8477,1,[7;32],4).is_err());
        }
    }
    #[test]
    fn rejects_gaps_overlap_wrong_generation_and_identity() {
        let e=Extent{start:6429,rows:2048,ready:227,identity:[7;32],generation:4};
        for (start,rows,id,generation) in [(6655,512,[7;32],4),(6657,512,[7;32],4),
            (6656,0,[7;32],4),(6656,513,[7;32],4),(6656,512,[8;32],4),(6656,512,[7;32],5)] {
            assert!(e.accept(start,rows,id,generation).is_err());
        }
    }
}
