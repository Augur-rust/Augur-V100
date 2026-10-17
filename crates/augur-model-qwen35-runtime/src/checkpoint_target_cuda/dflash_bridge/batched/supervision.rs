//! Diagnostic-only target logits; canonical labels stop at the first rejection.
use super::*;
use augur_backend::CudaQwen35VocabTopkArgsV1;
const K:usize=16;
const ROWS:usize=7;
const CANDIDATE_BYTES:usize=ROWS*K*16;
const IDS_BYTES:usize=ROWS*2*4;
const TOTAL_BYTES:usize=CANDIDATE_BYTES+2*IDS_BYTES;

fn addresses(ids:&[u32],labels:&[Option<u32>])->Result<Vec<(usize,usize)>> {
    if ids.len()!=8*K || labels.len()!=8 || ids.iter().any(|&id|id as usize>=V)
        || labels.iter().flatten().any(|&id|id as usize>=V) {
        return Err("selector supervision shape/token bounds".into());
    }
    let mut absent=false;let mut result=Vec::new();
    for position in 1..8 {
        if labels[position].is_none(){absent=true;continue;}
        if absent{return Err("selector label beyond first unknown prefix".into());}
        for candidate in 0..K {
            let id=ids[position*K+candidate] as usize;
            // One aligned four-float cell per candidate; discard extra lanes.
            result.push((((position-1)*K+candidate)*16,((position-1)*V+(id&!3))*4));
        }
    }
    Ok(result)
}

pub(crate) struct Workspace {
    stream:Rc<OwnedCudaStream>,source:Buffer,scratch:OwnedCudaBuffer,
    top2:CudaQwen35VocabTopkArgsV1,
}
impl Workspace {
    pub(super) fn new(executor:&Executor)->Result<Self> {
        if executor.width!=8 || executor.pending.is_some() || executor.parked {
            return Err("supervision requires quiescent B8 executor".into());
        }
        let stream=Rc::clone(&executor.stream);let source=Rc::clone(&executor.logits);
        let scratch=OwnedCudaBuffer::allocate(stream.device_ordinal(),TOTAL_BYTES)?;
        scratch.copy_from_host_sync(&stream,&vec![0;TOTAL_BYTES])?;
        let logits=CudaF32View::new(source.view(0,ROWS*V*4)?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,ROWS*V)?;
        let top2=CudaQwen35VocabTopkArgsV1::bind(&stream,logits,
            scratch.view(CANDIDATE_BYTES,IDS_BYTES)?,
            scratch.view(CANDIDATE_BYTES+IDS_BYTES,IDS_BYTES)?,ROWS,2)?;
        Ok(Self{stream,source,scratch,top2})
    }
    pub(super) fn matches_executor(&self,executor:&Executor)->bool {
        Rc::ptr_eq(&self.stream,&executor.stream) && Rc::ptr_eq(&self.source,&executor.logits) && executor.width==8
    }
    pub(crate) fn read(&self,ids:&[u32],labels:&[Option<u32>])->Result<Value> {
        let layout=addresses(ids,labels)?;
        if layout.is_empty(){return Ok(json!({"rows":vec![Value::Null;8],"D2H_bytes":0}));}
        let copies=layout.iter().map(|&(destination,source)|
            Ok((&self.scratch,destination,self.source.view(source,16)?)))
            .collect::<Result<Vec<_>>>()?;
        // SAFETY: retained source/scratch owners, disjoint aligned destinations,
        // same stream; readback and Drop drain before either owner is freed.
        unsafe{OwnedCudaBuffer::copy_independent_batch_async_unchecked(&self.stream,&copies)?;}
        self.top2.submit(&self.stream)?;
        let mut raw=vec![0;TOTAL_BYTES];self.scratch.copy_to_host_sync(&self.stream,&mut raw)?;
        let float=|offset:usize|f32::from_le_bytes(raw[offset..offset+4].try_into().unwrap());
        let token=|offset:usize|u32::from_le_bytes(raw[offset..offset+4].try_into().unwrap());
        let mut rows=vec![Value::Null;8];
        for position in 1..8 {
            let Some(label)=labels[position] else{continue;};
            let scores:Vec<_>=(0..K).map(|k|float(((position-1)*K+k)*16+(ids[position*K+k] as usize%4)*4)).collect();
            let top_ids:Vec<_>=(0..2).map(|k|token(CANDIDATE_BYTES+((position-1)*2+k)*4)).collect();
            let top_scores:Vec<_>=(0..2).map(|k|float(CANDIDATE_BYTES+IDS_BYTES+((position-1)*2+k)*4)).collect();
            if top_ids[0]!=label || top_ids.iter().any(|&id|id as usize>=V)
                || scores.iter().chain(&top_scores).any(|v|!v.is_finite()) || top_scores[0]<top_scores[1] {
                return Err("teacher logits/label row mismatch or nonfinite result".into());
            }
            rows[position]=json!({"position":position,"target_logit_row":position-1,
                "label":label,"candidate_logits":scores,"top2_ids":top_ids,
                "top2_logits":top_scores,"top1_margin":top_scores[0]-top_scores[1]});
        }
        Ok(json!({"rows":rows,"D2H_bytes":TOTAL_BYTES,
            "source":"same_round_actual_B8_verify_logits","post_first_rejection_masked":true,
            "label_only_not_draft_input":true,"candidate_set":"unchanged_draft_top16"}))
    }
}
impl Drop for Workspace{fn drop(&mut self){let _=self.stream.synchronize();}}
impl Executor {
    pub(crate) fn selector_supervision_workspace(&self)->Result<Workspace>{Workspace::new(self)}
}
#[test]
fn supervision_candidates_use_predecessor_logit_row_and_aligned_cells(){
    let ids:Vec<u32>=(0..128).collect();let labels=vec![Some(0),Some(3),Some(7),None,None,None,None,None];
    let layout=addresses(&ids,&labels).unwrap();assert_eq!(layout.len(),32);
    for position in 1..=2{for candidate in 0..K{
        let(dst,src)=layout[(position-1)*K+candidate];let id=ids[position*K+candidate] as usize;
        assert_eq!(src/4+id%4,(position-1)*V+id);assert_eq!((src%16,dst%16),(0,0));
        assert!(src+16<=ROWS*V*4 && dst+16<=CANDIDATE_BYTES);
    }}
}
#[test]
fn supervision_unknown_suffix_and_invalid_tokens_are_rejected(){
    let mut ids:Vec<u32>=(0..128).collect();let mut labels=vec![Some(0),Some(1),None,None,None,None,None,None];
    labels[3]=Some(9);assert!(addresses(&ids,&labels).is_err());
    labels[3]=None;ids[24]=V as u32;assert!(addresses(&ids,&labels).is_err());
    ids[24]=24;labels[1]=None;assert!(addresses(&ids,&labels).unwrap().is_empty());
}
