//! Request-owned image embeddings. Replace only authenticated placeholder rows
//! after the text gather, before layer0; cache slots remain physical token slots.
use super::*;
#[derive(Debug, PartialEq, Eq)]
struct Copy { dst_row:usize, src_row:usize, rows:usize }
fn copies(indices:&[usize], first:usize, rows:usize)->Result<Vec<Copy>> {
    let end=first.checked_add(rows).ok_or("image chunk extent overflow")?;
    if rows==0 || indices.windows(2).any(|x|x[0]>=x[1]) {return Err("invalid image row order/chunk".into());}
    let mut out:Vec<Copy>=Vec::new();
    for (src,&position) in indices.iter().enumerate().filter(|(_,p)| **p>=first&&**p<end) {
        let dst=position-first;
        if let Some(last)=out.last_mut() {
            if last.dst_row+last.rows==dst && last.src_row+last.rows==src {last.rows+=1;continue;}
        }
        out.push(Copy {dst_row:dst,src_row:src,rows:1});
    }
    Ok(out)
}
pub(in crate::checkpoint_target_cuda) struct DeviceRows {
    stream:Rc<OwnedCudaStream>, source:OwnedCudaBuffer, indices:Vec<usize>,
    token_sha256:String, image_identity:serde_json::Value,
}
impl DeviceRows {
    pub(in crate::checkpoint_target_cuda) fn upload(stream:Rc<OwnedCudaStream>,tokens:&[u32],indices:Vec<usize>,
        embeddings:&[f32],identity:serde_json::Value)->Result<Self> {
        if indices.is_empty() || indices.len()>crate::augur_session::images::MAX_IMAGE_TOKENS || embeddings.len()!=indices.len()*H
            || indices.windows(2).any(|x|x[0]>=x[1])
            || indices.iter().any(|&i| tokens.get(i)!=Some(&248056))
            || tokens.iter().filter(|&&x|x==248056).count()!=indices.len()
            || embeddings.iter().any(|x|!x.is_finite()) {
            return Err("image placeholder/embedding/finite contract mismatch".into());
        }
        let source=OwnedCudaBuffer::allocate(stream.device_ordinal(),embeddings.len()*4)?;
        let bytes:Vec<u8>=embeddings.iter().flat_map(|x|x.to_le_bytes()).collect();
        source.copy_from_host_sync(&stream,&bytes)?;
        Ok(Self {stream,source,indices,token_sha256:ids_hash(tokens),image_identity:identity})
    }
    pub(in crate::checkpoint_target_cuda) fn validate(&self,stream:&Rc<OwnedCudaStream>,tokens:&[u32])->Result<()> {
        if !Rc::ptr_eq(stream,&self.stream)||self.token_sha256!=ids_hash(tokens) {return Err("foreign image input owner/tokens".into());}
        Ok(())
    }
    pub(in crate::checkpoint_target_cuda) fn submit(&self,destination:&OwnedCudaBuffer,first:usize,rows:usize)->Result<()> {
        for c in copies(&self.indices,first,rows)? {
            // SAFETY: sorted validated row indices, disjoint request-owned source
            // and phase-owned destination; both outlive the phase stream drain.
            unsafe {destination.copy_from_device_async_unchecked(&self.stream,c.dst_row*H*4,
                self.source.view(c.src_row*H*4,c.rows*H*4)?)?;}
        }
        Ok(())
    }
    pub(in crate::checkpoint_target_cuda) fn receipt(&self)->serde_json::Value {
        serde_json::json!({"image_embedding_rows":self.indices.len(),"device_embedding_bytes":self.source.len_bytes(),
            "token_sha256":self.token_sha256,"identity":self.image_identity,"image_embeddings_consumed":true})
    }
}
impl Drop for DeviceRows {fn drop(&mut self){let _=self.stream.synchronize();}}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn image_scatter_crosses_chunks_and_preserves_intervening_text() {
        let indices=vec![1,2,3,6,7,8,9,4351,4352,4353];
        assert_eq!(copies(&indices,2,6).unwrap(),vec![Copy{dst_row:0,src_row:1,rows:2},Copy{dst_row:4,src_row:3,rows:2}]);
        assert_eq!(copies(&indices,4352,2).unwrap(),vec![Copy{dst_row:0,src_row:8,rows:2}]);
        let mut actual=Vec::new();
        for first in (0..4360).step_by(7) {
            for c in copies(&indices,first,7).unwrap() {
                for i in 0..c.rows {actual.push((first+c.dst_row+i,c.src_row+i));}
            }
        }
        assert_eq!(actual,indices.iter().enumerate().map(|(i,&p)|(p,i)).collect::<Vec<_>>());
        assert!(copies(&[2,2],0,4).is_err());assert!(copies(&[2,1],0,4).is_err());
        assert!(copies(&[],usize::MAX,1).is_err());assert!(copies(&[],0,0).is_err());
    }
}
