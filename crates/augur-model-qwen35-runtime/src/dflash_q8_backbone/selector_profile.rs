//! The sole admitted embedded gate profile; no generic F32-matrix exception.
use super::{err,DFlashBlockProducerError,GgufManifest,GgufValue};
use sha2::{Digest,Sha256};
pub const GATE_SHA256:&str="0bcb694a4e84aa304d888829742fdef92941771ac5e97917ca5dc4388f6821c7";
pub const GATE_PROFILE:&str="f32_gate045_v1";
const PREFIX:&str="quasar.selector_gate.";
pub(super) fn admit(m:&GgufManifest)->Result<bool,DFlashBlockProducerError> {
    let tensor=m.tensor("selector_hidden.weight").ok_or_else(||err("missing selector gate"))?;
    let fields=["profile","sha256","folded_factor","runtime_scale","scope"];
    let supplied=fields.iter().filter(|k|m.metadata(&format!("{PREFIX}{k}")).is_some()).count();
    if tensor.ggml_type==8 && supplied==0 {return Ok(false);}
    let text=|key:&str|m.metadata(&format!("{PREFIX}{key}")).and_then(GgufValue::as_str);
    let number=|key:&str|m.metadata(&format!("{PREFIX}{key}")).and_then(GgufValue::as_f64);
    if tensor.ggml_type!=0 || tensor.dimensions!=[5120,256] || tensor.byte_len!=256*5120*4
        || supplied!=fields.len() || text("profile")!=Some(GATE_PROFILE) || text("sha256")!=Some(GATE_SHA256)
        || text("scope")!=Some("global_gate045_v1") || number("folded_factor")!=Some(f64::from(0.45f32))
        || number("runtime_scale")!=Some(1.0) {
        return Err(err("embedded selector requires authenticated F32 gate0.45, global scope and runtime scale1"));
    }
    Ok(true)
}
pub(super) fn validate_bytes(bytes:&[u8])->Result<(),DFlashBlockProducerError> {
    if bytes.len()!=256*5120*4 || format!("{:x}",Sha256::digest(bytes))!=GATE_SHA256
        || !bytes.chunks_exact(4).all(|b|f32::from_le_bytes(b.try_into().unwrap()).is_finite()) {
        return Err(err("embedded gate payload does not match admitted original FP32 gate0.45"));
    }
    Ok(())
}
/// CPU audit oracle, not the CUDA serving kernel.
pub(super) fn matvec_f64(bytes:&[u8],rows:usize,cols:usize,input:&[f32],batch:usize)->Result<Vec<f32>,DFlashBlockProducerError> {
    if rows.checked_mul(cols).and_then(|n|n.checked_mul(4))!=Some(bytes.len())
        || cols.checked_mul(batch)!=Some(input.len()) || rows==0 || cols==0 || batch==0 {
        return Err(err("F32 gate oracle geometry"));
    }
    let mut out=vec![0.;rows.checked_mul(batch).ok_or_else(||err("F32 gate extent"))?];
    for b in 0..batch {for r in 0..rows {
        let mut sum=0f64;
        for c in 0..cols {let start=(r*cols+c)*4;sum+=f64::from(f32::from_le_bytes(bytes[start..start+4].try_into().unwrap()))*f64::from(input[b*cols+c]);}
        out[b*rows+r]=sum as f32;
    }}
    Ok(out)
}
#[cfg(test)]mod tests {
    use super::*;
    #[test]fn payload_cannot_be_promoted_from_an_arbitrary_float_or_q8_gate() {
        assert!(validate_bytes(&[0;34]).is_err());
        assert!(validate_bytes(&vec![0;256*5120*4]).is_err());
    }
    #[test]fn f32_oracle_batches_without_changing_matrix_orientation() {
        let w:Vec<u8>=[1f32,2.,3.,4.].into_iter().flat_map(f32::to_le_bytes).collect();
        assert_eq!(matvec_f64(&w,2,2,&[1.,0.,0.,1.],2).unwrap(),[1.,3.,2.,4.]);
        assert!(matvec_f64(&w,2,2,&[1.],2).is_err());
    }
    fn fixture(kind:u32, fields:&[(&str,Option<&str>,f32)], dims:[u64;2])->GgufManifest {
        use std::{fs,io::Write,sync::atomic::{AtomicU64,Ordering}};
        static ID:AtomicU64=AtomicU64::new(0);
        fn text(out:&mut Vec<u8>,s:&str){out.extend_from_slice(&(s.len() as u64).to_le_bytes());out.extend_from_slice(s.as_bytes());}
        let mut out=b"GGUF".to_vec();out.extend_from_slice(&3u32.to_le_bytes());out.extend_from_slice(&1u64.to_le_bytes());out.extend_from_slice(&(fields.len() as u64).to_le_bytes());
        for (key,value,number) in fields {
            text(&mut out,&format!("{PREFIX}{key}"));out.extend_from_slice(&(if value.is_some(){8u32}else{6u32}).to_le_bytes());
            if let Some(value)=value {text(&mut out,value);}else{out.extend_from_slice(&number.to_le_bytes());}
        }
        text(&mut out,"selector_hidden.weight");out.extend_from_slice(&2u32.to_le_bytes());
        for d in dims {out.extend_from_slice(&d.to_le_bytes());}
        out.extend_from_slice(&kind.to_le_bytes());out.extend_from_slice(&0u64.to_le_bytes());
        out.resize(out.len().next_multiple_of(32),0);
        let payload=dims[0]*dims[1]*if kind==0{4}else{34}/if kind==0{1}else{32};
        let path=std::env::temp_dir().join(format!("augur-embedded-gate-{}-{}.gguf",std::process::id(),ID.fetch_add(1,Ordering::Relaxed)));
        let mut file=fs::OpenOptions::new().write(true).create_new(true).open(&path).unwrap();file.write_all(&out).unwrap();file.set_len(out.len() as u64+payload).unwrap();drop(file);
        let manifest=GgufManifest::inspect(&path,augur_model_qwen35::GgufAuditOptions {hash_file:false,..Default::default()}).unwrap();fs::remove_file(path).unwrap();manifest
    }
    #[test]fn metadata_cannot_relabel_q8_or_unscaled_f32_as_the_calibrated_gate() {
        let fields=[("profile",Some(GATE_PROFILE),0.),("sha256",Some(GATE_SHA256),0.),
            ("folded_factor",None,0.45),("runtime_scale",None,1.),("scope",Some("global_gate045_v1"),0.)];
        assert!(admit(&fixture(0,&fields,[5120,256])).unwrap());
        assert!(!admit(&fixture(8,&[],[5120,256])).unwrap());
        assert!(admit(&fixture(8,&fields,[5120,256])).is_err());
        assert!(admit(&fixture(0,&[],[5120,256])).is_err());
        assert!(admit(&fixture(0,&fields[..4],[5120,256])).is_err());
        assert!(admit(&fixture(0,&fields,[256,5120])).is_err());
        for index in [0,1,2,3,4] {
            let mut bad=fields;
            if bad[index].1.is_some() {bad[index].1=Some("wrong");}else{bad[index].2=0.5;}
            assert!(admit(&fixture(0,&bad,[5120,256])).is_err());
        }
    }

}
