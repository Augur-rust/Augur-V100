//! Experimental gate bytes are explicitly supplied and SHA-bound, never inferred.
use super::{err,Result};
use sha2::{Digest,Sha256};
pub(super) const KEY:&str="__original_selector_gate_f32";
const BYTES:usize=256*5120*4;
fn validate(bytes:&[u8],sha:&str)->Result<()> {
    if bytes.len()!=BYTES || sha.len()!=64 || !sha.bytes().all(|v|v.is_ascii_hexdigit())
        || format!("{:x}",Sha256::digest(bytes))!=sha {
        return Err("selector gate F32 shape or SHA256 mismatch".into());
    }
    if !bytes.chunks_exact(4).all(|v|f32::from_le_bytes(v.try_into().unwrap()).is_finite()) {
        return Err("selector gate contains non-finite weights".into());
    }
    Ok(())
}
fn accumulation(value:Option<&str>,has_gate:bool)->Result<bool> {
    match (value,has_gate) {
        (None,_) | (Some("fp32"),true)=>Ok(false),
        (Some("fp64"),true)=>Ok(true),
        _=>Err("selector gate accumulation requires original gate and fp32 or fp64".into()),
    }
}
pub(super) fn load(embedded:Option<&[u8]>)->Result<Option<(Vec<u8>,String,bool)>> {
    let path=std::env::var_os("AUGUR_DFLASH_SELECTOR_GATE_F32");
    let sha=std::env::var_os("AUGUR_DFLASH_SELECTOR_GATE_F32_SHA256");
    let mode=std::env::var("AUGUR_DFLASH_SELECTOR_GATE_ACCUMULATION").ok();
    if let Some(bytes)=embedded {
        if path.is_some() || sha.is_some() || mode.as_deref().is_some_and(|v|v!="fp32") {
            return Err("embedded gate cannot be overridden, double-calibrated or use non-FP32 accumulation".into());
        }
        if !cfg!(feature="qwen35-prefill-cublas") {return Err("embedded F32 gate requires prefill cuBLAS support".into());}
        let digest=crate::dflash_q8_backbone::GATE_SHA256;
        validate(bytes,digest)?;
        return Ok(Some((bytes.to_vec(),digest.into(),false)));
    }
    let fp64=accumulation(mode.as_deref(),path.is_some()&&sha.is_some())?;
    match (path,sha) {
        (None,None)=>Ok(None),
        (Some(path),Some(sha))=>{
            if !cfg!(feature="qwen35-prefill-cublas") {return Err("F32 gate requires prefill cuBLAS support".into());}
            let sha=sha.into_string().map_err(|_|"non-Unicode selector gate SHA")?;
            if std::fs::metadata(&path).map_err(err)?.len()!=BYTES as u64 {return Err("selector gate F32 file size mismatch".into());}
            let bytes=std::fs::read(path).map_err(err)?;validate(&bytes,&sha)?;Ok(Some((bytes,sha,fp64)))
        }
        _=>Err("selector gate requires both path and SHA256".into()),
    }
}
#[cfg(test)]mod tests {
    use super::*;
    #[test]fn accumulation_requires_explicit_gate_and_known_mode() {
        assert!(!accumulation(None,false).unwrap());assert!(!accumulation(Some("fp32"),true).unwrap());
        assert!(accumulation(Some("fp64"),true).unwrap());assert!(accumulation(Some("fp64"),false).is_err());
        assert!(accumulation(Some("mixed"),true).is_err());
    }
    #[test]fn gate_bytes_reject_corruption_shape_and_nonfinite() {
        let mut b=vec![0u8;BYTES];let sha=format!("{:x}",Sha256::digest(&b));assert!(validate(&b,&sha).is_ok());
        assert!(validate(&b[..BYTES-4],&sha).is_err());assert!(validate(&b,&"0".repeat(64)).is_err());
        b[..4].copy_from_slice(&f32::NAN.to_le_bytes());let sha=format!("{:x}",Sha256::digest(&b));assert!(validate(&b,&sha).is_err());
    }
}

pub(super) const CODEBOOK_KEY:&str="__original_selector_codebooks_bf16";
pub(super) const CODEBOOK_BYTES:usize=248320*256*2;
pub(super) fn load_codebooks(has_original_gate:bool)->Result<Option<(Vec<u8>,String)>> {
    let path=std::env::var_os("AUGUR_DFLASH_SELECTOR_CODEBOOKS_BF16");
    let sha=std::env::var("AUGUR_DFLASH_SELECTOR_CODEBOOKS_BF16_SHA256").ok();
    match (path,sha) {
        (None,None)=>Ok(None),
        (Some(path),Some(sha)) if has_original_gate=>{
            if std::fs::metadata(&path).map_err(err)?.len()!=(2*CODEBOOK_BYTES) as u64 {return Err("original codebook size".into());}
            let bytes=std::fs::read(path).map_err(err)?;
            if sha.len()!=64 || format!("{:x}",Sha256::digest(&bytes))!=sha {return Err("original codebook SHA".into());}
            if !bytes.chunks_exact(2).all(|b|u16::from_le_bytes(b.try_into().unwrap())&0x7f80!=0x7f80) {return Err("nonfinite original codebook".into());}
            Ok(Some((bytes,sha)))
        }
        _=>Err("original codebooks require path, SHA and authenticated original gate".into()),
    }
}
