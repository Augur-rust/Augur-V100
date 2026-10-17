//! Test/benchmark-only native prefix cache. Uses the existing bounded transfer
//! store with an explicit target-only identity; no placeholder draft binding.
use super::*;
use crate::session_checkpoint::{self as transfer, CaptureSource, RestoreTarget, Component, TransferPlan, Manifest};
use crate::session_checkpoint::filesystem::{LocalCheckpointWriter, LocalCheckpointReader};
use std::io;
struct Span<'a> { name: String, buffer: &'a OwnedCudaBuffer, bytes: usize }
struct Io<'a> { stream: &'a OwnedCudaStream, spans: Vec<Span<'a>>, metadata: Vec<u8>, plan: TransferPlan }
impl Drop for Io<'_> { fn drop(&mut self) { let _ = self.stream.synchronize(); } }
fn err(e: impl std::fmt::Display) -> io::Error { io::Error::other(e.to_string()) }
impl Io<'_> {
    fn extent(&self, c: usize, off: u64, n: usize) -> io::Result<usize> {
        let start = usize::try_from(off).map_err(err)?;
        let component = self.plan.components.get(c).ok_or_else(|| err("cache component"))?;
        if n == 0 || start.checked_add(n).is_none_or(|end| end as u64 > component.bytes) { return Err(err("cache extent")); }
        Ok(start)
    }
    fn state_digest(&mut self) -> io::Result<String> {
        let mut hash = Sha256::new();
        for c in 0..self.plan.components.len() {
            let mut off = 0; let bytes = self.plan.components[c].bytes;
            hash.update(self.plan.components[c].name.as_bytes());hash.update(bytes.to_le_bytes());
            while off < bytes {
                let mut chunk = vec![0; (bytes-off).min(transfer::MAX_CHUNK_BYTES as u64) as usize];
                CaptureSource::read_exact(self,c,off,&mut chunk)?;hash.update(&chunk);off += chunk.len() as u64;
            }
        }
        Ok(format!("{:x}",hash.finalize()))
    }
}
impl CaptureSource for Io<'_> {
    fn begin(&mut self) -> io::Result<TransferPlan> { self.stream.synchronize().map_err(err)?; Ok(self.plan.clone()) }
    fn read_exact(&mut self,c:usize,off:u64,out:&mut[u8]) -> io::Result<()> {
        let start=self.extent(c,off,out.len())?;
        if c==0 {out.copy_from_slice(&self.metadata[start..start+out.len()]);}
        else {self.spans[c-1].buffer.view(start,out.len()).map_err(err)?.copy_to_host_sync(self.stream,out).map_err(err)?;}
        Ok(())
    }
    fn finish(&mut self)->io::Result<TransferPlan>{self.stream.synchronize().map_err(err)?;Ok(self.plan.clone())}
    fn abort(&mut self)->io::Result<()>{self.stream.synchronize().map_err(err)}
}
impl RestoreTarget for Io<'_> {
    fn begin(&mut self,p:&TransferPlan)->io::Result<()>{if *p!=self.plan{return Err(err("cache independently derived identity"));}self.stream.synchronize().map_err(err)}
    fn write_exact(&mut self,c:usize,off:u64,bytes:&[u8])->io::Result<()> {
        let start=self.extent(c,off,bytes.len())?;
        if c==0 {if self.metadata[start..start+bytes.len()]!=*bytes{return Err(err("cache descriptor mismatch"));}}
        else {self.spans[c-1].buffer.copy_range_from_host_sync(self.stream,start,bytes).map_err(err)?;}Ok(())
    }
    fn finish_unpublished(&mut self)->io::Result<()>{self.stream.synchronize().map_err(err)}
    fn abort(&mut self)->io::Result<()>{self.stream.synchronize().map_err(err)}
}
fn descriptor(t:&CheckpointCudaTarget,ids:&[u32])->Value {
    assert!(ids.len()>0 && ids.len()<262144 && ids.iter().all(|&id| (id as usize)<t.logits.len_bytes()/4));
    json!({"schema":"quasar-offline-native-prefix-v1",
        "producer_sha256":hash(&std::fs::read(std::env::current_exe().unwrap()).unwrap()),
        "manifest_sha256":std::env::var("AUGUR_MANIFEST_SHA256").unwrap(),
        "attention_mode":t.attention_mode().label(),"kv_identity":t.attention_mode().kv_identity(),
        "capacity":262144,"numerical_profile":"quasar-native-quasar-qpn-b4352-exact-norm-fused-input-batched-ffn8-v1",
        "dflash_loaded":false,"feature_ring":false,"logits_current":true,
        "token_ids_sha256":hash(&ids.iter().flat_map(|id|id.to_le_bytes()).collect::<Vec<_>>()),"token_ids":ids})
}
fn state_io<'a>(t:&'a CheckpointCudaTarget,metadata:&Value)->Io<'a> {
    let n=metadata["token_ids"].as_array().unwrap().len();assert!(n>0 && n<t.capacity);
    assert_eq!(t.capacity,262144);assert_eq!(t.layers.len(),64);assert!(t.dflash_capture.is_none() && t.augur_decode.is_some());
    assert_eq!(metadata["kv_identity"],t.attention_mode().kv_identity());
    let physical=n.div_ceil(256)*256;let mut spans=Vec::new();
    for (index,layer) in t.layers.iter().enumerate(){match layer{
        Layer::Linear(l)=>for(role,buffer)in l.checkpoint_buffers(){spans.push(Span{name:format!("layer{index:02}.{role}"),buffer,bytes:buffer.len_bytes()});},
        Layer::Full(l)=>for(role,buffer)in l.checkpoint_buffers(){assert_eq!(buffer.len_bytes(),262144*1088);spans.push(Span{name:format!("layer{index:02}.{role}"),buffer,bytes:physical*1088});},
    }}
    spans.push(Span{name:"target.logits".into(),buffer:&t.logits,bytes:t.logits.len_bytes()});
    let raw=serde_json::to_vec(metadata).unwrap();let mut components=vec![Component{name:"augur.offline-native-prefix".into(),bytes:raw.len() as u64}];
    components.extend(spans.iter().map(|s|Component{name:s.name.clone(),bytes:s.bytes as u64}));
    let mut domain=b"QUASAR-OFFLINE-NATIVE-PREFIX-V1\0".to_vec();domain.extend_from_slice(&raw);
    let plan=TransferPlan{binding_sha256:hash(&domain),components};plan.validate().unwrap();
    Io{stream:&t.stream,spans,metadata:raw,plan}
}
fn continuation(t:&mut CheckpointCudaTarget)->Vec<u32>{let mut ids=vec![t.greedy_next_token().unwrap()];for _ in 0..16{t.consume_selected_augur().unwrap();ids.push(t.greedy_next_token().unwrap());}ids}
pub(super) fn preserve(t:&mut CheckpointCudaTarget,ids:&[u32],root:&Path)->Value {
    assert_eq!(t.consumed,ids.len());t.validate_cursor().unwrap();assert!(!t.poisoned && t.augur_rotated_kv);
    let metadata=descriptor(t,ids);let before_logits=t.logits_snapshot().unwrap();let pending=t.greedy_next_token().unwrap();
    std::fs::create_dir(root).unwrap();save(root,"descriptor.json",&metadata);
    let tick=Instant::now();
    let (saved,state_sha)={let mut source=state_io(t,&metadata);let state_sha=source.state_digest().unwrap();
        let plan=source.plan.clone();let mut writer=LocalCheckpointWriter::create(&root.join("native"),&plan).unwrap();
        (transfer::save(&mut source,&mut writer,&plan,transfer::MAX_CHUNK_BYTES).unwrap(),state_sha)};
    let save_and_hash_s=tick.elapsed().as_secs_f64();let want=continuation(t);
    t.reset_empty().unwrap();assert_eq!(t.consumed,0);
    let mut wrong=metadata.clone();wrong["kv_identity"]="wrong-layout".into();
    // A mismatched independently supplied plan is rejected before target mutation.
    let mut wrong_plan=state_io(t,&metadata).plan.clone();wrong_plan.binding_sha256=hash(&serde_json::to_vec(&wrong).unwrap());
    assert!(LocalCheckpointReader::open(&root.join("native"),&saved.manifest_sha256,&wrong_plan).is_err());
    let tick=Instant::now();t.poisoned=true;
    let (restored,after_sha)={let mut destination=state_io(t,&metadata);let plan=destination.plan.clone();
        let mut reader=LocalCheckpointReader::open(&root.join("native"),&saved.manifest_sha256,&plan).unwrap();
        let raw=reader.manifest_bytes().to_vec();Manifest::decode(&raw,&saved.manifest_sha256,&plan).unwrap();
        let restored=transfer::restore(&mut reader,&mut destination,&raw,&saved.manifest_sha256,&plan).unwrap();
        (restored,destination.state_digest().unwrap())};
    let restore_and_hash_s=tick.elapsed().as_secs_f64();assert_eq!(state_sha,after_sha);
    for layer in &mut t.layers{match layer{Layer::Linear(l)=>l.checkpoint_resume(ids.len()).unwrap(),Layer::Full(l)=>l.checkpoint_resume(ids.len()).unwrap()}}
    t.consumed=ids.len();t.augur_rotated_kv=true;t.selected=None;t.logits_current=true;t.poisoned=false;
    t.info["active_kv_format"]=t.attention_mode().kv_identity().into();t.validate_cursor().unwrap();
    let got_logits=t.logits_snapshot().unwrap();assert_eq!(before_logits.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),got_logits.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
    assert_eq!(pending,t.greedy_next_token().unwrap());assert_eq!(want,continuation(t));
    let receipt=json!({"status":"PASS_OFFLINE_NATIVE_PREFIX","prompt_tokens":ids.len(),"physical_tokens":ids.len().div_ceil(256)*256,
        "descriptor_sha256":hash(&std::fs::read(root.join("descriptor.json")).unwrap()),"binding_sha256":state_io(t,&metadata).plan.binding_sha256,
        "native_manifest_sha256":saved.manifest_sha256,"payload_bytes":saved.payload_bytes,"components":saved.components,"chunks":saved.chunks,
        "staging_bytes":saved.staging_bytes,"save_and_full_state_hash_s":save_and_hash_s,"restore_and_full_state_hash_s":restore_and_hash_s,
        "full_state_sha256":state_sha,"restore_full_state_exact":true,"logits_exact":true,"continuation_tokens":want,
        "continuation_exact":true,"wrong_identity_rejected":true,"restored_payload_bytes":restored.payload_bytes,
        "timing_scope":"offline save/restore plus full-state hashing; excluded from fresh-prefill TPS",
        "cross_build_reuse":false,"dflash_loaded":false});save(root,"cache-receipt.json",&receipt);receipt
}

fn restore_prefix(t:&mut CheckpointCudaTarget,ids:&[u32],root:&Path,descriptor_sha:&str,manifest_sha:&str)->(transfer::TransferReceipt,f64) {
    assert_eq!(t.consumed,0);t.validate_cursor().unwrap();assert!(!t.poisoned);
    let raw=std::fs::read(root.join("descriptor.json")).unwrap();assert_eq!(hash(&raw),descriptor_sha);
    let stored:Value=serde_json::from_slice(&raw).unwrap();let expected=descriptor(t,ids);
    assert_eq!(stored,expected,"offline prefix build/model/numerics/token identity mismatch");
    let plan=state_io(t,&expected).plan.clone();
    let mut reader=LocalCheckpointReader::open(&root.join("native"),manifest_sha,&plan).unwrap();
    let raw=reader.manifest_bytes().to_vec();Manifest::decode(&raw,manifest_sha,&plan).unwrap();
    t.poisoned=true;let tick=Instant::now();
    let restored={let mut destination=state_io(t,&expected);transfer::restore(&mut reader,&mut destination,&raw,manifest_sha,&plan).unwrap()};
    for layer in &mut t.layers{match layer{Layer::Linear(l)=>l.checkpoint_resume(ids.len()).unwrap(),Layer::Full(l)=>l.checkpoint_resume(ids.len()).unwrap()}}
    t.consumed=ids.len();t.augur_rotated_kv=true;t.selected=None;t.logits_current=true;t.poisoned=false;
    t.info["active_kv_format"]=t.attention_mode().kv_identity().into();t.validate_cursor().unwrap();
    t.logits_snapshot().unwrap();t.greedy_next_token().unwrap();
    (restored,tick.elapsed().as_secs_f64())
}
#[test]
#[ignore="exclusive SM70: authenticated offline native prefix restore and decode, no fresh-prefill score"]
fn native_prefix_cache_decode() {
    let out=PathBuf::from(std::env::var_os("AUGUR_NATIVE_DECODE_OUT").unwrap());
    let raw=std::fs::read(std::env::var_os("AUGUR_NATIVE_DECODE_INPUT").unwrap()).unwrap();
    assert_eq!(hash(&raw),std::env::var("AUGUR_NATIVE_DECODE_INPUT_SHA256").unwrap());
    let input:Value=serde_json::from_slice(&raw).unwrap();let cache=&input["offline_prefix_restore"];
    let ids=super::super::transfer_tests::ids(&input["cases"][0]["token_ids"]);
    let raw=std::fs::read(std::env::var_os("AUGUR_MANIFEST").unwrap()).unwrap();
    assert_eq!(hash(&raw),std::env::var("AUGUR_MANIFEST_SHA256").unwrap());
    let manifest:Value=serde_json::from_slice(&raw).unwrap();
    let identity:CheckpointExpectedIdentity=serde_json::from_value(manifest["checkpoint"].clone()).unwrap();
    let initial=cuda_device_allocation_snapshot();let tick=Instant::now();
    let mut t=CheckpointCudaTarget::open_with_execution_policy(std::env::var_os("AUGUR_MODEL_ROOT").unwrap(),
        Qwen35CheckpointProfile::QuasarNvfp4,identity,262144,
        CheckpointProjectionPolicy::AugurQpnAllWorkspaceShapeSmallPairF64,CheckpointKvPolicy::Q8_0).unwrap();
    t.enable_exact_warp_tree_norm().unwrap();t.enable_fused_input_qpn_norm().unwrap();t.enable_batched_ffn8().unwrap();t.enable_augur_decode().unwrap();
    let load_s=tick.elapsed().as_secs_f64();
    let root=PathBuf::from(cache["root"].as_str().unwrap());
    let (restored,restore_s)=restore_prefix(&mut t,&ids,&root,cache["descriptor_sha256"].as_str().unwrap(),cache["native_manifest_sha256"].as_str().unwrap());
    let expected=cache["continuation_tokens"].as_array().unwrap().iter().map(|v|v.as_u64().unwrap()as u32).collect::<Vec<_>>();
    let eos=super::super::transfer_tests::ids(&input["eos"]);
    let budget=input["cases"][0]["max_new_tokens"].as_u64().unwrap_or(128) as usize;
    let mut decoded=super::decode_current(&mut t,ids.len(),budget,&eos,false);
    let got=decoded["output"].as_array().unwrap().iter().map(|v|v.as_u64().unwrap()as u32).collect::<Vec<_>>();
    assert!(got.len()>=expected.len());assert_eq!(&got[..expected.len()],expected.as_slice());
    let info=t.info().clone();drop(t);let closed=cuda_device_allocation_snapshot();
    assert_eq!(closed.live_bytes,initial.live_bytes);assert_eq!(closed.live_allocations,initial.live_allocations);
    decoded["status"]="PASS_OFFLINE_NEW_OWNER_RESTORE".into();
    decoded["prompt_tokens"]=ids.len().into();decoded["load_s"]=load_s.into();decoded["restore_s"]=restore_s.into();
    decoded["restored_payload_bytes"]=restored.payload_bytes.into();decoded["continuation_tokens"]=expected.into();
    decoded["continuation_exact"]=true.into();decoded["fresh_prefill_executed"]=false.into();
    decoded["dflash_loaded"]=false.into();decoded["target"]=info;decoded["allocation_closed"]=true.into();
    decoded["timing_scope"]="cache restore plus shared native decode_current body; no fresh prefill, not HTTP TTFT".into();
    save(&out,"offline-restore-complete.json",&decoded);;
}

// Partition diagnostics share this test-only authenticated cache owner.
mod partition_error;

/// Bounded full-state diagnostic for explicit same-build scheduling comparisons.
/// No file payload is written and this must stay outside all timing enclosures.
pub(super) fn diagnostic_digest(t: &CheckpointCudaTarget, ids: &[u32]) -> String {
    assert_eq!(t.consumed, ids.len());
    let metadata = descriptor(t, ids);
    state_io(t, &metadata).state_digest().unwrap()
}
