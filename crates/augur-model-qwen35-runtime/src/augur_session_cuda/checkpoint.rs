//! Product binding for immutable native checkpoints and bounded input entries.
use super::*;
use crate::augur_session::checkpoint::{self as contract,Bindings,Boundary,Descriptor,Envelope,RestoreRequest};
use crate::session_checkpoint::filesystem::{http::{HttpCheckpointStore,StoreBudget},LocalCheckpointReader};
use std::{collections::BTreeMap,io::Read,path::{Path,PathBuf},time::Instant};
use sha2::{Digest,Sha256};
use crate::session_checkpoint::ParallelRestoreSchedule;
use serde_json::json;
const PREFIX_QUOTA:usize=8;
const PREFIX_BYTES:u64=64*1024*1024*1024;
pub(super) struct Manager {bindings:Bindings,store:Option<HttpCheckpointStore>,prefix:Option<Prefix>,restore_timing:bool,restore_workers:usize,restore_schedule:ParallelRestoreSchedule}
struct Saved {descriptor:Descriptor,id:String,sha:String}
struct Prefix {
    tokens:Option<usize>,prompt_end:bool,store:HttpCheckpointStore,saved:Vec<Saved>,
    unavailable:BTreeMap<String,String>,persistent:bool,index_rebuild_s:f64,recovered_entries:usize,
}
/// Every math/build/draft binding participates; unrelated builds get a distinct
/// directory and can never promote each other's state into the live index.
fn prefix_namespace(bindings:&Bindings)->Result<String> {
    bindings.validate()?;
    let mut bytes=b"QUASAR-INPUT-PREFIX-NAMESPACE-V1\0".to_vec();
    bytes.extend_from_slice(&serde_json::to_vec(bindings)?);Ok(session::digest(&bytes))
}
impl Prefix {
    fn open(store:HttpCheckpointStore,bindings:&Bindings,tokens:Option<usize>,prompt_end:bool,persistent:bool)->Result<Self> {
        let started=Instant::now();let scan=store.scan_opaque()?;
        let mut saved=Vec::new();let mut unavailable:BTreeMap<_,_>=scan.unavailable.into_iter()
            .map(|v|(v.checkpoint_id,v.reason)).collect();
        for metadata in scan.published {
            let decoded=(|| -> Result<Descriptor> {
                let e=Envelope::decode(&metadata.bytes,&metadata.checkpoint_sha256,bindings)?;
                if e.checkpoint_id!=metadata.checkpoint_id
                    || session::digest(&e.descriptor.encode()?)!=metadata.checkpoint_id
                    || !matches!(e.descriptor.boundary,Boundary::InputPrefix{..}|Boundary::InputEnd{..}) {
                    return Err("persistent input cache ID/boundary".into());
                }
                Ok(e.descriptor)
            })();
            match decoded {
                Ok(descriptor)=>saved.push(Saved{descriptor,id:metadata.checkpoint_id,sha:metadata.checkpoint_sha256}),
                Err(error)=>{unavailable.insert(metadata.checkpoint_id,error.to_string());},
            }
        }
        let recovered_entries=saved.len();
        Ok(Self{tokens,prompt_end,store,saved,unavailable,persistent,index_rebuild_s:started.elapsed().as_secs_f64(),recovered_entries})
    }
}
fn sha_file(p:&Path)->Result<String>{
    let mut f=std::fs::File::open(p)?;let mut h=Sha256::new();let mut buf=vec![0;1024*1024];
    loop {let n=f.read(&mut buf)?;if n==0{break;}h.update(&buf[..n]);}Ok(format!("{:x}",h.finalize()))
}
fn env_value(k:&str)->Result<Option<String>>{match std::env::var(k){Ok(v) if !v.is_empty()=>Ok(Some(v)),Err(std::env::VarError::NotPresent)=>Ok(None),_=>Err(format!("invalid {k}").into())}}
impl Manager {
    pub(super) fn configure(evidence:&Path,manifest:&str,attention:&str)->Result<Option<Self>> {
        let restore_timing=match env_value("AUGUR_CACHE_RESTORE_DIAGNOSTIC")?.as_deref() {
            None|Some("0")=>false,Some("1")=>true,_=>return Err("invalid AUGUR_CACHE_RESTORE_DIAGNOSTIC".into())};
        let restore_workers=env_value("AUGUR_CACHE_RESTORE_WORKERS")?.map(|s|s.parse::<usize>()).transpose()?.unwrap_or(1);
        if !matches!(restore_workers,1|2|4|8) || (restore_timing && restore_workers!=1) {
            return Err("restore workers must be 1/2/4/8; diagnostic lane requires 1".into());}
        let restore_schedule=ParallelRestoreSchedule::parse(env_value("AUGUR_CACHE_RESTORE_SCHEDULE")?.as_deref().unwrap_or("batch"))?;
        if restore_schedule==ParallelRestoreSchedule::Pipeline && (restore_timing || restore_workers<2) {
            return Err("pipeline restore requires parallel verification without diagnostic timing".into());
        }
        let root=env_value("AUGUR_CHECKPOINT_STORE")?;
        let persistent_root=env_value("AUGUR_INPUT_PREFIX_CACHE_ROOT")?;
        let prefix=env_value("AUGUR_INPUT_PREFIX_CACHE_TOKENS")?.map(|v|v.parse::<usize>()).transpose()?;
        let prompt_end=match env_value("AUGUR_INPUT_PREFIX_CACHE_PROMPT_END")?.as_deref() {
            None|Some("0")=>false,Some("1")=>true,_=>return Err("invalid AUGUR_INPUT_PREFIX_CACHE_PROMPT_END".into())};
        if persistent_root.is_some() && prefix.is_none() && !prompt_end {return Err("prefix cache root requires tokens or prompt end capture".into());}
        if root.is_none() && prefix.is_none() && !prompt_end{return Ok(None);}
        if let Some(n)=prefix{contract::validate_prefix_tokens(n)?;}
        let bindings=Bindings{producer_sha256:sha_file(&std::env::current_exe()?)?,manifest_sha256:manifest.into(),profile:session::requested_profile()?.into(),
            draft_config_sha256:std::env::var("AUGUR_DRAFT_CONFIG_SHA256")?,draft_weights_sha256:std::env::var("AUGUR_DRAFT_SHA256")?,decode_attention:attention.into(),sampling:crate::augur_sampling::Mode::from_env()?.label().into(),sampling_seed:crate::augur_sampling::process_seed()?};
        bindings.validate()?;
        let store=root.map(|p|HttpCheckpointStore::open(&PathBuf::from(p),StoreBudget{max_checkpoints:8,max_reserved_bytes:PREFIX_BYTES})).transpose()?;
        let prefix=if prefix.is_some() || prompt_end {
            let persistent=persistent_root.is_some();
            let dir=if let Some(root)=persistent_root {
                let root=PathBuf::from(root);if !root.is_absolute(){return Err("prefix cache root must be absolute".into());}
                let dir=root.join(prefix_namespace(&bindings)?);std::fs::create_dir_all(&dir)?;dir
            }else{let dir=evidence.join("input-prefix-store");std::fs::create_dir(&dir)?;dir};
            let store=HttpCheckpointStore::open(&dir,StoreBudget{max_checkpoints:PREFIX_QUOTA,max_reserved_bytes:PREFIX_BYTES})?;
            Some(Prefix::open(store,&bindings,prefix,prompt_end,persistent)?)
        }else{None};
        Ok(Some(Self{bindings,store,prefix,restore_timing,restore_workers,restore_schedule}))
    }
    pub(super) fn ready(&self)->Value {json!({"save_restore_enabled":self.store.is_some(),"scope":"private_exact_build_text_only",
        "input_prefix_restore_workers":self.restore_workers,"input_prefix_restore_schedule":self.restore_schedule.label(),"input_prefix_restore_diagnostic":self.restore_timing,"input_prefix_tokens":self.prefix.as_ref().and_then(|p|p.tokens),"input_prefix_prompt_end":self.prefix.as_ref().is_some_and(|p|p.prompt_end),
        "input_prefix_scope":"bounded_multi_entry_exact_input","input_prefix_quota":PREFIX_QUOTA,"input_prefix_reserved_byte_quota":PREFIX_BYTES,
        "input_prefix_persistent_index":self.prefix.as_ref().is_some_and(|p|p.persistent),
        "input_prefix_index_rebuild_s":self.prefix.as_ref().map(|p|p.index_rebuild_s),
        "input_prefix_recovered_entries":self.prefix.as_ref().map(|p|p.recovered_entries),
        "input_prefix_unavailable_entries":self.prefix.as_ref().map(|p|&p.unavailable),
        "input_prefix_index_source":"sealed_envelopes_metadata_only_payload_checked_on_restore","bindings":self.bindings,"checkpoint_quota":8,"checkpoint_reserved_byte_quota":PREFIX_BYTES})}
    pub(super) fn bindings(&self)->Option<Bindings>{self.store.as_ref().map(|_|self.bindings.clone())}
    pub(super) fn save(&mut self,t:&mut CheckpointCudaTarget,d:&Descriptor,id:&str)->Result<Value>{
        d.validate(&self.bindings)?;save(self.store.as_mut().ok_or("checkpoint store disabled")?,t,d,id)
    }
    pub(super) fn restore(&mut self,t:&mut CheckpointCudaTarget,r:&RestoreRequest)->Result<(Descriptor,Value)>{
        let store=self.store.as_ref().ok_or("checkpoint store disabled")?;
        let read=store.read_opaque(&r.checkpoint_id,&r.checkpoint_sha256)?;
        let envelope=Envelope::decode(&read.bytes,&r.checkpoint_sha256,&self.bindings)?;
        if envelope.checkpoint_id!=r.checkpoint_id || !matches!(&envelope.descriptor.boundary,Boundary::Session{snapshot,..} if snapshot.session_id!=r.session_id) {
            return Err("checkpoint needs a generated boundary and a new session ID".into());}
        let plan=t.checkpoint_plan(&envelope.descriptor)?;
        if plan.validate()?!=read.payload_bytes {return Err("checkpoint native reservation extent".into());}
        let native=t.checkpoint_restore(&envelope.descriptor,&read.native_path(),&envelope.native_manifest_sha256)?;
        read.verify_namespace()?;
        Ok((envelope.descriptor,json!({"checkpoint_id":r.checkpoint_id,"checkpoint_sha256":r.checkpoint_sha256,
            "native_manifest_sha256":native.manifest_sha256,"payload_bytes":native.payload_bytes,"staging_bytes":native.staging_bytes,"no_prefix_replay":true})))
    }
    pub(super) fn prefill(&mut self,t:&mut CheckpointCudaTarget,ids:&[u32],cancel:&SessionCancellationToken)->Result<Option<Value>> {
        let Some(p)=self.prefix.as_mut() else{return Ok(None);};
        // Same-session append and vision continue to use their own live state.
        if t.consumed_tokens()!=0 || (!p.prompt_end && p.tokens.is_none_or(|n|ids.len()<n)){return Ok(None);}
        cancel.check("before Augur input cache lookup")?;
        let started=Instant::now();let mut capture_s=0.0;let mut restore_s=0.0;let mut captures=Vec::new();
        let mut matched=0;let mut restored_bytes=0;let mut failures=Vec::new();let mut restore_diagnostic=None;let mut restore_staging_bytes=0;
        loop {
            cancel.check("before Augur input cache candidate")?;
            let Some(index)=contract::longest_prefix(p.saved.iter().map(|s|&s.descriptor),&self.bindings,ids,1)? else{break;};
            let tick=Instant::now();
            match restore_input(&p.store,t,&p.saved[index],&self.bindings,self.restore_timing,self.restore_workers,self.restore_schedule) {
                Ok((bytes,diagnostic,staging))=>{restore_staging_bytes=staging;matched=p.saved[index].descriptor.tokens().len();restored_bytes=bytes;restore_diagnostic=diagnostic;restore_s+=tick.elapsed().as_secs_f64();break;},
                Err(error)=>{
                    // A native read may have partially written state and poisoned
                    // the owner. Reset must succeed before any shorter retry.
                    t.reset_empty()?;
                    let failed=p.saved.remove(index);let reason=error.to_string();
                    failures.push(json!({"checkpoint_id":failed.id,"tokens":failed.descriptor.tokens().len(),"reason":reason}));
                    p.unavailable.insert(failed.id,reason);restore_s+=tick.elapsed().as_secs_f64();
                    cancel.check("after Augur failed cache restore")?;
                }
            }
        }
        let hit=matched>0;
        let cursor=if hit {matched}else if let Some(n)=p.tokens.filter(|&n|n<=ids.len()) {
            session::SessionTarget::prefill(t,&ids[..n],cancel)?;
            let d=Descriptor{schema:contract::SCHEMA.into(),bindings:self.bindings.clone(),boundary:Boundary::InputPrefix{token_ids:ids[..n].to_vec()}};
            let tick=Instant::now();if let Some(v)=capture(p,t,d)?{captures.push(v);}capture_s+=tick.elapsed().as_secs_f64();n
        }else{0};
        cancel.check("before Augur cached suffix prefill")?;
        // A complete input hit resumes saved logits; do not send an empty GEMM.
        let suffix=if cursor<ids.len(){session::SessionTarget::prefill(t,&ids[cursor..],cancel)?}else{None};
        if p.prompt_end && !p.saved.iter().any(|s|s.descriptor.tokens()==ids) {
            cancel.check("before Augur prompt end capture")?;
            let d=Descriptor{schema:contract::SCHEMA.into(),bindings:self.bindings.clone(),boundary:Boundary::InputEnd{token_ids:ids.to_vec()}};
            let tick=Instant::now();if let Some(v)=capture(p,t,d)?{captures.push(v);}capture_s+=tick.elapsed().as_secs_f64();
        }
        let captured=captures.iter().filter(|v|v["stored"].as_bool()==Some(true)).filter_map(|v|v["tokens"].as_u64()).max().unwrap_or(0);
        Ok(Some(json!({"input_prefix_cache":{"hit":hit,"matched_tokens":matched,"captured_tokens":captured,"captures":captures,
            "capture_s":capture_s,"restore_s":restore_s,"restore_diagnostic":restore_diagnostic,"restore_verification_workers":self.restore_workers,"restore_schedule":self.restore_schedule.label(),"restore_staging_bytes":restore_staging_bytes,"restored_payload_bytes":restored_bytes,"complete_wall_s":started.elapsed().as_secs_f64(),
            "suffix_tokens":ids.len()-cursor,"new_prefill_tokens":ids.len()-matched,"recomputed_prefix_tokens":if hit{0}else{cursor},
            "partial_match":matched>0 && matched<ids.len(),"exact_input_hit":matched==ids.len(),"restored_logits":hit && cursor==ids.len(),
            "entries":p.saved.len(),"unavailable_entries":p.unavailable.len(),"restore_failures":failures,
            "persistent_index":p.persistent,"recovered_entries":p.recovered_entries,"longest_prefix_search":true,"source":"native_gdn_conv_kv_feature_logits_state"},"suffix_prefill":suffix})))
    }
}
fn restore_input(store:&HttpCheckpointStore,t:&mut CheckpointCudaTarget,s:&Saved,bindings:&Bindings,diagnostic:bool,workers:usize,schedule:ParallelRestoreSchedule)->Result<(u64,Option<Value>,usize)> {
    let started=diagnostic.then(Instant::now);
    let read=store.read_opaque(&s.id,&s.sha)?;
    let e=Envelope::decode(&read.bytes,&s.sha,bindings)?;
    if e.checkpoint_id!=s.id || e.descriptor!=s.descriptor {return Err("input cache descriptor changed".into());}
    let plan=t.checkpoint_plan(&s.descriptor)?;
    if plan.validate()?!=read.payload_bytes{return Err("prefix reservation extent".into());}
    let envelope_s=started.map(|t|t.elapsed().as_secs_f64());
    let (native,mut detail)=if diagnostic {let (r,d)=t.checkpoint_restore_with_timing(&s.descriptor,&read.native_path(),&e.native_manifest_sha256)?;(r,Some(d))}
        else if workers>1 {(t.checkpoint_restore_parallel_scheduled(&s.descriptor,&read.native_path(),&e.native_manifest_sha256,workers,schedule)?,None)}
        else {(t.checkpoint_restore(&s.descriptor,&read.native_path(),&e.native_manifest_sha256)?,None)};
    let tick=diagnostic.then(Instant::now);read.verify_namespace()?;
    if let Some(d)=&mut detail {d["manager_envelope_s"]=envelope_s.into();d["manager_namespace_s"]=tick.map(|t|t.elapsed().as_secs_f64()).into();
        d["manager_complete_s"]=started.map(|t|t.elapsed().as_secs_f64()).into();}
    Ok((native.payload_bytes,detail,native.staging_bytes))
}
fn capture(p:&mut Prefix,t:&mut CheckpointCudaTarget,d:Descriptor)->Result<Option<Value>> {
    if p.saved.iter().any(|s|s.descriptor.tokens()==d.tokens()){return Ok(None);}
    let id=session::digest(&d.encode()?);
    if p.unavailable.contains_key(&id) {return Ok(Some(json!({"stored":false,"tokens":d.tokens().len(),"reason":"cache_entry_unavailable","checkpoint_id":id})));}
    let bytes=t.checkpoint_plan(&d)?.validate()?;
    if !p.store.has_capacity_for(bytes)? {return Ok(Some(json!({"stored":false,"tokens":d.tokens().len(),"reason":"cache_quota","payload_bytes":bytes})));}
    let saved=save(&mut p.store,t,&d,&id)?;
    let sha=saved["checkpoint_sha256"].as_str().ok_or("prefix checkpoint digest")?.into();
    let report=json!({"stored":true,"tokens":d.tokens().len(),"boundary":if matches!(&d.boundary,Boundary::InputEnd{..}){"input_end"}else{"literal_b4352"},"checkpoint":saved});
    p.saved.push(Saved{descriptor:d,id,sha});Ok(Some(report))
}
fn save(store:&mut HttpCheckpointStore,t:&mut CheckpointCudaTarget,d:&Descriptor,id:&str)->Result<Value>{
    let plan=t.checkpoint_plan(d)?;let slot=store.reserve(id,plan.validate()?)?;
    let native=t.checkpoint_save(d,&slot.native_path())?;
    let e=Envelope{schema:"quasar-checkpoint-envelope-v1".into(),checkpoint_id:id.into(),native_manifest_sha256:native.manifest_sha256.clone(),descriptor:d.clone()};
    let sha=slot.publish_opaque(id,&e.encode()?,&native.manifest_sha256,&plan)?;
    Ok(json!({"checkpoint_id":id,"checkpoint_sha256":sha,"native_manifest_sha256":native.manifest_sha256,
        "payload_bytes":native.payload_bytes,"staging_bytes":native.staging_bytes,"chunks":native.chunks,"components":native.components}))
}

#[cfg(test)]
mod tests;
