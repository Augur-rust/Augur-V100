//! Versioned Augur boundary descriptors. Token history is never reconstructed
//! from text, and a prefix hit is never a generated-answer cache or GDN truncation.
use super::*;
use crate::session_checkpoint::{Component,TransferPlan};
pub const SCHEMA:&str="quasar-native-checkpoint-v1";
pub const PREFIX_BLOCK:usize=4352;
pub const MAX_PREFIX_BLOCK_TOKENS:usize=(CAPACITY-1)/PREFIX_BLOCK*PREFIX_BLOCK;

#[derive(Clone,Debug,PartialEq,Eq,Serialize,Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bindings {
    pub producer_sha256:String,
    pub manifest_sha256:String,
    pub profile:String,
    pub draft_config_sha256:String,
    pub draft_weights_sha256:String,
    pub decode_attention:String,
    #[serde(default="default_sampling")]
    pub sampling:String,
}
fn default_sampling()->String {crate::augur_sampling::GREEDY.into()}
impl Bindings {
    pub fn validate(&self)->Result<()> {
        if [&self.producer_sha256,&self.manifest_sha256,&self.draft_config_sha256,&self.draft_weights_sha256]
            .iter().any(|s|!valid_digest(s)||s.bytes().all(|b|b==b'0'))
            || ![crate::augur_sampling::GREEDY,crate::augur_sampling::PRESET].contains(&self.sampling.as_str())
            || ![PROFILE,DFLASH_B8_PROFILE].contains(&self.profile.as_str())
            || !["tile8_b1", "gqa8_planar", "gqa8_c4_planar"].contains(&self.decode_attention.as_str()) {return Err("checkpoint live binding".into());}
        Ok(())
    }
}
#[derive(Clone,Debug,PartialEq,Eq,Serialize,Deserialize)]
#[serde(tag="kind",rename_all="snake_case",deny_unknown_fields)]
pub enum Boundary {
    InputPrefix {token_ids:Vec<u32>},
    /// Exact input boundary, including a short prompt or a non-B4352 tail.
    InputEnd {token_ids:Vec<u32>},
    Session {snapshot:SessionSnapshot,last_receipt_sha256:String},
}
#[derive(Clone,Debug,PartialEq,Eq,Serialize,Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Descriptor {pub schema:String,pub bindings:Bindings,pub boundary:Boundary}
impl Descriptor {
    pub fn tokens(&self)->&[u32] {match &self.boundary {
        Boundary::InputPrefix{token_ids}|Boundary::InputEnd{token_ids}=>token_ids,Boundary::Session{snapshot,..}=>&snapshot.consumed_token_ids}}
    pub fn pending(&self)->Option<u32> {match &self.boundary {Boundary::Session{snapshot,..}=>Some(snapshot.pending_token),_=>None}}
    pub fn validate(&self,expected:&Bindings)->Result<()> {
        expected.validate()?;
        if self.schema!=SCHEMA || &self.bindings!=expected || self.tokens().is_empty()
            || self.tokens().len()>=context_capacity()? || self.tokens().iter().any(|&t|t>=VOCAB) {
            return Err("checkpoint schema/live identity/token extent".into());
        }
        match &self.boundary {
            Boundary::InputPrefix{token_ids}=>validate_prefix_tokens(token_ids.len())?,
            Boundary::InputEnd{..}=>{},
            Boundary::Session{snapshot:s,last_receipt_sha256}=>{
                if s.profile!=expected.profile || s.manifest_sha256!=expected.manifest_sha256
                    || !valid_id(&s.session_id) || s.revision==0 || s.pending_token>=VOCAB
                    || s.consumed_input_sha256!=token_digest(&s.consumed_token_ids)
                    || s.image_input_sha256.is_some() || !valid_digest(last_receipt_sha256) {
                    return Err("checkpoint exact text session ledger".into());
                }
            }
        }
        Ok(())
    }
    pub fn encode(&self)->Result<Vec<u8>> {self.validate(&self.bindings)?;Ok(serde_json::to_vec(self)?)}
    /// State inventory is independently derived by the live target, never read
    /// back from the untrusted native manifest and promoted to expected state.
    pub(crate) fn plan(&self,state:Vec<Component>)->Result<TransferPlan> {
        let raw=self.encode()?;
        let mut components=vec![Component{name:"augur.boundary".into(),bytes:raw.len() as u64}];
        components.extend(state);
        let mut bound=b"QUASAR-NATIVE-BOUNDARY-V1\0".to_vec();bound.extend_from_slice(&raw);
        let plan=TransferPlan{binding_sha256:digest(&bound),components};plan.validate()?;Ok(plan)
    }
    pub fn lookup_prefix(&self,expected:&Bindings,input:&[u32],output_reserve:usize)->Result<bool> {
        self.validate(expected)?;
        if !matches!(&self.boundary,Boundary::InputPrefix{..}|Boundary::InputEnd{..}) {return Err("session is not an input cache".into());}
        if input.is_empty() || input.iter().any(|&t|t>=VOCAB) || output_reserve==0
            || !fits_context(input.len().checked_add(output_reserve))? {return Err("cache request capacity".into());}
        Ok(input.len()>=self.tokens().len() && input.starts_with(self.tokens()))
    }
}
pub fn validate_prefix_tokens(n:usize)->Result<()> {
    if !(PREFIX_BLOCK..=MAX_PREFIX_BLOCK_TOKENS).contains(&n) || n%PREFIX_BLOCK!=0 || n>=context_capacity()? {
        return Err("Augur input cache requires a complete B4352 boundary below capacity".into());}
    Ok(())
}
/// Select only a complete checkpoint boundary. A longer nonmatching entry must
/// not hide a shorter matching KV/GDN state. Equal input lengths reuse logits.
pub(crate) fn longest_prefix<'a>(entries:impl IntoIterator<Item=&'a Descriptor>,expected:&Bindings,input:&[u32],output_reserve:usize)->Result<Option<usize>> {
    if input.is_empty() || input.iter().any(|&t|t>=VOCAB) || output_reserve==0
        || !fits_context(input.len().checked_add(output_reserve))? {return Err("cache request capacity".into());}
    let mut best:Option<(usize,usize)>=None;
    for (index,d) in entries.into_iter().enumerate() {
        if d.lookup_prefix(expected,input,output_reserve)? && best.is_none_or(|(_,n)|d.tokens().len()>n) {
            best=Some((index,d.tokens().len()));
        }
    }
    Ok(best.map(|(index,_)|index))
}
#[derive(Clone,Debug,Serialize,Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Envelope {pub schema:String,pub checkpoint_id:String,pub native_manifest_sha256:String,pub descriptor:Descriptor}
impl Envelope {
    pub fn encode(&self)->Result<Vec<u8>> {
        if self.schema!="quasar-checkpoint-envelope-v1" || !valid_digest(&self.checkpoint_id)
            || !valid_digest(&self.native_manifest_sha256) {return Err("checkpoint envelope".into());}
        self.descriptor.validate(&self.descriptor.bindings)?;Ok(serde_json::to_vec(self)?)
    }
    pub fn decode(raw:&[u8],expected_sha:&str,bindings:&Bindings)->Result<Self> {
        if raw.len()>4*1024*1024 || digest(raw)!=expected_sha {return Err("checkpoint envelope length/hash".into());}
        let v:Self=serde_json::from_slice(raw)?;v.descriptor.validate(bindings)?;
        if v.encode()?!=raw {return Err("checkpoint canonical envelope bytes".into());}Ok(v)
    }
}
#[derive(Clone,Debug,Serialize,Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SaveRequest {pub request_id:String,pub session_id:String,pub revision:u64,pub checkpoint_id:String}
#[derive(Clone,Debug,Serialize,Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RestoreRequest {pub request_id:String,pub session_id:String,pub checkpoint_id:String,pub checkpoint_sha256:String}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn sampling_modes_cannot_share_checkpoints() {
        let greedy=bindings();let mut sampled=greedy.clone();
        sampled.sampling=crate::augur_sampling::PRESET.into();
        sampled.validate().unwrap();assert_ne!(greedy,sampled);
        sampled.sampling="unknown".into();assert!(sampled.validate().is_err());
    }
    fn bindings()->Bindings {Bindings{producer_sha256:"a".repeat(64),manifest_sha256:"b".repeat(64),profile:PROFILE.into(),
        draft_config_sha256:"c".repeat(64),draft_weights_sha256:"d".repeat(64),decode_attention:"tile8_b1".into(),sampling:crate::augur_sampling::GREEDY.into()}}
    fn prefix()->Descriptor {Descriptor{schema:SCHEMA.into(),bindings:bindings(),boundary:Boundary::InputPrefix{token_ids:vec![7;PREFIX_BLOCK]}}}
    #[test] fn exact_complete_boundary_only() {
        let p=prefix();let mut ids=p.tokens().to_vec();assert!(p.lookup_prefix(&bindings(),&ids,8).unwrap());
        ids.push(9);assert!(p.lookup_prefix(&bindings(),&ids,8).unwrap());ids[0]=8;
        assert!(!p.lookup_prefix(&bindings(),&ids,8).unwrap());
        for n in [0,512,2048,4096,8192,17409,CAPACITY,usize::MAX] {assert!(validate_prefix_tokens(n).is_err());}
        for n in [4352,8704,13056,17408,30464,MAX_PREFIX_BLOCK_TOKENS] {validate_prefix_tokens(n).unwrap();}
    }
    #[test] fn exact_prompt_tail_has_its_own_boundary_contract() {
        for n in [1,73,32760,261632,CAPACITY-1] {
            let d=Descriptor{boundary:Boundary::InputEnd{token_ids:vec![7;n]},..prefix()};
            d.validate(&bindings()).unwrap();
            assert!(d.lookup_prefix(&bindings(),d.tokens(),1).unwrap());
            if n+8>CAPACITY {assert!(d.lookup_prefix(&bindings(),d.tokens(),8).is_err());}
        }
        let empty=Descriptor{boundary:Boundary::InputEnd{token_ids:vec![]},..prefix()};
        assert!(empty.validate(&bindings()).is_err());
        let invalid=Descriptor{boundary:Boundary::InputEnd{token_ids:vec![VOCAB]},..prefix()};
        assert!(invalid.validate(&bindings()).is_err());
    }
    #[test] fn longest_complete_prefix_handles_document_branches_and_equal_input() {
        let short=prefix();
        let mut longer=short.tokens().to_vec();longer.extend(vec![9;PREFIX_BLOCK]);
        let long=Descriptor{boundary:Boundary::InputPrefix{token_ids:longer},..prefix()};
        let mut input=short.tokens().to_vec();input.extend(vec![8;PREFIX_BLOCK]);input.push(3);
        let tail=Descriptor{boundary:Boundary::InputEnd{token_ids:input.clone()},..prefix()};
        let entries=[long,short,tail];
        assert_eq!(longest_prefix(entries[..2].iter(),&bindings(),&input,8).unwrap(),Some(1));
        assert_eq!(longest_prefix(entries.iter(),&bindings(),&input,8).unwrap(),Some(2));
        input.push(4);
        assert_eq!(longest_prefix(entries.iter(),&bindings(),&input,8).unwrap(),Some(2));
        input[0]=6;
        assert_eq!(longest_prefix(entries.iter(),&bindings(),&input,8).unwrap(),None);
        assert!(longest_prefix(entries.iter(),&bindings(),&[],8).is_err());
        let mut wrong=bindings();wrong.producer_sha256="e".repeat(64);
        assert!(longest_prefix(entries.iter(),&wrong,&input,8).is_err());
    }
    #[test] fn model_build_policy_and_capacity_are_bound() {
        let p=prefix();let mut b=bindings();b.producer_sha256="e".repeat(64);assert!(p.validate(&b).is_err());
        b=bindings();b.profile=DFLASH_B8_PROFILE.into();assert!(p.validate(&b).is_err());
        assert!(p.lookup_prefix(&bindings(),&vec![7;CAPACITY],8).is_err());
        assert!(p.lookup_prefix(&bindings(),&[VOCAB],8).is_err());
    }
    #[test] fn planar_cache_is_bound_to_the_complete_owner_mode() {
        let legacy = prefix();
        let mut planar = legacy.clone();
        planar.bindings.decode_attention = "gqa8_planar".into();
        planar.validate(&planar.bindings).unwrap();
        assert!(legacy.validate(&planar.bindings).is_err());
        assert!(planar.validate(&legacy.bindings).is_err());
        let state = vec![Component { name: "layer03.key_cache".into(), bytes: 1088 * 4352 }];
        assert_ne!(legacy.plan(state.clone()).unwrap().binding_sha256,
                   planar.plan(state).unwrap().binding_sha256);
        let mut invalid = planar.bindings.clone();
        invalid.decode_attention = "unknown_planar".into();
        assert!(invalid.validate().is_err());
    }
    #[test] fn c4_and_c8_caches_reject_each_other_with_the_same_producer() {
        let mut c8=prefix();c8.bindings.decode_attention="gqa8_planar".into();
        let mut c4=c8.clone();c4.bindings.decode_attention="gqa8_c4_planar".into();
        c4.validate(&c4.bindings).unwrap();c8.validate(&c8.bindings).unwrap();
        assert!(c4.validate(&c8.bindings).is_err());assert!(c8.validate(&c4.bindings).is_err());
        let state=vec![Component{name:"layer03.key_cache".into(),bytes:1088*4352}];
        assert_ne!(c4.plan(state.clone()).unwrap().binding_sha256,c8.plan(state).unwrap().binding_sha256);
    }
    #[test] fn envelope_is_canonical_not_untrusted_manifest_authority() {
        let e=Envelope{schema:"quasar-checkpoint-envelope-v1".into(),checkpoint_id:"e".repeat(64),native_manifest_sha256:"f".repeat(64),descriptor:prefix()};
        let raw=e.encode().unwrap();Envelope::decode(&raw,&digest(&raw),&bindings()).unwrap();
        let mut trailing=raw.clone();trailing.push(b' ');assert!(Envelope::decode(&trailing,&digest(&trailing),&bindings()).is_err());
        assert!(Envelope::decode(&raw,&"0".repeat(64),&bindings()).is_err());
    }
}

#[cfg(all(windows,feature="qwen35-augur-http"))]
impl<T:SessionTarget> Session<T> {
    pub(crate) fn save_checkpoint(&mut self,r:&SaveRequest,persist:impl FnOnce(&serde_json::Value)->Result<String>)->Result<serde_json::Value>{
        let b=self.target.checkpoint_bindings().ok_or("checkpoint storage disabled")?;
        let a=self.active.as_ref().ok_or("checkpoint requires active session")?;
        if self.poisoned || self.cancelled_turn.is_some() || !valid_id(&r.request_id) || self.requests.contains(&r.request_id)
            || !valid_digest(&r.checkpoint_id) || a.snapshot.session_id!=r.session_id || a.snapshot.revision!=r.revision {
            return Err("checkpoint save identity/state".into());}
        let d=Descriptor{schema:SCHEMA.into(),bindings:b,boundary:Boundary::Session{snapshot:a.snapshot.clone(),last_receipt_sha256:a.receipt_sha256.clone()}};
        d.validate(&d.bindings)?;self.requests.insert(r.request_id.clone());
        let started=Instant::now();
        let saved=match self.target.save_checkpoint(&d,&r.checkpoint_id){Ok(v)=>v,Err(e)=>{
            if self.target.drain().is_err(){let _=self.retire();}return Err(e);
        }};
        let receipt=serde_json::json!({"schema":"quasar-checkpoint-save-receipt-v1","request_id":r.request_id,
            "session":self.snapshot(),"checkpoint":saved,"save_wall_s":started.elapsed().as_secs_f64(),"state_unchanged":true});
        let hash=persist(&receipt)?;
        if !valid_digest(&hash){return Err("checkpoint save receipt digest".into());}
        Ok(serde_json::json!({"receipt":receipt,"receipt_sha256":hash}))
    }
    pub(crate) fn restore_checkpoint(&mut self,r:&RestoreRequest,persist:impl FnOnce(&serde_json::Value)->Result<String>)->Result<serde_json::Value>{
        let b=self.target.checkpoint_bindings().ok_or("checkpoint storage disabled")?;
        if self.poisoned || self.active.is_some() || self.cancelled_turn.is_some() || self.target.consumed_tokens()!=0
            || !valid_id(&r.request_id) || !valid_id(&r.session_id) || self.requests.contains(&r.request_id)
            || self.sessions.contains(&r.session_id) || !valid_digest(&r.checkpoint_id) || !valid_digest(&r.checkpoint_sha256) {
            return Err("checkpoint restore requires a fresh unpublished session ID/owner boundary".into());}
        self.requests.insert(r.request_id.clone());self.sessions.insert(r.session_id.clone());
        // After admission, any read, copy, drain or durable-publication failure
        // retires the unpublished owner. Never silently replay a corrupt cache.
        self.poisoned=true;let started=Instant::now();
        let (d,detail)=self.target.restore_checkpoint(r)?;d.validate(&b)?;
        let Boundary::Session{mut snapshot,..}=d.boundary else{return Err("input checkpoint cannot publish generated session".into());};
        if snapshot.session_id==r.session_id || self.target.consumed_tokens()!=snapshot.consumed_token_ids.len()
            || self.target.greedy_next_token()?!=snapshot.pending_token {return Err("restored session boundary mismatch".into());}
        let previous_session=snapshot.session_id.clone();snapshot.session_id=r.session_id.clone();snapshot.revision=1;
        self.target.drain()?;
        let receipt=serde_json::json!({"schema":"quasar-checkpoint-restore-receipt-v1","request_id":r.request_id,
            "session":snapshot,"source_session_id":previous_session,"checkpoint":detail,"restored_prefix_tokens":snapshot.consumed_token_ids.len(),
            "recomputed_prefix_tokens":0,"restore_wall_s":started.elapsed().as_secs_f64(),"draft_context":"fresh_local_epoch_reprojected_on_next_proposal"});
        let hash=persist(&receipt)?;if !valid_digest(&hash){return Err("restore receipt digest".into());}
        self.active=Some(Active{snapshot,receipt_sha256:hash.clone()});self.poisoned=false;
        Ok(serde_json::json!({"receipt":receipt,"receipt_sha256":hash}))
    }
}

#[cfg(all(windows,feature="qwen35-augur-http"))]
impl Session<crate::augur_session_cuda::CudaTarget> {
    pub(crate) fn configure_checkpoint(&mut self,evidence:&Path)->Result<serde_json::Value>{
        self.target.configure_checkpoint(evidence,&self.manifest_sha256)
    }
}

pub fn storage_requested()->bool {cfg!(all(windows,feature="qwen35-augur-http")) && std::env::var_os("AUGUR_CHECKPOINT_STORE").is_some()}
