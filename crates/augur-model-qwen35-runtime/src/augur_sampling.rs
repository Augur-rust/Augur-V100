//! Fixed target sampling contract. Deterministic draft proposals are accepted
//! only when they equal a draw from the target conditional distribution.
//! A position-indexed PRNG makes rollback, prefix restore and B1/B8 grouping
//! independent of how many uncommitted speculative rows were evaluated.
use serde_json::{json,Value};
pub const PRESET:&str="ninfer_thinking_v1";
pub const GREEDY:&str="greedy";
pub const TOP_K:usize=20;
pub const TOP_P:f64=0.95;
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum Mode { Greedy, Thinking }
impl Mode {
    pub fn from_env()->Result<Self,Box<dyn std::error::Error>> {
        match std::env::var("AUGUR_SAMPLING").as_deref() {
            Err(std::env::VarError::NotPresent)|Ok("greedy")=>Ok(Self::Greedy),
            Ok(PRESET)=>Ok(Self::Thinking),
            _=>Err("AUGUR_SAMPLING must be greedy or ninfer_thinking_v1".into()),
        }
    }
    pub fn label(self)->&'static str {match self {Self::Greedy=>GREEDY,Self::Thinking=>PRESET}}
    pub fn description(self)->Value {
        match self {
            Self::Greedy=>json!({"preset":GREEDY,"temperature":0.0,"top_p":1.0,"greedy":true}),
            Self::Thinking=>json!({"preset":PRESET,"temperature":1.0,"top_p":TOP_P,"top_k":TOP_K,
                "min_p":0.0,"presence_penalty":0.0,"frequency_penalty":0.0,"seed":0,"greedy":false,
                "filter_order":"temperature_top_k_top_p","rng":"splitmix64_absolute_position_v1",
                "verification":"deterministic_draft_target_sample_equality_v1",
                "counter_scope":"absolute_next_token_position_including_prompt",
                "seed_replay_scope":"same_numerical_path_and_prefix"}),
        }
    }
}
/// The sampling contract a decode actually runs: the target preset plus the speculative
/// verification selected by the environment. Health, API metrics and receipts all report
/// this one value, so no surface can advertise a contract the owner does not execute.
pub fn effective_description()->Result<Value,Box<dyn std::error::Error>> {
    use crate::augur_speculative_sampling as S;
    let mode=Mode::from_env()?;
    let positions=S::positions_from_env()?;
    let calibration=S::ProposalCalibration::from_env(positions)?;
    let block=S::block_from_env()?;
    if positions>0 && mode!=Mode::Thinking {return Err("probability prefix requires target sampling".into());}
    if block && positions!=7 {return Err("block verification requires all7 probability proposals".into());}
    let mut v=mode.description();
    if mode==Mode::Thinking {v["seed"]=json!(process_seed()?);v["seed_scope"]=json!("process");}
    if positions>0 {
        v["verification"]=json!(if block {S::BLOCK_CONTRACT} else {S::CONTRACT});
        v["probability_prefix_positions"]=json!(positions);
        v["proposal_calibration"]=json!(calibration);
        v["rng"]=json!("splitmix64_absolute_position_independent_purpose_domains_v1");
    }
    Ok(v)
}
/// One immutable seed per native owner. Invalid values fail before sampling.
fn parse_seed(value:Option<&str>)->Result<u64,&'static str> {
    match value {
        None=>Ok(0),
        Some(v) if !v.is_empty() && v.bytes().all(|b|b.is_ascii_digit())=>v.parse().map_err(|_|"AUGUR_SAMPLING_SEED must be a u64 decimal integer"),
        _=>Err("AUGUR_SAMPLING_SEED must be a u64 decimal integer"),
    }
}
pub(crate) fn process_seed()->Result<u64,&'static str> {
    static SEED:std::sync::OnceLock<Result<u64,&'static str>>=std::sync::OnceLock::new();
    *SEED.get_or_init(||match std::env::var("AUGUR_SAMPLING_SEED") {
        Ok(v)=>parse_seed(Some(&v)),Err(std::env::VarError::NotPresent)=>parse_seed(None),
        _=>Err("AUGUR_SAMPLING_SEED must be a u64 decimal integer"),
    })
}
/// XOR the run seed into each purpose domain; seed=0 exactly retains the old stream.
pub(crate) fn uniform_seeded(seed:u64,domain:u64,position:usize)->f64 {uniform(seed^domain,position)}
pub(crate) fn process_uniform(domain:u64,position:usize)->Result<f64,&'static str> {
    Ok(uniform_seeded(process_seed()?,domain,position))
}
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub(crate) struct Selection {pub id:u32,pub top1:u32}
/// Uniform in [0,1), with separate counters for each absolute token position.
pub(crate) fn uniform(seed:u64,position:usize)->f64 {
    let mut x=seed.wrapping_add((position as u64).wrapping_add(1).wrapping_mul(0x9e3779b97f4a7c15));
    x=(x^(x>>30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x=(x^(x>>27)).wrapping_mul(0x94d049bb133111eb);
    x^=x>>31;
    ((x>>11) as f64)*(1.0/9007199254740992.0)
}
pub(crate) fn select(ids:&[u32],logits:&[f32],u:f64)->Result<Selection,&'static str> {
    if ids.len()!=TOP_K || logits.len()!=TOP_K || !(0.0..1.0).contains(&u) {return Err("invalid sampling shape/uniform");}
    for i in 0..TOP_K {
        if ids[i]>=248320 || !logits[i].is_finite() || ids[..i].contains(&ids[i]) {return Err("invalid sampling candidates");}
        if i>0 && (logits[i]>logits[i-1] || (logits[i]==logits[i-1] && ids[i]<ids[i-1])) {return Err("unsorted sampling candidates");}
    }
    let mut weights=[0.0;TOP_K];let mut total=0.0;
    for i in 0..TOP_K {weights[i]=(f64::from(logits[i])-f64::from(logits[0])).exp();total+=weights[i];}
    let mut retained=0.0;let mut end=0;
    while end<TOP_K {retained+=weights[end];end+=1;if retained>=TOP_P*total {break;}}
    let threshold=u*retained;let mut cumulative=0.0;
    for i in 0..end {cumulative+=weights[i];if threshold<cumulative || i+1==end {return Ok(Selection{id:ids[i],top1:ids[0]});}}
    Err("empty sampling support")
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn nucleus_keeps_crossing_token_and_has_expected_mass() {
        let ids:Vec<_>=(0..20).collect();let mut logits=[-100.0;20];
        logits[0]=0.5f32.ln();logits[1]=0.3f32.ln();logits[2]=0.16f32.ln();logits[3]=0.04f32.ln();
        let mut counts=[0usize;20];
        for i in 0..96000 {counts[select(&ids,&logits,(i as f64+0.5)/96000.0).unwrap().id as usize]+=1;}
        assert_eq!(&counts[..4],&[50000,30000,16000,0]);
        assert!(select(&ids,&logits,1.0).is_err());
        logits[19]=f32::NAN;assert!(select(&ids,&logits,0.1).is_err());
    }
    #[test] fn deterministic_proposal_equality_preserves_target_distribution() {
        // Independent enumeration: q is a point mass at token 1. Match is
        // accepted; mismatch uses the same target draw, never a fresh draw.
        let mut counts=[0usize;3];let mut accepted=0;
        for i in 0..10000 {let u=(i as f64+0.5)/10000.0;let target=if u<0.2 {0} else if u<0.7 {1} else {2};
            let draft=1;let output=if draft==target {accepted+=1;draft} else {target};counts[output]+=1;}
        assert_eq!(counts,[2000,5000,3000]);assert_eq!(accepted,5000);
    }
    #[test] fn seeds_are_validated_and_change_all_purpose_streams() {
        assert_eq!(parse_seed(None),Ok(0));assert_eq!(parse_seed(Some("18446744073709551615")),Ok(u64::MAX));
        for value in ["","-1","+1","1.0","true","18446744073709551616"," 1"] {assert!(parse_seed(Some(value)).is_err());}
        let domains=[0,crate::augur_speculative_sampling::PROPOSAL,crate::augur_speculative_sampling::ACCEPT,
            crate::augur_speculative_sampling::CORRECTION,crate::augur_speculative_sampling::BONUS];
        for domain in domains {for position in [0,512,32760,262000] {
            assert_eq!(uniform_seeded(0,domain,position),uniform(domain,position));
            assert_ne!(uniform_seeded(1,domain,position),uniform_seeded(2,domain,position));
            assert_ne!(uniform_seeded(0,domain,position),uniform_seeded(1,domain,position));
        }}
    }
    #[test] fn position_rng_survives_speculation_rollback_and_restore() {
        let serial:Vec<_>=(900..932).map(|p|uniform(0,p)).collect();
        for rejected in 1..8 {for _ in 0..7 {let _:Vec<_>=(900..908).map(|p|uniform(0,p)).collect();}
            assert_eq!(uniform(0,900+rejected),serial[rejected]);}
        assert!(serial.iter().all(|u|(0.0..1.0).contains(u)));
        assert_eq!(uniform(0,0).to_bits(),0.8833108082136426f64.to_bits());
        let mean=(0..100000).map(|p|uniform(0,p)).sum::<f64>()/100000.0;
        assert!((mean-0.5).abs()<0.005);
    }
}

/// Audit only committed/reachable selections, never unverified speculative tails.
/// The digest can be reconstructed from a turn receipt without any logits or
/// hidden state. It is scoped to one request and includes the pending EOS/token.
#[derive(Default)]
pub(crate) struct Trace {
    hash:sha2::Sha256,first:Option<usize>,last:Option<(usize,u32)>,count:usize,
    non_argmax:usize,top1_unknown:usize,
}
impl Trace {
    pub(crate) fn record(&mut self,position:usize,id:u32,top1:Option<u32>)->Result<(),&'static str> {
        use sha2::Digest;
        if let Some((previous,token))=self.last {
            if position==previous {return if id==token {Ok(())}else{Err("sampled pending token changed")};}
            if position!=previous+1 {return Err("sampled trajectory has a gap or rollback");}
        }
        if self.first.is_none(){self.first=Some(position);}
        self.hash.update((position as u64).to_le_bytes());self.hash.update(id.to_le_bytes());
        self.last=Some((position,id));self.count+=1;
        self.non_argmax+=usize::from(top1.is_some_and(|x|x!=id));self.top1_unknown+=usize::from(top1.is_none());Ok(())
    }
    pub(crate) fn detail(&self)->Value {
        use sha2::Digest;
        json!({"schema":"quasar-sampled-selection-trace-v1","scope":"request_reachable_selections_including_pending",
            "encoding":"concatenated_u64_le_absolute_position_u32_le_token_id",
            "first_position":self.first,"last_position":self.last.map(|x|x.0),"selections":self.count,
            "sha256":format!("{:x}",self.hash.clone().finalize()),"non_argmax_selections":self.non_argmax,
            "top1_unobserved_selections":self.top1_unknown,"uncommitted_speculative_tail_excluded":true})
    }
}
#[cfg(test)] mod trace_tests {
    use super::*;
    #[test] fn serial_and_committed_b8_have_identical_trajectory_digest() {
        let mut serial=Trace::default();let mut batched=Trace::default();
        for (i,id) in [10,11,12,88,15].into_iter().enumerate(){serial.record(1024+i,id,Some(10)).unwrap();}
        batched.record(1024,10,Some(10)).unwrap();
        // Two accepted drafts then a sampled rejection; unchecked tail omitted.
        for (i,id) in [11,12,88].into_iter().enumerate(){batched.record(1025+i,id,Some(10)).unwrap();}
        batched.record(1027,88,None).unwrap(); // next round returns cached pending token
        batched.record(1028,15,Some(10)).unwrap();
        assert_eq!(serial.detail(),batched.detail());
        assert!(batched.record(1028,99,None).is_err());
        assert!(batched.record(1030,15,None).is_err());
        assert!(batched.record(1026,12,None).is_err());
    }
}
