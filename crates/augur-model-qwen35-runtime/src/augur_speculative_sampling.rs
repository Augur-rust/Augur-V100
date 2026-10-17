//! Sparse speculative rejection sampling. The proposal carries its actual q;
//! target top-k/top-p remains unchanged. No rejected draft token enters state.
use crate::augur_sampling::{self,Selection,TOP_K,TOP_P};
pub(crate) const PROPOSAL:u64=0x445241465450524f;
pub(crate) const ACCEPT:u64=0x414343455054414e;
pub(crate) const CORRECTION:u64=0x434f525245435449;
pub(crate) const BONUS:u64=0x424f4e5553544f4b;
pub(crate) const CONTRACT:&str="sparse_p_over_q_positive_residual_v1";
pub(crate) fn positions_from_env()->Result<usize,Box<dyn std::error::Error>> {
    match std::env::var("AUGUR_DFLASH_PROBABILITY_PREFIX").as_deref() {
        Err(std::env::VarError::NotPresent)|Ok("0")=>Ok(0),
        Ok("1")=>Ok(1),Ok("3")=>Ok(3),Ok("7")=>Ok(7),
        _=>Err("probability prefix must be 0, 1, 3 or 7".into()),
    }
}
/// Proposal calibration is not target temperature and never changes gate weights.
/// Row k (1..=7) uses temperatures[k-1] and pair_scales[k-1]; row1 has no draft predecessor.
#[derive(Debug,Clone,Copy,PartialEq,serde::Serialize)]
pub(crate) struct ProposalCalibration {pub temperatures:[f32;7],pub pair_scales:[f32;7]}
impl ProposalCalibration {
    pub const DEFAULT:Self=Self{temperatures:[1.0;7],pair_scales:[1.0;7]};
    pub fn pos234(temperature:f32,pair_scale:f32)->Self {
        let mut c=Self::DEFAULT;for i in 1..4 {c.temperatures[i]=temperature;c.pair_scales[i]=pair_scale;}c
    }
    fn checked(self,positions:usize)->Result<Self,&'static str> {
        let bounded=self.temperatures.iter().all(|t|t.is_finite() && (0.5..=1.5).contains(t))
            && self.pair_scales.iter().all(|s|s.is_finite() && (0.0..=2.0).contains(s));
        if !bounded || self.pair_scales[0]!=1.0 || (self!=Self::DEFAULT && positions!=7) {
            return Err("invalid proposal calibration; requires bounded values, row1 pair scale 1 and all7 probability proposals");
        }
        Ok(self)
    }
    /// Per-position lists take precedence; the legacy pos2-4 pair remains accepted but cannot be mixed.
    pub fn from_env(positions:usize)->Result<Self,Box<dyn std::error::Error>> {
        fn text(name:&str)->Result<Option<String>,Box<dyn std::error::Error>> {
            match std::env::var(name) {Ok(v)=>Ok(Some(v)),Err(std::env::VarError::NotPresent)=>Ok(None),Err(e)=>Err(e.into())}
        }
        fn list(v:&str)->Result<[f32;7],Box<dyn std::error::Error>> {
            let x:Vec<f32>=v.split(',').map(|x|x.trim().parse()).collect::<Result<_,_>>()?;
            x.try_into().map_err(|_|"proposal calibration list must have exactly seven values".into())
        }
        let temps=text("AUGUR_DFLASH_PROBABILITY_TEMPERATURES")?;let scales=text("AUGUR_DFLASH_PROBABILITY_PAIR_SCALES")?;
        let t234=text("AUGUR_DFLASH_PROBABILITY_POS234_TEMPERATURE")?;let s234=text("AUGUR_DFLASH_PROBABILITY_POS234_PAIR_SCALE")?;
        let c=if temps.is_some() || scales.is_some() {
            if t234.is_some() || s234.is_some() {return Err("per-position and pos234 proposal calibration cannot be mixed".into());}
            Self{temperatures:temps.as_deref().map(list).transpose()?.unwrap_or([1.0;7]),
                pair_scales:scales.as_deref().map(list).transpose()?.unwrap_or([1.0;7])}
        } else {
            Self::pos234(t234.as_deref().map(str::parse).transpose()?.unwrap_or(1.0),s234.as_deref().map(str::parse).transpose()?.unwrap_or(1.0))
        };
        Ok(c.checked(positions)?)
    }
}
#[derive(Debug)]
pub(crate) struct Proposal {pub ids:Vec<u32>,pub q:Vec<f64>,pub start:usize,pub positions:usize,pub calibration:ProposalCalibration}
impl Proposal {
    pub fn validate(&self,path:&[u32])->Result<(),&'static str> {
        self.calibration.checked(self.positions)?;
        if path.len()!=8 || self.ids.len()!=128 || self.q.len()!=128 || !(1..=7).contains(&self.positions)
            || self.start.checked_add(8).is_none() {return Err("invalid probability proposal shape");}
        for row in 1..8 {
            let ids=&self.ids[row*16..(row+1)*16];let q=&self.q[row*16..(row+1)*16];
            validate_q(ids,q,path[row])?;
            if row>self.positions && q[ids.iter().position(|id|*id==path[row]).unwrap()]!=1.0 {
                return Err("deterministic proposal tail is not one-hot");
            }
        }
        Ok(())
    }
}
fn validate_q(ids:&[u32],q:&[f64],draft:u32)->Result<f64,&'static str> {
    if ids.len()!=q.len() || ids.is_empty() {return Err("invalid q shape");}
    for (i,(&id,&prob)) in ids.iter().zip(q).enumerate() {
        if id>=248320 || ids[..i].contains(&id) || !prob.is_finite() || !(0.0..=1.0).contains(&prob) {return Err("invalid q support");}
    }
    if (q.iter().sum::<f64>()-1.0).abs()>1e-12 {return Err("q must be normalized");}
    let qd=ids.iter().position(|id|*id==draft).map(|i|q[i]).unwrap_or(0.0);
    if qd<=0.0 {return Err("proposed token has zero q");}Ok(qd)
}
fn draw(ids:&[u32],weights:&[f64],u:f64)->Result<u32,&'static str> {
    if ids.len()!=weights.len() || !(0.0..1.0).contains(&u) {return Err("invalid sparse draw");}
    let total: f64=weights.iter().sum();
    if !total.is_finite() || total<=0.0 || weights.iter().any(|p|!p.is_finite() || *p<0.0) {return Err("empty/nonfinite residual");}
    let threshold=u*total;let mut sum=0.0;let mut last=None;
    for (&id,&w) in ids.iter().zip(weights) {if w>0.0 {last=Some(id);}sum+=w;if threshold<sum {return Ok(id);}}
    last.ok_or("empty sparse support")
}
/// Target probabilities use precisely the baseline's top-p crossing rule.
fn target(ids:&[u32],logits:&[f32])->Result<[f64;TOP_K],&'static str> {
    augur_sampling::select(ids,logits,0.0)?;
    let mut weights=[0.0;TOP_K];let mut total=0.0;
    for i in 0..TOP_K {weights[i]=(f64::from(logits[i])-f64::from(logits[0])).exp();total+=weights[i];}
    let mut retained=0.0;let mut end=0;
    while end<TOP_K {retained+=weights[end];end+=1;if retained>=TOP_P*total {break;}}
    for (i,w) in weights.iter_mut().enumerate(){*w=if i<end{*w/retained}else{0.0};}Ok(weights)
}
fn verify_sparse(ids:&[u32],p:&[f64],q_ids:&[u32],q:&[f64],draft:u32,accept_u:f64,correction_u:f64)->Result<u32,&'static str> {
    if ids.len()!=p.len() || ids.is_empty() || !(0.0..1.0).contains(&accept_u) || !(0.0..1.0).contains(&correction_u)
        || p.iter().any(|x|!x.is_finite() || *x<0.0) || (p.iter().sum::<f64>()-1.0).abs()>1e-12 {return Err("invalid p or verifier uniform");}
    let qd=validate_q(q_ids,q,draft)?;
    let pd=ids.iter().position(|id|*id==draft).map(|i|p[i]).unwrap_or(0.0);
    if pd>=qd || accept_u*qd<pd {return Ok(draft);}
    let residual:Vec<_>=ids.iter().zip(p).map(|(&id,&pt)| {
        let qt=q_ids.iter().position(|x|*x==id).map(|i|q[i]).unwrap_or(0.0);(pt-qt).max(0.0)
    }).collect();
    let corrected=draw(ids,&residual,correction_u)?;
    if corrected==draft {return Err("rejection correction cannot select the rejected draft");}Ok(corrected)
}
pub(crate) fn verify(ids:&[u32],logits:&[f32],proposal:&Proposal,row:usize,draft:u32,position:usize)->Result<Selection,&'static str> {
    if !(1..8).contains(&row) || proposal.start.checked_add(row)!=Some(position) {return Err("foreign probability position");}
    let probabilities=target(ids,logits)?;
    let id=verify_sparse(ids,&probabilities,&proposal.ids[row*16..(row+1)*16],&proposal.q[row*16..(row+1)*16],draft,
        augur_sampling::uniform(ACCEPT,position),augur_sampling::uniform(CORRECTION,position))?;
    Ok(Selection{id,top1:ids[0]})
}
/// Block verification (Sun et al., ICLR 2025, Algorithm 2) is opt-in; token verification stays default.
pub(crate) const BLOCK_CONTRACT:&str="block_verification_sun2025_sparse_v1";
pub(crate) fn block_from_env()->Result<bool,Box<dyn std::error::Error>> {
    match std::env::var("AUGUR_DFLASH_VERIFICATION").as_deref() {
        Err(std::env::VarError::NotPresent)|Ok("token")=>Ok(false),
        Ok("block")=>Ok(true),
        _=>Err("AUGUR_DFLASH_VERIFICATION must be token or block".into()),
    }
}
fn sparse_mass(q_ids:&[u32],q:&[f64],id:u32)->f64 {q_ids.iter().position(|x|*x==id).map(|i|q[i]).unwrap_or(0.0)}
type Row<'a>=(&'a [u32],&'a [f64]);
const BLOCK_SNAP:f64=1e-12;
/// Residual draw; an empty residual has exact probability zero and can only be reached through
/// floating-point rounding, so it falls back to the target conditional (error is O(rounding)).
fn block_residual_draw(row:Row,q:Row,scale:f64,u:f64)->Result<u32,&'static str> {
    let w=block_residual(row,q,scale);
    if w.iter().sum::<f64>()>0.0 {draw(row.0,&w,u)} else {draw(row.0,row.1,u)}
}
/// Residual weights max(scale*P - Q, 0) over P's support (tokens outside P only lose mass).
fn block_residual(row:Row,q:Row,scale:f64)->Vec<f64> {
    row.0.iter().zip(row.1).map(|(&id,&pt)|(scale*pt-sparse_mass(q.0,q.1,id)).max(0.0)).collect()
}
/// Block terms for one drafted path. rows[r] (r in 0..=gamma) is the target nucleus at the
/// position of draft r+1 (rows[gamma] is the bonus position); q[r] is the proposal for draft r+1.
/// Returns (p_0..p_gamma, h_0..h_gamma) with p_i=min(p_{i-1}P(X_i)/Q(X_i),1), h_0=1,
/// h_gamma=p_gamma and h_i=r_i/(r_i+1-p_i) where r_i is the residual mass (0/0 -> 0).
fn block_terms(rows:&[Row],q:&[Row],drafts:&[u32])->Result<(Vec<f64>,Vec<f64>),&'static str> {
    let gamma=drafts.len();
    if gamma==0 || rows.len()!=gamma+1 || q.len()!=gamma {return Err("invalid block verification shape");}
    for (ids,p) in rows {
        if ids.len()!=p.len() || ids.is_empty() || p.iter().any(|x|!x.is_finite() || *x<0.0)
            || (p.iter().sum::<f64>()-1.0).abs()>1e-12 {return Err("invalid block target row");}
    }
    let mut pr=vec![1.0f64;gamma+1];
    for i in 0..gamma {
        let qd=validate_q(q[i].0,q[i].1,drafts[i])?;
        let pd=rows[i].0.iter().position(|id|*id==drafts[i]).map(|k|rows[i].1[k]).unwrap_or(0.0);
        // Snap rounding of an exact 1 (e.g. (a/b)*(b/a)) so zero-probability branches stay unreachable.
        let v=(pr[i]*pd/qd).min(1.0);pr[i+1]=if v>1.0-BLOCK_SNAP {1.0} else {v};
    }
    let mut h=vec![1.0f64;gamma+1];h[gamma]=pr[gamma];
    for i in 1..gamma {
        let r:f64=block_residual(rows[i],q[i],pr[i]).iter().sum();let den=r+1.0-pr[i];
        h[i]=if den>0.0 {r/den} else {0.0};
    }
    Ok((pr,h))
}
/// Scan i=gamma..1 and accept the longest sub-block whose coin succeeds (i=0 always succeeds),
/// then emit one residual (tau<gamma) or bonus (tau=gamma) token drawn with `next_u(tau)`.
fn block_decide(rows:&[Row],q:&[Row],drafts:&[u32],accept_u:&[f64],next_u:impl Fn(usize)->f64)
    ->Result<(usize,u32),&'static str> {
    let gamma=drafts.len();
    if accept_u.len()!=gamma || accept_u.iter().any(|u|!(0.0..1.0).contains(u)) {return Err("invalid block coins");}
    let (pr,h)=block_terms(rows,q,drafts)?;
    let tau=(1..=gamma).rev().find(|&i|accept_u[i-1]<h[i]).unwrap_or(0);
    let u=next_u(tau);
    if !(0.0..1.0).contains(&u) {return Err("invalid block next uniform");}
    let next=if tau==gamma {draw(rows[gamma].0,rows[gamma].1,u)?} else {block_residual_draw(rows[tau],q[tau],pr[tau],u)?};
    Ok((tau,next))
}
/// Verify a full B8 probability path with block verification. Returns tau (accepted drafts) and
/// eight selections: 0..tau are the accepted drafts, tau is the residual/bonus token, later rows
/// repeat it only to keep shape and are never committed. The residual token may equal the
/// rejected draft (block residuals are not disjoint from it), so callers must use tau, never
/// draft/selection equality, to locate the commit boundary.
pub(crate) fn verify_block(target_rows:&[(&[u32],&[f32])],proposal:&Proposal,path:&[u32])->Result<(usize,Vec<Selection>),&'static str> {
    let width=path.len();
    if width!=8 || target_rows.len()!=width {return Err("block verification requires B8");}
    if proposal.positions!=7 {return Err("block verification requires all7 probability proposals");}
    proposal.validate(path)?;
    let probs:Vec<[f64;TOP_K]>=target_rows.iter().map(|(ids,logits)|target(ids,logits)).collect::<Result<_,_>>()?;
    let rows:Vec<(&[u32],&[f64])>=target_rows.iter().zip(&probs).map(|((ids,_),p)|(*ids,&p[..])).collect();
    let q:Vec<(&[u32],&[f64])>=(1..width).map(|r|(&proposal.ids[r*16..(r+1)*16],&proposal.q[r*16..(r+1)*16])).collect();
    let drafts=&path[1..];
    // Coins reuse the ACCEPT domain at each draft's absolute position. The single emitted draw uses
    // CORRECTION (residual) or BONUS (full block) at its absolute position, like token verification.
    let accept_u:Vec<f64>=(1..width).map(|r|augur_sampling::uniform(ACCEPT,proposal.start+r)).collect();
    let start=proposal.start;let gamma=width-1;
    let (tau,next)=block_decide(&rows,&q,drafts,&accept_u,|tau|augur_sampling::uniform(if tau==gamma {BONUS} else {CORRECTION},start+tau+1))?;
    let out=(0..width).map(|r|Selection{id:if r<tau {drafts[r]} else {next},top1:target_rows[r].0[0]}).collect();
    Ok((tau,out))
}
#[cfg(test)] mod tests {
    use super::*;
    type Table=std::collections::HashMap<Vec<u32>,Vec<f64>>;
    fn extend(p:&Table,prefix:Vec<u32>,len:usize,mass:f64,out:&mut std::collections::HashMap<Vec<u32>,f64>) {
        if prefix.len()==len {*out.entry(prefix).or_default()+=mass;return;}
        for (t,&pt) in p[&prefix].iter().enumerate() {if pt>0.0 {let mut x=prefix.clone();x.push(t as u32);extend(p,x,len,mass*pt,out);}}
    }
    /// Exact law of block verification over all drafts, coin outcomes and residual draws,
    /// extended by target sampling to length gamma+1, must equal the target joint law.
    #[test] fn block_verification_preserves_target_joint_law_exactly() {
        const V:usize=3;let ids:Vec<u32>=(0..V as u32).collect();
        let mut seed=0x1234_5678_9abc_def0u64;
        let mut rand=move||{seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;seed};
        for case in 0..90 {
            let gamma=1+case%3;let (mut p,mut q)=(Table::new(),Table::new());
            let mut stack=vec![vec![]];
            while let Some(c)=stack.pop() {
                let mut dist=||{let mut w:Vec<f64>=(0..V).map(|_|(rand()%10) as f64).collect();
                    if w.iter().sum::<f64>()==0.0 {w[0]=1.0;}let z:f64=w.iter().sum();w.into_iter().map(|x|x/z).collect::<Vec<_>>()};
                let pd=dist();let mut qd=dist();
                match rand()%5 {0=>qd=pd.clone(),1=>{let k=(rand()%V as u64) as usize;qd=(0..V).map(|i|f64::from(u8::from(i==k))).collect();},_=>{}}
                if c.len()<gamma {for t in 0..V as u32 {let mut d=c.clone();d.push(t);stack.push(d);}}
                p.insert(c.clone(),pd);q.insert(c,qd);
            }
            let mut drafts=vec![vec![]];
            for _ in 0..gamma {drafts=drafts.into_iter().flat_map(|d:Vec<u32>|(0..V as u32).map(move|t|{let mut x=d.clone();x.push(t);x})).collect();}
            let mut law=std::collections::HashMap::new();
            for d in drafts {
                let dm:f64=(0..gamma).map(|i|q[&d[..i].to_vec()][d[i] as usize]).product();
                if dm==0.0 {continue;}
                let rows:Vec<Row>=(0..=gamma).map(|i|(&ids[..],&p[&d[..i].to_vec()][..])).collect();
                let qs:Vec<Row>=(0..gamma).map(|i|(&ids[..],&q[&d[..i].to_vec()][..])).collect();
                let (pr,h)=block_terms(&rows,&qs,&d).unwrap();
                let mut remaining=dm;
                for tau in (0..=gamma).rev() {
                    let take=remaining*h[tau];remaining-=take;if take==0.0 {continue;}
                    let mut w=if tau==gamma {p[&d].clone()} else {block_residual(rows[tau],qs[tau],pr[tau])};
                    if w.iter().sum::<f64>()<=0.0 {assert!(take<1e-12,"empty residual with real mass {take}");w=p[&d[..tau].to_vec()].clone();}
                    let z:f64=w.iter().sum();
                    for (t,&wt) in w.iter().enumerate() {if wt>0.0 {let mut x=d[..tau].to_vec();x.push(t as u32);extend(&p,x,gamma+1,take*wt/z,&mut law);}}
                }
                assert!(remaining.abs()<1e-15);
            }
            let mut joint=std::collections::HashMap::new();extend(&p,vec![],gamma+1,1.0,&mut joint);
            let keys:std::collections::BTreeSet<_>=joint.keys().chain(law.keys()).cloned().collect();
            for k in keys {let (a,b)=(joint.get(&k).copied().unwrap_or(0.0),law.get(&k).copied().unwrap_or(0.0));assert!((a-b).abs()<1e-12,"case {case} {k:?} {a} {b}");}
        }
    }
    #[test] fn block_decide_accepts_longest_and_fails_closed() {
        let ids=[0u32,1];let p=[0.5,0.5];let q=[0.5,0.5];
        let rows:Vec<Row>=vec![(&ids,&p);4];let qs:Vec<Row>=vec![(&ids,&q);3];
        // p==q: every p_i=1 and h_gamma=1, so the whole block is always accepted with a bonus draw.
        for u in [0.0,0.3,0.999] {let (tau,_)=block_decide(&rows,&qs,&[0,1,0],&[u,u,u],|t|{assert_eq!(t,3);0.2}).unwrap();assert_eq!(tau,3);}
        // A draft with zero target mass makes p_3=0, so tau<3 regardless of coins.
        let pz=[1.0,0.0];let rows2:Vec<Row>=vec![(&ids,&p),(&ids,&p),(&ids,&pz),(&ids,&p)];
        for u in [0.0,0.5,0.99] {let (tau,next)=block_decide(&rows2,&qs,&[0,1,1],&[u,u,u],|_|0.9).unwrap();assert!(tau<3);if tau==2 {assert_eq!(next,0);}}
        assert!(block_decide(&rows,&qs,&[0,1,0],&[0.1,1.0,0.1],|_|0.1).is_err());
        assert!(block_decide(&rows,&qs,&[0,1,0],&[0.1,0.1,0.1],|_|1.0).is_err());
        assert!(block_decide(&rows,&qs,&[0,1],&[0.1,0.1],|_|0.1).is_err());
    }
    #[test] fn proposal_calibration_is_bounded_and_cannot_silently_affect_deterministic_tail() {
        for temperature in [0.5,0.875,1.0,1.5] {for pair_scale in [0.0,1.0,1.5,2.0] {
            assert!(ProposalCalibration::pos234(temperature,pair_scale).checked(7).is_ok());
        }}
        for temperature in [f32::NAN,f32::INFINITY,0.0,0.49,1.51] {assert!(ProposalCalibration::pos234(temperature,1.0).checked(7).is_err());}
        for pair_scale in [f32::NAN,f32::INFINITY,-0.01,2.01] {assert!(ProposalCalibration::pos234(1.0,pair_scale).checked(7).is_err());}
        for positions in [0,1,3] {assert!(ProposalCalibration::pos234(0.875,1.5).checked(positions).is_err());}
        let tiered=ProposalCalibration{temperatures:[0.875,0.875,0.75,0.625,0.625,0.5,0.625],pair_scales:[1.0,1.5,1.5,1.5,1.5,1.5,1.5]};
        assert!(tiered.checked(7).is_ok());
        let mut bad=tiered;bad.pair_scales[0]=1.5;assert!(bad.checked(7).is_err());
        for i in 0..7 {let mut bad=tiered;bad.temperatures[i]=0.49;assert!(bad.checked(7).is_err());}
        assert!(ProposalCalibration::DEFAULT.checked(1).is_ok());
    }
    #[test] fn proposal_contract_rejects_bad_cursor_and_non_onehot_tail() {
        let path=[200000,1,2,3,4,5,6,7];
        let mut proposal=Proposal{ids:(0..8).flat_map(|_|0..16u32).collect(),q:vec![0.0;128],start:100,positions:1,calibration:ProposalCalibration::DEFAULT};
        for row in 1..8 {proposal.q[row*16+row]=1.0;}
        proposal.validate(&path).unwrap();
        proposal.q[32+2]=0.5;proposal.q[32+3]=0.5;assert!(proposal.validate(&path).is_err());
        proposal.q[32+2]=1.0;proposal.q[32+3]=0.0;proposal.start=usize::MAX;assert!(proposal.validate(&path).is_err());
        proposal.start=100;proposal.positions=0;assert!(proposal.validate(&path).is_err());
        proposal.positions=1;proposal.ids.pop();assert!(proposal.validate(&path).is_err());
    }
    #[test] fn sparse_rejection_preserves_target_with_partial_support() {
        // Exact discrete quadrature, independent expected p. q includes a token
        // absent from p, and p includes a token absent from q.
        let ids=[0,1,2];let p=[0.2,0.5,0.3];let qi=[0,1,3];let q=[0.6,0.2,0.2];
        let mut count=[0;4];let mut accepted=0;
        for a in 0..100 {for b in 0..300 {for c in 0..30 {
            let d=if a<60{0}else if a<80{1}else{3};
            let out=verify_sparse(&ids,&p,&qi,&q,d,(b as f64+0.5)/300.0,(c as f64+0.5)/30.0).unwrap();
            count[out as usize]+=1;accepted+=usize::from(out==d);
        }}}
        assert_eq!(count,[180000,450000,270000,0]);assert_eq!(accepted,360000);
    }
    #[test] fn equal_and_deterministic_q_and_disjoint_support() {
        for u in [0.0,0.2,0.999999] {
            for d in [0,1] {assert_eq!(verify_sparse(&[0,1],&[0.4,0.6],&[0,1],&[0.4,0.6],d,u,u).unwrap(),d);}
            assert_eq!(verify_sparse(&[2],&[1.0],&[1],&[1.0],1,u,u).unwrap(),2);
            assert_eq!(verify_sparse(&[0,1],&[0.4,0.6],&[0],&[1.0],0,0.9,u).unwrap(),1);
        }
    }
    #[test] fn probabilities_fail_closed() {
        for q in [[0.0,0.0],[f64::NAN,1.0],[-0.1,1.1],[0.1,0.8]] {
            assert!(verify_sparse(&[0,1],&[0.5,0.5],&[0,1],&q,0,0.4,0.4).is_err());
        }
        assert!(verify_sparse(&[0,1],&[0.5,0.5],&[0,0],&[0.5,0.5],0,0.4,0.4).is_err());
        assert!(verify_sparse(&[0,1],&[0.5,0.5],&[0,1],&[0.0,1.0],0,0.4,0.4).is_err());
    }
    #[test] fn target_nucleus_matches_baseline_and_rng_domains_are_separate() {
        let ids:Vec<u32>=(0..20).collect();let mut logits=[-100.0;20];
        for (i,w) in [0.5f32,0.3,0.16,0.04].iter().enumerate(){logits[i]=w.ln();}
        let p=target(&ids,&logits).unwrap();assert_eq!(p[3],0.0);
        for i in 0..9600 {let u=(i as f64+0.5)/9600.0;assert_eq!(draw(&ids,&p,u).unwrap(),augur_sampling::select(&ids,&logits,u).unwrap().id);}
        for pos in 0..1024 {let us=[PROPOSAL,ACCEPT,CORRECTION,BONUS].map(|s|augur_sampling::uniform(s,pos));
            for i in 0..4 {assert!((0.0..1.0).contains(&us[i]));assert!(!us[..i].contains(&us[i]));}}
    }
}
