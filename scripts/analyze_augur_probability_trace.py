"""Independent oracle for sparse p/q traces; never treat speculative tail as labels."""
import hashlib,json,math,struct
from pathlib import Path
from audit_augur_sampling_trajectory import audit
from analyze_augur_sampling_rejections import nucleus,f32
MASK=(1<<64)-1
DOMAINS={'proposal':0x445241465450524f,'accept':0x414343455054414e,'correction':0x434f525245435449,'bonus':0x424f4e5553544f4b}
def uniform(purpose,position):
    x=(DOMAINS[purpose]+(position+1)*0x9e3779b97f4a7c15)&MASK
    x=((x^(x>>30))*0xbf58476d1ce4e5b9)&MASK
    x=((x^(x>>27))*0x94d049bb133111eb)&MASK
    return ((x^(x>>31))>>11)/9007199254740992

def draw(distribution,u):
    total=sum(distribution.values());assert total>0
    cumulative=0;last=None
    for token,p in distribution.items():
        assert p>=0 and math.isfinite(p)
        if p>0:last=token
        cumulative+=p
        if u*total<cumulative:return token
    return last

def verify(p,q,draft,position):
    assert q[draft]>0 and abs(sum(q.values())-1)<1e-12
    if uniform('accept',position)*q[draft]<p.get(draft,0) or p.get(draft,0)>=q[draft]:return draft
    corrected=draw({t:max(v-q.get(t,0),0) for t,v in p.items()},uniform('correction',position))
    assert corrected!=draft
    return corrected

BLOCK_CONTRACT='block_verification_sun2025_sparse_v1'
def block_verify(P,Q,drafts,start):
    """Sun et al. 2025 Algorithm 2. P[i] (i=0..gamma) target dicts, Q[i] (i<gamma) proposal dicts,
    drafts X_1..X_gamma at absolute positions start+1..start+gamma. Returns (tau, emitted token)."""
    g=len(drafts);pr=[1.0]
    for i in range(g):
        assert Q[i].get(drafts[i],0)>0 and abs(sum(Q[i].values())-1)<1e-12
        v=min(pr[-1]*P[i].get(drafts[i],0.0)/Q[i][drafts[i]],1.0);pr.append(1.0 if v>1.0-1e-12 else v)
    def res(i):return {t:max(pr[i]*v-Q[i].get(t,0.0),0.0) for t,v in P[i].items()}
    tau=0
    for i in range(g,0,-1):
        if i==g:h=pr[g]
        else:
            r=sum(res(i).values());den=r+1-pr[i];h=r/den if den>0 else 0.0
        if uniform('accept',start+i)<h:tau=i;break
    if tau==g:return tau,draw(P[g],uniform('bonus',start+g+1))
    w=res(tau)
    if sum(w.values())<=0:w=P[tau]  # zero-probability rounding branch, mirrors the runtime
    return tau,draw(w,uniform('correction',start+tau+1))

DEFAULT={'temperatures':[1.0]*7,'pair_scales':[1.0]*7}
def normalize_calibration(c):
    """Accept per-position {'temperatures','pair_scales'} or legacy pos2-4 {'temperature','pair_scale'}."""
    if c is None:return DEFAULT
    if 'temperatures' in c:
        t=[f32(x) for x in c['temperatures']];s=[f32(x) for x in c['pair_scales']]
    else:
        t=[1.0]+[f32(c['temperature'])]*3+[1.0]*3;s=[1.0]+[f32(c['pair_scale'])]*3+[1.0]*3
    assert len(t)==7 and len(s)==7 and s[0]==1.0
    assert all(math.isfinite(x) and .5<=x<=1.5 for x in t) and all(math.isfinite(x) and 0<=x<=2 for x in s)
    return {'temperatures':t,'pair_scales':s}

def proposal_distribution(scores,slot,calibration):
    """scores already contain the calibrated pair term; only temperature is applied here."""
    assert len(scores)==16 and 1<=slot<=7
    temp=normalize_calibration(calibration)['temperatures'][slot-1]
    weights=[math.exp((v-max(scores))/temp) for v in scores];total=sum(weights)
    return [w/total for w in weights]

def analyze(trace_path,receipt_path):
    receipt=json.loads(Path(receipt_path).read_bytes());identity=audit(receipt)
    assert identity['native_selection_trace_verified']
    contract=receipt['decode_detail']['sampling']['verification']
    assert contract in ('sparse_p_over_q_positive_residual_v1',BLOCK_CONTRACT);block=contract==BLOCK_CONTRACT
    trace=[json.loads(l) for l in Path(trace_path).read_text(encoding='utf8').splitlines() if l]
    d=receipt['decode_detail']['dflash'];assert len(trace)==d['rounds']
    output=receipt['output_token_ids'];selected=output if receipt['session']['pending_token_published'] else output+[receipt['session']['pending_token']]
    expected=dict(enumerate(selected,identity['prompt_tokens']));previous=None;previous_round=None
    buckets=[[0,0,0,0] for _ in range(7)];checked=0;rejections=0;maximum_q_error=0
    for row in trace:
        assert row['schema']=='quasar-probability-rejection-trace-v2'
        sampling=receipt['decode_detail']['sampling']
        raw=row.get('proposal_calibration',row.get('proposal_pos234'))
        calibration=normalize_calibration(raw)
        assert calibration==normalize_calibration(sampling.get('proposal_calibration',sampling.get('proposal_pos234')))
        assert row['probability_positions']==7 or calibration==DEFAULT
        cursor=row['start_consumed'];path=row['path'];emitted=row['output_tokens'];rejected=row['first_rejection']
        if previous is not None:assert cursor==previous and row['round']==previous_round+1
        previous=row['end_consumed'];previous_round=row['round']
        assert all(expected[cursor+i]==t for i,t in enumerate(emitted))
        slots=list(range(1,len(emitted)+(1 if rejected else 0)));assert row['canonical_draft_slots']==slots
        reject_slot=rejected['offset'] if rejected else None
        if block:
            assert row['probability_positions']==7
            Ps=[nucleus(r['ids'],r['logits']) for r in row['target_rows']];Qs=[]
            for slot in range(1,8):
                ids=row['draft_candidates'][slot*16:(slot+1)*16];scores=[f32(x) for x in row['draft_transitions'][slot*16:(slot+1)*16]]
                actual=row['proposal_q'][slot*16:(slot+1)*16];probs=proposal_distribution(scores,slot,calibration)
                error=max(abs(a-b) for a,b in zip(actual,probs));maximum_q_error=max(error,maximum_q_error)
                assert error<2e-12 and path[slot]==draw(dict(zip(ids,probs)),uniform('proposal',cursor+slot))
                Qs.append({t:v for t,v in zip(ids,actual) if v>0})
            tau,nxt=block_verify(Ps,Qs,path[1:],cursor)
            assert row['verified_ids'][:tau]==path[1:tau+1] and row['verified_ids'][tau]==nxt
            # EOS/budget can stop publication earlier; otherwise the boundary must be exactly tau.
            limit=len(emitted)-1 if not rejected else rejected['offset']-1
            assert limit<=tau and (rejected is None or rejected['offset']==tau+1)
            for slot in range(1,8):
                buckets[slot-1][0 if slot<len(emitted) else 1 if slot==reject_slot else 2 if rejected else 3]+=1
                if slot in slots:checked+=1;assert expected[cursor+slot]==row['verified_ids'][slot-1]
            if row['end_consumed']==cursor+8:assert expected[cursor+8]==nxt and tau==7
            if rejected:
                rejections+=1;assert rejected['offset']==len(emitted) and expected[cursor+reject_slot]==rejected['correct_target_token']
            continue
        for slot in range(1,8):
            buckets[slot-1][0 if slot<len(emitted) else 1 if slot==reject_slot else 2 if rejected else 3]+=1
            ids=row['draft_candidates'][slot*16:(slot+1)*16]
            scores=[f32(x) for x in row['draft_transitions'][slot*16:(slot+1)*16]]
            actual=row['proposal_q'][slot*16:(slot+1)*16];assert len(set(ids))==16
            best=max(range(16),key=lambda i:scores[i])
            if slot<=row['probability_positions']:
                probs=proposal_distribution(scores,slot,calibration)
                proposal=draw(dict(zip(ids,probs)),uniform('proposal',cursor+slot))
            else:
                probs=[float(i==best) for i in range(16)];proposal=ids[best]
            error=max(abs(a-b) for a,b in zip(actual,probs));maximum_q_error=max(error,maximum_q_error)
            assert error<2e-12 and path[slot]==proposal
            trow=row['target_rows'][slot-1];p=nucleus(trow['ids'],trow['logits']);q=dict(zip(ids,actual))
            verified=verify(p,q,proposal,cursor+slot);assert row['verified_ids'][slot-1]==verified
            if slot in slots:
                checked+=1;assert expected[cursor+slot]==verified
                assert (slot<len(emitted))==(verified==proposal)
        # Bonus is only canonical if every proposal has been accepted and committed.
        trow=row['target_rows'][-1];bonus=draw(nucleus(trow['ids'],trow['logits']),uniform('bonus',cursor+8))
        assert row['verified_ids'][-1]==bonus
        if row['end_consumed']==cursor+8:assert expected[cursor+8]==bonus
        if rejected:
            rejections+=1;assert rejected['offset']==len(emitted)
            assert expected[cursor+reject_slot]==rejected['correct_target_token']
    assert buckets==d['by_position_accepted_rejected_unchecked_censored']
    assert checked==d['checked_draft_tokens'] and sum(x[0] for x in buckets)==d['accepted_draft_tokens']
    return dict(schema='augur-probability-oracle-v1',status='PASS',verification=contract,rounds=len(trace),checked=checked,first_rejections=rejections,
        by_position=buckets,max_q_absolute_error=maximum_q_error,commit_pending_alignment_verified=True,
        raw_trace_sha256=hashlib.sha256(Path(trace_path).read_bytes()).hexdigest(),diagnostic_TPS_not_performance=True)
