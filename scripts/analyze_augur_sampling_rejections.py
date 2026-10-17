"""Audit sampled DFlash rejection evidence without evaluating speculative tails.

Offline p/q overlaps below are conditional diagnostics on the observed path,
not acceptance/TPS predictions for a changed rollout. Current q is a point mass.
"""
import argparse, collections, hashlib, json, math, struct
from pathlib import Path
from audit_augur_sampling_trajectory import audit

CLOSE_THINK = 248069
TEMPERATURES = (0.5, 1.0, 2.0)

def uniform(position):
    mask = (1 << 64) - 1
    x = ((position + 1) * 0x9e3779b97f4a7c15) & mask
    x = ((x ^ (x >> 30)) * 0xbf58476d1ce4e5b9) & mask
    x = ((x ^ (x >> 27)) * 0x94d049bb133111eb) & mask
    return ((x ^ (x >> 31)) >> 11) / (1 << 53)

def f32(value):
    # serde f32 decimal JSON must be rounded back to binary32 before f64 math.
    return struct.unpack("<f", struct.pack("<f", value))[0]

def nucleus(ids, logits):
    logits = [f32(x) for x in logits]
    if len(ids) != 20 or len(set(ids)) != 20 or len(logits) != 20 or any(type(t) is not int or not 0 <= t < 248320 for t in ids):
        raise ValueError('invalid target top20 shape')
    if any(not math.isfinite(x) for x in logits): raise ValueError('nonfinite logits')
    if list(zip(ids, logits)) != sorted(zip(ids, logits), key=lambda x: (-x[1], x[0])):
        raise ValueError('target ordering')
    w = [math.exp(x - logits[0]) for x in logits]
    total = sum(w); retained = 0; end = 0
    for v in w:
        retained += v; end += 1
        if retained >= .95 * total: break
    return dict(zip(ids[:end], (x / retained for x in w[:end])))

def draw(p, u):
    cumulative = 0
    for token, mass in p.items():
        cumulative += mass
        if u < cumulative: return token
    return next(reversed(p))

def softmax(ids, scores, temperature):
    w = [math.exp((x - max(scores)) / temperature) for x in scores]
    z = sum(w)
    return dict(zip(ids, (x / z for x in w)))

def classify(draft, target, top1, candidates):
    if draft == target: return 'accepted'
    if target not in candidates: return 'sampled_target_missing_from_draft_top16'
    if draft == top1: return 'sampled_divergence_with_draft_at_target_top1'
    return 'sampled_target_in_candidates_but_selector_chose_other_non_top1'

def mean(xs): return sum(xs) / len(xs) if xs else None

def summarize(rows):
    rejected = [x for x in rows if not x['accepted']]
    return {
        'checked': len(rows), 'accepted': sum(x['accepted'] for x in rows), 'first_rejections': len(rejected),
        'observed_conditional_hit_rate': mean([x['accepted'] for x in rows]),
        'first_rejection_categories': dict(collections.Counter(x['category'] for x in rejected)),
        'first_rejection_sample_equals_target_top1': sum(x['sample_is_top1'] for x in rejected),
        'first_rejection_target_top1_in_draft_candidates': sum(x['top1_in_candidates'] for x in rejected),
        'first_rejection_draft_is_target_top1': sum(x['draft_is_top1'] for x in rejected),
        'sampled_target_unary_rank_histogram':dict(sorted(collections.Counter(str(x.get('sample_unary_rank')) for x in rejected).items())),
        'sampled_target_selector_rank_histogram': dict(sorted(collections.Counter(str(x['sample_selector_rank']) for x in rejected).items())),
        'draft_equals_target_top1_fraction': mean([x['draft_is_top1'] for x in rows]),
        'mean_current_deterministic_accept_probability': mean([x['p_draft'] for x in rows]),
        'mean_best_in_candidate_accept_probability_oracle': mean([x['p_best_candidate'] for x in rows]),
        'mean_target_top1_probability_oracle': mean([x['p_top1'] for x in rows]),
        'mean_candidate_probability_coverage': mean([x['coverage'] for x in rows]),
        'mean_target_entropy_nats': mean([x['entropy'] for x in rows]),
        'mean_target_support': mean([x['support'] for x in rows]),
        'target_singleton_support_checked':sum(x['support']==1 for x in rows),
        'target_singleton_support_rejections':sum(x['support']==1 for x in rejected),
        'draft_zero_target_probability':sum(x['p_draft']==0 for x in rows),
        'draft_outside_target_top20':sum(not x['draft_in_target_top20'] for x in rows),
        'hypothetical_position_q_policy_overlap':{name:mean([x['policy_overlap'][name] for x in rows]) for name in ['all_deterministic','all_stochastic_T1','pos1_stochastic_T1','pos1_to_3_stochastic_T1','draft_confidence_below_0.95_stochastic_T1']},
        'hypothetical_stochastic_q_overlap': {source: {str(t): mean([x['overlap'][source][str(t)] for x in rows]) for t in TEMPERATURES} for source in ('selector', 'unary')},
    }

def analyze(trace_path, receipt_path):
    receipt = json.loads(Path(receipt_path).read_bytes()); identity = audit(receipt)
    if not identity['native_selection_trace_verified']: raise ValueError('sampling trace not verified')
    trace = [json.loads(line) for line in Path(trace_path).read_text(encoding='utf8').splitlines() if line]
    d = receipt['decode_detail']['dflash']; output = receipt['output_token_ids']; start = identity['prompt_tokens']
    if len(trace) != d['rounds']: raise ValueError('round count mismatch')
    selected = output if receipt['session']['pending_token_published'] else output + [receipt['session']['pending_token']]
    expected = dict(enumerate(selected, start)); labels=[]; buckets=[[0,0,0,0] for _ in range(7)]; observed_output=[]
    close_at = next((start+i for i,t in enumerate(selected) if t == CLOSE_THINK), None)
    previous_round = None; previous_end = None
    for row in trace:
        if row['schema'] != 'quasar-sampling-rejection-trace-v1' or row['draft_scores_are_proposal_probabilities']:
            raise ValueError('trace schema/proposal semantics')
        if previous_round is not None and row['round'] != previous_round+1: raise ValueError('round gap')
        previous_round = row['round']
        cursor=row['start_consumed']; path=row['path']; emitted=row['output_tokens']; rejection=row['first_rejection']
        if previous_end is not None and cursor != previous_end: raise ValueError('committed cursor gap or rollback')
        previous_end = row['end_consumed']
        if len(path)!=8 or len(row['target_rows'])!=8 or row['selector_path']!='greedy' or not row['gate_enabled']:
            raise ValueError('expected fixed B8 global greedy gate proposal')
        if any(expected.get(cursor+i)!=t for i,t in enumerate(emitted)): raise ValueError('round/output mismatch')
        observed_output.extend(emitted)
        reject_slot = rejection['offset'] if rejection else None
        checked = list(range(1, len(emitted)+(1 if rejection else 0)))
        if row['canonical_draft_slots'] != checked: raise ValueError('canonical mask mismatch')
        if rejection and (reject_slot!=len(emitted) or expected[cursor+reject_slot]!=rejection['correct_target_token']):
            raise ValueError('rejection boundary mismatch')
        for slot in range(1,8):
            bucket=0 if slot<len(emitted) else 1 if slot==reject_slot else 2 if rejection else 3
            buckets[slot-1][bucket]+=1
        for slot in checked:
            trow=row['target_rows'][slot-1]; pos=cursor+slot
            if trow['absolute_position']!=pos or trow['uniform']!=uniform(pos): raise ValueError('absolute RNG mismatch')
            ids=trow['ids']; p=nucleus(ids,trow['logits']); sampled=draw(p,uniform(pos))
            if trow['top1'] != ids[0]: raise ValueError('target top1 mismatch')
            if sampled != trow['selected'] or sampled != expected[pos]: raise ValueError('target draw mismatch')
            cand=row['draft_candidates'][slot*16:(slot+1)*16]
            # CUDA transitions output ALREADY contains unary + scaled pair score.
            score=[f32(x) for x in row['draft_transitions'][slot*16:(slot+1)*16]]
            unary=[f32(x) for x in row['draft_unary'][slot*16:(slot+1)*16]]
            if len(cand)!=16 or len(set(cand))!=16 or any(not math.isfinite(v) for v in score+unary): raise ValueError('draft row')
            chosen=cand[max(range(16),key=lambda i:score[i])]
            if chosen!=path[slot]: raise ValueError('selector path/rank reconstruction mismatch')
            ranks=sorted(range(16),key=lambda i:(-score[i],i)); rank=cand.index(sampled) if sampled in cand else None
            top1=ids[0]; accepted=slot<len(emitted)
            if accepted!=(chosen==sampled): raise ValueError('acceptance mismatch')
            overlap={}
            for source,scores in [('selector',score),('unary',unary)]:
                overlap[source]={}
                for temp in TEMPERATURES:
                    q=softmax(cand,scores,temp)
                    overlap[source][str(temp)]=sum(min(p.get(c,0),mass) for c,mass in q.items())
            q1=softmax(cand,score,1.0);q_peak=max(q1.values());base_p=p.get(chosen,0);stochastic_overlap=overlap['selector']['1.0']
            policy_overlap={'all_deterministic':base_p,'all_stochastic_T1':stochastic_overlap,
                'pos1_stochastic_T1':stochastic_overlap if slot==1 else base_p,
                'pos1_to_3_stochastic_T1':stochastic_overlap if slot<=3 else base_p,
                'draft_confidence_below_0.95_stochastic_T1':stochastic_overlap if q_peak<.95 else base_p}
            labels.append({'round':row['round'],'position':slot,'absolute_position':pos,
                'phase':'thinking' if close_at is None or pos<=close_at else 'answer',
                'accepted':accepted,'category':classify(chosen,sampled,top1,cand),
                'sample_is_top1':sampled==top1,'draft_is_top1':chosen==top1,'top1_in_candidates':top1 in cand,
                'sample_selector_rank':None if rank is None else ranks.index(rank)+1,
                'p_draft':p.get(chosen,0),'p_best_candidate':max(p.get(c,0) for c in cand),'p_top1':p[top1],
                'coverage':sum(p.get(c,0) for c in cand),'entropy':-sum(v*math.log(v) for v in p.values()),
                'support':len(p),'overlap':overlap,'policy_overlap':policy_overlap,'draft_in_target_top20':chosen in ids,'selector_margin':score[ranks[0]]-score[ranks[1]],
                'sample_unary_rank':None if rank is None else rank+1})
    if buckets!=d['by_position_accepted_rejected_unchecked_censored']: raise ValueError('position ledger mismatch')
    if sum(x['accepted'] for x in labels)!=d['accepted_draft_tokens'] or len(labels)!=d['checked_draft_tokens']:
        raise ValueError('receipt acceptance totals mismatch')
    # A final published pending token can be outside the round's consumed output.
    if output[:len(observed_output)]!=observed_output: raise ValueError('round sequence does not reconstruct output')
    phases={phase:summarize([x for x in labels if x['phase']==phase]) for phase in ('thinking','answer')}
    return {'schema':'quasar-english-rejection-ledger-v1','identity':identity,'diagnostic_only':True,'json_f32_logits_restored_to_binary32_before_f64_math':True,
        'trace_sha256':hashlib.sha256(Path(trace_path).read_bytes()).hexdigest(),
        'receipt_sha256':hashlib.sha256(Path(receipt_path).read_bytes()).hexdigest(),
        'rounds':len(trace),'first_close_think_output_offset':None if close_at is None else close_at-start,
        'output_tokens':len(output),'stop_reason':receipt['stop_reason'],
        'accepted_over_proposed':d['accepted_over_proposed'],'accepted_over_checked':d['accepted_over_checked'],
        'by_position_accepted_rejected_unchecked_censored':buckets,
        'all_checked':summarize(labels),'phases':phases,
        'by_position':{str(i):summarize([x for x in labels if x['position']==i]) for i in range(1,8)},
        'thinking_by_position':{str(i):summarize([x for x in labels if x['phase']=='thinking' and x['position']==i]) for i in range(1,8)},
        'thinking_entropy_buckets':{name:summarize([x for x in labels if x['phase']=='thinking' and low<=x['entropy']<high]) for name,low,high in [('0_to_0.25',0,.25),('0.25_to_0.75',.25,.75),('0.75_to_1.5',.75,1.5),('1.5_plus',1.5,float('inf'))]},
        'thinking_windows_1024':{str(i):summarize([x for x in labels if x['phase']=='thinking' and (x['absolute_position']-start)//1024==i]) for i in sorted({(x['absolute_position']-start)//1024 for x in labels if x['phase']=='thinking'})},
        'limitations':['Post-first-rejection rows never used as labels.','Current draft proposal is deterministic: true q is a point mass, not softmax of gate scores.','Softmax q and overlap are offline hypotheses on observed conditional prefixes, not changed-rollout acceptance/TPS.','Oracle candidate choice and coverage use target knowledge and are not available before verification.','Diagnostics add D2H/file IO: do not compare their TPS with performance runs.','Rejection categories are observations, not independent causal attribution.']}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trace',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=analyze(a.trace,a.receipt)
    with a.out.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({'status':'PASS','rounds':result['rounds'],'checked':result['all_checked']['checked'],'first_rejections':result['all_checked']['first_rejections']}))
if __name__=='__main__':main()
