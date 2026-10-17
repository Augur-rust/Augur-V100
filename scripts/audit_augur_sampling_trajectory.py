"""Audit committed token alignment; compare sampling and greedy as separate laws.

Reads only local native receipts. Emits hashes/counters, never prompts or tokens.
Does not claim B1/B8 numerical equivalence or derive logits from output strings.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def digest_tokens(tokens):
    return hashlib.sha256(b''.join(struct.pack('<I',t) for t in tokens)).hexdigest()


def audit(receipt):
    if receipt.get('schema') != 'quasar-token-turn-receipt-v1':
        raise ValueError('not a generation receipt')
    output=receipt['output_token_ids']; session=receipt['session']; history=session['consumed_token_ids']
    published=session['pending_token_published']; pending=session['pending_token']
    if type(published) is not bool or any(type(t) is not int or not 0<=t<248320 for t in output+history+[pending]):
        raise ValueError('invalid token or pending state')
    if not output and published: raise ValueError('published pending without output')
    consumed_output=output[:-1] if published else output
    start=len(history)-len(consumed_output)
    if start<0 or history[start:]!=consumed_output: raise ValueError('committed output/history mismatch')
    if published and pending!=output[-1]: raise ValueError('published pending mismatch')
    if receipt['stop_reason']!=('length' if published else 'eos'): raise ValueError('stop boundary mismatch')
    selected=output if published else output+[pending]
    if receipt['selected_tokens_including_eos']!=len(selected): raise ValueError('selected count mismatch')
    sampler=receipt.get('decode_detail',{}).get('sampling',{})
    preset=sampler.get('preset'); mode='sampling' if preset=='ninfer_thinking_v1' else 'greedy' if preset=='greedy' else 'unknown'
    hashed=hashlib.sha256()
    for offset,token in enumerate(selected):hashed.update(struct.pack('<QI',start+offset,token))
    trace=sampler.get('trajectory'); verified=False
    if trace is not None:
        if mode!='sampling' or trace.get('schema')!='quasar-sampled-selection-trace-v1':raise ValueError('trace mode/schema mismatch')
        if (trace.get('first_position')!=start or trace.get('last_position')!=start+len(selected)-1 or
            trace.get('selections')!=len(selected) or trace.get('sha256')!=hashed.hexdigest() or
            trace.get('uncommitted_speculative_tail_excluded') is not True):raise ValueError('sampled selections do not align with committed receipt')
        verified=True
    dflash=receipt.get('decode_detail',{}).get('dflash',{})
    proposed=dflash.get('proposed_draft_tokens'); accepted=dflash.get('accepted_draft_tokens')
    return dict(request_id=receipt['request_id'],decode_mode=mode,sampling_preset=preset,
        prompt_tokens=start,input_token_sha256=digest_tokens(history[:start]),output_tokens=len(output),
        output_token_sha256=digest_tokens(output),selection_trajectory_sha256=hashed.hexdigest(),
        native_selection_trace_verified=verified,commit_pending_alignment_verified=True,
        selector_policy=(dflash.get('english_selector') or {}).get('policy'),
        width_policy=(dflash.get('adaptive_gate_decisions') or {}).get('policy'),
        accepted_drafts=accepted,proposed_drafts=proposed,
        accepted_over_proposed=accepted/proposed if type(proposed) is int and proposed>0 and type(accepted) is int else None,
        semantics='sampling_and_greedy_are_separate_target_laws',
        logits_or_hidden_states_audited=False,B1_B8_numerical_equivalence_claim=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipts',nargs='+',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); rows=[]
    for file in args.receipts:
        raw=file.read_bytes();row=audit(json.loads(raw));row['receipt_sha256']=hashlib.sha256(raw).hexdigest();rows.append(row)
    result=dict(schema='quasar-trajectory-audit-v1',groups={mode:[r for r in rows if r['decode_mode']==mode] for mode in ('sampling','greedy','unknown')},
        cross_mode_TPS_or_acceptance_aggregation=False)
    with args.out.open('x',encoding='utf8') as file:json.dump(result,file,ensure_ascii=False,indent=2)
    print(json.dumps(dict(status='PASS',receipts=len(rows),native_sampling_traces_verified=sum(r['native_selection_trace_verified'] for r in rows)),ensure_ascii=False))

if __name__=='__main__':main()
