"""Independent receipt checks for opt-in DFlash measured_decode_cost_v1.
This validates accounting, not prediction quality, speed gains, or equivalence.
"""
import math

POLICY = 'measured_decode_cost_v1'
DEFERRED_POLICY = 'deferred_decode_cost_v2'
FIXED_POLICY = 'fixed_b8_v1'
CLOCK = 'after_synchronized_selection_before_publication_to_next_synchronized_selection'


def require(ok, message):
    if not ok:
        raise ValueError('DFlash cost ledger: ' + message)


def count(value, label):
    require(type(value) is int and value >= 0, 'invalid ' + label)
    return value


def sample(value, lane, probe=False):
    transactions = count(value['completed_transactions'], 'transactions')
    tokens = count(value['consumed_tokens'], 'consumed tokens')
    wall = count(value['wall_ns'], 'wall ns')
    require((transactions == 0) == (tokens == 0) == (wall == 0), 'empty sample mismatch')
    require(tokens == transactions if lane == 'B1' else transactions <= tokens <= 8*transactions,
            'lane width mismatch')
    seconds = value['wall_s']
    require(type(seconds) in (int, float) and math.isfinite(seconds)
            and math.isclose(seconds, wall/1e9, rel_tol=1e-12, abs_tol=1e-12), 'wall unit mismatch')
    mean = value['ns_per_consumed_token']
    if tokens:
        require(type(mean) in (int, float) and math.isfinite(mean)
                and math.isclose(mean, wall/tokens, rel_tol=1e-12), 'cost denominator mismatch')
    else:
        require(mean is None, 'empty sample has a mean')
    if probe:
        require(transactions == (16 if lane == 'B1' else 8), 'incomplete probe used for a decision')
    return transactions, tokens, wall


def validate(receipt):
    d = receipt['decode_detail']['dflash']
    c = d['adaptive_gate_decisions']
    require(c['policy'] in (POLICY,DEFERRED_POLICY,FIXED_POLICY) and c['decision_policy_unchanged'] is False, 'policy identity')
    deferred = c['policy'] == DEFERRED_POLICY
    fixed = c['policy'] == FIXED_POLICY
    require(d['active'] is True and d['block_width'] == 8, 'cost gate without active B8')
    rounds = count(d['rounds'], 'B8 rounds')
    accepted = count(d['accepted_draft_tokens'], 'accepted drafts')
    checked = count(d['checked_draft_tokens'], 'checked drafts')
    proposed = count(d['proposed_draft_tokens'], 'proposed drafts')
    require(proposed == 7*rounds and accepted <= checked <= proposed, 'draft denominators')
    require(d['adaptive_gate_min_rounds'] is None
            and d['adaptive_gate_threshold_accepted_over_checked'] is None, 'legacy threshold attached to cost policy')
    for key, expected in [('B1_sample_steps',16), ('B8_sample_rounds',8),
                          ('recheck_after_consumed_tokens',512), ('maximum_decisions',4),
                          ('hysteresis_percent',2)]:
        require(count(c[key],key) == (0 if fixed else expected), 'unregistered controller parameter')
    require(c['clock'] == CLOCK and c['request_drained'] is True and c['interval_open'] is False,
            'incomplete or incompatible timing boundary')
    for key, expected in [('includes_publication_setup_and_next_selection',True),
                          ('extra_GPU_synchronization',False), ('prefill_excluded',True),
                          ('same_prefix_B1_counterfactual_measured',False),
                          ('all_sample_output_is_published_and_charged',True),
                          ('not_whole_decode_or_HTTP_wall',True)]:
        require(c[key] is expected, 'unsupported measurement scope: ' + key)
    b1 = sample(c['all_completed_B1'], 'B1')
    b8 = sample(c['all_completed_B8'], 'B8')
    initial = sample(c['initial_B8_before_cost_sampling'],'B8') if 'initial_B8_before_cost_sampling' in c else (0,0,0)
    if not deferred:
        require(initial == (0,0,0), 'immediate policy includes deferred warmup')
    require(all(x<=y for x,y in zip(initial,b8)), 'unpaid initial B8 work')
    trigger = c.get('first_cost_probe_trigger')
    if fixed:
        require(trigger is None and c.get('low_acceptance_trigger') is None, 'fixed B8 sampled a trigger')
    if deferred:
        require(trigger in (None,'legacy_low_acceptance','initial_512_consumed_tokens'), 'initial trigger')
        if trigger == 'legacy_low_acceptance':
            observed=c['low_acceptance_trigger']
            rounds=count(observed['rounds'],'trigger rounds')
            a=count(observed['accepted'],'trigger accepted');k=count(observed['checked'],'trigger checked')
            require(rounds==initial[0] and rounds>=8 and a==initial[1]-rounds
                    and a<=k<=7*rounds and k>0 and a/k<.8, 'unmeasured low-acceptance trigger')
        elif trigger == 'initial_512_consumed_tokens':
            require(512<=initial[1]<=519 and c.get('low_acceptance_trigger') is None, 'initial cadence')
        else:
            require(initial[1]<512 and c.get('low_acceptance_trigger') is None, 'missed initial trigger')
    tail = c['terminal_incomplete_intervals']
    nt = count(tail['transactions'], 'terminal transactions')
    tokens = count(tail['consumed_tokens'], 'terminal consumed tokens')
    ns = count(tail['wall_ns'], 'terminal wall')
    selection = count(tail['selection_without_consume_wall_ns'], 'terminal selection wall')
    require(tail['used_for_decisions'] is False, 'incomplete tail used for a decision')
    if nt == 0:
        require(tail['lane'] is None and tokens == ns == 0, 'empty terminal transaction')
    else:
        require(nt == 1 and tail['lane'] in ('B1','B8') and selection == 0, 'invalid terminal lane')
        require(tokens == 1 if tail['lane'] == 'B1' else 1 <= tokens <= 8, 'terminal width')
    start = count(c['start_consumed_tokens'], 'start cursor')
    end = count(c['terminal_consumed_tokens'], 'terminal cursor')
    require(end - start == b1[1] + b8[1] + tokens, 'consumed cursor sum')
    require(end == len(receipt['session']['consumed_token_ids']), 'session cursor mismatch')
    output = len(receipt['output_token_ids'])
    require(receipt['stop_reason'] in ('length','eos'), 'unsupported successful stop')
    pending = int(receipt['stop_reason'] == 'length')
    require(end - start == output - pending, 'EOS/pending token counted as consumed')
    require(b8[0] + int(tail['lane'] == 'B8') == d['rounds'], 'B8 rounds clock mismatch')
    b8_consumed = b8[1] + (tokens if tail['lane'] == 'B8' else 0)
    b8_pending = int(pending and tail['lane'] == 'B8')
    require(b8_consumed == d['rounds'] + d['accepted_draft_tokens'] - b8_pending,
            'B8 accepted-token accounting mismatch')
    decisions = c['decisions']
    require(type(decisions) is list and len(decisions) <= 4, 'unbounded controller decisions')
    if fixed:
        require(not decisions, 'fixed B8 contains adaptive decisions')
        if c['forced_native_reason'] is None:
            require(b1 == (0,0,0) and tail['lane'] != 'B1' and c['next_lane_if_more_tokens'] == 'B8',
                    'fixed B8 contains unguarded native work')
    incumbent = 'B1'
    previous_end = None
    probe_totals = [[0,0,0],[0,0,0]]
    for decision in decisions:
        a, b = sample(decision['B1'],'B1',True), sample(decision['B8'],'B8',True)
        begin = count(decision['probe_start_consumed'],'probe start')
        finish = count(decision['decision_consumed'],'decision cursor')
        require(start <= begin < finish <= end and finish-begin == a[1]+b[1], 'probe extent')
        if previous_end is None:
            require(begin == start + initial[1], 'first probe discarded initial output')
        else:
            gap = begin - previous_end
            require(gap == 512 if incumbent == 'B1' else 512 <= gap <= 519, 'probe cadence')
        require(decision['incumbent'] == incumbent, 'decision incumbent drift')
        left = b[2] * a[1] * 100
        right = a[2] * b[1] * (98 if incumbent == 'B1' else 102)
        selected = 'B8' if left < right else 'B1'
        require(decision['selected'] == selected, 'cost decision does not follow measured samples')
        incumbent, previous_end = selected, finish
        for target, values in zip(probe_totals,(a,b)):
            for i in range(3): target[i] += values[i]
    if deferred and decisions:
        require(trigger is not None, 'decision without a sampled trigger')
    for i in range(3):probe_totals[1][i]+=initial[i]
    for sampled, total in zip(probe_totals,(b1,b8)):
        require(all(x <= y for x,y in zip(sampled,total)), 'uncharged sampling work')
    require(c['forced_native_reason'] in (None,'vision','capacity'), 'unexpected native guard')
    require(c['next_lane_if_more_tokens'] in ('B1','B8'), 'invalid next lane')
    if c['forced_native_reason'] is not None:
        require(c['next_lane_if_more_tokens'] == 'B1', 'guard failed to select native')
    whole, prefill = receipt['whole_wall_s_before_receipt'], receipt['prefill_wall_s']
    require(all(type(v) in (int,float) and math.isfinite(v) and v >= 0 for v in (whole,prefill))
            and whole > prefill, 'invalid generation wall')
    generation = whole - prefill
    measured = (b1[2] + b8[2] + ns + selection) / 1e9
    require(measured <= generation + 1e-6, 'overlapping costs exceed full generation')
    return dict(policy=c['policy'], completed_B1_transactions=b1[0],completed_B8_transactions=b8[0],
                completed_decisions=len(decisions),measured_decode_interval_s=measured,
                native_generation_s=generation,unassigned_generation_s=max(0.,generation-measured),
                semantic_quality_qualified=False,performance_gain_qualified=False)
