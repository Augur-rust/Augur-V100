"""Closed Nerkyor text/token boundary, independent of QUASAR product profiles.

Pinned manifests select the 512-token DFlash or 4096-token no-DFlash greedy/F32
research profile. No MTP,
vision, sampled decoding, old gate calibration, restore or history truncation.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import re
import struct
import uuid
import qwen35_chat as C

PROFILE = 'nerkyor_f32kv512_core_b8prefill_dflash_greedy_session_research_v2'
EXECUTION = 'nerkyor_original_nvfp4_core_f32_kv_b8_prefill_pending_v2'
POLICY = 'checkpoint_nerkyor_bf16_gate_f32_cta_research_v1'
NO_DFLASH_PROFILE = 'nerkyor_f32kv4096_qpn_reassoc_no_dflash_greedy_session_research_v1'
NO_DFLASH_EXECUTION = 'nerkyor_qpn_source_reassoc_f32_kv4096_no_dflash_v1'
NO_DFLASH_POLICY = 'checkpoint_nerkyor_qpn_source_pair_reassoc_f64_gate_f32_research_v1'
GATE = 'bf16_f32_fma_cta256_tree_research_v1'
CAPACITY, MAX_OUTPUT, VOCAB = 512, 486, 248320
MAX_FRAME, MAX_TEXT_BYTES = 256 * 1024, 64 * 1024
SCHEMA = 'nerkyor-token-session-jsonl-v1'
RECOVERY_SCHEMA = 'nerkyor-token-session-jsonl-v2'
RECOVERY_CONTRACT = 'bound_request_unpublished_discard_or_committed_close_v1'
GENERATION = 'greedy_lowest_id_tie_nonfinite_fail_closed'
PENDING = 'consumed_history_excludes_pending_append_consumes_pending_once'


def session_settings(template):
    no_dflash = getattr(template, 'manifest', {}).get('execution_profile') == NO_DFLASH_EXECUTION
    return dict(profile=NO_DFLASH_PROFILE if no_dflash else PROFILE,
                execution=NO_DFLASH_EXECUTION if no_dflash else EXECUTION,
                policy=NO_DFLASH_POLICY if no_dflash else POLICY,
                capacity=4096 if no_dflash else CAPACITY,
                max_output=3891 if no_dflash else MAX_OUTPUT,
                dflash=not no_dflash)


def read_bounded(path, maximum=MAX_FRAME):
    with Path(path).open('rb') as file:
        raw = file.read(maximum + 1)
    C.require(len(raw) <= maximum, 'Nerkyor file/frame bound exceeded')
    return raw


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def same_json(left, right):
    # Python equality aliases bool/int and integer/float; the wire contract does not.
    return canonical_json(left) == canonical_json(right)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def valid_id(value):
    return isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,64}', value) is not None


def valid_tokens(ids):
    return type(ids) is list and all(type(i) is int and 0 <= i < VOCAB for i in ids)


def token_sha(ids):
    C.require(valid_tokens(ids), 'invalid Nerkyor token vector')
    return C.sha256(struct.pack('<' + str(len(ids)) + 'I', *ids))


def execution_contract(manifest):
    return dict(schema='nerkyor-fp32-gate-core-b8-prefill-dflash-research-identity-v2', profile='nerkyor_w4a16',
        policy=POLICY, activation_dtype='F32', kv_dtype='F32', gate_numeric_contract=GATE,
        authenticated_checkpoint=manifest['checkpoint'], draft_file_sha256=manifest['draft_gguf_sha256'],
        candidate_publication=False, product_admitted=False,
        target_execution=dict(profile=EXECUTION, width=8, activation_dtype='F32', kv_dtype='F32',
            projection='retained_original_nvfp4_exact_f32_batch_v1', gate=GATE,
            attention='core_tu_batched_qk_norm_split_rope_scalar_fma_attention_v2',
            gdn='checkpoint_contiguous3_f32_recorded_prefix_replay_v1', target_head='llama_mmvf8_bf16_f32_v1',
            draft_head='llama_mmvf8_bf16_f32_v1', scalar_boundary='same_gate_f32_scalar_initial_tail_pending_consume_v1',
            prefill='scalar_initial1_full_b8_commit_scalar_tail_v1',
            selection=GENERATION, final_token='selected_unconsumed',
            proposal='owner_generation_capture_epoch_revision_cursor_sealed', publication='research_only'))


class Template(C.ChatTemplate):
    """Render retained, hashed source bytes, not an inferred Qwen chat format."""
    def __init__(self, model_root, manifest_path, manifest_sha256):
        from jinja2 import StrictUndefined
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        from tokenizers import Tokenizer
        C.require(C.digest_string(manifest_sha256), 'caller-pinned manifest SHA256 required')
        raw = read_bounded(manifest_path, 1024 * 1024)
        C.require(C.sha256(raw) == manifest_sha256, 'Nerkyor manifest digest mismatch')
        m = C.strict_json(raw)
        keys = {'schema', 'execution_profile', 'checkpoint', 'tokenizer_sha256', 'tokenizer_config_sha256',
                'generation_config_sha256', 'chat_template_sha256', 'draft_config_sha256', 'draft_gguf_sha256'}
        C.require(set(m) == keys and m['schema'] == 'nerkyor-session-model-manifest-v1'
                  and m['execution_profile'] in (EXECUTION, NO_DFLASH_EXECUTION), 'wrong Nerkyor model/profile manifest')
        checkpoint = m['checkpoint']
        names = {f'text-{i:02}.safetensors' for i in range(1, 6)} | {'mtp-bf16.safetensors', 'vision-bf16.safetensors'}
        C.require(type(checkpoint) is dict and set(checkpoint) == {'config_sha256', 'index_sha256', 'shards'}
                  and type(checkpoint['shards']) is dict and set(checkpoint['shards']) == names,
                  'wrong source checkpoint names')
        hashes = [m[k] for k in keys if k.endswith('_sha256')] + [checkpoint['config_sha256'], checkpoint['index_sha256']] + list(checkpoint['shards'].values())
        C.require(all(C.digest_string(h) for h in hashes), 'invalid Nerkyor source digest')
        root = Path(model_root)
        source = {}
        for key, name, bound in [('tokenizer_sha256', 'tokenizer.json', 64 * 1024 * 1024),
                                ('tokenizer_config_sha256', 'tokenizer_config.json', 4 * 1024 * 1024),
                                ('generation_config_sha256', 'generation_config.json', 1024 * 1024),
                                ('chat_template_sha256', 'chat_template.jinja', 4 * 1024 * 1024)]:
            source[key] = read_bounded(root / name, bound)
            C.require(C.sha256(source[key]) == m[key], 'Nerkyor metadata digest mismatch: ' + name)
        config = C.strict_json(source['tokenizer_config_sha256'])
        template_text = source['chat_template_sha256'].decode('utf-8')
        C.require(template_text and ('chat_template' not in config or config['chat_template'] == template_text),
                  'separate and embedded templates disagree')
        generation = C.strict_json(source['generation_config_sha256'])
        C.require(valid_tokens(generation.get('eos_token_id')) and generation['eos_token_id'] == [248046, 248044], 'wrong target EOS export')
        self.tokenizer = Tokenizer.from_str(source['tokenizer_sha256'].decode('utf-8'))
        for text, i in [('<|im_end|>', 248046), ('<|endoftext|>', 248044), ('<|audio_start|>', 248070)]:
            C.require(self.tokenizer.token_to_id(text) == i, 'Nerkyor special-token mismatch')
        self.eos, self.end = {248046, 248044}, 248046
        self.manifest, self.manifest_sha256 = m, manifest_sha256
        self.profile = session_settings(self)['profile']
        self.identity = {k: m[k] for k in keys if k.endswith('_sha256')}
        env = ImmutableSandboxedEnvironment(undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)
        def fail(message):
            raise C.ProtocolError('Nerkyor template rejected: ' + str(message))
        env.globals['raise_exception'] = fail
        self.template = env.from_string(template_text)

    def render_messages(self, messages, generation=True):
        # Explicit speed-oriented text mode. No hidden system/reasoning injection.
        return super().render(messages, add_generation_prompt=generation, enable_thinking=False)


def verify_ready(frame, template):
    return _verify_ready(frame, template, SCHEMA, False)


def verify_recovery_ready(frame, template):
    return _verify_ready(frame, template, RECOVERY_SCHEMA, True)


def _verify_ready(frame, template, schema, recovery):
    settings = session_settings(template)
    expected = dict(event='ready', schema=schema, profile=settings['profile'], manifest_sha256=template.manifest_sha256,
        capacity=settings['capacity'], maximum_output_tokens=settings['max_output'], generation=GENERATION,
        activation_dtype='F32', kv='F32', lm_head='original_bf16', eos_token_ids=[248046, 248044],
        cooperative_input_cancellation=True, request_bound_cancellation_discard=recovery,
        input_frame_limit_bytes=64 * 1024, pending_contract=PENDING,
        persistence='create_new_fsync_turn_receipts_not_gpu_restore', tokenizer_execution='caller_owned_token_ids',
        decoded_text_output=False, dflash_enabled=settings['dflash'], mtp_enabled=False, vision_enabled=False,
        restore_enabled=False, product_ready=False, actual_long_history_qualified=False, http_enabled=False)
    if recovery:
        expected['request_cancellation_contract'] = RECOVERY_CONTRACT
    for key, value in expected.items():
        C.require(type(frame.get(key)) is type(value) and frame[key] == value, 'Nerkyor ready mismatch: ' + key)
    target = frame.get('target', {})
    C.require(type(target) is dict, 'invalid target metadata object')
    for key, value in dict(profile='nerkyor_w4a16', session_profile=settings['profile'], session_manifest_sha256=template.manifest_sha256,
            policy=settings['policy'], activation_dtype='F32', kv_dtype='F32', kv_capacity=settings['capacity'],
            nerkyor_target_execution=settings['execution'], full_shard_hashes_verified=True, product_admitted=False,
            product_ready=False, vision_enabled=False).items():
        C.require(type(target.get(key)) is type(value) and target[key] == value, 'Nerkyor target mismatch: ' + key)
    C.require(target.get('source_identity') == template.manifest['checkpoint'], 'target checkpoint differs from pinned manifest')
    if not settings['dflash']:
        C.require(target.get('dflash_loaded') is False and target.get('dflash_enabled') is False
                  and target.get('dflash_feature_capture') is None, 'no-DFlash owner loaded draft/capture')
        C.require(target.get('qpn_source_pair_exact_projection_count') == 400
                  and target.get('qpn_source_pair_arithmetic') == 'source_f32_pair_reassoc_k16_even_odd_f64_fold_v1',
                  'wrong no-DFlash projection arithmetic')
        return
    contract = execution_contract(template.manifest)
    capture = target.get('dflash_feature_capture', {})
    C.require(type(capture) is dict, 'invalid capture metadata object')
    expected_hash = C.sha256(canonical_json(contract))
    C.require(same_json(capture.get('source_identity_contract'), contract) and capture.get('source_identity_sha256') == expected_hash,
              'Nerkyor numerical source seal mismatch')


class Ledger:
    def __init__(self, template, system=None):
        self.template = template
        self.session_id = 'nerkyor-' + uuid.uuid4().hex
        self.messages = [] if system is None else [template.message('system', system)]
        self.history, self.terminal = [], None
        self.prefix, self.last_text, self.poisoned = '', '', False

    def plan_user(self, text, budget):
        C.require(isinstance(text, str) and len(text.encode('utf-8')) <= MAX_TEXT_BYTES, 'text input byte limit')
        return self.plan_messages(self.messages + [self.template.message('user', text)], budget)

    def plan_messages(self, messages, budget):
        """Plan normalized full history; only an exact prefix may reuse native IDs.

        Public request validation belongs to the caller. This guard separately
        prevents accidental last-user-only forwarding into a resident session.
        Planning is CPU-only and never truncates or changes the live ledger.
        """
        C.require(not self.poisoned and type(budget) is int and 1 <= budget <= session_settings(self.template)['max_output'], 'invalid session/output budget')
        C.require(type(messages) is list and messages and all(type(m) is dict for m in messages)
                  and messages[-1].get('role') == 'user', 'normalized full user-ended history required')
        if self.terminal is not None:
            C.require(same_json(messages[:-1], self.messages), 'full history differs from the resident transcript')
        messages = copy.deepcopy(messages)
        full = self.template.render_messages(messages)
        if self.terminal is None:
            ids = self.template.encode(full)
            inp = dict(kind='open', prompt_token_ids=ids)
        else:
            prior = self.template.render_messages(self.messages, generation=False)
            C.require(prior.endswith('<|im_end|>\n') and full.startswith(prior), 'canonical template rewrote prior transcript')
            pending = self.terminal['receipt']['session']['pending_token']
            suffix = full[len(prior) - 1:]
            eos = self.terminal['receipt']['stop_reason'] == 'eos'
            if eos:
                C.require(pending == self.template.end, 'non-message EOS needs a new session')
            else:
                suffix = '<|im_end|>' + suffix
            suffix_ids = self.template.encode(suffix)
            C.require(suffix_ids, 'empty canonical append suffix')
            ids = self.history + [pending] + suffix_ids
            inp = dict(kind='append', revision=self.terminal['receipt']['session']['revision'],
                       exact_history=self.history.copy(), suffix_token_ids=suffix_ids)
            # Preserve the actual generated BPE IDs and whitespace, never re-encode history.
            full = self.prefix + self.last_text + ('<|im_end|>' if eos else '') + suffix
        C.require(valid_tokens(ids) and 0 < len(ids) < session_settings(self.template)['capacity'], 'context exhausted; reset explicitly, no history truncation')
        C.require(len(full.encode('utf-8')) <= MAX_TEXT_BYTES, 'canonical text byte limit')
        request = dict(request_id='turn-' + uuid.uuid4().hex, session_id=self.session_id,
                       max_new_tokens=min(budget, session_settings(self.template)['capacity']-len(ids)), input=inp, selector_policy=None)
        return dict(request=request, input_ids=ids, messages=messages, prefix=full)

    def accept(self, plan, committed, text):
        self.terminal = copy.deepcopy(committed)
        self.history = committed['receipt']['session']['consumed_token_ids'].copy()
        self.messages = copy.deepcopy(plan['messages']) + [dict(role='assistant', content=text, reasoning_content='', tool_calls=[])]
        self.prefix, self.last_text = plan['prefix'], text


def verify_token(frame, plan, tokens, eos):
    t = frame.get('token', {})
    C.require(type(t) is dict, 'invalid token object')
    C.require(frame.get('event') == 'token' and frame.get('product_ready') is False
              and t.get('request_id') == plan['request']['request_id'] and t.get('session_id') == plan['request']['session_id']
              and type(t.get('sequence')) is int and t['sequence'] == len(tokens)
              and type(t.get('token_id')) is int and 0 <= t['token_id'] < VOCAB and t['token_id'] not in eos
              and len(tokens) < plan['request']['max_new_tokens'], 'foreign/duplicate/invalid token event')
    tokens.append(t['token_id'])


def verify_terminal(frame, plan, ledger, tokens):
    request = plan['request']
    settings = session_settings(ledger.template)
    C.require(frame.get('event') == 'terminal' and frame.get('product_ready') is False
              and frame.get('request_id') == request['request_id'] and frame.get('session_id') == ledger.session_id,
              'foreign terminal event')
    committed = frame.get('committed', {})
    C.require(type(committed) is dict and type(committed.get('receipt')) is dict, 'invalid committed receipt object')
    r, sha = committed.get('receipt', {}), committed.get('receipt_sha256')
    s = r.get('session', {})
    C.require(type(s) is dict, 'invalid session snapshot object')
    C.require(C.digest_string(sha) and r.get('schema') == 'nerkyor-token-turn-receipt-v1'
              and r.get('request_id') == request['request_id'] and valid_tokens(r.get('output_token_ids'))
              and r['output_token_ids'] == tokens and r.get('product_ready') is False
              and r.get('vision_enabled') is False and r.get('dflash_enabled') is settings['dflash'], 'Nerkyor receipt output/claims')
    previous = ledger.terminal
    C.require(r.get('previous_receipt_sha256') == (previous['receipt_sha256'] if previous else None), 'broken receipt chain')
    C.require(s.get('profile') == settings['profile'] and s.get('manifest_sha256') == ledger.template.manifest_sha256
              and s.get('session_id') == ledger.session_id and type(s.get('revision')) is int
              and s['revision'] == (previous['receipt']['session']['revision']+1 if previous else 1)
              and s.get('image_input_sha256') is None, 'foreign receipt snapshot')
    pending, published = s.get('pending_token'), s.get('pending_token_published')
    C.require(type(pending) is int and 0 <= pending < VOCAB and type(published) is bool, 'bad pending token')
    selected = len(tokens) + int(not published)
    C.require(type(r.get('selected_tokens_including_eos')) is int and r['selected_tokens_including_eos'] == selected
              and 1 <= selected <= request['max_new_tokens'], 'selection count exceeds budget')
    if published:
        C.require(r.get('stop_reason') == 'length' and len(tokens) == request['max_new_tokens'] and tokens[-1] == pending,
                  'bad length-pending contract')
        history = plan['input_ids'] + tokens[:-1]
    else:
        C.require(r.get('stop_reason') == 'eos' and pending in ledger.template.eos, 'bad EOS-pending contract')
        history = plan['input_ids'] + tokens
    C.require(valid_tokens(s.get('consumed_token_ids')) and s['consumed_token_ids'] == history
              and s.get('consumed_input_sha256') == token_sha(history), 'exact consumed history mismatch')
    d = r.get('decode_detail', {})
    C.require(type(d) is dict, 'invalid decode detail object')
    for key, value in dict(profile=settings['profile'], execution_profile=settings['execution'], dflash_enabled=settings['dflash'], sampling='greedy',
                          kv_dtype='F32', capacity=settings['capacity'], selector='own_q8_default_uncalibrated' if settings['dflash'] else 'target_greedy',
                          product_ready=False, mtp_executed=False).items():
        C.require(type(d.get(key)) is type(value) and d[key] == value, 'wrong decode detail: ' + key)
    if not settings['dflash']:
        C.require(d.get('dflash_loaded') is False and d.get('rounds') == 0
                  and d.get('first_prefill_selection_excluded') is True, 'no-DFlash decode claims')
        steps = selected - 1
        seconds, tps = d.get('decode_wall_s'), d.get('decode_tokens_per_second')
        C.require(type(d.get('decode_steps')) is int and d['decode_steps'] == steps, 'decode step count includes prefill token')
        import math
        C.require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0, 'invalid decode duration')
        C.require((steps == 0 and seconds == 0 and tps is None) or
                  (steps > 0 and seconds > 0 and type(tps) in (int, float) and math.isfinite(tps)
                   and abs(tps - steps / seconds) <= max(1e-9, tps * 1e-12)), 'invalid decode TPS')
        prefill = r.get('prefill_detail', {})
        count = len(plan['input_ids']) - (len(request['input']['exact_history']) if request['input']['kind'] == 'append' else 0)
        for key, value in dict(profile=settings['profile'], execution_profile=settings['execution'],
                prefill='qpn_source_reassoc_scalar_v1', tokens=count, scalar_tokens=count,
                b8_full_blocks=0, dflash_loaded=False, product_ready=False).items():
            C.require(type(prefill.get(key)) is type(value) and prefill[key] == value, 'wrong no-DFlash prefill: ' + key)
        return copy.deepcopy(committed)
    prefill = r.get('prefill_detail', {})
    C.require(type(prefill) is dict, 'missing core-B8 prefill receipt')
    for key, value in dict(profile=PROFILE, execution_profile=EXECUTION,
                          prefill='scalar_initial1_full_b8_commit_scalar_tail_v1', product_ready=False).items():
        C.require(type(prefill.get(key)) is type(value) and prefill[key] == value, 'wrong prefill detail: ' + key)
    inp = request['input']
    initial = int(inp['kind'] == 'open')
    count = len(plan['input_ids']) - (0 if initial else len(inp['exact_history']))
    expected = dict(tokens=count, scalar_tokens=initial+(count-initial)%8, b8_full_blocks=(count-initial)//8)
    for key, value in expected.items():
        C.require(type(prefill.get(key)) is int and prefill[key] == value, 'wrong prefill accounting: ' + key)
    C.require(prefill.get('source_identity_sha256') == C.sha256(canonical_json(execution_contract(ledger.template.manifest))),
              'foreign core-B8 prefill source identity')
    return copy.deepcopy(committed)


def persisted_receipt(root, request_id, expected_sha, inline=None):
    C.require(valid_id(request_id) and C.digest_string(expected_sha), 'invalid receipt file identity')
    raw = read_bounded(Path(root) / 'turns' / (request_id + '.json'))
    value = C.strict_json(raw)
    C.require(C.sha256(raw) == expected_sha and (inline is None or same_json(value, inline)), 'durable receipt mismatch')
    return value


def verify_stopped(frame, manifest_sha, normal, profile=PROFILE):
    C.require(frame.get('event') == 'owner_stopped' and frame.get('schema') == 'nerkyor-owner-stopped-v1'
              and frame.get('profile') == profile and frame.get('manifest_sha256') == manifest_sha
              and frame.get('product_ready') is False and frame.get('allocation_closure') is True
              and frame.get('drain_error') is None, 'Nerkyor owner cleanup not proved')
    for name in ['live_bytes', 'live_allocations']:
        before, after = frame.get(name+'_before'), frame.get(name+'_after')
        C.require(type(before) is int and type(after) is int and 0 <= before == after, 'CUDA allocation closure mismatch')
    if normal:
        C.require(frame.get('reason') == 'shutdown' and frame.get('error') is None, 'owner shutdown did not finish normally')


class Decoder:
    """Only emit append-stable complete UTF-8, then verify the final full decode."""
    def __init__(self, tokenizer):
        self.tokenizer, self.tokens, self.text = tokenizer, [], ''

    def push(self, token):
        self.tokens.append(token)
        decoded = self.tokenizer.decode(self.tokens, skip_special_tokens=False)
        C.require(decoded.startswith(self.text), 'decoder rewrote published text')
        delta = decoded[len(self.text):]
        if '\ufffd' in delta:
            return ''
        self.text = decoded
        return delta

    def finish(self):
        decoded = self.tokenizer.decode(self.tokens, skip_special_tokens=False)
        C.require(decoded.startswith(self.text), 'terminal decode rewrote text')
        delta, self.text = decoded[len(self.text):], decoded
        return delta


def verify_cancellation(frame, plan, ledger, tokens, cancel_id, terminal=None):
    """Closed recovery-v2 acknowledgement. Never equates a durable turn with rollback.

    Callers must verify each token/terminal first and then call
    persisted_cancellation before making the owner available to another request.
    This validator does not mutate or revive a ledger.
    """
    req = plan['request']
    C.require(valid_id(cancel_id) and cancel_id != req['request_id'], 'invalid cancellation id')
    for key, value in dict(event='cancelled', request_id=req['request_id'], session_id=req['session_id'],
                           owner_retired=False, product_ready=False).items():
        C.require(type(frame.get(key)) is type(value) and frame[key] == value, 'bad cancellation frame: ' + key)
    committed = frame.get('resolution')
    C.require(type(committed) is dict and set(committed) == {'receipt', 'receipt_sha256'}
              and C.digest_string(committed['receipt_sha256']), 'invalid cancellation resolution envelope')
    r = committed['receipt']
    expected = dict(schema='nerkyor-request-cancel-resolution-v1', contract=RECOVERY_CONTRACT, profile=PROFILE,
        request_id=cancel_id, cancelled_request_id=req['request_id'], session_id=req['session_id'],
        manifest_sha256=ledger.template.manifest_sha256, submitted_tokens_after_resolution=0, active_session=False,
        owner_reusable=True, previous_history_reusable=False, weights_reloaded=False, product_ready=False)
    C.require(type(r) is dict and set(r) == set(expected) | {'outcome', 'cleanup_proof',
              'original_turn_receipt_sha256', 'submitted_tokens_before_resolution'}, 'invalid cancellation resolution fields')
    for key, value in expected.items():
        C.require(type(r.get(key)) is type(value) and r[key] == value, 'bad cancellation receipt: ' + key)
    before = r['submitted_tokens_before_resolution']
    C.require(type(before) is int and 0 <= before <= CAPACITY, 'invalid pre-resolution cursor')
    outcome = r['outcome']
    C.require(outcome in ('unpublished_discarded', 'committed_session_closed', 'unadmitted_session_closed', 'unadmitted_empty'),
              'unknown cancellation outcome')
    proof = r['cleanup_proof']
    if outcome == 'unadmitted_empty':
        C.require(not tokens and terminal is None and ledger.terminal is None and before == 0
                  and req['input']['kind'] == 'open' and proof is None
                  and r['original_turn_receipt_sha256'] is None, 'unadmitted-empty has live or committed state')
        return copy.deepcopy(committed)
    C.require(type(proof) is dict and set(proof) == {'receipt', 'receipt_sha256'}
              and C.digest_string(proof['receipt_sha256']) and type(proof['receipt']) is dict,
              'missing cleanup proof')
    receipt = proof['receipt']
    expected = dict(request_id=cancel_id, session_id=req['session_id'], manifest_sha256=ledger.template.manifest_sha256)
    if outcome == 'unpublished_discarded':
        C.require(terminal is None and r['original_turn_receipt_sha256'] is None, 'cannot discard a committed turn')
        C.require(not tokens or before >= len(plan['input_ids']) + len(tokens) - 1, 'discard cursor precedes published tokens')
        expected.update(schema='nerkyor-session-cancellation-discard-v1', cancelled_request_id=req['request_id'],
            submitted_tokens_before_discard=before, submitted_tokens_after_discard=0, drain_completed=True,
            reset_completed=True, session_discarded=True, previous_history_reusable=False, product_ready=False, dflash_enabled=True)
        C.require(set(receipt) == set(expected) | {'cancellation_boundary'}, 'unknown discard proof fields')
        C.require(type(receipt['cancellation_boundary']) is str and 0 < len(receipt['cancellation_boundary']) <= 256,
                  'missing typed cancellation boundary')
    else:
        if outcome == 'committed_session_closed':
            C.require(type(terminal) is dict and r['original_turn_receipt_sha256'] == terminal.get('receipt_sha256'),
                      'late close lacks verified irreversible terminal')
            previous = terminal
            C.require(previous['receipt']['request_id'] == req['request_id'], 'late close refers to a foreign turn')
        else:
            C.require(not tokens and terminal is None and r['original_turn_receipt_sha256'] is None
                      and ledger.terminal is not None and req['input']['kind'] == 'append',
                      'unadmitted-close has no unchanged prior session')
            previous = ledger.terminal
        snap = previous['receipt']['session']
        C.require(snap['session_id'] == req['session_id'] and before == len(snap['consumed_token_ids']), 'close cursor/session mismatch')
        expected.update(schema='nerkyor-session-close-v1', revision=snap['revision'],
                        last_turn_receipt_sha256=previous['receipt_sha256'], reset_completed=True)
        C.require(set(receipt) == set(expected), 'unknown close proof fields')
    for key, value in expected.items():
        C.require(type(receipt.get(key)) is type(value) and receipt[key] == value, 'bad cleanup proof: ' + key)
    return copy.deepcopy(committed)


def persisted_cancellation(root, committed):
    """Require the envelope, cleanup proof and any irreversible turn on disk."""
    r = committed['receipt']; rid = r['request_id']
    C.require(valid_id(rid) and valid_id(r['cancelled_request_id']), 'invalid cancellation file identity')
    raw = read_bounded(Path(root) / 'cancellations' / (rid + '.json'))
    C.require(C.sha256(raw) == committed['receipt_sha256'] and same_json(C.strict_json(raw), r),
              'durable cancellation resolution mismatch')
    if r['cleanup_proof'] is not None:
        proof = r['cleanup_proof']
        persisted_receipt(root, rid, proof['receipt_sha256'], proof['receipt'])
    original = Path(root) / 'turns' / (r['cancelled_request_id'] + '.json')
    if r['outcome'] == 'committed_session_closed':
        persisted_receipt(root, r['cancelled_request_id'], r['original_turn_receipt_sha256'])
    else:
        C.require(not original.exists(), 'unpublished/unadmitted turn has a successful or partial receipt')
    return copy.deepcopy(committed)