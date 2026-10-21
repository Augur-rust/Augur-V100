"""Augur-only canonical text/tool Chat Completions over the Rust token HTTP owner.

Original BF16 head, full 262144 KV8; native-advertised greedy or fixed target sampling. DFlash/Vision require explicit profiles; restore remains closed.
Reuses the existing authenticated serial API/tool validation, not its GGUF ledger.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import threading
import time
import uuid
from types import SimpleNamespace
import qwen35_api as A
import qwen35_chat as C
import qwen35_tools as T
import augur_reasoning as R
import augur_vision as V

SAMPLING_PRESET = 'ninfer_thinking_v1'
# (verification, rng) pairs the native owner may advertise; all preserve the target distribution.
SAMPLING_CONTRACTS = {('deterministic_draft_target_sample_equality_v1', 'splitmix64_absolute_position_v1'),
    ('sparse_p_over_q_positive_residual_v1', 'splitmix64_absolute_position_independent_purpose_domains_v1'),
    ('block_verification_sun2025_sparse_v1', 'splitmix64_absolute_position_independent_purpose_domains_v1')}
SAMPLING_VALUES = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
                       presence_penalty=0.0, frequency_penalty=0.0, seed=0)


PROFILE = 'quasar_256ki_kv8_bf16_head_greedy_v1'
AUGUR_PROFILE = 'quasar_b4352_rotated_decode_http_v1'
DFLASH_PROFILE = 'quasar_b4352_dflash_b8_packed_http_v1'
PROFILES = (PROFILE, AUGUR_PROFILE, DFLASH_PROFILE)
CAPACITY = 262144
MAX_OUTPUT = 249036
DEFAULT_OUTPUT = 256


def configure_context(capacity=262144, default_output=256):
    from augur_launch_config import CONTEXT_CHOICES
    C.require(type(capacity) is int and capacity in CONTEXT_CHOICES, 'invalid context choice')
    C.require(type(default_output) is int and 1 <= default_output <= capacity * 95 // 100,
              'default output exceeds selected context budget')
    global CAPACITY, MAX_OUTPUT, DEFAULT_OUTPUT
    CAPACITY, MAX_OUTPUT, DEFAULT_OUTPUT = capacity, capacity * 95 // 100, default_output



def token_sha(ids):
    C.require(type(ids) is list and all(type(t) is int and 0 <= t < 248320 for t in ids), 'invalid Augur tokens')
    # Preserve the native little-endian u32 wire digest without per-token byte objects.
    return hashlib.sha256(struct.pack('<' + str(len(ids)) + 'I', *ids)).hexdigest()


class Template(C.ChatTemplate):
    def __init__(self, root, manifest_path, expected_sha256, profile=PROFILE, vision=False):
        C.require(profile in PROFILES, 'unsupported Augur profile')
        self.profile = profile
        self.dflash = profile == DFLASH_PROFILE
        self.vision = vision
        C.require(C.digest_string(expected_sha256), 'pinned Augur manifest required')
        raw = Path(manifest_path).read_bytes()
        C.require(len(raw) <= 1024 * 1024 and C.sha256(raw) == expected_sha256, 'Augur manifest hash mismatch')
        m = C.strict_json(raw)
        C.require(m.get('schema') == 'quasar-session-model-manifest-v1'
                  and m.get('upstream_repo') == 'QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4', 'Augur manifest profile')
        super().__init__(root)
        for local, bound in [('tokenizer_sha256', 'tokenizer_sha256'), ('template_sha256', 'chat_template_sha256'),
                             ('generation_sha256', 'generation_config_sha256')]:
            C.require(self.identity[local] == m.get(bound), 'Augur client metadata mismatch: ' + local)
        config_raw = (Path(root) / 'tokenizer_config.json').read_bytes()
        C.require(C.sha256(config_raw) == m.get('tokenizer_config_sha256'), 'QUASAR tokenizer config hash')
        config = C.strict_json(config_raw)
        C.require(isinstance(config.get('chat_template'), str)
                  and C.sha256(config['chat_template'].encode()) == self.identity['template_sha256'], 'QUASAR template copies disagree')
        self.manifest, self.manifest_sha256 = m, expected_sha256

    def render(self, messages, tools=None, add_generation_prompt=True):
        thinking = getattr(self, 'reasoning_policy', R.policy())
        if thinking['enabled']:
            messages = R.instruct(messages, thinking['language'])
        if self.vision:
            import augur_vision as V
            return V.render(self, messages, tools, add_generation_prompt,
                            enable_thinking=thinking['enabled'], reasoning_effort=thinking['effort'])
        return super().render(messages, tools=tools, add_generation_prompt=add_generation_prompt,
                              enable_thinking=thinking['enabled'], reasoning_effort=thinking['effort'])


class Transport(A.AbortableTransport):
    max_frame = 4 * 1024 * 1024  # full256Ki consumed-token receipt, not a reduced context
    # One bounded JSON frame is emitted for every token. The legacy 16 MiB
    # aggregate ceiling cuts off ~88K output tokens even with 256Ki capacity.
    # 128 MiB covers 249036 frames with two <=64-byte IDs plus the final receipt.
    # The parser streams frames; this ceiling does not preallocate a response.
    max_response = 128 * 1024 * 1024
    max_body = V.MAX_REQUEST_BYTES  # image RGB-hex ingress; not response framing

    def __init__(self, url, timeout, profile=PROFILE):
        super().__init__(url, timeout)
        C.require(profile in PROFILES, 'unsupported Augur profile')
        self.profile = profile

    def health(self):
        conn, response = self._connect('GET', '/health')
        try:
            raw = response.read(65537)
            C.require(len(raw) <= 65536, 'oversized Augur health')
            value = C.strict_json(raw)
            C.require(value.get('profile') == self.profile and value.get('schema') == 'quasar-token-http-v1', 'wrong Augur HTTP profile')
            return value
        finally:
            response.close(); conn.close()

    def close_session(self, ledger):
        rid = 'close-' + uuid.uuid4().hex
        frames = list(self.frames('/augur/v1/close', dict(request_id=rid, session_id=ledger.session_id,
            revision=ledger.terminal['session']['revision'])))
        C.require(len(frames) == 1 and frames[0][0].get('event') == 'closed'
                  and frames[0][0].get('request_id') == rid
                  and frames[0][0].get('session_id') == ledger.session_id
                  and C.digest_string(frames[0][0].get('receipt_sha256')), 'Augur close terminal/framing')

    def shutdown(self):
        frames = list(self.frames('/augur/v1/shutdown', {}))
        C.require(len(frames) == 1 and frames[0][0].get('event') == 'shutdown', 'Augur shutdown/framing')
        return frames[0][0]


class Ledger:
    def __init__(self, template, policy=None):
        self.template, self.tool_policy = template, copy.deepcopy(policy)
        self.reasoning_policy = copy.deepcopy(getattr(template, 'reasoning_policy', R.policy()))
        self.session_id = 'augur-' + uuid.uuid4().hex
        self.messages, self.history, self.terminal = [], [], None
        self.prefix, self.last_text, self.poisoned = '', '', False

    def render(self, messages, generation=True):
        if self.tool_policy is not None:
            return T.Ledger.render(self, messages, generation)
        return self.template.render(messages, add_generation_prompt=generation)

    def plan(self, messages, budget):
        C.require(not self.poisoned and type(budget) is int and 1 <= budget <= MAX_OUTPUT, 'Augur session/output budget')
        full = self.render(messages)
        if self.terminal is None:
            ids = self.template.encode(full)
            inp = dict(kind='open', prompt_token_ids=ids)
        else:
            C.require(messages[:len(self.messages)] == self.messages, 'Augur exact transcript mismatch')
            pending = self.terminal['session']['pending_token']
            if self.tool_policy is not None:
                C.require(self.terminal['stop_reason'] == 'eos' and pending == self.template.end, 'tool append needs message EOS')
                canonical = self.render(self.messages, generation=False)
                C.require(canonical.endswith('<|im_end|>\n') and full.startswith(canonical), 'tool template rewrote history')
                suffix = full[len(canonical)-1:]
                # Retain actual model XML/tokens, not re-serialized API arguments.
                full = self.prefix + self.last_text + '<|im_end|>' + suffix
            else:
                canonical = self.render(self.messages, generation=False)
                C.require(canonical.endswith('<|im_end|>\n') and full.startswith(canonical),
                          'QUASAR template rewrote transcript history')
                # Templates trim assistant text, while generated IDs may encode
                # leading/trailing spaces. Authenticate the rendered transcript
                # boundary, then append only its new suffix to the exact native
                # history. Never re-tokenize previously generated text.
                suffix = full[len(canonical)-1:]
                if self.terminal['stop_reason'] == 'eos':
                    C.require(pending == self.template.end, 'non-message EOS cannot be continued')
                else:
                    suffix = '<|im_end|>' + suffix
                full = self.prefix + self.last_text + ('<|im_end|>' if self.terminal['stop_reason'] == 'eos' else '') + suffix
            suffix_ids = self.template.encode(suffix)
            C.require(suffix_ids, 'empty Augur suffix')
            ids = self.history + [pending] + suffix_ids
            # Generated BPE tokens need not be the tokenizer's preferred encoding
            # of their decoded text. Preserve the native exact IDs, never replace
            # them by re-encoding the transcript. Only the new canonical suffix is
            # encoded here; its exact transcript prefix was checked above.
            inp = dict(kind='append', revision=self.terminal['session']['revision'],
                exact_history=self.history.copy(), suffix_token_ids=suffix_ids)
        token_sha(ids)
        image_identity = None
        if getattr(self.template, 'vision', False):
            import augur_vision as V
            images = V.collect(messages)
            if images:
                image_identity = V.identity(images)
                if self.terminal is None:
                    inp = dict(kind='open_images', prompt_token_ids=ids, images=images)
                else:
                    C.require(image_identity == self.terminal['session'].get('image_input_sha256'), 'image history identity changed')
                    inp.update(kind='append_images', image_input_sha256=image_identity)
        C.require(ids and len(ids) < CAPACITY, 'Augur selected context budget exhausted; no history truncation')
        # The HTTP ceiling is a maximum, not a reservation of unavailable context.
        budget = min(budget, CAPACITY - len(ids))
        C.require(len(full.encode()) <= 8 * 1024 * 1024, 'canonical prompt byte bound')
        payload = dict(request_id='turn-' + uuid.uuid4().hex, session_id=self.session_id, max_new_tokens=budget, input=inp)
        return dict(path='/augur/v1/turn', payload=payload, input_ids=ids, messages=copy.deepcopy(messages), prefix=full,
                    image_identity=image_identity)

    def accept(self, plan, terminal, text):
        C.require(not self.poisoned, 'uncertain Augur ledger')
        self.history = terminal['session']['consumed_token_ids'].copy()
        self.terminal = terminal
        self.prefix, self.last_text = plan['prefix'], text
        self.messages = plan['messages'] + [dict(role='assistant', content=text, reasoning_content='', tool_calls=[])]


class Decoder:
    """Publish only append-stable decoded text; bind a terminal UTF8 flush to EOF."""
    def __init__(self, tokenizer, fast=False):
        self.tokenizer, self.tokens, self.text = tokenizer, [], ''
        self.fast, self.last_decoded = fast, ''

    def push(self, token):
        self.tokens.append(token)
        decoded = self.tokenizer.decode(self.tokens, skip_special_tokens=False)
        self.last_decoded = decoded
        C.require(decoded.startswith(self.text), 'Augur decoder rewrote published text')
        delta = decoded[len(self.text):]
        if '\ufffd' in delta:
            return ''
        self.text = decoded
        return delta

    def finish(self):
        decoded = self.last_decoded if self.fast else self.tokenizer.decode(self.tokens, skip_special_tokens=False)
        C.require(decoded.startswith(self.text), 'terminal decode rewrote published text')
        delta = decoded[len(self.text):]
        self.text = decoded
        return delta


class StreamDecoder:
    """Native tokenizer streaming state; one final full decode checks the text boundary."""
    def __init__(self, tokenizer):
        from tokenizers.decoders import DecodeStream
        self.tokenizer, self.stream = tokenizer, DecodeStream(skip_special_tokens=False)
        self.tokens, self.parts, self.pending = [], [], ''
        self._text = None

    @property
    def text(self):
        if self._text is None: self._text = ''.join(self.parts)
        return self._text

    def push(self, token):
        self.tokens.append(token)
        self.pending += self.stream.step(self.tokenizer, token) or ''
        if '\ufffd' in self.pending: return ''
        delta, self.pending = self.pending, ''
        if delta:
            self.parts.append(delta); self._text = None
        return delta

    def finish(self):
        decoded = self.tokenizer.decode(self.tokens, skip_special_tokens=False)
        published = self.text
        C.require(decoded.startswith(published), 'terminal streaming decode rewrote published text')
        delta = decoded[len(published):]
        self.parts, self._text, self.pending = [decoded], decoded, ''
        return delta


class ByteLevelTailDecoder:
    """Decode only the uncommitted UTF8 suffix for the pinned ByteLevel tokenizer."""
    def __init__(self, tokenizer):
        from tokenizers.decoders import ByteLevel
        C.require(isinstance(tokenizer.decoder, ByteLevel), 'incremental suffix requires ByteLevel decoder')
        self.tokenizer, self.tokens, self.pending_ids, self.parts = tokenizer, [], [], []
        self.pending_text, self._text = '', None

    @property
    def text(self):
        if self._text is None: self._text = ''.join(self.parts)
        return self._text

    def push(self, token):
        self.tokens.append(token); self.pending_ids.append(token)
        self.pending_text = self.tokenizer.decode(self.pending_ids, skip_special_tokens=False)
        if '\ufffd' in self.pending_text: return ''
        # ByteLevel concatenates token bytes. A complete UTF8 prefix cannot be
        # modified by later bytes, so only incomplete suffix IDs stay pending.
        delta = self.pending_text
        self.pending_ids.clear(); self.pending_text = ''
        if delta: self.parts.append(delta); self._text = None
        return delta

    def finish(self):
        # A truncated final codepoint follows the old full decoder's replacement
        # behavior; it must not silently disappear at EOS or length.
        delta = self.pending_text
        if delta: self.parts.append(delta)
        self.pending_ids.clear(); self.pending_text = ''
        self._text = ''.join(self.parts)
        return delta


class ConsumedDigest:
    """Hash validated input once; hold the last generated token pending until EOS/length."""
    def __init__(self, ids):
        self.digest = hashlib.sha256(struct.pack('<' + str(len(ids)) + 'I', *ids))
        self.pending = None
        self.count = 0

    def push(self, token):
        if self.pending is not None: self.digest.update(struct.pack('<I', self.pending))
        self.pending = token
        self.count += 1

    def final(self, published):
        digest = self.digest.copy()
        if not published and self.pending is not None: digest.update(struct.pack('<I', self.pending))
        return digest.hexdigest()


def _copy_receipt_value(value):
    """Copy JSON receipt containers without invoking deepcopy's general graph walk.

    Native receipts are JSON-shaped (dict/list/scalar). The previous implementation
    used ``copy.deepcopy`` with two memo overrides so the token vectors were not
    shared. That is correct but needlessly pays deepcopy's generic dispatch cost on
    the terminal path. Keep the same independent-container guarantee for the JSON
    contract, and retain a defensive fallback for tests or future non-JSON values.
    """
    if isinstance(value, dict):
        return {key: _copy_receipt_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_receipt_value(item) for item in value]
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return copy.deepcopy(value)


def copy_verified_receipt(receipt):
    # Both vectors have already been compared with validated immutable token IDs.
    # Copy every JSON container so the retained native receipt cannot alias the
    # terminal frame or its nested session/detail dictionaries.
    return _copy_receipt_value(receipt)


def validate_dflash_receipt(receipt, profile):
    active = profile == DFLASH_PROFILE
    C.require(receipt.get('dflash_enabled') is active, 'Augur DFlash receipt capability mismatch')
    detail = receipt.get('decode_detail') or {}
    d = detail.get('dflash') or {}
    if not active:
        C.require(d.get('active', False) is False, 'unexpected active DFlash detail')
        return
    C.require(d.get('profile') == profile and d.get('active') is True
              and d.get('block_width') == 8 and d.get('target_projection') == 'packed_b8'
              and d.get('draft_projection') == 'llama' and d.get('draft_head') == 'llama_b8'
              and d.get('context') == 'retained' and d.get('B1_numerical_equivalence_qualified') is False,
              'Augur DFlash numerical policy mismatch')
    keys = ('rounds', 'proposed_draft_tokens', 'accepted_draft_tokens', 'checked_draft_tokens')
    C.require(all(type(d.get(k)) is int and d[k] >= 0 for k in keys), 'invalid DFlash counters')
    C.require(d['proposed_draft_tokens'] == d['rounds'] * 7
              and d['accepted_draft_tokens'] <= d['checked_draft_tokens'] <= d['proposed_draft_tokens'],
              'DFlash acceptance denominator mismatch')
    gate = d.get('adaptive_gate_decisions')
    if gate is not None:
        C.require(isinstance(gate, dict) and gate.get('policy') in
                  ('legacy_cumulative_checked_acceptance', 'measured_decode_cost_v1', 'deferred_decode_cost_v2', 'fixed_b8_v1'),
                  'unsupported DFlash adaptive policy')
        if gate['policy'] in ('measured_decode_cost_v1', 'deferred_decode_cost_v2', 'fixed_b8_v1'):
            from augur_dflash_cost_accounting import validate
            validate(receipt)


def input_prefix_cached_tokens(receipt, plan):
    detail = receipt.get('prefill_detail')
    cache = detail.get('input_prefix_cache') if isinstance(detail, dict) else None
    if cache is None:
        return 0
    C.require(isinstance(cache, dict) and plan['payload']['input']['kind'] == 'open'
              and not plan.get('image_identity'), 'unexpected Augur prefix cache receipt')
    matched, count = cache.get('matched_tokens'), len(plan['input_ids'])
    C.require(type(matched) is int and 0 <= matched <= count
              and cache.get('hit') is (matched > 0)
              and cache.get('exact_input_hit') is (matched == count)
              and type(cache.get('new_prefill_tokens')) is int
              and cache['new_prefill_tokens'] == count - matched
              and cache.get('source') == 'native_gdn_conv_kv_feature_logits_state',
              'invalid Augur prefix cache accounting')
    return matched


def verify_terminal(frame, plan, ledger, tokens, ceiling, consumed_digest=None):
    rid, sid = plan['payload']['request_id'], ledger.session_id
    C.require(frame.get('event') == 'terminal' and frame.get('request_id') == rid and frame.get('session_id') == sid
              and frame.get('product_ready') is False, 'Augur terminal identity/claims')
    committed = frame.get('committed', {})
    r = committed.get('receipt', {}); s = r.get('session', {})
    reference = frame.get('receipt_transport') == 'bound_receipt_v1'
    compact = frame.get('receipt_vectors')
    if reference:
        C.require(isinstance(compact, dict) and compact.get('schema') == 'from_stream_v1'
                  and set(compact) == {'schema','output_tokens','consumed_tokens'}
                  and type(compact.get('output_tokens')) is int and compact['output_tokens'] == len(tokens)
                  and type(compact.get('consumed_tokens')) is int, 'Augur compact receipt vectors')
        C.require('output_token_ids' not in r and 'consumed_token_ids' not in s,
                  'Augur compact receipt must omit vectors')
    else:
        C.require(frame.get('receipt_transport', 'inline_v1') == 'inline_v1'
                  and compact is None and r.get('output_token_ids') == tokens, 'Augur receipt/output mismatch')
    C.require(C.digest_string(committed.get('receipt_sha256')) and r.get('schema') == 'quasar-token-turn-receipt-v1'
              and r.get('request_id') == rid, 'Augur receipt identity')
    for key in ('product_ready',):
        C.require(r.get(key) is False, 'unsupported Augur receipt claim: ' + key)
    C.require(r.get('vision_enabled') is bool(plan.get('image_identity'))
              and s.get('image_input_sha256') == plan.get('image_identity'), 'QUASAR image receipt identity')
    previous = ledger.terminal
    C.require(r.get('previous_receipt_sha256') == (previous['native_report_sha256'] if previous else None), 'Augur receipt chain')
    C.require(s.get('profile') == getattr(ledger.template, 'profile', PROFILE) and s.get('manifest_sha256') == ledger.template.manifest_sha256
              and s.get('session_id') == sid and type(s.get('revision')) is int
              and s['revision'] == (previous['session']['revision'] + 1 if previous else 1), 'Augur session identity/revision')
    pending, published = s.get('pending_token'), s.get('pending_token_published')
    C.require(type(pending) is int and pending in ledger.template.vocab and type(published) is bool, 'Augur pending identity')
    selected = len(tokens) + int(not published)
    C.require(type(r.get('selected_tokens_including_eos')) is int and r['selected_tokens_including_eos'] == selected
              and 1 <= selected <= ceiling, 'Augur selection count/budget')
    if published:
        C.require(r.get('stop_reason') == 'length' and len(tokens) == ceiling and tokens[-1] == pending, 'length pending mismatch')
        history = plan['input_ids'] + tokens[:-1]
    else:
        C.require(r.get('stop_reason') == 'eos' and pending in ledger.template.eos, 'EOS pending mismatch')
        history = plan['input_ids'] + tokens
    if reference:
        C.require(compact.get('consumed_tokens') == len(history), 'Augur compact history count')
        r['output_token_ids'] = tokens.copy()
        s['consumed_token_ids'] = history.copy()
    if consumed_digest is not None:
        C.require(consumed_digest.count == len(tokens), 'Augur streamed digest token count')
    expected_hash = consumed_digest.final(published) if consumed_digest is not None else token_sha(history)
    C.require(s.get('consumed_token_ids') == history and s.get('consumed_input_sha256') == expected_hash, 'Augur exact consumed ledger')
    validate_dflash_receipt(r, getattr(ledger.template, 'profile', PROFILE))
    return dict(schema='quasar-api-native-terminal-v1', session=s, stop_reason=r['stop_reason'],
                cached_input_tokens=input_prefix_cached_tokens(r, plan),
                visible_token_ids=tokens.copy(), native_report_sha256=committed['receipt_sha256'],
                native_receipt=copy_verified_receipt(r) if consumed_digest is not None else copy.deepcopy(r), native_terminal_wall_s=frame.get('wall_s_through_receipt'),
                native_pre_receipt_unix_ns=frame.get('native_pre_receipt_unix_ns'),
                native_receipt_commit_s=frame.get('native_receipt_commit_s'),
                producer_before_terminal_unix_ns=frame.get('producer_before_terminal_unix_ns'),
                adapter_terminal_received_unix_ns=frame.get('adapter_terminal_received_unix_ns'),
                adapter_before_terminal_unix_ns=frame.get('adapter_before_terminal_unix_ns'),
                adapter_terminal_check_s=frame.get('adapter_terminal_check_s'),
                adapter_protocol_check_s=frame.get('adapter_protocol_check_s'),
                adapter_receipt_readback_s=frame.get('adapter_receipt_readback_s'))


class Controller(A.Controller):
    def cached_prompt_tokens(self, meta, plan, terminal):
        if meta['session_mode'] == 'canonical_full_history_open':
            return terminal['cached_input_tokens']
        return super().cached_prompt_tokens(meta, plan, terminal)

    def __init__(self, template, backend, model, timeout=14400, metrics=None, transport_factory=None, health_factory=None):
        self.template, self.backend, self.model, self.metrics = template, backend, model, metrics
        # auto: guard greedy decoding only. A greedy loop is an absorbing state; a sampled target
        # leaves any cycle on its own, so the guard would only spend time on text it never rejects.
        self.reasoning_loop_guard = os.environ.get('AUGUR_REASONING_LOOP_GUARD', 'auto')
        C.require(self.reasoning_loop_guard in ('auto', 'on', 'off'), 'reasoning loop guard must be auto, on or off')
        self.http_decode_policy = os.environ.get('AUGUR_HTTP_DECODE', 'legacy')
        C.require(self.http_decode_policy in ('legacy', 'stream_v1', 'bytelevel_v2'), 'invalid HTTP decode policy')
        self.profile = getattr(template, 'profile', PROFILE)
        self.english_selector_experiment = R.selector_experiment(os.environ, self.profile == DFLASH_PROFILE)
        self.factory = transport_factory or (lambda: Transport(backend, timeout, self.profile))
        raw_health_factory = health_factory or transport_factory or (lambda: Transport(backend, min(timeout, A.HEALTH_TIMEOUT), self.profile))
        # Revalidate the pinned model on recovery too. An operator-restarted
        # backend at the same port must not emit another model's partial text.
        self.health_factory = lambda: SimpleNamespace(
            health=lambda: self.check_health(raw_health_factory().health()))
        self.busy, self.active_lock = threading.Lock(), threading.Lock()
        self.active, self.current, self.faulted = None, None, False
        self.failed_request = None
        self.restore_used, self.restore_required = False, False
        self.health = self.health_factory().health()
        finish_policy = os.environ.get('AUGUR_HTTP_FINISH', 'legacy')
        C.require(finish_policy in ('legacy', 'fused_v1', 'receipt_handle_v2', 'receipt_ref_v3'), 'unsupported HTTP finish policy')
        self.fast_finish = finish_policy != 'legacy'
        self.http_finish_policy = finish_policy
        if self.fast_finish:
            # Account cold tool dependency initialization in startup, not the
            # first accepted tool request. No tool schemas/arguments are trusted.
            from jsonschema import Draft202012Validator
        self.request_budgets = True
        self.max_body = V.MAX_REQUEST_BYTES if getattr(template, 'vision', False) else A.MAX_BODY
        self.limits = dict(capacity=CAPACITY, initial_output_limit=DEFAULT_OUTPUT, continuation_output_limit=DEFAULT_OUTPUT,
            initial_max_output_limit=MAX_OUTPUT, continuation_max_output_limit=MAX_OUTPUT, request_output_budget_supported=True)

    def check_health(self, h):
        template = self.template
        C.require(h.get('profile') == self.profile and h.get('capacity') == CAPACITY and h.get('maximum_output_tokens') == MAX_OUTPUT
                  and h.get('manifest_sha256') == template.manifest_sha256 and h.get('model_manifest') == template.manifest
                  and h.get('kv') == 'Q8_0' and h.get('lm_head') == 'original_bf16', 'Augur API/native identity mismatch')
        for key in ('product_ready', 'qualified', 'actual_long_history_qualified', 'restore_enabled'):
            C.require(h.get(key) is False, 'unverified Augur capability: ' + key)
        C.require(h.get('dflash_enabled') is (self.profile == DFLASH_PROFILE), 'Augur DFlash capability mismatch')
        C.require(h.get('vision_enabled') is getattr(template, 'vision', False), 'QUASAR Vision capability mismatch')
        sampling = h.get('sampling', dict(preset='greedy', greedy=True))
        C.require(type(sampling) is dict and sampling.get('preset') in ('greedy', SAMPLING_PRESET),
                  'Unsupported native target sampler')
        if sampling.get('preset') == SAMPLING_PRESET:
            C.require(sampling.get('greedy') is False and all(
                type(sampling.get(k)) in (int, float) and sampling.get(k) == v
                for k, v in SAMPLING_VALUES.items() if k != 'seed') and
                type(sampling.get('seed')) is int and 0 <= sampling['seed'] <= (1 << 64) - 1 and
                (sampling.get('verification'), sampling.get('rng')) in SAMPLING_CONTRACTS, 'Native sampling contract mismatch')
        return h

    def capabilities(self):
        vision = getattr(self.template, 'vision', False)
        return dict(model=self.model, profile=self.profile, text_only=not vision,
            greedy=self.health.get('sampling', {}).get('preset', 'greedy') == 'greedy',
            sampling=self.health.get('sampling', dict(preset='greedy', temperature=0.0, top_p=1.0, greedy=True)),
            n=1, stream=True,
            thinking=True, thinking_enabled_by_default=False, default_thinking_language=R.DEFAULT_LANGUAGE,
            thinking_languages=list(R.LANGUAGES), thinking_policy_version=R.POLICY_VERSION,
            reasoning_content=True, thinking_budget_includes_reasoning_and_answer=True,
            full_kv_capacity=CAPACITY, lm_head='original_bf16', dflash_enabled=self.profile == DFLASH_PROFILE, vision_forward=vision,
            exact_history_reuse=True, tool_calls=True, image_tool_calls=vision, tool_execution='client_only',
            image_min_pixels=dict(low=65536,auto=524288,high=524288) if vision else None,
            image_count_limit=None, image_history_policy='preserve_all_within_resource_budget' if vision else None,
            image_max_pixels_per_image=V.MAX_IMAGE_PIXELS if vision else None,
            image_max_total_pixels=V.MAX_TOTAL_IMAGE_PIXELS if vision else None,
            image_max_tokens=V.MAX_IMAGE_TOKENS if vision else None,
            request_max_bytes=self.max_body, max_messages=C.MAX_MESSAGES,
            user_steering_after_tool_results=True,
            tool_delta_publication='validated_complete_calls_after_native_http_eof', strict_constrained_generation=False,
            startup_checkpoint_restore=False, arbitrary_output_budget=True, native_limits=self.limits,
            completion_authority='native_generation_only_not_tool_execution_or_code_verification',
            request_half_close='treated_as_cancel', pipelining=False,
            product_ready=False, qualified=False, actual_long_history_qualified=False, visible_tps=False)

    def restore_saved(self, *args):
        raise A.ApiError('Augur persistent state restore is not implemented.', 409, 'restore_unavailable')

    def recover_faulted(self):
        h = self.health_factory().health()
        if h.get('owner_resident') is False:
            self.current, self.faulted, self.failed_request = None, False, None
            return dict(mode='retired_owner_fresh_open')
        value = h.get('last_cancel_discard')
        receipt = value.get('receipt') if isinstance(value, dict) else None
        if not (self.failed_request and isinstance(receipt, dict)
                and h.get('owner_resident') is True and h.get('session_active') is False
                and h.get('request_bound_cancellation_discard') is True
                and C.digest_string(h.get('producer_sha256'))
                and h.get('producer_sha256') == self.health.get('producer_sha256')
                and C.digest_string(value.get('receipt_sha256'))
                and receipt.get('schema') == 'quasar-session-cancellation-discard-v1'
                and receipt.get('cancelled_request_id') == self.failed_request[0]
                and receipt.get('session_id') == self.failed_request[1]
                and receipt.get('manifest_sha256') == self.template.manifest_sha256
                and type(receipt.get('request_id')) is str and 1 <= len(receipt['request_id']) <= 64
                and receipt.get('drain_completed') is True and receipt.get('reset_completed') is True
                and receipt.get('session_discarded') is True
                and receipt.get('previous_history_reusable') is False
                and receipt.get('product_ready') is False and receipt.get('dflash_enabled') is (self.profile == DFLASH_PROFILE)
                and type(receipt.get('submitted_tokens_after_discard')) is int
                and receipt['submitted_tokens_after_discard'] == 0
                and type(receipt.get('submitted_tokens_before_discard')) is int
                and 0 <= receipt['submitted_tokens_before_discard'] <= CAPACITY
                and type(receipt.get('cancellation_boundary')) is str and 0 < len(receipt['cancellation_boundary']) < 256):
            raise A.ApiError('Native state is uncertain; a bound cancellation-discard receipt or owner retirement is required.',
                             503, 'native_state_uncertain')
        self.current, self.faulted, self.failed_request = None, False, None
        return dict(mode='verified_cancel_discard_fresh_session', receipt_sha256=value['receipt_sha256'])

    def validate_request(self, payload):
        A.need(type(payload) is dict, 'A JSON object is required.')
        clean = dict(payload)
        sampled = self.health.get('sampling', {}).get('preset') == SAMPLING_PRESET
        if sampled:
            # Each native owner has a fixed preset and seed. Accept only parameters it
            # actually executes, including omitted parameters using that preset.
            expected_values = dict(SAMPLING_VALUES, seed=self.health['sampling']['seed'])
            for key, expected in expected_values.items():
                value = clean.pop(key, None)
                if value is not None:
                    valid = type(value) is int if key in ('seed', 'top_k') else A.number(value)
                    A.need(valid and value == expected,
                           f'This service uses {key}={expected}; restart with a different supported preset to change it.', key)
        # Client-returned reasoning is display metadata. Native append keeps
        # its original token ledger; never accept client-authored hidden state.
        if isinstance(clean.get('messages'), list):
            clean['messages'] = []
            for message in payload['messages']:
                if isinstance(message, dict) and message.get('role') == 'assistant' and 'reasoning_content' in message:
                    A.need(type(message['reasoning_content']) is str,
                           'reasoning_content must be text.', 'messages')
                    message = {k: v for k, v in message.items() if k != 'reasoning_content'}
                clean['messages'].append(message)
        effort = clean.pop('reasoning_effort', None)
        enabled = clean.pop('enable_thinking', effort is not None and effort not in ('off', 'none'))
        language = clean.pop('thinking_language', R.DEFAULT_LANGUAGE)
        try:
            thinking = R.policy(enabled, language, effort)
        except ValueError as error:
            raise A.ApiError(str(error), param='enable_thinking/thinking_language/reasoning_effort') from error
        if getattr(self.template, 'vision', False):
            import augur_vision as V
            result = V.validate_request(clean, self.model)
        else:
            result = super().validate_request(clean)
        A.need(result.get('budget') is None or result['budget'] <= MAX_OUTPUT,
               'Token ceiling exceeds the selected context output limit.', 'max_completion_tokens')
        result['reasoning_policy'] = thinking
        result['sampling'] = self.health.get('sampling', dict(preset='greedy', greedy=True))
        return result

    def loop_guard_active(self):
        if self.reasoning_loop_guard != 'auto':
            return self.reasoning_loop_guard == 'on'
        return self.health.get('sampling', {}).get('preset', 'greedy') == 'greedy'

    def output_adapter(self, request, content_emit, emit):
        if request['reasoning_policy']['enabled']:
            return R.Output(content_emit, emit, loop_guard=self.loop_guard_active())
        return None

    def prepare(self, request):
        messages, policy = request['messages'], request.get('tool_policy')
        current = self.current
        thinking = request.get('reasoning_policy', R.policy())
        reused = (current is not None and not current.poisoned and current.tool_policy == policy
                  and current.reasoning_policy == thinking
                  and len(messages) > len(current.messages) and messages[:len(current.messages)] == current.messages)
        if reused and policy is None:
            reused = len(messages) == len(current.messages) + 1
        if reused and policy is not None:
            reused = current.terminal['stop_reason'] == 'eos' and current.terminal['session']['pending_token'] == self.template.end
        if reused and getattr(self.template, 'vision', False):
            import augur_vision as V
            reused = V.collect(messages) == V.collect(current.messages)
        request_template = copy.copy(self.template)
        request_template.reasoning_policy = copy.deepcopy(thinking)
        ledger = Ledger(request_template, policy)
        if reused:
            for key in ('session_id', 'messages', 'prefix', 'last_text', 'history', 'terminal'):
                setattr(ledger, key, copy.deepcopy(getattr(current, key)))
        cap = self.output_budget(request, reused)
        try:
            plan = ledger.plan(messages, cap)
        except (ValueError, TypeError) as e:
            raise A.ApiError(str(e), code='context_or_template_limit', param='messages') from e
        selector = R.selector_request(self.english_selector_experiment, thinking, plan.get('image_identity') is not None)
        if selector is not None:
            if selector == R.ENGLISH_GATE_POLICY:
                C.require(plan['prefix'].endswith('<think>\n'), 'English selector requires open thinking template')
            plan['payload']['selector_policy'] = selector
        return ledger, plan, reused, plan['payload']['max_new_tokens']

    def run_native_turn(self, transport, ledger, plan, emit, ceiling):
        self.failed_request = None
        started, first, last, first_count = time.perf_counter(), None, None, 0
        fast_decode = self.http_decode_policy != 'legacy'
        decoder, terminal = Decoder(self.template.tokenizer, fast=fast_decode), None
        if fast_decode:
            from tokenizers import Tokenizer
            if isinstance(self.template.tokenizer, Tokenizer):
                from tokenizers.decoders import ByteLevel
                decoder = ByteLevelTailDecoder(self.template.tokenizer) if self.http_decode_policy == 'bytelevel_v2' and isinstance(self.template.tokenizer.decoder, ByteLevel) else StreamDecoder(self.template.tokenizer)
        consumed_digest = ConsumedDigest(plan['input_ids']) if fast_decode else None
        terminal_received_s = terminal_verify_s = None
        terminal_received_unix_ns = None
        text_decode_ns = 0
        frames = transport.frames(plan['path'], plan['payload'])
        try:
            for frame, elapsed in frames:
                C.require(not transport.cancelled.is_set() and terminal is None, 'cancelled or post-terminal frame')
                if frame.get('event') == 'token':
                    t = frame.get('token', {})
                    token = t.get('token_id')
                    C.require(frame.get('product_ready') is False and t.get('request_id') == plan['payload']['request_id']
                        and t.get('session_id') == ledger.session_id and type(t.get('sequence')) is int
                        and t['sequence'] == len(decoder.tokens) and len(decoder.tokens) < ceiling
                        and type(token) is int and token in self.template.vocab and token not in self.template.eos, 'foreign/out-of-order token frame')
                    if consumed_digest is not None: consumed_digest.push(token)
                    decode_started_ns = time.perf_counter_ns()
                    delta = decoder.push(token)
                    text_decode_ns += time.perf_counter_ns() - decode_started_ns
                    if delta:
                        if first is None: first = elapsed
                        last = elapsed
                        emit(delta)
                    if elapsed == first: first_count = len(decoder.tokens)
                elif frame.get('event') == 'error':
                    C.require(frame.get('request_id') == plan['payload']['request_id']
                        and frame.get('session_id') == ledger.session_id
                        and frame.get('product_ready') is False, 'foreign native error frame')
                    # This is a failed request, never a terminal or a committed completion.
                    # Details stay in private native stderr; do not echo arbitrary payloads.
                    raise C.ProtocolError('native engine failed before a committed terminal; see native owner stderr.log')
                else:
                    terminal_received_s = elapsed
                    terminal_received_unix_ns = str(time.time_ns())
                    verify_started = time.perf_counter()
                    terminal = verify_terminal(frame, plan, ledger, decoder.tokens, ceiling, consumed_digest)
                    terminal_verify_s = time.perf_counter() - verify_started
            C.require(terminal is not None and not transport.cancelled.is_set(), 'native HTTP ended without committed terminal')
            # Only the complete zero HTTP chunk authorizes decoder flush/ledger adoption.
            terminal_decode_started_ns = time.perf_counter_ns()
            flush = decoder.finish()
            terminal_decode_s = (time.perf_counter_ns() - terminal_decode_started_ns) * 1e-9
            if flush:
                elapsed = time.perf_counter() - started
                if first is None: first, first_count = elapsed, len(decoder.tokens)
                last = elapsed
                emit(flush)
            ledger.accept(plan, terminal, decoder.text)
            return terminal, decoder.text, dict(native_request_id=plan['payload']['request_id'],
                sampling=self.health.get('sampling', dict(preset='greedy', greedy=True)),
                decode_mode='sampling' if self.health.get('sampling', {}).get('preset') == SAMPLING_PRESET else 'greedy',
                native_report_sha256=terminal['native_report_sha256'], native_ttft_s=first,
                native_pre_receipt_unix_ns=terminal.get('native_pre_receipt_unix_ns'),
                native_receipt_commit_s=terminal.get('native_receipt_commit_s'),
                native_terminal_received_s=terminal_received_s, native_terminal_verify_s=terminal_verify_s,
                bridge_terminal_received_unix_ns=terminal_received_unix_ns,
                producer_before_terminal_unix_ns=terminal.get('producer_before_terminal_unix_ns'),
                adapter_terminal_received_unix_ns=terminal.get('adapter_terminal_received_unix_ns'),
                adapter_before_terminal_unix_ns=terminal.get('adapter_before_terminal_unix_ns'),
                adapter_terminal_check_s=terminal.get('adapter_terminal_check_s'),
                adapter_protocol_check_s=terminal.get('adapter_protocol_check_s'),
                adapter_receipt_readback_s=terminal.get('adapter_receipt_readback_s'),
                http_decode_policy=self.http_decode_policy, text_decoder=type(decoder).__name__,
                text_decode_s=text_decode_ns * 1e-9, terminal_decode_s=terminal_decode_s,
                frame_clock=C.FRAME_CLOCK, native_http_wall_s=time.perf_counter()-started,
                visible_tokens=len(decoder.tokens), post_first_tps=(len(decoder.tokens)-first_count)/(last-first)
                if first is not None and last > first else None)
        except BaseException:
            ledger.poisoned = True
            self.failed_request = (plan['payload']['request_id'], ledger.session_id)
            raise
        finally:
            frames.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', default='http://127.0.0.1:8293')
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--model', default='quasar-27b-256ki')
    parser.add_argument('--port', type=int, default=8294)
    parser.add_argument('--timeout', type=float, default=14400)
    parser.add_argument('--metrics-file', type=Path)
    parser.add_argument('--profile', choices=list(PROFILES), default=PROFILE)
    parser.add_argument('--vision', action='store_true')
    args = parser.parse_args(argv)
    key = os.environ.get('AUGUR_QWEN35_API_KEY', '')
    A.validate_key(key)
    C.require(0 <= args.port <= 65535 and math.isfinite(args.timeout) and args.timeout > 0, 'invalid port/timeout')
    C.require(1 <= len(args.model) <= 128 and all(c.isascii() and (c.isalnum() or c in '-_.') for c in args.model), 'plain model ID required')
    template = Template(args.model_dir, args.manifest, args.manifest_sha256, args.profile, args.vision)
    metrics = args.metrics_file.open('x', encoding='utf8') if args.metrics_file else None
    controller = server = None
    try:
        controller = Controller(template, args.backend, args.model, args.timeout, metrics=metrics)
        server = A.Server(('127.0.0.1', args.port), controller, key)
        print(json.dumps(dict(event='augur_text_api_listening', port=server.server_address[1], model=args.model,
                              manifest_sha256=args.manifest_sha256, product_ready=False, qualified=False)), flush=True)
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        if controller is not None: controller.close()
        if server is not None: server.server_close()
        if metrics is not None: metrics.close()


if __name__ == '__main__': main()
