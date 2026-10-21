"""Loopback, single-user Chat Completions bridge to the native Qwen service.

Text only, greedy, n=1. Exact full-transcript matches reuse the native token
ledger; unknown histories get a canonical fresh session. No GPU process is
started here. The backend's finite output/owner limits remain explicit.
"""
from __future__ import annotations

import argparse
import copy
import hmac
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import select
import socket
import sys
import threading
import time
import uuid

import qwen35_chat as C
import qwen35_persistence as P
import qwen35_tools as T

MAX_BODY = 8 * 1024 * 1024
HEALTH_TIMEOUT = 5.0


def validate_key(api_key):
    C.require(isinstance(api_key, str) and 16 <= len(api_key) <= 256 and api_key.isascii()
              and all(33 <= ord(c) <= 126 for c in api_key),
              'a private printable ASCII API key of 16..256 characters is required')


class ApiError(ValueError):
    def __init__(self, message, status=400, code='invalid_request', param=None):
        super().__init__(message)
        self.status, self.code, self.param = status, code, param

    def response(self):
        return {'error': {'message': str(self), 'type': 'invalid_request_error' if self.status < 500 else 'server_error',
                          'param': self.param, 'code': self.code}}


def need(condition, message, param=None):
    if not condition:
        raise ApiError(message, param=param)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def restore_inputs(template, path, expected_sha256, profile_path):
    """Validate the operator-pinned private bundle before any backend IO."""
    C.require(C.digest_string(expected_sha256), 'restore requires an external lowercase SHA256')
    bundle = P.read_bundle(path, template, expected_sha256)
    with P.plain_path(profile_path).open('rb') as stream:
        raw = stream.read(65537)
    C.require(len(raw) <= 65536, 'checkpoint profile exceeds native bound')
    profile = C.strict_json(raw)
    C.require(profile.get('schema') == 'cq-qwen35-http-checkpoint-profile-v1'
              and type(profile.get('bindings')) is dict, 'native checkpoint profile required')
    for client_key, native_key in (('tokenizer_sha256', 'tokenizer_sha256'),
                                  ('template_sha256', 'chat_template_sha256'),
                                  ('generation_sha256', 'generation_config_sha256')):
        C.require(template.identity[client_key] == profile['bindings'].get(native_key),
                  'client/profile template identity mismatch: ' + client_key)
    profile_sha = C.sha256(raw)
    C.require(bundle['checkpoint']['profile_sha256'] == profile_sha, 'saved checkpoint profile mismatch')
    return bundle, profile_sha


def validate_request(value, model):
    need(type(value) is dict, 'A JSON object is required.')
    allowed = {'model', 'messages', 'stream', 'stream_options', 'temperature', 'top_p', 'n',
               'max_tokens', 'max_completion_tokens', 'response_format', 'store', 'logprobs',
               'presence_penalty', 'frequency_penalty', 'stop', 'tools', 'tool_choice', 'parallel_tool_calls'}
    unknown = sorted(set(value) - allowed)
    need(not unknown, 'Unsupported parameter: ' + ', '.join(unknown), unknown[0] if unknown else None)
    need(value.get('model') == model, 'Unknown local model.', 'model')
    need(type(value.get('stream', False)) is bool, 'stream must be a boolean.', 'stream')
    for key, expected in (('temperature', 0), ('top_p', 1), ('presence_penalty', 0), ('frequency_penalty', 0)):
        if value.get(key) is not None:
            need(number(value[key]) and value[key] == expected, f'Only {key}={expected} is supported.', key)
    if value.get('n') is not None:
        need(type(value['n']) is int and value['n'] == 1, 'Only n=1 is supported.', 'n')
    for key in ('store', 'logprobs'):
        need(value.get(key) is None or value[key] is False, f'{key}=true is not supported.', key)
    need(value.get('stop') is None, 'Custom stop sequences are not supported.', 'stop')
    need(value.get('response_format') is None or value['response_format'] == {'type': 'text'},
         'Only text response format is supported; JSON schema is not enforced.', 'response_format')
    need(not (value.get('max_tokens') is not None and value.get('max_completion_tokens') is not None),
         'Specify only one token ceiling.', 'max_completion_tokens')
    budget = value.get('max_completion_tokens')
    if budget is None:
        budget = value.get('max_tokens')
    need(budget is None or type(budget) is int and 0 < budget <= 249036,
         'Token ceiling must be a positive bounded integer.', 'max_completion_tokens')
    stream_options = value.get('stream_options')
    if stream_options is not None:
        need(value.get('stream') is True and type(stream_options) is dict
             and set(stream_options) <= {'include_usage'}
             and type(stream_options.get('include_usage', False)) is bool,
             'stream_options supports only boolean include_usage with stream=true.', 'stream_options')
    messages = value.get('messages')
    need(type(messages) is list and bool(messages), 'messages must be a non-empty array.', 'messages')
    need(len(messages) <= C.MAX_MESSAGES,
         f'messages contains {len(messages)} entries; maximum is {C.MAX_MESSAGES}.', 'messages')
    if any(k in value for k in ('tools','tool_choice','parallel_tool_calls')) or any(
            isinstance(m,dict) and (m.get('role')=='tool' or m.get('tool_calls')) for m in messages):
        try:
            selected = T.policy(value)
            if selected['tools'] or any(isinstance(m,dict) and (m.get('role')=='tool' or m.get('tool_calls')) for m in messages):
                result = T.normalize_messages(messages,selected['tools'])
                return dict(messages=result,stream=value.get('stream',False),budget=budget,
                    include_usage=bool(stream_options and stream_options.get('include_usage')),
                    tool_policy=selected)
        except (ValueError,TypeError,KeyError,RecursionError) as error:
            raise ApiError(str(error),param='tools/messages') from error
    result = []
    for index, message in enumerate(messages):
        allowed_fields = ({'role', 'content', 'tool_calls'} if type(message) is dict
                          and message.get('role') == 'assistant' else {'role', 'content'})
        need(type(message) is dict and {'role', 'content'} <= set(message) <= allowed_fields,
             'Messages must contain only role and text content.', 'messages')
        if 'tool_calls' in message:
            need(message['tool_calls'] is None or type(message['tool_calls']) is list
                 and not message['tool_calls'], 'Nonempty calls require declared tools.', 'messages')
        role, text = message['role'], message['content']
        need(role in ('system', 'developer', 'user', 'assistant'), 'Unsupported message role.', 'messages')
        if type(text) is list:
            need(text and all(type(p) is dict and set(p) == {'type', 'text'} and
                 p['type'] == 'text' and isinstance(p['text'], str) for p in text),
                 'Only text content parts are supported.', 'messages')
            text = ''.join(p['text'] for p in text)
        need(isinstance(text, str), 'Only text message content is supported.', 'messages')
        if role in ('system', 'developer'):
            need(index == 0, 'One leading system or developer message is supported.', 'messages')
            role = 'system'  # the model has one canonical system-role slot
        try:
            if role == 'assistant' and text == '':
                # A native first-token EOS legitimately confirms an empty answer.
                normalized = C.ChatTemplate.message(role, '_')
                normalized['content'] = ''
                result.append(normalized)
            else:
                result.append(C.ChatTemplate.message(role, text))
        except ValueError as error:
            raise ApiError(str(error), param='messages') from error
    offset = int(result[0]['role'] == 'system')
    need(len(result) > offset and (len(result) - offset) % 2 == 1,
         'A conversation must end with a user message.', 'messages')
    for index, message in enumerate(result[offset:]):
        need(message['role'] == ('user' if index % 2 == 0 else 'assistant'),
             'User and assistant messages must alternate.', 'messages')
    return dict(messages=result, stream=value.get('stream', False), budget=budget,
                include_usage=bool(stream_options and stream_options.get('include_usage')))


class AbortableTransport(C.Transport):
    """Interrupt a blocked native read when the downstream client disappears."""
    max_body = MAX_BODY
    def __init__(self, url, timeout):
        super().__init__(url, timeout)
        self.cancelled = threading.Event()
        self._lock = threading.Lock()
        self._socket = None

    def abort(self):
        self.cancelled.set()
        with self._lock:
            sock = self._socket
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def _connect(self, method, path, payload=None):
        C.require(not self.cancelled.is_set(), 'downstream cancelled')
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode('utf-8')
        C.require(body is None or len(body) <= self.max_body, 'native request exceeds byte bound')
        conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        try:
            conn.connect()
            with self._lock:
                self._socket = conn.sock
            if self.cancelled.is_set():
                self.abort()
                raise C.ProtocolError('downstream cancelled')
            conn.request(method, path, body=body, headers={'Content-Type': 'application/json', 'Connection': 'close'})
            response = conn.getresponse()
            if response.status != 200:
                detail = response.read(8192).decode('utf-8', errors='replace')
                raise C.ProtocolError(f'HTTP {response.status}: {detail}')
            return conn, response
        except BaseException:
            conn.close()
            raise


def native_turn(transport, ledger, plan, emit, ceiling):
    verifier = C.StreamVerifier(plan['payload']['request_id'], ledger.template)
    started, first, last, first_count = time.perf_counter(), None, None, 0
    frames = transport.frames(plan['path'], plan['payload'])
    try:
        for frame, elapsed in frames:
            C.require(not transport.cancelled.is_set(), 'downstream cancelled')
            verifier.push(frame)
            if frame['event'] == 'text':
                C.require(len(verifier.tokens) <= ceiling, 'native output exceeded advertised ceiling')
                if first is None:
                    first = elapsed
                if elapsed == first:
                    first_count += len(frame['token_ids'])
                last = elapsed
                emit(frame['text'])
        terminal, text = verifier.finish()  # includes complete native HTTP framing
        C.require(len(verifier.tokens) + int(terminal['stop_reason'] == 'eos') <= ceiling,
                  'native completion exceeded advertised ceiling')
        C.require(not transport.cancelled.is_set(), 'downstream cancelled')
        ledger.accept(plan, terminal, text)
        return terminal, text, dict(native_request_id=plan['payload']['request_id'],
            native_report_sha256=terminal['native_report_sha256'], native_ttft_s=first,
            frame_clock=C.FRAME_CLOCK,
            native_http_wall_s=time.perf_counter() - started, visible_tokens=len(verifier.tokens),
            post_first_tps=((len(verifier.tokens) - first_count) / (last - first)
                            if first is not None and last > first else None))
    except BaseException:
        ledger.poisoned = True
        raise
    finally:
        frames.close()


class Controller:
    def __init__(self, template, backend, model, timeout=14400, transport_factory=None, metrics=None,
                 health_factory=None):
        self.template, self.backend, self.model = template, backend, model
        self.factory = transport_factory or (lambda: AbortableTransport(backend, timeout))
        self.health_factory = health_factory or transport_factory or (
            lambda: AbortableTransport(backend, min(timeout, HEALTH_TIMEOUT)))
        self.metrics = metrics
        self.busy = threading.Lock()
        self.active_lock = threading.Lock()
        self.active = None
        self.current, self.faulted = None, False
        self.restore_used, self.restore_required = False, False
        self.health = self.health_factory().health()
        C.require(self.health.get('profile') == 'resident', 'API requires the native resident profile')
        self.limits = self.health.get('session_limits', {})
        self.request_budgets = self.limits.get('request_output_budget_supported', False)
        C.require(type(self.request_budgets) is bool, 'invalid native request-budget capability')
        C.require(self.limits.get('capacity') == 196608 and self.limits.get('draft_history_capacity') == 2048
                  and self.limits.get('initialized_history') is True,
                  'native full KV/draft-history contract mismatch')
        for key in ('initial_output_limit', 'continuation_output_limit'):
            C.require(type(self.limits.get(key)) is int and self.limits[key] > 0, 'native output budget missing')
        for stage in ('initial','continuation'):
            key=stage+'_max_output_limit'
            if key in self.limits:
                C.require(self.request_budgets and type(self.limits[key]) is int
                          and self.limits[stage+'_output_limit']<=self.limits[key]<=196608,
                          'invalid native extended output budget')

    def restore_saved(self, bundle, profile_sha):
        """Startup-only restore; never adopt a Python ledger without native proof."""
        if not self.busy.acquire(blocking=False):
            raise ApiError('One generation is already active.', 409, 'single_user_busy')
        transport = None
        try:
            C.require(self.current is None and not self.faulted and not self.restore_used,
                      'restore requires a fresh API controller')
            bundle = copy.deepcopy(P.validate_bundle(bundle, self.template))
            C.require(bundle['state']['profile'] == 'resident' and C.digest_string(profile_sha)
                      and bundle['checkpoint']['profile_sha256'] == profile_sha,
                      'resident saved checkpoint/profile required')
            # Check that this completed transcript can be submitted via this API,
            # including its message-count and control-token restrictions.
            messages = [dict(role=m['role'], content=m['content']) for m in bundle['state']['messages']]
            validate_request(dict(model=self.model, messages=messages + [dict(role='user', content='Resume')]),
                             self.model)
            C.require(self.health_factory().health().get('owner_resident') is False,
                      'restore requires a fresh idle native owner')
            transport = self.factory()
            with self.active_lock:
                self.active = transport
            self.restore_used, self.restore_required = True, True
            started = time.perf_counter()
            try:
                ledger, frame = P.restore(transport, bundle, self.template, profile_sha)
                C.require(not transport.cancelled.is_set(), 'API restore cancelled')
            except BaseException:
                self.current, self.faulted = None, True
                transport.abort()
                raise
            self.current = ledger
            return dict(event='api_session_restored', session_id=ledger.session_id,
                checkpoint_id=frame['checkpoint_id'], profile_sha256=profile_sha,
                consumed_tokens=len(ledger.history), prefix_recompute_tokens=frame['prefix_recompute_tokens'],
                restore_http_wall_s=time.perf_counter() - started,
                first_request_requires_exact_transcript=True,
                product_ready=False, qualified=False, semantic_qualification=False, visible_tps=False)
        finally:
            with self.active_lock:
                self.active = None
            self.busy.release()

    def abort(self):
        with self.active_lock:
            active = self.active
        if active is not None:
            active.abort()

    def capabilities(self):
        return dict(model=self.model, text_only=True, greedy=True, n=1, stream=True,
            full_kv_capacity=196608, draft_history=2048, native_limits=self.limits,
            exact_history_reuse=True, tool_calls=True, vision_forward=False,
            max_messages=C.MAX_MESSAGES, user_steering_after_tool_results=True,
            tool_call_types=['function'], tool_choice=['auto','none','required','named_function'],
            parallel_tool_calls=True, tool_argument_schema_validation=True,
            strict_constrained_generation=False, tool_execution='client_only',
            tool_delta_publication='validated_complete_calls_after_native_eof',
            tool_session_checkpoint_restore=False,
            completion_authority='native_generation_only_not_tool_execution_or_code_verification',
            startup_checkpoint_restore=True, first_request_requires_exact_transcript=self.restore_required,
            arbitrary_output_budget=self.request_budgets,
            request_output_budget_supported=self.request_budgets,
            openai_api_scope='chat_completions_text_function_tools_v2',
            request_half_close='treated_as_cancel', pipelining=False,
            product_ready=False, qualified=False, visible_tps=False)

    def output_budget(self,request,reused):
        stage='continuation' if reused else 'initial'
        cap=self.limits[stage+'_output_limit']
        need(self.request_budgets or request['budget'] is None or request['budget']>=cap,
             f'This backend uses a fixed {cap}-token ceiling for this request.','max_completion_tokens')
        if self.request_budgets and request['budget'] is not None:
            maximum=self.limits.get(stage+'_max_output_limit')
            if maximum is None:
                # Old frozen owners expose only their smaller fixed maximum.
                # Do not invent new capability for those binaries.
                cap=min(cap,request['budget'])
            else:
                need(request['budget']<=maximum,
                     f'This backend supports at most {maximum} output tokens.','max_completion_tokens')
                cap=request['budget']
        return cap

    def prepare(self, request):
        messages = request['messages']
        if request.get('tool_policy') is not None:
            policy=request['tool_policy']
            reused=(isinstance(self.current,T.Ledger) and not self.current.poisoned
                and self.current.tool_policy==policy and len(messages)>len(self.current.messages)
                and messages[:len(self.current.messages)]==self.current.messages
                and self.current.terminal['stop_reason']=='eos'
                and self.current.terminal['session']['pending_token']==self.template.end)
            if self.restore_required:
                raise ApiError('Tool sessions cannot adopt a text-only saved checkpoint.',409,'restored_history_mismatch')
            ledger=T.Ledger(self.template,policy)
            if reused:
                for key in ('session_id','messages','prefix','last_text','history','terminal'):
                    setattr(ledger,key,copy.deepcopy(getattr(self.current,key)))
            cap=self.output_budget(request,reused)
            try:
                plan=ledger.plan_messages(messages,output_reserve=max(134,cap+6))
            except ValueError as error:
                raise ApiError(str(error),code='context_or_template_limit',param='messages') from error
            if self.request_budgets and request['budget'] is not None:plan['payload']['max_new_tokens']=cap
            return ledger,plan,reused,cap
        reused = (self.current is not None and not self.current.poisoned
                  and not isinstance(self.current,T.Ledger)
                  and messages[:-1] == self.current.messages)
        if self.restore_required and not reused:
            raise ApiError('The first restored request must contain the exact saved transcript plus one user message.',
                           409, 'restored_history_mismatch', 'messages')
        if reused:
            # Preserve tokenizer/template objects, copy only request-owned state.
            ledger = C.Ledger(self.template, 'resident')
            for key in ('session_id', 'messages', 'prefix', 'last_text', 'history', 'terminal'):
                setattr(ledger, key, copy.deepcopy(getattr(self.current, key)))
        else:
            ledger = C.Ledger(self.template, 'resident')
            ledger.messages = copy.deepcopy(messages[:-1])
        cap = self.output_budget(request,reused)
        try:
            plan = ledger.plan(messages[-1]['content'],output_reserve=max(134,cap+6))
        except ValueError as error:
            raise ApiError(str(error), code='context_or_template_limit', param='messages') from error
        if self.request_budgets and request['budget'] is not None:
            plan['payload']['max_new_tokens'] = cap
        return ledger, plan, reused, cap

    def run_native_turn(self, transport, ledger, plan, emit, ceiling):
        """Backend-specific verification hook; the legacy contract stays default."""
        return native_turn(transport, ledger, plan, emit, ceiling)

    def validate_request(self, payload):
        return validate_request(payload, self.model)

    def output_adapter(self, request, content_emit, emit):
        return None

    def recover_faulted(self):
        if self.health_factory().health().get('owner_resident') is not False:
            raise ApiError('Native state is uncertain; wait for owner retirement or explicitly restart the backend.',
                           503, 'native_state_uncertain')
        self.current, self.faulted = None, False

    def cached_prompt_tokens(self, meta, plan, terminal):
        return meta['cached_tokens']

    def complete(self, payload, begin, emit, deliver, watch):
        validation_started = time.perf_counter()
        request = self.validate_request(payload)
        validation_s = time.perf_counter() - validation_started
        if not self.busy.acquire(blocking=False):
            raise ApiError('One generation is already active.', 409, 'single_user_busy')
        started, native_done, submitted = time.perf_counter(), False, False
        ledger, transport, stop = None, None, None
        restore_required = self.restore_required
        event = dict(event='chat_completion', request_id='chatcmpl-' + uuid.uuid4().hex,
                     model=self.model, product_ready=False, qualified=False, visible_tps=False)
        try:
            if self.restore_required and (self.faulted or self.current is None):
                raise ApiError('Restored state is uncertain; restart with a fresh native owner and the saved bundle.',
                               503, 'restored_state_uncertain')
            if self.faulted:
                recovery = self.recover_faulted()
                if recovery is not None:
                    event['native_recovery'] = recovery
            ledger, plan, reused, cap = self.prepare(request)
            if not reused and self.current is not None:
                try:
                    self.factory().close_session(self.current)
                except BaseException:
                    self.current, self.faulted = None, True
                    raise
                self.current = None
            transport = self.factory()
            with self.active_lock:
                self.active = transport
            stop = watch(transport.abort)
            meta = dict(id=event['request_id'], created=int(time.time()), model=self.model,
                        cached_tokens=len(ledger.history) if reused else 0, output_ceiling=cap,
                        session_mode='exact_history_append' if reused else 'canonical_full_history_open')
            if request.get('reasoning_policy') is not None:
                meta['reasoning_policy'] = request['reasoning_policy']
            begin(meta, request)
            submitted = True
            tools=T.Output(request['tool_policy'],emit) if request.get('tool_policy') is not None else None
            content_emit = tools.push if tools else emit
            adapter = self.output_adapter(request, content_emit, emit)
            terminal, text, timings = self.run_native_turn(transport, ledger, plan, adapter.push if adapter else content_emit, cap)
            native_done = True
            native_done_at = time.perf_counter()
            if adapter:
                text = adapter.finish(terminal['stop_reason'])
                request['response_reasoning_content'] = adapter.reasoning
                request['thinking_complete'] = adapter.closed
                reasoning_counts = adapter.token_counts(self.template.tokenizer, terminal['visible_token_ids'])
                ledger.messages[-1]['content'] = text
                timings.update(adapter.timings())
                timings['reasoning_policy'] = request['reasoning_policy']
            if tools:
                content,calls=tools.finish(terminal['stop_reason'])
                if calls:
                    ledger.messages[-1]=dict(role='assistant',content=content,reasoning_content='',tool_calls=calls)
                    request['response_tool_calls']=T.public_calls(calls)
                text=content or (None if calls else '')
            # A normal client may close immediately after reading the final
            # response. Stop the native-abort watcher before publishing that
            # boundary; write errors still retire the completed native session.
            watch_stop_started = time.perf_counter()
            stop()
            timings['cancel_watcher_stop_s'] = time.perf_counter() - watch_stop_started
            stop = None
            count = len(terminal['visible_token_ids']) + int(terminal['stop_reason'] == 'eos')
            usage = dict(prompt_tokens=len(plan['input_ids']), completion_tokens=count,
                         total_tokens=len(plan['input_ids']) + count,
                         prompt_tokens_details={'cached_tokens': self.cached_prompt_tokens(meta, plan, terminal)})
            if adapter:
                usage['completion_tokens_details'] = dict(reasoning_counts,
                    eos_tokens=int(terminal['stop_reason'] == 'eos'))
            finish = 'tool_calls' if request.get('response_tool_calls') else ('stop' if terminal['stop_reason'] == 'eos' else 'length')
            if tools:tools.publish()
            C.require(not transport.cancelled.is_set(), 'client disconnected before response publication')
            # Final bytes can reach the client before deliver() returns. Adopt
            # the native ledger under the busy lock, rolling back on failure.
            self.current = ledger
            self.restore_required = False
            deliver_started = time.perf_counter()
            deliver(meta, request, text, finish, usage)
            timings['final_response_write_s'] = time.perf_counter() - deliver_started
            timings['post_native_to_final_write_s'] = time.perf_counter() - native_done_at
            timings['request_validation_s'] = validation_s
            timings['http_finish_policy'] = getattr(self, 'http_finish_policy', 'legacy')
            C.require(not transport.cancelled.is_set(), 'client disconnected before response publication')
            event.update(status='completed', **timings, usage=usage, session_mode=meta['session_mode'])
            return event
        except BaseException as error:
            event.update(status='failed', error_type=type(error).__name__)
            if isinstance(error, C.GenerationFailure):
                event.update(error_code=error.code, error_counters=error.counters)
            if submitted:
                self.current, self.faulted = None, True
                self.restore_required = restore_required
                if native_done and ledger is not None and not ledger.poisoned:
                    try:
                        self.factory().close_session(ledger)
                        self.faulted = False
                    except (OSError, ValueError, http.client.HTTPException):
                        pass
            raise
        finally:
            if stop is not None:
                stop()
            with self.active_lock:
                self.active = None
            event['api_wall_s'] = time.perf_counter() - started
            if self.metrics:
                try:
                    self.metrics.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')
                    self.metrics.flush()
                except (OSError, ValueError):
                    print('API metrics write failed; this request has no complete metrics receipt.', file=sys.stderr)
            self.busy.release()

    def close(self):
        self.abort()
        if self.busy.acquire(blocking=False):
            try:
                if self.current is not None and not self.current.poisoned:
                    self.factory().close_session(self.current)
                    self.current = None
            finally:
                self.busy.release()


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, address, controller, api_key):
        C.require(address[0] == '127.0.0.1', 'only explicit IPv4 loopback binding is supported')
        validate_key(api_key)
        self.controller, self.api_key = controller, api_key
        self.started = int(time.time())
        self.slots = threading.BoundedSemaphore(16)
        super().__init__(address, Handler)

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'QwenTextBridge/1'
    sys_version = ''

    def setup(self):
        super().setup()
        self.connection.settimeout(15)
        self.response_started = False
        if getattr(self.server.controller, 'fast_finish', False):
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def log_message(self, *_):
        pass  # no automatic request/header/transcript logging

    def authorized(self):
        self.close_connection = True
        values = self.headers.get_all('Authorization', [])
        if len(values) != 1 or not values[0].isascii() or not hmac.compare_digest(values[0], 'Bearer ' + self.server.api_key):
            raise ApiError('A valid local Bearer key is required.', 401, 'invalid_api_key')
        if self.headers.get('Origin') is not None:
            raise ApiError('Browser cross-origin access is not enabled.', 403, 'origin_not_allowed')

    def json_response(self, status, value):
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(raw)))
        self.send_header('Connection', 'close')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.response_started = True
        self.wfile.write(raw)
        self.wfile.flush()

    def do_GET(self):
        try:
            self.authorized()
            model = dict(id=self.server.controller.model, object='model', created=self.server.started, owned_by='local')
            if self.path == '/v1/models':
                value = dict(object='list', data=[model])
            elif self.path == '/v1/models/' + model['id']:
                value = model
            elif self.path == '/cq/capabilities':
                value = self.server.controller.capabilities()
            elif self.path == '/health':
                value = dict(status=('busy' if self.server.controller.busy.locked() else
                                     'degraded' if self.server.controller.faulted else 'ready'),
                             scope='bridge_state_not_live_native_probe',
                             native_state_uncertain=self.server.controller.faulted, product_ready=False, qualified=False)
            else:
                raise ApiError('Unknown endpoint.', 404, 'not_found')
            self.json_response(200, value)
        except ApiError as error:
            self.json_response(error.status, error.response())
        except (OSError, http.client.HTTPException):
            pass

    def watch(self, abort):
        stopped = threading.Event()
        abort_guard = threading.Lock()
        fast = getattr(self.server.controller, 'fast_finish', False)

        def cancel_if_active():
            with abort_guard:
                if not stopped.is_set():
                    abort()

        def monitor():
            while not stopped.is_set():
                try:
                    # Let EOF/RST wake the watcher immediately. Sleeping before
                    # a zero-time select left a 0-100 ms cancel admission race.
                    ready, _, errors = select.select([self.connection], [], [self.connection], .1)
                    if errors or ready:
                        # Do not consume pipelined bytes. This API closes each
                        # connection after one request; EOF/RST aborts the turn.
                        if errors or self.connection.recv(1, socket.MSG_PEEK) == b'':
                            cancel_if_active()
                            return
                        # Unexpected pipelined data stays unread; avoid a busy loop.
                        stopped.wait(.1)
                except OSError:
                    cancel_if_active()
                    return

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()

        def stop():
            stopped.set()
            if fast:
                # Fence any in-flight abort without waiting for an already
                # disabled watcher thread to be scheduled merely to exit.
                with abort_guard:
                    pass
            else:
                thread.join(timeout=1)

        return stop

    def sse(self, value):
        raw = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, allow_nan=False)
        self.wfile.write(('data: ' + raw + '\n\n').encode('utf-8'))
        self.wfile.flush()

    def do_POST(self):
        try:
            self.authorized()
            if self.path != '/v1/chat/completions':
                raise ApiError('Only /v1/chat/completions is supported.', 404, 'not_found')
            need(not self.headers.get_all('Transfer-Encoding'), 'Chunked request bodies are not supported.')
            lengths = self.headers.get_all('Content-Length', [])
            need(len(lengths) == 1 and 1 <= len(lengths[0]) <= 8 and lengths[0].isascii()
                 and lengths[0].isdigit(), 'One bounded Content-Length is required.')
            size = int(lengths[0])
            need(0 < size <= getattr(self.server.controller, 'max_body', MAX_BODY), 'Request body exceeds its byte limit.')
            types = self.headers.get_all('Content-Type', [])
            need(len(types) == 1 and types[0].split(';')[0].strip().lower() == 'application/json',
                 'application/json is required.')
            need(not self.headers.get_all('Content-Encoding'), 'Encoded request bodies are not supported.')
            raw = self.rfile.read(size)
            need(len(raw) == size, 'Truncated request body.')
            try:
                payload = C.strict_json(raw)
            except (ValueError, UnicodeError, RecursionError) as error:
                raise ApiError('Malformed or non-finite JSON.') from error
            need(type(payload) is dict, 'A JSON object is required.')
            streaming = payload.get('stream') is True
            metadata, options = {}, {}
            terminal_events = []
            fast_finish = getattr(self.server.controller, 'fast_finish', False)

            def chunk(delta, finish=None, usage=None, empty=False):
                value = dict(id=metadata['id'], object='chat.completion.chunk', created=metadata['created'],
                    model=metadata['model'], choices=[] if empty else [dict(index=0, delta=delta, logprobs=None, finish_reason=finish)])
                if options.get('include_usage'):
                    value['usage'] = usage
                return value

            def begin(meta, request):
                metadata.update(meta)
                options.update(request)
                if streaming:
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
                    self.send_header('Cache-Control', 'no-cache, no-transform')
                    self.send_header('Connection', 'close')
                    self.send_header('X-Accel-Buffering', 'no')
                    self.send_header('X-AUGUR-Session-Mode', meta['session_mode'])
                    self.send_header('X-AUGUR-Native-Output-Ceiling', str(meta['output_ceiling']))
                    self.end_headers()
                    self.response_started = True
                    self.sse(chunk({'role': 'assistant', 'content': ''}))

            def emit(text):
                if streaming:
                    event = chunk(text if isinstance(text, dict) else {'content': text})
                    if fast_finish and isinstance(text, dict) and 'tool_calls' in text:
                        # Tool deltas are produced only after verified native
                        # EOF. Publish them atomically with the final SSE group.
                        terminal_events.append(event)
                    else:
                        self.sse(event)

            def deliver(meta, request, text, finish, usage):
                if streaming:
                    events = terminal_events + [chunk({}, finish)]
                    if request['include_usage']:
                        events.append(chunk({}, usage=usage, empty=True))
                    events.append('[DONE]')
                    if fast_finish:
                        raw = ''.join('data: ' + (event if isinstance(event, str) else
                            json.dumps(event, ensure_ascii=False, allow_nan=False)) + '\n\n' for event in events)
                        self.wfile.write(raw.encode('utf-8'))
                        self.wfile.flush()
                    else:
                        for event in events:
                            self.sse(event)
                else:
                    message={'role':'assistant','content':text}
                    if 'response_reasoning_content' in request:
                        message['reasoning_content'] = request['response_reasoning_content']
                    if request.get('response_tool_calls'):
                        message['tool_calls']=request['response_tool_calls']
                    self.json_response(200, dict(id=meta['id'], object='chat.completion', created=meta['created'],
                        model=meta['model'], choices=[dict(index=0, message=message,
                        logprobs=None, finish_reason=finish)], usage=usage))

            self.server.controller.complete(payload, begin, emit, deliver, self.watch)
        except (ApiError, ValueError, KeyError, TypeError, OSError, http.client.HTTPException) as error:
            api = (error if isinstance(error, ApiError) else
                   ApiError(str(error), error.status, error.code) if isinstance(error, C.GenerationFailure) else
                   ApiError('Native request failed; no completion was committed.', 502, 'native_generation_failed'))
            try:
                if self.response_started:
                    self.sse(api.response())  # never send [DONE] for an incomplete response
                else:
                    self.json_response(api.status, api.response())
            except (OSError, http.client.HTTPException):
                pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', default='http://127.0.0.1:8193')
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--model', default='qwen35-27b-resident')
    parser.add_argument('--port', type=int, default=8194)
    parser.add_argument('--timeout', type=float, default=14400)
    parser.add_argument('--metrics-file', type=Path, help='fresh private JSONL path; no prompt text or keys')
    parser.add_argument('--restore-file', type=Path, help='private completed session; restore before listening')
    parser.add_argument('--expected-state-sha256', help='required external SHA256 of --restore-file')
    parser.add_argument('--checkpoint-profile', type=Path, help='native checkpoint profile paired with the saved state')
    args = parser.parse_args(argv)
    key = os.environ.get('AUGUR_QWEN35_API_KEY', '')
    validate_key(key)  # fail before tokenizer IO, metrics creation or a backend connection
    C.require(isinstance(args.model, str) and 1 <= len(args.model) <= 128 and
              all(c.isascii() and (c.isalnum() or c in '-_.') for c in args.model), 'bounded plain model identifier required')
    C.require(math.isfinite(args.timeout) and args.timeout > 0 and 0 <= args.port <= 65535, 'invalid timeout or port')
    restore_options = (args.restore_file, args.expected_state_sha256, args.checkpoint_profile)
    C.require(all(value is None for value in restore_options) or all(value is not None for value in restore_options),
              '--restore-file, --expected-state-sha256 and --checkpoint-profile must be supplied together')
    if args.restore_file is not None:
        C.require(C.digest_string(args.expected_state_sha256), 'restore requires an external lowercase SHA256')
    template = C.ChatTemplate(args.model_dir)
    saved = restore_inputs(template, *restore_options) if args.restore_file is not None else None
    metrics = args.metrics_file.open('x', encoding='utf-8') if args.metrics_file else None
    controller = server = None
    try:
        controller = Controller(template, args.backend, args.model, args.timeout, metrics=metrics)
        if saved is not None:
            restored = controller.restore_saved(*saved)
            print(json.dumps(dict(restored, client_state_sha256=args.expected_state_sha256)), flush=True)
        server = Server(('127.0.0.1', args.port), controller, key)
        print(json.dumps(dict(event='text_api_listening', port=server.server_address[1], model=args.model,
                              scope='chat_completions_text_subset_v1', product_ready=False, qualified=False)), flush=True)
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        if controller is not None:
            controller.close()
        if server is not None:
            server.server_close()
        if metrics is not None:
            metrics.close()


if __name__ == '__main__':
    main()
