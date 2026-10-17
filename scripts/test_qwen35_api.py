"""CPU contract and actual loopback HTTP/SSE tests; no model or CUDA loads."""
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import http.client
import io
import json
import os
from pathlib import Path
import socket
import struct
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import qwen35_api as A
import qwen35_chat as C
import qwen35_persistence as P
from test_qwen35_chat import Template, fixture
from test_qwen35_persistence import live_ledger, restore_frame, save_frame, PROFILE

MODEL, KEY = 'test-qwen', 'local-test-key-not-a-real-secret'


def request(messages=None, **options):
    return dict(model=MODEL, messages=messages or [dict(role='user', content='Hi')], **options)


class Backend:
    def __init__(self):
        self.template, self.native = Template(), None
        self.mode = 'ok'
        self.loaded, self.loads, self.closes = False, 0, 0
        self.requests = []
        self.entered, self.aborted = threading.Event(), threading.Event()
        self.close_failure = False
        self.text = 'OK'
        self.request_budgets = False
        self.saved_bundle = None
        self.bad_restore = False

    def factory(self):
        backend = self

        class Transport:
            def __init__(self):
                self.cancelled = threading.Event()

            def health(self):
                return dict(profile='resident', owner_resident=backend.loaded,
                    session_limits=dict(capacity=196608, draft_history_capacity=2048, initialized_history=True,
                                        request_output_budget_supported=backend.request_budgets,
                                        initial_output_limit=32, continuation_output_limit=16))

            def abort(self):
                self.cancelled.set()
                backend.aborted.set()
                backend.native, backend.loaded = None, False

            def close_session(self, ledger):
                backend.closes += 1
                if backend.close_failure:
                    raise C.ProtocolError('close failure')
                backend.native = None
                ledger.poisoned = True

            def frames(self, path, payload):
                backend.requests.append((path, copy.deepcopy(payload)))
                if not backend.loaded:
                    backend.loaded = True
                    backend.loads += 1
                if path == '/v1/session/restore':
                    C.require(backend.saved_bundle is not None, 'unexpected restore')
                    frame = restore_frame(payload, backend.saved_bundle)
                    if backend.bad_restore:
                        frame['native_manifest_sha256'] = '0' * 64
                    else:
                        state = backend.saved_bundle['state']
                        ledger = C.Ledger(backend.template, 'resident')
                        for field in P.STATE_KEYS:
                            setattr(ledger, field, copy.deepcopy(state[field]))
                        ledger.session_id = payload['session_id']
                        ledger.terminal['session'] = copy.deepcopy(frame['session'])
                        backend.native = ledger
                    yield frame, .01
                    return
                if path.endswith('/open'):
                    ledger = C.Ledger(backend.template, 'resident')
                    ledger.session_id = payload['session_id']
                    ids = backend.template.encode(payload['prompt'])
                else:
                    ledger = backend.native
                    C.require(ledger is not None and ledger.session_id == payload['session_id'], 'unknown session')
                    C.require(payload['exact_history'] == ledger.history, 'foreign exact history')
                    ids = ledger.history + [ledger.terminal['session']['pending_token']] + backend.template.encode(payload['suffix'])
                plan = dict(payload=payload, input_ids=ids)
                text, stop = backend.text, 'max_new_tokens' if backend.mode == 'budget' else 'eos'
                budget = payload.get('max_new_tokens')
                if budget is not None and len(backend.template.encode(text)) >= budget:
                    text, stop = text[:budget], 'max_new_tokens'
                frames = fixture(ledger, plan, text=text, stop=stop)
                if text == '':
                    frames = [frames[0], dict(frames[-1], sequence=1)]
                backend.entered.set()
                if backend.mode == 'silent':
                    self.cancelled.wait(3)
                    if not self.cancelled.is_set():
                        raise RuntimeError('test cancellation deadline')
                    raise C.ProtocolError('cancelled native read')
                for i, frame in enumerate(frames):
                    if backend.mode == 'native_error' and frame['event'] == 'terminal':
                        frame = dict(frame, event='error')
                    if backend.mode == 'missing_terminal' and frame['event'] == 'terminal':
                        break
                    yield frame, .01 + i * .02
                if backend.mode == 'truncated_http':
                    raise C.ProtocolError('truncated final HTTP chunk')
                if backend.mode in ('ok', 'budget'):
                    visible = backend.template.encode(text)
                    ledger.history = ids + (visible[:-1] if stop == 'max_new_tokens' else visible)
                    ledger.terminal = frames[-1]
                    backend.native = ledger

        return Transport()


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.backend = Backend()
        self.controller = A.Controller(self.backend.template, 'http://127.0.0.1', MODEL,
                                       transport_factory=self.backend.factory)
        self.addCleanup(self.controller.close)

    def complete(self, value=None, deliver=None):
        result = {}
        def delivered(meta, options, text, finish, usage):
            result.update(meta=meta, options=options, text=text, finish=finish, usage=usage)
        result['event'] = self.controller.complete(value or request(), lambda *_: None, lambda _: None,
            deliver or delivered, lambda abort: lambda: None)
        return result

    def saved(self):
        ledger = live_ledger()
        self.backend.template = ledger.template
        self.controller.template = ledger.template
        bundle = P.make_bundle(P.snapshot(ledger), ledger.template,
                               save_frame(dict(checkpoint_id='7' * 64, request_id='saved'), ledger))
        self.backend.saved_bundle = bundle
        return bundle

    def test_startup_restore_exact_transcript_then_normal_history(self):
        bundle = self.saved()
        receipt = self.controller.restore_saved(bundle, PROFILE)
        self.assertEqual(receipt['consumed_tokens'], len(bundle['state']['history']))
        self.assertEqual(receipt['prefix_recompute_tokens'], 0)
        self.assertTrue(self.controller.restore_required)
        self.assertEqual([path for path, _ in self.backend.requests], ['/v1/session/restore'])
        saved = self.controller.current.history.copy()
        messages = [dict(role=m['role'], content=m['content']) for m in bundle['state']['messages']]
        for changed in (messages[:-1] + [dict(role='assistant', content='wrong'), dict(role='user', content='Next')],
                        [dict(role='user', content='Unrelated')]):
            with self.assertRaisesRegex(A.ApiError, 'exact saved transcript'):
                self.complete(request(changed))
        self.assertEqual(len(self.backend.requests), 1)
        self.assertIsNotNone(self.backend.native)
        self.assertTrue(self.controller.restore_required)
        first = self.complete(request(messages + [dict(role='user', content='Next')]))
        self.assertEqual(first['meta']['session_mode'], 'exact_history_append')
        self.assertEqual(self.backend.requests[-1][1]['exact_history'], saved)
        self.assertEqual(first['usage']['prompt_tokens_details']['cached_tokens'], len(saved))
        self.assertEqual(self.backend.loads, 1)
        self.assertFalse(self.controller.restore_required)
        self.assertEqual(self.complete(request([dict(role='user', content='Different')]))['meta']['session_mode'],
                         'canonical_full_history_open')

    def test_restore_rejects_live_owner_or_bad_profile_before_native_restore(self):
        bundle = self.saved()
        self.backend.loaded = True
        with self.assertRaisesRegex(ValueError, 'fresh idle native owner'):
            self.controller.restore_saved(bundle, PROFILE)
        self.assertEqual(self.backend.requests, [])
        self.backend.loaded = False
        with self.assertRaisesRegex(ValueError, 'resident saved checkpoint/profile'):
            self.controller.restore_saved(bundle, '0' * 64)
        self.assertEqual(self.backend.requests, [])
        self.assertFalse(self.controller.restore_used)

    def test_restore_failure_never_adopts_ledger_or_falls_back_to_open(self):
        bundle = self.saved()
        self.backend.bad_restore = True
        with self.assertRaisesRegex(ValueError, 'manifest binding mismatch'):
            self.controller.restore_saved(bundle, PROFILE)
        self.assertIsNone(self.controller.current)
        self.assertTrue(self.controller.faulted)
        self.assertTrue(self.controller.restore_required)
        self.assertTrue(self.backend.aborted.is_set())
        with self.assertRaisesRegex(A.ApiError, 'Restored state is uncertain'):
            self.complete()
        self.assertEqual(len(self.backend.requests), 1)

    def test_restore_inputs_reject_missing_external_hash_and_profile_drift_without_backend_io(self):
        bundle = self.saved()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state, profile = root / 'state.json', root / 'profile.json'
            raw = P.canonical(bundle)
            state.write_bytes(raw)
            profile.write_bytes(P.canonical(dict(schema='cq-qwen35-http-checkpoint-profile-v1',
                bindings=dict(tokenizer_sha256='a' * 64, chat_template_sha256='b' * 64,
                              generation_config_sha256='c' * 64))))
            with self.assertRaisesRegex(ValueError, 'external lowercase SHA256'):
                A.restore_inputs(self.backend.template, state, None, profile)
            with self.assertRaisesRegex(ValueError, 'external client file hash mismatch'):
                A.restore_inputs(self.backend.template, state, '0' * 64, profile)
            with self.assertRaisesRegex(ValueError, 'saved checkpoint profile mismatch'):
                A.restore_inputs(self.backend.template, state, C.sha256(raw), profile)
            bundle['checkpoint']['profile_sha256'] = C.sha256(profile.read_bytes())
            bundle['content_sha256'] = C.sha256(P.canonical({k: v for k, v in bundle.items() if k != 'content_sha256'}))
            raw = P.canonical(bundle)
            state.write_bytes(raw)
            loaded, pin = A.restore_inputs(self.backend.template, state, C.sha256(raw), profile)
            self.assertEqual(loaded, bundle)
            self.assertEqual(pin, bundle['checkpoint']['profile_sha256'])
        self.assertEqual(self.backend.requests, [])

    def test_restore_is_not_reentrant_and_existing_conversation_is_preserved(self):
        bundle = self.saved()
        self.controller.busy.acquire()
        try:
            with self.assertRaisesRegex(A.ApiError, 'already active'):
                self.controller.restore_saved(bundle, PROFILE)
        finally:
            self.controller.busy.release()
        self.controller.restore_saved(bundle, PROFILE)
        current = self.controller.current
        with self.assertRaisesRegex(ValueError, 'fresh API controller'):
            self.controller.restore_saved(bundle, PROFILE)
        self.assertIs(self.controller.current, current)
        self.assertEqual(len(self.backend.requests), 1)

    def test_failed_first_append_never_reopens_saved_history_even_after_retirement(self):
        bundle = self.saved()
        self.controller.restore_saved(bundle, PROFILE)
        self.backend.mode = 'missing_terminal'
        messages = [dict(role=m['role'], content=m['content']) for m in bundle['state']['messages']]
        with self.assertRaises(ValueError):
            self.complete(request(messages + [dict(role='user', content='Next')]))
        self.backend.loaded, self.backend.mode = False, 'ok'
        with self.assertRaisesRegex(A.ApiError, 'Restored state is uncertain'):
            self.complete()
        self.assertEqual([p for p, _ in self.backend.requests], ['/v1/session/restore', '/v1/session/append'])

    def test_exact_history_reuses_native_tokens_without_old_answer_tokenization(self):
        first = self.complete()
        saved = self.controller.current.history.copy()
        full = [dict(role='user', content='Hi'), dict(role='assistant', content='OK'), dict(role='user', content='Next')]
        original = self.backend.template.encode
        encoded = []
        def encode(text):
            encoded.append(text)
            return original(text)
        with patch.object(self.backend.template, 'encode', side_effect=encode):
            second = self.complete(request(full))
        self.assertEqual(second['meta']['session_mode'], 'exact_history_append')
        self.assertEqual(self.backend.requests[-1][0], '/v1/session/append')
        self.assertEqual(self.backend.requests[-1][1]['exact_history'], saved)
        self.assertEqual(second['usage']['prompt_tokens_details']['cached_tokens'], len(saved))
        self.assertFalse(any('Hi' in text for text in encoded))
        self.assertEqual((self.backend.loads, self.backend.closes), (1, 0))
        self.assertNotEqual(first['meta']['id'], second['meta']['id'])

    def test_unknown_history_closes_old_session_but_keeps_model_loaded(self):
        self.complete()
        result = self.complete(request([dict(role='user', content='Different')]))
        self.assertEqual(result['meta']['session_mode'], 'canonical_full_history_open')
        self.assertEqual((self.backend.loads, self.backend.closes), (1, 1))

    def test_unknown_complete_transcript_is_a_real_canonical_prompt(self):
        messages = [dict(role='system', content='Be brief.'), dict(role='user', content='A'),
                    dict(role='assistant', content='B'), dict(role='user', content='C')]
        self.complete(request(messages))
        self.assertIn('assistant\nB', self.backend.requests[0][1]['prompt'])
        self.assertEqual([m['content'] for m in self.controller.current.messages], ['Be brief.', 'A', 'B', 'C', 'OK'])

    def test_smaller_budget_rejects_before_replacing_current_history(self):
        self.complete()
        old = self.controller.current
        with self.assertRaisesRegex(A.ApiError, 'fixed 32-token'):
            self.complete(request([dict(role='user', content='Different')], max_tokens=8))
        self.assertIs(self.controller.current, old)
        self.assertEqual(self.backend.closes, 0)
        self.assertEqual(len(self.backend.requests), 1)

    def test_invalid_long_context_preserves_existing_session(self):
        self.complete()
        old = self.controller.current
        with self.assertRaises(A.ApiError):
            self.complete(request([dict(role='user', content='x' * 200000)]))
        self.assertIs(self.controller.current, old)
        self.assertEqual(self.backend.closes, 0)

    def test_native_missing_terminal_or_http_truncation_never_commits(self):
        for mode in ('missing_terminal', 'truncated_http'):
            with self.subTest(mode=mode):
                backend = Backend()
                backend.mode = mode
                controller = A.Controller(backend.template, 'http://127.0.0.1', MODEL, transport_factory=backend.factory)
                delivered = []
                with self.assertRaises(ValueError):
                    controller.complete(request(), lambda *_: None, lambda _: None,
                                        lambda *args: delivered.append(args), lambda _: lambda: None)
                self.assertEqual(delivered, [])
                self.assertIsNone(controller.current)
                self.assertTrue(controller.faulted)
                self.assertFalse(controller.busy.locked())

    def test_uncertain_live_owner_blocks_then_retired_owner_allows_fresh_request(self):
        self.backend.mode = 'missing_terminal'
        with self.assertRaises(ValueError):
            self.complete()
        with self.assertRaisesRegex(A.ApiError, 'Native state is uncertain'):
            self.complete()
        self.backend.loaded, self.backend.mode = False, 'ok'
        self.complete()
        self.assertFalse(self.controller.faulted)

    def test_failed_session_close_does_not_loop_on_old_poisoned_ledger(self):
        self.complete()
        self.backend.close_failure = True
        with self.assertRaisesRegex(ValueError, 'close failure'):
            self.complete(request([dict(role='user', content='Different')]))
        self.assertIsNone(self.controller.current)
        self.assertTrue(self.controller.faulted)
        with self.assertRaises(A.ApiError):
            self.complete()
        self.assertEqual(self.backend.closes, 1)

    def test_failed_delivery_closes_confirmed_native_session_without_replaying(self):
        def fail(*_):
            raise BrokenPipeError('peer lost')
        with self.assertRaises(BrokenPipeError):
            self.complete(deliver=fail)
        self.assertIsNone(self.controller.current)
        self.assertFalse(self.controller.faulted)
        self.assertEqual((len(self.backend.requests), self.backend.closes), (1, 1))

    def test_restored_state_is_committed_before_final_delivery(self):
        bundle = self.saved()
        self.controller.restore_saved(bundle, PROFILE)
        messages = [dict(role=m['role'], content=m['content']) for m in bundle['state']['messages']]
        messages.append(dict(role='user', content='Next'))

        def deliver(*_):
            self.assertFalse(self.controller.restore_required)
            self.assertIsNotNone(self.controller.current)
            self.assertEqual(self.controller.current.messages[-1]['content'], 'OK')
            self.assertTrue(self.controller.busy.locked())

        self.complete(request(messages), deliver=deliver)

    def test_restored_failed_delivery_preserves_uncertainty_gate(self):
        bundle = self.saved()
        self.controller.restore_saved(bundle, PROFILE)
        messages = [dict(role=m['role'], content=m['content']) for m in bundle['state']['messages']]
        messages.append(dict(role='user', content='Next'))

        def fail(*_):
            raise BrokenPipeError('peer lost')

        with self.assertRaises(BrokenPipeError):
            self.complete(request(messages), deliver=fail)
        self.assertTrue(self.controller.restore_required)
        self.assertIsNone(self.controller.current)
        self.assertFalse(self.controller.faulted)
        with self.assertRaisesRegex(A.ApiError, 'Restored state is uncertain'):
            self.complete()
        self.assertEqual((len(self.backend.requests), self.backend.closes), (2, 1))

    def test_cancelled_completed_native_turn_never_publishes_success(self):
        delivered = []

        def watch(abort):
            return abort

        with self.assertRaisesRegex(ValueError, 'client disconnected before response publication'):
            self.controller.complete(request(), lambda *_: None, lambda _: None,
                lambda *args: delivered.append(args), watch)
        self.assertEqual(delivered, [])
        self.assertIsNone(self.controller.current)
        self.assertFalse(self.controller.busy.locked())

    def test_usage_counts_native_pending_eos_and_cached_prompt_separately(self):
        result = self.complete()
        usage = result['usage']
        self.assertEqual(usage['completion_tokens'], 3)  # two visible tokens + EOS
        self.assertEqual(usage['total_tokens'], usage['prompt_tokens'] + 3)
        self.assertEqual(usage['prompt_tokens_details']['cached_tokens'], 0)

    def test_short_api_and_native_wall_use_precise_clock(self):
        with patch.object(A.time, 'monotonic', return_value=20.0), \
             patch.object(A.time, 'perf_counter', side_effect=(100.0 + i*.001 for i in range(32))):
            result = self.complete()
        self.assertAlmostEqual(result['event']['native_http_wall_s'], .001)
        self.assertAlmostEqual(result['event']['api_wall_s'], .009)
        self.assertAlmostEqual(result['event']['request_validation_s'], .001)
        self.assertAlmostEqual(result['event']['cancel_watcher_stop_s'], .001)
        self.assertAlmostEqual(result['event']['final_response_write_s'], .001)

    def test_native_post_first_excludes_all_tokens_in_initial_delivery(self):
        for times, expected in (([1., 1., 2.], 1.), ([1., 1., 1.], None)):
            with self.subTest(times=times):
                ledger = C.Ledger(Template())
                plan = ledger.plan('Hi')
                values = fixture(ledger, plan, text='ABC')

                class Transport:
                    cancelled = threading.Event()

                    def frames(self, path, payload):
                        yield values[0], .5
                        for index, (char, elapsed) in enumerate(zip('ABC', times), 1):
                            yield dict(values[1], sequence=index, text=char, token_ids=[ord(char)]), elapsed
                        yield dict(values[-1], sequence=4), 3.

                _, text, metrics = A.native_turn(Transport(), ledger, plan, lambda _: None, 32)
                self.assertEqual(text, 'ABC')
                self.assertEqual(metrics['post_first_tps'], expected)
                self.assertEqual(metrics['frame_clock'], C.FRAME_CLOCK)

    def test_metrics_failure_does_not_lock_out_future_requests(self):
        class FailedMetrics:
            def write(self, text):
                raise OSError('full')
        self.controller.metrics = FailedMetrics()
        with patch('sys.stderr', new=io.StringIO()):
            self.complete()
        self.assertFalse(self.controller.busy.locked())
        self.assertIsNotNone(self.controller.current)

    def test_concurrent_request_is_rejected_without_native_submission(self):
        self.controller.busy.acquire()
        try:
            with self.assertRaises(A.ApiError) as result:
                self.complete()
            self.assertEqual(result.exception.status, 409)
            self.assertFalse(self.backend.requests)
        finally:
            self.controller.busy.release()

    def test_native_output_over_ceiling_has_no_successful_terminal(self):
        self.backend.text = 'X' * 33
        with self.assertRaisesRegex(ValueError, 'advertised ceiling'):
            self.complete()
        self.assertIsNone(self.controller.current)

    def test_native_budget_capability_forwards_one_token_and_preserves_continuation(self):
        self.backend.request_budgets = True
        self.controller = A.Controller(self.backend.template, 'http://127.0.0.1', MODEL,
                                       transport_factory=self.backend.factory)
        self.addCleanup(self.controller.close)
        first = self.complete(request(max_completion_tokens=1))
        self.assertEqual(first['text'], 'O')
        self.assertEqual(first['finish'], 'length')
        self.assertEqual(first['usage']['completion_tokens'], 1)
        self.assertEqual(self.backend.requests[-1][1]['max_new_tokens'], 1)
        history = self.controller.current.history.copy()
        second = self.complete(request([dict(role='user', content='Hi'), dict(role='assistant', content='O'),
                                        dict(role='user', content='Next')], max_tokens=1))
        self.assertEqual(second['meta']['session_mode'], 'exact_history_append')
        self.assertEqual(self.backend.requests[-1][1]['exact_history'], history)
        self.assertEqual(self.backend.requests[-1][1]['max_new_tokens'], 1)
        self.assertEqual(self.backend.loads, 1)

    def test_extended_continuation_is_explicit_preserves_ids_and_rejects_oversize(self):
        self.backend.request_budgets=True
        def factory():
            transport=self.backend.factory();health=transport.health
            def extended():
                value=health();value['session_limits'].update(initial_max_output_limit=32,
                    continuation_max_output_limit=1024)
                return value
            transport.health=extended;return transport
        self.controller=A.Controller(self.backend.template,'http://127.0.0.1',MODEL,transport_factory=factory)
        self.addCleanup(self.controller.close)
        self.complete()
        history=self.controller.current.history.copy();session=self.controller.current.session_id
        messages=[dict(role='user',content='Hi'),dict(role='assistant',content='OK'),dict(role='user',content='Next')]
        self.backend.text='X'*180
        result=self.complete(request(messages,max_completion_tokens=512))
        self.assertEqual(result['finish'],'stop')
        self.assertEqual(result['text'],'X'*180)
        self.assertEqual(result['meta']['output_ceiling'],512)
        self.assertEqual(self.backend.requests[-1][1]['max_new_tokens'],512)
        self.assertEqual(self.backend.requests[-1][1]['exact_history'],history)
        self.assertEqual(self.controller.current.session_id,session)
        self.assertEqual(self.backend.loads,1)
        current=self.controller.current;count=len(self.backend.requests)
        follow=messages+[dict(role='assistant',content='X'*180),dict(role='user',content='More')]
        with self.assertRaises(A.ApiError):self.complete(request(follow,max_completion_tokens=1025))
        self.assertIs(self.controller.current,current)
        self.assertEqual(len(self.backend.requests),count)

    def test_extended_budget_reserve_rejects_before_native_or_history_mutation(self):
        self.backend.request_budgets=True
        def factory():
            transport=self.backend.factory();health=transport.health
            def extended():
                value=health();value['session_limits']['continuation_max_output_limit']=1024
                return value
            transport.health=extended;return transport
        self.controller=A.Controller(self.backend.template,'http://127.0.0.1',MODEL,transport_factory=factory)
        self.addCleanup(self.controller.close);self.complete()
        current=self.controller.current;current.history=[ord('A')]*196300
        before=current.history.copy();count=len(self.backend.requests)
        value=request([dict(role='user',content='Hi'),dict(role='assistant',content='OK'),
                       dict(role='user',content='Next')],max_completion_tokens=512)
        with self.assertRaises(A.ApiError):self.complete(value)
        self.assertIs(self.controller.current,current)
        self.assertEqual(current.history,before)
        self.assertEqual(len(self.backend.requests),count)

    def test_empty_eos_answer_can_be_continued_without_rebuilding_history(self):
        self.backend.text = ''
        first = self.complete()
        self.assertEqual(first['text'], '')
        self.assertEqual(first['finish'], 'stop')
        self.assertEqual(first['usage']['completion_tokens'], 1)
        self.backend.text = 'OK'
        second = self.complete(request([dict(role='user', content='Hi'), dict(role='assistant', content=''),
                                        dict(role='user', content='Next')]))
        self.assertEqual(second['meta']['session_mode'], 'exact_history_append')

    def test_budget_stop_keeps_pending_token_exact_across_next_request(self):
        self.backend.mode = 'budget'
        result = self.complete()
        self.assertEqual(result['finish'], 'length')
        self.assertEqual(result['usage']['completion_tokens'], 2)
        saved = self.controller.current.history.copy()
        self.assertEqual(saved[-1], ord('O'))
        self.complete(request([dict(role='user', content='Hi'), dict(role='assistant', content='OK'),
                               dict(role='user', content='Next')]))
        self.assertEqual(self.backend.requests[-1][1]['exact_history'], saved)
        self.assertTrue(self.backend.requests[-1][1]['suffix'].startswith('<|im_end|>\n'))

    def test_native_error_requires_retirement_before_fresh_recovery(self):
        self.backend.mode = 'native_error'
        with self.assertRaises(ValueError):
            self.complete()
        self.assertTrue(self.controller.faulted)
        self.assertIsNone(self.controller.current)
        with self.assertRaises(A.ApiError):
            self.complete()
        self.backend.loaded, self.backend.mode = False, 'ok'
        self.complete()
        self.assertFalse(self.controller.faulted)
        self.assertEqual(self.backend.requests[-1][0], '/v1/session/open')

    def test_health_timeout_is_separate_from_long_generation_timeout(self):
        with patch.object(A, 'AbortableTransport', return_value=self.backend.factory()) as transport:
            controller = A.Controller(self.backend.template, 'http://127.0.0.1', MODEL, timeout=14400)
            transport.assert_called_once_with('http://127.0.0.1', A.HEALTH_TIMEOUT)
            controller.faulted = True
            controller.complete(request(), lambda *_: None, lambda _: None, lambda *_: None,
                                lambda _: lambda: None)
            self.assertEqual([call.args[1] for call in transport.call_args_list],
                             [A.HEALTH_TIMEOUT, A.HEALTH_TIMEOUT, 14400])
            controller.close()


class AdmissionTests(unittest.TestCase):
    def test_startup_restore_flags_must_be_complete_before_io(self):
        for options in (['--restore-file', 'unused'], ['--expected-state-sha256', '1' * 64],
                        ['--checkpoint-profile', 'unused'],
                        ['--restore-file', 'unused', '--expected-state-sha256', 'BAD', '--checkpoint-profile', 'unused']):
            with self.subTest(options=options), patch.dict(os.environ, {'AUGUR_QWEN35_API_KEY': KEY}), \
                    patch.object(C, 'ChatTemplate') as template, patch.object(A, 'Controller') as controller:
                with self.assertRaises(ValueError):
                    A.main(['--model-dir', 'unused', *options])
                template.assert_not_called()
                controller.assert_not_called()

    def test_startup_restore_failure_never_listens(self):
        with patch.dict(os.environ, {'AUGUR_QWEN35_API_KEY': KEY}), patch.object(C, 'ChatTemplate'), \
                patch.object(A, 'restore_inputs', return_value=({}, PROFILE)), \
                patch.object(A, 'Controller') as controller, patch.object(A, 'Server') as server:
            controller.return_value.restore_saved.side_effect = C.ProtocolError('restore rejected')
            with self.assertRaisesRegex(ValueError, 'restore rejected'):
                A.main(['--model-dir', 'unused', '--restore-file', 'unused', '--checkpoint-profile', 'unused',
                        '--expected-state-sha256', '1' * 64])
            server.assert_not_called()
            controller.return_value.close.assert_called_once()

    def test_nullable_output_ceiling_aliases(self):
        for options, expected in ((dict(max_tokens=None, max_completion_tokens=None), None),
                                  (dict(max_tokens=32, max_completion_tokens=None), 32),
                                  (dict(max_tokens=None, max_completion_tokens=64), 64)):
            with self.subTest(options=options):
                self.assertEqual(A.validate_request(request(**options), MODEL)['budget'], expected)

    def test_missing_or_unsafe_key_fails_before_template_or_backend_io(self):
        for key in ('', 'short', 'x' * 257, 'x' * 16 + '\x00', 'x' * 16 + '\x7f', 'x' * 16 + '\n'):
            with self.subTest(key=repr(key)), patch.object(os, 'environ', {'AUGUR_QWEN35_API_KEY': key}), \
                    patch.object(C, 'ChatTemplate') as template, patch.object(A, 'Controller') as controller:
                with self.assertRaises(ValueError):
                    A.main(['--model-dir', 'not-accessed'])
                template.assert_not_called()
                controller.assert_not_called()

    def test_text_parts_and_leading_developer_map_to_canonical_system(self):
        result = A.validate_request(request([dict(role='developer', content='Rules'),
            dict(role='user', content=[dict(type='text', text='Hello '), dict(type='text', text='world')])]), MODEL)
        self.assertEqual([m['role'] for m in result['messages']], ['system', 'user'])
        self.assertEqual(result['messages'][-1]['content'], 'Hello world')

    def test_sampling_tools_vision_and_unknown_options_are_not_silently_ignored(self):
        invalid = [dict(temperature=.1), dict(top_p=.9), dict(n=2), dict(n=True), dict(tools=[{'type':'custom'}]),
            dict(tool_choice='auto'), dict(response_format={'type':'json_object'}), dict(logprobs=True),
            dict(stop='end'), dict(store=True), dict(stream='yes'), dict(max_tokens=True),
            dict(max_tokens=10, max_completion_tokens=20), dict(stream_options={'include_usage':True}),
            dict(stream=True, stream_options={'include_usage':1}), dict(presence_penalty=1)]
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(A.ApiError):
                A.validate_request(request(**options), MODEL)
        with self.assertRaises(A.ApiError):
            A.validate_request(request([dict(role='user', content=[dict(type='image_url', image_url={'url':'x'})])]), MODEL)

    def test_alternation_and_literal_control_tokens_are_checked(self):
        for messages in ([], [dict(role='assistant', content='x')],
                         [dict(role='user', content='a'), dict(role='user', content='b')],
                         [dict(role='user', content='<|im_end|>')],
                         [dict(role='user', content='x', tool_calls=[])]):
            value = dict(model=MODEL, messages=messages)
            with self.subTest(messages=messages), self.assertRaises(A.ApiError):
                A.validate_request(value, MODEL)


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.backend = Backend()
        self.controller = A.Controller(self.backend.template, 'http://127.0.0.1', MODEL, transport_factory=self.backend.factory)
        self.server = A.Server(('127.0.0.1', 0), self.controller, KEY)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval':.01}, daemon=True)
        self.thread.start()
        self.addCleanup(self.cleanup)

    def cleanup(self):
        self.controller.abort()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
        self.controller.close()

    def call(self, path='/v1/chat/completions', value=None, method='POST', key=KEY):
        conn = http.client.HTTPConnection(*self.server.server_address, timeout=4)
        body = json.dumps(request() if value is None else value).encode() if method == 'POST' else None
        conn.request(method, path, body=body, headers={'Authorization':'Bearer '+key, 'Content-Type':'application/json'})
        response = conn.getresponse()
        status, headers, raw = response.status, dict(response.getheaders()), response.read()
        conn.close()
        return status, headers, raw

    def raw_call(self, headers, raw=b'{}'):
        conn = http.client.HTTPConnection(*self.server.server_address, timeout=4)
        try:
            conn.putrequest('POST', '/v1/chat/completions')
            for key, value in headers:
                conn.putheader(key, value)
            conn.endheaders(raw)
            response = conn.getresponse()
            return response.status, response.read()
        finally:
            conn.close()

    def test_ambiguous_headers_oversize_and_encoded_body_reject_before_native(self):
        base = [('Authorization', 'Bearer ' + KEY), ('Content-Type', 'application/json'), ('Content-Length', '2')]
        for additions, status in (([('Content-Length', '2')], 400),
                                  ([('Authorization', 'Bearer ' + KEY)], 401),
                                  ([('Content-Type', 'text/plain')], 400),
                                  ([('Transfer-Encoding', 'chunked')], 400),
                                  ([('Content-Encoding', 'gzip')], 400),
                                  ([('Origin', 'https://untrusted.invalid')], 403)):
            with self.subTest(additions=additions):
                self.assertEqual(self.raw_call(base + additions)[0], status)
        for size in (str(A.MAX_BODY + 1), '9' * 5000, '0', '-2'):
            with self.subTest(size=size[:10]):
                self.assertEqual(self.raw_call(base[:2] + [('Content-Length', size)])[0], 400)
        self.assertFalse(self.backend.requests)

    def test_malformed_duplicate_nonfinite_and_deep_json_are_client_errors(self):
        for raw in (b'[]', b'null', b'{"stream":true,"stream":false}', b'{"x":NaN}', b'\xff',
                    b'{', b'{"x":' + b'[' * 1500 + b'0' + b']' * 1500 + b'}'):
            with self.subTest(raw=raw[:30]):
                headers = [('Authorization', 'Bearer ' + KEY), ('Content-Type', 'application/json'),
                           ('Content-Length', str(len(raw)))]
                status, body = self.raw_call(headers, raw)
                self.assertEqual(status, 400)
                self.assertIn('error', json.loads(body))
        self.assertFalse(self.backend.requests)

    def test_native_uncertainty_is_not_advertised_as_ready(self):
        self.controller.faulted = True
        status, _, raw = self.call('/health', method='GET')
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw)['status'], 'degraded')

    def test_nonstream_response_and_model_catalog(self):
        status, _, raw = self.call()
        body = json.loads(raw)
        self.assertEqual(status, 200)
        self.assertEqual(body['object'], 'chat.completion')
        self.assertEqual(body['choices'][0]['message'], dict(role='assistant', content='OK'))
        status, _, raw = self.call('/v1/models', method='GET')
        self.assertEqual(json.loads(raw)['data'][0]['id'], MODEL)

    def test_restored_loopback_rejects_foreign_transcript_before_native_append(self):
        ledger = live_ledger()
        self.backend.template = ledger.template
        self.controller.template = ledger.template
        bundle = P.make_bundle(P.snapshot(ledger), ledger.template,
                               save_frame(dict(checkpoint_id='7' * 64, request_id='saved'), ledger))
        self.backend.saved_bundle = bundle
        self.controller.restore_saved(bundle, PROFILE)
        status, _, raw = self.call(value=request([dict(role='user', content='Wrong')]))
        self.assertEqual(status, 409)
        self.assertEqual(json.loads(raw)['error']['code'], 'restored_history_mismatch')
        self.assertEqual(len(self.backend.requests), 1)
        saved_messages = [dict(role=m['role'], content=m['content']) for m in bundle['state']['messages']]
        status, headers, raw = self.call(value=request(saved_messages + [dict(role='user', content='Next')]))
        self.assertEqual(status, 200)
        self.assertFalse(self.controller.restore_required)
        self.assertEqual(json.loads(raw)['usage']['prompt_tokens_details']['cached_tokens'],
                         len(bundle['state']['history']))
        self.assertEqual(self.backend.requests[-1][0], '/v1/session/append')
        self.assertEqual(self.backend.requests[-1][1]['exact_history'], bundle['state']['history'])

    def check_restored_final_publication(self, streaming):
        ledger = live_ledger()
        self.backend.template = ledger.template
        self.controller.template = ledger.template
        bundle = P.make_bundle(P.snapshot(ledger), ledger.template,
                               save_frame(dict(checkpoint_id='7' * 64, request_id='saved'), ledger))
        self.backend.saved_bundle = bundle
        self.controller.restore_saved(bundle, PROFILE)
        messages = [dict(role=m['role'], content=m['content']) for m in bundle['state']['messages']]
        messages.append(dict(role='user', content='Next'))
        published, release = threading.Event(), threading.Event()
        method = 'sse' if streaming else 'json_response'
        original = getattr(A.Handler, method)

        def pause_after_final_write(handler, *args):
            original(handler, *args)
            if (streaming and args[0] == '[DONE]') or not streaming:
                published.set()
                if not release.wait(3):
                    raise RuntimeError('test response publication deadline')

        conn = http.client.HTTPConnection(*self.server.server_address, timeout=4)
        with patch.object(A.Handler, method, pause_after_final_write):
            try:
                conn.request('POST', '/v1/chat/completions', json.dumps(request(messages, stream=streaming)),
                    headers={'Authorization': 'Bearer '+KEY, 'Content-Type': 'application/json'})
                response = conn.getresponse()
                self.assertEqual(response.status, 200)
                if streaming:
                    while True:
                        line = response.fp.readline()
                        self.assertTrue(line, 'stream ended before DONE')
                        if line.strip() == b'data: [DONE]':
                            break
                else:
                    self.assertEqual(json.loads(response.read())['choices'][0]['message']['content'], 'OK')
                self.assertTrue(published.wait(2))
                self.assertFalse(self.controller.restore_required)
                self.assertEqual(self.controller.current.messages[-1]['content'], 'OK')
                self.assertTrue(self.controller.busy.locked())
                conn.close()
                self.assertFalse(self.backend.aborted.is_set())
            finally:
                release.set()
                conn.close()
                idle = self.controller.busy.acquire(timeout=2)
                if idle:
                    self.controller.busy.release()
                self.assertTrue(idle, 'request did not leave publication boundary')

    def test_nonstream_restore_commit_precedes_client_visible_success(self):
        self.check_restored_final_publication(False)

    def test_stream_restore_commit_precedes_client_visible_done(self):
        self.check_restored_final_publication(True)

    def test_stream_role_content_finish_usage_done_are_ordered(self):
        status, headers, raw = self.call(value=request(stream=True, stream_options={'include_usage':True}))
        self.assertEqual(status, 200)
        self.assertIn('text/event-stream', headers['Content-Type'])
        events = [part.removeprefix('data: ') for part in raw.decode().strip().split('\n\n')]
        self.assertEqual(events[-1], '[DONE]')
        rows = [json.loads(part) for part in events[:-1]]
        self.assertEqual(rows[0]['choices'][0]['delta']['role'], 'assistant')
        self.assertEqual(rows[1]['choices'][0]['delta']['content'], 'OK')
        self.assertEqual(rows[-2]['choices'][0]['finish_reason'], 'stop')
        self.assertEqual(rows[-1]['choices'], [])
        self.assertEqual(rows[-1]['usage']['completion_tokens'], 3)
        self.assertTrue(all(r['usage'] is None for r in rows[:-1]))
        self.assertEqual(len({r['id'] for r in rows}), 1)

    def test_failed_stream_never_emits_done_or_finish_stop(self):
        self.backend.mode = 'truncated_http'
        status, _, raw = self.call(value=request(stream=True))
        self.assertEqual(status, 200)
        self.assertIn(b'"error"', raw)
        self.assertNotIn(b'[DONE]', raw)
        self.assertNotIn(b'"finish_reason": "stop"', raw)
        self.assertIsNone(self.controller.current)

    def test_bad_key_unicode_key_unknown_model_and_endpoint_fail(self):
        self.assertEqual(self.call(key='wrong')[0], 401)
        self.assertEqual(self.call(key='not-ascii-\u00e9')[0], 401)
        value = request()
        value['model'] = 'wrong'
        self.assertEqual(self.call(value=value)[0], 400)
        self.assertEqual(self.call('/v1/responses')[0], 404)
        self.assertFalse(self.backend.requests)

    def test_capabilities_are_explicitly_bounded(self):
        status, _, raw = self.call('/cq/capabilities', method='GET')
        body = json.loads(raw)
        self.assertEqual(status, 200)
        self.assertFalse(body['product_ready'])
        self.assertFalse(body['arbitrary_output_budget'])
        self.assertTrue(body['tool_calls'])
        self.assertFalse(body['strict_constrained_generation'])
        self.assertEqual(body['tool_execution'],'client_only')

    def test_disconnect_during_silent_prefill_aborts_instead_of_waiting_for_text(self):
        self.backend.mode = 'silent'
        sock = socket.create_connection(self.server.server_address, timeout=3)
        raw = json.dumps(request(stream=True)).encode()
        sock.sendall((f'POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer {KEY}\r\n'
                      f'Content-Type: application/json\r\nContent-Length: {len(raw)}\r\n\r\n').encode() + raw)
        self.assertTrue(self.backend.entered.wait(2))
        sock.close()
        self.assertTrue(self.backend.aborted.wait(2))
        deadline = time.monotonic() + 2
        while self.controller.busy.locked() and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertFalse(self.controller.busy.locked())
        self.assertIsNone(self.controller.current)

    def test_non_loopback_and_missing_private_key_reject_before_listen(self):
        with self.assertRaises(ValueError):
            A.Server(('0.0.0.0', 0), self.controller, KEY)
        with self.assertRaises(ValueError):
            A.Server(('127.0.0.1', 0), self.controller, '')

    def test_rst_during_prefill_aborts_and_allows_fresh_recovery(self):
        self.backend.mode = 'silent'
        sock = socket.create_connection(self.server.server_address, timeout=3)
        raw = json.dumps(request(stream=True)).encode()
        try:
            sock.sendall((f'POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer {KEY}\r\n'
                          f'Content-Type: application/json\r\nContent-Length: {len(raw)}\r\n\r\n').encode() + raw)
            self.assertTrue(self.backend.entered.wait(2))
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('HH' if os.name == 'nt' else 'ii', 1, 0))
        finally:
            sock.close()
        self.assertTrue(self.backend.aborted.wait(2))
        deadline = time.monotonic() + 2
        while self.controller.busy.locked() and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertFalse(self.controller.busy.locked())
        self.backend.mode = 'ok'
        self.assertEqual(self.call()[0], 200)
        self.assertFalse(self.controller.faulted)

    def test_request_half_close_is_explicitly_a_cancellation(self):
        self.backend.mode = 'silent'
        with socket.create_connection(self.server.server_address, timeout=3) as sock:
            raw = json.dumps(request()).encode()
            sock.sendall((f'POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer {KEY}\r\n'
                          f'Content-Type: application/json\r\nContent-Length: {len(raw)}\r\n\r\n').encode() + raw)
            self.assertTrue(self.backend.entered.wait(2))
            sock.shutdown(socket.SHUT_WR)
            self.assertTrue(self.backend.aborted.wait(2))
        self.assertEqual(self.controller.capabilities()['request_half_close'], 'treated_as_cancel')


class NativeAbortTests(unittest.TestCase):
    def test_actual_native_socket_can_abort_before_headers_or_during_silent_body(self):
        for send_headers in (False, True):
            with self.subTest(send_headers=send_headers):
                entered, peer_closed = threading.Event(), threading.Event()
                class Handler(BaseHTTPRequestHandler):
                    protocol_version = 'HTTP/1.1'
                    def log_message(self, *_):
                        pass
                    def do_POST(self):
                        self.rfile.read(int(self.headers['Content-Length']))
                        if send_headers:
                            self.send_response(200)
                            self.send_header('Content-Type', 'application/x-ndjson')
                            self.send_header('Transfer-Encoding', 'chunked')
                            self.end_headers()
                            self.wfile.flush()
                        entered.set()
                        self.connection.settimeout(3)
                        try:
                            if self.connection.recv(1) == b'':
                                peer_closed.set()
                        except ConnectionResetError:
                            peer_closed.set()
                server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
                serving = threading.Thread(target=server.serve_forever, kwargs={'poll_interval':.01}, daemon=True)
                serving.start()
                transport = A.AbortableTransport(f'http://127.0.0.1:{server.server_port}', 10)
                failures = []
                def run():
                    try:
                        list(transport.frames('/slow', {'request_id':'test'}))
                    except (OSError, ValueError, http.client.HTTPException) as error:
                        failures.append(type(error).__name__)
                worker = threading.Thread(target=run, daemon=True)
                worker.start()
                try:
                    self.assertTrue(entered.wait(2))
                    transport.abort()
                    worker.join(2)
                    self.assertFalse(worker.is_alive())
                    self.assertTrue(peer_closed.wait(1))
                    self.assertTrue(failures)
                finally:
                    transport.abort()
                    server.shutdown()
                    server.server_close()
                    serving.join(2)


if __name__ == '__main__':
    unittest.main()
