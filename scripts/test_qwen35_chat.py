"""CPU protocol and real TCP tests; no model/GPU needed."""
import copy
import io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import qwen35_chat as C


class Template:
    end, eos, vocab = 1000, {1000}, set(range(1101))
    identity = {'tokenizer_sha256': 'a' * 64}
    message = staticmethod(C.ChatTemplate.message)

    def encode(self, text):
        return [ord(c) for c in text]

    def render(self, messages):
        return ''.join(f'<|im_start|>{m["role"]}\n{m["content"]}<|im_end|>\n'
                       for m in messages) + '<|im_start|>assistant\n'


def fixture(ledger, plan, text='OK', stop='eos'):
    visible = ledger.template.encode(text)
    history = plan['input_ids'] + (visible if stop == 'eos' else visible[:-1])
    identity = dict(schema=C.SCHEMA, request_id=plan['payload']['request_id'],
                    request_sha256='b' * 64)
    begin = dict(identity, event='begin', sequence=0, tokenizer_sha256='a' * 64)
    middle = dict(identity, event='text', sequence=1, token_ids=visible, text=text)
    terminal = dict(identity, event='terminal', sequence=2, stop_reason=stop,
                    visible_token_ids=visible, text_sha256=C.sha256(text.encode()),
                    native_report_sha256='c' * 64,
                    session=dict(session_id=ledger.session_id, capacity=196608,
                        revision=1 if ledger.terminal is None else ledger.terminal['session']['revision'] + 2,
                        pending_token=1000 if stop == 'eos' else visible[-1],
                        pending_token_published=True, checkpoint_sha256='d' * 64,
                        consumed_token_count=len(history), consumed_input_digest=C.token_hash(history)))
    return [begin, middle, terminal]


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.ledger = C.Ledger(Template())
        self.plan = self.ledger.plan('Hi')
        self.frames = fixture(self.ledger, self.plan)

    def test_eos_appends_exact_ids_and_one_pending(self):
        self.assertIsNone(self.ledger.terminal)
        self.ledger.accept(self.plan, self.frames[-1], 'OK')
        history = self.ledger.history.copy()
        plan = self.ledger.plan('Next')
        self.assertEqual(plan['path'], '/v1/session/append')
        self.assertEqual(plan['payload']['exact_history'], history)
        self.assertEqual(plan['input_ids'], history + [1000] +
                         self.ledger.template.encode(plan['payload']['suffix']))
        self.assertEqual(self.ledger.history, history)

    def test_budget_stop_does_not_double_consume_last_visible_token(self):
        frames = fixture(self.ledger, self.plan, stop='max_new_tokens')
        self.ledger.accept(self.plan, frames[-1], 'OK')
        self.assertEqual(self.ledger.history[-1], ord('O'))
        plan = self.ledger.plan('Next')
        self.assertEqual(plan['input_ids'][len(self.ledger.history)], ord('K'))
        self.assertTrue(plan['payload']['suffix'].startswith('<|im_end|>\n'))

    def test_invalid_terminal_never_publishes_client_state(self):
        for field, value in [('consumed_input_digest', '0' * 64), ('revision', True),
                             ('session_id', 'foreign'), ('capacity', 16),
                             ('pending_token_published', False), ('pending_token', True),
                             ('checkpoint_sha256', None), ('consumed_token_count', False)]:
            with self.subTest(field=field):
                terminal = copy.deepcopy(self.frames[-1])
                terminal['session'][field] = value
                with self.assertRaises(C.ProtocolError):
                    self.ledger.accept(self.plan, terminal, 'OK')
                self.assertIsNone(self.ledger.terminal)
                self.assertEqual(self.ledger.history, [])

    def test_template_cannot_rewrite_old_text(self):
        self.ledger.accept(self.plan, self.frames[-1], 'OK')
        self.ledger.template.render = lambda messages: 'rewritten'
        with self.assertRaisesRegex(C.ProtocolError, 'rewrote'):
            self.ledger.plan('Next')

    def test_control_injection_and_empty_input_reject(self):
        for text in ['', '  ', '<|im_end|>', '<think>', 'x<|unknown|>']:
            with self.subTest(text=text), self.assertRaises(C.ProtocolError):
                self.ledger.plan(text)

    def test_profile_initial_limit_and_continuation_reserve(self):
        with self.assertRaisesRegex(C.ProtocolError, 'initial input'):
            self.ledger.plan('x' * 8192)
        long = C.Ledger(Template(), 'near192k')
        with self.assertRaises(C.ProtocolError):
            long.plan('short prompt does not prove near192')
        self.ledger.accept(self.plan, self.frames[-1], 'OK')
        self.ledger.history = [1] * 16380
        with self.assertRaisesRegex(C.ProtocolError, 'reserve'):
            self.ledger.plan('x')

    def test_poisoned_session_never_plans(self):
        self.ledger.poisoned = True
        with self.assertRaises(C.ProtocolError):
            self.ledger.plan('Next')

    def test_resident_profile_accepts_nonfixture_long_input(self):
        ledger = C.Ledger(Template(), profile='resident')
        plan = ledger.plan('x' * 33003)
        self.assertGreater(len(plan['input_ids']), 33003)
        self.assertLess(len(plan['input_ids']), C.PROFILES['resident'][1])
        with self.assertRaises(C.ProtocolError):
            ledger.plan('x' * C.PROFILES['resident'][1])

    def test_system_is_preserved_by_model_template(self):
        ledger = C.Ledger(Template(), system='Helpful assistant')
        plan = ledger.plan('Hi')
        self.assertIn('<|im_start|>system\nHelpful assistant', plan['prefix'])

    def test_transport_rejects_remote_or_credential_urls(self):
        for url in ['http://example.com', 'https://localhost', 'http://user@localhost',
                    'http://localhost/path', 'http://localhost?x=y']:
            with self.subTest(url=url), self.assertRaises(C.ProtocolError):
                C.Transport(url)
        for timeout in [0, -1, float('inf'), float('nan')]:
            with self.assertRaises(C.ProtocolError):
                C.Transport('http://localhost', timeout)

    def test_terminal_controls_are_displayed_not_executed(self):
        self.assertEqual(C.terminal_text('a\x1b[31mb\r\x00\x9b\n\t'),
                         'a\\x1b[31mb\\x0d\\x00\\x9b\n\t')
        self.assertEqual(C.terminal_text('青竹'), '青竹')


class StreamTests(unittest.TestCase):
    def setUp(self):
        ledger = C.Ledger(Template())
        plan = ledger.plan('Hi')
        self.frames = fixture(ledger, plan)
        self.rid = plan['payload']['request_id']

    def verify(self, frames, template=None):
        verifier = C.StreamVerifier(self.rid, template or Template())
        for frame in frames:
            verifier.push(frame)
        return verifier.finish()

    def test_valid_stream(self):
        self.assertEqual(self.verify(self.frames)[1], 'OK')

    def test_unicode_completed_frame_keeps_all_buffered_token_ids(self):
        frames = copy.deepcopy(self.frames)
        frames[1].update(token_ids=[3, 6, 1, 2], text='A €')
        frames[2].update(visible_token_ids=[3, 6, 1, 2],
                         text_sha256=C.sha256('A €'.encode('utf-8')))
        terminal, text = self.verify(frames)
        self.assertEqual(text, 'A €')
        self.assertEqual(terminal['visible_token_ids'], [3, 6, 1, 2])
        frames[2]['visible_token_ids'] = [3, 2]
        with self.assertRaises(C.ProtocolError):
            self.verify(frames)

    def flush_fixture(self, stop='max_new_tokens'):
        frames = copy.deepcopy(self.frames)
        frames[0]['terminal_text_flush_supported'] = True
        frames[1].update(token_ids=[65, 226], text='A\ufffd', terminal_decode_flush=stop)
        frames[2].update(visible_token_ids=[65, 226], stop_reason=stop,
                         text_sha256=C.sha256('A\ufffd'.encode('utf-8')))
        template = Template()
        template.tokenizer = SimpleNamespace(decode=lambda ids, skip_special_tokens:
            bytes(ids).decode('utf-8', errors='replace'))
        return frames, template

    def test_terminal_flush_uses_exact_full_token_decode(self):
        for stop in ['eos', 'max_new_tokens']:
            with self.subTest(stop=stop):
                frames, template = self.flush_fixture(stop)
                terminal, text = self.verify(frames, template)
                self.assertEqual(text, 'A\ufffd')
                self.assertEqual(terminal['visible_token_ids'], [65, 226])

    def test_terminal_flush_requires_capability_reason_and_correct_decode(self):
        for index, key, value in [
            (0, 'terminal_text_flush_supported', False),
            (0, 'terminal_text_flush_supported', 1),
            (1, 'terminal_decode_flush', None), (1, 'terminal_decode_flush', True),
            (1, 'terminal_decode_flush', 'cancelled'),
            (1, 'text', 'B\ufffd'), (1, 'token_ids', [66, 226]),
            (2, 'stop_reason', 'eos')]:
            with self.subTest(index=index, key=key, value=value):
                frames, template = self.flush_fixture()
                frames[index][key] = value
                with self.assertRaises(C.ProtocolError):
                    self.verify(frames, template)
        frames, template = self.flush_fixture()
        del frames[0]['terminal_text_flush_supported']
        with self.assertRaises(C.ProtocolError):
            self.verify(frames, template)

    def test_terminal_flush_cannot_continue_text_or_omit_terminal(self):
        frames, template = self.flush_fixture()
        with self.assertRaises(C.ProtocolError):
            self.verify(frames[:-1], template)
        extra = copy.deepcopy(frames[1])
        extra.update(sequence=2, token_ids=[66], text='B')
        del extra['terminal_decode_flush']
        frames[2]['sequence'] = 3
        with self.assertRaises(C.ProtocolError):
            self.verify([frames[0], frames[1], extra, frames[2]], template)

    def test_terminal_flush_decode_error_is_protocol_error(self):
        frames, template = self.flush_fixture()
        template.tokenizer.decode = mock.Mock(side_effect=ValueError('decoder failure'))
        with self.assertRaises(C.ProtocolError):
            self.verify(frames, template)

    def test_binding_mutations_reject(self):
        for index, key, value in [
            (0, 'tokenizer_sha256', 'f' * 64), (0, 'sequence', True),
            (1, 'request_id', 'foreign'), (1, 'request_sha256', 'f' * 64),
            (1, 'sequence', 9), (1, 'token_ids', [True]), (1, 'token_ids', [1000]),
            (1, 'text', '\ufffd'), (2, 'visible_token_ids', [True]),
            (2, 'text_sha256', 'f' * 64), (2, 'native_report_sha256', 'bad')]:
            with self.subTest(key=key):
                frames = copy.deepcopy(self.frames)
                frames[index][key] = value
                with self.assertRaises(C.ProtocolError):
                    self.verify(frames)

    def test_incomplete_reordered_and_extra_streams_reject(self):
        for frames in [[], self.frames[:-1], self.frames[1:],
                       self.frames + [self.frames[-1]], [self.frames[0], self.frames[2]]]:
            with self.subTest(frames=len(frames)), self.assertRaises(C.ProtocolError):
                self.verify(frames)

    def test_strict_json_duplicate_and_nonfinite(self):
        for raw in ['{"x":1,"x":2}', '{"x":NaN}', '[]']:
            with self.assertRaises(C.ProtocolError):
                C.strict_json(raw)


class ReadClockTests(unittest.TestCase):
    @staticmethod
    def chunk(raw):
        return f'{len(raw):X}\r\n'.encode() + raw + b'\r\n'

    def parse(self, blocks):
        response = mock.Mock()
        unread = list(blocks)

        def read(n):
            if not unread:
                return b''
            block = unread[0]
            raw, unread[0] = block[:n], block[n:]
            if not unread[0]:
                unread.pop(0)
            return raw

        response.fp.read1.side_effect = read
        clock = SimpleNamespace(perf_counter=mock.Mock(side_effect=(100 + i * .001 for i in range(1000))))
        with mock.patch.object(C, 'time', clock):
            rows = list(C.Transport('http://127.0.0.1:1')._native_frames(response, 100.0))
        return rows, response

    def test_one_read_across_http_chunks_has_one_arrival_time(self):
        a, b, c = (b'{"i":1}\n', b'{"i":2}\n', b'{"i":3}\n')
        rows, response = self.parse([self.chunk(a + b) + self.chunk(c) + b'0\r\n\r\n'])
        self.assertEqual([r[0]['i'] for r in rows], [1, 2, 3])
        self.assertEqual(len(set(r[1] for r in rows)), 1)
        self.assertEqual(response.fp.read1.call_count, 1)
        self.assertGreater(rows[0][1], 0)

    def test_split_frame_uses_read_that_completes_its_newline(self):
        a, b = b'{"i":1}\n', b'{"i":2}\n'
        rows, response = self.parse([self.chunk(a) + self.chunk(b[:-1]),
                                    self.chunk(b[-1:]) + b'0\r\n\r\n'])
        self.assertEqual([r[0]['i'] for r in rows], [1, 2])
        self.assertLess(rows[0][1], rows[1][1])
        self.assertEqual(response.fp.read1.call_count, 2)

    def test_every_wire_split_preserves_complete_chunked_frames(self):
        wire = self.chunk(b'{"i":1}\n') + self.chunk(b'{"i":2}\n') + b'0\r\n\r\n'
        for offset in range(1, len(wire)):
            with self.subTest(offset=offset):
                rows, _ = self.parse([wire[:offset], wire[offset:]])
                self.assertEqual([r[0]['i'] for r in rows], [1, 2])
                self.assertLessEqual(rows[0][1], rows[1][1])

    def test_malformed_or_truncated_buffered_wire_cannot_finish(self):
        for wire in (b'8;x=1\r\n{"i":1}\n\r\n0\r\n\r\n',
                     self.chunk(b'{"i":1}\n')[:-2] + b'XX0\r\n\r\n',
                     self.chunk(b'{"i":1}\n') + b'0\r\n',
                     self.chunk(b'{"i":1}') + b'0\r\n\r\n'):
            with self.subTest(wire=wire), self.assertRaises(C.ProtocolError):
                self.parse([wire])

    def test_read_completing_after_deadline_cannot_publish_frame(self):
        response = mock.Mock()
        response.fp.read1.return_value = self.chunk(b'{"i":1}\n') + b'0\r\n\r\n'
        clock = SimpleNamespace(perf_counter=mock.Mock(side_effect=[100.0, 102.0]))
        with mock.patch.object(C, 'time', clock), self.assertRaisesRegex(C.ProtocolError, 'deadline exceeded'):
            list(C.Transport('http://127.0.0.1:1', timeout=1)._native_frames(response, 100.0))

    def test_initial_delivery_batch_is_excluded_from_post_first_rate(self):
        for times, expected in (([1., 1., 2.], 1.), ([1., 1., 1.], None)):
            with self.subTest(times=times):
                ledger, transport = C.Ledger(Template()), C.Transport('http://127.0.0.1:1')

                def frames(path, payload):
                    values = fixture(ledger, dict(payload=payload, input_ids=ledger.template.encode(payload['prompt'])), text='ABC')
                    yield values[0], .5
                    for index, (char, elapsed) in enumerate(zip('ABC', times), 1):
                        yield dict(values[1], sequence=index, text=char, token_ids=[ord(char)]), elapsed
                    yield dict(values[-1], sequence=4), 3.

                with mock.patch.object(transport, 'frames', frames):
                    text, metrics = transport.turn(ledger, 'Hi', lambda _: None)
                self.assertEqual(text, 'ABC')
                self.assertEqual(metrics['post_first_tps'], expected)
                self.assertEqual(metrics['frame_clock'], C.FRAME_CLOCK)


class SocketTests(unittest.TestCase):
    def run_server(self, mode='normal'):
        emitted = threading.Event()
        observed = []
        ledger = C.Ledger(Template())
        if mode == 'unicode':
            ledger.template.vocab = ledger.template.vocab | {ord('青'), ord('竹')}

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *_):
                pass

            def do_POST(self):
                request = C.strict_json(self.rfile.read(int(self.headers['Content-Length'])))
                observed.append(request)
                # Match the client's real request identity and initial canonical bytes.
                plan = dict(payload=request, input_ids=ledger.template.encode(request['prompt']))
                frames = fixture(ledger, plan, text='青竹' if mode == 'unicode' else 'OK')
                if mode == 'bad-history':
                    frames[-1]['session']['consumed_input_digest'] = '0' * 64
                if mode == 'rejected':
                    frames = [dict(event='rejected', detail='test rejection')]
                self.send_response(200)
                self.send_header('Content-Type', 'application/x-ndjson')
                self.send_header('Transfer-Encoding', 'chunked')
                self.send_header('Connection', 'close')
                self.end_headers()
                try:
                    for index, frame in enumerate(frames):
                        if mode == 'missing-terminal' and frame['event'] == 'terminal':
                            break
                        raw = (json.dumps(frame, ensure_ascii=False) + '\n').encode()
                        # Split a single JSON frame across two HTTP chunks, and write
                        # UTF-8/NDJSON separately from transport chunk boundaries.
                        parts = [bytes([b]) for b in raw] if mode == 'unicode' else (raw[:7], raw[7:])
                        for part in parts:
                            self.wfile.write(f'{len(part):X}\r\n'.encode() + part + b'\r\n')
                            self.wfile.flush()
                        if index == 1 and mode == 'normal':
                            observed.append(('streamed_before_terminal', emitted.wait(2)))
                    if mode != 'truncated-http':
                        self.wfile.write(b'0\r\n' if mode == 'truncated-final-crlf' else b'0\r\n\r\n')
                        self.wfile.flush()
                except OSError:
                    pass
                self.close_connection = True

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        transport = C.Transport(f'http://127.0.0.1:{server.server_port}', timeout=4)
        return ledger, transport, emitted, observed

    def test_real_tcp_streams_before_terminal_and_accepts_complete_framing(self):
        ledger, transport, emitted, observed = self.run_server()
        output = []
        text, metrics = transport.turn(ledger, 'Hi', lambda t: (output.append(t), emitted.set()))
        self.assertEqual(text, 'OK')
        self.assertEqual(output, ['OK'])
        self.assertIn(('streamed_before_terminal', True), observed)
        self.assertEqual(metrics['visible_tokens'], 2)
        self.assertGreater(metrics['http_wall_s'], 0)
        self.assertIsNotNone(ledger.terminal)

    def test_sub_tick_native_metrics_and_deadline_use_one_precise_clock(self):
        ledger, transport, emitted, _ = self.run_server()
        clock = SimpleNamespace(monotonic=mock.Mock(return_value=9000.0),
                                perf_counter=mock.Mock(side_effect=(100.0 + i * .000001
                                                                    for i in range(1000))))
        with mock.patch.object(C, 'time', clock):
            # Deliberately different epochs: mixing these clocks must fail.
            text, metrics = transport.turn(ledger, 'Hi', lambda _: emitted.set())
            clock.monotonic.assert_not_called()
        self.assertEqual(text, 'OK')
        self.assertGreater(metrics['ttft_s'], 0)
        self.assertGreater(metrics['http_wall_s'], metrics['ttft_s'])
        self.assertLess(metrics['http_wall_s'], .001)

    def test_native_framing_deadline_uses_precise_request_start(self):
        response = mock.Mock()
        transport = C.Transport('http://127.0.0.1:1', timeout=1)
        clock = SimpleNamespace(monotonic=mock.Mock(return_value=50.0),
                                perf_counter=mock.Mock(return_value=51.001))
        with mock.patch.object(C, 'time', clock):
            with self.assertRaisesRegex(C.ProtocolError, 'deadline exceeded'):
                next(transport._native_frames(response, started=50.0))
            clock.monotonic.assert_not_called()
        response.fp.read1.assert_not_called()
        response.fp.raw._sock.settimeout.assert_not_called()

    def test_no_terminal_bad_ledger_and_truncated_http_never_commit(self):
        for mode in ['missing-terminal', 'bad-history', 'truncated-http', 'truncated-final-crlf', 'rejected']:
            with self.subTest(mode=mode):
                ledger, transport, _, _ = self.run_server(mode)
                with self.assertRaises((C.ProtocolError, C.http.client.HTTPException)):
                    transport.turn(ledger, 'Hi', lambda _: None)
                self.assertTrue(ledger.poisoned)
                self.assertIsNone(ledger.terminal)
                self.assertEqual(ledger.history, [])

    def test_callback_interrupt_closes_stream_and_poisons_client(self):
        ledger, transport, _, _ = self.run_server('interrupt')

        def interrupt(_):
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            transport.turn(ledger, 'Hi', interrupt)
        self.assertTrue(ledger.poisoned)
        self.assertIsNone(ledger.terminal)

    def test_utf8_tokens_across_single_byte_http_chunks(self):
        ledger, transport, _, _ = self.run_server('unicode')
        text, _ = transport.turn(ledger, 'Hi', lambda _: None)
        self.assertEqual(text, '青竹')


class CommandLineTests(unittest.TestCase):
    def check_exit(self, inputs, expected_code):
        closed = []

        class FakeTransport:
            def __init__(self, *_):
                pass

            def health(self):
                return {'profile': 'growing'}

            def turn(self, ledger, text, emit):
                plan = ledger.plan(text)
                frames = fixture(ledger, plan)
                ledger.accept(plan, frames[-1], 'OK')
                emit('OK')
                return 'OK', {'test_only': True}

            def close_session(self, ledger):
                closed.append(ledger.session_id)
                ledger.poisoned = True

        with mock.patch.object(C, 'Transport', FakeTransport), \
             mock.patch.object(C, 'ChatTemplate', lambda _: Template()), \
             mock.patch.object(C.sys.stdin, 'isatty', return_value=True), \
             mock.patch('builtins.input', side_effect=inputs), \
             mock.patch.object(C.sys, 'stdout', io.StringIO()), \
             mock.patch.object(C.sys, 'stderr', io.StringIO()):
            self.assertEqual(C.main(['--model-dir', 'fixture']), expected_code)
        self.assertEqual(len(closed), 1)

    def test_ctrl_c_between_turns_closes_healthy_session(self):
        self.check_exit(['Hi', KeyboardInterrupt], 130)

    def test_local_followup_error_closes_healthy_session(self):
        self.check_exit(['Hi', '<|im_end|>'], 1)

    def test_quit_closes_exactly_once(self):
        self.check_exit(['Hi', '/quit'], 0)


if __name__ == '__main__':
    unittest.main()
