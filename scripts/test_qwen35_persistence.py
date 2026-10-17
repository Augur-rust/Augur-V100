"""Client state/HTTP checkpoint boundary tests. No model, CUDA or weights."""
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import qwen35_chat as C
import qwen35_persistence as P
from test_qwen35_chat import Template, fixture

PROFILE = '1' * 64


def live_ledger(stop='eos'):
    template = Template()
    template.identity = dict(tokenizer_sha256='a' * 64, template_sha256='b' * 64, generation_sha256='c' * 64)
    template.tokenizer = SimpleNamespace(decode=lambda ids, skip_special_tokens:
                                         ''.join(chr(t) for t in ids if t not in template.eos))
    ledger = C.Ledger(template, 'resident')
    plan = ledger.plan('Remember version 17.')
    terminal = fixture(ledger, plan, stop=stop)[-1]
    ledger.accept(plan, terminal, 'OK')
    return ledger


def save_frame(payload, ledger):
    return dict(schema='qwen35-http-checkpoint-terminal-v1', event='session_saved', **payload,
        owner_id='2' * 64, owner_epoch=101, profile_sha256=PROFILE,
        http_manifest_sha256='3' * 64, native_manifest_sha256='4' * 64,
        session=copy.deepcopy(ledger.terminal['session']), **dict.fromkeys(P.CLOSED_CLAIMS, False))


def rejected_save_frame(payload, ledger):
    return dict(schema='qwen35-http-checkpoint-admission-rejection-v1', event='rejected',
        operation='save', before_submit=True, capture_started=False, session_preserved=True, **payload,
        owner_id='2' * 64, owner_epoch=101, profile_sha256=PROFILE,
        session=copy.deepcopy(ledger.terminal['session']), detail='checkpoint admission: insufficient disk space',
        **dict.fromkeys(P.CLOSED_CLAIMS, False))


def restore_frame(payload, bundle):
    state, checkpoint = bundle['state'], bundle['checkpoint']
    session = copy.deepcopy(state['terminal']['session'])
    session.update(session_id=payload['session_id'], checkpoint_sha256='5' * 64)
    return dict(schema='qwen35-http-checkpoint-terminal-v1', event='session_restored', **payload,
        revision=session['revision'], owner_id='6' * 64, owner_epoch=202, profile_sha256=PROFILE,
        native_manifest_sha256=checkpoint['native_manifest_sha256'], session=session,
        source_owner_id=checkpoint['owner_id'], source_session_id=state['session_id'],
        target_seal_verified=True, client_delivery_acknowledged=False,
        restored_prefix_tokens=len(state['history']), prefix_recompute_tokens=0,
        native_restore=dict(fresh_native_session_owner=True, device_pending_token_verified=True,
            draft_rebuild_submitted_and_drained=True, old_cuda_authority_restored=False,
            draft_rebuild_outputs_published=False, restored_prefix_tokens=len(state['history']),
            prefix_recompute_tokens=0), **dict.fromkeys(P.CLOSED_CLAIMS, False))


class Transport:
    def __init__(self, respond, extra=False, truncate=False, profile='resident'):
        self.respond, self.extra, self.truncate, self.profile = respond, extra, truncate, profile
        self.calls, self.closed = [], False

    def health(self):
        return dict(profile=self.profile)

    def frames(self, path, payload):
        self.calls.append((path, copy.deepcopy(payload)))
        try:
            frame = self.respond(payload)
            yield frame, 0.1
            if self.extra:
                yield frame, 0.2
            if self.truncate:
                raise C.ProtocolError('truncated final HTTP chunk')
        finally:
            self.closed = True


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / 'private-session.json'
        self.ledger = live_ledger()

    def saved(self, stop='eos'):
        if stop != 'eos':
            self.ledger = live_ledger(stop)
        transport = Transport(lambda p: save_frame(p, self.ledger))
        result = P.save(transport, self.ledger, self.path, PROFILE, '7' * 64)
        return result, P.read_bundle(self.path, self.ledger.template, result['file_sha256'])

    def test_default_checkpoint_id_matches_native_hex_abi(self):
        transport = Transport(lambda p: save_frame(p, self.ledger))
        result = P.save(transport, self.ledger, self.path, PROFILE)
        self.assertRegex(result['checkpoint']['checkpoint_id'], r'^[0-9a-f]{64}$')
        self.assertEqual(transport.calls[0][1]['checkpoint_id'], result['checkpoint']['checkpoint_id'])

    def test_invalid_checkpoint_id_rejected_before_file_or_native_request(self):
        for value in ('checkpoint-' + 'a' * 32, 'a' * 32, 'a' * 63, 'a' * 65, 'A' * 64, 'g' * 64,
                      '../escape', True):
            transport = Transport(lambda p: save_frame(p, self.ledger))
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, '64 lowercase hex'):
                P.save(transport, self.ledger, self.path, PROFILE, value)
            self.assertEqual(transport.calls, [])
            self.assertEqual(list(self.root.iterdir()), [])
            self.assertFalse(self.ledger.poisoned)

    def test_saved_named_checkpoint_cannot_reenter_native_restore(self):
        _, bundle = self.saved()
        bundle['checkpoint']['checkpoint_id'] = 'named-checkpoint'
        bundle['content_sha256'] = P.sha256(P.canonical({k: v for k, v in bundle.items() if k != 'content_sha256'}))
        transport = Transport(lambda p: restore_frame(p, bundle))
        with self.assertRaisesRegex(ValueError, 'checkpoint ID'):
            P.restore(transport, bundle, self.ledger.template, PROFILE)
        self.assertEqual(transport.calls, [])

    def test_save_complete_file_and_fresh_owner_restore_exact_ids(self):
        result, bundle = self.saved()
        self.assertFalse(self.ledger.poisoned)
        self.assertEqual(list(self.root.iterdir()), [self.path])
        transport = Transport(lambda p: restore_frame(p, bundle))
        with patch.object(self.ledger.template, 'encode', side_effect=AssertionError('old text retokenized')):
            restored, frame = P.restore(transport, bundle, self.ledger.template, PROFILE)
        self.assertTrue(transport.closed)
        self.assertNotEqual(restored.session_id, self.ledger.session_id)
        self.assertEqual(restored.history, self.ledger.history)
        self.assertEqual(restored.messages, self.ledger.messages)
        self.assertEqual(restored.terminal['schema'], P.ANCHOR)
        next_plan = restored.plan('What version?')
        self.assertEqual(next_plan['path'], '/v1/session/append')
        self.assertEqual(next_plan['payload']['exact_history'], self.ledger.history)
        self.assertEqual(next_plan['input_ids'][len(self.ledger.history)], self.ledger.template.end)

    def test_budget_stop_restore_consumes_pending_once(self):
        _, bundle = self.saved('max_new_tokens')
        restored, _ = P.restore(Transport(lambda p: restore_frame(p, bundle)), bundle, self.ledger.template, PROFILE)
        plan = restored.plan('Continue.')
        self.assertEqual(restored.history[-1], ord('O'))
        self.assertEqual(plan['input_ids'][len(restored.history)], ord('K'))

    def test_truncated_or_duplicate_control_terminal_never_publishes(self):
        for extra, truncate in ((True, False), (False, True)):
            self.ledger = live_ledger()
            transport = Transport(lambda p: save_frame(p, self.ledger), extra=extra, truncate=truncate)
            with self.subTest(extra=extra, truncate=truncate), self.assertRaises(ValueError):
                P.save(transport, self.ledger, self.path, PROFILE)
            self.assertFalse(self.path.exists())
            self.assertEqual(list(self.root.iterdir()), [])
            self.assertTrue(transport.closed)
            self.assertTrue(self.ledger.poisoned)

    def test_changed_save_boundary_and_authority_rejected(self):
        mutations = [lambda f: f['session'].update(revision=5), lambda f: f.update(qualified=True),
                     lambda f: f.update(profile_sha256='0' * 64), lambda f: f.update(request_id='foreign'),
                     lambda f: f.update(owner_epoch=True), lambda f: f.update(http_manifest_sha256='bad')]
        for mutate in mutations:
            self.ledger = live_ledger()
            def reply(payload):
                frame = save_frame(payload, self.ledger)
                mutate(frame)
                return frame
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                P.save(Transport(reply), self.ledger, self.path, PROFILE)
            self.assertFalse(self.path.exists())
            self.assertTrue(self.ledger.poisoned)

    def test_bound_admission_rejection_preserves_ledger_for_chat_and_explicit_save_retry(self):
        state = P.snapshot(self.ledger)
        transport = Transport(lambda p: rejected_save_frame(p, self.ledger))
        with self.assertRaisesRegex(P.CheckpointAdmissionRejected, 'live session retained.*disk space'):
            P.save(transport, self.ledger, self.path, PROFILE)
        self.assertTrue(transport.closed)
        self.assertEqual(P.snapshot(self.ledger), state)
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertEqual(self.ledger.plan('Continue.')['path'], '/v1/session/append')
        P.save(Transport(lambda p: save_frame(p, self.ledger)), self.ledger, self.path, PROFILE)
        self.assertTrue(self.path.is_file())

    def test_unbound_malformed_or_uncertain_rejection_still_poisons(self):
        mutations = [lambda f: f.pop('schema'), lambda f: f.update(event='session_saved'),
            lambda f: f.update(before_submit=1), lambda f: f.update(capture_started=True),
            lambda f: f.update(session_preserved=False), lambda f: f.update(operation='restore'),
            lambda f: f.update(request_id='other'), lambda f: f.update(session_id='other'),
            lambda f: f.update(checkpoint_id='9'*64), lambda f: f.update(revision=True),
            lambda f: f.update(profile_sha256='0'*64), lambda f: f.update(owner_epoch=True),
            lambda f: f['session'].update(pending_token=0),
            lambda f: f['session'].update(pending_token_published=1),
            lambda f: f.update(qualified=True), lambda f: f.update(detail=''),
            lambda f: f.update(detail='x'*8193), lambda f: f.update(unrecognized_authority=True)]
        for mutate in mutations:
            self.ledger = live_ledger()
            def reply(payload):
                frame = rejected_save_frame(payload, self.ledger)
                mutate(frame)
                return frame
            with self.subTest(mutate=mutate), self.assertRaises(ValueError) as raised:
                P.save(Transport(reply), self.ledger, self.path, PROFILE)
            self.assertNotIsInstance(raised.exception, P.CheckpointAdmissionRejected)
            self.assertTrue(self.ledger.poisoned)
            self.assertEqual(list(self.root.iterdir()), [])

    def test_admission_rejection_requires_complete_single_frame_http_response(self):
        for extra, truncate in [(True, False), (False, True)]:
            self.ledger = live_ledger()
            with self.subTest(extra=extra, truncate=truncate), self.assertRaises(ValueError) as raised:
                P.save(Transport(lambda p: rejected_save_frame(p, self.ledger), extra=extra,
                    truncate=truncate), self.ledger, self.path, PROFILE)
            self.assertNotIsInstance(raised.exception, P.CheckpointAdmissionRejected)
            self.assertTrue(self.ledger.poisoned)
            self.assertEqual(list(self.root.iterdir()), [])

    def test_existing_local_file_rejects_before_gpu_request(self):
        self.path.write_text('existing', encoding='utf-8')
        transport = Transport(lambda _: self.fail('should not send'))
        with self.assertRaises(ValueError):
            P.save(transport, self.ledger, self.path, PROFILE)
        self.assertEqual(transport.calls, [])
        self.assertEqual(self.path.read_text(), 'existing')
        self.assertFalse(self.ledger.poisoned)

    def test_invalid_or_uncertain_ledger_rejects_before_saving(self):
        self.ledger.poisoned = True
        transport = Transport(lambda _: self.fail('should not send'))
        with self.assertRaises(ValueError):
            P.save(transport, self.ledger, self.path, PROFILE)
        self.assertEqual(transport.calls, [])

    def test_invalid_explicit_checkpoint_name_is_not_replaced_silently(self):
        for value in ('', '../outside', False, 'x' * 65):
            transport = Transport(lambda _: self.fail('should not send'))
            with self.subTest(value=value), self.assertRaises(ValueError):
                P.save(transport, self.ledger, self.path, PROFILE, value)
            self.assertEqual(transport.calls, [])

    def test_interrupt_during_control_poisons_and_removes_pending_client_file(self):
        def interrupt(_):
            raise KeyboardInterrupt
        transport = Transport(interrupt)
        with self.assertRaises(KeyboardInterrupt):
            P.save(transport, self.ledger, self.path, PROFILE)
        self.assertTrue(transport.closed)
        self.assertTrue(self.ledger.poisoned)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_file_publication_failure_preserves_recoverable_bundle_not_false_success(self):
        with patch.object(P.os, 'link', side_effect=OSError('disk error')):
            with self.assertRaises(P.LocalStateWriteError) as error:
                P.save(Transport(lambda p: save_frame(p, self.ledger)), self.ledger, self.path, PROFILE)
        P.validate_bundle(error.exception.bundle, self.ledger.template)
        self.assertFalse(self.ledger.poisoned)
        self.assertFalse(self.path.exists())
        self.assertEqual(list(self.root.iterdir()), [])
        result = P.publish_bundle(self.path, error.exception.bundle, self.ledger.template)
        P.read_bundle(self.path, self.ledger.template, result['file_sha256'])

    def test_client_size_rejection_precedes_gpu_capture(self):
        transport = Transport(lambda _: self.fail('should not send'))
        with patch.object(P, 'MAX_BYTES', 128), self.assertRaisesRegex(ValueError, 'before native capture'):
            P.save(transport, self.ledger, self.path, PROFILE)
        self.assertEqual(transport.calls, [])
        self.assertEqual(list(self.root.iterdir()), [])

    def test_atomic_install_does_not_overwrite_a_racing_file(self):
        link = P.os.link
        def raced(source, destination):
            self.path.write_bytes(b'existing-racer')
            return link(source, destination)
        with patch.object(P.os, 'link', side_effect=raced), self.assertRaises(P.LocalStateWriteError):
            P.save(Transport(lambda p: save_frame(p, self.ledger)), self.ledger, self.path, PROFILE)
        self.assertEqual(self.path.read_bytes(), b'existing-racer')
        self.assertEqual(list(self.root.iterdir()), [self.path])
        self.assertFalse(self.ledger.poisoned)

    def tcp(self, truncate=False):
        ledger = self.ledger
        bundle = []
        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'
            def log_message(self, *args):
                pass
            def do_GET(self):
                raw = b'{"profile":"resident"}'
                self.send_response(200)
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if not C.digest_string(payload.get('checkpoint_id')):
                    self.send_error(400, 'native checkpoint ID grammar')
                    return
                if self.path.endswith('/save'):
                    frame = save_frame(payload, ledger)
                    bundle[:] = [P.make_bundle(P.snapshot(ledger), ledger.template, frame)]
                else:
                    frame = restore_frame(payload, bundle[0])
                raw = P.canonical(frame) + b'\n'
                self.send_response(200)
                self.send_header('Content-Type', 'application/x-ndjson')
                self.send_header('Transfer-Encoding', 'chunked')
                self.send_header('Connection', 'close')
                self.end_headers()
                for part in (raw[:19], raw[19:]):
                    self.wfile.write(f'{len(part):X}\r\n'.encode() + part + b'\r\n')
                self.wfile.write(b'0\r\n' if truncate else b'0\r\n\r\n')
                self.wfile.flush()
                self.close_connection = True
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return C.Transport(f'http://127.0.0.1:{server.server_port}', timeout=3)

    def test_actual_tcp_save_restore_complete_framing(self):
        transport = self.tcp()
        result = P.save(transport, self.ledger, self.path, PROFILE)
        value = P.read_bundle(self.path, self.ledger.template, result['file_sha256'])
        restored, frame = P.restore(transport, value, self.ledger.template, PROFILE)
        self.assertEqual(restored.history, self.ledger.history)
        self.assertEqual(frame['prefix_recompute_tokens'], 0)

    def test_actual_tcp_missing_final_crlf_is_not_a_saved_client_state(self):
        with self.assertRaises(ValueError):
            P.save(self.tcp(truncate=True), self.ledger, self.path, PROFILE)
        self.assertTrue(self.ledger.poisoned)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_external_checksum_and_payload_checksum(self):
        _, bundle = self.saved()
        with self.assertRaisesRegex(ValueError, 'external'):
            P.read_bundle(self.path, self.ledger.template, '0' * 64)
        bundle['state']['history'][0] += 1
        with self.assertRaisesRegex(ValueError, 'checksum'):
            P.validate_bundle(bundle, self.ledger.template)

    def test_duplicate_json_keys_or_unknown_root_reject(self):
        self.path.write_text('{"schema":"a","schema":"b"}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            P.read_bundle(self.path, self.ledger.template)
        self.path.unlink()
        _, bundle = self.saved()
        bundle['unrecognized_authority'] = True
        with self.assertRaisesRegex(ValueError, 'schema'):
            P.validate_bundle(bundle, self.ledger.template)

    def test_template_mismatch_and_corrupt_exact_history_are_rejected(self):
        _, bundle = self.saved()
        template = copy.copy(self.ledger.template)
        template.identity = {'tokenizer_sha256': '0' * 64}
        with self.assertRaisesRegex(ValueError, 'identity'):
            P.validate_bundle(bundle, template)
        for change in (lambda s: s['history'].__setitem__(0, 99),
                       lambda s: s.update(prefix='different'),
                       lambda s: s['terminal'].update(event='incomplete'),
                       lambda s: s['terminal']['session'].update(pending_token=True)):
            value = copy.deepcopy(bundle)
            change(value['state'])
            value['content_sha256'] = P.sha256(P.canonical({k:v for k,v in value.items() if k != 'content_sha256'}))
            with self.subTest(change=change), self.assertRaises(ValueError):
                P.validate_bundle(value, self.ledger.template)

    def test_restore_boundaries_and_fresh_authority(self):
        _, bundle = self.saved()
        mutations = [lambda f: f.update(owner_id=bundle['checkpoint']['owner_id']),
            lambda f: f.update(source_session_id='another-session'),
            lambda f: f.update(native_manifest_sha256='0' * 64),
            lambda f: f.update(prefix_recompute_tokens=1),
            lambda f: f.update(target_seal_verified=False),
            lambda f: f['session'].update(consumed_input_digest='0' * 64),
            lambda f: f['native_restore'].update(old_cuda_authority_restored=True),
            lambda f: f['native_restore'].update(prefix_recompute_tokens=1)]
        original = P.canonical(bundle)
        for mutate in mutations:
            def reply(payload):
                frame = restore_frame(payload, bundle)
                mutate(frame)
                return frame
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                P.restore(Transport(reply), bundle, self.ledger.template, PROFILE)
            self.assertEqual(P.canonical(bundle), original)

    def test_restore_profile_rejected_before_request(self):
        _, bundle = self.saved()
        transport = Transport(lambda _: self.fail('should not send'), profile='growing')
        with self.assertRaisesRegex(ValueError, 'profile'):
            P.restore(transport, bundle, self.ledger.template, PROFILE)
        self.assertEqual(transport.calls, [])

    def test_nested_symlink_path_rejected(self):
        actual, alias = self.root / 'actual', self.root / 'alias'
        actual.mkdir()
        try:
            alias.symlink_to(actual, target_is_directory=True)
        except OSError as error:
            self.skipTest(str(error))
        with self.assertRaisesRegex(ValueError, 'reparse'):
            P.save(Transport(lambda _: self.fail('should not send')), self.ledger, alias / 'state.json', PROFILE)


if __name__ == '__main__':
    unittest.main()
