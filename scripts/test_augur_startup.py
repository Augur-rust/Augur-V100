"""Startup failure/lease regressions; never load a model or stop a real engine."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid
import augur_startup as S


class Diagnostics(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.language = patch.object(S.C, 'LANG', 'en')
        self.language.start(); self.addCleanup(self.language.stop)

    def write(self, path, text):
        p = self.root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding='utf8')

    def test_native_oom_overrides_generic_traceback_without_echoing_logs(self):
        self.write('launcher.stderr.log', 'ProtocolError: terminal identity/claims; PRIVATE_KEY')
        self.write('launch/native/owner-0/stderr.log',
            'Error: CudaRuntimeError { operation: "cuda_device_alloc", raw_cuda_status: 2 }')
        detail = S.failure_detail(self.root, 1)
        self.assertIn('GPU memory', detail)
        self.assertNotIn('PRIVATE_KEY', detail)
        self.assertNotIn('identity/claims', detail)

    def test_duplicate_guard_message_is_actionable_and_private(self):
        self.write('launcher.stderr.log', 'augur_engine_already_running: PRIVATE_COMMAND')
        detail = S.failure_detail(self.root, 1)
        self.assertIn('another port/version', detail)
        self.assertNotIn('PRIVATE_COMMAND', detail)

    def test_missing_or_unrecognized_logs_keep_exit_code(self):
        self.assertIn('exit code 17', S.failure_detail(self.root, 17))
        self.write('launcher.stderr.log', 'PRIVATE_KEY not a known CUDA error')
        self.assertIn('exit code 17', S.failure_detail(self.root, 17))
        self.assertNotIn('PRIVATE_KEY', S.failure_detail(self.root, 17))
        self.write('launch/native/owner-0/stderr.log',
            'operation: "cuda_device_alloc", raw_cuda_status: 20 }')
        self.assertNotIn('GPU memory', S.failure_detail(self.root, 17))

    def test_log_tail_is_bounded(self):
        self.write('large.log', 'a' * 32768 + 'final error')
        self.assertEqual(S.tail(self.root / 'large.log', 11), 'final error')

    def test_retry_requires_retirement_of_every_failed_native_owner(self):
        self.assertTrue(S.failed_start_retired(self.root, {}))
        (self.root / 'launch').mkdir()
        self.assertFalse(S.failed_start_retired(self.root, {}))
        good = dict(confirmed_exit=True, exit_code=0, forced=False, cleanup_errors=[],
            native_shutdown=dict(no_owner=True), native_retirements=[dict(confirmed_exit=True,
            reason='startup_failure', force_attempted=False)])
        self.assertTrue(S.failed_start_retired(self.root, good))
        for name, value in [('confirmed_exit', False), ('forced', True), ('exit_code', 1),
                ('cleanup_errors', ['error']), ('native_shutdown', None), ('native_retirements', [])]:
            with self.subTest(field=name):
                self.assertFalse(S.failed_start_retired(self.root, dict(good, **{name:value})))
        bad = copy.deepcopy(good)
        bad['native_retirements'].append(dict(confirmed_exit=False, reason='startup_failure', force_attempted=False))
        self.assertFalse(S.failed_start_retired(self.root, bad))
        bad = copy.deepcopy(good)
        bad['native_retirements'][0]['reason'] = 'request_failure'
        self.assertFalse(S.failed_start_retired(self.root, bad))
        self.write('launch/ready.json', '{}')
        self.assertFalse(S.failed_start_retired(self.root, good))


@unittest.skipUnless(os.name == 'nt', 'Windows named mutex')
class LeaseTests(unittest.TestCase):
    def spawn_holder(self, name):
        code = """import sys
from unittest.mock import patch
import augur_startup as S
with patch.object(S, 'native_owners', return_value=[]), S.EngineLease(sys.argv[1]):
    print('ready', flush=True)
    sys.stdin.readline()
"""
        p = subprocess.Popen([sys.executable, '-X', 'utf8', '-B', '-c', code, name],
            cwd=Path(S.__file__).parent, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
        def finish():
            if not p.stdin.closed: p.stdin.close()
            try: p.wait(timeout=10)
            finally: p.stdout.close(); p.stderr.close()
        self.addCleanup(finish)
        self.assertEqual(p.stdout.readline().strip(), b'ready')
        return p

    def test_other_process_is_rejected_and_can_retry_after_exit(self):
        name = r'Local\Augur.Test.' + uuid.uuid4().hex
        p = self.spawn_holder(name)
        with patch.object(S, 'native_owners', return_value=[]):
            with self.assertRaisesRegex(RuntimeError, 'augur_engine_already_running'):
                with S.EngineLease(name): pass
            p.stdin.close(); p.wait(timeout=10)
            with S.EngineLease(name): pass

    def test_legacy_owner_rejection_releases_lease(self):
        name = r'Local\Augur.Test.' + uuid.uuid4().hex
        with patch.object(S, 'native_owners', return_value=[123]):
            with self.assertRaisesRegex(RuntimeError, 'native engine PID 123'):
                with S.EngineLease(name): pass
        with patch.object(S, 'native_owners', return_value=[]), S.EngineLease(name): pass

    def test_real_process_snapshot_is_read_only(self):
        result = S.native_owners()
        self.assertTrue(all(type(pid) is int and pid > 0 for pid in result))
        self.assertEqual(result, sorted(set(result)))


if __name__ == '__main__': unittest.main()
