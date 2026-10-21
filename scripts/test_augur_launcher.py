"""CPU tests for local management, configuration and owner lifecycle (no GPU)."""
import copy
import http.client
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
import augur_launcher as L
import augur_launch_config as C


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.value = C.defaults(self.root)

    def test_six_physical_context_choices(self):
        for n in C.CONTEXT_CHOICES:
            value = dict(self.value, context_tokens=n, default_output_tokens=n * 95 // 100)
            self.assertEqual(C.normalize(value, self.root)['context_tokens'], n)

    def test_invalid_contexts(self):
        for n in (True, 0, 8193, 16384, '8192', 8192.0, 262145):
            with self.assertRaises(ValueError):
                C.normalize(dict(self.value, context_tokens=n), self.root)

    def test_output_boundaries(self):
        for n in C.CONTEXT_CHOICES:
            for limit in (0, True, n * 95 // 100 + 1):
                with self.assertRaises(ValueError):
                    C.normalize(dict(self.value, context_tokens=n, default_output_tokens=limit), self.root)

    def test_unknown_sampling_or_shell_fields_rejected(self):
        for name in ('temperature', 'top_p', 'top_k', 'command', 'environment'):
            with self.assertRaises(ValueError):
                C.normalize(dict(self.value, **{name:0}), self.root)

    def test_bad_paths_and_port(self):
        for p in ('relative/path', 'C:\\line\npath', '\\\\remote\\weights'):
            with self.assertRaises(ValueError):
                C.normalize(dict(self.value, model_dir=p), self.root)
        for port in (0, 1023, 65536, True):
            with self.assertRaises(ValueError):
                C.normalize(dict(self.value, port=port), self.root)

    def test_atomic_save_and_defaults(self):
        destination = self.root / 'private/config.json'
        C.atomic_json(destination, self.value)
        changed = dict(self.value, context_tokens=8192)
        C.atomic_json(destination, changed)
        self.assertEqual(json.loads(destination.read_bytes()), changed)
        self.assertFalse(list(destination.parent.glob('*.tmp')))
        self.assertIn('user-data', C.defaults(self.root)['quant_output_dir'])

    def test_missing_models_fail_before_launch(self):
        for function in (C.target_preflight, C.draft_environment, C.quantization_source):
            with self.assertRaises(ValueError):
                function(self.value)

    def test_runtime_manifest_is_distinct_from_modelcard(self):
        self.assertEqual(json.loads(C.MANIFEST.read_bytes())['schema'], 'quasar-session-model-manifest-v1')
        self.assertEqual(json.loads((C.MODEL_CARD/'artifact-manifest.json').read_bytes())['role'], 'documentation_only_not_runtime_admission')

    def test_output_limit_reaches_python_owner(self):
        import augur_api as Q
        original = (Q.CAPACITY, Q.DEFAULT_OUTPUT)
        try:
            for n in C.CONTEXT_CHOICES:
                Q.configure_context(n, 128)
                self.assertEqual((Q.CAPACITY, Q.MAX_OUTPUT, Q.DEFAULT_OUTPUT), (n, n*95//100, 128))
            for value in (True, 16384, '8192'):
                with self.assertRaises(ValueError):
                    Q.configure_context(value)
        finally:
            Q.configure_context(*original)


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manager = L.Manager(self.root/'package', self.root/'private', verify=False)
        self.env = self.fake_environment()  # never touch the real HKCU environment

    def test_invalid_saved_config_preserved(self):
        self.manager.config_path.write_text('{broken', encoding='utf8')
        new = L.Manager(self.root/'package', self.root/'private', verify=False)
        self.assertIsNotNone(new.error)
        self.assertEqual(new.config_path.read_text(), '{broken')

    def test_invalid_candidate_does_not_stop_owner(self):
        with patch.object(self.manager, '_preflight', side_effect=ValueError('bad model')), patch.object(self.manager, '_stop_engine') as stop:
            self.manager._work('restart', self.manager.config)
            stop.assert_not_called()
            self.assertEqual(self.manager.error, 'bad model')

    def test_stop_closes_only_owned_pipe_and_requires_cleanup(self):
        process = Mock()
        process.poll.return_value = None
        process.returncode = 0
        process.stdin = io.BytesIO()
        self.manager.process = process
        self.manager.run_dir = self.root / 'run'
        C.atomic_json(self.manager.run_dir/'launch/stopped.json', dict(cleanup_verified=True))
        self.manager._stop_engine()
        self.assertTrue(process.stdin.closed)
        process.wait.assert_called_once_with(timeout=300)
        self.assertIsNone(self.manager.process)
        self.assertTrue(self.manager.last_cleanup['cleanup_verified'])

    def test_unconfirmed_shutdown_never_starts_replacement(self):
        with patch.object(self.manager, '_preflight', return_value=({},{})), patch.object(self.manager, '_stop_engine', side_effect=RuntimeError('not drained')), patch.object(self.manager, '_start_engine') as start:
            self.manager._work('restart', self.manager.config)
            start.assert_not_called()
            self.assertFalse(self.manager.config_path.exists())

    def test_shutdown_timeout_retains_owner(self):
        process = Mock()
        process.poll.return_value = None
        process.stdin = io.BytesIO()
        process.wait.side_effect = subprocess.TimeoutExpired('owned engine', 300)
        self.manager.process = process
        with self.assertRaises(subprocess.TimeoutExpired):
            self.manager._stop_engine()
        self.assertIs(self.manager.process, process)

    def test_empty_startup_stdout_reports_oom_and_retires_failed_start(self):
        process = Mock()
        process.stdout = io.BytesIO()
        process.stdin = io.BytesIO()
        process.wait.return_value = process.poll.return_value = process.returncode = 1
        config = dict(self.manager.config, cache_enabled=False)
        admission = dict(manifest='manifest.json', manifest_sha256='a' * 64)
        def spawn(*args, **kwargs):
            run = self.manager.run_dir
            owner = run / 'launch/native/owner-0'
            owner.mkdir(parents=True)
            (owner/'stderr.log').write_text('operation: "cuda_device_alloc", raw_cuda_status: 2 }')
            C.atomic_json(run/'launch/stopped.json', dict(exit_code=0, confirmed_exit=True,
                forced=False, cleanup_errors=[], cleanup_verified=False,
                native_shutdown=dict(no_owner=True), native_retirements=[dict(
                    confirmed_exit=True, reason='startup_failure', force_attempted=False)]))
            return process
        with patch.object(L.subprocess, 'Popen', side_effect=spawn), patch.object(C, 'LANG', 'en'):
            with self.assertRaisesRegex(RuntimeError, 'GPU memory'):
                self.manager._start_engine(config, admission, {})
        self.assertEqual(self.manager.phase, 'startup_failed')
        self.assertTrue(process.stdin.closed)
        self.manager._stop_engine()
        self.assertIsNone(self.manager.process)
        self.assertFalse(self.manager.last_cleanup['cleanup_verified'])

    def test_invalid_startup_receipt_never_exposes_raw_output(self):
        for raw in (b'PRIVATE_KEY not json', b'[]', b'null'):
            with self.subTest(raw=raw):
                process = Mock()
                process.stdout, process.stdin = io.BytesIO(raw), io.BytesIO()
                with patch.object(L.subprocess, 'Popen', return_value=process), patch.object(C, 'LANG', 'en'):
                    with self.assertRaisesRegex(RuntimeError, 'invalid startup receipt') as caught:
                        self.manager._start_engine(dict(self.manager.config, cache_enabled=False),
                            dict(manifest='manifest.json', manifest_sha256='a' * 64), {})
                self.assertNotIn('PRIVATE_KEY', str(caught.exception))
                self.assertIsNone(self.manager.ready)
                process.stdout.close()

    def test_child_reuses_current_bridge_and_explicit_native_runtime(self):
        process = Mock()
        process.stdin = io.BytesIO()
        process.stdout = io.BytesIO(json.dumps(dict(event='augur_product_ready',
            context_tokens=self.manager.config['context_tokens'])).encode() + b'\n')
        self.addCleanup(process.stdin.close)
        self.addCleanup(process.stdout.close)
        with patch.object(self.manager, 'sync_api_key'), patch.object(self.manager, '_start_monitor'), \
                patch.object(L.subprocess, 'Popen', return_value=process) as popen:
            self.manager._start_engine(dict(self.manager.config, cache_enabled=False),
                dict(manifest='manifest.json', manifest_sha256='a' * 64), {})
        args = popen.call_args.args[0]
        self.assertEqual(Path(args[5]), Path(L.__file__).with_name('augur_portable.py'))
        self.assertEqual(args[6:9], ['engine', '--runtime-root', str(self.manager.package)])
        self.assertIn('--stop-on-stdin', args)

    def test_one_operation_at_a_time(self):
        self.manager.operation = 'quantize'
        with self.assertRaises(RuntimeError):
            self.manager.submit('stop', {})

    def test_saved_quantization_settings_do_not_hide_occupied_port(self):
        self.manager.process = Mock()
        self.manager.process.poll.return_value = None
        self.manager.active_config = dict(self.manager.config, port=8294)
        self.manager.config = dict(self.manager.config, port=8297)
        candidate = dict(self.manager.config)
        with patch.object(C, 'target_preflight', return_value={}), patch.object(C, 'draft_environment', return_value={}), patch.object(L.socket, 'socket') as socket:
            socket.return_value.__enter__.return_value.bind.side_effect = OSError('port occupied')
            with self.assertRaises(OSError):
                self.manager._preflight(candidate)

    def test_monitor_tracks_this_run_and_survives_engine_stop(self):
        self.manager.run_dir = self.root / 'specific-run'
        self.manager.run_dir.mkdir()
        import augur_monitor as M
        with patch.object(M.Sampler, 'loop'):
            self.manager._start_monitor(dict(self.manager.config, port=8301))
            server, sampler, _, _ = self.manager.monitor
            self.assertEqual(sampler.runs_root, self.manager.run_dir)
            self.assertEqual(sampler.engine_port, 8301)
            self.assertEqual(self.manager.monitor_url, 'http://127.0.0.1:' + str(server.server_port) + '/')
            connection = http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=3)
            connection.request('GET','/api/metrics')
            response = connection.getresponse()
            self.assertEqual(response.status,200)
            self.assertEqual(json.loads(response.read())['status'],'starting')
            connection.close()
            self.manager._stop_engine()
            self.assertIsNotNone(self.manager.monitor_url)
            self.manager._close_monitor()
            self.assertIsNone(self.manager.monitor)
            self.assertIsNone(self.manager.monitor_url)

    def test_operation_clock_survives_polling_and_freezes_after_success(self):
        entered, release = threading.Event(), threading.Event()
        def preflight(candidate):
            self.manager.stage('validating', 'checking')
            entered.set()
            release.wait(3)
            return {}, {}
        with patch.object(self.manager, '_preflight', side_effect=preflight), patch.object(L.time, 'monotonic', return_value=100) as clock:
            self.manager.submit('validate', self.manager.config)
            self.assertTrue(entered.wait(2))
            clock.return_value = 107.25
            state = self.manager.state()
            self.assertEqual(state['progress']['elapsed_s'], 7.25)
            self.assertEqual(state['progress']['stage'], 'validating')
            self.assertEqual(state['progress']['status'], 'running')
            release.set()
            self.manager.worker.join(3)
            clock.return_value = 200
            state = self.manager.state()
            self.assertIsNone(state['operation'])
            self.assertEqual(state['progress']['status'], 'complete')
            self.assertEqual(state['progress']['elapsed_s'], 7.25)

    def test_failed_operation_exposes_error_and_final_elapsed_time(self):
        with patch.object(self.manager, '_preflight', side_effect=ValueError('invalid model')):
            self.manager.submit('validate', self.manager.config)
            self.manager.worker.join(3)
        state = self.manager.state()
        self.assertEqual(state['progress']['status'], 'failed')
        self.assertEqual(state['error'], 'invalid model')
        self.assertGreaterEqual(state['progress']['elapsed_s'], 0)
        self.assertIsNone(state['operation'])

    def test_exited_engine_never_advertises_a_chat_ready_receipt(self):
        self.manager.process = Mock()
        self.manager.process.poll.return_value = 1
        self.manager.phase = 'running'
        self.manager.ready = dict(browser_url='http://127.0.0.1:8294/#key=old')
        state = self.manager.state()
        self.assertFalse(state['running'])
        self.assertIsNone(state['ready'])
        self.assertEqual(state['phase'], 'service_exited')
        self.manager.process = None

    def test_monitor_url_survives_restart_and_persists_across_launcher_start(self):
        import augur_monitor as M
        with patch.object(M.Sampler, 'loop'):
            self.manager.run_dir = self.root / 'first-run'
            self.manager._start_monitor(dict(self.manager.config, port=8301))
            original = self.manager.monitor
            url = self.manager.monitor_url
            self.manager.run_dir = self.root / 'second-run'
            self.manager._start_monitor(dict(self.manager.config, port=8302))
            self.assertEqual(self.manager.monitor_url, url)
            self.assertIs(self.manager.monitor[0], original[0])
            self.assertTrue(original[1].closed.is_set())
            self.assertEqual(self.manager.monitor[1].runs_root, self.root / 'second-run')
            self.assertEqual(self.manager.monitor[1].engine_port, 8302)
            self.manager.monitor[1].state = dict(status='new_run')
            connection = http.client.HTTPConnection('127.0.0.1', original[0].server_port, timeout=3)
            connection.request('GET', '/api/metrics')
            self.assertEqual(json.loads(connection.getresponse().read())['status'], 'new_run')
            connection.close()
            self.manager._close_monitor()
            self.manager._start_monitor(self.manager.config)
            self.assertEqual(self.manager.monitor_url, url)
            self.manager._close_monitor()

    def test_occupied_saved_monitor_port_uses_available_port(self):
        import augur_monitor as M
        with L.socket.socket() as occupied, patch.object(M.Sampler, 'loop'):
            occupied.bind(('127.0.0.1', 0))
            occupied.listen()
            C.atomic_json(self.manager.data / 'monitor-port.json', dict(port=occupied.getsockname()[1]))
            self.manager.run_dir = self.root / 'run'
            self.manager._start_monitor(self.manager.config)
            self.assertNotEqual(self.manager.monitor[0].server_port, occupied.getsockname()[1])
            self.manager._close_monitor()

    def test_api_key_is_stable_across_managers_and_not_in_public_state(self):
        key_path = self.manager.api_key_file()
        key = key_path.read_text(encoding='ascii').strip()
        self.assertGreaterEqual(len(key), 32)
        other = L.Manager(self.manager.package, self.manager.data, verify=False)
        self.assertEqual(other.api_key_file().read_text(), key_path.read_text())
        self.assertNotIn(key, json.dumps(other.state()))

    def test_invalid_saved_key_is_preserved_and_rejected_before_stop(self):
        path = self.manager.data / 'api-key.txt'
        path.write_text('bad key')
        with patch.object(self.manager, '_stop_engine') as stop:
            self.manager._work('restart', self.manager.config)
            stop.assert_not_called()
        self.assertIn('api-key.txt', self.manager.error)
        self.assertEqual(path.read_text(), 'bad key')

    def test_autostart_only_with_saved_readable_configuration(self):
        with patch.object(self.manager, 'submit') as submit:
            self.assertFalse(L.autostart(self.manager, True))  # first run: wait for the page
            C.atomic_json(self.manager.config_path, self.manager.config)
            self.assertFalse(L.autostart(self.manager, False))
            submit.assert_not_called()
            self.assertTrue(L.autostart(self.manager, True))
            submit.assert_called_once_with('restart', self.manager.config)
            self.manager.error = 'unreadable'
            self.assertFalse(L.autostart(self.manager, True))
            self.assertEqual(submit.call_count, 1)

    def test_autostart_rejection_is_shown_not_raised(self):
        C.atomic_json(self.manager.config_path, self.manager.config)
        with patch.object(self.manager, 'submit', side_effect=ValueError('missing model')):
            self.assertFalse(L.autostart(self.manager, True))
        self.assertEqual(self.manager.error, 'missing model')

    def fake_environment(self, initial=None):
        store = dict(initial or {})
        fake = Mock()
        fake.read.side_effect = store.get
        fake.write.side_effect = lambda values: store.update(values) or True
        fake.remove.side_effect = lambda values: [store.pop(n, None) for n in values]
        patcher = patch.object(L, 'E', fake)
        patcher.start(); self.addCleanup(patcher.stop)
        clean = patch.dict(os.environ, {}, clear=False); clean.start(); self.addCleanup(clean.stop)
        os.environ.pop('AUGUR_API_KEY', None)
        return store

    def test_key_is_exported_to_user_environment_by_default(self):
        store = self.env
        key = self.manager.sync_api_key()
        self.assertEqual(store['AUGUR_API_KEY'], key)
        self.assertEqual(store['AUGUR_BASE_URL'], 'http://127.0.0.1:8294/v1')
        self.assertEqual(store['AUGUR_MODEL'], 'quasar-27b')
        self.assertEqual((self.manager.data/'api-key.txt').read_text().strip(), key)
        self.assertNotIn(key, (self.manager.data/'env-export.json').read_text())
        self.assertTrue(self.manager.connection()['env_exported'])
        self.assertNotIn(key, json.dumps(self.manager.state()))

    def test_user_set_variable_wins_but_stale_export_does_not(self):
        mine = 'u' * 40
        store = self.env; store['AUGUR_API_KEY'] = mine
        self.assertEqual(self.manager.sync_api_key(), mine)
        self.assertEqual((self.manager.data/'api-key.txt').read_text().strip(), mine)
        with patch.object(self.manager, '_preflight'), patch.object(self.manager, '_stop_engine'):
            self.manager._rekey()
        fresh = (self.manager.data/'api-key.txt').read_text().strip()
        self.assertNotEqual(fresh, mine)
        self.assertEqual(store['AUGUR_API_KEY'], fresh)
        self.assertEqual(self.manager.sync_api_key(), fresh)  # exported value is not re-adopted

    def test_invalid_user_variable_is_ignored(self):
        self.env['AUGUR_API_KEY'] = 'short'
        self.assertNotEqual(self.manager.sync_api_key(), 'short')

    def test_switching_export_off_retracts_only_our_values(self):
        store = self.env
        self.manager.sync_api_key()
        self.manager.config = dict(self.manager.config, export_env=False)
        self.manager.sync_api_key()
        self.assertNotIn('AUGUR_API_KEY', store)
        self.assertFalse(self.manager.connection()['env_exported'])
        store['AUGUR_API_KEY'] = 'v' * 40  # set by the user afterwards: left alone
        self.manager.sync_api_key()
        self.assertEqual(store['AUGUR_API_KEY'], 'v' * 40)

    def test_rekey_restarts_a_running_engine_with_the_same_config(self):
        process = Mock(); process.poll.return_value = None
        self.manager.process, self.manager.active_config = process, dict(self.manager.config)
        with patch.object(self.manager, '_preflight', return_value=('a', 'd')), \
             patch.object(self.manager, '_stop_engine') as stop, patch.object(self.manager, '_start_engine') as start:
            self.manager._rekey()
        stop.assert_called_once()
        start.assert_called_once_with(self.manager.config, 'a', 'd')
        self.manager.process = None

    def test_saved_config_without_export_switch_defaults_on(self):
        legacy = {k: v for k, v in self.manager.config.items() if k != 'export_env'}
        self.assertTrue(C.normalize(legacy, self.manager.package)['export_env'])

    def test_first_run_defaults_to_chinese_and_asks(self):
        state = self.manager.state()
        self.assertEqual(state['language'], 'zh-CN')
        self.assertFalse(state['language_chosen'])
        self.assertEqual(C.L('中文', 'English'), '中文')
        # The language choice must never look like a saved engine configuration.
        C.save_language(self.manager.data, 'en')
        self.addCleanup(setattr, C, 'LANG', 'zh-CN')
        self.assertFalse(self.manager.config_path.exists())
        self.assertFalse(L.autostart(self.manager, True))
        again = L.Manager(self.manager.package, self.manager.data, verify=False)
        self.assertEqual(again.state()['language'], 'en')
        self.assertTrue(again.state()['language_chosen'])
        self.assertIn('Choose', again.message)

    def test_unreadable_preferences_fall_back_to_chinese(self):
        (self.manager.data / 'preferences.json').write_text('{bad', encoding='utf8')
        self.assertEqual(C.load_preferences(self.manager.data), dict(language='zh-CN', chosen=False))

    def test_idle_stop_keeps_configuration(self):
        previous = copy.deepcopy(self.manager.config)
        self.manager._stop_engine()
        self.assertEqual(self.manager.config, previous)
        self.assertEqual(self.manager.phase, 'stopped')

    def test_quantization_reuses_only_verified_output(self):
        output = self.root/'existing';output.mkdir()
        candidate = dict(self.manager.config, quant_output_dir=str(output))
        with patch.object(C, 'quantization_source', return_value='a'*64), patch.object(C, 'draft_environment', return_value={}) as check, patch.object(subprocess, 'run') as run:
            self.manager._quantize(candidate)
            run.assert_not_called()
            check.assert_called_once()
            self.assertEqual(self.manager.config['draft_quantized_dir'], str(output))

    def test_quantization_does_not_overwrite_invalid_existing_output(self):
        output = self.root/'existing';output.mkdir();(output/'mine').write_text('keep')
        candidate = dict(self.manager.config, quant_output_dir=str(output))
        with patch.object(C, 'quantization_source', return_value='a'*64), patch.object(C, 'draft_environment', side_effect=ValueError('invalid')), patch.object(subprocess, 'run') as run:
            with self.assertRaises(ValueError): self.manager._quantize(candidate)
            run.assert_not_called()
            self.assertEqual((output/'mine').read_text(), 'keep')


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.manager = Mock()
        self.manager.state.return_value = dict(running=False)
        self.manager.submit.return_value = dict(accepted=True)
        self.server = L.Server(('127.0.0.1',0), self.manager, 'private-token')
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True);self.thread.start()
        self.addCleanup(self.shutdown)

    def shutdown(self):
        self.server.shutdown();self.server.server_close();self.thread.join()

    def request(self, path, value=None, **headers):
        h={'Authorization':'Bearer private-token'};h.update(headers)
        connection = http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=3)
        try:
            if value is None: connection.request('GET',path,headers=h)
            else:
                h.setdefault('Content-Type','application/json')
                connection.request('POST',path,json.dumps(value),h)
            response=connection.getresponse();return response.status,response.read()
        finally: connection.close()

    def test_auth_on_state_and_stop(self):
        self.assertEqual(self.request('/api/state')[0],200)
        self.assertEqual(self.request('/api/state',Authorization='Bearer wrong')[0],403)
        self.assertEqual(self.request('/api/stop',{},Authorization='')[0],403)
        self.manager.submit.assert_not_called()

    def test_cross_origin_and_dns_rebinding(self):
        for headers in ({'Origin':'https://evil.invalid'},{'Host':'evil.invalid'}):
            self.assertEqual(self.request('/api/stop',{},**headers)[0],403)
        self.manager.submit.assert_not_called()

    def test_stop_has_fixed_operation(self):
        self.assertEqual(self.request('/api/stop',{})[0],202)
        self.manager.submit.assert_called_once_with('stop',{})
        self.assertEqual(self.request('/api/stop',{'pid':4})[0],400)

    def test_bound_body_and_type(self):
        self.assertEqual(self.request('/api/save-restart',[],**{'Content-Type':'application/json'})[0],400)
        self.assertEqual(self.request('/api/stop',{},**{'Content-Type':'text/plain'})[0],400)
        self.assertEqual(self.request('/api/stop',{'data':'x'*70000})[0],400)

    def test_static_assets_and_no_traversal(self):
        self.assertEqual(self.request('/')[0],200)
        for route in ('/open/chat', '/open/monitor', '/service.js'):
            code, data = self.request(route, Authorization='')
            self.assertEqual(code, 200)
            self.assertNotIn(b'private-token', data)
        self.assertEqual(self.request('/../config.json')[0],404)
        self.assertNotIn(b'private-token', self.request('/')[1])

    def test_quit_requires_auth_and_stops_serving(self):
        self.manager.operation = None
        self.assertEqual(self.request('/api/quit',{},Authorization='')[0],403)
        self.assertEqual(self.request('/api/quit',{'now':1})[0],400)
        self.assertFalse(self.server.quitting.is_set())
        self.assertEqual(self.request('/api/quit',{})[0],202)
        self.thread.join(3)
        self.assertFalse(self.thread.is_alive())

    def test_quit_waits_for_running_operation(self):
        self.manager.operation = 'quantize'
        self.assertEqual(self.request('/api/quit',{})[0],409)
        self.assertFalse(self.server.quitting.is_set())

    def test_connection_requires_setup_bearer(self):
        self.manager.connection.return_value = dict(api_key='k' * 40)
        self.assertEqual(self.request('/api/connection', Authorization='')[0], 403)
        code, data = self.request('/api/connection')
        self.assertEqual(code, 200)
        self.assertIn(b'k' * 40, data)
        self.assertEqual(self.request('/api/rekey', {'key': 'x'})[0], 400)
        self.assertEqual(self.request('/api/rekey', {})[0], 202)
        self.manager.submit.assert_called_once_with('rekey', {})

    def test_language_preference_is_saved_and_validated(self):
        with tempfile.TemporaryDirectory() as data:
            self.manager.data = Path(data)
            self.manager.lock = threading.RLock()
            self.addCleanup(setattr, C, 'LANG', 'zh-CN')
            self.assertEqual(self.request('/api/language', {'language': 'fr'})[0], 400)
            self.assertEqual(self.request('/api/language', {'language': 'en', 'x': 1})[0], 400)
            code, data_bytes = self.request('/api/language', {'language': 'en'})
            self.assertEqual(code, 200)
            self.assertEqual(json.loads(data_bytes), dict(language='en', chosen=True))
            self.assertEqual(C.load_preferences(data), dict(language='en', chosen=True))
            self.assertEqual(C.L('中文', 'English'), 'English')

    def test_busy_is_conflict(self):
        self.manager.submit.side_effect=RuntimeError('busy')
        self.assertEqual(self.request('/api/stop',{})[0],409)


if __name__ == '__main__': unittest.main()
