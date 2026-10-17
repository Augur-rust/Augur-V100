"""Local UI boundary tests; not real-GPU or model capability evidence."""
import http.client
import json
from pathlib import Path
import signal
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import augur_product as P
import qwen35_chat as C


class CacheRestorePolicy(unittest.TestCase):
    def test_default_eight_and_explicit_overrides_do_not_mutate_environment(self):
        environment = {}
        self.assertEqual(P.cache_restore_policy(environment), (8, False, 'product_default'))
        self.assertEqual(environment, {})
        for workers in ('1', '2', '4', '8'):
            self.assertEqual(P.cache_restore_policy({'AUGUR_CACHE_RESTORE_WORKERS': workers}),
                             (int(workers), False, 'environment_override'))

    def test_diagnostic_defaults_to_serial_and_invalid_policy_is_rejected(self):
        self.assertEqual(P.cache_restore_policy({'AUGUR_CACHE_RESTORE_DIAGNOSTIC':'1'}),
                         (1, True, 'diagnostic_default'))
        for value in ('', '0', '3', '16', '8 ', 8, True, None):
            with self.subTest(value=value), self.assertRaises(C.ProtocolError):
                P.cache_restore_policy({'AUGUR_CACHE_RESTORE_WORKERS':value})
        with self.assertRaises(C.ProtocolError):
            P.cache_restore_policy({'AUGUR_CACHE_RESTORE_DIAGNOSTIC':'1', 'AUGUR_CACHE_RESTORE_WORKERS':'8'})
        with self.assertRaises(C.ProtocolError):
            P.cache_restore_policy({'AUGUR_CACHE_RESTORE_DIAGNOSTIC':'on'})

    def test_native_receipt_must_confirm_selected_policy_and_bound_producer(self):
        capability=dict(input_prefix_restore_workers=8, input_prefix_restore_diagnostic=False,
                        bindings=dict(producer_sha256='a'*64))
        self.assertTrue(P.verify_cache_restore_policy(dict(checkpoint_capabilities=capability),8,False,'a'*64,True))
        self.assertFalse(P.verify_cache_restore_policy({},8,False,'a'*64,False))
        disabled=dict(checkpoint_capabilities=dict(input_prefix_tokens=None, save_restore_enabled=False))
        self.assertFalse(P.verify_cache_restore_policy(disabled,8,False,'a'*64,False))
        with self.assertRaises(C.ProtocolError):
            P.verify_cache_restore_policy(disabled,8,False,'a'*64,True)
        for native, workers, diagnostic, producer in [({},8,False,'a'*64),
                (dict(checkpoint_capabilities=capability),1,False,'a'*64),
                (dict(checkpoint_capabilities=capability),8,True,'a'*64),
                (dict(checkpoint_capabilities=capability),8,False,'b'*64)]:
            with self.assertRaises(C.ProtocolError):
                P.verify_cache_restore_policy(native,workers,diagnostic,producer,True)


class CancellationPolicy(unittest.TestCase):
    def test_product_default_is_discard_without_modifying_parent_environment(self):
        environment = {}
        self.assertEqual(P.cancellation_policy(environment), 'discard')
        self.assertEqual(environment, {})

    def test_explicit_rollback_and_discard_are_preserved(self):
        for mode in ('retire', 'discard'):
            self.assertEqual(P.cancellation_policy({'AUGUR_CANCEL_RECOVERY': mode}), mode)

    def test_invalid_override_cannot_silently_select_default(self):
        for mode in ('', 'DISCARD', 'discard ', '0', None, True):
            with self.subTest(mode=mode), self.assertRaises(C.ProtocolError):
                P.cancellation_policy({'AUGUR_CANCEL_RECOVERY': mode})

    def test_default_requires_exact_live_native_discard_capability(self):
        P.verify_cancellation_policy(dict(owner_resident=True, request_bound_cancellation_discard=True), 'discard')
        for health in ({}, dict(owner_resident=False, request_bound_cancellation_discard=True),
                       dict(owner_resident=True), dict(owner_resident=True, request_bound_cancellation_discard=1),
                       dict(owner_resident=True, request_bound_cancellation_discard=False)):
            with self.subTest(health=health), self.assertRaises(C.ProtocolError):
                P.verify_cancellation_policy(health, 'discard')

    def test_rollback_accepts_legacy_absence_but_never_an_enabled_discard_owner(self):
        P.verify_cancellation_policy(dict(owner_resident=True), 'retire')
        P.verify_cancellation_policy(dict(owner_resident=True, request_bound_cancellation_discard=False), 'retire')
        for mode, health in [('unknown', dict(owner_resident=True)),
                             ('retire', dict(owner_resident=True, request_bound_cancellation_discard=True))]:
            with self.assertRaises(C.ProtocolError):
                P.verify_cancellation_policy(health, mode)


class Product(unittest.TestCase):
    def setUp(self):
        controller=SimpleNamespace(model='quasar-27b',faulted=False,busy=threading.Lock(),capabilities=lambda:dict(vision_forward=True))
        self.key='unit-test-key-long-enough-123456'
        self.server=P.Server(('127.0.0.1',0),controller,self.key)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()

    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join()

    def get(self,path,**headers):
        connection=http.client.HTTPConnection('127.0.0.1',self.server.server_address[1],timeout=5)
        connection.request('GET',path,headers=headers);response=connection.getresponse()
        result=response.status,dict(response.getheaders()),response.read();connection.close();return result

    def test_static_assets_and_no_secret_bootstrap(self):
        for path in P.FILES:
            status,headers,body=self.get(path)
            self.assertEqual(status,200);self.assertNotIn(self.key.encode(),body)
            self.assertIn("frame-ancestors 'none'",headers['Content-Security-Policy'])
            self.assertEqual(headers['Cache-Control'],'no-store')
        self.assertEqual(self.get('/../../augur_product.py')[0],401)

    def test_authenticated_same_origin_and_foreign_origin_rejection(self):
        origin='http://127.0.0.1:'+str(self.server.server_address[1])
        self.assertEqual(self.get('/v1/models')[0],401)
        status,_,body=self.get('/v1/models',Authorization='Bearer '+self.key,Origin=origin)
        self.assertEqual(status,200);self.assertEqual(json.loads(body)['data'][0]['id'],'quasar-27b')
        self.assertEqual(self.get('/v1/models',Authorization='Bearer '+self.key,Origin='http://evil.test')[0],403)
        self.assertEqual(self.get('/',Host='evil.test')[0],403)
        self.assertEqual(self.get('/',**{'Sec-Fetch-Site':'cross-site'})[0],403)
        self.assertEqual(self.get('/',Origin='http://evil.test')[0],403)

    def post(self, path, body=b'', **headers):
        connection=http.client.HTTPConnection('127.0.0.1',self.server.server_address[1],timeout=5)
        headers.setdefault('Content-Length',str(len(body)))
        connection.request('POST',path,body=body or None,headers=headers);response=connection.getresponse()
        result=response.status,dict(response.getheaders()),response.read();connection.close();return result

    def test_stop_route_requires_the_key_and_answers_before_shutting_down(self):
        origin='http://127.0.0.1:'+str(self.server.server_address[1])
        # Unauthenticated, cross-origin and parameterised stops are all refused, and none of them
        # set the flag: a refused request must not leave the service half-stopped.
        self.assertEqual(self.post('/augur/v1/stop')[0],401)
        self.assertEqual(self.post('/augur/v1/stop',Authorization='Bearer '+self.key,Origin='http://evil.test')[0],403)
        self.assertEqual(self.post('/augur/v1/stop',b'{"now":true}',Authorization='Bearer '+self.key)[0],400)
        self.assertFalse(self.server.stop_requested.is_set())
        status,_,body=self.post('/augur/v1/stop',Authorization='Bearer '+self.key,Origin=origin)
        self.assertEqual(status,202);self.assertEqual(json.loads(body),dict(accepted=True,operation='stop'))
        self.assertTrue(self.server.stop_requested.wait(2))

    def test_repeated_stop_requests_drain_once(self):
        # The 202 is answered before the drain, so a second call can still land while the socket
        # is closing. It must not start a second drain: shutdown() is what releases the GPU.
        with patch.object(self.server,'shutdown') as shutdown:
            self.server.request_stop();self.server.request_stop()
            self.assertEqual(shutdown.call_count,1)

    def test_unknown_post_route_is_not_treated_as_a_stop(self):
        status,_,body=self.post('/v1/chat/completions',b'{}',Authorization='Bearer '+self.key)
        self.assertNotEqual(status,202)
        self.assertFalse(self.server.stop_requested.is_set())

    def test_bad_build_identity_fails_before_any_child_spawn(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'build-receipt.json').write_text(json.dumps(dict(feature='legacy')))
            with self.assertRaises(C.ProtocolError):P.verified_build(root)

    def test_frozen_source_tamper_and_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'source').mkdir()
            source=root/'source'/'native.rs'; source.write_text('pinned')
            binary=root/'owner'; binary.write_bytes(b'pinned executable')
            identity=dict(files={'native.rs':P.sha(source)})
            P.save(root/'source-identity.json',identity)
            P.save(root/'build-receipt.json',dict(feature='augur-model-qwen35-runtime/qwen35-augur-vision',
                source_sha256=P.sha(root/'source-identity.json'),binaries={'owner':dict(path=str(binary),sha256=P.sha(binary))}))
            P.save(root/'cpu-complete.json',dict(status='PASS',build_sha256=P.sha(root/'build-receipt.json')))
            self.assertEqual(P.verified_build(root)['binaries']['owner']['sha256'],P.sha(binary))
            source.write_text('tampered')
            with self.assertRaises(C.ProtocolError):P.verified_build(root)
            identity['files']={'../owner':P.sha(binary)}
            (root/'source-identity.json').write_text(json.dumps(identity))
            receipt=json.loads((root/'build-receipt.json').read_bytes()); receipt['source_sha256']=P.sha(root/'source-identity.json')
            (root/'build-receipt.json').write_text(json.dumps(receipt))
            (root/'cpu-complete.json').write_text(json.dumps(dict(status='PASS',build_sha256=P.sha(root/'build-receipt.json'))))
            with self.assertRaises(C.ProtocolError):P.verified_build(root)

    def test_snapshot_asset_is_served_from_the_bound_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'index.html').write_bytes(b'frozen page')
            self.server.assets=root
            self.assertEqual(self.get('/')[2],b'frozen page')

    def test_cleanup_error_does_not_skip_later_teardown(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); native=root/'native'/'owner-0'; native.mkdir(parents=True)
            P.save(native/'owner-stopped.json',dict(allocation_closure=True,live_allocations_after=0,live_bytes_after=0,drain_error=None))
            P.save(native/'retirement.json',dict(confirmed_exit=True,native_cleanup_verified=True))
            process=Mock(pid=42,returncode=None)
            process.poll.side_effect=lambda:process.returncode
            process.wait.side_effect=lambda **kw:setattr(process,'returncode',0)
            controller=Mock(busy=threading.Lock()); controller.abort.side_effect=OSError('abort probe')
            server=Mock(); transport=Mock()
            stopped=P.cleanup(process,server,controller,None,'http://127.0.0.1:1',root,lambda:transport)
            controller.close.assert_called_once(); server.server_close.assert_called_once()
            transport.shutdown.assert_called_once(); process.wait.assert_called_once()
            self.assertTrue(stopped['allocation_closure']); self.assertTrue(stopped['confirmed_exit'])
            self.assertFalse(stopped['cleanup_verified'])
            self.assertEqual(stopped['cleanup_errors'][0]['phase'],'controller_abort')

    def test_missing_allocation_receipt_is_not_verified_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            process=Mock(pid=42,returncode=0); process.poll.return_value=0
            stopped=P.cleanup(process,None,None,None,None,Path(directory))
            self.assertFalse(stopped['allocation_closure']); self.assertFalse(stopped['cleanup_verified'])


class ProductSignals(unittest.TestCase):
    def test_signal_only_records_request_and_restores_previous_handlers(self):
        previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
        with P.StopSignals() as stop:
            signal.raise_signal(signal.SIGTERM)
            signal.raise_signal(signal.SIGINT)
            self.assertEqual(stop.receipt(), dict(signal='SIGTERM', count=2, interrupted_error=None))
            with self.assertRaises(P.StopRequested): stop.check()
        for signum, handler in previous.items():
            self.assertEqual(signal.getsignal(signum), handler)

    def test_worker_exception_is_propagated_without_a_stop_request(self):
        stop = P.StopSignals()
        def fail(): raise ValueError('real worker failure')
        with self.assertRaisesRegex(ValueError, 'real worker failure'):
            stop.run(fail, Mock())

    def test_startup_abort_retries_until_transport_is_active(self):
        stop = P.StopSignals()
        released = threading.Event()
        attempts = []
        def work():
            stop.receive(signal.SIGTERM, None)
            if not released.wait(5): raise RuntimeError('test drain timeout')
            raise OSError('cancelled startup socket')
        def abort():
            attempts.append(True)
            if len(attempts) == 2:
                stop.receive(signal.SIGTERM, None)
                released.set()
        with self.assertRaises(P.StopRequested): stop.run(work, abort)
        self.assertEqual(len(attempts), 2)
        self.assertEqual(stop.receipt()['count'], 2)
        self.assertEqual(stop.receipt()['interrupted_error']['type'], 'OSError')

    def test_server_shutdown_runs_off_the_serving_thread(self):
        controller = SimpleNamespace(model='quasar-27b', faulted=False, busy=threading.Lock())
        server = P.Server(('127.0.0.1', 0), controller, 'signal-test-key-long-enough-12345')
        stop = P.StopSignals()
        serving = []
        def serve():
            serving.append(threading.get_ident())
            stop.receive(signal.SIGTERM, None)
            server.serve_forever(poll_interval=.01)
        def shutdown():
            self.assertNotEqual(threading.get_ident(), serving[0])
            server.shutdown()
        try:
            with self.assertRaises(P.StopRequested): stop.run(serve, shutdown)
        finally:
            server.server_close()

    def test_pre_owner_shutdown_requires_explicit_no_owner_and_no_spawn_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def clean(shutdown):
                process = Mock(pid=42, returncode=None)
                process.poll.side_effect = lambda: process.returncode
                process.wait.side_effect = lambda **kw: setattr(process, 'returncode', 0)
                transport = Mock()
                transport.shutdown.return_value = shutdown
                return P.cleanup(process, None, None, None, 'http://127.0.0.1:1', root, lambda: transport)
            self.assertFalse(clean(dict(event='shutdown'))['cleanup_verified'])
            stopped = clean(dict(event='shutdown', no_owner=True))
            self.assertTrue(stopped['cleanup_verified'])
            self.assertTrue(stopped['native_owner_never_spawned'])
            self.assertFalse(stopped['allocation_closure'])
            (root/'native'/'owner-0').mkdir(parents=True)
            self.assertFalse(clean(dict(event='shutdown', no_owner=True))['cleanup_verified'])

    def test_signal_during_child_creation_retains_handle_and_runs_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stop = P.StopSignals()
            process = Mock(pid=42)
            process.stdout.readline.return_value = b'{"address":"127.0.0.1:23456"}\n'
            def spawn(*args, **kwargs):
                stop.receive(signal.SIGTERM, None)
                return process
            binding = dict(binaries=dict(augur_session_http=dict(path='adapter'),
                augur_session_jsonl=dict(path='producer', sha256='a'*64)))
            environment = {key: 'test-only' for key in P.DRAFT_KEYS}
            argv = ['--build', str(root), '--model-dir', str(root), '--manifest', str(root/'manifest'),
                    '--manifest-sha256', 'a'*64, '--evidence', str(root/'run'), '--port', '0']
            with patch.dict(P.os.environ, environment), patch.object(P, 'EngineLease'), \
                 patch.object(P, 'StopSignals', return_value=stop), \
                 patch.object(P, 'verified_build', return_value=binding), patch.object(P.Q, 'Template'), \
                 patch.object(P, 'snapshot_public', return_value={}), patch.object(P, 'sha', return_value='a'*64), \
                 patch.object(P.subprocess, 'Popen', side_effect=spawn), patch.object(P.Q, 'Controller') as controller, \
                 patch.object(P, 'cleanup', return_value=dict(cleanup_verified=True)) as cleanup:
                P.main(argv)
            controller.assert_not_called()
            self.assertIs(cleanup.call_args.args[0], process)
            self.assertEqual(cleanup.call_args.args[4], 'http://127.0.0.1:23456')
            stopped = json.loads((root/'run'/'stopped.json').read_bytes())
            self.assertEqual(stopped['stop_request']['signal'], 'SIGTERM')
            self.assertFalse((root/'run'/'ready.json').exists())


if __name__=='__main__':unittest.main()


class DflashSelection(unittest.TestCase):
    def test_resident_default_and_explicit_b8_profile(self):
        self.assertEqual(P.dflash_profile({}),P.Q.AUGUR_PROFILE)
        self.assertEqual(P.dflash_profile({'AUGUR_HTTP_DFLASH':'resident_only'}),P.Q.AUGUR_PROFILE)
        self.assertEqual(P.dflash_profile({'AUGUR_HTTP_DFLASH':'batched8'}),P.Q.DFLASH_PROFILE)

    def test_bad_dflash_mode_cannot_fallback_to_resident(self):
        for value in ('','BATCHED8','batched8 ',True,None,8):
            with self.subTest(value=value),self.assertRaises(C.ProtocolError):
                P.dflash_profile({'AUGUR_HTTP_DFLASH':value})
