"""CPU HTTP/security tests; fake native owner, no GPU qualification implied."""
import copy
import http.client
import io
import json
import queue
import subprocess
import sys
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock
import nerkyor_api as H
import nerkyor_chat as N
import nerkyor_protocol as P
from test_nerkyor_chat import FakeTextTemplate


def payload(text='hi', **kwargs):
    return dict(model=H.MODEL, messages=[dict(role='user', content=text)], max_tokens=12, **kwargs)


class FakeEngine:
    def __init__(self):
        self.template = FakeTextTemplate()
        self.poisoned = self.closed = False
        self.process = SimpleNamespace(poll=lambda: 0 if self.closed else None)
        self.cancel_requested = threading.Event()
        self.plans, self.resets, self.aborts = [], [], []
        self.entered, self.release = threading.Event(), threading.Event()
        self.release.set()

    request_cancel = N.Engine.request_cancel

    def generate_plan(self, ledger, plan, emit):
        self.plans.append(copy.deepcopy(plan)); self.entered.set()
        while not self.release.wait(.01):
            if self.cancel_requested.is_set():
                raise P.C.ProtocolError('cancelled')
        text = '42'
        receipt = dict(stop_reason='eos', selected_tokens_including_eos=2,
            session=dict(revision=(ledger.terminal['receipt']['session']['revision']+1 if ledger.terminal else 1),
                consumed_token_ids=plan['input_ids']+[101], pending_token=248046))
        committed = dict(receipt=receipt, receipt_sha256='f'*64)
        ledger.accept(plan, committed, text)
        emit(text)
        return dict(text=text, committed=committed)

    def close_session(self, ledger):
        self.resets.append(ledger.session_id); ledger.poisoned = True

    def abort(self):
        self.aborts.append(threading.get_ident()); self.closed = self.poisoned = True

    def close(self):
        self.closed = True


def run(controller, value, **kwargs):
    captured = {}
    begin = lambda meta, request: captured.update(meta=meta, request=request)
    deliver = lambda meta, request, text, finish, usage: captured.update(text=text, finish=finish, usage=usage)
    event = controller.complete(value, kwargs.get('begin', begin), kwargs.get('emit', lambda t: None),
        kwargs.get('deliver', deliver), kwargs.get('watch', lambda abort: lambda: None))
    return captured, event


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEngine(); self.controller = H.Controller(self.engine)

    def test_full_unknown_history_and_system_are_not_discarded(self):
        value = payload()
        value['messages'] = [dict(role='system', content='a rule'), dict(role='user', content='first'),
            dict(role='assistant', content='prior answer'), dict(role='user', content='next')]
        result, _ = run(self.controller, value)
        plan = self.engine.plans[-1]
        self.assertEqual([m['content'] for m in plan['messages']], ['a rule','first','prior answer','next'])
        self.assertEqual(plan['request']['input']['kind'], 'open')
        self.assertEqual(result['usage']['completion_tokens'], 2)

    def test_exact_append_preserves_ids_then_edited_history_resets(self):
        run(self.controller, payload())
        old = self.controller.current
        history = copy.deepcopy(old.history)
        value = payload(); value['messages'] += [dict(role='assistant', content='42'), dict(role='user', content='again')]
        result, _ = run(self.controller, value)
        self.assertEqual(result['meta']['session_mode'], 'exact_history_append')
        self.assertEqual(self.engine.plans[-1]['input_ids'][:len(history)+1], history+[248046])
        self.assertEqual(result['usage']['prompt_tokens_details']['cached_tokens'], len(history))
        value['messages'][0]['content'] = 'different'
        result, _ = run(self.controller, value)
        self.assertEqual(result['meta']['session_mode'], 'fresh_history_reset')
        self.assertEqual(self.engine.resets, [old.session_id])
        self.assertIn('different', self.engine.plans[-1]['prefix'])

    def test_bad_budget_and_overflow_do_not_reset_live_session(self):
        run(self.controller, payload()); old = self.controller.current
        for value in [dict(payload(),max_tokens=487),payload('z'*800),dict(payload(),seed=0)]:
            with self.assertRaises(H.A.ApiError): run(self.controller, value)
            self.assertIs(self.controller.current,old)
        self.assertFalse(self.engine.resets); self.assertFalse(self.controller.faulted)
        self.assertEqual(len(self.engine.plans),1)

    def test_live_ledger_rejects_last_user_only_direct_planning(self):
        run(self.controller, payload())
        with self.assertRaises(P.C.ProtocolError):
            self.controller.current.plan_messages([P.C.ChatTemplate.message('user','next')],12)

    def test_unsupported_settings_are_rejected_not_ignored(self):
        for key in ('seed','top_k','min_p','tools','tool_choice','parallel_tool_calls',
                    'enable_thinking','thinking_language','reasoning_effort'):
            for value in (None,False,0):
                with self.subTest(key=key,value=value), self.assertRaises(H.A.ApiError):
                    H.validate_request(dict(payload(),**{key:value}),H.MODEL)
        for extra in ({'temperature':1},{'max_tokens':True},{'stream':1},{'model':'quasar-27b'}):
            with self.assertRaises(H.A.ApiError): H.validate_request(dict(payload(),**extra),H.MODEL)

    def test_capabilities_do_not_claim_product_or_sampling(self):
        c = self.controller.capabilities()
        for name in ('product_ready','qualified','mtp_enabled','tool_calls','thinking','sampling','request_bound_cancellation_discard'):
            self.assertIs(c[name],False)
        self.assertEqual(c['native_limits']['total_context_tokens'],512)

    def test_concurrent_admission_is_nonblocking_and_does_not_touch_engine(self):
        with self.controller.busy:
            with self.assertRaises(H.A.ApiError) as caught: run(self.controller,payload())
        self.assertEqual((caught.exception.status,caught.exception.code),(409,'single_user_busy'))
        self.assertFalse(self.engine.plans)

    def test_cancel_watcher_only_signals_and_execution_thread_alone_retires(self):
        ids = []
        def watch(abort):
            def callback(): ids.append(threading.get_ident()); abort()
            thread = threading.Thread(target=callback); thread.start(); thread.join()
            return lambda: None
        owner_thread = threading.get_ident()
        with self.assertRaises(P.C.ProtocolError): run(self.controller,payload(),watch=watch)
        self.assertNotEqual(ids,[owner_thread]); self.assertEqual(self.engine.aborts,[owner_thread])
        self.assertFalse(self.engine.plans)
        with self.assertRaises(H.A.ApiError) as error: run(self.controller,payload())
        self.assertEqual(error.exception.status,503)

    def test_publication_failure_retires_even_after_native_completion(self):
        def fail(*args): raise OSError('closed downstream')
        with self.assertRaises(OSError): run(self.controller,payload(),deliver=fail)
        self.assertIsNone(self.controller.current); self.assertTrue(self.controller.faulted)
        self.assertEqual(len(self.engine.aborts),1)

    def test_late_cancel_is_fenced_before_completion(self):
        delivered = []
        with self.assertRaises(P.C.ProtocolError):
            run(self.controller,payload(),watch=lambda abort: abort,deliver=lambda *a: delivered.append(a))
        self.assertFalse(delivered); self.assertEqual(len(self.engine.aborts),1)

    def test_header_failure_before_native_mutation_preserves_owner(self):
        def fail(*args): raise OSError('closed')
        with self.assertRaises(OSError): run(self.controller,payload(),begin=fail)
        self.assertFalse(self.engine.plans); self.assertFalse(self.controller.faulted)

    def test_metrics_do_not_include_transcript_or_credentials(self):
        metrics=io.StringIO();self.controller.metrics=metrics
        run(self.controller,payload('private-test-prompt'))
        event=json.loads(metrics.getvalue())
        self.assertEqual(event['status'],'completed')
        self.assertNotIn('private-test-prompt',metrics.getvalue()); self.assertNotIn('42',event)

    def test_signal_interrupts_native_read_without_queue_or_process_teardown_on_watcher(self):
        engine=N.Engine.__new__(N.Engine)
        engine.frames=queue.Queue();engine.cancel_requested=threading.Event()
        def signal(): time.sleep(.02);engine.request_cancel()
        thread=threading.Thread(target=signal);thread.start()
        before=time.monotonic()
        with self.assertRaises(P.C.ProtocolError):engine._read(time.monotonic()+10)
        thread.join();self.assertLess(time.monotonic()-before,.5)
        self.assertFalse(hasattr(engine,'process'))


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.engine=FakeEngine();self.controller=H.Controller(self.engine)
        self.key='cpu-test-key-not-a-credential'
        self.server=H.W.Server(('127.0.0.1',0),self.controller,self.key)
        self.thread=threading.Thread(target=self.server.serve_forever,kwargs=dict(poll_interval=.01))
        self.thread.start();self.port=self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join();self.controller.close()

    def query(self,path='/v1/chat/completions',value=None,method='POST',headers=None,raw=None):
        conn=http.client.HTTPConnection('127.0.0.1',self.port,timeout=3)
        h={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'}
        if headers:h.update(headers)
        body=raw if raw is not None else json.dumps(value if value is not None else payload())
        conn.request(method,path,body=body if method=='POST' else None,headers=h)
        response=conn.getresponse();data=response.read();status=response.status;conn.close()
        return status,data

    def test_authentication_and_origin_and_host(self):
        for headers,code in [({'Authorization':'Bearer bad'},401),({'Origin':'http://example.com'},403),
                             ({'Host':'localhost:'+str(self.port)},403),({'Sec-Fetch-Site':'cross-site'},403)]:
            self.assertEqual(self.query(headers=headers)[0],code)
        self.assertEqual(self.query(headers={'Origin':'http://127.0.0.1:'+str(self.port)})[0],200)

    def test_http_nonobject_duplicate_and_nonfinite_json_are_400(self):
        for raw in ('[]','null','1','{"a":1,"a":2}','{"x":NaN}'):
            self.assertEqual(self.query(raw=raw)[0],400)
        self.assertFalse(self.engine.plans)

    def test_json_and_sse_usage_done_and_same_origin_assets(self):
        status,body=self.query();self.assertEqual(status,200)
        self.assertEqual(json.loads(body)['choices'][0]['message']['content'],'42')
        status,body=self.query(value=payload(stream=True,stream_options={'include_usage':True}))
        self.assertEqual(status,200);self.assertTrue(body.endswith(b'data: [DONE]\n\n'))
        self.assertIn(b'"completion_tokens": 2',body)
        self.assertEqual(self.query(path='/',method='GET')[0],200)
        self.assertEqual(self.query(path='/assets/app.js',method='GET')[0],200)

    def test_failed_stream_has_error_but_never_done(self):
        def fail(*args): raise P.C.ProtocolError('fake failure')
        self.engine.generate_plan=fail
        status,body=self.query(value=payload(stream=True))
        self.assertEqual(status,200);self.assertIn(b'"error"',body);self.assertNotIn(b'[DONE]',body)
        self.assertTrue(self.controller.faulted)

    def test_retired_owner_reports_degraded_and_rejects_new_work(self):
        self.engine.abort()
        self.assertEqual(self.query()[0],503)
        status,body=self.query(path='/health',method='GET')
        self.assertEqual(json.loads(body)['status'],'degraded')
        self.assertFalse(self.engine.plans)


class ParentPipeTests(unittest.TestCase):
    def child(self, extra):
        code = 'import nerkyor_api as H, sys, time; ' + extra
        return subprocess.Popen([sys.executable, '-c', code], cwd=Path(__file__).parent,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))

    def test_http_style_exit_while_parent_keeps_stdin_open_does_not_abort_python(self):
        child=self.child('H.watch_parent_stdin(lambda: None, sys.stdin.fileno()); time.sleep(.05)')
        try:
            self.assertEqual(child.wait(timeout=5),0)
            self.assertEqual(child.stderr.read(),b'')
        finally:
            if child.poll() is None:child.kill();child.wait(timeout=5)
            child.stdin.close();child.stdout.close();child.stderr.close()

    def test_parent_signal_runs_stop_callback_and_finishes_watcher(self):
        child=self.child('t=H.watch_parent_stdin(lambda: print("stopped",flush=True),sys.stdin.fileno()); t.join(3); assert not t.is_alive()')
        stdout,stderr=child.communicate(b'\n',timeout=5)
        self.assertEqual(child.returncode,0)
        self.assertEqual(stdout.strip(),b'stopped');self.assertEqual(stderr,b'')


if __name__=='__main__':unittest.main()
