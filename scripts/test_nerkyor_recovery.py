import copy
import json
from pathlib import Path
import tempfile
import unittest
import nerkyor_protocol as P
from test_nerkyor_chat import packet, ready, template


def cancellation(outcome='unpublished_discarded'):
    plan, ledger, tokens, final = packet()
    terminal = None
    previous = None
    if outcome == 'committed_session_closed':
        terminal = P.verify_terminal(final, plan, ledger, tokens)
        previous = terminal
    elif outcome == 'unadmitted_session_closed':
        previous = P.verify_terminal(final, plan, ledger, tokens)
        ledger.accept(plan, previous, 'answer')
        plan = copy.deepcopy(plan);plan['request']['request_id'] = 'r2';plan['request']['input']['kind'] = 'append'
        tokens = []
    elif outcome == 'unadmitted_empty':
        tokens = []
    r = dict(schema='nerkyor-request-cancel-resolution-v1', contract=P.RECOVERY_CONTRACT, profile=P.PROFILE,
        request_id='cancel1', cancelled_request_id=plan['request']['request_id'], session_id=ledger.session_id,
        manifest_sha256=ledger.template.manifest_sha256, outcome=outcome,
        original_turn_receipt_sha256=terminal['receipt_sha256'] if terminal else None,
        submitted_tokens_before_resolution=0 if outcome == 'unadmitted_empty' else 3,
        submitted_tokens_after_resolution=0, active_session=False, owner_reusable=True,
        previous_history_reusable=False, weights_reloaded=False, product_ready=False)
    proof = dict(request_id='cancel1', session_id=ledger.session_id, manifest_sha256=ledger.template.manifest_sha256)
    if outcome == 'unpublished_discarded':
        proof.update(schema='nerkyor-session-cancellation-discard-v1', cancelled_request_id=plan['request']['request_id'],
            cancellation_boundary='before token publication', submitted_tokens_before_discard=3,
            submitted_tokens_after_discard=0, drain_completed=True, reset_completed=True, session_discarded=True,
            previous_history_reusable=False, product_ready=False, dflash_enabled=True)
    elif previous is not None:
        proof.update(schema='nerkyor-session-close-v1', revision=1,
                     last_turn_receipt_sha256=previous['receipt_sha256'], reset_completed=True)
    r['cleanup_proof'] = None if outcome == 'unadmitted_empty' else dict(receipt=proof,receipt_sha256='d'*64)
    frame = dict(event='cancelled', request_id=plan['request']['request_id'], session_id=ledger.session_id,
                 resolution=dict(receipt=r,receipt_sha256='e'*64), owner_retired=False, product_ready=False)
    return frame, plan, ledger, tokens, terminal


class RecoveryProtocolTests(unittest.TestCase):
    def test_ready_versions_never_alias_each_other_or_arithmetic_identity(self):
        t = template();old = ready(t)
        new = dict(old, schema=P.RECOVERY_SCHEMA, request_bound_cancellation_discard=True, request_cancellation_contract=P.RECOVERY_CONTRACT)
        P.verify_ready(old,t);P.verify_recovery_ready(new,t)
        with self.assertRaises(P.C.ProtocolError):P.verify_ready(new,t)
        with self.assertRaises(P.C.ProtocolError):P.verify_recovery_ready(old,t)
        for key,value in [('request_bound_cancellation_discard',1),('request_cancellation_contract','generic-reset'),('profile','quasar')]:
            bad=copy.deepcopy(new);bad[key]=value
            with self.assertRaises(P.C.ProtocolError):P.verify_recovery_ready(bad,t)

    def test_four_outcomes_keep_distinct_commit_semantics(self):
        for outcome in ('unpublished_discarded','committed_session_closed','unadmitted_session_closed','unadmitted_empty'):
            f,p,l,t,c=cancellation(outcome)
            result=P.verify_cancellation(f,p,l,t,'cancel1',c)
            self.assertEqual(result['receipt']['outcome'],outcome)
            self.assertFalse(l.poisoned)  # Validator itself grants no lifecycle mutation.
            f['resolution']['receipt']['outcome']='foreign'
            self.assertEqual(result['receipt']['outcome'],outcome)

    def test_foreign_ids_schema_types_and_reset_claims_rejected(self):
        f,p,l,t,c=cancellation()
        for key,value in [('schema','quasar-session-cancellation-discard-v1'),('request_id','other'),
            ('cancelled_request_id','wrong'),('session_id','foreign'),('manifest_sha256','b'*64),
            ('owner_reusable',1),('product_ready',0),('submitted_tokens_after_resolution',False),
            ('submitted_tokens_before_resolution',True),('previous_history_reusable',True),('weights_reloaded',True)]:
            bad=copy.deepcopy(f);bad['resolution']['receipt'][key]=value
            with self.assertRaises(P.C.ProtocolError):P.verify_cancellation(bad,p,l,t,'cancel1',c)
        for key,value in [('drain_completed',False),('reset_completed',1),('dflash_enabled',False),
            ('cancellation_boundary',''),('cancelled_request_id','wrong'),('submitted_tokens_before_discard',2)]:
            bad=copy.deepcopy(f);bad['resolution']['receipt']['cleanup_proof']['receipt'][key]=value
            with self.assertRaises(P.C.ProtocolError):P.verify_cancellation(bad,p,l,t,'cancel1',c)

    def test_durable_commit_cannot_be_reclassified_as_unpublished(self):
        f,p,l,t,c=cancellation('committed_session_closed')
        with self.assertRaises(P.C.ProtocolError):P.verify_cancellation(f,p,l,t,'cancel1',None)
        f['resolution']['receipt']['outcome']='unpublished_discarded'
        with self.assertRaises(P.C.ProtocolError):P.verify_cancellation(f,p,l,t,'cancel1',c)

    def test_close_proof_must_bind_previous_committed_revision(self):
        for outcome in ('committed_session_closed','unadmitted_session_closed'):
            f,p,l,t,c=cancellation(outcome)
            for key,value in [('revision',True),('last_turn_receipt_sha256','b'*64),('session_id','wrong')]:
                bad=copy.deepcopy(f);bad['resolution']['receipt']['cleanup_proof']['receipt'][key]=value
                with self.assertRaises(P.C.ProtocolError):P.verify_cancellation(bad,p,l,t,'cancel1',c)
        f,p,l,t,c=cancellation('unadmitted_empty')
        with self.assertRaises(P.C.ProtocolError):P.verify_cancellation(f,p,l,[1],'cancel1',None)

    def test_unknown_proof_fields_outcomes_and_early_flags_fail_closed(self):
        f,p,l,t,c=cancellation()
        for mutate in [lambda r:r.update(outcome='reset_any_error'),lambda r:r.update(extra=True),
                       lambda r:r.update(cleanup_proof=None),lambda r:r['cleanup_proof']['receipt'].update(extra=1)]:
            bad=copy.deepcopy(f);mutate(bad['resolution']['receipt'])
            with self.assertRaises(P.C.ProtocolError):P.verify_cancellation(bad,p,l,t,'cancel1',c)
        with self.assertRaises(P.C.ProtocolError):P.verify_cancellation(f,p,l,t,'r1',c)

    def test_durable_envelope_and_cleanup_are_both_required_and_no_success_may_exist(self):
        f,p,l,t,c=cancellation();committed=P.verify_cancellation(f,p,l,t,'cancel1',c)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'turns').mkdir();(root/'cancellations').mkdir()
            def write(path,value):
                raw=json.dumps(value).encode();path.write_bytes(raw);return P.C.sha256(raw)
            proof=committed['receipt']['cleanup_proof']
            proof['receipt_sha256']=write(root/'turns/cancel1.json',proof['receipt'])
            committed['receipt_sha256']=write(root/'cancellations/cancel1.json',committed['receipt'])
            P.persisted_cancellation(root,committed)
            (root/'turns/r1.json').write_text('{}',encoding='utf-8')
            with self.assertRaises(P.C.ProtocolError):P.persisted_cancellation(root,committed)
            (root/'turns/r1.json').unlink()
            (root/'turns/cancel1.json').write_text('{}',encoding='utf-8')
            with self.assertRaises(P.C.ProtocolError):P.persisted_cancellation(root,committed)
# Execution/HTTP ownership tests use no CUDA and never claim performance evidence.
import queue
import threading
import time
from types import SimpleNamespace
from unittest import mock
import nerkyor_recovery as R
import nerkyor_api as H
from test_nerkyor_api import FakeEngine, payload, run


class FakeRecoveryEngine(FakeEngine):
    def __init__(self):
        super().__init__();self.resolutions=[]

    def resolved(self, ledger, outcome):
        ledger.poisoned=True
        self.resolutions.append((threading.get_ident(),outcome))
        return dict(receipt=dict(outcome=outcome),receipt_sha256='e'*64)

    def generate_plan(self, ledger, plan, emit, cancel=None):
        self.entered.set()
        while not self.release.wait(.01):
            if cancel.is_set():
                raise R.RequestCancelled(self.resolved(ledger,'unpublished_discarded'))
        if cancel.is_set():raise R.RequestCancelled()
        try:return super().generate_plan(ledger,plan,emit)
        except R.DISCONNECTS:
            raise R.RequestCancelled(self.resolved(ledger,'unpublished_discarded'))

    def cancel_completed(self, ledger, plan, committed):
        return self.resolved(ledger,'committed_session_closed')


class RecoveryControllerTests(unittest.TestCase):
    def setUp(self):
        self.engine=FakeRecoveryEngine();self.controller=H.RecoverableController(self.engine)

    def test_capabilities_keep_recovery_separate_from_full_product(self):
        c=self.controller.capabilities()
        self.assertTrue(c['request_bound_cancellation_discard']);self.assertFalse(c['product_ready'])
        self.assertEqual(c['cancellation_commit_race'],'close_irreversible_terminal')
        self.assertFalse(c['automatically_restarted'])

    def test_cancel_then_next_request_and_stale_watcher_cannot_cancel_new_request(self):
        callbacks=[];errors=[]
        self.engine.release.clear()
        def work():
            try:run(self.controller,payload(),watch=lambda cb: callbacks.append(cb) or (lambda:None))
            except BaseException as e:errors.append(e)
        first=threading.Thread(target=work);first.start();self.assertTrue(self.engine.entered.wait(2))
        callbacks[0]();first.join(2);self.assertFalse(first.is_alive())
        self.assertIsInstance(errors.pop(),R.RequestCancelled)
        self.assertFalse(self.controller.faulted);self.assertFalse(self.engine.cancel_requested.is_set());self.assertFalse(self.engine.aborts)
        self.assertIsNone(self.controller.current)
        self.engine.entered.clear();second=threading.Thread(target=work);second.start();self.assertTrue(self.engine.entered.wait(2))
        callbacks[0]()  # Old callback captures only its retired request Event.
        self.engine.release.set();second.join(2);self.assertFalse(second.is_alive());self.assertFalse(errors)
        self.assertIsNotNone(self.controller.current);self.assertFalse(self.controller.faulted)

    def test_cancel_before_submission_preserves_prior_session(self):
        run(self.controller,payload());previous=self.controller.current
        with self.assertRaises(R.RequestCancelled):
            run(self.controller,payload(),watch=lambda cb: cb() or (lambda:None))
        self.assertIs(self.controller.current,previous);self.assertFalse(self.engine.resolutions)
        self.assertFalse(self.engine.aborts);self.assertFalse(self.controller.faulted)

    def test_late_signal_closes_committed_session_without_owner_retirement(self):
        with self.assertRaises(R.RequestCancelled):
            run(self.controller,payload(),watch=lambda cb:(lambda:cb()))
        self.assertEqual(self.engine.resolutions[-1][1],'committed_session_closed')
        self.assertIsNone(self.controller.current);self.assertFalse(self.controller.faulted);self.assertFalse(self.engine.aborts)
        run(self.controller,payload());self.assertFalse(self.controller.faulted)

    def test_late_socket_write_failure_recovers_but_unknown_failure_retires(self):
        def gone(*args):raise BrokenPipeError('downstream closed')
        with self.assertRaises(R.RequestCancelled):run(self.controller,payload(),deliver=gone)
        self.assertEqual(self.engine.resolutions[-1][1],'committed_session_closed');self.assertFalse(self.controller.faulted)
        def unknown(*args):raise ValueError('unexpected validator/publication error')
        with self.assertRaises(ValueError):run(self.controller,payload(),deliver=unknown)
        self.assertTrue(self.controller.faulted);self.assertTrue(self.engine.aborts)


class RecoveryEngineTests(unittest.TestCase):
    def engine(self, ledger, root):
        e=object.__new__(R.Engine);e.template=ledger.template
        e.template.tokenizer=SimpleNamespace(decode=lambda ids,**kwargs:''.join(map(str,ids)))
        e.frames=queue.Queue();e.cancel_requested=threading.Event();e.closed=e.poisoned=False;e.last_completed=None
        e.turn_timeout=e.stop_grace=1;e.evidence=root;e.native=root/'native'
        (e.native/'turns').mkdir(parents=True);(e.native/'cancellations').mkdir()
        e.process=SimpleNamespace(poll=lambda:None,pid=123)
        e.abort=mock.Mock();return e

    def test_signal_after_dequeue_does_not_drop_the_native_frame(self):
        e=object.__new__(R.Engine);e.cancel_requested=threading.Event();flag=threading.Event();observed=[]
        class Frames:
            def get(self,**kwargs):flag.set();return b'{"event":"token"}\n'
        e.frames=Frames()
        frame=e._read_request([time.monotonic()+1],lambda:observed.append(flag.is_set()))
        self.assertEqual(frame['event'],'token');self.assertEqual(observed,[False,True])

    def test_protocol_error_with_cancel_flag_never_authorizes_reuse(self):
        plan,ledger,_,_=packet()
        with tempfile.TemporaryDirectory() as d:
            e=self.engine(ledger,Path(d));flag=threading.Event();writes=[]
            def write(value):
                writes.append(value)
                if value['op']=='turn':flag.set();e.frames.put(b'{"event":"token","token":{}}\n')
            e._write=write
            with self.assertRaises(P.C.ProtocolError):e.generate_plan(ledger,plan,cancel=flag)
            self.assertEqual([x['op'] for x in writes],['turn','cancel_turn'])
            e.abort.assert_called_once();self.assertTrue(e.poisoned)

    def test_verified_durable_discard_keeps_owner_and_poisons_only_old_ledger(self):
        frame,plan,ledger,_,_=cancellation()
        with tempfile.TemporaryDirectory() as d:
            e=self.engine(ledger,Path(d));flag=threading.Event();writes=[];deltas=[]
            def save(path,value):
                raw=json.dumps(value).encode();path.write_bytes(raw);return P.C.sha256(raw)
            def write(value):
                writes.append(value)
                if value['op']=='turn':
                    e.frames.put(P.canonical_json(dict(event='token',product_ready=False,token=dict(request_id='r1',session_id=ledger.session_id,sequence=0,token_id=3))))
                    flag.set()
                else:
                    r=frame['resolution']['receipt'];r['request_id']=value['request_id']
                    proof=r['cleanup_proof'];proof['receipt']['request_id']=value['request_id']
                    proof['receipt_sha256']=save(e.native/'turns'/(value['request_id']+'.json'),proof['receipt'])
                    frame['resolution']['receipt_sha256']=save(e.native/'cancellations'/(value['request_id']+'.json'),r)
                    e.frames.put(P.canonical_json(frame))
            e._write=write
            with self.assertRaises(R.RequestCancelled) as caught:e.generate_plan(ledger,plan,deltas.append,cancel=flag)
            self.assertEqual(caught.exception.resolution['receipt']['outcome'],'unpublished_discarded')
            e.abort.assert_not_called();self.assertFalse(e.poisoned);self.assertTrue(ledger.poisoned);self.assertFalse(deltas)
            self.assertEqual(len(list((e.evidence/'request-cancellations').glob('*.json'))),1)

if __name__ == '__main__':
    unittest.main()
