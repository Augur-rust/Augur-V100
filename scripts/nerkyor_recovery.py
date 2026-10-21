"""Opt-in request-scoped cancellation for the retained Nerkyor recovery-v2 producer.

Only the execution thread reads/writes native frames or resets state. Watchers
hold a fresh Event per HTTP request, never a reusable owner cancellation flag.
"""
from __future__ import annotations
import copy
from pathlib import Path
import queue
import threading
import time
import uuid
import nerkyor_chat as N
import nerkyor_protocol as P
C = P.C
DISCONNECTS = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)


class RequestCancelled(C.ProtocolError):
    def __init__(self, resolution=None):
        super().__init__('Request cancelled; native session discarded.' if resolution is not None
                         else 'Request cancelled before native submission.')
        self.resolution = copy.deepcopy(resolution)


class Engine(N.Engine):
    def __init__(self, **kwargs):
        self.last_completed = None
        super().__init__(**kwargs)

    def _verify_ready(self, frame):
        P.verify_recovery_ready(frame, self.template)

    def _read_request(self, deadline, signal):
        # Unlike owner retirement, request cancellation MUST NOT drop a dequeued
        # token/terminal frame. Every native frame is verified exactly once.
        while True:
            C.require(not self.cancel_requested.is_set(), 'owner shutdown requested')
            signal()
            remaining = deadline[0] - time.monotonic()
            C.require(remaining > 0, 'Nerkyor request/control response timeout')
            try:
                raw = self.frames.get(timeout=min(remaining, .05))
                break
            except queue.Empty:
                continue
        C.require(not self.cancel_requested.is_set(), 'owner shutdown requested')
        signal()
        C.require(raw is not None, 'native owner exited before a complete frame')
        if isinstance(raw, BaseException):
            raise raw
        return C.strict_json(raw)

    def _acknowledge(self, frame, plan, ledger, tokens, control, terminal):
        committed = P.verify_cancellation(frame, plan, ledger, tokens, control['request_id'], terminal)
        P.persisted_cancellation(self.native, committed)
        C.require(self.process.poll() is None, 'owner exited while publishing cancellation resolution')
        folder = self.evidence / 'request-cancellations'
        folder.mkdir(exist_ok=True)
        N.save_new(folder / (control['request_id'] + '.json'), dict(schema='nerkyor-client-cancellation-verified-v1',
            request_id=plan['request']['request_id'], session_id=ledger.session_id,
            native_pid=self.process.pid, resolution=committed, native_restarted=False, product_ready=False))
        self.last_completed = None
        ledger.poisoned = True
        return committed

    def generate_plan(self, ledger, plan, on_delta=None, cancel=None):
        C.require(ledger.template is self.template and not ledger.poisoned and not self.closed and not self.poisoned,
                  'foreign/retired recovery owner')
        C.require(plan['request']['session_id'] == ledger.session_id, 'plan belongs to another session')
        cancel = cancel if cancel is not None else threading.Event()
        if cancel.is_set():
            raise RequestCancelled()  # No wire write, no state-reset authority, prior session remains intact.
        tokens, decoder = [], P.Decoder(self.template.tokenizer)
        deadline = [time.monotonic() + self.turn_timeout]
        started = time.perf_counter()
        submitted = resolved = False
        control = terminal = None

        def signal():
            nonlocal control
            if cancel.is_set() and control is None:
                request = plan['request']
                value = dict(op='cancel_turn', request_id='cancel-' + uuid.uuid4().hex,
                             cancelled_request_id=request['request_id'], session_id=request['session_id'])
                self._write(value)
                control = value
                deadline[0] = time.monotonic() + self.stop_grace

        def emit(delta):
            if delta and on_delta is not None and not cancel.is_set():
                try:
                    on_delta(delta)
                except DISCONNECTS:
                    cancel.set()
            signal()

        try:
            submitted = True  # Partial/failed pipe writes are uncertain and cannot be recovered by a flag.
            self.last_completed = None
            self._write(dict(op='turn', request=plan['request']))
            while True:
                frame = self._read_request(deadline, signal)
                if frame.get('event') == 'token':
                    C.require(terminal is None, 'token after irreversible terminal')
                    P.verify_token(frame, plan, tokens, self.template.eos)
                    emit(decoder.push(tokens[-1]))
                elif frame.get('event') == 'terminal':
                    C.require(terminal is None, 'duplicate irreversible terminal')
                    terminal = P.verify_terminal(frame, plan, ledger, tokens)
                    P.persisted_receipt(self.native, plan['request']['request_id'],
                                        terminal['receipt_sha256'], terminal['receipt'])
                    emit(decoder.finish())
                    signal()
                    if control is not None:
                        continue  # Native will close the committed session; never pretend unpublished discard.
                    ledger.accept(plan, terminal, decoder.text)
                    self.last_completed = (plan['request']['request_id'], ledger.session_id, terminal['receipt_sha256'])
                    return dict(text=decoder.text, committed=terminal, input_tokens=len(plan['input_ids']),
                                output_tokens=len(tokens), client_wall_s=time.perf_counter()-started)
                elif frame.get('event') == 'cancelled':
                    C.require(control is not None, 'unsolicited cancellation authority')
                    resolution = self._acknowledge(frame, plan, ledger, tokens, control, terminal)
                    resolved = True
                    raise RequestCancelled(resolution)
                elif (frame.get('event') == 'rejected' and not tokens and terminal is None
                      and frame.get('request_id') == plan['request']['request_id']
                      and frame.get('session_id') == ledger.session_id and frame.get('state_unchanged') is True
                      and frame.get('owner_retired') is False and frame.get('product_ready') is False):
                    signal()
                    if control is not None:
                        continue  # A matched late cancel resolves an unadmitted request separately.
                    submitted = False
                    raise C.ProtocolError('native request rejected before mutation')
                else:
                    raise C.ProtocolError('unexpected/failed native request; owner cannot be reused')
        except BaseException:
            if submitted and not resolved:
                self.poisoned = ledger.poisoned = True
                self.abort()
            raise

    def cancel_completed(self, ledger, plan, committed):
        """Resolve a downstream disconnect after generate_plan returned a durable turn."""
        request = plan['request']
        C.require(self.last_completed == (request['request_id'], ledger.session_id, committed['receipt_sha256'])
                  and P.same_json(ledger.terminal, committed), 'late cancellation is not the last verified turn')
        control = dict(op='cancel_turn', request_id='cancel-' + uuid.uuid4().hex,
                       cancelled_request_id=request['request_id'], session_id=ledger.session_id)
        try:
            self._write(control)
            frame = self._read_request([time.monotonic()+self.stop_grace], lambda: None)
            resolution = self._acknowledge(frame, plan, ledger, committed['receipt']['output_token_ids'], control, committed)
            return resolution
        except BaseException:
            self.poisoned = ledger.poisoned = True
            self.abort()
            raise

    def close_session(self, ledger):
        super().close_session(ledger)
        self.last_completed = None