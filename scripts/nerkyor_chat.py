"""Pinned local text CLI for the independent Nerkyor + DFlash2 owner.

Experimental: 512 tokens total, greedy, explicit no-thinking template, F32
activation/KV. No HTTP listener, model download, MTP, sampling or auto-restart.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import uuid
import nerkyor_protocol as P
C = P.C


def save_new(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write('\n'); file.flush(); os.fsync(file.fileno())


def producer_environment(environment):
    # The explicit model profile has no inherited QUASAR policy/calibration.
    result = {k: v for k, v in environment.items()
              if not k.upper().startswith(('AUGUR_', 'QWEN35_')) and k.upper() != 'NERKYOR_API_KEY'}
    result['AUGUR_SHARED_HEAD_KERNEL'] = 'llama'
    return result


class Engine:
    def __init__(self, *, producer, producer_sha256, model_root, draft_root, manifest,
                 manifest_sha256, evidence_root, startup_timeout=180, turn_timeout=180, stop_grace=60):
        for seconds in (startup_timeout, turn_timeout, stop_grace):
            C.require(type(seconds) in (int, float) and math.isfinite(seconds) and 0 < seconds <= 86400, 'invalid timeout')
        C.require(C.digest_string(producer_sha256) and P.file_sha(producer) == producer_sha256, 'producer executable digest mismatch')
        self.template = P.Template(model_root, manifest, manifest_sha256)
        self.evidence = Path(evidence_root).resolve()
        self.evidence.mkdir(parents=True, exist_ok=False)
        self.native = self.evidence / 'native'
        self.frames, self.reader_stop = queue.Queue(maxsize=64), threading.Event()
        self.cancel_requested = threading.Event()
        self.process, self.reader, self.stderr_file = None, None, None
        self.turn_timeout, self.stop_grace = turn_timeout, stop_grace
        self.poisoned, self.closed, self.forced = False, False, False
        self.cleanup, self.ready = None, None
        self.producer_sha256 = producer_sha256
        try:
            self.stderr_file = (self.evidence / 'native-stderr.log').open('xb')
            command = [str(Path(producer).resolve()), '--model-root', str(Path(model_root).resolve()),
                       '--draft-root', str(Path(draft_root).resolve()), '--manifest', str(Path(manifest).resolve()),
                       '--manifest-sha256', manifest_sha256, '--evidence-root', str(self.native)]
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self.stderr_file, env=producer_environment(os.environ),
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            self.reader = threading.Thread(target=self._pump, name='nerkyor-stdout', daemon=True)
            self.reader.start()
            frame = self._read(time.monotonic() + startup_timeout)
            self._verify_ready(frame)
            C.require(P.same_json(C.strict_json(P.read_bounded(self.native / 'ready.json')), frame), 'ready receipt differs from stream')
            self.ready = frame
        except BaseException:
            self.abort()
            raise

    def _verify_ready(self, frame):
        P.verify_ready(frame, self.template)

    def _queue(self, item):
        while not self.reader_stop.is_set():
            try:
                self.frames.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def _pump(self):
        try:
            while not self.reader_stop.is_set():
                raw = self.process.stdout.readline(P.MAX_FRAME + 1)
                if not raw:
                    self._queue(None)
                    return
                if len(raw) > P.MAX_FRAME or not raw.endswith(b'\n'):
                    raise C.ProtocolError('invalid/oversized native frame')
                self._queue(raw)
        except BaseException as error:
            self._queue(error)

    def _read(self, deadline):
        # Only the execution thread reads frames or retires the subprocess.
        # HTTP disconnect watchers may signal, but may not drain this queue.
        while True:
            C.require(not self.cancel_requested.is_set(), 'Nerkyor request cancelled; owner retirement required')
            remaining = deadline - time.monotonic()
            C.require(remaining > 0, 'Nerkyor native response timeout')
            try:
                raw = self.frames.get(timeout=min(remaining, 0.05))
                break
            except queue.Empty:
                continue
        C.require(not self.cancel_requested.is_set(), 'Nerkyor request cancelled; owner retirement required')
        C.require(raw is not None, 'native owner exited before a complete response')
        if isinstance(raw, BaseException):
            raise C.ProtocolError('native frame reader failed') from raw
        return C.strict_json(raw)

    def request_cancel(self):
        """Thread-safe signal only. The executing owner performs all cleanup."""
        self.cancel_requested.set()

    def _write(self, value):
        C.require(not self.cancel_requested.is_set(), 'Nerkyor owner cancellation requested')
        C.require(not self.closed and not self.poisoned and self.process.poll() is None, 'Nerkyor owner is not usable')
        raw = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8') + b'\n'
        C.require(len(raw) <= 64 * 1024, 'native command byte limit')
        self.process.stdin.write(raw); self.process.stdin.flush()

    def generate(self, ledger, text, max_new_tokens=128, on_delta=None):
        C.require(ledger.template is self.template and not self.closed and not self.poisoned, 'foreign/retired text owner')
        plan = ledger.plan_user(text, max_new_tokens)  # CPU-only rejection must leave owner usable.
        return self.generate_plan(ledger, plan, on_delta)

    def generate_plan(self, ledger, plan, on_delta=None):
        C.require(ledger.template is self.template and not ledger.poisoned
                  and not self.closed and not self.poisoned, 'foreign/retired text owner')
        C.require(plan['request']['session_id'] == ledger.session_id, 'plan belongs to a different session')
        tokens, decoder = [], P.Decoder(self.template.tokenizer)
        submitted = False
        started = time.perf_counter()
        try:
            submitted = True  # A failed/partial pipe write is uncertain too.
            self._write(dict(op='turn', request=plan['request']))
            deadline = time.monotonic() + self.turn_timeout
            while True:
                frame = self._read(deadline)
                if frame.get('event') == 'token':
                    P.verify_token(frame, plan, tokens, self.template.eos)
                    delta = decoder.push(tokens[-1])
                    if delta and on_delta is not None:
                        on_delta(delta)
                elif frame.get('event') == 'terminal':
                    committed = P.verify_terminal(frame, plan, ledger, tokens)
                    P.persisted_receipt(self.native, plan['request']['request_id'],
                                        committed['receipt_sha256'], committed['receipt'])
                    delta = decoder.finish()
                    if delta and on_delta is not None:
                        on_delta(delta)
                    ledger.accept(plan, committed, decoder.text)
                    return dict(text=decoder.text, committed=committed, input_tokens=len(plan['input_ids']),
                                output_tokens=len(tokens), client_wall_s=time.perf_counter()-started)
                elif (frame.get('event') == 'rejected' and not tokens
                      and frame.get('request_id') == plan['request']['request_id']
                      and frame.get('session_id') == ledger.session_id and frame.get('state_unchanged') is True
                      and frame.get('owner_retired') is False and frame.get('product_ready') is False):
                    submitted = False
                    raise C.ProtocolError('native request rejected before mutation')
                else:
                    raise C.ProtocolError('unexpected/failed native turn; no text completion committed')
        except BaseException:
            if submitted:
                self.poisoned = ledger.poisoned = True
                self.abort()
            raise

    def close_session(self, ledger):
        C.require(ledger.template is self.template and not ledger.poisoned, 'foreign/uncertain text ledger')
        if ledger.terminal is None:
            ledger.poisoned = True
            return
        rid, sid = 'close-' + uuid.uuid4().hex, ledger.session_id
        revision = ledger.terminal['receipt']['session']['revision']
        try:
            self._write(dict(op='close', request_id=rid, session_id=sid, revision=revision))
            frame = self._read(time.monotonic() + self.turn_timeout)
            C.require(frame.get('event') == 'closed' and frame.get('request_id') == rid and frame.get('session_id') == sid
                      and type(frame.get('revision')) is int and frame['revision'] == revision
                      and frame.get('product_ready') is False, 'invalid close/reset acknowledgement')
            r = P.persisted_receipt(self.native, rid, frame.get('receipt_sha256'))
            C.require(r.get('schema') == 'nerkyor-session-close-v1' and r.get('request_id') == rid
                      and r.get('session_id') == sid and type(r.get('revision')) is int and r['revision'] == revision
                      and r.get('manifest_sha256') == self.template.manifest_sha256
                      and r.get('last_turn_receipt_sha256') == ledger.terminal['receipt_sha256']
                      and r.get('reset_completed') is True, 'invalid durable close receipt')
            ledger.poisoned = True
        except BaseException:
            self.poisoned = ledger.poisoned = True
            self.abort()
            raise

    def _observe_cleanup(self, frame, normal):
        P.verify_stopped(frame, self.template.manifest_sha256, normal, P.session_settings(self.template)['profile'])
        C.require(P.same_json(C.strict_json(P.read_bounded(self.native / 'owner-stopped.json')), frame),
                  'owner cleanup receipt differs from stream')
        self.cleanup = frame

    def _finish(self, normal):
        self.reader_stop.set()
        if self.reader is not None:
            self.reader.join(timeout=2)
        if self.process is not None:
            for pipe in (self.process.stdin, self.process.stdout):
                if pipe is not None:
                    try: pipe.close()
                    except (OSError, ValueError): pass
        if self.stderr_file is not None:
            self.stderr_file.close()
        self.closed = True
        save_new(self.evidence / 'client-lifecycle.json', dict(schema='nerkyor-cli-lifecycle-v1',
            profile=P.session_settings(self.template)['profile'], producer_sha256=self.producer_sha256, manifest_sha256=self.template.manifest_sha256,
            child_pid=self.process.pid if self.process is not None else None,
            child_exit_code=self.process.poll() if self.process is not None else None,
            child_exit_confirmed=self.process is None or self.process.poll() is not None,
            normal_shutdown=normal, forced_termination=self.forced, allocation_closure_proved=self.cleanup is not None,
            automatically_restarted=False, product_ready=False))

    def close(self):
        if self.closed:
            return
        if self.poisoned:
            self.abort()
            return
        try:
            self._write(dict(op='shutdown'))
            self._observe_cleanup(self._read(time.monotonic() + self.stop_grace), normal=True)
            C.require(self.process.wait(timeout=self.stop_grace) == 0, 'native shutdown returned failure')
            self._finish(normal=True)
        except BaseException:
            self.abort()
            raise

    def abort(self):
        if self.closed:
            return
        self.poisoned = True
        if self.process is not None:
            try: self.process.stdin.close()
            except (OSError, ValueError, AttributeError): pass
            deadline = time.monotonic() + self.stop_grace
            # Drain output while waiting, so a bounded reader queue cannot trap
            # the child in a blocked stdout write during cooperative retirement.
            while time.monotonic() < deadline:
                try:
                    raw = self.frames.get(timeout=0.1)
                    if isinstance(raw, bytes):
                        frame = C.strict_json(raw)
                        if frame.get('event') == 'owner_stopped':
                            self._observe_cleanup(frame, normal=False)
                except (queue.Empty, C.ProtocolError, OSError, ValueError):
                    pass
                if self.process.poll() is not None and self.frames.empty() and (self.reader is None or not self.reader.is_alive()):
                    break
            if self.process.poll() is None:
                self.forced = True
                self.process.kill()
            self.process.wait(timeout=10)
        self._finish(normal=False)

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        if kind is None:
            self.close()
        else:
            self.abort()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['producer', 'model-root', 'draft-root', 'manifest', 'evidence-root']:
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--producer-sha256', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    prompts = parser.add_mutually_exclusive_group(required=True)
    prompts.add_argument('--prompt', action='append', help='May repeat for exact-history multi-turn generation')
    prompts.add_argument('--interactive', action='store_true', help='/reset or /quit; Ctrl+C retires the owner')
    parser.add_argument('--system')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--json', action='store_true', help='One verified completion JSON per turn; no provisional text')
    parser.add_argument('--startup-timeout', type=float, default=180)
    parser.add_argument('--turn-timeout', type=float, default=180)
    args = parser.parse_args(argv)
    C.require(1 <= args.max_new_tokens <= 3891, 'max-new-tokens outside supported profiles')
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
    with Engine(producer=args.producer, producer_sha256=args.producer_sha256,
                model_root=args.model_root, draft_root=args.draft_root, manifest=args.manifest,
                manifest_sha256=args.manifest_sha256, evidence_root=args.evidence_root,
                startup_timeout=args.startup_timeout, turn_timeout=args.turn_timeout) as engine:
        ledger = P.Ledger(engine.template, args.system)
        def generate(text):
            delta = None if args.json else lambda text: print(C.terminal_text(text), end='', flush=True)
            result = engine.generate(ledger, text, args.max_new_tokens, delta)
            if args.json:
                print(json.dumps(dict(event='completion', profile=engine.template.profile, product_ready=False, **result), ensure_ascii=False, allow_nan=False), flush=True)
            else:
                print(flush=True)
        if args.prompt is not None:
            for prompt in args.prompt:
                generate(prompt)
        else:
            while True:
                try:
                    text = input('you> ')
                except EOFError:
                    break
                if text == '/quit':
                    break
                if text == '/reset':
                    engine.close_session(ledger)
                    ledger = P.Ledger(engine.template, args.system)
                    continue
                if not text.strip():
                    continue
                try:
                    generate(text)
                except C.ProtocolError as error:
                    if engine.poisoned:
                        raise
                    print('Request rejected: ' + C.terminal_text(str(error)), file=sys.stderr)
        engine.close_session(ledger)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nCancelled; the native owner was retired.', file=sys.stderr)
        raise SystemExit(130)
    except (C.ProtocolError, OSError, ValueError) as error:
        print('Nerkyor: ' + C.terminal_text(str(error)), file=sys.stderr)
        raise SystemExit(2)
