"""Independent, bounded Nerkyor HTTP owner using Augur's same-origin chat UI.

512-token research profile, greedy/no-thinking, original model template, F32
activation/KV, own DFlash2. Default cancellation retires this owner; opt-in
recovery-v2 retains weights only after a durable, request-bound cleanup proof.
Neither path auto-restarts an owner.
This is NOT the QUASAR product profile or a long-context qualification.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time
import uuid
import augur_product as W
from augur_startup import EngineLease
import nerkyor_chat as N
import nerkyor_protocol as P
import nerkyor_recovery as R
import qwen35_api as A
C = P.C
MODEL = 'nerkyor-coder390-dflash2-research'
MAX_BODY = 128 * 1024


def validate_request(payload, model):
    A.need(type(payload) is dict, 'A JSON object is required.')
    # Do not loosen the old controller's guards or route this model through its
    # sampling/tool/QPN paths. Explicitly reject unsupported knobs even if null.
    for name in ('tools', 'tool_choice', 'parallel_tool_calls', 'seed', 'top_k', 'min_p',
                 'enable_thinking', 'thinking_language', 'reasoning_effort'):
        A.need(name not in payload, 'Unsupported Nerkyor parameter: ' + name, name)
    if type(payload.get('messages')) is list:
        for message in payload['messages']:
            if type(message) is dict:
                A.need(message.get('role') != 'tool' and 'tool_calls' not in message,
                       'Nerkyor accepts text messages only; tool history is unsupported.', 'messages')
    request = A.validate_request(payload, model)
    budget = request['budget'] if request['budget'] is not None else 128
    A.need(type(budget) is int and 1 <= budget <= P.MAX_OUTPUT,
           'Nerkyor output ceiling must be 1..486; total context is 512 tokens.', 'max_completion_tokens')
    request['budget'] = budget
    return request


class Controller:
    max_body = MAX_BODY
    fast_finish = True
    request_recovery = False

    def __init__(self, engine, model=MODEL, metrics=None):
        self.engine, self.model, self.metrics = engine, model, metrics
        self.busy, self.closing = threading.Lock(), threading.Event()
        self.current, self.retired = None, False

    @property
    def faulted(self):
        return (self.retired or self.engine.poisoned or self.engine.closed or self.engine.cancel_requested.is_set()
                or self.engine.process.poll() is not None)

    def capabilities(self):
        value = dict(schema='nerkyor-bounded-text-http-v1', model=self.model, profile=P.PROFILE,
            product_ready=False, qualified=False, http_enabled=True,
            scope='bounded_research_bridge_not_full_product', generation='greedy',
            activation_dtype='F32', kv_dtype='F32', dflash_enabled=True, mtp_enabled=False,
            vision_forward=False, tool_calls=False, thinking=False, sampling=False,
            full_history_required=True, unknown_history='canonical_fresh_prefill_no_truncation',
            cancellation='retire_owner_manual_service_restart_required',
            request_bound_cancellation_discard=False, automatically_restarted=False,
            native_limits=dict(total_context_tokens=P.CAPACITY, initial_max_output_limit=P.MAX_OUTPUT),
            ui=dict(label='Nerkyor 27B · DFlash2', tools=False, thinking=False,
                    thinking_parameters=False,
                    notice='实验性：总上下文 512 tokens；greedy / no-thinking；取消生成会退役引擎，需手动重启服务。'))

        if self.request_recovery:
            value.update(schema='nerkyor-bounded-text-http-v2', cancellation='discard_session_keep_resident_weights',
                         request_bound_cancellation_discard=True, cancellation_commit_race='close_irreversible_terminal')
            value['ui']['notice'] = '实验性：总上下文 512 tokens；greedy / no-thinking；取消后清空本次会话状态，模型仍驻留，可继续提问。'
        return value

    def complete(self, payload, begin, emit, deliver, watch):
        request = validate_request(payload, self.model)
        if not self.busy.acquire(blocking=False):
            raise A.ApiError('One request is already active.', 409, 'single_user_busy')
        stop = None
        mutated = False
        request_cancel = threading.Event() if self.request_recovery else self.engine.cancel_requested
        event = dict(schema='nerkyor-http-request-v1', id='chatcmpl-' + uuid.uuid4().hex,
                     status='rejected', product_ready=False)
        started = time.perf_counter()
        try:
            if self.closing.is_set() or self.faulted:
                raise A.ApiError('Nerkyor owner is retired; manually restart this service.', 503, 'owner_retired')
            exact = (self.current is not None and not self.current.poisoned
                     and P.same_json(request['messages'][:-1], self.current.messages))
            # A non-message EOS cannot be appended; fresh canonical history is
            # explicit, never an attempt to reuse a foreign pending token.
            if exact and self.current.terminal['receipt']['stop_reason'] == 'eos':
                exact = self.current.terminal['receipt']['session']['pending_token'] == self.engine.template.end
            ledger = self.current if exact else P.Ledger(self.engine.template)
            try:
                plan = ledger.plan_messages(request['messages'], request['budget'])
            except C.ProtocolError as error:
                raise A.ApiError(str(error), 400, 'context_or_history_rejected', 'messages') from error
            mode = 'exact_history_append' if exact else ('fresh_history_reset' if self.current else 'fresh')
            meta = dict(id=event['id'], created=int(time.time()), model=self.model,
                        session_mode=mode, output_ceiling=plan['request']['max_new_tokens'])
            event.update(session_mode=mode, prompt_tokens=len(plan['input_ids']),
                         output_ceiling=meta['output_ceiling'])
            # Watcher callback only signals an Event; no concurrent stdout read,
            # stdin close, GPU teardown, or subprocess replacement is allowed.
            stop = watch(request_cancel.set if self.request_recovery else self.engine.request_cancel)
            if self.request_recovery and request_cancel.is_set():
                raise R.RequestCancelled()
            C.require(not self.engine.cancel_requested.is_set(), 'downstream cancelled before admission')
            begin(meta, request)
            mutated = True  # Includes uncertain writes and resetting a prior session.
            if not exact and self.current is not None:
                self.engine.close_session(self.current)
                self.current = None
            result = (self.engine.generate_plan(ledger, plan, emit, cancel=request_cancel) if self.request_recovery
                      else self.engine.generate_plan(ledger, plan, emit))
            # Fence a late disconnect before exposing verified completion/DONE.
            stop(); stop = None
            if self.request_recovery and request_cancel.is_set():
                raise R.RequestCancelled(self.engine.cancel_completed(ledger, plan, result['committed']))
            C.require(not self.engine.cancel_requested.is_set(), 'downstream cancelled before completion')
            receipt = result['committed']['receipt']
            completion = receipt['selected_tokens_including_eos']
            usage = dict(prompt_tokens=len(plan['input_ids']), completion_tokens=completion,
                total_tokens=len(plan['input_ids']) + completion,
                prompt_tokens_details=dict(cached_tokens=len(plan['request']['input']['exact_history']) if exact else 0))
            finish = 'stop' if receipt['stop_reason'] == 'eos' else 'length'
            self.current = ledger
            try:
                deliver(meta, request, result['text'], finish, usage)
            except R.DISCONNECTS:
                if self.request_recovery:
                    raise R.RequestCancelled(self.engine.cancel_completed(ledger, plan, result['committed']))
                raise
            event.update(status='completed', usage=usage,
                         receipt_sha256=result['committed']['receipt_sha256'],
                         request_id=plan['request']['request_id'])
            return event
        except R.RequestCancelled as error:
            if stop is not None:
                try:
                    stop(); stop = None
                except BaseException:
                    self.current, self.retired = None, True
                    self.engine.abort()
                    raise
            if not self.request_recovery:
                self.current, self.retired = None, True
                self.engine.abort()
                raise
            event.update(status='cancelled', request_id=plan['request']['request_id'], owner_reusable=not self.faulted,
                         native_request_submitted=error.resolution is not None)
            if error.resolution is not None:
                self.current = None
                event.update(recovery_outcome=error.resolution['receipt']['outcome'],
                             recovery_receipt_sha256=error.resolution['receipt_sha256'])
            raise
        except BaseException as error:
            if stop is not None:
                stop(); stop = None
            event.update(status='failed' if mutated else 'rejected', error_type=type(error).__name__)
            if mutated or self.engine.cancel_requested.is_set():
                self.current, self.retired = None, True
                self.engine.abort()  # This thread is the sole owner/queue consumer.
            raise
        finally:
            if stop is not None:
                stop()
            event['api_wall_s'] = time.perf_counter() - started
            if self.metrics is not None:
                try:
                    self.metrics.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')
                    self.metrics.flush()
                except (OSError, ValueError):
                    print('Nerkyor HTTP metrics write failed.', file=sys.stderr)
            self.busy.release()

    def close(self):
        self.closing.set()
        if self.busy.locked():
            self.engine.request_cancel()
        # Native cleanup is never performed concurrently with a request. Reads
        # wake within 50ms of the event and cooperative retirement is bounded.
        with self.busy:
            if self.engine.cancel_requested.is_set() or self.engine.poisoned:
                self.engine.abort()
            else:
                self.engine.close()
            self.current, self.retired = None, True


class RecoverableController(Controller):
    request_recovery = True


def watch_parent_stdin(on_stop, fd):
    """A raw descriptor read must not hold Python's buffered stdin lock.

    The parent may keep its pipe open while HTTP requests stop the service.
    A daemon blocked in sys.stdin.buffer.readline() can abort CPython during
    finalization. os.read releases the GIL and owns no buffered IO lock.
    Any input byte or EOF is the explicit parent stop signal.
    """
    def wait_parent():
        try:
            os.read(fd, 1)
        except OSError:
            pass
        on_stop()
    thread = threading.Thread(target=wait_parent, name='nerkyor-parent-stop', daemon=True)
    thread.start()
    return thread


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('producer', 'model-root', 'draft-root', 'manifest', 'evidence-root'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--producer-sha256', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--startup-timeout', type=float, default=180)
    parser.add_argument('--turn-timeout', type=float, default=180)
    parser.add_argument('--stop-on-stdin', action='store_true')
    parser.add_argument('--request-cancel-recovery', action='store_true', help='Requires the pinned recovery-v2 native producer')
    parser.add_argument('--open-browser', action='store_true', help='Open same-origin UI; credentials stay in URL fragment, never logs')
    args = parser.parse_args(argv)
    key = os.environ.get('NERKYOR_API_KEY', '')
    A.validate_key(key)  # Before tokenizer/GPU IO; never print or persist the key.
    C.require(0 <= args.port <= 65535, 'invalid port')
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
    evidence = args.evidence_root.resolve()
    engine = server = controller = None
    with EngineLease():
        evidence.mkdir(parents=True, exist_ok=False)
        with (evidence / 'http-metrics.jsonl').open('x', encoding='utf-8') as metrics:
            try:
                engine_type = R.Engine if args.request_cancel_recovery else N.Engine
                controller_type = RecoverableController if args.request_cancel_recovery else Controller
                engine = engine_type(producer=args.producer, producer_sha256=args.producer_sha256,
                    model_root=args.model_root, draft_root=args.draft_root, manifest=args.manifest,
                    manifest_sha256=args.manifest_sha256, evidence_root=evidence / 'engine',
                    startup_timeout=args.startup_timeout, turn_timeout=args.turn_timeout)
                controller = controller_type(engine, metrics=metrics)
                # Reuse authenticated loopback HTTP/SSE and exact same-origin
                # static assets, never the QUASAR Controller or native adapter.
                server = W.Server(('127.0.0.1', args.port), controller, key)
                if args.stop_on_stdin:
                    watch_parent_stdin(server.request_stop, sys.stdin.fileno())
                ready = dict(event='nerkyor_http_listening', port=server.server_address[1], model=MODEL,
                    profile=P.PROFILE, product_ready=False, producer_sha256=args.producer_sha256,
                    manifest_sha256=args.manifest_sha256, cancellation=controller.capabilities()['cancellation'],
                    capabilities=controller.capabilities())
                N.save_new(evidence / 'listening.json', ready)
                print(json.dumps(ready, ensure_ascii=False), flush=True)
                if args.open_browser:
                    import webbrowser
                    from urllib.parse import quote
                    # Fragments are not sent in HTTP. No capability URL is
                    # printed, recorded in evidence, or returned to API clients.
                    webbrowser.open('http://127.0.0.1:' + str(server.server_address[1]) + '/#key=' + quote(key, safe=''))
                try:
                    server.serve_forever(poll_interval=.1)
                except KeyboardInterrupt:
                    pass
            finally:
                if server is not None:
                    server.server_close()
                if controller is not None:
                    controller.close()
                elif engine is not None:
                    engine.abort()
                if engine is not None:
                    N.save_new(evidence / 'http-stopped.json', dict(product_ready=False,
                        native_exit_confirmed=engine.process.poll() is not None,
                        native_exit_code=engine.process.poll(), forced_termination=engine.forced,
                        allocation_closure_proved=engine.cleanup is not None, automatically_restarted=False))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (C.ProtocolError, OSError, ValueError, RuntimeError) as error:
        # No request bodies, credentials, native command lines or private URLs.
        print('Nerkyor HTTP failed: ' + type(error).__name__, file=sys.stderr)
        raise SystemExit(2)
