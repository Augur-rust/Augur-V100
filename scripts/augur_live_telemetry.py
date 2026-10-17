"""Optional token-count observer for the Windows launcher; never stores text or token IDs."""
import collections
import json
import os
from pathlib import Path
import threading
import time


class Tracker:
    def __init__(self, destination):
        self.destination = Path(destination)
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.samples = collections.deque(maxlen=4096)
        self.started = None
        self.ended = None
        self.first = None
        self.count = 0
        self.request_id = None
        self.status = 'idle'
        self.decode_mode = 'unknown'
        self.sampling_preset = None
        self.input_tokens = self.capacity = self.committed_used = self.consumed = None
        self.closed = threading.Event()
        self.thread = threading.Thread(target=self.publish, daemon=True)
        self.thread.start()

    def start(self, request_id, input_tokens=None, capacity=None, sampling=None):
        with self.lock:
            self.request_id, self.status = request_id, 'prefill'
            preset = sampling.get('preset') if isinstance(sampling, dict) else None
            self.sampling_preset = preset if preset in ('greedy', 'ninfer_thinking_v1') else None
            self.decode_mode = 'sampling' if preset == 'ninfer_thinking_v1' else 'greedy' if preset == 'greedy' else 'unknown'
            self.started, self.first, self.count = time.monotonic(), None, 0
            self.ended = None
            self.samples.clear()
            valid = type(input_tokens) is int and type(capacity) is int and 0 <= input_tokens < capacity
            self.input_tokens = input_tokens if valid else None
            self.capacity = capacity if valid else None
            self.committed_used = self.consumed = None

    def token(self):
        now = time.monotonic()
        with self.lock:
            if self.first is None:
                self.first = now
                self.samples.append((now, 0))
            self.count += 1
            self.status = 'decode'
            self.samples.append((now, self.count))

    def end(self, status, terminal=None):
        with self.lock:
            self.status = status
            self.ended = time.monotonic()
            session = (terminal or {}).get('session') or {}
            history = session.get('consumed_token_ids')
            if status == 'completed' and isinstance(history, list) and type(session.get('pending_token')) is int:
                self.consumed = len(history)
                self.committed_used = self.consumed + 1

    def snapshot(self):
        now = time.monotonic()
        with self.lock:
            while len(self.samples) > 1 and self.samples[1][0] < now - 5:
                self.samples.popleft()
            active = self.status in ('prefill', 'decode')
            rolling = None
            if self.samples and active:
                tick, count = self.samples[0]
                span = now - tick
                if span >= .25:
                    rolling = (self.count - count) / span
            return dict(schema=1, producer_pid=os.getpid(), updated_unix_s=time.time(),
                        request_id=self.request_id, status=self.status, output_tokens=self.count,
                        decode_mode=self.decode_mode, sampling_preset=self.sampling_preset,
                        elapsed_s=(self.ended if self.ended is not None else now)-self.started if self.started is not None else None,
                        ttft_s=self.first-self.started if self.first is not None else None,
                        rolling_5s_tps=rolling,
                        context_capacity_tokens=self.capacity, input_tokens=self.input_tokens,
                        context_used_tokens=(self.committed_used if self.committed_used is not None else
                            self.input_tokens + self.count if active and self.input_tokens is not None else None),
                        context_consumed_tokens=self.consumed,
                        context_clock='input_plus_validated_output_pending_included_at_commit', 
                        clock='validated_native_non_eos_tokens_at_http_frame_observer',
                        text_or_token_ids_included=False)

    def publish(self):
        while not self.closed.is_set():
            try:
                temp = self.destination.with_suffix('.tmp')
                temp.write_text(json.dumps(self.snapshot(), allow_nan=False), encoding='utf8')
                os.replace(temp, self.destination)
            except OSError:
                # Monitoring IO cannot make a model request fail.
                pass
            self.closed.wait(.5)


class ObservedTransport:
    def __init__(self, transport, tracker):
        self.transport, self.tracker = transport, tracker

    def __getattr__(self, name):
        return getattr(self.transport, name)

    def frames(self, path, payload):
        stream = self.transport.frames(path, payload)
        try:
            for frame, elapsed in stream:
                yield frame, elapsed
                # Resumption means the controller accepted and processed this frame.
                if frame.get('event') == 'token':
                    self.tracker.token()
        finally:
            close = getattr(stream, 'close', None)
            if close is not None:
                close()


def install(destination):
    """Observe the selected public controller without changing the frozen runtime files."""
    import augur_api
    tracker = Tracker(destination)
    original = augur_api.Controller.run_native_turn
    if getattr(original, '_augur_observed', False):
        raise RuntimeError('Telemetry observer already installed')

    def observed(self, transport, ledger, plan, emit, ceiling):
        tracker.start(plan['payload']['request_id'], len(plan['input_ids']), self.limits.get('capacity'), self.health.get('sampling'))
        try:
            result = original(self, ObservedTransport(transport, tracker), ledger, plan, emit, ceiling)
        except BaseException:
            tracker.end('failed_or_cancelled')
            raise
        else:
            tracker.end('completed', result[0])
            return result
    observed._augur_observed = True
    augur_api.Controller.run_native_turn = observed
    return tracker
