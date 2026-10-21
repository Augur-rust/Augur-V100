"""Streaming terminal client for the Rust resident Qwen session service.

No CUDA/model process is created here. The service owns inference and KV state;
this client retains its exact consumed-token ledger, never reconstructed text.
Requires tokenizers and jinja2. Run with --help for the bounded current scope.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
from pathlib import Path
import re
import struct
import sys
import time
from urllib.parse import urlsplit
import uuid


# Object-count safety bound, not a conversation-turn budget. HTTP byte and
# token-context limits remain authoritative; never truncate messages to fit.
MAX_MESSAGES = 65_536
MAX_FRAME = 1_048_576
MAX_RESPONSE = 16 * 1024 * 1024
FRAME_CLOCK = 'http_read1_completion_perf_counter_same_timestamp_per_read'
SCHEMA = 'qwen35-committed-text-stream-v1'
# These are advertised native protocol profiles, not qualification claims.
PROFILES = {
    'resident': (1, 196608 - 1030, 196608),
    'growing': (1, 8192, 16384),
    'canonical8k': (8192, 8192, 16384),
    'context32k': (32768, 32768, 36864),
    'context64k': (65536, 65536, 69632),
    'near192k': (194560, 194560, 196608),
}


class ProtocolError(ValueError):
    pass


class GenerationFailure(ProtocolError):
    """Allowlisted public errors and numeric diagnostics; never include user text."""
    definitions = {
        'native_response_limit': (502, 'Native response exceeded its transport byte limit; no completion was committed.'),
        'reasoning_repetition': (422, 'Repeated reasoning detected; generation was stopped after a long exact repetition. No completion was committed.'),
    }

    def __init__(self, code, **counters):
        status, message = self.definitions[code]
        if any(type(value) is not int or value < 0 for value in counters.values()):
            raise ValueError('generation diagnostics must be nonnegative integer counts')
        self.code, self.status, self.counters = code, status, counters
        super().__init__(message)


def require(ok, detail):
    if not ok:
        raise ProtocolError(detail)


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result

    def constant(word):
        raise ProtocolError('non-finite JSON: ' + word)

    result = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    require(type(result) is dict, 'JSON object required')
    return result


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def digest_string(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def token_hash(ids):
    require(type(ids) is list and all(type(i) is int and 0 <= i < 2**32 for i in ids),
            'invalid u32 token ledger')
    return sha256(b'QWEN35-SESSION-CONSUMED-U32LE-V1\0' +
                  b''.join(struct.pack('<I', i) for i in ids))


def terminal_text(text):
    """Display untrusted model text without executing terminal control sequences."""
    return ''.join(c if c in ('\n', '\t') or (ord(c) >= 32 and not 127 <= ord(c) <= 159)
                   else f'\\x{ord(c):02x}' for c in text)


class ChatTemplate:
    """Use the supplied model template, including a full-transcript round trip."""
    def __init__(self, model_dir):
        from jinja2 import StrictUndefined
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        from tokenizers import Tokenizer

        root = Path(model_dir)
        paths = {key: root / name for key, name in (
            ('tokenizer', 'tokenizer.json'), ('template', 'chat_template.jinja'),
            ('generation', 'generation_config.json'))}
        raw = {key: path.read_bytes() for key, path in paths.items()}
        self.identity = {key + '_sha256': sha256(value) for key, value in raw.items()}
        self.tokenizer = Tokenizer.from_str(raw['tokenizer'].decode('utf-8'))
        eos = strict_json(raw['generation'])['eos_token_id']
        eos = eos if isinstance(eos, list) else [eos]
        require(eos and all(type(t) is int and t >= 0 for t in eos), 'invalid EOS IDs')
        self.eos = set(eos)
        self.end = self.tokenizer.token_to_id('<|im_end|>')
        require(type(self.end) is int and self.end in self.eos, 'message-end/EOS mismatch')
        self.vocab = set(self.tokenizer.get_vocab().values())
        env = ImmutableSandboxedEnvironment(undefined=StrictUndefined,
                                            trim_blocks=True, lstrip_blocks=True)

        def fail(message):
            raise ProtocolError('chat template rejected: ' + str(message))

        env.globals['raise_exception'] = fail
        self.template = env.from_string(raw['template'].decode('utf-8'))

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def render(self, messages, tools=None, add_generation_prompt=True, enable_thinking=False, reasoning_effort=None):
        # Public text-only histories legitimately omit optional assistant
        # fields. StrictUndefined must not turn those messages into a template
        # error; keep private defaults local and never mutate the transcript.
        rendered_messages = [dict({'reasoning_content': '', 'tool_calls': []}, **message)
                             for message in messages]
        effort = {} if reasoning_effort is None else dict(reasoning_effort=reasoning_effort)
        return self.template.render(messages=rendered_messages, tools=tools, enable_thinking=enable_thinking,
                                    preserve_thinking=True, add_generation_prompt=add_generation_prompt,
                                    add_vision_id=False, **effort)

    @staticmethod
    def message(role, text):
        require(isinstance(text, str) and text.strip(), 'nonempty text required')
        require(not re.search(r'<\|[^<>\r\n]*\|>|</?think>', text),
                'literal model control tokens are not accepted in text chat')
        return dict(role=role, content=text, reasoning_content='', tool_calls=[])


class Ledger:
    def __init__(self, template, profile='growing', system=None):
        require(profile in PROFILES, 'unsupported server profile')
        self.template, self.profile = template, profile
        self.messages = [] if system is None else [template.message('system', system)]
        self.session_id = 'chat-' + uuid.uuid4().hex
        self.history, self.terminal = [], None
        self.prefix, self.last_text, self.poisoned = '', '', False

    def plan(self, text, output_reserve=134):
        require(type(output_reserve) is int and output_reserve >= 134, 'invalid continuation output reserve')
        require(not self.poisoned, 'session is uncertain; start an explicit new session')
        message = self.template.message('user', text)
        messages = self.messages + [message]
        full = self.template.render(messages)
        rid = 'turn-' + uuid.uuid4().hex
        lo, hi, ceiling = PROFILES[self.profile]
        if self.terminal is None:
            ids = self.template.encode(full)
            require(lo <= len(ids) <= hi, f'{self.profile} initial input requires {lo}..{hi} tokens')
            require(len(full.encode('utf-8')) <= 4 * 1024 * 1024, 'prompt exceeds native byte limit')
            payload = dict(request_id=rid, session_id=self.session_id, prompt=full)
            path = '/v1/session/open'
        else:
            terminal = self.terminal
            pending = terminal['session']['pending_token']
            if terminal['stop_reason'] == 'eos':
                require(pending == self.template.end, 'non-message EOS cannot be continued')
            delivered = self.prefix + self.last_text
            if terminal['stop_reason'] == 'eos':
                delivered += '<|im_end|>'
            require(full.startswith(delivered), 'template rewrote committed conversation history')
            suffix = full[len(delivered):]
            suffix_ids = self.template.encode(suffix)
            require(0 < len(suffix_ids) <= 2048 and len(suffix.encode('utf-8')) <= 16384,
                    'follow-up exceeds native suffix limit (2048 tokens / 16 KiB)')
            ids = self.history + [pending] + suffix_ids
            require(len(ids) + output_reserve <= ceiling, 'session has insufficient continuation reserve')
            payload = dict(request_id=rid, session_id=self.session_id,
                           revision=terminal['session']['revision'],
                           exact_history=self.history.copy(), suffix=suffix)
            path = '/v1/session/append'
        return dict(path=path, payload=payload, input_ids=ids, messages=messages, prefix=full)

    def accept(self, plan, terminal, text):
        require(not self.poisoned, 'cannot publish an uncertain session')
        meta = terminal['session']
        visible, pending = terminal['visible_token_ids'], meta['pending_token']
        require(meta.get('session_id') == self.session_id and meta.get('capacity') == 196608,
                'terminal session/capacity mismatch')
        require(type(meta.get('revision')) is int and meta['revision'] >
                (self.terminal['session']['revision'] if self.terminal else 0), 'stale revision')
        require(meta.get('pending_token_published') is True and type(pending) is int
                and pending in self.template.vocab, 'pending token missing or invalid')
        require(digest_string(meta.get('checkpoint_sha256')), 'missing checkpoint hash')
        require(all(t not in self.template.eos for t in visible), 'visible EOS token')
        if terminal['stop_reason'] == 'eos':
            require(pending in self.template.eos, 'EOS pending mismatch')
            history = plan['input_ids'] + visible
        else:
            require(terminal['stop_reason'] == 'max_new_tokens' and visible
                    and visible[-1] == pending, 'output budget/pending mismatch')
            history = plan['input_ids'] + visible[:-1]
        require(type(meta.get('consumed_token_count')) is int
                and meta['consumed_token_count'] == len(history), 'consumed token count mismatch')
        require(meta.get('consumed_input_digest') == token_hash(history), 'consumed token hash mismatch')
        require(len(history) <= PROFILES[self.profile][2], 'history exceeds server profile')
        # Publish all client state only after complete HTTP framing and terminal validation.
        self.history, self.terminal = history, terminal
        self.prefix, self.last_text = plan['prefix'], text
        self.messages = plan['messages'] + [dict(role='assistant', content=text,
                                                reasoning_content='', tool_calls=[])]


class StreamVerifier:
    def __init__(self, request_id, template):
        self.request_id, self.template = request_id, template
        self.sequence, self.identity = 0, None
        self.tokens, self.parts, self.terminal = [], [], None
        self.flush_supported, self.final_flush = False, None

    def push(self, frame):
        require(self.terminal is None, 'frame after terminal')
        require(frame.get('event') != 'rejected', 'server rejected request: ' + str(frame.get('detail', '')))
        require(frame.get('schema') == SCHEMA and frame.get('request_id') == self.request_id,
                'foreign frame schema/request')
        require(type(frame.get('sequence')) is int and frame['sequence'] == self.sequence,
                'out-of-order frame')
        if self.sequence == 0:
            require(frame.get('event') == 'begin' and digest_string(frame.get('request_sha256')),
                    'missing begin identity')
            require(frame.get('tokenizer_sha256') == self.template.identity['tokenizer_sha256'],
                    'server/client tokenizer mismatch')
            self.identity = frame['request_sha256']
            self.flush_supported = frame.get('terminal_text_flush_supported', False)
            require(type(self.flush_supported) is bool, 'invalid terminal text-flush capability')
        else:
            require(frame.get('request_sha256') == self.identity, 'request identity changed')
            if frame.get('event') == 'text':
                require(self.final_flush is None, 'text after terminal decode flush')
                ids, text = frame.get('token_ids'), frame.get('text')
                require(type(ids) is list and ids and all(type(t) is int and t in self.template.vocab
                        and t not in self.template.eos for t in ids), 'invalid visible token IDs')
                require(isinstance(text, str) and text, 'invalid visible text')
                flush = frame.get('terminal_decode_flush')
                if 'terminal_decode_flush' in frame:
                    require(self.flush_supported and type(flush) is str
                            and flush in ('eos', 'max_new_tokens'), 'invalid terminal decode flush')
                    try:
                        decoded = self.template.tokenizer.decode(self.tokens + ids, skip_special_tokens=True)
                    except Exception as error:
                        raise ProtocolError('terminal token decode failed') from error
                    require(decoded == ''.join(self.parts) + text, 'terminal decode does not match exact token IDs')
                    self.final_flush = flush
                else:
                    require('\ufffd' not in text, 'unbound replacement text')
                self.tokens.extend(ids)
                self.parts.append(text)
            else:
                require(frame.get('event') == 'terminal' and isinstance(frame.get('session'), dict),
                        'unexpected stream event')
                require(self.final_flush is None or frame.get('stop_reason') == self.final_flush,
                        'terminal stop differs from text flush')
                ids = frame.get('visible_token_ids')
                require(type(ids) is list and all(type(t) is int for t in ids)
                        and ids == self.tokens, 'terminal visible token mismatch')
                require(frame.get('text_sha256') == sha256(''.join(self.parts).encode('utf-8')),
                        'terminal text hash mismatch')
                require(digest_string(frame.get('native_report_sha256')), 'missing native report binding')
                self.terminal = frame
        self.sequence += 1

    def finish(self):
        require(self.terminal is not None, 'HTTP stream ended without terminal; no history committed')
        return self.terminal, ''.join(self.parts)


class Transport:
    max_frame = MAX_FRAME
    max_response = MAX_RESPONSE

    def __init__(self, url, timeout=1800, trace_root=None):
        parsed = urlsplit(url)
        require(parsed.scheme == 'http' and parsed.hostname in ('127.0.0.1', 'localhost', '::1')
                and not parsed.username and not parsed.password and not parsed.query
                and not parsed.fragment and parsed.path in ('', '/'),
                'use a loopback HTTP URL; this service has no remote authentication/TLS')
        require(math.isfinite(timeout) and timeout > 0, 'timeout must be positive and finite')
        self.host, self.port = parsed.hostname, parsed.port or 80
        self.timeout, self.trace_root = timeout, trace_root

    def _connect(self, method, path, payload=None):
        conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode('utf-8')
        require(body is None or len(body) <= 8 * 1024 * 1024, 'HTTP request exceeds native bound')
        try:
            conn.request(method, path, body=body, headers={
                'Content-Type': 'application/json', 'Connection': 'close'})
            response = conn.getresponse()
            if response.status != 200:
                detail = response.read(8192).decode('utf-8', errors='replace')
                raise ProtocolError(f'HTTP {response.status}: {detail}')
            return conn, response
        except BaseException:
            conn.close()
            raise

    def health(self):
        conn, response = self._connect('GET', '/health')
        try:
            raw = response.read(65537)
            require(len(raw) <= 65536, 'oversized health response')
            result = strict_json(raw)
            require(result.get('profile') in PROFILES, 'server advertises an unsupported profile')
            return result
        finally:
            response.close()
            conn.close()

    def _native_frames(self, response, started):
        # HTTPResponse.readline uses peek(), which can turn IncompleteRead into
        # EOF. Parse the native adapter's bounded chunk framing explicitly so a
        # terminal JSON object without the final 0 CRLF CRLF cannot commit state.
        reader = response.fp
        buffered, wire_offset, arrived = b'', 0, None

        def read_part(n):
            nonlocal buffered, wire_offset, arrived
            remaining = self.timeout - (time.perf_counter() - started)
            require(remaining > 0, 'request deadline exceeded')
            if wire_offset == len(buffered):
                reader.raw._sock.settimeout(remaining)
                buffered = reader.read1(65536)
                arrived = time.perf_counter() - started
                require(arrived < self.timeout, 'request deadline exceeded')
                require(bool(buffered), 'truncated HTTP chunk framing')
                wire_offset = 0
            end = min(wire_offset + n, len(buffered))
            raw, wire_offset = buffered[wire_offset:end], end
            return raw, arrived

        def exact(n):
            parts = bytearray()
            while len(parts) < n:
                raw, _ = read_part(n - len(parts))
                parts.extend(raw)
            return bytes(parts)

        total, pending = 0, bytearray()
        while True:
            size_line = bytearray()
            while not size_line.endswith(b'\r\n'):
                require(len(size_line) < 128, 'oversized HTTP chunk header')
                size_line.extend(exact(1))
            require(re.fullmatch(rb'[0-9a-fA-F]+\r\n', size_line) is not None,
                    'native HTTP chunk header invalid (extensions are not supported)')
            size = int(size_line[:-2], 16)
            if size == 0:
                require(exact(2) == b'\r\n', 'native final chunk delimiter/trailers invalid')
                require(not pending, 'incomplete NDJSON frame')
                break
            total += size
            if total > self.max_response:
                raise GenerationFailure('native_response_limit', response_bytes=total, limit_bytes=self.max_response)
            while size:
                raw, received = read_part(min(size, 65536))
                size -= len(raw)
                pending.extend(raw)
                while b'\n' in pending:
                    offset = pending.index(b'\n')
                    require(offset + 1 <= self.max_frame, 'oversized NDJSON frame')
                    line = bytes(pending[:offset + 1])
                    del pending[:offset + 1]
                    # Parsing and callbacks are not new arrivals. Frames completed
                    # by one read share its clock across HTTP chunk boundaries.
                    yield strict_json(line), received
                require(len(pending) <= self.max_frame, 'oversized incomplete NDJSON frame')
            require(exact(2) == b'\r\n', 'invalid HTTP chunk delimiter')

    def frames(self, path, payload):
        started = time.perf_counter()
        conn, response = self._connect('POST', path, payload)
        trace = None
        try:
            require(response.getheader('Content-Type', '').split(';')[0] == 'application/x-ndjson',
                    'expected NDJSON streaming response')
            require(response.chunked, 'expected complete chunked HTTP framing')
            if self.trace_root:
                rid = payload.get('request_id', 'shutdown')
                trace = (self.trace_root / (rid + '.jsonl')).open('x', encoding='utf-8')
                (self.trace_root / (rid + '-request.json')).write_text(
                    json.dumps(payload, ensure_ascii=False), encoding='utf-8')
            for frame, elapsed in self._native_frames(response, started):
                if trace:
                    trace.write(json.dumps(dict(elapsed_s=elapsed, frame=frame),
                                           ensure_ascii=False) + '\n')
                    trace.flush()
                yield frame, elapsed
        finally:
            if trace:
                trace.close()
            response.close()
            conn.close()

    def turn(self, ledger, text, emit):
        plan = ledger.plan(text)
        verifier = StreamVerifier(plan['payload']['request_id'], ledger.template)
        started, first, last, first_count = time.perf_counter(), None, None, 0
        frames = self.frames(plan['path'], plan['payload'])
        try:
            for frame, elapsed in frames:
                verifier.push(frame)
                if frame['event'] == 'text':
                    if first is None:
                        first = elapsed
                    if elapsed == first:
                        first_count += len(frame['token_ids'])
                    last = elapsed
                    emit(frame['text'])
            terminal, output = verifier.finish()
            ledger.accept(plan, terminal, output)
        except BaseException:
            # Partial text may already be visible, but never becomes a reusable ledger.
            ledger.poisoned = True
            raise
        finally:
            frames.close()
        metrics = dict(request_id=plan['payload']['request_id'],
                       session_id=ledger.session_id, revision=terminal['session']['revision'],
                       input_tokens=len(plan['input_ids']), visible_tokens=len(verifier.tokens),
                       consumed_tokens=len(ledger.history), stop_reason=terminal['stop_reason'],
                       frame_clock=FRAME_CLOCK,
                       ttft_s=first, http_wall_s=time.perf_counter() - started,
                       post_first_tps=((len(verifier.tokens) - first_count) / (last - first)
                                       if first is not None and last > first else None),
                       native_report_sha256=terminal['native_report_sha256'],
                       product_ready=False, qualified=False)
        if self.trace_root:
            (self.trace_root / (plan['payload']['request_id'] + '-metrics.json')).write_text(
                json.dumps(metrics), encoding='utf-8')
        return output, metrics

    def close_session(self, ledger):
        if ledger.terminal is None:
            return
        require(not ledger.poisoned, 'uncertain session cannot be reused or closed as healthy')
        rid = 'close-' + uuid.uuid4().hex
        payload = dict(request_id=rid, session_id=ledger.session_id,
                       revision=ledger.terminal['session']['revision'])
        try:
            frames = list(self.frames('/v1/session/close', payload))
            require(len(frames) == 1 and frames[0][0].get('event') == 'session_closed'
                    and frames[0][0].get('request_id') == rid
                    and frames[0][0].get('session_id') == ledger.session_id, 'session close unconfirmed')
        finally:
            ledger.poisoned = True  # This ledger is retired even if the response was lost.

    def shutdown(self):
        # Explicit option only: never stop a user's resident service implicitly.
        conn, response = self._connect('POST', '/v1/shutdown', {})
        try:
            raw = response.read(MAX_FRAME + 1)
            require(len(raw) <= MAX_FRAME, 'oversized shutdown response')
            frame = strict_json(raw)
            require((frame.get('event') == 'service_closed' and frame.get('cleanup_ok') is True
                     and frame.get('success') is True)
                    or (frame.get('status') == 'stopped' and frame.get('owner_resident') is False),
                    'native shutdown cleanup unconfirmed')
            return frame
        finally:
            response.close()
            conn.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True,
                        help='contains tokenizer.json, chat_template.jinja, generation_config.json')
    parser.add_argument('--url', default='http://127.0.0.1:8193')
    parser.add_argument('--message', '-m', action='append', default=[],
                        help='send a turn; repeat for a same-session conversation')
    parser.add_argument('--message-file', type=Path, action='append', default=[],
                        help='send UTF-8 file contents, after any --message turns')
    parser.add_argument('--system', help='optional system text, rendered by the model template')
    parser.add_argument('--timeout', type=float, default=1800, help='total deadline per HTTP request')
    parser.add_argument('--trace-root', type=Path,
                        help='NEW private directory for raw requests/frames and timing; opt-in')
    parser.add_argument('--shutdown-on-exit', action='store_true',
                        help='explicitly shut down the service and release its resident model')
    args = parser.parse_args(argv)
    template = ChatTemplate(args.model_dir)
    transport = Transport(args.url, args.timeout, args.trace_root)
    health = transport.health()
    if args.trace_root:
        args.trace_root.mkdir(parents=True, exist_ok=False)
        (args.trace_root / 'client.json').write_text(json.dumps(dict(
            schema='qwen35-chat-client-run-v1', client_sha256=sha256(Path(__file__).read_bytes()),
            model_dir=str(args.model_dir.resolve()), template_identity=template.identity,
            server_url=args.url, health=health, no_thinking=True,
            product_ready=False, qualified=False), ensure_ascii=False), encoding='utf-8')
    ledger = Ledger(template, health['profile'], args.system)
    print(f"Connected: {args.url}, profile={health['profile']}; text-only, no thinking. "
          'This client does not grant product qualification.', file=sys.stderr)
    messages = args.message + [p.read_text(encoding='utf-8') for p in args.message_file]
    interactive = not messages and sys.stdin.isatty()
    if not messages and not interactive:
        messages = [sys.stdin.read()]

    def emit(text):
        sys.stdout.write(terminal_text(text))
        sys.stdout.flush()

    code = 0
    try:
        while True:
            if interactive:
                try:
                    text = input('\nYou (/new, /quit)> ')
                except EOFError:
                    break
                if text.strip() == '/quit':
                    break
                if text.strip() == '/new':
                    transport.close_session(ledger)
                    ledger = Ledger(template, health['profile'], args.system)
                    continue
                if not text.strip():
                    continue
            elif messages:
                text = messages.pop(0)
            else:
                break
            _, metrics = transport.turn(ledger, text, emit)
            print()
            print(json.dumps(metrics, ensure_ascii=False), file=sys.stderr)
        transport.close_session(ledger)
    except KeyboardInterrupt:
        print('\nCancelled. Partial output is not reusable; no automatic retry.', file=sys.stderr)
        code = 130
    except (ValueError, OSError, http.client.HTTPException, KeyError, TypeError) as error:
        print(f'\nChat stopped: {error}. No automatic replay.', file=sys.stderr)
        code = 1
    finally:
        # Local input errors or Ctrl+C while awaiting input do not disconnect an
        # in-flight request. Explicitly close that healthy session before exit.
        if ledger.terminal is not None and not ledger.poisoned:
            try:
                transport.close_session(ledger)
            except (ValueError, OSError, http.client.HTTPException) as error:
                print(f'Session close unconfirmed: {error}', file=sys.stderr)
                code = code or 1
        if args.shutdown_on_exit:
            try:
                result = transport.shutdown()
                print('Service shutdown confirmed.', file=sys.stderr)
                if args.trace_root:
                    (args.trace_root / 'shutdown.json').write_text(json.dumps(result), encoding='utf-8')
            except (ValueError, OSError, http.client.HTTPException) as error:
                print(f'Shutdown unconfirmed: {error}', file=sys.stderr)
                code = code or 1
    return code


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, ImportError) as error:
        print(f'Cannot start chat: {error}', file=sys.stderr)
        sys.exit(1)
