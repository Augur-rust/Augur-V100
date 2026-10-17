"""Private client state paired with a completed native HTTP SSD checkpoint.

This module does not load a model, change runtime ownership, or retokenize old
answers. A new owner must restore the server checkpoint before a client ledger
becomes usable. The checksum detects file corruption; it is not authentication
or encryption. CLI integration and real GPU recovery require separate tests.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import re
import stat
import uuid

from qwen35_chat import (Ledger, PROFILES, SCHEMA as STREAM_SCHEMA, digest_string,
                        require, sha256, strict_json, token_hash)

SCHEMA = 'qwen35-chat-client-checkpoint-v1'
ANCHOR = 'qwen35-chat-restored-anchor-v1'
MAX_BYTES = 32 * 1024 * 1024
CLOSED_CLAIMS = ('qualified', 'product_ready', 'semantic_qualification', 'visible_tps')
STATE_KEYS = ('session_id', 'profile', 'messages', 'prefix', 'last_text', 'history', 'terminal')
SAME_BOUNDARY = ('revision', 'capacity', 'consumed_token_count', 'consumed_input_digest',
                 'pending_token', 'pending_token_published')


class CheckpointAdmissionRejected(ValueError):
    """A fully framed, identity-bound refusal before capture; no checkpoint saved."""


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def valid_id(value):
    return isinstance(value, str) and re.fullmatch('[A-Za-z0-9_-]{1,64}', value) is not None


def plain_path(path):
    path = Path(path).absolute()
    for item in (*reversed(path.parents), path):
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        require(not stat.S_ISLNK(info.st_mode) and
                not getattr(info, 'st_file_attributes', 0) & 0x400, 'linked/reparse client state path')
    return path


def boundary(meta, history, template):
    require(type(meta) is dict and valid_id(meta.get('session_id')), 'invalid session identity')
    require(type(meta.get('revision')) is int and meta['revision'] > 0 and
            type(meta.get('capacity')) is int and meta['capacity'] == 196608, 'invalid session revision/capacity')
    require(type(history) is list and 0 < len(history) <= 196608 and
            all(type(t) is int and t in template.vocab for t in history), 'invalid consumed history')
    require(type(meta.get('consumed_token_count')) is int and meta['consumed_token_count'] == len(history)
            and meta.get('consumed_input_digest') == token_hash(history), 'exact history boundary mismatch')
    require(type(meta.get('pending_token')) is int and meta['pending_token'] in template.vocab and
            meta.get('pending_token_published') is True and digest_string(meta.get('checkpoint_sha256')),
            'invalid pending/checkpoint boundary')


def validate_state(state, template):
    require(type(state) is dict and set(state) == set(STATE_KEYS), 'closed client state schema required')
    require(valid_id(state['session_id']) and isinstance(state['profile'], str)
            and state['profile'] in PROFILES, 'invalid saved session/profile')
    terminal, messages, history = state['terminal'], state['messages'], state['history']
    require(type(terminal) is dict and terminal.get('schema') in (STREAM_SCHEMA, ANCHOR), 'confirmed terminal required')
    require(terminal.get('event') == ('terminal' if terminal['schema'] == STREAM_SCHEMA else 'restored_session_anchor'),
            'unconfirmed client boundary event')
    boundary(terminal.get('session'), history, template)
    meta = terminal['session']
    require(meta['session_id'] == state['session_id'] and len(history) <= PROFILES[state['profile']][2],
            'client session/profile extent mismatch')
    visible = terminal.get('visible_token_ids')
    require(type(visible) is list and all(type(t) is int and t in template.vocab and t not in template.eos
                                        for t in visible), 'invalid last visible tokens')
    stop = terminal.get('stop_reason')
    pending = meta['pending_token']
    require((stop == 'eos' and pending in template.eos) or
            (stop == 'max_new_tokens' and visible and pending == visible[-1]), 'last stop/pending mismatch')
    require(isinstance(state['last_text'], str) and isinstance(state['prefix'], str) and
            terminal.get('text_sha256') == sha256(state['last_text'].encode()) and
            digest_string(terminal.get('native_report_sha256')), 'last generation text/report binding')
    require(template.tokenizer.decode(visible, skip_special_tokens=True) == state['last_text'],
            'visible token/text mismatch')
    consumed_visible = visible if stop == 'eos' else visible[:-1]
    require(not consumed_visible or history[-len(consumed_visible):] == consumed_visible,
            'last visible output absent from consumed tail')
    require(type(messages) is list and 2 <= len(messages) <= 129, 'bounded completed conversation required')
    offset = int(messages[0].get('role') == 'system') if isinstance(messages[0], dict) else 0
    require((len(messages) - offset) % 2 == 0, 'complete user/assistant pairs required')
    for index, message in enumerate(messages):
        expected = 'system' if index < offset else ('user' if (index - offset) % 2 == 0 else 'assistant')
        require(type(message) is dict and set(message) == {'role', 'content', 'reasoning_content', 'tool_calls'}
                and message['role'] == expected and isinstance(message['content'], str)
                and message['reasoning_content'] == '' and message['tool_calls'] == [], 'unsupported saved message')
        if expected != 'assistant':
            template.message(expected, message['content'])
    require(messages[-1]['content'] == state['last_text'] and
            template.render(messages[:-1]) == state['prefix'], 'canonical saved transcript mismatch')


def snapshot(ledger):
    require(not ledger.poisoned and ledger.terminal is not None, 'only confirmed healthy sessions can be saved')
    state = copy.deepcopy({key: getattr(ledger, key) for key in STATE_KEYS})
    validate_state(state, ledger.template)
    return state


def control(transport, payload, operation, expected_profile, ledger=None):
    require(operation in ('save', 'restore') and digest_string(expected_profile), 'checkpoint operation/profile pin')
    require(digest_string(payload.get('checkpoint_id')), 'checkpoint ID must be 64 lowercase hex digits')
    prior_boundary = (copy.deepcopy(ledger.terminal['session'])
                      if operation == 'save' and ledger is not None and ledger.terminal is not None else None)
    frames = transport.frames('/v1/session/' + operation, payload)
    try:
        rows = list(frames)  # final HTTP chunk must arrive before publishing anything
        require(len(rows) == 1, 'one complete checkpoint terminal required')
        frame = rows[0][0]
        if type(frame) is dict and frame.get('schema') == 'qwen35-http-checkpoint-admission-rejection-v1':
            expected_keys = {'schema', 'event', 'operation', 'before_submit', 'capture_started',
                'session_preserved', 'request_id', 'session_id', 'revision', 'checkpoint_id',
                'profile_sha256', 'owner_id', 'owner_epoch', 'session', 'detail', *CLOSED_CLAIMS}
            require(set(frame) == expected_keys and operation == 'save' and prior_boundary is not None,
                    'unsupported checkpoint admission rejection')
            require(frame['event'] == 'rejected' and frame['operation'] == 'save'
                    and frame['before_submit'] is True and frame['capture_started'] is False
                    and frame['session_preserved'] is True, 'uncertain checkpoint rejection boundary')
            for key in ('request_id', 'session_id', 'checkpoint_id'):
                require(frame[key] == payload[key], 'checkpoint rejection identity mismatch: ' + key)
            require(frame['profile_sha256'] == expected_profile and digest_string(frame['owner_id'])
                    and type(frame['owner_epoch']) is int and 0 < frame['owner_epoch'] < 2**64,
                    'checkpoint rejection owner/profile mismatch')
            require(type(frame['revision']) is int and frame['revision'] == payload['revision']
                    and type(frame['session']) is dict
                    and canonical(frame['session']) == canonical(prior_boundary)
                    and canonical(ledger.terminal['session']) == canonical(prior_boundary),
                    'checkpoint rejection changed confirmed session')
            require(all(frame[key] is False for key in CLOSED_CLAIMS)
                    and type(frame['detail']) is str and 0 < len(frame['detail']) <= 8192,
                    'checkpoint rejection claim/detail drift')
            return frame  # Fully drained and verified; save() reports failure, never a saved bundle.
        expected_event = 'session_saved' if operation == 'save' else 'session_restored'
        require(type(frame) is dict and frame.get('schema') == 'qwen35-http-checkpoint-terminal-v1'
                and frame.get('event') == expected_event, 'unexpected checkpoint terminal')
        for key in ('request_id', 'session_id', 'checkpoint_id'):
            require(frame.get(key) == payload[key], 'checkpoint response identity mismatch: ' + key)
        require(frame.get('profile_sha256') == expected_profile and digest_string(frame.get('owner_id')) and
                type(frame.get('owner_epoch')) is int and 0 < frame['owner_epoch'] < 2**64,
                'checkpoint owner/profile mismatch')
        for key in ('http_manifest_sha256', 'native_manifest_sha256'):
            require(digest_string(frame.get(key)), 'missing checkpoint manifest binding')
        require(all(frame.get(key) is False for key in CLOSED_CLAIMS), 'checkpoint authority drift')
        require(type(frame.get('session')) is dict and frame['session'].get('session_id') == payload['session_id']
                and type(frame.get('revision')) is int and frame['revision'] > 0
                and frame['revision'] == frame['session'].get('revision'), 'checkpoint revision mismatch')
        return frame
    except BaseException:
        if ledger is not None:
            ledger.poisoned = True
        raise
    finally:
        frames.close()


def make_bundle(state, template, saved):
    value = dict(schema=SCHEMA, template_identity=copy.deepcopy(template.identity),
                 state=state, checkpoint={key: saved[key] for key in (
                     'checkpoint_id', 'http_manifest_sha256', 'native_manifest_sha256',
                     'owner_id', 'owner_epoch', 'profile_sha256')})
    value['content_sha256'] = sha256(canonical(value))
    require(len(canonical(value)) <= MAX_BYTES, 'client state exceeds file bound')
    return value


def validate_bundle(value, template):
    require(type(value) is dict and set(value) == {'schema', 'template_identity', 'state', 'checkpoint', 'content_sha256'}
            and value['schema'] == SCHEMA, 'unsupported client checkpoint schema')
    require(value['template_identity'] == template.identity, 'tokenizer/template/generation identity mismatch')
    require(value['content_sha256'] == sha256(canonical({k: v for k, v in value.items() if k != 'content_sha256'})),
            'client state checksum mismatch')
    checkpoint = value['checkpoint']
    require(type(checkpoint) is dict and set(checkpoint) == {'checkpoint_id', 'http_manifest_sha256',
        'native_manifest_sha256', 'owner_id', 'owner_epoch', 'profile_sha256'}, 'closed checkpoint reference required')
    require(digest_string(checkpoint['checkpoint_id']) and type(checkpoint['owner_epoch']) is int
            and 0 < checkpoint['owner_epoch'] < 2**64, 'invalid checkpoint ID/epoch')
    require(all(digest_string(checkpoint[k]) for k in ('http_manifest_sha256', 'native_manifest_sha256',
                                                     'owner_id', 'profile_sha256')), 'invalid checkpoint digest')
    validate_state(value['state'], template)
    return value


def read_bundle(path, template, expected_sha256=None):
    path = plain_path(path)
    path_before = path.stat()
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= MAX_BYTES, 'bounded regular client state required')
        raw = stream.read(MAX_BYTES + 1)
        after = os.fstat(stream.fileno())
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    # On Windows stat ctime can mean creation time while fstat ctime means
    # metadata-change time. Compare each API's ctime to itself, not to the other.
    require(len(raw) == before.st_size and identity(before) == identity(after)
            and identity(before)[:4] == identity(path_before)[:4]
            and identity(path_before) == identity(path.stat()),
            'client file changed while reading')
    plain_path(path)
    if expected_sha256 is not None:
        require(digest_string(expected_sha256) and sha256(raw) == expected_sha256, 'external client file hash mismatch')
    return validate_bundle(strict_json(raw), template)


class LocalStateWriteError(OSError):
    """Native save succeeded; retain the bundle to retry *local* publication only."""
    def __init__(self, bundle, cause):
        super().__init__('native checkpoint saved but client file publication failed: ' + str(cause))
        self.bundle = bundle


def publish_bundle(path, value, template):
    """Retry a saved bundle's local publication without another native capture."""
    value = copy.deepcopy(validate_bundle(value, template))
    raw = canonical(value)
    require(len(raw) <= MAX_BYTES, 'client state exceeds file bound')
    path = plain_path(path)
    require(path.parent.is_dir() and not path.exists(), 'fresh client file in an existing directory required')
    pending = path.with_name(path.name + '.pending-' + uuid.uuid4().hex)
    descriptor = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, 'wb') as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        plain_path(path)
        plain_path(pending)
        os.link(pending, path)
        read_bundle(path, template, sha256(raw))
        return dict(path=str(path), file_sha256=sha256(raw), checkpoint=copy.deepcopy(value['checkpoint']))
    finally:
        pending.unlink(missing_ok=True)


def save(transport, ledger, path, expected_profile_sha256, checkpoint_id=None):
    state = snapshot(ledger)
    require(len(canonical(state)) + len(canonical(ledger.template.identity)) + 4096 <= MAX_BYTES,
            'client state too large before native capture')
    require(digest_string(expected_profile_sha256), 'explicit checkpoint profile hash required')
    # The native save/restore ABI uses a 32-byte lowercase-hex store key, not
    # the more permissive request/session identifier grammar.
    checkpoint_id = os.urandom(32).hex() if checkpoint_id is None else checkpoint_id
    require(digest_string(checkpoint_id), 'checkpoint ID must be 64 lowercase hex digits')
    path = plain_path(path)
    require(path.parent.is_dir() and not path.exists(), 'fresh client file in an existing directory required')
    pending = path.with_name(path.name + '.pending-' + uuid.uuid4().hex)
    # Reserve a writable local file before requesting the expensive GPU capture.
    descriptor = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    bundle = None
    try:
        with os.fdopen(descriptor, 'wb') as output:
            payload = dict(request_id='save-' + uuid.uuid4().hex, session_id=state['session_id'],
                           revision=state['terminal']['session']['revision'], checkpoint_id=checkpoint_id)
            saved = control(transport, payload, 'save', expected_profile_sha256, ledger)
            try:
                require(saved['session'] == state['terminal']['session'] and saved['revision'] == payload['revision'],
                        'save changed the confirmed boundary')
                require(canonical(snapshot(ledger)) == canonical(state), 'client changed during save')
            except BaseException:
                ledger.poisoned = True
                raise
            if saved['event'] == 'rejected':
                raise CheckpointAdmissionRejected('checkpoint not captured; live session retained: ' + saved['detail'])
            bundle = make_bundle(state, ledger.template, saved)
            raw = canonical(bundle)
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        plain_path(path)
        plain_path(pending)
        os.link(pending, path)  # atomic create, never overwrite an existing file
        read_bundle(path, ledger.template, sha256(raw))
        return dict(path=str(path), file_sha256=sha256(raw), checkpoint=copy.deepcopy(bundle['checkpoint']))
    except (OSError, ValueError) as error:
        if bundle is not None:
            raise LocalStateWriteError(bundle, error) from error
        raise
    finally:
        pending.unlink(missing_ok=True)


def restore(transport, value, template, expected_profile_sha256):
    value = copy.deepcopy(validate_bundle(value, template))
    checkpoint, state = value['checkpoint'], value['state']
    require(checkpoint['profile_sha256'] == expected_profile_sha256, 'restore profile differs from saved profile')
    require(transport.health()['profile'] == state['profile'], 'destination session profile differs from saved profile')
    ledger = Ledger(template, state['profile'])  # fresh client/native session ID
    payload = dict(request_id='restore-' + uuid.uuid4().hex, session_id=ledger.session_id,
                   checkpoint_id=checkpoint['checkpoint_id'], http_manifest_sha256=checkpoint['http_manifest_sha256'])
    frame = control(transport, payload, 'restore', expected_profile_sha256, ledger)
    try:
        require(frame['http_manifest_sha256'] == checkpoint['http_manifest_sha256'] and
                frame['native_manifest_sha256'] == checkpoint['native_manifest_sha256'] and
                frame.get('source_owner_id') == checkpoint['owner_id'] and frame['owner_id'] != checkpoint['owner_id'] and
                frame.get('source_session_id') == state['session_id'] and ledger.session_id != state['session_id'],
                'fresh owner/session or saved manifest binding mismatch')
        require(frame.get('target_seal_verified') is True and frame.get('client_delivery_acknowledged') is False,
                'restore target seal/publication contract')
        boundary(frame['session'], state['history'], template)
        for key in SAME_BOUNDARY:
            require(type(frame['session'].get(key)) is type(state['terminal']['session'][key]) and
                    frame['session'][key] == state['terminal']['session'][key], 'restored boundary drift: ' + key)
        require(type(frame.get('restored_prefix_tokens')) is int and frame['restored_prefix_tokens'] == len(state['history'])
                and type(frame.get('prefix_recompute_tokens')) is int and frame['prefix_recompute_tokens'] == 0,
                'restore must consume the exact saved state, not recompute the prefix')
        native = frame.get('native_restore')
        require(type(native) is dict and all(native.get(k) is True for k in (
            'fresh_native_session_owner', 'device_pending_token_verified', 'draft_rebuild_submitted_and_drained')) and
            all(native.get(k) is False for k in ('old_cuda_authority_restored', 'draft_rebuild_outputs_published')),
            'native restore capability/rebuild contract')
        require(type(native.get('restored_prefix_tokens')) is int and native['restored_prefix_tokens'] == len(state['history'])
                and type(native.get('prefix_recompute_tokens')) is int and native['prefix_recompute_tokens'] == 0,
                'native/HTTP restored extent mismatch')
        for key in ('messages', 'prefix', 'last_text', 'history'):
            setattr(ledger, key, copy.deepcopy(state[key]))
        ledger.terminal = copy.deepcopy(state['terminal'])
        ledger.terminal.update(schema=ANCHOR, event='restored_session_anchor', session=frame['session'],
                               restore_request_id=payload['request_id'], restored_from_checkpoint=checkpoint['checkpoint_id'])
        # This is an explicit restore anchor, never a forged generation terminal.
        snapshot(ledger)
        return ledger, frame
    except BaseException:
        ledger.poisoned = True
        raise
