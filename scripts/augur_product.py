"""Pinned Augur launcher and same-origin local chat; speculative DFlash is off."""
from __future__ import annotations
import time
MODULE_STARTED = time.perf_counter()
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import secrets
import shutil
import signal
import subprocess
import threading
import augur_api as Q
import augur_launch_config as LC
from augur_startup import EngineLease
import qwen35_api as A
import qwen35_chat as C

ASSETS = Path(__file__).with_name('augur_web')
DRAFT_KEYS = ('AUGUR_DRAFT_CONFIG', 'AUGUR_DRAFT_CONFIG_SHA256',
              'AUGUR_DRAFT_PATH', 'AUGUR_DRAFT_SHA256')
FILES = {'/': ('index.html', 'text/html; charset=utf-8'),
         '/assets/app.js': ('app.js', 'text/javascript; charset=utf-8'),
         '/assets/style.css': ('style.css', 'text/css; charset=utf-8'),
         '/assets/i18n.js': ('i18n.js', 'text/javascript; charset=utf-8'),
         '/assets/lucide.min.js': ('vendor/lucide.min.js', 'text/javascript; charset=utf-8')}


def dflash_profile(environment):
    mode = environment.get('AUGUR_HTTP_DFLASH', 'resident_only')
    C.require(mode in ('resident_only', 'batched8'), 'AUGUR_HTTP_DFLASH must be resident_only or batched8')
    return Q.DFLASH_PROFILE if mode == 'batched8' else Q.AUGUR_PROFILE


def cache_restore_policy(environment):
    diagnostic = environment.get('AUGUR_CACHE_RESTORE_DIAGNOSTIC', '0')
    C.require(diagnostic in ('0', '1'), 'invalid cache restore diagnostic mode')
    value = environment.get('AUGUR_CACHE_RESTORE_WORKERS', '1' if diagnostic == '1' else '8')
    C.require(value in ('1', '2', '4', '8'), 'cache restore workers must be 1/2/4/8')
    C.require(diagnostic != '1' or value == '1', 'restore diagnostic lane requires one verifier')
    selection = ('environment_override' if 'AUGUR_CACHE_RESTORE_WORKERS' in environment
                 else 'diagnostic_default' if diagnostic == '1' else 'product_default')
    return int(value), diagnostic == '1', selection


def verify_cache_restore_policy(native, workers, diagnostic, producer_sha, configured):
    capability = native.get('checkpoint_capabilities')
    disabled = dict(input_prefix_tokens=None, save_restore_enabled=False)
    if capability is None or capability == disabled:
        C.require(not configured, 'configured checkpoint manager missing from native receipt')
        return False
    C.require(type(capability.get('input_prefix_restore_workers')) is int
              and capability['input_prefix_restore_workers'] == workers
              and capability.get('input_prefix_restore_diagnostic') is diagnostic
              and capability.get('bindings', {}).get('producer_sha256') == producer_sha,
              'native checkpoint restore policy differs from product selection')
    return True


def cancellation_policy(environment):
    """Product default only; legacy/native standalone owners keep their contract."""
    value = environment.get('AUGUR_CANCEL_RECOVERY', 'discard')
    C.require(value in ('discard', 'retire'),
              'AUGUR_CANCEL_RECOVERY must be discard or retire')
    return value


def verify_cancellation_policy(health, mode):
    C.require(mode in ('discard', 'retire'), 'invalid requested cancellation policy')
    C.require(health.get('owner_resident') is True,
              'cancellation policy requires a resident owner after startup smoke')
    C.require(health.get('request_bound_cancellation_discard', False) is (mode == 'discard'),
              'native cancellation capability differs from selected product policy')


class StopRequested(Exception):
    pass


class StopSignals:
    def __init__(self):
        self.signum, self.count, self.interrupted_error = None, 0, None
        self.previous = {}

    def receive(self, signum, frame):
        # No locks or raised exceptions: a signal can interrupt resource creation.
        if self.signum is None:
            self.signum = signum
        self.count += 1

    def __enter__(self):
        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                self.previous[signum] = signal.signal(signum, self.receive)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *args):
        for signum, previous in self.previous.items():
            signal.signal(signum, previous)
        self.previous.clear()

    def check(self):
        if self.signum is not None:
            raise StopRequested()

    def run(self, action, on_stop):
        self.check()
        results = queue.Queue()

        def work():
            try:
                results.put((True, action()))
            except BaseException as error:
                results.put((False, error))

        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        deadline = None
        while True:
            try:
                success, result = results.get(timeout=.05)
                break
            except queue.Empty:
                if self.signum is not None:
                    if deadline is None:
                        deadline = time.monotonic() + 90
                    # Retry abort until the startup transport has become active.
                    on_stop()
                    if time.monotonic() >= deadline:
                        raise TimeoutError('interrupted product worker did not drain')
        worker.join()
        if self.signum is not None and not success:
            self.interrupted_error = dict(type=type(result).__name__, error=str(result))
        self.check()
        if not success:
            raise result
        return result

    def receipt(self):
        return dict(signal=signal.Signals(self.signum).name if self.signum is not None else None,
                    count=self.count, interrupted_error=self.interrupted_error)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)


def verified_build(root):
    root = Path(root).resolve()
    receipt = C.strict_json((root / 'build-receipt.json').read_bytes())
    C.require(receipt['feature'].endswith('/qwen35-augur-vision'), 'Vision build required')
    cpu = C.strict_json((root / 'cpu-complete.json').read_bytes())
    C.require(cpu.get('status') == 'PASS' and cpu['build_sha256'] == sha(root / 'build-receipt.json'), 'CPU receipt mismatch')
    C.require(receipt['source_sha256'] == sha(root / 'source-identity.json'), 'source identity mismatch')
    source = C.strict_json((root / 'source-identity.json').read_bytes())
    for name, expected in source['files'].items():
        path = (root / 'source' / name).resolve()
        C.require(path.is_relative_to(root / 'source') and sha(path) == expected,
                  'frozen source mismatch: ' + name)
    portable_path = root / 'portable-build.json'
    portable = None
    if portable_path.exists():
        portable = C.strict_json(portable_path.read_bytes())
        C.require(portable.get('schema') == 'quasar-windows-portable-build-v1'
                  and portable.get('build_receipt_sha256') == sha(root / 'build-receipt.json')
                  and portable.get('cpu_receipt_sha256') == sha(root / 'cpu-complete.json')
                  and portable.get('source_identity_sha256') == receipt['source_sha256'],
                  'portable build receipt mismatch')
        C.require(isinstance(portable.get('binaries'), dict)
                  and set(portable['binaries']) == set(receipt['binaries']), 'portable binary inventory mismatch')
    resolved = {}
    for name, binary in receipt['binaries'].items():
        if portable is None:
            path = Path(binary['path']).resolve()
        else:
            mapped = portable['binaries'][name]
            relative = mapped.get('path') if isinstance(mapped, dict) else None
            C.require(isinstance(relative, str) and relative.startswith('bin/')
                      and ':' not in relative and '\\' not in relative
                      and all(part not in ('', '.', '..') for part in relative.split('/')),
                      'unsafe portable binary path: ' + name)
            path = (root / relative).resolve()
            C.require(path.is_relative_to(root / 'bin') and mapped.get('sha256') == binary['sha256'],
                      'portable binary mapping mismatch: ' + name)
        C.require(path.is_relative_to(root) and sha(path) == binary['sha256'], 'binary identity mismatch: ' + name)
        resolved[name] = dict(binary, path=str(path))
    # Preserve the original receipts byte-for-byte; only returned launch paths change.
    return dict(receipt, binaries=resolved)


def snapshot_public(out):
    root = Path(__file__).parent
    files = sorted(set(root.glob('augur*.py')) | set(root.glob('qwen35*.py')) |
                   {p for p in ASSETS.rglob('*') if p.is_file()})
    identities = {}
    for path in files:
        name = str(path.relative_to(root))
        destination = out / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        identities[name] = sha(destination)
        C.require(identities[name] == sha(path), 'public source changed during snapshot')
    return identities


def cleanup(process, server, controller, timer, url, evidence, transport_factory=None):
    errors, forced = [], False
    shutdown = None

    def attempt(phase, action):
        try:
            return action()
        except BaseException as error:
            errors.append(dict(phase=phase, type=type(error).__name__, error=str(error)))

    if timer:
        attempt('timer_cancel', timer.cancel)
    if controller:
        attempt('controller_abort', controller.abort)
        deadline = time.monotonic() + 90
        while controller.busy.locked() and time.monotonic() < deadline:
            time.sleep(.05)
        if controller.busy.locked():
            errors.append(dict(phase='controller_drain', error='controller remained busy'))
        attempt('controller_close', controller.close)
    if server:
        attempt('public_server_close', server.server_close)
    if process and process.poll() is None:
        if url:
            factory = transport_factory or (lambda: Q.Transport(url, 90, getattr(controller, 'profile', Q.AUGUR_PROFILE)))
            shutdown = attempt('native_shutdown', lambda: factory().shutdown())
        attempt('native_wait', lambda: process.wait(timeout=90))
        if process.poll() is None:
            forced = True
            if os.name == 'nt':
                attempt('native_force', lambda: subprocess.run(
                    ['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True, timeout=30, check=True))
            else:
                attempt('native_force', process.terminate)
            attempt('native_force_wait', lambda: process.wait(timeout=30))
    closures, retirements = [], []
    for name, target in [('owner-stopped.json', closures), ('retirement.json', retirements)]:
        for path in (evidence / 'native').rglob(name):
            value = attempt('read_' + name, lambda p=path: C.strict_json(p.read_bytes()))
            if value is not None:
                target.append(value)
    allocation_closure = bool(closures) and all(c.get('allocation_closure') is True and
        c.get('live_allocations_after') == 0 and c.get('live_bytes_after') == 0 and
        c.get('drain_error') is None for c in closures)
    retired = bool(retirements) and all(r.get('confirmed_exit') is True and
        r.get('native_cleanup_verified') is True for r in retirements)
    confirmed_exit = process is not None and process.poll() is not None
    never_spawned = (isinstance(shutdown, dict) and shutdown.get('no_owner') is True and
        not any((evidence / 'native').glob('owner-*')) and not closures and not retirements)
    return dict(exit_code=process.returncode if process else None, adapter_pid=process.pid if process else None,
        confirmed_exit=confirmed_exit, forced=forced, cleanup_errors=errors,
        native_closures=closures, native_retirements=retirements, allocation_closure=allocation_closure,
        native_owner_never_spawned=never_spawned,
        native_shutdown=shutdown if isinstance(shutdown, dict) else None,
        cleanup_verified=confirmed_exit and process.returncode == 0 and
            ((allocation_closure and retired) or never_spawned) and not forced and not errors,
        product_ready=False)


class Handler(A.Handler):
    def local_origin(self):
        authority = '127.0.0.1:' + str(self.server.server_address[1])
        hosts = self.headers.get_all('Host', [])
        origins = self.headers.get_all('Origin', [])
        if hosts != [authority] or (origins and origins != ['http://' + authority]):
            raise A.ApiError('Only the local application origin is accepted.', 403, 'origin_not_allowed')
        if self.headers.get('Sec-Fetch-Site') == 'cross-site':
            raise A.ApiError('Cross-site access is not accepted.', 403, 'origin_not_allowed')

    def authorized(self):
        self.local_origin()
        # The shared API rejects every browser Origin. This product accepts only
        # its own exact origin, then retains the shared constant-time key check.
        del self.headers['Origin']
        super().authorized()

    def do_POST(self):
        if self.path != '/augur/v1/stop':
            return super().do_POST()
        try:
            self.close_connection = True
            self.authorized()
            lengths = self.headers.get_all('Content-Length', [])
            if self.headers.get('Transfer-Encoding') is not None or len(lengths) != 1:
                raise A.ApiError('The stop request needs exactly one Content-Length.')
            size = int(lengths[0]) if lengths[0].isdigit() else -1
            if not 0 <= size <= 1024:
                raise A.ApiError('The stop request carries no parameters.')
            raw = self.rfile.read(size) if size else b''
            if raw.strip() not in (b'', b'{}'):
                raise A.ApiError('The stop request carries no parameters.')
            self.json_response(202, dict(accepted=True, operation='stop'))
            # Answer before the drain starts: the caller needs the 202 to tell an accepted stop
            # from a refused one, and serve_forever must not shut down inside a request thread.
            threading.Thread(target=self.server.request_stop, daemon=True).start()
        except A.ApiError as error:
            self.json_response(error.status, error.response())
        except (OSError, ValueError, ConnectionError):
            pass

    def do_GET(self):
        if self.path not in FILES:
            return super().do_GET()
        try:
            self.close_connection = True
            self.local_origin()
            name, mime = FILES[self.path]
            raw = (getattr(self.server, 'assets', ASSETS) / name).read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(raw)))
            self.send_header('Connection', 'close')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' data:; connect-src 'self'; style-src 'self'; script-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
            self.end_headers()
            self.wfile.write(raw)
        except A.ApiError as error:
            self.json_response(error.status, error.response())
        except (OSError, ConnectionError):
            pass


class Server(A.Server):
    def __init__(self, address, controller, api_key):
        # The public product owns its own stop switch; set before the socket accepts anything.
        self.stop_requested = threading.Event()
        super().__init__(address, controller, api_key)

    def request_stop(self):
        """Cooperative stop from the served page or a scripted client: drain, release the GPU,
        write stopped.json, then let launch() return normally so cleanup runs on the normal path."""
        if self.stop_requested.is_set():
            return
        self.stop_requested.set()
        self.shutdown()

    def finish_request(self, request, client_address):
        Handler(request, client_address, self)


def launch(argv, stop):
    started = MODULE_STARTED
    stages = dict(module_imports_s=time.perf_counter() - started)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', required=True, type=Path)
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--evidence', required=True, type=Path)
    parser.add_argument('--draft-environment', type=Path)
    parser.add_argument('--port', type=int, default=8294)
    parser.add_argument('--context-tokens', type=int, default=262144, choices=(8192,32768,65536,131072,196608,262144))
    parser.add_argument('--default-output-tokens', type=int, default=256)
    parser.add_argument('--stop-on-stdin', action='store_true', help='owned launcher pipe: stop cooperatively on stop/EOF')
    parser.add_argument('--prefix-cache-root', type=Path, help='optional persistent exact-input prefix cache')
    parser.add_argument('--prefix-cache-scope', default='chat',
                        help='one path segment under the cache root; runs in different scopes never share entries')
    parser.add_argument('--serve-seconds', type=float, help='bounded smoke-run duration; omit for interactive use')
    args = parser.parse_args(argv)
    C.require(0 <= args.port <= 65535, 'invalid port')
    Q.configure_context(args.context_tokens, args.default_output_tokens)
    os.environ['AUGUR_CONTEXT_TOKENS'] = str(args.context_tokens)
    if args.prefix_cache_root:
        try:
            scope = LC.cache_scope(args.prefix_cache_scope)
        except ValueError as error:
            raise SystemExit('invalid --prefix-cache-scope: ' + str(error))
        os.environ['AUGUR_INPUT_PREFIX_CACHE_ROOT'] = str((args.prefix_cache_root.resolve() / scope))
        os.environ['AUGUR_INPUT_PREFIX_CACHE_PROMPT_END'] = '1'
    with EngineLease():
        return launch_owned(args, stop, started, stages)


def launch_owned(args, stop, started, stages):
    if args.stop_on_stdin:
        def wait_parent():
            import sys
            sys.stdin.readline()
            stop.receive(signal.SIGTERM, None)
        threading.Thread(target=wait_parent, daemon=True).start()
    C.require(args.serve_seconds is None or 0 < args.serve_seconds <= 3600, 'invalid smoke duration')
    recovery_mode = cancellation_policy(os.environ)
    restore_workers, restore_diagnostic, restore_selection = cache_restore_policy(os.environ)
    profile = dflash_profile(os.environ)
    dflash_active = profile == Q.DFLASH_PROFILE
    stop.check()
    tick = time.perf_counter()
    binding = verified_build(args.build)
    stages['verified_build_s'] = time.perf_counter() - tick
    stop.check()
    tick = time.perf_counter()
    template = Q.Template(args.model_dir, args.manifest, args.manifest_sha256, profile, vision=True)
    stages['template_s'] = time.perf_counter() - tick
    stop.check()
    env = os.environ.copy()
    env.setdefault('AUGUR_SAMPLING', Q.SAMPLING_PRESET)
    C.require(env['AUGUR_SAMPLING'] in ('greedy', Q.SAMPLING_PRESET), 'Unsupported target sampling preset')
    env['AUGUR_CANCEL_RECOVERY'] = recovery_mode
    env['AUGUR_CACHE_RESTORE_WORKERS'] = str(restore_workers)
    if args.draft_environment:
        source = C.strict_json(args.draft_environment.read_bytes())
        source = source.get('environment', source)
        env.update({k: source[k] for k in DRAFT_KEYS})
    C.require(all(env.get(k) for k in DRAFT_KEYS), 'authenticated resident-draft environment required')
    if env.get('CUDA_PATH'):
        env['PATH'] = str(Path(env['CUDA_PATH']) / 'bin') + os.pathsep + env['PATH']
    args.evidence.mkdir(exist_ok=False)
    adapter = binding['binaries']['augur_session_http']
    producer = binding['binaries']['augur_session_jsonl']
    command = [adapter['path'], '--bind', '127.0.0.1:0', '--producer', producer['path'],
               '--producer-sha256', producer['sha256'], '--model-root', str(args.model_dir.resolve()),
               '--manifest', str(args.manifest.resolve()), '--manifest-sha256', args.manifest_sha256,
               '--evidence-root', str(args.evidence.resolve() / 'native')]
    tick = time.perf_counter()
    public_files = snapshot_public(args.evidence / 'public-source')
    save(args.evidence / 'identity.json', dict(build_receipt_sha256=sha(args.build / 'build-receipt.json'),
        manifest_sha256=args.manifest_sha256, launcher_sha256=sha(__file__),
        public_source_files={p.name: sha(p) for p in Path(__file__).parent.glob('augur*.py')},
        assets={str(p.relative_to(ASSETS)): sha(p) for p in ASSETS.rglob('*') if p.is_file()},
        public_snapshot_files=public_files,
        cancellation_recovery=recovery_mode,
        cache_restore_workers=restore_workers, cache_restore_diagnostic=restore_diagnostic,
        cache_restore_selection=restore_selection,
        cancellation_selection='environment_override' if 'AUGUR_CANCEL_RECOVERY' in os.environ else 'product_default',
        dflash_active=dflash_active, profile=profile,
        sampling_preset=env['AUGUR_SAMPLING'],
        selector_gate=dict(policy=env.get('AUGUR_DFLASH_ENGLISH_GATE'),
            source=env.get('AUGUR_DFLASH_SELECTOR_GATE_SOURCE'),
            sha256=env.get('AUGUR_DFLASH_SELECTOR_GATE_F32_SHA256'),
            accumulation=env.get('AUGUR_DFLASH_SELECTOR_GATE_ACCUMULATION'),
            runtime_scale=env.get('AUGUR_DFLASH_SELECTOR_SCALE')),
        product_ready=False, command=command))
    stages['public_snapshot_s'] = time.perf_counter() - tick
    process = server = controller = timer = None
    run_error = None
    url = None
    key = os.environ.get('AUGUR_QWEN35_API_KEY') or secrets.token_urlsafe(32)
    A.validate_key(key)
    with (args.evidence / 'adapter.stderr.log').open('xb') as log, (args.evidence / 'api-metrics.jsonl').open('x', encoding='utf8') as metrics:
        try:
            stop.check()
            tick = time.perf_counter()
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log, env=env,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            save(args.evidence / 'process.json', dict(adapter_pid=process.pid, launcher_pid=os.getpid(), command=command))
            responses = queue.Queue()
            threading.Thread(target=lambda: responses.put(process.stdout.readline()), daemon=True).start()
            listening = C.strict_json(responses.get(timeout=30))
            stages['adapter_listen_s'] = time.perf_counter() - tick
            save(args.evidence / 'listening.json', listening)
            url = 'http://' + listening['address']
            stop.check()
            tick = time.perf_counter()
            controller = Q.Controller(template, url, 'quasar-27b', timeout=14400, metrics=metrics)
            stages['controller_health_s'] = time.perf_counter() - tick
            tick = time.perf_counter()
            warm = {}
            stop.run(lambda: controller.complete(dict(model=controller.model, messages=[dict(role='user', content='Reply with the single word ready.')],
                max_completion_tokens=16), lambda *a: None, lambda *a: None,
                lambda meta, request, text, finish, usage: warm.update(text=text, finish=finish, usage=usage),
                lambda abort: lambda: None), controller.abort)
            C.require(warm['text'].strip().rstrip('.').lower() == 'ready', 'startup language smoke failed')
            policy_health = controller.health_factory().health()
            verify_cancellation_policy(policy_health, recovery_mode)
            native_ready_paths = list((args.evidence / 'native').glob('owner-*/native/ready.json'))
            C.require(len(native_ready_paths) == 1, 'single native ready receipt required')
            native_ready = C.strict_json(native_ready_paths[0].read_bytes())
            cache_configured = (any(env.get(k) for k in ('AUGUR_CHECKPOINT_STORE',
                'AUGUR_INPUT_PREFIX_CACHE_ROOT', 'AUGUR_INPUT_PREFIX_CACHE_TOKENS'))
                or env.get('AUGUR_INPUT_PREFIX_CACHE_PROMPT_END') == '1')
            cache_enabled = verify_cache_restore_policy(native_ready, restore_workers, restore_diagnostic,
                producer['sha256'], cache_configured)
            save(args.evidence / 'native-policy.json', dict(requested=recovery_mode, health=policy_health,
                cache_restore_workers=restore_workers, cache_restore_diagnostic=restore_diagnostic,
                cache_restore_selection=restore_selection, checkpoint_manager_enabled=cache_enabled,
                native_ready_sha256=sha(native_ready_paths[0])))
            stages['warm_completion_s'] = time.perf_counter() - tick
            tick = time.perf_counter()
            server = Server(('127.0.0.1', args.port), controller, key)
            server.assets = args.evidence / 'public-source' / 'augur_web'
            stop.check()
            stages['public_bind_s'] = time.perf_counter() - tick
            launch = dict(event='augur_product_ready', url='http://127.0.0.1:' + str(server.server_address[1]),
                launch_to_first_usable_response_s=time.perf_counter()-started,
                startup_clock='local_module_entry_to_public_bind_after_native_language_smoke',
                external_process_to_first_http_response_measured=False, stages=stages,
                native_owner_resident=True, vision_initialized=True, dflash_active=dflash_active, profile=profile,
                context_tokens=Q.CAPACITY, maximum_output_tokens=Q.MAX_OUTPUT,
                cancellation_recovery=recovery_mode,
                cache_restore_workers=restore_workers, cache_restore_diagnostic=restore_diagnostic,
                cache_restore_selection=restore_selection, checkpoint_manager_enabled=cache_enabled,
                product_ready=False, numerical_qualified=False)
            save(args.evidence / 'ready.json', launch)
            # The bearer capability is a URL fragment, not an HTTP request/query.
            print(json.dumps(dict(launch, browser_url=launch['url'] + '/#key=' + key)), flush=True)
            if args.serve_seconds:
                timer = threading.Timer(args.serve_seconds, server.shutdown)
                timer.daemon = True
                timer.start()
            def stop_serving():
                controller.abort()
                server.shutdown()

            stop.run(lambda: server.serve_forever(poll_interval=.2), stop_serving)
        except (KeyboardInterrupt, StopRequested):
            pass
        except BaseException as error:
            run_error = error
            save(args.evidence / 'failure.json', dict(type=type(error).__name__, error=str(error)))
            raise
        finally:
            if process:
                stopped = cleanup(process, server, controller, timer, url, args.evidence)
                stopped['stop_request'] = stop.receipt()
                save(args.evidence / 'stopped.json', stopped)
                if run_error is None:
                    C.require(stopped['cleanup_verified'], 'product cleanup could not be verified')


def main(argv=None):
    with StopSignals() as stop:
        try:
            launch(argv, stop)
        except StopRequested:
            pass


if __name__ == '__main__':
    main()
