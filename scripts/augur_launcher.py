"""Loopback-only setup UI owning one cooperative Augur subprocess.

The random fragment bearer, strict Host/Origin checks, bounded JSON and fixed
operations protect local process/file capabilities. Never executes shell text
from a request. Closing a browser tab does not stop inference.
"""
from __future__ import annotations
import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
import queue
import secrets
import socket
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import augur_launch_config as C
from augur_startup import failure_detail, failed_start_retired

ASSETS = Path(__file__).with_name('augur_setup_web')
CREATE_NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)


def valid_key(key):
    return (type(key) is str and 32 <= len(key) <= 512
            and all(33 <= ord(c) <= 126 for c in key))


def digest(key):
    return hashlib.sha256(key.encode('ascii', 'replace')).hexdigest()


def write_key(path, key):
    if not valid_key(key):
        raise ValueError(C.L('API Key 至少 32 个可见 ASCII 字符，不能含空白。', 'The API key needs at least 32 visible ASCII characters and no whitespace.'))
    path = Path(path)
    temporary = path.with_name('.api-key-' + uuid.uuid4().hex + '.tmp')
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w', encoding='ascii') as stream:
        stream.write(key + '\n')
    os.replace(temporary, path)


class UserEnvironment:
    """Per-user environment (HKCU\\Environment); no administrator rights, no system variables."""
    def read(self, name):
        if os.name != 'nt':
            return None
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as key:
                value, kind = winreg.QueryValueEx(key, name)
            return value if kind == winreg.REG_SZ and type(value) is str else None
        except OSError:
            return None

    def write(self, values):
        if os.name != 'nt':
            return False
        import winreg
        changed = False
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment', 0, winreg.KEY_SET_VALUE) as key:
            for name, value in values.items():
                if self.read(name) != value:
                    winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value)
                    changed = True
        if changed:
            self.broadcast()
        return True

    def remove(self, values):
        if os.name != 'nt':
            return
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment', 0, winreg.KEY_SET_VALUE) as key:
            for name in values:
                try:
                    winreg.DeleteValue(key, name)
                except OSError:
                    pass
        self.broadcast()

    @staticmethod
    def broadcast():
        # Tell Explorer so programs started afterwards inherit the new values.
        import ctypes
        result = ctypes.c_size_t()
        ctypes.windll.user32.SendMessageTimeoutW(0xFFFF, 0x001A, 0, 'Environment', 0x0002, 2000, ctypes.byref(result))


E = UserEnvironment()


class Manager:
    def __init__(self, package, data, *, verify=True, cache_scope='chat'):
        self.package, self.data = Path(package).resolve(), Path(data).resolve()
        self.cache_scope = C.cache_scope(cache_scope)
        self.data.mkdir(parents=True, exist_ok=True)
        self.config_path = self.data / 'config.json'
        self.preferences = C.load_preferences(self.data)
        self.config = C.defaults(self.package)
        self.error = None
        if self.config_path.exists():
            try:
                self.config = C.normalize(json.loads(self.config_path.read_bytes()), self.package)
            except (ValueError, OSError) as error:
                self.error = C.L('已保存配置无法读取，原文件未覆盖：', 'The saved configuration could not be read; the file was left unchanged: ') + str(error)
        if verify:
            import augur_product as P
            P.verified_build(self.package)
        self.lock = threading.RLock()
        self.operation = None
        self.worker = None
        self.process = None
        self.active_config = None
        self.ready = None
        self.run_dir = None
        self.closed = False
        self.phase = 'stopped'
        self.message = C.L('请选择模型路径，检查后启动。', 'Choose the model folders, check them, then start.')
        self.last_cleanup = None
        self.monitor = None
        self.monitor_url = None
        self.monitor_error = None
        self.progress = None

    def state(self):
        with self.lock:
            running = self.process is not None and self.process.poll() is None
            progress = None
            if self.progress is not None:
                progress = dict(self.progress)
                now = progress.pop('finished_at', None) or time.monotonic()
                progress['elapsed_s'] = max(0, now - progress.pop('started_at'))
                progress['stage_elapsed_s'] = max(0, now - progress.pop('stage_started_at'))
            phase = 'service_exited' if self.phase == 'running' and not running else self.phase
            return dict(config=self.config, operation=self.operation, phase=phase, progress=progress,
                        message=self.message, error=self.error, running=running,
                        ready=self.ready if running else None, cleanup=self.last_cleanup,
                        active_config=self.active_config if running else None,
                        context_choices=list(C.CONTEXT_CHOICES),
                        config_path=str(self.config_path), language=C.LANG,
                        language_chosen=self.preferences['chosen'], sampling='ninfer_thinking_v1_gate045_fixed_b8',
                        monitor_url=self.monitor_url, monitor_error=self.monitor_error)

    def update(self, **values):
        with self.lock:
            for name, value in values.items():
                setattr(self, name, value)

    def stage(self, name, message):
        with self.lock:
            if self.progress is not None:
                self.progress.update(stage=name, stage_started_at=time.monotonic())
            self.message = message

    def submit(self, action, config=None):
        if action not in ('validate', 'restart', 'stop', 'quantize', 'rekey'):
            raise ValueError(C.L('不支持的操作。', 'Unsupported operation.'))
        candidate = None if action in ('stop', 'rekey') else C.normalize(config, self.package)
        with self.lock:
            if self.closed or self.operation is not None:
                raise RuntimeError(C.L('已有操作正在进行，请等待结束。', 'Another operation is running; wait for it to finish.'))
            self.operation, self.error = action, None
            now = time.monotonic()
            self.progress = dict(action=action, stage='queued', status='running',
                                 started_at=now, stage_started_at=now, finished_at=None)
            self.worker = threading.Thread(target=self._work, args=(action, candidate), daemon=False)
            self.worker.start()
        return dict(accepted=True, operation=action)

    def api_key_file(self):
        """Private stable credential shared by chat and API clients across restarts."""
        path = self.data / 'api-key.txt'
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(descriptor, 'w', encoding='ascii') as stream:
                stream.write(secrets.token_urlsafe(32) + '\n')
        key = path.read_text(encoding='ascii').strip()
        if not valid_key(key):
            raise ValueError(C.L('本机 api-key.txt 无效；保留原文件，请修正后重试。', 'The local api-key.txt is invalid. It was kept; fix it and try again.'))
        return path

    def sync_api_key(self, config=None):
        """One local key, reachable two ways: the private api-key.txt and (by default) the user
        environment. A valid AUGUR_API_KEY the user set themselves wins over the file; a value
        this launcher exported earlier never does, so a reset is not undone by a stale variable."""
        config = config or self.config
        path = self.data / 'api-key.txt'
        record = self.data / 'env-export.json'
        try:
            exported = json.loads(record.read_bytes()).get('sha256')
        except (OSError, ValueError, AttributeError):
            exported = None
        user = E.read('AUGUR_API_KEY') or os.environ.get('AUGUR_API_KEY', '')
        current = path.read_text(encoding='ascii').strip() if path.is_file() else None
        if valid_key(user) and user != current and digest(user) != exported:
            write_key(path, user)
        key = self.api_key_file().read_text(encoding='ascii').strip()
        values = dict(AUGUR_API_KEY=key, AUGUR_BASE_URL='http://127.0.0.1:' + str(config['port']) + '/v1',
                      AUGUR_MODEL='quasar-27b')
        if config['export_env']:
            if E.write(values):
                C.atomic_json(record, dict(sha256=digest(key)))
        elif exported is not None and digest(E.read('AUGUR_API_KEY') or '') == exported:
            E.remove(values)  # switched off: retract only what this launcher wrote
            record.unlink(missing_ok=True)
        return key

    def connection(self):
        """Values another agent needs; served only behind the setup-page bearer."""
        with self.lock:
            config = self.active_config or self.config
        key = self.api_key_file().read_text(encoding='ascii').strip()
        return dict(base_url='http://127.0.0.1:' + str(config['port']) + '/v1', model='quasar-27b',
                    api_key=key, env_exported=bool(config['export_env']) and E.read('AUGUR_API_KEY') == key,
                    key_file=str(self.data / 'api-key.txt'))

    def _rekey(self):
        self.stage('rekey', C.L('正在生成新的 API Key。', 'Generating a new API key.'))
        write_key(self.data / 'api-key.txt', secrets.token_urlsafe(32))
        self.sync_api_key()
        if self.process is not None and self.process.poll() is None and self.active_config:
            config = dict(self.active_config)
            admission, draft = self._preflight(config)
            self._stop_engine()
            self._start_engine(config, admission, draft)
        self.update(message=C.L('已更换 API Key。旧密钥立即失效；已打开的终端或 Agent 需要重新读取。', 'API key replaced. The old key stops working now; reopen terminals or agents to pick up the new one.'))

    def _preflight(self, candidate):
        self.stage('validating', C.L('正在校验主模型和 DFlash 文件身份；现有服务继续运行。', 'Checking the target model and DFlash files; the current service keeps running.'))
        self.api_key_file()  # Validate before stopping the current owner.
        admission = C.target_preflight(candidate)
        draft = C.draft_environment(candidate)
        if candidate['cache_enabled']:
            free = C.writable_directory(C.prefix_cache_root(candidate['cache_dir'], candidate['context_tokens'], self.cache_scope))
            # Conservative free-space gate for one long entry plus a new capture.
            # The detailed cache bill is documented separately; quotas remain native.
            if free < 24 * 1024**3 and candidate['context_tokens'] == 262144:
                raise ValueError(C.L('256K 持久化缓存建议至少预留 24 GiB 空闲空间；请选择空间足够的缓存盘。', 'A 256K persistent cache needs at least 24 GiB free; choose a cache drive with more space.'))
        if candidate['port'] != (self.active_config.get('port') if self.active_config and self.process and self.process.poll() is None else None):
            with socket.socket() as probe:
                if os.name == 'nt':
                    probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                probe.bind(('127.0.0.1', candidate['port']))
        return admission, draft

    def _work(self, action, candidate):
        try:
            if action == 'stop':
                self._stop_engine()
            elif action == 'rekey':
                self._rekey()
            elif action == 'quantize':
                self._quantize(candidate)
            else:
                admission, draft = self._preflight(candidate)
                if action == 'validate':
                    self.update(message=C.L('模型与配置校验通过；尚未保存或重启。', 'Model and settings check passed; nothing was saved or restarted.'))
                else:
                    self._stop_engine()
                    if self.closed:
                        raise RuntimeError(C.L('启动器正在退出，未启动新服务。', 'The launcher is exiting; no new service was started.'))
                    self.stage('saving', C.L('正在保存配置。', 'Saving settings.'))
                    C.atomic_json(self.config_path, candidate)
                    self.update(config=candidate)
                    self._start_engine(candidate, admission, draft)
        except Exception as error:
            self.update(error=str(error), message=C.L('操作未完成；请检查错误信息。', 'The operation did not finish; see the error below.'))
        finally:
            with self.lock:
                if self.progress is not None:
                    self.progress.update(status='failed' if self.error else 'complete',
                                         finished_at=time.monotonic())
                self.operation = None

    def _stop_engine(self):
        self.stage('stopping', C.L('正在关闭当前服务并确认 GPU 资源释放。', 'Stopping the service and confirming the GPU is released.'))
        process = self.process
        if process is None:
            self.update(phase='stopped', message=C.L('服务已关闭，配置页保留。', 'Service stopped. This page stays open.'), ready=None)
            return
        failed_start = self.phase == 'startup_failed' and self.ready is None
        self.update(phase='stopping', message=C.L('正在中断生成、排空请求并释放 GPU，请等待清理确认。', 'Interrupting generation, draining requests and releasing the GPU; wait for cleanup to be confirmed.'))
        if process.poll() is None:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()  # private pipe EOF, never a global process-name kill
            process.wait(timeout=300)
        receipt = self.run_dir / 'launch/stopped.json'
        result = json.loads(receipt.read_bytes()) if receipt.is_file() else {}
        self.update(last_cleanup=result, ready=None)
        retired_failure = failed_start and failed_start_retired(self.run_dir, result)
        if not retired_failure and (process.returncode != 0 or result.get('cleanup_verified') is not True):
            self.update(phase='cleanup_unconfirmed')
            raise RuntimeError(C.L('引擎已退出但资源清理未通过验证；为避免重叠占用，禁止启动新 owner。请查看本次运行日志。', 'The engine exited but cleanup could not be verified. To avoid overlapping GPU use, no new engine will start. See this run\'s log.'))
        if process.stdout:
            process.stdout.close()
        self.update(process=None, active_config=None, phase='stopped', message=(
            C.L('失败的启动进程已退出，可重新启动。', 'The failed startup process has exited; ready to retry.') if retired_failure else
            C.L('服务已关闭，GPU 分配清理已确认。', 'Service stopped; GPU cleanup confirmed.')))

    def _start_engine(self, config, admission, draft):
        self.run_dir = self.data / 'runs' / (time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8])
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.sync_api_key(config)
        draft_file = self.run_dir / 'draft-environment.json'
        C.atomic_json(draft_file, draft)
        arguments = [str(self.package / 'runtime/python/python.exe'), '-I', '-X', 'utf8', '-B',
                     str(self.package / 'source/scripts/augur_portable.py'), 'engine',
                     '--model-dir', config['model_dir'], '--manifest', admission['manifest'],
                     '--manifest-sha256', admission['manifest_sha256'], '--draft-environment', str(draft_file),
                     '--context-tokens', str(config['context_tokens']),
                     '--default-output-tokens', str(config['default_output_tokens']),
                     '--port', str(config['port']), '--vision-residency', config['vision_residency'],
                     '--evidence', str(self.run_dir / 'launch'), '--live-file', str(self.data / 'live.json'),
                     '--api-key-file', str(self.api_key_file()), '--stop-on-stdin']
        if config['cache_enabled']:
            arguments += ['--prefix-cache-root', str(C.prefix_cache_root(config['cache_dir'], config['context_tokens'], self.cache_scope))]
        self.stage('loading', C.L('正在加载模型、DFlash 与 KV；启动完成后可进入聊天。', 'Loading the model, DFlash and KV; chat opens when this finishes.'))
        self.update(phase='starting', ready=None)
        with (self.run_dir / 'launcher.stderr.log').open('xb') as errors:
            process = subprocess.Popen(arguments, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=errors, creationflags=CREATE_NO_WINDOW)
        self.update(process=process, active_config=dict(config))
        results = queue.Queue()
        def read_ready():
            try:
                raw = process.stdout.readline(1048577)
                if len(raw) > 1048576:
                    raise ValueError(C.L('启动回执过大。', 'Startup receipt too large.'))
                if not raw.strip():
                    try:
                        code = process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        code = process.poll()
                    raise ValueError(failure_detail(self.run_dir, code))
                try:
                    value = json.loads(raw)
                except (ValueError, UnicodeError) as error:
                    raise ValueError(C.L('引擎返回了无效的启动回执。', 'The engine returned an invalid startup receipt.')) from error
                if not isinstance(value, dict):
                    raise ValueError(C.L('引擎返回了无效的启动回执。', 'The engine returned an invalid startup receipt.'))
                if value.get('event') != 'augur_product_ready' or value.get('context_tokens') != config['context_tokens']:
                    raise ValueError(C.L('引擎启动回执与所选 KV 配置不一致。', 'The engine startup receipt does not match the selected KV capacity.'))
                results.put((True, value))
            except Exception as error:
                results.put((False, error))
        threading.Thread(target=read_ready, daemon=True).start()
        try:
            try:
                success, value = results.get(timeout=300)
            except queue.Empty as error:
                raise RuntimeError(C.L('引擎启动超过 300 秒；请查看 ', 'Engine startup exceeded 300 seconds; see ') + str(self.run_dir / 'launcher.stderr.log')) from error
            if not success:
                raise RuntimeError(C.L('引擎启动失败：', 'Engine failed to start: ') + str(value) + C.L('；请查看 ', '; see ') + str(self.run_dir / 'launcher.stderr.log'))
        except Exception:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            self.update(phase='startup_failed')
            raise
        self.update(ready=value, phase='running')
        self.stage('monitor', C.L('模型已就绪，正在连接实时监控。', 'Model ready; connecting the live monitor.'))
        try:
            self._start_monitor(config)
        except Exception as error:
            self.update(monitor_error=C.L('监控未能启动：', 'Monitor failed to start: ') + str(error))
        self.stage('ready', C.L('服务已启动，聊天和监控入口已更新。', 'Service started; chat and monitor links are ready.') if self.monitor_url else C.L('模型已启动；监控启动失败，请查看提示。', 'Model started, but the monitor failed; see the note.'))

    def _close_monitor(self):
        monitor = self.monitor
        if monitor is None:
            return
        server, sampler, serving, sampling = monitor
        sampler.closed.set()
        server.shutdown()
        server.server_close()
        serving.join(timeout=5)
        sampling.join(timeout=5)
        self.update(monitor=None, monitor_url=None)

    def _start_monitor(self, config):
        import augur_monitor as M
        if self.monitor is not None:
            # Keep the listening socket and existing browser URL across engine restarts.
            server, old_sampler, serving, old_sampling = self.monitor
            old_sampler.closed.set()
            old_sampling.join(timeout=5)
            sampler = M.Sampler(self.run_dir, self.data / 'live.json', config['port'])
            server.sampler = sampler
        else:
            port_file = self.data / 'monitor-port.json'
            port = 0
            try:
                saved = json.loads(port_file.read_bytes()).get('port')
                if type(saved) is int and 1024 <= saved <= 65535:
                    port = saved
            except (OSError, ValueError, AttributeError):
                pass
            try:
                server, sampler = M.create_server(self.run_dir, self.data / 'live.json', config['port'], port)
            except OSError:
                if not port:
                    raise
                server, sampler = M.create_server(self.run_dir, self.data / 'live.json', config['port'], 0)
            serving = threading.Thread(target=server.serve_forever, kwargs={'poll_interval':.2}, daemon=True)
            serving.start()
            try:
                C.atomic_json(port_file, dict(port=server.server_port))
            except OSError:
                pass  # The active monitor still works when its preferred port cannot be saved.
        sampling = threading.Thread(target=sampler.loop, daemon=True)
        sampling.start()
        self.update(monitor=(server,sampler,serving,sampling),
                    monitor_url='http://127.0.0.1:' + str(server.server_port) + '/', monitor_error=None)

    def _quantize(self, config):
        self.stage('quantizing', C.L('正在检查并量化 DFlash 权重；量化使用 CPU，不改动原文件。', 'Checking and quantizing the DFlash weights on the CPU; the original files are not changed.'))
        expected = C.quantization_source(config)
        if not config['quant_output_dir']:
            raise ValueError(C.L('请选择量化输出目录。', 'Choose an output folder for quantization.'))
        output = Path(config['quant_output_dir'])
        if output.exists():
            candidate = dict(config, draft_quantized_dir=str(output))
            C.draft_environment(candidate)  # reuse only complete admitted output; never overwrite
        else:
            C.writable_directory(output.parent)
            exe = self.package / 'bin/augur-quantize.exe'
            common = [config['draft_source_dir'], config['model_dir']]
            plan = subprocess.run([str(exe), 'dflash-plan', *common, '--weights-sha256', expected],
                                  capture_output=True, timeout=300, creationflags=CREATE_NO_WINDOW)
            if plan.returncode:
                raise ValueError(plan.stderr.decode('utf8', 'replace')[-6000:])
            self.update(message=C.L('正在执行 Rust Q8_0＋FP32 gate0.45 量化，请勿关闭启动器。', 'Running the Rust Q8_0 + FP32 gate0.45 quantizer; keep the launcher running.'))
            log_dir = self.data / 'quantization' / uuid.uuid4().hex
            log_dir.mkdir(parents=True, exist_ok=False)
            (log_dir / 'plan.json').write_bytes(plan.stdout)
            with (log_dir / 'stdout.log').open('xb') as out, (log_dir / 'stderr.log').open('xb') as err:
                result = subprocess.run([str(exe), 'dflash-convert', *common, str(output), '--weights-sha256', expected],
                                        stdout=out, stderr=err, timeout=3600, creationflags=CREATE_NO_WINDOW)
            if result.returncode:
                raise RuntimeError(C.L('量化失败，未切换模型；请查看 ', 'Quantization failed; the model was not switched. See ') + str(log_dir))
            candidate = dict(config, draft_quantized_dir=str(output))
            C.draft_environment(candidate)
        C.atomic_json(self.config_path, candidate)
        self.update(config=candidate, message=C.L('量化结果已验证并保存。点击“保存并重启服务”应用；原始权重保留。', 'Quantized draft verified and saved. Click "Save and restart" to use it; the original weights are kept.'))

    def close(self):
        self.closed = True
        if self.worker and self.worker.is_alive():
            self.worker.join()  # conversion or cooperative retirement must finish before exiting
        try:
            if self.process:
                self._stop_engine()
        finally:
            self._close_monitor()


def choose_directory():
    if os.name != 'nt':
        raise ValueError(C.L('目录选择框仅用于 Windows；也可以直接填写绝对路径。', 'The folder picker is Windows-only; you can type an absolute path instead.'))
    # Constant script, no request-derived shell text. The console stays hidden;
    # only the folder-picker requested by the user is interactive.
    script = """Add-Type -AssemblyName System.Windows.Forms;
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false);
$d = New-Object System.Windows.Forms.FolderBrowserDialog;
$d.Description = 'Select a local model or data directory';
if ($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { [Console]::Write($d.SelectedPath) }; $d.Dispose()"""
    system = Path(os.environ.get('SystemRoot', r'C:\Windows'))
    result = subprocess.run([str(system / 'System32/WindowsPowerShell/v1.0/powershell.exe'),
                             '-NoProfile', '-STA', '-Command', script], capture_output=True,
                            timeout=300, creationflags=CREATE_NO_WINDOW)
    if result.returncode:
        raise RuntimeError(C.L('目录选择框未完成，请直接粘贴本机路径。', 'The folder picker did not finish; paste a local path instead.'))
    return result.stdout.decode('utf-8-sig').strip()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # no request tokens or filesystem paths in access logs

    def send(self, code, value, content_type='application/json; charset=utf-8'):
        raw = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(code)
        for name, val in [('Content-Type', content_type), ('Content-Length', str(len(raw))),
                          ('Cache-Control', 'no-store'), ('X-Content-Type-Options', 'nosniff'),
                          ('Referrer-Policy', 'no-referrer'), ('Connection', 'close'),
                          ('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")]:
            self.send_header(name, val)
        self.end_headers()
        self.wfile.write(raw)

    def guard(self, auth=True):
        authority = '127.0.0.1:' + str(self.server.server_port)
        if self.headers.get_all('Host', []) != [authority]:
            raise PermissionError('Invalid Host')
        origins = self.headers.get_all('Origin', [])
        if origins and origins != ['http://' + authority]:
            raise PermissionError('Invalid Origin')
        if auth:
            values = self.headers.get_all('Authorization', [])
            if len(values) != 1 or not hmac.compare_digest(values[0], 'Bearer ' + self.server.key):
                raise PermissionError(C.L('需要本地启动授权，请从启动器打开配置页。', 'Local launcher authorization required; open this page from the launcher.'))

    def do_GET(self):
        try:
            if self.path == '/api/state':
                self.guard()
                return self.send(200, self.server.manager.state())
            if self.path == '/api/connection':
                self.guard()
                return self.send(200, self.server.manager.connection())
            self.guard(auth=False)
            files = {'/': ('index.html', 'text/html; charset=utf-8'),
                     '/open/chat': ('service.html', 'text/html; charset=utf-8'),
                     '/open/monitor': ('service.html', 'text/html; charset=utf-8'),
                     '/service.js': ('service.js', 'text/javascript; charset=utf-8'),
                     '/i18n.js': ('i18n.js', 'text/javascript; charset=utf-8'),
                     '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                     '/style.css': ('style.css', 'text/css; charset=utf-8')}
            if self.path not in files:
                return self.send(404, dict(error='Not found'))
            name, mime = files[self.path]
            self.send(200, (ASSETS / name).read_bytes(), mime)
        except PermissionError as error:
            self.send(403, dict(error=str(error)))
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        try:
            self.guard()
            lengths = self.headers.get_all('Content-Length', [])
            if self.headers.get('Transfer-Encoding') or len(lengths) != 1:
                raise ValueError(C.L('请求必须有唯一 Content-Length。', 'The request needs exactly one Content-Length.'))
            if self.headers.get('Content-Type') != 'application/json':
                raise ValueError(C.L('仅支持 application/json。', 'Only application/json is accepted.'))
            length = int(lengths[0])
            if not 0 < length <= 65536:
                raise ValueError(C.L('配置请求过大。', 'Request too large.'))
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError(C.L('请求内容不完整。', 'Incomplete request body.'))
            value = json.loads(raw, parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
            if type(value) is not dict:
                raise ValueError(C.L('请求必须是对象。', 'The request must be a JSON object.'))
            if self.path == '/api/browse':
                if value:
                    raise ValueError(C.L('目录选择不接受额外参数。', 'The folder picker takes no parameters.'))
                if not self.server.dialog_lock.acquire(blocking=False):
                    raise RuntimeError(C.L('已有目录选择框打开。', 'A folder picker is already open.'))
                try:
                    return self.send(200, dict(path=choose_directory()))
                finally:
                    self.server.dialog_lock.release()
            if self.path == '/api/language':
                if set(value) != {'language'}:
                    raise ValueError(C.L('语言设置只接受 language。', 'Language settings take only "language".'))
                manager = self.server.manager
                with manager.lock:
                    manager.preferences = C.save_language(manager.data, value['language'])
                return self.send(200, manager.preferences)
            if self.path == '/api/quit':
                if value:
                    raise ValueError(C.L('退出不接受额外参数。', 'Quit takes no parameters.'))
                self.server.quit()
                return self.send(202, dict(accepted=True, operation='quit'))
            actions = {'/api/validate':'validate', '/api/save-restart':'restart',
                       '/api/stop':'stop', '/api/quantize':'quantize', '/api/rekey':'rekey'}
            action = actions.get(self.path)
            if not action:
                return self.send(404, dict(error='Not found'))
            if action in ('stop', 'rekey') and value:
                raise ValueError(C.L('该操作不接受额外参数。', 'This operation takes no parameters.'))
            self.send(202, self.server.manager.submit(action, value))
        except PermissionError as error:
            self.send(403, dict(error=str(error)))
        except RuntimeError as error:
            self.send(409, dict(error=str(error)))
        except (ValueError, OSError) as error:
            self.send(400, dict(error=str(error)))


class Server(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, address, manager, key):
        self.manager, self.key = manager, key
        self.dialog_lock = threading.Lock()
        self.quitting = threading.Event()
        super().__init__(address, Handler)

    def quit(self):
        """Windowless start has no console to close: the page retires the engine, then the launcher.
        Cleanup runs in main() after serve_forever returns, exactly as for Ctrl+C."""
        if self.quitting.is_set():
            return
        if self.manager.operation is not None:
            raise RuntimeError(C.L('已有操作正在进行，请等待结束后再退出。', 'An operation is running; wait for it to finish before quitting.'))
        self.quitting.set()
        threading.Thread(target=self.shutdown, daemon=True).start()
    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(10)
        return connection, address


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--port', type=int, default=0, help='setup page port; independent of engine API port')
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--autostart', action='store_true', help='start the engine with the saved configuration')
    parser.add_argument('--cache-scope', default='chat',
                        help='persistent prefix-cache scope: runs using different scopes never reuse each other')
    args = parser.parse_args(argv)
    try:
        args.cache_scope = C.cache_scope(args.cache_scope)
    except ValueError as error:
        parser.error(str(error))
    if not 0 <= args.port <= 65535:
        parser.error('invalid setup port')
    data = args.data_root or Path(os.environ.get('LOCALAPPDATA', Path.home())) / 'Augur'
    data.mkdir(parents=True, exist_ok=True)
    # One launcher per user: a second start reopens the running setup page instead of a new owner.
    running = running_launcher(data)
    if running:
        print('Augur is already running: ' + running, flush=True)
        if not args.no_browser:
            webbrowser.open(running)
        return
    manager = Manager(args.build, data, cache_scope=args.cache_scope)
    if manager.error is None:
        try:
            manager.sync_api_key()  # environment ready for other agents even before the engine loads
        except (OSError, ValueError) as error:
            manager.update(error=C.L('API Key 未能同步：', 'The API key could not be synced: ') + str(error))
    key = secrets.token_urlsafe(32)
    server = Server(('127.0.0.1', args.port), manager, key)
    url = 'http://127.0.0.1:' + str(server.server_port) + '/#key=' + key
    instance = data / 'launcher.json'
    C.atomic_json(instance, dict(pid=os.getpid(), url=url))
    print('Augur settings (local private link): ' + url, flush=True)
    print('Keep this window open, or use "Quit Augur" in the web page.', flush=True)
    autostart(manager, args.autostart)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        manager.close()
        try:
            if json.loads(instance.read_bytes()).get('pid') == os.getpid():
                instance.unlink()
        except (OSError, ValueError):
            pass


def autostart(manager, enabled):
    """Double-click start: a configuration saved by the user is loaded without another click.
    A first run (no saved file) or an unreadable file waits for the setup page instead."""
    if not enabled or manager.error is not None or not manager.config_path.is_file():
        return False
    try:
        manager.submit('restart', manager.config)
    except (ValueError, RuntimeError) as error:
        manager.update(error=str(error), message=C.L('已保存的配置未能自动启动；请检查后手动启动。', 'The saved settings could not start automatically; check them and start by hand.'))
        return False
    return True


def running_launcher(data):
    """URL of a live launcher recorded in the private data directory, if its page still answers."""
    try:
        record = json.loads((Path(data) / 'launcher.json').read_bytes())
        url = record['url']
        host, port = url.split('//', 1)[1].split('/', 1)[0].split(':')
        if host != '127.0.0.1':
            return None
        with socket.create_connection((host, int(port)), timeout=1):
            return url
    except (OSError, ValueError, KeyError, IndexError):
        return None


if __name__ == '__main__':
    main()
