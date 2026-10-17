"""Loopback-only read-only GPU and Augur receipt monitor; no credentials or prompts served."""
import argparse
import collections
import csv
import io
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ASSETS = Path(__file__).with_name('augur_monitor_web')


def finite_ratio(a, b):
    if not (type(a) in (int, float) and type(b) in (int, float)
            and math.isfinite(a) and math.isfinite(b) and a >= 0 and b > 0):
        return None
    return a / b


def read_json(path):
    return json.loads(path.read_bytes())


def token_count(value):
    return type(value) is int and value >= 0


def prefill_accounting(detail):
    """Count executed input rows, never cached tokens or merely reserved KV capacity."""
    if not isinstance(detail, dict):
        return None, None, 'unknown'
    if 'language_prefill' in detail:
        return prefill_accounting(detail['language_prefill'])
    cache = detail.get('input_prefix_cache')
    plan = detail.get('plan')
    cached_count = None
    planned_count = None
    if cache is not None:
        if not isinstance(cache, dict) or not token_count(cache.get('new_prefill_tokens')):
            return None, None, 'invalid_native_count'
        cached_count = cache['new_prefill_tokens']
        if cache.get('exact_input_hit') is True and cached_count != 0:
            return None, None, 'conflicting_native_counts'
    if isinstance(plan, dict) and 'widths' in plan:
        widths = plan['widths']
        if (not isinstance(widths, list)
                or any(not token_count(n) or n == 0 for n in widths)):
            return None, None, 'invalid_native_count'
        planned_count = sum(widths)
    if cached_count is not None and planned_count is not None and cached_count != planned_count:
        return None, None, 'conflicting_native_counts'
    if cached_count is not None:
        return cached_count, 'native_input_prefix_cache', 'counted'
    if planned_count is not None:
        return planned_count, 'native_plan_widths', 'counted'
    return None, None, 'unknown'


def native_prefill_seconds(detail, tokens):
    """Use execution counters covering exactly the rows in the numerator.

    Native execute includes upload, forward/head and draft-feature capture, ending
    at stream synchronization. It excludes phase setup/release and disk snapshots.
    Older split-prefix receipts omit the first execution: never divide all rows by
    only the suffix clock or substitute the outer request wall clock.
    """
    if tokens == 0:
        return 0.0
    if not token_count(tokens) or not isinstance(detail, dict):
        return None
    if 'language_prefill' in detail:
        return native_prefill_seconds(detail['language_prefill'], tokens)
    if 'suffix_prefill' in detail:
        parts = [part for part in (detail.get('prefix_prefill'), detail.get('suffix_prefill'))
                 if part is not None]
    else:
        parts = [detail]
    total_tokens, seconds = 0, 0.0
    for part in parts:
        if not isinstance(part, dict):
            return None
        plan = part.get('plan')
        widths = plan.get('widths') if isinstance(plan, dict) else None
        elapsed = part.get('execute_with_upload_head_capture_seconds')
        if (not isinstance(widths, list) or not widths
                or any(not token_count(n) or n == 0 for n in widths)
                or type(elapsed) not in (int, float)
                or not math.isfinite(elapsed) or elapsed <= 0):
            return None
        total_tokens += sum(widths)
        seconds += elapsed
    return seconds if total_tokens == tokens and seconds > 0 and math.isfinite(seconds) else None


def set_prefill_rate(row):
    tokens = row['prefill_tokens']
    row['prefill_tps'] = finite_ratio(tokens, row['prefill_s']) if tokens != 0 else None
    row['prefill_rate_status'] = ('not_executed' if tokens == 0 else
        'measured' if row['prefill_tps'] is not None else
        'invalid_clock' if tokens is not None else row['prefill_count_status'])
    row['prefill_clock'] = 'native_execute_with_upload_head_capture_seconds'
    return row


def receipt_context(receipt):
    session = receipt.get('session') or {}
    consumed = session.get('consumed_token_ids')
    if isinstance(consumed, list) and type(session.get('pending_token')) is int:
        return len(consumed) + 1
    return None


def context_accounting(live, latest, ready, online):
    """Logical token occupancy, not allocated VRAM or cumulative server throughput."""
    capacity = ready.get('capacity')
    allocated = (ready.get('target') or {}).get('kv_cache_bytes')
    used = None
    scope = 'unknown'
    if live is not None:
        capacity = live.get('context_capacity_tokens') or capacity
        if live.get('status') in ('prefill', 'decode', 'completed'):
            used = live.get('context_used_tokens')
            scope = 'prefill_input' if live.get('status') == 'prefill' else ('live' if live.get('status') == 'decode' else 'last_completed')
        else:
            scope = 'failed_or_cancelled'
    elif latest is not None:
        used = latest.get('context_used_tokens')
        scope = 'last_completed'
    if type(capacity) is not int or capacity <= 0:
        capacity = None
    if not token_count(used) or capacity is None or used > capacity:
        used = None
    threshold = capacity * 95 // 100 if capacity is not None else None
    return dict(capacity_tokens=capacity, used_tokens=used, scope=scope,
                utilization=finite_ratio(used, capacity), compression_threshold_tokens=threshold,
                remaining_to_compression_tokens=max(0, threshold-used) if used is not None else None,
                compression_due=used >= threshold if used is not None else None,
                kv_allocated_bytes=allocated if token_count(allocated) and online else 0 if not online else None,
                engine_online=online, compression_owner='client_at_request_boundary',
                compression_automatic_in_engine=False)


def decode_identity(sampling):
    """Whitelist mode metadata; never expose native token/hidden-state diagnostics."""
    preset = sampling.get('preset') if isinstance(sampling, dict) else None
    mode = 'sampling' if preset == 'ninfer_thinking_v1' else 'greedy' if preset == 'greedy' else 'unknown'
    return dict(decode_mode=mode, sampling_preset=preset if mode != 'unknown' else None)


def summarize_receipt(receipt):
    detail = receipt.get('decode_detail', {}).get('dflash') or {}
    count = len(receipt.get('output_token_ids', []))
    prefill = receipt.get('prefill_wall_s', 0)
    whole = receipt.get('whole_wall_s_before_receipt', 0)
    seconds = whole - prefill if (type(whole) in (int, float) and type(prefill) in (int, float)) else None
    new_tokens, source, count_status = prefill_accounting(receipt.get('prefill_detail'))
    return set_prefill_rate(dict(**decode_identity(receipt.get('decode_detail', {}).get('sampling')),
                request_id=receipt.get('request_id'), output_tokens=count,
                context_used_tokens=receipt_context(receipt),
                vision_residency=receipt.get('decode_detail', {}).get('vision_residency'),
                selected_including_eos=receipt.get('selected_tokens_including_eos'),
                prefill_s=native_prefill_seconds(receipt.get('prefill_detail'), new_tokens),
                prefill_wall_s=prefill, prefill_tokens=new_tokens, prefill_count_source=source,
                prefill_count_status=count_status, generation_s=seconds, native_generation_tps=finite_ratio(count, seconds),
                completed_unix_s=int(receipt.get('generation_complete_unix_ns', 0))/1e9,
                stop=receipt.get('stop_reason'), dflash_active=detail.get('active', False),
                rounds=detail.get('rounds'), accepted=detail.get('accepted_draft_tokens'),
                proposed=detail.get('proposed_draft_tokens'), checked=detail.get('checked_draft_tokens'),
                accepted_over_proposed=finite_ratio(detail.get('accepted_draft_tokens'), detail.get('proposed_draft_tokens')),
                accepted_over_checked=finite_ratio(detail.get('accepted_draft_tokens'), detail.get('checked_draft_tokens')),
                mean_accepted_drafts=finite_ratio(detail.get('accepted_draft_tokens'), detail.get('rounds')),
                policy=(detail.get('adaptive_gate_decisions') or {}).get('policy')))


def attach_http_metrics(row, metrics):
    """Join only counters and timings; never expose request or response bodies."""
    if row.get('decode_mode', 'unknown') == 'unknown' and isinstance(metrics.get('sampling'), dict):
        row.update(decode_identity(metrics['sampling']))
    usage = metrics.get('usage') or {}
    prompt = usage.get('prompt_tokens')
    cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens')
    valid_input = token_count(prompt) and token_count(cached) and cached <= prompt
    # Historical receipts without native counters can use canonical full-history input.
    # Appends may include a pending token and are not inferred from total HTTP usage.
    if (row['prefill_count_status'] == 'unknown' and valid_input
            and metrics.get('session_mode') == 'canonical_full_history_open'):
        row.update(prefill_tokens=prompt-cached, prefill_count_source='http_uncached_prompt_tokens',
                   prefill_count_status='counted')
    row.update(http_wall_s=metrics.get('api_wall_s'), http_ttft_s=metrics.get('native_ttft_s'),
               http_output_tps=finite_ratio(row['output_tokens'], metrics.get('api_wall_s')),
               input_tokens=prompt, cached_tokens=cached,
               input_cache_coverage=finite_ratio(cached, prompt) if valid_input else None,
               session_mode=metrics.get('session_mode'))
    return set_prefill_rate(row)


class Sampler:
    def __init__(self, runs_root, live_file, engine_port):
        self.runs_root, self.live_file, self.engine_port = runs_root, live_file, engine_port
        self.lock, self.closed = threading.Lock(), threading.Event()
        self.state = dict(status='starting', history=[])
        self.history = collections.deque(maxlen=150)
        self.receipts = {}

    def collect(self):
        errors = []
        gpu = None
        try:
            proc = subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,memory.free',
                                   '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3,
                                  creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), check=True)
            fields = next(csv.reader(io.StringIO(proc.stdout)))
            def number(index):
                try: return float(fields[index].strip())
                except (ValueError, IndexError): return None
            gpu = dict(index=number(0), name=fields[1].strip(), utilization=number(2),
                       used_mib=number(3), total_mib=number(4), temperature_c=number(5), power_w=number(6),
                       free_mib=number(7), free_memory_source="nvidia-smi.memory.free")
        except (OSError, subprocess.SubprocessError, StopIteration, IndexError):
            errors.append('GPU sampling unavailable')
        try:
            with socket.create_connection(('127.0.0.1', self.engine_port), timeout=.4): pass
            online = True
        except OSError: online = False
        run = None
        try:
            candidates = [p for p in self.runs_root.iterdir() if p.is_dir() and (p/'api-metrics.jsonl').is_file()]
            run = max(candidates, key=lambda p: p.name) if candidates else None
        except OSError: errors.append('Run directory unavailable')
        live = None
        try:
            live = read_json(self.live_file)
            live['fresh'] = time.time() - live.get('updated_unix_s', 0) < 4 and online
        except (OSError, ValueError): pass
        latest, recent, active = None, [], None
        ready = {}
        if run:
            try:
                ready_paths = list(run.glob('native/owner-*/native/ready.json'))
                if ready_paths:
                    ready = read_json(max(ready_paths, key=lambda p: p.stat().st_mtime_ns))
            except (OSError, ValueError, TypeError):
                pass
            try:
                paths = sorted(run.glob('native/owner-*/native/turns/turn-*.json'), key=lambda p:p.stat().st_mtime_ns, reverse=True)[:40]
                for p in paths:
                    stat = p.stat()
                    if not stat.st_size:
                        if active is None: active = dict(request_id=p.stem, started_unix_s=stat.st_mtime)
                        continue
                    identity=(str(p),stat.st_size,stat.st_mtime_ns)
                    if identity not in self.receipts:
                        self.receipts[identity]=summarize_receipt(read_json(p))
                    recent.append(dict(self.receipts[identity]))
                self.receipts={k:v for k,v in self.receipts.items() if any(str(p)==k[0] for p in paths)}
                metrics=[]
                with (run/'api-metrics.jsonl').open('rb') as stream:
                    stream.seek(max(0, stream.seek(0,2)-2*1024*1024))
                    for line in stream.read().splitlines():
                        try: metrics.append(json.loads(line))
                        except ValueError: continue
                by_id={m.get('native_request_id'):m for m in metrics}
                for row in recent:
                    m=by_id.get(row['request_id'],{})
                    attach_http_metrics(row, m)
                latest=recent[0] if recent else None
                if active and latest and latest['completed_unix_s'] > active['started_unix_s']: active=None
            except (OSError, ValueError, TypeError): errors.append('Waiting for a complete receipt')
        real_live=live if live and live.get('fresh') else None
        if real_live and real_live.get('decode_mode', 'unknown') == 'unknown':
            real_live.update(decode_identity((ready.get('target') or {}).get('sampling')))
        if real_live and real_live['status'] == 'failed_or_cancelled':
            status='failed_or_cancelled'; active=None
        elif real_live and real_live['status'] in ('prefill','decode'): status=real_live['status']
        elif online and active: status='running_without_token_observer'
        else: status='idle' if online else 'offline'
        # Keep the last budget visible after shutdown without treating it as live GPU occupancy.
        context_live = real_live
        if context_live is None and not online and live is not None:
            context_live = live
        context = context_accounting(context_live, latest, ready, online)
        point=dict(time=time.time(),gpu=gpu['utilization'] if gpu else None,
                   vram_gib=gpu['used_mib']/1024 if gpu and gpu['used_mib'] is not None else None,
                   tps=real_live.get('rolling_5s_tps') if real_live else None,
                   decode_mode=real_live.get('decode_mode', 'unknown') if real_live else 'unknown')
        self.history.append(point)
        return dict(schema=1, updated_unix_s=time.time(), status=status, engine_online=online,
                    gpu=gpu, live=real_live, active=active, latest=latest, recent=recent[:8], context=context,
                    history=list(self.history),errors=errors,run=run.name if run else None,
                    dflash_scope='latest_completed_request',token_observer_connected=real_live is not None)

    def loop(self):
        while not self.closed.is_set():
            try: value=self.collect()
            except Exception: value=dict(status='sampling_error',errors=['Monitor sampling failed'],updated_unix_s=time.time())
            with self.lock: self.state=value
            self.closed.wait(2)


def create_server(runs_root, live_file, engine_port, port=0):
    sampler=Sampler(runs_root,live_file,engine_port)
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            host=self.headers.get('Host','')
            if host not in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'):
                self.send_error(403);return
            route=self.path.split('?',1)[0]
            if route=='/api/metrics':
                current = self.server.sampler
                with current.lock: data=json.dumps(current.state,allow_nan=False).encode()
                mime='application/json; charset=utf-8'
            elif route in ('/','/app.js','/i18n.js','/style.css'):
                name={'/':'index.html','/app.js':'app.js','/i18n.js':'i18n.js','/style.css':'style.css'}[route]
                data=(ASSETS/name).read_bytes()
                mime={'/':'text/html','/app.js':'text/javascript','/i18n.js':'text/javascript','/style.css':'text/css'}[route]+'; charset=utf-8'
            else: self.send_error(404);return
            self.send_response(200)
            self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers();self.wfile.write(data)
        def do_POST(self):
            # Loopback-only stop for the lifecycle script. The Host check above rejects any other
            # origin, and a GET-only page cannot reach here without a same-origin script.
            host=self.headers.get('Host','')
            if host not in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'):
                self.send_error(403);return
            if self.path.split('?',1)[0]!='/api/stop':
                self.send_error(404);return
            length=self.headers.get('Content-Length','0')
            if not (length.isdigit() and int(length)<=1024):
                self.send_error(413);return
            self.rfile.read(int(length))
            self.send_response(202);self.send_header('Content-Length','0');self.send_header('Connection','close')
            self.end_headers()
            threading.Thread(target=self.server.shutdown,daemon=True).start()
        def log_message(self,*args): pass
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    server.sampler = sampler
    return server,sampler


def serve(args):
    server,sampler=create_server(args.runs_root,args.live_file,args.engine_port,args.port)
    threading.Thread(target=sampler.loop,daemon=True).start()
    print(json.dumps(dict(url=f'http://127.0.0.1:{server.server_port}/',mode='read_only')),flush=True)
    try: server.serve_forever()
    finally: sampler.closed.set();server.server_close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--live-file',type=Path,required=True)
    p.add_argument('--port',type=int,default=8295);p.add_argument('--engine-port',type=int,default=8294)
    serve(p.parse_args())
