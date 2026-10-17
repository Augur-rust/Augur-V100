"""Augur decode benchmark: interleaved A/B arms over a fixed prompt suite.

Each arm starts one product owner on a private port with the Windows preset plus the arm's
overrides, sends every suite case once, stops the owner and checks that cleanup was verified.
Arms run in the given order (use ABBA, --order base,new,new,base) so drift cancels.

Native decode TPS = output tokens / native decode seconds (prefill and HTTP excluded).
Tokens per round counts the verifier's correction or bonus token. Every result carries the
selection-trajectory hash from the native receipt, so bit-identical arms are checked directly.
Results hold counters and hashes only; generated text is not saved. Run with the package's bundled
Python (runtime/python/python.exe) and stop any other owner on the GPU first.

  python scripts/augur_bench.py --build BUILD --model-dir MODEL --draft-environment DRAFT.json \\
      --out runs/glue --arm base: --arm fused:AUGUR_GDN_INPUT_KERNEL=fused --order base,fused,fused,base

With --endpoint the harness measures a product that is already serving instead (no A/B arms):

  python scripts/augur_bench.py --endpoint http://127.0.0.1:8294 --api-key-file KEY \\
      --evidence RUN/launch --suite bench/scenarios.json --repeats 2 --out runs/scenarios
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from audit_augur_sampling_trajectory import audit  # noqa: E402

SUITE = ROOT / 'bench/english-thinking-14.json'
MANIFEST = ROOT / 'model-cards/qwen3.8-27b-quasar-nvfp4/runtime-manifest.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_arm(text):
    name, _, rest = text.partition(':')
    if not name or not name.replace('_', '').replace('-', '').isalnum():
        raise argparse.ArgumentTypeError('arm name must be alphanumeric: ' + text)
    env = {}
    for item in filter(None, rest.split(',')):
        key, eq, value = item.partition('=')
        if not eq or not key.startswith('AUGUR_'):
            raise argparse.ArgumentTypeError('arm overrides must be AUGUR_NAME=value: ' + item)
        env[key] = value
    return name, env


def http(url, key, payload=None, timeout=30):
    request = urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode(),
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


class Owner:
    """One product owner for one arm, stopped cooperatively through its stdin pipe."""

    def __init__(self, args, label, env, key):
        self.evidence = args.out / label / 'launch'
        self.evidence.parent.mkdir(parents=True)
        child = dict(os.environ, AUGUR_BENCH_OVERRIDES=json.dumps(env), AUGUR_QWEN35_API_KEY=key,
                     AUGUR_BENCH_RUNTIME=str(args.runtime or ''))
        command = [sys.executable, '-I', '-X', 'utf8', '-B', str(HERE / 'augur_bench.py'), '--serve',
                   '--build', str(args.build), '--model-dir', str(args.model_dir),
                   '--manifest', str(MANIFEST), '--manifest-sha256', sha(MANIFEST),
                   '--draft-environment', str(args.draft_environment), '--evidence', str(self.evidence),
                   '--context-tokens', str(args.context_tokens), '--port', str(args.port), '--stop-on-stdin']
        self.log = (args.out / label / 'owner.log').open('xb')
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=self.log, stderr=subprocess.STDOUT, env=child)
        deadline = time.monotonic() + args.startup_timeout
        while not (self.evidence / 'ready.json').exists():
            if self.process.poll() is not None:
                raise RuntimeError(f'owner for arm {label} exited during startup; see owner.log')
            if time.monotonic() > deadline:
                self.stop()
                raise TimeoutError(f'owner for arm {label} did not become ready')
            time.sleep(0.25)

    def stop(self):
        if self.process.poll() is None:
            self.process.stdin.write(b'stop\n')
            self.process.stdin.flush()
            self.process.wait(timeout=900)
        self.log.close()
        stopped = json.loads((self.evidence / 'stopped.json').read_bytes())
        if stopped.get('cleanup_verified') is not True:
            raise RuntimeError('owner cleanup was not verified; do not start another owner')

    def receipt(self, response_id):
        metrics = [json.loads(line) for line in (self.evidence / 'api-metrics.jsonl').read_text(encoding='utf-8').splitlines()]
        metric = next(m for m in reversed(metrics) if m.get('request_id') == response_id)
        path = next((self.evidence / 'native').glob('owner-*/native/turns/' + metric['native_request_id'] + '.json'))
        return json.loads(path.read_bytes()), metric


class Live:
    """A product owner this harness did not start: read its receipts, never stop it."""

    def __init__(self, evidence):
        self.evidence = evidence

    receipt = Owner.receipt

    def stop(self):
        pass


def serving_prefix_scope(evidence):
    """The serving product's own report on its prefix store. Absent or unreadable reads as off."""
    try:
        frames = sorted((Path(evidence) / 'native').glob('owner-*/native/ready.json'))
        frame = json.loads(frames[0].read_bytes())
    except (OSError, ValueError, IndexError):
        return dict(persistent=False)
    return dict(persistent=bool(frame.get('input_prefix_persistent_index')),
                scope=frame.get('input_prefix_scope'), quota=frame.get('input_prefix_quota'))


def run_case(url, key, request, case, seed=None):
    payload = dict(request, messages=case['messages'], max_completion_tokens=case['max_completion_tokens'])
    if seed is not None:
        payload['seed'] = seed
    started = time.perf_counter()
    response = http(url + '/v1/chat/completions', key, payload, timeout=3600)
    wall = time.perf_counter() - started
    return response, wall


def result_row(arm, case, response, wall, receipt, metric):
    trace = audit(receipt)
    if not trace['native_selection_trace_verified']:
        raise RuntimeError('native selection trace was not verified')
    dflash, sampling = receipt['decode_detail']['dflash'], receipt['decode_detail']['sampling']
    seconds = receipt['whole_wall_s_before_receipt'] - receipt['prefill_wall_s']
    tokens = len(receipt['output_token_ids'])
    cached = response['usage']['prompt_tokens_details']['cached_tokens']
    prefill = receipt['prefill_wall_s']
    return dict(arm=arm, case=case, prompt_tokens=trace['prompt_tokens'], cached_prompt_tokens=cached,
        prefill_s=prefill, prefill_tps=(trace['prompt_tokens'] - cached) / prefill if prefill > 0 else None,
        ttft_s=metric.get('native_ttft_s'), output_tokens=tokens, rounds=dflash['rounds'],
        native_decode_s=seconds, native_tps=tokens / seconds, http_tps=tokens / wall,
        tokens_per_round=tokens / dflash['rounds'], accepted_over_proposed=dflash['accepted_over_proposed'],
        finish=response['choices'][0]['finish_reason'], verification=sampling.get('verification'),
        proposal_calibration=sampling.get('proposal_calibration'),
        kernels=dict(shared_head=dflash.get('shared_head_kernel'), draft_block=dflash.get('draft_block_projection_kernel'),
                     gdn_input=dflash.get('gdn_input_kernel')),
        verify_gpu_component_bill=dflash.get('verify_gpu_component_bill'),
        trajectory_sha256=trace['selection_trajectory_sha256'], output_sha256=trace['output_token_sha256'])


def pool(rows):
    tokens = sum(r['output_tokens'] for r in rows)
    seconds = sum(r['native_decode_s'] for r in rows)
    rounds = sum(r['rounds'] for r in rows)
    return dict(native_tps=tokens / seconds, tokens_per_round=tokens / rounds,
                ms_per_round=1e3 * seconds / rounds, output_tokens=tokens, rounds=rounds)


def check_cold_prefix(rows, required):
    """A warm prefill makes TTFT and HTTP TPS incomparable across runs while leaving native decode TPS
    untouched, so a measured run either states it was cold or refuses to stand behind its numbers.
    Only the first pass is held to that: a second pass over the same prompts is meant to hit the store,
    and its native decode TPS remains comparable."""
    cold = [r for r in rows if r.get('pass', 1) == 1]
    warm = [r['case'] for r in cold if r.get('contaminated')]
    reused = [r['case'] for r in rows if r.get('pass', 1) > 1 and r.get('contaminated')]
    if warm and required:
        raise SystemExit(f'{len(warm)} case(s) were served from a persisted prefix on their first request '
                         f'({", ".join(warm)}); restart the product with a fresh --prefix-cache-scope and rerun')
    return dict(cold_prefill=not warm, contaminated_cases=warm, warm_repeat_cases=reused)


def summarize(rows, order):
    """Pooled per-arm counters, each arm against the first, per-case trajectory identity."""
    arms = list(dict.fromkeys(order))
    cases = list(dict.fromkeys(r['case'] for r in rows))
    pick = lambda arm, case=None: [r for r in rows if r['arm'] == arm and case in (None, r['case'])]
    pooled = {a: pool(pick(a)) for a in arms}
    base, comparisons = arms[0], {}
    for arm in arms[1:]:
        per_case = {}
        for case in cases:
            hashes = {r['trajectory_sha256'] for r in pick(base, case) + pick(arm, case)}
            gain = pool(pick(arm, case))['native_tps'] / pool(pick(base, case))['native_tps'] - 1
            per_case[case] = dict(gain_pct=100 * gain, identical_trajectory=len(hashes) == 1)
        gains = [v['gain_pct'] / 100 for v in per_case.values()]
        comparisons[arm] = dict(vs=base,
            gain_pct=100 * (pooled[arm]['native_tps'] / pooled[base]['native_tps'] - 1),
            ms_saved_per_round=pooled[base]['ms_per_round'] - pooled[arm]['ms_per_round'],
            geomean_case_gain_pct=100 * (math.exp(sum(map(math.log1p, gains)) / len(gains)) - 1),
            min_case_gain_pct=100 * min(gains), max_case_gain_pct=100 * max(gains),
            identical_trajectory_cases=sum(v['identical_trajectory'] for v in per_case.values()),
            per_case=per_case)
    spread = {}
    for arm in arms:
        values = [100 * (max(t) / min(t) - 1) for case in cases
                  if len(t := [r['native_tps'] for r in pick(arm, case)]) > 1]
        spread[arm] = max(values) if values else None
    return dict(arms=pooled, comparisons=comparisons, max_repeat_spread_pct=spread, cases=len(cases))


def report(summary):
    lines = [f"{arm:12s} {p['native_tps']:7.2f} TPS  {p['ms_per_round']:6.2f} ms/round  {p['tokens_per_round']:.3f} tokens/round"
             for arm, p in summary['arms'].items()]
    for arm, c in summary['comparisons'].items():
        lines.append(f"{arm} vs {c['vs']}: {c['gain_pct']:+.2f}% pooled, geomean {c['geomean_case_gain_pct']:+.2f}%, "
                     f"range {c['min_case_gain_pct']:+.1f}..{c['max_case_gain_pct']:+.1f}%, "
                     f"identical trajectories {c['identical_trajectory_cases']}/{summary['cases']}")
    return '\n'.join(lines)


def serve(argv):
    """Child entry: the Windows preset first (it clears AUGUR_*), then the arm's overrides."""
    overrides = json.loads(os.environ.pop('AUGUR_BENCH_OVERRIDES', '{}'))
    key = os.environ.get('AUGUR_QWEN35_API_KEY')
    runtime = os.environ.pop('AUGUR_BENCH_RUNTIME', '')
    if runtime:
        import augur_portable
        augur_portable.activate(runtime)
    import run_augur_windows
    run_augur_windows.configure()
    os.environ.update(overrides)
    if key:
        os.environ['AUGUR_QWEN35_API_KEY'] = key
    import augur_product
    return augur_product.main(argv)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ['--serve']:
        return serve(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--build', type=Path)
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--draft-environment', type=Path)
    parser.add_argument('--out', type=Path, required=True, help='new directory for this run')
    parser.add_argument('--arm', type=parse_arm, action='append', default=[],
                        help='NAME:AUGUR_X=v,AUGUR_Y=w; NAME: alone runs the preset')
    parser.add_argument('--order', help='arm names, e.g. base,new,new,base')
    parser.add_argument('--endpoint', help='measure an already running product (e.g. http://127.0.0.1:8294) '
                        'instead of starting owners; needs --api-key-file and --evidence (its run launch directory)')
    parser.add_argument('--api-key-file', type=Path)
    parser.add_argument('--evidence', type=Path, help='launch directory of the running product')
    parser.add_argument('--repeats', type=int, default=1, help='endpoint mode: requests per case')
    parser.add_argument('--seeds', help='comma-separated seeds; each case is sent once per seed, back to back '
                        '(the first request of a case is the cold prefill, later ones may hit the prefix cache)')
    parser.add_argument('--require-cold-prefix', action=argparse.BooleanOptionalAction, default=True,
                        help='fail the run when a case was served from a persisted prefix (default: on). '
                             'A warm prefill inflates TTFT and HTTP TPS; native decode TPS is unaffected')
    parser.add_argument('--runtime', type=Path, default=ROOT.parent if (ROOT.parent / 'runtime/native').is_dir() else None,
                        help='Windows package root whose bundled native runtime to load (default: this package)')
    parser.add_argument('--suite', type=Path, default=SUITE)
    parser.add_argument('--cases', help='comma-separated subset of suite case names')
    parser.add_argument('--max-tokens', type=int, help='override every case budget (smoke runs)')
    parser.add_argument('--port', type=int, default=8304)
    parser.add_argument('--context-tokens', type=int, default=262144)
    parser.add_argument('--startup-timeout', type=float, default=600)
    args = parser.parse_args(argv)
    if args.endpoint:
        if not (args.api_key_file and args.evidence) or args.arm or args.order:
            parser.error('--endpoint needs --api-key-file and --evidence, and takes no arms')
        args.arm, args.order = [('live', {})], ','.join(['live'] * args.repeats)
    elif not (args.build and args.model_dir and args.draft_environment and args.arm and args.order):
        parser.error('owner mode needs --build, --model-dir, --draft-environment, --arm and --order')
    arms = dict(args.arm)
    order = args.order.split(',')
    if len(arms) != len(args.arm) or any(a not in arms for a in order):
        parser.error('--order must name declared arms, and arm names must be unique')
    suite = json.loads(args.suite.read_bytes())
    if suite.get('schema') != 'augur-bench-suite-v1':
        parser.error('unsupported suite schema')
    cases = suite['cases']
    if args.cases:
        wanted = args.cases.split(',')
        cases = [c for c in cases if c['name'] in wanted]
        if len(cases) != len(wanted):
            parser.error('unknown case in --cases')
    if args.max_tokens:
        cases = [dict(c, max_completion_tokens=args.max_tokens) for c in cases]
    seeds = [int(s) for s in args.seeds.split(',')] if args.seeds else [None]
    args.out.mkdir(parents=True, exist_ok=False)
    if args.endpoint:
        key, url = args.api_key_file.read_text(encoding='ascii').strip(), args.endpoint.rstrip('/')
        build_id = json.loads((args.evidence / 'identity.json').read_bytes()).get('build_receipt_sha256')
        prefix = serving_prefix_scope(args.evidence)
        if prefix['persistent'] and args.require_cold_prefix:
            print('WARNING: the serving product persists an input-prefix store; restart it with a fresh '
                  '--prefix-cache-scope for a cold run. Cases that hit will be marked contaminated.', file=sys.stderr, flush=True)
    else:
        for name in ('build', 'model_dir', 'draft_environment'):
            setattr(args, name, getattr(args, name).resolve())
        key = secrets.token_urlsafe(32)  # per-run loopback key; never written to disk
        url = f'http://127.0.0.1:{args.port}'
        build_id = sha(args.build / 'build-receipt.json')
    registration = dict(schema='augur-bench-run-v1', suite=suite['name'], suite_sha256=sha(args.suite),
        cases=[c['name'] for c in cases], arms=arms, order=order, endpoint=bool(args.endpoint),
        build_receipt_sha256=build_id, bench_sha256=sha(__file__))
    (args.out / 'registration.json').write_text(json.dumps(registration, indent=1) + '\n', encoding='utf-8')
    rows = []
    for index, arm in enumerate(order):
        label = f'{index + 1}-{arm}'
        owner = Live(args.evidence) if args.endpoint else Owner(args, label, arms[arm], key)
        try:
            for case, seed in ((c, s) for c in cases for s in seeds):
                response, wall = run_case(url, key, suite['request'], case, seed)
                row = result_row(arm, case['name'], response, wall, *owner.receipt(response['id']))
                row['seed'] = suite['request'].get('seed') if seed is None else seed
                row['pass'] = index + 1
                if row['cached_prompt_tokens']:
                    row['contaminated'] = True
                    print(f"  !! {case['name']} reused {row['cached_prompt_tokens']} cached prompt tokens; "
                          f"prefill and HTTP TPS for this row are not cold", file=sys.stderr, flush=True)
                rows.append(row)
                with (args.out / 'results.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(row) + '\n')
                print(f"{label:12s} {case['name']:28s} {row['prompt_tokens']:7d} prompt  {row['native_tps']:7.2f} TPS  "
                      f"{row['tokens_per_round']:.3f} tokens/round", flush=True)
        finally:
            owner.stop()
    summary = check_cold_prefix(rows, args.require_cold_prefix)
    summary = dict(summarize(rows, order), **summary)
    (args.out / 'summary.json').write_text(json.dumps(dict(registration=registration, **summary), indent=1) + '\n',
                                           encoding='utf-8')
    print(report(summary))


if __name__ == '__main__':
    main()
