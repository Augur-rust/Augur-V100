"""Selected Windows Augur fixed-B8 preset. Model inputs remain explicit."""
import argparse
import hashlib
import os
from pathlib import Path
import sys

SETTINGS = {
    "AUGUR_HTTP_DFLASH": "batched8",
    "AUGUR_HTTP_DFLASH_PLAN": "park",
    "AUGUR_DFLASH_GATE": "fixed_b8",
    "AUGUR_DFLASH_TOPK": "batch8",
    "AUGUR_DFLASH_TOPK_TILED": "1",
    "AUGUR_SAMPLING_TOPK": "tiled",
    "AUGUR_DFLASH_SELECTOR_PATH": "greedy",
    "AUGUR_DFLASH_SELECTOR_SCALE": "1",
    "AUGUR_DFLASH_PINNED_D2H": "0",
    "AUGUR_DFLASH_GPU_BILL": "0",
    "AUGUR_DFLASH_VERIFY_BILL": "0",
    "AUGUR_B1_LINEAR_GRAPH": "1",
    "AUGUR_B1_FULL_GRAPH": "1",
    "AUGUR_B1_HEAD_GRAPH": "1",
    "AUGUR_HTTP_DECODE_ATTENTION": "gqa8_planar",
    "AUGUR_CACHE_RESTORE_WORKERS": "8",
    "AUGUR_CACHE_RESTORE_SCHEDULE": "pipeline",
    "AUGUR_SHARD_WORKERS": "5",
    "AUGUR_BF16_STARTUP": "streamed",
    "AUGUR_QPN_STARTUP": "direct",
    "AUGUR_HTTP_FINISH": "receipt_ref_v3",
    "AUGUR_HTTP_DECODE": "bytelevel_v2",
    # Lossless sparse speculative sampling: all seven drafts carry their actual proposal q and are
    # verified by p/q rejection with a positive residual. Target sampling stays T=1/top_p=.95/top_k=20.
    "AUGUR_DFLASH_PROBABILITY_PREFIX": "7",
    # Per-position proposal temperature and pair scale (proposal-only; measured 14-task English
    # thinking: +20.1% native TPS vs greedy proposals, 3.64 tokens/round). Row 1 pair scale must be 1.
    "AUGUR_DFLASH_PROBABILITY_TEMPERATURES": "0.875,0.875,0.875,0.75,0.75,0.75,0.75",
    "AUGUR_DFLASH_PROBABILITY_PAIR_SCALES": "1,1.5,1.5,1.5,1.5,1.5,1.5",
    # Block verification (Sun et al. 2025) is available as an explicit opt-in only.
    "AUGUR_DFLASH_VERIFICATION": "token",
    # Shared BF16 vocabulary head (draft + B8 verifier): register-blocked FP32 CUDA-core kernel,
    # 16-byte BF16 vectors, exact FP32 products. 8.12 -> 3.23 ms per B8 head; "llama" restores MMVF<8>.
    "AUGUR_SHARED_HEAD_KERNEL": "rw4",
    # Draft B8 Q8_0 block projections: Augur direct kernel reading the original 34-byte blocks in
    # place with FP32 activations (no Q8_1 staging). Draft layers 12.85 -> 7.39 ms; "llama" restores MMVQ.
    "AUGUR_DFLASH_Q8_KERNEL": "direct",
    # Direct-B8 GDN glue: [residual add] -> rmsnorm -> FP16 stage -> grouped QPN projections in two
    # launches, trimmed alpha/beta and replay records stored in place. Bit-identical to the separate
    # sequence; 50.31 -> 46.96 ms per B8 round (+7.1%). "separate" restores the launch-per-op path.
    "AUGUR_GDN_INPUT_KERNEL": "fused"
}

def gate_identity(path):
    from augur_reasoning import ENGLISH_GATE_SHA256
    path = Path(path).resolve(strict=True)
    if not path.is_file() or path.stat().st_size != 256 * 5120 * 4:
        raise ValueError('selector gate must contain 256 x 5120 FP32 weights')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != ENGLISH_GATE_SHA256:
        raise ValueError('selector gate SHA256 does not match the selected gate0.45 baseline')
    return dict(path=str(path), sha256=digest)


def configure(selector_gate_file=None, vision_residency="on_demand", sampling_seed=0):
    if type(sampling_seed) is not int or not 0 <= sampling_seed <= (1 << 64) - 1:
        raise ValueError("sampling seed must be an unsigned 64-bit integer")
    if vision_residency not in ("auto", "on_demand"):
        raise ValueError("vision residency must be auto or on_demand")
    from augur_reasoning import GLOBAL_GATE_POLICY
    gate = gate_identity(selector_gate_file) if selector_gate_file is not None else None
    # Select exactly the measured preset; stale experiment flags must not leak in.
    for key in list(os.environ):
        if key.startswith('AUGUR_'):
            del os.environ[key]
    os.environ.update(SETTINGS)
    os.environ["AUGUR_SAMPLING_SEED"] = str(sampling_seed)
    os.environ["AUGUR_VISION_RESIDENCY"] = vision_residency
    os.environ.update(AUGUR_DFLASH_ENGLISH_GATE=GLOBAL_GATE_POLICY,
        AUGUR_DFLASH_SELECTOR_GATE_ACCUMULATION='fp32',
        AUGUR_DFLASH_SELECTOR_GATE_SOURCE='external_file' if gate else 'embedded_gguf')
    if gate:
        os.environ.update(AUGUR_DFLASH_SELECTOR_GATE_F32=gate['path'],
            AUGUR_DFLASH_SELECTOR_GATE_F32_SHA256=gate['sha256'])
    if 'CUDA_PATH' in os.environ:
        os.environ['PATH'] = str(Path(os.environ['CUDA_PATH'])/'bin')+os.pathsep+os.environ.get('PATH', '')

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    help_requested = '--help' in argv or '-h' in argv
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--api-key-file', type=Path)
    parser.add_argument('--selector-gate-file', type=Path,
        help='optional legacy SHA256-pinned gate; default is the Rust-quantized embedded FP32 gate0.45')
    parser.add_argument('--sampling-seed', type=int, default=0,
        help='immutable native target/proposal/verification seed for this service (default 0)')
    parser.add_argument('--live-file', type=Path,
        help='optional private telemetry output consumed by augur_monitor.py')
    parser.add_argument("--vision-residency", choices=("auto", "on_demand"), default="on_demand",
        help="default on_demand releases after encoding; optional auto retains weights/plans until prefill memory pressure")
    options, remaining = parser.parse_known_args(argv)
    if help_requested:
        parser.print_help()
        import augur_product
        return augur_product.main(remaining)
    configure(options.selector_gate_file, options.vision_residency, options.sampling_seed)
    if options.api_key_file:
        key = options.api_key_file.read_text(encoding='ascii').strip()
        if len(key) < 32 or any(c.isspace() for c in key):
            raise ValueError('local API key must contain at least 32 non-whitespace ASCII characters')
        os.environ['AUGUR_QWEN35_API_KEY'] = key
    import augur_product
    tracker = None
    try:
        if options.live_file:
            from augur_live_telemetry import install
            tracker = install(options.live_file.resolve())
        return augur_product.main(remaining)
    finally:
        if tracker is not None:
            tracker.closed.set()

if __name__ == '__main__':
    main()
