# Augur repository layout

The project keeps its Rust workspace. Model cards and the documentation index use a simple per-model/reference layout; they do not introduce a generic model loader or copy another engine's artifact format.

```text
Augur/
  README.md / README.zh-CN.md      overview, supported target, measured results
  BUILDING.md / CONTRIBUTING.md    build and contributor entry points
  SECURITY.md / NOTICE             trust boundaries and attribution
  Cargo.toml / Cargo.lock          Rust workspace and pinned dependencies
  crates/                         Rust components and native CUDA sources
  tools/augur-quantize/           standalone Rust draft quantizer
  model-cards/
    README.md                     supported-model index (one target)
    qwen3.8-27b-quasar-nvfp4/
      README.md / README.zh-CN.md  model integration cards
      artifact-manifest.json      recorded identities, not runtime admission
  scripts/                        build, launch, monitor, packaging and Python tests
  bench/                          public benchmark prompt suites for scripts/augur_bench.py
  integrations/dsh/                client profile and integration patch
  docs/
    README.md                     documentation index
    public/                       public provenance and measurement summaries
```

## Implementation boundaries

| Location | Role |
|---|---|
| `crates/augur-cuda-sys/native/` | CUDA/C++ operators, runtime shim and attributed vendor sources |
| `crates/augur-cuda-sys/src/` | Rust FFI declarations and ABI types |
| `crates/augur-backend/` | Device owners, streams, events and CUDA wrappers |
| `crates/augur-model-qwen35-runtime/` | Target/draft/Vision execution, sessions, KV/GDN state and service examples |
| `crates/augur-model-qwen35/`, `augur-core`, `augur-formats` | Model metadata and shared types/formats |
| Other workspace crates | Graph/compiler/execution/reference/repack support; not additional targets |
| `scripts/augur_product.py`, `augur_api.py` | Product control, request validation and HTTP translation |
| `scripts/run_augur_windows.py`, `augur_portable.py` | Launch and isolated portable-runtime bootstrap |
| `scripts/augur_monitor.py`, `augur_live_telemetry.py` | Read-only monitoring and telemetry; no extra model owner |

Crate and file names use the `augur` prefix; `qwen35` and `quasar` name the model family and its checkpoint, not the engine. Support is defined by [the model-card index](../model-cards/README.md), not by every historical crate or fixture.

## Tests and release inputs

Rust tests stay alongside their crates. Python product/release tests live in `scripts/test_*.py`; there is no duplicate top-level test harness. [CONTRIBUTING](../CONTRIBUTING.md) explains CPU and GPU evidence boundaries.

Public-source selection is explicit in [prepare_augur_release.py](../scripts/prepare_augur_release.py). A development checkout can contain `evidence/`, `target/`, old experiments and external build outputs; these are not automatically published. Model-card selection is an exact file allowlist, not a recursive copy of a weight directory.

After source or naming changes, freeze a new identity. Build receipts, performance summaries and old archives must not be silently rewritten to match new names. The Windows build entry is [build_augur_windows.py](../scripts/build_augur_windows.py); see [BUILDING](../BUILDING.md).
