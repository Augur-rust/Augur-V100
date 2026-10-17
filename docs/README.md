# Augur documentation / 文档导航

Start with the [project README](../README.md) / [中文首页](../README.zh-CN.md). The current supported target is **qwen3.8-27b-quasar-nvfp4** only.

## Build and use

| Guide | Purpose |
|---|---|
| [Windows one-click setup](AUGUR_WINDOWS_USER_GUIDE.zh-CN.md) | Model paths, quantization, monitor and cache disk budgeting |
| [Build entry](../BUILDING.md) | Freeze public source, build the Windows GPU owner and run CPU checks |
| [Windows guide](AUGUR_WINDOWS_SOURCE_GUIDE.md) | Start, choose Vision residency, monitor and package |
| [DSH integration](../integrations/dsh/README.md) | Client configuration, thinking controls and compaction |
| [Rust quantizer](../tools/augur-quantize/README.md) | Convert the paired draft to Q8_0 with embedded FP32 gate0.45 |

## Models and measurements

| Reference | Purpose |
|---|---|
| [Supported models](../model-cards/README.md) | One directory per supported target; cards and recorded identities |
| [Model provenance](public/AUGUR_MODEL_SOURCES.md) | Upstream sources, target/draft hashes, distribution checks |
| [Performance](../README.md#performance) | Current decode results and how each step was measured |
| [Benchmark harness](../scripts/augur_bench.py) | Interleaved A/B arms over [the public prompt suite](../bench/english-thinking-14.json) |
| [Probability drafts, 2026-10-16](AUGUR_ENGLISH_PROPOSAL_CALIBRATION_20261016.md) | Sampled English thinking: calibration, verification, 14-task results |
| [Shared head kernel, 2026-10-16](AUGUR_SHARED_HEAD_RW4_20261016.md) | Register-blocked FP32 BF16 head, tensor-core comparison, ABBA |
| [Draft Q8_0 direct kernel, 2026-10-16](AUGUR_DRAFT_Q8_B8_DIRECT_20261016.md) | In-place Q8_0 B8 draft projections with FP32 activations, ABBA |
| [Fused GDN glue, 2026-10-16](AUGUR_GDN_GLUE_B8_20261016.md) | Bit-identical norm → stage → grouped projection programs for the B8 verifier, ABBA |
| [Gate0.45 evaluation](AUGUR_EMBEDDED_GATE045_ENGLISH_ACCEPTANCE_20261015.md) | Method, outputs and counter definitions |
| [Public measurement data](public/AUGUR_ENGLISH_THINKING_GATE045_20261015.json) | Counters and bound evaluation identity |
| [Vision memory checks](AUGUR_VISION_RESIDENCY_AND_LONG_PREFILL_20261015.md) | Residency and actual 8K/32K/64K inputs |
| [Source release and prefill ledger](AUGUR_SOURCE_RELEASE_README.md) | Delivery scope and historical prefill peak |

## Contribute and release

- [Repository layout](REPOSITORY_LAYOUT.md): Rust workspace, CUDA boundary, product entry points and tests.
- [Release preparation](RELEASING.md) and [changelog](../CHANGELOG.md): source selection, artifact verification and publication checklist.
- [Contributing](../CONTRIBUTING.md): change scope and evidence requirements.
- [Security](../SECURITY.md) and [unsafe trust boundaries](UNSAFE_TRUST_BOUNDARIES.md).
- [Runtime dependency provenance](public/RUNTIME_DEPENDENCIES.md), [engine notices](../NOTICE).

This index points to current public entry points. A research file elsewhere in a development checkout is not a supported-model entry, a deployment instruction, or a new performance baseline. Private runs, weights and caches are not shipped as documentation.
