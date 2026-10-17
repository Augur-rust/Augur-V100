# Contributing to Augur

## Scope

The current supported target is [qwen3.8-27b-quasar-nvfp4](model-cards/qwen3.8-27b-quasar-nvfp4/README.md), with the paired DFlash accelerator. A crate name, shared architecture string or experiment does not imply support for other weights. A new target needs explicit implementation, a model card, authenticated identities and separate qualification.

Start with [BUILDING](BUILDING.md), [repository layout](docs/REPOSITORY_LAYOUT.md) and [documentation](docs/README.md). Preserve the Rust workspace; avoid unrelated moves, broad formatting and abstractions for hypothetical models.

## Changes and checks

1. Update affected callers, tests, model metadata and documentation together.
2. Test Python changes with relevant `scripts/test_*.py` suites. The frozen public suite must work without a private development directory; retain explicit skip reasons.
3. For Rust/CUDA/ABI or feature changes, build the selected Windows feature and run CPU tests from the same frozen source. Ignored GPU tests are not executed CPU evidence.
4. For execution or mathematical changes, obtain matching GPU functional/numerical and timing evidence. Record model/build hashes, hardware, prompt/output counts, cache state, Vision/DFlash modes and acceptance denominators.
5. Keep native prefill TPS, target-only decode, DFlash-assisted generation, HTTP TTFT and end-to-end throughput separate. Report regressions and failed/excluded cases, not only the fastest result.
6. Model-card measurements must reference public counters and a bound evaluation identity. Do not copy upstream or other-engine benchmark scores into this engine's results.

The selected launch policy is fixed B8 with Q8_0 + embedded FP32 gate0.45; Vision is on-demand by default. Changing defaults requires coordinated tests and documentation.

## Unsafe, ownership and attribution

Read [unsafe/CUDA boundaries](docs/UNSAFE_TRUST_BOUNDARIES.md). New unsafe code needs concrete owner, range, aliasing, completion and failure-path reasoning. Comments do not repair unenforced memory-safety preconditions in safe APIs. Preserve vendor license/NOTICE files and attribution.

## Release hygiene

- Keep weights, caches, keys, browser capability URLs, conversations and private runs outside public source.
- Public additions belong in [prepare_augur_release.py](scripts/prepare_augur_release.py), with selection/link checks. Model cards use an exact file allowlist, not broad JSON or binary inclusion.
- Freeze a new source identity after changes; preserve historical archives and measurement receipts.
- Card manifests are documentation only; loading still requires an authenticated admission manifest and matching artifacts.
- A local candidate is not a public release. Publishing, signing, clean-machine checks and redistribution permissions are separate steps.

For security concerns, follow [SECURITY](SECURITY.md); do not include secrets or private prompts in public reports.
