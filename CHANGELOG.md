# Changelog

## v0.0.1-alpha — 2026-10-17

- Cooperative stop shared by the launcher, monitor, chat and benchmark client; cancellation waits for the active native generation to retire before the next request.
- Explicit persistent prefix-cache scope, with a public cold/warm long-context ledger from 8K through 256K.
- Simplified Chinese and English UI, including a saved first-run language preference.
- One per-user API key, automatic local chat sign-in, optional user environment export, agent connection settings and key rotation.
- Double-click `启动服务.exe`, automatic startup from saved settings, and a Quit action that retires the engine and launcher.
- Source and Windows delivery retain independent file manifests and validation receipts. Existing performance measurements remain tied to their original builds; publication packaging does not create new GPU performance results.

### Engine and Windows features prepared on 2026-10-16

### Decoding

- Exact speculative sampling on fixed B8: all 7 draft rows propose from a 16-candidate distribution `q`; the target accepts with `min(1, p/q)` and corrects from the positive residual. Target sampling: temperature 1, top-p 0.95, top-k 20, seed 0.
- Per-row proposal temperature and gate pair-term scale (`AUGUR_DFLASH_PROBABILITY_TEMPERATURES`, `AUGUR_DFLASH_PROBABILITY_PAIR_SCALES`). Defaults: T = 0.875×3, 0.75×4; pair scale = 1, 1.5×6. A refit for the direct Q8_0 draft found no setting worth more than 0.4%. [Details](docs/AUGUR_ENGLISH_PROPOSAL_CALIBRATION_20261016.md).
- Opt-in block verification (`AUGUR_DFLASH_VERIFICATION=block`, Sun et al. 2025) on the sparse support.
- Shared BF16 vocabulary head (draft and verifier) on a register-blocked FP32 CUDA-core kernel: 8.12 → 3.23 ms per head. `AUGUR_SHARED_HEAD_KERNEL=llama` selects MMVF<8>. [Details](docs/AUGUR_SHARED_HEAD_RW4_20261016.md).
- Draft Q8_0 projections read the original 34-byte blocks in place with FP32 activations: draft layers 12.9 → 7.5 ms per round. `AUGUR_DFLASH_Q8_KERNEL=llama` selects MMVQ<8>. [Details](docs/AUGUR_DRAFT_Q8_B8_DIRECT_20261016.md).
- Verifier GDN glue fused into grouped [residual add] → RMSNorm → FP16 stage → projection programs, bit-identical to the separate launches: 50.3 → 47.0 ms per round. `AUGUR_GDN_INPUT_KERNEL=separate` selects the separate launches. [Details](docs/AUGUR_GDN_GLUE_B8_20261016.md).
- 14 long English-thinking segments: 46.21 → 75.92 native TPS against a deterministic-draft baseline.
- Health, API metrics and receipts report one effective sampling contract, including the speculative verification actually in use.

### Tools

- `scripts/augur_bench.py` with the public suite `bench/english-thinking-14.json`: interleaved A/B arms, pooled counters, trajectory-identity checks, cleanup verification per arm.

### Windows package

- Browser setup: target and draft paths, Rust draft quantization, KV capacity from 8K to 256K, restart progress, chat and monitor.
- Self-contained Windows x64/SM70 package with isolated Python, pinned dependencies and CUDA/MSVC runtime files. Model weights and the NVIDIA driver are supplied separately.
- Q8_0 draft with an embedded FP32 selector gate (0.45), fixed B8 for text, and on-demand Vision.
- Cancellation reaches the native owner; client requests wait in a cancellable queue while the previous generation drains.
- Source and Windows archives with file identities and third-party notices.

### Validation scope

GPU measurements cover one Windows SM70 host and the identified binaries. Clean-machine qualification is a separate step. See [release preparation](docs/RELEASING.md) and [security boundaries](SECURITY.md).
