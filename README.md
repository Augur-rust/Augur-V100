# Augur

**v0.0.1-alpha** · First public alpha release.

[简体中文](README.zh-CN.md)

Augur is a Rust/CUDA inference engine that runs a 27B NVFP4 reasoning model on a single Volta GPU (SM70, 32 GiB). It samples exactly from the target model's distribution and uses a small draft model to produce several tokens per target pass.

**75.9 tokens/s** native decode on 14 long English-thinking segments, sampled at temperature 1, top-p 0.95, top-k 20. That is 1.64× a conventional deterministic-draft baseline on the same GPU, with the output distribution unchanged.

You can see the project demos here: https://x.com/Quaereee

## Performance

Hardware: one Tesla V100 32 GB (SM70), Windows, TCC driver. Target: [QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4), original weights and BF16 output head. Draft: DFlash2, converted by Augur to Q8_0 with an FP32 selector gate (see [below](#a-draft-converted-for-this-engine)). Native TPS = output tokens / native decode seconds, prefill excluded.

**Sampling is never relaxed for a benchmark.** Every number on this page is measured under the product contract (temperature 1, top-p 0.95, top-k 20) with exact step-by-step distribution sampling: each emitted token is a sample from the target model's own distribution.

### Long context, 8K to 256K

Two passes over every prompt. Pass 1 is cold — nothing from it is in any cache. Pass 2 is sent back-to-back to show what the persisted prefix store does to time-to-first-token; the decode rate it reports is measured the same way and is unaffected by the reuse. Output is 1,024 tokens per case with thinking off.

| case | prompt tokens | cold prefill | cold TTFT | native decode | warm re-run native decode |
|---|---:|---:|---:|---:|---:|
| code_8k | 7,353 | 984.2 tok/s | 7.47 s | 83.19 tok/s | 83.28 tok/s |
| code_32k | 31,506 | 934.0 tok/s | 33.74 s | 91.63 tok/s | 91.83 tok/s |
| code_186k | 186,247 | 533.1 tok/s | 349.43 s | 59.38 tok/s | 59.41 tok/s |
| code_256k | 260,230 | 437.8 tok/s | 594.47 s | 55.81 tok/s | 55.99 tok/s |
| en-doc_8k | 7,630 | 929.4 tok/s | 8.21 s | 64.66 tok/s | 64.65 tok/s |
| en-doc_32k | 31,834 | 914.0 tok/s | 34.84 s | 54.08 tok/s | 54.16 tok/s |
| en-doc_193k | 192,954 | 520.5 tok/s | 370.72 s | 45.39 tok/s | 45.45 tok/s |
| en-doc_256k | 260,214 | 455.2 tok/s | 571.66 s | 34.98 tok/s | 34.95 tok/s |

Both contexts fit on the one card. A warm re-run returns 256K in 3.70 s of prefill and 186K in 2.67 s, against 594 s and 349 s cold; the store keeps whole prefixes up to its quota, so the second pass pays only the first token. The one exception is `en-doc_256k`, whose re-run missed the store after the quota evicted it and paid the full cold prefill again — its native decode rate is unaffected, because that rate excludes prefill. Decode rate is repeat-stable to 0.33% across the whole run, which is why the cold and warm columns agree: the prefix store changes when the first token arrives, not how fast the rest of them come.

The round itself gets more expensive with length: 49.8 ms at 8K, 53.2 ms at 32K, 75.0 ms at 186K and 85.3 ms at 256K on the code cases, with the same shape on the English document. Only the 16 full-attention layers grow with context; the 48 Gated DeltaNet layers stay fixed, and the difference is what the longer round costs. Acceptance moves on its own axis: the draft lands 4.15 tokens per round on code at 8K, 4.88 at 32K and 4.76 at 256K, and between 2.88 and 3.46 across the document cases. So the decode rate follows the cost of the round rather than the quality of the proposals, which is why 32K — where acceptance is highest — also posts the best code rate in the table. Each case's counters and trajectory hash are in the [ledger](docs/public/AUGUR_LONG_CONTEXT_DECODE_20261017.json).

### Measurement ledgers

Each ledger has the method, per-task counters and trajectory hashes.

- [Speed milestones](docs/public/AUGUR_GDN_GLUE_B8_20261016.json): 46.21 → 75.92 native TPS on 14 English-thinking segments in four interleaved A/B/B/A steps ([probability drafts](docs/AUGUR_ENGLISH_PROPOSAL_CALIBRATION_20261016.md), [vocabulary head](docs/AUGUR_SHARED_HEAD_RW4_20261016.md), [draft Q8_0](docs/AUGUR_DRAFT_Q8_B8_DIRECT_20261016.md), [verifier glue](docs/AUGUR_GDN_GLUE_B8_20261016.md)).
- [Long-context component bill](docs/public/AUGUR_CONTEXT_ACCEPTANCE_20261016.json): 8K to 256K with CUDA-event timing per verify pass. Only the 16 full-attention layers grow with context (8.8 → 44.4 ms per pass); the 48 Gated DeltaNet layers stay at 21.8 ms.

## How it works

### One owner, closed programs

A single Rust process owns the GPU, the model weights, the KV and recurrent state, and every buffer. The CUDA code sits behind a C ABI where each operation is a pair: a `preflight` call validates device ownership, extents, alignment and aliasing once, and a `prevalidated_submit` repeats the cheap checks and enqueues. Rust types bind each submission to the plan, the stream and the retained owners, so a launch cannot outlive its buffers. Hot paths allocate nothing, and the GPU tests assert zero hot allocations.

Every request ends in a receipt. It records the sampling contract, the kernels that ran, acceptance counters and a hash of the sampled selection trajectory. The HTTP adapter checks the receipt before it answers, and benchmarks compare those hashes to prove when two builds produce identical output.

### Speculative decoding that stays exact

Each round, the draft proposes 7 tokens and the target verifies all 8 positions in one batched pass (fixed B8). Every draft row samples from a 16-candidate distribution `q`. The target accepts token *x* with probability `min(1, p(x)/q(x))` and on rejection samples a correction from the normalized positive residual `max(p − q, 0)`. This is standard speculative rejection sampling, so the output is distributed exactly as the target's own sampling at temperature 1, top-p 0.95 and top-k 20.

Proposal quality only affects speed. Each draft row uses its own temperature and gate pair-term scale (`T = 0.875×3, 0.75×4`, pair scale `1, 1.5×6`), fitted on traced rounds with leave-one-task-out validation. Proposal, acceptance, correction and bonus draws use separate SplitMix64 streams keyed by absolute token position, so rollback and prefix restore never shift the random sequence.

### A draft converted for this engine

Augur converts the original DFlash2 safetensors itself (`augur-quantize dflash-convert`) instead of using a llama.cpp GGUF conversion. The two files agree on 80 of 81 tensors byte for byte; the difference is the selector gate, the 256 × 5,120 projection that ranks each draft row's 16 candidates.

| | llama.cpp conversion | Augur conversion |
|---|---|---|
| 5-layer backbone, codebooks | 48 matrices Q8_0 | same bytes |
| Norms, convolution bases | 32 tensors F32 | same bytes |
| Selector gate | Q8_0, scale 1 | FP32 from the original BF16, × 0.45 folded in once |
| Gate contraction at run time | Q8_1-quantized input, MMVQ | FP32 input and weights, cuBLAS FP32 (pedantic) |

The factor 0.45 was chosen by cross-validation on training prompts and frozen before held-out prompts were run; on those it gave fewer verify passes for the same output. It is written into the GGUF with the gate's SHA-256, and the loader checks the hash, the folded factor and a runtime scale of exactly 1, so the factor cannot be applied twice. The gate costs 5 MiB of FP32 and applies to all text. Like every proposal change, it only affects speed: the target still verifies every token. [Converter](tools/augur-quantize/README.md) · [Gate evaluation](docs/AUGUR_EMBEDDED_GATE045_ENGLISH_ACCEPTANCE_20261015.md)

### Kernels where the time goes

At 8 tokens per pass, almost every matrix is bandwidth-bound. Augur's kernels are written around that.

- **Vocabulary head.** The 248,320 × 5,120 BF16 head runs twice per round, once for the draft and once for the verifier. Each warp keeps 4 rows in registers, loads 16-byte BF16 vectors, widens them exactly to FP32 and reuses every weight across all 8 activation rows. 8.12 → 3.23 ms per head, about 790 GB/s, with no copy of the weights. An FP16 tensor-core version was slower (4.05 ms) and about 10× less accurate.
- **Draft projections.** The draft's Q8_0 blocks are read in place: aligned 32-bit loads plus funnel shifts recover any 34-byte block offset, and one `PRMT` and one `FADD` turn each code into an exact FP32 value. Activations stay FP32, with no Q8_1 staging. Draft layers 12.9 → 7.5 ms per round, and dot-product error against FP64 falls from about 2e-3 to 3e-8.
- **NVFP4 on Volta.** Volta has no FP4 hardware. Building on NInfer's QPN layout, the target's E2M1 weights and E4M3 scales are expanded inside the tile to FP16 for SM70 tensor-core MMA with FP32 accumulation, and the tensor scale is applied once per output. Split-K is fixed per shape and merged in a fixed order, so results are deterministic.
- **Verifier glue.** Each of the 48 Gated DeltaNet layers used to wrap its projections in a dozen small launches: norm, FP16 staging, two 2-CTA gate projections, strided trims and rollback-record copies. Two closed programs now run [residual add] → RMSNorm → FP16 stage → all projections of that input in one grouped grid. Trimmed outputs and rollback records are stored by the same write. The arithmetic is unchanged: the projection tiles stay in their original compilation unit with the original compiler flags, and a GPU test compares every output byte. 50.3 → 47.0 ms per round.

### Numerical contracts

Every change is measured as either bit-identical or deliberately numerics-changing, and the two kinds are reported separately. Bit-identical changes (the head kernel on 13 of 14 tasks, the glue fusion on 14 of 14) are checked by trajectory hash. Numerics-changing ones (the Q8_0 draft) only alter proposals. Because acceptance always divides by the `q` that was actually sampled, they can change speed but not the output distribution.

## Get started

On Windows, extract the full package, run `Verify-Package.cmd`, then double-click `启动服务.exe` ("start service"). The setup page asks for the target model folder and the original DFlash2 folder, converts the draft with the bundled Rust quantizer, and starts a loopback-only, key-authenticated Chat Completions endpoint with a chat page and a monitor. The package bundles its own Python, dependencies and CUDA runtime. You provide the NVIDIA driver and the model weights. Later starts reuse the saved configuration, and the page's Quit button retires the engine and the launcher. The package ships no API key: one per-user key is generated on first start, kept in `%LOCALAPPDATA%\Augur\api-key.txt` and, by default, exported as the user variables `AUGUR_API_KEY`, `AUGUR_BASE_URL` and `AUGUR_MODEL`, so the chat page signs in by itself and other OpenAI-compatible agents can read it.

- [Windows guide](docs/AUGUR_WINDOWS_SOURCE_GUIDE.md) · [Step-by-step setup (Chinese)](docs/AUGUR_WINDOWS_USER_GUIDE.zh-CN.md) · [Build from source](BUILDING.md)
- [Model card](model-cards/qwen3.8-27b-quasar-nvfp4/README.md) · [Model sources and hashes](docs/public/AUGUR_MODEL_SOURCES.md)

## Technical reports

| Report | Topic |
|---|---|
| [Probability drafts](docs/AUGUR_ENGLISH_PROPOSAL_CALIBRATION_20261016.md) | Exact sparse rejection sampling, per-row calibration, block verification |
| [Vocabulary head](docs/AUGUR_SHARED_HEAD_RW4_20261016.md) | Register-blocked FP32 kernel, tensor-core comparison |
| [Draft Q8_0 kernel](docs/AUGUR_DRAFT_Q8_B8_DIRECT_20261016.md) | In-place block reads, exact code conversion, calibration refit |
| [Verifier glue](docs/AUGUR_GDN_GLUE_B8_20261016.md) | Grouped norm/stage/projection programs and cross-unit bit identity |
| [Unsafe trust boundaries](docs/UNSAFE_TRUST_BOUNDARIES.md) | Preflight and submit contracts for every native entry |

Each report links a JSON file with per-task counters and hashes. The [documentation index](docs/README.md) lists the rest.

## Scope and limits

- Supported target: qwen3.8-27b-quasar-nvfp4 (API alias `quasar-27b`). Platform: Windows, MSVC, SM70.
- Measurements come from one GPU. Clean-machine qualification of the Windows package is a separate release step.
- The decode figures are fixed-length thinking segments that measure speed, not answer quality. Three hard tasks still average under 3 tokens per pass. Longer contexts spend more of each pass in attention, which shrinks the relative gains above.
- The engine serves one local session owner. Tool calls are proposed to the client, not executed by Augur.

## License

Augur is available under either the [MIT License](LICENSE-MIT) or the [Apache License 2.0](LICENSE-APACHE), at your option. Portions adapted from other projects keep their own licenses, as listed in [NOTICE](NOTICE). Model weights and vendor runtimes are not covered by the engine license. See [Security](SECURITY.md) and [Contributing](CONTRIBUTING.md).

## Acknowledgments

- [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp): attention tiles, the Q8_0 decoder and the reference projection kernels that Augur's own kernels are measured against.
- NInfer: the V100 QPN weight layout and Volta NVFP4 projection design (Apache-2.0, attribution in `crates/augur-cuda-sys/native/licenses`).
- [Qwen](https://huggingface.co/Qwen) and [QUASAR-QAT](https://huggingface.co/QUASAR-QAT): the base model and its NVFP4 quantization-aware checkpoint.
- [DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) (incoai, mirrored by z-lab): the draft model.
- Sun et al., *Block Verification Accelerates Speculative Decoding* (ICLR 2025): the opt-in block verifier.
- [Lucide](https://lucide.dev): icons in the web UI.
