# qwen3.8-27b-quasar-nvfp4 — Augur model card

[中文](README.zh-CN.md) · [Supported models](../README.md) · [Artifact manifest](artifact-manifest.json)

Updated: October 15, 2026. **This is the only currently supported target.** This card describes an engine integration, not a newly trained checkpoint or a claim of authorship of the upstream model.

## Identity and architecture

| Item | Value |
|---|---|
| Target identifier | `qwen3.8-27b-quasar-nvfp4` |
| Default product API alias | `quasar-27b` |
| Upstream checkpoint | `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` |
| Recorded checkpoint revision | `d9af17e20644c5b9a6d66dff1d14f349e2e61b05` |
| Upstream-declared base | `Qwen/Qwen3.8-27B` |
| Parameter-size label | 27B; this card does not independently recount parameters |
| Text stack | 64 layers: 48 GDN / linear-attention + 16 full-attention layers |
| Hidden / vocabulary sizes | 5,120 / 248,320 |
| Full-attention heads | 24 query heads / 4 KV heads |
| Weights | Original admitted NVFP4 safetensors; original BF16 output head |
| Modalities in this product | Text and images; text, reasoning and tool-call outputs |

The admitted config uses `qwen3_5` / `Qwen3_5ForConditionalGeneration`. These implementation identifiers do **not** advertise general support for other Qwen checkpoints. Model names and interchangeable tensor shapes do not replace admitted weight/config/tokenizer/template hashes. Recorded identities are in the adjacent manifest; small metadata was rehashed for this card, but large weight shards were not rehashed in this documentation pass.

## Selected execution configuration

- Windows x64, Rust/MSVC, CUDA SM70. The measured system is one Tesla V100 32 GB. The repository name does not qualify every V100 capacity or every NVIDIA GPU; 16 GiB deployment is not established by these results.
- B4352 prefill; QPN target, GDN and planar GQA paths. NVFP4 is the source-weight format, not a claim of native Blackwell NVFP4 instructions on SM70.
- Target weights stay unchanged. Rust converts the paired DFlash2 to Q8_0 with an embedded FP32 selector gate: 48 Q8_0 matrices + 33 FP32 tensors. Gate0.45 is folded once from the original floating-point gate; runtime scale is 1. This changes the draft artifact, not the target quantization.
- Fixed B8 is the selected text preset. Image requests use the native-target safeguard. Vision defaults to `on_demand`; reclaimable residency via `auto` is explicit opt-in.
- Generation samples from the target (temperature 1, top-p 0.95, top-k 20, min-p 0, no presence/frequency penalty, seed 0), using fixed B8 probability drafts and exact sparse p/q verification. Thinking is off by default; when enabled, English is the default reasoning language and Chinese is available. The final-answer language is independent of that preference.

The paired draft is not a second supported target. Draft/gate SHA256 values and conversion details are in [the artifact manifest](artifact-manifest.json) and [the Rust quantizer guide](../../tools/augur-quantize/README.md).

## Context and API

Capacity is **262,144 token positions**. Input/history, visual tokens, generated reasoning and the final answer share it. The API output ceiling is **249,036 tokens**, further limited by remaining context; it does not reserve that many free output positions after any prompt. There is no unlimited output or implicit server-side history truncation. Client-side compaction is a separate workflow.

The recommended product launcher exposes `quasar-27b`. Example request body for its authenticated `/v1/chat/completions` endpoint:

```json
{
  "model": "quasar-27b",
  "messages": [{"role": "user", "content": "Explain why binary search needs a sorted input."}],
  "max_completion_tokens": 2048,
  "stream": true,
  "enable_thinking": true,
  "thinking_language": "en"
}
```

This is a documented Chat Completions subset, not blanket compatibility with every API field. Images use bounded PNG/JPEG data URLs; audio/video inputs are not advertised by this integration. Clients authorize and execute tool calls. See the [Windows guide](../../docs/AUGUR_WINDOWS_SOURCE_GUIDE.md) and [DSH integration](../../integrations/dsh/README.md).

## Recorded performance and application checks

These are **existing October 15, 2026 measurements**, not a new GPU run of renamed source. Single 32 GiB SM70 GPU, Windows, fixed B8, embedded FP32 gate0.45, two measured embedded runs per case. Rates use summed tokens / summed wall time, not best-run selection.

| Task | Native generation TPS | HTTP request TPS | Acceptance | Checked-draft hit |
|---|---:|---:|---:|---:|
| English: conditional probability | 78.66 | 73.40 | 56.67% | 85.14% |
| English: non-adjacent strings | 88.11 | 81.64 | 65.08% | 88.28% |
| English: HTHH waiting time | 65.29 | 63.74 | 45.33% | 79.39% |
| Python ordered_counts | 127.63 | 81.23 | 98.41% | 98.41% |
| JavaScript zipLongest | 113.07 | 86.12 | 85.71% | 97.44% |
| Chinese probability reasoning | 92.42 | 79.44 | 68.05% | 91.88% |
| Tool call | 127.49 | 42.43 | 100.00% | 100.00% |
| Tool-result continuation | 92.05 | 16.01 | 71.43% | 100.00% |

Native generation TPS includes DFlash acceleration and excludes prefill; it is **not target-only decode TPS**. HTTP TPS includes the entire request. Acceptance = accepted/proposed; checked-draft hit = accepted/checked; anchors are excluded. English figures cover the whole request including its short final answer. The three English prompts contain 156–168 tokens, not 8K–256K contexts. Code/tool outputs are short; the continuation is only six tokens and reuses its session.

Python/JavaScript executable checks, the listed reasoning answers, a tool-call/result exchange and a small OCR fixture passed their recorded checks. These are application cases, not broad benchmark scores. Prompts, counts, TTFT/EOF and build identity are in [the public measurement summary](../../docs/public/AUGUR_ENGLISH_THINKING_GATE045_20261015.json) and [the evaluation report](../../docs/AUGUR_EMBEDDED_GATE045_ENGLISH_ACCEPTANCE_20261015.md).

Historical uncached native 8K prefill peak: **1,183.09 TPS** for 8,176 input tokens, with DFlash not loaded. This is a separate configuration, not fixed-B8 HTTP throughput; see [the prefill ledger](../../docs/AUGUR_SOURCE_RELEASE_README.md). Vision memory checks cover actual 8K/32K/64K inputs with a full-capacity KV allocation, not a tested 256K-input workload; see [the residency report](../../docs/AUGUR_VISION_RESIDENCY_AND_LONG_PREFILL_20261015.md).

## Limitations, attribution and distribution

- No guarantee of correct reasoning, safe generated code, complete long answers or correct tool arguments for arbitrary prompts. Review outputs; external agents need their own permissions and execution isolation.
- Context capacity is not a qualification of all lengths, languages, image budgets and speculative paths. Packed-B8 arithmetic is not asserted to be bitwise identical to scalar target execution.
- This integration does not independently establish upstream training-data composition, safety/bias evaluations or standardized benchmark scores. Upstream claims are not Augur engine results.
- The upstream card declares Apache-2.0. Engine source licensing and third-party notices are separate. Weights, credentials and private sessions are not included. Immutable upstream identity and redistribution checks are tracked in [model provenance](../../docs/public/AUGUR_MODEL_SOURCES.md). The descriptive manifest cannot replace the startup admission manifest.
- Use the local authenticated entry point. Read [SECURITY](../../SECURITY.md) and [unsafe/CUDA boundaries](../../docs/UNSAFE_TRUST_BOUNDARIES.md) before embedding or exposing it.

Build/run: [BUILDING](../../BUILDING.md). Upstream publisher and source links: [QUASAR model sources](../../docs/public/AUGUR_MODEL_SOURCES.md).
