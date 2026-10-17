# Probability proposals and per-row calibration for sampled English thinking (2026-10-16)

[Public counters and hashes](public/AUGUR_ENGLISH_PROPOSAL_CALIBRATION_20261016.json)

## Result

With the target sampling contract unchanged (temperature=1, top-p=0.95, top-k=20, min-p=0, no penalties, seed=0), the new default raises native decode throughput on 14 long English-thinking segments from **46.21 to 55.49 TPS (+20.1%)** and output tokens per verify round from **3.03 to 3.64**. Every task is faster than the previous product path; per-task gains range from +6.6% to +41.9%.

| Arm | What changes | Native TPS | Tokens / round | vs product |
|---|---|---:|---:|---:|
| G0 | Previous product: deterministic draft, token-equality acceptance | 46.21 | 3.028 | — |
| B7 | All 7 draft rows proposed from `q = softmax(score)`, sparse `min(1, p/q)` acceptance | 53.64 | 3.519 | +16.1% |
| **TQ3** | B7 plus per-row temperature and pair-term scale (**new default**) | **55.49** | **3.642** | **+20.1%** |
| BQ3 | TQ3 calibration with block verification (opt-in) | 55.87 | 3.670 | +20.9% |

Hardware: single Tesla V100 32 GB (SM70), Windows, TCC. Target: QUASAR Qwen3.8-27B NVFP4. Draft: DFlash2 Q8_0 with embedded FP32 gate0.45, fixed B8, weights unchanged and not retrained.

## Method

- 14 English reasoning prompts: 6 used during calibration screening, 8 held out and run only after the setting was frozen. On the 8 holdouts alone, TQ3 is +3.9% over B7.
- One request per task per arm, each generating 3,072 tokens (all segments reached the length limit). These are fixed-length long-thinking segments, not completed answers or a quality benchmark.
- Native decode TPS = output tokens / native decode seconds, summed over tasks. Prefill and HTTP overhead are excluded. No prefix-cache replay.
- Tokens per round = output tokens / verify rounds, including the target's correction or bonus token.

## What the default does

1. **Probability proposals on every draft row.** Each of the 7 draft positions samples from a 16-candidate distribution `q` built by the selector walk, and the target accepts with `min(1, p/q)`. On rejection the correction comes from the normalized positive residual `max(p - q, 0)`. This is standard speculative rejection sampling, so the output distribution equals the target's sampling distribution.
2. **Per-row calibration.** Before the softmax, row `r` uses temperature `T[r]` and scales the gate pair term by `s[r]`. Default: `T = 0.875, 0.875, 0.875, 0.75, 0.75, 0.75, 0.75` and `s = 1, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5`. Row 1 has no predecessor, so its pair scale must be 1. Calibration only reshapes `q`; it cannot bias the output distribution because acceptance always divides by the `q` that was actually sampled.
3. **Independent random streams.** Proposal, acceptance, correction and bonus draws use separate SplitMix64 domains keyed by absolute token position.

Configuration (defaults set by `scripts/run_augur_windows.py`):

```text
AUGUR_DFLASH_PROBABILITY_PREFIX=7
AUGUR_DFLASH_PROBABILITY_TEMPERATURES=0.875,0.875,0.875,0.75,0.75,0.75,0.75
AUGUR_DFLASH_PROBABILITY_PAIR_SCALES=1,1.5,1.5,1.5,1.5,1.5,1.5
AUGUR_DFLASH_VERIFICATION=token
```

## Block verification stays opt-in

`AUGUR_DFLASH_VERIFICATION=block` implements block verification (Sun et al., ICLR 2025, Algorithm 2) on the sparse Top-20 / Top-16 support. It is also distribution-preserving and checked by an exact joint-law test. It averaged +0.7% over TQ3 but varied more between tasks (−13.5% to +8.4%), so it is not the default.

## Verification

- Rust: exact joint-law tests for token and block verification over enumerated small distributions; CUDA oracle comparing the v1/v2/v3 proposal ABIs; negative ABI tests for null pointers, non-finite values and a row-1 pair scale other than 1.
- Python: an independent re-implementation replays recorded rounds and checks the recorded acceptance decisions.
- GPU: every receipt records the calibration, the verification contract and a selection-trajectory hash.

## Limits and what was rejected

- Three hard tasks remain below 3 tokens per round (2.71, 2.75, 2.87). Raising them likely needs a better draft rather than more proposal tuning.
- Learned proposal shapes, a pos1-only second draft chain and a CUDA Graph for the 5 draft layers were measured or estimated and not adopted: each gained less than about 1–2% after cost.
- The largest remaining draft cost is the shared BF16 vocabulary head in B8 (about 8 ms of a 21 ms proposal round). It is the next optimization target.
