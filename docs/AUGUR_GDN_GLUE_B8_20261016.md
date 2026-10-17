# Fused GDN glue for the B8 verifier (2026-10-16)

[Public counters and hashes](public/AUGUR_GDN_GLUE_B8_20261016.json)

## Result

Each B8 verify round runs 48 Gated DeltaNet layers. Around their NVFP4 projections sat a dozen small launches per layer: rmsnorm, FP16 staging, two 2-CTA gate projections, strided 64→48 trims and replay-record copies. Two closed native programs now cover that glue. On 14 long English-thinking segments, the round drops from **50.31 to 46.96 ms** and native decode goes from **70.86 to 75.92 TPS (+7.1%)**. All 14 tasks produce the same trajectory hash in all four arms: the change is bit-identical.

| GDN glue | Input group | Post-attention | ms / round | Native TPS |
|---|---|---|---:|---:|
| Separate | norm, 2 stages, 4 projections, 6 copies | add, norm, stage, gate/up | 50.31 | 70.86 |
| **Fused norm group (new default)** | **2 launches** | **2 launches** | **46.96** | **75.92** |

Every task gained between +7.0% and +7.2%, and the repeat spread was at most 0.3%. Tokens per round stayed at 3.565 because the sampled tokens did not change.

## Programs

`augur_cuda_qwen35_qpn_norm_group_b8_*_v1` runs [optional residual add] → rmsnorm → FP16 stage → 2 to 4 projections of that normalized input, as two launches:

- `qpn_norm_stage_b8`: 8 CTAs × 256 threads. It keeps `prefill_rmsnorm_kernel`'s per-thread serial sum and shared-memory tree, caches the row in registers, and writes the F32 output and its FP16 RNE stage in one pass. The residual add, when present, is one FP32 add per element and is stored back to the norm input.
- `project_group`: one grid over the tiles of every member. It runs the same `project_tile<8,half,true>` code as the separate projection, so the 2-CTA alpha/beta projections now share a grid with qkv and z instead of leaving the GPU idle. A member may keep only the first 48 of its 64 rows, and may store the same words a second time to a replay record. Both copies disappear.

The GDN input group (qkv 10240, z 6144, alpha 64→48, beta 64→48) and the post-attention group (residual, then gate and up 17408) each use one program.

## Bit identity

The plan TU is built with `--fmad=false --ftz=false` and the QPN TU with default flags. To keep the projection arithmetic exactly the same, the tiles stay in the QPN TU and the plan TU enqueues them through a private descriptor (`augur_qwen35_qpn_group.h`). The GPU test `fused_norm_group_b8_is_bit_identical_to_separate_glue` compares the norm plane, every member output, every record and the residual against the separate sequence byte for byte, on 3 inputs with outlier channels. It also checks zero hot allocations and rejects aliasing records, wrong trims and single-member groups.

## Configuration and checks

- `AUGUR_GDN_INPUT_KERNEL=fused` (Windows default) or `separate`. The engine default stays `separate`. Fused requires width 8, packed B8 projections and no captured B8 GDN graph (`AUGUR_B8_GDN_GRAPH=1`). Receipts record `kernels.gdn_input`.
- ABI: every range, alias and shape is validated at bind, before any enqueue. See [unsafe trust boundaries](UNSAFE_TRUST_BOUNDARIES.md).

Measured on a single Tesla V100 32 GB (SM70), Windows, with the [direct Q8_0 draft](AUGUR_DRAFT_Q8_B8_DIRECT_20261016.md), [rw4 shared head](AUGUR_SHARED_HEAD_RW4_20261016.md) and [probability drafts](AUGUR_ENGLISH_PROPOSAL_CALIBRATION_20261016.md).
