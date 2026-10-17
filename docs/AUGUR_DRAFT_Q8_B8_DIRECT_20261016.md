# Direct Q8_0 kernel for the draft B8 projections (2026-10-16)

[Public counters and hashes](public/AUGUR_DRAFT_Q8_B8_DIRECT_20261016.json)

## Result

The DFlash draft has 5 Q8_0 layers; every B8 round projects 8 rows through all of them. A new kernel cuts the draft layers from **12.9 ms to 7.5 ms per round** with no repack and no extra VRAM. On 14 long English-thinking segments, native decode goes from **65.62 to 70.77 TPS (+7.8%)**, saving 5.4 ms per round.

| Draft projection kernel | Draft layers (GPU) | Tokens / round | ms / round | Native TPS |
|---|---:|---:|---:|---:|
| llama.cpp-derived Q8_1 MMVQ<8> | 12.91 ms | 3.661 | 55.79 | 65.62 |
| **Augur Q8_0 B8 direct (new default)** | **7.48 ms** | **3.565** | **50.38** | **70.77** |

Unlike the shared head change, this one changes the draft's numbers, so no task kept its trajectory. Acceptance moved per task in both directions: 11 of 14 tasks got faster (range −2.7% to +21.7%, geometric mean +8.1%). Pooled tokens per round fell 2.6%, which gives back part of the 9.7% cheaper round. Output stays an exact sample of the target distribution either way: the verifier and its sampling parameters are unchanged. Repeat spread within each kernel was at most 0.2%.

## Kernel

llama's MMVQ<8> first quantizes each FP32 activation row to Q8_1, then runs int8 dot products. At B8 that path reached about 138 GB/s of weight bandwidth. The direct kernel:

- Reads the original 34-byte Q8_0 blocks in place. Each lane owns 16 codes of a block; five aligned 32-bit loads plus `__funnelshift_r` recover them whatever the 2-byte block offset.
- Converts each signed code to FP32 exactly with one `PRMT` and one `FADD` (`2^23 + (b xor 0x80) − (2^23 + 128)`).
- Multiplies by the FP32 activations directly, with no activation quantization. Each 16-code partial sum is scaled once by the block's FP16 scale, which is `w = d·q` exactly as Q8_0 defines it.
- Keeps 8 rows per warp in registers and reuses every weight for all 8 activation rows. Short matrices and the 17,408-column down projection split K across 4 warps; the rest split across 2. Partials merge in a fixed order, so results are deterministic.

On real layer weights against an FP64 reference, the error falls from about 2e-3 to 3e-8 of the dot-product mass. A screen of 5-layer variants (r1–r6) put a repacked layout at 4.66 ms, but it needed about 1.7 GB of extra VRAM, so the in-place layout (about 5.0 ms in isolation) was kept.

## Configuration and checks

- `AUGUR_DFLASH_Q8_KERNEL=direct` (Windows default) or `llama` (restores MMVQ<8>). Only the B8 block projections switch; context rows keep llama dispatch. Receipts record `draft_block_projection_kernel`.
- GPU test (`dflash_q8_b8_direct_matches_f64_on_all_block_shapes`): every draft block shape with random FP16 scales and outlier activation channels. It checks every output against FP64, guard bytes on all buffers, zero hot allocations, and rejection of 7-row and misaligned input.
- ABI: same argument struct and range checks as the llama projection, plus exactly 8 tokens, `rows % 64 == 0`, `cols % 512 == 0` and 16-byte input alignment. See [unsafe trust boundaries](UNSAFE_TRUST_BOUNDARIES.md).

## Limits

The per-row proposal temperatures and pair scales were fitted for the old draft numerics. A same-state refit on direct-kernel traces (6 tasks, 18,009 draft rows, grid T ∈ {0.5…1}, pair scale ∈ {1…2.5}, leave-one-task-out) found at most +0.41% expected accepted tokens per round over the current setting, with the same optimum in every fold: T ≈ 0.875 and pair scale 1.25 on rows 2–5. That is far below rollout noise, so the calibration was kept. Measured on a single Tesla V100 32 GB (SM70), Windows, with the [rw4 shared head](AUGUR_SHARED_HEAD_RW4_20261016.md) and [probability drafts](AUGUR_ENGLISH_PROPOSAL_CALIBRATION_20261016.md).
