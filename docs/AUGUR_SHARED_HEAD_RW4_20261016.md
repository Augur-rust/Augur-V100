# Register-blocked FP32 kernel for the shared BF16 vocabulary head (2026-10-16)

[Public counters and hashes](public/AUGUR_SHARED_HEAD_RW4_20261016.json)

## Result

The shared 248,320 × 5,120 BF16 head runs twice per B8 round: once for the draft and once in the target verifier. A new kernel cuts each B8 head from **8.12 ms to 3.23 ms**. On 14 long English-thinking segments, native decode goes from **55.53 to 65.59 TPS (+18.1%)**, saving 9.8 ms per round. The weights, the numerics contract and VRAM use are unchanged.

| Kernel | Verify head (GPU) | Draft head + TopK + selector | ms / round | Native TPS |
|---|---:|---:|---:|---:|
| llama.cpp-derived MMVF<8> | 8.12 ms | 21.3 ms | 65.58 | 55.53 |
| **Augur FP32 rw4 (new default)** | **3.23 ms** | **16.5 ms** | **55.82** | **65.59** |

On 13 of the 14 tasks, all four ABBA runs produced byte-identical selection trajectories and outputs. These are pure timing comparisons, and every one gained between +17.3% and +17.6% (pooled +17.5%). On the 14th task the two kernels diverged after token 2,592 of 3,072. Both runs stayed valid samples from the target distribution. The divergent continuation happened to accept more drafts (3.06 vs 2.87 tokens per round), which explains that task's +25.4%. Repeat spread within each kernel was at most 0.2%.

## Kernel

Each warp owns 4 vocabulary rows. Each lane loads 8 BF16 weights per row as one 16-byte vector, widens them to FP32 with shifts (exact), and reuses every weight for all 8 activation rows. Products are exact FP32 FMAs and accumulation stays in FP32. The kernel reads the original BF16 tensor with no copy or conversion pass and adds no workspace, transfer or synchronization. Against MMVF<8>, only the summation order changes.

The old kernel launched one 256-thread CTA per vocabulary row. That reached about 306 GB/s of weight bandwidth. The new kernel reaches about 790 GB/s on the same data.

## Why not FP16 tensor cores

Volta tensor cores were the first candidate. A standalone screen on 61,440 real head rows against an FP64 reference:

| Arm | Full-vocab estimate | Max error / (‖w‖·‖x‖) |
|---|---:|---:|
| MMVF<8> | 8.32 ms | 1.4e-8 |
| FP16 WMMA, activation hi/lo compensation | 4.05 ms | 4.9e-7 |
| FP16 WMMA, no compensation | 4.05 ms | 1.5e-5 (one Top-20 flip on another slice) |
| **FP32 rw4** | **3.22 ms** | **4.8e-8** |

At B8 the head is bandwidth-bound: each weight is used only 8 times. Tensor cores remove arithmetic that is not the bottleneck, but they add BF16→FP16 conversion and staging. They also need an extra compensation pass to match FP32 accuracy. Register blocking on CUDA cores ended up both faster and about 10× more accurate. BF16→FP16 conversion is exact at scale 2¹⁶ for all but 8 of 1.27 billion weights, so the tensor-core path stays an option for wider batches.

## Configuration and checks

- `AUGUR_SHARED_HEAD_KERNEL=rw4` (Windows default) or `llama` (restores MMVF<8>). One setting binds both the draft head and the B8 verifier. Each receipt records `shared_head_kernel.draft` and `.verify`.
- GPU test (`f32_rw4_head_matches_mmvf8_and_f64_reference`): full-vocabulary geometry with outlier activations. It compares against MMVF<8> on every logit and against an FP64 reference on sampled rows. It also checks that the Top-20 sets match for every column, rejects misaligned input at bind, and verifies output guard bytes and zero hot allocations.
- ABI: same argument struct and range checks as MMVF<8>, plus 16-byte weight/input alignment. See [unsafe trust boundaries](UNSAFE_TRUST_BOUNDARIES.md).

Measured on a single Tesla V100 32 GB (SM70), Windows, with the 2026-10-16 decoding defaults ([probability drafts](AUGUR_ENGLISH_PROPOSAL_CALIBRATION_20261016.md)). Longer contexts spend more time in attention per round, so the relative gain will be smaller there.
