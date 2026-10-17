# QUASAR unsafe / CUDA 信任边界

更新：2026-10-15。状态：**首轮边界梳理与关键入口注释，不是全仓 unsafe 审计通过证明。**

本轮不修改 kernel、精度、矩阵形状、调度或同步行为，不增加在线检查开销。冻结的旧源码包、二进制和测试回执保持不变；此文与注释须在下一次源码冻结时纳入新身份，不能反写旧包的 SHA256。

## 1. 边界图

```text
不可信请求 / 图片 / 工具结果
           |
  HTTP 鉴权 + 模板/长度/结构校验
           |
  单 owner 会话协议：请求、取消、提交、退场
           |
  target / draft / selector / vision 资源 owners
           |
  Rust buffer、stream、event、plan 包装器
           |  此层仍有未编码的生命周期约束，见 GAP-*。
  unsafe Rust FFI / C ABI
           |
  C++ / CUDA kernels / cuBLAS / CUDA runtime / NVIDIA driver
```

最外层校验不能替代底层指针安全；底层范围检查也不能代替最外层鉴权。原生依赖属于可信计算基，并不因为在 Rust 进程中运行就获得借用检查器的保护。**没有出现 `unsafe` 关键字的 C++/CUDA、Python 原生扩展与 DLL 同样在边界之内。**

## 2. 必须维持的契约

| ID / 边界 | 代码已有的检查或机制 | 仍需 owner / 调用协议保证 |
|---|---|---|
| **TB-ABI**：Rust ↔ C/CUDA ABI | `repr(C)`、部分 size/align/offset 断言、版本/struct_size 字段、各入口参数校验 | 实际编译 feature/调用约定一致；host 指针存储有效、对齐、范围足够且在调用期间存活；不能把任意非零地址或已销毁句柄交给 native。返回错误码不能补救先发生的非法解引用。 |
| **TB-DEVICE**：设备地址与 layout | device ordinal、长度、checked arithmetic、布局和部分非重叠检查；owner incarnation ID | 地址仍属于原来的活跃分配；host 地址与 device 地址不可混用；GPU 读写方向、对齐、stride、pitch、shape 与真实分配一致。相同地址/相同长度不等于相同 allocation。 |
| **TB-ASYNC**：提交与完成 | stream/event、submission epoch、`CudaStreamCompletionProof`、部分 pending 状态机 | 提交成功仅代表入队；同 stream 排序不自动阻止另一 stream/CPU 访问。所有被访问资源必须保留到覆盖相应提交的终结完成证明，不能拿另一 stream、旧 event 或旧 epoch 的成功结果代替。 |
| **TB-HOST**：pinned/mapped H2D、D2H | host allocation owner、部分 event wait / proof-bound read；分配 shim 初始化 host 字节 | GPU 写入完成前不能读取；GPU 正在读取期间不能修改/释放源。Rust 的 `&mut` 不能排除持有 device view 的 GPU 并发访问。见 GAP-MAPPED。 |
| **TB-GRAPH**：capture / replay | graph handle owner、设备/stream 检查、部分执行计划保留 | capture 结束不延长 closure 内借用。权重、KV、hidden、scratch、host staging 地址在每次 replay 与其完成期间必须稳定；更换/回收资源前要停止旧 graph 使用并重建相关 plan。 |
| **TB-STATE**：target / draft / selector / cache | 会话 cursor、proposal/commit、poisoned 状态与特定缓存身份检查 | memory safety 不等于模型状态正确。hidden 层号、token 位置、KV/GDN 边界、accepted 前缀和 selector 数值口径需一致；不得提交尚未验证的 draft 状态或把错误路径标记为已提交。 |
| **TB-ERROR**：取消、部分提交与释放 | 显式 close/drain、部分 quarantine/leak 路径、产品 cleanup 回执 | 失败/取消不等于 GPU 已停止访问资源。不能为使计数归零而提前 free，也不能吞掉 completion failure 后恢复复用；策略须逐条路径审查。`Drop` 的 best-effort 尝试不是成功证明。 |
| **TB-FILES**：模型、缓存、DLL 与转换器 | 选定产品身份/SHA、格式/范围校验、冻结构建映射、部分文件锁及独占创建 | 信任摘要的来源、保护目录不被并发替换。哈希不是签名、沙箱或恶意格式证明。Win32 空闲空间查询是瞬时观察，不是预留磁盘容量。 |

### 异步操作的关键顺序

```text
验证元数据和真实 owners
 -> 保持资源/访问权限
 -> 提交，记录确切 stream + epoch
 -> 排除冲突访问
 -> 获取覆盖此次提交的完成证明
 -> 允许读取 D2H / 复用 / 提交状态 / 释放
```

若提交了一部分再报错，资源义务仍然存在；不能因为函数返回 `Err` 就直接当作“从未使用”。完成不可证明时应保持不可复用状态，并执行该路径规定的 quarantine/owner 退场策略。**本轮没有证明所有历史 API 都已实现这条规则。**

## 3. 可定位的实现入口

- [原始 FFI 与 Safety 声明](../crates/augur-cuda-sys/src/lib.rs)、[C ABI 头文件](../crates/augur-cuda-sys/native/augur_cuda_runtime.h)、[native runtime](../crates/augur-cuda-sys/native/augur_cuda_runtime.cu)：host handle、device token、返回码与结构体约定。
- [CUDA 资源与完成协议](../crates/augur-backend/src/cuda_runtime.rs)：`CudaBufferView`、`OwnedCudaGraphExec`、`CudaD2dBatch`、`OwnedPinnedH2dBuffer`、`OwnedPinnedD2hBuffer`、`CudaStreamCompletionProof`、mapped host slices。
- [BF16 shared-head 预绑定计划](../crates/augur-backend/src/cuda_qwen35_prefill_cublas/dflash_head.rs)：几何/stream 检查与 buffer 生命周期是两个不同问题，见 GAP-VIEW。
- [B8 verifier 的 pinned D2H](../crates/augur-model-qwen35-runtime/src/checkpoint_target_cuda/dflash_bridge/batched.rs)：提交所用 `self.ids` owner 保持存活；读取经 `wait_and_read`，错误不能转换成“已完成”。
- [prefill phase](../crates/augur-model-qwen35-runtime/src/checkpoint_target_cuda/augur_prefill.rs)：最后 hidden 行的异步拷贝依赖 phase/target owners；成功路径同步后释放，并在传播执行结果前调用 phase.finish。失败 drain 仍须单独审查。
- [Vision 驻留回收](../crates/augur-model-qwen35-runtime/src/checkpoint_target_cuda/image_prefill/residency.rs)、[视觉 encoder](../crates/augur-model-qwen35-runtime/src/vision_cuda.rs)：默认按需；可选常驻回收必须经过 encoder 关闭协议。显存不足不是提前销毁活跃 graph/buffer 的许可。
- [检查点 Win32 空间查询](../crates/augur-model-qwen35-runtime/src/session_checkpoint/available_space.rs)、[Rust 量化器](../tools/augur-quantize/src/lib.rs)：同步 FFI 输入/输出存储与错误检查；不涉及把路径或字节直接当作 GPU 指针。

这些是首轮定位入口，不是完整清单；没有逐个复核所有 feature-gated kernel、unsafe block 和依赖库。

## 4. 明确未关闭的类型/API 缺口

### GAP-VIEW：非拥有 view 与 safe 预绑定调用

`CudaBufferView` 是 `Copy`，没有关联 `OwnedCudaBuffer` 的借用生命周期。范围/device/owner 元数据不能阻止 owner 先释放。部分 safe `bind` / `submit` API，例如 `CudaDflashBf16HeadV1`，仅保存 raw 参数，不保留 weight/input/output owners。

当前受控执行路径依赖外围 plan/target owner 维持约束。这不能推出该 API 对任意安全 Rust 调用序列都是 sound。**给 safe 函数增加“调用者必须保证”的注释，不会自动修复内存安全前提泄漏。** 后续应选择：编码 borrow/owner lease，或缩小可见性并将必要的 unsafe 契约显式放到真正调用边界；对真实调用点补证明和测试，而不是全仓机械加 `unsafe`。

### GAP-GRAPH：graph handle 所有权不等于捕获资源所有权

`OwnedCudaGraphExec` 持有 graph handle，但并未在其类型中持有所有捕获 buffer。safe capture/launch 的签名不表达这些资源的 replay 生命周期。复用/驱逐图时仍需审查 enclosing plan，未来应引入资源 lease 或等价的明确 unsafe 边界。

### GAP-MAPPED：Rust 借用不能覆盖设备访问

`OwnedCudaMappedHostBuffer::as_slice/as_mut_slice` 与 `device_view` 之间没有完整的 CPU/GPU 访问状态机。`&self` 或 `&mut self` 只能解释 Rust 侧别名，不能单独证明 GPU 未持有旧地址并发写/读。应以完成 proof / 状态 guard 控制切换，或暴露明确 unsafe 的底层访问；本轮只披露和补注释，未改 API。

### GAP-ERROR：异常退场不是统一证明

部分 pinned/D2D 路径在不能证明完成时显式保持 pending 或泄漏隔离；其他 `Drop` 路径只是忽略 `close` / synchronize 的错误。不能把一个模块的 quarantine 保证推广到所有资源。需继续逐路径覆盖 allocation、preflight、submit、event-record、wait、graph-capture、取消与 close 故障，并检查“不错误地读/复用/释放”。

以上是可从接口与实现确认的契约缺口/审计范围，不是声称已验证某个 HTTP 可利用漏洞，也不是宣称不存在漏洞。

## 5. 新增或修改 unsafe 的评审要求

- `unsafe fn` 写 `# Safety`：列出具体 caller 义务；每个 `unsafe` block 写 `// SAFETY:` 解释本调用为何满足，不能只说“已检查”或“为了性能”。
- 至少注明：owner/分配实例、设备、有效字节范围与对齐、读写/别名、异步持续时间、completion 来源、成功/部分失败/取消时的释放归属。
- 除非已经有具体证明，不添加 `unsafe impl Send/Sync`。锁能否覆盖 GPU 尚未完成的访问，也要单独分析。
- 优先使用现有 ownership/completion 协议，避免为一处提速扩大 raw pointer API。不要用无条件同步掩盖未知契约；需要同步时要说明正确性原因，另列延迟账单。
- ABI/layout/长度/偏移用 checked arithmetic 和匹配的断言/负例；错误路径和资源替换与成功路径同等重要。
- 评审安全、数值、性能分开：无 OOM、GPU 分配归零、token 相同、接受率上涨，都不能单独证明无 UB。CPU 单测也不能覆盖 CUDA kernel；ignored GPU 测试不能算已执行。Miri/常规 Rust 分析不能代替实际 GPU/native 验证。
- 发布只报告做过的检查；缺口没有关闭就保留，不把 `SAFETY` 注释数量当作安全认证。

## 6. 本轮交付与限制

本轮添加公开边界文档、README/SECURITY 入口、关键源码注释及发布白名单检查；对变更的 Rust/C 入口验证为注释行变动，不改变执行逻辑。没有补跑 GPU，也没有以旧 GPU 回执认证新的源码 SHA。后续优先处理 GAP-VIEW/GAP-GRAPH 的所有权表达，再处理 mapped host 与异常退场的统一协议。


## 7. 2026-10-15：分块精确 Top-K 的新增边界

`augur_cuda_qwen35_vocab_topk_tiled_submit_v1` 是独立 ABI 入口；原 Top-K v1 保留。额外 scratch 必须与 logits、ids、scores 三个范围完全不重叠，设备一致、4-byte 对齐，长度精确为 rows × parts × K × 8；K 限制为 16/20，parts 限制为 8/16/32/64。Rust 与 native 两侧都校验这些约束和地址溢出。

scratch 的前半是每块候选 ID，后半是原始 FP32 分数字节。两次 kernel 提交在同一 stream 上有序执行，不在热路径申请/释放内存，不增加 host 拷贝。`Sampler` 与 `SharedHead` 持有分配 owner；正常读取、commit 和既有退场协议覆盖两次提交。第一次提交成功后第二次提交失败，第一阶段仍可能在队列中，调用者仍须保持 scratch 与原 owners 到 stream 完成；错误码不授予提前释放许可。原有 GAP-VIEW/GAP-ERROR 审计限制继续适用。

本次执行了独立排序 GPU oracle（含 tie、signed zero、subnormal、NaN/Inf）、scratch 别名/长度/ordinal/溢出负例及真实固定 B8 采样轨迹对照。验证保留 token-id tie 和 score bits；不能把这些测试扩展为全仓 FFI soundness 证明。

## 2026-10-16 补充：实验性稀疏概率草稿

此补充描述新实验入口，不改变前文历史审计范围，也不表示全仓审计完成。

- `cuda_qwen35_dflash_q8_selector.rs::submit_probability_prefix` 新增第8个 device view：8×16 FP64 q，恰好1024字节、8字节对齐。同设备、溢出、与全部7个既有视图的重叠均在 Rust 与 native 两层检查。caller 仍须保持所有 owners 存活到同 stream 完成，不能把通过范围校验理解为生命周期证明。
- CUDA walk 的路径和 q 属于同一次条件提案；不能用别轮 q、softmax 重建的猜测 q 或另一 selector 政策的 q 验收。host 读取必须在该生产 stream 完成后进行。
- q 的归一化、有限性、唯一候选、实际选择 token 的正概率、绝对位置，以及后续确定性行的 one-hot 由 runtime 检查。q 随 move-only resident proposal 绑定 owner/generation，不能独立授予 commit 权限。
- 概率验收只决定“接受前缀＋唯一 pending 修正/bonus”，KV/GDN 回放仍由原有提交协议执行；拒绝后的 speculative tail 不得发布。失败仍走 poisoned/drain 边界。
- gate 权重仍为 FP32；q 中 FP64 是小候选概率计算的数值口径，与 prefill 精度门槛无关。
- 证据及尚未完成的产品边界见 `AUGUR_SPARSE_PROBABILITY_VERIFY_20261016.md`。

### Pos2–4 proposal calibration (experimental, 2026-10-16)

`augur_cuda_qwen35_dflash_q8_selector_probability_submit_v2` keeps v1 unchanged. Its eight device ranges retain the same exact size, ordinal, alignment, non-aliasing and lifetime checks. Both Rust and native validate finite temperature `[0.5,1.5]`, pair scale `[0,2]`, and all-seven probability positions before launch. Calibration changes only proposal rows2–4 in the same fused sequential walk. Actual sampled predecessor is used for the next row; the saved q is the actual calibrated distribution, not uncalibrated scores. No new allocation, host transfer or kernel launch is introduced relative to v1. Target sampling and residual rejection remain unchanged. Invalid parameters fail closed, including direct raw ABI tests. Raw traces and receipts carry calibration identity; independent replay rejects mismatches. Neither v1 nor this experiment is a release-default promise.

`augur_cuda_qwen35_dflash_q8_selector_probability_submit_v3` generalizes v2 to per-position proposal calibration for rows 1–7. The two host pointers must reference exactly seven finite floats each; the native entry rejects null pointers, reads both arrays synchronously before returning and copies them into kernel parameters by value, so caller stack lifetime is sufficient and no device allocation or host transfer is added. Rust and native both require positions==7, temperature `[0.5,1.5]`, pair scale `[0,2]`, and row-1 pair scale exactly 1 (row 1 has no draft predecessor). v2 now maps onto the same per-row table with identity values outside rows 2–4; scale 1 and temperature 1 are exact identities in the kernel arithmetic. Device range, alignment, non-aliasing and lifetime obligations are unchanged from v2. Traces and receipts record `proposal_calibration`; the replay oracle accepts legacy `proposal_pos234` and rejects mismatches.

### Shared BF16 head, register-blocked FP32 kernel (2026-10-16)

`augur_cuda_dflash_bf16_head_f32_rw4_{preflight,submit}_v1` take the same `augur_cuda_bf16_f32_matvec_args_v1` and the same exact B8 geometry, ordinal, SM70, range and non-aliasing checks as the MMVF<8> entry, plus 16-byte alignment of the weight and input bases, because each lane issues 16-byte vector loads. Misaligned bases fail closed at both preflight and submit, and the Rust bind rejects them before any launch. The grid tiles 248,320 rows exactly (static assert), so no row guard is needed and no thread reads or writes outside the declared views; the GPU test checks output guard bytes on both sides. The kernel adds no allocation, host transfer or synchronization. Lifetime and exclusive-output obligations are unchanged (GAP-VIEW). `AUGUR_SHARED_HEAD_KERNEL` selects the schedule at bind time for both the draft head and the B8 verifier; receipts record the bound kernel for each side.

### Draft Q8_0 B8 direct projection (2026-10-16)

`augur_cuda_dflash_q8_b8_direct_{preflight,submit}_v1` take the same `augur_cuda_qwen35_prefill_f32_projection_args_v1` and run the same `llama_projection_metadata_valid` ordinal, plan, range and non-aliasing checks as the llama draft projection entry. They additionally require exactly 8 tokens, `rows % 64 == 0`, `cols % 512 == 0` and a 16-byte-aligned FP32 input base, because the kernel tiles rows without guards and issues 16-byte activation loads. Weight code words are read with aligned 32-bit loads that stay inside each 544-byte row step. The entry uses no plan scratch, so it cannot race the shared llama workspace, but the caller must still keep weight, input and output alive until the same stream completes. The Rust wrapper rejects non-8 widths and misaligned views before reaching native code; the GPU test asserts both rejections and guard bytes around every buffer.

### Direct-B8 norm group (2026-10-16)

`augur_cuda_qwen35_qpn_norm_group_b8_{preflight,prevalidated_submit}_v1` take one 512-byte `augur_cuda_qwen35_qpn_norm_group_b8_args_v1` (static assert on both sides). Both entries require exactly 8 tokens, 5120 norm columns, 2 to 4 profile2/SplitK8 members whose input is the norm output, and member shapes of 10240, 6144 or 17408 full rows or 64 rows trimmed to 48. Unused member slots must be all zero. Each member's untrimmed producer extent goes through the existing `qpn_prefill_valid` check; the trimmed outputs, optional records, optional residual operands, norm planes and the plan's FP16 staging buffer are then checked together for extent and mutual disjointness, and preflight also confirms device ownership of every range. Submission repeats every scalar, extent and overlap check but not driver queries. The grouped projection descriptor (`augur_qwen35_qpn_group.h`) is private: only this validated entry builds it, and only `augur_qpn_project_group_b8_enqueue` consumes it, by value. The program writes the plan's staging buffer, so the caller must not run another plan consumer on a different stream concurrently (same obligation as the separate stage). The Rust `CudaQwen35AugurNormGroupB8V1::bind` adds plan and stream identity, default norm arithmetic, direct-B8 members and residual-output = norm-input checks, runs native preflight at bind, retains every owner, and synchronizes the stream on drop.
