# Augur

**v0.0.1-alpha** · 首个公开 Alpha 版本。

[English](README.md)

Augur 是一个 Rust/CUDA 推理引擎，在单张 Volta GPU（SM70，32 GiB）上运行 27B NVFP4 推理模型。它严格按主模型的分布采样，同时用小草稿模型让主模型每次前向产出多个 token。

在 14 段长英文思维链上，以 temperature 1、top-p 0.95、top-k 20 采样，原生解码 **75.9 tokens/s**。是同一 GPU 上常规确定性草稿基线的 1.64 倍，输出分布不变。

## 性能

硬件：单张 Tesla V100 32 GB（SM70），Windows，TCC 驱动。主模型：[QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4)，原始权重与 BF16 输出头。草稿：DFlash2，由 Augur 转换为 Q8_0，selector gate 为 FP32（见[下文](#为本引擎转换的草稿)）。原生 TPS = 输出 token / 原生解码秒数，不含 prefill。

**不会为了测试而放宽采样参数。** 本页所有数字都在产品契约下测得（temperature 1、top-p 0.95、top-k 20），采用逐步分布精确采样：每个输出 token 都是主模型自身分布的样本。

### 长上下文，8K 到 256K

每个提示各发两次。第一次是冷的——此前任何缓存里都没有它。第二次紧接其后发出，用来显示持久化前缀存储对首 token 时间的作用；它报告的解码速度按同样方法测量，不受这次复用影响。每个用例输出 1,024 个 token，不开启思考。

| 用例 | prompt token | 冷 prefill | 冷 TTFT | 原生解码 | 复跑原生解码 |
|---|---:|---:|---:|---:|---:|
| code_8k | 7,353 | 984.2 tok/s | 7.47 s | 83.19 tok/s | 83.28 tok/s |
| code_32k | 31,506 | 934.0 tok/s | 33.74 s | 91.63 tok/s | 91.83 tok/s |
| code_186k | 186,247 | 533.1 tok/s | 349.43 s | 59.38 tok/s | 59.41 tok/s |
| code_256k | 260,230 | 437.8 tok/s | 594.47 s | 55.81 tok/s | 55.99 tok/s |
| en-doc_8k | 7,630 | 929.4 tok/s | 8.21 s | 64.66 tok/s | 64.65 tok/s |
| en-doc_32k | 31,834 | 914.0 tok/s | 34.84 s | 54.08 tok/s | 54.16 tok/s |
| en-doc_193k | 192,954 | 520.5 tok/s | 370.72 s | 45.39 tok/s | 45.45 tok/s |
| en-doc_256k | 260,214 | 455.2 tok/s | 571.66 s | 34.98 tok/s | 34.95 tok/s |

两种长度都装在这一张卡上。复跑时 256K 的 prefill 只需 3.70 s、186K 只需 2.67 s，而冷启分别是 594 s 和 349 s；存储按配额保留完整前缀，第二次只为首 token 付费。唯一的例外是 `en-doc_256k`：它的复跑被配额挤出存储、未命中，又付了一次完整的冷 prefill；但它的原生解码速度不受影响，因为该速度不含 prefill。整个运行的解码速度重复稳定性在 0.33% 以内，这正是冷热两列一致的原因：前缀存储改变的是第一个 token 何时到达，而不是其余 token 走多快。

每轮本身会随长度变贵：代码用例在 8K 时 49.8 ms，32K 时 53.2 ms，186K 时 75.0 ms，256K 时 85.3 ms，英文文档是同样的形状。随上下文增长的只有 16 个全注意力层，48 个 Gated DeltaNet 层保持固定，差值就是更长那一轮的成本。接受率走的是另一个轴：草稿在 8K 的代码上每轮命中 4.15 个 token，32K 时 4.88，256K 时 4.76，文档用例则落在 2.88 到 3.46 之间。所以解码速度跟随的是每轮成本，而不是提案质量——这也解释了为什么接受率最高的 32K 同时给出表里最好的代码速度。每个用例的计数与轨迹哈希见[账本](docs/public/AUGUR_LONG_CONTEXT_DECODE_20261017.json)。

### 测量账单

每份账单包含方法、逐任务计数与轨迹哈希。

- [提速里程碑](docs/public/AUGUR_GDN_GLUE_B8_20261016.json)：14 段英文思维链上，四步交错 A/B/B/A 测量，从 46.21 提升到 75.92 原生 TPS（[概率草稿](docs/AUGUR_ENGLISH_PROPOSAL_CALIBRATION_20261016.md)、[词表头](docs/AUGUR_SHARED_HEAD_RW4_20261016.md)、[草稿 Q8_0](docs/AUGUR_DRAFT_Q8_B8_DIRECT_20261016.md)、[验证胶水](docs/AUGUR_GDN_GLUE_B8_20261016.md)）。
- [长上下文分项账单](docs/public/AUGUR_CONTEXT_ACCEPTANCE_20261016.json)：8K 到 256K，用 CUDA event 计时每次验证前向。只有 16 个全注意力层随上下文增长（每次前向 8.8 → 44.4 ms），48 个 Gated DeltaNet 层保持 21.8 ms。

## 工作原理

### 单一所有者，闭合程序

一个 Rust 进程独占 GPU、模型权重、KV 与循环状态以及所有缓冲区。CUDA 代码位于 C ABI 之后，每个操作都是一对调用：`preflight` 一次性校验设备归属、范围、对齐与别名；`prevalidated_submit` 重复廉价检查后入队。Rust 类型把每次提交绑定到计划、流和被持有的所有者上，launch 不可能比它的缓冲区活得更久。热路径零分配，GPU 测试会断言这一点。

每个请求都以一份回执结束，记录采样契约、实际运行的 kernel、接受计数和采样选择轨迹的哈希。HTTP 适配层先校验回执再回复；基准测试比较这些哈希，以证明两个构建何时输出完全一致。

### 保持精确的投机解码

每一轮草稿提出 7 个 token，主模型在一次批量前向中验证全部 8 个位置（固定 B8）。每个草稿行从 16 个候选的分布 `q` 中采样。主模型以概率 `min(1, p(x)/q(x))` 接受 token *x*，拒绝时从归一化的正残差 `max(p − q, 0)` 中采样修正。这是标准的投机拒绝采样，输出分布与主模型自身在 temperature 1、top-p 0.95、top-k 20 下的采样完全相同。

提案质量只影响速度。每个草稿行有自己的温度和门控成对项缩放（`T = 0.875×3, 0.75×4`，pair scale `1, 1.5×6`），在追踪的轮次上拟合，并做留一任务交叉验证。提案、接受、修正与奖励抽样使用独立的 SplitMix64 流，按绝对 token 位置取值，因此回滚和前缀恢复不会改变随机序列。

### 为本引擎转换的草稿

Augur 用自己的转换器（`augur-quantize dflash-convert`）直接转换原始 DFlash2 safetensors，不使用 llama.cpp 转出的 GGUF。两份文件 81 个张量中有 80 个逐字节相同，差别在 selector gate，也就是为每个草稿行的 16 个候选打分的那个 256 × 5,120 投影。

| | llama.cpp 转换 | Augur 转换 |
|---|---|---|
| 5 层主干、codebook | 48 个矩阵 Q8_0 | 字节相同 |
| Norm、卷积基 | 32 个张量 F32 | 字节相同 |
| Selector gate | Q8_0，缩放 1 | 由原始 BF16 展开为 FP32，× 0.45 只折叠一次 |
| 运行时 gate 计算 | 输入量化为 Q8_1，MMVQ | FP32 输入与权重，cuBLAS FP32（pedantic） |

系数 0.45 在训练提示上交叉验证选出，跑留出提示前已经冻结；在留出提示上，相同输出所需的验证前向更少。系数连同 gate 的 SHA-256 写进 GGUF，加载器校验哈希、折叠系数以及运行时缩放必须正好为 1，所以系数不会被重复乘。gate 占 5 MiB FP32，对全部文本生效。和其他提案改动一样，它只影响速度：每个 token 仍由主模型验证。[转换器](tools/augur-quantize/README.md) · [gate 评估](docs/AUGUR_EMBEDDED_GATE045_ENGLISH_ACCEPTANCE_20261015.md)

### 把 kernel 写在耗时处

每次前向只有 8 个 token，几乎所有矩阵都受带宽限制。Augur 的 kernel 围绕这一点设计。

- **词表头。** 248,320 × 5,120 的 BF16 词表头每轮运行两次，草稿一次、验证一次。每个 warp 把 4 行留在寄存器里，读取 16 字节 BF16 向量，精确展开为 FP32，每个权重复用于全部 8 行激活。每次从 8.12 ms 降到 3.23 ms，约 790 GB/s，不复制权重。FP16 tensor core 版本更慢（4.05 ms），误差约大 10 倍。
- **草稿投影。** 草稿的 Q8_0 块原地读取：对齐 32 位读取加 funnel shift 适配任意 34 字节块偏移，一条 `PRMT` 加一条 `FADD` 把码值精确转换为 FP32。激活保持 FP32，不经 Q8_1 暂存。草稿层每轮从 12.9 ms 降到 7.5 ms，相对 FP64 的点积误差从约 2e-3 降到 3e-8。
- **Volta 上的 NVFP4。** Volta 没有 FP4 硬件。在 NInfer 的 QPN 布局基础上，主模型的 E2M1 权重和 E4M3 缩放在 tile 内展开为 FP16，供 SM70 tensor core MMA 以 FP32 累加，张量缩放在每个输出上只乘一次。Split-K 按形状固定，按固定顺序合并，结果确定。
- **验证胶水。** 48 个 Gated DeltaNet 层原本在投影周围各有十几次小 launch：norm、FP16 暂存、两个 2-CTA 的门控投影、跨步裁剪和回滚记录拷贝。现在两个闭合程序在一个分组网格里完成 [残差加法] → RMSNorm → FP16 暂存 → 该输入的全部投影。裁剪输出和回滚记录由同一次写入完成。算术不变：投影 tile 留在原编译单元、沿用原编译参数，GPU 测试逐字节比较每个输出。每轮从 50.3 ms 降到 47.0 ms。

### 数值契约

每项改动都被测量并归为逐位一致或有意改变数值两类，分别报告。逐位一致的改动（词表头在 14 个任务中的 13 个，胶水融合在全部 14 个）用轨迹哈希检验。改变数值的改动（Q8_0 草稿）只影响提案；由于接受概率总是除以实际采样所用的 `q`，它们能改变速度，但不能改变输出分布。

## 开始使用

在 Windows 上完整解压发布包，运行 `Verify-Package.cmd`，再双击 `启动服务.exe`。设置页面会要求选择主模型目录和原始 DFlash2 目录，用自带的 Rust 量化器转换草稿，然后启动只监听本机、需密钥认证的 Chat Completions 接口，并提供聊天页面和监控页面。发布包自带 Python、依赖和 CUDA 运行时；NVIDIA 驱动和模型权重需自行准备。之后再双击会直接沿用已保存的配置启动；页面上的“退出 Augur”会关闭服务并结束启动器。发布包不含 API Key：首次启动时为当前用户生成一个，保存在 `%LOCALAPPDATA%\Augur\api-key.txt`，默认同时写入用户环境变量 `AUGUR_API_KEY`、`AUGUR_BASE_URL`、`AUGUR_MODEL`。聊天页自动登录，其他 OpenAI 兼容的 Agent 直接读取即可。

- [Windows 指南](docs/AUGUR_WINDOWS_SOURCE_GUIDE.md) · [分步设置](docs/AUGUR_WINDOWS_USER_GUIDE.zh-CN.md) · [从源码构建](BUILDING.md)
- [模型卡](model-cards/qwen3.8-27b-quasar-nvfp4/README.zh-CN.md) · [模型来源与哈希](docs/public/AUGUR_MODEL_SOURCES.md)

## 技术报告

| 报告 | 主题 |
|---|---|
| [概率草稿](docs/AUGUR_ENGLISH_PROPOSAL_CALIBRATION_20261016.md) | 精确稀疏拒绝采样、逐行校准、块验证 |
| [词表头](docs/AUGUR_SHARED_HEAD_RW4_20261016.md) | 寄存器分块 FP32 kernel，与 tensor core 的对比 |
| [草稿 Q8_0 kernel](docs/AUGUR_DRAFT_Q8_B8_DIRECT_20261016.md) | 原地块读取、精确码值转换、校准重拟合 |
| [验证胶水](docs/AUGUR_GDN_GLUE_B8_20261016.md) | 分组 norm/暂存/投影程序与跨编译单元逐位一致 |
| [unsafe 信任边界](docs/UNSAFE_TRUST_BOUNDARIES.md) | 每个原生入口的 preflight 与提交契约 |

每份报告都链接一份 JSON，含逐任务计数与哈希。其余文档见[文档索引](docs/README.md)。

## 范围与限制

- 支持的主模型：qwen3.8-27b-quasar-nvfp4（API 别名 `quasar-27b`）。平台：Windows、MSVC、SM70。
- 测量来自单张 GPU。Windows 发布包的干净机器验证是单独的发布步骤。
- 解码数字是固定长度的思维片段，衡量速度而非答案质量。三个难题平均每次前向仍不到 3 个 token。上下文越长，每次前向花在注意力上的比例越大，上述相对提升会缩小。
- 引擎服务一个本地会话所有者。工具调用由客户端执行，Augur 只负责提出。

## 许可

Augur 可按 [MIT 许可](LICENSE-MIT)或 [Apache License 2.0](LICENSE-APACHE) 任选其一使用。改编自其他项目的部分保留各自的许可，见 [NOTICE](NOTICE)。模型权重和厂商运行时不受引擎许可覆盖。另见[安全说明](SECURITY.md)与[贡献指南](CONTRIBUTING.md)。

## 致谢

- [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)：注意力 tile、Q8_0 解码器，以及 Augur 自有 kernel 用作对照的参考投影 kernel。
- NInfer：V100 QPN 权重布局与 Volta NVFP4 投影设计（Apache-2.0，署名见 `crates/augur-cuda-sys/native/licenses`）。
- [Qwen](https://huggingface.co/Qwen) 与 [QUASAR-QAT](https://huggingface.co/QUASAR-QAT)：基础模型及其 NVFP4 量化感知训练检查点。
- [DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)（incoai，z-lab 镜像）：草稿模型。
- Sun 等，*Block Verification Accelerates Speculative Decoding*（ICLR 2025）：可选的块验证器。
- [Lucide](https://lucide.dev)：网页界面图标。
