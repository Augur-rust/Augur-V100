# qwen3.8-27b-quasar-nvfp4 — Augur 模型卡

[English](README.md) · [支持模型](../README.md) · [制品身份清单](artifact-manifest.json)

更新：2026-10-15。**当前仅支持这一个 target 模型。** 本卡说明引擎对上游模型的适配，不是重新训练的模型发布，也不将上游模型归为本项目原创。

## 模型身份与结构

| 项目 | 当前内容 |
|---|---|
| 模型标识 | `qwen3.8-27b-quasar-nvfp4` |
| 默认产品 API 别名 | `quasar-27b` |
| 上游 checkpoint | `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` |
| 已记录上游 revision | `d9af17e20644c5b9a6d66dff1d14f349e2e61b05` |
| 上游声明的基础模型 | `Qwen/Qwen3.8-27B` |
| 参数规模标签 | 27B；本卡未独立重新统计参数总数 |
| 文本层 | 64 层：48 层 GDN／线性注意力＋16 层全注意力 |
| Hidden／词表大小 | 5,120／248,320 |
| 全注意力头 | 24 个 query head／4 个 KV head |
| Target 权重 | 原始已准入 NVFP4 safetensors；原始 BF16 输出 head |
| 当前产品模态 | 文本、图片输入；文本、思考与工具调用输出 |

准入配置中的架构字符串为 `qwen3_5`／`Qwen3_5ForConditionalGeneration`，不代表自动支持其他 Qwen 权重。相同结构、相似文件名或另一个量化版本都不能替代已绑定的权重、config、tokenizer 和模板身份。相邻制品清单记录了这些 SHA256；本轮复核了小型元数据，未重新哈希大体积权重分片。

## 当前推理配置

- Windows x64、Rust/MSVC、CUDA SM70；实测为单张 Tesla V100 32 GB。仓库名不代表所有 V100 容量或所有 NVIDIA 显卡均已验收，也未据此确认 16 GiB 部署可行。
- Prefill B4352；QPN target、GDN 与 planar GQA 路径。NVFP4 指输入权重格式，不是在 SM70 上使用 Blackwell 原生 NVFP4 指令的声明。
- Target 权重不变。配套 DFlash2 由 Rust 直接转换为 Q8_0＋内置 FP32 selector gate：48 个 Q8_0 矩阵＋33 个 FP32 张量。gate 从原始浮点权重仅折入一次 0.45，运行时 scale=1；这是 draft 制品的处理，不是重新量化 target。
- 文本产品预设为固定 B8。图片请求走原生 target 保护路径。Vision 默认 `on_demand`，可显式选择 `auto` 可回收常驻。
- 默认 target 采样为 temperature=1、top-p=0.95、top-k=20、min-p=0、presence/frequency penalty=0、seed=0；采用固定 B8 概率草稿及精确的稀疏 p/q 验收。Thinking 默认关闭；启用后默认英文，可选择中文。思考语言偏好不强制最终答案语言。

DFlash 是配套加速器，不是另一个已支持的主模型。量化方案和 draft／gate SHA256 见 [制品清单](artifact-manifest.json) 与 [Rust 量化器说明](../../tools/augur-quantize/README.md)。

## 上下文与调用

容量为 **262,144 token positions**。输入／历史、视觉 token、生成的思考与最终答案共同消耗这份容量。API 输出上限为 **249,036 token**，还要受剩余上下文约束，并不是任意长输入之后都能额外输出这么多 token。不存在无限输出或服务端默认静默裁剪历史；客户端压缩属于单独流程。

推荐产品启动器的模型别名为 `quasar-27b`。带认证的 `/v1/chat/completions` 请求体示例：

```json
{
  "model": "quasar-27b",
  "messages": [{"role": "user", "content": "解释二分查找为什么要求输入有序。"}],
  "max_completion_tokens": 2048,
  "stream": true,
  "enable_thinking": true,
  "thinking_language": "en"
}
```

这里只实现文档列出的 Chat Completions 子集，不承诺任意 API 字段兼容。图片使用有资源上限的 PNG/JPEG data URL；本适配不宣称支持音频／视频输入。工具调用由客户端授权并执行。见 [Windows 指南](../../docs/AUGUR_WINDOWS_SOURCE_GUIDE.md) 与 [DSH 集成](../../integrations/dsh/README.md)。

## 已记录的性能与功能检查

下表沿用 **2026-10-15 已有测量**，不冒充本轮重命名源码的新 GPU 验收。单卡 32 GiB SM70、Windows、固定 B8、内置 FP32 gate0.45，每个用例两次内置测量，按总 token／总墙钟汇总，而不是挑最快一轮。

| 任务 | 原生生成 TPS | HTTP 整请求 TPS | 接受率 | 已检查草稿命中率 |
|---|---:|---:|---:|---:|
| 英文：条件概率 | 78.66 | 73.40 | 56.67% | 85.14% |
| 英文：非相邻组合 | 88.11 | 81.64 | 65.08% | 88.28% |
| 英文：HTHH 等待时间 | 65.29 | 63.74 | 45.33% | 79.39% |
| Python ordered_counts | 127.63 | 81.23 | 98.41% | 98.41% |
| JavaScript zipLongest | 113.07 | 86.12 | 85.71% | 97.44% |
| 中文概率推理 | 92.42 | 79.44 | 68.05% | 91.88% |
| 工具调用 | 127.49 | 42.43 | 100.00% | 100.00% |
| 工具结果续写 | 92.05 | 16.01 | 71.43% | 100.00% |

这里的原生生成 TPS **包含 DFlash 加速且不含 prefill**，不是纯 target decode TPS。HTTP TPS 含完整请求。接受率为 accepted/proposed，已检查草稿命中率为 accepted/checked，anchor 不算 draft。英文思考指标覆盖含短答案的整请求，不冒充纯 thinking 计数；三道英文输入仅 156～168 token，不是 8K～256K 上下文。代码／工具输出较短，工具结果续写仅 6 token，且复用了已有会话。

已记录用例中的 Python／JavaScript 执行检查、所列推理答案、工具调用及结果续写、少量 OCR 样例通过。不据此宣称任意提示词、所有代码或完整多模态 benchmark 均已通过。完整提示词、计数、TTFT／EOF 与构建身份见 [公开测量汇总](../../docs/public/AUGUR_ENGLISH_THINKING_GATE045_20261015.json) 和 [验收报告](../../docs/AUGUR_EMBEDDED_GATE045_ENGLISH_ACCEPTANCE_20261015.md)。

历史原生无缓存 8K prefill 峰值 **1,183.09 TPS**，实际输入 8,176 token，未加载 DFlash；它与固定 B8 HTTP 吞吐不是同一配置。见 [prefill 账单](../../docs/AUGUR_SOURCE_RELEASE_README.md)。Vision 显存检查覆盖实际 8K／32K／64K prefill＋完整容量 KV 分配，不能等同于已经跑过 256K 输入，见 [驻留报告](../../docs/AUGUR_VISION_RESIDENCY_AND_LONG_PREFILL_20261015.md)。

## 限制、归属与分发

- 不保证任意提示词上的推理正确、代码安全、长思考完整收尾或工具参数正确。生成结果需要复核，外部 Agent 需要自行实施权限控制和执行隔离。
- 容量声明不等于所有长度、语言、图片预算与草稿组合的质量／性能验收。Packed-B8 计算口径也不宣称与标量 target 逐位一致。
- 本适配未独立确认上游训练数据组成、安全评估、偏差指标或标准 benchmark 分数；上游声明与本引擎实测分开，不将其他引擎成绩搬成本项目成绩。
- 上游模型卡声明 Apache-2.0；引擎源码许可、第三方署名与模型权重许可各自独立。仓库不包含权重、密钥和私有会话。不可变上游身份与再分发核验状态见 [来源记录](../../docs/public/AUGUR_MODEL_SOURCES.md)。本卡的描述性 manifest 不是启动用 `--manifest`，不能绕过准入。
- 使用本地带认证的产品入口；嵌入或开放服务前阅读 [SECURITY](../../SECURITY.md) 与 [unsafe／CUDA 边界](../../docs/UNSAFE_TRUST_BOUNDARIES.md)。

构建运行见 [BUILDING](../../BUILDING.md)；上游作者与原始来源链接见 [QUASAR 模型来源](../../docs/public/AUGUR_MODEL_SOURCES.md)。
