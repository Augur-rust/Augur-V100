# Augur Windows 源码候选（2026-10-15）

仓库名称为 `Augur`，默认主模型为 `quasar-27b`。这是源码交付候选，不是已完成最终验收的安装包，也尚未公开发布。

## 当前交付边界

- 源码包含 Rust/CUDA 引擎、Python HTTP/UI、固定 B8＋gate0.45 启动预设、可搬移 Windows 二进制打包工具、只读监控、DSH 配置与版本绑定的适配补丁。
- 不包含模型或 DFlash 权重、运行时、凭据、真实对话、训练/验证样本或本地证据目录。
- 保留现有 gate0.45 作为产品验收基线。rank4 v7 已因整任务留出交叉验证出现旧前缀损失而放弃，不再搜索新的 rank4 候选；生产 gate 未被替换。
- `rust-toolchain.toml`、`Cargo.lock` 和 Python 依赖清单一并交付。Python 最低 3.11，本机产品回归使用 3.13.9；训练环境 `qwen_env` 的 3.10 不能代替产品运行环境。

## Selector gate：与 llama.cpp 参考路径的区别

产品基线为 Rust 直接生成的 **Q8_0 主干＋GGUF 内置 FP32 gate0.45**。gate 从原始浮点权重展开并在 FP32 中折入 0.45 一次，运行时 scale=1；默认用于全部 DFlash 文本生成，vision 保留原生保护。与本地 llama.cpp b11552 参考相比，差异是 QUASAR 明确的 gate 校准与身份契约。

## Prefill 实测峰值

B4352 的 8K 档历史单次峰值为 **1,183.09 TPS**：实际新计算 **8,176 token / 6.9107113 秒**。同一组两次测量分别为 6.9107113 / 7.0084983 秒，按总 token / 总墙钟汇总为 **1,174.78 TPS**。

测量日期为 2026-10-11，物理 KV 容量 262,144；完整原生 fresh prefill，无前缀缓存复用，未加载 DFlash、未启用草稿特征捕获，不经过 HTTP。峰值属于该历史构建与用例，并非当前固定 B8 产品请求的吞吐。

原始账单目录：`F:/quasar-context-staircase-8k64k-native-20261011-v1/`。`complete.json` 状态为 `PASS`；两份样本均已按其中的 SHA256 重新核对：

| 原始样本 | SHA256 |
|---|---|
| `code_8k-r0.json`（峰值） | `94449f60b57e24f372ca76200b7a57dd7fb7170e2d6a52323ba7fa202ae79ebe` |
| `code_8k-r1.json` | `0f303cae43356b71fd870f5d5e79cc8353563b68efcbc551f8763b5cc4ef38ac` |

构建身份：`quasar-cache95-persistence-build-20261011-v1`；二进制 SHA256：`af256eac886c4a2892118cda9a8893cbba7f42289ff228d84cda479032cfb153`。

## 多模态与显存驻留

支持图像输入；主模型与 DFlash 常驻显存，视觉编码器按需加载，完成编码后释放 GPU 显存。

## 启动与监控

完整步骤见 [Windows 源码使用说明](AUGUR_WINDOWS_SOURCE_GUIDE.md)。固定 B8 预设为 `scripts/run_augur_windows.py`；它使用 batch8 Top-K、固定 B8 的七位置概率草稿与 p/q 验收、target temperature=1/top-p=0.95/top-k=20 采样、B4352 prefill、262144 物理 KV、gqa8_planar、bytelevel_v2 和 receipt_ref_v3。性能自适应关闭，vision/capacity 保护保留。预设清除本进程遗留 `AUGUR_` 实验变量，默认由 Rust 加载器校验内置 gate0.45（旧格式才需 `--selector-gate-file`），通过 `--live-file` 接入实时 token 观测。

DSH 示例保持用户选择的 249036 输出上限；输入与生成之和仍须满足真实物理上下文预算。95% 压缩是客户端轮间策略，不代表单次尚未结束的生成能够无限延长。详见 `integrations/dsh/README.md`。

监控模块和静态网页随源码交付。GPU 指标为采样值，实时 TPS 需要有效 observer 文件；没有有效 observer 时不应显示伪造实时 TPS。DFlash 接受率使用最新已完成请求回执，`accepted/proposed` 与 `accepted/checked` 分开显示，均不等于输入前缀缓存命中率。

## 证据不能跨版本冒用

历史固定 B8 账单见 [HTTP 账单](AUGUR_FIXED_B8_HTTP_LEDGER_20261013.md)，其数据只属于文中绑定的 v24 构建和用例，不自动适用于本源码候选。CPU 测试、源码哈希核对或导入成功都不能证明新 GPU 二进制的质量和 TPS。物理 256K 容量也不等于所有长度的质量已验收。

正式发布前仍须完成候选构建、同构建功能/GPU/HTTP 验收、依赖许可清单与干净环境验证。保留 `product_ready=false`、`published=false`，直至有对应证据。模型和第三方 runtime 的许可单独核实，源码许可不重新授权它们。

## 许可和隐私

保留 LICENSE-MIT、LICENSE-APACHE、NOTICE，以及 llama.cpp、NInfer 和 Lucide 的单独许可。源码包逐文件记 SHA256；禁止把 API key、原始会话、缓存快照和模型权重加入包内。
