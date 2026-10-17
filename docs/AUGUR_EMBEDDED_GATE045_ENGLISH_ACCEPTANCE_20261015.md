# QUASAR Rust 内置 FP32 gate0.45 与英文思考验收（2026-10-15）

## 已执行的方案

- Rust 从原始 `model.safetensors` 直接生成一个 GGUF，不调用 Python 或 llama.cpp 进行转换。
- 48 个矩阵 Q8_0，32 个 norm/conv 张量精确展开为 F32；selector gate 从原始 BF16 展开到 FP32 后只乘一次 FP32 0.45，成为第 33 个 F32 张量。
- GGUF 记录 gate profile、SHA256、folded factor、runtime scale1 和 `global_gate045_v1`。加载器验证所有字段、形状、字节 SHA、有限值；不接受任意 FP32 矩阵替换，也不允许对内置 gate 叠加外置覆盖。
- 默认全局文本 gate0.45；英文／中文 thinking、代码、工具调用和关闭 thinking 后保持该 gate。图片仍走原生 target 保护。
- 使用现有 FP32 cuBLAS pedantic kernel，不加在线 MLP、额外 gate 乘法或训练增量。

## 量化与字节核验

转换耗时 17.3218082 s，输出 2,060,264,832 字节。80 个非 gate 张量与原准入 Q8 GGUF 逐字节一致；内置 gate 与原准入外置 gate 逐字节一致，也与独立从原始 BF16 展开再乘 FP32 0.45 的结果一致。

- GGUF SHA256: `33d9842166866dae79eeff0026bf90991f438ef0aaa2578429d950f536b0d797`
- Gate SHA256: `0bcb694a4e84aa304d888829742fdef92941771ac5e97917ca5dc4388f6821c7`
- 原始 draft safetensors SHA256: `67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c`
- 实测构建回执 SHA256: `2fe739d46f48c0fb571e2e1c8b94e7134e71a94186acf22bdc7fcb8bf9bdd621`

## 英文思考成绩

| 任务 | 思考／总输出 token | 草稿接受率 | 已检查草稿命中率 | 原生生成 TPS | HTTP 整请求 TPS | HTTP 首个有效输出 | HTTP EOF |
|---|---:|---:|---:|---:|---:|---:|---:|
| 条件概率 | 445 / 452 | 56.67% | 85.14% | **78.66** | 73.40 | 410.5 ms | 6.158 s |
| 非相邻组合计数 | 445 / 450 | 65.08% | 88.28% | **88.11** | 81.64 | 403.5 ms | 5.512 s |
| HTHH 等待时间期望 | 1131 / 1135 | 45.33% | 79.39% | **65.29** | 63.74 | 420.7 ms | 17.807 s |

三个任务均有正确完整答案并自然 EOS。概率题条件样本空间为 120+60+6，目标为 60，结果 10/31；组合题为 C(11,6)=462；HTHH 的前缀状态递推给出期望 18。已人工核对模型的对应思考内容和边界转换；本次不声称覆盖所有语义题型。

表为内置两轮的汇总；原生 TPS = 非 EOS 输出 token / (native whole_wall_s_before_receipt − prefill_wall_s)；HTTP TPS 包含完整请求；首个有效输出指客户端首次收到 reasoning/content/tool 非空 delta，不是首个最终答案。接受率计整个请求含短答案，不单独伪装成纯 thinking 计数；anchor 不算 draft。终止位置产生的 censored/unchecked 计数保留于 JSON。输入仅 156～168 token，不能替代 8K～256K 性能或超长思考验收。

## 外置与内置 ABBA

4 个独立 GPU owner，顺序外置 A0、内置 B1、内置 B2、外置 A3；后两轮反转文本用例顺序。同一构建、target、B8、batch8 Top-K、greedy、scale1；仅 gate 存储位置／GGUF 身份改变。没有训练、重挑 gate factor 或使用 rank4 验证集。每个 owner 先做一次短 warmup，44 个请求含 warmup；10 个非 warmup case ID、40 项机器检查全部通过。同一 case 的四次完整 token ID 相同（含 thinking）。四轮 owner 均正常清理。

| 用例 | 外置 TPS | 内置 TPS | 内置接受率 | 内置已检查命中率 |
|---|---:|---:|---:|---:|
| after_vision | 5.62 | 5.63 | 0.00% | 不适用 |
| en_conditional_probability | 78.64 | 78.66 | 56.67% | 85.14% |
| en_nonadjacent_combinations | 88.05 | 88.11 | 65.08% | 88.28% |
| en_pattern_waiting_time | 65.25 | 65.29 | 45.33% | 79.39% |
| holdout_javascript_zip_longest | 113.06 | 113.07 | 85.71% | 97.44% |
| holdout_python_histogram | 127.58 | 127.63 | 98.41% | 98.41% |
| tool_lookup | 127.58 | 127.49 | 100.00% | 100.00% |
| tool_result | 92.11 | 92.05 | 71.43% | 100.00% |
| vision_ocr | 35.25 | 35.25 | 不适用 | 不适用 |
| zh_control | 92.35 | 92.42 | 68.05% | 91.88% |

较长等待时间任务的接受率为 45.33%，低于两道短题；这是本次测得的剩余瓶颈，不做已最优的结论。外置与内置接受计数完全一致，速度差在本轮约 0.1% 的量级，不能将内置本身当作提速。

## 原始证据与可分享汇总

本机完整原始目录：`E:/quasar-embedded-gate045-20261015-v1/`。其中 `measurements/{A0,B1,B2,A3}/` 含 HTTP SSE、原生回执、token ID、工具检查与退场记录；`weight-audit.json` 含逐张量身份；`run_abba.py` 和 registration 绑定请求、顺序与设置。可分享汇总见 [计数、提示词和 SHA256](public/AUGUR_ENGLISH_THINKING_GATE045_20261015.json)，不包含模型权重、API key 或用户会话。源码／二进制候选的后续重新冻结与打包记录单独保存，不改写本次实测构建。
