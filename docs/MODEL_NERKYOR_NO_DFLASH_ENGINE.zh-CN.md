# Nerkyor source-reassoc 无 DFlash 会话入口

本轮以 `5e9e7c8` 为基线，将 `NerkyorQpnSourceReassocResearch` 接入实际 token 会话与文本 CLI。使用独立的 manifest execution identity `nerkyor_qpn_source_reassoc_f32_kv4096_no_dflash_v1`，对应 session profile `nerkyor_f32kv4096_qpn_reassoc_no_dflash_greedy_session_research_v1`。

## 执行方式

- 原始 NVFP4 码值/scale 的 QPN 物理布局，源 FP32 pair 运算与重排 FP64 求和；mode 4，不是 Tensor Core profile 8/9。
- FP32 activation/KV，4096 token 总容量，greedy，使用本模型 tokenizer、chat template 和 EOS。
- 不分配或加载 DFlash draft 权重，也不建立 feature capture；MTP 不执行。manifest 仍绑定原模型和已有 draft 配置元数据。
- exact-history append 保留 pending token 语义；预算不足在 GPU 提交前拒绝，close/reset 回收会话状态。
- 原 DFlash/512 manifest 继续选择原执行路径；两个 profile 的 ready/turn/cleanup 回执互相不能混用。

## 计时口径

实际会话 `decode_detail` 增加 `decode_steps`、`decode_wall_s`、`decode_tokens_per_second`。每一步从 `consume_selected` 开始，到下一次 greedy 选择和 4 字节 token 读回完成；累加这些区间。首个输出来自 prefill，不计入 decode 步数。输出 N 个 token 且 length 停止时，分子为 N−1；如果遇到 EOS，则 EOS 的选择按实际计算步计数。

该计时排除加载、prefill、tokenizer、文本输出及回执写盘。另行记录客户端首末 token 到达之间的 TPS，以观察输出通道开销。会话保留 EOS；原生固定 128 步的对照测试忽略 EOS，仅用于吞吐诊断。

QUASAR 对照使用其现有 QPN、Q8_0 KV 和 262144 容量路径；Nerkyor 使用 FP32 KV/4096。两者不是相同 KV 算法的消融实验。

## 本地入口与证据

保留构建目录：`evidence/model-portability/nerkyor-coder390-20261018/implementation-gpu/no-dflash-session-20261019-102528`（开发机私有证据目录，不随 Release 分发）。

启动脚本为该目录下 `launch-nerkyor-no-dflash.ps1`。支持 `-Prompt`、`-Interactive`、`-MaxNewTokens` 与 `-Json`；每次运行建立新的证据目录。该脚本固定 producer 和 manifest SHA256。

本轮数值检查：同构建128输入+32步，3076组状态全部逐字节一致，64000次候选投影全部走QPN source-pair，无原NVFP4回退。该记录绑定原研究构建；本次发布不新增或宣称 no-DFlash TPS 基准结果。
