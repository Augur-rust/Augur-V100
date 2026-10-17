# 固定 B8 HTTP EOF 账单（2026-10-13）

用户决定：active DFlash 默认固定 B8，取消性能自适应；vision/capacity 保护仍保留。验收构建为 F:/quasar-fixed-b8-build-20261013-v24，原始 ABBA 为 F:/quasar-fixed-b8-http-20261013-v24-abba/complete.json。

## 同构建正常计时

顺序为 deferred / fixed / fixed / deferred；两题均启用英文 thinking，预算 16384，自然 EOS。GPU/host 细分 profiling 关闭，pinned D2H 关闭。每种策略重复的输出 token IDs、接受计数和轮数一致。以下取固定策略两次均值。

| 指标 | Python | Probability |
|---|---:|---:|
| HTTP EOF，秒 | 12.062659 | 16.722131 |
| prefill，秒 | 0.380510 | 0.383619 |
| native generation，秒 | 11.653610 | 16.314762 |
| draft proposal，秒 | 3.970610 | 5.606716 |
| verify+commit，秒 | 7.666792 | 10.685276 |
| 生成结束→客户端 EOF，毫秒 | 1.983 | 2.479 |
| native decode TPS | 66.331 | 68.404 |
| 输出 token / 完整 HTTP EOF | 64.082 | 66.738 |
| 输出 token（不含 EOS） | 773 | 1116 |
| thinking / boundary / answer | 706 / 1 / 66 | 728 / 1 / 387 |
| EOS，额外计数 | 1 | 1 |
| B8 轮数 | 184 | 256 |
| B1 consumed token | 0 | 0 |
| accepted / proposed | 589 / 1288 = 45.73% | 860 / 1792 = 47.99% |
| accepted / checked | 589 / 730 = 80.68% | 860 / 1055 = 81.52% |

## 差异解释

固定 B8 后两题完整 EOF 差 4.659472 秒。Probability 多 343 个输出 token，其中 thinking 多 22、最终答案多 321；多执行 72 轮。其 native TPS 和接受率均略高。每轮均摊 native generation 为 Python 63.335 ms、Probability 63.730 ms。以 Python 每轮成本分解：新增 72 轮约 4.560108 秒，余下每轮成本差约 0.101044 秒。此为账单算术分解，不是因果消融。

HTTP EOF 减 native generation 仍包括 prefill，不能称为 HTTP finish。真实收尾约 2 ms，无秒级差距。新增 token 中答案占 93.6%，不等于墙钟归因占比。

旧 deferred 的 Python EOF 11.087/11.097 秒、684 token；Probability EOF 26.154/26.129 秒、1206 token。固定后分别为 773/1116 token，不能将旧新 EOF 差冒充同输出 kernel 提速。此前 Probability 波动来自成本门控短窗口穿过阈值后改变 B1/B8 分配；诊断 profiling 本身可改变选择。

## 验证及失败记录

- Python 两次完整答案实际执行空数组、首次出现顺序、中文字符串三组测试均通过，输入不变。
- Probability 两次最终答案相同，完整条件样本空间为 9，其中一红一蓝 6，结果 2/3；人工核对未把“至少一红”混同为“第一球红”。
- v22 漏补枚举启动分支导致构建失败，修复后重建；v23 warmup 缺少 completion_tokens_details 导致报告提取失败，修复测量脚本后重跑 v24。原失败证据保留，未修改冻结源/收据。
- v24 Rust backend 237 passed/71 ignored、runtime 645 passed/46 ignored、Python 283 tests OK；ignored GPU 用例不算通过。
- pinned D2H v21 没有稳定收益，仍为关闭的实验项。此报告不授予任意提示词语义质量资格。

产品验收另见 AUGUR_WINDOWS_PRODUCT_ACCEPTANCE_20261013.md。
