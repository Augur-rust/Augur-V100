# QUASAR 本地 DSH 配置（Windows）

`augur-provider.json` 是无凭据的推荐配置，默认主模型为 `quasar-27b`，loopback API 为 `http://127.0.0.1:8294/v1`。它引用 `AUGUR_LOCAL_API_KEY`，不存储真实 key。模型上下文 262144、输出上限 249036；支持 off/low/high/xhigh 档位。`low` 传入模型模板的 low，`high` 映射到模板支持的 medium，`xhigh` 传入 xhigh，`off` 关闭思考；开启思考但未指定强度时保留模板的 xhigh 默认值。强度通过 `reasoning_effort` 传递，文本与 Vision 使用同一映射。该设置影响模板提示，不代表固定的思考 token 配额。

`configure_augur_dsh.mjs` 会核对本机 DSH schema、备份现有文件并写入本地配置/凭据，因此应关闭 DSH 后再运行。源码包本身不含用户配置或凭据：

```powershell
node scripts/configure_augur_dsh.mjs --install "C:/Path/To/DSH Desktop Beta" --home "C:/Users/USERNAME/.dsh-beta" --key-file "C:/Augur/private/local.key"
```

引擎需使用同一个本地 key 文件启动。不要公开该文件、私有 profile 或 `.credentials.yaml`。

## 版本绑定的兼容补丁

`dsh-patch.json` 与 `scripts/patch_augur_dsh.mjs` 仅适用于其中标记的 `@deepseek-ai/dsh-llm-pi-ai` 版本和源码摘要，遇到不同版本/摘要必须拒绝，而不是强行替换。先使用 `--check` 检查；只有核对过原始版本后才在独立安装副本上应用并进行工具/多轮/思考档位测试。此补丁不是任意新版 DSH 的通用适配，也不代表当前全部 DSH 后续改动已经被这一份历史补丁覆盖。

## 自动压缩

`configure_augur_dsh.mjs` 同时安装 `augur-compaction.json` 中的 QUASAR 专用压缩策略：总上下文 **262144**、95% 压缩阈值 **249036**、`outputReservationTokens=0`、余量 `13107`、保留尾部 `16384`、摘要输出最多 `8192`。脚本先验证本机 provider 与 compaction schema，旧 DSH 不支持这些字段时拒绝写入；它不会修改 compaction 插件实现。已有其他模型策略保留，QUASAR 同目标策略替换。

已经配置好本机 provider 和凭据时，可只对齐上下文与压缩策略，不读取或修改凭据、API 地址及默认模型：

```powershell
node scripts/configure_augur_dsh.mjs --install "C:/Path/To/DSH Desktop Beta" --home "C:/Users/USERNAME/.dsh-beta" --context-only true
```

DSH 请求到的引擎也必须选择 **256K**；引擎改成其他档位时，这份固定256K provider 不会自动随之修改。生效需重启 DSH。监控的压缩提示线按实际引擎容量计算，并显示当前请求的输入加已输出 token（完成后包含待消费尾 token）；DSH 使用客户端请求估算，因此两者触发时刻可能有差异。

输出 249036 是配置上限，非保证剩余空间。自动压缩发生在请求/轮次边界，不意味着单次流式思考可以无限生成。工具和图片仍须遵守服务端实际约束；不可依靠删除错误消息掩盖预算不足。

新版 DSH 将压缩器放在 Agent 预设的隔离组内，宿主上的 `compaction-basic` 已禁用。配置脚本同时更新已安装的 standard、ptc、cordis 等包含压缩器的预设，保留各预设的完整插件列表、用户修改和 `!!js` 条件。预设覆盖固定了当前列表；升级 DSH 后应重新核对预设变化。配置完成后重启 DSH，让已有会话重新按新预设装载；只修改宿主上的同名配置不会修复会话中的压缩预算。


## 同任务循环恢复及本机请求排队

配置脚本默认安装本仓库 `reasoning-recovery` 插件，仅作用于 `augur-local/quasar-27b`。已核验 DSH 核心版本为0.2.1-alpha.2；其他版本拒绝安装，须重新验收。插件文件按内容哈希安装到 DSH 的 `node_modules/@augur` 下，配置文件带备份，旧插件不覆盖。`--reasoning-recovery false` 可跳过安装，但不会自动删除已有插件配置。

恢复只响应服务端的 `reasoning_repetition` 信号。服务端默认只在贪婪解码时检测重复（`AUGUR_REASONING_LOOP_GUARD=auto`），默认的采样解码不会发出该信号，插件此时不介入。重复思考信号触发最多两次恢复：追加可追溯的提示，第一次临时low/8192，第二次off/16384（同时服从更小的用户上限）。恢复保留同一轮及已完成工具结果，失败思考不加入后续模型历史。第二次仍循环或截断时才终止本轮。只对精确匹配的错误恢复，不重试普通错误，不接管用户停止；成功后的正常请求恢复原配置。计数来自持久会话投影，压缩不会重置当前轮的恢复额度。

同一DSH宿主的Augur请求使用可取消FIFO队列，覆盖不同会话与压缩调用。切换会话不自动取消旧会话。停止后的HTTP清理可能稍晚于DSH界面结束，新请求遇到接纳前409会等待重试（最多30秒）；已经开始输出的请求不重放。需要有限的provider重试策略，不支持与无限 `always` 重试混用。

验收命令（模型流由固定测试脚本提供，使用已安装的真实AgentLoop、session、工具和Loader）：

```powershell
node integrations/dsh/reasoning-recovery/test.mjs "C:/Path/To/DSH Desktop Beta"
node integrations/dsh/reasoning-recovery/single-flight.test.mjs
```

这些测试验证流程和取消语义，不能当成任意提示词的模型脱环成功率。


## 工具改名与历史兼容

当前请求的 `tools` 只规定模型下一次可以生成的工具调用。已完成的历史记录可以包含旧名称、已卸载工具或旧参数格式；服务端保留这些历史调用和对应结果，检查名称、JSON对象、调用ID及结果配对，不再用当前工具声明重新校验旧记录。新生成的调用仍严格校验当前声明及参数schema，不会因为历史出现过旧工具就重新授权它。切换工具集合后会重新建立与新提示对应的状态，不复用不匹配的KV。

## 长任务与中途补充输入（2026-10-21）

Chat Completions 文本和工具历史共用 `MAX_MESSAGES=65536` 对象数保护，不再使用原来的129条上限。这个数不是可用上下文保证：HTTP请求字节数、模板长度和所选模型token容量仍各自检查；服务端不会截断历史、删除旧消息或丢弃工具结果。文本检查点接受请求加最终assistant的完整会话，避免聊天可继续而保存仍被旧上限拒绝。

新版服务允许在**完整工具结果组之后**直接追加user补充要求；缺少或重复tool_call_id、未完成的并行调用组仍拒绝。推荐provider设 `requiresAssistantAfterToolResult=false`，不再为补充输入虚构 `I have processed the tool results.`。旧聊天里已经存在的兼容assistant记录不主动删除。切换该兼容项会改变旧会话的序列化历史，首个请求可能需要一次完整历史重建；不能把这次迁移成本误称为每次steer都无需prefill。

**部署顺序：先更新服务端，再更新DSH配置。** 使用旧服务端时保持原来的true，否则它会拒绝tool后直接跟user。新版 `/cq/capabilities` 应返回 `max_messages=65536` 与 `user_steering_after_tool_results=true` 后，才将实际所用provider（本机可能叫quasar-local，而非augur-local）的兼容项改为false。不要为改这一个选项覆盖API地址、凭据、默认模型或用户预设。DSH配置加载后方可生效；不要强制终止进行中的工具操作。

DSH的steer在下一步安全点接收补充输入，不等同于Stop或取消当前工具。在已检查的本机会话里，补充输入留在turn 3，推理服务连续48次使用exact_history_append；这不支持“每次补充都重新加载模型”的判断。模型收到新要求后重新组织思路仍可能发生，删除兼容assistant并不等于已经证实彻底消除了该行为。没有改采样、thinking档位、系统提示或已完成工具记录。

回归覆盖长文本、长工具历史、工具组完整性、HTTP/SSE、文本检查点和steering后的native会话/token复用。运行 `python -m unittest discover -s scripts -p test_qwen35_long_history.py -v`，并结合现有API、工具、持久化、Augur与Vision测试。模型GPU实测、DSH交互端到端与新的冻结发布包验收仍是不同阶段，不能用CPU测试替代。


### 本机最小桥接部署验收（2026-10-21）

已使用旧安装的原生二进制/运行库和最小 Python overlay 完成真实 GPU 验证：131/133 条文本历史成功，续接为 `exact_history_append`；生成真实工具调用后直接追加工具结果及 user 补充，在 `tool_choice=auto` 不变时继续复用；176 条/44 组工具历史的 SSE 正常结束。原冻结安装、数值策略、模型/缓存配置不变。原安装版本与仓库 HEAD 有其他差异，部署回归使用版本匹配的旧测试加本次新增测试，不将不适用的 HEAD 测试冒充通过。

实际 `quasar-local` profile 已仅修改上述兼容项。DSH 默认 profile HMR 路径存在，桌面和后台工具未重启；本次没有独立确认进程内配置或原 DSH 会话交互端到端。旧会话首次迁移可触发完整 prefill。使用本机新增的“Augur（长对话修复版）”入口，原安装入口仍为旧版本；回退到旧服务之前应先恢复兼容项为 true。公开验证范围见 [发布验收说明](../../docs/CHAT_CONTINUITY_RELEASE_20261021.md)；私有 profile、密钥和原始会话回执不分发。
