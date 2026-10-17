# Augur Windows 源码使用与验收边界

仓库名称为 `Augur`；现有 `augur-*` 工具、`augur-*` crate 前缀及模型标识保持不变。

## 环境

- Windows x64、MSVC C++、`rust-toolchain.toml` 指定的 Rust、CUDA 12.8 / SM70 构建环境。
- Python **3.11 或更高**。本机产品回归环境为 Python 3.13.9；Python 3.10 缺少此项目使用的 `tomllib` 和 `hashlib.file_digest`。
- 安装 `container/augur/requirements.in` 中的依赖；这是依赖版本清单，不是完整 Windows wheel 哈希锁。
- 自行提供已准入的 QUASAR 模型目录、模型 manifest 及可信 SHA256、Rust 量化的 Q8_0＋内置 FP32 gate0.45 GGUF 及对应 DFlash 环境记录。不能凭一个新计算的摘要绕过权重身份准入。

本源码候选不附带开发工具、模型或二进制。只对现有 Windows / 32GiB SM70 环境有本地历史证据，其他设备尚未验证。

## 独立构建

在源码目录执行，输出目录须是新的独立目录；不要覆盖旧构建或证据。`AUGUR_VCVARS64` 可指定实际的 Visual Studio 环境脚本。

```powershell
$env:AUGUR_CUDA_GENCODE = 'arch=compute_70,code=sm_70'
python -B scripts/build_augur_windows.py --stage build --feature augur-model-qwen35-runtime/qwen35-augur-vision --out C:/Quasar/build-01 --target-dir C:/Quasar/cargo-target-01
python -B scripts/build_augur_windows.py --stage cpu --out C:/Quasar/build-01
```

构建工具从 `augur-release-source.json` 校验逐文件 SHA，因此解压后的源码可以不带 Git 元数据。新构建必须重新取得相应 CPU/GPU/产品回执，不能拷贝历史回执冒用。

## 固定 B8 启动

```powershell
python -B scripts/run_augur_windows.py --build C:/Quasar/build-01 --model-dir C:/Models/quasar --manifest C:/Quasar/inputs/model-manifest.json --manifest-sha256 YOUR_ADMITTED_SHA256 --draft-environment C:/Quasar/inputs/draft-environment.json --live-file C:/Quasar/private/live.json --evidence C:/Quasar/runs/new-run --port 8294
```

所有占位路径需替换为实际已准入输入。仅监听 loopback；stdout 的 `browser_url` 含本次 bearer capability，不能公开。需要 DSH 固定本地 key 时加 `--api-key-file <PRIVATE_KEY_FILE>`；不要把 key 写进源码、截图或测试日志。Rust 加载器校验 GGUF 内 gate 的 256×5120 FP32 形状、校准元数据及固定 SHA256，默认对全部 DFlash 文本生成使用 gate0.45；运行时 scale=1。包含英文／中文 thinking、代码和工具调用；vision 仍走原生 target 保护路径。旧 Q8 GGUF 可通过 `--selector-gate-file` 显式使用已验收的外置 gate，但不能对内置版叠加覆盖。

## Vision 显存策略

默认 `--vision-residency on_demand`：每次图片编码后释放视觉编码器的 GPU 权重与执行计划，CPU 权重继续保留。可显式选择 `--vision-residency auto`：启动后保留视觉编码器，图片临时工作区用完释放；长 prefill 显存不足时先回收 Vision，下次图片请求恢复。两者均不降低图片分辨率或计算精度。未传选项或遗留实验环境均不会自动启用常驻。

本机 32 GiB、完整 262144-token KV、B4352、DFlash B8 条件下，8K／32K／64K 全量 prefill 与恢复测试通过。保留 B8 验证计划时最低采样空闲仅约 133 MiB，不宜再让其他 GPU 程序抢占显存。详见 [Vision 显存验收](AUGUR_VISION_RESIDENCY_AND_LONG_PREFILL_20261015.md)。

## 只读监控

```powershell
python -B scripts/augur_monitor.py --runs-root C:/Quasar/runs --live-file C:/Quasar/private/live.json --engine-port 8294 --port 8295
```

监控读取现有回执并采样 GPU；不会加载模型。启动器提供 `--live-file` 安装 token observer；路径须与监控的 `--live-file` 一致。Prefill TPS 以实际新计算 token 除以原生 prefill 墙钟；缓存 token 不计入分子。没有新 prefill 时显示状态复用，不制造无限 TPS。没有 observer、文件失效或引擎离线时，实时 TPS 为空；最近完成请求的 DFlash 分母账单仍可显示。observer 自身的开销须另行同构建比较，不能默认当作零。

## Windows 二进制候选包

完成冻结构建和 CPU 测试后，可以生成可搬移的二进制候选：

```powershell
python -B scripts/package_augur_windows.py --build C:/Quasar/build-01 --out C:/Quasar/windows-candidate --zip C:/Quasar/windows-candidate.zip
```

包内保留原始 build/CPU/source 回执，`portable-build.json` 只将已绑定的二进制 SHA256 映射到包内 `bin/`，不会改写原回执或伪造验收。解压到不同路径后，使用 `Start-Augur-Engine.cmd` 并传入上述模型、manifest、draft、evidence 等参数（无需再次指定 `--build`）。

### 带运行库的便携候选

发布打包增加 `--runtime <VERIFIED_RUNTIME>` 与 `--notices <VERIFIED_NOTICES>` 后，包内包含 CPython 3.13.16、完整锁定 wheel 闭包、CUDA 12.8.1 cudart/cuBLAS 和 Microsoft VC143 CRT。用户不必安装 Rust、MSVC 编译器、Python、Conda 或 CUDA Toolkit；仍须 Windows x64、兼容的 NVIDIA 驱动与 SM70 GPU，并自行提供准入模型。

1. 解压完整 ZIP（不能只复制 exe），可放在含空格／中文的本地目录。
2. 执行 `Verify-Package.cmd`，校验包内文件和 Python/native 依赖导入。
3. 执行 `Start-Augur-Engine.cmd --help`，然后传入上文模型、manifest、draft 环境和新的 evidence 目录。不要再次传 `--build`。
4. 监控使用 `Start-Monitor.cmd --runs-root RUNS --live-file PRIVATE_LIVE_JSON --engine-port 8294 --port 8295`；引擎须传相同 `--live-file`。
5. 停止时 Ctrl+C，等待正常退场。evidence 保存回执；实时文件和 capability URL 不要公开。

便携入口固定使用 `runtime/python/python.exe`，不读取 `AUGUR_PYTHON`，清除继承的 Python/CUDA 开发路径，以包内 native 和 Windows 系统目录构造 PATH。源码入口仍支持自行安装的依赖。默认 Vision 按需加载，显式 `--vision-residency auto` 才启用回收式常驻。

`THIRD_PARTY/dependency-inventory.json` 和 `runtime/runtime-manifest.json` 列出两份 Cargo.lock、Python、CUDA、MSVC 的版本／来源／SHA256 与许可证。`windows-distribution.json` 绑定包内文件；最终 ZIP 摘要另列 `CHECKSUMS.sha256`。这提供内容校验，不等同于代码签名或发布者身份认证。

开发机上的隔离 PATH 与搬移测试不能替代真正的干净 Windows 机器验收；其他 GPU 架构不在本 SM70 候选的验收范围。未传 `--runtime` 的旧开发候选仍需系统 Python/CUDA 运行库。模型、驱动、密钥和聊天数据不分发。

## DSH 与上下文

见 `integrations/dsh/README.md`。推荐模型 `quasar-27b`，思考档位 off/low/high/xhigh；开启思考后的默认语言为英文，语言和质量须由实际输出验证。输出上限 249036 是上限而非每次可用预算。物理 KV 为 262144，prompt、历史和本次生成必须共同装入。客户端压缩应在下一轮请求前完成；不会在一个仍在流式生成的回答中无损重写 target 状态。

## 必须分开记录的验收

1. 冻结源码、二进制、原 gate 和模型身份；不把未准入 rank4 打入默认配置。
2. 真实代码执行、工具调用/结果闭环、vision/OCR、多轮图片、取消后恢复、EOF 与错误恢复。
3. 长思考是否自然结束、有无最终答案；长度截断不是通过。
4. 冷/热缓存 TTFT、完整 HTTP EOF、HTTP 收尾、native decode TPS 分开。接受率同时列 accepted/proposed 与 accepted/checked。
5. 启动时间和退场资源释放；干净 Windows 环境从解压源码完成构建和部署。
6. 许可证/依赖完整性、秘密与私人数据扫描、源包 SHA256，然后才可宣称正式发布。

Linux 镜像不在本轮范围。单元测试通过只表示对应代码边界通过，不是上述 GPU/语义验收的替代品。

## Rust 原生 DFlash 量化

Windows 构建工具同时输出 `augur-quantize.exe`，二进制包位于 `bin/`；无需 Python/llama.cpp 转换权重。

```powershell
bin/augur-quantize.exe dflash-plan C:/Models/quasar/z-lab-dflash C:/Models/quasar --weights-sha256 ORIGINAL_SAFETENSORS_SHA256
bin/augur-quantize.exe dflash-convert C:/Models/quasar/z-lab-dflash C:/Models/quasar C:/Models/quasar/dflash-q8-fp32-gate045 --weights-sha256 ORIGINAL_SAFETENSORS_SHA256
```

输出目录必须尚不存在，父目录已存在。得到 `model-q8_0.gguf`、不变的 `config.json` 和转换回执。新建 draft-environment JSON，以转换回执的输出路径／SHA256、config 路径／SHA256 填入 `AUGUR_DRAFT_PATH`、`AUGUR_DRAFT_SHA256`、`AUGUR_DRAFT_CONFIG`、`AUGUR_DRAFT_CONFIG_SHA256`；不要改写旧准入记录。gate0.45 只折入一次，启动器固定运行时 scale=1。主模型与草稿权重不随源码或二进制包分发。
