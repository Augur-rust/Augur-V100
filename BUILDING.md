# Build Augur

Current target: [qwen3.8-27b-quasar-nvfp4](model-cards/qwen3.8-27b-quasar-nvfp4/README.md). This is the current Windows source-build entry point; model weights are not needed to compile or run non-GPU tests.

## Prerequisites

- Windows x64, MSVC C++ tools and the Rust toolchain pinned in `rust-toolchain.toml`.
- CUDA 12.8 with the SM70 build path; compatible NVIDIA hardware/driver for subsequent GPU execution.
- Python 3.11+ and the product Python dependencies listed in `container/augur/requirements.in` for the builder/tests. The verified portable package carries its own runtime; source-build dependencies are separate.
- A trusted local toolchain and pinned Cargo dependencies. Source preparation is not supply-chain attestation or a claim of bitwise-reproducible builds.

## Freeze, build, test

From a development Git checkout, freeze a new public source context. Choose fresh absolute paths outside the checkout; create their parent directory first. Do not overwrite previous builds or receipts.

```powershell
python -B scripts/prepare_augur_release.py --out C:/Augur/source-01
```

From that frozen source directory (or the root of a verified source archive), build the selected Vision-capable product and run its CPU checks:

```powershell
$env:AUGUR_VCVARS64 = 'C:/Your/VisualStudio/VC/Auxiliary/Build/vcvars64.bat'
$env:CUDA_PATH = 'C:/Your/CUDA'
$env:AUGUR_CUDA_GENCODE = 'arch=compute_70,code=sm_70'
python -B scripts/build_augur_windows.py --stage build --feature augur-model-qwen35-runtime/qwen35-augur-vision --out C:/Augur/build-01 --target-dir C:/Augur/cargo-target-01
if ($LASTEXITCODE -ne 0) { throw 'Native build failed' }
python -B scripts/build_augur_windows.py --stage cpu --feature augur-model-qwen35-runtime/qwen35-augur-vision --out C:/Augur/build-01
if ($LASTEXITCODE -ne 0) { throw 'CPU checks failed' }
```

Replace toolchain placeholders with actual installed locations. The source manifest binds file identities; an exported source archive does not need Git metadata to build. A fresh development checkout should freeze first instead of reusing a stale manifest.

Public CPU checks run from the selected source directory with those Python dependencies and Node.js 22+ available:

```powershell
python -X utf8 -B scripts/check_augur_source.py
```

This command runs the shipped Python suite, browser chat and DSH queue regressions, and launcher/monitor JavaScript syntax checks. It does not load weights or start inference. The same command is configured in the GitHub workflow. For archive preparation and the initial public import, see [RELEASING](docs/RELEASING.md).

## Run and qualify

Follow the [Windows guide](docs/AUGUR_WINDOWS_SOURCE_GUIDE.md) for admitted inputs, authentication, launch, monitoring and portable packaging. The descriptive model-card manifest is **not** the loader's `--manifest` input.

CPU tests and successful linking do not qualify GPU inference, quality, TPS or clean-machine deployment. GPU measurements must use matching source/build/model identities. Do not attach old performance receipts to a newly renamed or recompiled binary. Keep weights, keys, private sessions and raw run receipts out of source archives.

See [CONTRIBUTING](CONTRIBUTING.md), [repository layout](docs/REPOSITORY_LAYOUT.md), and [SECURITY](SECURITY.md).
