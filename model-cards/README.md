# Supported models / 支持模型

Augur currently supports **one target model**. Each supported target has its own card directory and a descriptive artifact manifest. Historical experiments and source module names do not expand this list.

当前只支持以下一个 target；DFlash 是配套草稿加速器，不是第二个独立支持的主模型。

| Target | Product API alias | Model card | Artifact identity |
|---|---|---|---|
| `qwen3.8-27b-quasar-nvfp4` | `quasar-27b` | [English](qwen3.8-27b-quasar-nvfp4/README.md) · [中文](qwen3.8-27b-quasar-nvfp4/README.zh-CN.md) | [Manifest](qwen3.8-27b-quasar-nvfp4/artifact-manifest.json) |

These directories contain documentation and recorded identities only, **not model weights**. The card manifest is not the runtime `--manifest` input or a replacement for runtime hash admission. The engine loads the original target safetensors; it does not load NInfer `.ninfer` artifacts.

See [build and run](../BUILDING.md), [documentation](../docs/README.md), and [model provenance](../docs/public/AUGUR_MODEL_SOURCES.md).
