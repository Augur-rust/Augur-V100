---
name: Bug report / 问题反馈
about: Report a reproducible Augur engine, launcher or client issue
title: ''
labels: ''
assignees: ''
---

## What happened / 现象

Describe the result and what you expected. Include minimal synthetic prompts only.

## Reproduce / 复现步骤

1. Package date/revision and archive SHA256:
2. Windows, GPU/VRAM and driver versions:
3. Model identity, KV capacity, Vision mode, fixed-B8/native mode:
4. Minimal steps and HTTP status/error code:

## Evidence / 证据

Paste a short redacted error or attach a sanitized receipt. For performance issues, separate native prefill/decode TPS from HTTP TTFT and throughput; state input/output token counts and cache state.

Remove API keys, setup URLs containing capabilities, personal paths, conversations and model weights. Never attach entire user-data or session directories. Follow the [security policy](../../SECURITY.md) for security reports.
