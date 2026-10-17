# Security and trust boundaries / 安全与信任边界

QUASAR is a local, single-owner GPU inference engine, **not a sandbox for arbitrary native code or a formally verified safe Rust library**. The selected product entry point is narrower than the set of exported low-level and experimental APIs.

## Read before embedding or exposing the engine

- [Unsafe/CUDA trust-boundary contract and open gaps](docs/UNSAFE_TRUST_BOUNDARIES.md)
- [Windows deployment guide](docs/AUGUR_WINDOWS_SOURCE_GUIDE.md)
- [Runtime dependencies and provenance](docs/public/RUNTIME_DEPENDENCIES.md)

The documented low-level API gaps include non-owning device views, captured graph resources, and CPU/GPU access to mapped host memory. **A `SAFETY` comment does not make a safe API with unenforced memory-safety preconditions sound.** The current controlled owner path and its tests are not a general library-wide memory-safety proof.

## Deployment assumptions

1. Use the loopback-only authenticated product launcher. Do not expose the native adapter, producer, or read-only monitor directly to an untrusted network. The monitor is not a multi-tenant authenticated service; even counters and request timing can reveal workload information.
2. Treat HTTP messages, tool results and images as untrusted inputs: retain existing limits, validation and error handling. A bearer key is authorization to use the service, not proof that payloads are safe.
3. Trust the publisher and provenance of executable code, CUDA/driver/runtime DLLs, model templates, manifests and local checkpoint inputs. Keep their directories protected from concurrent replacement. Matching a supplied SHA256 establishes byte identity, **not** who supplied it or that those bytes are harmless.
4. Keep API keys, browser capability URLs, conversations, checkpoints and raw request receipts private. They do not belong in an open-source/release archive. Model weights are not included in the distribution.
5. Generated tool calls are data for the client to authorize and execute. This engine is not an authorization boundary for an external agent's filesystem, shell or network privileges.
6. Successful kernel submission, a memory-usage counter, or `Drop` running does not prove completion. Do not suppress cancellation, poisoned-owner, failed-drain or shutdown-verification errors.

## Scope and reporting

This documentation pass identifies boundaries and outstanding audit items; it is not a complete security audit, a clean-machine certificate, or proof of universal CUDA memory safety. Native/CUDA dependencies remain part of the trusted computing base. See the linked contract for what is checked in code and what is not encoded in types.

No dedicated security-reporting contact is configured in this source snapshot. Before a public launch, maintainers should publish a private reporting route. Do not post credentials, private prompts, model data, or an immediately usable exploit in public issues; contact the repository's maintainers privately when a channel is available.

中文：本项目的 `unsafe` 边界不能仅写成“为了性能”。应逐项说明资源 owner、范围、别名、设备、异步完成与异常退场条件；尚未由类型或检查表达的前提必须保留为待修复项，不能用注释宣称已安全。详见上面的专项说明。
