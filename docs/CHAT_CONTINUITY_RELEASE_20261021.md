# Chat continuity release validation — 2026-10-21

## Scope

The text API, tool-history normalization and completed text checkpoints now share a 65,536-message object-count safety bound. Request bytes, template length and the model token budget remain independently bounded. No history is silently removed.

A complete tool-result group can be followed by a real user supplement, without a synthetic assistant acknowledgement. Missing/duplicate tool IDs and incomplete parallel groups still fail closed. Capabilities advertise `max_messages` and `user_steering_after_tool_results`.

Deploy the backend before changing the active DSH provider to `requiresAssistantAfterToolResult: false`. The active provider may have a different name from the repository example. Preserve credentials, model selection and unrelated settings. Existing conversations may require one complete prefill after serialization migration. A changed tool policy can also intentionally invalidate reuse.

## Recorded checks

- Development CPU suite: 195 tests, 193 passed and 2 skipped.
- Installed-version minimal bridge regression suite: 193 tests, 191 passed and 2 skipped, including nine long-history tests and the Augur tool-boundary continuation regression.
- Live V100 native inference: 131-message text request and 133-message continuation both returned HTTP 200; continuation used `exact_history_append` and reused 1,633 tokens.
- Native tool generation, completed tool result, then direct user supplement: returned `OK`, used `exact_history_append`, and reused 329 tokens with the same `tool_choice=auto` policy.
- Live SSE with 176 messages and 44 completed tool calls: HTTP 200, `finish_reason=stop`, usage and `[DONE]` received.
- Old owner shut down cooperatively with verified cleanup before the new bridge started. Original native binaries, frozen source/build receipts, numerical policy, model/cache settings and API key remained unchanged.

These are protocol and inference smoke tests, not a new throughput benchmark, long-context quality evaluation or independent clean-machine qualification. The tested Windows bridge update uses existing native binaries; it does not execute the additional Nerkyor research source included in the repository.

## Explicit limitations

The running DSH profile file was updated without stopping an existing background tool. The installed profile watcher supports reload, but the in-memory compat value and a user interaction in the original DSH conversation were not independently observed. Do not claim end-to-end DSH acceptance from these backend checks alone. Model replanning is not necessarily a process restart and is not eliminated by this protocol fix.

The full current-development test suite is not interchangeable with the older installed-version suite: newer native error-frame and nonzero seed checks target different interfaces. Version-matched tests were used for the minimal bridge update, with mismatched attempts retained privately rather than reported as passes.

The optional Nerkyor source-pair/no-DFlash path is research-only: 4,096-token F32 activation/KV, greedy selection, no MTP or Vision, no DFlash allocation. Historical numerical/performance evidence stays bound to its original builds; it is not re-qualified by publishing this release.

## Publication validation (2026-10-21)

The selected public Python/browser/queue checks are rerun on the clean publication tree. Native CPU checks use `--lib --no-default-features --features augur-model-qwen35-runtime/qwen35-session-checkpoint` and an isolated writable TEMP directory. The model crate passed; the runtime suite reported 306 passed, one ignored and three Windows symlink-negative tests blocked by missing symlink privilege (error 1314). Those three are not counted as passes. An earlier feature-less invocation and the default TEMP permissions were unsuitable; their failures are not hidden by a blanket green native-test claim. No current native GPU rebuild or benchmark is part of this bridge release.
