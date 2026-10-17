## Problem and behavior

Describe the trigger, old behavior and resulting behavior.

## Validation

List commands and actual results; distinguish CPU checks, GPU tests and measurements that were not run. For numerical/performance changes, include model/build identities, prompt/output counts, cache state, acceptance denominators and regressions.

## Review notes

- Keep model weights, keys, private sessions and raw user logs out of the diff.
- Preserve vendor licenses and update affected public documentation/model cards.
- For unsafe/CUDA changes, explain ownership, bounds, stream completion and error cleanup; consult [trust boundaries](../docs/UNSAFE_TRUST_BOUNDARIES.md).
