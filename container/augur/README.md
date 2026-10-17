# Experimental QUASAR SM70 Image

Default main model: `quasar-27b`. This recipe keeps Vision enabled, streamed
original BF16 head/embedding, B4352 prefill, physical262144 Q8 KV and tile8 B1
attention. Draft weights are authenticated and resident but **DFlash is inactive**.
`product_ready=false` and `numerical_qualified=false` remain explicit.

HTTP/Vision defaults to lossless direct checkpoint-to-QPN GPU layout packing.
`AUGUR_QPN_STARTUP=ggml` selects the CPU-layout control/rollback; `direct`
selects the default explicitly. This changes weight loading, not inference
arithmetic or model identity. Windows first-image ABBA receipts are separate
from the still-unverified Linux/container startup contract.

The source context contains no model/draft weights or private admission files.
Linux compilation, installed image GPU smoke and registry publication are not
yet verified. A source preparation receipt is not a successful image receipt.
The tested target is a32GiB SM70 GPU; other hardware profiles are not qualified.

## Prepare And Build

Run these commands from the source repository on a machine with Python3.12+,
Git and Cargo. Choose fresh output paths. Preparation downloads only Linux
CPython3.12/universal Python wheels from PyPI. It captures allowlisted dirty
source without resetting the working tree, preserving native licenses and the
local Lucide script/license. Every source file and wheel is hashed.

```sh
python -B scripts/prepare_augur_release.py --out /work/augur-context --download-wheels
python -B scripts/resolve_augur_image_bases.py --out /work/augur-bases
python -B scripts/verify_augur_release.py --context /work/augur-context \
  --base-lock /work/augur-bases --node /usr/bin/node \
  --out /work/augur-release-checks
python -B scripts/build_augur_image.py --context /work/augur-context \
  --base-lock /work/augur-bases --tag augur:sm70 \
  --out /work/augur-image-build
```

The builder requires Docker and fixes `linux/amd64`. The base lock binds Rust
1.96.0-slim-bookworm and CUDA12.8.1-devel/runtime-ubuntu24.04 to exact Registry
index bytes, amd64 manifests and config bytes. Tokens are not written to evidence.
This checks downloaded metadata identity, not a publisher signature or image
runtime. Python installation is offline/hash-locked and runs `pip check`.
Native compilation and CPU tests must pass before the final image is produced.
The build also collects authenticated Cargo, locked-wheel and native notice
texts into `/opt/augur/notices`, which the runtime stage retains. The collector
uses checksum-matching `.crate` archives, not unverified registry source folders.
APT packages are not snapshot-pinned: do not claim bitwise reproducible builds.

For a later source snapshot, use `--wheel-context /work/augur-context` instead
of `--download-wheels`. This validates the earlier complete context and copies
the same wheel bytes/lock with provenance; it does not resolve new dependencies.
The verifier records host-Python/Web regressions, locked Cargo metadata, wheel
license records and native notices. Host tests are not installation tests of
those Linux wheels, and a dependency inventory is not complete license clearance.

## External Inputs

Use legally obtained QUASAR config/tokenizer/template/generation files, index
and all five original safetensors shards in a read-only directory. Supply a
trusted `quasar-session-model-manifest-v1` manifest and its externally known
SHA256. Do not hash an unreviewed replacement and treat it as trusted admission.

Mount legally obtained draft config/weights at `/draft`. The read-only
`draft-environment.json` must contain these four fields (directly or under
`environment`); hashes are the admitted source hashes, not invented placeholders:

```json
{
  "AUGUR_DRAFT_CONFIG": "/draft/config-f16.json",
  "AUGUR_DRAFT_CONFIG_SHA256": "<admitted-config-sha256>",
  "AUGUR_DRAFT_PATH": "/draft/qwen35-27b-dflash2-q8_0.gguf",
  "AUGUR_DRAFT_SHA256": "<admitted-weights-sha256>"
}
```

Windows absolute draft paths do not work inside Linux. Change only the mount
paths, preserving the verified files/hashes. Never include API keys in this JSON.
The input directory must contain `model-manifest.json` and
`draft-environment.json`. Runtime UID10001 needs read access to all inputs and
write access to a separate evidence directory. Do not chmod model files writable
to make startup pass. Model acquisition/license clearance is not supplied by
this recipe, and source licenses do not relicense weights or CUDA.

## Linux GPU Smoke

Use a Linux Docker/NVIDIA Container Toolkit host with exclusive ownership of the
qualified GPU. This recipe binds loopback inside the container. On Linux,
`--network host` preserves local access without making the server public;
ordinary `-p` forwarding does not expose a loopback-only container listener.
Do not silently change the application bind to `0.0.0.0`.

```sh
docker run --rm --stop-timeout 300 --gpus '"device=0"' --network host \
  --mount type=bind,src=/srv/augur/model,dst=/model,readonly \
  --mount type=bind,src=/srv/augur/draft,dst=/draft,readonly \
  --mount type=bind,src=/srv/augur/inputs,dst=/inputs,readonly \
  --mount type=bind,src=/srv/augur/evidence,dst=/evidence \
  --env AUGUR_MANIFEST_SHA256="$AUGUR_MANIFEST_SHA256" \
  --env AUGUR_PORT=8294 augur:sm70 --serve-seconds 120
```

Use the private `browser_url` emitted to stdout for the local browser. Its
fragment is a bearer capability; do not commit/log/share it as release evidence.
The image does not execute model-generated arbitrary shell commands.

The product launcher defaults to request-bound cancellation discard for supported
decode cancellations. It verifies the live native capability after the startup
smoke before publishing readiness. Cancelled conversation history is discarded;
the next request opens a fresh session. Unsupported prefill/encoder cancellation,
compute errors and failed drain/reset/receipt validation still retire the owner.
Use `--env AUGUR_CANCEL_RECOVERY=retire` for the explicit reload-on-Stop rollback.
Standalone native examples retain their opt-in contract. No DFlash activation or
numerical qualification is implied by this latency policy.

Before calling the image usable, retain the immutable image ID/build receipt,
real text and held-out PNG/JPEG answers, actual tool execution+continuation,
HTTP/SSE streaming, sustained Stop/recovery and a clean bounded shutdown. Verify
`stopped.json`: exit0, unforced retirement, empty cleanup errors and every native
closure with zero live allocations/bytes. Test occupied port, bad manifest/hash,
missing input and malformed/oversized image failure paths. Measure process-to-
first-text and first-image responses separately; host-ready Vision is not an
executed encoder. Windows timings must not be relabeled as container timings.

The launcher handles SIGTERM/SIGINT by recording the request, cancelling active
generation, stopping the public server from a different thread and writing the
native shutdown receipts. Repeated signals do not interrupt resource teardown.
The grace period covers startup cancellation and bounded native drains; Docker's
short default stop grace is not the acceptance contract. Test `docker stop` during
native startup and active SSE as well as after readiness on the Linux host.
An interrupted startup must not publish `ready.json`. If no native owner ever
spawned, cleanup requires an explicit adapter `no_owner` response and no owner
directories; missing allocation receipts alone never qualify cleanup. Windows
in-process signal injection is retained separately and is not Docker evidence.

## Licenses And Release Boundary

Keep root `LICENSE-APACHE`, `LICENSE-MIT`, `NOTICE`, native llama/ggml MIT license,
NInfer Apache2.0 notice/license and Lucide/Feather ISC notices. Python wheels
retain their embedded license records. Audit Cargo transitive licenses, CUDA/
base OS redistribution conditions and model/draft rights before publication.
`dependency-inventory.json` from the verifier binds locked registry checksums,
declared Cargo/Python license information and embedded notice hashes. CUDA/base OS
and model/draft rights remain separate acceptance items.
The standalone collector can prepare and revalidate the exact notice bundle on
the host before a Linux build:

```sh
python -B scripts/prepare_augur_notices.py --context /work/augur-context \
  --out /work/augur-notices
python -B scripts/prepare_augur_notices.py --context /work/augur-context \
  --out /work/augur-notices --verify-only
```

Each Cargo archive is authenticated against the original lock checksum before
reading; unsafe archive paths/links, duplicate paths, size overruns and byte drift
are rejected without extracting code. Source context and wheel hashes are checked
before and after collection. The resolved Cargo workspace is broader than a
proven linked-binary dependency set. Missing embedded wheel notices are listed,
never silently replaced with the license of a same-named Cargo package. The
currently locked tokenizers0.22.2 wheel has no embedded notice, although the
separately locked Rust tokenizers0.22.2 crate does contain a retained LICENSE.
Native dependencies inside wheels need separate review; this bundle does not
declare license clearance or redistribute model/draft weights.
No image, weights or registry publication has been qualified by this document.
