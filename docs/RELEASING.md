# Prepare and publish Augur

This guide separates an editable repository from frozen source and Windows artifacts. The repository name is **Augur**. Public model support, defaults and recorded benchmarks are described in the [README](../README.md) and [model cards](../model-cards/README.md).

## Editable repository

Use the prepared clean repository directory for the first public import. It contains the selected source, documentation, repository templates and third-party notices, without development Git history. Do not upload an entire research checkout or its hidden directories. The exact selection is maintained in [prepare_augur_release.py](../scripts/prepare_augur_release.py); only named GitHub/dotfiles are admitted.

Generated archive/build manifests are not source files for ongoing development. They are omitted from the editable directory and recreated when freezing a release. Keep the dependency Cargo.lock files. A source archive includes its immutable manifest so it can be verified and built without Git.

Install the dependencies in container/augur/requirements.in and Node.js 22+, then run:

```powershell
python -X utf8 -B scripts/check_augur_source.py
```

This runs shipped Python checks, browser chat regressions, DSH queue tests and JavaScript syntax checks. No GPU, model, running service or DSH installation is needed. The workflow uses Windows, Python 3.13 and immutable GitHub Action commits. The full recovery integration test additionally requires the supported DSH installation; see [DSH integration](../integrations/dsh/README.md). Native build/CPU/GPU verification is described in [BUILDING](../BUILDING.md).

## Frozen source and Windows package

1. Freeze a new selected source directory and build the Windows Vision-capable feature as documented in BUILDING. Preserve the source identity and native build/CPU receipts.
2. Prepare and verify the runtime and third-party notice bundle using the scripts in the [Windows source guide](AUGUR_WINDOWS_SOURCE_GUIDE.md). Licenses/lockfiles must match the exact frozen inputs.
3. Package the Windows release with package_augur_windows.py and export source with export_augur_windows_source.py. Test from extracted artifacts, not only the working repository.
4. Include ZIP and tar.gz variants with identical per-member bytes. Publish SHA256SUMS.txt and a validation summary beside them. Check names/size/type, reject traversal and duplicate members, and verify all manifest hashes.
5. Preserve validation scope: CPU success, GPU function, measured performance and deployment on another Windows machine are distinct results. If reusing native executables for a documentation/Python-only release, prove native-source equality, bind their hashes and record why the reuse is valid. Do not call that a native rebuild or a new performance benchmark.

## Public-upload checklist

- Import only the clean tree; retain MIT/Apache license files, NOTICE, vendor exceptions and THIRD_PARTY records. Model weights and the NVIDIA driver are separate inputs.
- Inspect staged files and archive inventories for weights, checkpoints, API keys, setup capability URLs, private sessions, user-data and raw failure payloads. .gitignore is convenience; the export allowlist and review remain required.
- Preserve the [unsafe trust-boundary notes](UNSAFE_TRUST_BOUNDARIES.md) and open audit items. Configure a real private vulnerability-reporting route on the hosting platform; no maintainer address is invented in this snapshot.
- Run the hosted CPU workflow after creating the repository. Local checks do not establish that a remote workflow has executed.
- For Windows delivery, run Verify-Package.cmd and qualify startup, text, images, tool roundtrips, cancellation and restart on the intended hardware. The recorded development host is not an independent clean-machine installation.
- Keep historical performance tables tied to their original identities and test conditions. No generic score is implied by a short-task or peak result.
- Use an explicit release tag and attach the four archives, checksums and release notes. Public upload, code signing and signing keys are not part of local archive preparation.

## User entry points

For Windows users, start with [the Chinese setup guide](AUGUR_WINDOWS_USER_GUIDE.zh-CN.md): verify/extract the entire package, double-click 启动服务.exe, choose model/draft paths, quantize when needed, then save and start. The package supplies Python and the CUDA runtime; it does not supply weights or the NVIDIA driver.

The guide also explains 256K persistent-cache disk budgeting and why multiple chat windows do not map one-to-one to cache entries. The monitor shows native prefill TPS, decode TPS and context/compaction progress.

Recent fixes are in the [changelog](../CHANGELOG.md). Recorded pre-release acceptance is in the [public receipt](public/AUGUR_RELEASE_ACCEPTANCE_20261015.json). Distribution-level VALIDATION.json records checks for each specific archive set; it is generated alongside the final files.
