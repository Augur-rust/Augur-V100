# Windows runtime dependency inventory

Updated: October 15, 2026.

The Windows RC packaging path accepts a separately verified runtime (`--runtime`) and dependency notices (`--notices`). It does not require system Python, Conda, Rust/MSVC build tools or a CUDA Toolkit installation at runtime. Windows x64/UCRT, a compatible NVIDIA driver/SM70 GPU and separately admitted model weights are still required. A development-host isolated-PATH smoke is not a clean-machine certification.

| Component | Locked selection / source |
|---|---|
| Native owner / quantizer | Frozen public source, original build/CPU receipts, package-local SHA256 mapping |
| Python | Official CPython **3.13.16** embeddable AMD64, explicit `_pth` isolation; no pip or user site in package |
| Python packages | tokenizers 0.22.2, Jinja2 3.1.6, Pillow 12.3.0, jsonschema 4.25.0, plus full Windows cp313 transitive closure in `requirements-windows.lock` |
| CUDA runtime | Official CUDA 12.8.1 redistribution metadata: cudart **12.8.90**, cuBLAS/cuBLASLt **12.8.4.1**; only runtime DLLs, no driver/SDK |
| Microsoft CRT | Signed DLLs from Visual Studio 2022 VC143 x64 REDIST **14.44.35112**; same recorded vcruntime beside Python, in native runtime, and app-local CRT beside native EXEs |
| Cargo | Windows workspace and standalone Rust quantizer Cargo.lock; original registry archives verified by checksum |
| Web | Packaged HTML/CSS/JS and the retained Lucide license; no required CDN |
| Driver / models | Not redistributed; original model identities remain separate from source/runtime licensing |

## Reproducible packaging inputs

`prepare_augur_windows_runtime.py` authenticates the selected upstream archive SHA256, each exact wheel against the official PyPI release record, and the installed Microsoft REDIST signatures. It rejects unexpected paths and writes a complete per-file manifest. Native EXEs have app-local signed CRT copies so Windows does not select an unrelated System32 version before PATH. CPython's bundled vcruntime is replaced by the signed selected REDIST version and recorded explicitly. Authenticode was checked for Microsoft CRT files, not claimed for unsigned CUDA DLLs or the engine executables.

`prepare_augur_windows_notices.py` collects both Cargo lockfile license sets, the exact Python wheel notices, native source attributions and vendor distribution terms. The tokenizers wheel lacks an embedded LICENSE: the exact same-version PyPI source archive supplies it, with a wheel/source/metadata hash proof. No source package is executed for this recovery. These third-party licenses remain applicable; the engine's MIT/Apache options do not relicense vendor components.

`audit_augur_windows_dependencies.py` walks direct and delay-load PE imports in every shipped EXE/DLL/PYD. Windows system APIs and the NVIDIA driver are classified as external prerequisites. Dynamic LoadLibrary paths and actual GPU functionality require separate smoke tests; a zero-unresolved static inventory alone is not deployment qualification.

The Windows package preserves build/CPU/source receipts, includes a `Verify-Package.cmd` command and separates content checksums from code signing. Final ZIPs need external SHA256 sums. No target/draft model, credential, user chat, offline session cache or developer environment is part of the distribution. Public upload is not performed by packaging.

## Remaining independent release gates

- Truly clean Windows machine installation with compatible driver/hardware.
- Hardware/driver combinations other than the measured SM70 32-GiB machine.
- Long-context quality/functionality combinations beyond the actual recorded cases.
- Model redistribution rights if anyone elects to distribute weights separately.
- Optional publisher code signing and external public-release upload.

Linux packaging remains outside the requested Windows deliverable.
