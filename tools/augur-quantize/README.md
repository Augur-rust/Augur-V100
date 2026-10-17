# Native Rust Q8_0 GGUF converter (Windows)

This CPU-only command converts **already mapped F32, F16 or BF16 GGUF v2/v3** tensors to Q8_0. It does not launch llama.cpp, Python, CUDA or an external quantizer. The generic GGUF lane is independently validated. The DFlash safetensors lane produces Q8_0 + an embedded calibrated FP32 gate, with separate format and inference validation receipts.

Build from this directory with:

    cargo build --release --locked --offline

Use the compiled executable:

    augur-quantize.exe plan input-float.gguf
    augur-quantize.exe convert input-float.gguf new-output-directory
    augur-quantize.exe convert input-float.gguf new-output-directory --chunk-bytes 1048576

The parent output directory must exist. The destination must not exist. Successful conversion publishes a directory containing model-q8_0.gguf and receipt.json. The input is pinned with Windows sharing locks and a full SHA256. Existing quantized inputs, nonfinite floating inputs and unrepresentable scales are rejected. A handled error cleans up the files created by the converter; abrupt process termination may leave a .augur-q8-partial-* directory, which is never automatically reused or deleted.

For generic architectures, every rank >= 2 float tensor whose first GGUF dimension is divisible by 32 is quantized. Mapped DFlash inputs retain F32 norms and convolution bases; non-matrix DFlash roles that are not already F32 are rejected and should use the safetensors lane. Other float tensors retain their original bytes and type. This explicit policy does not imply the HF tensor mapping, transformations or norm policy of every architecture. All original metadata records preserve their wire representation except general.file_type (MOSTLY_Q8_0 = 7) and general.quantization_version (2). GGML tensor type Q8_0 is 8.

Payload reads are bounded by the configured chunk size (4 KiB to 8 MiB, multiples of 128); the default is 1 MiB. GGUF header parsing has an independent 128 MiB limit. This is not a claim of 1 MiB total process memory. The space plan includes the entire new output plus 128 MiB for receipt and publication overhead. Both original weights and completed prior outputs remain intact.

The receipt records input/output and converter executable SHA256, all tensor sizes and actions, conversion wall time, per-tensor maximum absolute error, RMSE and relative L2 error. It explicitly leaves model semantic and DFlash acceptance qualification false. The result is a mixed Q8_0/float GGUF, according to the policy above.

Validation uses scripts/validate_augur_rust_q8.py with the user's fixed llama.cpp b11552 source, extracts quantize_row_q8_0_ref without modifying its arithmetic, compiles it with MSVC /fp:strict, and compares all output block bytes. The harness uses the official local GGUF reader and records source and executable hashes. It includes half-scale boundaries, deterministic random samples and samples from the original DFlash tensors. This developer validation requires Python and MSVC; running the built converter does not.

The offline Q8_0 scale helper follows the existing model crate's portable ggml-compatible nearest-even conversion. The general F16Bits constructor and runtime inference arithmetic are unchanged. Reference: the local ggml/src/ggml-quants.c and ggml/src/ggml-impl.h in F:/llama.cpp-b11552/llama.cpp-b11552 (ggml/llama.cpp MIT licensing and notices remain applicable).

## Original DFlash safetensors: Q8_0 + FP32 gate0.45

The new commands map the original five-layer DFlash2DraftModel directly in Rust:

    augur-quantize.exe dflash-plan DRAFT_DIR TARGET_DIR --weights-sha256 ORIGINAL_SHA256
    augur-quantize.exe dflash-convert DRAFT_DIR TARGET_DIR NEW_OUTPUT_DIR --weights-sha256 ORIGINAL_SHA256

DRAFT_DIR contains the original config.json and model.safetensors. TARGET_DIR supplies config.json, tokenizer.json, tokenizer_config.json and chat_template.jinja; its target weights are never copied or quantized. The admitted profile is the QUASAR/Qwen3.5 27B target with the original five-layer DFlash2 configuration. Unsupported RoPE, architecture, tokenizer/preprocessor and tensor layouts fail explicitly. Sharded draft inputs and patched mask embeddings are not supported by this entry point.

The exact 81-tensor inventory is enforced. Forty-eight two-dimensional matrices become Q8_0; 32 norms and three-dimensional convolution bases expand exactly to F32. The selector gate expands from its original F32/F16/BF16 tensor and is multiplied by FP32 0.45 exactly once, becoming the 33rd F32 tensor. Its SHA256, folded factor, runtime scale1 and global text policy are stored in GGUF metadata. Never quantize then dequantize the gate, or multiply by 0.45 again at runtime. HF row-major payload order is retained and only GGUF dimension descriptors reverse order. NeoX Q/K is required, so no hidden RoPE permutation is applied. Target layer metadata uses HF layer IDs plus one. Tokenizer vocabulary, added-token roles, padding entries, merges, specials and the template are derived from authenticated target files, with a strict admitted qwen35 pre-tokenizer fingerprint. Model source/license claims are not inferred from directory names.

The new directory contains model-q8_0.gguf, the unchanged draft config.json and receipt.json. It is never installed as the selected draft automatically. Input weights require a full SHA256 and remain pinned during conversion. Metadata files also stay read-locked on Windows. The selector identity plan uses a bounded 5 MiB expanded gate buffer plus its source buffer. Streaming conversion buffers are bounded by --chunk-bytes (128 bytes to 16 MiB, multiples of 128); parsed tokenizer metadata has separate bounded storage. No intermediate float GGUF or whole-model float shadow is needed.

Developer validation: scripts/validate_augur_rust_dflash.py compares every tensor byte and all mathematical/tokenizer metadata against the user's fixed llama conversion. Names/provenance fields and metadata order may differ, so matching whole-file SHA is not required. The oracle retains both input identities and reports any mismatch before admission. GPU loading, acceptance and TPS remain separate tests.

The Windows engine build includes this CLI and its CPU tests. The October 15 tensor audit found all other 80 tensors byte-identical to the previously admitted Q8 draft, and the embedded gate byte-identical to the admitted external FP32 gate0.45. English-thinking HTTP measurements are documented in the root README; format conversion alone does not establish model quality.
