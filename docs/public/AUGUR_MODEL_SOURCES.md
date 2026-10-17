# QUASAR target and DFlash source records

Updated: October 15, 2026. This is the current Windows source candidate's provenance record. It is not a completed weight-redistribution review.

## Default target

The current manifest names [QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4), at recorded revision d9af17e20644c5b9a6d66dff1d14f349e2e61b05. The local model card identifies Qwen/Qwen3.8-27B as its base and declares apache-2.0. These statements were read from local artifacts in this review; remote revision/license revalidation is pending after public API connection timeouts.

The owner manifest binds five safetensors shards, index, config, tokenizer, tokenizer config, generation config and chat template. Original target weights and the original BF16 head are the intended inputs. A repository name is not a replacement for these identities.

These are the hashes recorded in the admitted manifest. The large files were not rehashed during the ongoing exclusive GPU timing run.

| Shard | Recorded SHA-256 |
|---|---|
| model-00001-of-00005.safetensors | 54d83c1d36631de231876217a8e0c2483eccee8746369a482b79442bdfc5d958 |
| model-00002-of-00005.safetensors | bb5c367ac5ba8b2ab68f07b9719f1c3ac528ada98c26523be6e2c07f82e299bc |
| model-00003-of-00005.safetensors | 6bfa358557579f538682984a667e243a72ab361d50c583fbf06ba3148653fda1 |
| model-00004-of-00005.safetensors | 260822c66711e7fbd9df0f96b0da0b54b01c2bd77612911885b33f88f18003d7 |
| model-00005-of-00005.safetensors | 08e4c9bcd8ca371240cae47c98e5e1a499a14caeab5ee2ca4a0a07c24c69a24b |

## Original draft and selected local Q8 derivative

The local card identifies [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) as a mirror of [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) and declares apache-2.0. The exact public draft commit and its LFS hashes still require online revalidation. The local folder name alone is not source proof.

| Artifact | Recorded identity |
|---|---|
| Original model.safetensors | SHA-256 67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c |
| Draft config.json | SHA-256 873e3556509b0da06e29654ba00d4944888d4b5e8a33afde25f7eb27d321e980 |
| Selected Rust Q8_0 + embedded FP32 gate0.45 GGUF | SHA-256 33d9842166866dae79eeff0026bf90991f438ef0aaa2578429d950f536b0d797; 2,060,264,832 bytes |
| Embedded 256 x 5120 FP32 gate0.45 | SHA-256 0bcb694a4e84aa304d888829742fdef92941771ac5e97917ca5dc4388f6821c7 |
| Historical llama-converted Q8_0 GGUF (not the release default) | SHA-256 5cf0923d4f481e82f38291d9057f5944a043c74142519cb28eb992ec796aaabd; 2,056,414,624 bytes |

This GGUF is a local derivative with a conversion receipt and tensor inventory. A conversion receipt alone does not qualify runtime loading, acceptance or TPS; those require separate evidence. The release selects the Rust safetensors-to-Q8 converter output, with 48 Q8_0 matrices and 33 FP32 tensors. Gate0.45 is folded once into FP32 source weights; runtime scale remains 1. The local paired GPU acceptance results are documented in AUGUR_EMBEDDED_GATE045_ENGLISH_ACCEPTANCE_20261015.md, not inferred from conversion alone.

The draft shares the target embedding/head path in this engine. Upstream descriptions of lossless speculation do not establish B1/B8 bitwise equivalence for the engine's separate packed-B8 arithmetic profile. Semantic and state-contract qualifications remain distinct.

## Source attribution and release checks

Existing source notices cover llama.cpp/ggml adaptations, the separately attributed Apache-2.0 NInfer QPN portions, and the local Lucide artifact. Preserve the actual files in crates/augur-cuda-sys/native/llama_attention_vendor, crates/augur-cuda-sys/native/licenses, scripts/augur_web/vendor and the root NOTICE. Exact adapted-source revisions and a final dependency inventory are still required; an unpacked directory label is not a verified Git revision.

Working evidence: evidence/analysis/quasar-release-provenance-20261012-v1/local-provenance.json. It binds small local records and records the failed online attempts. Public export must remove machine-specific/private evidence paths rather than copying that manifest verbatim.

Pending before redistribution: retrieve official target/base/draft metadata at immutable revisions; compare upstream LFS hashes with admitted local weight identities; retain upstream license/NOTICE files; complete native/Python/web dependency attribution and runtime distribution checks. No weights were packaged or uploaded by this review.
