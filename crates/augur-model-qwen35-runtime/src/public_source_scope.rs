//! Public source-scope validation for build metadata, never product authority.
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, fs, path::Path};

pub const SCOPE_FILE: &str = "qwen35-public-source-scope.json";
pub const ATTESTATION_SCHEMA: &str = "qwen35-public-frozen-source-build-input-v2";
pub const UPSTREAM: &str = "b4c5dd2ee156551872713ed5cce92f3601ddacd6";
pub const PACKAGES: &[&str] = &[
    "augur-backend",
    "augur-compiler",
    "augur-core",
    "augur-cuda-sys",
    "augur-exec",
    "augur-formats",
    "augur-graph",
    "augur-kernel-api",
    "augur-kernels",
    "augur-model-qwen35",
    "augur-model-qwen35-runtime",
    "augur-reference",
    "augur-repack",
];
pub const ENTRYPOINTS: &[&str] = &[
    "qwen35_q8_draft_target_stream_cuda",
    "qwen35_resident_session_http",
];
pub const RUNTIME_FEATURES: &[&str] = &[
    "cuda",
    "qwen35-autoregressive-cuda",
    "qwen35-batched-prefill-handoff",
    "qwen35-bf16-warp-row",
    "qwen35-device-resident-handoff",
    "qwen35-device-token-handoff",
    "qwen35-draft-private-capture-only",
    "qwen35-draft-private-long",
    "qwen35-draft-private-state",
    "qwen35-draft-target-verify",
    "qwen35-draft-top16-batch",
    "qwen35-frozen-build",
    "qwen35-full-attention",
    "qwen35-full-attention-parallel-exp",
    "qwen35-full-target-cuda",
    "qwen35-fused-ffn-swiglu",
    "qwen35-fused-head-norm-rope",
    "qwen35-gdn-warp-row",
    "qwen35-grouped-gqa-online-softmax",
    "qwen35-kv8-batched-attention",
    "qwen35-kv8-chunk-prefill",
    "qwen35-kv8-full-attention",
    "qwen35-linear-attention",
    "qwen35-linear-layer",
    "qwen35-linear-step",
    "qwen35-mixed-prefix-cuda",
    "qwen35-native-f16-scale",
    "qwen35-nvfp4-fused-lb10",
    "qwen35-optimized-state-bridge",
    "qwen35-output-head-acceptance",
    "qwen35-prefill-cublas",
    "qwen35-q8-packed-load",
    "qwen35-raw-committed-state",
    "qwen35-resident-text-service",
    "qwen35-session-checkpoint",
    "qwen35-session-context32-audit",
    "qwen35-session-context64-audit",
    "qwen35-session-continuation",
    "qwen35-session-near192-audit",
    "qwen35-split-rmsnorm",
    "qwen35-target-split-rope",
    "qwen35-top16-state-client",
    "qwen35-warp-full-attention",
    "qwen35-zero-copy-layer-handoff",
    "qwen35-zero-copy-native-f16-scale",
];
pub const NATIVE_FILES: &[&str] = &[
    "crates/augur-cuda-sys/native/augur_add_cuda.cu",
    "crates/augur-cuda-sys/native/augur_cuda_internal.h",
    "crates/augur-cuda-sys/native/augur_cuda_runtime.cu",
    "crates/augur-cuda-sys/native/augur_cuda_runtime.h",
    "crates/augur-cuda-sys/native/augur_gqa_repeat_cuda.cu",
    "crates/augur-cuda-sys/native/augur_m120_qk_exact_sm70.generated.cu",
    "crates/augur-cuda-sys/native/augur_m121_q4k_dp4a_sm70.cu",
    "crates/augur-cuda-sys/native/augur_m306_k14_async_fault_sm70.cu",
    "crates/augur-cuda-sys/native/augur_nvfp4_cutlass_sm120.cu",
    "crates/augur-cuda-sys/native/augur_nvfp4_sm70.cu",
    "crates/augur-cuda-sys/native/augur_nvfp4_stub.cu",
    "crates/augur-cuda-sys/native/augur_prefill_m128_qk_projection_sm70.cu",
    "crates/augur-cuda-sys/native/augur_q4k_cuda.cu",
    "crates/augur-cuda-sys/native/augur_qwen35_canonical_prefill.cuh",
    "crates/augur-cuda-sys/native/augur_qwen35_canonical_prefill_scores.cuh",
    "crates/augur-cuda-sys/native/augur_qwen35_device_token_handoff_sm70.cu",
    "crates/augur-cuda-sys/native/augur_qwen35_dflash_acceptance_sm70.cu",
    "crates/augur-cuda-sys/native/augur_qwen35_exact_ffn_down_row_reuse.cuh",
    "crates/augur-cuda-sys/native/augur_qwen35_exact_projection_row_reuse.cuh",
    "crates/augur-cuda-sys/native/augur_qwen35_exact_small_batch_tiles.cuh",
    "crates/augur-cuda-sys/native/augur_qwen35_gdn_input_b8.cuh",
    "crates/augur-cuda-sys/native/augur_qwen35_kv8_segment_v2_abi.h",
    "crates/augur-cuda-sys/native/augur_qwen35_kv8_segment_v2_sm70.cu",
    "crates/augur-cuda-sys/native/augur_qwen35_linear_step_sm70.cu",
    "crates/augur-cuda-sys/native/augur_qwen35_nvfp4_grouped_decode.cuh",
    "crates/augur-cuda-sys/native/augur_qwen35_nvfp4_qpn_abi.h",
    "crates/augur-cuda-sys/native/augur_qwen35_nvfp4_qpn_gemm.cuh",
    "crates/augur-cuda-sys/native/augur_qwen35_nvfp4_qpn_layout.cuh",
    "crates/augur-cuda-sys/native/augur_qwen35_nvfp4_qpn_sm70.cu",
    "crates/augur-cuda-sys/native/augur_qwen35_prefill_compensated_tensor.cuh",
    "crates/augur-cuda-sys/native/augur_qwen35_prefill_cublas_sm70.cu",
    "crates/augur-cuda-sys/native/augur_qwen35_prefill_integer_mma.cuh",
    "crates/augur-cuda-sys/native/augur_qwen35_prefill_integer_q8.cuh",
    "crates/augur-cuda-sys/native/augur_qwen35_prefill_llama_activation_sm70.cu",
    "crates/augur-cuda-sys/native/augur_qwen35_prefill_llama_mmq_sm70.cu",
    "crates/augur-cuda-sys/native/augur_qwen35_prefill_llama_small_sm70.cuh",
    "crates/augur-cuda-sys/native/augur_qwen35_qpn_group.h",
    "crates/augur-cuda-sys/native/augur_qwen35_qpn_prefill.cuh",
    "crates/augur-cuda-sys/native/augur_qwen35_replay_fold_shared.cuh",
    "crates/augur-cuda-sys/native/augur_qwen35_rotated_kv_abi.h",
    "crates/augur-cuda-sys/native/augur_qwen35_rotated_kv_gqa8_sm70.cu",
    "crates/augur-cuda-sys/native/augur_qwen35_rotated_kv_gqa8_sm70.cuh",
    "crates/augur-cuda-sys/native/augur_qwen35_rotated_kv_input_sm70.cuh",
    "crates/augur-cuda-sys/native/augur_qwen35_rotated_kv_planar_gqa8_sm70.cu",
    "crates/augur-cuda-sys/native/augur_qwen35_rotated_kv_planar_layout.cuh",
    "crates/augur-cuda-sys/native/augur_qwen35_rotated_kv_planar_sm70.cu",
    "crates/augur-cuda-sys/native/augur_qwen35_rotated_kv_policy_sm70.cu",
    "crates/augur-cuda-sys/native/augur_qwen35_rotated_kv_sm70.cu",
    "crates/augur-cuda-sys/native/augur_qwen35_rotated_kv_small_sm70.cu",
    "crates/augur-cuda-sys/native/augur_qwen35_row_q8_input.cuh",
    "crates/augur-cuda-sys/native/augur_qwen35_row_q8_input_mmq_sm70.cu",
    "crates/augur-cuda-sys/native/augur_qwen35_vision_affine.cuh",
    "crates/augur-cuda-sys/native/augur_qwen35_vision_control.cuh",
    "crates/augur-cuda-sys/native/augur_rmsnorm_warp_tree5120.cuh",
    "crates/augur-cuda-sys/native/licenses/NINFER-APACHE-2.0.txt",
    "crates/augur-cuda-sys/native/licenses/NINFER-NOTICE.txt",
    "crates/augur-cuda-sys/native/llama_attention_vendor/LICENSE",
    "crates/augur-cuda-sys/native/llama_attention_vendor/README.md",
    "crates/augur-cuda-sys/native/llama_attention_vendor/augur_gqa8_cohort_sm70.cuh",
    "crates/augur-cuda-sys/native/llama_attention_vendor/augur_packed_causal_sm70.cuh",
    "crates/augur-cuda-sys/native/llama_attention_vendor/augur_packed_tile_sm70.cuh",
    "crates/augur-cuda-sys/native/llama_attention_vendor/augur_quantize_mmq_device.cuh",
    "crates/augur-cuda-sys/native/llama_attention_vendor/augur_vec_sync_sm70.cuh",
    "crates/augur-cuda-sys/native/llama_attention_vendor/common.cuh",
    "crates/augur-cuda-sys/native/llama_attention_vendor/convert.cuh",
    "crates/augur-cuda-sys/native/llama_attention_vendor/cp-async.cuh",
    "crates/augur-cuda-sys/native/llama_attention_vendor/fattn-common.cuh",
    "crates/augur-cuda-sys/native/llama_attention_vendor/fattn-mma-reference.cuh",
    "crates/augur-cuda-sys/native/llama_attention_vendor/fattn-swizzle.cuh",
    "crates/augur-cuda-sys/native/llama_attention_vendor/fattn-tile-reference.cuh",
    "crates/augur-cuda-sys/native/llama_attention_vendor/fattn-vec-reference.cuh",
    "crates/augur-cuda-sys/native/llama_attention_vendor/ggml-alloc.h",
    "crates/augur-cuda-sys/native/llama_attention_vendor/ggml-backend.h",
    "crates/augur-cuda-sys/native/llama_attention_vendor/ggml-common.h",
    "crates/augur-cuda-sys/native/llama_attention_vendor/ggml-cuda.h",
    "crates/augur-cuda-sys/native/llama_attention_vendor/ggml-impl.h",
    "crates/augur-cuda-sys/native/llama_attention_vendor/ggml.h",
    "crates/augur-cuda-sys/native/llama_attention_vendor/gguf.h",
    "crates/augur-cuda-sys/native/llama_attention_vendor/mma.cuh",
    "crates/augur-cuda-sys/native/llama_attention_vendor/mmq-config-ampere.cuh",
    "crates/augur-cuda-sys/native/llama_attention_vendor/mmq-config-blackwell.cuh",
    "crates/augur-cuda-sys/native/llama_attention_vendor/mmq-config-cdna.cuh",
    "crates/augur-cuda-sys/native/llama_attention_vendor/mmq-config-pascal-dp4a.cuh",
    "crates/augur-cuda-sys/native/llama_attention_vendor/mmq-config-pascal-older.cuh",
    "crates/augur-cuda-sys/native/llama_attention_vendor/mmq-config-rdna2.cuh",
    "crates/augur-cuda-sys/native/llama_attention_vendor/mmq-config-rdna3-5.cuh",
    "crates/augur-cuda-sys/native/llama_attention_vendor/mmq-config-rdna3.cuh",
    "crates/augur-cuda-sys/native/llama_attention_vendor/mmq-config-rdna4.cuh",
    "crates/augur-cuda-sys/native/llama_attention_vendor/mmq-load-tiles.cuh",
    "crates/augur-cuda-sys/native/llama_attention_vendor/mmq-vec-dot.cuh",
    "crates/augur-cuda-sys/native/llama_attention_vendor/mmq.cuh",
    "crates/augur-cuda-sys/native/llama_attention_vendor/vecdotq.cuh",
    "crates/augur-cuda-sys/native/llama_attention_vendor/vendors/cuda.h",
    "crates/augur-cuda-sys/native/nv_nvfp4_cutlass_sm120.cu",
    "crates/augur-cuda-sys/native/nv_nvfp4_stub.cu",
];
pub const CLIENT_FILES: &[&str] = &[
    "PERSISTENT_CHAT.md",
    "STARTUP_OPTIMIZATION.md",
    "crates/augur-model-qwen35/examples/pinned_startup_bill.rs",
    "docs/QWEN35_DECODE_COMBO_LONG_CONTEXT.md",
    "docs/QWEN35_DFLASH_SHORT_UNROLL_EXPERIMENT.md",
    "docs/QWEN35_DFLASH_TOP16_LOCAL_EXPERIMENT.md",
    "docs/QWEN35_FEATURE_WINDOW_SHIFT_20261006.md",
    "docs/QWEN35_GQA8_MMA_DIAGNOSTIC.md",
    "docs/QWEN35_INPUT_OUTPUT_DOWN_AUDIT.md",
    "docs/QWEN35_LLAMA_INPUT_NORM_DISPOSITION_20261004.md",
    "docs/QWEN35_LLAMA_INPUT_ROWS_AUDIT_V1.md",
    "docs/QWEN35_LLAMA_SMALL_INPUT_ROWS_V1.md",
    "docs/QWEN35_NEAR192_DENOMINATOR.md",
    "docs/QWEN35_NINFER_MIXED_FIRST_20261008.md",
    "docs/QWEN35_OPEN_CODE_TOOL_QUALIFICATION.md",
    "docs/QWEN35_ROTATED_KV_NATIVE_V1.md",
    "docs/QWEN35_ROTATED_KV_VENDOR_BINDING.json",
    "docs/QWEN35_SUBMISSION_BUNDLE.md",
    "docs/QWEN35_TARGET_REPLAY_AND_ADAPTIVE_DFLASH_PLAN_20261004.md",
    "docs/QWEN35_TARGET_STATIC_COLS_EXPERIMENT.md",
    "docs/QWEN35_TEXT_API.md",
    "scripts/experiments/qwen35_ninfer_pair/benchmark.cu",
    "scripts/experiments/qwen35_ninfer_pair/checkpoint_branch_ledger.py",
    "scripts/experiments/qwen35_ninfer_pair/checkpoint_context_fixture.py",
    "scripts/experiments/qwen35_ninfer_pair/checkpoint_full_fixture.py",
    "scripts/experiments/qwen35_ninfer_pair/checkpoint_layer_fixture.py",
    "scripts/experiments/qwen35_ninfer_pair/checkpoint_math_fixture.py",
    "scripts/experiments/qwen35_ninfer_pair/checkpoint_target_fixture.py",
    "scripts/experiments/qwen35_ninfer_pair/contract.cu",
    "scripts/experiments/qwen35_ninfer_pair/exact_batch_bench.cu",
    "scripts/experiments/qwen35_ninfer_pair/expand_bench.cu",
    "scripts/experiments/qwen35_ninfer_pair/run.py",
    "scripts/experiments/qwen35_ninfer_pair/run_checkpoint_projection.py",
    "scripts/experiments/qwen35_ninfer_pair/run_contract.py",
    "scripts/experiments/qwen35_ninfer_pair/run_exact_batch.py",
    "scripts/experiments/qwen35_ninfer_pair/run_http_smoke.py",
    "scripts/experiments/qwen35_ninfer_pair/run_retained_ffn.py",
    "scripts/experiments/qwen35_ninfer_pair/run_rust_abi.py",
    "scripts/experiments/qwen35_ninfer_pair/run_target_smoke.py",
    "scripts/experiments/qwen35_ninfer_pair/test_checkpoint_branch_ledger.py",
    "scripts/experiments/qwen35_ninfer_pair/test_checkpoint_kv8_math.py",
    "scripts/experiments/qwen35_ninfer_pair/test_qpn_postscale_math.py",
    "scripts/qwen35_action_contract.py",
    "scripts/qwen35_api.py",
    "scripts/qwen35_augur_checkpoint.py",
    "scripts/qwen35_chat.py",
    "scripts/qwen35_checkpoint_profile.py",
    "scripts/qwen35_dflash_trajectory.py",
    "scripts/qwen35_nvidia_checkpoint.py",
    "scripts/qwen35_persistence.py",
    "scripts/qwen35_session.py",
    "scripts/qwen35_tools.py",
    "scripts/test_qwen35_action_contract.py",
    "scripts/test_qwen35_api.py",
    "scripts/test_qwen35_api_sdk.py",
    "scripts/test_qwen35_augur_checkpoint.py",
    "scripts/test_qwen35_chat.py",
    "scripts/test_qwen35_checkpoint_profile.py",
    "scripts/test_qwen35_dflash_short_unroll.py",
    "scripts/test_qwen35_dflash_top16_local.py",
    "scripts/test_qwen35_dflash_trajectory.py",
    "scripts/test_qwen35_kv8_staging.py",
    "scripts/test_qwen35_nvidia_checkpoint.py",
    "scripts/test_qwen35_persistence.py",
    "scripts/test_qwen35_session.py",
    "scripts/test_qwen35_target_static_cols.py",
    "scripts/test_qwen35_tools.py",
];

fn require(ok: bool, reason: &str) -> Result<(), String> {
    if ok {
        Ok(())
    } else {
        Err(reason.to_owned())
    }
}

fn strings(value: &Value) -> Result<Vec<&str>, String> {
    value
        .as_array()
        .ok_or("string array required")?
        .iter()
        .map(|v| {
            v.as_str()
                .ok_or_else(|| "string element required".to_owned())
        })
        .collect()
}

fn lowercase_hex(value: &Value, length: usize) -> bool {
    value.as_str().is_some_and(|s| {
        s.len() == length
            && s.bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    })
}

fn allowed_source(name: &str) -> bool {
    if name.is_empty()
        || name.len() > 2048
        || !name
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"_-./".contains(&c))
    {
        return false;
    }
    let parts: Vec<_> = name.split('/').collect();
    if parts
        .iter()
        .any(|p| p.is_empty() || *p == "." || *p == ".." || *p == ".git" || p.ends_with('.'))
    {
        return false;
    }
    if [
        "Cargo.toml",
        "Cargo.lock",
        "rust-toolchain.toml",
        "LICENSE-MIT",
        "LICENSE-APACHE",
        "NOTICE",
        "README.md",
        "BUILDING.md",
        ".gitignore",
        SCOPE_FILE,
        "docs/public/QWEN_MODEL_SOURCES.md",
        "scripts/build-public-qwen.py",
    ]
    .contains(&name)
        || CLIENT_FILES.contains(&name)
    {
        return true;
    }
    if parts.len() < 3 || parts[0] != "crates" || !PACKAGES.contains(&parts[1]) {
        return false;
    }
    if parts.len() == 3 && ["Cargo.toml", "build.rs", "build_plan.rs"].contains(&parts[2]) {
        return true;
    }
    if ["src", "tests"].contains(&parts[2]) {
        return name.ends_with(".rs");
    }
    if parts[2] == "native" {
        return NATIVE_FILES.contains(&name);
    }
    if parts[1] == "augur-model-qwen35-runtime" && parts[2] == "examples" {
        return ENTRYPOINTS
            .iter()
            .any(|e| name == format!("crates/augur-model-qwen35-runtime/examples/{e}.rs"))
            || parts.len() > 4 && parts[3] == "resident_service" && name.ends_with(".rs");
    }
    false
}

/// Validate the closed declaration against the complete attested source map.
/// The build script additionally hashes every file and checks the actual tree.
pub fn validate_value(
    scope: &Value,
    attestation: &Value,
    sources: &Map<String, Value>,
) -> Result<(), String> {
    require(
        attestation["schema"] == ATTESTATION_SCHEMA,
        "public V2 attestation required",
    )?;
    require(
        lowercase_hex(&attestation["source_commit"], 40)
            && attestation["source_commit"] != UPSTREAM,
        "independent public source commit required",
    )?;
    require(
        attestation["source_dirty"].as_bool() == Some(false),
        "clean public source required",
    )?;
    require(
        attestation["profile"] == "release"
            && attestation["opt_level"] == "3"
            && attestation["cuda_arch"] == "arch=compute_70,code=sm_70"
            && attestation["target"] == "x86_64-pc-windows-msvc",
        "public build profile drift",
    )?;
    let object = scope.as_object().ok_or("scope object required")?;
    let keys: BTreeSet<_> = object.keys().map(String::as_str).collect();
    require(
        keys == BTreeSet::from([
            "schema",
            "policy",
            "upstream_source_commit",
            "packages",
            "product_entrypoints",
            "files",
            "native_files",
            "runtime_features",
            "claims",
        ]),
        "scope keys drift",
    )?;
    require(
        scope["schema"] == "qwen35-public-source-scope-v1"
            && scope["policy"] == "qwen35-sm70-serial-text-source-v1"
            && scope["upstream_source_commit"] == UPSTREAM,
        "public scope identity drift",
    )?;
    require(
        strings(&scope["packages"])? == PACKAGES,
        "exact public package set required",
    )?;
    require(
        strings(&scope["product_entrypoints"])? == ENTRYPOINTS,
        "complete producer/HTTP entrypoints required",
    )?;
    require(
        strings(&scope["runtime_features"])? == RUNTIME_FEATURES
            && strings(&attestation["resolved_features"])? == RUNTIME_FEATURES
            && strings(&attestation["requested_features"])? == RUNTIME_FEATURES,
        "public runtime feature drift",
    )?;
    require(
        strings(&scope["native_files"])? == NATIVE_FILES,
        "closed native scope required",
    )?;
    let claims = scope["claims"].as_object().ok_or("scope claims required")?;
    require(
        claims.len() == 4
            && [
                "product_ready",
                "qualified",
                "semantic_qualification",
                "visible_tps",
            ]
            .iter()
            .all(|key| claims.get(*key).and_then(Value::as_bool) == Some(false)),
        "scope cannot grant authority",
    )?;
    let names = strings(&scope["files"])?;
    require(
        names.windows(2).all(|p| p[0] < p[1]),
        "sorted unique source names required",
    )?;
    require(
        names.iter().all(|name| allowed_source(name)),
        "unapproved source path",
    )?;
    let casefold: BTreeSet<_> = names.iter().map(|name| name.to_ascii_lowercase()).collect();
    require(casefold.len() == names.len(), "case-aliased source names")?;
    require(
        names.iter().copied().collect::<BTreeSet<_>>()
            == sources.keys().map(String::as_str).collect(),
        "complete source namespace binding required",
    )?;
    require(
        sources.values().all(|value| lowercase_hex(value, 64)),
        "source hashes required",
    )?;
    for path in [
        "Cargo.toml",
        "Cargo.lock",
        "rust-toolchain.toml",
        "LICENSE-MIT",
        "LICENSE-APACHE",
        "NOTICE",
        SCOPE_FILE,
        "README.md",
        "BUILDING.md",
        "scripts/build-public-qwen.py",
        "docs/public/QWEN_MODEL_SOURCES.md",
        "crates/augur-model-qwen35-runtime/build.rs",
        "crates/augur-model-qwen35-runtime/src/public_source_scope.rs",
    ] {
        require(
            sources.contains_key(path),
            "required public root or identity source missing",
        )?;
    }
    if RUNTIME_FEATURES.contains(&"qwen35-session-checkpoint") {
        require(
            sources.contains_key("scripts/qwen35_chat.py"),
            "current product chat client missing",
        )?;
    }
    for package in PACKAGES {
        require(
            sources.contains_key(&format!("crates/{package}/Cargo.toml"))
                && sources.contains_key(&format!("crates/{package}/src/lib.rs")),
            "package implementation missing",
        )?;
    }
    for entrypoint in ENTRYPOINTS {
        require(
            sources.contains_key(&format!(
                "crates/augur-model-qwen35-runtime/examples/{entrypoint}.rs"
            )),
            "complete public entrypoint source missing",
        )?;
    }
    for path in NATIVE_FILES {
        require(sources.contains_key(*path), "native source missing")?;
    }
    require(
        names.iter().any(|name| {
            name.starts_with("crates/augur-model-qwen35-runtime/examples/resident_service/")
        }),
        "resident service implementation missing",
    )?;
    Ok(())
}

pub fn validate(
    root: &Path,
    attestation: &Value,
    sources: &Map<String, Value>,
) -> Result<(), String> {
    let path = root.join(SCOPE_FILE);
    let metadata = fs::symlink_metadata(&path).map_err(|e| e.to_string())?;
    require(
        metadata.is_file()
            && !metadata.file_type().is_symlink()
            && metadata.len() <= 2 * 1024 * 1024,
        "bounded regular scope file required",
    )?;
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        require(
            metadata.file_attributes() & 0x400 == 0,
            "scope reparse rejected",
        )?;
    }
    let raw = fs::read(path).map_err(|e| e.to_string())?;
    let digest = format!("{:x}", Sha256::digest(&raw));
    require(
        attestation["public_source_scope_sha256"].as_str() == Some(digest.as_str())
            && sources.get(SCOPE_FILE).and_then(Value::as_str) == Some(digest.as_str()),
        "scope byte binding drift",
    )?;
    let scope: Value = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
    require(
        serde_json::to_vec(&scope).map_err(|e| e.to_string())? == raw,
        "canonical scope bytes required",
    )?;
    validate_value(&scope, attestation, sources)
}
