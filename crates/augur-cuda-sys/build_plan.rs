#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CudaSource {
    pub(crate) path: &'static str,
    pub(crate) object_stem: &'static str,
}

const CUDA_RERUN_ENV_VARS: &[&str] = &[
    "CUDA_PATH",
    "CUDA_HOME",
    "AUGUR_CUDA_NVCC",
    "AUGUR_CUDA_HOST_COMPILER",
    "AUGUR_CUDA_GENCODE",
];

const BLACKWELL_RERUN_ENV_VARS: &[&str] = &["AUGUR_CUTLASS_ROOT"];

const BASE_CUDA_SOURCES: &[CudaSource] = &[
    CudaSource {
        path: "native/augur_cuda_runtime.cu",
        object_stem: "augur_cuda_runtime",
    },
    CudaSource {
        path: "native/augur_q4k_cuda.cu",
        object_stem: "augur_q4k_cuda",
    },
    CudaSource {
        path: "native/augur_add_cuda.cu",
        object_stem: "augur_add_cuda",
    },
    CudaSource {
        path: "native/augur_gqa_repeat_cuda.cu",
        object_stem: "augur_gqa_repeat_cuda",
    },
    CudaSource {
        path: "native/augur_m120_qk_exact_sm70.generated.cu",
        object_stem: "augur_m120_qk_exact_sm70",
    },
    CudaSource {
        path: "native/augur_m121_q4k_dp4a_sm70.cu",
        object_stem: "augur_m121_q4k_dp4a_sm70",
    },
    CudaSource {
        path: "native/augur_nvfp4_sm70.cu",
        object_stem: "augur_nvfp4_sm70",
    },
];

pub(crate) const NVFP4_CUTLASS_SOURCE: CudaSource = CudaSource {
    path: "native/nv_nvfp4_cutlass_sm120.cu",
    object_stem: "nv_nvfp4",
};

pub(crate) const NVFP4_STUB_SOURCE: CudaSource = CudaSource {
    path: "native/nv_nvfp4_stub.cu",
    object_stem: "nv_nvfp4",
};

pub(crate) fn selected_rerun_env_vars(
    cuda_enabled: bool,
    blackwell_nvfp4_enabled: bool,
) -> impl Iterator<Item = &'static str> {
    CUDA_RERUN_ENV_VARS
        .iter()
        .copied()
        .filter(move |_| cuda_enabled)
        .chain(
            BLACKWELL_RERUN_ENV_VARS
                .iter()
                .copied()
                .filter(move |_| cuda_enabled && blackwell_nvfp4_enabled),
        )
}

pub(crate) fn selected_nvfp4_source(
    blackwell_nvfp4_enabled: bool,
    cutlass_available: bool,
) -> Option<CudaSource> {
    blackwell_nvfp4_enabled.then_some(if cutlass_available {
        NVFP4_CUTLASS_SOURCE
    } else {
        NVFP4_STUB_SOURCE
    })
}

pub(crate) fn object_file_name(source: CudaSource, target_os: &str) -> String {
    let extension = if target_os == "windows" { "obj" } else { "o" };
    format!("{}.{extension}", source.object_stem)
}

pub(crate) fn selected_cuda_sources_public() -> Vec<CudaSource> {
    BASE_CUDA_SOURCES.to_vec()
}

pub(crate) const PUBLIC_FEATURE_ENV: &[&str] = &[
    "CARGO_FEATURE_CUDA",
    "CARGO_FEATURE_DEFAULT",
    "CARGO_FEATURE_M306_K17_BF16_WARP_ROW",
    "CARGO_FEATURE_QWEN35_DEVICE_TOKEN_HANDOFF",
    "CARGO_FEATURE_QWEN35_FULL_ATTENTION_PARALLEL_EXP",
    "CARGO_FEATURE_QWEN35_FUSED_FFN_SWIGLU",
    "CARGO_FEATURE_QWEN35_FUSED_HEAD_NORM_ROPE",
    "CARGO_FEATURE_QWEN35_GDN_WARP_ROW",
    "CARGO_FEATURE_QWEN35_GROUPED_GQA_ONLINE_SOFTMAX",
    "CARGO_FEATURE_QWEN35_KV8_BATCHED_ATTENTION",
    "CARGO_FEATURE_QWEN35_KV8_CHUNK_PREFILL",
    "CARGO_FEATURE_QWEN35_KV8_FULL_ATTENTION",
    "CARGO_FEATURE_QWEN35_LINEAR_STEP",
    "CARGO_FEATURE_QWEN35_NATIVE_F16_SCALE",
    "CARGO_FEATURE_QWEN35_NVFP4_FUSED_LB10",
    "CARGO_FEATURE_QWEN35_PREFILL_CUBLAS",
    "CARGO_FEATURE_QWEN35_AUGUR_LARGE_PREFILL",
    "CARGO_FEATURE_QWEN35_VISION_AFFINE",
    "CARGO_FEATURE_QWEN35_Q8_PACKED_LOAD",
    "CARGO_FEATURE_QWEN35_SPLIT_RMSNORM",
    "CARGO_FEATURE_QWEN35_TARGET_SPLIT_ROPE",
    "CARGO_FEATURE_QWEN35_WARP_FULL_ATTENTION",
];

pub(crate) fn public_feature_env_is_allowed(name: &str) -> bool {
    PUBLIC_FEATURE_ENV.contains(&name)
}

#[cfg(test)]
pub(crate) fn known_cuda_sources() -> Vec<CudaSource> {
    vec![
        CudaSource {
            path: "native/augur_q4k_cuda.cu",
            object_stem: "augur_q4k_cuda",
        },
        CudaSource {
            path: "native/augur_add_cuda.cu",
            object_stem: "augur_add_cuda",
        },
        CudaSource {
            path: "native/augur_cuda_runtime.cu",
            object_stem: "augur_cuda_runtime",
        },
        CudaSource {
            path: "native/augur_gqa_repeat_cuda.cu",
            object_stem: "augur_gqa_repeat_cuda",
        },
        CudaSource {
            path: "native/augur_m120_qk_exact_sm70.generated.cu",
            object_stem: "augur_m120_qk_exact_sm70",
        },
        CudaSource {
            path: "native/augur_m121_q4k_dp4a_sm70.cu",
            object_stem: "augur_m121_q4k_dp4a_sm70",
        },
        CudaSource {
            path: "native/augur_m306_k14_async_fault_sm70.cu",
            object_stem: "augur_m306_k14_async_fault_sm70",
        },
        CudaSource {
            path: "native/augur_nvfp4_cutlass_sm120.cu",
            object_stem: "augur_nvfp4_cutlass_sm120",
        },
        CudaSource {
            path: "native/augur_nvfp4_sm70.cu",
            object_stem: "augur_nvfp4_sm70",
        },
        CudaSource {
            path: "native/augur_nvfp4_stub.cu",
            object_stem: "augur_nvfp4_stub",
        },
        CudaSource {
            path: "native/augur_prefill_m128_qk_projection_sm70.cu",
            object_stem: "augur_prefill_m128_qk_projection_sm70",
        },
        CudaSource {
            path: "native/augur_qwen35_device_token_handoff_sm70.cu",
            object_stem: "augur_qwen35_device_token_handoff_sm70",
        },
        CudaSource {
            path: "native/augur_qwen35_dflash_acceptance_sm70.cu",
            object_stem: "augur_qwen35_dflash_acceptance_sm70",
        },
        CudaSource {
            path: "native/augur_qwen35_kv8_segment_v2_sm70.cu",
            object_stem: "augur_qwen35_kv8_segment_v2_sm70",
        },
        CudaSource {
            path: "native/augur_qwen35_linear_step_sm70.cu",
            object_stem: "augur_qwen35_linear_step_sm70",
        },
        CudaSource {
            path: "native/augur_qwen35_prefill_cublas_sm70.cu",
            object_stem: "augur_qwen35_prefill_cublas_sm70",
        },
        CudaSource {
            path: "native/augur_qwen35_nvfp4_qpn_sm70.cu",
            object_stem: "augur_qwen35_nvfp4_qpn_sm70",
        },
        CudaSource {
            path: "native/nv_nvfp4_cutlass_sm120.cu",
            object_stem: "nv_nvfp4_cutlass_sm120",
        },
        CudaSource {
            path: "native/nv_nvfp4_stub.cu",
            object_stem: "nv_nvfp4_stub",
        },
    ]
}
