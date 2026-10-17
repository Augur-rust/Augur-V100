use std::{
    collections::VecDeque,
    env,
    ffi::OsString,
    path::{Path, PathBuf},
    process::{Child, Command, ExitStatus},
    thread,
    time::Duration,
};

mod build_plan;

use build_plan::{
    object_file_name, selected_cuda_sources_public, selected_nvfp4_source,
    selected_rerun_env_vars, NVFP4_CUTLASS_SOURCE,
};

fn main() {
    // Validate even CPU-only builds; ambient variables cannot restore retired features.
    for (key, _) in env::vars_os() {
        let name = key.to_string_lossy();
        if name.starts_with("CARGO_FEATURE_") {
            assert!(
                build_plan::public_feature_env_is_allowed(&name),
                "augur-cuda-sys public profile rejects feature environment: {}", name
            );
        }
    }
    publish_resolved_features();
    let cuda_enabled = env::var_os("CARGO_FEATURE_CUDA").is_some();
    let jobserver = cuda_enabled.then(inherited_jobserver).flatten();

    println!("cargo:rerun-if-changed=build_plan.rs");

    let blackwell_nvfp4_enabled = env::var_os("CARGO_FEATURE_BLACKWELL_NVFP4").is_some();
    for name in selected_rerun_env_vars(cuda_enabled, blackwell_nvfp4_enabled) {
        println!("cargo:rerun-if-env-changed={name}");
    }

    if !cuda_enabled {
        return;
    }

    println!("cargo:rerun-if-changed=native/augur_dflash_llama_bf16_head.cuh");
    println!("cargo:rerun-if-changed=native/augur_dflash_q8_b8_direct.cuh");
    println!("cargo:rerun-if-changed=native/augur_nvfp4_small_gate.cuh");
    println!("cargo:rerun-if-changed=native/augur_nvfp4_small_gate_pair.cuh");
    println!("cargo:rerun-if-changed=native/augur_rmsnorm_warp_tree5120.cuh");
    let mut cuda_sources = selected_cuda_sources_public();
    if env::var_os("CARGO_FEATURE_QWEN35_LINEAR_STEP").is_some() {
        cuda_sources.push(build_plan::CudaSource {
            path: "native/augur_qwen35_linear_step_sm70.cu",
            object_stem: "augur_qwen35_linear_step_sm70",
        });
    }
    if env::var_os("CARGO_FEATURE_QWEN35_DEVICE_TOKEN_HANDOFF").is_some() {
        cuda_sources.push(build_plan::CudaSource {
            path: "native/augur_qwen35_device_token_handoff_sm70.cu",
            object_stem: "augur_qwen35_device_token_handoff_sm70",
        });
        cuda_sources.push(build_plan::CudaSource {
            path: "native/augur_qwen35_dflash_acceptance_sm70.cu",
            object_stem: "augur_qwen35_dflash_acceptance_sm70",
        });
    }
    let qwen35_prefill_cublas_enabled =
        env::var_os("CARGO_FEATURE_QWEN35_PREFILL_CUBLAS").is_some();
    if qwen35_prefill_cublas_enabled {
        cuda_sources.push(build_plan::CudaSource {
            path: "native/augur_qwen35_nvfp4_qpn_sm70.cu",
            object_stem: "augur_qwen35_nvfp4_qpn_sm70",
        });
        println!("cargo:rerun-if-changed=native/augur_qwen35_nvfp4_qpn_abi.h");
        println!("cargo:rerun-if-changed=native/augur_qwen35_nvfp4_qpn_layout.cuh");
        println!("cargo:rerun-if-changed=native/augur_qwen35_nvfp4_qpn_gemm.cuh");
        println!("cargo:rerun-if-changed=native/augur_qwen35_qpn_prefill.cuh");
        println!("cargo:rerun-if-changed=native/augur_qwen35_qpn_group.h");
        println!("cargo:rerun-if-changed=native/augur_qwen35_gdn_input_b8.cuh");
        println!("cargo:rerun-if-changed=native/augur_qwen35_augur_prefill.cuh");
        println!("cargo:rerun-if-changed=native/augur_qwen35_augur_prefill_policy.cuh");
        cuda_sources.push(build_plan::CudaSource {
            path: "native/augur_qwen35_prefill_cublas_sm70.cu",
            object_stem: "augur_qwen35_prefill_cublas_sm70",
        });
        cuda_sources.push(build_plan::CudaSource {
            path: "native/augur_qwen35_prefill_llama_activation_sm70.cu",
            object_stem: "augur_qwen35_prefill_llama_activation_sm70",
        });
        cuda_sources.push(build_plan::CudaSource {
            path: "native/augur_qwen35_prefill_llama_mmq_sm70.cu",
            object_stem: "augur_qwen35_prefill_llama_mmq_sm70",
        });
        cuda_sources.push(build_plan::CudaSource {
            path: "native/augur_qwen35_row_q8_input_mmq_sm70.cu",
            object_stem: "augur_qwen35_row_q8_input_mmq_sm70",
        });
        cuda_sources.push(build_plan::CudaSource {
            path: "native/augur_qwen35_rotated_kv_sm70.cu",
            object_stem: "augur_qwen35_rotated_kv_sm70",
        });
        cuda_sources.push(build_plan::CudaSource {
            path: "native/augur_qwen35_rotated_kv_policy_sm70.cu",
            object_stem: "augur_qwen35_rotated_kv_policy_sm70",
        });
        cuda_sources.push(build_plan::CudaSource {
            path: "native/augur_qwen35_rotated_kv_small_sm70.cu",
            object_stem: "augur_qwen35_rotated_kv_small_sm70",
        });
        println!("cargo:rerun-if-changed=native/augur_qwen35_rotated_kv_abi.h");
        cuda_sources.push(build_plan::CudaSource {
            path: "native/augur_qwen35_rotated_kv_gqa8_sm70.cu",
            object_stem: "augur_qwen35_rotated_kv_gqa8_sm70",
        });
        println!("cargo:rerun-if-changed=native/augur_qwen35_rotated_kv_gqa8_sm70.cuh");
        cuda_sources.push(build_plan::CudaSource {
            path: "native/augur_qwen35_rotated_kv_planar_sm70.cu",
            object_stem: "augur_qwen35_rotated_kv_planar_sm70",
        });
        cuda_sources.push(build_plan::CudaSource {
            path: "native/augur_qwen35_rotated_kv_planar_gqa8_sm70.cu",
            object_stem: "augur_qwen35_rotated_kv_planar_gqa8_sm70",
        });
        cuda_sources.push(build_plan::CudaSource {
            path: "native/augur_qwen35_rotated_kv_planar_gqa8_c4_sm70.cu",
            object_stem: "augur_qwen35_rotated_kv_planar_gqa8_c4_sm70",
        });
        println!("cargo:rerun-if-changed=native/augur_qwen35_rotated_kv_planar_layout.cuh");
        println!("cargo:rerun-if-changed=native/augur_qwen35_prefill_llama_small_sm70.cuh");
        println!("cargo:rerun-if-changed=native/llama_attention_vendor");
    }
    if env::var_os("CARGO_FEATURE_QWEN35_KV8_FULL_ATTENTION").is_some() {
        cuda_sources.push(build_plan::CudaSource {
            path: "native/augur_qwen35_kv8_segment_v2_sm70.cu",
            object_stem: "augur_qwen35_kv8_segment_v2_sm70",
        });
    }
    let cutlass_root = if blackwell_nvfp4_enabled {
        env::var_os("AUGUR_CUTLASS_ROOT").map(PathBuf::from)
    } else {
        None
    };
    let nvfp4_source = selected_nvfp4_source(blackwell_nvfp4_enabled, cutlass_root.is_some());

    println!("cargo:rerun-if-changed=native/augur_cuda_runtime.h");
    println!("cargo:rerun-if-changed=native/augur_cuda_internal.h");
    println!("cargo:rerun-if-changed=native/augur_qwen35_rotated_kv_input_sm70.cuh");
    println!("cargo:rerun-if-changed=native/augur_qwen35_exact_small_batch_tiles.cuh");
    println!("cargo:rerun-if-changed=native/augur_qwen35_replay_fold_shared.cuh");
    println!("cargo:rerun-if-changed=native/augur_qwen35_augur_prefill_conv.cuh");
    println!("cargo:rerun-if-changed=native/augur_qwen35_augur_prefill_delta.cuh");
    println!("cargo:rerun-if-changed=native/augur_qwen35_augur_prefill_pack.cuh");
    println!("cargo:rerun-if-changed=native/augur_qwen35_augur_silu_pack.cuh");
    println!("cargo:rerun-if-changed=native/augur_qwen35_augur_prefill_norm.cuh");
    println!("cargo:rerun-if-changed=native/augur_qwen35_prefill_attention_policy.cuh");
    println!("cargo:rerun-if-changed=native/augur_qwen35_exact_projection_row_reuse.cuh");
    println!("cargo:rerun-if-changed=native/augur_qwen35_exact_ffn_down_row_reuse.cuh");
    println!("cargo:rerun-if-changed=native/augur_qwen35_nvfp4_grouped_decode.cuh");
    println!("cargo:rerun-if-changed=native/augur_qwen35_prefill_compensated_tensor.cuh");
    println!("cargo:rerun-if-changed=native/augur_qwen35_prefill_integer_mma.cuh");
    println!("cargo:rerun-if-changed=native/augur_qwen35_prefill_integer_q8.cuh");
    println!("cargo:rerun-if-changed=native/augur_qwen35_canonical_prefill.cuh");
    println!("cargo:rerun-if-changed=native/augur_qwen35_canonical_prefill_scores.cuh");
    println!("cargo:rerun-if-changed=native/augur_qwen35_vision_affine.cuh");
    println!("cargo:rerun-if-changed=native/augur_dflash_selector_gate_f32.cuh");
    println!("cargo:rerun-if-changed=native/augur_qwen35_vision_control.cuh");
    println!("cargo:rerun-if-changed=native/augur_qwen35_kv8_segment_v2_abi.h");
    for source in cuda_sources.iter().copied().chain(nvfp4_source) {
        println!("cargo:rerun-if-changed={}", source.path);
    }

    let target_os = required_env("CARGO_CFG_TARGET_OS");
    let target_env = required_env("CARGO_CFG_TARGET_ENV");

    if target_os != "windows" && target_os != "linux" {
        panic!("augur-cuda-sys: CUDA build supports only Windows and Linux");
    }

    if target_os == "windows" && target_env != "msvc" {
        panic!("augur-cuda-sys: Windows CUDA build requires the MSVC Rust target");
    }

    let cuda_root = cuda_root();
    let nvcc = env::var_os("AUGUR_CUDA_NVCC")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            cuda_root.join("bin").join(if target_os == "windows" {
                "nvcc.exe"
            } else {
                "nvcc"
            })
        });

    if !nvcc.is_file() {
        panic!(
            "augur-cuda-sys: nvcc not found at {}; set AUGUR_CUDA_NVCC or CUDA_PATH",
            nvcc.display()
        );
    }

    let out_dir = PathBuf::from(required_env("OUT_DIR"));

    let library = out_dir.join(if target_os == "windows" {
        "augur_cuda_runtime.lib"
    } else {
        "libaugur_cuda_runtime.a"
    });

    let gencode = env::var_os("AUGUR_CUDA_GENCODE").map(|value| value.to_string_lossy().into_owned());

    let mut objects = Vec::new();
    let mut compile_commands = Vec::new();
    for source in cuda_sources {
        let object = out_dir.join(object_file_name(source, &target_os));
        compile_commands.push(cuda_compile_command(
            &nvcc,
            source.path,
            &object,
            &cuda_root,
            &target_os,
            gencode.as_deref(),
        ));
        objects.push(object);
    }

    if let Some(source) = nvfp4_source {
        let object = out_dir.join(object_file_name(source, &target_os));
        if source == NVFP4_CUTLASS_SOURCE {
            let cutlass_root = cutlass_root
                .as_deref()
                .expect("CUTLASS source requires root");
            let header = cutlass_root.join("include/cutlass/cutlass.h");
            if !header.is_file() {
                panic!(
                    "augur-cuda-sys: AUGUR_CUTLASS_ROOT does not contain include/cutlass/cutlass.h: {}",
                    cutlass_root.display()
                );
            }
            compile_commands.push(nvfp4_cutlass_compile_command(
                &nvcc,
                &object,
                &cuda_root,
                cutlass_root,
                &target_os,
                gencode.as_deref(),
            ));
        } else {
            compile_commands.push(cuda_compile_command(
                &nvcc,
                source.path,
                &object,
                &cuda_root,
                &target_os,
                gencode.as_deref(),
            ));
        }
        objects.push(object);
    }

    run_compile_commands(compile_commands, jobserver);

    let mut archive = Command::new(&nvcc);
    archive.arg("--lib").args(&objects).arg("-o").arg(&library);

    if let Some(host_compiler) = env::var_os("AUGUR_CUDA_HOST_COMPILER") {
        archive.arg("-ccbin").arg(host_compiler);
    }

    run(&mut archive, "archive selected CUDA objects");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=augur_cuda_runtime");

    let cuda_lib_dir = if target_os == "windows" {
        cuda_root.join("lib").join("x64")
    } else {
        cuda_root.join("lib64")
    };

    println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    if target_os == "linux" {
        println!("cargo:rustc-link-lib=dylib=stdc++");
        println!("cargo:rustc-link-lib=dylib=dl");
    }
    if qwen35_prefill_cublas_enabled {
        println!("cargo:rustc-link-lib=dylib=cublas");
    }
}

fn publish_resolved_features() {
    let mut features = env::vars_os()
        .filter_map(|(name, _)| {
            name.to_str()
                .and_then(|name| name.strip_prefix("CARGO_FEATURE_"))
                .map(|name| name.to_ascii_lowercase().replace('_', "-"))
        })
        .collect::<Vec<_>>();
    features.sort();
    features.dedup();
    println!(
        "cargo:rustc-env=AUGUR_CUDA_SYS_RESOLVED_FEATURES={}",
        features.join(",")
    );
}

fn nvfp4_cutlass_compile_command(
    nvcc: &Path,
    object: &Path,
    cuda_root: &Path,
    cutlass_root: &Path,
    target_os: &str,
    gencode: Option<&str>,
) -> CompileCommand {
    const REQUIRED_GENCODE: &str = "arch=compute_120a,code=sm_120a";

    let mut compile = Command::new(nvcc);
    compile
        .arg("-c")
        .arg(NVFP4_CUTLASS_SOURCE.path)
        .arg("-o")
        .arg(object)
        .arg("-std=c++17")
        .arg("--expt-relaxed-constexpr")
        .arg("-I")
        .arg("native")
        .arg("-I")
        .arg(cuda_root.join("include"))
        .arg("-I")
        .arg(cutlass_root.join("include"))
        .arg("-I")
        .arg(cutlass_root.join("tools/util/include"));

    let code = gencode.unwrap_or(REQUIRED_GENCODE);
    if code != REQUIRED_GENCODE {
        panic!(
            "augur-cuda-sys: the real M52 CUTLASS NVFP4 path requires \
             AUGUR_CUDA_GENCODE={REQUIRED_GENCODE}; found {code}"
        );
    }
    compile.arg(format!("-gencode={code}"));
    if target_os == "windows" {
        compile
            .arg("-Xcompiler=/EHsc")
            .arg("-Xcompiler=/Zc:__cplusplus")
            .arg("-Xcompiler=/Zc:preprocessor")
            .arg("-Xcompiler=/bigobj");
        if let Some(host_compiler) = env::var_os("AUGUR_CUDA_HOST_COMPILER") {
            compile.arg("-ccbin").arg(host_compiler);
        }
    } else {
        compile.arg("-Xcompiler=-fPIC");
    }
    CompileCommand::new(compile, format!("compile {}", NVFP4_CUTLASS_SOURCE.path))
}

fn cuda_compile_command(
    nvcc: &Path,
    source: &str,
    object: &Path,
    cuda_root: &Path,
    target_os: &str,
    gencode: Option<&str>,
) -> CompileCommand {
    let mut compile = Command::new(nvcc);
    compile
        .arg("-c")
        .arg(source)
        .arg("-o")
        .arg(object)
        .arg(if source.starts_with("native/augur_qwen35_rotated_kv_")
            || source == "native/augur_qwen35_prefill_llama_mmq_sm70.cu"
            || source == "native/augur_qwen35_row_q8_input_mmq_sm70.cu"
            { "-std=c++17" } else { "-std=c++14" })
        .arg("-I")
        .arg("native")
        .arg("-I")
        .arg(cuda_root.join("include"));

    if let Some(code) = gencode {
        compile.arg(format!("-gencode={code}"));
    }
    if env::var_os("CARGO_FEATURE_M306_K15_Q5_PAIRED_PROJECTION").is_some() {
        compile.arg("-DAUGUR_CUDA_M306_K15_Q5_PAIRED_PROJECTION=1");
    }
    if env::var_os("CARGO_FEATURE_FAST_CUDA_SCHEDULE_SPIN_CANDIDATE").is_some() {
        compile.arg("-DAUGUR_CUDA_FAST_SCHEDULE_SPIN_CANDIDATE=1");
    }
    if env::var_os("CARGO_FEATURE_M306_K22_Q5_FP32_KAHAN").is_some() {
        compile.arg("-DAUGUR_CUDA_M306_K22_Q5_FP32_KAHAN=1");
    }
    if env::var_os("CARGO_FEATURE_M306_K17_BF16_WARP_ROW").is_some() {
        compile.arg("-DAUGUR_CUDA_M306_K17_BF16_WARP_ROW=1");
    }
    if env::var_os("CARGO_FEATURE_QWEN35_Q8_PACKED_LOAD").is_some() {
        compile.arg("-DAUGUR_CUDA_QWEN35_Q8_PACKED_LOAD=1");
    }
    if env::var_os("CARGO_FEATURE_QWEN35_DIAGNOSTIC_Q8_AWAY").is_some() {
        compile.arg("-DAUGUR_CUDA_QWEN35_DIAGNOSTIC_Q8_AWAY=1");
    }
    if env::var_os("CARGO_FEATURE_QWEN35_DIAGNOSTIC_SPLIT_ROPE").is_some()
        || env::var_os("CARGO_FEATURE_QWEN35_TARGET_SPLIT_ROPE").is_some()
    {
        compile.arg("-DAUGUR_CUDA_QWEN35_DIAGNOSTIC_SPLIT_ROPE=1");
    }
    if env::var_os("CARGO_FEATURE_QWEN35_NATIVE_F16_SCALE").is_some() {
        compile.arg("-DAUGUR_CUDA_QWEN35_NATIVE_F16_SCALE=1");
    }
    if env::var_os("CARGO_FEATURE_QWEN35_NVFP4_FUSED_LB10").is_some() {
        compile.arg("-DAUGUR_CUDA_QWEN35_NVFP4_FUSED_LB10=1");
    }
    if env::var_os("CARGO_FEATURE_QWEN35_GDN_WARP_ROW").is_some() {
        compile.arg("-DAUGUR_CUDA_QWEN35_GDN_WARP_ROW=1");
    }
    if env::var_os("CARGO_FEATURE_QWEN35_FUSED_HEAD_NORM_ROPE").is_some() {
        compile.arg("-DAUGUR_CUDA_QWEN35_FUSED_HEAD_NORM_ROPE=1");
    }
    if env::var_os("CARGO_FEATURE_QWEN35_WARP_FULL_ATTENTION").is_some() {
        compile.arg("-DAUGUR_CUDA_QWEN35_WARP_FULL_ATTENTION=1");
    }
    if env::var_os("CARGO_FEATURE_QWEN35_FULL_ATTENTION_PARALLEL_EXP").is_some() {
        compile.arg("-DAUGUR_CUDA_QWEN35_FULL_ATTENTION_PARALLEL_EXP=1");
    }
    if env::var_os("CARGO_FEATURE_QWEN35_GROUPED_GQA_ONLINE_SOFTMAX").is_some() {
        compile.arg("-DAUGUR_CUDA_QWEN35_GROUPED_GQA_ONLINE_SOFTMAX=1");
    }
    if env::var_os("CARGO_FEATURE_QWEN35_KV8_FULL_ATTENTION").is_some() {
        compile.arg("-DAUGUR_CUDA_QWEN35_KV8_FULL_ATTENTION=1");
    }
    if env::var_os("CARGO_FEATURE_QWEN35_FUSED_FFN_SWIGLU").is_some() {
        compile.arg("-DAUGUR_CUDA_QWEN35_FUSED_FFN_SWIGLU=1");
    }
    if env::var_os("CARGO_FEATURE_QWEN35_PREFILL_CUBLAS").is_some() {
        compile.arg("-DAUGUR_CUDA_QWEN35_PREFILL_CUBLAS=1");
    }
    if env::var_os("CARGO_FEATURE_QWEN35_VISION_AFFINE").is_some() {
        compile.arg("-DAUGUR_CUDA_QWEN35_VISION_AFFINE=1");
    }
    if env::var_os("CARGO_FEATURE_QWEN35_VISION_AFFINE").is_some() {
        compile.arg("-DAUGUR_CUDA_QWEN35_VISION_AFFINE=1");
    }
    if env::var_os("CARGO_FEATURE_QWEN35_AUGUR_LARGE_PREFILL").is_some() {
        compile.arg("-DAUGUR_CUDA_QWEN35_AUGUR_LARGE_PREFILL=1");
    }
    if env::var_os("CARGO_FEATURE_QWEN35_KV8_TOKEN_PAIR").is_some() {
        compile.arg("-DAUGUR_CUDA_QWEN35_KV8_TOKEN_PAIR=1");
    }
    if env::var_os("CARGO_FEATURE_QWEN35_TC_PREFIX_ATTENTION").is_some() {
        compile.arg("-DAUGUR_CUDA_QWEN35_TC_PREFIX_ATTENTION=1");
    }
    if env::var_os("CARGO_FEATURE_SINGLE_SEQUENCE_FAST_EXPERT_WARP_SHARED").is_some() {
        compile.arg("-DAUGUR_CUDA_SINGLE_SEQUENCE_FAST_EXPERT_WARP_SHARED=1");
    }
    if env::var_os("CARGO_FEATURE_SINGLE_SEQUENCE_FAST_EXPERT_DP4A").is_some() {
        compile.arg("-DAUGUR_CUDA_SINGLE_SEQUENCE_FAST_EXPERT_DP4A=1");
    }
    if env::var_os("CARGO_FEATURE_CTX6D_ROUTE_QUANTIZE_FUSED_CANDIDATE").is_some() {
        compile.arg("-DAUGUR_CUDA_CTX6D_ROUTE_QUANTIZE_FUSED_CANDIDATE=1");
    }
    if env::var_os("CARGO_FEATURE_CTX6D_EXPERT_GRAPH_CANDIDATE").is_some() {
        compile.arg("-DAUGUR_CUDA_CTX6D_EXPERT_GRAPH_CANDIDATE=1");
    }
    if env::var_os("CARGO_FEATURE_SINGLE_SEQUENCE_FAST_Q5_DP4A").is_some() {
        compile.arg("-DAUGUR_CUDA_SINGLE_SEQUENCE_FAST_Q5_DP4A=1");
    }
    // Word-pack candidates only change packed-byte reconstruction.  They keep
    // the existing shared staging and are deliberately opt-in so their
    // numerical identity can be qualified independently of the product path.
    let shared_word_pack = env::var_os("CARGO_FEATURE_FAST_EXPERT_WORD_PACK_V1").is_some()
        || env::var_os("CARGO_FEATURE_FAST_EXPERT_SHARED_WORD_PACK_CANDIDATE").is_some();
    if shared_word_pack {
        compile.arg("-DAUGUR_CUDA_SINGLE_SEQUENCE_FAST_EXPERT_SHARED_WORD_PACK=1");
    }
    let direct_q5_word_pack = env::var_os("CARGO_FEATURE_FAST_EXPERT_WORD_PACK_V1").is_some()
        || env::var_os("CARGO_FEATURE_FAST_EXPERT_Q5_DIRECT_WORD_PACK_CANDIDATE").is_some();
    if direct_q5_word_pack {
        compile.arg("-DAUGUR_CUDA_SINGLE_SEQUENCE_FAST_EXPERT_Q5_DIRECT_WORD_PACK=1");
    }
    if env::var_os("CARGO_FEATURE_SINGLE_SEQUENCE_FAST_EXPERT_WIDTH512").is_some() {
        compile.arg("-DAUGUR_CUDA_SINGLE_SEQUENCE_FAST_EXPERT_WIDTH512=1");
    }
    if env::var_os("CARGO_FEATURE_SINGLE_SEQUENCE_FAST_EXPERT_WIDTH768").is_some() {
        compile.arg("-DAUGUR_CUDA_SINGLE_SEQUENCE_FAST_EXPERT_WIDTH768=1");
    }
    if env::var_os("CARGO_FEATURE_SINGLE_SEQUENCE_FAST_EXPERT_WIDTH1024").is_some() {
        compile.arg("-DAUGUR_CUDA_SINGLE_SEQUENCE_FAST_EXPERT_WIDTH1024=1");
    }
    if env::var_os("CARGO_FEATURE_SINGLE_SEQUENCE_FAST_EXPERT_WIDTH1536").is_some() {
        compile.arg("-DAUGUR_CUDA_SINGLE_SEQUENCE_FAST_EXPERT_WIDTH1536=1");
    }
    let ffn_q1_width_overrides = [
        ("CARGO_FEATURE_FFN_Q1_WIDTH2048", "AUGUR_CUDA_FFN_Q1_WIDTH2048"),
        ("CARGO_FEATURE_FFN_Q1_WIDTH1536", "AUGUR_CUDA_FFN_Q1_WIDTH1536"),
        ("CARGO_FEATURE_FFN_Q1_WIDTH1024", "AUGUR_CUDA_FFN_Q1_WIDTH1024"),
        ("CARGO_FEATURE_FFN_Q1_WIDTH768", "AUGUR_CUDA_FFN_Q1_WIDTH768"),
        ("CARGO_FEATURE_FFN_Q1_WIDTH512", "AUGUR_CUDA_FFN_Q1_WIDTH512"),
    ];
    let selected_ffn_q1_widths = ffn_q1_width_overrides
        .iter()
        .filter(|(feature, _)| env::var_os(feature).is_some())
        .collect::<Vec<_>>();
    if selected_ffn_q1_widths.len() > 1 {
        panic!("FFN-Q1 width overrides are mutually exclusive");
    }
    if let Some((_, define)) = selected_ffn_q1_widths.first() {
        compile.arg(format!("-D{define}=1"));
    }
    if env::var_os("CARGO_FEATURE_SINGLE_SEQUENCE_FAST_KV_WINDOW32").is_some() {
        compile.arg("-DAUGUR_CUDA_SINGLE_SEQUENCE_FAST_KV_WINDOW32=1");
    }
    if env::var_os("CARGO_FEATURE_SINGLE_SEQUENCE_FAST_KV_WINDOW16").is_some() {
        compile.arg("-DAUGUR_CUDA_SINGLE_SEQUENCE_FAST_KV_WINDOW16=1");
    }
    if env::var_os("CARGO_FEATURE_SINGLE_SEQUENCE_FAST_KV_WINDOW8").is_some() {
        compile.arg("-DAUGUR_CUDA_SINGLE_SEQUENCE_FAST_KV_WINDOW8=1");
    }
    if env::var_os("CARGO_FEATURE_SINGLE_SEQUENCE_FAST_Q6_DP4A").is_some() {
        compile.arg("-DAUGUR_CUDA_SINGLE_SEQUENCE_FAST_Q6_DP4A=1");
    }
    if env::var_os("CARGO_FEATURE_SINGLE_SEQUENCE_FAST_Q6_HIGH2_DP4A").is_some() {
        compile.arg("-DAUGUR_CUDA_SINGLE_SEQUENCE_FAST_Q6_HIGH2_DP4A=1");
    }
    if env::var_os("CARGO_FEATURE_SINGLE_SEQUENCE_FAST_Q6_HIGH2_WIDTH512_DP4A").is_some() {
        // The width512 successor reuses the proven high2 unpacker, but keeps
        // a distinct Cargo/runtime identity from the old V27 component.
        compile.arg("-DAUGUR_CUDA_SINGLE_SEQUENCE_FAST_Q6_HIGH2_DP4A=1");
    }
    if env::var_os("CARGO_FEATURE_SINGLE_SEQUENCE_FAST_Q6_HIGH2_REGISTER_DP4A").is_some() {
        compile.arg("-DAUGUR_CUDA_SINGLE_SEQUENCE_FAST_Q6_HIGH2_REGISTER_DP4A=1");
    }
    if env::var_os("CARGO_FEATURE_SINGLE_SEQUENCE_FAST_RATIO4_FUSED").is_some() {
        compile.arg("-DAUGUR_CUDA_SINGLE_SEQUENCE_FAST_RATIO4_FUSED=1");
    }
    if env::var_os("CARGO_FEATURE_CTX12_RATIO128_SKIP_ROW_VALIDATION_CANDIDATE").is_some() {
        compile.arg("-DAUGUR_CUDA_CTX12_RATIO128_SKIP_ROW_VALIDATION_CANDIDATE=1");
    }
    if env::var_os("CARGO_FEATURE_CTX13_RATIO128_HEADS8_CANDIDATE").is_some() {
        compile.arg("-DAUGUR_CUDA_CTX13_RATIO128_HEADS8_CANDIDATE=1");
    }
    if env::var_os("CARGO_FEATURE_CTX23_RATIO128_PARALLEL_DEQUANT_CANDIDATE").is_some() {
        compile.arg("-DAUGUR_CUDA_CTX23_RATIO128_PARALLEL_DEQUANT_CANDIDATE=1");
    }
    if env::var_os("CARGO_FEATURE_CTX28_RATIO128_SCALE_PLAN_MERGE_CANDIDATE").is_some() {
        compile.arg("-DAUGUR_CUDA_CTX28_RATIO128_SCALE_PLAN_MERGE_CANDIDATE=1");
    }
    if env::var_os("CARGO_FEATURE_P4_EXPERT_MICROSLICE_GPU_REPACK").is_some() {
        compile.arg("-DAUGUR_CUDA_P4_EXPERT_MICROSLICE_GPU_REPACK=1");
    }
    if env::var_os("CARGO_FEATURE_CTX30_PAIRED_Q4_DP4A_CANDIDATE").is_some() {
        compile.arg("-DAUGUR_CUDA_CTX30_PAIRED_Q4_DP4A_CANDIDATE=1");
    }
    if env::var_os("CARGO_FEATURE_CTX34_EXPERT_Q5_RANK6_FUSED_CANDIDATE").is_some() {
        compile.arg("-DAUGUR_CUDA_CTX34_EXPERT_Q5_RANK6_FUSED_CANDIDATE=1");
    }
    if env::var_os("CARGO_FEATURE_CTX36_EXPERT_DOUBLE_BUFFER_CANDIDATE").is_some() {
        compile.arg("-DAUGUR_CUDA_CTX36_EXPERT_DOUBLE_BUFFER_CANDIDATE=1");
    }
    if env::var_os("CARGO_FEATURE_CTX37_RATIO128_DISTRIBUTED_ROW_STAGE_CANDIDATE").is_some() {
        compile.arg("-DAUGUR_CUDA_CTX37_RATIO128_DISTRIBUTED_ROW_STAGE_CANDIDATE=1");
    }
    if env::var_os("CARGO_FEATURE_CTX38_RATIO128_DIRECT_DEQUANT_CANDIDATE").is_some() {
        compile.arg("-DAUGUR_CUDA_CTX38_RATIO128_DIRECT_DEQUANT_CANDIDATE=1");
    }
    if env::var_os("CARGO_FEATURE_CTX39_RATIO128_SHORT_HEADS4_CANDIDATE").is_some() {
        // CTX39 changes only the Rust dispatch topology; the existing H4 CUDA
        // entry point is already compiled by the retained CTX28 path.
        compile.arg("-DAUGUR_CUDA_CTX39_RATIO128_SHORT_HEADS4_CANDIDATE=1");
    }
    if env::var_os("CARGO_FEATURE_CTX40_RATIO128_VECTOR_DIRECT_PINGPONG_CANDIDATE").is_some() {
        compile.arg("-DAUGUR_CUDA_CTX40_RATIO128_VECTOR_DIRECT_PINGPONG_CANDIDATE=1");
    }
    if env::var_os("CARGO_FEATURE_CTX41_RATIO128_REGISTER_DIRECT_CANDIDATE").is_some() {
        compile.arg("-DAUGUR_CUDA_CTX41_RATIO128_REGISTER_DIRECT_CANDIDATE=1");
    }
    if env::var_os("CARGO_FEATURE_CTX42_RATIO128_STRIDED_REGISTER_CANDIDATE").is_some() {
        compile.arg("-DAUGUR_CUDA_CTX42_RATIO128_STRIDED_REGISTER_CANDIDATE=1");
    }
    if env::var_os("CARGO_FEATURE_CTX44_RATIO128_H2_STRIDED_REGISTER_CANDIDATE").is_some() {
        compile.arg("-DAUGUR_CUDA_CTX44_RATIO128_H2_STRIDED_REGISTER_CANDIDATE=1");
    }
    if env::var_os("CARGO_FEATURE_CTX49_RATIO4_H2_WARP_LOCAL_ROW_CANDIDATE").is_some() {
        compile.arg("-DAUGUR_CUDA_CTX49_RATIO4_H2_WARP_LOCAL_ROW_CANDIDATE=1");
    }
    if env::var_os("CARGO_FEATURE_CTX15_RATIO4_SKIP_ROW_VALIDATION_CANDIDATE").is_some() {
        compile.arg("-DAUGUR_CUDA_CTX15_RATIO4_SKIP_ROW_VALIDATION_CANDIDATE=1");
    }
    if env::var_os("CARGO_FEATURE_CTX32_KV8_DSA_EXACT_H2_CANDIDATE").is_some() {
        compile.arg("-DAUGUR_CUDA_CTX32_KV8_DSA_EXACT_H2_CANDIDATE=1");
    }
    if env::var_os("CARGO_FEATURE_CTX33_KV8_DSA_EXACT_ROW_STAGE_CANDIDATE").is_some() {
        compile.arg("-DAUGUR_CUDA_CTX33_KV8_DSA_EXACT_ROW_STAGE_CANDIDATE=1");
    }
    if env::var_os("CARGO_FEATURE_KV8_DSA_RATIO4_HOST_TIER_CANDIDATE").is_some() {
        compile.arg("-DAUGUR_CUDA_KV8_DSA_RATIO4_HOST_TIER_CANDIDATE=1");
    }
    if env::var_os("CARGO_FEATURE_CTX17_RATIO4_WARP_TOPK_CANDIDATE").is_some() {
        compile.arg("-DAUGUR_CUDA_CTX17_RATIO4_WARP_TOPK_CANDIDATE=1");
    }
    if env::var_os("CARGO_FEATURE_CTX18_RATIO4_DISTRIBUTED_TOPK_CANDIDATE").is_some() {
        compile.arg("-DAUGUR_CUDA_CTX18_RATIO4_DISTRIBUTED_TOPK_CANDIDATE=1");
    }
    if env::var_os("CARGO_FEATURE_CTX31_RATIO4_COOPERATIVE_SELECTOR_CANDIDATE").is_some() {
        compile.arg("-DAUGUR_CUDA_CTX31_RATIO4_COOPERATIVE_SELECTOR_CANDIDATE=1");
    }
    if env::var_os("CARGO_FEATURE_FAST_EXPERT_Q4_UNROLL_CANDIDATE").is_some() {
        compile.arg("-DAUGUR_CUDA_SINGLE_SEQUENCE_FAST_EXPERT_Q4_UNROLL_CANDIDATE=1");
    }
    if env::var_os("CARGO_FEATURE_FAST_EXPERT_CTA256_CANDIDATE").is_some() {
        compile.arg("-DAUGUR_CUDA_SINGLE_SEQUENCE_FAST_EXPERT_CTA256_CANDIDATE=1");
    }
    if env::var_os("CARGO_FEATURE_FAST_EXPERT_REGISTER_STAGING_CANDIDATE").is_some() {
        compile.arg("-DAUGUR_CUDA_SINGLE_SEQUENCE_FAST_EXPERT_REGISTER_STAGING_CANDIDATE=1");
    }
    if env::var_os("CARGO_FEATURE_FAST_EXPERT_DUAL_ROW_CANDIDATE").is_some() {
        compile.arg("-DAUGUR_CUDA_SINGLE_SEQUENCE_FAST_EXPERT_DUAL_ROW_CANDIDATE=1");
    }
    let promoted_word_pack = env::var_os("CARGO_FEATURE_FAST_EXPERT_WORD_PACK_V1").is_some();
    if promoted_word_pack
        || env::var_os("CARGO_FEATURE_FAST_EXPERT_SHARED_WORD_PACK_CANDIDATE").is_some()
    {
        compile.arg("-DAUGUR_CUDA_SINGLE_SEQUENCE_FAST_EXPERT_SHARED_WORD_PACK=1");
    }
    if promoted_word_pack
        || env::var_os("CARGO_FEATURE_FAST_EXPERT_Q5_DIRECT_WORD_PACK_CANDIDATE").is_some()
    {
        compile.arg("-DAUGUR_CUDA_SINGLE_SEQUENCE_FAST_EXPERT_Q5_DIRECT_WORD_PACK=1");
    }
    if env::var_os("CARGO_FEATURE_BATCH01_B8_Q5_ROW_TILE").is_some() {
        compile.arg("-DAUGUR_CUDA_BATCH01_B8_Q5_ROW_TILE=1");
    }
    if env::var_os("CARGO_FEATURE_BATCH01_B4_Q6_ROW_REUSE").is_some() {
        compile.arg("-DAUGUR_CUDA_BATCH01_B4_Q6_ROW_REUSE=1");
    }
    if env::var_os("CARGO_FEATURE_BATCH01_B4_Q6_BLOCK_DIAGONAL_ROW_REUSE").is_some() {
        compile.arg("-DAUGUR_CUDA_BATCH01_B4_Q6_BLOCK_DIAGONAL_ROW_REUSE=1");
    }
    if env::var_os("CARGO_FEATURE_DSPARK_FAST_Q6_ROW_REUSE").is_some() {
        compile.arg("-DAUGUR_CUDA_DSPARK_FAST_Q6_ROW_REUSE=1");
    }
    if env::var_os("CARGO_FEATURE_SINGLE_SEQUENCE_FAST_PREFILL_Q6_WMMA_M16").is_some() {
        compile.arg("-DAUGUR_CUDA_SINGLE_SEQUENCE_FAST_PREFILL_Q6_WMMA_M16=1");
    }
    if env::var_os("CARGO_FEATURE_SINGLE_SEQUENCE_FAST_PREFILL_Q6_STAGED_WMMA_M16").is_some() {
        compile.arg("-DAUGUR_CUDA_SINGLE_SEQUENCE_FAST_PREFILL_Q6_STAGED_WMMA_M16=1");
    }
    if env::var_os("CARGO_FEATURE_SINGLE_SEQUENCE_FAST_PREFILL_Q6_STAGED_WMMA_M128").is_some() {
        compile.arg("-DAUGUR_CUDA_SINGLE_SEQUENCE_FAST_PREFILL_Q6_STAGED_WMMA_M128=1");
    }
    if env::var_os("CARGO_FEATURE_SINGLE_SEQUENCE_FAST_PREFILL_Q6_STAGED_WMMA_M128_ROWGROUP2")
        .is_some()
    {
        compile.arg("-DAUGUR_CUDA_SINGLE_SEQUENCE_FAST_PREFILL_Q6_STAGED_WMMA_M128_ROWGROUP2=1");
    }
    if env::var_os("CARGO_FEATURE_SINGLE_SEQUENCE_FAST_PREFILL_Q6_STAGED_WMMA_M128_ROWGROUP4")
        .is_some()
    {
        compile.arg("-DAUGUR_CUDA_SINGLE_SEQUENCE_FAST_PREFILL_Q6_STAGED_WMMA_M128_ROWGROUP4=1");
    }
    if env::var_os("CARGO_FEATURE_SINGLE_SEQUENCE_FAST_PREFILL_ATTENTION_PRELUDE_M128").is_some() {
        compile.arg("-DAUGUR_CUDA_SINGLE_SEQUENCE_FAST_PREFILL_ATTENTION_PRELUDE_M128=1");
    }
    // RowGroup4's four WMMA accumulators otherwise trigger a one-word ptxas
    // spill on SM70. Keep the resource contract explicit while staying below
    // the candidate's 96-register/thread ceiling.
    if source == "native/augur_m146_r1_attention_sm70.cu"
        && env::var_os("CARGO_FEATURE_SINGLE_SEQUENCE_FAST_PREFILL_Q6_STAGED_WMMA_M128_ROWGROUP4")
            .is_some()
    {
        compile.arg("-maxrregcount=96");
    }
    if env::var_os("CARGO_FEATURE_SINGLE_SEQUENCE_FAST_PREFILL_ROUTE_RECEIPT_ONLY").is_some() {
        compile.arg("-DAUGUR_CUDA_SINGLE_SEQUENCE_FAST_PREFILL_ROUTE_RECEIPT_ONLY=1");
    }
    if env::var_os("CARGO_FEATURE_SINGLE_SEQUENCE_FAST_PREFILL_ROUTE_ROW8_V14").is_some() {
        compile.arg("-DAUGUR_CUDA_SINGLE_SEQUENCE_FAST_PREFILL_ROUTE_ROW8_V14=1");
    }
    if env::var_os("CARGO_FEATURE_P0_PREFILL_V21_RATIO4_FAST_EXACT_QDQ").is_some() {
        compile.arg("-DAUGUR_CUDA_P0_PREFILL_V21_RATIO4_FAST_EXACT_QDQ=1");
    }
    if env::var_os("CARGO_FEATURE_P0_PREFILL_V22_PROJECTION_RESUME_BATCH_EXACT").is_some() {
        compile.arg("-DAUGUR_CUDA_P0_PREFILL_V22_PROJECTION_RESUME_BATCH_EXACT=1");
    }
    if env::var_os("CARGO_FEATURE_P0_PREFILL_V23_PROJECTION_POST_BATCH_EXACT").is_some() {
        compile.arg("-DAUGUR_CUDA_P0_PREFILL_V23_PROJECTION_POST_BATCH_EXACT=1");
    }
    if env::var_os("CARGO_FEATURE_P0_PREFILL_V24_RESIDENT_WOA_M128").is_some() {
        compile.arg("-DAUGUR_CUDA_P0_PREFILL_V24_RESIDENT_WOA_M128=1");
    }
    if env::var_os("CARGO_FEATURE_P0_PREFILL_V27_Q5_ROW128_REUSE").is_some() {
        compile.arg("-DAUGUR_CUDA_P0_PREFILL_V27_Q5_ROW128_REUSE=1");
    }
    if env::var_os("CARGO_FEATURE_P0_PREFILL_V28_RAW_RATIO128_PROJECTION_POST_BATCH_EXACT")
        .is_some()
    {
        compile.arg("-DAUGUR_CUDA_P0_PREFILL_V28_RAW_RATIO128_PROJECTION_POST_BATCH_EXACT=1");
    }
    if env::var_os("CARGO_FEATURE_P0_PREFILL_V29_Q5_BLOCK_STAGING").is_some() {
        compile.arg("-DAUGUR_CUDA_P0_PREFILL_V29_Q5_BLOCK_STAGING=1");
    }
    if env::var_os("CARGO_FEATURE_P0_PREFILL_V30_Q5_INPUT_BROADCAST").is_some() {
        compile.arg("-DAUGUR_CUDA_P0_PREFILL_V30_Q5_INPUT_BROADCAST=1");
    }
    if env::var_os("CARGO_FEATURE_P0_PREFILL_V31_CTA_OCCUPANCY_FILL").is_some() {
        compile.arg("-DAUGUR_CUDA_P0_PREFILL_V31_CTA_OCCUPANCY_FILL=1");
    }
    if env::var_os("CARGO_FEATURE_P0_PREFILL_V32_Q5_CTA8").is_some() {
        compile.arg("-DAUGUR_CUDA_P0_PREFILL_V32_Q5_CTA8=1");
    }
    if env::var_os("CARGO_FEATURE_P0_PREFILL_V33_Q5_CTA8_GRID192").is_some() {
        compile.arg("-DAUGUR_CUDA_P0_PREFILL_V33_Q5_CTA8_GRID192=1");
    }
    if env::var_os("CARGO_FEATURE_P0_PREFILL_V34_Q8_QUANTIZE_SHARD4").is_some() {
        compile.arg("-DAUGUR_CUDA_P0_PREFILL_V34_Q8_QUANTIZE_SHARD4=1");
    }
    if env::var_os("CARGO_FEATURE_P0_PREFILL_V36_M1024_LAYER_UNION_WHOLE_FFN").is_some() {
        compile.arg("-DAUGUR_CUDA_P0_PREFILL_V36_M1024_LAYER_UNION_WHOLE_FFN=1");
    }
    if env::var_os("CARGO_FEATURE_P0_PREFILL_V37_M1024_Q8_FAST_UNION").is_some() {
        compile.arg("-DAUGUR_CUDA_P0_PREFILL_V37_M1024_Q8_FAST_UNION=1");
    }
    if env::var_os("CARGO_FEATURE_BLACKWELL_NVFP4").is_some() {
        compile.arg("-DAUGUR_CUDA_BLACKWELL_NVFP4=1");
    }

    if source == "native/augur_m304_r3_expert_route_sm70.cu"
        || source == "native/augur_m306_k11_hc_exact_sm70.cu"
        || source == "native/augur_m306_k12_learned_router_exact_sm70.cu"
    {
        compile.arg("--ftz=false");
    }
    if source == "native/augur_m306_k11_hc_exact_sm70.cu" {
        compile.arg("--fmad=false");
    }
    if source == "native/augur_qwen35_linear_step_sm70.cu"
        || source == "native/augur_qwen35_prefill_cublas_sm70.cu"
        || source == "native/augur_qwen35_kv8_segment_v2_sm70.cu"
    {
        // The page producer shares the V1 numerical identity: implicit FMA
        // contraction changes online-softmax recurrence across page cuts.
        compile.arg("--fmad=false").arg("--ftz=false");
    }

    // Match only isolated opt-in llama SwiGLU and exact-input hidden5120 norm.
    // Never apply this to other projection/norm/GDN/attention/residual paths.
    if source == "native/augur_qwen35_prefill_llama_activation_sm70.cu" {
        compile.arg("--use_fast_math");
    }
    // The actual llama MMQ family has a separate fast-math identity. Keep the
    // original-input MMVQ/GEMM and all default arithmetic flags unchanged.
    if source == "native/augur_qwen35_prefill_llama_mmq_sm70.cu"
        || source == "native/augur_qwen35_row_q8_input_mmq_sm70.cu" {
        compile.arg("--use_fast_math").arg("--extended-lambda")
            .arg("-DGGML_CUDA_PEER_MAX_BATCH_SIZE=128").arg("-DNDEBUG");
        if target_os == "windows" {
            compile.arg("-Xcompiler=/Zc:preprocessor");
        }
    }
    // Separate opt-in arithmetic identity; never changes existing V3/V4 kernels.
    if source.starts_with("native/augur_qwen35_rotated_kv_") {
        compile.arg("--use_fast_math").arg("--extended-lambda")
            .arg("-DGGML_CUDA_PEER_MAX_BATCH_SIZE=128").arg("-DNDEBUG");
        if target_os == "windows" {
            compile.arg("-Xcompiler=/Zc:preprocessor");
        }
    }

    if target_os == "windows" {
        compile.arg("-Xcompiler=/EHsc");

        if let Some(host_compiler) = env::var_os("AUGUR_CUDA_HOST_COMPILER") {
            compile.arg("-ccbin").arg(host_compiler);
        }
    } else {
        compile.arg("-Xcompiler=-fPIC");
    }

    CompileCommand::new(compile, format!("compile {source}"))
}

struct CompileCommand {
    command: Command,
    action: String,
}

impl CompileCommand {
    fn new(command: Command, action: impl Into<String>) -> Self {
        Self {
            command,
            action: action.into(),
        }
    }
}

struct RunningCommand {
    index: usize,
    action: String,
    rendered: String,
    child: Child,
    uses_implicit_token: bool,
    _token: Option<jobserver::Acquired>,
}

struct CommandOutcome {
    index: usize,
    action: String,
    rendered: String,
    status: Result<ExitStatus, String>,
}

fn inherited_jobserver() -> Option<jobserver::Client> {
    // SAFETY: this is called before the build script starts threads or child
    // processes, so the inherited jobserver handles cannot race with other users.
    unsafe { jobserver::Client::from_env() }
}

fn run_compile_commands(commands: Vec<CompileCommand>, jobserver: Option<jobserver::Client>) {
    if commands.len() < 2 || jobserver.is_none() {
        for mut compile in commands {
            run(&mut compile.command, &compile.action);
        }
        return;
    }

    let jobserver = jobserver.expect("checked above");
    let mut pending = commands.into_iter().enumerate().collect::<VecDeque<_>>();
    let mut running = Vec::<RunningCommand>::new();
    let mut outcomes = Vec::<CommandOutcome>::new();
    let mut implicit_token_available = true;
    let mut jobserver_available = true;
    let mut stop_spawning = false;

    while !pending.is_empty() || !running.is_empty() {
        while !stop_spawning && !pending.is_empty() {
            let (token, uses_implicit_token) = if implicit_token_available {
                implicit_token_available = false;
                (None, true)
            } else if jobserver_available {
                match jobserver.try_acquire() {
                    Ok(Some(token)) => (Some(token), false),
                    Ok(None) => break,
                    Err(error) => {
                        println!(
                            "cargo:warning=augur-cuda-sys: disabling parallel native build: {error}"
                        );
                        jobserver_available = false;
                        break;
                    }
                }
            } else {
                break;
            };

            let (index, mut compile) = pending.pop_front().expect("pending command");
            let rendered = render_command(&compile.command);
            match compile.command.spawn() {
                Ok(child) => running.push(RunningCommand {
                    index,
                    action: compile.action,
                    rendered,
                    child,
                    uses_implicit_token,
                    _token: token,
                }),
                Err(error) => {
                    implicit_token_available |= uses_implicit_token;
                    outcomes.push(CommandOutcome {
                        index,
                        action: compile.action,
                        rendered,
                        status: Err(error.to_string()),
                    });
                    stop_spawning = true;
                    pending.clear();
                    break;
                }
            }
        }

        if running.is_empty() {
            continue;
        }

        let mut completed = None;
        for (running_index, command) in running.iter_mut().enumerate() {
            match command.child.try_wait() {
                Ok(Some(status)) => {
                    completed = Some((running_index, Ok(status)));
                    break;
                }
                Ok(None) => {}
                Err(error) => {
                    let status = command.child.wait().map_err(|wait_error| {
                        format!("{error}; follow-up wait also failed: {wait_error}")
                    });
                    completed = Some((running_index, status));
                    break;
                }
            }
        }

        let Some((running_index, status)) = completed else {
            thread::sleep(Duration::from_millis(10));
            continue;
        };

        let command = running.remove(running_index);
        implicit_token_available |= command.uses_implicit_token;
        if status.as_ref().map_or(true, |status| !status.success()) {
            stop_spawning = true;
            pending.clear();
        }
        outcomes.push(CommandOutcome {
            index: command.index,
            action: command.action,
            rendered: command.rendered,
            status,
        });
    }

    outcomes.sort_by_key(|outcome| outcome.index);
    for outcome in outcomes {
        match outcome.status {
            Ok(status) if status.success() => {}
            Ok(status) => panic!(
                "augur-cuda-sys: failed to {}; status: {}; command: {}",
                outcome.action, status, outcome.rendered
            ),
            Err(error) => panic!(
                "augur-cuda-sys: failed to {}: {}; command: {}",
                outcome.action, error, outcome.rendered
            ),
        }
    }
}

fn required_env(name: &str) -> String {
    env::var(name).unwrap_or_else(|_| panic!("augur-cuda-sys: missing Cargo variable {name}"))
}

fn cuda_root() -> PathBuf {
    env::var_os("CUDA_PATH")
        .or_else(|| env::var_os("CUDA_HOME"))
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            panic!("augur-cuda-sys: CUDA_PATH or CUDA_HOME must be set when feature `cuda` is enabled")
        })
}

fn run(command: &mut Command, action: &str) {
    let rendered = render_command(command);
    let status = command.status().unwrap_or_else(|error| {
        panic!(
            "augur-cuda-sys: failed to {}: {}; command: {}",
            action, error, rendered
        )
    });

    if !status.success() {
        panic!(
            "augur-cuda-sys: failed to {}; status: {}; command: {}",
            action, status, rendered
        );
    }
}

fn render_command(command: &Command) -> String {
    let mut parts = Vec::<OsString>::new();
    parts.push(command.get_program().to_os_string());
    parts.extend(command.get_args().map(OsString::from));

    parts
        .iter()
        .map(|part| part.to_string_lossy())
        .collect::<Vec<_>>()
        .join(" ")
}
