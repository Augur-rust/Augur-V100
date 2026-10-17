#[path = "../build_plan.rs"]
mod build_plan;

use build_plan::{
    known_cuda_sources, object_file_name, public_feature_env_is_allowed,
    selected_cuda_sources_public, selected_nvfp4_source, selected_rerun_env_vars,
    NVFP4_CUTLASS_SOURCE, NVFP4_STUB_SOURCE, PUBLIC_FEATURE_ENV,
};
use std::{collections::HashSet, path::Path};

#[test]
fn environment_invalidation_follows_cuda_activation() {
    assert!(selected_rerun_env_vars(false, false)
        .collect::<Vec<_>>()
        .is_empty());
    assert!(selected_rerun_env_vars(false, true)
        .collect::<Vec<_>>()
        .is_empty());
    assert_eq!(
        selected_rerun_env_vars(true, false).collect::<Vec<_>>(),
        [
            "CUDA_PATH",
            "CUDA_HOME",
            "AUGUR_CUDA_NVCC",
            "AUGUR_CUDA_HOST_COMPILER",
            "AUGUR_CUDA_GENCODE"
        ]
    );
    assert_eq!(
        selected_rerun_env_vars(true, true).last(),
        Some("AUGUR_CUTLASS_ROOT")
    );
}

#[test]
fn base_cuda_selection_is_exactly_general_math_and_shared_runtime() {
    let paths: Vec<_> = selected_cuda_sources_public()
        .into_iter()
        .map(|s| s.path)
        .collect();
    assert_eq!(
        paths,
        [
            "native/augur_cuda_runtime.cu",
            "native/augur_q4k_cuda.cu",
            "native/augur_add_cuda.cu",
            "native/augur_gqa_repeat_cuda.cu",
            "native/augur_m120_qk_exact_sm70.generated.cu",
            "native/augur_m121_q4k_dp4a_sm70.cu",
            "native/augur_nvfp4_sm70.cu"
        ]
    );
}

#[test]
fn frozen_qwen_and_general_features_are_accepted_exactly() {
    let unique: HashSet<_> = PUBLIC_FEATURE_ENV.iter().copied().collect();
    assert_eq!(unique.len(), PUBLIC_FEATURE_ENV.len());
    for name in PUBLIC_FEATURE_ENV {
        assert!(name.starts_with("CARGO_FEATURE_"));
        assert!(public_feature_env_is_allowed(name));
        assert!(!public_feature_env_is_allowed(&format!("{name}_UNKNOWN")));
    }
    for name in [
        "CARGO_FEATURE_CUDA",
        "CARGO_FEATURE_QWEN35_LINEAR_STEP",
        "CARGO_FEATURE_M306_K17_BF16_WARP_ROW",
    ] {
        assert!(public_feature_env_is_allowed(name));
    }
}

#[test]
fn private_unknown_and_unsupported_feature_admission_fails_closed() {
    for name in [
        "CARGO_FEATURE_SINGLE_SEQUENCE_FAST_ROUTE",
        "CARGO_FEATURE_REQUEST_MLA_1K",
        "CARGO_FEATURE_DSPARK_FAST_Q6_ROW_REUSE",
        "CARGO_FEATURE_P4_EXPERT_MICROSLICE_GPU_REPACK",
        "CARGO_FEATURE_BLACKWELL_NVFP4",
        "CARGO_FEATURE_UNRECOGNIZED",
        "cuda",
        "",
    ] {
        assert!(!public_feature_env_is_allowed(name), "{name}");
    }
}

#[test]
fn feature_environment_guard_precedes_cpu_early_return() {
    let source = include_str!("../build.rs");
    let check = source
        .find("build_plan::public_feature_env_is_allowed(&name)")
        .unwrap();
    assert!(check < source.find("if !cuda_enabled").unwrap());
    assert!(!source.contains("selected_cuda_sources_with_prefill_cublas"));
}

#[test]
fn known_sources_exist_and_source_map_is_unique() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let sources = known_cuda_sources();
    assert_eq!(sources.len(), 19);
    assert_eq!(
        sources.iter().map(|s| s.path).collect::<HashSet<_>>().len(),
        sources.len()
    );
    for source in sources {
        assert!(root.join(source.path).is_file(), "{}", source.path);
        assert!(!source.path.contains("width6"));
        assert!(!source.path.contains("causal_mla"));
        assert!(!source.path.contains("learned_router"));
    }
}

#[test]
fn general_nvfp4_implementation_selection_and_object_names_are_preserved() {
    assert_eq!(selected_nvfp4_source(false, false), None);
    assert_eq!(selected_nvfp4_source(false, true), None);
    assert_eq!(selected_nvfp4_source(true, false), Some(NVFP4_STUB_SOURCE));
    assert_eq!(
        selected_nvfp4_source(true, true),
        Some(NVFP4_CUTLASS_SOURCE)
    );
    for extra in [None, Some(NVFP4_STUB_SOURCE), Some(NVFP4_CUTLASS_SOURCE)] {
        let mut sources = selected_cuda_sources_public();
        sources.extend(extra);
        for os in ["windows", "linux"] {
            assert_eq!(
                sources
                    .iter()
                    .map(|s| object_file_name(*s, os))
                    .collect::<HashSet<_>>()
                    .len(),
                sources.len()
            );
        }
    }
    assert_eq!(
        object_file_name(NVFP4_STUB_SOURCE, "windows"),
        "nv_nvfp4.obj"
    );
    assert_eq!(object_file_name(NVFP4_STUB_SOURCE, "linux"), "nv_nvfp4.o");
}
