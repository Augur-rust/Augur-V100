//! AVX2 kernels.
//!
//! M27:
//! - Q4K × f32 row-dot primitive;
//! - complete Q4K × f32 Identity QMatmul;
//! - registration and ExecutionPlan integration.
//!
//! M28-B:
//! - DualScale × Q8Hot block/row main-path primitives;
//! - no registration;
//! - no lowering;
//! - no ExecutionPlan routing.

mod dual_scale_q8hot;
mod q4k_f32;

pub use dual_scale_q8hot::{
    dot_dual_scale_block_q8k_hot_avx2_prevalidated, dot_dual_scale_hot_q8k_hot_avx2_prevalidated,
    launch_qmatvec_dual_scale_q8k_hot_cpu_avx2,
    launch_qmatvec_dual_scale_q8k_hot_cpu_avx2_prevalidated, qmatvec_dual_scale_q8k_hot_cpu_avx2,
};

pub use q4k_f32::{
    dot_q4k_hot_f32_avx2_prevalidated, launch_qmatmul_q4k_f32_avx2, qmatmul_q4k_f32_avx2,
};
