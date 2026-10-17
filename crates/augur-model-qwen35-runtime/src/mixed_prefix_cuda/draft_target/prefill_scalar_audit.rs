//! Opt-in native session experiment. Align prefill GDN alpha/beta to decode.
//! Reuses canonical BF16-weight/F32-input matvec without activation Q8 loss.
//! Warp-row builds batch independent token rows; scalar bindings are retained.
//! Does not batch/reorder the recurrent update or change decode/verifier paths.
use super::*;

pub(super) fn install(d: &Driver, mode: &mut AutoregressivePrefillMode) -> Result<Value> {
    let AutoregressivePrefillMode::Batched(prefill) = mode else {
        return Err(invalid(
            "canonical GDN scalar audit requires batched prefill",
        ));
    };
    if prefill.direct_activation
        || prefill.tensor_core_prefix_attention
        || prefill.workspace.max_tokens != 512
        || d.plans.len() != 64
    {
        return Err(invalid("canonical scalar audit frozen prefill geometry"));
    }
    let w = &prefill.workspace;
    let mut rows_total = 0usize;
    let mut bound_ops = 0usize;
    for chunk in &mut prefill.schedule.chunks {
        if chunk.tokens == 0 || chunk.tokens > 512 {
            return Err(invalid("scalar audit chunk extent"));
        }
        rows_total += chunk.tokens;
        for (op, plan) in chunk.operations.iter_mut().zip(&d.plans) {
            if let (PrefillLayerOps::Linear(op), DevicePlan::Linear(p)) = (op, plan) {
                if op.canonical_gdn_scalars.is_some() {
                    return Err(invalid("scalar audit already installed"));
                }
                let projection = p
                    .projection
                    .as_ref()
                    .ok_or_else(|| invalid("scalar weight owner"))?;
                let bind = |row: usize,
                            weight: &OwnedCudaBuffer,
                            output: &OwnedCudaBuffer|
                 -> Result<CudaBf16F32MatvecArgsV1> {
                    Ok(CudaBf16F32MatvecArgsV1::bind(
                        &d.stream,
                        CudaBf16WeightViewV1::new(
                            view(weight)?,
                            PhysicalLayoutId::DEVICE_CONTIGUOUS_BF16_CUDA_V1,
                            linear::VALUE_HEADS,
                            H,
                        )?,
                        CudaF32View::new(
                            w.normalized.view(row * H * 4, H * 4)?,
                            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                            H,
                        )?,
                        CudaF32View::new(
                            output.view(row * linear::VALUE_HEADS * 4, linear::VALUE_HEADS * 4)?,
                            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                            linear::VALUE_HEADS,
                        )?,
                    )?)
                };
                op.canonical_gdn_scalars = Some(
                    (0..chunk.tokens)
                        .map(|row| {
                            Ok([
                                bind(row, &projection.alpha_weight, &w.alpha_raw)?,
                                bind(row, &projection.beta_weight, &w.beta_raw)?,
                            ])
                        })
                        .collect::<Result<Vec<_>>>()?,
                );
                if cfg!(feature = "qwen35-bf16-warp-row") {
                    let batch = |weight: &OwnedCudaBuffer, output: &OwnedCudaBuffer| -> Result<_> {
                        Ok(augur_backend::CudaBf16F32MatvecRowsArgsV1::bind(
                            &d.stream,
                            CudaBf16WeightViewV1::new(view(weight)?,
                                PhysicalLayoutId::DEVICE_CONTIGUOUS_BF16_CUDA_V1, linear::VALUE_HEADS, H)?,
                            CudaF32View::new(w.normalized.view(0, chunk.tokens * H * 4)?,
                                PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1, chunk.tokens * H)?,
                            CudaF32View::new(output.view(0, chunk.tokens * linear::VALUE_HEADS * 4)?,
                                PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1, chunk.tokens * linear::VALUE_HEADS)?,
                            chunk.tokens)?)
                    };
                    op.canonical_gdn_prefill_batch = Some([
                        batch(&projection.alpha_weight, &w.alpha_raw)?,
                        batch(&projection.beta_weight, &w.beta_raw)?,
                    ]);
                }
                bound_ops += 2 * chunk.tokens;
            }
        }
    }
    if bound_ops != 96 * rows_total {
        return Err(invalid("scalar audit layer/row coverage"));
    }
    Ok(
        json!({"schema":"qwen35-native-prefill-canonical-scalars-audit-v1",
        "policy":"decode_BF16_F32_matvec_no_Q8_activation_row_launch_prototype",
        "row_launch_policy":if cfg!(feature="qwen35-bf16-warp-row") {"canonical_fp64_token_grid_v1"}else{"scalar_reference_v1"},
        "prefix_tokens":rows_total,"linear_layers":48,"bound_alpha_beta_rows":bound_ops,
        "default_path_changed":false,"numerical_tolerance_changed":false,
        "throughput_qualified":false,"product_ready":false}),
    )
}
