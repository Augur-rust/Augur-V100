//! Explicit request-only installation; default prefix and decode are unchanged.
use super::*;
use serde_json::{json, Value};

pub(super) fn eligible(position: usize, rows: usize) -> bool {
    position >= 512 && (128..=512).contains(&rows)
}

pub(super) fn install(
    mode: &mut AutoregressivePrefillMode,
    stream: &OwnedCudaStream,
    capacity: usize,
) -> Result<Value, Qwen35MixedPrefixCudaError> {
    let AutoregressivePrefillMode::Batched(prefill) = mode else {
        return Err(invalid("DGEMM scores require a real batched prefix"));
    };
    if !prefill.simt_shared_softmax
        || prefill.tensor_core_prefix_attention
        || prefill.direct_activation
        || prefill.workspace.dgemm_score_scratch.is_some()
    {
        return Err(invalid("DGEMM scores incompatible prefix policy"));
    }
    let elements = CudaQwen35PrefillKv8ChunkAttentionV1::dgemm_score_elements(
        prefill.workspace.max_tokens,
        capacity,
    )?;
    let scratch = OwnedCudaBuffer::allocate(stream.device_ordinal(), elements * 4)?;
    let mut position = 0;
    let mut selected = 0;
    let mut retained = 0;
    for chunk in &mut prefill.schedule.chunks {
        for op in &mut chunk.operations {
            if let PrefillLayerOps::Full(full) = op {
                if !full.simt_shared_softmax
                    || full.simt_softmax_tile16
                    || full.simt_unit_previous_scale
                {
                    return Err(invalid("DGEMM scores incompatible full-layer policy"));
                }
                if eligible(position, chunk.tokens) {
                    let n = CudaQwen35PrefillKv8ChunkAttentionV1::dgemm_score_elements(
                        chunk.tokens,
                        capacity,
                    )?;
                    let view = CudaF32View::new(
                        scratch.view(0, n * 4)?,
                        PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                        n,
                    )?;
                    full.dgemm_attention = Some(full.attention.with_dgemm_scores(
                        stream,
                        &prefill.workspace.cublas,
                        view,
                    )?);
                    selected += 1;
                } else {
                    retained += 1;
                }
            }
        }
        position += chunk.tokens;
    }
    if selected == 0 {
        return Err(invalid("DGEMM request has no eligible prefix work"));
    }
    // No operations have been submitted during installation. If any bind
    // fails the caller exits without using the partly built schedule.
    prefill.workspace.dgemm_score_scratch = Some(scratch);
    Ok(json!({"schema":"qwen35-prefill-dgemm-scores-request-v1",
        "policy":"position_ge512_rows128to512_otherwise_retained_v1",
        "scratch_bytes":elements*4,"bound_candidate_layer_ops":selected,
        "bound_retained_layer_ops":retained,"prefix_tokens":position,
        "workspace_owner":"prefill_only_released_before_auxiliary_and_decode",
        "score_policy":"fp64_dgemm_cast_fp32_scale_then_ordered_fp32_consumer",
        "numerical_tolerance_changed":false,"model_semantics_qualified":false,"product_ready":false}))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dispatch_keeps_short_prefix_and_small_tails_on_control() {
        for (p, n) in [
            (0, 512),
            (511, 512),
            (512, 1),
            (8192, 17),
            (194304, 127),
            (512, 513),
        ] {
            assert!(!eligible(p, n));
        }
        for (p, n) in [(512, 512), (8192, 128), (194304, 512)] {
            assert!(eligible(p, n));
        }
    }
}
