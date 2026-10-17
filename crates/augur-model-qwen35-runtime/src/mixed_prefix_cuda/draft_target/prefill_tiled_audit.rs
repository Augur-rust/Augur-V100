//! Diagnostic layer-major512 schedule using existing canonical native math.
//! Only explicit native callers install this; defaults and small-batch bounds stay intact.
use super::*;

pub(super) fn install_projections_and_norms(operations: &mut [PrefillLayerOps]) -> Result<()> {
    install_projection_policy(operations, false)
}

pub(super) fn install_grid(operations: &mut [PrefillLayerOps]) -> Result<()> {
    install_projection_policy(operations, true)
}

fn install_projection_policy(operations: &mut [PrefillLayerOps], grid: bool) -> Result<()> {
    if operations.len() != 64 {
        return Err(invalid("tiled canonical audit needs all64 layers"));
    }
    let mut projections = 0;
    for operation in operations {
        let (norms, matrices): ([_; 2], Vec<_>) = match operation {
            PrefillLayerOps::Linear(p) => { let ffn = p.ffn.legacy_mut()?; (
                [&mut p.attn_norm, &mut p.post_norm],
                vec![
                    &mut p.qkv,
                    &mut p.z,
                    &mut p.output,
                    &mut ffn.ffn_gate,
                    &mut ffn.ffn_up,
                    &mut ffn.ffn_down,
                ],
            ) },
            PrefillLayerOps::Full(p) => { let ffn = p.ffn.legacy_mut()?; (
                [&mut p.attn_norm, &mut p.post_norm],
                vec![
                    &mut p.q,
                    &mut p.k,
                    &mut p.v,
                    &mut p.output,
                    &mut ffn.ffn_gate,
                    &mut ffn.ffn_up,
                    &mut ffn.ffn_down,
                ],
            ) },
            
        };
        for norm in norms {
            norm.enable_canonical_prefill_rows_audit()?;
        }
        for projection in matrices {
            if grid { projection.enable_canonical_grid_rows_audit()?; }
            else { projection.enable_canonical_tiled_rows_audit()?; }
            projections += 1;
        }
    }
    if projections != 400 {
        return Err(invalid("tiled canonical projection coverage"));
    }
    Ok(())
}

pub(super) fn submit_layer(
    d: &Driver,
    w: &PrefillWorkspace,
    operation: &PrefillLayerOps,
    plan: &DevicePlan,
    rows: usize,
) -> Result<()> {
    match (operation, plan) {
        (PrefillLayerOps::Full(p), DevicePlan::Full(plan)) => {
            batched_verify::full_attention::submit_rows_audit(d, w, p, plan, rows)
        }
        (PrefillLayerOps::Linear(_), DevicePlan::Linear(_)) => {
            operation.submit(&d.stream, &w.cublas)
        }
        _ => Err(invalid("tiled canonical operation/owner mismatch")),
    }
}
