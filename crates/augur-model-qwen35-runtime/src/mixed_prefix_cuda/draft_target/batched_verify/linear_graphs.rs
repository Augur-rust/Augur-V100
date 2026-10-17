//! Group three position-independent linear layers and their record/feature
//! copies. Full attention, acceptance and actual prefix replay remain eager.
use super::*;
use augur_backend::OwnedCudaGraphExec;

pub(in super::super) fn switch(value: Option<&std::ffi::OsStr>) -> Result<bool> {
    match value {
        None => Ok(false),
        Some(v) if v == "1" => Ok(true),
        _ => Err(invalid(
            "verify linear graphs require AUGUR_QWEN35_VERIFY_LINEAR_GRAPHS=1",
        )),
    }
}

pub(super) struct LinearGraphs {
    groups: Vec<OwnedCudaGraphExec>,
    pub(super) nodes: u64,
    pub(super) capture_ms: f64,
}

fn validate_topology(linear: &[bool]) -> Result<()> {
    if linear.len() != 64 || linear.iter().enumerate().any(|(i, &v)| v != (i % 4 != 3)) {
        return Err(invalid(
            "grouped graph requires sixteen linear/linear/linear/full regions",
        ));
    }
    Ok(())
}

impl LinearGraphs {
    pub(super) fn capture(
        d: &Driver,
        core: &BatchedVerifyAudit,
        operations: &[PrefillLayerOps],
        records: &ReplaySsm,
    ) -> Result<Self> {
        if operations.len() != 64
            || core.profile.is_some()
            || core.canonical_profile.is_some()
            || core.numerics.is_some()
            || d.reference_boundary.is_some()
            || d.prefix_boundary_audit.is_some()
            || core.batch_rows != B
            || core.workspace.cublas.paired_ffn_enabled()
            || core.exact_input_owner
            || d.verified_features.is_none()
        {
            return Err(invalid(
                "linear graphs require fixed unaudited canonical layers",
            ));
        }
        #[cfg(feature = "qwen35-raw-committed-state")]
        if d.raw_state.is_some() {
            return Err(invalid(
                "grouped graph does not capture raw-state audit side effects",
            ));
        }
        let topology: Vec<_> = operations
            .iter()
            .map(|op| matches!(op, PrefillLayerOps::Linear(_)))
            .collect();
        validate_topology(&topology)?;
        if operations
            .iter()
            .enumerate()
            .any(|(i, op)| i % 4 == 3 && !matches!(op, PrefillLayerOps::Full(_)))
        {
            return Err(invalid(
                "grouped graph rejects alternate full-layer topology",
            ));
        }
        let start = Instant::now();
        let mut groups = Vec::with_capacity(16);
        let mut nodes = 0;
        for first in (0..64).step_by(4) {
            let mut operation_error = None;
            let graph = OwnedCudaGraphExec::capture(&d.stream, || {
                    let submit = (|| -> Result<()> {
                        for layer in first..first + 3 {
                            let PrefillLayerOps::Linear(p) = &operations[layer] else {
                                return Err(invalid("grouped graph linear topology changed"));
                            };
                            p.submit_with_activation_policy(&d.stream, &core.workspace.cublas, false)?;
                            // Capture before the next layer reuses qkv/alpha/beta
                            // or hidden buffers. No cursor or host stamp is baked.
                            records.capture_graph_copy(d, &core.workspace, layer)?;
                            core.capture_verified_layer(d, layer)?;
                        }
                        Ok(())
                    })();
                    submit.map_err(|error| {
                            operation_error = Some(error.to_string());
                            CudaRuntimeError::invalid_metadata("canonical_grouped_linear_graph_capture")
                        })
                })
                .map_err(|error| {
                    invalid(format!(
                        "verify grouped linear graph first layer {first}: {error}; operation={operation_error:?}"
                    ))
                })?;
            nodes += graph.node_count();
            groups.push(graph);
        }
        Ok(Self {
            groups,
            nodes,
            capture_ms: start.elapsed().as_secs_f64() * 1000.0,
        })
    }

    // The enclosing BatchPlanBinding authenticates stream, owner, capacity and
    // generation before any graph launch. State/scratch/weight addresses are
    // owned by the same retained execution and never rebound under a graph.
    pub(super) fn launch_group(
        &self,
        first: usize,
        d: &Driver,
        records: &ReplaySsm,
    ) -> Result<usize> {
        if first >= 64 || first % 4 != 0 {
            return Err(invalid(
                "grouped graph must start at a three-linear-layer region",
            ));
        }
        let graph = self
            .groups
            .get(first / 4)
            .ok_or_else(|| invalid("grouped graph extent"))?;
        for layer in first..first + 3 {
            records.mark_graph_capture(d, layer)?;
        }
        // Any launch failure poisons the enclosing Driver; marked records
        // cannot be sealed/committed on that failed transaction.
        graph.launch(&d.stream)?;
        Ok(first + 3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn verify_linear_graph_switch_is_explicit_and_strict() {
        assert!(!switch(None).unwrap());
        assert!(switch(Some("1".as_ref())).unwrap());
        for value in ["", "0", "true", " 1", "1 "] {
            assert!(switch(Some(value.as_ref())).is_err());
        }
    }

    #[test]
    fn grouped_linear_graph_requires_exact_regions() {
        let valid: Vec<_> = (0..64).map(|i| i % 4 != 3).collect();
        assert!(validate_topology(&valid).is_ok());
        assert!(validate_topology(&valid[..63]).is_err());
        for index in 0..64 {
            let mut invalid = valid.clone();
            invalid[index] = !invalid[index];
            assert!(validate_topology(&invalid).is_err());
        }
    }
}
