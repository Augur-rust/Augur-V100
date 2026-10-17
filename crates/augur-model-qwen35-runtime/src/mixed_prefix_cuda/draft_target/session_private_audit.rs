//! Bounded selected-phase audit; no acceptance overrides or duplicate GPU owner.
use super::*;
use std::collections::BTreeSet;

fn phase(round: usize, rows: usize, previous: Option<&Value>) -> Result<&'static str> {
    if round == 0 {
        if rows != 2048 || previous.is_some() {
            return Err(invalid("initial session private phase"));
        }
        return Ok("initial");
    }
    let previous = previous.ok_or_else(|| invalid("missing natural preceding transaction"))?;
    if !(1..=B).contains(&rows)
        || previous["committed_token_count"].as_u64() != Some(rows as u64)
        || previous["publication_stop_reason"].as_str().is_some()
    {
        return Err(invalid(
            "session private reuse must match natural committed rows",
        ));
    }
    Ok(if rows == B {
        "after_full_commit"
    } else {
        "after_partial_commit"
    })
}

pub(super) struct SessionPrivateAudit {
    root: PathBuf,
    role: &'static str,
    seen: BTreeSet<&'static str>,
    frames: Vec<Value>,
}

impl SessionPrivateAudit {
    pub(super) fn new(root: &Path, role: &'static str) -> Result<Self> {
        if !root.is_absolute()
            || root.exists()
            || !matches!(role, "session-retained-rebind" | "session-fresh-recompute")
        {
            return Err(invalid("exclusive session private audit path/role"));
        }
        Ok(Self {
            root: root.into(),
            role,
            seen: BTreeSet::new(),
            frames: Vec::new(),
        })
    }

    pub(super) fn capture_selected(
        &mut self,
        draft: &DFlashQ8CudaBlock,
        d: &Driver,
        round: usize,
        previous: Option<&Value>,
        base_history: &[u32],
        output: &[u32],
    ) -> Result<()> {
        let rows = draft.context_projection_rows();
        let selected = phase(round, rows, previous)?;
        if self.seen.contains(selected) {
            return Ok(());
        }
        if self.frames.len() >= 3 || d.context_rows != 2048 {
            return Err(invalid("private frame bound"));
        }
        // Exclude the final pending output from consumed input; never retokenize.
        let mut consumed = base_history.to_vec();
        if !output.is_empty() {
            if output.last() != Some(&d.anchor) {
                return Err(invalid("private pending output binding"));
            }
            consumed.extend_from_slice(&output[..output.len() - 1]);
        }
        if consumed.len() != d.cursor || consumed.iter().any(|&t| t as usize >= V) {
            return Err(invalid("private exact input ledger extent"));
        }
        if let Some(tx) = previous {
            if tx["cursor_after"].as_u64() != Some(d.cursor as u64)
                || tx["generation_after"].as_u64() != Some(d.generation)
                || tx["emitted_token_ids"]
                    .as_array()
                    .and_then(|a| a.last())
                    .and_then(Value::as_u64)
                    != Some(d.anchor as u64)
            {
                return Err(invalid("private previous transaction live binding"));
            }
        }
        // Read the actual live target feature ring with its submission proof.
        // No mid-round full target-state or performance claim is made.
        d.stream.synchronize()?;
        let proof = d.stream.submission_receipt()?;
        let mut host = vec![0u8; 1024 * 1024];
        let mut h = Sha256::new();
        for offset in (0..d.features.len_bytes()).step_by(host.len()) {
            let count = host.len().min(d.features.len_bytes() - offset);
            let done = d.features.copy_range_to_host_sync_with_submission_proof(
                &d.stream,
                offset,
                &mut host[..count],
                proof,
            )?;
            if !done.covers_submission(proof) || !done.belongs_to_stream(&d.stream)? {
                return Err(invalid("private target feature completion proof"));
            }
            h.update(&host[..count]);
        }
        let directory = if selected == "initial" {
            self.root.clone()
        } else {
            let name = self
                .root
                .file_name()
                .and_then(|x| x.to_str())
                .ok_or_else(|| invalid("private directory name"))?;
            self.root.with_file_name(format!("{name}-{selected}"))
        };
        let binding = draft
            .capture_native_session_private_audit(&directory, self.role)
            .map_err(invalid)?;
        self.frames.push(json!({"phase":selected,"round":round,"new_rows":rows,
            "previous_transaction":previous,"binding":binding,
            "exact_consumed_token_sha256":committed_session::token_hash(&consumed),
            "target_checkpoint":{"cursor":d.cursor,"generation":d.generation,"pending_token":d.anchor,
                "request_identity":sha256_hex(&d.identity),"feature_bytes_sha256":format!("{:x}",h.finalize())}}));
        self.seen.insert(selected);
        Ok(())
    }

    pub(super) fn first_binding(&self) -> Option<Value> {
        self.frames.first().map(|f| f["binding"].clone())
    }
    pub(super) fn report(self) -> Value {
        json!({"schema":"qwen35-native-session-private-series-v1","frames":self.frames,
            "selection":"initial_and_first_natural_full_and_partial_commit",
            "acceptance_forced":false,"duplicate_gpu_workspace":false,
            "all_rounds_captured":false,"latency_qualified":false,"product_ready":false})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn selected_phase_requires_natural_commit_extent() {
        assert_eq!(phase(0, 2048, None).unwrap(), "initial");
        for rows in 1..=8 {
            let tx = json!({"committed_token_count":rows,"publication_stop_reason":null});
            assert_eq!(
                phase(1, rows, Some(&tx)).unwrap(),
                if rows == 8 {
                    "after_full_commit"
                } else {
                    "after_partial_commit"
                }
            );
            assert!(phase(1, rows + 1, Some(&tx)).is_err());
        }
    }
    #[test]
    fn malformed_or_terminal_reuse_never_captured() {
        assert!(phase(0, 8, None).is_err());
        assert!(phase(1, 8, None).is_err());
        assert!(phase(
            1,
            8,
            Some(&json!({"committed_token_count":8,"publication_stop_reason":"eos"}))
        )
        .is_err());
        assert!(phase(1, 0, Some(&json!({"committed_token_count":0}))).is_err());
    }
}
