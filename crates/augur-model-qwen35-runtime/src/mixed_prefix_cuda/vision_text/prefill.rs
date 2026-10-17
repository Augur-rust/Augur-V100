//! Mixed image/text rows enter the existing language prefill backbone directly.
//! Never re-embed image placeholders or reconstruct one-dimensional RoPE.
//! Prefix and scalar last prompt row use the same request-private KV/GDN owner.
use super::*;
use std::ffi::OsStr;

pub(super) fn parse_rows(value: Option<&OsStr>) -> Result<usize> {
    match value.and_then(OsStr::to_str) {
        None if value.is_none() => Ok(1),
        Some("1") => Ok(1),
        Some("64") => Ok(64),
        Some("128") => Ok(128),
        Some("256") => Ok(256),
        _ => Err(invalid(
            "Vision prefill rows must be exactly 1, 64, 128 or 256",
        )),
    }
}

pub(super) fn policy(rows: usize) -> &'static str {
    if rows == 1 {
        "sequential_fp32_state_v1"
    } else {
        "mixed_rows_cublas_q8_inputs_fp32_state_v1"
    }
}

pub(super) fn parse_shared_attention(value: Option<&OsStr>) -> Result<bool> {
    match value.and_then(OsStr::to_str) {
        None if value.is_none() => Ok(false),
        Some("baseline") => Ok(false),
        Some("shared") if cfg!(feature = "qwen35-kv8-chunk-prefill") => Ok(true),
        _ => Err(invalid(
            "Vision prefill attention requires baseline or compiled KV8 shared",
        )),
    }
}
pub(super) fn attention_policy(shared: bool) -> &'static str {
    if shared {
        "kv8_grouped_shared_softmax_ordered"
    } else {
        "kv8_grouped_online_softmax_retained"
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Geometry {
    prefix_rows: usize,
    chunk_rows: usize,
}
impl Geometry {
    fn new(
        rows: usize,
        prompt_rows: usize,
        hidden_len: usize,
        rope_capacity: usize,
    ) -> Result<Self> {
        if !matches!(rows, 1 | 64 | 128 | 256)
            || prompt_rows == 0
            || rope_capacity < prompt_rows
            || rope_capacity > MAX_CONTEXT
            || prompt_rows.checked_mul(linear::HIDDEN) != Some(hidden_len)
        {
            return Err(invalid("Vision mixed prefill extent/position policy"));
        }
        let prefix_rows = if rows == 1 { 0 } else { prompt_rows - 1 };
        Ok(Self {
            prefix_rows,
            chunk_rows: rows.min(prefix_rows),
        })
    }
}

pub(super) struct PreparedMixedRows {
    // Bound operations die before the owning buffers and cuBLAS scratch.
    schedule: PrefillChunkSchedule,
    workspace: PrefillWorkspace,
    geometry: Geometry,
    profile: Option<profile::Profile>,
}
impl PreparedMixedRows {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare(
        rows: usize,
        profiled: bool,
        shared_attention: bool,
        stream: &OwnedCudaStream,
        plans: &[DevicePlan],
        model: &Qwen35Config,
        mrope: &Qwen35ImageMropePlan,
        prompt_rows: usize,
        hidden: &[f32],
    ) -> Result<Option<Self>> {
        let geometry = Geometry::new(rows, prompt_rows, hidden.len(), mrope.capacity())?;
        if hidden.iter().any(|x| !x.is_finite()) || plans.len() != FULL_TARGET_LAYER_COUNT {
            return Err(invalid(
                "Vision mixed prefill finite input/all-layer coverage",
            ));
        }
        if geometry.prefix_rows == 0 {
            return Ok(None);
        }
        let workspace = PrefillWorkspace::new_with_rope_capacity(
            stream,
            geometry.chunk_rows,
            mrope.capacity(),
        )?;
        let (cos, sin) = mrope.tables()?;
        let expected = mrope.capacity() * (full::ROTARY_DIM / 2);
        if cos.len() != expected || sin.len() != expected {
            return Err(invalid("Vision mixed prefill MRoPE table extent"));
        }
        workspace
            .rope_cos
            .copy_from_host_sync(stream, &f32_bytes(&cos))?;
        workspace
            .rope_sin
            .copy_from_host_sync(stream, &f32_bytes(&sin))?;
        let mut schedule = build_prefill_chunk_schedule(
            &workspace,
            plans,
            stream,
            geometry.prefix_rows,
            geometry.chunk_rows,
            model.rms_norm_eps as f32,
        )?;
        if shared_attention {
            #[cfg(not(feature = "qwen35-kv8-chunk-prefill"))]
            return Err(invalid(
                "Vision shared attention requires KV8 chunk prefill",
            ));
            #[cfg(feature = "qwen35-kv8-chunk-prefill")]
            for chunk in &mut schedule.chunks {
                for op in &mut chunk.operations {
                    if let PrefillLayerOps::Full(p) = op {
                        p.simt_shared_softmax = true;
                    }
                }
            }
        }
        Ok(Some(Self {
            schedule,
            workspace,
            geometry,
            profile: profiled
                .then(|| profile::Profile::new(stream))
                .transpose()?,
        }))
    }

    pub(super) fn submit(
        &mut self,
        stream: &OwnedCudaStream,
        start: &OwnedCudaEvent,
        stop: &OwnedCudaEvent,
        terminal: &OwnedCudaEvent,
        hidden: &[f32],
        cancelled: &impl Fn() -> bool,
    ) -> Result<(f64, f64, usize, Option<Value>)> {
        if hidden.len() != self.geometry.prefix_rows * linear::HIDDEN {
            return Err(invalid("Vision mixed prefill submitted row extent"));
        }
        let started = Instant::now();
        let mut offset = 0;
        let mut cuda_ms = 0.0;
        for chunk in &self.schedule.chunks {
            check_cancel(cancelled)?;
            let end = offset + chunk.tokens * linear::HIDDEN;
            self.workspace
                .hidden_a
                .copy_from_host_sync(stream, &f32_bytes(&hidden[offset..end]))?;
            start.record(stream)?;
            for (layer, operation) in chunk.operations.iter().enumerate() {
                check_cancel(cancelled)?;
                if let Some(profile) = &mut self.profile {
                    profile.submit(layer, operation, stream, &self.workspace.cublas)?;
                } else {
                    operation.submit(stream, &self.workspace.cublas)?;
                }
            }
            stop.record(stream)?;
            let receipt = stream.submission_receipt()?;
            terminal.record_for_submission(stream, receipt)?;
            let mut polls = 0;
            let completion = loop {
                if let Some(completion) = stream.query_event_completion(terminal, receipt)? {
                    break completion;
                }
                polls += 1;
                if polls >= MAX_CHUNKED_PREFILL_COMPLETION_POLLS {
                    return Err(Qwen35MixedPrefixCudaError::CompletionPollLimit);
                }
                bounded_completion_poll_backoff(polls);
            };
            if !completion.covers_submission(receipt) || !completion.belongs_to_stream(stream)? {
                return Err(invalid("Vision mixed prefill owner/epoch completion"));
            }
            let elapsed = f64::from(start.elapsed_ms(stop)?);
            if !elapsed.is_finite() || elapsed <= 0.0 {
                return Err(invalid("Vision mixed prefill CUDA timing"));
            }
            cuda_ms += elapsed;
            if let Some(profile) = &mut self.profile {
                profile.collect_chunk()?;
            }
            offset = end;
        }
        if offset != hidden.len() {
            return Err(invalid("Vision mixed prefill incomplete schedule"));
        }
        check_cancel(cancelled)?;
        Ok((
            started.elapsed().as_secs_f64() * 1000.0,
            cuda_ms,
            self.schedule.chunks.len(),
            self.profile
                .as_ref()
                .map(|p| p.summary(cuda_ms))
                .transpose()?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefill_switch_is_explicit_bounded_and_exact() {
        assert_eq!(parse_rows(None).unwrap(), 1);
        for rows in [1, 64, 128, 256] {
            assert_eq!(
                parse_rows(Some(OsStr::new(&rows.to_string()))).unwrap(),
                rows
            );
        }
        for invalid in ["", "0", "8", "32", "512", "0128", "128 ", "true"] {
            assert!(parse_rows(Some(OsStr::new(invalid))).is_err());
        }
    }
    #[test]
    fn shared_attention_switch_is_exact_and_feature_gated() {
        assert!(!parse_shared_attention(None).unwrap());
        assert!(!parse_shared_attention(Some(OsStr::new("baseline"))).unwrap());
        assert_eq!(
            parse_shared_attention(Some(OsStr::new("shared"))).is_ok(),
            cfg!(feature = "qwen35-kv8-chunk-prefill")
        );
        for value in ["", "1", "true", "shared ", "SHARED"] {
            assert!(parse_shared_attention(Some(OsStr::new(value))).is_err());
        }
    }

    #[test]
    fn mixed_rows_leave_one_real_prompt_row_for_unchanged_decode() {
        for prompt in [1, 2, 63, 64, 65, 128, 305, 1024, 4032] {
            for width in [64, 128, 256] {
                let g = Geometry::new(width, prompt, prompt * linear::HIDDEN, 4096).unwrap();
                assert_eq!(g.prefix_rows + 1, prompt);
                if g.prefix_rows > 0 {
                    let actual = (0..g.prefix_rows)
                        .step_by(g.chunk_rows)
                        .flat_map(|start| start..(start + g.chunk_rows).min(g.prefix_rows))
                        .chain(g.prefix_rows..prompt)
                        .collect::<Vec<_>>();
                    assert_eq!(actual, (0..prompt).collect::<Vec<_>>());
                }
            }
        }
    }

    #[test]
    fn geometry_rejects_drift_overflow_empty_and_out_of_context() {
        for (rows, prompt, hidden, capacity) in [
            (32, 100, 100 * linear::HIDDEN, 4096),
            (128, 0, 0, 4096),
            (128, 100, 99 * linear::HIDDEN, 4096),
            (128, 100, 100 * linear::HIDDEN, 99),
            (128, 4097, 4097 * linear::HIDDEN, 4097),
            (128, usize::MAX, usize::MAX, 4096),
        ] {
            assert!(Geometry::new(rows, prompt, hidden, capacity).is_err());
        }
        assert_eq!(
            Geometry::new(1, 305, 305 * linear::HIDDEN, 4096)
                .unwrap()
                .prefix_rows,
            0
        );
    }
}
