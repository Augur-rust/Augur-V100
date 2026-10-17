//! Default-off observer; preserves the existing exact audit's failure decision.
//! Comparing B8 projections to serial recomputation is NOT isolated fold correctness.
use super::*;

pub(super) fn directory(value: Option<&std::ffi::OsStr>, audit: bool) -> Result<Option<PathBuf>> {
    if value.is_some() && !audit {
        return Err(invalid(
            "replay numeric observer requires exact replay audit",
        ));
    }
    fixed_history_logits::directory(value)
}

struct Plane {
    layer: usize,
    plane: usize,
    kv8: bool,
    offset: usize,
    bytes: Vec<u8>,
}

fn states(d: &Driver) -> Result<Vec<Plane>> {
    let receipt = d.stream.submission_receipt()?;
    let mut result = Vec::with_capacity(128);
    for (layer, plan) in d.plans.iter().enumerate() {
        for (plane, buffer) in plan.terminal_state_buffers().into_iter().enumerate() {
            let kv8 = matches!(plan, DevicePlan::Full(_));
            let (offset, bytes) = state_range(kv8, buffer.len_bytes(), d.cursor, d.capacity)?;
            let mut host = vec![0; bytes];
            let proof = buffer
                .view(offset, bytes)?
                .copy_to_host_sync_with_submission_proof(&d.stream, &mut host, receipt)?;
            if !proof.covers_submission(receipt) || !proof.belongs_to_stream(&d.stream)? {
                return Err(invalid("replay observer state owner/epoch"));
            }
            result.push(Plane {
                layer,
                plane,
                kv8,
                offset,
                bytes: host,
            });
        }
    }
    Ok(result)
}

fn decoded(plane: &Plane) -> Result<Vec<f32>> {
    if !plane.kv8 {
        if !plane.bytes.len().is_multiple_of(4) {
            return Err(invalid("observer F32 extent"));
        }
        return Ok(f32_vec(&plane.bytes));
    }
    if !plane.bytes.len().is_multiple_of(Q8_0_BLOCK_BYTES) {
        return Err(invalid("observer Q8 extent"));
    }
    let mut output = vec![0.; plane.bytes.len() / Q8_0_BLOCK_BYTES * Q8_0_BLOCK_ELEMENTS];
    for (block, values) in plane
        .bytes
        .chunks_exact(Q8_0_BLOCK_BYTES)
        .zip(output.chunks_exact_mut(Q8_0_BLOCK_ELEMENTS))
    {
        dequantize_q8_0_block(block, values)?;
    }
    Ok(output)
}

fn difference(actual: &[f32], reference: &[f32]) -> Result<Value> {
    if actual.is_empty()
        || actual.len() != reference.len()
        || actual.iter().chain(reference).any(|v| !v.is_finite())
    {
        return Err(invalid("replay observer metric extent/nonfinite"));
    }
    let (mut maximum, mut square, mut reference_square, mut unequal) =
        (0.0f64, 0.0f64, 0.0f64, 0usize);
    for (&a, &r) in actual.iter().zip(reference) {
        let delta = f64::from(a) - f64::from(r);
        maximum = maximum.max(delta.abs());
        square += delta * delta;
        reference_square += f64::from(r) * f64::from(r);
        unequal += usize::from(a.to_bits() != r.to_bits());
    }
    Ok(
        json!({"elements":actual.len(),"unequal_f32_bits":unequal,"max_abs":maximum,
        "rmse":(square / actual.len() as f64).sqrt(),
        "relative_l2":if reference_square > 0. { Some((square / reference_square).sqrt()) } else { None },
        "reference_l2":reference_square.sqrt(),"all_finite":true}),
    )
}

fn feature_rows(d: &Driver, buffer: &OwnedCudaBuffer, committed: usize) -> Result<Vec<f32>> {
    read_f32_view(
        &d.stream,
        buffer.view(0, committed * 5 * H * 4)?,
        d.stream.submission_receipt()?,
    )
}

fn save_f32(root: &Path, name: &str, values: &[f32]) -> Result<Value> {
    let bytes = values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    fixed_history_logits::write_new(&root.join(name), &bytes)?;
    Ok(json!({"file":name,"elements":values.len(),"sha256":sha256_hex(&bytes)}))
}

pub(super) struct Before {
    directory: PathBuf,
    cursor: usize,
    generation: u64,
    committed: usize,
    path: Vec<u32>,
    states: Vec<Plane>,
    logits: Vec<f32>,
    features: Vec<f32>,
    verified_next: u32,
    serial_attention_rows: usize,
    vector_attention_rows: usize,
}

impl Before {
    pub(super) fn capture(
        d: &Driver,
        committed: usize,
        root: &Path,
        serial_attention_rows: usize,
        vector_attention_rows: usize,
        head: &output_head::PackedOutputHead,
    ) -> Result<Self> {
        if !(1..=B).contains(&committed) {
            return Err(invalid("numeric accepted prefix extent"));
        }
        let states = states(d)?;
        let logits = head.diagnostic_target_row(d, committed - 1)?;
        let verified_next = words(&d.stream, &d.verify_ids, (committed - 1) * 4, 1)?[0];
        target_only_reference::summarize_logits(&logits, verified_next, committed - 1)?;
        let features = feature_rows(
            d,
            d.verified_features
                .as_ref()
                .ok_or_else(|| invalid("numeric verified features absent"))?,
            committed,
        )?;
        Ok(Self {
            directory: root.to_path_buf(),
            cursor: d.cursor,
            generation: d.generation,
            committed,
            path: words(&d.stream, &d.path, 0, B)?,
            states,
            logits,
            features,
            verified_next,
            serial_attention_rows,
            vector_attention_rows,
        })
    }

    pub(super) fn finish(
        self,
        d: &Driver,
        features: &OwnedCudaBuffer,
        folded_fingerprint: &str,
        serial_fingerprint: &str,
    ) -> Result<()> {
        if self.cursor != d.cursor || self.generation != d.generation {
            return Err(invalid("numeric observer target identity changed"));
        }
        let serial_states = states(d)?;
        if self.states.len() != 128 || serial_states.len() != self.states.len() {
            return Err(invalid("numeric observer all-layer extent"));
        }
        let mut planes = Vec::with_capacity(128);
        for (a, r) in self.states.iter().zip(&serial_states) {
            if (a.layer, a.plane, a.kv8, a.offset, a.bytes.len())
                != (r.layer, r.plane, r.kv8, r.offset, r.bytes.len())
            {
                return Err(invalid("numeric observer state pairing"));
            }
            let numeric = difference(&decoded(a)?, &decoded(r)?)?;
            planes.push(json!({"layer":a.layer,"plane":a.plane,"storage":if a.kv8 {"Q8_0_KV"} else {"FP32_history_or_state"},
                "offset_bytes":a.offset,"bytes":a.bytes.len(),"bytes_exact":a.bytes==r.bytes,
                "unequal_bytes":a.bytes.iter().zip(&r.bytes).filter(|(x,y)|x!=y).count(),
                "folded_sha256":sha256_hex(&a.bytes),"serial_sha256":sha256_hex(&r.bytes),"numeric":numeric}));
        }
        let serial_logits =
            read_f32_buffer(&d.stream, &d.head.logits, d.stream.submission_receipt()?)?;
        let serial_features = feature_rows(d, features, self.committed)?;
        let logits_difference = difference(&self.logits, &serial_logits)?;
        let serial_next = serial_logits
            .iter()
            .enumerate()
            .max_by(|(i, a), (j, b)| a.total_cmp(b).then_with(|| j.cmp(i)))
            .ok_or_else(|| invalid("numeric empty logits"))?
            .0 as u32;
        let mut feature_difference = Vec::with_capacity(self.committed * 5);
        for row in 0..self.committed {
            for (slot, &layer) in DFLASH_TARGET_CAPTURE_LAYER_IDS.iter().enumerate() {
                let start = (row * 5 + slot) * H;
                feature_difference.push(json!({"row":row,"layer":layer,
                    "numeric":difference(&self.features[start..start+H], &serial_features[start..start+H])?}));
            }
        }
        // Revalidate the existing absolute directory before creating a unique evidence child.
        fixed_history_logits::directory(Some(self.directory.as_os_str()))?;
        let root = self.directory.join(format!(
            "cursor-{}-generation-{}",
            self.cursor, self.generation
        ));
        std::fs::create_dir(&root).map_err(Qwen35Error::from)?;
        let report = json!({"schema":"qwen35-replay-same-checkpoint-numerics-v1",
            "source_commit":crate::build_identity::source_commit(),"target_identity":sha256_hex(&d.identity),
            "cursor":self.cursor,"generation":self.generation,"committed_inputs":self.committed,"path_ids":self.path,
            "diagnostic_serial_attention_rows_submitted":self.serial_attention_rows,
            "candidate_vector_attention_rows_submitted":self.vector_attention_rows,
            "fixed_tile8_attention_layers":d.plans.iter().filter(|plan| matches!(plan,
                DevicePlan::Full(p) if p.rotated_attention.as_ref().is_some_and(|a| a.tile8_rows))).count(),
            "folded_state_fingerprint":folded_fingerprint,"serial_state_fingerprint":serial_fingerprint,
            "planes":planes,"feature_rows":feature_difference,"logits_difference":logits_difference,
            "batched_summary":target_only_reference::summarize_logits(&self.logits,self.verified_next,self.committed-1)?,
            "serial_summary":target_only_reference::summarize_logits(&serial_logits,serial_next,self.committed-1)?,
            "serial_selection_scope":"CPU_argmax_inventory_only_not_publication",
            "batched_logits":save_f32(&root,"batched-logits.f32le.bin",&self.logits)?,
            "serial_logits":save_f32(&root,"serial-logits.f32le.bin",&serial_logits)?,
            "batched_features":save_f32(&root,"batched-features.f32le.bin",&self.features)?,
            "serial_features":save_f32(&root,"serial-features.f32le.bin",&serial_features)?,
            "same_starting_checkpoint":true,"different_projection_arithmetic":true,
            "isolated_record_fold_correctness_proof":false,"original_exact_audit_decision_unchanged":true,
            "host_serialized_diagnostic":true,"performance_evidence":false,
            "product_ready":false,"qualified":false,"semantic_qualification":false,"visible_tps":false});
        let bytes = serde_json::to_vec_pretty(&report).map_err(|e| invalid(e.to_string()))?;
        fixed_history_logits::write_new(&root.join("report.json"), &bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn observer_requires_explicit_audit_and_valid_directory() {
        assert!(directory(None, false).unwrap().is_none());
        assert!(directory(Some(std::ffi::OsStr::new("relative")), false).is_err());
        assert!(directory(Some(std::ffi::OsStr::new("relative")), true).is_err());
    }
    #[test]
    fn metrics_handle_equal_signed_zero_and_zero_reference() {
        let equal = difference(&[1., 2.], &[1., 2.]).unwrap();
        assert_eq!(equal["max_abs"], 0.);
        assert_eq!(equal["unequal_f32_bits"], 0);
        let zero = difference(&[-0., 1.], &[0., 0.]).unwrap();
        assert_eq!(zero["unequal_f32_bits"], 2);
        assert!(zero["relative_l2"].is_null());
        assert_eq!(zero["max_abs"], 1.);
    }
    #[test]
    fn metrics_reject_extent_and_nonfinite() {
        for (a, r) in [
            (vec![], vec![]),
            (vec![1.], vec![1., 2.]),
            (vec![f32::NAN], vec![0.]),
            (vec![0.], vec![f32::INFINITY]),
        ] {
            assert!(difference(&a, &r).is_err());
        }
    }

    #[test]
    fn kv_decoder_uses_q8_blocks_not_raw_float_bits() {
        let mut bytes = vec![0; Q8_0_BLOCK_BYTES];
        bytes[0..2].copy_from_slice(&0x3c00u16.to_le_bytes()); // FP16 scale1
        bytes[2] = 127;
        bytes[3] = (-128i8) as u8;
        let plane = Plane {
            layer: 3,
            plane: 0,
            kv8: true,
            offset: 0,
            bytes,
        };
        let values = decoded(&plane).unwrap();
        assert_eq!(values.len(), 32);
        assert_eq!(values[0], 127.);
        assert_eq!(values[1], -128.);
        assert!(values[2..].iter().all(|v| *v == 0.));
        let short = Plane {
            bytes: vec![0; 3],
            ..plane
        };
        assert!(decoded(&short).is_err());
    }
}
