//! Default-OFF exact packed head for draft scoring and target verification.
//! Reuses the execution workspace's Q8 rows; owns only the eight logit rows.
use super::*;
use augur_backend::CudaQwen35PackedSmallBatchProjectionV1;

pub(super) const POLICY: &str = "packed_q8_exact_full8_draft_and_target_v1";
pub(super) const LOGIT_BYTES: usize = B * V * 4;

pub(super) struct PackedOutputHead {
    logits: OwnedCudaBuffer,
    projection: HeadBatchProjection,
    target_top1: Vec<CudaQwen35VocabTop1ArgsV1>,
    target_top1_batch: Option<CudaQwen35VocabTop1ArgsV1>,
    input_prepare: Option<PreparedHeadInput>,
    draft_stores: std::cell::RefCell<Option<augur_backend::CudaQwen35HeadPrepareB8V1>>,
    draft_top16: Vec<CudaQwen35VocabTopkArgsV1>,
    draft_top16_batch: Option<CudaQwen35VocabTopkArgsV1>,
    // Host-only, bounded observation. Never read by a selector or publication.
    verify_logits_diagnostic: Option<PathBuf>,
}

struct PreparedHeadInput {
    // Same runtime-TU 256-leaf row reduction as DeviceOutputHeadPlan::norm_op;
    // deliberately NOT the llama-input prefill norm arithmetic.
    target_norm: CudaQwen35PrefillRmsnormV1,
    target_quantize: CudaQ8_1ActivationQuantizeArgsV1,
    target_stores: Option<augur_backend::CudaQwen35HeadPrepareB8V1>,
}

enum HeadBatchProjection {
    Block(CudaQwen35PackedSmallBatchProjectionV1),
    Planar(augur_backend::CudaQwen35PlanarQ8HeadProjectionV1),
}
impl HeadBatchProjection {
    fn submit(&self, stream: &OwnedCudaStream) -> std::result::Result<(), augur_backend::CudaRuntimeError> {
        match self { Self::Block(p) => p.submit(stream), Self::Planar(p) => p.submit(stream) }
    }
}

impl PackedOutputHead {
    fn observe_extent(cursor: usize, generation: u64) -> bool {
        cursor <= 512 && generation <= 8
    }

    pub(super) fn observe_verify_logits(&self, d: &Driver) -> Result<bool> {
        let Some(directory) = &self.verify_logits_diagnostic else { return Ok(false); };
        if !Self::observe_extent(d.cursor, d.generation) { return Ok(false); }
        fixed_history_logits::directory(Some(directory.as_os_str()))?;
        let receipt = d.stream.submission_receipt()?;
        let logits = read_f32_view(&d.stream, self.logits.view(0, LOGIT_BYTES)?, receipt)?;
        if logits.len() != B * V { return Err(invalid("verify logits observer extent")); }
        let path = words(&d.stream, &d.path, 0, B)?;
        let selected = words(&d.stream, &d.verify_ids, 0, B)?;
        let summaries = (0..B).map(|row| target_only_reference::summarize_logits(
            &logits[row*V..(row+1)*V], selected[row], row)).collect::<Result<Vec<_>>>()?;
        let identity = sha256_hex(&d.identity);
        let root = directory.join(format!("{identity}-cursor-{}-generation-{}",d.cursor,d.generation));
        std::fs::create_dir(&root).map_err(Qwen35Error::from)?;
        let bytes = logits.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>();
        fixed_history_logits::write_new(&root.join("verify-logits.f32le.bin"), &bytes)?;
        let report = json!({"schema":"qwen35-bounded-verify-logits-observer-v1",
            "source_commit":crate::build_identity::source_commit(),"target_identity":identity,
            "cursor":d.cursor,"generation":d.generation,"path_ids":path,"selected_ids":selected,
            "rows":B,"vocab":V,"logits_file":"verify-logits.f32le.bin",
            "logits_sha256":sha256_hex(&bytes),"summaries":summaries,
            "capture_bounds":{"maximum_cursor":512,"maximum_generation":8},
            "host_serialized_diagnostic":true,"performance_evidence":false,
            "publication_policy_changed":false,"product_ready":false,"qualified":false,
            "semantic_qualification":false,"visible_tps":false});
        fixed_history_logits::write_new(&root.join("report.json"),
            &serde_json::to_vec_pretty(&report).map_err(|e|invalid(e.to_string()))?)?;
        Ok(true)
    }

    // Only called by the explicit host-serialized replay numerical observer.
    pub(super) fn diagnostic_target_row(&self, d: &Driver, row: usize) -> Result<Vec<f32>> {
        if row >= B { return Err(invalid("diagnostic target row extent")); }
        read_f32_view(&d.stream, self.logits.view(row * V * 4, V * 4)?, d.stream.submission_receipt()?)
    }

    pub(super) fn new(d: &Driver, workspace: &PrefillWorkspace, epsilon: f32) -> Result<Self> {
        let before = cuda_device_allocation_snapshot();
        let logits = OwnedCudaBuffer::allocate(d.stream.device_ordinal(), LOGIT_BYTES)?;
        let projection = if d.head.output_weight_layout == PhysicalLayoutId::DEVICE_QWEN35_Q8_CODE_SCALE_PLANES_V1 {
            HeadBatchProjection::Planar(augur_backend::CudaQwen35PlanarQ8HeadProjectionV1::bind(
                &d.stream, augur_backend::CudaQwen35PlanarQ8HeadWeightV1::new(
                    view(&d.head.output_weight)?, d.head.output_weight_layout, V, H)?,
                prefill_q8_1_view(&workspace.normalized_q8_1, B*H)?,
                plan_f32_view(&logits, B*V)?, B)?)
        } else { HeadBatchProjection::Block(CudaQwen35PackedSmallBatchProjectionV1::bind(
            &d.stream,
            CudaQwen35PrefillWeightV1::Q8_0(CudaQ8_0WeightViewV1::new(
                view(&d.head.output_weight)?,
                PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_0_CUDA_V1,
                V,
                H,
            )?),
            prefill_q8_1_view(&workspace.normalized_q8_1, B * H)?,
            plan_f32_view(&logits, B * V)?,
            B,
            true,
        )?) };
        let mut target_top1 = Vec::with_capacity(B);
        let mut draft_top16 = Vec::with_capacity(B);
        for row in 0..B {
            let row_logits = CudaF32View::new(
                logits.view(row * V * 4, V * 4)?,
                PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                V,
            )?;
            // Keep the canonical per-row top1/top16 arithmetic and tie rules.
            target_top1.push(CudaQwen35VocabTop1ArgsV1::bind(
                &d.stream,
                row_logits,
                d.verify_ids.view(row * 4, 4)?,
                1,
            )?);
            draft_top16.push(CudaQwen35VocabTopkArgsV1::bind(
                &d.stream,
                row_logits,
                d.candidate_ids.view(row * 64, 64)?,
                d.unary.view(row * 64, 64)?,
                1,
                16,
            )?);
        }
        let target_top1_batch = if d.batched_target_top1 {
            Some(CudaQwen35VocabTop1ArgsV1::bind(
                &d.stream, plan_f32_view(&logits, B * V)?,
                d.verify_ids.view(0, B * 4)?, B)?)
        } else { None };
        let input_prepare = if d.batched_output_head_prepare {
            let mut target_norm = CudaQwen35PrefillRmsnormV1::bind(
                &d.stream, plan_f32_view(&workspace.hidden_a, B * H)?,
                plan_f32_view(&d.head.norm_weight, H)?,
                plan_f32_view(&workspace.normalized, B * H)?, B, H, epsilon)?;
            target_norm.enable_canonical_small_batch()?;
            let target_quantize = CudaQ8_1ActivationQuantizeArgsV1::bind(
                &d.stream, plan_f32_view(&workspace.normalized, B * H)?,
                prefill_q8_1_view(&workspace.normalized_q8_1, B * H)?)?;
            let target_stores = if d.head_prepare_stores {
                Some(augur_backend::CudaQwen35HeadPrepareB8V1::bind_target(
                    &d.stream, plan_f32_view(&workspace.hidden_a, B*H)?, plan_f32_view(&d.head.norm_weight, H)?,
                    plan_f32_view(&workspace.normalized, B*H)?, prefill_q8_1_view(&workspace.normalized_q8_1, B*H)?,
                    plan_f32_view(d.plans[63].final_hidden_buffer(), H)?, plan_f32_view(&d.head.normalized_hidden, H)?,
                    prefill_q8_1_view(&d.head.activation_q8_1, H)?, epsilon)?)
            } else { None };
            Some(PreparedHeadInput { target_norm, target_quantize, target_stores })
        } else { None };
        let draft_top16_batch = if d.draft_top16_batch {
            Some(CudaQwen35VocabTopkArgsV1::bind(
                &d.stream,
                plan_f32_view(&logits, B * V)?,
                d.candidate_ids.view(0, B * 64)?,
                d.unary.view(0, B * 64)?,
                B,
                16,
            )?)
        } else {
            None
        };
        let after = cuda_device_allocation_snapshot();
        if after
            .successful_allocations
            .checked_sub(before.successful_allocations)
            != Some(1)
            || after.successful_frees != before.successful_frees
            || after.live_bytes.checked_sub(before.live_bytes) != Some(LOGIT_BYTES as u64)
        {
            return Err(invalid(
                "packed output head must own exactly one logit buffer",
            ));
        }
        Ok(Self {
            logits,
            projection,
            target_top1,
            target_top1_batch,
            input_prepare,
            draft_stores: std::cell::RefCell::new(None),
            draft_top16,
            draft_top16_batch,
            verify_logits_diagnostic: fixed_history_logits::directory(
                std::env::var_os("AUGUR_QWEN35_VERIFY_LOGITS_DIAGNOSTIC_DIR").as_deref())?,
        })
    }

    fn stage_activation(&self, d: &Driver, workspace: &PrefillWorkspace, row: usize) -> Result<()> {
        d.head.quantize_op.submit_for_plan(&d.stream)?;
        copy(
            &d.stream,
            &workspace.normalized_q8_1,
            row * H / 32 * 36,
            view(&d.head.activation_q8_1)?,
        )
    }

    fn preserve_last_logits(&self, d: &Driver) -> Result<()> {
        // Reference-boundary probes and the serial/replay path observe this
        // scratch. Norm/activation scratch already contain the final row.
        copy(
            &d.stream,
            &d.head.logits,
            0,
            self.logits.view((B - 1) * V * 4, V * 4)?,
        )
    }

    fn preserve_last_activation(&self, d: &Driver, workspace: &PrefillWorkspace) -> Result<()> {
        copy(&d.stream, &d.head.activation_q8_1, 0,
            workspace.normalized_q8_1.view((B - 1) * H / 32 * 36, H / 32 * 36)?)
    }

    pub(super) fn submit_draft(
        &self,
        d: &Driver,
        workspace: &PrefillWorkspace,
        draft: &DFlashQ8CudaBlock,
    ) -> Result<()> {
        // Driver has already waited on the draft producer's exact event epoch.
        // Draft final hidden is already normalized. NEVER apply target norm.
        if self.input_prepare.is_some() {
            if d.head_prepare_stores {
                let input = draft.final_hidden_rows_device().map_err(invalid)?;
                let mut cached = self.draft_stores.borrow_mut();
                if cached.is_none() {
                    *cached = Some(augur_backend::CudaQwen35HeadPrepareB8V1::bind_draft(
                        &d.stream, input, prefill_q8_1_view(&workspace.normalized_q8_1, B*H)?,
                        plan_f32_view(&d.head.normalized_hidden, H)?, prefill_q8_1_view(&d.head.activation_q8_1, H)?)?);
                }
                let bound = cached.as_ref().ok_or_else(|| invalid("draft head store binding missing"))?;
                bound.require_input(input)?;
                bound.submit(&d.stream)?;
            } else {
            // Independent 32-element quantization blocks keep their exact
            // inputs and reduction order; avoid eight scratch round trips.
            CudaQ8_1ActivationQuantizeArgsV1::bind(
                &d.stream, draft.final_hidden_rows_device().map_err(invalid)?,
                prefill_q8_1_view(&workspace.normalized_q8_1, B * H)?)?
                .submit_for_plan(&d.stream)?;
            copy(&d.stream, &d.head.normalized_hidden, 0,
                draft.final_hidden_row_device(B - 1).map_err(invalid)?.buffer_view())?;
            self.preserve_last_activation(d, workspace)?;
            }
            d.output_head_draft_prepare_batches.set(d.output_head_draft_prepare_batches.get() + 1);
        } else {
            for row in 0..B {
                copy(&d.stream, &d.head.normalized_hidden, 0,
                    draft.final_hidden_row_device(row).map_err(invalid)?.buffer_view())?;
                self.stage_activation(d, workspace, row)?;
            }
        }
        self.projection.submit(&d.stream)?;
        if let Some(top) = &self.draft_top16_batch {
            top.submit(&d.stream)?;
        } else {
            for top in &self.draft_top16 {
                top.submit(&d.stream)?;
            }
        }
        self.preserve_last_logits(d)
    }

    pub(super) fn submit_target(&self, d: &Driver, workspace: &PrefillWorkspace) -> Result<()> {
        if let Some(prepare) = &self.input_prepare {
            if let Some(stores) = &prepare.target_stores {
                stores.submit(&d.stream)?;
            } else {
            prepare.target_norm.submit(&d.stream)?;
            prepare.target_quantize.submit_for_plan(&d.stream)?;
            // Preserve all three final-row scratches observed by scalar replay
            // and boundary audits. No normalized/logit boundary is elided.
            copy(&d.stream, d.plans[63].final_hidden_buffer(), 0,
                workspace.hidden_a.view((B - 1) * H * 4, H * 4)?)?;
            copy(&d.stream, &d.head.normalized_hidden, 0,
                workspace.normalized.view((B - 1) * H * 4, H * 4)?)?;
            self.preserve_last_activation(d, workspace)?;
            }
            d.output_head_target_prepare_batches.set(d.output_head_target_prepare_batches.get() + 1);
        } else {
            for row in 0..B {
                copy(&d.stream, d.plans[63].final_hidden_buffer(), 0,
                    workspace.hidden_a.view(row * H * 4, H * 4)?)?;
                d.head.norm_op.submit_for_plan(&d.stream)?;
                self.stage_activation(d, workspace, row)?;
            }
        }
        self.projection.submit(&d.stream)?;
        if let Some(top) = &self.target_top1_batch {
            top.submit(&d.stream)?;
            d.target_top1_batches.set(d.target_top1_batches.get() + 1);
        } else {
            for top in &self.target_top1 { top.submit(&d.stream)?; }
        }
        self.preserve_last_logits(d)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn verify_logits_observer_is_bounded_to_short_early_rounds() {
        assert!(PackedOutputHead::observe_extent(47, 1));
        assert!(PackedOutputHead::observe_extent(512, 8));
        assert!(!PackedOutputHead::observe_extent(513, 1));
        assert!(!PackedOutputHead::observe_extent(47, 9));
        assert!(!PackedOutputHead::observe_extent(196600, 1));
    }
    use super::*;

    fn component_read(stream: &OwnedCudaStream, buffer: &OwnedCudaBuffer) -> Vec<u8> {
        let mut bytes = vec![0; buffer.len_bytes()];
        buffer.copy_to_host_sync(stream, &mut bytes).unwrap();
        bytes
    }

    fn component_f32(buffer: &OwnedCudaBuffer, offset: usize, n: usize) -> CudaF32View {
        CudaF32View::new(buffer.view(offset, n * 4).unwrap(),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1, n).unwrap()
    }

    fn component_q8(buffer: &OwnedCudaBuffer, offset: usize, n: usize) -> CudaQ8_1ActivationViewV1 {
        CudaQ8_1ActivationViewV1::new(buffer.view(offset, n / 32 * 36).unwrap(),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_1_ACTIVATION_CUDA_V1, n).unwrap()
    }

    fn component_time(stream: &OwnedCudaStream, mut submit: impl FnMut()) -> f32 {
        for _ in 0..3 { submit(); }
        stream.synchronize().unwrap();
        let begin = OwnedCudaEvent::create(0).unwrap();
        let end = OwnedCudaEvent::create(0).unwrap();
        begin.record(stream).unwrap();
        for _ in 0..20 { submit(); }
        end.record(stream).unwrap();
        end.synchronize().unwrap();
        begin.elapsed_ms(&end).unwrap() / 20.0
    }

    #[test]
    #[ignore = "requires exclusive SM70 GPU and explicit real norm/logits fixtures"]
    fn output_head_batch_native_components() {
        let before = cuda_device_allocation_snapshot();
        {
            let stream = OwnedCudaStream::create(0).unwrap();
            let model = PathBuf::from(std::env::var_os("AUGUR_QWEN35_HEAD_COMPONENT_GGUF").unwrap());
            let manifest = GgufManifest::inspect(&model,
                GgufAuditOptions { hash_file: false, ..Default::default() }).unwrap();
            let weight = manifest.open_tensor_reader(&model).unwrap()
                .read_tensor_f32("output_norm.weight").unwrap();
            assert_eq!(weight.len(), H);
            let weight: Vec<u8> = weight.iter().flat_map(|x| x.to_le_bytes()).collect();
            println!("{}", json!({"event":"real_norm_weight", "bytes":weight.len(),
                "sha256":format!("{:x}", Sha256::digest(&weight)), "model":model,
                "scope":"header_bound_tensor_payload_not_new_full_model_hash"}));
            let alloc = |n: usize| {
                let b = OwnedCudaBuffer::allocate(0, n).unwrap();
                b.copy_from_host_sync(&stream, &vec![0xa5; n]).unwrap(); b
            };
            let w = alloc(H * 4); w.copy_from_host_sync(&stream, &weight).unwrap();
            let x = alloc(B * H * 4);
            let scalar_x = alloc(H * 4);
            let scalar_norm = alloc(H * 4);
            let scalar_q8 = alloc(H / 32 * 36);
            let old_norm = alloc(128 + B * H * 4);
            let new_norm = alloc(128 + B * H * 4);
            let old_q8 = alloc(128 + B * H / 32 * 36);
            let new_q8 = alloc(128 + B * H / 32 * 36);
            // Test the actually compiled scalar plan (split norm in the full
            // product recipe), not merely another batched-norm implementation.
            let norm = CudaPlanRmsnormF32::bind(&stream, component_f32(&scalar_x,0,H),
                component_f32(&w,0,H), component_f32(&scalar_norm,0,H),1e-6).unwrap();
            let quantize = CudaQ8_1ActivationQuantizeArgsV1::bind(&stream,
                component_f32(&scalar_norm,0,H),component_q8(&scalar_q8,0,H)).unwrap();
            let mut batch_norm = CudaQwen35PrefillRmsnormV1::bind(&stream,
                component_f32(&x,0,B*H),component_f32(&w,0,H),
                component_f32(&new_norm,64,B*H),B,H,1e-6).unwrap();
            batch_norm.enable_canonical_small_batch().unwrap();
            let batch_quantize = CudaQ8_1ActivationQuantizeArgsV1::bind(&stream,
                component_f32(&new_norm,64,B*H),component_q8(&new_q8,64,B*H)).unwrap();
            let bound_before = cuda_device_allocation_snapshot();
            let stores = augur_backend::CudaQwen35HeadPrepareB8V1::bind_target(&stream,
                component_f32(&x,0,B*H),component_f32(&w,0,H),component_f32(&new_norm,64,B*H),
                component_q8(&new_q8,64,B*H),component_f32(&scalar_x,0,H),
                component_f32(&scalar_norm,0,H),component_q8(&scalar_q8,0,H),1e-6).unwrap();
            let draft_stores = augur_backend::CudaQwen35HeadPrepareB8V1::bind_draft(&stream,
                component_f32(&x,0,B*H),component_q8(&new_q8,64,B*H),
                component_f32(&scalar_norm,0,H),component_q8(&scalar_q8,0,H)).unwrap();
            assert_eq!(bound_before.successful_allocations,cuda_device_allocation_snapshot().successful_allocations);
            let submit_old = |capture_norm: bool| {
                for row in 0..B {
                    copy(&stream,&scalar_x,0,x.view(row*H*4,H*4).unwrap()).unwrap();
                    norm.submit_for_plan(&stream).unwrap();
                    quantize.submit_for_plan(&stream).unwrap();
                    copy(&stream,&old_q8,64+row*H/32*36,view(&scalar_q8).unwrap()).unwrap();
                    if capture_norm {
                        copy(&stream,&old_norm,64+row*H*4,view(&scalar_norm).unwrap()).unwrap();
                    }
                }
            };
            let submit_new = || {
                batch_norm.submit(&stream).unwrap();
                batch_quantize.submit_for_plan(&stream).unwrap();
                copy(&stream,&scalar_x,0,x.view((B-1)*H*4,H*4).unwrap()).unwrap();
                copy(&stream,&scalar_norm,0,new_norm.view(64+(B-1)*H*4,H*4).unwrap()).unwrap();
                copy(&stream,&scalar_q8,0,new_q8.view(64+(B-1)*H/32*36,H/32*36).unwrap()).unwrap();
            };
            for case in 0..14 {
                let input: Vec<u8> = if case >= 9 {
                    let root = PathBuf::from(std::env::var_os("AUGUR_QWEN35_BOUNDARY_CAPTURE_ROOT").unwrap());
                    let layer = [0,31,50,51,63][case-9];
                    let bytes = std::fs::read(root.join(format!("layer{layer}-ffn_out-full.f32le.bin"))).unwrap();
                    assert!(bytes.len() >= B*H*4); bytes[..B*H*4].to_vec()
                } else { (0..B*H).flat_map(|i| {
                    let value = match case {
                        0..=2 => ((i*13+case*37)%2053) as f32 / 83.0 - 12.0,
                        3 => if i%2==0 {0.0} else {-0.0},
                        4 => f32::from_bits((i as u32%29)+1),
                        5 => if i%2==0 {1e18} else {-1e-18},
                        6 if i%H==31 => f32::INFINITY,
                        7 if i%H==256 => f32::from_bits(0x7fc01234),
                        8 if i%H==H-1 => f32::NEG_INFINITY,
                        _ => ((i*17%113) as f32 - 56.0)/19.0,
                    }; value.to_le_bytes()
                }).collect() };
                x.copy_from_host_sync(&stream,&input).unwrap();
                let allocation_before = cuda_device_allocation_snapshot();
                submit_old(true); stream.synchronize().unwrap();
                let last = (component_read(&stream,&scalar_x),component_read(&stream,&scalar_norm),
                    component_read(&stream,&scalar_q8));
                submit_new(); stream.synchronize().unwrap();
                assert_eq!(component_read(&stream,&old_norm),component_read(&stream,&new_norm),"norm case {case}");
                assert_eq!(component_read(&stream,&old_q8),component_read(&stream,&new_q8),"Q8 case {case}");
                assert_eq!(last,(component_read(&stream,&scalar_x),component_read(&stream,&scalar_norm),
                    component_read(&stream,&scalar_q8)),"scratch case {case}");
                // Poison all outputs before testing the producer-side stores;
                // preserve the existing guard bytes around batch destinations.
                for b in [&new_norm,&new_q8] {
                    let mut bytes = vec![0xa5; b.len_bytes()];
                    let n = bytes.len(); bytes[64..n-64].fill(0xff);
                    b.copy_from_host_sync(&stream,&bytes).unwrap();
                }
                for b in [&scalar_x,&scalar_norm,&scalar_q8] {
                    b.copy_from_host_sync(&stream,&vec![0xff;b.len_bytes()]).unwrap();
                }
                stores.submit(&stream).unwrap(); stream.synchronize().unwrap();
                assert_eq!(component_read(&stream,&old_norm),component_read(&stream,&new_norm),"stored norm case {case}");
                assert_eq!(component_read(&stream,&old_q8),component_read(&stream,&new_q8),"stored Q8 case {case}");
                assert_eq!(last,(component_read(&stream,&scalar_x),component_read(&stream,&scalar_norm),
                    component_read(&stream,&scalar_q8)),"producer scratch case {case}");
                for b in [&new_norm,&new_q8] {
                    let bytes=component_read(&stream,b);
                    assert!(bytes[..64].iter().chain(bytes[bytes.len()-64..].iter()).all(|&x|x==0xa5));
                }
                assert_eq!(component_read(&stream,&x),input);
                assert_eq!(component_read(&stream,&w),weight);
                let allocation_after=cuda_device_allocation_snapshot();
                assert_eq!(allocation_before.successful_allocations,allocation_after.successful_allocations);
                assert_eq!(allocation_before.successful_frees,allocation_after.successful_frees);
                // Draft is already normalized: compare row-quantize directly
                // with one contiguous pack, without applying target norm.
                for row in 0..B {
                    CudaQ8_1ActivationQuantizeArgsV1::bind(&stream,
                        component_f32(&x,row*H*4,H),component_q8(&old_q8,64+row*H/32*36,H))
                        .unwrap().submit_for_plan(&stream).unwrap();
                }
                CudaQ8_1ActivationQuantizeArgsV1::bind(&stream,component_f32(&x,0,B*H),
                    component_q8(&new_q8,64,B*H)).unwrap().submit_for_plan(&stream).unwrap();
                stream.synchronize().unwrap();
                assert_eq!(component_read(&stream,&old_q8),component_read(&stream,&new_q8),"draft pack case {case}");
                let mut poison = vec![0xa5;new_q8.len_bytes()];
                let n = poison.len(); poison[64..n-64].fill(0xff);
                new_q8.copy_from_host_sync(&stream,&poison).unwrap();
                for b in [&scalar_norm,&scalar_q8] { b.copy_from_host_sync(&stream,&vec![0xff;b.len_bytes()]).unwrap(); }
                draft_stores.require_input(component_f32(&x,0,B*H)).unwrap();
                draft_stores.submit(&stream).unwrap(); stream.synchronize().unwrap();
                let expected = component_read(&stream,&old_q8);
                assert_eq!(expected,component_read(&stream,&new_q8),"draft producer Q8 case {case}");
                assert_eq!(component_read(&stream,&scalar_norm),input[(B-1)*H*4..],"draft last normalized case {case}");
                assert_eq!(component_read(&stream,&scalar_q8),expected[64+(B-1)*H/32*36..64+B*H/32*36],"draft last Q8 case {case}");
                assert_eq!(component_read(&stream,&scalar_x),last.0,"draft must not touch target hidden");
                let after = cuda_device_allocation_snapshot();
                assert_eq!(allocation_before.successful_allocations,after.successful_allocations);
                assert_eq!(allocation_before.successful_frees,after.successful_frees);
                assert_eq!(allocation_before.live_bytes,after.live_bytes);
                println!("{}",json!({"event":"prepare_case","case":case,"exact":true}));
                if case==1 {
                    let off=component_time(&stream,||submit_old(false));
                    let on=component_time(&stream,submit_new);
                    println!("{}",json!({"event":"prepare_timing","off_ms":off,"on_ms":on,
                        "scope":"stream_envelope_including_enqueue_gaps_not_HTTP"}));
                    let stored=component_time(&stream,||stores.submit(&stream).unwrap());
                    println!("{}",json!({"event":"producer_store_timing","batched_ms":on,"stored_ms":stored,
                        "scope":"stream_envelope_including_enqueue_gaps_not_HTTP"}));
                }
            }

            let real_path=PathBuf::from(std::env::var_os("AUGUR_QWEN35_HEAD_COMPONENT_LOGITS").unwrap());
            let real=std::fs::read(real_path).unwrap(); assert_eq!(real.len(),V*4);
            let logits=alloc(B*V*4);
            let scalar_ids=alloc(128+B*4); let batch_ids=alloc(128+B*4);
            let top= (0..B).map(|row|CudaQwen35VocabTop1ArgsV1::bind(&stream,
                component_f32(&logits,row*V*4,V),scalar_ids.view(64+row*4,4).unwrap(),1).unwrap())
                .collect::<Vec<_>>();
            let batch=CudaQwen35VocabTop1ArgsV1::bind(&stream,component_f32(&logits,0,B*V),
                batch_ids.view(64,B*4).unwrap(),B).unwrap();
            for case in 0..8 {
                let mut data=Vec::with_capacity(B*V*4);
                for row in 0..B {
                    if case==0 {data.extend_from_slice(&real);continue;}
                    for i in 0..V {
                        let x=match case {
                            1=>((i*17+row*5)%2011) as f32/97.0,
                            2=>1.0,
                            3=>if i%2==0 {0.0} else {-0.0},
                            4=>f32::from_bits((i%31+1) as u32),
                            5 if i==row*257=>f32::NAN,
                            6 if i==V-1-row=>f32::INFINITY,
                            7 if i==row=>f32::NEG_INFINITY,
                            _=>-3.0,
                        };data.extend_from_slice(&x.to_le_bytes());
                    }
                }
                logits.copy_from_host_sync(&stream,&data).unwrap();
                for op in &top {op.submit(&stream).unwrap();}
                batch.submit(&stream).unwrap(); stream.synchronize().unwrap();
                assert_eq!(component_read(&stream,&scalar_ids),component_read(&stream,&batch_ids));
                let ids=component_read(&stream,&batch_ids);
                assert!(ids[..64].iter().chain(ids[ids.len()-64..].iter()).all(|&x|x==0xa5));
                if case>=5 {assert!(ids[64..64+B*4].iter().all(|&x|x==255));}
                assert_eq!(component_read(&stream,&logits),data);
                println!("{}",json!({"event":"top1_case","case":case,"exact":true}));
                if case==0 {
                    let off=component_time(&stream,||for op in &top {op.submit(&stream).unwrap();});
                    let on=component_time(&stream,||batch.submit(&stream).unwrap());
                    println!("{}",json!({"event":"top1_timing","off_ms":off,"on_ms":on,
                        "scope":"stream_envelope_including_enqueue_gaps_not_HTTP"}));
                }
            }
            stream.synchronize().unwrap();
        }
        let after=cuda_device_allocation_snapshot();
        assert_eq!(after.live_bytes,before.live_bytes);
        assert_eq!(after.live_allocations,before.live_allocations);
        assert_eq!(after.successful_allocations-before.successful_allocations,
            after.successful_frees-before.successful_frees);
        println!("{}",json!({"event":"head_component_cleanup",
            "allocations":after.successful_allocations-before.successful_allocations,
            "frees":after.successful_frees-before.successful_frees,"live_bytes_delta":0}));
    }
    #[test]
    fn draft_top16_batch_requires_feature_and_exact_opt_in() {
        use std::ffi::OsStr;
        assert!(!draft_top16_batch_switch(None, false).unwrap());
        assert!(!draft_top16_batch_switch(None, true).unwrap());
        assert!(draft_top16_batch_switch(Some(OsStr::new("1")), true).unwrap());
        assert!(draft_top16_batch_switch(Some(OsStr::new("1")), false).is_err());
        for value in ["", "0", "true", "1 ", "8"] {
            assert!(draft_top16_batch_switch(Some(OsStr::new(value)), true).is_err());
        }
    }

    #[test]
    fn draft_top16_batch_preserves_contiguous_output_extent() {
        assert_eq!(B, 8);
        assert_eq!(B * 16 * 4, 512);
        assert_eq!(LOGIT_BYTES, B * V * 4);
    }

    #[test]
    fn head_workspace_is_only_eight_logit_rows_and_reuses_q8_activation() {
        assert_eq!(LOGIT_BYTES, 7_946_240);
        assert_eq!(B * H / 32 * 36, 46_080);
        assert_eq!(POLICY, "packed_q8_exact_full8_draft_and_target_v1");
    }

    #[test]
    fn output_head_batches_preserve_last_row_extents_without_new_storage() {
        assert_eq!((B - 1) * H * 4 + H * 4, B * H * 4);
        assert_eq!((B - 1) * H / 32 * 36 + H / 32 * 36, B * H / 32 * 36);
        assert_eq!((B - 1) * V * 4 + V * 4, LOGIT_BYTES);
        assert_eq!(B * core::mem::size_of::<u32>(), 32);
    }

    #[test]
    fn output_head_batch_flags_fail_closed_outside_exact_recipe() {
        use std::ffi::OsStr;
        assert!(!submission_batch_switch(None, false).unwrap());
        assert!(submission_batch_switch(Some(OsStr::new("1")), true).unwrap());
        assert!(submission_batch_switch(Some(OsStr::new("1")), false).is_err());
        for value in ["", "0", "true", "1 "] {
            assert!(submission_batch_switch(Some(OsStr::new(value)), true).is_err());
        }
    }
}
