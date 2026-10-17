//! Real global input chunks; feature history is an intersection, not a scheduler.
use super::*;

pub(super) fn capture(d: &mut Driver, prompt: &[usize], epsilon: f32, theta: f32,
    mut capture_fc:Option<&mut crate::dflash_q8_backbone_cuda::capture_fc::CaptureFc>,
) -> Result<Value> {
    let history = d.context_rows;
    let (batch_start, feature_start) = history_capture::exact_input_schedule(prompt.len(), history)?;
    if !d.llama_exact_input_rows || !d.llama_history_full || !d.llama_full_original
        || !d.llama_bf16_scalars || !d.llama_rotated_kv {
        return Err(invalid("exact input capture requires its complete recipe"));
    }
    let before = cuda_device_allocation_snapshot();
    let timer = Instant::now();
    d.poisoned = true;
    let mut owner = CaptureOwner { driver:d, core:None, prefix_profile:None, drained:false };
    owner.driver.cursor = batch_start;
    owner.core = Some(BatchedVerifyAudit::allocate_exact_input(owner.driver, prompt.len(), epsilon, theta)?);
    let mut bind_calls = 0;
    let mut blocks = 0;
    let mut tail_rows = 0;
    let mut qpn_projections = 0;
    let feature_copies_before = owner.driver.history_feature_copy_enqueues.get();
    {
        let core = owner.core.as_ref().ok_or_else(|| invalid("exact input core"))?;
        let d = &mut *owner.driver;
        let mut bound_rows = 0;
        let mut operations = Vec::new();
        for cursor in (batch_start..prompt.len()).step_by(history_capture::WIDTH) {
            d.check_session_cancellation("exact input block admission")?;
            d.cursor = cursor;
            let rows = (prompt.len()-cursor).min(history_capture::WIDTH);
            if rows != bound_rows {
                operations = core.bind_operations_rows(d,rows)?;
                let selected = prefill_input_integer::qpn_projection_count(&operations);
                if bind_calls > 0 && selected != qpn_projections {
                    return Err(invalid("exact input FFN profile changed across widths"));
                }
                qpn_projections = selected;
                bound_rows = rows;
                bind_calls += 1;
            }
            let ids = prompt[cursor..cursor+rows].iter().flat_map(|&id| (id as u32).to_le_bytes()).collect::<Vec<_>>();
            core.batch_ids.copy_from_host_sync(&d.stream,&ids)?;
            if blocks == 1 {
                eprintln!("QWEN35_HISTORY_CAPTURE_PROGRESS={}",json!({
                    "schema":"qwen35-history-capture-progress-v1","pid":std::process::id(),
                    "completed_rows":cursor-feature_start,"history_rows":history,
                    "history_start":feature_start,"prompt_tokens":prompt.len(),
                    "batch_width":history_capture::WIDTH,"proof":"same_stream_h2d_sync_after_history_backbone",
                    "visible_tokens":0,"exact_input_rows":true}));
            }
            core.submit_exact_input_history(d,&operations,rows)?;
            if let Some(fc)=capture_fc.as_mut() {
                let first=cursor.max(feature_start);
                let last=(cursor+rows).min(prompt.len());
                if last>first {
                    fc.submit_ready(&d.stream,first,last-first,d.identity,d.generation).map_err(invalid)?;
                }
            }
            blocks += 1;
            tail_rows = rows;
        }
        copy(&d.stream,d.plans[63].final_hidden_buffer(),0,
            core.workspace.hidden_output(63,tail_rows)?.buffer_view()
                .subview((tail_rows-1)*H*4,H*4)?)?;
        d.head.submit(&d.stream)?;
        d.top1(view(&d.token)?)?;
    }
    owner.driver.stream.synchronize()?;
    owner.drained = true;
    let allocated = cuda_device_allocation_snapshot();
    let q8_input_pack_elisions = owner.core.as_ref().unwrap().workspace.cublas.q8_input_pack_elisions();
    let gdn_bf16_pair_submissions = owner.core.as_ref().unwrap().workspace.cublas.gdn_bf16_pair_submissions();
    drop(owner.core.take());
    let after = cuda_device_allocation_snapshot();
    let workspace = allocated.live_bytes.checked_sub(before.live_bytes).ok_or_else(|| invalid("exact workspace counter"))?;
    let allocations = after.successful_allocations.checked_sub(before.successful_allocations).ok_or_else(|| invalid("exact allocation counter"))?;
    let frees = after.successful_frees.checked_sub(before.successful_frees).ok_or_else(|| invalid("exact free counter"))?;
    if workspace == 0 || allocations != frees || before.live_bytes != after.live_bytes
        || before.live_allocations != after.live_allocations {
        return Err(invalid("exact input workspace cleanup"));
    }
    owner.driver.cursor=prompt.len();
    owner.driver.poisoned=false;
    Ok(json!({"schema":"qwen35-canonical-exact-input-history-v1",
        "arithmetic_policy":if qpn_projections>0 {"qpn_original_f16_ffn_llama_other_realM_global512_history_v1"}
            else {"llama_original_realM_global512_history_intersection_llama_norm_v1"},
        "rotated_full_attention":true,"llama_full_history_requested":true,
        "llama_exact_input_rows_requested":true,
        "llama_full_original_selected_projection_count":400-qpn_projections,
        "llama_full_original_executed_projection_count":(400-qpn_projections)*blocks,
        "qpn_ffn_selected_projection_count":qpn_projections,
        "qpn_ffn_executed_projection_count":qpn_projections*blocks,
        "llama_bf16_scalars_selected_projection_count":96,
        "llama_bf16_scalars_executed_projection_count":96*blocks,
        "llama_full_original_swiglu_count":(64-qpn_projections/3)*blocks,
        "input_norm_arithmetic":"llama_sm70_hidden5120_rms1024_xor_fast_v1",
        "llama_input_norm_selected_count":128,
        "llama_input_norm_executed_count":128*blocks,
        "verifier_and_decode_llama_history_math":false,
        "feature_start":feature_start,"feature_end":prompt.len(),"feature_rows":history,
        "batch_width":history_capture::WIDTH,"batch_count":blocks,"final_real_rows":tail_rows,
        "serial_tail_rows":0,"global_boundary_audit":false,
        "input_batch_start":batch_start,"input_rows":prompt.len()-batch_start,
        "first_capture_source_row":feature_start-batch_start,"batch_feature_rows":history,"scalar_feature_rows":0,
        "tail_arithmetic":"explicit_input_mmvq_mmq_gemm_by_real_rows",
        "input_attention_dispatch":"llama_sm70_vec_tile_mma_streamk_v1",
        "input_attention_physical_kv_padding":256,
        "input_attention_mma_meta_reserved_bytes":25559040,
        "global_remainder_dispatch_qualified":false,"native_m1_fused_launch_reproduced":false,
        "output_head_calls":1,"feature_d2d_bytes":history*5*H*4,
        "feature_copy_calls":owner.driver.history_feature_copy_enqueues.get()-feature_copies_before,
        "history_batched_feature_copies":owner.driver.history_batched_feature_copies,
        "q8_input_pack_elisions":q8_input_pack_elisions,
        "gdn_bf16_pair_submissions":gdn_bf16_pair_submissions,
        "capture_destination":"driver_initialized_features_direct","extra_feature_staging_bytes":0,
        "verifier_batch_width":B,"generated_warmup_tokens":0,"padding_tokens":0,
        "publication_enabled":false,"operation_bind_calls":bind_calls,
        "conditioning_wall_ms":timer.elapsed().as_secs_f64()*1000.0,
        "workspace_bytes":workspace,"workspace_released_bytes":workspace,
        "live_bytes_before":before.live_bytes,"live_bytes_with_workspace":allocated.live_bytes,
        "live_bytes_after":after.live_bytes,"allocations":allocations,"frees":frees,
        "workspace_released_before_auxiliary_and_decode":true,"product_qualification":false}))
}
