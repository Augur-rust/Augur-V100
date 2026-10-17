//! Separate B8 speed-first qualification lane: never weakens canonical B1 tests.
use super::*;
use crate::checkpoint_target_cuda::transfer_tests::ids;
#[derive(Clone)]
enum Replay {Prompt(Vec<u32>),Selected(u32),Batch(Vec<u32>,usize)}
fn flatten(ops:&[Replay])->Vec<u32> {
    let mut ids=Vec::new();for op in ops {match op {
        Replay::Prompt(v)=>ids.extend(v),Replay::Selected(id)=>ids.push(*id),Replay::Batch(v,n)=>ids.extend(&v[..*n]),
    }}ids
}
fn restore(t:&mut CheckpointCudaTarget,b:&mut AugurDflashBridge,ops:&[Replay])->f64 {
    // Retire the quiescent request plan before the original exclusive KV reset.
    // This setup/rebinding is outside timed generation; no reset contract is weakened.
    b.release_batched_verifier().unwrap();
    t.reset_empty().unwrap();let start=Instant::now();
    for op in ops {match op {
        Replay::Prompt(ids)=>consume_prompt(t,ids),Replay::Selected(id)=>{assert_eq!(t.greedy_next_token().unwrap(),*id);consume_selected(t);},
        Replay::Batch(path,n)=>{b.prepare_batched_verifier(t).unwrap();b.replay_batched_inputs(t,path,*n).unwrap();},
    }}b.prepare_batched_verifier(t).unwrap();t.stream.synchronize().unwrap();assert_eq!(t.consumed,flatten(ops).len());start.elapsed().as_secs_f64()
}
fn resident(t:&mut CheckpointCudaTarget,b:&mut AugurDflashBridge,budget:usize,eos:&[u32],audit:bool)->Run {
    resident_with_hidden_file(t,b,budget,eos,audit,None)
}
fn resident_with_hidden_file(t:&mut CheckpointCudaTarget,b:&mut AugurDflashBridge,budget:usize,eos:&[u32],audit:bool,hidden_path:Option<&std::path::Path>)->Run {
    resident_with_teacher_replay(t,b,budget,eos,audit,hidden_path,None)
}
fn resident_with_teacher_replay(t:&mut CheckpointCudaTarget,b:&mut AugurDflashBridge,budget:usize,eos:&[u32],audit:bool,hidden_path:Option<&std::path::Path>,teacher_replay:Option<&complete_teacher::TeacherReplay<'_>>)->Run {
    assert!(hidden_path.is_none() || audit);
    let mut hidden_writer=hidden_path.map(|p|std::fs::OpenOptions::new().write(true).create_new(true).open(p).unwrap());
    let mut hidden_offset=0usize;
    let supervision_enabled=hidden_path.is_some() && std::env::var("AUGUR_SELECTOR_SUPERVISION").ok().as_deref()==Some("1");
    let tail_path=hidden_path.filter(|_|supervision_enabled).map(|path| {
        let name=path.file_stem().unwrap().to_str().unwrap().strip_suffix("-hidden").unwrap();
        path.with_file_name(format!("{name}-target-tail.f32"))
    });
    let mut tail_writer=tail_path.as_ref().map(|p|std::fs::OpenOptions::new().write(true).create_new(true).open(p).unwrap());
    let mut tail_offset=0usize;
    let supervision=supervision_enabled.then(||b.batched.as_ref().unwrap().selector_supervision_workspace().unwrap());
    assert!(teacher_replay.is_none() || supervision_enabled);
    let before=cuda_device_allocation_snapshot();let start=Instant::now();let mut output=Vec::new();let mut rounds=Vec::new();
    let stop=loop {
        let begin=start.elapsed().as_secs_f64();let anchor=t.greedy_next_token().unwrap();if eos.contains(&anchor){break "eos";}
        let p=b.propose_resident(t).unwrap();let path=p.path().to_vec();let proposal_end=start.elapsed().as_secs_f64();
        // Read the already produced candidates only in the independent audit replay.
        let head=audit.then(||if hidden_path.is_some(){b.head.read_candidate_diagnostics(anchor).unwrap()}else{b.head.read_diagnostics(anchor).unwrap()});
        let selector_gates = audit.then(|| {
            let draft = b.resident.as_ref().unwrap().draft.as_ref().unwrap();
            let (source, _, _) = draft.selector_inputs_device().unwrap();
            assert_eq!(source.len_bytes(), 8 * 256 * 4);
            let mut bytes = vec![0u8; source.len_bytes()];
            source.copy_to_host_sync_with_submission_proof(
                draft.stream(), &mut bytes, draft.stream().submission_receipt().unwrap(),
            ).unwrap();
            let gates: Vec<f32> = bytes.chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap())).collect();
            assert!(gates.iter().all(|v| v.is_finite()));
            gates
        });
        let selector_input = (audit && (hidden_path.is_some() || std::env::var("AUGUR_DFLASH_GATE_NUMERICAL_AUDIT").ok().as_deref()==Some("1"))).then(|| {
            let draft=b.resident.as_ref().unwrap().draft.as_ref().unwrap();
            let source=draft.final_hidden_rows_device().unwrap().buffer_view();
            assert_eq!(source.len_bytes(),8*5120*4);
            let mut bytes=vec![0u8;source.len_bytes()];
            source.copy_to_host_sync_with_submission_proof(draft.stream(),&mut bytes,draft.stream().submission_receipt().unwrap()).unwrap();
            let values:Vec<f32>=bytes.chunks_exact(4).map(|v|f32::from_le_bytes(v.try_into().unwrap())).collect();
            assert!(values.iter().all(|v|v.is_finite()));
            if let Some(writer)=hidden_writer.as_mut() {
                writer.write_all(&bytes).unwrap();let offset=hidden_offset;hidden_offset+=bytes.len();
                json!({"shape":[8,5120],"dtype":"f32_le","stage":"draft_final_rmsnorm_before_gate_and_shared_head",
                    "file":hidden_path.unwrap().file_name().unwrap().to_str().unwrap(),"offset_bytes":offset,"bytes":bytes.len(),
                    "sha256":format!("{:x}",Sha256::digest(&bytes))})
            } else {json!({"shape":[8,5120],"dtype":"f32","stage":"draft_final_rmsnorm_before_gate_and_shared_head","values":values})}
        });
        if let Some(h)=&head { assert_eq!(h.path,path); }
        let extent=p.initialized_feature_range();
        let consumed_target=tail_writer.as_mut().map(|writer| {
            let draft=b.resident.as_ref().unwrap().draft.as_ref().unwrap();
            let (first,end,generation,bytes)=draft.selector_consumed_target_tail().unwrap();
            assert_eq!((first,end),extent);assert_eq!(end,t.consumed);assert!(end>0);
            writer.write_all(&bytes).unwrap();let offset=tail_offset;tail_offset+=bytes.len();
            json!({"file":tail_path.as_ref().unwrap().file_name().unwrap().to_str().unwrap(),
                "offset_bytes":offset,"bytes":bytes.len(),"shape":[5120],"dtype":"f32_le",
                "sha256":format!("{:x}",Sha256::digest(&bytes)),"request_generation":generation,
                "proposal_generation":p.generation,"context_start":first,"consumed_end_exclusive":end,
                "absolute_position":end-1,"pending_anchor_excluded":true,
                "stage":"target_post_residual_5_layers_fc_then_hidden_norm_last_consumed",
                "available_before_current_verify":true})
        });
        let complete_teacher=teacher_replay.and_then(|replay|replay.teacher_path(rounds.len(),&output,t.consumed,&path))
            .map(|forced|b.probe_complete_b8_teacher(t,&p,forced,&head.as_ref().unwrap().candidate_ids,supervision.as_ref().unwrap()).unwrap());
        let v=b.verify_resident_batched(t,p,budget-output.len(),eos).unwrap();
        let mut rejection=v.first_rejection.clone();
        let mut collection=Value::Null;
        if let Some(head)=head {
            if let Some(reject)=&v.first_rejection {
                let offset=reject["offset"].as_u64().unwrap() as usize;
                assert_eq!(offset,v.output_tokens.len());
                let correct=reject["correct_target_token"].as_u64().unwrap() as u32;
                rejection=Some(classify_rejection(&head,offset,correct));
            }
            let rejected=rejection.as_ref().map(|r|r["correct_target_token"].as_u64().unwrap() as u32);
            let terminal_eos=(v.stop=="eos_pending").then(||t.greedy_next_token().unwrap());
            let labels=label_mask(&v.output_tokens,rejected,terminal_eos);
            let teacher=supervision.as_ref().map(|workspace|workspace.read(&head.candidate_ids,&labels).unwrap());
            collection=json!({"initialized_feature_range":extent,"draft_query_absolute_start":extent.1,
                "proposal_path":path,"candidate_ids":head.candidate_ids,"unary_scores":head.unary_scores,
                "selector_scores":head.transition_scores,"canonical_target_labels":labels,
                "selector_gate_f32":selector_gates.unwrap(),"selector_gate_shape":[8,256],
                "selector_input":selector_input,"selector_gate_precision":b.weights.selector_gate_precision(),
                "target_teacher":teacher,"consumed_target_input":consumed_target,
                "complete_block_teacher":complete_teacher,
                "selector_transition_scale":b.head.selector_scale,"selector_path":b.head.selector_path,
                "selector_early_position_scales":(b.head.selector_path=="position").then_some(b.head.selector_early_scales),
                "target_label_policy":"B8 causal prefix; not B1 equivalence",
                "row0_is_anchor_not_acceptance_trial":true,"EOS_comparison_censored":terminal_eos.is_some(),
                "post_first_rejection_labels_masked":true,"old_target_samples":false});
        }
        assert!(!v.output_tokens.is_empty());
        rounds.push(json!({"output_start":output.len(),"canonical_tokens":v.output_tokens,"proposal_path":path,
            "target_label_policy":"B8 speed-first causal prefix; not B1 numerical equivalence",
            "first_rejection":rejection,"stop":v.stop,"start_consumed":v.start_consumed,"end_consumed":v.end_consumed,
            "wall_s":start.elapsed().as_secs_f64()-begin,"proposal_and_anchor_s":proposal_end-begin,
            "batched_verification_commit_s":v.batch_verify_commit_s,"collection":collection}));
        if let Some(replay)=teacher_replay {replay.check_round(rounds.len()-1,rounds.last().unwrap());}
        let progress_before=output.len()/512;output.extend(v.output_tokens);
        if hidden_path.is_some() && output.len()/512>progress_before {println!("AUGUR_SELECTOR_PROGRESS={}",json!({"output_tokens":output.len(),"rounds":rounds.len()}));std::io::stdout().flush().unwrap();}
        if v.stop=="eos_pending"{break "eos";}if v.stop=="length_pending"{break "length";}
    };
    t.stream.synchronize().unwrap();let wall=start.elapsed().as_secs_f64();let after=cuda_device_allocation_snapshot();
    assert_eq!(before.successful_allocations,after.successful_allocations);assert_eq!(before.successful_frees,after.successful_frees);
    if let Some(writer)=tail_writer.as_mut(){writer.flush().unwrap();}
    let pending=t.greedy_next_token().unwrap();Run {output,cursor:t.consumed,logits:t.logits_snapshot().unwrap(),pending,stop,
        wire:json!({"lane":if audit{"audit_replay_NOT_timing"}else{"resident"},"generation_wall_s":wall,"rounds":rounds,"GPU_allocations":0,"GPU_frees":0,"optional_diagnostic_IO":audit,
            "supervision_schema":supervision_enabled.then_some("quasar-selector-supervision-v1"),
            "target_tail_file":tail_path.as_ref().map(|p|p.file_name().unwrap().to_str().unwrap()),
            "target_tail_bytes":tail_offset,"target_tail_layout":"round_major_5120_f32_le",
            "complete_teacher_schema":teacher_replay.map(|_|"quasar-complete-b8-conditional-teacher-v1"),
            "complete_teacher_stride":teacher_replay.map(|r|r.stride),
            "probe_baseline_rounds_exact":teacher_replay.is_some()})}
}
fn accept_case(t:&mut CheckpointCudaTarget,b:&mut AugurDflashBridge,case:&Value,ops:&[Replay],eos:&[u32],initialized:bool)->(Vec<u32>,u32,&'static str,Value) {
    let budget=case["max_new_tokens"].as_u64().unwrap() as usize;let name=case["id"].as_str().unwrap();
    // Verify actual same-owner append against the recorded mixed B8/B1 schedule.
    let mut batch_ref=initialized.then(||resident(t,b,budget,eos,false));
    let mut serial_ref=None;let mut timed=Vec::new();
    for rep in 0..2 {for lane in if rep==0 {[false,true]}else{[true,false]} {
        let setup=restore(t,b,ops);let mut r=if lane{resident(t,b,budget,eos,false)}else{no_draft(t,budget,eos)};
        let reference=if lane{&mut batch_ref}else{&mut serial_ref};if let Some(expected)=reference.as_ref(){equal(&r,expected);}
        r.wire["repeat"]=json!(rep);r.wire["setup_exact_replay_wall_s"]=json!(setup);r.wire["content_tokens"]=json!(r.output.len());
        r.wire["output_tokens"]=json!(r.output);timed.push(r.wire.clone());if reference.is_none(){*reference=Some(r);}
    }}
    restore(t,b,ops);let audit=resident(t,b,budget,eos,true);equal(&audit,batch_ref.as_ref().unwrap());
    for r in timed.iter().filter(|r|r["lane"]=="resident") {
        let a=r["rounds"].as_array().unwrap();let b=audit.wire["rounds"].as_array().unwrap();assert_eq!(a.len(),b.len());
        for (a,b) in a.iter().zip(b) {
            for k in ["canonical_tokens","proposal_path","stop","start_consumed","end_consumed"]{assert_eq!(a[k],b[k]);}
            assert_eq!(a["first_rejection"].is_null(),b["first_rejection"].is_null());
            if !a["first_rejection"].is_null() {
                // Audit adds ranks/scores; compare the underlying accepted prefix and actual target label.
                for k in ["offset","correct_target_token","proposed_token","canonical_labels_after_this_rejection"] {
                    assert_eq!(a["first_rejection"][k],b["first_rejection"][k]);
                }
            }
        }
    }
    let serial=serial_ref.unwrap();let input=flatten(ops);
    let result=json!({"id":name,"family":case["family"],"thinking":case["thinking"],"input_tokens":input.len(),
        "input_token_ids":input,"input_u32le_sha256":hash_tokens(&input),"output_tokens":audit.output,
        "pending_token":audit.pending,"stop":audit.stop,"final_consumed_cursor":audit.cursor,
        "same_target_tokens_cursor_logits_exact":false,"B8_same_policy_repeat_tokens_cursor_logits_exact":true,
        "B1_tokens_equal":audit.output==serial.output,"B1_final_logits_equal":audit.logits==serial.logits,
        "B1_first_token_difference":audit.output.iter().zip(&serial.output).position(|(a,b)|a!=b),
        "no_dflash_output_tokens":serial.output,"no_dflash_stop":serial.stop,"no_dflash_pending_token":serial.pending,
        "same_owner_continuation_checked":initialized,"target_execution_profile":super::super::batched::PROFILE,"target_projection_policy":b.batched.as_ref().unwrap().projection_policy,
        "exact_replay_operations":ops.iter().map(|op|match op{Replay::Prompt(v)=>json!({"kind":"prompt_B4352_literal_tail","tokens":v}),
            Replay::Selected(id)=>json!({"kind":"selected_B1","token":id}),Replay::Batch(v,n)=>json!({"kind":"verified_B8_prefix","path":v,"committed":n})}).collect::<Vec<_>>(),
        "timed":timed,"audit":audit.wire,"HTTP_or_Session_commit_authority":false});
    (audit.output,audit.pending,audit.stop,result)
}
#[test]
#[ignore = "exclusive full256Ki current Augur B8 migration acceptance/wall"]
fn real_batched8_acceptance_and_separate_wall() {
    let current_target=true;
    use crate::checkpoint_target_cuda::transfer_tests::{ids, open};
    let raw = std::fs::read(std::env::var_os("AUGUR_ACCEPTANCE_INPUT").unwrap()).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&raw)),
        std::env::var("AUGUR_ACCEPTANCE_INPUT_SHA256").unwrap()
    );
    let probes: Value = serde_json::from_slice(&raw).unwrap();
    assert_eq!(probes["schema"], if current_target {
        "quasar-batched8-thinking-first-probes-v1"
    } else { "quasar-english-thinking-first-probes-v1" });
    let fixture: Value = serde_json::from_slice(
        &std::fs::read(std::env::var_os("AUGUR_CHECKPOINT_TARGET_REFERENCE").unwrap()).unwrap(),
    )
    .unwrap();
    assert_eq!(
        probes["tokenizer_sha256"],
        fixture["metadata"]["tokenizer.json"]["sha256"]
    );
    assert_eq!(
        probes["template_sha256"],
        fixture["metadata"]["chat_template.jinja"]["sha256"]
    );
    let cfg = std::fs::read(std::env::var_os("AUGUR_DRAFT_CONFIG").unwrap()).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&cfg)),
        std::env::var("AUGUR_DRAFT_CONFIG_SHA256").unwrap()
    );
    let q8 = DFlashQ8Backbone::load_pinned(
        std::env::var_os("AUGUR_DRAFT_PATH").unwrap(),
        DFlashConfig::parse_f16_opt_in(&cfg).unwrap(),
    )
    .unwrap();
    assert_eq!(
        q8.manifest().file_sha256.as_ref().unwrap(),
        &std::env::var("AUGUR_DRAFT_SHA256").unwrap()
    );
    let before = cuda_device_allocation_snapshot();
    let load_start = Instant::now();
    let mut target = open(&fixture);
    assert_eq!(target.capacity, 262144);
    if current_target {
        #[cfg(feature = "qwen35-augur-http")]
        target.enable_augur_decode().unwrap();
        target.enable_augur_feature_capture().unwrap();
        #[cfg(not(feature = "qwen35-augur-http"))]
        panic!("current-target test requires rotated decoder feature");
    } else {
        target.enable_dflash_feature_capture().unwrap();
    }
    let target_attention_mode = target.info()["attention_mode"].clone();
    if current_target {
        assert!(["tile8_b1", "gqa8_planar", "gqa8_c4_planar"].contains(&target_attention_mode.as_str().unwrap()));
        assert_eq!(probes["target_attention_mode"], target_attention_mode, "requested vs actual target attention");
    }
    let weights = DFlashQ8CudaWeights::upload(&q8, 0).unwrap();
    let mut bridge = AugurDflashBridge::new(&target, &weights).unwrap();
    let roots = probes["cases"].as_array().unwrap();
    // Explicit short component screen is not the full acceptance collection.
    if current_target && probes["probe_set"] == "projection_screen" {
        assert_eq!(roots.iter().map(|r|r["id"].as_str().unwrap()).collect::<Vec<_>>(),
            vec!["reason_probability","code_python","tool_add"]);
        for case in &roots[..2] { assert_eq!(case["max_new_tokens"],128); }
    } else if probes["probe_set"] == "english_selector_adaptation" {
        assert!(current_target);adaptation::validate(&probes).unwrap();
    } else if probes["probe_set"] == "english_gate_screen" {
        assert!(current_target);
        assert_eq!(roots.iter().map(|r|r["id"].as_str().unwrap()).collect::<Vec<_>>(),vec!["reason_transaction","reason_binary_search"]);
        for case in roots {assert_eq!(case["family"],"english_thinking");assert_eq!(case["thinking"],true);
            assert!(case["followup"].is_null());assert!((1..=4096).contains(&case["max_new_tokens"].as_u64().unwrap()));}
    } else if probes["probe_set"] == "english_selector_holdout" {
        assert!(current_target);
        assert_eq!(roots.len(), 6, "five unseen reasoning roots plus tool closure");
        assert_eq!(probes["holdout_manifest_sha256"].as_str().unwrap().len(), 64);
        let thinking: Vec<_> = roots.iter().filter(|r| r["family"] == "english_thinking").collect();
        assert_eq!(thinking.len(), 5);
        for case in thinking {
            assert_eq!(case["max_new_tokens"], 4096);
            assert_eq!(case["thinking"], true);
            assert_eq!(case["experiment_partition"], "holdout_not_for_calibration");
            assert!(case["followup"].is_null());
        }
    } else if probes["probe_set"] == "english_thinking_full" {
        // Focused collection: four English-thinking families plus tool closure.
        assert_eq!(roots.len(), 5);
    } else {
        assert!(probes["probe_set"].is_null() || probes["probe_set"] == "full");
        assert_eq!(roots.len(), 8);
    }
    let eos = ids(&fixture["eos_token_ids"]);
    consume_prompt(&mut target, &ids(&roots[0]["token_ids"]));
    if current_target {
        // Exercise the real product's preallocated-but-previously-inactive path.
        bridge.retain_prefill_benchmark_workspace(&mut target).unwrap();
        assert!(target.consume_selected().is_err(), "legacy decoder must keep rejecting rotated KV");
    }
    bridge.prepare_batched_verifier(&target).unwrap();
    let cold_before = cuda_device_allocation_snapshot();
    let cold = bridge.propose_resident(&mut target).unwrap();
    if current_target {
        let after = cuda_device_allocation_snapshot();
        assert_eq!(after.successful_allocations, cold_before.successful_allocations);
        assert_eq!(after.successful_frees, cold_before.successful_frees);
        assert_eq!(bridge.generation, 2, "unused startup seed is not a generation proposal");
    }
    let rollback_probe=cold.path().to_vec();
    bridge.discard_resident(cold).unwrap();
    bridge.audit_batched_zero_rollback(&mut target,&rollback_probe).unwrap();
    save("zero-rollback.json",&json!({"pass":true,"GDN_state_history_exact":true,"physical_KV_8row_tail_exact":true,"capture_identity_unchanged":true,"timing_evidence":false}));
    let cold_load_and_workspace_s = load_start.elapsed().as_secs_f64();
    let mut results = Vec::new();
    let stdin = std::io::stdin();
    let mut control = stdin.lock();
    for root_case in roots {
        let mut case = root_case.clone();
        let mut ops = vec![Replay::Prompt(ids(&case["token_ids"]))];
        let mut initialized = false;
        loop {
            let (_output, pending, stop, result) =
                if probes["probe_set"]=="english_selector_adaptation" {
                    assert!(!initialized);adaptation::run_case(&mut target,&mut bridge,&case,&ops,&eos,probes["adaptation_mode"].as_str().unwrap())
                } else {accept_case(&mut target, &mut bridge, &case, &ops, &eos, initialized)};
            let name = case["id"].as_str().unwrap().to_owned();
            let digest = save(&format!("{name}.json"), &result);
            results.push(json!({"id":name,"sha256":digest,"family":case["family"],"thinking":case["thinking"]}));
            println!(
                "AUGUR_ACCEPTANCE_CASE={}",
                json!({"id":name,"sha256":digest})
            );
            std::io::stdout().flush().unwrap();
            let mut line = String::new();
            assert!(
                control.read_line(&mut line).unwrap() > 0,
                "controller disappeared"
            );
            assert!(line.len() < 1024 * 1024);
            let command: Value = serde_json::from_str(&line).unwrap();
            if command["append"] == false {
                break;
            }
            assert_eq!(command["append"], true);
            assert_eq!(stop, "eos");
            assert_eq!(pending, 248046);
            let next = command["case"].clone();
            assert_eq!(next["parent_case"], name);
            let suffix = ids(&next["suffix_token_ids"]);
            assert!(!suffix.is_empty() && suffix.len() < 4096);
            // Consume the real pending EOS and only the newly rendered suffix.
            assert_eq!(target.greedy_next_token().unwrap(), pending);
            consume_selected(&mut target);
            consume_prompt(&mut target, &suffix);
            for row in result["audit"]["rounds"].as_array().unwrap() {
                ops.push(Replay::Batch(ids(&row["proposal_path"]), (row["end_consumed"].as_u64().unwrap()-row["start_consumed"].as_u64().unwrap()) as usize));
            }
            ops.push(Replay::Selected(pending));
            ops.push(Replay::Prompt(suffix));
            assert_eq!(target.consumed, flatten(&ops).len());
            case = next;
            initialized = true;
        }
    }
    let target_width = bridge.batched.as_ref().unwrap().width;
    let mut info = bridge.resident_workspace_info();
    info["target_projection_policy"]=json!(bridge.batched.as_ref().unwrap().projection_policy);
    let peak = cuda_device_allocation_snapshot();
    drop(bridge);
    drop(weights);
    drop(target);
    let after = cuda_device_allocation_snapshot();
    assert_eq!(after.live_bytes, before.live_bytes);
    assert_eq!(after.live_allocations, before.live_allocations);
    save(
        "native-complete.json",
        &json!({"status":if probes["probe_set"]=="english_selector_adaptation" {"PASS_AUGUR_SELECTOR_ADAPTATION"} else if current_target {"PASS_AUGUR_BATCHED8_NATIVE_SCREEN"} else {"PASS_ENGLISH_THINKING_FIRST_NATIVE_ACCEPTANCE"},"results":results,
        "current_product_target":current_target,"target_attention_mode":target_attention_mode,"prefill_max_batch":if current_target {4352} else {8},
        "cold_load_and_workspace_s":cold_load_and_workspace_s,"workspace":info,"target_capacity":262144,
        "owned_peak_bytes":peak.peak_live_bytes,"steady_owned_bytes":peak.live_bytes-before.live_bytes,
        "vision_loaded":false,"native_HTTP_or_Session_commits":false,"target_width":target_width,
        "target_verifier":format!("B{target_width}_recorded_projection_prefix"),"numerics_changed_from_B1":true,
        "cleanup_live_bytes_delta":after.live_bytes-before.live_bytes,"cleanup_live_allocations_delta":after.live_allocations-before.live_allocations}),
    );
}



#[cfg(windows)]
mod cache_thinking;

mod adaptation;

mod complete_teacher;
