//! Exclusive native target test of the production input cache manager. Draft
//! bindings are explicit test-only markers; no draft weights or HTTP are measured.
use super::*;
use augur_backend::cuda_device_allocation_snapshot;
use augur_model_qwen35::Qwen35CheckpointProfile;
use crate::checkpoint_full_cuda::CheckpointKvPolicy;
use crate::checkpoint_projection_cuda::CheckpointProjectionPolicy;
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {id:String,token_ids:Vec<u32>}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {schema:String,cases:Vec<Case>}
#[derive(serde::Serialize,Deserialize)]
struct Fingerprint {sha:String,logits:Vec<f32>,features:Vec<u8>,tokens:Vec<u32>}
fn descriptor(bindings:&Bindings,ids:&[u32])->Descriptor {
    Descriptor{schema:contract::SCHEMA.into(),bindings:bindings.clone(),boundary:Boundary::InputEnd{token_ids:ids.to_vec()}}
}
fn audit(t:&mut CheckpointCudaTarget,b:&Bindings,ids:&[u32])->Fingerprint {
    assert_eq!(t.consumed_tokens(),ids.len());
    let sha=t.checkpoint_test_state_sha256(&descriptor(b,ids)).unwrap();
    let logits=t.logits_snapshot().unwrap();let features=t.dflash_feature_snapshot().unwrap().little_endian_f32;
    let mut tokens=vec![t.greedy_next_token().unwrap()];
    for _ in 0..16 {t.consume_selected_augur().unwrap();tokens.push(t.greedy_next_token().unwrap());}
    t.drain_session_work().unwrap();Fingerprint{sha,logits,features,tokens}
}
fn open_target(identity:augur_model_qwen35::CheckpointExpectedIdentity)->CheckpointCudaTarget {
    let mut t=CheckpointCudaTarget::open_with_execution_policy(std::env::var_os("AUGUR_MODEL_ROOT").unwrap(),
        Qwen35CheckpointProfile::QuasarNvfp4,identity,262144,
        CheckpointProjectionPolicy::AugurQpnAllWorkspaceShapeSmallPairF64,CheckpointKvPolicy::Q8_0).unwrap();
    t.enable_exact_warp_tree_norm().unwrap();t.enable_fused_input_qpn_norm().unwrap();t.enable_batched_ffn8().unwrap();
    t.enable_augur_decode().unwrap();t.enable_augur_feature_capture().unwrap();t
}
fn manager(root:&Path,bindings:&Bindings,budget:StoreBudget)->Manager {
    std::fs::create_dir(root).unwrap();
    let store=HttpCheckpointStore::open(root,budget).unwrap();
    Manager{bindings:bindings.clone(),store:None,restore_timing:false,restore_workers:1,restore_schedule:ParallelRestoreSchedule::Batch,prefix:Some(Prefix::open(store,bindings,Some(30464),true,false).unwrap())}
}
fn write(root:&Path,name:&str,v:&Value) {
    std::fs::write(root.join(name),serde_json::to_vec_pretty(v).unwrap()).unwrap();
}
fn check(t:&mut CheckpointCudaTarget,m:&mut Manager,b:&Bindings,c:&Case,reference:&Fingerprint,label:&str,hit:bool,matched:usize,root:&Path)->Value {
    t.reset_empty().unwrap();let cancel=SessionCancellationToken::new();let tick=Instant::now();
    let detail=m.prefill(t,&c.token_ids,&cancel).unwrap().unwrap();t.drain_session_work().unwrap();let prefill_s=tick.elapsed().as_secs_f64();
    let first=t.greedy_next_token().unwrap();let native_ttft_s=tick.elapsed().as_secs_f64();assert_eq!(first,reference.tokens[0]);
    assert_eq!(detail["input_prefix_cache"]["hit"],hit,"{label}");
    assert_eq!(detail["input_prefix_cache"]["matched_tokens"],matched,"{label}");
    let got=audit(t,b,&c.token_ids);
    assert_eq!(got.sha,reference.sha,"{label}: complete KV/GDN/conv/feature/logits state");
    assert_eq!(got.logits,reference.logits,"{label}: input logits");
    assert_eq!(got.features,reference.features,"{label}: chronological features");
    assert_eq!(got.tokens,reference.tokens,"{label}: continuation");
    let result=json!({"id":label,"input_tokens":c.token_ids.len(),"manager_prefill_wall_s":prefill_s,"native_prefill_to_first_token_s":native_ttft_s,
        "prefill_detail":detail,"state_sha256":got.sha,"state_exact":true,"logits_exact":true,
        "features_exact":true,"continuation_tokens":got.tokens,"continuation_exact":true,"http":false,"draft_loaded":false});
    write(root,&format!("{label}.json"),&result);eprintln!("PASS_CACHE_MANAGER {label} hit={hit} matched={matched} wall={prefill_s:.6}");result
}
#[test]
#[ignore="exclusive SM70: production cache manager 32K exact input, branch fallback, multiple documents, quota, state/continuation"]
fn cache_manager_full_state_and_wall() {
    let root=PathBuf::from(std::env::var_os("AUGUR_CACHE_MANAGER_OUT").unwrap());std::fs::create_dir(&root).unwrap();
    let manifest_raw=std::fs::read(std::env::var_os("AUGUR_MANIFEST").unwrap()).unwrap();
    assert_eq!(session::digest(&manifest_raw),std::env::var("AUGUR_MANIFEST_SHA256").unwrap());
    let manifest:Value=serde_json::from_slice(&manifest_raw).unwrap();
    let identity:augur_model_qwen35::CheckpointExpectedIdentity=serde_json::from_value(manifest["checkpoint"].clone()).unwrap();
    let raw=std::fs::read(std::env::var_os("AUGUR_CACHE_MANAGER_INPUT").unwrap()).unwrap();
    assert_eq!(session::digest(&raw),std::env::var("AUGUR_CACHE_MANAGER_INPUT_SHA256").unwrap());
    let input:Input=serde_json::from_slice(&raw).unwrap();assert_eq!(input.schema,"quasar-cache-manager-input-v1");
    assert_eq!(input.cases.len(),4);let cases=&input.cases;
    assert_eq!(cases[0].token_ids.len(),32760);assert_eq!(&cases[0].token_ids[..30464],&cases[1].token_ids[..30464]);
    assert_ne!(cases[0].token_ids,cases[1].token_ids);assert_ne!(&cases[0].token_ids[..30464],&cases[2].token_ids[..30464]);
    assert!(cases[3].token_ids.len()<4352);
    let bindings=Bindings{producer_sha256:sha_file(&std::env::current_exe().unwrap()).unwrap(),manifest_sha256:session::digest(&manifest_raw),
        profile:session::PROFILE.into(),draft_config_sha256:session::digest(b"TEST-ONLY-NO-DRAFT-CONFIG"),
        draft_weights_sha256:session::digest(b"TEST-ONLY-NO-DRAFT-WEIGHTS"),decode_attention:"gqa8_planar".into(),sampling:crate::augur_sampling::GREEDY.into()};
    bindings.validate().unwrap();let before=cuda_device_allocation_snapshot();let started=Instant::now();
    let mut t=open_target(identity.clone());let load_s=started.elapsed().as_secs_f64();let cancel=SessionCancellationToken::new();
    let mut references=Vec::new();let mut reference_bill=Vec::new();
    for c in cases {
        t.reset_empty().unwrap();let tick=Instant::now();
        session::SessionTarget::prefill(&mut t,&c.token_ids,&cancel).unwrap();t.drain_session_work().unwrap();let prefill_s=tick.elapsed().as_secs_f64();
        let fingerprint=audit(&mut t,&bindings,&c.token_ids);
        let value=json!({"id":c.id,"input_tokens":c.token_ids.len(),"uncached_prefill_wall_s":prefill_s,
            "state_sha256":fingerprint.sha,"continuation_tokens":fingerprint.tokens,"timing_excludes_state_audit":true});
        write(&root,&format!("reference-{}.json",c.id),&value);eprintln!("PASS_CACHE_REFERENCE {} wall={prefill_s:.6}",c.id);
        reference_bill.push(value);references.push(fingerprint);
    }
    let mut m=manager(&root.join("multi-entry"),&bindings,StoreBudget{max_checkpoints:PREFIX_QUOTA,max_reserved_bytes:PREFIX_BYTES});
    let mut rows=Vec::new();
    for (i,label,hit,matched) in [
        (0,"cold-a",false,0),(0,"exact-a-1",true,32760),(1,"branch-a",true,30464),
        (2,"cold-b",false,0),(0,"alternate-a",true,32760),(2,"alternate-b",true,cases[2].token_ids.len()),
        (3,"short-cold",false,0),(3,"short-exact",true,cases[3].token_ids.len()),(0,"exact-a-2",true,32760),
    ] {rows.push(check(&mut t,&mut m,&bindings,&cases[i],&references[i],label,hit,matched,&root));}
    // Same owner append must keep the live state; manager must decline without mutation.
    let consumed=t.consumed_tokens();assert!(m.prefill(&mut t,&[7,8],&cancel).unwrap().is_none());assert_eq!(consumed,t.consumed_tokens());
    t.reset_empty().unwrap();let cancelled=SessionCancellationToken::new();cancelled.request_cancel();
    assert!(m.prefill(&mut t,&cases[0].token_ids,&cancelled).is_err());assert_eq!(t.consumed_tokens(),0);
    let mut q=manager(&root.join("quota"),&bindings,StoreBudget{max_checkpoints:1,max_reserved_bytes:PREFIX_BYTES});
    rows.push(check(&mut t,&mut q,&bindings,&cases[3],&references[3],"quota-first",false,0,&root));
    // Reuse another short real prompt to prove skipped captures do not fail inference.
    let prefix=&Case{id:"short-distinct".into(),token_ids:cases[0].token_ids[..73].to_vec()};
    t.reset_empty().unwrap();session::SessionTarget::prefill(&mut t,&prefix.token_ids,&cancel).unwrap();
    let distinct=audit(&mut t,&bindings,&prefix.token_ids);
    let skipped=check(&mut t,&mut q,&bindings,prefix,&distinct,"quota-full",false,0,&root);
    assert_eq!(skipped["prefill_detail"]["input_prefix_cache"]["captures"][0]["reason"],"cache_quota");rows.push(skipped);
    // Release the target and restore the cached prompt into a separately loaded owner.
    drop(t);let restarted=Instant::now();let mut t=open_target(identity);let new_owner_load_s=restarted.elapsed().as_secs_f64();
    rows.push(check(&mut t,&mut m,&bindings,&cases[0],&references[0],"new-owner-exact",true,32760,&root));
    let target=t.info().clone();drop(t);drop(q);drop(m);let after=cuda_device_allocation_snapshot();
    assert_eq!(before.live_bytes,after.live_bytes);assert_eq!(before.live_allocations,after.live_allocations);
    let complete=json!({"status":"PASS_CACHE_MANAGER_NATIVE", "bindings":bindings,"capacity":262144,"load_s":load_s,"new_owner_load_s":new_owner_load_s,
        "reference":reference_bill,"results":rows,"same_owner_append_bypassed":true,"cancel_before_lookup_unmutated":true,
        "target":target,"allocation_closure":true,"live_bytes_after":after.live_bytes,"live_allocations_after":after.live_allocations,
        "draft_loaded":false,"draft_bindings":"TEST-ONLY structural markers; no production asset identity or acceptance claim",
        "http_measured":false,"persistent_index_tested":false,"cross_build_reuse":false,"actual_request_hit_rate_measured":false});
    write(&root,"complete.json",&complete);eprintln!("PASS_CACHE_MANAGER_NATIVE all state, continuation, quota, fresh-owner and allocation checks");
}

fn marker_bindings()->Bindings {
    Bindings{producer_sha256:session::digest(b"test-build"),manifest_sha256:session::digest(b"test-model"),
        profile:session::PROFILE.into(),draft_config_sha256:session::digest(b"test-draft-config"),
        draft_weights_sha256:session::digest(b"test-draft-weights"),decode_attention:"gqa8_planar".into(),sampling:crate::augur_sampling::GREEDY.into()}
}
#[test]
fn persistent_namespace_binds_every_live_identity_field() {
    let b=marker_bindings();let expected=prefix_namespace(&b).unwrap();
    for i in 0..6 {
        let mut changed=b.clone();
        match i {0=>changed.producer_sha256=session::digest(b"next-build"),1=>changed.manifest_sha256=session::digest(b"next-model"),
            2=>changed.profile=session::DFLASH_B8_PROFILE.into(),3=>changed.draft_config_sha256=session::digest(b"next-config"),
            4=>changed.draft_weights_sha256=session::digest(b"next-draft"),_=>changed.decode_attention="tile8_b1".into()}
        assert_ne!(expected,prefix_namespace(&changed).unwrap(),"identity field {i}");
    }
}
#[test]
fn persistent_index_rejects_changed_bindings_and_non_descriptor_ids() {
    use std::sync::atomic::{AtomicU64,Ordering};static NEXT:AtomicU64=AtomicU64::new(1);
    let root=std::env::temp_dir().join(format!("augur-prefix-index-{}-{}",std::process::id(),NEXT.fetch_add(1,Ordering::SeqCst)));
    std::fs::create_dir(&root).unwrap();let bindings=marker_bindings();
    let good=descriptor(&bindings,&[1,2,3]);let mut wrong=good.clone();wrong.bindings.producer_sha256=session::digest(b"wrong producer");
    let mut store=HttpCheckpointStore::open(&root,StoreBudget{max_checkpoints:4,max_reserved_bytes:PREFIX_BYTES}).unwrap();
    for (d,id) in [(&good,session::digest(&good.encode().unwrap())),(&wrong,session::digest(&wrong.encode().unwrap())),(&good,session::digest(b"wrong ID"))] {
        // Only the sealed envelope is materialized. This checks bootstrap
        // admission, never treats synthetic metadata as a restorable state.
        let slot=store.reserve(&id,16).unwrap();std::fs::create_dir(slot.native_path()).unwrap();
        let e=Envelope{schema:"quasar-checkpoint-envelope-v1".into(),checkpoint_id:id.clone(),native_manifest_sha256:session::digest(b"synthetic payload"),descriptor:d.clone()};
        let path=root.join(&id);std::fs::write(path.join("http.blob"),e.encode().unwrap()).unwrap();
        std::fs::hard_link(path.join("http.blob"),path.join("http.json")).unwrap();drop(slot);
    }
    drop(store);let store=HttpCheckpointStore::open(&root,StoreBudget{max_checkpoints:4,max_reserved_bytes:PREFIX_BYTES}).unwrap();
    let p=Prefix::open(store,&bindings,None,true,true).unwrap();
    assert_eq!(p.recovered_entries,1);assert_eq!(p.saved[0].descriptor,good);assert_eq!(p.unavailable.len(),2);
    assert_eq!(contract::longest_prefix(p.saved.iter().map(|s|&s.descriptor),&bindings,&[1,2,3],1).unwrap(),Some(0));
}

#[test]
#[ignore="exclusive SM70: two-process persistent seed/reopen, full state and late-payload corruption recovery"]
fn cache_manager_persistent_restart() {
    let root=PathBuf::from(std::env::var_os("AUGUR_CACHE_MANAGER_OUT").unwrap());std::fs::create_dir(&root).unwrap();
    let phase=std::env::var("AUGUR_CACHE_RESTART_PHASE").unwrap();assert!(["seed","reopen"].contains(&phase.as_str()));
    let common=root.parent().unwrap();
    let manifest_raw=std::fs::read(std::env::var_os("AUGUR_MANIFEST").unwrap()).unwrap();
    assert_eq!(session::digest(&manifest_raw),std::env::var("AUGUR_MANIFEST_SHA256").unwrap());
    let manifest:Value=serde_json::from_slice(&manifest_raw).unwrap();
    let identity:augur_model_qwen35::CheckpointExpectedIdentity=serde_json::from_value(manifest["checkpoint"].clone()).unwrap();
    let raw=std::fs::read(std::env::var_os("AUGUR_CACHE_MANAGER_INPUT").unwrap()).unwrap();
    assert_eq!(session::digest(&raw),std::env::var("AUGUR_CACHE_MANAGER_INPUT_SHA256").unwrap());
    let input:Input=serde_json::from_slice(&raw).unwrap();assert_eq!(input.schema,"quasar-cache-manager-input-v1");
    let long=&input.cases[0];let short=&input.cases[3];assert_eq!(long.token_ids.len(),32760);assert!(short.token_ids.len()<4352);
    let before=cuda_device_allocation_snapshot();
    // Use the actual startup configuration and persistent namespace; metadata
    // scan completes before the CUDA target exists.
    let tick=Instant::now();let mut m=Manager::configure(&root,&session::digest(&manifest_raw),"gqa8_planar").unwrap().unwrap();
    let configure_s=tick.elapsed().as_secs_f64();let ready=m.ready();write(&root,"ready.json",&ready);let b=m.bindings.clone();
    let tick=Instant::now();let mut t=open_target(identity);let load_s=tick.elapsed().as_secs_f64();let cancel=SessionCancellationToken::new();
    let mut rows=Vec::new();let mut corruption=None;
    if phase=="seed" {
        assert_eq!(m.prefix.as_ref().unwrap().recovered_entries,0);
        let mut references=Vec::new();
        for c in [long,short] {
            t.reset_empty().unwrap();let tick=Instant::now();session::SessionTarget::prefill(&mut t,&c.token_ids,&cancel).unwrap();
            t.drain_session_work().unwrap();let prefill_s=tick.elapsed().as_secs_f64();let reference=audit(&mut t,&b,&c.token_ids);
            write(&root,&format!("reference-{}.json",c.id),&json!({"uncached_prefill_wall_s":prefill_s,"fingerprint":reference}));
            rows.push(check(&mut t,&mut m,&b,c,&reference,&format!("seed-{}",c.id),false,0,&root));references.push(reference);
        }
        let refs=json!({"schema":"quasar-persistent-cache-references-v1","bindings":b,"long":references[0],"short":references[1]});
        let bytes=serde_json::to_vec(&refs).unwrap();std::fs::write(common.join("references.json"),&bytes).unwrap();
        write(common,"references-identity.json",&json!({"sha256":session::digest(&bytes),"producer_sha256":b.producer_sha256}));
    }else{
        assert_eq!(m.prefix.as_ref().unwrap().recovered_entries,3);assert!(m.prefix.as_ref().unwrap().unavailable.is_empty());
        let reference_bytes=std::fs::read(common.join("references.json")).unwrap();
        let identity:Value=serde_json::from_slice(&std::fs::read(common.join("references-identity.json")).unwrap()).unwrap();
        assert_eq!(session::digest(&reference_bytes),identity["sha256"].as_str().unwrap());
        let references:Value=serde_json::from_slice(&reference_bytes).unwrap();assert_eq!(serde_json::to_value(&b).unwrap(),references["bindings"]);
        let long_ref:Fingerprint=serde_json::from_value(references["long"].clone()).unwrap();
        let short_ref:Fingerprint=serde_json::from_value(references["short"].clone()).unwrap();
        rows.push(check(&mut t,&mut m,&b,long,&long_ref,"restart-long",true,32760,&root));
        rows.push(check(&mut t,&mut m,&b,short,&short_ref,"restart-short",true,short.token_ids.len(),&root));
        // Corrupt the final logits chunk so earlier native components have
        // already been copied into the owner when verification rejects it.
        let id=m.prefix.as_ref().unwrap().saved.iter().find(|s|s.descriptor.tokens()==long.token_ids).unwrap().id.clone();
        let native=PathBuf::from(std::env::var_os("AUGUR_INPUT_PREFIX_CACHE_ROOT").unwrap()).join(prefix_namespace(&b).unwrap()).join(&id).join("native");
        let manifest:Value=serde_json::from_slice(&std::fs::read(native.join("manifest.json")).unwrap()).unwrap();
        let chunk=manifest["chunks"].as_array().unwrap().last().unwrap();
        let component=chunk["component"].as_u64().unwrap() as usize;
        assert_eq!(manifest["plan"]["components"][component]["name"],"target.logits");
        assert!(component>0,"late corruption must follow native state writes");
        let name=chunk["sha256"].as_str().unwrap();
        let path=native.join(format!("{name}.chunk"));let mut bytes=std::fs::read(&path).unwrap();
        let original_sha=session::digest(&bytes);bytes[0]^=1;std::fs::write(&path,&bytes).unwrap();
        let fallback=check(&mut t,&mut m,&b,long,&long_ref,"corrupt-long-fallback",true,30464,&root);
        assert_eq!(fallback["prefill_detail"]["input_prefix_cache"]["restore_failures"].as_array().unwrap().len(),1);
        assert_eq!(fallback["prefill_detail"]["input_prefix_cache"]["captures"][0]["reason"],"cache_entry_unavailable");rows.push(fallback);
        rows.push(check(&mut t,&mut m,&b,long,&long_ref,"quarantined-long-fallback",true,30464,&root));
        // Quarantine is an owner index decision; a later owner rediscovers the
        // seal lazily, detects the bad payload and also recovers the shorter state.
        drop(m);m=Manager::configure(&root,&session::digest(&manifest_raw),"gqa8_planar").unwrap().unwrap();
        rows.push(check(&mut t,&mut m,&b,long,&long_ref,"reopened-corrupt-fallback",true,30464,&root));
        corruption=Some(json!({"checkpoint_id":id,"chunk":name,"original_file_sha256":original_sha,"corrupt_file_sha256":session::digest(&bytes),
            "component":chunk["component"],"partial_native_write_exercised":true,"corrupt_slot_retained":true,"shorter_state_exact":true}));
        // Cancellation must escape and leave the already-empty owner unchanged.
        t.reset_empty().unwrap();let cancelled=SessionCancellationToken::new();cancelled.request_cancel();
        assert!(m.prefill(&mut t,&long.token_ids,&cancelled).is_err());assert_eq!(t.consumed_tokens(),0);
    }
    drop(t);drop(m);let after=cuda_device_allocation_snapshot();assert_eq!(before.live_bytes,after.live_bytes);assert_eq!(before.live_allocations,after.live_allocations);
    let complete=json!({"status":"PASS_CACHE_PERSISTENT_RESTART","phase":phase,"bindings":b,"configure_s":configure_s,"load_s":load_s,
        "ready":ready,"results":rows,"corruption":corruption,"allocation_closure":true,"live_bytes_after":after.live_bytes,"live_allocations_after":after.live_allocations,
        "draft_loaded":false,"http_measured":false,"cross_build_reuse":false});write(&root,"complete.json",&complete);
    eprintln!("PASS_CACHE_PERSISTENT_RESTART phase={phase}");
}

#[test]
#[ignore="exclusive SM70: late failed restore scrubs all physical KV before retry"]
fn late_restore_failure_resets_uncommitted_physical_kv() {
    let root=PathBuf::from(std::env::var_os("AUGUR_RESTORE_ZERO_OUT").unwrap());std::fs::create_dir(&root).unwrap();
    let manifest_raw=std::fs::read(std::env::var_os("AUGUR_MANIFEST").unwrap()).unwrap();
    assert_eq!(session::digest(&manifest_raw),std::env::var("AUGUR_MANIFEST_SHA256").unwrap());
    let manifest:Value=serde_json::from_slice(&manifest_raw).unwrap();
    let identity:augur_model_qwen35::CheckpointExpectedIdentity=serde_json::from_value(manifest["checkpoint"].clone()).unwrap();
    let raw=std::fs::read(std::env::var_os("AUGUR_RESTORE_ZERO_INPUT").unwrap()).unwrap();
    assert_eq!(session::digest(&raw),std::env::var("AUGUR_RESTORE_ZERO_INPUT_SHA256").unwrap());
    let input:Value=serde_json::from_slice(&raw).unwrap();
    let case=input["cases"].as_array().unwrap().iter().find(|c|c["id"]=="code_32k").unwrap();
    let ids:Vec<u32>=serde_json::from_value(case["token_ids"].clone()).unwrap();assert_eq!(ids.len(),32760);
    let b=Bindings{producer_sha256:sha_file(&std::env::current_exe().unwrap()).unwrap(),manifest_sha256:session::digest(&manifest_raw),
        profile:session::PROFILE.into(),draft_config_sha256:std::env::var("AUGUR_DRAFT_CONFIG_SHA256").unwrap(),
        draft_weights_sha256:std::env::var("AUGUR_DRAFT_SHA256").unwrap(),decode_attention:"gqa8_planar".into(),sampling:crate::augur_sampling::GREEDY.into()};b.validate().unwrap();
    let before=cuda_device_allocation_snapshot();let mut target=open_target(identity);
    session::SessionTarget::prefill(&mut target,&ids,&SessionCancellationToken::new()).unwrap();
    let d=descriptor(&b,&ids);let reference=target.checkpoint_test_state_sha256(&d).unwrap();
    let native=root.join("native");let saved=target.checkpoint_save(&d,&native).unwrap();
    let transfer:Value=serde_json::from_slice(&std::fs::read(native.join("manifest.json")).unwrap()).unwrap();
    let chunk=transfer["chunks"].as_array().unwrap().last().unwrap();let component=chunk["component"].as_u64().unwrap() as usize;
    assert_eq!(transfer["plan"]["components"][component]["name"],"target.logits");
    let path=native.join(format!("{}.chunk",chunk["sha256"].as_str().unwrap()));let good=std::fs::read(&path).unwrap();
    fn continuation(t:&mut CheckpointCudaTarget)->Vec<u32> {
        let mut ids=vec![t.greedy_next_token().unwrap()];
        for _ in 0..127 {t.consume_selected_augur().unwrap();ids.push(t.greedy_next_token().unwrap());}
        t.drain_session_work().unwrap();ids
    }
    let expected_output=continuation(&mut target);
    let mut bad=good.clone();bad[0]^=1;let mut rows=Vec::new();
    for fault in ["late_corruption","late_missing","wrong_manifest_sha"] {
        if fault=="late_corruption" {std::fs::write(&path,&bad).unwrap();}
        if fault=="late_missing" {std::fs::rename(&path,root.join("held-late.chunk")).unwrap();}
        for (workers,schedule) in [(1,ParallelRestoreSchedule::Batch),(8,ParallelRestoreSchedule::Batch),(8,ParallelRestoreSchedule::Pipeline)] {
            target.reset_empty().unwrap();
            let expected_sha=if fault=="wrong_manifest_sha" {"0".repeat(64)} else {saved.manifest_sha256.clone()};
            let failure=if workers==1 {target.checkpoint_restore(&d,&native,&expected_sha)}
                else {target.checkpoint_restore_parallel_scheduled(&d,&native,&expected_sha,workers,schedule)};
            let error=failure.unwrap_err().to_string();
            if fault=="late_corruption" {assert!(error.contains("chunk content mismatch"));}
            assert_eq!(target.consumed_tokens(),0);target.reset_empty().unwrap();
            let checked=target.checkpoint_test_zero_kv_rows(ids.len()).unwrap();assert_eq!(checked,1140850688);
            rows.push(json!({"fault":fault,"workers":workers,"schedule":schedule.label(),"error":error,
                "uncommitted_cursor":0,"zero_checked_kv_bytes":checked,"physical_padding_zero":true}));
        }
        if fault=="late_missing" {std::fs::rename(root.join("held-late.chunk"),&path).unwrap();}
        std::fs::write(&path,&good).unwrap();
    }
    let mut good_rows=Vec::new();
    for schedule in [ParallelRestoreSchedule::Batch,ParallelRestoreSchedule::Pipeline] {
        target.reset_empty().unwrap();
        let restored=target.checkpoint_restore_parallel_scheduled(&d,&native,&saved.manifest_sha256,8,schedule).unwrap();
        let restored_sha=target.checkpoint_test_state_sha256(&d).unwrap();assert_eq!(restored_sha,reference);
        let output=continuation(&mut target);assert_eq!(output,expected_output);
        good_rows.push(json!({"schedule":schedule.label(),"full_state_sha256":restored_sha,
            "output_tokens":output.len(),"output_sha256":session::digest(&output.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>()),
            "staging_bytes":restored.staging_bytes,"payload_bytes":restored.payload_bytes,"state_exact":true,"continuation128_exact":true}));
    }
    drop(target);let after=cuda_device_allocation_snapshot();assert_eq!(before.live_bytes,after.live_bytes);assert_eq!(before.live_allocations,after.live_allocations);
    write(&root,"complete.json",&json!({"status":"PASS_LATE_RESTORE_ZERO_AND_RECOVERY","results":rows,"good_restores":good_rows,"recovered_full_state_exact":true,"continuation128_exact":true,
        "bindings":b,"corrupt_copy_repaired":true,"allocation_closure":true,"live_bytes_after":after.live_bytes,"draft_loaded":false,"http_measured":false}));
}
