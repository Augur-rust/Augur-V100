//! Actual Augur owner checks with authenticated model, no draft weights.
use super::*;
use crate::augur_session::{self as session, checkpoint::{Bindings,Boundary,Descriptor}};
use std::{fs, path::PathBuf};
fn fingerprint(t:&mut CheckpointCudaTarget)->(Vec<f32>,Vec<u8>) {
    (t.logits_snapshot().unwrap(),t.dflash_feature_snapshot().unwrap().little_endian_f32)
}
fn prefill(t:&mut CheckpointCudaTarget,ids:&[u32])->Value {
    let plan=t.plan_augur_sequence(ids,ids.chunks(4352).map(|c|c.len()).collect()).unwrap();
    serde_json::to_value(t.consume_sequence_augur(ids,&plan,
        crate::checkpoint_target_cuda::augur_prefill::Budget::default(),|_|false).unwrap()).unwrap()
}
#[test]
#[ignore="exclusive Augur SM70: planar owner prefill/B1/B8 rollback/park/native checkpoint"]
fn planar_owner_prefill_decode_b8_and_checkpoint() {
    let root=PathBuf::from(std::env::var_os("AUGUR_PLANAR_OWNER_OUT").unwrap());
    fs::create_dir(&root).unwrap();
    let manifest_raw=fs::read(std::env::var_os("AUGUR_MANIFEST").unwrap()).unwrap();
    assert_eq!(session::digest(&manifest_raw),std::env::var("AUGUR_MANIFEST_SHA256").unwrap());
    let manifest:Value=serde_json::from_slice(&manifest_raw).unwrap();
    let identity=serde_json::from_value(manifest["checkpoint"].clone()).unwrap();
    let raw=fs::read(std::env::var_os("AUGUR_PLANAR_OWNER_INPUT").unwrap()).unwrap();
    let input:Value=serde_json::from_slice(&raw).unwrap();
    let ids:Vec<u32>=serde_json::from_value(input["token_ids"].clone()).unwrap();
    assert_eq!(ids.len(),4359);
    let before=cuda_device_allocation_snapshot();
    let mut t=CheckpointCudaTarget::open_with_execution_policy(std::env::var_os("AUGUR_MODEL_ROOT").unwrap(),
        augur_model_qwen35::Qwen35CheckpointProfile::QuasarNvfp4,identity,262144,
        crate::checkpoint_projection_cuda::CheckpointProjectionPolicy::AugurQpnAllWorkspaceShapeSmallPairF64,
        crate::checkpoint_full_cuda::CheckpointKvPolicy::Q8_0).unwrap();
    t.enable_exact_warp_tree_norm().unwrap();t.enable_fused_input_qpn_norm().unwrap();t.enable_batched_ffn8().unwrap();
    t.enable_augur_decode().unwrap();
    assert_eq!(t.attention_mode(),full::AttentionMode::Gqa8Planar);
    t.enable_augur_feature_capture().unwrap();
    assert!(t.enable_augur_decode().is_err());
    let receipt=prefill(&mut t,&ids);
    assert_eq!(receipt["plan"]["widths"],json!([4352,7]));
    assert_eq!(receipt["plan"]["kv_identity"],t.attention_mode().kv_identity());
    let mut wrong=receipt["plan"].clone();
    wrong["kv_identity"]=augur_backend::cuda_qwen35_rotated_kv::QWEN35_ROTATED_KV_IDENTITY.into();
    let wrong=serde_json::from_value(wrong).unwrap();
    assert!(t.consume_sequence_augur(&ids,&wrong,crate::checkpoint_target_cuda::augur_prefill::Budget::default(),|_|false).is_err());
    let selected=t.greedy_next_token().unwrap();t.consume_selected_augur().unwrap();
    assert_eq!(t.consumed,4360);t.logits_snapshot().unwrap();
    let path=&ids[..B];
    let mut executor=Executor::new_with_projection(&t,"packed_b8").unwrap();
    let initial=fingerprint(&mut t);
    let first=executor.execute(&mut t,path).unwrap();executor.commit(&mut t,0).unwrap();
    for p in &executor.programs {if let Program::Full(_,c)=p {c.audit_zero_restored(t.consumed).unwrap();}}
    assert_eq!(initial,fingerprint(&mut t));
    assert_eq!(first,executor.execute(&mut t,path).unwrap());executor.commit(&mut t,0).unwrap();
    let mut rounds=Vec::new();
    for accepted in [1,7,8] {
        let start=t.consumed;
        let logits=executor.execute(&mut t,path).unwrap();executor.commit(&mut t,accepted).unwrap();
        assert_eq!(t.consumed,start+accepted);t.validate_cursor().unwrap();t.logits_snapshot().unwrap();
        rounds.push(json!({"accepted":accepted,"start":start,"end":t.consumed,"target_ids":logits}));
    }
    executor.park().unwrap();t.reset_empty().unwrap();
    assert_eq!(t.attention_mode(),full::AttentionMode::Gqa8Planar);
    prefill(&mut t,&ids);t.greedy_next_token().unwrap();t.consume_selected_augur().unwrap();
    executor.resume(&t).unwrap();
    assert_eq!(first,executor.execute(&mut t,path).unwrap());executor.commit(&mut t,0).unwrap();
    drop(executor);
    // Exact text native boundary, same owner mode; no synthetic draft admission.
    let pending=t.greedy_next_token().unwrap();
    let mut history=ids.clone();history.push(selected);
    let d=Descriptor{schema:session::checkpoint::SCHEMA.into(),
        bindings:Bindings{producer_sha256:session::digest(&fs::read(std::env::current_exe().unwrap()).unwrap()),
            manifest_sha256:session::digest(&manifest_raw),profile:session::PROFILE.into(),
            draft_config_sha256:"a".repeat(64),draft_weights_sha256:"b".repeat(64),decode_attention:"gqa8_planar".into(),sampling:crate::augur_sampling::GREEDY.into()},
        boundary:Boundary::Session{snapshot:session::SessionSnapshot{profile:session::PROFILE.into(),
            manifest_sha256:session::digest(&manifest_raw),session_id:"planar-owner-structural".into(),revision:1,
            consumed_input_sha256:session::digest(&history.iter().flat_map(|id| id.to_le_bytes()).collect::<Vec<_>>()),consumed_token_ids:history,
            pending_token:pending,pending_token_published:true,image_input_sha256:None},last_receipt_sha256:"c".repeat(64)}};
    let prior=fingerprint(&mut t);
    let native=t.checkpoint_save(&d,&root.join("native")).unwrap();
    t.reset_empty().unwrap();
    let mut wrong=d.clone();wrong.bindings.decode_attention="tile8_b1".into();
    assert!(t.checkpoint_plan(&wrong).is_err());assert_eq!(t.consumed,0);
    t.checkpoint_restore(&d,&root.join("native"),&native.manifest_sha256).unwrap();
    assert_eq!(prior,fingerprint(&mut t));
    assert_eq!(t.info["active_kv_format"],t.attention_mode().kv_identity());
    assert_eq!(pending,t.greedy_next_token().unwrap());t.consume_selected_augur().unwrap();t.logits_snapshot().unwrap();
    let info=t.info().clone();drop(t);
    let after=cuda_device_allocation_snapshot();
    assert_eq!(before.live_bytes,after.live_bytes);assert_eq!(before.live_allocations,after.live_allocations);
    let result=json!({"status":"PASS_PLANAR_OWNER", "capacity":262144,"prefill":receipt,
        "b8_prefix_commits":rounds,"zero_rollback_exact":true,"park_resume_replay_exact":true,
        "same_owner_native_restore_exact":true,"draft_loaded":false,"draft_metadata":"structural placeholders",
        "full_256k_prompt_measured":false,"target":info,"allocation_closed":true});
    fs::write(root.join("complete.json"),serde_json::to_vec_pretty(&result).unwrap()).unwrap();
    eprintln!("PASS_PLANAR_OWNER capacity=262144 B4352+7 B1 B8 rollback park restore allocations_closed");
}
