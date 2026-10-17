//! Independent adaptation collection/evaluation; never impersonates full B1 qualification.
use super::*;
pub(super) fn validate(probes:&Value)->std::result::Result<(),String> {
    if !matches!(probes["adaptation_mode"].as_str(),Some("collect"|"evaluate")){return Err("adaptation mode".into());}
    let rows=probes["cases"].as_array().ok_or("adaptation cases")?;
    if rows.is_empty() || rows.len()>16{return Err("adaptation case count".into());}
    let mut names=std::collections::BTreeSet::new();
    for c in rows {
        let name=c["id"].as_str().ok_or("adaptation id")?;
        if name.is_empty() || name.len()>64 || !name.bytes().all(|v|v.is_ascii_lowercase()||v.is_ascii_digit()||v==b'_') || !names.insert(name){return Err("adaptation id syntax/duplicate".into());}
        if c["family"]!="english_thinking" || c["thinking"]!=true || !c["followup"].is_null()
            || !matches!(c["experiment_partition"].as_str(),Some("train"|"validation"|"holdout_not_for_training"))
            || !c["max_new_tokens"].as_u64().is_some_and(|n|(1..=8192).contains(&n)) {return Err("adaptation case contract".into());}
        let input=c["token_ids"].as_array().ok_or("adaptation tokens")?;
        if input.is_empty() || input.len()+c["max_new_tokens"].as_u64().unwrap() as usize+8>262144 {return Err("adaptation capacity".into());}
    }
    Ok(())
}
pub(super) fn run_case(t:&mut CheckpointCudaTarget,b:&mut AugurDflashBridge,case:&Value,ops:&[Replay],eos:&[u32],mode:&str)->(Vec<u32>,u32,&'static str,Value) {
    let budget=case["max_new_tokens"].as_u64().unwrap() as usize;let name=case["id"].as_str().unwrap();
    let complete_teacher=std::env::var("AUGUR_COMPLETE_TEACHER").as_deref()==Ok("1");
    if complete_teacher {assert_eq!(mode,"evaluate");assert_eq!(case["experiment_partition"],"train");
        assert_eq!(std::env::var("AUGUR_SELECTOR_SUPERVISION").as_deref(),Ok("1"));}
    let mut timed=Vec::new();let mut reference=None;
    if mode=="evaluate" {for rep in 0..2 {
        let setup=restore(t,b,ops);let mut r=resident(t,b,budget,eos,false);
        if let Some(prior)=reference.as_ref(){equal(&r,prior);}
        r.wire["repeat"]=json!(rep);r.wire["setup_exact_replay_wall_s"]=json!(setup);
        r.wire["content_tokens"]=json!(r.output.len());r.wire["output_tokens"]=json!(r.output);
        timed.push(r.wire.clone());if reference.is_none(){reference=Some(r);}
    }} else {assert_eq!(mode,"collect");}
    restore(t,b,ops);
    let path=std::path::PathBuf::from(std::env::var_os("AUGUR_BRIDGE_OUTPUT").unwrap()).join(format!("{name}-hidden.f32"));
    let replay=complete_teacher.then(||complete_teacher::TeacherReplay{case_id:name,reference:reference.as_ref().unwrap(),
        stride:std::env::var("AUGUR_COMPLETE_TEACHER_STRIDE").unwrap().parse().unwrap()});
    let r=resident_with_teacher_replay(t,b,budget,eos,true,Some(&path),replay.as_ref());
    if let Some(prior)=reference.as_ref(){equal(&r,prior);}
    let input=flatten(ops);
    let result=json!({"schema":"quasar-selector-adaptation-case-v1","id":name,"family":case["family"],"thinking":true,
        "experiment_partition":case["experiment_partition"],"collection_only":mode=="collect","B1_comparison_performed":false,
        "input_tokens":input.len(),"input_token_ids":input,"input_u32le_sha256":hash_tokens(&input),
        "output_tokens":r.output,"pending_token":r.pending,"stop":r.stop,"final_consumed_cursor":r.cursor,
        "B8_same_policy_repeat_tokens_cursor_logits_exact":mode=="evaluate",
        "complete_teacher_probe_enabled":complete_teacher,"complete_teacher_probe_repeat_state_exact":complete_teacher,
        "target_execution_profile":super::super::super::batched::PROFILE,"target_projection_policy":b.batched.as_ref().unwrap().projection_policy,
        "hidden_file":path.file_name().unwrap().to_str().unwrap(),"hidden_bytes":std::fs::metadata(&path).unwrap().len(),
        "hidden_layout":"round_major_8_5120_f32_le","timed":timed,"audit":r.wire,"HTTP_or_Session_commit_authority":false});
    (r.output,r.pending,r.stop,result)
}
#[test]fn adaptation_case_partition_and_bounds_are_explicit(){
    let mut p=json!({"adaptation_mode":"collect","cases":[{"id":"train_graph","family":"english_thinking","thinking":true,
        "experiment_partition":"train","max_new_tokens":3072,"token_ids":[1,2],"followup":null}]});
    assert!(validate(&p).is_ok());p["cases"][0]["experiment_partition"]=json!("unmarked");assert!(validate(&p).is_err());
    p["cases"][0]["experiment_partition"]=json!("validation");p["cases"][0]["id"]=json!("../escape");assert!(validate(&p).is_err());
}

#[test]fn compact_missing_candidate_does_not_invent_a_logit(){
    let h=HeadDiagnostics{candidate_ids:(0..128).collect(),unary_scores:vec![0.;128],transition_scores:vec![0.;128],
        path:vec![0,16,32,48,64,80,96,112],logits:Vec::new(),normalized_hidden:Vec::new()};
    let missing=classify_rejection(&h,1,999);assert!(missing["correct_draft_head_logit"].is_null());
    assert_eq!(missing["reason"],"top16_coverage_missing");
    let present=classify_rejection(&h,1,17);assert_eq!(present["correct_draft_head_logit"],0.0);
}
