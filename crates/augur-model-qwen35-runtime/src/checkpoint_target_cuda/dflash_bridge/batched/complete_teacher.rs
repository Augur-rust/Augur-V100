//! Test-only conditional teacher probe; preserves the live proposal and target.
use super::*;

fn validate_path(width:usize,path:&[u32],anchor:u32,candidates:&[u32])->Result<()> {
    if width!=8 || path.len()!=8 || path[0]!=anchor || path.iter().any(|&id|id as usize>=V)
        || candidates.len()!=128 || candidates.iter().any(|&id|id as usize>=V) {
        return Err("complete teacher path/anchor/candidate admission".into());
    }
    Ok(())
}
fn logit_sha(values:&[f32])->String {
    let bytes:Vec<u8>=values.iter().flat_map(|v|v.to_bits().to_le_bytes()).collect();
    format!("{:x}",Sha256::digest(&bytes))
}
impl AugurDflashBridge {
    pub(crate) fn probe_complete_b8_teacher(&mut self,t:&mut CheckpointCudaTarget,
        p:&AugurResidentDraftProposal,teacher_path:&[u32],candidates:&[u32],
        workspace:&supervision::Workspace)->Result<Value> {
        self.validate_target(t)?;
        if !Rc::ptr_eq(&self.instance,&p.instance) || self.live_proposal!=Some(p.generation)
            || !Rc::ptr_eq(&self.stream,&p.stream)
            || t.dflash_capture.as_ref().unwrap().committed_identity()?!=p.identity.capture_identity() {
            return Err("complete teacher stale/foreign live proposal".into());
        }
        let executor=self.batched.as_ref().ok_or("complete teacher missing B8 executor")?;
        if executor.pending.is_some() || executor.parked || !workspace.matches_executor(executor) {
            return Err("complete teacher stale/foreign workspace or executor".into());
        }
        let anchor=t.greedy_next_token()?;
        validate_path(executor.width,teacher_path,anchor,candidates)?;
        if p.path[0]!=anchor {return Err("complete teacher proposal anchor changed".into());}
        let identity=t.dflash_capture.as_ref().unwrap().committed_identity()?;
        let cursor=t.consumed;let selected=t.selected;let current=t.logits_current;
        let logits=t.logits_snapshot()?;let before_sha=logit_sha(&logits);
        self.poisoned=true;
        let result=(|| {
            let b=self.batched.as_mut().unwrap();
            let ids=b.execute(t,teacher_path)?;
            // Actual target argmax after each forced predecessor, never the old
            // continuation token copied into a gold label.
            let mut labels=vec![None;8];
            for position in 1..8 {labels[position]=Some(ids[position-1]);}
            let mut teacher=workspace.read(candidates,&labels)?;
            b.commit(t,0)?;
            for program in &b.programs {match program {
                Program::Gdn(_,state)=>state.audit_zero_restored()?,
                Program::Full(_,state)=>state.audit_zero_restored(cursor)?,
            }}
            if identity!=t.dflash_capture.as_ref().unwrap().committed_identity()? || t.consumed!=cursor {
                return Err("complete teacher altered target/capture cursor".into());
            }
            let after=t.logits_snapshot()?;
            if after.len()!=logits.len() || after.iter().zip(&logits).any(|(a,b)|a.to_bits()!=b.to_bits()) {
                return Err("complete teacher altered committed logits bytes".into());
            }
            // Zero commit invalidates only the cached selected token. Restore
            // that metadata after proving all underlying state was unchanged.
            t.selected=selected;t.logits_current=current;
            teacher["schema"]=json!("quasar-complete-b8-conditional-teacher-v1");
            teacher["source"]=json!("actual_B8_target_probe_on_forced_reference_prefix");
            teacher["post_first_rejection_masked"]=json!(false);
            teacher["observed_acceptance_labels"]=json!(false);
            teacher["labels_are_conditional_on_forced_prefix"]=json!(true);
            teacher["teacher_only_forced_path"]=json!(teacher_path);
            teacher["selector_predecessor_ids"]=json!(&teacher_path[..7]);
            teacher["continuation_top1_agreement"]=json!((1..8).map(|j|ids[j-1]==teacher_path[j]).collect::<Vec<_>>());
            teacher["label_source"]=json!("actual_probe_top1_not_forced_continuation");
            teacher["target_start_consumed"]=json!(cursor);
            teacher["proposal_generation"]=json!(p.generation);
            teacher["rollback"]=json!({"committed_tokens":0,"consumed_cursor_unchanged":true,
                "capture_identity_unchanged":true,"target_logits_bytes_unchanged":true,
                "GDN_state_and_conv_bytes_unchanged":true,"KV_rejected_tail_bytes_unchanged":true,
                "selection_metadata_unchanged":t.selected==selected && t.logits_current==current,
                "before_logits_sha256":before_sha,"after_logits_sha256":logit_sha(&after),
                "GDN_layers":b.programs.iter().filter(|p|matches!(p,Program::Gdn(..))).count(),
                "full_attention_layers":b.programs.iter().filter(|p|matches!(p,Program::Full(..))).count()});
            Ok(teacher)
        })();
        match result {
            Ok(mut value)=>{
                if self.live_proposal!=Some(p.generation) {t.poisoned=true;return Err("complete teacher consumed live proposal".into());}
                value["live_proposal_preserved"]=json!(true);self.poisoned=false;Ok(value)
            },
            Err(error)=>{t.poisoned=true;Err(error)},
        }
    }
}
#[test]
fn complete_teacher_admission_requires_original_anchor_and_full_b8() {
    let ids:Vec<u32>=(0..128).collect();let path=vec![17,1,2,3,4,5,6,7];
    assert!(validate_path(8,&path,17,&ids).is_ok());
    assert!(validate_path(4,&path,17,&ids).is_err());
    assert!(validate_path(8,&path[..7],17,&ids).is_err());
    assert!(validate_path(8,&path,18,&ids).is_err());
    let mut invalid=path.clone();invalid[7]=V as u32;
    assert!(validate_path(8,&invalid,17,&ids).is_err());
}
#[test]
fn complete_teacher_logits_fence_is_bitwise() {
    assert_ne!(logit_sha(&[0.0]),logit_sha(&[-0.0]));
    assert_eq!(logit_sha(&[1.0,2.0]),logit_sha(&[1.0,2.0]));
}
