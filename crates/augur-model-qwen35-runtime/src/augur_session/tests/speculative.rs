use super::*;
struct Batched { inner:Fake, width:usize, corrupt:&'static str }
impl SessionTarget for Batched {
    fn profile(&self)->&'static str {DFLASH_B8_PROFILE}
    fn dflash_enabled(&self)->bool {true}
    fn consumed_tokens(&self)->usize {self.inner.consumed_tokens()}
    fn consume_sequence(&mut self,ids:&[u32])->Result<()> {self.inner.consume_sequence(ids)}
    fn greedy_next_token(&mut self)->Result<u32> {self.inner.greedy_next_token()}
    fn consume_selected(&mut self)->Result<()> {self.inner.consume_selected()}
    fn drain(&mut self)->Result<()> {self.inner.drain()}
    fn reset_empty(&mut self)->Result<()> {self.inner.reset_empty()}
    fn speculative_step(&mut self,anchor:u32,remaining:usize,eos:&[u32],_:&SessionCancellationToken)->Result<Option<SpeculativeStep>> {
        let start=self.consumed_tokens();let mut tokens=vec![anchor];
        loop {
            if tokens.len()==remaining {break;}
            self.inner.consume_selected()?;
            if tokens.len()==self.width {break;}
            let next=self.inner.greedy_next_token()?;
            if eos.contains(&next) {break;}
            tokens.push(next);
        }
        let mut consumed=self.consumed_tokens()-start;
        match self.corrupt {
            "empty"=>tokens.clear(),"anchor"=>tokens[0]+=1,"wide"=>tokens.resize(9,1),
            "eos"=>tokens.push(EOS),"vocab"=>tokens.push(VOCAB),"count"=>consumed+=1,
            "cursor"=>self.inner.0.borrow_mut().ids.push(1),_=>{}
        }
        Ok(Some(SpeculativeStep{tokens,consumed}))
    }
}
fn batch(width:usize)->(Session<Batched>,Rc<RefCell<FakeState>>) {
    let s=Rc::new(RefCell::new(FakeState::default()));
    (Session::new(Batched{inner:Fake(s.clone()),width,corrupt:""},"a".repeat(64),vec![EOS]).unwrap(),s)
}
fn br(o:&mut Session<Batched>,r:&TurnRequest)->CommittedTurn {
    let mut emitted=Vec::new();let t=o.turn(r,&SessionCancellationToken::new(),|e|{assert_eq!(e.sequence,emitted.len());emitted.push(e.token_id);Ok(())},persist).unwrap();
    assert_eq!(emitted,t.receipt.output_token_ids);assert!(t.receipt.dflash_enabled);t
}
fn same(actual:&TurnReceipt,expected:&TurnReceipt) {
    assert_eq!(actual.output_token_ids,expected.output_token_ids);assert_eq!(actual.stop_reason,expected.stop_reason);
    assert_eq!(actual.selected_tokens_including_eos,expected.selected_tokens_including_eos);
    let mut e=expected.session.clone();e.profile=DFLASH_B8_PROFILE.into();assert_eq!(actual.session,e);
}
#[test] fn explicit_mode_rejects_silent_fallbacks() {
    assert!(!parse_dflash_mode(None).unwrap());assert!(!parse_dflash_mode(Some("resident_only")).unwrap());
    assert!(parse_dflash_mode(Some("batched8")).unwrap());for s in ["","true","8","batched8 ","BATCHED8"] {assert!(parse_dflash_mode(Some(s)).is_err());}
}
#[test] fn every_prefix_width_and_length_has_single_anchor_and_exact_pending_append() {
    for width in 1..=8 {for budget in 1..=25 {
        let(mut b,state)=batch(width);let(mut base,_)=owner();let mut r=open();r.max_new_tokens=budget;
        let a=br(&mut b,&r);let e=run(&mut base,&r);same(&a.receipt,&e.receipt);
        assert_eq!(state.borrow().ids,a.receipt.session.consumed_token_ids);
        let mut next=append(&a.receipt.session);next.max_new_tokens=budget;
        let aa=br(&mut b,&next);let ee=run(&mut base,&next);same(&aa.receipt,&ee.receipt);
        assert_eq!(aa.receipt.previous_receipt_sha256,Some(a.receipt_sha256));
    }}
}
#[test] fn eos_at_every_block_position_stays_unpublished_and_unconsumed() {
    for width in 1..=8 {for offset in 0..=17 {
        let(mut b,bs)=batch(width);let(mut base,es)=owner();bs.borrow_mut().eos_at=Some(2+offset);es.borrow_mut().eos_at=Some(2+offset);
        let mut r=open();r.max_new_tokens=24;let a=br(&mut b,&r);let e=run(&mut base,&r);same(&a.receipt,&e.receipt);
        assert!(!a.receipt.output_token_ids.contains(&EOS));assert!(!a.receipt.session.pending_token_published);
    }}
}
#[test] fn invalid_speculative_steps_never_persist_or_reuse_a_session() {
    for corrupt in ["empty","anchor","wide","eos","vocab","count","cursor"] {
        let(mut b,_)=batch(8);b.target.corrupt=corrupt;let mut r=open();r.max_new_tokens=20;let mut persisted=false;
        assert!(b.turn(&r,&SessionCancellationToken::new(),|_|Ok(()),|_|{persisted=true;Ok("a".repeat(64))}).is_err());
        assert!(!persisted);assert!(b.poisoned());assert!(b.snapshot().is_none());
    }
}
#[test] fn failed_extra_token_publication_is_not_a_successful_transaction() {
    let(mut b,_)=batch(8);let mut r=open();r.max_new_tokens=20;let mut persisted=false;
    assert!(b.turn(&r,&SessionCancellationToken::new(),|e|if e.sequence==1 {Err("closed stream".into())}else{Ok(())},|_|{persisted=true;Ok("a".repeat(64))}).is_err());
    assert!(!persisted);assert!(b.poisoned());assert!(b.snapshot().is_none());
}
#[cfg(feature="qwen35-augur-http")]
#[test] fn cancellation_during_extra_publication_discards_whole_completed_block() {
    let(mut b,state)=batch(8);let mut r=open();r.max_new_tokens=20;let cancel=SessionCancellationToken::new();let mut persisted=false;
    let e=b.turn(&r,&cancel,|e|{if e.sequence==1 {cancel.request_cancel();}Ok(())},|_|{persisted=true;Ok("a".repeat(64))}).unwrap_err();
    assert!(e.is::<crate::session_cancellation::SessionCancellationRequested>());assert!(!persisted);assert!(b.poisoned());
    assert_eq!(state.borrow().ids.len(),10);
    let d=b.discard_cancelled_turn("discard","r1","s1",|r|Ok(digest(&serde_json::to_vec(r)?))).unwrap();
    assert!(d.receipt.dflash_enabled);assert_eq!(d.receipt.submitted_tokens_before_discard,10);assert!(state.borrow().ids.is_empty());
    let mut fresh=open();fresh.request_id="fresh-r".into();fresh.session_id="fresh-s".into();
    let a=br(&mut b,&fresh);let(mut reference,_)=batch(8);let e=br(&mut reference,&fresh);same(&a.receipt,&e.receipt);
}

#[test] fn explicit_plan_cache_mode_rejects_unknown_values() {
    assert!(!parse_dflash_plan_cache(None).unwrap());assert!(!parse_dflash_plan_cache(Some("recreate")).unwrap());
    assert!(parse_dflash_plan_cache(Some("park")).unwrap());
    for value in ["","true","park ","PARK","resident"] {assert!(parse_dflash_plan_cache(Some(value)).is_err());}
}

#[test] fn parked_plan_default_only_applies_to_active_b8_and_has_explicit_rollback() {
    assert!(!resolve_dflash_plan_cache(false,None).unwrap());assert!(resolve_dflash_plan_cache(true,None).unwrap());
    assert!(!resolve_dflash_plan_cache(true,Some("recreate")).unwrap());assert!(resolve_dflash_plan_cache(true,Some("park")).unwrap());
    assert!(!resolve_dflash_plan_cache(false,Some("recreate")).unwrap());assert!(resolve_dflash_plan_cache(false,Some("park")).is_err());
    for active in [true,false] {assert!(resolve_dflash_plan_cache(active,Some("invalid")).is_err());}
}
