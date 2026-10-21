use super::*;
use std::{cell::RefCell, rc::Rc};
const EOS: u32 = 248046;
#[derive(Default)]
struct FakeState {
    ids: Vec<u32>,
    selected: Option<u32>,
    calls: usize,
    drains: usize,
    image_prefills: usize,
    image_history_prefills: usize,
    eos_at: Option<usize>,
    fail: &'static str,
    cancel: Option<SessionCancellationToken>,
}
struct Fake(Rc<RefCell<FakeState>>);
impl SessionTarget for Fake {
    fn supports_images(&self)->bool {true}
    fn prefill_images(&mut self,ids:&[u32],_:&[augur_model_qwen35::Qwen35PreparedF16Image],
        _: &SessionCancellationToken)->Result<Option<serde_json::Value>> {
        self.0.borrow_mut().image_prefills+=1;
        self.consume_sequence(ids)?;Ok(Some(serde_json::json!({"test_only":true})))
    }
    fn prefill_image_history(&mut self,ids:&[u32],cancel:&SessionCancellationToken)->Result<Option<serde_json::Value>> {
        self.0.borrow_mut().image_history_prefills+=1;
        self.prefill(ids,cancel)
    }
    fn consumed_tokens(&self) -> usize {
        self.0.borrow().ids.len()
    }
    fn consume_sequence(&mut self, ids: &[u32]) -> Result<()> {
        let mut s = self.0.borrow_mut();
        s.calls += 1;
        if s.fail == "prefill" {
            return Err("injected prefill failure".into());
        }
        s.ids.extend_from_slice(ids);
        s.selected = None;
        if let Some(c) = &s.cancel {
            c.request_cancel();
        }
        Ok(())
    }
    fn greedy_next_token(&mut self) -> Result<u32> {
        let mut s = self.0.borrow_mut();
        s.calls += 1;
        let id = if s.fail == "nonfinite" {
            VOCAB
        } else if s.eos_at == Some(s.ids.len()) {
            EOS
        } else {
            s.ids
                .iter()
                .fold(0u32, |v, &t| v.wrapping_mul(31).wrapping_add(t))
                % 97
        };
        s.selected = Some(id);
        Ok(id)
    }
    fn consume_selected(&mut self) -> Result<()> {
        let mut s = self.0.borrow_mut();
        s.calls += 1;
        let id = s.selected.take().ok_or("missing selection")?;
        if s.fail != "cursor" {
            s.ids.push(id);
        }
        Ok(())
    }
    fn drain(&mut self) -> Result<()> {
        let mut s = self.0.borrow_mut();
        s.drains += 1;
        if s.fail == "drain" { return Err("injected drain failure".into()); }
        Ok(())
    }
    fn reset_empty(&mut self) -> Result<()> {
        let mut s = self.0.borrow_mut();
        s.calls += 1;
        if s.fail == "reset" {
            return Err("injected reset failure".into());
        }
        if s.fail == "reset_cursor" { return Ok(()); }
        s.ids.clear();
        s.selected = None;
        Ok(())
    }
}
#[cfg(any(feature = "qwen35-augur-http", feature = "qwen35-nerkyor-cancel-recovery"))]
mod cancellation_discard;
#[cfg(feature = "qwen35-augur-vision")]
mod cancellation_gpu_fallback;
fn owner() -> (Session<Fake>, Rc<RefCell<FakeState>>) {
    let state = Rc::new(RefCell::new(FakeState::default()));
    (
        Session::new(Fake(state.clone()), "a".repeat(64), vec![EOS]).unwrap(),
        state,
    )
}
fn open() -> TurnRequest {
    TurnRequest {
        selector_policy: None, request_id: "r1".into(),
        session_id: "s1".into(),
        max_new_tokens: 3,
        input: TurnInput::Open {
            prompt_token_ids: vec![1, 2],
        },
    }
}
fn persist(r: &TurnReceipt) -> Result<String> {
    Ok(digest(&serde_json::to_vec(r)?))
}
fn run(o: &mut Session<Fake>, r: &TurnRequest) -> CommittedTurn {
    o.turn(r, &SessionCancellationToken::new(), |_| Ok(()), persist)
        .unwrap()
}
fn append(a: &SessionSnapshot) -> TurnRequest {
    TurnRequest {
        selector_policy: None, request_id: "r2".into(),
        session_id: a.session_id.clone(),
        max_new_tokens: 3,
        input: TurnInput::Append {
            revision: a.revision,
            exact_history: a.consumed_token_ids.clone(),
            suffix_token_ids: vec![8, 9],
        },
    }
}
#[test]
fn image_identity_is_required_for_retained_append_and_reset_clears_it() {
    let (mut o,state)=owner();let mut r=open();let rgb=vec![0u8;32*32*3];
    r.input=TurnInput::OpenImages {prompt_token_ids:vec![248053,248056,248054],images:vec![images::ImageInput {
        width:32,height:32,rgb8_hex:"00".repeat(rgb.len()),rgb8_sha256:digest(&rgb) }]};
    let first=run(&mut o,&r);assert!(first.receipt.vision_enabled);
    let hash=first.receipt.session.image_input_sha256.clone().unwrap();let calls=state.borrow().calls;
    let mut next=append(&first.receipt.session);
    assert!(o.turn(&next,&SessionCancellationToken::new(),|_|Ok(()),persist).is_err());
    assert_eq!(state.borrow().calls,calls);assert!(!o.poisoned());
    next.input=TurnInput::AppendImages {revision:1,exact_history:first.receipt.session.consumed_token_ids.clone(),
        suffix_token_ids:vec![9],image_input_sha256:"a".repeat(64)};
    assert!(o.turn(&next,&SessionCancellationToken::new(),|_|Ok(()),persist).is_err());
    if let TurnInput::AppendImages {image_input_sha256,..}=&mut next.input {*image_input_sha256=hash.clone();}
    let second=run(&mut o,&next);assert_eq!(second.receipt.session.image_input_sha256,Some(hash));
    assert_eq!(state.borrow().image_prefills,1); // The encoder is not replayed.
    assert_eq!(state.borrow().image_history_prefills,1); // Decode protection survives append.
    o.close("close","s1",2,|r|Ok(digest(&serde_json::to_vec(r)?))).unwrap();
    let mut fresh=open();fresh.request_id="r3".into();fresh.session_id="s2".into();
    let text=run(&mut o,&fresh);assert!(!text.receipt.vision_enabled);assert_eq!(text.receipt.session.image_input_sha256,None);
    assert_eq!(state.borrow().image_history_prefills,1); // Fresh text is a normal text path.
}
#[test]
fn text_requests_cannot_smuggle_image_placeholders() {
    let (mut o,state)=owner();let mut r=open();
    r.input=TurnInput::Open {prompt_token_ids:vec![248053,248056,248054]};
    assert!(o.turn(&r,&SessionCancellationToken::new(),|_|Ok(()),persist).is_err());
    assert_eq!(state.borrow().calls,0);assert!(!o.poisoned());
}
#[test]
fn exact_pending_append_matches_fresh_replay_and_persistence_is_commit_point() {
    let (mut o, state) = owner();
    let a = run(&mut o, &open());
    assert_eq!(a.receipt.output_token_ids.len(), 3);
    assert_eq!(a.receipt.session.consumed_token_ids.len(), 4);
    assert!(a.receipt.session.pending_token_published);
    assert_eq!(state.borrow().ids, a.receipt.session.consumed_token_ids);
    let mut prefix = state.borrow().ids.clone();
    prefix.push(a.receipt.session.pending_token);
    prefix.extend([8, 9]);
    let b = run(&mut o, &append(&a.receipt.session));
    assert_eq!(b.receipt.previous_receipt_sha256, Some(a.receipt_sha256));
    assert_eq!(b.receipt.session.revision, 2);
    assert_eq!(
        &b.receipt.session.consumed_token_ids[..prefix.len()],
        &prefix
    );
    let (mut replay, _) = owner();
    let mut r = open();
    r.input = TurnInput::Open {
        prompt_token_ids: prefix,
    };
    let fresh = run(&mut replay, &r);
    assert_eq!(b.receipt.output_token_ids, fresh.receipt.output_token_ids);
    assert_eq!(
        b.receipt.session.consumed_token_ids,
        fresh.receipt.session.consumed_token_ids
    );
    assert_eq!(
        b.receipt.session.pending_token,
        fresh.receipt.session.pending_token
    );
    assert_eq!(
        b.receipt.session.consumed_input_sha256,
        fresh.receipt.session.consumed_input_sha256
    );
}
#[test]
fn eos_is_unpublished_pending_and_consumed_once_on_append() {
    let (mut o, state) = owner();
    state.borrow_mut().eos_at = Some(2);
    let a = run(&mut o, &open());
    assert!(a.receipt.output_token_ids.is_empty());
    assert_eq!(a.receipt.stop_reason, "eos");
    assert_eq!(a.receipt.selected_tokens_including_eos, 1);
    assert!(!a.receipt.session.pending_token_published);
    assert_eq!(a.receipt.session.pending_token, EOS);
    let b = run(&mut o, &append(&a.receipt.session));
    assert_eq!(
        &b.receipt.session.consumed_token_ids[..5],
        &[1, 2, EOS, 8, 9]
    );
}
#[test]
fn admission_rejects_without_any_target_mutation_and_legacy_capacity_is_unchanged() {
    assert_eq!(crate::resident_session_profile::CAPACITY, 196608);
    assert_eq!(CAPACITY, 262144);
    let (mut o, state) = owner();
    let a = run(&mut o, &open());
    let good = append(&a.receipt.session);
    let before = state.borrow().calls;
    let mut bad = vec![open()];
    let mut r = good.clone();
    r.request_id = "../escape".into();
    bad.push(r);
    let mut r = good.clone();
    r.session_id = "wrong".into();
    bad.push(r);
    let mut r = good.clone();
    r.max_new_tokens = 0;
    bad.push(r);
    let mut r = good.clone();
    r.max_new_tokens = MAX_OUTPUT + 1;
    bad.push(r);
    for kind in 0..3 {
        let mut r = good.clone();
        if let TurnInput::Append {
            revision,
            exact_history,
            suffix_token_ids,
        } = &mut r.input
        {
            match kind {
                0 => *revision += 1,
                1 => exact_history[0] ^= 1,
                _ => suffix_token_ids.push(VOCAB),
            }
        }
        bad.push(r);
    }
    for r in bad {
        assert!(o
            .turn(
                &r,
                &SessionCancellationToken::new(),
                |_| panic!("emit"),
                |_| panic!("persist")
            )
            .is_err());
        assert!(!o.poisoned());
        assert_eq!(o.snapshot(), Some(&a.receipt.session));
        assert_eq!(state.borrow().calls, before);
    }
    run(&mut o, &good);
}
#[test]
fn full_capacity_budget_and_last_invalid_input_checked_before_cuda() {
    let (mut o, state) = owner();
    for ids in [vec![0; CAPACITY], vec![1, VOCAB], vec![]] {
        let mut r = open();
        r.input = TurnInput::Open {
            prompt_token_ids: ids,
        };
        assert!(o
            .turn(
                &r,
                &SessionCancellationToken::new(),
                |_| panic!(),
                |_| panic!()
            )
            .is_err());
        assert_eq!(state.borrow().calls, 0);
        assert!(!o.poisoned());
    }
    let mut r = open();
    r.input = TurnInput::Open {
        prompt_token_ids: vec![0; CAPACITY - 3],
    };
    let a = run(&mut o, &r);
    assert_eq!(a.receipt.session.consumed_token_ids.len(), CAPACITY - 1);
    let before = state.borrow().calls;
    assert!(o
        .turn(
            &append(&a.receipt.session),
            &SessionCancellationToken::new(),
            |_| panic!(),
            |_| panic!()
        )
        .is_err());
    assert_eq!(state.borrow().calls, before);
}
#[test]
fn cancellation_before_admission_preserves_state_during_execution_retires_owner() {
    let (mut o, state) = owner();
    let c = SessionCancellationToken::new();
    c.request_cancel();
    assert!(o.turn(&open(), &c, |_| panic!(), |_| panic!()).is_err());
    assert_eq!(state.borrow().calls, 0);
    assert!(!o.poisoned());
    let c = SessionCancellationToken::new();
    state.borrow_mut().cancel = Some(c.clone());
    let mut r = open();
    r.input = TurnInput::Open {
        prompt_token_ids: vec![1; PREFILL_CHUNK * 2],
    };
    assert!(o.turn(&r, &c, |_| panic!(), |_| panic!()).is_err());
    assert_eq!(state.borrow().ids.len(), PREFILL_CHUNK);
    assert!(o.poisoned());
    assert!(o.snapshot().is_none());
    assert!(state.borrow().drains > 0);
    assert!(o
        .turn(
            &open(),
            &SessionCancellationToken::new(),
            |_| panic!(),
            |_| panic!()
        )
        .is_err());
}
#[test]
fn cancellation_after_publication_does_not_consume_or_commit_pending() {
    let (mut o, state) = owner();
    let c = SessionCancellationToken::new();
    let result = o.turn(
        &open(),
        &c,
        |_| {
            c.request_cancel();
            Ok(())
        },
        |_| panic!("uncommitted"),
    );
    assert!(result.is_err());
    assert!(o.poisoned());
    assert_eq!(state.borrow().ids, [1, 2]);
}
#[test]
fn sink_and_receipt_failures_never_publish_a_reusable_revision() {
    for failure in ["sink", "receipt", "bad_digest"] {
        let (mut o, state) = owner();
        let result = o.turn(
            &open(),
            &SessionCancellationToken::new(),
            |_| {
                if failure == "sink" {
                    Err("disconnected".into())
                } else {
                    Ok(())
                }
            },
            |_| {
                if failure == "bad_digest" {
                    Ok("wrong".into())
                } else {
                    Err("fsync failed".into())
                }
            },
        );
        assert!(result.is_err());
        assert!(o.poisoned());
        assert!(o.snapshot().is_none());
        assert!(state.borrow().drains > 0);
    }
}
#[test]
fn nonfinite_submit_or_cursor_failure_is_fail_closed() {
    for fail in ["prefill", "nonfinite", "cursor"] {
        let (mut o, state) = owner();
        state.borrow_mut().fail = fail;
        assert!(o
            .turn(
                &open(),
                &SessionCancellationToken::new(),
                |_| Ok(()),
                |_| panic!()
            )
            .is_err());
        assert!(o.poisoned());
        assert!(o.snapshot().is_none());
        assert!(state.borrow().drains > 0);
    }
}
#[test]
fn close_requires_exact_revision_resets_then_persists_and_never_reuses_session_id() {
    let (mut o, state) = owner();
    let a = run(&mut o, &open());
    assert!(o.close("close1", "s1", 2, |_| panic!()).is_err());
    assert!(!o.poisoned());
    o.close("close1", "s1", 1, |r| {
        assert_eq!(r.last_turn_receipt_sha256, a.receipt_sha256);
        assert!(state.borrow().ids.is_empty());
        Ok("b".repeat(64))
    })
    .unwrap();
    assert!(o.snapshot().is_none());
    assert!(!o.poisoned());
    let mut r = open();
    r.request_id = "r3".into();
    assert!(o
        .turn(
            &r,
            &SessionCancellationToken::new(),
            |_| panic!(),
            |_| panic!()
        )
        .is_err());
    r.session_id = "s2".into();
    run(&mut o, &r);
    assert!(o.close("close2", "s2", 1, |_| Err("fsync".into())).is_err());
    assert!(o.poisoned());
}
#[test]
fn strict_protocol_and_bounded_identity() {
    let mut v = serde_json::to_value(open()).unwrap();
    v["unknown"] = true.into();
    assert!(serde_json::from_value::<TurnRequest>(v).is_err());
    let mut v = serde_json::to_value(open()).unwrap();
    v["input"]["unknown"] = true.into();
    assert!(serde_json::from_value::<TurnRequest>(v).is_err());
    for id in ["", "a/b", "a\\b", "中文", "a.b", &"x".repeat(65)] {
        assert!(!valid_id(id));
    }
    assert!(valid_id("ABC_xyz-012"));
    assert!(!valid_digest(&"A".repeat(64)));
    assert!(valid_digest(&"a".repeat(64)));
}
#[test]
fn receipts_are_create_new_byte_hashed_and_not_gpu_restore_authority() {
    let path = std::env::temp_dir().join(format!(
        "augur-augur-receipt-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&path).unwrap();
    let value = serde_json::json!({"profile":PROFILE,"unicode":"中文","product_ready":false});
    let hash = persist_receipt(&path, "r1", &value).unwrap();
    let bytes = std::fs::read(path.join("r1.json")).unwrap();
    assert_eq!(digest(&bytes), hash);
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&bytes).unwrap(),
        value
    );
    assert!(persist_receipt(&path, "r1", &value).is_err());
    assert!(persist_receipt(&path, "../escape", &value).is_err());
    assert_eq!(std::fs::read(path.join("r1.json")).unwrap(), bytes);
    std::fs::remove_file(path.join("r1.json")).unwrap();
    std::fs::remove_dir(path).unwrap();
}

#[test]
fn manifest_pins_all_metadata_and_exact_shard_namespace_before_cuda() {
    use super::manifest::{load, AugurManifest};
    let path = std::env::temp_dir().join(format!(
        "augur-augur-manifest-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&path).unwrap();
    let files = [
        ("tokenizer.json", b"{}".as_slice()),
        (
            "tokenizer_config.json",
            br#"{"chat_template":"template"}"#.as_slice(),
        ),
        ("chat_template.jinja", b"template".as_slice()),
        (
            "generation_config.json",
            br#"{"eos_token_id":[248046,248044],"do_sample":true}"#.as_slice(),
        ),
    ];
    for (name, bytes) in files {
        std::fs::write(path.join(name), bytes).unwrap();
    }
    let mut m = AugurManifest {
        schema: "quasar-session-model-manifest-v1".into(),
        upstream_repo: "QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4".into(),
        upstream_revision: "a".repeat(40),
        checkpoint: augur_model_qwen35::CheckpointExpectedIdentity {
            config_sha256: "b".repeat(64),
            index_sha256: "c".repeat(64),
            shards: (1..=5)
                .map(|i| (format!("model-{i:05}-of-00005.safetensors"), "d".repeat(64)))
                .collect(),
        },
        tokenizer_sha256: digest(files[0].1),
        tokenizer_config_sha256: digest(files[1].1),
        chat_template_sha256: digest(files[2].1),
        generation_config_sha256: digest(files[3].1),
    };
    let raw = serde_json::to_vec(&m).unwrap();
    let manifest_path = path.join("manifest.json");
    std::fs::write(&manifest_path, &raw).unwrap();
    let admitted = load(&path, &manifest_path, &digest(&raw)).unwrap();
    assert_eq!(admitted.eos_token_ids(), [248046, 248044]);
    assert_eq!(admitted.model_root(), path);
    assert_eq!(admitted.raw(), raw);
    assert!(load(&path, &manifest_path, &"0".repeat(64)).is_err());
    std::fs::write(path.join("chat_template.jinja"), b"different").unwrap();
    assert!(load(&path, &manifest_path, &digest(&raw)).is_err());
    m.chat_template_sha256 = digest(b"different");
    let raw = serde_json::to_vec(&m).unwrap();
    std::fs::write(&manifest_path, &raw).unwrap();
    // Even two independently hashed templates must agree, not silently choose one.
    assert!(load(&path, &manifest_path, &digest(&raw)).is_err());
    m.checkpoint
        .shards
        .remove("model-00001-of-00005.safetensors");
    m.checkpoint
        .shards
        .insert("../outside.safetensors".into(), "d".repeat(64));
    assert!(m.validate().is_err());
    for (name, _) in files {
        std::fs::remove_file(path.join(name)).unwrap();
    }
    std::fs::remove_file(manifest_path).unwrap();
    std::fs::remove_dir(path).unwrap();
}

// A wide prefill receives the whole prefix once, not outer 128-row chunks.
struct WholePrefillFake {
    inner: Fake,
    calls: Rc<RefCell<Vec<Vec<u32>>>>,
}
impl SessionTarget for WholePrefillFake {
    fn consumed_tokens(&self) -> usize {
        self.inner.consumed_tokens()
    }
    fn consume_sequence(&mut self, _: &[u32]) -> Result<()> {
        Err("outer chunk loop must not run".into())
    }
    fn prefill(
        &mut self,
        ids: &[u32],
        cancel: &SessionCancellationToken,
    ) -> Result<Option<serde_json::Value>> {
        cancel.check("whole prefix")?;
        let first = self.inner.consumed_tokens();
        self.calls.borrow_mut().push(ids.to_vec());
        self.inner.consume_sequence(ids)?;
        Ok(Some(serde_json::json!({"first":first,"rows":ids.len()})))
    }
    fn greedy_next_token(&mut self) -> Result<u32> {
        self.inner.greedy_next_token()
    }
    fn consume_selected(&mut self) -> Result<()> {
        self.inner.consume_selected()
    }
    fn drain(&mut self) -> Result<()> {
        self.inner.drain()
    }
    fn reset_empty(&mut self) -> Result<()> {
        self.inner.reset_empty()
    }
}
#[test]
fn whole_prefix_prefill_receipt_and_pending_append_are_preserved() {
    let calls = Rc::new(RefCell::new(Vec::new()));
    let target = WholePrefillFake {
        inner: Fake(Rc::new(RefCell::new(FakeState::default()))),
        calls: calls.clone(),
    };
    let mut owner = Session::new(target, "a".repeat(64), vec![EOS]).unwrap();
    let mut request = open();
    request.input = TurnInput::Open {
        prompt_token_ids: vec![1; 8236],
    };
    request.max_new_tokens = 1;
    let first = owner
        .turn(
            &request,
            &SessionCancellationToken::new(),
            |_| Ok(()),
            persist,
        )
        .unwrap();
    assert_eq!(calls.borrow().len(), 1);
    assert_eq!(calls.borrow()[0].len(), 8236);
    assert_eq!(
        first.receipt.prefill_detail,
        Some(serde_json::json!({"first":0,"rows":8236}))
    );
    let request = append(&first.receipt.session);
    let second = owner
        .turn(
            &request,
            &SessionCancellationToken::new(),
            |_| Ok(()),
            persist,
        )
        .unwrap();
    assert_eq!(
        calls.borrow()[1],
        vec![first.receipt.session.pending_token, 8, 9]
    );
    assert_eq!(
        second.receipt.prefill_detail,
        Some(serde_json::json!({"first":8236,"rows":3}))
    );
    assert_eq!(second.receipt.session.revision, 2);
}

mod speculative;

#[test]
fn reserved_receipt_never_overwrites_a_collision_or_partial_commit() {
    use std::io::{Seek,SeekFrom,Write};
    let root=std::env::temp_dir().join(format!("augur-reserved-receipt-{}-{}",std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    std::fs::create_dir(&root).unwrap(); let value=serde_json::json!({"unicode":"中文","revision":1});
    assert!(reserve_receipt(&root,"../escape").is_err());
    let mut file=reserve_receipt(&root,"r1").unwrap();
    assert!(reserve_receipt(&root,"r1").is_err());
    let hash=persist_reserved_receipt(&mut file,&value).unwrap();
    let bytes=std::fs::read(root.join("r1.json")).unwrap();
    assert_eq!(hash,digest(&bytes)); assert!(bytes.ends_with(b"\n"));
    assert_eq!(serde_json::from_slice::<serde_json::Value>(&bytes).unwrap(),value);
    assert!(persist_reserved_receipt(&mut file,&value).is_err());
    assert_eq!(std::fs::read(root.join("r1.json")).unwrap(),bytes);
    let mut partial=reserve_receipt(&root,"partial").unwrap();
    partial.write_all(b"{").unwrap(); partial.seek(SeekFrom::Start(0)).unwrap();
    assert!(persist_reserved_receipt(&mut partial,&value).is_err());
    assert_eq!(std::fs::read(root.join("partial.json")).unwrap(),b"{");
    drop(partial); drop(file);
    std::fs::remove_file(root.join("partial.json")).unwrap();
    std::fs::remove_file(root.join("r1.json")).unwrap(); std::fs::remove_dir(root).unwrap();
}

#[test]
fn receipt_admission_runs_only_after_validation_and_before_target_work() {
    let (mut owner,state)=owner(); let mut bad=open(); bad.max_new_tokens=0;
    let called=std::cell::Cell::new(false);
    assert!(owner.turn_with_admission(&bad,&SessionCancellationToken::new(),
        || {called.set(true);Ok(())}, |_|Ok(()),persist).is_err());
    assert!(!called.get()); assert_eq!(state.borrow().calls,0);
    // Reservation failure does not admit the request or destroy the live state.
    assert!(owner.turn_with_admission(&open(),&SessionCancellationToken::new(),
        || Err("injected receipt reservation failure".into()), |_|Ok(()),persist).is_err());
    assert!(!owner.poisoned()); assert_eq!(state.borrow().calls,0);
    let result=owner.turn_with_admission(&open(),&SessionCancellationToken::new(),
        || {assert_eq!(state.borrow().calls,0);called.set(true);Ok(())}, |_|Ok(()),persist).unwrap();
    assert!(called.get()); assert_eq!(result.receipt.output_token_ids.len(),3);
    called.set(false);
    assert!(owner.turn_with_admission(&open(),&SessionCancellationToken::new(),
        || {called.set(true);Ok(())}, |_|Ok(()),persist).is_err());
    assert!(!called.get());
}

#[test]
fn admits_192ki_output_budget_without_forcing_full_generation() {
    let (mut owner, state) = owner();
    let mut request = open();
    request.max_new_tokens = 196608;
    state.borrow_mut().eos_at = Some(2);
    let result = run(&mut owner, &request);
    assert!(result.receipt.output_token_ids.is_empty());
    assert_eq!(result.receipt.stop_reason, "eos");
}
