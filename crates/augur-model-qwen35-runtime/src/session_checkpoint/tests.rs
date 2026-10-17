use super::*;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::io::Cursor;
use std::rc::Rc;

type Trace = Rc<RefCell<Vec<&'static str>>>;

#[cfg(windows)]
#[path = "filesystem_tests.rs"]
mod filesystem_tests;

fn plan() -> TransferPlan {
    // Synthetic byte fixtures exercise transport only, not Qwen state completeness.
    TransferPlan {
        binding_sha256: sha256(b"independently bound complete contract fixture"),
        components: vec![
            Component {
                name: "target.fixture".into(),
                bytes: 17,
            },
            Component {
                name: "draft.fixture".into(),
                bytes: 5,
            },
            Component {
                name: "session.ledger".into(),
                bytes: 3,
            },
        ],
    }
}

struct Source {
    plan: TransferPlan,
    data: Vec<Vec<u8>>,
    trace: Trace,
    fail: Option<&'static str>,
    abort_fails: bool,
    largest_read: usize,
}

impl Source {
    fn new(plan: TransferPlan, trace: Trace) -> Self {
        let data = plan
            .components
            .iter()
            .enumerate()
            .map(|(i, c)| (0..c.bytes).map(|v| ((v + i as u64) % 251) as u8).collect())
            .collect();
        Self {
            plan,
            data,
            trace,
            fail: None,
            abort_fails: false,
            largest_read: 0,
        }
    }

    fn step(&self, name: &'static str) -> io::Result<()> {
        self.trace.borrow_mut().push(name);
        require(self.fail != Some(name), name)
    }
}

impl CaptureSource for Source {
    fn begin(&mut self) -> io::Result<TransferPlan> {
        self.step("capture begin")?;
        let mut plan = self.plan.clone();
        if self.fail == Some("begin drift") {
            plan.binding_sha256 = sha256(b"different begin");
        }
        Ok(plan)
    }
    fn read_exact(&mut self, component: usize, offset: u64, output: &mut [u8]) -> io::Result<()> {
        self.step("capture read")?;
        self.largest_read = self.largest_read.max(output.len());
        let start = usize::try_from(offset).map_err(io::Error::other)?;
        output.copy_from_slice(&self.data[component][start..start + output.len()]);
        Ok(())
    }
    fn finish(&mut self) -> io::Result<TransferPlan> {
        self.step("capture finish")?;
        let mut plan = self.plan.clone();
        if self.fail == Some("finish drift") {
            plan.binding_sha256 = sha256(b"different finish");
        }
        Ok(plan)
    }
    fn abort(&mut self) -> io::Result<()> {
        self.trace.borrow_mut().push("capture abort");
        require(!self.abort_fails, "capture abort failed")
    }
}

#[derive(Default)]
struct Store {
    chunks: BTreeMap<String, Vec<u8>>,
    manifest: Option<Vec<u8>>,
    trace: Trace,
    fail: Option<&'static str>,
    abort_fails: bool,
}

impl CheckpointWriter for Store {
    fn write_chunk(&mut self, digest: &str, bytes: &[u8]) -> io::Result<()> {
        self.trace.borrow_mut().push("chunk write");
        require(self.fail != Some("chunk write"), "injected chunk write")?;
        if let Some(prior) = self.chunks.get(digest) {
            require(prior == bytes, "immutable collision")?;
        } else {
            self.chunks.insert(digest.into(), bytes.into());
        }
        Ok(())
    }
    fn seal_manifest(&mut self, bytes: &[u8], digest: &str) -> io::Result<()> {
        self.trace.borrow_mut().push("manifest seal");
        require(self.fail != Some("manifest seal"), "injected manifest seal")?;
        require(
            self.manifest.is_none() && sha256(bytes) == digest,
            "no clobber manifest",
        )?;
        self.manifest = Some(bytes.into());
        Ok(())
    }
    fn abort(&mut self) -> io::Result<()> {
        self.trace.borrow_mut().push("store abort");
        require(!self.abort_fails, "store abort failed")
    }
}

impl CheckpointReader for Store {
    type ChunkReader = Cursor<Vec<u8>>;
    fn open_chunk(&mut self, digest: &str) -> io::Result<Self::ChunkReader> {
        self.trace.borrow_mut().push("chunk open");
        self.chunks
            .get(digest)
            .cloned()
            .map(Cursor::new)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "chunk missing"))
    }
    fn finish_read(&mut self) -> io::Result<()> {
        self.trace.borrow_mut().push("store final verification");
        require(
            self.fail != Some("store final verification"),
            "late storage drift",
        )
    }
}

#[derive(Default)]
struct Target {
    trace: Trace,
    data: Vec<Vec<u8>>,
    fail: Option<&'static str>,
    abort_fails: bool,
    writes: usize,
    complete: bool,
    poisoned: bool,
}

impl RestoreTarget for Target {
    fn begin(&mut self, expected: &TransferPlan) -> io::Result<()> {
        self.trace.borrow_mut().push("restore begin");
        require(self.fail != Some("restore begin"), "injected begin")?;
        self.data = expected
            .components
            .iter()
            .map(|c| vec![0; c.bytes as usize])
            .collect();
        Ok(())
    }
    fn write_exact(&mut self, component: usize, offset: u64, bytes: &[u8]) -> io::Result<()> {
        self.trace.borrow_mut().push("restore write");
        self.writes += 1;
        // Intentionally fail after mutation to test irreversible abort.
        let start = offset as usize;
        self.data[component][start..start + bytes.len()].copy_from_slice(bytes);
        require(
            self.fail != Some("restore write"),
            "injected post-write failure",
        )
    }
    fn finish_unpublished(&mut self) -> io::Result<()> {
        self.trace.borrow_mut().push("restore finish");
        require(
            self.fail != Some("restore finish"),
            "injected final drain failure",
        )?;
        self.complete = true;
        Ok(())
    }
    fn abort(&mut self) -> io::Result<()> {
        self.trace.borrow_mut().push("restore abort");
        self.poisoned = true;
        self.complete = false;
        self.data.clear();
        require(!self.abort_fails, "destination abort failed")
    }
}

fn saved() -> (Source, Store, TransferReceipt) {
    let trace = Trace::default();
    let expected = plan();
    let mut source = Source::new(expected.clone(), trace.clone());
    let mut store = Store {
        trace,
        ..Store::default()
    };
    let receipt = save(&mut source, &mut store, &expected, 7).unwrap();
    (source, store, receipt)
}

fn assert_bad_manifest(bytes: &[u8], hash: &str, expected: &TransferPlan) {
    let mut store = Store::default();
    let mut target = Target::default();
    let err = restore(&mut store, &mut target, bytes, hash, expected).unwrap_err();
    assert_eq!(err.phase, "restore preflight");
    assert!(target.trace.borrow().is_empty());
    assert!(store.trace.borrow().is_empty());
}

fn mutate_manifest(f: impl FnOnce(&mut Manifest)) {
    let (source, store, _) = saved();
    let mut manifest: Manifest = serde_json::from_slice(store.manifest.as_ref().unwrap()).unwrap();
    f(&mut manifest);
    let bytes = serde_json::to_vec(&manifest).unwrap();
    assert_bad_manifest(&bytes, &sha256(&bytes), &source.plan);
}

#[test]
fn complete_roundtrip_exact_payload_and_order() {
    let (source, mut store, saved) = saved();
    let bytes = store.manifest.clone().unwrap();
    let mut target = Target::default();
    let restored = restore(
        &mut store,
        &mut target,
        &bytes,
        &saved.manifest_sha256,
        &source.plan,
    )
    .unwrap();
    assert_eq!(saved, restored);
    assert_eq!(restored.payload_bytes, 25);
    assert_eq!(restored.chunks, 5);
    assert_eq!(restored.staging_bytes, 7);
    assert_eq!(target.data, source.data);
    assert!(target.complete && !target.poisoned);
    let trace = source.trace.borrow();
    assert!(
        trace.iter().position(|s| *s == "capture finish").unwrap()
            < trace.iter().position(|s| *s == "manifest seal").unwrap()
    );
}

#[test]
fn large_component_uses_bounded_staging_with_exact_short_tail() {
    let expected = TransferPlan {
        binding_sha256: sha256(b"big fixture"),
        components: vec![Component {
            name: "test.large".into(),
            bytes: MAX_CHUNK_BYTES as u64 + 3,
        }],
    };
    let mut source = Source::new(expected.clone(), Trace::default());
    let mut store = Store::default();
    let receipt = save(&mut source, &mut store, &expected, MAX_CHUNK_BYTES).unwrap();
    assert_eq!(source.largest_read, MAX_CHUNK_BYTES);
    assert_eq!(receipt.chunks, 2);
    let manifest: Manifest = serde_json::from_slice(store.manifest.as_ref().unwrap()).unwrap();
    assert_eq!(manifest.chunks[1].bytes, 3);
}

#[test]
fn capture_failure_or_identity_drift_never_seals() {
    for phase in [
        "capture begin",
        "capture read",
        "capture finish",
        "begin drift",
        "finish drift",
    ] {
        let expected = plan();
        let mut source = Source::new(expected.clone(), Trace::default());
        source.fail = Some(phase);
        let mut store = Store::default();
        save(&mut source, &mut store, &expected, 7).unwrap_err();
        assert!(store.manifest.is_none());
        assert_eq!(
            source
                .trace
                .borrow()
                .iter()
                .filter(|s| **s == "capture abort")
                .count(),
            1
        );
        assert_eq!(store.trace.borrow().last().unwrap(), &"store abort");
        assert!(!store.trace.borrow().contains(&"manifest seal"));
    }
}

#[test]
fn storage_failure_drains_capture_and_aborts_store() {
    for phase in ["chunk write", "manifest seal"] {
        let expected = plan();
        let mut source = Source::new(expected.clone(), Trace::default());
        let mut store = Store {
            fail: Some(phase),
            ..Store::default()
        };
        let err = save(&mut source, &mut store, &expected, 7).unwrap_err();
        assert_eq!(err.phase, phase);
        assert!(store.manifest.is_none());
        assert!(source.trace.borrow().contains(&"capture abort"));
        assert!(store.trace.borrow().contains(&"store abort"));
    }
}

#[test]
fn both_capture_and_storage_cleanup_errors_remain_visible() {
    let expected = plan();
    let mut source = Source::new(expected.clone(), Trace::default());
    source.fail = Some("capture read");
    source.abort_fails = true;
    let mut store = Store {
        abort_fails: true,
        ..Store::default()
    };
    let err = save(&mut source, &mut store, &expected, 7).unwrap_err();
    assert_eq!(err.phase, "capture read");
    assert_eq!(err.abort_errors.len(), 2);
    assert!(err.to_string().contains("capture abort failed"));
    assert!(err.to_string().contains("store abort failed"));
}

#[test]
fn invalid_save_quotas_do_not_touch_source_or_store() {
    for chunk_bytes in [0, MAX_CHUNK_BYTES + 1] {
        let expected = plan();
        let mut source = Source::new(expected.clone(), Trace::default());
        let mut store = Store::default();
        let err = save(&mut source, &mut store, &expected, chunk_bytes).unwrap_err();
        assert_eq!(err.phase, "save preflight");
        assert!(source.trace.borrow().is_empty() && store.trace.borrow().is_empty());
    }
    let mut expected = plan();
    expected.components[0].bytes = MAX_CHUNKS as u64 + 1;
    assert!(chunk_count(&expected, 1).is_err());
    expected.components[0].bytes = u64::MAX;
    assert!(expected.validate().is_err());
}

#[test]
fn role_namespace_and_payload_quota_are_closed() {
    for role in ["", "../x", "x/y", "x\\y", "x:y", "x\0y", "X", "a..b"] {
        let mut expected = plan();
        expected.components[0].name = role.into();
        assert!(expected.validate().is_err(), "role {role:?}");
    }
    let mut expected = plan();
    expected.components[0].name = expected.components[1].name.clone();
    assert!(expected.validate().is_err());
    let mut expected = plan();
    expected.components[0].bytes = 0;
    assert!(expected.validate().is_err());
    let mut expected = plan();
    expected.components[0].bytes = MAX_PAYLOAD_BYTES;
    assert!(expected.validate().is_err());
}

#[test]
fn wrong_external_manifest_hash_has_no_destination_side_effects() {
    let (source, store, _) = saved();
    assert_bad_manifest(
        store.manifest.as_ref().unwrap(),
        &sha256(b"wrong"),
        &source.plan,
    );
}

#[test]
fn wrong_session_binding_is_rejected_before_begin() {
    let (source, store, saved) = saved();
    let mut expected = source.plan;
    expected.binding_sha256 = sha256(b"another model/build/cursor/namespace");
    assert_bad_manifest(
        store.manifest.as_ref().unwrap(),
        &saved.manifest_sha256,
        &expected,
    );
}

#[test]
fn unknown_root_or_nested_fields_fail_closed() {
    let (source, store, _) = saved();
    for nested in [false, true] {
        let mut value: serde_json::Value =
            serde_json::from_slice(store.manifest.as_ref().unwrap()).unwrap();
        let object = if nested {
            value["plan"].as_object_mut().unwrap()
        } else {
            value.as_object_mut().unwrap()
        };
        object.insert("product_ready".into(), true.into());
        let bytes = serde_json::to_vec(&value).unwrap();
        assert_bad_manifest(&bytes, &sha256(&bytes), &source.plan);
    }
}

#[test]
fn duplicate_fields_noncanonical_bytes_and_truncation_reject() {
    let (source, store, _) = saved();
    let original = store.manifest.unwrap();
    let text = String::from_utf8(original.clone()).unwrap();
    let duplicate = text
        .replacen("{", &format!("{{\"schema\":\"{SCHEMA}\","), 1)
        .into_bytes();
    let mut whitespace = original.clone();
    whitespace.push(b'\n');
    let truncated = original[..original.len() - 1].to_vec();
    for bytes in [duplicate, whitespace, truncated] {
        assert_bad_manifest(&bytes, &sha256(&bytes), &source.plan);
    }
}

#[test]
fn bounded_manifest_input_rejects_oversize_before_parsing() {
    let bytes = vec![b' '; MAX_MANIFEST_BYTES + 1];
    assert_bad_manifest(&bytes, &sha256(&bytes), &plan());
}

#[test]
fn missing_or_extra_component_rejected() {
    mutate_manifest(|m| {
        m.plan.components.pop();
    });
    mutate_manifest(|m| {
        m.plan.components.push(Component {
            name: "unexpected".into(),
            bytes: 1,
        })
    });
}

#[test]
fn chunk_gap_overlap_reorder_and_wrong_component_rejected() {
    mutate_manifest(|m| m.chunks[1].offset += 1);
    mutate_manifest(|m| m.chunks[1].offset -= 1);
    mutate_manifest(|m| m.chunks.swap(0, 1));
    mutate_manifest(|m| m.chunks[0].component = 1);
}

#[test]
fn chunk_missing_extra_or_wrong_extent_rejected() {
    mutate_manifest(|m| {
        m.chunks.pop();
    });
    mutate_manifest(|m| m.chunks.push(m.chunks[0].clone()));
    mutate_manifest(|m| m.chunks[0].bytes = 0);
    mutate_manifest(|m| m.chunks[0].bytes += 1);
    mutate_manifest(|m| m.chunk_bytes = MAX_CHUNK_BYTES + 1);
}

#[test]
fn role_and_offset_are_bound_into_each_content_address() {
    let a = Component {
        name: "a".into(),
        bytes: 14,
    };
    let b = Component {
        name: "b".into(),
        bytes: 14,
    };
    assert_ne!(chunk_digest(&a, 0, b"same"), chunk_digest(&b, 0, b"same"));
    assert_ne!(chunk_digest(&a, 0, b"same"), chunk_digest(&a, 4, b"same"));
}

#[test]
fn corrupt_missing_short_and_extended_chunk_never_reaches_target_write() {
    for mutation in ["corrupt", "missing", "short", "extended"] {
        let (source, mut store, saved) = saved();
        let bytes = store.manifest.clone().unwrap();
        let manifest: Manifest = serde_json::from_slice(&bytes).unwrap();
        let key = &manifest.chunks[0].sha256;
        match mutation {
            "corrupt" => store.chunks.get_mut(key).unwrap()[0] ^= 1,
            "missing" => {
                store.chunks.remove(key);
            }
            "short" => {
                store.chunks.get_mut(key).unwrap().pop();
            }
            "extended" => store.chunks.get_mut(key).unwrap().push(0),
            _ => unreachable!(),
        }
        let mut target = Target::default();
        restore(
            &mut store,
            &mut target,
            &bytes,
            &saved.manifest_sha256,
            &source.plan,
        )
        .unwrap_err();
        assert_eq!(target.writes, 0, "{mutation}");
        assert!(target.poisoned && !target.complete);
    }
}

#[test]
fn late_corruption_discards_partially_written_destination() {
    let (source, mut store, saved) = saved();
    let bytes = store.manifest.clone().unwrap();
    let manifest: Manifest = serde_json::from_slice(&bytes).unwrap();
    store.chunks.get_mut(&manifest.chunks[1].sha256).unwrap()[0] ^= 1;
    let mut target = Target::default();
    let err = restore(
        &mut store,
        &mut target,
        &bytes,
        &saved.manifest_sha256,
        &source.plan,
    )
    .unwrap_err();
    assert_eq!(err.phase, "chunk verify");
    assert_eq!(target.writes, 1);
    assert!(target.poisoned && target.data.is_empty());
}

#[test]
fn destination_begin_write_and_finish_failures_always_abort() {
    for phase in ["restore begin", "restore write", "restore finish"] {
        let (source, mut store, saved) = saved();
        let bytes = store.manifest.clone().unwrap();
        let mut target = Target {
            fail: Some(phase),
            ..Target::default()
        };
        restore(
            &mut store,
            &mut target,
            &bytes,
            &saved.manifest_sha256,
            &source.plan,
        )
        .unwrap_err();
        assert!(target.poisoned && !target.complete && target.data.is_empty());
        assert_eq!(
            target
                .trace
                .borrow()
                .iter()
                .filter(|v| **v == "restore abort")
                .count(),
            1
        );
    }
}

#[test]
fn destination_abort_failure_does_not_hide_primary_error() {
    let (source, mut store, saved) = saved();
    let bytes = store.manifest.clone().unwrap();
    let mut target = Target {
        fail: Some("restore write"),
        abort_fails: true,
        ..Target::default()
    };
    let err = restore(
        &mut store,
        &mut target,
        &bytes,
        &saved.manifest_sha256,
        &source.plan,
    )
    .unwrap_err();
    assert_eq!(err.phase, "restore write");
    assert_eq!(err.abort_errors.len(), 1);
    assert!(err.to_string().contains("post-write failure"));
    assert!(err.to_string().contains("destination abort failed"));
}

#[test]
fn late_store_verification_failure_discards_the_destination() {
    let (source, mut store, saved) = saved();
    store.fail = Some("store final verification");
    let bytes = store.manifest.clone().unwrap();
    let mut target = Target::default();
    let error = restore(
        &mut store,
        &mut target,
        &bytes,
        &saved.manifest_sha256,
        &source.plan,
    )
    .unwrap_err();
    assert_eq!(error.phase, "store final verification");
    assert_eq!(target.writes, saved.chunks);
    assert!(target.poisoned && !target.complete);
    assert!(!target.trace.borrow().contains(&"restore finish"));
}

#[test]
fn diagnostic_restore_preserves_receipt_payload_and_operation_order() {
    let run = |timed| {
        let (source, mut store, saved) = saved();
        let bytes = store.manifest.clone().unwrap();
        store.trace.borrow_mut().clear();
        let mut target = Target { trace: store.trace.clone(), ..Target::default() };
        let (receipt, timing) = if timed {
            let (r,t) = restore_with_timing(&mut store,&mut target,&bytes,&saved.manifest_sha256,&source.plan).unwrap();
            (r,Some(t))
        } else { (restore(&mut store,&mut target,&bytes,&saved.manifest_sha256,&source.plan).unwrap(),None) };
        assert!(target.complete && !target.poisoned);
        assert_eq!(target.data,source.data);
        let trace=store.trace.borrow().clone();
        (receipt,target.data,trace,timing)
    };
    let (a,data_a,trace_a,_) = run(false);
    let (b,data_b,trace_b,timing) = run(true);
    assert_eq!(a,b);assert_eq!(data_a,data_b);assert_eq!(trace_a,trace_b);
    let t=timing.unwrap();
    let sum=t.preflight_s+t.staging_allocation_s+t.destination_begin_s+t.chunk_read_s+t.chunk_verify_s
        +t.destination_write_s+t.namespace_verification_s+t.destination_finish_s;
    assert!(sum.is_finite() && sum>=0.0 && t.complete_s>=sum);
}

#[test]
fn diagnostic_restore_keeps_preflight_corruption_and_abort_boundaries() {
    for case in ["manifest","chunk","write","namespace","finish"] {
        let mut outcomes=Vec::new();
        for timed in [false,true] {
            let (source,mut store,saved)=saved();let bytes=store.manifest.clone().unwrap();
            let mut target=Target::default();
            let hash=if case=="manifest" {"0".repeat(64)} else {saved.manifest_sha256.clone()};
            if case=="chunk" {for payload in store.chunks.values_mut(){payload[0]^=1;}}
            if case=="write" {target.fail=Some("restore write");}
            if case=="namespace" {store.fail=Some("store final verification");}
            if case=="finish" {target.fail=Some("restore finish");}
            let result=if timed {restore_with_timing(&mut store,&mut target,&bytes,&hash,&source.plan).map(|(r,_)|r)}
                else {restore(&mut store,&mut target,&bytes,&hash,&source.plan)};
            let error=result.unwrap_err();assert!(!target.complete);
            outcomes.push((error.phase,error.abort_errors,target.writes,target.poisoned,target.trace.borrow().clone()));
        }
        assert_eq!(outcomes[0],outcomes[1],"{case}");
    }
}

#[test]
fn parallel_restore_verifies_every_chunk_and_preserves_payload_with_bounded_buffers() {
    for workers in [2,4,8] {
        let (source,mut store,saved)=saved();let bytes=store.manifest.clone().unwrap();let mut target=Target::default();
        let got=restore_parallel_verified(&mut store,&mut target,&bytes,&saved.manifest_sha256,&source.plan,workers).unwrap();
        assert_eq!(got.manifest_sha256,saved.manifest_sha256);assert_eq!(got.payload_bytes,saved.payload_bytes);
        assert_eq!(got.components,saved.components);assert_eq!(got.chunks,saved.chunks);
        assert_eq!(got.staging_bytes,workers.min(got.chunks)*saved.staging_bytes);
        assert_eq!(target.data,source.data);assert_eq!(target.writes,got.chunks);
        assert!(target.complete && !target.poisoned);
    }
}

#[test]
fn parallel_restore_rejects_corruption_and_drains_all_failure_paths() {
    for workers in [2,4,8] {
        for case in ["manifest","missing","chunk","tail","begin","write","namespace","finish"] {
            let (source,mut store,saved)=saved();let bytes=store.manifest.clone().unwrap();let mut target=Target::default();
            let hash=if case=="manifest" {"0".repeat(64)} else {saved.manifest_sha256.clone()};
            if case=="missing" {store.chunks.clear();}
            if case=="chunk" {for payload in store.chunks.values_mut(){payload[0]^=1;}}
            if case=="tail" {for payload in store.chunks.values_mut(){payload.push(0);}}
            if case=="begin" {target.fail=Some("restore begin");}
            if case=="write" {target.fail=Some("restore write");}
            if case=="namespace" {store.fail=Some("store final verification");}
            if case=="finish" {target.fail=Some("restore finish");}
            let error=restore_parallel_verified(&mut store,&mut target,&bytes,&hash,&source.plan,workers).unwrap_err();
            assert!(!target.complete,"{workers}/{case}");
            if case=="manifest" {assert_eq!(error.phase,"restore preflight");assert!(target.trace.borrow().is_empty());}
            else {assert!(target.poisoned,"{workers}/{case}");assert_eq!(target.trace.borrow().last(),Some(&"restore abort"));}
        }
    }
    for workers in [0,1,9,usize::MAX] {
        let (source,mut store,saved)=saved();let bytes=store.manifest.clone().unwrap();let mut target=Target::default();
        assert_eq!(restore_parallel_verified(&mut store,&mut target,&bytes,&saved.manifest_sha256,&source.plan,workers).unwrap_err().phase,"restore preflight");
        assert!(target.trace.borrow().is_empty());
    }
}

#[test]
fn parallel_restore_late_missing_or_corrupt_chunk_drains_inflight_workers() {
    for workers in [2,4,8] {
        for missing in [false,true] {
            let (source,mut store,saved)=saved();let raw=store.manifest.clone().unwrap();
            let manifest=Manifest::decode(&raw,&saved.manifest_sha256,&source.plan).unwrap();
            // The second block fails after the first verification job is sent.
            let digest=&manifest.chunks[1].sha256;
            if missing {store.chunks.remove(digest);} else {store.chunks.get_mut(digest).unwrap()[0]^=1;}
            let mut target=Target::default();
            let err=restore_parallel_verified(&mut store,&mut target,&raw,&saved.manifest_sha256,&source.plan,workers).unwrap_err();
            assert_eq!(err.phase,if missing {"chunk read"} else {"chunk verify"});
            assert!(target.poisoned && !target.complete);assert_eq!(target.trace.borrow().last(),Some(&"restore abort"));
        }
    }
}

#[test]
fn pipeline_large_multiwindow_restore_matches_baseline_with_bounded_ordered_slots() {
    #[derive(Default)]
    struct Counts { outstanding: usize, peak: usize, written: usize }
    struct ReadAudit { store: Store, counts: Rc<RefCell<Counts>>, width: usize }
    impl CheckpointReader for ReadAudit {
        type ChunkReader = Cursor<Vec<u8>>;
        fn open_chunk(&mut self, digest: &str) -> io::Result<Self::ChunkReader> {
            let mut counts=self.counts.borrow_mut();
            counts.outstanding+=1;counts.peak=counts.peak.max(counts.outstanding);
            assert!(counts.outstanding<=self.width,"read-ahead exceeded staging budget");
            self.store.open_chunk(digest)
        }
        fn finish_read(&mut self)->io::Result<()> {self.store.finish_read()}
    }
    struct WriteAudit { target: Target, counts: Rc<RefCell<Counts>>, expected: Vec<(usize,u64,usize)> }
    impl RestoreTarget for WriteAudit {
        fn begin(&mut self,p:&TransferPlan)->io::Result<()> {self.target.begin(p)}
        fn write_exact(&mut self,c:usize,offset:u64,bytes:&[u8])->io::Result<()> {
            let mut counts=self.counts.borrow_mut();
            assert_eq!(self.expected[counts.written],(c,offset,bytes.len()));
            assert!(counts.outstanding>0);counts.outstanding-=1;counts.written+=1;
            self.target.write_exact(c,offset,bytes)
        }
        fn finish_unpublished(&mut self)->io::Result<()> {
            assert_eq!(self.counts.borrow().outstanding,0);
            self.target.finish_unpublished()
        }
        fn abort(&mut self)->io::Result<()> {self.target.abort()}
    }
    for workers in [2,4,8] {
        let expected=TransferPlan { binding_sha256:sha256(b"multi-window actual SHA fixture"),
            components:vec![Component{name:"fixture.kv".into(),bytes:131089},
                Component{name:"fixture.gdn".into(),bytes:51314},
                Component{name:"fixture.logits".into(),bytes:65535}] };
        let trace=Trace::default();let mut source=Source::new(expected.clone(),trace.clone());
        let mut store=Store {trace,..Store::default()};
        let saved=save(&mut source,&mut store,&expected,4096).unwrap();
        let raw=store.manifest.clone().unwrap();
        let manifest=Manifest::decode(&raw,&saved.manifest_sha256,&expected).unwrap();
        assert!(manifest.chunks.len()>workers*4);
        let mut baseline=Target::default();
        let base=restore_parallel_verified(&mut store,&mut baseline,&raw,&saved.manifest_sha256,&expected,workers).unwrap();
        let counts=Rc::new(RefCell::new(Counts::default()));
        let mut reader=ReadAudit{store,counts:counts.clone(),width:workers};
        let mut target=WriteAudit{target:Target::default(),counts:counts.clone(),
            expected:manifest.chunks.iter().map(|c|(c.component,c.offset,c.bytes)).collect()};
        let got=restore_pipelined_verified(&mut reader,&mut target,&raw,&saved.manifest_sha256,&expected,workers).unwrap();
        assert_eq!(target.target.data,source.data);assert_eq!(target.target.data,baseline.data);
        assert_eq!(got.manifest_sha256,base.manifest_sha256);assert_eq!(got.payload_bytes,base.payload_bytes);
        assert_eq!(got.staging_bytes,base.staging_bytes);assert_eq!(got.staging_bytes,workers*4096);
        assert_eq!(counts.borrow().peak,workers);assert_eq!(counts.borrow().written,got.chunks);
        assert!(target.target.complete && !target.target.poisoned);
    }
}

#[test]
fn pipeline_late_failures_across_refill_boundaries_never_publish() {
    for workers in [2,4,8] {
        for broken in [0,7,8,15,16,32] {
            let expected=TransferPlan{binding_sha256:sha256(b"late refill corruption"),
                components:vec![Component{name:"fixture.kv".into(),bytes:4096*33}]};
            let trace=Trace::default();let mut source=Source::new(expected.clone(),trace.clone());
            let mut store=Store{trace,..Store::default()};
            let saved=save(&mut source,&mut store,&expected,4096).unwrap();
            let raw=store.manifest.clone().unwrap();
            let manifest=Manifest::decode(&raw,&saved.manifest_sha256,&expected).unwrap();
            store.chunks.get_mut(&manifest.chunks[broken].sha256).unwrap()[0]^=1;
            let mut target=Target::default();
            let error=restore_pipelined_verified(&mut store,&mut target,&raw,&saved.manifest_sha256,&expected,workers).unwrap_err();
            assert_eq!(error.phase,"chunk verify");assert!(!target.complete && target.poisoned);
            assert!(target.data.is_empty());assert_eq!(target.trace.borrow().last(),Some(&"restore abort"));
        }
    }
}

#[test]
fn pipeline_postwrite_failure_preserves_abort_error() {
    let (source,mut store,saved)=saved();let raw=store.manifest.clone().unwrap();
    let mut target=Target{fail:Some("restore write"),abort_fails:true,..Target::default()};
    let error=restore_pipelined_verified(&mut store,&mut target,&raw,&saved.manifest_sha256,&source.plan,8).unwrap_err();
    assert_eq!(error.phase,"restore write");assert_eq!(error.abort_errors.len(),1);
    assert!(target.poisoned && !target.complete && target.data.is_empty());
}
