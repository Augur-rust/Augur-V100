use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT: AtomicUsize = AtomicUsize::new(1);

fn boundary() -> Boundary {
    Boundary {
        model: "a".repeat(64),
        config: "b".repeat(64),
        numerical_identity: "c".repeat(64),
        prompt_ids: "d".repeat(64),
        committed_input_ids: "e".repeat(64),
        cursor: 5639,
        anchor: 42,
        context_start: 3591,
        context_rows: HISTORY,
    }
}

fn output() -> std::path::PathBuf {
    let base = std::env::var_os("AUGUR_CAPTURE_TEST_OUTPUT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    base.join(format!(
        "augur-state-capture-test-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::SeqCst)
    ))
}

struct Mock {
    snap: Snapshot,
    reads: usize,
    drains: usize,
    quiesced: bool,
    fail_read: bool,
    fail_quiesce: bool,
    fail_drain: bool,
    drift: bool,
    inject: Option<std::path::PathBuf>,
    pointers: Vec<usize>,
}

fn source() -> Mock {
    Mock {
        snap: Snapshot {
            boundary: boundary(),
            owner: "fixture-owner-1".into(),
            generation: 9,
            backend: "cpu_fixture_not_cuda".into(),
            planes: vec![
                plane("fixture_key".into(), Scope::TargetState, 44, 28),
                plane("fixture_state".into(), Scope::TargetState, 20, 20),
            ],
        },
        reads: 0,
        drains: 0,
        quiesced: false,
        fail_read: false,
        fail_quiesce: false,
        fail_drain: false,
        drift: false,
        inject: None,
        pointers: Vec::new(),
    }
}

fn budget() -> Budget {
    Budget {
        chunk_bytes: 16,
        max_total_bytes: 64,
        max_chunks: 5,
    }
}

impl ReadbackSource for Mock {
    fn quiesce(&mut self) -> io::Result<()> {
        if self.fail_quiesce {
            return Err(invalid("mock quiesce failure"));
        }
        self.quiesced = true;
        Ok(())
    }
    fn snapshot(&self) -> io::Result<Snapshot> {
        require(self.quiesced, "not quiesced")?;
        Ok(self.snap.clone())
    }
    fn read_completed(&mut self, plane: usize, offset: usize, out: &mut [u8]) -> io::Result<()> {
        require(self.quiesced, "read ordering")?;
        self.reads += 1;
        self.pointers.push(out.as_ptr() as usize);
        if self.fail_read {
            return Err(invalid("mock submission proof failure"));
        }
        for (index, byte) in out.iter_mut().enumerate() {
            *byte = ((plane * 47 + offset + index) % 256) as u8;
        }
        if self.drift {
            self.snap.generation += 1;
        }
        if let Some(path) = self.inject.take() {
            fs::write(path, b"undeclared")?;
        }
        Ok(())
    }
    fn drain(&mut self) -> io::Result<()> {
        self.drains += 1;
        if self.fail_drain {
            Err(invalid("mock drain failure"))
        } else {
            Ok(())
        }
    }
}

#[test]
fn all_target_planes_and_extents() {
    let inventory = target_inventory(5639).unwrap();
    assert_eq!(inventory.len(), 195);
    assert_eq!(
        inventory
            .iter()
            .filter(|p| p.scope == Scope::TargetState)
            .count(),
        128
    );
    assert_eq!(
        inventory[..128]
            .iter()
            .map(|p| p.physical_bytes)
            .sum::<usize>(),
        7001997312
    );
    assert_eq!(
        inventory[..128]
            .iter()
            .map(|p| p.committed_bytes)
            .sum::<usize>(),
        353220608
    );
    assert_eq!(inventory[128].committed_bytes, 209715200);
}

#[test]
fn capacity_future_empty_only_at_full_cursor() {
    assert!(target_inventory(CAPACITY)
        .unwrap()
        .iter()
        .all(|p| p.physical_bytes == p.committed_bytes));
    assert!(target_inventory(5639)
        .unwrap()
        .iter()
        .any(|p| p.physical_bytes > p.committed_bytes));
}

#[test]
fn invalid_cursor_rejected() {
    for n in [0, HISTORY - 1, CAPACITY + 1, usize::MAX] {
        assert!(target_inventory(n).is_err());
    }
}

#[test]
fn exact_boundary_match() {
    boundary().matches(&boundary()).unwrap();
}

#[test]
fn ledger_mismatch_rejected() {
    let mut other = boundary();
    other.committed_input_ids = "f".repeat(64);
    assert!(boundary().matches(&other).is_err());
}

#[test]
fn short_history_rejected() {
    let mut b = boundary();
    b.context_rows = 1024;
    assert!(b.validate().is_err());
}

#[test]
fn feature_position_mismatch_rejected() {
    let mut b = boundary();
    b.context_start += 1;
    assert!(b.validate().is_err());
}

#[test]
fn uppercase_digest_rejected() {
    let mut b = boundary();
    b.model = "A".repeat(64);
    assert!(b.validate().is_err());
}

#[test]
fn anchor_outside_vocabulary_rejected() {
    let mut b = boundary();
    b.anchor = 248320;
    assert!(b.validate().is_err());
}

#[test]
fn unknown_boundary_field_rejected() {
    let mut value = serde_json::to_value(boundary()).unwrap();
    value["authority"] = true.into();
    assert!(serde_json::from_value::<Boundary>(value).is_err());
}

#[test]
fn bool_cursor_rejected() {
    let mut value = serde_json::to_value(boundary()).unwrap();
    value["cursor"] = true.into();
    assert!(serde_json::from_value::<Boundary>(value).is_err());
}

#[test]
fn bounded_readback_and_exact_bytes() {
    let mut src = source();
    let out = output();
    let r = capture(&mut src, &out, &budget()).unwrap();
    assert_eq!(r.total_bytes, 64);
    assert_eq!(r.chunks.len(), 5);
    assert_eq!(src.drains, 1);
    assert!(src.pointers.iter().all(|p| *p == src.pointers[0]));
    assert_eq!(r.host_staging_bytes, 16);
    assert!(out.join("complete.json").exists());
    assert!(!out.join("receipt.pending").exists());
    assert!(
        !r.target_inventory_complete
            && !r.model_state_equivalence_qualified
            && !r.product_tps_permitted
    );
    for c in &r.chunks {
        let raw = fs::read(out.join(&c.file)).unwrap();
        assert_eq!(raw.len(), c.bytes);
        assert_eq!(format!("{:x}", Sha256::digest(&raw)), c.sha256);
        for (i, b) in raw.iter().enumerate() {
            assert_eq!(*b, ((c.plane * 47 + c.offset + i) % 256) as u8);
        }
    }
}

#[test]
fn committed_and_future_split_never_crosses_boundary() {
    let r = capture(&mut source(), &output(), &budget()).unwrap();
    for c in &r.chunks {
        let p = &r.snapshot.planes[c.plane];
        if c.region == "committed" {
            assert!(c.offset + c.bytes <= p.committed_bytes);
        } else {
            assert_eq!(c.region, "future_capacity");
            assert!(c.offset >= p.committed_bytes);
        }
    }
}

#[test]
fn staging_limit_rejected_before_read() {
    for size in [0, 3, MAX_CHUNK + 4] {
        let mut src = source();
        let mut b = budget();
        b.chunk_bytes = size;
        let out = output();
        assert!(capture(&mut src, &out, &b).is_err());
        assert_eq!(src.reads, 0);
        assert!(!out.exists());
    }
}

#[test]
fn byte_budget_rejected() {
    let mut src = source();
    let mut b = budget();
    b.max_total_bytes = 63;
    assert!(capture(&mut src, &output(), &b).is_err());
    assert_eq!(src.reads, 0);
}

#[test]
fn chunk_budget_counts_region_split() {
    let mut src = source();
    let mut b = budget();
    b.max_chunks = 4;
    assert!(capture(&mut src, &output(), &b).is_err());
    assert_eq!(src.reads, 0);
}

#[test]
fn source_drift_is_not_published() {
    let mut src = source();
    src.drift = true;
    let out = output();
    assert!(capture(&mut src, &out, &budget()).is_err());
    assert_eq!(src.drains, 1);
    assert!(!out.join("complete.json").exists());
}

#[test]
fn invalid_submission_proof_drains_without_publication() {
    let mut src = source();
    src.fail_read = true;
    let out = output();
    assert!(capture(&mut src, &out, &budget()).is_err());
    assert_eq!(src.drains, 1);
    assert!(!out.join("complete.json").exists());
}

#[test]
fn failed_quiescence_never_reads() {
    let mut src = source();
    src.fail_quiesce = true;
    let out = output();
    assert!(capture(&mut src, &out, &budget()).is_err());
    assert_eq!(src.reads, 0);
    assert_eq!(src.drains, 1);
}

#[test]
fn failed_drain_not_hidden() {
    let mut src = source();
    src.fail_drain = true;
    let out = output();
    let error = capture(&mut src, &out, &budget()).unwrap_err();
    assert!(error.to_string().contains("drain failed"));
    assert!(!out.join("complete.json").exists());
}

#[test]
fn undeclared_file_rejected() {
    let mut src = source();
    let out = output();
    src.inject = Some(out.join("authority.json"));
    assert!(capture(&mut src, &out, &budget()).is_err());
    assert!(!out.join("complete.json").exists());
}

#[test]
fn existing_output_never_overwritten() {
    let out = output();
    fs::create_dir(&out).unwrap();
    fs::write(out.join("sentinel"), b"keep").unwrap();
    assert!(capture(&mut source(), &out, &budget()).is_err());
    assert_eq!(fs::read(out.join("sentinel")).unwrap(), b"keep");
}

#[test]
fn duplicate_plane_rejected() {
    let mut src = source();
    src.snap.planes[1].name = src.snap.planes[0].name.clone();
    assert!(capture(&mut src, &output(), &budget()).is_err());
    assert_eq!(src.reads, 0);
}

#[test]
fn path_like_plane_rejected() {
    let mut src = source();
    src.snap.planes[0].name = "../escape".into();
    assert!(capture(&mut src, &output(), &budget()).is_err());
}

#[test]
fn extent_overflow_rejected() {
    let mut src = source();
    src.snap.planes[0].committed_bytes = 48;
    assert!(capture(&mut src, &output(), &budget()).is_err());
}

#[test]
fn relative_output_rejected() {
    assert!(capture(&mut source(), Path::new("relative-output"), &budget()).is_err());
}

#[test]
fn changed_generation_does_not_mean_different_token_boundary() {
    let a = source();
    let mut b = source();
    b.snap.generation += 1;
    assert_ne!(a.snap, b.snap);
    a.snap.boundary.matches(&b.snap.boundary).unwrap();
}

#[test]
fn namespace_rejects_nested_directory() {
    let out = output();
    fs::create_dir(&out).unwrap();
    fs::create_dir(out.join("unexpected")).unwrap();
    assert!(namespace(&out, &BTreeSet::new()).is_err());
}

#[cfg(windows)]
#[test]
fn nested_symlink_rejected() {
    let root = output();
    fs::create_dir(&root).unwrap();
    let real = root.join("real");
    fs::create_dir(&real).unwrap();
    let alias = root.join("alias");
    std::os::windows::fs::symlink_dir(&real, &alias).unwrap();
    assert!(capture(&mut source(), &alias.join("capture"), &budget()).is_err());
    assert!(!real.join("capture").exists());
}

fn mutate_receipt(out: &Path, change: impl FnOnce(&mut serde_json::Value)) {
    let path = out.join("complete.json");
    let mut value: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    change(&mut value);
    fs::write(path, serde_json::to_vec(&value).unwrap()).unwrap();
}

#[test]
fn independently_verify_all_stored_bytes() {
    let out = output();
    capture(&mut source(), &out, &budget()).unwrap();
    let r = verify_capture(&out, &budget()).unwrap();
    assert_eq!(r.total_bytes, 64);
}

#[test]
fn payload_corruption_rejected() {
    let out = output();
    capture(&mut source(), &out, &budget()).unwrap();
    fs::write(out.join("chunk00000000.bin"), [255_u8; 16]).unwrap();
    assert!(verify_capture(&out, &budget()).is_err());
}

#[test]
fn payload_truncation_rejected() {
    let out = output();
    capture(&mut source(), &out, &budget()).unwrap();
    fs::write(out.join("chunk00000000.bin"), [0_u8; 4]).unwrap();
    assert!(verify_capture(&out, &budget()).is_err());
}

#[test]
fn missing_chunk_rejected() {
    let out = output();
    capture(&mut source(), &out, &budget()).unwrap();
    fs::remove_file(out.join("chunk00000000.bin")).unwrap();
    assert!(verify_capture(&out, &budget()).is_err());
}

#[test]
fn unknown_receipt_authority_rejected() {
    let out = output();
    capture(&mut source(), &out, &budget()).unwrap();
    mutate_receipt(&out, |v| v["extra_authority"] = true.into());
    assert!(verify_capture(&out, &budget()).is_err());
}

#[test]
fn true_qualification_rejected() {
    for key in [
        "draft_private_state_qualified",
        "model_state_equivalence_qualified",
        "product_tps_permitted",
        "target_inventory_complete",
    ] {
        let out = output();
        capture(&mut source(), &out, &budget()).unwrap();
        mutate_receipt(&out, |v| v[key] = true.into());
        assert!(verify_capture(&out, &budget()).is_err(), "{key}");
    }
}

#[test]
fn chunk_offset_drift_rejected() {
    let out = output();
    capture(&mut source(), &out, &budget()).unwrap();
    mutate_receipt(&out, |v| v["chunks"][1]["offset"] = 12.into());
    assert!(verify_capture(&out, &budget()).is_err());
}

#[test]
fn future_capacity_cannot_be_relabeled_committed() {
    let out = output();
    capture(&mut source(), &out, &budget()).unwrap();
    mutate_receipt(&out, |v| v["chunks"][2]["region"] = "committed".into());
    assert!(verify_capture(&out, &budget()).is_err());
}

#[test]
fn retained_partial_files_have_no_success_receipt() {
    let mut src = source();
    let out = output();
    src.inject = Some(out.join("surplus"));
    assert!(capture(&mut src, &out, &budget()).is_err());
    assert!(out.join("chunk00000000.bin").exists());
    assert!(verify_capture(&out, &budget()).is_err());
}

#[test]
fn receipt_publication_collision_does_not_overwrite() {
    let mut src = source();
    let out = output();
    src.inject = Some(out.join("complete.json"));
    assert!(capture(&mut src, &out, &budget()).is_err());
    assert_eq!(fs::read(out.join("complete.json")).unwrap(), b"undeclared");
    assert!(verify_capture(&out, &budget()).is_err());
}
