use super::*;
use crate::session_checkpoint::http::tests::{fixture, native};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(1);

fn root() -> PathBuf {
    let base = std::env::var_os("AUGUR_CHECKPOINT_TEST_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let root = base.join(format!(
        "qwen-http-checkpoint-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::SeqCst)
    ));
    fs::create_dir(&root).unwrap();
    root
}

fn budget() -> StoreBudget {
    StoreBudget {
        max_checkpoints: 2,
        max_reserved_bytes: 2 * (MAX_PAYLOAD_BYTES + OVERHEAD),
    }
}

#[test]
fn startup_profile_and_template_are_hash_bound_and_pinned() {
    let base = root();
    let template = base.join("chat.jinja");
    fs::write(&template, b"canonical template fixture").unwrap();
    let mut bindings = native().bindings;
    bindings.chat_template_sha256 = sha256(b"canonical template fixture");
    let math = native().math;
    bindings.numerical_contract_sha256 = sha256(&serde_json::to_vec(&math).unwrap());
    let profile = HttpCheckpointProfile {
        schema: "cq-qwen35-http-checkpoint-profile-v1".into(),
        store_root: base.clone(),
        chat_template: template.clone(),
        budget: budget(),
        bindings,
        math,
    };
    let raw = serde_json::to_vec(&profile).unwrap();
    let path = base.join("profile.json");
    fs::write(&path, &raw).unwrap();
    assert!(PinnedHttpCheckpointProfile::open(&path, &sha256(b"wrong")).is_err());
    let pinned = PinnedHttpCheckpointProfile::open(&path, &sha256(&raw)).unwrap();
    assert!(OpenOptions::new().write(true).open(&path).is_err());
    assert!(OpenOptions::new().write(true).open(&template).is_err());
    assert_eq!(
        pinned.profile.bindings.chat_template_sha256,
        profile.bindings.chat_template_sha256
    );
    drop(pinned);
    for (key, value) in [
        ("schema", serde_json::json!("unknown")),
        ("authority", serde_json::json!(true)),
        ("store_root", serde_json::json!("relative-store")),
    ] {
        let mut v = serde_json::to_value(&profile).unwrap();
        v[key] = value;
        let bytes = serde_json::to_vec(&v).unwrap();
        fs::write(&path, &bytes).unwrap();
        assert!(
            PinnedHttpCheckpointProfile::open(&path, &sha256(&bytes)).is_err(),
            "{key}"
        );
    }
    let mut p = profile;
    p.bindings.numerical_contract_sha256 = sha256(b"wrong-math");
    let bytes = serde_json::to_vec(&p).unwrap();
    fs::write(&path, &bytes).unwrap();
    assert!(PinnedHttpCheckpointProfile::open(&path, &sha256(&bytes)).is_err());
}

#[test]
fn exclusive_store_owner_survives_until_all_slot_leases_drop() {
    let root = root();
    let mut store = HttpCheckpointStore::open(&root, budget()).unwrap();
    assert!(HttpCheckpointStore::open(&root, budget()).is_err());
    let slot = store.reserve(&sha256(b"one"), 1).unwrap();
    drop(store);
    assert!(HttpCheckpointStore::open(&root, budget()).is_err());
    drop(slot);
    HttpCheckpointStore::open(&root, budget()).unwrap();
}

#[test]
fn count_and_byte_backpressure_happen_before_directory_creation() {
    for b in [
        StoreBudget {
            max_checkpoints: 1,
            max_reserved_bytes: MAX_PAYLOAD_BYTES + OVERHEAD,
        },
        StoreBudget {
            max_checkpoints: 2,
            max_reserved_bytes: OVERHEAD + 10,
        },
    ] {
        let root = root();
        let mut store = HttpCheckpointStore::open(&root, b).unwrap();
        let first = store.reserve(&sha256(b"first"), 10).unwrap();
        drop(first);
        assert!(store.reserve(&sha256(b"second"), 1).is_err());
        assert!(!root.join(sha256(b"second")).exists());
    }
}

#[test]
fn failed_or_partial_slots_remain_charged_across_reopen() {
    let root = root();
    let b = StoreBudget {
        max_checkpoints: 1,
        max_reserved_bytes: MAX_PAYLOAD_BYTES + OVERHEAD,
    };
    let mut store = HttpCheckpointStore::open(&root, b).unwrap();
    let id = sha256(b"interrupted");
    drop(store.reserve(&id, 100).unwrap());
    drop(store);
    let mut store = HttpCheckpointStore::open(&root, b).unwrap();
    assert_eq!(store.usage().unwrap(), (1, 100 + OVERHEAD));
    assert!(store.reserve(&sha256(b"next"), 1).is_err());
    assert!(store
        .read(&id, &sha256(b"absent"), &native().bindings, &native().math)
        .is_err());
}

#[test]
fn disk_backpressure_has_no_partial_slot_and_healthy_retry_works() {
    let root = root();
    let mut store = HttpCheckpointStore::open(&root, budget()).unwrap();
    let id = sha256(b"disk-full");
    let error = store
        .reserve_with_available(&id, 100, |_| Ok(100 + OVERHEAD - 1))
        .err()
        .unwrap();
    assert_eq!(error.kind(), io::ErrorKind::StorageFull);
    assert!(!root.join(&id).exists());
    assert_eq!(store.usage().unwrap(), (0, 0));
    let slot = store.reserve(&id, 100).unwrap();
    assert_eq!(store.usage().unwrap(), (1, 100 + OVERHEAD));
    drop(slot);
}

#[test]
fn failed_disk_query_does_not_reserve_or_capture() {
    let root = root();
    let mut store = HttpCheckpointStore::open(&root, budget()).unwrap();
    let id = sha256(b"disk-query-failed");
    assert!(store
        .reserve_with_available(&id, 100, |_| {
            Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "injected disk query failure",
            ))
        })
        .is_err());
    assert!(!root.join(id).exists());
    assert_eq!(store.usage().unwrap(), (0, 0));
}

#[test]
fn unreserved_interrupted_directory_and_unknown_entries_fail_closed() {
    for name in [sha256(b"unreserved"), "unexpected.txt".into()] {
        let root = root();
        fs::create_dir(root.join(name)).unwrap();
        assert!(HttpCheckpointStore::open(&root, budget()).is_err());
    }
}

#[test]
fn reservation_is_no_clobber_and_invalid_ids_are_not_paths() {
    let root = root();
    let mut store = HttpCheckpointStore::open(&root, budget()).unwrap();
    let id = sha256(b"slot");
    let slot = store.reserve(&id, 100).unwrap();
    let before = fs::read(slot.directory.root.join(RESERVATION)).unwrap();
    assert!(store.reserve(&id, 100).is_err());
    assert_eq!(
        fs::read(slot.directory.root.join(RESERVATION)).unwrap(),
        before
    );
    for id in ["../x", "F:/x", "x:ads", "X", "", "00"] {
        assert!(store.reserve(id, 1).is_err());
    }
    assert!(store.reserve(&sha256(b"overflow"), u64::MAX).is_err());
}

#[test]
fn reparse_store_or_leaf_is_rejected() {
    use std::os::windows::fs::symlink_dir;
    let base = root();
    let target = root();
    let alias = base.join("alias");
    symlink_dir(&target, &alias).unwrap();
    assert!(HttpCheckpointStore::open(&alias, budget()).is_err());
    let base = root();
    let id = sha256(b"alias");
    symlink_dir(&target, base.join(id)).unwrap();
    assert!(HttpCheckpointStore::open(&base, budget()).is_err());
}

#[test]
fn incomplete_publication_does_not_create_outer_seal() {
    let root = root();
    let mut store = HttpCheckpointStore::open(&root, budget()).unwrap();
    let h = fixture();
    let slot = store
        .reserve(
            &h.checkpoint_id,
            native().transfer_plan().unwrap().validate().unwrap(),
        )
        .unwrap();
    assert!(slot.publish(&h).is_err());
    assert!(!root.join(&h.checkpoint_id).join(OUTER_SEAL).exists());
}

struct ZeroSource {
    plan: TransferPlan,
    metadata: Vec<u8>,
}
impl CaptureSource for ZeroSource {
    fn begin(&mut self) -> io::Result<TransferPlan> {
        Ok(self.plan.clone())
    }
    fn read_exact(&mut self, component: usize, offset: u64, output: &mut [u8]) -> io::Result<()> {
        if component == 0 {
            output.copy_from_slice(&self.metadata[offset as usize..offset as usize + output.len()]);
        } else {
            output.fill(0);
        }
        Ok(())
    }
    fn finish(&mut self) -> io::Result<TransferPlan> {
        Ok(self.plan.clone())
    }
    fn abort(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn real_files_publish_last_restore_metadata_and_detect_late_namespace() {
    // Exact native inventory with synthetic zero planes. This is filesystem /
    // metadata qualification only, not model semantics or CUDA restore evidence.
    let root = root();
    let mut store = HttpCheckpointStore::open(&root, budget()).unwrap();
    let descriptor = native();
    let plan = descriptor.transfer_plan().unwrap();
    let mut h = fixture();
    let slot = store
        .reserve(&h.checkpoint_id, plan.validate().unwrap())
        .unwrap();
    let mut source = ZeroSource {
        plan: plan.clone(),
        metadata: descriptor.encode().unwrap(),
    };
    let mut writer = LocalCheckpointWriter::create(&slot.native_path(), &plan).unwrap();
    let receipt = save(&mut source, &mut writer, &plan, MAX_CHUNK_BYTES).unwrap();
    drop(writer);
    assert!(store
        .read(
            &h.checkpoint_id,
            &sha256(b"not-sealed"),
            &h.bindings,
            &h.math
        )
        .is_err());
    h.native_manifest_sha256 = receipt.manifest_sha256;
    // Independently derive the expected seal for our synthetic zero payload.
    use sha2::{Digest, Sha256};
    let zero = [0u8; 65536];
    let mut target = Sha256::new();
    target.update(b"QWEN35-SESSION-TARGET-STATE-V1\0");
    target.update((descriptor.ledger.consumed.len() as u64).to_le_bytes());
    for p in &mut h.target_seal.state_plane_hashes {
        target.update((p.layer as u64).to_le_bytes());
        target.update((p.plane as u64).to_le_bytes());
        target.update(p.bytes.to_le_bytes());
        let mut hash = Sha256::new();
        let mut left = p.bytes;
        while left > 0 {
            let n = left.min(zero.len() as u64) as usize;
            target.update(&zero[..n]);
            hash.update(&zero[..n]);
            left -= n as u64;
        }
        p.sha256 = format!("{:x}", hash.finalize());
    }
    h.target_seal.target_committed_state_sha256 = format!("{:x}", target.finalize());
    h.target_seal.feature_bytes_sha256 = sha256(&vec![0; plan.components[129].bytes as usize]);
    let mut forged = h.target_seal.clone();
    forged.state_plane_hashes[0].sha256 = sha256(b"forged");
    assert!(verify_native_payload(
        &slot.native_path(),
        &h.native_manifest_sha256,
        &plan,
        &forged,
        descriptor.ledger.consumed.len()
    )
    .is_err());
    let sha = slot.publish(&h).unwrap();
    assert!(store
        .read(&h.checkpoint_id, &sha256(b"wrong"), &h.bindings, &h.math)
        .is_err());
    let read = store
        .read(&h.checkpoint_id, &sha, &h.bindings, &h.math)
        .unwrap();
    assert_eq!(read.envelope, h);
    assert_eq!(read.descriptor, descriptor);
    assert!(OpenOptions::new()
        .write(true)
        .open(root.join(&h.checkpoint_id).join(OUTER_SEAL))
        .is_err());
    assert!(fs::rename(root.join(&h.checkpoint_id), root.join("moved")).is_err());
    read.verify_namespace().unwrap();
    fs::write(root.join(&h.checkpoint_id).join("late.tmp"), b"late").unwrap();
    assert!(read.verify_namespace().is_err());
}

#[test]
#[ignore = "requires an explicitly hash-bound archived source receipt; CPU ABI/readback only"]
fn archived_native_seal_matches_typed_abi_and_payload_bytes() {
    use crate::session_checkpoint::http::TargetStateSeal;
    let path =
        PathBuf::from(std::env::var_os("AUGUR_CHECKPOINT_READBACK_SOURCE").expect("source path"));
    let expected =
        std::env::var("AUGUR_CHECKPOINT_READBACK_SOURCE_SHA256").expect("external source hash");
    let parent = DirectoryLease::acquire(path.parent().unwrap()).unwrap();
    let mut file = parent
        .read_file(path.file_name().unwrap().to_str().unwrap(), 8 * 1024 * 1024)
        .unwrap();
    let raw = bounded_bytes(&mut file, 8 * 1024 * 1024).unwrap();
    assert_eq!(sha256(&raw), expected);
    let source: serde_json::Value = serde_json::from_slice(&raw).unwrap();
    for saved in [&source, &source["ready"]] {
        let root = Path::new(saved["snapshot"].as_str().unwrap());
        let manifest = saved["manifest_sha256"].as_str().unwrap();
        let raw = LocalCheckpointReader::read_session_descriptor(root, manifest).unwrap();
        let descriptor: SessionDescriptor = serde_json::from_slice(&raw).unwrap();
        descriptor.validate().unwrap();
        assert_eq!(raw, descriptor.encode().unwrap());
        assert_eq!(
            serde_json::to_value(&descriptor.bindings).unwrap(),
            source["bindings"]
        );
        let seal = TargetStateSeal::from_checkpoint(&saved["checkpoint"], &descriptor).unwrap();
        verify_native_payload(
            root,
            manifest,
            &descriptor.transfer_plan().unwrap(),
            &seal,
            descriptor.ledger.consumed.len(),
        )
        .unwrap();
        println!(
            "CPU_NATIVE_SEAL_ABI_PAYLOAD_PASS cursor={} planes={} product_ready=false",
            descriptor.ledger.consumed.len(),
            seal.state_plane_hashes.len()
        );
    }
}

#[test]
fn optional_capture_capacity_check_is_nonmutating_and_matches_reservation() {
    for b in [
        StoreBudget { max_checkpoints: 1, max_reserved_bytes: MAX_PAYLOAD_BYTES + OVERHEAD },
        StoreBudget { max_checkpoints: 2, max_reserved_bytes: OVERHEAD + 10 },
    ] {
        let root = root();
        let mut store = HttpCheckpointStore::open(&root, b).unwrap();
        assert!(store.has_capacity_for(10).unwrap());
        assert_eq!(store.usage().unwrap(), (0, 0));
        assert!(store.has_capacity_for(0).is_err());
        assert!(store.has_capacity_for(u64::MAX).is_err());
        assert_eq!(store.usage().unwrap(), (0, 0));
        drop(store.reserve(&sha256(b"optional-first"), 10).unwrap());
        assert!(!store.has_capacity_for(1).unwrap());
        assert_eq!(store.usage().unwrap(), (1, OVERHEAD + 10));
        assert!(store.reserve(&sha256(b"optional-second"), 1).is_err());
        assert!(!root.join(sha256(b"optional-second")).exists());
        drop(store);
        let reopened = HttpCheckpointStore::open(&root, b).unwrap();
        assert!(!reopened.has_capacity_for(1).unwrap());
    }
}

#[test]
fn optional_capture_capacity_check_preserves_corruption_failure() {
    let root = root();
    let store = HttpCheckpointStore::open(&root, budget()).unwrap();
    fs::write(root.join("unexpected.txt"), b"not a reservation").unwrap();
    assert!(store.has_capacity_for(1).is_err());
}
