use super::*;
use crate::session_checkpoint::filesystem::{LocalCheckpointReader, LocalCheckpointWriter};
use std::fs::{self, OpenOptions};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(1);

fn fresh_path() -> PathBuf {
    let base = std::env::var_os("AUGUR_CHECKPOINT_TEST_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    assert!(base.is_absolute() && base.is_dir());
    base.join(format!(
        "qwen-checkpoint-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::SeqCst)
    ))
}

fn disk_saved() -> (PathBuf, Source, TransferReceipt) {
    let root = fresh_path();
    let expected = plan();
    let mut source = Source::new(expected.clone(), Trace::default());
    let mut writer = LocalCheckpointWriter::create(&root, &expected).unwrap();
    let receipt = save(&mut source, &mut writer, &expected, 7).unwrap();
    drop(writer);
    (root, source, receipt)
}

#[test]
fn actual_files_roundtrip_into_unpublished_target() {
    let (root, source, receipt) = disk_saved();
    let mut reader =
        LocalCheckpointReader::open(&root, &receipt.manifest_sha256, &source.plan).unwrap();
    let bytes = reader.manifest_bytes().to_vec();
    let mut target = Target::default();
    assert_eq!(
        restore(
            &mut reader,
            &mut target,
            &bytes,
            &receipt.manifest_sha256,
            &source.plan
        )
        .unwrap(),
        receipt
    );
    assert_eq!(target.data, source.data);
    assert_eq!(fs::read_dir(&root).unwrap().count(), receipt.chunks + 2);
    assert_eq!(fs::read(root.join("manifest.blob")).unwrap(), bytes);
    assert_eq!(fs::read(root.join("manifest.json")).unwrap(), bytes);
}

#[test]
fn checkpoint_root_and_manifest_never_overwrite_prior_seal() {
    let (root, source, receipt) = disk_saved();
    let before = fs::read(root.join("manifest.json")).unwrap();
    assert!(LocalCheckpointWriter::create(&root, &source.plan).is_err());
    assert_eq!(
        sha256(&fs::read(root.join("manifest.json")).unwrap()),
        receipt.manifest_sha256
    );
    assert_eq!(fs::read(root.join("manifest.json")).unwrap(), before);
}

#[test]
fn incomplete_capture_has_no_restorable_manifest_and_writer_stays_aborted() {
    let root = fresh_path();
    let expected = plan();
    let mut source = Source::new(expected.clone(), Trace::default());
    source.fail = Some("capture finish");
    let mut writer = LocalCheckpointWriter::create(&root, &expected).unwrap();
    assert!(save(&mut source, &mut writer, &expected, 7).is_err());
    assert!(!root.join("manifest.json").exists());
    assert!(writer.write_chunk(&sha256(b"new"), b"new").is_err());
    assert!(LocalCheckpointReader::open(&root, &sha256(b"absent"), &expected).is_err());
}

#[test]
fn corrupt_stored_chunk_rejects_before_seal_publication() {
    let (source, memory, receipt) = saved();
    let bytes = memory.manifest.unwrap();
    let manifest: Manifest = serde_json::from_slice(&bytes).unwrap();
    let root = fresh_path();
    let mut writer = LocalCheckpointWriter::create(&root, &source.plan).unwrap();
    for (index, chunk) in manifest.chunks.iter().enumerate() {
        let mut payload = memory.chunks[&chunk.sha256].clone();
        if index == 0 {
            payload[0] ^= 1;
        }
        writer.write_chunk(&chunk.sha256, &payload).unwrap();
    }
    assert!(writer
        .seal_manifest(&bytes, &receipt.manifest_sha256)
        .is_err());
    assert!(!root.join("manifest.json").exists());
    writer.abort().unwrap();
}

#[test]
fn extra_file_before_open_is_not_ignored() {
    let (root, source, receipt) = disk_saved();
    fs::write(root.join("unexpected.tmp"), b"late").unwrap();
    assert!(LocalCheckpointReader::open(&root, &receipt.manifest_sha256, &source.plan).is_err());
}

#[test]
fn late_namespace_insertion_aborts_after_payload_copy() {
    let (root, source, receipt) = disk_saved();
    let mut reader =
        LocalCheckpointReader::open(&root, &receipt.manifest_sha256, &source.plan).unwrap();
    let bytes = reader.manifest_bytes().to_vec();
    fs::write(root.join("late.tmp"), b"late").unwrap();
    let mut target = Target::default();
    let error = restore(
        &mut reader,
        &mut target,
        &bytes,
        &receipt.manifest_sha256,
        &source.plan,
    )
    .unwrap_err();
    assert_eq!(error.phase, "store final verification");
    assert_eq!(target.writes, receipt.chunks);
    assert!(target.poisoned && !target.complete);
}

#[test]
fn held_chunk_and_manifests_reject_concurrent_write_handles() {
    let (root, source, receipt) = disk_saved();
    let mut reader =
        LocalCheckpointReader::open(&root, &receipt.manifest_sha256, &source.plan).unwrap();
    let manifest: Manifest = serde_json::from_slice(reader.manifest_bytes()).unwrap();
    let key = &manifest.chunks[0].sha256;
    let _chunk = reader.open_chunk(key).unwrap();
    for name in [
        format!("{key}.chunk"),
        "manifest.json".into(),
        "manifest.blob".into(),
    ] {
        assert!(OpenOptions::new()
            .write(true)
            .open(root.join(name))
            .is_err());
    }
}

#[test]
fn held_directory_prevents_rename_or_parent_redirection() {
    let (root, source, receipt) = disk_saved();
    let _reader =
        LocalCheckpointReader::open(&root, &receipt.manifest_sha256, &source.plan).unwrap();
    let replacement = root.with_file_name(format!(
        "{}-moved",
        root.file_name().unwrap().to_string_lossy()
    ));
    assert_eq!(root.parent(), replacement.parent());
    assert!(!replacement.exists());
    assert!(fs::rename(&root, &replacement).is_err());
    assert!(root.is_dir());
}

#[test]
fn nested_directory_symlink_is_rejected_before_creating_output() {
    let root = fresh_path();
    fs::create_dir(&root).unwrap();
    let real = root.join("real");
    fs::create_dir(&real).unwrap();
    let alias = root.join("alias");
    std::os::windows::fs::symlink_dir(&real, &alias)
        .expect("Windows symlink test requires privilege; never count an unrun negative as PASS");
    assert!(LocalCheckpointWriter::create(&alias.join("snapshot"), &plan()).is_err());
    assert!(!real.join("snapshot").exists());
}

#[test]
fn manifest_file_symlink_is_rejected() {
    let root = fresh_path();
    fs::create_dir(&root).unwrap();
    let real = root.join("real");
    fs::create_dir(&real).unwrap();
    let bytes = b"synthetic untrusted manifest";
    fs::write(real.join("manifest.blob"), bytes).unwrap();
    std::os::windows::fs::symlink_file(real.join("manifest.blob"), root.join("manifest.json"))
        .expect("Windows symlink test requires privilege");
    assert!(LocalCheckpointReader::open(&root, &sha256(bytes), &plan()).is_err());
}

#[test]
fn unsafe_roots_and_manifest_size_fail_closed() {
    for root in [
        PathBuf::from("relative"),
        fresh_path().join(".."),
        fresh_path().with_extension("bad:stream"),
    ] {
        assert!(LocalCheckpointWriter::create(&root, &plan()).is_err());
    }
    let root = fresh_path();
    fs::create_dir(&root).unwrap();
    fs::write(
        root.join("manifest.json"),
        vec![b' '; MAX_MANIFEST_BYTES + 1],
    )
    .unwrap();
    assert!(LocalCheckpointReader::open(&root, &sha256(b"wrong"), &plan()).is_err());
}

#[test]
fn late_chunk_corruption_is_detected_and_destination_poisoned() {
    let (root, source, receipt) = disk_saved();
    let mut reader =
        LocalCheckpointReader::open(&root, &receipt.manifest_sha256, &source.plan).unwrap();
    let bytes = reader.manifest_bytes().to_vec();
    let manifest: Manifest = serde_json::from_slice(&bytes).unwrap();
    let filename = root.join(format!("{}.chunk", manifest.chunks[1].sha256));
    let mut payload = fs::read(&filename).unwrap();
    payload[0] ^= 1;
    fs::write(&filename, payload).unwrap();
    let mut target = Target::default();
    assert!(restore(
        &mut reader,
        &mut target,
        &bytes,
        &receipt.manifest_sha256,
        &source.plan
    )
    .is_err());
    assert_eq!(target.writes, 1);
    assert!(target.poisoned && target.data.is_empty());
}

#[test]
fn descriptor_bootstrap_verifies_bytes_without_granting_session_admission() {
    let root = fresh_path();
    let mut expected = plan();
    expected.components[0].name = "session.metadata".into();
    let mut source = Source::new(expected.clone(), Trace::default());
    let mut writer = LocalCheckpointWriter::create(&root, &expected).unwrap();
    let receipt = save(&mut source, &mut writer, &expected, 7).unwrap();
    drop(writer);
    let bytes =
        LocalCheckpointReader::read_session_descriptor(&root, &receipt.manifest_sha256).unwrap();
    assert_eq!(bytes, source.data[0]);
    // Transport integrity is deliberately weaker than typed native-state validity.
    assert!(
        serde_json::from_slice::<crate::session_checkpoint::metadata::SessionDescriptor>(&bytes)
            .is_err()
    );
}

#[test]
fn descriptor_bootstrap_rejects_wrong_role_and_content_drift() {
    let (root, _, receipt) = disk_saved();
    assert!(
        LocalCheckpointReader::read_session_descriptor(&root, &receipt.manifest_sha256).is_err()
    );
    let root = fresh_path();
    let mut expected = plan();
    expected.components[0].name = "session.metadata".into();
    let mut source = Source::new(expected.clone(), Trace::default());
    let mut writer = LocalCheckpointWriter::create(&root, &expected).unwrap();
    let receipt = save(&mut source, &mut writer, &expected, 7).unwrap();
    drop(writer);
    let manifest: Manifest =
        serde_json::from_slice(&fs::read(root.join("manifest.json")).unwrap()).unwrap();
    let file = root.join(format!("{}.chunk", manifest.chunks[0].sha256));
    fs::write(file, [1; 7]).unwrap();
    assert!(
        LocalCheckpointReader::read_session_descriptor(&root, &receipt.manifest_sha256).is_err()
    );
}

#[test]
fn input_prefix_bootstrap_and_session_bootstrap_reject_each_other() {
    for role in ["session.metadata", "input_prefix.metadata"] {
        let root = fresh_path();
        let mut expected = plan();
        expected.components[0].name = role.into();
        let mut source = Source::new(expected.clone(), Trace::default());
        let mut writer = LocalCheckpointWriter::create(&root, &expected).unwrap();
        let receipt = save(&mut source, &mut writer, &expected, 7).unwrap();
        drop(writer);
        let session =
            LocalCheckpointReader::read_session_descriptor(&root, &receipt.manifest_sha256);
        let prefix =
            LocalCheckpointReader::read_input_prefix_descriptor(&root, &receipt.manifest_sha256);
        if role == "session.metadata" {
            assert_eq!(session.unwrap(), source.data[0]);
            assert!(prefix.is_err());
        } else {
            assert!(session.is_err());
            assert_eq!(prefix.unwrap(), source.data[0]);
        }
    }
}

#[test]
fn input_prefix_bootstrap_rejects_bad_hash_and_chunk_drift() {
    let root = fresh_path();
    let mut expected = plan();
    expected.components[0].name = "input_prefix.metadata".into();
    let mut source = Source::new(expected.clone(), Trace::default());
    let mut writer = LocalCheckpointWriter::create(&root, &expected).unwrap();
    let receipt = save(&mut source, &mut writer, &expected, 7).unwrap();
    drop(writer);
    assert!(LocalCheckpointReader::read_input_prefix_descriptor(&root, &"a".repeat(64)).is_err());
    let manifest: Manifest =
        serde_json::from_slice(&fs::read(root.join("manifest.json")).unwrap()).unwrap();
    fs::write(
        root.join(format!("{}.chunk", manifest.chunks[0].sha256)),
        [1; 7],
    )
    .unwrap();
    assert!(
        LocalCheckpointReader::read_input_prefix_descriptor(&root, &receipt.manifest_sha256)
            .is_err()
    );
}
