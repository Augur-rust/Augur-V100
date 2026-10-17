use super::*;
use augur_backend::{cuda_host_boundary_snapshot, CudaHostBoundarySnapshot};
use serde_json::{json, Value};
use std::io::Write;

fn mark(ledger: &mut Ledger, first: usize, rows: usize) {
    for layer in 0..5 {
        ledger.check_layer(layer, first, rows).unwrap();
        ledger.mark_layer(layer, first, rows);
    }
}
#[test]
fn segments_preserve_absolute_order_at_every_wrap_and_width() {
    for first in (0..HISTORY_ROWS + 9).chain([260091, 262136]) {
        for rows in (1..=8).chain([HISTORY_ROWS]) {
            let actual: Vec<_> = ring_segments(first, rows)
                .unwrap()
                .iter()
                .flat_map(|s| (0..s.rows).map(move |i| (s.source_row + i, s.destination_row + i)))
                .collect();
            let expected: Vec<_> = (0..rows).map(|i| (i, (first + i) % HISTORY_ROWS)).collect();
            assert_eq!(actual, expected);
        }
    }
    assert!(ring_segments(0, 0).is_err());
    assert!(ring_segments(0, HISTORY_ROWS + 1).is_err());
    assert!(ring_segments(usize::MAX, 1).is_err());
}
#[test]
fn ledger_rejects_missing_duplicate_reordered_stale_and_out_of_range_rows() {
    let mut ledger = Ledger::new(262144).unwrap();
    assert!(ledger.range().is_err());
    assert!(ledger.begin(1, 1).is_err());
    assert!(ledger.begin(0, 0).is_err());
    ledger.begin(0, 8).unwrap();
    assert!(ledger.begin(0, 1).is_err());
    assert!(ledger.check_layer(1, 0, 8).is_err());
    assert!(ledger.check_layer(0, 0, 9).is_err());
    assert!(ledger.check_layer(5, 0, 8).is_err());
    ledger.check_layer(0, 0, 8).unwrap();
    ledger.mark_layer(0, 0, 8);
    assert!(ledger.check_layer(0, 0, 8).is_err());
    assert!(ledger.commit().is_err());
    assert!(ledger.range().is_err());
    for layer in 1..5 {
        ledger.check_layer(layer, 0, 8).unwrap();
        ledger.mark_layer(layer, 0, 8);
    }
    ledger.commit().unwrap();
    assert_eq!(ledger.range().unwrap(), (0, 8));
    ledger.positions[1] = Some(2049);
    assert!(ledger.range().is_err());
}
#[test]
fn full_256ki_metadata_keeps_only_initialized_last_2048_and_reset_invalidates() {
    let mut ledger = Ledger::new(262144).unwrap();
    for first in (0..262144).step_by(128) {
        ledger.begin(first, 128).unwrap();
        for pos in (first..first + 128).step_by(8) {
            mark(&mut ledger, pos, 8);
        }
        ledger.commit().unwrap();
        let end = first + 128;
        assert_eq!(
            ledger.range().unwrap(),
            (end.saturating_sub(HISTORY_ROWS), end.min(HISTORY_ROWS))
        );
    }
    assert!(ledger.begin(262144, 1).is_err());
    let generation = ledger.generation;
    ledger.reset().unwrap();
    assert_eq!(ledger.epoch, 2);
    assert_eq!(ledger.generation, generation + 1);
    assert!(ledger.range().is_err());
    ledger.begin(0, 3).unwrap();
    mark(&mut ledger, 0, 3);
    ledger.commit().unwrap();
    assert_eq!(ledger.range().unwrap(), (0, 3));
    assert!(ledger.positions[3..].iter().all(Option::is_none));
}
#[test]
fn partial_failed_operation_requires_explicit_reset_and_generation_never_wraps() {
    let mut ledger = Ledger::new(256).unwrap();
    ledger.begin(0, 3).unwrap();
    mark(&mut ledger, 0, 2);
    assert!(ledger.commit().is_err());
    assert!(ledger.begin(0, 3).is_err());
    ledger.reset().unwrap();
    ledger.begin(0, 1).unwrap();
    mark(&mut ledger, 0, 1);
    ledger.commit().unwrap();
    ledger.generation = u64::MAX;
    assert!(ledger.begin(1, 1).is_err());
    assert!(ledger.reset().is_err());
    assert_eq!(ledger.end, 1);
    assert!(Ledger::new(0).is_err());
    assert!(Ledger::new(262145).is_err());
}

fn save(name: &str, value: &Value) {
    let root = std::path::PathBuf::from(
        std::env::var_os("AUGUR_CAPTURE_OUTPUT").expect("new output directory"),
    );
    assert!(root.is_dir());
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(root.join(name))
        .unwrap();
    file.write_all(&serde_json::to_vec_pretty(value).unwrap())
        .unwrap();
    file.sync_all().unwrap();
}
pub(super) fn pattern(pos: usize, layer: usize, column: usize) -> f32 {
    f32::from_bits(0x3f000000 | ((pos * 997 + layer * 137 + column * 17) % 0x7fffff) as u32)
}
fn no_added_host_transfers(a: CudaHostBoundarySnapshot, b: CudaHostBoundarySnapshot) {
    assert_eq!(a.synchronous_d2h_bytes, b.synchronous_d2h_bytes);
    assert_eq!(a.synchronous_h2d_bytes, b.synchronous_h2d_bytes);
    assert_eq!(a.host_waits, b.host_waits);
}
pub(super) fn assert_pattern(s: &FeatureSnapshot) {
    for (i, bytes) in s.little_endian_f32.chunks_exact(4).enumerate() {
        let pos = s.receipt.context_start_position + i / WIDTH;
        let layer = (i % WIDTH) / H;
        let col = i % H;
        assert_eq!(bytes, pattern(pos, layer, col).to_le_bytes());
    }
}
#[test]
#[ignore = "exclusive CUDA; bounded synthetic device ring, NOT real long target history"]
fn gpu_synthetic_capture_b8_remainders_2048_wrap_reset_cleanup() {
    let before = cuda_device_allocation_snapshot();
    let stream = Rc::new(OwnedCudaStream::create(0).unwrap());
    let mut capture = Capture::new(Rc::clone(&stream), 262144, [1; 32]).unwrap();
    let source = OwnedCudaBuffer::allocate(0, 8 * LAYER_BYTES).unwrap();
    capture.begin(0, 2057).unwrap();
    // Start with width3 so a later B8 copy straddles slot2047/0, rather
    // than testing only a wrap exactly between two aligned batches.
    for first in std::iter::once(0).chain((3..2057).step_by(8)) {
        let rows = if first == 0 { 3 } else { 8.min(2057 - first) };
        for layer in 0..5 {
            let values: Vec<u8> = (0..8 * H)
                .flat_map(|i| {
                    if i / H < rows {
                        pattern(first + i / H, layer, i % H)
                    } else {
                        f32::NAN
                    }
                    .to_le_bytes()
                })
                .collect();
            source.copy_from_host_sync(&stream, &values).unwrap();
            let b = cuda_host_boundary_snapshot();
            capture
                .layer(DFLASH_TARGET_CAPTURE_LAYER_IDS[layer], first, rows, &source)
                .unwrap();
            no_added_host_transfers(b, cuda_host_boundary_snapshot());
        }
    }
    capture.commit().unwrap();
    let snapshot = capture.snapshot().unwrap();
    assert_eq!(snapshot.receipt.context_start_position, 9);
    assert_eq!(snapshot.receipt.context_len, 2048);
    assert_pattern(&snapshot);
    capture.reset().unwrap();
    assert!(capture.snapshot().is_err());
    capture.begin(0, 3).unwrap();
    for layer in 0..5 {
        let values: Vec<u8> = (0..8 * H)
            .flat_map(|i| {
                if i / H < 3 {
                    pattern(i / H, layer, i % H)
                } else {
                    f32::NAN
                }
                .to_le_bytes()
            })
            .collect();
        source.copy_from_host_sync(&stream, &values).unwrap();
        capture
            .layer(DFLASH_TARGET_CAPTURE_LAYER_IDS[layer], 0, 3, &source)
            .unwrap();
    }
    capture.commit().unwrap();
    let reset = capture.snapshot().unwrap();
    assert_eq!(reset.receipt.context_len, 3);
    assert_eq!(reset.owner_epoch, 2);
    assert_pattern(&reset);
    stream.synchronize().unwrap();
    let live = cuda_device_allocation_snapshot();
    drop(source);
    drop(capture);
    drop(stream);
    let after = cuda_device_allocation_snapshot();
    assert_eq!(after.live_bytes, before.live_bytes);
    assert_eq!(after.live_allocations, before.live_allocations);
    save(
        "synthetic.json",
        &json!({"status":"PASS_SYNTHETIC_DEVICE_CAPTURE", "real_long_history":false,
        "rows_written":2057, "window":snapshot.receipt, "reset_window":reset.receipt,
        "post_reset_epoch":reset.owner_epoch, "capture_added_host_transfers":false,
        "live_bytes_including_test_source":live.live_bytes - before.live_bytes,
        "cleanup_live_bytes_delta":after.live_bytes - before.live_bytes,
        "cleanup_live_allocations_delta":after.live_allocations - before.live_allocations}),
    );
}
fn boundary_delta(a: CudaHostBoundarySnapshot, b: CudaHostBoundarySnapshot) -> Value {
    json!({"host_waits":b.host_waits-a.host_waits,
        "stream_synchronizations":b.stream_synchronizations-a.stream_synchronizations,
        "synchronous_d2h_copies":b.synchronous_d2h_copies-a.synchronous_d2h_copies,
        "synchronous_d2h_bytes":b.synchronous_d2h_bytes-a.synchronous_d2h_bytes,
        "synchronous_h2d_copies":b.synchronous_h2d_copies-a.synchronous_h2d_copies,
        "synchronous_h2d_bytes":b.synchronous_h2d_bytes-a.synchronous_h2d_bytes})
}
fn expected_bytes(rows: usize, values: &[(usize, usize, Vec<f32>)]) -> Vec<u8> {
    let mut reference = vec![0; rows * ROW_BYTES];
    let mut seen = vec![false; rows * 5];
    for (position, layer, hidden) in values {
        let slot = DFLASH_TARGET_CAPTURE_LAYER_IDS
            .iter()
            .position(|id| id == layer)
            .unwrap();
        assert!(*position < rows && hidden.len() == H && !seen[position * 5 + slot]);
        seen[position * 5 + slot] = true;
        for (col, value) in hidden.iter().enumerate() {
            let offset = position * ROW_BYTES + slot * LAYER_BYTES + col * 4;
            reference[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
    }
    assert!(seen.iter().all(|v| *v));
    reference
}
fn values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
#[test]
#[ignore = "exclusive full262144 KV8 Augur; source-bound post-layer capture admission only"]
fn gpu_real_augur_capture_scalar_b8_pending_failure_reset_and_math() {
    use super::super::tests::bytes;
    use super::super::transfer_tests::{ids, open};
    let fixture_path = std::env::var_os("AUGUR_CHECKPOINT_TARGET_REFERENCE").expect("fixture");
    let fixture_bytes = std::fs::read(fixture_path).unwrap();
    let case: Value = serde_json::from_slice(&fixture_bytes).unwrap();
    assert_eq!(case["profile"], "quasar");
    let short = ids(&case["token_ids"]);
    assert_eq!(short.len(), 4);
    let tokens: Vec<u32> = short.iter().copied().cycle().take(19).collect();
    let before = cuda_device_allocation_snapshot();
    let mut target = open(&case);
    assert_eq!(target.capacity, 262144);
    assert_eq!(target.info()["ffn_batch_max_rows"], 8);
    assert_eq!(
        target.info()["qpn_input_staging_schedule"],
        "fused_cached_norm_qkv_v1"
    );
    let boundary_before = cuda_host_boundary_snapshot();
    target.consume_sequence(&tokens).unwrap();
    let boundary_after = cuda_host_boundary_snapshot();
    let baseline_boundary = boundary_delta(boundary_before, boundary_after);
    let logits = target.logits_snapshot().unwrap();
    let pending = target.greedy_next_token().unwrap();
    target.consume_selected().unwrap();
    let selected_logits = target.logits_snapshot().unwrap();
    target.reset_empty().unwrap();
    target.enable_dflash_feature_capture().unwrap();
    assert!(target.enable_dflash_feature_capture().is_err());
    assert!(target.dflash_feature_snapshot().is_err());
    assert!(!target.poisoned);
    let installed = target.info().clone();
    assert_eq!(
        installed["dflash_feature_capture"]["added_live_bytes"],
        2 * HISTORY_ROWS * ROW_BYTES
    );
    target.ffn_batch.as_mut().unwrap().diagnostic_capture =
        Some(std::cell::RefCell::new(Vec::new()));
    target.consume_sequence(&tokens).unwrap();
    let direct = target
        .ffn_batch
        .as_mut()
        .unwrap()
        .diagnostic_capture
        .take()
        .unwrap()
        .into_inner();
    let batched = target.dflash_feature_snapshot().unwrap();
    assert_eq!(batched.little_endian_f32, expected_bytes(19, &direct));
    assert_eq!(
        target.logits_snapshot().unwrap(),
        logits,
        "capture may not change target logits"
    );
    assert_eq!(target.greedy_next_token().unwrap(), pending);
    let still = target.dflash_feature_snapshot().unwrap();
    assert_eq!(
        still.receipt, batched.receipt,
        "selected pending is not a consumed feature row"
    );
    target.consume_selected().unwrap();
    let selected = target.dflash_feature_snapshot().unwrap();
    assert_eq!(selected.receipt.context_len, 20);
    assert_eq!(target.logits_snapshot().unwrap(), selected_logits);
    target.reset_empty().unwrap();
    assert!(target.dflash_feature_snapshot().is_err());
    let boundary_before = cuda_host_boundary_snapshot();
    target.consume_sequence(&tokens).unwrap();
    let captured_boundary = boundary_delta(boundary_before, cuda_host_boundary_snapshot());
    assert_eq!(
        baseline_boundary, captured_boundary,
        "device capture adds no host transfer/wait"
    );
    let scalar = target
        .consume(pending, true, &DFLASH_TARGET_CAPTURE_LAYER_IDS)
        .unwrap();
    let mut combined = direct;
    combined.extend(
        scalar
            .into_iter()
            .map(|(layer, hidden)| (19, layer, hidden)),
    );
    let repeated = target.dflash_feature_snapshot().unwrap();
    assert_eq!(repeated.little_endian_f32, expected_bytes(20, &combined));
    assert_eq!(repeated.little_endian_f32, selected.little_endian_f32);
    assert!(repeated.receipt.request_generation > selected.receipt.request_generation);
    assert_eq!(repeated.owner_epoch, selected.owner_epoch + 1);
    // Interrupted B8 after a capture layer must never publish partial rows.
    target.reset_empty().unwrap();
    target.ffn_batch.as_mut().unwrap().fail_after_layer = Some(19);
    assert!(target.consume_sequence(&tokens[..8]).is_err());
    assert!(target.poisoned && target.dflash_feature_snapshot().is_err());
    target.reset_empty().unwrap();
    target.ffn_batch.as_mut().unwrap().fail_after_layer = None;
    // Independent original tensor/reference qualification, unchanged 1e-3.
    let mut comparisons = Vec::new();
    let mut direct_scalar = Vec::new();
    for (position, token) in short.iter().enumerate() {
        let captured = target
            .consume(*token, true, &DFLASH_TARGET_CAPTURE_LAYER_IDS)
            .unwrap();
        for (layer, actual) in &captured {
            let reference = bytes(&case["layers"][*layer]);
            let reference: Vec<f64> = reference[position * H * 8..(position + 1) * H * 8]
                .chunks_exact(8)
                .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
                .collect();
            let error2: f64 = actual
                .iter()
                .zip(&reference)
                .map(|(a, b)| (f64::from(*a) - b).powi(2))
                .sum();
            let norm2: f64 = reference.iter().map(|v| v * v).sum();
            let max_abs = actual
                .iter()
                .zip(&reference)
                .map(|(a, b)| (f64::from(*a) - b).abs())
                .fold(0., f64::max);
            let max_ref = reference.iter().map(|v| v.abs()).fold(0., f64::max);
            let rel_l2 = (error2 / norm2.max(1e-300)).sqrt();
            let scaled_max = max_abs / (1. + max_ref);
            assert!(
                actual.iter().all(|v| v.is_finite()) && reference.iter().all(|v| v.is_finite())
            );
            assert!(
                rel_l2 <= 1e-3 && scaled_max <= 1e-3,
                "position={position} layer={layer} rel={rel_l2} scaled={scaled_max}"
            );
            comparisons.push(
                json!({"position":position,"layer":layer,"rel_l2":rel_l2,"scaled_max":scaled_max}),
            );
        }
        direct_scalar.extend(
            captured
                .into_iter()
                .map(|(layer, hidden)| (position, layer, hidden)),
        );
    }
    let recovered = target.dflash_feature_snapshot().unwrap();
    assert_eq!(
        recovered.little_endian_f32,
        expected_bytes(4, &direct_scalar)
    );
    assert!(values(&recovered.little_endian_f32)
        .iter()
        .all(|v| v.is_finite()));
    target.reset_empty().unwrap();
    drop(target);
    let after = cuda_device_allocation_snapshot();
    assert_eq!(after.live_bytes, before.live_bytes);
    assert_eq!(after.live_allocations, before.live_allocations);
    save(
        "real-target.json",
        &json!({"status":"PASS_AUGUR_DEVICE_CAPTURE_ADMISSION", "target":installed,
        "fixture_sha256":format!("{:x}",Sha256::digest(&fixture_bytes)),
        "batched_rows":19,"batched_direct_d2h_bitwise_match":true,"scalar_direct_d2h_bitwise_match":true,
        "capture_off_on_logits_bitwise_match":true,
        "capture_off_host_boundary":baseline_boundary,"capture_on_host_boundary":captured_boundary,"pending_token_excluded_until_consume":true,
        "device_selected_and_explicit_consumption_bitwise_match":true,
        "injected_partial_batch_failure_rejected_then_reset":true,
        "batched_receipt":batched.receipt,"selected_receipt":selected.receipt,
        "repeated_receipt":repeated.receipt,"independent_layer_math":comparisons,
        "limits_unchanged":1e-3,"dflash_acceptance_measured":false,"real_long_history":false,
        "cleanup_live_bytes_delta":after.live_bytes-before.live_bytes,
        "cleanup_live_allocations_delta":after.live_allocations-before.live_allocations}),
    );
}

#[cfg(feature = "qwen35-augur-dflash-bridge")]
#[test]
fn owner_identity_rejects_foreign_stale_and_false_payload_substitution() {
    let a = OwnerFeatureIdentity { source: [1;32], owner: 7, epoch: 1, revision: 2, start:0, rows:17 };
    assert!(a.validate_successor(a).is_ok());
    for bad in [OwnerFeatureIdentity{source:[2;32],..a}, OwnerFeatureIdentity{owner:8,..a},
        OwnerFeatureIdentity{revision:1,..a}, OwnerFeatureIdentity{rows:16,revision:3,..a},
        OwnerFeatureIdentity{rows:18,..a}, OwnerFeatureIdentity{epoch:2,..a}] {
        assert!(a.validate_successor(bad).is_err());
    }
    let next = OwnerFeatureIdentity { revision:3, rows:25, ..a };
    assert!(a.validate_successor(next).is_ok());
    let reset = OwnerFeatureIdentity {epoch:2,revision:4,rows:3,..a};
    assert!(next.validate_successor(reset).is_ok());
    assert!(reset.validate_successor(next).is_err());
}

#[cfg(feature = "qwen35-augur-dflash-bridge")]
#[test]
fn retained_owner_context_handles_append_roll_repeat_reset_and_disjoint_windows() {
    let a = OwnerFeatureIdentity { source:[1;32], owner:7, epoch:1, revision:2, start:0, rows:17 };
    assert_eq!(a.retained_context_rows(a).unwrap(), (17, 0));
    assert_eq!(a.retained_context_rows(OwnerFeatureIdentity {revision:3, rows:25, ..a}).unwrap(), (17, 0));
    let full = OwnerFeatureIdentity {revision:20, rows:2048, ..a};
    let rolled = OwnerFeatureIdentity {revision:21, start:7, ..full};
    assert_eq!(full.retained_context_rows(rolled).unwrap(), (2041, 7));
    assert_eq!(rolled.retained_context_rows(OwnerFeatureIdentity {revision:22, start:4096, ..rolled}).unwrap(), (0, 0));
    assert_eq!(rolled.retained_context_rows(OwnerFeatureIdentity {epoch:2, revision:22, start:0, rows:3, ..rolled}).unwrap(), (0, 0));
    for bad in [OwnerFeatureIdentity {source:[2;32], ..rolled},
        OwnerFeatureIdentity {owner:8, ..rolled}, OwnerFeatureIdentity {revision:19, ..rolled}] {
        assert!(rolled.retained_context_rows(bad).is_err());
    }
    let shifted = OwnerFeatureIdentity {start:100, rows:100, ..a};
    assert!(shifted.retained_context_rows(OwnerFeatureIdentity {start:99, rows:105, revision:3, ..shifted}).is_err());
}
