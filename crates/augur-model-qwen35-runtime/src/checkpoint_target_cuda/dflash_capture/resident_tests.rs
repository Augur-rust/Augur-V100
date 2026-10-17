//! Synthetic ring seam only: not real target long-history/acceptance evidence.
use super::*;
use crate::resident_growing_history::{GrowingHistory, GrowingHistoryConfig};
use crate::{
    DFlashDraftBlockInput, DFlashQ8Backbone, DFlashQ8CudaBlock, DFlashQ8CudaWeights,
    DFLASH_SEED_MASK_TOKEN,
};
use augur_backend::cuda_host_boundary_snapshot;
use augur_model_qwen35::DFlashConfig;
use serde_json::{json, Value};
use std::io::Write;

fn append(capture: &mut Capture, source: &OwnedCudaBuffer, first: usize, rows: usize) {
    capture.begin(first, rows).unwrap();
    for layer in 0..5 {
        let wire: Vec<u8> = (0..2048 * H)
            .flat_map(|i| {
                (if i / H < rows {
                    super::tests::pattern(first + i / H, layer, i % H)
                } else {
                    f32::NAN
                })
                .to_le_bytes()
            })
            .collect();
        source.copy_from_host_sync(&capture.stream, &wire).unwrap();
        capture
            .layer(DFLASH_TARGET_CAPTURE_LAYER_IDS[layer], first, rows, source)
            .unwrap();
    }
    capture.commit().unwrap();
}
fn configured(mut draft: DFlashQ8CudaBlock) -> DFlashQ8CudaBlock {
    draft.enable_context_rows_launch().unwrap();
    draft.enable_block_rows_launch().unwrap();
    draft.enable_q8_tile4().unwrap();
    draft
}
#[test]
#[ignore = "exclusive Q8 + synthetic capture: device lease wrap/reset equality, not real long history"]
fn gpu_owner_lease_wrap_reset_and_payload_separation() {
    let cfg = std::fs::read(std::env::var_os("AUGUR_DRAFT_CONFIG").unwrap()).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&cfg)),
        std::env::var("AUGUR_DRAFT_CONFIG_SHA256").unwrap()
    );
    let q8 = DFlashQ8Backbone::load_pinned(
        std::env::var_os("AUGUR_DRAFT_PATH").unwrap(),
        DFlashConfig::parse_f16_opt_in(&cfg).unwrap(),
    )
    .unwrap();
    assert_eq!(
        q8.manifest().file_sha256.as_ref().unwrap(),
        &std::env::var("AUGUR_DRAFT_SHA256").unwrap()
    );
    let before = cuda_device_allocation_snapshot();
    let weights = DFlashQ8CudaWeights::upload(&q8, 0).unwrap();
    let stream = Rc::new(OwnedCudaStream::create(0).unwrap());
    let source = OwnedCudaBuffer::allocate(0, 2048 * LAYER_BYTES).unwrap();
    let noise = OwnedCudaBuffer::allocate(0, 8 * H * 4).unwrap();
    let host_noise = vec![0.125f32; 8 * H];
    noise
        .copy_from_host_sync(
            &stream,
            &host_noise
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let mut capture = Capture::new(Rc::clone(&stream), 262144, [37; 32]).unwrap();
    let mut resident: Option<DFlashQ8CudaBlock> = None;
    let mut observations = Vec::new();
    for (generation, first, rows) in [(1u64, 0, 2046), (2, 2046, 8), (3, 0, 3)] {
        if generation == 3 {
            capture.reset().unwrap();
        }
        append(&mut capture, &source, first, rows);
        let snapshot = capture.snapshot().unwrap();
        super::tests::assert_pattern(&snapshot);
        let mut receipt = snapshot.receipt.clone();
        receipt.request_generation = generation;
        let boundary = cuda_host_boundary_snapshot();
        let alloc = cuda_device_allocation_snapshot();
        let lease = capture.device_lease().unwrap();
        let identity = lease.identity();
        if let Some(draft) = &mut resident {
            draft
                .refresh_augur_owner(&lease, noise.view(0, 8 * H * 4).unwrap(), 42, generation)
                .unwrap();
        } else {
            resident = Some(configured(
                DFlashQ8CudaBlock::from_augur_owner(
                    &weights,
                    &lease,
                    noise.view(0, 8 * H * 4).unwrap(),
                    42,
                    generation,
                )
                .unwrap(),
            ));
        }
        drop(lease);
        let delta = cuda_host_boundary_snapshot()
            .checked_delta_since(boundary)
            .unwrap();
        assert_eq!(delta.synchronous_d2h_bytes, 0);
        assert_eq!(
            delta.synchronous_h2d_bytes,
            if generation == 1 { (H * 4) as u64 } else { 0 }
        ); // cold zero-row initialization only
        if generation > 1 {
            assert_eq!(
                cuda_device_allocation_snapshot().successful_allocations,
                alloc.successful_allocations
            );
        }
        let draft = resident.as_mut().unwrap();
        draft.prepare_context().unwrap();
        draft.submit().unwrap();
        draft.synchronize().unwrap();
        assert_eq!(
            draft.submitted_augur_identity().unwrap(),
            (generation, 42, identity)
        );
        assert!(draft.submitted_identity().is_err());
        assert_eq!(
            draft.trajectory_input_features().unwrap().2,
            snapshot.little_endian_f32
        );
        let mut tokens = vec![DFLASH_SEED_MASK_TOKEN; 8];
        tokens[0] = 42;
        let seed = DFlashDraftBlockInput {
            request_generation: generation,
            absolute_position_start: identity.end(),
            anchor_token: 42,
            block_tokens: tokens,
            mask_token_id: DFLASH_SEED_MASK_TOKEN,
            target_feature_sha256: receipt.feature_sha256,
        };
        let history = GrowingHistory::from_extent(
            GrowingHistoryConfig::default_2048(),
            receipt.context_start_position,
            receipt.context_len,
        )
        .unwrap();
        // Independent legacy payload-SHA constructor, same rows/policy/noise.
        let mut reference = configured(
            DFlashQ8CudaBlock::from_resident_growing_device_features(
                &weights,
                &seed,
                &receipt,
                capture
                    .chronological
                    .view(0, snapshot.little_endian_f32.len())
                    .unwrap(),
                &host_noise,
                history,
                2048,
                &stream,
            )
            .unwrap(),
        );
        reference.prepare_context().unwrap();
        reference.submit().unwrap();
        reference.synchronize().unwrap();
        let a = draft.readback_diagnostics().unwrap();
        let b = reference.readback_diagnostics().unwrap();
        assert_eq!(a.hidden, b.hidden);
        assert_eq!(a.selector_hidden, b.selector_hidden);
        let lease = capture.device_lease().unwrap();
        assert!(reference
            .refresh_augur_owner(
                &lease,
                noise.view(0, 8 * H * 4).unwrap(),
                42,
                generation + 1
            )
            .is_err());
        assert!(draft
            .refresh_augur_owner(&lease, noise.view(0, 8 * H * 4).unwrap(), 42, generation)
            .is_err());
        observations.push(
            json!({"generation":generation,"epoch":identity.capture_identity().0,
            "start":identity.start(),"rows":identity.rows(),"exclusive_end":identity.end(),
            "features_hidden_selector_bitwise_equal":true,"feature_payload_D2H_bytes":0}),
        );
    }
    // Lease unavailable after a partially initialized transaction.
    capture.begin(3, 1).unwrap();
    assert!(capture.device_lease().is_err());
    drop(resident);
    drop(capture);
    drop(noise);
    drop(source);
    drop(stream);
    drop(weights);
    let after = cuda_device_allocation_snapshot();
    assert_eq!(after.live_bytes, before.live_bytes);
    assert_eq!(after.live_allocations, before.live_allocations);
    let result: Value = json!({"status":"PASS_SYNTHETIC_OWNER_LEASE_WRAP_RESET","observations":observations,
        "real_target_long_history":false,"model_acceptance_sample":false,"dirty_capture_rejected":true,
        "cross_provenance_and_nonincreasing_generation_rejected":true,
        "cleanup_live_bytes_delta":after.live_bytes-before.live_bytes,"cleanup_live_allocations_delta":after.live_allocations-before.live_allocations});
    let root = std::path::PathBuf::from(std::env::var_os("AUGUR_BRIDGE_OUTPUT").unwrap());
    let mut out = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(root.join("lease-rollover.json"))
        .unwrap();
    out.write_all(&serde_json::to_vec_pretty(&result).unwrap())
        .unwrap();
    out.sync_all().unwrap();
}
