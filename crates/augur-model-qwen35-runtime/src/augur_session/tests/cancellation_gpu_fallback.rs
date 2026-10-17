//! Deterministic real-GPU cancellation boundaries; not HTTP timing evidence.
use super::*;
use crate::{
    checkpoint_full_cuda::CheckpointKvPolicy,
    checkpoint_projection_cuda::CheckpointProjectionPolicy,
    checkpoint_target_cuda::{
        dflash_bridge::AugurDflashBridge, augur_prefill::Budget, CheckpointCudaTarget,
    },
    augur_session_cuda::augur_http,
};
use augur_backend::cuda_device_allocation_snapshot;
use augur_model_qwen35::{Qwen35CheckpointProfile, Qwen35PreparedF16Image};
use serde_json::{json, Value};
use std::{cell::Cell, fs, path::PathBuf};

struct RealTarget {
    // The draft holds views into target memory and must drop first.
    _draft: AugurDflashBridge,
    target: CheckpointCudaTarget,
    mode: &'static str,
    boundary: Option<Value>,
    reset_calls: usize,
}
impl SessionTarget for RealTarget {
    fn consumed_tokens(&self) -> usize {
        self.target.consumed_tokens()
    }
    fn consume_sequence(&mut self, ids: &[u32]) -> Result<()> {
        self.target.consume_sequence(ids)
    }
    fn prefill(&mut self, ids: &[u32], cancel: &SessionCancellationToken) -> Result<Option<Value>> {
        let widths =
            augur_backend::cuda_qwen35_prefill_cublas::augur_prefill_widths(ids.len())?;
        let plan = self.target.plan_augur_sequence(ids, widths)?;
        let mode = self.mode;
        let boundary = &mut self.boundary;
        let receipt = self.target.consume_sequence_augur(
            ids,
            &plan,
            Budget::default(),
            |progress| {
                if mode == "prefill" && progress.first == 4352 && progress.layer == 5 {
                    *boundary = Some(
                        json!({"first":progress.first,"rows":progress.rows,"layer":progress.layer}),
                    );
                    cancel.request_cancel();
                }
                cancel.is_requested()
            },
        )?;
        Ok(Some(serde_json::to_value(receipt)?))
    }
    fn supports_images(&self) -> bool {
        true
    }
    fn prefill_images(
        &mut self,
        ids: &[u32],
        images: &[Qwen35PreparedF16Image],
        cancel: &SessionCancellationToken,
    ) -> Result<Option<Value>> {
        let polls = Cell::new(0usize);
        let result = self.target.prefill_images(ids, images, || {
            polls.set(polls.get() + 1);
            // before_load + before_allocation + block0 + block1 + block2.
            // Two actual encoder blocks have been submitted before cancellation.
            if polls.get() == 5 {
                cancel.request_cancel();
            }
            cancel.is_requested()
        });
        self.boundary = Some(json!({"polls":polls.get(),"submitted_encoder_blocks":2}));
        Ok(Some(result?))
    }
    fn greedy_next_token(&mut self) -> Result<u32> {
        self.target.greedy_next_token()
    }
    fn consume_selected(&mut self) -> Result<()> {
        self.target.consume_selected_augur()
    }
    fn drain(&mut self) -> Result<()> {
        self.target.drain_session_work()
    }
    fn reset_empty(&mut self) -> Result<()> {
        self.reset_calls += 1;
        self.target.reset_empty()
    }
}

#[test]
#[ignore = "exclusive real Augur GPU: partial second prefill chunk, encoder cancellation and post-commit cancellation"]
fn unsupported_cancellation_and_late_commit_require_retirement_on_real_gpu() -> Result<()> {
    let root = PathBuf::from(std::env::var("AUGUR_MODEL_ROOT")?);
    let manifest_path = PathBuf::from(std::env::var("AUGUR_MANIFEST")?);
    let manifest_hash = std::env::var("AUGUR_MANIFEST_SHA256")?;
    let evidence = PathBuf::from(std::env::var("AUGUR_CANCEL_FALLBACK_EVIDENCE")?);
    fs::create_dir(&evidence)?;
    let before = cuda_device_allocation_snapshot();
    assert_eq!((before.live_bytes, before.live_allocations), (0, 0));
    let mut results = Vec::new();
    for mode in ["prefill", "encoder", "late_commit"] {
        let out = evidence.join(mode);
        fs::create_dir(&out)?;
        let admitted = manifest::load(&root, &manifest_path, &manifest_hash)?;
        let draft_load = augur_http::DraftLoad::begin()?;
        let mut target = CheckpointCudaTarget::open_with_execution_policy(
            &root,
            Qwen35CheckpointProfile::QuasarNvfp4,
            admitted.manifest().checkpoint.clone(),
            CAPACITY,
            CheckpointProjectionPolicy::AugurQpnAllWorkspaceShapeSmallPairF64,
            CheckpointKvPolicy::Q8_0,
        )?;
        target.enable_exact_warp_tree_norm()?;
        target.enable_fused_input_qpn_norm()?;
        target.enable_batched_ffn8()?;
        let info = target.info().clone();
        let (draft, info) = augur_http::prepare(&mut target, info, draft_load)?;
        persist_receipt(&out, "target-info", &info)?;
        let wrapped = RealTarget {
            _draft: draft,
            target,
            mode,
            boundary: None,
            reset_calls: 0,
        };
        let mut owner = Session::new(
            wrapped,
            manifest_hash.clone(),
            admitted.eos_token_ids().to_vec(),
        )?;
        let input = if mode == "encoder" {
            let rgb = vec![128u8; 256 * 256 * 3];
            let mut ids = vec![248053];
            ids.extend(vec![248056; 64]);
            ids.extend([248054, 42]);
            TurnInput::OpenImages {
                prompt_token_ids: ids,
                images: vec![images::ImageInput {
                    width: 256,
                    height: 256,
                    rgb8_hex: "80".repeat(rgb.len()),
                    rgb8_sha256: digest(&rgb),
                }],
            }
        } else {
            TurnInput::Open {
                prompt_token_ids: vec![42; if mode == "prefill" { 8236 } else { 32 }],
            }
        };
        let request = TurnRequest {
            selector_policy: None, request_id: mode.into(),
            session_id: format!("{mode}-session"),
            max_new_tokens: 2,
            input,
        };
        let cancel = SessionCancellationToken::new();
        let resident = cuda_device_allocation_snapshot();
        let mut published = 0usize;
        let mut persisted = false;
        let result = owner.turn(
            &request,
            &cancel,
            |_| {
                published += 1;
                Ok(())
            },
            |receipt| {
                if mode != "late_commit" {
                    return Err("cancelled operation reached persistence".into());
                }
                let hash = persist_receipt(&out, &receipt.request_id, receipt)?;
                persisted = true;
                // Cancellation after the durable commit point must not revoke it.
                cancel.request_cancel();
                Ok(hash)
            },
        );
        assert!(cancel.is_requested());
        let (error, committed_hash) = if mode == "late_commit" {
            let committed = result?;
            assert!(persisted && owner.snapshot().is_some() && !owner.poisoned());
            (None, Some(committed.receipt_sha256))
        } else {
            let error = result.unwrap_err();
            assert!(!error.is::<crate::session_cancellation::SessionCancellationRequested>());
            assert!(owner.poisoned() && owner.snapshot().is_none());
            assert!(!persisted && published == 0 && !out.join(format!("{mode}.json")).exists());
            if mode == "prefill" {
                assert!(error.to_string().contains("augur prefill cancelled"));
                assert_eq!(
                    owner.target.boundary,
                    Some(json!({"first":4352,"rows":3884,"layer":5}))
                );
            } else {
                assert!(error
                    .to_string()
                    .contains("Vision request cancelled during block submission"));
                assert_eq!(
                    owner.target.boundary,
                    Some(json!({"polls":5,"submitted_encoder_blocks":2}))
                );
            }
            (Some(error.to_string()), None)
        };
        let after = cuda_device_allocation_snapshot();
        assert_eq!(
            (resident.live_bytes, resident.live_allocations),
            (after.live_bytes, after.live_allocations)
        );
        let discard_error = owner
            .discard_cancelled_turn("forbidden-discard", mode, &request.session_id, |_| {
                Err("discard must not persist".into())
            })
            .unwrap_err();
        assert_eq!(owner.target.reset_calls, 0);
        owner.retire()?;
        assert!(owner.poisoned() && owner.snapshot().is_none());
        assert!(owner
            .discard_cancelled_turn(
                "forbidden-after-retire",
                mode,
                &request.session_id,
                |_| Err("discard must not persist".into())
            )
            .is_err());
        let report = json!({"case":mode,"boundary":owner.target.boundary,"error":error,
            "committed_receipt_sha256":committed_hash,"published_tokens":published,
            "discard_rejected":discard_error.to_string(),"reset_calls":owner.target.reset_calls,
            "retired":true,"temporary_allocations_closed":true});
        drop(owner);
        let closed = cuda_device_allocation_snapshot();
        assert_eq!((closed.live_bytes, closed.live_allocations), (0, 0));
        persist_receipt(&out, "complete", &report)?;
        eprintln!("AUGUR_GPU_CANCEL_FALLBACK {report}");
        results.push(report);
    }
    persist_receipt(
        &evidence,
        "complete",
        &json!({"gpu_pass":true,"cases":results,
        "live_allocations":0,"live_bytes":0,"http_verified":false,"performance_qualified":false,
        "product_ready":false,"numerical_qualified":false,"dflash_active":false}),
    )?;
    Ok(())
}
