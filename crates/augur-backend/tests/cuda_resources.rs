#![cfg(feature = "cuda")]

use augur_cuda_sys::{self, AUGUR_CUDA_STATUS_INVALID_ARGUMENT, AUGUR_CUDA_STATUS_OK};

use augur_backend::{
    cuda_host_boundary_snapshot, CudaD2dBatch, CudaHostBoundarySnapshot, 
    CudaRuntimeErrorKind, OwnedCudaBuffer,
    OwnedCudaEvent, OwnedCudaStream, 
};

#[test]
fn allocate_copy_roundtrip() {
    let stream = OwnedCudaStream::create(0).unwrap();
    let buffer = OwnedCudaBuffer::allocate(0, 4096).unwrap();

    let input = (0..4096).map(|value| value as u8).collect::<Vec<_>>();

    let mut output = vec![0_u8; input.len()];

    buffer.copy_from_host_sync(&stream, &input).unwrap();
    buffer.copy_to_host_sync(&stream, &mut output).unwrap();

    assert_eq!(output, input);
}

#[test]
fn d2d_batch_uses_one_exact_terminal_proof() {
    let stream = OwnedCudaStream::create(0).unwrap();
    let other_stream = OwnedCudaStream::create(0).unwrap();
    let mut source = OwnedCudaBuffer::allocate(0, 64).unwrap();
    let mut destination = OwnedCudaBuffer::allocate(0, 64).unwrap();
    let mut overflow = OwnedCudaBuffer::allocate(0, 64).unwrap();
    let input = (0..64).map(|value| value as u8).collect::<Vec<_>>();
    source.copy_from_host_sync(&stream, &input).unwrap();

    {
        let mut batch = CudaD2dBatch::<2>::new(&stream).unwrap();
        let source_owner = batch.register(&mut source).unwrap();
        let destination_owner = batch.register(&mut destination).unwrap();
        let capacity_error = batch.register(&mut overflow).unwrap_err();
        assert_eq!(capacity_error.kind, CudaRuntimeErrorKind::OutOfBounds);

        batch
            .copy(destination_owner, 0, source_owner, 0, 32)
            .unwrap();
        batch
            .copy(destination_owner, 32, source_owner, 32, 32)
            .unwrap();
        assert_eq!(batch.owner_count(), 2);
        assert_eq!(batch.submission_count(), 2);

        let receipt = batch.completion_receipt().unwrap();
        assert_eq!(
            batch
                .copy(destination_owner, 0, source_owner, 0, 1)
                .unwrap_err()
                .kind,
            CudaRuntimeErrorKind::InvalidMetadata
        );
        let wrong_stream_error = other_stream.synchronize_submission(receipt).unwrap_err();
        assert_eq!(
            wrong_stream_error.kind,
            CudaRuntimeErrorKind::InvalidMetadata
        );

        let before = cuda_host_boundary_snapshot();
        let proof = stream.synchronize_submission(receipt).unwrap();
        let delta = cuda_host_boundary_snapshot()
            .checked_delta_since(before)
            .unwrap();
        assert_eq!(
            delta,
            CudaHostBoundarySnapshot {
                host_waits: 1,
                stream_synchronizations: 1,
                ..CudaHostBoundarySnapshot::default()
            }
        );
        batch.release_after_completion(&proof).unwrap();
        assert_eq!(
            batch.release_after_completion(&proof).unwrap_err().kind,
            CudaRuntimeErrorKind::InvalidMetadata
        );
    }

    let mut output = [0_u8; 64];
    destination.copy_to_host_sync(&stream, &mut output).unwrap();
    assert_eq!(output.as_slice(), input.as_slice());
}

#[test]
fn d2d_batch_registered_view_requires_and_retains_its_owner() {
    let stream = OwnedCudaStream::create(0).unwrap();
    let mut source = OwnedCudaBuffer::allocate(0, 64).unwrap();
    let mut destination = OwnedCudaBuffer::allocate(0, 64).unwrap();
    let unregistered = OwnedCudaBuffer::allocate(0, 64).unwrap();
    let input = (0..64).map(|value| value as u8).collect::<Vec<_>>();
    source.copy_from_host_sync(&stream, &input).unwrap();
    let source_view = source.view(8, 32).unwrap();
    let unregistered_view = unregistered.view(0, 16).unwrap();

    {
        let mut batch = CudaD2dBatch::<2>::new(&stream).unwrap();
        batch.register(&mut source).unwrap();
        let destination_owner = batch.register(&mut destination).unwrap();
        assert_eq!(
            batch
                .copy_from_registered_view(destination_owner, 0, unregistered_view)
                .unwrap_err()
                .kind,
            CudaRuntimeErrorKind::InvalidMetadata
        );
        batch
            .copy_from_registered_view(destination_owner, 16, source_view)
            .unwrap();

        let mut output = [0_u8; 32];
        let proof = batch
            .copy_owner_range_to_host_sync(destination_owner, 16, &mut output)
            .unwrap();
        assert_eq!(output.as_slice(), &input[8..40]);
        batch.release_after_completion(&proof).unwrap();
    }
}

#[test]
fn d2d_batch_registered_view_preflight_is_side_effect_free() {
    let stream = OwnedCudaStream::create(0).unwrap();
    let mut source = OwnedCudaBuffer::allocate(0, 64).unwrap();
    let mut destination = OwnedCudaBuffer::allocate(0, 64).unwrap();
    let unregistered = OwnedCudaBuffer::allocate(0, 64).unwrap();
    let source_view = source.view(0, 64).unwrap();
    let unregistered_view = unregistered.view(0, 64).unwrap();

    {
        let mut batch = CudaD2dBatch::<2>::new(&stream).unwrap();
        batch.register(&mut source).unwrap();
        batch.register(&mut destination).unwrap();
        batch.validate_registered_view(source_view).unwrap();
        assert_eq!(
            batch
                .validate_registered_view(unregistered_view)
                .unwrap_err()
                .kind,
            CudaRuntimeErrorKind::InvalidMetadata
        );
        assert_eq!(batch.submission_count(), 0);
    }

    source.copy_from_host_sync(&stream, &[1_u8; 64]).unwrap();
    destination
        .copy_from_host_sync(&stream, &[2_u8; 64])
        .unwrap();
}

#[test]
fn d2d_batch_registered_views_fill_the_m289_owner_bound() {
    const SOURCE_COUNT: usize = 43 + 21 * 2 + 20;

    let stream = OwnedCudaStream::create(0).unwrap();
    let mut sources = (0..SOURCE_COUNT)
        .map(|index| {
            let source = OwnedCudaBuffer::allocate(0, 4).unwrap();
            source
                .copy_from_host_sync(&stream, &(index as u32).to_le_bytes())
                .unwrap();
            source
        })
        .collect::<Vec<_>>();
    let source_views = sources
        .iter()
        .map(|source| source.view(0, 4).unwrap())
        .collect::<Vec<_>>();
    let mut destination = OwnedCudaBuffer::allocate(0, SOURCE_COUNT * 4).unwrap();

    {
        let mut batch = CudaD2dBatch::<106>::new(&stream).unwrap();
        let destination_owner = batch.register(&mut destination).unwrap();
        for source in &mut sources {
            batch.register(source).unwrap();
        }
        assert_eq!(batch.owner_count(), 106);
        for (index, source) in source_views.into_iter().enumerate() {
            batch
                .copy_from_registered_view(destination_owner, index * 4, source)
                .unwrap();
        }

        let mut output = vec![0_u8; SOURCE_COUNT * 4];
        let proof = batch
            .copy_owner_range_to_host_sync(destination_owner, 0, &mut output)
            .unwrap();
        for (index, value) in output.chunks_exact(4).enumerate() {
            assert_eq!(u32::from_le_bytes(value.try_into().unwrap()), index as u32);
        }
        batch.release_after_completion(&proof).unwrap();
    }
}

#[test]
fn d2d_batch_external_owners_reuse_an_existing_stream_completion() {
    let stream = OwnedCudaStream::create(0).unwrap();
    let source = OwnedCudaBuffer::allocate(0, 64).unwrap();
    let destination = OwnedCudaBuffer::allocate(0, 64).unwrap();
    let input = (0..64).map(|value| value as u8).collect::<Vec<_>>();
    source.copy_from_host_sync(&stream, &input).unwrap();
    let source_view = source.view(0, input.len()).unwrap();

    {
        let mut batch = CudaD2dBatch::<2>::new(&stream).unwrap();
        // SAFETY: both allocations remain live and unchanged through the
        // exact-stream completion and batch release below.
        let destination_owner = unsafe { batch.register_external(&destination) }.unwrap();
        // SAFETY: same step-lifetime owner contract as the destination.
        unsafe { batch.register_external(&source) }.unwrap();
        batch
            .copy_from_registered_view(destination_owner, 0, source_view)
            .unwrap();
        assert_eq!(
            destination.view(0, input.len()).unwrap_err().kind,
            CudaRuntimeErrorKind::InvalidMetadata
        );

        let before = cuda_host_boundary_snapshot();
        stream.synchronize().unwrap();
        let after_completion = cuda_host_boundary_snapshot();
        assert_eq!(
            after_completion.checked_delta_since(before).unwrap(),
            CudaHostBoundarySnapshot {
                host_waits: 1,
                stream_synchronizations: 1,
                ..CudaHostBoundarySnapshot::default()
            }
        );
        // SAFETY: the successful blocking operation above completed this
        // batch's exact stream after its only D2D submission.
        unsafe { batch.release_after_external_stream_completion() }.unwrap();
        assert_eq!(cuda_host_boundary_snapshot(), after_completion);
    }

    let mut output = [0_u8; 64];
    destination.copy_to_host_sync(&stream, &mut output).unwrap();
    assert_eq!(output.as_slice(), input.as_slice());
}

#[test]
fn d2d_batch_rejects_zero_capacity_foreign_handles_and_overlap() {
    let stream = OwnedCudaStream::create(0).unwrap();
    assert_eq!(
        CudaD2dBatch::<0>::new(&stream).unwrap_err().kind,
        CudaRuntimeErrorKind::InvalidDimensions
    );

    let mut first_owner = OwnedCudaBuffer::allocate(0, 64).unwrap();
    let mut second_owner = OwnedCudaBuffer::allocate(0, 64).unwrap();
    let mut first_batch = CudaD2dBatch::<1>::new(&stream).unwrap();
    let foreign = first_batch.register(&mut first_owner).unwrap();
    let mut second_batch = CudaD2dBatch::<1>::new(&stream).unwrap();
    let local = second_batch.register(&mut second_owner).unwrap();
    assert_eq!(
        second_batch
            .copy(local, 0, foreign, 0, 16)
            .unwrap_err()
            .kind,
        CudaRuntimeErrorKind::InvalidMetadata
    );
    assert_eq!(
        first_batch.completion_receipt().unwrap_err().kind,
        CudaRuntimeErrorKind::InvalidMetadata
    );
    assert_eq!(
        second_batch.copy(local, 8, local, 0, 16).unwrap_err().kind,
        CudaRuntimeErrorKind::OverlappingDeviceRanges
    );
    assert_eq!(
        second_batch.completion_receipt().unwrap_err().kind,
        CudaRuntimeErrorKind::InvalidMetadata
    );
}

#[test]
fn d2d_batch_reuses_existing_d2h_completion_boundary() {
    let stream = OwnedCudaStream::create(0).unwrap();
    let mut source = OwnedCudaBuffer::allocate(0, 64).unwrap();
    let mut destination = OwnedCudaBuffer::allocate(0, 64).unwrap();
    let terminal = OwnedCudaBuffer::allocate(0, 4).unwrap();
    let input = (0..64).map(|value| value as u8).collect::<Vec<_>>();
    source.copy_from_host_sync(&stream, &input).unwrap();
    terminal
        .copy_from_host_sync(&stream, &[1, 2, 3, 4])
        .unwrap();

    {
        let mut batch = CudaD2dBatch::<2>::new(&stream).unwrap();
        let source_owner = batch.register(&mut source).unwrap();
        let destination_owner = batch.register(&mut destination).unwrap();
        batch
            .copy(destination_owner, 0, source_owner, 0, input.len())
            .unwrap();
        let receipt = batch.completion_receipt().unwrap();

        let before = cuda_host_boundary_snapshot();
        let mut terminal_host = [0_u8; 4];
        let proof = terminal
            .copy_to_host_sync_with_submission_proof(&stream, &mut terminal_host, receipt)
            .unwrap();
        assert_eq!(terminal_host, [1, 2, 3, 4]);
        assert_eq!(
            cuda_host_boundary_snapshot()
                .checked_delta_since(before)
                .unwrap(),
            CudaHostBoundarySnapshot {
                host_waits: 1,
                synchronous_d2h_copies: 1,
                synchronous_d2h_bytes: 4,
                ..CudaHostBoundarySnapshot::default()
            }
        );
        batch.release_after_completion(&proof).unwrap();
    }

    let mut output = [0_u8; 64];
    destination.copy_to_host_sync(&stream, &mut output).unwrap();
    assert_eq!(output.as_slice(), input.as_slice());
}

#[test]
fn d2d_batch_accepts_a_later_same_stream_completion_proof() {
    let stream = OwnedCudaStream::create(0).unwrap();
    let mut source = OwnedCudaBuffer::allocate(0, 64).unwrap();
    let mut destination = OwnedCudaBuffer::allocate(0, 64).unwrap();
    let terminal = OwnedCudaBuffer::allocate(0, 4).unwrap();
    source.copy_from_host_sync(&stream, &[7_u8; 64]).unwrap();
    terminal
        .copy_from_host_sync(&stream, &[1, 2, 3, 4])
        .unwrap();

    let mut batch = CudaD2dBatch::<2>::new(&stream).unwrap();
    let source_owner = batch.register(&mut source).unwrap();
    let destination_owner = batch.register(&mut destination).unwrap();
    batch
        .copy(destination_owner, 0, source_owner, 0, 64)
        .unwrap();

    let _ = batch.completion_receipt().unwrap();
    let later_receipt = stream.submission_receipt().unwrap();
    let proof = terminal
        .copy_to_host_sync_with_submission_proof(&stream, &mut [0_u8; 4], later_receipt)
        .unwrap();
    batch.release_after_covering_completion(&proof).unwrap();

    let mut output = [0_u8; 64];
    destination.copy_to_host_sync(&stream, &mut output).unwrap();
    assert_eq!(output, [7_u8; 64]);
}

#[test]
fn d2d_batch_rejects_a_proof_obtained_before_the_batch_was_sealed() {
    let stream = OwnedCudaStream::create(0).unwrap();
    let mut source = OwnedCudaBuffer::allocate(0, 64).unwrap();
    let mut destination = OwnedCudaBuffer::allocate(0, 64).unwrap();
    let terminal = OwnedCudaBuffer::allocate(0, 4).unwrap();
    source.copy_from_host_sync(&stream, &[7_u8; 64]).unwrap();
    terminal
        .copy_from_host_sync(&stream, &[1, 2, 3, 4])
        .unwrap();

    let mut batch = CudaD2dBatch::<2>::new(&stream).unwrap();
    let source_owner = batch.register(&mut source).unwrap();
    let destination_owner = batch.register(&mut destination).unwrap();

    let premature_receipt = stream.submission_receipt().unwrap();
    let premature_proof = terminal
        .copy_to_host_sync_with_submission_proof(&stream, &mut [0_u8; 4], premature_receipt)
        .unwrap();
    batch
        .copy(destination_owner, 0, source_owner, 0, 64)
        .unwrap();
    assert_eq!(
        batch
            .release_after_covering_completion(&premature_proof)
            .unwrap_err()
            .kind,
        CudaRuntimeErrorKind::InvalidMetadata
    );

    let _ = batch.completion_receipt().unwrap();
    assert_eq!(
        batch
            .release_after_covering_completion(&premature_proof)
            .unwrap_err()
            .kind,
        CudaRuntimeErrorKind::InvalidMetadata
    );
    let terminal_receipt = stream.submission_receipt().unwrap();
    let terminal_proof = terminal
        .copy_to_host_sync_with_submission_proof(&stream, &mut [0_u8; 4], terminal_receipt)
        .unwrap();
    let before_release = cuda_host_boundary_snapshot();
    batch
        .release_after_covering_completion(&terminal_proof)
        .unwrap();
    assert_eq!(
        cuda_host_boundary_snapshot()
            .checked_delta_since(before_release)
            .unwrap(),
        CudaHostBoundarySnapshot::default()
    );

    let mut output = [0_u8; 64];
    destination.copy_to_host_sync(&stream, &mut output).unwrap();
    assert_eq!(output, [7_u8; 64]);
}

#[test]
fn d2d_batch_terminal_d2h_can_read_a_pending_owner() {
    let stream = OwnedCudaStream::create(0).unwrap();
    let mut source = OwnedCudaBuffer::allocate(0, 64).unwrap();
    let mut destination = OwnedCudaBuffer::allocate(0, 64).unwrap();
    let input = (0..64).map(|value| value as u8).collect::<Vec<_>>();
    source.copy_from_host_sync(&stream, &input).unwrap();

    {
        let mut batch = CudaD2dBatch::<2>::new(&stream).unwrap();
        let source_owner = batch.register(&mut source).unwrap();
        let destination_owner = batch.register(&mut destination).unwrap();
        batch
            .copy(destination_owner, 0, source_owner, 0, input.len())
            .unwrap();

        assert_eq!(
            batch
                .copy_owner_range_to_host_sync(destination_owner, 0, &mut [])
                .unwrap_err()
                .kind,
            CudaRuntimeErrorKind::OutOfBounds
        );
        let before = cuda_host_boundary_snapshot();
        let mut output = [0_u8; 32];
        let proof = batch
            .copy_owner_range_to_host_sync(destination_owner, 16, &mut output)
            .unwrap();
        assert_eq!(output.as_slice(), &input[16..48]);
        assert_eq!(
            cuda_host_boundary_snapshot()
                .checked_delta_since(before)
                .unwrap(),
            CudaHostBoundarySnapshot {
                host_waits: 1,
                synchronous_d2h_copies: 1,
                synchronous_d2h_bytes: 32,
                ..CudaHostBoundarySnapshot::default()
            }
        );
        assert_eq!(
            batch
                .copy(destination_owner, 0, source_owner, 0, 1)
                .unwrap_err()
                .kind,
            CudaRuntimeErrorKind::InvalidMetadata
        );
        batch.release_after_completion(&proof).unwrap();
    }

    let mut output = [0_u8; 64];
    destination.copy_to_host_sync(&stream, &mut output).unwrap();
    assert_eq!(output.as_slice(), input.as_slice());
}

#[test]
fn d2d_batch_reuses_existing_event_completion_boundary() {
    let stream = OwnedCudaStream::create(0).unwrap();
    let event = OwnedCudaEvent::create(0).unwrap();
    let mut source = OwnedCudaBuffer::allocate(0, 64).unwrap();
    let mut destination = OwnedCudaBuffer::allocate(0, 64).unwrap();
    source.copy_from_host_sync(&stream, &[7_u8; 64]).unwrap();

    let mut batch = CudaD2dBatch::<2>::new(&stream).unwrap();
    let source_owner = batch.register(&mut source).unwrap();
    let destination_owner = batch.register(&mut destination).unwrap();
    batch
        .copy(destination_owner, 0, source_owner, 0, 64)
        .unwrap();
    let receipt = batch.completion_receipt().unwrap();

    let before = cuda_host_boundary_snapshot();
    let proof = event
        .record_and_synchronize_submission(&stream, receipt)
        .unwrap();
    assert_eq!(
        cuda_host_boundary_snapshot()
            .checked_delta_since(before)
            .unwrap(),
        CudaHostBoundarySnapshot {
            host_waits: 1,
            event_synchronizations: 1,
            ..CudaHostBoundarySnapshot::default()
        }
    );
    batch.release_after_completion(&proof).unwrap();
}

#[test]
fn d2d_completion_failure_does_not_release_batch() {
    let stream = OwnedCudaStream::create(0).unwrap();
    let mut source = OwnedCudaBuffer::allocate(0, 64).unwrap();
    let mut destination = OwnedCudaBuffer::allocate(0, 64).unwrap();
    let terminal = OwnedCudaBuffer::allocate(0, 4).unwrap();
    source.copy_from_host_sync(&stream, &[9_u8; 64]).unwrap();

    let mut batch = CudaD2dBatch::<2>::new(&stream).unwrap();
    let source_owner = batch.register(&mut source).unwrap();
    let destination_owner = batch.register(&mut destination).unwrap();
    batch
        .copy(destination_owner, 0, source_owner, 0, 64)
        .unwrap();
    let receipt = batch.completion_receipt().unwrap();
    assert_eq!(
        terminal
            .copy_to_host_sync_with_submission_proof(&stream, &mut [0_u8; 5], receipt)
            .unwrap_err()
            .kind,
        CudaRuntimeErrorKind::OutOfBounds
    );

    let proof = stream.synchronize_submission(receipt).unwrap();
    batch.release_after_completion(&proof).unwrap();
}

#[test]
fn view_copy_is_synchronous_and_counted() {
    let stream = OwnedCudaStream::create(0).unwrap();
    let buffer = OwnedCudaBuffer::allocate(0, 64).unwrap();
    let input = (0..64).map(|value| value as u8).collect::<Vec<_>>();
    buffer.copy_from_host_sync(&stream, &input).unwrap();

    let before = cuda_host_boundary_snapshot();
    let mut output = [0_u8; 16];
    buffer
        .view(12, output.len())
        .unwrap()
        .copy_to_host_sync(&stream, &mut output)
        .unwrap();
    let delta = cuda_host_boundary_snapshot()
        .checked_delta_since(before)
        .unwrap();

    assert_eq!(output, input[12..28]);
    assert_eq!(
        delta,
        CudaHostBoundarySnapshot {
            host_waits: 1,
            synchronous_d2h_copies: 1,
            synchronous_d2h_bytes: 16,
            ..CudaHostBoundarySnapshot::default()
        }
    );
}

#[test]
fn zero_length_buffer_has_no_device_allocation() {
    let buffer = OwnedCudaBuffer::allocate(0, 0).unwrap();

    assert!(buffer.is_empty());
    assert_eq!(buffer.len_bytes(), 0);
    assert!(buffer.device_ptr().is_none());
}

#[test]
fn oversized_copy_is_rejected() {
    let stream = OwnedCudaStream::create(0).unwrap();
    let buffer = OwnedCudaBuffer::allocate(0, 16).unwrap();

    let input = [0_u8; 17];

    let error = buffer.copy_from_host_sync(&stream, &input).unwrap_err();

    assert_eq!(error.kind, CudaRuntimeErrorKind::OutOfBounds);
    assert_eq!(error.status, AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
}

#[test]
fn negative_device_ordinal_is_rejected() {
    let error = OwnedCudaBuffer::allocate(-1, 16).unwrap_err();

    assert_eq!(error.kind, CudaRuntimeErrorKind::InvalidDeviceOrdinal);
}

#[test]
fn explicit_close_is_one_shot() {
    let mut stream = OwnedCudaStream::create(0).unwrap();

    stream.close().unwrap();

    let error = stream.close().unwrap_err();
    assert_eq!(error.kind, CudaRuntimeErrorKind::ClosedResource);
}

#[test]
fn stream_wait_event_rejects_closed_resources() {
    let mut stream = OwnedCudaStream::create(0).unwrap();
    let mut event = OwnedCudaEvent::create(0).unwrap();
    event.close().unwrap();

    let error = stream.wait_event(&event).unwrap_err();
    assert_eq!(error.kind, CudaRuntimeErrorKind::ClosedResource);

    let event = OwnedCudaEvent::create(0).unwrap();
    stream.close().unwrap();
    let error = stream.wait_event(&event).unwrap_err();
    assert_eq!(error.kind, CudaRuntimeErrorKind::ClosedResource);
}

#[test]
#[ignore = "requires at least two CUDA devices; run explicitly on multi-GPU hosts"]
fn host_side_ordinal_mismatch_is_rejected() {
    let stream = OwnedCudaStream::create(0).unwrap();

    let mut count = 0;
    let result = unsafe { augur_cuda_sys::augur_cuda_device_count(&mut count) };
    assert_eq!(result.status, AUGUR_CUDA_STATUS_OK);

    assert!(
        count >= 2,
        "multi-GPU probe requires at least two CUDA devices; detected {count}"
    );

    let buffer = OwnedCudaBuffer::allocate(1, 16).unwrap();

    let error = buffer
        .copy_from_host_sync(&stream, &[0_u8; 16])
        .unwrap_err();

    assert_eq!(error.operation, "cuda_memcpy_h2d");
    assert_eq!(error.kind, CudaRuntimeErrorKind::DeviceMismatch);
    assert_eq!(error.status, AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
    assert_eq!(error.raw_cuda_status, 0);
}
