#![cfg(feature = "cuda")]

use augur_backend::{
    cuda_host_boundary_snapshot, CudaHostBoundarySnapshot, CudaRuntimeErrorKind, OwnedCudaBuffer,
    OwnedCudaStream, OwnedPinnedH2dBuffer,
};

#[test]
fn pinned_owner_submits_without_a_stream_wide_h2d_wait() {
    let stream = OwnedCudaStream::create(0).unwrap();
    let mut stage = OwnedPinnedH2dBuffer::allocate(0, 64).unwrap();
    let destination = OwnedCudaBuffer::allocate(0, 64).unwrap();

    for generation in 0..2_u8 {
        let expected = (0..64)
            .map(|index| (index as u8).wrapping_add(generation))
            .collect::<Vec<_>>();
        stage.writable().unwrap().copy_from_slice(&expected);
        stage.submit(&stream).unwrap();
        assert!(stage.pending());

        // SAFETY: `stage`, `destination`, and `stream` remain live until the
        // synchronous D2H copy below completes the upload and its consumer.
        unsafe {
            destination
                .copy_from_device_async_unchecked(&stream, 0, stage.device_view().unwrap())
                .unwrap();
        }
        let mut actual = vec![0_u8; 64];
        let proof = destination
            .copy_to_host_sync_with_submission_proof(
                &stream,
                &mut actual,
                stream.submission_receipt().unwrap(),
            )
            .unwrap();

        // The D2H above is the externally owned terminal wait and completes
        // the upload plus its consumer on the same stream.
        stage.release_after_covering_completion(&proof).unwrap();
        assert!(!stage.pending());
        assert_eq!(actual, expected);
    }
}

#[test]
fn pinned_owner_rejects_stale_and_wrong_stream_completion_proofs() {
    let stream = OwnedCudaStream::create(0).unwrap();
    let other_stream = OwnedCudaStream::create(0).unwrap();
    let mut stage = OwnedPinnedH2dBuffer::allocate(0, 64).unwrap();
    let terminal = OwnedCudaBuffer::allocate(0, 4).unwrap();
    let other_terminal = OwnedCudaBuffer::allocate(0, 4).unwrap();
    terminal
        .copy_from_host_sync(&stream, &[1, 2, 3, 4])
        .unwrap();
    other_terminal
        .copy_from_host_sync(&other_stream, &[1, 2, 3, 4])
        .unwrap();

    let stale_receipt = stream.submission_receipt().unwrap();
    stage.writable().unwrap().fill(7);
    stage.submit(&stream).unwrap();

    let stale_proof = terminal
        .copy_to_host_sync_with_submission_proof(&stream, &mut [0_u8; 4], stale_receipt)
        .unwrap();
    assert_eq!(
        stage
            .release_after_covering_completion(&stale_proof)
            .unwrap_err()
            .kind,
        CudaRuntimeErrorKind::InvalidMetadata
    );
    assert!(stage.pending());

    let wrong_stream_receipt = other_stream.submission_receipt().unwrap();
    let wrong_stream_proof = other_terminal
        .copy_to_host_sync_with_submission_proof(
            &other_stream,
            &mut [0_u8; 4],
            wrong_stream_receipt,
        )
        .unwrap();
    assert_eq!(
        stage
            .release_after_covering_completion(&wrong_stream_proof)
            .unwrap_err()
            .kind,
        CudaRuntimeErrorKind::InvalidMetadata
    );
    assert!(stage.pending());

    let terminal_receipt = stream.submission_receipt().unwrap();
    let terminal_proof = terminal
        .copy_to_host_sync_with_submission_proof(&stream, &mut [0_u8; 4], terminal_receipt)
        .unwrap();
    let before_release = cuda_host_boundary_snapshot();
    stage
        .release_after_covering_completion(&terminal_proof)
        .unwrap();
    assert_eq!(
        cuda_host_boundary_snapshot()
            .checked_delta_since(before_release)
            .unwrap(),
        CudaHostBoundarySnapshot::default()
    );
    assert!(!stage.pending());
}
