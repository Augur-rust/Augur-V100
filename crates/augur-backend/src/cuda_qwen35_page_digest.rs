#![cfg(feature = "qwen35-kv8-full-attention")]

//! Compact device-side receipt for a Qwen35 KV8 page.
//!
//! The kernel returns four u64 lanes (32 bytes) instead of copying the whole
//! page back to the host. It is an audit receipt only; callers still need an
//! owner/epoch event proof and must keep the page slot alive until the ordered
//! digest submission completes.

use crate::{CudaBufferView, CudaRuntimeError, OwnedCudaBuffer, OwnedCudaStream};

pub const QWEN35_KV8_PAGE_DIGEST_BYTES: usize = 32;

/// Enqueue a compact digest over the exact input allocation extent.
pub fn submit_page_digest_v1(
    stream: &OwnedCudaStream,
    input: &OwnedCudaBuffer,
    output: &OwnedCudaBuffer,
) -> Result<(), CudaRuntimeError> {
    submit_page_digest_input_view_into_view_v1(
        stream,
        input.view(0, input.len_bytes())?,
        output.view(0, output.len_bytes())?,
    )
}

/// Enqueue a digest into one 32-byte slice of a larger device receipt buffer.
/// The output view is intentionally bounded so one terminal D2H can collect
/// digests for all recurrent/KV state planes.
pub fn submit_page_digest_into_view_v1(
    stream: &OwnedCudaStream,
    input: &OwnedCudaBuffer,
    output: CudaBufferView,
) -> Result<(), CudaRuntimeError> {
    submit_page_digest_input_view_into_view_v1(stream, input.view(0, input.len_bytes())?, output)
}

/// Enqueue a digest over an exact subview of the input allocation.
///
/// This is the active-extent entry point for request-local rings whose
/// physical allocation is larger than the currently valid logical history.
/// The kernel receives the view address and length, so uninitialized physical
/// tail rows are never included in the receipt.
pub fn submit_page_digest_input_view_into_view_v1(
    stream: &OwnedCudaStream,
    input: CudaBufferView,
    output: CudaBufferView,
) -> Result<(), CudaRuntimeError> {
    submit_page_digest_view_policy(stream, input, output, false)
}

/// Opt-in multi-CTA implementation of the EXACT V1 byte receipt. No additional
/// HBM workspace or transfers. Ownership/epoch/lifetime obligations are the
/// same as the serial-CTA entry. A partial enqueue failure requires draining
/// the stream before releasing owners; it is not a healthy retry boundary.
pub fn submit_page_digest_parallel_into_view_v1(
    stream: &OwnedCudaStream,
    input: &OwnedCudaBuffer,
    output: CudaBufferView,
) -> Result<(), CudaRuntimeError> {
    submit_page_digest_parallel_input_view_into_view_v1(
        stream,
        input.view(0, input.len_bytes())?,
        output,
    )
}

/// Parallel-CTA variant of [`submit_page_digest_input_view_into_view_v1`].
pub fn submit_page_digest_parallel_input_view_into_view_v1(
    stream: &OwnedCudaStream,
    input: CudaBufferView,
    output: CudaBufferView,
) -> Result<(), CudaRuntimeError> {
    submit_page_digest_view_policy(stream, input, output, true)
}

fn submit_page_digest_view_policy(
    stream: &OwnedCudaStream,
    input: CudaBufferView,
    output: CudaBufferView,
    parallel: bool,
) -> Result<(), CudaRuntimeError> {
    if input.len_bytes() == 0 || output.len_bytes() != QWEN35_KV8_PAGE_DIGEST_BYTES {
        return Err(CudaRuntimeError::invalid_argument(
            "qwen35 terminal digest extent",
        ));
    }
    if input.device_ordinal() != stream.device_ordinal()
        || output.device_ordinal() != stream.device_ordinal()
    {
        return Err(CudaRuntimeError::device_mismatch());
    }
    if input.overlaps(output, "qwen35 terminal digest")? {
        return Err(CudaRuntimeError::invalid_argument(
            "qwen35 terminal digest alias",
        ));
    }
    let mut args = augur_cuda_sys::AugurCudaQwen35Kv8PageDigestArgsV1::new();
    args.input = input.checked_address("qwen35 terminal digest input")?;
    args.input_len_bytes = input.len_bytes() as u64;
    args.output = output.checked_address("qwen35 terminal digest output")?;
    args.output_len_bytes = QWEN35_KV8_PAGE_DIGEST_BYTES as u64;
    let raw_stream = stream.raw()?;
    let result = unsafe {
        augur_cuda_sys::augur_cuda_qwen35_kv8_page_digest_preflight_v1(
            stream.device_ordinal(),
            raw_stream,
            &args,
        )
    };
    CudaRuntimeError::from_result("cuda_qwen35_terminal_digest_preflight_v1", result)?;
    // SAFETY: preflight checked same-device live disjoint extents. Both paths
    // enqueue only on this stream, retaining neither host pointers nor owners.
    let result = unsafe {
        if parallel {
            augur_cuda_sys::augur_cuda_qwen35_kv8_page_digest_parallel_submit_v1(
                stream.device_ordinal(),
                raw_stream,
                &args,
            )
        } else {
            augur_cuda_sys::augur_cuda_qwen35_kv8_page_digest_submit_v1(
                stream.device_ordinal(),
                raw_stream,
                &args,
            )
        }
    };
    CudaRuntimeError::from_result("cuda_qwen35_terminal_digest_submit_v1", result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{cuda_device_properties, CudaComputeCapability};

    fn expected(bytes: &[u8]) -> [u64; 4] {
        let mut lanes = [
            0xcbf29ce484222325_u64,
            0x84222325cbf29ce4,
            0x9e3779b185ebca87,
            0xd6e8feb86659fd93,
        ];
        for (index, &byte) in bytes.iter().enumerate() {
            let lane = index & 3;
            let mut value = u64::from(byte)
                .wrapping_add(0x9e3779b97f4a7c15)
                .wrapping_add((index as u64).wrapping_mul(0x100000001b3))
                ^ ((lane as u64) << 61);
            value ^= value >> 30;
            value = value.wrapping_mul(0xbf58476d1ce4e5b9);
            value ^= value >> 27;
            value = value.wrapping_mul(0x94d049bb133111eb);
            lanes[lane] ^= value ^ (value >> 31);
        }
        lanes
    }

    #[test]
    fn page_digest_reference_has_fixed_known_answers() {
        assert_eq!(
            expected(&[]),
            [
                0xcbf29ce484222325,
                0x84222325cbf29ce4,
                0x9e3779b185ebca87,
                0xd6e8feb86659fd93
            ]
        );
        // Byte extent and absolute position are part of the digest, including
        // zero-valued bytes; this is not an unordered checksum of byte values.
        assert_ne!(expected(&[0]), expected(&[0, 0]));
        assert_ne!(expected(&[1, 0]), expected(&[0, 1]));
    }

    #[test]
    #[ignore = "requires an exclusive SM70 CUDA window"]
    fn parallel_page_digest_sm70_matches_cpu_and_single_cta_bits() {
        use crate::{cuda_device_allocation_snapshot, cuda_host_boundary_snapshot, OwnedCudaEvent};
        let device = cuda_device_properties(0).expect("CUDA device");
        assert_eq!(device.compute_capability, CudaComputeCapability::SM70);
        let before = cuda_device_allocation_snapshot();
        {
            let stream = OwnedCudaStream::create(0).unwrap();
            let scalar = OwnedCudaBuffer::allocate(0, 48).unwrap();
            let parallel = OwnedCudaBuffer::allocate(0, 48).unwrap();
            let start = OwnedCudaEvent::create(0).unwrap();
            let stop = OwnedCudaEvent::create(0).unwrap();
            let sizes = [
                1,
                3,
                4,
                7,
                31,
                32,
                255,
                256,
                257,
                4095,
                4096,
                4097,
                8191,
                8192,
                8193,
                512 * 1088,
                1024 * 1024 + 3,
                2048 * 25600 * 4,
                196608 * 1088,
            ];
            for bytes in sizes {
                let host: Vec<u8> = (0..bytes)
                    .map(|i| ((i * 37 + i / 251) % 256) as u8)
                    .collect();
                let input = OwnedCudaBuffer::allocate(0, bytes).unwrap();
                input.copy_from_host_sync(&stream, &host).unwrap();
                let mut wanted = [0xa5; 48];
                for (index, word) in expected(&host).into_iter().enumerate() {
                    wanted[8 + index * 8..16 + index * 8].copy_from_slice(&word.to_le_bytes());
                }
                scalar.copy_from_host_sync(&stream, &[0xa5; 48]).unwrap();
                parallel.copy_from_host_sync(&stream, &[0xa5; 48]).unwrap();
                let submit = |is_parallel: bool| {
                    if is_parallel {
                        submit_page_digest_parallel_into_view_v1(
                            &stream,
                            &input,
                            parallel.view(8, 32).unwrap(),
                        )
                    } else {
                        submit_page_digest_into_view_v1(
                            &stream,
                            &input,
                            scalar.view(8, 32).unwrap(),
                        )
                    }
                    .unwrap();
                };
                let check = || {
                    for output in [&scalar, &parallel] {
                        let mut got = [0u8; 48];
                        output.copy_to_host_sync(&stream, &mut got).unwrap();
                        assert_eq!(got, wanted, "bytes={bytes}, CPU/bit/canary");
                    }
                };
                for _ in 0..2 {
                    let allocation = cuda_device_allocation_snapshot();
                    let boundary = cuda_host_boundary_snapshot();
                    submit(false);
                    submit(true);
                    assert_eq!(cuda_host_boundary_snapshot(), boundary);
                    assert_eq!(cuda_device_allocation_snapshot(), allocation);
                    check(); // Repetition detects missing seed reset / stale output.
                }
                assert!(submit_page_digest_parallel_into_view_v1(
                    &stream,
                    &input,
                    parallel.view(8, 31).unwrap()
                )
                .is_err());
                assert!(submit_page_digest_parallel_into_view_v1(
                    &stream,
                    &input,
                    parallel.view(1, 32).unwrap()
                )
                .is_err());
                if bytes >= 32 {
                    assert!(submit_page_digest_parallel_into_view_v1(
                        &stream,
                        &input,
                        input.view(0, 32).unwrap()
                    )
                    .is_err());
                }
                if bytes <= 8193 {
                    let mut got = vec![0u8; bytes];
                    input.copy_to_host_sync(&stream, &mut got).unwrap();
                    assert_eq!(got, host, "input must not be modified");
                }
                println!("PAGE_DIGEST_EXACT_COMPONENT_PASS bytes={bytes} repeated=2 cpu=true canary=true no_submit_allocation_or_host_boundary=true");
                if bytes >= 512 * 1088 {
                    for trial in 0..3 {
                        let order = if trial % 2 == 0 {
                            [false, true]
                        } else {
                            [true, false]
                        };
                        for is_parallel in order {
                            submit(is_parallel);
                            stream.synchronize().unwrap();
                            start.record(&stream).unwrap();
                            for _ in 0..3 {
                                submit(is_parallel);
                            }
                            stop.record(&stream).unwrap();
                            stop.synchronize().unwrap();
                            let ms = start.elapsed_ms(&stop).unwrap() / 3.0;
                            assert!(ms.is_finite() && ms > 0.0);
                            println!("PAGE_DIGEST_COMPONENT_TIMING_ONLY bytes={bytes} trial={trial} parallel={is_parallel} mean_cuda_ms={ms:.6}");
                        }
                        check();
                    }
                }
                // Changed payload must replace, not XOR onto, the old receipt.
                let mut changed = host;
                changed[bytes - 1] ^= 0x81;
                input.copy_from_host_sync(&stream, &changed).unwrap();
                submit_page_digest_parallel_into_view_v1(
                    &stream,
                    &input,
                    parallel.view(8, 32).unwrap(),
                )
                .unwrap();
                let mut got = [0u8; 48];
                parallel.copy_to_host_sync(&stream, &mut got).unwrap();
                let mut changed_wanted = [0xa5; 48];
                for (index, word) in expected(&changed).into_iter().enumerate() {
                    changed_wanted[8 + index * 8..16 + index * 8]
                        .copy_from_slice(&word.to_le_bytes());
                }
                assert_ne!(changed_wanted, wanted);
                assert_eq!(got, changed_wanted);
            }
            stream.synchronize().unwrap();
        }
        let after = cuda_device_allocation_snapshot();
        assert_eq!(after.live_bytes, before.live_bytes);
        assert_eq!(after.live_allocations, before.live_allocations);
        assert_eq!(
            after.successful_allocations - before.successful_allocations,
            after.successful_frees - before.successful_frees
        );
        println!(
            "PAGE_DIGEST_COMPONENT_CLEANUP live_bytes={} allocations={} frees={}",
            after.live_bytes,
            after.successful_allocations - before.successful_allocations,
            after.successful_frees - before.successful_frees
        );
    }

    #[test]
    #[ignore = "requires an exclusive SM70 CUDA window"]
    fn native_page_digest_sm70_matches_cpu_receipt() {
        let device = cuda_device_properties(0).expect("CUDA device");
        assert_eq!(device.compute_capability, CudaComputeCapability::SM70);
        let stream = OwnedCudaStream::create(0).expect("CUDA stream");
        let input_host: Vec<u8> = (0_usize..(512 * 1_088))
            .map(|index| (index.wrapping_mul(37) % 251) as u8)
            .collect();
        let input = OwnedCudaBuffer::allocate(0, input_host.len()).expect("input");
        let output = OwnedCudaBuffer::allocate(0, QWEN35_KV8_PAGE_DIGEST_BYTES).expect("output");
        input
            .copy_from_host_sync(&stream, &input_host)
            .expect("upload");
        submit_page_digest_v1(&stream, &input, &output).expect("digest submit");
        let mut output_host = vec![0_u8; QWEN35_KV8_PAGE_DIGEST_BYTES];
        output
            .copy_to_host_sync(&stream, &mut output_host)
            .expect("digest readback");
        let expected = expected(&input_host);
        let actual = output_host
            .chunks_exact(8)
            .map(|chunk| u64::from_le_bytes(chunk.try_into().expect("digest lane")))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    #[ignore = "requires an exclusive SM70 CUDA window"]
    fn active_page_digest_sm70_ignores_mutated_physical_tail() {
        let device = cuda_device_properties(0).unwrap();
        assert_eq!(device.compute_capability, CudaComputeCapability::SM70);
        let stream = OwnedCudaStream::create(0).unwrap();
        let input = OwnedCudaBuffer::allocate(0, 4096).unwrap();
        let output = OwnedCudaBuffer::allocate(0, QWEN35_KV8_PAGE_DIGEST_BYTES).unwrap();
        for active in [3, 17, 2047, 2048, 2049] {
            let mut bytes = (0..4096).map(|i| (i % 251) as u8).collect::<Vec<_>>();
            let reference = expected(&bytes[8..8 + active]);
            for tail in [0x00, 0xa5] {
                bytes[..8].fill(tail);
                bytes[8 + active..].fill(tail);
                input.copy_from_host_sync(&stream, &bytes).unwrap();
                for parallel in [false, true] {
                    let source = input.view(8, active).unwrap();
                    let destination = output.view(0, QWEN35_KV8_PAGE_DIGEST_BYTES).unwrap();
                    if parallel {
                        submit_page_digest_parallel_input_view_into_view_v1(
                            &stream,
                            source,
                            destination,
                        )
                        .unwrap();
                    } else {
                        submit_page_digest_input_view_into_view_v1(&stream, source, destination)
                            .unwrap();
                    }
                    let mut actual = [0u8; QWEN35_KV8_PAGE_DIGEST_BYTES];
                    output.copy_to_host_sync(&stream, &mut actual).unwrap();
                    let actual = actual
                        .chunks_exact(8)
                        .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
                        .collect::<Vec<_>>();
                    assert_eq!(actual, reference);
                }
            }
        }
        stream.synchronize().unwrap();
    }
}
