//! Exact startup transfer from authenticated descriptors, with bounded host RAM.
use super::{bf16_payload_is_finite, Result};
use augur_backend::{OwnedCudaBuffer, OwnedCudaStream, PinnedH2dLane};
use augur_model_qwen35::CheckpointArtifact;
use std::time::Instant;

const CHUNK_BYTES: usize = 16 * 1024 * 1024;
const LANES: usize = 2;

pub(crate) fn enabled() -> Result<bool> {
    match std::env::var("AUGUR_BF16_STARTUP") {
        Ok(value) if value == "pageable" => Ok(false),
        Ok(value) if value == "streamed" => Ok(true),
        Err(std::env::VarError::NotPresent) => Ok(true),
        _ => Err("AUGUR_BF16_STARTUP must be pageable or streamed".into()),
    }
}

fn validate_geometry(length: usize, chunk_bytes: usize) -> Result<()> {
    if length == 0 || !length.is_multiple_of(2)
        || chunk_bytes == 0 || !chunk_bytes.is_multiple_of(2) || chunk_bytes > CHUNK_BYTES
    {
        return Err("invalid bounded BF16 transfer extent".into());
    }
    Ok(())
}

pub(crate) fn upload_authenticated(
    reader: &mut CheckpointArtifact,
    name: &str,
    stream: &OwnedCudaStream,
) -> Result<(OwnedCudaBuffer, serde_json::Value)> {
    let record = reader.tensor(name)?;
    if record.dtype != "BF16" {
        return Err("streamed tensor requires authenticated BF16 storage".into());
    }
    let length = usize::try_from(record.byte_len()?)?;
    upload_chunks(stream, length, CHUNK_BYTES, |offset, bytes| {
        reader.read_tensor_range_into(name, u64::try_from(offset)?, bytes)?;
        Ok(())
    })
}

fn upload_chunks(
    stream: &OwnedCudaStream,
    length: usize,
    chunk_bytes: usize,
    mut read: impl FnMut(usize, &mut [u8]) -> Result<()>,
) -> Result<(OwnedCudaBuffer, serde_json::Value)> {
    validate_geometry(length, chunk_bytes)?;
    let started = Instant::now();
    let destination = OwnedCudaBuffer::allocate(stream.device_ordinal(), length)?;
    let device_allocate_s = started.elapsed().as_secs_f64();
    let capacity = length.min(chunk_bytes);
    let tick = Instant::now();
    let mut lanes = (0..LANES).map(|_| PinnedH2dLane::new(&destination, stream, capacity))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let pinned_setup_s = tick.elapsed().as_secs_f64();
    let (mut read_s, mut scan_s, mut reuse_wait_s, mut enqueue_s) = (0.0, 0.0, 0.0, 0.0);
    let mut chunks = 0;
    let transfer = (|| -> Result<()> {
        for offset in (0..length).step_by(capacity) {
            let len = capacity.min(length - offset);
            let lane = &mut lanes[chunks % LANES];
            let tick = Instant::now();
            let host = &mut lane.writable()?[..len];
            reuse_wait_s += tick.elapsed().as_secs_f64();
            let tick = Instant::now();
            read(offset, host)?;
            read_s += tick.elapsed().as_secs_f64();
            let tick = Instant::now();
            if !bf16_payload_is_finite(host) {
                return Err("nonfinite streamed BF16 payload".into());
            }
            scan_s += tick.elapsed().as_secs_f64();
            let tick = Instant::now();
            lane.submit(offset, len)?;
            enqueue_s += tick.elapsed().as_secs_f64();
            chunks += 1;
        }
        Ok(())
    })();
    // Drain both slots even after a late read/validation error; DMA owns their bytes.
    let tick = Instant::now();
    let drained: Vec<_> = lanes.iter_mut().map(PinnedH2dLane::wait).collect();
    let drain_s = tick.elapsed().as_secs_f64();
    let tick = Instant::now();
    drop(lanes);
    let pinned_release_s = tick.elapsed().as_secs_f64();
    transfer?;
    for result in drained { result?; }
    let detail = serde_json::json!({
        "mode":"authenticated_bf16_streamed_v1", "original_bytes":length,
        "chunk_bytes":capacity, "chunks":chunks, "pinned_slots":LANES,
        "pinned_host_bytes":capacity * LANES, "whole_tensor_pageable_bytes":0,
        "device_allocate_s":device_allocate_s, "pinned_setup_s":pinned_setup_s,
        "pinned_payload_read_s":read_s, "finite_scan_s":scan_s,
        "reuse_wait_s":reuse_wait_s, "async_enqueue_s":enqueue_s,
        "final_drain_s":drain_s, "pinned_release_s":pinned_release_s,
        "complete_s":started.elapsed().as_secs_f64(), "original_bf16_bytes_preserved":true,
    });
    Ok((destination, detail))
}

#[cfg(test)]
mod tests {
    use super::*;
    use augur_backend::cuda_device_allocation_snapshot;

    #[test]
    fn rejects_empty_odd_and_unbounded_chunks() {
        for (length, chunk) in [(0, 64), (3, 64), (128, 0), (128, 3), (128, CHUNK_BYTES + 2)] {
            assert!(validate_geometry(length, chunk).is_err());
        }
        for (length, chunk) in [(2, 64), (128, 64), (130, 64), (CHUNK_BYTES + 2, CHUNK_BYTES)] {
            validate_geometry(length, chunk).unwrap();
        }
    }

    #[test]
    #[ignore = "exclusive CUDA; exact streamed bytes, tail and late-error retirement"]
    fn streamed_bf16_preserves_bytes_and_drains_late_failures() {
        let before = cuda_device_allocation_snapshot();
        {
            let stream = OwnedCudaStream::create(0).unwrap();
            let source: Vec<u8> = (0..99_u16).flat_map(|i| (i ^ 0x8000).to_le_bytes()).collect();
            let mut offsets = Vec::new();
            let (device, detail) = upload_chunks(&stream, source.len(), 64, |offset, bytes| {
                offsets.push((offset, bytes.len()));
                bytes.copy_from_slice(&source[offset..offset + bytes.len()]);
                Ok(())
            }).unwrap();
            assert_eq!(offsets, [(0, 64), (64, 64), (128, 64), (192, 6)]);
            assert_eq!(detail["chunks"], 4);
            let mut actual = vec![0; source.len()];
            device.copy_to_host_sync(&stream, &mut actual).unwrap();
            assert_eq!(actual, source);
            drop(device);
            for invalid in [0x7f80_u16, 0xff80, 0x7fc1] {
                let result = upload_chunks(&stream, 198, 64, |offset, bytes| {
                    bytes.fill(0);
                    if offset == 192 { bytes[..2].copy_from_slice(&invalid.to_le_bytes()); }
                    Ok(())
                });
                assert!(result.unwrap_err().to_string().contains("nonfinite"));
            }
            let result = upload_chunks(&stream, 198, 64, |offset, bytes| {
                if offset == 192 { return Err("injected late authenticated read failure".into()); }
                bytes.fill(0);
                Ok(())
            });
            assert!(result.unwrap_err().to_string().contains("injected late"));
        }
        let after = cuda_device_allocation_snapshot();
        assert_eq!(after.live_allocations, before.live_allocations);
        assert_eq!(after.live_bytes, before.live_bytes);
        assert_eq!(after.successful_allocations - before.successful_allocations,
            after.successful_frees - before.successful_frees);
    }
}
