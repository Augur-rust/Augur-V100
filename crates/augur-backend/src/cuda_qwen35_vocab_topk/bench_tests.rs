//! Diagnostic comparison only: this is not target/HTTP throughput.
use super::*;
use crate::{cuda_device_allocation_snapshot, cuda_host_boundary_snapshot, OwnedCudaBuffer};
use augur_core::PhysicalLayoutId;
use std::time::Instant;

#[test]
#[ignore = "requires an idle SM70 device; component wall only"]
fn eight_scalar_vs_batched8_top16() -> Result<(), Box<dyn std::error::Error>> {
    const ROWS: usize = 8;
    const K: usize = 16;
    const ITERS: usize = 32;
    let vocab = QWEN35_DFLASH_ACCEPTANCE_VOCAB;
    let initial = cuda_device_allocation_snapshot();
    {
        let stream = OwnedCudaStream::create(0)?;
        let input = OwnedCudaBuffer::allocate(0, ROWS * vocab * 4)?;
        let ids = OwnedCudaBuffer::allocate(0, ROWS * K * 4)?;
        let scores = OwnedCudaBuffer::allocate(0, ROWS * K * 4)?;
        let view = |offset, elements| -> Result<CudaF32View, CudaRuntimeError> {
            CudaF32View::new(
                input.view(offset * 4, elements * 4)?,
                PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                elements,
            )
        };
        let batch = CudaQwen35VocabTopkArgsV1::bind(
            &stream,
            view(0, ROWS * vocab)?,
            ids.view(0, ROWS * K * 4)?,
            scores.view(0, ROWS * K * 4)?,
            ROWS,
            K,
        )?;
        let scalar = (0..ROWS)
            .map(|r| {
                CudaQwen35VocabTopkArgsV1::bind(
                    &stream,
                    view(r * vocab, vocab)?,
                    ids.view(r * K * 4, K * 4)?,
                    scores.view(r * K * 4, K * 4)?,
                    1,
                    K,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let submit = |batched: bool| -> Result<(), CudaRuntimeError> {
            if batched {
                batch.submit(&stream)?;
            } else {
                for row in &scalar {
                    row.submit(&stream)?;
                }
            }
            Ok(())
        };
        let read = || -> Result<(Vec<u8>, Vec<u8>), CudaRuntimeError> {
            let mut a = vec![0; ROWS * K * 4];
            let mut b = a.clone();
            ids.copy_to_host_sync(&stream, &mut a)?;
            scores.copy_to_host_sync(&stream, &mut b)?;
            Ok((a, b))
        };
        // Finite ties, signed zero/subnormals, and invalid rows are all compared.
        // A separate existing test checks B8 against a complete independent sort.
        let mut seed = 0x182f_c537_u32;
        let mut values = vec![0.0_f32; ROWS * vocab];
        for v in &mut values {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            *v = ((seed % 65536) as i32 - 32768) as f32 * 0.03125;
        }
        values[0..vocab].fill(-1.0);
        for (i, bits) in [0, 0x80000000, 1, 2, 0x80000001, 0x007fffff, 0x00800000]
            .iter()
            .enumerate()
        {
            values[200000 + i] = f32::from_bits(*bits);
        }
        for invalid in [true, false] {
            values[vocab - 1] = if invalid { f32::NAN } else { f32::MAX };
            values[2 * vocab - 1] = if invalid { f32::INFINITY } else { f32::MAX };
            values[3 * vocab - 1] = if invalid { f32::NEG_INFINITY } else { f32::MAX };
            let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
            input.copy_from_host_sync(&stream, &bytes)?;
            submit(false)?;
            let reference = read()?;
            submit(true)?;
            assert_eq!(reference, read()?, "batched top16 bytes differ");
            let mut input_after = vec![0; bytes.len()];
            input.copy_to_host_sync(&stream, &mut input_after)?;
            assert_eq!(input_after, bytes, "logits mutated");
        }
        for _ in 0..4 {
            submit(false)?;
            submit(true)?;
        }
        stream.synchronize()?;
        // Two ABBA blocks: the clock includes submission and exactly one final
        // drain per sample. No readbacks, allocations, or source uploads inside.
        let before = cuda_device_allocation_snapshot();
        let boundary = cuda_host_boundary_snapshot();
        for (sample, batched) in [false, true, true, false, false, true, true, false]
            .into_iter()
            .enumerate()
        {
            let start = Instant::now();
            for _ in 0..ITERS {
                submit(batched)?;
            }
            stream.synchronize()?;
            println!(
                "TOPK_COMPONENT sample={} batched={} rounds={} wall_s={:.9}",
                sample,
                batched,
                ITERS,
                start.elapsed().as_secs_f64()
            );
        }
        let delta = cuda_host_boundary_snapshot()
            .checked_delta_since(boundary)
            .ok_or("counter regression")?;
        assert_eq!(delta.stream_synchronizations, 8);
        assert_eq!(delta.host_waits, 8);
        assert_eq!(delta.synchronous_h2d_bytes, 0);
        assert_eq!(delta.synchronous_d2h_bytes, 0);
        assert_eq!(delta.synchronous_h2d_copies, 0);
        assert_eq!(delta.synchronous_d2h_copies, 0);
        assert_eq!(delta.event_synchronizations, 0);
        let after = cuda_device_allocation_snapshot();
        assert_eq!(before.successful_allocations, after.successful_allocations);
        assert_eq!(before.successful_frees, after.successful_frees);
    }
    let end = cuda_device_allocation_snapshot();
    assert_eq!(end.live_bytes, initial.live_bytes);
    assert_eq!(end.live_allocations, initial.live_allocations);
    println!("TOPK_COMPONENT exact=true allocations_closed=true target_TPS_claim=false");
    Ok(())
}
