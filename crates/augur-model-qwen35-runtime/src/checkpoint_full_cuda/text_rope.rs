use super::{Buffer, OwnedCudaBuffer, OwnedCudaStream, Rc, Result};

/// Host-only originals of the shared buffers overwritten by image MRoPE.
pub(super) struct RetainedTextRope {
    cos: Vec<u8>,
    sin: Vec<u8>,
}

impl RetainedTextRope {
    pub(super) fn from_f32(cos: &[f32], sin: &[f32]) -> Result<Self> {
        if cos.is_empty()
            || cos.len() % 32 != 0
            || cos.len() != sin.len()
            || cos.iter().chain(sin).any(|value| !value.is_finite())
        {
            return Err("invalid retained text RoPE tables".into());
        }
        Ok(Self {
            cos: cos.iter().flat_map(|value| value.to_le_bytes()).collect(),
            sin: sin.iter().flat_map(|value| value.to_le_bytes()).collect(),
        })
    }

    pub(super) fn upload(&self, stream: &OwnedCudaStream) -> Result<(Buffer, Buffer)> {
        let cos = OwnedCudaBuffer::allocate(stream.device_ordinal(), self.cos.len())?;
        let sin = OwnedCudaBuffer::allocate(stream.device_ordinal(), self.sin.len())?;
        self.restore_into(stream, &cos, &sin)?;
        Ok((Rc::new(cos), Rc::new(sin)))
    }

    pub(super) fn restore_into(
        &self,
        stream: &OwnedCudaStream,
        cos: &OwnedCudaBuffer,
        sin: &OwnedCudaBuffer,
    ) -> Result<()> {
        // Validate both destinations before the first mutation. Transfer failure
        // still propagates to reset_empty, which must keep the target poisoned.
        if cos.len_bytes() != self.cos.len() || sin.len_bytes() != self.sin.len() {
            return Err("retained text RoPE destination bounds".into());
        }
        cos.copy_from_host_sync(stream, &self.cos)?;
        sin.copy_from_host_sync(stream, &self.sin)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use augur_backend::cuda_device_allocation_snapshot;

    fn original(capacity: usize, absolute_start: usize, theta: f64) -> (Vec<f32>, Vec<f32>) {
        let mut cos = Vec::with_capacity(capacity * 32);
        let mut sin = Vec::with_capacity(capacity * 32);
        for position in absolute_start..absolute_start + capacity {
            for pair in 0..32 {
                let inverse = theta.powf(-2.0 * pair as f64 / 64.0) as f32;
                let phase = position as f32 * inverse;
                cos.push(phase.cos());
                sin.push(phase.sin());
            }
        }
        (cos, sin)
    }

    #[test]
    fn retained_text_bytes_match_the_original_restore_math() {
        for (capacity, start, theta) in [
            (1, 0, 10_000_000.0),
            (64, 2000, 10_000_000.0),
            (3, 262141, 10_000_000.0),
            (17, 1023, 1_000_000.0),
        ] {
            let (cos, sin) = original(capacity, start, theta);
            let retained = RetainedTextRope::from_f32(&cos, &sin).unwrap();
            let mut expected_cos = Vec::new();
            let mut expected_sin = Vec::new();
            for position in start..start + capacity {
                for pair in 0..32 {
                    let inverse = theta.powf(-2.0 * pair as f64 / 64.0) as f32;
                    let phase = position as f32 * inverse;
                    expected_cos.extend_from_slice(&phase.cos().to_le_bytes());
                    expected_sin.extend_from_slice(&phase.sin().to_le_bytes());
                }
            }
            assert_eq!(retained.cos, expected_cos);
            assert_eq!(retained.sin, expected_sin);
            assert_eq!(retained.cos.len() + retained.sin.len(), capacity * 32 * 8);
        }
    }

    #[test]
    fn retained_text_bytes_preserve_signed_zero_and_subnormal_bits() {
        let mut cos = vec![0.0; 32];
        cos[..4].copy_from_slice(&[-0.0, f32::from_bits(1), f32::from_bits(0x80000001), 1.0]);
        let retained = RetainedTextRope::from_f32(&cos, &cos).unwrap();
        for (bytes, value) in retained.cos.chunks_exact(4).zip(&cos) {
            assert_eq!(
                u32::from_le_bytes(bytes.try_into().unwrap()),
                value.to_bits()
            );
        }
        assert_eq!(retained.cos, retained.sin);
    }

    #[test]
    fn invalid_or_nonfinite_retained_tables_are_rejected() {
        for (cos, sin) in [
            (vec![], vec![]),
            (vec![0.0; 31], vec![0.0; 31]),
            (vec![0.0; 32], vec![0.0; 64]),
            (vec![f32::NAN; 32], vec![0.0; 32]),
            (vec![0.0; 32], vec![f32::INFINITY; 32]),
        ] {
            assert!(RetainedTextRope::from_f32(&cos, &sin).is_err());
        }
    }

    #[test]
    #[ignore = "real GPU: exact full262144 text table restore, shared allocation and bounds checks"]
    fn retained_text_rope_restores_full_capacity_on_real_gpu() -> Result<()> {
        let before = cuda_device_allocation_snapshot();
        {
            let stream = OwnedCudaStream::create(0)?;
            let (cos, sin) = original(262144, 0, 10_000_000.0);
            let retained = RetainedTextRope::from_f32(&cos, &sin)?;
            assert_eq!(retained.cos.len() + retained.sin.len(), 64 * 1024 * 1024);
            let (cos, sin) = retained.upload(&stream)?;
            let wrong_sin = OwnedCudaBuffer::allocate(0, retained.sin.len() - 4)?;
            let replacement = vec![0x55; retained.cos.len()];
            cos.copy_from_host_sync(&stream, &replacement)?;
            assert!(retained.restore_into(&stream, &cos, &wrong_sin).is_err());
            let mut actual = vec![0; retained.cos.len()];
            cos.copy_to_host_sync(&stream, &mut actual)?;
            assert_eq!(
                actual, replacement,
                "invalid second destination must not mutate the first"
            );
            drop(wrong_sin);
            let resident = cuda_device_allocation_snapshot();
            for round in 0..3 {
                cos.copy_from_host_sync(&stream, &replacement)?;
                sin.copy_from_host_sync(&stream, &replacement)?;
                let started = std::time::Instant::now();
                retained.restore_into(&stream, &cos, &sin)?;
                let elapsed = started.elapsed().as_secs_f64();
                let after = cuda_device_allocation_snapshot();
                assert_eq!(resident.live_bytes, after.live_bytes);
                assert_eq!(resident.live_allocations, after.live_allocations);
                cos.copy_to_host_sync(&stream, &mut actual)?;
                assert_eq!(actual, retained.cos);
                sin.copy_to_host_sync(&stream, &mut actual)?;
                assert_eq!(actual, retained.sin);
                eprintln!("RETAINED_TEXT_ROPE round={round} bytes=67108864 restore_s={elapsed:.9} exact=true new_gpu_allocations=0");
            }
            stream.synchronize()?;
        }
        let after = cuda_device_allocation_snapshot();
        assert_eq!(before.live_bytes, after.live_bytes);
        assert_eq!(before.live_allocations, after.live_allocations);
        Ok(())
    }
}
