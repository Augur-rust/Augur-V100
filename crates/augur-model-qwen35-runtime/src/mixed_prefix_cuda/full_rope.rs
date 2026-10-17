//! Request-local immutable position tables. Sharing changes ownership, not math.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Binding {
    stream_owner: usize,
    theta_bits: u32,
    positions: usize,
    rotary_dim: usize,
}

impl Binding {
    fn new(
        owner: usize,
        theta: f32,
        positions: usize,
        rotary_dim: usize,
    ) -> Result<Self, Qwen35MixedPrefixCudaError> {
        if owner == 0
            || !theta.is_finite()
            || theta <= 1.0
            || positions == 0
            || rotary_dim == 0
            || !rotary_dim.is_multiple_of(2)
            || positions
                .checked_mul(rotary_dim)
                .and_then(|n| n.checked_mul(4))
                .is_none_or(|n| n > isize::MAX as usize)
        {
            return Err(invalid("shared RoPE binding extent/configuration"));
        }
        Ok(Self {
            stream_owner: owner,
            theta_bits: theta.to_bits(),
            positions,
            rotary_dim,
        })
    }
    fn validate(self, other: Self) -> Result<(), Qwen35MixedPrefixCudaError> {
        if self != other {
            return Err(invalid("shared RoPE stream/position/numerical identity"));
        }
        Ok(())
    }
}

pub(super) struct FullRopeTables {
    pub(super) cos: OwnedCudaBuffer,
    pub(super) sin: OwnedCudaBuffer,
    binding: Binding,
    position_identity_sha256: Option<String>,
}

impl FullRopeTables {
    pub(super) fn new(
        stream: &OwnedCudaStream,
        theta: f32,
        positions: usize,
        rotary_dim: usize,
    ) -> Result<Self, Qwen35MixedPrefixCudaError> {
        let binding = Binding::new(stream.owner_instance_key()?, theta, positions, rotary_dim)?;
        // Use precisely the existing per-position FP32 trigonometry.
        let (cos, sin) = native_rope_table_span(theta, positions, rotary_dim)?;
        let cos_device = alloc_f32(stream.device_ordinal(), cos.len())?;
        let sin_device = alloc_f32(stream.device_ordinal(), sin.len())?;
        cos_device.copy_from_host_sync(stream, &f32_bytes(&cos))?;
        sin_device.copy_from_host_sync(stream, &f32_bytes(&sin))?;
        Ok(Self {
            cos: cos_device,
            sin: sin_device,
            binding,
            position_identity_sha256: None,
        })
    }
    pub(super) fn new_image(
        stream: &OwnedCudaStream,
        schedule: &augur_model_qwen35::Qwen35ImageMropePlan,
    ) -> Result<Self, Qwen35MixedPrefixCudaError> {
        let binding = Binding::new(
            stream.owner_instance_key()?,
            10_000_000.0,
            schedule.capacity(),
            64,
        )?;
        let (cos, sin) = schedule.tables()?;
        let cos_device = alloc_f32(stream.device_ordinal(), cos.len())?;
        let sin_device = alloc_f32(stream.device_ordinal(), sin.len())?;
        cos_device.copy_from_host_sync(stream, &f32_bytes(&cos))?;
        sin_device.copy_from_host_sync(stream, &f32_bytes(&sin))?;
        Ok(Self {
            cos: cos_device,
            sin: sin_device,
            binding,
            position_identity_sha256: Some(schedule.identity_sha256().to_owned()),
        })
    }
    pub(super) fn validate(
        &self,
        stream: &OwnedCudaStream,
        theta: f32,
        positions: usize,
        rotary_dim: usize,
    ) -> Result<(), Qwen35MixedPrefixCudaError> {
        if self.position_identity_sha256.is_some() {
            return Err(invalid(
                "image MRoPE tables require explicit position-schedule binding",
            ));
        }
        self.binding.validate(Binding::new(
            stream.owner_instance_key()?,
            theta,
            positions,
            rotary_dim,
        )?)
    }
    pub(super) fn validate_image(
        &self,
        stream: &OwnedCudaStream,
        theta: f32,
        positions: usize,
        rotary_dim: usize,
        schedule: &augur_model_qwen35::Qwen35ImageMropePlan,
    ) -> Result<(), Qwen35MixedPrefixCudaError> {
        self.binding.validate(Binding::new(
            stream.owner_instance_key()?,
            theta,
            positions,
            rotary_dim,
        )?)?;
        if positions != schedule.capacity()
            || self.position_identity_sha256.as_deref() != Some(schedule.identity_sha256())
        {
            return Err(invalid("image MRoPE position-schedule identity drift"));
        }
        Ok(())
    }
    pub(super) fn len_bytes(&self) -> usize {
        self.cos.len_bytes() + self.sin.len_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sharing_requires_exact_stream_position_and_numerical_binding() {
        let b = Binding::new(7, 10_000_000.0, 196608, 64).unwrap();
        assert!(b.validate(b).is_ok());
        for other in [
            Binding {
                stream_owner: 8,
                ..b
            },
            Binding {
                theta_bits: b.theta_bits + 1,
                ..b
            },
            Binding {
                positions: 196607,
                ..b
            },
            Binding {
                rotary_dim: 128,
                ..b
            },
        ] {
            assert!(b.validate(other).is_err());
        }
        for (owner, theta, positions, dim) in [
            (0, 2.0, 1, 64),
            (1, f32::NAN, 1, 64),
            (1, 1.0, 1, 64),
            (1, 2.0, 0, 64),
            (1, 2.0, 1, 0),
            (1, 2.0, 1, 63),
            (1, 2.0, usize::MAX, 64),
        ] {
            assert!(Binding::new(owner, theta, positions, dim).is_err());
        }
    }
    #[test]
    fn full_192k_table_accounting_is_not_occupied_context_evidence() {
        let b = Binding::new(7, 10_000_000.0, 196608, 64).unwrap();
        let one = b.positions * b.rotary_dim * 4;
        assert_eq!(one, 50_331_648);
        assert_eq!(one * 15, 754_974_720);
    }

    #[cfg(feature = "qwen35-prefill-cublas")]
    #[test]
    #[ignore = "exclusive CUDA workspace/table component; not HTTP or occupied-context qualification"]
    fn gpu_verify_rope_lease_exact_owner_lifetime_and_no_upload() {
        let before = cuda_device_allocation_snapshot();
        let stream = OwnedCudaStream::create(0).unwrap();
        let foreign = OwnedCudaStream::create(0).unwrap();
        let theta = 10_000_000.0_f32;
        for positions in [32, 8192, 196608] {
            let tables = Arc::new(FullRopeTables::new(&stream, theta, positions, 64).unwrap());
            let weak = Arc::downgrade(&tables);
            let start = Instant::now();
            let baseline_before = cuda_device_allocation_snapshot();
            let baseline = PrefillWorkspace::new_with_rope_capacity(&stream, 8, positions).unwrap();
            baseline.upload_rope_tables(&stream, positions, 0, theta).unwrap();
            let baseline_ms = start.elapsed().as_secs_f64() * 1000.0;
            let baseline_after = cuda_device_allocation_snapshot();
            let start = Instant::now();
            let shared_before = cuda_device_allocation_snapshot();
            let shared = PrefillWorkspace::new_with_rope_lease(
                &stream, 8, positions, Some((Arc::clone(&tables), theta))).unwrap();
            let boundary = augur_backend::cuda_host_boundary_snapshot();
            shared.upload_rope_tables(&stream, positions, 0, theta).unwrap();
            let boundary_after = augur_backend::cuda_host_boundary_snapshot();
            assert_eq!(boundary_after, boundary, "lease validation must not call CUDA or upload");
            let shared_ms = start.elapsed().as_secs_f64() * 1000.0;
            let shared_after = cuda_device_allocation_snapshot();
            assert_eq!((baseline_after.successful_allocations-baseline_before.successful_allocations)
                -(shared_after.successful_allocations-shared_before.successful_allocations), 2);
            assert_eq!((baseline_after.live_bytes-baseline_before.live_bytes)
                -(shared_after.live_bytes-shared_before.live_bytes), tables.len_bytes() as u64);
            assert_eq!(shared.rope_reuse_report()["avoided_table_h2d_bytes"], tables.len_bytes());
            for (old, new) in [(&baseline.rope_cos, &shared.rope_cos), (&baseline.rope_sin, &shared.rope_sin)] {
                let mut expected = vec![0u8; old.len_bytes()];
                let mut actual = vec![0u8; new.len_bytes()];
                old.copy_to_host_sync(&stream, &mut expected).unwrap();
                new.copy_to_host_sync(&stream, &mut actual).unwrap();
                assert_eq!(expected, actual, "all table bytes, not only selected positions");
            }
            let reject_before = cuda_device_allocation_snapshot();
            for (owner, count, th) in [(&foreign, positions, theta),
                (&stream, positions-1, theta), (&stream, positions, f32::from_bits(theta.to_bits()+1))] {
                assert!(PrefillWorkspace::new_with_rope_lease(owner, 8, count,
                    Some((Arc::clone(&tables), th))).is_err());
            }
            assert!(shared.upload_rope_tables(&stream, positions, 1, theta).is_err());
            assert!(shared.upload_rope_tables(&stream, positions+1, 0, theta).is_err());
            assert!(shared.upload_rope_tables(&foreign, positions, 0, theta).is_err());
            assert!(shared.upload_rope_tables(&stream, positions, 0, f32::from_bits(theta.to_bits()+1)).is_err());
            assert_eq!(cuda_device_allocation_snapshot(), reject_before);
            drop(tables);
            assert!(weak.upgrade().is_some(), "workspace must retain source owner");
            let mut probe = vec![0u8; shared.rope_cos.len_bytes()];
            shared.rope_cos.copy_to_host_sync(&stream, &mut probe).unwrap();
            stream.synchronize().unwrap();
            drop(shared);
            assert!(weak.upgrade().is_none(), "final table lease must release");
            drop(baseline);
            println!("{}", serde_json::json!({"event":"verify_rope_lease_case",
                "positions":positions,"full_table_bits_exact":true,"rejected_bindings":7,
                "avoided_allocations":2,"avoided_bytes":positions*64*4,
                "baseline_workspace_ms":baseline_ms,"shared_workspace_ms":shared_ms,
                "scope":"component_construction_wall_not_HTTP"}));
        }
        let mut image = FullRopeTables::new(&stream, theta, 32, 64).unwrap();
        image.position_identity_sha256 = Some("synthetic-image-schedule-negative-case".into());
        assert!(PrefillWorkspace::new_with_rope_lease(&stream, 8, 32,
            Some((Arc::new(image), theta))).is_err());
        drop(foreign);
        drop(stream);
        let after = cuda_device_allocation_snapshot();
        assert_eq!(after.live_bytes, before.live_bytes);
        assert_eq!(after.live_allocations, before.live_allocations);
        assert_eq!(after.successful_allocations-before.successful_allocations,
            after.successful_frees-before.successful_frees);
        println!("{}", serde_json::json!({"event":"verify_rope_lease_cleanup",
            "allocations":after.successful_allocations-before.successful_allocations,
            "frees":after.successful_frees-before.successful_frees,"live_bytes_delta":0}));
    }

    #[test]
    #[ignore = "isolated CUDA table allocation/readback only; not occupied-context qualification"]
    fn gpu_192k_shared_rope_tables_roundtrip_and_release() {
        let before = cuda_device_allocation_snapshot();
        let stream = OwnedCudaStream::create(0).unwrap();
        let tables = Arc::new(FullRopeTables::new(&stream, 10_000_000.0, 196608, 64).unwrap());
        let leases = (0..16).map(|_| Arc::clone(&tables)).collect::<Vec<_>>();
        let allocated = cuda_device_allocation_snapshot();
        assert_eq!(allocated.live_bytes - before.live_bytes, 50_331_648);
        assert_eq!(allocated.live_allocations - before.live_allocations, 2);
        let (cos, sin) = native_rope_table_span(10_000_000.0, 196608, 64).unwrap();
        for (owner, expected) in [(&tables.cos, cos), (&tables.sin, sin)] {
            let mut got = vec![0u8; owner.len_bytes()];
            owner.copy_to_host_sync(&stream, &mut got).unwrap();
            assert_eq!(got, f32_bytes(&expected));
        }
        drop(tables);
        assert_eq!(Arc::strong_count(&leases[0]), 16);
        assert_eq!(
            cuda_device_allocation_snapshot().live_bytes,
            allocated.live_bytes
        );
        stream.synchronize().unwrap();
        drop(leases);
        drop(stream);
        let after = cuda_device_allocation_snapshot();
        assert_eq!(after.live_bytes, before.live_bytes);
        assert_eq!(after.live_allocations, before.live_allocations);
        assert_eq!(after.successful_frees - before.successful_frees, 2);
        println!("ROPE_192K_COMPONENT bytes=50331648 owners=2 leases=16 full_table_bits=EXACT terminal_live_delta=0 occupied_context_qualification=false");
    }
}
