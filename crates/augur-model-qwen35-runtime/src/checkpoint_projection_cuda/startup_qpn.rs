//! Lossless startup layout selection; the inference numerical profile is unchanged.
use super::*;
use augur_backend::cuda_qwen35_qpn::OwnedCudaQwen35QpnWeightV1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Mode {
    Ggml,
    Direct,
}

impl Mode {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "ggml" => Ok(Self::Ggml),
            "direct" => Ok(Self::Direct),
            _ => Err("AUGUR_QPN_STARTUP must be ggml or direct".into()),
        }
    }
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Ggml => "authenticated_nvfp4_ggml_to_qpn_v1",
            Self::Direct => "authenticated_nvfp4_direct_to_qpn_v1",
        }
    }
}

fn default_mode() -> Mode {
    if cfg!(feature = "qwen35-augur-http") {
        Mode::Direct
    } else {
        Mode::Ggml
    }
}

pub(crate) fn mode() -> Result<Mode> {
    #[cfg(feature = "qwen35-augur-http")]
    {
        return match std::env::var("AUGUR_QPN_STARTUP") {
            Ok(value) => Mode::parse(&value),
            Err(std::env::VarError::NotPresent) => Ok(default_mode()),
            Err(error) => Err(error.into()),
        };
    }
    #[cfg(not(feature = "qwen35-augur-http"))]
    Ok(default_mode())
}

pub(crate) fn upload(
    stream: &OwnedCudaStream,
    packed: &[u8],
    scales: &[u8],
    rows: usize,
    cols: usize,
    global: CheckpointNvfp4Scale,
) -> Result<OwnedCudaQwen35QpnWeightV1> {
    let matrix = CheckpointNvfp4Matrix::new(rows, cols, packed, scales, global)?;
    let multiplier = match global {
        CheckpointNvfp4Scale::CompressedTensors {
            weight_global_scale,
            ..
        } => 1.0 / weight_global_scale,
        CheckpointNvfp4Scale::ModelOpt { weight_scale_2, .. }
        | CheckpointNvfp4Scale::ModelOptW4A16 { weight_scale_2 } => weight_scale_2,
    };
    Ok(match mode()? {
        Mode::Ggml => {
            let mut ggml = vec![0; rows * (cols / 64) * 36];
            matrix.to_ggml_planes_into(&mut ggml)?;
            OwnedCudaQwen35QpnWeightV1::upload_ggml(stream, &ggml, rows, cols, multiplier)?
        }
        Mode::Direct => OwnedCudaQwen35QpnWeightV1::upload_checkpoint_nvfp4(
            stream, packed, scales, rows, cols, multiplier,
        )?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn startup_modes_are_distinct_and_invalid_values_fail_closed() {
        assert_eq!(
            default_mode(),
            if cfg!(feature = "qwen35-augur-http") {
                Mode::Direct
            } else {
                Mode::Ggml
            }
        );
        assert_eq!(Mode::parse("ggml").unwrap(), Mode::Ggml);
        assert_eq!(Mode::parse("direct").unwrap(), Mode::Direct);
        assert_ne!(Mode::Ggml.label(), Mode::Direct.label());
        for bad in ["", "DIRECT", "cached", "direct "] {
            assert!(Mode::parse(bad).is_err());
        }
    }

    #[test]
    #[ignore = "exclusive SM70; full direct/GGML physical-byte parity at real FFN shapes"]
    fn checkpoint_direct_pack_matches_ggml_qpn_bytes() {
        use augur_backend::cuda_device_allocation_snapshot;
        use sha2::{Digest, Sha256};
        let before = cuda_device_allocation_snapshot();
        {
            let stream = OwnedCudaStream::create(0).unwrap();
            for (rows, cols) in [
                (32, 64),
                (64, 128),
                (1024, 5120),
                (17408, 5120),
                (5120, 17408),
            ] {
                let packed: Vec<u8> = (0..rows * cols / 2)
                    .map(|i| ((i * 73 + i / 8 * 19 + i / 256 * 97) & 255) as u8)
                    .collect();
                let scales: Vec<u8> = (0..rows * cols / 16)
                    .map(|i| ((i * 43 + i / 127 * 11) % 127) as u8)
                    .collect();
                let global = CheckpointNvfp4Scale::CompressedTensors {
                    weight_global_scale: 4.0,
                    input_global_scale: 1.0,
                };
                let matrix =
                    CheckpointNvfp4Matrix::new(rows, cols, &packed, &scales, global).unwrap();
                let mut ggml = vec![0; rows * cols / 64 * 36];
                matrix.to_ggml_planes_into(&mut ggml).unwrap();
                let control =
                    OwnedCudaQwen35QpnWeightV1::upload_ggml(&stream, &ggml, rows, cols, 0.25)
                        .unwrap();
                let candidate = OwnedCudaQwen35QpnWeightV1::upload_checkpoint_nvfp4(
                    &stream, &packed, &scales, rows, cols, 0.25,
                )
                .unwrap();
                assert_eq!(control.layout(), candidate.layout());
                assert_eq!(control.tensor_scale(), candidate.tensor_scale());
                let mut expected = vec![0; control.len_bytes()];
                let mut actual = vec![0; candidate.len_bytes()];
                control
                    .shared_storage()
                    .copy_to_host_sync(&stream, &mut expected)
                    .unwrap();
                candidate
                    .shared_storage()
                    .copy_to_host_sync(&stream, &mut actual)
                    .unwrap();
                assert!(
                    actual == expected,
                    "all physical code/scale bytes for {rows}x{cols}"
                );
                eprintln!(
                    "DIRECT_QPN_PARITY {rows}x{cols} bytes={} sha256={:x} PASS",
                    actual.len(),
                    Sha256::digest(&actual)
                );
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
        eprintln!(
            "DIRECT_QPN_ALLOCATION_CLOSURE live_bytes={} live_allocations={} PASS",
            after.live_bytes, after.live_allocations
        );
    }
}
