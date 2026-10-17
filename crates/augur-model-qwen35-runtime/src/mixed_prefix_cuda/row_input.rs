//! Same-byte input-only row layout. No request-time conversion or cached guesses.
use super::*;
type Result<T> = std::result::Result<T, Qwen35MixedPrefixCudaError>;
pub(super) fn switch(value: Option<&std::ffi::OsStr>, supported: bool) -> Result<bool> {
    match value {
        None => Ok(false),
        Some(v) if v == "0" => Ok(false),
        Some(v) if v == "1" && supported => Ok(true),
        _ => Err(invalid(
            "AUGUR_QWEN35_ROW_Q8_INPUT requires 0/1 and prefill backend",
        )),
    }
}
pub(super) fn selected() -> Result<bool> {
    switch(
        std::env::var_os("AUGUR_QWEN35_ROW_Q8_INPUT").as_deref(),
        cfg!(feature = "qwen35-prefill-cublas"),
    )
}
pub(super) fn layout(enabled: bool) -> PhysicalLayoutId {
    if enabled {
        PhysicalLayoutId::DEVICE_QWEN35_Q8_ROW_CODE_SCALE_PLANES_V1
    } else {
        PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_0_CUDA_V1
    }
}
// A wide prefix on row-layout weights cannot temporarily bind legacy grid or
// integer kernels. Select its final original-input recipe from retained plans.
pub(super) fn direct_prefix_binding(
    row_layers: usize, total_layers: usize, rows: usize, complete_recipe: bool,
) -> Result<bool> {
    if row_layers == 0 { return Ok(false); }
    if row_layers != 64 || total_layers != 64 || rows != 512 || !complete_recipe {
        return Err(invalid("row input wide prefix requires all64 retained layers and complete B512 original-input recipe"));
    }
    Ok(true)
}
pub(super) fn encode(bytes: &[u8], rows: usize, cols: usize) -> Result<Vec<u8>> {
    if cols != 5120 || !matches!(rows, 1024 | 6144 | 10240 | 12288) || bytes.len() != rows * 5440 {
        return Err(invalid("row Q8 input codec extent/shape"));
    }
    let mut out = vec![0u8; bytes.len()];
    for (src, dst) in bytes.chunks_exact(5440).zip(out.chunks_exact_mut(5440)) {
        let (codes, scales) = dst.split_at_mut(5120);
        for (b, wire) in src.chunks_exact(34).enumerate() {
            codes[b * 32..b * 32 + 32].copy_from_slice(&wire[2..]);
            scales[b * 2..b * 2 + 2].copy_from_slice(&wire[..2]);
        }
    }
    Ok(out)
}
pub(super) enum Scalar {
    Wire(CudaQ8_0Q8_1MatvecArgsV1),
    #[cfg(feature = "qwen35-prefill-cublas")]
    Row(augur_backend::CudaQwen35RowQ8InputProjectionV1),
}
impl Scalar {
    pub fn bind(
        stream: &OwnedCudaStream,
        buffer: &OwnedCudaBuffer,
        physical: PhysicalLayoutId,
        rows: usize,
        cols: usize,
        input: CudaQ8_1ActivationViewV1,
        output: CudaF32View,
    ) -> Result<Self> {
        #[cfg(feature = "qwen35-prefill-cublas")]
        if physical == layout(true) {
            return Ok(Self::Row(
                augur_backend::CudaQwen35RowQ8InputProjectionV1::bind(
                    stream,
                    augur_backend::CudaQwen35RowQ8InputWeightV1::new(
                        buffer.view(0, buffer.len_bytes())?,
                        physical,
                        rows,
                        cols,
                    )?,
                    input,
                    output,
                    1,
                )?,
            ));
        }
        Ok(Self::Wire(CudaQ8_0Q8_1MatvecArgsV1::bind(
            stream,
            CudaQ8_0WeightViewV1::new(buffer.view(0, buffer.len_bytes())?, physical, rows, cols)?,
            input,
            output,
        )?))
    }
    pub fn submit_for_plan(
        &self,
        stream: &OwnedCudaStream,
    ) -> std::result::Result<(), augur_backend::CudaRuntimeError> {
        match self {
            Self::Wire(p) => p.submit_for_plan(stream),
            #[cfg(feature = "qwen35-prefill-cublas")]
            Self::Row(p) => p.submit(stream),
        }
    }
}
#[cfg(feature = "qwen35-prefill-cublas")]
pub(super) fn prefill_weight(
    buffer: &OwnedCudaBuffer,
    physical: PhysicalLayoutId,
    rows: usize,
    cols: usize,
) -> Result<CudaQwen35PrefillWeightV1> {
    if physical == layout(true) {
        Ok(CudaQwen35PrefillWeightV1::RowQ8Input(
            augur_backend::CudaQwen35RowQ8InputWeightV1::new(
                buffer.view(0, buffer.len_bytes())?,
                physical,
                rows,
                cols,
            )?,
        ))
    } else {
        Ok(CudaQwen35PrefillWeightV1::Q8_0(CudaQ8_0WeightViewV1::new(
            buffer.view(0, buffer.len_bytes())?,
            physical,
            rows,
            cols,
        )?))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn row_input_prefix_binding_is_direct_and_recipe_bounded() {
        assert!(!direct_prefix_binding(0,64,512,true).unwrap());
        assert!(!direct_prefix_binding(0,64,128,false).unwrap());
        assert!(direct_prefix_binding(64,64,512,true).unwrap());
        for (selected,total,rows,recipe) in [(63,64,512,true),(64,63,512,true),
            (64,64,128,true),(64,64,8,true),(64,64,513,true),(64,64,512,false)] {
            assert!(direct_prefix_binding(selected,total,rows,recipe).is_err());
        }
    }
    #[test]
    fn row_input_switch_is_strict_and_feature_bounded() {
        use std::ffi::OsStr;
        assert!(!switch(None, false).unwrap());
        assert!(!switch(Some(OsStr::new("0")), false).unwrap());
        assert!(switch(Some(OsStr::new("1")), true).unwrap());
        assert!(switch(Some(OsStr::new("1")), false).is_err());
        for s in ["", "true", "01", " 1", "1 "] {
            assert!(switch(Some(OsStr::new(s)), true).is_err());
        }
    }
    #[test]
    fn row_input_codec_preserves_every_bit_and_row_permutation() {
        let rows: usize = 1024;
        let source: Vec<u8> = (0..rows * 5440)
            .map(|i| (i.wrapping_mul(71) ^ (i >> 8)) as u8)
            .collect();
        let encoded = encode(&source, rows, 5120).unwrap();
        let mut recovered = vec![0; source.len()];
        for r in 0..rows {
            for b in 0..160 {
                let dst = r * 5440 + b * 34;
                recovered[dst..dst + 2].copy_from_slice(
                    &encoded[r * 5440 + 5120 + b * 2..r * 5440 + 5120 + b * 2 + 2],
                );
                recovered[dst + 2..dst + 34]
                    .copy_from_slice(&encoded[r * 5440 + b * 32..r * 5440 + b * 32 + 32]);
            }
        }
        assert_eq!(recovered, source);
        let reorder = |bytes: &[u8]| {
            let mut v = Vec::with_capacity(bytes.len());
            for r in (0..rows).step_by(2).chain((1..rows).step_by(2)) {
                v.extend_from_slice(&bytes[r * 5440..(r + 1) * 5440]);
            }
            v
        };
        assert_eq!(
            encode(&reorder(&source), rows, 5120).unwrap(),
            reorder(&encoded)
        );
        for (r, c) in [(17408, 5120), (1024, 6144), (0, 5120), (1023, 5120)] {
            assert!(encode(&source, r, c).is_err());
        }
        assert!(encode(&source[..source.len() - 1], rows, 5120).is_err());
    }
}
