//! Lossless checkpoint planes and independent F64 weight-value references.
//!
//! Compressed-tensors NVFP4 divides by weight_global_scale; ModelOpt NVFP4
//! multiplies by weight_scale_2. These are distinct typed contracts, not
//! interchangeable metadata names. Activation scales are retained but are not
//! applied to weight decoding. This module does not implement W4A4 inference.
use crate::Qwen35Error;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CheckpointNvfp4Scale {
    CompressedTensors {
        weight_global_scale: f32,
        input_global_scale: f32,
    },
    ModelOpt {
        weight_scale_2: f32,
        input_scale: f32,
    },
}

impl CheckpointNvfp4Scale {
    fn validate(self) -> Result<(), Qwen35Error> {
        let (w, x) = match self {
            Self::CompressedTensors {
                weight_global_scale,
                input_global_scale,
            } => (weight_global_scale, input_global_scale),
            Self::ModelOpt {
                weight_scale_2,
                input_scale,
            } => (weight_scale_2, input_scale),
        };
        if !w.is_finite() || w <= 0.0 || !x.is_finite() || x <= 0.0 {
            return Err(Qwen35Error::invalid(
                "checkpoint.scale",
                "expected finite positive scalars",
            ));
        }
        Ok(())
    }

    /// F64 mathematical reference, not an assertion about a device rounding order.
    fn apply(self, block_value: f64) -> f64 {
        match self {
            Self::CompressedTensors {
                weight_global_scale,
                ..
            } => block_value / f64::from(weight_global_scale),
            Self::ModelOpt { weight_scale_2, .. } => block_value * f64::from(weight_scale_2),
        }
    }
}

fn elements(rows: usize, cols: usize) -> Result<usize, Qwen35Error> {
    if rows == 0 || cols == 0 {
        return Err(Qwen35Error::invalid("checkpoint.shape", "empty matrix"));
    }
    rows.checked_mul(cols).ok_or(Qwen35Error::SizeOverflow)
}

fn e4m3(byte: u8) -> f64 {
    let bits = byte & 127;
    let exponent = bits >> 3;
    let fraction = f64::from(bits & 7);
    let magnitude = if exponent == 0 {
        fraction * 2.0_f64.powi(-9)
    } else {
        (1.0 + fraction / 8.0) * 2.0_f64.powi(i32::from(exponent) - 7)
    };
    if byte & 128 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

fn e2m1(code: u8) -> f64 {
    const MAGNITUDE: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let value = MAGNITUDE[(code & 7) as usize];
    if code & 8 != 0 {
        -value
    } else {
        value
    }
}

/// Adjacent low/high nibbles in row-major K order and one E4M3 scale per16.
/// Rows need not be a QPN tile multiple: Augur's 48-row gates are valid here.
pub struct CheckpointNvfp4Matrix<'a> {
    rows: usize,
    cols: usize,
    packed: &'a [u8],
    scales: &'a [u8],
    global: CheckpointNvfp4Scale,
}

impl<'a> CheckpointNvfp4Matrix<'a> {
    pub fn new(
        rows: usize,
        cols: usize,
        packed: &'a [u8],
        scales: &'a [u8],
        global: CheckpointNvfp4Scale,
    ) -> Result<Self, Qwen35Error> {
        let count = elements(rows, cols)?;
        global.validate()?;
        if !cols.is_multiple_of(16) || packed.len() != count / 2 || scales.len() != count / 16 {
            return Err(Qwen35Error::invalid(
                "checkpoint.nvfp4",
                "shape/plane extent mismatch",
            ));
        }
        if scales.iter().any(|&byte| byte >= 127) {
            return Err(Qwen35Error::invalid(
                "checkpoint.nvfp4.scale",
                "negative or NaN block scale",
            ));
        }
        Ok(Self {
            rows,
            cols,
            packed,
            scales,
            global,
        })
    }

    pub fn global_scale(&self) -> CheckpointNvfp4Scale {
        self.global
    }

    pub fn decode_f64_into(&self, output: &mut [f64]) -> Result<(), Qwen35Error> {
        if output.len() != self.rows * self.cols {
            return Err(Qwen35Error::invalid(
                "checkpoint.output",
                "decoded extent mismatch",
            ));
        }
        for (index, value) in output.iter_mut().enumerate() {
            let code = (self.packed[index / 2] >> ((index % 2) * 4)) & 15;
            *value = self
                .global
                .apply(e2m1(code) * e4m3(self.scales[index / 16]));
        }
        Ok(())
    }

    /// Reorder codes/scales to GGML36 without folding any global scalar or
    /// requantizing. Callers MUST retain global_scale() as part of the weight.
    /// This supplies an input plane for a later QPN adapter, not a GGUF file.
    pub fn to_ggml_planes_into(&self, output: &mut [u8]) -> Result<(), Qwen35Error> {
        let count = self.rows * self.cols;
        if !self.cols.is_multiple_of(64) || output.len() != count / 64 * 36 {
            return Err(Qwen35Error::invalid(
                "checkpoint.ggml",
                "requires K%64 and exact output extent",
            ));
        }
        for ((block, packed), scales) in output.chunks_exact_mut(36)
            .zip(self.packed.chunks_exact(32)).zip(self.scales.chunks_exact(4)) {
            block[..4].copy_from_slice(scales);
            for (src, dst) in packed.chunks_exact(8).zip(block[4..].chunks_exact_mut(8)) {
                dst.copy_from_slice(&transpose_nibble_pairs(src).to_le_bytes());
            }
        }
        Ok(())
    }
}

// Lossless bit permutation: adjacent HF nibbles -> ggml split-half8 lanes.
// Spread each source byte into one u16 lane, then interleave the even/odd codes.
fn transpose_nibble_pairs(src: &[u8]) -> u64 {
    let lo = u32::from_le_bytes(src[..4].try_into().unwrap());
    let hi = u32::from_le_bytes(src[4..8].try_into().unwrap());
    let even = (lo & 0x0f0f0f0f) | ((hi & 0x0f0f0f0f) << 4);
    let odd = ((lo >> 4) & 0x0f0f0f0f) | (hi & 0xf0f0f0f0);
    fn spread(x: u32) -> u64 {
        let x = u64::from(x);
        let x = (x | (x << 16)) & 0x0000ffff0000ffff;
        (x | (x << 8)) & 0x00ff00ff00ff00ff
    }
    spread(even) | (spread(odd) << 8)
}

/// ModelOpt's attention FP8 planes use a scalar multiplier, not NVFP4 block scales.
pub struct CheckpointFp8Matrix<'a> {
    data: &'a [u8],
    weight_scale: f32,
    input_scale: f32,
}
impl<'a> CheckpointFp8Matrix<'a> {
    pub fn new(
        rows: usize,
        cols: usize,
        data: &'a [u8],
        weight_scale: f32,
        input_scale: f32,
    ) -> Result<Self, Qwen35Error> {
        if elements(rows, cols)? != data.len()
            || data.iter().any(|v| v & 127 == 127)
            || !weight_scale.is_finite()
            || weight_scale <= 0.0
            || !input_scale.is_finite()
            || input_scale <= 0.0
        {
            return Err(Qwen35Error::invalid(
                "checkpoint.fp8",
                "invalid geometry, value or scale",
            ));
        }
        Ok(Self {
            data,
            weight_scale,
            input_scale,
        })
    }
    pub fn scales(&self) -> (f32, f32) {
        (self.weight_scale, self.input_scale)
    }
    pub fn decode_f64_into(&self, output: &mut [f64]) -> Result<(), Qwen35Error> {
        if output.len() != self.data.len() {
            return Err(Qwen35Error::invalid(
                "checkpoint.fp8.output",
                "extent mismatch",
            ));
        }
        for (dst, &byte) in output.iter_mut().zip(self.data) {
            *dst = e4m3(byte) * f64::from(self.weight_scale);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn augur(w: f32) -> CheckpointNvfp4Scale {
        CheckpointNvfp4Scale::CompressedTensors {
            weight_global_scale: w,
            input_global_scale: 3.0,
        }
    }
    fn nvidia(w: f32) -> CheckpointNvfp4Scale {
        CheckpointNvfp4Scale::ModelOpt {
            weight_scale_2: w,
            input_scale: 7.0,
        }
    }
    #[test]
    fn global_operations_are_distinct_and_activation_scale_is_not_folded() {
        let packed = [0x72; 8]; // 1,6 repeated; E4M3 0x38 is1.
        for (profile, expected) in [(augur(2.0), [0.5, 3.0]), (nvidia(2.0), [2.0, 12.0])] {
            let m = CheckpointNvfp4Matrix::new(1, 16, &packed, &[0x38], profile).unwrap();
            let mut values = [0.0; 16];
            m.decode_f64_into(&mut values).unwrap();
            assert_eq!(&values[..2], &expected);
            assert_eq!(m.global_scale(), profile);
        }
    }
    #[test]
    fn small48_row_gate_has_no_artificial_qpn_padding_requirement() {
        let packed = vec![0xA2; 48 * 5120 / 2];
        let scales = vec![0x38; 48 * 5120 / 16];
        let m = CheckpointNvfp4Matrix::new(48, 5120, &packed, &scales, augur(4.0)).unwrap();
        let mut values = vec![0.0; 48 * 5120];
        m.decode_f64_into(&mut values).unwrap();
        assert!(values.chunks_exact(2).all(|v| v == [0.25, -0.25]));
    }
    #[test]
    fn lossless_plane_reorder_keeps_nibbles_scales_and_global_separate() {
        let packed: Vec<u8> = (0..64).map(|i| (i * 37) as u8).collect();
        let scales = [1, 16, 56, 126, 2, 8, 64, 100];
        let m = CheckpointNvfp4Matrix::new(2, 64, &packed, &scales, nvidia(0.125)).unwrap();
        let mut wire = vec![0; 72];
        m.to_ggml_planes_into(&mut wire).unwrap();
        for i in 0..128 {
            let b = &wire[(i / 64) * 36..][..36];
            let lane = i % 16;
            let code = (b[4 + (i % 64 / 16) * 8 + lane % 8] >> (lane / 8 * 4)) & 15;
            assert_eq!(code, (packed[i / 2] >> (i % 2 * 4)) & 15);
            assert_eq!(b[i % 64 / 16], scales[i / 16]);
        }
        assert_eq!(m.global_scale(), nvidia(0.125));
    }
    #[test]
    fn invalid_shape_extent_and_scale_reject() {
        for w in [0.0, -1.0, f32::INFINITY, f32::NAN] {
            assert!(CheckpointNvfp4Matrix::new(1, 16, &[0; 8], &[0], augur(w)).is_err());
        }
        for scale in [127, 128, 255] {
            assert!(CheckpointNvfp4Matrix::new(1, 16, &[0; 8], &[scale], nvidia(1.0)).is_err());
        }
        assert!(CheckpointNvfp4Matrix::new(1, 15, &[0; 8], &[0], nvidia(1.0)).is_err());
        assert!(CheckpointNvfp4Matrix::new(usize::MAX, 16, &[], &[], nvidia(1.0)).is_err());
        let m = CheckpointNvfp4Matrix::new(1, 16, &[0; 8], &[0], nvidia(1.0)).unwrap();
        assert!(m.to_ggml_planes_into(&mut [0; 9]).is_err());
        assert!(m.decode_f64_into(&mut [0.0; 15]).is_err());
    }
    #[test]
    fn fp8_subnormal_sign_max_and_nan() {
        let m = CheckpointFp8Matrix::new(1, 6, &[0, 128, 1, 129, 126, 254], 2.0, 9.0).unwrap();
        let mut v = [0.0; 6];
        m.decode_f64_into(&mut v).unwrap();
        assert_eq!(v, [0.0, -0.0, 1.0 / 256.0, -1.0 / 256.0, 896.0, -896.0]);
        assert!(v[1].is_sign_negative());
        assert_eq!(m.scales(), (2.0, 9.0));
        for byte in [127, 255] {
            assert!(CheckpointFp8Matrix::new(1, 1, &[byte], 1.0, 1.0).is_err());
        }
    }
}

#[cfg(test)]
mod startup_repack_tests {
    use super::*;
    fn original(packed: &[u8], scales: &[u8], output: &mut [u8]) {
        for (block_index, block) in output.chunks_exact_mut(36).enumerate() {
            block[..4].copy_from_slice(&scales[block_index*4..block_index*4+4]);
            for group in 0..4 {
                let first=block_index*64+group*16;
                for lane in 0..8 {
                    let lo=(packed[(first+lane)/2] >> ((lane%2)*4)) & 15;
                    let hi=(packed[(first+lane+8)/2] >> ((lane%2)*4)) & 15;
                    block[4+group*8+lane]=lo|(hi<<4);
                }
            }
        }
    }
    #[test]
    fn repack_exhaustive_byte_pairs_all_four_lanes() {
        for bits in 0..=u16::MAX {
            let [a,b]=bits.to_le_bytes();
            for lane in 0..4 {
                let mut src=[0u8;8];src[lane]=a;src[lane+4]=b;
                let actual=transpose_nibble_pairs(&src).to_le_bytes();
                let mut expected=[0u8;8];expected[2*lane]=(a&15)|((b&15)<<4);
                expected[2*lane+1]=(a>>4)|(b&0xf0);assert_eq!(actual,expected);
            }
        }
    }
    #[test]
    fn repack_real_shape_planes_are_byte_exact() {
        for (rows,cols) in [(1,64),(48,5120),(32,17408),(96,6144)] {
            let packed:Vec<u8>=(0..rows*cols/2).map(|i| ((i*73+i/17+11)%256) as u8).collect();
            let scales:Vec<u8>=(0..rows*cols/16).map(|i| (i%127) as u8).collect();
            let matrix=CheckpointNvfp4Matrix::new(rows,cols,&packed,&scales,
                CheckpointNvfp4Scale::CompressedTensors {weight_global_scale:2.0,input_global_scale:3.0}).unwrap();
            let mut actual=vec![0;rows*cols/64*36];let mut expected=actual.clone();
            matrix.to_ggml_planes_into(&mut actual).unwrap();original(&packed,&scales,&mut expected);
            assert_eq!(actual,expected);
        }
    }
}
