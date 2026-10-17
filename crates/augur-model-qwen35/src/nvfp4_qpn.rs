// SPDX-License-Identifier: Apache-2.0
//! Bit-preserving GGML NVFP4 <-> Volta QPN physical layout.
//!
//! A tensor remains exactly 36 bytes per 64 weights: the code plane comes
//! first, followed by the scale plane. Tensor-level scales are NOT folded
//! into this codec. This is not a new quantizer or permission to execute an
//! existing GGML kernel on QPN bytes.
//!
//! Geometry follows NInfer's `nvfp4_prepack_sm70.cu` (Apache-2.0). The reference
//! implementation below is intentionally CPU-only and bounded by the caller's
//! buffers; the device loader should use the equivalent GPU permutation.

use crate::Qwen35Error;

const ORDER: [usize; 16] = [0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen35Nvfp4QpnGeometry {
    rows: usize,
    cols: usize,
    code_bytes: usize,
    scale_bytes: usize,
}

impl Qwen35Nvfp4QpnGeometry {
    pub fn new(rows: usize, cols: usize) -> Result<Self, Qwen35Error> {
        if rows == 0 || cols == 0 || !rows.is_multiple_of(32) || !cols.is_multiple_of(64) {
            return Err(Qwen35Error::invalid(
                "nvfp4_qpn.shape",
                "requires positive rows%32=0 and cols%64=0",
            ));
        }
        let elements = rows.checked_mul(cols).ok_or(Qwen35Error::SizeOverflow)?;
        Ok(Self {
            rows,
            cols,
            code_bytes: elements / 2,
            scale_bytes: elements / 16,
        })
    }

    pub const fn rows(self) -> usize {
        self.rows
    }
    pub const fn cols(self) -> usize {
        self.cols
    }
    pub const fn code_bytes(self) -> usize {
        self.code_bytes
    }
    pub const fn scale_bytes(self) -> usize {
        self.scale_bytes
    }
    pub const fn payload_bytes(self) -> usize {
        self.code_bytes + self.scale_bytes
    }

    fn check_extent(self, bytes: &[u8]) -> Result<(), Qwen35Error> {
        if bytes.len() != self.payload_bytes() {
            return Err(Qwen35Error::invalid(
                "nvfp4_qpn.extent",
                "payload extent differs from geometry",
            ));
        }
        Ok(())
    }

    /// Index of one 16-value group in the QPN code/scale planes.
    fn group_index(self, row: usize, group: usize) -> usize {
        let r = row % 32;
        let lane = (r / 8) * 4 + (r & 3) + (r & 4) * 4;
        ((row / 32) * (self.cols / 16) + group) * 32 + lane
    }

    /// Convert into a same-sized disjoint destination. Reject invalid scale
    /// words before changing any destination byte; zero scales are supported.
    pub fn encode_ggml_into(self, source: &[u8], dest: &mut [u8]) -> Result<(), Qwen35Error> {
        self.check_extent(source)?;
        self.check_extent(dest)?;
        for block in source.chunks_exact(36) {
            if block[..4].iter().any(|scale| *scale >= 127) {
                return Err(Qwen35Error::invalid(
                    "nvfp4_qpn.scale",
                    "expected finite nonnegative E4M3FN scale",
                ));
            }
        }
        for row in 0..self.rows {
            for group in 0..self.cols / 16 {
                let block = (row * (self.cols / 64) + group / 4) * 36;
                let codes = &source[block + 4 + (group % 4) * 8..][..8];
                let index = self.group_index(row, group);
                for byte in 0..8 {
                    let a = ORDER[2 * byte];
                    let b = ORDER[2 * byte + 1];
                    let low = (codes[a % 8] >> ((a / 8) * 4)) & 15;
                    let high = (codes[b % 8] >> ((b / 8) * 4)) & 15;
                    dest[index * 8 + byte] = low | (high << 4);
                }
                dest[self.code_bytes + index] = source[block + group % 4];
            }
        }
        Ok(())
    }

    /// Recover the canonical GGML wire bytes for exact roundtrip validation.
    pub fn decode_ggml_into(self, source: &[u8], dest: &mut [u8]) -> Result<(), Qwen35Error> {
        self.check_extent(source)?;
        self.check_extent(dest)?;
        if source[self.code_bytes..].iter().any(|scale| *scale >= 127) {
            return Err(Qwen35Error::invalid(
                "nvfp4_qpn.scale",
                "expected finite nonnegative E4M3FN scale",
            ));
        }
        dest.fill(0);
        for row in 0..self.rows {
            for group in 0..self.cols / 16 {
                let block = (row * (self.cols / 64) + group / 4) * 36;
                let index = self.group_index(row, group);
                dest[block + group % 4] = source[self.code_bytes + index];
                for (position, &natural) in ORDER.iter().enumerate() {
                    let code = (source[index * 8 + position / 2] >> ((position % 2) * 4)) & 15;
                    dest[block + 4 + (group % 4) * 8 + natural % 8] |= code << ((natural / 8) * 4);
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(g: Qwen35Nvfp4QpnGeometry) -> Vec<u8> {
        let mut data: Vec<u8> = (0..g.payload_bytes())
            .map(|i| ((i * 73 + i / 251) & 255) as u8)
            .collect();
        for (index, block) in data.chunks_exact_mut(36).enumerate() {
            for (sub, scale) in block[..4].iter_mut().enumerate() {
                *scale = ((index + sub * 23) % 127) as u8;
            }
        }
        data
    }

    #[test]
    fn plane_extent_and_shapes_are_checked() {
        for (n, k) in [(0, 64), (32, 0), (31, 64), (32, 63), (usize::MAX - 31, 64)] {
            assert!(Qwen35Nvfp4QpnGeometry::new(n, k).is_err());
        }
        for (n, k) in [(32, 64), (17408, 5120), (5120, 17408), (48 * 128, 5120)] {
            let g = Qwen35Nvfp4QpnGeometry::new(n, k).unwrap();
            assert_eq!(g.payload_bytes(), n * (k / 64) * 36);
            assert_eq!(g.code_bytes() + g.scale_bytes(), g.payload_bytes());
        }
    }

    #[test]
    fn all_rows_groups_and_nibbles_roundtrip_without_requantization() {
        for (n, k) in [(32, 64), (64, 128), (128, 5120), (32, 17408)] {
            let g = Qwen35Nvfp4QpnGeometry::new(n, k).unwrap();
            let original = fixture(g);
            let mut packed = vec![0; g.payload_bytes()];
            let mut restored = vec![0; g.payload_bytes()];
            g.encode_ggml_into(&original, &mut packed).unwrap();
            g.decode_ggml_into(&packed, &mut restored).unwrap();
            assert_eq!(restored, original);
        }
    }

    #[test]
    fn qpn_quadpair_lane_and_code_words_have_an_independent_known_answer() {
        let g = Qwen35Nvfp4QpnGeometry::new(32, 64).unwrap();
        let mut source = vec![0; g.payload_bytes()];
        for row in 0..32 {
            for group in 0..4 {
                source[row * 36 + group] = (row * 4 + group) as u8 % 127;
                for byte in 0..8 {
                    source[row * 36 + 4 + group * 8 + byte] = (byte | ((byte + 8) << 4)) as u8;
                }
            }
        }
        let mut dest = vec![0; source.len()];
        g.encode_ggml_into(&source, &mut dest).unwrap();
        assert_eq!(
            &dest[..8],
            &[0x20, 0x64, 0x31, 0x75, 0xa8, 0xec, 0xb9, 0xfd]
        );
        // QPN lanes 0,4,16,31 refer to physical rows 0,8,4,31.
        for (lane, row) in [(0, 0), (4, 8), (16, 4), (31, 31)] {
            assert_eq!(dest[g.code_bytes() + lane], (row * 4) as u8);
            assert_eq!(dest[g.code_bytes() + 32 + lane], (row * 4 + 1) as u8);
        }
    }

    #[test]
    fn invalid_scale_and_short_buffers_cannot_create_a_partial_layout() {
        let g = Qwen35Nvfp4QpnGeometry::new(32, 64).unwrap();
        let mut source = fixture(g);
        let mut dest = vec![0xa5; g.payload_bytes()];
        for bad in [127, 128, 255] {
            source[31 * 36 + 3] = bad;
            assert!(g.encode_ggml_into(&source, &mut dest).is_err());
            assert!(dest.iter().all(|b| *b == 0xa5));
        }
        assert!(g
            .encode_ggml_into(&source[..source.len() - 1], &mut dest)
            .is_err());
    }
}
