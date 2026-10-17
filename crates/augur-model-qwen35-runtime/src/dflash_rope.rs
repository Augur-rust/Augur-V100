//! DFlash2 checkpoint-layout RoPE, separate from the target model's MRoPE.
//!
//! The supported Qwen3 draft stores Q/K (and their norm weights) in NeoX
//! half-split order. No row permutation is applied to its safetensors or GGUF.

pub const DFLASH_ROPE_LAYOUT: &str = "neox_half_split";
pub const DFLASH_ROPE_HEAD_DIM: usize = 128;
pub const DFLASH_ROPE_THETA: f32 = 10_000_000.0;

/// Rotate one or more complete, already-normalized heads at one absolute
/// position. Validate before mutation so rejected input is left unchanged.
pub fn apply_dflash_neox_rope(values: &mut [f32], position: usize) -> Result<(), &'static str> {
    if values.is_empty() || !values.len().is_multiple_of(DFLASH_ROPE_HEAD_DIM) {
        return Err("DFlash RoPE requires complete 128-dimensional heads");
    }
    if position >= 262_144 || values.iter().any(|value| !value.is_finite()) {
        return Err("DFlash RoPE position or non-finite input");
    }
    let half = DFLASH_ROPE_HEAD_DIM / 2;
    for head in values.chunks_exact_mut(DFLASH_ROPE_HEAD_DIM) {
        for pair in 0..half {
            // Round a high-precision inverse frequency once to F32 on both
            // CPU and CUDA. Platform powf errors otherwise grow with absolute
            // position and do not cancel across a full 2048-token window.
            let exponent = -((2 * pair) as f64) / DFLASH_ROPE_HEAD_DIM as f64;
            let inverse_frequency = (DFLASH_ROPE_THETA as f64).powf(exponent) as f32;
            let angle = position as f32 * inverse_frequency;
            let (sin, cos) = angle.sin_cos();
            let x = head[pair];
            let y = head[pair + half];
            head[pair] = x * cos - y * sin;
            head[pair + half] = x * sin + y * cos;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basis_pairs_first_and_second_halves_not_adjacent_dimensions() {
        let mut values = vec![0.0; DFLASH_ROPE_HEAD_DIM];
        values[0] = 1.0;
        apply_dflash_neox_rope(&mut values, 1).unwrap();
        assert!((values[0] - 1.0f32.cos()).abs() < 1.0e-7);
        assert!((values[64] - 1.0f32.sin()).abs() < 1.0e-7);
        assert_eq!(values[1], 0.0); // the former adjacent-pair implementation fails here
    }

    #[test]
    fn agrees_with_independent_rotate_half_formula_at_absolute_positions() {
        let input = (0..3 * DFLASH_ROPE_HEAD_DIM)
            .map(|i| ((i * 17 % 101) as f32 - 50.0) / 50.0)
            .collect::<Vec<_>>();
        for position in [0, 1, 4096, 6144, 196_600, 262_136] {
            let mut actual = input.clone();
            apply_dflash_neox_rope(&mut actual, position).unwrap();
            // Independently express HF rotate_half: [-second_half, first_half]
            // multiplied by duplicated sin frequencies, plus x*cos. Use F64
            // throughout the oracle, not the production pair-update formula.
            let mut squared_error = 0.0;
            let mut squared_reference = 0.0;
            for (index, &got) in actual.iter().enumerate() {
                let head_start = index / 128 * 128;
                let dim = index % 128;
                let paired = if dim < 64 { dim + 64 } else { dim - 64 };
                let rotated =
                    f64::from(input[head_start + paired]) * if dim < 64 { -1.0 } else { 1.0 };
                let frequency = 10_000_000.0f64.powf(-((dim % 64) as f64) / 64.0);
                let angle = position as f64 * frequency;
                let expected = f64::from(input[index]) * angle.cos() + rotated * angle.sin();
                squared_error += (f64::from(got) - expected).powi(2);
                squared_reference += expected.powi(2);
            }
            // F32 angle formation itself loses precision at long positions.
            assert!(
                (squared_error / squared_reference).sqrt() < 0.004,
                "position={position}"
            );
        }
    }

    #[test]
    fn invalid_extent_position_and_nonfinite_input_do_not_mutate() {
        for (mut values, position) in [
            (vec![], 0),
            (vec![1.0; 127], 0),
            (vec![1.0; 128], 262_144),
            (vec![f32::INFINITY; 128], 0),
            (vec![f32::NAN; 128], 0),
        ] {
            let before = values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert!(apply_dflash_neox_rope(&mut values, position).is_err());
            assert_eq!(
                before,
                values.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }
}
