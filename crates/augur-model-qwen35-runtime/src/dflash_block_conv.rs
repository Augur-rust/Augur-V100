//! CPU oracle for DFlash2 block-local grouped dynamic convolution.
//!
//! DFlash2 runs its two-tap convolutions over the fixed draft block.  The
//! first draft token in that block has an explicit zero-padded left boundary; no
//! request or previous block history is implicitly consumed.  This helper
//! contains only the arithmetic boundary.  It does not produce tokens, query
//! frames, logits, or acceptance decisions.

use augur_model_qwen35::DFlashConfig;
use std::fmt;

pub const DFLASH_BLOCK_CONV_SCHEMA: &str = "qwen35-dflash-block-local-grouped-conv-cpu-v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DFlashBlockConvError {
    Invalid(String),
}

impl fmt::Display for DFlashBlockConvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => write!(f, "invalid DFlash block convolution: {message}"),
        }
    }
}

impl std::error::Error for DFlashBlockConvError {}

/// Apply one DFlash2 grouped dynamic-convolution side to a complete block.
///
/// Layouts are row-major and match the safetensors/CUDA boundary:
///
/// * `hidden`: `[block_size, hidden_size]`
/// * `dynamic`: `[block_size, 2, conv_kernel_size, groups]`
/// * `base`: `[2, conv_kernel_size, hidden_size]`
///
/// For output token `t`, tap `k` reads token `t-k`; reads before token zero
/// are zero.  The dynamic coefficient is selected from the output token `t`
/// and is never borrowed from another request or a prior block.
pub fn apply_block_local_grouped_conv(
    config: &DFlashConfig,
    hidden: &[f32],
    dynamic: &[f32],
    base: &[f32],
    side: usize,
) -> Result<Vec<f32>, DFlashBlockConvError> {
    config
        .validate_exact_profile()
        .map_err(|error| invalid(error.to_string()))?;
    if side >= 2 {
        return Err(invalid("convolution side must be 0 or 1"));
    }
    let block_size = config.block_size;
    let hidden_size = config.hidden_size;
    let kernel_size = config.conv_kernel_size;
    let group_size = config.conv_group_size;
    if block_size == 0 || hidden_size == 0 || kernel_size == 0 || group_size == 0 {
        return Err(invalid("convolution geometry must be nonzero"));
    }
    if !hidden_size.is_multiple_of(group_size) {
        return Err(invalid("hidden size is not divisible by group size"));
    }
    if kernel_size > block_size {
        return Err(invalid("convolution kernel exceeds block size"));
    }
    let groups = hidden_size / group_size;
    let expected_hidden = block_size
        .checked_mul(hidden_size)
        .ok_or_else(|| invalid("hidden extent overflow"))?;
    let expected_dynamic = block_size
        .checked_mul(2)
        .and_then(|value| value.checked_mul(kernel_size))
        .and_then(|value| value.checked_mul(groups))
        .ok_or_else(|| invalid("dynamic extent overflow"))?;
    let expected_base = 2usize
        .checked_mul(kernel_size)
        .and_then(|value| value.checked_mul(hidden_size))
        .ok_or_else(|| invalid("base extent overflow"))?;
    if hidden.len() != expected_hidden
        || dynamic.len() != expected_dynamic
        || base.len() != expected_base
    {
        return Err(invalid(format!(
            "extent mismatch hidden={} (expected {}), dynamic={} (expected {}), base={} (expected {})",
            hidden.len(), expected_hidden, dynamic.len(), expected_dynamic, base.len(), expected_base
        )));
    }
    if hidden
        .iter()
        .chain(dynamic)
        .chain(base)
        .any(|value| !value.is_finite())
    {
        return Err(invalid("convolution inputs must be finite"));
    }

    let dynamic_stride = 2 * kernel_size * groups;
    let mut output = vec![0.0_f32; expected_hidden];
    for token in 0..block_size {
        let hidden_out = &mut output[token * hidden_size..(token + 1) * hidden_size];
        for group in 0..groups {
            for lane in 0..group_size {
                let index = group * group_size + lane;
                let mut value = 0.0_f32;
                for tap in 0..kernel_size {
                    let source_token = match token.checked_sub(tap) {
                        Some(source) => source,
                        None => continue,
                    };
                    let source = hidden[source_token * hidden_size + index];
                    let dynamic_index =
                        token * dynamic_stride + (side * kernel_size + tap) * groups + group;
                    let base_index = (side * kernel_size + tap) * hidden_size + index;
                    value += (base[base_index] + dynamic[dynamic_index]) * source;
                }
                hidden_out[index] = value;
            }
        }
    }
    if output.iter().any(|value| !value.is_finite()) {
        return Err(invalid("convolution output is non-finite"));
    }
    Ok(output)
}

fn invalid(message: impl Into<String>) -> DFlashBlockConvError {
    DFlashBlockConvError::Invalid(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> DFlashConfig {
        DFlashConfig {
            architecture: "DFlash2DraftModel".into(),
            model_type: "qwen3".into(),
            dtype: "float16".into(),
            hidden_size: 5120,
            intermediate_size: 17408,
            num_hidden_layers: 5,
            num_target_layers: 64,
            attention_heads: 32,
            kv_heads: 8,
            head_dim: 128,
            max_position_embeddings: 262_144,
            vocab_size: 248_320,
            block_size: 8,
            conv_group_size: 16,
            conv_kernel_size: 2,
            selector_rank: 256,
            selector_top_k: 16,
            target_layer_ids: vec![5, 19, 33, 47, 61],
        }
    }

    #[test]
    fn block_boundary_is_zero_padded_and_tap_order_is_causal() {
        let cfg = config();
        let hidden = (0..cfg.block_size * cfg.hidden_size)
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        let groups = cfg.hidden_size / cfg.conv_group_size;
        let mut dynamic = vec![0.0; cfg.block_size * 2 * cfg.conv_kernel_size * groups];
        let mut base = vec![0.0; 2 * cfg.conv_kernel_size * cfg.hidden_size];
        // Side 0 tap 0 is one; tap 1 is ten.  The first row must not see a
        // previous-block value, while row one sees row zero.
        base[0] = 1.0;
        base[cfg.hidden_size] = 10.0;
        let output = apply_block_local_grouped_conv(&cfg, &hidden, &dynamic, &base, 0).unwrap();
        assert_eq!(output[0], hidden[0]);
        assert_eq!(
            output[cfg.hidden_size],
            hidden[cfg.hidden_size] + 10.0 * hidden[0]
        );
        assert_eq!(
            output[2 * cfg.hidden_size],
            hidden[2 * cfg.hidden_size] + 10.0 * hidden[cfg.hidden_size]
        );
        dynamic[4 * groups] = 2.0;
        let output_with_dynamic =
            apply_block_local_grouped_conv(&cfg, &hidden, &dynamic, &base, 0).unwrap();
        assert_eq!(
            output_with_dynamic[cfg.hidden_size],
            3.0 * hidden[cfg.hidden_size] + 10.0 * hidden[0]
        );
    }

    #[test]
    fn sides_are_independent_and_malformed_inputs_fail_closed() {
        let cfg = config();
        let hidden = vec![1.0; cfg.block_size * cfg.hidden_size];
        let groups = cfg.hidden_size / cfg.conv_group_size;
        let dynamic = vec![0.0; cfg.block_size * 2 * cfg.conv_kernel_size * groups];
        let mut base = vec![0.0; 2 * cfg.conv_kernel_size * cfg.hidden_size];
        base[cfg.hidden_size] = 1.0;
        base[2 * cfg.hidden_size] = 7.0;
        let side0 = apply_block_local_grouped_conv(&cfg, &hidden, &dynamic, &base, 0).unwrap();
        let side1 = apply_block_local_grouped_conv(&cfg, &hidden, &dynamic, &base, 1).unwrap();
        assert_ne!(side0, side1);
        assert!(apply_block_local_grouped_conv(&cfg, &hidden, &dynamic, &base, 2).is_err());
        assert!(apply_block_local_grouped_conv(
            &cfg,
            &hidden[..hidden.len() - 1],
            &dynamic,
            &base,
            0
        )
        .is_err());
        let mut nonfinite = hidden.clone();
        nonfinite[3] = f32::NAN;
        assert!(apply_block_local_grouped_conv(&cfg, &nonfinite, &dynamic, &base, 0).is_err());
    }
}
