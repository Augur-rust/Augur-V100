//! CPU oracle for the DFlash2 block attention boundary.
//!
//! The draft attention consumes the accepted target context followed by the
//! current block. Queries belong only to the current block; keys and values
//! are split into a context prefix and block suffix.  This helper assumes the
//! caller has already applied Q/K RMSNorm and RoPE, and performs only GQA,
//! visibility, sliding-window masking, stable softmax, and V reduction.
//! DFlash2 is non-causal inside the draft block. The causal entry point is
//! retained only as a legacy arithmetic reference, not a draft-model policy.

use augur_model_qwen35::DFlashConfig;
use std::fmt;

pub const DFLASH_BLOCK_ATTENTION_SCHEMA: &str = "qwen35-dflash-block-causal-gqa-attention-cpu-v1";
pub const DFLASH_BLOCK_NONCAUSAL_ATTENTION_SCHEMA: &str =
    "qwen35-dflash-block-noncausal-standard-swa-attention-cpu-v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DFlashBlockAttentionError {
    Invalid(String),
}

impl fmt::Display for DFlashBlockAttentionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => write!(f, "invalid DFlash block attention: {message}"),
        }
    }
}

impl std::error::Error for DFlashBlockAttentionError {}

/// Absolute positions and left-window policy for a draft block.
/// Extents and overflow are validated by the attention entry points.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DFlashBlockWindow {
    pub context_start_position: usize,
    pub block_start_position: usize,
    pub sliding_window: usize,
}

/// Compute block queries against an accepted context plus the current block.
///
/// `q` has shape `[block, attention_heads, head_dim]`; `k` and `v` have shape
/// `[context + block, kv_heads, head_dim]`.  The context must end immediately
/// before `block_start_position`.  Keys are visible when they are causal and
/// within `sliding_window` absolute positions of the query.
pub fn apply_block_causal_gqa_attention(
    config: &DFlashConfig,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    context_len: usize,
    window: DFlashBlockWindow,
) -> Result<Vec<f32>, DFlashBlockAttentionError> {
    apply_block_gqa_attention(config, q, k, v, context_len, window, true)
}

/// DFlash2 visibility: all current-block keys (including future mask slots)
/// are visible, subject only to the standard *left* sliding-window bound.
/// This is not symmetric SWA and does not expose future target tokens.
pub fn apply_block_noncausal_gqa_attention(
    config: &DFlashConfig,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    context_len: usize,
    window: DFlashBlockWindow,
) -> Result<Vec<f32>, DFlashBlockAttentionError> {
    apply_block_gqa_attention(config, q, k, v, context_len, window, false)
}

fn apply_block_gqa_attention(
    config: &DFlashConfig,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    context_len: usize,
    window: DFlashBlockWindow,
    causal: bool,
) -> Result<Vec<f32>, DFlashBlockAttentionError> {
    let DFlashBlockWindow {
        context_start_position,
        block_start_position,
        sliding_window,
    } = window;
    config
        .validate_exact_profile()
        .map_err(|error| invalid(error.to_string()))?;
    if sliding_window == 0 {
        return Err(invalid("sliding window must be nonzero"));
    }
    if context_start_position
        .checked_add(context_len)
        .ok_or_else(|| invalid("context position overflow"))?
        != block_start_position
    {
        return Err(invalid("context does not end at block start"));
    }
    let block_len = config.block_size;
    if block_start_position
        .checked_add(block_len)
        .is_none_or(|end| end > config.max_position_embeddings)
    {
        return Err(invalid("attention block exceeds position capacity"));
    }
    let q_width = config
        .attention_heads
        .checked_mul(config.head_dim)
        .ok_or_else(|| invalid("query geometry overflow"))?;
    let kv_width = config
        .kv_heads
        .checked_mul(config.head_dim)
        .ok_or_else(|| invalid("key/value geometry overflow"))?;
    let total_len = context_len
        .checked_add(block_len)
        .ok_or_else(|| invalid("attention sequence overflow"))?;
    let expected_q = block_len
        .checked_mul(q_width)
        .ok_or_else(|| invalid("query extent overflow"))?;
    let expected_kv = total_len
        .checked_mul(kv_width)
        .ok_or_else(|| invalid("key/value extent overflow"))?;
    if q.len() != expected_q || k.len() != expected_kv || v.len() != expected_kv {
        return Err(invalid(format!(
            "extent mismatch q={} (expected {}), k={} (expected {}), v={} (expected {})",
            q.len(),
            expected_q,
            k.len(),
            expected_kv,
            v.len(),
            expected_kv
        )));
    }
    if q.iter().chain(k).chain(v).any(|value| !value.is_finite()) {
        return Err(invalid("attention inputs must be finite"));
    }

    let groups = config.attention_heads / config.kv_heads;
    if groups == 0 || !config.attention_heads.is_multiple_of(config.kv_heads) {
        return Err(invalid("attention heads are not divisible by KV heads"));
    }
    let scale = (config.head_dim as f64).sqrt().recip();
    let mut output = vec![0.0_f32; expected_q];

    for token in 0..block_len {
        let query_position = block_start_position
            .checked_add(token)
            .ok_or_else(|| invalid("query position overflow"))?;
        for head in 0..config.attention_heads {
            let kv_head = head / groups;
            let q_base = (token * config.attention_heads + head) * config.head_dim;
            let out_base = q_base;
            let mut max_score = f64::NEG_INFINITY;
            let mut scores = Vec::with_capacity(total_len);
            for key_index in 0..total_len {
                let key_position = if key_index < context_len {
                    context_start_position + key_index
                } else {
                    block_start_position + (key_index - context_len)
                };
                let visible = if key_position > query_position {
                    !causal
                } else {
                    query_position - key_position < sliding_window
                };
                if !visible {
                    scores.push(f64::NEG_INFINITY);
                    continue;
                }
                let k_base = (key_index * config.kv_heads + kv_head) * config.head_dim;
                let mut dot = 0.0_f64;
                for dim in 0..config.head_dim {
                    dot += f64::from(q[q_base + dim]) * f64::from(k[k_base + dim]);
                }
                let score = dot * scale;
                max_score = max_score.max(score);
                scores.push(score);
            }
            if !max_score.is_finite() {
                return Err(invalid("query has no visible key"));
            }
            let mut denominator = 0.0_f64;
            for score in &mut scores {
                if score.is_finite() {
                    *score = (*score - max_score).exp();
                    denominator += *score;
                } else {
                    *score = 0.0;
                }
            }
            if !denominator.is_finite() || denominator <= 0.0 {
                return Err(invalid("attention softmax denominator is invalid"));
            }
            for score in &mut scores {
                *score /= denominator;
            }
            for (key_index, &weight) in scores.iter().enumerate() {
                if weight == 0.0 {
                    continue;
                }
                let v_base = (key_index * config.kv_heads + kv_head) * config.head_dim;
                for dim in 0..config.head_dim {
                    output[out_base + dim] += (weight * f64::from(v[v_base + dim])) as f32;
                }
            }
        }
    }
    if output.iter().any(|value| !value.is_finite()) {
        return Err(invalid("attention output is non-finite"));
    }
    Ok(output)
}

fn invalid(message: impl Into<String>) -> DFlashBlockAttentionError {
    DFlashBlockAttentionError::Invalid(message.into())
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
    fn causal_gqa_averages_only_context_and_current_prefix() {
        let cfg = config();
        let context_len = 1;
        let total_len = context_len + cfg.block_size;
        let q = vec![0.0; cfg.block_size * cfg.attention_heads * cfg.head_dim];
        let mut k = vec![0.0; total_len * cfg.kv_heads * cfg.head_dim];
        let mut v = vec![0.0; total_len * cfg.kv_heads * cfg.head_dim];
        for row in 0..total_len {
            for head in 0..cfg.kv_heads {
                v[(row * cfg.kv_heads + head) * cfg.head_dim] = row as f32 + 1.0;
            }
        }
        let output = apply_block_causal_gqa_attention(
            &cfg,
            &q,
            &k,
            &v,
            1,
            DFlashBlockWindow {
                context_start_position: 99,
                block_start_position: 100,
                sliding_window: 2048,
            },
        )
        .unwrap();
        assert!((output[0] - 1.5).abs() < 1.0e-6);
        assert!((output[cfg.attention_heads * cfg.head_dim] - 2.0).abs() < 1.0e-6);
        assert!((output[7 * cfg.attention_heads * cfg.head_dim] - 5.0).abs() < 1.0e-6);
        k[total_len - 1] = 1000.0;
        let changed = apply_block_causal_gqa_attention(
            &cfg,
            &q,
            &k,
            &v,
            1,
            DFlashBlockWindow {
                context_start_position: 99,
                block_start_position: 100,
                sliding_window: 2048,
            },
        )
        .unwrap();
        assert!((changed[0] - output[0]).abs() < 1.0e-6);
    }

    #[test]
    fn sliding_window_and_geometry_are_fail_closed() {
        let cfg = config();
        let q = vec![0.0; cfg.block_size * cfg.attention_heads * cfg.head_dim];
        let k = vec![0.0; (3 + cfg.block_size) * cfg.kv_heads * cfg.head_dim];
        let v = vec![1.0; (3 + cfg.block_size) * cfg.kv_heads * cfg.head_dim];
        assert!(apply_block_causal_gqa_attention(
            &cfg,
            &q,
            &k,
            &v,
            3,
            DFlashBlockWindow {
                context_start_position: 10,
                block_start_position: 13,
                sliding_window: 2
            }
        )
        .is_ok());
        assert!(apply_block_causal_gqa_attention(
            &cfg,
            &q,
            &k,
            &v,
            3,
            DFlashBlockWindow {
                context_start_position: 11,
                block_start_position: 13,
                sliding_window: 2
            }
        )
        .is_err());
        assert!(apply_block_causal_gqa_attention(
            &cfg,
            &q[..q.len() - 1],
            &k,
            &v,
            3,
            DFlashBlockWindow {
                context_start_position: 10,
                block_start_position: 13,
                sliding_window: 2
            }
        )
        .is_err());
        let mut bad = q.clone();
        bad[0] = f32::NAN;
        assert!(apply_block_causal_gqa_attention(
            &cfg,
            &bad,
            &k,
            &v,
            3,
            DFlashBlockWindow {
                context_start_position: 10,
                block_start_position: 13,
                sliding_window: 2
            }
        )
        .is_err());
    }

    #[test]
    fn noncausal_block_includes_future_mask_slots_and_only_left_swa_bound() {
        let cfg = config();
        let context = 3;
        let q_width = cfg.attention_heads * cfg.head_dim;
        let kv_width = cfg.kv_heads * cfg.head_dim;
        let q = vec![0.0; cfg.block_size * q_width];
        let k = vec![0.0; (context + cfg.block_size) * kv_width];
        let mut v = vec![0.0; k.len()];
        for row in 0..context + cfg.block_size {
            for head in 0..cfg.kv_heads {
                v[row * kv_width + head * cfg.head_dim] = row as f32;
            }
        }
        let actual = apply_block_noncausal_gqa_attention(
            &cfg,
            &q,
            &k,
            &v,
            context,
            DFlashBlockWindow {
                context_start_position: 97,
                block_start_position: 100,
                sliding_window: 2,
            },
        )
        .unwrap();
        // At position 100, positions 99..107 are visible: rows 2..10, mean 6.
        // At position 107 only rows 9 and 10 remain: mean 9.5.
        for head in 0..cfg.attention_heads {
            assert!((actual[head * cfg.head_dim] - 6.0).abs() < 1.0e-5);
            assert!((actual[7 * q_width + head * cfg.head_dim] - 9.5).abs() < 1.0e-5);
        }
        v[10 * kv_width] += 90.0;
        let changed = apply_block_noncausal_gqa_attention(
            &cfg,
            &q,
            &k,
            &v,
            context,
            DFlashBlockWindow {
                context_start_position: 97,
                block_start_position: 100,
                sliding_window: 2,
            },
        )
        .unwrap();
        assert!((changed[0] - actual[0] - 10.0).abs() < 1.0e-5);
        assert_eq!(changed[4 * cfg.head_dim], actual[4 * cfg.head_dim]);
        assert!(apply_block_noncausal_gqa_attention(
            &cfg,
            &q,
            &k,
            &v,
            context,
            DFlashBlockWindow {
                context_start_position: usize::MAX - 3,
                block_start_position: usize::MAX,
                sliding_window: 2
            }
        )
        .is_err());
    }

    #[test]
    fn noncausal_nonzero_scores_match_independent_f64_dense_reference() {
        let cfg = config();
        let qw = cfg.attention_heads * cfg.head_dim;
        let kw = cfg.kv_heads * cfg.head_dim;
        let q = (0..8 * qw)
            .map(|i| (i % 17) as f32 * 0.025 - 0.2)
            .collect::<Vec<_>>();
        let k = (0..11 * kw)
            .map(|i| (i % 23) as f32 * 0.02 - 0.15)
            .collect::<Vec<_>>();
        let v = (0..11 * kw)
            .map(|i| (i % 29) as f32 * 0.125 - 1.0)
            .collect::<Vec<_>>();
        let actual = apply_block_noncausal_gqa_attention(
            &cfg,
            &q,
            &k,
            &v,
            3,
            DFlashBlockWindow {
                context_start_position: 4093,
                block_start_position: 4096,
                sliding_window: 2,
            },
        )
        .unwrap();
        for row in 0..8 {
            for h in 0..cfg.attention_heads {
                let qbase = row * qw + h * cfg.head_dim;
                // Explicit dense mask for these absolute positions: rows
                // [2+query_row, 11), including all future block noise rows.
                let logits = (row + 2..11)
                    .map(|key| {
                        let kb = key * kw + (h / 4) * cfg.head_dim;
                        (0..cfg.head_dim)
                            .map(|d| f64::from(q[qbase + d]) * f64::from(k[kb + d]))
                            .sum::<f64>()
                            / (cfg.head_dim as f64).sqrt()
                    })
                    .collect::<Vec<_>>();
                let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights = logits.iter().map(|l| (l - max).exp()).collect::<Vec<_>>();
                let sum = weights.iter().sum::<f64>();
                for d in 0..cfg.head_dim {
                    let expected = (row + 2..11)
                        .zip(&weights)
                        .map(|(key, w)| {
                            w * f64::from(v[key * kw + (h / 4) * cfg.head_dim + d]) / sum
                        })
                        .sum::<f64>();
                    assert!((f64::from(actual[qbase + d]) - expected).abs() < 1.0e-6);
                }
            }
        }
    }
}
