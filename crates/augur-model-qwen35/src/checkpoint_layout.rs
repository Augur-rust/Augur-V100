//! Lossless source-row permutations. No scale folding or re-quantization.
use crate::{LoadedCheckpointLinear, Qwen35Error};

fn query_gate_order(heads: usize, dim: usize) -> Vec<usize> {
    (0..2)
        .flat_map(|part| {
            (0..heads).flat_map(move |head| (0..dim).map(move |i| head * 2 * dim + part * dim + i))
        })
        .collect()
}
fn reorder(data: &[u8], row_bytes: usize, order: &[usize]) -> Result<Vec<u8>, Qwen35Error> {
    if row_bytes == 0
        || order.len().checked_mul(row_bytes) != Some(data.len())
        || order.iter().any(|&r| r >= order.len())
    {
        return Err(Qwen35Error::invalid(
            "checkpoint.query_gate",
            "row extent mismatch",
        ));
    }
    let mut out = Vec::with_capacity(data.len());
    for &row in order {
        out.extend_from_slice(&data[row * row_bytes..(row + 1) * row_bytes]);
    }
    Ok(out)
}

impl LoadedCheckpointLinear {
    /// Original `[head][query,gate][dimension]` to `[query,gate][head][dimension]`.
    /// Every packed code and block scale moves together; global/input scales
    /// stay unchanged. The projection output adopts this explicit planar order.
    pub fn separate_query_gate_rows(self) -> Result<Self, Qwen35Error> {
        let spec = self.spec();
        if !spec.name.ends_with(".self_attn.q_proj") || spec.rows != 12288 || spec.cols != 5120 {
            return Err(Qwen35Error::invalid(
                &spec.name,
                "requires full-attention source q/g projection",
            ));
        }
        let order = query_gate_order(24, 256);
        Ok(match self {
            Self::Nvfp4 {
                spec,
                packed,
                scales,
                global,
            } => Self::Nvfp4 {
                packed: reorder(&packed, spec.cols / 2, &order)?,
                scales: reorder(&scales, spec.cols / 16, &order)?,
                spec,
                global,
            },
            Self::Fp8 {
                spec,
                weights,
                weight_scale,
                input_scale,
            } => Self::Fp8 {
                weights: reorder(&weights, spec.cols, &order)?,
                spec,
                weight_scale,
                input_scale,
            },
            Self::Bf16 { spec, weights } => Self::Bf16 {
                weights: reorder(&weights, spec.cols * 2, &order)?,
                spec,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_query_gate_row_permutation_is_exact_and_bijective() {
        let order = query_gate_order(3, 2);
        assert_eq!(order, [0, 1, 4, 5, 8, 9, 2, 3, 6, 7, 10, 11]);
        let mut sorted = order.clone();
        sorted.sort();
        assert_eq!(sorted, (0..12).collect::<Vec<_>>());
        let data = (0..24).collect::<Vec<u8>>();
        let out = reorder(&data, 2, &order).unwrap();
        for (destination, &source) in order.iter().enumerate() {
            assert_eq!(
                &out[destination * 2..destination * 2 + 2],
                &data[source * 2..source * 2 + 2]
            );
        }
        assert!(reorder(&data[..23], 2, &order).is_err());
        assert!(reorder(&data, 0, &order).is_err());
        let mut bad = order;
        bad[0] = 12;
        assert!(reorder(&data, 2, &bad).is_err());
    }
}
