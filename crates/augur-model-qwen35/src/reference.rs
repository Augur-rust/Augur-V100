use crate::error::Qwen35Error;

/// Small owned CPU tensor used by the Qwen numerical reference path.
///
/// It is deliberately separate from the production graph/tensor types.  The
/// reference owns ordinary f32 values and has no device, allocator, or model
/// loading side effects.
#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceTensor {
    shape: Vec<usize>,
    values: Vec<f32>,
}

impl ReferenceTensor {
    pub fn new(
        shape: impl Into<Vec<usize>>,
        values: impl Into<Vec<f32>>,
    ) -> Result<Self, Qwen35Error> {
        let shape = shape.into();
        let values = values.into();
        let expected = element_count(&shape)?;
        if values.len() != expected {
            return Err(Qwen35Error::invalid(
                "reference.tensor_values",
                format!(
                    "shape {:?} requires {expected} values, got {}",
                    shape,
                    values.len()
                ),
            ));
        }
        if values.iter().any(|value| value.is_nan()) {
            return Err(Qwen35Error::invalid(
                "reference.tensor_values",
                "NaN values are not allowed",
            ));
        }
        Ok(Self { shape, values })
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    pub fn into_values(self) -> Vec<f32> {
        self.values
    }
}

/// Row-major matrix multiplication where `lhs` is `[M, K]` and `rhs` is
/// stored in GGUF-style row-major `[N, K]` form.  The result is `[M, N]`.
pub fn matmul_row_major(
    lhs: &ReferenceTensor,
    rhs: &ReferenceTensor,
) -> Result<ReferenceTensor, Qwen35Error> {
    if lhs.shape.len() != 2 || rhs.shape.len() != 2 {
        return Err(Qwen35Error::invalid(
            "reference.matmul",
            "both operands must be rank-2",
        ));
    }
    let (m, k) = (lhs.shape[0], lhs.shape[1]);
    let (n, rhs_k) = (rhs.shape[0], rhs.shape[1]);
    if k != rhs_k {
        return Err(Qwen35Error::invalid(
            "reference.matmul",
            format!("inner dimensions differ: lhs K={k}, rhs K={rhs_k}"),
        ));
    }
    let output_len = m.checked_mul(n).ok_or(Qwen35Error::SizeOverflow)?;
    let mut values = vec![0.0; output_len];
    for row in 0..m {
        for output in 0..n {
            let mut sum = 0.0f32;
            for column in 0..k {
                sum += lhs.values[row * k + column] * rhs.values[output * k + column];
            }
            values[row * n + output] = sum;
        }
    }
    ReferenceTensor::new(vec![m, n], values)
}

/// Add two tensors elementwise for a residual connection.
///
/// Residual paths are deliberately explicit in the reference layer.  Shape
/// equality and finiteness are checked before the result is committed so a
/// malformed branch cannot silently produce a plausible hidden state.
pub fn add_residual(
    input: &ReferenceTensor,
    residual: &ReferenceTensor,
) -> Result<ReferenceTensor, Qwen35Error> {
    if input.shape != residual.shape {
        return Err(Qwen35Error::invalid(
            "reference.residual.shape",
            format!(
                "input shape {:?} must match residual shape {:?}",
                input.shape, residual.shape
            ),
        ));
    }
    if input
        .values
        .iter()
        .chain(residual.values.iter())
        .any(|value| !value.is_finite())
    {
        return Err(Qwen35Error::invalid(
            "reference.residual.input",
            "residual operands must be finite",
        ));
    }
    let mut values = Vec::with_capacity(input.values.len());
    for (&left, &right) in input.values.iter().zip(&residual.values) {
        let sum = left + right;
        if !sum.is_finite() {
            return Err(Qwen35Error::invalid(
                "reference.residual.output",
                "residual sum is not finite",
            ));
        }
        values.push(sum);
    }
    ReferenceTensor::new(input.shape.clone(), values)
}

/// RMS-normalize each row along the final dimension and apply a 1-D weight.
pub fn rms_norm_last_dim(
    input: &ReferenceTensor,
    weight: &ReferenceTensor,
    epsilon: f32,
) -> Result<ReferenceTensor, Qwen35Error> {
    if input.shape.is_empty() || weight.shape.as_slice() != [*input.shape.last().unwrap()] {
        return Err(Qwen35Error::invalid(
            "reference.rms_norm",
            "weight must be one-dimensional and match the input's final width",
        ));
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(Qwen35Error::invalid(
            "reference.rms_norm.epsilon",
            "epsilon must be finite and positive",
        ));
    }
    if input.values.iter().any(|value| !value.is_finite()) {
        return Err(Qwen35Error::invalid(
            "reference.rms_norm.input",
            "input values must be finite",
        ));
    }
    if weight.values.iter().any(|value| !value.is_finite()) {
        return Err(Qwen35Error::invalid(
            "reference.rms_norm.weight",
            "weight values must be finite",
        ));
    }
    let width = *input.shape.last().unwrap();
    let mut values = Vec::with_capacity(input.values.len());
    for row in input.values.chunks_exact(width) {
        let mut sum_squares = 0.0f32;
        for &value in row {
            let square = value * value;
            if !square.is_finite() {
                return Err(Qwen35Error::invalid(
                    "reference.rms_norm",
                    "sum of squares is not finite",
                ));
            }
            sum_squares += square;
            if !sum_squares.is_finite() {
                return Err(Qwen35Error::invalid(
                    "reference.rms_norm",
                    "sum of squares is not finite",
                ));
            }
        }
        let mean_square = sum_squares / width as f32;
        let scale = (mean_square + epsilon).sqrt().recip();
        if !scale.is_finite() {
            return Err(Qwen35Error::invalid(
                "reference.rms_norm",
                "normalization scale is not finite",
            ));
        }
        for (&value, &factor) in row.iter().zip(weight.values.iter()) {
            let normalized = value * scale * factor;
            if !normalized.is_finite() {
                return Err(Qwen35Error::invalid(
                    "reference.rms_norm",
                    "normalized value is not finite",
                ));
            }
            values.push(normalized);
        }
    }
    ReferenceTensor::new(input.shape.clone(), values)
}

/// L2-normalize each row along the final dimension.
///
/// This follows ggml's `ggml_l2_norm` contract used by Qwen's DeltaNet path:
/// `x * rsqrt(max(sum(x * x), epsilon * epsilon))`.  The epsilon is therefore
/// a lower bound on the vector norm, rather than the mean-square epsilon used
/// by [`rms_norm_last_dim`].  Callers representing GGML's `[width, heads, ...]`
/// layout should make `width` the final dimension of the row-major reference
/// tensor before calling this function.
pub fn l2_norm_last_dim(
    input: &ReferenceTensor,
    epsilon: f32,
) -> Result<ReferenceTensor, Qwen35Error> {
    if input.shape.is_empty() {
        return Err(Qwen35Error::invalid(
            "reference.l2_norm",
            "input must have at least one dimension",
        ));
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(Qwen35Error::invalid(
            "reference.l2_norm.epsilon",
            "epsilon must be finite and positive",
        ));
    }
    let epsilon_squared = epsilon * epsilon;
    if !epsilon_squared.is_finite() || epsilon_squared <= 0.0 {
        return Err(Qwen35Error::invalid(
            "reference.l2_norm.epsilon",
            "epsilon squared must be finite and positive",
        ));
    }
    if input.values.iter().any(|value| !value.is_finite()) {
        return Err(Qwen35Error::invalid(
            "reference.l2_norm.input",
            "input values must be finite",
        ));
    }

    let width = *input.shape.last().unwrap();
    let mut values = Vec::with_capacity(input.values.len());
    for row in input.values.chunks_exact(width) {
        let mut sum_squares = 0.0f32;
        for &value in row {
            let square = value * value;
            if !square.is_finite() {
                return Err(Qwen35Error::invalid(
                    "reference.l2_norm",
                    "sum of squares is not finite",
                ));
            }
            sum_squares += square;
            if !sum_squares.is_finite() {
                return Err(Qwen35Error::invalid(
                    "reference.l2_norm",
                    "sum of squares is not finite",
                ));
            }
        }
        let denominator = sum_squares.max(epsilon_squared);
        let scale = denominator.sqrt().recip();
        if !scale.is_finite() {
            return Err(Qwen35Error::invalid(
                "reference.l2_norm",
                "normalization scale is not finite",
            ));
        }
        for &value in row {
            let normalized = value * scale;
            if !normalized.is_finite() {
                return Err(Qwen35Error::invalid(
                    "reference.l2_norm",
                    "normalized value is not finite",
                ));
            }
            values.push(normalized);
        }
    }
    ReferenceTensor::new(input.shape.clone(), values)
}

/// Apply Qwen's gated recurrent-output normalization:
/// `RMSNorm(input, weight) * SiLU(gate)`.
///
/// The gate is elementwise and must have exactly the same shape as `input`.
/// Keeping this composition as one checked reference operation prevents a
/// future executor from silently applying the gate before normalization.
pub fn rms_norm_gated_last_dim(
    input: &ReferenceTensor,
    weight: &ReferenceTensor,
    gate: &ReferenceTensor,
    epsilon: f32,
) -> Result<ReferenceTensor, Qwen35Error> {
    if gate.shape != input.shape {
        return Err(Qwen35Error::invalid(
            "reference.rms_norm_gated.gate",
            format!(
                "gate shape {:?} must match input shape {:?}",
                gate.shape, input.shape
            ),
        ));
    }
    if gate.values.iter().any(|value| !value.is_finite()) {
        return Err(Qwen35Error::invalid(
            "reference.rms_norm_gated.gate",
            "gate values must be finite",
        ));
    }

    let normalized = rms_norm_last_dim(input, weight, epsilon)?;
    let mut values = Vec::with_capacity(normalized.values.len());
    for (&value, &gate_value) in normalized.values.iter().zip(gate.values.iter()) {
        let gated = value * (gate_value / (1.0 + (-gate_value).exp()));
        if !gated.is_finite() {
            return Err(Qwen35Error::invalid(
                "reference.rms_norm_gated",
                "gated normalized value is not finite",
            ));
        }
        values.push(gated);
    }
    ReferenceTensor::new(input.shape.clone(), values)
}

/// Apply the exact scalar SiLU formula used by the CPU oracle.
pub fn silu(input: &ReferenceTensor) -> Result<ReferenceTensor, Qwen35Error> {
    let values = input
        .values
        .iter()
        .map(|&value| value / (1.0 + (-value).exp()))
        .collect::<Vec<_>>();
    ReferenceTensor::new(input.shape.clone(), values)
}

/// Stable softmax over the final dimension.
pub fn softmax_last_dim(input: &ReferenceTensor) -> Result<ReferenceTensor, Qwen35Error> {
    if input.shape.is_empty() {
        return Err(Qwen35Error::invalid(
            "reference.softmax",
            "softmax requires at least one dimension",
        ));
    }
    let width = *input.shape.last().unwrap();
    let mut output = vec![0.0f32; input.values.len()];
    for (source, target) in input
        .values
        .chunks_exact(width)
        .zip(output.chunks_exact_mut(width))
    {
        let positive_infinity_count = source
            .iter()
            .filter(|value| value.is_infinite() && value.is_sign_positive())
            .count();
        if positive_infinity_count > 0 {
            let probability = 1.0 / positive_infinity_count as f32;
            for (input, output) in source.iter().zip(target.iter_mut()) {
                *output = if input.is_infinite() && input.is_sign_positive() {
                    probability
                } else {
                    0.0
                };
            }
            continue;
        }
        let maximum = source.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if maximum == f32::NEG_INFINITY {
            return Err(Qwen35Error::invalid(
                "reference.softmax",
                "a row cannot contain only negative infinity",
            ));
        }
        let mut denominator = 0.0f32;
        for (input, output) in source.iter().zip(target.iter_mut()) {
            *output = (*input - maximum).exp();
            denominator += *output;
        }
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(Qwen35Error::invalid(
                "reference.softmax",
                "softmax denominator is not finite and positive",
            ));
        }
        for output in target {
            *output /= denominator;
        }
    }
    ReferenceTensor::new(input.shape.clone(), output)
}

/// Gather one token embedding from a GGML row-major `[hidden, vocab]` matrix.
///
/// GGML keeps dimension zero contiguous, so each vocabulary column occupies a
/// contiguous `hidden`-value slice.  Keeping this orientation explicit avoids
/// accidentally treating the converter's `[K, N]` layout as a conventional
/// Rust `[N, K]` matrix.  The returned tensor is shaped `[1, hidden]`.
pub fn embedding_lookup(
    weights: &ReferenceTensor,
    token_id: usize,
) -> Result<ReferenceTensor, Qwen35Error> {
    if weights.shape.len() != 2 || weights.shape[0] == 0 || weights.shape[1] == 0 {
        return Err(Qwen35Error::invalid(
            "reference.embedding.weights",
            format!(
                "expected non-empty rank-2 [hidden, vocab] weights, got {:?}",
                weights.shape
            ),
        ));
    }
    let hidden = weights.shape[0];
    let vocab = weights.shape[1];
    if token_id >= vocab {
        return Err(Qwen35Error::invalid(
            "reference.embedding.token_id",
            format!("token id {token_id} is outside vocabulary size {vocab}"),
        ));
    }
    let start = token_id
        .checked_mul(hidden)
        .ok_or(Qwen35Error::SizeOverflow)?;
    let end = start.checked_add(hidden).ok_or(Qwen35Error::SizeOverflow)?;
    let values = weights
        .values
        .get(start..end)
        .ok_or_else(|| {
            Qwen35Error::invalid(
                "reference.embedding.weights",
                "backing values are truncated",
            )
        })?
        .to_vec();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(Qwen35Error::invalid(
            "reference.embedding.weights",
            "embedding values must be finite",
        ));
    }
    ReferenceTensor::new(vec![1, hidden], values)
}

/// Apply ordinary text RoPE to every row of a tensor along its final
/// dimension.  `rotary_dim` must be even and may be smaller than the head
/// width; untouched tail components are copied verbatim.
///
/// Qwen's multimodal position handling is intentionally outside this helper.
/// This is the checked one-dimensional text path used by the first CPU
/// attention reference.
pub fn rope_interleaved_last_dim(
    input: &ReferenceTensor,
    position: usize,
    rotary_dim: usize,
    theta: f32,
) -> Result<ReferenceTensor, Qwen35Error> {
    rope_with_pairing(input, position, rotary_dim, theta, false)
}

/// Split the rotary prefix in half; MRoPE axis-frequency interleaving is a
/// different concept from coordinate pairing. Nonrotary head tail is kept.
pub fn rope_split_half_last_dim(
    input: &ReferenceTensor,
    position: usize,
    rotary_dim: usize,
    theta: f32,
) -> Result<ReferenceTensor, Qwen35Error> {
    rope_with_pairing(input, position, rotary_dim, theta, true)
}

pub const TARGET_ROPE_SPLIT_HALF: bool = cfg!(feature = "target-split-rope");

/// Explicit compiled target numerical identity; draft RoPE is separate.
pub fn target_rope_last_dim(
    input: &ReferenceTensor,
    position: usize,
    rotary_dim: usize,
    theta: f32,
) -> Result<ReferenceTensor, Qwen35Error> {
    rope_with_pairing(input, position, rotary_dim, theta, TARGET_ROPE_SPLIT_HALF)
}

fn rope_with_pairing(
    input: &ReferenceTensor,
    position: usize,
    rotary_dim: usize,
    theta: f32,
    split_half: bool,
) -> Result<ReferenceTensor, Qwen35Error> {
    if input.shape.is_empty() {
        return Err(Qwen35Error::invalid(
            "reference.rope.input",
            "input must have at least one dimension",
        ));
    }
    let width = *input.shape.last().unwrap();
    if rotary_dim == 0 || !rotary_dim.is_multiple_of(2) || rotary_dim > width {
        return Err(Qwen35Error::invalid(
            "reference.rope.rotary_dim",
            format!("rotary dimension {rotary_dim} must be even and within width {width}"),
        ));
    }
    if !theta.is_finite() || theta <= 1.0 {
        return Err(Qwen35Error::invalid(
            "reference.rope.theta",
            "theta must be finite and greater than one",
        ));
    }
    if input.values.iter().any(|value| !value.is_finite()) {
        return Err(Qwen35Error::invalid(
            "reference.rope.input",
            "input values must be finite",
        ));
    }

    let position = position as f32;
    let mut values = input.values.clone();
    let half = rotary_dim / 2;
    for row in values.chunks_exact_mut(width) {
        for pair in 0..half {
            let exponent = -2.0 * pair as f32 / rotary_dim as f32;
            let angle = position * theta.powf(exponent);
            let (sin, cos) = angle.sin_cos();
            let x_index = if split_half { pair } else { pair * 2 };
            let y_index = if split_half { pair + half } else { x_index + 1 };
            let x = row[x_index];
            let y = row[y_index];
            let rotated_x = x * cos - y * sin;
            let rotated_y = x * sin + y * cos;
            if !rotated_x.is_finite() || !rotated_y.is_finite() {
                return Err(Qwen35Error::invalid(
                    "reference.rope.output",
                    "rotation produced a non-finite value",
                ));
            }
            row[x_index] = rotated_x;
            row[y_index] = rotated_y;
        }
    }
    ReferenceTensor::new(input.shape.clone(), values)
}

/// A request-local grouped-query-attention KV cache for the CPU reference.
///
/// Keys and values are stored token-major as
/// `[position][kv_head][head_dim]`.  Query heads are mapped to contiguous
/// repeated KV groups (`query_head / (query_heads / kv_heads)`), matching the
/// standard GQA layout used by llama.cpp.  The cache owns no allocator or
/// device resource; it is deliberately a small numerical/state contract that
/// a future CUDA owner can implement without changing graph semantics.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen35FullAttentionState {
    key_value_heads: usize,
    head_dim: usize,
    max_positions: usize,
    keys: Vec<f32>,
    values: Vec<f32>,
}

/// Exact CPU checkpoint for [`Qwen35FullAttentionState`].
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen35FullAttentionCheckpoint {
    keys: Vec<f32>,
    values: Vec<f32>,
}

impl Qwen35FullAttentionCheckpoint {
    pub fn keys(&self) -> &[f32] {
        &self.keys
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

impl Qwen35FullAttentionState {
    pub fn zeros(
        key_value_heads: usize,
        head_dim: usize,
        max_positions: usize,
    ) -> Result<Self, Qwen35Error> {
        if key_value_heads == 0 || head_dim == 0 || max_positions == 0 {
            return Err(Qwen35Error::invalid(
                "reference.full_attention.shape",
                "KV head count, head dimension, and maximum positions must be nonzero",
            ));
        }
        let per_token = key_value_heads
            .checked_mul(head_dim)
            .ok_or(Qwen35Error::SizeOverflow)?;
        max_positions
            .checked_mul(per_token)
            .ok_or(Qwen35Error::SizeOverflow)?;
        Ok(Self {
            key_value_heads,
            head_dim,
            max_positions,
            keys: Vec::new(),
            values: Vec::new(),
        })
    }

    pub fn key_value_heads(&self) -> usize {
        self.key_value_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn max_positions(&self) -> usize {
        self.max_positions
    }

    pub fn position_count(&self) -> usize {
        let per_token = self.key_value_heads * self.head_dim;
        self.keys.len() / per_token
    }

    pub fn keys(&self) -> &[f32] {
        &self.keys
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    pub fn checkpoint(&self) -> Qwen35FullAttentionCheckpoint {
        Qwen35FullAttentionCheckpoint {
            keys: self.keys.clone(),
            values: self.values.clone(),
        }
    }

    pub fn restore(
        &mut self,
        checkpoint: &Qwen35FullAttentionCheckpoint,
    ) -> Result<(), Qwen35Error> {
        let per_token = self.key_value_heads * self.head_dim;
        if checkpoint.keys.len() != checkpoint.values.len()
            || !checkpoint.keys.len().is_multiple_of(per_token)
        {
            return Err(Qwen35Error::invalid(
                "reference.full_attention.checkpoint",
                "checkpoint key/value geometry is inconsistent",
            ));
        }
        let positions = checkpoint.keys.len() / per_token;
        if positions > self.max_positions {
            return Err(Qwen35Error::invalid(
                "reference.full_attention.checkpoint",
                format!(
                    "checkpoint has {positions} positions, maximum is {}",
                    self.max_positions
                ),
            ));
        }
        if checkpoint
            .keys
            .iter()
            .chain(checkpoint.values.iter())
            .any(|value| !value.is_finite())
        {
            return Err(Qwen35Error::invalid(
                "reference.full_attention.checkpoint",
                "checkpoint values must be finite",
            ));
        }
        self.keys.clone_from(&checkpoint.keys);
        self.values.clone_from(&checkpoint.values);
        Ok(())
    }

    /// Append one already-projected KV token.  The cache is unchanged on any
    /// validation failure.
    pub fn append_token(&mut self, key: &[f32], value: &[f32]) -> Result<(), Qwen35Error> {
        let per_token = self
            .key_value_heads
            .checked_mul(self.head_dim)
            .ok_or(Qwen35Error::SizeOverflow)?;
        if key.len() != per_token || value.len() != per_token {
            return Err(Qwen35Error::invalid(
                "reference.full_attention.append",
                format!(
                    "expected {per_token} key/value values, got {} and {}",
                    key.len(),
                    value.len()
                ),
            ));
        }
        if self.position_count() >= self.max_positions {
            return Err(Qwen35Error::invalid(
                "reference.full_attention.append",
                "maximum KV positions exceeded",
            ));
        }
        if key.iter().chain(value.iter()).any(|item| !item.is_finite()) {
            return Err(Qwen35Error::invalid(
                "reference.full_attention.append",
                "key/value values must be finite",
            ));
        }
        self.keys.extend_from_slice(key);
        self.values.extend_from_slice(value);
        Ok(())
    }

    /// Append a KV token and attend the supplied query atomically.
    pub fn step(
        &mut self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        query_heads: usize,
    ) -> Result<Vec<f32>, Qwen35Error> {
        let checkpoint = self.checkpoint();
        let result = self
            .append_token(key, value)
            .and_then(|_| self.attend(query, query_heads));
        if result.is_err() {
            let restored = self.restore(&checkpoint);
            debug_assert!(restored.is_ok());
        }
        result
    }

    /// Attend one query against all cached positions, returning
    /// `[query_heads][head_dim]` values.
    pub fn attend(&self, query: &[f32], query_heads: usize) -> Result<Vec<f32>, Qwen35Error> {
        if query_heads == 0 || !query_heads.is_multiple_of(self.key_value_heads) {
            return Err(Qwen35Error::invalid(
                "reference.full_attention.query_heads",
                format!(
                    "query heads {query_heads} must be nonzero and divisible by KV heads {}",
                    self.key_value_heads
                ),
            ));
        }
        let query_values = query_heads
            .checked_mul(self.head_dim)
            .ok_or(Qwen35Error::SizeOverflow)?;
        if query.len() != query_values {
            return Err(Qwen35Error::invalid(
                "reference.full_attention.query",
                format!("expected {query_values} values, got {}", query.len()),
            ));
        }
        let positions = self.position_count();
        if positions == 0 {
            return Err(Qwen35Error::invalid(
                "reference.full_attention.query",
                "cannot attend an empty KV cache",
            ));
        }
        if query.iter().any(|item| !item.is_finite()) {
            return Err(Qwen35Error::invalid(
                "reference.full_attention.query",
                "query values must be finite",
            ));
        }

        let group_size = query_heads / self.key_value_heads;
        let per_token = self.key_value_heads * self.head_dim;
        let scale = (self.head_dim as f32).sqrt().recip();
        let mut output = vec![0.0f32; query_values];
        for query_head in 0..query_heads {
            let kv_head = query_head / group_size;
            let query_start = query_head * self.head_dim;
            let mut scores = Vec::with_capacity(positions);
            for position in 0..positions {
                let key_start = position * per_token + kv_head * self.head_dim;
                let mut score = 0.0f32;
                for component in 0..self.head_dim {
                    score += query[query_start + component] * self.keys[key_start + component];
                }
                score *= scale;
                if !score.is_finite() {
                    return Err(Qwen35Error::invalid(
                        "reference.full_attention.score",
                        "attention score is not finite",
                    ));
                }
                scores.push(score);
            }
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut denominator = 0.0f32;
            for score in &mut scores {
                *score = (*score - maximum).exp();
                denominator += *score;
            }
            if !denominator.is_finite() || denominator <= 0.0 {
                return Err(Qwen35Error::invalid(
                    "reference.full_attention.softmax",
                    "attention denominator is not finite and positive",
                ));
            }
            for score in &mut scores {
                *score /= denominator;
            }
            let output_start = query_start;
            for (position, probability) in scores.into_iter().enumerate() {
                let value_start = position * per_token + kv_head * self.head_dim;
                for component in 0..self.head_dim {
                    output[output_start + component] +=
                        probability * self.values[value_start + component];
                }
            }
        }
        if output.iter().any(|item| !item.is_finite()) {
            return Err(Qwen35Error::invalid(
                "reference.full_attention.output",
                "attention output is not finite",
            ));
        }
        Ok(output)
    }
}

/// Round an f32 to the representable BF16 value, matching the reference
/// executor's round-to-nearest-even boundary behavior.
pub fn bf16_round(value: f32) -> f32 {
    if !value.is_finite() {
        return value;
    }
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000;
    f32::from_bits(rounded)
}

/// Request-local Gated DeltaNet state in the layout used by llama.cpp:
/// `state[value_head][output_component][input_component]`.  Q/K may have
/// fewer heads than V (Qwen uses 16 Q/K heads and 48 V heads); each value head
/// uses `value_head % key_heads` for its Q/K head.  Keeping this layout
/// explicit makes a CPU result directly comparable with a future CUDA kernel.
#[derive(Clone, Debug, PartialEq)]
pub struct GatedDeltaNetState {
    heads: usize,
    value_dim: usize,
    values: Vec<f32>,
}

impl GatedDeltaNetState {
    pub fn zeros(heads: usize, value_dim: usize) -> Result<Self, Qwen35Error> {
        if heads == 0 || value_dim == 0 {
            return Err(Qwen35Error::invalid(
                "reference.gated_delta_net.state",
                "heads and value dimension must be nonzero",
            ));
        }
        let matrix_elements = value_dim
            .checked_mul(value_dim)
            .ok_or(Qwen35Error::SizeOverflow)?;
        let elements = heads
            .checked_mul(matrix_elements)
            .ok_or(Qwen35Error::SizeOverflow)?;
        Ok(Self {
            heads,
            value_dim,
            values: vec![0.0; elements],
        })
    }

    pub fn heads(&self) -> usize {
        self.heads
    }

    pub fn value_dim(&self) -> usize {
        self.value_dim
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Copy the state for an atomic speculative checkpoint.
    pub fn checkpoint(&self) -> Vec<f32> {
        self.values.clone()
    }

    pub fn restore(&mut self, checkpoint: &[f32]) -> Result<(), Qwen35Error> {
        if checkpoint.len() != self.values.len() {
            return Err(Qwen35Error::invalid(
                "reference.gated_delta_net.checkpoint",
                format!(
                    "expected {} values, got {}",
                    self.values.len(),
                    checkpoint.len()
                ),
            ));
        }
        if checkpoint.iter().any(|value| !value.is_finite()) {
            return Err(Qwen35Error::invalid(
                "reference.gated_delta_net.checkpoint",
                "checkpoint contains a non-finite value",
            ));
        }
        self.values.copy_from_slice(checkpoint);
        Ok(())
    }

    /// Apply one causal Gated DeltaNet step.
    ///
    /// `q` and `k` are concatenated key-head-major vectors and `v` is a
    /// value-head-major vector.  Their per-head dimensions must match the
    /// state's `value_dim`, as required by the DeltaNet recurrence.  `beta`
    /// accepts one broadcast value or one value per value head.  `gate`
    /// accepts one broadcast scalar, one scalar per value head, or one decay
    /// value per input component per value head.  The latter is the KDA form;
    /// it scales state columns before the delta update.  The state update is
    /// performed before the returned attention vector is read, matching the
    /// reference kernel.  The returned vector includes the `1 / sqrt(value_dim)`
    /// output scale; callers should provide unscaled Q.  A failed step is
    /// atomic: the receiver is unchanged.
    pub fn step(
        &mut self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        gate: &[f32],
        beta: &[f32],
    ) -> Result<Vec<f32>, Qwen35Error> {
        let value_elements = self
            .heads
            .checked_mul(self.value_dim)
            .ok_or(Qwen35Error::SizeOverflow)?;
        if q.len() != k.len() || q.is_empty() || !q.len().is_multiple_of(self.value_dim) {
            return Err(Qwen35Error::invalid(
                "reference.gated_delta_net.qk",
                format!(
                    "q and k must have equal, nonzero lengths divisible by {}, got {} and {}",
                    self.value_dim,
                    q.len(),
                    k.len()
                ),
            ));
        }
        let key_heads = q.len() / self.value_dim;
        if !self.heads.is_multiple_of(key_heads) {
            return Err(Qwen35Error::invalid(
                "reference.gated_delta_net.heads",
                format!(
                    "value-head count {} must be divisible by key-head count {}",
                    self.heads, key_heads
                ),
            ));
        }
        for (name, values, expected) in [
            ("q", q, q.len()),
            ("k", k, k.len()),
            ("v", v, value_elements),
        ] {
            if values.len() != expected {
                return Err(Qwen35Error::invalid(
                    format!("reference.gated_delta_net.{name}"),
                    format!("expected {expected} values, got {}", values.len()),
                ));
            }
            if values.iter().any(|value| !value.is_finite()) {
                return Err(Qwen35Error::invalid(
                    format!("reference.gated_delta_net.{name}"),
                    "inputs must be finite",
                ));
            }
        }
        let valid_gate_len =
            gate.len() == 1 || gate.len() == self.heads || gate.len() == value_elements;
        if !valid_gate_len {
            return Err(Qwen35Error::invalid(
                "reference.gated_delta_net.gate",
                format!(
                    "expected 1, {}, or {} values, got {}",
                    self.heads,
                    value_elements,
                    gate.len()
                ),
            ));
        }
        let valid_beta_len = beta.len() == 1 || beta.len() == self.heads;
        if !valid_beta_len {
            return Err(Qwen35Error::invalid(
                "reference.gated_delta_net.beta",
                format!("expected 1 or {} values, got {}", self.heads, beta.len()),
            ));
        }
        if gate.iter().any(|value| !value.is_finite())
            || beta.iter().any(|value| !value.is_finite())
        {
            return Err(Qwen35Error::invalid(
                "reference.gated_delta_net.parameters",
                "gate and beta must be finite",
            ));
        }

        // Precompute decays so an invalid exponential cannot partially mutate
        // a multi-head state.
        let mut decays = vec![1.0f32; value_elements];
        let vector_gate = gate.len() == value_elements;
        for head in 0..self.heads {
            if vector_gate {
                for component in 0..self.value_dim {
                    let decay = gate[head * self.value_dim + component].exp();
                    if !decay.is_finite() {
                        return Err(Qwen35Error::invalid(
                            "reference.gated_delta_net.gate",
                            "exp(gate) is not finite",
                        ));
                    }
                    decays[head * self.value_dim + component] = decay;
                }
            } else {
                let gate_value = if gate.len() == 1 { gate[0] } else { gate[head] };
                let decay = gate_value.exp();
                if !decay.is_finite() {
                    return Err(Qwen35Error::invalid(
                        "reference.gated_delta_net.gate",
                        "exp(gate) is not finite",
                    ));
                }
                decays[head * self.value_dim..(head + 1) * self.value_dim].fill(decay);
            }
        }

        let scale = 1.0f32 / (self.value_dim as f32).sqrt();
        let mut output = vec![0.0f32; value_elements];
        let matrix_stride = self
            .value_dim
            .checked_mul(self.value_dim)
            .ok_or(Qwen35Error::SizeOverflow)?;
        let mut next_values = self.values.clone();
        for head in 0..self.heads {
            let matrix = &mut next_values[head * matrix_stride..(head + 1) * matrix_stride];
            let decay = &decays[head * self.value_dim..(head + 1) * self.value_dim];
            for row in matrix.chunks_exact_mut(self.value_dim) {
                for (column, factor) in row.iter_mut().zip(decay) {
                    *column *= *factor;
                }
            }

            let beta_value = if beta.len() == 1 { beta[0] } else { beta[head] };
            let key_head = head % key_heads;
            let q_head = &q[key_head * self.value_dim..(key_head + 1) * self.value_dim];
            let k_head = &k[key_head * self.value_dim..(key_head + 1) * self.value_dim];
            let v_head = &v[head * self.value_dim..(head + 1) * self.value_dim];
            let mut delta = vec![0.0f32; self.value_dim];
            for (row_index, row) in matrix.chunks_exact(self.value_dim).enumerate() {
                let mut projection = 0.0f32;
                for (&state_value, &key) in row.iter().zip(k_head) {
                    projection += state_value * key;
                }
                delta[row_index] = (v_head[row_index] - projection) * beta_value;
            }
            for (row_index, row) in matrix.chunks_exact_mut(self.value_dim).enumerate() {
                for (column, &key) in row.iter_mut().zip(k_head) {
                    *column += key * delta[row_index];
                }
            }
            for (row_index, row) in matrix.chunks_exact(self.value_dim).enumerate() {
                let mut projection = 0.0f32;
                for (&state_value, &query) in row.iter().zip(q_head) {
                    projection += state_value * query;
                }
                output[head * self.value_dim + row_index] = projection * scale;
            }
        }
        if next_values.iter().any(|value| !value.is_finite())
            || output.iter().any(|value| !value.is_finite())
        {
            return Err(Qwen35Error::invalid(
                "reference.gated_delta_net",
                "state update produced a non-finite value",
            ));
        }
        self.values = next_values;
        Ok(output)
    }
}

/// Request-local causal depthwise-convolution history used before the
/// DeltaNet recurrence.  The flat storage follows GGML's contiguous layout:
/// `history[channel][tap]`, oldest tap first within each channel.  This
/// matches llama.cpp's `ggml_ssm_conv` input window and keeps the state
/// independent from the graph or device executor.
#[derive(Clone, Debug, PartialEq)]
pub struct CausalConv1dState {
    kernel_size: usize,
    channels: usize,
    history: Vec<f32>,
}

impl CausalConv1dState {
    pub fn zeros(kernel_size: usize, channels: usize) -> Result<Self, Qwen35Error> {
        if kernel_size == 0 || channels == 0 {
            return Err(Qwen35Error::invalid(
                "reference.causal_conv1d.state",
                "kernel size and channel count must be nonzero",
            ));
        }
        let history_len = kernel_size
            .checked_sub(1)
            .and_then(|taps| taps.checked_mul(channels))
            .ok_or(Qwen35Error::SizeOverflow)?;
        Ok(Self {
            kernel_size,
            channels,
            history: vec![0.0; history_len],
        })
    }

    pub fn kernel_size(&self) -> usize {
        self.kernel_size
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn history(&self) -> &[f32] {
        &self.history
    }

    pub fn checkpoint(&self) -> Vec<f32> {
        self.history.clone()
    }

    pub fn restore(&mut self, checkpoint: &[f32]) -> Result<(), Qwen35Error> {
        if checkpoint.len() != self.history.len() {
            return Err(Qwen35Error::invalid(
                "reference.causal_conv1d.checkpoint",
                format!(
                    "expected {} values, got {}",
                    self.history.len(),
                    checkpoint.len()
                ),
            ));
        }
        if checkpoint.iter().any(|value| !value.is_finite()) {
            return Err(Qwen35Error::invalid(
                "reference.causal_conv1d.checkpoint",
                "checkpoint contains a non-finite value",
            ));
        }
        self.history.copy_from_slice(checkpoint);
        Ok(())
    }

    /// Apply one causal depthwise-convolution step.
    ///
    /// `input` has one value per channel.  `weights` is GGML-contiguous
    /// `[channel][kernel_tap]`, with tap zero oldest and the final tap the
    /// current input.  The returned vector is the raw convolution output;
    /// Qwen applies SiLU as a separate operation.  A failed step is atomic.
    pub fn step(&mut self, input: &[f32], weights: &[f32]) -> Result<Vec<f32>, Qwen35Error> {
        validate_conv_inputs(self.kernel_size, self.channels, input, weights)?;
        let mut next_history = self.history.clone();
        let output = conv_step_on_history(
            self.kernel_size,
            self.channels,
            &mut next_history,
            input,
            weights,
        )?;
        self.history = next_history;
        Ok(output)
    }

    /// Process a contiguous `[tokens][channels]` input block.  The whole
    /// block commits atomically, which makes it suitable for a speculative
    /// prefill/checkpoint boundary.
    pub fn process(&mut self, inputs: &[f32], weights: &[f32]) -> Result<Vec<f32>, Qwen35Error> {
        if !inputs.len().is_multiple_of(self.channels) {
            return Err(Qwen35Error::invalid(
                "reference.causal_conv1d.input",
                format!(
                    "input length {} is not divisible by {} channels",
                    inputs.len(),
                    self.channels
                ),
            ));
        }
        validate_conv_weights(self.kernel_size, self.channels, weights)?;
        if inputs.iter().any(|value| !value.is_finite()) {
            return Err(Qwen35Error::invalid(
                "reference.causal_conv1d.input",
                "inputs must be finite",
            ));
        }

        let token_count = inputs.len() / self.channels;
        let mut next_history = self.history.clone();
        let mut output = Vec::with_capacity(inputs.len());
        for token in inputs.chunks_exact(self.channels) {
            output.extend(conv_step_on_history(
                self.kernel_size,
                self.channels,
                &mut next_history,
                token,
                weights,
            )?);
        }
        debug_assert_eq!(output.len(), token_count * self.channels);
        self.history = next_history;
        Ok(output)
    }
}

/// Checkpoint for [`Qwen35LinearAttentionBlock`].
///
/// The two state vectors are kept separate because the convolution history
/// and DeltaNet matrix have different layouts and lifetimes in the model
/// runtime.  A checkpoint is an owned, immutable snapshot suitable for a
/// speculative rollback boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen35LinearAttentionCheckpoint {
    conv_history: Vec<f32>,
    gated_delta_values: Vec<f32>,
}

impl Qwen35LinearAttentionCheckpoint {
    pub fn conv_history(&self) -> &[f32] {
        &self.conv_history
    }

    pub fn gated_delta_values(&self) -> &[f32] {
        &self.gated_delta_values
    }
}

/// Borrowed inputs and numerical parameters for one
/// [`Qwen35LinearAttentionBlock::step`] call.
#[derive(Clone, Copy, Debug)]
pub struct Qwen35LinearAttentionInputs<'a> {
    pub qkv_mixed: &'a [f32],
    pub conv_weights: &'a [f32],
    pub z: &'a [f32],
    pub decay_gate: &'a [f32],
    pub beta: &'a [f32],
    pub norm_weight: &'a [f32],
    pub epsilon: f32,
}

/// CPU-only composition of one Qwen35 linear-attention (Gated DeltaNet) step.
///
/// The block starts after the learned projections and deliberately owns only
/// request-local recurrent state.  Its step order mirrors llama.cpp's
/// `build_layer_attn_linear` path:
///
/// ```text
/// causal depthwise conv -> SiLU -> split Q/K/V -> L2(Q/K)
/// -> Gated DeltaNet -> RMSNorm -> SiLU(z) gate
/// ```
///
/// `qkv_mixed` is the contiguous `[Q, K, V]` projection output, with Q and K
/// containing `key_heads * head_dim` values each and V containing
/// `value_heads * head_dim` values.  `conv_weights` uses the GGML-contiguous
/// `[channel][tap]` layout accepted by [`CausalConv1dState`].  The returned
/// vector is the gated recurrent output in value-head-major order; the learned
/// `ssm_out` projection is intentionally left to a separately checked matmul
/// so its GGUF orientation remains explicit.  A failed step restores both
/// recurrent states atomically.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen35LinearAttentionBlock {
    conv: CausalConv1dState,
    gated_delta: GatedDeltaNetState,
    key_heads: usize,
    value_heads: usize,
    head_dim: usize,
}

impl Qwen35LinearAttentionBlock {
    pub fn zeros(
        kernel_size: usize,
        key_heads: usize,
        value_heads: usize,
        head_dim: usize,
    ) -> Result<Self, Qwen35Error> {
        if key_heads == 0 || value_heads == 0 || head_dim == 0 {
            return Err(Qwen35Error::invalid(
                "reference.linear_attention.shape",
                "head counts and head dimension must be nonzero",
            ));
        }
        if !value_heads.is_multiple_of(key_heads) {
            return Err(Qwen35Error::invalid(
                "reference.linear_attention.heads",
                format!(
                    "value-head count {value_heads} must be divisible by key-head count {key_heads}"
                ),
            ));
        }
        let key_values = key_heads
            .checked_mul(head_dim)
            .ok_or(Qwen35Error::SizeOverflow)?;
        let value_values = value_heads
            .checked_mul(head_dim)
            .ok_or(Qwen35Error::SizeOverflow)?;
        let conv_channels = key_values
            .checked_mul(2)
            .and_then(|values| values.checked_add(value_values))
            .ok_or(Qwen35Error::SizeOverflow)?;
        Ok(Self {
            conv: CausalConv1dState::zeros(kernel_size, conv_channels)?,
            gated_delta: GatedDeltaNetState::zeros(value_heads, head_dim)?,
            key_heads,
            value_heads,
            head_dim,
        })
    }

    pub fn key_heads(&self) -> usize {
        self.key_heads
    }

    pub fn value_heads(&self) -> usize {
        self.value_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn conv_channels(&self) -> usize {
        self.conv.channels()
    }

    pub fn conv_history(&self) -> &[f32] {
        self.conv.history()
    }

    pub fn gated_delta_values(&self) -> &[f32] {
        self.gated_delta.values()
    }

    pub fn checkpoint(&self) -> Qwen35LinearAttentionCheckpoint {
        Qwen35LinearAttentionCheckpoint {
            conv_history: self.conv.checkpoint(),
            gated_delta_values: self.gated_delta.checkpoint(),
        }
    }

    pub fn restore(
        &mut self,
        checkpoint: &Qwen35LinearAttentionCheckpoint,
    ) -> Result<(), Qwen35Error> {
        if checkpoint.conv_history.len() != self.conv.history().len() {
            return Err(Qwen35Error::invalid(
                "reference.linear_attention.checkpoint",
                format!(
                    "expected {} convolution values, got {}",
                    self.conv.history().len(),
                    checkpoint.conv_history.len()
                ),
            ));
        }
        if checkpoint.gated_delta_values.len() != self.gated_delta.values().len() {
            return Err(Qwen35Error::invalid(
                "reference.linear_attention.checkpoint",
                format!(
                    "expected {} DeltaNet values, got {}",
                    self.gated_delta.values().len(),
                    checkpoint.gated_delta_values.len()
                ),
            ));
        }
        if checkpoint
            .conv_history
            .iter()
            .chain(checkpoint.gated_delta_values.iter())
            .any(|value| !value.is_finite())
        {
            return Err(Qwen35Error::invalid(
                "reference.linear_attention.checkpoint",
                "checkpoint values must be finite",
            ));
        }

        // Length and finiteness were checked above, so these two operations
        // cannot fail independently and leave a partially restored block.
        self.conv.restore(&checkpoint.conv_history)?;
        self.gated_delta.restore(&checkpoint.gated_delta_values)?;
        Ok(())
    }

    /// Execute one post-projection linear-attention step.
    pub fn step(
        &mut self,
        inputs: Qwen35LinearAttentionInputs<'_>,
    ) -> Result<Vec<f32>, Qwen35Error> {
        let checkpoint = self.checkpoint();
        let result = self.step_inner(inputs);
        if result.is_err() {
            // The checkpoint dimensions are intrinsic to self, so rollback
            // cannot fail unless an internal invariant has been corrupted.
            let restored = self.restore(&checkpoint);
            debug_assert!(restored.is_ok());
        }
        result
    }

    /// Execute one step and apply an explicit GGUF-row-major `ssm_out`
    /// projection.  `output_weight` must have shape `[hidden, value_values]`;
    /// the returned tensor has shape `[1, hidden]`.  The projection is kept out
    /// of [`Self::step`] so callers can inspect the recurrent vector before
    /// committing it to the model hidden width.
    pub fn step_projected(
        &mut self,
        inputs: Qwen35LinearAttentionInputs<'_>,
        output_weight: &ReferenceTensor,
    ) -> Result<ReferenceTensor, Qwen35Error> {
        let value_values = self
            .value_heads
            .checked_mul(self.head_dim)
            .ok_or(Qwen35Error::SizeOverflow)?;
        if output_weight.shape.len() != 2 || output_weight.shape[1] != value_values {
            return Err(Qwen35Error::invalid(
                "reference.linear_attention.output_weight",
                format!(
                    "expected rank-2 shape [hidden, {value_values}], got {:?}",
                    output_weight.shape
                ),
            ));
        }
        if output_weight.values.iter().any(|value| !value.is_finite()) {
            return Err(Qwen35Error::invalid(
                "reference.linear_attention.output_weight",
                "weights must be finite",
            ));
        }

        let checkpoint = self.checkpoint();
        let result = self.step(inputs).and_then(|recurrent| {
            let recurrent = ReferenceTensor::new(vec![1, value_values], recurrent)?;
            let projected = matmul_row_major(&recurrent, output_weight)?;
            if projected.values.iter().any(|value| !value.is_finite()) {
                return Err(Qwen35Error::invalid(
                    "reference.linear_attention.output",
                    "output projection produced a non-finite value",
                ));
            }
            Ok(projected)
        });
        if result.is_err() {
            let restored = self.restore(&checkpoint);
            debug_assert!(restored.is_ok());
        }
        result
    }

    /// Process a contiguous prefill batch of post-projection inputs.
    ///
    /// Each item advances the same request-local recurrent state.  Results are
    /// returned as a flat `[tokens][value_heads * head_dim]` vector.  The
    /// entire batch is atomic: a failure in any later token restores the
    /// convolution history and DeltaNet matrix from before the call.
    pub fn process(
        &mut self,
        inputs: &[Qwen35LinearAttentionInputs<'_>],
    ) -> Result<Vec<f32>, Qwen35Error> {
        let value_values = self
            .value_heads
            .checked_mul(self.head_dim)
            .ok_or(Qwen35Error::SizeOverflow)?;
        let output_capacity = inputs
            .len()
            .checked_mul(value_values)
            .ok_or(Qwen35Error::SizeOverflow)?;
        let checkpoint = self.checkpoint();
        let mut output = Vec::with_capacity(output_capacity);
        for &input in inputs {
            match self.step_inner(input) {
                Ok(values) => output.extend(values),
                Err(error) => {
                    let restored = self.restore(&checkpoint);
                    debug_assert!(restored.is_ok());
                    return Err(error);
                }
            }
        }
        debug_assert_eq!(output.len(), output_capacity);
        Ok(output)
    }

    fn step_inner(
        &mut self,
        inputs: Qwen35LinearAttentionInputs<'_>,
    ) -> Result<Vec<f32>, Qwen35Error> {
        let Qwen35LinearAttentionInputs {
            qkv_mixed,
            conv_weights,
            z,
            decay_gate,
            beta,
            norm_weight,
            epsilon,
        } = inputs;
        let key_values = self
            .key_heads
            .checked_mul(self.head_dim)
            .ok_or(Qwen35Error::SizeOverflow)?;
        let value_values = self
            .value_heads
            .checked_mul(self.head_dim)
            .ok_or(Qwen35Error::SizeOverflow)?;
        let expected_qkv = key_values
            .checked_mul(2)
            .and_then(|values| values.checked_add(value_values))
            .ok_or(Qwen35Error::SizeOverflow)?;
        if qkv_mixed.len() != expected_qkv {
            return Err(Qwen35Error::invalid(
                "reference.linear_attention.qkv",
                format!("expected {expected_qkv} values, got {}", qkv_mixed.len()),
            ));
        }
        if z.len() != value_values {
            return Err(Qwen35Error::invalid(
                "reference.linear_attention.z",
                format!("expected {value_values} values, got {}", z.len()),
            ));
        }
        if norm_weight.len() != self.head_dim {
            return Err(Qwen35Error::invalid(
                "reference.linear_attention.norm_weight",
                format!(
                    "expected {} values, got {}",
                    self.head_dim,
                    norm_weight.len()
                ),
            ));
        }
        for (name, values) in [
            ("qkv", qkv_mixed),
            ("z", z),
            ("decay_gate", decay_gate),
            ("beta", beta),
            ("norm_weight", norm_weight),
        ] {
            if values.iter().any(|value| !value.is_finite()) {
                return Err(Qwen35Error::invalid(
                    format!("reference.linear_attention.{name}"),
                    "values must be finite",
                ));
            }
        }

        let conv_output = self.conv.step(qkv_mixed, conv_weights)?;
        let mut activated = Vec::with_capacity(conv_output.len());
        for value in conv_output {
            let output = value / (1.0 + (-value).exp());
            if !output.is_finite() {
                return Err(Qwen35Error::invalid(
                    "reference.linear_attention.conv_silu",
                    "activation produced a non-finite value",
                ));
            }
            activated.push(output);
        }

        let q = ReferenceTensor::new(
            vec![self.key_heads, self.head_dim],
            activated[..key_values].to_vec(),
        )?;
        let k = ReferenceTensor::new(
            vec![self.key_heads, self.head_dim],
            activated[key_values..key_values * 2].to_vec(),
        )?;
        let q = l2_norm_last_dim(&q, epsilon)?;
        let k = l2_norm_last_dim(&k, epsilon)?;
        let v = &activated[key_values * 2..];

        let recurrent = self
            .gated_delta
            .step(q.values(), k.values(), v, decay_gate, beta)?;
        let recurrent = ReferenceTensor::new(vec![self.value_heads, self.head_dim], recurrent)?;
        let z = ReferenceTensor::new(vec![self.value_heads, self.head_dim], z.to_vec())?;
        let norm_weight = ReferenceTensor::new(vec![self.head_dim], norm_weight.to_vec())?;
        Ok(rms_norm_gated_last_dim(&recurrent, &norm_weight, &z, epsilon)?.into_values())
    }
}

fn validate_conv_inputs(
    kernel_size: usize,
    channels: usize,
    input: &[f32],
    weights: &[f32],
) -> Result<(), Qwen35Error> {
    if input.len() != channels {
        return Err(Qwen35Error::invalid(
            "reference.causal_conv1d.input",
            format!("expected {channels} values, got {}", input.len()),
        ));
    }
    if input.iter().any(|value| !value.is_finite()) {
        return Err(Qwen35Error::invalid(
            "reference.causal_conv1d.input",
            "inputs must be finite",
        ));
    }
    validate_conv_weights(kernel_size, channels, weights)
}

fn validate_conv_weights(
    kernel_size: usize,
    channels: usize,
    weights: &[f32],
) -> Result<(), Qwen35Error> {
    let expected = kernel_size
        .checked_mul(channels)
        .ok_or(Qwen35Error::SizeOverflow)?;
    if weights.len() != expected {
        return Err(Qwen35Error::invalid(
            "reference.causal_conv1d.weights",
            format!("expected {expected} values, got {}", weights.len()),
        ));
    }
    if weights.iter().any(|value| !value.is_finite()) {
        return Err(Qwen35Error::invalid(
            "reference.causal_conv1d.weights",
            "weights must be finite",
        ));
    }
    Ok(())
}

fn conv_step_on_history(
    kernel_size: usize,
    channels: usize,
    history: &mut [f32],
    input: &[f32],
    weights: &[f32],
) -> Result<Vec<f32>, Qwen35Error> {
    let mut output = vec![0.0f32; channels];
    for channel in 0..channels {
        let mut sum = 0.0f32;
        for tap in 0..kernel_size {
            let sample = if tap + 1 == kernel_size {
                input[channel]
            } else {
                history[channel * (kernel_size - 1) + tap]
            };
            sum += sample * weights[channel * kernel_size + tap];
        }
        if !sum.is_finite() {
            return Err(Qwen35Error::invalid(
                "reference.causal_conv1d",
                "convolution output is not finite",
            ));
        }
        output[channel] = sum;
    }

    if kernel_size > 1 {
        let history_taps = kernel_size - 1;
        for (channel, &value) in input.iter().enumerate() {
            let start = channel * history_taps;
            let end = start + history_taps;
            history.copy_within((start + 1)..end, start);
            history[end - 1] = value;
        }
    }
    Ok(output)
}

fn element_count(shape: &[usize]) -> Result<usize, Qwen35Error> {
    shape.iter().try_fold(1usize, |count, &dimension| {
        if dimension == 0 {
            return Err(Qwen35Error::invalid(
                "reference.shape",
                "dimensions must be nonzero",
            ));
        }
        count
            .checked_mul(dimension)
            .ok_or(Qwen35Error::SizeOverflow)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_uses_gguf_row_orientation() {
        let lhs = ReferenceTensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let rhs = ReferenceTensor::new(vec![2, 3], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let output = matmul_row_major(&lhs, &rhs).unwrap();
        assert_eq!(output.shape(), &[2, 2]);
        assert_eq!(output.values(), &[50.0, 68.0, 122.0, 167.0]);
    }

    #[test]
    fn residual_add_requires_matching_finite_shapes() {
        let input = ReferenceTensor::new(vec![1, 2], vec![1.0, 2.0]).unwrap();
        let residual = ReferenceTensor::new(vec![1, 2], vec![3.0, 4.0]).unwrap();
        assert_eq!(
            add_residual(&input, &residual).unwrap().values(),
            &[4.0, 6.0]
        );

        let wrong_shape = ReferenceTensor::new(vec![2], vec![3.0, 4.0]).unwrap();
        assert!(add_residual(&input, &wrong_shape).is_err());
        let overflowing = ReferenceTensor::new(vec![1, 2], vec![f32::MAX, 0.0]).unwrap();
        let large_residual = ReferenceTensor::new(vec![1, 2], vec![f32::MAX, 0.0]).unwrap();
        assert!(add_residual(&overflowing, &large_residual).is_err());
    }

    #[test]
    fn rms_norm_and_silu_are_deterministic() {
        let input = ReferenceTensor::new(vec![1, 2], vec![3.0, 4.0]).unwrap();
        let weight = ReferenceTensor::new(vec![2], vec![1.0, 2.0]).unwrap();
        let normalized = rms_norm_last_dim(&input, &weight, 1.0e-6).unwrap();
        assert!((normalized.values()[0] - 0.8485281).abs() < 1.0e-6);
        assert!((normalized.values()[1] - 2.2627416).abs() < 1.0e-6);
        let activated = silu(&input).unwrap();
        assert!((activated.values()[0] - 2.8577223).abs() < 1.0e-6);
    }

    #[test]
    fn embedding_lookup_uses_gguf_column_contiguity() {
        let weights = ReferenceTensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(embedding_lookup(&weights, 0).unwrap().values(), &[1.0, 2.0]);
        assert_eq!(embedding_lookup(&weights, 1).unwrap().values(), &[3.0, 4.0]);
        assert_eq!(embedding_lookup(&weights, 2).unwrap().values(), &[5.0, 6.0]);
        assert!(embedding_lookup(&weights, 3).is_err());
        let wrong_rank =
            ReferenceTensor::new(vec![6], (1..=6).map(|v| v as f32).collect::<Vec<_>>()).unwrap();
        assert!(embedding_lookup(&wrong_rank, 0).is_err());
        let nonfinite = ReferenceTensor::new(vec![2, 1], vec![f32::INFINITY, 0.0]).unwrap();
        assert!(embedding_lookup(&nonfinite, 0).is_err());
    }

    #[test]
    fn split_half_rope_is_not_adjacent_and_respects_compiled_target_policy() {
        let input = ReferenceTensor::new(vec![1, 6], vec![1., 0., 0., 0., 7., 8.]).unwrap();
        let split = rope_split_half_last_dim(&input, 1, 4, 10000.).unwrap();
        assert!((split.values()[0] - 1f32.cos()).abs() < 1e-7);
        assert!((split.values()[2] - 1f32.sin()).abs() < 1e-7);
        assert_eq!(split.values()[1], 0.);
        assert_eq!(&split.values()[4..], &[7., 8.]);
        let adjacent = rope_interleaved_last_dim(&input, 1, 4, 10000.).unwrap();
        assert_ne!(split.values(), adjacent.values());
        let selected = target_rope_last_dim(&input, 1, 4, 10000.).unwrap();
        assert_eq!(
            selected.values(),
            if TARGET_ROPE_SPLIT_HALF {
                split.values()
            } else {
                adjacent.values()
            }
        );
        for dim in [0, 3, 8] {
            assert!(rope_split_half_last_dim(&input, 1, dim, 10000.).is_err());
        }
    }

    #[test]
    fn split_half_rope_covers_target_heads_positions_and_unrotated_tail() {
        let values: Vec<f32> = (0..512)
            .map(|i| ((i * 37 % 127) as f32 - 63.) / 32.)
            .collect();
        let input = ReferenceTensor::new(vec![2, 256], values.clone()).unwrap();
        for position in [0, 1, 5638, 196607, 262136] {
            let rotated = rope_split_half_last_dim(&input, position, 64, 10000000.).unwrap();
            for head in 0..2 {
                let base = head * 256;
                assert_eq!(
                    &rotated.values()[base + 64..base + 256],
                    &values[base + 64..base + 256]
                );
                for pair in 0..32 {
                    let angle = position as f32 * 10000000f32.powf(-2. * pair as f32 / 64.);
                    let (s, c) = angle.sin_cos();
                    let x = values[base + pair];
                    let y = values[base + pair + 32];
                    assert!((rotated.values()[base + pair] - (x * c - y * s)).abs() < 1e-6);
                    assert!((rotated.values()[base + pair + 32] - (x * s + y * c)).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn rope_interleaved_rotates_pairs_and_preserves_tail() {
        let input = ReferenceTensor::new(vec![1, 4], vec![1.0, 0.0, 7.0, 8.0]).unwrap();
        let at_zero = rope_interleaved_last_dim(&input, 0, 2, 10_000.0).unwrap();
        assert_eq!(at_zero.values(), input.values());
        let at_one = rope_interleaved_last_dim(&input, 1, 2, 10_000.0).unwrap();
        assert!((at_one.values()[0] - 1.0f32.cos()).abs() < 1.0e-6);
        assert!((at_one.values()[1] - 1.0f32.sin()).abs() < 1.0e-6);
        assert_eq!(&at_one.values()[2..], &[7.0, 8.0]);
        assert!(rope_interleaved_last_dim(&input, 0, 3, 10_000.0).is_err());
        assert!(rope_interleaved_last_dim(&input, 0, 6, 10_000.0).is_err());
        assert!(rope_interleaved_last_dim(&input, 0, 2, 1.0).is_err());
    }

    #[test]
    fn full_attention_maps_contiguous_query_groups_to_kv_heads() {
        let mut state = Qwen35FullAttentionState::zeros(2, 1, 2).unwrap();
        state.append_token(&[1.0, 0.0], &[10.0, 20.0]).unwrap();
        state.append_token(&[0.0, 1.0], &[30.0, 40.0]).unwrap();
        let output = state.attend(&[0.0, 0.0, 0.0, 0.0], 4).unwrap();
        // Zero queries produce uniform probabilities.  Query heads 0/1 share
        // KV head 0, while heads 2/3 share KV head 1.
        assert_eq!(output, vec![20.0, 20.0, 30.0, 30.0]);
        assert_eq!(state.position_count(), 2);
        assert_eq!(state.keys(), &[1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn full_attention_step_is_atomic_and_checkpointable() {
        let mut state = Qwen35FullAttentionState::zeros(1, 2, 1).unwrap();
        let before = state.checkpoint();
        // The append succeeds, but the malformed query makes attention fail;
        // step() must restore the empty cache.
        assert!(state.step(&[0.0], &[1.0, 0.0], &[2.0, 3.0], 1).is_err());
        assert_eq!(state.position_count(), 0);
        assert_eq!(state.checkpoint(), before);

        state.append_token(&[1.0, 0.0], &[2.0, 3.0]).unwrap();
        let checkpoint = state.checkpoint();
        assert!(state
            .append_token(&[f32::INFINITY, 0.0], &[2.0, 3.0])
            .is_err());
        assert_eq!(state.checkpoint(), checkpoint);
        state.restore(&before).unwrap();
        assert_eq!(state.position_count(), 0);
        assert!(state.append_token(&[1.0, 0.0], &[2.0, 3.0]).is_ok());
        assert!(state.append_token(&[1.0, 0.0], &[2.0, 3.0]).is_err());
    }

    #[test]
    fn l2_norm_matches_ggml_row_formula_and_epsilon_floor() {
        let input = ReferenceTensor::new(vec![2, 2], vec![3.0, 4.0, 0.0, 5.0]).unwrap();
        let output = l2_norm_last_dim(&input, 1.0e-6).unwrap();
        assert!((output.values()[0] - 0.6).abs() < 1.0e-6);
        assert!((output.values()[1] - 0.8).abs() < 1.0e-6);
        assert_eq!(output.values()[2..], [0.0, 1.0]);

        // ggml clamps the norm denominator to epsilon, so a zero row remains
        // finite and zero instead of producing a divide-by-zero NaN.
        let zero = ReferenceTensor::new(vec![1, 3], vec![0.0, 0.0, 0.0]).unwrap();
        assert_eq!(l2_norm_last_dim(&zero, 0.25).unwrap().values(), &[0.0; 3]);
    }

    #[test]
    fn l2_norm_rejects_invalid_epsilon_and_nonfinite_reduction() {
        let input = ReferenceTensor::new(vec![1, 2], vec![1.0, 2.0]).unwrap();
        assert!(l2_norm_last_dim(&input, 0.0).is_err());
        assert!(l2_norm_last_dim(&input, f32::NAN).is_err());
        assert!(l2_norm_last_dim(&input, f32::MAX).is_err());

        let nonfinite = ReferenceTensor::new(vec![1, 2], vec![f32::INFINITY, 0.0]).unwrap();
        assert!(l2_norm_last_dim(&nonfinite, 1.0e-6).is_err());
        let overflowing = ReferenceTensor::new(vec![1, 2], vec![f32::MAX, f32::MAX]).unwrap();
        assert!(l2_norm_last_dim(&overflowing, 1.0e-6).is_err());
    }

    #[test]
    fn rms_norm_gated_applies_silu_after_normalization() {
        let input = ReferenceTensor::new(vec![1, 2], vec![1.0, 0.0]).unwrap();
        let weight = ReferenceTensor::new(vec![2], vec![1.0, 1.0]).unwrap();
        let gate = ReferenceTensor::new(vec![1, 2], vec![1.0, 0.0]).unwrap();
        let output = rms_norm_gated_last_dim(&input, &weight, &gate, 1.0e-6).unwrap();
        let expected = (0.5f32 + 1.0e-6).sqrt().recip() * (1.0 / (1.0 + (-1.0f32).exp()));
        assert!((output.values()[0] - expected).abs() < 2.0e-6);
        assert_eq!(output.values()[1], 0.0);
    }

    #[test]
    fn rms_norm_gated_rejects_shape_and_finite_contract_violations() {
        let input = ReferenceTensor::new(vec![1, 2], vec![1.0, 0.0]).unwrap();
        let weight = ReferenceTensor::new(vec![2], vec![1.0, 1.0]).unwrap();
        let wrong_shape = ReferenceTensor::new(vec![2], vec![0.0, 0.0]).unwrap();
        assert!(rms_norm_gated_last_dim(&input, &weight, &wrong_shape, 1.0e-6).is_err());

        let bad_gate = ReferenceTensor::new(vec![1, 2], vec![f32::INFINITY, 0.0]).unwrap();
        assert!(rms_norm_gated_last_dim(&input, &weight, &bad_gate, 1.0e-6).is_err());
        let bad_weight = ReferenceTensor::new(vec![2], vec![f32::INFINITY, 1.0]).unwrap();
        let gate = ReferenceTensor::new(vec![1, 2], vec![1.0, 0.0]).unwrap();
        assert!(rms_norm_gated_last_dim(&input, &bad_weight, &gate, 1.0e-6).is_err());
    }

    #[test]
    fn linear_attention_block_matches_explicit_llama_order() {
        let mut block = Qwen35LinearAttentionBlock::zeros(3, 1, 2, 2).unwrap();
        assert_eq!(block.conv_channels(), 8);
        let conv_weights = [
            0.0, 0.0, 1.0, // q head, component 0
            0.0, 0.0, 1.0, // q head, component 1
            0.0, 0.0, 1.0, // k head, component 0
            0.0, 0.0, 1.0, // k head, component 1
            0.0, 0.0, 1.0, // value head 0, component 0
            0.0, 0.0, 1.0, // value head 0, component 1
            0.0, 0.0, 1.0, // value head 1, component 0
            0.0, 0.0, 1.0, // value head 1, component 1
        ];
        let qkv = [1.0, -2.0, 3.0, 4.0, -1.0, 2.0, 0.5, -3.0];
        let z = [1.0, 0.0, -1.0, 0.5];
        let decay = [0.0, 0.0];
        let beta = [0.5, 0.75];
        let norm_weight = [1.0, 1.25];
        let epsilon = 1.0e-6;
        let checkpoint = block.checkpoint();
        let actual = block
            .step(Qwen35LinearAttentionInputs {
                qkv_mixed: &qkv,
                conv_weights: &conv_weights,
                z: &z,
                decay_gate: &decay,
                beta: &beta,
                norm_weight: &norm_weight,
                epsilon,
            })
            .unwrap();

        // Rebuild the same sequence with the individual public primitives.
        let mut conv = CausalConv1dState::zeros(3, 8).unwrap();
        let raw = conv.step(&qkv, &conv_weights).unwrap();
        let activated = raw
            .iter()
            .map(|&value| value / (1.0 + (-value).exp()))
            .collect::<Vec<_>>();
        let q = l2_norm_last_dim(
            &ReferenceTensor::new(vec![1, 2], activated[0..2].to_vec()).unwrap(),
            epsilon,
        )
        .unwrap();
        let k = l2_norm_last_dim(
            &ReferenceTensor::new(vec![1, 2], activated[2..4].to_vec()).unwrap(),
            epsilon,
        )
        .unwrap();
        let mut gated_delta = GatedDeltaNetState::zeros(2, 2).unwrap();
        let recurrent = gated_delta
            .step(q.values(), k.values(), &activated[4..], &decay, &beta)
            .unwrap();
        let expected = rms_norm_gated_last_dim(
            &ReferenceTensor::new(vec![2, 2], recurrent).unwrap(),
            &ReferenceTensor::new(vec![2], norm_weight.to_vec()).unwrap(),
            &ReferenceTensor::new(vec![2, 2], z.to_vec()).unwrap(),
            epsilon,
        )
        .unwrap()
        .into_values();
        assert_eq!(actual, expected);
        assert_ne!(block.checkpoint(), checkpoint);
        block.restore(&checkpoint).unwrap();
        assert_eq!(block.checkpoint(), checkpoint);

        let output_weight = ReferenceTensor::new(
            vec![3, 4],
            vec![
                1.0, 0.0, 0.0, 0.0, // hidden row 0
                0.0, 1.0, 0.0, 0.0, // hidden row 1
                0.0, 0.0, 1.0, 1.0, // hidden row 2
            ],
        )
        .unwrap();
        let mut projected_block = Qwen35LinearAttentionBlock::zeros(3, 1, 2, 2).unwrap();
        let projected = projected_block
            .step_projected(
                Qwen35LinearAttentionInputs {
                    qkv_mixed: &qkv,
                    conv_weights: &conv_weights,
                    z: &z,
                    decay_gate: &decay,
                    beta: &beta,
                    norm_weight: &norm_weight,
                    epsilon,
                },
                &output_weight,
            )
            .unwrap();
        let expected_projected = matmul_row_major(
            &ReferenceTensor::new(vec![1, 4], expected).unwrap(),
            &output_weight,
        )
        .unwrap();
        assert_eq!(projected, expected_projected);
    }

    #[test]
    fn linear_attention_block_rolls_back_on_late_failure() {
        let mut block = Qwen35LinearAttentionBlock::zeros(3, 1, 2, 2).unwrap();
        let conv_weights = [
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
        ];
        let qkv = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let z = [0.0; 4];
        let before = block.checkpoint();

        // The convolution succeeds, but the wrong norm width is detected
        // before the recurrent state can be committed.
        assert!(block
            .step(Qwen35LinearAttentionInputs {
                qkv_mixed: &qkv,
                conv_weights: &conv_weights,
                z: &z,
                decay_gate: &[0.0],
                beta: &[1.0],
                norm_weight: &[1.0],
                epsilon: 1.0e-6,
            })
            .is_err());
        assert_eq!(block.checkpoint(), before);

        // An overflowing Q/K norm is a late failure after convolution; both
        // recurrent states must still be restored to the original checkpoint.
        let overflowing = [f32::MAX; 8];
        assert!(block
            .step(Qwen35LinearAttentionInputs {
                qkv_mixed: &overflowing,
                conv_weights: &conv_weights,
                z: &z,
                decay_gate: &[0.0],
                beta: &[1.0],
                norm_weight: &[1.0, 1.0],
                epsilon: 1.0e-6,
            })
            .is_err());
        assert_eq!(block.checkpoint(), before);
    }

    #[test]
    fn linear_attention_block_process_matches_repeated_steps() {
        let conv_weights = [0.0f32, 0.0, 1.0].repeat(8);
        let qkv_a = [1.0, -2.0, 3.0, 4.0, -1.0, 2.0, 0.5, -3.0];
        let qkv_b = [0.25, 1.5, -2.0, 0.75, 3.0, -0.5, 2.5, 1.0];
        let z_a = [1.0, 0.0, -1.0, 0.5];
        let z_b = [0.0, 0.5, 1.0, -0.5];
        let decay = [0.0, 0.0];
        let beta = [0.5, 0.75];
        let norm_weight = [1.0, 1.25];
        let inputs = [
            Qwen35LinearAttentionInputs {
                qkv_mixed: &qkv_a,
                conv_weights: &conv_weights,
                z: &z_a,
                decay_gate: &decay,
                beta: &beta,
                norm_weight: &norm_weight,
                epsilon: 1.0e-6,
            },
            Qwen35LinearAttentionInputs {
                qkv_mixed: &qkv_b,
                conv_weights: &conv_weights,
                z: &z_b,
                decay_gate: &decay,
                beta: &beta,
                norm_weight: &norm_weight,
                epsilon: 1.0e-6,
            },
        ];

        let mut batched = Qwen35LinearAttentionBlock::zeros(3, 1, 2, 2).unwrap();
        let batched_output = batched.process(&inputs).unwrap();
        let mut stepped = Qwen35LinearAttentionBlock::zeros(3, 1, 2, 2).unwrap();
        let mut stepped_output = Vec::new();
        for input in inputs {
            stepped_output.extend(stepped.step(input).unwrap());
        }
        assert_eq!(batched_output, stepped_output);
        assert_eq!(batched.checkpoint(), stepped.checkpoint());
    }

    #[test]
    fn linear_attention_block_process_is_atomic_across_tokens() {
        let conv_weights = [0.0f32, 0.0, 1.0].repeat(8);
        let qkv = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let z = [0.0; 4];
        let valid = Qwen35LinearAttentionInputs {
            qkv_mixed: &qkv,
            conv_weights: &conv_weights,
            z: &z,
            decay_gate: &[0.0],
            beta: &[1.0],
            norm_weight: &[1.0, 1.0],
            epsilon: 1.0e-6,
        };
        let invalid = Qwen35LinearAttentionInputs {
            norm_weight: &[1.0],
            ..valid
        };
        let mut block = Qwen35LinearAttentionBlock::zeros(3, 1, 2, 2).unwrap();
        let before = block.checkpoint();
        assert!(block.process(&[valid, invalid]).is_err());
        assert_eq!(block.checkpoint(), before);
        assert!(block.process(&[]).unwrap().is_empty());
        assert_eq!(block.checkpoint(), before);
    }

    #[test]
    fn linear_attention_block_rejects_invalid_geometry_and_checkpoint() {
        assert!(Qwen35LinearAttentionBlock::zeros(3, 0, 2, 2).is_err());
        assert!(Qwen35LinearAttentionBlock::zeros(3, 2, 3, 2).is_err());
        assert!(Qwen35LinearAttentionBlock::zeros(3, 1, 2, 0).is_err());

        let mut block = Qwen35LinearAttentionBlock::zeros(3, 1, 2, 2).unwrap();
        let before = block.checkpoint();
        let wrong_length = Qwen35LinearAttentionCheckpoint {
            conv_history: vec![0.0],
            gated_delta_values: before.gated_delta_values.clone(),
        };
        assert!(block.restore(&wrong_length).is_err());
        assert_eq!(block.checkpoint(), before);

        let mut nonfinite = before.clone();
        nonfinite.gated_delta_values[0] = f32::INFINITY;
        assert!(block.restore(&nonfinite).is_err());
        assert_eq!(block.checkpoint(), before);

        let bad_weight = ReferenceTensor::new(vec![3, 3], vec![0.0; 9]).unwrap();
        let inputs = Qwen35LinearAttentionInputs {
            qkv_mixed: &[1.0; 8],
            conv_weights: &[0.0; 24],
            z: &[0.0; 4],
            decay_gate: &[0.0],
            beta: &[1.0],
            norm_weight: &[1.0, 1.0],
            epsilon: 1.0e-6,
        };
        assert!(block.step_projected(inputs, &bad_weight).is_err());
        assert_eq!(block.checkpoint(), before);
    }

    #[test]
    fn softmax_handles_large_values_and_positive_infinity() {
        let input = ReferenceTensor::new(vec![1, 3], vec![1000.0, 1001.0, 999.0]).unwrap();
        let output = softmax_last_dim(&input).unwrap();
        let sum = output.values().iter().sum::<f32>();
        assert!((sum - 1.0).abs() < 1.0e-6);

        let input =
            ReferenceTensor::new(vec![1, 3], vec![f32::INFINITY, 1.0, f32::INFINITY]).unwrap();
        let output = softmax_last_dim(&input).unwrap();
        assert_eq!(output.values(), &[0.5, 0.0, 0.5]);
    }

    #[test]
    fn invalid_shapes_and_all_masked_softmax_fail_closed() {
        assert!(ReferenceTensor::new(vec![2, 0], Vec::<f32>::new()).is_err());
        let input =
            ReferenceTensor::new(vec![1, 2], vec![f32::NEG_INFINITY, f32::NEG_INFINITY]).unwrap();
        assert!(softmax_last_dim(&input).is_err());
        let input = ReferenceTensor::new(vec![1, 2], vec![1.0, 2.0]).unwrap();
        let weight = ReferenceTensor::new(vec![3], vec![1.0, 1.0, 1.0]).unwrap();
        assert!(rms_norm_last_dim(&input, &weight, 1.0e-6).is_err());
    }

    #[test]
    fn bf16_round_is_nearest_even_for_finite_values() {
        assert_eq!(bf16_round(1.0), 1.0);
        assert_eq!(bf16_round(f32::INFINITY), f32::INFINITY);
        assert_eq!(bf16_round(f32::NAN).to_bits(), f32::NAN.to_bits());
    }

    #[test]
    fn gated_delta_net_step_matches_transposed_state_formula() {
        let mut state = GatedDeltaNetState::zeros(1, 2).unwrap();
        let output = state
            .step(&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0], &[0.0], &[0.5])
            .unwrap();
        assert_eq!(state.values(), &[7.5, 10.0, 9.0, 12.0]);
        assert!((output[0] - 19.445436).abs() < 1.0e-5);
        assert!((output[1] - 23.334524).abs() < 1.0e-5);
    }

    #[test]
    fn gated_delta_net_checkpoint_restores_exactly() {
        let mut state = GatedDeltaNetState::zeros(2, 2).unwrap();
        state
            .step(
                &[1.0, 0.0, 0.0, 1.0],
                &[0.0, 1.0, 1.0, 0.0],
                &[1.0, 2.0, 3.0, 4.0],
                &[0.0, 0.0],
                &[0.5, 0.25],
            )
            .unwrap();
        let checkpoint = state.checkpoint();
        state
            .step(
                &[0.5, 0.5, 0.5, 0.5],
                &[1.0, 1.0, 1.0, 1.0],
                &[2.0, 2.0, 2.0, 2.0],
                &[0.0],
                &[0.5],
            )
            .unwrap();
        state.restore(&checkpoint).unwrap();
        assert_eq!(state.values(), checkpoint.as_slice());
        assert!(state.restore(&checkpoint[..3]).is_err());
    }

    #[test]
    fn gated_delta_net_repeats_key_heads_for_qwen_grouped_values() {
        let mut state = GatedDeltaNetState::zeros(4, 2).unwrap();
        let q = [1.0, 0.0, 0.0, 1.0];
        let k = [1.0, 0.0, 0.0, 1.0];
        let v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let output = state.step(&q, &k, &v, &[0.0; 8], &[1.0; 4]).unwrap();

        // Value heads 0/2 use key head 0; 1/3 use key head 1.
        assert_eq!(state.values()[0..4], [1.0, 0.0, 2.0, 0.0]);
        assert_eq!(state.values()[4..8], [0.0, 3.0, 0.0, 4.0]);
        assert_eq!(state.values()[8..12], [5.0, 0.0, 6.0, 0.0]);
        assert_eq!(state.values()[12..16], [0.0, 7.0, 0.0, 8.0]);
        assert_eq!(output.len(), 8);
    }

    #[test]
    fn gated_delta_net_rejects_bad_shape_without_mutating_state() {
        let mut state = GatedDeltaNetState::zeros(2, 2).unwrap();
        let before = state.checkpoint();
        assert!(state
            .step(
                &[1.0, 2.0, 3.0],
                &[1.0, 2.0, 3.0],
                &[1.0; 4],
                &[0.0],
                &[1.0]
            )
            .is_err());
        assert_eq!(state.values(), before.as_slice());

        // A finite input can still overflow the update; that failure is also
        // atomic and must not expose a partially updated head.
        assert!(state
            .step(
                &[1.0, 0.0],
                &[f32::MAX, 0.0],
                &[f32::MAX, f32::MAX, f32::MAX, f32::MAX],
                &[0.0],
                &[f32::MAX],
            )
            .is_err());
        assert_eq!(state.values(), before.as_slice());
    }

    #[test]
    fn causal_conv_matches_oldest_to_newest_window_formula() {
        let mut state = CausalConv1dState::zeros(3, 2).unwrap();
        let weights = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0];

        assert_eq!(state.step(&[1.0, 2.0], &weights).unwrap(), [3.0, 60.0]);
        assert_eq!(state.history(), &[0.0, 1.0, 0.0, 2.0]);
        assert_eq!(state.step(&[3.0, 4.0], &weights).unwrap(), [11.0, 160.0]);
        assert_eq!(state.history(), &[1.0, 3.0, 2.0, 4.0]);
        assert_eq!(state.step(&[5.0, 6.0], &weights).unwrap(), [22.0, 280.0]);
    }

    #[test]
    fn causal_conv_process_matches_repeated_steps_and_checkpoint() {
        let weights = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let inputs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut block = CausalConv1dState::zeros(3, 2).unwrap();
        let block_output = block.process(&inputs, &weights).unwrap();

        let mut stepped = CausalConv1dState::zeros(3, 2).unwrap();
        let mut stepped_output = Vec::new();
        for token in inputs.chunks_exact(2) {
            stepped_output.extend(stepped.step(token, &weights).unwrap());
        }
        assert_eq!(block_output, stepped_output);
        assert_eq!(block.history(), stepped.history());

        let checkpoint = block.checkpoint();
        block.step(&[7.0, 8.0], &weights).unwrap();
        block.restore(&checkpoint).unwrap();
        assert_eq!(block.history(), checkpoint.as_slice());
    }

    #[test]
    fn causal_conv_rejects_invalid_or_overflowing_input_atomically() {
        let mut state = CausalConv1dState::zeros(3, 2).unwrap();
        let before = state.checkpoint();
        let weights = [0.0, 0.0, f32::MAX, 0.0, 0.0, 0.0];

        assert!(state.step(&[1.0], &weights).is_err());
        assert_eq!(state.history(), before.as_slice());
        assert!(state.step(&[f32::MAX, 0.0], &weights).is_err());
        assert_eq!(state.history(), before.as_slice());

        // The first token is valid, but the second overflows; process() must
        // leave the history at its pre-call checkpoint.
        let process_weights = [0.0, 0.0, f32::MAX, 0.0, 0.0, 0.0];
        assert!(state
            .process(&[1.0, 0.0, f32::MAX, 0.0], &process_weights)
            .is_err());
        assert_eq!(state.history(), before.as_slice());
        assert!(CausalConv1dState::zeros(0, 2).is_err());
        assert!(CausalConv1dState::zeros(2, 0).is_err());
    }
}
