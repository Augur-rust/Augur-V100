//! Reusable, descriptor-bound GGUF input and output boundaries.
//!
//! The providers in this module deliberately stop at the model input/output
//! edges.  They do not own a tokenizer, a request ledger, CUDA resources, or a
//! sampler.  A caller supplies one identity-bound [`GgufTensorReader`] so the
//! embedding row and streamed output matrix use the same artifact identity.

use crate::config::Qwen35Config;
use crate::gguf::{GgufManifest, GgufTensor, GgufTensorReader};
use crate::layer::Qwen35Projection;
use crate::quant::{
    dequantize_tensor, quantized_matvec_activation_bf16, quantized_matvec_activation_q8_0,
    quantized_matvec_fused_simd, Qwen35QuantType, GGML_TYPE_BF16, GGML_TYPE_F32, GGML_TYPE_NVFP4,
    GGML_TYPE_Q8_0,
};
use crate::reference::{rms_norm_last_dim, ReferenceTensor};
use crate::Qwen35Error;
use std::path::Path;

/// Default number of output rows held in one bounded logits chunk.
pub const DEFAULT_OUTPUT_HEAD_CHUNK_ROWS: usize = 256;
/// Hard upper bound for a streamed output chunk.
pub const MAX_OUTPUT_HEAD_CHUNK_ROWS: usize = 16_384;

const TOKEN_EMBEDDING: &str = "token_embd.weight";
const OUTPUT_NORM: &str = "output_norm.weight";
const OUTPUT_WEIGHT: &str = "output.weight";

/// One authenticated embedding row and its original GGUF bytes.
#[derive(Clone, Debug, PartialEq)]
pub struct GgufEmbeddingRow {
    pub token_id: usize,
    pub values: Vec<f32>,
    pub raw_bytes: Vec<u8>,
}

/// Numerical boundary used by an owned GGUF projection.
///
/// The native mode preserves crystal-quant's existing f32-activation ABI.
/// The GGML modes are opt-in. `GgmlQ8_0Activation` mirrors temporary
/// activation-Q8_0 conversion for every Q8_0 and NVFP4 projection, while
/// `GgmlFfnQ8_0Activation` lets a layer loader select that boundary only for
/// its FFN projections. `GgmlCpuActivation` also mirrors the BF16 activation
/// and AVX2 reduction boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GgufProjectionNumerics {
    NativeF32Activation,
    GgmlQ8_0Activation,
    GgmlFfnQ8_0Activation,
    GgmlOutputFfnQ8_0Activation,
    GgmlCpuActivation,
}

/// An owned quantized projection loaded from one identity-bound GGUF reader.
///
/// `GgufProjection` is the zero-copy form used by short-lived probes. This
/// owned form keeps the descriptor and payload together so a request executor
/// can retain a contiguous layer without self-referential lifetimes.
pub struct OwnedGgufProjection {
    descriptor: GgufTensor,
    payload: Vec<u8>,
    input_multiplier: f32,
    output_multiplier: f32,
    input_width: usize,
    output_width: usize,
    numerics: GgufProjectionNumerics,
}

impl OwnedGgufProjection {
    /// Read and bind one rank-2 GGUF projection from an existing reader.
    pub fn from_reader(
        reader: &mut GgufTensorReader<'_>,
        tensor_name: &str,
        input_multiplier: f32,
        output_multiplier: f32,
    ) -> Result<Self, Qwen35Error> {
        Self::from_reader_with_numerics(
            reader,
            tensor_name,
            input_multiplier,
            output_multiplier,
            GgufProjectionNumerics::NativeF32Activation,
        )
    }

    /// Read and bind one projection with an explicit numerical boundary.
    pub fn from_reader_with_numerics(
        reader: &mut GgufTensorReader<'_>,
        tensor_name: &str,
        input_multiplier: f32,
        output_multiplier: f32,
        numerics: GgufProjectionNumerics,
    ) -> Result<Self, Qwen35Error> {
        let descriptor = reader
            .tensor(tensor_name)
            .ok_or_else(|| Qwen35Error::Gguf(format!("missing projection {tensor_name:?}")))?
            .clone();
        let payload = reader.read_tensor_bytes(tensor_name)?;
        Self::from_parts_with_numerics(
            descriptor,
            payload,
            input_multiplier,
            output_multiplier,
            numerics,
        )
    }

    /// Bind an already-read payload after validating its rank, type, and
    /// exact GGML byte geometry.
    pub fn from_parts(
        descriptor: GgufTensor,
        payload: Vec<u8>,
        input_multiplier: f32,
        output_multiplier: f32,
    ) -> Result<Self, Qwen35Error> {
        Self::from_parts_with_numerics(
            descriptor,
            payload,
            input_multiplier,
            output_multiplier,
            GgufProjectionNumerics::NativeF32Activation,
        )
    }

    /// Bind an already-read payload with an explicit numerical boundary.
    pub fn from_parts_with_numerics(
        descriptor: GgufTensor,
        payload: Vec<u8>,
        input_multiplier: f32,
        output_multiplier: f32,
        numerics: GgufProjectionNumerics,
    ) -> Result<Self, Qwen35Error> {
        if descriptor.dimensions.len() != 2 {
            return Err(Qwen35Error::invalid(
                "owned_projection.dimensions",
                format!("expected rank-2 [K, N], got {:?}", descriptor.dimensions),
            ));
        }
        let input_width =
            usize::try_from(descriptor.dimensions[0]).map_err(|_| Qwen35Error::SizeOverflow)?;
        let output_width =
            usize::try_from(descriptor.dimensions[1]).map_err(|_| Qwen35Error::SizeOverflow)?;
        if input_width == 0 || output_width == 0 {
            return Err(Qwen35Error::invalid(
                "owned_projection.dimensions",
                "projection widths must be nonzero",
            ));
        }
        let quant_type = Qwen35QuantType::from_ggml_type(descriptor.ggml_type)?;
        let row_bytes = row_payload_bytes(input_width, quant_type)?;
        let expected_bytes = row_bytes
            .checked_mul(output_width)
            .ok_or(Qwen35Error::SizeOverflow)?;
        if payload.len() != expected_bytes
            || descriptor.byte_len
                != u64::try_from(expected_bytes).map_err(|_| Qwen35Error::SizeOverflow)?
        {
            return Err(Qwen35Error::invalid(
                "owned_projection.payload",
                format!("expected {expected_bytes} bytes, got {}", payload.len()),
            ));
        }
        validate_multiplier(input_multiplier, "owned_projection.input_multiplier")?;
        validate_multiplier(output_multiplier, "owned_projection.output_multiplier")?;
        Ok(Self {
            descriptor,
            payload,
            input_multiplier,
            output_multiplier,
            input_width,
            output_width,
            numerics,
        })
    }

    pub fn descriptor(&self) -> &GgufTensor {
        &self.descriptor
    }

    pub fn payload(&self) -> &[u8] {
        &self.payload
    }

    pub fn input_width(&self) -> usize {
        self.input_width
    }

    pub fn output_width(&self) -> usize {
        self.output_width
    }

    pub fn numerics(&self) -> GgufProjectionNumerics {
        self.numerics
    }
}

impl Qwen35Projection for OwnedGgufProjection {
    fn input_width(&self) -> usize {
        self.input_width
    }

    fn output_width(&self) -> usize {
        self.output_width
    }

    fn matvec(&self, input: &[f32]) -> Result<Vec<f32>, Qwen35Error> {
        if input.len() != self.input_width || input.iter().any(|value| !value.is_finite()) {
            return Err(Qwen35Error::invalid(
                "owned_projection.input",
                format!("expected {} finite values", self.input_width),
            ));
        }
        let scaled_input;
        let effective_input = if self.input_multiplier == 1.0 {
            input
        } else {
            scaled_input = input
                .iter()
                .map(|value| *value * self.input_multiplier)
                .collect::<Vec<_>>();
            if scaled_input.iter().any(|value| !value.is_finite()) {
                return Err(Qwen35Error::invalid(
                    "owned_projection.input_multiplier",
                    "scaled input is not finite",
                ));
            }
            &scaled_input
        };
        let mut output = match (self.numerics, self.descriptor.ggml_type) {
            (
                GgufProjectionNumerics::GgmlQ8_0Activation
                | GgufProjectionNumerics::GgmlFfnQ8_0Activation
                | GgufProjectionNumerics::GgmlOutputFfnQ8_0Activation
                | GgufProjectionNumerics::GgmlCpuActivation,
                GGML_TYPE_Q8_0 | GGML_TYPE_NVFP4,
            ) => {
                quantized_matvec_activation_q8_0(&self.descriptor, &self.payload, effective_input)?
            }
            (GgufProjectionNumerics::GgmlCpuActivation, GGML_TYPE_BF16) => {
                quantized_matvec_activation_bf16(&self.descriptor, &self.payload, effective_input)?
            }
            _ => quantized_matvec_fused_simd(&self.descriptor, &self.payload, effective_input)?,
        };
        for value in &mut output {
            *value *= self.output_multiplier;
            if !value.is_finite() {
                return Err(Qwen35Error::invalid(
                    "owned_projection.output_multiplier",
                    "scaled output is not finite",
                ));
            }
        }
        Ok(output)
    }
}

/// Metadata for a dense token-embedding matrix in GGUF `[hidden, vocab]`
/// orientation.
#[derive(Clone, Debug, PartialEq)]
pub struct GgufEmbeddingRowProvider {
    descriptor: GgufTensor,
    row_count: usize,
    row_width: usize,
    row_bytes: usize,
}

impl GgufEmbeddingRowProvider {
    /// Bind the standard Qwen35 token embedding descriptor.
    pub fn for_qwen35(manifest: &GgufManifest, config: &Qwen35Config) -> Result<Self, Qwen35Error> {
        config.validate_exact_profile()?;
        Self::from_manifest(
            manifest,
            TOKEN_EMBEDDING,
            config.vocab_size,
            config.hidden_size,
        )
    }

    /// Bind an embedding descriptor with explicit `[width, rows]` geometry.
    ///
    /// The first implementation accepts only dense F32/BF16 rows.  A
    /// quantized embedding would need an explicit row dequantization policy;
    /// silently treating a block payload as dense values would be unsafe.
    pub fn from_manifest(
        manifest: &GgufManifest,
        tensor_name: &str,
        row_count: usize,
        row_width: usize,
    ) -> Result<Self, Qwen35Error> {
        if row_count == 0 || row_width == 0 {
            return Err(Qwen35Error::invalid(
                "embedding.geometry",
                "row count and row width must be nonzero",
            ));
        }
        let descriptor = manifest
            .tensor(tensor_name)
            .ok_or_else(|| Qwen35Error::Gguf(format!("missing embedding tensor {tensor_name:?}")))?
            .clone();
        validate_embedding_descriptor(&descriptor, row_count, row_width)?;
        let row_bytes = dense_row_bytes(&descriptor, row_width)?;
        Ok(Self {
            descriptor,
            row_count,
            row_width,
            row_bytes,
        })
    }

    pub fn tensor(&self) -> &GgufTensor {
        &self.descriptor
    }

    pub fn row_count(&self) -> usize {
        self.row_count
    }

    pub fn row_width(&self) -> usize {
        self.row_width
    }

    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    /// Read and decode one row while retaining the exact source bytes.
    pub fn read_row_with_bytes(
        &self,
        reader: &mut GgufTensorReader<'_>,
        token_id: usize,
    ) -> Result<GgufEmbeddingRow, Qwen35Error> {
        ensure_reader_tensor(reader, &self.descriptor)?;
        if token_id >= self.row_count {
            return Err(Qwen35Error::invalid(
                "embedding.token_id",
                format!("{token_id} is outside row count {}", self.row_count),
            ));
        }
        let byte_offset = token_id
            .checked_mul(self.row_bytes)
            .ok_or(Qwen35Error::SizeOverflow)?;
        let raw_bytes = reader.read_tensor_slice(
            &self.descriptor.name,
            u64::try_from(byte_offset).map_err(|_| Qwen35Error::SizeOverflow)?,
            u64::try_from(self.row_bytes).map_err(|_| Qwen35Error::SizeOverflow)?,
        )?;
        let mut values = vec![0.0_f32; self.row_width];
        match self.descriptor.ggml_type {
            GGML_TYPE_BF16 | GGML_TYPE_F32 => {
                let decoded = dequantize_tensor(&self.descriptor_for_row(), &raw_bytes)?;
                values.copy_from_slice(&decoded);
            }
            _ => {
                return Err(Qwen35Error::invalid(
                    "embedding.ggml_type",
                    "dense embedding provider received an unsupported type",
                ));
            }
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(Qwen35Error::Gguf(format!(
                "embedding row {token_id} contains a non-finite value"
            )));
        }
        Ok(GgufEmbeddingRow {
            token_id,
            values,
            raw_bytes,
        })
    }

    /// Read and decode one row without retaining a second copy for callers
    /// that only need the F32 input vector.
    pub fn read_row(
        &self,
        reader: &mut GgufTensorReader<'_>,
        token_id: usize,
    ) -> Result<Vec<f32>, Qwen35Error> {
        Ok(self.read_row_with_bytes(reader, token_id)?.values)
    }

    fn descriptor_for_row(&self) -> GgufTensor {
        let mut descriptor = self.descriptor.clone();
        descriptor.dimensions = vec![self.row_width as u64];
        descriptor.byte_len = self.row_bytes as u64;
        descriptor
    }
}

/// A final RMSNorm plus a bounded, row-streamed `output.weight` head.
///
/// The head stores only the final norm and one tensor descriptor.  Weight
/// chunks are read through the caller's identity-bound reader for each call,
/// so a full 248K-row output matrix is never materialized in memory.
#[derive(Clone, Debug)]
pub struct GgufStreamedOutputHead {
    output_norm_descriptor: GgufTensor,
    output_norm_payload: Vec<u8>,
    output_norm: Vec<f32>,
    output_weight: GgufTensor,
    hidden_size: usize,
    vocab_size: usize,
    row_bytes: usize,
    chunk_rows: usize,
    rms_norm_eps: f32,
    numerics: GgufProjectionNumerics,
}

impl GgufStreamedOutputHead {
    /// Bind the standard Qwen35 final norm and output matrix.
    pub fn for_qwen35(
        reader: &mut GgufTensorReader<'_>,
        config: &Qwen35Config,
        chunk_rows: usize,
    ) -> Result<Self, Qwen35Error> {
        Self::for_qwen35_with_numerics(
            reader,
            config,
            chunk_rows,
            GgufProjectionNumerics::NativeF32Activation,
        )
    }

    /// Bind the standard Qwen35 final norm and output matrix with an explicit
    /// projection ABI.
    pub fn for_qwen35_with_numerics(
        reader: &mut GgufTensorReader<'_>,
        config: &Qwen35Config,
        chunk_rows: usize,
        numerics: GgufProjectionNumerics,
    ) -> Result<Self, Qwen35Error> {
        config.validate_exact_profile()?;
        Self::from_reader_with_numerics(
            reader,
            OUTPUT_NORM,
            OUTPUT_WEIGHT,
            config.hidden_size,
            config.vocab_size,
            config.rms_norm_eps as f32,
            chunk_rows,
            numerics,
        )
    }

    /// Bind a final norm and output matrix with explicit geometry.
    pub fn from_reader(
        reader: &mut GgufTensorReader<'_>,
        output_norm_name: &str,
        output_weight_name: &str,
        hidden_size: usize,
        vocab_size: usize,
        rms_norm_eps: f32,
        chunk_rows: usize,
    ) -> Result<Self, Qwen35Error> {
        Self::from_reader_with_numerics(
            reader,
            output_norm_name,
            output_weight_name,
            hidden_size,
            vocab_size,
            rms_norm_eps,
            chunk_rows,
            GgufProjectionNumerics::NativeF32Activation,
        )
    }

    /// Bind a final norm and output matrix with an explicit projection ABI.
    #[allow(clippy::too_many_arguments)]
    pub fn from_reader_with_numerics(
        reader: &mut GgufTensorReader<'_>,
        output_norm_name: &str,
        output_weight_name: &str,
        hidden_size: usize,
        vocab_size: usize,
        rms_norm_eps: f32,
        chunk_rows: usize,
        numerics: GgufProjectionNumerics,
    ) -> Result<Self, Qwen35Error> {
        if hidden_size == 0 || vocab_size == 0 {
            return Err(Qwen35Error::invalid(
                "output_head.geometry",
                "hidden size and vocabulary size must be nonzero",
            ));
        }
        if chunk_rows == 0 || chunk_rows > MAX_OUTPUT_HEAD_CHUNK_ROWS {
            return Err(Qwen35Error::invalid(
                "output_head.chunk_rows",
                format!("must be in 1..={MAX_OUTPUT_HEAD_CHUNK_ROWS}, got {chunk_rows}"),
            ));
        }
        if !rms_norm_eps.is_finite() || rms_norm_eps <= 0.0 {
            return Err(Qwen35Error::invalid(
                "output_head.rms_norm_eps",
                "must be finite and positive",
            ));
        }
        let norm = reader
            .tensor(output_norm_name)
            .ok_or_else(|| Qwen35Error::Gguf(format!("missing output norm {output_norm_name:?}")))?
            .clone();
        let output_weight = reader
            .tensor(output_weight_name)
            .ok_or_else(|| {
                Qwen35Error::Gguf(format!("missing output weight {output_weight_name:?}"))
            })?
            .clone();
        if norm.dimensions.as_slice() != [hidden_size as u64]
            || (norm.ggml_type != GGML_TYPE_F32 && norm.ggml_type != GGML_TYPE_BF16)
        {
            return Err(Qwen35Error::invalid(
                "output_head.norm",
                format!(
                    "expected dense [{hidden_size}] F32/BF16 norm, got {:?} type {}",
                    norm.dimensions, norm.ggml_type
                ),
            ));
        }
        if output_weight.dimensions.as_slice() != [hidden_size as u64, vocab_size as u64] {
            return Err(Qwen35Error::invalid(
                "output_head.weight",
                format!(
                    "expected [{hidden_size}, {vocab_size}], got {:?}",
                    output_weight.dimensions
                ),
            ));
        }
        let quant_type = Qwen35QuantType::from_ggml_type(output_weight.ggml_type)?;
        let row_bytes = row_payload_bytes(hidden_size, quant_type)?;
        let expected_payload = row_bytes
            .checked_mul(vocab_size)
            .ok_or(Qwen35Error::SizeOverflow)?;
        if output_weight.byte_len
            != u64::try_from(expected_payload).map_err(|_| Qwen35Error::SizeOverflow)?
        {
            return Err(Qwen35Error::Gguf(
                "output weight payload geometry is inconsistent".into(),
            ));
        }
        ensure_reader_tensor(reader, &norm)?;
        ensure_reader_tensor(reader, &output_weight)?;
        let norm_payload = reader.read_tensor_bytes(&norm.name)?;
        let output_norm = dequantize_tensor(&norm, &norm_payload)?;
        if output_norm.len() != hidden_size || output_norm.iter().any(|v| !v.is_finite()) {
            return Err(Qwen35Error::Gguf(
                "output norm decoded values are invalid".into(),
            ));
        }
        Ok(Self {
            output_norm_descriptor: norm,
            output_norm_payload: norm_payload,
            output_norm,
            output_weight,
            hidden_size,
            vocab_size,
            row_bytes,
            chunk_rows,
            rms_norm_eps,
            numerics,
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    pub fn chunk_rows(&self) -> usize {
        self.chunk_rows
    }

    pub fn chunk_count(&self) -> usize {
        self.vocab_size.div_ceil(self.chunk_rows)
    }

    pub fn numerics(&self) -> GgufProjectionNumerics {
        self.numerics
    }

    pub fn output_norm(&self) -> &[f32] {
        &self.output_norm
    }

    pub fn output_norm_tensor(&self) -> &GgufTensor {
        &self.output_norm_descriptor
    }

    pub fn output_norm_payload(&self) -> &[u8] {
        &self.output_norm_payload
    }

    pub fn output_weight(&self) -> &GgufTensor {
        &self.output_weight
    }

    /// Apply the final RMSNorm to a transformer hidden state.
    pub fn normalized_hidden(&self, hidden: &[f32]) -> Result<Vec<f32>, Qwen35Error> {
        if hidden.len() != self.hidden_size {
            return Err(Qwen35Error::invalid(
                "output_head.hidden",
                format!("expected {}, got {}", self.hidden_size, hidden.len()),
            ));
        }
        if hidden.iter().any(|value| !value.is_finite()) {
            return Err(Qwen35Error::invalid(
                "output_head.hidden",
                "values must be finite",
            ));
        }
        let input = ReferenceTensor::new(vec![self.hidden_size], hidden.to_vec())?;
        let norm = ReferenceTensor::new(vec![self.hidden_size], self.output_norm.clone())?;
        Ok(rms_norm_last_dim(&input, &norm, self.rms_norm_eps)?.into_values())
    }

    /// Return the descriptor for a bounded output-row chunk.
    pub fn chunk_descriptor(
        &self,
        row_start: usize,
        row_count: usize,
    ) -> Result<GgufTensor, Qwen35Error> {
        if row_count == 0 || row_count > self.chunk_rows {
            return Err(Qwen35Error::invalid(
                "output_head.chunk",
                "row count must be nonzero and no larger than configured chunk_rows",
            ));
        }
        let row_end = row_start
            .checked_add(row_count)
            .ok_or(Qwen35Error::SizeOverflow)?;
        if row_end > self.vocab_size {
            return Err(Qwen35Error::invalid(
                "output_head.chunk",
                format!("range [{row_start}, {row_end}) exceeds vocabulary"),
            ));
        }
        let byte_offset = row_start
            .checked_mul(self.row_bytes)
            .ok_or(Qwen35Error::SizeOverflow)?;
        let byte_len = row_count
            .checked_mul(self.row_bytes)
            .ok_or(Qwen35Error::SizeOverflow)?;
        let mut descriptor = self.output_weight.clone();
        descriptor.dimensions[1] =
            u64::try_from(row_count).map_err(|_| Qwen35Error::SizeOverflow)?;
        descriptor.relative_offset = descriptor
            .relative_offset
            .checked_add(u64::try_from(byte_offset).map_err(|_| Qwen35Error::SizeOverflow)?)
            .ok_or(Qwen35Error::SizeOverflow)?;
        descriptor.absolute_offset = descriptor
            .absolute_offset
            .checked_add(u64::try_from(byte_offset).map_err(|_| Qwen35Error::SizeOverflow)?)
            .ok_or(Qwen35Error::SizeOverflow)?;
        descriptor.byte_len = u64::try_from(byte_len).map_err(|_| Qwen35Error::SizeOverflow)?;
        Ok(descriptor)
    }

    /// Read one bounded output-weight chunk through the bound reader.
    pub fn read_weight_chunk(
        &self,
        reader: &mut GgufTensorReader<'_>,
        row_start: usize,
        row_count: usize,
    ) -> Result<Vec<u8>, Qwen35Error> {
        ensure_reader_tensor(reader, &self.output_weight)?;
        let descriptor = self.chunk_descriptor(row_start, row_count)?;
        let byte_offset = row_start
            .checked_mul(self.row_bytes)
            .ok_or(Qwen35Error::SizeOverflow)?;
        reader.read_tensor_slice(
            &self.output_weight.name,
            u64::try_from(byte_offset).map_err(|_| Qwen35Error::SizeOverflow)?,
            descriptor.byte_len,
        )
    }

    /// Apply final norm and stream all output rows into a logits vector.
    pub fn logits(
        &self,
        reader: &mut GgufTensorReader<'_>,
        hidden: &[f32],
    ) -> Result<Vec<f32>, Qwen35Error> {
        let normalized = self.normalized_hidden(hidden)?;
        self.logits_from_normalized_hidden(reader, &normalized)
    }

    /// Project a row that already passed its producer's final norm.
    /// DFlash borrows only the target output matrix, not its final norm.
    pub fn logits_from_normalized_hidden(
        &self,
        reader: &mut GgufTensorReader<'_>,
        normalized: &[f32],
    ) -> Result<Vec<f32>, Qwen35Error> {
        if normalized.len() != self.hidden_size || normalized.iter().any(|value| !value.is_finite())
        {
            return Err(Qwen35Error::Gguf(
                "post-norm output-head input has invalid extent or non-finite values".into(),
            ));
        }
        let mut logits = Vec::new();
        logits
            .try_reserve_exact(self.vocab_size)
            .map_err(|_| Qwen35Error::Gguf("logits allocation failed".into()))?;
        for row_start in (0..self.vocab_size).step_by(self.chunk_rows) {
            let row_count = self.chunk_rows.min(self.vocab_size - row_start);
            let descriptor = self.chunk_descriptor(row_start, row_count)?;
            let payload = self.read_weight_chunk(reader, row_start, row_count)?;
            let chunk = match (self.numerics, descriptor.ggml_type) {
                (
                    GgufProjectionNumerics::GgmlQ8_0Activation
                    | GgufProjectionNumerics::GgmlFfnQ8_0Activation
                    | GgufProjectionNumerics::GgmlOutputFfnQ8_0Activation
                    | GgufProjectionNumerics::GgmlCpuActivation,
                    GGML_TYPE_Q8_0 | GGML_TYPE_NVFP4,
                ) => quantized_matvec_activation_q8_0(&descriptor, &payload, normalized)?,
                (GgufProjectionNumerics::GgmlCpuActivation, GGML_TYPE_BF16) => {
                    quantized_matvec_activation_bf16(&descriptor, &payload, normalized)?
                }
                _ => quantized_matvec_fused_simd(&descriptor, &payload, normalized)?,
            };
            logits.extend_from_slice(&chunk);
        }
        if logits.len() != self.vocab_size || logits.iter().any(|value| !value.is_finite()) {
            return Err(Qwen35Error::Gguf(
                "streamed output head produced invalid logits".into(),
            ));
        }
        Ok(logits)
    }
}

fn row_payload_bytes(
    hidden_size: usize,
    quant_type: Qwen35QuantType,
) -> Result<usize, Qwen35Error> {
    if !hidden_size.is_multiple_of(quant_type.block_elements()) {
        return Err(Qwen35Error::invalid(
            "output_head.weight",
            format!(
                "hidden size {hidden_size} is not divisible by block width {}",
                quant_type.block_elements()
            ),
        ));
    }
    hidden_size
        .checked_div(quant_type.block_elements())
        .and_then(|blocks| blocks.checked_mul(quant_type.block_bytes()))
        .ok_or(Qwen35Error::SizeOverflow)
}

fn validate_multiplier(multiplier: f32, field: &str) -> Result<(), Qwen35Error> {
    if !multiplier.is_finite() || multiplier <= 0.0 {
        return Err(Qwen35Error::invalid(
            field,
            "multiplier must be finite and positive",
        ));
    }
    Ok(())
}

fn validate_embedding_descriptor(
    descriptor: &GgufTensor,
    row_count: usize,
    row_width: usize,
) -> Result<(), Qwen35Error> {
    if descriptor.dimensions.as_slice() != [row_width as u64, row_count as u64] {
        return Err(Qwen35Error::invalid(
            "embedding.geometry",
            format!(
                "expected [{row_width}, {row_count}], got {:?}",
                descriptor.dimensions
            ),
        ));
    }
    if descriptor.ggml_type != GGML_TYPE_BF16 && descriptor.ggml_type != GGML_TYPE_F32 {
        return Err(Qwen35Error::invalid(
            "embedding.ggml_type",
            format!(
                "expected BF16 ({GGML_TYPE_BF16}) or F32 ({GGML_TYPE_F32}), got {}",
                descriptor.ggml_type
            ),
        ));
    }
    let row_bytes = dense_row_bytes(descriptor, row_width)?;
    let expected_payload = row_bytes
        .checked_mul(row_count)
        .ok_or(Qwen35Error::SizeOverflow)?;
    if descriptor.byte_len
        != u64::try_from(expected_payload).map_err(|_| Qwen35Error::SizeOverflow)?
    {
        return Err(Qwen35Error::Gguf(format!(
            "embedding tensor {:?} payload geometry is inconsistent",
            descriptor.name
        )));
    }
    Ok(())
}

fn dense_row_bytes(descriptor: &GgufTensor, row_width: usize) -> Result<usize, Qwen35Error> {
    let element_bytes = match descriptor.ggml_type {
        GGML_TYPE_BF16 => 2,
        GGML_TYPE_F32 => 4,
        _ => {
            return Err(Qwen35Error::invalid(
                "embedding.ggml_type",
                "expected a dense F32 or BF16 tensor",
            ));
        }
    };
    row_width
        .checked_mul(element_bytes)
        .ok_or(Qwen35Error::SizeOverflow)
}

fn ensure_reader_tensor(
    reader: &GgufTensorReader<'_>,
    expected: &GgufTensor,
) -> Result<(), Qwen35Error> {
    match reader.tensor(&expected.name) {
        Some(actual) if actual == expected => Ok(()),
        Some(_) => Err(Qwen35Error::Gguf(format!(
            "reader descriptor for {:?} differs from provider binding",
            expected.name
        ))),
        None => Err(Qwen35Error::Gguf(format!(
            "reader does not contain bound tensor {:?}",
            expected.name
        ))),
    }
}

/// Convenience helper for callers that already have a path and only need a
/// one-shot row.  The manifest remains the authority for all offsets and
/// identity checks.
pub fn read_qwen35_embedding_row(
    manifest: &GgufManifest,
    path: impl AsRef<Path>,
    config: &Qwen35Config,
    token_id: usize,
) -> Result<GgufEmbeddingRow, Qwen35Error> {
    let provider = GgufEmbeddingRowProvider::for_qwen35(manifest, config)?;
    let mut reader = manifest.open_tensor_reader(path)?;
    provider.read_row_with_bytes(&mut reader, token_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn descriptor(name: &str, dimensions: Vec<u64>, ggml_type: u32, byte_len: u64) -> GgufTensor {
        GgufTensor {
            name: name.into(),
            dimensions,
            ggml_type,
            relative_offset: 32,
            absolute_offset: 128,
            byte_len,
        }
    }

    #[test]
    fn embedding_descriptor_requires_dense_geometry() {
        let good = descriptor("emb", vec![4, 8], GGML_TYPE_BF16, 64);
        assert!(validate_embedding_descriptor(&good, 8, 4).is_ok());
        assert_eq!(dense_row_bytes(&good, 4).unwrap(), 8);
        let mut bad = good.clone();
        bad.ggml_type = 8;
        assert!(validate_embedding_descriptor(&bad, 8, 4).is_err());
        assert!(validate_embedding_descriptor(&good, 7, 4).is_err());
    }

    #[test]
    fn output_chunk_descriptor_preserves_matrix_orientation() {
        let weight = descriptor("output", vec![32, 10], 8, 34 * 10);
        let head = GgufStreamedOutputHead {
            output_norm_descriptor: descriptor("norm", vec![32], GGML_TYPE_F32, 32 * 4),
            output_norm_payload: vec![0; 32 * 4],
            output_norm: vec![1.0; 32],
            output_weight: weight,
            hidden_size: 32,
            vocab_size: 10,
            row_bytes: 34,
            chunk_rows: 3,
            rms_norm_eps: 1.0e-6,
            numerics: GgufProjectionNumerics::NativeF32Activation,
        };
        let chunk = head.chunk_descriptor(4, 3).unwrap();
        assert_eq!(chunk.dimensions, vec![32, 3]);
        assert_eq!(chunk.relative_offset, 32 + 4 * 34);
        assert_eq!(chunk.absolute_offset, 128 + 4 * 34);
        assert_eq!(chunk.byte_len, 3 * 34);
        assert!(head.chunk_descriptor(8, 3).is_err());
    }

    #[test]
    fn output_chunk_rows_are_bounded() {
        const {
            assert!(MAX_OUTPUT_HEAD_CHUNK_ROWS >= DEFAULT_OUTPUT_HEAD_CHUNK_ROWS);
        }
        assert!(row_payload_bytes(5120, Qwen35QuantType::Q8_0).is_ok());
        assert!(row_payload_bytes(31, Qwen35QuantType::Q8_0).is_err());
    }

    #[test]
    fn owned_projection_matches_checked_quantized_matvec() {
        let descriptor = descriptor("owned.q8", vec![32, 2], crate::quant::GGML_TYPE_Q8_0, 68);
        let payload = q8_payload(32, 2);
        let input: Vec<f32> = (0..32).map(|value| value as f32 / 8.0).collect();
        let scaled_input: Vec<f32> = input.iter().map(|value| value * 2.0).collect();
        let expected = quantized_matvec_fused_simd(&descriptor, &payload, &scaled_input).unwrap();
        let projection = OwnedGgufProjection::from_parts(descriptor, payload, 2.0, 0.5).unwrap();
        assert_eq!(projection.input_width(), 32);
        assert_eq!(projection.output_width(), 2);
        let actual = projection.matvec(&input).unwrap();
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected * 0.5).abs() <= 1.0e-5);
        }
    }

    #[test]
    fn owned_projection_requires_explicit_activation_q8_numerics() {
        let descriptor = descriptor("owned.q8", vec![32, 1], crate::quant::GGML_TYPE_Q8_0, 34);
        let payload = q8_payload(32, 1);
        let mut input = vec![0.0_f32; 32];
        input[0] = 127.0;
        input[1] = 2.5;
        let expected = quantized_matvec_activation_q8_0(&descriptor, &payload, &input).unwrap();

        let native =
            OwnedGgufProjection::from_parts(descriptor.clone(), payload.clone(), 1.0, 1.0).unwrap();
        let compatible = OwnedGgufProjection::from_parts_with_numerics(
            descriptor,
            payload,
            1.0,
            1.0,
            GgufProjectionNumerics::GgmlQ8_0Activation,
        )
        .unwrap();
        assert_eq!(
            native.numerics(),
            GgufProjectionNumerics::NativeF32Activation
        );
        assert_eq!(
            compatible.numerics(),
            GgufProjectionNumerics::GgmlQ8_0Activation
        );
        assert_eq!(compatible.matvec(&input).unwrap(), expected);
        assert_ne!(native.matvec(&input).unwrap(), expected);
    }

    #[test]
    fn owned_projection_requires_full_ggml_mode_for_bf16_activation() {
        let descriptor = descriptor("owned.bf16", vec![8, 4], GGML_TYPE_BF16, 64);
        let mut payload = Vec::with_capacity(64);
        for _ in 0..32 {
            payload.extend_from_slice(&0x3f80_u16.to_le_bytes());
        }
        let input = vec![1.001_f32; 8];

        let q8_only = OwnedGgufProjection::from_parts_with_numerics(
            descriptor.clone(),
            payload.clone(),
            1.0,
            1.0,
            GgufProjectionNumerics::GgmlQ8_0Activation,
        )
        .unwrap();
        let compatible = OwnedGgufProjection::from_parts_with_numerics(
            descriptor,
            payload,
            1.0,
            1.0,
            GgufProjectionNumerics::GgmlCpuActivation,
        )
        .unwrap();
        assert!(q8_only
            .matvec(&input)
            .unwrap()
            .iter()
            .all(|value| *value > 8.0));
        assert_eq!(compatible.matvec(&input).unwrap(), vec![8.0; 4]);
    }

    #[test]
    fn owned_projection_rejects_bad_geometry_payload_and_multipliers() {
        let descriptor = descriptor("owned.q8", vec![32, 2], crate::quant::GGML_TYPE_Q8_0, 68);
        let payload = q8_payload(32, 2);

        let mut rank_one = descriptor.clone();
        rank_one.dimensions.pop();
        assert!(OwnedGgufProjection::from_parts(rank_one, payload.clone(), 1.0, 1.0).is_err());

        let mut unsupported = descriptor.clone();
        unsupported.ggml_type = 999;
        assert!(OwnedGgufProjection::from_parts(unsupported, payload.clone(), 1.0, 1.0).is_err());

        assert!(OwnedGgufProjection::from_parts(
            descriptor.clone(),
            payload[..payload.len() - 1].to_vec(),
            1.0,
            1.0,
        )
        .is_err());

        let mut wrong_byte_len = descriptor.clone();
        wrong_byte_len.byte_len -= 1;
        assert!(
            OwnedGgufProjection::from_parts(wrong_byte_len, payload.clone(), 1.0, 1.0).is_err()
        );

        for (input_multiplier, output_multiplier) in [
            (0.0, 1.0),
            (-1.0, 1.0),
            (f32::NAN, 1.0),
            (1.0, 0.0),
            (1.0, -1.0),
            (1.0, f32::INFINITY),
        ] {
            assert!(OwnedGgufProjection::from_parts(
                descriptor.clone(),
                payload.clone(),
                input_multiplier,
                output_multiplier,
            )
            .is_err());
        }
    }

    #[test]
    fn owned_projection_rejects_non_finite_inputs_and_scaled_outputs() {
        let descriptor = descriptor("owned.q8", vec![32, 1], crate::quant::GGML_TYPE_Q8_0, 34);
        let payload = q8_payload(32, 1);
        let projection =
            OwnedGgufProjection::from_parts(descriptor.clone(), payload.clone(), 1.0, 1.0).unwrap();
        assert!(projection.matvec(&[f32::INFINITY; 32]).is_err());
        assert!(projection.matvec(&[f32::NAN; 32]).is_err());

        let scaled = OwnedGgufProjection::from_parts(descriptor, payload, 2.0, 1.0).unwrap();
        let mut huge = vec![f32::MAX; 32];
        assert!(scaled.matvec(&huge).is_err());
        huge[0] = f32::NAN;
        assert!(scaled.matvec(&huge).is_err());
    }

    #[test]
    fn providers_read_dense_rows_and_stream_quantized_logits() {
        let path = temp_path("providers");
        fs::write(&path, synthetic_boundary_file()).unwrap();
        let manifest = GgufManifest::inspect(
            &path,
            crate::GgufAuditOptions {
                hash_file: true,
                ..Default::default()
            },
        )
        .unwrap();
        let embedding = GgufEmbeddingRowProvider::from_manifest(&manifest, "emb", 2, 4).unwrap();
        let mut reader = manifest.open_tensor_reader(&path).unwrap();
        let row = embedding.read_row_with_bytes(&mut reader, 1).unwrap();
        assert_eq!(row.token_id, 1);
        assert_eq!(row.values, vec![5.0, 6.0, 7.0, 8.0]);
        assert_eq!(row.raw_bytes.len(), 8);

        let head =
            GgufStreamedOutputHead::from_reader(&mut reader, "norm", "out", 32, 3, 1.0e-6, 2)
                .unwrap();
        assert_eq!(head.numerics(), GgufProjectionNumerics::NativeF32Activation);
        assert_eq!(head.chunk_count(), 2);
        let logits = head.logits(&mut reader, &[1.0; 32]).unwrap();
        assert_eq!(logits.len(), 3);
        assert!(logits.iter().all(|value| value.is_finite()));

        let compatible = GgufStreamedOutputHead::from_reader_with_numerics(
            &mut reader,
            "norm",
            "out",
            32,
            3,
            1.0e-6,
            2,
            GgufProjectionNumerics::GgmlCpuActivation,
        )
        .unwrap();
        assert_eq!(
            compatible.numerics(),
            GgufProjectionNumerics::GgmlCpuActivation
        );
        let boundary_input = (0..32)
            .map(|index| {
                if index == 0 {
                    127.0
                } else {
                    index as f32 / 7.0
                }
            })
            .collect::<Vec<_>>();
        assert_ne!(
            head.logits(&mut reader, &boundary_input).unwrap(),
            compatible.logits(&mut reader, &boundary_input).unwrap()
        );

        let owned = OwnedGgufProjection::from_reader(&mut reader, "out", 1.0, 1.0).unwrap();
        // Projection only must preserve the producer's normalized values.
        let direct = head
            .logits_from_normalized_hidden(&mut reader, &boundary_input)
            .unwrap();
        assert_eq!(direct, owned.matvec(&boundary_input).unwrap());
        assert_ne!(direct, head.logits(&mut reader, &boundary_input).unwrap());
        let normalized = head.normalized_hidden(&boundary_input).unwrap();
        assert_eq!(
            head.logits(&mut reader, &boundary_input).unwrap(),
            head.logits_from_normalized_hidden(&mut reader, &normalized)
                .unwrap()
        );
        for invalid in [vec![0.0; 31], vec![f32::NAN; 32], vec![f32::INFINITY; 32]] {
            assert!(head
                .logits_from_normalized_hidden(&mut reader, &invalid)
                .is_err());
        }
        assert_eq!(owned.input_width(), 32);
        assert_eq!(owned.output_width(), 3);
        assert_eq!(owned.payload().len(), 3 * 34);
        assert!(owned
            .matvec(&[1.0; 32])
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));
        fs::remove_file(path).unwrap();
    }

    fn temp_path(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "augur-qwen35-weights-{label}-{}-{}.gguf",
            std::process::id(),
            TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn put_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    fn put_tensor(bytes: &mut Vec<u8>, name: &str, dimensions: &[u64], kind: u32, offset: u64) {
        put_string(bytes, name);
        bytes.extend_from_slice(&(dimensions.len() as u32).to_le_bytes());
        for dimension in dimensions {
            bytes.extend_from_slice(&dimension.to_le_bytes());
        }
        bytes.extend_from_slice(&kind.to_le_bytes());
        bytes.extend_from_slice(&offset.to_le_bytes());
    }

    fn synthetic_boundary_file() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&crate::gguf::GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&crate::gguf::GGUF_VERSION_3.to_le_bytes());
        bytes.extend_from_slice(&3_u64.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        put_tensor(&mut bytes, "emb", &[4, 2], GGML_TYPE_BF16, 0);
        put_tensor(&mut bytes, "norm", &[32], GGML_TYPE_F32, 32);
        put_tensor(&mut bytes, "out", &[32, 3], 8, 192);
        let data_offset = (bytes.len() as u64).div_ceil(32) * 32;
        bytes.resize(data_offset as usize + 192 + 3 * 34, 0);

        let embedding_offset = data_offset as usize;
        for (index, value) in [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            .into_iter()
            .enumerate()
        {
            let value = (value.to_bits() >> 16) as u16;
            bytes[embedding_offset + index * 2..embedding_offset + index * 2 + 2]
                .copy_from_slice(&value.to_le_bytes());
        }
        let norm_offset = embedding_offset + 32;
        for index in 0..32 {
            bytes[norm_offset + index * 4..norm_offset + index * 4 + 4]
                .copy_from_slice(&1.0_f32.to_le_bytes());
        }
        let output_offset = embedding_offset + 192;
        for row in 0..3 {
            let row_offset = output_offset + row * 34;
            bytes[row_offset..row_offset + 2].copy_from_slice(&0x3c00_u16.to_le_bytes());
            for lane in 0..32 {
                bytes[row_offset + 2 + lane] = (lane as u8).wrapping_add(row as u8 + 1);
            }
        }
        bytes
    }

    fn q8_payload(input_width: usize, output_width: usize) -> Vec<u8> {
        let blocks_per_row = input_width / crate::quant::Q8_0_BLOCK_ELEMENTS;
        let mut payload =
            vec![0_u8; output_width * blocks_per_row * crate::quant::Q8_0_BLOCK_BYTES];
        for row in 0..output_width {
            for block in 0..blocks_per_row {
                let start = (row * blocks_per_row + block) * crate::quant::Q8_0_BLOCK_BYTES;
                payload[start..start + 2].copy_from_slice(&0x3c00_u16.to_le_bytes());
                for lane in 0..crate::quant::Q8_0_BLOCK_ELEMENTS {
                    payload[start + 2 + lane] = (lane as u8).wrapping_add(row as u8 + 1);
                }
            }
        }
        payload
    }
}
