//! Complete, identity-bound Q8_0/F32 DFlash2 weights and same-weight CPU oracle.
//! No tensor is silently substituted with the target model or F16 checkpoint.

use crate::dflash_backbone::expected_tensor_specs;
use crate::dflash_block_producer::{produce_dflash_block_from_weights, DFlashBlockWeights};
use crate::{DFlashBlockProducerError, DFlashBlockProducerInput, DFlashBlockProducerOutput};
use augur_model_qwen35::{
    dequantize_tensor, DFlashConfig, GgufAuditOptions, GgufManifest, GgufTensor, GgufValue,
};
use half::f16;
mod selector_profile;
pub use selector_profile::{GATE_SHA256, GATE_PROFILE};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

pub const DFLASH_Q8_BLOCK_PRODUCER_SCHEMA: &str = "qwen35-dflash-five-layer-block-q8-cpu-v1";

pub struct DFlashQ8Backbone {
    config: DFlashConfig,
    manifest: GgufManifest,
    // Source (HF) tensor names; descriptor names remain the exact GGUF names.
    tensors: BTreeMap<String, (GgufTensor, Vec<u8>)>,
}

impl DFlashQ8Backbone {
    pub fn load(
        path: impl AsRef<Path>,
        config: DFlashConfig,
    ) -> Result<Self, DFlashBlockProducerError> {
        Self::load_impl(path.as_ref(), config, false)
    }

    pub fn load_pinned(
        path: impl AsRef<Path>, config: DFlashConfig,
    ) -> Result<Self, DFlashBlockProducerError> {
        Self::load_impl(path.as_ref(), config, true)
    }

    fn load_impl(path: &Path, config: DFlashConfig, pinned: bool) -> Result<Self, DFlashBlockProducerError> {
        config.validate_exact_profile().map_err(err)?;
        let mut artifact = if pinned {
            Some(augur_model_qwen35::GgufPinnedArtifact::open(path, GgufAuditOptions::default()).map_err(err)?)
        } else { None };
        let manifest = if let Some(a) = &artifact { a.manifest().clone() }
            else { GgufManifest::inspect(path, GgufAuditOptions::default()).map_err(err)? };
        validate_metadata(&manifest)?;
        let embedded_gate=selector_profile::admit(&manifest)?;
        let specs = expected_tensor_specs(&config);
        let expected = specs
            .keys()
            .map(|name| gguf_name(name).unwrap())
            .collect::<BTreeSet<_>>();
        let actual = manifest
            .tensors
            .iter()
            .map(|tensor| tensor.name.clone())
            .collect::<BTreeSet<_>>();
        if expected != actual || manifest.tensors.len() != specs.len() {
            return Err(err("Q8 backbone requires the exact 81-tensor set"));
        }
        let mut tensors = BTreeMap::new();
        {
            let mut reader = if let Some(a) = artifact.as_mut() { a.tensor_reader().map_err(err)? }
                else { manifest.open_tensor_reader(path).map_err(err)? };
            for (name, shape) in specs {
                let gguf_name = gguf_name(&name).ok_or_else(|| err("unmapped tensor"))?;
                let tensor = manifest
                    .tensor(&gguf_name)
                    .ok_or_else(|| err("missing tensor"))?;
                validate_tensor(tensor, &shape)?;
                let bytes = reader.read_tensor_bytes(&gguf_name).map_err(err)?;
                if embedded_gate && tensor.name=="selector_hidden.weight" { selector_profile::validate_bytes(&bytes)?; }
                tensors.insert(name, (tensor.clone(), bytes));
            }
        }
        Ok(Self {
            config,
            manifest,
            tensors,
        })
    }

    /// Validated original F32 gate with calibration already folded in the GGUF.
    pub fn embedded_selector_gate(&self) -> Option<&[u8]> {
        self.tensors.get("candidate_selector.hidden_projection.weight")
            .filter(|(t,_)|t.ggml_type==0).map(|(_,b)|b.as_slice())
    }
    pub fn config(&self) -> &DFlashConfig {
        &self.config
    }
    pub fn manifest(&self) -> &GgufManifest {
        &self.manifest
    }
    pub fn payload_bytes(&self) -> usize {
        self.tensors.values().map(|(_, b)| b.len()).sum()
    }
    /// Bounded same-Q8 selector oracle: dequantize only one requested row.
    pub fn selector_codebook_row(
        &self,
        predecessor: bool,
        token: u32,
    ) -> Result<Vec<f32>, DFlashBlockProducerError> {
        let name = if predecessor {
            "candidate_selector.predecessor_codebook"
        } else {
            "candidate_selector.successor_codebook"
        };
        let (_, bytes) = self.tensor(name)?;
        selector_q8_row(bytes, token)
    }
    #[cfg(feature = "cuda")]
    pub(crate) fn tensors(&self) -> impl Iterator<Item = (&str, &GgufTensor, &[u8])> {
        self.tensors
            .iter()
            .map(|(n, (t, b))| (n.as_str(), t, b.as_slice()))
    }
    pub(crate) fn tensor(
        &self,
        name: &str,
    ) -> Result<(&GgufTensor, &[u8]), DFlashBlockProducerError> {
        self.tensors
            .get(name)
            .map(|(t, b)| (t, b.as_slice()))
            .ok_or_else(|| err(format!("unknown Q8 tensor {name}")))
    }
    pub(crate) fn read_tensor(
        &self,
        name: &str,
        shape: &[usize],
    ) -> Result<Vec<f32>, DFlashBlockProducerError> {
        let (tensor, bytes) = self.tensor(name)?;
        validate_tensor(tensor, shape)?;
        if tensor.ggml_type != 0 {
            return Err(err(
                "vector/base read requires F32, not a dequantized dense copy",
            ));
        }
        dequantize_tensor(tensor, bytes).map_err(err)
    }
    pub(crate) fn matvec_batch(
        &self,
        name: &str,
        rows: usize,
        cols: usize,
        input: &[f32],
        batch: usize,
    ) -> Result<Vec<f32>, DFlashBlockProducerError> {
        let (tensor, bytes) = self.tensor(name)?;
        validate_tensor(tensor, &[rows, cols])?;
        if tensor.ggml_type==0 {return selector_profile::matvec_f64(bytes,rows,cols,input,batch);}
        q8_matvec_f64(bytes, rows, cols, input, batch)
    }
}

pub fn produce_dflash_q8_block_cpu(
    weights: &DFlashQ8Backbone,
    input: DFlashBlockProducerInput<'_>,
) -> Result<DFlashBlockProducerOutput, DFlashBlockProducerError> {
    let mut result =
        produce_dflash_block_from_weights(DFlashBlockWeights::Q8(weights), input, |a| {
            a.cpu_reference()
        })?;
    result.schema = DFLASH_Q8_BLOCK_PRODUCER_SCHEMA.into();
    Ok(result)
}

fn err(value: impl std::fmt::Display) -> DFlashBlockProducerError {
    DFlashBlockProducerError::Invalid(value.to_string())
}

fn validate_metadata(manifest: &GgufManifest) -> Result<(), DFlashBlockProducerError> {
    if manifest
        .metadata("general.architecture")
        .and_then(GgufValue::as_str)
        != Some("dflash")
    {
        return Err(err("GGUF architecture is not dflash"));
    }
    for (key, expected) in [
        ("block_count", 5),
        ("context_length", 262144),
        ("embedding_length", 5120),
        ("feed_forward_length", 17408),
        ("attention.head_count", 32),
        ("attention.head_count_kv", 8),
        ("attention.key_length", 128),
        ("attention.value_length", 128),
        ("block_size", 8),
        ("conv_kernel_size", 2),
        ("conv_group_size", 16),
        ("selector_rank", 256),
        ("selector_top_k", 16),
        ("attention.sliding_window", 2048),
    ] {
        if manifest
            .metadata(&format!("dflash.{key}"))
            .and_then(GgufValue::as_u64)
            != Some(expected)
        {
            return Err(err(format!("DFlash GGUF metadata mismatch: {key}")));
        }
    }
    if manifest.metadata("dflash.attention.causal") != Some(&GgufValue::Bool(false))
        || manifest
            .metadata("dflash.rope.freq_base")
            .and_then(GgufValue::as_f64)
            != Some(10_000_000.0)
        || manifest
            .metadata("dflash.attention.layer_norm_rms_epsilon")
            .and_then(GgufValue::as_f64)
            .is_none_or(|v| (v - 1e-6).abs() > 1e-12)
    {
        return Err(err("DFlash causal/RoPE/epsilon mismatch"));
    }
    for (key, expected) in [
        ("dflash.target_layers", vec![6, 20, 34, 48, 62]), // GGUF stores HF ids + 1
        ("dflash.rope.dimension_sections", vec![64, 0, 0, 0]),
    ] {
        let Some(GgufValue::Array(values)) = manifest.metadata(key) else {
            return Err(err(format!("missing {key}")));
        };
        if values
            .iter()
            .map(GgufValue::as_u64)
            .collect::<Option<Vec<_>>>()
            != Some(expected)
        {
            return Err(err(format!("DFlash metadata mismatch: {key}")));
        }
    }
    if manifest.metadata("dflash.attention.sliding_window_pattern")
        != Some(&GgufValue::Array(vec![GgufValue::Bool(true); 5]))
    {
        return Err(err("DFlash sliding window pattern mismatch"));
    }
    Ok(())
}

fn validate_tensor(tensor: &GgufTensor, shape: &[usize]) -> Result<(), DFlashBlockProducerError> {
    let expected_type = if shape.len() == 2 && !(tensor.name=="selector_hidden.weight" && shape==[256,5120] && tensor.ggml_type==0) { 8 } else { 0 };
    let dimensions = shape.iter().rev().map(|&v| v as u64).collect::<Vec<_>>();
    let elements = shape
        .iter()
        .try_fold(1usize, |a, &v| a.checked_mul(v))
        .ok_or_else(|| err("tensor size overflow"))?;
    let bytes = if expected_type == 8 {
        if !shape[1].is_multiple_of(32) {
            return Err(err("Q8 columns unaligned"));
        }
        elements / 32 * 34
    } else {
        elements
            .checked_mul(4)
            .ok_or_else(|| err("tensor byte overflow"))?
    };
    if tensor.dimensions != dimensions
        || tensor.ggml_type != expected_type
        || tensor.byte_len != bytes as u64
    {
        return Err(err(format!(
            "Q8 tensor geometry/type mismatch: {}",
            tensor.name
        )));
    }
    Ok(())
}

fn gguf_name(name: &str) -> Option<String> {
    let global = match name {
        "fc.weight" => Some("fc.weight"),
        "hidden_norm.weight" => Some("enc.output_norm.weight"),
        "norm.weight" => Some("output_norm.weight"),
        "candidate_selector.hidden_projection.weight" => Some("selector_hidden.weight"),
        "candidate_selector.predecessor_codebook" => Some("selector_predecessor.weight"),
        "candidate_selector.successor_codebook" => Some("selector_successor.weight"),
        _ => None,
    };
    if let Some(mapped) = global {
        return Some(mapped.into());
    }
    let rest = name.strip_prefix("layers.")?;
    let (layer, suffix) = rest.split_once('.')?;
    if layer.parse::<usize>().ok()? >= 5 {
        return None;
    }
    let suffix = match suffix {
        "attention_conv.base_kernel" => "attn_conv_base",
        "attention_conv.kernel_projection.weight" => "attn_conv_proj.weight",
        "input_layernorm.weight" => "attn_norm.weight",
        "post_attention_layernorm.weight" => "ffn_norm.weight",
        "mlp_conv.base_kernel" => "ffn_conv_base",
        "mlp_conv.kernel_projection.weight" => "ffn_conv_proj.weight",
        "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.up_proj.weight" => "ffn_up.weight",
        "mlp.down_proj.weight" => "ffn_down.weight",
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.o_proj.weight" => "attn_output.weight",
        "self_attn.q_norm.weight" => "attn_q_norm.weight",
        "self_attn.k_norm.weight" => "attn_k_norm.weight",
        _ => return None,
    };
    Some(format!("blk.{layer}.{suffix}"))
}

fn selector_q8_row(bytes: &[u8], token: u32) -> Result<Vec<f32>, DFlashBlockProducerError> {
    const ROW: usize = 8 * 34;
    if bytes.len() != 248320 * ROW || token >= 248320 {
        return Err(err("selector Q8 codebook extent/token"));
    }
    let row = &bytes[token as usize * ROW..(token as usize + 1) * ROW];
    let mut values = Vec::with_capacity(256);
    for block in row.chunks_exact(34) {
        let scale = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        if !scale.is_finite() {
            return Err(err("non-finite selector Q8 scale"));
        }
        values.extend(block[2..].iter().map(|&q| f32::from(q as i8) * scale));
    }
    Ok(values)
}

fn q8_matvec_f64(
    bytes: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
    batch: usize,
) -> Result<Vec<f32>, DFlashBlockProducerError> {
    if rows == 0
        || cols == 0
        || !cols.is_multiple_of(32)
        || batch == 0
        || rows.checked_mul(cols / 32).and_then(|n| n.checked_mul(34)) != Some(bytes.len())
        || batch.checked_mul(cols) != Some(input.len())
        || input.iter().any(|x| !x.is_finite())
    {
        return Err(err("Q8 CPU matvec geometry/finite input"));
    }
    let mut output = vec![
        0.0;
        batch
            .checked_mul(rows)
            .ok_or_else(|| err("output extent overflow"))?
    ];
    for row in 0..rows {
        let row_bytes = &bytes[row * (cols / 32) * 34..(row + 1) * (cols / 32) * 34];
        // One unpacked row, not a full F32 model shadow.
        let mut decoded = Vec::with_capacity(cols);
        for block in row_bytes.chunks_exact(34) {
            let scale = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
            if !scale.is_finite() {
                return Err(err("non-finite Q8 scale"));
            }
            decoded.extend(block[2..].iter().map(|&q| (q as i8) as f32 * scale));
        }
        for item in 0..batch {
            let x = &input[item * cols..(item + 1) * cols];
            let sum = decoded
                .iter()
                .zip(x)
                .map(|(&w, &v)| f64::from(w) * f64::from(v))
                .sum::<f64>() as f32;
            if !sum.is_finite() {
                return Err(err("non-finite Q8 matvec output"));
            }
            output[item * rows + row] = sum;
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn selector_row_is_bounded_signed_and_rejects_nonfinite_scale() {
        let mut bytes = vec![0u8; 248320 * 272];
        let o = 248319 * 272 + 7 * 34;
        bytes[o..o + 2].copy_from_slice(&f16::from_f32(-0.5).to_bits().to_le_bytes());
        bytes[o + 2] = (-128i8) as u8;
        let row = selector_q8_row(&bytes, 248319).unwrap();
        assert_eq!(row.len(), 256);
        assert_eq!(row[224], 64.0);
        assert_eq!(row[0], 0.0);
        assert!(selector_q8_row(&bytes, 248320).is_err());
        assert!(selector_q8_row(&bytes[..bytes.len() - 1], 0).is_err());
        bytes[o..o + 2].copy_from_slice(&f16::NAN.to_bits().to_le_bytes());
        assert!(selector_q8_row(&bytes, 248319).is_err());
    }
    #[test]
    fn quantized_oracle_keeps_signed_codes_scale_and_batch_orientation() {
        let mut bytes = Vec::new();
        for scale in [0.5, -0.25] {
            bytes.extend_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
            bytes.extend((0..32).map(|n| (n as i8 - 16) as u8));
        }
        let x = [vec![1.0; 32], (0..32).map(|v| v as f32).collect()].concat();
        assert_eq!(
            q8_matvec_f64(&bytes, 2, 32, &x, 2).unwrap(),
            [-8.0, 4.0, 1240.0, -620.0]
        );
        assert!(q8_matvec_f64(&bytes[..67], 2, 32, &x, 2).is_err());
        bytes[..2].copy_from_slice(&f16::INFINITY.to_bits().to_le_bytes());
        assert!(q8_matvec_f64(&bytes, 2, 32, &x, 2).is_err());
    }
    #[test]
    fn names_and_shapes_do_not_silently_transpose_or_substitute() {
        assert_eq!(
            gguf_name("layers.4.self_attn.o_proj.weight").as_deref(),
            Some("blk.4.attn_output.weight")
        );
        assert!(gguf_name("layers.5.self_attn.q_proj.weight").is_none());
        let mut t = GgufTensor {
            name: "q".into(),
            dimensions: vec![5120, 4096],
            ggml_type: 8,
            relative_offset: 0,
            absolute_offset: 0,
            byte_len: 5120 * 4096 / 32 * 34,
        };
        assert!(validate_tensor(&t, &[4096, 5120]).is_ok());
        assert!(validate_tensor(&t, &[5120, 4096]).is_err());
        t.ggml_type = 0;
        assert!(validate_tensor(&t, &[4096, 5120]).is_err());
    }
}
