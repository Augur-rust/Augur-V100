//! CPU-only decoding of the GGUF quantized types used by the Qwen3.8 lane.
//!
//! The byte layouts in this module intentionally mirror llama.cpp's GGML
//! wire formats.  They are not interchangeable with the compact formats in
//! `augur-formats`: GGML NVFP4 has four per-sub-block UE4M3 scales followed by
//! 32 packed nibbles (36 bytes total), while the project compact format has a
//! different envelope.

use crate::{GgufManifest, GgufTensor, GgufTensorReader, Qwen35Error};
use serde::Serialize;

/// GGML type tag for one little-endian f32 value per four bytes.
pub const GGML_TYPE_F32: u32 = 0;
/// GGML type tag for a 32-value block with an FP16 scale and int8 values.
pub const GGML_TYPE_Q8_0: u32 = 8;
/// GGML type tag for one little-endian BF16 value per two bytes.
pub const GGML_TYPE_BF16: u32 = 30;
/// GGML type tag for a 64-value NVFP4 block.
pub const GGML_TYPE_NVFP4: u32 = 40;

pub const Q8_0_BLOCK_ELEMENTS: usize = 32;
pub const Q8_0_BLOCK_BYTES: usize = 34;
pub const NVFP4_BLOCK_ELEMENTS: usize = 64;
pub const NVFP4_SUB_BLOCK_ELEMENTS: usize = 16;
pub const NVFP4_BLOCK_BYTES: usize = 36;
pub const F32_ELEMENT_BYTES: usize = 4;
pub const BF16_ELEMENT_BYTES: usize = 2;

const NVFP4_SUB_BLOCKS: usize = NVFP4_BLOCK_ELEMENTS / NVFP4_SUB_BLOCK_ELEMENTS;
const NVFP4_SUB_BLOCK_PACKED_BYTES: usize = NVFP4_SUB_BLOCK_ELEMENTS / 2;

/// GGML quantized type tags accepted by the Qwen CPU reference decoder.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum Qwen35QuantType {
    F32,
    Q8_0,
    Bf16,
    Nvfp4,
}

impl Qwen35QuantType {
    pub fn from_ggml_type(ggml_type: u32) -> Result<Self, Qwen35Error> {
        match ggml_type {
            GGML_TYPE_F32 => Ok(Self::F32),
            GGML_TYPE_Q8_0 => Ok(Self::Q8_0),
            GGML_TYPE_BF16 => Ok(Self::Bf16),
            GGML_TYPE_NVFP4 => Ok(Self::Nvfp4),
            other => Err(Qwen35Error::invalid(
                "ggml_type",
                format!(
                    "unsupported Qwen reference type {other}; expected F32 (0), Q8_0 (8), BF16 (30), or NVFP4 (40)"
                ),
            )),
        }
    }

    pub const fn ggml_type(self) -> u32 {
        match self {
            Self::F32 => GGML_TYPE_F32,
            Self::Q8_0 => GGML_TYPE_Q8_0,
            Self::Bf16 => GGML_TYPE_BF16,
            Self::Nvfp4 => GGML_TYPE_NVFP4,
        }
    }

    pub const fn block_elements(self) -> usize {
        match self {
            Self::F32 => 1,
            Self::Q8_0 => Q8_0_BLOCK_ELEMENTS,
            Self::Bf16 => 1,
            Self::Nvfp4 => NVFP4_BLOCK_ELEMENTS,
        }
    }

    pub const fn block_bytes(self) -> usize {
        match self {
            Self::F32 => F32_ELEMENT_BYTES,
            Self::Q8_0 => Q8_0_BLOCK_BYTES,
            Self::Bf16 => BF16_ELEMENT_BYTES,
            Self::Nvfp4 => NVFP4_BLOCK_BYTES,
        }
    }
}

/// Decode one GGML Q8_0 block into exactly 32 f32 values.
pub fn dequantize_q8_0_block(bytes: &[u8], output: &mut [f32]) -> Result<(), Qwen35Error> {
    require_exact_len("q8_0 block", bytes.len(), Q8_0_BLOCK_BYTES)?;
    require_exact_len("q8_0 output", output.len(), Q8_0_BLOCK_ELEMENTS)?;

    let scale = f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
    if !scale.is_finite() {
        return Err(Qwen35Error::invalid(
            "q8_0.scale",
            "FP16 scale must be finite",
        ));
    }
    for (index, destination) in output.iter_mut().enumerate() {
        let code = i8::from_ne_bytes([bytes[2 + index]]);
        *destination = f32::from(code) * scale;
    }
    Ok(())
}

/// Decode one GGML F32 value sequence.  The input and output lengths must
/// match exactly and every decoded value must be finite.
pub fn dequantize_f32(bytes: &[u8], output: &mut [f32]) -> Result<(), Qwen35Error> {
    if output.is_empty() {
        return Err(Qwen35Error::invalid(
            "f32 output",
            "must contain at least one element",
        ));
    }
    let expected_bytes = output
        .len()
        .checked_mul(F32_ELEMENT_BYTES)
        .ok_or(Qwen35Error::SizeOverflow)?;
    require_exact_len("f32 payload", bytes.len(), expected_bytes)?;
    for (index, destination) in output.iter_mut().enumerate() {
        *destination = decode_f32_value(bytes, index)?;
    }
    Ok(())
}

/// Decode one GGML BF16 value sequence.  The input and output lengths must
/// match exactly; unlike a block format, BF16 has one value per two bytes.
pub fn dequantize_bf16(bytes: &[u8], output: &mut [f32]) -> Result<(), Qwen35Error> {
    if output.is_empty() {
        return Err(Qwen35Error::invalid(
            "bf16 output",
            "must contain at least one element",
        ));
    }
    let expected_bytes = output
        .len()
        .checked_mul(BF16_ELEMENT_BYTES)
        .ok_or(Qwen35Error::SizeOverflow)?;
    require_exact_len("bf16 payload", bytes.len(), expected_bytes)?;
    for (index, destination) in output.iter_mut().enumerate() {
        *destination = decode_bf16_value(bytes, index)?;
    }
    Ok(())
}

/// Decode one GGML NVFP4 block into exactly 64 f32 values.
///
/// Each 16-value sub-block owns one unsigned UE4M3 scale byte.  The packed
/// bytes store the low nibbles for the first eight values and the high
/// nibbles for the second eight values, matching `dequantize_row_nvfp4` in
/// llama.cpp.  llama.cpp's UE4M3 helper returns the scale multiplied by 0.5
/// because its shared FP4 lookup table stores doubled E2M1 values; this
/// decoder uses ordinary E2M1 values, so it deliberately uses the raw UE4M3
/// scale instead.
pub fn dequantize_nvfp4_block(bytes: &[u8], output: &mut [f32]) -> Result<(), Qwen35Error> {
    require_exact_len("nvfp4 block", bytes.len(), NVFP4_BLOCK_BYTES)?;
    require_exact_len("nvfp4 output", output.len(), NVFP4_BLOCK_ELEMENTS)?;

    for sub_block in 0..NVFP4_SUB_BLOCKS {
        let scale = ue4m3_scale_to_f32(bytes[sub_block]);
        let packed_start = NVFP4_SUB_BLOCKS + sub_block * NVFP4_SUB_BLOCK_PACKED_BYTES;
        let output_start = sub_block * NVFP4_SUB_BLOCK_ELEMENTS;
        for index in 0..NVFP4_SUB_BLOCK_PACKED_BYTES {
            let packed = bytes[packed_start + index];
            output[output_start + index] = e2m1_value(packed & 0x0f) * scale;
            output[output_start + NVFP4_SUB_BLOCK_PACKED_BYTES + index] =
                e2m1_value(packed >> 4) * scale;
        }
    }
    Ok(())
}

/// Decode a complete tensor payload using its GGUF descriptor.
///
/// GGML stores the contiguous row width in `dimensions[0]`.  Block formats
/// must have a block-aligned row width; accepting a block that crosses rows
/// would produce a numerically plausible but layout-incompatible tensor.
pub fn dequantize_tensor(tensor: &GgufTensor, bytes: &[u8]) -> Result<Vec<f32>, Qwen35Error> {
    let (quant_type, element_count, _) = validate_tensor_payload(tensor, bytes.len())?;

    let mut output = Vec::new();
    output
        .try_reserve_exact(element_count)
        .map_err(|_| Qwen35Error::Gguf("dequantized tensor allocation failed".into()))?;
    output.resize(element_count, 0.0);
    match quant_type {
        Qwen35QuantType::F32 => dequantize_f32(bytes, &mut output)?,
        Qwen35QuantType::Q8_0 => {
            for (block, destination) in bytes
                .chunks_exact(Q8_0_BLOCK_BYTES)
                .zip(output.chunks_exact_mut(Q8_0_BLOCK_ELEMENTS))
            {
                dequantize_q8_0_block(block, destination)?;
            }
        }
        Qwen35QuantType::Bf16 => dequantize_bf16(bytes, &mut output)?,
        Qwen35QuantType::Nvfp4 => {
            for (block, destination) in bytes
                .chunks_exact(NVFP4_BLOCK_BYTES)
                .zip(output.chunks_exact_mut(NVFP4_BLOCK_ELEMENTS))
            {
                dequantize_nvfp4_block(block, destination)?;
            }
        }
    }
    Ok(output)
}

/// Compute one GGUF matrix-vector product directly from quantized bytes.
///
/// The descriptor uses GGML's `[K, N]` dimensions: `dimensions[0]` is the
/// contiguous input width and `dimensions[1]` is the number of output rows.
/// This is the orientation used by the Qwen converter and is the inverse of
/// the `[N, K]` shape held by [`crate::ReferenceTensor`].  Only a single
/// block-sized scratch array is used, so this path does not materialize a
/// full f32 copy of the weight tensor.
pub fn quantized_matvec(
    tensor: &GgufTensor,
    bytes: &[u8],
    input: &[f32],
) -> Result<Vec<f32>, Qwen35Error> {
    let (quant_type, k, n, blocks_per_row) = validate_matvec_inputs(tensor, bytes, input)?;

    let mut output = Vec::new();
    output
        .try_reserve_exact(n)
        .map_err(|_| Qwen35Error::Gguf("quantized matvec output allocation failed".into()))?;
    output.resize(n, 0.0);
    for (row, destination) in output.iter_mut().enumerate() {
        let row_block_start = row
            .checked_mul(blocks_per_row)
            .ok_or(Qwen35Error::SizeOverflow)?;
        let mut sum = 0.0f32;
        match quant_type {
            Qwen35QuantType::F32 => {
                let row_start = row.checked_mul(k).ok_or(Qwen35Error::SizeOverflow)?;
                for (column, input_value) in input.iter().enumerate() {
                    let value = decode_f32_value(bytes, row_start + column)?;
                    dot_accumulate(&mut sum, value, *input_value)?;
                }
            }
            Qwen35QuantType::Q8_0 => {
                let mut decoded = [0.0f32; Q8_0_BLOCK_ELEMENTS];
                for block in 0..blocks_per_row {
                    let block_index = row_block_start
                        .checked_add(block)
                        .ok_or(Qwen35Error::SizeOverflow)?;
                    let start = block_index
                        .checked_mul(Q8_0_BLOCK_BYTES)
                        .ok_or(Qwen35Error::SizeOverflow)?;
                    dequantize_q8_0_block(&bytes[start..start + Q8_0_BLOCK_BYTES], &mut decoded)?;
                    let input_start = block * Q8_0_BLOCK_ELEMENTS;
                    for lane in 0..Q8_0_BLOCK_ELEMENTS {
                        dot_accumulate(&mut sum, decoded[lane], input[input_start + lane])?;
                    }
                }
            }
            Qwen35QuantType::Bf16 => {
                let row_start = row.checked_mul(k).ok_or(Qwen35Error::SizeOverflow)?;
                for (column, input_value) in input.iter().enumerate() {
                    let value = decode_bf16_value(bytes, row_start + column)?;
                    dot_accumulate(&mut sum, value, *input_value)?;
                }
            }
            Qwen35QuantType::Nvfp4 => {
                let mut decoded = [0.0f32; NVFP4_BLOCK_ELEMENTS];
                for block in 0..blocks_per_row {
                    let block_index = row_block_start
                        .checked_add(block)
                        .ok_or(Qwen35Error::SizeOverflow)?;
                    let start = block_index
                        .checked_mul(NVFP4_BLOCK_BYTES)
                        .ok_or(Qwen35Error::SizeOverflow)?;
                    dequantize_nvfp4_block(&bytes[start..start + NVFP4_BLOCK_BYTES], &mut decoded)?;
                    let input_start = block * NVFP4_BLOCK_ELEMENTS;
                    for lane in 0..NVFP4_BLOCK_ELEMENTS {
                        dot_accumulate(&mut sum, decoded[lane], input[input_start + lane])?;
                    }
                }
            }
        }
        if !sum.is_finite() {
            return Err(Qwen35Error::invalid(
                "quantized_matvec.output",
                format!("row {row} accumulated a non-finite value"),
            ));
        }
        *destination = sum;
    }
    Ok(output)
}

/// Compute one GGUF matrix-vector product while fusing dequantization with
/// the dot product.
///
/// This is a CPU reference/benchmark path.  It deliberately keeps the same
/// checked descriptor, payload, and finite-value contract as
/// [`quantized_matvec`], but does not write a decoded scratch block.  The
/// accumulation order is therefore allowed to differ by normal f32 rounding
/// from the materialized baseline; callers comparing the two paths must use a
/// documented numerical tolerance rather than bit equality.
pub fn quantized_matvec_fused(
    tensor: &GgufTensor,
    bytes: &[u8],
    input: &[f32],
) -> Result<Vec<f32>, Qwen35Error> {
    let (quant_type, k, n, blocks_per_row) = validate_matvec_inputs(tensor, bytes, input)?;

    let mut output = Vec::new();
    output
        .try_reserve_exact(n)
        .map_err(|_| Qwen35Error::Gguf("fused quantized matvec output allocation failed".into()))?;
    output.resize(n, 0.0);

    for (row, destination) in output.iter_mut().enumerate() {
        let row_block_start = row
            .checked_mul(blocks_per_row)
            .ok_or(Qwen35Error::SizeOverflow)?;
        let sum = match quant_type {
            Qwen35QuantType::F32 => fused_f32_row(bytes, input, row, k)?,
            Qwen35QuantType::Q8_0 => fused_q8_row(bytes, input, row_block_start, blocks_per_row)?,
            Qwen35QuantType::Bf16 => fused_bf16_row(bytes, input, row, k)?,
            Qwen35QuantType::Nvfp4 => {
                fused_nvfp4_row(bytes, input, row_block_start, blocks_per_row)?
            }
        };
        if !sum.is_finite() {
            return Err(Qwen35Error::invalid(
                "quantized_matvec_fused.output",
                format!("row {row} accumulated a non-finite value"),
            ));
        }
        *destination = sum;
    }
    Ok(output)
}

/// Dispatch the fused reference to an AVX2 implementation when the current
/// x86_64 CPU provides it.  Q8_0 and NVFP4 have independently checked SIMD
/// layouts; BF16 and F32 continue to use the checked fused-scalar path.
/// Non-x86 targets always use the scalar path.
pub fn quantized_matvec_fused_simd(
    tensor: &GgufTensor,
    bytes: &[u8],
    input: &[f32],
) -> Result<Vec<f32>, Qwen35Error> {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // The feature check immediately above is the precondition for these
        // target-feature-specialized functions.
        return match tensor.ggml_type {
            GGML_TYPE_Q8_0 => unsafe { quantized_matvec_fused_avx2_q8(tensor, bytes, input) },
            GGML_TYPE_NVFP4 => unsafe { quantized_matvec_fused_avx2_nvfp4(tensor, bytes, input) },
            _ => quantized_matvec_fused(tensor, bytes, input),
        };
    }
    quantized_matvec_fused(tensor, bytes, input)
}

/// Quantize one activation vector to GGML Q8_0 blocks using llama.cpp's x86
/// runtime convention.
///
/// This is an explicit numerical-compatibility boundary, not the default
/// projection ABI. Each 32-value block derives integer codes from the full
/// precision `127 / abs_max` multiplier while storing the block scale as
/// round-to-nearest-even FP16. x86 llama.cpp uses the same split convention in
/// `quantize_row_q8_0` before a quantized matrix-vector product.
pub fn quantize_q8_0_activation(input: &[f32]) -> Result<Vec<u8>, Qwen35Error> {
    if input.is_empty() || !input.len().is_multiple_of(Q8_0_BLOCK_ELEMENTS) {
        return Err(Qwen35Error::invalid(
            "q8_0_activation.input",
            format!(
                "expected a non-empty multiple of {Q8_0_BLOCK_ELEMENTS} finite values, got {}",
                input.len()
            ),
        ));
    }
    if input.iter().any(|value| !value.is_finite()) {
        return Err(Qwen35Error::invalid(
            "q8_0_activation.input",
            "activation values must be finite",
        ));
    }

    let block_count = input.len() / Q8_0_BLOCK_ELEMENTS;
    let payload_bytes = block_count
        .checked_mul(Q8_0_BLOCK_BYTES)
        .ok_or(Qwen35Error::SizeOverflow)?;
    let mut payload = Vec::new();
    payload
        .try_reserve_exact(payload_bytes)
        .map_err(|_| Qwen35Error::Gguf("Q8_0 activation allocation failed".into()))?;

    for block in input.chunks_exact(Q8_0_BLOCK_ELEMENTS) {
        let abs_max = block
            .iter()
            .fold(0.0_f32, |current, value| current.max(value.abs()));
        let scale = abs_max / 127.0;
        let inverse_scale = if abs_max != 0.0 { 127.0 / abs_max } else { 0.0 };
        payload.extend_from_slice(&f32_to_f16(scale).to_le_bytes());
        for value in block {
            let code = (*value * inverse_scale)
                .round_ties_even()
                .clamp(-128.0, 127.0) as i8;
            payload.push(code as u8);
        }
    }
    debug_assert_eq!(payload.len(), payload_bytes);
    Ok(payload)
}

/// Multiply a Q8_0 or NVFP4 weight matrix by an f32 activation after applying
/// the explicit llama.cpp activation-Q8_0 boundary.
///
/// The eight f32 accumulators, four-code grouping, fused multiply-add, and
/// final reduction tree mirror the AVX2 `ggml_vec_dot_q8_0_q8_0` path used by
/// the local llama.cpp parity oracle. Keeping this as a separately named API
/// prevents a diagnostic parity experiment from silently changing the
/// engine's existing f32-activation projection behavior.
pub fn quantized_matvec_activation_q8_0(
    tensor: &GgufTensor,
    bytes: &[u8],
    input: &[f32],
) -> Result<Vec<f32>, Qwen35Error> {
    let (quant_type, _k, output_width, blocks_per_row) =
        validate_matvec_inputs(tensor, bytes, input)?;
    if !matches!(quant_type, Qwen35QuantType::Q8_0 | Qwen35QuantType::Nvfp4) {
        return Err(Qwen35Error::invalid(
            "q8_0_activation.weight_type",
            format!(
                "expected Q8_0 ({GGML_TYPE_Q8_0}) or NVFP4 ({GGML_TYPE_NVFP4}) weight, got {}",
                tensor.ggml_type
            ),
        ));
    }

    let activation = quantize_q8_0_activation(input)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(output_width)
        .map_err(|_| Qwen35Error::Gguf("Q8_0 activation matvec output allocation failed".into()))?;
    output.resize(output_width, 0.0);

    for (row, destination) in output.iter_mut().enumerate() {
        let row_block_start = row
            .checked_mul(blocks_per_row)
            .ok_or(Qwen35Error::SizeOverflow)?;
        let mut accumulator = [0.0_f32; 8];
        match quant_type {
            Qwen35QuantType::Q8_0 => {
                for block in 0..blocks_per_row {
                    let weight_block = row_block_start
                        .checked_add(block)
                        .ok_or(Qwen35Error::SizeOverflow)?;
                    let weight_start = weight_block
                        .checked_mul(Q8_0_BLOCK_BYTES)
                        .ok_or(Qwen35Error::SizeOverflow)?;
                    let activation_start = block
                        .checked_mul(Q8_0_BLOCK_BYTES)
                        .ok_or(Qwen35Error::SizeOverflow)?;
                    let weight_block = &bytes[weight_start..weight_start + Q8_0_BLOCK_BYTES];
                    let activation_block =
                        &activation[activation_start..activation_start + Q8_0_BLOCK_BYTES];
                    let weight_scale =
                        f16_to_f32(u16::from_le_bytes([weight_block[0], weight_block[1]]));
                    let activation_scale = f16_to_f32(u16::from_le_bytes([
                        activation_block[0],
                        activation_block[1],
                    ]));
                    let combined_scale = weight_scale * activation_scale;
                    if !combined_scale.is_finite() {
                        return Err(Qwen35Error::invalid(
                            "q8_0_activation.scale",
                            "combined weight and activation scale must be finite",
                        ));
                    }

                    for (lane, lane_accumulator) in accumulator.iter_mut().enumerate() {
                        let code_start = lane * 4;
                        let mut integer_dot = 0_i32;
                        for code in code_start..code_start + 4 {
                            let weight_code =
                                i32::from(i8::from_ne_bytes([weight_block[2 + code]]));
                            let activation_code =
                                i32::from(i8::from_ne_bytes([activation_block[2 + code]]));
                            integer_dot += weight_code * activation_code;
                        }
                        *lane_accumulator =
                            combined_scale.mul_add(integer_dot as f32, *lane_accumulator);
                    }
                }
            }
            Qwen35QuantType::Nvfp4 => {
                for block in 0..blocks_per_row {
                    let weight_block = row_block_start
                        .checked_add(block)
                        .ok_or(Qwen35Error::SizeOverflow)?;
                    let weight_start = weight_block
                        .checked_mul(NVFP4_BLOCK_BYTES)
                        .ok_or(Qwen35Error::SizeOverflow)?;
                    let weight_block = &bytes[weight_start..weight_start + NVFP4_BLOCK_BYTES];

                    // llama.cpp evaluates sub-blocks 0/1 against the first
                    // activation block and then sub-blocks 2/3 against the
                    // second, issuing one vector FMA for each pair.
                    for pair in 0..2 {
                        let activation_block_index = block
                            .checked_mul(2)
                            .and_then(|value| value.checked_add(pair))
                            .ok_or(Qwen35Error::SizeOverflow)?;
                        let activation_start = activation_block_index
                            .checked_mul(Q8_0_BLOCK_BYTES)
                            .ok_or(Qwen35Error::SizeOverflow)?;
                        let activation_block =
                            &activation[activation_start..activation_start + Q8_0_BLOCK_BYTES];
                        let activation_scale = f16_to_f32(u16::from_le_bytes([
                            activation_block[0],
                            activation_block[1],
                        ]));

                        for sub_block_in_pair in 0..2 {
                            let sub_block = pair * 2 + sub_block_in_pair;
                            let combined_scale = ue4m3_scale_to_f32(weight_block[sub_block])
                                * 0.5
                                * activation_scale;
                            if !combined_scale.is_finite() {
                                return Err(Qwen35Error::invalid(
                                    "q8_0_activation.scale",
                                    "combined NVFP4 and activation scale must be finite",
                                ));
                            }
                            let packed_start =
                                NVFP4_SUB_BLOCKS + sub_block * NVFP4_SUB_BLOCK_PACKED_BYTES;
                            let activation_code_start = sub_block_in_pair * 16;
                            let accumulator_start = sub_block_in_pair * 4;
                            for group in 0..4 {
                                let code_start = group * 4;
                                let mut integer_dot = 0_i32;
                                for code in code_start..code_start + 4 {
                                    let packed_index = code % NVFP4_SUB_BLOCK_PACKED_BYTES;
                                    let packed = weight_block[packed_start + packed_index];
                                    let nibble = if code < NVFP4_SUB_BLOCK_PACKED_BYTES {
                                        packed & 0x0f
                                    } else {
                                        packed >> 4
                                    };
                                    let weight_code = e2m1_doubled_code(nibble);
                                    let activation_code = i32::from(i8::from_ne_bytes([
                                        activation_block[2 + activation_code_start + code],
                                    ]));
                                    integer_dot += weight_code * activation_code;
                                }
                                let lane = accumulator_start + group;
                                accumulator[lane] =
                                    combined_scale.mul_add(integer_dot as f32, accumulator[lane]);
                            }
                        }
                    }
                }
            }
            _ => unreachable!("weight type checked above"),
        }

        // hsum_float_8: first combine the two 128-bit halves, then the high
        // pair, then the remaining adjacent pair.
        let pair_0 = accumulator[4] + accumulator[0];
        let pair_1 = accumulator[5] + accumulator[1];
        let pair_2 = accumulator[6] + accumulator[2];
        let pair_3 = accumulator[7] + accumulator[3];
        let half_0 = pair_0 + pair_2;
        let half_1 = pair_1 + pair_3;
        let sum = half_0 + half_1;
        if !sum.is_finite() {
            return Err(Qwen35Error::invalid(
                "q8_0_activation.output",
                format!("row {row} accumulated a non-finite value"),
            ));
        }
        *destination = sum;
    }
    Ok(output)
}

/// Multiply a BF16 weight matrix by an f32 activation after converting the
/// activation to BF16 with round-to-nearest-even.
///
/// The eight accumulators and reduction tree match the AVX2 tinyBLAS path
/// selected by the local llama.cpp CPU oracle for Qwen's 5120x48 alpha and
/// beta projections. This compatibility path is opt-in and deliberately
/// separate from the native f32-activation reference.
pub fn quantized_matvec_activation_bf16(
    tensor: &GgufTensor,
    bytes: &[u8],
    input: &[f32],
) -> Result<Vec<f32>, Qwen35Error> {
    let (quant_type, input_width, output_width, _) = validate_matvec_inputs(tensor, bytes, input)?;
    if quant_type != Qwen35QuantType::Bf16 {
        return Err(Qwen35Error::invalid(
            "bf16_activation.weight_type",
            format!(
                "expected BF16 ({GGML_TYPE_BF16}) weight, got {}",
                tensor.ggml_type
            ),
        ));
    }
    if !input_width.is_multiple_of(8) || !output_width.is_multiple_of(4) {
        return Err(Qwen35Error::invalid(
            "bf16_activation.geometry",
            "local llama.cpp AVX2 tinyBLAS parity requires K divisible by 8 and N divisible by 4",
        ));
    }

    let activation = input
        .iter()
        .map(|value| f32::from_bits(u32::from(f32_to_bf16(*value)) << 16))
        .collect::<Vec<_>>();
    let mut output = Vec::new();
    output
        .try_reserve_exact(output_width)
        .map_err(|_| Qwen35Error::Gguf("BF16 activation matvec output allocation failed".into()))?;
    output.resize(output_width, 0.0);

    for (row, destination) in output.iter_mut().enumerate() {
        let row_start = row
            .checked_mul(input_width)
            .ok_or(Qwen35Error::SizeOverflow)?;
        let mut accumulator = [0.0_f32; 8];
        for column_start in (0..input_width).step_by(8) {
            for lane in 0..8 {
                let weight = decode_bf16_value(bytes, row_start + column_start + lane)?;
                accumulator[lane] =
                    weight.mul_add(activation[column_start + lane], accumulator[lane]);
            }
        }

        let pair_0 = accumulator[4] + accumulator[0];
        let pair_1 = accumulator[5] + accumulator[1];
        let pair_2 = accumulator[6] + accumulator[2];
        let pair_3 = accumulator[7] + accumulator[3];
        let half_0 = pair_0 + pair_2;
        let half_1 = pair_1 + pair_3;
        let sum = half_0 + half_1;
        if !sum.is_finite() {
            return Err(Qwen35Error::invalid(
                "bf16_activation.output",
                format!("row {row} accumulated a non-finite value"),
            ));
        }
        *destination = sum;
    }
    Ok(output)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn quantized_matvec_fused_avx2_q8(
    tensor: &GgufTensor,
    bytes: &[u8],
    input: &[f32],
) -> Result<Vec<f32>, Qwen35Error> {
    use std::arch::x86_64::{
        __m128i, __m256, _mm256_add_ps, _mm256_castps256_ps128, _mm256_cvtepi32_ps,
        _mm256_cvtepi8_epi32, _mm256_extractf128_ps, _mm256_loadu_ps, _mm256_mul_ps,
        _mm256_setzero_ps, _mm_add_ps, _mm_cvtss_f32, _mm_hadd_ps, _mm_loadl_epi64,
    };

    let (quant_type, _k, n, blocks_per_row) = validate_matvec_inputs(tensor, bytes, input)?;
    debug_assert_eq!(quant_type, Qwen35QuantType::Q8_0);

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn horizontal_sum(value: __m256) -> f32 {
        let low = _mm256_castps256_ps128(value);
        let high = _mm256_extractf128_ps(value, 1);
        let sum128 = _mm_add_ps(low, high);
        let pair_sum = _mm_hadd_ps(sum128, sum128);
        let total = _mm_hadd_ps(pair_sum, pair_sum);
        _mm_cvtss_f32(total)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn block_dot(block: &[u8], input: &[f32]) -> f32 {
        debug_assert_eq!(block.len(), Q8_0_BLOCK_BYTES);
        debug_assert_eq!(input.len(), Q8_0_BLOCK_ELEMENTS);
        let mut accumulator = _mm256_setzero_ps();
        for chunk in 0..4 {
            let codes = _mm_loadl_epi64(block.as_ptr().add(2 + chunk * 8).cast::<__m128i>());
            let codes_i32 = _mm256_cvtepi8_epi32(codes);
            let codes_f32 = _mm256_cvtepi32_ps(codes_i32);
            let activation = _mm256_loadu_ps(input.as_ptr().add(chunk * 8));
            accumulator = _mm256_add_ps(accumulator, _mm256_mul_ps(codes_f32, activation));
        }
        horizontal_sum(accumulator)
    }

    let mut output = Vec::new();
    output
        .try_reserve_exact(n)
        .map_err(|_| Qwen35Error::Gguf("AVX2 quantized matvec output allocation failed".into()))?;
    output.resize(n, 0.0);
    for (row, destination) in output.iter_mut().enumerate() {
        let row_block_start = row
            .checked_mul(blocks_per_row)
            .ok_or(Qwen35Error::SizeOverflow)?;
        let mut sum = 0.0f32;
        for block in 0..blocks_per_row {
            let block_index = row_block_start
                .checked_add(block)
                .ok_or(Qwen35Error::SizeOverflow)?;
            let start = block_index
                .checked_mul(Q8_0_BLOCK_BYTES)
                .ok_or(Qwen35Error::SizeOverflow)?;
            let end = start
                .checked_add(Q8_0_BLOCK_BYTES)
                .ok_or(Qwen35Error::SizeOverflow)?;
            let block_bytes = bytes.get(start..end).ok_or_else(|| {
                Qwen35Error::invalid(
                    "quantized_matvec_fused_simd.payload",
                    "block exceeds payload",
                )
            })?;
            let scale = f16_to_f32(u16::from_le_bytes([block_bytes[0], block_bytes[1]]));
            if !scale.is_finite() {
                return Err(Qwen35Error::invalid(
                    "q8_0.scale",
                    "FP16 scale must be finite",
                ));
            }
            let input_start = block
                .checked_mul(Q8_0_BLOCK_ELEMENTS)
                .ok_or(Qwen35Error::SizeOverflow)?;
            let block_input = &input[input_start..input_start + Q8_0_BLOCK_ELEMENTS];
            let raw_dot = block_dot(block_bytes, block_input);
            if !raw_dot.is_finite() {
                return Err(Qwen35Error::invalid(
                    "quantized_matvec_fused_simd.product",
                    "AVX2 block dot is not finite",
                ));
            }
            let scaled = raw_dot * scale;
            if !scaled.is_finite() {
                return Err(Qwen35Error::invalid(
                    "quantized_matvec_fused_simd.product",
                    "scaled block dot is not finite",
                ));
            }
            sum += scaled;
            if !sum.is_finite() {
                return Err(Qwen35Error::invalid(
                    "quantized_matvec_fused_simd.accumulator",
                    "row accumulator is not finite",
                ));
            }
        }
        *destination = sum;
    }
    Ok(output)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn quantized_matvec_fused_avx2_nvfp4(
    tensor: &GgufTensor,
    bytes: &[u8],
    input: &[f32],
) -> Result<Vec<f32>, Qwen35Error> {
    use std::arch::x86_64::{
        __m128i, __m256, __m256i, _mm256_add_ps, _mm256_and_si256, _mm256_castps256_ps128,
        _mm256_castsi256_ps, _mm256_cvtepu8_epi32, _mm256_extractf128_ps, _mm256_loadu_ps,
        _mm256_mul_ps, _mm256_permutevar8x32_ps, _mm256_set1_epi32, _mm256_set1_ps, _mm256_setr_ps,
        _mm256_setzero_ps, _mm256_slli_epi32, _mm256_srli_epi32, _mm256_xor_ps, _mm_add_ps,
        _mm_cvtss_f32, _mm_hadd_ps, _mm_loadl_epi64,
    };

    let (quant_type, _k, n, blocks_per_row) = validate_matvec_inputs(tensor, bytes, input)?;
    debug_assert_eq!(quant_type, Qwen35QuantType::Nvfp4);

    // The positive E2M1 magnitudes fit in one eight-lane permute table.  The
    // high bit of each nibble is then applied as the IEEE sign bit.  This
    // avoids a memory gather for every packed byte while preserving the
    // exact wire-format lookup used by the scalar decoder.
    let e2m1_magnitudes = _mm256_setr_ps(0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0);
    let magnitude_mask = _mm256_set1_epi32(0x07);
    let sign_nibble_mask = _mm256_set1_epi32(0x08);

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn decode_nibbles(
        codes: __m256i,
        magnitudes: __m256,
        magnitude_mask: __m256i,
        sign_nibble_mask: __m256i,
    ) -> (__m256, __m256) {
        let low_magnitude_codes = _mm256_and_si256(codes, magnitude_mask);
        let high_codes = _mm256_srli_epi32(codes, 4);
        let high_magnitude_codes = _mm256_and_si256(high_codes, magnitude_mask);
        let low_magnitudes = _mm256_permutevar8x32_ps(magnitudes, low_magnitude_codes);
        let high_magnitudes = _mm256_permutevar8x32_ps(magnitudes, high_magnitude_codes);
        let low_sign = _mm256_slli_epi32(_mm256_and_si256(codes, sign_nibble_mask), 28);
        let high_sign = _mm256_slli_epi32(_mm256_and_si256(high_codes, sign_nibble_mask), 28);
        (
            _mm256_xor_ps(low_magnitudes, _mm256_castsi256_ps(low_sign)),
            _mm256_xor_ps(high_magnitudes, _mm256_castsi256_ps(high_sign)),
        )
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn horizontal_sum(value: __m256) -> f32 {
        let low = _mm256_castps256_ps128(value);
        let high = _mm256_extractf128_ps(value, 1);
        let sum128 = _mm_add_ps(low, high);
        let pair_sum = _mm_hadd_ps(sum128, sum128);
        let total = _mm_hadd_ps(pair_sum, pair_sum);
        _mm_cvtss_f32(total)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn sub_block_products(
        packed: __m128i,
        input_low: *const f32,
        input_high: *const f32,
        scale: f32,
        magnitudes: __m256,
        magnitude_mask: __m256i,
        sign_nibble_mask: __m256i,
    ) -> __m256 {
        let codes = _mm256_cvtepu8_epi32(packed);
        let (low_values, high_values) =
            decode_nibbles(codes, magnitudes, magnitude_mask, sign_nibble_mask);
        let low_input = _mm256_loadu_ps(input_low);
        let high_input = _mm256_loadu_ps(input_high);
        let scale = _mm256_set1_ps(scale);
        _mm256_add_ps(
            _mm256_mul_ps(_mm256_mul_ps(low_values, scale), low_input),
            _mm256_mul_ps(_mm256_mul_ps(high_values, scale), high_input),
        )
    }

    let mut output = Vec::new();
    output.try_reserve_exact(n).map_err(|_| {
        Qwen35Error::Gguf("AVX2 NVFP4 quantized matvec output allocation failed".into())
    })?;
    output.resize(n, 0.0);
    for (row, destination) in output.iter_mut().enumerate() {
        let row_block_start = row
            .checked_mul(blocks_per_row)
            .ok_or(Qwen35Error::SizeOverflow)?;
        let mut accumulator = _mm256_setzero_ps();
        for block in 0..blocks_per_row {
            let block_index = row_block_start
                .checked_add(block)
                .ok_or(Qwen35Error::SizeOverflow)?;
            let start = block_index
                .checked_mul(NVFP4_BLOCK_BYTES)
                .ok_or(Qwen35Error::SizeOverflow)?;
            let end = start
                .checked_add(NVFP4_BLOCK_BYTES)
                .ok_or(Qwen35Error::SizeOverflow)?;
            let block_bytes = bytes.get(start..end).ok_or_else(|| {
                Qwen35Error::invalid(
                    "quantized_matvec_fused_simd.payload",
                    "NVFP4 block exceeds payload",
                )
            })?;
            let input_block_start = block
                .checked_mul(NVFP4_BLOCK_ELEMENTS)
                .ok_or(Qwen35Error::SizeOverflow)?;
            for sub_block in 0..NVFP4_SUB_BLOCKS {
                let scale = ue4m3_scale_to_f32(block_bytes[sub_block]);
                if !scale.is_finite() {
                    return Err(Qwen35Error::invalid(
                        "nvfp4.scale",
                        "UE4M3 scale must be finite",
                    ));
                }
                let packed_start = NVFP4_SUB_BLOCKS
                    .checked_add(sub_block * NVFP4_SUB_BLOCK_PACKED_BYTES)
                    .ok_or(Qwen35Error::SizeOverflow)?;
                let packed =
                    _mm_loadl_epi64(block_bytes.as_ptr().add(packed_start).cast::<__m128i>());
                let lane_start = input_block_start
                    .checked_add(sub_block * NVFP4_SUB_BLOCK_ELEMENTS)
                    .ok_or(Qwen35Error::SizeOverflow)?;
                let products = sub_block_products(
                    packed,
                    input.as_ptr().add(lane_start),
                    input
                        .as_ptr()
                        .add(lane_start + NVFP4_SUB_BLOCK_PACKED_BYTES),
                    scale,
                    e2m1_magnitudes,
                    magnitude_mask,
                    sign_nibble_mask,
                );
                accumulator = _mm256_add_ps(accumulator, products);
            }
        }
        let sum = horizontal_sum(accumulator);
        if !sum.is_finite() {
            return Err(Qwen35Error::invalid(
                "quantized_matvec_fused_simd.accumulator",
                "AVX2 NVFP4 row accumulator is not finite",
            ));
        }
        *destination = sum;
    }
    Ok(output)
}

fn validate_matvec_inputs(
    tensor: &GgufTensor,
    bytes: &[u8],
    input: &[f32],
) -> Result<(Qwen35QuantType, usize, usize, usize), Qwen35Error> {
    if tensor.dimensions.len() != 2 {
        return Err(Qwen35Error::invalid(
            "quantized_matvec.tensor.dimensions",
            format!("expected rank 2 [K, N], got {:?}", tensor.dimensions),
        ));
    }
    let (quant_type, _, _) = validate_tensor_payload(tensor, bytes.len())?;
    let k = usize::try_from(tensor.dimensions[0]).map_err(|_| Qwen35Error::SizeOverflow)?;
    let n = usize::try_from(tensor.dimensions[1]).map_err(|_| Qwen35Error::SizeOverflow)?;
    require_exact_len("quantized_matvec input", input.len(), k)?;
    if input.iter().any(|value| !value.is_finite()) {
        return Err(Qwen35Error::invalid(
            "quantized_matvec.input",
            "input values must be finite",
        ));
    }
    Ok((quant_type, k, n, k / quant_type.block_elements()))
}

#[inline]
fn fused_q8_row(
    bytes: &[u8],
    input: &[f32],
    row_block_start: usize,
    blocks_per_row: usize,
) -> Result<f32, Qwen35Error> {
    let mut sum = 0.0f32;
    for block in 0..blocks_per_row {
        let block_index = row_block_start
            .checked_add(block)
            .ok_or(Qwen35Error::SizeOverflow)?;
        let start = block_index
            .checked_mul(Q8_0_BLOCK_BYTES)
            .ok_or(Qwen35Error::SizeOverflow)?;
        let end = start
            .checked_add(Q8_0_BLOCK_BYTES)
            .ok_or(Qwen35Error::SizeOverflow)?;
        let block_bytes = bytes.get(start..end).ok_or_else(|| {
            Qwen35Error::invalid("quantized_matvec_fused.payload", "block exceeds payload")
        })?;
        let scale = f16_to_f32(u16::from_le_bytes([block_bytes[0], block_bytes[1]]));
        if !scale.is_finite() {
            return Err(Qwen35Error::invalid(
                "q8_0.scale",
                "FP16 scale must be finite",
            ));
        }
        let input_start = block
            .checked_mul(Q8_0_BLOCK_ELEMENTS)
            .ok_or(Qwen35Error::SizeOverflow)?;
        let mut block_sum = 0.0f32;
        for lane in 0..Q8_0_BLOCK_ELEMENTS {
            let code = f32::from(i8::from_ne_bytes([block_bytes[2 + lane]]));
            let product = code * input[input_start + lane];
            if !product.is_finite() {
                return Err(Qwen35Error::invalid(
                    "quantized_matvec_fused.product",
                    "product is not finite",
                ));
            }
            block_sum += product;
            if !block_sum.is_finite() {
                return Err(Qwen35Error::invalid(
                    "quantized_matvec_fused.accumulator",
                    "block accumulator is not finite",
                ));
            }
        }
        let scaled_block = block_sum * scale;
        if !scaled_block.is_finite() {
            return Err(Qwen35Error::invalid(
                "quantized_matvec_fused.product",
                "scaled block product is not finite",
            ));
        }
        sum += scaled_block;
        if !sum.is_finite() {
            return Err(Qwen35Error::invalid(
                "quantized_matvec_fused.accumulator",
                "row accumulator is not finite",
            ));
        }
    }
    Ok(sum)
}

#[inline]
fn fused_f32_row(bytes: &[u8], input: &[f32], row: usize, k: usize) -> Result<f32, Qwen35Error> {
    let row_start = row.checked_mul(k).ok_or(Qwen35Error::SizeOverflow)?;
    let mut sum = 0.0f32;
    for (column, input_value) in input.iter().enumerate() {
        let value = decode_f32_value(bytes, row_start + column)?;
        dot_accumulate_fused(&mut sum, value, *input_value)?;
    }
    Ok(sum)
}

#[inline]
fn fused_bf16_row(bytes: &[u8], input: &[f32], row: usize, k: usize) -> Result<f32, Qwen35Error> {
    let row_start = row.checked_mul(k).ok_or(Qwen35Error::SizeOverflow)?;
    let mut sum = 0.0f32;
    for (column, input_value) in input.iter().enumerate() {
        let value = decode_bf16_value(bytes, row_start + column)?;
        dot_accumulate_fused(&mut sum, value, *input_value)?;
    }
    Ok(sum)
}

#[inline]
fn fused_nvfp4_row(
    bytes: &[u8],
    input: &[f32],
    row_block_start: usize,
    blocks_per_row: usize,
) -> Result<f32, Qwen35Error> {
    let mut sum = 0.0f32;
    for block in 0..blocks_per_row {
        let block_index = row_block_start
            .checked_add(block)
            .ok_or(Qwen35Error::SizeOverflow)?;
        let start = block_index
            .checked_mul(NVFP4_BLOCK_BYTES)
            .ok_or(Qwen35Error::SizeOverflow)?;
        let end = start
            .checked_add(NVFP4_BLOCK_BYTES)
            .ok_or(Qwen35Error::SizeOverflow)?;
        let block_bytes = bytes.get(start..end).ok_or_else(|| {
            Qwen35Error::invalid("quantized_matvec_fused.payload", "block exceeds payload")
        })?;
        let input_start = block
            .checked_mul(NVFP4_BLOCK_ELEMENTS)
            .ok_or(Qwen35Error::SizeOverflow)?;
        for sub_block in 0..NVFP4_SUB_BLOCKS {
            let scale = ue4m3_scale_to_f32(block_bytes[sub_block]);
            if !scale.is_finite() {
                return Err(Qwen35Error::invalid(
                    "nvfp4.scale",
                    "UE4M3 scale must be finite",
                ));
            }
            let packed_start = NVFP4_SUB_BLOCKS
                .checked_add(sub_block * NVFP4_SUB_BLOCK_PACKED_BYTES)
                .ok_or(Qwen35Error::SizeOverflow)?;
            let lane_start = input_start
                .checked_add(sub_block * NVFP4_SUB_BLOCK_ELEMENTS)
                .ok_or(Qwen35Error::SizeOverflow)?;
            let mut sub_sum = 0.0f32;
            for index in 0..NVFP4_SUB_BLOCK_PACKED_BYTES {
                let packed = block_bytes[packed_start + index];
                let low_product = e2m1_value(packed & 0x0f) * input[lane_start + index];
                let high_product = e2m1_value(packed >> 4)
                    * input[lane_start + NVFP4_SUB_BLOCK_PACKED_BYTES + index];
                if !low_product.is_finite() || !high_product.is_finite() {
                    return Err(Qwen35Error::invalid(
                        "quantized_matvec_fused.product",
                        "product is not finite",
                    ));
                }
                sub_sum += low_product;
                sub_sum += high_product;
                if !sub_sum.is_finite() {
                    return Err(Qwen35Error::invalid(
                        "quantized_matvec_fused.accumulator",
                        "sub-block accumulator is not finite",
                    ));
                }
            }
            let scaled_sub_sum = sub_sum * scale;
            if !scaled_sub_sum.is_finite() {
                return Err(Qwen35Error::invalid(
                    "quantized_matvec_fused.product",
                    "scaled sub-block product is not finite",
                ));
            }
            sum += scaled_sub_sum;
            if !sum.is_finite() {
                return Err(Qwen35Error::invalid(
                    "quantized_matvec_fused.accumulator",
                    "row accumulator is not finite",
                ));
            }
        }
    }
    Ok(sum)
}

#[inline]
fn dot_accumulate_fused(sum: &mut f32, lhs: f32, rhs: f32) -> Result<(), Qwen35Error> {
    let product = lhs * rhs;
    if !product.is_finite() {
        return Err(Qwen35Error::invalid(
            "quantized_matvec_fused.product",
            "product is not finite",
        ));
    }
    *sum += product;
    if sum.is_finite() {
        Ok(())
    } else {
        Err(Qwen35Error::invalid(
            "quantized_matvec_fused.accumulator",
            "accumulator is not finite",
        ))
    }
}

/// Read and decode one tensor through an identity-bound GGUF reader.
pub fn read_dequantized_tensor(
    reader: &mut GgufTensorReader<'_>,
    name: &str,
) -> Result<Vec<f32>, Qwen35Error> {
    let tensor = reader
        .tensor(name)
        .ok_or_else(|| Qwen35Error::Gguf(format!("unknown tensor {name:?}")))?
        .clone();
    let bytes = reader.read_tensor_bytes(name)?;
    dequantize_tensor(&tensor, &bytes)
}

impl<'a> GgufTensorReader<'a> {
    /// Read one tensor's payload and decode it into CPU f32 values.
    pub fn read_tensor_f32(&mut self, name: &str) -> Result<Vec<f32>, Qwen35Error> {
        read_dequantized_tensor(self, name)
    }
}

impl GgufManifest {
    /// Open an inspected GGUF and decode one tensor into CPU f32 values.
    pub fn read_tensor_f32(
        &self,
        path: impl AsRef<std::path::Path>,
        name: &str,
    ) -> Result<Vec<f32>, Qwen35Error> {
        let mut reader = self.open_tensor_reader(path)?;
        reader.read_tensor_f32(name)
    }
}

fn tensor_element_count(tensor: &GgufTensor) -> Result<usize, Qwen35Error> {
    if tensor.dimensions.is_empty() || tensor.dimensions.len() > 4 {
        return Err(Qwen35Error::invalid(
            "tensor.dimensions",
            "expected one through four nonzero dimensions",
        ));
    }
    tensor
        .dimensions
        .iter()
        .try_fold(1usize, |count, dimension| {
            if *dimension == 0 {
                return Err(Qwen35Error::invalid(
                    "tensor.dimensions",
                    "dimensions must be nonzero",
                ));
            }
            let dimension = usize::try_from(*dimension).map_err(|_| Qwen35Error::SizeOverflow)?;
            count
                .checked_mul(dimension)
                .ok_or(Qwen35Error::SizeOverflow)
        })
}

fn validate_tensor_payload(
    tensor: &GgufTensor,
    payload_len: usize,
) -> Result<(Qwen35QuantType, usize, usize), Qwen35Error> {
    let quant_type = Qwen35QuantType::from_ggml_type(tensor.ggml_type)?;
    let element_count = tensor_element_count(tensor)?;
    if !element_count.is_multiple_of(quant_type.block_elements()) {
        return Err(Qwen35Error::invalid(
            "tensor.element_count",
            format!(
                "element count {element_count} is not divisible by {} for {:?}",
                quant_type.block_elements(),
                quant_type
            ),
        ));
    }
    let expected_bytes = element_count
        .checked_div(quant_type.block_elements())
        .and_then(|blocks| blocks.checked_mul(quant_type.block_bytes()))
        .ok_or(Qwen35Error::SizeOverflow)?;
    let expected_bytes_u64 =
        u64::try_from(expected_bytes).map_err(|_| Qwen35Error::SizeOverflow)?;
    if tensor.byte_len != expected_bytes_u64 {
        return Err(Qwen35Error::invalid(
            "tensor.byte_len",
            format!(
                "expected {expected_bytes} for {:?}, got {}",
                quant_type, tensor.byte_len
            ),
        ));
    }
    require_exact_len("tensor payload", payload_len, expected_bytes)?;

    if quant_type.block_elements() > 1 {
        let row_width = tensor.dimensions.first().copied().ok_or_else(|| {
            Qwen35Error::invalid("tensor.dimensions", "at least one dimension is required")
        })?;
        let row_width = usize::try_from(row_width).map_err(|_| Qwen35Error::SizeOverflow)?;
        if !row_width.is_multiple_of(quant_type.block_elements()) {
            return Err(Qwen35Error::invalid(
                "tensor.dimensions[0]",
                format!(
                    "row width {row_width} is not divisible by {} for {:?}",
                    quant_type.block_elements(),
                    quant_type
                ),
            ));
        }
    }
    Ok((quant_type, element_count, expected_bytes))
}

fn require_exact_len(field: &str, actual: usize, expected: usize) -> Result<(), Qwen35Error> {
    if actual == expected {
        Ok(())
    } else {
        Err(Qwen35Error::invalid(
            field,
            format!("expected {expected} bytes/elements, got {actual}"),
        ))
    }
}

fn decode_f32_value(bytes: &[u8], index: usize) -> Result<f32, Qwen35Error> {
    let offset = index
        .checked_mul(F32_ELEMENT_BYTES)
        .ok_or(Qwen35Error::SizeOverflow)?;
    let end = offset
        .checked_add(F32_ELEMENT_BYTES)
        .ok_or(Qwen35Error::SizeOverflow)?;
    if end > bytes.len() {
        return Err(Qwen35Error::invalid(
            "f32 payload",
            format!("element {index} exceeds payload"),
        ));
    }
    let value = f32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ]);
    if value.is_finite() {
        Ok(value)
    } else {
        Err(Qwen35Error::invalid(
            format!("f32.value[{index}]"),
            "decoded value must be finite",
        ))
    }
}

fn decode_bf16_value(bytes: &[u8], index: usize) -> Result<f32, Qwen35Error> {
    let offset = index
        .checked_mul(BF16_ELEMENT_BYTES)
        .ok_or(Qwen35Error::SizeOverflow)?;
    let end = offset
        .checked_add(BF16_ELEMENT_BYTES)
        .ok_or(Qwen35Error::SizeOverflow)?;
    if end > bytes.len() {
        return Err(Qwen35Error::invalid(
            "bf16 payload",
            format!("element {index} exceeds payload"),
        ));
    }
    let bits = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
    let value = f32::from_bits(u32::from(bits) << 16);
    if value.is_finite() {
        Ok(value)
    } else {
        Err(Qwen35Error::invalid(
            format!("bf16.value[{index}]"),
            "decoded value must be finite",
        ))
    }
}

fn dot_accumulate(sum: &mut f32, lhs: f32, rhs: f32) -> Result<(), Qwen35Error> {
    let product = lhs * rhs;
    if !product.is_finite() {
        return Err(Qwen35Error::invalid(
            "quantized_matvec.product",
            "product is not finite",
        ));
    }
    *sum += product;
    if sum.is_finite() {
        Ok(())
    } else {
        Err(Qwen35Error::invalid(
            "quantized_matvec.accumulator",
            "accumulator is not finite",
        ))
    }
}

fn e2m1_value(nibble: u8) -> f32 {
    // The table is the ordinary E2M1 representation.  It is equivalent to
    // llama.cpp's doubled integer table multiplied by its half-scale helper.
    const VALUES: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    VALUES[(nibble & 0x0f) as usize]
}

fn e2m1_doubled_code(nibble: u8) -> i32 {
    const VALUES: [i32; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];
    VALUES[(nibble & 0x0f) as usize]
}

fn ue4m3_scale_to_f32(value: u8) -> f32 {
    // This follows ggml_ue4m3_to_fp32 before its final * 0.5 convention.
    if value == 0 || value == 0x7f {
        return 0.0;
    }
    let exponent = (value >> 3) & 0x0f;
    let mantissa = value & 0x07;
    if exponent == 0 {
        f32::from(mantissa) * 2.0_f32.powi(-9)
    } else {
        (1.0 + f32::from(mantissa) / 8.0) * 2.0_f32.powi(i32::from(exponent) - 7)
    }
}

pub(crate) fn f32_to_f16(value: f32) -> u16 {
    // Portable round-to-nearest-even conversion used by ggml when a native
    // F16C conversion is unavailable. It is bit-equivalent for finite values
    // and keeps this diagnostic ABI deterministic on non-x86 hosts.
    let scale_to_inf = f32::from_bits(0x7780_0000); // 2^112
    let scale_to_zero = f32::from_bits(0x0880_0000); // 2^-110
    let mut base = (value.abs() * scale_to_inf) * scale_to_zero;

    let bits = value.to_bits();
    let shifted = bits.wrapping_add(bits);
    let sign = bits & 0x8000_0000;
    let mut bias = shifted & 0xff00_0000;
    if bias < 0x7100_0000 {
        bias = 0x7100_0000;
    }
    base += f32::from_bits((bias >> 1).wrapping_add(0x0780_0000));
    let base_bits = base.to_bits();
    let exponent = (base_bits >> 13) & 0x0000_7c00;
    let mantissa = base_bits & 0x0000_0fff;
    let nonsign = exponent + mantissa;
    ((sign >> 16)
        | if shifted > 0xff00_0000 {
            0x0000_7e00
        } else {
            nonsign
        }) as u16
}

fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16
}

pub(crate) fn f16_to_f32(bits: u16) -> f32 {
    let sign = (u32::from(bits & 0x8000)) << 16;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let mantissa = u32::from(bits & 0x03ff);
    let result_bits = if exponent == 0 {
        if mantissa == 0 {
            sign
        } else {
            // Normalize a half subnormal before shifting it into the f32
            // exponent/mantissa fields.
            let mut mantissa = mantissa;
            let mut exponent = 0i32;
            while mantissa & 0x0400 == 0 {
                mantissa <<= 1;
                exponent -= 1;
            }
            mantissa &= 0x03ff;
            let exponent = u32::try_from(127 - 15 + 1 + exponent).unwrap_or(0);
            sign | (exponent << 23) | (mantissa << 13)
        }
    } else if exponent == 0x1f {
        sign | 0x7f80_0000 | (mantissa << 13)
    } else {
        sign | ((exponent + 112) << 23) | (mantissa << 13)
    };
    f32::from_bits(result_bits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GgufTensor;

    fn tensor(dimensions: &[u64], ggml_type: u32) -> GgufTensor {
        let elements = dimensions.iter().product::<u64>();
        let (block_elements, block_bytes): (u64, u64) = match ggml_type {
            GGML_TYPE_F32 => (1, F32_ELEMENT_BYTES as u64),
            GGML_TYPE_Q8_0 => (32, 34),
            GGML_TYPE_BF16 => (1, 2),
            GGML_TYPE_NVFP4 => (64, 36),
            _ => (1, 64),
        };
        GgufTensor {
            name: "fixture".into(),
            dimensions: dimensions.to_vec(),
            ggml_type,
            relative_offset: 0,
            absolute_offset: 0,
            byte_len: (elements / block_elements) * block_bytes,
        }
    }

    #[test]
    fn q8_0_matches_little_endian_half_scale_and_signed_codes() {
        let mut bytes = vec![0_u8; Q8_0_BLOCK_BYTES];
        bytes[0..2].copy_from_slice(&0x3800_u16.to_le_bytes()); // 0.5
        bytes[2] = 0x80; // -128, exercises signed conversion
        bytes[3] = 0x01;
        bytes[2 + 31] = 0x7f;
        let mut output = [0.0; Q8_0_BLOCK_ELEMENTS];
        dequantize_q8_0_block(&bytes, &mut output).unwrap();
        assert_eq!(output[0], -64.0);
        assert_eq!(output[1], 0.5);
        assert_eq!(output[31], 63.5);
    }

    #[test]
    fn q8_activation_quantization_matches_llama_rounding_contract() {
        let mut input = [0.0_f32; Q8_0_BLOCK_ELEMENTS];
        input[0] = 127.0;
        input[1] = 2.5;
        input[2] = 3.5;
        input[3] = -2.5;
        input[4] = -3.5;

        let payload = quantize_q8_0_activation(&input).unwrap();
        assert_eq!(payload.len(), Q8_0_BLOCK_BYTES);
        assert_eq!(&payload[..2], &0x3c00_u16.to_le_bytes());
        assert_eq!(i8::from_ne_bytes([payload[2]]), 127);
        assert_eq!(i8::from_ne_bytes([payload[3]]), 2);
        assert_eq!(i8::from_ne_bytes([payload[4]]), 4);
        assert_eq!(i8::from_ne_bytes([payload[5]]), -2);
        assert_eq!(i8::from_ne_bytes([payload[6]]), -4);
        assert!(payload[7..].iter().all(|code| *code == 0));
    }

    #[test]
    fn q8_activation_matvec_keeps_compatibility_abi_separate() {
        let descriptor = tensor(&[32, 1], GGML_TYPE_Q8_0);
        let mut weights = vec![0_u8; Q8_0_BLOCK_BYTES];
        weights[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        weights[2..].fill(1);
        let mut input = [0.0_f32; Q8_0_BLOCK_ELEMENTS];
        input[0] = 127.0;
        input[1] = 2.5;

        let compatible = quantized_matvec_activation_q8_0(&descriptor, &weights, &input).unwrap();
        let native = quantized_matvec(&descriptor, &weights, &input).unwrap();
        assert_eq!(compatible, vec![129.0]);
        assert_eq!(native, vec![129.5]);
    }

    #[test]
    fn nvfp4_matvec_uses_the_same_activation_q8_boundary() {
        let descriptor = tensor(&[64, 1], GGML_TYPE_NVFP4);
        let mut weights = vec![0_u8; NVFP4_BLOCK_BYTES];
        weights[..NVFP4_SUB_BLOCKS].fill(0x38); // raw UE4M3 scale 1.0
        weights[NVFP4_SUB_BLOCKS..].fill(0x22); // E2M1 value 1.0 in both nibbles
        let mut input = [0.0_f32; NVFP4_BLOCK_ELEMENTS];
        input[0] = 127.0;
        input[1] = 2.5;

        let compatible = quantized_matvec_activation_q8_0(&descriptor, &weights, &input).unwrap();
        let native = quantized_matvec(&descriptor, &weights, &input).unwrap();
        assert_eq!(compatible, vec![129.0]);
        assert_eq!(native, vec![129.5]);
    }

    #[test]
    fn bf16_matvec_rounds_activation_and_uses_eight_lane_reduction() {
        let descriptor = tensor(&[8, 4], GGML_TYPE_BF16);
        let one = 0x3f80_u16.to_le_bytes();
        let mut weights = Vec::with_capacity(8 * 4 * BF16_ELEMENT_BYTES);
        for _ in 0..8 * 4 {
            weights.extend_from_slice(&one);
        }
        let input = [1.001_f32; 8];

        let compatible = quantized_matvec_activation_bf16(&descriptor, &weights, &input).unwrap();
        let native = quantized_matvec(&descriptor, &weights, &input).unwrap();
        assert_eq!(compatible, vec![8.0; 4]);
        assert!(native.iter().all(|value| *value > 8.0));
    }

    #[test]
    fn q8_0_rejects_nonfinite_scale_and_wrong_lengths() {
        let mut bytes = vec![0_u8; Q8_0_BLOCK_BYTES];
        bytes[0..2].copy_from_slice(&0x7c00_u16.to_le_bytes()); // +inf
        assert!(dequantize_q8_0_block(&bytes, &mut [0.0; Q8_0_BLOCK_ELEMENTS]).is_err());
        assert!(dequantize_q8_0_block(&bytes[..Q8_0_BLOCK_BYTES - 1], &mut [0.0; 32]).is_err());
        assert!(dequantize_q8_0_block(&bytes, &mut [0.0; 31]).is_err());
    }

    #[test]
    fn bf16_is_little_endian_and_rejects_nonfinite_values() {
        let bytes = [0x80, 0x3f, 0x00, 0xc0]; // 1.0, -2.0
        let mut output = [0.0; 2];
        dequantize_bf16(&bytes, &mut output).unwrap();
        assert_eq!(output, [1.0, -2.0]);

        let infinity = [0x80, 0x7f];
        assert!(dequantize_bf16(&infinity, &mut [0.0]).is_err());
    }

    #[test]
    fn f32_is_little_endian_and_rejects_nonfinite_values() {
        let bytes = [0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0xc0];
        let mut output = [0.0; 2];
        dequantize_f32(&bytes, &mut output).unwrap();
        assert_eq!(output, [1.0, -2.0]);

        let infinity = [0x00, 0x00, 0x80, 0x7f];
        assert!(dequantize_f32(&infinity, &mut [0.0]).is_err());
    }

    #[test]
    fn nvfp4_preserves_subblock_and_low_high_nibble_order() {
        let mut bytes = vec![0_u8; NVFP4_BLOCK_BYTES];
        bytes[0] = 0x38; // raw scale 1.0
        bytes[1] = 0x40; // raw scale 2.0
        bytes[2] = 0x30; // raw scale 0.5
        bytes[3] = 0x7e; // raw scale 448.0
        bytes[4] = 0x71; // low=1 (0.5), high=7 (6.0)
        bytes[12] = 0x71; // same nibbles in sub-block 1
        bytes[20] = 0x71; // same nibbles in sub-block 2
        bytes[28] = 0x71; // same nibbles in sub-block 3
        let mut output = [0.0; NVFP4_BLOCK_ELEMENTS];
        dequantize_nvfp4_block(&bytes, &mut output).unwrap();
        assert_eq!(output[0], 0.5);
        assert_eq!(output[8], 6.0);
        assert_eq!(output[16], 1.0); // sub-block 1 scale is 2
        assert_eq!(output[24], 12.0);
        assert_eq!(output[32], 0.25); // sub-block 2 scale is 0.5
        assert_eq!(output[56], 2688.0); // sub-block 3 scale is 448, high nibble
    }

    #[test]
    fn nvfp4_zero_scale_encoding_is_finite() {
        let bytes = vec![0x7f; NVFP4_BLOCK_BYTES];
        let mut output = [1.0; NVFP4_BLOCK_ELEMENTS];
        dequantize_nvfp4_block(&bytes, &mut output).unwrap();
        assert!(output.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn tensor_decoder_enforces_row_geometry_and_payload_size() {
        let valid = tensor(&[64, 2], GGML_TYPE_NVFP4);
        assert_eq!(dequantize_tensor(&valid, &[0; 72]).unwrap().len(), 128);

        let bad_row = tensor(&[48, 2], GGML_TYPE_NVFP4);
        assert!(dequantize_tensor(&bad_row, &[0; 72]).is_err());

        let bad_payload = tensor(&[64], GGML_TYPE_Q8_0);
        assert!(dequantize_tensor(&bad_payload, &[0; Q8_0_BLOCK_BYTES - 1]).is_err());

        let mut bad_descriptor = tensor(&[64], GGML_TYPE_Q8_0);
        bad_descriptor.byte_len += 1;
        assert!(dequantize_tensor(&bad_descriptor, &[0; Q8_0_BLOCK_BYTES * 2]).is_err());

        let unsupported = tensor(&[32], 1);
        assert!(dequantize_tensor(&unsupported, &[0; 64]).is_err());
    }

    #[test]
    fn quantized_matvec_matches_materialized_reference_for_all_types() {
        let q8_descriptor = tensor(&[32, 2], GGML_TYPE_Q8_0);
        let mut q8_bytes = vec![0_u8; Q8_0_BLOCK_BYTES * 2];
        q8_bytes[0..2].copy_from_slice(&0x3c00_u16.to_le_bytes()); // 1.0
        q8_bytes[2] = 2;
        q8_bytes[2 + 1] = 3;
        q8_bytes[Q8_0_BLOCK_BYTES..Q8_0_BLOCK_BYTES + 2].copy_from_slice(&0x3800_u16.to_le_bytes()); // 0.5
        q8_bytes[Q8_0_BLOCK_BYTES + 2] = 4;
        q8_bytes[Q8_0_BLOCK_BYTES + 3] = 5;
        let q8_input: Vec<f32> = (0..32).map(|value| value as f32 / 8.0).collect();
        assert_matvec_matches_decode(&q8_descriptor, &q8_bytes, &q8_input);

        let nv_descriptor = tensor(&[64, 2], GGML_TYPE_NVFP4);
        let mut nv_bytes = vec![0_u8; NVFP4_BLOCK_BYTES * 2];
        nv_bytes[0] = 0x38;
        nv_bytes[4] = 0x21;
        nv_bytes[NVFP4_BLOCK_BYTES] = 0x40;
        nv_bytes[NVFP4_BLOCK_BYTES + 4] = 0x43;
        let nv_input: Vec<f32> = (0..64).map(|value| value as f32 / 16.0).collect();
        assert_matvec_matches_decode(&nv_descriptor, &nv_bytes, &nv_input);

        let bf_descriptor = tensor(&[3, 2], GGML_TYPE_BF16);
        let bf_values = [1.0_f32, -2.0, 0.5, 3.0, 4.0, -1.0];
        let mut bf_bytes = Vec::with_capacity(bf_values.len() * 2);
        for value in bf_values {
            let bits = value.to_bits();
            bf_bytes.extend_from_slice(&((bits >> 16) as u16).to_le_bytes());
        }
        let bf_input = [0.25, 2.0, -1.0];
        assert_matvec_matches_decode(&bf_descriptor, &bf_bytes, &bf_input);

        let f32_descriptor = tensor(&[3, 2], GGML_TYPE_F32);
        let f32_values = [1.0_f32, -2.0, 0.5, 3.0, 4.0, -1.0];
        let mut f32_bytes = Vec::with_capacity(f32_values.len() * F32_ELEMENT_BYTES);
        for value in f32_values {
            f32_bytes.extend_from_slice(&value.to_le_bytes());
        }
        let f32_input = [0.25, 2.0, -1.0];
        assert_matvec_matches_decode(&f32_descriptor, &f32_bytes, &f32_input);
    }

    #[test]
    fn avx2_nvfp4_matvec_matches_checked_fused_reference_across_blocks() {
        #[cfg(not(target_arch = "x86_64"))]
        return;
        #[cfg(target_arch = "x86_64")]
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }

        let descriptor = tensor(&[128, 5], GGML_TYPE_NVFP4);
        let mut bytes = vec![0_u8; NVFP4_BLOCK_BYTES * 2 * 5];
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte = match index % NVFP4_BLOCK_BYTES {
                0 => 0x30,
                1 => 0x38,
                2 => 0x40,
                3 => 0x48,
                packed => (packed as u8)
                    .wrapping_mul(37)
                    .wrapping_add((index / NVFP4_BLOCK_BYTES) as u8),
            };
        }
        let input: Vec<f32> = (0..128)
            .map(|index| ((index % 29) as f32 - 14.0) / 7.0)
            .collect();
        let expected = quantized_matvec_fused(&descriptor, &bytes, &input).unwrap();
        let actual = quantized_matvec_fused_simd(&descriptor, &bytes, &input).unwrap();
        assert_eq!(actual.len(), expected.len());
        for (row, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            let tolerance = 1.0e-4 + 1.0e-4 * actual.abs().max(expected.abs());
            assert!(
                (actual - expected).abs() <= tolerance,
                "row {row}: SIMD {actual} differs from fused reference {expected}"
            );
        }
    }

    #[test]
    fn quantized_matvec_rejects_rank_input_and_nonfinite_contract_violations() {
        let descriptor = tensor(&[32, 1], GGML_TYPE_Q8_0);
        let bytes = vec![0_u8; Q8_0_BLOCK_BYTES];
        assert!(quantized_matvec(&descriptor, &bytes, &[0.0; 31]).is_err());
        assert!(quantized_matvec(&descriptor, &bytes, &[f32::NAN; 32]).is_err());

        let rank_one = tensor(&[32], GGML_TYPE_Q8_0);
        assert!(quantized_matvec(&rank_one, &bytes, &[0.0; 32]).is_err());
    }

    fn assert_matvec_matches_decode(tensor: &GgufTensor, bytes: &[u8], input: &[f32]) {
        let decoded = dequantize_tensor(tensor, bytes).unwrap();
        let expected: Vec<f32> = decoded
            .chunks_exact(tensor.dimensions[0] as usize)
            .map(|row| {
                row.iter()
                    .zip(input)
                    .map(|(weight, value)| weight * value)
                    .sum()
            })
            .collect();
        let actual = quantized_matvec(tensor, bytes, input).unwrap();
        let fused = quantized_matvec_fused(tensor, bytes, input).unwrap();
        assert_eq!(actual.len(), expected.len());
        assert_eq!(fused.len(), expected.len());
        for ((actual, fused), expected) in actual.iter().zip(&fused).zip(expected) {
            assert!(
                (actual - expected).abs() <= 1.0e-5,
                "{actual} != {expected}"
            );
            assert!(
                (fused - actual).abs() <= 1.0e-4,
                "fused {fused} != scalar {actual}"
            );
        }
    }

    #[test]
    fn fused_matvec_rejects_nonfinite_q8_scale() {
        let descriptor = tensor(&[32, 1], GGML_TYPE_Q8_0);
        let mut bytes = vec![0_u8; Q8_0_BLOCK_BYTES];
        bytes[..2].copy_from_slice(&0x7c00_u16.to_le_bytes());
        assert!(quantized_matvec_fused(&descriptor, &bytes, &[1.0; 32]).is_err());
    }

    #[test]
    fn fused_matvec_rejects_nonfinite_input_before_reading_payload() {
        let descriptor = tensor(&[64, 1], GGML_TYPE_NVFP4);
        let bytes = vec![0_u8; NVFP4_BLOCK_BYTES];
        assert!(quantized_matvec_fused(&descriptor, &bytes, &[f32::INFINITY; 64]).is_err());
    }

    #[test]
    fn bf16_tensor_has_no_artificial_block_alignment_requirement() {
        let descriptor = tensor(&[3, 2], GGML_TYPE_BF16);
        let bytes = vec![
            0x00, 0x3f, 0x00, 0x3f, 0x00, 0x3f, 0x00, 0x3f, 0x00, 0x3f, 0x00, 0x3f,
        ];
        let output = dequantize_tensor(&descriptor, &bytes).unwrap();
        assert_eq!(output, vec![0.5; 6]);
    }

    #[test]
    fn fp16_conversion_covers_zero_subnormal_normal_and_special() {
        assert_eq!(f32_to_f16(0.0), 0);
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(-2.0), 0xc000);
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7c00);
        assert_eq!(f32_to_f16(f32::NAN) & 0x7fff, 0x7e00);
        assert_eq!(f16_to_f32(0), 0.0);
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0x0001).to_bits(), 0x3380_0000);
        assert!(f16_to_f32(0x7c00).is_infinite());
        assert!(f16_to_f32(0x7e00).is_nan());
    }

    #[test]
    fn type_geometry_is_explicit() {
        assert_eq!(Qwen35QuantType::F32.block_bytes(), 4);
        assert_eq!(Qwen35QuantType::Q8_0.block_bytes(), 34);
        assert_eq!(Qwen35QuantType::Nvfp4.block_elements(), 64);
        assert_eq!(Qwen35QuantType::Bf16.ggml_type(), 30);
    }
}
