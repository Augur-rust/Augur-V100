#![cfg(feature = "qwen35-prefill-cublas")]

//! Stream-owned batched Qwen projection through cuBLAS on SM70.
//!
//! Quantized weights and Q8_1 activations remain in their resident wire
//! layouts. Default submissions decode them into two plan-owned FP16 workspaces
//! and emits a row-major F32 token batch. The workspaces and cuBLAS handle are
//! reused across projections; this component performs no allocation at submit.
//! Explicit small-batch quantized projections bypass materialization and cuBLAS.

mod dflash;
mod dflash_head;
mod selector_gate;
pub use dflash_head::{CudaDflashBf16HeadKernel,CudaDflashBf16HeadV1};
mod planar_head;
pub use planar_head::{CudaQwen35PlanarQ8HeadProjectionV1, CudaQwen35PlanarQ8HeadWeightV1};
mod row_input;
mod qpn;
mod augur;
pub use augur::{CudaQwen35AugurProjectionV1,CudaQwen35AugurNormGroupB8V1,NormGroupMemberV1,AUGUR_PREFILL_PROFILE,augur_prefill_widths,augur_prefill_widths_for_batch,AUGUR_MAX_BATCH,AUGUR_DEFAULT_BATCH};
pub use qpn::{CudaQwen35QpnPrefillProjectionV1, CudaQwen35QpnRetainedProjectionV1,
    CudaQwen35QpnFfnWorkspaceV1, CudaQwen35QpnFfnProgramV1, CudaQwen35QpnFfnGraphV1,
    CudaQwen35QpnScalePlacement, CudaQwen35NormQpnGroupV1};
#[cfg(test)]
mod submission_cleanup_tests;
#[cfg(test)]
mod replay_fold_tests;
pub use row_input::{CudaQwen35RowQ8InputProjectionV1, CudaQwen35RowQ8InputWeightV1};




#[cfg(feature = "qwen35-kv8-batched-attention")]
mod canonical_attention;
#[cfg(feature = "qwen35-kv8-batched-attention")]
pub use canonical_attention::{CudaQwen35CanonicalPrefillV1, Qwen35CanonicalPrefillScorePolicy};

use crate::{
    cuda_device_properties, CudaBf16WeightViewV1, CudaBufferView, CudaComputeCapability,
    CudaF32View, CudaNvfp4Sm70WeightViewV1, CudaQ8_0WeightViewV1, CudaQ8_1ActivationViewV1,
    CudaQ8_1ActivationQuantizeArgsV1,
    CudaRuntimeError, CudaRuntimeErrorKind, OwnedCudaBuffer, OwnedCudaEvent, OwnedCudaStream,
};
#[cfg(all(
    feature = "qwen35-prefill-cublas",
    feature = "qwen35-kv8-full-attention"
))]
use augur_cuda_sys::AugurCudaQwen35PrefillKv8ChunkAttentionArgsV1;
#[cfg(test)]
use augur_cuda_sys::AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_MAX_TOKENS_V1;
use augur_cuda_sys::{
    self, AugurCudaQwen35PrefillCublasPlanArgsV1, AugurCudaQwen35PrefillCublasPlanV1,
    AugurCudaQwen35PrefillCublasProjectionArgsV1, AugurCudaQwen35PrefillFullAttentionArgsV1,
    AugurCudaQwen35PrefillF32ProjectionArgsV1,
    AugurCudaQwen35PrefillLinearChunkArgsV1, AugurCudaQwen35PrefillRmsnormArgsV1,
    AUGUR_CUDA_QWEN35_FULL_ATTENTION_HEAD_DIM_V1, AUGUR_CUDA_QWEN35_FULL_ATTENTION_KV_ELEMENTS_V1,
    AUGUR_CUDA_QWEN35_FULL_ATTENTION_KV_HEADS_V1, AUGUR_CUDA_QWEN35_FULL_ATTENTION_MAX_CAPACITY_V1,
    AUGUR_CUDA_QWEN35_FULL_ATTENTION_QUERY_ELEMENTS_V1, AUGUR_CUDA_QWEN35_FULL_ATTENTION_QUERY_HEADS_V1,
    AUGUR_CUDA_QWEN35_FULL_ATTENTION_ROTARY_DIM_V1,
    AUGUR_CUDA_QWEN35_LINEAR_STEP_CONV_HISTORY_ELEMENTS_V1,
    AUGUR_CUDA_QWEN35_LINEAR_STEP_CONV_WEIGHT_ELEMENTS_V1, AUGUR_CUDA_QWEN35_LINEAR_STEP_HEAD_DIM_V1,
    AUGUR_CUDA_QWEN35_LINEAR_STEP_OUTPUT_ELEMENTS_V1, AUGUR_CUDA_QWEN35_LINEAR_STEP_QKV_ELEMENTS_V1,
    AUGUR_CUDA_QWEN35_LINEAR_STEP_STATE_ELEMENTS_V1, AUGUR_CUDA_QWEN35_LINEAR_STEP_VALUE_HEADS_V1,
    AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_COMPILED_MAX_TOKENS_V1,
    AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_BF16_V1, AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_NVFP4_V1,
    AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1,
};

#[cfg(feature = "qwen35-kv8-full-attention")]
use augur_cuda_sys::{
    AugurCudaQwen35PrefillKv8CacheQuantizeArgsV1, AUGUR_CUDA_QWEN35_KV8_CACHE_ROW_BYTES_V3,
    AUGUR_CUDA_QWEN35_KV8_FULL_ATTENTION_MAX_CAPACITY_V3,
};

const OPERATION: &str = "cuda_qwen35_prefill_cublas_v1";
#[cfg(all(
    feature = "qwen35-prefill-cublas",
    feature = "qwen35-kv8-full-attention"
))]
const KV8_CHUNK_OPERATION: &str = "cuda_qwen35_prefill_kv8_chunk_attention_v1";

const F16_BYTES: usize = 2;

pub const QWEN35_COMPARE_SUMMARY_BYTES: usize = 64;

/// Diagnostic comparison only. Format 0 is F32; format 1 is decoded Q8_0.
pub fn submit_qwen35_compare_summary(
    stream: &OwnedCudaStream,
    candidate: CudaBufferView,
    reference: CudaBufferView,
    output: CudaBufferView,
    format: u32,
) -> Result<(), CudaRuntimeError> {
    if format > 1
        || candidate.len_bytes() == 0
        || candidate.len_bytes() != reference.len_bytes()
        || candidate.len_bytes() % (if format == 0 { 4 } else { 34 }) != 0
        || output.len_bytes() != QWEN35_COMPARE_SUMMARY_BYTES
        || candidate.overlaps(output, OPERATION)?
        || reference.overlaps(output, OPERATION)?
    {
        return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
    }
    if [candidate, reference, output]
        .iter()
        .any(|v| v.device_ordinal() != stream.device_ordinal())
    {
        return Err(CudaRuntimeError::device_mismatch());
    }
    // SAFETY: exact nonempty extents/owners above, native validation repeats
    // alignment and device checks; caller retains owners to stream completion.
    let result = unsafe {
        augur_cuda_sys::augur_cuda_qwen35_compare_summary_v1(
            stream.device_ordinal(),
            stream.raw()?,
            candidate.checked_address(OPERATION)?,
            reference.checked_address(OPERATION)?,
            candidate.len_bytes() as u64,
            output.checked_address(OPERATION)?,
            output.len_bytes() as u64,
            format,
        )
    };
    CudaRuntimeError::from_result("qwen35_compare_summary", result)
}





pub const QWEN35_PREFILL_LINEAR_KEY_HEADS: usize = 16;
pub const QWEN35_PREFILL_LINEAR_VALUE_HEADS: usize = AUGUR_CUDA_QWEN35_LINEAR_STEP_VALUE_HEADS_V1;
pub const QWEN35_PREFILL_LINEAR_HEAD_DIM: usize = AUGUR_CUDA_QWEN35_LINEAR_STEP_HEAD_DIM_V1;
pub const QWEN35_PREFILL_LINEAR_QKV_ELEMENTS: usize = AUGUR_CUDA_QWEN35_LINEAR_STEP_QKV_ELEMENTS_V1;
pub const QWEN35_PREFILL_LINEAR_OUTPUT_ELEMENTS: usize =
    AUGUR_CUDA_QWEN35_LINEAR_STEP_OUTPUT_ELEMENTS_V1;
pub const QWEN35_PREFILL_LINEAR_STATE_ELEMENTS: usize =
    AUGUR_CUDA_QWEN35_LINEAR_STEP_STATE_ELEMENTS_V1;
pub const QWEN35_PREFILL_LINEAR_HISTORY_ELEMENTS: usize =
    AUGUR_CUDA_QWEN35_LINEAR_STEP_CONV_HISTORY_ELEMENTS_V1;
pub const QWEN35_PREFILL_LINEAR_CONV_WEIGHT_ELEMENTS: usize =
    AUGUR_CUDA_QWEN35_LINEAR_STEP_CONV_WEIGHT_ELEMENTS_V1;
pub const QWEN35_PREFILL_FULL_QUERY_HEADS: usize = AUGUR_CUDA_QWEN35_FULL_ATTENTION_QUERY_HEADS_V1;
pub const QWEN35_PREFILL_FULL_KV_HEADS: usize = AUGUR_CUDA_QWEN35_FULL_ATTENTION_KV_HEADS_V1;
pub const QWEN35_PREFILL_FULL_HEAD_DIM: usize = AUGUR_CUDA_QWEN35_FULL_ATTENTION_HEAD_DIM_V1;
pub const QWEN35_PREFILL_FULL_ROTARY_DIM: usize = AUGUR_CUDA_QWEN35_FULL_ATTENTION_ROTARY_DIM_V1;
pub const QWEN35_PREFILL_FULL_QUERY_ELEMENTS: usize =
    AUGUR_CUDA_QWEN35_FULL_ATTENTION_QUERY_ELEMENTS_V1;
pub const QWEN35_PREFILL_FULL_KV_ELEMENTS: usize = AUGUR_CUDA_QWEN35_FULL_ATTENTION_KV_ELEMENTS_V1;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CudaQwen35PrefillWeightV1 {
    Q8_0(CudaQ8_0WeightViewV1),
    RowQ8Input(CudaQwen35RowQ8InputWeightV1),
    Bf16(CudaBf16WeightViewV1),
    Nvfp4 {
        view: CudaNvfp4Sm70WeightViewV1,
        tensor_scale: f32,
    },
}

impl CudaQwen35PrefillWeightV1 {
    fn view(self) -> CudaBufferView {
        match self {
            Self::Q8_0(view) => view.buffer_view(),
            Self::RowQ8Input(view) => view.view,
            Self::Bf16(view) => view.buffer_view(),
            Self::Nvfp4 { view, .. } => view.buffer_view(),
        }
    }

    fn rows(self) -> usize {
        match self {
            Self::Q8_0(view) => view.rows() as usize,
            Self::RowQ8Input(view) => view.rows,
            Self::Bf16(view) => view.rows() as usize,
            Self::Nvfp4 { view, .. } => view.rows() as usize,
        }
    }

    fn cols(self) -> usize {
        match self {
            Self::Q8_0(view) => view.cols() as usize,
            Self::RowQ8Input(_) => 5120,
            Self::Bf16(view) => view.cols() as usize,
            Self::Nvfp4 { view, .. } => view.cols() as usize,
        }
    }

    fn format(self) -> u32 {
        match self {
            Self::Q8_0(_) => AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1,
            Self::RowQ8Input(_) => row_input::FORMAT,
            Self::Bf16(_) => AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_BF16_V1,
            Self::Nvfp4 { .. } => AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_NVFP4_V1,
        }
    }

    fn tensor_scale(self) -> f32 {
        match self {
            Self::Q8_0(_) | Self::RowQ8Input(_) | Self::Bf16(_) => 1.0,
            Self::Nvfp4 { tensor_scale, .. } => tensor_scale,
        }
    }
}

pub struct CudaQwen35PrefillCublasPlanV1 {
    raw: *mut AugurCudaQwen35PrefillCublasPlanV1,
    decoded_weight_f16: OwnedCudaBuffer,
    decoded_input_f16: OwnedCudaBuffer,
    device_ordinal: i32,
    stream_owner_instance_key: usize,
    max_tokens: usize,
    paired_ffn: bool,
    paired_ffn_submissions: core::cell::Cell<u64>,
    skip_unused_q8: bool,
    full_ffn_input_reuse: bool,
    gdn_bf16_pair: bool,
    gdn_bf16_pair_submissions: core::cell::Cell<u64>,
    q8_input_pack_elisions: core::cell::Cell<u64>,
    direct_b8_only: bool,
}

impl core::fmt::Debug for CudaQwen35PrefillCublasPlanV1 {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("CudaQwen35PrefillCublasPlanV1")
            .field("device_ordinal", &self.device_ordinal)
            .field("max_tokens", &self.max_tokens)
            .field("decoded_weight_bytes", &self.decoded_weight_f16.len_bytes())
            .field("decoded_input_bytes", &self.decoded_input_f16.len_bytes())
            .field("plan_bound", &!self.raw.is_null())
            .finish()
    }
}

impl CudaQwen35PrefillCublasPlanV1 {


    pub const fn compiled_max_tokens() -> usize {
        AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_COMPILED_MAX_TOKENS_V1 as usize
    }

    pub fn new(
        stream: &OwnedCudaStream,
        max_weight_elements: usize,
        max_input_columns: usize,
        max_tokens: usize,
    ) -> Result<Self, CudaRuntimeError> {
        let paired_ffn = paired_ffn_switch(std::env::var_os("AUGUR_QWEN35_PAIRED_NVFP4_FFN").as_deref())?;
        let skip_unused_q8 = paired_ffn_switch(std::env::var_os("AUGUR_QWEN35_SKIP_UNUSED_PREFILL_Q8").as_deref())?;
        let full_ffn_input_reuse = paired_ffn_switch(std::env::var_os("AUGUR_QWEN35_FULL_FFN_INPUT_REUSE").as_deref())?;
        let gdn_bf16_pair = paired_ffn_switch(std::env::var_os("AUGUR_QWEN35_GDN_BF16_PAIR").as_deref())?;
        // SAFETY: pure constant query, no pointers, CUDA calls or allocation.
        let native_limit =
            unsafe { augur_cuda_sys::augur_cuda_qwen35_prefill_cublas_compiled_max_tokens_v1() };
        if native_limit != AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_COMPILED_MAX_TOKENS_V1 {
            return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        if cuda_device_properties(stream.device_ordinal())?.compute_capability
            != CudaComputeCapability::SM70
        {
            return Err(validation(CudaRuntimeErrorKind::UnsupportedCapability));
        }
        if max_weight_elements == 0
            || max_input_columns == 0
            || max_tokens == 0
            || max_tokens > AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_COMPILED_MAX_TOKENS_V1 as usize
            || !max_input_columns.is_multiple_of(64)
        {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let weight_bytes = max_weight_elements
            .checked_mul(F16_BYTES)
            .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))?;
        let input_bytes = max_input_columns
            .checked_mul(max_tokens)
            .and_then(|elements| elements.checked_mul(F16_BYTES))
            .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))?;
        let decoded_weight_f16 = OwnedCudaBuffer::allocate(stream.device_ordinal(), weight_bytes)?;
        let decoded_input_f16 = OwnedCudaBuffer::allocate(stream.device_ordinal(), input_bytes)?;
        let weight_view = decoded_weight_f16.view(0, weight_bytes)?;
        let input_view = decoded_input_f16.view(0, input_bytes)?;

        let mut args = AugurCudaQwen35PrefillCublasPlanArgsV1::new();
        args.max_tokens = max_tokens
            .try_into()
            .map_err(|_| validation(CudaRuntimeErrorKind::InvalidDimensions))?;
        args.decoded_weight_f16 = weight_view.checked_address(OPERATION)?;
        args.decoded_weight_len_bytes = weight_bytes
            .try_into()
            .map_err(|_| validation(CudaRuntimeErrorKind::RangeOverflow))?;
        args.decoded_input_f16 = input_view.checked_address(OPERATION)?;
        args.decoded_input_len_bytes = input_bytes
            .try_into()
            .map_err(|_| validation(CudaRuntimeErrorKind::RangeOverflow))?;

        let mut raw = core::ptr::null_mut();
        // SAFETY: both workspaces are live, disjoint device allocations bound
        // to this stream's ordinal. The returned plan cannot outlive them.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_cublas_plan_create_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &args,
                &mut raw,
            )
        };
        CudaRuntimeError::from_result(OPERATION, result)?;
        if raw.is_null() {
            return Err(CudaRuntimeError::null_handle());
        }
        Ok(Self {
            raw,
            decoded_weight_f16,
            decoded_input_f16,
            device_ordinal: stream.device_ordinal(),
            stream_owner_instance_key: stream.owner_instance_key()?,
            max_tokens,
            paired_ffn,
            paired_ffn_submissions: core::cell::Cell::new(0),
            skip_unused_q8,
            full_ffn_input_reuse,
            gdn_bf16_pair,
            gdn_bf16_pair_submissions: core::cell::Cell::new(0),
            q8_input_pack_elisions: core::cell::Cell::new(0),
            direct_b8_only: false,
        })
    }

    /// Profile2 packed B8 consumes only staged FP16 input, not expanded weights.
    /// Retain a 16-byte disjoint ABI placeholder instead of the largest matrix.
    pub fn new_direct_b8(stream:&OwnedCudaStream,max_input_columns:usize)->Result<Self,CudaRuntimeError> {
        let mut plan=Self::new(stream,8,max_input_columns,8)?;
        plan.direct_b8_only=true;Ok(plan)
    }

    pub const fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    pub fn bind_projection(
        &self,
        stream: &OwnedCudaStream,
        weight: CudaQwen35PrefillWeightV1,
        input: CudaQ8_1ActivationViewV1,
        output: CudaF32View,
        tokens: usize,
    ) -> Result<CudaQwen35PrefillCublasProjectionV1, CudaRuntimeError> {
        self.require_stream(stream)?;
        let raw = bind_projection_raw(
            self.device_ordinal,
            self.max_tokens,
            self.decoded_weight_f16
                .view(0, self.decoded_weight_f16.len_bytes())?,
            self.decoded_input_f16
                .view(0, self.decoded_input_f16.len_bytes())?,
            weight,
            input,
            output,
            tokens,
        )?;
        // SAFETY: the Rust binding checked the live allocation identities,
        // exact extents and pairwise disjoint ranges. Native preflight repeats
        // device allocation and SM70 validation without launching work.
        let result = unsafe {
            if raw.weight_format == row_input::FORMAT {
                augur_cuda_sys::augur_cuda_qwen35_row_q8_input_binding_preflight_v1(
                    self.device_ordinal, stream.raw()?, self.raw, &raw,
                )
            } else { augur_cuda_sys::augur_cuda_qwen35_prefill_cublas_projection_preflight_v1(
                self.device_ordinal,
                stream.raw()?,
                self.raw,
                &raw,
            ) }
        };
        CudaRuntimeError::from_result(OPERATION, result)?;
        Ok(CudaQwen35PrefillCublasProjectionV1 {
            raw,
            weight_decode_lanes: 1,
            quantized_small_batch: false,
            exact_projection_tiles: false,
            projection_row_reuse: false,
            ffn_down_row_reuse: false,
            canonical_tiled_rows: false,
            canonical_grid_rows: false,
            tensor_ffn_down: false,
            tensor_ffn_down_compensated: false,
            integer_mma_ffn: false,
            integer_mma_input: false,
            llama_f16_input: false,
            original_input: None,
            llama_full_original: false,
            llama_exact_input_rows: false,
            device_ordinal: self.device_ordinal,
            stream_owner_instance_key: self.stream_owner_instance_key,
            plan_instance_key: self.raw as usize,
        })
    }

    /// Replace an admitted Q8-input projection with a separately typed original
    /// F32 source. Retains the same stream, plan, weight/output and scratch.
    /// Caller retains the normalized activation owner through stream completion.
    pub fn bind_original_input_audit(
        &self,
        stream: &OwnedCudaStream,
        projection: &mut CudaQwen35PrefillCublasProjectionV1,
        input: CudaF32View,
    ) -> Result<(), CudaRuntimeError> {
        self.require_projection(stream, projection)?;
        if input.buffer_view().device_ordinal() != self.device_ordinal {
            return Err(CudaRuntimeError::device_mismatch());
        }
        let mut candidate = *projection;
        candidate.enable_llama_f16_input_audit()?;
        let raw = original_input_raw(&candidate.raw, input)?;
        // SAFETY: checked live F32 view, exact extent and admitted plan/stream.
        // Native preflight verifies all five device ranges, disjointness, SM70,
        // scratch capacity and bounded input shapes before mutating the binding.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_input_original_preflight_v1(
                self.device_ordinal, stream.raw()?, self.raw, &raw,
            )
        };
        CudaRuntimeError::from_result(OPERATION, result)?;
        candidate.original_input = Some(raw);
        *projection = candidate;
        Ok(())
    }

    /// Prefix-only original-F32 projection route. All views retain the existing
    /// owner/plan and scratch; an alternate output permits safe workspace reuse.
    pub fn bind_llama_full_original_audit(
        &self,
        stream: &OwnedCudaStream,
        projection: &mut CudaQwen35PrefillCublasProjectionV1,
        input: CudaF32View,
        output: Option<CudaF32View>,
    ) -> Result<(), CudaRuntimeError> {
        self.bind_llama_original_impl(stream, projection, input, output, false)
    }

    /// Experimental real-input dispatch, 1..=512 rows. Never a speculative
    /// B8 route; smaller DLL dispatch families require separate implementations.
    pub fn bind_llama_exact_input_rows_audit(
        &self,
        stream: &OwnedCudaStream,
        projection: &mut CudaQwen35PrefillCublasProjectionV1,
        input: CudaF32View,
        output: Option<CudaF32View>,
    ) -> Result<(), CudaRuntimeError> {
        self.bind_llama_original_impl(stream, projection, input, output, true)
    }

    fn bind_llama_original_impl(
        &self,
        stream: &OwnedCudaStream,
        projection: &mut CudaQwen35PrefillCublasProjectionV1,
        input: CudaF32View,
        output: Option<CudaF32View>,
        exact_input_rows: bool,
    ) -> Result<(), CudaRuntimeError> {
        self.require_projection(stream, projection)?;
        if projection.llama_full_original || projection.original_input.is_some() {
            return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        if input.buffer_view().device_ordinal() != self.device_ordinal
            || output.is_some_and(|v| v.buffer_view().device_ordinal() != self.device_ordinal)
        {
            return Err(CudaRuntimeError::device_mismatch());
        }
        let mut candidate = *projection;
        let raw = if exact_input_rows {
            llama_original_rows_raw(&candidate.raw, input, output, true)?
        } else {
            llama_full_original_raw(&candidate.raw, input, output)?
        };
        // SAFETY: same live plan/stream as the original binding. Native preflight
        // repeats exact extents, all five disjoint allocation ranges and scratch.
        CudaRuntimeError::from_result(OPERATION, unsafe {
            if raw.weight_format == row_input::FORMAT {
                augur_cuda_sys::augur_cuda_qwen35_row_q8_input_original_preflight_v1(
                    self.device_ordinal, stream.raw()?, self.raw, &raw,
                )
            } else if exact_input_rows {
                augur_cuda_sys::augur_cuda_qwen35_prefill_llama_projection_rows_preflight_v1(
                    self.device_ordinal, stream.raw()?, self.raw, &raw,
                )
            } else {
                augur_cuda_sys::augur_cuda_qwen35_prefill_llama_projection_preflight_v1(
                    self.device_ordinal, stream.raw()?, self.raw, &raw,
                )
            }
        })?;
        candidate.raw.output_f32 = raw.output_f32;
        candidate.raw.output_len_bytes = raw.output_len_bytes;
        candidate.weight_decode_lanes = 1;
        candidate.quantized_small_batch = false;
        candidate.exact_projection_tiles = false;
        candidate.projection_row_reuse = false;
        candidate.ffn_down_row_reuse = false;
        candidate.canonical_tiled_rows = false;
        candidate.canonical_grid_rows = false;
        candidate.tensor_ffn_down = false;
        candidate.tensor_ffn_down_compensated = false;
        candidate.integer_mma_ffn = false;
        candidate.integer_mma_input = false;
        candidate.llama_f16_input = false;
        candidate.original_input = Some(raw);
        candidate.llama_full_original = true;
        candidate.llama_exact_input_rows = exact_input_rows;
        *projection = candidate;
        Ok(())
    }

    /// Only omit an external Q8 pack if every listed consumer is immutably
    /// bound to this exact original F32 input. Internal llama packing is kept.
    /// The caller must list all consumers before scratch reuse; no pointer cache
    /// survives this call, and mixed/native B8 consumers retain the old pack.
    pub fn submit_required_q8_input(
        &self,
        stream: &OwnedCudaStream,
        quantize: &CudaQ8_1ActivationQuantizeArgsV1,
        consumers: &[&CudaQwen35PrefillCublasProjectionV1],
    ) -> Result<(), CudaRuntimeError> {
        if consumers.is_empty() || consumers.len() > 4 {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        if self.skip_unused_q8 {
            for consumer in consumers { self.require_projection(stream, consumer)?; }
            if consumers.iter().all(|p| original_input_covers_pack(p, quantize.as_raw())) {
                self.q8_input_pack_elisions.set(self.q8_input_pack_elisions.get() + 1);
                return Ok(());
            }
        }
        quantize.submit_for_plan(stream)
    }

    pub const fn full_ffn_input_reuse_enabled(&self) -> bool { self.full_ffn_input_reuse }

    pub fn q8_input_pack_elisions(&self) -> u64 { self.q8_input_pack_elisions.get() }

    pub fn gdn_bf16_pair_submissions(&self) -> u64 { self.gdn_bf16_pair_submissions.get() }

    /// A diagnostic that observes packed scratch makes it a live output even
    /// when projections themselves read F32. Retain its old materialization.
    pub fn retain_q8_for_observation(&mut self) { self.skip_unused_q8 = false; }

    /// Pack one immutable original-F32 input once for a bounded prefix group.
    /// Reuse never survives this call, including when a later call has the same
    /// address with new contents. Legacy, BF16 and small-row paths stay scalar.
    pub fn submit_shared_input_group(
        &self,
        stream: &OwnedCudaStream,
        projections: &[&CudaQwen35PrefillCublasProjectionV1],
    ) -> Result<(), CudaRuntimeError> {
        if !(2..=3).contains(&projections.len()) {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        for projection in projections {
            self.require_projection(stream, projection)?;
        }
        if self.paired_ffn && projections.len() == 2
            && paired_ffn_eligible(projections[0], projections[1])
        {
            // SAFETY: immutable individual bindings retain this plan/stream
            // and all owners. Native checks shared input and cross aliases.
            CudaRuntimeError::from_result(OPERATION, unsafe {
                augur_cuda_sys::augur_cuda_qwen35_ffn_nvfp4_pair_prevalidated_submit_v1(
                    self.device_ordinal, stream.raw()?, self.raw,
                    &projections[0].raw, &projections[1].raw,
                )
            })?;
            self.paired_ffn_submissions.set(self.paired_ffn_submissions.get() + 1);
            return Ok(());
        }
        let gdn_pair = self.gdn_bf16_pair && projections.len() == 2
            && projections.iter().all(|p| p.raw.weight_format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_BF16_V1);
        let eligible = |p: &CudaQwen35PrefillCublasProjectionV1| {
            shared_original_input_group_rows(
                p.llama_full_original_enabled(), p.llama_exact_input_rows, p.raw.tokens)
                && (gdn_pair || matches!(p.raw.weight_format,
                    AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1
                    | AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_NVFP4_V1 | row_input::FORMAT))
                && p.weight_decode_lanes == 1 && !p.quantized_small_batch
                && !p.exact_projection_tiles && !p.projection_row_reuse
                && !p.canonical_tiled_rows && !p.canonical_grid_rows
                && !p.tensor_ffn_down && !p.tensor_ffn_down_compensated
                && !p.integer_mma_ffn && !p.integer_mma_input && !p.llama_f16_input
        };
        if projections.iter().all(|p| eligible(p)) {
            let first = *projections[0].original_input.as_ref()
                .ok_or_else(|| validation(CudaRuntimeErrorKind::InvalidMetadata))?;
            let mut group = [first; 3];
            let mut same_input = true;
            for (index, projection) in projections.iter().enumerate() {
                let raw = *projection.original_input.as_ref()
                    .ok_or_else(|| validation(CudaRuntimeErrorKind::InvalidMetadata))?;
                same_input &= raw.input_f32 == first.input_f32
                    && raw.input_len_bytes == first.input_len_bytes
                    && raw.tokens == first.tokens && raw.cols == first.cols;
                group[index] = raw;
            }
            let row_count=group[..projections.len()].iter().filter(|p|p.weight_format==row_input::FORMAT).count();
            if same_input && (row_count==0 || row_count==projections.len()) {
                // SAFETY: all bindings retain this plan/stream and their owners.
                // Native admission checks the complete group and cross aliases
                // before launching. Same-stream calls cannot interleave scratch.
                CudaRuntimeError::from_result(OPERATION, unsafe {
                    if row_count!=0 { augur_cuda_sys::augur_cuda_qwen35_row_q8_input_group_submit_v1(
                        self.device_ordinal, stream.raw()?, self.raw, group.as_ptr(), projections.len() as u32,
                    ) } else { augur_cuda_sys::augur_cuda_qwen35_prefill_llama_projection_group_rows_prevalidated_submit_v1(
                        self.device_ordinal, stream.raw()?, self.raw,
                        group.as_ptr(), projections.len() as u32,
                    ) }
                })?;
                if gdn_pair {
                    self.gdn_bf16_pair_submissions.set(self.gdn_bf16_pair_submissions.get() + 1);
                }
                return Ok(());
            }
        }
        for projection in projections {
            self.submit(stream, projection)?;
        }
        Ok(())
    }

    pub fn submit(
        &self,
        stream: &OwnedCudaStream,
        projection: &CudaQwen35PrefillCublasProjectionV1,
    ) -> Result<(), CudaRuntimeError> {
        self.require_projection(stream, projection)?;
        if projection.raw.weight_format==row_input::FORMAT {
            return self.submit_row_input(stream,projection);
        }
        if projection.llama_full_original {
            if projection.weight_decode_lanes != 1 || projection.quantized_small_batch
                || projection.exact_projection_tiles || projection.projection_row_reuse
                || projection.canonical_tiled_rows || projection.canonical_grid_rows
                || projection.tensor_ffn_down || projection.tensor_ffn_down_compensated
                || projection.integer_mma_ffn || projection.integer_mma_input || projection.llama_f16_input
            {
                return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
            }
            let raw = projection.original_input.as_ref()
                .ok_or_else(|| validation(CudaRuntimeErrorKind::InvalidMetadata))?;
            // SAFETY: same plan/owner and immutable preflighted F32 views. The
            // runtime keeps the workspace and weight owners through completion.
            return CudaRuntimeError::from_result(OPERATION, unsafe {
                if projection.llama_exact_input_rows {
                    augur_cuda_sys::augur_cuda_qwen35_prefill_llama_projection_rows_prevalidated_submit_v1(
                        self.device_ordinal, stream.raw()?, self.raw, raw,
                    )
                } else {
                    augur_cuda_sys::augur_cuda_qwen35_prefill_llama_projection_prevalidated_submit_v1(
                        self.device_ordinal, stream.raw()?, self.raw, raw,
                    )
                }
            });
        }
        if projection.integer_mma_ffn || projection.integer_mma_input || projection.llama_f16_input {
            if projection.weight_decode_lanes != 1
                || projection.quantized_small_batch
                || projection.exact_projection_tiles
                || projection.projection_row_reuse
                || projection.canonical_tiled_rows
                || projection.canonical_grid_rows
                || projection.tensor_ffn_down
                || projection.tensor_ffn_down_compensated
                || usize::from(projection.integer_mma_ffn)
                    + usize::from(projection.integer_mma_input)
                    + usize::from(projection.llama_f16_input) != 1
            {
                return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
            }
            // SAFETY: same live plan/stream and immutable disjoint views as
            // bind_projection; native admission repeats shape and scratch bounds.
            let result = unsafe {
                if let Some(raw) = &projection.original_input {
                    augur_cuda_sys::augur_cuda_qwen35_prefill_input_original_prevalidated_submit_v1(
                        self.device_ordinal, stream.raw()?, self.raw, raw,
                    )
                } else if projection.llama_f16_input {
                    augur_cuda_sys::augur_cuda_qwen35_prefill_input_llama_prevalidated_submit_v1(
                        self.device_ordinal, stream.raw()?, self.raw, &projection.raw,
                    )
                } else if projection.integer_mma_input {
                    augur_cuda_sys::augur_cuda_qwen35_prefill_input_integer_prevalidated_submit_v1(
                        self.device_ordinal, stream.raw()?, self.raw, &projection.raw,
                    )
                } else {
                    augur_cuda_sys::augur_cuda_qwen35_prefill_integer_mma_prevalidated_submit_v1(
                        self.device_ordinal, stream.raw()?, self.raw, &projection.raw,
                    )
                }
            };
            return CudaRuntimeError::from_result(OPERATION, result);
        }
        if projection.tensor_ffn_down {
            if projection.weight_decode_lanes != 1
                || projection.quantized_small_batch
                || projection.exact_projection_tiles
                || projection.projection_row_reuse
                || projection.canonical_tiled_rows
                || projection.canonical_grid_rows
            {
                return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
            }
            // SAFETY: immutable same-plan/stream binding retained all original
            // views and full scratch extents. Native admission fixes the two
            // measured FFN-down shapes and the 896-row working tile.
            let result = unsafe {
                if projection.tensor_ffn_down_compensated {
                    augur_cuda_sys::augur_cuda_qwen35_prefill_down_compensated_prevalidated_submit_v1(
                        self.device_ordinal,
                        stream.raw()?,
                        self.raw,
                        &projection.raw,
                    )
                } else {
                    augur_cuda_sys::augur_cuda_qwen35_prefill_down_tensor_prevalidated_submit_v1(
                        self.device_ordinal,
                        stream.raw()?,
                        self.raw,
                        &projection.raw,
                    )
                }
            };
            return CudaRuntimeError::from_result(OPERATION, result);
        }
        if projection.canonical_grid_rows {
            // SAFETY: explicit bounded policy, fully preflighted immutable plan,
            // same retained owners; native code checks shape and <=512 extent.
            let result = unsafe {
                augur_cuda_sys::augur_cuda_qwen35_canonical_prefill_grid_prevalidated_submit_v1(
                    self.device_ordinal,
                    stream.raw()?,
                    self.raw,
                    &projection.raw,
                )
            };
            return CudaRuntimeError::from_result(OPERATION, result);
        }
        if projection.canonical_tiled_rows {
            for start in (0..projection.raw.tokens).step_by(8) {
                let raw = canonical_projection_tile(&projection.raw, start)?;
                // SAFETY: the full projection passed bind preflight. Each
                // checked contiguous slice retains the same stream/plan/weight
                // owners, disjoint buffers, and canonical native <=8-row ABI.
                let result = unsafe {
                    augur_cuda_sys::augur_cuda_qwen35_exact_projection_tiles_prevalidated_submit_v1(
                        self.device_ordinal,
                        stream.raw()?,
                        self.raw,
                        &raw,
                    )
                };
                CudaRuntimeError::from_result(OPERATION, result)?;
            }
            return Ok(());
        }
        // SAFETY: bind_projection completed native preflight for the immutable
        // args and exact plan. All owners must remain live through completion.
        let result = unsafe {
            if projection.ffn_down_row_reuse {
                augur_cuda_sys::augur_cuda_qwen35_ffn_down_row_reuse_prevalidated_submit_v1(
                    self.device_ordinal, stream.raw()?, self.raw, &projection.raw,
                )
            } else if projection.projection_row_reuse {
                augur_cuda_sys::augur_cuda_qwen35_projection_row_reuse_prevalidated_submit_v1(
                    self.device_ordinal,
                    stream.raw()?,
                    self.raw,
                    &projection.raw,
                )
            } else if projection.exact_projection_tiles {
                augur_cuda_sys::augur_cuda_qwen35_exact_projection_tiles_prevalidated_submit_v1(
                    self.device_ordinal,
                    stream.raw()?,
                    self.raw,
                    &projection.raw,
                )
            } else if projection.quantized_small_batch {
                augur_cuda_sys::augur_cuda_qwen35_quantized_small_batch_prevalidated_submit_v1(
                    self.device_ordinal,
                    stream.raw()?,
                    self.raw,
                    &projection.raw,
                )
            } else if projection.weight_decode_lanes == 4 {
                augur_cuda_sys::augur_cuda_qwen35_prefill_cublas_projection_quad_prevalidated_submit_v1(
                    self.device_ordinal,
                    stream.raw()?,
                    self.raw,
                    &projection.raw,
                )
            } else if projection.weight_decode_lanes == 2 {
                augur_cuda_sys::augur_cuda_qwen35_prefill_cublas_projection_pair_prevalidated_submit_v1(
                    self.device_ordinal,
                    stream.raw()?,
                    self.raw,
                    &projection.raw,
                )
            } else {
                augur_cuda_sys::augur_cuda_qwen35_prefill_cublas_projection_prevalidated_submit_v1(
                    self.device_ordinal,
                    stream.raw()?,
                    self.raw,
                    &projection.raw,
                )
            }
        };
        CudaRuntimeError::from_result(OPERATION, result)
    }

    pub const fn paired_ffn_enabled(&self) -> bool { self.paired_ffn }

    pub fn paired_ffn_submissions(&self) -> u64 { self.paired_ffn_submissions.get() }

    /// Independent diagnostic replay only. Records completion of weight decode,
    /// input decode and GEMM; ordinary submit remains the uninstrumented path.
    /// Events/plan/views must live through the caller's stream terminal proof.
    pub fn submit_profiled(
        &self,
        stream: &OwnedCudaStream,
        projection: &CudaQwen35PrefillCublasProjectionV1,
        after_stages: [&OwnedCudaEvent; 3],
    ) -> Result<(), CudaRuntimeError> {
        self.require_projection(stream, projection)?;
        if projection.quantized_small_batch
            || projection.canonical_tiled_rows
            || projection.canonical_grid_rows
            || projection.tensor_ffn_down
            || projection.integer_mma_ffn
            || projection.integer_mma_input
            || projection.llama_f16_input
        {
            // A fused DP4A projection has no weight-decode/input-decode/GEMM
            // stages. Do not emit misleading legacy stage timings.
            return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        for event in &after_stages {
            if event.device_ordinal() != self.device_ordinal {
                return Err(CudaRuntimeError::device_mismatch());
            }
        }
        let keys = [
            after_stages[0].owner_instance_key()?,
            after_stages[1].owner_instance_key()?,
            after_stages[2].owner_instance_key()?,
        ];
        if keys[0] == keys[1] || keys[0] == keys[2] || keys[1] == keys[2] {
            return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        for (stage, event) in after_stages.into_iter().enumerate() {
            // SAFETY: full native preflight completed at binding. This single
            // method preserves 0 -> 1 -> 2 order on the same plan/stream/views,
            // with no interposed projection or scratch use. Event metadata is
            // recorded through OwnedCudaEvent, never bypassed through raw FFI.
            let result = unsafe {
                if projection.weight_decode_lanes == 4 {
                    augur_cuda_sys::augur_cuda_qwen35_prefill_cublas_projection_quad_profile_stage_prevalidated_submit_v1(
                        self.device_ordinal, stream.raw()?, self.raw, &projection.raw, stage as u32,
                    )
                } else if projection.weight_decode_lanes == 2 {
                    augur_cuda_sys::augur_cuda_qwen35_prefill_cublas_projection_pair_profile_stage_prevalidated_submit_v1(
                        self.device_ordinal, stream.raw()?, self.raw, &projection.raw, stage as u32,
                    )
                } else {
                    augur_cuda_sys::augur_cuda_qwen35_prefill_cublas_projection_profile_stage_prevalidated_submit_v1(
                        self.device_ordinal, stream.raw()?, self.raw, &projection.raw, stage as u32,
                    )
                }
            };
            CudaRuntimeError::from_result(OPERATION, result)?;
            event.record(stream)?;
        }
        Ok(())
    }





    pub fn execute_sync(
        &self,
        stream: &OwnedCudaStream,
        projection: &CudaQwen35PrefillCublasProjectionV1,
    ) -> Result<(), CudaRuntimeError> {
        self.require_projection(stream, projection)?;
        if projection.raw.weight_format==row_input::FORMAT {
            self.submit(stream,projection)?;
            return stream.synchronize();
        }
        if projection.tensor_ffn_down || projection.integer_mma_ffn
            || projection.integer_mma_input || projection.llama_f16_input {
            self.submit(stream, projection)?;
            return stream.synchronize();
        }
        // SAFETY: native entry repeats full device-range validation and drains
        // the owner stream before returning.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_cublas_projection_v1(
                self.device_ordinal,
                stream.raw()?,
                self.raw,
                &projection.raw,
            )
        };
        CudaRuntimeError::from_result(OPERATION, result)
    }











    fn require_projection(
        &self,
        stream: &OwnedCudaStream,
        projection: &CudaQwen35PrefillCublasProjectionV1,
    ) -> Result<(), CudaRuntimeError> {
        self.require_stream(stream)?;
        if projection.device_ordinal != self.device_ordinal
            || projection.stream_owner_instance_key != self.stream_owner_instance_key
            || projection.plan_instance_key != self.raw as usize
        {
            return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        Ok(())
    }

    fn require_stream(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.device_ordinal {
            return Err(CudaRuntimeError::device_mismatch());
        }
        if stream.owner_instance_key()? != self.stream_owner_instance_key || self.raw.is_null() {
            return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        Ok(())
    }
}

impl Drop for CudaQwen35PrefillCublasPlanV1 {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // Native cuBLAS destruction is ordered before the owned scratch
            // allocations are dropped. Callers must drain submitted work.
            unsafe { augur_cuda_sys::augur_cuda_qwen35_prefill_cublas_plan_destroy_v1(self.raw) };
            self.raw = core::ptr::null_mut();
        }
    }
}

/// Stream-bound packed 1..8-token projection, with no cuBLAS or FP16 scratch.
/// The enclosing execution owner must retain all three buffers through the
/// covering stream terminal proof, including error/drain paths.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CudaQwen35PackedSmallBatchProjectionV1 {
    raw: AugurCudaQwen35PrefillCublasProjectionArgsV1,
    device_ordinal: i32,
    stream_owner_instance_key: usize,
    exact_tiles: bool,
}

impl CudaQwen35PackedSmallBatchProjectionV1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        weight: CudaQwen35PrefillWeightV1,
        input: CudaQ8_1ActivationViewV1,
        output: CudaF32View,
        tokens: usize,
        exact_tiles: bool,
    ) -> Result<Self, CudaRuntimeError> {
        let raw =
            bind_packed_projection_raw(stream.device_ordinal(), weight, input, output, tokens)?;
        // SAFETY: Rust checked format, extents, device identity and disjointness.
        // Native preflight checks live allocations and SM70 without a kernel launch.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_packed_small_batch_projection_preflight_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &raw,
                u32::from(exact_tiles),
            )
        };
        CudaRuntimeError::from_result(OPERATION, result)?;
        Ok(Self {
            raw,
            device_ordinal: stream.device_ordinal(),
            stream_owner_instance_key: stream.owner_instance_key()?,
            exact_tiles,
        })
    }

    pub const fn as_raw(&self) -> &AugurCudaQwen35PrefillCublasProjectionArgsV1 {
        &self.raw
    }

    pub fn submit(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.device_ordinal {
            return Err(CudaRuntimeError::device_mismatch());
        }
        if stream.owner_instance_key()? != self.stream_owner_instance_key {
            return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        // SAFETY: bind preflighted these immutable views on this exact stream.
        // The enclosing owner retains them through completion; no scratch/plan
        // can be replaced here and the submit performs no allocation.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_packed_small_batch_projection_prevalidated_submit_v1(
                self.device_ordinal,
                stream.raw()?,
                &self.raw,
                u32::from(self.exact_tiles),
            )
        };
        CudaRuntimeError::from_result(OPERATION, result)
    }
}

fn bind_packed_projection_raw(
    ordinal: i32,
    weight: CudaQwen35PrefillWeightV1,
    input: CudaQ8_1ActivationViewV1,
    output: CudaF32View,
    tokens: usize,
) -> Result<AugurCudaQwen35PrefillCublasProjectionArgsV1, CudaRuntimeError> {
    if !(1..=8).contains(&tokens)
        || !matches!(
            weight,
            CudaQwen35PrefillWeightV1::Q8_0(_) | CudaQwen35PrefillWeightV1::RowQ8Input(_) | CudaQwen35PrefillWeightV1::Nvfp4 { .. }
        )
        || weight.cols().checked_mul(tokens) != Some(input.elements())
        || weight.rows().checked_mul(tokens) != Some(output.element_count())
    {
        return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
    }
    let scale = weight.tensor_scale();
    if !scale.is_finite() || scale <= 0.0 {
        return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
    }
    validate_disjoint_views(
        ordinal,
        &[weight.view(), input.buffer_view(), output.buffer_view()],
    )?;
    let mut raw = AugurCudaQwen35PrefillCublasProjectionArgsV1::new();
    raw.weight_format = weight.format();
    raw.rows = u32::try_from(weight.rows())
        .map_err(|_| validation(CudaRuntimeErrorKind::InvalidDimensions))?;
    raw.cols = u32::try_from(weight.cols())
        .map_err(|_| validation(CudaRuntimeErrorKind::InvalidDimensions))?;
    raw.tokens = tokens as u32;
    raw.tensor_scale = scale;
    bind_raw_view(&mut raw.weight, &mut raw.weight_len_bytes, weight.view())?;
    bind_raw_view(
        &mut raw.input_q8_1,
        &mut raw.input_len_bytes,
        input.buffer_view(),
    )?;
    bind_raw_view(
        &mut raw.output_f32,
        &mut raw.output_len_bytes,
        output.buffer_view(),
    )?;
    Ok(raw)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CudaQwen35PrefillCublasProjectionV1 {
    raw: AugurCudaQwen35PrefillCublasProjectionArgsV1,
    weight_decode_lanes: u8,
    quantized_small_batch: bool,
    exact_projection_tiles: bool,
    projection_row_reuse: bool,
    ffn_down_row_reuse: bool,
    canonical_tiled_rows: bool,
    canonical_grid_rows: bool,
    tensor_ffn_down: bool,
    tensor_ffn_down_compensated: bool,
    integer_mma_ffn: bool,
    integer_mma_input: bool,
    llama_f16_input: bool,
    original_input: Option<AugurCudaQwen35PrefillF32ProjectionArgsV1>,
    llama_full_original: bool,
    llama_exact_input_rows: bool,
    device_ordinal: i32,
    stream_owner_instance_key: usize,
    plan_instance_key: usize,
}

fn paired_ffn_switch(value: Option<&std::ffi::OsStr>) -> Result<bool, CudaRuntimeError> {
    match value {
        None => Ok(false),
        Some(v) if v == "1" => Ok(true),
        _ => Err(validation(CudaRuntimeErrorKind::InvalidMetadata)),
    }
}

fn paired_ffn_eligible(
    gate: &CudaQwen35PrefillCublasProjectionV1,
    up: &CudaQwen35PrefillCublasProjectionV1,
) -> bool {
    let member = |p: &CudaQwen35PrefillCublasProjectionV1| {
        p.raw.weight_format == AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_NVFP4_V1
            && p.raw.tokens == 8 && p.raw.rows == 17408 && p.raw.cols == 5120
            && p.weight_decode_lanes == 1 && p.quantized_small_batch
            && p.exact_projection_tiles && p.projection_row_reuse
            && !p.canonical_tiled_rows && !p.canonical_grid_rows
            && !p.tensor_ffn_down && !p.tensor_ffn_down_compensated
            && !p.integer_mma_ffn && !p.integer_mma_input && !p.llama_f16_input
            && p.original_input.is_none() && !p.llama_full_original && !p.llama_exact_input_rows
    };
    member(gate) && member(up) && gate.raw.input_q8_1 == up.raw.input_q8_1
        && gate.raw.input_len_bytes == up.raw.input_len_bytes
}

fn original_input_covers_pack(
    p: &CudaQwen35PrefillCublasProjectionV1,
    q: &augur_cuda_sys::AugurCudaQ8_1ActivationQuantizeArgsV1,
) -> bool {
    let Some(input) = p.original_input.as_ref() else { return false; };
    p.llama_full_original && p.weight_decode_lanes == 1
        && !p.quantized_small_batch && !p.exact_projection_tiles && !p.projection_row_reuse
        && !p.canonical_tiled_rows && !p.canonical_grid_rows && !p.tensor_ffn_down
        && !p.tensor_ffn_down_compensated && !p.integer_mma_ffn && !p.integer_mma_input
        && !p.llama_f16_input && (1..=512).contains(&input.tokens)
        && input.tokens == p.raw.tokens && input.cols == p.raw.cols
        && input.input_f32 == q.input_f32 && input.input_len_bytes == q.input_len_bytes
        && u64::from(input.tokens) * u64::from(input.cols) == u64::from(q.elements)
        && p.raw.input_q8_1 == q.output_q8_1 && p.raw.input_len_bytes == q.output_len_bytes
}

impl CudaQwen35PrefillCublasProjectionV1 {
    pub const fn llama_exact_input_rows_enabled(&self) -> bool {
        self.llama_exact_input_rows && self.llama_full_original_enabled()
    }
    pub const fn llama_full_original_enabled(&self) -> bool {
        self.llama_full_original && self.original_input.is_some()
    }
    /// Explicit numerical alternative to an already admitted B512 input.
    /// Q8 operands are rounded to F16 and GEMM accumulates/outputs F32.
    /// Selection is prefix-only at the runtime caller, not a global override.
    pub fn enable_llama_f16_input_audit(&mut self) -> Result<(), CudaRuntimeError> {
        if !self.integer_mma_input || self.llama_f16_input || self.raw.weight % 4 != 0 {
            return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        // Re-run admission on a copy so incompatible selector mutation fails
        // without partially changing this binding.
        let mut candidate = *self;
        candidate.integer_mma_input = false;
        candidate.enable_integer_mma_input_audit()?;
        candidate.integer_mma_input = false;
        candidate.llama_f16_input = true;
        *self = candidate;
        Ok(())
    }

    pub const fn llama_f16_input_enabled(&self) -> bool {
        self.llama_f16_input && self.original_input.is_none()
    }

    pub const fn llama_original_input_enabled(&self) -> bool {
        self.llama_f16_input && self.original_input.is_some()
    }

    /// Measured B512 Q8 input-only path; independent selector and accounting.
    /// Uses the retained plan scratch, no resident weight copy or new owner.
    pub fn enable_integer_mma_input_audit(&mut self) -> Result<(), CudaRuntimeError> {
        if self.weight_decode_lanes != 1
            || self.quantized_small_batch
            || self.exact_projection_tiles
            || self.projection_row_reuse
            || self.tensor_ffn_down
            || self.tensor_ffn_down_compensated
            || self.integer_mma_ffn
            || self.integer_mma_input
            || self.llama_f16_input
            || self.raw.tensor_scale != 1.0
            || self.integer_mma_input_scratch_bytes() == 0
        {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        self.canonical_tiled_rows = false;
        self.canonical_grid_rows = false;
        self.integer_mma_input = true;
        Ok(())
    }

    pub fn integer_mma_input_scratch_bytes(&self) -> u64 {
        // SAFETY: pure scalar selector; no CUDA call or pointer dereference.
        unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_input_integer_scratch_bytes_v1(
                self.raw.weight_format, self.raw.rows, self.raw.cols, self.raw.tokens,
            )
        }
    }

    pub const fn integer_mma_input_enabled(&self) -> bool {
        self.integer_mma_input
    }

    /// Default-off exact integer-MMA FFN. Measured NVFP4 B128/B256/B512
    /// and Q8 B256/B512 only; scratch comes from this retained plan.
    pub fn enable_integer_mma_ffn_audit(&mut self) -> Result<(), CudaRuntimeError> {
        if self.weight_decode_lanes != 1
            || self.quantized_small_batch
            || self.exact_projection_tiles
            || self.projection_row_reuse
            || self.tensor_ffn_down
            || self.tensor_ffn_down_compensated
            || self.integer_mma_ffn
            || self.integer_mma_ffn_scratch_bytes() == 0
            || self.integer_mma_input
            || self.llama_f16_input
        {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        self.canonical_tiled_rows = false;
        self.canonical_grid_rows = false;
        self.integer_mma_ffn = true;
        Ok(())
    }

    pub fn integer_mma_ffn_scratch_bytes(&self) -> u64 {
        // SAFETY: pure selector; does not dereference pointers or call CUDA.
        unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_integer_mma_scratch_bytes_v1(
                self.raw.weight_format,
                self.raw.rows,
                self.raw.cols,
                self.raw.tokens,
            )
        }
    }

    pub const fn integer_mma_ffn_enabled(&self) -> bool {
        self.integer_mma_ffn
    }

    /// Higher-precision default-off NVFP4 B128 down-only experiment. No Q8
    /// substitution; uses the same plan with a borrowed low-input scratch.
    pub fn enable_tensor_ffn_down_compensated_audit(&mut self) -> Result<(), CudaRuntimeError> {
        if self.tensor_ffn_down_compensated_tile_rows() != 896 {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        self.enable_tensor_ffn_down_audit()?;
        self.tensor_ffn_down_compensated = true;
        Ok(())
    }

    pub fn tensor_ffn_down_compensated_tile_rows(&self) -> u32 {
        // SAFETY: pure shape selector, no device call or pointer dereference.
        unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_down_compensated_tile_rows_v1(
                self.raw.weight_format,
                self.raw.rows,
                self.raw.cols,
                self.raw.tokens,
            )
        }
    }

    /// Explicit numerical experiment, not a canonical arithmetic policy.
    /// It only replaces FFN-down for real B64/B128 rows; gate/up and Q8_1
    /// packing are not changed. The existing plan scratch/owner is reused.
    pub fn enable_tensor_ffn_down_audit(&mut self) -> Result<(), CudaRuntimeError> {
        if self.weight_decode_lanes != 1
            || self.quantized_small_batch
            || self.exact_projection_tiles
            || self.projection_row_reuse
            || self.tensor_ffn_down
            || self.tensor_ffn_down_tile_rows() == 0
            || self.integer_mma_ffn
        {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        if self.integer_mma_input || self.llama_f16_input {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        self.canonical_tiled_rows = false;
        self.canonical_grid_rows = false;
        self.tensor_ffn_down = true;
        Ok(())
    }

    pub fn tensor_ffn_down_tile_rows(&self) -> u32 {
        // SAFETY: pure shape selector; no CUDA call or pointer dereference.
        unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_down_tensor_tile_rows_v1(
                self.raw.weight_format,
                self.raw.rows,
                self.raw.cols,
                self.raw.tokens,
            )
        }
    }

    pub const fn tensor_ffn_down_enabled(&self) -> bool {
        self.tensor_ffn_down
    }

    /// Same canonical tiled math in one two-dimensional launch. Diagnostic only.
    pub fn enable_canonical_grid_rows_audit(&mut self) -> Result<(), CudaRuntimeError> {
        canonical_projection_tile(&self.raw, 0)?;
        if self.weight_decode_lanes != 1
            || self.quantized_small_batch
            || self.exact_projection_tiles
            || self.tensor_ffn_down
            || self.integer_mma_ffn
        {
            return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        if self.integer_mma_input || self.llama_f16_input {
            return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        // SAFETY: pure native metadata selector, no CUDA or pointer dereference.
        let tile = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_exact_projection_tile_width_v1(
                self.raw.weight_format,
                self.raw.rows,
                self.raw.cols,
                8,
            )
        };
        if tile == 0 {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        self.canonical_tiled_rows = false;
        self.canonical_grid_rows = true;
        Ok(())
    }
    /// Explicit diagnostic schedule: slice a bound <=512-row projection into
    /// exact canonical <=8-row native calls, with no activation/materialization
    /// change. The legacy small-batch admission remains capped at eight.
    pub fn enable_canonical_tiled_rows_audit(&mut self) -> Result<(), CudaRuntimeError> {
        if self.weight_decode_lanes != 1
            || self.quantized_small_batch
            || self.exact_projection_tiles
            || self.tensor_ffn_down
            || self.integer_mma_ffn
        {
            return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        if self.integer_mma_input || self.llama_f16_input {
            return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        for start in (0..self.raw.tokens).step_by(8) {
            canonical_projection_tile(&self.raw, start)?;
        }
        if self.raw.tokens == 0 {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        self.canonical_tiled_rows = true;
        Ok(())
    }
    /// Exact, default-off full8/full4 scheduling on a frozen native shape set.
    /// Unknown shapes and token tails retain the canonical small-batch kernel.
    pub fn enable_exact_projection_tiles(&mut self) -> Result<(), CudaRuntimeError> {
        self.enable_quantized_small_batch()?;
        self.exact_projection_tiles = true;
        Ok(())
    }
    /// Default-off full8 input/gate-up row2 reuse. Other shapes and tails use
    /// the existing exact tile selector; arithmetic and owner binding do not change.
    pub fn enable_projection_row_reuse(&mut self) -> Result<(), CudaRuntimeError> {
        if self.canonical_tiled_rows
            || self.canonical_grid_rows
            || self.tensor_ffn_down
            || self.integer_mma_ffn
        {
            return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        if self.integer_mma_input || self.llama_f16_input {
            return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        self.enable_exact_projection_tiles()?;
        self.projection_row_reuse = true;
        Ok(())
    }
    /// Independent default-off full8 FFN-down row2 reuse. No tail padding,
    /// workspace, precision or owner-contract change.
    pub fn enable_ffn_down_row_reuse(&mut self) -> Result<(), CudaRuntimeError> {
        if self.raw.rows != 5120 || self.raw.cols != 17408 || self.raw.tokens != 8
            || !matches!(self.raw.weight_format, 1 | 2)
            || !self.projection_row_reuse || !self.exact_projection_tiles
        {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        // Recheck the complete canonical policy; no mutually exclusive path
        // may bypass this selector, even when called on a copied binding.
        self.enable_projection_row_reuse()?;
        self.ffn_down_row_reuse = true;
        Ok(())
    }
    pub fn ffn_down_row_reuse_enabled(&self) -> bool {
        self.ffn_down_row_reuse
    }
    /// Pure metadata report of the kernel actually selected by this binding.
    pub fn projection_row_reuse_width(&self) -> u32 {
        if !self.projection_row_reuse {
            return 0;
        }
        if self.raw.weight_format==row_input::FORMAT {
            return if self.raw.tokens==8 {2} else {0};
        }
        // SAFETY: pure selector, no pointer arguments, CUDA calls or dereference.
        unsafe {
            augur_cuda_sys::augur_cuda_qwen35_projection_row_reuse_width_v1(
                self.raw.weight_format,
                self.raw.rows,
                self.raw.cols,
                self.raw.tokens,
            )
        }
    }
    /// Opt-in small-N DP4A path. The canonical per-row FMA/reduction order is
    /// retained; no FP16 weight/activation materialization is performed.
    pub fn enable_quantized_small_batch(&mut self) -> Result<(), CudaRuntimeError> {
        if !quantized_small_batch_supported(self.raw.weight_format, self.raw.tokens)
            || self.weight_decode_lanes != 1
            || self.tensor_ffn_down
            || self.integer_mma_ffn
        {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        if self.integer_mma_input || self.llama_f16_input {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        self.quantized_small_batch = true;
        Ok(())
    }
    /// Explicit diagnostic policy; does not change ranges, precision or GEMM.
    pub fn enable_paired_weight_decode(&mut self) {
        self.weight_decode_lanes = 2;
    }

    /// Four scalar RN conversions followed by one aligned eight-byte store.
    pub fn enable_quad_weight_decode(&mut self) {
        self.weight_decode_lanes = 4;
    }

    pub const fn raw(&self) -> &AugurCudaQwen35PrefillCublasProjectionArgsV1 {
        &self.raw
    }
}

// Pure checked row slicing. Do not let a native prevalidated call escape the
// originally bound input/output extents or reuse a different arithmetic format.
fn canonical_projection_tile(
    raw: &AugurCudaQwen35PrefillCublasProjectionArgsV1,
    start: u32,
) -> Result<AugurCudaQwen35PrefillCublasProjectionArgsV1, CudaRuntimeError> {
    let invalid = || validation(CudaRuntimeErrorKind::InvalidDimensions);
    if !(1..=512).contains(&raw.tokens)
        || start >= raw.tokens
        || start % 8 != 0
        || raw.cols == 0
        || raw.cols % 32 != 0
        || raw.rows == 0
        || !quantized_small_batch_supported(raw.weight_format, 1)
    {
        return Err(invalid());
    }
    let input_stride = u64::from(raw.cols / 32) * 36;
    let output_stride = u64::from(raw.rows) * 4;
    if raw.input_len_bytes != input_stride * u64::from(raw.tokens)
        || raw.output_len_bytes != output_stride * u64::from(raw.tokens)
    {
        return Err(invalid());
    }
    // Check whole extents as well as start addresses, before any GPU call.
    raw.input_q8_1
        .checked_add(raw.input_len_bytes)
        .ok_or_else(invalid)?;
    raw.output_f32
        .checked_add(raw.output_len_bytes)
        .ok_or_else(invalid)?;
    let mut tile = *raw;
    tile.tokens = (raw.tokens - start).min(8);
    tile.input_q8_1 = raw
        .input_q8_1
        .checked_add(u64::from(start) * input_stride)
        .ok_or_else(invalid)?;
    tile.output_f32 = raw
        .output_f32
        .checked_add(u64::from(start) * output_stride)
        .ok_or_else(invalid)?;
    tile.input_len_bytes = u64::from(tile.tokens) * input_stride;
    tile.output_len_bytes = u64::from(tile.tokens) * output_stride;
    Ok(tile)
}

fn quantized_small_batch_supported(format: u32, tokens: u32) -> bool {
    (1..=8).contains(&tokens)
        && matches!(
            format,
            AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1
                | AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_NVFP4_V1
                | row_input::FORMAT
        )
}

























#[derive(Clone, Copy, Debug)]
pub struct CudaQwen35PrefillRmsnormV1 {
    raw: AugurCudaQwen35PrefillRmsnormArgsV1,
    device_ordinal: i32,
    stream_owner_instance_key: usize,
    canonical_rows: bool,
    llama_input_rows: bool,
}

impl CudaQwen35PrefillRmsnormV1 {
    /// Prevalidated descriptor for closed native programs that embed this norm.
    /// Only the default arithmetic is admitted; audit/llama variants are rejected there.
    pub fn as_raw(&self) -> &AugurCudaQwen35PrefillRmsnormArgsV1 { &self.raw }
    pub fn is_default_arithmetic(&self) -> bool { !self.canonical_rows && !self.llama_input_rows }
    /// Explicit same-DLL input arithmetic; never enabled by legacy B8/decode.
    pub fn enable_llama_exact_input_rows_audit(&mut self) -> Result<(), CudaRuntimeError> {
        if !(1..=512).contains(&self.raw.tokens) || self.raw.cols != 5120
            || self.raw.epsilon_bits != 1e-6_f32.to_bits() {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        self.canonical_rows = false;
        self.llama_input_rows = true;
        Ok(())
    }
    pub fn llama_exact_input_rows_enabled(&self) -> bool { self.llama_input_rows }
    /// Explicit <=512-row diagnostic; batches independent canonical row norms.
    /// Does not change default nor legacy eight-row verifier admission.
    pub fn enable_canonical_prefill_rows_audit(&mut self) -> Result<(), CudaRuntimeError> {
        if !(1..=512).contains(&self.raw.tokens) || self.llama_input_rows {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        self.canonical_rows = true;
        Ok(())
    }
    /// Diagnostic arithmetic alignment with the canonical runtime translation unit.
    /// No allocation or implicit promotion; bounded to the eight-row verifier.
    pub fn enable_canonical_small_batch(&mut self) -> Result<(), CudaRuntimeError> {
        if !(1..=8).contains(&self.raw.tokens) || self.llama_input_rows {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        self.canonical_rows = true;
        Ok(())
    }
    pub fn bind(
        stream: &OwnedCudaStream,
        input: CudaF32View,
        weight: CudaF32View,
        output: CudaF32View,
        tokens: usize,
        cols: usize,
        epsilon: f32,
    ) -> Result<Self, CudaRuntimeError> {
        let elements = tokens
            .checked_mul(cols)
            .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))?;
        let views = [
            input.buffer_view(),
            weight.buffer_view(),
            output.buffer_view(),
        ];
        if tokens == 0
            || tokens > AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_COMPILED_MAX_TOKENS_V1 as usize
            || cols == 0
            || input.element_count() != elements
            || weight.element_count() != cols
            || output.element_count() != elements
            || !epsilon.is_finite()
            || epsilon <= 0.0
        {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        validate_disjoint_views(stream.device_ordinal(), &views)?;

        let mut raw = AugurCudaQwen35PrefillRmsnormArgsV1::new();
        raw.tokens = u32::try_from(tokens)
            .map_err(|_| validation(CudaRuntimeErrorKind::InvalidDimensions))?;
        raw.cols =
            u32::try_from(cols).map_err(|_| validation(CudaRuntimeErrorKind::InvalidDimensions))?;
        bind_raw_view(
            &mut raw.input_f32,
            &mut raw.input_len_bytes,
            input.buffer_view(),
        )?;
        bind_raw_view(
            &mut raw.weight_f32,
            &mut raw.weight_len_bytes,
            weight.buffer_view(),
        )?;
        bind_raw_view(
            &mut raw.output_f32,
            &mut raw.output_len_bytes,
            output.buffer_view(),
        )?;
        raw.epsilon_bits = epsilon.to_bits();
        // SAFETY: Rust checked exact extents, device identity, disjointness,
        // and scalar bounds. Native preflight additionally validates live
        // allocation ranges and the SM70 device contract.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_rmsnorm_preflight_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &raw,
            )
        };
        CudaRuntimeError::from_result(OPERATION, result)?;
        Ok(Self {
            raw,
            device_ordinal: stream.device_ordinal(),
            stream_owner_instance_key: stream.owner_instance_key()?,
            canonical_rows: false,
            llama_input_rows: false,
        })
    }

    pub fn submit(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.require_stream(stream)?;
        if self.llama_input_rows {
            // SAFETY: bind verified live disjoint extents and the owner; the
            // explicit selector additionally bounds rows/width/epsilon.
            let result = unsafe {
                augur_cuda_sys::augur_cuda_qwen35_prefill_llama_input_rmsnorm_prevalidated_submit_v1(
                    self.device_ordinal, stream.raw()?, &self.raw,
                )
            };
            return CudaRuntimeError::from_result(OPERATION, result);
        }
        if self.canonical_rows {
            let mut args = augur_cuda_sys::AugurCudaRmsnormF32ArgsV1::new();
            args.input = self.raw.input_f32;
            args.input_len_bytes = self.raw.input_len_bytes;
            args.weight = self.raw.weight_f32;
            args.weight_len_bytes = self.raw.weight_len_bytes;
            args.output = self.raw.output_f32;
            args.output_len_bytes = self.raw.output_len_bytes;
            args.elements = self.raw.cols;
            args.epsilon_bits = self.raw.epsilon_bits;
            // SAFETY: bind checked live disjoint extents; admission bounds rows
            // to <=512. The caller retains the same buffers through completion.
            // Each CTA retains canonical per-row arithmetic in the original TU.
            let result = unsafe {
                augur_cuda_sys::augur_cuda_rmsnorm_f32_rows_submit_v1(
                    self.device_ordinal,
                    stream.raw()?,
                    &args,
                    self.raw.tokens,
                )
            };
            return CudaRuntimeError::from_result(OPERATION, result);
        }
        // SAFETY: bind completed native preflight. The caller retains all
        // backing buffers until a covering stream completion proof.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_rmsnorm_prevalidated_submit_v1(
                self.device_ordinal,
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result(OPERATION, result)
    }

    fn require_stream(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.device_ordinal {
            return Err(CudaRuntimeError::device_mismatch());
        }
        if stream.owner_instance_key()? != self.stream_owner_instance_key {
            return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        Ok(())
    }
}

/// B8 head producer-side last-row stores. No scratch is elided or re-aliased.
#[derive(Debug)]
pub struct CudaQwen35HeadPrepareB8V1 {
    raw: augur_cuda_sys::AugurCudaQwen35HeadPrepareB8ArgsV1,
    device_ordinal: i32,
    stream_owner: usize,
    submissions: core::cell::Cell<u64>,
}

impl CudaQwen35HeadPrepareB8V1 {
    pub fn bind_target(stream: &OwnedCudaStream, input: CudaF32View, weight: CudaF32View,
        normalized: CudaF32View, quantized: CudaQ8_1ActivationViewV1,
        last_hidden: CudaF32View, last_normalized: CudaF32View,
        last_quantized: CudaQ8_1ActivationViewV1, epsilon: f32) -> Result<Self, CudaRuntimeError> {
        if input.element_count() != 8*5120 || weight.element_count() != 5120 || normalized.element_count() != 8*5120
            || quantized.elements() != 8*5120 || last_hidden.element_count() != 5120
            || last_normalized.element_count() != 5120 || last_quantized.elements() != 5120 {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let views = [input.buffer_view(), weight.buffer_view(), normalized.buffer_view(), quantized.buffer_view(),
            last_hidden.buffer_view(), last_normalized.buffer_view(), last_quantized.buffer_view()];
        validate_disjoint_views(stream.device_ordinal(), &views)?;
        let mut a = augur_cuda_sys::AugurCudaQwen35HeadPrepareB8ArgsV1::new();
        a.mode = 1; a.epsilon_bits = epsilon.to_bits();
        bind_raw_view(&mut a.input, &mut a.input_len_bytes, views[0])?;
        bind_raw_view(&mut a.weight, &mut a.weight_len_bytes, views[1])?;
        bind_raw_view(&mut a.normalized, &mut a.normalized_len_bytes, views[2])?;
        bind_raw_view(&mut a.quantized, &mut a.quantized_len_bytes, views[3])?;
        bind_raw_view(&mut a.last_hidden, &mut a.last_hidden_len_bytes, views[4])?;
        bind_raw_view(&mut a.last_normalized, &mut a.last_normalized_len_bytes, views[5])?;
        bind_raw_view(&mut a.last_quantized, &mut a.last_quantized_len_bytes, views[6])?;
        Self::bind_raw(stream, a)
    }

    pub fn bind_draft(stream: &OwnedCudaStream, input: CudaF32View, quantized: CudaQ8_1ActivationViewV1,
        last_normalized: CudaF32View, last_quantized: CudaQ8_1ActivationViewV1) -> Result<Self, CudaRuntimeError> {
        if input.element_count() != 8*5120 || quantized.elements() != 8*5120
            || last_normalized.element_count() != 5120 || last_quantized.elements() != 5120 {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let views = [input.buffer_view(), quantized.buffer_view(), last_normalized.buffer_view(), last_quantized.buffer_view()];
        validate_disjoint_views(stream.device_ordinal(), &views)?;
        let mut a = augur_cuda_sys::AugurCudaQwen35HeadPrepareB8ArgsV1::new(); a.mode = 2;
        bind_raw_view(&mut a.input, &mut a.input_len_bytes, views[0])?;
        bind_raw_view(&mut a.quantized, &mut a.quantized_len_bytes, views[1])?;
        bind_raw_view(&mut a.last_normalized, &mut a.last_normalized_len_bytes, views[2])?;
        bind_raw_view(&mut a.last_quantized, &mut a.last_quantized_len_bytes, views[3])?;
        Self::bind_raw(stream, a)
    }

    fn bind_raw(stream: &OwnedCudaStream, raw: augur_cuda_sys::AugurCudaQwen35HeadPrepareB8ArgsV1) -> Result<Self, CudaRuntimeError> {
        let stream_owner = stream.owner_instance_key()?;
        // SAFETY: typed disjoint views retain exact live allocations. Native
        // preflight checks fixed geometry, mode and backing allocation extents.
        CudaRuntimeError::from_result(OPERATION, unsafe {
            augur_cuda_sys::augur_cuda_qwen35_head_prepare_b8_preflight_v1(stream.device_ordinal(), stream.raw()?, &raw)
        })?;
        Ok(Self { raw, device_ordinal: stream.device_ordinal(), stream_owner, submissions: core::cell::Cell::new(0) })
    }

    /// A reused draft binding must still refer to this producer's live view.
    /// This caches metadata only; input values are consumed afresh each submit.
    pub fn require_input(&self, input: CudaF32View) -> Result<(), CudaRuntimeError> {
        let mut ptr = 0; let mut bytes = 0;
        bind_raw_view(&mut ptr, &mut bytes, input.buffer_view())?;
        if input.buffer_view().device_ordinal() != self.device_ordinal || input.element_count() != 8*5120
            || ptr != self.raw.input || bytes != self.raw.input_len_bytes {
            return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        Ok(())
    }

    pub fn submit(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.device_ordinal || stream.owner_instance_key()? != self.stream_owner {
            return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        let next = self.submissions.get().checked_add(1).ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))?;
        // SAFETY: preflight proves immutable ranges, exact mode and stream owner;
        // all workspace/model/draft owners remain live through completion.
        CudaRuntimeError::from_result(OPERATION, unsafe {
            augur_cuda_sys::augur_cuda_qwen35_head_prepare_b8_submit_v1(self.device_ordinal, stream.raw()?, &self.raw)
        })?;
        self.submissions.set(next); Ok(())
    }

    pub fn eager_submissions(&self) -> u64 { self.submissions.get() }
}

#[derive(Clone, Copy, Debug)]
pub struct CudaQwen35PrefillLinearChunkViewsV1 {
    pub qkv: CudaF32View,
    pub z: CudaF32View,
    pub conv_weights: CudaF32View,
    pub conv_history: CudaF32View,
    pub alpha_raw: CudaF32View,
    pub beta_raw: CudaF32View,
    pub dt_bias: CudaF32View,
    pub ssm_a: CudaF32View,
    pub state: CudaF32View,
    pub norm_weight: CudaF32View,
    pub activated: CudaF32View,
    pub recurrent: CudaF32View,
    pub output: CudaF32View,
}

#[derive(Clone, Copy, Debug)]
pub struct CudaQwen35PrefillLinearChunkV1 {
    raw: AugurCudaQwen35PrefillLinearChunkArgsV1,
    device_ordinal: i32,
    stream_owner_instance_key: usize,
    checkpoint_sources: Option<[u64; 4]>,
}

impl CudaQwen35PrefillLinearChunkV1 {
    /// Explicit source-checkpoint math: contiguous 3:1 GDN head groups and
    /// Q/K squared-norm +1e-6. The legacy interleaved ABI remains unchanged.
    /// Legacy checkpoint-fold bindings cannot be converted to this profile.
    pub fn for_checkpoint_source(self) -> Result<CudaQwen35CheckpointPrefillLinearChunkV1, CudaRuntimeError> {
        if self.checkpoint_sources.is_some() {
            return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        Ok(CudaQwen35CheckpointPrefillLinearChunkV1 { inner: self })
    }
    /// Derive a checked prefix of this already preflighted full-B8 checkpoint
    /// binding. Pointers and checkpoints cannot change; only row-dependent
    /// extents shrink. The same caller-owned allocations must stay alive.
    /// There are no new native pointer queries, allocations or submissions.
    pub fn checkpoint_prefix_from_b8(&self, stream: &OwnedCudaStream, tokens: usize)
        -> Result<Self, CudaRuntimeError> {
        self.require_stream(stream)?;
        if self.raw.tokens != 8 || self.checkpoint_sources.is_none() || !(1..=8).contains(&tokens) {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let mut next = *self;
        macro_rules! shrink {
            ($len:ident, $width:expr) => {{
                let row_bytes = ($width as u64).checked_mul(4)
                    .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))?;
                if self.raw.$len != row_bytes * 8 {
                    return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
                }
                next.raw.$len = row_bytes * tokens as u64;
            }};
        }
        shrink!(qkv_len_bytes, AUGUR_CUDA_QWEN35_LINEAR_STEP_QKV_ELEMENTS_V1);
        shrink!(z_len_bytes, AUGUR_CUDA_QWEN35_LINEAR_STEP_OUTPUT_ELEMENTS_V1);
        shrink!(alpha_raw_len_bytes, AUGUR_CUDA_QWEN35_LINEAR_STEP_VALUE_HEADS_V1);
        shrink!(beta_raw_len_bytes, AUGUR_CUDA_QWEN35_LINEAR_STEP_VALUE_HEADS_V1);
        shrink!(activated_len_bytes, AUGUR_CUDA_QWEN35_LINEAR_STEP_QKV_ELEMENTS_V1);
        shrink!(recurrent_len_bytes, AUGUR_CUDA_QWEN35_LINEAR_STEP_OUTPUT_ELEMENTS_V1);
        shrink!(output_len_bytes, AUGUR_CUDA_QWEN35_LINEAR_STEP_OUTPUT_ELEMENTS_V1);
        next.raw.tokens = tokens as u32;
        Ok(next)
    }

    /// Bind the chunk and immutable checkpoints in one native preflight. Avoid
    /// inspecting all thirteen chunk allocations twice during request setup.
    pub fn bind_with_checkpoint_fold(
        stream: &OwnedCudaStream, views: CudaQwen35PrefillLinearChunkViewsV1,
        tokens: usize, epsilon: f32, history: CudaF32View, state: CudaF32View,
    ) -> Result<Self, CudaRuntimeError> {
        Self::bind_impl(stream, views, tokens, epsilon, Some((history, state)))
    }

    /// Bind immutable initial history/state once, without allocating. The caller
    /// retains these checkpoint owners as well as all ordinary chunk owners.
    pub fn bind_checkpoint_fold(mut self, stream: &OwnedCudaStream,
        history: CudaF32View, state: CudaF32View) -> Result<Self, CudaRuntimeError> {
        self.require_stream(stream)?;
        validate_disjoint_views(stream.device_ordinal(),
            &[history.buffer_view(), state.buffer_view()])?;
        let (mut h, mut hb, mut s, mut sb) = (0, 0, 0, 0);
        bind_raw_view(&mut h, &mut hb, history.buffer_view())?;
        bind_raw_view(&mut s, &mut sb, state.buffer_view())?;
        // SAFETY: typed device views retain their declared extents. Native
        // preflight proves exact size, device allocation and all cross-aliases.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_linear_checkpoint_fold_preflight_v1(
                self.device_ordinal, stream.raw()?, &self.raw, h, hb, s, sb)
        };
        CudaRuntimeError::from_result(OPERATION, result)?;
        self.checkpoint_sources = Some([h, hb, s, sb]);
        Ok(self)
    }

    /// Fold directly from the bound immutable checkpoint. No intermediate
    /// working-state restore, allocation, pointer query or host wait occurs.
    pub fn submit_checkpoint_fold(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.require_stream(stream)?;
        let [h, hb, s, sb] = self.checkpoint_sources
            .ok_or_else(|| validation(CudaRuntimeErrorKind::InvalidMetadata))?;
        // SAFETY: bind_checkpoint_fold validated these exact read-only sources
        // and all operands; caller keeps owners live through stream completion.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_linear_checkpoint_fold_prevalidated_submit_v1(
                self.device_ordinal, stream.raw()?, &self.raw, h, hb, s, sb)
        };
        CudaRuntimeError::from_result(OPERATION, result)
    }

    /// Replay a prefix of recorded FP32 QKV/alpha/beta into restored state.
    /// Only conv_history, activated scratch and state are written. The caller
    /// retains all bound owners and supplies records from this same stream.
    /// No query readout, gated norm, host wait or allocation is performed.
    pub fn submit_state_fold(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.require_stream(stream)?;
        // SAFETY: bind validates the same exact disjoint ranges as submit.
        // The fold only reads/writes a subset of those prevalidated ranges.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_linear_chunk_fold_prevalidated_submit_v1(
                self.device_ordinal,
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result(OPERATION, result)
    }

    /// Exact 1..8-row ReplaySSM candidate. In addition to the default state-fold
    /// writes, the first tokens*48*3 output words become dead scalar scratch.
    /// All owners must outlive the same-stream work, as for submit_state_fold.
    pub fn submit_state_fold_shared_scalars(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.require_stream(stream)?;
        if !(1..=8).contains(&self.raw.tokens) {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        // SAFETY: bind proves every exact extent and all disjoint ranges,
        // including output; this operation needs only a bounded prefix of it.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_linear_chunk_fold_shared_prevalidated_submit_v1(
                self.device_ordinal, stream.raw()?, &self.raw,
            )
        };
        CudaRuntimeError::from_result(OPERATION, result)
    }

    /// Explicit diagnostic candidate. The output extent temporarily holds four
    /// scalars per token/head, then is overwritten after the recurrent kernel
    /// completes on this same bound stream. bind's disjoint-range proof applies.
    pub fn submit_shared_scalars(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.require_stream(stream)?;
        // SAFETY: same native preflight, owners and completion lease as submit;
        // no additional pointers or allocations and no cross-stream scratch use.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_linear_chunk_shared_scalars_prevalidated_submit_v1(
                self.device_ordinal,
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result(OPERATION, result)
    }

    pub fn bind(
        stream: &OwnedCudaStream,
        views: CudaQwen35PrefillLinearChunkViewsV1,
        tokens: usize,
        epsilon: f32,
    ) -> Result<Self, CudaRuntimeError> {
        Self::bind_impl(stream, views, tokens, epsilon, None)
    }

    fn bind_impl(
        stream: &OwnedCudaStream, views: CudaQwen35PrefillLinearChunkViewsV1,
        tokens: usize, epsilon: f32, checkpoints: Option<(CudaF32View, CudaF32View)>,
    ) -> Result<Self, CudaRuntimeError> {
        let token_extent = |width: usize| {
            tokens
                .checked_mul(width)
                .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))
        };
        let expected = [
            token_extent(AUGUR_CUDA_QWEN35_LINEAR_STEP_QKV_ELEMENTS_V1)?,
            token_extent(AUGUR_CUDA_QWEN35_LINEAR_STEP_OUTPUT_ELEMENTS_V1)?,
            AUGUR_CUDA_QWEN35_LINEAR_STEP_CONV_WEIGHT_ELEMENTS_V1,
            AUGUR_CUDA_QWEN35_LINEAR_STEP_CONV_HISTORY_ELEMENTS_V1,
            token_extent(AUGUR_CUDA_QWEN35_LINEAR_STEP_VALUE_HEADS_V1)?,
            token_extent(AUGUR_CUDA_QWEN35_LINEAR_STEP_VALUE_HEADS_V1)?,
            AUGUR_CUDA_QWEN35_LINEAR_STEP_VALUE_HEADS_V1,
            AUGUR_CUDA_QWEN35_LINEAR_STEP_VALUE_HEADS_V1,
            AUGUR_CUDA_QWEN35_LINEAR_STEP_STATE_ELEMENTS_V1,
            AUGUR_CUDA_QWEN35_LINEAR_STEP_HEAD_DIM_V1,
            token_extent(AUGUR_CUDA_QWEN35_LINEAR_STEP_QKV_ELEMENTS_V1)?,
            token_extent(AUGUR_CUDA_QWEN35_LINEAR_STEP_OUTPUT_ELEMENTS_V1)?,
            token_extent(AUGUR_CUDA_QWEN35_LINEAR_STEP_OUTPUT_ELEMENTS_V1)?,
        ];
        let typed = [
            views.qkv,
            views.z,
            views.conv_weights,
            views.conv_history,
            views.alpha_raw,
            views.beta_raw,
            views.dt_bias,
            views.ssm_a,
            views.state,
            views.norm_weight,
            views.activated,
            views.recurrent,
            views.output,
        ];
        if tokens == 0
            || tokens > AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_COMPILED_MAX_TOKENS_V1 as usize
            || !epsilon.is_finite()
            || epsilon <= 0.0
            || typed
                .iter()
                .zip(expected)
                .any(|(view, elements)| view.element_count() != elements)
        {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let raw_views = typed.map(CudaF32View::buffer_view);
        validate_disjoint_views(stream.device_ordinal(), &raw_views)?;

        let mut raw = AugurCudaQwen35PrefillLinearChunkArgsV1::new();
        raw.tokens = u32::try_from(tokens)
            .map_err(|_| validation(CudaRuntimeErrorKind::InvalidDimensions))?;
        macro_rules! bind {
            ($ptr:ident, $len:ident, $view:expr) => {
                bind_raw_view(&mut raw.$ptr, &mut raw.$len, $view.buffer_view())?
            };
        }
        bind!(qkv_f32, qkv_len_bytes, views.qkv);
        bind!(z_f32, z_len_bytes, views.z);
        bind!(conv_weights_f32, conv_weights_len_bytes, views.conv_weights);
        bind!(conv_history_f32, conv_history_len_bytes, views.conv_history);
        bind!(alpha_raw_f32, alpha_raw_len_bytes, views.alpha_raw);
        bind!(beta_raw_f32, beta_raw_len_bytes, views.beta_raw);
        bind!(dt_bias_f32, dt_bias_len_bytes, views.dt_bias);
        bind!(ssm_a_f32, ssm_a_len_bytes, views.ssm_a);
        bind!(state_f32, state_len_bytes, views.state);
        bind!(norm_weight_f32, norm_weight_len_bytes, views.norm_weight);
        bind!(activated_f32, activated_len_bytes, views.activated);
        bind!(recurrent_f32, recurrent_len_bytes, views.recurrent);
        bind!(output_f32, output_len_bytes, views.output);
        raw.epsilon_bits = epsilon.to_bits();
        let bound = Self {
            raw,
            device_ordinal: stream.device_ordinal(),
            stream_owner_instance_key: stream.owner_instance_key()?,
            checkpoint_sources: None,
        };
        if let Some((history, state)) = checkpoints {
            // This preflight validates ALL chunk operands plus checkpoints.
            // No weakened trust in uninspected pointers, just one inspection.
            return bound.bind_checkpoint_fold(stream, history, state);
        }
        // SAFETY: Rust establishes exact fixed-profile extents and disjoint
        // device views. Native preflight pins the allocation ranges.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_linear_chunk_preflight_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &raw,
            )
        };
        CudaRuntimeError::from_result(OPERATION, result)?;
        Ok(bound)
    }

    pub fn submit(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.require_stream(stream)?;
        // SAFETY: bind completed native preflight and all referenced owners
        // remain live until the caller's stream completion proof.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_linear_chunk_prevalidated_submit_v1(
                self.device_ordinal,
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result(OPERATION, result)
    }

    /// Explicit numerical-contract candidate. Uses a device-side higher
    /// precision recurrent accumulation while preserving the established
    /// FP32 state/output ABI. This is never selected implicitly.
    pub fn submit_precise(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.require_stream(stream)?;
        // SAFETY: bind completed native preflight and all referenced owners
        // remain live until the caller's stream completion proof.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_linear_chunk_precise_prevalidated_submit_v1(
                self.device_ordinal,
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result(OPERATION, result)
    }

    fn require_stream(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.device_ordinal {
            return Err(CudaRuntimeError::device_mismatch());
        }
        if stream.owner_instance_key()? != self.stream_owner_instance_key {
            return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        Ok(())
    }
}

/// Caller retains all thirteen owners until stream completion, as for the
/// legacy chunk. Distinct wrapper prevents accidental legacy submit dispatch.
#[derive(Clone, Copy, Debug)]
pub struct CudaQwen35CheckpointPrefillLinearChunkV1 {
    inner: CudaQwen35PrefillLinearChunkV1,
}
impl CudaQwen35CheckpointPrefillLinearChunkV1 {
    pub const PROFILE_LABEL: &'static str = "quasar_checkpoint_contiguous3_gdn_shared_scalars_f32_v1";
    pub fn submit(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.inner.require_stream(stream)?;
        // SAFETY: identical validated geometry/owners, separate source math.
        CudaRuntimeError::from_result(OPERATION, unsafe {
            augur_cuda_sys::augur_cuda_qwen35_checkpoint_prefill_linear_chunk_prevalidated_submit_v1(
                self.inner.device_ordinal, stream.raw()?, &self.inner.raw)
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CudaQwen35PrefillFullAttentionViewsV1 {
    pub q_full: CudaF32View,
    pub k_raw: CudaF32View,
    pub value: CudaF32View,
    pub q_norm_weight: CudaF32View,
    pub k_norm_weight: CudaF32View,
    pub rope_cos: CudaF32View,
    pub rope_sin: CudaF32View,
    pub key_cache: CudaF32View,
    pub value_cache: CudaF32View,
    pub query: CudaF32View,
    pub gate: CudaF32View,
    pub attention: CudaF32View,
    pub gated: CudaF32View,
}

/// Explicit canonical-score research execution; no implicit global switch.
pub struct CudaNerkyorCanonicalF32KvAttentionV1 { inner: CudaQwen35PrefillFullAttentionV1 }
impl CudaNerkyorCanonicalF32KvAttentionV1 {
    pub fn submit(&self, stream:&OwnedCudaStream)->Result<(),CudaRuntimeError> {
        self.inner.require_stream(stream)?;
        // SAFETY: immutable checked descriptor; caller retains every buffer and
        // one stream owner through completion, as for the existing prefill ABI.
        let result=unsafe {augur_cuda_sys::augur_cuda_nerkyor_prefill_full_attention_canonical_submit_v1(
            self.inner.device_ordinal,stream.raw()?,&self.inner.raw)};
        CudaRuntimeError::from_result("nerkyor_canonical_f32_kv_attention",result)
    }
}
impl CudaQwen35PrefillFullAttentionV1 {
    pub fn for_nerkyor_canonical_f32_kv(self)->Result<CudaNerkyorCanonicalF32KvAttentionV1,CudaRuntimeError> {
        if self.raw.tokens>8 || !cfg!(feature="qwen35-warp-full-attention") || !cfg!(feature="qwen35-target-split-rope") {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        Ok(CudaNerkyorCanonicalF32KvAttentionV1{inner:self})
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CudaQwen35PrefillFullAttentionV1 {
    raw: AugurCudaQwen35PrefillFullAttentionArgsV1,
    device_ordinal: i32,
    stream_owner_instance_key: usize,
}

impl CudaQwen35PrefillFullAttentionV1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        views: CudaQwen35PrefillFullAttentionViewsV1,
        tokens: usize,
        position_base: usize,
        capacity: usize,
        epsilon: f32,
    ) -> Result<Self, CudaRuntimeError> {
        let token_extent = |width: usize| {
            tokens
                .checked_mul(width)
                .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))
        };
        let cache_extent = capacity
            .checked_mul(AUGUR_CUDA_QWEN35_FULL_ATTENTION_KV_ELEMENTS_V1)
            .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))?;
        let expected = [
            token_extent(AUGUR_CUDA_QWEN35_FULL_ATTENTION_QUERY_ELEMENTS_V1 * 2)?,
            token_extent(AUGUR_CUDA_QWEN35_FULL_ATTENTION_KV_ELEMENTS_V1)?,
            token_extent(AUGUR_CUDA_QWEN35_FULL_ATTENTION_KV_ELEMENTS_V1)?,
            AUGUR_CUDA_QWEN35_FULL_ATTENTION_HEAD_DIM_V1,
            AUGUR_CUDA_QWEN35_FULL_ATTENTION_HEAD_DIM_V1,
            token_extent(AUGUR_CUDA_QWEN35_FULL_ATTENTION_ROTARY_DIM_V1 / 2)?,
            token_extent(AUGUR_CUDA_QWEN35_FULL_ATTENTION_ROTARY_DIM_V1 / 2)?,
            cache_extent,
            cache_extent,
            token_extent(AUGUR_CUDA_QWEN35_FULL_ATTENTION_QUERY_ELEMENTS_V1)?,
            token_extent(AUGUR_CUDA_QWEN35_FULL_ATTENTION_QUERY_ELEMENTS_V1)?,
            token_extent(AUGUR_CUDA_QWEN35_FULL_ATTENTION_QUERY_ELEMENTS_V1)?,
            token_extent(AUGUR_CUDA_QWEN35_FULL_ATTENTION_QUERY_ELEMENTS_V1)?,
        ];
        let typed = [
            views.q_full,
            views.k_raw,
            views.value,
            views.q_norm_weight,
            views.k_norm_weight,
            views.rope_cos,
            views.rope_sin,
            views.key_cache,
            views.value_cache,
            views.query,
            views.gate,
            views.attention,
            views.gated,
        ];
        let end = position_base
            .checked_add(tokens)
            .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))?;
        if tokens == 0
            || tokens > AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_COMPILED_MAX_TOKENS_V1 as usize
            || capacity == 0
            || capacity > AUGUR_CUDA_QWEN35_FULL_ATTENTION_MAX_CAPACITY_V1
            || end > capacity
            || !epsilon.is_finite()
            || epsilon <= 0.0
            || typed
                .iter()
                .zip(expected)
                .any(|(view, elements)| view.element_count() != elements)
        {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let raw_views = typed.map(CudaF32View::buffer_view);
        validate_disjoint_views(stream.device_ordinal(), &raw_views)?;

        let mut raw = AugurCudaQwen35PrefillFullAttentionArgsV1::new();
        macro_rules! bind {
            ($ptr:ident, $len:ident, $view:expr) => {
                bind_raw_view(&mut raw.$ptr, &mut raw.$len, $view.buffer_view())?
            };
        }
        bind!(q_full_f32, q_full_len_bytes, views.q_full);
        bind!(k_raw_f32, k_raw_len_bytes, views.k_raw);
        bind!(value_f32, value_len_bytes, views.value);
        bind!(
            q_norm_weight_f32,
            q_norm_weight_len_bytes,
            views.q_norm_weight
        );
        bind!(
            k_norm_weight_f32,
            k_norm_weight_len_bytes,
            views.k_norm_weight
        );
        bind!(rope_cos_f32, rope_cos_len_bytes, views.rope_cos);
        bind!(rope_sin_f32, rope_sin_len_bytes, views.rope_sin);
        bind!(key_cache_f32, key_cache_len_bytes, views.key_cache);
        bind!(value_cache_f32, value_cache_len_bytes, views.value_cache);
        bind!(query_f32, query_len_bytes, views.query);
        bind!(gate_f32, gate_len_bytes, views.gate);
        bind!(attention_f32, attention_len_bytes, views.attention);
        bind!(gated_f32, gated_len_bytes, views.gated);
        raw.tokens = u32::try_from(tokens)
            .map_err(|_| validation(CudaRuntimeErrorKind::InvalidDimensions))?;
        raw.position_base = u32::try_from(position_base)
            .map_err(|_| validation(CudaRuntimeErrorKind::InvalidDimensions))?;
        raw.capacity = u32::try_from(capacity)
            .map_err(|_| validation(CudaRuntimeErrorKind::InvalidDimensions))?;
        raw.query_heads = AUGUR_CUDA_QWEN35_FULL_ATTENTION_QUERY_HEADS_V1 as u32;
        raw.kv_heads = AUGUR_CUDA_QWEN35_FULL_ATTENTION_KV_HEADS_V1 as u32;
        raw.head_dim = AUGUR_CUDA_QWEN35_FULL_ATTENTION_HEAD_DIM_V1 as u32;
        raw.rotary_dim = AUGUR_CUDA_QWEN35_FULL_ATTENTION_ROTARY_DIM_V1 as u32;
        raw.epsilon_bits = epsilon.to_bits();
        // SAFETY: exact extents, disjoint device ranges, causal bounds, and
        // scalar geometry are checked above. Native preflight pins the ranges.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_full_attention_preflight_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &raw,
            )
        };
        CudaRuntimeError::from_result(OPERATION, result)?;
        Ok(Self {
            raw,
            device_ordinal: stream.device_ordinal(),
            stream_owner_instance_key: stream.owner_instance_key()?,
        })
    }

    pub fn submit(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.require_stream(stream)?;
        // SAFETY: bind completed native preflight. All owners remain live
        // until the caller obtains a covering stream completion proof.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_full_attention_prevalidated_submit_v1(
                self.device_ordinal,
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result(OPERATION, result)
    }

    pub const fn raw(&self) -> &AugurCudaQwen35PrefillFullAttentionArgsV1 {
        &self.raw
    }

    fn require_stream(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.device_ordinal {
            return Err(CudaRuntimeError::device_mismatch());
        }
        if stream.owner_instance_key()? != self.stream_owner_instance_key {
            return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        Ok(())
    }
}

#[cfg(all(
    feature = "qwen35-prefill-cublas",
    feature = "qwen35-kv8-full-attention"
))]
#[derive(Clone, Copy, Debug)]
pub struct CudaQwen35PrefillKv8ChunkAttentionViewsV1 {
    pub q_full: CudaF32View,
    pub k_raw: CudaF32View,
    pub value: CudaF32View,
    pub q_norm_weight: CudaF32View,
    pub k_norm_weight: CudaF32View,
    pub rope_cos: CudaF32View,
    pub rope_sin: CudaF32View,
    pub key_work: CudaF32View,
    pub value_work: CudaF32View,
    pub key_cache: CudaBufferView,
    pub value_cache: CudaBufferView,
    pub query: CudaF32View,
    pub gate: CudaF32View,
    pub attention: CudaF32View,
    pub gated: CudaF32View,
}

#[cfg(all(
    feature = "qwen35-prefill-cublas",
    feature = "qwen35-kv8-full-attention"
))]
#[derive(Clone, Copy, Debug)]
pub struct CudaQwen35PrefillKv8ChunkAttentionV1 {
    raw: AugurCudaQwen35PrefillKv8ChunkAttentionArgsV1,
    device_ordinal: i32,
    stream_owner_instance_key: usize,
    score_workspace: Option<(u64, u64, bool)>,
    dgemm_workspace: Option<(u64, u64, usize)>,
}

#[cfg(all(
    feature = "qwen35-prefill-cublas",
    feature = "qwen35-kv8-full-attention"
))]
impl CudaQwen35PrefillKv8ChunkAttentionV1 {
    pub fn dgemm_score_elements(tokens: usize, capacity: usize) -> Result<usize, CudaRuntimeError> {
        let scores = Self::tiled_score_elements(tokens, capacity)?;
        let rows = tokens.min(128);
        let keys = capacity.min(8192);
        // Shapes were bounded by tiled_score_elements; double staging is
        // counted as two F32 elements solely for this byte-extent interface.
        Ok(scores + 2 * (rows * 6 * 256 + keys * 256 + rows * 6 * keys))
    }

    pub fn with_dgemm_scores(
        self,
        stream: &OwnedCudaStream,
        plan: &CudaQwen35PrefillCublasPlanV1,
        scores: CudaF32View,
    ) -> Result<Self, CudaRuntimeError> {
        self.require_stream(stream)?;
        plan.require_stream(stream)?;
        let elements =
            Self::dgemm_score_elements(self.raw.tokens as usize, self.raw.capacity as usize)?;
        let view = scores.buffer_view();
        if scores.element_count() != elements || view.len_bytes() != elements * 4 {
            return Err(kv8_chunk_validation(
                CudaRuntimeErrorKind::InvalidDimensions,
            ));
        }
        validate_disjoint_views(self.device_ordinal, &[view])?;
        let address = view.checked_address(KV8_CHUNK_OPERATION)?;
        let bytes = view.len_bytes() as u64;
        // SAFETY: native preflight verifies alignment, live extents and
        // disjointness from base views and both plan-owned workspaces.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_kv8_tiled_dgemm_preflight_v1(
                self.device_ordinal,
                stream.raw()?,
                plan.raw,
                &self.raw,
                address,
                bytes,
            )
        };
        CudaRuntimeError::from_result(KV8_CHUNK_OPERATION, result)?;
        Ok(Self {
            dgemm_workspace: Some((address, bytes, plan.raw as usize)),
            ..self
        })
    }

    pub fn submit_tiled_dgemm(
        &self,
        stream: &OwnedCudaStream,
        plan: &CudaQwen35PrefillCublasPlanV1,
    ) -> Result<(), CudaRuntimeError> {
        self.require_stream(stream)?;
        plan.require_stream(stream)?;
        let (address, bytes, plan_key) = self
            .dgemm_workspace
            .ok_or_else(|| kv8_chunk_validation(CudaRuntimeErrorKind::InvalidMetadata))?;
        if plan_key != plan.raw as usize {
            return Err(kv8_chunk_validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        // SAFETY: exact scratch/base views and plan are preflighted and bound
        // to this owner; caller keeps all owners live until terminal completion.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_kv8_tiled_dgemm_prevalidated_submit_v1(
                self.device_ordinal,
                stream.raw()?,
                plan.raw,
                &self.raw,
                address,
                bytes,
            )
        };
        CudaRuntimeError::from_result(KV8_CHUNK_OPERATION, result)
    }

    /// Exact-sized, bounded score workspace for the explicit two-stage experiment.
    pub fn staged_score_elements(
        tokens: usize,
        capacity: usize,
    ) -> Result<usize, CudaRuntimeError> {
        if tokens == 0
            || tokens > AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_COMPILED_MAX_TOKENS_V1 as usize
            || capacity == 0
            || capacity > AUGUR_CUDA_QWEN35_KV8_FULL_ATTENTION_MAX_CAPACITY_V3
        {
            return Err(kv8_chunk_validation(
                CudaRuntimeErrorKind::InvalidDimensions,
            ));
        }
        tokens
            .min(16)
            .checked_mul(AUGUR_CUDA_QWEN35_FULL_ATTENTION_QUERY_HEADS_V1)
            .and_then(|n| n.checked_mul(capacity))
            .ok_or_else(|| kv8_chunk_validation(CudaRuntimeErrorKind::RangeOverflow))
    }

    pub fn tiled_score_elements(tokens: usize, capacity: usize) -> Result<usize, CudaRuntimeError> {
        Self::staged_score_elements(tokens, capacity)?; // shared geometry validation
        tokens
            .min(128)
            .checked_mul(AUGUR_CUDA_QWEN35_FULL_ATTENTION_QUERY_HEADS_V1)
            .and_then(|n| n.checked_mul(capacity.min(8192) + 2))
            .ok_or_else(|| kv8_chunk_validation(CudaRuntimeErrorKind::RangeOverflow))
    }

    pub fn with_tiled_scores(
        self,
        stream: &OwnedCudaStream,
        scores: CudaF32View,
    ) -> Result<Self, CudaRuntimeError> {
        self.with_score_workspace(stream, scores, true)
    }

    /// Bind caller-owned scratch; its owner must remain live until covering
    /// stream completion, and may only be reused in this same ordered stream.
    pub fn with_staged_scores(
        self,
        stream: &OwnedCudaStream,
        scores: CudaF32View,
    ) -> Result<Self, CudaRuntimeError> {
        self.with_score_workspace(stream, scores, false)
    }

    fn with_score_workspace(
        self,
        stream: &OwnedCudaStream,
        scores: CudaF32View,
        tiled: bool,
    ) -> Result<Self, CudaRuntimeError> {
        self.require_stream(stream)?;
        let elements = if tiled {
            Self::tiled_score_elements(self.raw.tokens as usize, self.raw.capacity as usize)?
        } else {
            Self::staged_score_elements(self.raw.tokens as usize, self.raw.capacity as usize)?
        };
        if scores.element_count() != elements || scores.buffer_view().len_bytes() != elements * 4 {
            return Err(kv8_chunk_validation(
                CudaRuntimeErrorKind::InvalidDimensions,
            ));
        }
        validate_disjoint_views(self.device_ordinal, &[scores.buffer_view()])?;
        let address = scores.buffer_view().checked_address(KV8_CHUNK_OPERATION)?;
        let bytes = scores.buffer_view().len_bytes() as u64;
        // SAFETY: native preflight checks all15 input/output ranges against the
        // exact scratch extent, allocation bounds, device and stream identity.
        let result = unsafe {
            if tiled {
                augur_cuda_sys::augur_cuda_qwen35_prefill_kv8_tiled_scores_preflight_v1(
                    self.device_ordinal,
                    stream.raw()?,
                    &self.raw,
                    address,
                    bytes,
                )
            } else {
                augur_cuda_sys::augur_cuda_qwen35_prefill_kv8_staged_scores_preflight_v1(
                    self.device_ordinal,
                    stream.raw()?,
                    &self.raw,
                    address,
                    bytes,
                )
            }
        };
        CudaRuntimeError::from_result(KV8_CHUNK_OPERATION, result)?;
        Ok(Self {
            score_workspace: Some((address, bytes, tiled)),
            ..self
        })
    }

    pub fn submit_staged_scores(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.submit_score_workspace(stream, false)
    }

    pub fn submit_tiled_scores(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.submit_score_workspace(stream, true)
    }

    /// Same bounded score/state ABI; the score CTA caches exact FP64 queries.
    pub fn submit_tiled_cached_query(
        &self,
        stream: &OwnedCudaStream,
    ) -> Result<(), CudaRuntimeError> {
        self.require_stream(stream)?;
        let (address, bytes, tiled) = self
            .score_workspace
            .ok_or_else(|| kv8_chunk_validation(CudaRuntimeErrorKind::InvalidMetadata))?;
        if !tiled {
            return Err(kv8_chunk_validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        // SAFETY: same preflighted tiled scratch/base extents and stream owner;
        // only block-local query storage differs from submit_tiled_scores.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_kv8_tiled_cached_query_prevalidated_submit_v1(
                self.device_ordinal,
                stream.raw()?,
                &self.raw,
                address,
                bytes,
            )
        };
        CudaRuntimeError::from_result(KV8_CHUNK_OPERATION, result)
    }

    fn submit_score_workspace(
        &self,
        stream: &OwnedCudaStream,
        tiled: bool,
    ) -> Result<(), CudaRuntimeError> {
        self.require_stream(stream)?;
        let (address, bytes, bound_tiled) = self
            .score_workspace
            .ok_or_else(|| kv8_chunk_validation(CudaRuntimeErrorKind::InvalidMetadata))?;
        if bound_tiled != tiled {
            return Err(kv8_chunk_validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        // SAFETY: scratch and base views were preflighted on this owner;
        // no allocations, host copies or synchronization are introduced here.
        let result = unsafe {
            if tiled {
                augur_cuda_sys::augur_cuda_qwen35_prefill_kv8_tiled_scores_prevalidated_submit_v1(
                    self.device_ordinal,
                    stream.raw()?,
                    &self.raw,
                    address,
                    bytes,
                )
            } else {
                augur_cuda_sys::augur_cuda_qwen35_prefill_kv8_staged_scores_prevalidated_submit_v1(
                    self.device_ordinal,
                    stream.raw()?,
                    &self.raw,
                    address,
                    bytes,
                )
            }
        };
        CudaRuntimeError::from_result(KV8_CHUNK_OPERATION, result)
    }

    /// Shared tile8 candidate: exact unit scale only for finite unchanged maxima.
    pub fn submit_shared_softmax_unit_scale(
        &self,
        stream: &OwnedCudaStream,
    ) -> Result<(), CudaRuntimeError> {
        self.require_stream(stream)?;
        // SAFETY: identical prevalidated views/owner; no layout or lifetime change.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_kv8_chunk_shared_softmax_unit_scale_prevalidated_submit_v1(
                self.device_ordinal, stream.raw()?, &self.raw,
            )
        };
        CudaRuntimeError::from_result(KV8_CHUNK_OPERATION, result)
    }

    /// Bounded tile16 variant; retains ordered FP64 QK and FP32 softmax/output.
    pub fn submit_shared_softmax_tile16(
        &self,
        stream: &OwnedCudaStream,
    ) -> Result<(), CudaRuntimeError> {
        self.require_stream(stream)?;
        // SAFETY: identical validated ranges and stream lease; only CTA shared
        // tile geometry changes. No persistent cache layout or owner mutation.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_kv8_chunk_shared_softmax_tile16_prevalidated_submit_v1(
                self.device_ordinal,
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result(KV8_CHUNK_OPERATION, result)
    }

    pub fn bind(
        stream: &OwnedCudaStream,
        views: CudaQwen35PrefillKv8ChunkAttentionViewsV1,
        tokens: usize,
        position_base: usize,
        capacity: usize,
        epsilon: f32,
    ) -> Result<Self, CudaRuntimeError> {
        let raw = bind_prefill_kv8_chunk_attention_raw(
            stream.device_ordinal(),
            views,
            tokens,
            position_base,
            capacity,
            epsilon,
        )?;
        // SAFETY: Rust binding checks exact chunk/cache extents, alignment,
        // device ownership and non-aliasing. Native preflight repeats the
        // live allocation and SM70 checks without launching work.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_kv8_chunk_attention_preflight_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &raw,
            )
        };
        CudaRuntimeError::from_result(KV8_CHUNK_OPERATION, result)?;
        Ok(Self {
            raw,
            device_ordinal: stream.device_ordinal(),
            stream_owner_instance_key: stream.owner_instance_key()?,
            score_workspace: None,
            dgemm_workspace: None,
        })
    }

    pub fn submit(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.require_stream(stream)?;
        // SAFETY: bind completed native preflight; all referenced allocations
        // remain live until a terminal completion proof is obtained.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_kv8_chunk_attention_prevalidated_submit_v1(
                self.device_ordinal,
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result(KV8_CHUNK_OPERATION, result)
    }

    /// Opt-in SIMT candidate; the ordinary submit remains the control.
    pub fn submit_shared_softmax(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.require_stream(stream)?;
        // SAFETY: same preflighted, live views and exact owner as submit().
        // Only the head-scalar softmax coefficient computation is shared.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_kv8_chunk_shared_softmax_prevalidated_submit_v1(
                self.device_ordinal,
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result(KV8_CHUNK_OPERATION, result)
    }

    /// Opt-in storage-only candidate: one register-indexable softmax state per
    /// owning thread. Same tile8, FP64 QK and ordered FP32 recurrence as shared.
    /// This entry does not select the candidate in any default request path.
    pub fn submit_register_softmax(
        &self,
        stream: &OwnedCudaStream,
    ) -> Result<(), CudaRuntimeError> {
        self.require_stream(stream)?;
        // SAFETY: identical preflighted views and covering owner/completion
        // lifetime as submit_shared_softmax; the candidate allocates no buffer.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_kv8_chunk_register_softmax_prevalidated_submit_v1(
                self.device_ordinal,
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result(KV8_CHUNK_OPERATION, result)
    }

    /// Explicit research candidate: one warp owns a query head and preserves
    /// the reference's FP64 reduction and ordered FP32 recurrence. No default
    /// runtime selection and no additional workspace allocation.
    pub fn submit_warp_ordered(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.require_stream(stream)?;
        // SAFETY: same preflighted views and owner/terminal lifetime as the
        // retained shared-softmax path; only native work ownership changes.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_kv8_chunk_warp_ordered_prevalidated_submit_v1(
                self.device_ordinal,
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result(KV8_CHUNK_OPERATION, result)
    }

    /// Explicit component candidate retaining cooperative KV tile decode while
    /// each warp owns one head's exact ordered attention recurrence.
    pub fn submit_warp_shared_kv(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.require_stream(stream)?;
        // SAFETY: identical preflighted views/owner and terminal lifetime;
        // two block barriers protect each cooperatively decoded KV tile.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_kv8_chunk_warp_shared_kv_prevalidated_submit_v1(
                self.device_ordinal,
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result(KV8_CHUNK_OPERATION, result)
    }

    /// Explicit candidate: parallel tile-prefix maxima and expf coefficients,
    /// retaining the control's ordered denominator/output fold and FP64 QK.
    /// The finite numerical contract must pass before any request selection.
    pub fn submit_parallel_scales(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.require_stream(stream)?;
        // SAFETY: same preflighted buffer extents and terminal owner lifetime;
        // native shared-memory phases are separated by block barriers.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_kv8_chunk_parallel_scales_prevalidated_submit_v1(
                self.device_ordinal,
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result(KV8_CHUNK_OPERATION, result)
    }







    fn require_stream(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.device_ordinal
            || stream.owner_instance_key()? != self.stream_owner_instance_key
        {
            return Err(kv8_chunk_validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        Ok(())
    }

    pub const fn raw(&self) -> &AugurCudaQwen35PrefillKv8ChunkAttentionArgsV1 {
        &self.raw
    }
}

#[cfg(all(
    feature = "qwen35-prefill-cublas",
    feature = "qwen35-kv8-full-attention"
))]
fn bind_prefill_kv8_chunk_attention_raw(
    device_ordinal: i32,
    views: CudaQwen35PrefillKv8ChunkAttentionViewsV1,
    tokens: usize,
    position_base: usize,
    capacity: usize,
    epsilon: f32,
) -> Result<AugurCudaQwen35PrefillKv8ChunkAttentionArgsV1, CudaRuntimeError> {
    let f32_extent = |elements: usize| {
        elements
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or_else(|| kv8_chunk_validation(CudaRuntimeErrorKind::RangeOverflow))
    };
    let token_f32_extent = |width: usize| {
        tokens
            .checked_mul(width)
            .and_then(|elements| elements.checked_mul(core::mem::size_of::<f32>()))
            .ok_or_else(|| kv8_chunk_validation(CudaRuntimeErrorKind::RangeOverflow))
    };
    let cache_bytes = capacity
        .checked_mul(AUGUR_CUDA_QWEN35_KV8_CACHE_ROW_BYTES_V3)
        .ok_or_else(|| kv8_chunk_validation(CudaRuntimeErrorKind::RangeOverflow))?;
    let end = position_base
        .checked_add(tokens)
        .ok_or_else(|| kv8_chunk_validation(CudaRuntimeErrorKind::RangeOverflow))?;
    let expected = [
        token_f32_extent(AUGUR_CUDA_QWEN35_FULL_ATTENTION_QUERY_ELEMENTS_V1 * 2)?,
        token_f32_extent(AUGUR_CUDA_QWEN35_FULL_ATTENTION_KV_ELEMENTS_V1)?,
        token_f32_extent(AUGUR_CUDA_QWEN35_FULL_ATTENTION_KV_ELEMENTS_V1)?,
        f32_extent(AUGUR_CUDA_QWEN35_FULL_ATTENTION_HEAD_DIM_V1)?,
        f32_extent(AUGUR_CUDA_QWEN35_FULL_ATTENTION_HEAD_DIM_V1)?,
        token_f32_extent(AUGUR_CUDA_QWEN35_FULL_ATTENTION_ROTARY_DIM_V1 / 2)?,
        token_f32_extent(AUGUR_CUDA_QWEN35_FULL_ATTENTION_ROTARY_DIM_V1 / 2)?,
        token_f32_extent(AUGUR_CUDA_QWEN35_FULL_ATTENTION_KV_ELEMENTS_V1)?,
        token_f32_extent(AUGUR_CUDA_QWEN35_FULL_ATTENTION_KV_ELEMENTS_V1)?,
        cache_bytes,
        cache_bytes,
        token_f32_extent(AUGUR_CUDA_QWEN35_FULL_ATTENTION_QUERY_ELEMENTS_V1)?,
        token_f32_extent(AUGUR_CUDA_QWEN35_FULL_ATTENTION_QUERY_ELEMENTS_V1)?,
        token_f32_extent(AUGUR_CUDA_QWEN35_FULL_ATTENTION_QUERY_ELEMENTS_V1)?,
        token_f32_extent(AUGUR_CUDA_QWEN35_FULL_ATTENTION_QUERY_ELEMENTS_V1)?,
    ];
    let typed = [
        views.q_full.buffer_view(),
        views.k_raw.buffer_view(),
        views.value.buffer_view(),
        views.q_norm_weight.buffer_view(),
        views.k_norm_weight.buffer_view(),
        views.rope_cos.buffer_view(),
        views.rope_sin.buffer_view(),
        views.key_work.buffer_view(),
        views.value_work.buffer_view(),
        views.key_cache,
        views.value_cache,
        views.query.buffer_view(),
        views.gate.buffer_view(),
        views.attention.buffer_view(),
        views.gated.buffer_view(),
    ];
    if tokens == 0
        || tokens > AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_COMPILED_MAX_TOKENS_V1 as usize
        || capacity == 0
        || capacity > AUGUR_CUDA_QWEN35_KV8_FULL_ATTENTION_MAX_CAPACITY_V3
        || end > capacity
        || !epsilon.is_finite()
        || epsilon <= 0.0
        || typed
            .iter()
            .zip(expected)
            .any(|(view, bytes)| view.len_bytes() != bytes)
    {
        return Err(kv8_chunk_validation(
            CudaRuntimeErrorKind::InvalidDimensions,
        ));
    }
    for cache in [views.key_cache, views.value_cache] {
        if cache.checked_address(KV8_CHUNK_OPERATION)? % core::mem::align_of::<u16>() as u64 != 0 {
            return Err(kv8_chunk_validation(
                CudaRuntimeErrorKind::MisalignedDeviceAddress,
            ));
        }
    }
    validate_disjoint_views(device_ordinal, &typed)?;

    let mut raw = AugurCudaQwen35PrefillKv8ChunkAttentionArgsV1::new();
    macro_rules! bind {
        ($ptr:expr, $len:expr, $view:expr) => {
            *$ptr = $view.checked_address(KV8_CHUNK_OPERATION)?;
            *$len = u64::try_from($view.len_bytes())
                .map_err(|_| kv8_chunk_validation(CudaRuntimeErrorKind::RangeOverflow))?;
        };
    }
    bind!(
        &mut raw.q_full_f32,
        &mut raw.q_full_len_bytes,
        views.q_full.buffer_view()
    );
    bind!(
        &mut raw.k_raw_f32,
        &mut raw.k_raw_len_bytes,
        views.k_raw.buffer_view()
    );
    bind!(
        &mut raw.value_f32,
        &mut raw.value_len_bytes,
        views.value.buffer_view()
    );
    bind!(
        &mut raw.q_norm_weight_f32,
        &mut raw.q_norm_weight_len_bytes,
        views.q_norm_weight.buffer_view()
    );
    bind!(
        &mut raw.k_norm_weight_f32,
        &mut raw.k_norm_weight_len_bytes,
        views.k_norm_weight.buffer_view()
    );
    bind!(
        &mut raw.rope_cos_f32,
        &mut raw.rope_cos_len_bytes,
        views.rope_cos.buffer_view()
    );
    bind!(
        &mut raw.rope_sin_f32,
        &mut raw.rope_sin_len_bytes,
        views.rope_sin.buffer_view()
    );
    bind!(
        &mut raw.key_work_f32,
        &mut raw.key_work_len_bytes,
        views.key_work.buffer_view()
    );
    bind!(
        &mut raw.value_work_f32,
        &mut raw.value_work_len_bytes,
        views.value_work.buffer_view()
    );
    bind!(
        &mut raw.key_cache_q8_0,
        &mut raw.key_cache_len_bytes,
        views.key_cache
    );
    bind!(
        &mut raw.value_cache_q8_0,
        &mut raw.value_cache_len_bytes,
        views.value_cache
    );
    bind!(
        &mut raw.query_f32,
        &mut raw.query_len_bytes,
        views.query.buffer_view()
    );
    bind!(
        &mut raw.gate_f32,
        &mut raw.gate_len_bytes,
        views.gate.buffer_view()
    );
    bind!(
        &mut raw.attention_f32,
        &mut raw.attention_len_bytes,
        views.attention.buffer_view()
    );
    bind!(
        &mut raw.gated_f32,
        &mut raw.gated_len_bytes,
        views.gated.buffer_view()
    );
    raw.tokens = u32::try_from(tokens)
        .map_err(|_| kv8_chunk_validation(CudaRuntimeErrorKind::InvalidDimensions))?;
    raw.position_base = u32::try_from(position_base)
        .map_err(|_| kv8_chunk_validation(CudaRuntimeErrorKind::InvalidDimensions))?;
    raw.capacity = u32::try_from(capacity)
        .map_err(|_| kv8_chunk_validation(CudaRuntimeErrorKind::InvalidDimensions))?;
    raw.query_heads = AUGUR_CUDA_QWEN35_FULL_ATTENTION_QUERY_HEADS_V1 as u32;
    raw.kv_heads = AUGUR_CUDA_QWEN35_FULL_ATTENTION_KV_HEADS_V1 as u32;
    raw.head_dim = AUGUR_CUDA_QWEN35_FULL_ATTENTION_HEAD_DIM_V1 as u32;
    raw.rotary_dim = AUGUR_CUDA_QWEN35_FULL_ATTENTION_ROTARY_DIM_V1 as u32;
    raw.epsilon_bits = epsilon.to_bits();
    Ok(raw)
}

#[cfg(feature = "qwen35-kv8-full-attention")]
#[derive(Clone, Copy, Debug)]
pub struct CudaQwen35PrefillKv8CacheQuantizeViewsV1 {
    pub key_cache_f32: CudaF32View,
    pub value_cache_f32: CudaF32View,
    pub key_cache_q8_0: CudaBufferView,
    pub value_cache_q8_0: CudaBufferView,
}

#[cfg(feature = "qwen35-kv8-full-attention")]
#[derive(Clone, Copy, Debug)]
pub struct CudaQwen35PrefillKv8CacheQuantizeV1 {
    raw: AugurCudaQwen35PrefillKv8CacheQuantizeArgsV1,
    device_ordinal: i32,
    stream_owner_instance_key: usize,
}

#[cfg(feature = "qwen35-kv8-full-attention")]
impl CudaQwen35PrefillKv8CacheQuantizeV1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        views: CudaQwen35PrefillKv8CacheQuantizeViewsV1,
        capacity: usize,
        position_base: usize,
        tokens: usize,
    ) -> Result<Self, CudaRuntimeError> {
        let raw = bind_prefill_kv8_cache_quantize_raw(
            stream.device_ordinal(),
            views,
            capacity,
            position_base,
            tokens,
        )?;
        // SAFETY: exact source/destination geometry, disjointness, ownership,
        // and bounded causal span were checked above; native preflight also
        // validates all four device ranges on exact SM70.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_kv8_cache_quantize_preflight_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &raw,
            )
        };
        CudaRuntimeError::from_result(OPERATION, result)?;
        Ok(Self {
            raw,
            device_ordinal: stream.device_ordinal(),
            stream_owner_instance_key: stream.owner_instance_key()?,
        })
    }

    pub fn submit(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.device_ordinal {
            return Err(CudaRuntimeError::device_mismatch());
        }
        if stream.owner_instance_key()? != self.stream_owner_instance_key {
            return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        // SAFETY: bind completed native preflight, and all owners remain live
        // until the enclosing prefill schedule obtains terminal completion.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_kv8_cache_quantize_prevalidated_submit_v1(
                self.device_ordinal,
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result(OPERATION, result)
    }

    pub const fn raw(&self) -> &AugurCudaQwen35PrefillKv8CacheQuantizeArgsV1 {
        &self.raw
    }
}

#[cfg(feature = "qwen35-kv8-full-attention")]
fn bind_prefill_kv8_cache_quantize_raw(
    device_ordinal: i32,
    views: CudaQwen35PrefillKv8CacheQuantizeViewsV1,
    capacity: usize,
    position_base: usize,
    tokens: usize,
) -> Result<AugurCudaQwen35PrefillKv8CacheQuantizeArgsV1, CudaRuntimeError> {
    let f32_elements = capacity
        .checked_mul(AUGUR_CUDA_QWEN35_FULL_ATTENTION_KV_ELEMENTS_V1)
        .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))?;
    let q8_bytes = capacity
        .checked_mul(AUGUR_CUDA_QWEN35_KV8_CACHE_ROW_BYTES_V3)
        .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))?;
    let end = position_base
        .checked_add(tokens)
        .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))?;
    if capacity == 0
        || capacity > AUGUR_CUDA_QWEN35_FULL_ATTENTION_MAX_CAPACITY_V1
        || tokens == 0
        || end > capacity
        || views.key_cache_f32.element_count() != f32_elements
        || views.value_cache_f32.element_count() != f32_elements
        || views.key_cache_q8_0.len_bytes() != q8_bytes
        || views.value_cache_q8_0.len_bytes() != q8_bytes
    {
        return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
    }
    let raw_views = [
        views.key_cache_f32.buffer_view(),
        views.value_cache_f32.buffer_view(),
        views.key_cache_q8_0,
        views.value_cache_q8_0,
    ];
    validate_disjoint_views(device_ordinal, &raw_views)?;

    let mut raw = AugurCudaQwen35PrefillKv8CacheQuantizeArgsV1::new();
    bind_raw_view(
        &mut raw.key_cache_f32,
        &mut raw.key_cache_f32_len_bytes,
        views.key_cache_f32.buffer_view(),
    )?;
    bind_raw_view(
        &mut raw.value_cache_f32,
        &mut raw.value_cache_f32_len_bytes,
        views.value_cache_f32.buffer_view(),
    )?;
    bind_raw_view(
        &mut raw.key_cache_q8_0,
        &mut raw.key_cache_q8_0_len_bytes,
        views.key_cache_q8_0,
    )?;
    bind_raw_view(
        &mut raw.value_cache_q8_0,
        &mut raw.value_cache_q8_0_len_bytes,
        views.value_cache_q8_0,
    )?;
    raw.capacity = capacity
        .try_into()
        .map_err(|_| validation(CudaRuntimeErrorKind::InvalidDimensions))?;
    raw.position_base = position_base
        .try_into()
        .map_err(|_| validation(CudaRuntimeErrorKind::InvalidDimensions))?;
    raw.tokens = tokens
        .try_into()
        .map_err(|_| validation(CudaRuntimeErrorKind::InvalidDimensions))?;
    raw.kv_heads = AUGUR_CUDA_QWEN35_FULL_ATTENTION_KV_HEADS_V1 as u32;
    raw.head_dim = AUGUR_CUDA_QWEN35_FULL_ATTENTION_HEAD_DIM_V1 as u32;
    Ok(raw)
}







fn bind_raw_view(
    address: &mut u64,
    length: &mut u64,
    view: CudaBufferView,
) -> Result<(), CudaRuntimeError> {
    *address = view.checked_address(OPERATION)?;
    *length = u64::try_from(view.len_bytes())
        .map_err(|_| validation(CudaRuntimeErrorKind::RangeOverflow))?;
    Ok(())
}

fn validate_disjoint_views(
    device_ordinal: i32,
    views: &[CudaBufferView],
) -> Result<(), CudaRuntimeError> {
    if views
        .iter()
        .any(|view| view.device_ordinal() != device_ordinal)
    {
        return Err(CudaRuntimeError::device_mismatch());
    }
    for (index, left) in views.iter().enumerate() {
        left.checked_range(OPERATION)?;
        for right in views.iter().skip(index + 1) {
            if left.overlaps(*right, OPERATION)? {
                return Err(validation(CudaRuntimeErrorKind::OverlappingDeviceRanges));
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn bind_projection_raw(
    device_ordinal: i32,
    max_tokens: usize,
    decoded_weight_f16: CudaBufferView,
    decoded_input_f16: CudaBufferView,
    weight: CudaQwen35PrefillWeightV1,
    input: CudaQ8_1ActivationViewV1,
    output: CudaF32View,
    tokens: usize,
) -> Result<AugurCudaQwen35PrefillCublasProjectionArgsV1, CudaRuntimeError> {
    bind_projection_view_raw(
        device_ordinal,
        max_tokens,
        decoded_weight_f16,
        decoded_input_f16,
        weight,
        input.buffer_view(),
        input.elements(),
        output,
        tokens,
    )
}



#[allow(clippy::too_many_arguments)]
fn bind_projection_view_raw(
    device_ordinal: i32,
    max_tokens: usize,
    decoded_weight_f16: CudaBufferView,
    decoded_input_f16: CudaBufferView,
    weight: CudaQwen35PrefillWeightV1,
    input_view: CudaBufferView,
    input_element_count: usize,
    output: CudaF32View,
    tokens: usize,
) -> Result<AugurCudaQwen35PrefillCublasProjectionArgsV1, CudaRuntimeError> {
    let rows = weight.rows();
    let cols = weight.cols();
    let input_elements = tokens
        .checked_mul(cols)
        .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))?;
    let output_elements = tokens
        .checked_mul(rows)
        .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))?;
    let decoded_weight_bytes = rows
        .checked_mul(cols)
        .and_then(|elements| elements.checked_mul(F16_BYTES))
        .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))?;
    let decoded_input_bytes = input_elements
        .checked_mul(F16_BYTES)
        .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))?;
    if tokens == 0
        || tokens > max_tokens
        || input_element_count != input_elements
        || output.element_count() != output_elements
        || decoded_weight_f16.len_bytes() < decoded_weight_bytes
        || decoded_input_f16.len_bytes() < decoded_input_bytes
    {
        return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
    }
    let weight_view = weight.view();
    let output_view = output.buffer_view();
    let views = [
        weight_view,
        input_view,
        output_view,
        decoded_weight_f16,
        decoded_input_f16,
    ];
    if views
        .iter()
        .any(|view| view.device_ordinal() != device_ordinal)
    {
        return Err(CudaRuntimeError::device_mismatch());
    }
    for left in 0..views.len() {
        for right in left + 1..views.len() {
            if views[left].overlaps(views[right], OPERATION)? {
                return Err(validation(CudaRuntimeErrorKind::OverlappingDeviceRanges));
            }
        }
    }
    let tensor_scale = weight.tensor_scale();
    if !tensor_scale.is_finite() || tensor_scale <= 0.0 {
        return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
    }

    let mut raw = AugurCudaQwen35PrefillCublasProjectionArgsV1::new();
    raw.weight_format = weight.format();
    raw.tokens = tokens
        .try_into()
        .map_err(|_| validation(CudaRuntimeErrorKind::InvalidDimensions))?;
    raw.rows = rows
        .try_into()
        .map_err(|_| validation(CudaRuntimeErrorKind::InvalidDimensions))?;
    raw.cols = cols
        .try_into()
        .map_err(|_| validation(CudaRuntimeErrorKind::InvalidDimensions))?;
    raw.weight = weight_view.checked_address(OPERATION)?;
    raw.weight_len_bytes = weight_view
        .len_bytes()
        .try_into()
        .map_err(|_| validation(CudaRuntimeErrorKind::RangeOverflow))?;
    raw.input_q8_1 = input_view.checked_address(OPERATION)?;
    raw.input_len_bytes = input_view
        .len_bytes()
        .try_into()
        .map_err(|_| validation(CudaRuntimeErrorKind::RangeOverflow))?;
    raw.output_f32 = output_view.checked_address(OPERATION)?;
    raw.output_len_bytes = output_view
        .len_bytes()
        .try_into()
        .map_err(|_| validation(CudaRuntimeErrorKind::RangeOverflow))?;
    raw.tensor_scale = tensor_scale;
    Ok(raw)
}

fn original_input_raw(
    projection: &AugurCudaQwen35PrefillCublasProjectionArgsV1,
    input: CudaF32View,
) -> Result<AugurCudaQwen35PrefillF32ProjectionArgsV1, CudaRuntimeError> {
    let count = (projection.tokens as usize).checked_mul(projection.cols as usize)
        .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))?;
    if count == 0 || input.element_count() != count {
        return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
    }
    let source = input.buffer_view();
    Ok(AugurCudaQwen35PrefillF32ProjectionArgsV1 {
        abi_version: 1,
        struct_size: core::mem::size_of::<AugurCudaQwen35PrefillF32ProjectionArgsV1>() as u32,
        weight_format: projection.weight_format, tokens: projection.tokens,
        rows: projection.rows, cols: projection.cols,
        weight: projection.weight, weight_len_bytes: projection.weight_len_bytes,
        input_f32: source.checked_address(OPERATION)?, input_len_bytes: source.len_bytes() as u64,
        output_f32: projection.output_f32, output_len_bytes: projection.output_len_bytes,
        tensor_scale: projection.tensor_scale, reserved: 0,
    })
}

fn llama_full_original_raw(
    projection: &AugurCudaQwen35PrefillCublasProjectionArgsV1,
    input: CudaF32View,
    output: Option<CudaF32View>,
) -> Result<AugurCudaQwen35PrefillF32ProjectionArgsV1, CudaRuntimeError> {
    llama_original_rows_raw(projection, input, output, false)
}

fn llama_original_rows_raw(
    projection: &AugurCudaQwen35PrefillCublasProjectionArgsV1,
    input: CudaF32View,
    output: Option<CudaF32View>,
    exact_input_rows: bool,
) -> Result<AugurCudaQwen35PrefillF32ProjectionArgsV1, CudaRuntimeError> {
    let row_input = projection.weight_format==row_input::FORMAT;
    let q8 = projection.weight_format == 1 || row_input;
    let nv4 = projection.weight_format == 2;
    let input_shape = q8 && projection.cols == 5120
        && matches!(projection.rows, 1024 | 6144 | 10240 | 12288);
    let output_shape = !row_input && q8 && projection.cols == 6144 && projection.rows == 5120;
    let ffn_shape = !row_input && (q8 || nv4) && matches!((projection.rows, projection.cols),
        (17408, 5120) | (5120, 17408));
    let gdn_bf16 = projection.weight_format == 3 && projection.rows == 48
        && projection.cols == 5120 && projection.tensor_scale == 1.0;
    let admitted_rows = if exact_input_rows {
        (1..=512).contains(&projection.tokens)
    } else {
        projection.tokens == 512
    };
    if !admitted_rows || !(input_shape || output_shape || ffn_shape || gdn_bf16)
        || projection.weight % 4 != 0 || !projection.tensor_scale.is_finite()
        || projection.tensor_scale <= 0.0 || (q8 && projection.tensor_scale != 1.0)
    {
        return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
    }
    let mut raw = original_input_raw(projection, input)?;
    if let Some(output) = output {
        if output.element_count() != projection.tokens as usize * projection.rows as usize {
            return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        validate_disjoint_views(input.buffer_view().device_ordinal(),
            &[input.buffer_view(), output.buffer_view()])?;
        raw.output_f32 = output.buffer_view().checked_address(OPERATION)?;
        raw.output_len_bytes = output.buffer_view().len_bytes() as u64;
    }
    Ok(raw)
}

// The wide B512 producer uses the original full-prefix binding, whereas tails
// use the exact-row binding. Both reach the identical F16/GEMM branch >=64.
const fn shared_original_input_group_rows(full_original: bool, exact_rows: bool, tokens: u32) -> bool {
    full_original && tokens >= 64 && tokens <= 512 && (exact_rows || tokens == 512)
}

fn validation(kind: CudaRuntimeErrorKind) -> CudaRuntimeError {
    CudaRuntimeError::validation(OPERATION, kind)
}

#[cfg(all(
    feature = "qwen35-prefill-cublas",
    feature = "qwen35-kv8-full-attention"
))]
fn kv8_chunk_validation(kind: CudaRuntimeErrorKind) -> CudaRuntimeError {
    CudaRuntimeError::validation(KV8_CHUNK_OPERATION, kind)
}



#[cfg(test)]
mod tests {
    #[test]
    fn head_prepare_stores_null_native_and_abi_contract() {
        assert_eq!(core::mem::size_of::<augur_cuda_sys::AugurCudaQwen35HeadPrepareB8ArgsV1>(),128);
        for preflight in [true,false] {
            let call = if preflight { augur_cuda_sys::augur_cuda_qwen35_head_prepare_b8_preflight_v1 }
                else { augur_cuda_sys::augur_cuda_qwen35_head_prepare_b8_submit_v1 };
            // SAFETY: null metadata/stream must reject before dereferencing.
            let r = unsafe { call(0,core::ptr::null_mut(),core::ptr::null()) };
            assert_eq!(r.status,augur_cuda_sys::AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
        }
    }

    #[test]
    #[ignore = "exclusive GPU; checks only metadata and backing allocation extents"]
    fn head_prepare_stores_native_owner_geometry_and_extent_rejections() {
        use super::*;
        use crate::OwnedCudaBuffer;
        const H: usize = 5120;
        let before = crate::cuda_device_allocation_snapshot();
        let mut rejections = 0;
        {
            let stream = OwnedCudaStream::create(0).unwrap();
            let foreign = OwnedCudaStream::create(0).unwrap();
            let sizes = [8*H*4,H*4,8*H*4,8*H/32*36,H*4,H*4,H/32*36];
            let buffers: Vec<_> = sizes.iter().map(|n| OwnedCudaBuffer::allocate(0,*n).unwrap()).collect();
            let f = |i:usize| CudaF32View::new(buffers[i].view(0,sizes[i]).unwrap(),
                augur_core::PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,sizes[i]/4).unwrap();
            let q = |i:usize| CudaQ8_1ActivationViewV1::new(buffers[i].view(0,sizes[i]).unwrap(),
                augur_core::PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_1_ACTIVATION_CUDA_V1,sizes[i]/36*32).unwrap();
            let allocated = crate::cuda_device_allocation_snapshot();
            let target = CudaQwen35HeadPrepareB8V1::bind_target(&stream,f(0),f(1),f(2),q(3),f(4),f(5),q(6),1e-6).unwrap();
            let draft = CudaQwen35HeadPrepareB8V1::bind_draft(&stream,f(0),q(3),f(5),q(6)).unwrap();
            for bound in [&target,&draft] {
                bound.require_input(f(0)).unwrap();
                assert!(bound.require_input(f(2)).is_err()); rejections += 1;
                assert!(bound.submit(&foreign).is_err()); rejections += 1;
                bound.submissions.set(u64::MAX); assert!(bound.submit(&stream).is_err()); rejections += 1;
                for i in 0..24 {
                    let mut a = bound.raw;
                    match i {
                        0 => a.abi_version=0, 1 => a.struct_size=0, 2 => a.mode=0,
                        3 => a.epsilon_bits=f32::NAN.to_bits(), 4 => a.epsilon_bits=(-1.0f32).to_bits(),
                        5 => a.input=0, 6 => a.input+=2, 7 => a.input=u64::MAX-3,
                        8 => a.input_len_bytes-=4, 9 => a.quantized_len_bytes+=1,
                        10 => a.last_normalized_len_bytes-=4, 11 => a.last_quantized_len_bytes+=4,
                        12 => a.quantized=a.input, 13 => a.last_normalized=a.input,
                        14 => a.last_quantized=a.quantized, 15 => a.last_quantized=a.last_normalized,
                        16 => a.last_normalized=0, 17 => a.last_quantized+=2,
                        18 => if a.mode==1 {a.weight=a.input} else {a.weight=4},
                        19 => a.weight_len_bytes+=4,
                        20 => if a.mode==1 {a.normalized=a.input} else {a.normalized=4},
                        21 => a.normalized_len_bytes+=4,
                        22 => if a.mode==1 {a.last_hidden=a.input} else {a.last_hidden=4},
                        23 => a.last_hidden_len_bytes+=4, _ => unreachable!(),
                    }
                    for preflight in [true,false] {
                        let call = if preflight { augur_cuda_sys::augur_cuda_qwen35_head_prepare_b8_preflight_v1 }
                            else { augur_cuda_sys::augur_cuda_qwen35_head_prepare_b8_submit_v1 };
                        // SAFETY: each mutation is invalid metadata and must reject before a kernel.
                        let result = unsafe {call(0,stream.raw().unwrap(),&a)};
                        assert_eq!(result.status,augur_cuda_sys::AUGUR_CUDA_STATUS_INVALID_ARGUMENT,"mode {} mutation {i}",bound.raw.mode);
                        rejections += 1;
                    }
                }
                let mut a = bound.raw; a.input += 4;
                let r = unsafe {augur_cuda_sys::augur_cuda_qwen35_head_prepare_b8_preflight_v1(0,stream.raw().unwrap(),&a)};
                assert_eq!(r.status,augur_cuda_sys::AUGUR_CUDA_STATUS_INVALID_ARGUMENT); rejections += 1;
            }
            let after = crate::cuda_device_allocation_snapshot();
            assert_eq!(allocated.successful_allocations,after.successful_allocations);
            assert_eq!(allocated.successful_frees,after.successful_frees);
            assert_eq!(allocated.live_bytes,after.live_bytes);
        }
        let after = crate::cuda_device_allocation_snapshot();
        assert_eq!(after.successful_allocations-before.successful_allocations,7);
        assert_eq!(after.successful_frees-before.successful_frees,7);
        assert_eq!(after.live_allocations,before.live_allocations);
        assert_eq!(after.live_bytes,before.live_bytes);
        println!("head_prepare_stores rejections={rejections} allocations=7 frees=7 live_delta=0");
    }

    #[test]
    fn paired_ffn_switch_is_explicit_and_default_off() {
        use super::*;
        assert!(!paired_ffn_switch(None).unwrap());
        assert!(paired_ffn_switch(Some(std::ffi::OsStr::new("1"))).unwrap());
        for value in ["0", "true", "", "2"] {
            assert!(paired_ffn_switch(Some(std::ffi::OsStr::new(value))).is_err());
        }
    }

    #[test]
    fn paired_ffn_excludes_every_noncanonical_policy_and_shape() {
        use super::*;
        let mut raw = AugurCudaQwen35PrefillCublasProjectionArgsV1::new();
        raw.weight_format = 2; raw.tokens = 8; raw.rows = 17408; raw.cols = 5120;
        raw.input_q8_1 = 4096; raw.input_len_bytes = 8 * 5760;
        let gate = CudaQwen35PrefillCublasProjectionV1 {
            raw, weight_decode_lanes: 1, quantized_small_batch: true,
            exact_projection_tiles: true, projection_row_reuse: true,
            ffn_down_row_reuse: false,
            canonical_tiled_rows: false, canonical_grid_rows: false,
            tensor_ffn_down: false, tensor_ffn_down_compensated: false,
            integer_mma_ffn: false, integer_mma_input: false, llama_f16_input: false,
            original_input: None, llama_full_original: false, llama_exact_input_rows: false,
            device_ordinal: 0, stream_owner_instance_key: 0, plan_instance_key: 0,
        };
        assert!(paired_ffn_eligible(&gate, &gate));
        for mutation in 0..19 {
            let mut bad = gate;
            match mutation {
                0 => bad.raw.weight_format = 1, 1 => bad.raw.tokens = 7,
                2 => bad.raw.rows -= 1, 3 => bad.raw.cols -= 1,
                4 => bad.weight_decode_lanes = 2, 5 => bad.quantized_small_batch = false,
                6 => bad.exact_projection_tiles = false, 7 => bad.projection_row_reuse = false,
                8 => bad.canonical_tiled_rows = true, 9 => bad.canonical_grid_rows = true,
                10 => bad.tensor_ffn_down = true, 11 => bad.tensor_ffn_down_compensated = true,
                12 => bad.integer_mma_ffn = true, 13 => bad.integer_mma_input = true,
                14 => bad.llama_f16_input = true, 15 => bad.llama_full_original = true,
                16 => bad.llama_exact_input_rows = true, 17 => bad.raw.input_q8_1 += 4,
                _ => bad.raw.input_len_bytes -= 4,
            }
            assert!(!paired_ffn_eligible(&gate, &bad), "mutation {mutation}");
            assert!(!paired_ffn_eligible(&bad, &gate), "mutation {mutation}");
        }
        let mut original = gate;
        original.original_input = Some(AugurCudaQwen35PrefillF32ProjectionArgsV1 {
            abi_version: 1, struct_size: core::mem::size_of::<AugurCudaQwen35PrefillF32ProjectionArgsV1>() as u32,
            weight_format: 2, tokens: 8, rows: 17408, cols: 5120,
            weight: 0, weight_len_bytes: 0, input_f32: 0, input_len_bytes: 0,
            output_f32: 0, output_len_bytes: 0, tensor_scale: 1.0, reserved: 0,
        });
        assert!(!paired_ffn_eligible(&gate, &original));
        for tokens in [0, 1, 4, 9, 32, 64, 512] {
            let mut other = gate; other.raw.tokens = tokens;
            assert!(!paired_ffn_eligible(&other, &other));
        }
    }
    #[test]
    fn shared_input_group_admits_real_b512_binding_and_exact_large_tails_only() {
        for full in [false, true] {
            for exact in [false, true] {
                for tokens in [0, 1, 8, 9, 63, 64, 65, 285, 511, 512, 513, u32::MAX] {
                    let expected = full && (64..=512).contains(&tokens) && (exact || tokens == 512);
                    assert_eq!(super::shared_original_input_group_rows(full, exact, tokens), expected);
                }
            }
        }
        assert!(super::shared_original_input_group_rows(true, false, 512));
        assert!(!super::shared_original_input_group_rows(true, false, 285));
    }

    #[test]
    fn llama_input_rows_bind_exact_extents_without_widening_legacy() {
        for tokens in [1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 24, 25, 32, 33,
            40, 41, 48, 49, 63, 64, 65, 127, 128, 129, 285, 511, 512] {
            for (format, rows, cols) in [(1, 1024, 5120), (1, 6144, 5120),
                (1, 10240, 5120), (1, 12288, 5120), (1, 5120, 6144),
                (1, 17408, 5120), (1, 5120, 17408), (2, 17408, 5120),
                (2, 5120, 17408), (3, 48, 5120)] {
                let mut raw = tiled_raw(tokens);
                raw.weight_format = format; raw.rows = rows; raw.cols = cols;
                raw.weight = 0x6000_0000; raw.tensor_scale = 1.0;
                let count = tokens as usize * cols as usize;
                let input = CudaF32View::new(test_buffer_view(0x2000_0000, 0, 0, count * 4),
                    PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1, count).unwrap();
                let out_count = tokens as usize * rows as usize;
                let output = CudaF32View::new(test_buffer_view(0x4000_0000, 0, 0, out_count * 4),
                    PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1, out_count).unwrap();
                let bound = llama_original_rows_raw(&raw, input, Some(output), true).unwrap();
                assert_eq!(bound.tokens, tokens);
                assert_eq!(bound.input_len_bytes, (count * 4) as u64);
                assert_eq!(bound.output_len_bytes, (out_count * 4) as u64);
                assert_eq!(bound.output_f32, 0x4000_0000);
                assert_eq!(llama_full_original_raw(&raw, input, Some(output)).is_ok(), tokens == 512);
            }
        }
    }

    #[test]
    fn llama_input_rows_reject_invalid_rows_and_padded_views() {
        let view = |address, count| CudaF32View::new(
            test_buffer_view(address, 0, 0, count * 4),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1, count).unwrap();
        let mut raw = tiled_raw(285);
        raw.weight_format = 1; raw.rows = 5120; raw.cols = 6144;
        raw.weight = 0x6000_0000; raw.tensor_scale = 1.0;
        for tokens in [0, 513, u32::MAX] {
            let mut bad = raw; bad.tokens = tokens;
            assert!(llama_original_rows_raw(&bad, view(0x2000_0000, 285*6144), None, true).is_err());
        }
        let input = view(0x2000_0000, 285*6144);
        for count in [284*5120, 286*5120, 512*5120] {
            assert!(llama_original_rows_raw(&raw, input, Some(view(0x4000_0000, count)), true).is_err());
        }
        assert!(llama_original_rows_raw(&raw, view(0x2000_0000, 512*6144), None, true).is_err());
        assert!(llama_original_rows_raw(&raw, input, Some(view(0x2000_0000, 285*5120)), true).is_err());
        for i in 0..5 {
            let mut bad = raw;
            match i { 0 => bad.weight_format = 0, 1 => bad.cols -= 1,
                2 => bad.rows -= 1, 3 => bad.tensor_scale = f32::NAN, _ => bad.weight += 1 }
            assert!(llama_original_rows_raw(&bad, input, None, true).is_err());
        }
    }

    #[test]
    fn llama_input_rows_null_owner_rejects_before_cuda() {
        for result in unsafe {[
            augur_cuda_sys::augur_cuda_qwen35_prefill_llama_projection_rows_preflight_v1(
                0, core::ptr::null_mut(), core::ptr::null_mut(), core::ptr::null()),
            augur_cuda_sys::augur_cuda_qwen35_prefill_llama_projection_rows_prevalidated_submit_v1(
                0, core::ptr::null_mut(), core::ptr::null_mut(), core::ptr::null()),
            augur_cuda_sys::augur_cuda_qwen35_prefill_llama_swiglu_rows_preflight_v1(
                0, core::ptr::null_mut(), core::ptr::null()),
            augur_cuda_sys::augur_cuda_qwen35_prefill_llama_swiglu_rows_prevalidated_submit_v1(
                0, core::ptr::null_mut(), core::ptr::null()),
        ]} { assert!(CudaRuntimeError::from_result(OPERATION, result).is_err()); }
    }

    #[test]
    fn llama_gdn_bf16_binding_is_only_measured_prefix_geometry() {
        let mut raw = tiled_raw(512);
        raw.weight_format = 3;
        raw.rows = 48;
        raw.cols = 5120;
        raw.weight = 0x6000_0000;
        raw.tensor_scale = 1.0;
        let input = CudaF32View::new(test_buffer_view(0x2000_0000, 0, 0, 512*5120*4),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1, 512*5120).unwrap();
        assert!(llama_full_original_raw(&raw, input, None).is_ok());
        for index in 0..8 {
            let mut bad = raw;
            match index {
                0 => bad.tokens = 8, 1 => bad.tokens = 511,
                2 => bad.rows = 47, 3 => bad.cols = 6144,
                4 => bad.tensor_scale = 0.5, 5 => bad.weight += 2,
                6 => bad.weight_format = 1, _ => bad.weight_format = 2,
            }
            assert!(llama_full_original_raw(&bad, input, None).is_err(), "case {index}");
        }
    }

    #[test]
    fn llama_full_original_binding_covers_only_measured_geometry() {
        for (format, rows, cols) in [(1,1024,5120),(1,6144,5120),(1,10240,5120),
            (1,12288,5120),(1,5120,6144),(1,17408,5120),(1,5120,17408),
            (2,17408,5120),(2,5120,17408)] {
            let mut raw = tiled_raw(512);
            raw.weight_format = format; raw.rows = rows; raw.cols = cols;
            raw.weight = 0x6000_0000; raw.tensor_scale = if format == 2 {0.001} else {1.0};
            let count = 512 * cols as usize;
            let input = CudaF32View::new(test_buffer_view(0x2000_0000, 0, 0, count*4),
                PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1, count).unwrap();
            let bound = llama_full_original_raw(&raw, input, None).unwrap();
            assert_eq!(bound.input_len_bytes, (count*4) as u64);
            assert_eq!(bound.tensor_scale, raw.tensor_scale);
            for bad_kind in 0..6 {
                let mut bad = raw;
                match bad_kind {0 => bad.tokens=8, 1 => bad.weight_format=3,
                    2 => bad.cols+=1, 3 => bad.weight+=2,
                    4 => bad.tensor_scale=f32::NAN, _ => bad.tensor_scale=0.0}
                assert!(llama_full_original_raw(&bad, input, None).is_err());
            }
        }
    }

    #[test]
    fn llama_full_original_relocated_output_is_disjoint_and_exact() {
        let mut raw = tiled_raw(512);
        raw.weight_format=1; raw.rows=5120; raw.cols=6144;
        raw.tensor_scale=1.0; raw.weight=0x6000_0000;
        let input = CudaF32View::new(test_buffer_view(0x2000_0000,0,0,512*6144*4),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,512*6144).unwrap();
        let output = |address,count| CudaF32View::new(test_buffer_view(address,0,0,count*4),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,count).unwrap();
        let relocated=llama_full_original_raw(&raw,input,Some(output(0x4000_0000,512*5120))).unwrap();
        assert_eq!(relocated.output_f32,0x4000_0000);
        assert_eq!(relocated.output_len_bytes,512*5120*4);
        assert!(llama_full_original_raw(&raw,input,Some(output(0x2000_0000,512*5120))).is_err());
        assert!(llama_full_original_raw(&raw,input,Some(output(0x4000_0000,511*5120))).is_err());
    }

    #[test]
    fn llama_full_original_null_owner_rejects_before_cuda() {
        for result in unsafe {[
            augur_cuda_sys::augur_cuda_qwen35_prefill_llama_projection_preflight_v1(
                0,core::ptr::null_mut(),core::ptr::null_mut(),core::ptr::null()),
            augur_cuda_sys::augur_cuda_qwen35_prefill_llama_projection_prevalidated_submit_v1(
                0,core::ptr::null_mut(),core::ptr::null_mut(),core::ptr::null()),
            augur_cuda_sys::augur_cuda_qwen35_prefill_llama_swiglu_preflight_v1(
                0,core::ptr::null_mut(),core::ptr::null()),
            augur_cuda_sys::augur_cuda_qwen35_prefill_llama_swiglu_prevalidated_submit_v1(
                0,core::ptr::null_mut(),core::ptr::null()),
        ]} { assert!(CudaRuntimeError::from_result(OPERATION,result).is_err()); }
    }

    #[test]
    fn original_input_binding_preserves_f32_extent_and_distinct_abi() {
        assert_eq!(core::mem::size_of::<AugurCudaQwen35PrefillF32ProjectionArgsV1>(), 80);
        let mut raw = tiled_raw(512);
        raw.cols = 5120;
        raw.rows = 10240;
        raw.input_q8_1 = 0x7000_0000;
        raw.input_len_bytes = 512 * 5120 / 32 * 36;
        let original = CudaF32View::new(
            test_buffer_view(0x2000_0000, 0, 0, 512 * 5120 * 4),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1, 512 * 5120,
        ).unwrap();
        let bound = original_input_raw(&raw, original).unwrap();
        assert_eq!(bound.input_f32, 0x2000_0000);
        assert_eq!(bound.input_len_bytes, 10_485_760);
        assert_ne!(bound.input_f32, raw.input_q8_1);
        assert_eq!(bound.output_f32, raw.output_f32);
        assert_eq!(bound.weight, raw.weight);
        assert_eq!(raw.input_len_bytes, 2_949_120);
        for tokens in [0, 8, 511, 513] {
            let mut bad = raw;
            bad.tokens = tokens;
            assert!(original_input_raw(&bad, original).is_err());
        }
    }

    #[test]
    fn original_input_null_owner_rejects_before_cuda() {
        for result in unsafe {[
            augur_cuda_sys::augur_cuda_qwen35_prefill_input_original_preflight_v1(
                0, core::ptr::null_mut(), core::ptr::null_mut(), core::ptr::null()),
            augur_cuda_sys::augur_cuda_qwen35_prefill_input_original_prevalidated_submit_v1(
                0, core::ptr::null_mut(), core::ptr::null_mut(), core::ptr::null()),
        ]} {
            assert!(CudaRuntimeError::from_result(OPERATION, result).is_err());
        }
    }

    #[test]
    fn input_integer_selector_is_b512_q8_only_and_rejects_null_owner() {
        use super::*;
        let select = |f, r, c, n| unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_input_integer_scratch_bytes_v1(f, r, c, n)
        };
        for r in [1024, 6144, 10240, 12288] {
            assert_eq!(select(1, r, 5120, 512), 11_468_800);
            for n in [0, 1, 8, 128, 256, 511, 513, 1024, u32::MAX] {
                assert_eq!(select(1, r, 5120, n), 0);
            }
            for f in [0, 2, 3, u32::MAX] {
                assert_eq!(select(f, r, 5120, 512), 0);
            }
            for c in [0, 1024, 5119, 5121, 17408, u32::MAX] {
                assert_eq!(select(1, r, c, 512), 0);
            }
        }
        for r in [0, 1023, 5120, 17408, u32::MAX] {
            assert_eq!(select(1, r, 5120, 512), 0);
        }
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_input_integer_prevalidated_submit_v1(
                0, core::ptr::null_mut(), core::ptr::null_mut(), core::ptr::null(),
            )
        };
        assert!(CudaRuntimeError::from_result(OPERATION, result).is_err());
    }

    #[test]
    fn input_integer_policy_cannot_leak_into_ffn_or_other_paths() {
        use super::*;
        let mut raw = tiled_raw(512);
        raw.weight_format = AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1;
        raw.rows = 10240;
        raw.cols = 5120;
        raw.tensor_scale = 1.0;
        let p = CudaQwen35PrefillCublasProjectionV1 {
            raw, weight_decode_lanes: 1, quantized_small_batch: false,
            exact_projection_tiles: false, projection_row_reuse: false,
            ffn_down_row_reuse: false,
            canonical_tiled_rows: false, canonical_grid_rows: true,
            tensor_ffn_down: false, tensor_ffn_down_compensated: false,
            integer_mma_ffn: false, integer_mma_input: false,
            llama_f16_input: false,
            original_input: None,
            llama_full_original: false,
            llama_exact_input_rows: false,
            device_ordinal: 0, stream_owner_instance_key: 0, plan_instance_key: 0,
        };
        let mut selected = p;
        selected.enable_integer_mma_input_audit().unwrap();
        assert!(selected.integer_mma_input_enabled());
        assert!(!selected.integer_mma_ffn_enabled());
        assert!(!selected.canonical_grid_rows && !selected.canonical_tiled_rows);
        assert!(selected.enable_integer_mma_input_audit().is_err());
        assert!(selected.enable_integer_mma_ffn_audit().is_err());
        assert!(selected.enable_tensor_ffn_down_audit().is_err());
        assert!(selected.enable_canonical_grid_rows_audit().is_err());
        assert!(selected.enable_canonical_tiled_rows_audit().is_err());
        assert!(selected.enable_quantized_small_batch().is_err());
        assert!(selected.enable_projection_row_reuse().is_err());
        assert!(p.clone().enable_llama_f16_input_audit().is_err());
        let mut unaligned = selected;
        unaligned.raw.weight += 2;
        assert!(unaligned.enable_llama_f16_input_audit().is_err());
        let null_result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_input_llama_prevalidated_submit_v1(
                0, core::ptr::null_mut(), core::ptr::null_mut(), core::ptr::null(),
            )
        };
        assert!(CudaRuntimeError::from_result(OPERATION, null_result).is_err());
        selected.enable_llama_f16_input_audit().unwrap();
        assert!(selected.llama_f16_input_enabled());
        assert!(!selected.integer_mma_input_enabled());
        assert!(selected.enable_llama_f16_input_audit().is_err());
        assert!(selected.enable_integer_mma_input_audit().is_err());
        assert!(selected.enable_integer_mma_ffn_audit().is_err());
        assert!(selected.enable_tensor_ffn_down_audit().is_err());
        assert!(selected.enable_canonical_grid_rows_audit().is_err());
        assert!(selected.enable_canonical_tiled_rows_audit().is_err());
        assert!(selected.enable_quantized_small_batch().is_err());
        assert!(selected.enable_projection_row_reuse().is_err());
        for policy in 0..10 {
            let mut conflict = p;
            match policy {
                0 => conflict.weight_decode_lanes = 2,
                1 => conflict.quantized_small_batch = true,
                2 => conflict.exact_projection_tiles = true,
                3 => conflict.projection_row_reuse = true,
                4 => conflict.tensor_ffn_down = true,
                5 => conflict.tensor_ffn_down_compensated = true,
                6 => conflict.integer_mma_ffn = true,
                7 => conflict.raw.tensor_scale = 0.5,
                8 => conflict.raw.rows = 17408,
                _ => conflict.raw.tokens = 256,
            }
            assert!(conflict.enable_integer_mma_input_audit().is_err());
        }
    }

    #[test]
    fn integer_mma_selector_is_bounded_and_null_owner_fails_before_cuda() {
        use super::*;
        let select = |f, r, c, n| unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_integer_mma_scratch_bytes_v1(f, r, c, n)
        };
        for (r, c) in [(17408, 5120), (5120, 17408)] {
            assert_eq!(select(2, r, c, 128), 20_054_016);
            assert_eq!(select(2, r, c, 256), 25_067_520);
            assert_eq!(select(2, r, c, 512), 35_094_528);
            assert_eq!(select(1, r, c, 256), 27_852_800);
            assert_eq!(select(1, r, c, 512), 38_993_920);
            for n in [0, 1, 8, 64, 127, 129, 255, 513, 1024, u32::MAX] {
                assert_eq!(select(2, r, c, n), 0);
                assert_eq!(select(1, r, c, n), 0);
            }
            for f in [0, 1, 3, u32::MAX] {
                assert_eq!(select(f, r, c, 128), 0);
            }
        }
        for (r, c) in [
            (0, 0),
            (5120, 5120),
            (17408, 17408),
            (17409, 5120),
            (5120, u32::MAX),
        ] {
            assert_eq!(select(2, r, c, 128), 0);
        }
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_integer_mma_prevalidated_submit_v1(
                0,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null(),
            )
        };
        assert!(CudaRuntimeError::from_result(OPERATION, result).is_err());
    }

    #[test]
    fn integer_mma_policy_excludes_conflicting_paths() {
        use super::*;
        let mut raw = tiled_raw(128);
        raw.weight_format = AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_NVFP4_V1;
        let p = CudaQwen35PrefillCublasProjectionV1 {
            raw,
            weight_decode_lanes: 1,
            quantized_small_batch: false,
            exact_projection_tiles: false,
            projection_row_reuse: false,
            ffn_down_row_reuse: false,
            canonical_tiled_rows: false,
            canonical_grid_rows: true,
            tensor_ffn_down: false,
            tensor_ffn_down_compensated: false,
            integer_mma_ffn: false,
            device_ordinal: 0,
            integer_mma_input: false,
            llama_f16_input: false,
            original_input: None,
            llama_full_original: false,
            llama_exact_input_rows: false,
            stream_owner_instance_key: 0,
            plan_instance_key: 0,
        };
        let mut selected = p;
        selected.enable_integer_mma_ffn_audit().unwrap();
        assert!(selected.integer_mma_ffn_enabled());
        assert!(!selected.canonical_grid_rows && !selected.canonical_tiled_rows);
        assert!(selected.enable_integer_mma_ffn_audit().is_err());
        assert!(selected.enable_tensor_ffn_down_audit().is_err());
        assert!(selected.enable_canonical_grid_rows_audit().is_err());
        assert!(selected.enable_canonical_tiled_rows_audit().is_err());
        assert!(selected.enable_quantized_small_batch().is_err());
        assert!(selected.enable_projection_row_reuse().is_err());
        for policy in 0..8 {
            let mut conflict = p;
            match policy {
                0 => conflict.weight_decode_lanes = 2,
                1 => conflict.weight_decode_lanes = 4,
                2 => conflict.quantized_small_batch = true,
                3 => conflict.exact_projection_tiles = true,
                4 => conflict.projection_row_reuse = true,
                5 => conflict.tensor_ffn_down = true,
                6 => conflict.tensor_ffn_down_compensated = true,
                _ => conflict.raw.tokens = 64,
            }
            assert!(conflict.enable_integer_mma_ffn_audit().is_err());
        }
    }

    #[test]
    fn compensated_ffn_down_is_only_nvfp4_b128_and_rejects_null_owner() {
        use super::*;
        let select = |format, rows, cols, tokens| unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_down_compensated_tile_rows_v1(
                format, rows, cols, tokens,
            )
        };
        assert_eq!(
            select(
                AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_NVFP4_V1,
                5120,
                17408,
                128
            ),
            896
        );
        for f in [
            0,
            AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1,
            AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_BF16_V1,
            u32::MAX,
        ] {
            assert_eq!(select(f, 5120, 17408, 128), 0);
        }
        for n in [0, 1, 8, 32, 64, 127, 129, 256, 512, u32::MAX] {
            assert_eq!(
                select(
                    AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_NVFP4_V1,
                    5120,
                    17408,
                    n
                ),
                0
            );
        }
        let rejected = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_down_compensated_prevalidated_submit_v1(
                0,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null(),
            )
        };
        assert!(CudaRuntimeError::from_result(OPERATION, rejected).is_err());
    }

    #[test]
    fn tensor_ffn_down_selector_rejects_unmeasured_shapes_before_cuda() {
        use super::*;
        let select = |format, rows, cols, tokens| unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_down_tensor_tile_rows_v1(format, rows, cols, tokens)
        };
        for format in [
            AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1,
            AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_NVFP4_V1,
        ] {
            for n in [64, 128] {
                assert_eq!(select(format, 5120, 17408, n), 896);
            }
            for n in [0, 1, 8, 16, 32, 63, 65, 127, 129, 256, 512, u32::MAX] {
                assert_eq!(select(format, 5120, 17408, n), 0);
            }
            for (rows, cols) in [(17408, 5120), (5121, 17408), (5120, 17407), (0, 17408)] {
                assert_eq!(select(format, rows, cols, 128), 0);
            }
        }
        for format in [0, AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_BF16_V1, 4, u32::MAX] {
            assert_eq!(select(format, 5120, 17408, 128), 0);
        }
    }

    #[test]
    fn tensor_ffn_down_rejects_missing_owner_before_cuda() {
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_down_tensor_prevalidated_submit_v1(
                0,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null(),
            )
        };
        assert!(super::CudaRuntimeError::from_result(super::OPERATION, result).is_err());
    }

    #[test]
    fn canonical_prefill_grid_rejects_null_before_cuda() {
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_canonical_prefill_grid_prevalidated_submit_v1(
                0,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null(),
            )
        };
        assert!(super::CudaRuntimeError::from_result(super::OPERATION, result).is_err());
    }
    fn tiled_raw(tokens: u32) -> super::AugurCudaQwen35PrefillCublasProjectionArgsV1 {
        use super::*;
        let mut raw = AugurCudaQwen35PrefillCublasProjectionArgsV1::new();
        raw.weight_format = AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1;
        raw.tokens = tokens;
        raw.rows = 17408;
        raw.cols = 5120;
        raw.input_q8_1 = 0x100000;
        raw.output_f32 = 0x10000000;
        raw.input_len_bytes = u64::from(tokens) * 5760;
        raw.output_len_bytes = u64::from(tokens) * 17408 * 4;
        raw
    }
    #[test]
    fn canonical_projection_tiles_partition_real_rows_without_padding() {
        use super::*;
        for rows in 1..=512 {
            let raw = tiled_raw(rows);
            let mut consumed = 0;
            for start in (0..rows).step_by(8) {
                let tile = canonical_projection_tile(&raw, start).unwrap();
                assert_eq!(tile.input_q8_1, raw.input_q8_1 + u64::from(start) * 5760);
                assert_eq!(
                    tile.output_f32,
                    raw.output_f32 + u64::from(start) * 17408 * 4
                );
                assert!((1..=8).contains(&tile.tokens));
                assert!(
                    tile.input_q8_1 + tile.input_len_bytes <= raw.input_q8_1 + raw.input_len_bytes
                );
                assert!(
                    tile.output_f32 + tile.output_len_bytes
                        <= raw.output_f32 + raw.output_len_bytes
                );
                consumed += tile.tokens;
            }
            assert_eq!(consumed, rows);
        }
    }
    #[test]
    fn canonical_projection_tiles_reject_extent_format_and_address_drift() {
        use super::*;
        for rows in [0, 513, u32::MAX] {
            assert!(canonical_projection_tile(&tiled_raw(rows), 0).is_err());
        }
        for start in [1, 7, 512, u32::MAX] {
            assert!(canonical_projection_tile(&tiled_raw(512), start).is_err());
        }
        for case in 0..6 {
            let mut r = tiled_raw(512);
            match case {
                0 => r.weight_format = AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_BF16_V1,
                1 => r.input_len_bytes -= 1,
                2 => r.output_len_bytes -= 1,
                3 => r.input_q8_1 = u64::MAX - 1,
                4 => r.output_f32 = u64::MAX - 1,
                _ => r.cols = 5119,
            }
            assert!(canonical_projection_tile(&r, 0).is_err());
        }
    }
    #[test]
    fn canonical_prefill_row_norm_keeps_legacy_eight_row_gate_closed() {
        use super::*;
        for tokens in [0, 1, 8, 9, 511, 512, 513, u32::MAX] {
            let mut raw = AugurCudaQwen35PrefillRmsnormArgsV1::new();
            raw.tokens = tokens;
            let mut op = CudaQwen35PrefillRmsnormV1 {
                raw,
                device_ordinal: 0,
                stream_owner_instance_key: 1,
                canonical_rows: false,
                llama_input_rows: false,
            };
            assert_eq!(
                op.enable_canonical_prefill_rows_audit().is_ok(),
                (1..=512).contains(&tokens)
            );
            op.canonical_rows = false;
            assert_eq!(
                op.enable_canonical_small_batch().is_ok(),
                (1..=8).contains(&tokens)
            );
        }
    }
    #[test]
    fn canonical_row_norm_selection_is_explicit_and_bounded() {
        use super::*;
        for tokens in [0, 1, 2, 8, 9, u32::MAX] {
            let mut raw = AugurCudaQwen35PrefillRmsnormArgsV1::new();
            raw.tokens = tokens;
            let mut op = CudaQwen35PrefillRmsnormV1 {
                raw,
                device_ordinal: 0,
                stream_owner_instance_key: 1,
                canonical_rows: false,
                llama_input_rows: false,
            };
            let expected = (1..=8).contains(&tokens);
            assert_eq!(op.enable_canonical_small_batch().is_ok(), expected);
            assert_eq!(op.canonical_rows, expected);
        }
    }
    #[test]
    fn llama_input_norm_selection_is_explicit_bounded_and_cannot_silently_revert() {
        use super::*;
        for tokens in [0, 1, 8, 9, 16, 63, 64, 285, 511, 512, 513, u32::MAX] {
            let mut raw = AugurCudaQwen35PrefillRmsnormArgsV1::new();
            raw.tokens = tokens; raw.cols = 5120; raw.epsilon_bits = 1e-6_f32.to_bits();
            let mut op = CudaQwen35PrefillRmsnormV1 {raw, device_ordinal: 0,
                stream_owner_instance_key: 1, canonical_rows: false, llama_input_rows: false};
            assert!(!op.llama_exact_input_rows_enabled());
            let admitted = (1..=512).contains(&tokens);
            assert_eq!(op.enable_llama_exact_input_rows_audit().is_ok(), admitted);
            assert_eq!(op.llama_exact_input_rows_enabled(), admitted);
            if admitted {
                assert!(!op.canonical_rows);
                assert!(op.enable_canonical_small_batch().is_err());
                assert!(op.enable_canonical_prefill_rows_audit().is_err());
            }
        }
    }
    #[test]
    fn llama_input_norm_rejects_other_width_or_epsilon() {
        use super::*;
        for (cols, epsilon) in [(0, 1e-6_f32), (128, 1e-6), (5119, 1e-6),
            (5121, 1e-6), (5120, 0.0), (5120, 1e-5), (5120, f32::NAN)] {
            let mut raw = AugurCudaQwen35PrefillRmsnormArgsV1::new();
            raw.tokens = 3; raw.cols = cols; raw.epsilon_bits = epsilon.to_bits();
            let mut op = CudaQwen35PrefillRmsnormV1 {raw, device_ordinal: 0,
                stream_owner_instance_key: 1, canonical_rows: true, llama_input_rows: false};
            assert!(op.enable_llama_exact_input_rows_audit().is_err());
            assert!(op.canonical_rows);
            assert!(!op.llama_exact_input_rows_enabled());
        }
    }
    fn compare_canonical_norm_rows(input: &[u8], weight: &[u8], rows: usize, cols: usize) {
        use super::*;
        use augur_core::PhysicalLayoutId;
        assert_eq!(input.len(), rows * cols * 4);
        assert_eq!(weight.len(), cols * 4);
        let stream = OwnedCudaStream::create(0).unwrap();
        let other = OwnedCudaStream::create(0).unwrap();
        let x = OwnedCudaBuffer::allocate(0, input.len()).unwrap();
        let w = OwnedCudaBuffer::allocate(0, weight.len()).unwrap();
        let control = OwnedCudaBuffer::allocate(0, input.len()).unwrap();
        let candidate = OwnedCudaBuffer::allocate(0, input.len()).unwrap();
        x.copy_from_host_sync(&stream, input).unwrap();
        w.copy_from_host_sync(&stream, weight).unwrap();
        let view = |buffer: &OwnedCudaBuffer, start: usize, count: usize| {
            CudaF32View::new(buffer.view(start * 4, count * 4).unwrap(),
                PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1, count).unwrap()
        };
        let mut batch = CudaQwen35PrefillRmsnormV1::bind(&stream,
            view(&x, 0, rows * cols), view(&w, 0, cols),
            view(&candidate, 0, rows * cols), rows, cols, 1e-6).unwrap();
        batch.enable_canonical_prefill_rows_audit().unwrap();
        assert!(batch.submit(&other).is_err(), "foreign stream owner accepted");
        let scalar: Vec<_> = (0..rows).map(|row| {
            crate::CudaRmsnormF32ArgsV1::bind(&stream, view(&x, row * cols, cols),
                view(&w, 0, cols), view(&control, row * cols, cols), 1e-6).unwrap()
        }).collect();
        for repetition in 0..3 {
            // Alternate enqueue order; every byte must be independently written.
            let poison = vec![0xff; input.len()];
            candidate.copy_from_host_sync(&stream, &poison).unwrap();
            if repetition % 2 == 0 { batch.submit(&stream).unwrap(); }
            for op in &scalar { op.submit_for_plan(&stream).unwrap(); }
            if repetition % 2 == 1 { batch.submit(&stream).unwrap(); }
            let mut expected = vec![0; input.len()];
            let mut actual = vec![0; input.len()];
            control.copy_to_host_sync(&stream, &mut expected).unwrap();
            candidate.copy_to_host_sync(&stream, &mut actual).unwrap();
            assert!(expected.chunks_exact(4).all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()));
            assert_eq!(actual, expected, "row batch changed output bits: {rows}x{cols}");
        }
        let mut unchanged = vec![0; input.len()];
        x.copy_to_host_sync(&stream, &mut unchanged).unwrap();
        assert_eq!(unchanged, input);
        let mut unchanged = vec![0; weight.len()];
        w.copy_to_host_sync(&stream, &mut unchanged).unwrap();
        assert_eq!(unchanged, weight);
        println!("canonical_row_norm_exact rows={rows} cols={cols} compared_bytes={}", input.len());
    }

    #[test]
    #[ignore = "requires exclusive SM70 GPU"]
    fn canonical_row_norm_gpu_stress_is_bitwise_scalar() {
        for rows in [1, 7, 8, 9, 31, 512] {
            for cols in [1, 31, 256, 5120] {
                let input: Vec<u8> = (0..rows * cols).flat_map(|i| {
                    let bits = (i as u32).wrapping_mul(1664525).wrapping_add(1013904223);
                    let value = match i % 29 {
                        0 => -0.0, 1 => f32::from_bits(1), 2 => -1e-38,
                        _ => f32::from_bits((bits & 0x807fffff) | ((112 + i % 24) as u32) << 23),
                    };
                    value.to_le_bytes()
                }).collect();
                let weight: Vec<u8> = (0..cols).flat_map(|i| {
                    (((i * 17 % 127) as f32 - 63.0) / 61.0).to_le_bytes()
                }).collect();
                compare_canonical_norm_rows(&input, &weight, rows, cols);
            }
        }
    }

    #[test]
    #[ignore = "requires exclusive SM70 GPU and bound real layer0 fixture"]
    fn canonical_row_norm_gpu_real_fixture_is_bitwise_scalar() {
        let root = std::path::PathBuf::from(std::env::var_os("AUGUR_QWEN35_NORM_FIXTURE_DIR")
            .expect("explicit real-weight/activation fixture required"));
        let input = std::fs::read(root.join("input.f32")).unwrap();
        let weight = std::fs::read(root.join("norm.f32")).unwrap();
        assert_eq!(input.len(), 512 * 5120 * 4);
        assert_eq!(weight.len(), 5120 * 4);
        for rows in [1, 7, 8, 9, 31, 512] {
            compare_canonical_norm_rows(&input[..rows * 5120 * 4], &weight, rows, 5120);
        }
    }

    #[test]
    #[ignore = "requires exclusive SM70 GPU"]
    fn canonical_row_norm_gpu_native_rejects_bad_extents() {
        use super::*;
        let stream = OwnedCudaStream::create(0).unwrap();
        let x = OwnedCudaBuffer::allocate(0, 8 * 5120 * 4).unwrap();
        let w = OwnedCudaBuffer::allocate(0, 5120 * 4).unwrap();
        let y = OwnedCudaBuffer::allocate(0, 8 * 5120 * 4).unwrap();
        let mut base = augur_cuda_sys::AugurCudaRmsnormF32ArgsV1::new();
        base.input = x.device_ptr().unwrap().raw_value();
        base.input_len_bytes = 8 * 5120 * 4;
        base.weight = w.device_ptr().unwrap().raw_value();
        base.weight_len_bytes = 5120 * 4;
        base.output = y.device_ptr().unwrap().raw_value();
        base.output_len_bytes = 8 * 5120 * 4;
        base.elements = 5120;
        base.epsilon_bits = 1e-6f32.to_bits();
        for case in 0..16 {
            let mut raw = base;
            let mut rows = 8;
            match case {
                0 => rows = 0, 1 => rows = 513, 2 => rows = u32::MAX,
                3 => raw.input_len_bytes -= 4, 4 => raw.output_len_bytes += 4,
                5 => raw.weight_len_bytes += 4, 6 => raw.input = u64::MAX - 3,
                7 => raw.output = raw.input, 8 => raw.output = raw.weight,
                9 => raw.input += 1, 10 => raw.weight = 0,
                11 => raw.epsilon_bits = f32::NAN.to_bits(),
                12 => raw.epsilon_bits = 0, 13 => raw.elements = 0,
                14 => raw.struct_size = 0, _ => raw.abi_version = 0,
            }
            // SAFETY: mutations must reject before any launch or dereference;
            // valid base allocations remain live for the entire test.
            let result = unsafe { augur_cuda_sys::augur_cuda_rmsnorm_f32_rows_submit_v1(
                0, stream.raw().unwrap(), &raw, rows) };
            assert_eq!(result.status, augur_cuda_sys::AUGUR_CUDA_STATUS_INVALID_ARGUMENT,
                "invalid case {case} admitted");
        }
        stream.synchronize().unwrap();
    }

    #[test]
    #[ignore = "requires exclusive SM70 GPU"]
    fn compare_summary_gpu_matches_known_f32_and_q8_values() {
        use super::*;
        let stream = OwnedCudaStream::create(0).unwrap();
        for format in [0, 1] {
            let (a, b) = if format == 0 {
                (
                    [1.0f32, 3.0, -2.0, f32::NAN]
                        .into_iter()
                        .flat_map(|x| x.to_le_bytes())
                        .collect::<Vec<_>>(),
                    [1.0f32, 1.0, -1.0, 4.0]
                        .into_iter()
                        .flat_map(|x| x.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
            } else {
                let mut a = vec![2u8; 34];
                let mut b = vec![1u8; 34];
                a[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
                b[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
                (a, b)
            };
            let left = OwnedCudaBuffer::allocate(0, a.len()).unwrap();
            let right = OwnedCudaBuffer::allocate(0, b.len()).unwrap();
            let output = OwnedCudaBuffer::allocate(0, 64).unwrap();
            left.copy_from_host_sync(&stream, &a).unwrap();
            right.copy_from_host_sync(&stream, &b).unwrap();
            submit_qwen35_compare_summary(
                &stream,
                left.view(0, a.len()).unwrap(),
                right.view(0, b.len()).unwrap(),
                output.view(0, 64).unwrap(),
                format,
            )
            .unwrap();
            let mut bytes = [0u8; 64];
            output.copy_to_host_sync(&stream, &mut bytes).unwrap();
            let u = |i: usize| u64::from_le_bytes(bytes[i * 8..(i + 1) * 8].try_into().unwrap());
            let expected = if format == 0 {
                [
                    4u64,
                    3,
                    1,
                    2.0f64.to_bits(),
                    5.0f64.to_bits(),
                    3.0f64.to_bits(),
                    14.0f64.to_bits(),
                    6.0f64.to_bits(),
                ]
            } else {
                [
                    32u64,
                    32,
                    0,
                    1.0f64.to_bits(),
                    32.0f64.to_bits(),
                    32.0f64.to_bits(),
                    128.0f64.to_bits(),
                    64.0f64.to_bits(),
                ]
            };
            assert_eq!((0..8).map(u).collect::<Vec<_>>(), expected);
        }
    }
    #[test]
    fn compare_summary_null_metadata_rejects_without_cuda() {
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_compare_summary_v1(
                0,
                core::ptr::null_mut(),
                0,
                0,
                0,
                0,
                0,
                0,
            )
        };
        assert!(super::CudaRuntimeError::from_result(super::OPERATION, result).is_err());
    }
    #[test]
    fn quantized_small_batch_native_entry_rejects_null_before_cuda() {
        // SAFETY: null metadata is rejected before device selection or launch.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_quantized_small_batch_prevalidated_submit_v1(
                0,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null(),
            )
        };
        assert!(super::CudaRuntimeError::from_result(super::OPERATION, result).is_err());
    }
    #[test]
    fn quantized_small_batch_policy_is_bounded_and_rejects_bf16() {
        use super::*;
        for format in [
            AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1,
            AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_NVFP4_V1,
        ] {
            for tokens in 1..=8 {
                assert!(quantized_small_batch_supported(format, tokens));
            }
            for tokens in [0, 9, u32::MAX] {
                assert!(!quantized_small_batch_supported(format, tokens));
            }
        }
        assert!(!quantized_small_batch_supported(
            AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_BF16_V1,
            8
        ));
        assert!(!quantized_small_batch_supported(u32::MAX, 8));
    }
    #[test]
    fn exact_projection_tiles_native_shape_set_and_tail_fallback() {
        let width = |format, rows, cols, tokens| unsafe {
            augur_cuda_sys::augur_cuda_qwen35_exact_projection_tile_width_v1(format, rows, cols, tokens)
        };
        for format in [1, 2] {
            assert_eq!(width(format, 17408, 5120, 8), 8);
            assert_eq!(width(format, 5120, 17408, 8), 4);
            for tokens in [0, 1, 2, 3, 4, 5, 6, 7, 9, u32::MAX] {
                assert_eq!(width(format, 17408, 5120, tokens), 0);
                assert_eq!(width(format, 5120, 17408, tokens), 0);
            }
            assert_eq!(width(format, 131, 1088, 8), 0);
            assert_eq!(width(format, 0, 5120, 8), 0);
        }
        for (rows, cols) in [
            (10240, 5120),
            (6144, 5120),
            (12288, 5120),
            (1024, 5120),
            (248320, 5120),
            (5120, 6144),
        ] {
            assert_eq!(width(1, rows, cols, 8), 8);
            assert_eq!(width(2, rows, cols, 8), 0);
        }
        assert_eq!(width(3, 17408, 5120, 8), 0);
        assert_eq!(width(u32::MAX, 17408, 5120, 8), 0);
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_exact_projection_tiles_prevalidated_submit_v1(
                0,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null(),
            )
        };
        assert!(super::CudaRuntimeError::from_result(super::OPERATION, result).is_err());
    }
    #[test]
    fn projection_row_reuse_native_shape_set_and_canonical_fallback() {
        let width = |format, rows, cols, tokens| unsafe {
            augur_cuda_sys::augur_cuda_qwen35_projection_row_reuse_width_v1(format, rows, cols, tokens)
        };
        for format in [1, 2] {
            assert_eq!(width(format, 17408, 5120, 8), 2);
            for tokens in [0, 1, 2, 3, 4, 5, 6, 7, 9, 512, u32::MAX] {
                assert_eq!(width(format, 17408, 5120, tokens), 0);
            }
            for (rows, cols) in [
                (5120, 17408),
                (5120, 6144),
                (1024, 5120),
                (248320, 5120),
                (131, 1088),
                (0, 5120),
            ] {
                assert_eq!(width(format, rows, cols, 8), 0);
            }
        }
        for rows in [10240, 6144, 12288] {
            assert_eq!(width(1, rows, 5120, 8), 2);
            assert_eq!(width(2, rows, 5120, 8), 0);
        }
        for format in [0, 3, u32::MAX] {
            assert_eq!(width(format, 17408, 5120, 8), 0);
        }
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_projection_row_reuse_prevalidated_submit_v1(
                0,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null(),
            )
        };
        assert!(super::CudaRuntimeError::from_result(super::OPERATION, result).is_err());
    }
    #[test]
    fn ffn_down_row_reuse_exact_shape_policy_and_null_native_rejection() {
        use super::*;
        let mut raw = AugurCudaQwen35PrefillCublasProjectionArgsV1::new();
        raw.weight_format = 2; raw.tokens = 8; raw.rows = 5120; raw.cols = 17408;
        let p = CudaQwen35PrefillCublasProjectionV1 {
            raw, weight_decode_lanes: 1, quantized_small_batch: true,
            exact_projection_tiles: true, projection_row_reuse: true,
            ffn_down_row_reuse: false,
            canonical_tiled_rows: false, canonical_grid_rows: false,
            tensor_ffn_down: false, tensor_ffn_down_compensated: false,
            integer_mma_ffn: false, integer_mma_input: false, llama_f16_input: false,
            original_input: None, llama_full_original: false, llama_exact_input_rows: false,
            device_ordinal: 0, stream_owner_instance_key: 0, plan_instance_key: 0,
        };
        assert!(!p.ffn_down_row_reuse_enabled());
        for format in [1, 2] {
            let mut selected = p;
            selected.raw.weight_format = format;
            selected.enable_ffn_down_row_reuse().unwrap();
            assert!(selected.ffn_down_row_reuse_enabled());
            assert_eq!(selected.projection_row_reuse_width(), 0);
        }
        for mutation in 0..16 {
            let mut bad = p;
            match mutation {
                0 => bad.raw.rows = 5121, 1 => bad.raw.cols = 5120,
                2 => bad.raw.tokens = 7, 3 => bad.raw.tokens = 9,
                4 => bad.raw.weight_format = 0, 5 => bad.raw.weight_format = 3,
                6 => bad.projection_row_reuse = false, 7 => bad.exact_projection_tiles = false,
                8 => bad.canonical_tiled_rows = true, 9 => bad.canonical_grid_rows = true,
                10 => bad.tensor_ffn_down = true, 11 => bad.integer_mma_ffn = true,
                12 => bad.integer_mma_input = true, 13 => bad.llama_f16_input = true,
                14 => bad.weight_decode_lanes = 2, _ => bad.raw.tokens = 0,
            }
            assert!(bad.enable_ffn_down_row_reuse().is_err(), "mutation {mutation}");
            assert!(!bad.ffn_down_row_reuse_enabled());
        }
        // SAFETY: null/invalid metadata must reject without any CUDA operation.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_ffn_down_row_reuse_prevalidated_submit_v1(
                0, core::ptr::null_mut(), core::ptr::null_mut(), core::ptr::null(),
            )
        };
        assert!(CudaRuntimeError::from_result(OPERATION, result).is_err());
    }
    #[test]
    fn compiled_microbatch_capability_matches_native_without_cuda() {
        // SAFETY: pure constant query; no pointer arguments or CUDA calls.
        let actual = unsafe { augur_cuda_sys::augur_cuda_qwen35_prefill_cublas_compiled_max_tokens_v1() };
        assert_eq!(
            actual,
            super::AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_COMPILED_MAX_TOKENS_V1
        );
        assert_eq!(super::AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_MAX_TOKENS_V1, 1024);
        assert_eq!(
            actual,
            if cfg!(feature = "qwen35-augur-large-prefill") {
                6400
            } else {
                1024
            }
        );
    }



    #[test]
    fn quad_projection_entries_reject_invalid_metadata_without_cuda() {
        // SAFETY: null metadata rejects before CUDA.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_cublas_projection_quad_prevalidated_submit_v1(
                0,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null(),
            )
        };
        assert!(super::CudaRuntimeError::from_result(super::OPERATION, result).is_err());
        for stage in [0, 1, 2, 3, u32::MAX] {
            // SAFETY: invalid metadata/stage admission has no CUDA side effect.
            let result = unsafe {
                augur_cuda_sys::augur_cuda_qwen35_prefill_cublas_projection_quad_profile_stage_prevalidated_submit_v1(
                    0, core::ptr::null_mut(), core::ptr::null_mut(), core::ptr::null(), stage,
                )
            };
            assert!(super::CudaRuntimeError::from_result(super::OPERATION, result).is_err());
        }
    }

    #[test]
    fn quad_nvfp4_word_load_is_aligned_and_preserves_all_scalar_nibbles() {
        for index in (0_usize..64 * 3).step_by(4) {
            let lane = index % 64;
            let base = index / 64 * 36;
            let offset = base + 4 + lane / 16 * 8 + (lane % 16 & 7);
            let shift = if lane % 16 < 8 { 0 } else { 4 };
            assert_eq!(offset % 4, 0);
            for packed in [0_u32, u32::MAX, 0x0123_4567, 0x89ab_cdef, 0x80ff_7f00] {
                let bytes = packed.to_le_bytes();
                for item in 0..4 {
                    let scalar_lane = lane + item;
                    assert_eq!(
                        offset + item,
                        base + 4 + scalar_lane / 16 * 8 + (scalar_lane % 16 & 7)
                    );
                    assert_eq!(lane / 16, scalar_lane / 16);
                    let scalar = if scalar_lane % 16 < 8 {
                        bytes[item] & 15
                    } else {
                        bytes[item] >> 4
                    };
                    assert_eq!(((packed >> (8 * item + shift)) & 15) as u8, scalar);
                }
            }
        }
    }

    #[cfg(feature = "qwen35-kv8-full-attention")]
    #[test]
    fn unit_scale_entry_rejects_null_without_cuda() {
        // SAFETY: null stream/args rejected before CUDA.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_kv8_chunk_shared_softmax_unit_scale_prevalidated_submit_v1(
                0, core::ptr::null_mut(), core::ptr::null(),
            )
        };
        assert!(super::CudaRuntimeError::from_result(super::OPERATION, result).is_err());
    }

    #[cfg(feature = "qwen35-kv8-full-attention")]
    #[test]
    fn dgemm_scores_workspace_bound_and_null_entries() {
        use super::CudaQwen35PrefillKv8ChunkAttentionV1 as Op;
        assert_eq!(
            Op::dgemm_score_elements(512, 196608).unwrap() * 4,
            169_369_600
        );
        assert!(Op::dgemm_score_elements(0, 1).is_err());
        assert!(Op::dgemm_score_elements(1, usize::MAX).is_err());
        // SAFETY: null metadata/plan must reject before CUDA access.
        unsafe {
            let a = augur_cuda_sys::augur_cuda_qwen35_prefill_kv8_tiled_dgemm_preflight_v1(
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null(),
                0,
                0,
            );
            let b = augur_cuda_sys::augur_cuda_qwen35_prefill_kv8_tiled_dgemm_prevalidated_submit_v1(
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null(),
                0,
                0,
            );
            assert_eq!(a.status, augur_cuda_sys::AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
            assert_eq!(b.status, augur_cuda_sys::AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
        }
    }

    #[cfg(feature = "qwen35-kv8-full-attention")]
    #[test]
    fn tiled_cached_query_entry_rejects_null_without_cuda() {
        // SAFETY: invalid null metadata must reject before CUDA access.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_kv8_tiled_cached_query_prevalidated_submit_v1(
                0,
                std::ptr::null_mut(),
                std::ptr::null(),
                0,
                0,
            )
        };
        assert_eq!(result.status, augur_cuda_sys::AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
    }

    #[cfg(feature = "qwen35-kv8-full-attention")]
    #[test]
    fn tiled_scores_workspace_bound_and_null_entries() {
        use super::CudaQwen35PrefillKv8ChunkAttentionV1 as Op;
        assert_eq!(
            Op::tiled_score_elements(512, 196608).unwrap() * 4,
            100_687_872
        );
        assert_eq!(
            Op::tiled_score_elements(129, 8193).unwrap(),
            128 * 24 * 8194
        );
        assert_eq!(Op::tiled_score_elements(7, 100).unwrap(), 7 * 24 * 102);
        for (tokens, capacity) in [(0, 1), (1, 0), (usize::MAX, 1), (1, usize::MAX)] {
            assert!(Op::tiled_score_elements(tokens, capacity).is_err());
        }
        // SAFETY: null metadata must be rejected before CUDA access.
        unsafe {
            let a = augur_cuda_sys::augur_cuda_qwen35_prefill_kv8_tiled_scores_preflight_v1(
                0,
                std::ptr::null_mut(),
                std::ptr::null(),
                0,
                0,
            );
            let b = augur_cuda_sys::augur_cuda_qwen35_prefill_kv8_tiled_scores_prevalidated_submit_v1(
                0,
                std::ptr::null_mut(),
                std::ptr::null(),
                0,
                0,
            );
            assert_eq!(a.status, augur_cuda_sys::AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
            assert_eq!(b.status, augur_cuda_sys::AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
        }
    }

    #[cfg(feature = "qwen35-kv8-full-attention")]
    #[test]
    fn staged_scores_workspace_bound_and_null_entries() {
        use super::CudaQwen35PrefillKv8ChunkAttentionV1 as Op;
        assert_eq!(
            Op::staged_score_elements(512, 196608).unwrap() * 4,
            301_989_888
        );
        assert_eq!(Op::staged_score_elements(7, 100).unwrap(), 7 * 24 * 100);
        for (tokens, capacity) in [(0, 1), (1, 0), (usize::MAX, 1), (1, usize::MAX)] {
            assert!(Op::staged_score_elements(tokens, capacity).is_err());
        }
        // SAFETY: invalid pointers are intentionally null and must be rejected
        // before any CUDA access on both preflight and submit entries.
        unsafe {
            let a = augur_cuda_sys::augur_cuda_qwen35_prefill_kv8_staged_scores_preflight_v1(
                0,
                std::ptr::null_mut(),
                std::ptr::null(),
                0,
                0,
            );
            let b = augur_cuda_sys::augur_cuda_qwen35_prefill_kv8_staged_scores_prevalidated_submit_v1(
                0,
                std::ptr::null_mut(),
                std::ptr::null(),
                0,
                0,
            );
            assert_eq!(a.status, augur_cuda_sys::AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
            assert_eq!(b.status, augur_cuda_sys::AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
        }
    }

    #[cfg(feature = "qwen35-kv8-full-attention")]
    #[test]
    fn parallel_scales_entry_rejects_null_without_cuda() {
        // SAFETY: null metadata must reject before CUDA submission.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_kv8_chunk_parallel_scales_prevalidated_submit_v1(
                0,
                std::ptr::null_mut(),
                std::ptr::null(),
            )
        };
        assert!(super::CudaRuntimeError::from_result(super::KV8_CHUNK_OPERATION, result).is_err());
    }

    #[cfg(feature = "qwen35-kv8-full-attention")]
    #[test]
    fn warp_shared_kv_entry_rejects_null_without_cuda() {
        // SAFETY: null metadata must reject before CUDA submission.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_kv8_chunk_warp_shared_kv_prevalidated_submit_v1(
                0,
                std::ptr::null_mut(),
                std::ptr::null(),
            )
        };
        assert!(super::CudaRuntimeError::from_result(super::KV8_CHUNK_OPERATION, result).is_err());
    }

    #[cfg(feature = "qwen35-kv8-full-attention")]
    #[test]
    fn warp_ordered_entry_rejects_null_without_cuda() {
        // SAFETY: null metadata must be rejected before any CUDA submission.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_kv8_chunk_warp_ordered_prevalidated_submit_v1(
                0,
                std::ptr::null_mut(),
                std::ptr::null(),
            )
        };
        assert!(super::CudaRuntimeError::from_result(super::KV8_CHUNK_OPERATION, result).is_err());
    }

    #[cfg(feature = "qwen35-kv8-full-attention")]
    #[test]
    fn register_softmax_entry_rejects_null_without_cuda() {
        // SAFETY: null stream/args must fail before any CUDA context or submit.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_kv8_chunk_register_softmax_prevalidated_submit_v1(
                0,
                core::ptr::null_mut(),
                core::ptr::null(),
            )
        };
        assert!(super::CudaRuntimeError::from_result(super::OPERATION, result).is_err());
    }



    #[test]
    fn unit_scale_shortcut_excludes_nonfinite_maxima() {
        let shortcut = |previous: f32, next: f32| previous.is_finite() && previous == next;
        for value in [-100.0_f32, -0.0, 0.0, 1.0, f32::MAX] {
            assert!(shortcut(value, value));
            assert_eq!((value - value).exp().to_bits(), 1.0_f32.to_bits());
        }
        assert!(shortcut(-0.0, 0.0));
        assert!(!shortcut(0.0, 1.0));
        assert!(!shortcut(f32::INFINITY, f32::INFINITY));
        assert!(!shortcut(f32::NEG_INFINITY, f32::NEG_INFINITY));
        assert!(!shortcut(f32::NAN, 1.0));
    }

    #[test]
    fn paired_projection_entries_reject_invalid_metadata_without_cuda() {
        // SAFETY: null plan/stream/args reject before any CUDA submission.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_cublas_projection_pair_prevalidated_submit_v1(
                0,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null(),
            )
        };
        assert!(super::CudaRuntimeError::from_result(super::OPERATION, result).is_err());
        for stage in [0, 1, 2, 3, u32::MAX] {
            // SAFETY: metadata/stage rejection has no CUDA side effect.
            let result = unsafe {
                augur_cuda_sys::augur_cuda_qwen35_prefill_cublas_projection_pair_profile_stage_prevalidated_submit_v1(
                    0, core::ptr::null_mut(), core::ptr::null_mut(), core::ptr::null(), stage,
                )
            };
            assert!(super::CudaRuntimeError::from_result(super::OPERATION, result).is_err());
        }
    }

    #[test]
    fn paired_nvfp4_addresses_preserve_every_adjacent_scalar_lane() {
        for index in (0_usize..64 * 3).step_by(2) {
            let block = index / 64;
            let lane = index % 64;
            let pair_scale = block * 36 + lane / 16;
            let pair_data = block * 36 + 4 + lane / 16 * 8 + (lane % 16 & 7);
            let pair_shift = if lane % 16 < 8 { 0 } else { 4 };
            for component in 0..2 {
                let i = index + component;
                let scalar_lane = i % 64;
                assert_eq!(pair_scale, i / 64 * 36 + scalar_lane / 16);
                assert_eq!(
                    pair_data + component,
                    i / 64 * 36 + 4 + scalar_lane / 16 * 8 + (scalar_lane % 16 & 7)
                );
                assert_eq!(pair_shift, if scalar_lane % 16 < 8 { 0 } else { 4 });
                for byte in 0_u8..=255 {
                    let scalar_nibble = if scalar_lane % 16 < 8 {
                        byte & 15
                    } else {
                        byte >> 4
                    };
                    assert_eq!((byte >> pair_shift) & 15, scalar_nibble);
                }
            }
        }
    }

    #[test]
    fn projection_profile_stages_reject_invalid_metadata_without_cuda() {
        for stage in [0, 1, 2, 3, u32::MAX] {
            // SAFETY: null plan/stream/args or invalid stage rejects before CUDA.
            let result = unsafe {
                augur_cuda_sys::augur_cuda_qwen35_prefill_cublas_projection_profile_stage_prevalidated_submit_v1(
                    0,
                    core::ptr::null_mut(),
                    core::ptr::null_mut(),
                    core::ptr::null(),
                    stage,
                )
            };
            assert!(super::CudaRuntimeError::from_result(super::OPERATION, result).is_err());
        }
    }

    #[cfg(feature = "qwen35-kv8-full-attention")]
    #[test]
    fn shared_softmax_tile16_native_entry_rejects_null_without_cuda() {
        // SAFETY: null stream/args reject during metadata admission.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_kv8_chunk_shared_softmax_tile16_prevalidated_submit_v1(
                0,
                core::ptr::null_mut(),
                core::ptr::null(),
            )
        };
        assert!(super::CudaRuntimeError::from_result(super::OPERATION, result).is_err());
    }

    #[test]
    fn shared_gdn_scalar_native_entry_rejects_null_stream_without_cuda() {
        // SAFETY: invalid null stream must be rejected before any CUDA action.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_prefill_linear_chunk_shared_scalars_prevalidated_submit_v1(
                0,
                core::ptr::null_mut(),
                core::ptr::null(),
            )
        };
        assert!(super::CudaRuntimeError::from_result(super::OPERATION, result).is_err());
    }

    use super::*;
    use crate::cuda_runtime::test_buffer_view;
    use augur_core::PhysicalLayoutId;

    const ROWS: usize = 64;
    const COLS: usize = 64;
    const TOKENS: usize = 8;







    fn q8_weight(address: u64) -> CudaQ8_0WeightViewV1 {
        CudaQ8_0WeightViewV1::new(
            test_buffer_view(address, 0, 0, ROWS * (COLS / 32) * 34),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_0_CUDA_V1,
            ROWS,
            COLS,
        )
        .unwrap()
    }

    fn nvfp4_weight(address: u64) -> CudaNvfp4Sm70WeightViewV1 {
        CudaNvfp4Sm70WeightViewV1::new(
            test_buffer_view(address, 0, 0, ROWS * (COLS / 64) * 36),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_NVFP4_CUDA_SM70_V1,
            ROWS,
            COLS,
        )
        .unwrap()
    }

    fn bf16_weight(address: u64) -> CudaBf16WeightViewV1 {
        CudaBf16WeightViewV1::new(
            test_buffer_view(address, 0, 0, ROWS * COLS * 2),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_BF16_CUDA_V1,
            ROWS,
            COLS,
        )
        .unwrap()
    }

    fn input(address: u64) -> CudaQ8_1ActivationViewV1 {
        CudaQ8_1ActivationViewV1::new(
            test_buffer_view(address, 0, 0, TOKENS * (COLS / 32) * 36),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_1_ACTIVATION_CUDA_V1,
            TOKENS * COLS,
        )
        .unwrap()
    }

    fn output(address: u64) -> CudaF32View {
        CudaF32View::new(
            test_buffer_view(address, 0, 0, TOKENS * ROWS * 4),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            TOKENS * ROWS,
        )
        .unwrap()
    }

    #[cfg(feature = "qwen35-kv8-full-attention")]
    fn f32_cache(address: u64, elements: usize) -> CudaF32View {
        CudaF32View::new(
            test_buffer_view(address, 0, 0, elements * core::mem::size_of::<f32>()),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            elements,
        )
        .unwrap()
    }

    #[cfg(feature = "qwen35-kv8-full-attention")]
    fn kv8_quantize_views(capacity: usize) -> CudaQwen35PrefillKv8CacheQuantizeViewsV1 {
        let f32_elements = capacity * AUGUR_CUDA_QWEN35_FULL_ATTENTION_KV_ELEMENTS_V1;
        let q8_bytes = capacity * AUGUR_CUDA_QWEN35_KV8_CACHE_ROW_BYTES_V3;
        CudaQwen35PrefillKv8CacheQuantizeViewsV1 {
            key_cache_f32: f32_cache(0x6000_0000, f32_elements),
            value_cache_f32: f32_cache(0x7000_0000, f32_elements),
            key_cache_q8_0: test_buffer_view(0x8000_0000, 0, 0, q8_bytes),
            value_cache_q8_0: test_buffer_view(0x9000_0000, 0, 0, q8_bytes),
        }
    }

    #[cfg(all(
        feature = "qwen35-prefill-cublas",
        feature = "qwen35-kv8-full-attention"
    ))]
    fn kv8_chunk_views(
        tokens: usize,
        capacity: usize,
    ) -> CudaQwen35PrefillKv8ChunkAttentionViewsV1 {
        let q_elements = tokens * AUGUR_CUDA_QWEN35_FULL_ATTENTION_QUERY_ELEMENTS_V1;
        let kv_elements = tokens * AUGUR_CUDA_QWEN35_FULL_ATTENTION_KV_ELEMENTS_V1;
        let rope_elements = tokens * (AUGUR_CUDA_QWEN35_FULL_ATTENTION_ROTARY_DIM_V1 / 2);
        let cache_bytes = capacity * AUGUR_CUDA_QWEN35_KV8_CACHE_ROW_BYTES_V3;
        CudaQwen35PrefillKv8ChunkAttentionViewsV1 {
            q_full: f32_cache(0x01_0000_0000, q_elements * 2),
            k_raw: f32_cache(0x02_0000_0000, kv_elements),
            value: f32_cache(0x03_0000_0000, kv_elements),
            q_norm_weight: f32_cache(0x04_0000_0000, AUGUR_CUDA_QWEN35_FULL_ATTENTION_HEAD_DIM_V1),
            k_norm_weight: f32_cache(0x05_0000_0000, AUGUR_CUDA_QWEN35_FULL_ATTENTION_HEAD_DIM_V1),
            rope_cos: f32_cache(0x06_0000_0000, rope_elements),
            rope_sin: f32_cache(0x07_0000_0000, rope_elements),
            key_work: f32_cache(0x08_0000_0000, kv_elements),
            value_work: f32_cache(0x09_0000_0000, kv_elements),
            key_cache: test_buffer_view(0x0a_0000_0000, 0, 0, cache_bytes),
            value_cache: test_buffer_view(0x0b_0000_0000, 0, 0, cache_bytes),
            query: f32_cache(0x0c_0000_0000, q_elements),
            gate: f32_cache(0x0d_0000_0000, q_elements),
            attention: f32_cache(0x0e_0000_0000, q_elements),
            gated: f32_cache(0x0f_0000_0000, q_elements),
        }
    }

    fn raw(weight: CudaQwen35PrefillWeightV1) -> AugurCudaQwen35PrefillCublasProjectionArgsV1 {
        bind_projection_raw(
            0,
            TOKENS,
            test_buffer_view(0x4000_0000, 0, 0, ROWS * COLS * 2),
            test_buffer_view(0x5000_0000, 0, 0, TOKENS * COLS * 2),
            weight,
            input(0x2000_0000),
            output(0x3000_0000),
            TOKENS,
        )
        .unwrap()
    }













    #[test]
    fn q8_nvfp4_and_bf16_wire_extents_bind_exactly() {
        let q8 = raw(CudaQwen35PrefillWeightV1::Q8_0(q8_weight(0x1000_0000)));
        assert_eq!(
            q8.weight_format,
            AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_Q8_0_V1
        );
        assert_eq!(q8.tensor_scale, 1.0);
        assert_eq!(q8.tokens, TOKENS as u32);
        assert_eq!(q8.rows, ROWS as u32);
        assert_eq!(q8.cols, COLS as u32);

        let nvfp4 = raw(CudaQwen35PrefillWeightV1::Nvfp4 {
            view: nvfp4_weight(0x1000_0000),
            tensor_scale: 0.5,
        });
        assert_eq!(
            nvfp4.weight_format,
            AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_NVFP4_V1
        );
        assert_eq!(nvfp4.tensor_scale, 0.5);

        let bf16 = raw(CudaQwen35PrefillWeightV1::Bf16(bf16_weight(0x1000_0000)));
        assert_eq!(
            bf16.weight_format,
            AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_BF16_V1
        );
        assert_eq!(bf16.weight_len_bytes, (ROWS * COLS * 2) as u64);
        assert_eq!(bf16.tensor_scale, 1.0);
    }

    #[test]
    fn packed_small_batch_binds_both_formats_and_all_tails_without_scratch() {
        for weight in [
            CudaQwen35PrefillWeightV1::Q8_0(q8_weight(0x1000_0000)),
            CudaQwen35PrefillWeightV1::Nvfp4 {
                view: nvfp4_weight(0x1000_0000),
                tensor_scale: 0.75,
            },
        ] {
            for tokens in 1..=8 {
                let input = CudaQ8_1ActivationViewV1::new(
                    test_buffer_view(0x2000_0000, 0, 0, tokens * COLS / 32 * 36),
                    PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_1_ACTIVATION_CUDA_V1,
                    tokens * COLS,
                )
                .unwrap();
                let output = CudaF32View::new(
                    test_buffer_view(0x3000_0000, 0, 0, tokens * ROWS * 4),
                    PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                    tokens * ROWS,
                )
                .unwrap();
                let raw = bind_packed_projection_raw(0, weight, input, output, tokens).unwrap();
                assert_eq!(raw.tokens, tokens as u32);
                assert_eq!(raw.weight_format, weight.format());
                assert_eq!(raw.weight_len_bytes, weight.view().len_bytes() as u64);
                assert_eq!(raw.output_len_bytes, (tokens * ROWS * 4) as u64);
            }
        }
    }

    #[test]
    fn packed_small_batch_head_extent_needs_no_decoded_weight_view() {
        let (rows, cols, tokens) = (248320, 5120, 8);
        let weight = CudaQ8_0WeightViewV1::new(
            test_buffer_view(0x10_0000_0000, 0, 0, rows * cols / 32 * 34),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_0_CUDA_V1,
            rows,
            cols,
        )
        .unwrap();
        let input = CudaQ8_1ActivationViewV1::new(
            test_buffer_view(0x20_0000_0000, 0, 0, tokens * cols / 32 * 36),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_1_ACTIVATION_CUDA_V1,
            tokens * cols,
        )
        .unwrap();
        let output = CudaF32View::new(
            test_buffer_view(0x30_0000_0000, 0, 0, tokens * rows * 4),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            tokens * rows,
        )
        .unwrap();
        let raw = bind_packed_projection_raw(
            0,
            CudaQwen35PrefillWeightV1::Q8_0(weight),
            input,
            output,
            tokens,
        )
        .unwrap();
        assert_eq!(raw.weight_len_bytes, 1_350_860_800);
        assert_eq!(raw.output_len_bytes, 7_946_240);
        assert_eq!(raw.input_len_bytes, 46_080);
    }

    #[test]
    fn packed_small_batch_rejects_format_extent_alias_and_device_drift() {
        let weight = CudaQwen35PrefillWeightV1::Q8_0(q8_weight(0x1000_0000));
        for tokens in [0, 7, 9, usize::MAX] {
            assert!(bind_packed_projection_raw(
                0,
                weight,
                input(0x2000_0000),
                output(0x3000_0000),
                tokens
            )
            .is_err());
        }
        assert!(bind_packed_projection_raw(
            0,
            CudaQwen35PrefillWeightV1::Bf16(bf16_weight(0x1000_0000)),
            input(0x2000_0000),
            output(0x3000_0000),
            TOKENS
        )
        .is_err());
        for scale in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(bind_packed_projection_raw(
                0,
                CudaQwen35PrefillWeightV1::Nvfp4 {
                    view: nvfp4_weight(0x1000_0000),
                    tensor_scale: scale
                },
                input(0x2000_0000),
                output(0x3000_0000),
                TOKENS
            )
            .is_err());
        }
        assert_eq!(
            bind_packed_projection_raw(0, weight, input(0x2000_0000), output(0x2000_0000), TOKENS)
                .unwrap_err()
                .kind,
            CudaRuntimeErrorKind::OverlappingDeviceRanges
        );
        assert_eq!(
            bind_packed_projection_raw(1, weight, input(0x2000_0000), output(0x3000_0000), TOKENS)
                .unwrap_err()
                .kind,
            CudaRuntimeErrorKind::DeviceMismatch
        );
    }

    #[test]
    fn packed_small_batch_native_null_entries_fail_without_cuda() {
        for exact_tiles in [0, 1, 2, u32::MAX] {
            // SAFETY: null stream/arguments reject before any CUDA call.
            for result in unsafe {
                [
                augur_cuda_sys::augur_cuda_qwen35_packed_small_batch_projection_preflight_v1(
                    0, core::ptr::null_mut(), core::ptr::null(), exact_tiles),
                augur_cuda_sys::augur_cuda_qwen35_packed_small_batch_projection_prevalidated_submit_v1(
                    0, core::ptr::null_mut(), core::ptr::null(), exact_tiles),
            ]
            } {
                assert!(CudaRuntimeError::from_result(OPERATION, result).is_err());
            }
        }
    }

    #[test]
    #[ignore = "requires exclusive SM70 GPU; CPU/null admission tests run by default"]
    fn packed_small_batch_native_live_metadata_and_stream_guards() {
        let mut stream = OwnedCudaStream::create(0).unwrap();
        let mut foreign = OwnedCudaStream::create(0).unwrap();
        let weight = OwnedCudaBuffer::allocate(0, ROWS * COLS / 32 * 34).unwrap();
        let activation = OwnedCudaBuffer::allocate(0, TOKENS * COLS / 32 * 36).unwrap();
        let output = OwnedCudaBuffer::allocate(0, TOKENS * ROWS * 4).unwrap();
        let before = crate::cuda_device_allocation_snapshot();
        let binding = CudaQwen35PackedSmallBatchProjectionV1::bind(
            &stream,
            CudaQwen35PrefillWeightV1::Q8_0(
                CudaQ8_0WeightViewV1::new(
                    weight.view(0, weight.len_bytes()).unwrap(),
                    PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_0_CUDA_V1,
                    ROWS,
                    COLS,
                )
                .unwrap(),
            ),
            CudaQ8_1ActivationViewV1::new(
                activation.view(0, activation.len_bytes()).unwrap(),
                PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_1_ACTIVATION_CUDA_V1,
                TOKENS * COLS,
            )
            .unwrap(),
            CudaF32View::new(
                output.view(0, output.len_bytes()).unwrap(),
                PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                TOKENS * ROWS,
            )
            .unwrap(),
            TOKENS,
            true,
        )
        .unwrap();
        let after = crate::cuda_device_allocation_snapshot();
        assert_eq!(before.successful_allocations, after.successful_allocations);
        assert_eq!(before.successful_frees, after.successful_frees);
        assert_eq!(before.live_bytes, after.live_bytes);
        assert!(binding.submit(&foreign).is_err());
        let mutations: &[fn(&mut AugurCudaQwen35PrefillCublasProjectionArgsV1)] = &[
            |a| a.abi_version = 0,
            |a| a.struct_size = 0,
            |a| a.reserved = 1,
            |a| a.tokens = 0,
            |a| a.tokens = 9,
            |a| a.rows = 0,
            |a| a.cols = 1,
            |a| a.weight_format = AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_WEIGHT_BF16_V1,
            |a| a.tensor_scale = f32::NAN,
            |a| a.tensor_scale = 0.5,
            |a| a.weight_len_bytes -= 1,
            |a| a.input_len_bytes -= 1,
            |a| a.output_len_bytes -= 1,
            |a| a.output_f32 = a.input_q8_1,
            |a| a.input_q8_1 += 2,
        ];
        for mutate in mutations {
            let mut raw = *binding.as_raw();
            mutate(&mut raw);
            // SAFETY: deliberately malformed metadata must reject before launch;
            // all original buffers and the stream remain live through the test.
            unsafe {
                let preflight =
                    augur_cuda_sys::augur_cuda_qwen35_packed_small_batch_projection_preflight_v1(
                        0,
                        stream.raw().unwrap(),
                        &raw,
                        1,
                    );
                assert!(CudaRuntimeError::from_result(OPERATION, preflight).is_err());
                let submit = augur_cuda_sys::augur_cuda_qwen35_packed_small_batch_projection_prevalidated_submit_v1(
                    0, stream.raw().unwrap(), &raw, 1);
                assert!(CudaRuntimeError::from_result(OPERATION, submit).is_err());
            }
        }
        // Valid extents but a shifted output exceeds the actual allocation;
        // this is a preflight-only check, never a prevalidated unsafe submit.
        let mut shifted = *binding.as_raw();
        shifted.output_f32 += 4;
        unsafe {
            let result = augur_cuda_sys::augur_cuda_qwen35_packed_small_batch_projection_preflight_v1(
                0,
                stream.raw().unwrap(),
                &shifted,
                1,
            );
            assert!(CudaRuntimeError::from_result(OPERATION, result).is_err());
            for policy in [2, u32::MAX] {
                let result = augur_cuda_sys::augur_cuda_qwen35_packed_small_batch_projection_preflight_v1(
                    0,
                    stream.raw().unwrap(),
                    binding.as_raw(),
                    policy,
                );
                assert!(CudaRuntimeError::from_result(OPERATION, result).is_err());
            }
        }
        foreign.close().unwrap();
        stream.close().unwrap();
    }







    #[test]
    fn binding_rejects_aliases_bad_scale_and_token_overflow() {
        let weight = CudaQwen35PrefillWeightV1::Nvfp4 {
            view: nvfp4_weight(0x1000_0000),
            tensor_scale: f32::NAN,
        };
        assert_eq!(
            bind_projection_raw(
                0,
                TOKENS,
                test_buffer_view(0x4000_0000, 0, 0, ROWS * COLS * 2),
                test_buffer_view(0x5000_0000, 0, 0, TOKENS * COLS * 2),
                weight,
                input(0x2000_0000),
                output(0x3000_0000),
                TOKENS,
            )
            .unwrap_err()
            .kind,
            CudaRuntimeErrorKind::InvalidMetadata
        );
        assert_eq!(
            bind_projection_raw(
                0,
                TOKENS - 1,
                test_buffer_view(0x4000_0000, 0, 0, ROWS * COLS * 2),
                test_buffer_view(0x5000_0000, 0, 0, TOKENS * COLS * 2),
                CudaQwen35PrefillWeightV1::Q8_0(q8_weight(0x1000_0000)),
                input(0x2000_0000),
                output(0x3000_0000),
                TOKENS,
            )
            .unwrap_err()
            .kind,
            CudaRuntimeErrorKind::InvalidDimensions
        );
        assert_eq!(
            bind_projection_raw(
                0,
                TOKENS,
                test_buffer_view(0x1000_0000, 0, 0, ROWS * COLS * 2),
                test_buffer_view(0x5000_0000, 0, 0, TOKENS * COLS * 2),
                CudaQwen35PrefillWeightV1::Q8_0(q8_weight(0x1000_0000)),
                input(0x2000_0000),
                output(0x3000_0000),
                TOKENS,
            )
            .unwrap_err()
            .kind,
            CudaRuntimeErrorKind::OverlappingDeviceRanges
        );
    }

    #[cfg(all(
        feature = "qwen35-prefill-cublas",
        feature = "qwen35-kv8-full-attention"
    ))]
    #[test]
    fn kv8_chunk_prefill_binds_chunk_extremes_at_256k_capacity() {
        const CAPACITY: usize = 262_144;
        for tokens in [
            1,
            8,
            AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_COMPILED_MAX_TOKENS_V1 as usize,
        ] {
            let position_base = CAPACITY - tokens;
            let raw = bind_prefill_kv8_chunk_attention_raw(
                0,
                kv8_chunk_views(tokens, CAPACITY),
                tokens,
                position_base,
                CAPACITY,
                1.0e-6,
            )
            .unwrap();
            assert_eq!(raw.tokens, tokens as u32);
            assert_eq!(raw.position_base, position_base as u32);
            assert_eq!(raw.capacity, CAPACITY as u32);
            assert_eq!(
                raw.key_cache_len_bytes,
                (CAPACITY * AUGUR_CUDA_QWEN35_KV8_CACHE_ROW_BYTES_V3) as u64
            );
            assert_eq!(
                raw.key_work_len_bytes,
                (tokens
                    * AUGUR_CUDA_QWEN35_FULL_ATTENTION_KV_ELEMENTS_V1
                    * core::mem::size_of::<f32>()) as u64
            );
        }
    }

    #[cfg(all(
        feature = "qwen35-prefill-cublas",
        feature = "qwen35-kv8-full-attention"
    ))]
    #[test]
    fn kv8_chunk_prefill_rejects_causal_and_scalar_bounds() {
        const CAPACITY: usize = 8;
        for (tokens, position_base, capacity, epsilon, expected) in [
            (
                0,
                0,
                CAPACITY,
                1.0e-6,
                CudaRuntimeErrorKind::InvalidDimensions,
            ),
            (
                2,
                7,
                CAPACITY,
                1.0e-6,
                CudaRuntimeErrorKind::InvalidDimensions,
            ),
            (
                2,
                usize::MAX,
                CAPACITY,
                1.0e-6,
                CudaRuntimeErrorKind::RangeOverflow,
            ),
            (1, 0, 0, 1.0e-6, CudaRuntimeErrorKind::InvalidDimensions),
            (1, 0, CAPACITY, 0.0, CudaRuntimeErrorKind::InvalidDimensions),
            (
                1,
                0,
                CAPACITY,
                f32::NAN,
                CudaRuntimeErrorKind::InvalidDimensions,
            ),
        ] {
            assert_eq!(
                bind_prefill_kv8_chunk_attention_raw(
                    0,
                    kv8_chunk_views(tokens.max(1), capacity.max(1)),
                    tokens,
                    position_base,
                    capacity,
                    epsilon,
                )
                .unwrap_err()
                .kind,
                expected
            );
        }

        let too_many = AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_COMPILED_MAX_TOKENS_V1 as usize + 1;
        assert_eq!(
            bind_prefill_kv8_chunk_attention_raw(
                0,
                kv8_chunk_views(too_many, too_many),
                too_many,
                0,
                too_many,
                1.0e-6,
            )
            .unwrap_err()
            .kind,
            CudaRuntimeErrorKind::InvalidDimensions
        );
        let over_capacity = AUGUR_CUDA_QWEN35_KV8_FULL_ATTENTION_MAX_CAPACITY_V3 + 1;
        assert_eq!(
            bind_prefill_kv8_chunk_attention_raw(
                0,
                kv8_chunk_views(1, over_capacity),
                1,
                0,
                over_capacity,
                1.0e-6,
            )
            .unwrap_err()
            .kind,
            CudaRuntimeErrorKind::InvalidDimensions
        );
    }

    #[cfg(all(
        feature = "qwen35-prefill-cublas",
        feature = "qwen35-kv8-full-attention"
    ))]
    #[test]
    fn kv8_chunk_prefill_rejects_extent_alias_alignment_and_device_drift() {
        const TOKENS: usize = 8;
        const CAPACITY: usize = 32;

        let mut bad_scratch = kv8_chunk_views(TOKENS, CAPACITY);
        bad_scratch.key_work = f32_cache(
            0x08_0000_0000,
            TOKENS * AUGUR_CUDA_QWEN35_FULL_ATTENTION_KV_ELEMENTS_V1 - 1,
        );
        assert_eq!(
            bind_prefill_kv8_chunk_attention_raw(0, bad_scratch, TOKENS, 0, CAPACITY, 1.0e-6,)
                .unwrap_err()
                .kind,
            CudaRuntimeErrorKind::InvalidDimensions
        );

        let mut bad_cache = kv8_chunk_views(TOKENS, CAPACITY);
        bad_cache.value_cache = test_buffer_view(
            0x0b_0000_0000,
            0,
            0,
            CAPACITY * AUGUR_CUDA_QWEN35_KV8_CACHE_ROW_BYTES_V3 - 1,
        );
        assert_eq!(
            bind_prefill_kv8_chunk_attention_raw(0, bad_cache, TOKENS, 0, CAPACITY, 1.0e-6,)
                .unwrap_err()
                .kind,
            CudaRuntimeErrorKind::InvalidDimensions
        );

        let mut aliased = kv8_chunk_views(TOKENS, CAPACITY);
        aliased.key_work = aliased.k_raw;
        assert_eq!(
            bind_prefill_kv8_chunk_attention_raw(0, aliased, TOKENS, 0, CAPACITY, 1.0e-6,)
                .unwrap_err()
                .kind,
            CudaRuntimeErrorKind::OverlappingDeviceRanges
        );

        let mut misaligned = kv8_chunk_views(TOKENS, CAPACITY);
        misaligned.key_cache = test_buffer_view(
            0x0a_0000_0001,
            0,
            0,
            CAPACITY * AUGUR_CUDA_QWEN35_KV8_CACHE_ROW_BYTES_V3,
        );
        assert_eq!(
            bind_prefill_kv8_chunk_attention_raw(0, misaligned, TOKENS, 0, CAPACITY, 1.0e-6,)
                .unwrap_err()
                .kind,
            CudaRuntimeErrorKind::MisalignedDeviceAddress
        );

        let mut foreign = kv8_chunk_views(TOKENS, CAPACITY);
        foreign.value_cache = test_buffer_view(
            0x0b_0000_0000,
            1,
            0,
            CAPACITY * AUGUR_CUDA_QWEN35_KV8_CACHE_ROW_BYTES_V3,
        );
        assert_eq!(
            bind_prefill_kv8_chunk_attention_raw(0, foreign, TOKENS, 0, CAPACITY, 1.0e-6,)
                .unwrap_err()
                .kind,
            CudaRuntimeErrorKind::DeviceMismatch
        );
    }

    #[cfg(feature = "qwen35-kv8-full-attention")]
    #[test]
    fn kv8_prefill_cache_quantizer_binds_exact_geometry() {
        const CAPACITY: usize = 8;
        let raw =
            bind_prefill_kv8_cache_quantize_raw(0, kv8_quantize_views(CAPACITY), CAPACITY, 3, 5)
                .unwrap();

        assert_eq!(raw.capacity, CAPACITY as u32);
        assert_eq!(raw.position_base, 3);
        assert_eq!(raw.tokens, 5);
        assert_eq!(
            raw.key_cache_f32_len_bytes,
            (CAPACITY * AUGUR_CUDA_QWEN35_FULL_ATTENTION_KV_ELEMENTS_V1 * core::mem::size_of::<f32>())
                as u64
        );
        assert_eq!(
            raw.key_cache_q8_0_len_bytes,
            (CAPACITY * AUGUR_CUDA_QWEN35_KV8_CACHE_ROW_BYTES_V3) as u64
        );
        assert_eq!(
            raw.kv_heads,
            AUGUR_CUDA_QWEN35_FULL_ATTENTION_KV_HEADS_V1 as u32
        );
        assert_eq!(
            raw.head_dim,
            AUGUR_CUDA_QWEN35_FULL_ATTENTION_HEAD_DIM_V1 as u32
        );
    }

    #[cfg(feature = "qwen35-kv8-full-attention")]
    #[test]
    fn kv8_prefill_cache_quantizer_rejects_invalid_causal_extent() {
        const CAPACITY: usize = 8;
        for (position_base, tokens, expected) in [
            (0, 0, CudaRuntimeErrorKind::InvalidDimensions),
            (7, 2, CudaRuntimeErrorKind::InvalidDimensions),
            (usize::MAX, 2, CudaRuntimeErrorKind::RangeOverflow),
        ] {
            assert_eq!(
                bind_prefill_kv8_cache_quantize_raw(
                    0,
                    kv8_quantize_views(CAPACITY),
                    CAPACITY,
                    position_base,
                    tokens,
                )
                .unwrap_err()
                .kind,
                expected
            );
        }
        assert_eq!(
            bind_prefill_kv8_cache_quantize_raw(0, kv8_quantize_views(CAPACITY), 0, 0, 1)
                .unwrap_err()
                .kind,
            CudaRuntimeErrorKind::InvalidDimensions
        );
    }

    #[cfg(feature = "qwen35-kv8-full-attention")]
    #[test]
    fn kv8_prefill_cache_quantizer_rejects_extent_and_device_drift() {
        const CAPACITY: usize = 8;
        let f32_elements = CAPACITY * AUGUR_CUDA_QWEN35_FULL_ATTENTION_KV_ELEMENTS_V1;
        let q8_bytes = CAPACITY * AUGUR_CUDA_QWEN35_KV8_CACHE_ROW_BYTES_V3;

        let mut bad_f32 = kv8_quantize_views(CAPACITY);
        bad_f32.key_cache_f32 = f32_cache(0x6000_0000, f32_elements - 1);
        assert_eq!(
            bind_prefill_kv8_cache_quantize_raw(0, bad_f32, CAPACITY, 0, 1)
                .unwrap_err()
                .kind,
            CudaRuntimeErrorKind::InvalidDimensions
        );

        let mut bad_q8 = kv8_quantize_views(CAPACITY);
        bad_q8.value_cache_q8_0 = test_buffer_view(0x9000_0000, 0, 0, q8_bytes - 1);
        assert_eq!(
            bind_prefill_kv8_cache_quantize_raw(0, bad_q8, CAPACITY, 0, 1)
                .unwrap_err()
                .kind,
            CudaRuntimeErrorKind::InvalidDimensions
        );

        let mut foreign = kv8_quantize_views(CAPACITY);
        foreign.value_cache_q8_0 = test_buffer_view(0x9000_0000, 1, 0, q8_bytes);
        assert_eq!(
            bind_prefill_kv8_cache_quantize_raw(0, foreign, CAPACITY, 0, 1)
                .unwrap_err()
                .kind,
            CudaRuntimeErrorKind::DeviceMismatch
        );
    }

    #[cfg(feature = "qwen35-kv8-full-attention")]
    #[test]
    fn kv8_prefill_cache_quantizer_rejects_source_destination_alias() {
        const CAPACITY: usize = 8;
        let mut views = kv8_quantize_views(CAPACITY);
        let q8_bytes = CAPACITY * AUGUR_CUDA_QWEN35_KV8_CACHE_ROW_BYTES_V3;
        views.key_cache_q8_0 = test_buffer_view(0x6000_0000, 0, 0, q8_bytes);

        assert_eq!(
            bind_prefill_kv8_cache_quantize_raw(0, views, CAPACITY, 0, 1)
                .unwrap_err()
                .kind,
            CudaRuntimeErrorKind::OverlappingDeviceRanges
        );
    }
}

#[cfg(test)]
mod checkpoint_chunk_tests;

#[cfg(feature = "qwen35-vision-affine")]
#[path = "cuda_qwen35_vision_affine.rs"]
mod vision_affine;
#[cfg(feature = "qwen35-vision-affine")]
pub use vision_affine::{CudaQwen35VisionAffineViewsV1, CudaQwen35VisionF16AffineV1};
