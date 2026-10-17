#![cfg(feature = "cuda")]

//! Checked binding for the opt-in Qwen35 layer-3 full-attention boundary.
//!
//! Q and K are supplied in the normalized/rotated layout produced by the
//! model reference path.  The native operation appends one KV row, performs
//! causal grouped-query attention, applies the output gate, and writes the
//! attention and gated vectors.  It deliberately does not own projection
//! weights or a request scheduler.

#[cfg(feature = "qwen35-kv8-full-attention")]
use crate::CudaBufferView;
use crate::{CudaF32View, CudaRuntimeError, CudaRuntimeErrorKind, OwnedCudaStream};
use augur_cuda_sys::{
    AugurCudaQwen35FullAttentionArgsV1, AUGUR_CUDA_QWEN35_FULL_ATTENTION_HEAD_DIM_V1,
    AUGUR_CUDA_QWEN35_FULL_ATTENTION_KV_ELEMENTS_V1, AUGUR_CUDA_QWEN35_FULL_ATTENTION_KV_HEADS_V1,
    AUGUR_CUDA_QWEN35_FULL_ATTENTION_MAX_CAPACITY_V1, AUGUR_CUDA_QWEN35_FULL_ATTENTION_QUERY_ELEMENTS_V1,
    AUGUR_CUDA_QWEN35_FULL_ATTENTION_QUERY_HEADS_V1, AUGUR_CUDA_QWEN35_FULL_ATTENTION_ROTARY_DIM_V1,
};
#[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
use augur_cuda_sys::{
    AugurCudaQwen35FullAttentionArgsV2, AUGUR_CUDA_QWEN35_FULL_ATTENTION_FUSED_Q8_BYTES_V2,
    AUGUR_CUDA_QWEN35_FULL_ATTENTION_MERGE_SEGMENTS_V2,
    AUGUR_CUDA_QWEN35_FULL_ATTENTION_MERGE_THREADS_V2,
    AUGUR_CUDA_QWEN35_FULL_ATTENTION_PARTIAL_ELEMENTS_V2,
    AUGUR_CUDA_QWEN35_FULL_ATTENTION_PARTIAL_STRIDE_V2, AUGUR_CUDA_QWEN35_FULL_ATTENTION_SPLITS_V2,
    AUGUR_CUDA_QWEN35_FULL_ATTENTION_STAGE1_THREADS_V2,
    AUGUR_CUDA_QWEN35_FULL_ATTENTION_TILE_POSITIONS_V2,
    AUGUR_CUDA_QWEN35_FULL_ATTENTION_WORKSPACE_ELEMENTS_V2,
};
#[cfg(feature = "qwen35-kv8-batched-attention")]
use augur_cuda_sys::{
    AugurCudaQwen35Kv8BatchedAttentionArgsV4, AUGUR_CUDA_QWEN35_KV8_BATCHED_ATTENTION_MAX_TOKENS_V4,
};
#[cfg(feature = "qwen35-kv8-full-attention")]
use augur_cuda_sys::{
    AugurCudaQwen35Kv8FullAttentionArgsV3, AUGUR_CUDA_QWEN35_KV8_BLOCKS_PER_HEAD_V3,
    AUGUR_CUDA_QWEN35_KV8_BLOCK_BYTES_V3, AUGUR_CUDA_QWEN35_KV8_BLOCK_ELEMENTS_V3,
    AUGUR_CUDA_QWEN35_KV8_CACHE_ROW_BYTES_V3, AUGUR_CUDA_QWEN35_KV8_FULL_ATTENTION_MAX_CAPACITY_V3,
    AUGUR_CUDA_QWEN35_KV8_HEAD_ROW_BYTES_V3,
};

const OPERATION: &str = "cuda_qwen35_full_attention_v1";
#[cfg(feature = "qwen35-kv8-full-attention")]
const KV8_OPERATION: &str = "cuda_qwen35_kv8_full_attention_v3";
#[cfg(feature = "qwen35-kv8-batched-attention")]
const KV8_BATCHED_OPERATION: &str = "cuda_qwen35_kv8_batched_attention_v4";

pub const QWEN35_FULL_ATTENTION_QUERY_HEADS: usize = AUGUR_CUDA_QWEN35_FULL_ATTENTION_QUERY_HEADS_V1;
pub const QWEN35_FULL_ATTENTION_KV_HEADS: usize = AUGUR_CUDA_QWEN35_FULL_ATTENTION_KV_HEADS_V1;
pub const QWEN35_FULL_ATTENTION_HEAD_DIM: usize = AUGUR_CUDA_QWEN35_FULL_ATTENTION_HEAD_DIM_V1;
pub const QWEN35_FULL_ATTENTION_ROTARY_DIM: usize = AUGUR_CUDA_QWEN35_FULL_ATTENTION_ROTARY_DIM_V1;
pub const QWEN35_FULL_ATTENTION_QUERY_ELEMENTS: usize =
    AUGUR_CUDA_QWEN35_FULL_ATTENTION_QUERY_ELEMENTS_V1;
pub const QWEN35_FULL_ATTENTION_KV_ELEMENTS: usize = AUGUR_CUDA_QWEN35_FULL_ATTENTION_KV_ELEMENTS_V1;
pub const QWEN35_FULL_ATTENTION_MAX_CAPACITY: usize = AUGUR_CUDA_QWEN35_FULL_ATTENTION_MAX_CAPACITY_V1;

#[derive(Clone, Copy, Debug)]
pub struct CudaQwen35FullAttentionViewsV1 {
    pub query: CudaF32View,
    pub gate: CudaF32View,
    pub key: CudaF32View,
    pub value: CudaF32View,
    pub key_cache: CudaF32View,
    pub value_cache: CudaF32View,
    pub attention: CudaF32View,
    pub gated: CudaF32View,
}

#[derive(Clone, Copy, Debug)]
pub struct CudaQwen35FullAttentionArgsV1 {
    raw: AugurCudaQwen35FullAttentionArgsV1,
    ordinal: i32,
}

impl CudaQwen35FullAttentionArgsV1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        views: CudaQwen35FullAttentionViewsV1,
        capacity: usize,
        position: usize,
    ) -> Result<Self, CudaRuntimeError> {
        if capacity == 0 || capacity > QWEN35_FULL_ATTENTION_MAX_CAPACITY || position >= capacity {
            return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        let cache_elements = capacity
            .checked_mul(QWEN35_FULL_ATTENTION_KV_ELEMENTS)
            .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))?;
        let expected = [
            (views.query, QWEN35_FULL_ATTENTION_QUERY_ELEMENTS),
            (views.gate, QWEN35_FULL_ATTENTION_QUERY_ELEMENTS),
            (views.key, QWEN35_FULL_ATTENTION_KV_ELEMENTS),
            (views.value, QWEN35_FULL_ATTENTION_KV_ELEMENTS),
            (views.key_cache, cache_elements),
            (views.value_cache, cache_elements),
            (views.attention, QWEN35_FULL_ATTENTION_QUERY_ELEMENTS),
            (views.gated, QWEN35_FULL_ATTENTION_QUERY_ELEMENTS),
        ];
        let ordinal = stream.device_ordinal();
        for (view, elements) in expected {
            if view.buffer_view().device_ordinal() != ordinal
                || view.element_count() != elements
                || view.buffer_view().checked_address(OPERATION)?
                    % core::mem::align_of::<f32>() as u64
                    != 0
            {
                return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
            }
            view.buffer_view().checked_range(OPERATION)?;
        }
        let buffers = [
            views.query.buffer_view(),
            views.gate.buffer_view(),
            views.key.buffer_view(),
            views.value.buffer_view(),
            views.key_cache.buffer_view(),
            views.value_cache.buffer_view(),
            views.attention.buffer_view(),
            views.gated.buffer_view(),
        ];
        for (index, left) in buffers.iter().enumerate() {
            for right in buffers.iter().skip(index + 1) {
                if left.overlaps(*right, OPERATION)? {
                    return Err(validation(CudaRuntimeErrorKind::OverlappingDeviceRanges));
                }
            }
        }

        let mut raw = AugurCudaQwen35FullAttentionArgsV1::new();
        macro_rules! bind_view {
            ($ptr:ident, $len:ident, $view:expr) => {{
                let view = $view.buffer_view();
                raw.$ptr = view.checked_address(OPERATION)?;
                raw.$len = u64::try_from(view.len_bytes())
                    .map_err(|_| validation(CudaRuntimeErrorKind::RangeOverflow))?;
            }};
        }
        bind_view!(query_f32, query_len_bytes, views.query);
        bind_view!(gate_f32, gate_len_bytes, views.gate);
        bind_view!(key_f32, key_len_bytes, views.key);
        bind_view!(value_f32, value_len_bytes, views.value);
        bind_view!(key_cache_f32, key_cache_len_bytes, views.key_cache);
        bind_view!(value_cache_f32, value_cache_len_bytes, views.value_cache);
        bind_view!(attention_f32, attention_len_bytes, views.attention);
        bind_view!(gated_f32, gated_len_bytes, views.gated);
        raw.capacity = u32::try_from(capacity)
            .map_err(|_| validation(CudaRuntimeErrorKind::InvalidMetadata))?;
        raw.position = u32::try_from(position)
            .map_err(|_| validation(CudaRuntimeErrorKind::InvalidMetadata))?;
        raw.query_heads = QWEN35_FULL_ATTENTION_QUERY_HEADS as u32;
        raw.kv_heads = QWEN35_FULL_ATTENTION_KV_HEADS as u32;
        raw.head_dim = QWEN35_FULL_ATTENTION_HEAD_DIM as u32;
        raw.rotary_dim = QWEN35_FULL_ATTENTION_ROTARY_DIM as u32;
        Ok(Self { raw, ordinal })
    }

    pub const fn as_raw(&self) -> &AugurCudaQwen35FullAttentionArgsV1 {
        &self.raw
    }

    pub fn submit_for_plan(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        // SAFETY: bind checked every device view, exact extent, alignment, and
        // disjointness. Owners must remain live until a completion proof.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_full_attention_submit_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result("cuda_qwen35_full_attention_submit_v1", result)
    }

    pub fn launch_sync(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        // SAFETY: same bind-established contract as submit_for_plan; the
        // native wrapper drains the stream before returning.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_full_attention_v1(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result(OPERATION, result)
    }

    fn validate_stream(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.ordinal {
            return Err(validation(CudaRuntimeErrorKind::DeviceMismatch));
        }
        Ok(())
    }
}

#[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
pub const QWEN35_FULL_ATTENTION_SPLITS_V2: usize = AUGUR_CUDA_QWEN35_FULL_ATTENTION_SPLITS_V2;
#[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
pub const QWEN35_FULL_ATTENTION_STAGE1_THREADS_V2: usize =
    AUGUR_CUDA_QWEN35_FULL_ATTENTION_STAGE1_THREADS_V2;
#[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
pub const QWEN35_FULL_ATTENTION_TILE_POSITIONS_V2: usize =
    AUGUR_CUDA_QWEN35_FULL_ATTENTION_TILE_POSITIONS_V2;
#[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
pub const QWEN35_FULL_ATTENTION_MERGE_SEGMENTS_V2: usize =
    AUGUR_CUDA_QWEN35_FULL_ATTENTION_MERGE_SEGMENTS_V2;
#[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
pub const QWEN35_FULL_ATTENTION_MERGE_THREADS_V2: usize =
    AUGUR_CUDA_QWEN35_FULL_ATTENTION_MERGE_THREADS_V2;
#[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
pub const QWEN35_FULL_ATTENTION_PARTIAL_STRIDE_V2: usize =
    AUGUR_CUDA_QWEN35_FULL_ATTENTION_PARTIAL_STRIDE_V2;
#[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
pub const QWEN35_FULL_ATTENTION_PARTIAL_ELEMENTS_V2: usize =
    AUGUR_CUDA_QWEN35_FULL_ATTENTION_PARTIAL_ELEMENTS_V2;
#[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
pub const QWEN35_FULL_ATTENTION_FUSED_Q8_BYTES_V2: usize =
    AUGUR_CUDA_QWEN35_FULL_ATTENTION_FUSED_Q8_BYTES_V2;
#[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
pub const QWEN35_FULL_ATTENTION_WORKSPACE_ELEMENTS_V2: usize =
    AUGUR_CUDA_QWEN35_FULL_ATTENTION_WORKSPACE_ELEMENTS_V2;

#[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
#[derive(Clone, Copy, Debug)]
pub struct CudaQwen35FullAttentionViewsV2 {
    pub query: CudaF32View,
    pub gate: CudaF32View,
    pub key: CudaF32View,
    pub value: CudaF32View,
    pub key_cache: CudaF32View,
    pub value_cache: CudaF32View,
    pub attention: CudaF32View,
    pub gated: CudaF32View,
    pub workspace: CudaF32View,
}

#[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
#[derive(Clone, Copy, Debug)]
pub struct CudaQwen35FullAttentionArgsV2 {
    raw: AugurCudaQwen35FullAttentionArgsV2,
    ordinal: i32,
}

#[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
impl CudaQwen35FullAttentionArgsV2 {
    pub fn bind(
        stream: &OwnedCudaStream,
        views: CudaQwen35FullAttentionViewsV2,
        capacity: usize,
        position: usize,
    ) -> Result<Self, CudaRuntimeError> {
        if capacity == 0 || capacity > QWEN35_FULL_ATTENTION_MAX_CAPACITY || position >= capacity {
            return Err(validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        let cache_elements = capacity
            .checked_mul(QWEN35_FULL_ATTENTION_KV_ELEMENTS)
            .ok_or_else(|| validation(CudaRuntimeErrorKind::RangeOverflow))?;
        let expected = [
            (views.query, QWEN35_FULL_ATTENTION_QUERY_ELEMENTS),
            (views.gate, QWEN35_FULL_ATTENTION_QUERY_ELEMENTS),
            (views.key, QWEN35_FULL_ATTENTION_KV_ELEMENTS),
            (views.value, QWEN35_FULL_ATTENTION_KV_ELEMENTS),
            (views.key_cache, cache_elements),
            (views.value_cache, cache_elements),
            (views.attention, QWEN35_FULL_ATTENTION_QUERY_ELEMENTS),
            (views.gated, QWEN35_FULL_ATTENTION_QUERY_ELEMENTS),
            (views.workspace, QWEN35_FULL_ATTENTION_WORKSPACE_ELEMENTS_V2),
        ];
        let ordinal = stream.device_ordinal();
        for (view, elements) in expected {
            if view.buffer_view().device_ordinal() != ordinal
                || view.element_count() != elements
                || view.buffer_view().checked_address(OPERATION)?
                    % core::mem::align_of::<f32>() as u64
                    != 0
            {
                return Err(validation(CudaRuntimeErrorKind::InvalidDimensions));
            }
            view.buffer_view().checked_range(OPERATION)?;
        }
        let buffers = [
            views.query.buffer_view(),
            views.gate.buffer_view(),
            views.key.buffer_view(),
            views.value.buffer_view(),
            views.key_cache.buffer_view(),
            views.value_cache.buffer_view(),
            views.attention.buffer_view(),
            views.gated.buffer_view(),
            views.workspace.buffer_view(),
        ];
        for (index, left) in buffers.iter().enumerate() {
            for right in buffers.iter().skip(index + 1) {
                if left.overlaps(*right, OPERATION)? {
                    return Err(validation(CudaRuntimeErrorKind::OverlappingDeviceRanges));
                }
            }
        }

        let mut raw = AugurCudaQwen35FullAttentionArgsV2::new();
        macro_rules! bind_view {
            ($ptr:ident, $len:ident, $view:expr) => {{
                let view = $view.buffer_view();
                raw.$ptr = view.checked_address(OPERATION)?;
                raw.$len = u64::try_from(view.len_bytes())
                    .map_err(|_| validation(CudaRuntimeErrorKind::RangeOverflow))?;
            }};
        }
        bind_view!(query_f32, query_len_bytes, views.query);
        bind_view!(gate_f32, gate_len_bytes, views.gate);
        bind_view!(key_f32, key_len_bytes, views.key);
        bind_view!(value_f32, value_len_bytes, views.value);
        bind_view!(key_cache_f32, key_cache_len_bytes, views.key_cache);
        bind_view!(value_cache_f32, value_cache_len_bytes, views.value_cache);
        bind_view!(attention_f32, attention_len_bytes, views.attention);
        bind_view!(gated_f32, gated_len_bytes, views.gated);
        bind_view!(workspace_f32, workspace_len_bytes, views.workspace);
        raw.capacity = u32::try_from(capacity)
            .map_err(|_| validation(CudaRuntimeErrorKind::InvalidMetadata))?;
        raw.position = u32::try_from(position)
            .map_err(|_| validation(CudaRuntimeErrorKind::InvalidMetadata))?;
        raw.query_heads = QWEN35_FULL_ATTENTION_QUERY_HEADS as u32;
        raw.kv_heads = QWEN35_FULL_ATTENTION_KV_HEADS as u32;
        raw.head_dim = QWEN35_FULL_ATTENTION_HEAD_DIM as u32;
        raw.rotary_dim = QWEN35_FULL_ATTENTION_ROTARY_DIM as u32;
        raw.split_count = QWEN35_FULL_ATTENTION_SPLITS_V2 as u32;
        Ok(Self { raw, ordinal })
    }

    pub const fn as_raw(&self) -> &AugurCudaQwen35FullAttentionArgsV2 {
        &self.raw
    }

    pub fn submit_for_plan(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        // SAFETY: bind checked exact extents, alignment, disjointness, and
        // device ownership. All owners outlive the terminal stream proof.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_full_attention_submit_v2(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result("cuda_qwen35_full_attention_submit_v2", result)
    }

    pub fn launch_sync(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        // SAFETY: same bind-established contract as submit_for_plan; the
        // native wrapper drains the stream before returning.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_full_attention_v2(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result("cuda_qwen35_full_attention_v2", result)
    }

    fn validate_stream(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.ordinal {
            return Err(validation(CudaRuntimeErrorKind::DeviceMismatch));
        }
        Ok(())
    }
}

#[cfg(feature = "qwen35-kv8-full-attention")]
pub const QWEN35_KV8_FULL_ATTENTION_MAX_CAPACITY: usize =
    AUGUR_CUDA_QWEN35_KV8_FULL_ATTENTION_MAX_CAPACITY_V3;
#[cfg(feature = "qwen35-kv8-full-attention")]
pub const QWEN35_KV8_BLOCK_ELEMENTS: usize = AUGUR_CUDA_QWEN35_KV8_BLOCK_ELEMENTS_V3;
#[cfg(feature = "qwen35-kv8-full-attention")]
pub const QWEN35_KV8_BLOCK_BYTES: usize = AUGUR_CUDA_QWEN35_KV8_BLOCK_BYTES_V3;
#[cfg(feature = "qwen35-kv8-full-attention")]
pub const QWEN35_KV8_BLOCKS_PER_HEAD: usize = AUGUR_CUDA_QWEN35_KV8_BLOCKS_PER_HEAD_V3;
#[cfg(feature = "qwen35-kv8-full-attention")]
pub const QWEN35_KV8_HEAD_ROW_BYTES: usize = AUGUR_CUDA_QWEN35_KV8_HEAD_ROW_BYTES_V3;
#[cfg(feature = "qwen35-kv8-full-attention")]
pub const QWEN35_KV8_CACHE_ROW_BYTES: usize = AUGUR_CUDA_QWEN35_KV8_CACHE_ROW_BYTES_V3;

#[cfg(feature = "qwen35-kv8-full-attention")]
#[derive(Clone, Copy, Debug)]
pub struct CudaQwen35Kv8FullAttentionViewsV3 {
    pub query: CudaF32View,
    pub gate: CudaF32View,
    pub key: CudaF32View,
    pub value: CudaF32View,
    pub key_cache: CudaBufferView,
    pub value_cache: CudaBufferView,
    pub attention: CudaF32View,
    pub gated: CudaF32View,
    pub workspace: CudaF32View,
}

#[cfg(feature = "qwen35-kv8-full-attention")]
#[derive(Clone, Copy, Debug)]
pub struct CudaQwen35Kv8FullAttentionArgsV3 {
    raw: AugurCudaQwen35Kv8FullAttentionArgsV3,
    ordinal: i32,
}

#[cfg(feature = "qwen35-kv8-full-attention")]
impl CudaQwen35Kv8FullAttentionArgsV3 {
    /// Reuse the already checked cache/workspace extents at another causal
    /// position. This does not bind a new pointer, allocation or stream.
    pub fn at_position(&self, position: usize) -> Result<Self, CudaRuntimeError> {
        if self.raw.capacity == 0
            || self.raw.capacity as usize > QWEN35_KV8_FULL_ATTENTION_MAX_CAPACITY
            || position >= self.raw.capacity as usize
        {
            return Err(kv8_validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        let mut next = *self;
        next.raw.position = u32::try_from(position)
            .map_err(|_| kv8_validation(CudaRuntimeErrorKind::RangeOverflow))?;
        Ok(next)
    }
    pub fn bind(
        stream: &OwnedCudaStream,
        views: CudaQwen35Kv8FullAttentionViewsV3,
        capacity: usize,
        position: usize,
    ) -> Result<Self, CudaRuntimeError> {
        if capacity == 0
            || capacity > QWEN35_KV8_FULL_ATTENTION_MAX_CAPACITY
            || position >= capacity
        {
            return Err(kv8_validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        let cache_bytes = capacity
            .checked_mul(QWEN35_KV8_CACHE_ROW_BYTES)
            .ok_or_else(|| kv8_validation(CudaRuntimeErrorKind::RangeOverflow))?;
        let ordinal = stream.device_ordinal();
        let f32_views = [
            (views.query, QWEN35_FULL_ATTENTION_QUERY_ELEMENTS),
            (views.gate, QWEN35_FULL_ATTENTION_QUERY_ELEMENTS),
            (views.key, QWEN35_FULL_ATTENTION_KV_ELEMENTS),
            (views.value, QWEN35_FULL_ATTENTION_KV_ELEMENTS),
            (views.attention, QWEN35_FULL_ATTENTION_QUERY_ELEMENTS),
            (views.gated, QWEN35_FULL_ATTENTION_QUERY_ELEMENTS),
            (views.workspace, QWEN35_FULL_ATTENTION_WORKSPACE_ELEMENTS_V2),
        ];
        for (view, elements) in f32_views {
            if view.buffer_view().device_ordinal() != ordinal
                || view.element_count() != elements
                || view.buffer_view().checked_address(KV8_OPERATION)?
                    % core::mem::align_of::<f32>() as u64
                    != 0
            {
                return Err(kv8_validation(CudaRuntimeErrorKind::InvalidDimensions));
            }
            view.buffer_view().checked_range(KV8_OPERATION)?;
        }
        for cache in [views.key_cache, views.value_cache] {
            if cache.device_ordinal() != ordinal
                || cache.len_bytes() != cache_bytes
                || cache.checked_address(KV8_OPERATION)? % core::mem::align_of::<u16>() as u64 != 0
            {
                return Err(kv8_validation(CudaRuntimeErrorKind::InvalidDimensions));
            }
            cache.checked_range(KV8_OPERATION)?;
        }
        let buffers = [
            views.query.buffer_view(),
            views.gate.buffer_view(),
            views.key.buffer_view(),
            views.value.buffer_view(),
            views.key_cache,
            views.value_cache,
            views.attention.buffer_view(),
            views.gated.buffer_view(),
            views.workspace.buffer_view(),
        ];
        for (index, left) in buffers.iter().enumerate() {
            for right in buffers.iter().skip(index + 1) {
                if left.overlaps(*right, KV8_OPERATION)? {
                    return Err(kv8_validation(
                        CudaRuntimeErrorKind::OverlappingDeviceRanges,
                    ));
                }
            }
        }

        let mut raw = AugurCudaQwen35Kv8FullAttentionArgsV3::new();
        macro_rules! bind_f32 {
            ($ptr:ident, $len:ident, $view:expr) => {{
                let view = $view.buffer_view();
                raw.$ptr = view.checked_address(KV8_OPERATION)?;
                raw.$len = u64::try_from(view.len_bytes())
                    .map_err(|_| kv8_validation(CudaRuntimeErrorKind::RangeOverflow))?;
            }};
        }
        macro_rules! bind_bytes {
            ($ptr:ident, $len:ident, $view:expr) => {{
                let view = $view;
                raw.$ptr = view.checked_address(KV8_OPERATION)?;
                raw.$len = u64::try_from(view.len_bytes())
                    .map_err(|_| kv8_validation(CudaRuntimeErrorKind::RangeOverflow))?;
            }};
        }
        bind_f32!(query_f32, query_len_bytes, views.query);
        bind_f32!(gate_f32, gate_len_bytes, views.gate);
        bind_f32!(key_f32, key_len_bytes, views.key);
        bind_f32!(value_f32, value_len_bytes, views.value);
        bind_bytes!(key_cache_q8_0, key_cache_len_bytes, views.key_cache);
        bind_bytes!(value_cache_q8_0, value_cache_len_bytes, views.value_cache);
        bind_f32!(attention_f32, attention_len_bytes, views.attention);
        bind_f32!(gated_f32, gated_len_bytes, views.gated);
        bind_f32!(workspace_f32, workspace_len_bytes, views.workspace);
        raw.capacity = u32::try_from(capacity)
            .map_err(|_| kv8_validation(CudaRuntimeErrorKind::InvalidMetadata))?;
        raw.position = u32::try_from(position)
            .map_err(|_| kv8_validation(CudaRuntimeErrorKind::InvalidMetadata))?;
        raw.query_heads = QWEN35_FULL_ATTENTION_QUERY_HEADS as u32;
        raw.kv_heads = QWEN35_FULL_ATTENTION_KV_HEADS as u32;
        raw.head_dim = QWEN35_FULL_ATTENTION_HEAD_DIM as u32;
        raw.rotary_dim = QWEN35_FULL_ATTENTION_ROTARY_DIM as u32;
        raw.split_count = QWEN35_FULL_ATTENTION_SPLITS_V2 as u32;
        Ok(Self { raw, ordinal })
    }

    pub const fn as_raw(&self) -> &AugurCudaQwen35Kv8FullAttentionArgsV3 {
        &self.raw
    }

    pub fn submit_for_plan(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        // SAFETY: bind fixed the Q8_0 cache geometry, all F32 extents,
        // alignment, non-aliasing, and device ownership. The caller retains
        // every owner through a terminal stream proof.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_kv8_full_attention_submit_v3(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result("cuda_qwen35_kv8_full_attention_submit_v3", result)
    }

    pub fn launch_sync(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        // SAFETY: same checked binding as submit_for_plan; the native wrapper
        // drains this stream before it returns.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_kv8_full_attention_v3(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result(KV8_OPERATION, result)
    }

    fn validate_stream(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.ordinal {
            return Err(kv8_validation(CudaRuntimeErrorKind::DeviceMismatch));
        }
        Ok(())
    }
}

/// Contiguous multi-row candidate binding.  The V4 native ABI preserves the
/// V3 arithmetic and split intervals while sharing historical KV tiles across
/// up to eight causal rows. The runtime selector remains explicit and opt-in.
#[cfg(feature = "qwen35-kv8-batched-attention")]
#[derive(Clone, Copy, Debug)]
pub struct CudaQwen35Kv8BatchedAttentionViewsV4 {
    pub query: CudaF32View,
    pub gate: CudaF32View,
    pub key: CudaF32View,
    pub value: CudaF32View,
    pub key_cache: CudaBufferView,
    pub value_cache: CudaBufferView,
    pub attention: CudaF32View,
    pub gated: CudaF32View,
    pub workspace: CudaF32View,
}

#[cfg(feature = "qwen35-kv8-batched-attention")]
#[derive(Clone, Copy, Debug)]
pub struct CudaQwen35Kv8BatchedAttentionArgsV4 {
    raw: AugurCudaQwen35Kv8BatchedAttentionArgsV4,
    ordinal: i32,
}

#[cfg(feature = "qwen35-kv8-batched-attention")]
impl CudaQwen35Kv8BatchedAttentionArgsV4 {
    pub fn bind(
        stream: &OwnedCudaStream,
        views: CudaQwen35Kv8BatchedAttentionViewsV4,
        capacity: usize,
        position_base: usize,
        tokens: usize,
    ) -> Result<Self, CudaRuntimeError> {
        Self::bind_rows(
            stream,
            views,
            capacity,
            position_base,
            tokens,
            AUGUR_CUDA_QWEN35_KV8_BATCHED_ATTENTION_MAX_TOKENS_V4,
            augur_cuda_sys::AUGUR_CUDA_QWEN35_KV8_BATCHED_ATTENTION_ARGS_VERSION_V4,
        )
    }

    // Shared address validation, not a widening of the public V4 operation.
    pub(crate) fn bind_rows(
        stream: &OwnedCudaStream,
        views: CudaQwen35Kv8BatchedAttentionViewsV4,
        capacity: usize,
        position_base: usize,
        tokens: usize,
        max_tokens: usize,
        abi_version: u32,
    ) -> Result<Self, CudaRuntimeError> {
        if capacity == 0
            || capacity > QWEN35_KV8_FULL_ATTENTION_MAX_CAPACITY
            || tokens == 0
            || tokens > max_tokens
            || position_base
                .checked_add(tokens)
                .is_none_or(|end| end > capacity)
        {
            return Err(kv8_validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        let ordinal = stream.device_ordinal();
        let cache_bytes = capacity
            .checked_mul(QWEN35_KV8_CACHE_ROW_BYTES)
            .ok_or_else(|| kv8_validation(CudaRuntimeErrorKind::RangeOverflow))?;
        let query_elements = tokens
            .checked_mul(QWEN35_FULL_ATTENTION_QUERY_ELEMENTS)
            .ok_or_else(|| kv8_validation(CudaRuntimeErrorKind::RangeOverflow))?;
        let kv_elements = tokens
            .checked_mul(QWEN35_FULL_ATTENTION_KV_ELEMENTS)
            .ok_or_else(|| kv8_validation(CudaRuntimeErrorKind::RangeOverflow))?;
        let workspace_elements = tokens
            .checked_mul(QWEN35_FULL_ATTENTION_WORKSPACE_ELEMENTS_V2)
            .ok_or_else(|| kv8_validation(CudaRuntimeErrorKind::RangeOverflow))?;
        let f32_views = [
            (views.query, query_elements),
            (views.gate, query_elements),
            (views.key, kv_elements),
            (views.value, kv_elements),
            (views.attention, query_elements),
            (views.gated, query_elements),
            (views.workspace, workspace_elements),
        ];
        for (view, elements) in f32_views {
            if view.buffer_view().device_ordinal() != ordinal
                || view.element_count() != elements
                || view.buffer_view().checked_address(KV8_BATCHED_OPERATION)?
                    % core::mem::align_of::<f32>() as u64
                    != 0
            {
                return Err(kv8_validation(CudaRuntimeErrorKind::InvalidDimensions));
            }
            view.buffer_view().checked_range(KV8_BATCHED_OPERATION)?;
        }
        for cache in [views.key_cache, views.value_cache] {
            if cache.device_ordinal() != ordinal
                || cache.len_bytes() != cache_bytes
                || cache.checked_address(KV8_BATCHED_OPERATION)?
                    % core::mem::align_of::<u16>() as u64
                    != 0
            {
                return Err(kv8_validation(CudaRuntimeErrorKind::InvalidDimensions));
            }
            cache.checked_range(KV8_BATCHED_OPERATION)?;
        }
        let buffers = [
            views.query.buffer_view(),
            views.gate.buffer_view(),
            views.key.buffer_view(),
            views.value.buffer_view(),
            views.key_cache,
            views.value_cache,
            views.attention.buffer_view(),
            views.gated.buffer_view(),
            views.workspace.buffer_view(),
        ];
        for (index, left) in buffers.iter().enumerate() {
            for right in buffers.iter().skip(index + 1) {
                if left.overlaps(*right, KV8_BATCHED_OPERATION)? {
                    return Err(kv8_validation(
                        CudaRuntimeErrorKind::OverlappingDeviceRanges,
                    ));
                }
            }
        }
        let mut raw = AugurCudaQwen35Kv8BatchedAttentionArgsV4::new();
        raw.abi_version = abi_version;
        macro_rules! bind_f32 {
            ($ptr:ident, $len:ident, $view:expr) => {{
                let view = $view.buffer_view();
                raw.$ptr = view.checked_address(KV8_BATCHED_OPERATION)?;
                raw.$len = u64::try_from(view.len_bytes())
                    .map_err(|_| kv8_validation(CudaRuntimeErrorKind::RangeOverflow))?;
            }};
        }
        macro_rules! bind_bytes {
            ($ptr:ident, $len:ident, $view:expr) => {{
                raw.$ptr = $view.checked_address(KV8_BATCHED_OPERATION)?;
                raw.$len = u64::try_from($view.len_bytes())
                    .map_err(|_| kv8_validation(CudaRuntimeErrorKind::RangeOverflow))?;
            }};
        }
        bind_f32!(query_f32, query_len_bytes, views.query);
        bind_f32!(gate_f32, gate_len_bytes, views.gate);
        bind_f32!(key_f32, key_len_bytes, views.key);
        bind_f32!(value_f32, value_len_bytes, views.value);
        bind_bytes!(key_cache_q8_0, key_cache_len_bytes, views.key_cache);
        bind_bytes!(value_cache_q8_0, value_cache_len_bytes, views.value_cache);
        bind_f32!(attention_f32, attention_len_bytes, views.attention);
        bind_f32!(gated_f32, gated_len_bytes, views.gated);
        bind_f32!(workspace_f32, workspace_len_bytes, views.workspace);
        raw.capacity = u32::try_from(capacity)
            .map_err(|_| kv8_validation(CudaRuntimeErrorKind::InvalidMetadata))?;
        raw.position_base = u32::try_from(position_base)
            .map_err(|_| kv8_validation(CudaRuntimeErrorKind::InvalidMetadata))?;
        raw.tokens = u32::try_from(tokens)
            .map_err(|_| kv8_validation(CudaRuntimeErrorKind::InvalidMetadata))?;
        Ok(Self { raw, ordinal })
    }

    pub const fn as_raw(&self) -> &AugurCudaQwen35Kv8BatchedAttentionArgsV4 {
        &self.raw
    }

    pub fn submit_for_plan(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_kv8_batched_attention_submit_v4(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result("cuda_qwen35_kv8_batched_attention_submit_v4", result)
    }

    pub fn launch_sync(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        self.validate_stream(stream)?;
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_kv8_batched_attention_v4(
                stream.device_ordinal(),
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result(KV8_BATCHED_OPERATION, result)
    }

    fn validate_stream(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.ordinal {
            return Err(kv8_validation(CudaRuntimeErrorKind::DeviceMismatch));
        }
        Ok(())
    }
}

fn validation(kind: CudaRuntimeErrorKind) -> CudaRuntimeError {
    CudaRuntimeError::validation(OPERATION, kind)
}

#[cfg(feature = "qwen35-kv8-full-attention")]
fn kv8_validation(kind: CudaRuntimeErrorKind) -> CudaRuntimeError {
    CudaRuntimeError::validation(KV8_OPERATION, kind)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_constants_are_fixed() {
        assert_eq!(QWEN35_FULL_ATTENTION_QUERY_HEADS, 24);
        assert_eq!(QWEN35_FULL_ATTENTION_KV_HEADS, 4);
        assert_eq!(QWEN35_FULL_ATTENTION_HEAD_DIM, 256);
        assert_eq!(QWEN35_FULL_ATTENTION_ROTARY_DIM, 64);
        assert_eq!(QWEN35_FULL_ATTENTION_QUERY_ELEMENTS, 6144);
        assert_eq!(QWEN35_FULL_ATTENTION_KV_ELEMENTS, 1024);
    }

    #[test]
    fn cache_extent_is_capacity_times_kv_row() {
        let capacity = 2;
        assert_eq!(capacity * QWEN35_FULL_ATTENTION_KV_ELEMENTS, 2048);
    }

    #[cfg(feature = "qwen35-grouped-gqa-online-softmax")]
    #[test]
    fn grouped_gqa_workspace_geometry_is_fixed() {
        assert_eq!(QWEN35_FULL_ATTENTION_SPLITS_V2, 72);
        assert_eq!(QWEN35_FULL_ATTENTION_STAGE1_THREADS_V2, 256);
        assert_eq!(QWEN35_FULL_ATTENTION_TILE_POSITIONS_V2, 8);
        assert_eq!(QWEN35_FULL_ATTENTION_MERGE_SEGMENTS_V2, 2);
        assert_eq!(QWEN35_FULL_ATTENTION_MERGE_THREADS_V2, 128);
        assert_eq!(QWEN35_FULL_ATTENTION_PARTIAL_STRIDE_V2, 258);
        assert_eq!(
            QWEN35_FULL_ATTENTION_PARTIAL_ELEMENTS_V2,
            QWEN35_FULL_ATTENTION_QUERY_HEADS
                * QWEN35_FULL_ATTENTION_SPLITS_V2
                * (QWEN35_FULL_ATTENTION_HEAD_DIM + 2)
        );
        assert_eq!(QWEN35_FULL_ATTENTION_FUSED_Q8_BYTES_V2, 6_912);
        assert_eq!(
            QWEN35_FULL_ATTENTION_WORKSPACE_ELEMENTS_V2 * 4,
            QWEN35_FULL_ATTENTION_PARTIAL_ELEMENTS_V2 * 4 + QWEN35_FULL_ATTENTION_FUSED_Q8_BYTES_V2
        );
        assert_eq!(QWEN35_FULL_ATTENTION_WORKSPACE_ELEMENTS_V2 * 4, 1_790_208);
    }

    #[cfg(feature = "qwen35-kv8-full-attention")]
    #[test]
    fn kv8_cache_geometry_matches_q8_0_memory_budget() {
        assert_eq!(QWEN35_KV8_FULL_ATTENTION_MAX_CAPACITY, 262_144);
        assert_eq!(QWEN35_KV8_BLOCK_ELEMENTS, 32);
        assert_eq!(QWEN35_KV8_BLOCK_BYTES, 34);
        assert_eq!(QWEN35_KV8_BLOCKS_PER_HEAD, 8);
        assert_eq!(QWEN35_KV8_HEAD_ROW_BYTES, 272);
        assert_eq!(QWEN35_KV8_CACHE_ROW_BYTES, 1_088);
        assert_eq!(
            QWEN35_KV8_CACHE_ROW_BYTES,
            QWEN35_FULL_ATTENTION_KV_HEADS
                * (QWEN35_FULL_ATTENTION_HEAD_DIM / QWEN35_KV8_BLOCK_ELEMENTS)
                * QWEN35_KV8_BLOCK_BYTES
        );
    }

    #[cfg(feature = "qwen35-kv8-full-attention")]
    #[test]
    fn kv8_position_reuse_keeps_descriptor_and_rejects_out_of_capacity() {
        // Synthetic descriptor only tests host metadata; it cannot be submitted.
        let mut raw = AugurCudaQwen35Kv8FullAttentionArgsV3::new();
        raw.capacity = 196608;
        raw.key_cache_q8_0 = 0x1000;
        raw.value_cache_q8_0 = 0x2000;
        raw.workspace_f32 = 0x3000;
        let bound = CudaQwen35Kv8FullAttentionArgsV3 { raw, ordinal: 0 };
        let last = bound.at_position(196607).unwrap();
        assert_eq!(bound.raw.position, 0);
        assert_eq!(last.raw.position, 196607);
        assert_eq!(last.raw.key_cache_q8_0, bound.raw.key_cache_q8_0);
        assert_eq!(last.raw.value_cache_q8_0, bound.raw.value_cache_q8_0);
        assert_eq!(last.raw.workspace_f32, bound.raw.workspace_f32);
        for position in [196608, usize::MAX] {
            assert!(bound.at_position(position).is_err());
        }
        raw.capacity = 0;
        assert!(CudaQwen35Kv8FullAttentionArgsV3 { raw, ordinal: 0 }
            .at_position(0)
            .is_err());
    }
}
