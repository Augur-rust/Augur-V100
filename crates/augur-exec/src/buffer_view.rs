use augur_backend::{Backend, CpuBackend};
use augur_core::{DType, PhysicalLayoutId, QuantFormatId};

#[derive(Clone, Debug)]
pub struct BufferView<B: Backend> {
    buffer_index: usize,
    offset: usize,
    len: usize,
    alignment: usize,
    dtype: DType,

    /// Quantisation interpretation metadata (M11a).
    ///
    /// M11a only preserves metadata. Execution kernels must not assume that
    /// `quant = Some(Q8K)` means a supported quantized execution path exists.
    quant: Option<QuantFormatId>,

    /// Lowered physical layout.
    physical_layout: PhysicalLayoutId,

    _backend: core::marker::PhantomData<B>,
}

impl<B: Backend> BufferView<B> {
    /// Full constructor with explicit quant / physical_layout.
    ///
    /// This is a low-level protocol constructor. Production lowering should
    /// prefer backend-specific constructors when available.
    pub fn new(
        buffer_index: usize,
        len: usize,
        dtype: DType,
        quant: Option<QuantFormatId>,
        physical_layout: PhysicalLayoutId,
    ) -> Self {
        Self {
            buffer_index,
            offset: 0,
            len,
            alignment: 64,
            dtype,
            quant,
            physical_layout,
            _backend: core::marker::PhantomData,
        }
    }

    /// Plain (non-quantized) buffer with the given dtype.
    pub fn plain(
        buffer_index: usize,
        len: usize,
        dtype: DType,
        physical_layout: PhysicalLayoutId,
    ) -> Self {
        Self::new(buffer_index, len, dtype, None, physical_layout)
    }

    pub fn with_range(mut self, offset: usize, len: usize) -> Option<Self> {
        offset.checked_add(len)?;
        self.offset = offset;
        self.len = len;
        Some(self)
    }

    pub const fn buffer_index(&self) -> usize {
        self.buffer_index
    }

    pub const fn offset(&self) -> usize {
        self.offset
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub const fn alignment(&self) -> usize {
        self.alignment
    }

    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    pub const fn quant(&self) -> Option<QuantFormatId> {
        self.quant
    }

    pub const fn physical_layout(&self) -> PhysicalLayoutId {
        self.physical_layout
    }

    pub fn is_plain_f32(&self) -> bool {
        self.dtype == DType::F32 && self.quant.is_none()
    }
}

impl BufferView<CpuBackend> {
    /// Returns true when this view describes a contiguous host f32 buffer
    /// accepted by CPU kernels.
    ///
    /// This predicate validates representation metadata only. It does not
    /// validate the concrete buffer owner, offset, length or aliasing.
    pub fn is_cpu_contiguous_f32(&self) -> bool {
        self.dtype == DType::F32
            && self.quant.is_none()
            && self.physical_layout == PhysicalLayoutId::HOST_CONTIGUOUS_F32
    }

    /// Returns true when this view describes the CPU portable Q8Hot
    /// activation representation.
    ///
    /// This predicate validates representation metadata only. Complete-owner
    /// shape, zero offset and concrete owner kind are validated during
    /// binding.
    pub fn is_cpu_q8hot(&self) -> bool {
        self.dtype == DType::I8
            && self.quant == Some(QuantFormatId::Q8K)
            && self.physical_layout == PhysicalLayoutId::Q8K_ACTIVATION_HOT_CPU_PORTABLE_V1
    }

    /// Dense contiguous CPU f32 buffer.
    pub fn dense_f32(buffer_index: usize, len: usize) -> Self {
        Self::new(
            buffer_index,
            len,
            DType::F32,
            None,
            PhysicalLayoutId::HOST_CONTIGUOUS_F32,
        )
    }

    /// CPU portable Q8Hot activation view.
    ///
    /// `len_elements` is the logical activation element count, not physical
    /// bytes and not block count.
    ///
    /// # M26-C3 mutable-view restriction
    ///
    /// This constructor represents the complete mutable Q8Hot owner:
    ///
    /// - `offset == 0`;
    /// - `len == owner.len_elements()`;
    /// - `len` must be a non-zero multiple of the Q8K block length.
    ///
    /// Mutable Q8Hot subviews are not supported. Supporting them requires a
    /// separately audited block-range writer and alias contract.
    pub fn q8hot_cpu(buffer_index: usize, len_elements: usize) -> Self {
        Self::new(
            buffer_index,
            len_elements,
            DType::I8,
            Some(QuantFormatId::Q8K),
            PhysicalLayoutId::Q8K_ACTIVATION_HOT_CPU_PORTABLE_V1,
        )
    }
}
