use crate::{DType, LogicalLayout, QuantFormatId, Shape, StorageId};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DeviceKind {
    Cpu,
    Cuda,
    Rocm,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct DeviceId {
    pub kind: DeviceKind,
    pub index: u16,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StorageRef {
    None,
    HostArena {
        id: StorageId,
        offset: usize,
        len: usize,
    },
    HostOwned {
        id: StorageId,
    },
    Device {
        device: DeviceId,
        id: StorageId,
    },
    External {
        id: StorageId,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QuantDesc {
    pub format: QuantFormatId,
    pub block: usize,
    pub version: u16,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorDesc {
    pub dtype: DType,
    pub shape: Shape,
    pub layout: LogicalLayout,
    pub storage: StorageRef,
    pub quant: Option<QuantDesc>,
}

impl StorageRef {
    pub fn id(&self) -> Option<StorageId> {
        match self {
            Self::None => None,
            Self::HostArena { id, .. }
            | Self::HostOwned { id }
            | Self::Device { id, .. }
            | Self::External { id } => Some(*id),
        }
    }

    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }
}

impl TensorDesc {
    pub fn f32_contiguous(shape: Shape) -> Self {
        Self {
            dtype: DType::F32,
            shape,
            layout: LogicalLayout::Contiguous,
            storage: StorageRef::None,
            quant: None,
        }
    }

    pub fn quantized_weight(
        shape: Shape,
        storage: StorageRef,
        format: QuantFormatId,
        block: usize,
        version: u16,
    ) -> Self {
        Self {
            dtype: DType::PackedU8,
            shape,
            layout: LogicalLayout::Contiguous,
            storage,
            quant: Some(QuantDesc {
                format,
                block,
                version,
            }),
        }
    }

    pub fn is_quantized_weight(&self) -> bool {
        self.dtype == DType::PackedU8 && self.quant.is_some() && !self.storage.is_none()
    }

    pub fn elem_count(&self) -> Option<usize> {
        self.shape.elem_count()
    }

    pub fn is_contiguous(&self) -> bool {
        matches!(self.layout, LogicalLayout::Contiguous)
    }

    /// True when this tensor carries no quantisation metadata.
    pub fn is_plain_f32(&self) -> bool {
        self.dtype == DType::F32 && self.quant.is_none()
    }

    /// True when this tensor carries any quantisation format.
    pub fn is_quantized(&self) -> bool {
        self.quant.is_some()
    }

    /// True when the quantisation format matches `format`.
    pub fn is_quant_format(&self, format: QuantFormatId) -> bool {
        self.quant.as_ref().map(|q| q.format) == Some(format)
    }
}
