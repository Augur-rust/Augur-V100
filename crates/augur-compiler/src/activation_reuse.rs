use augur_backend::CpuBackend;
use augur_core::{DType, PhysicalLayoutId, QuantFormatId, TensorId};
use augur_exec::{BufferView, TemporaryBufferId};

/// Exact semantic identity of an activation preparation protocol.
///
/// The variant name includes backend, architecture, output protocol and
/// version. A non-BitExact preparation implementation must use a different
/// semantic kind even if it produces the same Rust owner type.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) enum CpuActivationPreparationKind {
    /// CPU portable:
    ///
    /// ```text
    /// HOST_CONTIGUOUS_F32
    /// → Q8K production quantization protocol
    /// → Q8K_ACTIVATION_HOT_CPU_PORTABLE_V1
    /// ```
    Q8KHotCpuPortableV1,
}

/// Exact compiler-private reuse identity for one prepared activation.
///
/// This type never enters Graph IR, GraphAnalysis, ExecutionPlan or a kernel.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) struct CpuActivationPreparationKey {
    /// Logical value identity.
    pub source_tensor: TensorId,

    /// Resolved physical source identity.
    pub source_buffer_index: usize,
    pub source_offset: usize,
    pub source_len: usize,

    /// Exact source representation contract.
    pub source_dtype: DType,
    pub source_quant: Option<QuantFormatId>,
    pub source_layout: PhysicalLayoutId,

    /// Encodes backend, architecture, output layout and numerical protocol.
    pub kind: CpuActivationPreparationKind,
}

impl CpuActivationPreparationKey {
    pub(crate) fn q8k_hot_cpu_portable_v1(
        source_tensor: TensorId,
        source: &BufferView<CpuBackend>,
    ) -> Option<Self> {
        if !source.is_cpu_contiguous_f32() {
            return None;
        }

        Some(Self {
            source_tensor,
            source_buffer_index: source.buffer_index(),
            source_offset: source.offset(),
            source_len: source.len(),
            source_dtype: source.dtype(),
            source_quant: source.quant(),
            source_layout: source.physical_layout(),
            kind: CpuActivationPreparationKind::Q8KHotCpuPortableV1,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PreparedQ8HotActivation {
    pub key: CpuActivationPreparationKey,
    pub temporary: TemporaryBufferId,
    pub buffer_index: usize,
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn same_tensor_different_source_slice_is_not_same_preparation() {
        let tensor = TensorId::new(3, 0);

        let first = CpuActivationPreparationKey::q8k_hot_cpu_portable_v1(
            tensor,
            &BufferView::<CpuBackend>::dense_f32(7, 256),
        )
        .unwrap();

        let mut second_view = BufferView::<CpuBackend>::dense_f32(7, 256);
        second_view = second_view.with_range(256, 256).unwrap();

        let second =
            CpuActivationPreparationKey::q8k_hot_cpu_portable_v1(tensor, &second_view).unwrap();

        assert_ne!(first, second);
    }

    #[test]
    fn different_tensor_same_buffer_is_not_same_preparation() {
        let view = BufferView::<CpuBackend>::dense_f32(4, 256);

        let first =
            CpuActivationPreparationKey::q8k_hot_cpu_portable_v1(TensorId::new(1, 0), &view)
                .unwrap();

        let second =
            CpuActivationPreparationKey::q8k_hot_cpu_portable_v1(TensorId::new(2, 0), &view)
                .unwrap();

        assert_ne!(first, second);
    }

    #[test]
    fn noncanonical_source_metadata_is_rejected() {
        let source = BufferView::<CpuBackend>::q8hot_cpu(1, 256);

        assert!(
            CpuActivationPreparationKey::q8k_hot_cpu_portable_v1(TensorId::new(0, 0), &source)
                .is_none()
        );
    }

    #[test]
    fn exact_preparation_key_is_hashable() {
        let source = BufferView::<CpuBackend>::dense_f32(4, 256);

        let key =
            CpuActivationPreparationKey::q8k_hot_cpu_portable_v1(TensorId::new(1, 0), &source)
                .unwrap();

        let mut keys = HashSet::new();

        assert!(keys.insert(key));
        assert!(!keys.insert(key));
    }
}
