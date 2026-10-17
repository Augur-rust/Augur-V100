use augur_core::TensorId;

/// Purpose of one executable-plan buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BufferPurpose {
    Input,
    Intermediate,

    /// Compiler-generated physical HotActivation.
    ///
    /// This is not a logical Graph tensor.
    HotActivation,

    Output,
    Constant,
}

/// Step coordinate used by the liveness end marker.
///
/// Graph tensor liveness is calculated in Graph topological-step space.
/// Compiler-generated physical temporaries are created after Graph analysis
/// and therefore use compiled-step space.
///
/// The two coordinate spaces must not be compared or merged as raw `usize`
/// values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BufferLifetimeEnd {
    GraphTopoStep(usize),
    CompiledStep(usize),
}

/// Plan-local identity of a compiler-generated physical temporary.
///
/// This identity does not enter Graph IR and is not a kernel identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct TemporaryBufferId(u32);

impl TemporaryBufferId {
    pub const fn new(index: u32) -> Self {
        Self(index)
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Logical identity represented by a physical buffer slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum BufferIdentity {
    Tensor(TensorId),
    Temporary(TemporaryBufferId),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BufferSlot {
    pub buffer_index: usize,
    pub identity: BufferIdentity,
    pub len_elements: usize,
    pub purpose: BufferPurpose,

    /// Inclusive liveness end in the coordinate space identified by the enum.
    pub reusable_after: Option<BufferLifetimeEnd>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorAllocation {
    pub tensor: TensorId,
    pub buffer_index: usize,
}

/// Allocation record for one compiler-generated physical temporary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TemporaryAllocation {
    pub temporary: TemporaryBufferId,
    pub buffer_index: usize,

    /// Compiled step that produces the temporary.
    pub producer_step: usize,
}

/// Auditable reassignment of one physical buffer between non-overlapping
/// logical identities.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BufferReuse {
    pub buffer_index: usize,
    pub previous: BufferIdentity,
    pub next: BufferIdentity,
    pub previous_reusable_after: BufferLifetimeEnd,
    pub next_first_use: BufferLifetimeEnd,
}

/// Immutable result of compiler-side memory planning.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MemoryPlan {
    buffers: Vec<BufferSlot>,

    /// Indexed only by Graph `TensorId`.
    tensor_allocations: Vec<Option<TensorAllocation>>,

    /// Indexed only by plan-local `TemporaryBufferId`.
    temporary_allocations: Vec<TemporaryAllocation>,

    reuse_transitions: Vec<BufferReuse>,
}

impl MemoryPlan {
    pub fn from_parts(
        buffers: Vec<BufferSlot>,
        tensor_allocations: Vec<Option<TensorAllocation>>,
        temporary_allocations: Vec<TemporaryAllocation>,
    ) -> Self {
        Self {
            buffers,
            tensor_allocations,
            temporary_allocations,
            reuse_transitions: Vec::new(),
        }
    }

    pub fn from_parts_with_reuse(
        buffers: Vec<BufferSlot>,
        tensor_allocations: Vec<Option<TensorAllocation>>,
        temporary_allocations: Vec<TemporaryAllocation>,
        reuse_transitions: Vec<BufferReuse>,
    ) -> Self {
        Self {
            buffers,
            tensor_allocations,
            temporary_allocations,
            reuse_transitions,
        }
    }

    pub fn buffers(&self) -> &[BufferSlot] {
        &self.buffers
    }

    pub fn tensor_allocations(&self) -> &[Option<TensorAllocation>] {
        &self.tensor_allocations
    }

    pub fn temporary_allocations(&self) -> &[TemporaryAllocation] {
        &self.temporary_allocations
    }

    pub fn reuse_transitions(&self) -> &[BufferReuse] {
        &self.reuse_transitions
    }

    pub fn buffer_for(&self, tensor: TensorId) -> Option<usize> {
        self.tensor_allocations
            .get(tensor.index as usize)?
            .as_ref()
            .filter(|allocation| allocation.tensor == tensor)
            .map(|allocation| allocation.buffer_index)
    }

    pub fn temporary_buffer(&self, temporary: TemporaryBufferId) -> Option<usize> {
        let allocation = self.temporary_allocations.get(temporary.index())?;

        if allocation.temporary != temporary {
            return None;
        }

        Some(allocation.buffer_index)
    }
}
