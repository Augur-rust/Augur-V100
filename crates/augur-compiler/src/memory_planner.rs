use augur_core::{CompileError, TensorId};
use augur_exec::{
    BufferIdentity, BufferLifetimeEnd, BufferPurpose, BufferReuse, BufferSlot, MemoryPlan,
    TemporaryAllocation, TemporaryBufferId, TensorAllocation,
};

/// Mutable compiler-side builder for an immutable [`MemoryPlan`].
#[derive(Clone, Debug)]
pub struct MemoryPlanner {
    buffers: Vec<BufferSlot>,

    tensor_allocations: Vec<Option<TensorAllocation>>,

    temporary_allocations: Vec<TemporaryAllocation>,

    current_identities: Vec<BufferIdentity>,

    reuse_transitions: Vec<BufferReuse>,
}

impl MemoryPlanner {
    pub fn new(tensor_count: usize) -> Self {
        Self {
            buffers: Vec::new(),
            tensor_allocations: vec![None; tensor_count],
            temporary_allocations: Vec::new(),
            current_identities: Vec::new(),
            reuse_transitions: Vec::new(),
        }
    }

    /// Returns the physical buffer index that the next successful assignment
    /// will receive.
    ///
    /// Lowering uses this before mutating either MemoryPlanner or the parallel
    /// `Vec<CpuBuffer>`, so a synchronization error is reported before either
    /// structure is changed.
    pub fn next_buffer_index(&self) -> usize {
        self.buffers.len()
    }

    /// Assigns a logical Graph tensor.
    ///
    /// `reusable_after_topo_step` is an inclusive Graph topological-step
    /// index.
    pub fn assign(
        &mut self,
        tensor: TensorId,
        len_elements: usize,
        purpose: BufferPurpose,
        reusable_after_topo_step: Option<usize>,
    ) -> Result<usize, CompileError> {
        if len_elements == 0 {
            return Err(CompileError::InvalidInput);
        }

        let tensor_index = tensor.index as usize;

        let allocation_slot = self
            .tensor_allocations
            .get_mut(tensor_index)
            .ok_or(CompileError::InvalidInput)?;

        if allocation_slot.is_some() {
            return Err(CompileError::InvalidInput);
        }

        let buffer_index = self.buffers.len();

        self.buffers.push(BufferSlot {
            buffer_index,
            identity: BufferIdentity::Tensor(tensor),
            len_elements,
            purpose,
            reusable_after: reusable_after_topo_step.map(BufferLifetimeEnd::GraphTopoStep),
        });
        self.current_identities.push(BufferIdentity::Tensor(tensor));

        *allocation_slot = Some(TensorAllocation {
            tensor,
            buffer_index,
        });

        Ok(buffer_index)
    }

    /// Assigns an intermediate Graph tensor, reusing a sufficient physical
    /// slot only when the previous inclusive lifetime ends before this tensor
    /// first becomes live.
    pub fn assign_reusing_tensor(
        &mut self,
        tensor: TensorId,
        len_elements: usize,
        purpose: BufferPurpose,
        first_topo_step: usize,
        last_topo_step: usize,
    ) -> Result<usize, CompileError> {
        if len_elements == 0
            || purpose != BufferPurpose::Intermediate
            || first_topo_step > last_topo_step
        {
            return Err(CompileError::InvalidInput);
        }
        let tensor_index = tensor.index as usize;
        if self
            .tensor_allocations
            .get(tensor_index)
            .ok_or(CompileError::InvalidInput)?
            .is_some()
        {
            return Err(CompileError::InvalidInput);
        }

        let candidate = self
            .buffers
            .iter()
            .enumerate()
            .filter_map(|(index, slot)| {
                let previous_end = match slot.reusable_after {
                    Some(BufferLifetimeEnd::GraphTopoStep(step)) => step,
                    Some(BufferLifetimeEnd::CompiledStep(_)) | None => return None,
                };
                (slot.purpose == BufferPurpose::Intermediate
                    && slot.len_elements >= len_elements
                    && previous_end < first_topo_step)
                    .then(|| (slot.len_elements - len_elements, index, previous_end))
            })
            .min_by_key(|&(waste, index, _)| (waste, index));

        let Some((_, buffer_index, previous_end)) = candidate else {
            return self.assign(tensor, len_elements, purpose, Some(last_topo_step));
        };

        let previous = *self
            .current_identities
            .get(buffer_index)
            .ok_or(CompileError::InvalidInput)?;
        let next = BufferIdentity::Tensor(tensor);
        self.reuse_transitions.push(BufferReuse {
            buffer_index,
            previous,
            next,
            previous_reusable_after: BufferLifetimeEnd::GraphTopoStep(previous_end),
            next_first_use: BufferLifetimeEnd::GraphTopoStep(first_topo_step),
        });
        self.buffers[buffer_index].reusable_after =
            Some(BufferLifetimeEnd::GraphTopoStep(last_topo_step));
        self.current_identities[buffer_index] = next;
        self.tensor_allocations[tensor_index] = Some(TensorAllocation {
            tensor,
            buffer_index,
        });
        Ok(buffer_index)
    }

    /// Assigns a compiler-generated physical temporary.
    ///
    /// `producer_step` is the inclusive compiled-step index that writes the
    /// temporary. Initial liveness ends at the producer. Consumers extend it
    /// through `extend_temporary_lifetime`.
    pub fn assign_temporary(
        &mut self,
        len_elements: usize,
        purpose: BufferPurpose,
        producer_step: usize,
    ) -> Result<(TemporaryBufferId, usize), CompileError> {
        if len_elements == 0 {
            return Err(CompileError::InvalidInput);
        }

        let temporary_index = u32::try_from(self.temporary_allocations.len())
            .map_err(|_| CompileError::InvalidInput)?;

        let temporary = TemporaryBufferId::new(temporary_index);

        let buffer_index = self.buffers.len();

        self.buffers.push(BufferSlot {
            buffer_index,
            identity: BufferIdentity::Temporary(temporary),
            len_elements,
            purpose,
            reusable_after: Some(BufferLifetimeEnd::CompiledStep(producer_step)),
        });
        self.current_identities
            .push(BufferIdentity::Temporary(temporary));

        self.temporary_allocations.push(TemporaryAllocation {
            temporary,
            buffer_index,
            producer_step,
        });

        Ok((temporary, buffer_index))
    }

    /// Extends one temporary through an inclusive compiled consumer step.
    ///
    /// Contract:
    ///
    /// - `consumer_step` is in compiled-step space;
    /// - it must not precede the temporary producer;
    /// - lifetime extension is monotonic;
    /// - passing an earlier consumer does not shrink the lifetime;
    /// - Graph-topological liveness markers are rejected rather than mixed
    ///   with compiled-step indexes.
    pub fn extend_temporary_lifetime(
        &mut self,
        temporary: TemporaryBufferId,
        consumer_step: usize,
    ) -> Result<(), CompileError> {
        let allocation = self
            .temporary_allocations
            .get(temporary.index())
            .ok_or(CompileError::InvalidInput)?;

        if allocation.temporary != temporary || consumer_step < allocation.producer_step {
            return Err(CompileError::InvalidInput);
        }

        let slot = self
            .buffers
            .get_mut(allocation.buffer_index)
            .ok_or(CompileError::InvalidInput)?;

        if slot.identity != BufferIdentity::Temporary(temporary) {
            return Err(CompileError::InvalidInput);
        }

        let current_end = match slot.reusable_after {
            Some(BufferLifetimeEnd::CompiledStep(step)) => step,
            Some(BufferLifetimeEnd::GraphTopoStep(_)) | None => {
                return Err(CompileError::InvalidInput);
            }
        };

        slot.reusable_after = Some(BufferLifetimeEnd::CompiledStep(
            current_end.max(consumer_step),
        ));

        Ok(())
    }

    pub fn buffer_for(&self, tensor: TensorId) -> Option<usize> {
        self.tensor_allocations
            .get(tensor.index as usize)?
            .as_ref()
            .filter(|allocation| allocation.tensor == tensor)
            .map(|allocation| allocation.buffer_index)
    }

    pub fn finish(self) -> MemoryPlan {
        MemoryPlan::from_parts_with_reuse(
            self.buffers,
            self.tensor_allocations,
            self.temporary_allocations,
            self.reuse_transitions,
        )
    }
}
