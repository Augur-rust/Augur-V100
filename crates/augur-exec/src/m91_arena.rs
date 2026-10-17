use crate::MemoryPlan;
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct M91ArenaBinding {
    pub buffer_index: usize,
    pub offset_bytes: usize,
    pub len_bytes: usize,
    pub alignment_bytes: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct M91ArenaPlan {
    bindings: Vec<M91ArenaBinding>,
    total_bytes: usize,
    max_alignment_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum M91ArenaError {
    EmptyPlan,
    MetadataLengthMismatch,
    InvalidBufferIndex { index: usize },
    ZeroBufferExtent { index: usize },
    ZeroElementWidth { index: usize },
    InvalidAlignment { index: usize },
    ArithmeticOverflow,
    ArenaLimit { required: usize, limit: usize },
    BufferOutOfRange { index: usize },
}

impl fmt::Display for M91ArenaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid M91 execution arena: {self:?}")
    }
}

impl std::error::Error for M91ArenaError {}

impl M91ArenaPlan {
    pub fn new(
        memory: &MemoryPlan,
        bytes_per_element: &[usize],
        alignments: &[usize],
        arena_limit_bytes: usize,
    ) -> Result<Self, M91ArenaError> {
        if memory.buffers().is_empty() {
            return Err(M91ArenaError::EmptyPlan);
        }
        if bytes_per_element.len() != memory.buffers().len()
            || alignments.len() != memory.buffers().len()
        {
            return Err(M91ArenaError::MetadataLengthMismatch);
        }

        let mut bindings = Vec::with_capacity(memory.buffers().len());
        let mut cursor = 0usize;
        let mut max_alignment_bytes = 1usize;
        for (index, slot) in memory.buffers().iter().enumerate() {
            if slot.buffer_index != index {
                return Err(M91ArenaError::InvalidBufferIndex { index });
            }
            if slot.len_elements == 0 {
                return Err(M91ArenaError::ZeroBufferExtent { index });
            }
            let element_width = bytes_per_element[index];
            if element_width == 0 {
                return Err(M91ArenaError::ZeroElementWidth { index });
            }
            let alignment = alignments[index];
            if alignment == 0 || !alignment.is_power_of_two() {
                return Err(M91ArenaError::InvalidAlignment { index });
            }
            let len_bytes = slot
                .len_elements
                .checked_mul(element_width)
                .ok_or(M91ArenaError::ArithmeticOverflow)?;
            let offset_bytes = align_up(cursor, alignment)?;
            cursor = offset_bytes
                .checked_add(len_bytes)
                .ok_or(M91ArenaError::ArithmeticOverflow)?;
            max_alignment_bytes = max_alignment_bytes.max(alignment);
            bindings.push(M91ArenaBinding {
                buffer_index: index,
                offset_bytes,
                len_bytes,
                alignment_bytes: alignment,
            });
        }
        let total_bytes = align_up(cursor, max_alignment_bytes)?;
        if total_bytes > arena_limit_bytes {
            return Err(M91ArenaError::ArenaLimit {
                required: total_bytes,
                limit: arena_limit_bytes,
            });
        }
        Ok(Self {
            bindings,
            total_bytes,
            max_alignment_bytes,
        })
    }

    pub fn bindings(&self) -> &[M91ArenaBinding] {
        &self.bindings
    }

    pub fn binding(&self, buffer_index: usize) -> Option<M91ArenaBinding> {
        self.bindings
            .get(buffer_index)
            .copied()
            .filter(|binding| binding.buffer_index == buffer_index)
    }

    pub const fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    pub const fn max_alignment_bytes(&self) -> usize {
        self.max_alignment_bytes
    }
}

#[derive(Debug)]
pub struct M91ExecutionArena {
    plan: M91ArenaPlan,
    storage: Box<[u8]>,
}

impl M91ExecutionArena {
    /// Performs the sole arena allocation before execution begins.
    pub fn new(plan: M91ArenaPlan) -> Self {
        let storage = vec![0; plan.total_bytes()].into_boxed_slice();
        Self { plan, storage }
    }

    pub const fn plan(&self) -> &M91ArenaPlan {
        &self.plan
    }

    pub fn view(&self, buffer_index: usize) -> Result<&[u8], M91ArenaError> {
        let binding = self
            .plan
            .binding(buffer_index)
            .ok_or(M91ArenaError::BufferOutOfRange {
                index: buffer_index,
            })?;
        Ok(&self.storage[binding.offset_bytes..binding.offset_bytes + binding.len_bytes])
    }

    pub fn view_mut(&mut self, buffer_index: usize) -> Result<&mut [u8], M91ArenaError> {
        let binding = self
            .plan
            .binding(buffer_index)
            .ok_or(M91ArenaError::BufferOutOfRange {
                index: buffer_index,
            })?;
        Ok(&mut self.storage[binding.offset_bytes..binding.offset_bytes + binding.len_bytes])
    }

    pub fn reset(&mut self) {
        self.storage.fill(0);
    }

    pub fn allocation_identity(&self) -> (*const u8, usize) {
        (self.storage.as_ptr(), self.storage.len())
    }
}

fn align_up(value: usize, alignment: usize) -> Result<usize, M91ArenaError> {
    let mask = alignment - 1;
    value
        .checked_add(mask)
        .map(|sum| sum & !mask)
        .ok_or(M91ArenaError::ArithmeticOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BufferIdentity, BufferPurpose, BufferSlot, MemoryPlan};
    use augur_core::TensorId;

    fn memory() -> MemoryPlan {
        MemoryPlan::from_parts(
            vec![
                BufferSlot {
                    buffer_index: 0,
                    identity: BufferIdentity::Tensor(TensorId::new(0, 0)),
                    len_elements: 3,
                    purpose: BufferPurpose::Input,
                    reusable_after: None,
                },
                BufferSlot {
                    buffer_index: 1,
                    identity: BufferIdentity::Tensor(TensorId::new(1, 0)),
                    len_elements: 5,
                    purpose: BufferPurpose::Output,
                    reusable_after: None,
                },
            ],
            vec![],
            vec![],
        )
    }

    #[test]
    fn plan_freezes_aligned_offsets_and_single_allocation() {
        let plan = M91ArenaPlan::new(&memory(), &[4, 2], &[16, 8], 64).unwrap();
        assert_eq!(plan.bindings()[0].offset_bytes, 0);
        assert_eq!(plan.bindings()[0].len_bytes, 12);
        assert_eq!(plan.bindings()[1].offset_bytes, 16);
        assert_eq!(plan.bindings()[1].len_bytes, 10);
        assert_eq!(plan.total_bytes(), 32);

        let mut arena = M91ExecutionArena::new(plan);
        let identity = arena.allocation_identity();
        arena.view_mut(1).unwrap().fill(7);
        assert!(arena.view(1).unwrap().iter().all(|&byte| byte == 7));
        arena.reset();
        assert_eq!(arena.allocation_identity(), identity);
        assert!(arena.view(1).unwrap().iter().all(|&byte| byte == 0));
    }

    #[test]
    fn invalid_alignment_metadata_and_limit_fail_closed() {
        assert_eq!(
            M91ArenaPlan::new(&memory(), &[4], &[16], 64),
            Err(M91ArenaError::MetadataLengthMismatch)
        );
        assert_eq!(
            M91ArenaPlan::new(&memory(), &[4, 2], &[3, 8], 64),
            Err(M91ArenaError::InvalidAlignment { index: 0 })
        );
        assert_eq!(
            M91ArenaPlan::new(&memory(), &[4, 2], &[16, 8], 31),
            Err(M91ArenaError::ArenaLimit {
                required: 32,
                limit: 31
            })
        );
    }
}
