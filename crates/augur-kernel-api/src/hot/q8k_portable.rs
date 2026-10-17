use core::marker::PhantomData;
use std::rc::Rc;

use augur_core::F16Bits;

use crate::HotLayoutError;

pub const Q8K_HOT_CPU_BLOCK_LEN: usize = 256;

/// Largest accepted absolute input value for the production CPU portable
/// Q8Hot quantizer.
///
/// Q8K stores:
///
/// ```text
/// d = f16(max_abs / 127)
/// ```
/// The accepted domain ensures that converting the scale to f16 remains
/// finite:
///
/// ```text
/// max_abs <= 65504 x 127 = 8,319,008
/// ```
///
/// This bound is part of the production preparation contract. The reference
/// codec may accept a wider finite f32 domain, but the production Q8Hot path
/// rejects inputs that would generate a non-finite scale.
pub const Q8K_HOT_CPU_MAX_INPUT_ABS: f32 = 65_504.0 * 127.0;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Q8KActivationHotCpuBlock {
    pub d: F16Bits,
    pub qs: [i8; Q8K_HOT_CPU_BLOCK_LEN],
}

impl Q8KActivationHotCpuBlock {
    pub const DECODED_LEN: usize = Q8K_HOT_CPU_BLOCK_LEN;

    /// Canonical zero representation.
    pub const ZERO: Self = Self {
        d: F16Bits::ZERO,
        qs: [0; Q8K_HOT_CPU_BLOCK_LEN],
    };
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Q8KActivationHotCpuVector {
    len_elements: usize,
    blocks: Vec<Q8KActivationHotCpuBlock>,
}

impl Q8KActivationHotCpuVector {
    pub fn new(
        len_elements: usize,
        blocks: Vec<Q8KActivationHotCpuBlock>,
    ) -> Result<Self, HotLayoutError> {
        let expected_blocks = block_count_for_len(len_elements)?;
        if blocks.len() != expected_blocks {
            return Err(HotLayoutError::BadBlockCount {
                expected: expected_blocks,
                actual: blocks.len(),
            });
        }
        Ok(Self {
            len_elements,
            blocks,
        })
    }

    /// Allocates a fixed-size canonical-zero Q8Hot owner.
    ///
    /// This constructor is intended for resource binding, plan construction
    /// or execution-context initialization. It must not be called from an
    /// ExecutionPlan timed execute path or from a kernel inner loop.
    ///
    /// After construction:
    ///
    /// - logical length is fixed;
    /// - block count is fixed;
    /// - the writer API cannot resize or reserve the backing allocation.
    pub fn new_zeroed(len_elements: usize) -> Result<Self, HotLayoutError> {
        let block_count = block_count_for_len(len_elements)?;
        let blocks = vec![Q8KActivationHotCpuBlock::ZERO; block_count];
        Ok(Self {
            len_elements,
            blocks,
        })
    }

    pub fn len_elements(&self) -> usize {
        self.len_elements
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    pub fn blocks(&self) -> &[Q8KActivationHotCpuBlock] {
        &self.blocks
    }

    pub fn block(&self, index: usize) -> Option<&Q8KActivationHotCpuBlock> {
        self.blocks.get(index)
    }

    /// Creates a fixed-shape producer view over existing Q8Hot blocks.
    ///
    /// The writer:
    ///
    /// - cannot resize the owner;
    /// - cannot reserve capacity;
    /// - cannot change logical length;
    /// - cannot change block count;
    /// - does not expose the backing Vec;
    /// - is conservatively neither Send nor Sync.
    ///
    /// This is a producer API. Callers are responsible for writing values
    /// that satisfy the Q8Hot numerical protocol.
    pub fn writer(&mut self) -> Q8KActivationHotCpuWriter<'_> {
        Q8KActivationHotCpuWriter {
            blocks: &mut self.blocks,
            _not_send_sync: PhantomData,
        }
    }
}

/// Fixed-shape mutable producer view for Q8Hot blocks.
///
/// This type deliberately does not implement:
///
/// - `DerefMut`;
/// - `AsMut<[...]>`;
/// - `Clone`;
/// - `Copy`;
/// - by-value `IntoIterator`.
///
/// It cannot mutate the owner shape or allocation capacity.
///
/// The `Rc` marker makes this writer conservatively `!Send + !Sync` without
/// an unsafe negative implementation. Future parallel quantization must use
/// explicitly disjoint block-range writers with a separate safety review.
pub struct Q8KActivationHotCpuWriter<'a> {
    blocks: &'a mut [Q8KActivationHotCpuBlock],
    _not_send_sync: PhantomData<Rc<()>>,
}

impl Q8KActivationHotCpuWriter<'_> {
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    pub fn iter_mut(&mut self) -> core::slice::IterMut<'_, Q8KActivationHotCpuBlock> {
        self.blocks.iter_mut()
    }
}

fn block_count_for_len(len_elements: usize) -> Result<usize, HotLayoutError> {
    if len_elements == 0 || !len_elements.is_multiple_of(Q8KActivationHotCpuBlock::DECODED_LEN) {
        return Err(HotLayoutError::BadShape);
    }
    Ok(len_elements / Q8KActivationHotCpuBlock::DECODED_LEN)
}
