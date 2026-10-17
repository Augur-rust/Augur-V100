//! Physical KV dirty-prefix tracking. Never use only the committed token cursor:
//! rotated pack kernels pad their writes and speculative commits can rewind it.
use super::{Buffer, Result};
use augur_backend::OwnedCudaStream;
use std::{collections::BTreeMap, rc::Rc};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct DirtyPrefix {
    capacity: usize,
    row_bytes: usize,
    rows: usize,
}
impl DirtyPrefix {
    pub(super) fn new(capacity: usize, row_bytes: usize) -> Result<Self> {
        if capacity == 0 || row_bytes == 0 || capacity.checked_mul(row_bytes).is_none() {
            return Err("invalid physical KV reset geometry".into());
        }
        Ok(Self {
            capacity,
            row_bytes,
            rows: 0,
        })
    }
    /// Called before the first possible cache write, not after successful submit.
    /// `alignment` bounds padding written by the selected kernel (up to 256 rows).
    pub(super) fn mark(&mut self, position: usize, rows: usize, alignment: usize) -> Result<()> {
        if rows == 0 || !alignment.is_power_of_two() {
            return Err("invalid physical KV write span".into());
        }
        let end = position
            .checked_add(rows)
            .ok_or("physical KV span overflow")?;
        let padded = end
            .checked_add(alignment - 1)
            .ok_or("physical KV padding overflow")?
            & !(alignment - 1);
        if end > self.capacity || padded > self.capacity {
            return Err("physical KV write exceeds capacity".into());
        }
        self.rows = self.rows.max(padded);
        Ok(())
    }
    fn clear_bytes(&self, committed: usize, poisoned: bool) -> usize {
        // Unknown/failed submissions and an uncovered logical frontier fail closed
        // to the old full-capacity scrub. No bytes become valid by changing metadata.
        let rows = if poisoned || committed > self.rows {
            self.capacity
        } else {
            self.rows
        };
        rows * self.row_bytes // constructor proved capacity*row_bytes cannot overflow
    }
}

pub(super) fn reset_buffers(
    stream: &OwnedCudaStream,
    buffers: &mut BTreeMap<&'static str, Buffer>,
    dirty: &mut DirtyPrefix,
    tokens: &mut usize,
    poisoned: &mut bool,
) -> Result<()> {
    let bytes = dirty.clear_bytes(*tokens, *poisoned);
    *poisoned = true;
    stream.synchronize()?;
    for name in ["key_cache", "value_cache"] {
        let buffer = Rc::get_mut(buffers.get_mut(name).ok_or("missing cache owner")?)
            .ok_or("checkpoint cache owner unexpectedly shared")?;
        if buffer.len_bytes() != dirty.capacity * dirty.row_bytes {
            return Err("physical KV reset allocation changed".into());
        }
        buffer.zero_prefix_sync(stream, bytes)?;
    }
    // Commit only after both physical banks complete. On any failure, retain the
    // dirty high-water mark and poisoned state; a retry scrubs full capacity.
    dirty.rows = 0;
    *tokens = 0;
    *poisoned = false;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use augur_backend::{cuda_device_allocation_snapshot, OwnedCudaBuffer};
    #[test]
    fn padding_and_rejected_tail_survive_logical_rewind() {
        let mut d = DirtyPrefix::new(262144, 1088).unwrap();
        d.mark(0, 255, 32).unwrap();
        d.mark(255, 8, 256).unwrap();
        assert_eq!(d.rows, 512);
        assert_eq!(d.clear_bytes(257, false), 512 * 1088);
        d.mark(257, 1, 256).unwrap();
        assert_eq!(d.rows, 512);
    }
    #[test]
    fn poison_or_missing_coverage_uses_full_capacity() {
        let mut d = DirtyPrefix::new(1024, 1088).unwrap();
        d.mark(0, 8, 256).unwrap();
        assert_eq!(d.clear_bytes(8, true), 1024 * 1088);
        assert_eq!(d.clear_bytes(257, false), 1024 * 1088);
        assert_eq!(d.clear_bytes(usize::MAX, false), 1024 * 1088);
    }
    #[test]
    fn bad_spans_do_not_advance_high_water() {
        let mut d = DirtyPrefix::new(262144, 1088).unwrap();
        d.mark(0, 73, 256).unwrap();
        let before = d;
        for (pos, rows, align) in [
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 3),
            (262144, 1, 256),
            (usize::MAX, 1, 1),
        ] {
            assert!(d.mark(pos, rows, align).is_err());
            assert_eq!(d, before);
        }
        assert!(DirtyPrefix::new(0, 1).is_err());
        assert!(DirtyPrefix::new(1, 0).is_err());
        assert!(DirtyPrefix::new(usize::MAX, 2).is_err());
    }
    #[test]
    fn full_capacity_and_ordinary_f32_rows() {
        let mut d = DirtyPrefix::new(262144, 1088).unwrap();
        d.mark(262143, 1, 256).unwrap();
        assert_eq!(d.clear_bytes(262144, false), 262144 * 1088);
        // Flash32's smaller legal capacity must not be rounded to 256 rows.
        let mut flash32 = DirtyPrefix::new(96, 1088).unwrap();
        flash32.mark(95, 1, 32).unwrap();
        assert_eq!(flash32.clear_bytes(96, false), 96 * 1088);
        let mut f = DirtyPrefix::new(7, 4096).unwrap();
        f.mark(0, 1, 1).unwrap();
        assert_eq!(f.clear_bytes(1, false), 4096);
        assert_eq!(f.clear_bytes(0, false), 4096); // rewind never drops dirty rows
        assert_eq!(DirtyPrefix::new(7, 4096).unwrap().clear_bytes(0, false), 0);
    }
    #[test]
    #[ignore = "exclusive CUDA device; exact physical reset/rollback/poison/lease guards"]
    fn physical_prefix_reset_covers_padding_rewind_poison_and_alias_failure() {
        let before = cuda_device_allocation_snapshot();
        let stream = OwnedCudaStream::create(0).unwrap();
        let mut buffers = BTreeMap::new();
        let capacity = 1024;
        let row_bytes = 1088;
        for name in ["key_cache", "value_cache"] {
            let mut b = OwnedCudaBuffer::allocate(0, capacity * row_bytes).unwrap();
            b.zero_sync(&stream).unwrap();
            buffers.insert(name, Rc::new(b));
        }
        let mut dirty = DirtyPrefix::new(capacity, row_bytes).unwrap();
        let mut tokens = 0;
        let mut poisoned = false;
        for scenario in 0..5 {
            dirty.mark(0, 255, 256).unwrap();
            dirty.mark(255, 8, 256).unwrap();
            tokens = 257; // emulate a B8 logical rewind with rejected physical rows
            for buffer in buffers.values() {
                buffer
                    .copy_from_host_sync(&stream, &vec![0x5a; 512 * row_bytes])
                    .unwrap();
            }
            if scenario == 1 || scenario == 2 {
                // Deliberately unknown bytes beyond recorded writes: full fallback.
                for buffer in buffers.values() {
                    buffer
                        .copy_from_host_sync(&stream, &vec![0x7f; capacity * row_bytes])
                        .unwrap();
                }
                if scenario == 1 {
                    poisoned = true;
                } else {
                    tokens = 513;
                }
            }
            if scenario == 3 {
                let alias = Rc::clone(&buffers["value_cache"]);
                assert!(reset_buffers(
                    &stream,
                    &mut buffers,
                    &mut dirty,
                    &mut tokens,
                    &mut poisoned
                )
                .is_err());
                assert!(poisoned);
                assert_eq!(dirty.rows, 512);
                assert_eq!(tokens, 257);
                drop(alias);
            }
            if scenario == 4 {
                dirty.mark(1023, 1, 256).unwrap();
                tokens = capacity;
                for buffer in buffers.values() {
                    buffer
                        .copy_from_host_sync(&stream, &vec![0xa5; capacity * row_bytes])
                        .unwrap();
                }
            }
            let allocated = cuda_device_allocation_snapshot();
            reset_buffers(
                &stream,
                &mut buffers,
                &mut dirty,
                &mut tokens,
                &mut poisoned,
            )
            .unwrap();
            assert_eq!(tokens, 0);
            assert!(!poisoned);
            assert_eq!(dirty.rows, 0);
            let after = cuda_device_allocation_snapshot();
            assert_eq!(
                allocated.successful_allocations,
                after.successful_allocations
            );
            assert_eq!(allocated.successful_frees, after.successful_frees);
            for buffer in buffers.values() {
                let mut actual = vec![0xff; capacity * row_bytes];
                buffer.copy_to_host_sync(&stream, &mut actual).unwrap();
                assert!(actual.iter().all(|&v| v == 0), "scenario {scenario}");
            }
            reset_buffers(
                &stream,
                &mut buffers,
                &mut dirty,
                &mut tokens,
                &mut poisoned,
            )
            .unwrap();
        }
        drop(buffers);
        let after = cuda_device_allocation_snapshot();
        assert_eq!(before.live_bytes, after.live_bytes);
        assert_eq!(before.live_allocations, after.live_allocations);
    }
}
