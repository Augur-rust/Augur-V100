//! CPU admission half of the real page owner. Issuing happens only after the
//! transport has fenced any previous physical use and submitted the new DMA.
//! Retiring is a consumer-release *submission*, not CUDA completion; the next
//! restore must still finish the physical release/readback before overwriting.
use super::{invalid, Qwen35MixedPrefixCudaError, LANES, PAGE};
use std::sync::atomic::{AtomicU64, Ordering};
static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PageBinding {
    pub step: usize,
    pub layer: usize,
    pub begin: usize,
    pub end: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PageLease {
    owner: u64,
    slot: usize,
    generation: u64,
    binding: PageBinding,
}

impl PageLease {
    pub fn generation(&self) -> u64 {
        self.generation
    }
    /// Diagnostic replay of a genuinely retired generation while preserving
    /// the current step/layer/position binding. This isolates generation
    /// validation instead of merely failing on an old step's identity.
    pub fn replay_generation(self, old: Self) -> Result<Self, Qwen35MixedPrefixCudaError> {
        if old.owner != self.owner || old.slot != self.slot || old.generation >= self.generation {
            return Err(invalid(
                "generation replay needs an older same-owner same-slot ticket",
            ));
        }
        Ok(Self {
            generation: old.generation,
            ..self
        })
    }
}

pub(super) struct PageLeaseBook {
    owner: u64,
    generations: [u64; LANES],
    live: [Option<(PageLease, bool)>; LANES], // bool: consumer acquired
    poisoned: bool,
    pub issued: u64,
    pub acquired: u64,
    pub retired: u64,
}

impl PageLeaseBook {
    pub fn new() -> Result<Self, Qwen35MixedPrefixCudaError> {
        let owner = NEXT_OWNER
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| invalid("page lease owner exhaustion"))?;
        Ok(Self {
            owner,
            generations: [0; LANES],
            live: [None; LANES],
            poisoned: false,
            issued: 0,
            acquired: 0,
            retired: 0,
        })
    }

    pub fn reusable(&self, slot: usize) -> Result<(), Qwen35MixedPrefixCudaError> {
        if self.poisoned
            || slot >= LANES
            || self.live[slot].is_some()
            || self.generations[slot] == u64::MAX
        {
            return Err(invalid("page lease owner poisoned, pending or exhausted"));
        }
        Ok(())
    }

    pub fn issue(
        &mut self,
        slot: usize,
        binding: PageBinding,
    ) -> Result<PageLease, Qwen35MixedPrefixCudaError> {
        self.reusable(slot)?;
        if binding.layer >= 64
            || binding.layer % 4 != 3
            || binding.end <= binding.begin
            || binding.end - binding.begin > PAGE
            || !binding.begin.is_multiple_of(PAGE)
        {
            return Err(invalid("page lease binding extent"));
        }
        self.generations[slot] += 1;
        let ticket = PageLease {
            owner: self.owner,
            slot,
            generation: self.generations[slot],
            binding,
        };
        self.live[slot] = Some((ticket, false));
        self.issued += 1;
        Ok(ticket)
    }

    pub fn acquire(
        &mut self,
        ticket: PageLease,
        binding: PageBinding,
    ) -> Result<(), Qwen35MixedPrefixCudaError> {
        if self.poisoned
            || ticket.slot >= LANES
            || ticket.owner != self.owner
            || ticket.binding != binding
            || self.live[ticket.slot] != Some((ticket, false))
        {
            self.poison();
            return Err(invalid(
                "page lease stale generation, owner, identity or phase",
            ));
        }
        self.live[ticket.slot] = Some((ticket, true));
        self.acquired += 1;
        Ok(())
    }

    pub fn retire(&mut self, ticket: PageLease) -> Result<(), Qwen35MixedPrefixCudaError> {
        if self.poisoned || ticket.slot >= LANES || self.live[ticket.slot] != Some((ticket, true)) {
            self.poison();
            return Err(invalid("page lease release does not match active consumer"));
        }
        self.live[ticket.slot] = None;
        self.retired += 1;
        Ok(())
    }

    pub fn terminal(&self) -> bool {
        !self.poisoned
            && self.live.iter().all(Option::is_none)
            && self.issued == self.acquired
            && self.acquired == self.retired
    }

    pub fn poison(&mut self) {
        self.poisoned = true;
    }
    pub fn cancel(&mut self) -> Result<(), Qwen35MixedPrefixCudaError> {
        self.poison();
        Err(invalid("page lease request cancelled"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn binding() -> PageBinding {
        PageBinding {
            step: 0,
            layer: 3,
            begin: 0,
            end: 512,
        }
    }
    #[test]
    fn complete_lifecycle_requires_consumer_release() {
        let mut b = PageLeaseBook::new().unwrap();
        let t = b.issue(0, binding()).unwrap();
        assert!(!b.terminal());
        assert!(b.reusable(0).is_err());
        b.acquire(t, binding()).unwrap();
        b.retire(t).unwrap();
        assert!(b.terminal());
    }
    #[test]
    fn stale_ticket_rejected_after_real_slot_reuse_pattern() {
        let mut b = PageLeaseBook::new().unwrap();
        let old = b.issue(0, binding()).unwrap();
        b.acquire(old, binding()).unwrap();
        b.retire(old).unwrap();
        let new = b.issue(0, binding()).unwrap();
        assert_ne!(old.generation, new.generation);
        let replay = new.replay_generation(old).unwrap();
        assert_eq!(replay.binding, new.binding);
        assert!(b.acquire(replay, binding()).is_err());
        assert!(b.acquire(new, binding()).is_err()); // sticky poison
    }
    #[test]
    fn foreign_owner_and_wrong_step_layer_position_reject() {
        let mut a = PageLeaseBook::new().unwrap();
        let t = a.issue(0, binding()).unwrap();
        let mut b = PageLeaseBook::new().unwrap();
        b.issue(0, binding()).unwrap();
        assert!(b.acquire(t, binding()).is_err());
        for binding in [
            PageBinding {
                step: 1,
                ..binding()
            },
            PageBinding {
                layer: 7,
                ..binding()
            },
            PageBinding {
                begin: 512,
                end: 1024,
                ..binding()
            },
        ] {
            let mut b = PageLeaseBook::new().unwrap();
            let t = b.issue(0, super::tests::binding()).unwrap();
            assert!(b.acquire(t, binding).is_err());
        }
    }
    #[test]
    fn cancel_blocks_issue_acquire_and_terminal_before_or_after_dma() {
        let mut b = PageLeaseBook::new().unwrap();
        assert!(b.cancel().is_err());
        assert!(b.issue(0, binding()).is_err());
        let mut b = PageLeaseBook::new().unwrap();
        let t = b.issue(0, binding()).unwrap();
        b.cancel().unwrap_err();
        assert!(b.acquire(t, binding()).is_err());
        assert!(!b.terminal());
    }
    #[test]
    fn duplicate_acquire_release_and_partial_geometry_reject() {
        let mut b = PageLeaseBook::new().unwrap();
        let t = b.issue(0, binding()).unwrap();
        b.acquire(t, binding()).unwrap();
        assert!(b.acquire(t, binding()).is_err());
        let mut b = PageLeaseBook::new().unwrap();
        let t = b.issue(0, binding()).unwrap();
        assert!(b.retire(t).is_err());
        let mut b = PageLeaseBook::new().unwrap();
        assert!(b.issue(4, binding()).is_err());
        assert!(b
            .issue(
                0,
                PageBinding {
                    end: 513,
                    ..binding()
                }
            )
            .is_err());
        assert!(b
            .issue(
                0,
                PageBinding {
                    begin: 512,
                    end: 513,
                    ..binding()
                }
            )
            .is_ok());
    }
}
