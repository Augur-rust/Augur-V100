//! Bounded active-HBM/RAM-cold KV scheduling contract.
//!
//! This module deliberately contains no CUDA objects. It is the state machine
//! that the CUDA page transport must satisfy before a page is allowed to
//! participate in a V2 partial/LSE attention submission. Keeping geometry and
//! lease rules independent makes it possible to compare the segmented path
//! with the contiguous V3 oracle without weakening either lifetime proof.

use std::collections::BTreeMap;

/// Storage class selected for a KV page.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvPageResidence {
    /// The page is retained in the active HBM window and needs no transport.
    ActiveHbm,
    /// The page is materialized from the RAM-cold snapshot through a lane.
    ColdRam,
}

/// Immutable page geometry used by the transport and V2 segment producer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvPageDescriptor {
    pub page_index: usize,
    pub start_position: usize,
    pub end_position: usize,
    pub residence: KvPageResidence,
    pub lane: usize,
}

/// One bounded page lease. A lease must reach `complete` or `cancel` before
/// its lane can be reused; a generation prevents stale completion after slot
/// reuse from being accepted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvPageLease {
    pub page_index: usize,
    pub lane: usize,
    pub generation: u64,
}

/// Compact deterministic page receipt used by the future device-side proof
/// path. The four lanes are intentionally independent so a single
/// corrupted byte changes the receipt with overwhelming practical
/// probability; this is an audit receipt, not a cryptographic publication
/// identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PageDigest(pub [u64; 4]);

impl PageDigest {
    pub fn from_bytes(bytes: &[u8]) -> Self {
        const SEEDS: [u64; 4] = [
            0xcbf29ce484222325,
            0x84222325cbf29ce4,
            0x9e3779b185ebca87,
            0xd6e8feb86659fd93,
        ];
        let mut lanes = SEEDS;
        for (index, &byte) in bytes.iter().enumerate() {
            let lane = index & 3;
            let mut value = u64::from(byte)
                .wrapping_add(0x9e3779b97f4a7c15)
                .wrapping_add((index as u64).wrapping_mul(0x100000001b3))
                ^ ((lane as u64) << 61);
            value ^= value >> 30;
            value = value.wrapping_mul(0xbf58476d1ce4e5b9);
            value ^= value >> 27;
            value = value.wrapping_mul(0x94d049bb133111eb);
            lanes[lane] ^= value ^ (value >> 31);
        }
        Self(lanes)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PageProofReceipt {
    pub lease: KvPageLease,
    pub expected: PageDigest,
}

/// Bounded proof ring decoupled from the H2D lane. A digest mismatch poisons
/// the ring permanently until the enclosing request is discarded; terminal
/// therefore cannot accidentally publish a partially verified request.
#[derive(Clone, Debug)]
pub struct PageProofRing {
    capacity: usize,
    pending: BTreeMap<usize, PageProofReceipt>,
    completed: usize,
    poisoned: bool,
}

impl PageProofRing {
    pub fn new(capacity: usize) -> Result<Self, &'static str> {
        if capacity == 0 {
            return Err("proof ring capacity must be nonzero");
        }
        Ok(Self {
            capacity,
            pending: BTreeMap::new(),
            completed: 0,
            poisoned: false,
        })
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    pub fn completed(&self) -> usize {
        self.completed
    }

    pub fn poisoned(&self) -> bool {
        self.poisoned
    }

    pub fn submit(
        &mut self,
        lease: KvPageLease,
        expected: PageDigest,
    ) -> Result<PageProofReceipt, &'static str> {
        if self.poisoned {
            return Err("proof ring is poisoned");
        }
        if self.pending.len() >= self.capacity {
            return Err("proof ring is full");
        }
        if self.pending.contains_key(&lease.page_index) {
            return Err("page proof already pending");
        }
        let receipt = PageProofReceipt { lease, expected };
        self.pending.insert(lease.page_index, receipt);
        Ok(receipt)
    }

    pub fn complete(
        &mut self,
        receipt: PageProofReceipt,
        actual: PageDigest,
    ) -> Result<(), &'static str> {
        match self.pending.get(&receipt.lease.page_index) {
            Some(current) if *current == receipt => {}
            _ => return Err("stale or unknown proof receipt"),
        }
        self.pending.remove(&receipt.lease.page_index);
        if actual != receipt.expected {
            self.poisoned = true;
            return Err("page proof digest mismatch");
        }
        self.completed = self
            .completed
            .checked_add(1)
            .ok_or("proof completion counter overflow")?;
        Ok(())
    }

    pub fn cancel(&mut self, receipt: PageProofReceipt) -> Result<(), &'static str> {
        match self.pending.get(&receipt.lease.page_index) {
            Some(current) if *current == receipt => {
                self.pending.remove(&receipt.lease.page_index);
                Ok(())
            }
            _ => Err("stale or unknown proof cancel"),
        }
    }

    pub fn terminal(&self) -> Result<(), &'static str> {
        if self.poisoned {
            return Err("proof ring is poisoned");
        }
        if !self.pending.is_empty() {
            return Err("proof receipts remain pending");
        }
        Ok(())
    }
}

/// A fail-closed planner/state machine for one decode cache.
#[derive(Clone, Debug)]
pub struct ActiveColdKvPlan {
    capacity_positions: usize,
    active_window: usize,
    page_positions: usize,
    lane_count: usize,
    pages: Vec<KvPageDescriptor>,
    pending: BTreeMap<usize, KvPageLease>,
    ready: BTreeMap<usize, KvPageLease>,
    lane_generation: Vec<u64>,
}

impl ActiveColdKvPlan {
    /// Build a tail-active plan. Pages intersecting the final
    /// `active_window` positions are retained in HBM; all earlier pages are
    /// RAM-cold. A page is never split across lanes.
    pub fn new(
        capacity_positions: usize,
        active_window: usize,
        page_positions: usize,
        lane_count: usize,
    ) -> Result<Self, &'static str> {
        if capacity_positions == 0
            || active_window == 0
            || active_window > capacity_positions
            || page_positions == 0
            || lane_count == 0
        {
            return Err("invalid active-cold geometry");
        }
        let page_count = capacity_positions.div_ceil(page_positions);
        let cold_cutoff = capacity_positions - active_window;
        let mut pages = Vec::with_capacity(page_count);
        for page_index in 0..page_count {
            let start_position = page_index * page_positions;
            let end_position = (start_position + page_positions).min(capacity_positions);
            let residence = if end_position > cold_cutoff {
                KvPageResidence::ActiveHbm
            } else {
                KvPageResidence::ColdRam
            };
            pages.push(KvPageDescriptor {
                page_index,
                start_position,
                end_position,
                residence,
                lane: page_index % lane_count,
            });
        }
        Ok(Self {
            capacity_positions,
            active_window,
            page_positions,
            lane_count,
            pages,
            pending: BTreeMap::new(),
            ready: BTreeMap::new(),
            lane_generation: vec![0; lane_count],
        })
    }

    pub fn capacity_positions(&self) -> usize {
        self.capacity_positions
    }

    pub fn active_window(&self) -> usize {
        self.active_window
    }

    pub fn page_positions(&self) -> usize {
        self.page_positions
    }

    pub fn lane_count(&self) -> usize {
        self.lane_count
    }

    pub fn pages(&self) -> &[KvPageDescriptor] {
        &self.pages
    }

    pub fn cold_pages(&self) -> impl Iterator<Item = KvPageDescriptor> + '_ {
        self.pages
            .iter()
            .copied()
            .filter(|page| page.residence == KvPageResidence::ColdRam)
    }

    pub fn active_pages(&self) -> impl Iterator<Item = KvPageDescriptor> + '_ {
        self.pages
            .iter()
            .copied()
            .filter(|page| page.residence == KvPageResidence::ActiveHbm)
    }

    /// Return the pages needed for an attention query at an absolute position.
    /// The page set is deterministic and includes every page intersecting the
    /// visible prefix `[0, position]`.
    pub fn demand_pages(&self, position: usize) -> Result<Vec<KvPageDescriptor>, &'static str> {
        if position >= self.capacity_positions {
            return Err("attention position exceeds cache capacity");
        }
        let last_page = position / self.page_positions;
        Ok(self.pages[..=last_page].to_vec())
    }

    /// Reserve one cold page on its predetermined lane. Active-HBM pages do
    /// not create leases because they have no host transport boundary.
    pub fn reserve(&mut self, page_index: usize) -> Result<Option<KvPageLease>, &'static str> {
        let page = *self
            .pages
            .get(page_index)
            .ok_or("page index exceeds cache geometry")?;
        if page.residence == KvPageResidence::ActiveHbm {
            return Ok(None);
        }
        if self.pending.contains_key(&page_index) {
            return Err("page already pending");
        }
        if self.pending.values().any(|lease| lease.lane == page.lane) {
            return Err("lane already has a pending page");
        }
        let generation = self.lane_generation[page.lane].saturating_add(1);
        self.lane_generation[page.lane] = generation;
        let lease = KvPageLease {
            page_index,
            lane: page.lane,
            generation,
        };
        self.pending.insert(page_index, lease);
        Ok(Some(lease))
    }

    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Number of DMA-complete pages that have not yet been promoted or
    /// terminally released. A ready page still owns its lane slot.
    pub fn ready_len(&self) -> usize {
        self.ready.len()
    }

    pub fn is_pending(&self, page_index: usize) -> bool {
        self.pending.contains_key(&page_index)
    }

    /// Validate the exact lease at submit time. The CUDA implementation must
    /// bind its stream/event receipt to this generation before DMA begins.
    pub fn submit(&self, lease: KvPageLease) -> Result<(), &'static str> {
        match self.pending.get(&lease.page_index) {
            Some(current) if *current == lease => Ok(()),
            _ => Err("stale or unknown page lease"),
        }
    }

    /// Record that the RAM→HBM transfer reached its ready event. This is
    /// intentionally separate from `complete`: consumers may submit the V2
    /// partial attention work as soon as the page is ready, while the lane
    /// remains owned until `promote`/`cancel` at the terminal boundary.
    pub fn mark_ready(&mut self, lease: KvPageLease) -> Result<(), &'static str> {
        match self.pending.get(&lease.page_index) {
            Some(current) if *current == lease => {
                self.ready.insert(lease.page_index, lease);
                Ok(())
            }
            _ => Err("stale or unknown page ready"),
        }
    }

    pub fn is_ready(&self, page_index: usize) -> bool {
        self.ready.contains_key(&page_index)
    }

    /// Promote a ready page after its consumer has recorded the device-side
    /// use. Promotion releases the bounded lane for reuse; the page's actual
    /// residency is tracked by the caller's HBM slot table.
    pub fn promote(&mut self, lease: KvPageLease) -> Result<(), &'static str> {
        match self.ready.get(&lease.page_index) {
            Some(current) if *current == lease => {
                self.ready.remove(&lease.page_index);
                self.pending.remove(&lease.page_index);
                Ok(())
            }
            _ => Err("page is not ready or lease is stale"),
        }
    }

    pub fn complete(&mut self, lease: KvPageLease) -> Result<(), &'static str> {
        self.finish(lease)
    }

    pub fn cancel(&mut self, lease: KvPageLease) -> Result<(), &'static str> {
        self.finish(lease)
    }

    fn finish(&mut self, lease: KvPageLease) -> Result<(), &'static str> {
        match self.pending.get(&lease.page_index) {
            Some(current) if *current == lease => {
                self.pending.remove(&lease.page_index);
                self.ready.remove(&lease.page_index);
                Ok(())
            }
            _ => Err("stale or unknown page terminal"),
        }
    }

    /// Terminal is valid only after every cold lease has reached a terminal
    /// state. This is the handoff point at which page slots may be reused or
    /// the decode workspace may be released.
    pub fn terminal(&self) -> Result<(), &'static str> {
        if self.pending.is_empty() {
            Ok(())
        } else {
            Err("pending page leases remain")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tail_active_geometry_and_lane_assignment_are_deterministic() {
        let plan = ActiveColdKvPlan::new(10, 3, 4, 4).unwrap();
        assert_eq!(plan.pages().len(), 3);
        assert_eq!(plan.pages()[0].residence, KvPageResidence::ColdRam);
        assert_eq!(plan.pages()[1].residence, KvPageResidence::ActiveHbm);
        assert_eq!(plan.pages()[2].residence, KvPageResidence::ActiveHbm);
        assert_eq!(plan.pages()[0].lane, 0);
        assert_eq!(plan.pages()[1].lane, 1);
    }

    #[test]
    fn demand_uses_absolute_position_and_includes_partial_tail_page() {
        let plan = ActiveColdKvPlan::new(10, 2, 4, 4).unwrap();
        let pages = plan.demand_pages(9).unwrap();
        assert_eq!(pages.len(), 3);
        assert_eq!(pages[2].start_position, 8);
        assert_eq!(pages[2].end_position, 10);
        assert!(plan.demand_pages(10).is_err());
    }

    #[test]
    fn exact_reserve_submit_complete_terminal_lifecycle() {
        let mut plan = ActiveColdKvPlan::new(32, 8, 8, 4).unwrap();
        let lease = plan.reserve(0).unwrap().unwrap();
        assert!(plan.submit(lease).is_ok());
        assert!(plan.terminal().is_err());
        plan.complete(lease).unwrap();
        assert!(plan.terminal().is_ok());
    }

    #[test]
    fn lane_is_bounded_and_duplicate_or_stale_completion_is_rejected() {
        let mut plan = ActiveColdKvPlan::new(64, 8, 8, 4).unwrap();
        let first = plan.reserve(0).unwrap().unwrap();
        assert!(plan.reserve(4).is_err());
        assert!(plan
            .submit(KvPageLease {
                generation: 0,
                ..first
            })
            .is_err());
        plan.cancel(first).unwrap();
        assert!(plan.cancel(first).is_err());
        let reused = plan.reserve(4).unwrap().unwrap();
        assert!(reused.generation > first.generation);
        assert!(plan.complete(first).is_err());
        plan.complete(reused).unwrap();
    }

    #[test]
    fn ready_is_distinct_from_terminal_and_promotion_releases_lane() {
        let mut plan = ActiveColdKvPlan::new(64, 8, 8, 1).unwrap();
        let first = plan.reserve(0).unwrap().unwrap();
        plan.submit(first).unwrap();
        plan.mark_ready(first).unwrap();
        assert_eq!(plan.pending_len(), 1);
        assert_eq!(plan.ready_len(), 1);
        assert!(plan.terminal().is_err());
        assert!(plan.reserve(1).is_err());
        plan.promote(first).unwrap();
        assert_eq!(plan.pending_len(), 0);
        assert_eq!(plan.ready_len(), 0);
        assert!(plan.terminal().is_ok());
        let reused = plan.reserve(1).unwrap().unwrap();
        assert!(reused.generation > first.generation);
    }

    #[test]
    fn ready_and_promotion_are_generation_bound() {
        let mut plan = ActiveColdKvPlan::new(64, 8, 8, 2).unwrap();
        let lease = plan.reserve(0).unwrap().unwrap();
        plan.submit(lease).unwrap();
        let stale = KvPageLease {
            generation: 0,
            ..lease
        };
        assert!(plan.mark_ready(stale).is_err());
        assert!(plan.promote(stale).is_err());
        plan.mark_ready(lease).unwrap();
        assert!(plan.promote(stale).is_err());
        plan.cancel(lease).unwrap();
        assert!(!plan.is_ready(lease.page_index));
    }

    #[test]
    fn proof_ring_is_independent_and_generation_bound() {
        let mut plan = ActiveColdKvPlan::new(64, 8, 8, 1).unwrap();
        let lease = plan.reserve(0).unwrap().unwrap();
        plan.submit(lease).unwrap();
        let mut ring = PageProofRing::new(2).unwrap();
        let bytes = vec![7_u8; 64];
        let receipt = ring.submit(lease, PageDigest::from_bytes(&bytes)).unwrap();
        assert_eq!(ring.pending_len(), 1);
        assert!(ring.terminal().is_err());
        assert!(ring
            .complete(
                PageProofReceipt {
                    lease: KvPageLease {
                        generation: 0,
                        ..lease
                    },
                    ..receipt
                },
                PageDigest::from_bytes(&bytes)
            )
            .is_err());
        ring.complete(receipt, PageDigest::from_bytes(&bytes))
            .unwrap();
        assert_eq!(ring.completed(), 1);
        assert!(ring.terminal().is_ok());
        plan.mark_ready(lease).unwrap();
        plan.promote(lease).unwrap();
    }

    #[test]
    fn proof_ring_digest_mismatch_poison_is_fail_closed() {
        let mut plan = ActiveColdKvPlan::new(64, 8, 8, 1).unwrap();
        let lease = plan.reserve(0).unwrap().unwrap();
        let mut ring = PageProofRing::new(1).unwrap();
        let receipt = ring
            .submit(lease, PageDigest::from_bytes(&[1, 2, 3]))
            .unwrap();
        assert!(ring
            .complete(receipt, PageDigest::from_bytes(&[1, 2, 4]))
            .is_err());
        assert!(ring.poisoned());
        assert!(ring.terminal().is_err());
        assert!(ring
            .submit(lease, PageDigest::from_bytes(&[1, 2, 3]))
            .is_err());
    }

    #[test]
    fn proof_ring_capacity_and_cancel_are_bounded() {
        let mut plan = ActiveColdKvPlan::new(128, 8, 8, 2).unwrap();
        let first = plan.reserve(0).unwrap().unwrap();
        let second = plan.reserve(1).unwrap().unwrap();
        let mut ring = PageProofRing::new(1).unwrap();
        let one = ring.submit(first, PageDigest::from_bytes(&[1])).unwrap();
        assert!(ring.submit(second, PageDigest::from_bytes(&[2])).is_err());
        ring.cancel(one).unwrap();
        assert_eq!(ring.pending_len(), 0);
        assert!(ring.terminal().is_ok());
    }

    #[test]
    fn active_pages_do_not_consume_transport_leases() {
        let mut plan = ActiveColdKvPlan::new(16, 8, 4, 4).unwrap();
        assert!(plan.reserve(2).unwrap().is_none());
        assert!(plan.reserve(3).unwrap().is_none());
        assert_eq!(plan.pending_len(), 0);
        assert_eq!(plan.active_pages().count(), 2);
        assert_eq!(plan.cold_pages().count(), 2);
    }

    #[test]
    fn invalid_geometry_fails_closed() {
        assert!(ActiveColdKvPlan::new(0, 1, 1, 1).is_err());
        assert!(ActiveColdKvPlan::new(8, 9, 1, 1).is_err());
        assert!(ActiveColdKvPlan::new(8, 1, 0, 1).is_err());
        assert!(ActiveColdKvPlan::new(8, 1, 1, 0).is_err());
    }
}
