//! Cost-gate candidate, connected only through an explicit adapter opt-in.
//! The caller must measure a completed decode transaction through the next
//! synchronized target selection, including token publication and plan setup.
//! Enqueue-only timings, prefill, and cached/generated answer tokens are invalid.
//! No language, task labels, target lookahead, or acceptance-rate inputs exist.
use std::time::Duration;

pub const B1_SAMPLE_STEPS: usize = 16;
pub const B8_SAMPLE_ROUNDS: usize = 8;
pub const RECHECK_AFTER_CONSUMED: usize = 512;
pub const MAX_DECISIONS: usize = 4;
pub const HYSTERESIS_PERCENT: u128 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lane { B1, B8 }
impl Lane { fn index(self) -> usize { match self { Self::B1 => 0, Self::B8 => 1 } } }

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Sample {
    pub transactions: usize,
    pub consumed_tokens: usize,
    pub wall_ns: u64,
}
impl Sample {
    fn add(self, consumed: usize, wall_ns: u64) -> Result<Self, &'static str> {
        Ok(Self {
            transactions: self.transactions.checked_add(1).ok_or("transaction overflow")?,
            consumed_tokens: self.consumed_tokens.checked_add(consumed).ok_or("token overflow")?,
            wall_ns: self.wall_ns.checked_add(wall_ns).ok_or("wall overflow")?,
        })
    }
    pub fn ns_per_consumed_token(self) -> Option<f64> {
        (self.consumed_tokens > 0).then(|| self.wall_ns as f64 / self.consumed_tokens as f64)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Decision {
    pub probe_start_consumed: usize,
    pub decision_consumed: usize,
    pub b1: Sample,
    pub b8: Sample,
    pub incumbent: Lane,
    pub selected: Lane,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase { SampleB1, SampleB8, Exploit { consumed: usize }, Hold }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Controller {
    origin: usize,
    cursor: usize,
    phase: Phase,
    selected: Lane,
    probe_start: usize,
    probe: [Sample; 2],
    total: [Sample; 2],
    decisions: [Option<Decision>; MAX_DECISIONS],
    decision_count: usize,
    forced_native: bool,
}
impl Controller {
    pub fn new(consumed: usize) -> Self {
        Self { origin: consumed, cursor: consumed, phase: Phase::SampleB1,
            selected: Lane::B1, probe_start: consumed, probe: [Sample::default(); 2],
            total: [Sample::default(); 2], decisions: [None; MAX_DECISIONS],
            decision_count: 0, forced_native: false }
    }
    pub fn next_lane(&self) -> Lane {
        match self.phase { Phase::SampleB1 => Lane::B1, Phase::SampleB8 => Lane::B8,
            Phase::Exploit { .. } | Phase::Hold => self.selected }
    }
    pub fn decisions(&self) -> &[Option<Decision>] { &self.decisions[..self.decision_count] }
    pub fn totals(&self) -> [Sample; 2] { self.total }
    pub fn consumed_since_start(&self) -> usize { self.cursor - self.origin }
    pub fn is_forced_native(&self) -> bool { self.forced_native }

    /// Capacity/vision/cancellation guards remain authoritative. A partial
    /// calibration never becomes a decision. The caller records an external
    /// reason in its receipt; this is not a cost-based disable decision.
    pub fn force_native(&mut self) {
        self.selected = Lane::B1;
        self.phase = Phase::Hold;
        self.forced_native = true;
    }

    /// Only complete intervals with actual consumed target tokens are admitted.
    /// Pending or EOS tokens are not a denominator. On an error, nothing changes.
    pub fn observe(&mut self, lane: Lane, start: usize, end: usize, wall: Duration)
        -> Result<(), &'static str>
    {
        if start != self.cursor { return Err("noncontiguous cursor; reset after restore"); }
        if lane != self.next_lane() { return Err("lane differs from requested policy"); }
        let consumed = end.checked_sub(start).ok_or("cursor went backwards")?;
        if consumed == 0 || consumed > 8 || (lane == Lane::B1 && consumed != 1) {
            return Err("invalid completed decode width");
        }
        let ns = u64::try_from(wall.as_nanos()).map_err(|_| "wall exceeds u64")?;
        if ns == 0 { return Err("zero or incomplete wall interval"); }
        let index = lane.index();
        let total = self.total[index].add(consumed, ns)?;
        let sampling = matches!(self.phase, Phase::SampleB1 | Phase::SampleB8);
        let probe = if sampling { self.probe[index].add(consumed, ns)? } else { self.probe[index] };
        let exploit_progress = match self.phase {
            Phase::Exploit { consumed: n } => n.checked_add(consumed).ok_or("progress overflow")?,
            _ => 0,
        };
        self.total[index] = total;
        if sampling { self.probe[index] = probe; }
        self.cursor = end;
        match self.phase {
            Phase::SampleB1 if probe.transactions == B1_SAMPLE_STEPS => self.phase = Phase::SampleB8,
            Phase::SampleB8 if probe.transactions == B8_SAMPLE_ROUNDS => {
                let b1 = self.probe[0];
                let b8 = self.probe[1];
                // Compare pooled nanoseconds/token exactly, without averaging TPS.
                // Maximum products fit u128: a u64 wall, <=64 tokens, and 102.
                let left = u128::from(b8.wall_ns) * b1.consumed_tokens as u128 * 100;
                let threshold = match self.selected {
                    Lane::B1 => 100 - HYSTERESIS_PERCENT,
                    Lane::B8 => 100 + HYSTERESIS_PERCENT,
                };
                let right = u128::from(b1.wall_ns) * b8.consumed_tokens as u128 * threshold;
                let selected = if left < right { Lane::B8 } else { Lane::B1 };
                self.decisions[self.decision_count] = Some(Decision {
                    probe_start_consumed: self.probe_start, decision_consumed: self.cursor,
                    b1, b8, incumbent: self.selected, selected,
                });
                self.decision_count += 1;
                self.selected = selected;
                self.phase = if self.decision_count == MAX_DECISIONS { Phase::Hold }
                    else { Phase::Exploit { consumed: 0 } };
            }
            Phase::Exploit { .. } => {
                if exploit_progress >= RECHECK_AFTER_CONSUMED {
                    self.phase = Phase::SampleB1;
                    self.probe_start = self.cursor;
                    self.probe = [Sample::default(); 2];
                } else { self.phase = Phase::Exploit { consumed: exploit_progress }; }
            }
            _ => {}
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn step(c: &mut Controller, width: usize, ns: u64) {
        let start = c.cursor;
        c.observe(c.next_lane(), start, start + width, Duration::from_nanos(ns)).unwrap();
    }
    fn probe(c: &mut Controller, b1_ns: u64, b8_ns: u64, b8_width: usize) {
        for _ in 0..B1_SAMPLE_STEPS { assert_eq!(c.next_lane(), Lane::B1); step(c, 1, b1_ns); }
        for _ in 0..B8_SAMPLE_ROUNDS { assert_eq!(c.next_lane(), Lane::B8); step(c, b8_width, b8_ns); }
    }
    fn exhaust_window(c: &mut Controller) {
        let lane = c.next_lane();
        for _ in 0..RECHECK_AFTER_CONSUMED { assert_eq!(c.next_lane(), lane); step(c, 1, 30); }
    }
    #[test]
    fn initially_samples_real_b1_then_real_b8() {
        let mut c = Controller::new(8192);
        for _ in 0..15 { step(&mut c, 1, 30); assert_eq!(c.next_lane(), Lane::B1); }
        step(&mut c, 1, 30);
        assert_eq!(c.next_lane(), Lane::B8);
        for _ in 0..7 { step(&mut c, 2, 40); assert_eq!(c.decisions().len(), 0); }
        step(&mut c, 2, 40);
        assert_eq!(c.next_lane(), Lane::B8);
        let d = c.decisions()[0].unwrap();
        assert_eq!(d.probe_start_consumed, 8192);
        assert_eq!(d.decision_consumed, 8224);
        assert_eq!(c.consumed_since_start(), 32);
        assert_eq!(d.b1.wall_ns, 480);
        assert_eq!(d.b8.wall_ns, 320);
    }
    #[test]
    fn high_acceptance_is_not_required_when_b8_is_cheaper() {
        let mut c = Controller::new(32768);
        probe(&mut c, 30, 50, 2);
        assert_eq!(c.next_lane(), Lane::B8);
    }
    #[test]
    fn high_acceptance_does_not_override_a_slower_b8() {
        let mut c = Controller::new(0);
        probe(&mut c, 30, 250, 8);
        assert_eq!(c.next_lane(), Lane::B1);
    }
    #[test]
    fn denominator_is_pooled_tokens_not_rounds_or_mean_tps() {
        let mut c = Controller::new(0);
        for _ in 0..16 { step(&mut c, 1, 30); }
        for i in 0..8 { step(&mut c, if i == 0 { 8 } else { 1 }, 50); }
        let d = c.decisions()[0].unwrap();
        assert_eq!(d.b8.consumed_tokens, 15);
        assert_eq!(d.b8.wall_ns, 400);
        assert_eq!(d.selected, Lane::B8); // 26.67 vs 30 ns/token
    }
    #[test]
    fn exact_two_percent_boundary_does_not_enter_b8() {
        let mut c = Controller::new(0);
        probe(&mut c, 100, 196, 2);
        assert_eq!(c.next_lane(), Lane::B1);
        let mut c = Controller::new(0);
        probe(&mut c, 100, 195, 2);
        assert_eq!(c.next_lane(), Lane::B8);
    }
    #[test]
    fn hysteresis_retains_b8_until_measured_regression() {
        let mut c = Controller::new(0);
        probe(&mut c, 100, 180, 2);
        exhaust_window(&mut c);
        probe(&mut c, 100, 201, 2);
        assert_eq!(c.next_lane(), Lane::B8);
        exhaust_window(&mut c);
        probe(&mut c, 100, 204, 2);
        assert_eq!(c.next_lane(), Lane::B1);
    }
    #[test]
    fn probes_are_finite_and_all_exploration_is_counted() {
        let mut c = Controller::new(8192);
        for i in 0..MAX_DECISIONS {
            probe(&mut c, 30, 40, 2);
            if i + 1 < MAX_DECISIONS { exhaust_window(&mut c); }
        }
        assert_eq!(c.decisions().len(), 4);
        for _ in 0..4096 { assert_eq!(c.next_lane(), Lane::B8); step(&mut c, 2, 40); }
        assert_eq!(c.decisions().len(), 4);
        let totals = c.totals();
        assert_eq!(totals[0].transactions, 64);
        assert_eq!(totals[1].transactions, 32 + 3*512 + 4096);
        assert_eq!(totals[0].consumed_tokens + totals[1].consumed_tokens, c.consumed_since_start());
    }
    #[test]
    fn changed_local_cost_can_restore_b8_after_native_selection() {
        let mut c = Controller::new(0);
        probe(&mut c, 30, 90, 2);
        assert_eq!(c.next_lane(), Lane::B1);
        exhaust_window(&mut c);
        probe(&mut c, 30, 40, 2);
        assert_eq!(c.next_lane(), Lane::B8);
    }
    #[test]
    fn errors_preserve_every_counter_and_decision() {
        let c = Controller::new(8192);
        for (lane, start, end, ns) in [
            (Lane::B1, 0, 1, 1), (Lane::B1, 8192, 8192, 1),
            (Lane::B1, 8192, 8194, 1), (Lane::B8, 8192, 8193, 1),
            (Lane::B1, 8192, 8191, 1), (Lane::B1, 8192, 8193, 0),
        ] {
            let mut candidate = c;
            assert!(candidate.observe(lane, start, end, Duration::from_nanos(ns)).is_err());
            assert_eq!(candidate, c);
        }
    }
    #[test]
    fn oversized_b8_and_duration_are_rejected() {
        let mut c = Controller::new(0);
        for _ in 0..16 { step(&mut c, 1, 1); }
        let before = c;
        assert!(c.observe(Lane::B8, 16, 25, Duration::from_nanos(1)).is_err());
        assert_eq!(c, before);
        assert!(c.observe(Lane::B8, 16, 17, Duration::MAX).is_err());
        assert_eq!(c, before);
    }
    #[test]
    fn sum_overflow_is_transactional() {
        let mut c = Controller::new(0);
        step(&mut c, 1, u64::MAX);
        let before = c;
        assert!(c.observe(Lane::B1, 1, 2, Duration::from_nanos(1)).is_err());
        assert_eq!(c, before);
    }
    #[test]
    fn guard_preempts_incomplete_probe_without_fabricating_a_decision() {
        let mut c = Controller::new(0);
        for _ in 0..16 { step(&mut c, 1, 30); }
        step(&mut c, 2, 40);
        c.force_native();
        for _ in 0..1024 { assert_eq!(c.next_lane(), Lane::B1); step(&mut c, 1, 30); }
        assert_eq!(c.decisions().len(), 0);
        assert!(c.is_forced_native());
        assert_eq!(c.totals()[1].transactions, 1);
    }
    #[test]
    fn restore_or_new_request_requires_fresh_controller() {
        let mut c = Controller::new(8192);
        probe(&mut c, 30, 40, 2);
        assert!(c.observe(Lane::B8, 8192, 8194, Duration::from_nanos(40)).is_err());
        c = Controller::new(8192);
        assert_eq!(c.next_lane(), Lane::B1);
        assert_eq!(c.consumed_since_start(), 0);
        assert_eq!(c.totals(), [Sample::default(); 2]);
        assert!(c.decisions().is_empty());
    }
    #[test]
    fn token_clock_rejects_an_unconsumed_pending_token() {
        let mut c = Controller::new(0);
        let before = c;
        assert!(c.observe(Lane::B1, 0, 0, Duration::from_nanos(30)).is_err());
        assert_eq!(before, c);
    }
}
