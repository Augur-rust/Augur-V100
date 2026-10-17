//! Cost-policy clock closes at the next existing synchronous token selection.
//! No GPU operation, synchronization, language label, or target lookahead here.
use super::cost_gate::{Controller, Lane, Sample};
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
struct Start { at: Instant, cursor: usize }
#[derive(Clone, Copy, Debug)]
struct Pending { start: Start, end: usize, lane: Lane }

#[derive(Default)]
pub(super) struct Runtime {
    control: Option<Controller>,
    deferred_start: bool,
    fixed_b8: bool,
    warmup: Sample,
    completed: [Sample; 2],
    first_probe_reason: Option<&'static str>,
    low_acceptance_trigger: Option<(usize, usize, usize)>,
    current: Option<Start>,
    pending: Option<Pending>,
    finished: bool,
    forced_native_reason: Option<&'static str>,
    start_cursor: Option<usize>,
    terminal_cursor: Option<usize>,
    terminal_lane: Option<Lane>,
    terminal_transactions: usize,
    terminal_consumed_tokens: usize,
    terminal_wall_ns: u64,
    terminal_selection_only_ns: u64,
}
impl Runtime {
    /// Account for real decode intervals without creating a sampling controller.
    pub(super) fn fixed() -> Self { Self {fixed_b8:true,..Self::default()} }
    pub(super) fn fixed_b8(&self) -> bool {self.fixed_b8}
    pub(super) fn deferred() -> Self { Self {deferred_start:true,..Self::default()} }
    pub(super) fn reset(&mut self) {
        *self = if self.fixed_b8 {Self::fixed()} else if self.deferred_start {Self::deferred()} else {Self::default()};
    }
    pub(super) fn deferred_start(&self) -> bool {self.deferred_start}
    pub(super) fn warmup(&self) -> Sample {self.warmup}
    pub(super) fn first_probe_reason(&self) -> Option<&'static str> {self.first_probe_reason}
    pub(super) fn low_acceptance_trigger(&self) -> Option<(usize,usize,usize)> {self.low_acceptance_trigger}
    pub(super) fn consider_legacy_fallback(&mut self,rounds:usize,accepted:usize,checked:usize) -> Result<(), &'static str> {
        if self.deferred_start && self.control.is_none() && self.forced_native_reason.is_none()
            && rounds>=8 && checked>0 && (accepted as f64)/(checked as f64)<0.8 {
            let current=self.current.ok_or("deferred probe requires synchronized selection")?;
            self.control=Some(Controller::new(current.cursor));
            self.first_probe_reason=Some("legacy_low_acceptance");
            self.low_acceptance_trigger=Some((rounds,accepted,checked));
        }
        Ok(())
    }
    pub(super) fn control(&self) -> Option<&Controller> { self.control.as_ref() }
    pub(super) fn reason(&self) -> Option<&'static str> { self.forced_native_reason }
    pub(super) fn start_cursor(&self) -> Option<usize> { self.start_cursor }
    pub(super) fn terminal_cursor(&self) -> Option<usize> { self.terminal_cursor }
    pub(super) fn terminal_lane(&self) -> Option<Lane> { self.terminal_lane }
    pub(super) fn finished(&self) -> bool { self.finished }
    pub(super) fn has_open_interval(&self) -> bool { self.current.is_some() || self.pending.is_some() }
    pub(super) fn terminal(&self) -> (usize, usize, u64, u64) {
        (self.terminal_transactions, self.terminal_consumed_tokens,
            self.terminal_wall_ns, self.terminal_selection_only_ns)
    }
    pub(super) fn force_native(&mut self, reason: &'static str) {
        self.forced_native_reason = Some(reason);
        if let Some(control) = &mut self.control { control.force_native(); }
    }
    pub(super) fn next_lane(&self) -> Result<Lane, &'static str> {
        if self.finished || self.current.is_none() || self.pending.is_some() {
            return Err("cost gate needs a current synchronized selection");
        }
        if self.forced_native_reason.is_some() {return Ok(Lane::B1);}
        if self.fixed_b8 || (self.deferred_start && self.control.is_none()) {return Ok(Lane::B8);}
        self.control.as_ref().map(Controller::next_lane).ok_or("cost gate not initialized")
    }
    /// Called after greedy_next_token has already synchronized its own stream,
    /// and before Session publishes the selected anchor. The same boundary is
    /// used for both lanes; publication and setup remain inside the interval.
    pub(super) fn selected(&mut self, cursor: usize, at: Instant) -> Result<(), &'static str> {
        if self.finished || self.current.is_some() {
            return Err("cost gate selection overlaps an unconsumed or terminal selection");
        }
        if let Some(pending) = self.pending {
            if cursor != pending.end { return Err("cost gate synchronized cursor mismatch"); }
            let wall = at.checked_duration_since(pending.start.at).ok_or("cost gate clock reversed")?;
            let ns=u64::try_from(wall.as_nanos()).map_err(|_|"cost clock overflow")?;
            if ns==0 {return Err("zero or incomplete cost interval");}
            let width=pending.end-pending.start.cursor;
            let index=if pending.lane==Lane::B1 {0}else{1};
            let total=add_sample(self.completed[index],width,ns)?;
            if let Some(control)=&mut self.control {
                control.observe(pending.lane,pending.start.cursor,pending.end,wall)?;
            } else if self.fixed_b8 {
                if pending.lane==Lane::B1 && self.forced_native_reason.is_none() {return Err("unguarded B1 in fixed B8");}
            } else if self.deferred_start {
                if pending.lane==Lane::B8 {
                    self.warmup=add_sample(self.warmup,width,ns)?;
                    if self.warmup.consumed_tokens>=512 && self.forced_native_reason.is_none() {
                        self.control=Some(Controller::new(cursor));
                        self.first_probe_reason=Some("initial_512_consumed_tokens");
                    }
                } else if self.forced_native_reason.is_none() {return Err("unrequested deferred B1");}
            } else {return Err("cost gate missing controller");}
            self.completed[index]=total;
            self.pending = None;
        } else if self.control.is_some() {
            return Err("cost gate missing preceding transaction");
        } else {
            if self.start_cursor.is_some() {return Err("deferred gate missing preceding transaction");}
            if !self.deferred_start && !self.fixed_b8 {
                let mut control=Controller::new(cursor);
                if self.forced_native_reason.is_some() {control.force_native();}
                self.control=Some(control);
                self.first_probe_reason=Some("request_start");
            }
            self.start_cursor = Some(cursor);
        }
        self.current = Some(Start { cursor, at });
        Ok(())
    }
    /// Called only after the actual target step succeeded. This deliberately
    /// records no duration: a B1 enqueue return does not establish completion.
    pub(super) fn enqueued(&mut self, lane: Lane, end: usize) -> Result<(), &'static str> {
        let start = self.current.ok_or("cost gate enqueue without selection")?;
        if self.pending.is_some() || self.finished || self.next_lane()? != lane {
            return Err("cost gate enqueue lane/order mismatch");
        }
        let count = end.checked_sub(start.cursor).ok_or("cost gate cursor reversed")?;
        if count == 0 || count > 8 || (lane == Lane::B1 && count != 1) {
            return Err("cost gate enqueue has invalid consumed extent");
        }
        self.pending = Some(Pending { start, end, lane });
        self.current = None;
        Ok(())
    }
    /// The existing terminal drain closes remaining work. A length-tail block
    /// has no next selection, so it is billed separately and never used to
    /// select a lane. An unconsumed final selection is not a generated token
    /// denominator (it may be EOS, a published length tail, or cancellation).
    pub(super) fn drained(&mut self, cursor: usize, at: Instant) -> Result<(), &'static str> {
        if self.finished { return Ok(()); }
        if let Some(pending) = self.pending {
            if cursor != pending.end { return Err("cost gate terminal cursor mismatch"); }
            let wall = at.checked_duration_since(pending.start.at).ok_or("cost gate clock reversed")?;
            let ns = u64::try_from(wall.as_nanos()).map_err(|_| "terminal clock overflow")?;
            self.terminal_lane = Some(pending.lane);
            self.terminal_transactions = 1;
            self.terminal_consumed_tokens = pending.end - pending.start.cursor;
            self.terminal_wall_ns = ns;
        } else if let Some(start) = self.current {
            if cursor != start.cursor { return Err("cost gate unconsumed terminal cursor mismatch"); }
            let wall = at.checked_duration_since(start.at).ok_or("cost gate clock reversed")?;
            self.terminal_selection_only_ns = u64::try_from(wall.as_nanos()).map_err(|_| "terminal clock overflow")?;
        }
        self.current = None;
        self.pending = None;
        self.terminal_cursor = Some(cursor);
        self.finished = true;
        Ok(())
    }
    pub(super) fn totals(&self) -> [Sample; 2] {
        self.completed
    }
}


fn add_sample(old: Sample, tokens: usize, ns: u64) -> Result<Sample, &'static str> {
    Ok(Sample { transactions: old.transactions.checked_add(1).ok_or("sample round overflow")?,
        consumed_tokens: old.consumed_tokens.checked_add(tokens).ok_or("sample token overflow")?,
        wall_ns: old.wall_ns.checked_add(ns).ok_or("sample wall overflow")? })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    #[test]
    fn fixed_b8_ignores_low_acceptance_time_noise_and_sampling_cadence() {
        let t=Instant::now();let mut r=Runtime::fixed();let mut cursor=8192;let mut elapsed=0;
        r.selected(cursor,t).unwrap();
        // Beyond several 512-token boundaries, including zero accepted drafts.
        for n in 0..600 {
            r.consider_legacy_fallback(n,0,n*7).unwrap();
            assert_eq!(r.next_lane().unwrap(),Lane::B8);
            cursor+=if n%2==0 {1}else{8};
            r.enqueued(Lane::B8,cursor).unwrap();
            elapsed+=if n%2==0 {300}else{20};
            r.selected(cursor,t+Duration::from_millis(elapsed)).unwrap();
            assert!(r.control().is_none());
        }
        assert_eq!(r.totals()[0],Sample::default());
        assert_eq!(r.totals()[1].transactions,600);
        assert_eq!(r.totals()[1].consumed_tokens,cursor-8192);
        assert_eq!(r.totals()[1].wall_ns,elapsed*1_000_000);
        assert!(r.first_probe_reason().is_none());
        assert!(r.low_acceptance_trigger().is_none());
        r.drained(cursor,t+Duration::from_millis(elapsed+1)).unwrap();
        assert!(r.finished());
    }
    #[test]
    fn fixed_b8_retains_guarded_native_and_clears_guard_on_reset() {
        for reason in ["vision","capacity"] {
            let t=Instant::now();let mut r=Runtime::fixed();
            r.selected(8192,t).unwrap();
            assert!(r.enqueued(Lane::B1,8193).is_err());
            r.force_native(reason);
            assert_eq!(r.next_lane().unwrap(),Lane::B1);
            r.enqueued(Lane::B1,8193).unwrap();
            r.selected(8193,t+Duration::from_millis(30)).unwrap();
            assert_eq!(r.totals()[0].consumed_tokens,1);
            r.drained(8193,t+Duration::from_millis(31)).unwrap();
            r.reset();r.selected(100,t).unwrap();
            assert!(r.fixed_b8());assert!(r.reason().is_none());
            assert_eq!(r.next_lane().unwrap(),Lane::B8);
            assert_eq!(r.totals(),[Sample::default();2]);
        }
    }
    #[test]
    fn enqueued_b1_is_not_timed_until_the_next_sync() {
        let t = Instant::now();
        let mut r = Runtime::default();
        r.selected(8192, t).unwrap();
        r.enqueued(Lane::B1, 8193).unwrap();
        assert_eq!(r.totals()[0].transactions, 0);
        r.selected(8193, t + Duration::from_millis(30)).unwrap();
        assert_eq!(r.totals()[0].wall_ns, 30_000_000);
        assert_eq!(r.totals()[0].consumed_tokens, 1);
    }
    #[test]
    fn both_lanes_charge_publication_setup_and_next_selection() {
        let t = Instant::now();
        let mut r = Runtime::default();
        r.selected(8192, t).unwrap();
        for n in 0..16 {
            r.enqueued(Lane::B1, 8193+n).unwrap();
            r.selected(8193+n, t + Duration::from_millis(30*(n as u64+1))).unwrap();
        }
        assert_eq!(r.next_lane().unwrap(), Lane::B8);
        r.enqueued(Lane::B8, 8211).unwrap();
        // 50ms covers the complete interval, not a 5ms host enqueue call.
        r.selected(8211, t + Duration::from_millis(530)).unwrap();
        assert_eq!(r.totals()[0].wall_ns, 480_000_000);
        assert_eq!(r.totals()[1].wall_ns, 50_000_000);
        assert_eq!(r.totals()[1].consumed_tokens, 3);
    }
    #[test]
    fn terminal_drain_cannot_finish_a_probe_or_count_pending_token() {
        let t = Instant::now();
        let mut r = Runtime::default();
        r.selected(8192, t).unwrap();
        r.enqueued(Lane::B1, 8193).unwrap();
        r.drained(8193, t + Duration::from_millis(30)).unwrap();
        assert_eq!(r.totals(), [Sample::default(); 2]);
        assert_eq!(r.terminal(), (1, 1, 30_000_000, 0));
        assert_eq!(r.start_cursor(), Some(8192));
        assert_eq!(r.terminal_cursor(), Some(8193));
        assert_eq!(r.terminal_lane(), Some(Lane::B1));
        assert!(r.control().unwrap().decisions().is_empty());
        assert!(!r.has_open_interval());
        assert!(r.finished());
    }
    #[test]
    fn single_output_or_eos_does_not_fabricate_decode_cost() {
        let t = Instant::now();
        let mut r = Runtime::default();
        r.selected(8192, t).unwrap();
        r.drained(8192, t + Duration::from_millis(2)).unwrap();
        assert_eq!(r.totals(), [Sample::default(); 2]);
        assert_eq!(r.terminal(), (0, 0, 0, 2_000_000));
        r.drained(8192, t + Duration::from_secs(4)).unwrap();
        assert_eq!(r.terminal(), (0, 0, 0, 2_000_000));
    }
    #[test]
    fn reset_clears_old_clocks_costs_and_guard_reason() {
        let t = Instant::now();
        let mut r = Runtime::default();
        r.selected(8192, t).unwrap(); r.enqueued(Lane::B1, 8193).unwrap();
        r.selected(8193, t + Duration::from_millis(30)).unwrap();
        r.force_native("vision"); r.reset();
        r.selected(32768, t + Duration::from_secs(2)).unwrap();
        assert_eq!(r.totals(), [Sample::default(); 2]);
        assert!(r.reason().is_none());
        assert!(!r.finished());
        assert_eq!(r.next_lane().unwrap(), Lane::B1);
    }
    #[test]
    fn guard_before_initialization_suppresses_all_speculation() {
        let t = Instant::now();
        let mut r = Runtime::default(); r.force_native("vision");
        r.selected(8192, t).unwrap();
        for n in 0..32 {
            r.enqueued(Lane::B1, 8193+n).unwrap();
            r.selected(8193+n, t + Duration::from_millis(n as u64+1)).unwrap();
            assert_eq!(r.next_lane().unwrap(), Lane::B1);
        }
        assert_eq!(r.reason(), Some("vision"));
        assert!(r.control().unwrap().is_forced_native());
        assert!(r.control().unwrap().decisions().is_empty());
    }
    #[test]
    fn protocol_errors_do_not_advance_the_measurement() {
        let t = Instant::now();
        let mut r = Runtime::default();
        assert!(r.enqueued(Lane::B1, 1).is_err());
        r.selected(8192, t).unwrap();
        assert!(r.selected(8192, t).is_err());
        assert!(r.enqueued(Lane::B8, 8193).is_err());
        assert!(r.enqueued(Lane::B1, 8192).is_err());
        r.enqueued(Lane::B1, 8193).unwrap();
        assert!(r.selected(8194, t+Duration::from_millis(1)).is_err());
        assert_eq!(r.totals(), [Sample::default(); 2]);
        r.selected(8193, t+Duration::from_millis(30)).unwrap();
        assert_eq!(r.totals()[0].transactions, 1);
    }
    #[test]
    fn no_selection_after_terminal_without_request_reset() {
        let t = Instant::now();
        let mut r = Runtime::default();
        r.drained(8192, t).unwrap();
        assert!(r.selected(8192, t).is_err());
        r.reset(); r.selected(8192, t).unwrap();
    }
    #[test]
    fn deferred_high_acceptance_short_request_does_not_sample_native() {
        let t=Instant::now();let mut r=Runtime::deferred();r.selected(100,t).unwrap();
        for n in 0..19 {
            assert_eq!(r.next_lane().unwrap(),Lane::B8);
            r.enqueued(Lane::B8,100+(n+1)*7).unwrap();
            r.selected(100+(n+1)*7,t+Duration::from_millis(60*(n as u64+1))).unwrap();
            r.consider_legacy_fallback(n+1,(n+1)*6,(n+1)*7).unwrap();
        }
        r.drained(233,t+Duration::from_millis(1141)).unwrap();
        assert_eq!(r.totals()[0].transactions,0);
        assert_eq!(r.totals()[1].transactions,19);
        assert_eq!(r.warmup().consumed_tokens,133);
        assert!(r.control().is_none());assert!(r.first_probe_reason().is_none());
    }
    #[test]
    fn deferred_legacy_fallback_triggers_real_cost_sampling_instead_of_disable() {
        let t=Instant::now();let mut r=Runtime::deferred();r.selected(100,t).unwrap();
        for n in 0..8 {
            r.enqueued(Lane::B8,102+2*n).unwrap();
            r.selected(102+2*n,t+Duration::from_millis(60*(n as u64+1))).unwrap();
            r.consider_legacy_fallback(n+1,n+1,2*(n+1)).unwrap();
        }
        assert_eq!(r.next_lane().unwrap(),Lane::B1);
        assert_eq!(r.first_probe_reason(),Some("legacy_low_acceptance"));
        assert_eq!(r.low_acceptance_trigger(),Some((8,8,16)));
        assert_eq!(r.warmup().consumed_tokens,16);
        assert_eq!(r.control().unwrap().consumed_since_start(),0);
        r.enqueued(Lane::B1,117).unwrap();
        r.selected(117,t+Duration::from_millis(510)).unwrap();
        assert_eq!(r.totals()[0].consumed_tokens,1);
        assert_eq!(r.totals()[1].consumed_tokens,16);
        assert_eq!(r.control().unwrap().consumed_since_start(),1);
    }
    #[test]
    fn deferred_long_high_acceptance_still_observes_actual_native_cost() {
        let t=Instant::now();let mut r=Runtime::deferred();r.selected(100,t).unwrap();
        for n in 0..64 {
            r.enqueued(Lane::B8,108+8*n).unwrap();
            r.selected(108+8*n,t+Duration::from_millis(60*(n as u64+1))).unwrap();
        }
        assert_eq!(r.next_lane().unwrap(),Lane::B1);
        assert_eq!(r.first_probe_reason(),Some("initial_512_consumed_tokens"));
        assert_eq!(r.warmup().consumed_tokens,512);
        assert_eq!(r.control().unwrap().consumed_since_start(),0);
        assert!(r.low_acceptance_trigger().is_none());
    }
    #[test]
    fn deferred_reset_preserves_policy_but_not_previous_probes() {
        let t=Instant::now();let mut r=Runtime::deferred();r.selected(100,t).unwrap();
        r.enqueued(Lane::B8,102).unwrap();r.selected(102,t+Duration::from_millis(60)).unwrap();
        r.consider_legacy_fallback(8,7,10).unwrap();r.reset();r.selected(8192,t).unwrap();
        assert!(r.deferred_start());assert_eq!(r.next_lane().unwrap(),Lane::B8);
        assert_eq!(r.warmup(),Sample::default());assert!(r.control().is_none());
        assert!(r.first_probe_reason().is_none());assert!(r.low_acceptance_trigger().is_none());
    }
    #[test]
    fn deferred_vision_guard_prevents_warmup_and_cost_sampling() {
        let t=Instant::now();let mut r=Runtime::deferred();r.force_native("vision");r.selected(100,t).unwrap();
        for n in 0..32 {
            assert_eq!(r.next_lane().unwrap(),Lane::B1);r.enqueued(Lane::B1,101+n).unwrap();
            r.selected(101+n,t+Duration::from_millis(30*(n as u64+1))).unwrap();
            r.consider_legacy_fallback(8,0,8).unwrap();
        }
        assert!(r.control().is_none());assert_eq!(r.warmup(),Sample::default());
        assert_eq!(r.totals()[0].transactions,32);
    }
    #[test]
    fn deferred_terminal_block_is_separate_from_warmup_and_probe_counts() {
        let t=Instant::now();let mut r=Runtime::deferred();r.selected(100,t).unwrap();
        r.enqueued(Lane::B8,103).unwrap();r.drained(103,t+Duration::from_millis(60)).unwrap();
        assert_eq!(r.totals(),[Sample::default();2]);assert_eq!(r.warmup(),Sample::default());
        assert_eq!(r.terminal(),(1,3,60_000_000,0));assert_eq!(r.terminal_lane(),Some(Lane::B8));
        assert!(r.control().is_none());
    }

}
