//! Explicit initialized-history destination, never a verifier publication target.
use super::*;

pub(super) const WIDTH: usize = 512;
pub(super) const HISTORY: usize = 2048;

// Scheduling is independent of feature capture. Prefix and this suffix meet
// on a global512 boundary; the final block contains only its real rows.
pub(super) fn exact_input_schedule(end: usize, history: usize) -> Result<(usize, usize)> {
    if end == 0 || history != end.min(HISTORY) {
        return Err(invalid("exact input history must be min(N,2048)"));
    }
    let start = end - history;
    Ok((start / WIDTH * WIDTH, start))
}

fn exact_feature_storage(active_rows: usize, capacity_rows: usize, bytes: usize) -> Result<()> {
    // A growing product history reserves2048 rows even for a16-token prompt.
    // Only active rows are captured/consumed; unused capacity is not padding.
    if active_rows == 0 || active_rows > capacity_rows || capacity_rows > HISTORY
        || bytes != capacity_rows * 5 * H * 4 {
        return Err(invalid("exact history active/physical feature extent"));
    }
    Ok(())
}

fn exact_feature_span(cursor: usize, end: usize, rows: usize, history: usize)
    -> Result<(usize, usize, usize)> {
    let (prefix, start) = exact_input_schedule(end, history)?;
    let remaining = end.checked_sub(cursor).ok_or_else(|| invalid("exact input cursor after end"))?;
    if cursor < prefix || cursor % WIDTH != 0 || remaining == 0 || rows != remaining.min(WIDTH) {
        return Err(invalid("exact input block extent/alignment"));
    }
    let block_end = cursor.checked_add(rows).ok_or_else(|| invalid("exact input block overflow"))?;
    let first = cursor.max(start);
    if first >= block_end { return Err(invalid("empty exact history intersection")); }
    Ok((first - cursor, first - start, block_end - first))
}

// Boundary-only isolation, not general remainder or scalar-kernel parity.
// Keep the existing scalar arithmetic, but move its single real token to the
// actual end of the input. Never silently decompose an arbitrary DLL remainder.
pub(super) fn global_boundary_schedule(end: usize) -> Result<(usize, usize, usize)> {
    let start = end.checked_sub(HISTORY).ok_or_else(|| invalid("global history underflow"))?;
    let tail = end % WIDTH;
    if tail > 1 {
        return Err(invalid("global boundary audit supports only N%512=0/1; arbitrary remainder unqualified"));
    }
    Ok((start / WIDTH * WIDTH, start, tail))
}

fn global_feature_span(cursor: usize, end: usize, rows: usize, history: usize)
    -> Result<(usize, usize, usize)> {
    let (prefix, start, _) = global_boundary_schedule(end)?;
    let block_end = cursor.checked_add(rows).ok_or_else(|| invalid("global block overflow"))?;
    if history != HISTORY || rows != WIDTH || cursor % WIDTH != 0
        || cursor < prefix || block_end > end || cursor >= end - end % WIDTH {
        return Err(invalid("global history intersection extent"));
    }
    let first = cursor.max(start);
    if first >= block_end { return Err(invalid("empty global history intersection")); }
    Ok((first - cursor, first - start, block_end - first))
}

pub(in super::super) fn requested(
    value: Option<&std::ffi::OsStr>,
    ffn_rows: Option<usize>,
    input_integer: bool,
    history_rows: usize,
) -> Result<bool> {
    match value {
        None => Ok(false),
        Some(v)
            if v == "1" && ffn_rows == Some(WIDTH) && input_integer && history_rows == HISTORY =>
        {
            Ok(true)
        }
        // Product-candidate dispatch: never pad a short/growing history and
        // never widen its B8 verifier. Invalid configuration still rejects.
        Some(v)
            if v == "auto"
                && ffn_rows == Some(WIDTH)
                && input_integer
                && (1..=HISTORY).contains(&history_rows) =>
        {
            Ok(history_rows == HISTORY)
        }
        _ => Err(invalid(
            "AUGUR_QWEN35_HISTORY_WIDE_CAPTURE requires 1 with2048 history or auto with1..2048 history, and B512 FFN/input",
        )),
    }
}

fn feature_start(cursor: usize, end: usize, rows: usize, history: usize) -> Result<usize> {
    let start = end
        .checked_sub(HISTORY)
        .ok_or_else(|| invalid("history underflow"))?;
    let offset = cursor
        .checked_sub(start)
        .ok_or_else(|| invalid("history cursor before start"))?;
    if history != HISTORY
        || rows != WIDTH
        || offset % WIDTH != 0
        || cursor.checked_add(rows).is_none_or(|value| value > end)
    {
        return Err(invalid("history block extent/alignment"));
    }
    Ok(offset)
}

/// Borrowed only for one same-stream block. This is not a new state/cursor owner.
pub(super) struct Destination<'a> {
    driver: &'a Driver,
    cursor: usize,
    end: usize,
    offset: usize,
    source_offset: usize,
    captured_rows: usize,
    global: bool,
    exact: bool,
    input_rows: usize,
}

impl<'a> Destination<'a> {
    pub(super) fn bind(d: &'a Driver, end: usize, rows: usize) -> Result<Self> {
        let offset = feature_start(d.cursor, end, rows, d.context_rows)?;
        if !d.wide_history_capture
            || !d.poisoned
            || d.integer_mma_ffn != Some(WIDTH)
            || !d.integer_mma_input
            || d.features.len_bytes() != HISTORY * 5 * H * 4
        {
            return Err(invalid("history capture owner/policy/feature extent"));
        }
        Ok(Self {
            driver: d,
            cursor: d.cursor,
            end,
            offset,
            source_offset: 0,
            captured_rows: WIDTH,
            global: false,
            exact: false,
            input_rows: rows,
        })
    }

    pub(super) fn bind_global(d: &'a Driver, end: usize, rows: usize) -> Result<Self> {
        let (source_offset, offset, captured_rows) = global_feature_span(d.cursor, end, rows, d.context_rows)?;
        if !d.llama_global_chunks || !d.llama_history_full || !d.wide_history_capture
            || !d.poisoned || d.integer_mma_ffn != Some(WIDTH) || !d.integer_mma_input
            || d.features.len_bytes() != HISTORY * 5 * H * 4 {
            return Err(invalid("global history owner/policy/feature extent"));
        }
        Ok(Self { driver: d, cursor: d.cursor, end, offset, source_offset, captured_rows,
            global: true, exact: false, input_rows: rows })
    }

    pub(super) fn bind_exact_input(d: &'a Driver, end: usize, rows: usize) -> Result<Self> {
        let (source_offset, offset, captured_rows) = exact_feature_span(d.cursor, end, rows, d.context_rows)?;
        if !d.llama_exact_input_rows || !d.llama_history_full || !d.poisoned {
            return Err(invalid("exact history owner/policy"));
        }
        exact_feature_storage(d.context_rows,
            d.growing_history.map_or(d.context_rows, GrowingHistory::capacity_rows), d.features.len_bytes())?;
        Ok(Self { driver: d, cursor: d.cursor, end, offset, source_offset, captured_rows,
            global: false, exact: true, input_rows: rows })
    }

    pub(super) fn validate(&self, d: &Driver, rows: usize, end: usize) -> Result<()> {
        let span = if self.exact {
            if !d.llama_exact_input_rows || !d.llama_history_full {
                return Err(invalid("exact history policy drift"));
            }
            exact_feature_span(d.cursor, end, rows, d.context_rows)?
        } else if self.global {
            if !d.llama_global_chunks || !d.llama_history_full {
                return Err(invalid("global history policy drift"));
            }
            global_feature_span(d.cursor, end, rows, d.context_rows)?
        } else {
            (0, feature_start(d.cursor, end, rows, d.context_rows)?, WIDTH)
        };
        if !core::ptr::eq(self.driver, d)
            || self.cursor != d.cursor
            || self.end != end
            || self.input_rows != rows
            || span != (self.source_offset, self.offset, self.captured_rows)
        {
            return Err(invalid("history capture destination binding drift"));
        }
        Ok(())
    }

    pub(super) fn capture_layer(&self, w: &PrefillWorkspace, layer: usize) -> Result<()> {
        let Some(slot) = DFLASH_TARGET_CAPTURE_LAYER_IDS
            .iter()
            .position(|&id| id == layer)
        else {
            return Ok(());
        };
        let source = w.hidden_output(layer, self.input_rows)?.buffer_view();
        if self.driver.history_batched_feature_copies {
            // SAFETY: the exact history intersection is bound to this driver;
            // source and strided destination are disjoint retained allocations.
            // CaptureOwner drains this stream before either owner can be dropped.
            unsafe { self.driver.features.copy_from_device_2d_async_unchecked(
                &self.driver.stream, (self.offset * 5 + slot) * H * 4, 5 * H * 4,
                source.subview(self.source_offset * H * 4, self.captured_rows * H * 4)?,
                H * 4, H * 4, self.captured_rows,
            )?; }
            self.driver.history_feature_copy_enqueues.set(
                self.driver.history_feature_copy_enqueues.get() + 1);
            return Ok(());
        }
        // Reuse the existing checked same-stream copy helper. The temporary
        // CaptureOwner drains before releasing w or any Driver allocation.
        for row in 0..self.captured_rows {
            copy(
                &self.driver.stream,
                &self.driver.features,
                ((self.offset + row) * 5 + slot) * H * 4,
                source.subview((self.source_offset + row) * H * 4, H * 4)?,
            )?;
            self.driver.history_feature_copy_enqueues.set(
                self.driver.history_feature_copy_enqueues.get() + 1);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsStr;

    #[test]
    fn exact_short_history_checks_active_extent_separately_from_reserved_capacity() {
        for active in 1..=HISTORY {
            assert!(exact_feature_storage(active, HISTORY, HISTORY * 5 * H * 4).is_ok());
            assert!(exact_feature_storage(active, active, active * 5 * H * 4).is_ok());
        }
        for (active, capacity, bytes) in [(0,HISTORY,HISTORY*5*H*4),
            (HISTORY+1,HISTORY,HISTORY*5*H*4), (16,15,15*5*H*4),
            (16,HISTORY,16*5*H*4), (16,HISTORY,HISTORY*5*H*4+4),
            (16,HISTORY+1,(HISTORY+1)*5*H*4), (1,usize::MAX,0)] {
            assert!(exact_feature_storage(active,capacity,bytes).is_err());
        }
    }
    #[test]
    fn exact_input_history_covers_every_remainder_once_without_padding() {
        for base in [0, 512, 2048, 8192, 32768, 65536, 195584] {
            for tail in 0..WIDTH {
                let end = base + tail;
                if end == 0 { continue; }
                let history = end.min(HISTORY);
                let (prefix, start) = exact_input_schedule(end, history).unwrap();
                assert_eq!(prefix % WIDTH, 0);
                assert_eq!(start + history, end);
                let mut captured = Vec::new();
                let mut executed = prefix;
                for cursor in (prefix..end).step_by(WIDTH) {
                    let rows = (end - cursor).min(WIDTH);
                    let (src, dst, count) = exact_feature_span(cursor, end, rows, history).unwrap();
                    assert!(src + count <= rows);
                    assert_eq!(cursor + src, start + dst);
                    captured.extend(dst..dst + count);
                    executed += rows;
                }
                assert_eq!(executed, end);
                assert_eq!(captured, (0..history).collect::<Vec<_>>());
            }
        }
    }
    #[test]
    fn exact_input_span_rejects_bad_history_cursor_and_padded_tail() {
        for (n,h) in [(0,0),(8,7),(2048,2047),(2049,2049)] {
            assert!(exact_input_schedule(n,h).is_err());
        }
        let end=8192+285;let history=HISTORY;
        assert!(exact_feature_span(8192,end,285,history).is_ok());
        for (cursor,rows) in [(8192,0),(8192,284),(8192,286),(8192,512),
            (8193,284),(0,512),(usize::MAX,512)] {
            assert!(exact_feature_span(cursor,end,rows,history).is_err());
        }
        assert!(exact_feature_span(0,8,8,8).is_ok());
        assert!(exact_feature_span(0,1,1,1).is_ok());
    }
    #[test]
    fn global_history_intersections_cover_exact_real_tail_once() {
        for end in [2048, 2049, 8192, 8193, 32768, 32769, 65536, 196097] {
            let (prefix, start, tail) = global_boundary_schedule(end).unwrap();
            assert_eq!(prefix % WIDTH, 0);
            assert_eq!(prefix + HISTORY + tail, end);
            let mut captured = Vec::new();
            for cursor in (prefix..end-tail).step_by(WIDTH) {
                let (source, offset, count) = global_feature_span(cursor, end, WIDTH, HISTORY).unwrap();
                assert_eq!(cursor + source, start + offset);
                assert_eq!(offset, captured.len());
                captured.extend(cursor + source..cursor + source + count);
            }
            captured.extend(end-tail..end);
            assert_eq!(captured, (start..end).collect::<Vec<_>>());
        }
    }
    #[test]
    fn global_boundary_never_silently_splits_or_pads_other_remainders() {
        for end in [0, 1, 2047, 2050, 8477, usize::MAX] {
            assert!(global_boundary_schedule(end).is_err());
        }
        for (cursor, end, rows, history) in [(30721,32769,512,2048), (30208,32769,512,2048),
            (32768,32769,512,2048), (30720,32769,1,2048), (30720,32769,512,2047),
            (usize::MAX,usize::MAX,512,2048)] {
            assert!(global_feature_span(cursor,end,rows,history).is_err());
        }
        // Legacy destination remains strict, even though the audit can capture a partial block.
        assert!(feature_start(30720,32769,512,2048).is_err());
        assert_eq!(global_feature_span(30720,32769,512,2048).unwrap(), (1,0,511));
    }
    #[test]
    fn resident_recipe_binds_real_prompt_history_not_profile_capacity() {
        for prompt_tokens in [1, 7, 8, 33, 97, 511, 2047, 2048, 12242] {
            let (start, initialized) = prompt_capture_extent(prompt_tokens, HISTORY).unwrap();
            assert_eq!(start + initialized, prompt_tokens);
            assert_eq!(
                requested(Some(OsStr::new("auto")), Some(WIDTH), true, initialized).unwrap(),
                prompt_tokens >= HISTORY
            );
            if prompt_tokens < HISTORY {
                assert!(requested(Some(OsStr::new("1")), Some(WIDTH), true, initialized).is_err());
            }
        }
    }
    #[test]
    fn wide_history_switch_preserves_default_and_verifier_boundaries() {
        assert!(!requested(None, None, false, 8).unwrap());
        assert!(requested(Some(OsStr::new("1")), Some(512), true, 2048).unwrap());
        for (rows, input, history) in [
            (None, true, 2048),
            (Some(256), true, 2048),
            (Some(512), false, 2048),
            (Some(512), true, 2047),
            (Some(512), true, 2049),
        ] {
            assert!(requested(Some(OsStr::new("1")), rows, input, history).is_err());
        }
        for value in ["", "0", "true", " 1", "1 "] {
            assert!(requested(Some(OsStr::new(value)), Some(512), true, 2048).is_err());
        }
    }
    #[test]
    fn wide_history_blocks_cover_only_the_real_initialized_tail() {
        for end in [2048, 4096, 12242, 196608] {
            for i in 0..4 {
                assert_eq!(
                    feature_start(end - 2048 + i * 512, end, 512, 2048).unwrap(),
                    i * 512
                );
            }
            for cursor in [end - 2047, end - 1, end] {
                assert!(feature_start(cursor, end, 512, 2048).is_err());
            }
            for rows in [0, 8, 128, 256, 513, 2048] {
                assert!(feature_start(end - 2048, end, rows, 2048).is_err());
            }
        }
        assert!(feature_start(0, 2047, 512, 2048).is_err());
        assert!(feature_start(0, 4096, 512, 2048).is_err());
        assert!(feature_start(usize::MAX, usize::MAX, 512, 2048).is_err());
    }

    #[test]
    fn automatic_history_dispatch_never_pads_short_or_growing_input() {
        let auto = Some(OsStr::new("auto"));
        for history in 1..=HISTORY {
            assert_eq!(
                requested(auto, Some(WIDTH), true, history).unwrap(),
                history == HISTORY
            );
        }
        for history in [0, HISTORY + 1, usize::MAX] {
            assert!(requested(auto, Some(WIDTH), true, history).is_err());
        }
        for history in [1, 7, 8, 511, 512, 2047, 2048] {
            for ffn in [None, Some(8), Some(256), Some(1024)] {
                assert!(requested(auto, ffn, true, history).is_err());
            }
            assert!(requested(auto, Some(WIDTH), false, history).is_err());
        }
        for value in ["AUTO", "auto ", " auto"] {
            assert!(requested(Some(OsStr::new(value)), Some(WIDTH), true, HISTORY).is_err());
        }
    }
}
