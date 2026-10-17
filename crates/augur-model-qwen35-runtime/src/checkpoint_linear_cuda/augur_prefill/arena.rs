//! One allocation, three disjoint-in-time stage groups. All use one serial stream.
//! Common residual/norm planes survive the stage boundary and never alias it.
use super::*;
pub(super) const COMMON: [(&str, usize); 5] = [
    ("normalized", H),
    ("attention", H),
    ("attention_residual", H),
    ("post_norm", H),
    ("ffn_output", H),
];
const GDN: [(&str, usize); 9] = [
    ("qkv", 10240),
    ("z", 6144),
    ("alpha64", 64),
    ("beta64", 64),
    ("alpha", 48),
    ("beta_raw", 48),
    ("activated", 10240),
    ("recurrent", 6144),
    ("linear_output", 6144),
];
const FULL: [(&str, usize); 9] = [
    ("q_full", 12288),
    ("k_raw", 1024),
    ("value", 1024),
    ("key_work", 1024),
    ("value_work", 1024),
    ("query", 6144),
    ("q_gate", 6144),
    ("attention", 6144),
    ("gated", 6144),
];
const FFN: [(&str, usize); 3] = [("gate", I), ("up", I), ("activation", I)];
#[derive(Clone, Copy, Debug)]
struct Extent {
    offset: usize,
    bytes: usize,
}
pub(super) struct Layout {
    common: BTreeMap<&'static str, Extent>,
    gdn: BTreeMap<&'static str, Extent>,
    full: BTreeMap<&'static str, Extent>,
    ffn: BTreeMap<&'static str, Extent>,
    pub(super) bytes: usize,
}
fn group(
    tokens: usize,
    start: usize,
    shapes: &[(&'static str, usize)],
) -> Result<(BTreeMap<&'static str, Extent>, usize)> {
    let mut offset = start;
    let mut result = BTreeMap::new();
    for &(name, cols) in shapes {
        offset = offset.checked_add(255).ok_or("arena alignment overflow")? & !255;
        let bytes = tokens
            .checked_mul(cols)
            .and_then(|n| n.checked_mul(4))
            .ok_or("arena extent overflow")?;
        result.insert(name, Extent { offset, bytes });
        offset = offset.checked_add(bytes).ok_or("arena size overflow")?;
    }
    Ok((result, offset))
}
impl Layout {
    fn new(tokens: usize, separate: bool) -> Result<Self> {
        if !(1..=augur_backend::cuda_qwen35_prefill_cublas::AUGUR_MAX_BATCH).contains(&tokens)
        {
            return Err("Augur arena width exceeds compiled capacity".into());
        }
        // Input normalization dies before post-attention normalization; attention
        // output dies at residual-add, before FFN output is produced.
        let compact_common = &COMMON[..3];
        let (mut common, base) = group(tokens, 0, if separate { &COMMON } else { compact_common })?;
        if !separate {
            common.insert("post_norm", common["normalized"]);
            common.insert("ffn_output", common["attention"]);
        }
        let (gdn, gdn_end) = group(tokens, base, &GDN)?;
        let (full, full_end) = group(tokens, if separate { gdn_end } else { base }, &FULL)?;
        let (mut ffn, ffn_end) = group(
            tokens,
            if separate { full_end } else { base },
            if separate { &FFN } else { &FFN[..2] },
        )?;
        if !separate {
            ffn.insert("activation", ffn["gate"]);
        }
        Ok(Self {
            common,
            gdn,
            full,
            ffn,
            bytes: gdn_end.max(full_end).max(ffn_end),
        })
    }
    pub(super) fn active(tokens: usize) -> Result<Self> {
        Self::new(tokens, separate())
    }
}
#[cfg(test)]
thread_local! { static SEPARATE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
fn separate() -> bool {
    #[cfg(test)]
    {
        return SEPARATE.with(|v| v.get());
    }
    #[cfg(not(test))]
    {
        false
    }
}
pub(crate) fn profile() -> &'static str {
    if separate() {
        "disjoint_stage_reference_v1"
    } else {
        "serial_stage_lifetime_arena_inplace_v2"
    }
}
/// CPU-thread-local test control, not getenv/native global numerical dispatch.
#[cfg(test)]
pub(crate) fn with_separate<T>(f: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            SEPARATE.with(|v| v.set(self.0));
        }
    }
    let _restore = Restore(SEPARATE.with(|v| v.replace(true)));
    f()
}
#[derive(Clone)]
pub(crate) struct Plane {
    pub(crate) owner: Buffer,
    pub(crate) offset: usize,
    bytes: usize,
}
impl Plane {
    pub(crate) fn view(&self, byte_offset: usize, elements: usize) -> Result<CudaF32View> {
        let bytes = elements.checked_mul(4).ok_or("arena view overflow")?;
        if byte_offset
            .checked_add(bytes)
            .is_none_or(|end| end > self.bytes)
        {
            return Err("arena view exceeds its stage plane".into());
        }
        prefix(
            &self.owner,
            self.offset
                .checked_add(byte_offset)
                .ok_or("arena view offset overflow")?,
            elements,
        )
    }
}
pub(super) struct Bound {
    pub(super) common_gdn: BTreeMap<&'static str, Plane>,
    pub(super) full: BTreeMap<&'static str, Plane>,
    pub(super) ffn: [Plane; 3],
}
pub(super) fn bind(stream: &OwnedCudaStream, layout: Layout) -> Result<Bound> {
    let owner = Rc::new(OwnedCudaBuffer::allocate(
        stream.device_ordinal(),
        layout.bytes,
    )?);
    let bind = |map: BTreeMap<&'static str, Extent>| {
        map.into_iter()
            .map(|(name, e)| {
                (
                    name,
                    Plane {
                        owner: Rc::clone(&owner),
                        offset: e.offset,
                        bytes: e.bytes,
                    },
                )
            })
            .collect::<BTreeMap<_, _>>()
    };
    let mut common_gdn = bind(layout.common);
    common_gdn.extend(bind(layout.gdn));
    let full = bind(layout.full);
    let mut ffn = bind(layout.ffn);
    Ok(Bound {
        common_gdn,
        full,
        ffn: [
            ffn.remove("gate").unwrap(),
            ffn.remove("up").unwrap(),
            ffn.remove("activation").unwrap(),
        ],
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    fn disjoint<'a>(extents: impl Iterator<Item = &'a Extent>, bytes: usize) {
        let mut sorted: Vec<_> = extents.collect();
        sorted.sort_by_key(|e| e.offset);
        let mut end = 0;
        for e in sorted {
            assert_eq!(e.offset % 256, 0);
            assert!(e.offset >= end);
            end = e.offset + e.bytes;
            assert!(end <= bytes);
        }
    }
    #[test]
    fn lifetime_groups_are_disjoint_within_every_stage_and_tail() {
        for n in [
            1, 7, 8, 9, 44, 512, 1024, 2048, 4096, 4118, 4352, 5120, 6400,
        ]
        .into_iter()
        .filter(|&n| n <= augur_backend::cuda_qwen35_prefill_cublas::AUGUR_MAX_BATCH)
        {
            let l = Layout::new(n, false).unwrap();
            for stage in [&l.gdn, &l.full] {
                disjoint(
                    [
                        &l.common["normalized"],
                        &l.common["attention"],
                        &l.common["attention_residual"],
                    ]
                    .into_iter()
                    .chain(stage.values()),
                    l.bytes,
                );
            }
            disjoint(
                [
                    &l.common["post_norm"],
                    &l.common["ffn_output"],
                    &l.common["attention_residual"],
                    &l.ffn["gate"],
                    &l.ffn["up"],
                ]
                .into_iter(),
                l.bytes,
            );
            assert_eq!(l.common["normalized"].offset, l.common["post_norm"].offset);
            assert_eq!(l.common["attention"].offset, l.common["ffn_output"].offset);
            assert_eq!(l.ffn["gate"].offset, l.ffn["activation"].offset);
            assert_eq!(l.gdn["qkv"].offset, l.full["q_full"].offset);
            assert_eq!(l.gdn["qkv"].offset, l.ffn["gate"].offset);
            if n % 16 == 0 {
                assert_eq!(l.bytes, n * (61440 + 163840));
            }
            let l = Layout::new(n, true).unwrap();
            disjoint(
                l.common
                    .values()
                    .chain(l.gdn.values())
                    .chain(l.full.values())
                    .chain(l.ffn.values()),
                l.bytes,
            );
        }
        assert!(Layout::new(0, false).is_err());
        assert!(Layout::new(usize::MAX, false).is_err());
    }
    #[test]
    fn reference_layout_control_restores_on_unwind() {
        assert!(!separate());
        let result = std::panic::catch_unwind(|| {
            with_separate(|| {
                assert!(separate());
                panic!("test unwind");
            })
        });
        assert!(result.is_err());
        assert!(!separate());
    }
}
