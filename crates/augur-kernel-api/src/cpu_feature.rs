/// CPU features required by a concrete kernel registration.
///
/// This is a selection-time value type. Feature detection must happen before
/// execution and must never enter a row/block/SIMD inner loop.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct CpuFeatureSet(u64);

impl CpuFeatureSet {
    pub const NONE: Self = Self(0);
    pub const AVX2: Self = Self(1 << 0);
    pub const FMA: Self = Self(1 << 1);
    pub const AVX512F: Self = Self(1 << 2);
    pub const AVX512_VNNI: Self = Self(1 << 3);

    pub const fn contains_all(self, required: Self) -> bool {
        (self.0 & required.0) == required.0
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn bits(self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_set_contains_required_subset() {
        let available = CpuFeatureSet::AVX2.union(CpuFeatureSet::FMA);

        assert!(available.contains_all(CpuFeatureSet::NONE));
        assert!(available.contains_all(CpuFeatureSet::AVX2));
        assert!(available.contains_all(CpuFeatureSet::FMA));
        assert!(available.contains_all(CpuFeatureSet::AVX2.union(CpuFeatureSet::FMA)));
        assert!(!available.contains_all(CpuFeatureSet::AVX512F));
    }
}
