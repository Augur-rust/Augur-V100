use augur_core::CompileError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BoundaryId(pub u16);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PadValue {
    Zero,
    NegativeInfinity,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QdqFormat {
    Fp8E4M3,
    Fp4E2M1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReshapeAttrs;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SliceAttrs {
    pub axis: usize,
    pub start: usize,
    pub len: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConcatAttrs {
    pub axis: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PadAttrs {
    pub axis: usize,
    pub before: usize,
    pub after: usize,
    pub value: PadValue,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BroadcastAddAttrs;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SoftmaxAttrs {
    pub axis: usize,
    pub boundary: Option<BoundaryId>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReduceSumAttrs {
    pub axis: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrecisionBoundaryAttrs {
    pub target: augur_core::DType,
    pub boundary: BoundaryId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RmsNormAttrs {
    pub epsilon: FiniteF32,
    pub boundary: BoundaryId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RopeAttrs {
    pub rope_dim: usize,
    pub position_start: usize,
    pub position_step: usize,
    pub base: FiniteF32,
    pub original_context: usize,
    pub factor: FiniteF32,
    pub beta_fast: FiniteF32,
    pub beta_slow: FiniteF32,
    pub boundary: BoundaryId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HadamardAttrs {
    pub normalized: bool,
    pub boundary: BoundaryId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QdqAttrs {
    pub format: QdqFormat,
    pub group_size: usize,
    pub boundary: BoundaryId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OrderedTopKTieBreak {
    LowerIndex,
}

/// Source-bound ordered selection semantics. Runtime kernels must preserve
/// the score order and use the declared tie-break; padding is an index value,
/// not a host-side sentinel injection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OrderedTopKAttrs {
    pub axis: usize,
    pub top_k: usize,
    pub causal: bool,
    pub padding: i32,
    pub index_base: i32,
    pub tie_break: OrderedTopKTieBreak,
}

/// Epilogue fused onto a QMatmul output.
///
/// Bias identity is represented by the third QMatmul input. It is not
/// duplicated in this enum.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QMatmulEpilogue {
    Identity,
    Bias,
    BiasSilu,
}

/// Logical QMatmul attributes.
///
/// Matrix dimensions are derived from the weight TensorDesc.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QMatmulAttrs {
    pub epilogue: QMatmulEpilogue,
}

/// Equality-safe finite f32 Graph attribute.
///
/// Graph attributes must remain structurally comparable. Storing the IEEE-754
/// bits also prevents NaN from violating `Eq`; construction rejects all
/// non-finite values before the attribute enters Graph IR.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FiniteF32(u32);

impl FiniteF32 {
    pub fn try_new(value: f32) -> Result<Self, CompileError> {
        if value.is_finite() {
            Ok(Self(value.to_bits()))
        } else {
            Err(CompileError::InvalidInput)
        }
    }

    pub const fn value(self) -> f32 {
        f32::from_bits(self.0)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClampAttrs {
    min: Option<FiniteF32>,
    max: Option<FiniteF32>,
}

impl ClampAttrs {
    pub fn try_new(min: Option<f32>, max: Option<f32>) -> Result<Self, CompileError> {
        let min = min.map(FiniteF32::try_new).transpose()?;
        let max = max.map(FiniteF32::try_new).transpose()?;
        if min.is_none() && max.is_none() {
            return Err(CompileError::InvalidInput);
        }
        if let (Some(min), Some(max)) = (min, max) {
            if min.value() > max.value() {
                return Err(CompileError::InvalidInput);
            }
        }
        Ok(Self { min, max })
    }

    pub const fn min(&self) -> Option<f32> {
        match self.min {
            Some(value) => Some(value.value()),
            None => None,
        }
    }

    pub const fn max(&self) -> Option<f32> {
        match self.max {
            Some(value) => Some(value.value()),
            None => None,
        }
    }

    pub fn is_valid(&self) -> bool {
        (self.min.is_some() || self.max.is_some())
            && match (self.min(), self.max()) {
                (Some(min), Some(max)) => min.is_finite() && max.is_finite() && min <= max,
                (Some(min), None) => min.is_finite(),
                (None, Some(max)) => max.is_finite(),
                (None, None) => false,
            }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FusedMlpActivation {
    Silu,
}

/// Generic gated-MLP semantics. Model identity and capability policy do not
/// belong in these attributes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FusedMlpAttrs {
    pub gate_clamp: ClampAttrs,
    pub up_clamp: ClampAttrs,
    pub activation: FusedMlpActivation,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Op {
    Add,
    Mul,
    Clamp(ClampAttrs),
    Silu,
    Matmul,
    QMatmul(QMatmulAttrs),
    Reshape(ReshapeAttrs),
    Slice(SliceAttrs),
    Concat(ConcatAttrs),
    Pad(PadAttrs),
    BroadcastAdd(BroadcastAddAttrs),
    Softmax(SoftmaxAttrs),
    ReduceSum(ReduceSumAttrs),
    PrecisionBoundary(PrecisionBoundaryAttrs),
    RmsNorm(RmsNormAttrs),
    Rope(RopeAttrs),
    Hadamard(HadamardAttrs),
    Qdq(QdqAttrs),
    OrderedTopK(OrderedTopKAttrs),
    FusedMlp(FusedMlpAttrs),
}
