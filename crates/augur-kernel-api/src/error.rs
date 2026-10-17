#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HotLayoutError {
    BadShape,
    BadBlockCount { expected: usize, actual: usize },
    BadPayloadBytes { expected: usize, actual: usize },
    BadReservedBytes { block: usize },
    BadOutlierCsr,
    NonFinitePayload { index: usize },
}

/// Stable error contract at the bound kernel-launch boundary.
///
/// Detailed reference/format errors remain owned by their implementation
/// crates. ExecutionPlan only needs stable launch-boundary diagnostics.
///
/// Q8Hot preparation errors are reported before the first output write, so
/// failed preparation leaves the destination owner unchanged.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KernelLaunchError {
    ReferenceFormat,

    BadActivationLength {
        expected: usize,
        actual: usize,
    },
    BadOutputLength {
        expected: usize,
        actual: usize,
    },
    BadBiasLength {
        expected: usize,
        actual: usize,
    },

    BadWeightShape,
    BadWeightBlockCount {
        expected: usize,
        actual: usize,
    },
    BadActivationBlockCount {
        expected: usize,
        actual: usize,
    },
    BadWeight,
    BadRuntimeConfig,
    BadWorkspace,

    /// The input contains NaN or infinity.
    ///
    /// `index` is the first invalid logical activation element.
    NonFiniteActivation {
        index: usize,
    },

    /// A finite input value is outside the production Q8Hot accepted domain.
    ///
    /// `index` is the first offending logical activation element.
    ActivationValueOutOfRange {
        index: usize,
    },

    /// The selected concrete kernel requires CPU features unavailable on the
    /// current host.
    ///
    /// Production lowering should prevent this in registered execution. This
    /// variant protects direct safe launch calls and stale/misbound launch
    /// targets.
    MissingCpuFeatures,
}
