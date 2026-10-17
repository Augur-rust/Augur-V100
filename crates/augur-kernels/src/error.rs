use augur_kernel_api::KernelLaunchError;

#[derive(Clone, Debug, PartialEq)]
pub enum KernelError {
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

    NonFiniteActivation {
        index: usize,
    },

    ActivationValueOutOfRange {
        index: usize,
    },

    /// The concrete kernel requires CPU features unavailable on this host.
    MissingCpuFeatures,
}

impl KernelError {
    pub fn into_launch_error(self) -> KernelLaunchError {
        match self {
            Self::BadActivationLength { expected, actual } => {
                KernelLaunchError::BadActivationLength { expected, actual }
            }
            Self::BadOutputLength { expected, actual } => {
                KernelLaunchError::BadOutputLength { expected, actual }
            }
            Self::BadBiasLength { expected, actual } => {
                KernelLaunchError::BadBiasLength { expected, actual }
            }
            Self::BadWeightShape => KernelLaunchError::BadWeightShape,
            Self::BadWeightBlockCount { expected, actual } => {
                KernelLaunchError::BadWeightBlockCount { expected, actual }
            }
            Self::BadActivationBlockCount { expected, actual } => {
                KernelLaunchError::BadActivationBlockCount { expected, actual }
            }
            Self::BadWeight => KernelLaunchError::BadWeight,
            Self::BadRuntimeConfig => KernelLaunchError::BadRuntimeConfig,
            Self::BadWorkspace => KernelLaunchError::BadWorkspace,
            Self::NonFiniteActivation { index } => KernelLaunchError::NonFiniteActivation { index },
            Self::ActivationValueOutOfRange { index } => {
                KernelLaunchError::ActivationValueOutOfRange { index }
            }
            Self::MissingCpuFeatures => KernelLaunchError::MissingCpuFeatures,
        }
    }
}
