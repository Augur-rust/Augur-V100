#![cfg(feature = "cuda")]

use augur_backend::CudaRuntimeError;
use augur_core::{FormatVersion, StorageId};
use augur_kernel_api::Q4KHotCudaStagingTensorV1;

use crate::CudaWeightBindingId;

#[derive(Debug)]
pub struct CudaQ4kWeightResource {
    pub storage: StorageId,
    pub format_version: FormatVersion,
    pub staging: Q4KHotCudaStagingTensorV1,
}

#[derive(Debug, Default)]
pub struct CudaWeightResources {
    resources: Vec<CudaQ4kWeightResource>,
}

impl CudaWeightResources {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_q4k_cuda_v1(
        &mut self,
        storage: StorageId,
        format_version: FormatVersion,
        staging: Q4KHotCudaStagingTensorV1,
    ) -> Result<(), CudaBindError> {
        if self
            .resources
            .iter()
            .any(|resource| resource.storage.index == storage.index)
        {
            return Err(CudaBindError::DuplicateStorage { storage });
        }

        self.resources.push(CudaQ4kWeightResource {
            storage,
            format_version,
            staging,
        });

        Ok(())
    }

    pub fn len(&self) -> usize {
        self.resources.len()
    }

    pub fn is_empty(&self) -> bool {
        self.resources.is_empty()
    }

    pub fn contains_exact(&self, storage: StorageId) -> bool {
        self.resources
            .iter()
            .any(|resource| resource.storage == storage)
    }

    pub(crate) fn get(
        &self,
        requested: StorageId,
    ) -> Result<&CudaQ4kWeightResource, CudaBindError> {
        let resource = self
            .resources
            .iter()
            .find(|resource| resource.storage.index == requested.index)
            .ok_or(CudaBindError::MissingStorage { storage: requested })?;

        if resource.storage != requested {
            return Err(CudaBindError::GenerationMismatch {
                requested,
                actual: resource.storage,
            });
        }

        Ok(resource)
    }

    pub(crate) fn take(
        &mut self,
        requested: StorageId,
    ) -> Result<CudaQ4kWeightResource, CudaBindError> {
        let index = self
            .resources
            .iter()
            .position(|resource| resource.storage.index == requested.index)
            .ok_or(CudaBindError::MissingStorage { storage: requested })?;

        let actual = self.resources[index].storage;

        if actual != requested {
            return Err(CudaBindError::GenerationMismatch { requested, actual });
        }

        Ok(self.resources.swap_remove(index))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CudaBindError {
    DuplicateStorage {
        storage: StorageId,
    },
    MissingStorage {
        storage: StorageId,
    },
    GenerationMismatch {
        requested: StorageId,
        actual: StorageId,
    },
    UnexpectedResourceCount {
        expected: usize,
        actual: usize,
    },
    UnexpectedStepCount {
        expected: usize,
        actual: usize,
    },
    UnexpectedRequirementCount {
        expected: usize,
        actual: usize,
    },
    InvalidBindingId {
        binding: CudaWeightBindingId,
    },
    RequirementMismatch {
        storage: StorageId,
    },
    StepRequirementMismatch {
        binding: CudaWeightBindingId,
    },
    UnusedRequirement {
        binding: CudaWeightBindingId,
    },
    InvalidMemoryPlan,
    InvalidStep {
        step: usize,
    },
    AliasedBuffers {
        step: usize,
        read: usize,
        write: usize,
    },
    ArithmeticOverflow,
    Runtime(CudaRuntimeError),
}

impl From<CudaRuntimeError> for CudaBindError {
    fn from(value: CudaRuntimeError) -> Self {
        Self::Runtime(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use augur_kernel_api::Q4K_HOT_CUDA_V1_BLOCK_BYTES;

    fn staging() -> Q4KHotCudaStagingTensorV1 {
        Q4KHotCudaStagingTensorV1::new(1, 256, vec![0; Q4K_HOT_CUDA_V1_BLOCK_BYTES]).unwrap()
    }

    #[test]
    fn stale_generation_does_not_take_valid_resource() {
        let valid = StorageId::new(0, 0);
        let stale = StorageId::new(0, 1);

        let mut resources = CudaWeightResources::new();
        resources
            .add_q4k_cuda_v1(valid, FormatVersion(1), staging())
            .unwrap();

        assert_eq!(
            resources.take(stale).unwrap_err(),
            CudaBindError::GenerationMismatch {
                requested: stale,
                actual: valid,
            },
        );

        assert!(resources.take(valid).is_ok());
    }

    #[test]
    fn duplicate_storage_index_is_rejected() {
        let mut resources = CudaWeightResources::new();

        resources
            .add_q4k_cuda_v1(StorageId::new(0, 0), FormatVersion(1), staging())
            .unwrap();

        assert_eq!(
            resources
                .add_q4k_cuda_v1(StorageId::new(0, 1), FormatVersion(1), staging())
                .unwrap_err(),
            CudaBindError::DuplicateStorage {
                storage: StorageId::new(0, 1),
            },
        );
    }
}
