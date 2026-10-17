use std::collections::HashSet;
use std::mem::size_of_val;
use std::sync::Arc;

use augur_core::{FormatVersion, PhysicalLayoutId, QuantFormatId, StorageId};
use augur_kernel_api::{
    DualScaleHotCpuBlock, DualScaleHotCpuTensor, Fp8E4M3ScaleGridHotCpuTensor, Iq2XxsHotCpuBlock,
    Iq2XxsHotCpuTensor, Mxfp4HotCpuTensor, Q2KHotCpuBlock, Q2KHotCpuTensor, Q4KHotCpuBlock,
    Q4KHotCpuTensor, Q8_0HotCpuBlock, Q8_0HotCpuTensor, FP8_E4M3_SCALE_TILE,
};

use crate::{
    BindError, CpuWeightRequirement, CpuWeightResourceKind, ResourceInsertError,
    ResourceLookupError, WeightBindingId,
};

/// Compiler/binding-side CPU weight resource.
///
/// The resource store owns concrete, already-repacked HotWeight values.
/// Binding consumes these values and moves them into stable `Arc` owners.
///
/// This type never enters `augur-exec`.
#[derive(Debug, PartialEq)]
pub enum CpuWeightResource {
    Q4KHotCpuPortable {
        version: FormatVersion,
        tensor: Q4KHotCpuTensor,
    },

    DualScaleHotCpuPortable {
        version: FormatVersion,
        tensor: DualScaleHotCpuTensor,
    },
    Q8_0HotCpuPortable {
        version: FormatVersion,
        tensor: Q8_0HotCpuTensor,
    },
    Q2KHotCpuPortable {
        version: FormatVersion,
        tensor: Q2KHotCpuTensor,
    },
    Iq2XxsHotCpuPortable {
        version: FormatVersion,
        tensor: Iq2XxsHotCpuTensor,
    },
    Mxfp4HotCpuPortable {
        version: FormatVersion,
        tensor: Mxfp4HotCpuTensor,
    },
    Fp8ScaleGridHotCpuPortable {
        version: FormatVersion,
        tensor: Fp8E4M3ScaleGridHotCpuTensor,
    },
}

impl CpuWeightResource {
    pub fn kind(&self) -> CpuWeightResourceKind {
        match self {
            Self::Q4KHotCpuPortable { .. } => CpuWeightResourceKind::Q4KHotCpuPortable,

            Self::DualScaleHotCpuPortable { .. } => CpuWeightResourceKind::DualScaleHotCpuPortable,
            Self::Q8_0HotCpuPortable { .. } => CpuWeightResourceKind::Q8_0HotCpuPortable,
            Self::Q2KHotCpuPortable { .. } => CpuWeightResourceKind::Q2KHotCpuPortable,
            Self::Iq2XxsHotCpuPortable { .. } => CpuWeightResourceKind::Iq2XxsHotCpuPortable,
            Self::Mxfp4HotCpuPortable { .. } => CpuWeightResourceKind::Mxfp4HotCpuPortable,
            Self::Fp8ScaleGridHotCpuPortable { .. } => {
                CpuWeightResourceKind::Fp8ScaleGridHotCpuPortable
            }
        }
    }

    pub fn quant_format(&self) -> QuantFormatId {
        match self {
            Self::Q4KHotCpuPortable { .. } => QuantFormatId::Q4K,

            Self::DualScaleHotCpuPortable { .. } => QuantFormatId::DualScale,
            Self::Q8_0HotCpuPortable { .. } => QuantFormatId::Q8_0,
            Self::Q2KHotCpuPortable { .. } => QuantFormatId::Q2K,
            Self::Iq2XxsHotCpuPortable { .. } => QuantFormatId::IQ2_XXS,
            Self::Mxfp4HotCpuPortable { .. } => QuantFormatId::MXFP4,
            Self::Fp8ScaleGridHotCpuPortable { .. } => QuantFormatId::FP8E4M3ScaleGrid,
        }
    }

    pub fn format_version(&self) -> FormatVersion {
        match self {
            Self::Q4KHotCpuPortable { version, .. }
            | Self::DualScaleHotCpuPortable { version, .. }
            | Self::Q8_0HotCpuPortable { version, .. }
            | Self::Q2KHotCpuPortable { version, .. }
            | Self::Iq2XxsHotCpuPortable { version, .. } => *version,
            Self::Mxfp4HotCpuPortable { version, .. }
            | Self::Fp8ScaleGridHotCpuPortable { version, .. } => *version,
        }
    }

    pub fn physical_layout(&self) -> PhysicalLayoutId {
        match self {
            Self::Q4KHotCpuPortable { .. } => Q4KHotCpuTensor::PHYSICAL_LAYOUT,

            Self::DualScaleHotCpuPortable { .. } => {
                PhysicalLayoutId::DUAL_SCALE_HOT_CPU_PORTABLE_V1
            }
            Self::Q8_0HotCpuPortable { .. } => Q8_0HotCpuTensor::PHYSICAL_LAYOUT,
            Self::Q2KHotCpuPortable { .. } => Q2KHotCpuTensor::PHYSICAL_LAYOUT,
            Self::Iq2XxsHotCpuPortable { .. } => Iq2XxsHotCpuTensor::PHYSICAL_LAYOUT,
            Self::Mxfp4HotCpuPortable { .. } => Mxfp4HotCpuTensor::PHYSICAL_LAYOUT,
            Self::Fp8ScaleGridHotCpuPortable { .. } => {
                Fp8E4M3ScaleGridHotCpuTensor::PHYSICAL_LAYOUT
            }
        }
    }

    pub fn rows(&self) -> usize {
        match self {
            Self::Q4KHotCpuPortable { tensor, .. } => tensor.rows(),

            Self::DualScaleHotCpuPortable { tensor, .. } => tensor.rows(),
            Self::Q8_0HotCpuPortable { tensor, .. } => tensor.rows(),
            Self::Q2KHotCpuPortable { tensor, .. } => tensor.rows(),
            Self::Iq2XxsHotCpuPortable { tensor, .. } => tensor.rows(),
            Self::Mxfp4HotCpuPortable { tensor, .. } => tensor.rows(),
            Self::Fp8ScaleGridHotCpuPortable { tensor, .. } => tensor.rows(),
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Self::Q4KHotCpuPortable { tensor, .. } => tensor.cols(),

            Self::DualScaleHotCpuPortable { tensor, .. } => tensor.cols(),
            Self::Q8_0HotCpuPortable { tensor, .. } => tensor.cols(),
            Self::Q2KHotCpuPortable { tensor, .. } => tensor.cols(),
            Self::Iq2XxsHotCpuPortable { tensor, .. } => tensor.cols(),
            Self::Mxfp4HotCpuPortable { tensor, .. } => tensor.cols(),
            Self::Fp8ScaleGridHotCpuPortable { tensor, .. } => tensor.cols(),
        }
    }
}

/// Generation-safe resource slot.
///
/// The slot retains its complete `StorageId` after the owner is consumed.
/// Therefore:
///
/// - an invalid index is distinguishable from an empty slot;
/// - a stale generation is distinguishable from an already-taken owner;
/// - a stale request cannot consume the valid owner.
#[derive(Debug)]
struct CpuWeightResourceSlot {
    id: StorageId,
    resource: Option<CpuWeightResource>,
}

/// Compiler/runtime-boundary weight resource input.
///
/// This store deliberately does not implement `Clone`: binding must move
/// concrete HotWeight owners into stable `Arc` owners instead of deep-cloning
/// their payload.
///
/// This type currently lives in `augur-compiler`. It may move to the model/runtime
/// resource layer in a later milestone.
#[derive(Debug, Default)]
pub struct CpuWeightResources {
    slots: Vec<CpuWeightResourceSlot>,
}

impl CpuWeightResources {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    pub fn add_q4k_hot_cpu_portable(
        &mut self,
        tensor: Q4KHotCpuTensor,
    ) -> Result<StorageId, ResourceInsertError> {
        self.add_q4k_hot_cpu_portable_versioned(FormatVersion(1), tensor)
    }

    pub fn add_q4k_hot_cpu_portable_versioned(
        &mut self,
        version: FormatVersion,
        tensor: Q4KHotCpuTensor,
    ) -> Result<StorageId, ResourceInsertError> {
        self.insert(CpuWeightResource::Q4KHotCpuPortable { version, tensor })
    }

    pub fn add_dual_scale_hot_cpu_portable(
        &mut self,
        tensor: DualScaleHotCpuTensor,
    ) -> Result<StorageId, ResourceInsertError> {
        self.add_dual_scale_hot_cpu_portable_versioned(FormatVersion(1), tensor)
    }

    pub fn add_dual_scale_hot_cpu_portable_versioned(
        &mut self,
        version: FormatVersion,
        tensor: DualScaleHotCpuTensor,
    ) -> Result<StorageId, ResourceInsertError> {
        self.insert(CpuWeightResource::DualScaleHotCpuPortable { version, tensor })
    }

    pub fn add_q8_0_hot_cpu_portable(
        &mut self,
        tensor: Q8_0HotCpuTensor,
    ) -> Result<StorageId, ResourceInsertError> {
        self.insert(CpuWeightResource::Q8_0HotCpuPortable {
            version: FormatVersion(1),
            tensor,
        })
    }

    pub fn add_q2k_hot_cpu_portable(
        &mut self,
        tensor: Q2KHotCpuTensor,
    ) -> Result<StorageId, ResourceInsertError> {
        self.insert(CpuWeightResource::Q2KHotCpuPortable {
            version: FormatVersion(1),
            tensor,
        })
    }

    pub fn add_iq2_xxs_hot_cpu_portable(
        &mut self,
        tensor: Iq2XxsHotCpuTensor,
    ) -> Result<StorageId, ResourceInsertError> {
        self.insert(CpuWeightResource::Iq2XxsHotCpuPortable {
            version: FormatVersion(1),
            tensor,
        })
    }

    pub fn add_mxfp4_hot_cpu_portable(
        &mut self,
        tensor: Mxfp4HotCpuTensor,
    ) -> Result<StorageId, ResourceInsertError> {
        self.insert(CpuWeightResource::Mxfp4HotCpuPortable {
            version: FormatVersion(1),
            tensor,
        })
    }

    pub fn add_fp8_scale_grid_hot_cpu_portable(
        &mut self,
        tensor: Fp8E4M3ScaleGridHotCpuTensor,
    ) -> Result<StorageId, ResourceInsertError> {
        self.insert(CpuWeightResource::Fp8ScaleGridHotCpuPortable {
            version: FormatVersion(1),
            tensor,
        })
    }

    fn insert(&mut self, resource: CpuWeightResource) -> Result<StorageId, ResourceInsertError> {
        let index =
            u32::try_from(self.slots.len()).map_err(|_| ResourceInsertError::TooManyResources)?;

        let id = StorageId {
            index,
            generation: 0,
        };

        self.slots.push(CpuWeightResourceSlot {
            id,
            resource: Some(resource),
        });

        Ok(id)
    }

    fn get(&self, requested: StorageId) -> Result<&CpuWeightResource, ResourceLookupError> {
        let slot = self
            .slots
            .get(requested.index as usize)
            .ok_or(ResourceLookupError::IndexOutOfRange { requested })?;

        if slot.id != requested {
            return Err(ResourceLookupError::GenerationMismatch {
                requested,
                actual: slot.id,
            });
        }

        slot.resource
            .as_ref()
            .ok_or(ResourceLookupError::AlreadyTaken { requested })
    }

    /// Moves one concrete owner out of the resource store.
    ///
    /// The complete `StorageId` is validated before the slot is modified.
    /// A stale-generation request therefore cannot consume or alter the valid
    /// resource.
    ///
    /// The slot retains its identity after the owner is taken, allowing
    /// repeated consumption to return `AlreadyTaken`.
    fn take(&mut self, requested: StorageId) -> Result<CpuWeightResource, ResourceLookupError> {
        let slot = self
            .slots
            .get_mut(requested.index as usize)
            .ok_or(ResourceLookupError::IndexOutOfRange { requested })?;

        if slot.id != requested {
            return Err(ResourceLookupError::GenerationMismatch {
                requested,
                actual: slot.id,
            });
        }

        slot.resource
            .take()
            .ok_or(ResourceLookupError::AlreadyTaken { requested })
    }
}

/// Concrete immutable owner produced by binding.
///
/// This enum is compiler-side binding state. It is not an execution-time
/// dynamic-dispatch abstraction.
#[derive(Debug)]
enum BoundCpuWeight {
    Q4K(Arc<Q4KHotCpuTensor>),

    DualScale(Arc<DualScaleHotCpuTensor>),
    Q8_0(Arc<Q8_0HotCpuTensor>),
    Q2K(Arc<Q2KHotCpuTensor>),
    Iq2Xxs(Arc<Iq2XxsHotCpuTensor>),
    Mxfp4(Arc<Mxfp4HotCpuTensor>),
    Fp8ScaleGrid(Arc<Fp8E4M3ScaleGridHotCpuTensor>),
}

/// Fully resolved owners indexed by `WeightBindingId`.
///
/// E3 consumes this table while converting compiled DualScale steps into
/// executable `PlannedOp` values.
#[derive(Debug)]
pub struct CpuBoundWeights {
    weights: Box<[BoundCpuWeight]>,
    accounting: CpuWeightBindingAccounting,
}

impl CpuBoundWeights {
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    pub fn accounting(&self) -> CpuWeightBindingAccounting {
        self.accounting
    }

    pub fn q4k_hot_cpu_portable(&self, binding: WeightBindingId) -> Option<&Arc<Q4KHotCpuTensor>> {
        match self.weights.get(binding.index())? {
            BoundCpuWeight::Q4K(owner) => Some(owner),
            _ => None,
        }
    }

    pub fn dual_scale_hot_cpu_portable(
        &self,
        binding: WeightBindingId,
    ) -> Option<&Arc<DualScaleHotCpuTensor>> {
        match self.weights.get(binding.index())? {
            BoundCpuWeight::DualScale(owner) => Some(owner),
            _ => None,
        }
    }

    pub fn q8_0_hot_cpu_portable(
        &self,
        binding: WeightBindingId,
    ) -> Option<&Arc<Q8_0HotCpuTensor>> {
        match self.weights.get(binding.index())? {
            BoundCpuWeight::Q8_0(owner) => Some(owner),
            _ => None,
        }
    }

    pub fn q2k_hot_cpu_portable(&self, binding: WeightBindingId) -> Option<&Arc<Q2KHotCpuTensor>> {
        match self.weights.get(binding.index())? {
            BoundCpuWeight::Q2K(owner) => Some(owner),
            _ => None,
        }
    }

    pub fn iq2_xxs_hot_cpu_portable(
        &self,
        binding: WeightBindingId,
    ) -> Option<&Arc<Iq2XxsHotCpuTensor>> {
        match self.weights.get(binding.index())? {
            BoundCpuWeight::Iq2Xxs(owner) => Some(owner),
            _ => None,
        }
    }

    pub fn mxfp4_hot_cpu_portable(
        &self,
        binding: WeightBindingId,
    ) -> Option<&Arc<Mxfp4HotCpuTensor>> {
        match self.weights.get(binding.index())? {
            BoundCpuWeight::Mxfp4(owner) => Some(owner),
            _ => None,
        }
    }

    pub fn fp8_scale_grid_hot_cpu_portable(
        &self,
        binding: WeightBindingId,
    ) -> Option<&Arc<Fp8E4M3ScaleGridHotCpuTensor>> {
        match self.weights.get(binding.index())? {
            BoundCpuWeight::Fp8ScaleGrid(owner) => Some(owner),
            _ => None,
        }
    }
}

/// Logical HotWeight payload accounting.
///
/// `hot_weight_payload_bytes` includes:
///
/// - Q4K Hot blocks;
/// - DualScale Hot blocks;
/// - DualScale CSR row pointers;
/// - DualScale CSR column indexes;
/// - DualScale CSR delta values.
/// - Q8_0, Q2_K and MXFP4 Hot blocks;
/// - FP8 scale-grid E4M3 values and decoded F32 scales.
///
/// It does not include:
///
/// - `Arc` allocation headers;
/// - `Vec` headers;
/// - allocator metadata;
/// - cache-line padding;
/// - allocator-specific overhead;
/// - transient binding scratch such as duplicate-detection tables.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CpuWeightBindingAccounting {
    pub unique_weight_count: usize,

    pub q4k_hot_block_count: usize,

    pub dual_scale_hot_block_count: usize,

    pub dual_scale_outlier_count: usize,

    pub q8_0_hot_block_count: usize,

    pub q2k_hot_block_count: usize,

    pub iq2_xxs_hot_block_count: usize,

    pub mxfp4_hot_block_count: usize,

    pub fp8_scale_grid_value_count: usize,

    pub fp8_scale_grid_scale_count: usize,

    pub hot_weight_payload_bytes: usize,
}

/// Consumes a resource store and produces concrete Arc-owned HotWeights.
///
/// Binding has two phases:
///
/// 1. validate every requirement and calculate accounting without consuming
///    any owner;
/// 2. take each owner and move it into an `Arc`.
///
/// This function does not:
///
/// - perform registry lookup;
/// - modify launch selection;
/// - read Graph IR;
/// - execute Compact repack;
/// - construct executable steps.
pub fn bind_cpu_weight_resources(
    requirements: &[CpuWeightRequirement],
    mut resources: CpuWeightResources,
) -> Result<CpuBoundWeights, BindError> {
    let accounting = preflight_requirements(requirements, &resources)?;

    let mut weights = Vec::with_capacity(requirements.len());

    for requirement in requirements {
        let resource = resources
            .take(requirement.storage)
            .map_err(BindError::ResourceLookup)?;

        match resource {
            CpuWeightResource::Q4KHotCpuPortable { version: _, tensor } => {
                weights.push(BoundCpuWeight::Q4K(Arc::new(tensor)));
            }

            CpuWeightResource::DualScaleHotCpuPortable { version: _, tensor } => {
                weights.push(BoundCpuWeight::DualScale(Arc::new(tensor)));
            }
            CpuWeightResource::Q8_0HotCpuPortable { version: _, tensor } => {
                weights.push(BoundCpuWeight::Q8_0(Arc::new(tensor)));
            }
            CpuWeightResource::Q2KHotCpuPortable { version: _, tensor } => {
                weights.push(BoundCpuWeight::Q2K(Arc::new(tensor)));
            }
            CpuWeightResource::Iq2XxsHotCpuPortable { version: _, tensor } => {
                weights.push(BoundCpuWeight::Iq2Xxs(Arc::new(tensor)));
            }
            CpuWeightResource::Mxfp4HotCpuPortable { version: _, tensor } => {
                weights.push(BoundCpuWeight::Mxfp4(Arc::new(tensor)));
            }
            CpuWeightResource::Fp8ScaleGridHotCpuPortable { version: _, tensor } => {
                weights.push(BoundCpuWeight::Fp8ScaleGrid(Arc::new(tensor)));
            }
        }
    }

    Ok(CpuBoundWeights {
        weights: weights.into_boxed_slice(),
        accounting,
    })
}

fn preflight_requirements(
    requirements: &[CpuWeightRequirement],
    resources: &CpuWeightResources,
) -> Result<CpuWeightBindingAccounting, BindError> {
    let mut accounting = CpuWeightBindingAccounting::default();

    // Binding-only allocation. This does not enter the executable plan or
    // execution path.
    let mut seen_storage = HashSet::with_capacity(requirements.len());

    for (index, requirement) in requirements.iter().enumerate() {
        if requirement.binding.index() != index {
            return Err(BindError::InvalidBindingId {
                binding: requirement.binding,
            });
        }

        if !seen_storage.insert(requirement.storage) {
            return Err(BindError::DuplicateStorageRequirement {
                storage: requirement.storage,
            });
        }

        let resource = resources
            .get(requirement.storage)
            .map_err(BindError::ResourceLookup)?;

        validate_requirement(requirement, resource)?;
        account_resource(&mut accounting, resource)?;
    }

    Ok(accounting)
}

fn account_resource(
    accounting: &mut CpuWeightBindingAccounting,
    resource: &CpuWeightResource,
) -> Result<(), BindError> {
    accounting.unique_weight_count = accounting
        .unique_weight_count
        .checked_add(1)
        .ok_or(BindError::AccountingOverflow)?;

    match resource {
        CpuWeightResource::Q4KHotCpuPortable { tensor, .. } => {
            accounting.q4k_hot_block_count = accounting
                .q4k_hot_block_count
                .checked_add(tensor.blocks().len())
                .ok_or(BindError::AccountingOverflow)?;

            add_payload_bytes(accounting, size_of_val(tensor.blocks()))?;
        }

        CpuWeightResource::DualScaleHotCpuPortable { tensor, .. } => {
            let block_count: usize = tensor.blocks().len();
            let outlier_count: usize = tensor.outliers().nnz();

            accounting.dual_scale_hot_block_count = accounting
                .dual_scale_hot_block_count
                .checked_add(block_count)
                .ok_or(BindError::AccountingOverflow)?;

            accounting.dual_scale_outlier_count = accounting
                .dual_scale_outlier_count
                .checked_add(outlier_count)
                .ok_or(BindError::AccountingOverflow)?;

            add_payload_bytes(accounting, size_of_val(tensor.blocks()))?;

            add_payload_bytes(accounting, size_of_val(tensor.outliers().row_ptr()))?;

            add_payload_bytes(accounting, size_of_val(tensor.outliers().col_idx()))?;

            add_payload_bytes(accounting, size_of_val(tensor.outliers().delta()))?;
        }
        CpuWeightResource::Q8_0HotCpuPortable { tensor, .. } => {
            accounting.q8_0_hot_block_count = accounting
                .q8_0_hot_block_count
                .checked_add(tensor.blocks().len())
                .ok_or(BindError::AccountingOverflow)?;
            add_payload_bytes(accounting, size_of_val(tensor.blocks()))?;
        }
        CpuWeightResource::Q2KHotCpuPortable { tensor, .. } => {
            accounting.q2k_hot_block_count = accounting
                .q2k_hot_block_count
                .checked_add(tensor.blocks().len())
                .ok_or(BindError::AccountingOverflow)?;
            add_payload_bytes(accounting, size_of_val(tensor.blocks()))?;
        }
        CpuWeightResource::Iq2XxsHotCpuPortable { tensor, .. } => {
            accounting.iq2_xxs_hot_block_count = accounting
                .iq2_xxs_hot_block_count
                .checked_add(tensor.blocks().len())
                .ok_or(BindError::AccountingOverflow)?;
            add_payload_bytes(accounting, size_of_val(tensor.blocks()))?;
        }
        CpuWeightResource::Mxfp4HotCpuPortable { tensor, .. } => {
            accounting.mxfp4_hot_block_count = accounting
                .mxfp4_hot_block_count
                .checked_add(tensor.blocks().len())
                .ok_or(BindError::AccountingOverflow)?;
            add_payload_bytes(accounting, size_of_val(tensor.blocks()))?;
        }
        CpuWeightResource::Fp8ScaleGridHotCpuPortable { tensor, .. } => {
            accounting.fp8_scale_grid_value_count = accounting
                .fp8_scale_grid_value_count
                .checked_add(tensor.values().len())
                .ok_or(BindError::AccountingOverflow)?;
            accounting.fp8_scale_grid_scale_count = accounting
                .fp8_scale_grid_scale_count
                .checked_add(tensor.scales().len())
                .ok_or(BindError::AccountingOverflow)?;
            add_payload_bytes(accounting, size_of_val(tensor.values()))?;
            add_payload_bytes(accounting, size_of_val(tensor.scales()))?;
        }
    }

    Ok(())
}

fn add_payload_bytes(
    accounting: &mut CpuWeightBindingAccounting,
    bytes: usize,
) -> Result<(), BindError> {
    accounting.hot_weight_payload_bytes = accounting
        .hot_weight_payload_bytes
        .checked_add(bytes)
        .ok_or(BindError::AccountingOverflow)?;

    Ok(())
}

fn validate_requirement(
    requirement: &CpuWeightRequirement,
    resource: &CpuWeightResource,
) -> Result<(), BindError> {
    let actual_kind = resource.kind();

    if requirement.kind != actual_kind {
        return Err(BindError::WrongResourceKind {
            storage: requirement.storage,
            expected: requirement.kind,
            actual: actual_kind,
        });
    }

    let actual_format = resource.quant_format();

    if requirement.quant_format != actual_format {
        return Err(BindError::FormatMismatch {
            storage: requirement.storage,
            expected: requirement.quant_format,
            actual: actual_format,
        });
    }

    let actual_version = resource.format_version();

    if requirement.format_version != actual_version {
        return Err(BindError::VersionMismatch {
            storage: requirement.storage,
            expected: requirement.format_version,
            actual: actual_version,
        });
    }

    let actual_layout = resource.physical_layout();

    if requirement.physical_layout != actual_layout {
        return Err(BindError::LayoutMismatch {
            storage: requirement.storage,
            expected: requirement.physical_layout,
            actual: actual_layout,
        });
    }

    if requirement.rows != resource.rows() || requirement.cols != resource.cols() {
        return Err(BindError::ShapeMismatch {
            storage: requirement.storage,
            expected_rows: requirement.rows,
            expected_cols: requirement.cols,
            actual_rows: resource.rows(),
            actual_cols: resource.cols(),
        });
    }

    match resource {
        CpuWeightResource::Q4KHotCpuPortable { tensor, .. } => {
            validate_q4k_tensor(requirement, tensor)
        }

        CpuWeightResource::DualScaleHotCpuPortable { tensor, .. } => {
            validate_dual_scale_tensor(requirement, tensor)
        }
        CpuWeightResource::Q8_0HotCpuPortable { tensor, .. } => validate_simple_tensor(
            requirement,
            Q8_0HotCpuBlock::DECODED_LEN,
            tensor.blocks_per_row(),
            tensor.blocks().len(),
        ),
        CpuWeightResource::Q2KHotCpuPortable { tensor, .. } => validate_simple_tensor(
            requirement,
            Q2KHotCpuBlock::DECODED_LEN,
            tensor.blocks_per_row(),
            tensor.blocks().len(),
        ),
        CpuWeightResource::Iq2XxsHotCpuPortable { tensor, .. } => validate_simple_tensor(
            requirement,
            Iq2XxsHotCpuBlock::DECODED_LEN,
            tensor.blocks_per_row(),
            tensor.blocks().len(),
        ),
        CpuWeightResource::Mxfp4HotCpuPortable { tensor, .. } => validate_simple_tensor(
            requirement,
            augur_kernel_api::MXFP4_HOT_CPU_BLOCK_LEN,
            tensor.blocks_per_row(),
            tensor.blocks().len(),
        ),
        CpuWeightResource::Fp8ScaleGridHotCpuPortable { tensor, .. } => {
            validate_fp8_scale_grid_tensor(requirement, tensor)
        }
    }
}

fn validate_fp8_scale_grid_tensor(
    requirement: &CpuWeightRequirement,
    tensor: &Fp8E4M3ScaleGridHotCpuTensor,
) -> Result<(), BindError> {
    if requirement.block_size != FP8_E4M3_SCALE_TILE {
        return Err(BindError::BlockSizeMismatch {
            storage: requirement.storage,
            expected: requirement.block_size,
            actual: FP8_E4M3_SCALE_TILE,
        });
    }
    let expected_values =
        requirement
            .rows
            .checked_mul(requirement.cols)
            .ok_or(BindError::InvalidWeightShape {
                storage: requirement.storage,
            })?;
    let expected_scale_rows = requirement.rows.div_ceil(FP8_E4M3_SCALE_TILE);
    if requirement.rows == 0
        || requirement.cols == 0
        || !requirement.cols.is_multiple_of(FP8_E4M3_SCALE_TILE)
        || tensor.values().len() != expected_values
        || tensor.scale_rows() != expected_scale_rows
        || tensor.scale_cols() != requirement.cols / FP8_E4M3_SCALE_TILE
        || tensor.scales().len() != tensor.scale_rows() * tensor.scale_cols()
    {
        return Err(BindError::InvalidWeightShape {
            storage: requirement.storage,
        });
    }
    Ok(())
}

fn validate_simple_tensor(
    requirement: &CpuWeightRequirement,
    decoded_len: usize,
    blocks_per_row: usize,
    block_count: usize,
) -> Result<(), BindError> {
    if requirement.block_size != decoded_len {
        return Err(BindError::BlockSizeMismatch {
            storage: requirement.storage,
            expected: requirement.block_size,
            actual: decoded_len,
        });
    }
    if requirement.rows == 0
        || requirement.cols == 0
        || !requirement.cols.is_multiple_of(decoded_len)
        || blocks_per_row != requirement.cols / decoded_len
        || block_count != requirement.rows.saturating_mul(blocks_per_row)
    {
        return Err(BindError::InvalidWeightShape {
            storage: requirement.storage,
        });
    }
    Ok(())
}

fn validate_q4k_tensor(
    requirement: &CpuWeightRequirement,
    tensor: &Q4KHotCpuTensor,
) -> Result<(), BindError> {
    if requirement.block_size != Q4KHotCpuBlock::DECODED_LEN {
        return Err(BindError::BlockSizeMismatch {
            storage: requirement.storage,
            expected: requirement.block_size,
            actual: Q4KHotCpuBlock::DECODED_LEN,
        });
    }

    if !requirement.cols.is_multiple_of(Q4KHotCpuBlock::DECODED_LEN) {
        return Err(BindError::InvalidWeightShape {
            storage: requirement.storage,
        });
    }

    let expected_blocks_per_row = requirement.cols / Q4KHotCpuBlock::DECODED_LEN;

    if tensor.blocks_per_row() != expected_blocks_per_row {
        return Err(BindError::InvalidWeightShape {
            storage: requirement.storage,
        });
    }

    let expected_blocks = requirement
        .rows
        .checked_mul(expected_blocks_per_row)
        .ok_or(BindError::InvalidWeightShape {
            storage: requirement.storage,
        })?;

    if tensor.blocks().len() != expected_blocks {
        return Err(BindError::InvalidWeightShape {
            storage: requirement.storage,
        });
    }

    Ok(())
}

/// Validates mutable/public DualScale shape metadata and immutable CSR
/// reachability before any owner is taken from the resource store.
///
/// Deep O(nnz) CSR validation belongs to `OutlierCsrHotCpu::new`. Binding
/// repeats only:
///
/// - tensor shape consistency;
/// - tensor/CSR cross-shape consistency;
/// - O(rows) checked CSR row reachability.
///
/// This mirrors the static part of M26-D's strong-error preflight without
/// moving kernel execution or the validated CSR token into `augur-exec`.
fn validate_dual_scale_tensor(
    requirement: &CpuWeightRequirement,
    tensor: &DualScaleHotCpuTensor,
) -> Result<(), BindError> {
    if requirement.block_size != DualScaleHotCpuBlock::DECODED_LEN {
        return Err(BindError::BlockSizeMismatch {
            storage: requirement.storage,
            expected: requirement.block_size,
            actual: DualScaleHotCpuBlock::DECODED_LEN,
        });
    }

    if requirement.rows == 0
        || requirement.cols == 0
        || !requirement
            .cols
            .is_multiple_of(DualScaleHotCpuBlock::DECODED_LEN)
    {
        return Err(BindError::InvalidWeightShape {
            storage: requirement.storage,
        });
    }

    let expected_blocks_per_row = requirement.cols / DualScaleHotCpuBlock::DECODED_LEN;

    if expected_blocks_per_row == 0 || tensor.blocks_per_row() != expected_blocks_per_row {
        return Err(BindError::InvalidWeightShape {
            storage: requirement.storage,
        });
    }

    let expected_blocks = requirement
        .rows
        .checked_mul(expected_blocks_per_row)
        .ok_or(BindError::InvalidWeightShape {
            storage: requirement.storage,
        })?;

    if tensor.blocks().len() != expected_blocks {
        return Err(BindError::InvalidWeightShape {
            storage: requirement.storage,
        });
    }

    let outliers = tensor.outliers();

    if outliers.rows() != requirement.rows || outliers.cols() != requirement.cols {
        return Err(BindError::InvalidWeightShape {
            storage: requirement.storage,
        });
    }

    // Cross-shape and validate_rows() already prove row reachability.
    // No additional ExactSizeIterator-based length assertion is needed.
    outliers
        .validate_rows()
        .ok_or(BindError::InvalidWeightShape {
            storage: requirement.storage,
        })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use augur_core::F16Bits;
    use augur_kernel_api::{
        OutlierCsrHotCpu, DUAL_SCALE_HOT_CPU_GROUPS, DUAL_SCALE_HOT_CPU_MEMBERSHIP_BYTES,
        DUAL_SCALE_HOT_CPU_QS_BYTES,
    };

    use crate::{dual_scale_requirement, q4k_requirement, WeightRequirementBuilder};

    fn q4k_tensor() -> Q4KHotCpuTensor {
        let block = Q4KHotCpuBlock {
            d: F16Bits::from_f32(1.0),
            dmin: F16Bits::from_f32(0.5),
            scale: [1; 8],
            min: [1; 8],
            qs: [0; 128],
        };

        Q4KHotCpuTensor::new(1, Q4KHotCpuBlock::DECODED_LEN, vec![block]).unwrap()
    }

    fn dual_scale_tensor() -> DualScaleHotCpuTensor {
        let rows = 1;
        let cols = DualScaleHotCpuBlock::DECODED_LEN;

        let block = DualScaleHotCpuBlock {
            scale_lo: [F16Bits::from_f32(1.0); DUAL_SCALE_HOT_CPU_GROUPS],
            scale_hi: [F16Bits::from_f32(2.0); DUAL_SCALE_HOT_CPU_GROUPS],
            membership: [0; DUAL_SCALE_HOT_CPU_MEMBERSHIP_BYTES],
            qs: [0; DUAL_SCALE_HOT_CPU_QS_BYTES],
        };

        let outliers =
            OutlierCsrHotCpu::new(rows, cols, vec![0, 0], Vec::new(), Vec::new()).unwrap();

        DualScaleHotCpuTensor::new(rows, cols, vec![block], outliers).unwrap()
    }

    fn dual_scale_requirements(
        storage: StorageId,
    ) -> (WeightBindingId, Box<[CpuWeightRequirement]>) {
        let mut builder = WeightRequirementBuilder::new();

        let binding = builder
            .intern(dual_scale_requirement(
                storage,
                FormatVersion(1),
                DualScaleHotCpuBlock::DECODED_LEN,
                1,
                DualScaleHotCpuBlock::DECODED_LEN,
            ))
            .unwrap();

        (binding, builder.finish())
    }

    #[test]
    fn stale_generation_does_not_consume_valid_resource() {
        let mut resources = CpuWeightResources::new();

        let valid = resources.add_q4k_hot_cpu_portable(q4k_tensor()).unwrap();

        let stale = StorageId {
            index: valid.index,
            generation: valid.generation + 1,
        };

        let error = resources.take(stale).unwrap_err();

        assert_eq!(
            error,
            ResourceLookupError::GenerationMismatch {
                requested: stale,
                actual: valid,
            }
        );

        assert!(resources.take(valid).is_ok());
    }

    #[test]
    fn taking_same_resource_twice_is_diagnostic() {
        let mut resources = CpuWeightResources::new();

        let id = resources.add_q4k_hot_cpu_portable(q4k_tensor()).unwrap();

        assert!(resources.take(id).is_ok());

        let error = resources.take(id).unwrap_err();

        assert_eq!(error, ResourceLookupError::AlreadyTaken { requested: id });
    }

    #[test]
    fn stale_generation_after_take_is_still_generation_mismatch() {
        let mut resources = CpuWeightResources::new();

        let valid = resources.add_q4k_hot_cpu_portable(q4k_tensor()).unwrap();

        assert!(resources.take(valid).is_ok());

        let stale = StorageId {
            index: valid.index,
            generation: valid.generation + 1,
        };

        assert_eq!(
            resources.take(stale),
            Err(ResourceLookupError::GenerationMismatch {
                requested: stale,
                actual: valid,
            }),
        );

        assert_eq!(
            resources.take(valid),
            Err(ResourceLookupError::AlreadyTaken { requested: valid }),
        );
    }

    #[test]
    fn binds_dual_scale_owner_into_arc() {
        let tensor = dual_scale_tensor();

        let expected_payload_bytes = size_of_val(tensor.blocks())
            + size_of_val(tensor.outliers().row_ptr())
            + size_of_val(tensor.outliers().col_idx())
            + size_of_val(tensor.outliers().delta());

        let mut resources = CpuWeightResources::new();

        let storage = resources.add_dual_scale_hot_cpu_portable(tensor).unwrap();

        let (binding, requirements) = dual_scale_requirements(storage);

        let bound = bind_cpu_weight_resources(&requirements, resources).unwrap();

        let owner = bound.dual_scale_hot_cpu_portable(binding).unwrap();

        assert_eq!(owner.rows(), 1);
        assert_eq!(owner.cols(), DualScaleHotCpuBlock::DECODED_LEN);
        assert_eq!(owner.blocks_per_row(), 1);

        assert_eq!(
            bound.accounting(),
            CpuWeightBindingAccounting {
                unique_weight_count: 1,
                q4k_hot_block_count: 0,
                dual_scale_hot_block_count: 1,
                dual_scale_outlier_count: 0,
                q8_0_hot_block_count: 0,
                q2k_hot_block_count: 0,
                iq2_xxs_hot_block_count: 0,
                mxfp4_hot_block_count: 0,
                fp8_scale_grid_value_count: 0,
                fp8_scale_grid_scale_count: 0,
                hot_weight_payload_bytes: expected_payload_bytes,
            }
        );
    }

    #[test]
    fn dual_scale_version_mismatch_is_preflighted_before_take() {
        let mut resources = CpuWeightResources::new();

        let storage = resources
            .add_dual_scale_hot_cpu_portable_versioned(FormatVersion(2), dual_scale_tensor())
            .unwrap();

        let mut builder = WeightRequirementBuilder::new();

        builder
            .intern(dual_scale_requirement(
                storage,
                FormatVersion(1),
                DualScaleHotCpuBlock::DECODED_LEN,
                1,
                DualScaleHotCpuBlock::DECODED_LEN,
            ))
            .unwrap();

        let requirements = builder.finish();

        assert!(matches!(
            preflight_requirements(&requirements, &resources),
            Err(BindError::VersionMismatch {
                storage: actual,
                expected: FormatVersion(1),
                actual: FormatVersion(2),
            }) if actual == storage
        ));

        // Preflight failure must not consume the owner.
        assert!(resources.take(storage).is_ok());
    }

    #[test]
    fn wrong_resource_kind_is_preflighted_before_take() {
        let mut resources = CpuWeightResources::new();

        let storage = resources.add_q4k_hot_cpu_portable(q4k_tensor()).unwrap();

        let mut builder = WeightRequirementBuilder::new();

        builder
            .intern(dual_scale_requirement(
                storage,
                FormatVersion(1),
                DualScaleHotCpuBlock::DECODED_LEN,
                1,
                DualScaleHotCpuBlock::DECODED_LEN,
            ))
            .unwrap();

        let requirements = builder.finish();

        assert!(matches!(
            preflight_requirements(&requirements, &resources),
            Err(BindError::WrongResourceKind {
                storage: actual,
                expected: CpuWeightResourceKind::DualScaleHotCpuPortable,
                actual: CpuWeightResourceKind::Q4KHotCpuPortable,
            }) if actual == storage
        ));

        assert!(resources.take(storage).is_ok());
    }

    #[test]
    fn q4k_requirement_remains_operational() {
        let mut resources = CpuWeightResources::new();

        let storage = resources.add_q4k_hot_cpu_portable(q4k_tensor()).unwrap();

        let mut builder = WeightRequirementBuilder::new();

        let binding = builder
            .intern(q4k_requirement(
                storage,
                FormatVersion(1),
                Q4KHotCpuBlock::DECODED_LEN,
                1,
                Q4KHotCpuBlock::DECODED_LEN,
            ))
            .unwrap();

        let requirements = builder.finish();

        let bound = bind_cpu_weight_resources(&requirements, resources).unwrap();

        assert!(bound.q4k_hot_cpu_portable(binding).is_some());
        assert!(bound.dual_scale_hot_cpu_portable(binding).is_none());
    }
}
