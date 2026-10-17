use core::fmt;

use augur_cuda_sys::{AugurCudaM68Bf16InterstageArgsV1, AugurCudaNvfp4W4a16ArgsV1};
use sha2::{Digest, Sha256};

use crate::cuda_nvfp4_family::{CudaNvfp4ExpertShapeFamilyV1, CudaNvfp4FamilyHostInputsV1};
use crate::{
    cuda_device_properties, CudaComputeCapability, CudaRuntimeError, CudaRuntimeErrorKind,
    OwnedCudaBuffer, OwnedCudaStream,
};

pub const M68_PLAN_CONTENT_SHA256: &str =
    "9f8548b9112d83b3a7cd4b6676ed1268a20312089fcb351f51e033aa5c346073";
pub const M68_CAPACITIES: [usize; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

const LAYER4_EXPERTS: [u16; 52] = [
    3, 5, 13, 14, 25, 27, 28, 40, 50, 51, 52, 58, 63, 64, 71, 76, 82, 85, 86, 87, 91, 93, 98, 101,
    102, 107, 108, 117, 120, 121, 131, 140, 144, 153, 163, 168, 170, 171, 174, 182, 183, 184, 187,
    199, 205, 208, 217, 228, 238, 242, 243, 255,
];
const LAYER4_W13_SCALE_BITS: [u32; 52] = [
    973727646, 973927375, 973378121, 979369984, 973727646, 982415848, 981766729, 973378121,
    981616933, 981167543, 981916526, 981666865, 981616933, 973128460, 972279613, 980668221,
    974975951, 974276901, 981467136, 981517068, 981766729, 974526562, 973428053, 974676358,
    973328189, 974526562, 980967814, 973877443, 982016390, 974576494, 973278257, 972878799,
    973128460, 973477986, 973927375, 974326833, 982116254, 973378121, 973677714, 973977307,
    981567000, 974676358, 973627782, 972778935, 980568357, 981467136, 973078528, 979869306,
    973677714, 973228325, 973378121, 982116254,
];
const LAYER4_W2_SCALE_BITS: [u32; 52] = [
    982166187, 975575138, 992602014, 976573781, 986160762, 993950184, 993201201, 979469848,
    987758592, 988657371, 992552082, 991004184, 988257914, 1030101090, 985861169, 985611508,
    983314627, 990305134, 978670933, 987858456, 998893470, 1002288859, 991603371, 983464424,
    982815305, 986060898, 974926019, 980967814, 995647878, 983913813, 993001472, 987758592,
    982415848, 1003437300, 986560219, 982415848, 996646522, 994749099, 983813949, 980568357,
    981517068, 991104049, 994549370, 978371340, 992901608, 981616933, 989356422, 971680427,
    977872018, 989356422, 976374053, 991503506,
];

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum M68StageV1 {
    W13,
    W2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum M68KernelFamilyV1 {
    DirectPackedW4A16,
    StaticScaleBoundW4A4,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct M68ShapeV1 {
    pub stage: M68StageV1,
    pub capacity: usize,
    pub n: usize,
    pub k: usize,
}

impl M68ShapeV1 {
    pub fn new(stage: M68StageV1, capacity: usize) -> Result<Self, M68BindError> {
        if !M68_CAPACITIES.contains(&capacity) {
            return Err(M68BindError::Shape);
        }
        let (n, k) = match stage {
            M68StageV1::W13 => (2048, 4096),
            M68StageV1::W2 => (4096, 2048),
        };
        Ok(Self {
            stage,
            capacity,
            n,
            k,
        })
    }

    pub const fn packed_weight_bytes(self) -> usize {
        self.n * self.k / 2
    }

    pub const fn logical_scale_bytes(self) -> usize {
        self.n * self.k / 16
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct M68RequirementV1<'a> {
    pub plan_content_sha256: &'a str,
    pub layer: u16,
    pub expert: u16,
    pub stage: M68StageV1,
    pub kernel: M68KernelFamilyV1,
    pub shape: M68ShapeV1,
    pub packed_weight_sha256: [u8; 32],
    pub input_scale_bits: Option<u32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct M68PackedWeightOwnerV1 {
    pub packed_values_len_bytes: usize,
    pub logical_scales_len_bytes: usize,
    pub packed_weight_sha256: [u8; 32],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct M68BoundRequirementV1<'a> {
    requirement: M68RequirementV1<'a>,
    owner: &'a M68PackedWeightOwnerV1,
}

impl<'a> M68BoundRequirementV1<'a> {
    pub fn requirement(&self) -> M68RequirementV1<'a> {
        self.requirement
    }

    pub fn owner(&self) -> &'a M68PackedWeightOwnerV1 {
        self.owner
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum M68BindError {
    PlanIdentity,
    Layer,
    Expert,
    Stage,
    KernelFamily,
    Shape,
    WeightIdentity,
    WeightExtent,
    Scale,
    KernelUnavailable,
}

impl fmt::Display for M68BindError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "M68 binding rejected: {self:?}")
    }
}

impl std::error::Error for M68BindError {}

pub fn m68_layer4_scale_bits(expert: u16, stage: M68StageV1) -> Option<u32> {
    let index = LAYER4_EXPERTS.binary_search(&expert).ok()?;
    Some(match stage {
        M68StageV1::W13 => LAYER4_W13_SCALE_BITS[index],
        M68StageV1::W2 => LAYER4_W2_SCALE_BITS[index],
    })
}

pub fn bind_m68_requirement<'a>(
    requirement: M68RequirementV1<'a>,
    owner: &'a M68PackedWeightOwnerV1,
    kernel_available: bool,
) -> Result<M68BoundRequirementV1<'a>, M68BindError> {
    if requirement.plan_content_sha256 != M68_PLAN_CONTENT_SHA256 {
        return Err(M68BindError::PlanIdentity);
    }
    if requirement.shape.stage != requirement.stage {
        return Err(M68BindError::Stage);
    }
    if !M68_CAPACITIES.contains(&requirement.shape.capacity) {
        return Err(M68BindError::Shape);
    }
    if owner.packed_weight_sha256 != requirement.packed_weight_sha256 {
        return Err(M68BindError::WeightIdentity);
    }
    if owner.packed_values_len_bytes != requirement.shape.packed_weight_bytes()
        || owner.logical_scales_len_bytes != requirement.shape.logical_scale_bytes()
    {
        return Err(M68BindError::WeightExtent);
    }
    if !kernel_available {
        return Err(M68BindError::KernelUnavailable);
    }
    match requirement.layer {
        3 => {
            if requirement.expert > 255 {
                return Err(M68BindError::Expert);
            }
            if requirement.kernel != M68KernelFamilyV1::DirectPackedW4A16 {
                return Err(M68BindError::KernelFamily);
            }
            if requirement.input_scale_bits.is_some() {
                return Err(M68BindError::Scale);
            }
        }
        4 => {
            if requirement.kernel != M68KernelFamilyV1::StaticScaleBoundW4A4 {
                return Err(M68BindError::KernelFamily);
            }
            let expected = m68_layer4_scale_bits(requirement.expert, requirement.stage)
                .ok_or(M68BindError::Expert)?;
            if requirement.input_scale_bits != Some(expected) {
                return Err(M68BindError::Scale);
            }
        }
        _ => return Err(M68BindError::Layer),
    }
    Ok(M68BoundRequirementV1 { requirement, owner })
}

fn cuda_validation(kind: CudaRuntimeErrorKind) -> CudaRuntimeError {
    CudaRuntimeError::validation("cuda_m68_w4a16", kind)
}

fn upload(stream: &OwnedCudaStream, bytes: &[u8]) -> Result<OwnedCudaBuffer, CudaRuntimeError> {
    let buffer = OwnedCudaBuffer::allocate(stream.device_ordinal(), bytes.len())?;
    buffer.copy_from_host_sync(stream, bytes)?;
    Ok(buffer)
}

fn address(buffer: &OwnedCudaBuffer) -> Result<u64, CudaRuntimeError> {
    buffer
        .device_ptr()
        .map(|pointer| pointer.raw_value())
        .ok_or_else(|| cuda_validation(CudaRuntimeErrorKind::InvalidLayout))
}

fn raw_shape(shape: M68ShapeV1) -> u32 {
    match (shape.stage, shape.capacity) {
        (M68StageV1::W13, 1) => augur_cuda_sys::AUGUR_CUDA_NVFP4_AOT_SHAPE_W13_M1_V1,
        (M68StageV1::W13, 2) => augur_cuda_sys::AUGUR_CUDA_NVFP4_AOT_SHAPE_W13_M2_V1,
        (M68StageV1::W13, 4) => augur_cuda_sys::AUGUR_CUDA_NVFP4_AOT_SHAPE_W13_M4_V1,
        (M68StageV1::W13, 8) => augur_cuda_sys::AUGUR_CUDA_NVFP4_AOT_SHAPE_W13_M8_V1,
        (M68StageV1::W13, 16) => augur_cuda_sys::AUGUR_CUDA_NVFP4_AOT_SHAPE_W13_M16_V1,
        (M68StageV1::W13, 32) => augur_cuda_sys::AUGUR_CUDA_NVFP4_AOT_SHAPE_W13_M32_V1,
        (M68StageV1::W13, 64) => augur_cuda_sys::AUGUR_CUDA_NVFP4_AOT_SHAPE_W13_M64_V1,
        (M68StageV1::W13, 128) => augur_cuda_sys::AUGUR_CUDA_NVFP4_AOT_SHAPE_W13_M128_V1,
        (M68StageV1::W2, 1) => augur_cuda_sys::AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M1_V1,
        (M68StageV1::W2, 2) => augur_cuda_sys::AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M2_V1,
        (M68StageV1::W2, 4) => augur_cuda_sys::AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M4_V1,
        (M68StageV1::W2, 8) => augur_cuda_sys::AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M8_V1,
        (M68StageV1::W2, 16) => augur_cuda_sys::AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M16_V1,
        (M68StageV1::W2, 32) => augur_cuda_sys::AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M32_V1,
        (M68StageV1::W2, 64) => augur_cuda_sys::AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M64_V1,
        (M68StageV1::W2, 128) => augur_cuda_sys::AUGUR_CUDA_NVFP4_AOT_SHAPE_W2_M128_V1,
        _ => 0,
    }
}

pub struct M68W4A16WeightHostV1<'a> {
    pub binding: M68BoundRequirementV1<'a>,
    pub packed_values: &'a [u8],
    pub logical_scales: &'a [u8],
    pub weight_global_scale: f32,
}

struct M68DeviceWeightV1 {
    values: OwnedCudaBuffer,
    scales: OwnedCudaBuffer,
}

pub struct CudaM68W4A16ExpertV1 {
    device_ordinal: i32,
    capacity: usize,
    valid_rows: usize,
    activation: OwnedCudaBuffer,
    route_weights: OwnedCudaBuffer,
    w1: M68DeviceWeightV1,
    w3: M68DeviceWeightV1,
    w2: M68DeviceWeightV1,
    gate: OwnedCudaBuffer,
    up: OwnedCudaBuffer,
    intermediate: OwnedCudaBuffer,
    output: OwnedCudaBuffer,
    w1_args: AugurCudaNvfp4W4a16ArgsV1,
    w3_args: AugurCudaNvfp4W4a16ArgsV1,
    interstage_args: AugurCudaM68Bf16InterstageArgsV1,
    w2_args: AugurCudaNvfp4W4a16ArgsV1,
}

fn upload_weight(
    stream: &OwnedCudaStream,
    input: &M68W4A16WeightHostV1<'_>,
) -> Result<M68DeviceWeightV1, CudaRuntimeError> {
    let requirement = input.binding.requirement();
    if input.packed_values.len() != requirement.shape.packed_weight_bytes()
        || input.logical_scales.len() != requirement.shape.logical_scale_bytes()
        || !input.weight_global_scale.is_finite()
        || input.weight_global_scale <= 0.0
    {
        return Err(cuda_validation(CudaRuntimeErrorKind::InvalidDimensions));
    }
    if host_weight_sha256(input.packed_values, input.logical_scales)
        != requirement.packed_weight_sha256
    {
        return Err(cuda_validation(CudaRuntimeErrorKind::InvalidMetadata));
    }
    Ok(M68DeviceWeightV1 {
        values: upload(stream, input.packed_values)?,
        scales: upload(stream, input.logical_scales)?,
    })
}

fn host_weight_sha256(values: &[u8], scales: &[u8]) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(values);
    digest.update(scales);
    digest.finalize().into()
}

fn gemm_args(
    activation: &OwnedCudaBuffer,
    weight: &M68DeviceWeightV1,
    output: &OwnedCudaBuffer,
    weight_global_scale: f32,
) -> Result<AugurCudaNvfp4W4a16ArgsV1, CudaRuntimeError> {
    let mut args = AugurCudaNvfp4W4a16ArgsV1::new();
    args.activation_bf16 = address(activation)?;
    args.activation_len_bytes = activation.len_bytes() as u64;
    args.weight_values = address(&weight.values)?;
    args.weight_values_len_bytes = weight.values.len_bytes() as u64;
    args.weight_scales = address(&weight.scales)?;
    args.weight_scales_len_bytes = weight.scales.len_bytes() as u64;
    args.output_bf16 = address(output)?;
    args.output_len_bytes = output.len_bytes() as u64;
    args.weight_global_scale = weight_global_scale;
    Ok(args)
}

impl CudaM68W4A16ExpertV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn bind_host(
        stream: &OwnedCudaStream,
        valid_rows: usize,
        activation_bf16: &[u8],
        route_weights: &[f32],
        swiglu_limit: f32,
        w1_input: M68W4A16WeightHostV1<'_>,
        w3_input: M68W4A16WeightHostV1<'_>,
        w2_input: M68W4A16WeightHostV1<'_>,
    ) -> Result<Self, CudaRuntimeError> {
        if cuda_device_properties(stream.device_ordinal())?.compute_capability
            != CudaComputeCapability::SM120
        {
            return Err(cuda_validation(CudaRuntimeErrorKind::UnsupportedCapability));
        }
        // SAFETY: Pure build-capability query with no pointer arguments.
        if unsafe { augur_cuda_sys::augur_cuda_nvfp4_sm120_build_capability_v1() } != 1 {
            return Err(cuda_validation(CudaRuntimeErrorKind::UnsupportedCapability));
        }
        let w1_requirement = w1_input.binding.requirement();
        let w3_requirement = w3_input.binding.requirement();
        let w2_requirement = w2_input.binding.requirement();
        let capacity = w1_requirement.shape.capacity;
        if w1_requirement.layer != 3
            || w3_requirement.layer != 3
            || w2_requirement.layer != 3
            || w1_requirement.kernel != M68KernelFamilyV1::DirectPackedW4A16
            || w3_requirement.kernel != M68KernelFamilyV1::DirectPackedW4A16
            || w2_requirement.kernel != M68KernelFamilyV1::DirectPackedW4A16
            || w1_requirement.expert != w3_requirement.expert
            || w1_requirement.expert != w2_requirement.expert
            || w1_requirement.stage != M68StageV1::W13
            || w3_requirement.stage != M68StageV1::W13
            || w2_requirement.stage != M68StageV1::W2
            || w3_requirement.shape.capacity != capacity
            || w2_requirement.shape.capacity != capacity
            || valid_rows == 0
            || valid_rows > capacity
            || activation_bf16.len() != capacity * 4096 * 2
            || route_weights.len() != capacity
            || route_weights
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
            || !swiglu_limit.is_finite()
            || swiglu_limit <= 0.0
        {
            return Err(cuda_validation(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let activation = upload(stream, activation_bf16)?;
        let mut route_bytes = Vec::with_capacity(capacity * 4);
        for value in route_weights {
            route_bytes.extend_from_slice(&value.to_ne_bytes());
        }
        let route_weights = upload(stream, &route_bytes)?;
        let w1 = upload_weight(stream, &w1_input)?;
        let w3 = upload_weight(stream, &w3_input)?;
        let w2 = upload_weight(stream, &w2_input)?;
        let gate = OwnedCudaBuffer::allocate(stream.device_ordinal(), capacity * 2048 * 2)?;
        let up = OwnedCudaBuffer::allocate(stream.device_ordinal(), capacity * 2048 * 2)?;
        let intermediate = OwnedCudaBuffer::allocate(stream.device_ordinal(), capacity * 2048 * 2)?;
        let output = OwnedCudaBuffer::allocate(stream.device_ordinal(), capacity * 4096 * 2)?;
        let w1_args = gemm_args(&activation, &w1, &gate, w1_input.weight_global_scale)?;
        let w3_args = gemm_args(&activation, &w3, &up, w3_input.weight_global_scale)?;
        let w2_args = gemm_args(&intermediate, &w2, &output, w2_input.weight_global_scale)?;
        let mut interstage_args = AugurCudaM68Bf16InterstageArgsV1::new();
        interstage_args.gate_bf16 = address(&gate)?;
        interstage_args.gate_len_bytes = gate.len_bytes() as u64;
        interstage_args.up_bf16 = address(&up)?;
        interstage_args.up_len_bytes = up.len_bytes() as u64;
        interstage_args.route_weights_f32 = address(&route_weights)?;
        interstage_args.route_weights_len_bytes = route_weights.len_bytes() as u64;
        interstage_args.output_bf16 = address(&intermediate)?;
        interstage_args.output_len_bytes = intermediate.len_bytes() as u64;
        interstage_args.capacity = capacity as u32;
        interstage_args.valid_rows = valid_rows as u32;
        interstage_args.swiglu_limit = swiglu_limit;
        Ok(Self {
            device_ordinal: stream.device_ordinal(),
            capacity,
            valid_rows,
            activation,
            route_weights,
            w1,
            w3,
            w2,
            gate,
            up,
            intermediate,
            output,
            w1_args,
            w3_args,
            interstage_args,
            w2_args,
        })
    }

    pub fn execute(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.device_ordinal {
            return Err(cuda_validation(CudaRuntimeErrorKind::DeviceMismatch));
        }
        let w13_shape = raw_shape(
            M68ShapeV1::new(M68StageV1::W13, self.capacity)
                .map_err(|_| cuda_validation(CudaRuntimeErrorKind::InvalidDimensions))?,
        );
        let w2_shape = raw_shape(
            M68ShapeV1::new(M68StageV1::W2, self.capacity)
                .map_err(|_| cuda_validation(CudaRuntimeErrorKind::InvalidDimensions))?,
        );
        // SAFETY: Arguments contain addresses owned by self and validated finite shapes.
        unsafe {
            CudaRuntimeError::from_result(
                "cuda_m68_w1",
                augur_cuda_sys::augur_cuda_nvfp4_w4a16_submit_v1(
                    self.device_ordinal,
                    stream.raw()?,
                    w13_shape,
                    &self.w1_args,
                ),
            )?;
            CudaRuntimeError::from_result(
                "cuda_m68_w3",
                augur_cuda_sys::augur_cuda_nvfp4_w4a16_submit_v1(
                    self.device_ordinal,
                    stream.raw()?,
                    w13_shape,
                    &self.w3_args,
                ),
            )?;
            CudaRuntimeError::from_result(
                "cuda_m68_interstage",
                augur_cuda_sys::augur_cuda_m68_bf16_interstage_submit_v1(
                    self.device_ordinal,
                    stream.raw()?,
                    &self.interstage_args,
                ),
            )?;
            CudaRuntimeError::from_result(
                "cuda_m68_w2",
                augur_cuda_sys::augur_cuda_nvfp4_w4a16_v1(
                    self.device_ordinal,
                    stream.raw()?,
                    w2_shape,
                    &self.w2_args,
                ),
            )?;
        }
        Ok(())
    }

    pub fn copy_output_to_host_sync(
        &self,
        stream: &OwnedCudaStream,
        destination: &mut [u8],
    ) -> Result<(), CudaRuntimeError> {
        if destination.len() != self.output.len_bytes() {
            return Err(cuda_validation(CudaRuntimeErrorKind::OutOfBounds));
        }
        self.output.copy_to_host_sync(stream, destination)
    }

    pub fn owner_addresses(&self) -> [u64; 12] {
        [
            &self.activation,
            &self.route_weights,
            &self.w1.values,
            &self.w1.scales,
            &self.w3.values,
            &self.w3.scales,
            &self.w2.values,
            &self.w2.scales,
            &self.gate,
            &self.up,
            &self.intermediate,
            &self.output,
        ]
        .map(|buffer| buffer.device_ptr().map_or(0, |pointer| pointer.raw_value()))
    }

    pub const fn valid_rows(&self) -> usize {
        self.valid_rows
    }
}

pub struct CudaM68ScaleBoundExpertV1 {
    expert: u16,
    inner: CudaNvfp4ExpertShapeFamilyV1,
}

impl CudaM68ScaleBoundExpertV1 {
    pub fn bind_host(
        stream: &OwnedCudaStream,
        w1_binding: M68BoundRequirementV1<'_>,
        w3_binding: M68BoundRequirementV1<'_>,
        w2_binding: M68BoundRequirementV1<'_>,
        inputs: CudaNvfp4FamilyHostInputsV1<'_>,
    ) -> Result<Self, CudaRuntimeError> {
        let w1 = w1_binding.requirement();
        let w3 = w3_binding.requirement();
        let w2 = w2_binding.requirement();
        if w1.layer != 4
            || w3.layer != 4
            || w2.layer != 4
            || w1.expert != w3.expert
            || w1.expert != w2.expert
            || w1.stage != M68StageV1::W13
            || w3.stage != M68StageV1::W13
            || w2.stage != M68StageV1::W2
            || w1.kernel != M68KernelFamilyV1::StaticScaleBoundW4A4
            || w3.kernel != M68KernelFamilyV1::StaticScaleBoundW4A4
            || w2.kernel != M68KernelFamilyV1::StaticScaleBoundW4A4
            || w1.input_scale_bits != Some(inputs.input_global_scale.to_bits())
            || w3.input_scale_bits != Some(inputs.input_global_scale.to_bits())
            || w2.input_scale_bits != Some(inputs.w2_activation_global_scale.to_bits())
        {
            return Err(cuda_validation(CudaRuntimeErrorKind::InvalidMetadata));
        }
        for (binding, values, scales) in [
            (w1_binding, inputs.w1_values, inputs.w1_scales_logical),
            (w3_binding, inputs.w3_values, inputs.w3_scales_logical),
            (w2_binding, inputs.w2_values, inputs.w2_scales_logical),
        ] {
            if host_weight_sha256(values, scales) != binding.requirement().packed_weight_sha256 {
                return Err(cuda_validation(CudaRuntimeErrorKind::InvalidMetadata));
            }
        }
        Ok(Self {
            expert: w1.expert,
            inner: CudaNvfp4ExpertShapeFamilyV1::bind_host(stream, inputs)?,
        })
    }

    pub fn execute(
        &self,
        stream: &OwnedCudaStream,
        valid_rows: usize,
    ) -> Result<(), CudaRuntimeError> {
        self.inner.execute(stream, valid_rows)
    }

    pub fn copy_output_to_host_sync(
        &self,
        stream: &OwnedCudaStream,
        capacity: usize,
        destination: &mut [u8],
    ) -> Result<(), CudaRuntimeError> {
        self.inner
            .copy_output_to_host_sync(stream, capacity, destination)
    }

    pub fn owner_addresses(&self) -> Vec<u64> {
        let mut result = self.inner.weight_owner_addresses().to_vec();
        result.push(self.inner.workspace_owner_address());
        result.extend(self.inner.arena_owner_addresses().into_iter().flatten());
        result
    }

    pub const fn expert(&self) -> u16 {
        self.expert
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn requirement(layer: u16, expert: u16, stage: M68StageV1) -> M68RequirementV1<'static> {
        let shape = M68ShapeV1::new(stage, 1).unwrap();
        M68RequirementV1 {
            plan_content_sha256: M68_PLAN_CONTENT_SHA256,
            layer,
            expert,
            stage,
            kernel: if layer == 3 {
                M68KernelFamilyV1::DirectPackedW4A16
            } else {
                M68KernelFamilyV1::StaticScaleBoundW4A4
            },
            shape,
            packed_weight_sha256: [7; 32],
            input_scale_bits: if layer == 4 {
                m68_layer4_scale_bits(expert, stage)
            } else {
                None
            },
        }
    }

    fn owner(stage: M68StageV1) -> M68PackedWeightOwnerV1 {
        let shape = M68ShapeV1::new(stage, 1).unwrap();
        M68PackedWeightOwnerV1 {
            packed_values_len_bytes: shape.packed_weight_bytes(),
            logical_scales_len_bytes: shape.logical_scale_bytes(),
            packed_weight_sha256: [7; 32],
        }
    }

    #[test]
    fn selected_requirement_table_is_complete_and_finite() {
        assert_eq!(LAYER4_EXPERTS.len(), 52);
        assert!(LAYER4_EXPERTS.windows(2).all(|pair| pair[0] < pair[1]));
        for expert in LAYER4_EXPERTS {
            for stage in [M68StageV1::W13, M68StageV1::W2] {
                let bits = m68_layer4_scale_bits(expert, stage).unwrap();
                assert!(f32::from_bits(bits).is_finite());
                assert!(f32::from_bits(bits) > 0.0);
                bind_m68_requirement(requirement(4, expert, stage), &owner(stage), true).unwrap();
            }
        }
        assert_eq!(M68_CAPACITIES.len() * 2, 16);
    }

    #[test]
    fn binder_rejects_drift_before_submission() {
        let stage = M68StageV1::W13;
        let weight = owner(stage);
        let mut value = requirement(4, LAYER4_EXPERTS[0], stage);
        value.input_scale_bits = Some(value.input_scale_bits.unwrap() + 1);
        assert_eq!(
            bind_m68_requirement(value, &weight, true),
            Err(M68BindError::Scale)
        );
        let mut value = requirement(4, 0, stage);
        value.input_scale_bits = Some(1);
        assert_eq!(
            bind_m68_requirement(value, &weight, true),
            Err(M68BindError::Expert)
        );
        let mut value = requirement(3, 0, stage);
        value.packed_weight_sha256[0] = 9;
        assert_eq!(
            bind_m68_requirement(value, &weight, true),
            Err(M68BindError::WeightIdentity)
        );
        assert_eq!(
            bind_m68_requirement(requirement(3, 0, stage), &weight, false),
            Err(M68BindError::KernelUnavailable)
        );
    }
}
