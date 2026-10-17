#![cfg(feature = "cuda")]

use augur_core::PhysicalLayoutId;

use crate::{
    CudaAddF32ArgsV1, CudaF32View, CudaQ4kF32DecodeArgsV1, CudaQ4kWeightViewV1,
    CudaQ5kF32DecodeArgsV1, CudaQ5kWeightViewV1, CudaQ6kF32DecodeArgsV1, CudaQ6kWeightViewV1,
    CudaRuntimeError, CudaRuntimeErrorKind, CudaSwigluF32ArgsV1, CudaWeightedAccumF32ArgsV1,
    OwnedCudaBuffer, OwnedCudaStream,
};

pub const M78_HIDDEN: usize = 4096;
pub const M78_INTERMEDIATE: usize = 2048;
pub const M78_SWIGLU_LIMIT: f32 = 10.0;

#[derive(Clone, Copy, Debug)]
pub struct CudaQkExpertWeightsV1 {
    pub w1: CudaQ4kWeightViewV1,
    pub w3: CudaQ4kWeightViewV1,
    pub w2: CudaQ5kWeightViewV1,
}

#[derive(Debug)]
pub struct CudaQkExpertF32V1 {
    device_ordinal: i32,
    gate: OwnedCudaBuffer,
    up: OwnedCudaBuffer,
}

#[derive(Debug)]
pub struct CudaRouteWeightedSixExpertF32V1 {
    device_ordinal: i32,
    expert: CudaQkExpertF32V1,
    expert_output: OwnedCudaBuffer,
}

impl CudaRouteWeightedSixExpertF32V1 {
    pub fn new(device_ordinal: i32) -> Result<Self, CudaRuntimeError> {
        Ok(Self {
            device_ordinal,
            expert: CudaQkExpertF32V1::new(device_ordinal)?,
            expert_output: OwnedCudaBuffer::allocate(device_ordinal, M78_HIDDEN * 4)?,
        })
    }

    pub fn launch(
        &self,
        stream: &OwnedCudaStream,
        experts: [CudaQkExpertWeightsV1; 6],
        route_weights: [f32; 6],
        input: CudaF32View,
        output: CudaF32View,
    ) -> Result<(), CudaRuntimeError> {
        let route_sum = route_weights.iter().sum::<f32>();
        if stream.device_ordinal() != self.device_ordinal
            || input.element_count() != M78_HIDDEN
            || output.element_count() != M78_HIDDEN
            || input.buffer_view().device_ptr() == output.buffer_view().device_ptr()
            || route_weights
                .iter()
                .any(|weight| !weight.is_finite() || *weight <= 0.0)
            || (route_sum - 1.5).abs() > 1.0e-4
        {
            return Err(CudaRuntimeError::validation(
                "cuda_route_weighted_six_expert_f32_v1",
                CudaRuntimeErrorKind::InvalidDimensions,
            ));
        }
        let expert_output = CudaF32View::new(
            self.expert_output.view(0, M78_HIDDEN * 4)?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            M78_HIDDEN,
        )?;
        for (rank, weights) in experts.into_iter().enumerate() {
            self.expert.launch(
                stream,
                weights.w1,
                weights.w3,
                weights.w2,
                input,
                expert_output,
            )?;
            CudaWeightedAccumF32ArgsV1::bind(
                stream,
                expert_output,
                output,
                route_weights[rank],
                rank == 0,
            )?
            .launch(stream)?;
        }
        Ok(())
    }

    pub fn scratch_bytes(&self) -> usize {
        self.expert.scratch_bytes() + M78_HIDDEN * 4
    }
}

#[derive(Debug)]
pub struct CudaQkSharedExpertF32V1 {
    device_ordinal: i32,
    gate: OwnedCudaBuffer,
    up: OwnedCudaBuffer,
}

#[derive(Clone, Copy, Debug)]
pub struct CudaQkSharedExpertWeightsV1 {
    pub w1: CudaQ5kWeightViewV1,
    pub w3: CudaQ5kWeightViewV1,
    pub w2: CudaQ6kWeightViewV1,
}

impl CudaQkSharedExpertF32V1 {
    pub fn new(device_ordinal: i32) -> Result<Self, CudaRuntimeError> {
        Ok(Self {
            device_ordinal,
            gate: OwnedCudaBuffer::allocate(device_ordinal, M78_INTERMEDIATE * 4)?,
            up: OwnedCudaBuffer::allocate(device_ordinal, M78_INTERMEDIATE * 4)?,
        })
    }

    pub fn launch(
        &self,
        stream: &OwnedCudaStream,
        w1: CudaQ5kWeightViewV1,
        w3: CudaQ5kWeightViewV1,
        w2: CudaQ6kWeightViewV1,
        input: CudaF32View,
        output: CudaF32View,
    ) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.device_ordinal
            || w1.rows() as usize != M78_INTERMEDIATE
            || w1.cols() as usize != M78_HIDDEN
            || w3.rows() as usize != M78_INTERMEDIATE
            || w3.cols() as usize != M78_HIDDEN
            || w2.rows() as usize != M78_HIDDEN
            || w2.cols() as usize != M78_INTERMEDIATE
            || input.element_count() != M78_HIDDEN
            || output.element_count() != M78_HIDDEN
        {
            return Err(CudaRuntimeError::validation(
                "cuda_qk_shared_expert_f32_v1",
                CudaRuntimeErrorKind::InvalidDimensions,
            ));
        }
        let gate = CudaF32View::new(
            self.gate.view(0, M78_INTERMEDIATE * 4)?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            M78_INTERMEDIATE,
        )?;
        let up = CudaF32View::new(
            self.up.view(0, M78_INTERMEDIATE * 4)?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            M78_INTERMEDIATE,
        )?;
        CudaQ5kF32DecodeArgsV1::bind(stream, w1, input, gate)?.submit_for_plan(stream)?;
        CudaQ5kF32DecodeArgsV1::bind(stream, w3, input, up)?.submit_for_plan(stream)?;
        CudaSwigluF32ArgsV1::bind(stream, gate, up, gate, M78_SWIGLU_LIMIT)?
            .submit_for_plan(stream)?;
        CudaQ6kF32DecodeArgsV1::bind(stream, w2, gate, output)?.launch(stream)
    }

    pub const fn scratch_bytes(&self) -> usize {
        M78_INTERMEDIATE * 8
    }
}

#[derive(Debug)]
pub struct CudaCompleteQkFfnF32V1 {
    device_ordinal: i32,
    routed: CudaRouteWeightedSixExpertF32V1,
    shared: CudaQkSharedExpertF32V1,
    routed_output: OwnedCudaBuffer,
    shared_output: OwnedCudaBuffer,
}

impl CudaCompleteQkFfnF32V1 {
    pub fn new(device_ordinal: i32) -> Result<Self, CudaRuntimeError> {
        Ok(Self {
            device_ordinal,
            routed: CudaRouteWeightedSixExpertF32V1::new(device_ordinal)?,
            shared: CudaQkSharedExpertF32V1::new(device_ordinal)?,
            routed_output: OwnedCudaBuffer::allocate(device_ordinal, M78_HIDDEN * 4)?,
            shared_output: OwnedCudaBuffer::allocate(device_ordinal, M78_HIDDEN * 4)?,
        })
    }

    pub fn launch(
        &self,
        stream: &OwnedCudaStream,
        routed_weights: [CudaQkExpertWeightsV1; 6],
        route_weights: [f32; 6],
        shared_weights: CudaQkSharedExpertWeightsV1,
        input: CudaF32View,
        output: CudaF32View,
    ) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.device_ordinal
            || input.element_count() != M78_HIDDEN
            || output.element_count() != M78_HIDDEN
            || input.buffer_view().device_ptr() == output.buffer_view().device_ptr()
        {
            return Err(CudaRuntimeError::validation(
                "cuda_complete_qk_ffn_f32_v1",
                CudaRuntimeErrorKind::InvalidDimensions,
            ));
        }
        let routed_output = CudaF32View::new(
            self.routed_output.view(0, M78_HIDDEN * 4)?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            M78_HIDDEN,
        )?;
        let shared_output = CudaF32View::new(
            self.shared_output.view(0, M78_HIDDEN * 4)?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            M78_HIDDEN,
        )?;
        self.routed
            .launch(stream, routed_weights, route_weights, input, routed_output)?;
        self.shared.launch(
            stream,
            shared_weights.w1,
            shared_weights.w3,
            shared_weights.w2,
            input,
            shared_output,
        )?;
        CudaAddF32ArgsV1::bind(stream, routed_output, shared_output, output)?.launch(stream)
    }

    pub fn scratch_bytes(&self) -> usize {
        self.routed.scratch_bytes()
            + self.shared.scratch_bytes()
            + self.routed_output.len_bytes()
            + self.shared_output.len_bytes()
    }
}

impl CudaQkExpertF32V1 {
    pub fn new(device_ordinal: i32) -> Result<Self, CudaRuntimeError> {
        Ok(Self {
            device_ordinal,
            gate: OwnedCudaBuffer::allocate(device_ordinal, M78_INTERMEDIATE * 4)?,
            up: OwnedCudaBuffer::allocate(device_ordinal, M78_INTERMEDIATE * 4)?,
        })
    }

    pub fn launch(
        &self,
        stream: &OwnedCudaStream,
        w1: CudaQ4kWeightViewV1,
        w3: CudaQ4kWeightViewV1,
        w2: CudaQ5kWeightViewV1,
        input: CudaF32View,
        output: CudaF32View,
    ) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.device_ordinal
            || w1.rows() as usize != M78_INTERMEDIATE
            || w1.cols() as usize != M78_HIDDEN
            || w3.rows() as usize != M78_INTERMEDIATE
            || w3.cols() as usize != M78_HIDDEN
            || w2.rows() as usize != M78_HIDDEN
            || w2.cols() as usize != M78_INTERMEDIATE
            || input.element_count() != M78_HIDDEN
            || output.element_count() != M78_HIDDEN
        {
            return Err(CudaRuntimeError::validation(
                "cuda_qk_expert_f32_v1",
                CudaRuntimeErrorKind::InvalidDimensions,
            ));
        }
        let gate = CudaF32View::new(
            self.gate.view(0, M78_INTERMEDIATE * 4)?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            M78_INTERMEDIATE,
        )?;
        let up = CudaF32View::new(
            self.up.view(0, M78_INTERMEDIATE * 4)?,
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            M78_INTERMEDIATE,
        )?;
        let gate_args = CudaQ4kF32DecodeArgsV1::bind(stream, w1, input, gate)?;
        let up_args = CudaQ4kF32DecodeArgsV1::bind(stream, w3, input, up)?;
        let activation = CudaSwigluF32ArgsV1::bind(stream, gate, up, gate, M78_SWIGLU_LIMIT)?;
        let down = CudaQ5kF32DecodeArgsV1::bind(stream, w2, gate, output)?;
        gate_args.submit_for_plan(stream)?;
        up_args.submit_for_plan(stream)?;
        activation.submit_for_plan(stream)?;
        down.launch(stream)
    }

    pub fn scratch_bytes(&self) -> usize {
        M78_INTERMEDIATE * 8
    }
}
