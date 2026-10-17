use std::sync::Arc;

use augur_backend::Backend;
use augur_kernel_api::{
    BoundDualScaleQ8Hot, DualScaleQ8HotCpuPrevalidatedLaunch, Fp8E4M3ScaleGridHotCpuTensor,
    Fp8FusedMlpF32CpuLaunch, Fp8FusedMlpF32Workspace, Fp8ScaleGridF32CpuLaunch, FusedMlpF32Config,
    Iq2XxsHotCpuTensor, Iq2XxsQ8HotCpuLaunch, Mxfp4F32CpuLaunch, Mxfp4HotCpuTensor,
    Q2KHotCpuTensor, Q2KQ8HotCpuLaunch, Q4KHotCpuTensor, Q8_0HotCpuTensor, Q8_0Q8HotCpuLaunch,
    QMatmulBiasF32Launch, QMatmulF32Launch, QMatmulF32PrefillLaunch, QuantizeQ8HotCpuLaunch,
};

use crate::BufferView;

#[derive(Clone, Debug)]
pub struct PlannedQMatmulF32Base<B: Backend> {
    weight: Arc<Q4KHotCpuTensor>,
    input: BufferView<B>,
    output: BufferView<B>,
    rows: usize,
    cols: usize,
}

impl<B: Backend> PlannedQMatmulF32Base<B> {
    pub fn new(
        weight: Arc<Q4KHotCpuTensor>,
        input: BufferView<B>,
        output: BufferView<B>,
        rows: usize,
        cols: usize,
    ) -> Self {
        Self {
            weight,
            input,
            output,
            rows,
            cols,
        }
    }
    pub fn weight(&self) -> &Arc<Q4KHotCpuTensor> {
        &self.weight
    }
    pub fn input(&self) -> &BufferView<B> {
        &self.input
    }
    pub fn output(&self) -> &BufferView<B> {
        &self.output
    }
    pub const fn rows(&self) -> usize {
        self.rows
    }
    pub const fn cols(&self) -> usize {
        self.cols
    }
}

#[derive(Clone, Debug)]
pub struct PlannedQMatmulF32Args<B: Backend> {
    base: PlannedQMatmulF32Base<B>,
    launch: QMatmulF32Launch,
}
impl<B: Backend> PlannedQMatmulF32Args<B> {
    pub fn new(base: PlannedQMatmulF32Base<B>, launch: QMatmulF32Launch) -> Self {
        Self { base, launch }
    }
    pub fn base(&self) -> &PlannedQMatmulF32Base<B> {
        &self.base
    }
    pub const fn launch(&self) -> QMatmulF32Launch {
        self.launch
    }
}

#[derive(Clone, Debug)]
pub struct PlannedQMatmulF32PrefillArgs<B: Backend> {
    weight: Arc<Q4KHotCpuTensor>,
    input: BufferView<B>,
    output: BufferView<B>,
    m: usize,
    rows: usize,
    cols: usize,
    launch: QMatmulF32PrefillLaunch,
}

impl<B: Backend> PlannedQMatmulF32PrefillArgs<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        weight: Arc<Q4KHotCpuTensor>,
        input: BufferView<B>,
        output: BufferView<B>,
        m: usize,
        rows: usize,
        cols: usize,
        launch: QMatmulF32PrefillLaunch,
    ) -> Self {
        Self {
            weight,
            input,
            output,
            m,
            rows,
            cols,
            launch,
        }
    }
    pub fn weight(&self) -> &Arc<Q4KHotCpuTensor> {
        &self.weight
    }
    pub fn input(&self) -> &BufferView<B> {
        &self.input
    }
    pub fn output(&self) -> &BufferView<B> {
        &self.output
    }
    pub const fn m(&self) -> usize {
        self.m
    }
    pub const fn rows(&self) -> usize {
        self.rows
    }
    pub const fn cols(&self) -> usize {
        self.cols
    }
    pub const fn launch(&self) -> QMatmulF32PrefillLaunch {
        self.launch
    }
}

#[derive(Clone, Debug)]
pub struct PlannedQMatmulBiasF32Args<B: Backend> {
    base: PlannedQMatmulF32Base<B>,
    bias: BufferView<B>,
    launch: QMatmulBiasF32Launch,
}
impl<B: Backend> PlannedQMatmulBiasF32Args<B> {
    pub fn new(
        base: PlannedQMatmulF32Base<B>,
        bias: BufferView<B>,
        launch: QMatmulBiasF32Launch,
    ) -> Self {
        Self { base, bias, launch }
    }
    pub fn base(&self) -> &PlannedQMatmulF32Base<B> {
        &self.base
    }
    pub fn bias(&self) -> &BufferView<B> {
        &self.bias
    }
    pub const fn launch(&self) -> QMatmulBiasF32Launch {
        self.launch
    }
}

#[derive(Clone, Debug)]
pub struct PlannedQuantizeQ8HotArgs<B: Backend> {
    input: BufferView<B>,
    output: BufferView<B>,
    launch: QuantizeQ8HotCpuLaunch,
}
impl<B: Backend> PlannedQuantizeQ8HotArgs<B> {
    pub fn new(
        input: BufferView<B>,
        output: BufferView<B>,
        launch: QuantizeQ8HotCpuLaunch,
    ) -> Self {
        Self {
            input,
            output,
            launch,
        }
    }
    pub fn input(&self) -> &BufferView<B> {
        &self.input
    }
    pub fn output(&self) -> &BufferView<B> {
        &self.output
    }
    pub const fn launch(&self) -> QuantizeQ8HotCpuLaunch {
        self.launch
    }
}

#[derive(Clone, Debug)]
pub struct PlannedDualScaleQ8HotArgs<B: Backend> {
    bound: BoundDualScaleQ8Hot,
    activation: BufferView<B>,
    output: BufferView<B>,
    rows: usize,
    cols: usize,
    blocks_per_row: usize,
    launch: DualScaleQ8HotCpuPrevalidatedLaunch,
}
impl<B: Backend> PlannedDualScaleQ8HotArgs<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        bound: BoundDualScaleQ8Hot,
        activation: BufferView<B>,
        output: BufferView<B>,
        rows: usize,
        cols: usize,
        blocks_per_row: usize,
        launch: DualScaleQ8HotCpuPrevalidatedLaunch,
    ) -> Self {
        Self {
            bound,
            activation,
            output,
            rows,
            cols,
            blocks_per_row,
            launch,
        }
    }
    pub fn bound(&self) -> &BoundDualScaleQ8Hot {
        &self.bound
    }
    pub fn weight(&self) -> &Arc<augur_kernel_api::DualScaleHotCpuTensor> {
        self.bound.weight_owner()
    }
    pub fn activation(&self) -> &BufferView<B> {
        &self.activation
    }
    pub fn output(&self) -> &BufferView<B> {
        &self.output
    }
    pub const fn rows(&self) -> usize {
        self.rows
    }
    pub const fn cols(&self) -> usize {
        self.cols
    }
    pub const fn blocks_per_row(&self) -> usize {
        self.blocks_per_row
    }
    pub const fn launch(&self) -> DualScaleQ8HotCpuPrevalidatedLaunch {
        self.launch
    }
}

#[derive(Clone, Debug)]
pub struct PlannedQ8_0Q8HotArgs<B: Backend> {
    weight: Arc<Q8_0HotCpuTensor>,
    activation: BufferView<B>,
    output: BufferView<B>,
    rows: usize,
    cols: usize,
    launch: Q8_0Q8HotCpuLaunch,
}

impl<B: Backend> PlannedQ8_0Q8HotArgs<B> {
    pub fn new(
        weight: Arc<Q8_0HotCpuTensor>,
        activation: BufferView<B>,
        output: BufferView<B>,
        rows: usize,
        cols: usize,
        launch: Q8_0Q8HotCpuLaunch,
    ) -> Self {
        Self {
            weight,
            activation,
            output,
            rows,
            cols,
            launch,
        }
    }

    pub fn weight(&self) -> &Arc<Q8_0HotCpuTensor> {
        &self.weight
    }

    pub fn activation(&self) -> &BufferView<B> {
        &self.activation
    }

    pub fn output(&self) -> &BufferView<B> {
        &self.output
    }

    pub const fn rows(&self) -> usize {
        self.rows
    }

    pub const fn cols(&self) -> usize {
        self.cols
    }

    pub const fn launch(&self) -> Q8_0Q8HotCpuLaunch {
        self.launch
    }
}

#[derive(Clone, Debug)]
pub struct PlannedQ2KQ8HotArgs<B: Backend> {
    weight: Arc<Q2KHotCpuTensor>,
    activation: BufferView<B>,
    output: BufferView<B>,
    rows: usize,
    cols: usize,
    launch: Q2KQ8HotCpuLaunch,
}

#[derive(Clone, Debug)]
pub struct PlannedIq2XxsQ8HotArgs<B: Backend> {
    weight: Arc<Iq2XxsHotCpuTensor>,
    activation: BufferView<B>,
    output: BufferView<B>,
    rows: usize,
    cols: usize,
    launch: Iq2XxsQ8HotCpuLaunch,
}

#[derive(Clone, Debug)]
pub struct PlannedMxfp4F32Args<B: Backend> {
    weight: Arc<Mxfp4HotCpuTensor>,
    input: BufferView<B>,
    output: BufferView<B>,
    rows: usize,
    cols: usize,
    launch: Mxfp4F32CpuLaunch,
}

impl<B: Backend> PlannedMxfp4F32Args<B> {
    pub fn new(
        weight: Arc<Mxfp4HotCpuTensor>,
        input: BufferView<B>,
        output: BufferView<B>,
        rows: usize,
        cols: usize,
        launch: Mxfp4F32CpuLaunch,
    ) -> Self {
        Self {
            weight,
            input,
            output,
            rows,
            cols,
            launch,
        }
    }

    pub fn weight(&self) -> &Arc<Mxfp4HotCpuTensor> {
        &self.weight
    }
    pub fn input(&self) -> &BufferView<B> {
        &self.input
    }
    pub fn output(&self) -> &BufferView<B> {
        &self.output
    }
    pub const fn rows(&self) -> usize {
        self.rows
    }
    pub const fn cols(&self) -> usize {
        self.cols
    }
    pub const fn launch(&self) -> Mxfp4F32CpuLaunch {
        self.launch
    }
}

#[derive(Clone, Debug)]
pub struct PlannedFp8ScaleGridF32Args<B: Backend> {
    weight: Arc<Fp8E4M3ScaleGridHotCpuTensor>,
    input: BufferView<B>,
    output: BufferView<B>,
    rows: usize,
    cols: usize,
    launch: Fp8ScaleGridF32CpuLaunch,
}

impl<B: Backend> PlannedFp8ScaleGridF32Args<B> {
    pub fn new(
        weight: Arc<Fp8E4M3ScaleGridHotCpuTensor>,
        input: BufferView<B>,
        output: BufferView<B>,
        rows: usize,
        cols: usize,
        launch: Fp8ScaleGridF32CpuLaunch,
    ) -> Self {
        Self {
            weight,
            input,
            output,
            rows,
            cols,
            launch,
        }
    }
    pub fn weight(&self) -> &Arc<Fp8E4M3ScaleGridHotCpuTensor> {
        &self.weight
    }
    pub fn input(&self) -> &BufferView<B> {
        &self.input
    }
    pub fn output(&self) -> &BufferView<B> {
        &self.output
    }
    pub const fn rows(&self) -> usize {
        self.rows
    }
    pub const fn cols(&self) -> usize {
        self.cols
    }
    pub const fn launch(&self) -> Fp8ScaleGridF32CpuLaunch {
        self.launch
    }
}

#[derive(Clone, Debug)]
pub struct PlannedFp8FusedMlpF32Args<B: Backend> {
    gate_weight: Arc<Fp8E4M3ScaleGridHotCpuTensor>,
    up_weight: Arc<Fp8E4M3ScaleGridHotCpuTensor>,
    down_weight: Arc<Fp8E4M3ScaleGridHotCpuTensor>,
    input: BufferView<B>,
    output: BufferView<B>,
    hidden: usize,
    intermediate: usize,
    config: FusedMlpF32Config,
    workspace: Fp8FusedMlpF32Workspace,
    launch: Fp8FusedMlpF32CpuLaunch,
}

impl<B: Backend> PlannedFp8FusedMlpF32Args<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gate_weight: Arc<Fp8E4M3ScaleGridHotCpuTensor>,
        up_weight: Arc<Fp8E4M3ScaleGridHotCpuTensor>,
        down_weight: Arc<Fp8E4M3ScaleGridHotCpuTensor>,
        input: BufferView<B>,
        output: BufferView<B>,
        hidden: usize,
        intermediate: usize,
        config: FusedMlpF32Config,
        workspace: Fp8FusedMlpF32Workspace,
        launch: Fp8FusedMlpF32CpuLaunch,
    ) -> Self {
        Self {
            gate_weight,
            up_weight,
            down_weight,
            input,
            output,
            hidden,
            intermediate,
            config,
            workspace,
            launch,
        }
    }
    pub fn input(&self) -> &BufferView<B> {
        &self.input
    }
    pub fn output(&self) -> &BufferView<B> {
        &self.output
    }
    pub const fn hidden(&self) -> usize {
        self.hidden
    }
    pub const fn intermediate(&self) -> usize {
        self.intermediate
    }
    pub fn workspace_payload_bytes(&self) -> usize {
        self.workspace.payload_bytes()
    }
    pub const fn config(&self) -> FusedMlpF32Config {
        self.config
    }
    pub const fn launch(&self) -> Fp8FusedMlpF32CpuLaunch {
        self.launch
    }
    pub fn weight_owners(&self) -> [Arc<Fp8E4M3ScaleGridHotCpuTensor>; 3] {
        [
            Arc::clone(&self.gate_weight),
            Arc::clone(&self.up_weight),
            Arc::clone(&self.down_weight),
        ]
    }
    pub fn launch_parts(
        &mut self,
    ) -> (
        &Fp8E4M3ScaleGridHotCpuTensor,
        &Fp8E4M3ScaleGridHotCpuTensor,
        &Fp8E4M3ScaleGridHotCpuTensor,
        &BufferView<B>,
        &BufferView<B>,
        FusedMlpF32Config,
        &mut Fp8FusedMlpF32Workspace,
        Fp8FusedMlpF32CpuLaunch,
    ) {
        (
            &self.gate_weight,
            &self.up_weight,
            &self.down_weight,
            &self.input,
            &self.output,
            self.config,
            &mut self.workspace,
            self.launch,
        )
    }
}

impl<B: Backend> PlannedQ2KQ8HotArgs<B> {
    pub fn new(
        weight: Arc<Q2KHotCpuTensor>,
        activation: BufferView<B>,
        output: BufferView<B>,
        rows: usize,
        cols: usize,
        launch: Q2KQ8HotCpuLaunch,
    ) -> Self {
        Self {
            weight,
            activation,
            output,
            rows,
            cols,
            launch,
        }
    }

    pub fn weight(&self) -> &Arc<Q2KHotCpuTensor> {
        &self.weight
    }

    pub fn activation(&self) -> &BufferView<B> {
        &self.activation
    }

    pub fn output(&self) -> &BufferView<B> {
        &self.output
    }

    pub const fn rows(&self) -> usize {
        self.rows
    }

    pub const fn cols(&self) -> usize {
        self.cols
    }

    pub const fn launch(&self) -> Q2KQ8HotCpuLaunch {
        self.launch
    }
}

impl<B: Backend> PlannedIq2XxsQ8HotArgs<B> {
    pub fn new(
        weight: Arc<Iq2XxsHotCpuTensor>,
        activation: BufferView<B>,
        output: BufferView<B>,
        rows: usize,
        cols: usize,
        launch: Iq2XxsQ8HotCpuLaunch,
    ) -> Self {
        Self {
            weight,
            activation,
            output,
            rows,
            cols,
            launch,
        }
    }

    pub fn weight(&self) -> &Arc<Iq2XxsHotCpuTensor> {
        &self.weight
    }

    pub fn activation(&self) -> &BufferView<B> {
        &self.activation
    }

    pub fn output(&self) -> &BufferView<B> {
        &self.output
    }

    pub const fn rows(&self) -> usize {
        self.rows
    }

    pub const fn cols(&self) -> usize {
        self.cols
    }

    pub const fn launch(&self) -> Iq2XxsQ8HotCpuLaunch {
        self.launch
    }
}

#[derive(Clone, Debug)]
pub enum PlannedOp<B: Backend> {
    AddContiguous {
        dst: BufferView<B>,
        a: BufferView<B>,
        b: BufferView<B>,
        n: usize,
    },
    MulContiguous {
        dst: BufferView<B>,
        a: BufferView<B>,
        b: BufferView<B>,
        n: usize,
    },
    ClampContiguous {
        dst: BufferView<B>,
        input: BufferView<B>,
        n: usize,
        min: Option<f32>,
        max: Option<f32>,
    },
    SiluContiguous {
        dst: BufferView<B>,
        input: BufferView<B>,
        n: usize,
    },
    QuantizeQ8Hot(PlannedQuantizeQ8HotArgs<B>),
    DualScaleQ8Hot(PlannedDualScaleQ8HotArgs<B>),
    Q8_0Q8Hot(PlannedQ8_0Q8HotArgs<B>),
    Q2KQ8Hot(PlannedQ2KQ8HotArgs<B>),
    Iq2XxsQ8Hot(PlannedIq2XxsQ8HotArgs<B>),
    Mxfp4F32(PlannedMxfp4F32Args<B>),
    Fp8ScaleGridF32(PlannedFp8ScaleGridF32Args<B>),
    Fp8FusedMlpF32(PlannedFp8FusedMlpF32Args<B>),
    QMatmulF32(PlannedQMatmulF32Args<B>),
    QMatmulF32Prefill(PlannedQMatmulF32PrefillArgs<B>),
    QMatmulBiasF32(PlannedQMatmulBiasF32Args<B>),
    QMatmulBiasSiluF32(PlannedQMatmulBiasF32Args<B>),
}
