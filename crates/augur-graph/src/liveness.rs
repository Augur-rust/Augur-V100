use crate::{topo_sort, Graph, ValidatedTopo};
use augur_core::{CompileError, TensorId};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LiveRange {
    pub first_step: usize,
    pub last_step: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TensorLiveRange {
    pub tensor: TensorId,
    pub range: LiveRange,
}

#[derive(Clone, Debug)]
pub struct Liveness {
    tensor_ranges: Vec<Option<TensorLiveRange>>,
}

impl Liveness {
    pub fn range(&self, tensor: TensorId) -> Option<LiveRange> {
        let entry = self.tensor_ranges.get(tensor.index as usize)?.as_ref()?;
        if entry.tensor == tensor {
            Some(entry.range)
        } else {
            None
        }
    }
}

pub fn analyze_liveness(graph: &Graph) -> Result<Liveness, CompileError> {
    let topo = topo_sort(graph)?;
    analyze_liveness_with_topo(graph, &topo)
}

pub fn analyze_liveness_with_topo(
    graph: &Graph,
    topo: &ValidatedTopo,
) -> Result<Liveness, CompileError> {
    let mut ranges = vec![None; graph.tensor_count()];

    for (step, &node_id) in topo.iter().enumerate() {
        let slot = graph.node(node_id).ok_or(CompileError::InvalidNode)?;
        for &input in &slot.node.inputs {
            graph.tensor(input).ok_or(CompileError::InvalidInput)?;
            touch(&mut ranges, input, step)?;
        }
        for &output in &slot.node.outputs {
            graph.tensor(output).ok_or(CompileError::InvalidOutput)?;
            touch(&mut ranges, output, step)?;
        }
    }

    let end_step = topo.len();
    for &output in graph.outputs() {
        graph.tensor(output).ok_or(CompileError::InvalidOutput)?;
        touch(&mut ranges, output, end_step)?;
    }

    Ok(Liveness {
        tensor_ranges: ranges,
    })
}

fn touch(
    ranges: &mut [Option<TensorLiveRange>],
    tensor: TensorId,
    step: usize,
) -> Result<(), CompileError> {
    let slot = ranges
        .get_mut(tensor.index as usize)
        .ok_or(CompileError::InvalidInput)?;
    match slot {
        Some(entry) => {
            if entry.tensor != tensor {
                return Err(CompileError::InvalidInput);
            }
            entry.range.first_step = entry.range.first_step.min(step);
            entry.range.last_step = entry.range.last_step.max(step);
        }
        None => {
            *slot = Some(TensorLiveRange {
                tensor,
                range: LiveRange {
                    first_step: step,
                    last_step: step,
                },
            });
        }
    }
    Ok(())
}
