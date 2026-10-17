use crate::{topo::topo_sort_validated, validate::validate_graph_for_analysis, Graph, Op};
use augur_core::{CompileError, NodeId, TensorId};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SafeFusionKind {
    SiluMul,
    ClampSiluMul,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SafeFusionGroup {
    pub kind: SafeFusionKind,
    pub nodes: Vec<NodeId>,
    pub output: TensorId,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SafeFusionDiscovery {
    groups: Vec<SafeFusionGroup>,
}

impl SafeFusionDiscovery {
    pub fn groups(&self) -> &[SafeFusionGroup] {
        &self.groups
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct ConsumerSummary {
    first: Option<NodeId>,
    shared: bool,
}

impl ConsumerSummary {
    fn observe(&mut self, consumer: NodeId) {
        match self.first {
            None => self.first = Some(consumer),
            Some(first) if first != consumer => self.shared = true,
            Some(_) => {}
        }
    }

    fn single(self) -> Option<NodeId> {
        if self.shared {
            None
        } else {
            self.first
        }
    }
}

/// Discovers conservative elementwise fusion groups without rewriting Graph.
///
/// A candidate is admitted only when every intermediate has one consumer and
/// is not a Graph output. Longest patterns win, and one node can belong to at
/// most one group. Backend lowering must still explicitly admit the group.
pub fn discover_safe_fusions(graph: &Graph) -> Result<SafeFusionDiscovery, CompileError> {
    let topo = {
        let validated = validate_graph_for_analysis(graph)?;
        topo_sort_validated(&validated)?
    };
    let mut consumers = vec![ConsumerSummary::default(); graph.tensor_count()];
    for slot in graph.nodes() {
        for &input in &slot.node.inputs {
            let summary = consumers
                .get_mut(input.index as usize)
                .ok_or(CompileError::InvalidInput)?;
            summary.observe(slot.id);
        }
    }

    let mut claimed = vec![false; graph.node_count()];
    let mut groups = Vec::new();
    for &node_id in topo.iter() {
        if claimed[node_id.index as usize] {
            continue;
        }
        let node = &graph.node(node_id).ok_or(CompileError::InvalidNode)?.node;
        let candidate = match node.op {
            Op::Clamp(_) => discover_clamp_silu_mul(graph, &consumers, node_id),
            Op::Silu => discover_silu_mul(graph, &consumers, node_id),
            _ => None,
        };
        let Some(group) = candidate else {
            continue;
        };
        if group
            .nodes
            .iter()
            .any(|id| claimed.get(id.index as usize).copied() != Some(false))
        {
            continue;
        }
        for id in &group.nodes {
            claimed[id.index as usize] = true;
        }
        groups.push(group);
    }
    Ok(SafeFusionDiscovery { groups })
}

fn discover_clamp_silu_mul(
    graph: &Graph,
    consumers: &[ConsumerSummary],
    clamp_id: NodeId,
) -> Option<SafeFusionGroup> {
    let clamp = &graph.node(clamp_id)?.node;
    let clamp_output = *clamp.outputs.first()?;
    let silu_id = single_consumer(graph, consumers, clamp_output)?;
    let silu = &graph.node(silu_id)?.node;
    if !matches!(silu.op, Op::Silu) || silu.inputs.as_slice() != [clamp_output] {
        return None;
    }
    let silu_output = *silu.outputs.first()?;
    let mul_id = single_consumer(graph, consumers, silu_output)?;
    let mul = &graph.node(mul_id)?.node;
    if !matches!(mul.op, Op::Mul) || !mul.inputs.contains(&silu_output) {
        return None;
    }
    Some(SafeFusionGroup {
        kind: SafeFusionKind::ClampSiluMul,
        nodes: vec![clamp_id, silu_id, mul_id],
        output: *mul.outputs.first()?,
    })
}

fn discover_silu_mul(
    graph: &Graph,
    consumers: &[ConsumerSummary],
    silu_id: NodeId,
) -> Option<SafeFusionGroup> {
    let silu = &graph.node(silu_id)?.node;
    let silu_output = *silu.outputs.first()?;
    let mul_id = single_consumer(graph, consumers, silu_output)?;
    let mul = &graph.node(mul_id)?.node;
    if !matches!(mul.op, Op::Mul) || !mul.inputs.contains(&silu_output) {
        return None;
    }
    Some(SafeFusionGroup {
        kind: SafeFusionKind::SiluMul,
        nodes: vec![silu_id, mul_id],
        output: *mul.outputs.first()?,
    })
}

fn single_consumer(
    graph: &Graph,
    consumers: &[ConsumerSummary],
    tensor: TensorId,
) -> Option<NodeId> {
    if graph.outputs().contains(&tensor) {
        return None;
    }
    consumers.get(tensor.index as usize)?.single()
}

#[cfg(test)]
mod tests {
    use super::*;
    use augur_core::{Shape, TensorDesc};

    fn desc() -> TensorDesc {
        TensorDesc::f32_contiguous(Shape::from_dims(&[16]))
    }

    #[test]
    fn longest_single_consumer_chain_is_discovered() {
        let mut graph = Graph::new();
        let x = graph.add_input(desc());
        let up = graph.add_input(desc());
        let clamp = graph.add_clamp(x, Some(-4.0), Some(4.0), desc()).unwrap();
        let silu = graph.add_silu(clamp, desc());
        let output = graph.add_mul(silu, up, desc());
        graph.try_set_output(output).unwrap();
        let discovery = discover_safe_fusions(&graph).unwrap();
        assert_eq!(discovery.groups().len(), 1);
        assert_eq!(discovery.groups()[0].kind, SafeFusionKind::ClampSiluMul);
        assert_eq!(discovery.groups()[0].nodes.len(), 3);
        assert_eq!(discovery.groups()[0].output, output);
    }

    #[test]
    fn shared_clamp_degrades_to_safe_silu_mul_subchain() {
        let mut graph = Graph::new();
        let x = graph.add_input(desc());
        let up = graph.add_input(desc());
        let clamp = graph.add_clamp(x, Some(-4.0), Some(4.0), desc()).unwrap();
        let silu = graph.add_silu(clamp, desc());
        let output = graph.add_mul(silu, up, desc());
        let branch = graph.add_add(clamp, up, desc());
        graph.try_set_output(output).unwrap();
        graph.try_set_output(branch).unwrap();
        let discovery = discover_safe_fusions(&graph).unwrap();
        assert_eq!(discovery.groups().len(), 1);
        assert_eq!(discovery.groups()[0].kind, SafeFusionKind::SiluMul);
        assert_eq!(discovery.groups()[0].nodes.len(), 2);
    }

    #[test]
    fn graph_output_intermediate_blocks_fusion() {
        let mut graph = Graph::new();
        let x = graph.add_input(desc());
        let up = graph.add_input(desc());
        let silu = graph.add_silu(x, desc());
        let output = graph.add_mul(silu, up, desc());
        graph.try_set_output(silu).unwrap();
        graph.try_set_output(output).unwrap();
        assert!(discover_safe_fusions(&graph).unwrap().groups().is_empty());
    }

    #[test]
    fn repeated_input_from_one_node_is_one_distinct_consumer() {
        let mut graph = Graph::new();
        let x = graph.add_input(desc());
        let silu = graph.add_silu(x, desc());
        let output = graph.add_mul(silu, silu, desc());
        graph.try_set_output(output).unwrap();

        let discovery = discover_safe_fusions(&graph).unwrap();
        assert_eq!(discovery.groups().len(), 1);
        assert_eq!(discovery.groups()[0].kind, SafeFusionKind::SiluMul);
        assert_eq!(discovery.groups()[0].nodes.len(), 2);
    }
}
