use std::ops::Deref;

use crate::{validate::ValidatedGraph, Graph};
use augur_core::{CompileError, NodeId};

#[derive(Clone, Debug)]
pub struct ValidatedTopo(Vec<NodeId>);

impl Deref for ValidatedTopo {
    type Target = [NodeId];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub fn topo_sort(graph: &Graph) -> Result<ValidatedTopo, CompileError> {
    let producers = collect_producers(graph)?;
    topo_sort_with_producers(graph, &producers, true)
}

pub(crate) fn topo_sort_validated(
    validated: &ValidatedGraph<'_>,
) -> Result<ValidatedTopo, CompileError> {
    topo_sort_with_producers(validated.graph(), validated.producers(), false)
}

fn collect_producers(graph: &Graph) -> Result<Vec<Option<NodeId>>, CompileError> {
    let mut producers = vec![None; graph.tensor_count()];
    for (node_index, slot) in graph.nodes().iter().enumerate() {
        if slot.id.index as usize != node_index || slot.node.id != slot.id {
            return Err(CompileError::InvalidNode);
        }
        for &output in &slot.node.outputs {
            graph.tensor(output).ok_or(CompileError::InvalidOutput)?;
            let producer = producers
                .get_mut(output.index as usize)
                .ok_or(CompileError::InvalidOutput)?;
            if producer.is_some() {
                return Err(CompileError::DuplicateProducer);
            }
            *producer = Some(slot.id);
        }
    }
    Ok(producers)
}

fn topo_sort_with_producers(
    graph: &Graph,
    producer_of_tensor: &[Option<NodeId>],
    validate_input_references: bool,
) -> Result<ValidatedTopo, CompileError> {
    let node_count = graph.node_count();
    let mut indegree = vec![0usize; node_count];
    let mut consumer_counts = vec![0usize; node_count];
    if producer_of_tensor.len() != graph.tensor_count() {
        return Err(CompileError::InvalidInput);
    }

    // First pass: build indegree / consumer counts using producer metadata.
    for (node_index, slot) in graph.nodes().iter().enumerate() {
        for &input in &slot.node.inputs {
            if validate_input_references {
                graph.tensor(input).ok_or(CompileError::InvalidInput)?;
            }
            let in_idx = input.index as usize;
            let producer = producer_of_tensor
                .get(in_idx)
                .ok_or(CompileError::InvalidInput)?;
            if let Some(producer) = producer {
                let producer_idx = producer.index as usize;
                if producer_idx >= node_count {
                    return Err(CompileError::InvalidNode);
                }
                indegree[node_index] += 1;
                consumer_counts[producer_idx] += 1;
            }
        }
    }

    let mut consumer_offsets = Vec::with_capacity(node_count + 1);
    consumer_offsets.push(0);
    let mut consumer_edge_count = 0;
    for &count in &consumer_counts {
        consumer_edge_count += count;
        consumer_offsets.push(consumer_edge_count);
    }
    let mut consumers = vec![0usize; consumer_edge_count];
    consumer_counts.fill(0);
    for (node_index, slot) in graph.nodes().iter().enumerate() {
        for &input in &slot.node.inputs {
            let in_idx = input.index as usize;
            if let Some(producer) = producer_of_tensor[in_idx] {
                let producer_idx = producer.index as usize;
                let write_index = consumer_offsets[producer_idx] + consumer_counts[producer_idx];
                consumers[write_index] = node_index;
                consumer_counts[producer_idx] += 1;
            }
        }
    }

    let mut order = Vec::with_capacity(node_count);
    for (idx, &degree) in indegree.iter().enumerate() {
        if degree == 0 {
            order.push(graph.nodes().get(idx).ok_or(CompileError::InvalidNode)?.id);
        }
    }

    let mut next = 0;
    while next < order.len() {
        let node_id = order[next];
        let idx = node_id.index as usize;
        // Final generation-safe check.
        if graph.node(node_id).is_none() {
            return Err(CompileError::InvalidNode);
        }
        for &consumer in &consumers[consumer_offsets[idx]..consumer_offsets[idx + 1]] {
            indegree[consumer] -= 1;
            if indegree[consumer] == 0 {
                order.push(
                    graph
                        .nodes()
                        .get(consumer)
                        .ok_or(CompileError::InvalidNode)?
                        .id,
                );
            }
        }
        next += 1;
    }

    if order.len() != node_count {
        return Err(CompileError::Cycle);
    }
    Ok(ValidatedTopo(order))
}
