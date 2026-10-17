//! Only position-independent operations are captured. Dynamic RoPE/KV/attention
//! stay between the two graphs, with the original poison and cursor handling.
use super::*;
use augur_backend::OwnedCudaGraphExec;

pub(crate) struct FullB1Graphs {
    input: OwnedCudaGraphExec,
    tail: OwnedCudaGraphExec,
}
impl FullB1Graphs {
    pub(crate) fn node_count(&self) -> u64 {
        self.input.node_count() + self.tail.node_count()
    }
}
fn capture(stream: &OwnedCudaStream, submit: impl FnOnce() -> Result<()>) -> Result<OwnedCudaGraphExec> {
    let mut submit_error = None;
    let graph = OwnedCudaGraphExec::capture(stream, || submit().map_err(|error| {
        submit_error = Some(error);
        augur_backend::CudaRuntimeError::invalid_metadata("checkpoint_fixed_b1_graph_capture")
    }));
    if let Some(error) = submit_error { return Err(error); }
    Ok(graph?)
}
impl CheckpointFullCudaLayer {
    fn submit_b1_input_operations(&self) -> Result<()> {
        let start = self.submit_input_prefix()?;
        for op in &self.input_ops[start..] { op.submit(&self.stream)?; }
        Ok(())
    }
    pub(crate) fn prepare_b1_graphs(&self) -> Result<FullB1Graphs> {
        if self.tokens != 0 || self.poisoned || self.b1_graphs.is_some()
            || self.input_norm_qpn.is_none() || self.post_norm_qpn.is_none() {
            return Err("full B1 graphs require a fresh finalized fused layer".into());
        }
        Ok(FullB1Graphs {
            input: capture(&self.stream, || self.submit_b1_input_operations())?,
            tail: capture(&self.stream, || self.submit_complete_ops(0))?,
        })
    }
    pub(crate) fn install_b1_graphs(&mut self, graphs: FullB1Graphs) {
        self.b1_graphs = Some(graphs);
    }
    pub(super) fn submit_b1_input(&self) -> Result<()> {
        if let Some(graphs) = &self.b1_graphs { graphs.input.launch(&self.stream)?; }
        else { self.submit_b1_input_operations()?; }
        Ok(())
    }
    pub(super) fn submit_b1_tail(&self) -> Result<()> {
        if let Some(graphs) = &self.b1_graphs { graphs.tail.launch(&self.stream)?; }
        else { self.submit_complete_ops(0)?; }
        Ok(())
    }
}
impl CheckpointCudaOutputHead {
    pub(crate) fn prepare_b1_graph(&self) -> Result<OwnedCudaGraphExec> {
        if self.b1_graph.is_some() { return Err("output head graph already installed".into()); }
        capture(&self.stream, || self.submit_head_operations())
    }
    pub(crate) fn install_b1_graph(&mut self, graph: OwnedCudaGraphExec) {
        self.b1_graph = Some(graph);
    }
}
