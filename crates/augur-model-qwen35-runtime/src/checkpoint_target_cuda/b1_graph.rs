//! Graph capture is staged before installation, while every layer is fresh.
use super::*;
fn graph_switch(name: &str) -> Result<bool> {
    let value = match std::env::var(name) {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => return Err(error.into()),
    };
    if name == "AUGUR_B1_LINEAR_GRAPH" { return b1_linear_graph_switch(value.as_deref()); }
    match value.as_deref() {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        _ => Err(format!("{name} must be 0 or 1").into()),
    }
}
impl CheckpointCudaTarget {
    pub(super) fn enable_fixed_b1_graphs(&mut self) -> Result<()> {
        let linear_enabled = graph_switch("AUGUR_B1_LINEAR_GRAPH")?;
        let full_enabled = graph_switch("AUGUR_B1_FULL_GRAPH")?;
        let head_enabled = graph_switch("AUGUR_B1_HEAD_GRAPH")?;
        let mut linear = Vec::new();
        let mut full = Vec::new();
        for (index, layer) in self.layers.iter().enumerate() {
            match layer {
                Layer::Linear(layer) if linear_enabled => linear.push((index, layer.prepare_b1_graph()?)),
                Layer::Full(layer) if full_enabled => full.push((index, layer.prepare_b1_graphs()?)),
                _ => (),
            }
        }
        if (linear_enabled && linear.len() != 48) || (full_enabled && full.len() != 16) {
            return Err("B1 graph layer inventory changed".into());
        }
        let head = if head_enabled { Some(self.head.prepare_b1_graph()?) } else { None };
        self.info["b1_linear_submission"] = if linear_enabled { "cuda_graph_same_ops_v1" } else { "individual_same_ops_v1" }.into();
        self.info["b1_linear_graph_layers"] = linear.len().into();
        self.info["b1_linear_graph_nodes"] = linear.iter().map(|(_, g)| g.node_count()).sum::<u64>().into();
        self.info["b1_full_submission"] = if full_enabled { "cuda_graph_fixed_input_tail_dynamic_attention_v1" } else { "individual_same_ops_v1" }.into();
        self.info["b1_full_graph_layers"] = full.len().into();
        self.info["b1_full_graph_nodes"] = full.iter().map(|(_, g)| g.node_count()).sum::<u64>().into();
        self.info["b1_head_graph_nodes"] = head.as_ref().map_or(0, |g| g.node_count()).into();
        for (index, graph) in linear {
            let Layer::Linear(layer) = &mut self.layers[index] else { unreachable!() };
            layer.install_b1_graph(graph);
        }
        for (index, graphs) in full {
            let Layer::Full(layer) = &mut self.layers[index] else { unreachable!() };
            layer.install_b1_graphs(graphs);
        }
        if let Some(graph) = head { self.head.install_b1_graph(graph); }
        Ok(())
    }
}
