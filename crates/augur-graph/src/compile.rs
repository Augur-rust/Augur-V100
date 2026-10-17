use crate::{
    analyze_liveness_with_topo, topo::topo_sort_validated, validate::validate_graph_for_analysis,
    Graph, Liveness, ValidatedTopo,
};
use augur_core::{CompileError, NodeId};

#[derive(Clone, Debug)]
pub struct GraphAnalysis {
    topo: ValidatedTopo,
    liveness: Liveness,
}

impl GraphAnalysis {
    pub fn topo(&self) -> &[NodeId] {
        &self.topo
    }
    pub fn liveness(&self) -> &Liveness {
        &self.liveness
    }
}

/// A validated graph bound to the exact analysis produced for it.
///
/// The graph cannot be mutably modified while this token is alive.
#[derive(Debug)]
pub struct AnalyzedGraph<'g> {
    graph: &'g Graph,
    analysis: GraphAnalysis,
}

impl<'g> AnalyzedGraph<'g> {
    pub fn graph(&self) -> &'g Graph {
        self.graph
    }
    pub fn analysis(&self) -> &GraphAnalysis {
        &self.analysis
    }
    pub fn topo(&self) -> &[NodeId] {
        self.analysis.topo()
    }
    pub fn liveness(&self) -> &Liveness {
        self.analysis.liveness()
    }
}

pub fn analyze_graph(graph: &Graph) -> Result<AnalyzedGraph<'_>, CompileError> {
    let topo = {
        let validated = validate_graph_for_analysis(graph)?;
        topo_sort_validated(&validated)?
    };
    let liveness = analyze_liveness_with_topo(graph, &topo)?;
    Ok(AnalyzedGraph {
        graph,
        analysis: GraphAnalysis { topo, liveness },
    })
}
