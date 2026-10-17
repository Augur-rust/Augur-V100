//! B8 GDN uses fixed addresses and dimensions. Capture identical operations,
//! including replay copies before the aliased FFN scratch overwrites them.
use super::*;
use augur_backend::OwnedCudaGraphExec;

pub(super) struct RecordedGraph {
    pub(super) exec: OwnedCudaGraphExec,
    // Retain every replay allocation captured by the graph and reject a foreign replay.
    pub(super) records: [Buffer;4],
}
impl Program {
    pub(crate) fn enable_recorded_b8_graph(&mut self,replay:&dflash::Replay)->Result<()> {
        if self.tokens!=8 || self.recorded_graph.is_some() {return Err("GDN graph requires uncaptured B8 program".into());}
        let records=replay.graph_records(&self.workspace)?;
        let mut original_error=None;
        let exec=OwnedCudaGraphExec::capture(&self.workspace.stream,||self.submit_operations(Some(replay)).map_err(|error|{
            original_error=Some(error);
            augur_backend::CudaRuntimeError::invalid_metadata("augur_b8_recorded_gdn_graph_capture")
        }));
        if let Some(error)=original_error {return Err(error);}
        self.recorded_graph=Some(RecordedGraph{exec:exec?,records});
        Ok(())
    }
    pub(crate) fn recorded_b8_graph_nodes(&self)->u64 {
        self.recorded_graph.as_ref().map_or(0,|g|g.exec.node_count())
    }
}
