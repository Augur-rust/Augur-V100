use augur_core::{NodeId, TensorId};
use smallvec::SmallVec;

use crate::Op;

#[derive(Clone, Debug)]
pub struct Node {
    pub id: NodeId,
    pub op: Op,
    pub inputs: SmallVec<[TensorId; 8]>,
    pub outputs: SmallVec<[TensorId; 2]>,
}
