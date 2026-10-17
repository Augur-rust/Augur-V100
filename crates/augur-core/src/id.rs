#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct TensorId {
    pub index: u32,
    pub generation: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct NodeId {
    pub index: u32,
    pub generation: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct StorageId {
    pub index: u32,
    pub generation: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct WeightId {
    pub index: u32,
    pub generation: u32,
}

impl TensorId {
    pub fn new(index: usize, generation: u32) -> Self {
        assert!(index <= u32::MAX as usize);
        Self {
            index: index as u32,
            generation,
        }
    }
}
impl NodeId {
    pub fn new(index: usize, generation: u32) -> Self {
        assert!(index <= u32::MAX as usize);
        Self {
            index: index as u32,
            generation,
        }
    }
}
impl StorageId {
    pub fn new(index: usize, generation: u32) -> Self {
        assert!(index <= u32::MAX as usize);
        Self {
            index: index as u32,
            generation,
        }
    }
}
impl WeightId {
    pub fn new(index: usize, generation: u32) -> Self {
        assert!(index <= u32::MAX as usize);
        Self {
            index: index as u32,
            generation,
        }
    }
}
