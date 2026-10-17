use smallvec::SmallVec;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Shape {
    dims: SmallVec<[usize; 4]>,
}

impl Shape {
    pub fn new(dims: impl Into<SmallVec<[usize; 4]>>) -> Self {
        Self { dims: dims.into() }
    }

    pub fn from_dims(dims: &[usize]) -> Self {
        Self {
            dims: SmallVec::from_slice(dims),
        }
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn elem_count(&self) -> Option<usize> {
        self.dims
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
    }

    pub fn same_shape(&self, other: &Self) -> bool {
        self.dims == other.dims
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LogicalLayout {
    Contiguous,
    Strided {
        stride: SmallVec<[isize; 4]>,
    },
    Transposed,
    View {
        base: crate::TensorId,
        offset: usize,
    },
}
