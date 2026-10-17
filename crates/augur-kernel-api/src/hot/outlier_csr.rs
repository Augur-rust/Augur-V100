use augur_core::F16Bits;

use crate::HotLayoutError;

/// Immutable, constructor-validated CPU Hot CSR owner.
///
/// Deep structural validation is performed by `new`. The internal arrays are
/// private and no mutable access is exposed after construction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OutlierCsrHotCpu {
    rows: usize,
    cols: usize,
    row_ptr: Vec<u32>,
    col_idx: Vec<u32>,
    delta: Vec<F16Bits>,
}

/// Borrowed, matched column/delta view for one CSR row.
#[derive(Clone, Copy, Debug)]
pub struct OutlierCsrHotCpuRow<'a> {
    col_idx: &'a [u32],
    delta: &'a [F16Bits],
}

/// Proof token that every logical CSR row is reachable through checked
/// indexing.
///
/// This token can only be created by `OutlierCsrHotCpu::validate_rows`.
/// It immutably borrows the CSR owner, so the validated structure cannot
/// change before or during traversal.
#[derive(Clone, Copy, Debug)]
pub struct ValidatedOutlierCsrRows<'a> {
    csr: &'a OutlierCsrHotCpu,
}

/// Iterator over a previously validated CSR owner.
///
/// `next` has no recoverable error path. All row boundaries were checked
/// before this iterator was created.
#[derive(Clone, Debug)]
pub struct ValidatedOutlierCsrRowIter<'a> {
    csr: &'a OutlierCsrHotCpu,
    next_row: usize,
}

impl OutlierCsrHotCpu {
    pub fn new(
        rows: usize,
        cols: usize,
        row_ptr: Vec<u32>,
        col_idx: Vec<u32>,
        delta: Vec<F16Bits>,
    ) -> Result<Self, HotLayoutError> {
        let expected_row_ptr_len = rows.checked_add(1).ok_or(HotLayoutError::BadOutlierCsr)?;
        if row_ptr.len() != expected_row_ptr_len {
            return Err(HotLayoutError::BadOutlierCsr);
        }
        if col_idx.len() != delta.len() {
            return Err(HotLayoutError::BadOutlierCsr);
        }
        let nnz = u32::try_from(col_idx.len()).map_err(|_| HotLayoutError::BadOutlierCsr)?;
        if row_ptr.first().copied() != Some(0) {
            return Err(HotLayoutError::BadOutlierCsr);
        }
        if row_ptr.last().copied() != Some(nnz) {
            return Err(HotLayoutError::BadOutlierCsr);
        }
        for pair in row_ptr.windows(2) {
            if pair[0] > pair[1] || pair[1] > nnz {
                return Err(HotLayoutError::BadOutlierCsr);
            }
        }
        for &column in &col_idx {
            let column = usize::try_from(column).map_err(|_| HotLayoutError::BadOutlierCsr)?;
            if column >= cols {
                return Err(HotLayoutError::BadOutlierCsr);
            }
        }
        for &value in &delta {
            if !value.to_f32().is_finite() {
                return Err(HotLayoutError::BadOutlierCsr);
            }
        }
        Ok(Self {
            rows,
            cols,
            row_ptr,
            col_idx,
            delta,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    pub fn nnz(&self) -> usize {
        self.col_idx.len()
    }
    pub fn row_ptr(&self) -> &[u32] {
        &self.row_ptr
    }
    pub fn col_idx(&self) -> &[u32] {
        &self.col_idx
    }
    pub fn delta(&self) -> &[F16Bits] {
        &self.delta
    }

    /// Returns one matched CSR row using only checked access.
    pub fn row(&self, row: usize) -> Option<OutlierCsrHotCpuRow<'_>> {
        if row >= self.rows {
            return None;
        }
        let next = row.checked_add(1)?;
        let start = usize::try_from(*self.row_ptr.get(row)?).ok()?;
        let end = usize::try_from(*self.row_ptr.get(next)?).ok()?;
        if start > end {
            return None;
        }
        let col_idx = self.col_idx.get(start..end)?;
        let delta = self.delta.get(start..end)?;
        if col_idx.len() != delta.len() {
            return None;
        }
        Some(OutlierCsrHotCpuRow { col_idx, delta })
    }

    /// Preflight: verify that all rows are reachable through checked access.
    ///
    /// On success, returns a `ValidatedOutlierCsrRows` token that proves
    /// every future `row()` call will succeed. The token borrows the CSR
    /// owner, making it structurally frozen during the token's lifetime.
    ///
    /// This is an O(rows) operation. It does not scan column indexes or
    /// delta values.
    pub fn validate_rows(&self) -> Option<ValidatedOutlierCsrRows<'_>> {
        for row in 0..self.rows {
            self.row(row)?;
        }
        Some(ValidatedOutlierCsrRows { csr: self })
    }

    /// Returns the constructor-established CSR proof in O(1).
    ///
    /// `OutlierCsrHotCpu` has private storage and exposes no mutation. `new`
    /// validates every row boundary, so a successfully bound immutable owner
    /// can reuse this proof on each execution without another row scan.
    pub fn prevalidated_rows(&self) -> ValidatedOutlierCsrRows<'_> {
        ValidatedOutlierCsrRows { csr: self }
    }

    /// Compatibility range accessor.
    pub fn row_range(&self, row: usize) -> Option<core::ops::Range<usize>> {
        if row >= self.rows {
            return None;
        }
        let next = row.checked_add(1)?;
        let start = usize::try_from(*self.row_ptr.get(row)?).ok()?;
        let end = usize::try_from(*self.row_ptr.get(next)?).ok()?;
        if start > end {
            return None;
        }
        self.col_idx.get(start..end)?;
        self.delta.get(start..end)?;
        Some(start..end)
    }
}

impl<'a> OutlierCsrHotCpuRow<'a> {
    pub fn len(&self) -> usize {
        self.col_idx.len()
    }
    pub fn is_empty(&self) -> bool {
        self.col_idx.is_empty()
    }
    pub fn entries(&self) -> impl ExactSizeIterator<Item = (&'a u32, &'a F16Bits)> + 'a {
        self.col_idx.iter().zip(self.delta.iter())
    }
}

impl<'a> ValidatedOutlierCsrRows<'a> {
    pub fn iter(&self) -> ValidatedOutlierCsrRowIter<'a> {
        ValidatedOutlierCsrRowIter {
            csr: self.csr,
            next_row: 0,
        }
    }

    /// Returns a validated row.
    ///
    /// `None` would indicate a preflight failure, which cannot happen because
    /// the validated token was created by `validate_rows`.
    pub fn get(&self, row: usize) -> Option<OutlierCsrHotCpuRow<'a>> {
        self.csr.row(row)
    }

    pub fn csr(&self) -> &'a OutlierCsrHotCpu {
        self.csr
    }
}

impl<'a> Iterator for ValidatedOutlierCsrRowIter<'a> {
    type Item = OutlierCsrHotCpuRow<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_row >= self.csr.rows {
            return None;
        }
        let row = self.next_row;
        self.next_row += 1;
        self.csr.row(row)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.csr.rows.saturating_sub(self.next_row);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for ValidatedOutlierCsrRowIter<'_> {}
