use augur_core::F16Bits;
use augur_kernel_api::{HotLayoutError, OutlierCsrHotCpu};

fn one() -> F16Bits {
    F16Bits::from_f32(1.0)
}

#[test]
fn valid_csr_rows_are_resolved_without_panicking_indexing() {
    let csr = OutlierCsrHotCpu::new(
        2,
        8,
        vec![0, 2, 3],
        vec![0, 7, 3],
        vec![one(), one(), one()],
    )
    .unwrap();
    let row0 = csr.row(0).unwrap();
    let row1 = csr.row(1).unwrap();
    assert_eq!(row0.len(), 2);
    assert_eq!(row1.len(), 1);
    let row0_entries: Vec<_> = row0.entries().map(|(&c, &d)| (c, d)).collect();
    assert_eq!(row0_entries, vec![(0, one()), (7, one())]);
    assert!(csr.row(2).is_none());
}

#[test]
fn empty_csr_is_valid() {
    let csr = OutlierCsrHotCpu::new(0, 8, vec![0], Vec::new(), Vec::new()).unwrap();
    assert_eq!(csr.rows(), 0);
    assert_eq!(csr.nnz(), 0);
    assert!(csr.row(0).is_none());
}

#[test]
fn validate_rows_pass_on_valid_csr() {
    let csr = OutlierCsrHotCpu::new(
        2,
        8,
        vec![0, 2, 3],
        vec![0, 7, 3],
        vec![one(), one(), one()],
    )
    .unwrap();
    assert!(csr.validate_rows().is_some());
}

#[test]
fn rejects_bad_row_pointer_length() {
    assert_eq!(
        OutlierCsrHotCpu::new(2, 8, vec![0, 0], Vec::new(), Vec::new()),
        Err(HotLayoutError::BadOutlierCsr)
    );
}

#[test]
fn rejects_nonzero_first_row_pointer() {
    assert_eq!(
        OutlierCsrHotCpu::new(1, 8, vec![1, 1], vec![0], vec![one()]),
        Err(HotLayoutError::BadOutlierCsr)
    );
}

#[test]
fn rejects_non_monotonic_row_pointer() {
    assert_eq!(
        OutlierCsrHotCpu::new(2, 8, vec![0, 2, 1], vec![0], vec![one()]),
        Err(HotLayoutError::BadOutlierCsr)
    );
}

#[test]
fn rejects_terminal_nnz_mismatch() {
    assert_eq!(
        OutlierCsrHotCpu::new(1, 8, vec![0, 0], vec![0], vec![one()]),
        Err(HotLayoutError::BadOutlierCsr)
    );
}

#[test]
fn rejects_column_delta_length_mismatch() {
    assert_eq!(
        OutlierCsrHotCpu::new(1, 8, vec![0, 1], vec![0], Vec::new()),
        Err(HotLayoutError::BadOutlierCsr)
    );
}

#[test]
fn rejects_column_outside_tensor() {
    assert_eq!(
        OutlierCsrHotCpu::new(1, 8, vec![0, 1], vec![8], vec![one()]),
        Err(HotLayoutError::BadOutlierCsr)
    );
}

#[test]
fn rejects_non_finite_delta() {
    let inf = F16Bits::from_f32(f32::INFINITY);
    assert_eq!(
        OutlierCsrHotCpu::new(1, 8, vec![0, 1], vec![0], vec![inf]),
        Err(HotLayoutError::BadOutlierCsr)
    );
}

#[test]
fn validated_token_iterator_covers_all_rows() {
    let csr = OutlierCsrHotCpu::new(
        3,
        16,
        vec![0, 2, 3, 3],
        vec![0, 7, 3],
        vec![one(), one(), one()],
    )
    .unwrap();
    let validated = csr.validate_rows().unwrap();
    let rows: Vec<_> = validated.iter().collect();
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0].len(), 2);
    assert!(rows[2].is_empty());
}
