use augur_backend::{CpuBuffer, CpuBufferData};
use augur_kernel_api::{HotLayoutError, Q8KActivationHotCpuBlock};

#[test]
fn cpu_buffer_owns_preallocated_q8hot() {
    let buffer = CpuBuffer::new_q8hot_zeroed(512).unwrap();
    let value = buffer.as_q8hot().unwrap();

    assert_eq!(value.len_elements(), 512);
    assert_eq!(value.block_count(), 2);
    assert!(value
        .blocks()
        .iter()
        .all(|b| b == &Q8KActivationHotCpuBlock::ZERO));
}

#[test]
fn cpu_buffer_exposes_fixed_q8hot_writer_owner() {
    let mut buffer = CpuBuffer::new_q8hot_zeroed(256).unwrap();
    let pointer_before = buffer.as_q8hot().unwrap().blocks().as_ptr();

    {
        let owner = buffer.as_q8hot_mut().unwrap();
        let mut writer = owner.writer();
        let block = writer.iter_mut().next().unwrap();
        block.qs.fill(7);
    }

    let owner = buffer.as_q8hot().unwrap();
    assert_eq!(owner.blocks().as_ptr(), pointer_before);
    assert!(owner.blocks()[0].qs.iter().all(|&v| v == 7));
}

#[test]
fn cpu_buffer_accepts_valid_q8hot_shape() {
    let buffer = CpuBuffer::new_q8hot_zeroed(512).unwrap();
    let q8hot = buffer.as_q8hot().unwrap();
    assert_eq!(q8hot.len_elements(), 512);
    assert_eq!(q8hot.block_count(), 2);
}

#[test]
fn cpu_buffer_rejects_invalid_q8hot_shape() {
    assert!(matches!(
        CpuBuffer::new_q8hot_zeroed(0),
        Err(HotLayoutError::BadShape)
    ));
    assert!(matches!(
        CpuBuffer::new_q8hot_zeroed(257),
        Err(HotLayoutError::BadShape)
    ));
}

#[test]
fn q8hot_buffer_variant_is_not_f32() {
    let buffer = CpuBuffer::new_q8hot_zeroed(256).unwrap();
    assert!(buffer.as_f32().is_none());
    assert!(matches!(buffer.data, CpuBufferData::Q8Hot(_)));
}
