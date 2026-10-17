use augur_core::F16Bits;
use augur_kernel_api::{HotLayoutError, Q8KActivationHotCpuBlock, Q8KActivationHotCpuVector};

#[test]
fn zeroed_owner_has_fixed_canonical_shape() {
    let owner = Q8KActivationHotCpuVector::new_zeroed(512).unwrap();

    assert_eq!(owner.len_elements(), 512);
    assert_eq!(owner.block_count(), 2);

    for block in owner.blocks() {
        assert_eq!(block, &Q8KActivationHotCpuBlock::ZERO);
        assert_eq!(block.d, F16Bits::ZERO);
        assert!(block.qs.iter().all(|&v| v == 0));
    }
}

#[test]
fn zeroed_owner_rejects_invalid_shape() {
    assert!(matches!(
        Q8KActivationHotCpuVector::new_zeroed(0),
        Err(HotLayoutError::BadShape)
    ));
    assert!(matches!(
        Q8KActivationHotCpuVector::new_zeroed(257),
        Err(HotLayoutError::BadShape)
    ));
}

#[test]
fn writer_overwrites_blocks_without_changing_owner_shape() {
    let mut owner = Q8KActivationHotCpuVector::new_zeroed(512).unwrap();

    let pointer_before = owner.blocks().as_ptr();
    let block_count_before = owner.block_count();
    let len_before = owner.len_elements();

    {
        let mut writer = owner.writer();
        assert_eq!(writer.block_count(), 2);
        assert!(!writer.is_empty());

        for (index, block) in writer.iter_mut().enumerate() {
            block.d = F16Bits::from_f32(index as f32 + 1.0);
            block.qs.fill(index as i8 + 1);
        }
    }

    assert_eq!(owner.blocks().as_ptr(), pointer_before);
    assert_eq!(owner.block_count(), block_count_before);
    assert_eq!(owner.len_elements(), len_before);
    assert_eq!(owner.blocks()[0].d, F16Bits::from_f32(1.0));
    assert!(owner.blocks()[0].qs.iter().all(|&v| v == 1));
    assert_eq!(owner.blocks()[1].d, F16Bits::from_f32(2.0));
    assert!(owner.blocks()[1].qs.iter().all(|&v| v == 2));
}

#[test]
fn repeated_writes_preserve_owner_allocation() {
    let mut owner = Q8KActivationHotCpuVector::new_zeroed(1024).unwrap();

    let pointer_before = owner.blocks().as_ptr();
    let block_count_before = owner.block_count();
    let len_before = owner.len_elements();

    for iteration in 0..16 {
        let mut writer = owner.writer();
        for block in writer.iter_mut() {
            block.d = F16Bits::from_f32(iteration as f32);
            block.qs.fill(iteration as i8);
        }
        assert_eq!(owner.blocks().as_ptr(), pointer_before);
        assert_eq!(owner.block_count(), block_count_before);
        assert_eq!(owner.len_elements(), len_before);
    }
}
