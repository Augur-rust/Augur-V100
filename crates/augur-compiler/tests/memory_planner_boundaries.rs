use augur_compiler::MemoryPlanner;
use augur_core::{CompileError, TensorId};
use augur_exec::{BufferIdentity, BufferLifetimeEnd, BufferPurpose};

#[test]
fn memory_planner_rejects_out_of_range_tensor() {
    let mut planner = MemoryPlanner::new(1);
    let invalid = TensorId::new(5, 0);
    assert_eq!(
        planner.assign(invalid, 4, BufferPurpose::Intermediate, None),
        Err(CompileError::InvalidInput),
    );
}

#[test]
fn memory_planner_rejects_duplicate_assignment() {
    let mut planner = MemoryPlanner::new(1);
    let tensor = TensorId::new(0, 0);
    planner
        .assign(tensor, 4, BufferPurpose::Input, Some(0))
        .unwrap();
    assert_eq!(
        planner.assign(tensor, 4, BufferPurpose::Output, None),
        Err(CompileError::InvalidInput),
    );
}

#[test]
fn finalized_plan_rejects_stale_generation_lookup() {
    let mut planner = MemoryPlanner::new(1);
    let tensor = TensorId::new(0, 0);
    planner
        .assign(tensor, 4, BufferPurpose::Input, Some(0))
        .unwrap();
    let plan = planner.finish();
    let stale = TensorId::new(0, 1);
    assert_eq!(plan.buffer_for(stale), None);
}

#[test]
fn temporary_lifetime_uses_inclusive_compiled_steps() {
    let mut planner = MemoryPlanner::new(0);
    let (temporary, buffer_index) = planner
        .assign_temporary(256, BufferPurpose::HotActivation, 3)
        .unwrap();

    assert_eq!(buffer_index, 0);

    planner.extend_temporary_lifetime(temporary, 5).unwrap();

    // Lifetime extension is monotonic. An earlier consumer must not shrink
    // the existing inclusive end.
    planner.extend_temporary_lifetime(temporary, 4).unwrap();

    let plan = planner.finish();

    assert_eq!(plan.temporary_buffer(temporary), Some(0));
    assert_eq!(
        plan.buffers()[0].identity,
        BufferIdentity::Temporary(temporary)
    );
    assert_eq!(plan.buffers()[0].purpose, BufferPurpose::HotActivation);
    assert_eq!(
        plan.buffers()[0].reusable_after,
        Some(BufferLifetimeEnd::CompiledStep(5))
    );
    assert_eq!(plan.temporary_allocations()[0].producer_step, 3);
}

#[test]
fn temporary_lifetime_rejects_consumer_before_producer() {
    let mut planner = MemoryPlanner::new(0);
    let (temporary, _) = planner
        .assign_temporary(256, BufferPurpose::HotActivation, 3)
        .unwrap();

    assert_eq!(
        planner.extend_temporary_lifetime(temporary, 2),
        Err(CompileError::InvalidInput)
    );
}

#[test]
fn graph_intermediate_reuse_requires_strictly_disjoint_inclusive_lifetimes() {
    let mut planner = MemoryPlanner::new(3);
    let first = TensorId::new(0, 0);
    let overlaps_at_boundary = TensorId::new(1, 0);
    let disjoint = TensorId::new(2, 0);

    assert_eq!(
        planner
            .assign_reusing_tensor(first, 256, BufferPurpose::Intermediate, 0, 1)
            .unwrap(),
        0
    );
    assert_eq!(
        planner
            .assign_reusing_tensor(overlaps_at_boundary, 256, BufferPurpose::Intermediate, 1, 2,)
            .unwrap(),
        1
    );
    assert_eq!(
        planner
            .assign_reusing_tensor(disjoint, 128, BufferPurpose::Intermediate, 2, 3)
            .unwrap(),
        0
    );

    let plan = planner.finish();
    assert_eq!(plan.buffers().len(), 2);
    assert_eq!(plan.buffer_for(first), Some(0));
    assert_eq!(plan.buffer_for(overlaps_at_boundary), Some(1));
    assert_eq!(plan.buffer_for(disjoint), Some(0));
    assert_eq!(plan.reuse_transitions().len(), 1);
    assert_eq!(plan.reuse_transitions()[0].buffer_index, 0);
    assert_eq!(
        plan.reuse_transitions()[0].previous_reusable_after,
        BufferLifetimeEnd::GraphTopoStep(1)
    );
    assert_eq!(
        plan.reuse_transitions()[0].next_first_use,
        BufferLifetimeEnd::GraphTopoStep(2)
    );
}

#[test]
fn graph_intermediate_reuse_rejects_non_intermediate_purpose() {
    let mut planner = MemoryPlanner::new(1);
    assert_eq!(
        planner.assign_reusing_tensor(TensorId::new(0, 0), 16, BufferPurpose::Output, 0, 0,),
        Err(CompileError::InvalidInput)
    );
}

#[test]
fn graph_intermediate_reuse_skips_smaller_slot_without_eager_subtraction() {
    let mut planner = MemoryPlanner::new(2);
    planner
        .assign_reusing_tensor(TensorId::new(0, 0), 8, BufferPurpose::Intermediate, 0, 0)
        .unwrap();
    let second = planner
        .assign_reusing_tensor(TensorId::new(1, 0), 16, BufferPurpose::Intermediate, 1, 1)
        .unwrap();
    assert_eq!(second, 1);
    assert_eq!(planner.finish().buffers().len(), 2);
}
