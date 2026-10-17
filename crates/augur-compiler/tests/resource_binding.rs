use std::sync::Arc;

use augur_compiler::{
    bind_cpu_weight_resources, BindError, CpuWeightResources, GraphCompiler, ResourceLookupError,
};
use augur_core::{F16Bits, FormatVersion, QuantFormatId, Shape, StorageId, StorageRef, TensorDesc};
use augur_graph::Graph;
use augur_kernel_api::{Q4KHotCpuBlock, Q4KHotCpuTensor};

fn q4k_tensor(rows: usize, cols: usize) -> Q4KHotCpuTensor {
    assert!(cols.is_multiple_of(Q4KHotCpuBlock::DECODED_LEN));

    let blocks_per_row = cols / Q4KHotCpuBlock::DECODED_LEN;
    let block_count = rows.checked_mul(blocks_per_row).unwrap();

    let blocks = (0..block_count)
        .map(|seed| Q4KHotCpuBlock {
            d: F16Bits::from_f32(1.0 + (seed % 3) as f32 * 0.125),
            dmin: F16Bits::from_f32(0.5),
            scale: [1; 8],
            min: [1; 8],
            qs: [seed as u8; 128],
        })
        .collect();

    Q4KHotCpuTensor::new(rows, cols, blocks).unwrap()
}

fn compile_q4k_plan(storage: StorageId, rows: usize, cols: usize) -> augur_compiler::CpuCompiledPlan {
    let mut graph = Graph::new();

    let activation = graph.add_input(TensorDesc::f32_contiguous(Shape::from_dims(&[cols])));

    let weight = graph
        .add_weight(TensorDesc::quantized_weight(
            Shape::from_dims(&[rows, cols]),
            StorageRef::HostOwned { id: storage },
            QuantFormatId::Q4K,
            256,
            1,
        ))
        .unwrap();

    let output = graph.add_qmatmul(
        activation,
        weight,
        TensorDesc::f32_contiguous(Shape::from_dims(&[rows])),
    );

    graph.try_set_output(output).unwrap();

    GraphCompiler::cpu_baseline()
        .compile_cpu_plan(&graph, vec![vec![0.0_f32; cols]])
        .unwrap()
}

#[test]
fn binding_moves_hot_weight_into_arc_owner() {
    let mut resources = CpuWeightResources::new();

    let storage = resources
        .add_q4k_hot_cpu_portable(q4k_tensor(2, 256))
        .unwrap();

    let compiled = compile_q4k_plan(storage, 2, 256);
    let binding = compiled.weight_requirements()[0].binding;

    let bound = bind_cpu_weight_resources(compiled.weight_requirements(), resources).unwrap();

    let owner = bound.q4k_hot_cpu_portable(binding).unwrap();

    assert_eq!(owner.rows(), 2);
    assert_eq!(owner.cols(), 256);
    assert_eq!(Arc::strong_count(owner), 1);

    let accounting = bound.accounting();
    assert_eq!(accounting.unique_weight_count, 1);
    assert_eq!(accounting.q4k_hot_block_count, 2);
    assert_eq!(
        accounting.hot_weight_payload_bytes,
        std::mem::size_of_val(owner.blocks())
    );
}

#[test]
fn bound_owner_outlives_consumed_resource_store() {
    let mut resources = CpuWeightResources::new();

    let storage = resources
        .add_q4k_hot_cpu_portable(q4k_tensor(1, 256))
        .unwrap();

    let compiled = compile_q4k_plan(storage, 1, 256);
    let binding = compiled.weight_requirements()[0].binding;

    let bound = bind_cpu_weight_resources(compiled.weight_requirements(), resources).unwrap();

    let owner = bound.q4k_hot_cpu_portable(binding).unwrap();

    assert_eq!(owner.rows(), 1);
    assert_eq!(owner.cols(), 256);
}

#[test]
fn missing_resource_is_rejected_before_take() {
    let storage = StorageId::new(17, 0);

    let compiled = compile_q4k_plan(storage, 1, 256);

    let error =
        bind_cpu_weight_resources(compiled.weight_requirements(), CpuWeightResources::new())
            .unwrap_err();

    assert_eq!(
        error,
        BindError::ResourceLookup(ResourceLookupError::IndexOutOfRange { requested: storage })
    );
}

#[test]
fn stale_generation_is_rejected() {
    let mut resources = CpuWeightResources::new();

    let actual = resources
        .add_q4k_hot_cpu_portable(q4k_tensor(1, 256))
        .unwrap();

    let requested = StorageId {
        index: actual.index,
        generation: actual.generation + 1,
    };

    let compiled = compile_q4k_plan(requested, 1, 256);

    let error = bind_cpu_weight_resources(compiled.weight_requirements(), resources).unwrap_err();

    assert_eq!(
        error,
        BindError::ResourceLookup(ResourceLookupError::GenerationMismatch { requested, actual })
    );
}

#[test]
fn version_mismatch_is_rejected() {
    let mut resources = CpuWeightResources::new();

    let storage = resources
        .add_q4k_hot_cpu_portable_versioned(FormatVersion(2), q4k_tensor(1, 256))
        .unwrap();

    let compiled = compile_q4k_plan(storage, 1, 256);

    let error = bind_cpu_weight_resources(compiled.weight_requirements(), resources).unwrap_err();

    assert_eq!(
        error,
        BindError::VersionMismatch {
            storage,
            expected: FormatVersion(1),
            actual: FormatVersion(2),
        }
    );
}

#[test]
fn shape_mismatch_is_rejected() {
    let mut resources = CpuWeightResources::new();

    let storage = resources
        .add_q4k_hot_cpu_portable(q4k_tensor(3, 256))
        .unwrap();

    let compiled = compile_q4k_plan(storage, 2, 256);

    let error = bind_cpu_weight_resources(compiled.weight_requirements(), resources).unwrap_err();

    assert_eq!(
        error,
        BindError::ShapeMismatch {
            storage,
            expected_rows: 2,
            expected_cols: 256,
            actual_rows: 3,
            actual_cols: 256,
        }
    );
}

#[test]
fn add_only_compiled_plan_binds_empty_weight_table() {
    let mut graph = Graph::new();

    let desc = TensorDesc::f32_contiguous(Shape::from_dims(&[4]));

    let a = graph.add_input(desc.clone());
    let b = graph.add_input(desc.clone());
    let output = graph.add_add(a, b, desc);

    graph.try_set_output(output).unwrap();

    let compiled = GraphCompiler::cpu_baseline()
        .compile_cpu_plan(
            &graph,
            vec![vec![1.0, 2.0, 3.0, 4.0], vec![10.0, 20.0, 30.0, 40.0]],
        )
        .unwrap();

    let bound =
        bind_cpu_weight_resources(compiled.weight_requirements(), CpuWeightResources::new())
            .unwrap();

    assert!(bound.is_empty());
    assert_eq!(bound.accounting(), Default::default());
}
