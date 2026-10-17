#![forbid(unsafe_op_in_unsafe_fn)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use augur_backend::CpuStream;
use augur_compiler::{
    bind_resident_moe_plan, preflight_resident_moe_plan, CpuResidentMoeCompiler,
    CpuResidentMoeResource, CpuResidentMoeResources, ResidentMoeBindError,
};
use augur_core::{DType, LogicalLayout, Shape, StorageId, StorageRef, TensorDesc};
use augur_graph::{ResidentMoeExpertDesc, ResidentMoeGraph, ResidentMoeTensorRole};
use augur_kernel_api::{ResidentMoeF32HotCpu, ResidentMoeF32HotExpert};
use augur_reference::{
    resident_moe_f32_reference, ResidentMoeReferenceConfig, ResidentMoeReferenceExpert,
    ResidentMoeReferenceWeights,
};

const HIDDEN: usize = 8;
const INTERMEDIATE: usize = 12;
const EXPERTS: usize = 4;
const TOP_K: usize = 2;
const ROUTED_SCALE: f32 = 1.25;

struct AuditAllocator;
thread_local! {
    static AUDIT: Cell<bool> = const { Cell::new(false) };
    static COUNTS: Cell<[u64; 4]> = const { Cell::new([0; 4]) };
}

fn record_allocation(index: usize) {
    AUDIT.with(|enabled| {
        if enabled.get() {
            COUNTS.with(|counts| {
                let mut value = counts.get();
                value[index] += 1;
                counts.set(value);
            });
        }
    });
}

unsafe impl GlobalAlloc for AuditAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_allocation(0);
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record_allocation(1);
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record_allocation(2);
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        record_allocation(3);
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: AuditAllocator = AuditAllocator;

fn audited(call: impl FnOnce()) -> [u64; 4] {
    COUNTS.with(|counts| counts.set([0; 4]));
    AUDIT.with(|enabled| enabled.set(true));
    call();
    AUDIT.with(|enabled| enabled.set(false));
    COUNTS.with(Cell::get)
}

#[derive(Clone)]
struct RawExpert {
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
}

#[derive(Clone)]
struct RawWeights {
    router: Vec<f32>,
    experts: Vec<RawExpert>,
    shared: RawExpert,
}

fn values(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|index| ((index * 19 + seed * 23 + 7) % 101) as f32 / 211.0 - 0.24)
        .collect()
}

fn raw_expert(seed: usize) -> RawExpert {
    RawExpert {
        gate: values(HIDDEN * INTERMEDIATE, seed),
        up: values(HIDDEN * INTERMEDIATE, seed + 11),
        down: values(HIDDEN * INTERMEDIATE, seed + 29),
    }
}

fn raw_weights(tied_router: bool) -> RawWeights {
    let router = if tied_router {
        vec![0.0; EXPERTS * HIDDEN]
    } else {
        values(EXPERTS * HIDDEN, 41)
    };
    RawWeights {
        router,
        experts: (0..EXPERTS).map(|index| raw_expert(index * 31)).collect(),
        shared: raw_expert(197),
    }
}

fn weight_desc(storage: StorageId, shape: &[usize]) -> TensorDesc {
    TensorDesc {
        dtype: DType::F32,
        shape: Shape::from_dims(shape),
        layout: LogicalLayout::Contiguous,
        storage: StorageRef::HostOwned { id: storage },
        quant: None,
    }
}

fn graph(tokens: usize) -> ResidentMoeGraph {
    let activation_shape = if tokens == 1 {
        Shape::from_dims(&[HIDDEN])
    } else {
        Shape::from_dims(&[tokens, HIDDEN])
    };
    let mut storage = 1;
    let router = weight_desc(StorageId::new(0, 0), &[EXPERTS, HIDDEN]);
    let experts = (0..EXPERTS)
        .map(|_| {
            let result = ResidentMoeExpertDesc {
                gate: weight_desc(StorageId::new(storage, 0), &[INTERMEDIATE, HIDDEN]),
                up: weight_desc(StorageId::new(storage + 1, 0), &[INTERMEDIATE, HIDDEN]),
                down: weight_desc(StorageId::new(storage + 2, 0), &[HIDDEN, INTERMEDIATE]),
            };
            storage += 3;
            result
        })
        .collect::<Vec<_>>();
    let shared_expert = ResidentMoeExpertDesc {
        gate: weight_desc(StorageId::new(storage, 0), &[INTERMEDIATE, HIDDEN]),
        up: weight_desc(StorageId::new(storage + 1, 0), &[INTERMEDIATE, HIDDEN]),
        down: weight_desc(StorageId::new(storage + 2, 0), &[HIDDEN, INTERMEDIATE]),
    };
    ResidentMoeGraph {
        input: TensorDesc::f32_contiguous(activation_shape.clone()),
        router,
        experts: experts.into_boxed_slice(),
        shared_expert,
        output: TensorDesc::f32_contiguous(activation_shape),
        top_k: TOP_K,
        routed_scale: ROUTED_SCALE,
    }
}

fn hot_expert(raw: &RawExpert) -> ResidentMoeF32HotExpert {
    ResidentMoeF32HotExpert::new(
        HIDDEN,
        INTERMEDIATE,
        raw.gate.clone(),
        raw.up.clone(),
        raw.down.clone(),
    )
    .unwrap()
}

fn resources(graph: &ResidentMoeGraph, raw: &RawWeights) -> CpuResidentMoeResources {
    let analysis = graph.analyze().unwrap();
    let weights = ResidentMoeF32HotCpu::new(
        HIDDEN,
        INTERMEDIATE,
        raw.router.clone(),
        raw.experts.iter().map(hot_expert).collect(),
        hot_expert(&raw.shared),
    )
    .unwrap();
    CpuResidentMoeResources::new(CpuResidentMoeResource {
        weight_storages: analysis.weight_storages,
        weights,
    })
}

fn reference_expert(value: &RawExpert) -> ResidentMoeReferenceExpert<'_> {
    ResidentMoeReferenceExpert {
        gate: &value.gate,
        up: &value.up,
        down: &value.down,
    }
}

fn reference_weights(raw: &RawWeights) -> ResidentMoeReferenceWeights<'_> {
    ResidentMoeReferenceWeights {
        router: &raw.router,
        experts: raw.experts.iter().map(reference_expert).collect(),
        shared_expert: reference_expert(&raw.shared),
    }
}

fn input(tokens: usize, replay: usize) -> Vec<f32> {
    (0..tokens * HIDDEN)
        .map(|index| ((index * 17 + replay * 13 + 3) % 47) as f32 / 31.0 - 0.7)
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let limit = 1.0e-6 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= limit,
            "index={index} actual={actual} expected={expected} limit={limit}"
        );
    }
}

fn run_case(tokens: usize) {
    let graph = graph(tokens);
    let raw = raw_weights(false);
    let compiled = CpuResidentMoeCompiler.compile(&graph).unwrap();
    let accounting = preflight_resident_moe_plan(&compiled, &resources(&graph, &raw)).unwrap();
    assert_eq!(accounting.logical_weight_count, 16);
    assert_eq!(accounting.weight_payload_bytes, 5888);
    assert_eq!(accounting.input_payload_bytes, tokens * HIDDEN * 4);
    assert_eq!(accounting.output_payload_bytes, tokens * HIDDEN * 4);
    assert_eq!(
        accounting.workspace_payload_bytes,
        if tokens == 1 { 196 } else { 260 }
    );

    let mut plan = bind_resident_moe_plan(compiled, resources(&graph, &raw)).unwrap();
    let owner_identity = plan.weight_owner_identity();
    let buffer_identity = plan.buffer_identity();
    for replay in 0..8 {
        let input = input(tokens, replay);
        let expected = resident_moe_f32_reference(
            ResidentMoeReferenceConfig {
                tokens,
                hidden: HIDDEN,
                intermediate: INTERMEDIATE,
                top_k: TOP_K,
                routed_scale: ROUTED_SCALE,
            },
            &reference_weights(&raw),
            &input,
        )
        .unwrap();
        plan.set_input(&input).unwrap();
        plan.execute_cpu(&mut CpuStream).unwrap();
        assert_close(plan.output(), &expected.output);
        assert_close(plan.router_logits(), &expected.router_logits);
        assert_eq!(plan.selected_expert_ids(), expected.selected_ids);
        assert_close(plan.selected_routing_weights(), &expected.selected_weights);
        assert_eq!(plan.expert_access_counts(), expected.expert_access_counts);
        assert_eq!(plan.shared_access_count(), expected.shared_access_count);
        assert_eq!(plan.weight_owner_identity(), owner_identity);
        assert_eq!(plan.buffer_identity(), buffer_identity);
        assert_eq!(plan.accounting(), accounting);
        assert_eq!(
            plan.expert_access_counts()
                .iter()
                .map(|count| *count as usize)
                .sum::<usize>(),
            tokens * TOP_K
        );
    }
}

#[test]
fn decode_and_small_prefill_match_independent_reference() {
    run_case(1);
    run_case(3);
}

#[test]
fn stable_tie_break_never_accesses_unselected_experts() {
    let graph = graph(3);
    let raw = raw_weights(true);
    let compiled = CpuResidentMoeCompiler.compile(&graph).unwrap();
    let mut plan = bind_resident_moe_plan(compiled, resources(&graph, &raw)).unwrap();
    plan.set_input(&input(3, 0)).unwrap();
    plan.execute_cpu(&mut CpuStream).unwrap();
    assert_eq!(plan.selected_expert_ids(), &[0, 1, 0, 1, 0, 1]);
    assert_eq!(plan.expert_access_counts(), &[3, 3, 0, 0]);
    assert_eq!(plan.shared_access_count(), 3);
}

#[test]
fn stale_generation_preflight_preserves_resident_owner() {
    let graph = graph(1);
    let raw = raw_weights(false);
    let compiled = CpuResidentMoeCompiler.compile(&graph).unwrap();
    let mut stale_storages = graph.analyze().unwrap().weight_storages;
    let required = stale_storages.experts[2].up;
    stale_storages.experts[2].up.generation += 1;
    let actual = stale_storages.experts[2].up;
    let resources = CpuResidentMoeResources::new(CpuResidentMoeResource {
        weight_storages: stale_storages,
        weights: ResidentMoeF32HotCpu::new(
            HIDDEN,
            INTERMEDIATE,
            raw.router.clone(),
            raw.experts.iter().map(hot_expert).collect(),
            hot_expert(&raw.shared),
        )
        .unwrap(),
    });
    assert_eq!(
        preflight_resident_moe_plan(&compiled, &resources).unwrap_err(),
        ResidentMoeBindError::GenerationMismatch {
            role: ResidentMoeTensorRole::ExpertUp(2),
            required,
            actual,
        }
    );
    assert!(resources.is_available());
}

#[test]
fn cpu_execute_is_rust_allocation_free() {
    let graph = graph(3);
    let raw = raw_weights(false);
    let compiled = CpuResidentMoeCompiler.compile(&graph).unwrap();
    let mut plan = bind_resident_moe_plan(compiled, resources(&graph, &raw)).unwrap();
    plan.set_input(&input(3, 0)).unwrap();
    let counts = audited(|| plan.execute_cpu(&mut CpuStream).unwrap());
    assert_eq!(counts, [0; 4]);
    println!("M36_CPU_EXEC_ALLOC={counts:?}");
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_steady_replay_is_rust_allocation_free() {
    use augur_compiler::{bind_cuda_resident_moe_plan, CudaResidentMoeCompiler};
    let graph = graph(3);
    let raw = raw_weights(false);
    let compiled = CudaResidentMoeCompiler::new(0)
        .unwrap()
        .compile(&graph)
        .unwrap();
    let plan = bind_cuda_resident_moe_plan(compiled, resources(&graph, &raw)).unwrap();
    let input = input(3, 0);
    let mut output = vec![0.0_f32; input.len()];
    let counts = audited(|| {
        plan.set_input_f32(&input).unwrap();
        plan.execute_cuda().unwrap();
        plan.read_output_f32(&mut output).unwrap();
    });
    assert_eq!(counts, [0; 4]);
    println!("M36_CUDA_STEADY_REPLAY_ALLOC={counts:?}");
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_decode_and_prefill_match_reference_and_preserve_owners() {
    use augur_compiler::{
        bind_cuda_resident_moe_plan, preflight_cuda_resident_moe_plan, CudaResidentMoeCompiler,
    };

    for tokens in [1, 3] {
        let graph = graph(tokens);
        let raw = raw_weights(false);
        let compiled = CudaResidentMoeCompiler::new(0)
            .unwrap()
            .compile(&graph)
            .unwrap();
        let accounting =
            preflight_cuda_resident_moe_plan(&compiled, &resources(&graph, &raw)).unwrap();
        assert_eq!(accounting.device_allocation_count, 8);
        assert_eq!(accounting.weight_payload_bytes, 5888);
        assert_eq!(
            accounting.workspace_payload_bytes,
            if tokens == 1 { 196 } else { 548 }
        );
        assert_eq!(accounting.event_count, 2);

        let plan = bind_cuda_resident_moe_plan(compiled, resources(&graph, &raw)).unwrap();
        let owner_identity = plan.owner_identity();
        for replay in 0..8 {
            let input = input(tokens, replay);
            let expected = resident_moe_f32_reference(
                ResidentMoeReferenceConfig {
                    tokens,
                    hidden: HIDDEN,
                    intermediate: INTERMEDIATE,
                    top_k: TOP_K,
                    routed_scale: ROUTED_SCALE,
                },
                &reference_weights(&raw),
                &input,
            )
            .unwrap();
            plan.set_input_f32(&input).unwrap();
            if replay == 0 {
                let timing = plan.execute_cuda_profiled().unwrap();
                assert!(timing.kernel_event_ms > 0.0);
                assert!(timing.complete_host >= timing.launch_host);
            } else {
                plan.execute_cuda().unwrap();
            }
            let mut output = vec![0.0; tokens * HIDDEN];
            let mut logits = vec![0.0; tokens * EXPERTS];
            let mut ids = vec![u32::MAX; tokens * TOP_K];
            let mut route_weights = vec![0.0; tokens * TOP_K];
            let mut access = vec![0; EXPERTS + 1];
            plan.read_output_f32(&mut output).unwrap();
            plan.read_router_logits(&mut logits).unwrap();
            plan.read_selected_ids(&mut ids).unwrap();
            plan.read_selected_weights(&mut route_weights).unwrap();
            plan.read_access_counts(&mut access).unwrap();
            assert_close(&output, &expected.output);
            assert_close(&logits, &expected.router_logits);
            assert_eq!(ids, expected.selected_ids);
            assert_close(&route_weights, &expected.selected_weights);
            assert_eq!(&access[..EXPERTS], expected.expert_access_counts);
            assert_eq!(access[EXPERTS], expected.shared_access_count);
            assert_eq!(plan.owner_identity(), owner_identity);
            assert_eq!(plan.accounting(), accounting);
        }
    }
}
