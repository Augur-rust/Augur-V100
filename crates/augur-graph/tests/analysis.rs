use augur_core::{Shape, TensorDesc};
use augur_graph::{analyze_graph, analyze_liveness, topo_sort, Graph};

#[test]
fn topo_sort_handles_add_chain() {
    let mut g = Graph::new();
    let desc = TensorDesc::f32_contiguous(Shape::from_dims(&[3]));
    let a = g.add_input(desc.clone());
    let b = g.add_input(desc.clone());
    let y1 = g.add_add(a, b, desc.clone());
    let y2 = g.add_add(y1, a, desc);
    g.try_set_output(y2).unwrap();

    let topo = topo_sort(&g).unwrap();
    assert_eq!(topo.len(), 2);
    assert_eq!(topo[0].index, 0);
    assert_eq!(topo[1].index, 1);
}

#[test]
fn topo_sort_handles_independent_nodes() {
    let mut g = Graph::new();
    let desc = TensorDesc::f32_contiguous(Shape::from_dims(&[2]));
    let a = g.add_input(desc.clone());
    let b = g.add_input(desc.clone());
    let c = g.add_input(desc.clone());
    let d = g.add_input(desc.clone());
    let y1 = g.add_add(a, b, desc.clone());
    let y2 = g.add_add(c, d, desc);
    g.try_set_output(y1).unwrap();
    g.try_set_output(y2).unwrap();

    let topo = topo_sort(&g).unwrap();
    assert_eq!(topo.len(), 2);
}

#[test]
fn topo_sort_preserves_fanout_and_duplicate_dependency_order() {
    let mut g = Graph::new();
    let desc = TensorDesc::f32_contiguous(Shape::from_dims(&[2]));
    let a = g.add_input(desc.clone());
    let b = g.add_input(desc.clone());
    let root = g.add_add(a, b, desc.clone());
    let duplicate = g.add_add(root, root, desc.clone());
    let fanout = g.add_add(root, a, desc.clone());
    let output = g.add_add(duplicate, fanout, desc);
    g.try_set_output(output).unwrap();

    let topo = topo_sort(&g).unwrap();
    assert!(topo.iter().map(|node| node.index).eq([0, 1, 2, 3]));
}

#[test]
fn liveness_tracks_chain_ranges() {
    let mut g = Graph::new();
    let desc = TensorDesc::f32_contiguous(Shape::from_dims(&[3]));
    let a = g.add_input(desc.clone());
    let b = g.add_input(desc.clone());
    let y1 = g.add_add(a, b, desc.clone());
    let y2 = g.add_add(y1, a, desc);
    g.try_set_output(y2).unwrap();

    let live = analyze_liveness(&g).unwrap();
    assert_eq!(live.range(a).unwrap().first_step, 0);
    assert_eq!(live.range(a).unwrap().last_step, 1);
    assert_eq!(live.range(b).unwrap().first_step, 0);
    assert_eq!(live.range(b).unwrap().last_step, 0);
    assert_eq!(live.range(y1).unwrap().first_step, 0);
    assert_eq!(live.range(y1).unwrap().last_step, 1);
    assert_eq!(live.range(y2).unwrap().first_step, 1);
    assert_eq!(live.range(y2).unwrap().last_step, 2);
}

#[test]
fn integrated_analysis_matches_standalone_topology_and_liveness() {
    let mut graph = Graph::new();
    let desc = TensorDesc::f32_contiguous(Shape::from_dims(&[3]));
    let a = graph.add_input(desc.clone());
    let b = graph.add_input(desc.clone());
    let root = graph.add_add(a, b, desc.clone());
    let left = graph.add_add(root, root, desc.clone());
    let right = graph.add_add(root, a, desc.clone());
    let output = graph.add_add(left, right, desc);
    graph.try_set_output(output).unwrap();

    let standalone_topo = topo_sort(&graph).unwrap();
    let standalone_liveness = analyze_liveness(&graph).unwrap();
    let integrated = analyze_graph(&graph).unwrap();

    assert!(integrated.topo().iter().eq(standalone_topo.iter()));
    for slot in graph.tensors() {
        assert_eq!(
            integrated.liveness().range(slot.id),
            standalone_liveness.range(slot.id)
        );
    }
}
