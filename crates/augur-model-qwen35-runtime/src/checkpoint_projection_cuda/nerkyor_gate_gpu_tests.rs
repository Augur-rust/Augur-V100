//! Small synthetic ownership test. This does not qualify model numerics.
use super::*;
use augur_backend::cuda_device_allocation_snapshot;
fn weight(
    stream: &Rc<OwnedCudaStream>,
    name: &str,
    policy: CheckpointProjectionPolicy,
) -> Rc<CheckpointScalarCudaWeight> {
    Rc::new(CheckpointScalarCudaWeight {
        storage: Storage::Bf16(upload(stream, &vec![0; 48 * 5120 * 2]).unwrap()),
        spec: CheckpointLinearSpec {
            name: name.into(),
            rows: 48,
            cols: 5120,
            storage: CheckpointLinearStorage::Bf16,
        },
        stream: Rc::clone(stream),
        policy,
    })
}
#[test]
#[ignore = "exclusive SM70: small retained gate owner/alias negative contracts"]
fn retained_gate_rejects_foreign_owners_and_keeps_both_weights_alive() {
    let before = cuda_device_allocation_snapshot();
    {
        let stream = Rc::new(OwnedCudaStream::create(0).unwrap());
        let foreign = Rc::new(OwnedCudaStream::create(0).unwrap());
        let policy = CheckpointProjectionPolicy::NerkyorBf16GateF32CtaResearch;
        let a = weight(
            &stream,
            "model.language_model.layers.0.linear_attn.in_proj_a",
            policy,
        );
        let b = weight(
            &stream,
            "model.language_model.layers.0.linear_attn.in_proj_b",
            policy,
        );
        let input = Rc::new(upload(&stream, &vec![0; 2 * 5120 * 4]).unwrap());
        let output_a = Rc::new(upload(&stream, &vec![0; 48 * 4]).unwrap());
        let output_b = Rc::new(upload(&stream, &vec![0; 48 * 4]).unwrap());
        let left = || {
            a.bind(Rc::clone(&input), 0, Rc::clone(&output_a), 0)
                .unwrap()
        };
        for (name, stream, policy) in [
            (
                "model.language_model.layers.1.linear_attn.in_proj_b",
                &stream,
                policy,
            ),
            (
                "model.language_model.layers.0.linear_attn.in_proj_a",
                &stream,
                policy,
            ),
            (
                "model.language_model.layers.0.linear_attn.in_proj_b",
                &foreign,
                policy,
            ),
            (
                "model.language_model.layers.0.linear_attn.in_proj_b",
                &stream,
                CheckpointProjectionPolicy::OriginalF32,
            ),
        ] {
            let wrong = weight(stream, name, policy);
            let right = wrong
                .bind(Rc::clone(&input), 0, Rc::clone(&output_b), 0)
                .unwrap();
            assert!(NerkyorBf16GatePair::bind(left(), right).is_err());
        }
        let foreign_input = Rc::new(upload(&stream, &vec![0; 5120 * 4]).unwrap());
        for (x, offset, y) in [
            (&foreign_input, 0, &output_b),
            (&input, 5120 * 4, &output_b),
            (&input, 0, &output_a),
        ] {
            let right = b.bind(Rc::clone(x), offset, Rc::clone(y), 0).unwrap();
            assert!(NerkyorBf16GatePair::bind(left(), right).is_err());
        }
        let weak_a = Rc::downgrade(&a);
        let weak_b = Rc::downgrade(&b);
        let weak_x = Rc::downgrade(&input);
        let pair = NerkyorBf16GatePair::bind(
            left(),
            b.bind(Rc::clone(&input), 0, Rc::clone(&output_b), 0)
                .unwrap(),
        )
        .unwrap();
        drop(a);
        drop(b);
        drop(input);
        drop(output_a);
        drop(output_b);
        assert!(
            weak_a.upgrade().is_some() && weak_b.upgrade().is_some() && weak_x.upgrade().is_some()
        );
        pair.submit_for_plan().unwrap();
        stream.synchronize().unwrap();
        drop(pair);
        assert!(
            weak_a.upgrade().is_none() && weak_b.upgrade().is_none() && weak_x.upgrade().is_none()
        );
    }
    let after = cuda_device_allocation_snapshot();
    assert_eq!(before.live_bytes, after.live_bytes);
    assert_eq!(before.live_allocations, after.live_allocations);
}
