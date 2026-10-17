#![cfg(feature = "cuda")]

use augur_backend::{CudaF32View, CudaSiluMulF32ArgsV1, OwnedCudaBuffer, OwnedCudaStream};
use augur_core::PhysicalLayoutId;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn f32_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(core::mem::size_of::<f32>())
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("f32 chunk")))
        .collect()
}

fn reference(gate: f32, up: f32) -> f32 {
    let sigmoid = if gate >= 0.0 {
        1.0 / (1.0 + (-gate).exp())
    } else {
        let exponent = gate.exp();
        exponent / (1.0 + exponent)
    };
    (gate * sigmoid) * up
}

#[test]
fn unclamped_silu_mul_matches_reference_and_replays() {
    let mut gate = vec![
        -20.0, -12.5, -10.0, -2.0, -0.5, 0.0, 0.5, 2.0, 10.0, 12.5, 20.0,
    ];
    gate.extend((0..245).map(|index| (index as f32 - 122.0) / 9.0));
    let up = gate
        .iter()
        .enumerate()
        .map(|(index, value)| 0.25 + value * 0.03125 + (index % 7) as f32 * 0.01)
        .collect::<Vec<_>>();
    let expected = gate
        .iter()
        .zip(&up)
        .map(|(&g, &u)| reference(g, u))
        .collect::<Vec<_>>();

    let stream = OwnedCudaStream::create(0).unwrap();
    let gate_bytes = f32_bytes(&gate);
    let up_bytes = f32_bytes(&up);
    let gate_buffer = OwnedCudaBuffer::allocate(0, gate_bytes.len()).unwrap();
    let up_buffer = OwnedCudaBuffer::allocate(0, up_bytes.len()).unwrap();
    let output_buffer = OwnedCudaBuffer::allocate(0, gate_bytes.len()).unwrap();
    gate_buffer
        .copy_from_host_sync(&stream, &gate_bytes)
        .unwrap();
    up_buffer.copy_from_host_sync(&stream, &up_bytes).unwrap();

    let args = CudaSiluMulF32ArgsV1::bind(
        &stream,
        CudaF32View::new(
            gate_buffer.view(0, gate_bytes.len()).unwrap(),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            gate.len(),
        )
        .unwrap(),
        CudaF32View::new(
            up_buffer.view(0, up_bytes.len()).unwrap(),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            up.len(),
        )
        .unwrap(),
        CudaF32View::new(
            output_buffer.view(0, gate_bytes.len()).unwrap(),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            gate.len(),
        )
        .unwrap(),
    )
    .unwrap();

    let mut captures = Vec::new();
    for _ in 0..2 {
        args.launch(&stream).unwrap();
        let mut bytes = vec![0_u8; gate_bytes.len()];
        output_buffer
            .copy_to_host_sync(&stream, &mut bytes)
            .unwrap();
        captures.push(bytes);
    }
    assert_eq!(captures[0], captures[1]);
    let actual = f32_values(&captures[0]);
    assert_eq!(actual.len(), expected.len());
    for (index, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
        assert!(got.is_finite(), "non-finite output at {index}: {got}");
        let tolerance = 3.0e-6_f32 + 3.0e-5_f32 * want.abs().max(1.0);
        assert!(
            (got - want).abs() <= tolerance,
            "index {index}: got {got}, want {want}, tolerance {tolerance}"
        );
    }
    // The values outside the old +/-10 policy ensure this is not silently
    // routed through the legacy clamped kernel.
    assert!((actual[10] - expected[10]).abs() <= 1.0e-3);
    assert!(actual[10].abs() > 15.0);
}

#[test]
fn submit_path_completes_without_changing_input() {
    let elements = 257;
    let gate = (0..elements)
        .map(|index| (index as f32 - 128.0) / 32.0)
        .collect::<Vec<_>>();
    let up = (0..elements)
        .map(|index| 1.0 - (index % 19) as f32 / 23.0)
        .collect::<Vec<_>>();
    let stream = OwnedCudaStream::create(0).unwrap();
    let gate_bytes = f32_bytes(&gate);
    let up_bytes = f32_bytes(&up);
    let gate_buffer = OwnedCudaBuffer::allocate(0, gate_bytes.len()).unwrap();
    let up_buffer = OwnedCudaBuffer::allocate(0, up_bytes.len()).unwrap();
    let output_buffer = OwnedCudaBuffer::allocate(0, gate_bytes.len()).unwrap();
    gate_buffer
        .copy_from_host_sync(&stream, &gate_bytes)
        .unwrap();
    up_buffer.copy_from_host_sync(&stream, &up_bytes).unwrap();
    let args = CudaSiluMulF32ArgsV1::bind(
        &stream,
        CudaF32View::new(
            gate_buffer.view(0, gate_bytes.len()).unwrap(),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            elements,
        )
        .unwrap(),
        CudaF32View::new(
            up_buffer.view(0, up_bytes.len()).unwrap(),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            elements,
        )
        .unwrap(),
        CudaF32View::new(
            output_buffer.view(0, gate_bytes.len()).unwrap(),
            PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            elements,
        )
        .unwrap(),
    )
    .unwrap();
    args.submit_for_plan(&stream).unwrap();
    let receipt = stream.submission_receipt().unwrap();
    stream.synchronize_submission(receipt).unwrap();

    let mut gate_after = vec![0_u8; gate_bytes.len()];
    let mut up_after = vec![0_u8; up_bytes.len()];
    gate_buffer
        .copy_to_host_sync(&stream, &mut gate_after)
        .unwrap();
    up_buffer.copy_to_host_sync(&stream, &mut up_after).unwrap();
    assert_eq!(gate_after, gate_bytes);
    assert_eq!(up_after, up_bytes);
}
