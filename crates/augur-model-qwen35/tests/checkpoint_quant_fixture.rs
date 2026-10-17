//! Actual checkpoint rows checked against separately captured Torch/NumPy values.
use augur_model_qwen35::{CheckpointFp8Matrix, CheckpointNvfp4Matrix, CheckpointNvfp4Scale};
use sha2::{Digest, Sha256};

#[test]
#[ignore = "CPU fixture from audited QUASAR/NVIDIA checkpoints; AUGUR_CHECKPOINT_QUANT_FIXTURE"]
fn actual_checkpoint_rows_match_independent_reference() {
    let path = std::env::var_os("AUGUR_CHECKPOINT_QUANT_FIXTURE").expect("fixture path");
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    assert_eq!(manifest["status"], "PASS_CPU_REFERENCE_EXTRACTION_ONLY");
    let mut elements = 0;
    for case in manifest["cases"].as_array().unwrap() {
        let read = |key: &str| {
            let bytes = std::fs::read(case[key]["path"].as_str().unwrap()).unwrap();
            assert_eq!(
                format!("{:x}", Sha256::digest(&bytes)),
                case[key]["sha256"].as_str().unwrap()
            );
            bytes
        };
        let rows = case["rows"].as_u64().unwrap() as usize;
        let cols = case["cols"].as_u64().unwrap() as usize;
        let weight = read("weight");
        let scales = read("scales");
        let reference = read("reference");
        let w = f32::from_bits(case["weight_scale_bits"].as_u64().unwrap() as u32);
        let x = f32::from_bits(case["input_scale_bits"].as_u64().unwrap() as u32);
        let mut actual = vec![0.0_f64; rows * cols];
        if case["family"] == "fp8" {
            CheckpointFp8Matrix::new(rows, cols, &weight, w, x)
                .unwrap()
                .decode_f64_into(&mut actual)
                .unwrap();
        } else {
            let global = if case["profile"] == "quasar" {
                CheckpointNvfp4Scale::CompressedTensors {
                    weight_global_scale: w,
                    input_global_scale: x,
                }
            } else {
                assert_eq!(case["profile"], "nvidia");
                CheckpointNvfp4Scale::ModelOpt {
                    weight_scale_2: w,
                    input_scale: x,
                }
            };
            CheckpointNvfp4Matrix::new(rows, cols, &weight, &scales, global)
                .unwrap()
                .decode_f64_into(&mut actual)
                .unwrap();
        }
        assert_eq!(reference.len(), actual.len() * 8);
        for (index, (a, bytes)) in actual.iter().zip(reference.chunks_exact(8)).enumerate() {
            let expected = f64::from_le_bytes(bytes.try_into().unwrap());
            assert!(a.is_finite() && expected.is_finite());
            assert_eq!(
                a.to_bits(),
                expected.to_bits(),
                "{} element{index}",
                case["name"]
            );
        }
        elements += actual.len();
        println!(
            "CHECKPOINT_MATH_CASE={} rows={rows} cols={cols} bit_equal=true",
            case["name"]
        );
    }
    assert_eq!(manifest["cases"].as_array().unwrap().len(), 18);
    println!("CHECKPOINT_MATH_ELEMENTS={elements}");
}
