//! Opt-in interoperability check against a real GPU-produced component fixture.
use augur_model_qwen35::Qwen35Nvfp4QpnGeometry;
use std::path::PathBuf;

#[test]
#[ignore = "requires explicit AUGUR_QWEN35_QPN_FIXTURE_ROOT from the GPU component run"]
fn real_ggml_weights_match_gpu_qpn_planes_exactly() {
    let root = PathBuf::from(std::env::var_os("AUGUR_QWEN35_QPN_FIXTURE_ROOT").expect("fixture root"));
    let g = Qwen35Nvfp4QpnGeometry::new(17408, 5120).unwrap();
    for projection in ["gate", "up"] {
        let ggml = std::fs::read(root.join(format!("{projection}.ggml"))).unwrap();
        let mut expected =
            std::fs::read(root.join(format!("{projection}.gpu_prepacked_codes"))).unwrap();
        expected.extend(
            std::fs::read(root.join(format!("{projection}.gpu_prepacked_scales"))).unwrap(),
        );
        let mut actual = vec![0; g.payload_bytes()];
        g.encode_ggml_into(&ggml, &mut actual).unwrap();
        assert!(
            actual == expected,
            "{projection}: Rust/GPU physical bytes differ"
        );
        g.decode_ggml_into(&actual, &mut expected).unwrap();
        assert!(expected == ggml, "{projection}: source words changed");
    }
}
