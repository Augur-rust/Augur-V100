use super::*;

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "augur-rust-q8-test-{}-{}",
            std::process::id(),
            TEMP_ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn input(&self, kind: u32, cols: u64, invalid: bool) -> PathBuf {
        let path = self.0.join("input.gguf");
        let mut header = Vec::new();
        header.extend_from_slice(b"GGUF");
        header.extend_from_slice(&3_u32.to_le_bytes());
        header.extend_from_slice(&2_u64.to_le_bytes());
        header.extend_from_slice(&5_u64.to_le_bytes());
        string(&mut header, "general.architecture");
        header.extend_from_slice(&8_u32.to_le_bytes());
        string(&mut header, "test");
        string(&mut header, "general.file_type");
        header.extend_from_slice(&4_u32.to_le_bytes());
        header.extend_from_slice(&0_u32.to_le_bytes());
        string(&mut header, "general.alignment");
        header.extend_from_slice(&4_u32.to_le_bytes());
        header.extend_from_slice(&64_u32.to_le_bytes());
        string(&mut header, "custom.float");
        header.extend_from_slice(&6_u32.to_le_bytes());
        header.extend_from_slice(&1.25_f32.to_le_bytes());
        string(&mut header, "custom.strings");
        header.extend_from_slice(&9_u32.to_le_bytes());
        header.extend_from_slice(&8_u32.to_le_bytes());
        header.extend_from_slice(&2_u64.to_le_bytes());
        string(&mut header, "one");
        string(&mut header, "二");
        let width = if kind == 0 { 4 } else { 2 };
        let matrix_bytes = cols * 34 * width;
        let norm_offset = align(matrix_bytes, 64).unwrap();
        string(&mut header, "blk.0.attn_q.weight");
        header.extend_from_slice(&2_u32.to_le_bytes());
        header.extend_from_slice(&cols.to_le_bytes());
        header.extend_from_slice(&34_u64.to_le_bytes());
        header.extend_from_slice(&kind.to_le_bytes());
        header.extend_from_slice(&0_u64.to_le_bytes());
        string(&mut header, "output_norm.weight");
        header.extend_from_slice(&1_u32.to_le_bytes());
        header.extend_from_slice(&4_u64.to_le_bytes());
        header.extend_from_slice(&0_u32.to_le_bytes());
        header.extend_from_slice(&norm_offset.to_le_bytes());
        let start = align(header.len() as u64, 64).unwrap();
        header.resize(start as usize, 0);
        for i in 0..cols * 34 {
            let value = if invalid && i == 33 {
                f32::NAN
            } else {
                (i % 32) as f32 - 16.0
            };
            match kind {
                0 => header.extend_from_slice(&value.to_le_bytes()),
                1 => header.extend_from_slice(&F16Bits::from_f32(value).to_bits().to_le_bytes()),
                30 => header.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes()),
                _ => unreachable!(),
            }
        }
        header.resize((start + norm_offset) as usize, 0);
        for value in [1.0_f32, 0.5, -0.25, 2.0] {
            header.extend_from_slice(&value.to_le_bytes());
        }
        fs::write(&path, header).unwrap();
        path
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        // Fixtures are unique; retain unknown files instead of recursively deleting.
        if let Ok(entries) = fs::read_dir(&self.0) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_dir() {
                    let _ = fs::remove_file(p.join("model-q8_0.gguf"));
                    let _ = fs::remove_file(p.join("receipt.json"));
                    let _ = fs::remove_dir(&p);
                } else {
                    let _ = fs::remove_file(p);
                }
            }
        }
        let _ = fs::remove_dir(&self.0);
    }
}

#[test]
fn f32_f16_bf16_convert_and_preserve_metadata_and_norm() {
    for kind in [0, 1, 30] {
        let fixture = Fixture::new();
        let input = fixture.input(kind, 32, false);
        let original = fs::read(&input).unwrap();
        let out = fixture.0.join("output");
        let receipt = convert(&input, &out, 4096).unwrap();
        let result = out.join("model-q8_0.gguf");
        let m = GgufManifest::inspect(&result, GgufAuditOptions::default()).unwrap();
        assert_eq!(m.alignment, 64);
        assert_eq!(m.metadata("general.file_type").unwrap().as_u64(), Some(7));
        assert_eq!(
            m.metadata("general.quantization_version").unwrap().as_u64(),
            Some(2)
        );
        assert_eq!(m.metadata("custom.float").unwrap().as_f64(), Some(1.25));
        assert_eq!(m.metadata("custom.strings").unwrap().array_len(), Some(2));
        let matrix = m.read_tensor(&result, "blk.0.attn_q.weight").unwrap();
        let expected =
            quantize_q8_0_block_ref(&(0..32).map(|i| i as f32 - 16.0).collect::<Vec<_>>())
                .unwrap()
                .to_le_bytes();
        assert_eq!(matrix, expected.repeat(34));
        let norm = m.read_tensor(&result, "output_norm.weight").unwrap();
        assert_eq!(
            norm,
            [1.0_f32, 0.5, -0.25, 2.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>()
        );
        assert_eq!(fs::read(&input).unwrap(), original);
        assert_eq!(receipt["output_sha256"].as_str(), m.file_sha256.as_deref());
        assert_eq!(receipt["plan"]["output_bytes"].as_u64(), Some(m.file_size));
    }
}

#[test]
fn conversion_is_independent_of_chunk_size() {
    let fixture = Fixture::new();
    let input = fixture.input(0, 32, false);
    let small = fixture.0.join("small");
    let large = fixture.0.join("large");
    let a = convert(&input, &small, 4096).unwrap();
    let b = convert(&input, &large, 8192).unwrap();
    assert_eq!(a["output_sha256"], b["output_sha256"]);
}

#[test]
fn failure_leaves_no_output_or_temporary_artifact() {
    let fixture = Fixture::new();
    let input = fixture.input(0, 32, true);
    let out = fixture.0.join("output");
    assert!(convert(&input, &out, 4096)
        .unwrap_err()
        .to_string()
        .contains("nonfinite"));
    assert!(!out.exists());
    assert_eq!(fs::read_dir(&fixture.0).unwrap().count(), 1);
}

#[test]
fn refuses_overwrite_requantization_and_incompatible_matrices() {
    let fixture = Fixture::new();
    let input = fixture.input(0, 32, false);
    let out = fixture.0.join("output");
    convert(&input, &out, 4096).unwrap();
    assert!(convert(&input, &out, 4096)
        .unwrap_err()
        .to_string()
        .contains("exists"));
    assert!(
        convert(&out.join("model-q8_0.gguf"), &fixture.0.join("again"), 4096)
            .unwrap_err()
            .to_string()
            .contains("expected float")
    );
    let second = Fixture::new();
    let input2 = second.input(0, 31, false);
    assert!(convert(&input2, &second.0.join("output"), 4096)
        .unwrap_err()
        .to_string()
        .contains("no Q8_0-compatible"));
}

#[test]
fn invalid_chunk_is_rejected_before_output_creation() {
    let fixture = Fixture::new();
    let input = fixture.input(0, 32, false);
    for chunk in [0, 1, 4097, 16 * 1024 * 1024] {
        assert!(convert(&input, &fixture.0.join("output"), chunk).is_err());
    }
    assert_eq!(fs::read_dir(&fixture.0).unwrap().count(), 1);
}

#[test]
fn mapped_dflash_keeps_three_dimensional_convolution_in_f32() {
    let fixture=Fixture::new();let input=fixture.0.join("mapped-dflash.gguf");
    let mut header=b"GGUF".to_vec();header.extend_from_slice(&3_u32.to_le_bytes());
    header.extend_from_slice(&2_u64.to_le_bytes());header.extend_from_slice(&1_u64.to_le_bytes());
    string(&mut header,"general.architecture");header.extend_from_slice(&8_u32.to_le_bytes());string(&mut header,"dflash");
    for (name,dims,offset) in [("blk.0.attn_conv_base",vec![32_u64,2,2],0_u64),
        ("blk.0.attn_q.weight",vec![32,1],512)] {
        string(&mut header,name);header.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in dims {header.extend_from_slice(&d.to_le_bytes());}
        header.extend_from_slice(&0_u32.to_le_bytes());header.extend_from_slice(&offset.to_le_bytes());
    }
    header.resize(align(header.len() as u64,32).unwrap() as usize,0);
    let base:Vec<u8>=(0..128).flat_map(|i|(i as f32/16.).to_le_bytes()).collect();
    header.extend_from_slice(&base);header.extend((0..32).flat_map(|i|(i as f32-16.).to_le_bytes()));
    fs::write(&input,header).unwrap();let out=fixture.0.join("converted");convert(&input,&out,4096).unwrap();
    let path=out.join("model-q8_0.gguf");let m=GgufManifest::inspect(&path,GgufAuditOptions::default()).unwrap();
    assert_eq!(m.tensor("blk.0.attn_conv_base").unwrap().ggml_type,0);
    assert_eq!(m.tensor("blk.0.attn_q.weight").unwrap().ggml_type,8);
    assert_eq!(m.read_tensor(&path,"blk.0.attn_conv_base").unwrap(),base);
}
