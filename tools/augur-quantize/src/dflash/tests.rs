use super::*;
fn config() -> DFlashConfig {
    DFlashConfig::parse(include_bytes!("fixtures/config.json")).unwrap()
}
fn records() -> Vec<SafetensorsRecord> {
    let h=strict_json::parse(include_bytes!("fixtures/safetensors-header.json")).unwrap();
    h.as_object().unwrap().iter().filter(|(name,_)|name.as_str()!="__metadata__").map(|(name,v)|SafetensorsRecord {
        name:name.clone(),dtype:v["dtype"].as_str().unwrap().into(),
        shape:v["shape"].as_array().unwrap().iter().map(|v|v.as_u64().unwrap()).collect(),
        data_offsets:[v["data_offsets"][0].as_u64().unwrap(),v["data_offsets"][1].as_u64().unwrap()],
    }).collect()
}
#[test]
fn names_shapes_and_roles_match_fixed_llama_inventory() {
    let actual=mapped_tensors(&records(),&config()).unwrap();
    let expected=strict_json::parse(include_bytes!("fixtures/llama-tensor-inventory.json")).unwrap();
    let expected=expected["tensors"].as_array().unwrap();
    assert_eq!(actual.len(),81);assert_eq!(expected.len(),81);
    for got in &actual {
        let want=expected.iter().find(|v|v["name"]==got.tensor.name).unwrap();
        assert_eq!(serde_json::json!(got.tensor.dimensions),want["shape"]);
        if got.tensor.name=="selector_hidden.weight" {assert_eq!(got.tensor.output_bytes,256*5120*4);assert_eq!(got.tensor.output_type,0);continue;}
        assert_eq!(got.tensor.output_bytes,want["bytes"].as_u64().unwrap());
        assert_eq!(got.tensor.output_type,if want["type"]=="Q8_0" {8}else{0});
    }
    assert_eq!(actual.iter().filter(|v|v.tensor.output_type==8).count(),48);
    assert_eq!(actual.iter().filter(|v|v.tensor.output_type==0).count(),33);
    let mut prefixed=records();for r in &mut prefixed {r.name=format!("model.{}",r.name);}
    let prefixed=mapped_tensors(&prefixed,&config()).unwrap();
    assert_eq!(actual.iter().map(|v|&v.tensor.name).collect::<Vec<_>>(),prefixed.iter().map(|v|&v.tensor.name).collect::<Vec<_>>());
}
#[test]
fn wrong_missing_extra_alias_and_quantized_inputs_fail() {
    let c=config();let original=records();
    let mut r=original.clone();r.pop();assert!(mapped_tensors(&r,&c).is_err());
    let mut r=original.clone();r[0].shape[0]+=1;assert!(mapped_tensors(&r,&c).is_err());
    let mut r=original.clone();r[0].dtype="I8".into();assert!(mapped_tensors(&r,&c).is_err());
    let mut r=original.clone();r[0].data_offsets[1]+=2;assert!(mapped_tensors(&r,&c).is_err());
    let mut r=original.clone();let mut extra=r[0].clone();extra.name=format!("model.{}",extra.name);r.push(extra);
    assert!(mapped_tensors(&r,&c).is_err());
    let mut r=original;r[0].name="lm_head.weight".into();assert!(mapped_tensors(&r,&c).is_err());
}
#[test]
fn conv_base_expands_to_f32_and_preserves_signed_zero() {
    let values=[0_u16,0x8000,0x3f80,0xbf80,0x0001,0x7f7f];
    let raw:Vec<u8>=values.iter().flat_map(|v|v.to_le_bytes()).collect();
    let expected:Vec<u8>=values.iter().flat_map(|v|(u32::from(*v)<<16).to_le_bytes()).collect();
    let mut bill=ErrorBill::default();assert_eq!(encode_chunk(&raw,30,false,&mut bill).unwrap(),expected);
    assert_eq!(bill.elements,0);
    for kind in [0,1,30] { assert!(encode_chunk(&[0;3],kind,false,&mut bill).is_err()); }
    assert!(encode_chunk(&0x7fc0_u16.to_le_bytes(),30,false,&mut bill).is_err());
    assert!(encode_chunk(&[0;62],30,true,&mut bill).is_err());
}
#[test]
fn chunking_does_not_change_q8_bytes() {
    let raw:Vec<u8>=(0..96).map(|i|(((i%32) as f32-16.).to_bits()>>16) as u16).flat_map(u16::to_le_bytes).collect();
    let mut bill=ErrorBill::default();let full=encode_chunk(&raw,30,true,&mut bill).unwrap();
    let mut split=Vec::new();let mut separate=ErrorBill::default();
    for chunk in raw.chunks(64) {split.extend(encode_chunk(chunk,30,true,&mut separate).unwrap());}
    assert_eq!(full,split);assert_eq!(bill.elements,96);assert_eq!(bill.squared,separate.squared);
}
fn metadata_inputs() -> (Value,Value,Value) {
    (strict_json::parse(include_bytes!("fixtures/tokenizer-small.json")).unwrap(),
     strict_json::parse(include_bytes!("fixtures/tokenizer-config.json")).unwrap(),
     serde_json::json!({"bos_token_id":248044}))
}
#[test]
fn target_tokenizer_profile_and_added_token_contracts() {
    let (t,c,target)=metadata_inputs();
    let mut m=Metadata::new();metadata::tokenizer(&mut m,&t,&c,&target,"template",248320).unwrap();
    let header=m.header(1).unwrap();assert!(header.len()>1024);
    let mut altered=t.clone();altered["pre_tokenizer"]["pretokenizers"][0]["invert"]=true.into();
    assert!(metadata::tokenizer(&mut Metadata::new(),&altered,&c,&target,"template",248320).is_err());
    let mut altered=t.clone();altered["added_tokens"][0]["content"]="changed".into();
    assert!(metadata::tokenizer(&mut Metadata::new(),&altered,&c,&target,"template",248320).is_err());
    let mut altered=t;altered["model"]["vocab"]["duplicate-id"]=0.into();
    assert!(metadata::tokenizer(&mut Metadata::new(),&altered,&c,&target,"template",248320).is_err());
    let mut m=Metadata::new();m.integer("x",1).unwrap();assert!(m.integer("x",2).is_err());
}

#[test]
fn gate_expands_original_then_folds_once_with_chunk_independent_f32_rounding() {
    let raw:Vec<u8>=[0x3f80u16,0xbf80,0x3e01,0x8000].into_iter().flat_map(u16::to_le_bytes).collect();
    let full=encode_gate(&raw,30).unwrap();
    let split:Vec<u8>=raw.chunks_exact(2).flat_map(|b|encode_gate(b,30).unwrap()).collect();
    assert_eq!(full,split);
    assert_eq!(&full[..4],&0.45f32.to_le_bytes());
    assert_eq!(&full[4..8],&(-0.45f32).to_le_bytes());
    assert_eq!(&full[12..16],&(-0.0f32).to_le_bytes());
    assert!(encode_gate(&[0],30).is_err());
    assert!(encode_gate(&f32::INFINITY.to_le_bytes(),0).is_err());
    assert!(encode_gate(&[0;34],8).is_err());
}
