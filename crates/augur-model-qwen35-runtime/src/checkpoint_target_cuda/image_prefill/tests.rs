use super::*;
use augur_model_qwen35::{prepare_qwen35_aligned_rgb8_f16,Qwen35ImageGrid};
use crate::augur_session::{manifest,valid_digest};
use std::path::PathBuf;
use std::io::Write;
fn save(path:&Path,value:&Value) {
    let mut f=std::fs::OpenOptions::new().write(true).create_new(true).open(path).unwrap();
    serde_json::to_writer_pretty(&mut f,value).unwrap();f.write_all(b"\n").unwrap();f.sync_all().unwrap();
}
fn generate(target:&mut CheckpointCudaTarget,tokenizer:&tokenizers::Tokenizer,eos:&[u32],limit:usize)->(Vec<u32>,String) {
    let mut output=Vec::new();
    for i in 0..limit {
        let id=target.greedy_next_token().unwrap();if eos.contains(&id) {break;}
        output.push(id);if i+1<limit {target.consume_selected_augur().unwrap();}
    }
    target.logits_snapshot().unwrap();
    let text=tokenizer.decode(&output,false).unwrap();(output,text)
}
#[test]
#[ignore = "real Augur full262144 plus resident DFlash: authenticated image-dependent native answers"]
fn real_augur_vision_language_ingress() {
    let fixture=PathBuf::from(std::env::var_os("AUGUR_VISION_FIXTURE").expect("fixture"));
    let out=PathBuf::from(std::env::var_os("AUGUR_VISION_OUTPUT").expect("output"));
    std::fs::create_dir(&out).unwrap();
    let value:Value=serde_json::from_slice(&std::fs::read(&fixture).unwrap()).unwrap();
    let root=PathBuf::from(value["model_root"].as_str().unwrap());
    let admitted=manifest::load(&root,Path::new(value["manifest"].as_str().unwrap()),value["manifest_sha256"].as_str().unwrap()).unwrap();
    let tokenizer=tokenizers::Tokenizer::from_file(root.join("tokenizer.json")).unwrap();
    let before=cuda_device_allocation_snapshot();
    let started=Instant::now();
    let draft_load=crate::augur_session_cuda::augur_http::DraftLoad::begin().unwrap();
    let mut target=CheckpointCudaTarget::open_with_execution_policy(&root,Qwen35CheckpointProfile::QuasarNvfp4,
        admitted.manifest().checkpoint.clone(),262144,CheckpointProjectionPolicy::AugurQpnAllWorkspaceShapeSmallPairF64,
        CheckpointKvPolicy::Q8_0).unwrap();
    target.enable_exact_warp_tree_norm().unwrap();target.enable_fused_input_qpn_norm().unwrap();target.enable_batched_ffn8().unwrap();
    let info=target.info().clone();
    let (draft,info)=crate::augur_session_cuda::augur_http::prepare(&mut target,info,draft_load).unwrap();
    save(&out.join("startup.json"),&json!({"target":info,"text_dflash_startup_s":started.elapsed().as_secs_f64()}));
    let text_ids:Vec<u32>=serde_json::from_value(value["text_ids"].clone()).unwrap();
    let plan=target.plan_augur_sequence(&text_ids,vec![text_ids.len()]).unwrap();
    target.consume_sequence_augur(&text_ids,&plan,augur_prefill::Budget::default(),|_|false).unwrap();
    let text_before=generate(&mut target,&tokenizer,admitted.eos_token_ids(),24);
    target.reset_empty().unwrap();
    let mut responses=Vec::new();
    for sample in value["cases"].as_array().unwrap() {
        let ids:Vec<u32>=serde_json::from_value(sample["ids"].clone()).unwrap();
        assert_eq!(tokenizer.encode(sample["canonical"].as_str().unwrap(),false).unwrap().get_ids(),&ids);
        let mut images=Vec::new();
        for image in sample["images"].as_array().unwrap() {
            let rgb=std::fs::read(image["rgb_path"].as_str().unwrap()).unwrap();
            assert_eq!(crate::augur_session::digest(&rgb),image["sha256"].as_str().unwrap());
            let grid=Qwen35ImageGrid::from_aligned_pixels(image["height"].as_u64().unwrap() as usize,
                image["width"].as_u64().unwrap() as usize).unwrap();
            images.push(prepare_qwen35_aligned_rgb8_f16(&rgb,grid,32*1024*1024).unwrap());
        }
        let image_started=Instant::now();
        let mut receipt=target.prefill_images(&ids,&images,||false).unwrap();
        // Test-only isolation of encoder vs language-path errors. The native
        // encoder is executed above but its embeddings are explicitly discarded.
        // These substituted-reference runs cannot qualify native Vision or TPS.
        if let Some(reference)=sample.get("diagnostic_embedding_reference") {
            let raw=std::fs::read(reference["path"].as_str().unwrap()).unwrap();
            assert_eq!(crate::augur_session::digest(&raw),reference["sha256"].as_str().unwrap());
            assert_eq!(raw.len()%4,0);
            let embeddings:Vec<f32>=raw.chunks_exact(4).map(|x|f32::from_le_bytes(x.try_into().unwrap())).collect();
            target.reset_empty().unwrap();
            let kinds:Vec<_>=ids.iter().map(|&id|u8::from(id==248056)).collect();
            let grids:Vec<_>=images.iter().map(|x|x.grid).collect();
            let layout=Qwen35ImagePromptLayout::new(&ids,&kinds,&grids).unwrap();
            let mrope=Qwen35ImageMropePlan::new(&layout,target.capacity,target.capacity*32*8).unwrap();
            let rows=augur_prefill::image_rows::DeviceRows::upload(Rc::clone(&target.stream),&ids,
                layout.image_token_indices().to_vec(),&embeddings,json!({"diagnostic_reference":reference})).unwrap();
            let width=sample.get("diagnostic_prefill_width").and_then(Value::as_u64)
                .map(|v|v as usize).unwrap_or(ids.len());
            assert!(width>0 && width<=ids.len());
            let mut widths=vec![width;ids.len()/width];
            if ids.len()%width!=0 {widths.push(ids.len()%width);}
            let plan=target.plan_augur_sequence(&ids,widths).unwrap();
            target.image_rope_active=true;target.rope.install_image(&mrope).unwrap();
            let prefill=target.consume_sequence_augur_image_rows(&ids,&plan,&rows,|_|false).unwrap();
            receipt=json!({"native_encoder_discarded":receipt,"diagnostic_reference_prefill":prefill,
                "native_vision_qualified":false,"performance_qualified":false,"product_ready":false});
        }
        let (output_ids,text)=generate(&mut target,&tokenizer,admitted.eos_token_ids(),32);
        let result=json!({"name":sample["name"],"receipt":receipt,"output_ids":output_ids,"text":text,
            "image_to_generation_s":image_started.elapsed().as_secs_f64(),"expected":sample["expected"]});
        save(&out.join(format!("{}.json",sample["name"].as_str().unwrap())),&result);
        eprintln!("AUGUR_VISION name={} text={text:?}",sample["name"]);
        responses.push(result);target.reset_empty().unwrap();
        // Cancellation before any mutation must leave the empty owner reusable.
        assert!(target.prefill_images(&ids,&images,||true).is_err());
        assert_eq!(target.consumed,0);assert!(!target.poisoned);assert!(!target.image_rope_active);
        let polls=std::cell::Cell::new(0usize);let before_cancel=cuda_device_allocation_snapshot();
        assert!(target.prefill_images(&ids,&images,||{polls.set(polls.get()+1);polls.get()>=5}).is_err());
        let after_cancel=cuda_device_allocation_snapshot();
        assert_eq!(before_cancel.live_bytes,after_cancel.live_bytes);
        assert_eq!(before_cancel.live_allocations,after_cancel.live_allocations);
        assert_eq!(target.consumed,0);assert!(!target.poisoned);assert!(!target.image_rope_active);
    }
    let plan=target.plan_augur_sequence(&text_ids,vec![text_ids.len()]).unwrap();
    target.consume_sequence_augur(&text_ids,&plan,augur_prefill::Budget::default(),|_|false).unwrap();
    let text_after=generate(&mut target,&tokenizer,admitted.eos_token_ids(),24);
    assert_eq!(text_before,text_after,"image reset must restore original text table arithmetic");
    assert!(valid_digest(target.info["vision_host"]["converted_sha256"].as_str().unwrap()));
    let host=target.info["vision_host"].clone();
    drop(draft);drop(target);
    let after=cuda_device_allocation_snapshot();
    assert_eq!(before.live_bytes,after.live_bytes);assert_eq!(before.live_allocations,after.live_allocations);
    let semantic=responses.iter().all(|r|r["text"].as_str().unwrap().trim().trim_end_matches('.').eq_ignore_ascii_case(r["expected"].as_str().unwrap()));
    save(&out.join("complete.json"),&json!({"status":"NATIVE_IMAGE_INGRESS_EXECUTED","semantic_cases_pass":semantic,
        "responses":responses,"vision_host":host,"text_reset_output_ids_equal":true,"text_output":text_before.1,
        "allocation_closure":true,"live_bytes_after":after.live_bytes,"live_allocations_after":after.live_allocations,
        "physical_kv_capacity":262144,"dflash_resident":true,"dflash_active":false,"http_qualified":false,"product_ready":false}));
    assert!(semantic,"real image answers differ from expected; keep raw outputs and do not qualify product");
}
