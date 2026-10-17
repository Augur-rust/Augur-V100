//! Test-only partition ledger. Fixed-token comparisons precede semantic grading.
use super::*;
use crate::checkpoint_target_cuda::augur_prefill::{Budget, Progress};
use crate::checkpoint_linear_cuda::augur_prefill::{Workspace, arena};
const H: usize = 5120;
fn blob(root:&Path,name:&str,raw:&[u8]) { let mut f=std::fs::OpenOptions::new().create_new(true).write(true).open(root.join(name)).unwrap();f.write_all(raw).unwrap(); }
fn floats(root:&Path,name:&str,x:&[f32]) { assert!(x.iter().all(|v|v.is_finite()));blob(root,name,&x.iter().flat_map(|x|x.to_le_bytes()).collect::<Vec<_>>()); }
fn checkpoint(t:&mut CheckpointCudaTarget,ids:&[u32],root:&Path)->Value {
    assert_eq!(t.consumed,ids.len());std::fs::create_dir(root).unwrap();let meta=descriptor(t,ids);save(root,"descriptor.json",&meta);
    let mut io=state_io(t,&meta);let digest=io.state_digest().unwrap();let plan=io.plan.clone();
    let mut writer=LocalCheckpointWriter::create(&root.join("native"),&plan).unwrap();
    let r=transfer::save(&mut io,&mut writer,&plan,transfer::MAX_CHUNK_BYTES).unwrap();
    let v=json!({"descriptor_sha256":hash(&std::fs::read(root.join("descriptor.json")).unwrap()),"manifest_sha256":r.manifest_sha256,"state_sha256":digest,"payload_bytes":r.payload_bytes});save(root,"identity.json",&v);v
}
fn rewind(t:&mut CheckpointCudaTarget,ids:&[u32],root:&Path,identity:&Value) {
    t.reset_empty().unwrap();restore_prefix(t,ids,root,identity["descriptor_sha256"].as_str().unwrap(),identity["manifest_sha256"].as_str().unwrap());
    let meta=descriptor(t,ids);assert_eq!(state_io(t,&meta).state_digest().unwrap(),identity["state_sha256"].as_str().unwrap(),"all restored KV/GDN/history/logits bytes");
}
fn capture(root:&Path,points:&[usize],p:Progress,s:&OwnedCudaStream,input:&OwnedCudaBuffer,output:&OwnedCudaBuffer,w:&Workspace)->Result<()> {
    let points:Vec<_>=points.iter().copied().filter(|&n|n>=p.first&&n<p.first+p.rows).collect();if points.is_empty(){return Ok(())}
    let mut index=Vec::new();
    let mut plane=|name:&str,cols:usize,owner:&OwnedCudaBuffer,offset:usize|->Result<()> {
        let mut all=Vec::new();for &row in &points {let mut raw=vec![0;cols*4];owner.view(offset+(row-p.first)*cols*4,raw.len())?.copy_to_host_sync(s,&mut raw)?;assert!(raw.chunks_exact(4).all(|b|f32::from_le_bytes(b.try_into().unwrap()).is_finite()),"nonfinite diagnostic {name}");all.extend_from_slice(&raw);}
        let file=format!("layer{:02}-first{}-{name}.f32",p.layer,p.first);blob(root,&file,&all);index.push(json!({"file":file,"name":name,"cols":cols,"rows":points,"sha256":hash(&all)}));Ok(())
    };
    plane("hidden_in",H,input,0)?;plane("hidden_out",H,output,0)?;
    for (name,cols) in [("normalized",H),("attention",H),("attention_residual",H),("post_norm",H),("ffn_output",H)] {let x=&w.planes[name];plane(name,cols,&x.owner,x.offset)?;}
    if p.layer%4!=3 {
        for (name,cols) in [("qkv",10240),("z",6144),("alpha",48),("beta_raw",48),("activated",10240),("recurrent",6144),("linear_output",6144)] {let x=&w.planes[name];plane(name,cols,&x.owner,x.offset)?;}
    } else {
        for (name,cols) in [("q_full",12288),("k_raw",1024),("value",1024),("query",6144),("q_gate",6144),("attention",6144),("gated",6144)] {let x=&w.full_planes[name];plane(&format!("full_{name}"),cols,&x.owner,x.offset)?;}
    }
    for (name,x) in ["ffn_gate","ffn_up","ffn_activation"].into_iter().zip(&w.ffn_planes){plane(name,17408,&x.owner,x.offset)?;}
    save(root,&format!("layer{:02}-first{}.json",p.layer,p.first),&json!({"layer":p.layer,"first":p.first,"width":p.rows,"planes":index}));Ok(())
}
fn append(t:&mut CheckpointCudaTarget,ids:&[u32],root:Option<&Path>,points:&[usize])->f64 {
    let tick=Instant::now();let plan=t.plan_augur_sequence(ids,ids.chunks(4352).map(|s|s.len()).collect()).unwrap();
    if let Some(root)=root {
        arena::with_separate(||t.consume_augur_impl(ids,&plan,Budget::default(),&mut |_|false,
            #[cfg(feature="qwen35-augur-vision")] None,
            Some(&mut |p,s,i,o,w|capture(root,points,p,s,i,o,w)))).unwrap();
    } else {t.consume_sequence_augur(ids,&plan,Budget::default(),|_|false).unwrap();}
    t.stream.synchronize().unwrap();tick.elapsed().as_secs_f64()
}
fn state_blobs(t:&CheckpointCudaTarget,root:&Path) {
    let mut index=Vec::new();for (i,layer) in t.layers.iter().enumerate(){if let Layer::Linear(l)=layer{for(name,b)in l.checkpoint_buffers(){let mut raw=vec![0;b.len_bytes()];b.view(0,raw.len()).unwrap().copy_to_host_sync(&t.stream,&mut raw).unwrap();let file=format!("state-{i:02}-{name}.f32");blob(root,&file,&raw);index.push(json!({"layer":i,"name":name,"file":file,"sha256":hash(&raw)}));}}}save(root,"state-index.json",&json!(index));
}
fn teacher(t:&mut CheckpointCudaTarget,reference:&[u32],root:&Path)->Value {
    let mut rows=Vec::new();for (step,&token) in reference.iter().enumerate(){
        let logits=t.logits_snapshot().unwrap();floats(root,&format!("logits-{step:03}.f32"),&logits);
        let mut order:Vec<_>=(0..logits.len()).collect();order.select_nth_unstable_by(8,|&a,&b|logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));order[..8].sort_by(|&a,&b|logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
        rows.push(json!({"step":step,"forced_token":token,"argmax":order[0],"margin":logits[order[0]]-logits[order[1]],"top8":order[..8].iter().map(|&i|json!([i,logits[i]])).collect::<Vec<_>>()}));
        if step+1<reference.len(){t.token.copy_from_host_sync(&t.stream,&token.to_le_bytes()).unwrap();t.selected=Some(token);t.consume_selected_augur().unwrap();}
    }json!(rows)
}
#[test]
#[ignore="exclusive SM70: 128K tail partition error ledger, authenticated cache, diagnostic I/O excluded from TPS"]
fn partition_numerical_ledger() {
    let out=PathBuf::from(std::env::var_os("AUGUR_NATIVE_DECODE_OUT").unwrap());
    let raw=std::fs::read(std::env::var_os("AUGUR_NATIVE_DECODE_INPUT").unwrap()).unwrap();assert_eq!(hash(&raw),std::env::var("AUGUR_NATIVE_DECODE_INPUT_SHA256").unwrap());let input:Value=serde_json::from_slice(&raw).unwrap();
    let ids=super::super::super::transfer_tests::ids(&input["token_ids"]);let common=input["common_tokens"].as_u64().unwrap()as usize;let split=input["split_tokens"].as_u64().unwrap()as usize;
    assert!(common>0&&common%4352==0&&split>common&&split<ids.len()&&ids.len()-common<=4352);
    let mut points:Vec<_>=(common..ids.len()).step_by(32).chain([split-1,split,ids.len()-1]).collect();points.sort_unstable();points.dedup();
    let raw=std::fs::read(std::env::var_os("AUGUR_MANIFEST").unwrap()).unwrap();assert_eq!(hash(&raw),std::env::var("AUGUR_MANIFEST_SHA256").unwrap());let manifest:Value=serde_json::from_slice(&raw).unwrap();
    let initial=cuda_device_allocation_snapshot();let mut t=CheckpointCudaTarget::open_with_execution_policy(std::env::var_os("AUGUR_MODEL_ROOT").unwrap(),Qwen35CheckpointProfile::QuasarNvfp4,serde_json::from_value(manifest["checkpoint"].clone()).unwrap(),262144,CheckpointProjectionPolicy::AugurQpnAllWorkspaceShapeSmallPairF64,CheckpointKvPolicy::Q8_0).unwrap();
    t.enable_exact_warp_tree_norm().unwrap();t.enable_fused_input_qpn_norm().unwrap();t.enable_batched_ffn8().unwrap();t.enable_augur_decode().unwrap();assert!(t.dflash_capture.is_none());
    append(&mut t,&ids[..common],None,&[]);println!("PARTITION common ready {common}");
    let base=out.join("common-cache");let base_id=checkpoint(&mut t,&ids[..common],&base);
    let plain_s=append(&mut t,&ids[common..],None,&[]);let plain_logits=t.logits_snapshot().unwrap();floats(&out,"plain-logits.f32",&plain_logits);
    let tick=Instant::now();let mut reference=vec![t.greedy_next_token().unwrap()];while reference.len()<128 {t.consume_selected_augur().unwrap();reference.push(t.greedy_next_token().unwrap());}let decode_s=tick.elapsed().as_secs_f64();save(&out,"reference.json",&json!({"tokens":reference,"plain_tail_seconds":plain_s,"decode127_seconds":decode_s}));
    let mut split_cache=None;let mut records=Vec::new();
    for mode in ["whole","segmented","restored"] {
        let dest=out.join(mode);std::fs::create_dir(&dest).unwrap();let tail_s;
        if mode=="restored"{let identity=split_cache.as_ref().unwrap();rewind(&mut t,&ids[..split],&out.join("split-cache"),identity);tail_s=append(&mut t,&ids[split..],Some(&dest),&points);}
        else {rewind(&mut t,&ids[..common],&base,&base_id);if mode=="whole"{tail_s=append(&mut t,&ids[common..],Some(&dest),&points);}else{let s=append(&mut t,&ids[common..split],Some(&dest),&points);split_cache=Some(checkpoint(&mut t,&ids[..split],&out.join("split-cache")));tail_s=s+append(&mut t,&ids[split..],Some(&dest),&points);}}
        let logits=t.logits_snapshot().unwrap();let plain_exact=logits.iter().zip(&plain_logits).all(|(a,b)|a.to_bits()==b.to_bits());if mode=="whole"{assert!(plain_exact,"disjoint diagnostic arena must match production logits before attribution");}
        state_blobs(&t,&dest);let forced=teacher(&mut t,&reference,&dest);save(&dest,"teacher.json",&forced);records.push(json!({"mode":mode,"plain_logits_exact":plain_exact,"diagnostic_tail_seconds_not_TPS":tail_s}));println!("PARTITION mode complete {mode}");
    }
    drop(t);let after=cuda_device_allocation_snapshot();assert_eq!(after.live_bytes,initial.live_bytes);assert_eq!(after.live_allocations,initial.live_allocations);
    save(&out,"complete.json",&json!({"status":"PASS_PARTITION_DIAGNOSTIC_COLLECTION_NOT_SEMANTIC_QUALIFICATION","records":records,"points":points,"common_tokens":common,"split_tokens":split,"prompt_tokens":ids.len(),"same_forced_token_sequence":true,"cache_full_state_hash_verified":true,"diagnostics_have_no_TPS_score":true,"live_bytes_after":after.live_bytes,"live_allocations_after":after.live_allocations}));
}
