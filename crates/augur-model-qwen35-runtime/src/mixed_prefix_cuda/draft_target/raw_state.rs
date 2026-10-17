//! Opt-in bounded same-owner capture. Never enables product or semantic claims.
use super::*;
use std::cell::{Cell, RefCell};
use std::collections::BTreeSet;
use std::io::Write;
#[path = "raw_state/io.rs"]
mod capture_io;

fn invalid(detail: impl std::fmt::Display) -> Qwen35MixedPrefixCudaError {
    super::invalid(detail.to_string())
}

pub(super) struct Options {
    directory: PathBuf,
    reference_offsets: Vec<usize>,
    reference: bool,
    exact_tiles_and_head: bool,
}

fn parse_offsets(value: &str) -> Result<Vec<usize>> {
    if value.is_empty() { return Err(invalid("reference capture offsets required")); }
    let values=value.split(',').map(|s|s.parse::<usize>().map_err(invalid)).collect::<Result<Vec<_>>>()?;
    if values.is_empty() || values.len()>3 || values.iter().any(|&v|v==0 || v>31) ||
       values.windows(2).any(|w|w[0]>=w[1]) { return Err(invalid("bounded unique increasing reference offsets")); }
    Ok(values)
}

#[cfg(feature="qwen35-optimized-state-bridge")]
fn bridge_phase(block:usize,cursor:usize,prior_count:Option<usize>) -> Result<Option<&'static str>> {
    if block>=1024 || !(8192..=9215).contains(&cursor) {return Err(invalid("bridge bounds"));}
    if block==0 {
        if cursor!=8192 || prior_count.is_some() {return Err(invalid("bridge initial identity"));}
        return Ok(Some("initial"));
    }
    let n=prior_count.ok_or_else(||invalid("bridge commit missing"))?;
    if !(1..=8).contains(&n) || cursor-8192<block {return Err(invalid("bridge commit extent"));}
    if cursor-8192<512 {return Ok(None);}
    Ok(Some(if n==8 {"late_full_after512"} else {"late_partial_after512"}))
}

fn committed_inputs(previous_anchor:u32,emitted:&[u32],device_path:&[u32]) -> Result<Vec<u32>> {
    if emitted.is_empty() || emitted.len()>B || previous_anchor as usize>=V || emitted.iter().any(|&t|t as usize>=V) {
        return Err(invalid("raw committed token range"));
    }
    let mut expected=vec![previous_anchor];expected.extend_from_slice(&emitted[..emitted.len()-1]);
    if expected!=device_path {return Err(invalid("raw committed input path mismatch"));}
    Ok(expected)
}

impl Options {
    pub(super) fn from_env(config:&Qwen35AutoregressiveCudaConfig, request:Option<&Qwen35DraftTargetRequestOptions>,
        history:usize, rounds:usize, vision:Option<&Path>, shared:bool, sink:bool) -> Result<Option<Self>> {
        let Some(path)=std::env::var_os("AUGUR_QWEN35_RAW_STATE_DIR") else { return Ok(None); };
        let r=request.ok_or_else(||invalid("raw capture requires request"))?;
        if history!=2048 || config.capacity!=196608 || (!cfg!(feature="qwen35-optimized-state-bridge") && !(2..=32).contains(&r.max_new_tokens)) ||
           (!cfg!(feature="qwen35-optimized-state-bridge") && rounds!=32) || vision.is_none() || shared || sink || !r.audit_committed_state ||
           !r.batched_prompt_capture || !r.prefill_dgemm_scores || !r.share_target_rope ||
           !r.target_split_rope || !r.request_position_budget || !r.retain_full_accept_state ||
           !r.batched_target_verify || (!cfg!(feature="qwen35-optimized-state-bridge") && r.long_generation_budget) || r.audit_batched_target_verify ||
           r.profile_batched_target_verify || r.profile_prefill_sparse || r.reference_boundary_directory.is_some() ||
           r.reference_serial_prefix || r.diagnostic_q8_away || r.diagnostic_split_rope ||
           (!cfg!(feature="qwen35-optimized-state-bridge") && (r.exact_projection_tiles || r.exact_output_head_batch)) {
            return Err(invalid("raw capture frozen short diagnostic axes"));
        }
        #[cfg(feature="qwen35-optimized-state-bridge")]
        if rounds!=1024 || r.max_new_tokens!=1024 || !r.long_generation_budget || r.target_only_reference ||
            r.exact_projection_tiles!=r.exact_output_head_batch ||
            std::env::var_os("AUGUR_QWEN35_DRAFT_PRIVATE_DIR").is_none() {
            return Err(invalid("policy bridge requires paired private capture and frozen long axes"));
        }
        let directory=PathBuf::from(path);
        if !directory.is_absolute() || directory.exists() || directory.components().any(|c|matches!(c,std::path::Component::ParentDir)) {
            return Err(invalid("raw capture requires new absolute output directory"));
        }
        let parent=directory.parent().ok_or_else(||invalid("raw capture parent"))?;
        for p in parent.ancestors() {
            let m=std::fs::symlink_metadata(p).map_err(Qwen35Error::from)?;
            #[cfg(windows)] { use std::os::windows::fs::MetadataExt;
                if m.file_attributes()&0x400!=0 { return Err(invalid("raw capture reparse parent")); }
            }
            if m.file_type().is_symlink() { return Err(invalid("raw capture symlink parent")); }
        }
        let reference_offsets=if r.target_only_reference {
            parse_offsets(&std::env::var("AUGUR_QWEN35_RAW_STATE_OFFSETS").map_err(invalid)?)?
        } else {
            if std::env::var_os("AUGUR_QWEN35_RAW_STATE_OFFSETS").is_some() { return Err(invalid("candidate offsets may not be externally forced")); }
            vec![]
        };
        Ok(Some(Self{directory,reference_offsets,reference:r.target_only_reference,exact_tiles_and_head:r.exact_projection_tiles && r.exact_output_head_batch}))
    }
}

pub(super) struct Capture {
    options: Options,
    hidden: OwnedCudaBuffer,
    positions: [Cell<usize>;64],
    ledger: RefCell<Vec<u32>>,
    initial_cursor:usize,
    initial_anchor:Cell<u32>,
    boundary:capture_io::Boundary,
    seen:RefCell<BTreeSet<String>>,
    records:RefCell<Vec<Value>>,
    auxiliary:RefCell<Value>,
}

fn write_json(path:&Path, value:&Value) -> Result<()> {
    let bytes=serde_json::to_vec_pretty(value).map_err(invalid)?;
    let mut file=std::fs::OpenOptions::new().create_new(true).write(true).open(path).map_err(Qwen35Error::from)?;
    file.write_all(&bytes).map_err(Qwen35Error::from)?;
    file.sync_all().map_err(Qwen35Error::from)?;
    Ok(())
}

impl Capture {
    pub(super) fn new(options:Options, config:&Qwen35AutoregressiveCudaConfig, model:&str,
        config_bytes:&[u8],tokenizer_bytes:&[u8],prompt:&[usize]) -> Result<Self> {
        if !(2048..=8192).contains(&prompt.len()) { return Err(invalid("raw capture short occupied prompt")); }
        let prompt=prompt.iter().map(|&t|t as u32).collect::<Vec<_>>();
        let prompt_raw=prompt.iter().flat_map(|t|t.to_le_bytes()).collect::<Vec<_>>();
        let mut numerical=json!({"policy":"existing_split_rope_native_canonical_quantization_v1",
            "config":sha256_hex(config_bytes),"tokenizer":sha256_hex(tokenizer_bytes),
            "conditioning":"existing_dgemm_prefix_batched_2048_capture_v1",
            "source_commit":config.source_commit,"arithmetic_changed":false,
            "target_projection_tile_policy":if options.exact_tiles_and_head {"canonical_backbone_prompt_full8_full4_v1"} else {"canonical_batch8_v1"},
            "output_head_projection_policy":if options.exact_tiles_and_head {"packed_q8_exact_full8_draft_and_target_v1"} else {"canonical_serial_v1"}});
        if !cfg!(feature="qwen35-optimized-state-bridge") {
            numerical.as_object_mut().unwrap().remove("target_projection_tile_policy");
            numerical.as_object_mut().unwrap().remove("output_head_projection_policy");
        }
        if cfg!(feature="qwen35-optimized-state-bridge") && prompt.len()!=8192 { return Err(invalid("bridge exact8192 prompt")); }
        std::fs::create_dir(&options.directory).map_err(Qwen35Error::from)?;
        write_json(&options.directory.join("preflight.json"),&json!({"schema":"qwen35-raw-state-runtime-preflight-v1",
            "source_commit":config.source_commit,"source_dirty":true,"max_boundaries":4,
            "max_capture_bytes_per_boundary":8_000_000_000u64,"requires_external_disk_preflight":true,
            "reference":options.reference,"reference_offsets":options.reference_offsets,"numerical_identity_fields":numerical,
            "model_qualification":false,"product_tps":false,
            "committed_ranges_only":cfg!(feature="qwen35-optimized-state-bridge"),
            "underlying_kv_capacity_unchanged":196608}))?;
        let initial_cursor=prompt.len();
        let boundary=capture_io::Boundary {model:model.into(),config:sha256_hex(config_bytes),
            numerical_identity:sha256_hex(&serde_json::to_vec(&numerical).map_err(invalid)?),
            prompt_ids:sha256_hex(&prompt_raw),committed_input_ids:sha256_hex(&prompt_raw),
            cursor:initial_cursor,anchor:0,context_start:initial_cursor-2048,context_rows:2048};
        Ok(Self{hidden:OwnedCudaBuffer::allocate(config.device_ordinal,64*H*4)?,
            positions:std::array::from_fn(|_|Cell::new(usize::MAX)),options,ledger:RefCell::new(prompt),
            initial_cursor,initial_anchor:Cell::new(0),boundary,seen:RefCell::new(BTreeSet::new()),
            records:RefCell::new(vec![]),auxiliary:RefCell::new(Value::Null)})
    }

    pub(super) fn stage_hidden(&self,d:&Driver,layer:usize,position:usize,source:CudaBufferView) -> Result<()> {
        if layer>=64 || source.len_bytes()!=H*4 { return Err(invalid("raw hidden extent")); }
        copy(&d.stream,&self.hidden,layer*H*4,source)?;
        self.positions[layer].set(position);
        Ok(())
    }

    pub(super) fn initial(&self,d:&Driver,draft:&DFlashQ8CudaBlock,dflash:&Value,vision:&Value) -> Result<()> {
        if dflash["resident_device_bytes"].as_u64().unwrap_or(0)==0 || vision["vision"]["resident_device_bytes"].as_u64().unwrap_or(0)==0 {
            return Err(invalid("raw capture requires real Q8 draft and Vision residency"));
        }
        *self.auxiliary.borrow_mut()=json!({"dflash":dflash,"vision":vision,"draft_workspace_bytes":draft.workspace_bytes(),
            "vision_forward_executed":false,"reference_draft_idle":self.options.reference});
        self.initial_anchor.set(d.anchor);
        if cfg!(feature="qwen35-optimized-state-bridge") { Ok(()) } else { self.save(d,draft,"initial",None) }
    }

    pub(super) fn candidate_commit(&self,d:&Driver,draft:&DFlashQ8CudaBlock,report:&Value,terminal:bool) -> Result<()> {
        let old=report["cursor_before"].as_u64().ok_or_else(||invalid("raw old cursor"))? as usize;
        let committed=report["committed_token_count"].as_u64().ok_or_else(||invalid("raw committed count"))? as usize;
        let emitted=report["emitted_token_ids"].as_array().ok_or_else(||invalid("raw emitted ledger"))?;
        if committed==0 || committed>B || emitted.len()!=committed || old!=self.ledger.borrow().len() || old+committed!=d.cursor {
            return Err(invalid("raw candidate input/output ledger extent"));
        }
        // Actual target inputs are [previous anchor, emitted outputs except last].
        // Verify them against the device path actually submitted, not an assumed output ledger.
        let emitted=emitted.iter().map(|t|t.as_u64().filter(|&x|x<V as u64).map(|x|x as u32).ok_or_else(||invalid("raw token"))).collect::<Result<Vec<_>>>()?;
        let expected=committed_inputs(self.initial_anchor.get(),&emitted,&words(&d.stream,&d.path,0,committed)?)?;
        self.ledger.borrow_mut().extend(expected);self.initial_anchor.set(d.anchor);
        if cfg!(feature="qwen35-optimized-state-bridge") { return Ok(()); }
        let strategy=report["commit_strategy"].as_str().ok_or_else(||invalid("raw strategy"))?;
        let label=match strategy {"retain_verified_full_block"=>"first_retain", "restore_replay_committed_prefix"=>"first_replay", _=>return Err(invalid("raw strategy unknown"))};
        let first=self.seen.borrow_mut().insert(label.into());
        if first || terminal { self.save(d,draft,if first {label}else{"terminal"},Some(report))?; }
        Ok(())
    }

    #[cfg(feature="qwen35-optimized-state-bridge")]
    pub(super) fn bridge_before_verify(&self,d:&Driver,draft:&DFlashQ8CudaBlock,block:usize,previous:Option<&Value>) -> Result<()> {
        let prior_count=if block==0 { None } else {
            let tx=previous.ok_or_else(||invalid("bridge previous transaction missing"))?;
            if tx["pass"]!=true || tx["cursor_after"].as_u64()!=Some(d.cursor as u64) ||
                tx["generation_after"].as_u64()!=Some(d.generation) || tx["next_anchor"].as_u64()!=Some(d.anchor as u64) ||
                !tx["publication_stop_reason"].is_null() { return Err(invalid("bridge prior transaction identity")); }
            Some(tx["committed_token_count"].as_u64().ok_or_else(||invalid("bridge prior commit count"))? as usize)
        };
        if let Some(label)=bridge_phase(block,d.cursor,prior_count)? {
            if self.seen.borrow_mut().insert(label.into()) { self.save(d,draft,label,previous)?; }
        }
        Ok(())
    }

    pub(super) fn reference_commit(&self,d:&Driver,draft:&DFlashQ8CudaBlock,input:u32,terminal:bool) -> Result<()> {
        if self.ledger.borrow().len()+1!=d.cursor || input!=self.initial_anchor.get() { return Err(invalid("raw independent reference ledger")); }
        self.ledger.borrow_mut().push(input);self.initial_anchor.set(d.anchor);
        let offset=d.cursor-self.initial_cursor;
        if self.options.reference_offsets.contains(&offset) { self.save(d,draft,"reference",None)?; }
        if terminal && self.options.reference_offsets.iter().any(|&x|x>offset) {
            return Err(invalid("independent reference ended before selected candidate boundary"));
        }
        Ok(())
    }

    fn save(&self,d:&Driver,draft:&DFlashQ8CudaBlock,label:&str,transaction:Option<&Value>) -> Result<()> {
        if self.records.borrow().len()>=4 || self.ledger.borrow().len()!=d.cursor ||
            self.positions.iter().any(|p|p.get()!=d.cursor-1) { return Err(invalid("raw boundary count/hidden positions/ledger")); }
        d.stream.synchronize()?;
        let directory=self.options.directory.join(format!("cursor{:06}",d.cursor));
        let mut boundary=self.boundary.clone();boundary.cursor=d.cursor;boundary.anchor=d.anchor;
        boundary.context_start=d.cursor-2048;
        let ledger=self.ledger.borrow().iter().flat_map(|t|t.to_le_bytes()).collect::<Vec<_>>();
        boundary.committed_input_ids=sha256_hex(&ledger);
        let full_inventory=capture_io::target_inventory(d.cursor).map_err(invalid)?;
        let planes=if cfg!(feature="qwen35-optimized-state-bridge") {
            capture_io::committed_inventory(d.cursor).map_err(invalid)?
        } else {full_inventory.clone()};
        let mut buffers=Vec::with_capacity(195);
        for p in &d.plans { for b in p.terminal_state_buffers() { buffers.push((b,0)); } }
        buffers.push((&d.features,0));buffers.push((&d.token,0));
        for layer in 0..64 {buffers.push((&self.hidden,layer*H*4));}
        buffers.push((&d.head.logits,0));
        let submission=d.stream.submission_receipt()?;
        let snapshot=capture_io::Snapshot{boundary,owner:format!("{submission:?}"),generation:d.generation,
            backend:if cfg!(feature="qwen35-optimized-state-bridge") {"actual_target_committed_range_submission_v1"} else {"actual_target_owner_submission_v1"}.into(),planes};
        if buffers.len()!=full_inventory.len() || buffers.iter().zip(&full_inventory).any(|((b,offset),p)|
            offset.checked_add(p.physical_bytes).is_none_or(|end|end>b.len_bytes())) {
            return Err(invalid("bridge underlying full target allocation extent"));
        }
        let mut source=Source{driver:d,buffers,snapshot,submission};
        let receipt=capture_io::capture(&mut source,&directory,&capture_io::Budget{chunk_bytes:capture_io::MAX_CHUNK,
            max_total_bytes:8_000_000_000,max_chunks:2048}).map_err(invalid)?;
        capture_io::verify_capture(&directory,&capture_io::Budget{chunk_bytes:capture_io::MAX_CHUNK,
            max_total_bytes:8_000_000_000,max_chunks:2048}).map_err(invalid)?;
        let draft_inventory=draft.raw_state_inventory().map_err(invalid)?;
        let record=json!({"label":label,"cursor":d.cursor,"offset":d.cursor-self.initial_cursor,
            "anchor":d.anchor,"generation":d.generation,"directory":directory,
            "receipt_sha256":sha256_hex(&std::fs::read(directory.join("complete.json")).map_err(Qwen35Error::from)?),
            "raw_target_bytes":receipt.total_bytes,"input_token_ids":*self.ledger.borrow(),
            "hidden_positions":self.positions.iter().map(Cell::get).collect::<Vec<_>>(),
            "transaction":transaction,"draft_inventory":draft_inventory,"draft_raw_equivalence_qualified":false,
            "committed_ranges_only":cfg!(feature="qwen35-optimized-state-bridge"),
            "full_allocation_inventory":full_inventory,
            "future_capacity_exported":!cfg!(feature="qwen35-optimized-state-bridge"),
            "block_before_verify":if cfg!(feature="qwen35-optimized-state-bridge") {Some(d.generation-1)} else {None}});
        write_json(&self.options.directory.join(format!("cursor{:06}.json",d.cursor)),&record)?;
        self.records.borrow_mut().push(record);Ok(())
    }

    pub(super) fn finish(&self,d:&Driver) -> Result<Value> {
        let result=json!({"schema":if cfg!(feature="qwen35-optimized-state-bridge") {"qwen35-policy-bridge-committed-runtime-v1"} else {"qwen35-raw-committed-runtime-capture-v1"},"records":*self.records.borrow(),
            "initial_cursor":self.initial_cursor,"final_cursor":d.cursor,"independent_greedy_reference":self.options.reference,
            "auxiliary":*self.auxiliary.borrow(),"diagnostic_hidden_staging_bytes":64*H*4,
            "host_payload_chunk_bytes":capture_io::MAX_CHUNK,"full_target_physical_capture":!cfg!(feature="qwen35-optimized-state-bridge"),
            "all_committed_target_planes":195,"future_capacity_exported":!cfg!(feature="qwen35-optimized-state-bridge"),
            "complete_draft_private_raw_capture":false,"state_equivalence_qualified":false,"product_tps_permitted":false});
        write_json(&self.options.directory.join("complete.json"),&result)?;Ok(result)
    }
}

struct Source<'a> { driver:&'a Driver,buffers:Vec<(&'a OwnedCudaBuffer,usize)>,snapshot:capture_io::Snapshot,
    submission:CudaStreamSubmissionReceipt }
impl capture_io::ReadbackSource for Source<'_> {
    fn quiesce(&mut self)->std::io::Result<()> {
        let proof=self.driver.stream.synchronize_submission(self.submission).map_err(std::io::Error::other)?;
        if !proof.covers_submission(self.submission) || !proof.belongs_to_stream(&self.driver.stream).map_err(std::io::Error::other)? {
            return Err(std::io::Error::other("raw owner proof"));
        } Ok(())
    }
    fn snapshot(&self)->std::io::Result<capture_io::Snapshot> {
        if self.driver.cursor!=self.snapshot.boundary.cursor || self.driver.generation!=self.snapshot.generation ||
            self.driver.anchor!=self.snapshot.boundary.anchor || self.buffers.len()!=self.snapshot.planes.len() {
            return Err(std::io::Error::other("raw live owner drift"));
        }
        for ((b,offset),p) in self.buffers.iter().zip(&self.snapshot.planes) {
            if offset.checked_add(p.physical_bytes).is_none_or(|end|end>b.len_bytes()) {return Err(std::io::Error::other("raw buffer extent"));}
        } Ok(self.snapshot.clone())
    }
    fn read_completed(&mut self,plane:usize,offset:usize,out:&mut[u8])->std::io::Result<()> {
        let (buffer,base)=self.buffers.get(plane).ok_or_else(||std::io::Error::other("raw plane"))?;
        let offset=base.checked_add(offset).ok_or_else(||std::io::Error::other("raw offset"))?;
        let proof=buffer.copy_range_to_host_sync_with_submission_proof(&self.driver.stream,offset,out,self.submission).map_err(std::io::Error::other)?;
        if !proof.covers_submission(self.submission) || !proof.belongs_to_stream(&self.driver.stream).map_err(std::io::Error::other)? {return Err(std::io::Error::other("raw read owner proof"));}
        Ok(())
    }
    fn drain(&mut self)->std::io::Result<()> {self.driver.stream.synchronize().map_err(std::io::Error::other)}
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn offsets_fail_closed() { for v in ["","0","32","2,2","3,2","1,2,3,4","true"] {assert!(parse_offsets(v).is_err());} }
    #[test] fn selected_natural_offsets(){assert_eq!(parse_offsets("1,8,31").unwrap(),vec![1,8,31]);}
    #[test] fn full_raw_inventory(){let x=capture_io::target_inventory(5639).unwrap();assert_eq!(x.len(),195);assert_eq!(x[..128].iter().map(|p|p.physical_bytes).sum::<usize>(),7001997312);}
    #[test] fn reject_input_is_previous_anchor_not_replacement(){assert_eq!(committed_inputs(42,&[9],&[42]).unwrap(),vec![42]);assert!(committed_inputs(42,&[9],&[9]).is_err());}
    #[test] fn full_accept_excludes_bonus_from_inputs(){assert_eq!(committed_inputs(42,&[1,2,3,4,5,6,7,8],&[42,1,2,3,4,5,6,7]).unwrap(),vec![42,1,2,3,4,5,6,7]);}
    #[test] fn clipped_prefix_is_not_full_path(){assert_eq!(committed_inputs(42,&[1,2],&[42,1]).unwrap(),vec![42,1]);assert!(committed_inputs(42,&[1,2],&[42,1,2]).is_err());}
    #[test] fn invalid_tokens_rejected(){assert!(committed_inputs(42,&[],&[]).is_err());assert!(committed_inputs(42,&[248320],&[42]).is_err());}
}

#[cfg(all(test,feature="qwen35-optimized-state-bridge"))]
mod bridge_tests {
    use super::*;
    #[test] fn selection_is_natural_and_late() {
        assert_eq!(bridge_phase(0,8192,None).unwrap(),Some("initial"));
        assert_eq!(bridge_phase(60,8703,Some(8)).unwrap(),None);
        assert_eq!(bridge_phase(70,8704,Some(7)).unwrap(),Some("late_partial_after512"));
        assert_eq!(bridge_phase(71,8712,Some(8)).unwrap(),Some("late_full_after512"));
    }
    #[test] fn invalid_boundaries_fail_closed() {
        for (b,c,n) in [(0,8193,None),(0,8192,Some(8)),(1,8192,Some(1)),(1024,9000,Some(8)),(1,9216,Some(8)),(1,8200,None),(1,8200,Some(0)),(1,8201,Some(9))] {
            assert!(bridge_phase(b,c,n).is_err());
        }
    }
    #[test] fn committed_inventory_preserves_complete_plane_names() {
        let a=capture_io::target_inventory(8722).unwrap();
        let b=capture_io::committed_inventory(8722).unwrap();
        assert_eq!(a.len(),195); assert_eq!(b.len(),195);
        for (full,live) in a.iter().zip(&b) {
            assert_eq!(full.name,live.name);assert_eq!(full.scope,live.scope);
            assert_eq!(full.committed_bytes,live.committed_bytes);assert_eq!(live.physical_bytes,live.committed_bytes);
        }
        assert_eq!(a.iter().map(|p|p.physical_bytes-p.committed_bytes).sum::<usize>(),(196608-8722)*1088*32);
        assert!(b.iter().map(|p|p.physical_bytes).sum::<usize>()<800_000_000);
    }
}
