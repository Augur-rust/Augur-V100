//! Native DFlash2 safetensors -> Q8_0 GGUF with the target tokenizer.
//! Only the admitted five-layer Augur draft profile is supported. All tensor
//! roles/shapes are checked; convolution bases and norms stay F32. The selector gate is original F32 * 0.45,
//! folded exactly once; it is never reconstructed from a Q8 gate.
use super::*;
use augur_model_qwen35::{DFlashConfig, SafetensorsPinnedShard, SafetensorsRecord};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
mod metadata;
mod strict_json;
use metadata::Metadata;
const GATE_PROFILE: &str = "f32_gate045_v1";
const GATE_FACTOR: f32 = 0.45;

struct SourceFile { path: PathBuf, raw: Vec<u8>, sha256: String, _lock: File }
impl SourceFile {
    fn open(path: &Path) -> Result<Self> {
        let path=path.canonicalize()?;
        let mut options=OpenOptions::new();options.read(true);
        #[cfg(windows)] { use std::os::windows::fs::OpenOptionsExt;options.share_mode(1); }
        let mut file=options.open(&path)?;
        let size=file.metadata()?.len();
        if !file.metadata()?.is_file() || size==0 || size>MAX_HEADER { return Err("metadata input size/type outside limits".into()); }
        let mut raw=Vec::new();Read::by_ref(&mut file).take(MAX_HEADER+1).read_to_end(&mut raw)?;
        if raw.len() as u64!=size { return Err("metadata file changed or exceeded budget".into()); }
        let sha256=format!("{:x}",Sha256::digest(&raw));
        Ok(Self {path,raw,sha256,_lock:file})
    }
}
#[derive(Debug, Serialize)]
pub struct MappedTensor {
    pub source_name: String,
    pub source_dtype: String,
    #[serde(flatten)] pub tensor: TensorPlan,
}
#[derive(Debug, Serialize)]
pub struct DflashPlan {
    pub schema: &'static str,
    pub weights_sha256: String,
    pub weights_bytes: u64,
    pub selector_gate_sha256: String,
    pub metadata_sources: Value,
    pub output_bytes: u64,
    pub additional_free_bytes_required: u64,
    pub chunk_bytes: usize,
    pub data_offset: u64,
    pub tensors: Vec<MappedTensor>,
    #[serde(skip)] header: Vec<u8>,
}
struct Inputs {
    weights: SafetensorsPinnedShard,
    draft: SourceFile,
    target: SourceFile,
    tokenizer: SourceFile,
    tokenizer_config: SourceFile,
    template: SourceFile,
    config: DFlashConfig,
}
impl Inputs {
    fn open(draft: &Path, target: &Path, weights_sha: &str) -> Result<Self> {
        if draft.join("model.safetensors.index.json").exists() || draft.join("mask_embedding.pt").exists() {
            return Err("this profile supports one original safetensors file and a shared unpatched target embedding".into());
        }
        let draft_config=SourceFile::open(&draft.join("config.json"))?;
        let config=DFlashConfig::parse_f16_opt_in(&draft_config.raw)?;
        let raw=strict_json::parse(&draft_config.raw)?;
        for key in ["attention_value_scale","output_multiplier","final_logit_softcapping","input_embedding_scale"] {
            if raw.get(key).is_some() || raw["dflash_config"].get(key).is_some() {
                return Err(format!("unsupported draft semantic override: {key}").into());
            }
        }
        if raw["sliding_window"]!=2048 || raw["attention_bias"]!=false || raw["tie_word_embeddings"]!=false {
            return Err("unsupported draft attention/embedding profile".into());
        }
        let target_config=SourceFile::open(&target.join("config.json"))?;
        let t=strict_json::parse(&target_config.raw)?;
        if t["architectures"]!=serde_json::json!(["Qwen3_5ForConditionalGeneration"]) ||
            t["text_config"]["hidden_size"]!=5120 || t["text_config"]["num_hidden_layers"]!=64 ||
            t["text_config"]["vocab_size"]!=248320 || !t["text_config"]["rope_parameters"]["mrope_section"].is_array() {
            return Err("target must be the admitted Qwen3.5/QUASAR 27B M-RoPE profile".into());
        }
        let tokenizer=SourceFile::open(&target.join("tokenizer.json"))?;
        let tokenizer_config=SourceFile::open(&target.join("tokenizer_config.json"))?;
        let template=SourceFile::open(&target.join("chat_template.jinja"))?;
        let weights=SafetensorsPinnedShard::open(draft.join("model.safetensors"),weights_sha)?;
        Ok(Self {weights,draft:draft_config,target:target_config,tokenizer,tokenizer_config,template,config})
    }
    fn identities(&self) -> Value {
        let mut sources=serde_json::Map::new();
        for (label,file) in [("draft_config",&self.draft),("target_config",&self.target),
            ("target_tokenizer",&self.tokenizer),("target_tokenizer_config",&self.tokenizer_config),
            ("target_template",&self.template)] {
            sources.insert(label.into(),serde_json::json!({"path":file.path,"sha256":file.sha256,"bytes":file.raw.len()}));
        }
        Value::Object(sources)
    }
}

fn specs(c: &DFlashConfig) -> BTreeMap<String,(String,Vec<u64>,bool)> {
    let mut out=BTreeMap::new();
    let mut add=|source: String,name: String,shape: Vec<u64>,q8: bool| { assert!(out.insert(source,(name,shape,q8)).is_none()); };
    let h=c.hidden_size as u64;let f=c.intermediate_size as u64;let d=c.head_dim as u64;
    let r=c.selector_rank as u64;let v=c.vocab_size as u64;
    add("fc.weight".into(),"fc.weight".into(),vec![h,h*c.target_layer_ids.len() as u64],true);
    add("hidden_norm.weight".into(),"enc.output_norm.weight".into(),vec![h],false);
    add("norm.weight".into(),"output_norm.weight".into(),vec![h],false);
    for (source,name,shape) in [("hidden_projection.weight","selector_hidden.weight",vec![r,h]),
        ("predecessor_codebook","selector_predecessor.weight",vec![v,r]),
        ("successor_codebook","selector_successor.weight",vec![v,r])] {
        add(format!("candidate_selector.{source}"),name.into(),shape,source!="hidden_projection.weight");
    }
    for i in 0..c.num_hidden_layers {
        for (source,name,shape,q8) in [
            ("attention_conv.base_kernel","attn_conv_base",vec![2,c.conv_kernel_size as u64,h],false),
            ("mlp_conv.base_kernel","ffn_conv_base",vec![2,c.conv_kernel_size as u64,h],false),
            ("attention_conv.kernel_projection.weight","attn_conv_proj.weight",vec![h/c.conv_group_size as u64*2*c.conv_kernel_size as u64,h],true),
            ("mlp_conv.kernel_projection.weight","ffn_conv_proj.weight",vec![h/c.conv_group_size as u64*2*c.conv_kernel_size as u64,h],true),
            ("input_layernorm.weight","attn_norm.weight",vec![h],false),
            ("post_attention_layernorm.weight","ffn_norm.weight",vec![h],false),
            ("self_attn.q_norm.weight","attn_q_norm.weight",vec![d],false),
            ("self_attn.k_norm.weight","attn_k_norm.weight",vec![d],false),
            ("self_attn.q_proj.weight","attn_q.weight",vec![c.attention_heads as u64*d,h],true),
            ("self_attn.k_proj.weight","attn_k.weight",vec![c.kv_heads as u64*d,h],true),
            ("self_attn.v_proj.weight","attn_v.weight",vec![c.kv_heads as u64*d,h],true),
            ("self_attn.o_proj.weight","attn_output.weight",vec![h,c.attention_heads as u64*d],true),
            ("mlp.gate_proj.weight","ffn_gate.weight",vec![f,h],true),
            ("mlp.up_proj.weight","ffn_up.weight",vec![f,h],true),
            ("mlp.down_proj.weight","ffn_down.weight",vec![h,f],true),
        ] { add(format!("layers.{i}.{source}"),format!("blk.{i}.{name}"),shape,q8); }
    }
    out
}
fn mapped_tensors(records: &[SafetensorsRecord], config: &DFlashConfig) -> Result<Vec<MappedTensor>> {
    let expected=specs(config);let mut remaining: BTreeSet<_>=expected.keys().cloned().collect();
    let mut output=Vec::new();let mut offset=0_u64;
    for record in records {
        // llama accepts both the original unprefixed names and an explicit model prefix.
        let source=record.name.strip_prefix("model.").unwrap_or(&record.name);
        let (name,shape,q8)=expected.get(source).ok_or_else(||format!("unexpected draft tensor: {}",record.name))?;
        if !remaining.remove(source) || record.shape!=*shape { return Err(format!("duplicate or wrong draft shape: {}",record.name).into()); }
        let (kind,width) = match record.dtype.as_str() {"F32"=>(0,4),"F16"=>(1,2),"BF16"=>(30,2),_=>return Err("draft tensors must be floating-point".into())};
        let elements=record.element_count()?;
        if record.byte_len()?!=elements.checked_mul(width).ok_or("tensor byte overflow")? { return Err("draft tensor byte extent mismatch".into()); }
        let dimensions: Vec<_>=record.shape.iter().copied().rev().collect();
        if *q8 && (dimensions.len()!=2 || dimensions[0]%32!=0) { return Err("draft matrix Q8 geometry".into()); }
        let bytes=if *q8 {elements/32*34} else {elements*4};
        offset=align(offset,32)?;
        output.push(MappedTensor {source_name:record.name.clone(),source_dtype:record.dtype.clone(),
            tensor:TensorPlan {name:name.clone(),dimensions,input_type:kind,output_type:if *q8 {8}else{0},
                input_bytes:record.byte_len()?,output_bytes:bytes,output_offset:offset,
                action:if *q8 {"native_q8_matrix"}else if name=="selector_hidden.weight" {"original_float_to_f32_times_045_once"}else{"exact_float_to_f32_norm_or_convolution"}}});
        offset=offset.checked_add(bytes).ok_or("draft output overflow")?;
    }
    if !remaining.is_empty() || output.len()!=81 { return Err("draft requires exactly 81 mapped tensors".into()); }
    Ok(output)
}
fn make_plan(inputs: &mut Inputs, chunk: usize) -> Result<DflashPlan> {
    if !(128..=16*1024*1024).contains(&chunk) || chunk%128!=0 { return Err("chunk bytes must be a 128-byte multiple within 128..16MiB".into()); }
    let records: Vec<_>=inputs.weights.manifest().tensors().cloned().collect();
    let tensors=mapped_tensors(&records,&inputs.config)?;
    let gate=tensors.iter().find(|v|v.tensor.name=="selector_hidden.weight").ok_or("missing gate")?;
    let mut gate_source=vec![0;usize::try_from(gate.tensor.input_bytes)?];
    inputs.weights.read_range_into(&gate.source_name,0,&mut gate_source)?;
    let selector_gate_sha256=format!("{:x}",Sha256::digest(encode_gate(&gate_source,gate.tensor.input_type)?));
    let mut m=Metadata::new();
    m.text("general.architecture","dflash")?;m.text("general.type","model")?;
    m.text("general.name","Augur DFlash2 Native Rust Q8_0 + FP32 gate0.45")?;
    // Do not infer license, organization or model origin from folder names.
    m.integer("general.file_type",7)?;m.integer("general.quantization_version",2)?;
    m.text("quasar.selector_gate.profile",GATE_PROFILE)?;
    m.text("quasar.selector_gate.sha256",&selector_gate_sha256)?;
    m.float("quasar.selector_gate.folded_factor",GATE_FACTOR)?;
    m.float("quasar.selector_gate.runtime_scale",1.0)?;
    m.text("quasar.selector_gate.scope","global_gate045_v1")?;
    m.text("quasar.source_weights.sha256",inputs.weights.manifest().file_sha256.as_deref().ok_or("missing source SHA")?)?;
    let c=&inputs.config;
    for (key,value) in [("block_count",c.num_hidden_layers),("context_length",c.max_position_embeddings),
        ("embedding_length",c.hidden_size),("feed_forward_length",c.intermediate_size),
        ("attention.head_count",c.attention_heads),("attention.head_count_kv",c.kv_heads),
        ("attention.key_length",c.head_dim),("attention.value_length",c.head_dim),("block_size",c.block_size),
        ("conv_kernel_size",c.conv_kernel_size),("conv_group_size",c.conv_group_size),
        ("selector_rank",c.selector_rank),("selector_top_k",c.selector_top_k),("attention.sliding_window",2048)] {
        m.integer(&format!("dflash.{key}"),u32::try_from(value)?)?;
    }
    m.boolean("dflash.attention.causal",false)?;
    m.float("dflash.rope.freq_base",10_000_000.)?;m.float("dflash.attention.layer_norm_rms_epsilon",1e-6)?;
    m.integers("dflash.target_layers",&c.target_layer_ids.iter().map(|v|(*v+1) as i32).collect::<Vec<_>>())?;
    m.integers("dflash.rope.dimension_sections",&[c.head_dim as i32/2,0,0,0])?;
    m.booleans("dflash.attention.sliding_window_pattern",&vec![true;c.num_hidden_layers])?;
    let target=strict_json::parse(&inputs.target.raw)?;
    metadata::tokenizer(&mut m,&strict_json::parse(&inputs.tokenizer.raw)?,
        &strict_json::parse(&inputs.tokenizer_config.raw)?,&target["text_config"],
        &std::str::from_utf8(&inputs.template.raw)?.replace("\r\n","\n"),c.vocab_size)?;
    let mut header=m.header(tensors.len())?;
    for mapped in &tensors {
        let t=&mapped.tensor;string(&mut header,&t.name);
        header.extend_from_slice(&(t.dimensions.len() as u32).to_le_bytes());
        for d in &t.dimensions {header.extend_from_slice(&d.to_le_bytes());}
        header.extend_from_slice(&t.output_type.to_le_bytes());header.extend_from_slice(&t.output_offset.to_le_bytes());
    }
    let data_offset=align(header.len() as u64,32)?;
    if data_offset>MAX_HEADER { return Err("draft GGUF header exceeds budget".into()); }
    header.resize(data_offset as usize,0);
    let last=&tensors.last().ok_or("no tensors")?.tensor;
    let output_bytes=data_offset.checked_add(last.output_offset).and_then(|v|v.checked_add(last.output_bytes)).ok_or("total draft length overflow")?;
    Ok(DflashPlan {schema:"quasar-rust-dflash-q8-f32gate-plan-v2",weights_sha256:inputs.weights.manifest().file_sha256.clone().ok_or("missing full source hash")?,
        weights_bytes:inputs.weights.manifest().file_size,selector_gate_sha256,metadata_sources:inputs.identities(),output_bytes,
        additional_free_bytes_required:output_bytes+MAX_HEADER,chunk_bytes:chunk,data_offset,tensors,header})
}
pub fn plan(draft: &Path, target: &Path, weights_sha: &str, chunk: usize) -> Result<DflashPlan> {
    make_plan(&mut Inputs::open(draft,target,weights_sha)?,chunk)
}

#[derive(Default)]
struct ErrorBill {max_abs: f64, squared: f64, reference_squared: f64, elements: u64}
fn encode_gate(raw: &[u8], kind: u32) -> Result<Vec<u8>> {
    let width=match kind {0=>4,1|30=>2,_=>return Err("gate must originate from F32/F16/BF16".into())};
    if raw.len()%width!=0 {return Err("unaligned gate source".into());}
    let mut out=Vec::with_capacity(raw.len()/width*4);
    for bytes in raw.chunks_exact(width) {
        let calibrated=decode(bytes,kind)? * GATE_FACTOR;
        if !calibrated.is_finite() {return Err("nonfinite calibrated selector gate".into());}
        out.extend_from_slice(&calibrated.to_le_bytes());
    }
    Ok(out)
}
fn encode_chunk(raw: &[u8], kind: u32, q8: bool, bill: &mut ErrorBill) -> Result<Vec<u8>> {
    let width=if kind==0 {4}else{2};
    if raw.len()%width!=0 || (q8 && raw.len()%(32*width)!=0) { return Err("unaligned draft input chunk".into()); }
    let mut out=Vec::with_capacity(if q8 {raw.len()/width/32*34}else{raw.len()/width*4});
    if !q8 {
        for value in raw.chunks_exact(width) {out.extend_from_slice(&decode(value,kind)?.to_le_bytes());}
        return Ok(out);
    }
    for raw in raw.chunks_exact(32*width) {
        let mut values=[0_f32;32];
        for (v,b) in values.iter_mut().zip(raw.chunks_exact(width)) {*v=decode(b,kind)?;}
        let block=quantize_q8_0_block_ref(&values).map_err(|e|format!("draft Q8 block: {e:?}"))?;
        let scale=block.d.to_f32();
        for (v,&q) in values.iter().zip(&block.qs) {
            let error=f64::from(scale*f32::from(q))-f64::from(*v);
            bill.max_abs=bill.max_abs.max(error.abs());bill.squared+=error*error;
            bill.reference_squared+=f64::from(*v).powi(2);bill.elements+=1;
        }
        out.extend_from_slice(&block.to_le_bytes());
    }
    Ok(out)
}
struct Pending {path: PathBuf,published: bool}
impl Drop for Pending {
    fn drop(&mut self) {
        if !self.published {
            for name in ["model-q8_0.gguf","config.json","receipt.json"] {let _=fs::remove_file(self.path.join(name));}
            let _=fs::remove_dir(&self.path);
        }
    }
}
pub fn convert(draft: &Path, target: &Path, destination: &Path, weights_sha: &str, chunk: usize) -> Result<Value> {
    let started=std::time::Instant::now();
    if destination.try_exists()? {return Err("destination exists; original draft will not be replaced".into());}
    let parent=destination.parent().filter(|p|!p.as_os_str().is_empty()).unwrap_or(Path::new(".")).canonicalize()?;
    let destination=parent.join(destination.file_name().ok_or("missing destination name")?);
    let mut inputs=Inputs::open(draft,target,weights_sha)?;
    let p=make_plan(&mut inputs,chunk)?;
    let available=free_bytes(&parent)?;
    if available<p.additional_free_bytes_required {return Err("insufficient free space for native draft output".into());}
    let path=parent.join(format!(".augur-dflash-partial-{}-{}",std::process::id(),TEMP_ID.fetch_add(1,Ordering::Relaxed)));
    fs::create_dir(&path)?;let mut pending=Pending {path,published:false};
    let output=pending.path.join("model-q8_0.gguf");
    let mut file=OpenOptions::new().write(true).create_new(true).open(&output)?;
    file.write_all(&p.header)?;let mut written=p.data_offset;let mut bills=Vec::new();
    let mut buffer=vec![0_u8;chunk];
    for mapped in &p.tensors {
        let t=&mapped.tensor;let location=p.data_offset+t.output_offset;
        let padding=location.checked_sub(written).ok_or("draft tensor offset regression")?;
        if padding>31 {return Err("unexpected draft tensor alignment gap".into());}
        file.write_all(&vec![0; padding as usize])?;written=location;
        let mut offset=0;let mut bill=ErrorBill::default();
        while offset<t.input_bytes {
            let count=(t.input_bytes-offset).min(chunk as u64) as usize;
            inputs.weights.read_range_into(&mapped.source_name,offset,&mut buffer[..count])?;
            let bytes=if t.name=="selector_hidden.weight" {encode_gate(&buffer[..count],t.input_type)?}
                else {encode_chunk(&buffer[..count],t.input_type,t.output_type==8,&mut bill)?};
            file.write_all(&bytes)?;written+=bytes.len() as u64;offset+=count as u64;
        }
        if written!=location+t.output_bytes {return Err("draft tensor write extent mismatch".into());}
        bills.push(serde_json::json!({"source":mapped.source_name,"tensor":t.name,"ggml_type":t.output_type,
            "quantized_elements":bill.elements,"max_abs_error":bill.max_abs,
            "rmse":if bill.elements>0 {(bill.squared/bill.elements as f64).sqrt()}else{0.},
            "relative_l2":if bill.reference_squared>0.0 {(bill.squared/bill.reference_squared).sqrt()}else{0.},
            "float_conversion_exact":t.output_type==0 && t.name!="selector_hidden.weight",
            "folded_factor":if t.name=="selector_hidden.weight" {Some(GATE_FACTOR)} else {None},
            "error_reference":"Q8 versus original; gate intentional calibration separately labeled"}));
    }
    if written!=p.output_bytes {return Err("draft output file extent mismatch".into());}
    file.sync_all()?;drop(file);
    let check=GgufManifest::inspect(&output,GgufAuditOptions::default())?;
    if check.file_size!=p.output_bytes || check.tensors.len()!=81 {return Err("new draft GGUF manifest mismatch".into());}
    for (got,want) in check.tensors.iter().zip(&p.tensors) {
        let t=&want.tensor;
        if got.name!=t.name || got.dimensions!=t.dimensions || got.ggml_type!=t.output_type ||
            got.byte_len!=t.output_bytes || got.relative_offset!=t.output_offset {return Err("new draft tensor descriptor mismatch".into());}
    }
    let mut config=OpenOptions::new().write(true).create_new(true).open(pending.path.join("config.json"))?;
    config.write_all(&inputs.draft.raw)?;config.sync_all()?;drop(config);
    let receipt=serde_json::json!({"schema":"quasar-rust-dflash-q8-f32gate-receipt-v2","status":"NATIVE_FORMAT_CONVERSION_COMPLETE",
        "source_weights":inputs.weights.manifest().path(),"source_weights_sha256":p.weights_sha256,
        "output":destination.join("model-q8_0.gguf"),"output_sha256":check.file_sha256,
        "config":destination.join("config.json"),"config_sha256":inputs.draft.sha256,
        "plan":p,"tensor_error_bill":bills,"converter_executable_sha256":file_sha256(&std::env::current_exe()?)?,
        "conversion_wall_s":started.elapsed().as_secs_f64(),"available_bytes_at_start":available,
        "native_rust_quantization":true,"external_converter_or_quantizer_executed":false,
        "norms_and_convolution_bases":"F32 exact BF16/F16/F32 expansion","matrices":"48 Q8_0; selector_hidden F32 with folded 0.45",
        "selector_gate": {"profile":GATE_PROFILE,"sha256":p.selector_gate_sha256,"folded_factor":GATE_FACTOR,"runtime_scale":1.0,"scope":"global_gate045_v1"},
        "shared_target_embeddings_and_head":true,"HF_rope_permutation":"none; checked NeoX profile",
        "vocab_source":"target HF tokenizer, strict admitted qwen35 pretokenizer",
        "independent_llama_oracle_passed":false,"GPU_loaded":false,"semantic_or_acceptance_qualified":false});
    let mut report=OpenOptions::new().write(true).create_new(true).open(pending.path.join("receipt.json"))?;
    report.write_all(&serde_json::to_vec_pretty(&receipt)?)?;report.sync_all()?;drop(report);
    if destination.try_exists()? {return Err("destination appeared while quantizing".into());}
    fs::rename(&pending.path,&destination)?;pending.published=true;Ok(receipt)
}
#[cfg(test)]
mod tests;
