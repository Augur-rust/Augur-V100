//! GGUF metadata for the admitted Qwen3.5/QUASAR tokenizer and DFlash2 profile.
use super::*;
use std::collections::{BTreeMap, BTreeSet};

pub(super) struct Metadata { entries: BTreeMap<String, Vec<u8>> }
impl Metadata {
    pub(super) fn new() -> Self { Self { entries: BTreeMap::new() } }
    fn put(&mut self, key: &str, kind: u32, value: &[u8]) -> Result<()> {
        let mut wire = kind.to_le_bytes().to_vec(); wire.extend_from_slice(value);
        if self.entries.insert(key.into(), wire).is_some() { return Err(format!("duplicate GGUF metadata: {key}").into()); }
        Ok(())
    }
    pub(super) fn text(&mut self, key: &str, value: &str) -> Result<()> {
        let mut bytes = Vec::new(); string(&mut bytes,value); self.put(key,8,&bytes)
    }
    pub(super) fn integer(&mut self, key: &str, value: u32) -> Result<()> { self.put(key,4,&value.to_le_bytes()) }
    pub(super) fn float(&mut self, key: &str, value: f32) -> Result<()> { self.put(key,6,&value.to_le_bytes()) }
    pub(super) fn boolean(&mut self, key: &str, value: bool) -> Result<()> { self.put(key,7,&[u8::from(value)]) }
    pub(super) fn strings(&mut self, key: &str, values: &[String]) -> Result<()> {
        let mut bytes = 8_u32.to_le_bytes().to_vec(); bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for v in values { string(&mut bytes,v); } self.put(key,9,&bytes)
    }
    pub(super) fn integers(&mut self, key: &str, values: &[i32]) -> Result<()> {
        let mut bytes = 5_u32.to_le_bytes().to_vec(); bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for v in values { bytes.extend_from_slice(&v.to_le_bytes()); } self.put(key,9,&bytes)
    }
    pub(super) fn booleans(&mut self, key: &str, values: &[bool]) -> Result<()> {
        let mut bytes = 7_u32.to_le_bytes().to_vec(); bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
        bytes.extend(values.iter().map(|v|u8::from(*v))); self.put(key,9,&bytes)
    }
    pub(super) fn header(&self, tensors: usize) -> Result<Vec<u8>> {
        let mut out = b"GGUF".to_vec(); out.extend_from_slice(&3_u32.to_le_bytes());
        out.extend_from_slice(&(tensors as u64).to_le_bytes());out.extend_from_slice(&(self.entries.len() as u64).to_le_bytes());
        for (key, wire) in &self.entries { string(&mut out,key);out.extend_from_slice(wire); }
        if out.len() as u64 > MAX_HEADER { return Err("GGUF metadata exceeds budget".into()); }
        Ok(out)
    }
}

fn require(ok: bool, what: &str) -> Result<()> { if ok { Ok(()) } else { Err(what.into()) } }
fn id_of(value: &Value, reverse: &BTreeMap<String, u32>) -> Result<u32> {
    let token = value.as_str().or_else(||value.get("content").and_then(Value::as_str)).ok_or("missing named special token")?;
    reverse.get(token).copied().ok_or_else(||format!("special token absent from vocabulary: {token}").into())
}
fn special_looking(token: &str) -> bool {
    matches!(token,"<pad>"|"<mask>"|"<2mass>"|"[@BOS@]") ||
        (token.starts_with("<|") && token.ends_with("|>")) ||
        (token.starts_with("<unused") && token.ends_with('>'))
}

pub(super) fn tokenizer(metadata: &mut Metadata, tokenizer: &Value, config: &Value,
    target: &Value, chat_template: &str, vocab: usize) -> Result<()> {
    require(vocab==248320,"only the admitted QUASAR vocabulary extent is supported")?;
    let model=&tokenizer["model"];
    require(model["type"]=="BPE" && model["dropout"].is_null() && model["unk_token"].is_null() &&
        model["continuing_subword_prefix"]=="" && model["end_of_word_suffix"]=="" &&
        model["fuse_unk"]==false && model["byte_fallback"]==false && model["ignore_merges"]==false,
        "unsupported tokenizer BPE behavior")?;
    require(tokenizer["normalizer"]==serde_json::json!({"type":"NFC"}),"unsupported tokenizer normalization")?;
    // Exact structural fingerprint of the locally admitted qwen35 pre-tokenizer.
    // Do not guess a pre-tokenizer label from a model directory name.
    let pre=serde_json::to_vec(&tokenizer["pre_tokenizer"])?;
    require(format!("{:x}",Sha256::digest(&pre))=="b36f1669c1fd93eaec200835899e8c42217d172fcb9dd3f4fe725920aec5a8b3",
        "unrecognized qwen35 pre-tokenizer; independent admission required")?;
    let byte_level=serde_json::json!({"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false});
    require(tokenizer["decoder"]==byte_level && tokenizer["post_processor"]==byte_level,
        "unsupported tokenizer decoder/postprocessor")?;
    require(tokenizer["truncation"].is_null() && tokenizer["padding"].is_null(),"tokenizer input truncation/padding is not admitted")?;
    let mut tokens: Vec<Option<String>>=vec![None;vocab];let mut types=vec![1_i32;vocab];
    let mut reverse=BTreeMap::new();
    for (token,id) in model["vocab"].as_object().ok_or("missing BPE vocabulary")? {
        let id=usize::try_from(id.as_u64().ok_or("invalid vocabulary token id")?)?;
        require(id<vocab && !token.is_empty(),"invalid vocabulary id/content")?;
        require(tokens[id].replace(token.clone()).is_none(),"duplicate vocabulary token id")?;
        require(reverse.insert(token.clone(),id as u32).is_none(),"duplicate vocabulary token string")?;
    }
    let added=tokenizer["added_tokens"].as_array().ok_or("missing added token list")?;
    let decoder=config["added_tokens_decoder"].as_object().ok_or("missing added token config")?;
    require(added.len()==decoder.len(),"tokenizer and added-token config disagree")?;
    let mut added_ids=BTreeSet::new();
    for token in added {
        let id=usize::try_from(token["id"].as_u64().ok_or("invalid added token id")?)?;
        let text=token["content"].as_str().ok_or("missing added token content")?;
        require(id<vocab && added_ids.insert(id) && !text.is_empty() && text.is_ascii(),"unsupported or duplicate added token")?;
        // Current profile's added tokens are literal ASCII; no external tokenizer
        // round trip or lossy Unicode normalization is needed.
        for key in ["normalized","lstrip","rstrip","single_word"] {
            require(token[key]==false,"unsupported added-token matching behavior")?;
        }
        let other=decoder.get(&id.to_string()).ok_or("added token absent from config")?;
        for key in ["content","normalized","lstrip","rstrip","single_word","special"] {
            require(other[key]==token[key],"added token definition drift")?;
        }
        if let Some(existing)=&tokens[id] { require(existing==text,"added token collides with base vocabulary id")?; }
        if let Some(existing)=reverse.get(text) { require(*existing==id as u32,"added token collides with base vocabulary string")?; }
        tokens[id]=Some(text.into());reverse.insert(text.into(),id as u32);
        let special=token["special"].as_bool().ok_or("missing special-token flag")?;
        types[id]=if special || special_looking(text) {3} else {4};
    }
    let tokens: Vec<String>=tokens.into_iter().enumerate().map(|(id,value)|value.unwrap_or_else(||{
        types[id]=5;format!("[PAD{id}]")
    })).collect();
    let merges=model["merges"].as_array().ok_or("missing BPE merges")?.iter().map(|value| {
        if let Some(text)=value.as_str() { return Ok(text.to_owned()); }
        let pair=value.as_array().ok_or("invalid BPE merge")?;
        require(pair.len()==2,"invalid BPE merge arity")?;
        Ok(format!("{} {}",pair[0].as_str().ok_or("merge first string")?,pair[1].as_str().ok_or("merge second string")?))
    }).collect::<Result<Vec<_>>>()?;
    require(!merges.is_empty(),"empty BPE merge table")?;
    let eos=id_of(&config["eos_token"],&reverse)?;let pad=id_of(&config["pad_token"],&reverse)?;
    let bos=if config["bos_token"].is_null() {
        u32::try_from(target["bos_token_id"].as_u64().ok_or("missing target BOS id")?)?
    } else { id_of(&config["bos_token"],&reverse)? };
    require(eos==248046 && pad==248044 && bos==248044 && reverse.get("<|audio_start|>")==Some(&248070),
        "QUASAR target special-token identity drift")?;
    require(config["add_bos_token"]==false && (config["add_eos_token"].is_null() || config["add_eos_token"]==false),
        "unsupported implicit special-token insertion")?;
    require(!chat_template.is_empty(),"missing target chat template")?;
    metadata.text("tokenizer.ggml.model","gpt2")?;metadata.text("tokenizer.ggml.pre","qwen35")?;
    metadata.strings("tokenizer.ggml.tokens",&tokens)?;metadata.integers("tokenizer.ggml.token_type",&types)?;
    metadata.strings("tokenizer.ggml.merges",&merges)?;
    for (key,id) in [("eos_token_id",eos),("padding_token_id",pad),("bos_token_id",bos),("mask_token_id",248070)] {
        metadata.integer(&format!("tokenizer.ggml.{key}"),id)?;
    }
    metadata.boolean("tokenizer.ggml.add_bos_token",false)?;metadata.boolean("tokenizer.ggml.add_eos_token",false)?;
    metadata.text("tokenizer.chat_template",chat_template)?;
    Ok(())
}
