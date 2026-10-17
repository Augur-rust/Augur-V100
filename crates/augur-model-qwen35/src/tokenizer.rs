use crate::identity::{read_stable_bytes, sha256_hex};
use crate::strict_json::StrictValue;
use crate::Qwen35Error;
use serde::Serialize;
use std::collections::BTreeSet;
use std::path::Path;

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TokenizerIdentity {
    pub tokenizer_json_bytes: u64,
    pub tokenizer_json_sha256: String,
    pub tokenizer_model: String,
    pub tokenizer_vocab_entries: usize,
    pub added_token_count: usize,
    pub maximum_token_id: u32,
    pub chat_template_bytes: u64,
    pub chat_template_sha256: String,
}

impl TokenizerIdentity {
    pub fn audit(
        tokenizer_path: impl AsRef<Path>,
        chat_template_path: impl AsRef<Path>,
    ) -> Result<Self, Qwen35Error> {
        let tokenizer_bytes = read_stable_bytes(tokenizer_path.as_ref())?;
        let tokenizer = StrictValue::parse(&tokenizer_bytes)?;
        let tokenizer_root = tokenizer.object("tokenizer.json")?;
        let model = tokenizer_root
            .get("model")
            .ok_or_else(|| Qwen35Error::MissingField("tokenizer.json.model".into()))?
            .object("tokenizer.json.model")?;
        let tokenizer_model = model
            .get("type")
            .ok_or_else(|| Qwen35Error::MissingField("tokenizer.json.model.type".into()))?
            .string("tokenizer.json.model.type")?
            .to_owned();
        let vocab = model
            .get("vocab")
            .ok_or_else(|| Qwen35Error::MissingField("tokenizer.json.model.vocab".into()))?
            .object("tokenizer.json.model.vocab")?;
        let mut vocab_ids = BTreeSet::new();
        for value in vocab.values() {
            vocab_ids.insert(value.u32("tokenizer.json.model.vocab")?);
        }
        let tokenizer_vocab_entries = vocab.len();
        if vocab_ids.len() != tokenizer_vocab_entries
            || vocab_ids.iter().copied().next() != Some(0)
            || vocab_ids.iter().copied().last()
                != Some((tokenizer_vocab_entries.saturating_sub(1)) as u32)
        {
            return Err(Qwen35Error::invalid(
                "tokenizer.json.model.vocab",
                "base BPE ids must be unique and contiguous from zero",
            ));
        }
        if tokenizer_model != "BPE" {
            return Err(Qwen35Error::invalid(
                "tokenizer.json.model.type",
                format!("expected BPE, got {tokenizer_model:?}"),
            ));
        }

        let added_tokens = tokenizer_root
            .get("added_tokens")
            .ok_or_else(|| Qwen35Error::MissingField("tokenizer.json.added_tokens".into()))?
            .array("tokenizer.json.added_tokens")?;
        let mut added_ids = BTreeSet::new();
        for (index, token) in added_tokens.iter().enumerate() {
            let object = token.object("tokenizer.json.added_tokens")?;
            let id = object
                .get("id")
                .ok_or_else(|| {
                    Qwen35Error::MissingField(format!("tokenizer.json.added_tokens[{index}].id"))
                })?
                .u32("tokenizer.json.added_tokens.id")?;
            if id < tokenizer_vocab_entries as u32 || !added_ids.insert(id) {
                return Err(Qwen35Error::invalid(
                    "tokenizer.json.added_tokens",
                    "added token ids must be unique and follow the base vocab",
                ));
            }
        }
        let maximum_token_id = added_ids
            .iter()
            .copied()
            .max()
            .unwrap_or(tokenizer_vocab_entries.saturating_sub(1) as u32);

        let chat_bytes = read_stable_bytes(chat_template_path.as_ref())?;
        let chat_template = std::str::from_utf8(&chat_bytes)
            .map_err(|_| Qwen35Error::invalid("chat_template.jinja", "must be UTF-8"))?;
        if chat_template.trim().is_empty() {
            return Err(Qwen35Error::invalid(
                "chat_template.jinja",
                "template must not be empty",
            ));
        }
        Ok(Self {
            tokenizer_json_bytes: tokenizer_bytes.len() as u64,
            tokenizer_json_sha256: sha256_hex(&tokenizer_bytes),
            tokenizer_model,
            tokenizer_vocab_entries,
            added_token_count: added_tokens.len(),
            maximum_token_id,
            chat_template_bytes: chat_bytes.len() as u64,
            chat_template_sha256: sha256_hex(&chat_bytes),
        })
    }
}
