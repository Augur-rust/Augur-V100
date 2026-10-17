//! Bind a diagnostic target export to a real anchor/mask draft seed.
//! This consumes only the exported suffix range (at most 2048 rows), never the
//! unexported prefix. Source binding does not authorize target verification.

use crate::{
    DFlashDraftBlockInput, DFlashTargetFeatureCaptureError, DFlashTargetFeatureCaptureReceipt,
    DFLASH_TARGET_CAPTURE_FEATURE_WIDTH, DFLASH_TARGET_CAPTURE_HIDDEN_SIZE,
    DFLASH_TARGET_CAPTURE_LAYER_IDS, DFLASH_TARGET_CAPTURE_SCHEMA,
};
use augur_model_qwen35::DFlashConfig;
use serde_json::Value;
use sha2::{Digest, Sha256};

pub const DFLASH_SEED_MASK_TOKEN: u32 = 248_070;
type Result<T> = std::result::Result<T, DFlashTargetFeatureCaptureError>;

#[derive(Clone, Debug)]
pub struct DFlashSourceSeed {
    pub seed: DFlashDraftBlockInput,
    pub capture: DFlashTargetFeatureCaptureReceipt,
    pub source_report_sha256: [u8; 32],
    pub source_commit: String,
    pub source_binary_sha256: String,
    pub source_dirty: bool,
    pub target_input_tokens: Vec<u32>,
}

impl DFlashSourceSeed {
    /// Bind exactly the feature rows emitted by the bounded full-chain runner.
    /// Full-chain V1/V2 and target-only V1 use generation 1; do not relabel it.
    pub fn from_target_report(
        config: &DFlashConfig,
        bytes: &[u8],
        feature_elements: usize,
        feature_sha256: [u8; 32],
        expected_gguf_sha256: &str,
        expected_config_sha256: &str,
    ) -> Result<Self> {
        config
            .validate_exact_profile()
            .map_err(|e| invalid(e.to_string()))?;
        let report: Value = serde_json::from_slice(bytes).map_err(|e| invalid(e.to_string()))?;
        let context_len = integer(&report["suffix_tokens"], "suffix_tokens")?;
        let start = integer(&report["prefix_positions"], "prefix_positions")?;
        if !(1..=2048).contains(&context_len) {
            return Err(invalid("source suffix extent must be in 1..=2048"));
        }
        let capture_only = report["schema"]
            == "crystal-quant.qwen35.full-chain-target-capture.component.v1"
            && report["status"] == "PASS_COMPONENT_FULL_CHAIN_TARGET_CAPTURE";
        let (schema, status) = if capture_only {
            (
                "crystal-quant.qwen35.full-chain-target-capture.component.v1",
                "PASS_COMPONENT_FULL_CHAIN_TARGET_CAPTURE",
            )
        } else if context_len == 1 {
            (
                "crystal-quant.qwen35.full-chain-active-cold-one-suffix.component.v1",
                "PASS_COMPONENT_FULL_CHAIN_ACTIVE_COLD_ONE_SUFFIX",
            )
        } else {
            (
                "crystal-quant.qwen35.full-chain-active-cold-multi-suffix.component.v2",
                "PASS_COMPONENT_FULL_CHAIN_ACTIVE_COLD_MULTI_SUFFIX",
            )
        };
        if report["schema"] != schema
            || report["status"] != status
            || report["complete_64_layer_chain_executed"] != true
        {
            return Err(invalid("source runner schema/status/complete-chain"));
        }
        if capture_only
            && (report["capture_only"] != true
                || report["component_only"] != true
                || report["diagnostic_only"] != true
                || report["claims"]["target_capture"] != true
                || report["full_draft_window_supplied"] != (context_len == 2048)
                || [
                    "candidate_executed",
                    "semantic_parity",
                    "rolling_decode",
                    "product_ready",
                    "default_cli",
                    "token_publication",
                    "kv_publication",
                    "visible_tps",
                    "context_32k_qualified",
                    "context_256k_qualified",
                ]
                .iter()
                .any(|key| report["claims"][key] != false))
        {
            return Err(invalid("target capture claim boundary"));
        }
        let gguf = decode_digest(expected_gguf_sha256)?;
        let config_digest = decode_digest(expected_config_sha256)?;
        if report["gguf_sha256"] != expected_gguf_sha256
            || report["config_sha256"] != expected_config_sha256
        {
            return Err(invalid("target source artifact/config mismatch"));
        }
        let export = &report["target_feature_export"];
        if export["source"] != "same_target_oracle_hidden_rows"
            || export["layer_ids"] != serde_json::json!(DFLASH_TARGET_CAPTURE_LAYER_IDS)
            || export["rows_per_step"] != 5
            || export["row_elements"] != DFLASH_TARGET_CAPTURE_HIDDEN_SIZE
            || integer(&export["elements"], "feature elements")? != feature_elements
            || feature_elements != context_len * DFLASH_TARGET_CAPTURE_FEATURE_WIDTH
            || decode_digest(
                export["sha256"]
                    .as_str()
                    .ok_or_else(|| invalid("feature digest"))?,
            )? != feature_sha256
        {
            return Err(invalid("target export feature geometry/digest"));
        }
        // Full-chain V1 uses root step fields; V2 and capture-only use steps.
        // Interpret the actual producer ABI, not a fabricated V1 steps array.
        let single_step = if context_len == 1 && !capture_only {
            if report.get("steps").is_some() {
                return Err(invalid("V1 source must use root step fields"));
            }
            vec![serde_json::json!({
                "step":0, "absolute_position":start,
                "pass":report["final_token_equal"],
                "final_token_equal":report["final_token_equal"],
                "oracle_top50":report["oracle_top50"]
            })]
        } else {
            Vec::new()
        };
        let steps = if context_len == 1 && !capture_only {
            &single_step
        } else {
            report["steps"]
                .as_array()
                .ok_or_else(|| invalid("source steps"))?
        };
        let tokens = report["oracle_input_tokens"]
            .as_array()
            .ok_or_else(|| invalid("source tokens"))?;
        if steps.len() != context_len || tokens.len() != context_len {
            return Err(invalid("source step/token count"));
        }
        let position = start
            .checked_add(context_len)
            .ok_or_else(|| invalid("source position overflow"))?;
        let mut input_tokens = Vec::with_capacity(context_len);
        for (index, step) in steps.iter().enumerate() {
            if integer(&step["step"], "step index")? != index
                || integer(&step["absolute_position"], "step position")? != start + index
            {
                return Err(invalid("source step position or numerical gate"));
            }
            if capture_only {
                if step["oracle_step_pass"] != true
                    || step["target_capture_only"] != true
                    || step["oracle_top50"]
                        .as_array()
                        .is_none_or(|top| top.len() != 50)
                {
                    return Err(invalid("target capture step gate/top50"));
                }
                let top = step["oracle_top50"].as_array().unwrap();
                for (rank, candidate) in top.iter().enumerate() {
                    token(candidate, config)?;
                    if top[..rank].contains(candidate) {
                        return Err(invalid("target capture duplicate top50 token"));
                    }
                }
            } else if step["pass"] != true || step["final_token_equal"] != true {
                return Err(invalid("source step position or numerical gate"));
            }
            input_tokens.push(token(&tokens[index], config)?);
        }
        let anchor = token(&steps[context_len - 1]["oracle_top50"][0], config)?;
        let source_commit = report["source_commit"]
            .as_str()
            .ok_or_else(|| invalid("source commit"))?;
        if ![8, 40].contains(&source_commit.len())
            || !source_commit.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return Err(invalid(
                "source commit must be an explicit 8/40 hex diagnostic label",
            ));
        }
        let binary = report["binary_sha256"]
            .as_str()
            .ok_or_else(|| invalid("source binary"))?;
        decode_digest(binary)?;
        let source_dirty = report["source_dirty"]
            .as_bool()
            .ok_or_else(|| invalid("source dirty flag"))?;
        let mut identity = Sha256::new();
        identity.update(b"qwen35-target-artifact-config-v1\0");
        identity.update(gguf);
        identity.update(config_digest);
        let capture = DFlashTargetFeatureCaptureReceipt {
            schema: DFLASH_TARGET_CAPTURE_SCHEMA.into(),
            request_generation: 1,
            context_start_position: start,
            context_len,
            target_layer_ids: DFLASH_TARGET_CAPTURE_LAYER_IDS.to_vec(),
            target_hidden_size: DFLASH_TARGET_CAPTURE_HIDDEN_SIZE,
            feature_width: DFLASH_TARGET_CAPTURE_FEATURE_WIDTH,
            feature_elements,
            source_target_model_sha256: identity.finalize().into(),
            feature_sha256,
        };
        let seed = DFlashDraftBlockInput {
            request_generation: 1,
            absolute_position_start: position,
            anchor_token: anchor,
            block_tokens: std::iter::once(anchor)
                .chain(std::iter::repeat_n(
                    DFLASH_SEED_MASK_TOKEN,
                    config.block_size - 1,
                ))
                .collect(),
            mask_token_id: DFLASH_SEED_MASK_TOKEN,
            target_feature_sha256: feature_sha256,
        };
        capture.validate_for_block(config, &seed, feature_elements, feature_sha256)?;
        Ok(Self {
            seed,
            capture,
            source_report_sha256: Sha256::digest(bytes).into(),
            source_commit: source_commit.into(),
            source_binary_sha256: binary.into(),
            source_dirty,
            target_input_tokens: input_tokens,
        })
    }
}

/// Read the two distinct target embedding rows, then build [anchor, mask x7].
/// The caller supplies a descriptor-bound reader, not an arbitrary hidden row.
pub fn embed_dflash_seed(
    config: &DFlashConfig,
    seed: &DFlashDraftBlockInput,
    mut read_row: impl FnMut(u32) -> std::result::Result<Vec<f32>, String>,
) -> Result<Vec<f32>> {
    seed.validate_initial_seed(config)
        .map_err(|e| invalid(e.to_string()))?;
    if seed.mask_token_id != DFLASH_SEED_MASK_TOKEN {
        return Err(invalid("seed mask identity"));
    }
    let anchor = read_row(seed.anchor_token).map_err(invalid)?;
    let mask = read_row(seed.mask_token_id).map_err(invalid)?;
    if anchor.len() != config.hidden_size
        || mask.len() != config.hidden_size
        || anchor.iter().chain(&mask).any(|v| !v.is_finite())
    {
        return Err(invalid("target embedding row extent/nonfinite"));
    }
    let mut output = Vec::with_capacity(config.block_size * config.hidden_size);
    output.extend_from_slice(&anchor);
    for _ in 1..config.block_size {
        output.extend_from_slice(&mask);
    }
    Ok(output)
}

fn integer(value: &Value, label: &str) -> Result<usize> {
    value
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| invalid(label))
}
fn token(value: &Value, config: &DFlashConfig) -> Result<u32> {
    let id = integer(value, "source token")?;
    if id >= config.vocab_size {
        return Err(invalid("source token outside vocabulary"));
    }
    Ok(id as u32)
}
fn decode_digest(value: &str) -> Result<[u8; 32]> {
    if value.len() != 64 || !value.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(invalid("source digest must be 64 hex digits"));
    }
    let mut result = [0; 32];
    for (i, out) in result.iter_mut().enumerate() {
        *out =
            u8::from_str_radix(&value[2 * i..2 * i + 2], 16).map_err(|e| invalid(e.to_string()))?;
    }
    if result == [0; 32] {
        return Err(invalid("source digest must be nonzero"));
    }
    Ok(result)
}
fn invalid(message: impl Into<String>) -> DFlashTargetFeatureCaptureError {
    DFlashTargetFeatureCaptureError::Invalid(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config() -> DFlashConfig {
        DFlashConfig {
            architecture: "DFlash2DraftModel".into(),
            model_type: "qwen3".into(),
            dtype: "float16".into(),
            hidden_size: 5120,
            intermediate_size: 17408,
            num_hidden_layers: 5,
            num_target_layers: 64,
            attention_heads: 32,
            kv_heads: 8,
            head_dim: 128,
            max_position_embeddings: 262144,
            vocab_size: 248320,
            block_size: 8,
            conv_group_size: 16,
            conv_kernel_size: 2,
            selector_rank: 256,
            selector_top_k: 16,
            target_layer_ids: vec![5, 19, 33, 47, 61],
        }
    }
    fn report() -> Value {
        serde_json::json!({
            "schema":"crystal-quant.qwen35.full-chain-active-cold-multi-suffix.component.v2",
            "status":"PASS_COMPONENT_FULL_CHAIN_ACTIVE_COLD_MULTI_SUFFIX",
            "complete_64_layer_chain_executed":true, "suffix_tokens":2,
            "prefix_positions":4096, "source_commit":"0af2e88d", "source_dirty":true,
            "binary_sha256":"44".repeat(32), "gguf_sha256":"11".repeat(32),
            "config_sha256":"22".repeat(32), "oracle_input_tokens":[123,124],
            "steps":[{"step":0,"absolute_position":4096,"pass":true,
                "final_token_equal":true,"oracle_top50":[271]},
                {"step":1,"absolute_position":4097,"pass":true,
                "final_token_equal":true,"oracle_top50":[38317]}],
            "target_feature_export":{"source":"same_target_oracle_hidden_rows",
                "layer_ids":[5,19,33,47,61], "rows_per_step":5,"row_elements":5120,
                "elements":51200,"sha256":"33".repeat(32)}
        })
    }
    fn bind(report: &Value) -> Result<DFlashSourceSeed> {
        DFlashSourceSeed::from_target_report(
            &config(),
            &serde_json::to_vec(report).unwrap(),
            51200,
            [0x33; 32],
            &"11".repeat(32),
            &"22".repeat(32),
        )
    }
    #[test]
    fn source_positions_and_anchor_are_derived_without_relabeling() {
        let bound = bind(&report()).unwrap();
        assert_eq!(bound.capture.context_start_position, 4096);
        assert_eq!(bound.seed.absolute_position_start, 4098);
        assert_eq!(
            bound.seed.block_tokens,
            vec![38317, 248070, 248070, 248070, 248070, 248070, 248070, 248070]
        );
        assert!(bound.source_dirty); // Never promote a diagnostic source label.
    }
    #[test]
    fn source_geometry_identity_status_and_position_drift_reject() {
        for (pointer, bad) in [
            ("/prefix_positions", serde_json::json!(4092)),
            ("/suffix_tokens", serde_json::json!(8)),
            (
                "/target_feature_export/sha256",
                serde_json::json!("55".repeat(32)),
            ),
            ("/gguf_sha256", serde_json::json!("55".repeat(32))),
            ("/config_sha256", serde_json::json!("55".repeat(32))),
            (
                "/target_feature_export/layer_ids",
                serde_json::json!([6, 20, 34, 48, 62]),
            ),
            ("/steps/0/absolute_position", serde_json::json!(4095)),
            ("/steps/1/oracle_top50/0", serde_json::json!(248320)),
            ("/steps/0/pass", serde_json::json!(false)),
            ("/source_dirty", serde_json::json!("false")),
            ("/status", serde_json::json!("FAIL")),
        ] {
            let mut bad_report = report();
            *bad_report.pointer_mut(pointer).unwrap() = bad;
            assert!(bind(&bad_report).is_err(), "{pointer}");
        }
    }

    #[test]
    fn source_context_contract_accepts_full_dflash_history_bound() {
        let mut source = report();
        let steps = source["steps"].as_array().unwrap().clone();
        let tokens = source["oracle_input_tokens"].as_array().unwrap().clone();
        let mut expanded_steps = Vec::with_capacity(2048);
        let mut expanded_tokens = Vec::with_capacity(2048);
        for index in 0..2048 {
            let mut step = steps[index % steps.len()].clone();
            step["step"] = serde_json::json!(index);
            step["absolute_position"] = serde_json::json!(4096 + index);
            expanded_steps.push(step);
            expanded_tokens.push(tokens[index % tokens.len()].clone());
        }
        source["suffix_tokens"] = serde_json::json!(2048);
        source["steps"] = serde_json::json!(expanded_steps);
        source["oracle_input_tokens"] = serde_json::json!(expanded_tokens);
        source["target_feature_export"]["elements"] = serde_json::json!(2048 * 25600);
        source["target_feature_export"]["sha256"] = serde_json::json!("11".repeat(32));
        let bound = DFlashSourceSeed::from_target_report(
            &config(),
            serde_json::to_string(&source).unwrap().as_bytes(),
            2048 * 25600,
            [0x11; 32],
            &"11".repeat(32),
            &"22".repeat(32),
        )
        .unwrap();
        assert_eq!(bound.capture.context_len, 2048);
        assert_eq!(bound.seed.absolute_position_start, 6144);
        // 2049 remains rejected by the explicit extent gate below.
        source["suffix_tokens"] = serde_json::json!(2049);
        assert!(DFlashSourceSeed::from_target_report(
            &config(),
            serde_json::to_string(&source).unwrap().as_bytes(),
            2049 * 25600,
            [0x11; 32],
            &"11".repeat(32),
            &"22".repeat(32),
        )
        .is_err());
    }
    #[test]
    fn embedding_reads_anchor_and_mask_not_repeated_anchor() {
        let bound = bind(&report()).unwrap();
        let mut ids = Vec::new();
        let values = embed_dflash_seed(&config(), &bound.seed, |id| {
            ids.push(id);
            Ok(vec![id as f32; 5120])
        })
        .unwrap();
        assert_eq!(ids, vec![38317, 248070]);
        assert!(values[..5120].iter().all(|v| *v == 38317.0));
        assert!(values[5120..].iter().all(|v| *v == 248070.0));
        assert!(embed_dflash_seed(&config(), &bound.seed, |_| Ok(vec![0.; 5119])).is_err());
        assert!(embed_dflash_seed(&config(), &bound.seed, |_| Ok(vec![f32::NAN; 5120])).is_err());
    }

    #[test]
    fn actual_one_suffix_schema_uses_root_token_fields() {
        let mut source = report();
        source["schema"] = serde_json::json!(
            "crystal-quant.qwen35.full-chain-active-cold-one-suffix.component.v1"
        );
        source["status"] = serde_json::json!("PASS_COMPONENT_FULL_CHAIN_ACTIVE_COLD_ONE_SUFFIX");
        source["suffix_tokens"] = serde_json::json!(1);
        source["oracle_input_tokens"] = serde_json::json!([123]);
        source["target_feature_export"]["elements"] = serde_json::json!(25600);
        source.as_object_mut().unwrap().remove("steps");
        source["final_token_equal"] = serde_json::json!(true);
        source["oracle_top50"] = serde_json::json!([38317]);
        let run = |source: &Value| {
            DFlashSourceSeed::from_target_report(
                &config(),
                &serde_json::to_vec(source).unwrap(),
                25600,
                [0x33; 32],
                &"11".repeat(32),
                &"22".repeat(32),
            )
        };
        let bound = run(&source).unwrap();
        assert_eq!(bound.seed.absolute_position_start, 4097);
        assert_eq!(bound.seed.anchor_token, 38317);
        source["final_token_equal"] = serde_json::json!(false);
        assert!(run(&source).is_err());
    }

    fn target_capture_report(rows: usize) -> Value {
        let mut source = report();
        source["schema"] =
            serde_json::json!("crystal-quant.qwen35.full-chain-target-capture.component.v1");
        source["status"] = serde_json::json!("PASS_COMPONENT_FULL_CHAIN_TARGET_CAPTURE");
        source["capture_only"] = serde_json::json!(true);
        source["component_only"] = serde_json::json!(true);
        source["diagnostic_only"] = serde_json::json!(true);
        source["full_draft_window_supplied"] = serde_json::json!(rows == 2048);
        source["suffix_tokens"] = serde_json::json!(rows);
        source["target_feature_export"]["elements"] = serde_json::json!(rows * 25600);
        source["oracle_input_tokens"] = serde_json::json!(vec![123; rows]);
        source["claims"] = serde_json::json!({
            "target_capture": true, "candidate_executed": false,
            "semantic_parity": false, "rolling_decode": false, "product_ready": false,
            "default_cli": false, "token_publication": false, "kv_publication": false,
            "visible_tps": false, "context_32k_qualified": false, "context_256k_qualified": false
        });
        source["steps"] = serde_json::json!((0..rows)
            .map(|step| serde_json::json!({
                "step": step,
                "absolute_position": 4096 + step,
                "oracle_step_pass": true,
                "oracle_top50": (0..50).map(|token| 38317 + token).collect::<Vec<_>>(),
                "target_capture_only": true
            }))
            .collect::<Vec<_>>());
        source
    }

    #[test]
    fn target_capture_schema_accepts_only_oracle_steps() {
        let mut source = target_capture_report(2);
        let bound = bind(&source).unwrap();
        assert_eq!(bound.seed.anchor_token, 38317);

        source["steps"][0]["oracle_step_pass"] = serde_json::json!(false);
        assert!(bind(&source).is_err());
        source["steps"][0]["oracle_step_pass"] = serde_json::json!(true);
        source["steps"][0]["oracle_top50"] = serde_json::json!([38317]);
        assert!(bind(&source).is_err());
        source["steps"][0]["oracle_top50"] =
            serde_json::json!((0..50).map(|token| 38317 + token).collect::<Vec<_>>());
        source["steps"][0]["absolute_position"] = serde_json::json!(4095);
        assert!(bind(&source).is_err());
    }

    #[test]
    fn target_capture_full_window_and_one_row_bind_without_candidate_parity() {
        for rows in [1, 2048] {
            let source = target_capture_report(rows);
            assert!(source.get("final_token_equal").is_none());
            assert!(source["steps"][0].get("final_token_equal").is_none());
            let bound = DFlashSourceSeed::from_target_report(
                &config(),
                &serde_json::to_vec(&source).unwrap(),
                rows * 25600,
                [0x33; 32],
                &"11".repeat(32),
                &"22".repeat(32),
            )
            .unwrap();
            assert_eq!(bound.capture.context_len, rows);
            assert_eq!(bound.seed.absolute_position_start, 4096 + rows);
            assert_eq!(bound.seed.anchor_token, 38317);
        }
    }

    #[test]
    fn target_capture_drift_missing_fields_and_authority_reject() {
        for (pointer, bad) in [
            ("/suffix_tokens", serde_json::json!(2049)),
            ("/steps", Value::Null),
            ("/steps/0/step", serde_json::json!(1)),
            ("/steps/0/oracle_top50/49", serde_json::json!(248320)),
            ("/steps/0/oracle_top50/1", serde_json::json!(38317)),
            ("/steps/0/target_capture_only", serde_json::json!(false)),
            ("/steps/0/oracle_step_pass", Value::Null),
            ("/full_draft_window_supplied", serde_json::json!(true)),
            ("/claims/candidate_executed", serde_json::json!(true)),
            ("/claims/semantic_parity", serde_json::json!(true)),
            ("/claims/product_ready", serde_json::json!(true)),
            ("/claims/visible_tps", serde_json::json!(true)),
            ("/claims/rolling_decode", Value::Null),
            ("/capture_only", serde_json::json!(false)),
            ("/status", serde_json::json!("FAIL")),
        ] {
            let mut source = target_capture_report(2);
            *source.pointer_mut(pointer).unwrap() = bad;
            assert!(bind(&source).is_err(), "{pointer}");
        }
    }
}
