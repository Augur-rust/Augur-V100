//! Opt-in exact-u32 fresh reference only. Does not widen HTTP admission.
use super::*;
use std::path::Path;

const MAX_INPUT_BYTES: u64 = 256 * 1024;
const MAX_LONG_INPUT_BYTES: u64 = 4 * 1024 * 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExactInput {
    schema: String,
    token_ids: Vec<u32>,
    consumed_input_digest: String,
}

fn input_digest(ids: &[u32]) -> String {
    let mut h = Sha256::new();
    h.update(b"QWEN35-SESSION-CONSUMED-U32LE-V1\0");
    for id in ids {
        h.update(id.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}

#[cfg(test)]
fn parse_input(raw: &[u8], expected_sha: &str) -> Result<ExactInput, Box<dyn Error>> {
    parse_input_profile(raw, expected_sha, false)
}
#[cfg(test)]
fn parse_input_profile(raw: &[u8], expected_sha: &str, growing: bool) -> Result<ExactInput, Box<dyn Error>> {
    parse_input_context(raw, expected_sha, growing, None)
}

fn parse_input_context(
    raw: &[u8], expected_sha: &str, growing: bool,
    long: Option<NativeLongReferenceProfile>,
) -> Result<ExactInput, Box<dyn Error>> {
    if let Some(profile) = long {
        profile.validate_selection(true, growing)?;
    }
    let byte_limit = if long.is_some() { MAX_LONG_INPUT_BYTES } else { MAX_INPUT_BYTES };
    if raw.len() as u64 > byte_limit || digest(raw) != expected_sha {
        return Err("reference input bytes/hash mismatch".into());
    }
    let input: ExactInput = serde_json::from_slice(raw)?;
    let (schema, bounds) = if let Some(profile) = long {
        let (minimum, maximum) = profile.bounds();
        (profile.schema(), minimum..=maximum)
    } else if growing {
        ("qwen35-growing-exact-history-reference-input-v1", 1..=16384)
    } else { ("qwen35-exact-history-reference-input-v1", 8193..=11264) };
    if input.schema != schema
        || !bounds.contains(&input.token_ids.len())
        || input.token_ids.iter().any(|&t| t >= 248320)
        || input.consumed_input_digest != input_digest(&input.token_ids)
    {
        return Err("reference schema/extent/token/digest mismatch".into());
    }
    Ok(input)
}

fn save(path: &Path, value: &Value) -> Result<(), Box<dyn Error>> {
    let mut bound = value.clone();
    if bound.is_object() {
        augur_model_qwen35_runtime::build_identity::annotate(&mut bound);
    }
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(&serde_json::to_vec_pretty(&bound)?)?;
    file.flush()?;
    Ok(())
}

pub fn run(
    config: Qwen35AutoregressiveCudaConfig,
    q8: PathBuf,
    draft_config: PathBuf,
    generation: PathBuf,
    evidence: PathBuf,
    input_path: &Path,
    expected_sha: &str,
    growing: bool,
    long: Option<NativeLongReferenceProfile>,
) -> Result<(), Box<dyn Error>> {
    if !evidence.is_absolute() || evidence.exists() || !input_path.is_absolute() {
        return Err("reference needs new absolute evidence and absolute input".into());
    }
    // External observer pins the file too; reject local alias/reparse components.
    for p in input_path.ancestors() {
        let m = fs::symlink_metadata(p)?;
        #[cfg(windows)]
        {
            use std::os::windows::fs::MetadataExt;
            if m.file_attributes() & 0x400 != 0 {
                return Err("reference input reparse path".into());
            }
        }
        if m.file_type().is_symlink() || (p == input_path && !m.is_file()) {
            return Err("reference input must be regular and non-symlink".into());
        }
    }
    let mut raw = Vec::new();
    let byte_limit = if long.is_some() { MAX_LONG_INPUT_BYTES } else { MAX_INPUT_BYTES };
    fs::File::open(input_path)?.take(byte_limit + 1).read_to_end(&mut raw)?;
    let input = parse_input_context(&raw, expected_sha, growing, long)?;
    fs::create_dir(&evidence)?;
    save(&evidence.join("input-receipt.json"), &json!({
        "input_sha256":expected_sha,"token_count":input.token_ids.len(),
        "consumed_input_digest":input.consumed_input_digest,
        "growing_initialized_history":growing,"reference_max_new_tokens":128,
        "long_reference_profile":long.map(|profile| profile.name()),
        "state_evidence":if long.is_some() {"128_state_plane_digests_no_raw_plane_dump"} else {"raw_planes_and_features"},
        "product_ready":false,"semantic_qualification":false}))?;
    let baseline = augur_backend::cuda_device_allocation_snapshot();
    let mut owner = None;
    let outcome = (|| -> Result<Value, Box<dyn Error>> {
        owner = Some(Qwen35ResidentDraftCuda::open_with_target_and_vision(&config, &q8, &draft_config)?);
        let resident = augur_backend::cuda_device_allocation_snapshot();
        save(&evidence.join("resident.json"), &allocation_json(resident))?;
        let resident_owner = owner.as_mut().ok_or("missing resident owner")?;
        let commits = RefCell::new(Vec::<Value>::new());
        let sink: &Qwen35VerifiedCommitSink<'_> = &|commit| {
            commits.borrow_mut().push(json!({"initial":commit.initial_anchor,
                "cursor_before":commit.cursor_before,"cursor_after":commit.cursor_after,
                "generation_before":commit.generation_before,"generation_after":commit.generation_after,
                "tokens":commit.token_ids,"stop_reason":commit.stop_reason}));
            Ok(())
        };
        let private = evidence.join("reference-private");
        let reference = if let Some(profile) = long {
            // These are the existing context-specific full-canonical methods;
            // no new numerical path, KV layout, acceptance or HTTP admission.
            match profile {
                NativeLongReferenceProfile::Context32 => resident_owner.fresh_recompute_native_session_context32_audit(
                    &config, "context32-exact-reference", options(&generation)?, &input.token_ids, &private, sink,
                )?,
                NativeLongReferenceProfile::Context64 => resident_owner.fresh_recompute_native_session_context64_audit(
                    &config, "context64-exact-reference", options(&generation)?, &input.token_ids, &private, sink,
                )?,
                NativeLongReferenceProfile::Near192 => resident_owner.fresh_recompute_native_session_near192_audit(
                    &config, "near192-exact-reference", options(&generation)?, &input.token_ids, &private, sink,
                )?,
            }
        } else if growing {
            resident_owner.fresh_recompute_native_growing_session_long_audit(
                &config, "growing-exact-reference", options(&generation)?, &input.token_ids,
                &evidence.join("reference-state"), sink,
            )?
        } else {
            resident_owner.fresh_recompute_native_session_long_audit(
                &config, "heldout-exact-reference", options(&generation)?, &input.token_ids,
                &evidence.join("reference-state"), &evidence.join("reference-private"), sink,
            )?
        };
        save(&evidence.join("fresh-reference.json"), &reference)?;
        save(&evidence.join("commits.json"), &json!(commits.into_inner()))?;
        let prompt: Vec<u32> = serde_json::from_value(reference["request_output"]["prompt_token_ids"].clone())?;
        if prompt != input.token_ids || reference["native_exact_token_reference"] != true {
            return Err("fresh reference exact ledger not consumed".into());
        }
        let close = resident_owner.close_native_session(&reference["session_checkpoint"])?;
        save(&evidence.join("close.json"), &close)?;
        let after = augur_backend::cuda_device_allocation_snapshot();
        if after.live_bytes != resident.live_bytes || after.live_allocations != resident.live_allocations {
            return Err("reference did not return immutable resident baseline".into());
        }
        Ok(json!({"exact_input_tokens":prompt.len(),"candidate_comparison":"NOT_RUN",
            "product_ready":false,"semantic_qualification":false,"qualified":false,"visible_tps":false}))
    })();
    drop(owner);
    let after = augur_backend::cuda_device_allocation_snapshot();
    let clean = after.live_bytes == baseline.live_bytes && after.live_allocations == baseline.live_allocations
        && after.successful_allocations.checked_sub(baseline.successful_allocations)
            == after.successful_frees.checked_sub(baseline.successful_frees);
    let schema = long.map(|profile| format!("qwen35-{}-exact-reference-v1", profile.name()))
        .unwrap_or_else(|| if growing {"qwen35-growing-exact-reference-v1"} else {"qwen35-heldout-exact-reference-v1"}.into());
    save(&evidence.join("complete.json"), &json!({
        "schema":schema,"long_reference_profile":long.map(|profile| profile.name()),
        "growing_initialized_history":growing,"success":outcome.is_ok() && clean,
        "result":outcome.as_ref().ok(),"error":outcome.as_ref().err().map(|e| e.to_string()),
        "cleanup_ok":clean,"before":allocation_json(baseline),"after":allocation_json(after),
        "product_ready":false,"semantic_qualification":false,"qualified":false,"visible_tps":false}))?;
    outcome?;
    if !clean { return Err("exact reference CUDA cleanup failed".into()); }
    println!("QWEN35_HELDOUT_EXACT_REFERENCE=EXECUTED_COMPARISON_REQUIRED");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> Value {
        let ids = vec![17u32; 8193];
        json!({"schema":"qwen35-exact-history-reference-input-v1",
            "token_ids":ids,"consumed_input_digest":input_digest(&ids)})
    }
    #[test]
    fn reference_binds_exact_u32_and_external_bytes() {
        let raw = serde_json::to_vec(&fixture()).unwrap();
        assert_eq!(parse_input(&raw, &digest(&raw)).unwrap().token_ids.len(), 8193);
        assert!(parse_input(&raw, &"0".repeat(64)).is_err());
    }

    #[test]
    fn growing_reference_schema_and_extent_cannot_alias_legacy() {
        for n in [0,1,7,31,2047,2048,2049,8193,11264,16384,16385] {
            let ids = vec![17u32;n];
            let value = json!({"schema":"qwen35-growing-exact-history-reference-input-v1",
                "token_ids":ids,"consumed_input_digest":input_digest(&ids)});
            let raw = serde_json::to_vec(&value).unwrap();
            assert_eq!(parse_input_profile(&raw,&digest(&raw),true).is_ok(), (1..=16384).contains(&n));
            assert!(parse_input(&raw,&digest(&raw)).is_err());
            assert!(parse_input_profile(&raw,&"0".repeat(64),true).is_err());
        }
        let raw = serde_json::to_vec(&fixture()).unwrap();
        assert!(parse_input_profile(&raw,&digest(&raw),true).is_err());
    }

    #[test]
    fn growing_reference_keeps_strict_tokens_digest_fields_and_duplicate_rejection() {
        let ids=vec![17u32;31];
        let value=json!({"schema":"qwen35-growing-exact-history-reference-input-v1",
            "token_ids":ids,"consumed_input_digest":input_digest(&ids)});
        for (key,bad) in [("token_ids",json!([true])),("token_ids",json!([248320])),
            ("token_ids",json!([-1])),("consumed_input_digest",json!("wrong")),("authority",json!(true))] {
            let mut changed=value.clone();changed[key]=bad;
            let raw=serde_json::to_vec(&changed).unwrap();
            assert!(parse_input_profile(&raw,&digest(&raw),true).is_err());
        }
        let raw=serde_json::to_vec(&value).unwrap();
        let mut duplicate=b"{\"schema\":\"duplicate\",".to_vec();
        duplicate.extend_from_slice(&raw[1..]);
        assert!(parse_input_profile(&duplicate,&digest(&duplicate),true).is_err());
    }
    #[test]
    fn reference_rejects_extent_domain_digest_and_unknown_keys() {
        for field in ["schema", "token_ids", "consumed_input_digest", "product_ready"] {
            let mut value = fixture();
            value[field] = if field == "token_ids" { json!([248320]) } else { json!(true) };
            let raw = serde_json::to_vec(&value).unwrap();
            assert!(parse_input(&raw, &digest(&raw)).is_err());
        }
        for ids in [vec![1u32;8192],vec![1;11265],vec![248320;8193]] {
            let mut value = fixture();
            value["token_ids"] = json!(ids);
            value["consumed_input_digest"] = json!(input_digest(&ids));
            let raw = serde_json::to_vec(&value).unwrap();
            assert!(parse_input(&raw, &digest(&raw)).is_err());
        }
    }
    #[test]
    fn reference_rejects_duplicate_keys_and_oversized_bytes() {
        let raw = serde_json::to_vec(&fixture()).unwrap();
        let mut duplicate = b"{\"schema\":\"duplicate\",".to_vec();
        duplicate.extend_from_slice(&raw[1..]);
        assert!(parse_input(&duplicate, &digest(&duplicate)).is_err());
        let large = vec![b' '; MAX_INPUT_BYTES as usize + 1];
        assert!(parse_input(&large, &digest(&large)).is_err());
    }
    #[test]
    fn selection_cannot_fall_through_into_http() {
        let absolute = std::env::temp_dir().join("reference.json");
        let sha = "a".repeat(64);
        assert!(validate_native_exact_reference(None,None,false).is_ok());
        assert!(validate_native_exact_reference(Some(&absolute),Some(&sha),true).is_ok());
        assert!(validate_native_exact_reference(Some(&absolute),Some(&sha),false).is_err());
        assert!(validate_native_exact_reference(None,Some(&sha),true).is_err());
        assert!(validate_native_exact_reference(Some(&absolute),None,true).is_err());
        assert!(validate_native_exact_reference(Some(Path::new("relative")),Some(&sha),true).is_err());
        assert!(validate_native_exact_reference(Some(&absolute),Some(&"A".repeat(64)),true).is_err());
    }

    fn long_fixture(profile: NativeLongReferenceProfile, count: usize) -> Value {
        let ids = vec![17u32; count];
        json!({"schema":profile.schema(),"token_ids":ids,
               "consumed_input_digest":input_digest(&ids)})
    }

    #[test]
    fn long_reference_selection_is_explicit_feature_bound_and_not_growing() {
        for profile in [NativeLongReferenceProfile::Context32, NativeLongReferenceProfile::Context64,
                        NativeLongReferenceProfile::Near192] {
            assert_eq!(NativeLongReferenceProfile::parse(profile.name()).unwrap(), profile);
            assert!(profile.validate_selection(false, false).is_err());
            assert!(profile.validate_selection(true, true).is_err());
            assert_eq!(profile.validate_selection(true, false).is_ok(), profile.compiled());
        }
        for value in ["", "32k", "context128", "CONTEXT32", "growing", "canonical8k"] {
            assert!(NativeLongReferenceProfile::parse(value).is_err());
        }
    }

    #[test]
    fn long_reference_bounds_preserve_runtime_headroom_and_profile_separation() {
        for (profile, lower, upper) in [
            (NativeLongReferenceProfile::Context32, 32769, 35840),
            (NativeLongReferenceProfile::Context64, 65537, 69632),
            (NativeLongReferenceProfile::Near192, 194561, 196474),
        ] {
            assert_eq!(profile.bounds(), (lower, upper));
            assert_eq!(profile.bootstrap_tokens(), lower - 1);
            for count in [lower - 1, lower, upper, upper + 1] {
                let raw = serde_json::to_vec(&long_fixture(profile, count)).unwrap();
                let valid = profile.compiled() && (lower..=upper).contains(&count);
                assert_eq!(parse_input_context(&raw, &digest(&raw), false, Some(profile)).is_ok(), valid);
                assert!(parse_input_context(&raw, &digest(&raw), true, Some(profile)).is_err());
                assert!(parse_input_profile(&raw, &digest(&raw), false).is_err());
                assert!(parse_input_profile(&raw, &digest(&raw), true).is_err());
            }
        }
        // Existing runtime reserves128+B-2 positions for128 visible outputs.
        assert_eq!(NativeLongReferenceProfile::Near192.bounds().1 + 128 + 8 - 2, 196608);
    }

    #[test]
    fn long_reference_cross_context_schema_and_legacy_aliases_reject() {
        for selected in [NativeLongReferenceProfile::Context32, NativeLongReferenceProfile::Context64,
                         NativeLongReferenceProfile::Near192] {
            let mut value = long_fixture(selected, selected.bounds().0);
            for schema in ["qwen35-exact-history-reference-input-v1",
                           "qwen35-growing-exact-history-reference-input-v1",
                           NativeLongReferenceProfile::Context32.schema(),
                           NativeLongReferenceProfile::Context64.schema(),
                           NativeLongReferenceProfile::Near192.schema()] {
                if schema == selected.schema() { continue; }
                value["schema"] = json!(schema);
                let raw = serde_json::to_vec(&value).unwrap();
                assert!(parse_input_context(&raw, &digest(&raw), false, Some(selected)).is_err());
            }
        }
    }

    #[test]
    fn long_reference_hash_typed_tokens_unknown_and_duplicate_fields_fail_closed() {
        let profile = NativeLongReferenceProfile::Context32;
        let base = long_fixture(profile, profile.bounds().0);
        for bad in [json!(true), json!(-1), json!(248320), json!(1.5), json!("17")] {
            let mut changed = base.clone();
            changed["token_ids"][0] = bad;
            let raw = serde_json::to_vec(&changed).unwrap();
            assert!(parse_input_context(&raw, &digest(&raw), false, Some(profile)).is_err());
        }
        for (key, bad) in [("consumed_input_digest", json!("wrong")), ("qualified", json!(true))] {
            let mut changed = base.clone(); changed[key] = bad;
            let raw = serde_json::to_vec(&changed).unwrap();
            assert!(parse_input_context(&raw, &digest(&raw), false, Some(profile)).is_err());
        }
        let raw = serde_json::to_vec(&base).unwrap();
        assert!(parse_input_context(&raw, &"0".repeat(64), false, Some(profile)).is_err());
        let mut duplicate = b"{\"schema\":\"duplicate\",".to_vec();
        duplicate.extend_from_slice(&raw[1..]);
        assert!(parse_input_context(&duplicate, &digest(&duplicate), false, Some(profile)).is_err());
    }

    #[test]
    fn long_reference_larger_json_does_not_widen_legacy_byte_bound() {
        let profile = NativeLongReferenceProfile::Context32;
        let mut raw = serde_json::to_vec(&long_fixture(profile, profile.bounds().0)).unwrap();
        raw.resize(MAX_INPUT_BYTES as usize + 1, b' ');
        assert_eq!(parse_input_context(&raw, &digest(&raw), false, Some(profile)).is_ok(), profile.compiled());
        assert!(parse_input_profile(&raw, &digest(&raw), false).is_err());
        assert!(parse_input_profile(&raw, &digest(&raw), true).is_err());
        raw.resize(MAX_LONG_INPUT_BYTES as usize + 1, b' ');
        assert!(parse_input_context(&raw, &digest(&raw), false, Some(profile)).is_err());
    }
}
