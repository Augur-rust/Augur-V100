//! Closed numerical identities, sealed to the installed capture rather than
//! accepting an arbitrary metadata schema or a policy label alone.
use super::*;
use crate::checkpoint_projection_cuda::CheckpointProjectionPolicy;
use augur_model_qwen35::CheckpointExpectedIdentity;
use serde_json::{json, Value};

pub(super) const B8_EXECUTION: &str = "nerkyor_original_nvfp4_canonical_f32_kv_b8_pending_v1";
pub(super) const CORE_B8_EXECUTION: &str = "nerkyor_original_nvfp4_core_f32_kv_b8_prefill_pending_v2";
pub(super) const B8_HEAD: &str = "llama_mmvf8_bf16_f32_v1";
fn b8_contract() -> Value {
    json!({"profile":B8_EXECUTION,"width":8,"projection":"retained_original_nvfp4_exact_f32_batch_v1",
        "gate":"bf16_f32_fma_cta256_tree_research_v1","activation_dtype":"F32","kv_dtype":"F32",
        "attention":"nerkyor_canonical_f64_warp_score_ordered_f32_softmax_split_rope_v1",
        "gdn":"checkpoint_contiguous3_f32_recorded_prefix_replay_v1","target_head":B8_HEAD,"draft_head":B8_HEAD,
        "scalar_boundary":"same_gate_f32_scalar_prefill_and_pending_consume_v1",
        "selection":"greedy_lowest_id_tie_nonfinite_fail_closed","final_token":"selected_unconsumed",
        "proposal":"owner_generation_capture_epoch_revision_cursor_sealed","publication":"research_only"})
}
fn core_b8_contract() -> Value {
    let mut contract=b8_contract();
    contract["profile"]=json!(CORE_B8_EXECUTION);
    contract["attention"]=json!("core_tu_batched_qk_norm_split_rope_scalar_fma_attention_v2");
    contract["prefill"]=json!("scalar_initial1_full_b8_commit_scalar_tail_v1");
    contract["scalar_boundary"]=json!("same_gate_f32_scalar_initial_tail_pending_consume_v1");
    contract
}
pub(super) fn require_b8_head() -> Result<()> {
    if crate::checkpoint_projection_cuda::shared_head_kernel()?.label() != B8_HEAD {
        return Err("B8 research identity pins the llama MMVF8 BF16/F32 head".into());
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ResearchPolicy {
    OriginalF32,
    GateF32Cta,
    GateF32CtaB8,
    CoreB8Prefill,
}
impl ResearchPolicy {
    pub(super) fn is_b8(self) -> bool { matches!(self,Self::GateF32CtaB8|Self::CoreB8Prefill) }
    pub(super) fn from_info(info: &Value) -> Result<Self> {
        if info["profile"] != "nerkyor_w4a16"
            || info["kv_dtype"] != "F32"
            || info["activation_dtype"] != "F32"
        {
            return Err("Nerkyor research requires this model and F32 activation/KV".into());
        }
        match info["policy"].as_str() {
            Some(label) if label == CheckpointProjectionPolicy::OriginalF32.label() => {
                if info.get("nerkyor_target_execution").is_some() {
                    return Err("OriginalF32 cannot claim a B8 execution identity".into());
                }
                Ok(Self::OriginalF32)
            }
            Some(label)
                if label == CheckpointProjectionPolicy::NerkyorBf16GateF32CtaResearch.label() =>
            {
                match info.get("nerkyor_target_execution") {
                    None => Ok(Self::GateF32Cta),
                    Some(v) if v.as_str() == Some(B8_EXECUTION) => Ok(Self::GateF32CtaB8),
                    Some(v) if v.as_str() == Some(CORE_B8_EXECUTION) => Ok(Self::CoreB8Prefill),
                    _ => Err("unknown Nerkyor execution identity".into()),
                }
            }
            _ => Err("unsupported Nerkyor DFlash research numerical policy".into()),
        }
    }
    pub(super) fn contract(self, checkpoint: &CheckpointExpectedIdentity, draft: &str) -> Value {
        // Preserve the original identity byte-for-byte. New arithmetic has its
        // own schema and source hash, never an alias for OriginalF32.
        let (schema, policy) = match self {
            Self::OriginalF32 => (
                "nerkyor-original-f32-dflash-research-identity-v1",
                CheckpointProjectionPolicy::OriginalF32,
            ),
            Self::GateF32CtaB8 => (
                "nerkyor-fp32-gate-b8-dflash-research-identity-v1",
                CheckpointProjectionPolicy::NerkyorBf16GateF32CtaResearch,
            ),
            Self::CoreB8Prefill => (
                "nerkyor-fp32-gate-core-b8-prefill-dflash-research-identity-v2",
                CheckpointProjectionPolicy::NerkyorBf16GateF32CtaResearch,
            ),
            Self::GateF32Cta => (
                "nerkyor-fp32-gate-dflash-research-identity-v1",
                CheckpointProjectionPolicy::NerkyorBf16GateF32CtaResearch,
            ),
        };
        let mut contract = json!({
            "schema":schema, "profile":"nerkyor_w4a16", "policy":policy.label(),
            "kv_dtype":"F32", "activation_dtype":"F32",
            "authenticated_checkpoint":checkpoint, "draft_file_sha256":draft,
            "product_admitted":false, "candidate_publication":false
        });
        if matches!(self, Self::GateF32Cta | Self::GateF32CtaB8 | Self::CoreB8Prefill) {
            contract["gate_numeric_contract"] = json!("bf16_f32_fma_cta256_tree_research_v1");
        }
        if self == Self::GateF32CtaB8 {
            contract["target_execution"] = b8_contract();
        }
        if self == Self::CoreB8Prefill { contract["target_execution"] = core_b8_contract(); }
        contract
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ResearchBinding {
    pub(super) policy: ResearchPolicy,
    pub(super) source_sha256: [u8; 32],
    pub(super) draft_sha256: String,
}
impl ResearchBinding {
    pub(super) fn read(target: &CheckpointCudaTarget) -> Result<Self> {
        let policy = ResearchPolicy::from_info(&target.info)?;
        let metadata = &target.info["dflash_feature_capture"];
        let contract = &metadata["source_identity_contract"];
        let draft = contract["draft_file_sha256"]
            .as_str()
            .ok_or("missing research draft identity")?;
        if draft.len() != 64
            || !draft
                .bytes()
                .all(|v| v.is_ascii_digit() || (b'a'..=b'f').contains(&v))
        {
            return Err("invalid research draft SHA256".into());
        }
        let expected = policy.contract(target._artifact.identity(), draft);
        let source_sha256: [u8; 32] = Sha256::digest(serde_json::to_vec(&expected)?).into();
        let capture = target
            .dflash_capture
            .as_ref()
            .ok_or("explicit Nerkyor research capture is not enabled")?;
        if *contract != expected
            || metadata["source_identity_sha256"]
                != format!("{:x}", Sha256::digest(serde_json::to_vec(&expected)?))
            || capture.source_identity_digest() != source_sha256
        {
            return Err(
                "Nerkyor capture numerical identity does not match its sealed owner".into(),
            );
        }
        Ok(Self {
            policy,
            source_sha256,
            draft_sha256: draft.to_owned(),
        })
    }
    pub(super) fn validate_weights(
        &self,
        target: &CheckpointCudaTarget,
        weights: &DFlashQ8CudaWeights,
    ) -> Result<()> {
        validate_draft(target, weights)?;
        if self.draft_sha256 != weights.file_sha256() {
            return Err("mismatched Nerkyor research draft".into());
        }
        Ok(())
    }
}
pub(super) fn validate_draft(
    target: &CheckpointCudaTarget,
    weights: &DFlashQ8CudaWeights,
) -> Result<()> {
    if weights.device_ordinal() != target.stream.device_ordinal()
        || weights.selector_gate_f32_sha256().is_some()
        || weights.selector_codebooks_bf16_sha256().is_some()
    {
        return Err("Nerkyor research requires an uncalibrated same-device draft".into());
    }
    weights.config().validate_exact_profile()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn checkpoint() -> CheckpointExpectedIdentity {
        CheckpointExpectedIdentity {
            config_sha256: "1".repeat(64),
            index_sha256: "2".repeat(64),
            shards: Default::default(),
        }
    }
    #[test]
    fn original_contract_is_compatible_and_gate_identity_is_distinct() {
        let checkpoint = checkpoint();
        let draft = "3".repeat(64);
        let original = ResearchPolicy::OriginalF32.contract(&checkpoint, &draft);
        let previous = json!({"schema":"nerkyor-original-f32-dflash-research-identity-v1",
            "profile":"nerkyor_w4a16","policy":"checkpoint_original_f32_scalar_v1",
            "kv_dtype":"F32","activation_dtype":"F32","authenticated_checkpoint":checkpoint,
            "draft_file_sha256":draft,"product_admitted":false,"candidate_publication":false});
        assert_eq!(
            serde_json::to_vec(&original).unwrap(),
            serde_json::to_vec(&previous).unwrap()
        );
        let gate = ResearchPolicy::GateF32Cta.contract(&checkpoint, &draft);
        assert_ne!(
            Sha256::digest(serde_json::to_vec(&original).unwrap()),
            Sha256::digest(serde_json::to_vec(&gate).unwrap())
        );
        assert_eq!(
            gate["gate_numeric_contract"],
            "bf16_f32_fma_cta256_tree_research_v1"
        );
        assert_eq!(gate["product_admitted"], false);
        assert_eq!(gate["candidate_publication"], false);
    }
    #[test]
    fn closed_policy_allowlist_rejects_profile_dtype_and_arbitrary_label() {
        for policy in [
            CheckpointProjectionPolicy::OriginalF32,
            CheckpointProjectionPolicy::NerkyorBf16GateF32CtaResearch,
        ] {
            let info = json!({"profile":"nerkyor_w4a16","policy":policy.label(),"kv_dtype":"F32","activation_dtype":"F32"});
            assert!(ResearchPolicy::from_info(&info).is_ok());
            for (key, value) in [
                ("profile", "quasar_nvfp4"),
                ("kv_dtype", "Q8_0"),
                ("activation_dtype", "BF16"),
                ("policy", "arbitrary-research-policy"),
                ("policy", CheckpointProjectionPolicy::NerkyorQpnWorkspaceShapeF64Research.label()),
                ("policy", CheckpointProjectionPolicy::NerkyorSourceSplit8F64Research.label()),
            ] {
                let mut invalid = info.clone();
                invalid[key] = json!(value);
                assert!(ResearchPolicy::from_info(&invalid).is_err(), "{key}");
            }
        }
    }
    #[test]
    fn binding_includes_policy_checkpoint_draft_and_gate_reduction() {
        let checkpoint = checkpoint();
        let draft = "3".repeat(64);
        let contract = ResearchPolicy::GateF32Cta.contract(&checkpoint, &draft);
        let hash =
            |c: &Value| -> [u8; 32] { Sha256::digest(serde_json::to_vec(c).unwrap()).into() };
        for key in [
            "policy",
            "schema",
            "draft_file_sha256",
            "gate_numeric_contract",
            "authenticated_checkpoint",
        ] {
            let mut changed = contract.clone();
            changed[key] = json!("foreign");
            assert_ne!(hash(&contract), hash(&changed), "{key}");
        }
    }
}

#[cfg(test)]
mod b8_tests {
    use super::*;
    #[test]
    fn b8_profile_is_closed_and_never_aliases_scalar_contracts() {
        let checkpoint: CheckpointExpectedIdentity = CheckpointExpectedIdentity {
            config_sha256: "1".repeat(64),
            index_sha256: "2".repeat(64),
            shards: Default::default(),
        };
        let draft = "4".repeat(64);
        let digest = |p: ResearchPolicy| {
            Sha256::digest(serde_json::to_vec(&p.contract(&checkpoint, &draft)).unwrap())
        };
        assert_ne!(
            digest(ResearchPolicy::GateF32CtaB8),
            digest(ResearchPolicy::GateF32Cta)
        );
        assert_ne!(
            digest(ResearchPolicy::GateF32CtaB8),
            digest(ResearchPolicy::OriginalF32)
        );
        let mut info = json!({"profile":"nerkyor_w4a16","policy":CheckpointProjectionPolicy::NerkyorBf16GateF32CtaResearch.label(),"kv_dtype":"F32","activation_dtype":"F32","nerkyor_target_execution":B8_EXECUTION});
        assert_eq!(
            ResearchPolicy::from_info(&info).unwrap(),
            ResearchPolicy::GateF32CtaB8
        );
        for v in [json!(null), json!("quasar_b8"), json!("unknown")] {
            info["nerkyor_target_execution"] = v;
            assert!(ResearchPolicy::from_info(&info).is_err());
        }
        info["nerkyor_target_execution"] = json!(B8_EXECUTION);
        info["policy"] = json!(CheckpointProjectionPolicy::OriginalF32.label());
        assert!(ResearchPolicy::from_info(&info).is_err());
    }
}


#[cfg(test)]
mod core_prefill_tests {
    use super::*;
    #[test]
    fn core_prefill_seal_cannot_alias_any_legacy_numerical_family() {
        let checkpoint=CheckpointExpectedIdentity { config_sha256:"1".repeat(64),index_sha256:"2".repeat(64),shards:Default::default() };
        let contract=ResearchPolicy::CoreB8Prefill.contract(&checkpoint,&"3".repeat(64));
        let digest=Sha256::digest(serde_json::to_vec(&contract).unwrap());
        for policy in [ResearchPolicy::OriginalF32,ResearchPolicy::GateF32Cta,ResearchPolicy::GateF32CtaB8] {
            assert_ne!(digest,Sha256::digest(serde_json::to_vec(&policy.contract(&checkpoint,&"3".repeat(64))).unwrap()));
        }
        assert_eq!(contract["target_execution"]["prefill"],"scalar_initial1_full_b8_commit_scalar_tail_v1");
        assert_eq!(contract["target_execution"]["profile"],CORE_B8_EXECUTION);
        let info=json!({"profile":"nerkyor_w4a16","policy":CheckpointProjectionPolicy::NerkyorBf16GateF32CtaResearch.label(),
            "kv_dtype":"F32","activation_dtype":"F32","nerkyor_target_execution":CORE_B8_EXECUTION});
        assert_eq!(ResearchPolicy::from_info(&info).unwrap(),ResearchPolicy::CoreB8Prefill);
        assert!(ResearchPolicy::CoreB8Prefill.is_b8());
        assert_eq!(ResearchPolicy::GateF32CtaB8.contract(&checkpoint,&"3".repeat(64))["target_execution"]["scalar_boundary"],
                   "same_gate_f32_scalar_prefill_and_pending_consume_v1");
    }
}
