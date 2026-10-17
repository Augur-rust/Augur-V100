//! Bind verification to the captured target's execution profile, not a guess
//! from its model name. Rotated KV must never enter the legacy B1 decoder.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TargetExecution {
    LegacyQpn,
    AugurRotatedQ8,
}

impl TargetExecution {
    fn from_contract(contract: &Value) -> Result<Self> {
        match contract["schema"].as_str() {
            Some("quasar-dflash-source-identity-v1") if contract["profile"] == "quasar_nvfp4" => {
                Ok(Self::LegacyQpn)
            }
            Some("quasar-quasar-speed-first-feature-identity-v1")
                if contract["numerical_profile"] == crate::checkpoint_target_cuda::augur_prefill::PROFILE => {
                Ok(Self::AugurRotatedQ8)
            }
            _ => Err("DFlash verifier requires a known exact target capture profile".into()),
        }
    }

    pub(super) fn for_target(target: &CheckpointCudaTarget) -> Result<Self> {
        let mode = Self::from_contract(&target.info["dflash_feature_capture"]["source_identity_contract"])?;
        // An empty reset owner has no live rotated KV yet; the capture contract
        // survives reset and still selects the future decoder. Residency-only
        // builds may bind this profile without compiling the actual verifier.
        if mode == Self::LegacyQpn && target.augur_rotated_kv {
            return Err("legacy capture cannot verify a rotated-KV target".into());
        }
        #[cfg(feature = "qwen35-augur-http")]
        if mode == Self::LegacyQpn && target.augur_decode.is_some() {
            return Err("legacy capture cannot adopt the rotated decoder".into());
        }
        Ok(mode)
    }

    pub(super) fn label(self) -> &'static str {
        match self {
            Self::LegacyQpn => "legacy_augur_qpn_canonical_B1",
            Self::AugurRotatedQ8 => "augur_rotated_q8_canonical_B1",
        }
    }

    pub(super) fn consume_selected(self, target: &mut CheckpointCudaTarget) -> Result<()> {
        if Self::for_target(target)? != self {
            return Err("DFlash target execution profile changed".into());
        }
        match self {
            Self::LegacyQpn => target.consume_selected(),
            Self::AugurRotatedQ8 => {
                #[cfg(feature = "qwen35-augur-http")]
                { target.consume_selected_augur() }
                #[cfg(not(feature = "qwen35-augur-http"))]
                { Err("rotated DFlash verification requires the Augur HTTP decoder feature".into()) }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_and_rotated_capture_contracts_select_distinct_verifiers() {
        assert_eq!(TargetExecution::from_contract(&json!({
            "schema":"quasar-dflash-source-identity-v1","profile":"quasar_nvfp4"
        })).unwrap(), TargetExecution::LegacyQpn);
        assert_eq!(TargetExecution::from_contract(&json!({
            "schema":"quasar-quasar-speed-first-feature-identity-v1",
            "numerical_profile":crate::checkpoint_target_cuda::augur_prefill::PROFILE
        })).unwrap(), TargetExecution::AugurRotatedQ8);
        assert_ne!(TargetExecution::LegacyQpn.label(), TargetExecution::AugurRotatedQ8.label());
    }

    #[test]
    fn model_name_alone_unknown_schema_or_wrong_math_profile_never_falls_back() {
        for contract in [Value::Null, json!({"profile":"quasar_nvfp4"}),
            json!({"schema":"quasar-dflash-source-identity-v1","profile":"legacy_gguf"}),
            json!({"schema":"quasar-quasar-speed-first-feature-identity-v1","numerical_profile":"unknown"}),
            json!({"schema":"unknown","numerical_profile":crate::checkpoint_target_cuda::augur_prefill::PROFILE})] {
            assert!(TargetExecution::from_contract(&contract).is_err());
        }
    }
}
