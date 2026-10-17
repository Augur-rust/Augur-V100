//! Opt-in admission only. Never changes arithmetic, state ownership or acceptance.
use super::*;
use std::ffi::OsStr;

// This scheduling-only experiment is admitted for live bounded resident
// generation, not scalar/reference audits or persistent input-only owners.
pub(super) fn live_prefill_batch_scope(
    growing: bool,
    full: bool,
    scalar: bool,
    output_ceiling: Option<usize>,
) -> bool {
    growing && full && !scalar && output_ceiling.is_some_and(|n| (1..=1024).contains(&n))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Mode {
    State,
    Client,
    Resident,
}

pub(super) fn parse(value: Option<&OsStr>, compiled: bool) -> Result<Option<Mode>> {
    match value {
        None => Ok(None),
        Some(v) if compiled && v == "state" => Ok(Some(Mode::State)),
        Some(v) if compiled && v == "client" => Ok(Some(Mode::Client)),
        Some(v)
            if compiled && cfg!(feature = "qwen35-resident-text-service") && v == "resident" =>
        {
            Ok(Some(Mode::Resident))
        }
        _ => Err(invalid(
            "top16 validation requires feature and exact state/client mode",
        )),
    }
}

#[derive(Clone, Copy)]
pub(super) struct Admission {
    pub frozen_request_shape: bool,
    pub other_diagnostic: bool,
    pub stage_profile: bool,
    pub shared_weights: bool,
    pub target_capture: bool,
    pub private_capture: bool,
    pub state_audit: bool,
    pub sink: bool,
}

impl Mode {
    /// A distinct consumer contract: retained input without any publication
    /// sink. Do not lie about sink presence to pass generated-request admission.
    pub(super) fn validate_input_prefix(self, a: Admission) -> Result<()> {
        if self != Self::Resident
            || !a.frozen_request_shape
            || !a.shared_weights
            || a.other_diagnostic
            || a.stage_profile
            || a.target_capture
            || a.private_capture
            || a.state_audit
            || a.sink
        {
            return Err(invalid(
                "input-prefix resident/no-publication admission mismatch",
            ));
        }
        Ok(())
    }

    pub(super) fn validate(self, a: Admission) -> Result<()> {
        let common = a.frozen_request_shape
            && !a.other_diagnostic
            && !a.stage_profile
            && (a.shared_weights == (self == Self::Resident));
        let specific = match self {
            Self::State => a.target_capture && a.private_capture && a.state_audit && !a.sink,
            Self::Client | Self::Resident => {
                !a.target_capture && !a.private_capture && !a.state_audit && a.sink
            }
        };
        if !common || !specific {
            return Err(invalid(
                "top16 state/client admission mismatch before allocation",
            ));
        }
        Ok(())
    }

    pub(super) fn validate_prompt(self, tokens: usize) -> Result<()> {
        let admitted = match self {
            Self::State => tokens == 8192,
            // This allows the declared ladder, not an automatic launch or qualification.
            Self::Client | Self::Resident => [8192, 32768, 65536, 194560].contains(&tokens),
        };
        if !admitted {
            return Err(invalid("top16 validation prompt outside declared ladder"));
        }
        Ok(())
    }

    pub(super) fn receipt(self, batch: bool) -> Value {
        json!({"schema":"qwen35-top16-validation-admission-v1",
            "mode":match self {Self::State=>"state",Self::Client=>"client",Self::Resident=>"resident"},
            "draft_top16_batch":batch,"numerical_or_acceptance_policy_changed":false,
            "shared_weight_service":self == Self::Resident,"product_qualification":false,
            "client_performance_qualified":false})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn live_prefill_batch_scope_preserves_audit_and_budget_boundaries() {
        for n in [1, 64, 1024] {
            assert!(live_prefill_batch_scope(true, true, false, Some(n)));
        }
        for n in [None, Some(0), Some(1025)] {
            assert!(!live_prefill_batch_scope(true, true, false, n));
        }
        assert!(!live_prefill_batch_scope(false, true, false, Some(64)));
        assert!(!live_prefill_batch_scope(true, false, false, Some(64)));
        assert!(!live_prefill_batch_scope(true, true, true, Some(64)));
    }
    fn state() -> Admission {
        Admission {
            frozen_request_shape: true,
            other_diagnostic: false,
            stage_profile: false,
            shared_weights: false,
            target_capture: true,
            private_capture: true,
            state_audit: true,
            sink: false,
        }
    }
    fn client() -> Admission {
        Admission {
            target_capture: false,
            private_capture: false,
            state_audit: false,
            sink: true,
            ..state()
        }
    }
    #[test]
    fn feature_off_and_exact_mode_parse() {
        assert_eq!(parse(None, false).unwrap(), None);
        assert_eq!(parse(None, true).unwrap(), None);
        for (v, mode) in [("state", Mode::State), ("client", Mode::Client)] {
            assert_eq!(parse(Some(OsStr::new(v)), true).unwrap(), Some(mode));
            assert!(parse(Some(OsStr::new(v)), false).is_err());
        }
        for v in ["", "1", "STATE", "client "] {
            assert!(parse(Some(OsStr::new(v)), true).is_err());
        }
    }
    #[test]
    fn exact_state_and_client_are_admitted() {
        assert!(Mode::State.validate(state()).is_ok());
        assert!(Mode::Client.validate(client()).is_ok());
        assert!(Mode::State.validate(client()).is_err());
        assert!(Mode::Client.validate(state()).is_err());
    }

    #[test]
    fn input_prefix_requires_resident_state_without_a_publication_sink() {
        let input = Admission {
            shared_weights: true,
            sink: false,
            ..client()
        };
        assert!(Mode::Resident.validate_input_prefix(input).is_ok());
        assert!(Mode::Resident.validate(input).is_err());
        assert!(Mode::Client.validate_input_prefix(input).is_err());
        assert!(Mode::State.validate_input_prefix(input).is_err());
        assert!(Mode::Resident
            .validate_input_prefix(Admission {
                sink: true,
                ..input
            })
            .is_err());
        for forbidden in [
            Admission {
                frozen_request_shape: false,
                ..input
            },
            Admission {
                shared_weights: false,
                ..input
            },
            Admission {
                other_diagnostic: true,
                ..input
            },
            Admission {
                stage_profile: true,
                ..input
            },
            Admission {
                target_capture: true,
                ..input
            },
            Admission {
                private_capture: true,
                ..input
            },
            Admission {
                state_audit: true,
                ..input
            },
        ] {
            assert!(Mode::Resident.validate_input_prefix(forbidden).is_err());
        }
    }
    #[test]
    fn resident_mode_has_separate_feature_and_requires_shared_weights() {
        let parsed = parse(Some(OsStr::new("resident")), true);
        assert_eq!(
            parsed.is_ok(),
            cfg!(feature = "qwen35-resident-text-service")
        );
        assert!(parse(Some(OsStr::new("resident")), false).is_err());
        assert!(Mode::Resident.validate(client()).is_err());
        let resident = Admission {
            shared_weights: true,
            ..client()
        };
        assert!(Mode::Resident.validate(resident).is_ok());
        assert!(Mode::Client.validate(resident).is_err());
        assert!(Mode::Resident
            .validate(Admission {
                private_capture: true,
                ..resident
            })
            .is_err());
        assert!(Mode::Resident
            .validate(Admission {
                sink: false,
                ..resident
            })
            .is_err());
        let report = Mode::Resident.receipt(true);
        assert_eq!(report["shared_weight_service"], true);
        assert_eq!(report["product_qualification"], false);
    }
    #[test]
    fn state_requires_both_captures_and_no_sink() {
        for a in [
            Admission {
                target_capture: false,
                ..state()
            },
            Admission {
                private_capture: false,
                ..state()
            },
            Admission {
                state_audit: false,
                ..state()
            },
            Admission {
                sink: true,
                ..state()
            },
        ] {
            assert!(Mode::State.validate(a).is_err());
        }
    }
    #[test]
    fn client_rejects_capture_audit_and_missing_sink() {
        for a in [
            Admission {
                target_capture: true,
                ..client()
            },
            Admission {
                private_capture: true,
                ..client()
            },
            Admission {
                state_audit: true,
                ..client()
            },
            Admission {
                sink: false,
                ..client()
            },
        ] {
            assert!(Mode::Client.validate(a).is_err());
        }
    }
    #[test]
    fn mixed_profile_shared_and_changed_axes_rejected() {
        for (mode, a) in [(Mode::State, state()), (Mode::Client, client())] {
            for bad in [
                Admission {
                    frozen_request_shape: false,
                    ..a
                },
                Admission {
                    other_diagnostic: true,
                    ..a
                },
                Admission {
                    stage_profile: true,
                    ..a
                },
                Admission {
                    shared_weights: true,
                    ..a
                },
            ] {
                assert!(mode.validate(bad).is_err());
            }
        }
    }
    #[test]
    fn state_and_client_occupied_lengths_not_capacity_labels() {
        assert!(Mode::State.validate_prompt(8192).is_ok());
        for n in [8192, 32768, 65536, 194560] {
            assert!(Mode::Client.validate_prompt(n).is_ok());
        }
        for n in [2048, 8191, 8193, 196352, 196608, 262144] {
            assert!(Mode::State.validate_prompt(n).is_err());
            assert!(Mode::Client.validate_prompt(n).is_err());
        }
        assert!(Mode::State.validate_prompt(194560).is_err());
    }
    #[test]
    fn receipt_cannot_grant_qualification() {
        for m in [Mode::State, Mode::Client] {
            for batch in [false, true] {
                let r = m.receipt(batch);
                assert_eq!(r["draft_top16_batch"], json!(batch));
                assert_eq!(r["product_qualification"], json!(false));
                assert_eq!(r["client_performance_qualified"], json!(false));
                assert_eq!(r["numerical_or_acceptance_policy_changed"], json!(false));
            }
        }
    }
}
