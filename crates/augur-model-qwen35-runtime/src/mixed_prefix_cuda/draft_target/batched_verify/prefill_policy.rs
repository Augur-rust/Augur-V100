//! One opt-in resident prefill recipe. Absence preserves every legacy switch.
//! Publication/default promotion is independent of selecting this recipe.
use super::*;
use std::ffi::OsString;

pub(in super::super) fn batched_head_rope_switch(
    value: Option<&std::ffi::OsStr>, compiled: bool, exact_input_recipe: bool,
) -> Result<bool> {
    match value {
        None => Ok(false),
        Some(value) if value == "1" && compiled && exact_input_recipe => Ok(true),
        _ => Err(invalid("batched head RoPE requires exact opt-in and the complete real-input B512 recipe")),
    }
}

#[cfg(test)]
#[test]
fn batched_head_rope_requires_exact_compiled_recipe_and_preserves_default() {
    use std::ffi::OsStr;
    for compiled in [false, true] {
        for recipe in [false, true] {
            assert!(!batched_head_rope_switch(None, compiled, recipe).unwrap());
            assert_eq!(batched_head_rope_switch(Some(OsStr::new("1")), compiled, recipe).is_ok(), compiled && recipe);
            for bad in ["", "0", "true", "01", "1 ", "auto"] {
                assert!(batched_head_rope_switch(Some(OsStr::new(bad)), compiled, recipe).is_err());
            }
        }
    }
    assert!(batched_head_rope_switch(Some(OsStr::new("1")), true, true).unwrap());
}

#[derive(Clone, Debug, Default)]
pub(in super::super) struct Switches {
    pub preset: Option<OsString>,
    pub rows: Option<OsString>,
    pub ffn: Option<OsString>,
    pub input: Option<OsString>,
    pub llama_f16_input: bool,
    pub llama_original_input: bool,
    pub llama_full_original: bool,
    pub llama_rotated_kv: bool,
    pub llama_bf16_scalars: bool,
    pub llama_history_full: bool,
    pub llama_global_chunks: bool,
    pub llama_exact_input_rows: bool,
    pub history: Option<OsString>,
    pub row_reuse: Option<OsString>,
    pub tensor_down: Option<OsString>,
    pub compensated: Option<OsString>,
    score_mode: Option<OsString>,
    attribution: Option<OsString>,
}

#[derive(Clone, Copy, Debug)]
pub(in super::super) struct Scope {
    pub canonical_session: bool,
    pub resident: bool,
    pub exact_batched_target: bool,
    pub vision: bool,
    pub capacity: usize,
    pub history_rows: usize,
}

impl Switches {
    pub(in super::super) fn from_env() -> Self {
        Self {
            preset: std::env::var_os("AUGUR_QWEN35_PREFILL_POLICY"),
            rows: std::env::var_os("AUGUR_QWEN35_CANONICAL_PREFILL_ROWS"),
            ffn: std::env::var_os("AUGUR_QWEN35_PREFILL_INTEGER_MMA"),
            input: std::env::var_os("AUGUR_QWEN35_PREFILL_INPUT_INTEGER"),
            llama_f16_input: false,
            llama_original_input: false,
            llama_full_original: false,
            llama_rotated_kv: false,
            llama_bf16_scalars: false,
            llama_history_full: false,
            llama_global_chunks: false,
            llama_exact_input_rows: false,
            history: std::env::var_os("AUGUR_QWEN35_HISTORY_WIDE_CAPTURE"),
            row_reuse: std::env::var_os("AUGUR_QWEN35_PROJECTION_ROW_REUSE"),
            tensor_down: std::env::var_os("AUGUR_QWEN35_PREFILL_FFN_DOWN_TENSOR"),
            compensated: std::env::var_os("AUGUR_QWEN35_PREFILL_FFN_DOWN_COMPENSATED"),
            score_mode: std::env::var_os("AUGUR_QWEN35_CANONICAL_PREFILL_SCORE_MODE"),
            attribution: std::env::var_os("AUGUR_QWEN35_CANONICAL_PREFIX_ATTRIBUTION"),
        }
    }

    pub(in super::super) fn resolve(mut self, scope: Scope) -> Result<Self> {
        let Some(preset) = self.preset.as_deref() else {
            return Ok(self);
        };
        if preset != "wide512-v1" && preset != "wide512-llama-q8-v1"
            && preset != "wide512-llama-original-v1" && preset != "wide512-llama-full-original-v1"
            && preset != "wide512-llama-dataflow-v1" && preset != "wide512-llama-bf16-gdn-v1"
            && preset != "wide512-llama-full-history-v1"
            && preset != "wide512-llama-global-boundary-audit-v1"
            && preset != "wide512-llama-exact-input-v1" {
            return Err(invalid("unknown AUGUR_QWEN35_PREFILL_POLICY"));
        }
        if [
            &self.rows,
            &self.ffn,
            &self.input,
            &self.history,
            &self.row_reuse,
            &self.tensor_down,
            &self.compensated,
            &self.score_mode,
            &self.attribution,
        ]
        .iter()
        .any(|value| value.is_some())
        {
            return Err(invalid(
                "prefill recipe cannot mix with individual prefill/row-reuse switches",
            ));
        }
        if !scope.canonical_session
            || !scope.resident
            || !scope.exact_batched_target
            || !scope.vision
            || scope.capacity != 196608
            || !(1..=2048).contains(&scope.history_rows)
        {
            return Err(invalid(
                "wide512 recipe requires the complete canonical resident text session",
            ));
        }
        self.llama_f16_input = preset == "wide512-llama-q8-v1";
        self.llama_original_input = preset == "wide512-llama-original-v1";
        self.llama_global_chunks = preset == "wide512-llama-global-boundary-audit-v1";
        self.llama_exact_input_rows = preset == "wide512-llama-exact-input-v1";
        self.llama_history_full = preset == "wide512-llama-full-history-v1"
            || self.llama_global_chunks || self.llama_exact_input_rows;
        self.llama_bf16_scalars = preset == "wide512-llama-bf16-gdn-v1" || self.llama_history_full;
        self.llama_rotated_kv = preset == "wide512-llama-dataflow-v1" || self.llama_bf16_scalars;
        self.llama_full_original = preset == "wide512-llama-full-original-v1" || self.llama_rotated_kv;
        self.rows = Some("512".into());
        self.ffn = Some("1".into());
        self.input = Some("1".into());
        self.history = Some("auto".into());
        self.row_reuse = Some("1".into());
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn exact_input_recipe_is_separate_from_boundary_audit_and_verifier() {
        for history_rows in [1, 7, 8, 9, 511, 512, 2048] {
            let mut s = scope(); s.history_rows = history_rows;
            let p = Switches { preset: Some("wide512-llama-exact-input-v1".into()),
                ..Default::default() }.resolve(s).unwrap();
            assert!(p.llama_exact_input_rows && p.llama_history_full
                && p.llama_bf16_scalars && p.llama_rotated_kv && p.llama_full_original);
            assert!(!p.llama_global_chunks);
            assert_eq!(p.rows.as_deref(), Some(std::ffi::OsStr::new("512")));
        }
        assert!(!Switches::default().resolve(scope()).unwrap().llama_exact_input_rows);
        let old = Switches { preset: Some("wide512-llama-global-boundary-audit-v1".into()),
            ..Default::default() }.resolve(scope()).unwrap();
        assert!(!old.llama_exact_input_rows && old.llama_global_chunks);
    }
    #[test]
    fn global_boundary_audit_is_explicit_and_inherits_history_math_only() {
        let p = Switches { preset: Some("wide512-llama-global-boundary-audit-v1".into()),
            ..Default::default() }.resolve(scope()).unwrap();
        assert!(p.llama_global_chunks && p.llama_history_full && p.llama_bf16_scalars
            && p.llama_rotated_kv && p.llama_full_original);
        for preset in [None, Some("wide512-v1"), Some("wide512-llama-bf16-gdn-v1"),
            Some("wide512-llama-full-history-v1")] {
            let old = Switches { preset: preset.map(Into::into), ..Default::default() }
                .resolve(scope()).unwrap();
            assert!(!old.llama_global_chunks);
        }
    }
    #[test]
    fn full_history_recipe_is_separate_and_preserves_all_older_recipes() {
        for name in [None, Some("wide512-v1"), Some("wide512-llama-q8-v1"),
            Some("wide512-llama-original-v1"), Some("wide512-llama-full-original-v1"),
            Some("wide512-llama-dataflow-v1"), Some("wide512-llama-bf16-gdn-v1")] {
            let p = Switches { preset: name.map(Into::into), ..Default::default() }
                .resolve(scope()).unwrap();
            assert!(!p.llama_history_full);
        }
        for rows in [1, 8, 97, 511, 2047, 2048] {
            let p = Switches { preset: Some("wide512-llama-full-history-v1".into()),
                ..Default::default() }.resolve(Scope { history_rows: rows, ..scope() }).unwrap();
            assert!(p.llama_history_full && p.llama_bf16_scalars
                && p.llama_full_original && p.llama_rotated_kv);
            assert!(!p.llama_original_input && !p.llama_f16_input);
            // Small/growing history remains on its existing path, without padding.
            assert_eq!(history_capture::requested(p.history.as_deref(), Some(512), true, rows).unwrap(), rows == 2048);
        }
    }
    #[test]
    fn full_history_recipe_rejects_scope_and_individual_overrides() {
        let recipe = || Switches { preset: Some("wide512-llama-full-history-v1".into()), ..Default::default() };
        for s in [Scope { capacity: 65536, ..scope() }, Scope { vision: false, ..scope() },
            Scope { resident: false, ..scope() }, Scope { exact_batched_target: false, ..scope() },
            Scope { canonical_session: false, ..scope() }, Scope { history_rows: 2049, ..scope() }] {
            assert!(recipe().resolve(s).is_err());
        }
        for index in 0..9 {
            let mut p = recipe();
            let fields = [&mut p.rows, &mut p.ffn, &mut p.input, &mut p.history,
                &mut p.row_reuse, &mut p.tensor_down, &mut p.compensated,
                &mut p.score_mode, &mut p.attribution];
            *fields.into_iter().nth(index).unwrap() = Some("1".into());
            assert!(p.resolve(scope()).is_err());
        }
    }
    #[test]
    fn bf16_scalar_recipe_is_separate_and_preserves_prior_recipes() {
        for name in [None, Some("wide512-v1"), Some("wide512-llama-dataflow-v1")] {
            let p = Switches { preset: name.map(Into::into), ..Default::default() }
                .resolve(scope()).unwrap();
            assert!(!p.llama_bf16_scalars);
        }
        let recipe = || Switches { preset: Some("wide512-llama-bf16-gdn-v1".into()), ..Default::default() };
        let p = recipe().resolve(scope()).unwrap();
        assert!(p.llama_bf16_scalars && p.llama_rotated_kv && p.llama_full_original);
        assert!(!p.llama_f16_input && !p.llama_original_input);
        assert!(recipe().resolve(Scope { capacity: 65536, ..scope() }).is_err());
        let mut conflict = recipe(); conflict.rows = Some("512".into());
        assert!(conflict.resolve(scope()).is_err());
    }

    use super::*;
    fn scope() -> Scope {
        Scope {
            canonical_session: true,
            resident: true,
            exact_batched_target: true,
            vision: true,
            capacity: 196608,
            history_rows: 2048,
        }
    }
    fn preset() -> Switches {
        Switches {
            preset: Some("wide512-v1".into()),
            ..Default::default()
        }
    }
    #[test]
    fn recipe_absence_preserves_default_and_explicit_b8_control() {
        let off = Switches::default().resolve(scope()).unwrap();
        assert!(off.rows.is_none() && off.ffn.is_none() && off.history.is_none());
        let control = Switches {
            rows: Some("8".into()),
            row_reuse: Some("1".into()),
            ..Default::default()
        }
        .resolve(scope())
        .unwrap();
        assert_eq!(control.rows.as_deref(), Some(std::ffi::OsStr::new("8")));
        assert!(control.ffn.is_none() && control.input.is_none() && control.history.is_none());
    }
    #[test]
    fn recipe_selects_identical_integer_math_and_only_full_history_capture() {
        for rows in [1, 7, 8, 511, 512, 2047, 2048] {
            let p = preset()
                .resolve(Scope {
                    history_rows: rows,
                    ..scope()
                })
                .unwrap();
            let ffn =
                prefill_integer_mma::requested(p.ffn.as_deref(), p.rows.as_deref(), true, false)
                    .unwrap();
            let input = prefill_input_integer::requested(p.input.as_deref(), ffn).unwrap();
            assert_eq!(ffn, Some(512));
            assert!(input);
            assert_eq!(
                history_capture::requested(p.history.as_deref(), ffn, input, rows).unwrap(),
                rows == 2048
            );
            assert!(projection_row_reuse_switch(p.row_reuse.as_deref(), true, true).unwrap());
            assert_eq!(prefill_width::select(512, 511).unwrap(), B);
        }
    }
    #[test]
    fn recipe_rejects_every_partial_or_conflicting_override() {
        for index in 0..9 {
            let mut p = preset();
            let fields = [
                &mut p.rows,
                &mut p.ffn,
                &mut p.input,
                &mut p.history,
                &mut p.row_reuse,
                &mut p.tensor_down,
                &mut p.compensated,
                &mut p.score_mode,
                &mut p.attribution,
            ];
            *fields.into_iter().nth(index).unwrap() = Some("1".into());
            assert!(p.resolve(scope()).is_err(), "conflict {index}");
        }
    }
    #[test]
    fn llama_recipe_is_explicit_and_preserves_integer_history_admission() {
        assert!(!Switches::default().resolve(scope()).unwrap().llama_f16_input);
        assert!(!preset().resolve(scope()).unwrap().llama_f16_input);
        let p = Switches {
            preset: Some("wide512-llama-q8-v1".into()), ..Default::default()
        }.resolve(scope()).unwrap();
        assert!(p.llama_f16_input);
        let ffn = prefill_integer_mma::requested(
            p.ffn.as_deref(), p.rows.as_deref(), true, false).unwrap();
        let input = prefill_input_integer::requested(p.input.as_deref(), ffn).unwrap();
        assert_eq!(ffn, Some(512));
        assert!(history_capture::requested(p.history.as_deref(), ffn, input, 2048).unwrap());
        assert!(Switches {preset: Some("wide512-llama-q8-v1".into()),
            input: Some("1".into()), ..Default::default()}.resolve(scope()).is_err());
        assert!(Switches {preset: Some("wide512-llama-q8-v1".into()),
            ..Default::default()}.resolve(Scope {capacity: 65536, ..scope()}).is_err());
    }
    #[test]
    fn original_input_recipe_is_separate_default_off_and_history_unchanged() {
        for name in [None, Some("wide512-v1"), Some("wide512-llama-q8-v1")] {
            let p = Switches {preset: name.map(Into::into), ..Default::default()}
                .resolve(scope()).unwrap();
            assert!(!p.llama_original_input);
        }
        let p = Switches {preset: Some("wide512-llama-original-v1".into()),
            ..Default::default()}.resolve(scope()).unwrap();
        assert!(p.llama_original_input && !p.llama_f16_input);
        let ffn = prefill_integer_mma::requested(
            p.ffn.as_deref(), p.rows.as_deref(), true, false).unwrap();
        let input = prefill_input_integer::requested(p.input.as_deref(), ffn).unwrap();
        assert_eq!(ffn, Some(512));
        assert!(history_capture::requested(p.history.as_deref(), ffn, input, 2048).unwrap());
        for s in [Scope {vision: false, ..scope()}, Scope {capacity: 65536, ..scope()}] {
            assert!(Switches {preset: Some("wide512-llama-original-v1".into()),
                ..Default::default()}.resolve(s).is_err());
        }
        assert!(Switches {preset: Some("wide512-llama-original-v1".into()),
            input: Some("1".into()), ..Default::default()}.resolve(scope()).is_err());
    }
    #[test]
    fn full_original_recipe_is_default_off_and_history_admission_unchanged() {
        for name in [None,Some("wide512-v1"),Some("wide512-llama-q8-v1"),Some("wide512-llama-original-v1")] {
            let p=Switches {preset:name.map(Into::into),..Default::default()}.resolve(scope()).unwrap();
            assert!(!p.llama_full_original);
        }
        let recipe=|| Switches {preset:Some("wide512-llama-full-original-v1".into()),..Default::default()};
        let p=recipe().resolve(scope()).unwrap();
        assert!(p.llama_full_original && !p.llama_original_input && !p.llama_f16_input);
        let ffn=prefill_integer_mma::requested(p.ffn.as_deref(),p.rows.as_deref(),true,false).unwrap();
        let input=prefill_input_integer::requested(p.input.as_deref(),ffn).unwrap();
        assert_eq!(ffn,Some(512));
        assert!(history_capture::requested(p.history.as_deref(),ffn,input,2048).unwrap());
        assert!(recipe().resolve(Scope {capacity:65536,..scope()}).is_err());
        assert!(recipe().resolve(Scope {vision:false,..scope()}).is_err());
        let mut bad=recipe();bad.input=Some("1".into());assert!(bad.resolve(scope()).is_err());
    }
    #[test]
    fn rotated_dataflow_recipe_binds_prefix_and_live_cache_without_changing_defaults() {
        for name in [None, Some("wide512-v1"), Some("wide512-llama-full-original-v1")] {
            let p = Switches { preset: name.map(Into::into), ..Default::default() }.resolve(scope()).unwrap();
            assert!(!p.llama_rotated_kv);
        }
        let recipe = || Switches { preset: Some("wide512-llama-dataflow-v1".into()), ..Default::default() };
        let selected = recipe().resolve(scope()).unwrap();
        assert!(selected.llama_rotated_kv && selected.llama_full_original);
        assert!(!selected.llama_original_input && !selected.llama_f16_input);
        assert_eq!(selected.rows.as_deref(), Some(std::ffi::OsStr::new("512")));
        assert!(recipe().resolve(Scope { capacity: 65536, ..scope() }).is_err());
        assert!(recipe().resolve(Scope { vision: false, ..scope() }).is_err());
        let mut mixed = recipe(); mixed.score_mode = Some("warp".into());
        assert!(mixed.resolve(scope()).is_err());
    }

    #[test]
    fn recipe_cannot_expand_diagnostic_geometry_or_authority() {
        for value in ["", "auto", "wide512", "wide512-v1 ", "WIDE512-V1"] {
            assert!(Switches {
                preset: Some(value.into()),
                ..Default::default()
            }
            .resolve(scope())
            .is_err());
        }
        for s in [
            Scope {
                canonical_session: false,
                ..scope()
            },
            Scope {
                resident: false,
                ..scope()
            },
            Scope {
                exact_batched_target: false,
                ..scope()
            },
            Scope {
                vision: false,
                ..scope()
            },
            Scope {
                capacity: 65536,
                ..scope()
            },
            Scope {
                history_rows: 0,
                ..scope()
            },
            Scope {
                history_rows: 2049,
                ..scope()
            },
        ] {
            assert!(preset().resolve(s).is_err());
        }
    }
}
