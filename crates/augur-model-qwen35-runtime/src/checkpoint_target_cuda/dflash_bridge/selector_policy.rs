//! Parse once at owner creation. Experimental scales never change mid-session.

pub(super) fn parse_early_scales(
    path: &str,
    uniform_scale: f32,
    value: Option<&str>,
) -> Result<[f32; 3], &'static str> {
    if path != "position" {
        return if value.is_some() {
            Err("DFlash early scales require selector path=position")
        } else {
            Ok([1.0; 3])
        };
    }
    if uniform_scale != 1.0 {
        return Err("DFlash position selector requires uniform scale=1; use explicit early scales");
    }
    let text = value.ok_or("DFlash position selector requires three explicit early scales")?;
    let mut parts = text.split(',');
    let mut scales = [1.0f32; 3];
    for scale in &mut scales {
        let part = parts.next().ok_or("DFlash position selector needs exactly three scales")?;
        *scale = part.trim().parse().map_err(|_| "invalid DFlash early scale")?;
        if !scale.is_finite() || !(0.0..=2.0).contains(scale) {
            return Err("DFlash early scales must be finite and within [0,2]");
        }
    }
    if parts.next().is_some() {
        return Err("DFlash position selector needs exactly three scales");
    }
    Ok(scales)
}

/// HTTP experiments must opt in separately from native selector experiments.
/// Position scaling is compatible with the global pinned gate; the legacy
/// within-block Q8/FP32 phase selector still requires its original greedy path.
pub(super) fn admit_http(path: &str, uniform_scale: f32, global_gate: bool,
    position_opt_in: Option<&str>) -> Result<(), &'static str> {
    match (path, position_opt_in) {
        ("greedy", None | Some("0")) if uniform_scale == 1.0 => Ok(()),
        ("position", Some("1")) if uniform_scale == 1.0 && global_gate => Ok(()),
        _ => Err("HTTP position selector requires explicit opt-in, global pinned gate and scale=1"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn http_position_requires_global_gate_and_independent_explicit_opt_in() {
        for global in [false, true] {
            assert!(admit_http("greedy", 1.0, global, None).is_ok());
            assert!(admit_http("greedy", 1.0, global, Some("0")).is_ok());
            assert!(admit_http("position", 1.0, global, None).is_err());
            assert!(admit_http("position", 1.0, global, Some("0")).is_err());
            assert!(admit_http("greedy", 1.0, global, Some("1")).is_err());
        }
        assert!(admit_http("position", 1.0, true, Some("1")).is_ok());
        assert!(admit_http("position", 1.0, false, Some("1")).is_err());
        for path in ["dp", "early", "tail075", "rankcomp", "unknown"] {
            assert!(admit_http(path, 1.0, true, Some("1")).is_err());
        }
        for scale in [0.45, f32::NAN, f32::INFINITY] {
            assert!(admit_http("position", scale, true, Some("1")).is_err());
        }
        assert!(admit_http("position", 1.0, true, Some("yes")).is_err());
    }

    #[test]
    fn position_scales_are_explicit_and_do_not_change_other_policies() {
        for path in ["greedy", "early", "dp", "tail075"] {
            assert_eq!(parse_early_scales(path, 0.5, None).unwrap(), [1.0; 3]);
            assert!(parse_early_scales(path, 1.0, Some("0.25,0.65,0.60")).is_err());
        }
        assert_eq!(parse_early_scales("position", 1.0, Some("0.25, 0.65, 0.60")).unwrap(), [0.25, 0.65, 0.6]);
        assert_eq!(parse_early_scales("position", 1.0, Some("0,1,2")).unwrap(), [0.0, 1.0, 2.0]);
    }

    #[test]
    fn invalid_or_ambiguous_scales_fail_before_owner_allocation() {
        assert!(parse_early_scales("position", 1.0, None).is_err());
        for scale in [0.5, f32::NAN, f32::INFINITY] {
            assert!(parse_early_scales("position", scale, Some("1,1,1")).is_err());
        }
        for text in ["", "1", "1,1", "1,1,1,1", "1,,1", "1,1,", "NaN,1,1", "1,inf,1", "1,1,-inf", "-0.1,1,1", "1,2.01,1", "1,1,junk"] {
            assert!(parse_early_scales("position", 1.0, Some(text)).is_err(), "{text}");
        }
    }
}
