fn finite_value(code: u8) -> Option<f32> {
    let exponent = (code >> 3) & 15;
    let mantissa = code & 7;
    if exponent == 15 && mantissa == 7 {
        return None;
    }
    let magnitude = if exponent == 0 {
        (f32::from(mantissa) / 8.0) * 2.0_f32.powi(-6)
    } else {
        (1.0 + f32::from(mantissa) / 8.0) * 2.0_f32.powi(i32::from(exponent) - 7)
    };
    Some(if code & 128 == 0 {
        magnitude
    } else {
        -magnitude
    })
}

fn exhaustive_nearest(value: f32) -> f32 {
    let mut best = 0.0_f32;
    let mut distance = value.abs();
    for code in 0..=u8::MAX {
        let Some(candidate) = finite_value(code) else {
            continue;
        };
        let candidate_distance = (value - candidate).abs();
        if candidate_distance < distance || (candidate_distance == distance && candidate < best) {
            distance = candidate_distance;
            best = candidate;
        }
    }
    best
}

fn positive_finite_value(code: u8) -> f32 {
    finite_value(code).expect("codes 0 through 126 are finite and positive")
}

fn fast_exact_nearest(value: f32) -> f32 {
    if value.is_nan() {
        return 0.0;
    }
    let magnitude = value.abs();
    let mut lower_code = 0_u8;
    let mut upper_code = 126_u8;
    while lower_code < upper_code {
        let middle = lower_code + (upper_code - lower_code).div_ceil(2);
        if positive_finite_value(middle) <= magnitude {
            lower_code = middle;
        } else {
            upper_code = middle - 1;
        }
    }
    let higher_code = lower_code.saturating_add(1).min(126);
    let lower = positive_finite_value(lower_code);
    let higher = positive_finite_value(higher_code);
    let selected = if magnitude - lower < higher - magnitude
        || (magnitude - lower == higher - magnitude && value >= 0.0)
    {
        lower
    } else {
        higher
    };
    if selected == 0.0 {
        0.0
    } else if value < 0.0 {
        -selected
    } else {
        selected
    }
}

fn assert_exact(value: f32) {
    assert_eq!(
        fast_exact_nearest(value).to_bits(),
        exhaustive_nearest(value).to_bits(),
        "FP8 nearest mismatch for value {value:?} bits 0x{:08x}",
        value.to_bits()
    );
}

fn clamp_like_cuda_qdq(value: f32) -> f32 {
    if value.is_nan() {
        -448.0
    } else {
        value.clamp(-448.0, 448.0)
    }
}

#[test]
fn fast_search_matches_every_codepoint_and_midpoint_tie() {
    for code in 0..=u8::MAX {
        if let Some(value) = finite_value(code) {
            assert_exact(value);
        }
    }
    for code in 0_u8..126 {
        let lower = positive_finite_value(code);
        let higher = positive_finite_value(code + 1);
        let midpoint = (lower + higher) * 0.5;
        for value in [
            midpoint.next_down(),
            midpoint,
            midpoint.next_up(),
            -midpoint.next_down(),
            -midpoint,
            -midpoint.next_up(),
        ] {
            assert_exact(value);
        }
        assert_eq!(fast_exact_nearest(midpoint).to_bits(), lower.to_bits());
        assert_eq!(fast_exact_nearest(-midpoint).to_bits(), (-higher).to_bits());
    }
}

#[test]
fn fast_search_matches_every_bf16_input_and_special_value() {
    for bits in 0..=u16::MAX {
        assert_exact(clamp_like_cuda_qdq(f32::from_bits(u32::from(bits) << 16)));
    }
    assert_exact(-0.0);
    assert_exact(0.0);
    assert_eq!(fast_exact_nearest(f32::NAN).to_bits(), 0.0_f32.to_bits());
    assert_exact(clamp_like_cuda_qdq(f32::NEG_INFINITY));
    assert_exact(clamp_like_cuda_qdq(f32::INFINITY));
    assert_exact(clamp_like_cuda_qdq(f32::from_bits(0xffc0_0001)));
}

#[test]
fn v21_switches_only_the_m287_ratio4_qdq_launch() {
    let source = include_str!("../native/augur_cuda_runtime.cu");
    let historical_m139 = source
        .split("// Historical M139 evidence binds this symbol")
        .nth(1)
        .expect("historical M139 marker")
        .split("// M287 owns the Graph-exact RMS path")
        .next()
        .expect("historical M139 body");
    assert!(historical_m139.contains("augur_m139_qdq<<<"));
    assert!(!historical_m139.contains("augur_m139_qdq_fast_exact<<<"));

    let m287 = source
        .split("augur_cuda_m287_ratio4_staged_submit_v1")
        .nth(1)
        .expect("M287 submit body")
        .split("augur_cuda_m287_ratio4_staged_v1")
        .next()
        .expect("M287 submit boundary");
    assert!(m287.contains("AUGUR_CUDA_P0_PREFILL_V21_RATIO4_FAST_EXACT_QDQ"));
    assert!(m287.contains("augur_m139_qdq_fast_exact<<<"));
    assert!(m287.contains("augur_m139_qdq<<<"));

    let m144 = source
        .split("augur_cuda_m288_ratio128_staged_submit_v1")
        .nth(1)
        .expect("M288 Ratio128 submit body");
    assert!(m144.contains("augur_m144_qdq<<<"));
    assert!(!m144.contains("augur_m139_qdq_fast_exact<<<"));
}
