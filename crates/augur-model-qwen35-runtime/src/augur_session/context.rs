//! Per-process KV extent. The public launcher selects before creating any owner.
use super::{Result, CAPACITY};
use std::sync::OnceLock;
pub const CONTEXT_CHOICES: [usize; 6] = [8192, 32768, 65536, 131072, 196608, 262144];
static SELECTED: OnceLock<std::result::Result<usize, String>> = OnceLock::new();
pub fn parse_context_capacity(value: Option<&str>) -> Result<usize> {
    let capacity = match value { None => CAPACITY, Some(v) => {
        if v.is_empty() || !v.bytes().all(|b| b.is_ascii_digit()) {
            return Err("invalid AUGUR_CONTEXT_TOKENS".into());
        }
        v.parse::<usize>()?
    }};
    if !CONTEXT_CHOICES.contains(&capacity) {
        return Err("context must be 8K, 32K, 64K, 128K, 192K or 256K".into());
    }
    Ok(capacity)
}
pub fn context_capacity() -> Result<usize> {
    match SELECTED.get_or_init(|| {
        match std::env::var("AUGUR_CONTEXT_TOKENS") {
            Ok(v) => parse_context_capacity(Some(&v)).map_err(|e| e.to_string()),
            Err(std::env::VarError::NotPresent) => Ok(CAPACITY),
            Err(e) => Err(e.to_string()),
        }
    }) { Ok(n) => Ok(*n), Err(e) => Err(e.clone().into()) }
}
pub fn context_output_limit() -> Result<usize> { Ok(context_capacity()? * 95 / 100) }
pub fn fits_context(tokens: Option<usize>) -> Result<bool> {
    let capacity = context_capacity()?;
    Ok(tokens.is_some_and(|n| n <= capacity))
}
#[cfg(test)] mod tests {
    use super::*;
    #[test] fn all_six_context_extents() {
        assert_eq!(parse_context_capacity(None).unwrap(), CAPACITY);
        for n in CONTEXT_CHOICES { assert_eq!(parse_context_capacity(Some(&n.to_string())).unwrap(), n); }
    }
    #[test] fn invalid_extents_fail_closed() {
        for v in ["", "0", "8193", "16384", "-8192", "+8192", "8192 ", "256k", "99999999999999999999999999999"] {
            assert!(parse_context_capacity(Some(v)).is_err(), "{v}");
        }
    }
    #[test] fn default_budget_and_overflow() {
        assert_eq!(context_output_limit().unwrap(), super::super::MAX_OUTPUT);
        assert!(!fits_context(None).unwrap());
        assert!(fits_context(Some(CAPACITY)).unwrap());
        assert!(!fits_context(Some(CAPACITY+1)).unwrap());
    }
}
