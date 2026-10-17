use std::env;

fn main() {
    let mut features = env::vars_os()
        .filter_map(|(name, _)| {
            name.to_str()
                .and_then(|name| name.strip_prefix("CARGO_FEATURE_"))
                .map(|name| name.to_ascii_lowercase().replace('_', "-"))
        })
        .collect::<Vec<_>>();
    features.sort();
    features.dedup();
    println!(
        "cargo:rustc-env=AUGUR_BACKEND_RESOLVED_FEATURES={}",
        features.join(",")
    );
}
