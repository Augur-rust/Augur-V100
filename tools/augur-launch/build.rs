//! Embed the lituus icon and version info with the Windows SDK resource compiler (rc.exe).
//! The build fails rather than silently shipping an exe without its icon.
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=augur.rc");
    println!("cargo:rerun-if-changed=augur.ico");
    println!("cargo:rerun-if-env-changed=AUGUR_RC");
    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR"));
    let res = out.join("augur.res");
    let rc = std::env::var_os("AUGUR_RC").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("rc.exe"));
    let status = Command::new(&rc).args(["/nologo", "/fo"]).arg(&res).arg("augur.rc").status()
        .unwrap_or_else(|e| panic!("augur-launch: cannot run {} ({e}); set AUGUR_RC to the Windows SDK rc.exe", rc.display()));
    assert!(status.success(), "augur-launch: rc.exe failed");
    println!("cargo:rustc-link-arg-bins={}", res.display());
}
