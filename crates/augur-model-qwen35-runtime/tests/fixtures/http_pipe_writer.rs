//! CPU-only native pipe writer for the HTTP adapter retirement test.
//! No runtime crate, CUDA context, model or cleanup-qualification claim.
use std::io::{self, Write};

fn main() {
    let mut output = io::stdout().lock();
    loop {
        if output
            .write_all(b"{\"event\":\"cpu_only_commit\"}\n")
            .and_then(|()| output.flush())
            .is_err()
        {
            std::process::exit(7);
        }
    }
}
