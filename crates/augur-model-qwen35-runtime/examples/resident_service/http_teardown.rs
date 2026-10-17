//! Bounded process retirement after closing an owner's input/output channels.
//!
//! A natural process exit is NOT proof of CUDA cleanup or successful cancellation.
//! Callers must independently bind and verify the native cleanup receipt. A grace
//! timeout falls back to containment and must never be relabelled graceful cleanup.
use std::{
    io,
    process::{Child, ExitStatus},
    thread,
    time::{Duration, Instant},
};

#[derive(Debug)]
pub(super) struct ProcessRetirement {
    pub(super) exit: ExitStatus,
    pub(super) force_attempted: bool,
    pub(super) elapsed: Duration,
}

pub(super) fn retire_after_disconnect(
    child: &mut Child,
    grace: Duration,
) -> io::Result<ProcessRetirement> {
    let started = Instant::now();
    loop {
        if let Some(exit) = child.try_wait()? {
            return Ok(ProcessRetirement {
                exit,
                force_attempted: false,
                elapsed: started.elapsed(),
            });
        }
        let Some(remaining) = grace.checked_sub(started.elapsed()) else {
            break;
        };
        if remaining.is_zero() {
            break;
        }
        thread::sleep(remaining.min(Duration::from_millis(10)));
    }
    // Recheck the exact child handle at the force boundary. A failed kill is safe
    // only if this handle now confirms exit; otherwise the adapter must stop.
    if let Some(exit) = child.try_wait()? {
        return Ok(ProcessRetirement {
            exit,
            force_attempted: false,
            elapsed: started.elapsed(),
        });
    }
    let exit = match child.kill() {
        Ok(()) => child.wait()?,
        Err(error) => child.try_wait()?.ok_or(error)?,
    };
    Ok(ProcessRetirement {
        exit,
        force_attempted: true,
        elapsed: started.elapsed(),
    })
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;
    use std::{os::windows::process::CommandExt, process::Command, process::Stdio};

    fn child(script: &str) -> Child {
        Command::new("powershell.exe")
            .args(["-NoProfile", "-NonInteractive", "-Command", script])
            .creation_flags(0x08000000)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap()
    }

    #[test]
    fn natural_error_exit_is_not_forced_or_assumed_success() {
        let mut child = child("exit 7");
        let result = retire_after_disconnect(&mut child, Duration::from_secs(5)).unwrap();
        assert!(!result.force_attempted);
        assert_eq!(result.exit.code(), Some(7));
        assert!(child.try_wait().unwrap().is_some());
    }

    #[test]
    fn stalled_child_is_bounded_and_reaped() {
        let mut child = child("Start-Sleep -Seconds 10");
        let result = retire_after_disconnect(&mut child, Duration::from_millis(75)).unwrap();
        assert!(result.force_attempted);
        assert!(result.elapsed < Duration::from_secs(3));
        assert!(child.try_wait().unwrap().is_some());
    }

    #[test]
    fn already_exited_child_does_not_trigger_force() {
        let mut child = child("exit 0");
        assert!(child.wait().unwrap().success());
        let result = retire_after_disconnect(&mut child, Duration::ZERO).unwrap();
        assert!(!result.force_attempted);
        assert!(result.exit.success());
    }
}
