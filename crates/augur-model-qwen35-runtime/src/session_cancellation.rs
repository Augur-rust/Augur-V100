//! Owner-local, monotonic cooperative cancellation notification.
//!
//! A token conveys no CUDA completion, cleanup or rollback authority. The
//! execution owner must stop new work, drain admitted work and discard/poison
//! unpublished state before releasing it. Never reuse a cancelled token.
use std::{
    error::Error,
    fmt,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

#[derive(Clone, Debug, Default)]
pub struct SessionCancellationToken {
    requested: Arc<AtomicBool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionCancellationRequested {
    pub boundary: &'static str,
}

impl fmt::Display for SessionCancellationRequested {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "session cancellation requested at {}", self.boundary)
    }
}

impl Error for SessionCancellationRequested {}

impl SessionCancellationToken {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns true only for the first cancellation request. No reset exists.
    pub fn request_cancel(&self) -> bool {
        !self.requested.swap(true, Ordering::AcqRel)
    }

    pub fn is_requested(&self) -> bool {
        self.requested.load(Ordering::Acquire)
    }

    pub fn check(&self, boundary: &'static str) -> Result<(), SessionCancellationRequested> {
        if self.is_requested() {
            Err(SessionCancellationRequested { boundary })
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancellation_is_monotonic_and_shared_only_by_explicit_clones() {
        let token = SessionCancellationToken::new();
        let same = token.clone();
        let other = SessionCancellationToken::new();
        assert!(token.check("before submit").is_ok());
        assert!(same.request_cancel());
        assert!(!token.request_cancel());
        assert_eq!(
            token.check("next batch").unwrap_err().boundary,
            "next batch"
        );
        assert!(!other.is_requested());
    }

    #[test]
    fn another_thread_can_notify_without_owning_execution_state() {
        let token = SessionCancellationToken::new();
        let peer = token.clone();
        std::thread::spawn(move || peer.request_cancel())
            .join()
            .unwrap();
        assert!(token.check("prefix batch").is_err());
    }
}
