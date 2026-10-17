//! Windows caller-available space, queried only on the pinned checkpoint root.
//! This is admission/backpressure, not an atomic disk-space reservation.

use std::{io, os::windows::ffi::OsStrExt, path::Path};

#[link(name = "kernel32")]
unsafe extern "system" {
    #[link_name = "GetDiskFreeSpaceExW"]
    fn get_disk_free_space(
        directory: *const u16,
        caller_available: *mut u64,
        total: *mut u64,
        volume_free: *mut u64,
    ) -> i32;
}

pub(super) fn available_bytes(root: &Path) -> io::Result<u64> {
    let mut wide = root.as_os_str().encode_wide().collect::<Vec<_>>();
    if wide.contains(&0) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "disk root contains NUL",
        ));
    }
    wide.push(0);
    let (mut available, mut total, mut volume_free) = (0, 0, 0);
    // SAFETY: the UTF-16 input is terminated and lives through this synchronous
    // call; all output pointers are valid u64-sized ULARGE_INTEGER storage.
    // The caller retains DirectoryLease for this path and its ancestors.
    let success =
        unsafe { get_disk_free_space(wide.as_ptr(), &mut available, &mut total, &mut volume_free) };
    if success == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(available)
}

pub(super) fn require_capacity(required: u64, available: u64) -> io::Result<()> {
    if available < required {
        return Err(io::Error::new(
            io::ErrorKind::StorageFull,
            format!("checkpoint needs {required} available disk bytes; only {available} remain"),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disk_bound_is_exact_and_does_not_overflow() {
        assert!(require_capacity(0, 0).is_ok());
        assert!(require_capacity(1, 1).is_ok());
        assert!(require_capacity(u64::MAX, u64::MAX).is_ok());
        assert_eq!(
            require_capacity(1, 0).unwrap_err().kind(),
            io::ErrorKind::StorageFull
        );
        assert!(require_capacity(u64::MAX, u64::MAX - 1).is_err());
    }

    #[test]
    fn actual_windows_query_and_bad_path_remain_distinct() {
        assert!(available_bytes(&std::env::temp_dir()).is_ok());
        assert!(available_bytes(Path::new("C:/not-a-disk-root-9225/not-present")).is_err());
    }
}
