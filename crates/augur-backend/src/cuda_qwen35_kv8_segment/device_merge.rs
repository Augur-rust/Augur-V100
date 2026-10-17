//! Device-resident F64 stable-LSE accumulator for V2 F32 segment partials.
//! This low-level opt-in interface does not create a page/terminal capability.

use crate::{CudaRuntimeError, OwnedCudaStream};
use core::mem;

pub const LSE_RESET: u32 = 0;
pub const LSE_ACCUMULATE: u32 = 1;
pub const LSE_FINALIZE: u32 = 2;
/// Import an already cumulative ordered prefix; never sum it with itself.
pub const LSE_IMPORT_ORDERED_PREFIX: u32 = 3;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AugurCudaQwen35Kv8LseMergeArgsV2 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub partial_max_f32: u64,
    pub partial_max_len_bytes: u64,
    pub partial_denominator_f32: u64,
    pub partial_denominator_len_bytes: u64,
    pub partial_weighted_f32: u64,
    pub partial_weighted_len_bytes: u64,
    pub accumulated_max_f64: u64,
    pub accumulated_max_len_bytes: u64,
    pub accumulated_denominator_f64: u64,
    pub accumulated_denominator_len_bytes: u64,
    pub accumulated_weighted_f64: u64,
    pub accumulated_weighted_len_bytes: u64,
    pub output_f32: u64,
    pub output_len_bytes: u64,
    pub status_u32: u64,
    pub status_len_bytes: u64,
    pub query_rows: u32,
    pub head_dim: u32,
    pub operation: u32,
    pub reserved: u32,
}

impl Default for AugurCudaQwen35Kv8LseMergeArgsV2 {
    fn default() -> Self {
        Self {
            abi_version: 2,
            struct_size: mem::size_of::<Self>() as u32,
            partial_max_f32: 0,
            partial_max_len_bytes: 0,
            partial_denominator_f32: 0,
            partial_denominator_len_bytes: 0,
            partial_weighted_f32: 0,
            partial_weighted_len_bytes: 0,
            accumulated_max_f64: 0,
            accumulated_max_len_bytes: 0,
            accumulated_denominator_f64: 0,
            accumulated_denominator_len_bytes: 0,
            accumulated_weighted_f64: 0,
            accumulated_weighted_len_bytes: 0,
            output_f32: 0,
            output_len_bytes: 0,
            status_u32: 0,
            status_len_bytes: 0,
            query_rows: 24,
            head_dim: 256,
            operation: LSE_RESET,
            reserved: 0,
        }
    }
}

const _: () = assert!(mem::size_of::<AugurCudaQwen35Kv8LseMergeArgsV2>() == 152);
const _: () = assert!(mem::align_of::<AugurCudaQwen35Kv8LseMergeArgsV2>() == 8);

pub fn validate_lse_merge_args_v2(a: &AugurCudaQwen35Kv8LseMergeArgsV2) -> Result<(), &'static str> {
    if a.abi_version != 2
        || a.struct_size as usize != mem::size_of_val(a)
        || a.query_rows == 0
        || a.query_rows > 24 * 64
        || a.query_rows % 24 != 0
        || a.head_dim != 256
        || a.operation > LSE_IMPORT_ORDERED_PREFIX
        || a.reserved != 0
    {
        return Err("LSE merge ABI/geometry/operation");
    }
    let rows = u64::from(a.query_rows);
    let values = rows * 256;
    let ranges = [
        (a.partial_max_f32, a.partial_max_len_bytes, rows * 4, 4),
        (
            a.partial_denominator_f32,
            a.partial_denominator_len_bytes,
            rows * 4,
            4,
        ),
        (
            a.partial_weighted_f32,
            a.partial_weighted_len_bytes,
            values * 4,
            4,
        ),
        (
            a.accumulated_max_f64,
            a.accumulated_max_len_bytes,
            rows * 8,
            8,
        ),
        (
            a.accumulated_denominator_f64,
            a.accumulated_denominator_len_bytes,
            rows * 8,
            8,
        ),
        (
            a.accumulated_weighted_f64,
            a.accumulated_weighted_len_bytes,
            values * 8,
            8,
        ),
        (a.output_f32, a.output_len_bytes, values * 4, 4),
        (a.status_u32, a.status_len_bytes, rows * 4, 4),
    ];
    for (i, &(start, bytes, expected, alignment)) in ranges.iter().enumerate() {
        if start == 0 || bytes != expected || start % alignment != 0 {
            return Err("LSE merge pointer/extent/alignment");
        }
        let end = start
            .checked_add(bytes)
            .ok_or("LSE merge pointer overflow")?;
        for &(other, other_bytes, _, _) in &ranges[..i] {
            let other_end = other
                .checked_add(other_bytes)
                .ok_or("LSE merge pointer overflow")?;
            if start < other_end && other < end {
                return Err("LSE merge aliased ranges");
            }
        }
    }
    Ok(())
}

/// Enqueue reset/accumulate/finalize without a host readback or synchronization.
///
/// # Safety
/// All ranges must be valid allocations on `stream`'s device and remain alive
/// until its terminal completion. Reset must precede accumulate/finalize, and
/// these buffers may only be accessed in order by this request's stream. The
/// caller must inspect terminal status (all rows zero) before any publication;
/// an invalid partial permanently poisons the row until a new reset.
pub unsafe fn submit_lse_merge_v2(
    stream: &OwnedCudaStream,
    args: &AugurCudaQwen35Kv8LseMergeArgsV2,
) -> Result<(), CudaRuntimeError> {
    validate_lse_merge_args_v2(args).map_err(CudaRuntimeError::invalid_argument)?;
    // SAFETY: the metadata is validated here; device lifetime and stream order
    // are the explicit caller obligations above. Native code snapshots args.
    let result = unsafe {
        augur_cuda_sys::augur_cuda_qwen35_kv8_lse_merge_submit_v2(
            stream.device_ordinal(),
            stream.raw()?,
            (args as *const AugurCudaQwen35Kv8LseMergeArgsV2).cast(),
        )
    };
    CudaRuntimeError::from_result("qwen35 kv8 device LSE merge", result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> AugurCudaQwen35Kv8LseMergeArgsV2 {
        AugurCudaQwen35Kv8LseMergeArgsV2 {
            partial_max_f32: 0x10000,
            partial_max_len_bytes: 96,
            partial_denominator_f32: 0x20000,
            partial_denominator_len_bytes: 96,
            partial_weighted_f32: 0x30000,
            partial_weighted_len_bytes: 24576,
            accumulated_max_f64: 0x40000,
            accumulated_max_len_bytes: 192,
            accumulated_denominator_f64: 0x50000,
            accumulated_denominator_len_bytes: 192,
            accumulated_weighted_f64: 0x60000,
            accumulated_weighted_len_bytes: 49152,
            output_f32: 0x70000,
            output_len_bytes: 24576,
            status_u32: 0x80000,
            status_len_bytes: 96,
            ..Default::default()
        }
    }

    #[test]
    fn merge_abi_size_and_all_operations() {
        assert_eq!(mem::size_of::<AugurCudaQwen35Kv8LseMergeArgsV2>(), 152);
        for operation in 0..=3 {
            let a = AugurCudaQwen35Kv8LseMergeArgsV2 {
                operation,
                ..fixture()
            };
            assert_eq!(validate_lse_merge_args_v2(&a), Ok(()));
        }
    }

    #[test]
    fn merge_rejects_metadata_pointer_and_alias_drift() {
        for case in 0..12 {
            let mut a = fixture();
            match case {
                0 => a.abi_version = 1,
                1 => a.struct_size -= 8,
                2 => a.query_rows = 0,
                3 => a.query_rows = 25,
                4 => a.query_rows = 24 * 65,
                5 => a.head_dim = 128,
                6 => a.operation = 4,
                7 => a.reserved = 1,
                8 => a.output_f32 = 0,
                9 => a.output_len_bytes -= 4,
                10 => a.accumulated_max_f64 += 4,
                _ => a.output_f32 = a.partial_weighted_f32,
            }
            assert!(validate_lse_merge_args_v2(&a).is_err(), "case {case}");
        }
        let mut a = fixture();
        a.output_f32 = u64::MAX - 3;
        assert!(validate_lse_merge_args_v2(&a).is_err());
    }

    #[test]
    #[ignore = "requires an exclusive SM70 GPU window"]
    fn native_device_merge_sm70_stable_and_poisoned_until_reset() {
        use crate::{cuda_device_properties, OwnedCudaBuffer};
        let device = cuda_device_properties(0).unwrap();
        assert_eq!(
            (
                device.compute_capability.major,
                device.compute_capability.minor
            ),
            (7, 0)
        );
        let stream = OwnedCudaStream::create(0).unwrap();
        let sizes = [96, 96, 24576, 192, 192, 49152, 24576, 96];
        let buffers: Vec<_> = sizes
            .into_iter()
            .map(|bytes| OwnedCudaBuffer::allocate(0, bytes).unwrap())
            .collect();
        let ptr = |i: usize| buffers[i].device_ptr().unwrap().raw_value();
        let mut a = AugurCudaQwen35Kv8LseMergeArgsV2 {
            partial_max_f32: ptr(0),
            partial_denominator_f32: ptr(1),
            partial_weighted_f32: ptr(2),
            accumulated_max_f64: ptr(3),
            accumulated_denominator_f64: ptr(4),
            accumulated_weighted_f64: ptr(5),
            output_f32: ptr(6),
            status_u32: ptr(7),
            ..fixture()
        };
        let upload = |m: f32, d: f32, n: f32| {
            for (i, value, count) in [(0, m, 24), (1, d, 24), (2, n, 6144)] {
                let bytes: Vec<_> = (0..count).flat_map(|_| value.to_le_bytes()).collect();
                buffers[i].copy_from_host_sync(&stream, &bytes).unwrap();
            }
        };
        let submit = |a: &AugurCudaQwen35Kv8LseMergeArgsV2| {
            // SAFETY: all buffers above are live, disjoint, and owned by this
            // test; all submits are ordered on stream and readbacks wait.
            unsafe {
                submit_lse_merge_v2(&stream, a).unwrap();
            }
        };
        let read = || {
            let mut out = vec![0; 24576];
            let mut status = vec![0; 96];
            buffers[6].copy_to_host_sync(&stream, &mut out).unwrap();
            buffers[7].copy_to_host_sync(&stream, &mut status).unwrap();
            let values: Vec<_> = out
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect();
            let statuses: Vec<_> = status
                .chunks_exact(4)
                .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
                .collect();
            (values, statuses)
        };
        submit(&a);
        upload(1000.0, 2.0, 6.0);
        a.operation = LSE_ACCUMULATE;
        submit(&a);
        upload(-1000.0, 4.0, 20.0);
        submit(&a);
        upload(1000.0, 1.0, 9.0);
        submit(&a);
        a.operation = LSE_FINALIZE;
        submit(&a);
        let (values, statuses) = read();
        assert!(values.iter().all(|v| *v == 5.0));
        assert!(statuses.iter().all(|v| *v == 0));
        for (m, d, n) in [
            (0.0, f32::NAN, 1.0),
            (f32::INFINITY, 1.0, 1.0),
            (0.0, 1.0, f32::NAN),
            (0.0, -1.0, 1.0),
            (0.0, 0.0, 0.0),
            (f32::NEG_INFINITY, 0.0, 1.0),
        ] {
            a.operation = LSE_RESET;
            submit(&a);
            upload(m, d, n);
            a.operation = LSE_ACCUMULATE;
            submit(&a);
            upload(0.0, 1.0, 1.0);
            submit(&a); // a later healthy page cannot erase a poisoned row
            a.operation = LSE_FINALIZE;
            submit(&a);
            let (values, statuses) = read();
            assert!(values.iter().all(|v| v.is_nan()));
            assert!(statuses.iter().all(|v| *v == 1));
        }
        a.operation = LSE_RESET;
        submit(&a);
        upload(f32::NEG_INFINITY, 0.0, 0.0); // canonical empty partial
        a.operation = LSE_ACCUMULATE;
        submit(&a);
        upload(0.0, 2.0, 6.0);
        submit(&a);
        a.operation = LSE_FINALIZE;
        submit(&a);
        let (values, statuses) = read();
        assert!(values.iter().all(|v| *v == 3.0));
        assert!(statuses.iter().all(|v| *v == 0));
        // Cumulative import must replace, not add to, existing state.
        upload(0.0, 4.0, 20.0);
        a.operation = LSE_IMPORT_ORDERED_PREFIX;
        submit(&a);
        a.operation = LSE_FINALIZE;
        submit(&a);
        let (values, statuses) = read();
        assert!(values.iter().all(|v| *v == 5.0));
        assert!(statuses.iter().all(|v| *v == 0));
        // An invalid import poisons the row even if followed by good input.
        upload(0.0, -1.0, 1.0);
        a.operation = LSE_IMPORT_ORDERED_PREFIX;
        submit(&a);
        upload(0.0, 1.0, 1.0);
        submit(&a);
        a.operation = LSE_FINALIZE;
        submit(&a);
        let (values, statuses) = read();
        assert!(values.iter().all(|v| v.is_nan()));
        assert!(statuses.iter().all(|v| *v == 1));
        stream.synchronize().unwrap();
    }
}
