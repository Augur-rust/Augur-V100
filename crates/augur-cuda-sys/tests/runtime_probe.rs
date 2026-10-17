#![cfg(feature = "cuda")]

use std::{ffi::CStr, ptr};

use augur_cuda_sys::*;

fn assert_ok(result: AugurCudaResultV1) {
    assert_eq!(
        result.status, AUGUR_CUDA_STATUS_OK,
        "CUDA bridge failed: project_status={}, raw_cuda_status={}",
        result.status, result.raw_cuda_status
    );
}

#[test]
fn abi_device_query_and_stream_lifecycle() {
    let abi = unsafe { augur_cuda_abi_version() };
    assert_eq!(abi, AUGUR_CUDA_ABI_VERSION_V1);

    let mut count = 0;
    assert_ok(unsafe { augur_cuda_device_count(&mut count) });
    assert!(count > 0, "CUDA test host has no visible devices");

    let mut properties = AugurCudaDevicePropertiesV1::new();
    assert_ok(unsafe { augur_cuda_get_device_properties(0, &mut properties) });

    assert_eq!(properties.ordinal, 0);
    assert!(properties.compute_major > 0);
    assert!(properties.total_global_memory_bytes > 0);

    let name = unsafe { CStr::from_ptr(properties.name.as_ptr()) }
        .to_str()
        .expect("CUDA device name must be UTF-8 for evidence output");

    assert!(!name.is_empty());

    let mut uuid = [0_u8; 16];
    assert_ok(unsafe { augur_cuda_get_device_uuid_v1(0, uuid.as_mut_ptr(), uuid.len() as u32) });
    assert!(uuid.iter().any(|byte| *byte != 0));

    eprintln!(
        "CUDA device: name={name}, ordinal={}, compute={}.{}, memory_bytes={}",
        properties.ordinal,
        properties.compute_major,
        properties.compute_minor,
        properties.total_global_memory_bytes
    );

    let mut stream: AugurCudaStream = ptr::null_mut();
    assert_ok(unsafe { augur_cuda_stream_create(0, &mut stream) });
    assert!(!stream.is_null());

    assert_ok(unsafe { augur_cuda_stream_synchronize(stream) });
    assert_ok(unsafe { augur_cuda_stream_destroy(stream) });
}

#[test]
fn null_arguments_are_rejected_without_runtime_submission() {
    let count_error = unsafe { augur_cuda_device_count(ptr::null_mut()) };
    assert_eq!(count_error.status, AUGUR_CUDA_STATUS_INVALID_ARGUMENT);

    let property_error = unsafe { augur_cuda_get_device_properties(0, ptr::null_mut()) };
    assert_eq!(property_error.status, AUGUR_CUDA_STATUS_INVALID_ARGUMENT);

    let mut uuid = [0_u8; 16];
    let uuid_null_error = unsafe { augur_cuda_get_device_uuid_v1(0, ptr::null_mut(), 16) };
    assert_eq!(uuid_null_error.status, AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
    let uuid_extent_error = unsafe { augur_cuda_get_device_uuid_v1(0, uuid.as_mut_ptr(), 15) };
    assert_eq!(uuid_extent_error.status, AUGUR_CUDA_STATUS_INVALID_ARGUMENT);

    let create_error = unsafe { augur_cuda_stream_create(0, ptr::null_mut()) };
    assert_eq!(create_error.status, AUGUR_CUDA_STATUS_INVALID_ARGUMENT);

    let sync_error = unsafe { augur_cuda_stream_synchronize(ptr::null_mut()) };
    assert_eq!(sync_error.status, AUGUR_CUDA_STATUS_INVALID_ARGUMENT);

    let wait_error = unsafe { augur_cuda_stream_wait_event_v1(ptr::null_mut(), ptr::null_mut()) };
    assert_eq!(wait_error.status, AUGUR_CUDA_STATUS_INVALID_ARGUMENT);

    let destroy_error = unsafe { augur_cuda_stream_destroy(ptr::null_mut()) };
    assert_eq!(destroy_error.status, AUGUR_CUDA_STATUS_INVALID_ARGUMENT);

    let pinned_alloc = unsafe { augur_cuda_host_alloc_pinned_v1(0, 4096, ptr::null_mut()) };
    assert_eq!(pinned_alloc.status, AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
    let pinned_free = unsafe { augur_cuda_host_free_pinned_v1(0, ptr::null_mut()) };
    assert_eq!(pinned_free.status, AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
    let async_h2d =
        unsafe { augur_cuda_memcpy_h2d_submit_v1(0, ptr::null_mut(), 0, 0, 0, ptr::null(), 0, 0) };
    assert_eq!(async_h2d.status, AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
    let d2d_2d = unsafe {
        augur_cuda_memcpy_d2d_2d_submit_v1(0, ptr::null_mut(), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    };
    assert_eq!(d2d_2d.status, AUGUR_CUDA_STATUS_INVALID_ARGUMENT);

    let graph_begin = unsafe { augur_cuda_graph_capture_begin_v1(0, ptr::null_mut()) };
    assert_eq!(graph_begin.status, AUGUR_CUDA_STATUS_INVALID_ARGUMENT);

    let graph_end = unsafe { augur_cuda_graph_capture_end_v1(0, ptr::null_mut(), ptr::null_mut()) };
    assert_eq!(graph_end.status, AUGUR_CUDA_STATUS_INVALID_ARGUMENT);

    let graph_launch = unsafe { augur_cuda_graph_launch_v1(0, ptr::null_mut(), ptr::null_mut()) };
    assert_eq!(graph_launch.status, AUGUR_CUDA_STATUS_INVALID_ARGUMENT);

    let graph_destroy = unsafe { augur_cuda_graph_destroy_v1(ptr::null_mut()) };
    assert_eq!(graph_destroy.status, AUGUR_CUDA_STATUS_INVALID_ARGUMENT);

    
}



#[test]
fn stream_wait_event_orders_two_nonblocking_streams() {
    let mut producer: AugurCudaStream = ptr::null_mut();
    let mut consumer: AugurCudaStream = ptr::null_mut();
    let mut ready: AugurCudaEvent = ptr::null_mut();
    assert_ok(unsafe { augur_cuda_stream_create(0, &mut producer) });
    assert_ok(unsafe { augur_cuda_stream_create(0, &mut consumer) });
    assert_ok(unsafe { augur_cuda_event_create(0, &mut ready) });

    assert_ok(unsafe { augur_cuda_event_record(ready, producer) });
    let mut complete = 0_u8;
    assert_ok(unsafe { augur_cuda_event_query_complete_v1(ready, &mut complete) });
    assert!(complete <= 1);
    assert_ok(unsafe { augur_cuda_stream_wait_event_v1(consumer, ready) });
    assert_ok(unsafe { augur_cuda_stream_synchronize(consumer) });
    assert_ok(unsafe { augur_cuda_event_query_complete_v1(ready, &mut complete) });
    assert_eq!(complete, 1);

    assert_ok(unsafe { augur_cuda_event_destroy(ready) });
    assert_ok(unsafe { augur_cuda_stream_destroy(consumer) });
    assert_ok(unsafe { augur_cuda_stream_destroy(producer) });
}

#[test]
fn device_properties_reject_wrong_abi_version() {
    let mut properties = AugurCudaDevicePropertiesV1::new();
    properties.abi_version += 1;

    let error = unsafe { augur_cuda_get_device_properties(0, &mut properties) };

    assert_eq!(error.status, AUGUR_CUDA_STATUS_ABI_VERSION_MISMATCH);
}

#[test]
fn extent_complete_sync_copies_reject_out_of_bounds_before_submission() {
    let source = [0_u8; 16];
    let mut destination = [0_u8; 16];
    let stream = ptr::null_mut();

    let h2d = unsafe {
        augur_cuda_memcpy_h2d_sync_v2(
            0,
            stream,
            0x1000,
            8,
            16,
            source.as_ptr(),
            source.len(),
            source.len(),
        )
    };
    assert_eq!(h2d.status, AUGUR_CUDA_STATUS_INVALID_ARGUMENT);

    let d2h = unsafe {
        augur_cuda_memcpy_d2h_sync_v2(
            0,
            stream,
            destination.as_mut_ptr(),
            destination.len(),
            0x1000,
            8,
            16,
            destination.len(),
        )
    };
    assert_eq!(d2h.status, AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
}

#[test]
fn pinned_host_async_h2d_lifecycle_roundtrips_exactly() {
    let mut stream: AugurCudaStream = ptr::null_mut();
    assert_ok(unsafe { augur_cuda_stream_create(0, &mut stream) });
    let mut host = ptr::null_mut();
    assert_ok(unsafe { augur_cuda_host_alloc_pinned_v1(0, 4096, &mut host) });
    let host_bytes = unsafe { std::slice::from_raw_parts_mut(host.cast::<u8>(), 4096) };
    for (index, value) in host_bytes.iter_mut().enumerate() {
        *value = (index % 251) as u8;
    }
    let mut device = 0_u64;
    assert_ok(unsafe { augur_cuda_device_alloc(0, 4096, &mut device) });
    assert_ok(unsafe {
        augur_cuda_memcpy_h2d_submit_v1(0, stream, device, 0, 4096, host, 4096, 4096)
    });
    assert_ok(unsafe { augur_cuda_stream_synchronize(stream) });
    let mut roundtrip = vec![0_u8; 4096];
    assert_ok(unsafe {
        augur_cuda_memcpy_d2h_sync_v2(
            0,
            stream,
            roundtrip.as_mut_ptr(),
            4096,
            device,
            0,
            4096,
            4096,
        )
    });
    assert_eq!(roundtrip, host_bytes);
    assert_ok(unsafe { augur_cuda_device_free(0, device) });
    assert_ok(unsafe { augur_cuda_host_free_pinned_v1(0, host) });
    assert_ok(unsafe { augur_cuda_stream_destroy(stream) });
}

#[test]
fn device_2d_copy_roundtrips_strided_columns_exactly() {
    let mut stream: AugurCudaStream = ptr::null_mut();
    assert_ok(unsafe { augur_cuda_stream_create(0, &mut stream) });
    let mut source = 0_u64;
    let mut destination = 0_u64;
    const SOURCE_LEN: usize = 4 * 8;
    const DESTINATION_LEN: usize = 4 * 12;
    assert_ok(unsafe { augur_cuda_device_alloc(0, SOURCE_LEN, &mut source) });
    assert_ok(unsafe { augur_cuda_device_alloc(0, DESTINATION_LEN, &mut destination) });

    let source_bytes = (0..SOURCE_LEN).map(|value| value as u8).collect::<Vec<_>>();
    let destination_bytes = vec![0_u8; DESTINATION_LEN];
    assert_ok(unsafe {
        augur_cuda_memcpy_h2d_sync_v2(
            0,
            stream,
            source,
            0,
            SOURCE_LEN,
            source_bytes.as_ptr(),
            source_bytes.len(),
            source_bytes.len(),
        )
    });
    assert_ok(unsafe {
        augur_cuda_memcpy_h2d_sync_v2(
            0,
            stream,
            destination,
            0,
            DESTINATION_LEN,
            destination_bytes.as_ptr(),
            destination_bytes.len(),
            destination_bytes.len(),
        )
    });
    let d2d_result = unsafe {
        augur_cuda_memcpy_d2d_2d_submit_v1(
            0,
            stream,
            destination,
            2,
            DESTINATION_LEN,
            12,
            source,
            1,
            SOURCE_LEN,
            8,
            4,
            4,
        )
    };
    assert_ok(d2d_result);
    assert_ok(unsafe { augur_cuda_stream_synchronize(stream) });

    let mut actual = vec![0_u8; DESTINATION_LEN];
    assert_ok(unsafe {
        augur_cuda_memcpy_d2h_sync_v2(
            0,
            stream,
            actual.as_mut_ptr(),
            actual.len(),
            destination,
            0,
            DESTINATION_LEN,
            DESTINATION_LEN,
        )
    });
    for row in 0..4 {
        assert_eq!(
            &actual[row * 12 + 2..row * 12 + 6],
            &source_bytes[row * 8 + 1..row * 8 + 5]
        );
    }
    assert_ok(unsafe { augur_cuda_device_free(0, destination) });
    assert_ok(unsafe { augur_cuda_device_free(0, source) });
    assert_ok(unsafe { augur_cuda_stream_destroy(stream) });
}
