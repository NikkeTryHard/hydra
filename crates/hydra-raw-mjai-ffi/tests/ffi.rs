use std::ffi::CString;
use std::ptr;

use hydra_raw_mjai_ffi::{
    HydraRawMjaiConfig, HydraRawMjaiError, HydraRawMjaiStats, hydra_raw_mjai_error_free,
    hydra_raw_mjai_stream_open,
};

fn empty_config(input: &CString) -> HydraRawMjaiConfig {
    HydraRawMjaiConfig {
        input_utf8: input.as_ptr(),
        split: 0,
        train_fraction: 0.9,
        batch_size: 1,
        max_games: 0,
        max_samples: 0,
        num_threads: 0,
        queue_bound: 1,
        augment: false,
    }
}

#[test]
fn ffi_open_rejects_null_config() {
    let mut handle = ptr::null_mut();
    let mut stats = HydraRawMjaiStats::default();
    let mut err = HydraRawMjaiError::default();

    let status =
        unsafe { hydra_raw_mjai_stream_open(ptr::null(), &mut handle, &mut stats, &mut err) };

    assert!(status < 0);
    assert!(handle.is_null());
    assert_eq!(err.code, status);
    assert!(!err.message_utf8.is_null());
    unsafe { hydra_raw_mjai_error_free(&mut err) };
    assert!(err.message_utf8.is_null());
}

#[test]
fn ffi_open_rejects_null_output_handle() {
    let input = CString::new(".").unwrap();
    let cfg = empty_config(&input);
    let mut stats = HydraRawMjaiStats::default();
    let mut err = HydraRawMjaiError::default();

    let status = unsafe { hydra_raw_mjai_stream_open(&cfg, ptr::null_mut(), &mut stats, &mut err) };

    assert!(status < 0);
    assert_eq!(err.code, status);
    unsafe { hydra_raw_mjai_error_free(&mut err) };
}

#[test]
fn ffi_open_rejects_zero_batch() {
    let input = CString::new(".").unwrap();
    let cfg = HydraRawMjaiConfig {
        batch_size: 0,
        ..empty_config(&input)
    };
    let mut handle = ptr::null_mut();
    let mut stats = HydraRawMjaiStats::default();
    let mut err = HydraRawMjaiError::default();

    let status = unsafe { hydra_raw_mjai_stream_open(&cfg, &mut handle, &mut stats, &mut err) };

    assert!(status < 0);
    assert!(handle.is_null());
    unsafe { hydra_raw_mjai_error_free(&mut err) };
}
