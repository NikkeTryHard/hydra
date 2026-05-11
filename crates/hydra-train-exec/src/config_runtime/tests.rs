use super::*;

#[test]
fn parse_train_device_accepts_cpu_cuda_and_indices() {
    assert_eq!(parse_train_device("cpu"), Ok(LibTorchDevice::Cpu));
    assert_eq!(parse_train_device(" CUDA "), Ok(LibTorchDevice::Cuda(0)));
    assert_eq!(parse_train_device("cuda:3"), Ok(LibTorchDevice::Cuda(3)));
}

#[test]
fn parse_train_device_rejects_invalid_values() {
    let err = parse_train_device("cuda:abc").expect_err("invalid cuda index should fail");
    assert!(err.contains("expected cpu, cuda, or cuda:<index>"));

    let err = parse_train_device("metal").expect_err("unsupported backend should fail");
    assert!(err.contains("unsupported HYDRA_TRAIN_DEVICE=metal"));
}
