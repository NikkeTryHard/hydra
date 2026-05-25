use std::{env, fs, path::PathBuf};

use hydra_core::{action::HYDRA_ACTION_SPACE, encoder::OBS_SIZE};
use hydra_model::onnx_policy::OnnxPolicyRuntime;
use safetensors::SafeTensors;

fn main() {
    let mut args = env::args_os().skip(1);
    let dir = PathBuf::from(
        args.next()
            .expect("usage: onnx_policy_parity <export-dir> [tol]"),
    );
    let tol = args
        .next()
        .map(|v| v.to_string_lossy().parse::<f32>().expect("tol must be f32"))
        .unwrap_or(1.0e-4);
    hydra_model::ort_init::init_ort_from_env().expect("ort init");
    let mut model = OnnxPolicyRuntime::load_dir(&dir).expect("model load");
    let fixture_bytes = fs::read(dir.join("parity_fixture.safetensors")).expect("fixture");
    let fixture = SafeTensors::deserialize(&fixture_bytes).expect("fixture parse");
    let obs = bytes_to_f32(fixture.tensor("obs").expect("obs").data());
    let expected = bytes_to_f32(
        fixture
            .tensor("policy_logits")
            .expect("policy_logits")
            .data(),
    );
    assert_eq!(obs.len() % OBS_SIZE, 0);
    let batch = obs.len() / OBS_SIZE;
    assert_eq!(expected.len(), batch * HYDRA_ACTION_SPACE);
    let actual = model.policy_logits_batch(&obs).expect("forward");
    let mut max_abs = 0.0f32;
    for (row, expected_row) in actual.iter().zip(expected.chunks_exact(HYDRA_ACTION_SPACE)) {
        for (&a, &e) in row.iter().zip(expected_row.iter()) {
            max_abs = max_abs.max((a - e).abs());
        }
    }
    assert!(max_abs <= tol, "parity max_abs {max_abs} > {tol}");
    println!("parity_ok batch={batch} max_abs={max_abs}");
}

fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}
