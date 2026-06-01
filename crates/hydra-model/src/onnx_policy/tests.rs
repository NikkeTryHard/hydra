use super::metadata::{
    OBS_CHANNELS, POLICY_ONLY_SCHEMA, PPO_POLICY_VALUE_SCHEMA, TILE_WIDTH, validate_metadata,
};
use super::*;

fn policy_only_metadata() -> OnnxPolicyMetadata {
    OnnxPolicyMetadata {
        schema_version: POLICY_ONLY_SCHEMA,
        format: "onnx".to_owned(),
        artifact: "policy.onnx".to_owned(),
        artifact_sha256: "0".repeat(64),
        input_name: "obs".to_owned(),
        output_name: Some("policy_logits".to_owned()),
        output_shape: Some(vec![
            OutputDim::Symbol("N".to_owned()),
            OutputDim::Value(hydra_core::action::HYDRA_ACTION_SPACE),
        ]),
        artifact_kind: None,
        outputs: None,
        encoder_shape: [OBS_CHANNELS, TILE_WIDTH],
        action_space: hydra_core::action::HYDRA_ACTION_SPACE,
        max_batch: 8,
        checkpoint_global_step: 0,
        checkpoint_samples_seen: 0,
        weight_source: "raw".to_owned(),
    }
}

fn policy_value_outputs(value_shape: Vec<OutputDim>) -> OnnxPolicyValueOutputs {
    OnnxPolicyValueOutputs {
        policy_logits: OnnxOutputMetadata {
            name: "policy_logits".to_owned(),
            dtype: "float32".to_owned(),
            shape: vec![
                OutputDim::Symbol("N".to_owned()),
                OutputDim::Value(hydra_core::action::HYDRA_ACTION_SPACE),
            ],
        },
        value: OnnxOutputMetadata {
            name: "value".to_owned(),
            dtype: "float32".to_owned(),
            shape: value_shape,
        },
    }
}

fn policy_value_metadata() -> OnnxPolicyMetadata {
    OnnxPolicyMetadata {
        schema_version: PPO_POLICY_VALUE_SCHEMA,
        output_name: None,
        output_shape: None,
        artifact_kind: Some("ppo_policy_value".to_owned()),
        outputs: Some(policy_value_outputs(vec![
            OutputDim::Symbol("N".to_owned()),
            OutputDim::Value(1),
        ])),
        ..policy_only_metadata()
    }
}

#[test]
fn metadata_accepts_schema_v2_policy_only() {
    validate_metadata(&policy_only_metadata()).unwrap();
}

#[test]
fn metadata_rejects_schema_v2_missing_output_name() {
    let mut metadata = policy_only_metadata();
    metadata.output_name = None;

    let err = validate_metadata(&metadata).unwrap_err().to_string();

    assert!(err.contains("schema v2 output_name"));
}

#[test]
fn metadata_rejects_schema_v2_missing_output_shape() {
    let mut metadata = policy_only_metadata();
    metadata.output_shape = None;

    let err = validate_metadata(&metadata).unwrap_err().to_string();

    assert!(err.contains("schema v2 output_shape"));
}

#[test]
fn metadata_parses_schema_v3_without_legacy_output_name() {
    let raw = r#"{
        "schema_version": 3,
        "format": "onnx",
        "artifact_kind": "ppo_policy_value",
        "artifact": "policy.onnx",
        "artifact_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        "input_name": "obs",
        "outputs": {
            "policy_logits": {"name": "policy_logits", "dtype": "float32", "shape": ["N", 46]},
            "value": {"name": "value", "dtype": "float32", "shape": ["N", 1]}
        },
        "encoder_shape": [192, 34],
        "action_space": 46,
        "max_batch": 4096,
        "checkpoint_global_step": 7,
        "checkpoint_samples_seen": 11,
        "weight_source": "raw"
    }"#;

    let metadata: OnnxPolicyMetadata = serde_json::from_str(raw).unwrap();

    assert!(metadata.output_name.is_none());
    validate_metadata(&metadata).unwrap();
}

#[test]
fn metadata_rejects_schema_v3_missing_value_output() {
    let mut metadata = policy_value_metadata();
    metadata.outputs = None;

    let err = validate_metadata(&metadata).unwrap_err().to_string();

    assert!(err.contains("missing outputs"));
}

#[test]
fn metadata_rejects_schema_v3_bad_value_shape() {
    let mut metadata = policy_value_metadata();
    metadata.outputs = Some(policy_value_outputs(vec![
        OutputDim::Symbol("N".to_owned()),
        OutputDim::Value(2),
    ]));

    let err = validate_metadata(&metadata).unwrap_err().to_string();

    assert!(err.contains("outputs.value.shape"));
}

#[test]
fn metadata_accepts_schema_v3_policy_value() {
    validate_metadata(&policy_value_metadata()).unwrap();
}
