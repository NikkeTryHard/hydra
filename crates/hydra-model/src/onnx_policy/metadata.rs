use hydra_core::action::HYDRA_ACTION_SPACE;
use serde::Deserialize;

use super::OnnxPolicyError;

pub(super) const OBS_CHANNELS: usize = 192;
pub(super) const TILE_WIDTH: usize = 34;
pub(super) const POLICY_ONLY_SCHEMA: u32 = 2;
pub(super) const PPO_POLICY_VALUE_SCHEMA: u32 = 3;

#[derive(Debug, Clone, Deserialize)]
pub struct OnnxOutputMetadata {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<OutputDim>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OnnxPolicyValueOutputs {
    pub policy_logits: OnnxOutputMetadata,
    pub value: OnnxOutputMetadata,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum OutputDim {
    Symbol(String),
    Value(usize),
}

#[derive(Debug, Clone, Deserialize)]
pub struct OnnxPolicyMetadata {
    pub schema_version: u32,
    pub format: String,
    pub artifact: String,
    pub artifact_sha256: String,
    pub input_name: String,
    #[serde(default)]
    pub output_name: Option<String>,
    #[serde(default)]
    pub output_shape: Option<Vec<OutputDim>>,
    #[serde(default)]
    pub artifact_kind: Option<String>,
    #[serde(default)]
    pub outputs: Option<OnnxPolicyValueOutputs>,
    pub encoder_shape: [usize; 2],
    pub action_space: usize,
    pub max_batch: usize,
    pub checkpoint_global_step: u64,
    pub checkpoint_samples_seen: u64,
    pub weight_source: String,
}
pub(super) fn validate_metadata(metadata: &OnnxPolicyMetadata) -> Result<(), OnnxPolicyError> {
    if metadata.schema_version != POLICY_ONLY_SCHEMA
        && metadata.schema_version != PPO_POLICY_VALUE_SCHEMA
    {
        return invalid(format!(
            "schema_version {} unsupported",
            metadata.schema_version
        ));
    }
    if metadata.format != "onnx" {
        return invalid(format!("format {:?} unsupported", metadata.format));
    }
    if metadata.artifact != "policy.onnx" {
        return invalid(format!("artifact {:?} unsupported", metadata.artifact));
    }
    if metadata.input_name != "obs" {
        return invalid("input name must be obs".to_owned());
    }
    if metadata.encoder_shape != [OBS_CHANNELS, TILE_WIDTH] {
        return invalid("encoder_shape must be [192,34]".to_owned());
    }
    if metadata.action_space != HYDRA_ACTION_SPACE {
        return invalid("action_space must be 46".to_owned());
    }
    if metadata.max_batch == 0 {
        return invalid("max_batch must be nonzero".to_owned());
    }
    if metadata.schema_version == POLICY_ONLY_SCHEMA {
        if metadata.output_name.as_deref() != Some("policy_logits") {
            return invalid("schema v2 output_name must be policy_logits".to_owned());
        }
        if metadata.output_shape.as_deref()
            != Some(&[
                OutputDim::Symbol("N".to_owned()),
                OutputDim::Value(HYDRA_ACTION_SPACE),
            ])
        {
            return invalid("schema v2 output_shape must be [N,46]".to_owned());
        }
        return Ok(());
    }
    if metadata.artifact_kind.as_deref() != Some("ppo_policy_value") {
        return invalid("schema v3 artifact_kind must be ppo_policy_value".to_owned());
    }
    let outputs = metadata
        .outputs
        .as_ref()
        .ok_or_else(|| OnnxPolicyError::Invalid("schema v3 metadata missing outputs".to_owned()))?;
    validate_output_metadata(
        &outputs.policy_logits,
        "outputs.policy_logits",
        "policy_logits",
        &[
            OutputDim::Symbol("N".to_owned()),
            OutputDim::Value(HYDRA_ACTION_SPACE),
        ],
    )?;
    validate_output_metadata(
        &outputs.value,
        "outputs.value",
        "value",
        &[OutputDim::Symbol("N".to_owned()), OutputDim::Value(1)],
    )?;
    Ok(())
}

pub(super) fn validate_output_metadata(
    output: &OnnxOutputMetadata,
    field: &str,
    expected_name: &str,
    expected_shape: &[OutputDim],
) -> Result<(), OnnxPolicyError> {
    if output.name != expected_name {
        return invalid(format!("{field}.name must be {expected_name}"));
    }
    if output.dtype != "float32" {
        return invalid(format!("{field}.dtype must be float32"));
    }
    if output.shape != expected_shape {
        return invalid(format!("{field}.shape must be {expected_shape:?}"));
    }
    Ok(())
}
pub(super) fn invalid<T>(msg: String) -> Result<T, OnnxPolicyError> {
    Err(OnnxPolicyError::Invalid(msg))
}
