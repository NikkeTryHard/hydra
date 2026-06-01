//! ONNX Runtime policy loader for Python-exported Hydra BC policies.

use std::{fs, path::Path};

use hydra_core::{action::HYDRA_ACTION_SPACE, encoder::OBS_SIZE};
use ndarray::ArrayView3;
use ort::{
    ep::CUDA,
    inputs,
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use thiserror::Error;

const OBS_CHANNELS: usize = 192;
const TILE_WIDTH: usize = 34;
const POLICY_ONLY_SCHEMA: u32 = 2;
const PPO_POLICY_VALUE_SCHEMA: u32 = 3;

#[derive(Debug, Error)]
pub enum OnnxPolicyError {
    #[error("failed to read {path}: {source}")]
    Read {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse metadata: {0}")]
    Metadata(#[from] serde_json::Error),
    #[error("ONNX Runtime error: {0}")]
    Ort(#[from] ort::Error),
    #[error("invalid ONNX policy artifact: {0}")]
    Invalid(String),
}

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

pub struct PolicyValueBatch {
    pub logits: Vec<[f32; HYDRA_ACTION_SPACE]>,
    pub values: Vec<f32>,
}

pub struct OnnxPolicyRuntime {
    metadata: OnnxPolicyMetadata,
    session: Session,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum OnnxPolicyDevice {
    Cpu,
    Cuda { device_id: i32 },
}

impl OnnxPolicyDevice {
    pub fn parse(raw: &str) -> Result<Self, OnnxPolicyError> {
        if raw == "cpu" {
            return Ok(Self::Cpu);
        }
        if raw == "cuda" {
            return Ok(Self::Cuda { device_id: 0 });
        }
        if let Some(id) = raw.strip_prefix("cuda:") {
            let device_id = id
                .parse::<i32>()
                .map_err(|_| OnnxPolicyError::Invalid(format!("invalid CUDA device {raw:?}")))?;
            if device_id < 0 {
                return Err(OnnxPolicyError::Invalid(format!(
                    "invalid CUDA device {raw:?}"
                )));
            }
            return Ok(Self::Cuda { device_id });
        }
        Err(OnnxPolicyError::Invalid(format!(
            "unsupported ONNX device {raw:?}"
        )))
    }
}

impl OnnxPolicyRuntime {
    pub fn load_dir(path: impl AsRef<Path>) -> Result<Self, OnnxPolicyError> {
        Self::load_dir_inner(path.as_ref(), OnnxPolicyDevice::Cpu)
    }

    pub fn load_dir_with_device(
        path: impl AsRef<Path>,
        device: OnnxPolicyDevice,
    ) -> Result<Self, OnnxPolicyError> {
        Self::load_dir_inner(path.as_ref(), device)
    }

    fn load_dir_inner(path: &Path, device: OnnxPolicyDevice) -> Result<Self, OnnxPolicyError> {
        let metadata_path = path.join("policy.json");
        let metadata_bytes = fs::read(&metadata_path).map_err(|source| OnnxPolicyError::Read {
            path: metadata_path.display().to_string(),
            source,
        })?;
        let metadata: OnnxPolicyMetadata = serde_json::from_slice(&metadata_bytes)?;
        validate_metadata(&metadata)?;
        let artifact_path = path.join(&metadata.artifact);
        let artifact_bytes = fs::read(&artifact_path).map_err(|source| OnnxPolicyError::Read {
            path: artifact_path.display().to_string(),
            source,
        })?;
        let digest = hex_sha256(&artifact_bytes);
        if digest != metadata.artifact_sha256 {
            return Err(OnnxPolicyError::Invalid(format!(
                "artifact sha256 mismatch: got {digest} expected {}",
                metadata.artifact_sha256
            )));
        }
        let mut builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|err| OnnxPolicyError::Invalid(err.to_string()))?
            .with_intra_threads(1)
            .map_err(|err| OnnxPolicyError::Invalid(err.to_string()))?
            .with_inter_threads(1)
            .map_err(|err| OnnxPolicyError::Invalid(err.to_string()))?;
        if let OnnxPolicyDevice::Cuda { device_id } = device {
            let cuda = CUDA::default()
                .with_device_id(device_id)
                .build()
                .error_on_failure();
            builder = builder
                .with_execution_providers([cuda])
                .map_err(|err| OnnxPolicyError::Invalid(format!("failed to enable CUDA: {err}")))?;
        }
        let session = builder.commit_from_file(&artifact_path)?;
        Ok(Self { metadata, session })
    }

    pub fn metadata(&self) -> &OnnxPolicyMetadata {
        &self.metadata
    }

    pub fn policy_logits_batch(
        &mut self,
        obs: &[f32],
    ) -> Result<Vec<[f32; HYDRA_ACTION_SPACE]>, OnnxPolicyError> {
        let logits_name = self.policy_logits_output_name()?.to_owned();
        let batch = self.validate_obs_batch(obs)?;
        let outputs = self.run_batch(obs, batch)?;
        extract_logits(&outputs, &logits_name, batch)
    }

    pub fn policy_value_batch(&mut self, obs: &[f32]) -> Result<PolicyValueBatch, OnnxPolicyError> {
        if self.metadata.schema_version != PPO_POLICY_VALUE_SCHEMA
            || self.metadata.artifact_kind.as_deref() != Some("ppo_policy_value")
        {
            return Err(OnnxPolicyError::Invalid(
                "policy_value_batch requires schema v3 ppo_policy_value artifact".to_owned(),
            ));
        }
        let outputs_metadata = self.metadata.outputs.as_ref().ok_or_else(|| {
            OnnxPolicyError::Invalid(
                "schema v3 ppo_policy_value metadata missing outputs".to_owned(),
            )
        })?;
        let logits_name = outputs_metadata.policy_logits.name.clone();
        let value_name = outputs_metadata.value.name.clone();
        let batch = self.validate_obs_batch(obs)?;
        let outputs = self.run_batch(obs, batch)?;
        let logits = extract_logits(&outputs, &logits_name, batch)?;
        let value = outputs
            .get(value_name.as_str())
            .ok_or_else(|| OnnxPolicyError::Invalid(format!("missing output {value_name}")))?;
        let (shape, data) = value.try_extract_tensor::<f32>()?;
        if shape.as_ref() != [batch as i64, 1] {
            return Err(OnnxPolicyError::Invalid(format!(
                "value output shape {:?} != [{batch}, 1]",
                shape.as_ref()
            )));
        }
        Ok(PolicyValueBatch {
            logits,
            values: data.to_vec(),
        })
    }

    fn validate_obs_batch(&self, obs: &[f32]) -> Result<usize, OnnxPolicyError> {
        if !obs.len().is_multiple_of(OBS_SIZE) {
            return Err(OnnxPolicyError::Invalid(format!(
                "obs length {} is not divisible by {OBS_SIZE}",
                obs.len()
            )));
        }
        let batch = obs.len() / OBS_SIZE;
        if batch == 0 || batch > self.metadata.max_batch {
            return Err(OnnxPolicyError::Invalid(format!(
                "batch {batch} outside 1..{}",
                self.metadata.max_batch
            )));
        }
        Ok(batch)
    }

    fn run_batch(
        &mut self,
        obs: &[f32],
        batch: usize,
    ) -> Result<ort::session::SessionOutputs<'_>, OnnxPolicyError> {
        let view = ArrayView3::from_shape((batch, OBS_CHANNELS, TILE_WIDTH), obs)
            .map_err(|err| OnnxPolicyError::Invalid(err.to_string()))?;
        let input = TensorRef::from_array_view(view)?;
        Ok(self
            .session
            .run(inputs![self.metadata.input_name.as_str() => input])?)
    }

    fn policy_logits_output_name(&self) -> Result<&str, OnnxPolicyError> {
        if self.metadata.schema_version == PPO_POLICY_VALUE_SCHEMA {
            let outputs = self.metadata.outputs.as_ref().ok_or_else(|| {
                OnnxPolicyError::Invalid("schema v3 metadata missing outputs".to_owned())
            })?;
            Ok(outputs.policy_logits.name.as_str())
        } else {
            self.metadata.output_name.as_deref().ok_or_else(|| {
                OnnxPolicyError::Invalid("schema v2 metadata missing output_name".to_owned())
            })
        }
    }
}

fn validate_metadata(metadata: &OnnxPolicyMetadata) -> Result<(), OnnxPolicyError> {
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

fn validate_output_metadata(
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

fn extract_logits(
    outputs: &ort::session::SessionOutputs<'_>,
    output_name: &str,
    batch: usize,
) -> Result<Vec<[f32; HYDRA_ACTION_SPACE]>, OnnxPolicyError> {
    let value = outputs
        .get(output_name)
        .ok_or_else(|| OnnxPolicyError::Invalid(format!("missing output {output_name}")))?;
    let (shape, data) = value.try_extract_tensor::<f32>()?;
    if shape.as_ref() != [batch as i64, HYDRA_ACTION_SPACE as i64] {
        return Err(OnnxPolicyError::Invalid(format!(
            "policy_logits output shape {:?} != [{batch}, {HYDRA_ACTION_SPACE}]",
            shape.as_ref()
        )));
    }
    let mut out = Vec::with_capacity(batch);
    for row in data.chunks_exact(HYDRA_ACTION_SPACE) {
        let mut logits = [0.0; HYDRA_ACTION_SPACE];
        logits.copy_from_slice(row);
        out.push(logits);
    }
    Ok(out)
}

fn invalid<T>(msg: String) -> Result<T, OnnxPolicyError> {
    Err(OnnxPolicyError::Invalid(msg))
}

fn hex_sha256(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    let digest = h.finalize();
    let mut out = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

#[cfg(test)]
mod tests {
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
                OutputDim::Value(HYDRA_ACTION_SPACE),
            ]),
            artifact_kind: None,
            outputs: None,
            encoder_shape: [OBS_CHANNELS, TILE_WIDTH],
            action_space: HYDRA_ACTION_SPACE,
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
                    OutputDim::Value(HYDRA_ACTION_SPACE),
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
}
