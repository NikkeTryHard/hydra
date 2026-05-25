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
const SCHEMA: u32 = 2;

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
pub struct OnnxPolicyMetadata {
    pub schema_version: u32,
    pub format: String,
    pub artifact: String,
    pub artifact_sha256: String,
    pub input_name: String,
    pub output_name: String,
    pub encoder_shape: [usize; 2],
    pub action_space: usize,
    pub max_batch: usize,
    pub checkpoint_global_step: u64,
    pub checkpoint_samples_seen: u64,
    pub weight_source: String,
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
        let view = ArrayView3::from_shape((batch, OBS_CHANNELS, TILE_WIDTH), obs)
            .map_err(|err| OnnxPolicyError::Invalid(err.to_string()))?;
        let input = TensorRef::from_array_view(view)?;
        let outputs = self
            .session
            .run(inputs![self.metadata.input_name.as_str() => input])?;
        let value = outputs
            .get(self.metadata.output_name.as_str())
            .ok_or_else(|| {
                OnnxPolicyError::Invalid(format!("missing output {}", self.metadata.output_name))
            })?;
        let (shape, data) = value.try_extract_tensor::<f32>()?;
        if shape.as_ref() != [batch as i64, HYDRA_ACTION_SPACE as i64] {
            return Err(OnnxPolicyError::Invalid(format!(
                "output shape {:?} != [{batch}, {HYDRA_ACTION_SPACE}]",
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
}

fn validate_metadata(metadata: &OnnxPolicyMetadata) -> Result<(), OnnxPolicyError> {
    if metadata.schema_version != SCHEMA {
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
    if metadata.input_name != "obs" || metadata.output_name != "policy_logits" {
        return invalid("input/output names must be obs/policy_logits".to_owned());
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
    Ok(())
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
