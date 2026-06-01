use std::{fs, path::Path};

use hydra_core::{action::HYDRA_ACTION_SPACE, encoder::OBS_SIZE};
use ndarray::ArrayView3;
use ort::{
    ep::CUDA,
    inputs,
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};
use sha2::{Digest, Sha256};

use super::device::OnnxPolicyDevice;
use super::extract::{extract_logits, extract_values};
use super::metadata::{OBS_CHANNELS, PPO_POLICY_VALUE_SCHEMA, TILE_WIDTH, validate_metadata};
use super::{OnnxPolicyError, OnnxPolicyMetadata, PolicyValueBatch};

pub struct OnnxPolicyRuntime {
    metadata: OnnxPolicyMetadata,
    session: Session,
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
        let values = extract_values(&outputs, &value_name, batch)?;
        Ok(PolicyValueBatch { logits, values })
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
