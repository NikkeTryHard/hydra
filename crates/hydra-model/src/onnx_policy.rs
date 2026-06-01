//! ONNX Runtime policy loader for Python-exported Hydra BC policies.

mod device;
mod extract;
mod metadata;
mod runtime;

pub use device::OnnxPolicyDevice;
pub use metadata::{OnnxOutputMetadata, OnnxPolicyMetadata, OnnxPolicyValueOutputs, OutputDim};
pub use runtime::OnnxPolicyRuntime;

use hydra_core::action::HYDRA_ACTION_SPACE;

use thiserror::Error;

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
pub struct PolicyValueBatch {
    pub logits: Vec<[f32; HYDRA_ACTION_SPACE]>,
    pub values: Vec<f32>,
}

#[cfg(test)]
mod tests;
