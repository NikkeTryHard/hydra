use super::OnnxPolicyError;

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
