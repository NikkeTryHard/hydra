use hydra_core::action::HYDRA_ACTION_SPACE;

use super::OnnxPolicyError;

pub(super) fn extract_logits(
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

pub(super) fn extract_values(
    outputs: &ort::session::SessionOutputs<'_>,
    output_name: &str,
    batch: usize,
) -> Result<Vec<f32>, OnnxPolicyError> {
    let value = outputs
        .get(output_name)
        .ok_or_else(|| OnnxPolicyError::Invalid(format!("missing output {output_name}")))?;
    let (shape, data) = value.try_extract_tensor::<f32>()?;
    if shape.as_ref() != [batch as i64, 1] {
        return Err(OnnxPolicyError::Invalid(format!(
            "value output shape {:?} != [{batch}, 1]",
            shape.as_ref()
        )));
    }
    Ok(data.to_vec())
}
