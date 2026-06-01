use super::{PythonLearnerVariant, PythonResidualProfileConfig};

pub(super) fn parse_python_variant(value: &str) -> Result<PythonLearnerVariant, String> {
    match value {
        "eager_fp32" => Ok(PythonLearnerVariant::EagerFp32),
        "eager_bf16" => Ok(PythonLearnerVariant::EagerBf16),
        "compile_default" => Ok(PythonLearnerVariant::CompileDefault),
        "compile_reduce_overhead" => Ok(PythonLearnerVariant::CompileReduceOverhead),
        "compile_max_autotune" => Ok(PythonLearnerVariant::CompileMaxAutotune),
        _ => Err(format!(
            "unsupported --python-variant value '{value}'; expected eager_fp32, eager_bf16, compile_default, compile_reduce_overhead, or compile_max_autotune"
        )),
    }
}

pub(super) fn parse_python_residual_profile(
    value: &str,
) -> Result<PythonResidualProfileConfig, String> {
    match value {
        "mish_se" => Ok(PythonResidualProfileConfig::MishSe),
        "silu_se" => Ok(PythonResidualProfileConfig::SiluSe),
        "relu_se" => Ok(PythonResidualProfileConfig::ReluSe),
        "mish_no_se" => Ok(PythonResidualProfileConfig::MishNoSe),
        "mish_eca" => Ok(PythonResidualProfileConfig::MishEca),
        "relu_no_se" => Ok(PythonResidualProfileConfig::ReluNoSe),
        "relu_no_norm_no_se" => Ok(PythonResidualProfileConfig::ReluNoNormNoSe),
        _ => Err(format!(
            "unsupported --python-residual-profile value '{value}'; expected mish_se, silu_se, relu_se, mish_no_se, mish_eca, relu_no_se, or relu_no_norm_no_se"
        )),
    }
}
