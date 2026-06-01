use crate::preflight::PreflightBenchTuple;

use super::super::{
    BcBackend, BenchmarkBaselineSource, ExperimentalBackboneProfileConfig,
    ExperimentalTrainBackend, PreflightProfile, default_backbone_se_every_n,
};
use super::common::{parse_positive_usize_text, parse_usize_flag_allowing_zero};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PreflightModeArg {
    Safe,
    Unsafe,
}

pub(super) fn parse_preflight_mode(value: &str) -> Result<PreflightModeArg, String> {
    match value {
        "safe" => Ok(PreflightModeArg::Safe),
        "unsafe" => Ok(PreflightModeArg::Unsafe),
        _ => Err(format!(
            "unsupported --preflight-mode value '{value}'; expected safe or unsafe"
        )),
    }
}

pub(super) fn parse_benchmark_source(value: &str) -> Result<BenchmarkBaselineSource, String> {
    match value {
        "mjai" | "raw" | "raw_mjai" => Ok(BenchmarkBaselineSource::Mjai),
        "bc-shards" | "bc_shards" | "shards" => Ok(BenchmarkBaselineSource::BcShards),
        "both" => Ok(BenchmarkBaselineSource::Both),
        _ => Err(format!(
            "unsupported --bench-source value '{value}'; expected mjai, bc-shards, or both"
        )),
    }
}

pub(super) fn parse_experimental_backend(value: &str) -> Result<ExperimentalTrainBackend, String> {
    match value {
        "libtorch" | "tch" => Ok(ExperimentalTrainBackend::LibTorch),
        "burn-cuda" | "burn_cuda" | "cuda" => Ok(ExperimentalTrainBackend::BurnCuda),
        _ => Err(format!(
            "unsupported --experimental-backend value '{value}'; expected libtorch or burn-cuda"
        )),
    }
}

pub(super) fn parse_bc_backend(value: &str) -> Result<BcBackend, String> {
    match value {
        "python" | "pytorch" => Ok(BcBackend::Python),
        "rust-burn" | "rust_burn" | "rust" | "burn" => Ok(BcBackend::RustBurn),
        _ => Err(format!(
            "unsupported --bc-backend value '{value}'; expected python or rust-burn"
        )),
    }
}

pub(super) fn parse_experimental_backbone_profile(
    value: &str,
) -> Result<ExperimentalBackboneProfileConfig, String> {
    let mut profile = ExperimentalBackboneProfileConfig {
        activation: hydra_train_types::config::BackboneActivationConfig::Mish,
        se_every_n: default_backbone_se_every_n(),
        norm: hydra_train_types::config::BackboneNormConfig::Both,
        num_blocks: None,
        hidden_channels: None,
    };
    for part in value.split(',') {
        let (key, raw) = part.split_once('=').ok_or_else(|| {
            format!("invalid --experimental-backbone-profile segment '{part}'; expected key=value")
        })?;
        match key {
            "activation" => {
                profile.activation = match raw {
                    "mish" => hydra_train_types::config::BackboneActivationConfig::Mish,
                    "silu" => hydra_train_types::config::BackboneActivationConfig::Silu,
                    "relu" => hydra_train_types::config::BackboneActivationConfig::Relu,
                    _ => return Err(format!("unsupported backbone activation '{raw}'")),
                };
            }
            "se_every_n" | "se-every-n" => {
                profile.se_every_n =
                    parse_usize_flag_allowing_zero("se_every_n", Some(raw.to_string()), false)?;
            }
            "norm" => {
                profile.norm = match raw {
                    "both" => hydra_train_types::config::BackboneNormConfig::Both,
                    "first_only" | "first-only" => {
                        hydra_train_types::config::BackboneNormConfig::FirstOnly
                    }
                    _ => return Err(format!("unsupported backbone norm '{raw}'")),
                };
            }
            "blocks" | "num_blocks" | "num-blocks" => {
                profile.num_blocks = Some(parse_usize_flag_allowing_zero(
                    "num_blocks",
                    Some(raw.to_string()),
                    false,
                )?);
            }
            "hidden" | "hidden_channels" | "hidden-channels" => {
                profile.hidden_channels = Some(parse_usize_flag_allowing_zero(
                    "hidden_channels",
                    Some(raw.to_string()),
                    false,
                )?);
            }
            _ => return Err(format!("unsupported backbone profile key '{key}'")),
        }
    }
    Ok(profile)
}

pub(super) fn parse_preflight_profile(value: &str) -> Result<PreflightProfile, String> {
    match value {
        "default" => Ok(PreflightProfile::Default),
        "fast-repeated-run" => Ok(PreflightProfile::FastRepeatedRun),
        _ => Err(format!(
            "unsupported --pf-profile value '{value}'; expected default or fast-repeated-run"
        )),
    }
}

pub(super) fn parse_preflight_bench_candidate_tuples(
    raw: &str,
) -> Result<Vec<PreflightBenchTuple>, String> {
    let mut out = Vec::new();
    for atom in raw.split(',') {
        let atom = atom.trim();
        if atom.is_empty() {
            return Err("--pf-candidate-tuples contains an empty tuple".to_string());
        }
        let mut fields = atom.split(':');
        let batch_size = parse_positive_usize_text(
            "--pf-candidate-tuples batch",
            fields.next().unwrap_or_default(),
        )?;
        let ring_batches = parse_positive_usize_text(
            "--pf-candidate-tuples ring",
            fields.next().unwrap_or_default(),
        )?;
        let loader_threads = parse_positive_usize_text(
            "--pf-candidate-tuples threads",
            fields.next().unwrap_or_default(),
        )?;
        let prefetch_batches = parse_positive_usize_text(
            "--pf-candidate-tuples prefetch",
            fields.next().unwrap_or_default(),
        )?;
        if fields.next().is_some() {
            return Err(format!(
                "invalid --pf-candidate-tuples tuple {atom}: expected batch:ring:threads:prefetch"
            ));
        }
        out.push(PreflightBenchTuple {
            batch_size,
            ring_batches,
            loader_threads,
            prefetch_batches,
        });
    }
    if out.is_empty() {
        return Err("--pf-candidate-tuples must contain at least one tuple".to_string());
    }
    Ok(out)
}
