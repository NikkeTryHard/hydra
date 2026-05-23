use std::path::PathBuf;

use crate::preflight::{PreflightBenchTuple, PreflightConfig, ProbeKind};

use super::{
    BcBackend, BenchmarkBaselineCliOptions, BenchmarkBaselineSource,
    ExperimentalBackboneProfileConfig, ExperimentalTrainBackend, PreflightCliOptions,
    PreflightProfile, ProbeBatchChildRequest, ProbeChildRequest, ProbeCliRequest,
    ProbeSingleChildRequest, PythonLearnerCliOptions, PythonLearnerInput, PythonLearnerVariant,
    PythonResidualProfileConfig, TrainCli, default_device, default_preflight_config_for_profile,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreflightModeArg {
    Safe,
    Unsafe,
}

fn normalize_long_flag(arg: &str) -> String {
    arg.replace('_', "-")
}

fn parse_preflight_mode(value: &str) -> Result<PreflightModeArg, String> {
    match value {
        "safe" => Ok(PreflightModeArg::Safe),
        "unsafe" => Ok(PreflightModeArg::Unsafe),
        _ => Err(format!(
            "unsupported --preflight-mode value '{value}'; expected safe or unsafe"
        )),
    }
}

fn parse_benchmark_source(value: &str) -> Result<BenchmarkBaselineSource, String> {
    match value {
        "mjai" | "raw" | "raw_mjai" => Ok(BenchmarkBaselineSource::Mjai),
        "bc-shards" | "bc_shards" | "shards" => Ok(BenchmarkBaselineSource::BcShards),
        "both" => Ok(BenchmarkBaselineSource::Both),
        _ => Err(format!(
            "unsupported --bench-source value '{value}'; expected mjai, bc-shards, or both"
        )),
    }
}

fn parse_experimental_backend(value: &str) -> Result<ExperimentalTrainBackend, String> {
    match value {
        "libtorch" | "tch" => Ok(ExperimentalTrainBackend::LibTorch),
        "burn-cuda" | "burn_cuda" | "cuda" => Ok(ExperimentalTrainBackend::BurnCuda),
        _ => Err(format!(
            "unsupported --experimental-backend value '{value}'; expected libtorch or burn-cuda"
        )),
    }
}

fn parse_bc_backend(value: &str) -> Result<BcBackend, String> {
    match value {
        "python" | "pytorch" => Ok(BcBackend::Python),
        "rust-burn" | "rust_burn" | "rust" | "burn" => Ok(BcBackend::RustBurn),
        _ => Err(format!(
            "unsupported --bc-backend value '{value}'; expected python or rust-burn"
        )),
    }
}

fn parse_python_variant(value: &str) -> Result<PythonLearnerVariant, String> {
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

fn parse_python_residual_profile(value: &str) -> Result<PythonResidualProfileConfig, String> {
    match value {
        "mish_se" => Ok(PythonResidualProfileConfig::MishSe),
        "silu_se" => Ok(PythonResidualProfileConfig::SiluSe),
        "relu_se" => Ok(PythonResidualProfileConfig::ReluSe),
        "mish_no_se" => Ok(PythonResidualProfileConfig::MishNoSe),
        "relu_no_se" => Ok(PythonResidualProfileConfig::ReluNoSe),
        "relu_no_norm_no_se" => Ok(PythonResidualProfileConfig::ReluNoNormNoSe),
        _ => Err(format!(
            "unsupported --python-residual-profile value '{value}'; expected mish_se, silu_se, relu_se, mish_no_se, relu_no_se, or relu_no_norm_no_se"
        )),
    }
}

fn parse_experimental_backbone_profile(
    value: &str,
) -> Result<ExperimentalBackboneProfileConfig, String> {
    let mut profile = ExperimentalBackboneProfileConfig {
        activation: super::BackboneActivationConfig::Mish,
        se_every_n: super::default_backbone_se_every_n(),
        norm: super::BackboneNormConfig::Both,
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
                    "mish" => super::BackboneActivationConfig::Mish,
                    "silu" => super::BackboneActivationConfig::Silu,
                    "relu" => super::BackboneActivationConfig::Relu,
                    _ => return Err(format!("unsupported backbone activation '{raw}'")),
                };
            }
            "se_every_n" | "se-every-n" => {
                profile.se_every_n =
                    parse_usize_flag_allowing_zero("se_every_n", Some(raw.to_string()), false)?;
            }
            "norm" => {
                profile.norm = match raw {
                    "both" => super::BackboneNormConfig::Both,
                    "first_only" | "first-only" => super::BackboneNormConfig::FirstOnly,
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

fn parse_preflight_profile(value: &str) -> Result<PreflightProfile, String> {
    match value {
        "default" => Ok(PreflightProfile::Default),
        "fast-repeated-run" => Ok(PreflightProfile::FastRepeatedRun),
        _ => Err(format!(
            "unsupported --pf-profile value '{value}'; expected default or fast-repeated-run"
        )),
    }
}

fn parse_positive_usize_text(flag: &str, raw: &str) -> Result<usize, String> {
    if raw.is_empty() || raw.starts_with('+') || raw.starts_with('-') {
        return Err(format!(
            "invalid {flag} value '{raw}': expected positive integer"
        ));
    }
    let value = raw
        .parse::<usize>()
        .map_err(|err| format!("invalid {flag} value '{raw}': {err}"))?;
    if value == 0 {
        return Err(format!("{flag} must be greater than 0"));
    }
    Ok(value)
}

fn parse_usize_flag_allowing_zero(
    flag: &str,
    value: Option<String>,
    allow_zero: bool,
) -> Result<usize, String> {
    let raw = value.ok_or_else(|| format!("missing value for {flag}"))?;
    if raw != raw.trim()
        || raw.contains(char::is_whitespace)
        || raw.starts_with('+')
        || raw.starts_with('-')
    {
        return Err(format!("invalid {flag} value '{raw}': expected integer"));
    }
    let parsed = raw
        .parse::<usize>()
        .map_err(|err| format!("invalid {flag} value '{raw}': {err}"))?;
    if !allow_zero && parsed == 0 {
        return Err(format!("{flag} must be greater than 0"));
    }
    Ok(parsed)
}

fn parse_u64_flag_allowing_zero(
    flag: &str,
    value: Option<String>,
    allow_zero: bool,
) -> Result<u64, String> {
    let raw = value.ok_or_else(|| format!("missing value for {flag}"))?;
    if raw != raw.trim()
        || raw.contains(char::is_whitespace)
        || raw.starts_with('+')
        || raw.starts_with('-')
    {
        return Err(format!("invalid {flag} value '{raw}': expected integer"));
    }
    let parsed = raw
        .parse::<u64>()
        .map_err(|err| format!("invalid {flag} value '{raw}': {err}"))?;
    if !allow_zero && parsed == 0 {
        return Err(format!("{flag} must be greater than 0"));
    }
    Ok(parsed)
}

fn parse_f64_flag(flag: &str, value: Option<String>) -> Result<f64, String> {
    let raw = value.ok_or_else(|| format!("missing value for {flag}"))?;
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed.contains(char::is_whitespace) {
        return Err(format!(
            "invalid {flag} value '{raw}': expected finite number"
        ));
    }
    let parsed = trimmed
        .parse::<f64>()
        .map_err(|err| format!("invalid {flag} value '{raw}': {err}"))?;
    if !parsed.is_finite() {
        return Err(format!("{flag} must be finite"));
    }
    Ok(parsed)
}

fn parse_bool_flag(flag: &str, value: Option<String>) -> Result<bool, String> {
    let raw = value.ok_or_else(|| format!("missing value for {flag}"))?;
    match raw.as_str() {
        "0" => Ok(false),
        "1" => Ok(true),
        _ => Err(format!("invalid {flag} value '{raw}'; expected 0 or 1")),
    }
}

fn parse_usize_range_list(flag: &str, raw: &str) -> Result<Vec<usize>, String> {
    let mut values = Vec::new();
    for segment in raw.trim().split(',') {
        let atom = segment.trim();
        if atom.is_empty() {
            return Err(format!("invalid {flag} value '{raw}': empty range segment"));
        }
        if atom.contains(char::is_whitespace) {
            return Err(format!(
                "invalid {flag} value '{raw}': whitespace inside range atom"
            ));
        }
        parse_usize_range_atom(flag, atom, &mut values)?;
    }
    Ok(values)
}

fn parse_usize_range_atom(flag: &str, atom: &str, out: &mut Vec<usize>) -> Result<(), String> {
    if let Some((start, rest)) = atom.split_once('-') {
        let (end, step, multiply) = if let Some((end, step)) = rest.split_once('+') {
            (end, parse_positive_usize_text(flag, step)?, false)
        } else if let Some((end, factor)) = rest.split_once('*') {
            (end, parse_positive_usize_text(flag, factor)?, true)
        } else {
            (rest, 1, false)
        };
        let start = parse_positive_usize_text(flag, start)?;
        let end = parse_positive_usize_text(flag, end)?;
        if start > end {
            return Err(format!("invalid {flag} range '{atom}': start exceeds end"));
        }
        if multiply && step == 1 {
            return Err(format!(
                "invalid {flag} range '{atom}': multiplicative step must be greater than 1"
            ));
        }
        let mut current = start;
        while current <= end {
            out.push(current);
            current = if multiply {
                current.checked_mul(step)
            } else {
                current.checked_add(step)
            }
            .ok_or_else(|| format!("invalid {flag} range '{atom}': overflow"))?;
        }
    } else {
        out.push(parse_positive_usize_text(flag, atom)?);
    }
    Ok(())
}

fn parse_f64_list(flag: &str, raw: &str) -> Result<Vec<f64>, String> {
    let mut values = Vec::new();
    for segment in raw.trim().split(',') {
        let atom = segment.trim();
        if atom.is_empty() {
            return Err(format!("invalid {flag} value '{raw}': empty float segment"));
        }
        if atom.contains(char::is_whitespace) {
            return Err(format!(
                "invalid {flag} value '{raw}': whitespace inside float atom"
            ));
        }
        let value = atom
            .parse::<f64>()
            .map_err(|err| format!("invalid {flag} value '{atom}': {err}"))?;
        if !value.is_finite() {
            return Err(format!("{flag} entries must be finite"));
        }
        if value <= 0.0 {
            return Err(format!("{flag} entries must be greater than 0"));
        }
        values.push(value);
    }
    Ok(values)
}

fn parse_preflight_bench_candidate_tuples(raw: &str) -> Result<Vec<PreflightBenchTuple>, String> {
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

pub fn usage(program: &str) -> String {
    format!(
        "Usage:\n  {program} <config.yaml>\n  {program} --experimental-python-learner --bc-shards-manifest <path> --output-dir <dir> [--device <cpu|cuda[:N]>] [--python-variant <eager_fp32|eager_bf16|compile_default|compile_reduce_overhead|compile_max_autotune>] [--python-residual-profile <mish_se|silu_se|relu_se|mish_no_se|relu_no_se|relu_no_norm_no_se>] [--python-warmup <N>] [--python-steps <N>] [--python-compile-fullgraph-check]\n  {program} --benchmark-baseline --bench-source <mjai|bc-shards|both> (--data-dir <dir>|--bc-shards-manifest <path>) [--output-dir <dir>] [--device <cpu|cuda[:N]>] [--bench-max-games <N>] [--bench-steps <N>]\n  {program} --preflight [--device <cpu|cuda[:N]>] [--output-dir <dir>] [--pf-candidate-tuples <batch:ring:threads:prefetch,...>] [--pf-warmup-steps <N>] [--pf-measure-steps <N>] [--pf-repetitions <N>] [--pf-output md]\n  {program} --list-devices\n  {program} <config.yaml> --delta-q-promotion [--delta-q-baseline-checkpoint <path>]\n  {program} <config.yaml> --probe-kind <train|validation|rl_games|rl_microbatch> --probe-candidate-microbatch <N> [--probe-warmup-steps <N>] [--probe-measure-steps <N>]\n"
    )
}

pub fn version(program: &str) -> String {
    format!("{program} {}", env!("CARGO_PKG_VERSION"))
}

fn parse_probe_kind(value: &str) -> Result<ProbeKind, String> {
    match value {
        "train" => Ok(ProbeKind::Train),
        "validation" => Ok(ProbeKind::Validation),
        "rl_games" => Ok(ProbeKind::RlGames),
        "rl_microbatch" => Ok(ProbeKind::RlMicrobatch),
        _ => Err(format!(
            "unsupported --probe-kind value '{value}'; expected train, validation, rl_games, or rl_microbatch"
        )),
    }
}

fn parse_usize_flag(flag: &str, value: Option<String>) -> Result<usize, String> {
    let raw = value.ok_or_else(|| format!("missing value for {flag}"))?;
    raw.parse::<usize>()
        .map_err(|err| format!("invalid {flag} value '{raw}': {err}"))
}

pub fn parse_args<I>(args: I) -> Result<TrainCli, String>
where
    I: IntoIterator<Item = String>,
{
    let mut args = args.into_iter();
    let program = args.next().unwrap_or_else(|| "train".to_string());
    let mut first = args.next().ok_or_else(|| usage(&program))?;
    if first == "--" {
        first = args.next().ok_or_else(|| usage(&program))?;
    }
    if first == "--list-devices" {
        if args.next().is_some() {
            return Err(
                "--list-devices cannot be combined with config path or train mode flags"
                    .to_string(),
            );
        }
        return Ok(TrainCli {
            config_path: None,
            list_devices: true,
            preflight: None,
            benchmark_baseline: None,
            delta_q_promotion: false,
            delta_q_baseline_checkpoint: None,
            probe_only: None,
            probe_child: None,
            experimental_backend: ExperimentalTrainBackend::LibTorch,
            python_learner: None,
            bc_backend: BcBackend::RustBurn,
        });
    }
    let mut config_path = None;
    let mut pending_arg = Some(first);
    let mut probe_kind = None;
    let mut candidate_microbatch = None;
    let mut warmup_steps = None;
    let mut measure_steps = None;
    let mut probe_attempts = None;
    let mut probe_result_path = None;
    let mut probe_results_path = None;
    let mut probe_manifest_cache_path = None;
    let mut probe_discovery_summary_path = None;
    let mut probe_discovery_index_path = None;
    let mut preflight_enabled = false;
    let mut preflight_mode = None;
    let mut preflight_profile = PreflightProfile::Default;
    let mut preflight_config = PreflightConfig::default();
    let mut unsafe_batch_seen = false;
    let mut unsafe_lr_seen = false;
    let mut unsafe_warmup_seen = false;
    let mut preflight_flag_seen = false;
    let mut unsafe_flag_seen = false;
    let mut delta_q_promotion = false;
    let mut delta_q_baseline_checkpoint = None;
    let mut preflight_output_dir = PathBuf::from("preflight_bench");
    let mut preflight_device = default_device();
    let mut benchmark_enabled = false;
    let mut benchmark_source = BenchmarkBaselineSource::Both;
    let mut benchmark_data_dir = None;
    let mut benchmark_bc_shards_manifest = None;
    let mut benchmark_output_dir = PathBuf::from("benchmark_baseline");
    let mut benchmark_device = default_device();
    let mut benchmark_max_games = 5_000usize;
    let mut benchmark_max_train_steps = 30usize;
    let mut benchmark_batch_size = 2048usize;
    let mut benchmark_microbatch_size = 256usize;
    let mut benchmark_validation_microbatch_size = 128usize;
    let mut benchmark_num_threads = 20usize;
    let mut benchmark_train_threads = 8usize;
    let mut benchmark_queue_bound = 256usize;
    let mut benchmark_shard_samples = 100_000usize;
    let mut benchmark_train_fraction = 0.9f32;
    let mut experimental_backend = ExperimentalTrainBackend::LibTorch;
    let mut benchmark_backbone_profile = None;
    let mut python_learner_enabled = false;
    let mut python_output_dir = None;
    let mut python_device = default_device();
    let mut python_variant = PythonLearnerVariant::default();
    let mut python_warmup_steps = 10usize;
    let mut python_steps = 30usize;
    let mut python_full_epoch = false;
    let mut python_checkpoint_out = None;
    let mut python_resume = None;
    let mut python_checkpoint_every_steps = 0usize;
    let mut python_compile_fullgraph_check = false;
    let mut python_oracle_critic_weight = 0.0f64;
    let mut python_safety_residual_weight = 0.0f64;
    let mut python_residual_profile = PythonResidualProfileConfig::default();
    let mut python_log_every_steps = super::default_log_every_n_steps();
    let mut python_keep_step_checkpoints = false;
    let mut python_tensorboard = true;
    let mut python_launch_tensorboard = false;
    let mut python_tensorboard_host = super::default_tensorboard_host();
    let mut python_tensorboard_port = super::default_tensorboard_port();
    let mut python_background = false;
    let mut bc_backend = None;
    while let Some(arg) = pending_arg.take().or_else(|| args.next()) {
        let normalized = normalize_long_flag(&arg);
        if !arg.starts_with('-') {
            if config_path.is_some() {
                return Err(usage(&program));
            }
            config_path = Some(PathBuf::from(arg));
            continue;
        }
        match normalized.as_str() {
            "--help" | "-h" => return Err(usage(&program)),
            "--version" | "-V" => return Err(version(&program)),
            "--list-devices" => {
                return Err(
                    "--list-devices cannot be combined with config path or train mode flags"
                        .to_string(),
                );
            }
            "--benchmark-baseline" | "--auto-benchmark" => benchmark_enabled = true,
            "--data-dir" => {
                benchmark_data_dir = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| "missing value for --data-dir".to_string())?,
                ));
            }
            "--bc-shards-manifest" => {
                let value = PathBuf::from(
                    args.next()
                        .ok_or_else(|| "missing value for --bc-shards-manifest".to_string())?,
                );
                benchmark_bc_shards_manifest = Some(value);
            }
            "--experimental-python-learner" => python_learner_enabled = true,
            "--bc-backend" => {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --bc-backend".to_string())?;
                bc_backend = Some(parse_bc_backend(&value)?);
            }
            "--legacy-rust-bc" => bc_backend = Some(BcBackend::RustBurn),
            "--python-variant" => {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --python-variant".to_string())?;
                python_variant = parse_python_variant(&value)?;
            }
            "--python-warmup" => {
                python_warmup_steps =
                    parse_usize_flag_allowing_zero("--python-warmup", args.next(), true)?;
            }
            "--python-steps" => {
                python_steps =
                    parse_usize_flag_allowing_zero("--python-steps", args.next(), false)?;
            }
            "--python-full-epoch" => python_full_epoch = true,
            "--python-checkpoint-out" => {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --python-checkpoint-out".to_string())?;
                python_checkpoint_out = Some(PathBuf::from(value));
            }
            "--python-resume" => {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --python-resume".to_string())?;
                python_resume = Some(PathBuf::from(value));
            }
            "--python-checkpoint-every-steps" => {
                python_checkpoint_every_steps = parse_usize_flag_allowing_zero(
                    "--python-checkpoint-every-steps",
                    args.next(),
                    true,
                )?;
            }
            "--python-log-every-steps" => {
                python_log_every_steps =
                    parse_usize_flag_allowing_zero("--python-log-every-steps", args.next(), true)?;
            }
            "--python-keep-step-checkpoints" => python_keep_step_checkpoints = true,
            "--python-no-tensorboard" => python_tensorboard = false,
            "--python-launch-tensorboard" => python_launch_tensorboard = true,
            "--python-tensorboard-host" => {
                python_tensorboard_host = args
                    .next()
                    .ok_or_else(|| "missing value for --python-tensorboard-host".to_string())?;
            }
            "--python-tensorboard-port" => {
                let value = parse_usize_flag_allowing_zero(
                    "--python-tensorboard-port",
                    args.next(),
                    false,
                )?;
                python_tensorboard_port = u16::try_from(value)
                    .map_err(|_| "--python-tensorboard-port must be <= 65535".to_string())?;
            }
            "--python-background" => python_background = true,
            "--python-compile-fullgraph-check" => python_compile_fullgraph_check = true,
            "--python-w-oracle-critic" => {
                python_oracle_critic_weight =
                    parse_f64_flag("--python-w-oracle-critic", args.next())?;
            }
            "--python-w-safety-residual" => {
                python_safety_residual_weight =
                    parse_f64_flag("--python-w-safety-residual", args.next())?;
            }
            "--python-residual-profile" => {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --python-residual-profile".to_string())?;
                python_residual_profile = parse_python_residual_profile(&value)?;
            }
            "--bench-source" => {
                benchmark_source = parse_benchmark_source(
                    &args
                        .next()
                        .ok_or_else(|| "missing value for --bench-source".to_string())?,
                )?;
            }
            "--experimental-backend" => {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --experimental-backend".to_string())?;
                experimental_backend = parse_experimental_backend(&value)?;
            }
            "--experimental-backbone-profile" => {
                let value = args.next().ok_or_else(|| {
                    "missing value for --experimental-backbone-profile".to_string()
                })?;
                benchmark_backbone_profile = Some(parse_experimental_backbone_profile(&value)?);
            }
            "--preflight" => preflight_enabled = true,
            "--preflight-mode" => {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --preflight-mode".to_string())?;
                preflight_mode = Some(parse_preflight_mode(&value)?);
            }
            "--pf-profile" => {
                preflight_flag_seen = true;
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --pf-profile".to_string())?;
                preflight_profile = parse_preflight_profile(&value)?;
                preflight_config = default_preflight_config_for_profile(preflight_profile);
                match preflight_mode {
                    Some(PreflightModeArg::Safe) => {
                        preflight_config.tuning_mode = crate::preflight::PreflightTuningMode::Safe
                    }
                    Some(PreflightModeArg::Unsafe) => {
                        preflight_config.tuning_mode = crate::preflight::PreflightTuningMode::Unsafe
                    }
                    None => {}
                }
            }
            "--pf-candidate-microbatch" => {
                return Err("--pf-candidate-microbatch is deprecated for benchmark preflight; use --pf-candidate-tuples <batch:ring:threads:prefetch,...>".to_string());
            }
            "--pf-candidate-tuples" => {
                preflight_flag_seen = true;
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --pf-candidate-tuples".to_string())?;
                preflight_config.bench_candidate_tuples =
                    parse_preflight_bench_candidate_tuples(&value)?;
            }
            "--pf-output" => {
                preflight_flag_seen = true;
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --pf-output".to_string())?;
                if value != "md" {
                    return Err("--pf-output only supports md in benchmark preflight".to_string());
                }
                preflight_config.bench_output = value;
            }
            "--output-dir" => {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --output-dir".to_string())?;
                if preflight_enabled {
                    preflight_flag_seen = true;
                    preflight_output_dir = PathBuf::from(value);
                } else if python_learner_enabled
                    || (benchmark_bc_shards_manifest.is_some() && !benchmark_enabled)
                {
                    python_output_dir = Some(PathBuf::from(value));
                } else if !benchmark_enabled {
                    preflight_flag_seen = true;
                    preflight_output_dir = PathBuf::from(value);
                } else {
                    benchmark_output_dir = PathBuf::from(value);
                }
            }
            "--device" => {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --device".to_string())?;
                if preflight_enabled {
                    preflight_flag_seen = true;
                    preflight_device = value;
                } else if python_learner_enabled
                    || (benchmark_bc_shards_manifest.is_some() && !benchmark_enabled)
                {
                    python_device = value;
                } else if !benchmark_enabled {
                    preflight_flag_seen = true;
                    preflight_device = value;
                } else {
                    benchmark_device = value;
                }
            }
            "--pf-min-microbatch" => {
                preflight_flag_seen = true;
                preflight_config.min_microbatch_size =
                    parse_usize_flag_allowing_zero("--pf-min-microbatch", args.next(), false)?;
            }
            "--pf-allow-explicit-microbatch-override" => {
                preflight_flag_seen = true;
                preflight_config.allow_override_explicit_microbatch =
                    parse_bool_flag("--pf-allow-explicit-microbatch-override", args.next())?;
            }
            "--pf-warmup-steps" => {
                preflight_flag_seen = true;
                preflight_config.warmup_steps =
                    parse_usize_flag_allowing_zero("--pf-warmup-steps", args.next(), false)?;
            }
            "--pf-measure-steps" => {
                preflight_flag_seen = true;
                preflight_config.measure_steps =
                    parse_usize_flag_allowing_zero("--pf-measure-steps", args.next(), false)?;
            }
            "--pf-required-successes" => {
                preflight_flag_seen = true;
                preflight_config.required_successes =
                    parse_usize_flag_allowing_zero("--pf-required-successes", args.next(), false)?;
            }
            "--pf-repetitions" => {
                preflight_flag_seen = true;
                preflight_config.required_successes =
                    parse_usize_flag_allowing_zero("--pf-repetitions", args.next(), false)?;
            }
            "--pf-noise-tolerance" => {
                preflight_flag_seen = true;
                preflight_config.measure_noise_tolerance_ratio =
                    parse_f64_flag("--pf-noise-tolerance", args.next())?;
            }
            "--pf-loader-rounds" => {
                preflight_flag_seen = true;
                preflight_config.loader_runtime_rounds =
                    parse_usize_flag_allowing_zero("--pf-loader-rounds", args.next(), true)?;
            }
            "--pf-loader-tuple-margin" => {
                preflight_flag_seen = true;
                preflight_config.loader_tuple_margin_ratio =
                    parse_f64_flag("--pf-loader-tuple-margin", args.next())?;
            }
            "--pf-loader-extra-samples" => {
                preflight_flag_seen = true;
                preflight_config.loader_tuple_extra_samples =
                    parse_usize_flag_allowing_zero("--pf-loader-extra-samples", args.next(), true)?;
            }
            "--pf-real-benchmark" => {
                preflight_flag_seen = true;
                preflight_config.real_benchmark_enabled =
                    parse_bool_flag("--pf-real-benchmark", args.next())?;
            }
            "--pf-real-benchmark-train-candidates" => {
                preflight_flag_seen = true;
                preflight_config.real_benchmark_train_candidates = parse_usize_flag_allowing_zero(
                    "--pf-real-benchmark-train-candidates",
                    args.next(),
                    false,
                )?;
            }
            "--pf-real-benchmark-validation-candidates" => {
                preflight_flag_seen = true;
                preflight_config.real_benchmark_validation_candidates =
                    parse_usize_flag_allowing_zero(
                        "--pf-real-benchmark-validation-candidates",
                        args.next(),
                        false,
                    )?;
            }
            "--pf-real-benchmark-loader-candidates" => {
                preflight_flag_seen = true;
                preflight_config.real_benchmark_loader_candidates = parse_usize_flag_allowing_zero(
                    "--pf-real-benchmark-loader-candidates",
                    args.next(),
                    false,
                )?;
            }
            "--pf-real-benchmark-max-finalists" => {
                preflight_flag_seen = true;
                preflight_config.real_benchmark_max_finalists = parse_usize_flag_allowing_zero(
                    "--pf-real-benchmark-max-finalists",
                    args.next(),
                    false,
                )?;
            }
            "--pf-real-benchmark-warmup-steps" => {
                preflight_flag_seen = true;
                preflight_config.real_benchmark_warmup_steps = parse_usize_flag_allowing_zero(
                    "--pf-real-benchmark-warmup-steps",
                    args.next(),
                    false,
                )?;
            }
            "--pf-real-benchmark-train-steps" => {
                preflight_flag_seen = true;
                preflight_config.real_benchmark_train_steps = parse_usize_flag_allowing_zero(
                    "--pf-real-benchmark-train-steps",
                    args.next(),
                    false,
                )?;
            }
            "--pf-real-benchmark-tie-margin" => {
                preflight_flag_seen = true;
                preflight_config.real_benchmark_tie_margin_ratio =
                    parse_f64_flag("--pf-real-benchmark-tie-margin", args.next())?;
            }
            "--pf-real-benchmark-extra-finalists" => {
                preflight_flag_seen = true;
                preflight_config.real_benchmark_extra_finalists = parse_usize_flag_allowing_zero(
                    "--pf-real-benchmark-extra-finalists",
                    args.next(),
                    false,
                )?;
            }
            "--pf-finalist-margin" => {
                preflight_flag_seen = true;
                preflight_config.finalist_margin_ratio =
                    parse_f64_flag("--pf-finalist-margin", args.next())?;
            }
            "--pf-finalist-max-candidates" => {
                preflight_flag_seen = true;
                preflight_config.finalist_max_candidates = parse_usize_flag_allowing_zero(
                    "--pf-finalist-max-candidates",
                    args.next(),
                    false,
                )?;
            }
            "--pf-finalist-extra-measure-steps" => {
                preflight_flag_seen = true;
                preflight_config.finalist_extra_measure_steps = parse_usize_flag_allowing_zero(
                    "--pf-finalist-extra-measure-steps",
                    args.next(),
                    false,
                )?;
            }
            "--pf-finalist-extra-successes" => {
                preflight_flag_seen = true;
                preflight_config.finalist_extra_successes = parse_usize_flag_allowing_zero(
                    "--pf-finalist-extra-successes",
                    args.next(),
                    false,
                )?;
            }
            "--pf-target-warmup-seconds" => {
                preflight_flag_seen = true;
                preflight_config.target_warmup_seconds =
                    parse_f64_flag("--pf-target-warmup-seconds", args.next())?;
            }
            "--pf-target-measure-seconds" => {
                preflight_flag_seen = true;
                preflight_config.target_measure_seconds =
                    parse_f64_flag("--pf-target-measure-seconds", args.next())?;
            }
            "--pf-max-adaptive-warmup-steps" => {
                preflight_flag_seen = true;
                preflight_config.max_adaptive_warmup_steps = parse_usize_flag_allowing_zero(
                    "--pf-max-adaptive-warmup-steps",
                    args.next(),
                    false,
                )?;
            }
            "--pf-max-adaptive-measure-steps" => {
                preflight_flag_seen = true;
                preflight_config.max_adaptive_measure_steps = parse_usize_flag_allowing_zero(
                    "--pf-max-adaptive-measure-steps",
                    args.next(),
                    false,
                )?;
            }
            "--pf-local-refinement" => {
                preflight_flag_seen = true;
                preflight_config.local_refinement_enabled =
                    parse_bool_flag("--pf-local-refinement", args.next())?;
            }
            "--pf-local-refinement-max-candidates" => {
                preflight_flag_seen = true;
                preflight_config.local_refinement_max_candidates = parse_usize_flag_allowing_zero(
                    "--pf-local-refinement-max-candidates",
                    args.next(),
                    false,
                )?;
            }
            "--pf-local-refinement-min-gap" => {
                preflight_flag_seen = true;
                preflight_config.local_refinement_min_gap = parse_usize_flag_allowing_zero(
                    "--pf-local-refinement-min-gap",
                    args.next(),
                    false,
                )?;
            }
            "--pf-local-refinement-extra-measure-steps" => {
                preflight_flag_seen = true;
                preflight_config.local_refinement_extra_measure_steps =
                    parse_usize_flag_allowing_zero(
                        "--pf-local-refinement-extra-measure-steps",
                        args.next(),
                        false,
                    )?;
            }
            "--pf-search-coordinate-rounds" => {
                preflight_flag_seen = true;
                preflight_config.search_coordinate_rounds = parse_usize_flag_allowing_zero(
                    "--pf-search-coordinate-rounds",
                    args.next(),
                    false,
                )?;
            }
            "--pf-search-top-k" => {
                preflight_flag_seen = true;
                preflight_config.search_top_k =
                    parse_usize_flag_allowing_zero("--pf-search-top-k", args.next(), false)?;
            }
            "--pf-fast-repeated-run-window" => {
                preflight_flag_seen = true;
                preflight_config.fast_repeated_run_candidate_window =
                    parse_usize_flag_allowing_zero(
                        "--pf-fast-repeated-run-window",
                        args.next(),
                        false,
                    )?;
            }
            "--pf-unsafe-batch-size" => {
                preflight_flag_seen = true;
                unsafe_flag_seen = true;
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --pf-unsafe-batch-size".to_string())?;
                if !unsafe_batch_seen {
                    preflight_config.unsafe_candidate_batch_sizes.clear();
                    unsafe_batch_seen = true;
                }
                preflight_config
                    .unsafe_candidate_batch_sizes
                    .extend(parse_usize_range_list("--pf-unsafe-batch-size", &value)?);
            }
            "--pf-unsafe-lr-scale" => {
                preflight_flag_seen = true;
                unsafe_flag_seen = true;
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --pf-unsafe-lr-scale".to_string())?;
                if !unsafe_lr_seen {
                    preflight_config.unsafe_candidate_lr_scales.clear();
                    unsafe_lr_seen = true;
                }
                preflight_config
                    .unsafe_candidate_lr_scales
                    .extend(parse_f64_list("--pf-unsafe-lr-scale", &value)?);
            }
            "--pf-unsafe-warmup-steps" => {
                preflight_flag_seen = true;
                unsafe_flag_seen = true;
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --pf-unsafe-warmup-steps".to_string())?;
                if !unsafe_warmup_seen {
                    preflight_config.unsafe_candidate_warmup_steps.clear();
                    unsafe_warmup_seen = true;
                }
                preflight_config
                    .unsafe_candidate_warmup_steps
                    .extend(parse_usize_range_list("--pf-unsafe-warmup-steps", &value)?);
            }
            "--bench-max-games" => {
                benchmark_max_games =
                    parse_usize_flag_allowing_zero("--bench-max-games", args.next(), false)?;
            }
            "--bench-steps" => {
                benchmark_max_train_steps =
                    parse_usize_flag_allowing_zero("--bench-steps", args.next(), false)?;
            }
            "--bench-batch-size" => {
                benchmark_batch_size =
                    parse_usize_flag_allowing_zero("--bench-batch-size", args.next(), false)?;
            }
            "--bench-microbatch-size" => {
                benchmark_microbatch_size =
                    parse_usize_flag_allowing_zero("--bench-microbatch-size", args.next(), false)?;
            }
            "--bench-validation-microbatch-size" => {
                benchmark_validation_microbatch_size = parse_usize_flag_allowing_zero(
                    "--bench-validation-microbatch-size",
                    args.next(),
                    false,
                )?;
            }
            "--bench-num-threads" => {
                benchmark_num_threads =
                    parse_usize_flag_allowing_zero("--bench-num-threads", args.next(), false)?;
            }
            "--bench-train-threads" => {
                benchmark_train_threads =
                    parse_usize_flag_allowing_zero("--bench-train-threads", args.next(), false)?;
            }
            "--bench-queue-bound" => {
                benchmark_queue_bound =
                    parse_usize_flag_allowing_zero("--bench-queue-bound", args.next(), false)?;
            }
            "--bench-shard-samples" => {
                benchmark_shard_samples =
                    parse_usize_flag_allowing_zero("--bench-shard-samples", args.next(), false)?;
            }
            "--bench-train-fraction" => {
                benchmark_train_fraction = args
                    .next()
                    .ok_or_else(|| "missing value for --bench-train-fraction".to_string())?
                    .parse::<f32>()
                    .map_err(|err| format!("invalid --bench-train-fraction: {err}"))?;
            }
            "--pf-rl-min-free-memory-bytes" => {
                preflight_flag_seen = true;
                preflight_config.rl_probe_min_free_memory_bytes = parse_u64_flag_allowing_zero(
                    "--pf-rl-min-free-memory-bytes",
                    args.next(),
                    true,
                )?;
            }
            "--pf-rl-memory-headroom-ratio" => {
                preflight_flag_seen = true;
                preflight_config.rl_probe_memory_headroom_ratio =
                    parse_f64_flag("--pf-rl-memory-headroom-ratio", args.next())?;
            }
            "--pf-rl-growth-safety-factor" => {
                preflight_flag_seen = true;
                preflight_config.rl_probe_growth_safety_factor =
                    parse_f64_flag("--pf-rl-growth-safety-factor", args.next())?;
            }
            "--delta-q-promotion" => delta_q_promotion = true,
            "--delta-q-baseline-checkpoint" => {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --delta-q-baseline-checkpoint".to_string())?;
                delta_q_baseline_checkpoint = Some(PathBuf::from(value));
            }
            "--probe-kind" => {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --probe-kind".to_string())?;
                probe_kind = Some(parse_probe_kind(&value)?);
            }
            "--probe-candidate-microbatch" => {
                candidate_microbatch = Some(parse_usize_flag(
                    "--probe-candidate-microbatch",
                    args.next(),
                )?);
            }
            "--probe-warmup-steps" => {
                warmup_steps = Some(parse_usize_flag("--probe-warmup-steps", args.next())?);
            }
            "--probe-measure-steps" => {
                measure_steps = Some(parse_usize_flag("--probe-measure-steps", args.next())?);
            }
            "--probe-attempts" => {
                probe_attempts = Some(parse_usize_flag("--probe-attempts", args.next())?);
            }
            "--probe-result-path" => {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --probe-result-path".to_string())?;
                probe_result_path = Some(PathBuf::from(value));
            }
            "--probe-results-path" => {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --probe-results-path".to_string())?;
                probe_results_path = Some(PathBuf::from(value));
            }
            "--probe-manifest-cache-path" => {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --probe-manifest-cache-path".to_string())?;
                probe_manifest_cache_path = Some(PathBuf::from(value));
            }
            "--probe-discovery-summary-path" => {
                let value = args.next().ok_or_else(|| {
                    "missing value for --probe-discovery-summary-path".to_string()
                })?;
                probe_discovery_summary_path = Some(PathBuf::from(value));
            }
            "--probe-discovery-index-path" => {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --probe-discovery-index-path".to_string())?;
                probe_discovery_index_path = Some(PathBuf::from(value));
            }
            _ => return Err(usage(&program)),
        }
    }

    let bc_backend = bc_backend.unwrap_or_else(|| {
        if benchmark_bc_shards_manifest.is_some() && !benchmark_enabled {
            BcBackend::Python
        } else {
            BcBackend::RustBurn
        }
    });
    let route_python_bc = python_learner_enabled || bc_backend == BcBackend::Python;

    if preflight_flag_seen && !preflight_enabled {
        return Err("--pf-* flags require --preflight".to_string());
    }
    let python_learner = if route_python_bc {
        if config_path.is_some() {
            return Err("Python BC learner mode does not accept a config path".to_string());
        }
        let bc_shards_manifest = benchmark_bc_shards_manifest
            .clone()
            .ok_or_else(|| "Python BC learner requires --bc-shards-manifest <path>".to_string())?;
        let output_dir = python_output_dir
            .clone()
            .unwrap_or_else(|| preflight_output_dir.clone());
        Some(PythonLearnerCliOptions {
            bc_shards_manifest: bc_shards_manifest.clone(),
            input: PythonLearnerInput::BcShards {
                manifest: bc_shards_manifest.clone(),
            },
            output_dir,
            device: python_device.clone(),
            batch_size: 2048,
            microbatch_size: 1024,
            variant: python_variant,
            residual_profile: python_residual_profile,
            hidden: 256,
            blocks: 10,
            bottleneck: 64,
            warmup_steps: python_warmup_steps,
            steps: Some(python_steps),
            full_epoch: python_full_epoch,
            validation_steps: 0,
            validation_every: 0,
            checkpoint_out: python_checkpoint_out.clone(),
            resume: python_resume.clone(),
            checkpoint_every_steps: python_checkpoint_every_steps,
            log_every_steps: python_log_every_steps,
            keep_step_checkpoints: python_keep_step_checkpoints,
            tensorboard: python_tensorboard,
            launch_tensorboard: python_launch_tensorboard,
            tensorboard_host: python_tensorboard_host.clone(),
            tensorboard_port: python_tensorboard_port,
            background: python_background,
            learning_rate: super::default_bc_learning_rate(),
            weight_decay: f64::from(super::default_bc_weight_decay()),
            compile_fullgraph_check: python_compile_fullgraph_check,
            oracle_critic_weight: python_oracle_critic_weight,
            safety_residual_weight: python_safety_residual_weight,
        })
    } else {
        if python_output_dir.is_some()
            || python_checkpoint_out.is_some()
            || python_resume.is_some()
            || python_checkpoint_every_steps != 0
            || python_log_every_steps != super::default_log_every_n_steps()
            || python_keep_step_checkpoints
            || !python_tensorboard
            || python_launch_tensorboard
            || python_tensorboard_host != super::default_tensorboard_host()
            || python_tensorboard_port != super::default_tensorboard_port()
            || python_background
            || python_compile_fullgraph_check
            || python_variant != PythonLearnerVariant::default()
            || python_warmup_steps != 10
            || python_steps != 30
            || python_full_epoch
            || python_oracle_critic_weight != 0.0
            || python_safety_residual_weight != 0.0
            || python_residual_profile != PythonResidualProfileConfig::default()
        {
            return Err("--python-* flags require Python BC backend".to_string());
        }
        None
    };

    let preflight = if preflight_enabled {
        if config_path.is_some() {
            return Err(
                "--preflight does not accept a config path; pass benchmark flags explicitly"
                    .to_string(),
            );
        }
        let mode = preflight_mode.unwrap_or(PreflightModeArg::Safe);
        match mode {
            PreflightModeArg::Safe => {
                if unsafe_flag_seen {
                    return Err("unsafe --pf-* flags require --preflight-mode unsafe".to_string());
                }
                preflight_config.tuning_mode = crate::preflight::PreflightTuningMode::Safe;
            }
            PreflightModeArg::Unsafe => {
                preflight_config.tuning_mode = crate::preflight::PreflightTuningMode::Unsafe;
            }
        }
        super::validate_preflight_config(&preflight_config)?;
        Some(PreflightCliOptions {
            preflight_config,
            profile: preflight_profile,
            output_dir: preflight_output_dir.clone(),
            device: preflight_device.clone(),
            bc_shards_manifest_path: benchmark_bc_shards_manifest.clone(),
            bc_backend,
            python_variant,
        })
    } else {
        None
    };
    let benchmark_baseline = if benchmark_enabled {
        if config_path.is_some() {
            return Err("--benchmark-baseline does not accept a config path".to_string());
        }
        if preflight_enabled {
            return Err("--benchmark-baseline cannot be combined with --preflight".to_string());
        }
        if matches!(experimental_backend, ExperimentalTrainBackend::BurnCuda) {
            if !(0.0..=1.0).contains(&benchmark_train_fraction) || benchmark_train_fraction == 0.0 {
                return Err(
                    "--bench-train-fraction must be greater than 0 and at most 1 for burn-cuda probe"
                        .to_string(),
                );
            }
        } else if !(0.0..1.0).contains(&benchmark_train_fraction) {
            return Err(
                "--bench-train-fraction must be greater than 0 and less than 1".to_string(),
            );
        }
        let needs_mjai = matches!(
            benchmark_source,
            BenchmarkBaselineSource::Mjai | BenchmarkBaselineSource::Both
        );
        let needs_existing_shards = matches!(benchmark_source, BenchmarkBaselineSource::BcShards);
        let data_dir = benchmark_data_dir.clone();
        let bc_shards_manifest_path = benchmark_bc_shards_manifest.clone();
        if needs_mjai && data_dir.is_none() {
            return Err("--bench-source mjai/both requires --data-dir <dir>".to_string());
        }
        if needs_existing_shards && bc_shards_manifest_path.is_none() {
            return Err(
                "--bench-source bc-shards requires --bc-shards-manifest <path>".to_string(),
            );
        }
        Some(BenchmarkBaselineCliOptions {
            data_dir,
            bc_shards_manifest_path,
            source: benchmark_source,
            output_dir: benchmark_output_dir.clone(),
            device: benchmark_device.clone(),
            max_games: benchmark_max_games,
            max_train_steps: benchmark_max_train_steps,
            batch_size: benchmark_batch_size,
            microbatch_size: benchmark_microbatch_size,
            validation_microbatch_size: benchmark_validation_microbatch_size,
            num_threads: benchmark_num_threads,
            train_threads: benchmark_train_threads,
            queue_bound: benchmark_queue_bound,
            shard_samples: benchmark_shard_samples,
            train_fraction: benchmark_train_fraction,
            experimental_backend,
            experimental_backbone_profile: benchmark_backbone_profile.clone(),
        })
    } else {
        if (benchmark_data_dir.is_some()
            || (benchmark_bc_shards_manifest.is_some() && bc_backend != BcBackend::RustBurn))
            && python_learner.is_none()
        {
            return Err(
                "--data-dir/--bc-shards-manifest requires --benchmark-baseline".to_string(),
            );
        }
        None
    };

    if preflight.is_some()
        && (probe_kind.is_some()
            || probe_result_path.is_some()
            || probe_results_path.is_some()
            || probe_attempts.is_some()
            || delta_q_promotion
            || delta_q_baseline_checkpoint.is_some())
    {
        return Err(format!(
            "{}\n--preflight cannot be combined with probe-only flags",
            usage(&program)
        ));
    }
    if benchmark_baseline.is_some()
        && (python_learner.is_some()
            || probe_kind.is_some()
            || probe_result_path.is_some()
            || probe_results_path.is_some()
            || probe_attempts.is_some()
            || delta_q_promotion
            || delta_q_baseline_checkpoint.is_some())
    {
        return Err(format!(
            "{}\n--benchmark-baseline cannot be combined with probe-only or promotion flags",
            usage(&program)
        ));
    }
    if delta_q_promotion
        && (probe_kind.is_some()
            || probe_result_path.is_some()
            || probe_results_path.is_some()
            || probe_attempts.is_some())
    {
        return Err(format!(
            "{}\n--delta-q-promotion cannot be combined with probe-only flags",
            usage(&program)
        ));
    }
    if delta_q_baseline_checkpoint.is_some() && !delta_q_promotion {
        return Err(format!(
            "{}\n--delta-q-baseline-checkpoint requires --delta-q-promotion",
            usage(&program)
        ));
    }
    if probe_result_path.is_some() && (probe_results_path.is_some() || probe_attempts.is_some()) {
        return Err(format!(
            "{}\ninternal probe child mode cannot combine --probe-result-path with --probe-attempts/--probe-results-path",
            usage(&program)
        ));
    }
    if probe_results_path.is_some() ^ probe_attempts.is_some() {
        return Err(format!(
            "{}\ninternal probe batch child mode requires both --probe-attempts and --probe-results-path",
            usage(&program)
        ));
    }
    if probe_discovery_summary_path.is_some() ^ probe_discovery_index_path.is_some() {
        return Err(format!(
            "{}\ninternal probe child mode requires both --probe-discovery-summary-path and --probe-discovery-index-path",
            usage(&program)
        ));
    }
    match (
        probe_kind,
        candidate_microbatch,
        probe_result_path,
        probe_results_path,
        probe_attempts,
    ) {
        (None, None, None, None, None) => Ok(TrainCli {
            config_path,
            list_devices: false,
            preflight,
            benchmark_baseline,
            delta_q_promotion,
            delta_q_baseline_checkpoint,
            probe_only: None,
            probe_child: None,
            experimental_backend,
            python_learner,
            bc_backend,
        }),
        (Some(kind), Some(candidate_microbatch), None, None, None) => Ok(TrainCli {
            config_path: Some(config_path.ok_or_else(|| usage(&program))?),
            list_devices: false,
            preflight: None,
            benchmark_baseline: None,
            delta_q_promotion: false,
            delta_q_baseline_checkpoint: None,
            probe_only: Some(ProbeCliRequest {
                kind,
                candidate_microbatch,
                warmup_steps,
                measure_steps,
            }),
            probe_child: None,
            experimental_backend,
            python_learner: None,
            bc_backend: BcBackend::RustBurn,
        }),
        (Some(kind), Some(candidate_microbatch), Some(result_path), None, None) => Ok(TrainCli {
            config_path: Some(config_path.ok_or_else(|| usage(&program))?),
            list_devices: false,
            preflight: None,
            benchmark_baseline: None,
            delta_q_promotion: false,
            delta_q_baseline_checkpoint: None,
            probe_only: None,
            probe_child: Some(ProbeChildRequest::Single(ProbeSingleChildRequest {
                request: ProbeCliRequest {
                    kind,
                    candidate_microbatch,
                    warmup_steps,
                    measure_steps,
                },
                result_path,
                manifest_cache_path: probe_manifest_cache_path,
                discovery_summary_path: probe_discovery_summary_path.clone(),
                discovery_index_path: probe_discovery_index_path.clone(),
            })),
            experimental_backend,
            python_learner: None,
            bc_backend: BcBackend::RustBurn,
        }),
        (Some(kind), Some(candidate_microbatch), None, Some(results_path), Some(attempts)) => {
            Ok(TrainCli {
                config_path: Some(config_path.ok_or_else(|| usage(&program))?),
                list_devices: false,
                preflight: None,
                benchmark_baseline: None,
                delta_q_promotion: false,
                delta_q_baseline_checkpoint: None,
                probe_only: None,
                probe_child: Some(ProbeChildRequest::Batch(ProbeBatchChildRequest {
                    request: ProbeCliRequest {
                        kind,
                        candidate_microbatch,
                        warmup_steps,
                        measure_steps,
                    },
                    attempts,
                    results_path,
                    manifest_cache_path: probe_manifest_cache_path,
                    discovery_summary_path: probe_discovery_summary_path,
                    discovery_index_path: probe_discovery_index_path,
                })),
                experimental_backend,
                python_learner: None,
                bc_backend: BcBackend::RustBurn,
            })
        }
        _ => Err(format!(
            "{}\nprobe-only mode requires both --probe-kind and --probe-candidate-microbatch",
            usage(&program)
        )),
    }
}
