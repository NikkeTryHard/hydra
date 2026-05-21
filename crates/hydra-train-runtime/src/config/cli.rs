use std::path::PathBuf;

use crate::preflight::{PreflightBenchTuple, PreflightConfig, ProbeKind};

use super::{
    BenchmarkBaselineCliOptions, BenchmarkBaselineSource, ExperimentalTrainBackend,
    PreflightCliOptions, PreflightProfile, ProbeBatchChildRequest, ProbeChildRequest,
    ProbeCliRequest, ProbeSingleChildRequest, TrainCli, default_device,
    default_preflight_config_for_profile,
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
        "Usage:\n  {program} <config.yaml>\n  {program} --benchmark-baseline --bench-source <mjai|bc-shards|both> (--data-dir <dir>|--bc-shards-manifest <path>) [--output-dir <dir>] [--device <cpu|cuda[:N]>] [--bench-max-games <N>] [--bench-steps <N>]\n  {program} --preflight [--device <cpu|cuda[:N]>] [--output-dir <dir>] [--pf-candidate-tuples <batch:ring:threads:prefetch,...>] [--pf-warmup-steps <N>] [--pf-measure-steps <N>] [--pf-repetitions <N>] [--pf-output md]\n  {program} --list-devices\n  {program} <config.yaml> --delta-q-promotion [--delta-q-baseline-checkpoint <path>]\n  {program} <config.yaml> --probe-kind <train|validation|rl_games|rl_microbatch> --probe-candidate-microbatch <N> [--probe-warmup-steps <N>] [--probe-measure-steps <N>]\n"
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
                benchmark_bc_shards_manifest =
                    Some(PathBuf::from(args.next().ok_or_else(|| {
                        "missing value for --bc-shards-manifest".to_string()
                    })?));
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
                if preflight_enabled || !benchmark_enabled {
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
                if preflight_enabled || !benchmark_enabled {
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

    if preflight_flag_seen && !preflight_enabled {
        return Err("--pf-* flags require --preflight".to_string());
    }
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
        })
    } else {
        if benchmark_data_dir.is_some() || benchmark_bc_shards_manifest.is_some() {
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
        && (probe_kind.is_some()
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
            })
        }
        _ => Err(format!(
            "{}\nprobe-only mode requires both --probe-kind and --probe-candidate-microbatch",
            usage(&program)
        )),
    }
}
