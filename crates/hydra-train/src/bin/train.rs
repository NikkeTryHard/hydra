use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use burn::backend::{Autodiff, LibTorch, libtorch::LibTorchDevice};
use burn::module::AutodiffModule;
use burn::optim::{GradientsAccumulator, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use colored::Colorize;
use hydra_train::data::pipeline::{
    StreamingLoaderConfig, scan_data_sources_with_progress, stream_train_epoch, stream_val_pass,
};
use hydra_train::data::sample::collate_samples;
use hydra_train::model::{HydraModel, HydraModelConfig};
use hydra_train::training::bc::{
    BCTrainerConfig, CheckpointMeta, policy_agreement, policy_agreement_counts,
    target_actions_from_policy_target, warmup_then_cosine_lr,
};
use hydra_train::training::losses::{HydraLoss, HydraLossConfig, LossBreakdown};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rayon::ThreadPoolBuilder;
use tboard::EventWriter;

type TrainBackend = Autodiff<LibTorch<f32>>;
type ValidBackend = <TrainBackend as burn::tensor::backend::AutodiffBackend>::InnerBackend;

#[derive(Debug, serde::Deserialize)]
struct TrainConfig {
    data_dir: PathBuf,
    output_dir: PathBuf,
    num_epochs: usize,
    #[serde(default = "default_batch_size")]
    batch_size: usize,
    #[serde(default)]
    microbatch_size: Option<usize>,
    #[serde(default)]
    validation_microbatch_size: Option<usize>,
    #[serde(default = "default_train_fraction")]
    train_fraction: f32,
    #[serde(default = "default_augment")]
    augment: bool,
    resume_checkpoint: Option<PathBuf>,
    #[serde(default = "default_seed")]
    seed: u64,
    #[serde(default)]
    advanced_loss: Option<AdvancedLossConfig>,
    #[serde(default = "default_device")]
    device: String,
    #[serde(default = "default_buffer_games")]
    buffer_games: usize,
    #[serde(default = "default_buffer_samples")]
    buffer_samples: usize,
    #[serde(default)]
    num_threads: Option<usize>,
    #[serde(default = "default_tensorboard")]
    tensorboard: bool,
    #[serde(default = "default_archive_queue_bound")]
    archive_queue_bound: usize,
    #[serde(default = "default_validation_every_n_epochs")]
    validation_every_n_epochs: usize,
    #[serde(default = "default_max_skip_logs_per_source")]
    max_skip_logs_per_source: usize,
    #[serde(default = "default_log_every_n_steps")]
    log_every_n_steps: usize,
    #[serde(default = "default_validate_every_n_steps")]
    validate_every_n_steps: usize,
    #[serde(default = "default_checkpoint_every_n_steps")]
    checkpoint_every_n_steps: usize,
    #[serde(default)]
    max_train_steps: Option<usize>,
    #[serde(default)]
    max_validation_batches: Option<usize>,
    #[serde(default)]
    max_validation_samples: Option<usize>,
}

#[derive(serde::Deserialize, Debug, Clone, Default)]
#[serde(deny_unknown_fields)]
struct AdvancedLossConfig {
    safety_residual: Option<f32>,
    belief_fields: Option<f32>,
    mixture_weight: Option<f32>,
    opponent_hand_type: Option<f32>,
    delta_q: Option<f32>,
}

#[derive(Default, serde::Serialize)]
struct ScalarAverages {
    total_loss: f64,
    policy_agreement: f64,
    loss_policy: f64,
    loss_value: f64,
    loss_grp: f64,
    loss_tenpai: f64,
    loss_danger: f64,
    loss_opp_next: f64,
    loss_score_pdf: f64,
    loss_score_cdf: f64,
    num_batches: usize,
}

#[derive(Clone, Copy, Default)]
struct BatchStats {
    total_loss: f64,
    policy_agreement: f64,
    loss_policy: f64,
    loss_value: f64,
    loss_grp: f64,
    loss_tenpai: f64,
    loss_danger: f64,
    loss_opp_next: f64,
    loss_score_pdf: f64,
    loss_score_cdf: f64,
}

#[derive(serde::Serialize)]
struct EpochLogEntry {
    epoch: usize,
    global_step: usize,
    lr: f64,
    train_total_loss: f64,
    train_policy_agreement: f64,
    train_loss_policy: f64,
    train_loss_value: f64,
    train_loss_grp: f64,
    train_loss_tenpai: f64,
    train_loss_danger: f64,
    train_loss_opp_next: f64,
    train_loss_score_pdf: f64,
    train_loss_score_cdf: f64,
    val_policy_agreement: Option<f64>,
    best_val_agreement: Option<f64>,
    num_batches: usize,
}

#[derive(serde::Serialize)]
struct StepLogEntry {
    global_step: usize,
    epoch: usize,
    lr: f64,
    train_total_loss: f64,
    train_policy_agreement: f64,
    train_loss_policy: f64,
    train_loss_value: f64,
    train_loss_grp: f64,
    train_loss_tenpai: f64,
    train_loss_danger: f64,
    train_loss_opp_next: f64,
    train_loss_score_pdf: f64,
    train_loss_score_cdf: f64,
    val_policy_agreement: Option<f64>,
    best_val_agreement: Option<f64>,
}

struct BannerStats {
    total_sources: usize,
    total_games: usize,
    train_count: usize,
    val_count: usize,
    accum_steps: usize,
    counts_exact: bool,
}

fn default_batch_size() -> usize {
    2048
}

fn default_train_fraction() -> f32 {
    0.9
}

fn default_augment() -> bool {
    true
}

fn default_seed() -> u64 {
    0
}

fn default_device() -> String {
    "cpu".to_string()
}

fn default_buffer_games() -> usize {
    50_000
}

fn default_buffer_samples() -> usize {
    32_768
}

fn default_tensorboard() -> bool {
    true
}

fn default_archive_queue_bound() -> usize {
    128
}

fn default_validation_every_n_epochs() -> usize {
    1
}

fn default_max_skip_logs_per_source() -> usize {
    32
}

fn default_log_every_n_steps() -> usize {
    50
}

fn default_validate_every_n_steps() -> usize {
    200
}

fn default_checkpoint_every_n_steps() -> usize {
    200
}

fn usage(program: &str) -> String {
    format!("Usage: {program} <config.json>")
}

fn parse_args<I>(args: I) -> Result<PathBuf, String>
where
    I: IntoIterator<Item = String>,
{
    let mut args = args.into_iter();
    let program = args.next().unwrap_or_else(|| "train".to_string());
    match (args.next(), args.next()) {
        (Some(config), None) => Ok(PathBuf::from(config)),
        _ => Err(usage(&program)),
    }
}

fn read_config(path: &Path) -> Result<TrainConfig, String> {
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed to read config {}: {err}", path.display()))?;
    serde_json::from_str(&raw)
        .map_err(|err| format!("failed to parse config {}: {err}", path.display()))
}

fn reject_blocked_advanced_loss_presence(field: &str, weight: Option<f32>) -> Result<(), String> {
    match weight {
        Some(_) => Err(format!(
            "advanced_loss.{field} is not supported in train.rs because this BC data path does not safely support it yet"
        )),
        None => Ok(()),
    }
}

fn build_loss_config(
    advanced_loss: Option<&AdvancedLossConfig>,
) -> Result<HydraLossConfig, String> {
    if let Some(cfg) = advanced_loss {
        reject_blocked_advanced_loss_presence("belief_fields", cfg.belief_fields)?;
        reject_blocked_advanced_loss_presence("mixture_weight", cfg.mixture_weight)?;
        reject_blocked_advanced_loss_presence("opponent_hand_type", cfg.opponent_hand_type)?;
        reject_blocked_advanced_loss_presence("delta_q", cfg.delta_q)?;
    }

    let safety_residual = advanced_loss
        .and_then(|cfg| cfg.safety_residual)
        .unwrap_or(0.0);

    let loss_config = HydraLossConfig::new().with_w_safety_residual(safety_residual);
    loss_config
        .validate()
        .map_err(|err| format!("invalid loss config: {err}"))?;
    Ok(loss_config)
}

fn parse_train_device(value: &str) -> LibTorchDevice {
    let value = value.trim().to_ascii_lowercase();
    if value == "cpu" {
        return LibTorchDevice::Cpu;
    }
    if value == "cuda" {
        return LibTorchDevice::Cuda(0);
    }
    if let Some(index) = value.strip_prefix("cuda:") {
        let index = index.parse::<usize>().unwrap_or_else(|_| {
            panic!("unsupported HYDRA_TRAIN_DEVICE={value}; expected cpu, cuda, or cuda:<index>")
        });
        return LibTorchDevice::Cuda(index);
    }
    panic!("unsupported HYDRA_TRAIN_DEVICE={value}; expected cpu, cuda, or cuda:<index>");
}

fn train_device(config_device: &str) -> LibTorchDevice {
    match env::var("HYDRA_TRAIN_DEVICE") {
        Ok(value) => parse_train_device(&value),
        Err(_) => parse_train_device(config_device),
    }
}

fn device_label(config_device: &str) -> String {
    match env::var("HYDRA_TRAIN_DEVICE") {
        Ok(value) => value,
        Err(_) => config_device.to_string(),
    }
}

fn configure_threads(num_threads: Option<usize>) -> Result<(), String> {
    let Some(num_threads) = num_threads else {
        return Ok(());
    };
    if num_threads == 0 {
        return Err("num_threads must be greater than 0".to_string());
    }
    match ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
    {
        Ok(()) => Ok(()),
        Err(err) if err.to_string().contains("initialized") => Ok(()),
        Err(err) => Err(format!("failed to configure rayon thread pool: {err}")),
    }
}

fn validate_config(config: &TrainConfig) -> Result<(), String> {
    if config.num_epochs == 0 {
        return Err("num_epochs must be greater than 0".to_string());
    }
    if config.batch_size == 0 {
        return Err("batch_size must be greater than 0".to_string());
    }
    if config.buffer_games == 0 {
        return Err("buffer_games must be greater than 0".to_string());
    }
    if config.buffer_samples == 0 {
        return Err("buffer_samples must be greater than 0".to_string());
    }
    if config.archive_queue_bound == 0 {
        return Err("archive_queue_bound must be greater than 0".to_string());
    }
    if config.validation_every_n_epochs == 0 {
        return Err("validation_every_n_epochs must be greater than 0".to_string());
    }
    if config.log_every_n_steps == 0 {
        return Err("log_every_n_steps must be greater than 0".to_string());
    }
    if config.validate_every_n_steps == 0 {
        return Err("validate_every_n_steps must be greater than 0".to_string());
    }
    if config.checkpoint_every_n_steps == 0 {
        return Err("checkpoint_every_n_steps must be greater than 0".to_string());
    }
    if let Some(max_train_steps) = config.max_train_steps
        && max_train_steps == 0
    {
        return Err("max_train_steps must be greater than 0 when set".to_string());
    }
    if let Some(max_validation_batches) = config.max_validation_batches
        && max_validation_batches == 0
    {
        return Err("max_validation_batches must be greater than 0 when set".to_string());
    }
    if let Some(max_validation_samples) = config.max_validation_samples
        && max_validation_samples == 0
    {
        return Err("max_validation_samples must be greater than 0 when set".to_string());
    }
    if let Some(microbatch_size) = config.microbatch_size
        && microbatch_size == 0
    {
        return Err("microbatch_size must be greater than 0".to_string());
    }
    if let Some(validation_microbatch_size) = config.validation_microbatch_size
        && validation_microbatch_size == 0
    {
        return Err("validation_microbatch_size must be greater than 0".to_string());
    }
    Ok(())
}

fn train_microbatch_size(config: &TrainConfig) -> usize {
    config.microbatch_size.unwrap_or(config.batch_size)
}

fn validation_microbatch_size(config: &TrainConfig) -> usize {
    config
        .validation_microbatch_size
        .unwrap_or_else(|| train_microbatch_size(config))
}

fn validation_sample_limit(config: &TrainConfig) -> Option<usize> {
    config.max_validation_samples.or_else(|| {
        config
            .max_validation_batches
            .map(|limit| limit.saturating_mul(validation_microbatch_size(config)))
    })
}

fn make_bar(len: u64, template: &str) -> Result<ProgressBar, String> {
    let pb = ProgressBar::new(len);
    let style = ProgressStyle::with_template(template)
        .map_err(|err| format!("failed to build progress style: {err}"))?
        .progress_chars("=> ");
    pb.set_style(style);
    Ok(pb)
}

fn make_spinner(template: &str) -> Result<ProgressBar, String> {
    let pb = ProgressBar::new_spinner();
    let style = ProgressStyle::with_template(template)
        .map_err(|err| format!("failed to build spinner style: {err}"))?
        .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ ");
    pb.set_style(style);
    pb.enable_steady_tick(Duration::from_millis(120));
    Ok(pb)
}

fn scalar1<B: Backend>(tensor: &Tensor<B, 1>) -> f64 {
    tensor.clone().into_scalar().elem::<f64>()
}

impl ScalarAverages {
    fn record_batch(&mut self, batch: BatchStats) {
        self.total_loss += batch.total_loss;
        self.policy_agreement += batch.policy_agreement;
        self.loss_policy += batch.loss_policy;
        self.loss_value += batch.loss_value;
        self.loss_grp += batch.loss_grp;
        self.loss_tenpai += batch.loss_tenpai;
        self.loss_danger += batch.loss_danger;
        self.loss_opp_next += batch.loss_opp_next;
        self.loss_score_pdf += batch.loss_score_pdf;
        self.loss_score_cdf += batch.loss_score_cdf;
        self.num_batches += 1;
    }

    fn finalize(mut self) -> Self {
        if self.num_batches == 0 {
            return self;
        }
        let denom = self.num_batches as f64;
        self.total_loss /= denom;
        self.policy_agreement /= denom;
        self.loss_policy /= denom;
        self.loss_value /= denom;
        self.loss_grp /= denom;
        self.loss_tenpai /= denom;
        self.loss_danger /= denom;
        self.loss_opp_next /= denom;
        self.loss_score_pdf /= denom;
        self.loss_score_cdf /= denom;
        self
    }
}

fn batch_stats_from_breakdown<B: Backend>(
    agreement: f64,
    breakdown: &LossBreakdown<B>,
) -> BatchStats {
    BatchStats {
        total_loss: scalar1(&breakdown.total),
        policy_agreement: agreement,
        loss_policy: scalar1(&breakdown.policy),
        loss_value: scalar1(&breakdown.value),
        loss_grp: scalar1(&breakdown.grp),
        loss_tenpai: scalar1(&breakdown.tenpai),
        loss_danger: scalar1(&breakdown.danger),
        loss_opp_next: scalar1(&breakdown.opp_next),
        loss_score_pdf: scalar1(&breakdown.score_pdf),
        loss_score_cdf: scalar1(&breakdown.score_cdf),
    }
}

fn optimizer_steps_for_samples(
    samples: usize,
    microbatch_size: usize,
    accum_steps: usize,
) -> usize {
    if samples == 0 {
        0
    } else {
        samples.div_ceil(microbatch_size).div_ceil(accum_steps)
    }
}

fn model_kind(config: &HydraModelConfig) -> &'static str {
    if config.is_learner() {
        "learner"
    } else {
        "actor"
    }
}

fn print_banner(
    model_config: &HydraModelConfig,
    config: &TrainConfig,
    device_name: &str,
    stats: &BannerStats,
) {
    let tb_dir = config.output_dir.join("tb");
    println!();
    println!(
        "{}",
        "========================================".bold().cyan()
    );
    println!("{}", "  HYDRA BC Training".bold().cyan());
    println!(
        "{}",
        "========================================".bold().cyan()
    );
    println!(
        "  {} {}",
        "Model:".white(),
        format!(
            "{} ({} blocks, {}ch)",
            model_kind(model_config),
            model_config.num_blocks,
            model_config.hidden_channels
        )
        .green()
    );
    println!("  {} {}", "Device:".white(), device_name.green());
    println!(
        "  {} {}",
        "Dataset:".white(),
        if stats.counts_exact {
            format!(
                "{} ({} sources, {} games)",
                config.data_dir.display(),
                stats.total_sources,
                stats.total_games
            )
        } else {
            format!(
                "{} ({} sources, game counts discovered during streaming)",
                config.data_dir.display(),
                stats.total_sources,
            )
        }
        .green()
    );
    println!(
        "  {} {}",
        "Train:".white(),
        if stats.counts_exact {
            format!(
                "{} games | Val: {} games",
                stats.train_count, stats.val_count
            )
        } else {
            "streaming split, counts unknown until load/validation".to_string()
        }
        .green()
    );
    println!(
        "  {} {}",
        "Buffer:".white(),
        format!(
            "{} samples (max {} games)",
            config.buffer_samples, config.buffer_games
        )
        .yellow()
    );
    println!(
        "  {} {}",
        "Batch:".white(),
        format!(
            "{} ({} x {} accum)",
            config.batch_size,
            config.microbatch_size.unwrap_or(config.batch_size),
            stats.accum_steps
        )
        .yellow()
    );
    println!(
        "  {} {}",
        "Epochs:".white(),
        config.num_epochs.to_string().yellow()
    );
    println!(
        "  {} {}",
        "Output:".white(),
        config.output_dir.display().to_string().green()
    );
    println!(
        "  {} {}",
        "TBoard:".white(),
        if config.tensorboard {
            tb_dir.display().to_string().green()
        } else {
            "disabled".yellow()
        }
    );
    println!(
        "{}",
        "========================================".bold().cyan()
    );
    println!();
}

fn save_checkpoint(
    model: &HydraModel<TrainBackend>,
    output_dir: &Path,
    epoch: usize,
    train_loss: f64,
    eval_agreement: f64,
) -> Result<(), String> {
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let checkpoint_base = output_dir.join("best_model");
    model
        .clone()
        .save_file(&checkpoint_base, &recorder)
        .map_err(|err| {
            format!(
                "failed to save checkpoint {}: {err}",
                checkpoint_base.display()
            )
        })?;

    let meta = CheckpointMeta::new(epoch as u32, train_loss, eval_agreement);
    let meta_path = output_dir.join("best_model.meta.json");
    let meta_json = serde_json::to_string_pretty(&meta)
        .map_err(|err| format!("failed to serialize checkpoint metadata: {err}"))?;
    fs::write(&meta_path, meta_json).map_err(|err| {
        format!(
            "failed to write checkpoint metadata {}: {err}",
            meta_path.display()
        )
    })
}

fn append_training_log(output_dir: &Path, entry: &EpochLogEntry) -> Result<(), String> {
    let path = output_dir.join("training_log.jsonl");
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|err| format!("failed to open training log {}: {err}", path.display()))?;
    let line = serde_json::to_string(entry)
        .map_err(|err| format!("failed to serialize training log entry: {err}"))?;
    writeln!(file, "{line}")
        .map_err(|err| format!("failed to append training log {}: {err}", path.display()))
}

fn append_step_log(output_dir: &Path, entry: &StepLogEntry) -> Result<(), String> {
    let path = output_dir.join("step_log.jsonl");
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|err| format!("failed to open step log {}: {err}", path.display()))?;
    let line = serde_json::to_string(entry)
        .map_err(|err| format!("failed to serialize step log entry: {err}"))?;
    writeln!(file, "{line}")
        .map_err(|err| format!("failed to append step log {}: {err}", path.display()))
}

fn log_tensorboard<W: Write>(
    tb: &mut EventWriter<W>,
    epoch: usize,
    train: &ScalarAverages,
    val_agreement: Option<f64>,
    lr: f64,
    best_val_agreement: Option<f64>,
) -> Result<(), String> {
    let step = epoch as i64;
    tb.write_scalar(step, "train/total_loss", train.total_loss as f32)
        .map_err(|err| format!("tensorboard write train/total_loss failed: {err}"))?;
    tb.write_scalar(
        step,
        "train/policy_agreement",
        train.policy_agreement as f32,
    )
    .map_err(|err| format!("tensorboard write train/policy_agreement failed: {err}"))?;
    if let Some(val_agreement) = val_agreement {
        tb.write_scalar(step, "val/policy_agreement", val_agreement as f32)
            .map_err(|err| format!("tensorboard write val/policy_agreement failed: {err}"))?;
    }
    tb.write_scalar(step, "lr", lr as f32)
        .map_err(|err| format!("tensorboard write lr failed: {err}"))?;
    if let Some(best_val_agreement) = best_val_agreement {
        tb.write_scalar(step, "train/best_val_agreement", best_val_agreement as f32)
            .map_err(|err| format!("tensorboard write train/best_val_agreement failed: {err}"))?;
    }
    tb.write_scalar(step, "train/loss_policy", train.loss_policy as f32)
        .map_err(|err| format!("tensorboard write train/loss_policy failed: {err}"))?;
    tb.write_scalar(step, "train/loss_value", train.loss_value as f32)
        .map_err(|err| format!("tensorboard write train/loss_value failed: {err}"))?;
    tb.write_scalar(step, "train/loss_grp", train.loss_grp as f32)
        .map_err(|err| format!("tensorboard write train/loss_grp failed: {err}"))?;
    tb.write_scalar(step, "train/loss_tenpai", train.loss_tenpai as f32)
        .map_err(|err| format!("tensorboard write train/loss_tenpai failed: {err}"))?;
    tb.write_scalar(step, "train/loss_danger", train.loss_danger as f32)
        .map_err(|err| format!("tensorboard write train/loss_danger failed: {err}"))?;
    tb.write_scalar(step, "train/loss_opp_next", train.loss_opp_next as f32)
        .map_err(|err| format!("tensorboard write train/loss_opp_next failed: {err}"))?;
    tb.write_scalar(step, "train/loss_score_pdf", train.loss_score_pdf as f32)
        .map_err(|err| format!("tensorboard write train/loss_score_pdf failed: {err}"))?;
    tb.write_scalar(step, "train/loss_score_cdf", train.loss_score_cdf as f32)
        .map_err(|err| format!("tensorboard write train/loss_score_cdf failed: {err}"))?;
    tb.flush()
        .map_err(|err| format!("tensorboard flush failed: {err}"))
}

fn format_progress_message(loss: f64, agreement: f64, lr: f64) -> String {
    format!("loss={loss:.4} agree={:.2}% lr={lr:.2e}", agreement * 100.0)
}

fn run_validation(
    model: &HydraModel<TrainBackend>,
    config: &TrainConfig,
    loader_config: &StreamingLoaderConfig,
    manifest: &hydra_train::data::pipeline::DataManifest,
    device: &<ValidBackend as Backend>::Device,
) -> Result<f64, String> {
    let model_valid = model.valid();
    let validation_batch_size = validation_microbatch_size(config);
    let validation_sample_limit = validation_sample_limit(config);
    let val_pb = if manifest.counts_exact {
        make_bar(
            manifest.val_count as u64,
            "[val-load] [{bar:40.cyan/blue}] {pos}/{len} games {msg}",
        )?
    } else {
        make_spinner("[val-load] {spinner:.cyan} {pos} games {msg}")?
    };
    let mut total_correct = 0usize;
    let mut total_samples = 0usize;

    for buffer_result in stream_val_pass(manifest, loader_config, Some(&val_pb)) {
        let buffer = buffer_result.map_err(|err| format!("validation stream failed: {err}"))?;
        for chunk in buffer.chunks(validation_batch_size) {
            if let Some(limit) = validation_sample_limit
                && total_samples >= limit
            {
                break;
            }
            let capped_chunk = if let Some(limit) = validation_sample_limit {
                let remaining = limit.saturating_sub(total_samples);
                &chunk[..chunk.len().min(remaining)]
            } else {
                chunk
            };
            if capped_chunk.is_empty() {
                break;
            }
            let Some((obs, targets)) = collate_samples::<ValidBackend>(capped_chunk, false, device)
            else {
                continue;
            };
            let output = model_valid.forward(obs);
            let target_actions = target_actions_from_policy_target(targets.policy_target);
            let (correct, samples) =
                policy_agreement_counts(output.policy_logits, targets.legal_mask, target_actions);
            total_correct += correct;
            total_samples += samples;
            val_pb.set_message(format!(
                "samples={} agree={:.2}%",
                total_samples,
                total_correct as f64 / total_samples.max(1) as f64 * 100.0
            ));
        }
        if let Some(limit) = validation_sample_limit
            && total_samples >= limit
        {
            break;
        }
    }

    val_pb.finish_with_message("validation pass complete".to_string());
    if total_samples == 0 {
        Ok(0.0)
    } else {
        Ok(total_correct as f64 / total_samples as f64)
    }
}

fn run() -> Result<(), String> {
    let config_path = parse_args(env::args())?;
    let config = read_config(&config_path)?;
    validate_config(&config)?;
    configure_threads(config.num_threads)?;

    fs::create_dir_all(&config.output_dir).map_err(|err| {
        format!(
            "failed to create output directory {}: {err}",
            config.output_dir.display()
        )
    })?;

    let loader_config = StreamingLoaderConfig {
        buffer_games: config.buffer_games,
        buffer_samples: config.buffer_samples,
        train_fraction: config.train_fraction,
        seed: config.seed,
        archive_queue_bound: config.archive_queue_bound,
        max_skip_logs_per_source: config.max_skip_logs_per_source,
    };

    let scan_sources_len = if config.data_dir.is_file() {
        1
    } else {
        fs::read_dir(&config.data_dir)
            .map_err(|err| {
                format!(
                    "failed to read data dir {}: {err}",
                    config.data_dir.display()
                )
            })?
            .filter_map(Result::ok)
            .filter_map(|entry| entry.file_type().ok().filter(|ft| ft.is_file()))
            .count()
    };
    let scan_pb = make_bar(
        scan_sources_len as u64,
        "[scan] [{bar:40.cyan/blue}] {pos}/{len} sources {msg}",
    )?;
    scan_pb.set_message("Scanning archives...".to_string());
    let manifest =
        scan_data_sources_with_progress(&config.data_dir, config.train_fraction, Some(&scan_pb))
            .map_err(|err| {
                format!(
                    "failed to scan MJAI data from {}: {err}",
                    config.data_dir.display()
                )
            })?;
    if manifest.counts_exact {
        scan_pb.finish_with_message(format!(
            "found {} train / {} val games",
            manifest.train_count, manifest.val_count
        ));
    } else {
        scan_pb.finish_with_message(format!(
            "found {} sources; exact game counts deferred to streaming load",
            manifest.sources.len()
        ));
    }

    let train_cfg = BCTrainerConfig::default_learner()
        .with_batch_size(config.batch_size)
        .with_lr(BCTrainerConfig::default_learner().lr);
    train_cfg
        .validate()
        .map_err(|err| format!("invalid trainer config: {err}"))?;

    let device_name = device_label(&config.device);
    let train_device = train_device(&config.device);
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let model_config = HydraModelConfig::learner();
    let mut model = model_config.init::<TrainBackend>(&train_device);
    if let Some(resume) = &config.resume_checkpoint {
        model = model
            .load_file(resume, &recorder, &train_device)
            .map_err(|err| format!("failed to load checkpoint {}: {err}", resume.display()))?;
    }

    let mut optimizer = train_cfg.optimizer_config().init();
    let loss_fn = HydraLoss::<TrainBackend>::new(build_loss_config(config.advanced_loss.as_ref())?);
    let total_steps = config.max_train_steps.unwrap_or(config.num_epochs.max(1));
    let microbatch_size = train_microbatch_size(&config);
    let accum_steps = config.batch_size.div_ceil(microbatch_size).max(1);
    let mut best_val_agreement = f64::NEG_INFINITY;
    let mut global_step = 0usize;
    let mut tb = if config.tensorboard {
        let tb_dir = config.output_dir.join("tb");
        fs::create_dir_all(&tb_dir).map_err(|err| {
            format!(
                "failed to create tensorboard dir {}: {err}",
                tb_dir.display()
            )
        })?;
        Some(EventWriter::create(&tb_dir).map_err(|err| format!("tensorboard init: {err}"))?)
    } else {
        None
    };

    let banner_stats = BannerStats {
        total_sources: manifest.sources.len(),
        total_games: manifest.total_games,
        train_count: manifest.train_count,
        val_count: manifest.val_count,
        accum_steps,
        counts_exact: manifest.counts_exact,
    };
    print_banner(&model_config, &config, &device_name, &banner_stats);

    for epoch in 0..config.num_epochs {
        let lr = warmup_then_cosine_lr(
            epoch,
            train_cfg.warmup_steps.min(total_steps),
            total_steps,
            train_cfg.lr,
            1e-6,
        );

        let multi = MultiProgress::new();
        let load_pb = if manifest.counts_exact {
            multi.add(make_bar(
                manifest.train_count as u64,
                &format!(
                    "[epoch {}/{} load] [{{bar:30.cyan/blue}}] {{pos}}/{{len}} games {{msg}}",
                    epoch + 1,
                    config.num_epochs
                ),
            )?)
        } else {
            multi.add(make_spinner(&format!(
                "[epoch {}/{} load] {{spinner:.cyan}} {{pos}} games {{msg}}",
                epoch + 1,
                config.num_epochs
            ))?)
        };
        let train_pb = multi.add(make_bar(
            1,
            &format!(
                "[epoch {}/{}] [{{bar:30.green/black}}] {{pos}}/{{len}} batches  {{msg}}",
                epoch + 1,
                config.num_epochs
            ),
        )?);

        let mut stats = ScalarAverages::default();
        let mut step_window = ScalarAverages::default();
        let mut accumulator: GradientsAccumulator<HydraModel<TrainBackend>> =
            GradientsAccumulator::new();
        let mut accum_current = 0usize;
        let mut pending_breakdowns: Vec<BatchStats> = Vec::new();
        let mut seen_samples = 0usize;
        let mut assumed_games_seen = 0usize;
        let mut remaining_games = manifest.train_count;

        for buffer_result in stream_train_epoch(&manifest, &loader_config, epoch, Some(&load_pb)) {
            let buffer = buffer_result.map_err(|err| format!("training stream failed: {err}"))?;
            if manifest.counts_exact {
                let assumed_games = remaining_games.min(config.buffer_games);
                remaining_games = remaining_games.saturating_sub(assumed_games);
                assumed_games_seen += assumed_games;
            }
            seen_samples += buffer.len();
            if manifest.counts_exact && assumed_games_seen > 0 {
                let estimated_total_samples =
                    seen_samples.saturating_mul(manifest.train_count) / assumed_games_seen.max(1);
                let estimated_steps = optimizer_steps_for_samples(
                    estimated_total_samples,
                    microbatch_size,
                    accum_steps,
                )
                .max(1);
                train_pb.set_length(estimated_steps as u64);
            } else if !manifest.counts_exact {
                load_pb.set_message(format!(
                    "samples={}  batches={}",
                    seen_samples, stats.num_batches
                ));
            }

            for chunk in buffer.chunks(microbatch_size) {
                let lr = warmup_then_cosine_lr(
                    global_step,
                    train_cfg.warmup_steps.min(total_steps),
                    total_steps,
                    train_cfg.lr,
                    1e-6,
                );

                let mut completed_step = false;

                {
                    let Some((obs, targets)) =
                        collate_samples::<TrainBackend>(chunk, config.augment, &train_device)
                    else {
                        continue;
                    };
                    let output = model.forward(obs.clone());
                    let agreement = policy_agreement(
                        output.policy_logits.clone(),
                        targets.legal_mask.clone(),
                        target_actions_from_policy_target(targets.policy_target.clone()),
                    );
                    let breakdown = loss_fn.total_loss(&output, &targets);
                    let batch_stats = batch_stats_from_breakdown(agreement, &breakdown);
                    let grads = breakdown.total.backward();
                    let grads = GradientsParams::from_grads(grads, &model);
                    accumulator.accumulate(&model, grads);
                    pending_breakdowns.push(batch_stats);
                    accum_current += 1;

                    if accum_current >= accum_steps {
                        let grads = accumulator.grads();
                        model = optimizer.step(lr, model, grads);
                        for batch_stats in pending_breakdowns.drain(..) {
                            stats.record_batch(batch_stats);
                            step_window.record_batch(batch_stats);
                        }
                        accum_current = 0;
                        global_step += 1;
                        train_pb.inc(1);
                        train_pb.set_message(format_progress_message(
                            stats.total_loss / stats.num_batches.max(1) as f64,
                            stats.policy_agreement / stats.num_batches.max(1) as f64,
                            lr,
                        ));
                        completed_step = true;
                    }
                }

                if completed_step {
                    let val_agreement = if global_step.is_multiple_of(config.validate_every_n_steps)
                    {
                        let agreement = run_validation(
                            &model,
                            &config,
                            &loader_config,
                            &manifest,
                            &train_device,
                        )?;
                        if agreement > best_val_agreement {
                            best_val_agreement = agreement;
                            save_checkpoint(
                                &model,
                                &config.output_dir,
                                global_step,
                                step_window.total_loss / step_window.num_batches.max(1) as f64,
                                agreement,
                            )?;
                        }
                        Some(agreement)
                    } else {
                        None
                    };

                    if global_step.is_multiple_of(config.log_every_n_steps) {
                        let window_stats = std::mem::take(&mut step_window).finalize();

                        println!(
                            "{} {} {} {} {} {}",
                            format!("step {}", global_step).bold().cyan(),
                            format!("train_loss={:.4}", window_stats.total_loss).green(),
                            format!("train_agree={:.2}%", window_stats.policy_agreement * 100.0)
                                .green(),
                            if let Some(val_agreement) = val_agreement {
                                format!("val_agree={:.2}%", val_agreement * 100.0)
                            } else {
                                "val_agree=skipped".to_string()
                            }
                            .bold()
                            .yellow(),
                            if best_val_agreement.is_finite() {
                                format!("best={:.2}%", best_val_agreement * 100.0)
                            } else {
                                "best=n/a".to_string()
                            }
                            .bold()
                            .magenta(),
                            format!("lr={:.2e}", lr).white(),
                        );

                        if let Some(ref mut tb) = tb {
                            log_tensorboard(
                                tb,
                                global_step,
                                &window_stats,
                                val_agreement,
                                lr,
                                if best_val_agreement.is_finite() {
                                    Some(best_val_agreement)
                                } else {
                                    None
                                },
                            )?;
                        }

                        let step_entry = StepLogEntry {
                            global_step,
                            epoch: epoch + 1,
                            lr,
                            train_total_loss: window_stats.total_loss,
                            train_policy_agreement: window_stats.policy_agreement,
                            train_loss_policy: window_stats.loss_policy,
                            train_loss_value: window_stats.loss_value,
                            train_loss_grp: window_stats.loss_grp,
                            train_loss_tenpai: window_stats.loss_tenpai,
                            train_loss_danger: window_stats.loss_danger,
                            train_loss_opp_next: window_stats.loss_opp_next,
                            train_loss_score_pdf: window_stats.loss_score_pdf,
                            train_loss_score_cdf: window_stats.loss_score_cdf,
                            val_policy_agreement: val_agreement,
                            best_val_agreement: if best_val_agreement.is_finite() {
                                Some(best_val_agreement)
                            } else {
                                None
                            },
                        };
                        append_step_log(&config.output_dir, &step_entry)?;
                    }

                    if global_step.is_multiple_of(config.checkpoint_every_n_steps) {
                        save_checkpoint(
                            &model,
                            &config.output_dir,
                            global_step,
                            stats.total_loss / stats.num_batches.max(1) as f64,
                            if best_val_agreement.is_finite() {
                                best_val_agreement
                            } else {
                                0.0
                            },
                        )?;
                    }

                    if let Some(max_train_steps) = config.max_train_steps
                        && global_step >= max_train_steps
                    {
                        break;
                    }
                }
            }

            if let Some(max_train_steps) = config.max_train_steps
                && global_step >= max_train_steps
            {
                break;
            }
        }

        if accum_current > 0 {
            let lr = warmup_then_cosine_lr(
                global_step,
                train_cfg.warmup_steps.min(total_steps),
                total_steps,
                train_cfg.lr,
                1e-6,
            );
            let grads = accumulator.grads();
            model = optimizer.step(lr, model, grads);
            for batch_stats in pending_breakdowns.drain(..) {
                stats.record_batch(batch_stats);
                step_window.record_batch(batch_stats);
            }
            global_step += 1;
            train_pb.inc(1);
        }

        load_pb.finish_with_message("training data stream complete".to_string());
        let train_stats = stats.finalize();
        train_pb.set_length(train_stats.num_batches.max(1) as u64);
        train_pb.finish_with_message(format_progress_message(
            train_stats.total_loss,
            train_stats.policy_agreement,
            lr,
        ));

        let should_validate =
            (epoch + 1) % config.validation_every_n_epochs == 0 || epoch + 1 == config.num_epochs;
        let val_agreement = if should_validate {
            let agreement =
                run_validation(&model, &config, &loader_config, &manifest, &train_device)?;
            if agreement > best_val_agreement {
                best_val_agreement = agreement;
                save_checkpoint(
                    &model,
                    &config.output_dir,
                    epoch + 1,
                    train_stats.total_loss,
                    agreement,
                )?;
            }
            Some(agreement)
        } else {
            None
        };

        if let Some(ref mut tb) = tb {
            log_tensorboard(
                tb,
                epoch + 1,
                &train_stats,
                val_agreement,
                lr,
                if best_val_agreement.is_finite() {
                    Some(best_val_agreement)
                } else {
                    None
                },
            )?;
        }

        let entry = EpochLogEntry {
            epoch: epoch + 1,
            global_step,
            lr,
            train_total_loss: train_stats.total_loss,
            train_policy_agreement: train_stats.policy_agreement,
            train_loss_policy: train_stats.loss_policy,
            train_loss_value: train_stats.loss_value,
            train_loss_grp: train_stats.loss_grp,
            train_loss_tenpai: train_stats.loss_tenpai,
            train_loss_danger: train_stats.loss_danger,
            train_loss_opp_next: train_stats.loss_opp_next,
            train_loss_score_pdf: train_stats.loss_score_pdf,
            train_loss_score_cdf: train_stats.loss_score_cdf,
            val_policy_agreement: val_agreement,
            best_val_agreement: if best_val_agreement.is_finite() {
                Some(best_val_agreement)
            } else {
                None
            },
            num_batches: train_stats.num_batches,
        };
        append_training_log(&config.output_dir, &entry)?;

        println!(
            "{} {} {} {} {} {}",
            format!("epoch {}/{}", epoch + 1, config.num_epochs)
                .bold()
                .cyan(),
            format!("train_loss={:.4}", train_stats.total_loss).green(),
            format!("train_agree={:.2}%", train_stats.policy_agreement * 100.0).green(),
            if let Some(val_agreement) = val_agreement {
                format!("val_agree={:.2}%", val_agreement * 100.0)
            } else {
                "val_agree=skipped".to_string()
            }
            .bold()
            .yellow(),
            if best_val_agreement.is_finite() {
                format!("best={:.2}%", best_val_agreement * 100.0)
            } else {
                "best=n/a".to_string()
            }
            .bold()
            .magenta(),
            format!("lr={:.2e}", lr).white(),
        );

        if let Some(max_train_steps) = config.max_train_steps
            && global_step >= max_train_steps
        {
            break;
        }
    }

    println!(
        "{} {}",
        "Finished BC training. Best validation agreement:"
            .bold()
            .cyan(),
        if best_val_agreement.is_finite() {
            format!("{:.2}%", best_val_agreement * 100.0)
        } else {
            "n/a".to_string()
        }
        .bold()
        .green()
    );

    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_args_accepts_single_config_path() {
        let args = vec!["train".to_string(), "config.json".to_string()];
        let parsed = parse_args(args).expect("single config arg should parse");
        assert_eq!(parsed, PathBuf::from("config.json"));
    }

    #[test]
    fn parse_args_rejects_missing_config() {
        let args = vec!["train".to_string()];
        let err = parse_args(args).expect_err("missing config should fail");
        assert!(err.contains("Usage:"));
    }

    #[test]
    fn parse_args_rejects_extra_args() {
        let args = vec![
            "train".to_string(),
            "config.json".to_string(),
            "extra".to_string(),
        ];
        let err = parse_args(args).expect_err("extra args should fail");
        assert!(err.contains("Usage:"));
    }

    #[test]
    fn read_config_applies_defaults() {
        let base = std::env::temp_dir().join(format!(
            "hydra_train_config_{}_{}.json",
            std::process::id(),
            default_seed()
        ));
        let json = r#"{
            "data_dir": "/tmp/data",
            "output_dir": "/tmp/out",
            "num_epochs": 3
        }"#;
        fs::write(&base, json).expect("write config");
        let cfg = read_config(&base).expect("read config");
        assert_eq!(cfg.data_dir, PathBuf::from("/tmp/data"));
        assert_eq!(cfg.output_dir, PathBuf::from("/tmp/out"));
        assert_eq!(cfg.num_epochs, 3);
        assert_eq!(cfg.batch_size, 2048);
        assert!(cfg.microbatch_size.is_none());
        assert!(cfg.validation_microbatch_size.is_none());
        assert!((cfg.train_fraction - 0.9).abs() < f32::EPSILON);
        assert!(cfg.augment);
        assert_eq!(cfg.seed, 0);
        assert_eq!(cfg.device, "cpu");
        assert_eq!(cfg.buffer_games, 50_000);
        assert_eq!(cfg.buffer_samples, 32_768);
        assert!(cfg.num_threads.is_none());
        assert!(cfg.tensorboard);
        assert_eq!(cfg.archive_queue_bound, 128);
        assert_eq!(cfg.validation_every_n_epochs, 1);
        assert_eq!(cfg.max_skip_logs_per_source, 32);
        assert!(cfg.max_validation_batches.is_none());
        assert!(cfg.max_validation_samples.is_none());
        assert!(cfg.advanced_loss.is_none());
        fs::remove_file(base).ok();
    }

    #[test]
    fn validation_microbatch_and_sample_limit_fallbacks_work() {
        let cfg = TrainConfig {
            data_dir: PathBuf::from("/tmp/data"),
            output_dir: PathBuf::from("/tmp/out"),
            num_epochs: 1,
            batch_size: 256,
            microbatch_size: Some(64),
            validation_microbatch_size: None,
            train_fraction: 0.9,
            augment: true,
            resume_checkpoint: None,
            seed: 0,
            advanced_loss: None,
            device: "cpu".to_string(),
            buffer_games: 16,
            buffer_samples: 128,
            num_threads: None,
            tensorboard: false,
            archive_queue_bound: 8,
            validation_every_n_epochs: 1,
            max_skip_logs_per_source: 4,
            log_every_n_steps: 10,
            validate_every_n_steps: 10,
            checkpoint_every_n_steps: 10,
            max_train_steps: None,
            max_validation_batches: Some(32),
            max_validation_samples: None,
        };

        assert_eq!(train_microbatch_size(&cfg), 64);
        assert_eq!(validation_microbatch_size(&cfg), 64);
        assert_eq!(validation_sample_limit(&cfg), Some(2048));

        let cfg = TrainConfig {
            validation_microbatch_size: Some(32),
            max_validation_batches: Some(32),
            max_validation_samples: Some(1500),
            ..cfg
        };

        assert_eq!(validation_microbatch_size(&cfg), 32);
        assert_eq!(validation_sample_limit(&cfg), Some(1500));
    }

    #[test]
    fn validate_config_rejects_zero_validation_microbatch_and_samples() {
        let cfg = TrainConfig {
            data_dir: PathBuf::from("/tmp/data"),
            output_dir: PathBuf::from("/tmp/out"),
            num_epochs: 1,
            batch_size: 256,
            microbatch_size: Some(64),
            validation_microbatch_size: Some(0),
            train_fraction: 0.9,
            augment: true,
            resume_checkpoint: None,
            seed: 0,
            advanced_loss: None,
            device: "cpu".to_string(),
            buffer_games: 16,
            buffer_samples: 128,
            num_threads: None,
            tensorboard: false,
            archive_queue_bound: 8,
            validation_every_n_epochs: 1,
            max_skip_logs_per_source: 4,
            log_every_n_steps: 10,
            validate_every_n_steps: 10,
            checkpoint_every_n_steps: 10,
            max_train_steps: None,
            max_validation_batches: None,
            max_validation_samples: Some(0),
        };

        let err = validate_config(&cfg).expect_err("zero validation controls should fail");
        assert!(
            err.contains("max_validation_samples") || err.contains("validation_microbatch_size")
        );
    }

    #[test]
    fn build_loss_config_defaults_match_baseline() {
        let loss = build_loss_config(None).expect("default loss config should build");
        assert!((loss.w_safety_residual - 0.0).abs() < f32::EPSILON);
        assert!((loss.w_belief_fields - 0.0).abs() < f32::EPSILON);
        assert!((loss.w_mixture_weight - 0.0).abs() < f32::EPSILON);
        assert!((loss.w_opponent_hand_type - 0.0).abs() < f32::EPSILON);
        assert!((loss.w_delta_q - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn build_loss_config_allows_safety_residual_only() {
        let advanced = AdvancedLossConfig {
            safety_residual: Some(0.1),
            ..Default::default()
        };
        let loss = build_loss_config(Some(&advanced)).expect("safety residual should be allowed");
        assert!((loss.w_safety_residual - 0.1).abs() < 1e-6);
        assert!((loss.w_belief_fields - 0.0).abs() < f32::EPSILON);
        assert!((loss.w_mixture_weight - 0.0).abs() < f32::EPSILON);
        assert!((loss.w_opponent_hand_type - 0.0).abs() < f32::EPSILON);
        assert!((loss.w_delta_q - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn build_loss_config_rejects_negative_safety_residual() {
        let advanced = AdvancedLossConfig {
            safety_residual: Some(-0.1),
            ..Default::default()
        };
        let err =
            build_loss_config(Some(&advanced)).expect_err("negative safety residual should fail");
        assert!(err.contains("invalid loss config"));
    }

    #[test]
    fn build_loss_config_rejects_belief_fields_activation() {
        let advanced = AdvancedLossConfig {
            belief_fields: Some(0.1),
            ..Default::default()
        };
        let err = build_loss_config(Some(&advanced))
            .expect_err("belief fields should remain blocked in train.rs");
        assert!(err.contains("advanced_loss.belief_fields"));
    }

    #[test]
    fn build_loss_config_rejects_belief_fields_even_at_zero() {
        let advanced = AdvancedLossConfig {
            belief_fields: Some(0.0),
            ..Default::default()
        };
        let err = build_loss_config(Some(&advanced))
            .expect_err("blocked belief fields key should be rejected even at zero");
        assert!(err.contains("advanced_loss.belief_fields"));
    }

    #[test]
    fn build_loss_config_rejects_mixture_weight_activation() {
        let advanced = AdvancedLossConfig {
            mixture_weight: Some(0.1),
            ..Default::default()
        };
        let err = build_loss_config(Some(&advanced))
            .expect_err("mixture weight should remain blocked in train.rs");
        assert!(err.contains("advanced_loss.mixture_weight"));
    }

    #[test]
    fn build_loss_config_rejects_opponent_hand_type_activation() {
        let advanced = AdvancedLossConfig {
            opponent_hand_type: Some(0.1),
            ..Default::default()
        };
        let err = build_loss_config(Some(&advanced))
            .expect_err("opponent hand type should remain blocked in train.rs");
        assert!(err.contains("advanced_loss.opponent_hand_type"));
    }

    #[test]
    fn build_loss_config_rejects_delta_q_activation() {
        let advanced = AdvancedLossConfig {
            delta_q: Some(0.1),
            ..Default::default()
        };
        let err = build_loss_config(Some(&advanced))
            .expect_err("delta_q should remain blocked in train.rs");
        assert!(err.contains("advanced_loss.delta_q"));
    }

    #[test]
    fn read_config_rejects_unknown_advanced_loss_field() {
        let base = std::env::temp_dir().join(format!(
            "hydra_train_bad_advanced_loss_{}_{}.json",
            std::process::id(),
            default_seed()
        ));
        let json = r#"{
            "data_dir": "/tmp/data",
            "output_dir": "/tmp/out",
            "num_epochs": 3,
            "advanced_loss": {
                "not_a_real_field": 0.1
            }
        }"#;
        fs::write(&base, json).expect("write config");
        let err = read_config(&base).expect_err("unknown advanced loss field should fail");
        assert!(err.contains("not_a_real_field"));
        fs::remove_file(base).ok();
    }

    #[test]
    fn train_device_prefers_env_override_then_config() {
        unsafe {
            env::remove_var("HYDRA_TRAIN_DEVICE");
        }
        assert_eq!(train_device("cpu"), LibTorchDevice::Cpu);
        assert_eq!(train_device("cuda:2"), LibTorchDevice::Cuda(2));

        unsafe {
            env::set_var("HYDRA_TRAIN_DEVICE", "cuda:0");
        }
        assert_eq!(train_device("cpu"), LibTorchDevice::Cuda(0));

        unsafe {
            env::set_var("HYDRA_TRAIN_DEVICE", "cpu");
        }
        assert_eq!(train_device("cuda:3"), LibTorchDevice::Cpu);

        unsafe {
            env::remove_var("HYDRA_TRAIN_DEVICE");
        }
    }

    #[test]
    #[should_panic(expected = "unsupported HYDRA_TRAIN_DEVICE")]
    fn train_device_rejects_invalid_env_value() {
        unsafe {
            env::set_var("HYDRA_TRAIN_DEVICE", "vulkan");
        }
        let _ = train_device("cpu");
        unsafe {
            env::remove_var("HYDRA_TRAIN_DEVICE");
        }
    }

    #[test]
    fn validation_agreement_is_sample_weighted_across_chunks() {
        let device: <burn::backend::ndarray::NdArray<f32> as Backend>::Device = Default::default();
        let logits = Tensor::<burn::backend::ndarray::NdArray<f32>, 2>::from_floats(
            [[5.0, 0.0], [5.0, 0.0], [5.0, 0.0], [5.0, 0.0], [0.0, 5.0]],
            &device,
        );
        let mask = Tensor::<burn::backend::ndarray::NdArray<f32>, 2>::ones([5, 2], &device);
        let targets = Tensor::<burn::backend::ndarray::NdArray<f32>, 1, Int>::from_ints(
            [0, 1, 1, 1, 1],
            &device,
        );

        let (chunk1_correct, chunk1_total) = policy_agreement_counts(
            logits.clone().slice([0..4, 0..2]),
            mask.clone().slice([0..4, 0..2]),
            targets.clone().slice([0..4]),
        );
        let (chunk2_correct, chunk2_total) = policy_agreement_counts(
            logits.slice([4..5, 0..2]),
            mask.slice([4..5, 0..2]),
            targets.slice([4..5]),
        );

        let weighted =
            (chunk1_correct + chunk2_correct) as f64 / (chunk1_total + chunk2_total) as f64;
        let naive_chunk_average = ((chunk1_correct as f64 / chunk1_total as f64)
            + (chunk2_correct as f64 / chunk2_total as f64))
            / 2.0;

        assert!((weighted - 0.4).abs() < 1e-12);
        assert!((naive_chunk_average - 0.625).abs() < 1e-12);
    }
}
