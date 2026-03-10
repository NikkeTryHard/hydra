use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use burn::backend::{libtorch::LibTorchDevice, Autodiff, LibTorch};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use hydra_train::data::mjai_loader::{MjaiDataset, MjaiGame};
use hydra_train::data::pipeline::{
    collate_sample_chunk, collect_samples, load_mjai_directory, shuffle_samples,
};
use hydra_train::data::sample::MjaiSample;
use hydra_train::model::{HydraModel, HydraModelConfig};
use hydra_train::training::bc::{
    policy_agreement, target_actions_from_policy_target, train_epoch, warmup_then_cosine_lr,
    BCTrainerConfig, CheckpointMeta,
};
use hydra_train::training::losses::{HydraLoss, HydraLossConfig};

type TrainBackend = Autodiff<LibTorch<f32>>;

#[derive(Debug, serde::Deserialize)]
struct TrainConfig {
    data_dir: PathBuf,
    output_dir: PathBuf,
    num_epochs: usize,
    #[serde(default = "default_batch_size")]
    batch_size: usize,
    #[serde(default)]
    microbatch_size: Option<usize>,
    #[serde(default = "default_train_fraction")]
    train_fraction: f32,
    #[serde(default = "default_augment")]
    augment: bool,
    resume_checkpoint: Option<PathBuf>,
    #[serde(default = "default_seed")]
    seed: u64,
    #[serde(default)]
    advanced_loss: Option<AdvancedLossConfig>,
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

fn train_device() -> LibTorchDevice {
    match env::var("HYDRA_TRAIN_DEVICE") {
        Ok(value) => parse_train_device(&value),
        Err(_) => LibTorchDevice::Cpu,
    }
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

fn copy_sample(sample: &MjaiSample) -> MjaiSample {
    MjaiSample {
        obs: sample.obs,
        action: sample.action,
        legal_mask: sample.legal_mask,
        placement: sample.placement,
        score_delta: sample.score_delta,
        grp_label: sample.grp_label,
        oracle_target: sample.oracle_target,
        tenpai: sample.tenpai,
        opp_next: sample.opp_next,
        danger: sample.danger,
        danger_mask: sample.danger_mask,
        safety_residual: sample.safety_residual,
        safety_residual_mask: sample.safety_residual_mask,
        belief_fields: sample.belief_fields,
        mixture_weights: sample.mixture_weights,
        belief_fields_present: sample.belief_fields_present,
        mixture_weights_present: sample.mixture_weights_present,
    }
}

fn dataset_from_games(games: &[MjaiGame], train_fraction: f32) -> MjaiDataset {
    let mut dataset = MjaiDataset::new(train_fraction);
    dataset.games = games
        .iter()
        .map(|game| MjaiGame {
            samples: game.samples.iter().map(copy_sample).collect(),
            final_scores: game.final_scores,
        })
        .collect();
    dataset
}

fn evaluate_policy_agreement(
    model: &HydraModel<TrainBackend>,
    samples: &[&MjaiSample],
    batch_size: usize,
    device: &<TrainBackend as Backend>::Device,
) -> f64 {
    if samples.is_empty() || batch_size == 0 {
        return 0.0;
    }

    let mut total = 0.0;
    let mut num_batches = 0usize;
    for chunk in samples.chunks(batch_size) {
        let Some((obs, targets)) = collate_sample_chunk::<TrainBackend>(chunk, false, device)
        else {
            continue;
        };
        let output = model.forward(obs.clone());
        total += policy_agreement(
            output.policy_logits.clone(),
            targets.legal_mask.clone(),
            target_actions_from_policy_target(targets.policy_target.clone()),
        );
        num_batches += 1;
    }

    if num_batches == 0 {
        0.0
    } else {
        total / num_batches as f64
    }
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

fn run() -> Result<(), String> {
    let config_path = parse_args(env::args())?;
    let config = read_config(&config_path)?;

    fs::create_dir_all(&config.output_dir).map_err(|err| {
        format!(
            "failed to create output directory {}: {err}",
            config.output_dir.display()
        )
    })?;

    let dataset = load_mjai_directory(&config.data_dir, config.train_fraction).map_err(|err| {
        format!(
            "failed to load MJAI data from {}: {err}",
            config.data_dir.display()
        )
    })?;
    let (train_games, val_games) = dataset.train_split();
    let train_dataset = dataset_from_games(train_games, 1.0);
    let val_dataset = dataset_from_games(val_games, 0.0);

    let train_cfg = BCTrainerConfig::default_learner()
        .with_batch_size(config.batch_size)
        .with_lr(BCTrainerConfig::default_learner().lr);
    train_cfg
        .validate()
        .map_err(|err| format!("invalid trainer config: {err}"))?;

    let train_device = train_device();
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

    let mut model = HydraModelConfig::learner().init::<TrainBackend>(&train_device);
    if let Some(resume) = &config.resume_checkpoint {
        model = model
            .load_file(resume, &recorder, &train_device)
            .map_err(|err| format!("failed to load checkpoint {}: {err}", resume.display()))?;
    }

    let mut optimizer = train_cfg.optimizer_config().init();
    let loss_fn = HydraLoss::<TrainBackend>::new(build_loss_config(config.advanced_loss.as_ref())?);
    let total_steps = config.num_epochs.max(1);
    let mut best_val_agreement = f64::NEG_INFINITY;
    let microbatch_size = config.microbatch_size.unwrap_or(config.batch_size);
    let accum_steps = if microbatch_size > 0 {
        (config.batch_size / microbatch_size).max(1)
    } else {
        1
    };

    println!(
        "Starting BC training with {} train games and {} val games (microbatch={}, accum={}x, effective_batch={})",
        train_dataset.num_games(),
        val_dataset.num_games(),
        microbatch_size,
        accum_steps,
        microbatch_size * accum_steps,
    );

    for epoch in 0..config.num_epochs {
        let mut train_samples = collect_samples(&train_dataset);
        shuffle_samples(&mut train_samples, config.seed.wrapping_add(epoch as u64));

        let lr = warmup_then_cosine_lr(
            epoch,
            train_cfg.warmup_steps.min(total_steps),
            total_steps,
            train_cfg.lr,
            1e-6,
        );
        let (next_model, train_stats) = train_epoch(
            model,
            &train_samples,
            microbatch_size,
            accum_steps,
            config.augment,
            &train_device,
            &loss_fn,
            lr,
            &mut optimizer,
        );
        model = next_model;

        let val_samples = collect_samples(&val_dataset);
        let val_agreement =
            evaluate_policy_agreement(&model, &val_samples, config.batch_size, &train_device);

        if val_agreement > best_val_agreement {
            best_val_agreement = val_agreement;
            save_checkpoint(
                &model,
                &config.output_dir,
                epoch + 1,
                train_stats.avg_loss,
                val_agreement,
            )?;
        }

        println!(
            "epoch {}/{} lr={:.3e} train={} val_agree={:.2}% best={:.2}%",
            epoch + 1,
            config.num_epochs,
            lr,
            train_stats.summary(),
            val_agreement * 100.0,
            best_val_agreement * 100.0,
        );
    }

    println!(
        "Finished BC training. Best validation agreement: {:.2}%",
        best_val_agreement * 100.0
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
        assert!((cfg.train_fraction - 0.9).abs() < f32::EPSILON);
        assert!(cfg.augment);
        assert_eq!(cfg.seed, 0);
        assert!(cfg.advanced_loss.is_none());
        fs::remove_file(base).ok();
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
    fn train_device_prefers_available_backend() {
        unsafe {
            env::remove_var("HYDRA_TRAIN_DEVICE");
        }
        assert_eq!(train_device(), LibTorchDevice::Cpu);

        unsafe {
            env::set_var("HYDRA_TRAIN_DEVICE", "cuda:0");
        }
        assert_eq!(train_device(), LibTorchDevice::Cuda(0));

        unsafe {
            env::set_var("HYDRA_TRAIN_DEVICE", "cpu");
        }
        assert_eq!(train_device(), LibTorchDevice::Cpu);

        unsafe {
            env::set_var("HYDRA_TRAIN_DEVICE", "cuda:3");
        }
        assert_eq!(train_device(), LibTorchDevice::Cuda(3));

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
        let _ = train_device();
        unsafe {
            env::remove_var("HYDRA_TRAIN_DEVICE");
        }
    }
}
