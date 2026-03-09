use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use burn::backend::{Autodiff, LibTorch};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use hydra_train::data::mjai_loader::{MjaiDataset, MjaiGame};
use hydra_train::data::pipeline::{
    build_batches, collect_samples, load_mjai_directory, shuffle_samples,
};
use hydra_train::data::sample::MjaiSample;
use hydra_train::model::{HydraModel, HydraModelConfig};
use hydra_train::training::bc::{
    policy_agreement, target_actions_from_policy_target, train_epoch, warmup_then_cosine_lr,
    BCTrainerConfig, CheckpointMeta,
};
use hydra_train::training::losses::{HydraLoss, HydraLossConfig};

type TrainBackend = Autodiff<LibTorch<f32>>;

#[derive(serde::Deserialize)]
struct TrainConfig {
    data_dir: PathBuf,
    output_dir: PathBuf,
    num_epochs: usize,
    #[serde(default = "default_batch_size")]
    batch_size: usize,
    #[serde(default = "default_train_fraction")]
    train_fraction: f32,
    #[serde(default = "default_augment")]
    augment: bool,
    resume_checkpoint: Option<PathBuf>,
    #[serde(default = "default_seed")]
    seed: u64,
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
    let batches = build_batches::<TrainBackend>(samples, batch_size, false, device);
    if batches.is_empty() {
        return 0.0;
    }

    let mut total = 0.0;
    for (obs, targets) in &batches {
        let output = model.forward(obs.clone());
        total += policy_agreement(
            output.policy_logits.clone(),
            targets.legal_mask.clone(),
            target_actions_from_policy_target(targets.policy_target.clone()),
        );
    }

    total / batches.len() as f64
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

    let train_device = Default::default();
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

    let mut model = HydraModelConfig::learner().init::<TrainBackend>(&train_device);
    if let Some(resume) = &config.resume_checkpoint {
        model = model
            .load_file(resume, &recorder, &train_device)
            .map_err(|err| format!("failed to load checkpoint {}: {err}", resume.display()))?;
    }

    let mut optimizer = train_cfg.optimizer_config().init();
    let loss_fn = HydraLoss::<TrainBackend>::new(HydraLossConfig::new());
    let total_steps = config.num_epochs.max(1);
    let mut best_val_agreement = f64::NEG_INFINITY;

    println!(
        "Starting BC training with {} train games and {} val games",
        train_dataset.num_games(),
        val_dataset.num_games()
    );

    for epoch in 0..config.num_epochs {
        let mut train_samples = collect_samples(&train_dataset);
        shuffle_samples(&mut train_samples, config.seed.wrapping_add(epoch as u64));
        let train_batches = build_batches::<TrainBackend>(
            &train_samples,
            config.batch_size,
            config.augment,
            &train_device,
        );

        let lr = warmup_then_cosine_lr(
            epoch,
            train_cfg.warmup_steps.min(total_steps),
            total_steps,
            train_cfg.lr,
            1e-6,
        );
        let (next_model, train_stats) =
            train_epoch(model, &train_batches, &loss_fn, lr, &mut optimizer);
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
        fs::remove_file(base).ok();
    }
}
