use std::collections::BTreeMap;
use std::time::Instant;

use burn::backend::libtorch::LibTorchDevice;
use burn::optim::{GradientsAccumulator, GradientsParams, Optimizer};
use colored::Colorize;
use hydra_train::config::PipelineState;
use hydra_train::data::pipeline::{DataManifest, StreamingLoaderConfig};
use hydra_train::model::HydraModelConfig;
use hydra_train::preflight::{LoaderRuntimeConfig, ProbeKind};
use hydra_train::selfplay::generate_self_play_rl_batch;
use hydra_train::training::distill::{DistillConfig, DistillState};
use hydra_train::training::drda::RebaseTracker;
use hydra_train::training::head_gates::{HeadActivationConfig, HeadActivationController};
use hydra_train::training::orchestrator::{
    live_exit_config_from_plan, maintenance_plan, rl_phase_train_step_with_controller,
};

use super::artifacts::RlArtifactPaths;
use super::config::{RlTrainConfig, TrainConfig, loader_runtime_config};
use super::config_runtime::rl_config_from_train_config;
use super::loss_policy::build_rl_loss_config;
use super::preflight_runtime::measure_samples_per_second;
use super::presentation::{
    format_preflight_summary_line, format_runtime_tuning_message, format_timed_phase_message,
    make_bar,
};
use super::probe_process::{execute_probe_request, rl_probe_result_path};
use super::probe_request::ProbeRequest;
use super::probe_summary::{best_probe_summary, format_probe_selection_summary};
use super::schedule::effective_lr;

pub(super) fn autotune_buffer_samples_candidates(config: &TrainConfig) -> Vec<usize> {
    let current = config.buffer_samples.max(1);
    let mut candidates = vec![
        current,
        current.saturating_mul(2),
        current.saturating_mul(4),
    ];
    candidates.retain(|value| *value > 0);
    candidates.sort_unstable();
    candidates.dedup();
    candidates
}

pub(super) fn autotune_buffer_games_candidates(config: &TrainConfig) -> Vec<usize> {
    let current = config.buffer_games.max(1);
    let mut candidates = vec![current, current.saturating_mul(2)];
    candidates.retain(|value| *value > 0);
    candidates.sort_unstable();
    candidates.dedup();
    candidates
}

pub(super) fn autotune_archive_queue_candidates(config: &TrainConfig) -> Vec<usize> {
    let current = config.archive_queue_bound.max(1);
    let mut candidates = vec![current / 2, current, current.saturating_mul(2)];
    candidates.retain(|value| *value > 0);
    candidates.sort_unstable();
    candidates.dedup();
    candidates
}

pub(super) fn autotune_rl_games_per_batch_candidates(rl: &RlTrainConfig) -> Vec<usize> {
    let current = rl.games_per_batch.max(1);
    let mut candidates = vec![
        current.saturating_mul(8),
        current.saturating_mul(4),
        current.saturating_mul(2),
        current,
        current / 2,
        1,
    ];
    candidates.retain(|value| *value > 0);
    candidates.dedup();
    candidates
}

pub(super) fn runtime_probe_loader_config(config: &TrainConfig) -> StreamingLoaderConfig {
    StreamingLoaderConfig {
        buffer_games: config.buffer_games,
        buffer_samples: config.buffer_samples,
        train_fraction: config.train_fraction,
        seed: config.seed,
        archive_queue_bound: config.archive_queue_bound,
        max_skip_logs_per_source: config.max_skip_logs_per_source,
        aggregate_skip_logs: true,
        exit_sidecar: None,
        exit_sidecar_source_net_hash: None,
        exit_sidecar_source_version: None,
    }
}

pub(super) fn measure_train_runtime_throughput(
    config: &TrainConfig,
    loader_config: &hydra_train::data::pipeline::StreamingLoaderConfig,
    manifest: &DataManifest,
    train_device: &LibTorchDevice,
) -> Result<f64, String> {
    let train_cfg = super::config::trainer_config_from_train_config(config);
    let mut model =
        hydra_train::model::HydraModelConfig::learner().init::<super::TrainBackend>(train_device);
    let mut optimizer = train_cfg.optimizer_config().init();
    let loss_fn = hydra_train::training::losses::HydraLoss::<super::TrainBackend>::new(
        super::loss_policy::build_loss_config(config.advanced_loss.as_ref())?,
    );
    let microbatch_size = config
        .microbatch_size
        .unwrap_or(config.batch_size)
        .min(config.batch_size)
        .max(1);
    let warmup_steps = config.preflight.warmup_steps.max(1);
    let measure_steps = config.preflight.measure_steps.max(1);
    let target_steps = warmup_steps + measure_steps;
    let mut completed_steps = 0usize;
    let mut pending_samples = std::collections::VecDeque::new();
    let mut measure_start = None;

    for buffer_result in
        hydra_train::data::pipeline::stream_train_epoch(manifest, loader_config, 0, None)
    {
        let buffer = buffer_result.map_err(|err| format!("runtime train stream failed: {err}"))?;
        pending_samples.extend(buffer);
        while pending_samples.len() >= config.batch_size {
            let logical_batch: Vec<hydra_train::data::sample::MjaiSample> =
                pending_samples.drain(..config.batch_size).collect();
            let logical_batch_len = logical_batch.len().max(1) as f32;
            let mut accumulator: GradientsAccumulator<
                hydra_train::model::HydraModel<super::TrainBackend>,
            > = GradientsAccumulator::new();
            for chunk in logical_batch.chunks(microbatch_size) {
                let Some((obs, targets)) = hydra_train::data::sample::collate_samples::<
                    super::TrainBackend,
                >(chunk, config.augment, train_device) else {
                    continue;
                };
                let output = model.forward(obs);
                let breakdown = loss_fn.total_loss(&output, &targets);
                let chunk_weight = chunk.len() as f32 / logical_batch_len;
                let grads = (breakdown.total * chunk_weight).backward();
                let grads = GradientsParams::from_grads(grads, &model);
                accumulator.accumulate(&model, grads);
            }
            let lr = effective_lr(&train_cfg, completed_steps, target_steps.max(1));
            let grads = accumulator.grads();
            model = optimizer.step(lr, model, grads);
            completed_steps += 1;
            if completed_steps == warmup_steps {
                measure_start = Some(Instant::now());
            }
            if completed_steps >= target_steps {
                let elapsed = measure_start
                    .map(|start| start.elapsed())
                    .unwrap_or_default();
                return Ok(measure_samples_per_second(
                    measure_steps * config.batch_size,
                    elapsed,
                ));
            }
        }
    }

    Err("not enough train data to finish runtime probe".to_string())
}

pub(super) fn measure_rl_runtime_throughput(
    config: &TrainConfig,
    rl: &RlTrainConfig,
    train_device: &LibTorchDevice,
) -> Result<f64, String> {
    let model_config = HydraModelConfig::learner();
    let mut model = model_config.init::<super::TrainBackend>(train_device);
    let mut optimizer = super::config::trainer_config_from_train_config(config)
        .optimizer_config()
        .init();
    let loss_fn = hydra_train::training::losses::HydraLoss::<super::TrainBackend>::new(
        build_rl_loss_config(config.advanced_loss.as_ref())?,
    );
    let rl_cfg = rl_config_from_train_config(rl);
    let mut state = PipelineState {
        phase: rl.phase.to_training_phase(),
        ..PipelineState::default()
    };
    let mut controller = HeadActivationController::new(HeadActivationConfig::default_with_params(
        model_config.estimated_params(),
    ));
    let mut rebase_tracker = RebaseTracker::default_phase2();
    let distill_state = DistillState::default();
    let distill_cfg = DistillConfig::fast_distill();
    let warmup_steps = config.preflight.warmup_steps.max(1);
    let measure_steps = config.preflight.measure_steps.max(1);
    let target_steps = warmup_steps + measure_steps;
    let mut completed_steps = 0usize;
    let mut measure_start = None;

    while completed_steps < target_steps {
        let elapsed_secs = completed_steps as u64;
        let plan = maintenance_plan(
            &state,
            &rebase_tracker,
            &distill_state,
            &distill_cfg,
            elapsed_secs,
            0.05,
        );
        let live_exit_cfg = live_exit_config_from_plan(&plan);
        let base_seed = config.seed.wrapping_add(completed_steps as u64 * 1009);
        let game_seeds: Vec<u64> = (0..rl.games_per_batch)
            .map(|idx| base_seed.wrapping_add(idx as u64))
            .collect();
        let batch = generate_self_play_rl_batch(
            &game_seeds,
            rl.temperature,
            base_seed,
            &model,
            train_device,
            &hydra_train::training::gae::GaeConfig::default(),
            live_exit_cfg,
        );

        controller.try_activate(hydra_train::training::head_gates::AdvancedHead::DeltaQ);
        let (next_model, _) = rl_phase_train_step_with_controller(
            &state,
            model,
            &batch,
            &rl_cfg,
            &loss_fn,
            &mut optimizer,
            Some(&mut controller),
        )
        .map_err(|err| format!("RL runtime probe step failed: {err}"))?;
        model = next_model;
        controller.tick_warmup();
        completed_steps += 1;
        state.total_games += rl.games_per_batch as u64;
        state.total_samples += batch.batch_size() as u64;
        state.increment_learner_version();
        rebase_tracker.tick(1.0);
        if completed_steps == warmup_steps {
            measure_start = Some(Instant::now());
        }
    }

    let elapsed = measure_start.map(|start| start.elapsed()).unwrap_or_default();
    Ok(measure_samples_per_second(measure_steps * rl.games_per_batch, elapsed))
}

pub(super) fn autotune_rl_games_per_batch(
    config_path: &std::path::Path,
    config: &TrainConfig,
    artifacts: &RlArtifactPaths,
    rl: &RlTrainConfig,
) -> Result<usize, String> {
    let candidates = autotune_rl_games_per_batch_candidates(rl);
    let progress = make_bar(
        candidates.len() as u64,
        "{spinner:.cyan} {msg} {wide_bar} {pos}/{len}",
    )?;
    let mut best = *candidates
        .first()
        .ok_or_else(|| "no RL games_per_batch candidates available".to_string())?;
    let mut best_score = f64::NEG_INFINITY;
    for &candidate in &candidates {
        progress.set_message(format_runtime_tuning_message(
            "rl_games_per_batch",
            candidate.to_string(),
            progress.position() as usize,
            progress.length().unwrap_or(1) as usize,
        ));
        let request = ProbeRequest {
            kind: ProbeKind::RlGames,
            candidate_microbatch: candidate,
            warmup_steps: config.preflight.warmup_steps,
            measure_steps: config.preflight.measure_steps,
        };
        let result_path = rl_probe_result_path(artifacts, ProbeKind::RlGames, candidate, 0);
        let result = execute_probe_request(
            config_path,
            request,
            &result_path,
            super::preflight_runtime::classify_probe_detail,
        )?;
        progress.inc(1);
        println!(
            "{}",
            super::presentation::format_probe_status_line(&result)
        );
        if result.status != hydra_train::preflight::ProbeStatus::Success {
            continue;
        }
        let score = result.measured_samples_per_second.unwrap_or(0.0);
        if score > best_score {
            best = candidate;
            best_score = score;
        }
        println!(
            "{}",
            format_runtime_tuning_message(
                "rl_games_per_batch",
                format!(
                    "candidate={} throughput={:.2} games/s best={} ({:.2} games/s)",
                    candidate, score, best, best_score
                ),
                progress.position() as usize,
                progress.length().unwrap_or(1) as usize,
            )
        );
    }
    progress.finish_with_message("runtime tuning rl_games_per_batch complete".green().to_string());
    let summaries = candidates
        .iter()
        .copied()
        .map(|candidate| hydra_train::preflight::ProbeResult {
            kind: ProbeKind::RlGames,
            candidate_microbatch: candidate,
            status: if candidate == best && best_score.is_finite() {
                hydra_train::preflight::ProbeStatus::Success
            } else {
                hydra_train::preflight::ProbeStatus::Oom
            },
            measured_samples_per_second: if candidate == best && best_score.is_finite() {
                Some(best_score)
            } else {
                None
            },
            elapsed_seconds: None,
            detail: String::new(),
        })
        .collect::<Vec<_>>();
    if let Some(summary) = best_probe_summary(&summaries) {
        println!(
            "{}",
            format_preflight_summary_line(
                "RL Preflight:",
                format_probe_selection_summary(ProbeKind::RlGames, &summary)
            )
        );
    }
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::autotune_rl_games_per_batch_candidates;
    use crate::config::RlTrainConfig;
    use crate::preflight_runtime::classify_probe_detail;
    use hydra_train::preflight::ProbeStatus;

    #[test]
    fn rl_games_per_batch_candidates_are_sorted_and_include_current() {
        let candidates = autotune_rl_games_per_batch_candidates(&RlTrainConfig {
            games_per_batch: 4,
            ..Default::default()
        });
        assert_eq!(candidates, vec![32, 16, 8, 4, 2, 1]);
    }

    #[test]
    fn rl_runtime_autotune_uses_probe_oom_classification() {
        assert_eq!(classify_probe_detail("CUDA out of memory"), ProbeStatus::Oom);
        assert_eq!(classify_probe_detail("oom while probing rl batch"), ProbeStatus::Oom);
    }
}

pub(super) fn tune_runtime_knob<T, F>(
    base: &TrainConfig,
    knob_name: &str,
    candidates: &[T],
    display: impl Fn(T) -> String,
    apply: impl Fn(&mut TrainConfig, T),
    score: &mut F,
) -> Result<T, String>
where
    T: Copy + Eq,
    F: FnMut(&TrainConfig) -> Result<f64, String>,
{
    let progress = make_bar(
        candidates.len() as u64,
        "{spinner:.cyan} {msg} {wide_bar} {pos}/{len}",
    )?;
    let mut best = candidates
        .first()
        .copied()
        .ok_or_else(|| format!("no candidates available for {knob_name}"))?;
    let mut best_score = f64::NEG_INFINITY;

    for candidate in candidates {
        progress.set_message(format_runtime_tuning_message(
            knob_name,
            display(*candidate),
            progress.position() as usize,
            progress.length().unwrap_or(1) as usize,
        ));
        let mut tuned = base.clone();
        apply(&mut tuned, *candidate);
        let candidate_score = score(&tuned)?;
        progress.inc(1);
        if candidate_score > best_score {
            best = *candidate;
            best_score = candidate_score;
        }
        println!(
            "{}",
            format_runtime_tuning_message(
                knob_name,
                format!(
                    "candidate={} throughput={:.2} samples/s best={} ({:.2} samples/s)",
                    display(*candidate),
                    candidate_score,
                    display(best),
                    best_score,
                ),
                progress.position() as usize,
                progress.length().unwrap_or(1) as usize,
            )
        );
    }
    progress.finish_with_message(
        format!("runtime tuning {knob_name} complete")
            .green()
            .to_string(),
    );
    Ok(best)
}

pub(super) fn score_runtime_tuple(
    config: &TrainConfig,
    manifest: &DataManifest,
    train_device: &LibTorchDevice,
    cache: &mut BTreeMap<(usize, usize, usize), Vec<f64>>,
) -> Result<f64, String> {
    let key = (
        config.archive_queue_bound,
        config.buffer_samples,
        config.buffer_games,
    );
    if let Some(scores) = cache.get(&key)
        && !scores.is_empty()
    {
        return Ok(score_tuple_samples_mean(scores));
    }
    let loader = runtime_probe_loader_config(config);
    let score = measure_train_runtime_throughput(config, &loader, manifest, train_device)?;
    cache.insert(key, vec![score]);
    Ok(score)
}

pub(super) fn push_runtime_tuple_sample(
    config: &TrainConfig,
    manifest: &DataManifest,
    train_device: &LibTorchDevice,
    cache: &mut BTreeMap<(usize, usize, usize), Vec<f64>>,
) -> Result<f64, String> {
    let key = (
        config.archive_queue_bound,
        config.buffer_samples,
        config.buffer_games,
    );
    let loader = runtime_probe_loader_config(config);
    let sample = measure_train_runtime_throughput(config, &loader, manifest, train_device)?;
    let samples = cache.entry(key).or_default();
    samples.push(sample);
    Ok(score_tuple_samples_mean(samples))
}

pub(super) fn score_tuple_samples_mean(scores: &[f64]) -> f64 {
    if scores.is_empty() {
        return 0.0;
    }
    scores.iter().sum::<f64>() / scores.len() as f64
}

pub(super) fn autotune_loader_runtime(
    config: &TrainConfig,
    manifest: &DataManifest,
    train_device: &LibTorchDevice,
) -> Result<LoaderRuntimeConfig, String> {
    let runtime_tuning_started = Instant::now();
    let mut tuned = config.clone();
    tuned.num_threads = loader_runtime_config(&tuned).num_threads;

    if let Some(num_threads) = tuned.num_threads
        && num_threads == 0
    {
        return Err("runtime autotune produced invalid num_threads=0".to_string());
    }

    let mut score_cache: BTreeMap<(usize, usize, usize), Vec<f64>> = BTreeMap::new();

    let queue_candidates = autotune_archive_queue_candidates(&tuned);
    let sample_candidates = autotune_buffer_samples_candidates(&tuned);
    let game_candidates = autotune_buffer_games_candidates(&tuned);

    let mut best_score = f64::NEG_INFINITY;
    let mut best_tuple = (
        tuned.archive_queue_bound,
        tuned.buffer_samples,
        tuned.buffer_games,
    );
    let mut coarse_scores = Vec::new();
    let coarse_started = Instant::now();
    println!(
        "{}",
        format_timed_phase_message(
            "runtime_coarse_search",
            &format!(
                "starting tuples={}",
                queue_candidates.len() * sample_candidates.len() * game_candidates.len()
            ),
            0.0,
        )
    );

    let coarse_progress = make_bar(
        (queue_candidates.len() * sample_candidates.len() * game_candidates.len()) as u64,
        "{spinner:.cyan} {msg} {wide_bar} {pos}/{len}",
    )?;
    for queue in &queue_candidates {
        for samples in &sample_candidates {
            for games in &game_candidates {
                coarse_progress.set_message(format_runtime_tuning_message(
                    "coarse_search",
                    format!("q={queue}, samples={samples}, games={games}"),
                    coarse_progress.position() as usize,
                    coarse_progress.length().unwrap_or(1) as usize,
                ));
                let mut candidate = tuned.clone();
                candidate.archive_queue_bound = *queue;
                candidate.buffer_samples = *samples;
                candidate.buffer_games = *games;
                let score =
                    score_runtime_tuple(&candidate, manifest, train_device, &mut score_cache)?;
                coarse_progress.inc(1);
                coarse_scores.push(((*queue, *samples, *games), score));
                if score > best_score {
                    best_score = score;
                    best_tuple = (*queue, *samples, *games);
                }
            }
        }
    }
    coarse_progress.finish_with_message("runtime coarse search complete".green().to_string());
    println!(
        "{}",
        format_timed_phase_message(
            "runtime_coarse_search",
            "complete",
            coarse_started.elapsed().as_secs_f64(),
        )
    );

    coarse_scores.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let close_tuples = coarse_scores
        .iter()
        .filter(|(_, score)| {
            *score >= best_score * (1.0 - config.preflight.loader_tuple_margin_ratio)
        })
        .take(2)
        .map(|(tuple, _)| *tuple)
        .collect::<Vec<_>>();
    if close_tuples.len() >= 2 {
        let refine_started = Instant::now();
        println!(
            "{}",
            format_preflight_summary_line(
                "Runtime refine:",
                format!(
                    "close_tuples={:?} extra_samples={}",
                    close_tuples,
                    config.preflight.loader_tuple_extra_samples.max(1)
                )
            )
        );
        for tuple in &close_tuples {
            let mut candidate = tuned.clone();
            candidate.archive_queue_bound = tuple.0;
            candidate.buffer_samples = tuple.1;
            candidate.buffer_games = tuple.2;
            for _ in 0..config.preflight.loader_tuple_extra_samples.max(1) {
                let averaged = push_runtime_tuple_sample(
                    &candidate,
                    manifest,
                    train_device,
                    &mut score_cache,
                )?;
                if averaged > best_score {
                    best_score = averaged;
                    best_tuple = *tuple;
                }
            }
        }
        println!(
            "{}",
            format_timed_phase_message(
                "runtime_refine",
                "complete",
                refine_started.elapsed().as_secs_f64(),
            )
        );
    }

    tuned.archive_queue_bound = best_tuple.0;
    tuned.buffer_samples = best_tuple.1;
    tuned.buffer_games = best_tuple.2;

    for _round in 0..config.preflight.loader_runtime_rounds.max(1) {
        let mut score = |candidate: &TrainConfig| {
            score_runtime_tuple(candidate, manifest, train_device, &mut score_cache)
        };

        let queue_candidates = autotune_archive_queue_candidates(&tuned);
        tuned.archive_queue_bound = tune_runtime_knob(
            &tuned,
            "archive_queue_bound",
            &queue_candidates,
            |value| value.to_string(),
            |cfg, value| cfg.archive_queue_bound = value,
            &mut score,
        )?;

        let sample_candidates = autotune_buffer_samples_candidates(&tuned);
        tuned.buffer_samples = tune_runtime_knob(
            &tuned,
            "buffer_samples",
            &sample_candidates,
            |value| value.to_string(),
            |cfg, value| cfg.buffer_samples = value,
            &mut score,
        )?;

        let game_candidates = autotune_buffer_games_candidates(&tuned);
        tuned.buffer_games = tune_runtime_knob(
            &tuned,
            "buffer_games",
            &game_candidates,
            |value| value.to_string(),
            |cfg, value| cfg.buffer_games = value,
            &mut score,
        )?;
    }

    println!(
        "{}",
        format_timed_phase_message(
            "runtime_tuning_total",
            "complete",
            runtime_tuning_started.elapsed().as_secs_f64(),
        )
    );

    Ok(loader_runtime_config(&tuned))
}
