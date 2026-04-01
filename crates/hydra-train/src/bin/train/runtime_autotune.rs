use std::collections::BTreeMap;
use std::time::Instant;
use burn::backend::libtorch::LibTorchDevice;
use burn::tensor::backend::AutodiffBackend;
use colored::Colorize;
use hydra_train::config::PipelineState;
use hydra_train::data::pipeline::{DataManifest, StreamingLoaderConfig};
use hydra_train::model::HydraModelConfig;
use hydra_train::preflight::LoaderRuntimeConfig;
use hydra_train::selfplay::{
    CooperativeSelfPlayCoordinator, CooperativeSelfPlayRequest, generate_self_play_rl_batch_reuse,
};
use hydra_train::training::distill::{DistillConfig, DistillState};
use hydra_train::training::drda::RebaseTracker;
use hydra_train::training::head_gates::{HeadActivationConfig, HeadActivationController};
use hydra_train::training::orchestrator::{
    live_exit_config_from_plan, maintenance_plan, rl_phase_train_step_with_controller,
};

use super::config::{RlTrainConfig, TrainConfig, loader_runtime_config};
use super::config_runtime::rl_config_from_train_config;
use super::loss_policy::build_rl_loss_config;
use super::preflight_runtime::{measure_samples_per_second, run_train_measurement_loop, TrainMeasurementSpec};
use super::presentation::{
    format_preflight_summary_line, format_runtime_tuning_message, format_timed_phase_message,
    make_bar,
};
use super::TrainBackend;

pub(super) type RuntimeTuple = (usize, usize, usize);

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(super) struct RuntimeTupleStats {
    pub count: usize,
    pub sum: f64,
}

impl RuntimeTupleStats {
    fn push(self, sample: f64) -> Self {
        Self {
            count: self.count + 1,
            sum: self.sum + sample,
        }
    }

    fn mean(self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct RankedLoaderRuntime {
    pub(super) loader: LoaderRuntimeConfig,
    pub(super) tuple: RuntimeTuple,
    pub(super) train_samples_per_second: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct LoaderRuntimeScoreSeed {
    pub(super) train_microbatch_size: usize,
    pub(super) tuple: RuntimeTuple,
    pub(super) warmup_steps: usize,
    pub(super) measure_steps: usize,
    pub(super) stats: RuntimeTupleStats,
}

fn apply_runtime_tuple(config: &mut TrainConfig, tuple: RuntimeTuple) {
    config.archive_queue_bound = tuple.0;
    config.buffer_samples = tuple.1;
    config.buffer_games = tuple.2;
}

fn current_runtime_tuple(config: &TrainConfig) -> RuntimeTuple {
    (
        config.archive_queue_bound,
        config.buffer_samples,
        config.buffer_games,
    )
}

fn current_train_runtime_microbatch(config: &TrainConfig) -> usize {
    config
        .microbatch_size
        .unwrap_or(config.batch_size)
        .min(config.batch_size)
        .max(1)
}

fn loader_runtime_from_tuple(config: &TrainConfig, tuple: RuntimeTuple) -> LoaderRuntimeConfig {
    LoaderRuntimeConfig {
        num_threads: Some(super::config::default_num_threads_for_system())
            .filter(|_| config.num_threads.is_none())
            .or(config.num_threads),
        buffer_games: tuple.2,
        buffer_samples: tuple.1,
        archive_queue_bound: tuple.0,
    }
}

fn loader_runtime_score_seed_matches(
    config: &TrainConfig,
    seed: LoaderRuntimeScoreSeed,
) -> bool {
    seed.train_microbatch_size == current_train_runtime_microbatch(config)
        && seed.tuple == current_runtime_tuple(config)
        && seed.warmup_steps == runtime_autotune_warmup_steps(config)
        && seed.measure_steps == runtime_autotune_measure_steps(config)
        && seed.stats.count > 0
}

fn seed_runtime_score_cache(
    config: &TrainConfig,
    cache: &mut BTreeMap<RuntimeTuple, RuntimeTupleStats>,
    seed: Option<LoaderRuntimeScoreSeed>,
) {
    let Some(seed) = seed.filter(|seed| loader_runtime_score_seed_matches(config, *seed)) else {
        return;
    };
    cache.entry(seed.tuple).or_insert(seed.stats);
}

fn should_refine_close_tuples(close_tuples: &[RuntimeTuple]) -> bool {
    close_tuples.len() >= 2
}

fn coarse_search_candidate_count(
    queue_candidates: &[usize],
    sample_candidates: &[usize],
    game_candidates: &[usize],
) -> usize {
    queue_candidates.len() * sample_candidates.len() * game_candidates.len()
}

fn validate_runtime_threads(config: &TrainConfig) -> Result<(), String> {
    if matches!(config.num_threads, Some(0)) {
        return Err("runtime autotune produced invalid num_threads=0".to_string());
    }
    Ok(())
}

fn should_start_measurement(completed_steps: usize, warmup_steps: usize) -> bool {
    completed_steps == warmup_steps
}

fn should_count_measured_samples(completed_steps: usize, warmup_steps: usize) -> bool {
    completed_steps > warmup_steps
}

fn runtime_autotune_warmup_steps(config: &TrainConfig) -> usize {
    config.preflight.warmup_steps.max(1)
}

fn runtime_autotune_measure_steps(config: &TrainConfig) -> usize {
    config.preflight.measure_steps.max(1)
}

#[cfg(test)]
fn measured_train_samples(measure_steps: usize, batch_size: usize) -> usize {
    measure_steps * batch_size
}

fn finalize_runtime_probe_throughput(
    measure_start: Option<Instant>,
    measured_samples: usize,
) -> f64 {
    let elapsed = measure_start
        .map(|start| start.elapsed())
        .unwrap_or_default();
    measure_samples_per_second(measured_samples, elapsed)
}

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
        delta_q_sidecar: None,
        delta_q_sidecar_source_net_hash: None,
        delta_q_sidecar_source_version: None,
    }
}

fn runtime_probe_model_config() -> HydraModelConfig {
    #[cfg(test)]
    {
        HydraModelConfig::new(1)
            .with_input_channels(hydra_train::config::INPUT_CHANNELS)
            .with_hidden_channels(4)
            .with_num_groups(4)
            .with_se_bottleneck(1)
    }

    #[cfg(not(test))]
    {
        HydraModelConfig::learner()
    }
}

fn measure_train_runtime_throughput_for_backend<B>(
    config: &TrainConfig,
    loader_config: &hydra_train::data::pipeline::StreamingLoaderConfig,
    manifest: &DataManifest,
    train_device: &LibTorchDevice,
) -> Result<f64, String>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
{
    run_train_measurement_loop::<B>(TrainMeasurementSpec {
        config,
        model_config: &runtime_probe_model_config(),
        candidate_microbatch: current_train_runtime_microbatch(config),
        warmup_steps: runtime_autotune_warmup_steps(config),
        measure_steps: runtime_autotune_measure_steps(config),
        loader_config,
        manifest,
        train_device,
        on_start: Box::new(|_candidate_microbatch, _warmup_steps, _measure_steps| Ok(())),
        on_step: Box::new(|_completed_steps, _candidate_microbatch, _request, _measure_start| Ok(())),
        on_measure_start: Box::new(|_candidate_microbatch, _measure_steps| Ok(())),
        insufficient_data: Box::new(|_candidate_microbatch| {
            "not enough train data to finish runtime probe".to_string()
        }),
    })
}

pub(super) fn measure_train_runtime_throughput(
    config: &TrainConfig,
    loader_config: &hydra_train::data::pipeline::StreamingLoaderConfig,
    manifest: &DataManifest,
    train_device: &LibTorchDevice,
) -> Result<f64, String> {
    match config.precision_mode {
        crate::config::PrecisionMode::Fp32 => measure_train_runtime_throughput_for_backend::<
            TrainBackend,
        >(
            config, loader_config, manifest, train_device
        ),
        crate::config::PrecisionMode::Bf16Autocast => {
            measure_train_runtime_throughput_for_backend::<TrainBackend>(
                config,
                loader_config,
                manifest,
                train_device,
            )
        }
    }
}

pub(super) fn measure_rl_runtime_throughput(
    config: &TrainConfig,
    rl: &RlTrainConfig,
    train_device: &LibTorchDevice,
) -> Result<f64, String> {
    match config.precision_mode {
        crate::config::PrecisionMode::Fp32 => {
            measure_rl_runtime_throughput_for_backend::<super::TrainBackend>(
                config,
                rl,
                train_device,
            )
        }
        crate::config::PrecisionMode::Bf16Autocast => {
            measure_rl_runtime_throughput_for_backend::<super::TrainBackend>(
                config,
                rl,
                train_device,
            )
        }
    }
}

fn measure_rl_runtime_throughput_for_backend<B>(
    config: &TrainConfig,
    rl: &RlTrainConfig,
    train_device: &LibTorchDevice,
) -> Result<f64, String>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
{
    let model_config = HydraModelConfig::learner();
    let mut model = model_config.init::<B>(train_device);
    let mut optimizer = super::config::trainer_config_from_train_config(config)
        .optimizer_config()
        .init();
    let loss_fn = hydra_train::training::losses::HydraLoss::<B>::new(
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
    let mut self_play_coordinator = CooperativeSelfPlayCoordinator::new();
    let warmup_steps = config.preflight.warmup_steps.max(1);
    let measure_steps = config.preflight.measure_steps.max(1);
    let target_steps = warmup_steps + measure_steps;
    let mut completed_steps = 0usize;
    let mut measure_start = None;
    let mut measure_samples = 0usize;

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
        let batch = generate_self_play_rl_batch_reuse(
            &mut self_play_coordinator,
            CooperativeSelfPlayRequest {
                game_seeds: &game_seeds,
                temperature: rl.temperature,
                rng_seed: base_seed,
                live_exit_cfg,
            },
            &model,
            train_device,
            &hydra_train::training::gae::GaeConfig::default(),
        );

        controller.try_activate(hydra_train::training::head_gates::AdvancedHead::DeltaQ);
        let batch_samples = batch.batch_size();
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
        state.total_samples += batch_samples as u64;
        state.increment_learner_version();
        rebase_tracker.tick(1.0);
        if should_start_measurement(completed_steps, warmup_steps) {
            measure_start = Some(Instant::now());
        }
        if should_count_measured_samples(completed_steps, warmup_steps) {
            measure_samples += batch_samples;
        }
    }

    Ok(finalize_runtime_probe_throughput(
        measure_start,
        measure_samples,
    ))
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
    let mut tuned = base.clone();

    for candidate in candidates {
        progress.set_message(format_runtime_tuning_message(
            knob_name,
            display(*candidate),
            progress.position() as usize,
            progress.length().unwrap_or(1) as usize,
        ));
        tuned.clone_from(base);
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
                format_runtime_knob_candidate_summary(
                    &display(*candidate),
                    candidate_score,
                    &display(best),
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

pub(super) fn format_runtime_knob_candidate_summary(
    candidate: &str,
    candidate_score: f64,
    best: &str,
    best_score: f64,
) -> String {
    format!(
        "candidate={candidate} throughput={candidate_score:.2} samples/s best={best} ({best_score:.2} samples/s)"
    )
}

pub(super) fn runtime_tuple_key(config: &TrainConfig) -> RuntimeTuple {
    current_runtime_tuple(config)
}

pub(super) fn rank_runtime_tuple_scores(scores: &mut [(RuntimeTuple, f64)]) {
    scores.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

pub(super) fn close_runtime_tuples(
    scores: &[(RuntimeTuple, f64)],
    best_score: f64,
    margin_ratio: f64,
    max_candidates: usize,
) -> Vec<RuntimeTuple> {
    let minimum_score = best_score * (1.0 - margin_ratio);
    scores
        .iter()
        .filter(|(_, score)| *score >= minimum_score)
        .take(max_candidates)
        .map(|(tuple, _)| *tuple)
        .collect()
}

pub(super) fn runtime_refine_sample_budget(extra_samples: usize) -> usize {
    extra_samples.max(1)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RuntimeRefineTopUpPlan {
    target_total_count: usize,
    missing_samples: usize,
}

fn runtime_refine_top_up_plan(
    stats: RuntimeTupleStats,
    extra_samples: usize,
) -> RuntimeRefineTopUpPlan {
    let target_total_count = runtime_refine_sample_budget(extra_samples).saturating_add(1);
    RuntimeRefineTopUpPlan {
        target_total_count,
        missing_samples: target_total_count.saturating_sub(stats.count),
    }
}

pub(super) fn format_runtime_refine_summary(
    close_tuples: &[RuntimeTuple],
    extra_samples: usize,
) -> String {
    format_preflight_summary_line(
        "Runtime refine:",
        format!(
            "close_tuples={close_tuples:?} extra_samples={}",
            runtime_refine_sample_budget(extra_samples)
        ),
    )
}

pub(super) fn score_runtime_tuple(
    config: &TrainConfig,
    manifest: &DataManifest,
    train_device: &LibTorchDevice,
    cache: &mut BTreeMap<(usize, usize, usize), RuntimeTupleStats>,
) -> Result<f64, String> {
    let key = runtime_tuple_key(config);
    if let Some(stats) = cache.get(&key)
        && stats.count > 0
    {
        return Ok(stats.mean());
    }
    let loader = runtime_probe_loader_config(config);
    let score = measure_train_runtime_throughput(config, &loader, manifest, train_device)?;
    cache.insert(
        key,
        RuntimeTupleStats {
            count: 1,
            sum: score,
        },
    );
    Ok(score)
}

pub(super) fn push_runtime_tuple_sample(
    config: &TrainConfig,
    manifest: &DataManifest,
    train_device: &LibTorchDevice,
    cache: &mut BTreeMap<(usize, usize, usize), RuntimeTupleStats>,
) -> Result<f64, String> {
    let key = runtime_tuple_key(config);
    let loader = runtime_probe_loader_config(config);
    let sample = measure_train_runtime_throughput(config, &loader, manifest, train_device)?;
    let stats = cache.entry(key).or_default();
    *stats = stats.push(sample);
    Ok(stats.mean())
}

fn refine_close_runtime_tuples<F>(
    tuned: &TrainConfig,
    close_tuples: &[RuntimeTuple],
    extra_samples: usize,
    score_cache: &mut BTreeMap<(usize, usize, usize), RuntimeTupleStats>,
    best_score: &mut f64,
    best_tuple: &mut RuntimeTuple,
    mut push_sample: F,
) -> Result<(), String>
where
    F: FnMut(
        &TrainConfig,
        &mut BTreeMap<(usize, usize, usize), RuntimeTupleStats>,
    ) -> Result<f64, String>,
{
    let mut candidate = tuned.clone();
    for tuple in close_tuples {
        candidate.clone_from(tuned);
        apply_runtime_tuple(&mut candidate, *tuple);
        let top_up_plan = runtime_refine_top_up_plan(
            score_cache.get(tuple).copied().unwrap_or_default(),
            extra_samples,
        );
        for _ in 0..top_up_plan.missing_samples {
            let averaged = push_sample(&candidate, score_cache)?;
            if averaged > *best_score {
                *best_score = averaged;
                *best_tuple = *tuple;
            }
        }
    }
    Ok(())
}

fn compare_runtime_tuple_scores(
    left: &(RuntimeTuple, RuntimeTupleStats),
    right: &(RuntimeTuple, RuntimeTupleStats),
) -> std::cmp::Ordering {
    right
        .1
        .mean()
        .partial_cmp(&left.1.mean())
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| left.0.cmp(&right.0))
}

fn ranked_loader_runtime_from_score_cache(
    base_config: &TrainConfig,
    tuned_config: &TrainConfig,
    score_cache: &BTreeMap<RuntimeTuple, RuntimeTupleStats>,
    limit: usize,
) -> Vec<RankedLoaderRuntime> {
    let shortlist_limit = shortlist_limit(limit);
    let tuned_loader = loader_runtime_config(tuned_config);
    let current_tuple = current_runtime_tuple(tuned_config);
    let mut ranked_tuples = score_cache
        .iter()
        .filter(|(_, stats)| stats.count > 0)
        .map(|(tuple, stats)| (*tuple, *stats))
        .collect::<Vec<_>>();
    ranked_tuples.sort_by(compare_runtime_tuple_scores);

    let mut seen = std::collections::BTreeSet::new();
    let mut ranked = ranked_tuples
        .into_iter()
        .filter_map(|(tuple, stats)| {
            seen.insert(tuple).then_some(RankedLoaderRuntime {
                loader: if tuple == current_tuple {
                    tuned_loader
                } else {
                    loader_runtime_from_tuple(base_config, tuple)
                },
                tuple,
                train_samples_per_second: stats.mean(),
            })
        })
        .take(shortlist_limit)
        .collect::<Vec<_>>();

    if ranked.is_empty() {
        ranked.push(RankedLoaderRuntime {
            loader: tuned_loader,
            tuple: current_tuple,
            train_samples_per_second: 0.0,
        });
    }

    ranked
}

#[cfg(test)]
pub(super) fn autotune_loader_runtime(
    config: &TrainConfig,
    manifest: &DataManifest,
    train_device: &LibTorchDevice,
) -> Result<LoaderRuntimeConfig, String> {
    Ok(
        autotune_ranked_loader_runtime(config, manifest, train_device, 1)?
            .into_iter()
            .next()
            .map(|ranked| ranked.loader)
            .unwrap_or_else(|| loader_runtime_config(config)),
    )
}

#[cfg(test)]
pub(super) fn autotune_ranked_loader_runtime(
    config: &TrainConfig,
    manifest: &DataManifest,
    train_device: &LibTorchDevice,
    limit: usize,
) -> Result<Vec<RankedLoaderRuntime>, String> {
    autotune_ranked_loader_runtime_with_seed(config, manifest, train_device, limit, None)
}

pub(super) fn autotune_ranked_loader_runtime_with_seed(
    config: &TrainConfig,
    manifest: &DataManifest,
    train_device: &LibTorchDevice,
    limit: usize,
    score_seed: Option<LoaderRuntimeScoreSeed>,
) -> Result<Vec<RankedLoaderRuntime>, String> {
    if config.preflight.loader_runtime_rounds == 0
        && config.preflight.loader_tuple_extra_samples == 0
        && limit <= 1
    {
        let loader = loader_runtime_config(config);
        return Ok(vec![RankedLoaderRuntime {
            loader,
            tuple: current_runtime_tuple(config),
            train_samples_per_second: 0.0,
        }]);
    }

    let runtime_tuning_started = Instant::now();
    let mut tuned = config.clone();
    tuned.num_threads = loader_runtime_config(&tuned).num_threads;
    validate_runtime_threads(&tuned)?;

    let mut score_cache: BTreeMap<(usize, usize, usize), RuntimeTupleStats> = BTreeMap::new();
    seed_runtime_score_cache(&tuned, &mut score_cache, score_seed);

    let queue_candidates = autotune_archive_queue_candidates(&tuned);
    let sample_candidates = autotune_buffer_samples_candidates(&tuned);
    let game_candidates = autotune_buffer_games_candidates(&tuned);

    let mut best_score = f64::NEG_INFINITY;
    let mut best_tuple = current_runtime_tuple(&tuned);
    let mut coarse_scores = Vec::new();
    let coarse_started = Instant::now();
    println!(
        "{}",
        format_timed_phase_message(
            "runtime_coarse_search",
            &format!(
                "starting tuples={}",
                coarse_search_candidate_count(
                    &queue_candidates,
                    &sample_candidates,
                    &game_candidates
                )
            ),
            0.0,
        )
    );

    let coarse_progress = make_bar(
        coarse_search_candidate_count(&queue_candidates, &sample_candidates, &game_candidates)
            as u64,
        "{spinner:.cyan} {msg} {wide_bar} {pos}/{len}",
    )?;
    let mut candidate = tuned.clone();
    for queue in &queue_candidates {
        for samples in &sample_candidates {
            for games in &game_candidates {
                coarse_progress.set_message(format_runtime_tuning_message(
                    "coarse_search",
                    format!("q={queue}, samples={samples}, games={games}"),
                    coarse_progress.position() as usize,
                    coarse_progress.length().unwrap_or(1) as usize,
                ));
                candidate.clone_from(&tuned);
                apply_runtime_tuple(&mut candidate, (*queue, *samples, *games));
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

    rank_runtime_tuple_scores(&mut coarse_scores);
    let close_tuples = close_runtime_tuples(
        &coarse_scores,
        best_score,
        config.preflight.loader_tuple_margin_ratio,
        2,
    );
    if should_refine_close_tuples(&close_tuples) {
        let refine_started = Instant::now();
        println!(
            "{}",
            format_runtime_refine_summary(
                &close_tuples,
                config.preflight.loader_tuple_extra_samples
            )
        );
        refine_close_runtime_tuples(
            &tuned,
            &close_tuples,
            config.preflight.loader_tuple_extra_samples,
            &mut score_cache,
            &mut best_score,
            &mut best_tuple,
            |candidate, cache| {
                push_runtime_tuple_sample(candidate, manifest, train_device, cache)
            },
        )?;
        println!(
            "{}",
            format_timed_phase_message(
                "runtime_refine",
                "complete",
                refine_started.elapsed().as_secs_f64(),
            )
        );
    }

    apply_runtime_tuple(&mut tuned, best_tuple);

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

    Ok(ranked_loader_runtime_from_score_cache(
        config,
        &tuned,
        &score_cache,
        limit,
    ))
}

fn shortlist_limit(limit: usize) -> usize {
    limit.max(1)
}

#[cfg(test)]
#[path = "runtime_autotune/tests.rs"]
mod tests;
