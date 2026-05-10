//! Preflight execution cache and selected-runtime helpers.
//!
//! Heavy probe loops still migrate incrementally, but cache identity, hit/miss
//! handling, and bootstrap runtime application live at the exec boundary so the
//! runtime crate remains protocol/fingerprint only.

use colored::Colorize;
use hydra_model::model::HydraModelConfig;
use hydra_train_runtime::config::{TrainConfig, default_num_threads_for_system};
use hydra_train_runtime::preflight::{
    EffectiveRuntimeConfig, ExplicitSettings, PreflightCacheEntry, PreflightCacheKey, ProbeResult,
    preflight_cache_key,
};
use std::path::Path;

use crate::advisory::RuntimeAdvisory;
use crate::artifacts::{
    BcArtifactPaths, PreflightBenchmarkPaths, PreflightBenchmarkReport, PreflightPaths,
    RlArtifactPaths, RlPreflightPaths, read_preflight_cache, write_preflight_benchmark_report,
    write_preflight_cache,
};
use crate::presentation::{format_preflight_summary_line, timestamped};
use crate::resume::ResumeContext;

/// Selected BC preflight execution result passed back to mode dispatch.
#[derive(Debug, Clone)]
pub struct PreflightRuntime {
    /// Effective runtime selected by cache/probes/benchmark.
    pub runtime: EffectiveRuntimeConfig,
    /// Probe results for train microbatch search; empty on cache hit.
    pub train_probe_results: Vec<ProbeResult>,
    /// Probe results for validation microbatch search; empty on cache hit.
    pub validation_probe_results: Vec<ProbeResult>,
    /// Stage-2 benchmark result when one was run and persisted.
    pub benchmark: Option<hydra_train_runtime::preflight::BenchmarkResult>,
    /// Runtime advisories derived from observed probe/benchmark data.
    pub advisories: Vec<RuntimeAdvisory>,
    /// Whether train/validation microbatch values were explicit in YAML.
    pub explicit: ExplicitSettings,
}

/// Selected RL preflight execution result passed back to mode dispatch.
#[derive(Debug, Clone)]
pub struct RlPreflightRuntime {
    /// Selected RL games per batch.
    pub selected_games_per_batch: usize,
    /// Selected RL microbatch size.
    pub selected_microbatch_size: usize,
    /// RL games probe results; empty on cache hit.
    pub rl_games_probe_results: Vec<ProbeResult>,
    /// RL microbatch probe results; empty on cache hit.
    pub rl_microbatch_probe_results: Vec<ProbeResult>,
}

/// Immutable BC preflight cache lookup context.
pub struct BcPreflightCacheContext {
    /// Cache key computed from selected-runtime inputs.
    pub cache_key: PreflightCacheKey,
    /// Paths used by BC preflight.
    pub paths: PreflightPaths,
    /// Explicit microbatch settings from config.
    pub explicit: ExplicitSettings,
}

/// Computes BC preflight cache context without executing probes.
pub fn bc_preflight_cache_context(
    config: &TrainConfig,
    model_config: &HydraModelConfig,
    device_label: &str,
    artifacts: &BcArtifactPaths,
) -> BcPreflightCacheContext {
    BcPreflightCacheContext {
        cache_key: preflight_cache_key(
            config,
            model_config,
            device_label,
            default_num_threads_for_system(),
        ),
        paths: PreflightPaths::new(artifacts),
        explicit: ExplicitSettings {
            train_microbatch_explicit: config.microbatch_size.is_some(),
            validation_microbatch_explicit: config.validation_microbatch_size.is_some(),
        },
    }
}

/// Returns a matching BC preflight cache entry, preserving cache-hit probe-skip semantics.
pub fn matching_bc_preflight_cache(
    context: &BcPreflightCacheContext,
) -> Result<Option<PreflightCacheEntry>, String> {
    let Some(cached) = read_preflight_cache(&context.paths.cache_path)? else {
        return Ok(None);
    };
    if cached.cache_key == context.cache_key {
        println!(
            "{}",
            format_preflight_summary_line(
                "Preflight cache hit:",
                format!(
                    "reusing cached probe results train_mb={} val_mb={} -- re-benchmarking to verify",
                    cached.runtime.selected.train_microbatch_size,
                    cached.runtime.selected.validation_microbatch_size,
                ),
            )
        );
        Ok(Some(cached))
    } else {
        Ok(None)
    }
}

/// Writes the BC preflight cache and optional benchmark report atomically.
pub fn persist_bc_preflight_runtime(
    cache_key: PreflightCacheKey,
    runtime: EffectiveRuntimeConfig,
    benchmark: Option<hydra_train_runtime::preflight::BenchmarkResult>,
    paths: &PreflightPaths,
    artifacts: &BcArtifactPaths,
) -> Result<(), String> {
    write_preflight_cache(
        &paths.cache_path,
        &PreflightCacheEntry {
            cache_key: cache_key.clone(),
            runtime,
            benchmark: benchmark.clone(),
        },
    )?;
    if let Some(benchmark) = benchmark {
        let benchmark_paths = PreflightBenchmarkPaths::new(artifacts);
        benchmark_paths.create_root_dir()?;
        write_preflight_benchmark_report(
            &benchmark_paths.report_path(),
            &PreflightBenchmarkReport {
                cache_key,
                runtime,
                benchmark,
            },
        )?;
    }
    Ok(())
}

/// Returns a matching RL preflight cache entry.
pub fn matching_rl_preflight_cache(
    config: &TrainConfig,
    model_config: &HydraModelConfig,
    artifacts: &RlArtifactPaths,
) -> Result<Option<PreflightCacheEntry>, String> {
    let paths = RlPreflightPaths::new(artifacts);
    let cache_key = preflight_cache_key(
        config,
        model_config,
        &config.device,
        default_num_threads_for_system(),
    );
    let Some(cached) = read_preflight_cache(&paths.cache_path)? else {
        return Ok(None);
    };
    Ok((cached.cache_key == cache_key).then_some(cached))
}

/// Applies a matching BC preflight selected runtime during epoch-boundary resume.
pub fn apply_cached_bc_runtime_if_matching(
    config: &mut TrainConfig,
    resume: &ResumeContext,
    artifacts: &BcArtifactPaths,
    model_config: &HydraModelConfig,
) -> Result<(), String> {
    let is_epoch_boundary_resume = resume
        .state
        .as_ref()
        .is_some_and(|state| state.skip_optimizer_steps_in_epoch == 0);
    if !is_epoch_boundary_resume {
        return Ok(());
    }

    let context = bc_preflight_cache_context(config, model_config, &config.device, artifacts);
    let Some(cached) = read_preflight_cache(&context.paths.cache_path)? else {
        return Ok(());
    };
    if cached.cache_key != context.cache_key {
        println!(
            "{}",
            timestamped(format!(
                "{} cache fingerprint mismatch, using config train_microbatch_size={:?} validation_microbatch_size={:?} buffer_games={} buffer_samples={} archive_queue_bound={} num_threads={:?}",
                "BC preflight skip:".bold().yellow(),
                config.microbatch_size,
                config.validation_microbatch_size,
                config.buffer_games,
                config.buffer_samples,
                config.archive_queue_bound,
                config.num_threads,
            ))
        );
        return Ok(());
    }

    let tuned_selected = cached.runtime.selected;
    let original_train = config.microbatch_size;
    let original_validation = config.validation_microbatch_size;
    if original_train != Some(tuned_selected.train_microbatch_size)
        || original_validation != Some(tuned_selected.validation_microbatch_size)
    {
        println!(
            "{}",
            timestamped(format!(
                "{} train_microbatch_size={:?} -> {} validation_microbatch_size={:?} -> {} accum_steps={} (epoch-boundary selected-runtime from preflight cache)",
                "BC preflight override:".bold().cyan(),
                original_train,
                tuned_selected.train_microbatch_size,
                original_validation,
                tuned_selected.validation_microbatch_size,
                tuned_selected.accum_steps,
            ))
        );
    }

    config.microbatch_size = Some(tuned_selected.train_microbatch_size);
    config.validation_microbatch_size = Some(tuned_selected.validation_microbatch_size);
    Ok(())
}

/// Persists RL preflight selected runtime in the shared preflight cache schema.
pub fn persist_rl_preflight_runtime(
    config: &TrainConfig,
    artifacts: &RlArtifactPaths,
    runtime: EffectiveRuntimeConfig,
) -> Result<(), String> {
    let paths = RlPreflightPaths::new(artifacts);
    let cache_key = preflight_cache_key(
        config,
        &HydraModelConfig::learner(),
        &config.device,
        default_num_threads_for_system(),
    );
    write_preflight_cache(
        &paths.cache_path,
        &PreflightCacheEntry {
            cache_key,
            runtime,
            benchmark: None,
        },
    )
}

/// Returns true when `path` exists; small test helper for migrated cache-path checks.
pub fn preflight_cache_exists(path: &Path) -> bool {
    path.exists()
}
