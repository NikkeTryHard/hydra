//! Exec-owned streaming MJAI data pipeline for preflight/epoch runners.
//!
//! This module mirrors the train facade streaming semantics without depending on `hydra-train`.
#![allow(
    missing_docs,
    reason = "migrated DTO seam preserves train-bin internal API names"
)]

use std::collections::BTreeMap;
use std::fs;
use std::io::{self, BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;

pub use hydra_data_core::{DataManifest, DataSource, GameLocator, SourceFilterConfig};
use indicatif::ProgressBar;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

use hydra_data_core::MjaiSample;
pub use hydra_replay_loader::ReplayTargetProfile;
use hydra_replay_loader::{
    MjaiGame, ReplayLoadPolicy, SidecarProvenance, load_game_from_path_with_policy,
    load_game_from_stream_with_policy, normalized_train_fraction,
};
use hydra_replay_sidecar::{DeltaQSidecarIndex, ExitSidecarIndex};
use hydra_sample_cache::{
    ParsedSampleCacheMetadata, is_parsed_sample_cache_file, load_parsed_sample_cache,
    read_parsed_sample_cache_metadata,
};
pub use hydra_train_runtime::data::validation_stream::StreamValMicrobatchIterator;

fn compact_identity(identity: &str) -> &str {
    identity.rsplit('/').next().unwrap_or(identity)
}

fn compact_error_message(err: &dyn std::fmt::Display) -> &'static str {
    let message = err.to_string();
    if message.contains("expected value") || message.contains("EOF while parsing") {
        "invalid-json"
    } else if message.contains("no valid decisions") || message.contains("no samples") {
        "no-decision-samples"
    } else if message.contains("unsupported") {
        "unsupported-replay"
    } else if message.contains("sidecar") {
        "sidecar-mismatch"
    } else {
        "parse-error"
    }
}

fn identity_for_archive_entry(archive_path: &Path, entry_path: &Path) -> io::Result<String> {
    let archive_name = archive_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "archive path has no filename")
        })?;
    Ok(format!("{archive_name}/{}", entry_path.display()))
}

fn is_tar_zst_file(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some(name) if name.ends_with(".tar.zst") || name.ends_with(".tzst")
    )
}

fn is_mjai_archive_entry(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some(name)
            if name.ends_with(".json")
                || name.ends_with(".json.gz")
                || name.ends_with(".json.zst")
    )
}

const MJAI_LOAD_THREAD_STACK_SIZE: usize = 8 * 1024 * 1024;
const MJAI_ARCHIVE_QUEUE_BOUND: usize = 128;

/// Configuration for the streaming loader.
#[derive(Debug, Clone)]
pub struct StreamingLoaderConfig {
    pub buffer_games: usize,
    pub buffer_samples: usize,
    pub train_fraction: f32,
    pub seed: u64,
    pub archive_queue_bound: usize,
    pub max_skip_logs_per_source: usize,
    pub aggregate_skip_logs: bool,
    pub source_filters: SourceFilterConfig,
    pub replay_target_profile: ReplayTargetProfile,
    pub exit_sidecar: Option<Arc<ExitSidecarIndex>>,
    pub exit_sidecar_source_net_hash: Option<u64>,
    pub exit_sidecar_source_version: Option<u32>,
    pub delta_q_sidecar: Option<Arc<DeltaQSidecarIndex>>,
    pub delta_q_sidecar_source_net_hash: Option<u64>,
    pub delta_q_sidecar_source_version: Option<u32>,
    pub num_threads: Option<usize>,
}

impl Default for StreamingLoaderConfig {
    fn default() -> Self {
        Self {
            buffer_games: 50_000,
            buffer_samples: 32_768,
            train_fraction: 0.9,
            seed: 0,
            archive_queue_bound: MJAI_ARCHIVE_QUEUE_BOUND,
            max_skip_logs_per_source: 32,
            aggregate_skip_logs: false,
            source_filters: SourceFilterConfig::default(),
            replay_target_profile: ReplayTargetProfile::minimal_bc(),
            exit_sidecar: None,
            exit_sidecar_source_net_hash: None,
            exit_sidecar_source_version: None,
            delta_q_sidecar: None,
            delta_q_sidecar_source_net_hash: None,
            delta_q_sidecar_source_version: None,
            num_threads: None,
        }
    }
}

impl StreamingLoaderConfig {
    fn effective_replay_target_profile(&self) -> ReplayTargetProfile {
        let has_exit = self.exit_sidecar.is_some();
        let has_delta_q = self.delta_q_sidecar.is_some();
        if self.replay_target_profile == ReplayTargetProfile::minimal_bc()
            && (has_exit || has_delta_q)
        {
            ReplayTargetProfile::with_optional_heads(
                false,
                false,
                false,
                false,
                has_exit,
                has_delta_q,
            )
        } else {
            self.replay_target_profile
        }
    }
}

#[derive(Clone, Copy)]
enum StreamSplit {
    Train,
    Validation,
}

enum SourceCursor {
    Archive {
        path: PathBuf,
        rx: mpsc::Receiver<MjaiGame>,
        handle: Option<thread::JoinHandle<io::Result<()>>>,
    },
}

#[derive(Debug, Clone, PartialEq)]
enum PendingSourceOpen {
    Archive(PathBuf),
    LooseBatch(Vec<PathBuf>),
    ParsedSampleCacheBatch(Vec<(PathBuf, String)>),
}

pub struct StreamEpochIterator {
    sources: Vec<DataSource>,
    config: StreamingLoaderConfig,
    split: StreamSplit,
    shuffle_buffers: bool,
    epoch: usize,
    yield_index: usize,
    next_source_index: usize,
    current_source: Option<SourceCursor>,
    progress: Option<ProgressBar>,
}

struct ArchiveEntryJob {
    sequence: usize,
    display_name: String,
    data: Vec<u8>,
}

struct ParsedArchiveGame {
    sequence: usize,
    display_name: String,
    result: io::Result<MjaiGame>,
}

struct SkipLogState {
    source: String,
    emitted: AtomicUsize,
    suppressed: AtomicUsize,
    max_logs: usize,
    aggregate_only: bool,
    reason_counts: std::sync::Mutex<BTreeMap<&'static str, usize>>,
}

struct ProducerLoadContext {
    queue_bound: usize,
    num_threads: Option<usize>,
    replay_target_profile: ReplayTargetProfile,
    exit_sidecar: Option<Arc<ExitSidecarIndex>>,
    exit_sidecar_source_net_hash: Option<u64>,
    exit_sidecar_source_version: Option<u32>,
    delta_q_sidecar: Option<Arc<DeltaQSidecarIndex>>,
    delta_q_sidecar_source_net_hash: Option<u64>,
    delta_q_sidecar_source_version: Option<u32>,
    skip_state: Arc<SkipLogState>,
}

struct LooseBatchWorkerContext {
    replay_target_profile: ReplayTargetProfile,
    exit_sidecar: Option<Arc<ExitSidecarIndex>>,
    exit_provenance: SidecarProvenance,
    delta_q_sidecar: Option<Arc<DeltaQSidecarIndex>>,
    delta_q_provenance: SidecarProvenance,
    skip_state: Arc<SkipLogState>,
}

struct ArchiveParsePolicy {
    replay_target_profile: ReplayTargetProfile,
    exit_sidecar: Option<Arc<ExitSidecarIndex>>,
    exit_provenance: SidecarProvenance,
    delta_q_sidecar: Option<Arc<DeltaQSidecarIndex>>,
    delta_q_provenance: SidecarProvenance,
}

impl LooseBatchWorkerContext {
    fn replay_load_policy(&self) -> ReplayLoadPolicy<'_> {
        ReplayLoadPolicy::new(
            self.replay_target_profile,
            self.exit_provenance,
            self.delta_q_provenance,
            self.exit_sidecar.as_deref(),
            self.delta_q_sidecar.as_deref(),
        )
    }
}

impl ArchiveParsePolicy {
    fn replay_load_policy(&self) -> ReplayLoadPolicy<'_> {
        ReplayLoadPolicy::new(
            self.replay_target_profile,
            self.exit_provenance,
            self.delta_q_provenance,
            self.exit_sidecar.as_deref(),
            self.delta_q_sidecar.as_deref(),
        )
    }
}

impl SkipLogState {
    fn utc_prefix() -> String {
        let ts = OffsetDateTime::now_utc()
            .format(&Rfc3339)
            .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_string());
        format!("[{ts}]")
    }

    fn new(source: String, max_logs: usize, aggregate_only: bool) -> Self {
        Self {
            source,
            emitted: AtomicUsize::new(0),
            suppressed: AtomicUsize::new(0),
            max_logs,
            aggregate_only,
            reason_counts: std::sync::Mutex::new(BTreeMap::new()),
        }
    }

    fn log_skip(&self, identity: &str, err: &dyn std::fmt::Display) {
        let reason = compact_error_message(err);
        if let Ok(mut counts) = self.reason_counts.lock() {
            *counts.entry(reason).or_insert(0) += 1;
        }
        if self.aggregate_only {
            self.suppressed.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let emitted = self.emitted.fetch_add(1, Ordering::Relaxed);
        if emitted < self.max_logs {
            eprintln!("Skipping {}: {}", compact_identity(identity), reason);
        } else {
            self.suppressed.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn flush_summary(&self) {
        if self.aggregate_only
            && let Ok(counts) = self.reason_counts.lock()
            && !counts.is_empty()
        {
            let summary = counts
                .iter()
                .map(|(reason, count)| format!("{reason}={count}"))
                .collect::<Vec<_>>()
                .join(", ");
            eprintln!(
                "{} [preflight:skip] source={} {}",
                Self::utc_prefix(),
                self.source,
                summary
            );
            return;
        }
        let suppressed = self.suppressed.load(Ordering::Relaxed);
        if suppressed > 0 {
            eprintln!(
                "Suppressed {suppressed} more replay skip logs from {}",
                self.source
            );
        }
    }
}

fn build_producer_load_context(
    config: &StreamingLoaderConfig,
    skip_source: String,
) -> ProducerLoadContext {
    ProducerLoadContext {
        queue_bound: config.archive_queue_bound.max(1),
        num_threads: config.num_threads,
        replay_target_profile: config.effective_replay_target_profile(),
        exit_sidecar: config.exit_sidecar.clone(),
        exit_sidecar_source_net_hash: config.exit_sidecar_source_net_hash,
        exit_sidecar_source_version: config.exit_sidecar_source_version,
        delta_q_sidecar: config.delta_q_sidecar.clone(),
        delta_q_sidecar_source_net_hash: config.delta_q_sidecar_source_net_hash,
        delta_q_sidecar_source_version: config.delta_q_sidecar_source_version,
        skip_state: Arc::new(SkipLogState::new(
            skip_source,
            config.max_skip_logs_per_source,
            config.aggregate_skip_logs,
        )),
    }
}

fn load_loose_game_with_policy(
    path: &Path,
    worker: &LooseBatchWorkerContext,
) -> io::Result<MjaiGame> {
    let policy = worker.replay_load_policy();
    load_game_from_path_with_policy(path, Some(&policy))
}

fn load_parsed_sample_cache_game(path: &Path) -> io::Result<MjaiGame> {
    load_parsed_sample_cache(path).map(|cache| MjaiGame {
        samples: cache.game.samples,
        final_scores: cache.game.final_scores,
    })
}

fn forward_loose_game_result(
    path: &Path,
    result: io::Result<MjaiGame>,
    game_tx: &mpsc::SyncSender<MjaiGame>,
    worker: &LooseBatchWorkerContext,
) {
    match result {
        Ok(game) => {
            let _ = game_tx.send(game);
        }
        Err(err) => {
            if let Ok(identity) = identity_for_loose_file(path) {
                worker.skip_state.log_skip(&identity, &err);
            }
        }
    }
}

struct LooseBatchStreamInput {
    paths: Vec<PathBuf>,
    split: StreamSplit,
    train_fraction: f32,
    progress: Option<ProgressBar>,
    queue_bound: usize,
    num_threads: Option<usize>,
    game_tx: mpsc::SyncSender<MjaiGame>,
    worker: LooseBatchWorkerContext,
}

struct ParsedSampleCacheBatchStreamInput {
    entries: Vec<(PathBuf, String)>,
    split: StreamSplit,
    train_fraction: f32,
    progress: Option<ProgressBar>,
    queue_bound: usize,
    num_threads: Option<usize>,
    game_tx: mpsc::SyncSender<MjaiGame>,
    worker: LooseBatchWorkerContext,
}

fn run_loose_batch_stream(input: LooseBatchStreamInput) -> io::Result<()> {
    let LooseBatchStreamInput {
        paths,
        split,
        train_fraction,
        progress,
        queue_bound,
        num_threads,
        game_tx,
        worker,
    } = input;
    let mut pool_builder = ThreadPoolBuilder::new().stack_size(MJAI_LOAD_THREAD_STACK_SIZE);
    if let Some(n) = num_threads {
        pool_builder = pool_builder.num_threads(n);
    }
    let pool = pool_builder.build().map_err(|err| {
        io::Error::other(format!("failed to build loose batch thread pool: {err}"))
    })?;

    let (path_tx, path_rx) = mpsc::sync_channel::<PathBuf>(queue_bound);

    let lister = thread::Builder::new()
        .name("mjai-loose-lister".into())
        .spawn(move || {
            for path in paths {
                if let Ok(identity) = identity_for_loose_file(&path)
                    && should_include_identity(&identity, train_fraction, &split)
                    && path_tx.send(path).is_err()
                {
                    break;
                }
            }
        })
        .map_err(|err| io::Error::other(format!("failed to spawn loose lister thread: {err}")))?;

    let worker_for_pool = LooseBatchWorkerContext {
        replay_target_profile: worker.replay_target_profile,
        exit_sidecar: worker.exit_sidecar.clone(),
        exit_provenance: worker.exit_provenance,
        delta_q_sidecar: worker.delta_q_sidecar.clone(),
        delta_q_provenance: worker.delta_q_provenance,
        skip_state: Arc::clone(&worker.skip_state),
    };

    pool.install(|| {
        path_rx.into_iter().par_bridge().for_each(|path| {
            let result = load_loose_game_with_policy(&path, &worker_for_pool);

            if let Some(pb) = &progress {
                pb.inc(1);
            }

            forward_loose_game_result(&path, result, &game_tx, &worker_for_pool);
        });
    });

    lister
        .join()
        .map_err(|_| io::Error::other("loose lister thread panicked"))?;
    worker.skip_state.flush_summary();
    Ok(())
}

fn run_parsed_sample_cache_batch_stream(
    input: ParsedSampleCacheBatchStreamInput,
) -> io::Result<()> {
    let ParsedSampleCacheBatchStreamInput {
        entries,
        split,
        train_fraction,
        progress,
        queue_bound,
        num_threads,
        game_tx,
        worker,
    } = input;
    let mut pool_builder = ThreadPoolBuilder::new().stack_size(MJAI_LOAD_THREAD_STACK_SIZE);
    if let Some(n) = num_threads {
        pool_builder = pool_builder.num_threads(n);
    }
    let pool = pool_builder.build().map_err(|err| {
        io::Error::other(format!(
            "failed to build parsed-sample cache thread pool: {err}"
        ))
    })?;

    let (entry_tx, entry_rx) = mpsc::sync_channel::<(PathBuf, String)>(queue_bound);

    let lister = thread::Builder::new()
        .name("parsed-sample-cache-lister".into())
        .spawn(move || {
            for (path, original_identity) in entries {
                if should_include_identity(&original_identity, train_fraction, &split)
                    && entry_tx.send((path, original_identity)).is_err()
                {
                    break;
                }
            }
        })
        .map_err(|err| {
            io::Error::other(format!(
                "failed to spawn parsed-sample cache lister thread: {err}"
            ))
        })?;

    let worker_for_pool = LooseBatchWorkerContext {
        replay_target_profile: worker.replay_target_profile,
        exit_sidecar: worker.exit_sidecar.clone(),
        exit_provenance: worker.exit_provenance,
        delta_q_sidecar: worker.delta_q_sidecar.clone(),
        delta_q_provenance: worker.delta_q_provenance,
        skip_state: Arc::clone(&worker.skip_state),
    };

    pool.install(|| {
        entry_rx
            .into_iter()
            .par_bridge()
            .for_each(|(path, original_identity)| {
                let result = load_parsed_sample_cache_game(&path);

                if let Some(pb) = &progress {
                    pb.inc(1);
                }

                match result {
                    Ok(game) => {
                        let _ = game_tx.send(game);
                    }
                    Err(err) => worker_for_pool
                        .skip_state
                        .log_skip(&original_identity, &err),
                }
            });
    });

    lister
        .join()
        .map_err(|_| io::Error::other("parsed-sample cache lister thread panicked"))?;
    worker.skip_state.flush_summary();
    Ok(())
}

fn enqueue_archive_entry_jobs<R: Read>(
    archive: &mut tar::Archive<R>,
    archive_path: &Path,
    should_include_entry: impl Fn(&str) -> bool,
    mut on_include_entry: impl FnMut(),
    mut on_read_error: impl FnMut(&str, &io::Error),
    job_tx: &mpsc::SyncSender<ArchiveEntryJob>,
) -> io::Result<()> {
    let mut sequence = 0usize;
    for entry_result in archive.entries()? {
        let mut entry = entry_result?;
        let entry_path = entry.path()?.into_owned();
        if !is_mjai_archive_entry(&entry_path) {
            continue;
        }

        let identity = identity_for_archive_entry(archive_path, &entry_path)?;
        if !should_include_entry(&identity) {
            continue;
        }

        on_include_entry();

        let mut data = Vec::with_capacity(entry.size() as usize);
        if let Err(err) = std::io::Read::read_to_end(&mut entry, &mut data) {
            on_read_error(&identity, &err);
            continue;
        }

        if job_tx
            .send(ArchiveEntryJob {
                sequence,
                display_name: identity,
                data,
            })
            .is_err()
        {
            break;
        }
        sequence += 1;
    }

    Ok(())
}

fn collect_parsed_archive_games_in_order(
    parsed_rx: mpsc::Receiver<ParsedArchiveGame>,
    ordered_tx: mpsc::SyncSender<MjaiGame>,
    skip_state: Arc<SkipLogState>,
) -> io::Result<()> {
    let mut next_sequence = 0usize;
    let mut pending = BTreeMap::new();
    for parsed in parsed_rx {
        pending.insert(parsed.sequence, parsed);
        while let Some(parsed) = pending.remove(&next_sequence) {
            match parsed.result {
                Ok(game) => ordered_tx.send(game).map_err(|_| {
                    io::Error::new(io::ErrorKind::BrokenPipe, "archive stream receiver dropped")
                })?,
                Err(err) => skip_state.log_skip(&parsed.display_name, &err),
            }
            next_sequence += 1;
        }
    }
    Ok(())
}

fn parse_archive_job(
    job: ArchiveEntryJob,
    policy: Option<&ArchiveParsePolicy>,
) -> (usize, String, io::Result<MjaiGame>) {
    let replay_policy = policy.map(ArchiveParsePolicy::replay_load_policy);
    let result = load_game_from_stream_with_policy(
        &job.display_name,
        BufReader::new(std::io::Cursor::new(job.data)),
        replay_policy.as_ref(),
    );
    (job.sequence, job.display_name, result)
}

fn next_seed(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    *seed
}

fn fnv1a_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

pub(crate) fn identity_for_loose_file(path: &Path) -> io::Result<String> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "loose file path does not have a recognizable filename: {}",
                    path.display()
                ),
            )
        })?;
    if let Some(parent) = path
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
    {
        Ok(format!("{parent}/{file_name}"))
    } else {
        Ok(file_name.to_owned())
    }
}

fn shuffle_sources(sources: &mut [DataSource], seed: u64) {
    let mut state = seed;
    for idx in (1..sources.len()).rev() {
        let swap_idx = (next_seed(&mut state) % (idx as u64 + 1)) as usize;
        sources.swap(idx, swap_idx);
    }
}

fn shuffle_owned_samples(samples: &mut [MjaiSample], seed: u64) {
    let mut state = seed;
    for idx in (1..samples.len()).rev() {
        let swap_idx = (next_seed(&mut state) % (idx as u64 + 1)) as usize;
        samples.swap(idx, swap_idx);
    }
}

fn should_include_identity(identity: &str, train_fraction: f32, split: &StreamSplit) -> bool {
    match split {
        StreamSplit::Train => is_train_game(identity, train_fraction),
        StreamSplit::Validation => !is_train_game(identity, train_fraction),
    }
}

fn source_matches_filters(source: &DataSource, filters: &SourceFilterConfig) -> bool {
    if filters.is_empty() {
        return true;
    }
    let path = match source {
        DataSource::ParsedSampleCache {
            original_source_path,
            ..
        } => original_source_path.to_string_lossy(),
        _ => source.path().to_string_lossy(),
    };
    let included = filters.include_source_patterns.is_empty()
        || filters
            .include_source_patterns
            .iter()
            .any(|pattern| path.contains(pattern));
    included
        && !filters
            .exclude_source_patterns
            .iter()
            .any(|pattern| path.contains(pattern))
}

fn stream_shuffle_seed(config: &StreamingLoaderConfig, epoch: usize, yield_index: usize) -> u64 {
    config
        .seed
        .wrapping_add(epoch as u64)
        .wrapping_mul(1_000_003)
        .wrapping_add(yield_index as u64)
}

fn scan_data_sources_with_fraction(
    data_dir: &Path,
    train_fraction: f32,
    source_filters: &SourceFilterConfig,
    progress: Option<&ProgressBar>,
) -> io::Result<DataManifest> {
    let sources = if data_dir.is_file() {
        if is_tar_zst_file(data_dir) || is_tar_file(data_dir) {
            vec![DataSource::Archive(data_dir.to_path_buf())]
        } else if is_mjai_file(data_dir) {
            vec![DataSource::LooseFile(data_dir.to_path_buf())]
        } else if is_parsed_sample_cache_file(data_dir) {
            vec![data_source_for_cache_path(data_dir)?]
        } else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "expected directory, MJAI file, parsed-sample cache file, or .tar/.tar.zst archive, got {}",
                    data_dir.display()
                ),
            ));
        }
    } else {
        scan_directory_sources(data_dir)?
    };
    let sources: Vec<DataSource> = sources
        .into_iter()
        .filter(|source| source_matches_filters(source, source_filters))
        .collect();

    let mut total_games = 0usize;
    let mut train_count = 0usize;
    let mut counts_exact = true;
    for source in &sources {
        match source {
            DataSource::LooseFile(path) => {
                total_games += 1;
                let identity = identity_for_loose_file(path)?;
                if is_train_game(&identity, train_fraction) {
                    train_count += 1;
                }
            }
            DataSource::ParsedSampleCache {
                original_identity, ..
            } => {
                total_games += 1;
                if is_train_game(original_identity, train_fraction) {
                    train_count += 1;
                }
            }
            DataSource::Archive(_) => {
                counts_exact = false;
            }
        }
        if let Some(pb) = progress {
            pb.inc(1);
        }
    }

    Ok(DataManifest {
        sources,
        total_games,
        train_count,
        val_count: total_games.saturating_sub(train_count),
        counts_exact,
    })
}

fn scan_directory_sources(dir: &Path) -> io::Result<Vec<DataSource>> {
    let mut sources = Vec::new();
    scan_directory_sources_recursive(dir, &mut sources)?;
    sources.sort_by(|a, b| a.path().cmp(b.path()));
    Ok(sources)
}

fn scan_directory_sources_recursive(dir: &Path, sources: &mut Vec<DataSource>) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let path = entry.path();
        if file_type.is_dir() {
            scan_directory_sources_recursive(&path, sources)?;
        } else if file_type.is_file() {
            if is_mjai_file(&path) {
                sources.push(DataSource::LooseFile(path));
            } else if is_parsed_sample_cache_file(&path) {
                sources.push(data_source_for_cache_path(&path)?);
            } else if is_tar_zst_file(&path) || is_tar_file(&path) {
                sources.push(DataSource::Archive(path));
            }
        }
    }
    Ok(())
}

fn data_source_for_cache_path(path: &Path) -> io::Result<DataSource> {
    let ParsedSampleCacheMetadata {
        original_identity,
        original_source_path,
        ..
    } = read_parsed_sample_cache_metadata(path)?;
    Ok(DataSource::ParsedSampleCache {
        path: path.to_path_buf(),
        original_identity,
        original_source_path,
    })
}

fn spawn_archive_stream(
    path: PathBuf,
    split: StreamSplit,
    train_fraction: f32,
    progress: Option<ProgressBar>,
    config: &StreamingLoaderConfig,
) -> io::Result<SourceCursor> {
    let producer = build_producer_load_context(config, path.display().to_string());
    let ProducerLoadContext {
        queue_bound: archive_queue_bound,
        num_threads,
        replay_target_profile,
        exit_sidecar,
        exit_sidecar_source_net_hash,
        exit_sidecar_source_version,
        delta_q_sidecar,
        delta_q_sidecar_source_net_hash,
        delta_q_sidecar_source_version,
        skip_state,
    } = producer;
    let parse_policy = if exit_sidecar.is_some() || delta_q_sidecar.is_some() {
        Some(ArchiveParsePolicy {
            replay_target_profile,
            exit_sidecar,
            exit_provenance: SidecarProvenance::new(
                exit_sidecar_source_net_hash,
                exit_sidecar_source_version,
            ),
            delta_q_sidecar,
            delta_q_provenance: SidecarProvenance::new(
                delta_q_sidecar_source_net_hash,
                delta_q_sidecar_source_version,
            ),
        })
    } else {
        None
    };
    let (tx, rx) = mpsc::sync_channel::<MjaiGame>(archive_queue_bound);
    let path_for_thread = path.clone();
    let handle = thread::Builder::new()
        .name(format!("mjai-stream-{}", path.display()))
        .stack_size(MJAI_LOAD_THREAD_STACK_SIZE)
        .spawn(move || -> io::Result<()> {
            let mut pool_builder = ThreadPoolBuilder::new().stack_size(MJAI_LOAD_THREAD_STACK_SIZE);
            if let Some(n) = num_threads {
                pool_builder = pool_builder.num_threads(n);
            }
            let pool = pool_builder.build().map_err(|err| {
                io::Error::other(format!(
                    "failed to build MJAI archive stream thread pool: {err}"
                ))
            })?;
            let file = fs::File::open(&path_for_thread)?;
            let reader: Box<dyn Read + Send> = if is_tar_zst_file(&path_for_thread) {
                let zstd = zstd::Decoder::new(file).map_err(|err| {
                    io::Error::other(format!(
                        "failed to open zstd archive {}: {err}",
                        path_for_thread.display()
                    ))
                })?;
                Box::new(zstd)
            } else {
                Box::new(file)
            };
            let mut archive = tar::Archive::new(reader);
            let (job_tx, job_rx) = mpsc::sync_channel::<ArchiveEntryJob>(archive_queue_bound);
            let (parsed_tx, parsed_rx) =
                mpsc::sync_channel::<ParsedArchiveGame>(archive_queue_bound);

            let parsed_tx_for_parse = parsed_tx.clone();
            let parser = thread::Builder::new()
                .name(format!("mjai-archive-parse-{}", path_for_thread.display()))
                .spawn(move || -> io::Result<()> {
                    pool.install(|| {
                        job_rx.into_iter().par_bridge().try_for_each(|job| {
                            let (sequence, display_name, result) =
                                parse_archive_job(job, parse_policy.as_ref());
                            parsed_tx_for_parse
                                .send(ParsedArchiveGame {
                                    sequence,
                                    display_name,
                                    result,
                                })
                                .map_err(|_| {
                                    io::Error::new(
                                        io::ErrorKind::BrokenPipe,
                                        "archive stream receiver dropped",
                                    )
                                })
                        })
                    })
                })
                .map_err(|err| {
                    io::Error::other(format!(
                        "failed to spawn archive parse thread {}: {err}",
                        path_for_thread.display()
                    ))
                })?;
            drop(parsed_tx);

            let skip_state_for_collect = Arc::clone(&skip_state);
            let ordered_tx = tx.clone();
            let collector = thread::Builder::new()
                .name(format!("mjai-archive-order-{}", path_for_thread.display()))
                .spawn(move || {
                    collect_parsed_archive_games_in_order(
                        parsed_rx,
                        ordered_tx,
                        skip_state_for_collect,
                    )
                })
                .map_err(|err| {
                    io::Error::other(format!(
                        "failed to spawn archive ordering thread {}: {err}",
                        path_for_thread.display()
                    ))
                })?;

            enqueue_archive_entry_jobs(
                &mut archive,
                &path_for_thread,
                |identity| should_include_identity(identity, train_fraction, &split),
                || {
                    if let Some(pb) = &progress {
                        pb.inc(1);
                    }
                },
                |identity, err| skip_state.log_skip(identity, err),
                &job_tx,
            )?;

            drop(job_tx);
            parser.join().map_err(|_| {
                io::Error::other(format!(
                    "archive parse thread panicked for {}",
                    path_for_thread.display()
                ))
            })??;
            collector.join().map_err(|_| {
                io::Error::other(format!(
                    "archive ordering thread panicked for {}",
                    path_for_thread.display()
                ))
            })??;
            skip_state.flush_summary();

            Ok(())
        })
        .map_err(|err| io::Error::other(format!("failed to spawn archive stream: {err}")))?;

    Ok(SourceCursor::Archive {
        path,
        rx,
        handle: Some(handle),
    })
}

fn spawn_loose_batch_stream(
    paths: Vec<PathBuf>,
    split: StreamSplit,
    train_fraction: f32,
    progress: Option<ProgressBar>,
    config: &StreamingLoaderConfig,
) -> io::Result<SourceCursor> {
    let producer = build_producer_load_context(config, "loose-batch".to_string());
    let ProducerLoadContext {
        queue_bound,
        num_threads,
        replay_target_profile,
        exit_sidecar,
        exit_sidecar_source_net_hash,
        exit_sidecar_source_version,
        delta_q_sidecar,
        delta_q_sidecar_source_net_hash,
        delta_q_sidecar_source_version,
        skip_state,
    } = producer;
    let (game_tx, game_rx) = mpsc::sync_channel::<MjaiGame>(queue_bound);
    let worker = LooseBatchWorkerContext {
        replay_target_profile,
        exit_sidecar,
        exit_provenance: SidecarProvenance::new(
            exit_sidecar_source_net_hash,
            exit_sidecar_source_version,
        ),
        delta_q_sidecar,
        delta_q_provenance: SidecarProvenance::new(
            delta_q_sidecar_source_net_hash,
            delta_q_sidecar_source_version,
        ),
        skip_state,
    };

    let handle = thread::Builder::new()
        .name("mjai-loose-batch".into())
        .stack_size(MJAI_LOAD_THREAD_STACK_SIZE)
        .spawn(move || {
            run_loose_batch_stream(LooseBatchStreamInput {
                paths,
                split,
                train_fraction,
                progress,
                queue_bound,
                num_threads,
                game_tx,
                worker,
            })
        })
        .map_err(|err| io::Error::other(format!("failed to spawn loose batch stream: {err}")))?;

    Ok(SourceCursor::Archive {
        path: PathBuf::from("<loose-batch>"),
        rx: game_rx,
        handle: Some(handle),
    })
}

fn spawn_parsed_sample_cache_batch_stream(
    entries: Vec<(PathBuf, String)>,
    split: StreamSplit,
    train_fraction: f32,
    progress: Option<ProgressBar>,
    config: &StreamingLoaderConfig,
) -> io::Result<SourceCursor> {
    let producer = build_producer_load_context(config, "parsed-sample-cache-batch".to_string());
    let ProducerLoadContext {
        queue_bound,
        num_threads,
        replay_target_profile,
        exit_sidecar,
        exit_sidecar_source_net_hash,
        exit_sidecar_source_version,
        delta_q_sidecar,
        delta_q_sidecar_source_net_hash,
        delta_q_sidecar_source_version,
        skip_state,
    } = producer;
    let (game_tx, game_rx) = mpsc::sync_channel::<MjaiGame>(queue_bound);
    let worker = LooseBatchWorkerContext {
        replay_target_profile,
        exit_sidecar,
        exit_provenance: SidecarProvenance::new(
            exit_sidecar_source_net_hash,
            exit_sidecar_source_version,
        ),
        delta_q_sidecar,
        delta_q_provenance: SidecarProvenance::new(
            delta_q_sidecar_source_net_hash,
            delta_q_sidecar_source_version,
        ),
        skip_state,
    };

    let handle = thread::Builder::new()
        .name("parsed-sample-cache-batch".into())
        .stack_size(MJAI_LOAD_THREAD_STACK_SIZE)
        .spawn(move || {
            run_parsed_sample_cache_batch_stream(ParsedSampleCacheBatchStreamInput {
                entries,
                split,
                train_fraction,
                progress,
                queue_bound,
                num_threads,
                game_tx,
                worker,
            })
        })
        .map_err(|err| {
            io::Error::other(format!(
                "failed to spawn parsed-sample cache batch stream: {err}"
            ))
        })?;

    Ok(SourceCursor::Archive {
        path: PathBuf::from("<parsed-sample-cache-batch>"),
        rx: game_rx,
        handle: Some(handle),
    })
}

impl StreamEpochIterator {
    fn new(
        manifest: &DataManifest,
        config: &StreamingLoaderConfig,
        split: StreamSplit,
        epoch: usize,
        progress: Option<&ProgressBar>,
        shuffle_buffers: bool,
    ) -> Self {
        let mut sources = manifest.sources.clone();
        if matches!(split, StreamSplit::Train) {
            shuffle_sources(&mut sources, config.seed.wrapping_add(epoch as u64));
        }
        Self {
            sources,
            config: config.clone(),
            split,
            shuffle_buffers,
            epoch,
            yield_index: 0,
            next_source_index: 0,
            current_source: None,
            progress: progress.cloned(),
        }
    }

    fn buffer_limit(&self) -> usize {
        self.config.buffer_games.max(1)
    }

    fn sample_limit(&self) -> usize {
        self.config.buffer_samples.max(1)
    }

    fn take_next_source_plan(&mut self) -> Option<PendingSourceOpen> {
        if self.next_source_index >= self.sources.len() {
            return None;
        }

        let source = self.sources[self.next_source_index].clone();
        self.next_source_index += 1;
        Some(match source {
            DataSource::Archive(path) => PendingSourceOpen::Archive(path),
            DataSource::LooseFile(first_path) => {
                let mut paths = vec![first_path];
                while self.next_source_index < self.sources.len() {
                    if let DataSource::LooseFile(path) = &self.sources[self.next_source_index] {
                        paths.push(path.clone());
                        self.next_source_index += 1;
                    } else {
                        break;
                    }
                }
                PendingSourceOpen::LooseBatch(paths)
            }
            DataSource::ParsedSampleCache {
                path,
                original_identity,
                ..
            } => {
                let mut entries = vec![(path, original_identity)];
                while self.next_source_index < self.sources.len() {
                    if let DataSource::ParsedSampleCache {
                        path,
                        original_identity,
                        ..
                    } = &self.sources[self.next_source_index]
                    {
                        entries.push((path.clone(), original_identity.clone()));
                        self.next_source_index += 1;
                    } else {
                        break;
                    }
                }
                PendingSourceOpen::ParsedSampleCacheBatch(entries)
            }
        })
    }

    fn open_next_source(&mut self) -> io::Result<()> {
        if self.current_source.is_some() || self.next_source_index >= self.sources.len() {
            return Ok(());
        }

        self.current_source = Some(
            match self
                .take_next_source_plan()
                .expect("source plan should exist when next_source_index is in range")
            {
                PendingSourceOpen::Archive(path) => spawn_archive_stream(
                    path,
                    self.split,
                    self.config.train_fraction,
                    self.progress.clone(),
                    &self.config,
                )?,
                PendingSourceOpen::LooseBatch(paths) => spawn_loose_batch_stream(
                    paths,
                    self.split,
                    self.config.train_fraction,
                    self.progress.clone(),
                    &self.config,
                )?,
                PendingSourceOpen::ParsedSampleCacheBatch(entries) => {
                    spawn_parsed_sample_cache_batch_stream(
                        entries,
                        self.split,
                        self.config.train_fraction,
                        self.progress.clone(),
                        &self.config,
                    )?
                }
            },
        );
        Ok(())
    }

    fn take_next_game(&mut self) -> io::Result<Option<MjaiGame>> {
        loop {
            self.open_next_source()?;

            let Some(source) = self.current_source.take() else {
                return Ok(None);
            };

            match source {
                SourceCursor::Archive {
                    path,
                    rx,
                    mut handle,
                } => match rx.recv() {
                    Ok(game) => {
                        self.current_source = Some(SourceCursor::Archive { path, rx, handle });
                        return Ok(Some(game));
                    }
                    Err(_) => {
                        if let Some(handle) = handle.take() {
                            handle.join().map_err(|_| {
                                io::Error::other(format!(
                                    "archive stream thread panicked for {}",
                                    path.display()
                                ))
                            })??;
                        }
                    }
                },
            }
        }
    }
}

impl Iterator for StreamEpochIterator {
    type Item = io::Result<Vec<MjaiSample>>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut games = Vec::new();
        let mut sample_count = 0usize;
        while games.len() < self.buffer_limit() && sample_count < self.sample_limit() {
            match self.take_next_game() {
                Ok(Some(game)) => {
                    sample_count += game.num_samples();
                    games.push(game);
                }
                Ok(None) => break,
                Err(err) => return Some(Err(err)),
            }
        }

        if games.is_empty() {
            return None;
        }

        let sample_capacity = games.iter().map(MjaiGame::num_samples).sum();
        let mut samples = Vec::with_capacity(sample_capacity);
        for game in games {
            samples.extend(game.samples);
        }

        if self.shuffle_buffers {
            let seed = stream_shuffle_seed(&self.config, self.epoch, self.yield_index);
            shuffle_owned_samples(&mut samples, seed);
        }
        self.yield_index += 1;
        Some(Ok(samples))
    }
}

/// Deterministic train/val assignment by hashing game identity.
pub fn is_train_game(identity: &str, train_fraction: f32) -> bool {
    let threshold = (normalized_train_fraction(train_fraction) * 1000.0).round() as u64;
    fnv1a_hash(identity.as_bytes()) % 1000 < threshold
}

/// Scan data_dir and return all GameLocators without loading any data.
pub fn scan_data_sources(data_dir: &Path) -> io::Result<DataManifest> {
    let default_config = StreamingLoaderConfig::default();
    scan_data_sources_with_fraction(
        data_dir,
        default_config.train_fraction,
        &default_config.source_filters,
        None,
    )
}

pub fn scan_data_sources_with_progress(
    data_dir: &Path,
    train_fraction: f32,
    source_filters: &SourceFilterConfig,
    progress: Option<&ProgressBar>,
) -> io::Result<DataManifest> {
    scan_data_sources_with_fraction(data_dir, train_fraction, source_filters, progress)
}

/// Stream training samples from the dataset, one buffer-full at a time.
pub fn stream_train_epoch(
    manifest: &DataManifest,
    config: &StreamingLoaderConfig,
    epoch: usize,
    progress: Option<&ProgressBar>,
) -> impl Iterator<Item = io::Result<Vec<MjaiSample>>> {
    StreamEpochIterator::new(manifest, config, StreamSplit::Train, epoch, progress, true)
}

/// Stream validation samples from the dataset, one buffer-full at a time.
pub fn stream_val_pass(
    manifest: &DataManifest,
    config: &StreamingLoaderConfig,
    progress: Option<&ProgressBar>,
) -> impl Iterator<Item = io::Result<Vec<MjaiSample>>> {
    StreamEpochIterator::new(
        manifest,
        config,
        StreamSplit::Validation,
        0,
        progress,
        false,
    )
}

pub fn stream_val_microbatches(
    manifest: &DataManifest,
    config: &StreamingLoaderConfig,
    microbatch_size: usize,
    progress: Option<&ProgressBar>,
) -> impl Iterator<Item = io::Result<Vec<MjaiSample>>> {
    StreamValMicrobatchIterator::new(stream_val_pass(manifest, config, progress), microbatch_size)
}

fn is_mjai_file(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some(name)
            if name.ends_with(".json")
                || name.ends_with(".json.gz")
                || name.ends_with(".json.zst")
    )
}

fn is_tar_file(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some(name) if name.ends_with(".tar")
    )
}
