use std::collections::BTreeMap;
use std::fs;
use std::io::{self, BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;

use burn::prelude::*;
pub use hydra_data_core::{DataManifest, DataSource, GameLocator, SourceFilterConfig};
use indicatif::ProgressBar;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

use crate::data::archive_helpers::{
    compact_error_message, compact_identity, identity_for_archive_entry, is_mjai_archive_entry,
    is_tar_zst_file,
};
use crate::data::mjai_loader::{
    MjaiDataset, MjaiGame, ReplayLoadPolicy, ReplayTargetProfile, SidecarProvenance,
    load_game_from_path, load_game_from_path_with_policy, load_game_from_stream_with_policy,
    normalized_train_fraction,
};
use crate::data::parsed_sample_cache::{
    ParsedSampleCacheMetadata, is_parsed_sample_cache_file, load_parsed_sample_cache,
    read_parsed_sample_cache_metadata,
};
use crate::data::sample::{MjaiSample, collate_sample_refs};

type CollatedTrainBatch<B> = Vec<(Tensor<B, 3>, HydraTargets<B>)>;
use crate::training::losses::HydraTargets;
use crate::training::replay_delta_q::DeltaQSidecarIndex;
use crate::training::replay_exit::ExitSidecarIndex;

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
    load_parsed_sample_cache(path).map(|cache| cache.game)
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

pub struct StreamValMicrobatchIterator<I> {
    inner: I,
    microbatch_size: usize,
    current: Vec<MjaiSample>,
    pending: Vec<MjaiSample>,
    pending_start: usize,
    exhausted: bool,
}

impl<I> StreamValMicrobatchIterator<I>
where
    I: Iterator<Item = io::Result<Vec<MjaiSample>>>,
{
    fn new(inner: I, microbatch_size: usize) -> Self {
        let microbatch_size = microbatch_size.max(1);
        Self {
            inner,
            microbatch_size,
            current: Vec::with_capacity(microbatch_size),
            pending: Vec::new(),
            pending_start: 0,
            exhausted: false,
        }
    }

    fn take_full_batch(&mut self) -> Option<Vec<MjaiSample>> {
        let pending_len = self.pending.len().saturating_sub(self.pending_start);
        if pending_len < self.microbatch_size {
            return None;
        }
        if self.pending_start > 0
            && (self.pending_start >= self.microbatch_size
                || self.pending_start * 2 >= self.pending.len())
        {
            self.pending.drain(..self.pending_start);
            self.pending_start = 0;
        }
        self.current.clear();
        self.current.extend(
            self.pending
                .drain(self.pending_start..self.pending_start + self.microbatch_size),
        );
        Some(std::mem::take(&mut self.current))
    }
}

impl<I> Iterator for StreamValMicrobatchIterator<I>
where
    I: Iterator<Item = io::Result<Vec<MjaiSample>>>,
{
    type Item = io::Result<Vec<MjaiSample>>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(batch) = self.take_full_batch() {
                return Some(Ok(batch));
            }

            if self.exhausted {
                if self.pending_start >= self.pending.len() {
                    return None;
                }
                self.current.clear();
                self.current
                    .extend(self.pending.drain(self.pending_start..));
                self.pending.clear();
                self.pending_start = 0;
                return Some(Ok(std::mem::take(&mut self.current)));
            }

            match self.inner.next() {
                Some(Ok(samples)) => {
                    if !samples.is_empty() {
                        self.pending.extend(samples);
                    }
                }
                Some(Err(err)) => return Some(Err(err)),
                None => self.exhausted = true,
            }
        }
    }
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

fn load_mjai_archive(
    path: &Path,
    train_fraction: f32,
    num_threads: Option<usize>,
) -> io::Result<MjaiDataset> {
    let mut pool_builder = ThreadPoolBuilder::new().stack_size(MJAI_LOAD_THREAD_STACK_SIZE);
    if let Some(n) = num_threads {
        pool_builder = pool_builder.num_threads(n);
    }
    let pool = pool_builder.build().map_err(|err| {
        io::Error::other(format!(
            "failed to build MJAI archive loader thread pool: {err}"
        ))
    })?;

    let path_buf = path.to_path_buf();
    let (job_tx, job_rx) = mpsc::sync_channel::<ArchiveEntryJob>(MJAI_ARCHIVE_QUEUE_BOUND);

    let producer = thread::Builder::new()
        .name("mjai-archive-reader".to_string())
        .stack_size(MJAI_LOAD_THREAD_STACK_SIZE)
        .spawn(move || -> io::Result<()> {
            let file = fs::File::open(&path_buf)?;
            let reader: Box<dyn Read> = if is_tar_zst_file(&path_buf) {
                let zstd = zstd::Decoder::new(file).map_err(|err| {
                    io::Error::other(format!(
                        "failed to open zstd archive {}: {err}",
                        path_buf.display()
                    ))
                })?;
                Box::new(zstd)
            } else {
                Box::new(file)
            };
            let mut archive = tar::Archive::new(reader);

            enqueue_archive_entry_jobs(
                &mut archive,
                &path_buf,
                |_| true,
                || {},
                |_, _| {},
                &job_tx,
            )?;

            Ok(())
        })
        .map_err(|err| io::Error::other(format!("failed to spawn archive reader: {err}")))?;

    let mut results: Vec<(usize, String, io::Result<MjaiGame>)> = pool.install(|| {
        job_rx
            .into_iter()
            .par_bridge()
            .map(|job| parse_archive_job(job, None))
            .collect()
    });

    producer.join().map_err(|_| {
        io::Error::other(format!(
            "archive reader thread panicked for {}",
            path.display()
        ))
    })??;

    let mut dataset = MjaiDataset::new(train_fraction);
    let mut skipped = 0usize;

    results.sort_by_key(|(sequence, _, _)| *sequence);

    for (_, display_name, result) in results {
        match result {
            Ok(game) => dataset.add_game(game),
            Err(err) => {
                eprintln!(
                    "Skipping {}: {}",
                    compact_identity(&display_name),
                    compact_error_message(&err)
                );
                skipped += 1;
            }
        }
    }

    println!(
        "Loaded {} MJAI games ({} samples, {} skipped) from archive {}",
        dataset.num_games(),
        dataset.num_samples(),
        skipped,
        path.display()
    );

    Ok(dataset)
}

pub fn load_mjai_directory(dir: &Path, train_fraction: f32) -> io::Result<MjaiDataset> {
    if dir.is_file() {
        if is_tar_zst_file(dir) || is_tar_file(dir) {
            return load_mjai_archive(dir, train_fraction, None);
        }
        if is_parsed_sample_cache_file(dir) {
            let cache = load_parsed_sample_cache(dir)?;
            let mut dataset = MjaiDataset::new(train_fraction);
            dataset.add_game(cache.game);
            return Ok(dataset);
        }
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "expected directory, parsed-sample cache file, or .tar/.tar.zst archive, got {}",
                dir.display()
            ),
        ));
    }

    let mut paths = Vec::new();
    let mut cache_paths = Vec::new();
    let mut archives = Vec::new();
    for source in scan_directory_sources(dir)? {
        match source {
            DataSource::LooseFile(path) => paths.push(path),
            DataSource::ParsedSampleCache { path, .. } => cache_paths.push(path),
            DataSource::Archive(path) => archives.push(path),
        }
    }

    let mut dataset = MjaiDataset::new(train_fraction);
    dataset.games.reserve(paths.len());
    let pool = ThreadPoolBuilder::new()
        .stack_size(MJAI_LOAD_THREAD_STACK_SIZE)
        .build()
        .map_err(|err| {
            io::Error::other(format!("failed to build MJAI loader thread pool: {err}"))
        })?;
    let results: Vec<_> = pool.install(|| {
        paths
            .par_iter()
            .map(|path| (path.clone(), load_game_from_path(path)))
            .collect()
    });

    let mut skipped = 0usize;
    for (path, result) in results {
        match result {
            Ok(game) => dataset.add_game(game),
            Err(err) => {
                eprintln!(
                    "Skipping {}: {}",
                    compact_identity(&path.display().to_string()),
                    compact_error_message(&err)
                );
                skipped += 1;
            }
        }
    }

    for cache_path in cache_paths {
        let cache = load_parsed_sample_cache(&cache_path)?;
        dataset.add_game(cache.game);
    }

    for archive in archives {
        let archive_dataset = load_mjai_archive(&archive, train_fraction, None)?;
        for game in archive_dataset.games {
            dataset.add_game(game);
        }
    }

    println!(
        "Loaded {} MJAI games ({} samples, {} skipped) from {}",
        dataset.num_games(),
        dataset.num_samples(),
        skipped,
        dir.display()
    );

    Ok(dataset)
}

pub fn collect_samples(dataset: &MjaiDataset) -> Vec<&MjaiSample> {
    dataset
        .games
        .iter()
        .flat_map(|game| game.samples.iter())
        .collect()
}

pub fn shuffle_samples(samples: &mut [&MjaiSample], seed: u64) {
    let mut state = seed;
    for idx in (1..samples.len()).rev() {
        let swap_idx = (next_seed(&mut state) % (idx as u64 + 1)) as usize;
        samples.swap(idx, swap_idx);
    }
}

pub fn build_batches<B: Backend>(
    samples: &[&MjaiSample],
    batch_size: usize,
    augment: bool,
    device: &B::Device,
) -> Result<CollatedTrainBatch<B>, String> {
    if samples.is_empty() || batch_size == 0 {
        return Ok(Vec::new());
    }

    samples
        .chunks(batch_size)
        .try_fold(Vec::new(), |mut batches, chunk| {
            if let Some(batch) = collate_sample_refs::<B>(chunk, augment, device)? {
                batches.push(batch);
            }
            Ok(batches)
        })
}

pub fn collate_sample_chunk<B: Backend>(
    samples: &[&MjaiSample],
    augment: bool,
    device: &B::Device,
) -> crate::data::sample::CollatedHydraBatch<B> {
    collate_sample_refs::<B>(samples, augment, device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use flate2::Compression;
    use flate2::write::GzEncoder;
    use hydra_core::action::HYDRA_ACTION_SPACE;
    use hydra_core::encoder::OBS_SIZE;
    use riichienv_core::replay::read_mjai_events;
    use std::fs::File;
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};
    use tar::Builder;
    use zstd::stream::Encoder as ZstdEncoder;

    use crate::data::mjai_loader::{MjaiDataset, MjaiGame, prepare_replay_decision, update_safety};
    use crate::training::replay_delta_q::{DeltaQSidecarIndex, ReplayDeltaQRecordV1};
    use crate::training::replay_exit::{
        ExitSidecarIndex, ReplayDecisionKey, ReplayExitRecordV1, legal_mask_digest_from_f32,
    };
    use crate::training::{live_exit, replay_delta_q, replay_exit};
    use hydra_core::encoder::ObservationEncoder;
    use hydra_core::safety::SafetyInfo;
    use riichienv_core::rule::GameRule;
    use riichienv_core::state::GameState;
    use std::os::unix::ffi::OsStringExt;

    type B = NdArray<f32>;

    fn dummy_sample(action: u8) -> MjaiSample {
        let mut legal_mask = [0.0f32; HYDRA_ACTION_SPACE];
        legal_mask[action as usize] = 1.0;

        MjaiSample {
            obs: [0.25; OBS_SIZE],
            action,
            legal_mask,
            placement: 0,
            score_delta: 0,
            grp_label: 0,
            oracle_target: None,
            tenpai: [0.0; 3],
            opp_next: [255; 3],
            danger: [0.0; 102],
            danger_mask: [1.0; 102],
            safety_residual: None,
            safety_residual_mask: None,
            exit_target: None,
            exit_mask: None,
            delta_q_target: None,
            delta_q_mask: None,
            belief_fields: None,
            mixture_weights: None,
            belief_fields_present: false,
            mixture_weights_present: false,
        }
    }

    fn dataset_with_samples(num_samples: usize) -> MjaiDataset {
        let mut dataset = MjaiDataset::new(0.9);
        dataset.add_game(MjaiGame {
            samples: (0..num_samples)
                .map(|idx| dummy_sample((idx % HYDRA_ACTION_SPACE) as u8))
                .collect(),
            final_scores: [25_000; 4],
        });
        dataset
    }

    fn valid_game_json() -> String {
        [
            r#"{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"dora_marker":"1m","tehais":[["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"],["1s","2s","3s","4s","5s","6s","7s","8s","9s","E","S","W","N"],["P","F","C","1m","1m","2m","2m","3m","3m","4m","4m","5m","5m"],["6p","6p","7p","7p","8p","8p","9p","9p","1s","1s","2s","2s","3s"]]}"#,
            r#"{"type":"end_kyoku"}"#,
        ]
        .join("\n")
    }

    fn replay_sidecar_guardrail_log() -> String {
        [
            r#"{"type":"start_game","names":["a","b","c","d"],"id":"game-1"}"#,
            r#"{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"dora_marker":"1m","tehais":[["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"],["1s","2s","3s","4s","5s","6s","7s","8s","9s","E","S","W","N"],["P","F","C","1m","1m","2m","2m","3m","3m","4m","4m","5m","5m"],["6p","6p","7p","7p","8p","8p","9p","9p","1s","1s","2s","2s","3s"]]}"#,
            r#"{"type":"dahai","actor":0,"pai":"4p","tsumogiri":false}"#,
            r#"{"type":"tsumo","actor":1,"pai":"P"}"#,
            r#"{"type":"dahai","actor":1,"pai":"P","tsumogiri":true}"#,
            r#"{"type":"ryukyoku"}"#,
            r#"{"type":"end_kyoku"}"#,
        ]
        .join("\n")
    }

    fn replay_guardrail_decisions_for_identity(
        identity: &str,
    ) -> Vec<(ReplayDecisionKey, u8, [f32; HYDRA_ACTION_SPACE])> {
        let events = read_mjai_events(std::io::Cursor::new(replay_sidecar_guardrail_log()))
            .expect("parse events");
        let mut state = GameState::new(0, true, Some(0), 0, GameRule::default_tenhou());
        let mut safety = std::array::from_fn(|_| SafetyInfo::default());
        let mut encoder = ObservationEncoder::new();
        let mut decisions = Vec::new();

        for (idx, event) in events.iter().enumerate() {
            if let Some(decision) =
                prepare_replay_decision(event, &mut state, &safety, &mut encoder)
                    .expect("prepare replay decision")
            {
                decisions.push((
                    ReplayDecisionKey {
                        source_hash: replay_exit::source_hash_from_identity(identity),
                        event_index: idx as u32,
                        actor: decision.actor as u8,
                        obs_hash: live_exit::obs_hash(&decision.obs_encoded),
                    },
                    decision.action_id,
                    decision.legal_mask_f32,
                ));
            }
            update_safety(&mut safety, event).expect("update safety");
            state.apply_mjai_event(event.clone());
        }

        decisions
    }

    fn synthetic_exit_records(
        identity: &str,
        source_net_hash: u64,
        source_version: u32,
    ) -> Vec<ReplayExitRecordV1> {
        replay_guardrail_decisions_for_identity(identity)
            .into_iter()
            .take(2)
            .map(|(key, action, legal_mask)| {
                let mut target = [0.0f32; HYDRA_ACTION_SPACE];
                let mut mask = [0.0f32; HYDRA_ACTION_SPACE];
                target[action as usize] = 1.0;
                mask[action as usize] = 1.0;
                ReplayExitRecordV1 {
                    version: 1,
                    semantics: replay_exit::REPLAY_EXIT_SEMANTICS_V1.to_string(),
                    provenance: replay_exit::REPLAY_EXIT_PROVENANCE.to_string(),
                    key,
                    action,
                    legal_mask_digest: legal_mask_digest_from_f32(&legal_mask),
                    source_net_hash,
                    source_version,
                    root_visit_count: 64,
                    legal_discard_count: legal_mask[..=36]
                        .iter()
                        .filter(|&&value| value > 0.0)
                        .count() as u8,
                    supported_actions: 1,
                    coverage: 1.0,
                    kl_to_base: 0.0,
                    target: target.to_vec(),
                    mask: mask.to_vec(),
                }
            })
            .collect()
    }

    fn synthetic_delta_q_records(
        identity: &str,
        source_net_hash: u64,
        source_version: u32,
    ) -> Vec<ReplayDeltaQRecordV1> {
        replay_guardrail_decisions_for_identity(identity)
            .into_iter()
            .take(2)
            .map(|(key, action, legal_mask)| {
                let mut target = [0.0f32; HYDRA_ACTION_SPACE];
                let mut mask = [0.0f32; HYDRA_ACTION_SPACE];
                target[action as usize] = 0.25;
                mask[action as usize] = 1.0;
                ReplayDeltaQRecordV1 {
                    version: 1,
                    semantics: replay_delta_q::REPLAY_DELTA_Q_SEMANTICS_V1.to_string(),
                    provenance: replay_delta_q::REPLAY_DELTA_Q_PROVENANCE.to_string(),
                    key,
                    action,
                    legal_mask_digest: legal_mask_digest_from_f32(&legal_mask),
                    source_net_hash,
                    source_version,
                    target: target.to_vec(),
                    mask: mask.to_vec(),
                }
            })
            .collect()
    }

    fn write_tar_zst_with_entries(path: &Path, entries: &[(&str, Vec<u8>)]) {
        let file = File::create(path).expect("create archive");
        let encoder = zstd::Encoder::new(file, 19).expect("create zstd encoder");
        let mut builder = Builder::new(encoder.auto_finish());
        for (name, data) in entries {
            let mut header = tar::Header::new_gnu();
            header.set_size(data.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder
                .append_data(&mut header, *name, data.as_slice())
                .expect("append tar entry");
        }
        builder.finish().expect("finish tar builder");
    }

    fn unique_temp_path(label: &str, suffix: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after epoch")
            .as_nanos();
        PathBuf::from("/home/cachybtw/tmp").join(format!("hydra_pipeline_{label}_{unique}{suffix}"))
    }

    fn single_archive_manifest(path: PathBuf) -> DataManifest {
        DataManifest {
            sources: vec![DataSource::Archive(path)],
            total_games: 1,
            train_count: 0,
            val_count: 1,
            counts_exact: false,
        }
    }

    fn single_cache_manifest(path: PathBuf, identity: &str) -> DataManifest {
        DataManifest {
            sources: vec![DataSource::ParsedSampleCache {
                path,
                original_identity: identity.to_string(),
                original_source_path: PathBuf::from(format!("/raw/{identity}")),
            }],
            total_games: 1,
            train_count: 0,
            val_count: 1,
            counts_exact: true,
        }
    }

    fn collect_streamed_samples(
        manifest: &DataManifest,
        config: &StreamingLoaderConfig,
    ) -> Vec<MjaiSample> {
        stream_val_pass(manifest, config, None)
            .collect::<io::Result<Vec<_>>>()
            .expect("stream validation pass")
            .into_iter()
            .flatten()
            .collect()
    }

    #[test]
    fn test_collect_samples_empty() {
        let dataset = MjaiDataset::new(0.9);
        let samples = collect_samples(&dataset);
        assert!(samples.is_empty());
    }

    #[test]
    fn test_build_batches_empty() {
        let device = Default::default();
        let batches = build_batches::<B>(&[], 4, false, &device).expect("empty batches succeed");
        assert!(batches.is_empty());
    }

    #[test]
    fn test_build_batches_creates_correct_count() {
        let dataset = dataset_with_samples(10);
        let samples = collect_samples(&dataset);
        let device = Default::default();
        let batches =
            build_batches::<B>(&samples, 4, false, &device).expect("batch build succeeds");
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].0.dims()[0], 4);
        assert_eq!(batches[1].0.dims()[0], 4);
        assert_eq!(batches[2].0.dims()[0], 2);
    }

    #[test]
    fn test_shuffle_samples_deterministic() {
        let dataset = dataset_with_samples(6);
        let mut a = collect_samples(&dataset);
        let mut b = collect_samples(&dataset);
        shuffle_samples(&mut a, 42);
        shuffle_samples(&mut b, 42);
        let actions_a: Vec<u8> = a.iter().map(|sample| sample.action).collect();
        let actions_b: Vec<u8> = b.iter().map(|sample| sample.action).collect();
        assert_eq!(actions_a, actions_b);
    }

    #[test]
    fn test_collate_sample_chunk_matches_requested_batch_size() {
        let dataset = dataset_with_samples(5);
        let samples = collect_samples(&dataset);
        let device = Default::default();
        let (obs, targets) = collate_sample_chunk::<B>(&samples[..3], false, &device)
            .expect("chunk should collate")
            .expect("chunk should be present");
        assert_eq!(obs.dims()[0], 3);
        assert_eq!(targets.policy_target.dims()[0], 3);
    }

    #[test]
    fn test_fraction_identity_and_split_helpers_cover_edge_cases() {
        assert_eq!(normalized_train_fraction(f32::NAN), 0.0);
        assert_eq!(normalized_train_fraction(-0.25), 0.0);
        assert_eq!(normalized_train_fraction(1.25), 1.0);

        let identity = "game_0001.mjai.json";
        let train_fraction = 0.6;
        assert_eq!(
            should_include_identity(identity, train_fraction, &StreamSplit::Train),
            is_train_game(identity, train_fraction)
        );
        assert_eq!(
            should_include_identity(identity, train_fraction, &StreamSplit::Validation),
            !is_train_game(identity, train_fraction)
        );

        let cfg = StreamingLoaderConfig {
            seed: 7,
            ..StreamingLoaderConfig::default()
        };
        assert_eq!(stream_shuffle_seed(&cfg, 2, 5), 9_000_032);
    }

    #[test]
    fn test_identity_helpers_and_hash_are_deterministic() {
        let loose = identity_for_loose_file(Path::new("/tmp/example.mjai.json"))
            .expect("filename should be valid utf-8");
        assert_eq!(loose, "tmp/example.mjai.json");

        let archive = identity_for_archive_entry(
            Path::new("/tmp/archive.tar.zst"),
            Path::new("nested/game.json"),
        )
        .expect("archive identity should build");
        assert_eq!(archive, "archive.tar.zst/nested/game.json");

        let bad_archive = PathBuf::from(std::ffi::OsString::from_vec(vec![0xFF]));
        let err = identity_for_archive_entry(&bad_archive, Path::new("game.json"))
            .expect_err("invalid utf-8 archive names should fail");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);

        assert_eq!(fnv1a_hash(b"hydra"), fnv1a_hash(b"hydra"));
        assert_ne!(fnv1a_hash(b"hydra"), fnv1a_hash(b"hydrb"));
    }

    #[test]
    fn scan_data_sources_uses_cache_metadata_for_split_counts() {
        let dir = unique_temp_path("cache_scan_dir", "");
        fs::create_dir_all(&dir).expect("create cache dir");

        let train_identity = "league/train_game.mjai.json";
        let val_identity = "league/val_game.mjai.json";
        let train_path = dir.join("train_game.mjai.samples.cache");
        let val_path = dir.join("val_game.mjai.samples.cache");

        let train_is_train = is_train_game(train_identity, 0.5);
        let (train_identity, val_identity) = if train_is_train == is_train_game(val_identity, 0.5) {
            (train_identity, "other/val_game.mjai.json")
        } else {
            (train_identity, val_identity)
        };

        crate::data::parsed_sample_cache::write_parsed_sample_cache(
            &train_path,
            Path::new("/raw/train_game.mjai.json"),
            train_identity,
            &MjaiGame {
                samples: vec![dummy_sample(1)],
                final_scores: [25_000; 4],
            },
        )
        .expect("write train cache");
        crate::data::parsed_sample_cache::write_parsed_sample_cache(
            &val_path,
            Path::new("/raw/val_game.mjai.json"),
            val_identity,
            &MjaiGame {
                samples: vec![dummy_sample(2)],
                final_scores: [25_000; 4],
            },
        )
        .expect("write val cache");

        let manifest =
            scan_data_sources_with_progress(&dir, 0.5, &SourceFilterConfig::default(), None)
                .expect("cache scan should succeed");
        assert_eq!(manifest.total_games, 2);
        assert!(manifest.counts_exact);
        assert_eq!(manifest.train_count + manifest.val_count, 2);
        assert!(
            manifest
                .sources
                .iter()
                .all(|source| matches!(source, DataSource::ParsedSampleCache { .. }))
        );

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn scan_data_sources_filters_cache_using_original_source_path() {
        let dir = unique_temp_path("cache_filter_dir", "");
        fs::create_dir_all(&dir).expect("create cache dir");
        let cache_path = dir.join("filtered_game.mjai.samples.cache");

        crate::data::parsed_sample_cache::write_parsed_sample_cache(
            &cache_path,
            Path::new("/raw/jade/season_1/filtered_game.mjai.json"),
            "season_1/filtered_game.mjai.json",
            &MjaiGame {
                samples: vec![dummy_sample(4)],
                final_scores: [25_000; 4],
            },
        )
        .expect("write cache file");

        let filters = SourceFilterConfig {
            include_source_patterns: vec!["jade/season_1".to_string()],
            exclude_source_patterns: Vec::new(),
        };
        let manifest = scan_data_sources_with_progress(&dir, 0.5, &filters, None)
            .expect("cache scan with filters should succeed");
        assert_eq!(manifest.sources.len(), 1);

        let exclude_filters = SourceFilterConfig {
            include_source_patterns: Vec::new(),
            exclude_source_patterns: vec!["jade/season_1".to_string()],
        };
        let excluded_manifest = scan_data_sources_with_progress(&dir, 0.5, &exclude_filters, None)
            .expect("cache scan with exclude filters should succeed");
        assert!(excluded_manifest.sources.is_empty());

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn stream_val_pass_loads_parsed_sample_cache_using_original_identity_split() {
        let cache_path = unique_temp_path("stream_cache", ".samples.cache");
        let base_identity = "league/cache_stream_game.mjai.json";
        let alt_identity = "league/cache_stream_game_alt.mjai.json";
        let identity = if is_train_game(base_identity, 0.0) {
            alt_identity
        } else {
            base_identity
        };
        crate::data::parsed_sample_cache::write_parsed_sample_cache(
            &cache_path,
            Path::new("/raw/cache_stream_game.mjai.json"),
            identity,
            &MjaiGame {
                samples: vec![dummy_sample(7), dummy_sample(9)],
                final_scores: [28_000, 26_000, 24_000, 22_000],
            },
        )
        .expect("write parsed-sample cache");

        let manifest = single_cache_manifest(cache_path.clone(), identity);
        let config = StreamingLoaderConfig {
            buffer_games: 1,
            buffer_samples: 32,
            train_fraction: 0.0,
            ..StreamingLoaderConfig::default()
        };

        let samples = collect_streamed_samples(&manifest, &config);
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].action, 7);
        assert_eq!(samples[1].action, 9);

        fs::remove_file(cache_path).ok();
    }

    #[test]
    fn test_scan_data_sources_with_fraction_on_single_files() {
        let loose_path = unique_temp_path("single_loose", ".mjai.json");
        fs::write(&loose_path, valid_game_json()).expect("write loose replay");
        let filters = SourceFilterConfig::default();

        let loose_manifest = scan_data_sources_with_fraction(&loose_path, 1.0, &filters, None)
            .expect("loose file should scan");
        assert_eq!(
            loose_manifest.sources,
            vec![DataSource::LooseFile(loose_path.clone())]
        );
        assert_eq!(loose_manifest.total_games, 1);
        assert_eq!(loose_manifest.train_count, 1);
        assert_eq!(loose_manifest.val_count, 0);
        assert!(loose_manifest.counts_exact);

        let archive_path = unique_temp_path("single_archive", ".tar.zst");
        write_tar_zst_with_entries(
            &archive_path,
            &[("game.mjai.json", valid_game_json().into_bytes())],
        );
        let archive_manifest = scan_data_sources_with_fraction(&archive_path, 0.5, &filters, None)
            .expect("archive file should scan");
        assert_eq!(
            archive_manifest.sources,
            vec![DataSource::Archive(archive_path.clone())]
        );
        assert_eq!(archive_manifest.total_games, 0);
        assert_eq!(archive_manifest.train_count, 0);
        assert_eq!(archive_manifest.val_count, 0);
        assert!(!archive_manifest.counts_exact);

        let invalid_path = unique_temp_path("single_invalid", ".txt");
        fs::write(&invalid_path, "not mjai").expect("write invalid input file");
        let err = scan_data_sources_with_fraction(&invalid_path, 0.5, &filters, None)
            .expect_err("non-mjai file should be rejected");
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);

        fs::remove_file(loose_path).ok();
        fs::remove_file(archive_path).ok();
        fs::remove_file(invalid_path).ok();
    }

    #[test]
    fn test_scan_data_sources_accepts_single_zstd_mjai_file() {
        let loose_path = unique_temp_path("single_loose_zstd", ".mjai.json.zst");
        let file = File::create(&loose_path).expect("create zstd loose replay");
        let mut encoder = ZstdEncoder::new(file, 1).expect("create zstd encoder");
        encoder
            .write_all(valid_game_json().as_bytes())
            .expect("write zstd loose replay");
        encoder.finish().expect("finish zstd loose replay");

        let filters = SourceFilterConfig::default();
        let manifest = scan_data_sources_with_fraction(&loose_path, 1.0, &filters, None)
            .expect("zstd loose replay should scan");

        assert_eq!(
            manifest.sources,
            vec![DataSource::LooseFile(loose_path.clone())]
        );
        assert_eq!(manifest.total_games, 1);
        fs::remove_file(loose_path).ok();
    }

    #[test]
    fn source_matches_filters_respects_include_and_exclude_patterns() {
        let source = DataSource::Archive(PathBuf::from("/data/majsoul-jade-mjai-2024.tar"));
        assert!(source_matches_filters(
            &source,
            &SourceFilterConfig::default()
        ));

        let include_only = SourceFilterConfig {
            include_source_patterns: vec!["jade".to_string()],
            exclude_source_patterns: Vec::new(),
        };
        assert!(source_matches_filters(&source, &include_only));

        let exclude_only = SourceFilterConfig {
            include_source_patterns: Vec::new(),
            exclude_source_patterns: vec!["jade".to_string()],
        };
        assert!(!source_matches_filters(&source, &exclude_only));

        let include_then_exclude = SourceFilterConfig {
            include_source_patterns: vec!["majsoul".to_string()],
            exclude_source_patterns: vec!["jade".to_string()],
        };
        assert!(!source_matches_filters(&source, &include_then_exclude));
    }

    #[test]
    fn test_load_mjai_directory_parallel_keeps_sorted_successes_and_skip_count() {
        let dir = unique_temp_path("loader", "");
        fs::create_dir_all(&dir).expect("create temp mjai dir");

        let valid_game = valid_game_json();

        let good_a = dir.join("a_valid.json");
        let good_b = dir.join("b_valid.json");
        let bad = dir.join("c_invalid.json");

        fs::write(&good_a, &valid_game).expect("write first valid game");
        fs::write(&good_b, &valid_game).expect("write second valid game");
        let mut file = fs::File::create(&bad).expect("create bad file");
        writeln!(file, "{{not valid json").expect("write invalid json");

        let dataset = load_mjai_directory(&dir, 0.5).expect("directory load should succeed");
        assert_eq!(dataset.num_games(), 2);
        assert_eq!(dataset.games.len(), 2);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_load_mjai_directory_reads_tar_zst_archive() {
        let archive_path = unique_temp_path("archive", ".tar.zst");

        let raw = valid_game_json();
        let mut gz = GzEncoder::new(Vec::new(), Compression::default());
        gz.write_all(raw.as_bytes()).expect("write gz payload");
        let gz_bytes = gz.finish().expect("finish gz payload");

        write_tar_zst_with_entries(
            &archive_path,
            &[
                ("game_a.mjai.json", raw.clone().into_bytes()),
                ("game_b.mjai.json.gz", gz_bytes),
                ("ignore.txt", b"nope".to_vec()),
            ],
        );

        let dataset = load_mjai_directory(&archive_path, 0.5).expect("archive load should succeed");
        assert_eq!(dataset.num_games(), 2);
        assert_eq!(dataset.games.len(), 2);

        fs::remove_file(&archive_path).ok();
    }

    #[test]
    fn test_load_mjai_directory_reads_mixed_dir_and_archives() {
        let dir = unique_temp_path("mixed", "");
        fs::create_dir_all(&dir).expect("create temp dir");

        let raw = valid_game_json();
        fs::write(dir.join("loose.json"), &raw).expect("write loose game");
        write_tar_zst_with_entries(
            &dir.join("pack.tar.zst"),
            &[("packed.mjai.json", raw.into_bytes())],
        );

        let dataset = load_mjai_directory(&dir, 0.5).expect("mixed load should succeed");
        assert_eq!(dataset.num_games(), 2);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_scan_directory_sources_recurses_into_subdirectories() {
        let dir = unique_temp_path("recursive_scan", "");
        let sub1 = dir.join("sub1");
        let sub2 = dir.join("sub2");
        fs::create_dir_all(&sub1).expect("create first subdir");
        fs::create_dir_all(&sub2).expect("create second subdir");
        fs::write(sub1.join("a.mjai.json.gz"), b"").expect("write first mjai file");
        fs::write(sub2.join("b.mjai.json"), b"").expect("write second mjai file");
        fs::write(dir.join("c.mjai.json.gz"), b"").expect("write root mjai file");
        fs::write(dir.join("d.mjai.json.zst"), b"").expect("write zstd mjai file");

        let sources = scan_directory_sources(&dir).expect("scan directory recursively");
        let loose_count = sources
            .iter()
            .filter(|s| matches!(s, DataSource::LooseFile(_)))
            .count();
        assert_eq!(
            loose_count, 4,
            "should find files in root and subdirectories"
        );

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_identity_for_loose_file_includes_parent_directory() {
        let id = identity_for_loose_file(std::path::Path::new("/data/subdir/game.mjai.json.gz"))
            .expect("nested path identity");
        assert_eq!(id, "subdir/game.mjai.json.gz");

        let id_no_parent = identity_for_loose_file(std::path::Path::new("/game.mjai.json.gz"))
            .expect("root path identity");
        assert_eq!(id_no_parent, "game.mjai.json.gz");
    }

    #[test]
    fn test_load_mjai_directory_recurses_into_subdirectories() {
        let dir = unique_temp_path("recursive_loader", "");
        let nested = dir.join("nested");
        fs::create_dir_all(&nested).expect("create nested dir");

        let raw = valid_game_json();
        fs::write(dir.join("root_game.mjai.json"), &raw).expect("write root game");
        fs::write(nested.join("nested_game.mjai.json"), &raw).expect("write nested game");

        let dataset =
            load_mjai_directory(&dir, 0.5).expect("recursive directory load should succeed");
        assert_eq!(dataset.num_games(), 2);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn stream_val_pass_keeps_delta_q_hydration_when_exit_provenance_mismatches() {
        let replay_log = replay_sidecar_guardrail_log();
        let archive_path = unique_temp_path("sidecar_stream", ".tar.zst");
        let entry_name = "game-1.mjai.json";
        let identity = format!(
            "{}/{}",
            archive_path
                .file_name()
                .and_then(|name| name.to_str())
                .expect("archive name"),
            entry_name
        );

        let exit_records = synthetic_exit_records(&identity, 123, 1);
        let delta_q_records = synthetic_delta_q_records(&identity, 456, 2);

        write_tar_zst_with_entries(&archive_path, &[(entry_name, replay_log.into_bytes())]);

        let manifest = single_archive_manifest(archive_path.clone());
        let config = StreamingLoaderConfig {
            train_fraction: 0.0,
            archive_queue_bound: 1,
            exit_sidecar: Some(Arc::new(ExitSidecarIndex::from_records(exit_records))),
            exit_sidecar_source_net_hash: Some(999),
            exit_sidecar_source_version: Some(99),
            delta_q_sidecar: Some(Arc::new(DeltaQSidecarIndex::from_records(delta_q_records))),
            delta_q_sidecar_source_net_hash: Some(456),
            delta_q_sidecar_source_version: Some(2),
            ..StreamingLoaderConfig::default()
        };

        let samples = collect_streamed_samples(&manifest, &config);

        assert!(samples.iter().all(|sample| sample.exit_target.is_none()));
        assert!(samples.iter().all(|sample| sample.exit_mask.is_none()));
        assert!(samples.iter().any(|sample| sample.delta_q_target.is_some()));
        assert!(samples.iter().any(|sample| sample.delta_q_mask.is_some()));

        fs::remove_file(&archive_path).ok();
    }

    #[test]
    fn stream_val_pass_keeps_exit_hydration_when_delta_q_provenance_mismatches() {
        let replay_log = replay_sidecar_guardrail_log();
        let archive_path = unique_temp_path("sidecar_stream_inverse", ".tar.zst");
        let entry_name = "game-1.mjai.json";
        let identity = format!(
            "{}/{}",
            archive_path
                .file_name()
                .and_then(|name| name.to_str())
                .expect("archive name"),
            entry_name
        );

        let exit_records = synthetic_exit_records(&identity, 123, 1);
        let delta_q_records = synthetic_delta_q_records(&identity, 456, 2);

        write_tar_zst_with_entries(&archive_path, &[(entry_name, replay_log.into_bytes())]);

        let manifest = single_archive_manifest(archive_path.clone());
        let config = StreamingLoaderConfig {
            train_fraction: 0.0,
            archive_queue_bound: 1,
            exit_sidecar: Some(Arc::new(ExitSidecarIndex::from_records(exit_records))),
            exit_sidecar_source_net_hash: Some(123),
            exit_sidecar_source_version: Some(1),
            delta_q_sidecar: Some(Arc::new(DeltaQSidecarIndex::from_records(delta_q_records))),
            delta_q_sidecar_source_net_hash: Some(999),
            delta_q_sidecar_source_version: Some(99),
            ..StreamingLoaderConfig::default()
        };

        let samples = collect_streamed_samples(&manifest, &config);

        assert!(samples.iter().any(|sample| sample.exit_target.is_some()));
        assert!(samples.iter().any(|sample| sample.exit_mask.is_some()));
        assert!(samples.iter().all(|sample| sample.delta_q_target.is_none()));
        assert!(samples.iter().all(|sample| sample.delta_q_mask.is_none()));

        fs::remove_file(&archive_path).ok();
    }

    #[test]
    fn take_next_source_plan_batches_adjacent_loose_files_until_archive_boundary() {
        let loose_a = PathBuf::from("/home/cachybtw/tmp/loose_a.mjai.json");
        let loose_b = PathBuf::from("/home/cachybtw/tmp/loose_b.mjai.json");
        let archive = PathBuf::from("/home/cachybtw/tmp/archive.tar.zst");
        let loose_c = PathBuf::from("/home/cachybtw/tmp/loose_c.mjai.json");
        let mut iter = StreamEpochIterator {
            sources: vec![
                DataSource::LooseFile(loose_a.clone()),
                DataSource::LooseFile(loose_b.clone()),
                DataSource::Archive(archive.clone()),
                DataSource::LooseFile(loose_c.clone()),
            ],
            config: StreamingLoaderConfig::default(),
            split: StreamSplit::Validation,
            shuffle_buffers: false,
            epoch: 0,
            yield_index: 0,
            next_source_index: 0,
            current_source: None,
            progress: None,
        };

        assert_eq!(
            iter.take_next_source_plan(),
            Some(PendingSourceOpen::LooseBatch(vec![loose_a, loose_b]))
        );
        assert_eq!(
            iter.take_next_source_plan(),
            Some(PendingSourceOpen::Archive(archive))
        );
        assert_eq!(
            iter.take_next_source_plan(),
            Some(PendingSourceOpen::LooseBatch(vec![loose_c]))
        );
        assert_eq!(iter.take_next_source_plan(), None);
    }
}
