use std::fs;
use std::io::{self, BufReader};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;

use burn::prelude::*;
use indicatif::ProgressBar;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;

use crate::data::mjai_loader::{MjaiDataset, MjaiGame, load_game_from_path, load_game_from_stream};
use crate::data::sample::{MjaiSample, collate_batch, collate_batch_augmented};
use crate::training::losses::HydraTargets;

const MJAI_LOAD_THREAD_STACK_SIZE: usize = 8 * 1024 * 1024;
const MJAI_ARCHIVE_QUEUE_BOUND: usize = 128;

/// Identifies a game's location for deterministic train/val splitting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GameLocator {
    /// For loose files: filename. For archive entries: "archive_name/entry_name"
    pub identity: String,
}

/// Configuration for the streaming loader.
#[derive(Debug, Clone)]
pub struct StreamingLoaderConfig {
    pub buffer_games: usize,
    pub buffer_samples: usize,
    pub train_fraction: f32,
    pub seed: u64,
    pub archive_queue_bound: usize,
    pub max_skip_logs_per_source: usize,
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
        }
    }
}

#[derive(Debug, Clone)]
pub struct DataManifest {
    pub sources: Vec<DataSource>,
    pub total_games: usize,
    pub train_count: usize,
    pub val_count: usize,
    pub counts_exact: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataSource {
    Archive(PathBuf),
    LooseFile(PathBuf),
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
    LooseFile {
        path: PathBuf,
    },
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
    display_name: String,
    data: Vec<u8>,
}

struct SkipLogState {
    source: String,
    emitted: AtomicUsize,
    suppressed: AtomicUsize,
    max_logs: usize,
}

fn compact_identity(identity: &str) -> &str {
    identity.rsplit('/').next().unwrap_or(identity)
}

fn compact_error_message(err: &dyn std::fmt::Display) -> &'static str {
    let raw = err.to_string();
    if raw.contains("Replay desync") {
        "replay desync"
    } else if raw.contains("replay observation failed") {
        "replay observation failed"
    } else if raw.contains("replay action conversion failed") {
        "replay action conversion failed"
    } else if raw.contains("hydra action mapping failed") {
        "hydra action mapping failed"
    } else if raw.contains("failed to parse MJAI events") {
        "invalid mjai events"
    } else if raw.contains("failed to load MJAI events") {
        "failed to load mjai events"
    } else if raw.contains("failed to inspect MJAI stream") {
        "failed to inspect mjai stream"
    } else {
        "load error"
    }
}

impl SkipLogState {
    fn new(source: String, max_logs: usize) -> Self {
        Self {
            source,
            emitted: AtomicUsize::new(0),
            suppressed: AtomicUsize::new(0),
            max_logs,
        }
    }

    fn log_skip(&self, identity: &str, err: &dyn std::fmt::Display) {
        let emitted = self.emitted.fetch_add(1, Ordering::Relaxed);
        if emitted < self.max_logs {
            eprintln!(
                "Skipping {}: {}",
                compact_identity(identity),
                compact_error_message(err)
            );
        } else {
            self.suppressed.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn flush_summary(&self) {
        let suppressed = self.suppressed.load(Ordering::Relaxed);
        if suppressed > 0 {
            eprintln!(
                "Suppressed {suppressed} more replay skip logs from {}",
                self.source
            );
        }
    }
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

fn normalized_train_fraction(train_fraction: f32) -> f32 {
    if train_fraction.is_finite() {
        train_fraction.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn identity_for_loose_file(path: &Path) -> io::Result<String> {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(ToOwned::to_owned)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid filename {}", path.display()),
            )
        })
}

fn identity_for_archive_entry(archive_path: &Path, entry_path: &Path) -> io::Result<String> {
    let archive_name = archive_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid archive name {}", archive_path.display()),
            )
        })?;
    Ok(format!("{archive_name}/{}", entry_path.display()))
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
    progress: Option<&ProgressBar>,
) -> io::Result<DataManifest> {
    let sources = if data_dir.is_file() {
        if is_tar_zst_file(data_dir) {
            vec![DataSource::Archive(data_dir.to_path_buf())]
        } else if is_mjai_file(data_dir) {
            vec![DataSource::LooseFile(data_dir.to_path_buf())]
        } else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "expected directory, MJAI file, or .tar.zst archive, got {}",
                    data_dir.display()
                ),
            ));
        }
    } else {
        scan_directory_sources(data_dir)?
    };

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
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let path = entry.path();
        if !file_type.is_file() {
            continue;
        }
        if is_mjai_file(&path) {
            sources.push(DataSource::LooseFile(path));
        } else if is_tar_zst_file(&path) {
            sources.push(DataSource::Archive(path));
        }
    }
    sources.sort_by(|a, b| data_source_path(a).cmp(data_source_path(b)));
    Ok(sources)
}

fn data_source_path(source: &DataSource) -> &Path {
    match source {
        DataSource::Archive(path) | DataSource::LooseFile(path) => path.as_path(),
    }
}

fn spawn_archive_stream(
    path: PathBuf,
    split: StreamSplit,
    train_fraction: f32,
    progress: Option<ProgressBar>,
    config: &StreamingLoaderConfig,
) -> io::Result<SourceCursor> {
    let archive_queue_bound = config.archive_queue_bound.max(1);
    let max_skip_logs_per_source = config.max_skip_logs_per_source;
    let (tx, rx) = mpsc::sync_channel::<MjaiGame>(archive_queue_bound);
    let path_for_thread = path.clone();
    let path_for_logs = path.display().to_string();
    let skip_state = Arc::new(SkipLogState::new(path_for_logs, max_skip_logs_per_source));
    let handle = thread::Builder::new()
        .name(format!("mjai-stream-{}", path.display()))
        .stack_size(MJAI_LOAD_THREAD_STACK_SIZE)
        .spawn(move || -> io::Result<()> {
            let pool = ThreadPoolBuilder::new()
                .stack_size(MJAI_LOAD_THREAD_STACK_SIZE)
                .build()
                .map_err(|err| {
                    io::Error::other(format!(
                        "failed to build MJAI archive stream thread pool: {err}"
                    ))
                })?;
            let file = fs::File::open(&path_for_thread)?;
            let zstd = zstd::Decoder::new(file).map_err(|err| {
                io::Error::other(format!(
                    "failed to open zstd archive {}: {err}",
                    path_for_thread.display()
                ))
            })?;
            let mut archive = tar::Archive::new(zstd);
            let (job_tx, job_rx) = mpsc::sync_channel::<ArchiveEntryJob>(archive_queue_bound);

            let skip_state_for_parse = Arc::clone(&skip_state);
            let parse_tx = tx.clone();
            let parser = thread::Builder::new()
                .name(format!("mjai-archive-parse-{}", path_for_thread.display()))
                .spawn(move || -> io::Result<()> {
                    pool.install(|| {
                        job_rx.into_iter().par_bridge().try_for_each(|job| {
                            match load_game_from_stream(BufReader::new(std::io::Cursor::new(
                                job.data,
                            ))) {
                                Ok(game) => parse_tx.send(game).map_err(|_| {
                                    io::Error::new(
                                        io::ErrorKind::BrokenPipe,
                                        "archive stream receiver dropped",
                                    )
                                }),
                                Err(err) => {
                                    skip_state_for_parse.log_skip(&job.display_name, &err);
                                    Ok(())
                                }
                            }
                        })
                    })
                })
                .map_err(|err| {
                    io::Error::other(format!(
                        "failed to spawn archive parse thread {}: {err}",
                        path_for_thread.display()
                    ))
                })?;

            for entry_result in archive.entries()? {
                let mut entry = entry_result?;
                let entry_path = entry.path()?.into_owned();
                if !is_mjai_archive_entry(&entry_path) {
                    continue;
                }

                let identity = identity_for_archive_entry(&path_for_thread, &entry_path)?;
                if !should_include_identity(&identity, train_fraction, &split) {
                    continue;
                }

                if let Some(pb) = &progress {
                    pb.inc(1);
                }

                let mut data = Vec::with_capacity(entry.size() as usize);
                if let Err(err) = std::io::Read::read_to_end(&mut entry, &mut data) {
                    skip_state.log_skip(&identity, &err);
                    continue;
                }
                if job_tx
                    .send(ArchiveEntryJob {
                        display_name: identity,
                        data,
                    })
                    .is_err()
                {
                    break;
                }
            }

            drop(job_tx);
            parser.join().map_err(|_| {
                io::Error::other(format!(
                    "archive parse thread panicked for {}",
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

    fn open_next_source(&mut self) -> io::Result<()> {
        if self.current_source.is_some() || self.next_source_index >= self.sources.len() {
            return Ok(());
        }

        let source = self.sources[self.next_source_index].clone();
        self.next_source_index += 1;
        self.current_source = Some(match source {
            DataSource::Archive(path) => spawn_archive_stream(
                path,
                self.split,
                self.config.train_fraction,
                self.progress.clone(),
                &self.config,
            )?,
            DataSource::LooseFile(path) => SourceCursor::LooseFile { path },
        });
        Ok(())
    }

    fn take_next_game(&mut self) -> io::Result<Option<MjaiGame>> {
        loop {
            self.open_next_source()?;

            let Some(source) = self.current_source.take() else {
                return Ok(None);
            };

            match source {
                SourceCursor::LooseFile { path } => {
                    let identity = identity_for_loose_file(&path)?;
                    if !should_include_identity(&identity, self.config.train_fraction, &self.split)
                    {
                        continue;
                    }
                    if let Some(pb) = &self.progress {
                        pb.inc(1);
                    }
                    let result = load_game_from_path(&path);
                    match result {
                        Ok(game) => return Ok(Some(game)),
                        Err(err) => {
                            eprintln!(
                                "Skipping {}: {}",
                                compact_identity(&identity),
                                compact_error_message(&err)
                            );
                            continue;
                        }
                    }
                }
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
    scan_data_sources_with_fraction(
        data_dir,
        StreamingLoaderConfig::default().train_fraction,
        None,
    )
}

pub fn scan_data_sources_with_progress(
    data_dir: &Path,
    train_fraction: f32,
    progress: Option<&ProgressBar>,
) -> io::Result<DataManifest> {
    scan_data_sources_with_fraction(data_dir, train_fraction, progress)
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

fn is_mjai_file(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some(name) if name.ends_with(".json") || name.ends_with(".json.gz")
    )
}

fn is_tar_zst_file(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some(name) if name.ends_with(".tar.zst") || name.contains(".tar-") && name.ends_with(".zst")
    )
}

fn is_mjai_archive_entry(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some(name) if name.ends_with(".json") || name.ends_with(".json.gz") || name.ends_with(".mjai.json") || name.ends_with(".mjai.json.gz")
    )
}

fn load_mjai_archive(path: &Path, train_fraction: f32) -> io::Result<MjaiDataset> {
    let pool = ThreadPoolBuilder::new()
        .stack_size(MJAI_LOAD_THREAD_STACK_SIZE)
        .build()
        .map_err(|err| {
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
            let zstd = zstd::Decoder::new(file).map_err(|err| {
                io::Error::other(format!(
                    "failed to open zstd archive {}: {err}",
                    path_buf.display()
                ))
            })?;
            let mut archive = tar::Archive::new(zstd);

            for entry_result in archive.entries()? {
                let mut entry = entry_result?;
                let entry_path = entry.path()?.into_owned();
                if !is_mjai_archive_entry(&entry_path) {
                    continue;
                }

                let mut data = Vec::with_capacity(entry.size() as usize);
                std::io::Read::read_to_end(&mut entry, &mut data)?;
                let display_name = format!("{} in {}", entry_path.display(), path_buf.display());

                if job_tx.send(ArchiveEntryJob { display_name, data }).is_err() {
                    break;
                }
            }

            Ok(())
        })
        .map_err(|err| io::Error::other(format!("failed to spawn archive reader: {err}")))?;

    let results: Vec<(String, io::Result<MjaiGame>)> = pool.install(|| {
        job_rx
            .into_iter()
            .par_bridge()
            .map(|job| {
                let result = load_game_from_stream(BufReader::new(std::io::Cursor::new(job.data)));
                (job.display_name, result)
            })
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

    for (display_name, result) in results {
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

fn clone_sample(sample: &MjaiSample) -> MjaiSample {
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

pub fn load_mjai_directory(dir: &Path, train_fraction: f32) -> io::Result<MjaiDataset> {
    if dir.is_file() {
        if is_tar_zst_file(dir) {
            return load_mjai_archive(dir, train_fraction);
        }
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "expected directory or .tar.zst archive, got {}",
                dir.display()
            ),
        ));
    }

    let mut paths = Vec::new();
    let mut archives = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let path = entry.path();
        if file_type.is_file() {
            if is_mjai_file(&path) {
                paths.push(path);
            } else if is_tar_zst_file(&path) {
                archives.push(path);
            }
        }
    }
    paths.sort();
    archives.sort();

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

    println!(
        "Loaded {} MJAI games ({} samples, {} skipped) from {}",
        dataset.num_games(),
        dataset.num_samples(),
        skipped,
        dir.display()
    );

    for archive in archives {
        let archive_dataset = load_mjai_archive(&archive, train_fraction)?;
        for game in archive_dataset.games {
            dataset.add_game(game);
        }
    }

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
) -> Vec<(Tensor<B, 3>, HydraTargets<B>)> {
    if samples.is_empty() || batch_size == 0 {
        return Vec::new();
    }

    samples
        .chunks(batch_size)
        .map(|chunk| {
            let owned: Vec<MjaiSample> = chunk.iter().map(|sample| clone_sample(sample)).collect();
            let batch = if augment {
                collate_batch_augmented(&owned, device)
            } else {
                collate_batch(&owned, device)
            };
            (batch.obs.clone(), batch.into_hydra_targets())
        })
        .collect()
}

pub fn collate_sample_chunk<B: Backend>(
    samples: &[&MjaiSample],
    augment: bool,
    device: &B::Device,
) -> Option<(Tensor<B, 3>, HydraTargets<B>)> {
    if samples.is_empty() {
        return None;
    }

    let owned: Vec<MjaiSample> = samples.iter().map(|sample| clone_sample(sample)).collect();
    let batch = if augment {
        collate_batch_augmented(&owned, device)
    } else {
        collate_batch(&owned, device)
    };
    Some((batch.obs.clone(), batch.into_hydra_targets()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use flate2::Compression;
    use flate2::write::GzEncoder;
    use hydra_core::action::HYDRA_ACTION_SPACE;
    use hydra_core::encoder::OBS_SIZE;
    use std::fs::File;
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};
    use tar::Builder;

    use crate::data::mjai_loader::{MjaiDataset, MjaiGame};

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

    #[test]
    fn test_collect_samples_empty() {
        let dataset = MjaiDataset::new(0.9);
        let samples = collect_samples(&dataset);
        assert!(samples.is_empty());
    }

    #[test]
    fn test_build_batches_empty() {
        let device = Default::default();
        let batches = build_batches::<B>(&[], 4, false, &device);
        assert!(batches.is_empty());
    }

    #[test]
    fn test_build_batches_creates_correct_count() {
        let dataset = dataset_with_samples(10);
        let samples = collect_samples(&dataset);
        let device = Default::default();
        let batches = build_batches::<B>(&samples, 4, false, &device);
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
        let (obs, targets) =
            collate_sample_chunk::<B>(&samples[..3], false, &device).expect("chunk should collate");
        assert_eq!(obs.dims()[0], 3);
        assert_eq!(targets.policy_target.dims()[0], 3);
    }

    #[test]
    fn test_compact_error_message_reduces_to_short_reason() {
        let raw = format!(
            "replay observation failed:\n  Replay desync:\n    phase: WaitAct\n    drawn: Some(128)\n    {}",
            "extra ".repeat(64)
        );
        let compact = compact_error_message(&raw);
        assert_eq!(compact, "replay desync");
    }

    #[test]
    fn test_compact_identity_uses_file_name_only() {
        let identity = "majsoul-jade-mjai-2021.tar.zst/./210614_44a21457_86ce_4215_9ac2_aeb845f15521.mjai.json";
        assert_eq!(
            compact_identity(identity),
            "210614_44a21457_86ce_4215_9ac2_aeb845f15521.mjai.json"
        );
    }

    #[test]
    fn test_load_mjai_directory_parallel_keeps_sorted_successes_and_skip_count() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("hydra_pipeline_loader_{unique}"));
        fs::create_dir_all(&dir).expect("create temp mjai dir");

        let valid_game = [
            r#"{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"dora_marker":"1m","tehais":[["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"],["1s","2s","3s","4s","5s","6s","7s","8s","9s","E","S","W","N"],["P","F","C","1m","1m","2m","2m","3m","3m","4m","4m","5m","5m"],["6p","6p","7p","7p","8p","8p","9p","9p","1s","1s","2s","2s","3s"]]}"#,
            r#"{"type":"end_kyoku"}"#,
        ]
        .join("\n");

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
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after epoch")
            .as_nanos();
        let archive_path =
            std::env::temp_dir().join(format!("hydra_pipeline_archive_{unique}.tar.zst"));

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
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("hydra_pipeline_mixed_{unique}"));
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
}
