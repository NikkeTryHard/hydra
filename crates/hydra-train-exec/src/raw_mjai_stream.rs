//! Direct raw-MJAI to Python host-batch stream.
//!
//! The wire format is intentionally little-endian and field-major so Python can wrap each field as
//! a NumPy view before the CUDA copy. Diagnostics must go to stderr; stdout is binary frames only.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc::{self, SyncSender, TrySendError},
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use hydra_bc_shards::BcShardHostScratch;
use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::encoder::OBS_SIZE;
use hydra_data_core::{DataManifest, DataSource};
use hydra_replay_loader::mjai_loader::{
    ReplayLoadPolicy, ReplayObservationProfile, SidecarProvenance, invalid_data,
    load_game_from_stream_into_sink,
};
use hydra_replay_sidecar::{DeltaQSidecarIndex, ExitSidecarIndex};
use rayon::ThreadPoolBuilder;
use rayon::iter::{ParallelBridge, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::data::sample::ReplayHostScratchSink;
use crate::data_pipeline::{
    compact_error_message, identity_for_archive_entry, identity_for_loose_file,
    is_mjai_archive_entry, is_train_game, scan_data_sources,
};

const STREAM_MAGIC: &[u8; 8] = b"HYRMB1\0\0";
const FRAME_KIND_HEADER: u8 = 1;
const FRAME_KIND_BATCH: u8 = 2;
const FRAME_KIND_PROGRESS: u8 = 3;
const FRAME_KIND_END: u8 = 4;
const FIELD_COUNT_BASE: u32 = 13;
const DTYPE_F32: u8 = 1;
const DTYPE_I64: u8 = 2;
const DTYPE_BOOL: u8 = 3;
const GRP_CLASS_COUNT: usize = 24;

/// Deterministic raw-MJAI dataset split streamed to Python.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum RawMjaiStreamSplit {
    /// Training split selected by hashed replay identity.
    Train,
    /// Validation split selected by hashed replay identity.
    Validation,
}

/// Configuration for streaming raw MJAI batches directly to stdout.
#[derive(Debug, Clone)]
pub struct RawMjaiBatchStreamConfig {
    /// Input loose file, directory, or archive path.
    pub input: PathBuf,
    /// Train or validation split to stream.
    pub split: RawMjaiStreamSplit,
    /// Train/validation split fraction.
    pub train_fraction: f32,
    /// Output batch size in rows.
    pub batch_size: usize,
    /// Optional maximum included train games.
    pub max_games: Option<usize>,
    /// Optional maximum output rows; current batch may stop at the cap.
    pub max_samples: Option<usize>,
    /// Worker threads for replay materialization.
    pub num_threads: Option<usize>,
    /// Bounded job/result queue depth.
    pub queue_bound: usize,
    /// Enable suit augmentation.
    pub augment: bool,
    /// Optional pre-scanned source manifest.
    pub source_manifest: Option<DataManifest>,
}

impl Default for RawMjaiBatchStreamConfig {
    fn default() -> Self {
        Self {
            input: PathBuf::from("."),
            split: RawMjaiStreamSplit::Train,
            train_fraction: 0.9,
            batch_size: 2048,
            max_games: None,
            max_samples: None,
            num_threads: None,
            queue_bound: 128,
            augment: false,
            source_manifest: None,
        }
    }
}

/// Summary emitted in progress and end frames.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct RawMjaiBatchStreamTotals {
    /// Train games accepted into the stream.
    pub loaded_games: u64,
    /// Train games skipped due loader/materialization errors.
    pub skipped_games: u64,
    /// Rows emitted to stdout.
    pub samples: u64,
    /// Batches emitted to stdout.
    pub batches: u64,
    /// True when max_games stopped planning.
    pub max_games_reached: bool,
    /// True when max_samples stopped streaming.
    pub max_samples_reached: bool,
}

#[derive(Debug, Clone)]
struct StreamPlanEntry {
    sequence: usize,
    identity: String,
    source: StreamPlanSource,
}

#[derive(Debug, Clone)]
enum StreamPlanSource {
    LooseFile {
        path: PathBuf,
    },
    ArchiveEntry {
        archive_path: PathBuf,
        entry_path: PathBuf,
    },
}

struct MaterializedStreamGame {
    sequence: usize,
    identity: String,
    result: io::Result<GameHostRows>,
}

struct GameHostRows {
    rows: usize,
    scratch: BcShardHostScratch,
}

/// Borrowed destination buffers for one Python-owned pinned host batch.
pub struct RawMjaiPinnedBatchView<'a> {
    /// Output observation buffer `[capacity_rows, 192, 34]`.
    pub obs: &'a mut [f32],
    /// Output action buffer `[capacity_rows]`.
    pub actions: &'a mut [i64],
    /// Output legal mask buffer `[capacity_rows, 46]`, encoded as 0/1 bytes.
    pub legal: &'a mut [u8],
    /// Output value target buffer `[capacity_rows]`.
    pub value: &'a mut [f32],
    /// Output GRP target buffer `[capacity_rows, 24]`.
    pub grp: &'a mut [f32],
    /// Output oracle target buffer `[capacity_rows, 4]`.
    pub oracle: &'a mut [f32],
    /// Output oracle mask buffer `[capacity_rows]`.
    pub oracle_mask: &'a mut [f32],
    /// Output tenpai target buffer `[capacity_rows, 3]`.
    pub tenpai: &'a mut [f32],
    /// Output opponent-next target buffer `[capacity_rows, 102]`.
    pub opp_next: &'a mut [f32],
    /// Output danger target buffer `[capacity_rows, 102]`.
    pub danger: &'a mut [f32],
    /// Output danger mask buffer `[capacity_rows, 102]`.
    pub danger_mask: &'a mut [f32],
    /// Output score-PDF target buffer `[capacity_rows, 64]`.
    pub score_pdf: &'a mut [f32],
    /// Output score-CDF target buffer `[capacity_rows, 64]`.
    pub score_cdf: &'a mut [f32],
    /// Row capacity shared by every field.
    pub capacity_rows: usize,
}

/// Materializes the first raw-MJAI batch directly into caller-owned host buffers.
pub fn fill_raw_mjai_pinned_one_batch(
    config: &RawMjaiBatchStreamConfig,
    dst: RawMjaiPinnedBatchView<'_>,
) -> io::Result<(usize, RawMjaiBatchStreamTotals)> {
    if config.batch_size == 0 {
        return Err(invalid_data("batch size must be > 0"));
    }
    if config.queue_bound == 0 {
        return Err(invalid_data("queue bound must be > 0"));
    }
    validate_pinned_view(&dst)?;
    let source_manifest = match &config.source_manifest {
        Some(manifest) => manifest.clone(),
        None => scan_data_sources(&config.input)?,
    };
    let (entries, max_games_reached) = build_stream_plan(&source_manifest, config)?;
    let mut totals = RawMjaiBatchStreamTotals {
        max_games_reached,
        ..RawMjaiBatchStreamTotals::default()
    };
    let mut sink = PinnedOneBatchWriter::new(dst, config.batch_size);
    materialize_stream_in_order_to_pinned(&entries, config, &mut sink, &mut totals)?;
    sink.finish(&mut totals)?;
    Ok((sink.rows_written, totals))
}

/// Per-call timing and byte counters for the persistent pinned bridge.
#[derive(Debug, Clone, Copy, Default)]
pub struct RawMjaiPinnedStreamStats {
    /// Number of times this stream ran scan/plan setup. Must stay `1` after open.
    pub open_count: u64,
    /// Wall-clock scan+plan+worker-start cost at open.
    pub open_scan_plan_ms: f64,
    /// Wall-clock time spent by the last `next_into` call.
    pub last_next_fill_ms: f64,
    /// Time the last `next_into` call spent waiting for materialized game rows.
    pub last_queue_wait_ms: f64,
    /// Bytes written into caller buffers by the last `next_into` call.
    pub last_bytes_filled: u64,
    /// Materialized games consumed by the last `next_into` call.
    pub last_games_consumed: u64,
}

/// Result from a persistent pinned stream `next_into` call.
#[derive(Debug, Clone, Copy, Default)]
pub struct RawMjaiPinnedNext {
    /// Rows written into caller buffers.
    pub rows: usize,
    /// Accumulated stream totals after this call.
    pub totals: RawMjaiBatchStreamTotals,
    /// Latest stream counters/timers.
    pub stats: RawMjaiPinnedStreamStats,
}

/// Persistent raw-MJAI materializer for caller-owned pinned host buffers.
pub struct RawMjaiPinnedStream {
    config: RawMjaiBatchStreamConfig,
    expected_count: usize,
    result_rx: mpsc::Receiver<io::Result<MaterializedStreamGame>>,
    producer: Option<JoinHandle<io::Result<()>>>,
    workers: Option<JoinHandle<()>>,
    stop: Arc<AtomicBool>,
    pending_results: BTreeMap<usize, MaterializedStreamGame>,
    current_game: Option<GameHostRows>,
    current_offset: usize,
    next_sequence: usize,
    stopped: bool,
    totals: RawMjaiBatchStreamTotals,
    stats: RawMjaiPinnedStreamStats,
}

impl RawMjaiPinnedStream {
    /// Opens a persistent stream, scanning/planning exactly once and starting workers once.
    pub fn open(config: RawMjaiBatchStreamConfig) -> io::Result<Self> {
        if config.batch_size == 0 {
            return Err(invalid_data("batch size must be > 0"));
        }
        if config.queue_bound == 0 {
            return Err(invalid_data("queue bound must be > 0"));
        }
        let opened = Instant::now();
        let source_manifest = match &config.source_manifest {
            Some(manifest) => manifest.clone(),
            None => scan_data_sources(&config.input)?,
        };
        let (entries, max_games_reached) = build_stream_plan(&source_manifest, &config)?;
        let entries: Vec<StreamPlanEntry> = entries
            .into_iter()
            .filter(|entry| matches!(entry.source, StreamPlanSource::LooseFile { .. }))
            .collect();
        let (result_rx, producer, workers, stop) =
            spawn_persistent_loose_workers(&entries, &config)?;
        Ok(Self {
            config,
            expected_count: entries.len(),
            result_rx,
            producer: Some(producer),
            workers: Some(workers),
            stop,
            pending_results: BTreeMap::new(),
            current_game: None,
            current_offset: 0,
            next_sequence: 0,
            stopped: false,
            totals: RawMjaiBatchStreamTotals {
                max_games_reached,
                ..RawMjaiBatchStreamTotals::default()
            },
            stats: RawMjaiPinnedStreamStats {
                open_count: 1,
                open_scan_plan_ms: opened.elapsed().as_secs_f64() * 1000.0,
                ..RawMjaiPinnedStreamStats::default()
            },
        })
    }

    /// Fills the next batch into caller-owned pinned host buffers without rescanning or replanning.
    pub fn next_into(&mut self, dst: RawMjaiPinnedBatchView<'_>) -> io::Result<RawMjaiPinnedNext> {
        validate_pinned_view(&dst)?;
        let started = Instant::now();
        let mut queue_wait_ms = 0.0;
        let mut games_consumed = 0u64;
        let mut writer = PinnedOneBatchWriter::new(dst, self.config.batch_size);
        while !writer.is_full() && !self.stopped && !self.totals.max_samples_reached {
            if self.current_game.is_none() {
                let wait_started = Instant::now();
                if !self.load_next_game()? {
                    self.stopped = true;
                    break;
                }
                queue_wait_ms += wait_started.elapsed().as_secs_f64() * 1000.0;
                games_consumed += 1;
            }
            if let Some(game) = self.current_game.as_ref() {
                let consumed = writer.push_game_rows(
                    game,
                    self.current_offset,
                    &self.config,
                    &mut self.totals,
                )?;
                self.current_offset += consumed;
                if self.current_offset >= game.rows {
                    self.current_game = None;
                    self.current_offset = 0;
                }
                if consumed == 0 {
                    break;
                }
            }
        }
        writer.finish(&mut self.totals)?;
        if writer.rows_written == 0 && self.stopped && self.next_sequence != self.expected_count {
            return Err(invalid_data(format!(
                "pinned stream materialized {} games, expected {}",
                self.next_sequence, self.expected_count
            )));
        }
        self.stats.last_next_fill_ms = started.elapsed().as_secs_f64() * 1000.0;
        self.stats.last_queue_wait_ms = queue_wait_ms;
        self.stats.last_bytes_filled = pinned_rows_bytes(writer.rows_written) as u64;
        self.stats.last_games_consumed = games_consumed;
        Ok(RawMjaiPinnedNext {
            rows: writer.rows_written,
            totals: self.totals,
            stats: self.stats,
        })
    }

    /// Returns current stream totals.
    pub fn totals(&self) -> RawMjaiBatchStreamTotals {
        self.totals
    }

    /// Returns current stream counters/timers.
    pub fn stats(&self) -> RawMjaiPinnedStreamStats {
        self.stats
    }

    fn load_next_game(&mut self) -> io::Result<bool> {
        loop {
            if let Some(game) = self.pending_results.remove(&self.next_sequence) {
                self.next_sequence += 1;
                match game.result {
                    Ok(rows) if rows.rows > 0 => {
                        self.totals.loaded_games += 1;
                        self.current_game = Some(rows);
                        self.current_offset = 0;
                        return Ok(true);
                    }
                    Ok(_) => {
                        self.totals.skipped_games += 1;
                        continue;
                    }
                    Err(err) => {
                        self.totals.skipped_games += 1;
                        eprintln!(
                            "raw MJAI pinned stream skipped {}: {}",
                            game.identity,
                            compact_error_message(&err)
                        );
                        continue;
                    }
                }
            }
            if self.next_sequence >= self.expected_count {
                return Ok(false);
            }
            let game = self.result_rx.recv().map_err(|_| {
                invalid_data("persistent pinned raw-MJAI worker channel closed early")
            })??;
            self.pending_results.insert(game.sequence, game);
        }
    }
}

impl Drop for RawMjaiPinnedStream {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(producer) = self.producer.take() {
            let _ = producer.join();
        }
        while self.result_rx.try_recv().is_ok() {}
        if let Some(workers) = self.workers.take() {
            let _ = workers.join();
        }
    }
}

type PersistentLooseWorkers = (
    mpsc::Receiver<io::Result<MaterializedStreamGame>>,
    JoinHandle<io::Result<()>>,
    JoinHandle<()>,
    Arc<AtomicBool>,
);

fn spawn_persistent_loose_workers(
    entries: &[StreamPlanEntry],
    config: &RawMjaiBatchStreamConfig,
) -> io::Result<PersistentLooseWorkers> {
    let pool = make_optional_pool(config.num_threads, "raw MJAI persistent pinned stream")?;
    let (job_tx, job_rx) = mpsc::sync_channel::<StreamPlanEntry>(config.queue_bound);
    let (result_tx, result_rx) =
        mpsc::sync_channel::<io::Result<MaterializedStreamGame>>(config.queue_bound);
    let stop = Arc::new(AtomicBool::new(false));
    let jobs = entries.to_vec();
    let producer_stop = Arc::clone(&stop);
    let producer = thread::Builder::new()
        .name("raw-mjai-pinned-persistent-producer".into())
        .spawn(move || -> io::Result<()> {
            for job in jobs {
                if send_job_cancelable(&job_tx, &producer_stop, job).is_err() {
                    break;
                }
            }
            Ok(())
        })
        .map_err(|err| io::Error::other(format!("failed to spawn pinned producer: {err}")))?;
    let augment = config.augment;
    let worker_stop = Arc::clone(&stop);
    let workers = thread::Builder::new()
        .name("raw-mjai-pinned-persistent-workers".into())
        .spawn(move || {
            pool.install(|| {
                job_rx.into_iter().par_bridge().for_each(|job| {
                    if worker_stop.load(Ordering::Acquire) {
                        return;
                    }
                    let result = materialize_stream_job(job, augment);
                    let _ = send_result_cancelable(&result_tx, &worker_stop, Ok(result));
                });
            });
        })
        .map_err(|err| io::Error::other(format!("failed to spawn pinned workers: {err}")))?;
    Ok((result_rx, producer, workers, stop))
}

fn send_job_cancelable(
    tx: &SyncSender<StreamPlanEntry>,
    stop: &AtomicBool,
    mut job: StreamPlanEntry,
) -> Result<(), ()> {
    while !stop.load(Ordering::Acquire) {
        match tx.try_send(job) {
            Ok(()) => return Ok(()),
            Err(TrySendError::Full(returned)) => {
                job = returned;
                thread::sleep(Duration::from_millis(1));
            }
            Err(TrySendError::Disconnected(_)) => return Err(()),
        }
    }
    Err(())
}

fn send_result_cancelable(
    tx: &SyncSender<io::Result<MaterializedStreamGame>>,
    stop: &AtomicBool,
    mut result: io::Result<MaterializedStreamGame>,
) -> Result<(), ()> {
    while !stop.load(Ordering::Acquire) {
        match tx.try_send(result) {
            Ok(()) => return Ok(()),
            Err(TrySendError::Full(returned)) => {
                result = returned;
                thread::sleep(Duration::from_millis(1));
            }
            Err(TrySendError::Disconnected(_)) => return Err(()),
        }
    }
    Err(())
}

fn pinned_rows_bytes(rows: usize) -> usize {
    rows * (OBS_SIZE * std::mem::size_of::<f32>()
        + std::mem::size_of::<i64>()
        + HYDRA_ACTION_SPACE * std::mem::size_of::<u8>()
        + std::mem::size_of::<f32>()
        + GRP_CLASS_COUNT * std::mem::size_of::<f32>()
        + 4 * std::mem::size_of::<f32>()
        + std::mem::size_of::<f32>()
        + 3 * std::mem::size_of::<f32>()
        + 102 * std::mem::size_of::<f32>()
        + 102 * std::mem::size_of::<f32>()
        + 102 * std::mem::size_of::<f32>()
        + 64 * std::mem::size_of::<f32>()
        + 64 * std::mem::size_of::<f32>())
}
/// Streams raw MJAI host batches as framed binary to `writer`.
pub fn stream_raw_mjai_batches<W: Write>(
    config: &RawMjaiBatchStreamConfig,
    writer: W,
) -> io::Result<RawMjaiBatchStreamTotals> {
    if config.batch_size == 0 {
        return Err(invalid_data("batch size must be > 0"));
    }
    if config.queue_bound == 0 {
        return Err(invalid_data("queue bound must be > 0"));
    }
    let source_manifest = match &config.source_manifest {
        Some(manifest) => manifest.clone(),
        None => scan_data_sources(&config.input)?,
    };
    let (entries, max_games_reached) = build_stream_plan(&source_manifest, config)?;
    let mut stream = BatchStreamWriter::new(writer, config.batch_size)?;
    let mut totals = RawMjaiBatchStreamTotals {
        max_games_reached,
        ..RawMjaiBatchStreamTotals::default()
    };
    materialize_stream_in_order(&entries, config, &mut stream, &mut totals)?;
    stream.finish(&mut totals)?;
    Ok(totals)
}

fn build_stream_plan(
    source_manifest: &DataManifest,
    config: &RawMjaiBatchStreamConfig,
) -> io::Result<(Vec<StreamPlanEntry>, bool)> {
    let mut out = Vec::new();
    let mut max_games_reached = false;
    for source in &source_manifest.sources {
        match source {
            DataSource::LooseFile(path) => {
                let identity = identity_for_loose_file(path)?;
                let is_train = is_train_game(&identity, config.train_fraction);
                if !split_includes(config.split, is_train) {
                    continue;
                }
                if config.max_games.is_some_and(|max| out.len() >= max) {
                    max_games_reached = true;
                    break;
                }
                out.push(StreamPlanEntry {
                    sequence: out.len(),
                    identity,
                    source: StreamPlanSource::LooseFile { path: path.clone() },
                });
            }
            DataSource::Archive(path) => {
                if enumerate_stream_archive_entries(path, config, &mut out)? {
                    max_games_reached = true;
                    break;
                }
            }
            DataSource::ParsedSampleCache { path, .. } => {
                return Err(invalid_data(format!(
                    "parsed-sample cache input is not supported by raw MJAI stream: {}",
                    path.display()
                )));
            }
        }
    }
    Ok((out, max_games_reached))
}

fn enumerate_stream_archive_entries(
    archive_path: &Path,
    config: &RawMjaiBatchStreamConfig,
    out: &mut Vec<StreamPlanEntry>,
) -> io::Result<bool> {
    let file = File::open(archive_path)?;
    let reader = archive_reader(archive_path, file)?;
    let mut archive = tar::Archive::new(reader);
    for entry_result in archive.entries()? {
        let entry = entry_result?;
        let entry_path = entry.path()?.into_owned();
        if !is_mjai_archive_entry(&entry_path) {
            continue;
        }
        let identity = identity_for_archive_entry(archive_path, &entry_path)?;
        let is_train = is_train_game(&identity, config.train_fraction);
        if !split_includes(config.split, is_train) {
            continue;
        }
        if config.max_games.is_some_and(|max| out.len() >= max) {
            return Ok(true);
        }
        out.push(StreamPlanEntry {
            sequence: out.len(),
            identity,
            source: StreamPlanSource::ArchiveEntry {
                archive_path: archive_path.to_path_buf(),
                entry_path,
            },
        });
    }
    Ok(false)
}

fn split_includes(split: RawMjaiStreamSplit, is_train: bool) -> bool {
    match split {
        RawMjaiStreamSplit::Train => is_train,
        RawMjaiStreamSplit::Validation => !is_train,
    }
}

fn materialize_stream_in_order<W: Write>(
    entries: &[StreamPlanEntry],
    config: &RawMjaiBatchStreamConfig,
    stream: &mut BatchStreamWriter<W>,
    totals: &mut RawMjaiBatchStreamTotals,
) -> io::Result<()> {
    for group in group_stream_entries_preserving_order(entries) {
        match group {
            StreamPlanGroup::Loose(entries) => {
                materialize_loose_stream_group(&entries, config, stream, totals)?;
            }
            StreamPlanGroup::Archive {
                archive_path,
                entries,
            } => materialize_archive_stream_group(&archive_path, &entries, config, stream, totals)?,
        }
        if totals.max_samples_reached {
            break;
        }
    }
    Ok(())
}

enum StreamPlanGroup<'a> {
    Loose(Vec<&'a StreamPlanEntry>),
    Archive {
        archive_path: PathBuf,
        entries: Vec<&'a StreamPlanEntry>,
    },
}

fn group_stream_entries_preserving_order(entries: &[StreamPlanEntry]) -> Vec<StreamPlanGroup<'_>> {
    let mut groups = Vec::new();
    let mut index = 0usize;
    while index < entries.len() {
        match &entries[index].source {
            StreamPlanSource::LooseFile { .. } => {
                let start = index;
                index += 1;
                while index < entries.len()
                    && matches!(entries[index].source, StreamPlanSource::LooseFile { .. })
                {
                    index += 1;
                }
                groups.push(StreamPlanGroup::Loose(
                    entries[start..index].iter().collect(),
                ));
            }
            StreamPlanSource::ArchiveEntry { archive_path, .. } => {
                let archive_path = archive_path.clone();
                let start = index;
                index += 1;
                while index < entries.len() {
                    match &entries[index].source {
                        StreamPlanSource::ArchiveEntry {
                            archive_path: next, ..
                        } if *next == archive_path => {
                            index += 1;
                        }
                        _ => break,
                    }
                }
                groups.push(StreamPlanGroup::Archive {
                    archive_path,
                    entries: entries[start..index].iter().collect(),
                });
            }
        }
    }
    groups
}

fn materialize_loose_stream_group<W: Write>(
    entries: &[&StreamPlanEntry],
    config: &RawMjaiBatchStreamConfig,
    stream: &mut BatchStreamWriter<W>,
    totals: &mut RawMjaiBatchStreamTotals,
) -> io::Result<()> {
    let pool = make_optional_pool(config.num_threads, "raw MJAI loose stream")?;
    let (job_tx, job_rx) = mpsc::sync_channel::<StreamPlanEntry>(config.queue_bound);
    let (result_tx, result_rx) =
        mpsc::sync_channel::<io::Result<MaterializedStreamGame>>(config.queue_bound);
    let jobs: Vec<StreamPlanEntry> = entries.iter().map(|entry| (*entry).clone()).collect();
    let producer = thread::Builder::new()
        .name("raw-mjai-stream-loose-producer".into())
        .spawn(move || -> io::Result<()> {
            for job in jobs {
                if job_tx.send(job).is_err() {
                    break;
                }
            }
            Ok(())
        })
        .map_err(|err| io::Error::other(format!("failed to spawn stream producer: {err}")))?;
    let augment = config.augment;
    let workers = thread::Builder::new()
        .name("raw-mjai-stream-loose-workers".into())
        .spawn(move || {
            pool.install(|| {
                job_rx.into_iter().par_bridge().for_each(|job| {
                    let result = materialize_stream_job(job, augment);
                    let _ = result_tx.send(Ok(result));
                });
            });
        })
        .map_err(|err| io::Error::other(format!("failed to spawn stream workers: {err}")))?;
    collect_stream_results(result_rx, entries.len(), config, stream, totals)?;
    producer
        .join()
        .map_err(|_| io::Error::other("stream producer thread panicked"))??;
    workers
        .join()
        .map_err(|_| io::Error::other("stream worker thread panicked"))?;
    Ok(())
}

fn materialize_archive_stream_group<W: Write>(
    archive_path: &Path,
    entries: &[&StreamPlanEntry],
    config: &RawMjaiBatchStreamConfig,
    stream: &mut BatchStreamWriter<W>,
    totals: &mut RawMjaiBatchStreamTotals,
) -> io::Result<()> {
    let pool = make_optional_pool(config.num_threads, "raw MJAI archive stream")?;
    let wanted: BTreeMap<PathBuf, StreamPlanEntry> = entries
        .iter()
        .map(|entry| match &entry.source {
            StreamPlanSource::ArchiveEntry { entry_path, .. } => {
                (entry_path.clone(), (*entry).clone())
            }
            StreamPlanSource::LooseFile { .. } => unreachable!("archive group contains loose file"),
        })
        .collect();
    let (job_tx, job_rx) = mpsc::sync_channel::<ArchiveStreamJob>(config.queue_bound);
    let (result_tx, result_rx) =
        mpsc::sync_channel::<io::Result<MaterializedStreamGame>>(config.queue_bound);
    let producer_path = archive_path.to_path_buf();
    let producer = thread::Builder::new()
        .name("raw-mjai-stream-archive-producer".into())
        .spawn(move || -> io::Result<()> {
            let file = File::open(&producer_path)?;
            let reader = archive_reader(&producer_path, file)?;
            let mut archive = tar::Archive::new(reader);
            for entry_result in archive.entries()? {
                let mut entry = entry_result?;
                let entry_path = entry.path()?.into_owned();
                let Some(plan) = wanted.get(&entry_path) else {
                    continue;
                };
                let mut data = Vec::with_capacity(entry.size() as usize);
                entry.read_to_end(&mut data)?;
                if job_tx
                    .send(ArchiveStreamJob {
                        plan: plan.clone(),
                        data,
                    })
                    .is_err()
                {
                    break;
                }
            }
            Ok(())
        })
        .map_err(|err| io::Error::other(format!("failed to spawn archive producer: {err}")))?;
    let augment = config.augment;
    let workers = thread::Builder::new()
        .name("raw-mjai-stream-archive-workers".into())
        .spawn(move || {
            pool.install(|| {
                job_rx.into_iter().par_bridge().for_each(|job| {
                    let result = materialize_archive_stream_job(job, augment);
                    let _ = result_tx.send(Ok(result));
                });
            });
        })
        .map_err(|err| io::Error::other(format!("failed to spawn archive workers: {err}")))?;
    collect_stream_results(result_rx, entries.len(), config, stream, totals)?;
    producer
        .join()
        .map_err(|_| io::Error::other("archive producer thread panicked"))??;
    workers
        .join()
        .map_err(|_| io::Error::other("archive worker thread panicked"))?;
    Ok(())
}

#[derive(Debug)]
struct ArchiveStreamJob {
    plan: StreamPlanEntry,
    data: Vec<u8>,
}

fn materialize_stream_job(job: StreamPlanEntry, augment: bool) -> MaterializedStreamGame {
    let mut scratch = BcShardHostScratch::new(0, false, false, false);
    let policy = raw_stream_policy();
    let result = match &job.source {
        StreamPlanSource::LooseFile { path } => File::open(path).and_then(|file| {
            let mut sink = ReplayHostScratchSink::new(&mut scratch, augment);
            load_game_from_stream_into_sink(&job.identity, file, Some(&policy), &mut sink)?;
            Ok(sink.finish().unwrap_or(0))
        }),
        StreamPlanSource::ArchiveEntry { .. } => {
            unreachable!("archive entry passed to loose worker")
        }
    }
    .map(|rows| GameHostRows { rows, scratch });
    MaterializedStreamGame {
        sequence: job.sequence,
        identity: job.identity,
        result,
    }
}

fn materialize_archive_stream_job(job: ArchiveStreamJob, augment: bool) -> MaterializedStreamGame {
    let mut scratch = BcShardHostScratch::new(0, false, false, false);
    let policy = raw_stream_policy();
    let result = (|| -> io::Result<usize> {
        let mut sink = ReplayHostScratchSink::new(&mut scratch, augment);
        load_game_from_stream_into_sink(
            &job.plan.identity,
            BufReader::new(std::io::Cursor::new(job.data)),
            Some(&policy),
            &mut sink,
        )?;
        Ok(sink.finish().unwrap_or(0))
    })()
    .map(|rows| GameHostRows { rows, scratch });
    MaterializedStreamGame {
        sequence: job.plan.sequence,
        identity: job.plan.identity,
        result,
    }
}

fn raw_stream_policy<'a>() -> ReplayLoadPolicy<'a> {
    ReplayLoadPolicy::new(
        hydra_replay_loader::ReplayTargetProfile::with_optional_heads(
            false, false, false, false, false, false,
        ),
        ReplayObservationProfile::BcMinimal,
        SidecarProvenance::default(),
        SidecarProvenance::default(),
        Option::<&'a ExitSidecarIndex>::None,
        Option::<&'a DeltaQSidecarIndex>::None,
    )
}

fn materialize_stream_in_order_to_pinned(
    entries: &[StreamPlanEntry],
    config: &RawMjaiBatchStreamConfig,
    stream: &mut PinnedOneBatchWriter<'_>,
    totals: &mut RawMjaiBatchStreamTotals,
) -> io::Result<()> {
    for group in group_stream_entries_preserving_order(entries) {
        match group {
            StreamPlanGroup::Loose(entries) => {
                materialize_loose_stream_group_to_pinned(&entries, config, stream, totals)?;
            }
            StreamPlanGroup::Archive {
                archive_path,
                entries,
            } => materialize_archive_stream_group_to_pinned(
                &archive_path,
                &entries,
                config,
                stream,
                totals,
            )?,
        }
        if stream.is_full() || totals.max_samples_reached {
            break;
        }
    }
    Ok(())
}

fn materialize_loose_stream_group_to_pinned(
    entries: &[&StreamPlanEntry],
    config: &RawMjaiBatchStreamConfig,
    stream: &mut PinnedOneBatchWriter<'_>,
    totals: &mut RawMjaiBatchStreamTotals,
) -> io::Result<()> {
    let pool = make_optional_pool(config.num_threads, "raw MJAI loose pinned stream")?;
    let (job_tx, job_rx) = mpsc::sync_channel::<StreamPlanEntry>(config.queue_bound);
    let (result_tx, result_rx) =
        mpsc::sync_channel::<io::Result<MaterializedStreamGame>>(config.queue_bound);
    let jobs: Vec<StreamPlanEntry> = entries.iter().map(|entry| (*entry).clone()).collect();
    let producer = thread::Builder::new()
        .name("raw-mjai-pinned-loose-producer".into())
        .spawn(move || -> io::Result<()> {
            for job in jobs {
                if job_tx.send(job).is_err() {
                    break;
                }
            }
            Ok(())
        })
        .map_err(|err| io::Error::other(format!("failed to spawn pinned producer: {err}")))?;
    let augment = config.augment;
    let workers = thread::Builder::new()
        .name("raw-mjai-pinned-loose-workers".into())
        .spawn(move || {
            pool.install(|| {
                job_rx.into_iter().par_bridge().for_each(|job| {
                    let result = materialize_stream_job(job, augment);
                    let _ = result_tx.send(Ok(result));
                });
            });
        })
        .map_err(|err| io::Error::other(format!("failed to spawn pinned workers: {err}")))?;
    collect_pinned_results(result_rx, entries.len(), config, stream, totals)?;
    producer
        .join()
        .map_err(|_| io::Error::other("pinned producer thread panicked"))??;
    workers
        .join()
        .map_err(|_| io::Error::other("pinned worker thread panicked"))?;
    Ok(())
}

fn materialize_archive_stream_group_to_pinned(
    archive_path: &Path,
    entries: &[&StreamPlanEntry],
    config: &RawMjaiBatchStreamConfig,
    stream: &mut PinnedOneBatchWriter<'_>,
    totals: &mut RawMjaiBatchStreamTotals,
) -> io::Result<()> {
    let pool = make_optional_pool(config.num_threads, "raw MJAI archive pinned stream")?;
    let wanted: BTreeMap<PathBuf, StreamPlanEntry> = entries
        .iter()
        .map(|entry| match &entry.source {
            StreamPlanSource::ArchiveEntry { entry_path, .. } => {
                (entry_path.clone(), (*entry).clone())
            }
            StreamPlanSource::LooseFile { .. } => unreachable!("archive group contains loose file"),
        })
        .collect();
    let (job_tx, job_rx) = mpsc::sync_channel::<ArchiveStreamJob>(config.queue_bound);
    let (result_tx, result_rx) =
        mpsc::sync_channel::<io::Result<MaterializedStreamGame>>(config.queue_bound);
    let producer_path = archive_path.to_path_buf();
    let producer = thread::Builder::new()
        .name("raw-mjai-pinned-archive-producer".into())
        .spawn(move || -> io::Result<()> {
            let file = File::open(&producer_path)?;
            let reader = archive_reader(&producer_path, file)?;
            let mut archive = tar::Archive::new(reader);
            for entry_result in archive.entries()? {
                let mut entry = entry_result?;
                let entry_path = entry.path()?.into_owned();
                let Some(plan) = wanted.get(&entry_path) else {
                    continue;
                };
                let mut data = Vec::with_capacity(entry.size() as usize);
                entry.read_to_end(&mut data)?;
                if job_tx
                    .send(ArchiveStreamJob {
                        plan: plan.clone(),
                        data,
                    })
                    .is_err()
                {
                    break;
                }
            }
            Ok(())
        })
        .map_err(|err| {
            io::Error::other(format!("failed to spawn pinned archive producer: {err}"))
        })?;
    let augment = config.augment;
    let workers = thread::Builder::new()
        .name("raw-mjai-pinned-archive-workers".into())
        .spawn(move || {
            pool.install(|| {
                job_rx.into_iter().par_bridge().for_each(|job| {
                    let result = materialize_archive_stream_job(job, augment);
                    let _ = result_tx.send(Ok(result));
                });
            });
        })
        .map_err(|err| {
            io::Error::other(format!("failed to spawn pinned archive workers: {err}"))
        })?;
    collect_pinned_results(result_rx, entries.len(), config, stream, totals)?;
    producer
        .join()
        .map_err(|_| io::Error::other("pinned archive producer thread panicked"))??;
    workers
        .join()
        .map_err(|_| io::Error::other("pinned archive worker thread panicked"))?;
    Ok(())
}

fn collect_pinned_results(
    results: mpsc::Receiver<io::Result<MaterializedStreamGame>>,
    expected_count: usize,
    config: &RawMjaiBatchStreamConfig,
    stream: &mut PinnedOneBatchWriter<'_>,
    totals: &mut RawMjaiBatchStreamTotals,
) -> io::Result<()> {
    let mut next = 0usize;
    let mut pending = BTreeMap::new();
    let mut stopped = false;
    for item in results {
        let game = item?;
        pending.insert(game.sequence, game);
        while let Some(game) = pending.remove(&next) {
            match game.result {
                Ok(rows) if rows.rows > 0 => {
                    totals.loaded_games += 1;
                    stream.push_game(rows, config, totals)?;
                }
                Ok(_) => {
                    totals.skipped_games += 1;
                }
                Err(err) => {
                    totals.skipped_games += 1;
                    eprintln!(
                        "raw MJAI pinned stream skipped {}: {}",
                        game.identity,
                        compact_error_message(&err)
                    );
                }
            }
            next += 1;
            if stream.is_full() || totals.max_samples_reached {
                stopped = true;
                break;
            }
        }
        if stopped {
            break;
        }
    }
    if !stopped && next != expected_count {
        return Err(invalid_data(format!(
            "pinned stream materialized {next} games, expected {expected_count}"
        )));
    }
    Ok(())
}

struct PinnedOneBatchWriter<'a> {
    dst: RawMjaiPinnedBatchView<'a>,
    batch_size: usize,
    pending: BcShardHostScratch,
    pending_rows: usize,
    rows_written: usize,
}

impl<'a> PinnedOneBatchWriter<'a> {
    fn new(dst: RawMjaiPinnedBatchView<'a>, batch_size: usize) -> Self {
        Self {
            dst,
            batch_size,
            pending: BcShardHostScratch::new(batch_size, false, false, false),
            pending_rows: 0,
            rows_written: 0,
        }
    }

    fn push_game(
        &mut self,
        mut game: GameHostRows,
        config: &RawMjaiBatchStreamConfig,
        totals: &mut RawMjaiBatchStreamTotals,
    ) -> io::Result<()> {
        let mut src_offset = 0usize;
        while src_offset < game.rows && !self.is_full() {
            if config
                .max_samples
                .is_some_and(|max| totals.samples as usize >= max)
            {
                totals.max_samples_reached = true;
                break;
            }
            let cap_remaining = config.max_samples.map_or(usize::MAX, |max| {
                max.saturating_sub(totals.samples as usize)
            });
            let pending_free = self.batch_size - self.pending_rows;
            let dst_free = self.dst.capacity_rows - self.rows_written;
            let take = (game.rows - src_offset)
                .min(pending_free)
                .min(dst_free)
                .min(cap_remaining);
            if take == 0 {
                if dst_free == 0 {
                    break;
                }
                totals.max_samples_reached = true;
                break;
            }
            copy_host_rows(
                &game.scratch,
                src_offset,
                &mut self.pending,
                self.pending_rows,
                take,
            );
            self.pending_rows += take;
            src_offset += take;
            if self.pending_rows == self.batch_size || self.pending_rows == dst_free {
                self.flush_pending(totals)?;
            }
        }
        game.scratch.reset(0);
        Ok(())
    }

    fn push_game_rows(
        &mut self,
        game: &GameHostRows,
        src_offset: usize,
        config: &RawMjaiBatchStreamConfig,
        totals: &mut RawMjaiBatchStreamTotals,
    ) -> io::Result<usize> {
        if self.is_full()
            || src_offset >= game.rows
            || config
                .max_samples
                .is_some_and(|max| totals.samples as usize >= max)
        {
            if config
                .max_samples
                .is_some_and(|max| totals.samples as usize >= max)
            {
                totals.max_samples_reached = true;
            }
            return Ok(0);
        }
        let cap_remaining = config.max_samples.map_or(usize::MAX, |max| {
            max.saturating_sub(totals.samples as usize)
        });
        let pending_free = self.batch_size - self.pending_rows;
        let dst_free = self.dst.capacity_rows - self.rows_written;
        let take = (game.rows - src_offset)
            .min(pending_free)
            .min(dst_free)
            .min(cap_remaining);
        if take == 0 {
            if dst_free != 0 {
                totals.max_samples_reached = true;
            }
            return Ok(0);
        }
        copy_host_rows(
            &game.scratch,
            src_offset,
            &mut self.pending,
            self.pending_rows,
            take,
        );
        self.pending_rows += take;
        if self.pending_rows == self.batch_size || self.pending_rows == dst_free {
            self.flush_pending(totals)?;
        }
        Ok(take)
    }

    fn finish(&mut self, totals: &mut RawMjaiBatchStreamTotals) -> io::Result<()> {
        if self.pending_rows > 0 {
            self.flush_pending(totals)?;
        }
        Ok(())
    }

    fn is_full(&self) -> bool {
        self.rows_written >= self.dst.capacity_rows
    }

    fn flush_pending(&mut self, totals: &mut RawMjaiBatchStreamTotals) -> io::Result<()> {
        copy_scratch_to_pinned(
            &mut self.dst,
            self.rows_written,
            &self.pending,
            self.pending_rows,
        )?;
        self.rows_written += self.pending_rows;
        totals.samples += self.pending_rows as u64;
        totals.batches += 1;
        self.pending.reset(self.batch_size);
        self.pending_rows = 0;
        Ok(())
    }
}

fn validate_pinned_view(dst: &RawMjaiPinnedBatchView<'_>) -> io::Result<()> {
    let rows = dst.capacity_rows;
    validate_len("obs", dst.obs.len(), rows * OBS_SIZE)?;
    validate_len("actions", dst.actions.len(), rows)?;
    validate_len("legal", dst.legal.len(), rows * HYDRA_ACTION_SPACE)?;
    validate_len("value", dst.value.len(), rows)?;
    validate_len("grp", dst.grp.len(), rows * GRP_CLASS_COUNT)?;
    validate_len("oracle", dst.oracle.len(), rows * 4)?;
    validate_len("oracle_mask", dst.oracle_mask.len(), rows)?;
    validate_len("tenpai", dst.tenpai.len(), rows * 3)?;
    validate_len("opp_next", dst.opp_next.len(), rows * 102)?;
    validate_len("danger", dst.danger.len(), rows * 102)?;
    validate_len("danger_mask", dst.danger_mask.len(), rows * 102)?;
    validate_len("score_pdf", dst.score_pdf.len(), rows * 64)?;
    validate_len("score_cdf", dst.score_cdf.len(), rows * 64)?;
    Ok(())
}

fn validate_len(name: &str, got: usize, expected: usize) -> io::Result<()> {
    if got != expected {
        return Err(invalid_data(format!(
            "pinned {name} length mismatch: got {got}, expected {expected}"
        )));
    }
    Ok(())
}

fn copy_scratch_to_pinned(
    dst: &mut RawMjaiPinnedBatchView<'_>,
    dst_row: usize,
    src: &BcShardHostScratch,
    rows: usize,
) -> io::Result<()> {
    if dst_row + rows > dst.capacity_rows {
        return Err(invalid_data("pinned batch destination capacity exceeded"));
    }
    copy_slice_rows(&src.obs_flat, OBS_SIZE, 0, dst.obs, dst_row, rows);
    copy_slice_rows(&src.actions, 1, 0, dst.actions, dst_row, rows);
    copy_legal_rows_to_bool(src, dst, dst_row, rows);
    copy_slice_rows(&src.value_target, 1, 0, dst.value, dst_row, rows);
    copy_slice_rows(
        &src.grp_target_flat,
        GRP_CLASS_COUNT,
        0,
        dst.grp,
        dst_row,
        rows,
    );
    copy_slice_rows(&src.oracle_target_flat, 4, 0, dst.oracle, dst_row, rows);
    copy_slice_rows(
        &src.oracle_target_mask,
        1,
        0,
        dst.oracle_mask,
        dst_row,
        rows,
    );
    copy_slice_rows(&src.tenpai_flat, 3, 0, dst.tenpai, dst_row, rows);
    copy_slice_rows(&src.opp_next_flat, 102, 0, dst.opp_next, dst_row, rows);
    copy_slice_rows(&src.danger_flat, 102, 0, dst.danger, dst_row, rows);
    copy_slice_rows(
        &src.danger_mask_flat,
        102,
        0,
        dst.danger_mask,
        dst_row,
        rows,
    );
    copy_slice_rows(&src.score_pdf_flat, 64, 0, dst.score_pdf, dst_row, rows);
    copy_slice_rows(&src.score_cdf_flat, 64, 0, dst.score_cdf, dst_row, rows);
    Ok(())
}

fn copy_legal_rows_to_bool(
    src: &BcShardHostScratch,
    dst: &mut RawMjaiPinnedBatchView<'_>,
    dst_row: usize,
    rows: usize,
) {
    let src_len = rows * HYDRA_ACTION_SPACE;
    let dst_start = dst_row * HYDRA_ACTION_SPACE;
    let dst_end = dst_start + src_len;
    for (out, value) in dst.legal[dst_start..dst_end]
        .iter_mut()
        .zip(&src.legal_mask_flat[..src_len])
    {
        *out = u8::from(*value != 0.0);
    }
}

fn collect_stream_results<W: Write>(
    results: mpsc::Receiver<io::Result<MaterializedStreamGame>>,
    expected_count: usize,
    config: &RawMjaiBatchStreamConfig,
    stream: &mut BatchStreamWriter<W>,
    totals: &mut RawMjaiBatchStreamTotals,
) -> io::Result<()> {
    let mut next = 0usize;
    let mut pending = BTreeMap::new();
    let mut stopped_on_sample_limit = false;
    for item in results {
        let game = item?;
        pending.insert(game.sequence, game);
        while let Some(game) = pending.remove(&next) {
            match game.result {
                Ok(rows) if rows.rows > 0 => {
                    totals.loaded_games += 1;
                    stream.push_game(rows, config, totals)?;
                }
                Ok(_) => {
                    totals.skipped_games += 1;
                    stream.write_progress(totals)?;
                }
                Err(err) => {
                    totals.skipped_games += 1;
                    eprintln!(
                        "raw MJAI stream skipped {}: {}",
                        game.identity,
                        compact_error_message(&err)
                    );
                    stream.write_progress(totals)?;
                }
            }
            next += 1;
            if totals.max_samples_reached {
                stopped_on_sample_limit = true;
                break;
            }
        }
        if stopped_on_sample_limit {
            break;
        }
    }
    if !stopped_on_sample_limit && next != expected_count {
        return Err(invalid_data(format!(
            "stream materialized {next} games, expected {expected_count}"
        )));
    }
    Ok(())
}

struct BatchStreamWriter<W: Write> {
    writer: BufWriter<W>,
    batch_size: usize,
    pending: BcShardHostScratch,
    pending_rows: usize,
    wrote_header: bool,
}

impl<W: Write> BatchStreamWriter<W> {
    fn new(writer: W, batch_size: usize) -> io::Result<Self> {
        let mut out = Self {
            writer: BufWriter::new(writer),
            batch_size,
            pending: BcShardHostScratch::new(batch_size, false, false, false),
            pending_rows: 0,
            wrote_header: false,
        };
        out.write_header()?;
        Ok(out)
    }

    fn write_header(&mut self) -> io::Result<()> {
        let mut payload = Vec::with_capacity(32);
        payload.extend_from_slice(STREAM_MAGIC);
        payload.extend_from_slice(&1u32.to_le_bytes());
        payload.extend_from_slice(&(self.batch_size as u64).to_le_bytes());
        payload.extend_from_slice(&0u32.to_le_bytes());
        payload.extend_from_slice(&FIELD_COUNT_BASE.to_le_bytes());
        self.write_frame(FRAME_KIND_HEADER, &payload)?;
        self.wrote_header = true;
        Ok(())
    }

    fn push_game(
        &mut self,
        mut game: GameHostRows,
        config: &RawMjaiBatchStreamConfig,
        totals: &mut RawMjaiBatchStreamTotals,
    ) -> io::Result<()> {
        let mut src_offset = 0usize;
        while src_offset < game.rows {
            if config
                .max_samples
                .is_some_and(|max| totals.samples as usize >= max)
            {
                totals.max_samples_reached = true;
                break;
            }
            let cap_remaining = config.max_samples.map_or(usize::MAX, |max| {
                max.saturating_sub(totals.samples as usize)
            });
            let pending_free = self.batch_size - self.pending_rows;
            let take = (game.rows - src_offset)
                .min(pending_free)
                .min(cap_remaining);
            if take == 0 {
                totals.max_samples_reached = true;
                break;
            }
            copy_host_rows(
                &game.scratch,
                src_offset,
                &mut self.pending,
                self.pending_rows,
                take,
            );
            self.pending_rows += take;
            src_offset += take;
            if self.pending_rows == self.batch_size {
                self.flush_pending(totals)?;
            }
        }
        game.scratch.reset(0);
        Ok(())
    }

    fn finish(&mut self, totals: &mut RawMjaiBatchStreamTotals) -> io::Result<()> {
        if self.pending_rows > 0 {
            self.flush_pending(totals)?;
        }
        let payload = serde_json::to_vec(totals)
            .map_err(|err| io::Error::other(format!("failed to encode end frame: {err}")))?;
        self.write_frame(FRAME_KIND_END, &payload)?;
        self.writer.flush()
    }

    fn flush_pending(&mut self, totals: &mut RawMjaiBatchStreamTotals) -> io::Result<()> {
        self.pending.batch_size = self.pending_rows;
        let payload = encode_batch_payload(&self.pending, self.pending_rows)?;
        self.write_frame(FRAME_KIND_BATCH, &payload)?;
        totals.samples += self.pending_rows as u64;
        totals.batches += 1;
        self.write_progress(totals)?;
        self.pending.reset(self.batch_size);
        self.pending_rows = 0;
        Ok(())
    }

    fn write_progress(&mut self, totals: &RawMjaiBatchStreamTotals) -> io::Result<()> {
        let payload = serde_json::to_vec(totals)
            .map_err(|err| io::Error::other(format!("failed to encode progress frame: {err}")))?;
        self.write_frame(FRAME_KIND_PROGRESS, &payload)
    }

    fn write_frame(&mut self, kind: u8, payload: &[u8]) -> io::Result<()> {
        self.writer.write_all(&[kind])?;
        self.writer
            .write_all(&(payload.len() as u64).to_le_bytes())?;
        self.writer.write_all(payload)?;
        Ok(())
    }
}

fn copy_host_rows(
    src: &BcShardHostScratch,
    src_offset: usize,
    dst: &mut BcShardHostScratch,
    dst_offset: usize,
    rows: usize,
) {
    copy_slice_rows(
        &src.obs_flat,
        OBS_SIZE,
        src_offset,
        &mut dst.obs_flat,
        dst_offset,
        rows,
    );
    copy_slice_rows(
        &src.actions,
        1,
        src_offset,
        &mut dst.actions,
        dst_offset,
        rows,
    );
    copy_slice_rows(
        &src.legal_mask_flat,
        HYDRA_ACTION_SPACE,
        src_offset,
        &mut dst.legal_mask_flat,
        dst_offset,
        rows,
    );
    copy_slice_rows(
        &src.value_target,
        1,
        src_offset,
        &mut dst.value_target,
        dst_offset,
        rows,
    );
    copy_slice_rows(
        &src.grp_target_flat,
        GRP_CLASS_COUNT,
        src_offset,
        &mut dst.grp_target_flat,
        dst_offset,
        rows,
    );
    copy_slice_rows(
        &src.oracle_target_flat,
        4,
        src_offset,
        &mut dst.oracle_target_flat,
        dst_offset,
        rows,
    );
    copy_slice_rows(
        &src.oracle_target_mask,
        1,
        src_offset,
        &mut dst.oracle_target_mask,
        dst_offset,
        rows,
    );
    copy_slice_rows(
        &src.tenpai_flat,
        3,
        src_offset,
        &mut dst.tenpai_flat,
        dst_offset,
        rows,
    );
    copy_slice_rows(
        &src.opp_next_flat,
        102,
        src_offset,
        &mut dst.opp_next_flat,
        dst_offset,
        rows,
    );
    copy_slice_rows(
        &src.danger_flat,
        102,
        src_offset,
        &mut dst.danger_flat,
        dst_offset,
        rows,
    );
    copy_slice_rows(
        &src.danger_mask_flat,
        102,
        src_offset,
        &mut dst.danger_mask_flat,
        dst_offset,
        rows,
    );
    copy_slice_rows(
        &src.score_pdf_flat,
        64,
        src_offset,
        &mut dst.score_pdf_flat,
        dst_offset,
        rows,
    );
    copy_slice_rows(
        &src.score_cdf_flat,
        64,
        src_offset,
        &mut dst.score_cdf_flat,
        dst_offset,
        rows,
    );
}

fn copy_slice_rows<T: Copy>(
    src: &[T],
    width: usize,
    src_row: usize,
    dst: &mut [T],
    dst_row: usize,
    rows: usize,
) {
    let src_start = src_row * width;
    let src_end = src_start + rows * width;
    let dst_start = dst_row * width;
    let dst_end = dst_start + rows * width;
    dst[dst_start..dst_end].copy_from_slice(&src[src_start..src_end]);
}

fn encode_batch_payload(scratch: &BcShardHostScratch, rows: usize) -> io::Result<Vec<u8>> {
    let mut payload = Vec::with_capacity(rows * 27_000);
    payload.extend_from_slice(&(rows as u64).to_le_bytes());
    payload.extend_from_slice(&0u32.to_le_bytes());
    payload.extend_from_slice(&FIELD_COUNT_BASE.to_le_bytes());
    write_f32_field(
        &mut payload,
        1,
        &[rows as u64, 192, 34],
        &scratch.obs_flat[..rows * OBS_SIZE],
    );
    write_i64_field(&mut payload, 2, &[rows as u64], &scratch.actions[..rows]);
    write_bool_from_f32_field(
        &mut payload,
        3,
        &[rows as u64, HYDRA_ACTION_SPACE as u64],
        &scratch.legal_mask_flat[..rows * HYDRA_ACTION_SPACE],
    );
    write_f32_field(
        &mut payload,
        4,
        &[rows as u64],
        &scratch.value_target[..rows],
    );
    write_f32_field(
        &mut payload,
        5,
        &[rows as u64, GRP_CLASS_COUNT as u64],
        &scratch.grp_target_flat[..rows * GRP_CLASS_COUNT],
    );
    write_f32_field(
        &mut payload,
        6,
        &[rows as u64, 4],
        &scratch.oracle_target_flat[..rows * 4],
    );
    write_f32_field(
        &mut payload,
        7,
        &[rows as u64],
        &scratch.oracle_target_mask[..rows],
    );
    write_f32_field(
        &mut payload,
        8,
        &[rows as u64, 3],
        &scratch.tenpai_flat[..rows * 3],
    );
    write_f32_field(
        &mut payload,
        9,
        &[rows as u64, 102],
        &scratch.opp_next_flat[..rows * 102],
    );
    write_f32_field(
        &mut payload,
        10,
        &[rows as u64, 102],
        &scratch.danger_flat[..rows * 102],
    );
    write_f32_field(
        &mut payload,
        11,
        &[rows as u64, 102],
        &scratch.danger_mask_flat[..rows * 102],
    );
    write_f32_field(
        &mut payload,
        12,
        &[rows as u64, 64],
        &scratch.score_pdf_flat[..rows * 64],
    );
    write_f32_field(
        &mut payload,
        13,
        &[rows as u64, 64],
        &scratch.score_cdf_flat[..rows * 64],
    );
    Ok(payload)
}

fn write_field_header(
    payload: &mut Vec<u8>,
    field_id: u16,
    dtype: u8,
    shape: &[u64],
    byte_len: usize,
) {
    payload.extend_from_slice(&field_id.to_le_bytes());
    payload.push(dtype);
    payload.push(shape.len() as u8);
    for dim in shape {
        payload.extend_from_slice(&dim.to_le_bytes());
    }
    payload.extend_from_slice(&(byte_len as u64).to_le_bytes());
}

fn write_f32_field(payload: &mut Vec<u8>, field_id: u16, shape: &[u64], values: &[f32]) {
    let byte_len = std::mem::size_of_val(values);
    write_field_header(payload, field_id, DTYPE_F32, shape, byte_len);
    for value in values {
        payload.extend_from_slice(&value.to_le_bytes());
    }
}

fn write_i64_field(payload: &mut Vec<u8>, field_id: u16, shape: &[u64], values: &[i64]) {
    let byte_len = std::mem::size_of_val(values);
    write_field_header(payload, field_id, DTYPE_I64, shape, byte_len);
    for value in values {
        payload.extend_from_slice(&value.to_le_bytes());
    }
}

fn write_bool_from_f32_field(payload: &mut Vec<u8>, field_id: u16, shape: &[u64], values: &[f32]) {
    write_field_header(payload, field_id, DTYPE_BOOL, shape, values.len());
    for value in values {
        payload.push(u8::from(*value != 0.0));
    }
}

fn make_optional_pool(threads: Option<usize>, name: &'static str) -> io::Result<rayon::ThreadPool> {
    let mut builder = ThreadPoolBuilder::new().thread_name(move |idx| format!("{name}-{idx}"));
    if let Some(threads) = threads {
        builder = builder.num_threads(threads);
    }
    builder
        .build()
        .map_err(|err| io::Error::other(format!("failed to build {name} thread pool: {err}")))
}

fn archive_reader(path: &Path, file: File) -> io::Result<Box<dyn Read + Send>> {
    if crate::data_pipeline::is_tar_zst_file(path) {
        Ok(Box::new(zstd::Decoder::new(file).map_err(|err| {
            io::Error::other(format!(
                "failed to open zstd archive {}: {err}",
                path.display()
            ))
        })?))
    } else {
        Ok(Box::new(file))
    }
}
