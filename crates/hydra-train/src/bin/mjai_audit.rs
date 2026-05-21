use std::collections::HashMap;
use std::fs;
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use hydra_replay_loader::archive_helpers::is_mjai_archive_entry;
use hydra_replay_loader::mjai_loader::{load_game_from_path, load_game_from_stream};
use hydra_sample_cache::is_parsed_sample_cache_file;
use hydra_train_exec::data_pipeline::{DataSource, scan_data_sources};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

const MJAI_AUDIT_THREAD_STACK_SIZE: usize = 8 * 1024 * 1024;

#[derive(Debug)]
struct AuditConfig {
    data_dir: PathBuf,
    threads: usize,
    failure_examples: usize,
    failure_inventory_dir: Option<PathBuf>,
    debug_first_failure: bool,
}

#[derive(Debug, Clone, PartialEq, Default)]
struct AuditTotals {
    loaded: usize,
    skipped: usize,
    samples: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct AuditRates {
    elapsed_secs: f64,
    files_per_sec: f64,
    samples_per_sec: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct FailureInventoryRecord {
    source: String,
    identity: String,
    error: String,
}

#[derive(Debug)]
struct FailureInventoryWriter {
    source: String,
    path: PathBuf,
    writer: BufWriter<fs::File>,
}

#[derive(Debug, Default)]
struct AuditSharedState {
    loaded: AtomicUsize,
    skipped: AtomicUsize,
    samples: AtomicUsize,
    error_buckets: Mutex<HashMap<String, usize>>,
    failure_examples: Mutex<Vec<(String, String)>>,
}

impl FailureInventoryWriter {
    fn create(inventory_dir: &Path, source_path: &Path) -> Result<Self, String> {
        fs::create_dir_all(inventory_dir).map_err(|err| {
            format!(
                "failed to create failure inventory dir {}: {err}",
                inventory_dir.display()
            )
        })?;
        let path = failure_inventory_path(inventory_dir, source_path);
        let file = fs::File::create(&path).map_err(|err| {
            format!(
                "failed to create failure inventory {}: {err}",
                path.display()
            )
        })?;
        Ok(Self {
            source: source_identity(source_path),
            path,
            writer: BufWriter::new(file),
        })
    }

    fn write_failure(&mut self, identity: &str, error: &str) -> Result<(), String> {
        serde_json::to_writer(
            &mut self.writer,
            &FailureInventoryRecord {
                source: self.source.clone(),
                identity: identity.to_string(),
                error: error.to_string(),
            },
        )
        .map_err(|err| {
            format!(
                "failed to write failure inventory {}: {err}",
                self.path.display()
            )
        })?;
        self.writer.write_all(b"\n").map_err(|err| {
            format!(
                "failed to write failure inventory {}: {err}",
                self.path.display()
            )
        })
    }

    fn finalize(self) -> Result<(), String> {
        let path = self.path;
        let file = self.writer.into_inner().map_err(|err| {
            format!(
                "failed to flush failure inventory {}: {err}",
                path.display()
            )
        })?;
        file.sync_all()
            .map_err(|err| format!("failed to sync failure inventory {}: {err}", path.display()))
    }
}

impl AuditSharedState {
    fn record_loaded(&self, sample_count: usize) {
        self.loaded.fetch_add(1, Ordering::Relaxed);
        self.samples.fetch_add(sample_count, Ordering::Relaxed);
    }

    fn totals(&self) -> AuditTotals {
        AuditTotals {
            loaded: self.loaded.load(Ordering::Relaxed),
            skipped: self.skipped.load(Ordering::Relaxed),
            samples: self.samples.load(Ordering::Relaxed),
        }
    }

    fn snapshot_error_buckets(&self) -> HashMap<String, usize> {
        self.error_buckets
            .lock()
            .expect("lock error buckets")
            .clone()
    }

    fn snapshot_failure_examples(&self) -> Vec<(String, String)> {
        self.failure_examples
            .lock()
            .expect("lock failure examples")
            .clone()
    }
}

fn usage(program: &str) -> String {
    format!(
        "Usage: {program} <data-dir> [--threads N] [--failure-examples N] [--failure-inventory-dir DIR] [--debug-first-failure]"
    )
}

fn parse_args<I>(args: I) -> Result<AuditConfig, String>
where
    I: IntoIterator<Item = String>,
{
    let mut args = args.into_iter();
    let program = args.next().unwrap_or_else(|| "mjai_audit".to_string());
    let Some(data_dir) = args.next() else {
        return Err(usage(&program));
    };

    let mut threads = 16usize;
    let mut failure_examples = 20usize;
    let mut failure_inventory_dir = None;
    let mut debug_first_failure = false;

    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--threads" => {
                let Some(value) = args.next() else {
                    return Err("missing value for --threads".to_string());
                };
                threads = value
                    .parse::<usize>()
                    .map_err(|err| format!("invalid --threads value {value:?}: {err}"))?;
                if threads == 0 {
                    return Err("--threads must be greater than 0".to_string());
                }
            }
            "--failure-examples" => {
                let Some(value) = args.next() else {
                    return Err("missing value for --failure-examples".to_string());
                };
                failure_examples = value
                    .parse::<usize>()
                    .map_err(|err| format!("invalid --failure-examples value {value:?}: {err}"))?;
            }
            "--failure-inventory-dir" => {
                let Some(value) = args.next() else {
                    return Err("missing value for --failure-inventory-dir".to_string());
                };
                failure_inventory_dir = Some(PathBuf::from(value));
            }
            "--debug-first-failure" => {
                debug_first_failure = true;
            }
            _ => return Err(format!("unknown argument {flag:?}\n{}", usage(&program))),
        }
    }

    Ok(AuditConfig {
        data_dir: PathBuf::from(data_dir),
        threads,
        failure_examples,
        failure_inventory_dir,
        debug_first_failure,
    })
}

fn is_archive_file(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some(name)
            if name.ends_with(".tar.zst")
                || name.contains(".tar-") && name.ends_with(".zst")
    )
}

fn collect_sources(data_dir: &Path) -> Result<Vec<DataSource>, String> {
    if data_dir.is_dir() {
        return scan_data_sources(data_dir)
            .map(|manifest| manifest.sources)
            .map_err(|err| format!("failed to scan data dir {}: {err}", data_dir.display()));
    }
    if data_dir.is_file() {
        if is_parsed_sample_cache_file(data_dir) {
            return Err(format!(
                "parsed-sample cache input is not supported by mjai_audit yet: {}",
                data_dir.display()
            ));
        }
        return Ok(vec![if is_archive_file(data_dir) {
            DataSource::Archive(data_dir.to_path_buf())
        } else {
            DataSource::LooseFile(data_dir.to_path_buf())
        }]);
    }
    scan_data_sources(data_dir)
        .map(|manifest| manifest.sources)
        .map_err(|err| format!("failed to scan data dir {}: {err}", data_dir.display()))
}

fn source_identity(path: &Path) -> String {
    path.display().to_string()
}

fn archive_entry_identity(archive_path: &Path, entry_path: &Path) -> String {
    format!("{}/{}", archive_path.display(), entry_path.display())
}

fn archive_entry_index_identity(archive_path: &Path, entry_index: usize) -> String {
    format!("{}#entry[{entry_index}]", archive_path.display())
}

fn fnv1a_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn inventory_name_component(value: &str) -> String {
    let sanitized = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-' | '_') {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();
    if sanitized.is_empty() {
        "source".to_string()
    } else {
        sanitized
    }
}

fn failure_inventory_path(inventory_dir: &Path, source_path: &Path) -> PathBuf {
    let source = source_identity(source_path);
    let source_name = source_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("source");
    let hash = fnv1a_hash(source.as_bytes());
    inventory_dir.join(format!(
        "{}-{hash:016x}.jsonl",
        inventory_name_component(source_name)
    ))
}

fn persist_failure_record(
    inventory_dir: Option<&Path>,
    source_path: &Path,
    inventory_writer: &mut Option<FailureInventoryWriter>,
    identity: &str,
    error: &str,
) -> Result<(), String> {
    let Some(inventory_dir) = inventory_dir else {
        return Ok(());
    };
    if inventory_writer.is_none() {
        *inventory_writer = Some(FailureInventoryWriter::create(inventory_dir, source_path)?);
    }
    inventory_writer
        .as_mut()
        .expect("failure inventory writer should exist")
        .write_failure(identity, error)
}

fn finalize_failure_inventory(
    inventory_writer: Option<FailureInventoryWriter>,
) -> Result<(), String> {
    if let Some(inventory_writer) = inventory_writer {
        inventory_writer.finalize()?;
    }
    Ok(())
}

fn record_failure(
    state: &AuditSharedState,
    config: &AuditConfig,
    source_path: &Path,
    inventory_writer: &mut Option<FailureInventoryWriter>,
    identity: String,
    error: String,
) -> Result<(), String> {
    persist_failure_record(
        config.failure_inventory_dir.as_deref(),
        source_path,
        inventory_writer,
        &identity,
        &error,
    )?;

    state.skipped.fetch_add(1, Ordering::Relaxed);
    let bucket = summarize_error(&error);
    {
        let mut buckets = state.error_buckets.lock().expect("lock error buckets");
        *buckets.entry(bucket).or_insert(0) += 1;
    }
    {
        let mut examples = state
            .failure_examples
            .lock()
            .expect("lock failure examples");
        if should_record_failure_example(examples.len(), config.failure_examples) {
            examples.push((identity, error));
        }
    }
    Ok(())
}

fn audit_loose_source(
    path: &Path,
    config: &AuditConfig,
    state: &AuditSharedState,
    inventory_writer: &mut Option<FailureInventoryWriter>,
) -> Result<(), String> {
    match load_game_from_path(path) {
        Ok(game) => state.record_loaded(game.num_samples()),
        Err(err) => {
            record_failure(
                state,
                config,
                path,
                inventory_writer,
                source_identity(path),
                err.to_string(),
            )?;
        }
    }
    Ok(())
}

fn audit_archive_source(
    path: &Path,
    config: &AuditConfig,
    state: &AuditSharedState,
    inventory_writer: &mut Option<FailureInventoryWriter>,
) -> Result<(), String> {
    let file = match fs::File::open(path) {
        Ok(file) => file,
        Err(err) => {
            record_failure(
                state,
                config,
                path,
                inventory_writer,
                source_identity(path),
                format!("failed to open archive {}: {err}", path.display()),
            )?;
            return Ok(());
        }
    };
    let zstd = match zstd::Decoder::new(file) {
        Ok(zstd) => zstd,
        Err(err) => {
            record_failure(
                state,
                config,
                path,
                inventory_writer,
                source_identity(path),
                format!("failed to decode archive {}: {err}", path.display()),
            )?;
            return Ok(());
        }
    };
    let mut archive = tar::Archive::new(zstd);

    let entries = match archive.entries() {
        Ok(entries) => entries,
        Err(err) => {
            record_failure(
                state,
                config,
                path,
                inventory_writer,
                source_identity(path),
                format!("failed to iterate archive {}: {err}", path.display()),
            )?;
            return Ok(());
        }
    };

    for (entry_index, entry_result) in entries.enumerate() {
        let entry = match entry_result {
            Ok(entry) => entry,
            Err(err) => {
                record_failure(
                    state,
                    config,
                    path,
                    inventory_writer,
                    archive_entry_index_identity(path, entry_index),
                    format!("failed to read archive entry in {}: {err}", path.display()),
                )?;
                break;
            }
        };
        let entry_path = match entry.path() {
            Ok(entry_path) => entry_path.into_owned(),
            Err(err) => {
                record_failure(
                    state,
                    config,
                    path,
                    inventory_writer,
                    archive_entry_index_identity(path, entry_index),
                    format!(
                        "failed to inspect archive entry in {}: {err}",
                        path.display()
                    ),
                )?;
                continue;
            }
        };
        if !is_mjai_archive_entry(&entry_path) {
            continue;
        }

        match load_game_from_stream(BufReader::new(entry)) {
            Ok(game) => state.record_loaded(game.num_samples()),
            Err(err) => {
                record_failure(
                    state,
                    config,
                    path,
                    inventory_writer,
                    archive_entry_identity(path, &entry_path),
                    err.to_string(),
                )?;
            }
        }
    }

    Ok(())
}

fn audit_source(
    source: &DataSource,
    config: &AuditConfig,
    state: &AuditSharedState,
) -> Result<(), String> {
    let mut inventory_writer = None;
    let audit_result = match source {
        DataSource::Archive(path) => {
            audit_archive_source(path, config, state, &mut inventory_writer)
        }
        DataSource::LooseFile(path) => {
            audit_loose_source(path, config, state, &mut inventory_writer)
        }
        DataSource::ParsedSampleCache { path, .. } => Err(format!(
            "parsed-sample cache input is not supported by mjai_audit yet: {}",
            path.display()
        )),
    };
    audit_result.as_ref()?;
    finalize_failure_inventory(inventory_writer)
}

fn summarize_error(err: &str) -> String {
    let summary = err.lines().next().unwrap_or(err).trim();
    if summary.is_empty() {
        "unknown error".to_string()
    } else {
        summary.to_string()
    }
}

fn throughput_per_second(units: usize, elapsed_secs: f64) -> f64 {
    if elapsed_secs > 0.0 {
        units as f64 / elapsed_secs
    } else {
        0.0
    }
}

fn sort_error_buckets(error_buckets: HashMap<String, usize>) -> Vec<(String, usize)> {
    let mut buckets = error_buckets.into_iter().collect::<Vec<_>>();
    buckets.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    buckets
}

fn should_record_failure_example(current_examples: usize, limit: usize) -> bool {
    current_examples < limit
}

fn total_files(totals: &AuditTotals) -> usize {
    totals.loaded + totals.skipped
}

fn compute_audit_rates(totals: &AuditTotals, elapsed_secs: f64) -> AuditRates {
    AuditRates {
        elapsed_secs,
        files_per_sec: throughput_per_second(total_files(totals), elapsed_secs),
        samples_per_sec: throughput_per_second(totals.samples, elapsed_secs),
    }
}

fn audit_totals_summary_line(totals: &AuditTotals) -> String {
    format!(
        "Audit complete: loaded={} skipped={} samples={} total={}",
        totals.loaded,
        totals.skipped,
        totals.samples,
        total_files(totals)
    )
}

fn audit_speed_summary_line(rates: &AuditRates) -> String {
    format!(
        "Speed: elapsed={:.2}s files_per_sec={:.2} samples_per_sec={:.2}",
        rates.elapsed_secs, rates.files_per_sec, rates.samples_per_sec
    )
}

fn failure_report_lines(buckets: &[(String, usize)], examples: &[(String, String)]) -> Vec<String> {
    if buckets.is_empty() {
        return vec!["No failures detected.".to_string()];
    }

    let mut lines = vec!["Top failure buckets:".to_string()];
    for (bucket, count) in buckets.iter().take(20) {
        lines.push(format!("  {count:>6}  {bucket}"));
    }
    if !examples.is_empty() {
        lines.push("Failure examples:".to_string());
        for (path, err) in examples {
            lines.push(format!("---\n{path}\n{err}"));
        }
    }
    lines
}

fn run() -> Result<(), String> {
    let started_at = Instant::now();
    let config = parse_args(std::env::args())?;

    let sources = collect_sources(&config.data_dir)?;
    let total = sources.len();
    if config.debug_first_failure {
        if total != 1 {
            return Err("--debug-first-failure requires a single replay file input".to_string());
        }
        let source = sources.first().expect("total checked above");
        let path = match source {
            DataSource::LooseFile(path) => path,
            DataSource::Archive(_) | DataSource::ParsedSampleCache { .. } => {
                return Err("--debug-first-failure supports loose replay files only".to_string());
            }
        };
        let file = fs::File::open(path)
            .map_err(|err| format!("failed to open replay {}: {err}", path.display()))?;
        let zstd = zstd::Decoder::new(file)
            .map_err(|err| format!("failed to decode replay {}: {err}", path.display()))?;
        match hydra_replay_loader::mjai_loader::debug_first_replay_failure_from_reader(
            BufReader::new(zstd),
        )
        .map_err(|err| format!("failed to debug replay {}: {err}", path.display()))?
        {
            Some(report) => println!("{report}"),
            None => println!("No replay failure detected."),
        }
        return Ok(());
    }
    println!(
        "Auditing {} data sources from {} with {} threads",
        total,
        config.data_dir.display(),
        config.threads
    );

    let progress = Arc::new(ProgressBar::new(total as u64));
    progress.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] [{wide_bar}] {pos}/{len} ({percent}%) eta {eta_precise}",
        )
        .expect("valid progress template")
        .progress_chars("=>-"),
    );

    let state = Arc::new(AuditSharedState::default());

    let pool = ThreadPoolBuilder::new()
        .num_threads(config.threads)
        .stack_size(MJAI_AUDIT_THREAD_STACK_SIZE)
        .build()
        .map_err(|err| format!("failed to build rayon pool: {err}"))?;

    let audit_results = {
        let progress = Arc::clone(&progress);
        let state = Arc::clone(&state);
        pool.install(|| {
            sources
                .par_iter()
                .map(|source| {
                    let result = audit_source(source, &config, state.as_ref());
                    progress.inc(1);
                    result
                })
                .collect::<Vec<_>>()
        })
    };

    progress.finish_and_clear();

    if let Some(err) = audit_results.into_iter().find_map(Result::err) {
        return Err(err);
    }

    let totals = state.totals();
    let rates = compute_audit_rates(&totals, started_at.elapsed().as_secs_f64());

    println!("{}", audit_totals_summary_line(&totals));
    println!("{}", audit_speed_summary_line(&rates));

    let buckets = sort_error_buckets(state.snapshot_error_buckets());
    let examples = state.snapshot_failure_examples();
    for line in failure_report_lines(&buckets, &examples) {
        println!("{line}");
    }

    Ok(())
}

fn exit_code_for_run_result(result: &Result<(), String>) -> i32 {
    if result.is_ok() { 0 } else { 1 }
}

fn main() {
    let result = run();
    if let Err(err) = &result {
        eprintln!("{err}");
        std::process::exit(exit_code_for_run_result(&result));
    }
}

#[cfg(test)]
#[path = "mjai_audit/tests.rs"]
mod tests;
