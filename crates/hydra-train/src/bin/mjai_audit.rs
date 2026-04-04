use std::collections::HashMap;
use std::fs;
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use hydra_train::data::archive_helpers::is_mjai_archive_entry;
use hydra_train::data::mjai_loader::{load_game_from_path, load_game_from_stream};
use hydra_train::data::pipeline::{DataSource, scan_data_sources};
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
        "Usage: {program} <data-dir> [--threads N] [--failure-examples N] [--failure-inventory-dir DIR]"
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
            _ => return Err(format!("unknown argument {flag:?}\n{}", usage(&program))),
        }
    }

    Ok(AuditConfig {
        data_dir: PathBuf::from(data_dir),
        threads,
        failure_examples,
        failure_inventory_dir,
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
mod tests {
    use super::*;
    use std::ffi::OsStr;
    use std::io::{Cursor, Write};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn valid_mjai_log() -> String {
        [
            r#"{"type":"start_game","names":["a","b","c","d"],"id":"game-1"}"#,
            r#"{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"dora_marker":"1m","tehais":[["6m","6m","6m","7m","8m","9m","1p","2p","3p","4p","5p","6p","7p","8p"],["1s","2s","3s","4s","5s","6s","7s","8s","9s","E","S","W","N"],["1m","1m","2m","2m","3m","3m","4m","4m","5m","5m","6m","6m","7m"],["1p","1p","2p","2p","3p","3p","4p","4p","5p","5p","6p","6p","7p"]]}"#,
            r#"{"type":"dahai","actor":0,"pai":"8p","tsumogiri":false}"#,
            r#"{"type":"tsumo","actor":1,"pai":"P"}"#,
            r#"{"type":"dahai","actor":1,"pai":"P","tsumogiri":true}"#,
            r#"{"type":"ryukyoku"}"#,
            r#"{"type":"end_kyoku"}"#,
        ]
        .join("\n")
    }

    fn unique_temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "hydra_mjai_audit_{name}_{}_{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    fn write_tar_zst_with_entries(path: &Path, entries: &[(&str, &[u8])]) {
        let tar_bytes = {
            let mut builder = tar::Builder::new(Vec::new());
            for (entry_name, entry_data) in entries {
                let mut header = tar::Header::new_gnu();
                header.set_size(entry_data.len() as u64);
                header.set_mode(0o644);
                header.set_cksum();
                builder
                    .append_data(&mut header, *entry_name, Cursor::new(*entry_data))
                    .expect("append archive entry");
            }
            builder.into_inner().expect("finish tar builder")
        };

        let file = fs::File::create(path).expect("create zstd archive");
        let mut encoder = zstd::Encoder::new(file, 0).expect("create zstd encoder");
        encoder
            .write_all(&tar_bytes)
            .expect("write tar payload into zstd stream");
        encoder.finish().expect("finish zstd stream");
    }

    fn cleanup_dir(dir: &Path) {
        if dir.exists() {
            fs::remove_dir_all(dir).expect("temp dir should be removable");
        }
    }

    fn read_inventory_records(path: &Path) -> Vec<FailureInventoryRecord> {
        fs::read_to_string(path)
            .expect("inventory file should be readable")
            .lines()
            .map(|line| {
                serde_json::from_str::<FailureInventoryRecord>(line)
                    .expect("inventory line should parse")
            })
            .collect()
    }

    #[test]
    fn parse_args_uses_defaults_and_accepts_overrides() {
        let cfg = parse_args(vec!["mjai_audit".to_string(), "/data".to_string()])
            .expect("default args should parse");
        assert_eq!(cfg.data_dir, PathBuf::from("/data"));
        assert_eq!(cfg.threads, 16);
        assert_eq!(cfg.failure_examples, 20);
        assert_eq!(cfg.failure_inventory_dir, None);

        let cfg = parse_args(vec![
            "mjai_audit".to_string(),
            "/data".to_string(),
            "--threads".to_string(),
            "8".to_string(),
            "--failure-examples".to_string(),
            "5".to_string(),
            "--failure-inventory-dir".to_string(),
            "/tmp/inventory".to_string(),
        ])
        .expect("override args should parse");
        assert_eq!(cfg.threads, 8);
        assert_eq!(cfg.failure_examples, 5);
        assert_eq!(
            cfg.failure_inventory_dir,
            Some(PathBuf::from("/tmp/inventory"))
        );
    }

    #[test]
    fn parse_args_rejects_bad_values_and_unknown_flags() {
        assert!(parse_args(vec!["mjai_audit".to_string()]).is_err());

        let zero = parse_args(vec![
            "mjai_audit".to_string(),
            "/data".to_string(),
            "--threads".to_string(),
            "0".to_string(),
        ])
        .expect_err("zero threads should fail");
        assert!(zero.contains("greater than 0"));

        let invalid = parse_args(vec![
            "mjai_audit".to_string(),
            "/data".to_string(),
            "--threads".to_string(),
            "abc".to_string(),
        ])
        .expect_err("non numeric threads should fail");
        assert!(invalid.contains("invalid --threads value"));

        let unknown = parse_args(vec![
            "mjai_audit".to_string(),
            "/data".to_string(),
            "--mystery".to_string(),
        ])
        .expect_err("unknown flag should fail");
        assert!(unknown.contains("unknown argument"));
    }

    #[test]
    fn path_classifiers_and_error_summary_match_expected_suffix_rules() {
        assert_eq!(
            usage("audit-bin"),
            "Usage: audit-bin <data-dir> [--threads N] [--failure-examples N] [--failure-inventory-dir DIR]"
        );

        assert!(is_archive_file(Path::new("dataset.tar.zst")));
        assert!(is_archive_file(Path::new("dataset.tar-0001.zst")));
        assert!(!is_archive_file(Path::new("dataset.zip")));

        assert!(is_mjai_archive_entry(Path::new("round.mjai.json")));
        assert!(is_mjai_archive_entry(Path::new("round.mjai.json.gz")));
        assert!(is_mjai_archive_entry(Path::new("round.json")));
        assert!(!is_mjai_archive_entry(Path::new("round.txt")));

        assert_eq!(summarize_error("first line\nsecond line"), "first line");
        assert_eq!(summarize_error("   \n  "), "unknown error");
    }

    #[test]
    fn collect_sources_accepts_single_files_and_directory_archives() {
        let dir = unique_temp_dir("collect_sources");
        let keep_json = dir.join("a.json");
        let keep_archive = dir.join("dataset.tar.zst");
        let keep_gz = dir.join("b.json.gz");
        let skip_txt = dir.join("note.txt");
        fs::write(&keep_json, valid_mjai_log()).expect("write json");
        fs::write(&keep_gz, b"gz").expect("write gz placeholder");
        fs::write(&skip_txt, b"note").expect("write txt");
        write_tar_zst_with_entries(&keep_archive, &[("good.json", valid_mjai_log().as_bytes())]);

        let single = collect_sources(&keep_json).expect("single file should be accepted");
        assert_eq!(single, vec![DataSource::LooseFile(keep_json.clone())]);

        let single_archive = collect_sources(&keep_archive).expect("archive should be accepted");
        assert_eq!(
            single_archive,
            vec![DataSource::Archive(keep_archive.clone())]
        );

        let listed = collect_sources(&dir).expect("directory collection should succeed");
        assert_eq!(
            listed,
            vec![
                DataSource::LooseFile(keep_json.clone()),
                DataSource::LooseFile(keep_gz.clone()),
                DataSource::Archive(keep_archive.clone()),
            ]
        );

        cleanup_dir(&dir);
    }

    #[test]
    fn collect_sources_reports_missing_directory_with_context() {
        let missing = std::env::temp_dir().join(format!(
            "hydra_missing_mjai_audit_{}_{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        let err = collect_sources(&missing).expect_err("missing dir should fail");
        assert!(err.contains("failed to scan data dir"));
        assert!(err.contains(&missing.display().to_string()));
    }

    #[test]
    fn archive_and_mjai_entry_classifiers_reject_directory_and_wrong_suffixes() {
        assert!(!is_archive_file(Path::new("dataset.tar")));
        assert!(!is_archive_file(Path::new("dataset.zst")));
        assert!(!is_archive_file(Path::new("dataset.tar.gz")));

        assert!(!is_mjai_archive_entry(Path::new("round.mjai")));
        assert!(!is_mjai_archive_entry(Path::new("round.log")));
        assert!(!is_mjai_archive_entry(Path::new("round.json.zst")));
    }

    #[test]
    fn parse_args_requires_flag_values_for_threads_and_failure_examples() {
        let threads_err = parse_args(vec![
            "mjai_audit".to_string(),
            "/data".to_string(),
            "--threads".to_string(),
        ])
        .expect_err("missing threads value should fail");
        assert!(threads_err.contains("missing value for --threads"));

        let examples_err = parse_args(vec![
            "mjai_audit".to_string(),
            "/data".to_string(),
            "--failure-examples".to_string(),
        ])
        .expect_err("missing failure-examples value should fail");
        assert!(examples_err.contains("missing value for --failure-examples"));

        let inventory_err = parse_args(vec![
            "mjai_audit".to_string(),
            "/data".to_string(),
            "--failure-inventory-dir".to_string(),
        ])
        .expect_err("missing failure inventory dir should fail");
        assert!(inventory_err.contains("missing value for --failure-inventory-dir"));
    }

    #[test]
    fn summary_helpers_cover_zero_positive_sorting_and_failure_limits() {
        assert_eq!(throughput_per_second(10, 0.0), 0.0);
        assert_eq!(throughput_per_second(12, 3.0), 4.0);

        let sorted = sort_error_buckets(HashMap::from([
            ("z bucket".to_string(), 2usize),
            ("a bucket".to_string(), 2usize),
            ("mid bucket".to_string(), 5usize),
        ]));
        assert_eq!(sorted[0], ("mid bucket".to_string(), 5));
        assert_eq!(sorted[1], ("a bucket".to_string(), 2));
        assert_eq!(sorted[2], ("z bucket".to_string(), 2));

        assert!(should_record_failure_example(0, 2));
        assert!(should_record_failure_example(1, 2));
        assert!(!should_record_failure_example(2, 2));
        assert!(!should_record_failure_example(0, 0));
    }

    #[test]
    fn report_helpers_cover_totals_rates_and_failure_rendering() {
        let totals = AuditTotals {
            loaded: 3,
            skipped: 2,
            samples: 40,
        };
        assert_eq!(total_files(&totals), 5);

        let rates = compute_audit_rates(&totals, 2.0);
        assert_eq!(rates.elapsed_secs, 2.0);
        assert_eq!(rates.files_per_sec, 2.5);
        assert_eq!(rates.samples_per_sec, 20.0);

        assert_eq!(
            audit_totals_summary_line(&totals),
            "Audit complete: loaded=3 skipped=2 samples=40 total=5"
        );
        assert_eq!(
            audit_speed_summary_line(&rates),
            "Speed: elapsed=2.00s files_per_sec=2.50 samples_per_sec=20.00"
        );

        let none = failure_report_lines(&[], &[]);
        assert_eq!(none, vec!["No failures detected.".to_string()]);

        let lines = failure_report_lines(
            &[("bucket a".to_string(), 3), ("bucket b".to_string(), 1)],
            &[("/tmp/a.json".to_string(), "boom".to_string())],
        );
        assert_eq!(lines[0], "Top failure buckets:");
        assert!(lines.iter().any(|line| line.contains("bucket a")));
        assert!(lines.iter().any(|line| line == "Failure examples:"));
        assert!(lines.iter().any(|line| line.contains("/tmp/a.json")));
    }

    #[test]
    fn exit_code_helper_distinguishes_success_from_failure() {
        assert_eq!(exit_code_for_run_result(&Ok(())), 0);
        assert_eq!(exit_code_for_run_result(&Err("boom".to_string())), 1);
    }

    #[test]
    fn collect_sources_accepts_non_mjai_single_file_inputs_too() {
        let dir = unique_temp_dir("single_file_any_suffix");
        let single = dir.join("notes.txt");
        fs::write(&single, b"hello").expect("single file fixture should write");

        let paths = collect_sources(&single).expect("single file path should be returned as-is");

        assert_eq!(paths, vec![DataSource::LooseFile(single.clone())]);

        cleanup_dir(&dir);
    }

    #[test]
    fn summarize_error_keeps_single_line_messages_intact() {
        assert_eq!(summarize_error("plain error"), "plain error");
    }

    #[test]
    fn run_succeeds_for_single_valid_file_path() {
        let dir = unique_temp_dir("single_valid_run");
        let path = dir.join("game.json");
        fs::write(&path, valid_mjai_log()).expect("valid mjai fixture should write");

        let args = vec![
            "mjai_audit".to_string(),
            path.display().to_string(),
            "--threads".to_string(),
            "1".to_string(),
            "--failure-examples".to_string(),
            "1".to_string(),
        ];

        let previous_args = std::env::args_os().collect::<Vec<_>>();
        let _ = previous_args;
        let result = {
            let parsed = parse_args(args).expect("args should parse");
            let totals = AuditTotals {
                loaded: 1,
                skipped: 0,
                samples: load_game_from_path(&path)
                    .expect("fixture should load")
                    .num_samples(),
            };
            let rates = compute_audit_rates(&totals, 1.0);
            assert_eq!(
                audit_totals_summary_line(&totals),
                format!(
                    "Audit complete: loaded=1 skipped=0 samples={} total=1",
                    totals.samples
                )
            );
            assert_eq!(parsed.data_dir, path);
            assert_eq!(parsed.threads, 1);
            assert_eq!(parsed.failure_examples, 1);
            assert_eq!(parsed.failure_inventory_dir, None);
            assert!(audit_speed_summary_line(&rates).contains("files_per_sec=1.00"));
            Ok::<(), String>(())
        };

        assert_eq!(result, Ok(()));

        cleanup_dir(&dir);
    }

    #[test]
    fn run_style_helpers_cover_failed_single_file_load_accounting() {
        let dir = unique_temp_dir("single_invalid_run");
        let path = dir.join("bad.json");
        fs::write(&path, "not-json").expect("invalid fixture should write");

        let err = match load_game_from_path(&path) {
            Ok(_) => panic!("invalid mjai fixture should fail to load"),
            Err(err) => err,
        };
        let summary = summarize_error(&err.to_string());
        let buckets = sort_error_buckets(HashMap::from([(summary.clone(), 1usize)]));
        let examples = vec![(path.display().to_string(), err.to_string())];
        let totals = AuditTotals {
            loaded: 0,
            skipped: 1,
            samples: 0,
        };

        assert_eq!(total_files(&totals), 1);
        assert_eq!(buckets[0], (summary, 1));
        let lines = failure_report_lines(&buckets, &examples);
        assert!(lines.iter().any(|line| line == "Top failure buckets:"));
        assert!(lines.iter().any(|line| line == "Failure examples:"));
        assert!(lines.iter().any(|line| line.contains("bad.json")));

        cleanup_dir(&dir);
    }

    #[test]
    fn archive_open_failure_string_includes_path() {
        let path = unique_temp_dir("missing_archive_open").join("dataset.tar.zst");

        let err = fs::File::open(&path)
            .map(|_| ())
            .map_err(|err| format!("failed to open archive {}: {err}", path.display()))
            .expect_err("missing archive should fail to open");

        assert!(err.contains("failed to open archive"));
        assert!(err.contains(path.to_string_lossy().as_ref()));
    }

    #[test]
    fn archive_decode_failure_string_includes_path() {
        let dir = unique_temp_dir("invalid_archive_decode");
        let path = dir.join("dataset.tar.zst");
        fs::write(&path, b"not a zstd archive").expect("invalid archive fixture should write");

        let file = fs::File::open(&path).expect("fixture should open");
        let mut decoder = zstd::Decoder::new(file).expect("decoder construction should succeed");
        let err = std::io::Read::read_to_end(&mut decoder, &mut Vec::new())
            .map(|_| ())
            .map_err(|err| format!("failed to decode archive {}: {err}", path.display()))
            .expect_err("invalid archive should fail when decoder is read");

        assert!(err.contains("failed to decode archive"));
        assert!(err.contains(path.to_string_lossy().as_ref()));

        cleanup_dir(&dir);
    }

    #[test]
    fn archive_mode_ignores_non_mjai_entries_and_caps_failure_examples() {
        let dir = unique_temp_dir("archive_mode_mixed_entries");
        let path = dir.join("dataset.tar.zst");

        write_tar_zst_with_entries(
            &path,
            &[
                ("good.json", valid_mjai_log().as_bytes()),
                ("bad1.json", b"not-json"),
                ("bad2.json", b"still-not-json"),
                ("notes.txt", b"hello"),
            ],
        );

        let config = AuditConfig {
            data_dir: path.clone(),
            threads: 1,
            failure_examples: 1,
            failure_inventory_dir: None,
        };
        let state = AuditSharedState::default();
        audit_archive_source(&path, &config, &state, &mut None).expect("archive audit should work");

        let totals = state.totals();
        let error_buckets = state.snapshot_error_buckets();
        let failure_examples = state.snapshot_failure_examples();
        assert_eq!(totals.loaded, 1);
        assert_eq!(totals.skipped, 2);
        assert!(totals.samples > 0);
        assert_eq!(failure_examples.len(), 1);
        assert_eq!(error_buckets.values().sum::<usize>(), 2);
        let lines = failure_report_lines(&sort_error_buckets(error_buckets), &failure_examples);
        assert!(lines.iter().any(|line| line == "Top failure buckets:"));
        assert!(lines.iter().any(|line| line == "Failure examples:"));
        assert!(
            lines
                .iter()
                .any(|line| line.contains("dataset.tar.zst/bad1.json")
                    || line.contains("dataset.tar.zst/bad2.json"))
        );

        cleanup_dir(&dir);
    }

    #[test]
    fn archive_entry_identity_includes_archive_path() {
        let archive = Path::new("/data/dataset.tar.zst");
        let entry = Path::new("nested/bad.json");

        assert_eq!(
            archive_entry_identity(archive, entry),
            "/data/dataset.tar.zst/nested/bad.json"
        );
    }

    #[test]
    fn failure_inventory_path_uses_source_name_and_hash_suffix() {
        let inventory_dir = Path::new("/tmp/inventory");
        let source = Path::new("/mnt/dev/dataset/dataset.tar.zst");
        let path = failure_inventory_path(inventory_dir, source);

        assert_eq!(path.parent(), Some(inventory_dir));
        let filename = path
            .file_name()
            .and_then(OsStr::to_str)
            .expect("utf8 filename");
        assert!(filename.starts_with("dataset.tar.zst-"));
        assert!(filename.ends_with(".jsonl"));
    }

    #[test]
    fn audit_source_persists_failure_inventory_for_loose_file_and_archive_entries() {
        let dir = unique_temp_dir("failure_inventory");
        let inventory_dir = dir.join("inventory");
        let loose = dir.join("bad.json");
        let archive = dir.join("dataset.tar.zst");
        fs::write(&loose, "not-json").expect("invalid loose fixture should write");
        write_tar_zst_with_entries(
            &archive,
            &[
                ("good.json", valid_mjai_log().as_bytes()),
                ("bad.json", b"not-json"),
            ],
        );
        let expected_loose_error = match load_game_from_path(&loose) {
            Ok(_) => panic!("invalid loose file should fail"),
            Err(err) => err.to_string(),
        };
        let expected_archive_error = {
            let file = fs::File::open(&archive).expect("archive fixture should open");
            let zstd = zstd::Decoder::new(file).expect("archive decoder should open");
            let mut tar = tar::Archive::new(zstd);
            let mut entries = tar.entries().expect("archive entries should iterate");
            let bad_entry = entries
                .find_map(|entry| {
                    let entry = entry.expect("archive entry should read");
                    let entry_path = entry.path().expect("archive path should read").into_owned();
                    (entry_path == std::path::Path::new("bad.json")).then_some(entry)
                })
                .expect("bad archive entry should exist");
            match load_game_from_stream(BufReader::new(bad_entry)) {
                Ok(_) => panic!("invalid archive entry should fail"),
                Err(err) => err.to_string(),
            }
        };

        let config = AuditConfig {
            data_dir: dir.clone(),
            threads: 2,
            failure_examples: 10,
            failure_inventory_dir: Some(inventory_dir.clone()),
        };
        let state = AuditSharedState::default();

        audit_source(&DataSource::LooseFile(loose.clone()), &config, &state)
            .expect("loose source audit should succeed");
        audit_source(&DataSource::Archive(archive.clone()), &config, &state)
            .expect("archive source audit should succeed");

        let loose_inventory = failure_inventory_path(&inventory_dir, &loose);
        let archive_inventory = failure_inventory_path(&inventory_dir, &archive);
        assert!(loose_inventory.exists());
        assert!(archive_inventory.exists());

        let loose_records = read_inventory_records(&loose_inventory);
        assert_eq!(loose_records.len(), 1);
        assert_eq!(loose_records[0].source, loose.display().to_string());
        assert_eq!(loose_records[0].identity, loose.display().to_string());
        assert_eq!(loose_records[0].error, expected_loose_error);

        let archive_records = read_inventory_records(&archive_inventory);
        assert_eq!(archive_records.len(), 1);
        assert_eq!(archive_records[0].source, archive.display().to_string());
        assert_eq!(
            archive_records[0].identity,
            format!("{}/bad.json", archive.display())
        );
        assert_eq!(archive_records[0].error, expected_archive_error);

        cleanup_dir(&dir);
    }

    #[test]
    fn audit_source_keeps_clean_sources_artifact_free() {
        let dir = unique_temp_dir("clean_source_inventory");
        let inventory_dir = dir.join("inventory");
        let loose = dir.join("good.json");
        let archive = dir.join("good-dataset.tar.zst");
        fs::write(&loose, valid_mjai_log()).expect("valid loose fixture should write");
        write_tar_zst_with_entries(&archive, &[("good.json", valid_mjai_log().as_bytes())]);

        let config = AuditConfig {
            data_dir: dir.clone(),
            threads: 2,
            failure_examples: 10,
            failure_inventory_dir: Some(inventory_dir.clone()),
        };
        let state = AuditSharedState::default();

        audit_source(&DataSource::LooseFile(loose.clone()), &config, &state)
            .expect("clean loose source audit should succeed");
        audit_source(&DataSource::Archive(archive.clone()), &config, &state)
            .expect("clean archive source audit should succeed");

        assert!(!failure_inventory_path(&inventory_dir, &loose).exists());
        assert!(!failure_inventory_path(&inventory_dir, &archive).exists());
        assert!(!inventory_dir.exists());

        cleanup_dir(&dir);
    }
}
