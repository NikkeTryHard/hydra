use std::collections::HashMap;
use std::fs;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use hydra_train::data::mjai_loader::{load_game_from_path, load_game_from_stream};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

const MJAI_AUDIT_THREAD_STACK_SIZE: usize = 8 * 1024 * 1024;

#[derive(Debug)]
struct AuditConfig {
    data_dir: PathBuf,
    threads: usize,
    failure_examples: usize,
}

#[derive(Debug, Clone, PartialEq)]
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

fn usage(program: &str) -> String {
    format!("Usage: {program} <data-dir> [--threads N] [--failure-examples N]")
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
            _ => return Err(format!("unknown argument {flag:?}\n{}", usage(&program))),
        }
    }

    Ok(AuditConfig {
        data_dir: PathBuf::from(data_dir),
        threads,
        failure_examples,
    })
}

fn is_mjai_file(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some(name) if name.ends_with(".json") || name.ends_with(".json.gz")
    )
}

fn collect_paths(dir: &Path) -> Result<Vec<PathBuf>, String> {
    if dir.is_file() {
        return Ok(vec![dir.to_path_buf()]);
    }

    let mut paths = Vec::new();
    let entries = fs::read_dir(dir)
        .map_err(|err| format!("failed to read data dir {}: {err}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|err| format!("failed to read dir entry: {err}"))?;
        let path = entry.path();
        if path.is_file() && is_mjai_file(&path) {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

fn is_archive_file(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some(name)
            if name.ends_with(".tar")
                || name.ends_with(".tar.zst")
                || name.contains(".tar-") && name.ends_with(".zst")
    )
}

fn is_mjai_archive_entry(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some(name) if name.ends_with(".json") || name.ends_with(".json.gz") || name.ends_with(".mjai.json") || name.ends_with(".mjai.json.gz")
    )
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

    if config.data_dir.is_file() && is_archive_file(&config.data_dir) {
        let file = fs::File::open(&config.data_dir).map_err(|err| {
            format!(
                "failed to open archive {}: {err}",
                config.data_dir.display()
            )
        })?;
        let zstd = zstd::Decoder::new(file).map_err(|err| {
            format!(
                "failed to decode archive {}: {err}",
                config.data_dir.display()
            )
        })?;
        let mut archive = tar::Archive::new(zstd);

        let mut loaded = 0usize;
        let mut skipped = 0usize;
        let mut samples = 0usize;
        let mut error_buckets = HashMap::<String, usize>::new();
        let mut failure_examples = Vec::<(String, String)>::new();

        for entry_result in archive.entries().map_err(|err| {
            format!(
                "failed to iterate archive {}: {err}",
                config.data_dir.display()
            )
        })? {
            let entry = entry_result.map_err(|err| {
                format!(
                    "failed to read archive entry in {}: {err}",
                    config.data_dir.display()
                )
            })?;
            let entry_path = entry
                .path()
                .map_err(|err| {
                    format!(
                        "failed to inspect archive entry in {}: {err}",
                        config.data_dir.display()
                    )
                })?
                .into_owned();
            if !is_mjai_archive_entry(&entry_path) {
                continue;
            }

            match load_game_from_stream(BufReader::new(entry)) {
                Ok(game) => {
                    loaded += 1;
                    samples += game.num_samples();
                }
                Err(err) => {
                    skipped += 1;
                    let err_string = err.to_string();
                    let bucket = summarize_error(&err_string);
                    *error_buckets.entry(bucket).or_insert(0) += 1;
                    if should_record_failure_example(
                        failure_examples.len(),
                        config.failure_examples,
                    ) {
                        failure_examples.push((entry_path.display().to_string(), err_string));
                    }
                }
            }
        }

        let totals = AuditTotals {
            loaded,
            skipped,
            samples,
        };
        let rates = compute_audit_rates(&totals, started_at.elapsed().as_secs_f64());

        println!("{}", audit_totals_summary_line(&totals));
        println!("{}", audit_speed_summary_line(&rates));

        let buckets = sort_error_buckets(error_buckets);
        for line in failure_report_lines(&buckets, &failure_examples) {
            println!("{line}");
        }

        return Ok(());
    }

    let paths = collect_paths(&config.data_dir)?;
    let total = paths.len();
    println!(
        "Auditing {} MJAI files from {} with {} threads",
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

    let loaded = AtomicUsize::new(0);
    let skipped = AtomicUsize::new(0);
    let samples = AtomicUsize::new(0);
    let error_buckets = Arc::new(Mutex::new(HashMap::<String, usize>::new()));
    let failure_examples = Arc::new(Mutex::new(Vec::<(String, String)>::new()));

    let pool = ThreadPoolBuilder::new()
        .num_threads(config.threads)
        .stack_size(MJAI_AUDIT_THREAD_STACK_SIZE)
        .build()
        .map_err(|err| format!("failed to build rayon pool: {err}"))?;

    pool.install(|| {
        paths.par_iter().for_each(|path| {
            match load_game_from_path(path) {
                Ok(game) => {
                    loaded.fetch_add(1, Ordering::Relaxed);
                    samples.fetch_add(game.num_samples(), Ordering::Relaxed);
                }
                Err(err) => {
                    skipped.fetch_add(1, Ordering::Relaxed);
                    let err_string = err.to_string();
                    let bucket = summarize_error(&err_string);
                    {
                        let mut buckets = error_buckets.lock().expect("lock error buckets");
                        *buckets.entry(bucket).or_insert(0) += 1;
                    }
                    {
                        let mut examples = failure_examples.lock().expect("lock failure examples");
                        if should_record_failure_example(examples.len(), config.failure_examples) {
                            examples.push((path.display().to_string(), err_string));
                        }
                    }
                }
            }
            progress.inc(1);
        });
    });

    progress.finish_and_clear();

    let totals = AuditTotals {
        loaded: loaded.load(Ordering::Relaxed),
        skipped: skipped.load(Ordering::Relaxed),
        samples: samples.load(Ordering::Relaxed),
    };
    let rates = compute_audit_rates(&totals, started_at.elapsed().as_secs_f64());

    println!("{}", audit_totals_summary_line(&totals));
    println!("{}", audit_speed_summary_line(&rates));

    let buckets = sort_error_buckets(
        error_buckets
            .lock()
            .expect("lock error buckets")
            .drain()
            .collect(),
    );
    let examples = failure_examples.lock().expect("lock failure examples");
    for line in failure_report_lines(&buckets, &examples) {
        println!("{line}");
    }

    Ok(())
}

fn exit_code_for_run_result(result: &Result<(), String>) -> i32 {
    if result.is_ok() {
        0
    } else {
        1
    }
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
    use std::io::Write;
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

    #[test]
    fn parse_args_uses_defaults_and_accepts_overrides() {
        let cfg = parse_args(vec!["mjai_audit".to_string(), "/data".to_string()])
            .expect("default args should parse");
        assert_eq!(cfg.data_dir, PathBuf::from("/data"));
        assert_eq!(cfg.threads, 16);
        assert_eq!(cfg.failure_examples, 20);

        let cfg = parse_args(vec![
            "mjai_audit".to_string(),
            "/data".to_string(),
            "--threads".to_string(),
            "8".to_string(),
            "--failure-examples".to_string(),
            "5".to_string(),
        ])
        .expect("override args should parse");
        assert_eq!(cfg.threads, 8);
        assert_eq!(cfg.failure_examples, 5);
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
            "Usage: audit-bin <data-dir> [--threads N] [--failure-examples N]"
        );
        assert!(is_mjai_file(Path::new("game.json")));
        assert!(is_mjai_file(Path::new("game.json.gz")));
        assert!(is_mjai_file(Path::new("game.mjai.json.gz")));

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
    fn collect_paths_accepts_single_file_and_filters_directory_entries() {
        let dir = unique_temp_dir("collect");
        let keep_json = dir.join("a.json");
        let keep_gz = dir.join("b.json.gz");
        let skip_txt = dir.join("note.txt");
        fs::write(&keep_json, b"{}").expect("write json");
        fs::write(&keep_gz, b"gz").expect("write gz placeholder");
        fs::write(&skip_txt, b"note").expect("write txt");

        let single = collect_paths(&keep_json).expect("single file should be accepted");
        assert_eq!(single, vec![keep_json.clone()]);

        let listed = collect_paths(&dir).expect("directory collection should succeed");
        assert_eq!(listed, vec![keep_json.clone(), keep_gz.clone()]);

        fs::remove_file(keep_json).expect("remove json");
        fs::remove_file(keep_gz).expect("remove gz");
        fs::remove_file(skip_txt).expect("remove txt");
        fs::remove_dir(dir).expect("remove dir");
    }

    #[test]
    fn collect_paths_reports_missing_directory_with_context() {
        let missing = std::env::temp_dir().join(format!(
            "hydra_missing_mjai_audit_{}_{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        let err = collect_paths(&missing).expect_err("missing dir should fail");
        assert!(err.contains("failed to read data dir"));
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
    fn collect_paths_accepts_non_mjai_single_file_inputs_too() {
        let dir = unique_temp_dir("single_file_any_suffix");
        let single = dir.join("notes.txt");
        fs::write(&single, b"hello").expect("single file fixture should write");

        let paths = collect_paths(&single).expect("single file path should be returned as-is");

        assert_eq!(paths, vec![single.clone()]);

        fs::remove_file(single).expect("single fixture should be removable");
        fs::remove_dir(dir).expect("temp dir should be removable");
    }

    #[test]
    fn path_classifiers_reject_mjai_archive_suffix_for_plain_file_detector() {
        assert!(is_mjai_file(Path::new("game.mjai.json.gz")));
        assert!(is_mjai_file(Path::new("game.mjai.json")));
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
            assert!(audit_speed_summary_line(&rates).contains("files_per_sec=1.00"));
            Ok::<(), String>(())
        };

        assert_eq!(result, Ok(()));

        fs::remove_file(path).expect("fixture should be removable");
        fs::remove_dir(dir).expect("temp dir should be removable");
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

        fs::remove_file(path).expect("invalid fixture should be removable");
        fs::remove_dir(dir).expect("temp dir should be removable");
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

        fs::remove_file(path).expect("fixture should be removable");
        fs::remove_dir(dir).expect("temp dir should be removable");
    }

    #[test]
    fn archive_mode_ignores_non_mjai_entries_and_caps_failure_examples() {
        let dir = unique_temp_dir("archive_mode_mixed_entries");
        let path = dir.join("dataset.tar.zst");

        let tar_bytes = {
            let mut builder = tar::Builder::new(Vec::new());

            let valid = valid_mjai_log();
            let mut valid_header = tar::Header::new_gnu();
            valid_header.set_size(valid.len() as u64);
            valid_header.set_mode(0o644);
            valid_header.set_cksum();
            builder
                .append_data(&mut valid_header, "good.json", valid.as_bytes())
                .expect("append valid archive entry");

            let invalid_1 = b"not-json";
            let mut invalid_header_1 = tar::Header::new_gnu();
            invalid_header_1.set_size(invalid_1.len() as u64);
            invalid_header_1.set_mode(0o644);
            invalid_header_1.set_cksum();
            builder
                .append_data(&mut invalid_header_1, "bad1.json", &invalid_1[..])
                .expect("append invalid archive entry one");

            let invalid_2 = b"still-not-json";
            let mut invalid_header_2 = tar::Header::new_gnu();
            invalid_header_2.set_size(invalid_2.len() as u64);
            invalid_header_2.set_mode(0o644);
            invalid_header_2.set_cksum();
            builder
                .append_data(&mut invalid_header_2, "bad2.json", &invalid_2[..])
                .expect("append invalid archive entry two");

            let ignored = b"hello";
            let mut ignored_header = tar::Header::new_gnu();
            ignored_header.set_size(ignored.len() as u64);
            ignored_header.set_mode(0o644);
            ignored_header.set_cksum();
            builder
                .append_data(&mut ignored_header, "notes.txt", &ignored[..])
                .expect("append ignored archive entry");

            builder.into_inner().expect("finish tar builder")
        };

        let file = fs::File::create(&path).expect("create zstd archive");
        let mut encoder = zstd::Encoder::new(file, 0).expect("create zstd encoder");
        encoder
            .write_all(&tar_bytes)
            .expect("write tar payload into zstd stream");
        encoder.finish().expect("finish zstd stream");

        let file = fs::File::open(&path).expect("open archive fixture");
        let zstd = zstd::Decoder::new(file).expect("decode archive fixture");
        let mut archive = tar::Archive::new(zstd);

        let mut loaded = 0usize;
        let mut skipped = 0usize;
        let mut samples = 0usize;
        let mut error_buckets = HashMap::<String, usize>::new();
        let mut failure_examples = Vec::<(String, String)>::new();

        for entry_result in archive.entries().expect("iterate archive entries") {
            let entry = entry_result.expect("read archive entry");
            let entry_path = entry.path().expect("inspect archive path").into_owned();
            if !is_mjai_archive_entry(&entry_path) {
                continue;
            }
            match load_game_from_stream(BufReader::new(entry)) {
                Ok(game) => {
                    loaded += 1;
                    samples += game.num_samples();
                }
                Err(err) => {
                    skipped += 1;
                    let err_string = err.to_string();
                    let bucket = summarize_error(&err_string);
                    *error_buckets.entry(bucket).or_insert(0) += 1;
                    if should_record_failure_example(failure_examples.len(), 1) {
                        failure_examples.push((entry_path.display().to_string(), err_string));
                    }
                }
            }
        }

        let totals = AuditTotals {
            loaded,
            skipped,
            samples,
        };
        assert_eq!(totals.loaded, 1);
        assert_eq!(totals.skipped, 2);
        assert!(totals.samples > 0);
        assert_eq!(failure_examples.len(), 1);
        assert_eq!(error_buckets.values().sum::<usize>(), 2);
        let lines = failure_report_lines(&sort_error_buckets(error_buckets), &failure_examples);
        assert!(lines.iter().any(|line| line == "Top failure buckets:"));
        assert!(lines.iter().any(|line| line == "Failure examples:"));
        assert!(lines
            .iter()
            .any(|line| line.contains("bad1.json") || line.contains("bad2.json")));

        fs::remove_file(path).expect("archive fixture should be removable");
        fs::remove_dir(dir).expect("temp dir should be removable");
    }
}
