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
        Some(name) if name.ends_with(".tar.zst") || name.contains(".tar-") && name.ends_with(".zst")
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
                    if failure_examples.len() < config.failure_examples {
                        failure_examples.push((entry_path.display().to_string(), err_string));
                    }
                }
            }
        }

        let elapsed_secs = started_at.elapsed().as_secs_f64();
        let total = loaded + skipped;
        let files_per_sec = if elapsed_secs > 0.0 {
            total as f64 / elapsed_secs
        } else {
            0.0
        };
        let samples_per_sec = if elapsed_secs > 0.0 {
            samples as f64 / elapsed_secs
        } else {
            0.0
        };

        println!(
            "Audit complete: loaded={} skipped={} samples={} total={}",
            loaded, skipped, samples, total
        );
        println!(
            "Speed: elapsed={:.2}s files_per_sec={:.2} samples_per_sec={:.2}",
            elapsed_secs, files_per_sec, samples_per_sec
        );

        let mut buckets = error_buckets.into_iter().collect::<Vec<_>>();
        buckets.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        if buckets.is_empty() {
            println!("No failures detected.");
        } else {
            println!("Top failure buckets:");
            for (bucket, count) in buckets.into_iter().take(20) {
                println!("  {count:>6}  {bucket}");
            }
            if !failure_examples.is_empty() {
                println!("Failure examples:");
                for (path, err) in failure_examples {
                    println!("---\n{path}\n{err}");
                }
            }
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
                        if examples.len() < config.failure_examples {
                            examples.push((path.display().to_string(), err_string));
                        }
                    }
                }
            }
            progress.inc(1);
        });
    });

    progress.finish_and_clear();

    let loaded = loaded.load(Ordering::Relaxed);
    let skipped = skipped.load(Ordering::Relaxed);
    let samples = samples.load(Ordering::Relaxed);
    let elapsed = started_at.elapsed();
    let elapsed_secs = elapsed.as_secs_f64();
    let files_per_sec = if elapsed_secs > 0.0 {
        total as f64 / elapsed_secs
    } else {
        0.0
    };
    let samples_per_sec = if elapsed_secs > 0.0 {
        samples as f64 / elapsed_secs
    } else {
        0.0
    };

    println!(
        "Audit complete: loaded={} skipped={} samples={} total={}",
        loaded, skipped, samples, total
    );
    println!(
        "Speed: elapsed={:.2}s files_per_sec={:.2} samples_per_sec={:.2}",
        elapsed_secs, files_per_sec, samples_per_sec
    );

    let mut buckets = error_buckets
        .lock()
        .expect("lock error buckets")
        .drain()
        .collect::<Vec<_>>();
    buckets.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    if buckets.is_empty() {
        println!("No failures detected.");
    } else {
        println!("Top failure buckets:");
        for (bucket, count) in buckets.into_iter().take(20) {
            println!("  {count:>6}  {bucket}");
        }

        let examples = failure_examples.lock().expect("lock failure examples");
        if !examples.is_empty() {
            println!("Failure examples:");
            for (path, err) in examples.iter() {
                println!("---\n{path}\n{err}");
            }
        }
    }

    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}
