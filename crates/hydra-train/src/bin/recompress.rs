//! Recompress `.tar.zst` archives of MJAI replay files into individual `.json.gz` files.
//!
//! Streams each archive entry through a bounded channel to rayon workers,
//! keeping memory usage constant regardless of archive size.
//!
//! Usage:
//!     recompress <output_dir> <archive1.tar.zst> [archive2.tar.zst ...]

use std::fs::{self, File};
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, mpsc};
use std::time::Instant;

use flate2::Compression;
use flate2::write::GzEncoder;
use indicatif::{HumanBytes, MultiProgress, ProgressBar, ProgressStyle};

const CHANNEL_BOUND: usize = 64;

struct RawEntry {
    name: String,
    data: Vec<u8>,
}

fn is_mjai_entry(name: &str) -> bool {
    name.ends_with(".json") || name.ends_with(".mjai.json")
}

fn compress_entry(entry: &RawEntry, output_dir: &Path) -> io::Result<u64> {
    let out_path = output_dir.join(&entry.name);
    let file = File::create(&out_path)?;
    let buf = BufWriter::new(file);
    let mut encoder = GzEncoder::new(buf, Compression::fast());
    encoder.write_all(&entry.data)?;
    encoder.finish()?.flush()?;
    let meta = fs::metadata(&out_path)?;
    Ok(meta.len())
}

fn process_archive(
    archive_path: &Path,
    output_dir: &Path,
    multi: &MultiProgress,
    total_files: &AtomicU64,
    total_bytes_out: &AtomicU64,
) -> io::Result<()> {
    let archive_name = archive_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("?");

    let pb = multi.add(ProgressBar::new_spinner());
    pb.set_style(
        ProgressStyle::with_template("{prefix:.bold.cyan} {pos} files ({msg})")
            .expect("valid template"),
    );
    pb.set_prefix(archive_name.to_string());
    pb.set_message("starting...");

    let (tx, rx) = mpsc::sync_channel::<RawEntry>(CHANNEL_BOUND);
    let rx = Mutex::new(rx);

    let pb_writer = pb.clone();
    let output_dir_owned = output_dir.to_path_buf();
    let archive_bytes = AtomicU64::new(0);
    let archive_files = AtomicU64::new(0);

    let archive_bytes_ref = &archive_bytes;
    let archive_files_ref = &archive_files;

    std::thread::scope(|s| {
        // Producer: single thread reads tar entries sequentially
        let producer = s.spawn(move || -> io::Result<()> {
            let file = File::open(archive_path)?;
            let zst_reader = zstd::Decoder::new(file)?;
            let mut archive = tar::Archive::new(zst_reader);

            for entry_result in archive.entries()? {
                let mut entry = entry_result?;
                let entry_path = entry.path()?.into_owned();
                let file_name = match entry_path.file_name().and_then(|n| n.to_str()) {
                    Some(name) if is_mjai_entry(name) => name.to_owned(),
                    _ => continue,
                };

                let size = entry.size();
                let mut data = Vec::with_capacity(size as usize);
                entry.read_to_end(&mut data)?;

                let raw = RawEntry {
                    name: format!("{file_name}.gz"),
                    data,
                };

                if tx.send(raw).is_err() {
                    break;
                }
            }
            Ok(())
        });

        // Consumers: rayon workers drain the channel and compress
        let num_workers = rayon::current_num_threads().max(1);
        let workers: Vec<_> = (0..num_workers)
            .map(|_| {
                let rx = &rx;
                let output_dir = &output_dir_owned;
                let pb = &pb_writer;
                s.spawn(move || {
                    loop {
                        let entry = {
                            let guard = rx.lock().expect("lock rx");
                            guard.recv()
                        };
                        let Ok(entry) = entry else { break };
                        match compress_entry(&entry, output_dir) {
                            Ok(bytes) => {
                                archive_bytes_ref.fetch_add(bytes, Ordering::Relaxed);
                            }
                            Err(err) => {
                                eprintln!("  error: {}: {err}", entry.name);
                            }
                        }
                        archive_files_ref.fetch_add(1, Ordering::Relaxed);
                        pb.inc(1);
                    }
                })
            })
            .collect();

        if let Err(err) = producer.join().expect("producer thread panicked") {
            eprintln!("  read error in {archive_name}: {err}");
        }
        for w in workers {
            w.join().expect("worker thread panicked");
        }
    });

    let files = archive_files.load(Ordering::Relaxed);
    let bytes = archive_bytes.load(Ordering::Relaxed);
    pb.set_message(format!("{} done", HumanBytes(bytes)));
    pb.finish();

    total_files.fetch_add(files, Ordering::Relaxed);
    total_bytes_out.fetch_add(bytes, Ordering::Relaxed);

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: recompress <output_dir> <archive1.tar.zst> [archive2.tar.zst ...]");
        std::process::exit(1);
    }

    let output_dir = PathBuf::from(&args[1]);
    let archive_paths: Vec<PathBuf> = args[2..].iter().map(PathBuf::from).collect();

    fs::create_dir_all(&output_dir)?;

    let multi = MultiProgress::new();
    let total_start = Instant::now();
    let total_files = AtomicU64::new(0);
    let total_bytes_out = AtomicU64::new(0);

    for archive_path in &archive_paths {
        process_archive(
            archive_path,
            &output_dir,
            &multi,
            &total_files,
            &total_bytes_out,
        )?;
    }

    let elapsed = total_start.elapsed();
    let files = total_files.load(Ordering::Relaxed);
    let bytes = total_bytes_out.load(Ordering::Relaxed);
    println!(
        "\nDone: {} files, {} written in {:.1}s",
        files,
        HumanBytes(bytes),
        elapsed.as_secs_f64()
    );

    Ok(())
}
