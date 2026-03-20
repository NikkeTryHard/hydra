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
use std::sync::{mpsc, Mutex};
use std::time::Instant;

use flate2::write::GzEncoder;
use flate2::Compression;
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
                s.spawn(move || loop {
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

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::read::GzDecoder;
    use std::io::Cursor;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "hydra_recompress_{}_{}",
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
    fn is_mjai_entry_matches_expected_suffixes() {
        assert!(is_mjai_entry("game.json"));
        assert!(is_mjai_entry("round.mjai.json"));
        assert!(!is_mjai_entry("game.json.gz"));
        assert!(!is_mjai_entry("round.mjai.json.gz"));
        assert!(!is_mjai_entry("notes.txt"));
    }

    #[test]
    fn compress_entry_writes_gzip_that_roundtrips_original_data() {
        let output_dir = unique_temp_dir();
        let entry = RawEntry {
            name: "sample.json.gz".to_string(),
            data: br#"{"hello":"world"}"#.to_vec(),
        };

        let written = compress_entry(&entry, &output_dir).expect("compress entry");
        assert!(written > 0);

        let out_path = output_dir.join(&entry.name);
        let compressed = fs::read(&out_path).expect("read compressed file");
        let mut decoded = Vec::new();
        GzDecoder::new(&compressed[..])
            .read_to_end(&mut decoded)
            .expect("decode gzip output");
        assert_eq!(decoded, entry.data);

        fs::remove_file(out_path).expect("remove gzip file");
        fs::remove_dir(output_dir).expect("remove temp dir");
    }

    #[test]
    fn compress_entry_handles_empty_payloads() {
        let output_dir = unique_temp_dir();
        let entry = RawEntry {
            name: "empty.json.gz".to_string(),
            data: Vec::new(),
        };

        let out_len = compress_entry(&entry, &output_dir).expect("compress empty payload");
        assert!(out_len > 0);

        let out_path = output_dir.join(&entry.name);
        let compressed = fs::read(&out_path).expect("read empty gzip file");
        let mut decoded = Vec::new();
        GzDecoder::new(&compressed[..])
            .read_to_end(&mut decoded)
            .expect("decode empty gzip output");
        assert!(decoded.is_empty());

        fs::remove_file(out_path).expect("remove gzip file");
        fs::remove_dir(output_dir).expect("remove temp dir");
    }

    #[test]
    fn process_archive_skips_non_mjai_entries_and_writes_only_matching_files() {
        let archive_dir = unique_temp_dir();
        let output_dir = unique_temp_dir();
        let archive_path = archive_dir.join("dataset.tar.zst");

        let tar_file = File::create(&archive_path).expect("create archive file");
        let zst = zstd::Encoder::new(tar_file, 1).expect("create zstd encoder");
        let mut builder = tar::Builder::new(zst);

        let mjai = br#"{"type":"start_game"}"#;
        let txt = b"ignore me";

        let mut header = tar::Header::new_gnu();
        header.set_size(mjai.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        builder
            .append_data(&mut header, "round.json", Cursor::new(mjai))
            .expect("append json entry");

        let mut header = tar::Header::new_gnu();
        header.set_size(txt.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        builder
            .append_data(&mut header, "note.txt", Cursor::new(txt))
            .expect("append txt entry");

        let zst = builder.into_inner().expect("finish tar builder");
        zst.finish().expect("finish zstd stream");

        let multi = MultiProgress::new();
        let total_files = AtomicU64::new(0);
        let total_bytes = AtomicU64::new(0);
        process_archive(
            &archive_path,
            &output_dir,
            &multi,
            &total_files,
            &total_bytes,
        )
        .expect("process archive");

        assert_eq!(total_files.load(Ordering::Relaxed), 1);
        assert!(output_dir.join("round.json.gz").exists());
        assert!(!output_dir.join("note.txt.gz").exists());
        assert!(total_bytes.load(Ordering::Relaxed) > 0);

        fs::remove_file(output_dir.join("round.json.gz")).expect("remove output gzip");
        fs::remove_dir(output_dir).expect("remove output dir");
        fs::remove_file(archive_path).expect("remove archive file");
        fs::remove_dir(archive_dir).expect("remove archive dir");
    }

    #[test]
    fn process_archive_gracefully_handles_missing_archive_without_outputs() {
        let output_dir = unique_temp_dir();
        let archive_path = output_dir.join("missing.tar.zst");
        let multi = MultiProgress::new();
        let total_files = AtomicU64::new(0);
        let total_bytes = AtomicU64::new(0);

        process_archive(
            &archive_path,
            &output_dir,
            &multi,
            &total_files,
            &total_bytes,
        )
        .expect("missing archive should be reported but not abort the wrapper");
        assert_eq!(total_files.load(Ordering::Relaxed), 0);
        assert_eq!(total_bytes.load(Ordering::Relaxed), 0);

        fs::remove_dir(output_dir).expect("remove output dir");
    }
}
