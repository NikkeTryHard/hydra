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
