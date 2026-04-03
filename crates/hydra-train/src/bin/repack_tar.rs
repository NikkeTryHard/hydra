use std::fs::{self, File};
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use flate2::Compression;
use flate2::write::GzEncoder;
use indicatif::{HumanBytes, ProgressBar, ProgressStyle};

const DEFAULT_MIN_FREE_GB: u64 = 180;

#[derive(Debug, Clone)]
struct Config {
    input_dir: PathBuf,
    output_dir: PathBuf,
    min_free_gb: u64,
    max_output_gb: Option<u64>,
}

fn parse_args() -> Result<Config, String> {
    let mut args = std::env::args().skip(1);
    let input_dir = args
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| "missing <input_dir>".to_string())?;
    let output_dir = args
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| "missing <output_dir>".to_string())?;

    let mut min_free_gb = DEFAULT_MIN_FREE_GB;
    let mut max_output_gb = None;

    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--min-free-gb" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--min-free-gb requires a value".to_string())?;
                min_free_gb = value
                    .parse::<u64>()
                    .map_err(|err| format!("invalid --min-free-gb value `{value}`: {err}"))?;
            }
            "--max-output-gb" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--max-output-gb requires a value".to_string())?;
                max_output_gb =
                    Some(value.parse::<u64>().map_err(|err| {
                        format!("invalid --max-output-gb value `{value}`: {err}")
                    })?);
            }
            other => {
                return Err(format!("unexpected argument: {other}"));
            }
        }
    }

    Ok(Config {
        input_dir,
        output_dir,
        min_free_gb,
        max_output_gb,
    })
}

fn is_archive_file(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some(name) if name.ends_with(".tar.zst")
    )
}

fn free_bytes(path: &Path) -> io::Result<u64> {
    let output = std::process::Command::new("df")
        .arg("-B1")
        .arg(path)
        .output()?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "df failed for {}",
            path.display()
        )));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout
        .lines()
        .nth(1)
        .ok_or_else(|| io::Error::other("unexpected df output"))?;
    let available = line
        .split_whitespace()
        .nth(3)
        .ok_or_else(|| io::Error::other("missing available column in df output"))?;
    available.parse::<u64>().map_err(|err| {
        io::Error::other(format!(
            "failed to parse available bytes `{available}`: {err}"
        ))
    })
}

fn gzip_entry_bytes<R: Read>(entry: &mut tar::Entry<'_, R>) -> io::Result<Vec<u8>> {
    let mut raw = Vec::with_capacity(entry.size() as usize);
    entry.read_to_end(&mut raw)?;
    let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
    encoder.write_all(&raw)?;
    encoder.finish().map_err(io::Error::other)
}

fn output_tar_name(input: &Path) -> io::Result<String> {
    let name = input
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "invalid archive name"))?;
    let stem = name
        .strip_suffix(".tar.zst")
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "expected .tar.zst suffix"))?;
    Ok(format!("{stem}.tar"))
}

fn repack_archive(
    input_path: &Path,
    output_path: &Path,
    min_free_bytes: u64,
    max_output_bytes: Option<u64>,
    output_bytes_so_far: u64,
) -> io::Result<u64> {
    let input = File::open(input_path)?;
    let decoder = zstd::Decoder::new(input)?;
    let mut archive = tar::Archive::new(decoder);
    let temp_path = output_path.with_extension("tar.partial");
    let output = File::create(&temp_path)?;
    let writer = BufWriter::new(output);
    let mut builder = tar::Builder::new(writer);

    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::with_template("{spinner:.green} {msg}").expect("valid spinner template"),
    );
    pb.enable_steady_tick(std::time::Duration::from_millis(120));
    pb.set_message(format!("repacking {}", input_path.display()));

    let mut entry_count = 0u64;
    let mut written_estimate = 0u64;
    for entry_result in archive.entries()? {
        let mut entry = entry_result?;
        let path = entry.path()?.into_owned();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let is_mjai = name.ends_with(".json") || name.ends_with(".mjai.json");
        if !is_mjai {
            continue;
        }

        let gz_bytes = gzip_entry_bytes(&mut entry)?;
        let free_before_write = free_bytes(output_path.parent().unwrap_or_else(|| Path::new(".")))?;
        if free_before_write <= min_free_bytes {
            return Err(io::Error::other(format!(
                "refusing to continue: free space {} is at/below reserve {}",
                HumanBytes(free_before_write),
                HumanBytes(min_free_bytes)
            )));
        }
        if let Some(limit) = max_output_bytes
            && output_bytes_so_far
                .saturating_add(written_estimate)
                .saturating_add(gz_bytes.len() as u64)
                > limit
        {
            return Err(io::Error::other(format!(
                "refusing to continue: projected output would exceed cap {}",
                HumanBytes(limit)
            )));
        }

        let mut header = tar::Header::new_gnu();
        header.set_size(gz_bytes.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        let gz_name = format!("{name}.gz");
        builder.append_data(&mut header, gz_name, gz_bytes.as_slice())?;
        written_estimate = written_estimate.saturating_add(gz_bytes.len() as u64 + 512);
        entry_count += 1;
        if entry_count.is_multiple_of(1000) {
            pb.set_message(format!(
                "repacking {} ({entry_count} entries, ~{})",
                input_path.display(),
                HumanBytes(written_estimate)
            ));
        }
    }

    builder.finish()?;
    let mut writer = builder.into_inner()?;
    writer.flush()?;
    drop(writer);
    fs::rename(&temp_path, output_path)?;
    let final_size = fs::metadata(output_path)?.len();
    pb.finish_with_message(format!(
        "repacked {} -> {} entries, {}",
        input_path.display(),
        entry_count,
        HumanBytes(final_size)
    ));
    Ok(final_size)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args().map_err(io::Error::other)?;
    fs::create_dir_all(&config.output_dir)?;

    let min_free_bytes = config.min_free_gb.saturating_mul(1024_u64.pow(3));
    let max_output_bytes = config
        .max_output_gb
        .map(|gb| gb.saturating_mul(1024_u64.pow(3)));

    let mut archives = Vec::new();
    for entry in fs::read_dir(&config.input_dir)? {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_file() && is_archive_file(&path) {
            archives.push(path);
        }
    }
    archives.sort();
    if archives.is_empty() {
        return Err(io::Error::new(io::ErrorKind::NotFound, "no .tar.zst archives found").into());
    }

    let start = Instant::now();
    let mut total_output = 0u64;
    for archive in archives {
        let output_name = output_tar_name(&archive)?;
        let output_path = config.output_dir.join(output_name);
        if output_path.exists() {
            let existing = fs::metadata(&output_path)?.len();
            println!(
                "skipping {} -> existing {}",
                output_path.display(),
                HumanBytes(existing)
            );
            total_output = total_output.saturating_add(existing);
            continue;
        }
        let size = repack_archive(
            &archive,
            &output_path,
            min_free_bytes,
            max_output_bytes,
            total_output,
        )?;
        total_output = total_output.saturating_add(size);
    }

    println!(
        "done: wrote {} in {:.1}s to {}",
        HumanBytes(total_output),
        start.elapsed().as_secs_f64(),
        config.output_dir.display()
    );
    Ok(())
}
