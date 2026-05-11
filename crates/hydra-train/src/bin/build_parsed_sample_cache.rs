use std::fs;
use std::path::{Path, PathBuf};

use hydra_replay_loader::archive_helpers::{
    compact_error_message, compact_identity, is_tar_zst_file,
};
use hydra_replay_loader::mjai_loader::load_game_from_path;
use hydra_sample_cache::{
    ParsedSampleCacheGame, parsed_sample_cache_file_name, write_parsed_sample_cache,
};
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;

#[derive(Debug)]
struct Cli {
    input: PathBuf,
    output_dir: PathBuf,
}

fn usage(program: &str) -> String {
    format!("Usage: {program} --input <loose-mjai-file|dir> --output-dir <dir>")
}

fn parse_args<I>(program: &str, args: I) -> Result<Cli, String>
where
    I: IntoIterator<Item = String>,
{
    let mut args = args.into_iter();
    let _ = args.next();

    let mut input = None;
    let mut output_dir = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input" => input = args.next().map(PathBuf::from),
            "--output-dir" => output_dir = args.next().map(PathBuf::from),
            _ => return Err(usage(program)),
        }
    }

    Ok(Cli {
        input: input.ok_or_else(|| usage(program))?,
        output_dir: output_dir.ok_or_else(|| usage(program))?,
    })
}

fn is_mjai_file(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some(name) if name.ends_with(".json") || name.ends_with(".json.gz")
    )
}

fn is_tar_file(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some(name) if name.ends_with(".tar")
    )
}

fn original_identity_for_loose_file(path: &Path) -> Result<String, String> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            format!(
                "loose file path does not have a recognizable filename: {}",
                path.display()
            )
        })?;
    Ok(path
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .map(|parent| format!("{parent}/{file_name}"))
        .unwrap_or_else(|| file_name.to_owned()))
}

fn collect_loose_mjai_files(input: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    if !input.is_dir() && (is_tar_file(input) || is_tar_zst_file(input)) {
        return Err(format!(
            "archive input not supported for parsed-sample cache builder yet: {}",
            input.display()
        ));
    }

    if input.is_file() {
        if !is_mjai_file(input) {
            return Err(format!(
                "expected loose MJAI file ending in .json or .json.gz, got {}",
                input.display()
            ));
        }
        out.push(input.to_path_buf());
        return Ok(());
    }

    let entries = fs::read_dir(input)
        .map_err(|err| format!("failed to read input directory {}: {err}", input.display()))?;
    for entry in entries {
        let entry = entry.map_err(|err| {
            format!(
                "failed to read entry under input directory {}: {err}",
                input.display()
            )
        })?;
        let file_type = entry.file_type().map_err(|err| {
            format!(
                "failed to read file type for {}: {err}",
                entry.path().display()
            )
        })?;
        let path = entry.path();
        if file_type.is_dir() {
            collect_loose_mjai_files(&path, out)?;
        } else if file_type.is_file() {
            if is_tar_file(&path) || is_tar_zst_file(&path) {
                return Err(format!(
                    "archive input not supported for parsed-sample cache builder yet: {}",
                    path.display()
                ));
            }
            if is_mjai_file(&path) {
                out.push(path);
            }
        }
    }
    Ok(())
}

fn build_one_cache(path: &Path, output_dir: &Path) -> Result<PathBuf, String> {
    let game = load_game_from_path(path)
        .map_err(|err| format!("failed to load loose MJAI {}: {err}", path.display()))?;
    let identity = original_identity_for_loose_file(path)?;
    let cache_game = ParsedSampleCacheGame {
        samples: game.samples,
        final_scores: game.final_scores,
    };
    let file_name = parsed_sample_cache_file_name(path).map_err(|err| {
        format!(
            "failed to derive cache filename for {}: {err}",
            path.display()
        )
    })?;
    let output_path = output_dir.join(file_name);
    write_parsed_sample_cache(&output_path, path, &identity, &cache_game).map_err(|err| {
        format!(
            "failed to write parsed-sample cache {}: {err}",
            output_path.display()
        )
    })?;
    Ok(output_path)
}

fn run() -> Result<(), String> {
    let cli = parse_args("build_parsed_sample_cache", std::env::args())?;
    fs::create_dir_all(&cli.output_dir).map_err(|err| {
        format!(
            "failed to create output directory {}: {err}",
            cli.output_dir.display()
        )
    })?;

    let mut inputs = Vec::new();
    collect_loose_mjai_files(&cli.input, &mut inputs)?;
    inputs.sort();

    if inputs.is_empty() {
        return Err(format!(
            "no loose MJAI files found under {}",
            cli.input.display()
        ));
    }

    let pool = ThreadPoolBuilder::new()
        .build()
        .map_err(|err| format!("failed to build parsed-sample cache thread pool: {err}"))?;

    let results = pool.install(|| {
        inputs
            .par_iter()
            .map(|path| (path.clone(), build_one_cache(path, &cli.output_dir)))
            .collect::<Vec<_>>()
    });

    let mut written = 0usize;
    let mut skipped = 0usize;
    for (path, result) in results {
        match result {
            Ok(output) => {
                println!("cached {} -> {}", path.display(), output.display());
                written += 1;
            }
            Err(err) => {
                skipped += 1;
                let display_path = path.display().to_string();
                println!(
                    "Skipping {}: {}",
                    compact_identity(&display_path),
                    compact_error_message(&err)
                );
            }
        }
    }

    println!(
        "Wrote {written} parsed-sample cache file(s) to {} (skipped={skipped})",
        cli.output_dir.display()
    );
    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_args_accepts_required_flags() {
        let cli = parse_args(
            "build_parsed_sample_cache",
            vec![
                "build_parsed_sample_cache".to_string(),
                "--input".to_string(),
                "replays".to_string(),
                "--output-dir".to_string(),
                "cache".to_string(),
            ],
        )
        .expect("args should parse");

        assert_eq!(cli.input, PathBuf::from("replays"));
        assert_eq!(cli.output_dir, PathBuf::from("cache"));
    }

    #[test]
    fn collect_loose_mjai_files_rejects_archive_input() {
        let err = collect_loose_mjai_files(Path::new("/data/replays.tar.zst"), &mut Vec::new())
            .expect_err("archive input should fail clearly");
        assert!(err.contains("archive input not supported"));
    }
}
