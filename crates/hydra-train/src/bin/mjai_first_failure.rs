use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

use hydra_train::data::archive_helpers::is_mjai_archive_entry;
use hydra_train::data::mjai_loader::load_game_from_stream;

fn usage(program: &str) -> String {
    format!("Usage: {program} <archive.tar.zst>")
}

fn run(path: PathBuf) -> Result<(), String> {
    let file = File::open(&path)
        .map_err(|err| format!("failed to open archive {}: {err}", path.display()))?;
    let zstd = zstd::Decoder::new(file)
        .map_err(|err| format!("failed to decode archive {}: {err}", path.display()))?;
    let mut archive = tar::Archive::new(zstd);

    let mut seen = 0usize;
    for entry_result in archive
        .entries()
        .map_err(|err| format!("failed to iterate archive {}: {err}", path.display()))?
    {
        let entry = entry_result
            .map_err(|err| format!("failed to read archive entry in {}: {err}", path.display()))?;
        let entry_path = entry
            .path()
            .map_err(|err| {
                format!(
                    "failed to inspect archive entry in {}: {err}",
                    path.display()
                )
            })?
            .into_owned();

        if !is_mjai_archive_entry(&entry_path) {
            continue;
        }

        seen += 1;
        match load_game_from_stream(BufReader::new(entry)) {
            Ok(_) => {}
            Err(err) => {
                println!("ARCHIVE: {}", path.display());
                println!("ENTRY: {}", entry_path.display());
                println!("SEEN: {}", seen);
                println!("ERROR:\n{}", err);
                return Ok(());
            }
        }
    }

    println!("ARCHIVE: {}", path.display());
    println!("No failures found after scanning {} MJAI entries.", seen);
    Ok(())
}

fn main() {
    let mut args = std::env::args();
    let program = args
        .next()
        .unwrap_or_else(|| "mjai_first_failure".to_string());
    let Some(path) = args.next() else {
        eprintln!("{}", usage(&program));
        std::process::exit(2);
    };
    if args.next().is_some() {
        eprintln!("{}", usage(&program));
        std::process::exit(2);
    }

    if let Err(err) = run(PathBuf::from(path)) {
        eprintln!("{err}");
        std::process::exit(1);
    }
}
