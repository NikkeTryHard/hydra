use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

use hydra_replay_loader::mjai_loader::debug_first_replay_failure_from_reader;

fn usage(program: &str) -> String {
    format!("Usage: {program} <replay.json>")
}

fn run(path: PathBuf) -> Result<(), String> {
    let file = File::open(&path)
        .map_err(|err| format!("failed to open replay file {}: {err}", path.display()))?;
    let reader = BufReader::new(file);
    match debug_first_replay_failure_from_reader(reader).map_err(|err| {
        format!(
            "failed to debug replay failure in {}: {err}",
            path.display()
        )
    })? {
        Some(report) => println!("{report}"),
        None => println!("No failure found."),
    }
    Ok(())
}

fn main() {
    let mut args = std::env::args();
    let program = args
        .next()
        .unwrap_or_else(|| "mjai_debug_failure".to_string());
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
