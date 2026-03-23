use std::fs::File;
use std::io::BufReader;

use hydra_train::data::mjai_loader::debug_first_replay_failure_from_reader;

fn usage(program: &str) -> String {
    format!("Usage: {program} <replay.json>")
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

    let file = File::open(&path).expect("open replay file");
    let reader = BufReader::new(file);
    match debug_first_replay_failure_from_reader(reader).expect("debug replay failure") {
        Some(report) => println!("{report}"),
        None => println!("No failure found."),
    }
}
