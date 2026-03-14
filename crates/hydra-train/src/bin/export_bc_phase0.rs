#[path = "export_bc_phase0/config.rs"]
mod config;
#[path = "export_bc_phase0/exporter.rs"]
mod exporter;
#[path = "export_bc_phase0/manifest.rs"]
mod manifest;
#[path = "export_bc_phase0/shard_writer.rs"]
mod shard_writer;

use std::env;

use self::config::{parse_args, read_config};
use self::exporter::run_export;

fn run() -> Result<(), String> {
    let cli = parse_args(env::args())?;
    let config = read_config(&cli.config_path)?;
    run_export(&config).map_err(|err| err.to_string())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}
