//! Emit direct raw-MJAI host batches as binary frames for Python BC training.

use std::io::{self, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use hydra_train_exec::raw_mjai_stream::{
    RawMjaiBatchStreamConfig, RawMjaiStreamSplit, stream_raw_mjai_batches,
};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            let _ = writeln!(io::stderr(), "raw MJAI stream failed: {err}");
            ExitCode::from(1)
        }
    }
}

fn run() -> io::Result<()> {
    let config = parse_args(std::env::args().skip(1))?;
    let stdout = io::stdout();
    let totals = stream_raw_mjai_batches(&config, stdout.lock())?;
    let _ = writeln!(
        io::stderr(),
        "raw MJAI stream complete: loaded_games={} skipped_games={} samples={} batches={}",
        totals.loaded_games,
        totals.skipped_games,
        totals.samples,
        totals.batches
    );
    Ok(())
}

fn parse_args(args: impl IntoIterator<Item = String>) -> io::Result<RawMjaiBatchStreamConfig> {
    let mut config = RawMjaiBatchStreamConfig::default();
    config.inputs.clear();
    let mut iter = args.into_iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--input" => config
                .inputs
                .push(PathBuf::from(next_value(&mut iter, "--input")?)),
            "--batch-size" => {
                config.batch_size =
                    parse_usize(&next_value(&mut iter, "--batch-size")?, "--batch-size")?
            }
            "--train-fraction" => {
                config.train_fraction = parse_f32(
                    &next_value(&mut iter, "--train-fraction")?,
                    "--train-fraction",
                )?;
            }
            "--split" => {
                config.split = parse_split(&next_value(&mut iter, "--split")?)?;
            }
            "--max-games" => {
                config.max_games = Some(parse_usize(
                    &next_value(&mut iter, "--max-games")?,
                    "--max-games",
                )?);
            }
            "--max-samples" => {
                config.max_samples = Some(parse_usize(
                    &next_value(&mut iter, "--max-samples")?,
                    "--max-samples",
                )?);
            }
            "--skip-games" => {
                config.skip_games =
                    parse_usize(&next_value(&mut iter, "--skip-games")?, "--skip-games")?;
            }
            "--num-threads" => {
                config.num_threads = Some(parse_usize(
                    &next_value(&mut iter, "--num-threads")?,
                    "--num-threads",
                )?);
            }
            "--queue-bound" => {
                config.queue_bound =
                    parse_usize(&next_value(&mut iter, "--queue-bound")?, "--queue-bound")?;
            }
            "--augment" => config.augment = true,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("unknown argument {other}"),
                ));
            }
        }
    }
    if config.inputs.is_empty() {
        config.inputs.push(PathBuf::from("."));
    }
    Ok(config)
}

fn next_value(iter: &mut impl Iterator<Item = String>, flag: &str) -> io::Result<String> {
    iter.next().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{flag} requires a value"),
        )
    })
}

fn parse_usize(value: &str, flag: &str) -> io::Result<usize> {
    value.parse::<usize>().map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("invalid {flag} value {value:?}: {err}"),
        )
    })
}

fn parse_f32(value: &str, flag: &str) -> io::Result<f32> {
    value.parse::<f32>().map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("invalid {flag} value {value:?}: {err}"),
        )
    })
}

fn parse_split(value: &str) -> io::Result<RawMjaiStreamSplit> {
    match value {
        "train" => Ok(RawMjaiStreamSplit::Train),
        "validation" => Ok(RawMjaiStreamSplit::Validation),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("invalid --split value {value:?}; expected train or validation"),
        )),
    }
}

fn print_help() {
    println!(
        "Usage: raw_mjai_stream --input PATH [--input PATH ...] [--batch-size N] [--max-games N] [--max-samples N] [--skip-games N] [--num-threads N] [--queue-bound N] [--train-fraction F] [--split train|validation] [--augment]"
    );
}
