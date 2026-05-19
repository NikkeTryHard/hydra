use std::path::PathBuf;

use hydra_train_runtime::timing_metrics::{
    TimingMetricsOptions, extract_timing_metrics_from_paths,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OutputFormat {
    Json,
    Csv,
}

#[derive(Debug)]
struct Cli {
    step_logs: Vec<PathBuf>,
    training_logs: Vec<PathBuf>,
    run_id: Option<String>,
    skip_initial_rows: usize,
    min_global_step: Option<usize>,
    format: OutputFormat,
}

enum ParsedArgs {
    Run(Cli),
    Help(String),
}

fn usage(program: &str) -> String {
    format!(
        "Usage: {program} (--step-log <path>|--training-log <path>)... [--run-id <string>] [--skip-initial-rows <usize>] [--min-global-step <usize>] [--format json|csv]"
    )
}

fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    match args.next() {
        Some(value) if !value.starts_with("--") => Ok(value),
        _ => Err(format!("missing value for {flag}")),
    }
}

fn parse_usize_flag(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<usize, String> {
    next_value(args, flag)?
        .parse::<usize>()
        .map_err(|err| format!("invalid {flag}: {err}"))
}

fn parse_format(value: &str) -> Result<OutputFormat, String> {
    match value {
        "json" => Ok(OutputFormat::Json),
        "csv" => Ok(OutputFormat::Csv),
        _ => Err(format!("invalid --format value: {value}")),
    }
}

fn parse_args<I>(program: &str, args: I) -> Result<ParsedArgs, String>
where
    I: IntoIterator<Item = String>,
{
    let mut args = args.into_iter();
    let _ = args.next();

    let mut step_logs = Vec::new();
    let mut training_logs = Vec::new();
    let mut run_id = None;
    let mut skip_initial_rows = 0usize;
    let mut min_global_step = None;
    let mut format = OutputFormat::Json;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--step-log" => step_logs.push(PathBuf::from(next_value(&mut args, "--step-log")?)),
            "--training-log" => {
                training_logs.push(PathBuf::from(next_value(&mut args, "--training-log")?))
            }
            "--run-id" => run_id = Some(next_value(&mut args, "--run-id")?),
            "--skip-initial-rows" => {
                skip_initial_rows = parse_usize_flag(&mut args, "--skip-initial-rows")?;
            }
            "--min-global-step" => {
                min_global_step = Some(parse_usize_flag(&mut args, "--min-global-step")?);
            }
            "--format" => format = parse_format(&next_value(&mut args, "--format")?)?,
            "--help" | "-h" => return Ok(ParsedArgs::Help(usage(program))),
            _ => return Err(format!("unknown argument: {arg}\n{}", usage(program))),
        }
    }

    if step_logs.is_empty() && training_logs.is_empty() {
        return Err(usage(program));
    }

    Ok(ParsedArgs::Run(Cli {
        step_logs,
        training_logs,
        run_id,
        skip_initial_rows,
        min_global_step,
        format,
    }))
}

fn run() -> Result<(), String> {
    let program = std::env::args()
        .next()
        .unwrap_or_else(|| "extract_timing_metrics".to_string());
    let cli = match parse_args(&program, std::env::args())? {
        ParsedArgs::Run(cli) => cli,
        ParsedArgs::Help(usage) => {
            println!("{usage}");
            return Ok(());
        }
    };
    let options = TimingMetricsOptions {
        run_id: cli.run_id,
        skip_initial_rows: cli.skip_initial_rows,
        min_global_step: cli.min_global_step,
    };
    let report = extract_timing_metrics_from_paths(&cli.step_logs, &cli.training_logs, &options)?;
    match cli.format {
        OutputFormat::Json => {
            serde_json::to_writer_pretty(std::io::stdout(), &report)
                .map_err(|err| format!("failed to write JSON report: {err}"))?;
            println!();
        }
        OutputFormat::Csv => write_csv(&report)?,
    }
    Ok(())
}

fn write_csv(
    report: &hydra_train_runtime::timing_metrics::TimingMetricsReport,
) -> Result<(), String> {
    let mut out = String::from(
        "run_id,path,line_number,scope,root_stage,global_step,epoch,complete_for_gate,missing_stages,window_steps,window_samples,steps_per_second,samples_per_second,elapsed_seconds,train_seconds,producer_wait_seconds,collation_seconds,h2d_transfer_seconds,h2d_pageable_to_pinned_seconds,h2d_tensor_materialize_seconds,h2d_stream_sync_seconds,forward_seconds,loss_seconds,backward_seconds,optimizer_step_seconds,metric_readback_seconds,validation_seconds,checkpoint_seconds,logging_seconds,producer_wait_pct,collation_pct,h2d_transfer_pct,input_starvation_pct,compute_pct,metric_readback_pct,validation_pct,checkpoint_pct,logging_pct,h2d_pageable_to_pinned_pct_of_h2d,h2d_tensor_materialize_pct_of_h2d,h2d_stream_sync_pct_of_h2d\n",
    );
    for row in &report.rows {
        push_csv_field(&mut out, row.run_id.as_deref().unwrap_or(""));
        push_csv_field(&mut out, &row.path);
        push_csv_field(&mut out, &row.line_number.to_string());
        push_csv_field(
            &mut out,
            match row.scope {
                hydra_train_runtime::timing_metrics::LogScope::Step => "step",
                hydra_train_runtime::timing_metrics::LogScope::Epoch => "epoch",
            },
        );
        push_csv_field(&mut out, &row.root_stage);
        push_csv_option_usize(&mut out, row.global_step);
        push_csv_option_usize(&mut out, row.epoch);
        push_csv_field(
            &mut out,
            if row.complete_for_gate {
                "true"
            } else {
                "false"
            },
        );
        push_csv_field(&mut out, &row.missing_stages.join(";"));
        push_csv_option_usize(&mut out, row.window_steps);
        push_csv_option_usize(&mut out, row.window_samples);
        push_csv_option_f64(&mut out, row.steps_per_second);
        push_csv_option_f64(&mut out, row.samples_per_second);
        push_csv_f64(&mut out, row.elapsed_seconds);
        push_csv_option_f64(&mut out, row.train_seconds);
        push_csv_option_f64(&mut out, row.producer_wait_seconds);
        push_csv_option_f64(&mut out, row.collation_seconds);
        push_csv_option_f64(&mut out, row.h2d_transfer_seconds);
        push_csv_option_f64(&mut out, row.h2d_pageable_to_pinned_seconds);
        push_csv_option_f64(&mut out, row.h2d_tensor_materialize_seconds);
        push_csv_option_f64(&mut out, row.h2d_stream_sync_seconds);
        push_csv_option_f64(&mut out, row.forward_seconds);
        push_csv_option_f64(&mut out, row.loss_seconds);
        push_csv_option_f64(&mut out, row.backward_seconds);
        push_csv_option_f64(&mut out, row.optimizer_step_seconds);
        push_csv_option_f64(&mut out, row.metric_readback_seconds);
        push_csv_option_f64(&mut out, row.validation_seconds);
        push_csv_option_f64(&mut out, row.checkpoint_seconds);
        push_csv_option_f64(&mut out, row.logging_seconds);
        push_csv_option_f64(&mut out, row.producer_wait_pct);
        push_csv_option_f64(&mut out, row.collation_pct);
        push_csv_option_f64(&mut out, row.h2d_transfer_pct);
        push_csv_option_f64(&mut out, row.input_starvation_pct);
        push_csv_option_f64(&mut out, row.compute_pct);
        push_csv_option_f64(&mut out, row.metric_readback_pct);
        push_csv_option_f64(&mut out, row.validation_pct);
        push_csv_option_f64(&mut out, row.checkpoint_pct);
        push_csv_option_f64(&mut out, row.logging_pct);
        push_csv_option_f64(&mut out, row.h2d_pageable_to_pinned_pct_of_h2d);
        push_csv_option_f64(&mut out, row.h2d_tensor_materialize_pct_of_h2d);
        push_last_csv_option_f64(&mut out, row.h2d_stream_sync_pct_of_h2d);
        out.push('\n');
    }
    print!("{out}");
    Ok(())
}

fn push_csv_option_usize(out: &mut String, value: Option<usize>) {
    match value {
        Some(value) => push_csv_field(out, &value.to_string()),
        None => push_csv_field(out, ""),
    }
}

fn push_csv_f64(out: &mut String, value: f64) {
    push_csv_field(out, &value.to_string());
}

fn push_csv_option_f64(out: &mut String, value: Option<f64>) {
    match value {
        Some(value) => push_csv_field(out, &value.to_string()),
        None => push_csv_field(out, ""),
    }
}

fn push_last_csv_option_f64(out: &mut String, value: Option<f64>) {
    match value {
        Some(value) => push_last_csv_field(out, &value.to_string()),
        None => push_last_csv_field(out, ""),
    }
}

fn push_csv_field(out: &mut String, value: &str) {
    push_last_csv_field(out, value);
    out.push(',');
}

fn push_last_csv_field(out: &mut String, value: &str) {
    if value.contains([',', '"', '\n', '\r']) {
        out.push('"');
        for ch in value.chars() {
            if ch == '"' {
                out.push('"');
            }
            out.push(ch);
        }
        out.push('"');
    } else {
        out.push_str(value);
    }
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}
