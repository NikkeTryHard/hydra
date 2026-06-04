use std::io::{self, Read};
use std::path::PathBuf;

use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_replay_loader::mjai_loader::{
    ReplayDecisionKind, ReplayDecisionPhase, ReplaySampleRecord, ReplaySampleSink,
    load_game_from_path, load_game_from_path_strict, load_game_from_reader,
    load_game_from_reader_strict, load_game_from_stream_strict_into_sink,
};
use serde::Serialize;

#[derive(Serialize)]
struct FixtureRow {
    index: usize,
    action_id: u8,
    legal_mask: Vec<u8>,
}

#[derive(Serialize)]
struct TraceRow {
    replay_path: Option<String>,
    index: usize,
    event_index: usize,
    source_event_type: &'static str,
    actor: usize,
    kind: &'static str,
    phase: &'static str,
    kyoku: u8,
    honba: u8,
    kyotaku: u32,
    oya: u8,
    round_wind: u8,
    response_target: Option<u8>,
    action_id: u8,
    legal_mask: Vec<u8>,
}

#[derive(Serialize)]
struct TraceOutput {
    action_space: usize,
    rows: Vec<TraceRow>,
}

struct TraceSink {
    replay_path: Option<String>,
    rows: Vec<TraceRow>,
}

fn trace_kind_name(kind: ReplayDecisionKind) -> &'static str {
    match kind {
        ReplayDecisionKind::ImplicitPass => "implicit_pass",
        ReplayDecisionKind::SampledEvent => "sampled_event",
    }
}

fn trace_phase_name(phase: ReplayDecisionPhase) -> &'static str {
    match phase {
        ReplayDecisionPhase::Normal => "normal",
        ReplayDecisionPhase::RiichiSelect => "riichi_select",
        ReplayDecisionPhase::KanSelect => "kan_select",
    }
}

impl ReplaySampleSink for TraceSink {
    fn push_sample(&mut self, sample: ReplaySampleRecord) -> io::Result<()> {
        let mut legal_mask = vec![0u8; HYDRA_ACTION_SPACE];
        for (dst, &src) in legal_mask.iter_mut().zip(sample.legal_mask.iter()) {
            *dst = u8::from(src > 0.0);
        }
        let index = self.rows.len();
        self.rows.push(TraceRow {
            replay_path: self.replay_path.clone(),
            index,
            event_index: sample.trace.event_index,
            source_event_type: sample.trace.source_event_type,
            actor: sample.trace.actor,
            kind: trace_kind_name(sample.trace.kind),
            phase: trace_phase_name(sample.trace.phase),
            kyoku: sample.trace.kyoku,
            honba: sample.trace.honba,
            kyotaku: sample.trace.kyotaku,
            oya: sample.trace.oya,
            round_wind: sample.trace.round_wind,
            response_target: sample.trace.response_target,
            action_id: sample.action,
            legal_mask,
        });
        Ok(())
    }
}

#[derive(Serialize)]
struct FixtureOutput {
    action_space: usize,
    rows: Vec<FixtureRow>,
}

fn usage(program: &str) -> String {
    format!("Usage: {program} [--strict] [--trace] [--batch PATH_LIST] [path]")
}

fn parse_args() -> Result<(bool, bool, Option<PathBuf>, Option<PathBuf>), String> {
    let mut strict = false;
    let mut trace = false;
    let mut batch = None;
    let mut path = None;
    let mut args = std::env::args();
    let program = args
        .next()
        .unwrap_or_else(|| "mjai_authority_fixture".to_string());
    while let Some(arg) = args.next() {
        if arg == "--strict" {
            strict = true;
        } else if arg == "--trace" {
            trace = true;
        } else if arg == "--batch" {
            let Some(list_path) = args.next() else {
                return Err(usage(&program));
            };
            batch = Some(PathBuf::from(list_path));
        } else if path.is_none() {
            path = Some(PathBuf::from(arg));
        } else {
            return Err(usage(&program));
        }
    }
    Ok((strict, trace, batch, path))
}

fn write_trace_for_path(
    path: PathBuf,
    rows: &mut Vec<TraceRow>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut sink = TraceSink {
        replay_path: Some(path.display().to_string()),
        rows: Vec::new(),
    };
    let file = std::fs::File::open(&path)?;
    load_game_from_stream_strict_into_sink(file, &mut sink)?;
    rows.extend(sink.rows);
    Ok(())
}

fn write_trace(
    path: Option<PathBuf>,
    batch: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut rows = Vec::new();
    if let Some(batch) = batch {
        let paths = std::fs::read_to_string(batch)?;
        for line in paths.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            write_trace_for_path(PathBuf::from(trimmed), &mut rows)?;
        }
    } else if let Some(path) = path {
        write_trace_for_path(path, &mut rows)?;
    } else {
        let mut sink = TraceSink {
            replay_path: None,
            rows: Vec::new(),
        };
        let mut input = Vec::new();
        io::stdin().read_to_end(&mut input)?;
        load_game_from_stream_strict_into_sink(io::Cursor::new(input), &mut sink)?;
        rows = sink.rows;
    }
    serde_json::to_writer(
        io::stdout(),
        &TraceOutput {
            action_space: HYDRA_ACTION_SPACE,
            rows,
        },
    )?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (strict, trace, batch, path) =
        parse_args().map_err(|err| io::Error::new(io::ErrorKind::InvalidInput, err))?;
    if trace {
        if !strict {
            return Err(
                io::Error::new(io::ErrorKind::InvalidInput, "--trace requires --strict").into(),
            );
        }
        return write_trace(path, batch);
    }
    if batch.is_some() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "--batch requires --trace").into());
    }
    let game = if let Some(path) = path {
        if strict {
            load_game_from_path_strict(&path)?
        } else {
            load_game_from_path(&path)?
        }
    } else {
        let mut input = String::new();
        io::stdin().read_to_string(&mut input)?;
        let reader = io::Cursor::new(input);
        if strict {
            load_game_from_reader_strict(reader)?
        } else {
            load_game_from_reader(reader)?
        }
    };

    let mut rows = Vec::with_capacity(game.samples.len());
    for (index, sample) in game.samples.iter().enumerate() {
        let mut legal_mask = vec![0u8; HYDRA_ACTION_SPACE];
        for (dst, &src) in legal_mask.iter_mut().zip(sample.legal_mask.iter()) {
            *dst = u8::from(src > 0.0);
        }
        rows.push(FixtureRow {
            index,
            action_id: sample.action,
            legal_mask,
        });
    }

    serde_json::to_writer(
        io::stdout(),
        &FixtureOutput {
            action_space: HYDRA_ACTION_SPACE,
            rows,
        },
    )?;
    Ok(())
}
