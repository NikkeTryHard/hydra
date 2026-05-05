//! Parsed-sample cache file format for BC raw-path reuse.

use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use crate::data::mjai_loader::{MjaiGame, invalid_data};
use crate::data::sample::MjaiSample;

#[cfg(test)]
use hydra_core::action::HYDRA_ACTION_SPACE;
#[cfg(test)]
use hydra_core::encoder::OBS_SIZE;

pub const PARSED_SAMPLE_CACHE_EXTENSION: &str = ".samples.cache";

const PARSED_SAMPLE_CACHE_MAGIC: &[u8; 8] = b"HPSCACHE";
const PARSED_SAMPLE_CACHE_VERSION: u32 = 1;

const FLAG_ORACLE_TARGET: u16 = 1 << 0;
const FLAG_SAFETY_RESIDUAL: u16 = 1 << 1;
const FLAG_SAFETY_RESIDUAL_MASK: u16 = 1 << 2;
const FLAG_EXIT_TARGET: u16 = 1 << 3;
const FLAG_EXIT_MASK: u16 = 1 << 4;
const FLAG_DELTA_Q_TARGET: u16 = 1 << 5;
const FLAG_DELTA_Q_MASK: u16 = 1 << 6;
const FLAG_BELIEF_FIELDS: u16 = 1 << 7;
const FLAG_MIXTURE_WEIGHTS: u16 = 1 << 8;

#[cfg(test)]
const DANGER_TARGET_SIZE: usize = 3 * 34;
#[cfg(test)]
const BELIEF_FIELD_SIZE: usize = 16 * 34;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ParsedSampleCacheMetadata {
    pub original_source_path: PathBuf,
    pub original_identity: String,
    pub sample_count: usize,
}

pub struct ParsedSampleCacheFile {
    pub metadata: ParsedSampleCacheMetadata,
    pub game: MjaiGame,
}

pub fn is_parsed_sample_cache_file(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some(name) if name.ends_with(PARSED_SAMPLE_CACHE_EXTENSION)
    )
}

pub fn parsed_sample_cache_file_name(source_path: &Path) -> io::Result<String> {
    let file_name = source_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            invalid_data(format!(
                "source path does not have a valid UTF-8 filename: {}",
                source_path.display()
            ))
        })?;
    let stem = file_name
        .strip_suffix(".json.gz")
        .or_else(|| file_name.strip_suffix(".json"))
        .ok_or_else(|| {
            invalid_data(format!(
                "expected loose MJAI file ending in .json or .json.gz, got {}",
                source_path.display()
            ))
        })?;
    Ok(format!("{stem}{PARSED_SAMPLE_CACHE_EXTENSION}"))
}

pub fn write_parsed_sample_cache(
    path: &Path,
    original_source_path: &Path,
    original_identity: &str,
    game: &MjaiGame,
) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    writer.write_all(PARSED_SAMPLE_CACHE_MAGIC)?;
    write_u32(&mut writer, PARSED_SAMPLE_CACHE_VERSION)?;
    write_u32(&mut writer, game.samples.len() as u32)?;

    let source_path = original_source_path.to_string_lossy();
    write_string(&mut writer, original_identity)?;
    write_string(&mut writer, &source_path)?;

    for score in game.final_scores {
        write_i32(&mut writer, score)?;
    }

    for sample in &game.samples {
        write_sample(&mut writer, sample)?;
    }
    writer.flush()?;
    Ok(())
}

pub fn read_parsed_sample_cache_metadata(path: &Path) -> io::Result<ParsedSampleCacheMetadata> {
    let mut reader = BufReader::new(File::open(path)?);
    let header = read_header_internal(&mut reader)?;
    let _ = read_final_scores(&mut reader)?;
    Ok(ParsedSampleCacheMetadata {
        original_source_path: PathBuf::from(header.original_source_path),
        original_identity: header.original_identity,
        sample_count: header.sample_count as usize,
    })
}

pub fn load_parsed_sample_cache(path: &Path) -> io::Result<ParsedSampleCacheFile> {
    let mut reader = BufReader::new(File::open(path)?);
    let header = read_header_internal(&mut reader)?;
    let final_scores = read_final_scores(&mut reader)?;
    let samples = (0..header.sample_count)
        .map(|_| read_sample(&mut reader))
        .collect::<io::Result<Vec<_>>>()?;
    Ok(ParsedSampleCacheFile {
        metadata: ParsedSampleCacheMetadata {
            original_source_path: PathBuf::from(header.original_source_path),
            original_identity: header.original_identity,
            sample_count: header.sample_count as usize,
        },
        game: MjaiGame {
            samples,
            final_scores,
        },
    })
}

struct ParsedSampleCacheHeader {
    sample_count: u32,
    original_identity: String,
    original_source_path: String,
}

fn read_header_internal(reader: &mut impl Read) -> io::Result<ParsedSampleCacheHeader> {
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != PARSED_SAMPLE_CACHE_MAGIC {
        return Err(invalid_data("parsed-sample cache magic mismatch"));
    }

    let version = read_u32(reader)?;
    if version != PARSED_SAMPLE_CACHE_VERSION {
        return Err(invalid_data(format!(
            "parsed-sample cache version {version} unsupported (expected {})",
            PARSED_SAMPLE_CACHE_VERSION
        )));
    }

    let sample_count = read_u32(reader)?;
    let original_identity = read_string(reader)?;
    let original_source_path = read_string(reader)?;

    Ok(ParsedSampleCacheHeader {
        sample_count,
        original_identity,
        original_source_path,
    })
}

fn read_final_scores(reader: &mut impl Read) -> io::Result<[i32; 4]> {
    let mut final_scores = [0i32; 4];
    for score in &mut final_scores {
        *score = read_i32(reader)?;
    }
    Ok(final_scores)
}

fn write_sample(writer: &mut impl Write, sample: &MjaiSample) -> io::Result<()> {
    write_f32_array(writer, &sample.obs)?;
    write_u8(writer, sample.action)?;
    write_f32_array(writer, &sample.legal_mask)?;
    write_u8(writer, sample.placement)?;
    write_i32(writer, sample.score_delta)?;
    write_u8(writer, sample.grp_label)?;
    write_f32_array(writer, &sample.tenpai)?;
    writer.write_all(&sample.opp_next)?;
    write_f32_array(writer, &sample.danger)?;
    write_f32_array(writer, &sample.danger_mask)?;
    write_bool(writer, sample.belief_fields_present)?;
    write_bool(writer, sample.mixture_weights_present)?;

    let mut flags = 0u16;
    if sample.oracle_target.is_some() {
        flags |= FLAG_ORACLE_TARGET;
    }
    if sample.safety_residual.is_some() {
        flags |= FLAG_SAFETY_RESIDUAL;
    }
    if sample.safety_residual_mask.is_some() {
        flags |= FLAG_SAFETY_RESIDUAL_MASK;
    }
    if sample.exit_target.is_some() {
        flags |= FLAG_EXIT_TARGET;
    }
    if sample.exit_mask.is_some() {
        flags |= FLAG_EXIT_MASK;
    }
    if sample.delta_q_target.is_some() {
        flags |= FLAG_DELTA_Q_TARGET;
    }
    if sample.delta_q_mask.is_some() {
        flags |= FLAG_DELTA_Q_MASK;
    }
    if sample.belief_fields.is_some() {
        flags |= FLAG_BELIEF_FIELDS;
    }
    if sample.mixture_weights.is_some() {
        flags |= FLAG_MIXTURE_WEIGHTS;
    }
    write_u16(writer, flags)?;

    write_optional_f32_array(writer, sample.oracle_target.as_ref())?;
    write_optional_f32_array(writer, sample.safety_residual.as_ref())?;
    write_optional_f32_array(writer, sample.safety_residual_mask.as_ref())?;
    write_optional_f32_array(writer, sample.exit_target.as_ref())?;
    write_optional_f32_array(writer, sample.exit_mask.as_ref())?;
    write_optional_f32_array(writer, sample.delta_q_target.as_ref())?;
    write_optional_f32_array(writer, sample.delta_q_mask.as_ref())?;
    write_optional_f32_array(writer, sample.belief_fields.as_ref())?;
    write_optional_f32_array(writer, sample.mixture_weights.as_ref())?;
    Ok(())
}

fn read_sample(reader: &mut impl Read) -> io::Result<MjaiSample> {
    let obs = read_f32_array(reader)?;
    let action = read_u8(reader)?;
    let legal_mask = read_f32_array(reader)?;
    let placement = read_u8(reader)?;
    let score_delta = read_i32(reader)?;
    let grp_label = read_u8(reader)?;
    let tenpai = read_f32_array(reader)?;
    let mut opp_next = [0u8; 3];
    reader.read_exact(&mut opp_next)?;
    let danger = read_f32_array(reader)?;
    let danger_mask = read_f32_array(reader)?;
    let belief_fields_present = read_bool(reader)?;
    let mixture_weights_present = read_bool(reader)?;
    let flags = read_u16(reader)?;

    Ok(MjaiSample {
        obs,
        action,
        legal_mask,
        placement,
        score_delta,
        grp_label,
        oracle_target: read_optional_f32_array(reader, flags & FLAG_ORACLE_TARGET != 0)?,
        tenpai,
        opp_next,
        danger,
        danger_mask,
        safety_residual: read_optional_f32_array(reader, flags & FLAG_SAFETY_RESIDUAL != 0)?,
        safety_residual_mask: read_optional_f32_array(
            reader,
            flags & FLAG_SAFETY_RESIDUAL_MASK != 0,
        )?,
        exit_target: read_optional_f32_array(reader, flags & FLAG_EXIT_TARGET != 0)?,
        exit_mask: read_optional_f32_array(reader, flags & FLAG_EXIT_MASK != 0)?,
        delta_q_target: read_optional_f32_array(reader, flags & FLAG_DELTA_Q_TARGET != 0)?,
        delta_q_mask: read_optional_f32_array(reader, flags & FLAG_DELTA_Q_MASK != 0)?,
        belief_fields: read_optional_f32_array(reader, flags & FLAG_BELIEF_FIELDS != 0)?,
        mixture_weights: read_optional_f32_array(reader, flags & FLAG_MIXTURE_WEIGHTS != 0)?,
        belief_fields_present,
        mixture_weights_present,
    })
}

fn write_optional_f32_array<const N: usize>(
    writer: &mut impl Write,
    values: Option<&[f32; N]>,
) -> io::Result<()> {
    if let Some(values) = values {
        write_f32_array(writer, values)?;
    }
    Ok(())
}

fn read_optional_f32_array<const N: usize>(
    reader: &mut impl Read,
    present: bool,
) -> io::Result<Option<[f32; N]>> {
    present.then(|| read_f32_array(reader)).transpose()
}

fn write_string(writer: &mut impl Write, value: &str) -> io::Result<()> {
    write_u32(writer, value.len() as u32)?;
    writer.write_all(value.as_bytes())
}

fn read_string(reader: &mut impl Read) -> io::Result<String> {
    let len = read_u32(reader)? as usize;
    let mut bytes = vec![0u8; len];
    reader.read_exact(&mut bytes)?;
    String::from_utf8(bytes)
        .map_err(|err| invalid_data(format!("invalid UTF-8 in cache metadata: {err}")))
}

fn write_bool(writer: &mut impl Write, value: bool) -> io::Result<()> {
    write_u8(writer, u8::from(value))
}

fn read_bool(reader: &mut impl Read) -> io::Result<bool> {
    match read_u8(reader)? {
        0 => Ok(false),
        1 => Ok(true),
        other => Err(invalid_data(format!(
            "invalid bool value in parsed-sample cache: {other}"
        ))),
    }
}

fn write_u8(writer: &mut impl Write, value: u8) -> io::Result<()> {
    writer.write_all(&[value])
}

fn read_u8(reader: &mut impl Read) -> io::Result<u8> {
    let mut buf = [0u8; 1];
    reader.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn write_u16(writer: &mut impl Write, value: u16) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn read_u16(reader: &mut impl Read) -> io::Result<u16> {
    let mut buf = [0u8; 2];
    reader.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn write_u32(writer: &mut impl Write, value: u32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn read_u32(reader: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn write_i32(writer: &mut impl Write, value: i32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn read_i32(reader: &mut impl Read) -> io::Result<i32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn write_f32(writer: &mut impl Write, value: f32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn read_f32(reader: &mut impl Read) -> io::Result<f32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn write_f32_array<const N: usize>(writer: &mut impl Write, values: &[f32; N]) -> io::Result<()> {
    for value in values {
        write_f32(writer, *value)?;
    }
    Ok(())
}

fn read_f32_array<const N: usize>(reader: &mut impl Read) -> io::Result<[f32; N]> {
    let mut out = [0.0f32; N];
    for value in &mut out {
        *value = read_f32(reader)?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_path(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after epoch")
            .as_nanos();
        PathBuf::from("/home/cachybtw/tmp")
            .join(format!("hydra_parsed_sample_cache_{label}_{unique}.cache"))
    }

    fn sample_with_optionals(action: u8) -> MjaiSample {
        let mut obs = [0.0f32; OBS_SIZE];
        obs[0] = 0.25;
        obs[OBS_SIZE - 1] = 0.75;

        let mut legal_mask = [0.0f32; HYDRA_ACTION_SPACE];
        legal_mask[action as usize] = 1.0;
        legal_mask[HYDRA_ACTION_SPACE - 1] = 1.0;

        let mut danger = [0.0f32; DANGER_TARGET_SIZE];
        danger[1] = 0.5;
        danger[51] = 0.25;
        let mut danger_mask = [0.0f32; DANGER_TARGET_SIZE];
        danger_mask[1] = 1.0;
        danger_mask[90] = 1.0;

        let mut oracle_target = [0.0f32; 4];
        oracle_target[0] = 0.4;
        oracle_target[3] = -0.2;

        let mut safety_residual = [0.0f32; HYDRA_ACTION_SPACE];
        safety_residual[3] = 0.8;
        let mut safety_residual_mask = [0.0f32; HYDRA_ACTION_SPACE];
        safety_residual_mask[3] = 1.0;

        let mut exit_target = [0.0f32; HYDRA_ACTION_SPACE];
        exit_target[2] = 1.0;
        let mut exit_mask = [0.0f32; HYDRA_ACTION_SPACE];
        exit_mask[2] = 1.0;

        let mut delta_q_target = [0.0f32; HYDRA_ACTION_SPACE];
        delta_q_target[5] = -0.25;
        let mut delta_q_mask = [0.0f32; HYDRA_ACTION_SPACE];
        delta_q_mask[5] = 1.0;

        let mut belief_fields = [0.0f32; BELIEF_FIELD_SIZE];
        belief_fields[0] = 0.1;
        belief_fields[BELIEF_FIELD_SIZE - 1] = 0.9;

        let mut mixture_weights = [0.0f32; 4];
        mixture_weights[0] = 0.7;
        mixture_weights[1] = 0.3;

        MjaiSample {
            obs,
            action,
            legal_mask,
            placement: 2,
            score_delta: -1_200,
            grp_label: 7,
            oracle_target: Some(oracle_target),
            tenpai: [0.1, 0.2, 0.3],
            opp_next: [1, 17, 255],
            danger,
            danger_mask,
            safety_residual: Some(safety_residual),
            safety_residual_mask: Some(safety_residual_mask),
            exit_target: Some(exit_target),
            exit_mask: Some(exit_mask),
            delta_q_target: Some(delta_q_target),
            delta_q_mask: Some(delta_q_mask),
            belief_fields: Some(belief_fields),
            mixture_weights: Some(mixture_weights),
            belief_fields_present: true,
            mixture_weights_present: true,
        }
    }

    fn assert_sample_eq(lhs: &MjaiSample, rhs: &MjaiSample) {
        assert_eq!(lhs.obs, rhs.obs);
        assert_eq!(lhs.action, rhs.action);
        assert_eq!(lhs.legal_mask, rhs.legal_mask);
        assert_eq!(lhs.placement, rhs.placement);
        assert_eq!(lhs.score_delta, rhs.score_delta);
        assert_eq!(lhs.grp_label, rhs.grp_label);
        assert_eq!(lhs.oracle_target, rhs.oracle_target);
        assert_eq!(lhs.tenpai, rhs.tenpai);
        assert_eq!(lhs.opp_next, rhs.opp_next);
        assert_eq!(lhs.danger, rhs.danger);
        assert_eq!(lhs.danger_mask, rhs.danger_mask);
        assert_eq!(lhs.safety_residual, rhs.safety_residual);
        assert_eq!(lhs.safety_residual_mask, rhs.safety_residual_mask);
        assert_eq!(lhs.exit_target, rhs.exit_target);
        assert_eq!(lhs.exit_mask, rhs.exit_mask);
        assert_eq!(lhs.delta_q_target, rhs.delta_q_target);
        assert_eq!(lhs.delta_q_mask, rhs.delta_q_mask);
        assert_eq!(lhs.belief_fields, rhs.belief_fields);
        assert_eq!(lhs.mixture_weights, rhs.mixture_weights);
        assert_eq!(lhs.belief_fields_present, rhs.belief_fields_present);
        assert_eq!(lhs.mixture_weights_present, rhs.mixture_weights_present);
    }

    #[test]
    fn parsed_sample_cache_round_trips_game_and_metadata() {
        let path = unique_temp_path("round_trip");
        let game = MjaiGame {
            samples: vec![sample_with_optionals(3), sample_with_optionals(9)],
            final_scores: [31_000, 27_000, 23_000, 19_000],
        };
        let original_source_path = PathBuf::from("/data/raw/league_a/game_0001.mjai.json.gz");
        let original_identity = "league_a/game_0001.mjai.json.gz";

        write_parsed_sample_cache(&path, &original_source_path, original_identity, &game)
            .expect("cache write should succeed");

        let metadata =
            read_parsed_sample_cache_metadata(&path).expect("cache metadata read should succeed");
        assert_eq!(metadata.original_source_path, original_source_path);
        assert_eq!(metadata.original_identity, original_identity);
        assert_eq!(metadata.sample_count, game.samples.len());

        let loaded = load_parsed_sample_cache(&path).expect("cache load should succeed");
        assert_eq!(loaded.metadata, metadata);
        assert_eq!(loaded.game.final_scores, game.final_scores);
        assert_eq!(loaded.game.samples.len(), game.samples.len());
        for (lhs, rhs) in loaded.game.samples.iter().zip(game.samples.iter()) {
            assert_sample_eq(lhs, rhs);
        }

        fs::remove_file(path).ok();
    }

    #[test]
    fn parsed_sample_cache_file_name_rewrites_mjai_suffix() {
        let file_name = parsed_sample_cache_file_name(Path::new("/data/game_0001.mjai.json.gz"))
            .expect("cache filename should build");
        assert_eq!(file_name, "game_0001.mjai.samples.cache");
        assert!(is_parsed_sample_cache_file(Path::new(&file_name)));
    }
}
