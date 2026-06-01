use std::io::{self, Read, Write};

use crate::limits::{
    FINAL_SCORE_COUNT, MAX_PARSED_SAMPLE_CACHE_SAMPLES, PARSED_SAMPLE_CACHE_MAGIC,
    PARSED_SAMPLE_CACHE_MAGIC_LEN, PARSED_SAMPLE_CACHE_VERSION,
};
use crate::primitives::{read_i32, read_string, read_u32, write_i32, write_string, write_u32};
use crate::{invalid_data, invalid_input};

pub(crate) struct ParsedSampleCacheHeader {
    pub(crate) sample_count: u32,
    pub(crate) original_identity: String,
    pub(crate) original_source_path: String,
}

pub(crate) fn write_header(
    writer: &mut impl Write,
    original_source_path: &std::path::Path,
    original_identity: &str,
    sample_count: usize,
) -> io::Result<()> {
    writer.write_all(PARSED_SAMPLE_CACHE_MAGIC)?;
    write_u32(writer, PARSED_SAMPLE_CACHE_VERSION)?;
    write_u32(writer, checked_sample_count(sample_count)?)?;

    let source_path = original_source_path.to_string_lossy();
    write_string(writer, original_identity)?;
    write_string(writer, &source_path)
}

pub(crate) fn checked_sample_count(sample_count: usize) -> io::Result<u32> {
    let sample_count = u32::try_from(sample_count).map_err(|_| {
        invalid_input(format!(
            "parsed-sample cache sample count {sample_count} exceeds u32 format capacity"
        ))
    })?;
    validate_sample_count(sample_count).map_err(|err| invalid_input(err.to_string()))?;
    Ok(sample_count)
}

pub(crate) fn validate_sample_count(sample_count: u32) -> io::Result<()> {
    if sample_count > MAX_PARSED_SAMPLE_CACHE_SAMPLES {
        return Err(invalid_data(format!(
            "parsed-sample cache sample count {sample_count} exceeds maximum {MAX_PARSED_SAMPLE_CACHE_SAMPLES}"
        )));
    }
    Ok(())
}

pub(crate) fn read_header_internal(reader: &mut impl Read) -> io::Result<ParsedSampleCacheHeader> {
    let mut magic = [0u8; PARSED_SAMPLE_CACHE_MAGIC_LEN];
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
    validate_sample_count(sample_count)?;
    let original_identity = read_string(reader)?;
    let original_source_path = read_string(reader)?;

    Ok(ParsedSampleCacheHeader {
        sample_count,
        original_identity,
        original_source_path,
    })
}

pub(crate) fn read_final_scores(reader: &mut impl Read) -> io::Result<[i32; 4]> {
    let mut final_scores = [0i32; FINAL_SCORE_COUNT];
    for score in &mut final_scores {
        *score = read_i32(reader)?;
    }
    Ok(final_scores)
}

pub(crate) fn write_final_scores(
    writer: &mut impl Write,
    final_scores: [i32; FINAL_SCORE_COUNT],
) -> io::Result<()> {
    for score in final_scores {
        write_i32(writer, score)?;
    }
    Ok(())
}
