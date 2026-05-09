//! JSONL readers for sidecar record streams.

use std::io;
use std::io::BufRead;

use serde::de::DeserializeOwned;

/// Reads newline-delimited JSON records, skipping blank lines.
pub fn read_jsonl_records<T>(reader: impl BufRead, sidecar_name: &str) -> io::Result<Vec<T>>
where
    T: DeserializeOwned,
{
    let mut records = Vec::new();
    for (line_idx, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let record = serde_json::from_str(&line).map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid {sidecar_name} line {}: {err}", line_idx + 1),
            )
        })?;
        records.push(record);
    }
    Ok(records)
}
