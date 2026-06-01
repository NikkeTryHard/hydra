use std::io::{self, Read, Write};

use crate::limits::MAX_PARSED_SAMPLE_CACHE_METADATA_STRING_LEN;
use crate::{invalid_data, invalid_input};

pub(crate) fn write_optional_f32_array<const N: usize>(
    writer: &mut impl Write,
    values: Option<&[f32; N]>,
) -> io::Result<()> {
    if let Some(values) = values {
        write_f32_array(writer, values)?;
    }
    Ok(())
}

pub(crate) fn read_optional_f32_array<const N: usize>(
    reader: &mut impl Read,
    present: bool,
) -> io::Result<Option<[f32; N]>> {
    present.then(|| read_f32_array(reader)).transpose()
}

pub(crate) fn write_string(writer: &mut impl Write, value: &str) -> io::Result<()> {
    if value.len() > MAX_PARSED_SAMPLE_CACHE_METADATA_STRING_LEN {
        return Err(invalid_input(format!(
            "parsed-sample cache metadata string length {} exceeds maximum {MAX_PARSED_SAMPLE_CACHE_METADATA_STRING_LEN}",
            value.len()
        )));
    }
    write_u32(
        writer,
        u32::try_from(value.len()).map_err(|_| {
            invalid_input(format!(
                "parsed-sample cache metadata string length {} exceeds u32 format capacity",
                value.len()
            ))
        })?,
    )?;
    writer.write_all(value.as_bytes())
}

pub(crate) fn read_string(reader: &mut impl Read) -> io::Result<String> {
    let len = read_u32(reader)? as usize;
    if len > MAX_PARSED_SAMPLE_CACHE_METADATA_STRING_LEN {
        return Err(invalid_data(format!(
            "parsed-sample cache metadata string length {len} exceeds maximum {MAX_PARSED_SAMPLE_CACHE_METADATA_STRING_LEN}"
        )));
    }
    let mut bytes = vec![0u8; len];
    reader.read_exact(&mut bytes)?;
    String::from_utf8(bytes)
        .map_err(|err| invalid_data(format!("invalid UTF-8 in cache metadata: {err}")))
}

pub(crate) fn write_bool(writer: &mut impl Write, value: bool) -> io::Result<()> {
    write_u8(writer, u8::from(value))
}

pub(crate) fn read_bool(reader: &mut impl Read) -> io::Result<bool> {
    match read_u8(reader)? {
        0 => Ok(false),
        1 => Ok(true),
        other => Err(invalid_data(format!(
            "invalid bool value in parsed-sample cache: {other}"
        ))),
    }
}

pub(crate) fn write_u8(writer: &mut impl Write, value: u8) -> io::Result<()> {
    writer.write_all(&[value])
}

pub(crate) fn read_u8(reader: &mut impl Read) -> io::Result<u8> {
    let mut buf = [0u8; 1];
    reader.read_exact(&mut buf)?;
    Ok(buf[0])
}

pub(crate) fn write_u16(writer: &mut impl Write, value: u16) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

pub(crate) fn read_u16(reader: &mut impl Read) -> io::Result<u16> {
    let mut buf = [0u8; 2];
    reader.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

pub(crate) fn write_u32(writer: &mut impl Write, value: u32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

pub(crate) fn read_u32(reader: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

pub(crate) fn write_i32(writer: &mut impl Write, value: i32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

pub(crate) fn read_i32(reader: &mut impl Read) -> io::Result<i32> {
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

pub(crate) fn write_f32_array<const N: usize>(
    writer: &mut impl Write,
    values: &[f32; N],
) -> io::Result<()> {
    for value in values {
        write_f32(writer, *value)?;
    }
    Ok(())
}

pub(crate) fn read_f32_array<const N: usize>(reader: &mut impl Read) -> io::Result<[f32; N]> {
    let mut out = [0.0f32; N];
    for value in &mut out {
        *value = read_f32(reader)?;
    }
    Ok(out)
}
