//! Primitive little-endian write helpers.

use std::io::{self, Write};

pub(super) fn write_zero_bytes<W: Write>(writer: &mut W, total: usize) -> io::Result<()> {
    const ZERO_CHUNK: [u8; 4096] = [0u8; 4096];
    let mut remaining = total;
    while remaining > 0 {
        let chunk = remaining.min(ZERO_CHUNK.len());
        writer.write_all(&ZERO_CHUNK[..chunk])?;
        remaining -= chunk;
    }
    Ok(())
}

pub(super) fn write_u8<W: Write>(writer: &mut W, value: u8) -> io::Result<()> {
    writer.write_all(&[value])
}

pub(super) fn write_u32_le<W: Write>(writer: &mut W, value: u32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

pub(super) fn write_u64_le<W: Write>(writer: &mut W, value: u64) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

pub(super) fn write_i32_le<W: Write>(writer: &mut W, value: i32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

pub(super) fn write_required_f32_slice<W: Write, const N: usize>(
    writer: &mut W,
    values: &[f32; N],
) -> io::Result<()> {
    for &value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}
