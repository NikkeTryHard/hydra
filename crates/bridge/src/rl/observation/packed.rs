//! Packed metadata layout for the mixed-transport actor path.
//!
//! The dense 26+uniform T256 planes stay unchanged; this module owns only the
//! packed metadata slab the GPU graphs consume. Layout is byte-exact:
//! `packed_kind` int64[C] at 0, `cu_seqlens` int32[B+1] at 8*C, align up to 8,
//! then `token_row` int64[C], `token_pos` int64[C], `valid` bool[C]. Tail
//! storage is deterministic zeros with `valid` False; real prefixes carry the
//! actual kind/row/position. `B` is the lease row count (<=128), `C` is the
//! token capacity (<=32768, 2048-classed by the owner, minimum 2048 in
//! production). No Python loops; one batched native fill.

use super::ObservationError;

/// Maximum packed rows per ticket (matches `TICKET_CAP`).
pub const PACKED_MAX_ROWS: usize = 128;
/// Maximum packed tokens per batch (graph bound).
pub const PACKED_MAX_TOKENS: usize = 32768;
/// Graph capacity quantum (owner classes `ceil(total/2048)*2048`, min 2048).
pub const PACKED_CAP_QUANTUM: usize = 2048;

const fn align_up(offset: usize, width: usize) -> usize {
    let mask = width - 1;
    (offset + mask) & !mask
}

/// Byte offsets for `(C, B)`: `(name, dtype, shape, byte_offset)` in fill
/// order. `B` is the lease row count, `C` the token capacity.
pub fn packed_layout(
    token_capacity: usize,
    batch: usize,
) -> Result<Vec<(String, String, Vec<usize>, usize)>, ObservationError> {
    if batch == 0 || batch > PACKED_MAX_ROWS {
        return Err(ObservationError(format!(
            "packed batch {batch} outside 1..={PACKED_MAX_ROWS}"
        )));
    }
    if token_capacity > PACKED_MAX_TOKENS {
        return Err(ObservationError(format!(
            "packed capacity {token_capacity} exceeds {PACKED_MAX_TOKENS}"
        )));
    }
    let mut out = Vec::with_capacity(5);
    let mut offset = 0usize;
    out.push((
        "packed_kind".to_string(),
        "int64".to_string(),
        vec![token_capacity],
        offset,
    ));
    offset += token_capacity * 8;
    // cu_seqlens int32[B+1] at 8*C (int32 width 4; base need not be 8-aligned
    // until the following int64 region aligns up).
    out.push((
        "cu_seqlens".to_string(),
        "int32".to_string(),
        vec![batch + 1],
        offset,
    ));
    offset += (batch + 1) * 4;
    offset = align_up(offset, 8);
    out.push((
        "token_row".to_string(),
        "int64".to_string(),
        vec![token_capacity],
        offset,
    ));
    offset += token_capacity * 8;
    out.push((
        "token_pos".to_string(),
        "int64".to_string(),
        vec![token_capacity],
        offset,
    ));
    offset += token_capacity * 8;
    out.push((
        "valid".to_string(),
        "bool".to_string(),
        vec![token_capacity],
        offset,
    ));
    offset += token_capacity;
    let _ = offset;
    Ok(out)
}

/// Total packed metadata bytes for `(C, B)`: last region end, no tail pad.
pub fn packed_bytes(token_capacity: usize, batch: usize) -> Result<usize, ObservationError> {
    let layout = packed_layout(token_capacity, batch)?;
    let (name, dtype, shape, off) = layout
        .last()
        .cloned()
        .ok_or_else(|| ObservationError("packed layout empty".to_string()))?;
    let width = match dtype.as_str() {
        "int64" | "float64" => 8,
        "int32" => 4,
        _ => 1,
    };
    let _ = name;
    Ok(off + shape.iter().product::<usize>() * width)
}
