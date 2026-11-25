//! Pack one prepared batch into the four little-endian wire byte arrays.
//!
//! Descriptor width is the maximum legal-action count in the batch. Every
//! unused descriptor slot is filled with int32 -1, which the Python boundary
//! recognizes as padding. No per-row Python object is created.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyByteArray;

use crate::features;

use super::batch::PreparedRow;

pub(super) type WireBuffers = (
    Py<PyByteArray>,
    Py<PyByteArray>,
    Py<PyByteArray>,
    Py<PyByteArray>,
);

pub(super) fn pack(py: Python<'_>, rows: &[PreparedRow]) -> PyResult<WireBuffers> {
    let k = rows.iter().map(|row| row.legal_count).max().unwrap_or(0);
    let token_width = features::TOKENS
        .checked_mul(features::WIDTH)
        .and_then(|count| count.checked_mul(4))
        .ok_or_else(|| PyValueError::new_err("token row geometry overflow"))?;
    let action_width = features::ACTION
        .checked_mul(4)
        .ok_or_else(|| PyValueError::new_err("action row geometry overflow"))?;
    let slot_bytes = rows
        .len()
        .checked_mul(4)
        .ok_or_else(|| PyValueError::new_err("slot batch geometry overflow"))?;
    let token_bytes = rows
        .len()
        .checked_mul(token_width)
        .ok_or_else(|| PyValueError::new_err("token batch geometry overflow"))?;
    let descriptor_bytes = rows
        .len()
        .checked_mul(k)
        .and_then(|count| count.checked_mul(action_width))
        .ok_or_else(|| PyValueError::new_err("descriptor batch geometry overflow"))?;

    let slots = PyByteArray::new_with(py, slot_bytes, |out| {
        let (chunks, remainder) = out.as_chunks_mut::<4>();
        if !remainder.is_empty() {
            return Err(PyRuntimeError::new_err("slot buffer is not int32 aligned"));
        }
        for (row, chunk) in rows.iter().zip(chunks) {
            let slot =
                i32::try_from(row.slot).map_err(|_| PyValueError::new_err("slot exceeds int32"))?;
            chunk.copy_from_slice(&slot.to_le_bytes());
        }
        Ok(())
    })?;
    let seats = PyByteArray::new_with(py, rows.len(), |out| {
        for (row, seat) in rows.iter().zip(out.iter_mut()) {
            *seat = row.seat;
        }
        Ok(())
    })?;
    let tokens = PyByteArray::new_with(py, token_bytes, |out| {
        let mut offset = 0usize;
        for row in rows {
            if row.tokens.len() != features::TOKENS * features::WIDTH {
                return Err(PyRuntimeError::new_err("token row length mismatch"));
            }
            for value in &row.tokens {
                out[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
                offset += 4;
            }
        }
        if offset != out.len() {
            return Err(PyRuntimeError::new_err("token row length mismatch"));
        }
        Ok(())
    })?;
    let descriptors = PyByteArray::new_with(py, descriptor_bytes, |out| {
        let padding = (-1i32).to_le_bytes();
        let mut offset = 0usize;
        for row in rows {
            if row.candidates.len() != row.legal_count * features::ACTION {
                return Err(PyRuntimeError::new_err("candidate row length mismatch"));
            }
            for action in 0..k {
                if action < row.legal_count {
                    let start = action * features::ACTION;
                    for value in &row.candidates[start..start + features::ACTION] {
                        out[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
                        offset += 4;
                    }
                } else {
                    for _ in 0..features::ACTION {
                        out[offset..offset + 4].copy_from_slice(&padding);
                        offset += 4;
                    }
                }
            }
        }
        if offset != out.len() {
            return Err(PyRuntimeError::new_err("descriptor row length mismatch"));
        }
        Ok(())
    })?;

    Ok((
        slots.unbind(),
        seats.unbind(),
        tokens.unbind(),
        descriptors.unbind(),
    ))
}
