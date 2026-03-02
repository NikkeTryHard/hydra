//! Batch observation encoder for training throughput.
//!
//! Pre-allocates a contiguous buffer for N observations and encodes
//! directly into slots, avoiding per-observation allocation.

use crate::encoder::{ObservationEncoder, OBS_SIZE};

/// Batch encoder that manages a contiguous buffer for multiple observations.
///
/// The buffer layout is `[batch_size, OBS_SIZE]` in row-major order,
/// matching the tensor layout expected by the training pipeline.
pub struct BatchEncoder {
    /// Contiguous buffer: batch_size * OBS_SIZE f32 values.
    buffer: Vec<f32>,
    /// Number of slots in the batch.
    batch_size: usize,
}

impl BatchEncoder {
    /// Creates a new batch encoder with the given batch size.
    ///
    /// Allocates a single contiguous buffer that is reused across
    /// training steps.
    #[inline]
    pub fn new(batch_size: usize) -> Self {
        Self {
            buffer: vec![0.0; batch_size * OBS_SIZE],
            batch_size,
        }
    }

    /// Returns the batch size.
    #[inline]
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Copies a pre-encoded observation into the batch at the given slot.
    ///
    /// # Panics
    ///
    /// Panics if `slot >= batch_size`.
    #[inline]
    pub fn copy_from_encoder(&mut self, slot: usize, src: &ObservationEncoder) {
        assert!(slot < self.batch_size, "slot {slot} >= batch_size {}", self.batch_size);
        let start = slot * OBS_SIZE;
        self.buffer[start..start + OBS_SIZE].copy_from_slice(src.as_slice());
    }

    /// Returns a mutable slice for a specific slot.
    ///
    /// This allows writing directly into the batch buffer.
    ///
    /// # Panics
    ///
    /// Panics if `slot >= batch_size`.
    #[inline]
    pub fn slot_mut(&mut self, slot: usize) -> &mut [f32] {
        assert!(slot < self.batch_size, "slot {slot} >= batch_size {}", self.batch_size);
        let start = slot * OBS_SIZE;
        &mut self.buffer[start..start + OBS_SIZE]
    }

    /// Returns the full batch as a contiguous slice.
    ///
    /// Layout: `[batch_size, OBS_SIZE]` row-major.
    /// Can be directly copied to GPU tensor memory.
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.buffer
    }

    /// Returns a mutable reference to the full buffer.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.buffer
    }

    /// Clears all slots to zero.
    #[inline]
    pub fn clear(&mut self) {
        self.buffer.fill(0.0);
    }

    /// Returns the total number of f32 values in the buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Returns true if the batch has zero capacity.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.batch_size == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_encoder_creates_correct_size() {
        let batch = BatchEncoder::new(32);
        assert_eq!(batch.len(), 32 * OBS_SIZE);
        assert_eq!(batch.batch_size(), 32);
    }

    #[test]
    fn batch_encoder_slot_isolation() {
        let mut batch = BatchEncoder::new(2);
        let slot0 = batch.slot_mut(0);
        slot0[0] = 1.0;
        let slot1 = batch.slot_mut(1);
        assert_eq!(slot1[0], 0.0);
    }

    #[test]
    fn batch_encoder_clear_zeros() {
        let mut batch = BatchEncoder::new(4);
        batch.slot_mut(2)[100] = 42.0;
        batch.clear();
        assert_eq!(batch.as_slice()[2 * OBS_SIZE + 100], 0.0);
    }
}
