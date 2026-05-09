//! Batch observation encoder for training throughput.
//!
//! Pre-allocates a contiguous buffer for N observations and encodes
//! directly into slots, avoiding per-observation allocation.

use crate::encoder::{OBS_SIZE, ObservationEncoder};

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
        assert!(
            slot < self.batch_size,
            "slot {slot} >= batch_size {}",
            self.batch_size
        );
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
        assert!(
            slot < self.batch_size,
            "slot {slot} >= batch_size {}",
            self.batch_size
        );
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

    #[test]
    fn batch_encoder_zero_batch_is_empty() {
        let batch = BatchEncoder::new(0);
        assert_eq!(batch.batch_size(), 0);
        assert_eq!(batch.len(), 0);
        assert!(batch.is_empty());
        assert!(batch.as_slice().is_empty());
    }

    #[test]
    fn batch_encoder_as_mut_slice_updates_buffer() {
        let mut batch = BatchEncoder::new(1);
        let buffer = batch.as_mut_slice();
        buffer[0] = 3.5;
        buffer[OBS_SIZE - 1] = 7.25;

        assert_eq!(batch.as_slice()[0], 3.5);
        assert_eq!(batch.as_slice()[OBS_SIZE - 1], 7.25);
    }

    #[test]
    #[should_panic(expected = "slot 0 >= batch_size 0")]
    fn batch_encoder_slot_mut_panics_when_slot_is_out_of_bounds() {
        let mut batch = BatchEncoder::new(0);
        let _ = batch.slot_mut(0);
    }

    #[test]
    #[should_panic(expected = "slot 1 >= batch_size 1")]
    fn batch_encoder_copy_from_encoder_panics_when_slot_is_out_of_bounds() {
        let mut batch = BatchEncoder::new(1);
        let src = ObservationEncoder::new();
        batch.copy_from_encoder(1, &src);
    }
}
