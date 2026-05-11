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
