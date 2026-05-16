use super::*;

#[test]
fn bit_writer_reader_crosses_byte_boundaries() {
    let cases = [0usize, 1, 5, 6, 13, 34, 46, 102, 136];
    for bits in cases {
        let mut bytes = vec![0u8; bits.div_ceil(8)];
        let mut writer = BitWriter::new(&mut bytes);
        for idx in 0..bits {
            writer.write_bit(idx % 3 == 0).expect("bit write");
        }
        let mut reader = BitReader::new(&bytes);
        for idx in 0..bits {
            assert_eq!(reader.read_bit().expect("bit read"), idx % 3 == 0);
        }
    }
}

#[test]
fn action_mask_round_trips_exact_one_bits() {
    let mut src = [0.0f32; HYDRA_ACTION_SPACE];
    src[0] = 1.0;
    src[7] = 1.0;
    src[45] = 1.0;
    let mut packed = [0u8; PACKED_ACTION_MASK_BYTES];
    pack_action_mask(&src, &mut packed).expect("pack action mask");
    let mut dst = [0.25f32; HYDRA_ACTION_SPACE];
    unpack_action_mask_into(&packed, &mut dst).expect("unpack action mask");
    assert_eq!(dst, src);
    assert_eq!(dst[0].to_bits(), 1.0f32.to_bits());
    assert_eq!(dst[1].to_bits(), 0.0f32.to_bits());
}

#[test]
fn spatial_mask_round_trips_102_bits() {
    let mut src = [0.0f32; 102];
    for idx in [0usize, 8, 31, 64, 101] {
        src[idx] = 1.0;
    }
    let mut packed = [0u8; PACKED_SPATIAL_MASK_BYTES];
    pack_spatial_mask(&src, &mut packed).expect("pack spatial mask");
    let mut dst = [0.0f32; 102];
    unpack_spatial_mask_into(&packed, &mut dst).expect("unpack spatial mask");
    assert_eq!(dst, src);
}

#[test]
fn tile_counts_round_trip_counts_zero_to_four() {
    let counts = std::array::from_fn(|idx| (idx % 5) as u8);
    let mut packed = [0u8; TILE34_COUNT_BYTES];
    pack_tile_counts(&counts, &mut packed).expect("pack counts");
    let mut decoded = [0u8; 34];
    unpack_tile_counts(&packed, &mut decoded).expect("unpack counts");
    assert_eq!(decoded, counts);
}

#[test]
fn invalid_count_and_non_binary_mask_hard_error() {
    let mut counts = [0u8; 34];
    counts[33] = 5;
    let mut packed_counts = [0u8; TILE34_COUNT_BYTES];
    assert!(matches!(
        pack_tile_counts(&counts, &mut packed_counts),
        Err(CompactEncodeError::CountOutOfRange {
            index: 33,
            value: 5
        })
    ));

    let mut mask = [0.0f32; HYDRA_ACTION_SPACE];
    mask[12] = 0.5;
    let mut packed_mask = [0u8; PACKED_ACTION_MASK_BYTES];
    assert!(matches!(
        pack_action_mask(&mask, &mut packed_mask),
        Err(CompactEncodeError::NonBinaryMask { index: 12 })
    ));
}

#[test]
fn invalid_small_enum_range_hard_errors() {
    assert!(validate_u8_range("action", 45, 46).is_ok());
    assert!(matches!(
        validate_u8_range("action", 46, 46),
        Err(CompactEncodeError::ValueOutOfRange {
            name: "action",
            value: 46,
            max_exclusive: 46
        })
    ));
}
