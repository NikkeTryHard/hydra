#![allow(clippy::expect_used, clippy::unwrap_used)] // integration harness: fixture-load expects are the test idiom
//! P4-B cold-shard acceptance: hot stage → compact write → mmap read →
//! collate → G2 hot-subset identity, plus geometry, tamper, and privileged
//! parquet-join proofs. Cold only; `format!` below is fixture scaffolding.

use std::fs;
use std::path::PathBuf;

use hydra_feed::fill::{N_PLANES, Scratch, StageJob, row_bytes, stage_one};
use hydra_feed::ledger::PLANE_LEGAL;
use hydra_shard::{
    CollateScratch, ColdSidecar, HotRowIter, ShardReader, ShardWriter, compact_row_bytes,
    decode_row_into_planes, decision_ids_for_game, dense_full26_row_bytes, encode_compact_row,
    join_privileged, parse_privileged_label, read_privileged_parquet, verify_hot_subset,
};

// ---------------------------------------------------------------------------
// F1 fixture (wall-less shape: 5 tsumogiri discards; mirrors feed fill_tests)
// ---------------------------------------------------------------------------

const TEHAIS: [&[&str]; 4] = [
    &[
        "1m", "1m", "1m", "1m", "5mr", "5m", "5m", "5m", "9m", "9m", "9m", "9m", "4p",
    ],
    &[
        "2m", "2m", "2m", "2m", "6m", "6m", "6m", "6m", "1p", "1p", "1p", "1p", "4p",
    ],
    &[
        "3m", "3m", "3m", "3m", "7m", "7m", "7m", "7m", "2p", "2p", "2p", "2p", "4p",
    ],
    &[
        "4m", "4m", "4m", "4m", "8m", "8m", "8m", "8m", "3p", "3p", "3p", "3p", "4p",
    ],
];

fn kyoku_line(tehais: &[&[&str]], oya: u8) -> String {
    let mut seats = Vec::new();
    for hand in tehais {
        let tiles: Vec<String> = hand.iter().map(|t| format!("{t:?}")).collect();
        seats.push(format!("[{}]", tiles.join(",")));
    }
    format!(
        "{{\"type\":\"start_kyoku\",\"bakaze\":\"E\",\"dora_marker\":\"F\",\"honba\":0,\"kyoku\":1,\"kyotaku\":0,\"oya\":{oya},\"scores\":[25000,25000,25000,25000],\"tehais\":[{}]}}",
        seats.join(",")
    )
}

fn game_text(pais: &[&str]) -> Vec<u8> {
    let mut lines = vec![
        "{\"type\":\"start_game\"}".to_string(),
        kyoku_line(&TEHAIS, 0),
    ];
    for (k, pai) in pais.iter().enumerate() {
        let actor = u8::try_from(k % 4).unwrap();
        lines.push(format!(
            "{{\"type\":\"tsumo\",\"actor\":{actor},\"pai\":{pai:?}}}"
        ));
        lines.push(format!(
            "{{\"type\":\"dahai\",\"actor\":{actor},\"pai\":{pai:?},\"tsumogiri\":true}}"
        ));
    }
    lines.push("{\"type\":\"end_game\"}".to_string());
    let mut out = lines.join("\n");
    out.push('\n');
    out.into_bytes()
}

fn tmp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("hydra-cold-{}-{name}", std::process::id()))
}

// ---------------------------------------------------------------------------
// Geometry legs (F9): exact strides + compact < dense
// ---------------------------------------------------------------------------

#[test]
fn geometry_compact_beats_dense() {
    assert_eq!(hydra_shard::LEGAL_PACKED_BYTES, 850);
    assert_eq!(hydra_shard::LEGAL_BITS, 6792);
    assert_eq!(compact_row_bytes(32), 1050);
    assert_eq!(compact_row_bytes(64), 1086);
    assert_eq!(compact_row_bytes(128), 1158);
    assert_eq!(compact_row_bytes(256), 1302);
    assert_eq!(dense_full26_row_bytes(256), 9558);
    assert!(compact_row_bytes(256) < dense_full26_row_bytes(256));
}

// ---------------------------------------------------------------------------
// G2: cold cache byte-identical to hot rows
// ---------------------------------------------------------------------------

#[test]
fn cold_cache_matches_hot_rows_g2() {
    let job = StageJob {
        game_idx: 7,
        object_id: 7,
        bytes: game_text(&["5pr", "5p", "5p", "5p", "6p"]),
        wall_digest: None,
    };
    let game = stage_one(&job).expect("F1 stages");
    assert_eq!(game.rows, 5);

    // Hot fill into caller buffers (exactly what the bridge hands torch).
    let mut scratch = Scratch::new();
    scratch.commit(&game);
    // History lens are tiny: bucket 32. Fill once with over-capacity caps to
    // learn t_len, then fill exactly (mirrors the bridge two-step).
    let t_len = {
        let mut big: [Vec<u8>; N_PLANES] = Default::default();
        let mut caps = [0usize; N_PLANES];
        let mut ptrs = [0u64; N_PLANES];
        let mut i = 0usize;
        while i < N_PLANES {
            big[i] = vec![0u8; 1 << 20];
            caps[i] = big[i].len();
            ptrs[i] = big[i].as_mut_ptr() as u64;
            i += 1;
        }
        // NOTE: this consumes the scratch; re-commit below for the real fill.
        let out = scratch.fill_pinned(&ptrs, &caps).expect("big fill");
        assert_eq!(out.rows, 5);
        out.t_len
    };
    assert_eq!(t_len, 32);
    scratch.commit(&game);
    let rows = game.rows as usize;
    let mut hot: [Vec<u8>; N_PLANES] = Default::default();
    let mut caps = [0usize; N_PLANES];
    let mut ptrs = [0u64; N_PLANES];
    let mut i = 0usize;
    while i < N_PLANES {
        hot[i] = vec![0xABu8; rows * row_bytes(i, t_len)];
        caps[i] = hot[i].len();
        ptrs[i] = hot[i].as_mut_ptr() as u64;
        i += 1;
    }
    let out = scratch.fill_pinned(&ptrs, &caps).expect("exact fill");
    assert_eq!(out.rows, 5);
    assert_eq!(out.t_len, 32);

    // Cold encode (neutral sidecars: legal = exactly the hot ids).
    let mut compact: Vec<u8> = Vec::new();
    for (seq, hot_row) in HotRowIter::new(&game).enumerate() {
        let hot_row = hot_row.expect("hot row");
        let sidecar = ColdSidecar::neutral_from_hot(&hot_row).expect("neutral");
        encode_compact_row(
            7,
            u32::try_from(seq).unwrap(),
            &hot_row,
            &sidecar,
            t_len,
            &mut compact,
        )
        .expect("encode");
    }
    assert_eq!(compact.len(), 5 * compact_row_bytes(t_len));

    // Write → mmap read (header + exact-len + sha verified on open).
    let path = tmp_path("g2.hshard");
    let mut writer = ShardWriter::create(&path, t_len, None).expect("create");
    assert!(writer.fits(5));
    writer.append_game(&compact).expect("append");
    let done = writer.finish().expect("finish");
    assert_eq!(done.rows, 5);
    assert!(done.payload_sha256.starts_with("sha256:"));

    let reader = ShardReader::open(&path).expect("open");
    assert_eq!(reader.rows(), 5);
    assert_eq!(reader.t_bucket(), 32);
    assert_eq!(reader.row_bytes(), compact_row_bytes(32));

    // Collate + G2: byte-identical hot subset (ints EXACT post-cast).
    let mut scratch = CollateScratch::new();
    scratch.collate_into(&reader).expect("collate");
    let batch = scratch.batch();
    assert_eq!(batch.rows, 5);
    assert_eq!(batch.keys_game, vec![7u32; 5]);
    assert_eq!(batch.keys_seq, vec![0, 1, 2, 3, 4]);
    // Neutral sidecar: cold mask popcount equals the hot legal_len exactly.
    let legal_plane = hydra_shard::full26_plane("legal_mask").expect("legal plane");
    let mut r = 0usize;
    while r < 5 {
        let mask = &batch.planes[legal_plane][r * 6792..(r + 1) * 6792];
        let pop: usize = mask.iter().map(|b| *b as usize).sum();
        let hot_len = usize::try_from(i64::from_le_bytes(
            hot[PLANE_LEGAL][r * 136 + 128..r * 136 + 136]
                .try_into()
                .expect("len bytes"),
        )).unwrap();
        assert_eq!(pop, hot_len);
        r += 1;
    }
    assert_eq!(
        verify_hot_subset(batch, &hot, t_len).expect("G2 green"),
        5
    );

    // Scratch swap reuse: caps recycle, second collate identical.
    let mut spare = hydra_shard::CollatedBatch::default();
    scratch.swap_batch(&mut spare);
    assert_eq!(spare.rows, 5);
    assert_eq!(scratch.batch().rows, 0);
    scratch.collate_into(&reader).expect("re-collate");
    assert_eq!(
        verify_hot_subset(scratch.batch(), &hot, t_len).expect("G2 again"),
        5
    );

    fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Tamper legs: bad magic / truncation / payload flip fail closed
// ---------------------------------------------------------------------------

#[test]
fn shard_tamper_fails_closed() {
    let job = StageJob {
        game_idx: 1,
        object_id: 1,
        bytes: game_text(&["5pr", "5p", "5p", "5p", "6p"]),
        wall_digest: None,
    };
    let game = stage_one(&job).expect("F1 stages");
    let mut compact: Vec<u8> = Vec::new();
    for (seq, hot_row) in HotRowIter::new(&game).enumerate() {
        let hot_row = hot_row.expect("hot row");
        let sidecar = ColdSidecar::neutral_from_hot(&hot_row).expect("neutral");
        encode_compact_row(
            1,
            u32::try_from(seq).unwrap(),
            &hot_row,
            &sidecar,
            32,
            &mut compact,
        )
        .expect("encode");
    }
    let path = tmp_path("tamper.hshard");
    let mut writer = ShardWriter::create(&path, 32, None).expect("create");
    writer.append_game(&compact).expect("append");
    writer.finish().expect("finish");
    let good = fs::read(&path).expect("read back");

    // Bad magic.
    let bad_path = tmp_path("tamper-magic.hshard");
    let mut bad = good.clone();
    bad[0] ^= 0xFF;
    fs::write(&bad_path, &bad).expect("write bad magic");
    let err = format!("{}", ShardReader::open(&bad_path).expect_err("bad magic"));
    assert!(err.contains("bad magic"), "got: {err}");

    // Truncation (exact-len assert).
    let trunc_path = tmp_path("tamper-trunc.hshard");
    fs::write(&trunc_path, &good[..good.len() - 1]).expect("write trunc");
    let err = format!("{}", ShardReader::open(&trunc_path).expect_err("trunc"));
    assert!(err.contains("length"), "got: {err}");

    // Payload flip (sha mismatch; header still parses).
    let flip_path = tmp_path("tamper-flip.hshard");
    let mut flip = good.clone();
    let last = flip.len() - 1;
    flip[last] ^= 0x01;
    fs::write(&flip_path, &flip).expect("write flip");
    let err = format!("{}", ShardReader::open(&flip_path).expect_err("flip"));
    assert!(err.contains("sha"), "got: {err}");

    // Torn append is rejected before any write.
    let torn_path = tmp_path("tamper-torn.hshard");
    let mut writer = ShardWriter::create(&torn_path, 32, None).expect("create");
    let err = format!(
        "{}",
        writer.append_game(&compact[..compact.len() - 7]).expect_err("torn")
    );
    assert!(err.contains("game-atomic"), "got: {err}");

    for p in [&path, &bad_path, &trunc_path, &flip_path, &torn_path] {
        fs::remove_file(p).ok();
    }
}

// ---------------------------------------------------------------------------
// Decode unit leg: compact → full-26 planes field geometry
// ---------------------------------------------------------------------------

#[test]
fn decode_full26_plane_geometry() {
    let job = StageJob {
        game_idx: 3,
        object_id: 3,
        bytes: game_text(&["5pr", "5p", "5p", "5p", "6p"]),
        wall_digest: None,
    };
    let game = stage_one(&job).expect("F1 stages");
    let hot_row = hydra_shard::hot_row_at(&game.planes, &game.hist_lens, 0).expect("row 0");
    let sidecar = ColdSidecar::neutral_from_hot(&hot_row).expect("neutral");
    let mut bytes: Vec<u8> = Vec::new();
    encode_compact_row(3, 0, &hot_row, &sidecar, 32, &mut bytes).expect("encode");
    let mut planes: [Vec<u8>; 26] = Default::default();
    let mut label: Vec<u8> = Vec::new();
    decode_row_into_planes(&bytes, 32, &mut planes, &mut label).expect("decode");
    assert_eq!(label.len(), 8);
    // Spot-check strides at T=32.
    let stride = |name: &str| {
        let p = hydra_shard::full26_plane(name).expect("field");
        hydra_shard::full26_plane_row_bytes(p, 32)
    };
    assert_eq!(planes[hydra_shard::full26_plane("actor").expect("f")].len(), 8);
    assert_eq!(stride("concealed_hand_counts"), 136);
    assert_eq!(stride("history_event_kind"), 256);
    assert_eq!(stride("history_mask"), 32);
    assert_eq!(stride("legal_mask"), 6792);
    assert_eq!(stride("scores"), 16);
    // dora verbatim + actor_seats == actor dup.
    assert_eq!(
        &planes[hydra_shard::full26_plane("dora_indicators").expect("f")],
        &game.planes[2][..20]
    );
    assert_eq!(
        &planes[hydra_shard::full26_plane("actor").expect("f")],
        &planes[hydra_shard::full26_plane("actor_seats").expect("f")]
    );
}

// ---------------------------------------------------------------------------
// Privileged parquet cold join (low-level fixture writer: RecordWriter-free)
// ---------------------------------------------------------------------------

fn write_string_parquet(path: &PathBuf, columns: &[(&str, Vec<Option<String>>)]) {
    use parquet::data_type::ByteArray;
    use parquet::file::properties::WriterProperties;
    use parquet::file::writer::SerializedFileWriter;
    use parquet::schema::types::Type;
    use std::sync::Arc;

    let fields: Vec<Arc<Type>> = columns
        .iter()
        .map(|(name, _)| {
            Arc::new(
                Type::primitive_type_builder(name, parquet::basic::Type::BYTE_ARRAY)
                    .with_repetition(parquet::basic::Repetition::OPTIONAL)
                    .with_converted_type(parquet::basic::ConvertedType::UTF8)
                    .build()
                    .expect("field"),
            )
        })
        .collect();
    let schema = Arc::new(
        Type::group_type_builder("privileged_schema")
            .with_fields(fields)
            .build()
            .expect("schema"),
    );
    let file = fs::File::create(path).expect("create parquet");
    let props = Arc::new(WriterProperties::builder().build());
    let mut writer = SerializedFileWriter::new(file, schema, props).expect("writer");
    let mut row_group = writer.next_row_group().expect("row group");
    for (_, values) in columns {
        let mut col = row_group.next_column().expect("column").expect("typed");
        let mut data: Vec<ByteArray> = Vec::new();
        let mut defs: Vec<i16> = Vec::new();
        for v in values {
            match v {
                Some(s) => {
                    data.push(ByteArray::from(s.as_str()));
                    defs.push(1);
                }
                None => defs.push(0),
            }
        }
        col.typed::<parquet::data_type::ByteArrayType>()
            .write_batch(&data, Some(&defs), None)
            .expect("write batch");
        col.close().expect("close column");
    }
    row_group.close().expect("close rg");
    writer.close().expect("close");
}

#[test]
fn privileged_parquet_join_cold() {
    let ids = decision_ids_for_game("g7", 5);
    let label = |ranks: &str, split: &str, wall: Option<&str>| {
        let mut s = format!("{{\"ranks\":{ranks},\"split\":\"{split}\"");
        if let Some(w) = wall {
            s.push_str(&format!(",\"wall_id\":\"{w}\""));
        }
        s.push('}');
        s
    };
    let path = tmp_path("priv-000.parquet");
    write_string_parquet(
        &path,
        &[
            (
                "decision_id",
                vec![Some(ids[0].clone()), Some(ids[1].clone()), Some(ids[2].clone())],
            ),
            (
                "privileged_label",
                vec![
                    Some(label("[1,2,3,4]", "train", None)),
                    Some(label("[4,3,2,1]", "train", Some("sha256:abc"))),
                    Some(label("[2,1,4,3]", "train", None)),
                ],
            ),
            (
                "full_world",
                vec![Some("{}".to_string()), None, Some("{}".to_string())],
            ),
        ],
    );
    let labels = read_privileged_parquet(&path).expect("read");
    assert_eq!(labels.len(), 3);
    assert_eq!(labels[0].ranks, [1, 2, 3, 4]);
    assert_eq!(labels[0].split, "train");
    assert_eq!(labels[0].wall_id, None);
    assert_eq!(labels[1].ranks, [4, 3, 2, 1]);
    assert_eq!(labels[1].wall_id.as_deref(), Some("sha256:abc"));

    // Join over the 5 collated row keys: first three hit, last two miss.
    let hits = join_privileged(&ids, &labels);
    assert_eq!(hits, vec![Some(0), Some(1), Some(2), None, None]);
    // Unknown key misses.
    assert_eq!(join_privileged(&["g7:d9999".to_string()], &labels), vec![None]);

    // Label parser rejects bad ranks fail-closed.
    assert!(parse_privileged_label("g7:d0000", "{\"ranks\":[1,1,2,3],\"split\":\"train\"}").is_err());
    assert!(parse_privileged_label("g7:d0000", "{\"split\":\"train\"}").is_err());

    fs::remove_file(&path).ok();
}

#[test]
fn privileged_parquet_missing_column_fails_closed() {
    let path = tmp_path("priv-bad.parquet");
    write_string_parquet(
        &path,
        &[(
            "decision_id",
            vec![Some("g7:d0000".to_string())],
        )],
    );
    let err = format!("{}", read_privileged_parquet(&path).expect_err("missing col"));
    assert!(err.contains("privileged_label"), "got: {err}");
    fs::remove_file(&path).ok();
}
