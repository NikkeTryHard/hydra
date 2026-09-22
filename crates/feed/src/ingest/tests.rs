// ---------------------------------------------------------------------------
// Tests (unit scope; oracle histogram agreement runs at P1-DONE)
// ---------------------------------------------------------------------------

use super::kinds::boundary_class;
use super::*;
use crate::gate::{
    GateReject, REASON_BLANK_LINE, REASON_BOUNDARY, REASON_FRAMING, REASON_UNKNOWN_KIND,
    REASON_WALL_PERM,
};

const GOOD_B: &str = include_str!("../../../../crates/tests/fixtures/s4/good-b.jsonl");
const WALLED_SYNTH: &str =
    include_str!("../../../../crates/tests/fixtures/s7/walled-synth.jsonl");
const WALLED_REAL: &str =
    include_str!("../../../../crates/tests/fixtures/s7/walled-real.jsonl");
const WALLED_ANKAN: &str =
    include_str!("../../../../crates/tests/fixtures/s7/walled-ankan.jsonl");
const WALLED_KAKAN: &str =
    include_str!("../../../../crates/tests/fixtures/s7/walled-kakan.jsonl");
const WALLED_POST: &str =
    include_str!("../../../../crates/tests/fixtures/s7/walled-post-terminal.jsonl");
const WALLED_TRUNC: &str =
    include_str!("../../../../crates/tests/fixtures/s7/walled-truncated.jsonl");
const Q_UNKNOWN: &str =
    include_str!("../../../../crates/tests/fixtures/s4/q-unknown-event.jsonl");
const Q_NO_PAI: &str =
    include_str!("../../../../crates/tests/fixtures/s4/q-tile-conservation.jsonl");
const Q_FRAMING: &str =
    include_str!("../../../../crates/tests/fixtures/s4/q-framing.jsonl");
const Q_TRUNCATED: &str =
    include_str!("../../../../crates/tests/fixtures/s4/q-truncated.jsonl");
const Q_WALL: &str =
    include_str!("../../../../crates/tests/fixtures/s4/q-wall-bearing.jsonl");

fn frame_both(
    bytes: &[u8],
    object_id: u32,
    game_idx: u32,
) -> (Result<ProjGame, GateReject>, Result<ProjGame, GateReject>) {
    // Separate arenas: each game borrows its own side (flip discipline).
    let mut a = Vec::new();
    let mut b = Vec::new();
    let r0 = frame_spans_serde(&mut a, bytes, object_id, game_idx);
    let r1 = frame_spans_fast(&mut b, bytes, object_id, game_idx);
    // Owned projections (lifetimes differ by arena); borrow ends on move.
    (proj(r0), proj(r1))
}

fn proj(r: Result<FramedGame<'_>, GateReject>) -> Result<ProjGame, GateReject> {
    r.map(|g| ProjGame {
        game_idx: g.game_idx,
        object_id: g.object_id,
        game_key: g.game_key,
        wall: g.wall,
        events: g.events.iter().map(ProjEvent::of).collect(),
    })
}

/// Concatenate fixture texts into one multi-game file image (test-only).
fn cat(parts: &[&str]) -> Vec<u8> {
    let mut v = Vec::new();
    for p in parts {
        v.extend_from_slice(p.as_bytes());
    }
    v
}

/// Paired per-path verdicts from [`frame_file_both`]
/// (named alias for clippy::type_complexity).
type BothVerdicts = (
    Vec<Result<ProjGame, GateReject>>,
    Vec<Result<ProjGame, GateReject>>,
);

/// Frame one file on both paths; project every per-game verdict so the
/// two paths compare owned (arenas differ). Pins the staged-count
/// invariant (`count == results.len()`) on every input.
fn frame_file_both(bytes: &[u8], object_id: u32, base: u32) -> BothVerdicts {
    let mut a = Vec::new();
    let mut b = Vec::new();
    let (r0, n0) = frame_file_serde(&mut a, bytes, object_id, base);
    let (r1, n1) = frame_file_fast(&mut b, bytes, object_id, base);
    assert_eq!(n0, u32::try_from(r0.len()).unwrap());
    assert_eq!(n1, u32::try_from(r1.len()).unwrap());
    (
        r0.into_iter().map(proj).collect(),
        r1.into_iter().map(proj).collect(),
    )
}

#[derive(Debug, PartialEq, Eq)]
struct ProjEvent {
    kind: u8,
    actor: u8,
    target: u8,
    pai: u8,
    consumed: ([u8; 4], u8),
    tsumogiri: bool,
    dora_span: Option<Vec<u8>>,
    wall_span: Option<Vec<u8>>,
}

impl ProjEvent {
    fn of(e: &StackedEvent<'_>) -> Self {
        Self {
            kind: e.kind,
            actor: e.actor,
            target: e.target,
            pai: e.pai,
            consumed: e.consumed,
            tsumogiri: e.tsumogiri,
            dora_span: e.dora_span.map(|s| s.to_vec()),
            wall_span: e.wall_span.map(|s| s.to_vec()),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct ProjGame {
    game_idx: u32,
    object_id: u32,
    game_key: u64,
    wall: Option<[u8; 136]>,
    events: Vec<ProjEvent>,
}

#[test]
fn f1_smoke_baseline_shape() {
    let mut arena = Vec::new();
    let g = frame_spans(&mut arena, GOOD_B.as_bytes(), 7, 3).unwrap();
    assert_eq!(g.events.len(), 14);
    assert_eq!(g.wall, None);
    assert_eq!(g.game_idx, 3);
    assert_eq!(g.object_id, 7);
    let kinds: Vec<u8> = g.events.iter().map(|e| e.kind).collect();
    assert_eq!(
        kinds,
        [
            KIND_START,
            KIND_START_KYOKU,
            KIND_TSUMO,
            KIND_DAHAI,
            KIND_TSUMO,
            KIND_DAHAI,
            KIND_TSUMO,
            KIND_DAHAI,
            KIND_TSUMO,
            KIND_DAHAI,
            KIND_TSUMO,
            KIND_HORA,
            KIND_END_KYOKU,
            KIND_END
        ]
    );
    // tsumogiri + hora seats.
    assert!(g.events[3].tsumogiri);
    assert_eq!(g.events[11].actor, 0);
    assert_eq!(g.events[11].target, 0);
    // Deterministic key.
    let mut arena2 = Vec::new();
    let g2 = frame_spans(&mut arena2, GOOD_B.as_bytes(), 7, 3).unwrap();
    assert_eq!(g.game_key, g2.game_key);
}

#[test]
fn f3_walled_shapes() {
    for text in [WALLED_SYNTH, WALLED_REAL, WALLED_ANKAN, WALLED_KAKAN] {
        let mut arena = Vec::new();
        let g = frame_spans(&mut arena, text.as_bytes(), 1, 0).unwrap();
        let wall = g.wall.unwrap();
        let mut sorted = wall;
        sorted.sort_unstable();
        for (i, v) in sorted.iter().enumerate() {
            assert_eq!(*v as usize, i);
        }
        assert!(g.events[0].wall_span.is_some());
        assert!(g.events[1].wall_span.is_none());
    }
    // Consumed capture on the ankan/pon lines.
    let mut arena = Vec::new();
    let g = frame_spans(&mut arena, WALLED_ANKAN.as_bytes(), 1, 0).unwrap();
    let ankan = g.events.iter().find(|e| e.kind == KIND_ANKAN).unwrap();
    assert_eq!(ankan.consumed, ([0, 0, 0, 0], 4));
    let mut arena = Vec::new();
    let g = frame_spans(&mut arena, WALLED_KAKAN.as_bytes(), 1, 0).unwrap();
    let pon = g.events.iter().find(|e| e.kind == KIND_PON).unwrap();
    assert_eq!(pon.actor, 1);
    assert_eq!(pon.target, 0);
    assert_eq!(pon.consumed, ([108, 108, 0, 0], 2));
}

#[test]
fn baseline_and_fast_agree_on_pinned_fixtures() {
    // Accept verdicts must project identically (fields + spans + wall).
    for text in [
        GOOD_B,
        WALLED_SYNTH,
        WALLED_REAL,
        WALLED_ANKAN,
        WALLED_KAKAN,
        WALLED_POST,
        WALLED_TRUNC,
        Q_UNKNOWN,
        Q_NO_PAI,
        Q_WALL,
    ] {
        let (r0, r1) = frame_both(text.as_bytes(), 11, 5);
        assert_eq!(r0, r1);
        assert!(r0.is_ok());
    }
    // Reject verdicts must agree exactly (reason + index).
    for text in [Q_FRAMING, Q_TRUNCATED] {
        let (r0, r1) = frame_both(text.as_bytes(), 11, 5);
        assert!(r0.is_err());
        assert_eq!(r0, r1);
    }
}

#[test]
fn framing_rejects_with_oracle_reasons() {
    let mut arena = Vec::new();
    // Blank line (q-framing).
    let e = frame_spans(&mut arena, Q_FRAMING.as_bytes(), 0, 0).unwrap_err();
    assert_eq!(e.reason, REASON_BLANK_LINE);
    assert_eq!(e.event_idx, 1);
    // Truncated (q-truncated: last record is start_kyoku, not end_game).
    let mut arena = Vec::new();
    let e = frame_spans(&mut arena, Q_TRUNCATED.as_bytes(), 0, 0).unwrap_err();
    assert_eq!(e.reason, REASON_BOUNDARY);
    assert_eq!(e.event_idx, 1);
    // Missing type.
    let mut arena = Vec::new();
    let e = frame_spans(
        &mut arena,
        b"{\"type\":\"start_game\"}\n{\"x\":1}\n{\"type\":\"end_game\"}\n",
        0,
        0,
    )
    .unwrap_err();
    assert_eq!(e.reason, REASON_UNKNOWN_KIND);
    assert_eq!(e.event_idx, 1);
    // Bad boundary.
    let mut arena = Vec::new();
    let e = frame_spans(
        &mut arena,
        b"{\"type\":\"dahai\",\"actor\":0,\"pai\":\"1m\"}\n{\"type\":\"end_game\"}\n",
        0,
        0,
    )
    .unwrap_err();
    assert_eq!(e.reason, REASON_BOUNDARY);
    // Double canonical start.
    let mut arena = Vec::new();
    let e = frame_spans(
        &mut arena,
        b"{\"type\":\"start_game\"}\n{\"type\":\"start_game\"}\n{\"type\":\"end_game\"}\n",
        0,
        0,
    )
    .unwrap_err();
    assert_eq!(e.reason, REASON_BOUNDARY);
    // Bad tile.
    let mut arena = Vec::new();
    let e = frame_spans(
        &mut arena,
        b"{\"type\":\"start_game\"}\n{\"type\":\"tsumo\",\"actor\":0,\"pai\":\"9x\"}\n{\"type\":\"end_game\"}\n",
        0,
        0,
    )
    .unwrap_err();
    assert_eq!(e.reason, REASON_FRAMING);
    assert_eq!(e.event_idx, 1);
    // Bad wall (duplicate id 0, missing 135).
    let mut wall = [0u8; 136];
    for (i, w) in wall.iter_mut().enumerate() {
        *w = u8::try_from(i % 135).unwrap();
    }
    assert!(!crate::gate::is_wall_perm136(&wall));
}

#[test]
fn unknown_kind_stacks_as_other_and_missing_pai_is_none() {
    let mut arena = Vec::new();
    let g = frame_spans(&mut arena, Q_UNKNOWN.as_bytes(), 0, 0).unwrap();
    assert_eq!(g.events[2].kind, KIND_OTHER);
    let mut arena = Vec::new();
    let g = frame_spans(&mut arena, Q_NO_PAI.as_bytes(), 0, 0).unwrap();
    assert_eq!(g.events[2].pai, NO_PAI);
}

#[test]
fn compressed_inputs_frame_identically() {
    let raw = GOOD_B.as_bytes();
    let zst = zstd::encode_all(raw, 3).unwrap();
    assert_eq!(&zst[..4], &[0x28, 0xB5, 0x2F, 0xFD]);
    let mut a = Vec::new();
    let mut b = Vec::new();
    let g0 = frame_spans_serde(&mut a, &zst, 9, 9).unwrap();
    let g1 = frame_spans_fast(&mut b, &zst, 9, 9).unwrap();
    assert_eq!(g0.events.len(), 14);
    assert_eq!(g0.events.len(), g1.events.len());
    assert_eq!(g0.wall, g1.wall);
    // gzip magic path.
    let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
    std::io::Write::write_all(&mut enc, raw).unwrap();
    let gz = enc.finish().unwrap();
    let mut c = Vec::new();
    let g2 = frame_spans(&mut c, &gz, 9, 9).unwrap();
    assert_eq!(g2.events.len(), 14);
    // Corrupt magic-present payload fails closed.
    let mut d = Vec::new();
    let e = frame_spans(&mut d, &[0x28, 0xB5, 0x2F, 0xFD, 0x00], 0, 0).unwrap_err();
    assert_eq!(e.reason, REASON_FRAMING);
}

#[test]
fn f6_packager_zst_rejects_like_decode_py() {
    // F6 (manifest 8a2f) decompresses to a single typeless envelope
    // line: decode.py rejects (missing `type`); both paths agree.
    const ZST: &[u8] = include_bytes!(
        "../../../../crates/packager/tests/fixtures/c.mjai.json.zst"
    );
    let mut a = Vec::new();
    let mut b = Vec::new();
    let e0 = frame_spans_serde(&mut a, ZST, 0, 0).unwrap_err();
    let e1 = frame_spans_fast(&mut b, ZST, 0, 0).unwrap_err();
    assert_eq!(e0.reason, REASON_UNKNOWN_KIND);
    assert_eq!(e0, e1);
}

#[test]
fn file_single_game_matches_spans_path() {
    // No special-casing: one game frames exactly like frame_spans(idx 0).
    let bytes = GOOD_B.as_bytes();
    let mut a = Vec::new();
    let mut c = Vec::new();
    let (mut f, n) = frame_file(&mut a, bytes, 7, 0);
    assert_eq!((f.len(), n), (1, 1));
    let g = frame_spans(&mut c, bytes, 7, 0).unwrap();
    let fg = f.remove(0).unwrap();
    assert_eq!(fg.events.len(), 14);
    assert_eq!(fg.events.len(), g.events.len());
    assert_eq!(fg.game_idx, 0);
    assert_eq!(fg.object_id, 7);
    assert_eq!(fg.wall, None);
    assert_eq!(fg.game_key, g.game_key);
    let kinds: Vec<u8> = fg.events.iter().map(|e| e.kind).collect();
    let kinds0: Vec<u8> = g.events.iter().map(|e| e.kind).collect();
    assert_eq!(kinds, kinds0);
}

#[test]
fn file_two_games_accept_in_order() {
    let bytes = cat(&[GOOD_B, WALLED_SYNTH]);
    let mut a = Vec::new();
    let (f, n) = frame_file(&mut a, &bytes, 7, 0);
    assert_eq!((f.len(), n), (2, 2));
    let g0 = f[0].as_ref().unwrap();
    let g1 = f[1].as_ref().unwrap();
    assert_eq!(g0.events.len(), 14);
    assert_eq!(g1.events.len(), 13);
    assert_eq!((g0.game_idx, g1.game_idx), (0, 1));
    assert_eq!(g0.wall, None);
    assert!(g1.wall.is_some());
    assert_ne!(g0.game_key, g1.game_key);
}

#[test]
fn file_mixed_valid_invalid_quarantine_independent() {
    // The blank-line game fails alone; siblings still sample.
    let bytes = cat(&[GOOD_B, Q_FRAMING, GOOD_B]);
    let mut a = Vec::new();
    let (f, n) = frame_file(&mut a, &bytes, 7, 0);
    assert_eq!((f.len(), n), (3, 3));
    assert_eq!(f[0].as_ref().unwrap().events.len(), 14);
    assert_eq!(f[2].as_ref().unwrap().events.len(), 14);
    assert_eq!(f[2].as_ref().unwrap().game_idx, 2);
    match &f[1] {
        Err(e) => {
            assert_eq!(e.reason, REASON_BLANK_LINE);
            assert_eq!(e.event_idx, 1);
        }
        Ok(_) => panic!(),
    }
}

#[test]
fn file_f8_shape_22_games() {
    // F8 packs 22 games/file: every game samples independently.
    let parts = [GOOD_B; 22];
    let bytes = cat(&parts);
    let mut a = Vec::new();
    let (f, n) = frame_file(&mut a, &bytes, 7, 0);
    assert_eq!((f.len(), n), (22, 22));
    for (i, r) in f.iter().enumerate() {
        let g = r.as_ref().unwrap();
        assert_eq!(g.events.len(), 14);
        assert_eq!(g.game_idx, u32::try_from(i).unwrap());
    }
}

#[test]
fn file_leading_junk_chunk_fails_alone() {
    // F6-style typeless envelope line, then a good game.
    let bytes = cat(&[
        "{\"origin\":\"precompressed\",\"note\":\"byte-copy contract\"}\n",
        GOOD_B,
    ]);
    let mut a = Vec::new();
    let (f, n) = frame_file(&mut a, &bytes, 7, 0);
    assert_eq!((f.len(), n), (2, 2));
    match &f[0] {
        Err(e) => {
            assert_eq!(e.reason, REASON_UNKNOWN_KIND);
            assert_eq!(e.event_idx, 0);
        }
        Ok(_) => panic!(),
    }
    assert_eq!(f[1].as_ref().unwrap().events.len(), 14);
    assert_eq!(f[1].as_ref().unwrap().game_idx, 1);
}

#[test]
fn file_unterminated_tail_fails_last_only() {
    let stripped = GOOD_B.strip_suffix('\n').unwrap();
    let bytes = cat(&[GOOD_B, stripped]);
    let mut a = Vec::new();
    let (f, n) = frame_file(&mut a, &bytes, 7, 0);
    assert_eq!((f.len(), n), (2, 2));
    assert_eq!(f[0].as_ref().unwrap().events.len(), 14);
    match &f[1] {
        Err(e) => assert_eq!(e.reason, REASON_FRAMING),
        Ok(_) => panic!(),
    }
}

#[test]
fn file_baseline_fast_agree() {
    let stripped = GOOD_B.strip_suffix('\n').unwrap();
    let inputs = [
        cat(&[GOOD_B]),
        cat(&[GOOD_B, WALLED_SYNTH]),
        cat(&[GOOD_B, Q_FRAMING, GOOD_B]),
        cat(&[GOOD_B; 22]),
        cat(&[
            "{\"origin\":\"precompressed\",\"note\":\"byte-copy contract\"}\n",
            GOOD_B,
        ]),
        cat(&[GOOD_B, stripped]),
    ];
    for bytes in &inputs {
        let (r0, r1) = frame_file_both(bytes, 11, 0);
        assert_eq!(r0, r1);
    }
    // Nonzero base also agrees (exercises the offset path on both).
    let (r0, r1) = frame_file_both(&inputs[2], 11, 41);
    assert_eq!(r0, r1);
    assert_eq!(r0.len(), 3);
    // Plus every pinned single-game fixture through the file path.
    for text in [
        GOOD_B,
        WALLED_SYNTH,
        WALLED_REAL,
        WALLED_ANKAN,
        WALLED_KAKAN,
        WALLED_POST,
        WALLED_TRUNC,
        Q_UNKNOWN,
        Q_NO_PAI,
        Q_WALL,
    ] {
        let (r0, r1) = frame_file_both(text.as_bytes(), 11, 0);
        assert_eq!(r0.len(), 1);
        assert_eq!(r0, r1);
        assert!(r0[0].is_ok());
    }
}

#[test]
fn file_base_offsets_key_globally_unique() {
    // Bridge advances game_seq by the staged count: consecutive files
    // must not collide. Rejects occupy keyspace too.
    fn idx_of(r: &Result<FramedGame<'_>, GateReject>) -> u32 {
        match r {
            Ok(g) => g.game_idx,
            Err(e) => e.game_idx,
        }
    }
    let bytes = cat(&[GOOD_B, Q_FRAMING, GOOD_B]);
    let mut a = Vec::new();
    let mut b = Vec::new();
    let (f0, n0) = frame_file(&mut a, &bytes, 7, 100);
    let (f1, n1) = frame_file(&mut b, &bytes, 7, 103);
    assert_eq!((n0, n1), (3, 3));
    let i0: Vec<u32> = f0.iter().map(idx_of).collect();
    let i1: Vec<u32> = f1.iter().map(idx_of).collect();
    assert_eq!(i0, [100, 101, 102]);
    assert_eq!(i1, [103, 104, 105]);
    // Same content, distinct keys: no cross-file collision.
    let k0 = f0[0].as_ref().unwrap().game_key;
    let k1 = f1[0].as_ref().unwrap().game_key;
    assert_ne!(k0, k1);
    // The middle reject sits at its keyed slot on both files.
    match &f0[1] {
        Err(e) => {
            assert_eq!(e.game_idx, 101);
            assert_eq!(e.reason, REASON_BLANK_LINE);
        }
        Ok(_) => panic!(),
    }
}

#[test]
fn file_base_saturates_without_wrap() {
    let bytes = cat(&[GOOD_B, GOOD_B]);
    let mut a = Vec::new();
    let (f, n) = frame_file(&mut a, &bytes, 7, u32::MAX);
    assert_eq!((f.len(), n), (2, 2));
    assert_eq!(f[0].as_ref().unwrap().game_idx, u32::MAX);
    assert_eq!(f[1].as_ref().unwrap().game_idx, u32::MAX);
    assert!(f.iter().all(|r| r.is_ok()));
}

#[test]
fn chunk_ranges_cover_bytes_exactly_once() {
    let bytes = cat(&[GOOD_B, Q_FRAMING, WALLED_SYNTH]);
    let mut a = Vec::new();
    let ranges = file_chunk_ranges(&mut a, &bytes, 0).unwrap();
    assert_eq!(ranges.len(), 3);
    assert_eq!(ranges[0].0, 0);
    assert_eq!(ranges[2].1, a.len());
    for w in ranges.windows(2) {
        assert_eq!(w[0].1, w[1].0);
    }
    for (s, e) in &ranges {
        assert!(s < e);
    }
}

#[test]
fn chunk_ranges_match_frame_file_chunks() {
    // Slicing the filled arena per range and framing each slice with
    // the single-game entry point equals the file path chunk-for-chunk
    // (verdicts, counts, walls, keys) — the staging seam is exact.
    let bytes = cat(&[GOOD_B, Q_FRAMING, WALLED_SYNTH]);
    let mut a = Vec::new();
    let ranges = file_chunk_ranges(&mut a, &bytes, 0).unwrap();
    let mut f = Vec::new();
    let (file_res, n) = frame_file(&mut f, &bytes, 7, 0);
    assert_eq!((ranges.len(), file_res.len(), n), (3, 3, 3));
    for (i, (s, e)) in ranges.iter().enumerate() {
        let mut c = Vec::new();
        let single = frame_spans(&mut c, &a[*s..*e], 7, u32::try_from(i).unwrap());
        match (&single, &file_res[i]) {
            (Ok(g1), Ok(g2)) => {
                assert_eq!(g1.events.len(), g2.events.len());
                assert_eq!(g1.wall, g2.wall);
                assert_eq!(g1.game_key, g2.game_key);
            }
            (Err(e1), Err(e2)) => assert_eq!(e1, e2),
            _ => panic!(),
        }
    }
}

#[test]
fn chunk_ranges_fail_closed_on_file_errors() {
    let mut a = Vec::new();
    let e = file_chunk_ranges(&mut a, b"", 5).unwrap_err();
    assert_eq!((e.game_idx, e.reason), (5, REASON_FRAMING));
    let mut b = Vec::new();
    let e = file_chunk_ranges(&mut b, &[0x28, 0xB5, 0x2F, 0xFD, 0x00], 5).unwrap_err();
    assert_eq!((e.game_idx, e.reason), (5, REASON_FRAMING));
}

#[test]
fn double_buffer_flip_never_clears_borrowed() {
    // Flip discipline: fill-one/clear-other across two arenas.
    let mut a = Vec::new();
    let mut b = Vec::new();
    let g0 = frame_spans(&mut a, GOOD_B.as_bytes(), 1, 0).unwrap();
    assert_eq!(g0.events.len(), 14);
    let g1 = frame_spans(&mut b, WALLED_SYNTH.as_bytes(), 2, 1).unwrap();
    assert!(g1.wall.is_some());
    // First game still fully readable while the second is framed.
    assert_eq!(g0.events[11].kind, KIND_HORA);
    assert_eq!(g0.events.len(), 14);
}

#[test]
fn kind_lut_matches_vocab_consts() {
    for t in START_TYPES {
        assert_eq!(kind_from_bytes(t.as_bytes()), KIND_START);
    }
    for t in END_TYPES {
        assert_eq!(kind_from_bytes(t.as_bytes()), KIND_END);
    }
    for (t, k) in [
        ("dahai", KIND_DAHAI),
        ("chi", KIND_CHI),
        ("pon", KIND_PON),
        ("daiminkan", KIND_DAIMINKAN),
        ("ankan", KIND_ANKAN),
        ("kakan", KIND_KAKAN),
        ("hora", KIND_HORA),
    ] {
        assert_eq!(kind_from_bytes(t.as_bytes()), k);
    }
    assert_eq!(kind_from_bytes(b"frobnicate"), KIND_OTHER);
    assert_eq!(boundary_class(b"start_game"), 1);
    assert_eq!(boundary_class(b"start"), 2);
    assert_eq!(boundary_class(b"end_game"), 3);
    assert_eq!(boundary_class(b"end"), 4);
    assert_eq!(boundary_class(b"dahai"), 0);
}

#[test]
fn wall_perm_helper_agrees_with_inline_check() {
    let mut arena = Vec::new();
    let g = frame_spans(&mut arena, WALLED_SYNTH.as_bytes(), 0, 0).unwrap();
    let wall = g.wall.unwrap();
    assert!(crate::gate::is_wall_perm136(&wall));
    assert!(!crate::gate::is_wall_perm136(&[0u8; 136]));
    let _ = REASON_WALL_PERM;
}
