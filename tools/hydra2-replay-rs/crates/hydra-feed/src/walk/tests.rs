// ---------------------------------------------------------------------------
// Tests (walk gates: F1 bitwise, ids, quarantine agreement).
// ---------------------------------------------------------------------------
use super::*;
use crate::gate::gate_game;
use crate::ingest::{KIND_OTHER, frame_spans};
use crate::ledger::{
    ACT_ANKAN, ACT_CHI, ACT_DAIMINKAN, ACT_DISCARD, ACT_KAKAN, ACT_PASS, ACT_PON,
    ACT_RIICHI_DISCARD, ACT_RON, ACT_TSUMO, ACT_TSUMOGIRI, GameCtx, H_CALL_WINDOW, Ledger,
    N_WALK_PLANES, PLANE_ACTOR, PLANE_CHOSEN, PLANE_CONCEALED, PLANE_DEALER, PLANE_DORA,
    PLANE_HIST_KIND, PLANE_LEGAL, PLANE_PHASE, PLANE_ROUND_WIND, PLANE_SCORES, RowSink,
    WALK_BARE_DORA, WALK_PAST_TERMINAL, WALK_TILE_CONSERVATION, WALK_TURN_ORDER,
    WALK_UNKNOWN_EVENT,
};

const TEHAIS_F1: [&[&str]; 4] = [
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

fn tsumo_line(actor: u8, pai: &str) -> String {
    format!("{{\"type\":\"tsumo\",\"actor\":{actor},\"pai\":{pai:?}}}")
}

fn dahai_line(actor: u8, pai: &str, tsumogiri: bool) -> String {
    format!("{{\"type\":\"dahai\",\"actor\":{actor},\"pai\":{pai:?},\"tsumogiri\":{tsumogiri}}}")
}

/// F1 golden content (wall-less): 5 tsumogiri discards, no hora.
fn f1_text() -> Vec<u8> {
    let mut lines = vec![
        "{\"type\":\"start_game\"}".to_string(),
        kyoku_line(&TEHAIS_F1, 0),
    ];
    for (actor, pai) in [(0, "5pr"), (1, "5p"), (2, "5p"), (3, "5p"), (0, "6p")] {
        lines.push(tsumo_line(actor, pai));
        lines.push(dahai_line(actor, pai, true));
    }
    lines.push("{\"type\":\"end_game\"}".to_string());
    let mut out = lines.join("\n");
    out.push('\n');
    out.into_bytes()
}

struct Walked {
    rows: u16,
    planes: [Vec<u8>; N_WALK_PLANES],
    hist_lens: Vec<u32>,
    ledger: Ledger,
}

fn walk_bytes(text: &[u8], game_idx: u32, digest: Option<(&str, std::sync::Arc<str>)>) -> Walked {
    let mut arena = Vec::new();
    let game = frame_spans(&mut arena, text, 7, game_idx).expect("frame");
    let verdict = gate_game(game_idx, &game.events).expect("gate ok");
    assert!(verdict.is_sample());
    let (wall_digest, digest_arc) = match digest {
        Some((d, a)) => (Some(d), Some(a)),
        None => (None, None),
    };
    let ctx = GameCtx {
        game_idx,
        wall_digest,
        digest_arc,
    };
    let mut ledger = Ledger::new();
    let mut planes: [Vec<u8>; N_WALK_PLANES] = Default::default();
    let mut hist_lens = Vec::new();
    let rows = {
        let mut sink = RowSink::new(&mut planes, &mut hist_lens);
        walk_game(&ctx, &game.events, &mut ledger, &mut sink).expect("walk")
    };
    Walked {
        rows,
        planes,
        hist_lens,
        ledger,
    }
}

fn rd_i64(plane: &[u8], row: usize) -> i64 {
    let o = row * 8;
    i64::from_le_bytes(plane[o..o + 8].try_into().unwrap())
}

fn rd_dora(plane: &[u8], row: usize, slot: usize) -> i32 {
    let o = row * 20 + slot * 4;
    i32::from_le_bytes(plane[o..o + 4].try_into().unwrap())
}

fn rd_score(plane: &[u8], row: usize, slot: usize) -> i32 {
    let o = row * 16 + slot * 4;
    i32::from_le_bytes(plane[o..o + 4].try_into().unwrap())
}

fn rd_u8(plane: &[u8], row: usize, slot: usize) -> u8 {
    plane[row * 34 + slot]
}

fn legal_of(w: &Walked, row: usize) -> Vec<u32> {
    let base = row * 136;
    let p = &w.planes[PLANE_LEGAL];
    let len = i64::from_le_bytes(p[base + 128..base + 136].try_into().unwrap()) as usize;
    let mut out = Vec::new();
    for k in 0..len {
        out.push(i32::from_le_bytes(p[base + k * 4..base + k * 4 + 4].try_into().unwrap()) as u32);
    }
    out
}

fn hist_of(w: &Walked, row: usize) -> Vec<i64> {
    let mut off = 0usize;
    for r in 0..row {
        off += w.hist_lens[r] as usize;
    }
    let p = &w.planes[PLANE_HIST_KIND];
    let mut out = Vec::new();
    for k in 0..w.hist_lens[row] as usize {
        out.push(i64::from_le_bytes(
            p[(off + k) * 8..(off + k) * 8 + 8].try_into().unwrap(),
        ));
    }
    out
}

#[test]
fn f1_five_decisions_collapsed_bitwise() {
    let w = walk_bytes(&f1_text(), 3, None);
    assert_eq!(w.rows, 5);
    // Wall-less collapse oracle pin: [192,193,193,193,196].
    let mut chosen = Vec::new();
    let mut actors = Vec::new();
    for r in 0..5 {
        chosen.push(rd_i64(&w.planes[PLANE_CHOSEN], r));
        actors.push(rd_i64(&w.planes[PLANE_ACTOR], r));
    }
    assert_eq!(chosen, [192, 193, 193, 193, 196]);
    assert_eq!(actors, [0, 1, 2, 3, 0]);
    // Scalars: draw phase, dealer 0, round wind E=0, seat winds 0..3.
    for r in 0..5 {
        assert_eq!(rd_i64(&w.planes[PLANE_PHASE], r), 1);
        assert_eq!(rd_i64(&w.planes[PLANE_DEALER], r), 0);
        assert_eq!(rd_i64(&w.planes[PLANE_ROUND_WIND], r), 0);
    }
    for r in 0..5 {
        for s in 0..4 {
            assert_eq!(rd_score(&w.planes[PLANE_SCORES], r, s), 25000);
        }
    }
    // Dora never revealed: all −1 (G3).
    for r in 0..5 {
        for k in 0..5 {
            assert_eq!(rd_dora(&w.planes[PLANE_DORA], r, k), -1);
        }
    }
    // Concealed row0 exact: 1m×4, 5m×4, 9m×4, 4p (draw excluded).
    let mut c0 = [0u8; 34];
    for t in 0..34 {
        c0[t] = rd_u8(&w.planes[PLANE_CONCEALED], 0, t);
    }
    assert_eq!(c0[0], 4);
    assert_eq!(c0[4], 4);
    assert_eq!(c0[8], 4);
    assert_eq!(c0[12], 1);
    // Histories: [game,round,advance,draw] then +2 per same-private
    // prefix; seat3's 5p discard opens a kamicha-chi window (3p×4+4p
    // held), so row3/row4 carry the call_window envelope.
    assert_eq!(w.hist_lens, [4, 6, 8, 11, 14]);
    assert_eq!(hist_of(&w, 0), [0, 1, 2, 3]);
    assert!(hist_of(&w, 3).contains(&(H_CALL_WINDOW as i64)));
    // Legal: sorted, non-empty, chosen ∈ set (G2 legality at fill).
    for r in 0..5 {
        let legal = legal_of(&w, r);
        assert!(!legal.is_empty());
        let mut sorted = legal.clone();
        sorted.sort_unstable();
        assert_eq!(legal, sorted);
        assert!(legal.contains(&(chosen[r] as u32)));
    }
    // Row0 offers the pool-first strings + twin.
    let l0 = legal_of(&w, 0);
    for id in [4, 20, 21, 36, 52, 56, 192] {
        assert!(l0.contains(&id), "row0 legal missing {id}: {l0:?}");
    }
}

#[test]
fn f1_walled_gate_agrees_on_claim_free_game() {
    // Walled regime reports true takes (wall oracle): seats 1-3 draw
    // pool takes 53, 54, 55, reported unfolded.
    let arc: std::sync::Arc<str> = std::sync::Arc::from("sha256:test-digest");
    let w = {
        let digest: &str = &arc;
        walk_bytes(&f1_text(), 3, Some((digest, std::sync::Arc::clone(&arc))))
    };
    assert_eq!(w.rows, 5);
    let chosen: Vec<i64> = (0..5).map(|r| rd_i64(&w.planes[PLANE_CHOSEN], r)).collect();
    assert_eq!(chosen, [192, 193, 194, 195, 196]);
}

#[test]
fn chosen_id_arithmetic_pinned() {
    assert_eq!(chosen_id(ACT_PASS, 0, 0), 0);
    assert_eq!(chosen_id(ACT_DISCARD, 0, 0), 4);
    assert_eq!(chosen_id(ACT_DISCARD, 135, 0), 139);
    assert_eq!(chosen_id(ACT_TSUMOGIRI, 52, 0), 192);
    assert_eq!(chosen_id(ACT_TSUMOGIRI, 53, 0), 193);
    assert_eq!(chosen_id(ACT_RIICHI_DISCARD, 56, 0), 332);
    assert_eq!(chosen_id(ACT_KAKAN, 0, 0), 6110);
    assert_eq!(chosen_id(ACT_TSUMO, 53, 0), 6707);
    assert_eq!(chosen_id(ACT_ANKAN, 0, 0), 6076);
    assert_eq!(chosen_id(ACT_ANKAN, 33, 0), 6109);
    assert_eq!(chosen_id(ACT_RON, 0, 0), 6246);
    assert_eq!(chosen_id(ACT_RON, 0, 1), 6247);
    assert_eq!(chosen_id(ACT_RON, 0, 2), 6248);
    assert_eq!(chosen_id(ACT_RON, 135, 2), 6653);
    assert_eq!(chosen_id(99, 0, 0), UNRESOLVED_ID);
    assert_eq!(chosen_id(ACT_DISCARD, 136, 0), UNRESOLVED_ID);
    assert_eq!(chosen_id(ACT_RON, 0, 3), UNRESOLVED_ID);
    assert_eq!(source_offset(0, 1), -1);
    assert_eq!(source_offset(1, 0), 1);
    assert_eq!(source_offset(2, 0), 2);
}

#[test]
fn claim_ids_match_published_table() {
    // chi: called 4m-copy0 (12), consumed [2m-copy0, 5m-aka] (8,16).
    assert_eq!(chosen_claim_id(ACT_CHI, 12, &[8, 16], -1), Some(812));
    // chi: first entry (called 1m, [2m,3m] copies 0,0).
    assert_eq!(chosen_claim_id(ACT_CHI, 0, &[4, 8], -1), Some(412));
    // pon: called 0 over [1,2] kamicha.
    assert_eq!(chosen_claim_id(ACT_PON, 0, &[1, 2], -1), Some(4444));
    assert_eq!(chosen_claim_id(ACT_PON, 1, &[0, 2], 1), Some(4454));
    // daiminkan: called 0 over the other three.
    assert_eq!(chosen_claim_id(ACT_DAIMINKAN, 0, &[1, 2, 3], 2), Some(5670));
    assert_eq!(
        chosen_claim_id(ACT_DAIMINKAN, 17, &[16, 18, 19], -1),
        Some(5719)
    );
    // ankan: full blocks.
    assert_eq!(chosen_claim_id(ACT_ANKAN, 0, &[0, 1, 2, 3], 0), Some(6076));
    // Rejects: bad offset, bad pattern, wrong counts, honors chi.
    assert_eq!(chosen_claim_id(ACT_CHI, 12, &[8, 16], 1), None);
    assert_eq!(chosen_claim_id(ACT_CHI, 12, &[8, 9], -1), None);
    assert_eq!(chosen_claim_id(ACT_CHI, 108, &[109, 110], -1), None);
    assert_eq!(chosen_claim_id(ACT_PON, 0, &[1, 2], 0), None);
    assert_eq!(chosen_claim_id(ACT_PON, 0, &[0, 1], -1), None);
    assert_eq!(chosen_claim_id(ACT_DAIMINKAN, 0, &[1, 2, 4], -1), None);
    assert_eq!(chosen_claim_id(ACT_ANKAN, 0, &[0, 1, 2, 4], 0), None);
}

fn custom_tehais() -> Vec<Vec<&'static str>> {
    vec![
        vec![
            "1m", "1m", "9m", "9m", "9m", "9m", "2p", "2p", "2p", "3p", "3p", "3p", "4p",
        ],
        vec![
            "1m", "1m", "6m", "6m", "6m", "6m", "1p", "1p", "1p", "1p", "7s", "7s", "7s",
        ],
        vec![
            "2m", "2m", "2m", "2m", "3m", "3m", "3m", "3m", "4p", "4p", "4p", "5s", "5s",
        ],
        vec![
            "7m", "7m", "7m", "7m", "8m", "8m", "8m", "8m", "E", "E", "E", "E", "F",
        ],
    ]
}

fn pon_v1_text() -> Vec<u8> {
    // Oracle-legal prefix: the pon row itself, kyoku closed without a
    // post-claim draw (V1 in the parity notes).
    let held = custom_tehais();
    let te: Vec<&[&str]> = held.iter().map(|v| &v[..]).collect();
    let mut lines = vec!["{\"type\":\"start_game\"}".to_string(), kyoku_line(&te, 0)];
    lines.push(tsumo_line(0, "5p"));
    lines.push(dahai_line(0, "1m", false));
    lines.push(
        "{\"type\":\"pon\",\"actor\":1,\"target\":0,\"pai\":\"1m\",\"consumed\":[\"1m\",\"1m\"]}"
            .to_string(),
    );
    lines.push("{\"type\":\"end_kyoku\"}".to_string());
    lines.push("{\"type\":\"end_game\"}".to_string());
    let mut out = lines.join("\n");
    out.push('\n');
    out.into_bytes()
}

fn pon_text() -> Vec<u8> {
    let held = custom_tehais();
    let te: Vec<&[&str]> = held.iter().map(|v| &v[..]).collect();
    let mut lines = vec!["{\"type\":\"start_game\"}".to_string(), kyoku_line(&te, 0)];
    lines.push(tsumo_line(0, "5p"));
    lines.push(dahai_line(0, "1m", false));
    lines.push(
        "{\"type\":\"pon\",\"actor\":1,\"target\":0,\"pai\":\"1m\",\"consumed\":[\"1m\",\"1m\"]}"
            .to_string(),
    );
    lines.push(tsumo_line(1, "6p"));
    lines.push(dahai_line(1, "6p", true));
    lines.push("{\"type\":\"end_game\"}".to_string());
    let mut out = lines.join("\n");
    out.push('\n');
    out.into_bytes()
}

#[test]
fn pon_claim_rows_match_oracle_exact() {
    // V1 vectors from the wall-less oracle, bit-for-bit: chosen, full
    // legal sets (discards + twin + riichi + ankan on row0), concealed
    // takes, histories, phases.
    let w = walk_bytes(&pon_v1_text(), 9, None);
    assert_eq!(w.rows, 2);
    let chosen: Vec<i64> = (0..2).map(|r| rd_i64(&w.planes[PLANE_CHOSEN], r)).collect();
    assert_eq!(chosen, [4, 4444]);
    assert_eq!(
        legal_of(&w, 0),
        [4, 36, 44, 48, 52, 57, 193, 308, 309, 310, 311, 6084]
    );
    assert_eq!(legal_of(&w, 1), [0, 4444]);
    assert_eq!(rd_i64(&w.planes[PLANE_PHASE], 0), 1);
    assert_eq!(rd_i64(&w.planes[PLANE_PHASE], 1), 2);
    assert_eq!(hist_of(&w, 0), [0, 1, 2, 3]);
    assert_eq!(hist_of(&w, 1), [0, 1, 2, 4, 7]);
    let mut c0 = [0u8; 34];
    let mut c1 = [0u8; 34];
    for t in 0..34 {
        c0[t] = rd_u8(&w.planes[PLANE_CONCEALED], 0, t);
        c1[t] = rd_u8(&w.planes[PLANE_CONCEALED], 1, t);
    }
    // Row0: 1m×2, 9m×4, 2p×3, 3p×3, 4p (drawn 5p excluded).
    assert_eq!(&c0[0..14], &[2, 0, 0, 0, 0, 0, 0, 0, 4, 0, 3, 3, 1, 0]);
    // Row1: capture-before-mutate — the consumed 1m pair is still
    // present (oracle row shows the same 13 takes).
    assert_eq!(c1[0], 2);
    assert_eq!(c1[5], 4);
    assert_eq!(c1[9], 4);
    assert_eq!(c1[24], 3);
    assert_eq!(c1.iter().map(|x| *x as u32).sum::<u32>(), 13);
    // Meld installed on seat 1: pon [1,2] + called 0.
    assert_eq!(w.ledger.meld_lens[1], 1);
    let meld = &w.ledger.melds[1][0];
    assert_eq!(meld.kind, ACT_PON);
    assert_eq!(meld.owner, 1);
    assert_eq!(meld.tiles, ([0, 1, 2, 0], 3));
}

#[test]
fn pon_ext_post_claim_draw_is_conservation_reject() {
    // Post-claim closure: a claimer tsumo after a non-kan claim with no
    // intervening discard and no rinshan pending deals a 15th ledger
    // take (12 concealed + 3 meld) the wall-less oracle desyncs on
    // (whole-game quarantine, zero rows). Fail-closed favors the
    // oracle: conservation reject, no taxonomy churn. The V1 prefix
    // (pon_claim_rows_match_oracle_exact) still walks 2 rows.
    let text = pon_text();
    let mut arena = Vec::new();
    let game = frame_spans(&mut arena, &text, 7, 9).expect("frame");
    let verdict = gate_game(9, &game.events).expect("gate ok");
    assert!(verdict.is_sample());
    let ctx = GameCtx::wall_less(9);
    let mut ledger = Ledger::new();
    let mut planes: [Vec<u8>; N_WALK_PLANES] = Default::default();
    let mut hist = Vec::new();
    let mut sink = RowSink::new(&mut planes, &mut hist);
    let err = walk_game(&ctx, &game.events, &mut ledger, &mut sink)
        .expect_err("post-claim draw must reject");
    // Events: 0 start, 1 kyoku, 2 tsumo(0), 3 dahai(0), 4 pon(1), 5 tsumo(1).
    assert_eq!(err.event_idx, 5);
    assert_eq!(err.reason, WALK_TILE_CONSERVATION);
    assert_eq!(err.name(), "tile-conservation");
}

fn chi_tehais() -> Vec<Vec<&'static str>> {
    // Pool-exact tehais (global <=4 per type): seat0 discards 3p, seat1
    // chis 2p-3p-4p kamicha.
    vec![
        vec![
            "3p", "3p", "1m", "1m", "1m", "2m", "2m", "2m", "7s", "7s", "7s", "E", "E",
        ],
        vec![
            "2p", "2p", "4p", "4p", "6m", "6m", "6m", "1p", "1p", "1p", "S", "S", "S",
        ],
        vec![
            "8m", "8m", "8m", "9m", "9m", "9m", "2s", "2s", "2s", "6s", "6s", "6s", "W",
        ],
        vec![
            "4s", "4s", "4s", "8s", "8s", "8s", "9s", "9s", "9s", "N", "N", "N", "F",
        ],
    ]
}

fn chi_text() -> Vec<u8> {
    let held = chi_tehais();
    let te: Vec<&[&str]> = held.iter().map(|v| &v[..]).collect();
    let mut lines = vec!["{\"type\":\"start_game\"}".to_string(), kyoku_line(&te, 0)];
    lines.push(tsumo_line(0, "5p"));
    lines.push(dahai_line(0, "3p", false));
    lines.push(
        "{\"type\":\"chi\",\"actor\":1,\"target\":0,\"pai\":\"3p\",\"consumed\":[\"2p\",\"4p\"]}"
            .to_string(),
    );
    lines.push(tsumo_line(1, "6p"));
    lines.push(dahai_line(1, "6p", true));
    lines.push("{\"type\":\"end_game\"}".to_string());
    let mut out = lines.join("\n");
    out.push('\n');
    out.into_bytes()
}

#[test]
fn chi_ext_post_claim_draw_is_conservation_reject() {
    // Same tripwire via chi (second non-kan claim kind): claimer tsumo
    // with no intervening discard rejects as tile-conservation.
    let text = chi_text();
    let mut arena = Vec::new();
    let game = frame_spans(&mut arena, &text, 7, 21).expect("frame");
    let verdict = gate_game(21, &game.events).expect("gate ok");
    assert!(verdict.is_sample());
    let ctx = GameCtx::wall_less(21);
    let mut ledger = Ledger::new();
    let mut planes: [Vec<u8>; N_WALK_PLANES] = Default::default();
    let mut hist = Vec::new();
    let mut sink = RowSink::new(&mut planes, &mut hist);
    let err = walk_game(&ctx, &game.events, &mut ledger, &mut sink)
        .expect_err("chi claimer draw must reject");
    assert_eq!(err.event_idx, 5);
    assert_eq!(err.reason, WALK_TILE_CONSERVATION);
}

fn chi_then_discard_text() -> Vec<u8> {
    // Claimant discards after chi (chi-twin of pon_then_discard): the
    // post-chi dahai row carries the remainder takes.
    let held = chi_tehais();
    let te: Vec<&[&str]> = held.iter().map(|v| &v[..]).collect();
    let mut lines = vec!["{\"type\":\"start_game\"}".to_string(), kyoku_line(&te, 0)];
    lines.push(tsumo_line(0, "5p"));
    lines.push(dahai_line(0, "3p", false));
    lines.push(
        "{\"type\":\"chi\",\"actor\":1,\"target\":0,\"pai\":\"3p\",\"consumed\":[\"2p\",\"4p\"]}"
            .to_string(),
    );
    lines.push(dahai_line(1, "6m", false));
    lines.push("{\"type\":\"end_game\"}".to_string());
    let mut out = lines.join("\n");
    out.push('\n');
    out.into_bytes()
}

#[test]
fn chi_remainder_offers_skip_only_called_string() {
    // Walled: seat1 chis 2p-3p-4p holding a second 2p and 4p. The
    // leftovers render non-called strings, so the post-chi dahai still
    // offers them (the wall oracle withholds only the just-claimed
    // string: a scripted-wall probe of the live engine deals chi
    // consuming {44,48} and offers discard-45 on the post-chi turn).
    let arc: std::sync::Arc<str> = std::sync::Arc::from("sha256:test-digest");
    let w = {
        let digest: &str = &arc;
        walk_bytes(
            &chi_then_discard_text(),
            23,
            Some((digest, std::sync::Arc::clone(&arc))),
        )
    };
    assert_eq!(w.rows, 3);
    assert_eq!(rd_i64(&w.planes[PLANE_CHOSEN], 2), 24);
    let legal = legal_of(&w, 2);
    assert!(legal.contains(&45), "2p remainder missing: {legal:?}");
    assert!(legal.contains(&53), "4p remainder missing: {legal:?}");
}

fn pon_then_discard_text() -> Vec<u8> {
    // Claimant discards before the next draw: tripwire disarmed, the
    // rotation walks (4 rows: dahai, pon, dahai, dahai).
    let held = custom_tehais();
    let te: Vec<&[&str]> = held.iter().map(|v| &v[..]).collect();
    let mut lines = vec!["{\"type\":\"start_game\"}".to_string(), kyoku_line(&te, 0)];
    lines.push(tsumo_line(0, "5p"));
    lines.push(dahai_line(0, "1m", false));
    lines.push(
        "{\"type\":\"pon\",\"actor\":1,\"target\":0,\"pai\":\"1m\",\"consumed\":[\"1m\",\"1m\"]}"
            .to_string(),
    );
    lines.push(dahai_line(1, "6m", false));
    lines.push(tsumo_line(2, "7p"));
    lines.push(dahai_line(2, "2m", false));
    lines.push("{\"type\":\"end_game\"}".to_string());
    let mut out = lines.join("\n");
    out.push('\n');
    out.into_bytes()
}

#[test]
fn post_claim_intervening_discard_disarms_tripwire() {
    // Guard against over-rejection: pon -> claimer discard -> rotation
    // draw walks normally.
    let w = walk_bytes(&pon_then_discard_text(), 22, None);
    assert_eq!(w.rows, 4);
    let chosen: Vec<i64> = (0..4).map(|r| rd_i64(&w.planes[PLANE_CHOSEN], r)).collect();
    assert_eq!(chosen[1], 4444);
}

#[test]
fn kyushu_offer_on_qualifying_first_draw() {
    // 13 distinct terminals/honors + first draw: the abort offer rides
    // (live parity; id pinned against the live engine + action table).
    let te = vec![
        &[
            "1m", "9m", "1p", "9p", "1s", "9s", "E", "S", "W", "N", "P", "F", "C",
        ][..],
        &[
            "2m", "3m", "4m", "2p", "3p", "4p", "2s", "3s", "4s", "6m", "7m", "8m", "6p",
        ][..],
        &[
            "2m", "3m", "4m", "2p", "3p", "4p", "2s", "3s", "4s", "6m", "7m", "8m", "7p",
        ][..],
        &[
            "2m", "3m", "4m", "2p", "3p", "4p", "2s", "3s", "4s", "6m", "7m", "8m", "8p",
        ][..],
    ];
    let mut lines = vec!["{\"type\":\"start_game\"}".to_string(), kyoku_line(&te, 0)];
    lines.push(tsumo_line(0, "2m"));
    lines.push(dahai_line(0, "2m", true));
    lines.push("{\"type\":\"end_game\"}".to_string());
    let mut out = lines.join("\n");
    out.push('\n');
    let w = walk_bytes(out.as_bytes(), 75, None);
    assert_eq!(w.rows, 1);
    let legal = legal_of(&w, 0);
    assert!(
        legal.contains(&ABORT_NINE_TERMINALS_ID),
        "kyushu offer missing"
    );
    assert_eq!(ABORT_NINE_TERMINALS_ID, 6790);
}

#[test]
fn kyushu_absent_without_nine_yaochu() {
    // F1 golden game (simple hands, no yaochu mass): no row offers abort.
    let w = walk_bytes(&f1_text(), 76, None);
    assert!(w.rows > 0);
    let mut r = 0usize;
    while r < w.rows as usize {
        assert!(!legal_of(&w, r).contains(&ABORT_NINE_TERMINALS_ID));
        r += 1;
    }
}

fn ankan_rinshan_tehais() -> Vec<Vec<&'static str>> {
    // Pool-exact tehais: seat0 holds a 9m quad for the ankan; every
    // other type is collision-free globally.
    vec![
        vec![
            "9m", "9m", "9m", "9m", "1m", "1m", "1m", "2m", "2m", "2m", "3p", "3p", "5p",
        ],
        vec![
            "4m", "4m", "4m", "6m", "6m", "6m", "7m", "7m", "7m", "1p", "1p", "1p", "E",
        ],
        vec![
            "3m", "3m", "3m", "8m", "8m", "8m", "2p", "2p", "2p", "4p", "4p", "4p", "S",
        ],
        vec![
            "6p", "6p", "6p", "7p", "7p", "7p", "8p", "8p", "8p", "9p", "9p", "9p", "N",
        ],
    ]
}

fn ankan_rinshan_text() -> Vec<u8> {
    let held = ankan_rinshan_tehais();
    let te: Vec<&[&str]> = held.iter().map(|v| &v[..]).collect();
    let mut lines = vec!["{\"type\":\"start_game\"}".to_string(), kyoku_line(&te, 0)];
    lines.push(tsumo_line(0, "6s"));
    lines.push(
        "{\"type\":\"ankan\",\"actor\":0,\"consumed\":[\"9m\",\"9m\",\"9m\",\"9m\"]}".to_string(),
    );
    lines.push(tsumo_line(0, "7s"));
    lines.push(dahai_line(0, "7s", true));
    lines.push("{\"type\":\"end_game\"}".to_string());
    let mut out = lines.join("\n");
    out.push('\n');
    out.into_bytes()
}

#[test]
fn post_claim_kan_rinshan_still_accepted() {
    // Guard against over-rejection: ankan arms rinshan pending (never
    // the non-kan tripwire), so the replacement draw + discard walk.
    let w = walk_bytes(&ankan_rinshan_text(), 23, None);
    assert_eq!(w.rows, 2);
    let chosen: Vec<i64> = (0..2).map(|r| rd_i64(&w.planes[PLANE_CHOSEN], r)).collect();
    assert_eq!(chosen[0], 6084);
    assert_eq!(w.ledger.meld_lens[0], 1);
}
fn tenpai_tehais() -> Vec<Vec<&'static str>> {
    vec![
        vec![
            "1m", "1m", "1m", "2m", "2m", "2m", "3m", "3m", "3m", "4m", "4m", "4m", "5p",
        ],
        vec![
            "6m", "6m", "6m", "6m", "7m", "7m", "7m", "7m", "1p", "1p", "1p", "2p", "5p",
        ],
        vec![
            "8m", "8m", "8m", "8m", "9m", "9m", "9m", "9m", "2p", "2p", "2p", "3p", "3p",
        ],
        vec![
            "1s", "1s", "1s", "1s", "2s", "2s", "2s", "2s", "E", "E", "E", "F", "F",
        ],
    ]
}

fn tsumo_win_text() -> Vec<u8> {
    let held = tenpai_tehais();
    let te: Vec<&[&str]> = held.iter().map(|v| &v[..]).collect();
    let mut lines = vec!["{\"type\":\"start_game\"}".to_string(), kyoku_line(&te, 0)];
    lines.push(tsumo_line(0, "5p"));
    // Tenhou-real tsumo: no flag, no pai, target == actor.
    lines.push(
        "{\"type\":\"hora\",\"actor\":0,\"target\":0,\"deltas\":[8000,-2000,-3000,-3000]}"
            .to_string(),
    );
    lines.push("{\"type\":\"end_kyoku\"}".to_string());
    lines.push("{\"type\":\"end_game\"}".to_string());
    let mut out = lines.join("\n");
    out.push('\n');
    out.into_bytes()
}

#[test]
fn declined_tsumo_win_still_offered() {
    // Winning draw declined (dahai instead of hora): the tsumo answer
    // still rides the draw row (engine offers wins taken or not).
    let held = tenpai_tehais();
    let te: Vec<&[&str]> = held.iter().map(|v| &v[..]).collect();
    let mut lines = vec!["{\"type\":\"start_game\"}".to_string(), kyoku_line(&te, 0)];
    lines.push(tsumo_line(0, "5p"));
    lines.push(dahai_line(0, "1m", false));
    lines.push("{\"type\":\"end_game\"}".to_string());
    let mut out = lines.join("\n");
    out.push('\n');
    let w = walk_bytes(out.as_bytes(), 14, None);
    assert_eq!(w.rows, 1);
    // Chosen is the 1m discard (take 0); tsumo on the drawn 5p (take 53).
    assert_eq!(rd_i64(&w.planes[PLANE_CHOSEN], 0), 4);
    let legal = legal_of(&w, 0);
    assert!(legal.contains(&6707), "declined tsumo missing: {legal:?}");
}

#[test]
fn tsumo_win_row_scores_and_shape() {
    let w = walk_bytes(&tsumo_win_text(), 11, None);
    assert_eq!(w.rows, 1);
    assert_eq!(rd_i64(&w.planes[PLANE_CHOSEN], 0), 6707);
    assert_eq!(rd_i64(&w.planes[PLANE_ACTOR], 0), 0);
    assert_eq!(rd_i64(&w.planes[PLANE_PHASE], 0), 1);
    // Pre-payoff observation: the hora row is the winning declaration,
    // so its scores plane reads the ledger BEFORE the win deltas apply
    // (post-payoff scores would leak the payout target into the input).
    let scores: Vec<i32> = (0..4)
        .map(|s| rd_score(&w.planes[PLANE_SCORES], 0, s))
        .collect();
    assert_eq!(scores, [25000, 25000, 25000, 25000]);
    let legal = legal_of(&w, 0);
    assert_eq!(
        legal,
        [
            4, 8, 12, 16, 57, 193, 276, 277, 278, 280, 281, 282, 284, 285, 286, 288, 289, 290, 329,
            330, 6707
        ]
    );
}

fn ron_win_text() -> Vec<u8> {
    let held = tenpai_tehais();
    let te: Vec<&[&str]> = held.iter().map(|v| &v[..]).collect();
    let mut lines = vec!["{\"type\":\"start_game\"}".to_string(), kyoku_line(&te, 0)];
    lines.push(tsumo_line(0, "6p"));
    lines.push(dahai_line(0, "6p", true));
    lines.push(tsumo_line(1, "7p"));
    lines.push(dahai_line(1, "5p", false));
    lines.push(
        "{\"type\":\"hora\",\"actor\":0,\"target\":1,\"deltas\":[8000,-8000,0,0]}".to_string(),
    );
    lines.push("{\"type\":\"end_kyoku\"}".to_string());
    lines.push("{\"type\":\"end_game\"}".to_string());
    let mut out = lines.join("\n");
    out.push('\n');
    out.into_bytes()
}

#[test]
fn ron_win_row_window_and_mask() {
    let w = walk_bytes(&ron_win_text(), 12, None);
    assert_eq!(w.rows, 3);
    // Tedashi 5p discards map to Discard ids (collapse 53 → 57).
    let chosen: Vec<i64> = (0..3).map(|r| rd_i64(&w.planes[PLANE_CHOSEN], r)).collect();
    assert_eq!(chosen, [196, 57, 6406]);
    // Exact oracle legal sets (log_replay wall-less): discards + twin +
    // riichi candidates + ankan quads, chosen forced.
    assert_eq!(
        legal_of(&w, 0),
        [
            4, 8, 12, 16, 57, 60, 196, 276, 277, 278, 280, 281, 282, 284, 285, 286, 288, 289, 290,
            329, 332
        ]
    );
    assert_eq!(legal_of(&w, 1), [24, 28, 40, 44, 57, 64, 200, 6081, 6082]);
    // Ron row: discard-response phase, turn actor = discarder, {pass,ron}.
    assert_eq!(rd_i64(&w.planes[PLANE_PHASE], 2), 2);
    assert_eq!(legal_of(&w, 2), [0, 6406]);
    // The taken ron proves the window: call_window in the ron history.
    assert_eq!(hist_of(&w, 2), [0, 1, 2, 3, 4, 2, 4, 7]);
    // Concealed row0: triples + single 5p (draw excluded); drawn 56.
    let mut c0 = [0u8; 34];
    for t in 0..34 {
        c0[t] = rd_u8(&w.planes[PLANE_CONCEALED], 0, t);
    }
    assert_eq!(&c0[0..14], &[3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]);
}

#[test]
fn hora_without_end_kyoku_quarantines() {
    // ron_win shape (hora straight into end_game): the wall oracle goes
    // terminal and rejects the trailing end (quarantine parity with
    // replay_expand's post-loop terminal check). Walled-only: the SIM
    // oracle walks the same shape.
    let held = tenpai_tehais();
    let te: Vec<&[&str]> = held.iter().map(|v| &v[..]).collect();
    let mut lines = vec!["{\"type\":\"start_game\"}".to_string(), kyoku_line(&te, 0)];
    lines.push(tsumo_line(0, "6p"));
    lines.push(dahai_line(0, "6p", true));
    lines.push(tsumo_line(1, "7p"));
    lines.push(dahai_line(1, "5p", false));
    lines.push(
        "{\"type\":\"hora\",\"actor\":0,\"target\":1,\"deltas\":[8000,-8000,0,0]}".to_string(),
    );
    lines.push("{\"type\":\"end_game\"}".to_string());
    let mut out = lines.join("\n");
    out.push('\n');
    let text = out.into_bytes();
    let mut arena = Vec::new();
    let game = frame_spans(&mut arena, &text, 7, 99).expect("frame");
    let verdict = gate_game(99, &game.events).expect("gate ok");
    assert!(verdict.is_sample());
    let arc: std::sync::Arc<str> = std::sync::Arc::from("sha256:test-digest");
    let ctx = GameCtx::walled(99, &arc, Some(std::sync::Arc::clone(&arc)));
    let mut ledger = Ledger::new();
    let mut planes: [Vec<u8>; N_WALK_PLANES] = Default::default();
    let mut hist_lens = Vec::new();
    let mut sink = RowSink::new(&mut planes, &mut hist_lens);
    let err = walk_game(&ctx, &game.events, &mut ledger, &mut sink).expect_err("must quarantine");
    assert_eq!(err.reason, WALK_PAST_TERMINAL);
}

fn riichi_text() -> Vec<u8> {
    let teanais: Vec<Vec<&str>> = vec![
        vec![
            "1m", "1m", "1m", "2m", "2m", "2m", "3m", "3m", "3m", "4m", "4m", "4m", "5p",
        ],
        vec![
            "6m", "6m", "6m", "6m", "7m", "7m", "7m", "7m", "1p", "1p", "1p", "1p", "2p",
        ],
        vec![
            "8m", "8m", "8m", "8m", "9m", "9m", "9m", "9m", "2p", "2p", "2p", "3p", "3p",
        ],
        vec![
            "1s", "1s", "1s", "1s", "2s", "2s", "2s", "2s", "3s", "3s", "3s", "4s", "4s",
        ],
    ];
    let te: Vec<&[&str]> = teanais.iter().map(|v| &v[..]).collect();
    let mut lines = vec!["{\"type\":\"start_game\"}".to_string(), kyoku_line(&te, 0)];
    lines.push(tsumo_line(0, "6p"));
    lines.push("{\"type\":\"reach\",\"actor\":0}".to_string());
    lines.push(dahai_line(0, "6p", false));
    lines.push(tsumo_line(1, "S"));
    lines.push(dahai_line(1, "S", true));
    lines.push(tsumo_line(2, "W"));
    lines.push(dahai_line(2, "W", true));
    lines.push(tsumo_line(3, "N"));
    lines.push(dahai_line(3, "N", true));
    lines.push(tsumo_line(0, "7p"));
    lines.push(dahai_line(0, "7p", true));
    lines.push("{\"type\":\"end_game\"}".to_string());
    let mut out = lines.join("\n");
    out.push('\n');
    out.into_bytes()
}

#[test]
fn riichi_declaration_and_forced_pair() {
    // Full oracle vectors (wall-less log_replay): riichi declaration,
    // three rotation discards (ankan quads + a riichi candidate ride),
    // then the true-take forced pair.
    let w = walk_bytes(&riichi_text(), 13, None);
    assert_eq!(w.rows, 5);
    let chosen: Vec<i64> = (0..5).map(|r| rd_i64(&w.planes[PLANE_CHOSEN], r)).collect();
    assert_eq!(chosen, [332, 252, 256, 260, 200]);
    assert_eq!(
        legal_of(&w, 0),
        [
            4, 8, 12, 16, 57, 60, 196, 276, 277, 278, 280, 281, 282, 284, 285, 286, 288, 289, 290,
            329, 332
        ]
    );
    assert_eq!(
        legal_of(&w, 1),
        [24, 28, 40, 44, 116, 252, 6081, 6082, 6085]
    );
    assert_eq!(legal_of(&w, 2), [32, 36, 44, 48, 120, 256, 392, 6083, 6084]);
    assert_eq!(
        legal_of(&w, 3),
        [
            76, 80, 84, 88, 124, 260, 352, 353, 354, 355, 396, 6094, 6095
        ]
    );
    // Forced row offers exactly the true-take pair.
    assert_eq!(legal_of(&w, 4), [64, 200]);
    // Riichi stick moved at declaration.
    let s0: Vec<i32> = (0..4)
        .map(|s| rd_score(&w.planes[PLANE_SCORES], 4, s))
        .collect();
    assert_eq!(s0, [24000, 25000, 25000, 25000]);
    assert_eq!(w.ledger.riichi[0], 1);
    // Forced-row history: no windows on the honor discards.
    assert_eq!(hist_of(&w, 4), [0, 1, 2, 3, 4, 2, 4, 2, 4, 2, 4, 2, 3]);
}

fn dora_text() -> Vec<u8> {
    let mut lines = vec![
        "{\"type\":\"start_game\"}".to_string(),
        kyoku_line(&TEHAIS_F1, 0),
    ];
    lines.push(tsumo_line(0, "5pr"));
    lines.push(dahai_line(0, "5pr", true));
    lines.push("{\"type\":\"dora\",\"dora_marker\":\"5p\"}".to_string());
    lines.push(tsumo_line(1, "5p"));
    lines.push(dahai_line(1, "5p", true));
    lines.push("{\"type\":\"end_game\"}".to_string());
    let mut out = lines.join("\n");
    out.push('\n');
    out.into_bytes()
}

#[test]
fn dora_indicator_plane_reveals_in_order() {
    let w = walk_bytes(&dora_text(), 14, None);
    assert_eq!(w.rows, 2);
    for k in 0..5 {
        assert_eq!(rd_dora(&w.planes[PLANE_DORA], 0, k), -1);
    }
    assert_eq!(rd_dora(&w.planes[PLANE_DORA], 1, 0), 53);
    for k in 1..5 {
        assert_eq!(rd_dora(&w.planes[PLANE_DORA], 1, k), -1);
    }
}

#[test]
fn rejects_are_closed_triples() {
    // Dahai before any kyoku: turn-order.
    let text = "{\"type\":\"start_game\"}\n{\"type\":\"dahai\",\"actor\":0,\"pai\":\"1m\"}\n{\"type\":\"end_game\"}\n";
    let mut arena = Vec::new();
    let game = frame_spans(&mut arena, text.as_bytes(), 0, 77).expect("frame");
    let ctx = GameCtx::wall_less(77);
    let mut ledger = Ledger::new();
    let mut planes: [Vec<u8>; N_WALK_PLANES] = Default::default();
    let mut hist = Vec::new();
    let mut sink = RowSink::new(&mut planes, &mut hist);
    let err = walk_game(&ctx, &game.events, &mut ledger, &mut sink).expect_err("reject");
    assert_eq!(err.game_idx, 77);
    assert_eq!(err.reason, WALK_TURN_ORDER);
    // Truncated game (no end_game): turn-order.
    let text2 = "{\"type\":\"start_game\"}\n{\"type\":\"end_game\"}\n";
    let mut arena2 = Vec::new();
    let game2 = frame_spans(&mut arena2, text2.as_bytes(), 0, 78).expect("frame");
    let ctx2 = GameCtx::wall_less(78);
    let mut ledger2 = Ledger::new();
    let mut planes2: [Vec<u8>; N_WALK_PLANES] = Default::default();
    let mut hist2 = Vec::new();
    let mut sink2 = RowSink::new(&mut planes2, &mut hist2);
    let err2 = walk_game(&ctx2, &game2.events, &mut ledger2, &mut sink2).expect_err("reject");
    assert_eq!(err2.reason, WALK_TURN_ORDER);
    // Unknown stacked kind (gate bypassed): unknown-event.
    let mut arena3 = Vec::new();
    let text3 = f1_text();
    let mut game3 = frame_spans(&mut arena3, &text3, 0, 79).expect("frame");
    game3.events[3].kind = KIND_OTHER;
    let ctx3 = GameCtx::wall_less(79);
    let mut ledger3 = Ledger::new();
    let mut planes3: [Vec<u8>; N_WALK_PLANES] = Default::default();
    let mut hist3 = Vec::new();
    let mut sink3 = RowSink::new(&mut planes3, &mut hist3);
    let err3 = walk_game(&ctx3, &game3.events, &mut ledger3, &mut sink3).expect_err("reject");
    assert_eq!(err3.reason, WALK_UNKNOWN_EVENT);
    assert_eq!(err3.event_idx, 3);
    // Bare dora at walk level (gate bypassed): bare-dora.
    let mut arena4 = Vec::new();
    let mut game4 = frame_spans(&mut arena4, &text3, 0, 80).expect("frame");
    game4.events[2].kind = crate::ingest::KIND_DORA;
    game4.events[2].dora_span = None;
    let ctx4 = GameCtx::wall_less(80);
    let mut ledger4 = Ledger::new();
    let mut planes4: [Vec<u8>; N_WALK_PLANES] = Default::default();
    let mut hist4 = Vec::new();
    let mut sink4 = RowSink::new(&mut planes4, &mut hist4);
    let err4 = walk_game(&ctx4, &game4.events, &mut ledger4, &mut sink4).expect_err("reject");
    assert_eq!(err4.reason, WALK_BARE_DORA);
    // Reason names join the closed sink vocabulary (G5).
    assert_eq!(err3.name(), "unknown-event");
    assert_eq!(err4.name(), "bare-dora");
    let _ = WALK_TILE_CONSERVATION;
}

#[test]
fn walk_is_bitwise_deterministic() {
    // G1 facet: two walks of one game produce identical planes.
    let text = f1_text();
    let a = walk_bytes(&text, 3, None);
    let b = walk_bytes(&text, 3, None);
    assert_eq!(a.rows, b.rows);
    assert_eq!(a.hist_lens, b.hist_lens);
    for p in 0..N_WALK_PLANES {
        assert_eq!(a.planes[p], b.planes[p]);
    }
}
#[test]
fn win_shape_units() {
    // Four triples + pair wins; junk does not.
    assert!(win_shape_14(
        &[0, 0, 0, 4, 4, 4, 8, 8, 8, 12, 12, 12, 52, 52],
        0
    ));
    // Seven pairs wins; honor quad + six pairs does not (E cannot
    // sequence and the pairs cannot triple: no pair+4-meld split).
    assert!(win_shape_14(
        &[0, 0, 4, 4, 8, 8, 12, 12, 16, 16, 20, 20, 24, 24],
        0
    ));
    assert!(!win_shape_14(
        &[108, 108, 108, 108, 4, 4, 8, 8, 12, 12, 16, 16, 20, 20],
        0
    ));
    // Open hand needs melds + pair from concealed.
    assert!(win_shape_14(&[0, 0, 0, 4, 4, 4, 8, 8, 8, 12, 12], 1));
    assert!(!win_shape_14(&[0, 0, 0, 4, 4, 4, 8, 8, 8, 12, 12], 2));
}

#[test]
fn legal_ids_sorted_contract() {
    // Bad seat: empty set, never a panic.
    let mut l = Ledger::new();
    let (ids, len) = legal_ids_sorted(9, &mut l, None);
    assert_eq!(len, 0);
    assert!(ids.iter().all(|x| *x == 0));
}
