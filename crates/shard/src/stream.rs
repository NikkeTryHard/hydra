//! Walled + wall-less stream state: tehais install, countdown wall, tracked hands.
//!
//! Reference (read-only): `src/hydra2/engines/riichienv/log_replay.py`
//! (`_take_next`, `_tracker_init`, `_split_kyoku_draws`, `_track_remove`,
//! `_track_remove_consumed`, `_tracked_consumed`, `_LIVE_WALL_BASE`) and
//! `src/hydra2/data/replay_expand.py` (`_schedule_for`, `ReplayExpander`
//! mode machine) plus `src/hydra2/data/decode.py` (wall extraction).
//! Take order reused exactly (tehais seats 0..3, then live draws in log
//! order, then rinshan draws); code written fresh.
//!
//! One framing gate serves both paths: [`parse_game`] accepts an optional
//! 136-tile wall (`wall`/`wall_tiles`/`tiles`, permutation 0..135) and
//! carries it on [`ParsedGame::wall_tiles`]. Wall-less games carry `None`
//! (SIM-mark derivation); walled games carry `Some` (real-digest
//! derivation). Malformed walls fail closed as `framing`, never silently
//! dropped.

use std::collections::HashMap;

use sha2::{Digest, Sha256};

use crate::mjai_event::{MjaiEvent, TRANSPARENT_KINDS};
use crate::tile::{copies_of_string, mjai_string_of};

/// Live-wall countdown base: 136 tiles minus 4x13 dealt minus the 14-tile
/// dead wall (tehais-install countdown semantics).
pub const LIVE_WALL_BASE: u32 = 136 - 52 - 14;

/// Rejection of a whole framed game before the walk starts.
///
/// S7: one framing gate for walled + wall-less. Valid 136-tile walls are
/// accepted (never quarantined); malformed walls fail as `Framing`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GameReject {
    Framing(String),
}

/// One framed game: validated events plus derived identity.
///
/// `wall_tiles` is `Some` permutation 0..135 exactly when the framed payload
/// carries a 136-int wall under `wall`/`wall_tiles`/`tiles` (first match in
/// event order, `wall` > `wall_tiles` > `tiles` per event, mirroring
/// `decode_game_object`). `None` is the wall-less path (SIM mark).
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedGame {
    pub game_id: String,
    pub object_id: String,
    pub events: Vec<MjaiEvent>,
    pub wall_tiles: Option<Vec<u8>>,
}

/// Parse framed JSONL into a game under strict line rules.
/// (mirrors `decode_game_object` + `_game_id_from_events`).
///
/// Wall extraction mirrors `decode_game_object`: the first event carrying a
/// 136-int list under `wall`/`wall_tiles`/`tiles` binds the wall. Entries
/// must be ints 0..135 permuting 0..135 exactly once (the `WallSchedule`
/// contract); anything else with a 136-length int-looking list fails closed
/// as framing. Non-136 lists under those keys are ignored (never walls).
pub fn parse_game(text: &str, object_id: &str) -> Result<ParsedGame, GameReject> {
    let events = crate::mjai_event::frame_events(text, object_id).map_err(GameReject::Framing)?;
    let raw: Vec<serde_json::Value> = text
        .split('\n')
        .filter(|line| !line.is_empty())
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()
        .map_err(|e| GameReject::Framing(format!("framing {object_id}: {e}")))?;
    let mut wall_tiles: Option<Vec<u8>> = None;
    'scan: for value in &raw {
        if let Some(obj) = value.as_object() {
            for key in ["wall", "wall_tiles", "tiles"] {
                if let Some(list) = obj.get(key).and_then(|v| v.as_array()) {
                    if list.len() != 136 {
                        continue;
                    }
                    if !list.iter().all(|x| x.is_i64() || x.is_u64()) {
                        continue;
                    }
                    let mut tiles: Vec<u8> = Vec::with_capacity(136);
                    for entry in list {
                        let tile = entry
                            .as_u64()
                            .or_else(|| entry.as_i64().map(|v| v.cast_unsigned()));
                        match tile {
                            Some(v) if v <= 135 => tiles.push(u8::try_from(v).unwrap_or(0)),
                            _ => {
                                return Err(GameReject::Framing(format!(
                                    "framing {object_id}: wall field {key:?} carries a non-tile id"
                                )));
                            }
                        }
                    }
                    let mut sorted = tiles.clone();
                    sorted.sort_unstable();
                    // Permutation 0..135 exactly once (u8 holds 0..135; the
                    // entry-wise range check above makes the u16 compare exact).
                    let mut ok = sorted.len() == 136;
                    if ok {
                        for (got, want) in sorted.iter().zip(0..136u16) {
                            if *got as u16 != want {
                                ok = false;
                                break;
                            }
                        }
                    }
                    if !ok {
                        return Err(GameReject::Framing(format!(
                            "framing {object_id}: wall field {key:?} must permute 0..135 exactly once"
                        )));
                    }
                    wall_tiles = Some(tiles);
                    break 'scan;
                }
            }
        }
    }
    let mut game_id: Option<String> = None;
    for event in &events {
        if let Some(gid) = event.game_id.as_ref().or(event.game_id_alt.as_ref())
            && !gid.is_empty() {
                game_id = Some(gid.clone());
                break;
            }
    }
    let game_id = game_id.unwrap_or_else(|| {
        let digest = Sha256::digest(object_id.as_bytes());
        format!("game-{:012x}", u128_from_prefix(&digest))
    });
    Ok(ParsedGame {
        game_id,
        object_id: object_id.to_string(),
        events,
        wall_tiles,
    })
}

fn u128_from_prefix(digest: &[u8]) -> u128 {
    // First 12 hex chars of sha256(object_id), like the Python fallback.
    let hex: String = digest.iter().take(6).map(|b| format!("{b:02x}")).collect();
    u128::from_str_radix(&hex, 16).unwrap_or(0)
}

/// Exposed meld record (minimal: kind, owner, tiles, source, called).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrackedMeld {
    pub kind: String,
    pub owner: u8,
    pub tiles: Vec<u8>,
    pub source: Option<u8>,
    pub called: Option<u8>,
}

/// Per-kyoku tracked state: hands, rivers, draws, melds, take ledger.
#[derive(Debug)]
pub struct KyokuTrack {
    pub ordinal: usize,
    pub take_taken: HashMap<String, usize>,
    pub tehais: [Vec<String>; 4],
    pub hands: [Vec<u8>; 4],
    pub rivers: [Vec<u8>; 4],
    pub drawn: [Option<u8>; 4],
    /// Turn-scoped drawn copy: `Some` only when the seat drew this turn and
    /// has not acted since (mirrors drained step drawn presence, which feeds
    /// concealed-hand construction but NOT the persistent drawn display).
    pub drawn_live: [Option<u8>; 4],
    pub dora: Vec<String>,
    pub melds: [Vec<TrackedMeld>; 4],
    pub riichi_declared: [bool; 4],
    pub drawer: Option<u8>,
    pub exp_drawer: Option<u8>,
    pub kan_pending: bool,
    pub draws: u32,
    pub live_queues: [Vec<u8>; 4],
    pub rinshan_queues: [Vec<u8>; 4],
    pub tsumo_counts: [u32; 4],
    pub first_draws: [Option<String>; 4],
}

fn is_end_or_boundary(kind: &str) -> bool {
    matches!(
        kind,
        "end_game" | "endGame" | "game_end" | "end" | "start_kyoku" | "end_kyoku"
    )
}

fn is_transparent(kind: &str) -> bool {
    TRANSPARENT_KINDS.contains(&kind)
}

/// Partition a kyoku's tsumo pais into live and rinshan draws.
///
/// A tsumo whose previous significant event (past transparent dora /
/// reach_accepted markers) is a kan is the rinshan replacement.
/// Malformed draws (bad actor, missing pai) fail closed, mirroring
/// `_split_kyoku_draws`.
pub fn split_draws(
    events: &[MjaiEvent],
    start_idx: usize,
    game_id: &str,
    ordinal: usize,
) -> Result<(Vec<String>, Vec<String>), TrackReject> {
    let mut live = Vec::new();
    let mut rinshan = Vec::new();
    let mut prev_kind = "start_kyoku";
    let mut idx = start_idx + 1;
    while idx < events.len() {
        let kind = events[idx].type_.as_str();
        if is_end_or_boundary(kind) {
            break;
        }
        if is_transparent(kind) {
            idx += 1;
            continue;
        }
        if kind == "tsumo" {
            // Shape mirrors `_split_kyoku_draws` fail-closed checks; codes
            // follow the walk's own draw convention (bad actor is
            // turn-order, missing pai is tile-conservation).
            if events[idx].actor_seat("tsumo").is_err() {
                return Err(TrackReject {
                    code: "turn-order".to_string(),
                    detail: desync_detail(game_id, ordinal, "malformed draw event"),
                });
            }
            let pai = events[idx].pai.clone().unwrap_or_default();
            if pai.is_empty() {
                return Err(fail_box(game_id, ordinal, "malformed draw event"));
            }
            if matches!(prev_kind, "ankan" | "kakan" | "daiminkan") {
                rinshan.push(pai);
            } else {
                live.push(pai);
            }
        }
        prev_kind = kind;
        idx += 1;
    }
    Ok((live, rinshan))
}

/// Classified take-ledger failure: a closed quarantine `code` plus the
/// oracle-style `detail` (byte-stable `"sim replay desync ..."` text).
///
/// The ledger owns the copy domain, so every pool/queue/ownership helper
/// here fails `tile-conservation`; kyoku-header seat coverage classifies
/// `turn-order` at its own site.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrackReject {
    pub code: String,
    pub detail: String,
}

impl KyokuTrack {
    /// Install tehais and preallocate the take ledger for one kyoku.
    pub fn install(
        events: &[MjaiEvent],
        ordinal: usize,
        start_idx: usize,
        tehais: &[Vec<String>],
        game_id: &str,
    ) -> Result<Self, TrackReject> {
        let fail = |code: &str, why: &str| TrackReject {
            code: code.to_string(),
            detail: format!(
                "sim replay desync game {game_id:?} kyoku {ordinal} start_kyoku: {why}"
            ),
        };
        if tehais.len() != 4 {
            return Err(fail(
                "turn-order",
                &format!("tehais cover {} seats, not 4", tehais.len()),
            ));
        }
        let mut track = KyokuTrack {
            ordinal,
            take_taken: HashMap::new(),
            tehais: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            hands: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            rivers: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            drawn: [None, None, None, None],
            drawn_live: [None, None, None, None],
            dora: Vec::new(),
            melds: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            riichi_declared: [false, false, false, false],
            drawer: None,
            exp_drawer: None,
            kan_pending: false,
            draws: 0,
            live_queues: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            rinshan_queues: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            tsumo_counts: [0, 0, 0, 0],
            first_draws: [None, None, None, None],
        };
        for (seat, teh) in tehais.iter().enumerate().take(4) {
            if teh.len() != 13 {
                return Err(fail(
                    "tile-conservation",
                    &format!("seat {seat} tehais hold {} tiles, not 13", teh.len()),
                ));
            }
            track.tehais[seat] = teh.clone();
            for pai in teh {
                let copy = track.take_next(pai, game_id)?;
                track.hands[seat].push(copy);
            }
        }
        let (live_draws, rinshan_draws) = split_draws(events, start_idx, game_id, ordinal)?;
        let mut live_copies = Vec::with_capacity(live_draws.len());
        for pai in &live_draws {
            live_copies.push(track.take_next(pai, game_id)?);
        }
        let mut rinshan_copies = Vec::with_capacity(rinshan_draws.len());
        for pai in &rinshan_draws {
            rinshan_copies.push(track.take_next(pai, game_id)?);
        }
        // Assign queued draws in log order (post-kan tsumo takes rinshan).
        let mut live_pos = 0usize;
        let mut rinshan_pos = 0usize;
        let mut prev_kind = "start_kyoku";
        let mut idx = start_idx + 1;
        while idx < events.len() {
            let kind = events[idx].type_.as_str();
            if is_end_or_boundary(kind) {
                break;
            }
            if is_transparent(kind) {
                idx += 1;
                continue;
            }
            if kind == "tsumo" {
                // Shape already validated by `split_draws` above over the
                // identical event walk, so only the seat resolves here.
                let seat = events[idx].actor_seat("tsumo").map_err(|e| TrackReject {
                    code: "turn-order".to_string(),
                    detail: desync_detail(game_id, ordinal, &e),
                })?;
                if matches!(prev_kind, "ankan" | "kakan" | "daiminkan") {
                    let copy = *rinshan_copies.get(rinshan_pos).ok_or_else(|| {
                        fail_box(game_id, ordinal, "rinshan draw without a rinshan tile")
                    })?;
                    rinshan_pos += 1;
                    track.rinshan_queues[seat as usize].push(copy);
                } else {
                    let copy = *live_copies.get(live_pos).ok_or_else(|| {
                        fail_box(game_id, ordinal, "live draw without a live tile")
                    })?;
                    live_pos += 1;
                    track.live_queues[seat as usize].push(copy);
                }
            }
            prev_kind = kind;
            idx += 1;
        }
        Ok(track)
    }

    /// Take the next pool copy of `pai` in global wall order (fail closed).
    pub fn take_next(&mut self, pai: &str, game_id: &str) -> Result<u8, TrackReject> {
        let pool = copies_of_string(pai)
            .map_err(|e| fail_box(game_id, self.ordinal, &e.to_string()))?;
        let taken = self.take_taken.get(pai).copied().unwrap_or(0);
        if taken >= pool.len() {
            return Err(fail_box(
                game_id,
                self.ordinal,
                &format!("tile string {pai:?} overused (tile conservation)"),
            ));
        }
        self.take_taken.insert(pai.to_string(), taken + 1);
        Ok(pool[taken])
    }

    /// Pop the queued draw copy for `seat`, checking the logged string.
    /// `after_kan` selects the rinshan queue (post-kan tsumo).
    pub fn pop_draw(&mut self, seat: u8, pai: &str, game_id: &str) -> Result<u8, TrackReject> {
        let queue = if self.kan_pending {
            &mut self.rinshan_queues[seat as usize]
        } else {
            &mut self.live_queues[seat as usize]
        };
        if queue.is_empty() {
            return Err(fail_box(
                game_id,
                self.ordinal,
                &format!("seat {seat} drew a different tile than logged"),
            ));
        }
        let copy = queue.remove(0);
        let rendered =
            mjai_string_of(copy).map_err(|e| fail_box(game_id, self.ordinal, &e.to_string()))?;
        if rendered != pai {
            return Err(fail_box(
                game_id,
                self.ordinal,
                &format!("seat {seat} drew a different tile than logged"),
            ));
        }
        Ok(copy)
    }

    /// Remove one tracked copy rendering `pai` (drawn tile preferred).
    pub fn remove_one(
        hand: &mut Vec<u8>,
        pai: &str,
        drawn: Option<u8>,
        game_id: &str,
        ordinal: usize,
        what: &str,
    ) -> Result<u8, TrackReject> {
        if let Some(d) = drawn
            && hand.contains(&d)
                && let Ok(rendered) = mjai_string_of(d)
                    && rendered == pai {
                        let pos = hand.iter().position(|t| *t == d).unwrap_or(0);
                        hand.remove(pos);
                        return Ok(d);
                    }
        if let Some(pos) = hand.iter().position(|t| {
            mjai_string_of(*t)
                .map(|rendered| rendered == pai)
                .unwrap_or(false)
        }) {
            Ok(hand.remove(pos))
        } else {
            Err(fail_box(
                game_id,
                ordinal,
                &format!("no tracked copy of discard {pai:?} in hand ({what})"),
            ))
        }
    }

    /// Resolve meld copies from the tracked hand (mirrors `_tracked_consumed`).
    ///
    /// Each logged string consumes one tracked tile rendering that exact
    /// string (red-aware); a picked copy colliding with `called` is swapped
    /// for an unused pool copy of the same string. Result is sorted.
    pub fn resolve_consumed(
        hand: &[u8],
        consumed_strings: &[String],
        needed: usize,
        called: Option<u8>,
        game_id: &str,
        ordinal: usize,
    ) -> Result<Vec<u8>, TrackReject> {
        let mut pool: Vec<u8> = hand.to_vec();
        let mut sorted_strings: Vec<String> = consumed_strings.to_vec();
        sorted_strings.sort();
        let mut picked: Vec<u8> = Vec::with_capacity(sorted_strings.len());
        for pai in &sorted_strings {
            let mut found = None;
            for (index, candidate) in pool.iter().enumerate() {
                if mjai_string_of(*candidate)
                    .map(|rendered| rendered == *pai)
                    .unwrap_or(false)
                {
                    found = Some(index);
                    break;
                }
            }
            match found {
                Some(index) => picked.push(pool.remove(index)),
                None => {
                    return Err(fail_box(
                        game_id,
                        ordinal,
                        &format!("no tracked copy left for meld tile {pai:?}"),
                    ));
                }
            }
        }
        if picked.len() != needed {
            return Err(fail_box(
                game_id,
                ordinal,
                &format!(
                    "claim needs {needed} consumed tiles, got {}",
                    picked.len()
                ),
            ));
        }
        if let Some(called_id) = called {
            for pos in 0..picked.len() {
                if picked[pos] == called_id {
                    let pai = mjai_string_of(picked[pos])
                        .map_err(|e| fail_box(game_id, ordinal, &e.to_string()))?;
                    let used: std::collections::HashSet<u8> =
                        picked.iter().copied().chain([called_id]).collect();
                    let mut swapped = false;
                    for candidate in copies_of_string(&pai)
                        .map_err(|e| fail_box(game_id, ordinal, &e.to_string()))?
                    {
                        if !used.contains(&candidate) {
                            picked[pos] = candidate;
                            swapped = true;
                            break;
                        }
                    }
                    if !swapped {
                        return Err(fail_box(
                            game_id,
                            ordinal,
                            &format!("no distinct copy left for meld tile {pai:?}"),
                        ));
                    }
                }
            }
        }
        picked.sort_unstable();
        Ok(picked)
    }
}

fn desync_detail(game_id: &str, kyoku: usize, why: &str) -> String {
    format!("sim replay desync game {game_id:?} kyoku {kyoku} {why}")
}

/// Ledger-copy failure (pool overuse, queue mismatch, untracked copies):
/// always `tile-conservation`; header/seat failures classify at their site.
fn fail_box(game_id: &str, kyoku: usize, why: &str) -> TrackReject {
    TrackReject {
        code: "tile-conservation".to_string(),
        detail: desync_detail(game_id, kyoku, why),
    }
}

/// Slice-S2 shared fixture (test-only): kan/rinshan kyoku with transparent
/// markers between kan and replacement draw. Mirrors the throwaway probe
/// (`/tmp/s2/oracle_dump.py`); oracle split is live `['1s','2s','6s']`,
/// rinshan `['3s','4s','5s']`.
#[cfg(test)]
pub(crate) const S2_TILES: [&str; 34] = [
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p", "5p",
    "6p", "7p", "8p", "9p", "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "E",
    "S", "W", "N", "P", "F", "C",
];

#[cfg(test)]
pub(crate) fn s2_tile_at(slot: usize) -> String {
    S2_TILES[slot % S2_TILES.len()].to_string()
}

#[cfg(test)]
pub(crate) fn s2_tehais_json() -> String {
    let hands: Vec<String> = (0..4)
        .map(|seat| {
            let tiles: Vec<String> =
                (0..13).map(|i| format!("{:?}", s2_tile_at(seat * 13 + i))).collect();
            format!("[{}]", tiles.join(","))
        })
        .collect();
    format!("[{}]", hands.join(","))
}

#[cfg(test)]
pub(crate) fn s2_kan_game_text() -> String {
    let t = s2_tehais_json();
    let d: Vec<String> = (52..58).map(s2_tile_at).collect();
    let lines = vec![
        "{\"type\":\"start_game\"}".to_string(),
        format!(
            "{{\"type\":\"start_kyoku\",\"bakaze\":\"E\",\"kyoku\":1,\"honba\":0,\"kyotaku\":0,\"oya\":0,\"scores\":[25000,25000,25000,25000],\"dora_marker\":\"3m\",\"tehais\":{t}}}"
        ),
        format!("{{\"type\":\"tsumo\",\"actor\":0,\"pai\":{:?}}}", d[0]),
        format!("{{\"type\":\"dahai\",\"actor\":0,\"pai\":{:?},\"tsumogiri\":true}}", d[0]),
        format!("{{\"type\":\"tsumo\",\"actor\":1,\"pai\":{:?}}}", d[1]),
        format!("{{\"type\":\"dahai\",\"actor\":1,\"pai\":{:?},\"tsumogiri\":true}}", d[1]),
        format!("{{\"type\":\"ankan\",\"actor\":0,\"consumed\":[{0:?},{0:?},{0:?},{0:?}]}}", d[0]),
        "{\"type\":\"dora\",\"dora_marker\":\"4m\"}".to_string(),
        format!("{{\"type\":\"tsumo\",\"actor\":0,\"pai\":{:?}}}", d[2]),
        format!("{{\"type\":\"dahai\",\"actor\":0,\"pai\":{:?},\"tsumogiri\":true}}", d[2]),
        format!("{{\"type\":\"kakan\",\"actor\":0,\"pai\":{:?}}}", d[0]),
        "{\"type\":\"reach_accepted\",\"actor\":1}".to_string(),
        format!("{{\"type\":\"tsumo\",\"actor\":0,\"pai\":{:?}}}", d[3]),
        format!("{{\"type\":\"dahai\",\"actor\":0,\"pai\":{:?},\"tsumogiri\":true}}", d[3]),
        format!(
            "{{\"type\":\"daiminkan\",\"actor\":2,\"pai\":{:?},\"consumed\":[{0:?},{0:?}],\"target\":0}}",
            d[3]
        ),
        format!("{{\"type\":\"tsumo\",\"actor\":2,\"pai\":{:?}}}", d[4]),
        format!("{{\"type\":\"dahai\",\"actor\":2,\"pai\":{:?},\"tsumogiri\":true}}", d[4]),
        format!("{{\"type\":\"tsumo\",\"actor\":3,\"pai\":{:?}}}", d[5]),
        format!("{{\"type\":\"dahai\",\"actor\":3,\"pai\":{:?},\"tsumogiri\":true}}", d[5]),
        "{\"type\":\"end_kyoku\"}".to_string(),
        "{\"type\":\"end_game\"}".to_string(),
    ];
    lines.join("\n") + "\n"
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mjai_event::frame_events;

    fn sample_game() -> ParsedGame {
        let text = concat!(
            "{\"type\":\"start_game\"}\n",
            "{\"type\":\"start_kyoku\",\"bakaze\":\"E\",\"kyoku\":1,\"honba\":0,\"kyotaku\":0,\"oya\":0,\"scores\":[25000,25000,25000,25000],\"dora_marker\":\"3m\",\"tehais\":[[\"1m\",\"2m\",\"3m\",\"4m\",\"5m\",\"6m\",\"7m\",\"8m\",\"9m\",\"E\",\"S\",\"W\",\"N\"],[\"1p\",\"2p\",\"3p\",\"4p\",\"5p\",\"6p\",\"7p\",\"8p\",\"9p\",\"P\",\"F\",\"C\",\"5mr\"],[\"1s\",\"2s\",\"3s\",\"4s\",\"5s\",\"6s\",\"7s\",\"8s\",\"9s\",\"E\",\"S\",\"W\",\"N\"],[\"1m\",\"1m\",\"1m\",\"9p\",\"9p\",\"9p\",\"1s\",\"1s\",\"1s\",\"P\",\"P\",\"P\",\"C\"]]}\n",
            "{\"type\":\"tsumo\",\"actor\":0,\"pai\":\"9m\"}\n",
            "{\"type\":\"dahai\",\"actor\":0,\"pai\":\"N\",\"tsumogiri\":false}\n",
            "{\"type\":\"end_kyoku\"}\n",
            "{\"type\":\"end_game\"}\n",
        );
        parse_game(text, "sample-tenhou-stem").unwrap()
    }

    #[test]
    fn game_id_falls_back_to_object_hash() {
        let game = sample_game();
        assert!(game.game_id.starts_with("game-"));
        assert_eq!(game.game_id.len(), 17);
        // Deterministic: same stem, same id.
        let again = parse_game(
            "{\"type\":\"start_game\"}\n{\"type\":\"end_game\"}\n",
            "sample-tenhou-stem",
        )
        .unwrap();
        assert_eq!(again.game_id, game.game_id);
    }

    #[test]
    fn tehais_install_takes_in_wall_order() {
        let game = sample_game();
        let tehais: Vec<Vec<String>> = game.events[1].tehais.clone().unwrap();
        let track = KyokuTrack::install(&game.events, 0, 1, &tehais, &game.game_id).unwrap();
        assert_eq!(track.hands[0].len(), 13);
        // First take of "1m" is the block base; the aka take is 16.
        assert_eq!(track.hands[0][0], 0);
        assert!(track.hands[1].contains(&16));
        // The live "9m" draw continues its pool past the tehai take.
        assert_eq!(track.take_taken.get("9m").copied().unwrap_or(0), 2);
        assert_eq!(track.live_queues[0], vec![33]);
    }

    #[test]
    fn aka_overuse_fails_closed() {
        let game = sample_game();
        let tehais: Vec<Vec<String>> = game.events[1].tehais.clone().unwrap();
        let mut track = KyokuTrack::install(&game.events, 0, 1, &tehais, &game.game_id).unwrap();
        // Seat 1 already holds the single "5mr" copy; a second take is an
        // impossible log and must quarantine, never wrap the pool.
        assert!(track.take_next("5mr", &game.game_id).is_err());
    }

    #[test]
    fn walled_games_parse_with_wall_tiles() {
        // S7: valid 136-tile walls are accepted (never quarantined); the
        // permutation rides `ParsedGame::wall_tiles` for real-digest
        // binding. Malformed walls fail closed as framing.
        let text = format!(
            "{{\"type\":\"start_game\",\"wall\":[{}]}}\n{{\"type\":\"end_game\"}}\n",
            (0..136)
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        let game = parse_game(&text, "walled").expect("valid wall parses");
        assert_eq!(
            game.wall_tiles,
            Some(
                (0..136u16)
                    .map(|v| u8::try_from(v).unwrap())
                    .collect::<Vec<u8>>()
            )
        );
        assert!(game.wall_tiles.as_ref().unwrap().contains(&16));
        // Duplicate tile: not a permutation -> framing.
        let bad = format!(
            "{{\"type\":\"start_game\",\"wall\":[{}]}}\n{{\"type\":\"end_game\"}}\n",
            (0..136).map(|_| "0").collect::<Vec<_>>().join(",")
        );
        assert!(matches!(parse_game(&bad, "bad-wall"), Err(GameReject::Framing(_))));
        // Wall-less games carry None (SIM-mark path).
        let plain = "{\"type\":\"start_game\"}\n{\"type\":\"end_game\"}\n";
        assert_eq!(parse_game(plain, "plain").unwrap().wall_tiles, None);
    }

    #[test]
    fn frame_events_smoke() {
        let text = "{\"type\":\"start_game\"}\n{\"type\":\"end_game\"}\n";
        assert_eq!(frame_events(text, "x").unwrap().len(), 2);
    }

    // --- Slice S2 pins: take order + draw partition vs the Python oracle ---
    //
    // The expected live/rinshan pais, queue ids, and take counts below are
    // the oracle's `_split_kyoku_draws` + `_tracker_init` outputs verbatim
    // (throwaway probe /tmp/s2/oracle_dump.py); the fixture text lives in
    // `s2_kan_game_text` above so the walk gate reuses it byte-identically.

    #[test]
    fn live_wall_base_is_seventy() {
        // 136 tiles minus 4x13 dealt minus the 14-tile dead wall.
        assert_eq!(LIVE_WALL_BASE, 70);
        assert_eq!(LIVE_WALL_BASE, 136 - 52 - 14);
    }

    #[test]
    fn kan_rinshan_partition_pins_oracle() {
        let game = parse_game(&s2_kan_game_text(), "s2-kan").unwrap();
        // Oracle `_split_kyoku_draws`: live ['1s','2s','6s'], rinshan
        // ['3s','4s','5s'] (dora / reach_accepted never reset the kan).
        let (live, rinshan) = split_draws(&game.events, 1, &game.game_id, 0).unwrap();
        assert_eq!(live, vec!["1s".to_string(), "2s".to_string(), "6s".to_string()]);
        assert_eq!(rinshan, vec!["3s".to_string(), "4s".to_string(), "5s".to_string()]);
        // Per-seat queues hold those copies in log order (seat 0 draws live
        // first, then two rinshan replacements).
        let tehais: Vec<Vec<String>> = game.events[1].tehais.clone().unwrap();
        let track = KyokuTrack::install(&game.events, 0, 1, &tehais, &game.game_id).unwrap();
        assert_eq!(track.live_queues[0], vec![73]);
        assert_eq!(track.rinshan_queues[0], vec![81, 85]);
        assert_eq!(track.live_queues[1], vec![77]);
        assert!(track.rinshan_queues[1].is_empty());
        assert!(track.live_queues[2].is_empty());
        assert_eq!(track.rinshan_queues[2], vec![90]);
        assert_eq!(track.live_queues[3], vec![93]);
        assert!(track.rinshan_queues[3].is_empty());
        // Take order: tehais seats 0..3 (52 takes), then live in log order,
        // then rinshan — 58 takes total.
        let taken: usize = track.take_taken.values().sum();
        assert_eq!(taken, 58);
        assert_eq!(track.take_taken.get("1m").copied().unwrap_or(0), 2);
        assert_eq!(track.take_taken.get("1s").copied().unwrap_or(0), 2);
        assert_eq!(track.take_taken.get("6s").copied().unwrap_or(0), 2);
        assert_eq!(track.take_taken.get("7s").copied().unwrap_or(0), 1);
    }

    #[test]
    fn split_stops_at_kyoku_boundary() {
        let text = concat!(
            "{\"type\":\"start_game\"}\n",
            "{\"type\":\"start_kyoku\",\"bakaze\":\"E\",\"kyoku\":1,\"honba\":0,\"kyotaku\":0,\"oya\":0,\"scores\":[25000,25000,25000,25000],\"dora_marker\":\"3m\",\"tehais\":[[\"1m\",\"2m\",\"3m\",\"4m\",\"5m\",\"6m\",\"7m\",\"8m\",\"9m\",\"E\",\"S\",\"W\",\"N\"],[\"1p\",\"2p\",\"3p\",\"4p\",\"5p\",\"6p\",\"7p\",\"8p\",\"9p\",\"P\",\"F\",\"C\",\"5mr\"],[\"1s\",\"2s\",\"3s\",\"4s\",\"5s\",\"6s\",\"7s\",\"8s\",\"9s\",\"E\",\"S\",\"W\",\"N\"],[\"1m\",\"1m\",\"1m\",\"9p\",\"9p\",\"9p\",\"1s\",\"1s\",\"1s\",\"P\",\"P\",\"P\",\"C\"]]}\n",
            "{\"type\":\"end_kyoku\"}\n",
            "{\"type\":\"tsumo\",\"actor\":0,\"pai\":\"1m\"}\n",
            "{\"type\":\"end_game\"}\n",
        );
        let game = parse_game(text, "s2-boundary").unwrap();
        let (live, rinshan) = split_draws(&game.events, 1, &game.game_id, 0).unwrap();
        assert!(live.is_empty());
        assert!(rinshan.is_empty());
    }

    #[test]
    fn malformed_draw_fails_closed() {
        // Codes follow the walk's draw convention (bad actor turn-order,
        // missing pai tile-conservation); install propagates the split.
        // Tsumo without a pai string.
        let text = concat!(
            "{\"type\":\"start_game\"}\n",
            "{\"type\":\"start_kyoku\",\"bakaze\":\"E\",\"kyoku\":1,\"honba\":0,\"kyotaku\":0,\"oya\":0,\"scores\":[25000,25000,25000,25000],\"dora_marker\":\"3m\",\"tehais\":[[\"1m\",\"2m\",\"3m\",\"4m\",\"5m\",\"6m\",\"7m\",\"8m\",\"9m\",\"E\",\"S\",\"W\",\"N\"],[\"1p\",\"2p\",\"3p\",\"4p\",\"5p\",\"6p\",\"7p\",\"8p\",\"9p\",\"P\",\"F\",\"C\",\"5mr\"],[\"1s\",\"2s\",\"3s\",\"4s\",\"5s\",\"6s\",\"7s\",\"8s\",\"9s\",\"E\",\"S\",\"W\",\"N\"],[\"1m\",\"1m\",\"1m\",\"9p\",\"9p\",\"9p\",\"1s\",\"1s\",\"1s\",\"P\",\"P\",\"P\",\"C\"]]}\n",
            "{\"type\":\"tsumo\",\"actor\":0}\n",
            "{\"type\":\"end_game\"}\n",
        );
        let game = parse_game(text, "s2-bad-pai").unwrap();
        // Missing pai is tile-conservation, on both the partition helper
        // and the install path that propagates it.
        let err = split_draws(&game.events, 1, &game.game_id, 0).unwrap_err();
        assert_eq!(err.code, "tile-conservation");
        assert!(err.detail.contains("malformed draw event"));
        let tehais: Vec<Vec<String>> = game.events[1].tehais.clone().unwrap();
        let err = KyokuTrack::install(&game.events, 0, 1, &tehais, &game.game_id).unwrap_err();
        assert_eq!(err.code, "tile-conservation");
        assert!(err.detail.contains("malformed draw event"));
        // Tsumo with an out-of-range actor.
        let text = text.replace(
            "{\"type\":\"tsumo\",\"actor\":0}",
            "{\"type\":\"tsumo\",\"actor\":7,\"pai\":\"1m\"}",
        );
        let game = parse_game(&text, "s2-bad-actor").unwrap();
        // A bad actor is turn-order, on both paths.
        let err = split_draws(&game.events, 1, &game.game_id, 0).unwrap_err();
        assert_eq!(err.code, "turn-order");
        let tehais: Vec<Vec<String>> = game.events[1].tehais.clone().unwrap();
        let err = KyokuTrack::install(&game.events, 0, 1, &tehais, &game.game_id).unwrap_err();
        assert_eq!(err.code, "turn-order");
    }
}
