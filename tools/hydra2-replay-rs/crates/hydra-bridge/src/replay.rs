//! Per-game plane replay over [`stage_one`] (game-pull training path).
//!
//! [`replay_game_planes`] frames, gates, and walks ONE game from raw lines
//! and serves its staged planes as ``{name: bytes}`` — the same canonical
//! 26 planes (same order, same LE bytes) the file fill commits, minus the
//! file walk.
//! Walled regime follows wall CONTENT (fill parity: a wall-bearing game
//! walks unfolded with ``wall_digest: None``). History planes are dense
//! ``[rows, t_len]`` with ``t_len`` from [`pick_t_len`], exactly like a
//! single-game fill, so Python assembles with the shared code path.
//!
//! Quarantine is data, never an error: a rejected game returns
//! ``quarantined=True`` with the feed reason string + event index, and the
//! caller counts it like any fill quarantine. Only malformed ARGUMENTS
//! raise (over-cap history fails closed like the fill path). The GIL is
//! released for the whole stage.

use hydra_feed::fill::{pick_t_len, row_bytes, stage_one, FillOut, Scratch, StageJob, N_PLANES, inject_wall};
use hydra_feed::gate::reason_name as gate_reason_name;
use hydra_feed::ledger::{walk_reason_name, PLANE_NAMES};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyDict};

/// Render a reject reason from either family (gate `REASON_*` 0–8, walk
/// `WALK_*` 10+; the families are disjoint by construction, both fall back
/// to `"other"` outside their range).
fn reject_name(reason: u8) -> &'static str {
    if reason < 10 {
        gate_reason_name(reason)
    } else {
        walk_reason_name(reason)
    }
}

/// Frame + gate + walk one game; serve staged planes (shared core behind
/// the two `replay_game_planes*` entry points).
///
/// `events` is the final game bytes (wall already bound by the caller).
/// Returns ``(planes, rows, t_len, quarantined, reason, event_idx)`` where
/// `planes` maps plane names (`hydra_feed::ledger::PLANE_NAMES` order) to
/// raw LE bytes (fixed planes ``[rows * stride]``, history planes dense
/// ``[rows * t_len]``) and `reason`/`event_idx`
/// describe the quarantine (empty/zero when clean).
fn run_planes(
    py: Python<'_>,
    events: Vec<u8>,
    game_idx: u32,
) -> PyResult<(Py<PyDict>, u32, u32, bool, String, u32)> {
    // Stage + fill run detached (pure CPU, no Python): only the dict build
    // below touches the interpreter.
    let staged: Result<Staged, String> = py.detach(|| stage_game(events, game_idx));
    staged_to_py(py, staged)
}

/// Staged-game transfer between the off-GIL core and the dict builders.
enum Staged {
    Rows {
        bufs: [Vec<u8>; N_PLANES],
        rows: u32,
        t_len: u32,
    },
    Quarantined {
        reason: String,
        event_idx: u32,
    },
}

/// Frame + gate + walk + fill one game with zero Python interaction.
///
/// Pure over its inputs; the caller arranges GIL state (one detach for a
/// whole batch, `py.detach` for single games). Wall binding (if any) is the
/// caller's job — `events` arrives final.
fn stage_game(events: Vec<u8>, game_idx: u32) -> Result<Staged, String> {
    let job = StageJob {
        game_idx,
        object_id: 0,
        bytes: events,
        wall_digest: None,
    };
    let game = match stage_one(&job) {
        Ok(g) => g,
        Err(reject) => {
            let stub = reject.stub;
            return Ok(Staged::Quarantined {
                reason: reject_name(stub.reason).to_string(),
                event_idx: stub.event_idx as u32,
            });
        }
    };
    let rows = game.rows as usize;
    let mut scratch = Scratch::new();
    scratch.commit(&game);
    let max_hist = scratch.max_hist();
    let t_len = match pick_t_len(max_hist) {
        Some(t) => t,
        None => {
            return Err(format!(
                "game {game_idx} history {max_hist} exceeds model bucket cap; \
                 rows are never truncated",
            ));
        }
    };
    // Caller bufs sized exactly (single game always fits: caps derive
    // from the staged rows themselves, never starve by construction).
    let mut bufs: [Vec<u8>; N_PLANES] = Default::default();
    let mut caps = [0usize; N_PLANES];
    let mut i = 0usize;
    while i < N_PLANES {
        let cap = rows.saturating_mul(row_bytes(i, t_len));
        caps[i] = cap;
        bufs[i] = vec![0u8; cap];
        i += 1;
    }
    let mut ptrs = [0u64; N_PLANES];
    let mut j = 0usize;
    while j < N_PLANES {
        // Soundness: the fill writes through these addresses, so they
        // must derive from as_mut_ptr (writing through as_ptr-derived
        // pointers is Tree-Borrows UB, even with equal addresses).
        ptrs[j] = bufs[j].as_mut_ptr() as u64;
        j += 1;
    }
    let out: FillOut = scratch
        .fill_pinned(&ptrs, &caps)
        .map_err(|e| format!("fill failed: {e:?}"))?;
    Ok(Staged::Rows {
        bufs,
        rows: out.rows,
        t_len: out.t_len as u32,
    })
}

/// Serve one staged result as Python planes (holds the GIL).
fn staged_to_py(
    py: Python<'_>,
    staged: Result<Staged, String>,
) -> PyResult<(Py<PyDict>, u32, u32, bool, String, u32)> {
    match staged {
        Err(message) => Err(pyo3::exceptions::PyValueError::new_err(message)),
        Ok(Staged::Quarantined { reason, event_idx }) => {
            let dict = PyDict::new(py);
            Ok((dict.into(), 0, 0, true, reason, event_idx))
        }
        Ok(Staged::Rows { bufs, rows, t_len }) => {
            let dict = PyDict::new(py);
            let mut k = 0usize;
            while k < N_PLANES {
                // Index 12 serves packed bytes; split ids/len like the fill
                // path does so callers see the unpacked `legal_ids` /
                // `legal_len` entries (never `legal_packed`).
                if k == 12 {
                    let packed = &bufs[12];
                    let mut ids = Vec::with_capacity(rows as usize * 128);
                    let mut lens = Vec::with_capacity(rows as usize * 8);
                    let mut r = 0usize;
                    while r < rows as usize {
                        let base = r * 136;
                        ids.extend_from_slice(&packed[base..base + 128]);
                        lens.extend_from_slice(&packed[base + 128..base + 136]);
                        r += 1;
                    }
                    dict.set_item("legal_ids", PyByteArray::new(py, &ids))?;
                    dict.set_item("legal_len", PyByteArray::new(py, &lens))?;
                } else {
                    dict.set_item(PLANE_NAMES[k], PyByteArray::new(py, &bufs[k]))?;
                }
                k += 1;
            }
            Ok((dict.into(), rows, t_len, false, String::new(), 0))
        }
    }
}

/// Frame + gate + walk one game; serve staged planes.
///
/// `events` is the raw game bytes exactly as the log framed them (wall-less
/// games, or walled games whose embedded wall is already the bound one).
#[pyfunction]
fn replay_game_planes(
    py: Python<'_>,
    events: Vec<u8>,
    game_idx: u32,
) -> PyResult<(Py<PyDict>, u32, u32, bool, String, u32)> {
    run_planes(py, events, game_idx)
}

/// Frame + gate + walk one game with an overriding wall; serve staged planes.
///
/// `events` is the raw framed game bytes; when `wall` is `Some`, its 136
/// entries replace the first event's wall in Rust (no Python re-serialization
/// of every event). `None` behaves exactly like [`replay_game_planes`].
#[pyfunction]
#[pyo3(signature = (events, game_idx, wall = None))]
fn replay_game_planes_wall(
    py: Python<'_>,
    events: Vec<u8>,
    game_idx: u32,
    wall: Option<Vec<u32>>,
) -> PyResult<(Py<PyDict>, u32, u32, bool, String, u32)> {
    let bytes = match wall {
        Some(w) => inject_wall(&events, &w),
        None => events,
    };
    run_planes(py, bytes, game_idx)
}

/// Expand a batch of games in one call; serve per-game staged planes.
///
/// `items` is `(events, game_idx, wall)` per game in pull order (`wall`
/// `Some` splices an overriding wall in Rust, `None` walks the embedded
/// content). Returns one ``(planes, rows, t_len, quarantined, reason,
/// event_idx)`` tuple per input game, in order, with the exact shapes
/// [`replay_game_planes_wall`] serves. The whole batch stages under a
/// single GIL release; only argument handling and the dict builds hold it.
/// Per-item stage failures ride as quarantine data (same text the serial
/// path raises), never as a call-level error — ordering and quarantine
/// accounting match the serial pull game-for-game.
#[pyfunction]
fn expand_games(
    py: Python<'_>,
    items: Vec<(Vec<u8>, u32, Option<Vec<u32>>)>,
) -> PyResult<Vec<(Py<PyDict>, u32, u32, bool, String, u32)>> {
    let staged_all: Vec<Result<Staged, String>> = py.detach(|| {
        items
            .into_iter()
            .map(|(events, game_idx, wall)| {
                let bytes = match wall {
                    Some(w) => inject_wall(&events, &w),
                    None => events,
                };
                stage_game(bytes, game_idx)
            })
            .collect()
    });
    let mut out = Vec::with_capacity(staged_all.len());
    for staged in staged_all {
        out.push(staged_to_py(py, staged)?);
    }
    Ok(out)
}

/// Register per-game replay fns on the extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(replay_game_planes, m)?)?;
    m.add_function(wrap_pyfunction!(replay_game_planes_wall, m)?)?;
    m.add_function(wrap_pyfunction!(expand_games, m)?)?;
    Ok(())
}

#[cfg(test)]
mod replay_tests {
    use super::*;

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

    fn tsumo_line(actor: u8, pai: &str) -> String {
        format!("{{\"type\":\"tsumo\",\"actor\":{actor},\"pai\":{pai:?}}}")
    }

    fn dahai_line(actor: u8, pai: &str) -> String {
        format!("{{\"type\":\"dahai\",\"actor\":{actor},\"pai\":{pai:?},\"tsumogiri\":true}}")
    }

    /// F1 shape (wall-less): 5 tsumogiri discards, no hora.
    fn f1_text() -> Vec<u8> {
        let mut lines = vec![
            "{\"type\":\"start_game\"}".to_string(),
            kyoku_line(&TEHAIS, 0),
        ];
        for (k, pai) in ["5pr", "5p", "5p", "5p", "6p"].iter().enumerate() {
            let actor = (k % 4) as u8;
            lines.push(tsumo_line(actor, pai));
            lines.push(dahai_line(actor, pai));
        }
        lines.push("{\"type\":\"end_game\"}".to_string());
        let mut text = lines.join("\n");
        text.push('\n');
        text.into_bytes()
    }

    #[test]
    fn stage_one_golden_ok() {
        let job = StageJob {
            game_idx: 7,
            object_id: 0,
            bytes: f1_text(),
            wall_digest: None,
        };
        let g = stage_one(&job).expect("golden stages");
        assert_eq!(g.rows, 5);
        let chosen: Vec<i64> = (0..5)
            .map(|r| {
                let o = r as usize * 8;
                i64::from_le_bytes(g.planes[6][o..o + 8].try_into().unwrap())
            })
            .collect();
        assert_eq!(chosen, [192, 193, 193, 193, 196]);
    }

    #[test]
    fn stage_one_garbage_quarantines() {
        let job = StageJob {
            game_idx: 9,
            object_id: 0,
            bytes: b"not json at all\n".to_vec(),
            wall_digest: None,
        };
        let err = stage_one(&job).expect_err("garbage quarantines");
        assert_eq!(err.stub.game_idx, 9);
    }
}
