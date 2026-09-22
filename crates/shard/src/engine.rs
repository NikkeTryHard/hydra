//! Narrow engine-out protocol: frozen seat-filtered answers for wall-less replay.
//!
//! Reference (read-only): `src/hydra2/engines/riichienv/log_replay.py`
//! (drained-oracle semantics, SIM mark v1) and
//! `src/hydra2/engines/riichienv/single_pass.py` (live-engine semantics,
//! SIM mark v2). The stock engine stays the SOLE rules authority: this
//! module freezes the narrow framed protocol the driver consults instead
//! of guessing — offered legals, riichi candidates, ankan/kakan
//! availability, ron/window state — and every answer is seat-filtered
//! (acting-seat tiles plus public booleans only; foreign concealed tiles
//! are unrepresentable in every output).
//!
//! Backend: the free functions below implement the drained-oracle (v1)
//! semantics natively, pinned rule-by-rule to the Python sites in the
//! table and locked by `tests/s6_engine.rs` plus the full `parity_84`
//! identity gate. The protocol is versioned ([`EngineVersion`]); v2 (live
//! single-pass deltas: kuikae-strict discards, complete chi, native
//! kyushu) rides the same query/answer types once its rules are pinned
//! against the live engine — until then requesting v2 fails closed
//! instead of approximating.
//! Provenance of each rule (Python site -> native rule):
//!
//! | Offer | Python authority | Native rule |
//! |---|---|---|
//! | discards | drained `raw_legals` via `legal_view` (`_expand_nonclaim_legals`) | one bit per distinct MJAI string in concealed (+ live-drawn string), pool-first tile id |
//! | tsumogiri twin | canonical-only twin iff slot tile == drawn (`expand_engine_legals`) | twin iff live drawn equals the pool-first id of its string |
//! | riichi_discard | `RIICHI` slot + `riichienv.check_riichi_candidates(own_hand)` | gate (live, closed modulo ankan, not riichi, score >= 1000, wall >= 4) + tenpai-discards via [`win_shape_14`] |
//! | ankan | drained ankan slots (draw-turn actions only) | live draw + wall > 0 + full quad + (riichi => drawn tile in quad) |
//! | kakan | drained kakan slots + prior-pon reference | prior pon + fourth copy present + not riichi |
//! | kyushu | drained engine never offers (0/42553 oracle bits) | never offered in masks (ryukyoku inference keeps [`kyushu_first_draw`]) |
//! | chi/pon/daiminkan (claim rows) | non-claim expansion + chosen forcing + raw-chi variants (~logged claim only) | `{pass, chosen}` exactly (ron bit iff chosen ron) |
//! | ron | stashed hora step (`{pass, ron}`) | `{pass, ron}` exactly |
//! | tsumo bit | drained hora slot on taken wins (never declined in corpus) | present iff the logged action is the tsumo win |
//! | call_window | throwaway-engine `window_open` (phase + responder legals) | responder pon/chi counts + riichi shape/river + log-ron on this discard |

use std::collections::{HashMap, HashSet};

use crate::stream::TrackedMeld;
use crate::tile::{copies_of_string, is_yaochu_norm, mjai_string_of, norm_pai, physical_of};

/// Wall-less backend version. V1 (drained-oracle) is the default and the
/// only backend pinned to the frozen oracle; V2 (live single-pass deltas)
/// fails closed until its rules are pinned rule-by-rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineVersion {
    V1,
    V2,
}

impl EngineVersion {
    /// Marker bound into `derivation_hash` instead of a wall digest.
    /// V1 matches the drained oracle; V2 matches the single-pass module.
    /// The two marks never mix silently: rows carry exactly one.
    pub fn derivation_mark(&self) -> &'static str {
        match self {
            EngineVersion::V1 => crate::parity::SIM_DERIVATION_MARK,
            EngineVersion::V2 => "sim-replay-wall-less-v2",
        }
    }
}

/// Live-wall countdown base (136 minus 4x13 dealt minus the 14-tile dead
/// wall). The wall gate (riichi/kan availability) reads this countdown.
pub const LIVE_WALL_BASE: u32 = crate::stream::LIVE_WALL_BASE;

/// Riichi declaration needs a full round left: no riichi with fewer than 4
/// wall tiles remaining (pinned: 46 closed-tenpai wall<4 rows withheld,
//  all 1553 riichi rows carry wall >= 4).
pub const RIICHI_MIN_WALL: u32 = 4;

/// Riichi costs 1000 points: withheld below (pinned: single score-900
/// closed-tenpai row withheld).
pub const RIICHI_MIN_SCORE: i32 = 1000;

/// Kan needs a replacement tile: withheld on an exhausted live wall
/// (pinned: 2 live-draw quad rows at wall 0 withheld).
pub const KAN_MIN_WALL: u32 = 1;

/// Classified engine failure: a named quarantine with game + kyoku + step
/// detail (mirrors the Python `ContractError` desync text). Callers map
/// this onto the `engine-desync` reason code, never a silent drop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineDesync {
    pub game_id: String,
    pub kyoku: i64,
    pub step: String,
    pub why: String,
}

impl EngineDesync {
    pub fn new(game_id: &str, kyoku: i64, step: &str, why: impl Into<String>) -> Self {
        EngineDesync {
            game_id: game_id.to_string(),
            kyoku,
            step: step.to_string(),
            why: why.into(),
        }
    }

    /// Oracle-style detail line (`sim replay desync game {id:?} kyoku
    /// {kyoku} {step}: {why}`), shared with the ledger rejects.
    pub fn detail(&self) -> String {
        format!(
            "sim replay desync game {:?} kyoku {} {}: {}",
            self.game_id, self.kyoku, self.step, self.why
        )
    }
}

/// Seat-filtered frozen inputs for one draw decision (mirrors `_SimStep`
/// actor fields plus the public table state the offer rules read).
/// Construction validates the firewall: the acting seat's hand must be
/// non-empty (an empty acting hand is a desync, never guessed).
///
/// `hand` is the collapsed take multiset (pool-first copy per string
/// occurrence, exactly like the drained `_distinct_copies` expansion the
/// oracle feeds its legal view): offer rules read held takes only, so
/// melded/discarded takes never offer. `drawn_true` carries the true
/// drawn take for the riichi-forced pair. Foreign seats contribute
/// nothing, so the seat filter holds by construction.
#[derive(Debug)]
pub struct SeatView<'a> {
    pub game_id: &'a str,
    pub kyoku: i64,
    pub step: &'static str,
    pub seat: u8,
    /// Collapsed hand takes (pool-first per string occurrence), live draw
    /// included as its collapsed copy. Mirrors `step.hand`.
    pub hand: Vec<u8>,
    /// Collapsed live draw (pool-first of the drawn string) when the seat
    /// drew this turn; mirrors the drained step tile the twin/forced/
    /// candidate rules read. Stale draws never ride here.
    pub drawn: Option<u8>,
    /// True drawn take (global pool order) when the seat drew this turn.
    /// Only the riichi-forced pair reads it; every other rule reads
    /// collapsed takes.
    pub drawn_true: Option<u8>,
    /// This seat's melds (open count + prior-pon lookup for kakan).
    pub melds: &'a [TrackedMeld],
    /// Whether this seat already declared riichi.
    pub riichi: bool,
    /// Tracked score at decision time (riichi sticks already deducted).
    pub score: i32,
    /// Live-wall countdown at decision time.
    pub wall: u32,
}
/// Constructor params for [`SeatView::new`] (11 fields; grouped so the
/// constructor stays under the too-many-arguments threshold).
pub struct SeatViewParams<'a> {
    pub game_id: &'a str,
    pub kyoku: i64,
    pub step: &'static str,
    pub seat: u8,
    pub hand: Vec<u8>,
    pub drawn: Option<u8>,
    pub drawn_true: Option<u8>,
    pub melds: &'a [TrackedMeld],
    pub riichi: bool,
    pub score: i32,
    pub wall: u32,
}

impl<'a> SeatView<'a> {
    pub fn new(params: SeatViewParams<'a>) -> Result<Self, EngineDesync> {
        let SeatViewParams {
            game_id,
            kyoku,
            step,
            seat,
            hand,
            drawn,
            drawn_true,
            melds,
            riichi,
            score,
            wall,
        } = params;
        if seat > 3 {
            return Err(EngineDesync::new(
                game_id,
                kyoku,
                step,
                format!("seat-filtered view for invalid seat {seat}"),
            ));
        }
        if hand.is_empty() && drawn.is_none() {
            return Err(EngineDesync::new(
                game_id,
                kyoku,
                step,
                format!("seat-filtered hands must cover seat {seat} (acting-seat hand is empty)"),
            ));
        }
        Ok(SeatView {
            game_id,
            kyoku,
            step,
            seat,
            hand,
            drawn,
            drawn_true,
            melds,
            riichi,
            score,
            wall,
        })
    }

    fn fail(&self, why: impl Into<String>) -> EngineDesync {
        EngineDesync::new(self.game_id, self.kyoku, self.step, why)
    }

    /// Closed for riichi purposes: no open meld (ankan never opens).
    /// Mirrors the engine, which offers RIICHI only on closed positions
    /// (pinned: 1285 closed + 20 single-ankan riichi rows, zero with
    /// chi/pon/daiminkan/kakan melds).
    fn closed_for_riichi(&self) -> bool {
        self.melds.iter().all(|m| m.kind == "ankan")
    }

    /// Number of open meld slots for shape evaluation (every meld,
    /// ankan included, permanently removes its tiles from concealed).
    fn open_count(&self) -> usize {
        self.melds.len()
    }
}

/// Seat-filtered draw offer: exact tile ids the engine would list (pool-
/// first per string, true copies for riichi candidates). Foreign seats
/// contribute nothing; public state contributes only booleans upstream.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DrawOffer {
    /// One pool-first tile id per offered discard string.
    pub discards: Vec<u8>,
    /// Tsumogiri twin of the live drawn tile, when offered.
    pub tsumogiri: Option<u8>,
    /// Riichi declaration candidates (true copies in hand).
    pub riichi: Vec<u8>,
    /// Offered ankan quads (full sorted blocks).
    pub ankan: Vec<Vec<u8>>,
    /// Offered kakan added tiles.
    pub kakan: Vec<u8>,
}

/// Responder snapshot for window evaluation (ledger-derived, never
/// row-bound: only booleans leave the engine).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResponderView<'a> {
    pub seat: u8,
    pub riichi: bool,
    /// True-copy concealed hand (live-drawn excluded, as responders hold
    /// no live draw at window time).
    pub hand: &'a [u8],
    /// True-copy river.
    pub river: &'a [u8],
    /// Open meld count (shape evaluation).
    pub open: usize,
}

// ---------------------------------------------------------------------------
// Win-shape evaluator (yaku-blind; port of `HandEvaluator.has_win_shape).
// ---------------------------------------------------------------------------

fn type_counts(ids: &[u8]) -> [u8; 34] {
    let mut counts = [0u8; 34];
    for id in ids {
        counts[(id / 4) as usize] += 1;
    }
    counts
}

fn melds_out(counts: &mut [u8; 34], need: u8) -> bool {
    if need == 0 {
        return counts.iter().all(|c| *c == 0);
    }
    let first = counts.iter().position(|c| *c > 0);
    let i = match first {
        Some(i) => i,
        None => return false,
    };
    // Triplet.
    if counts[i] >= 3 {
        counts[i] -= 3;
        if melds_out(counts, need - 1) {
            return true;
        }
        counts[i] += 3;
    }
    // Sequence (suit tiles only: types 0..27, low 0..6 within suit).
    let suit = i / 9;
    let pos = i % 9;
    if suit < 3 && pos <= 6 && counts[i + 1] > 0 && counts[i + 2] > 0 {
        counts[i] -= 1;
        counts[i + 1] -= 1;
        counts[i + 2] -= 1;
        if melds_out(counts, need - 1) {
            return true;
        }
        counts[i] += 1;
        counts[i + 1] += 1;
        counts[i + 2] += 1;
    }
    false
}

/// Yaku-blind standard-shape win: (4 - open) melds + a pair from
/// concealed, or seven-pairs / thirteen-orphans when fully concealed.
/// Port of `HandEvaluator.has_win_shape` for the row values the driver
/// carries (tile types only; aka shares its type). Conformance is locked
/// by `tests/fixtures/s6/shapes.jsonl` (engine-labeled, incl. aka,
/// seven-pairs-with-quad, thirteen-orphans edges).
pub fn win_shape_14(concealed: &[u8], open_melds: usize) -> bool {
    if open_melds > 4 {
        return false;
    }
    let counts = type_counts(concealed);
    if open_melds == 0 {
        // Seven pairs (four-of-a-kind cannot serve as two pairs).
        if concealed.len() == 14
            && counts.iter().all(|c| *c == 0 || *c == 2)
            && counts.iter().filter(|c| **c == 2).count() == 7
        {
            return true;
        }
        // Thirteen orphans.
        let terminals: [usize; 13] = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];
        if concealed.len() == 14
            && terminals.iter().all(|t| counts[*t] >= 1)
            && counts.iter().sum::<u8>() == 14
            && counts.contains(&2)
        {
            return true;
        }
    }
    // proof: open melds 0..4, need 0..4 fits in u8.
    #[allow(clippy::cast_possible_truncation)]
    let need_melds = 4 - open_melds as u8;
    for pair in 0..34 {
        if counts[pair] >= 2 {
            let mut rest = counts;
            rest[pair] -= 2;
            if melds_out(&mut rest, need_melds) {
                return true;
            }
        }
    }
    false
}

/// Tenpai discards of a 14-tile hand: one entry per physical copy whose
/// removal leaves a 13-tile tenpai (some completion wins by shape).
/// Port of `riichienv.check_riichi_candidates` intersected with the hand
/// (pinned: exact candidate sets on all 1553 oracle riichi rows, incl.
/// per-copy pairs, aka, seven-pairs completions).
pub fn tenpai_discards(hand14: &[u8], open_melds: usize) -> Vec<u8> {
    // Tenpai-ness depends only on the removed tile's TYPE, so test each
    // distinct type once, then expand to every copy in hand.
    let mut by_type: HashMap<u8, Vec<u8>> = HashMap::new();
    for tile in hand14 {
        by_type.entry(tile / 4).or_default().push(*tile);
    }
    let mut types: Vec<u8> = by_type.keys().copied().collect();
    types.sort_unstable();
    let mut out = Vec::new();
    for ty in types {
        // 13 tiles after removing one copy of this type.
        let mut rest: Vec<u8> = Vec::with_capacity(hand14.len() - 1);
        let mut skipped = false;
        for tile in hand14 {
            if !skipped && tile / 4 == ty {
                skipped = true;
                continue;
            }
            rest.push(*tile);
        }
        // Tenpai iff some kind completes the shape. Kind representatives
        // are type-level (aka shares its type), so `kind * 4` suffices.
        // No fifth copy exists: skip waits on types already holding all four
        // copies (dead waits; same gate as the hot walk's count-space trials).
        let mut rest_counts = [0u8; 34];
        for tile in rest.iter() {
            rest_counts[(tile / 4) as usize] += 1;
        }
        let mut tenpai = false;
        for kind in 0..34u8 {
            if rest_counts[kind as usize] >= 4 {
                continue;
            }
            let mut trial = rest.clone();
            trial.push(kind * 4);
            if win_shape_14(&trial, open_melds) {
                tenpai = true;
                break;
            }
        }
        if tenpai {
            let mut copies = by_type[&ty].clone();
            copies.sort_unstable();
            out.extend(copies);
        }
    }
    out.sort_unstable();
    out
}

/// Nine-terminals first-draw inference (ryukyoku-reason path only, never a
/// mask offer: the drained engine offers no kyushu bit on any of the
/// 42553 rows). Mirrors `_thin_kyushu`: first draw of the seat with 9+
/// terminal/honor kinds across tehais plus the first draw.
pub fn kyushu_first_draw(tehais: &[String], first_draw: &str, tsumo_count: usize) -> bool {
    if tsumo_count != 1 {
        return false;
    }
    let mut kinds = HashSet::new();
    for raw in tehais.iter().chain([&first_draw.to_string()]) {
        let normed = norm_pai(raw);
        if is_yaochu_norm(normed) {
            kinds.insert(normed.to_string());
        }
    }
    kinds.len() >= 9
}

// ---------------------------------------------------------------------------
// Draw offers (native v1 backend).
// ---------------------------------------------------------------------------
/// Native v1 draw offer for one seat-filtered view.
///
/// Rules (each pinned to corpus counts in the module docs):
/// - discards: one pool-first tile per distinct string in the held takes
///   (stale-held strings included; melded/discarded takes are absent so
///   never offered);
/// - tsumogiri twin: the pool-first drawn string when live (mirrors the
///   slot==drawn twin on collapsed takes);
/// - riichi candidates behind the closed/score/wall/live gate (never when
///   already riichi);
/// - ankan behind live + wall + quad, plus same-wait under riichi (quad
///   contains the drawn take and no prior melds);
/// - kakan behind live + prior pon + fourth copy (never under riichi);
/// - kyushu never (drained-engine quirk; see [`kyushu_first_draw`]).
///
/// Riichi-forced tsumogiri discards (true takes) bypass this offer via
/// [`forced_pair`]: only `do_dahai` under riichi takes that path, pinned
/// by the 2191 forced-pair-only rows.
pub fn draw_offer(view: &SeatView<'_>) -> Result<DrawOffer, EngineDesync> {
    // Distinct strings in the held takes (pool-first tile each).
    let mut seen = HashSet::new();
    let mut discards = Vec::new();
    for tile in &view.hand {
        let pai = match mjai_string_of(*tile) {
            Ok(p) => p,
            Err(_) => continue,
        };
        if seen.insert(pai.clone())
            && let Some(first) = copies_of_string(&pai).ok().and_then(|p| p.first().copied()) {
                discards.push(first);
            }
    }
    discards.sort_unstable();
    let live = view.drawn.is_some();
    // Tsumogiri twin of the pool-first drawn string whenever live (the
    // drained twin rides the collapsed take on every live row: 38610/38610
    // live rows carry it).
    let tsumogiri = match view.drawn {
        Some(drawn) => {
            let pai =
                mjai_string_of(drawn).map_err(|e| view.fail(format!("bad drawn tile: {e}")))?;
            copies_of_string(&pai)
                .map_err(|e| view.fail(format!("bad drawn string: {e}")))?
                .first()
                .copied()
        }
        None => None,
    };
    let mut offer = DrawOffer {
        discards,
        tsumogiri,
        ..DrawOffer::default()
    };
    // Riichi declaration candidates behind the engine gate (never when
    // already riichi: declaration rows are pre-riichi by construction).
    if live
        && !view.riichi
        && view.closed_for_riichi()
        && view.score >= RIICHI_MIN_SCORE
        && view.wall >= RIICHI_MIN_WALL
    {
        let open = view.open_count();
        offer.riichi = tenpai_discards(&view.hand, open)
            .into_iter()
            .filter(|t| view.hand.contains(t))
            .collect();
    }
    // Kan answers need a live draw turn (kans are draw-turn actions:
    // 30/30 kakan rows live, post-claim rows withheld) and a live wall
    // (replacement tile).
    if live && view.wall >= KAN_MIN_WALL {
        let mut quads = ankan_quads(&view.hand);
        // Same-wait under riichi: the quad must contain the just-drawn
        // take and the seat must hold no prior melds (8/8 offered quads
        // contain it with clean melds; the withheld containing one sits
        // behind a prior ankan, the withheld non-containing one misses
        // the drawn take).
        if view.riichi {
            let drawn = view.drawn_true;
            quads.retain(|quad| {
                view.melds.is_empty() && drawn.map(|d| quad.contains(&d)).unwrap_or(false)
            });
        }
        offer.ankan = quads;
        // Added-kan answers need a visible prior pon plus the fourth copy,
        // and never appear under riichi (0/30 kakan rows riichi).
        if !view.riichi {
            offer.kakan = kakan_adds(view.melds, &view.hand);
        }
    }
    Ok(offer)
}

/// Forced pair for a drawn string: exactly one discard plus its tsumogiri
/// twin on the given tile id. Callers pass the true drawn take for
/// riichi-forced discards (`do_dahai`, pinned by the 2191 forced-pair-only
/// rows) and the collapsed take everywhere else (tsumo/ankan rows under
/// riichi read pool-first takes, pinned by the 138 riichi tsumo/ankan
/// rows).
pub fn forced_pair(tile: u8) -> DrawOffer {
    DrawOffer {
        discards: vec![tile],
        tsumogiri: Some(tile),
        ..DrawOffer::default()
    }
}

/// V2 one-shot kuikae gate: discards to withhold after a chi claim. A
/// discard is withheld when its type is same-suit within ±1 of any take
/// of the just-claimed chi (pinned: all 175 v1-vs-v2 withheld discards
/// are chi-melded (71), same-type-as-chi (19), or chi-adjacent (85); zero
/// pon/kan neighborhoods, zero distance ≥2). Pon/kan claims set no gate.
pub fn kuikae_withheld(chi_takes: &[u8], discards: &[u8]) -> Vec<u8> {
    let types: Vec<(u8, u8)> = chi_takes.iter().map(|t| (t / 4, (t / 4) / 9)).collect();
    discards
        .iter()
        .filter(|d| {
            let ty = *d / 4;
            if ty >= 27 {
                return false;
            }
            let suit = ty / 9;
            types
                .iter()
                .any(|(ct, cs)| *cs == suit && ((*ct).cast_signed() - ty.cast_signed()).abs() <= 1)
        })
        .copied()
        .collect()
}
/// V2 complete-chi enumeration: every sequence pattern containing the
/// called tile whose two consumed types are both held (max-held take per
/// type; live takes sort descending-first). Pinned by the v2-only chi
/// bits (d0190 [44,51], d0492 [57,67]+[67,68], d0299 [8,17], d0282 all
/// three positions). Honors never chi.
pub fn complete_chi(hand_true: &[u8], called: u8) -> Vec<Vec<u8>> {
    let ct = called / 4;
    if ct >= 27 {
        return Vec::new();
    }
    let suit = ct / 9;
    let rank = ct % 9;
    let mut max_held: HashMap<u8, u8> = HashMap::new();
    for t in hand_true {
        if *t == called {
            continue;
        }
        let ty = *t / 4;
        if ty / 9 != suit {
            continue;
        }
        max_held.entry(ty).and_modify(|m| *m = (*m).max(*t)).or_insert(*t);
    }
    let mut positions: Vec<(i64, i64)> = Vec::new();
    if rank >= 2 {
        positions.push((rank as i64 - 2, rank as i64 - 1));
    }
    if (1..=7).contains(&rank) {
        positions.push((rank as i64 - 1, rank as i64 + 1));
    }
    if rank <= 6 {
        positions.push((rank as i64 + 1, rank as i64 + 2));
    }
    let mut out = Vec::new();
    for (lo, hi) in positions {
        // proof: sequence bounds 1..8 fit in u8.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let lot = suit * 9 + lo as u8;
        // proof: sequence bounds 1..8 fit in u8.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let hit = suit * 9 + hi as u8;
        if let (Some(a), Some(b)) = (max_held.get(&lot), max_held.get(&hit)) {
            let mut pair = vec![*a, *b];
            pair.sort_unstable();
            out.push(pair);
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Distinct terminal/honor types in a 14-tile hand. V2 offers the
/// nine-terminals abort on a first draw with nine or more (standard
/// kyushu; pinned: exactly the 4 v2-only abort rows; v1 never offers).
pub const KYUSHU_DISTINCT_YAOCHU: usize = 9;

pub fn distinct_yaochu(hand14: &[u8]) -> usize {
    let mut set = HashSet::new();
    for t in hand14 {
        let ty = *t / 4;
        if ty >= 27 || ty % 9 == 0 || ty % 9 == 8 {
            set.insert(ty);
        }
    }
    set.len()
}



/// Full sorted quad blocks among the collapsed hand takes.
fn ankan_quads(hand: &[u8]) -> Vec<Vec<u8>> {
    let mut by_type: HashMap<u8, Vec<u8>> = HashMap::new();
    for tile in hand {
        by_type.entry(tile / 4).or_default().push(*tile);
    }
    let mut out = Vec::new();
    for tiles in by_type.values() {
        if tiles.len() == 4 {
            let mut quad = tiles.clone();
            quad.sort_unstable();
            out.push(quad);
        }
    }
    out.sort();
    out
}

/// Added-kan tiles: the one pool copy of a prior pon's type absent from
/// the pon triple, when present in hand. Mirrors the oracle's
/// deterministic added-copy resolution (contract-valid on legal logs).
fn kakan_adds(melds: &[TrackedMeld], hand: &[u8]) -> Vec<u8> {
    let copies: &[u8] = hand;
    let mut out = Vec::new();
    for meld in melds {
        if meld.kind != "pon" {
            continue;
        }
        let meld_type = match meld.tiles.first() {
            Some(t) => t / 4,
            None => continue,
        };
        let owned: HashSet<u8> = meld.tiles.iter().copied().collect();
        // Pool copies of this type: derive from any copy present.
        let probe = match copies.iter().find(|t| *t / 4 == meld_type) {
            Some(t) => *t,
            None => continue,
        };
        let pai = match mjai_string_of(probe) {
            Ok(p) => p,
            Err(_) => continue,
        };
        let mut pool = match copies_of_string(&pai) {
            Ok(p) => p,
            Err(_) => continue,
        };
        if pai.len() == 2 && pai.as_bytes()[0] == b'5'
            && let Ok(step) = physical_of(&pai) {
                let base = (step / 4) * 4;
                pool = vec![base, base + 1, base + 2, base + 3];
            }
        let missing: Vec<u8> = pool.into_iter().filter(|c| !owned.contains(c)).collect();
        if missing.len() == 1 && copies.contains(&missing[0]) {
            out.push(missing[0]);
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

// ---------------------------------------------------------------------------
// Claim / ron offers (native v1 backend).
// ---------------------------------------------------------------------------

/// Claim-response offer: the offset-None pass plus the logged claim
/// itself. The drained engine offers no chi enumeration beyond the
/// stashed (logged) claim — string-distinct chi rejoining is a v2
/// (live-engine) semantic, never v1 (pinned: 190 kamicha-legal + 31
/// non-kamicha extras vanish; zero missing chi/pon/daiminkan globally).
/// Ron never rides a non-ron claim row (0/42553).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaimOffer {
    /// Always true in v1 (every claim row carries pass id 0).
    pub pass: bool,
}

/// Ron-response offer: exactly `{pass, ron}` (pinned: all ron rows carry
/// precisely these two bits).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RonOffer {
    pub pass: bool,
}

// ---------------------------------------------------------------------------
// Window evaluation (native v1 backend).
// ---------------------------------------------------------------------------

/// Whether the engine opens a claim window after a discard: some
/// responder holds a legal response (pon counts, kamicha chi, or a
/// furiten-clean riichi shape). Ports the responder rules the throwaway
/// engine answers natively; ron by a NON-riichi responder is invisible to
/// those rules, so the walk ORs the log fact (a logged ron proves the
/// window was open — sound, since the engine must have offered it).
/// Corpus-complete: every oracle window opens by one of these rules
/// (113/113 trailing ron windows via the log fact; zero draw-row history
/// gaps). Kan windows never emit (`chankan` suppresses pon/chi exactly
/// like the oracle, which additionally never emits post-kakan).
pub fn window_open(
    discarder: u8,
    tile_norm: &str,
    responders: &[ResponderView<'_>],
    chankan: bool,
) -> bool {
    for r in responders {
        if r.seat == discarder {
            continue;
        }
        if r.riichi {
            if !shape_win_with(r.hand, tile_norm, r.open) {
                continue;
            }
            // River furiten proxy (mirrors `_responder_eval`): winning
            // tile seen, or any river tile completing the shape.
            let river_norms: HashSet<String> = r
                .river
                .iter()
                .filter_map(|t| mjai_string_of(*t).ok())
                .map(|p| norm_pai(&p).to_string())
                .collect();
            if river_norms.contains(tile_norm) {
                continue;
            }
            let mut furiten = false;
            for kind in &river_norms {
                if shape_win_with(r.hand, kind, r.open) {
                    furiten = true;
                    break;
                }
            }
            if !furiten {
                return true;
            }
            continue;
        }
        if chankan {
            continue;
        }
        let mut counts: HashMap<String, usize> = HashMap::new();
        for id in r.hand {
            if let Ok(pai) = mjai_string_of(*id) {
                *counts.entry(norm_pai(&pai).to_string()).or_insert(0) += 1;
            }
        }
        if counts.get(tile_norm).copied().unwrap_or(0) >= 2 {
            return true;
        }
        if r.seat == (discarder + 1) % 4 && chi_offered(r.hand, tile_norm) {
            return true;
        }
    }
    false
}

/// Kamicha chi availability for one responder hand (normalized counting).
/// Mirrors `_chi_offered` exactly (patterns per called value, aka folds).
pub fn chi_offered(hand: &[u8], tile_norm: &str) -> bool {
    if tile_norm.len() != 2 {
        return false;
    }
    let bytes = tile_norm.as_bytes();
    if !bytes[0].is_ascii_digit() || !"mps".contains(bytes[1] as char) {
        return false;
    }
    let value = (bytes[0] - b'0') as i64;
    let suit = bytes[1] as char;
    if !(1..=9).contains(&value) {
        return false;
    }
    let mut counts: HashMap<String, usize> = HashMap::new();
    for id in hand {
        if let Ok(pai) = mjai_string_of(*id) {
            *counts.entry(norm_pai(&pai).to_string()).or_insert(0) += 1;
        }
    }
    let mut patterns: Vec<(i64, i64)> = Vec::new();
    if value >= 3 {
        patterns.push((value - 2, value - 1));
    }
    if (2..=8).contains(&value) {
        patterns.push((value - 1, value + 1));
    }
    if value <= 7 {
        patterns.push((value + 1, value + 2));
    }
    patterns.into_iter().any(|(low, high)| {
        counts.get(&format!("{low}{suit}")).copied().unwrap_or(0) > 0
            && counts.get(&format!("{high}{suit}")).copied().unwrap_or(0) > 0
    })
}

/// Shape win with one extra tile of `tile_norm` (pool-first copy),
/// mirroring `_responder_eval`'s shape gate.
fn shape_win_with(hand: &[u8], tile_norm: &str, open: usize) -> bool {
    let win_id = match copies_of_string(tile_norm).ok().and_then(|p| p.first().copied()) {
        Some(id) => id,
        None => return false,
    };
    let mut concealed = hand.to_vec();
    concealed.push(win_id);
    win_shape_14(&concealed, open)
}
