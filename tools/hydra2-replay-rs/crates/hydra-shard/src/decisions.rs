//! Log-order decision walker: framed MJAI events -> decision rows.
//!
//! Reference (read-only): `src/hydra2/engines/riichienv/log_replay.py`
//! row-construction order (`_walk_game` dispatch, `_do_*` handlers,
//! `_capture_row` capture-then-emit, `_peek_window_claim`,
//! `_resolve_dora` string rule, `_TRACK` take order). Order reused; code
//! written fresh.
//!
//! Engine-owned surfaces (offered legals, riichi candidates, ankan/kakan
//! availability, ron/window state) come from the frozen engine-out
//! protocol (`crate::engine`, native v1 backend pinning the drained-oracle
//! semantics rule-by-rule) — never guessed from the log. The take ledger
//! still owns ORDER (tehais seats 0..3, live draws, rinshan draws);
//! everything else (ids, seats, chosen tiles, concealed strings, dora
//! strings, history kinds, wall countdown, tracked scores) mirrors the
//! oracle rule.

use std::collections::{HashMap, HashSet};
use crate::engine::{
    EngineDesync, EngineVersion, ResponderView, SeatView, draw_offer, kyushu_first_draw, window_open,
};
use crate::parity::{ChosenAction, ReplayRow};
use crate::mjai_event::{SKIP_TYPES, TRANSPARENT_KINDS, is_claim_kind, is_row_kind};
use crate::stream::{KyokuTrack, ParsedGame};
use crate::tile::{copies_of_string, mjai_string_of, norm_pai, physical_of};

// ---------------------------------------------------------------------------
// Action table (loaded from the published artifact, never re-generated).
// ---------------------------------------------------------------------------

/// One template row of `configs/contracts/action_table_v1.json`.
#[derive(Debug, Clone)]
pub struct Template {
    pub kind: String,
    pub tile: Option<u8>,
    pub called_tile: Option<u8>,
    pub consumed: Vec<u8>,
    pub source_offset: Option<i8>,
    pub declares_riichi: bool,
    pub meld_ref_required: bool,
}

/// Published v1 action-table envelope identity (`ACTION_TABLE_ARTIFACT_TYPE`).
pub const ACTION_TABLE_ARTIFACT_TYPE: &str = "hydra2.action_table";
/// Supported table schema (`ACTION_TABLE_SCHEMA_VERSION`, v1 only).
pub const ACTION_TABLE_SCHEMA_VERSION: &str = "1.0.0";
/// Exact template entry keys (`_TEMPLATE_JSON_FIELDS`).
const TEMPLATE_JSON_FIELDS: [&str; 7] = [
    "called_tile",
    "consumed_tiles",
    "declares_riichi",
    "kind",
    "meld_ref_required",
    "source_offset",
    "tile",
];
/// Frozen kind -> ordinal mapping (SPEC 6.1, `ACTION_KIND_ORDINALS`).
const ACTION_KIND_ORDINALS: [(&str, u32); 13] = [
    ("pass", 0),
    ("discard", 1),
    ("tsumogiri", 2),
    ("riichi_discard", 3),
    ("chi", 4),
    ("pon", 5),
    ("daiminkan", 6),
    ("ankan", 7),
    ("kakan", 8),
    ("ron", 9),
    ("tsumo", 10),
    ("abort_nine_terminals", 11),
    ("accept_abortive_draw", 12),
];

/// Generation-order action table with template -> id lookup.
#[derive(Debug)]
pub struct ActionTable {
    index: HashMap<String, u32>,
    pub len: usize,
    /// Content digest (`table.digest` on the Python `_table` path): sha256
    /// over the RFC 8785 canonical bytes of the digest-free payload
    /// (`{schema_version, actions}` in file order). Binds
    /// `action_table_hash`; byte-equal to Python on the pinned artifact.
    pub digest: String,
}

fn template_key(
    kind: &str,
    tile: Option<u8>,
    called: Option<u8>,
    consumed: &[u8],
    offset: Option<i8>,
    riichi: bool,
    meldref: bool,
) -> String {
    format!(
        "{kind}|{}|{}|{}|{}|{riichi}|{meldref}",
        tile.map(|t| t.to_string()).unwrap_or_default(),
        called.map(|t| t.to_string()).unwrap_or_default(),
        consumed
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(","),
        offset.map(|o| o.to_string()).unwrap_or_default(),
    )
}

fn kind_ordinal(kind: &str) -> Option<u32> {
    ACTION_KIND_ORDINALS
        .iter()
        .find(|(name, _)| *name == kind)
        .map(|(_, ordinal)| *ordinal)
}

/// SPEC 6.3 generation order: lexicographic, `None` before integers
/// (`template_sort_key`).
fn template_sort_key(
    template: &Template,
) -> (
    u32,
    (u8, i64),
    (u8, i64),
    Vec<u8>,
    (u8, i64),
    bool,
    bool,
) {
    fn none_first(value: Option<i64>) -> (u8, i64) {
        match value {
            None => (0, 0),
            Some(v) => (1, v),
        }
    }
    (
        kind_ordinal(&template.kind).unwrap_or(u32::MAX),
        none_first(template.tile.map(|v| v as i64)),
        none_first(template.called_tile.map(|v| v as i64)),
        template.consumed.clone(),
        none_first(template.source_offset.map(|v| v as i64)),
        template.declares_riichi,
        template.meld_ref_required,
    )
}

/// Digest-free template document (`_template_to_json` field mapping).
fn template_json(template: &Template) -> serde_json::Value {
    fn opt_tile(value: Option<u8>) -> serde_json::Value {
        value.map_or(serde_json::Value::Null, |v| serde_json::Value::from(v as u64))
    }
    serde_json::json!({
        "called_tile": opt_tile(template.called_tile),
        "consumed_tiles": template.consumed,
        "declares_riichi": template.declares_riichi,
        "kind": template.kind,
        "meld_ref_required": template.meld_ref_required,
        "source_offset": template.source_offset.map(|v| v as i64).map_or(serde_json::Value::Null, serde_json::Value::from),
        "tile": opt_tile(template.tile),
    })
}

/// Content digest over the digest-free payload in the given template order
/// (`compute_table_digest`).
fn table_content_digest(schema_version: &str, templates: &[Template]) -> String {
    let doc = serde_json::json!({
        "schema_version": schema_version,
        "actions": templates.iter().map(template_json).collect::<Vec<_>>(),
    });
    crate::parity::digest_text(&crate::parity::canonical_json_bytes(&doc))
}

/// Strict template parse (`_template_from_json`): exact key set, known
/// kind, null-or-ranged tile ids (`bool` rejected), ranged offsets, bool
/// flags. Anything else fails closed.
fn parse_template(entry: &serde_json::Value, id: usize) -> Result<Template, String> {
    let obj = entry
        .as_object()
        .ok_or_else(|| format!("action table row {id}: template must be a JSON object"))?;
    if obj.len() != TEMPLATE_JSON_FIELDS.len()
        || !TEMPLATE_JSON_FIELDS.iter().all(|key| obj.contains_key(*key))
    {
        return Err(format!(
            "action table row {id}: template entries must hold exactly {TEMPLATE_JSON_FIELDS:?}"
        ));
    }
    let kind = obj
        .get("kind")
        .and_then(|v| v.as_str())
        .filter(|kind| kind_ordinal(kind).is_some())
        .ok_or_else(|| {
            format!("action table row {id}: unknown template kind {:?}", obj.get("kind"))
        })?;
    let tile = opt_tile_value(obj.get("tile"), id, "tile")?;
    let called_tile = opt_tile_value(obj.get("called_tile"), id, "called_tile")?;
    let consumed_raw = obj
        .get("consumed_tiles")
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("action table row {id}: consumed_tiles must be an array of ints"))?;
    let mut consumed = Vec::with_capacity(consumed_raw.len());
    for value in consumed_raw {
        consumed.push(
            value
                .as_u64()
                .filter(|v| *v <= 135)
                .map(|v| v as u8)
                .ok_or_else(|| format!("action table row {id}: bad consumed tile"))?,
        );
    }
    let source_offset = match obj.get("source_offset") {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::Number(n)) => match n.as_i64() {
            Some(v) if (-1..=2).contains(&v) => Some(v as i8),
            _ => return Err(format!("action table row {id}: source_offset invalid: {n}")),
        },
        Some(other) => {
            return Err(format!("action table row {id}: source_offset invalid: {other}"));
        }
    };
    let declares_riichi = obj
        .get("declares_riichi")
        .and_then(|v| v.as_bool())
        .ok_or_else(|| format!("action table row {id}: declares_riichi must be a bool"))?;
    let meld_ref_required = obj
        .get("meld_ref_required")
        .and_then(|v| v.as_bool())
        .ok_or_else(|| format!("action table row {id}: meld_ref_required must be a bool"))?;
    Ok(Template {
        kind: kind.to_string(),
        tile,
        called_tile,
        consumed,
        source_offset,
        declares_riichi,
        meld_ref_required,
    })
}

fn opt_tile_value(
    value: Option<&serde_json::Value>,
    id: usize,
    name: &str,
) -> Result<Option<u8>, String> {
    match value.unwrap_or(&serde_json::Value::Null) {
        serde_json::Value::Null => Ok(None),
        serde_json::Value::Number(n) => n
            .as_u64()
            .filter(|v| *v <= 135)
            .map(|v| Some(v as u8))
            .ok_or_else(|| format!("action table row {id}: {name} must be null or int 0..135")),
        other => Err(format!("action table row {id}: {name} must be null or int, got {other}")),
    }
}

#[cfg(test)]
fn opt_u8(value: &serde_json::Value) -> Result<Option<u8>, String> {
    match value {
        serde_json::Value::Null => Ok(None),
        serde_json::Value::Number(n) => n
            .as_u64()
            .filter(|v| *v <= 135)
            .map(|v| Some(v as u8))
            .ok_or_else(|| format!("action table tile out of range: {n}")),
        other => Err(format!("action table tile must be int-or-null: {other}")),
    }
}

impl ActionTable {
    /// Load + fully verify the published v1 artifact (6792 rows).
    ///
    /// Mirrors `load_action_table` / `_table_from_document`: envelope
    /// identity, exact payload keys, strict template checks, declared
    /// digest match, and the generation-order rebuild check (a row swap
    /// with an updated digest still fails closed). The verified CONTENT
    /// digest is stored and binds `action_table_hash`.
    pub fn load_json(text: &str) -> Result<Self, String> {
        let doc: serde_json::Value =
            serde_json::from_str(text).map_err(|e| format!("action table parse: {e}"))?;
        let root = doc
            .as_object()
            .ok_or_else(|| "action table: artifact must be a JSON object".to_string())?;
        if root.get("artifact_type").and_then(|v| v.as_str()) != Some(ACTION_TABLE_ARTIFACT_TYPE) {
            return Err(format!(
                "action table: artifact_type must be {ACTION_TABLE_ARTIFACT_TYPE:?}, got {:?}",
                root.get("artifact_type")
            ));
        }
        if root.get("compatibility").and_then(|v| v.as_str()) != Some("exact") {
            return Err(format!(
                "action table: unsupported compatibility {:?}",
                root.get("compatibility")
            ));
        }
        let version = root
            .get("schema_version")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "action table: schema_version must be a string".to_string())?;
        if version.split('.').next() != Some("1") {
            return Err(format!("action table: unknown major schema version {version:?}"));
        }
        if version != ACTION_TABLE_SCHEMA_VERSION {
            return Err(format!(
                "action table: schema_version {version:?} newer than supported {ACTION_TABLE_SCHEMA_VERSION:?}"
            ));
        }
        let payload = root
            .get("payload")
            .and_then(|v| v.as_object())
            .ok_or_else(|| "action table: payload must be an object".to_string())?;
        if payload.len() != 3
            || !["schema_version", "actions", "digest"]
                .iter()
                .all(|key| payload.contains_key(*key))
        {
            return Err(
                "action table: payload must hold exactly schema_version/actions/digest".to_string(),
            );
        }
        if payload.get("schema_version").and_then(|v| v.as_str()) != Some(version) {
            return Err(format!(
                "action table: payload schema_version {:?} != envelope {version:?}",
                payload.get("schema_version")
            ));
        }
        let declared = payload
            .get("digest")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "action table: digest must be a string".to_string())?
            .to_string();
        let actions = payload
            .get("actions")
            .and_then(|v| v.as_array())
            .filter(|arr| !arr.is_empty())
            .ok_or_else(|| "action table: actions must be a non-empty array".to_string())?;
        let mut templates = Vec::with_capacity(actions.len());
        for (id, entry) in actions.iter().enumerate() {
            templates.push(parse_template(entry, id)?);
        }
        let recomputed = table_content_digest(version, &templates);
        if recomputed != declared {
            return Err(format!(
                "action table: declared digest {declared:?} != recomputed {recomputed:?}"
            ));
        }
        // Rebuild check (`build_action_table` sorts then digests): file
        // order must already be generation order.
        let mut ordered = templates.clone();
        ordered.sort_by(|a, b| template_sort_key(a).cmp(&template_sort_key(b)));
        if table_content_digest(version, &ordered) != declared {
            return Err(
                "action table: templates are not in generation order (rebuilt digest differs)"
                    .to_string(),
            );
        }
        let mut table = ActionTable {
            index: HashMap::new(),
            len: 0,
            digest: declared,
        };
        for (id, template) in templates.iter().enumerate() {
            table.index.insert(
                template_key(
                    &template.kind,
                    template.tile,
                    template.called_tile,
                    &template.consumed,
                    template.source_offset,
                    template.declares_riichi,
                    template.meld_ref_required,
                ),
                id as u32,
            );
        }
        if table.index.len() != templates.len() {
            return Err("action table: duplicate action templates are rejected".to_string());
        }
        table.len = table.index.len();
        Ok(table)
    }

    /// Test-only lenient loader: index-only over `payload.actions`, no
    /// envelope or digest checks. Unit-test scaffolding for synthetic
    /// tables; every production path MUST use [`load_json`](Self::load_json).
    #[cfg(test)]
    pub(crate) fn load_unverified_json(text: &str) -> Result<Self, String> {
        let doc: serde_json::Value =
            serde_json::from_str(text).map_err(|e| format!("action table parse: {e}"))?;
        let actions = doc
            .pointer("/payload/actions")
            .and_then(|v| v.as_array())
            .ok_or_else(|| "action table: missing payload.actions".to_string())?;
        let mut index = HashMap::new();
        for (id, entry) in actions.iter().enumerate() {
            let kind = entry
                .get("kind")
                .and_then(|v| v.as_str())
                .ok_or_else(|| format!("action table row {id}: missing kind"))?;
            let tile = opt_u8(entry.get("tile").unwrap_or(&serde_json::Value::Null))?;
            let called = opt_u8(entry.get("called_tile").unwrap_or(&serde_json::Value::Null))?;
            let consumed: Vec<u8> = entry
                .get("consumed_tiles")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .map(|v| {
                            v.as_u64()
                                .filter(|x| *x <= 135)
                                .map(|x| x as u8)
                                .ok_or_else(|| format!("action table row {id}: bad consumed tile"))
                        })
                        .collect::<Result<Vec<u8>, String>>()
                })
                .transpose()?
                .unwrap_or_default();
            let offset = match entry.get("source_offset").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Null => None,
                serde_json::Value::Number(n) => n
                    .as_i64()
                    .filter(|v| (-1..=2).contains(v))
                    .map(|v| v as i8),
                other => return Err(format!("action table row {id}: bad offset {other}")),
            };
            if offset.is_none() && !entry.get("source_offset").unwrap_or(&serde_json::Value::Null).is_null() {
                return Err(format!("action table row {id}: bad offset"));
            }
            let riichi = entry
                .get("declares_riichi")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let meldref = entry
                .get("meld_ref_required")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            index.insert(
                template_key(kind, tile, called, &consumed, offset, riichi, meldref),
                id as u32,
            );
        }
        let len = index.len();
        Ok(ActionTable {
            index,
            len,
            digest: String::new(),
        })
    }

    pub fn lookup(
        &self,
        kind: &str,
        tile: Option<u8>,
        called: Option<u8>,
        consumed: &[u8],
        offset: Option<i8>,
        riichi: bool,
        meldref: bool,
    ) -> Option<u32> {
        self.index
            .get(&template_key(kind, tile, called, consumed, offset, riichi, meldref))
            .copied()
    }
}

/// Relative seat offset: (source - actor) mod 4 with 3 folded to -1.
pub fn source_offset(source: u8, actor: u8) -> i8 {
    match (source + 4 - actor) % 4 {
        3 => -1,
        delta => delta as i8,
    }
}

// ---------------------------------------------------------------------------
// Walker.
// ---------------------------------------------------------------------------

/// Whole-game quarantine with a stable reason code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalkReject {
    pub game_id: String,
    pub code: String,
    pub detail: String,
}

impl WalkReject {
    /// Classified rejection: `code` is the closed taxonomy
    /// (`bare-dora | double-ron | unknown-event | tile-conservation |
    /// turn-order | claim-no-offer | draw-past-wall | kyushu-ambiguous |
    /// unmapped-ryukyoku-reason | engine-desync`; `framing` /
    /// `wall-bearing` live on `GameReject`, `action-id-unresolved` is
    /// row-level). `step` keeps the oracle's event-step vocabulary in
    /// `detail` so downstream `_quarantine_class` normalization keeps
    /// working.
    pub(crate) fn new(game_id: &str, kyoku: i64, code: &str, step: &str, why: &str) -> Self {
        WalkReject {
            game_id: game_id.to_string(),
            code: code.to_string(),
            detail: format!("sim replay desync game {game_id:?} kyoku {kyoku} {step}: {why}"),
        }
    }
}

fn is_end(kind: &str) -> bool {
    matches!(kind, "end_game" | "endGame" | "game_end" | "end")
}

fn is_start(kind: &str) -> bool {
    matches!(
        kind,
        "start_game" | "startGame" | "game_start" | "start"
    )
}

fn in_skip(kind: &str) -> bool {
    SKIP_TYPES.contains(&kind)
}

fn is_transparent(kind: &str) -> bool {
    TRANSPARENT_KINDS.contains(&kind)
}

struct Walker<'a> {
    game: &'a ParsedGame,
    table: &'a ActionTable,
    seq: u32,
    round_idx: i64,
    hand_index: i64,
    kyoku_ordinal: i64,
    track: Option<KyokuTrack>,
    dealer: u8,
    histories: [Vec<String>; 4],
    last_kind: Option<String>,
    last_discard: Option<(u8, u8)>,
    opened_by_discard: bool,
    decided: bool,
    terminal: bool,
    rows: Vec<ReplayRow>,
    /// Tracked scores per seat (kyoku headers, -1000 per riichi
    /// declaration, plus hora/ryukyoku deltas). Feeds the engine's riichi
    /// availability gate (score >= 1000), mirroring `tracked_scores`.
    scores: [i32; 4],
    /// Selected backend: V1 (pinned drained-oracle, bug-compatible) or V2
    /// (single-pass live-engine semantics). V1 is the frozen default.
    version: crate::engine::EngineVersion,
    /// V2 one-shot immediate filter: (claimer, just-claimed called take,
    /// just-claimed meld takes, chi takes for the neighborhood gate, empty
    /// unless chi). Set on chi/pon claims, cleared on the next discard or
    /// kyoku start.
    kuikae_gate: Option<(u8, u8, Vec<u8>, Vec<u8>)>,
    /// Whether each seat has made a decision this kyoku (any emitted row).
    /// Feeds V2 first-draw detection for the kyushu offer.
    acted: [bool; 4],
    /// Claim/kan count this kyoku (any seat). Feeds V2 kyushu (no calls).
    claims: u32,
    /// S7 walled binding: `Some` real wall digest when the framed game
    /// carries a 136-tile wall (`replay-{game_id}` schedule), `None` on the
    /// wall-less path (SIM mark). Minted once at walk start via
    /// `wall_schedule_digest`; every emitted row clones it so the
    /// derivation retires the SIM mark for walled rows only.
    wall_digest: Option<String>,
}

/// S7 port of the `ReplayExpander` mode machine
/// (`replay_expand.ReplayExpander._step_decision`: `expected-actor` /
/// `window-pending` / `draw-mode` / `terminal` against engine answers).
///
/// The engine owns legality; the log owns order. `mode` is derived from
/// engine answers on ledger state (window responders via `window_open` /
/// per-seat offer predicates, draw expectation via the tracked drawer),
/// never from ledger guesses alone:
/// - `Terminal`: the kyoku was decided (hora/ryukyoku) — no decision is
///   pending, so any further row event fails closed (post-terminal rows);
/// - `Window { pending }`: a discard offer is live with a non-empty sorted
///   responder set — the next significant event must be a matching claim on
///   that discard (earlier pending seats pass mechanically, in order) or the
///   window passes with no row;
/// - `Draw { expected }`: the tracked drawer owns the next draw decision —
///   claims outside any window fail closed here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WalkerMode {
    /// No live kyoku yet (before the first `start_kyoku`).
    Idle,
    /// Kyoku decided: no decision pending until the next `start_kyoku`.
    Terminal,
    /// Claim window live on `discarder`'s `tile` with sorted `pending`.
    Window { discarder: u8, tile: u8, pending: Vec<u8> },
    /// Draw decision owned by `expected`.
    Draw { expected: u8 },
}

impl<'a> Walker<'a> {
    fn fail(&self, code: &str, step: &str, why: &str) -> WalkReject {
        WalkReject::new(&self.game.game_id, self.kyoku_ordinal, code, step, why)
    }

    fn track(&self, step: &str) -> Result<&KyokuTrack, WalkReject> {
        self.track
            .as_ref()
            .ok_or_else(|| self.fail("turn-order", step, "no live kyoku"))
    }

    fn track_mut(&mut self, step: &str) -> Result<&mut KyokuTrack, WalkReject> {
        let ordinal = self.kyoku_ordinal;
        self.track.as_mut().ok_or_else(|| {
            WalkReject::new(&self.game.game_id, ordinal, "turn-order", step, "no live kyoku")
        })
    }

    fn push_public(&mut self, kind: &str) {
        for seat in 0..4 {
            self.histories[seat].push(kind.to_string());
        }
        self.last_kind = Some(kind.to_string());
    }

    fn live_wall(&self) -> u32 {
        let draws = self.track.as_ref().map(|t| t.draws).unwrap_or(0);
        crate::stream::LIVE_WALL_BASE.saturating_sub(draws)
    }

    /// Reported (string-canonical, pool-first) concealed multiset for `seat`.
    fn reported_hand(&self, seat: u8) -> Result<Vec<u8>, WalkReject> {
        let track = self.track("row")?;
        // Collapse every true copy to its string-canonical id, then expand
        // per-string pools in hand order (mirrors `_distinct_copies`).
        let mut per_string: HashMap<String, usize> = HashMap::new();
        let mut out = Vec::with_capacity(track.hands[seat as usize].len());
        // First pass: count occurrences per rendered string in hand order.
        let rendered: Vec<String> = track.hands[seat as usize]
            .iter()
            .map(|t| mjai_string_of(*t).map_err(|e| self.fail("tile-conservation", "row", &e.to_string())))
            .collect::<Result<_, _>>()?;
        // Assign pool copies per occurrence (order-independent multiset).
        let mut pools: HashMap<String, Vec<u8>> = HashMap::new();
        for pai in &rendered {
            let entry = pools.entry(pai.clone()).or_insert_with(|| {
                copies_of_string(pai).unwrap_or_default()
            });
            let used = per_string.get(pai).copied().unwrap_or(0);
            if used < entry.len() {
                out.push(entry[used]);
            } else {
                // Overused strings keep the verbatim collapsed id.
                out.push(copies_of_string(pai).unwrap_or_default().first().copied().unwrap_or(0));
            }
            per_string.insert(pai.clone(), used + 1);
        }
        out.sort_unstable();
        Ok(out)
    }

    fn reported_drawn(&self, seat: u8) -> Result<Option<u8>, WalkReject> {
        let track = self.track("row")?;
        match track.drawn[seat as usize] {
            None => Ok(None),
            Some(d) => {
                let pai = mjai_string_of(d).map_err(|e| self.fail("tile-conservation", "row", &e.to_string()))?;
                Ok(copies_of_string(&pai)
                    .map_err(|e| self.fail("tile-conservation", "row", &e.to_string()))?
                    .first()
                    .copied())
            }
        }
    }

    fn concealed_for_row(&self, seat: u8) -> Result<Vec<u8>, WalkReject> {
        let mut hand = self.reported_hand(seat)?;
        // Mirror `_concealed_for_build` on collapsed takes: the drained
        // step hand is pool-prefix, so the collapsed drawn copy is always
        // present exactly when the seat drew this turn. Stale draws persist
        // in the display but never subtract from the concealed hand.
        let live = self.track("row")?.drawn_live[seat as usize].is_some();
        if live {
            if let Some(drawn) = self.reported_drawn(seat)? {
                if let Some(pos) = hand.iter().position(|t| *t == drawn) {
                    hand.remove(pos);
                }
            }
        }
        Ok(hand)
    }
    /// keeping the oracle's `game + kyoku + step` detail vocabulary.
    fn desync(&self, err: EngineDesync) -> WalkReject {
        WalkReject::new(&err.game_id, err.kyoku, "engine-desync", &err.step, &err.why)
    }

    /// Seat-filtered engine view for a draw decision: the collapsed hand
    /// takes (pool-first per string occurrence, exactly like the drained
    /// `_distinct_copies` expansion) plus the collapsed live draw and the
    /// true drawn take for riichi-forced answers.
    fn seat_view(&self, seat: u8, step: &'static str) -> Result<SeatView<'_>, WalkReject> {
        let hand = self.reported_hand(seat)?;
        let track = self.track(step)?;
        let live_take = track.drawn_live[seat as usize];
        let live = live_take.is_some();
        // Collapsed live draw (pool-first of the drawn string); the true
        // take rides alongside for riichi-forced answers.
        let drawn = if live { self.reported_drawn(seat)? } else { None };
        SeatView::new(
            &self.game.game_id,
            self.kyoku_ordinal,
            step,
            seat,
            hand,
            drawn,
            live_take,
            &track.melds[seat as usize],
            track.riichi_declared[seat as usize],
            self.scores[seat as usize],
            self.live_wall(),
        )
        .map_err(|err| self.desync(err))
    }
    fn draw_mask_ids(
        &self,
        seat: u8,
        offer: &crate::engine::DrawOffer,
        chosen: &ChosenAction,
        log_win: bool,
    ) -> Vec<u32> {
        let mut mask = std::collections::HashSet::new();
        for tile in &offer.discards {
            if let Some(id) = self.table.lookup("discard", Some(*tile), None, &[], None, false, false)
            {
                mask.insert(id);
            }
        }
        if let Some(tile) = offer.tsumogiri {
            if let Some(id) =
                self.table.lookup("tsumogiri", Some(tile), None, &[], None, false, false)
            {
                mask.insert(id);
            }
        }
        for tile in &offer.riichi {
            if let Some(id) =
                self.table.lookup("riichi_discard", Some(*tile), None, &[], None, true, false)
            {
                mask.insert(id);
            }
        }
        for quad in &offer.ankan {
            if let Some(id) = self.table.lookup("ankan", None, None, quad, None, false, false) {
                mask.insert(id);
            }
        }
        for tile in &offer.kakan {
            if let Some(id) = self.table.lookup("kakan", Some(*tile), None, &[], None, false, true)
            {
                mask.insert(id);
            }
        }
        if log_win {
            if let ChosenAction::Tsumo { tile } = chosen {
                if let Some(id) = self.table.lookup("tsumo", Some(*tile), None, &[], None, false, false)
                {
                    mask.insert(id);
                }
            }
        }
        if let Some(id) = chosen.lookup(self.table, seat) {
            mask.insert(id);
        }
        let mut out: Vec<u32> = mask.into_iter().collect();
        out.sort_unstable();
        out
    }

    /// Claim-response mask: exactly `{pass, chosen}`. The drained engine
    /// offers no chi enumeration beyond the stashed (logged) claim, and
    /// ron never rides a non-ron claim row (both corpus-proven).
    fn claim_mask_ids(&self, seat: u8, chosen: &ChosenAction) -> Vec<u32> {
        let mut mask = std::collections::HashSet::new();
        if let Some(id) = self.table.lookup("pass", None, None, &[], None, false, false) {
            mask.insert(id);
        }
        if let Some(id) = chosen.lookup(self.table, seat) {
            mask.insert(id);
        }
        let mut out: Vec<u32> = mask.into_iter().collect();
        out.sort_unstable();
        out
    }

    /// Ron-response mask: exactly `{pass, ron}` (pinned on every ron row).
    fn ron_mask_ids(&self, seat: u8, tile: u8, discarder: u8) -> Vec<u32> {
        let mut mask = std::collections::HashSet::new();
        if let Some(id) = self.table.lookup("pass", None, None, &[], None, false, false) {
            mask.insert(id);
        }
        let delta = (discarder + 4 - seat) % 4;
        let offset = if delta == 3 { -1 } else { delta as i8 };
        if let Some(id) = self.table.lookup("ron", Some(tile), None, &[], Some(offset), false, false)
        {
            mask.insert(id);
        }
        let mut out: Vec<u32> = mask.into_iter().collect();
        out.sort_unstable();
        out
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "row capture takes explicit context args; grouping churns cold walker"
    )]
    fn capture_row(
        &mut self,
        seat: u8,
        phase: &str,
        turn_actor: u8,
        chosen: ChosenAction,
        mask: &[u32],
    ) -> Result<(), WalkReject> {
        let seq = self.seq;
        self.seq += 1;
        self.acted[seat as usize] = true;
        let decision_id = format!("{}:d{seq:04}", self.game.game_id);
        let round_id = format!("{}:h{:02}", self.game.game_id, self.round_idx.max(0));
        let concealed = self.concealed_for_row(seat)?;
        let concealed_hand: Vec<String> = concealed
            .iter()
            .map(|t| mjai_string_of(*t).map_err(|e| self.fail("tile-conservation", "row", &e.to_string())))
            .collect::<Result<_, _>>()?;
        let drawn_id = self.reported_drawn(seat)?;
        let drawn_tile = drawn_id
            .map(|d| mjai_string_of(d).map_err(|e| self.fail("tile-conservation", "row", &e.to_string())))
            .transpose()?;
        let track = self.track("row")?;
        let dora_indicators: Vec<Option<String>> = (0..5)
            .map(|i| track.dora.get(i).cloned())
            .collect();
        let (tile, called, consumed) = match &chosen {
            ChosenAction::Discard { tile }
            | ChosenAction::Tsumogiri { tile }
            | ChosenAction::RiichiDiscard { tile } => (Some(*tile), None, vec![]),
            ChosenAction::Chi { called, consumed } => (None, Some(*called), consumed.clone()),
            ChosenAction::Pon { called, consumed, .. } => (None, Some(*called), consumed.clone()),
            ChosenAction::Daiminkan { called, consumed, .. } => {
                (None, Some(*called), consumed.clone())
            }
            ChosenAction::Ankan { consumed } => (None, None, consumed.clone()),
            ChosenAction::Kakan { tile } => (Some(*tile), None, vec![]),
            ChosenAction::Ron { tile, .. } => (Some(*tile), None, vec![]),
            ChosenAction::Tsumo { tile } => (Some(*tile), None, vec![]),
            ChosenAction::Pass => (None, None, vec![]),
        };
        let kind = chosen.kind_str().to_string();
        let declares_riichi = matches!(chosen, ChosenAction::RiichiDiscard { .. });
        let offset = match &chosen {
            ChosenAction::Chi { .. } => Some(-1),
            ChosenAction::Pon { source, .. }
            | ChosenAction::Daiminkan { source, .. }
            | ChosenAction::Ron { source, .. } => Some(source_offset(*source, seat)),
            _ => None,
        };
        let meldref = matches!(chosen, ChosenAction::Kakan { .. });
        let chosen_action_id =
            self.table
                .lookup(&kind, tile, called, &consumed, offset, declares_riichi, meldref);
        let chosen_unresolved = if chosen_action_id.is_none() {
            Some(format!("action-id-unresolved:{kind}"))
        } else {
            None
        };
        let mut legal_mask: Vec<u32> = mask.to_vec();
        legal_mask.sort_unstable();
        legal_mask.dedup();
        self.rows.push(ReplayRow {
            game_id: self.game.game_id.clone(),
            round_id,
            decision_id,
            seat,
            phase: phase.to_string(),
            turn_actor,
            chosen,
            chosen_action_id,
            chosen_unresolved,
            legal_mask,
            history_kinds: self.histories[seat as usize].clone(),
            concealed_hand,
            drawn_tile,
            dora_indicators,
            wall_remaining: self.live_wall(),
            wall_digest: self.wall_digest.clone(),
        });
        Ok(())
    }

    // -- engine window evaluation ------------------------------------------

    /// Responder snapshots for one window query (ledger-derived inputs to
    /// the engine's window rule; only booleans leave the engine, so the
    /// seat filter holds by construction).
    fn responders(&self, step: &str) -> Result<Vec<ResponderView<'_>>, WalkReject> {
        let track = self.track(step)?;
        let mut out = Vec::with_capacity(4);
        for seat in 0..4u8 {
            out.push(ResponderView {
                seat,
                riichi: track.riichi_declared[seat as usize],
                hand: &track.hands[seat as usize],
                river: &track.rivers[seat as usize],
                open: track.melds[seat as usize].len(),
            });
        }
        Ok(out)
    }

    /// Next window claim on `discarder`'s tile, if the log shows one
    /// (mirrors `_peek_window_claim`: scans past transparent events; a
    /// draw decision, another offer, a terminal, or any boundary ends the
    /// window with no claim).
    fn peek_window_claim(&self, from_idx: usize, discarder: u8) -> Option<&crate::mjai_event::MjaiEvent> {
        let mut idx = from_idx;
        while idx < self.game.events.len() {
            let event = &self.game.events[idx];
            let kind = event.type_.as_str();
            if is_transparent(kind) {
                idx += 1;
                continue;
            }
            if is_claim_kind(kind) {
                if event.target_seat().ok() == Some(discarder) {
                    return Some(event);
                }
                return None;
            }
            if kind == "hora" && !event.tsumo {
                if event.actor == event.target {
                    return None;
                }
                if event.target_seat().ok() == Some(discarder) {
                    return Some(event);
                }
                return None;
            }
            return None;
        }
        None
    }

    /// Whether the log takes a ron on `discarder`'s tile at the window
    /// opening at `from_idx`. A logged ron proves the engine's window was
    /// open (the engine must have offered it): sound window-open fact for
    /// the responder-ron cases the responder rules cannot see.
    fn peek_window_ron(&self, from_idx: usize, discarder: u8) -> bool {
        match self.peek_window_claim(from_idx, discarder) {
            Some(event) if event.type_.as_str() == "hora" => true,
            _ => false,
        }
    }

    /// Emit the public `call_window` envelope when the engine opens a
    /// window after a discard (responder rules) or the log proves one (a
    /// ron taken on this discard). Kan windows never emit envelopes: the
    /// adapter grammar routes kakan straight to ron, and
    /// `opened_by_discard` is false there on both sides.
    fn open_window(&mut self, discarder: u8, tile: u8, chankan: bool, from_idx: usize) {
        let tile_norm = mjai_string_of(tile).map(|p| norm_pai(&p).to_string()).unwrap_or_default();
        let engine_open = self
            .responders("window")
            .map(|views| window_open(discarder, &tile_norm, &views, chankan))
            .unwrap_or(false);
        let log_ron = self.peek_window_ron(from_idx, discarder);
        if (engine_open || log_ron) && self.opened_by_discard && self.last_kind.as_deref() == Some("discard")
        {
            self.push_public("call_window");
        }
    }

    // -- S7 mode machine (ReplayExpander port) ------------------------------
    //
    // Reference: `replay_expand.ReplayExpander._step_decision` /
    // `_step_window` / `_step_draw` plus the adapter's
    // `_expected_actor_or_none` / `_detect_decision` (draw vs window vs
    // terminal). The engine owns legality (responder predicates, drawer
    // expectation); the log owns order (claim matching, reach-collapse
    // indices, terminal checks). Both walled and wall-less rows flow through
    // this gate so the fork stays unified.

    /// Sorted pending responders for a live discard offer.
    ///
    /// Mirrors the adapter's `WaitResponse` responder set (seats whose engine
    /// observation would list a legal response): pon counts, kamicha chi
    /// availability, or a furiten-clean riichi shape — plus the log-ron fact
    /// (a logged ron proves the engine offered it, sound since the engine
    /// must have). Kan (`chankan`) windows suppress pon/chi exactly like the
    /// oracle, which additionally never emits post-kakan envelopes.
    pub(crate) fn window_pending(&self, discarder: u8, tile: u8, chankan: bool) -> Vec<u8> {
        let Ok(track) = self.track("window") else {
            return Vec::new();
        };
        let tile_norm = mjai_string_of(tile).map(|p| norm_pai(&p).to_string()).unwrap_or_default();
        let mut pending = Vec::new();
        for seat in 0..4u8 {
            if seat == discarder {
                continue;
            }
            let hand = &track.hands[seat as usize];
            let river = &track.rivers[seat as usize];
            let open = track.melds[seat as usize].len();
            let riichi = track.riichi_declared[seat as usize];
            // Pon: two held copies of the called type (aka folds).
            let mut pon = false;
            let mut chi = false;
            if !chankan {
                let mut held = 0u8;
                for t in hand.iter() {
                    if let Ok(p) = mjai_string_of(*t) {
                        if norm_pai(&p) == tile_norm {
                            held += 1;
                        }
                    }
                }
                if held >= 2 {
                    pon = true;
                }
                // Chi: kamicha only, via the engine's chi predicate.
                if seat == (discarder + 1) % 4 && !tile_norm.is_empty() {
                    // Honors never chi; the predicate folds aka.
                    chi = crate::engine::chi_offered(hand, &tile_norm);
                }
            }
            // Ron shape: the discard completes a winning shape for a
            // closed-or-open hand. Responder-ron by non-riichi seats is
            // invisible to the responder rules, so the mode keeps the
            // responder in pending only when the shape holds AND (riichi
            // declared OR the log takes the ron here). The log fact is
            // resolved by the caller via `peek_window_ron`; here we keep the
            // shape gate so the set stays engine-derived.
            let mut completed = hand.clone();
            completed.push(tile);
            let shape = crate::engine::win_shape_14(&completed, open);
            // Furiten-clean riichi shape is the visible ron offer; the log
            // fact covers non-riichi ron separately at the window gate.
            let ron_offered = shape && riichi && !river.iter().any(|r| {
                mjai_string_of(*r).map(|p| norm_pai(&p).to_string()).unwrap_or_default() == tile_norm
            });
            if pon || chi || ron_offered {
                pending.push(seat);
            }
        }
        // Log-ron soundness: a logged ron on this discard proves the window
        // was open for its winner even when the responder rules cannot see
        // non-riichi ron. The caller ORs this at the gate (`open_window`
        // already does); `pending` itself stays engine-derived so the mode
        // never guesses from the log alone.
        pending.sort_unstable();
        pending
    }

    /// Current mode against engine answers (never ledger guesses alone).
    ///
    /// - `Terminal` once the kyoku is decided (post-terminal rows fail
    ///   closed) or before the first kyoku (`Idle` maps to a turn-order
    ///   failure at use sites);
    /// - `Window` while a discard offer is live with non-empty pending;
    /// - `Draw` otherwise, owned by the tracked drawer.
    pub(crate) fn walker_mode(&self) -> WalkerMode {
        if self.track.is_none() || self.kyoku_ordinal < 0 {
            return WalkerMode::Idle;
        }
        if self.decided {
            return WalkerMode::Terminal;
        }
        if let Some((discarder, tile)) = self.last_discard {
            if self.opened_by_discard {
                let pending = self.window_pending(discarder, tile, false);
                // Kan windows (opened_by_discard false) never surface here;
                // chankan ron still resolves via the ron path's shape gate.
                if !pending.is_empty() {
                    return WalkerMode::Window {
                        discarder,
                        tile,
                        pending,
                    };
                }
            }
        }
        let expected = self
            .track
            .as_ref()
            .and_then(|t| t.drawer.or(t.exp_drawer).or(Some(self.dealer)))
            .unwrap_or(self.dealer);
        WalkerMode::Draw { expected }
    }

    /// Expected decision actor, if any (`None` on terminal/idle — mirrors
    /// `sim._expected_actor_or_none`: terminal => None, window => min
    /// pending, draw => drawer).
    #[allow(
        dead_code,
        reason = "mirrors sim expected-actor for walled gate; wall-less keeps frozen checks"
    )]
    pub(crate) fn expected_actor_or_none(&self) -> Option<u8> {
        match self.walker_mode() {
            WalkerMode::Idle | WalkerMode::Terminal => None,
            WalkerMode::Window { pending, .. } => pending.into_iter().min(),
            WalkerMode::Draw { expected } => Some(expected),
        }
    }

    /// Assert the log event at `idx` may act in the current mode (fail
    /// closed like `_step_decision`): post-terminal rows, claims outside any
    /// window, and draw-actor mismatches are named desyncs, never drops.
    /// Reach-collapse callers pass the declaration index for error detail.
    /// Walled-only gate (wall-less keeps its frozen turn-order checks so
    /// PG-S6 parity holds bit-for-bit); both paths share the vocabulary and
    /// the `walker_mode` construction.
    pub(crate) fn check_mode_for(&self, kind: &str, actor: u8, idx: usize) -> Result<(), WalkReject> {
        // Wall-less keeps frozen behavior: no extra gate.
        if self.wall_digest.is_none() {
            return Ok(());
        }
        let is_ron = kind == "hora"
            && self.game.events.get(idx).map(|e| !e.tsumo).unwrap_or(false);
        match self.walker_mode() {
            WalkerMode::Terminal => Err(self.fail(
                "turn-order",
                kind,
                &format!("log carries row-events past kyoku terminal at index {idx}"),
            )),
            WalkerMode::Window { discarder, pending, tile, .. } => {
                if crate::mjai_event::is_claim_kind(kind) {
                    // Pass placement: earlier pending seats pass mechanically
                    // in sorted order ahead of the claim (no rows, never
                    // synthesized) — the claim must sit inside the pending
                    // set, exactly like `_step_window`.
                    if !pending.contains(&actor) {
                        return Err(self.fail(
                            "claim-no-offer",
                            kind,
                            &format!("logged {kind} by seat {actor} outside the claim window [{discarder:?}->{pending:?}]"),
                        ));
                    }
                } else if is_ron {
                    // Log-ron soundness: the responder rules cannot see
                    // non-riichi ron, so a logged ron with a winning shape on
                    // the offered tile proves the window (sound: the engine
                    // must have offered it). Riichi ron must sit in pending;
                    // non-riichi ron passes on shape alone.
                    if pending.contains(&actor) {
                    } else {
                        let track = self.track("hora")?;
                        let mut completed = track.hands[actor as usize].clone();
                        completed.push(tile);
                        if !crate::engine::win_shape_14(&completed, track.melds[actor as usize].len()) {
                            return Err(self.fail(
                                "claim-no-offer",
                                kind,
                                &format!("logged {kind} by seat {actor} outside the claim window [{discarder:?}->{pending:?}]"),
                            ));
                        }
                    }
                }
                // Draws inside a window are the mechanical-pass path: the
                // window must have no logged claim (checked by the caller via
                // `peek_window_claim`); the mode itself never blocks the pass.
                Ok(())
            }
            WalkerMode::Draw { expected: _ } => {
                if crate::mjai_event::is_claim_kind(kind) {
                    return Err(self.fail(
                        "claim-no-offer",
                        kind,
                        &format!("logged {kind} outside any claim window at index {idx}"),
                    ));
                }
                if is_ron {
                    // Chankan: kakan sets `last_discard` with
                    // `opened_by_discard == false` (no envelope, grammar
                    // routes kakan straight to ron). A ron on that tile with
                    // a winning shape is the kan-response path, not a
                    // claim-outside-window failure.
                    if let Some((discarder, tile)) = self.last_discard {
                        let event = self.game.events.get(idx);
                        let target_ok = event
                            .and_then(|e| e.target_seat().ok())
                            .map(|t| t == discarder)
                            .unwrap_or(false);
                        if target_ok {
                            if let Ok(track) = self.track("hora") {
                                let mut completed = track.hands[actor as usize].clone();
                                completed.push(tile);
                                if crate::engine::win_shape_14(&completed, track.melds[actor as usize].len()) {
                                    return Ok(());
                                }
                            }
                        }
                    }
                    return Err(self.fail(
                        "claim-no-offer",
                        kind,
                        &format!("logged ron outside any claim window at index {idx}"),
                    ));
                }
                Ok(())
            }
            WalkerMode::Idle => Err(self.fail("turn-order", kind, "decision before the first start_kyoku")),
        }
    }
}

// ---------------------------------------------------------------------------
// Entry point.
// ---------------------------------------------------------------------------

/// Pre-scan: bare kan-dora markers and double ron quarantine whole games,
/// exactly like `replay_game` (fail closed before any row is built).
fn prescan(game: &ParsedGame) -> Result<(), WalkReject> {
    let mut kyoku_idx: i64 = -1;
    let mut pending_ron: Option<(u8, u8)> = None;
    for event in &game.events {
        let kind = event.type_.as_str();
        if kind == "start_kyoku" {
            kyoku_idx += 1;
            pending_ron = None;
            continue;
        }
        if kind == "dora" {
            match &event.dora_marker {
                Some(m) if !m.is_empty() => {}
                _ => {
                    return Err(WalkReject::new(
                        &game.game_id,
                        kyoku_idx,
                        "bare-dora",
                        "dora",
                        "kan-dora indicator unrecoverable: dora event without a dora_marker",
                    ));
                }
            }
            continue;
        }
        if kind == "hora" {
            let actor = seat_value(&event.actor);
            let target = seat_value(&event.target);
            let tsumo_flag = event.tsumo;
            if !tsumo_flag {
                if let (Some(a), Some(t)) = (actor, target) {
                    if a != t {
                        if let Some((prev_a, prev_t)) = pending_ron {
                            if prev_t == t && prev_a != a {
                                return Err(WalkReject::new(
                                    &game.game_id,
                                    kyoku_idx,
                                    "double-ron",
                                    "hora",
                                    "double ron on one discard is quarantined (single-winner pipeline)",
                                ));
                            }
                        }
                        pending_ron = Some((a, t));
                        continue;
                    }
                }
            }
            pending_ron = None;
            continue;
        }
        if is_transparent(kind) {
            continue;
        }
        pending_ron = None;
    }
    Ok(())
}

fn seat_value(value: &Option<serde_json::Value>) -> Option<u8> {
    match value {
        Some(serde_json::Value::Number(n)) => {
            let seat = n.as_u64()?;
            if seat <= 3 { Some(seat as u8) } else { None }
        }
        _ => None,
    }
}

fn int_value(value: &Option<serde_json::Value>) -> Option<i64> {
    match value {
        Some(serde_json::Value::Number(n)) => n.as_i64(),
        _ => None,
    }
}
/// Replay one framed game into decision rows (walled + wall-less, one gate).
///
/// S7 walled path: `sim.reset(wall)` is the wall-digest mint
/// (`replay-{game_id}` schedule via `wall_schedule_digest`); each logged
/// decision is one `sim.apply` in log order with the identical sequence
/// Python emits — pass placement in sorted pending order (no rows), claim
/// matching on exact strings + consumed + source, reach collapsed onto its
/// declaration dahai (`nxt + 1`), terminal checks (`end_game` break,
/// post-terminal rows fail closed). Wall-less games (`wall_tiles: None`)
/// keep the SIM-mark derivation; walled games bind the real digest.
pub fn walk_game(game: &ParsedGame, table: &ActionTable) -> Result<Vec<ReplayRow>, WalkReject> {
    walk_game_with_version(game, table, EngineVersion::V1)
}

/// Version-selected replay: [`EngineVersion::V1`] runs the pinned
/// drained-oracle backend; [`EngineVersion::V2`] runs the single-pass
/// live-engine semantics (kuikae-strict, complete-chi, kyushu, take
/// conservation). V1 is the frozen default for parity. Both versions share
/// the S7 mode machine and the walled/wall-less digest binding.
pub fn walk_game_with_version(
    game: &ParsedGame,
    table: &ActionTable,
    version: EngineVersion,
) -> Result<Vec<ReplayRow>, WalkReject> {
    prescan(game)?;
    // S7 `reset(wall)`: mint the real digest once per game (never a
    // placeholder). Wall-less stays `None` (SIM mark).
    let wall_digest: Option<String> = match &game.wall_tiles {
        None => None,
        Some(tiles) => {
            if tiles.len() != 136 {
                return Err(WalkReject::new(
                    &game.game_id,
                    -1,
                    "tile-conservation",
                    "wall",
                    "wall must carry 136 tiles",
                ));
            }
            let schedule = crate::parity::schedule_id_for(&game.game_id);
            Some(crate::parity::wall_schedule_digest(&schedule, tiles))
        }
    };
    let mut walker = Walker {
        game,
        table,
        seq: 0,
        round_idx: -1,
        hand_index: -1,
        kyoku_ordinal: -1,
        track: None,
        dealer: 0,
        histories: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
        last_kind: None,
        last_discard: None,
        opened_by_discard: false,
        decided: false,
        terminal: false,
        rows: Vec::new(),
        scores: [0; 4],
        version,
        kuikae_gate: None,
        acted: [false; 4],
        claims: 0,
        wall_digest,
    };
    let total = game.events.len();
    let mut idx = 0usize;
    while idx < total {
        let kind = game.events[idx].type_.clone();
        if is_end(&kind) {
            if walker.track.is_none() || walker.kyoku_ordinal < 0 {
                return Err(walker.fail("turn-order", "end_game", "game ends before any kyoku"));
            }
            walker.terminal = true;
            break;
        }
        if kind == "start_kyoku" {
            do_start_kyoku(&mut walker, idx)?;
            idx += 1;
            continue;
        }
        if is_start(&kind) {
            idx += 1;
            continue;
        }
        if walker.track.is_none() || walker.kyoku_ordinal < 0 {
            return Err(walker.fail("turn-order", &kind, "decision before the first start_kyoku"));
        }
        match kind.as_str() {
            "ryukyoku" => {
                do_ryukyoku(&mut walker, idx)?;
                idx += 1;
            }
            "end_kyoku" => {
                walker.push_public("round_end");
                walker.track_mut("end_kyoku")?.drawer = None;
                idx += 1;
            }
            "tsumo" => {
                do_tsumo(&mut walker, idx)?;
                idx += 1;
            }
            "dora" => {
                let marker = game.events[idx].dora_marker.clone().unwrap_or_default();
                if marker.is_empty() {
                    return Err(walker.fail("bare-dora", "dora", "dora event without a dora_marker"));
                }
                walker.track_mut("dora")?.dora.push(marker);
                walker.push_public("dora_revealed");
                idx += 1;
            }
            "reach_accepted" => {
                game.events[idx].actor_seat("reach_accepted").map_err(|e| {
                    walker.fail("turn-order", "reach_accepted", &format!("event actor must be 0..3 ({e})"))
                })?;
                walker.push_public("riichi_accepted");
                idx += 1;
            }
            _ if in_skip(&kind) => {
                idx += 1;
            }
            "reach" => {
                // Collapse with the following declaration dahai (mirrors
                // `_skip_index`: transparent + skip events never reset the
                // collapse; the declaration index `nxt + 1` is the resume
                // point, exactly like `_step_draw`'s reach path).
                let mut nxt = idx + 1;
                while nxt < total && in_skip(game.events[nxt].type_.as_str()) {
                    nxt += 1;
                }
                if nxt >= total {
                    return Err(walker.fail("turn-order", "reach", "declaration missing"));
                }
                if game.events[nxt].type_ != "dahai" {
                    return Err(walker.fail("turn-order", "reach", "declaration is not a dahai"));
                }
                // S7 mode gate (`_step_draw` reach path): the declarer must
                // own the draw decision; post-terminal reach fails closed.
                if walker.wall_digest.is_some() {
                    if let Ok(actor) = game.events[idx].actor_seat("reach") {
                        walker.check_mode_for("reach", actor, idx)?;
                    }
                }
                do_reach(&mut walker, idx, nxt)?;
                idx = nxt + 1;
            }
            _ if is_row_kind(&kind) => {
                // S7 mode gate (`_step_decision`): expected-actor /
                // window-pending / draw-mode / terminal against engine
                // answers. Wall-less skips (frozen parity); walled fails
                // closed on post-terminal rows and claims outside any
                // window, with pass placement in sorted pending order.
                if walker.wall_digest.is_some() {
                    if let Ok(actor) = game.events[idx].actor_seat(&kind) {
                        walker.check_mode_for(&kind, actor, idx)?;
                    }
                }
                match kind.as_str() {
                    k if is_claim_kind(k) => do_claim(&mut walker, idx, k)?,
                    "ankan" => do_ankan(&mut walker, idx)?,
                    "kakan" => do_kakan(&mut walker, idx)?,
                    "hora" => do_hora(&mut walker, idx)?,
                    _ => do_dahai(&mut walker, idx)?,
                }
                idx += 1;
            }
            _ => return Err(walker.fail("unknown-event", &kind, &format!("unmapped mjai event type {kind:?}"))),
        }
    }
    if !walker.terminal {
        return Err(walker.fail("turn-order", "walk", "game never reached end_game"));
    }
    Ok(walker.rows)
}

/// Engine-free row count: one per row kind with `reach` collapsed onto its
/// declaration dahai (mirrors `_count_row_decisions` + `_skip_index`).
/// The walk must emit exactly this many rows on mutually-ok games; on
/// truncated games it still counts the prefix the walk would have emitted.
pub fn count_row_decisions(game: &ParsedGame) -> Result<usize, WalkReject> {
    let total = game.events.len();
    let mut count = 0usize;
    let mut kyoku_idx: i64 = -1;
    let mut idx = 0usize;
    while idx < total {
        let kind = game.events[idx].type_.as_str();
        if is_end(kind) {
            break;
        }
        if kind == "start_kyoku" {
            kyoku_idx += 1;
        }
        if kind == "reach" {
            match skip_index(game, idx + 1, kyoku_idx)? {
                None => {
                    return Err(WalkReject::new(
                        &game.game_id,
                        kyoku_idx,
                        "turn-order",
                        "reach",
                        "reach event without a following dahai",
                    ));
                }
                Some(nxt) => {
                    if game.events[nxt].type_ != "dahai" {
                        return Err(WalkReject::new(
                            &game.game_id,
                            kyoku_idx,
                            "turn-order",
                            "reach",
                            "reach event must be followed by its declaration dahai",
                        ));
                    }
                    count += 1;
                    idx = nxt + 1;
                    continue;
                }
            }
        }
        if is_row_kind(kind) {
            count += 1;
        } else if !in_skip(kind) {
            return Err(WalkReject::new(
                &game.game_id,
                kyoku_idx,
                "unknown-event",
                kind,
                &format!("unmapped mjai event type {kind:?}"),
            ));
        }
        idx += 1;
    }
    Ok(count)
}

/// First index >= start holding a decision, reach, or end event (None when
/// the tail holds none). Mirrors `_skip_index`.
fn skip_index(game: &ParsedGame, start: usize, kyoku: i64) -> Result<Option<usize>, WalkReject> {
    let mut idx = start;
    while idx < game.events.len() {
        let kind = game.events[idx].type_.as_str();
        if is_end(kind) || is_row_kind(kind) || kind == "reach" {
            return Ok(Some(idx));
        }
        if !in_skip(kind) {
            return Err(WalkReject::new(
                &game.game_id,
                kyoku,
                "unknown-event",
                kind,
                &format!("unmapped mjai event type {kind:?}"),
            ));
        }
        idx += 1;
    }
    Ok(None)
}

// ---------------------------------------------------------------------------
// Handlers.
// ---------------------------------------------------------------------------

fn do_start_kyoku(walker: &mut Walker, idx: usize) -> Result<(), WalkReject> {
    let event = &walker.game.events[idx];
    let ordinal = walker.kyoku_ordinal + 1;
    let game_id = walker.game.game_id.clone();
    let bad = |code: &str, why: String| WalkReject::new(&game_id, ordinal, code, "start_kyoku", &why);
    let oya = int_value(&event.oya).filter(|v| (0..=3).contains(v)).ok_or_else(|| {
        bad("turn-order", "malformed hand header: oya must be 0..3".to_string())
    })? as u8;
    for (name, value) in [("honba", &event.honba), ("kyotaku", &event.kyotaku)] {
        if int_value(value).filter(|v| *v >= 0).is_none() {
            return Err(bad("turn-order", format!("malformed hand header: {name} must be a non-negative int")));
        }
    }
    let scores_raw = event.scores.clone().ok_or_else(|| bad("turn-order", "scores must cover 4 seats".to_string()))?;
    if scores_raw.len() != 4 || scores_raw.iter().any(|v| v.as_i64().is_none()) {
        return Err(bad("turn-order", "scores must cover 4 seats".to_string()));
    }
    let scores = [
        scores_raw[0].as_i64().unwrap_or(0) as i32,
        scores_raw[1].as_i64().unwrap_or(0) as i32,
        scores_raw[2].as_i64().unwrap_or(0) as i32,
        scores_raw[3].as_i64().unwrap_or(0) as i32,
    ];
    let bakaze = event.bakaze.clone().unwrap_or_default();
    if !matches!(bakaze.as_str(), "E" | "S" | "W" | "N") {
        return Err(bad("turn-order", format!("unknown bakaze {bakaze:?}")));
    }
    let tehais = event.tehais.clone().ok_or_else(|| bad("turn-order", "malformed hand header: tehais".to_string()))?;
    let track = KyokuTrack::install(&walker.game.events, ordinal as usize, idx, &tehais, &game_id)
        .map_err(|err| WalkReject {
            game_id: game_id.clone(),
            code: err.code,
            detail: err.detail,
        })?;
    walker.kyoku_ordinal = ordinal;
    walker.round_idx += 1;
    walker.hand_index += 1;
    walker.dealer = oya;
    walker.track = Some(track);
    walker.scores = scores;
    walker.last_discard = None;
    walker.opened_by_discard = false;
    walker.decided = false;
    walker.kuikae_gate = None;
    walker.acted = [false; 4];
    walker.claims = 0;
    walker.histories = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    if ordinal == 0 {
        walker.push_public("game_start");
    }
    walker.push_public("round_start");
    Ok(())
}

fn check_drawer(walker: &mut Walker, seat: u8, what: &str) -> Result<(), WalkReject> {
    let track = walker.track_mut(what)?;
    match track.drawer {
        None => track.drawer = Some(seat),
        Some(d) if d == seat => {}
        Some(_) => {
            return Err(walker.fail("turn-order", what, &format!("{what}: actor {seat} != expected decision seat")));
        }
    }
    Ok(())
}

fn do_tsumo(walker: &mut Walker, idx: usize) -> Result<(), WalkReject> {
    let (actor, pai) = {
        let event = &walker.game.events[idx];
        let actor = event
            .actor_seat("tsumo")
            .map_err(|e| walker.fail("turn-order", "tsumo", &e))?;
        let pai = event.pai.clone().unwrap_or_default();
        if pai.is_empty() {
            return Err(walker.fail("tile-conservation", "tsumo", "draw without a pai string"));
        }
        (actor, pai)
    };
    if walker.decided {
        return Err(walker.fail("turn-order", "tsumo", "draw after the kyoku was decided"));
    }
    let expected = if walker.track("tsumo")?.kan_pending {
        walker.track("tsumo")?.exp_drawer
    } else if let Some((last, _)) = walker.last_discard {
        Some((last + 1) % 4)
    } else {
        Some(walker.track("tsumo")?.exp_drawer.unwrap_or(walker.dealer))
    };
    if Some(actor) != expected {
        return Err(walker.fail(
            "turn-order",
            "tsumo",
            &format!("seat {actor} tsumo breaks turn order (out-of-turn draw)"),
        ));
    }
    if walker.track("tsumo")?.draws >= 70 {
        return Err(walker.fail("draw-past-wall", "tsumo", "draw past the end of the wall"));
    }
    let game_id = walker.game.game_id.clone();
    let copy = walker
        .track_mut("tsumo")
        .and_then(|track| {
            track.pop_draw(actor, &pai, &game_id).map_err(|err| WalkReject {
                game_id: game_id.clone(),
                code: err.code,
                detail: err.detail,
            })
        })?;
    {
        let track = walker.track_mut("tsumo")?;
        track.hands[actor as usize].push(copy);
        track.drawn[actor as usize] = Some(copy);
        track.drawn_live[actor as usize] = Some(copy);
        track.draws += 1;
        track.drawer = Some(actor);
        track.exp_drawer = Some(actor);
        track.kan_pending = false;
        track.tsumo_counts[actor as usize] += 1;
        if track.first_draws[actor as usize].is_none() {
            track.first_draws[actor as usize] = Some(pai.clone());
        }
    }
    if walker.last_discard.map(|(s, _)| s) == Some(actor) {
        walker.last_discard = None;
    }
    walker.push_public("turn_advance");
    walker.histories[actor as usize].push("draw_tile".to_string());
    Ok(())
}

/// Remove the discard copy from the tracked hand (exact value preferred,
/// drawn-preferred string match otherwise) and record the river.
fn apply_discard(
    walker: &mut Walker,
    actor: u8,
    tile: u8,
    drawn: Option<u8>,
) -> Result<(), WalkReject> {
    let game_id = walker.game.game_id.clone();
    let ordinal = walker.kyoku_ordinal;
    let in_hand = walker
        .track("dahai")
        .map(|track| track.hands[actor as usize].contains(&tile))
        .unwrap_or(false);
    let track = walker.track_mut("dahai")?;
    let hand = &mut track.hands[actor as usize];
    if in_hand {
        let pos = hand.iter().position(|t| *t == tile).unwrap_or(0);
        hand.remove(pos);
    } else {
        let pai = mjai_string_of(tile).unwrap_or_default();
        KyokuTrack::remove_one(hand, &pai, drawn, &game_id, ordinal as usize, "dahai")
            .map_err(|err| WalkReject {
                game_id: game_id.clone(),
                code: err.code,
                detail: err.detail,
            })?;
    }
    track.rivers[actor as usize].push(tile);
    track.exp_drawer = Some(actor);
    Ok(())
}

fn do_dahai(walker: &mut Walker, idx: usize) -> Result<(), WalkReject> {
    let (actor, pai, tsumogiri_flag) = {
        let event = &walker.game.events[idx];
        let actor = event
            .actor_seat("dahai")
            .map_err(|e| walker.fail("turn-order", "dahai", &e))?;
        let pai = event.pai.clone().unwrap_or_default();
        if pai.is_empty() {
            return Err(walker.fail("tile-conservation", "dahai", "discard without a pai string"));
        }
        (actor, pai, event.tsumogiri)
    };
    if walker.decided {
        return Err(walker.fail("turn-order", "dahai", "discard after the kyoku was decided"));
    }
    check_drawer(walker, actor, "dahai")?;
    let drawn = walker.track("dahai")?.drawn[actor as usize];
    let collapsed = physical_of(&pai).map_err(|e| walker.fail("tile-conservation", "dahai", &e.to_string()))?;
    // Post-reach discards report the true drawn take (never the collapsed
    // twin), mirroring the drained forced answers. Pre-reach discards
    // report the collapsed id, exactly like the yielded steps.
    let (kind, tile) = if walker.track("dahai")?.riichi_declared[actor as usize] {
        let drawn_true = walker.track("dahai")?.drawn_live[actor as usize];
        let drawn_pai = drawn_true
            .map(|d| mjai_string_of(d).map_err(|e| walker.fail("tile-conservation", "dahai", &e.to_string())))
            .transpose()?;
        if drawn_pai.as_deref() != Some(pai.as_str()) {
            return Err(walker.fail("turn-order", "dahai", "forced discard is not the drawn tile"));
        }
        ("tsumogiri", drawn_true.unwrap_or(collapsed))
    } else if tsumogiri_flag {
        ("tsumogiri", collapsed)
    } else {
        ("discard", collapsed)
    };
    let chosen = match kind {
        "tsumogiri" => ChosenAction::Tsumogiri { tile },
        _ => ChosenAction::Discard { tile },
    };
    // Engine offer: riichi-forced rows take the true-take forced pair
    // (nothing else is offered on forced discards: 2191/2191 pair-only
    // rows); every other row reads the pool-first offer.
    let mask = if walker.track("dahai")?.riichi_declared[actor as usize] {
        let forced_take = walker.track("dahai")?.drawn_live[actor as usize].ok_or_else(|| {
            walker.fail("engine-desync", "dahai", "riichi draw decision without a live drawn tile")
        })?;
        let offer = crate::engine::forced_pair(forced_take);
        walker.draw_mask_ids(actor, &offer, &chosen, false)
    } else {
        let view = walker.seat_view(actor, "dahai")?;
        let mut offer = draw_offer(&view).map_err(|err| walker.desync(err))?;
        let mut kyushu = false;
        if walker.version == crate::engine::EngineVersion::V2 {
            // One-shot immediate filter after chi/pon claims. Withholds the
            // just-claimed CALLED take (live takes never re-offer it) plus
            // (chi only) non-melded takes near the chi. Consumed takes stay
            // offered even when duped in hand (pinned: v2 offers them).
            if let Some((_, called, meld, chi)) = walker.kuikae_gate.clone() {
                let mut cands = offer.discards.clone();
                if let Some(t) = offer.tsumogiri {
                    cands.push(t);
                }
                let meldset: std::collections::HashSet<u8> = meld.into_iter().collect();
                let mut withheld: Vec<u8> = vec![called];
                for w in crate::engine::kuikae_withheld(&chi, &cands) {
                    if !meldset.contains(&w) {
                        withheld.push(w);
                    }
                }
                offer.discards.retain(|d| !withheld.contains(d));
                if let Some(t) = offer.tsumogiri {
                    if withheld.contains(&t) {
                        offer.tsumogiri = None;
                    }
                }
            }
            // Standard kyushu: first draw, no calls yet, 9+ distinct
            // terminals/honors offers the nine-terminals abort.
            let live = walker.track("dahai")?.drawn_live[actor as usize].is_some();
            if live && !walker.acted[actor as usize] && walker.claims == 0 {
                let hand14 = walker.reported_hand(actor)?;
                if crate::engine::distinct_yaochu(&hand14) >= crate::engine::KYUSHU_DISTINCT_YAOCHU {
                    kyushu = true;
                }
            }
        }
        let mut mask = walker.draw_mask_ids(actor, &offer, &chosen, false);
        if kyushu {
            if let Some(id) =
                walker.table.lookup("abort_nine_terminals", None, None, &[], None, false, false)
            {
                mask.push(id);
                mask.sort_unstable();
            }
        }
        mask
    };
    walker.capture_row(actor, "draw_decision", actor, chosen, &mask)?;
    walker.kuikae_gate = None;
    walker.track_mut("dahai")?.drawn_live[actor as usize] = None;
    apply_discard(walker, actor, tile, drawn)?;
    walker.push_public("discard");
    walker.last_discard = Some((actor, tile));
    walker.opened_by_discard = true;
    walker.track_mut("dahai")?.drawer = Some(actor);
    walker.open_window(actor, tile, false, idx + 1);
    Ok(())
}

fn do_reach(walker: &mut Walker, reach_idx: usize, decl_idx: usize) -> Result<(), WalkReject> {
    let actor = walker.game.events[reach_idx]
        .actor_seat("reach")
        .map_err(|e| walker.fail("turn-order", "reach", &e))?;
    if walker.decided {
        return Err(walker.fail("turn-order", "reach", "declaration after the kyoku was decided"));
    }
    let (dactor, pai) = {
        let decl = &walker.game.events[decl_idx];
        let dactor = decl
            .actor_seat("dahai")
            .map_err(|e| walker.fail("turn-order", "reach-declaration", &e))?;
        let pai = decl.pai.clone().unwrap_or_default();
        if pai.is_empty() {
            return Err(walker.fail("tile-conservation", "reach", "declaration without a pai string"));
        }
        (dactor, pai)
    };
    if dactor != actor {
        return Err(walker.fail("turn-order", "reach", "declaration actor differs from reach actor"));
    }
    check_drawer(walker, actor, "reach")?;
    // Declaration discard: the yielded reach step resolves drawn-preferred
    // from collapsed ids, so the tile is always the string-canonical id
    // (ownership still enforced against the tracked hand, fail closed).
    let drawn = walker.track("reach")?.drawn[actor as usize];
    let declaration_tile = physical_of(&pai).map_err(|e| walker.fail("tile-conservation", "reach", &e.to_string()))?;
    {
        let track = walker.track("reach")?;
        let owned = track.hands[actor as usize]
            .iter()
            .any(|t| mjai_string_of(*t).map(|p| p == pai).unwrap_or(false));
        if !owned {
            return Err(walker.fail("tile-conservation", "reach", "declaration discard not owned"));
        }
    }
    let chosen = ChosenAction::RiichiDiscard { tile: declaration_tile };
    let view = walker.seat_view(actor, "reach")?;
    let offer = draw_offer(&view).map_err(|err| walker.desync(err))?;
    if !offer.riichi.contains(&declaration_tile) {
        return Err(walker.fail(
            "engine-desync",
            "reach",
            &format!("declaration discard {declaration_tile} is not a riichi candidate"),
        ));
    }
    let mask = walker.draw_mask_ids(actor, &offer, &chosen, false);
    walker.capture_row(actor, "draw_decision", actor, chosen, &mask)?;
    walker.track_mut("reach")?.drawn_live[actor as usize] = None;
    apply_discard(walker, actor, declaration_tile, drawn)?;
    walker.push_public("discard");
    walker.last_discard = Some((actor, declaration_tile));
    walker.opened_by_discard = true;
    walker.track_mut("reach")?.drawer = Some(actor);
    walker.track_mut("reach")?.riichi_declared[actor as usize] = true;
    walker.scores[actor as usize] -= 1000;
    walker.open_window(actor, declaration_tile, false, decl_idx + 1);
    Ok(())
}

fn do_claim(walker: &mut Walker, idx: usize, kind: &str) -> Result<(), WalkReject> {
    let (actor, pai, consumed_raw, target) = {
        let event = &walker.game.events[idx];
        let actor = event
            .actor_seat(kind)
            .map_err(|e| walker.fail("turn-order", kind, &e))?;
        let pai = event.pai.clone().unwrap_or_default();
        if pai.is_empty() {
            return Err(walker.fail("tile-conservation", kind, "logged claim without a pai string"));
        }
        let consumed = event.consumed.clone().unwrap_or_default();
        let target = event.target_seat().map_err(|e| walker.fail("turn-order", kind, &e))?;
        (actor, pai, consumed, target)
    };
    if walker.decided {
        return Err(walker.fail("turn-order", kind, "claim after the kyoku was decided"));
    }
    let (discarder, called) = match walker.last_discard {
        Some(pair) => pair,
        None => return Err(walker.fail("claim-no-offer", kind, "claim without a live discard offer")),
    };
    if target != discarder {
        return Err(walker.fail("claim-no-offer", kind, &format!("claim target {target} != discarder {discarder}")));
    }
    let called_pai = mjai_string_of(called).map_err(|e| walker.fail("tile-conservation", kind, &e.to_string()))?;
    if called_pai != pai {
        return Err(walker.fail("claim-no-offer", kind, "claim names a different tile than the offer"));
    }
    let needed = if kind == "daiminkan" { 3 } else { 2 };
    // Pool-first consumed resolution with ownership enforced on the
    // tracked hand (mirrors `_tracked_consumed` outcome).
    let consumed = resolve_claim_tiles(walker, actor, &consumed_raw, needed, Some(called))?;
    let chosen = match kind {
        "chi" => ChosenAction::Chi { called, consumed: consumed.clone() },
        "pon" => ChosenAction::Pon { called, consumed: consumed.clone(), source: discarder },
        _ => ChosenAction::Daiminkan { called, consumed: consumed.clone(), source: discarder },
    };
    let v2 = walker.version == crate::engine::EngineVersion::V2;
    // Claim-response offer. V1 offers exactly `{pass, chosen}` (no chi
    // enumeration beyond the logged claim). V2 additionally enumerates
    // every legal chi pattern (complete-chi, kamicha-gated). Melded-take
    // withholding rides the one-shot immediate filter in `do_dahai`.
    let mut mask = walker.claim_mask_ids(actor, &chosen);
    if v2 && actor == (discarder + 1) % 4 {
        // Complete-chi: every sequence position containing the called
        // tile, with min-held takes per consumed type (live takes are
        // pool-first-held; pinned by d0282 [48,53]+[53,60]+[60,64] and
        // d0299 [8,17]). Patterns whose takes are absent resolve to
        // nothing and are skipped.
        let hand_true = walker.track(kind)?.hands[actor as usize].clone();
        for pair in crate::engine::complete_chi(&hand_true, called) {
            let probe = ChosenAction::Chi { called, consumed: pair };
            if let Some(id) = probe.lookup(walker.table, actor) {
                if !mask.contains(&id) {
                    mask.push(id);
                }
            }
        }
        mask.sort_unstable();
    }
    walker.capture_row(actor, "discard_response", discarder, chosen, &mask)?;
    walker.track_mut(kind)?.drawn_live[actor as usize] = None;
    // Meld record (sorted tiles, like the oracle's VisibleMeld).
    let mut tiles = consumed.clone();
    tiles.push(called);
    tiles.sort_unstable();
    let game_id = walker.game.game_id.clone();
    let ordinal = walker.kyoku_ordinal;
    let kind_owned = kind.to_string();
    {
        let track = walker.track_mut(&kind_owned)?;
        let hand = &mut track.hands[actor as usize];
        for tile in &consumed {
            if let Some(pos) = hand.iter().position(|t| t == tile) {
                hand.remove(pos);
            } else {
                let pai = mjai_string_of(*tile).unwrap_or_default();
                KyokuTrack::remove_one(hand, &pai, None, &game_id, ordinal as usize, &kind_owned)
                    .map_err(|err| WalkReject {
                        game_id: game_id.clone(),
                        code: err.code,
                        detail: err.detail,
                    })?;
            }
        }
        track.melds[actor as usize].push(crate::stream::TrackedMeld {
            kind: kind_owned,
            owner: actor,
            tiles,
            source: Some(discarder),
            called: Some(called),
        });
        track.drawer = Some(actor);
        track.exp_drawer = Some(actor);
        track.kan_pending = kind == "daiminkan";
    }
    if walker.version == crate::engine::EngineVersion::V2 && (kind == "chi" || kind == "pon") {
        let gate: Vec<u8> = walker.track(kind)?.melds[actor as usize].last().map(|m| m.tiles.clone()).unwrap_or_default();
        // Chi gates both the meld takes and their same-suit neighborhood;
        // pon gates only the meld takes (no kuikae neighborhood, observed).
        let chi = if kind == "chi" { gate.clone() } else { Vec::new() };
        walker.kuikae_gate = Some((actor, called, gate, chi));
    }
    walker.push_public(kind);
    walker.last_discard = None;
    Ok(())
}
/// Pool-first consumed resolution: sorted strings take the smallest pool
/// copies (the engine reports string-canonical ids), after enforcing
/// ownership against the tracked hand. A picked copy colliding with the
/// called tile is swapped for an unused pool copy of the same string.
fn resolve_claim_tiles(
    walker: &Walker,
    seat: u8,
    consumed_strings: &[String],
    needed: usize,
    called: Option<u8>,
) -> Result<Vec<u8>, WalkReject> {
    let step = "claim";
    let track = walker.track(step)?;
    let hand = &track.hands[seat as usize];
    let mut sorted = consumed_strings.to_vec();
    sorted.sort();
    if sorted.len() != needed {
        return Err(walker.fail("tile-conservation", step, &format!("claim needs {needed} consumed tiles")));
    }
    // Ownership: the tracked hand must render every consumed string.
    let mut remaining: HashMap<String, usize> = HashMap::new();
    for id in hand {
        if let Ok(pai) = mjai_string_of(*id) {
            *remaining.entry(pai).or_insert(0) += 1;
        }
    }
    let mut picked: Vec<u8> = Vec::with_capacity(sorted.len());
    let mut picked_counts: HashMap<String, usize> = HashMap::new();
    for pai in &sorted {
        let have = remaining.get(pai).copied().unwrap_or(0);
        let used = picked_counts.get(pai).copied().unwrap_or(0);
        if used >= have {
            return Err(walker.fail("tile-conservation", step, &format!("no tracked copy left for meld tile {pai:?}")));
        }
        let pool = copies_of_string(pai).map_err(|e| walker.fail("tile-conservation", step, &e.to_string()))?;
        let take = pool.get(used).copied().ok_or_else(|| {
            walker.fail("tile-conservation", step, &format!("no tracked copy left for meld tile {pai:?}"))
        })?;
        picked.push(take);
        picked_counts.insert(pai.clone(), used + 1);
    }
    if let Some(called_id) = called {
        for pos in 0..picked.len() {
            if picked[pos] == called_id {
                let pai = mjai_string_of(picked[pos])
                    .map_err(|e| walker.fail("tile-conservation", step, &e.to_string()))?;
                let used: HashSet<u8> = picked.iter().copied().chain([called_id]).collect();
                let mut swapped = false;
                for candidate in copies_of_string(&pai)
                    .map_err(|e| walker.fail("tile-conservation", step, &e.to_string()))?
                {
                    if !used.contains(&candidate) {
                        picked[pos] = candidate;
                        swapped = true;
                        break;
                    }
                }
                if !swapped {
                    return Err(
                        walker.fail("tile-conservation", step, &format!("no distinct copy left for meld tile {pai:?}"))
                    );
                }
            }
        }
    }
    picked.sort_unstable();
    Ok(picked)
}

fn do_ankan(walker: &mut Walker, idx: usize) -> Result<(), WalkReject> {
    let (actor, consumed_raw) = {
        let event = &walker.game.events[idx];
        let actor = event
            .actor_seat("ankan")
            .map_err(|e| walker.fail("turn-order", "ankan", &e))?;
        (actor, event.consumed.clone().unwrap_or_default())
    };
    if walker.decided {
        return Err(walker.fail("turn-order", "ankan", "kan after the kyoku was decided"));
    }
    check_drawer(walker, actor, "ankan")?;
    let block = resolve_claim_tiles(walker, actor, &consumed_raw, 4, None)?;
    let base = (block[0] / 4) * 4;
    if block != vec![base, base + 1, base + 2, base + 3] {
        return Err(walker.fail("tile-conservation", "ankan", &format!("ankan tiles {block:?} are not one block")));
    }
    let chosen = ChosenAction::Ankan { consumed: block.clone() };
    // Engine draw offer. Under riichi the row offers the pool-first forced
    // pair (never full discards), like tsumo wins; otherwise the full
    // pool-first offer.
    let view = walker.seat_view(actor, "ankan")?;
    let offer = draw_offer(&view).map_err(|err| walker.desync(err))?;
    let mask = if walker.track("ankan")?.riichi_declared[actor as usize] {
        // Kans need a live draw turn; without one the log is illegal
        // (post-claim kan) and fails closed.
        let live_drawn = view.drawn.ok_or_else(|| {
            walker.fail("engine-desync", "ankan", "kan without a live draw")
        })?;
        let forced = crate::engine::forced_pair(live_drawn);
        // The logged quad is always offered (log authority, mirroring the
        // oracle's chosen forcing); other quads ride the same-wait offer.
        // Membership fails closed only when no quad is offerable at all
        // (exhausted wall or no full quad in hand).
        let mut quads = offer.ankan;
        if !quads.iter().any(|quad| *quad == block) {
            if view.wall < crate::engine::KAN_MIN_WALL {
                return Err(walker.fail(
                    "engine-desync",
                    "ankan",
                    &format!("logged quad {block:?} is not an offered ankan"),
                ));
            }
            let mut have = block.clone();
            have.sort_unstable();
            quads.push(have);
        }
        let combined = crate::engine::DrawOffer {
            discards: forced.discards,
            tsumogiri: forced.tsumogiri,
            ankan: quads,
            ..Default::default()
        };
        walker.draw_mask_ids(actor, &combined, &chosen, false)
    } else {
        // Match-offered check (mirrors `_strict_row`): the logged quad
        // must sit in the engine's ankan answers, else desync.
        if !offer.ankan.iter().any(|quad| *quad == block) {
            return Err(walker.fail(
                "engine-desync",
                "ankan",
                &format!("logged quad {block:?} is not an offered ankan"),
            ));
        }
        walker.draw_mask_ids(actor, &offer, &chosen, false)
    };
    walker.capture_row(actor, "draw_decision", actor, chosen, &mask)?;
    walker.track_mut("ankan")?.drawn_live[actor as usize] = None;
    let game_id = walker.game.game_id.clone();
    let ordinal = walker.kyoku_ordinal;
    {
        let track = walker.track_mut("ankan")?;
        let hand = &mut track.hands[actor as usize];
        for tile in &block {
            if let Some(pos) = hand.iter().position(|t| t == tile) {
                hand.remove(pos);
            } else {
                let pai = mjai_string_of(*tile).unwrap_or_default();
                KyokuTrack::remove_one(hand, &pai, None, &game_id, ordinal as usize, "ankan")
                    .map_err(|err| WalkReject {
                        game_id: game_id.clone(),
                        code: err.code,
                        detail: err.detail,
                    })?;
            }
        }
        track.melds[actor as usize].push(crate::stream::TrackedMeld {
            kind: "ankan".to_string(),
            owner: actor,
            tiles: block,
            source: None,
            called: None,
        });
        track.drawer = Some(actor);
        track.exp_drawer = Some(actor);
        track.kan_pending = true;
    }
    walker.push_public("ankan");
    walker.claims += 1;
    Ok(())
}

fn do_kakan(walker: &mut Walker, idx: usize) -> Result<(), WalkReject> {
    let (actor, pai) = {
        let event = &walker.game.events[idx];
        let actor = event
            .actor_seat("kakan")
            .map_err(|e| walker.fail("turn-order", "kakan", &e))?;
        let pai = event.pai.clone().unwrap_or_default();
        if pai.is_empty() {
            return Err(walker.fail("tile-conservation", "kakan", "logged kakan without a pai string"));
        }
        (actor, pai)
    };
    if walker.decided {
        return Err(walker.fail("turn-order", "kakan", "kan after the kyoku was decided"));
    }
    check_drawer(walker, actor, "kakan")?;
    // The added fourth copy is the one pool copy of this type absent from
    // the prior pon triple (deterministic, contract-valid).
    let added = {
        let track = walker.track("kakan")?;
        let step_tile = physical_of(&pai).map_err(|e| walker.fail("tile-conservation", "kakan", &e.to_string()))?;
        let added_type = step_tile / 4;
        let prior = track.melds[actor as usize]
            .iter()
            .find(|m| m.kind == "pon" && m.tiles.first().map(|t| t / 4) == Some(added_type))
            .ok_or_else(|| walker.fail("tile-conservation", "kakan", &format!("no prior pon owned by seat {actor}")))?;
        let mut pool = copies_of_string(&pai).map_err(|e| walker.fail("tile-conservation", "kakan", &e.to_string()))?;
        if pai.len() == 2 && pai.as_bytes()[0] == b'5' {
            let base = (physical_of(&pai).map_err(|e| walker.fail("tile-conservation", "kakan", &e.to_string()))? / 4) * 4;
            pool = vec![base, base + 1, base + 2, base + 3];
        }
        let owned: HashSet<u8> = prior.tiles.iter().copied().collect();
        let missing: Vec<u8> = pool.into_iter().filter(|c| !owned.contains(c)).collect();
        if missing.len() != 1 {
            return Err(walker.fail("tile-conservation", "kakan", "prior pon leaves no single added copy"));
        }
        missing[0]
    };
    let chosen = ChosenAction::Kakan { tile: added };
    let view = walker.seat_view(actor, "kakan")?;
    let offer = draw_offer(&view).map_err(|err| walker.desync(err))?;
    let mask = walker.draw_mask_ids(actor, &offer, &chosen, false);
    walker.capture_row(actor, "draw_decision", actor, chosen, &mask)?;
    walker.track_mut("kakan")?.drawn_live[actor as usize] = None;
    let game_id = walker.game.game_id.clone();
    let ordinal = walker.kyoku_ordinal;
    {
        let track = walker.track_mut("kakan")?;
        // The meld record keeps the prior pon (mirrors the oracle, which
        // never rewrites state melds on kakan); the added copy leaves the hand.
        let hand = &mut track.hands[actor as usize];
        if let Some(pos) = hand.iter().position(|t| *t == added) {
            hand.remove(pos);
        } else {
            let owed = mjai_string_of(added).unwrap_or_default();
            KyokuTrack::remove_one(hand, &owed, None, &game_id, ordinal as usize, "kakan")
                .map_err(|err| WalkReject {
                    game_id: game_id.clone(),
                    code: err.code,
                    detail: err.detail,
                })?;
        }
        track.drawer = Some(actor);
        track.exp_drawer = Some(actor);
        track.kan_pending = true;
    }
    walker.push_public("kakan");
    walker.claims += 1;
    walker.last_discard = Some((actor, added));
    walker.opened_by_discard = false;
    // Kan windows never emit envelopes (the adapter grammar routes kakan
    // straight to ron); the window still resolves responders upstream.
    walker.open_window(actor, added, true, idx + 1);
    Ok(())
}

fn do_hora(walker: &mut Walker, idx: usize) -> Result<(), WalkReject> {
    let (winner, tsumo_flag, target, deltas) = {
        let event = &walker.game.events[idx];
        let winner = event
            .actor_seat("hora")
            .map_err(|e| walker.fail("turn-order", "hora", &e))?;
        let target = event.target_seat().map_err(|e| walker.fail("turn-order", "hora", &e))?;
        let deltas = event.deltas.clone().unwrap_or_default();
        if deltas.len() != 4 || deltas.iter().any(|d| d.as_i64().is_none()) {
            return Err(walker.fail("turn-order", "hora", "hora without a 4-seat deltas quad"));
        }
        (winner, event.tsumo, target, deltas)
    };
    // Tenhou-real tsumo wins carry no flag (and no pai); target==actor
    // marks them, while ron names the discarder.
    let self_draw = tsumo_flag || target == winner;
    if walker.decided {
        return Err(walker.fail("turn-order", "hora", "win after the kyoku was decided"));
    }
    // Tracked scores follow the log deltas (riichi sticks already moved at
    // declaration time), mirroring `tracked_scores`.
    for (seat, delta) in deltas.iter().enumerate() {
        if let Some(add) = delta.as_i64() {
            walker.scores[seat] += add as i32;
        }
    }
    if self_draw {
        check_drawer(walker, winner, "hora")?;
        let drawn = walker
            .reported_drawn(winner)?
            .ok_or_else(|| walker.fail("tile-conservation", "hora", "tsumo win without a drawn tile"))?;
        let chosen = ChosenAction::Tsumo { tile: drawn };
        // Engine draw offer (post-reach rows collapse to the forced pair
        // inside the engine) plus the taken-win tsumo answer. The shape
        // gate is the fail-closed net: no win holds without a winning
        // shape, so a shapeless logged win desyncs instead of emitting.
        let track = walker.track("hora")?;
        // Shape gate over the collapsed takes (mirrors the oracle's
        // collapsed hand the evaluator reads).
        let hand14 = walker.reported_hand(winner)?;
        if !crate::engine::win_shape_14(&hand14, track.melds[winner as usize].len()) {
            return Err(walker.fail(
                "engine-desync",
                "hora",
                "tsumo win without a winning shape",
            ));
        }
        let mask = if walker.track("hora")?.riichi_declared[winner as usize] {
            // Tsumo wins under riichi offer the pool-first forced pair
            // (never full discards), plus the taken-win tsumo answer and
            // same-wait ankan answers (none occur in corpus, kept for
            // engine fidelity).
            let view = walker.seat_view(winner, "hora")?;
            let offer = draw_offer(&view).map_err(|err| walker.desync(err))?;
            let forced = crate::engine::forced_pair(drawn);
            let combined = crate::engine::DrawOffer {
                discards: forced.discards,
                tsumogiri: forced.tsumogiri,
                ankan: offer.ankan,
                ..Default::default()
            };
            walker.draw_mask_ids(winner, &combined, &chosen, true)
        } else {
            let view = walker.seat_view(winner, "hora")?;
            let offer = draw_offer(&view).map_err(|err| walker.desync(err))?;
            walker.draw_mask_ids(winner, &offer, &chosen, true)
        };
        walker.capture_row(winner, "draw_decision", winner, chosen, &mask)?;
        walker.push_public("tsumo");
    } else {
        let (discarder, tile) = match walker.last_discard {
            Some(pair) => pair,
            None => return Err(walker.fail("claim-no-offer", "hora", "ron without a live discard offer")),
        };
        if target != discarder {
            return Err(walker.fail(
                "claim-no-offer",
                "hora",
                &format!("ron target {target} != discarder {discarder}"),
            ));
        }
        let phase = if walker.opened_by_discard {
            "discard_response"
        } else {
            "kan_response"
        };
        // Shape gate, as above: the discard must complete a winning shape
        // for the winner's tracked hand.
        let track = walker.track("hora")?;
        let mut completed = track.hands[winner as usize].clone();
        completed.push(tile);
        if !crate::engine::win_shape_14(&completed, track.melds[winner as usize].len()) {
            return Err(walker.fail(
                "engine-desync",
                "hora",
                "ron win without a winning shape",
            ));
        }
        let chosen = ChosenAction::Ron { tile, source: discarder };
        let mask = walker.ron_mask_ids(winner, tile, discarder);
        walker.capture_row(winner, phase, discarder, chosen, &mask)?;
        walker.push_public("ron");
    }
    walker.decided = true;
    walker.track_mut("hora")?.drawer = None;
    walker.track_mut("hora")?.exp_drawer = None;
    walker.track_mut("hora")?.kan_pending = false;
    Ok(())
}

fn do_ryukyoku(walker: &mut Walker, idx: usize) -> Result<(), WalkReject> {
    let (reason_raw, deltas) = {
        let event = &walker.game.events[idx];
        (event.reason.clone(), event.deltas.clone().unwrap_or_default())
    };
    if walker.decided {
        return Err(walker.fail("turn-order", "ryukyoku", "draw after the kyoku was decided"));
    }
    if deltas.len() != 4 || deltas.iter().any(|d| d.as_i64().is_none()) {
        return Err(walker.fail("turn-order", "ryukyoku", "draw without a 4-seat deltas quad"));
    }
    for (seat, delta) in deltas.iter().enumerate() {
        if let Some(add) = delta.as_i64() {
            walker.scores[seat] += add as i32;
        }
    }
    // Tenhou-real MJAI omits the reason for wall exhaustion and first-turn
    // kyushu aborts; every other abort carries one. The kyushu inference
    // calls the engine's first-draw rule (never a mask offer: the drained
    // engine answers no kyushu bit on any row).
    let reason = match reason_raw {
        Some(r) if !r.is_empty() => r,
        _ => {
            let track = walker.track("ryukyoku")?;
            let draws = track.draws;
            if draws >= 60 {
                "exhaustive_draw".to_string()
            } else if draws <= 4 {
                let kyushu = match track.drawer {
                    Some(seat) => track.first_draws[seat as usize].clone().map(|first| {
                        kyushu_first_draw(
                            &track.tehais[seat as usize],
                            &first,
                            track.tsumo_counts[seat as usize] as usize,
                        )
                    }),
                    None => None,
                };
                match kyushu {
                    Some(true) => "kyushu_kyuhai".to_string(),
                    _ => {
                        return Err(walker.fail(
                            "kyushu-ambiguous",
                            "ryukyoku",
                            "draw without a reason string outside kyushu/exhaustive shape",
                        ));
                    }
                }
            } else {
                return Err(walker.fail(
                    "kyushu-ambiguous",
                    "ryukyoku",
                    "draw without a reason string outside kyushu/exhaustive shape",
                ));
            }
        }
    };
    // History envelope kind follows the rules-manifest reason classes.
    let draw_end = matches!(reason.as_str(), "exhaustive_draw" | "nagashi_mangan");
    let abortive = matches!(
        reason.as_str(),
        "kyushu_kyuhai"
            | "kyuushu_kyuuhai"
            | "suucha_riichi"
            | "sanchaho"
            | "sanchahou"
            | "suukaikan"
            | "suukansansen"
            | "suufon_renda"
            | "sufuurenta"
    );
    if !draw_end && !abortive {
        return Err(walker.fail("unmapped-ryukyoku-reason", "ryukyoku", &format!("unmapped ryukyoku reason {reason:?}")));
    }
    walker.push_public(if draw_end { "draw_end" } else { "abortive_draw" });
    walker.decided = true;
    let track = walker.track_mut("ryukyoku")?;
    track.drawer = None;
    track.exp_drawer = None;
    track.kan_pending = false;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod walk_skeleton_tests {
    //! Slice S2 gate: row COUNT + decision/round id sequences equal
    //! `_count_row_decisions` (+ the `d{seq:04}` / `h{idx:02}` scheme), and
    //! the walk skeleton (start_kyoku / END / terminal-required /
    //! reach-collapse / unknown-event) quarantines like the oracle.
    //!
    //! Fixture W mirrors /tmp/s2/oracle_dump.py (throwaway probe): its
    //! count (5), decision ids, and round ids below are the oracle outputs
    //! verbatim. Masks/observations are excluded from this gate.
    use super::*;
    use crate::stream::parse_game;

    const TILES: [&str; 34] = [
        "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p", "5p",
        "6p", "7p", "8p", "9p", "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "E",
        "S", "W", "N", "P", "F", "C",
    ];

    fn tile_at(slot: usize) -> String {
        TILES[slot % TILES.len()].to_string()
    }

    fn tehais_json() -> String {
        let hands: Vec<String> = (0..4)
            .map(|seat| {
                let tiles: Vec<String> = match seat {
                    // Seat 1: 13-tile shanpon tenpai (567p/123s/456s + 7s/8s
                    // pairs), so the logged 2s declaration below is a
                    // genuine riichi candidate (engine match-offered rule).
                    1 => [
                        "5p", "6p", "7p", "1s", "2s", "3s", "4s", "5s", "6s", "7s", "7s",
                        "8s", "8s",
                    ]
                    .iter()
                    .map(|t| format!("{t:?}"))
                    .collect(),
                    // Seat 2: EEE/SSS/WWW/PPP + FF pair, so the logged E
                    // tsumo below completes a genuine win (shape gate).
                    2 => [
                        "E", "E", "S", "S", "S", "W", "W", "W", "P", "P", "P", "F",
                        "F",
                    ]
                    .iter()
                    .map(|t| format!("{t:?}"))
                    .collect(),
                    _ => (0..13).map(|i| format!("{:?}", tile_at(seat * 13 + i))).collect(),
                };
                format!("[{}]", tiles.join(","))
            })
            .collect();
        format!("[{}]", hands.join(","))
    }

    fn start_kyoku(oya: u8, no: u8) -> String {
        format!(
            "{{\"type\":\"start_kyoku\",\"bakaze\":\"E\",\"kyoku\":{no},\"honba\":0,\"kyotaku\":0,\"oya\":{oya},\"scores\":[25000,25000,25000,25000],\"dora_marker\":\"3m\",\"tehais\":{}}}",
            tehais_json()
        )
    }

    fn empty_table() -> ActionTable {
        ActionTable::load_unverified_json("{\"payload\":{\"actions\":[]}}").unwrap()
    }

    /// Two-kyoku walk: dahai, reach (transparent dora before its
    /// declaration), tsumo-hora, then dahai/dahai + exhaustive ryukyoku.
    fn two_kyoku_text() -> String {
        let w: Vec<String> = (52..57).map(tile_at).collect();
        let lines = vec![
            "{\"type\":\"start_game\"}".to_string(),
            start_kyoku(0, 1),
            format!("{{\"type\":\"tsumo\",\"actor\":0,\"pai\":{:?}}}", w[0]),
            format!("{{\"type\":\"dahai\",\"actor\":0,\"pai\":{:?},\"tsumogiri\":true}}", w[0]),
            format!("{{\"type\":\"tsumo\",\"actor\":1,\"pai\":{:?}}}", w[1]),
            "{\"type\":\"reach\",\"actor\":1}".to_string(),
            "{\"type\":\"dora\",\"dora_marker\":\"5p\"}".to_string(),
            format!("{{\"type\":\"dahai\",\"actor\":1,\"pai\":{:?},\"tsumogiri\":true}}", w[1]),
            "{\"type\":\"tsumo\",\"actor\":2,\"pai\":\"E\"}".to_string(),
            format!(
                "{{\"type\":\"hora\",\"actor\":2,\"target\":2,\"tsumo\":true,\"deltas\":[8000,-2000,-2000,-4000]}}"
            ),
            "{\"type\":\"end_kyoku\"}".to_string(),
            start_kyoku(1, 2),
            format!("{{\"type\":\"tsumo\",\"actor\":1,\"pai\":{:?}}}", w[3]),
            format!("{{\"type\":\"dahai\",\"actor\":1,\"pai\":{:?},\"tsumogiri\":true}}", w[3]),
            format!("{{\"type\":\"tsumo\",\"actor\":2,\"pai\":{:?}}}", w[4]),
            format!("{{\"type\":\"dahai\",\"actor\":2,\"pai\":{:?},\"tsumogiri\":true}}", w[4]),
            "{\"type\":\"ryukyoku\",\"reason\":\"exhaustive_draw\",\"deltas\":[0,0,0,0]}".to_string(),
            "{\"type\":\"end_kyoku\"}".to_string(),
            "{\"type\":\"end_game\"}".to_string(),
        ];
        lines.join("\n") + "\n"
    }

    #[test]
    fn count_and_ids_match_oracle_on_two_kyoku() {
        let game = parse_game(&two_kyoku_text(), "s2-walk").unwrap();
        let table = empty_table();
        // Oracle `_count_row_decisions` == 5 on this fixture.
        assert_eq!(count_row_decisions(&game).unwrap(), 5);
        let rows = walk_game(&game, &table).unwrap();
        assert_eq!(rows.len(), 5);
        // Decision ids stay positional and gapless across the kyoku seam.
        let ids: Vec<&str> = rows.iter().map(|r| r.decision_id.as_str()).collect();
        for (seq, id) in ids.iter().enumerate() {
            assert_eq!(*id, format!("{}:d{seq:04}", game.game_id));
        }
        // Round ids flip at the kyoku boundary (3 rows in h00, 2 in h01).
        let rounds: Vec<&str> = rows.iter().map(|r| r.round_id.as_str()).collect();
        assert_eq!(
            rounds,
            vec![
                format!("{}:h00", game.game_id),
                format!("{}:h00", game.game_id),
                format!("{}:h00", game.game_id),
                format!("{}:h01", game.game_id),
                format!("{}:h01", game.game_id),
            ]
        );
        assert_eq!(
            rows.iter().map(|r| r.seat).collect::<Vec<_>>(),
            vec![0, 1, 2, 1, 2]
        );
        // Countdown wall: 70 minus per-kyoku draws at capture time (kyoku 1
        // rows land after 1, 2, and 3 draws; kyoku 2 resets the countdown).
        assert_eq!(
            rows.iter().map(|r| r.wall_remaining).collect::<Vec<_>>(),
            vec![69, 68, 67, 69, 68]
        );
    }

    #[test]
    fn kan_fixture_count_matches_oracle() {
        // Count-only (no row walk): the kan/rinshan kyoku carries 9 row
        // decisions per the oracle (6 dahai + ankan + kakan + daiminkan).
        let text = crate::stream::s2_kan_game_text();
        let game = parse_game(&text, "s2-kan").unwrap();
        assert_eq!(count_row_decisions(&game).unwrap(), 9);
    }

    #[test]
    fn unknown_event_quarantines_both_paths() {
        let lines = vec![
            "{\"type\":\"start_game\"}".to_string(),
            start_kyoku(0, 1),
            format!("{{\"type\":\"tsumo\",\"actor\":0,\"pai\":{:?}}}", tile_at(52)),
            "{\"type\":\"frobnicator\",\"actor\":0}".to_string(),
            "{\"type\":\"end_game\"}".to_string(),
        ];
        let game = parse_game(&(lines.join("\n") + "\n"), "s2-unknown").unwrap();
        // The oracle `_count_row_decisions` raises unmapped here too.
        let count_err = count_row_decisions(&game).unwrap_err();
        assert_eq!(count_err.code, "unknown-event");
        let walk_err = walk_game(&game, &empty_table()).unwrap_err();
        assert_eq!(walk_err.code, "unknown-event");
    }

    #[test]
    fn truncated_game_fails_terminal_required_but_counts_prefix() {
        let mut game = parse_game(&two_kyoku_text(), "s2-trunc").unwrap();
        game.events.pop();
        assert_ne!(game.events.last().map(|e| e.type_.as_str()), Some("end_game"));
        let err = walk_game(&game, &empty_table()).unwrap_err();
        assert_eq!(err.code, "turn-order");
        assert!(err.detail.contains("game never reached end_game"));
        // The engine-free count still reports the prefix honestly.
        assert_eq!(count_row_decisions(&game).unwrap(), 5);
    }

    #[test]
    fn end_before_any_kyoku_is_turn_order() {
        let game = parse_game("{\"type\":\"start_game\"}\n{\"type\":\"end_game\"}\n", "s2-empty").unwrap();
        let err = walk_game(&game, &empty_table()).unwrap_err();
        assert_eq!(err.code, "turn-order");
        assert_eq!(count_row_decisions(&game).unwrap(), 0);
    }

    #[test]
    fn reach_without_declaration_fails_both_paths() {
        let lines = vec![
            "{\"type\":\"start_game\"}".to_string(),
            start_kyoku(0, 1),
            format!("{{\"type\":\"tsumo\",\"actor\":0,\"pai\":{:?}}}", tile_at(52)),
            "{\"type\":\"reach\",\"actor\":0}".to_string(),
            "{\"type\":\"end_game\"}".to_string(),
        ];
        let game = parse_game(&(lines.join("\n") + "\n"), "s2-reach").unwrap();
        assert_eq!(walk_game(&game, &empty_table()).unwrap_err().code, "turn-order");
        assert_eq!(count_row_decisions(&game).unwrap_err().code, "turn-order");
    }
}

#[cfg(test)]
mod s3_chosen_resolution_tests {
    //! Slice S3 gate (PG-S3): chosen-action resolution on ledger copies.
    //!
    //! Pins `resolve_claim_tiles` (pool-first smallest-copy + called-collision
    //! swap, matching the `_tracked_consumed` outcome on the copy-collapsed
    //! oracle wall), `apply_discard` (exact-then-drawn-preferred, matching
    //! `_track_discard`), and the `do_*` dispatch order
    //! (tsumo/dahai/reach/claim/ankan/kakan/hora/ryukyoku). Matching stays
    //! ledger-copy based: no engine offer is consulted (that is S6).
    //!
    //! KNOWN-RESIDUAL (recorded, never fixed here — fixing any of these would
    //! reimplement engine rules in a second language; S6 deletes the
    //! approximations with the engine-answer bridge):
    //! - kuikae-discards: the live engine withholds kuikae-illegal discards
    //!   after melds; the wall-less driver offers every string-kind instead
    //!   (probe class `mask-extra/kuikae-discards`, post-claim rows).
    //! - complete-chi variants: string-distinct chi enumeration rejoins here
    //!   while the replay engine offers fewer (probe class
    //!   `mask-extra/chi-variants`, claim rows).
    //! - kyushu proxy over-fire + missing riichi-declaration candidates: win
    //!   evaluation is engine-owned (probe classes `mask-extra/kyushu-proxy`
    //!   and `mask-missing/riichi-candidates`).
    //!
    //! Copy-identity battery: red discard/tsumogiri, ankan strings, chi
    //! order-permutation, kakan-from-pon — smallest-physical-id choice,
    //! byte-identical to the Python oracle on the copy-collapsed wall
    //! (every occurrence of one MJAI string reports its base id, so the
    //! collapsed id IS the oracle id).
    use super::*;
    use crate::stream::parse_game;

    const TILES: [&str; 34] = [
        "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p", "5p",
        "6p", "7p", "8p", "9p", "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "E",
        "S", "W", "N", "P", "F", "C",
    ];

    fn tile_at(slot: usize) -> String {
        TILES[slot % TILES.len()].to_string()
    }

    /// Cycling tehais (52 distinct slot tiles over 34 kinds, at most two
    /// copies per kind, never red) — overuse-safe for simple games.
    fn cycling_tehais() -> String {
        let hands: Vec<String> = (0..4)
            .map(|seat| {
                let tiles: Vec<String> =
                    (0..13).map(|i| format!("{:?}", tile_at(seat * 13 + i))).collect();
                format!("[{}]", tiles.join(","))
            })
            .collect();
        format!("[{}]", hands.join(","))
    }

    fn tehais_of(seats: &[&[&str]]) -> String {
        let hands: Vec<String> = seats
            .iter()
            .map(|hand| {
                let tiles: Vec<String> = hand.iter().map(|t| format!("{t:?}")).collect();
                format!("[{}]", tiles.join(","))
            })
            .collect();
        format!("[{}]", hands.join(","))
    }

    fn start_kyoku_of(oya: u8, no: u8, tehais: &str) -> String {
        format!(
            "{{\"type\":\"start_kyoku\",\"bakaze\":\"E\",\"kyoku\":{no},\"honba\":0,\"kyotaku\":0,\"oya\":{oya},\"scores\":[25000,25000,25000,25000],\"dora_marker\":\"3m\",\"tehais\":{tehais}}}"
        )
    }

    fn game_of(lines: &[String], id: &str) -> ParsedGame {
        parse_game(&(lines.join("\n") + "\n"), id).unwrap()
    }

    fn tpl(
        kind: &str,
        tile: Option<u8>,
        called: Option<u8>,
        consumed: &[u8],
        offset: Option<i8>,
        riichi: bool,
        meldref: bool,
    ) -> String {
        let consumed_json =
            consumed.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(",");
        let tile_json = tile.map(|t| t.to_string()).unwrap_or_else(|| "null".to_string());
        let called_json = called.map(|t| t.to_string()).unwrap_or_else(|| "null".to_string());
        let offset_json = offset.map(|o| o.to_string()).unwrap_or_else(|| "null".to_string());
        format!(
            "{{\"kind\":{kind:?},\"tile\":{tile_json},\"called_tile\":{called_json},\"consumed_tiles\":[{consumed_json}],\"source_offset\":{offset_json},\"declares_riichi\":{riichi},\"meld_ref_required\":{meldref}}}"
        )
    }

    fn table_of(entries: &[String]) -> ActionTable {
        ActionTable::load_unverified_json(&format!("{{\"payload\":{{\"actions\":[{}]}}}}", entries.join(",")))
            .unwrap()
    }

    /// Assemble a fully-declared envelope over synthetic entries: strict
    /// parse, digest in file order, declared digest spliced in.
    fn verified_table_of(entries: &[String]) -> Result<ActionTable, String> {
        let actions: Vec<serde_json::Value> = entries
            .iter()
            .map(|e| serde_json::from_str(e).expect("entry json"))
            .collect();
        let mut templates = Vec::new();
        for (id, entry) in actions.iter().enumerate() {
            templates.push(parse_template(entry, id)?);
        }
        let digest = table_content_digest(ACTION_TABLE_SCHEMA_VERSION, &templates);
        let text = format!(
            "{{\"artifact_type\":\"hydra2.action_table\",\"schema_version\":\"1.0.0\",\"compatibility\":\"exact\",\"payload\":{{\"schema_version\":\"1.0.0\",\"actions\":[{}],\"digest\":{digest:?}}}}}",
            entries.join(",")
        );
        ActionTable::load_json(&text)
    }

    #[test]
    fn verified_loader_accepts_sorted_and_binds_content_digest() {
        let entries = vec![
            tpl("pass", None, None, &[], None, false, false),
            tpl("discard", Some(44), None, &[], None, false, false),
        ];
        let table = verified_table_of(&entries).expect("sorted synthetic table loads");
        assert_eq!(table.len, 2);
        // Stored digest is the content digest over file order (what binds
        // `action_table_hash`), not a file-bytes hash.
        let actions: Vec<serde_json::Value> = entries
            .iter()
            .map(|e| serde_json::from_str(e).unwrap())
            .collect();
        let templates: Vec<Template> = actions
            .iter()
            .enumerate()
            .map(|(id, e)| parse_template(e, id).unwrap())
            .collect();
        assert_eq!(
            table.digest,
            table_content_digest(ACTION_TABLE_SCHEMA_VERSION, &templates)
        );
        assert_eq!(
            table.lookup("discard", Some(44), None, &[], None, false, false),
            Some(1)
        );
    }

    #[test]
    fn verified_loader_rejects_unsorted_and_duplicates() {
        // Generation order is pass(0) before discard(1): reversed file
        // order with a MATCHING declared digest still fails closed via the
        // rebuild check.
        let reversed = vec![
            tpl("discard", Some(44), None, &[], None, false, false),
            tpl("pass", None, None, &[], None, false, false),
        ];
        let err = verified_table_of(&reversed).unwrap_err();
        assert!(err.contains("generation order"), "unexpected: {err}");
        // Duplicate templates collapse the index: rejected.
        let duplicated = vec![
            tpl("pass", None, None, &[], None, false, false),
            tpl("pass", None, None, &[], None, false, false),
        ];
        let err = verified_table_of(&duplicated).unwrap_err();
        assert!(err.contains("duplicate"), "unexpected: {err}");
    }

    #[test]
    fn verified_loader_rejects_tamper_and_bad_envelopes() {
        let entries = vec![
            tpl("pass", None, None, &[], None, false, false),
            tpl("discard", Some(44), None, &[], None, false, false),
        ];
        // Rebuild the declared envelope text for mutation.
        let actions: Vec<serde_json::Value> = entries
            .iter()
            .map(|e| serde_json::from_str(e).unwrap())
            .collect();
        let templates: Vec<Template> = actions
            .iter()
            .enumerate()
            .map(|(id, e)| parse_template(e, id).unwrap())
            .collect();
        let digest = table_content_digest(ACTION_TABLE_SCHEMA_VERSION, &templates);
        let text = format!(
            "{{\"artifact_type\":\"hydra2.action_table\",\"schema_version\":\"1.0.0\",\"compatibility\":\"exact\",\"payload\":{{\"schema_version\":\"1.0.0\",\"actions\":[{}],\"digest\":{digest:?}}}}}",
            entries.join(",")
        );
        // One template byte flipped: declared-vs-recomputed mismatch.
        let tampered = text.replacen("\"tile\":44", "\"tile\":45", 1);
        assert_ne!(tampered, text);
        let err = ActionTable::load_json(&tampered).unwrap_err();
        assert!(err.contains("declared digest"), "unexpected: {err}");
        // Declared digest flipped instead: same mismatch, other direction.
        let zero_digest = format!("sha256:{}", "0".repeat(64));
        let tampered = text.replacen(&digest, &zero_digest, 1);
        assert!(ActionTable::load_json(&tampered).unwrap_err().contains("declared digest"));
    }
    fn kinds_of(rows: &[ReplayRow]) -> Vec<&str> {
        rows.iter().map(|r| r.chosen.kind_str()).collect()
    }

    fn assert_ids_exact(rows: &[ReplayRow], ids: &[u32]) {
        assert_eq!(rows.len(), ids.len(), "row count must match the id vector");
        for (row, want) in rows.iter().zip(ids) {
            assert_eq!(row.chosen_action_id, Some(*want), "chosen id for {}", row.decision_id);
            assert_eq!(row.chosen_unresolved, None, "no unresolved chosen for {}", row.decision_id);
        }
    }

    /// Minimal installed walker for direct `resolve_*` / `apply_*` pins: a
    fn walker_for<'g>(
        game: &'g ParsedGame,
        table: &'g ActionTable,
        tehais: &[Vec<String>],
    ) -> Walker<'g> {
        let track = KyokuTrack::install(&game.events, 0, 1, tehais, &game.game_id).unwrap();
        Walker {
            game,
            table,
            seq: 0,
            round_idx: 0,
            hand_index: 0,
            kyoku_ordinal: 0,
            track: Some(track),
            dealer: 0,
            histories: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            last_kind: None,
            last_discard: None,
            opened_by_discard: false,
            decided: false,
            terminal: false,
            rows: Vec::new(),
            scores: [25000; 4],
            version: crate::engine::EngineVersion::V1,
            kuikae_gate: None,
            acted: [false; 4],
            claims: 0,
            wall_digest: None,
        }
    }


    fn direct_tehais() -> Vec<Vec<String>> {
        [
            &["1m", "2m", "3m", "4m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p", "6p"][..],
            &["1s", "2s", "3s", "4s", "6s", "7s", "8s", "9s", "E", "S", "W", "N", "P"][..],
            &["F", "C", "E", "S", "W", "N", "P", "F", "C", "7p", "8p", "9p", "6s"][..],
            &["E", "S", "W", "N", "P", "F", "C", "5m", "5p", "7p", "8p", "9p", "2m"][..],
        ]
        .iter()
        .map(|hand| hand.iter().map(|t| t.to_string()).collect())
        .collect()
    }

    fn direct_game() -> ParsedGame {
        let seats: Vec<&[&str]> = vec![
            &["1m", "2m", "3m", "4m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p", "6p"],
            &["1s", "2s", "3s", "4s", "6s", "7s", "8s", "9s", "E", "S", "W", "N", "P"],
            &["F", "C", "E", "S", "W", "N", "P", "F", "C", "7p", "8p", "9p", "6s"],
            &["E", "S", "W", "N", "P", "F", "C", "5m", "5p", "7p", "8p", "9p", "2m"],
        ];
        let text = format!(
            "{{\"type\":\"start_game\"}}\n{}\n{{\"type\":\"end_game\"}}\n",
            start_kyoku_of(0, 1, &tehais_of(&seats))
        );
        parse_game(&text, "s3-direct").unwrap()
    }

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|t| t.to_string()).collect()
    }

    // -- resolve_claim_tiles: pool-first smallest-copy -----------------------

    #[test]
    fn resolve_takes_smallest_pool_copy_not_hand_position() {
        // Hand holds later copies of "2m" (pool [4,5,6,7]); the oracle's
        // copy-collapsed wall reports every "2m" as its base id, so the
        // resolved copy is pool-first (4), never the hand position (5).
        let game = direct_game();
        let table = ActionTable::load_unverified_json("{\"payload\":{\"actions\":[]}}").unwrap();
        let mut walker = walker_for(&game, &table, &direct_tehais());
        walker.track_mut("test").unwrap().hands[1] = vec![5, 6, 40];
        let picked = resolve_claim_tiles(&walker, 1, &strings(&["2m"]), 1, None).unwrap();
        assert_eq!(picked, vec![4]);
    }

    #[test]
    fn resolve_called_collision_swaps_to_unused_pool_copy() {
        // Pool-first picks [44, 45] for two "3p", but 44 is the called tile:
        // the collision swaps to the first unused pool copy (46).
        let game = direct_game();
        let table = ActionTable::load_unverified_json("{\"payload\":{\"actions\":[]}}").unwrap();
        let mut walker = walker_for(&game, &table, &direct_tehais());
        walker.track_mut("test").unwrap().hands[1] = vec![44, 45, 40];
        let picked =
            resolve_claim_tiles(&walker, 1, &strings(&["3p", "3p"]), 2, Some(44)).unwrap();
        assert_eq!(picked, vec![45, 46]);
    }

    #[test]
    fn resolve_chi_consumed_order_permutation_is_identity() {
        // The log may list consumed tiles in either order; resolution sorts
        // strings first, so both spellings resolve to the same copies.
        let game = direct_game();
        let table = ActionTable::load_unverified_json("{\"payload\":{\"actions\":[]}}").unwrap();
        let mut walker = walker_for(&game, &table, &direct_tehais());
        walker.track_mut("test").unwrap().hands[1] = vec![0, 8, 40];
        let forward = resolve_claim_tiles(&walker, 1, &strings(&["3m", "1m"]), 2, Some(48)).unwrap();
        let backward = resolve_claim_tiles(&walker, 1, &strings(&["1m", "3m"]), 2, Some(48)).unwrap();
        assert_eq!(forward, vec![0, 8]);
        assert_eq!(backward, vec![0, 8]);
    }

    #[test]
    fn resolve_red_pools_stay_disjoint() {
        // "5p" (pool [53,54,55]) and "5pr" (pool [52]) never share copies.
        let game = direct_game();
        let table = ActionTable::load_unverified_json("{\"payload\":{\"actions\":[]}}").unwrap();
        let mut walker = walker_for(&game, &table, &direct_tehais());
        walker.track_mut("test").unwrap().hands[0] = vec![52, 53, 40];
        let plain = resolve_claim_tiles(&walker, 0, &strings(&["5p"]), 1, None).unwrap();
        let aka = resolve_claim_tiles(&walker, 0, &strings(&["5pr"]), 1, None).unwrap();
        assert_eq!(plain, vec![53]);
        assert_eq!(aka, vec![52]);
    }

    #[test]
    fn resolve_ownership_and_arity_fail_closed() {
        let game = direct_game();
        let table = ActionTable::load_unverified_json("{\"payload\":{\"actions\":[]}}").unwrap();
        let mut walker = walker_for(&game, &table, &direct_tehais());
        walker.track_mut("test").unwrap().hands[1] = vec![40, 41, 42];
        // No tracked "E" copy left.
        let missing = resolve_claim_tiles(&walker, 1, &strings(&["E"]), 1, None).unwrap_err();
        assert_eq!(missing.code, "tile-conservation");
        // One logged string cannot satisfy a pair claim.
        let arity = resolve_claim_tiles(&walker, 1, &strings(&["2p"]), 2, None).unwrap_err();
        assert_eq!(arity.code, "tile-conservation");
    }

    // -- apply_discard: exact-then-drawn-preferred ---------------------------

    #[test]
    fn apply_discard_prefers_exact_copy_over_drawn_twin() {
        // Hand holds both the collapsed id (53) and the drawn twin (54) of
        // "5p": the exact copy leaves, the drawn twin stays.
        let game = direct_game();
        let table = ActionTable::load_unverified_json("{\"payload\":{\"actions\":[]}}").unwrap();
        let mut walker = walker_for(&game, &table, &direct_tehais());
        walker.track_mut("test").unwrap().hands[0] = vec![53, 54, 40];
        apply_discard(&mut walker, 0, 53, Some(54)).unwrap();
        let track = walker.track("test").unwrap();
        assert_eq!(track.hands[0], vec![54, 40]);
        assert_eq!(track.rivers[0], vec![53]);
        assert_eq!(track.exp_drawer, Some(0));
    }

    #[test]
    fn apply_discard_falls_back_to_drawn_preferred_string_match() {
        // Collapsed id (53) absent: the drawn copy rendering "5p" leaves.
        let game = direct_game();
        let table = ActionTable::load_unverified_json("{\"payload\":{\"actions\":[]}}").unwrap();
        let mut walker = walker_for(&game, &table, &direct_tehais());
        walker.track_mut("test").unwrap().hands[0] = vec![54, 55, 40];
        apply_discard(&mut walker, 0, 53, Some(54)).unwrap();
        let track = walker.track("test").unwrap();
        assert_eq!(track.hands[0], vec![55, 40]);
        assert_eq!(track.rivers[0], vec![53]);
    }

    #[test]
    fn apply_discard_keeps_red_and_normal_copies_distinct() {
        let game = direct_game();
        let table = ActionTable::load_unverified_json("{\"payload\":{\"actions\":[]}}").unwrap();
        let mut walker = walker_for(&game, &table, &direct_tehais());
        // Normal "5p" discard removes 53 and keeps the aka 52.
        walker.track_mut("test").unwrap().hands[0] = vec![52, 53, 40];
        apply_discard(&mut walker, 0, 53, None).unwrap();
        assert_eq!(walker.track("test").unwrap().hands[0], vec![52, 40]);
        // Red "5pr" discard removes exactly the aka.
        walker.track_mut("test").unwrap().hands[0] = vec![52, 53, 40];
        apply_discard(&mut walker, 0, 52, Some(52)).unwrap();
        assert_eq!(walker.track("test").unwrap().hands[0], vec![53, 40]);
    }

    // -- end-to-end copy-identity battery ------------------------------------

    #[test]
    fn red_tsumogiri_and_normal_discard_report_collapsed_ids() {
        let lines = vec![
            "{\"type\":\"start_game\"}".to_string(),
            start_kyoku_of(0, 1, &cycling_tehais()),
            "{\"type\":\"tsumo\",\"actor\":0,\"pai\":\"5pr\"}".to_string(),
            "{\"type\":\"dahai\",\"actor\":0,\"pai\":\"5pr\",\"tsumogiri\":true}".to_string(),
            format!("{{\"type\":\"tsumo\",\"actor\":1,\"pai\":{:?}}}", tile_at(53)),
            "{\"type\":\"dahai\",\"actor\":1,\"pai\":\"5p\",\"tsumogiri\":false}".to_string(),
            "{\"type\":\"ryukyoku\",\"reason\":\"exhaustive_draw\",\"deltas\":[0,0,0,0]}".to_string(),
            "{\"type\":\"end_kyoku\"}".to_string(),
            "{\"type\":\"end_game\"}".to_string(),
        ];
        let game = game_of(&lines, "s3-red");
        assert_eq!(count_row_decisions(&game).unwrap(), 2);
        let table = table_of(&[tpl("tsumogiri", Some(52), None, &[], None, false, false), tpl("discard", Some(53), None, &[], None, false, false)]);
        let rows = walk_game(&game, &table).unwrap();
        assert_eq!(kinds_of(&rows), vec!["tsumogiri", "discard"]);
        assert_eq!(rows[0].chosen, ChosenAction::Tsumogiri { tile: 52 });
        assert_eq!(rows[0].phase, "draw_decision");
        // Plain "5p" reports the collapsed second copy (53), never the aka.
        assert_eq!(rows[1].chosen, ChosenAction::Discard { tile: 53 });
        assert_ids_exact(&rows, &[0, 1]);
    }

    #[test]
    fn pon_then_kakan_resolves_missing_pon_copy() {
        let tehais = tehais_of(&[
            &["3p", "1m", "2m", "4m", "6m", "7m", "8m", "9m", "1p", "2p", "4p", "6p", "7p"],
            &["3p", "3p", "3p", "1s", "2s", "3s", "4s", "6s", "7s", "8s", "9s", "E", "S"],
            &["W", "N", "P", "F", "C", "1m", "2m", "4m", "6m", "7m", "8m", "9m", "1p"],
            &["E", "S", "W", "N", "P", "F", "C", "1s", "2s", "4s", "6s", "7s", "8s"],
        ]);
        let lines = vec![
            "{\"type\":\"start_game\"}".to_string(),
            start_kyoku_of(0, 1, &tehais),
            "{\"type\":\"tsumo\",\"actor\":0,\"pai\":\"9m\"}".to_string(),
            "{\"type\":\"dahai\",\"actor\":0,\"pai\":\"3p\",\"tsumogiri\":false}".to_string(),
            "{\"type\":\"pon\",\"actor\":1,\"target\":0,\"pai\":\"3p\",\"consumed\":[\"3p\",\"3p\"]}".to_string(),
            "{\"type\":\"tsumo\",\"actor\":1,\"pai\":\"7m\"}".to_string(),
            "{\"type\":\"kakan\",\"actor\":1,\"pai\":\"3p\"}".to_string(),
            "{\"type\":\"tsumo\",\"actor\":1,\"pai\":\"8m\"}".to_string(),
            "{\"type\":\"dahai\",\"actor\":1,\"pai\":\"8m\",\"tsumogiri\":true}".to_string(),
            "{\"type\":\"tsumo\",\"actor\":2,\"pai\":\"9s\"}".to_string(),
            "{\"type\":\"dahai\",\"actor\":2,\"pai\":\"9s\",\"tsumogiri\":true}".to_string(),
            "{\"type\":\"ryukyoku\",\"reason\":\"exhaustive_draw\",\"deltas\":[0,0,0,0]}".to_string(),
            "{\"type\":\"end_kyoku\"}".to_string(),
            "{\"type\":\"end_game\"}".to_string(),
        ];
        let game = game_of(&lines, "s3-pon-kakan");
        assert_eq!(count_row_decisions(&game).unwrap(), 5);
        let table = table_of(&[
            tpl("discard", Some(44), None, &[], None, false, false),
            tpl("pon", None, Some(44), &[45, 46], Some(-1), false, false),
            tpl("kakan", Some(47), None, &[], None, false, true),
            tpl("tsumogiri", Some(28), None, &[], None, false, false),
            tpl("tsumogiri", Some(104), None, &[], None, false, false),
        ]);
        let rows = walk_game(&game, &table).unwrap();
        assert_eq!(kinds_of(&rows), vec!["discard", "pon", "kakan", "tsumogiri", "tsumogiri"]);
        assert_eq!(rows[0].chosen, ChosenAction::Discard { tile: 44 });
        // Called 44 collides with pool-first 44: the consumed pair swaps to
        // the unused copies, and the kakan adds the one copy missing from
        // the prior pon triple.
        assert_eq!(
            rows[1].chosen,
            ChosenAction::Pon { called: 44, consumed: vec![45, 46], source: 0 }
        );
        assert_eq!(rows[1].phase, "discard_response");
        assert_eq!(rows[2].chosen, ChosenAction::Kakan { tile: 47 });
        assert_eq!(rows[3].chosen, ChosenAction::Tsumogiri { tile: 28 });
        // Seat 2 never held collapsed 104: the drawn copy (105) leaves while
        // the row still reports the collapsed id.
        assert_eq!(rows[4].chosen, ChosenAction::Tsumogiri { tile: 104 });
        assert_ids_exact(&rows, &[0, 1, 2, 3, 4]);
    }

    #[test]
    fn chi_consumed_permutation_resolves_sorted_pair() {
        let tehais = tehais_of(&[
            &["4p", "1m", "2m", "3m", "6m", "7m", "8m", "9m", "1p", "2p", "6p", "7p", "8p"],
            &["3p", "5p", "1s", "2s", "3s", "4s", "6s", "7s", "8s", "9s", "E", "S", "W"],
            &["N", "P", "F", "C", "1m", "2m", "3m", "6m", "7m", "8m", "9m", "1p", "2p"],
            &["E", "S", "W", "N", "P", "F", "C", "1s", "2s", "4s", "6s", "7s", "9s"],
        ]);
        let lines = vec![
            "{\"type\":\"start_game\"}".to_string(),
            start_kyoku_of(0, 1, &tehais),
            "{\"type\":\"tsumo\",\"actor\":0,\"pai\":\"9m\"}".to_string(),
            "{\"type\":\"dahai\",\"actor\":0,\"pai\":\"4p\",\"tsumogiri\":false}".to_string(),
            "{\"type\":\"chi\",\"actor\":1,\"target\":0,\"pai\":\"4p\",\"consumed\":[\"5p\",\"3p\"]}".to_string(),
            "{\"type\":\"tsumo\",\"actor\":1,\"pai\":\"1s\"}".to_string(),
            "{\"type\":\"dahai\",\"actor\":1,\"pai\":\"1s\",\"tsumogiri\":true}".to_string(),
            "{\"type\":\"ryukyoku\",\"reason\":\"exhaustive_draw\",\"deltas\":[0,0,0,0]}".to_string(),
            "{\"type\":\"end_kyoku\"}".to_string(),
            "{\"type\":\"end_game\"}".to_string(),
        ];
        let game = game_of(&lines, "s3-chi");
        assert_eq!(count_row_decisions(&game).unwrap(), 3);
        let table = table_of(&[
            tpl("discard", Some(48), None, &[], None, false, false),
            tpl("chi", None, Some(48), &[44, 53], Some(-1), false, false),
            tpl("tsumogiri", Some(72), None, &[], None, false, false),
        ]);
        let rows = walk_game(&game, &table).unwrap();
        assert_eq!(kinds_of(&rows), vec!["discard", "chi", "tsumogiri"]);
        // Log order ["5p","3p"] still resolves to the sorted smallest pair.
        assert_eq!(rows[1].chosen, ChosenAction::Chi { called: 48, consumed: vec![44, 53] });
        assert_eq!(rows[1].phase, "discard_response");
        assert_ids_exact(&rows, &[0, 1, 2]);
    }

    #[test]
    fn ankan_strings_resolve_to_one_physical_block() {
        let tehais = tehais_of(&[
            &["7s", "7s", "7s", "1m", "2m", "3m", "4m", "6m", "8m", "9m", "1p", "2p", "3p"],
            &["1s", "2s", "3s", "4s", "6s", "8s", "9s", "E", "S", "W", "N", "P", "F"],
            &["C", "E", "S", "W", "N", "P", "F", "C", "1m", "2m", "3m", "4m", "6m"],
            &["8m", "9m", "1p", "2p", "3p", "4p", "6p", "7p", "8p", "9p", "1s", "2s", "5m"],
        ]);
        let lines = vec![
            "{\"type\":\"start_game\"}".to_string(),
            start_kyoku_of(0, 1, &tehais),
            "{\"type\":\"tsumo\",\"actor\":0,\"pai\":\"7s\"}".to_string(),
            "{\"type\":\"ankan\",\"actor\":0,\"consumed\":[\"7s\",\"7s\",\"7s\",\"7s\"]}".to_string(),
            "{\"type\":\"tsumo\",\"actor\":0,\"pai\":\"5s\"}".to_string(),
            "{\"type\":\"dahai\",\"actor\":0,\"pai\":\"5s\",\"tsumogiri\":true}".to_string(),
            "{\"type\":\"ryukyoku\",\"reason\":\"exhaustive_draw\",\"deltas\":[0,0,0,0]}".to_string(),
            "{\"type\":\"end_kyoku\"}".to_string(),
            "{\"type\":\"end_game\"}".to_string(),
        ];
        let game = game_of(&lines, "s3-ankan");
        assert_eq!(count_row_decisions(&game).unwrap(), 2);
        let table = table_of(&[
            tpl("ankan", None, None, &[96, 97, 98, 99], None, false, false),
            tpl("tsumogiri", Some(89), None, &[], None, false, false),
        ]);
        let rows = walk_game(&game, &table).unwrap();
        assert_eq!(kinds_of(&rows), vec!["ankan", "tsumogiri"]);
        assert_eq!(rows[0].chosen, ChosenAction::Ankan { consumed: vec![96, 97, 98, 99] });
        assert_eq!(rows[0].phase, "draw_decision");
        // Drawn 5s is the pool-first take (no 5s anywhere in tehais), so
        // the tsumogiri twin is the true take itself.
        assert_eq!(rows[1].chosen, ChosenAction::Tsumogiri { tile: 89 });
        assert_ids_exact(&rows, &[0, 1]);
    }

    #[test]
    fn daiminkan_claim_resolves_called_collision_triple() {
        let tehais = tehais_of(&[
            &["6s", "1m", "2m", "3m", "4m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"],
            &["6s", "6s", "6s", "1s", "2s", "3s", "4s", "7s", "8s", "9s", "E", "S", "W"],
            &["N", "P", "F", "C", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p", "6p"],
            &["E", "S", "W", "N", "P", "F", "C", "7p", "8p", "9p", "1s", "2s", "5s"],
        ]);
        let lines = vec![
            "{\"type\":\"start_game\"}".to_string(),
            start_kyoku_of(0, 1, &tehais),
            "{\"type\":\"tsumo\",\"actor\":0,\"pai\":\"9m\"}".to_string(),
            "{\"type\":\"dahai\",\"actor\":0,\"pai\":\"6s\",\"tsumogiri\":false}".to_string(),
            "{\"type\":\"daiminkan\",\"actor\":1,\"target\":0,\"pai\":\"6s\",\"consumed\":[\"6s\",\"6s\",\"6s\"]}".to_string(),
            "{\"type\":\"tsumo\",\"actor\":1,\"pai\":\"1m\"}".to_string(),
            "{\"type\":\"dahai\",\"actor\":1,\"pai\":\"1m\",\"tsumogiri\":true}".to_string(),
            "{\"type\":\"ryukyoku\",\"reason\":\"exhaustive_draw\",\"deltas\":[0,0,0,0]}".to_string(),
            "{\"type\":\"end_kyoku\"}".to_string(),
            "{\"type\":\"end_game\"}".to_string(),
        ];
        let game = game_of(&lines, "s3-daiminkan");
        assert_eq!(count_row_decisions(&game).unwrap(), 3);
        let table = table_of(&[
            tpl("discard", Some(92), None, &[], None, false, false),
            tpl("daiminkan", None, Some(92), &[93, 94, 95], Some(-1), false, false),
            tpl("tsumogiri", Some(0), None, &[], None, false, false),
        ]);
        let rows = walk_game(&game, &table).unwrap();
        assert_eq!(kinds_of(&rows), vec!["discard", "daiminkan", "tsumogiri"]);
        assert_eq!(
            rows[1].chosen,
            ChosenAction::Daiminkan { called: 92, consumed: vec![93, 94, 95], source: 0 }
        );
        assert_ids_exact(&rows, &[0, 1, 2]);
    }

    // -- do_* dispatch order --------------------------------------------------

    #[test]
    fn reach_collapses_declaration_into_one_riichi_row() {
        // Seat 0 holds a genuine tenpai (123m/456m/789m + 1p pair + 1s/6m
        // floaters): discarding either 6m keeps the shanpon wait, so the
        // logged 6m declaration is a true engine candidate. Seat 1 holds
        // no 2s, so its drawn 2s is the pool-first take (tsumogiri twin).
        let tehais = tehais_of(&[
            &["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "1s", "6m"],
            &["3s", "4s", "5s", "6s", "7s", "8s", "9s", "E", "S", "W", "N", "P", "F"],
            &["9s", "E", "S", "W", "N", "P", "F", "C", "1m", "2m", "3m", "4m", "5m"],
            &["6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p"],
        ]);
        let lines = vec![
            "{\"type\":\"start_game\"}".to_string(),
            start_kyoku_of(0, 1, &tehais),
            format!("{{\"type\":\"tsumo\",\"actor\":0,\"pai\":{:?}}}", tile_at(52)),
            "{\"type\":\"reach\",\"actor\":0}".to_string(),
            "{\"type\":\"dahai\",\"actor\":0,\"pai\":\"6m\",\"tsumogiri\":true}".to_string(),
            format!("{{\"type\":\"tsumo\",\"actor\":1,\"pai\":{:?}}}", tile_at(53)),
            format!("{{\"type\":\"dahai\",\"actor\":1,\"pai\":{:?},\"tsumogiri\":true}}", tile_at(53)),
            "{\"type\":\"ryukyoku\",\"reason\":\"exhaustive_draw\",\"deltas\":[0,0,0,0]}".to_string(),
            "{\"type\":\"end_kyoku\"}".to_string(),
            "{\"type\":\"end_game\"}".to_string(),
        ];
        let game = game_of(&lines, "s3-reach");
        // The reach plus its declaration dahai count (and walk) as exactly
        // one row: no orphan dahai row, no zero-row reach.
        assert_eq!(count_row_decisions(&game).unwrap(), 2);
        let table = table_of(&[
            tpl("riichi_discard", Some(20), None, &[], None, true, false),
            tpl("tsumogiri", Some(76), None, &[], None, false, false),
        ]);
        let rows = walk_game(&game, &table).unwrap();
        assert_eq!(kinds_of(&rows), vec!["riichi_discard", "tsumogiri"]);
        // The declaration reports the collapsed id, exactly like yielded steps.
        assert_eq!(rows[0].chosen, ChosenAction::RiichiDiscard { tile: 20 });
        assert_eq!(rows[1].chosen, ChosenAction::Tsumogiri { tile: 76 });
        assert_ids_exact(&rows, &[0, 1]);
    }

    #[test]
    fn ron_and_tsumo_hora_close_dispatch() {
        // Ron answers the live discard offer from the discard_response phase.
        // Seat 1 holds 11p/234p/567p/88p/999p: the logged 1p discard
        // completes a genuine win (shape gate), and its 1p pair opens the
        // window by pon counts.
        let ron_tehais = tehais_of(&[
            &["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"],
            &["1p", "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "8p", "9p", "9p", "9p"],
            &["9s", "E", "S", "W", "N", "P", "F", "C", "1m", "2m", "3m", "4m", "5m"],
            &["6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p"],
        ]);
        let ron_lines = vec![
            "{\"type\":\"start_game\"}".to_string(),
            start_kyoku_of(0, 1, &ron_tehais),
            format!("{{\"type\":\"tsumo\",\"actor\":0,\"pai\":{:?}}}", tile_at(52)),
            "{\"type\":\"dahai\",\"actor\":0,\"pai\":\"1p\",\"tsumogiri\":false}".to_string(),
            "{\"type\":\"hora\",\"actor\":1,\"target\":0,\"deltas\":[-8000,12000,-2000,-2000]}".to_string(),
            "{\"type\":\"end_kyoku\"}".to_string(),
            "{\"type\":\"end_game\"}".to_string(),
        ];
        let ron_game = game_of(&ron_lines, "s3-ron");
        assert_eq!(count_row_decisions(&ron_game).unwrap(), 2);
        let ron_table = table_of(&[
            tpl("discard", Some(36), None, &[], None, false, false),
            tpl("ron", Some(36), None, &[], Some(-1), false, false),
        ]);
        let ron_rows = walk_game(&ron_game, &ron_table).unwrap();
        assert_eq!(kinds_of(&ron_rows), vec!["discard", "ron"]);
        assert_eq!(ron_rows[1].chosen, ChosenAction::Ron { tile: 36, source: 0 });
        assert_eq!(ron_rows[1].phase, "discard_response");
        assert_ids_exact(&ron_rows, &[0, 1]);
        // Tsumo answers the seat's own draw from the draw_decision phase and
        // reports the string-canonical first copy of the drawn string. Seat
        // 0 holds 111m/234m/567m/23s/99m: the logged first-take 1s draw
        // completes a genuine win (shape gate) under the drained pool
        // discipline (drawn take-index below the hand count, as on every
        // corpus draw).
        let tsumo_tehais = tehais_of(&[
            &["1m", "1m", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "2s", "3s", "9m", "9m"],
            &["3p", "2s", "3s", "4s", "6s", "7s", "8s", "9s", "E", "S", "W", "N", "P"],
            &["9s", "E", "S", "W", "N", "P", "F", "C", "1m", "2m", "3m", "4m", "5m"],
            &["6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p"],
        ]);
        let tsumo_lines = vec![
            "{\"type\":\"start_game\"}".to_string(),
            start_kyoku_of(0, 1, &tsumo_tehais),
            format!("{{\"type\":\"tsumo\",\"actor\":0,\"pai\":{:?}}}", tile_at(52)),
            "{\"type\":\"hora\",\"actor\":0,\"target\":0,\"tsumo\":true,\"deltas\":[12000,-2000,-4000,-6000]}".to_string(),
            "{\"type\":\"end_kyoku\"}".to_string(),
            "{\"type\":\"end_game\"}".to_string(),
        ];
        let tsumo_game = game_of(&tsumo_lines, "s3-tsumo-hora");
        assert_eq!(count_row_decisions(&tsumo_game).unwrap(), 1);
        let tsumo_table =
            table_of(&[tpl("tsumo", Some(72), None, &[], None, false, false)]);
        let tsumo_rows = walk_game(&tsumo_game, &tsumo_table).unwrap();
        assert_eq!(kinds_of(&tsumo_rows), vec!["tsumo"]);
        assert_eq!(tsumo_rows[0].chosen, ChosenAction::Tsumo { tile: 72 });
        assert_eq!(tsumo_rows[0].phase, "draw_decision");
        assert_ids_exact(&tsumo_rows, &[0]);
    }

    #[test]
    fn decided_kyoku_rejects_late_rows() {
        // do_ryukyoku decides the kyoku (emitting no row itself); any later
        // row event in the same kyoku fails closed instead of appending.
        let lines = vec![
            "{\"type\":\"start_game\"}".to_string(),
            start_kyoku_of(0, 1, &cycling_tehais()),
            format!("{{\"type\":\"tsumo\",\"actor\":0,\"pai\":{:?}}}", tile_at(52)),
            format!("{{\"type\":\"dahai\",\"actor\":0,\"pai\":{:?},\"tsumogiri\":true}}", tile_at(52)),
            "{\"type\":\"ryukyoku\",\"reason\":\"exhaustive_draw\",\"deltas\":[0,0,0,0]}".to_string(),
            format!("{{\"type\":\"dahai\",\"actor\":1,\"pai\":{:?},\"tsumogiri\":true}}", tile_at(53)),
            "{\"type\":\"end_game\"}".to_string(),
        ];
        let game = game_of(&lines, "s3-decided");
        let table = ActionTable::load_unverified_json("{\"payload\":{\"actions\":[]}}").unwrap();
        let err = walk_game(&game, &table).unwrap_err();
        assert_eq!(err.code, "turn-order");
    }
}
