//! rl/core: native four-player hanchan driver with dense hand ordinals.
//!
//! The game owns one cohort member: immutable context validated at open,
//! explicit hand-index ownership (the caller assigns each ordinal once, dense
//! `0..N-1`), engine seeds derived from the schedule and the ordinal, and a
//! fresh engine state per hand. Hand zero uses the schedule tiles; later hands
//! use continuation walls; the engine's own auto-advanced round is discarded
//! at every boundary. New engine log records are parsed once into the typed
//! arena with game-owned per-seat cursors, using consume-after-translate
//! discipline. Claim windows buffer responders and step once with the combined
//! map; a missing responder means pass; multi-winner runs merge; riichi needs
//! a declaration step plus a discard step; added-quad tiles are captured at
//! apply time. Draws come from the step-time queue; discards resolve from the
//! river cursor, then a single-tile hand diff, then the mjai-string fallback.
//! Window resolution stages envelopes, heads them with a resolution marker
//! when the window opened from a discard, and propagates hand carries found
//! inside the staged batch. Any engine error after any step fails closed.
//! Terminal output carries final scores, resolved ranks, and point deltas,
//! with placement utilities from `rank_values[rank - 1]`. No Python, no opaque
//! RNG replay (replay is canonical action prefixes).
//!
//! Layout: this facade keeps the public surface, the shared private types,
//! the small free helpers, and the open plus identity methods. Behavior lives
//! in the children, which see this module's items via `super::` and expose
//! cross-child helpers as `pub(super)` only:
//! - `row`: captured row serialization (`to_json`/`from_json`) plus row
//!   construction (`capture_row` with transition provenance).
//! - `driver`: hand lifecycle (`open_hand`, `reopen`, `drain`, `detect`),
//!   legal views, observations, applies, replay, and the envelope and error
//!   helpers shared with translation.
//! - `translate`: mjai batch translation to typed envelopes plus carry facts.
//! - `tiles`: exact tile resolution backing translation.
//! - `tests`: native behavior tests.

mod driver;
mod row;
mod tiles;
mod translate;

#[cfg(test)]
mod tests;

use std::collections::{HashMap, VecDeque};

use hydra_feed::census::ActionKind;
use riichienv_core::action::Action as CoreAction;
use riichienv_core::rule::GameRule;
use riichienv_core::state::GameState;
use sha2::{Digest, Sha256};

pub use super::actions::CanonicalAction;
use super::events::{EventArena, TypedEvent, Visibility};
use super::observation::ObservationError;

/// Four-player hanchan mode id (`GameType.YON_HANCHAN` = 2).
pub const GAME_MODE_YON_HANCHAN: u8 = 2;
/// Tile domain top.
pub const TILE_ID_MAX: u8 = 135;

/// Native failure (fail-closed text).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreError(pub String);

impl std::fmt::Display for CoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "rl core: {}", self.0)
    }
}

impl std::error::Error for CoreError {}

impl From<ObservationError> for CoreError {
    fn from(e: ObservationError) -> Self {
        CoreError(e.0)
    }
}

impl From<super::actions::ActionError> for CoreError {
    fn from(e: super::actions::ActionError) -> Self {
        CoreError(e.0)
    }
}

impl From<super::events::EventError> for CoreError {
    fn from(e: super::events::EventError) -> Self {
        CoreError(e.0)
    }
}

/// Only cyclic rotations keep canonical adjacency equal to engine turn order:
/// seat `s` must map to `(perm[0] + s) % 4`.
pub fn check_cyclic_rotation(perm: [u8; 4]) -> Result<[usize; 4], CoreError> {
    let mut sorted = perm.to_vec();
    sorted.sort_unstable();
    if sorted != vec![0, 1, 2, 3] {
        return Err(CoreError("seat_permutation must arrange all four seats".to_string()));
    }
    if (0..4).any(|seat| perm[seat] != (perm[0] + seat as u8) % 4) {
        return Err(CoreError(
            "non-cyclic seat_permutation; only rotations of (0,1,2,3) are supported".to_string(),
        ));
    }
    let mut inv = [0usize; 4];
    for (seat, pid) in perm.iter().enumerate() {
        inv[*pid as usize] = seat;
    }
    Ok(inv)
}

/// Immutable game context, validated once at open before any hand starts.
#[derive(Debug, Clone)]
pub struct GameContext {
    pub rules_id: String,
    pub rules_hash: String,
    pub action_table_hash: String,
    pub event_schema_hash: String,
    pub observation_schema_hash: String,
    pub packet_boundary_hash: String,
    pub adapter_version: String,
    pub schedule_id: String,
    pub schedule_digest: String,
    pub schedule_tiles: [u8; 136],
    pub seat_perm: [u8; 4],
    pub rank_values: [f64; 4],
    pub behavior_ckpt: [u8; 32],
}

impl GameContext {
    pub fn validate(&self) -> Result<(), CoreError> {
        if self.rules_id.is_empty() || self.schedule_id.is_empty() {
            return Err(CoreError("rules/schedule ids must be non-empty".to_string()));
        }
        if self.schedule_digest.is_empty() {
            return Err(CoreError("schedule digest must be non-empty".to_string()));
        }
        for v in self.rank_values {
            if !v.is_finite() {
                return Err(CoreError("rank_values must be finite".to_string()));
            }
        }
        check_cyclic_rotation(self.seat_perm)?;
        if self.schedule_tiles.iter().any(|t| *t > TILE_ID_MAX) {
            return Err(CoreError("schedule_tiles must hold physical ids 0..=135".to_string()));
        }
        Ok(())
    }
    pub fn digest_noprefix(&self) -> String {
        self.schedule_digest.trim_start_matches("sha256:").to_string()
    }
}

/// One settlement fact: hand score deltas with the payer and the winners.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SettlementFact {
    pub kind: String,
    pub deltas: [i32; 4],
    pub payer: Option<u8>,
    pub winners: Vec<u8>,
}

/// Terminal outcome: final scores, ranks, point deltas, and placement utilities.
#[derive(Debug, Clone)]
pub struct TerminalOutcome {
    pub final_scores: [i32; 4],
    pub ranks: [u8; 4],
    pub point_deltas: [i32; 4],
    pub utility_g4: [f64; 4],
}

/// Compact captured row: one committed decision's facts for the artifact tape.
/// The tape stores these names plus its terminal tail; `row_id` stays zeroed
/// at capture and the seal fills it with the hash of the canonical row bytes
/// after terminal binding and sorting; `return_g4` and `raw_advantage_f32`
/// are `None` at capture and frozen later at terminal binding.
#[derive(Debug, Clone)]
pub struct GameRow {
    pub wall_id: String,
    pub game_id: String,
    pub round_id: String,
    pub trajectory_step: u64,
    pub actor: u8,
    pub decision_id: String,
    pub sequence: u64,
    pub row_id: [u8; 32],
    pub round_index: u64,
    pub round_wind: i64,
    pub honba: u32,
    pub kyotaku: u32,
    pub dealer: u8,
    pub scores: [i32; 4],
    pub live_remaining: u32,
    pub hand_number: u32,
    pub turn_actor: u8,
    pub phase_id: u8,
    pub concealed: Vec<u8>,
    pub drawn: Option<u8>,
    pub dora5: [i32; 5],
    pub ippatsu_bits: u8,
    pub furiten_id: u8,
    pub can_tsumo: bool,
    pub can_riichi: bool,
    pub legal_ids: Vec<u32>,
    pub legal_mask_bits: Vec<u8>,
    pub selected: u32,
    pub selected_logp: f32,
    pub value4: [f32; 4],
    pub cursor_before: u64,
    pub cursor_after: u64,
    pub obs_hash: [u8; 32],
    pub event_prefix_len: u32,
    pub derivation: [u8; 32],
    pub action_table_hash: [u8; 32],
    pub action_table_hash_hex: String,
    pub behavior_ckpt: [u8; 32],
    pub transition_provenance: [u8; 32],
    pub rules_id: String,
    pub rules_hash: String,
    pub event_schema_hash: String,
    pub observation_schema_hash: String,
    pub packet_boundary_hash: String,
    pub return_g4: Option<[f64; 4]>,
    pub raw_advantage_f32: Option<f32>,
}

/// Pending decision after a drain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PendingDecision {
    Draw { seat: u8 },
    Window { pending: Vec<u8> },
}

struct BufferedDecision {
    engine: CoreAction,
    canonical_id: u32,
}

struct Stamp {
    id: u32,
    actor: u8,
    kind: ActionKind,
}

/// Native full-game simulator (one cohort member).
pub struct NativeGame {
    ctx: GameContext,
    inv: [usize; 4],
    game_id: String,
    state: GameState,
    cursor: usize,
    hand_index: u64,
    seq: u64,
    arena: EventArena,
    staging: Option<Vec<TypedEvent>>,
    mode: Option<GameMode>,
    decision_seat: Option<u8>,
    pending: Vec<u8>,
    buffered: HashMap<u8, BufferedDecision>,
    views: HashMap<u8, (Vec<(u32, CanonicalAction)>, Vec<bool>)>,
    last_discard: (Option<u8>, Option<u8>),
    dahai_cursor: [usize; 4],
    draw_queue: VecDeque<u8>,
    pre_step_hands: HashMap<u8, Vec<u8>>,
    kakan_added: HashMap<u8, u8>,
    /// Per-seat drawn tile retained for the hand (oracle rule: every
    /// `draw_tile` envelope sets its seat; discards never clear it; only a
    /// new hand resets it). The live engine clears `drawn_tile` on discard,
    /// so reading engine state here would blank responder observations.
    drawn_cache: [Option<u8>; 4],
    window_offered_ids: Vec<u32>,
    window_opened_by_discard: bool,
    /// Arena length when the current window was freshly detected (identity
    /// for re-detects: answers arriving across lanes must not wipe it).
    window_arena_len: usize,
    stamp: Option<Stamp>,
    ippatsu: [bool; 4],
    riichi_states: [String; 4],
    kan_count: u32,
    applied: Vec<CanonicalAction>,
    settlements: Vec<SettlementFact>,
    starting_scores: [i32; 4],
    terminal: bool,
    outcome: Option<TerminalOutcome>,
    captured: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum GameMode {
    Draw,
    Window,
}

impl NativeGame {
    /// Open a cohort game: validate context, derive game id, open hand 0,
    /// emit `game_start`, drain to the first decision.
    pub fn open(ctx: GameContext, starting_points: i32) -> Result<Self, CoreError> {
        ctx.validate()?;
        let inv = check_cyclic_rotation(ctx.seat_perm)?;
        let game_id = derive_game_id(&ctx)?;
        let starting = [starting_points; 4];
        let seed_tiles = ctx.schedule_tiles;
        let perm0 = ctx.seat_perm[0];
        let mut game = Self {
            inv,
            game_id,
            state: GameState::new(GAME_MODE_YON_HANCHAN, false, Some(0), 0, GameRule::default_tenhou()),
            cursor: 0,
            hand_index: 0,
            seq: 0,
            arena: EventArena::new(),
            staging: None,
            mode: None,
            decision_seat: None,
            pending: vec![],
            buffered: HashMap::new(),
            views: HashMap::new(),
            last_discard: (None, None),
            dahai_cursor: [0; 4],
            drawn_cache: [None; 4],
            draw_queue: VecDeque::new(),
            pre_step_hands: HashMap::new(),
            kakan_added: HashMap::new(),
            window_offered_ids: vec![],
            window_opened_by_discard: false,
            window_arena_len: 0,
            stamp: None,
            ippatsu: [false; 4],
            riichi_states: [
                "none".to_string(),
                "none".to_string(),
                "none".to_string(),
                "none".to_string(),
            ],
            kan_count: 0,
            applied: vec![],
            settlements: vec![],
            starting_scores: starting,
            terminal: false,
            outcome: None,
            captured: 0,
            ctx,
        };
        game.open_hand(0, &seed_tiles, perm0, 0, 0, engine_order_of(starting, game.inv), 0)?;
        let game_start = game.make_envelope(
            "game_start",
            None,
            Visibility::Public,
            vec![0, 1, 2, 3],
            TypedPayload {
                round_index: Some(0),
                scores: Some(starting),
                ..TypedPayload::empty()
            },
            vec![],
        )?;
        game.emit(game_start)?;
        game.drain()?;
        Ok(game)
    }

    // -- identity ---------------------------------------------------------

    pub fn game_id(&self) -> &str {
        &self.game_id
    }

    pub fn hand_index(&self) -> u64 {
        self.hand_index
    }

    pub fn is_terminal(&self) -> bool {
        self.terminal
    }

    pub fn outcome(&self) -> Option<&TerminalOutcome> {
        self.outcome.as_ref()
    }

    pub fn applied_prefix(&self) -> &[CanonicalAction] {
        &self.applied
    }

    /// Per-hand settlement facts (engine deltas as recorded; control
    /// snapshots and independent audits consume this ledger).
    pub fn settlements(&self) -> &[SettlementFact] {
        &self.settlements
    }

    pub fn captured_count(&self) -> u64 {
        self.captured
    }

    /// Reinstate the row counter on resume (sealed rows already exist;
    /// replay rebuilds state only, so the counter is restored explicitly to
    /// keep future `trajectory_step`/digests continuous).
    pub fn set_captured(&mut self, n: u64) {
        self.captured = n;
    }

    /// Whether the core recorded an engine error (worker poison check).
    pub fn state_last_error(&self) -> bool {
        self.state.last_error.is_some()
    }

    pub fn event_len(&self) -> usize {
        self.arena.len()
    }

    pub fn arena(&self) -> &EventArena {
        &self.arena
    }

    /// Move the event arena out for tape retention at seal (the game is
    /// terminal then; rows ref prefixes by length into the moved arena).
    pub fn take_arena(&mut self) -> EventArena {
        std::mem::take(&mut self.arena)
    }

    /// Compact state digest for snapshot/restore verification.
    pub fn state_digest(&self) -> [u8; 32] {
        let mut hands: Vec<Vec<u8>> = Vec::with_capacity(4);
        for pid in 0..4 {
            let mut h = self.state.players[pid].hand.clone();
            h.sort_unstable();
            hands.push(h);
        }
        let doc = serde_json::json!({
            "ordinal": self.hand_index,
            "oya": self.state.oya,
            "honba": self.state.honba,
            "kyotaku": self.state.riichi_sticks,
            "scores": [self.state.players[0].score, self.state.players[1].score,
                       self.state.players[2].score, self.state.players[3].score],
            "hands": hands,
            "phase": self.state.phase as u8,
            "current": self.state.current_player,
            "events": self.arena.len(),
            "applied": self.applied.len(),
            "captured": self.captured,
        });
        let bytes = hydra_feed::canon::canonical_bytes_value(&doc, "rl:game-state")
            .unwrap_or_default();
        Sha256::digest(&bytes).into()
    }

}

/// Cohort hand-carry facts in engine seat order, taken from the record that
/// follows a hand boundary.
#[derive(Debug, Clone)]
struct HandCarry {
    oya: u8,
    honba: u8,
    kyotaku: u32,
    scores: [i32; 4],
    bakaze: String,
}

enum BatchOutcome {
    Continue,
    Carry(HandCarry),
    Terminal,
}

/// Closed payload builder with explicit defaults for every envelope field.
/// The event actor travels as the `make_envelope` `actor` parameter, never
/// inside the payload: every construction site passed the same value twice
/// and the envelope builder only read the parameter (live-reference checked;
/// `rustc` reported the field as never read).
#[derive(Debug, Clone, Default)]
struct TypedPayload {
    tile: Option<u8>,
    action_id: Option<u32>,
    source_seat: Option<u8>,
    consumed_tiles: Vec<u8>,
    offered_action_ids: Vec<u32>,
    accepted_action_ids: Vec<u32>,
    round_index: Option<u64>,
    scores: Option<[i32; 4]>,
    reason: Option<String>,
}

impl TypedPayload {
    fn empty() -> Self {
        Self::default()
    }
}

// -- free helpers --------------------------------------------------------------

fn engine_order_of(canonical: [i32; 4], inv: [usize; 4]) -> [i32; 4] {
    [canonical[inv[0]], canonical[inv[1]], canonical[inv[2]], canonical[inv[3]]]
}

fn canonical_scores(state: &GameState, inv: [usize; 4]) -> [i32; 4] {
    let mut perm = [0u8; 4];
    for (engine, seat) in inv.iter().enumerate() {
        perm[*seat] = engine as u8;
    }
    [
        state.players[perm[0] as usize].score,
        state.players[perm[1] as usize].score,
        state.players[perm[2] as usize].score,
        state.players[perm[3] as usize].score,
    ]
}

fn seat_winds_for(dealer: u8) -> [i64; 4] {
    [0, 1, 2, 3].map(|offset| 27 + ((dealer + offset) % 4) as i64)
}

/// LSB-first bit packing over canonical 6792 action order (bit i = action i
/// legal; agreed with ArtifactStorage).
pub fn pack_bits_lsb(mask: &[bool]) -> Vec<u8> {
    let mut out = vec![0u8; mask.len().div_ceil(8)];
    for (i, flag) in mask.iter().enumerate() {
        if *flag {
            out[i / 8] |= 1 << (i % 8);
        }
    }
    out
}

fn derive_game_id(ctx: &GameContext) -> Result<String, CoreError> {
    let doc = serde_json::json!({
        "rules_hash": ctx.rules_hash,
        "wall_digest": ctx.schedule_digest,
        "seat_permutation": ctx.seat_perm.to_vec(),
        "adapter_version": ctx.adapter_version,
    });
    let bytes = hydra_feed::canon::canonical_bytes_value(&doc, "rl:game-seed")
        .map_err(|e| CoreError(format!("game id canon failed: {e:?}")))?;
    let digest: [u8; 32] = Sha256::digest(&bytes).into();
    Ok(format!(
        "hydra2-riichienv-{}",
        digest[..8].iter().map(|b| format!("{b:02x}")).collect::<String>()
    ))
}
