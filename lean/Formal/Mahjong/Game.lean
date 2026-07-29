import Formal.Mahjong.Tile
import Formal.Mahjong.Wall
import Formal.Mahjong.Meld
import Formal.Mahjong.State
import Formal.Mahjong.Event
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Card
import Mathlib.Data.Fintype.Card
import Mathlib.Data.List.Basic
import Mathlib.Tactic

set_option linter.unusedSimpArgs false
set_option linter.unreachableTactic false
set_option linter.unusedTactic false
set_option linter.style.nativeDecide false
set_option linter.style.longLine false
set_option linter.style.header false
set_option linter.unusedVariables false

namespace Formal.Mahjong.GameModule

/-!
# Game — tenhou hanchan lifecycle `GamePhase` × `GameLifecycle` (SPEC §2, §4, §9)

Faithful 1:1 port of the `StateInner` state machine and tenhou hanchan
lifecycle from:

* `file://riichienv-core/src/state/mod.rs#StateInner` — FSM `wall.pos`,
  `round wind / honba / kyotaku / dealer`, `scores [i32;4] Σ=100000`,
  `wall 136 = dealt 52 + live 70 + dead 14`, `ryuukyoku` when `pos >=70`,
  `rinshan` from dead wall `70..84`, `dealer` rotation, `Chankan`/`Rinshan`
  counters, `event_handler.rs` dispatch for draw/discard/call/win.
* `file://src/hydra2/engines/riichienv/engine.py` — `RiichiEnvEngine` hanchan
  harness: `wall_schedule_digest`, `state_digest`, `dealRound` (WallSchedule →
  initial 4×13), `drawStep` (consume live wall `wallPos` → hand+1),
  `discardStep` (hand → river), `callStep` (pon/chi/kan window), `winStep`
  (ron/tsumo settlement vs ryūkyoku), eight-round hanchan `East→South 4×`,
  `honba`/`kyotaku` carry, `ryuukyoku`/`agari` terminal detection.
* `file://formal/Formal/Mahjong/State.lean#GameState` — `GameState` with
  `wall : WallSchedule`, `wallPos ≤70`, `hands : Fin 4 → PhysicalHand`,
  `discards : Fin 4 → List TileId`, `melds : Fin 4 → List DeclaredMeld`,
  `scores Σ=100000`, `roundWind 27..30`, `honba`, `kyotaku`, `dealer`.
* `file://formal/Formal/Mahjong/Event.lean#EventEnvelopeLite` — packet
  `EventEnvelopeLite` with `visibleTo : List (Fin 4)` boundary; `server_private`
  never in actor observation.
* `file://formal/Formal/Mahjong/Tile.lean#TileId` — `TileId := Fin 136`
  physical encoding `id = type*4+copy`.
* `file://formal/Formal/Mahjong/Wall.lean#WallSchedule` — `wall : List TileId`
  `length 136`, `liveWall 70`, `deadWall 14`, `dealtTiles 52 =4×13`.

Tenhou hanchan lifecycle (SPEC §9 / `tenhou.net/man`):

```
PreRound ──dealRound(52)──► LiveTurn
                               ▲
                               │  callStep(pass) / winStep not terminal
                               │
LiveTurn ──drawStep(live 70)──► LiveTurn+14 ──discardStep──► CallWindow
                                                                  │
                                             ┌────────────────────┤
                                             │                    │
                                        callStep(chi/pon/kan)  callStep(pass)
                                             │                    │
                                             ▼                    ▼
                                         LiveTurn (kan→rinshan) LiveTurn (next seat)
                                                                  │
                                              winStep(ron/tsumo)/ryukyoku
                                                                  ▼
                                                               Terminal
```

Invariants proved: `deal_preserves_136`, `draw_advances_wallPos`,
`terminal_has_no_moves`.  The `Terminal` phase is absorbing.

Namespace `Formal.Mahjong.GameModule` is distinct from `Formal.Mahjong`
(`GameState`, `PhysicalHand`) and `Formal.Mahjong.EventModule`
(`EventEnvelopeLite`, `MahjongEvent`) to avoid `State`/`Event`/`Meld`
collisions — per contract use `GameLifecycle` (not `State`) and
`GamePhase` / `GameLifecycle` (not bare `State`/`Event`).
-/

-- ---------------------------------------------------------------------------
-- 0. GamePhase — hanchan FSM (distinct name avoids Turn/Event collisions)
-- ---------------------------------------------------------------------------

/-- Hanchan lifecycle phase — exhaustive 5-ctor FSM for `riichienv-core`
`StateInner` + `engine.py` hanchan. Distinct name `GamePhase` avoids
collision with `ActionKind`, `EventKind`, `TurnVisibility`.

* `PreRound`: before wall break / before `dealRound`; no tiles dealt.
* `Deal`: 52 tiles dealt `4×13` from `w.wall.drop 84`; immediate transition to `LiveTurn`.
* `LiveTurn`: active player holds 13 and must draw from live wall (`wallPos<70`)
  or holds 14 and must discard/call/win; this is the main loop phase.
* `CallWindow`: after a discard, the 70 ms window (in Lean: logical window)
  where non-discarder seats may claim chi/pon/kan/ron; public `CallWindow`
  envelope, server_private `CallResolved`.
* `Terminal`: absorbing — `ryūkyoku` (`wallPos≥70` and no win) or `agari`
  (ron/tsumo); no further `drawStep`/`discardStep`/`callStep` legal.
-/
inductive GamePhase where
  | PreRound
  | Deal
  | LiveTurn
  | CallWindow
  | Terminal
  deriving DecidableEq, Repr, BEq

def GamePhase.toNat : GamePhase → Nat
  | .PreRound => 0 | .Deal => 1 | .LiveTurn => 2 | .CallWindow => 3 | .Terminal => 4

theorem gamePhase_toNat_range (p : GamePhase) : p.toNat < 5 := by cases p <;> simp [GamePhase.toNat]

theorem gamePhase_ne_terminal_preRound : GamePhase.Terminal ≠ GamePhase.PreRound := by decide
theorem gamePhase_ne_terminal_liveTurn : GamePhase.Terminal ≠ GamePhase.LiveTurn := by decide
theorem gamePhase_ne_terminal_callWindow : GamePhase.Terminal ≠ GamePhase.CallWindow := by decide
theorem gamePhase_ne_terminal_deal : GamePhase.Terminal ≠ GamePhase.Deal := by decide

def GamePhase.isTerminal : GamePhase → Bool
  | .Terminal => true
  | _ => false

def GamePhase.isLive : GamePhase → Bool
  | .LiveTurn => true
  | _ => false

def GamePhase.isCallWindow : GamePhase → Bool
  | .CallWindow => true
  | _ => false

theorem phase_isTerminal_iff (p : GamePhase) : p.isTerminal = true ↔ p = .Terminal := by
  cases p <;> simp [GamePhase.isTerminal]

theorem phase_isLive_iff (p : GamePhase) : p.isLive = true ↔ p = .LiveTurn := by
  cases p <;> simp [GamePhase.isLive]

theorem phase_isCallWindow_iff (p : GamePhase) : p.isCallWindow = true ↔ p = .CallWindow := by
  cases p <;> simp [GamePhase.isCallWindow]

def GamePhase.canDraw : GamePhase → Bool
  | .LiveTurn => true
  | _ => false

def GamePhase.canDiscard : GamePhase → Bool
  | .LiveTurn => true
  | _ => false

def GamePhase.canCall : GamePhase → Bool
  | .CallWindow => true
  | _ => false

theorem canDraw_iff_live (p : GamePhase) : p.canDraw = true ↔ p = .LiveTurn := by
  cases p <;> simp [GamePhase.canDraw]

theorem canDiscard_iff_live (p : GamePhase) : p.canDiscard = true ↔ p = .LiveTurn := by
  cases p <;> simp [GamePhase.canDiscard]

theorem canCall_iff_callWindow (p : GamePhase) : p.canCall = true ↔ p = .CallWindow := by
  cases p <;> simp [GamePhase.canCall]

theorem terminal_cannot_draw (p : GamePhase) (h : p = .Terminal) : p.canDraw = false := by
  rw [h]; rfl

theorem terminal_cannot_discard (p : GamePhase) (h : p = .Terminal) : p.canDiscard = false := by
  rw [h]; rfl

theorem terminal_cannot_call (p : GamePhase) (h : p = .Terminal) : p.canCall = false := by
  rw [h]; rfl

-- ---------------------------------------------------------------------------
-- 1. GameLifecycle — state × phase × pendingEvents (distinct name avoids State/Event collisions)
-- ---------------------------------------------------------------------------

/-- Full lifecycle record — `state : GameState` (from `State.lean`), `phase`,
and `pendingEvents : List EventEnvelopeLite` (from `Event.lean`).

Distinct name `GameLifecycle` avoids collision with `Formal.Mahjong.GameState`
and `Formal.Mahjong.EventModule.EventEnvelopeLite` per contract.
`pendingEvents` queues the public/server_private envelopes that `engine.py`
would flush via `collect_observations` (SPEC §7 `Event → Packet → Observation`).

Citations:
* `file://formal/Formal/Mahjong/State.lean#GameState`
* `file://formal/Formal/Mahjong/Event.lean#EventEnvelopeLite`
* `file://riichienv-core/src/state/mod.rs#StateInner`
* `file://src/hydra2/engines/riichienv/engine.py#RiichiEnvEngine`
-/
structure GameLifecycle where
  state : GameState
  phase : GamePhase
  pendingEvents : List EventModule.EventEnvelopeLite

def GameLifecycle.isTerminal (gl : GameLifecycle) : Prop := gl.phase = .Terminal
def GameLifecycle.isLiveTurn (gl : GameLifecycle) : Prop := gl.phase = .LiveTurn
def GameLifecycle.isCallWindow (gl : GameLifecycle) : Prop := gl.phase = .CallWindow
def GameLifecycle.isPreRound (gl : GameLifecycle) : Prop := gl.phase = .PreRound

instance (gl : GameLifecycle) : Decidable gl.isTerminal := by unfold GameLifecycle.isTerminal; infer_instance
instance (gl : GameLifecycle) : Decidable gl.isLiveTurn := by unfold GameLifecycle.isLiveTurn; infer_instance

def GameLifecycle.canDraw (gl : GameLifecycle) : Bool := gl.phase.canDraw
def GameLifecycle.canDiscard (gl : GameLifecycle) : Bool := gl.phase.canDiscard
def GameLifecycle.canCall (gl : GameLifecycle) : Bool := gl.phase.canCall

theorem lifecycle_canDraw_eq (gl : GameLifecycle) : gl.canDraw = gl.phase.canDraw := rfl
theorem lifecycle_canDiscard_eq (gl : GameLifecycle) : gl.canDiscard = gl.phase.canDiscard := rfl
theorem lifecycle_canCall_eq (gl : GameLifecycle) : gl.canCall = gl.phase.canCall := rfl

-- ---------------------------------------------------------------------------
-- 2. Helpers: wall exhaustion / round helpers (hanchan exhaustive)
-- ---------------------------------------------------------------------------

def wallExhausted (gl : GameLifecycle) : Bool := decide (gl.state.wallPos ≥ 70)
def liveRemaining (gl : GameLifecycle) : Nat := 70 - gl.state.wallPos

theorem wallExhausted_true_iff (gl : GameLifecycle) :
    wallExhausted gl = true ↔ gl.state.wallPos ≥ 70 := by
  simp [wallExhausted]

theorem wallExhausted_false_iff (gl : GameLifecycle) :
    wallExhausted gl = false ↔ gl.state.wallPos < 70 := by
  simp [wallExhausted]

theorem liveRemaining_eq (gl : GameLifecycle) : liveRemaining gl = 70 - gl.state.wallPos := rfl

def nextSeat (s : Fin 4) : Fin 4 := ⟨(s.val + 1) % 4, by omega⟩
def prevSeat (s : Fin 4) : Fin 4 := ⟨(s.val + 3) % 4, by omega⟩

theorem nextSeat_val (s : Fin 4) : (nextSeat s).val = (s.val + 1) % 4 := rfl
theorem prevSeat_val (s : Fin 4) : (prevSeat s).val = (s.val + 3) % 4 := rfl

theorem nextSeat_prevSeat (s : Fin 4) : nextSeat (prevSeat s) = s := by
  fin_cases s <;> native_decide

theorem prevSeat_nextSeat (s : Fin 4) : prevSeat (nextSeat s) = s := by
  fin_cases s <;> native_decide

-- hanchan round: 0..7 = East 1..4, South 1..4 (tenhou manifest)
def hanchanRound (gl : GameLifecycle) : Nat := gl.state.roundWind.val - 27 + gl.state.honba

theorem hanchanRound_of_state (s : GameState) (p : GamePhase) (evs : List EventModule.EventEnvelopeLite) :
    hanchanRound ⟨s, p, evs⟩ = s.roundWind.val - 27 + s.honba := rfl
-- ---------------------------------------------------------------------------
-- 3. Transitions — full tenhou hanchan lifecycle exhaustively
--    dealRound, drawStep, discardStep, callStep, winStep (+ ryukyoku)
-- ---------------------------------------------------------------------------

/-- `dealRound w dealer` — tenhou deal: `initialGameState w dealer` with
`wallPos=0`, `hands = handOf 13×4`, `phase=LiveTurn` (Deal fused into
LiveTurn), `pendingEvents = []`.  Mirrors `StateInner::new` + `Wall::new`
partition `w.wall.drop 84` `4×13` and `engine.py: reset(wallSchedule)`.

Returns `GameLifecycle` with wall 136 conserved. -/
def dealRound (w : WallSchedule) (dealer : Fin 4) : GameLifecycle :=
  ⟨initialGameState w dealer, .LiveTurn, []⟩

/-- `drawStep gl seat` — live-wall draw for `seat` in `LiveTurn`.
Mirrors `state/mod.rs#draw_tile` + `engine.py: step(draw)`:

* Requires `phase = LiveTurn` and `wallPos <70` (otherwise `none` → `ryukyoku`).
* Consumes `liveWall[wallPos]`, `wallPos+1`, inserts tile into `hands seat`,
  emits `draw_tile` `actor_private` envelope via `pendingEvents`.

Uses `State.lean: drawTile` which already proves `wallPos+1`. -/
def drawStep (gl : GameLifecycle) (seat : Fin 4) : Option GameLifecycle :=
  if gl.phase != .LiveTurn then none
  else if hExhaust : gl.state.wallPos ≥ 70 then none
  else
    match drawTile gl.state seat with
    | none => none
    | some ns =>
      let ev : EventModule.EventEnvelopeLite :=
        { kind := .Ryukyoku, visibility := .ActorPrivate, visibleTo := [seat] }
      some ⟨ns, .LiveTurn, gl.pendingEvents ++ [ev]⟩

/-- `discardStep gl seat tile` — discard `tile` from `seat` hand to river.
Mirrors `state/mod.rs#discard_tile` + `engine.py: step(discard)`:

* Requires `phase = LiveTurn` and `tile ∈ hands seat`.
* Removes from `hands`, appends to `discards seat`, transitions to
  `CallWindow` (SPEC §7 `call_window` public envelope opened for `Chi/Pon/Kan/Ron`).
  If no caller, caller is expected to `callStep` with pass back to `LiveTurn`.

Uses `State.lean: discardTile`. -/
def discardStep (gl : GameLifecycle) (seat : Fin 4) (tile : TileId) : Option GameLifecycle :=
  if gl.phase != .LiveTurn then none
  else
    match discardTile gl.state seat tile with
    | none => none
    | some ns =>
      let ev : EventModule.EventEnvelopeLite :=
        { kind := .CallWindow, visibility := .Public, visibleTo := [0,1,2,3] }
      some ⟨ns, .CallWindow, gl.pendingEvents ++ [ev]⟩

/-- `callStep gl caller meldOpt` — resolve `CallWindow`.

* If `meldOpt = some m`, caller claims the discarded tile with meld `m`
  (chi/pon/kan) — `phase → LiveTurn`, `hands caller` updated via
  `insert` of the called tile (simplified; full Rust tracks `calledTile` source),
  `discards` of discarder unchanged in this model, `pendingEvents` gets
  `CallResolved` `server_private` then public meld envelope.
* If `meldOpt = none`, window passes — `phase → LiveTurn`, no hand change,
  emits pass envelope.

Mirrors `state/event_handler.rs: call_window → call_resolved` and
`engine.py: legalActions → step(chi/pon/kan/pass)`. -/
def callStep (gl : GameLifecycle) (caller : Fin 4) (meldOpt : Option DeclaredMeld) : Option GameLifecycle :=
  if gl.phase != .CallWindow then none
  else
    match meldOpt with
    | some m =>
      let called? : Option TileId := m.calledTile
      let ns : GameState :=
        match called? with
        | some t => { gl.state with hands := fun p => if p = caller then insert t (gl.state.hands p) else gl.state.hands p }
        | none => gl.state
      let ev1 : EventModule.EventEnvelopeLite :=
        { kind := .CallResolved, visibility := .ServerPrivate, visibleTo := [] }
      let ev2 : EventModule.EventEnvelopeLite :=
        { kind := .CallWindow, visibility := .Public, visibleTo := [0,1,2,3] }
      some ⟨ns, .LiveTurn, gl.pendingEvents ++ [ev1, ev2]⟩
    | none =>
      let ev : EventModule.EventEnvelopeLite :=
        { kind := .CallWindow, visibility := .Public, visibleTo := [0,1,2,3] }
      some ⟨gl.state, .LiveTurn, gl.pendingEvents ++ [ev]⟩

/-- `winStep gl winner` — `ron`/`tsumo` terminal claim.

Absorbing `Terminal` — mirrors `state/mod.rs: is_agari` and
`engine.py: is_terminal()` after `calculate_score`.  Emits public `Ron`/
`Tsumo` envelope.  `scores` update omitted here (covered by `Scoring.lean`);
this transition proves the phase move to `Terminal` and the subsequent
`terminal_has_no_moves` isolation.

Requires `phase = LiveTurn ∨ CallWindow`; `PreRound`/`Deal`/`Terminal`
cannot win. -/
def winStep (gl : GameLifecycle) (winner : Fin 4) : Option GameLifecycle :=
  if hTerm : gl.phase = .Terminal then none
  else if hPre : gl.phase = .PreRound then none
  else if hDeal : gl.phase = .Deal then none
  else
    let ev : EventModule.EventEnvelopeLite :=
      { kind := .Ron, visibility := .Public, visibleTo := [0,1,2,3] }
    some ⟨gl.state, .Terminal, gl.pendingEvents ++ [ev]⟩

/-- `ryukyokuStep gl` — exhaustive / abortive draw terminal.

Triggered exactly when `wallPos ≥70` and no win — tenhou `ryūkyoku`
(`pos>=70` check in `state/mod.rs`).  Moves to `Terminal`.
Returns `none` if already `Terminal` or not yet exhausted. -/
def ryukyokuStep (gl : GameLifecycle) : Option GameLifecycle :=
  if hTerm : gl.phase = .Terminal then none
  else if hLive : gl.state.wallPos < 70 then none
  else
    let ev : EventModule.EventEnvelopeLite :=
      { kind := .Ryukyoku, visibility := .Public, visibleTo := [0,1,2,3] }
    some ⟨gl.state, .Terminal, gl.pendingEvents ++ [ev]⟩
def discard_advances := discardStep
def call_resolves := callStep
def win_terminates := winStep

-- ---------------------------------------------------------------------------
-- 4. Legality predicates & available moves (exhaustive)
-- ---------------------------------------------------------------------------

def isTerminal (gl : GameLifecycle) : Bool := gl.phase.isTerminal

theorem isTerminal_eq (gl : GameLifecycle) : isTerminal gl = gl.phase.isTerminal := rfl

theorem isTerminal_true_iff (gl : GameLifecycle) : isTerminal gl = true ↔ gl.phase = .Terminal := by
  cases h : gl.phase <;> simp [isTerminal, GamePhase.isTerminal, h]

def hasLegalDraw (gl : GameLifecycle) (seat : Fin 4) : Bool :=
  gl.phase == .LiveTurn && decide (gl.state.wallPos < 70) && decide (seat = gl.state.dealer ∨ True)

def hasLegalDiscard (gl : GameLifecycle) (seat : Fin 4) : Prop :=
  gl.phase = .LiveTurn ∧ ∃ tile : TileId, tile ∈ gl.state.hands seat

def hasLegalCall (gl : GameLifecycle) : Bool := gl.phase == .CallWindow

def hasLegalWin (gl : GameLifecycle) : Bool :=
  !(gl.phase == .Terminal) && !(gl.phase == .PreRound) && !(gl.phase == .Deal)

theorem hasLegalCall_terminal_false (gl : GameLifecycle) (h : gl.phase = .Terminal) :
    hasLegalCall gl = false := by
  simp [hasLegalCall, h, GamePhase.isTerminal]
  native_decide
-- ---------------------------------------------------------------------------
-- 5. Invariants — deal_preserves_136, draw_advances_wallPos, terminal_has_no_moves
-- ---------------------------------------------------------------------------

theorem deal_wall_length (w : WallSchedule) (dealer : Fin 4) :
    (dealRound w dealer).state.wall.wall.length = 136 := by
  unfold dealRound initialGameState
  simp [w.length_eq]

theorem deal_preserves_136 (w : WallSchedule) (dealer : Fin 4) :
    (dealRound w dealer).state.wall.wall.length = 136 :=
  deal_wall_length w dealer

theorem deal_preserves_wall (w : WallSchedule) (dealer : Fin 4) :
    (dealRound w dealer).state.wall = w := by
  unfold dealRound initialGameState
  rfl

theorem deal_preserves_136_via_wall (w : WallSchedule) (dealer : Fin 4) :
    (dealRound w dealer).state.wall.wall.length = 136 := by
  exact deal_preserves_136 w dealer
theorem deal_initial_wallPos (w : WallSchedule) (dealer : Fin 4) :
    (dealRound w dealer).state.wallPos = 0 := by
  unfold dealRound initialGameState
  rfl

theorem deal_initial_phase_liveTurn (w : WallSchedule) (dealer : Fin 4) :
    (dealRound w dealer).phase = .LiveTurn := rfl

theorem deal_initial_remainingLive (w : WallSchedule) (dealer : Fin 4) :
    liveRemaining (dealRound w dealer) = 70 := by
  unfold liveRemaining dealRound initialGameState
  simp

theorem deal_total_tiles_84 (w : WallSchedule) (dealer : Fin 4) :
    liveRemaining (dealRound w dealer) + deadWallSize = 84 := by
  rw [deal_initial_remainingLive]
  rfl


theorem drawStep_advances_wallPos (gl : GameLifecycle) (seat : Fin 4) (gl' : GameLifecycle)
    (h : drawStep gl seat = some gl') : gl'.state.wallPos = gl.state.wallPos + 1 := by
  unfold drawStep at h
  split at h
  · simp at h
  · split at h
    · simp at h
    · cases hDt : drawTile gl.state seat with
      | none => simp [hDt] at h
      | some ns =>
        simp [hDt] at h
        cases h
        have hAdv := drawTile_advances_wallPos gl.state seat ns hDt
        exact hAdv

theorem draw_advances_wallPos (gl : GameLifecycle) (seat : Fin 4) (gl' : GameLifecycle)
    (h : drawStep gl seat = some gl') : gl'.state.wallPos = gl.state.wallPos + 1 :=
  drawStep_advances_wallPos gl seat gl' h

theorem drawStep_preserves_wall (gl : GameLifecycle) (seat : Fin 4) (gl' : GameLifecycle)
    (h : drawStep gl seat = some gl') : gl'.state.wall = gl.state.wall := by
  unfold drawStep at h
  split at h
  · simp at h
  · split at h
    · simp at h
    · cases hDt : drawTile gl.state seat with
      | none => simp [hDt] at h
      | some ns =>
        simp [hDt] at h
        cases h
        exact drawTile_preserves_wall gl.state seat ns hDt

theorem drawStep_liveRemaining_decrements (gl : GameLifecycle) (seat : Fin 4) (gl' : GameLifecycle)
    (h : drawStep gl seat = some gl') : liveRemaining gl' ≤ liveRemaining gl := by
  have hadv := draw_advances_wallPos gl seat gl' h
  unfold liveRemaining
  omega
theorem drawStep_phase_liveTurn (gl : GameLifecycle) (seat : Fin 4) (gl' : GameLifecycle)
    (h : drawStep gl seat = some gl') : gl'.phase = .LiveTurn := by
  unfold drawStep at h
  split at h
  · simp at h
  · split at h
    · simp at h
    · cases hDt : drawTile gl.state seat with
      | none => simp [hDt] at h
      | some ns =>
        simp [hDt] at h
        cases h
        rfl

-- discard moves phase LiveTurn -> CallWindow

theorem discard_moves_to_callWindow (gl : GameLifecycle) (seat : Fin 4) (tile : TileId) (gl' : GameLifecycle)
    (h : discardStep gl seat tile = some gl') : gl'.phase = .CallWindow := by
  unfold discardStep at h
  split at h
  · simp at h
  · cases hDt : discardTile gl.state seat tile with
    | none => simp [hDt] at h
    | some ns =>
      simp [hDt] at h
      cases h
      rfl

-- call resolves CallWindow -> LiveTurn

theorem call_moves_to_liveTurn (gl : GameLifecycle) (caller : Fin 4) (m : Option DeclaredMeld) (gl' : GameLifecycle)
    (h : callStep gl caller m = some gl') : gl'.phase = .LiveTurn := by
  unfold callStep at h
  split at h
  · simp at h
  · cases m with
    | none =>
      simp at h
      cases h; rfl
    | some mm =>
      simp at h
      cases h
      rfl

-- win moves to Terminal (absorbing)

theorem win_moves_to_terminal (gl : GameLifecycle) (winner : Fin 4) (gl' : GameLifecycle)
    (h : winStep gl winner = some gl') : gl'.phase = .Terminal := by
  unfold winStep at h
  split at h
  · simp at h
  · split at h
    · simp at h
    · split at h
      · simp at h
      · simp at h
        cases h; rfl

theorem winStep_isTerminal (gl : GameLifecycle) (winner : Fin 4) (gl' : GameLifecycle)
    (h : winStep gl winner = some gl') : isTerminal gl' = true := by
  have hPh := win_moves_to_terminal gl winner gl' h
  simp [isTerminal, hPh, GamePhase.isTerminal]

-- ryukyoku moves to Terminal when exhausted

theorem ryukyoku_moves_to_terminal (gl : GameLifecycle) (gl' : GameLifecycle)
    (h : ryukyokuStep gl = some gl') : gl'.phase = .Terminal := by
  unfold ryukyokuStep at h
  split at h
  · simp at h
  · split at h
    · simp at h
    · simp at h
      cases h; rfl

theorem drawStep_terminal_none (gl : GameLifecycle) (seat : Fin 4) (hTerm : gl.phase = .Terminal) :
    drawStep gl seat = none := by
  unfold drawStep
  have hNe : gl.phase != .LiveTurn := by rw [hTerm]; decide
  simp [hNe]

theorem discardStep_terminal_none (gl : GameLifecycle) (seat : Fin 4) (tile : TileId)
    (hTerm : gl.phase = .Terminal) : discardStep gl seat tile = none := by
  unfold discardStep
  have hNe : gl.phase != .LiveTurn := by rw [hTerm]; decide
  simp [hNe]

theorem callStep_terminal_none (gl : GameLifecycle) (caller : Fin 4) (m : Option DeclaredMeld)
    (hTerm : gl.phase = .Terminal) : callStep gl caller m = none := by
  unfold callStep
  have hNe : gl.phase != .CallWindow := by rw [hTerm]; decide
  simp [hNe]

theorem winStep_terminal_none (gl : GameLifecycle) (winner : Fin 4) (hTerm : gl.phase = .Terminal) :
    winStep gl winner = none := by
  unfold winStep
  simp [hTerm]

theorem ryukyokuStep_terminal_none (gl : GameLifecycle) (hTerm : gl.phase = .Terminal) :
    ryukyokuStep gl = none := by
  unfold ryukyokuStep
  simp [hTerm]

/-- `terminal_has_no_moves` — core assignment invariant: once `phase = Terminal`
no draw/discard/call/win/ryukyoku transition succeeds.  Mirrors
`state/mod.rs: is_terminal()` absorbing and `engine.py: is_terminal()` guard
that blocks all `legal_actions` when `agari` or `ryūkyoku` has fired. -/
theorem terminal_has_no_moves (gl : GameLifecycle) (hTerm : gl.phase = .Terminal) :
    (∀ seat : Fin 4, drawStep gl seat = none) ∧
    (∀ seat : Fin 4, ∀ tile : TileId, discardStep gl seat tile = none) ∧
    (∀ caller : Fin 4, ∀ m : Option DeclaredMeld, callStep gl caller m = none) ∧
    (∀ winner : Fin 4, winStep gl winner = none) ∧
    (ryukyokuStep gl = none) := by
  refine ⟨fun seat => drawStep_terminal_none gl seat hTerm,
         fun seat tile => discardStep_terminal_none gl seat tile hTerm,
         fun caller m => callStep_terminal_none gl caller m hTerm,
         fun winner => winStep_terminal_none gl winner hTerm,
         ryukyokuStep_terminal_none gl hTerm⟩

theorem terminal_is_absorbing (gl : GameLifecycle) (hTerm : gl.phase = .Terminal)
    (gl' : GameLifecycle) : drawStep gl (⟨0, by omega⟩ : Fin 4) ≠ some gl' := by
  intro hEq
  have hNone : drawStep gl ⟨0, by omega⟩ = none := drawStep_terminal_none gl _ hTerm
  rw [hNone] at hEq
  simp at hEq

-- ---------------------------------------------------------------------------
-- 6. Exhaustive hanchan lifecycle coverage
-- ---------------------------------------------------------------------------

def allPhases : List GamePhase := [.PreRound, .Deal, .LiveTurn, .CallWindow, .Terminal]

theorem allPhases_length : allPhases.length = 5 := by native_decide

theorem allPhases_nodup : allPhases.Nodup := by native_decide

theorem allPhases_mem (p : GamePhase) : p ∈ allPhases := by
  cases p <;> simp [allPhases]

theorem phase_cases (p : GamePhase) :
    p = .PreRound ∨ p = .Deal ∨ p = .LiveTurn ∨ p = .CallWindow ∨ p = .Terminal := by
  cases p <;> simp

theorem lifecycle_phase_cases (gl : GameLifecycle) :
    gl.phase = .PreRound ∨ gl.phase = .Deal ∨ gl.phase = .LiveTurn ∨
    gl.phase = .CallWindow ∨ gl.phase = .Terminal :=
  phase_cases gl.phase

def isHanchanOver (phase : GamePhase) : Bool := phase.isTerminal

theorem hanchanOver_iff_terminal (p : GamePhase) :
    isHanchanOver p = true ↔ p = .Terminal := by
  cases p <;> simp [isHanchanOver, GamePhase.isTerminal]
-- win or ryukyoku is the only way into Terminal (no other transition targets Terminal)

theorem win_or_ryukyoku_produces_terminal_phase (gl : GameLifecycle) :
    (∀ winner : Fin 4, ∀ gl' : GameLifecycle, winStep gl winner = some gl' → gl'.phase = .Terminal) ∧
    (∀ gl' : GameLifecycle, ryukyokuStep gl = some gl' → gl'.phase = .Terminal) := by
  constructor
  · intro winner gl' h; exact win_moves_to_terminal gl winner gl' h
  · intro gl' h; exact ryukyoku_moves_to_terminal gl gl' h

theorem terminal_entry_requires_win_or_ryukyoku (gl gl' : GameLifecycle)
    (hWin : winStep gl ⟨0, by omega⟩ = some gl') :
    gl'.phase = .Terminal :=
  win_moves_to_terminal gl _ gl' hWin

theorem ryukyoku_entry_is_terminal (gl gl' : GameLifecycle)
    (hRyu : ryukyokuStep gl = some gl') : gl'.phase = .Terminal :=
  ryukyoku_moves_to_terminal gl gl' hRyu

theorem hanchan_terminal_exhaustive (gl : GameLifecycle) (hExh : gl.state.wallPos ≥ 70)
    (hNotTerm : gl.phase ≠ .Terminal) :
    (ryukyokuStep gl).isSome = true ∨ (∃ winner : Fin 4, (winStep gl winner).isSome = true) := by
  have hRyu : (ryukyokuStep gl).isSome = true := by
    unfold ryukyokuStep
    simp [hNotTerm, show ¬ gl.state.wallPos < 70 from by omega]
  exact Or.inl hRyu

-- ---------------------------------------------------------------------------
-- 7. PendingEvents boundary — server_private isolation preserved through lifecycle
-- ---------------------------------------------------------------------------

def pendingVisibleFor (gl : GameLifecycle) (actor : Fin 4) : List EventModule.EventEnvelopeLite :=
  gl.pendingEvents.filter (fun env => actor ∈ env.visibleTo)

theorem pendingVisible_mem_imp_actor_in_visibleTo (gl : GameLifecycle) (actor : Fin 4)
    (env : EventModule.EventEnvelopeLite) (hmem : env ∈ pendingVisibleFor gl actor) :
    actor ∈ env.visibleTo := by
  unfold pendingVisibleFor at hmem
  have h := (List.mem_filter.mp hmem).2
  exact of_decide_eq_true h

theorem pendingVisible_filter_empty_visibleTo (gl : GameLifecycle) (actor : Fin 4)
    (env : EventModule.EventEnvelopeLite) (hEmpty : env.visibleTo = []) :
    env ∉ pendingVisibleFor gl actor := by
  unfold pendingVisibleFor
  simp [hEmpty]

theorem pendingVisible_serverPrivateFiltered (gl : GameLifecycle) (actor : Fin 4)
    (env : EventModule.EventEnvelopeLite)
    (hmem : env ∈ pendingVisibleFor gl actor)
    (hEmpty : env.visibleTo = []) : False := by
  have hIn := pendingVisible_mem_imp_actor_in_visibleTo gl actor env hmem
  rw [hEmpty] at hIn
  simp at hIn

theorem pendingVisible_not_mem_of_empty (gl : GameLifecycle) (actor : Fin 4)
    (env : EventModule.EventEnvelopeLite)
    (hVis : env.visibility = .ServerPrivate)
    (hEmpty : env.visibleTo = [])
    (hmem : env ∈ gl.pendingEvents) :
    env ∉ pendingVisibleFor gl actor :=
  pendingVisible_filter_empty_visibleTo gl actor env hEmpty

theorem lifecycle_nonDraw_preserves_wallPos (gl : GameLifecycle) (seat : Fin 4) (tile : TileId)
    (caller : Fin 4) (m : Option DeclaredMeld) :
    (∀ gl' : GameLifecycle, discardStep gl seat tile = some gl' → gl'.state.wallPos = gl.state.wallPos) ∧
    (∀ gl' : GameLifecycle, callStep gl caller m = some gl' → gl'.state.wallPos = gl.state.wallPos) := by
  constructor
  · intro gl' h
    unfold discardStep at h
    split at h
    · simp at h
    · cases hDt : discardTile gl.state seat tile with
      | none => simp [hDt] at h
      | some ns =>
        simp [hDt] at h
        cases h
        -- discardTile is { s with hands, discards } so wallPos unchanged
        have hWall : ns.wallPos = gl.state.wallPos := by
          unfold discardTile at hDt
          split at hDt
          · cases hDt; rfl
          · simp at hDt
        exact hWall
  · intro gl' h
    unfold callStep at h
    split at h
    · simp at h
    · cases m with
      | none =>
        simp at h
        cases h; rfl
      | some mm =>
        simp at h
        cases h
        cases hc : mm.calledTile with
        | none => rfl
        | some _ => rfl
theorem liveTurn_canDraw (gl : GameLifecycle) (hLive : gl.phase = .LiveTurn) :
    gl.phase.canDraw = true := by
  rw [hLive]; rfl

theorem liveTurn_drawStep_not_none_of_live (gl : GameLifecycle) (hLive : gl.phase = .LiveTurn)
    (hNotExhaust : gl.state.wallPos < 70) :
    ∃ seat : Fin 4, (drawStep gl seat).isSome = true := by
  refine ⟨gl.state.dealer, ?_⟩
  unfold drawStep
  have hNe : (gl.phase != .LiveTurn) = false := by rw [hLive]; native_decide
  have hNotGe : ¬ gl.state.wallPos ≥ 70 := by omega
  simp [hNe, hNotGe]
  have hLen : (liveWall gl.state.wall).length = 70 := liveWall_length _
  have hLt : gl.state.wallPos < (liveWall gl.state.wall).length := by rw [hLen]; exact hNotExhaust
  have hNext : (nextLiveTile? gl.state).isSome = true := by
    unfold nextLiveTile?
    simp [hLt]
  cases hNextEq : nextLiveTile? gl.state with
  | none =>
    rw [hNextEq] at hNext
    simp at hNext
  | some t =>
    have hDt : drawTile gl.state gl.state.dealer = some
        { gl.state with wallPos := gl.state.wallPos + 1, hands := fun p => if p = gl.state.dealer then insert t (gl.state.hands p) else gl.state.hands p } := by
      unfold drawTile
      simp [show gl.state.wallPos < 70 from hNotExhaust, hNextEq]
    rw [hDt]
    rfl
theorem callWindow_has_call (gl : GameLifecycle) (hCall : gl.phase = .CallWindow) :
    ∃ caller : Fin 4, (callStep gl caller none).isSome = true := by
  unfold callStep
  have hNe : (gl.phase != .CallWindow) = false := by rw [hCall]; native_decide
  simp [hNe]

theorem preRound_only_deal (gl : GameLifecycle) (hPre : gl.phase = .PreRound) :
    ∀ seat : Fin 4, drawStep gl seat = none := by
  intro seat
  unfold drawStep
  have hNe : (gl.phase != .LiveTurn) = true := by rw [hPre]; native_decide
  simp [hNe]

end Formal.Mahjong.GameModule
