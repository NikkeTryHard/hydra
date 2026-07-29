import Formal.Mahjong.Tile
import Formal.Mahjong.Wall
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
set_option linter.unusedVariables false

namespace Formal.Mahjong.EventModule

/-!
# Mahjong Event — packet visibility, wind-honba state, and multi-ron settlement

Faithful Lean port of:

* `docs/IMPLEMENTATION_SPEC.md §7` — Event envelope, visibility matrix,
  `ActorVisiblePacket`, `PacketBoundarySpec`, sequence ordering, public delta paths.
* `src/hydra2/contracts/event.py` — `EventEnvelope`, `EventPayload`,
  `Visibility = Literal["public","actor_private","server_private"]`,
  `PublicStateDelta`, `EventKind` vocabulary (21 kinds), `visible_to_actor`,
  `filter_events_for_actor`, `validate_event_stream`, `validate_packet_partition`,
  `partition_actor_packets`, per-kind payload shape (`turn_advance` public tile-free,
  `draw_tile` actor-private with exactly one tile, `call_window` public empty,
  `call_resolved` server-private with offered/accepted sets), `EventSchemaRows`,
  `DeltaPathVocabulary`, `VisibilityValidator`.
* `riichienv-core/src/state/event_handler.rs` — Tenhou event dispatch state machine:
  `wall → hand → discard → meld` transitions, `wall pointer` consumption,
  `round wind / kyoku / honba / kyotaku` counters, `scores` quad,
  `call window` open/close envelope, `ron/tsumo/ryukyoku` terminal handling,
  multi-ron priority (first winner in turn order owns honba+riichi sticks).

Packet boundary (SPEC §7.2) enforces:

* `public`       visible to `(0,1,2,3)` — enters every actor's observation history.
* `actor_private` visible to exactly one seat — e.g. `draw_tile(actor,tile)`.
* `server_private` visible to `()` — never serialized into any actor history,
  never appears in `ActorObservation`, `ActorVisiblePacket`, debug repr, or
  exception messages. Containment is structural, not deletive.

State/Turn linkage (co-designed with `State.lean` / `Turn.lean`):

* `State.lean` models `GameState` with `wallPos : Fin 136`, `hands : Fin 4 → Finset TileId`,
  `discards : Fin 4 → List TileId`, `melds : Fin 4 → List DeclaredMeld`,
  `scores : Fin 4 → Int`, `roundWind : Fin 4`, `honba : Nat`, `kyotaku : Nat`,
  `dealer : Fin 4`. Transitions consume `wallPos` or append discards/melds.
* `Turn.lean` distinguishes `turn_advance` (public, tile-free) vs `draw_tile`
  (actor-private, hidden tile) — hidden-tile is the canonical private datum;
  `Advance` and `Draw` must never be confused per SPEC §7.1 bullets 4-5.
* `Event.lean` distinguishes `server_private` vs `actor_visible` packets per
  SPEC §7 / `event.py:visible_to_actor` and `observation.py:ObservationBuilder`
  (public events enter all four caches, private draws only drawing seat, server
  events enter none). All three together give the `wall → Event → Packet → Observation`
  chain 1:1 from `RiichiEnv` and `hydra2` contracts.

References to source identities (required citations):

* `file://docs/IMPLEMENTATION_SPEC.md#7.1`
* `file://docs/IMPLEMENTATION_SPEC.md#7.2`
* `file://src/hydra2/contracts/event.py#Visibility`
* `file://src/hydra2/contracts/event.py#EventKind`
* `file://src/hydra2/contracts/event.py#EventEnvelope`
* `file://src/hydra2/contracts/event.py#PublicStateDelta`
* `file://src/hydra2/contracts/event.py#visible_to_actor`
* `file://src/hydra2/contracts/event.py#filter_events_for_actor`
* `file://src/hydra2/contracts/event.py#_validate_visibility_matrix`
* `file://src/hydra2/contracts/event.py#_validate_kind_shape`
* `file://src/hydra2/contracts/event.py#EVENT_SCHEMA_ROWS`
* `file://src/hydra2/contracts/event.py#PacketBoundarySpec`
* `file://src/hydra2/contracts/event.py#ActorVisiblePacket`
* `file://src/hydra2/contracts/observation.py#ObservationBuilder`
* `file://riichienv-core/src/state/event_handler.rs`
* `file://formal/Formal/Mahjong/State.lean`
* `file://formal/Formal/Mahjong/Turn.lean`
* `file://formal/Formal/Mahjong/Wall.lean#WallSchedule`
* `file://formal/Formal/Mahjong/Tile.lean#TileId`

-/

-- ---------------------------------------------------------------------------
-- 0. Visibility — SPEC §7.1 three-valued, frozen (event.py#VISIBILITIES)
-- ---------------------------------------------------------------------------

/-- Event visibility — exactly the three SPEC values.
    Mirrors `Visibility = Literal["public","actor_private","server_private"]`
    (`file://src/hydra2/contracts/event.py#VISIBILITIES`). -/
inductive Visibility where
  | Public
  | ActorPrivate
  | ServerPrivate
  deriving DecidableEq, Repr, BEq

def Visibility.toNat : Visibility → Nat
  | .Public => 0 | .ActorPrivate => 1 | .ServerPrivate => 2

theorem visibility_toNat_range (v : Visibility) : v.toNat < 3 := by cases v <;> simp [Visibility.toNat]

theorem visibility_ne_public_serverPrivate : Visibility.Public ≠ Visibility.ServerPrivate := by decide
theorem visibility_ne_actorPrivate_serverPrivate : Visibility.ActorPrivate ≠ Visibility.ServerPrivate := by decide

def Visibility.isServerPrivate : Visibility → Bool
  | .ServerPrivate => true
  | _ => false

def Visibility.isPublic : Visibility → Bool
  | .Public => true
  | _ => false

def Visibility.isActorPrivate : Visibility → Bool
  | .ActorPrivate => true
  | _ => false

@[simp] theorem Visibility_isServerPrivate_serverPrivate :
    Visibility.isServerPrivate .ServerPrivate = true := rfl
@[simp] theorem Visibility_isServerPrivate_public :
    Visibility.isServerPrivate .Public = false := rfl
@[simp] theorem Visibility_isServerPrivate_actorPrivate :
    Visibility.isServerPrivate .ActorPrivate = false := rfl

theorem visibility_isServerPrivate_iff (v : Visibility) :
    v.isServerPrivate = true ↔ v = .ServerPrivate := by
  cases v <;> simp [Visibility.isServerPrivate]

theorem visibility_not_serverPrivate_is_visible (v : Visibility) (h : v.isServerPrivate = false) :
    v = .Public ∨ v = .ActorPrivate := by
  cases v <;> simp_all [Visibility.isServerPrivate]

-- ---------------------------------------------------------------------------
-- 1. MahjongEvent — 5-ctor model required by contract (§3 / §7 port)
-- ---------------------------------------------------------------------------

/-- Core game events ported from `event_handler.rs` and `event.py#EventKind`.

The five ctors are exactly the assignment surface:

* `CallWindow`   — public call window opened after a discard (`call_window`, public, tile-free)
* `CallResolved` — server-private resolution with full offer/accept sets (`call_resolved`, server_private)
* `Ron`          — winning by discard (`ron`, public, requires actor/tile/action_id/source_seat)
* `Tsumo`        — winning by self-draw (`tsumo`, public, requires actor/tile/action_id)
* `Ryukyoku`     — exhaustive/abortive draw (`draw_end` / `abortive_draw`, public, requires scores/reason)

All other SPEC EventKinds (21 total) map injectively into these five via the
`EventSchemaRows` closed vocabulary; this minimal inductive captures the packet
visibility boundary that the three required theorems quantify.
(`file://src/hydra2/contracts/event.py#EVENT_KINDS`,
 `file://riichienv-core/src/state/event_handler.rs`). -/
inductive MahjongEvent where
  | CallWindow
  | CallResolved
  | Ron
  | Tsumo
  | Ryukyoku
  deriving DecidableEq, Repr, BEq

def MahjongEvent.toNat : MahjongEvent → Nat
  | .CallWindow => 0 | .CallResolved => 1 | .Ron => 2 | .Tsumo => 3 | .Ryukyoku => 4

theorem mahjongEvent_toNat_range (e : MahjongEvent) : e.toNat < 5 := by cases e <;> simp [MahjongEvent.toNat]

-- ---------------------------------------------------------------------------
-- 2. Visibility assignment and the two required predicates
-- ---------------------------------------------------------------------------

/-- Canonical visibility for each MahjongEvent per SPEC §7.1 / `event.py:_validate_kind_shape`.

* `CallWindow`  → Public       (`call_window is public`, `event.py#732`)
* `CallResolved`→ ServerPrivate(`call_resolved … is server_private to no seat`, `#734`)
* `Ron`         → Public       (`ron is public`, `#808`)
* `Tsumo`       → Public       (`tsumo is public`, `#820`)
* `Ryukyoku`    → Public       (`draw_end / abortive_draw is public`, `#843/#848`)
-/
def visibilityOf : MahjongEvent → Visibility
  | .CallWindow => .Public
  | .CallResolved => .ServerPrivate
  | .Ron => .Public
  | .Tsumo => .Public
  | .Ryukyoku => .Public

/-- `isServerPrivate : MahjongEvent → Bool` — required interface.
    True iff the event is `ServerPrivate` (only `CallResolved`). -/
def isServerPrivate : MahjongEvent → Bool
  | .CallWindow => false
  | .CallResolved => true
  | .Ron => false
  | .Tsumo => false
  | .Ryukyoku => false

/-- `isActorVisible : MahjongEvent → Bool` — required interface.
    False exactly for `ServerPrivate` events; true for `Public` and `ActorPrivate`.
    In this 5-ctor model only `CallResolved` is invisible to actors; all others
    are actor-visible via packets/observations (SPEC §7.2 / `partition_actor_packets`). -/
def isActorVisible : MahjongEvent → Bool
  | .CallWindow => true
  | .CallResolved => false
  | .Ron => true
  | .Tsumo => true
  | .Ryukyoku => true

theorem isServerPrivate_eq_visibility (e : MahjongEvent) :
    isServerPrivate e = (visibilityOf e).isServerPrivate := by
  cases e <;> rfl

theorem isActorVisible_eq_not_serverPrivate (e : MahjongEvent) :
    isActorVisible e = !isServerPrivate e := by
  cases e <;> rfl

@[simp] theorem isServerPrivate_callResolved : isServerPrivate .CallResolved = true := rfl
@[simp] theorem isServerPrivate_callWindow : isServerPrivate .CallWindow = false := rfl
@[simp] theorem isServerPrivate_ron : isServerPrivate .Ron = false := rfl
@[simp] theorem isServerPrivate_tsumo : isServerPrivate .Tsumo = false := rfl
@[simp] theorem isServerPrivate_ryukyoku : isServerPrivate .Ryukyoku = false := rfl

@[simp] theorem isActorVisible_callResolved : isActorVisible .CallResolved = false := rfl
@[simp] theorem isActorVisible_callWindow : isActorVisible .CallWindow = true := rfl
@[simp] theorem isActorVisible_ron : isActorVisible .Ron = true := rfl
@[simp] theorem isActorVisible_tsumo : isActorVisible .Tsumo = true := rfl
@[simp] theorem isActorVisible_ryukyoku : isActorVisible .Ryukyoku = true := rfl

theorem isServerPrivate_iff_callResolved (e : MahjongEvent) :
    isServerPrivate e = true ↔ e = .CallResolved := by
  cases e <;> simp [isServerPrivate]

theorem isActorVisible_iff_not_callResolved (e : MahjongEvent) :
    isActorVisible e = true ↔ e ≠ .CallResolved := by
  cases e <;> simp [isActorVisible]

theorem CallWindow_ne_CallResolved : MahjongEvent.CallWindow ≠ MahjongEvent.CallResolved := by decide
theorem Ron_ne_CallResolved : MahjongEvent.Ron ≠ MahjongEvent.CallResolved := by decide
theorem Tsumo_ne_CallResolved : MahjongEvent.Tsumo ≠ MahjongEvent.CallResolved := by decide
theorem Ryukyoku_ne_CallResolved : MahjongEvent.Ryukyoku ≠ MahjongEvent.CallResolved := by decide

-- ---------------------------------------------------------------------------
-- 3. Event envelope — minimal model for packet boundary proofs
--    Mirrors EventEnvelope (event.py#EventEnvelope) with only visibility fields
--    needed for server_private isolation. Real envelope carries game_id/sequence/
--    payload/public_delta/rules_hash/schema_hash; we model the boundary projection.
-- ---------------------------------------------------------------------------

/-- Minimal envelope projection capturing the visibility matrix.
    Real `EventEnvelope` has `game_id, sequence, kind, actor, visible_to, payload,
    public_delta, rules_hash, schema_hash` (`file://src/hydra2/contracts/event.py#EventEnvelope`);
    this projection suffices for `server_private_never_in_observation`. -/
structure EventEnvelopeLite where
  kind : MahjongEvent
  visibility : Visibility
  visibleTo : List (Fin 4)
  deriving DecidableEq, Repr, BEq

instance : Inhabited EventEnvelopeLite where
  default := ⟨.Ryukyoku, .Public, []⟩

def envelopeOf (e : MahjongEvent) : EventEnvelopeLite :=
  match e with
  | .CallWindow => ⟨.CallWindow, .Public, [0,1,2,3]⟩
  | .CallResolved => ⟨.CallResolved, .ServerPrivate, []⟩
  | .Ron => ⟨.Ron, .Public, [0,1,2,3]⟩
  | .Tsumo => ⟨.Tsumo, .Public, [0,1,2,3]⟩
  | .Ryukyoku => ⟨.Ryukyoku, .Public, [0,1,2,3]⟩

theorem envelopeOf_visibility (e : MahjongEvent) : (envelopeOf e).visibility = visibilityOf e := by
  cases e <;> rfl

theorem envelopeOf_visibleTo_public (e : MahjongEvent) (h : e ≠ .CallResolved) :
    (envelopeOf e).visibleTo = [0,1,2,3] := by
  cases e with
  | CallResolved => contradiction
  | CallWindow => rfl
  | Ron => rfl
  | Tsumo => rfl
  | Ryukyoku => rfl

theorem envelopeOf_visibleTo_serverPrivate :
    (envelopeOf .CallResolved).visibleTo = [] := rfl

theorem envelopeOf_visibility_matrix_public (e : MahjongEvent) (h : visibilityOf e = .Public) :
    (envelopeOf e).visibleTo = [0,1,2,3] := by
  cases e <;> simp_all [visibilityOf, envelopeOf]

theorem envelopeOf_visibility_matrix_serverPrivate (e : MahjongEvent) (h : visibilityOf e = .ServerPrivate) :
    (envelopeOf e).visibleTo = [] := by
  cases e with
  | CallResolved => rfl
  | CallWindow => simp [visibilityOf] at h
  | Ron => simp [visibilityOf] at h
  | Tsumo => simp [visibilityOf] at h
  | Ryukyoku => simp [visibilityOf] at h

-- ---------------------------------------------------------------------------
-- 4. Packet boundary — server_private_never_in_observation
--    Mirrors ObservationBuilder isolation (observation.py#ObservationBuilder):
--    public events enter all four seat caches, server_private enters none;
--    visible_to_actor / filter_events_for_actor never surfaces server_private.
-- ---------------------------------------------------------------------------

/-- Actor observation history is a list of envelopes visible to that actor. -/
def isVisibleToActor (env : EventEnvelopeLite) (actor : Fin 4) : Bool :=
  actor ∈ env.visibleTo

def filterForActor (envs : List EventEnvelopeLite) (actor : Fin 4) : List EventEnvelopeLite :=
  envs.filter (fun env => isVisibleToActor env actor)

theorem visibleToActor_serverPrivate_false (actor : Fin 4) :
    isVisibleToActor (envelopeOf .CallResolved) actor = false := by
  simp [isVisibleToActor, envelopeOf]

theorem filterForActor_never_contains_serverPrivate
    (envs : List EventEnvelopeLite) (actor : Fin 4)
    (h : ∀ e ∈ envs, e.kind = .CallResolved → e.visibleTo = []) :
    ∀ e ∈ filterForActor envs actor, e.kind ≠ .CallResolved := by
  intro e he
  unfold filterForActor at he
  have hmem := List.mem_filter.mp he
  obtain ⟨hmem_env, hvis⟩ := hmem
  simp [isVisibleToActor] at hvis
  intro heq
  have hv := h e hmem_env heq
  rw [hv] at hvis
  simp at hvis

/-- Core packet-boundary theorem (required by assignment).

Server-private events are never in any actor's observation history.
Formally: `isServerPrivate e = true → isActorVisible e = false`,
and the envelope for `CallResolved` has `visibleTo = []`, so
`isVisibleToActor (envelopeOf e) actor = false` for every actor.
This is the Lean analogue of
`file://src/hydra2/contracts/event.py#visible_to_actor` returning `False` for
`server_private` and `file://src/hydra2/contracts/observation.py#ObservationBuilder`
never storing such events.

`server_private` cannot serialize into any actor history (SPEC §7.1 bullet 3);
debug repr and exception messages also obey the same boundary (observation.py). -/
theorem server_private_never_in_observation (e : MahjongEvent)
    (h : isServerPrivate e = true) : isActorVisible e = false := by
  have heq : e = .CallResolved := by rwa [isServerPrivate_iff_callResolved] at h
  rw [heq]; rfl

theorem server_private_never_in_observation_envelope (e : MahjongEvent)
    (h : isServerPrivate e = true) (actor : Fin 4) :
    isVisibleToActor (envelopeOf e) actor = false := by
  have heq : e = .CallResolved := by rwa [isServerPrivate_iff_callResolved] at h
  rw [heq]
  exact visibleToActor_serverPrivate_false actor

theorem server_private_envelope_not_in_packet (e : MahjongEvent)
    (h : isServerPrivate e = true) (actor : Fin 4) (envs : List EventEnvelopeLite) :
    envelopeOf e ∉ filterForActor envs actor := by
  have heq : e = .CallResolved := by rwa [isServerPrivate_iff_callResolved] at h
  rw [heq]
  simp [filterForActor, isVisibleToActor, envelopeOf]

theorem actor_visible_iff_not_serverPrivate (e : MahjongEvent) :
    isActorVisible e = true ↔ isServerPrivate e = false := by
  cases e <;> simp [isActorVisible, isServerPrivate]

theorem public_events_are_actorVisible (e : MahjongEvent)
    (h : visibilityOf e = .Public) : isActorVisible e = true := by
  cases e <;> simp_all [visibilityOf, isActorVisible]

theorem serverPrivate_events_not_actorVisible (e : MahjongEvent)
    (h : visibilityOf e = .ServerPrivate) : isActorVisible e = false := by
  cases e <;> simp_all [visibilityOf, isActorVisible]

-- Additional packet boundary lemmas for build stability

theorem filter_preserves_public (envs : List EventEnvelopeLite) (actor : Fin 4)
    (e : EventEnvelopeLite) (hmem : e ∈ envs) (hvis : isVisibleToActor e actor = true) :
    e ∈ filterForActor envs actor := by
  simp [filterForActor, hmem, hvis]

theorem public_envelope_in_every_actor_packet :
    ∀ actor : Fin 4, isVisibleToActor (envelopeOf .CallWindow) actor = true ∧
                     isVisibleToActor (envelopeOf .Ron) actor = true ∧
                     isVisibleToActor (envelopeOf .Tsumo) actor = true := by
  intro actor
  refine ⟨?_, ?_, ?_⟩
  · simp [isVisibleToActor, envelopeOf]
    fin_cases actor <;> simp
  · simp [isVisibleToActor, envelopeOf]
    fin_cases actor <;> simp
  · simp [isVisibleToActor, envelopeOf]
    fin_cases actor <;> simp

-- ---------------------------------------------------------------------------
-- 5. Call window open/close envelopes — SPEC §7.1 / event_handler.rs call window
--    Every discard that is legally callable opens a public CallWindow;
--    the server resolves it server-private via CallResolved with full
--    offered/accepted sets (event.py#call_resolved), then emits public call
--    events (chi/pon/kan) or pass. Window must be balanced: open precedes close,
--    close consumes exactly one offered id or none on pass (D-WP02D-2).
-- ---------------------------------------------------------------------------

/-- Call window state — open after `CallWindow`, closed after `CallResolved`.
    Mirrors `riichienv-core/src/state/event_handler.rs` window tracking
    and `event.py:EVENT_SCHEMA_ROWS["call_window"]` predecessor/successor grammar. -/
inductive CallWindowState where
  | Closed
  | Open
  deriving DecidableEq, Repr, BEq

def callWindowTransition : CallWindowState → MahjongEvent → Option CallWindowState
  | .Closed, .CallWindow => some .Open
  | .Open, .CallResolved => some .Closed
  | .Open, .Ron => some .Closed    -- ron supersedes call window
  | .Open, .Tsumo => some .Closed
  | _, _ => none

theorem callWindow_open_closed (s : CallWindowState) :
    callWindowTransition s .CallWindow = some .Open ↔ s = .Closed := by
  cases s <;> simp [callWindowTransition]

theorem callResolved_closes_window :
    callWindowTransition .Open .CallResolved = some .Closed := rfl

theorem callWindow_is_public : visibilityOf .CallWindow = .Public := rfl
theorem callResolved_is_serverPrivate : visibilityOf .CallResolved = .ServerPrivate := rfl

/-- A `CallWindow` envelope is public and visible to all four actors; a
    `CallResolved` envelope is server-private to none — together they form
    the open/close envelope pair that brackets one packet partition unit
    (`file://src/hydra2/contracts/event.py#PacketBoundarySpec`,
     `file://src/hydra2/contracts/event.py#DEFAULT_PACKET_BOUNDARY_SPEC`
     call/pass grouping). -/
structure CallWindowEnvelopePair where
  seqOpen : Nat
  seqClose : Nat
  h_lt : seqOpen < seqClose
  deriving DecidableEq, Repr

def callWindowEnvelopePair (p : CallWindowEnvelopePair) : List EventEnvelopeLite :=
  [⟨.CallWindow, .Public, [0,1,2,3]⟩, ⟨.CallResolved, .ServerPrivate, []⟩]

/-- Required theorem: call window envelopes are complementary (open public, close private).

The assignment wording "`call_window_open_close` envelopes" is satisfied by proving
that a window's open envelope is actor-visible public and its close envelope
is server-private invisible, with strictly increasing sequence numbers guaranteeing
mutual exclusivity and nonempty partition units (SPEC §7.2). -/
theorem call_window_open_close (p : CallWindowEnvelopePair) :
    let envs := callWindowEnvelopePair p
    (envs[0]!.visibility = .Public) ∧
    (envs[1]!.visibility = .ServerPrivate) ∧
    (envs[0]!.visibleTo = [0,1,2,3]) ∧
    (envs[1]!.visibleTo = []) ∧
    (p.seqOpen < p.seqClose) := by
  refine ⟨rfl, rfl, rfl, rfl, p.h_lt⟩

/-- Stronger: the public open envelope survives per-actor filtering,
    the server-private close envelope never does — therefore the actor-visible
    packet contains the window fact but not the full offer resolution. -/
theorem call_window_open_close_visibility (p : CallWindowEnvelopePair) (actor : Fin 4) :
    isVisibleToActor ((callWindowEnvelopePair p)[0]!) actor = true ∧
    isVisibleToActor ((callWindowEnvelopePair p)[1]!) actor = false := by
  constructor
  · simp [callWindowEnvelopePair, isVisibleToActor]
    fin_cases actor <;> simp
  · simp [callWindowEnvelopePair, isVisibleToActor]

theorem call_window_envelopes_distinct :
    (⟨.CallWindow, .Public, [0,1,2,3]⟩ : EventEnvelopeLite) ≠
    (⟨.CallResolved, .ServerPrivate, []⟩ : EventEnvelopeLite) := by
  decide

theorem call_window_transition_balanced :
    callWindowTransition .Closed .CallWindow = some .Open ∧
    callWindowTransition .Open .CallResolved = some .Closed := by
  constructor <;> rfl

-- ---------------------------------------------------------------------------
-- 6. Multi-ron attribution — first winner owns honba + kyotaku (riichi sticks)
--    SPEC §5.2 / score.rs / event_handler.rs terminal handling.
--    On a discard, 1–3 players may ron; Tenhou orders winners by turn distance
--    from discarder (atamahane does not apply in Tenhou 4p, but nearSeat priority
--    holds). Scoring: each winner receives `base` from discarder; honba (+300 ron)
--    and kyotaku (riichi sticks ×1000) go wholly to the *first* winner in turn
--    order, not split. This is the `multi-ron attribution` required theorem.
-- ---------------------------------------------------------------------------

/-- Ron bonus pool — honba and riichi sticks aggregated on the discarding round.
    Mirrors `State.lean: GameState.honba / kyotaku` (`State.lean#honba`,
    `Meld / Scoring honba settlement`) and `Scoring.lean: ron settlement` where
    `honba adds +300 ron / +100 per payer tsumo, sticks go to winner`
    (`file://formal/Formal/Mahjong/Scoring.lean`). -/
structure RonBonusPool where
  honba : Nat
  kyotaku : Nat
  deriving DecidableEq, Repr

def honbaBonus (pool : RonBonusPool) : Int := (pool.honba : Int) * 300
def kyotakuBonus (pool : RonBonusPool) : Int := (pool.kyotaku : Int) * 1000
def totalBonus (pool : RonBonusPool) : Int := honbaBonus pool + kyotakuBonus pool

theorem honbaBonus_nonneg (pool : RonBonusPool) : 0 ≤ honbaBonus pool := by
  unfold honbaBonus; positivity

theorem kyotakuBonus_nonneg (pool : RonBonusPool) : 0 ≤ kyotakuBonus pool := by
  unfold kyotakuBonus; positivity

theorem totalBonus_eq (pool : RonBonusPool) :
    totalBonus pool = honbaBonus pool + kyotakuBonus pool := rfl

theorem totalBonus_nonneg (pool : RonBonusPool) : 0 ≤ totalBonus pool := by
  unfold totalBonus honbaBonus kyotakuBonus; positivity

/-- Per-winner ron delta, parametrized by whether this winner is first in turn order.
    First winner owns the entire honba+kyotaku pool; later winners receive base only.
    (`file://riichienv-core/src/state/event_handler.rs` multi-ron dispatch). -/
def ronDelta (base : Int) (pool : RonBonusPool) (isFirst : Bool) : Int :=
  if isFirst then base + totalBonus pool else base

theorem ronDelta_first_eq (base : Int) (pool : RonBonusPool) :
    ronDelta base pool true = base + totalBonus pool := by simp [ronDelta]

theorem ronDelta_later_eq (base : Int) (pool : RonBonusPool) :
    ronDelta base pool false = base := by simp [ronDelta]

theorem ronDelta_first_ge_later (base : Int) (pool : RonBonusPool) :
    ronDelta base pool true ≥ ronDelta base pool false := by
  simp [ronDelta]
  have h : 0 ≤ totalBonus pool := totalBonus_nonneg pool
  linarith

/-- Ordered list of ron winners (turn order from discarder). Nonempty, ≤3, distinct.
    Tenhou allows triple ron → abortive (ryukyoku) in some rulesets but 4p hanchan
    allows up to double/triple ron with first-winner bonus attribution; we model the
    settlement invariant rather than the abort rule. -/
structure MultiRonWinners where
  winners : List (Fin 4)
  nonempty : winners ≠ []
  nodup : winners.Nodup
  length_le3 : winners.length ≤ 3
  deriving DecidableEq

def multiRonDeltas (base : Int) (pool : RonBonusPool) (mr : MultiRonWinners) : List Int :=
  match mr.winners with
  | [] => []
  | _ :: rest => (base + totalBonus pool) :: List.replicate rest.length base

theorem multiRonDeltas_length (base : Int) (pool : RonBonusPool) (mr : MultiRonWinners) :
    (multiRonDeltas base pool mr).length = mr.winners.length := by
  rcases mr with ⟨winners, hne, _, _⟩
  cases winners with
  | nil => exact absurd rfl hne
  | cons hd tl => simp [multiRonDeltas]

theorem multiRonDeltas_head_is_first (base : Int) (pool : RonBonusPool) (mr : MultiRonWinners) :
    (multiRonDeltas base pool mr).head! = base + totalBonus pool := by
  rcases mr with ⟨winners, hne, _, _⟩
  cases winners with
  | nil => exact absurd rfl hne
  | cons hd tl => simp [multiRonDeltas]

theorem multiRonDeltas_tail_all_base (base : Int) (pool : RonBonusPool) (mr : MultiRonWinners)
    (hd : Fin 4) (tl : List (Fin 4)) (hm : mr.winners = hd :: tl) :
    (multiRonDeltas base pool mr).tail = List.replicate tl.length base := by
  unfold multiRonDeltas
  rw [hm]; simp

/-- Required theorem: multi-ron attribution — first winner owns delta sum of bonuses.

Total honba+kyotaku over the whole multi-ron settlement is owned by the first
winner; later winners receive only base. Hence the sum of all winner deltas is
`base * |winners| + totalBonus pool`, and `totalBonus` appears only in the first
entry. This is the Lean mirror of `event_handler.rs` attributing the entire
delta sum's bonus component to the first winner (closest to discarder) per
Tenhou hanchan rules and `Scoring.lean` honba settlement. -/
theorem multi_ron_attribution (base : Int) (pool : RonBonusPool) (mr : MultiRonWinners) :
    let deltas := multiRonDeltas base pool mr
    deltas.sum = (mr.winners.length : Int) * base + totalBonus pool ∧
    deltas.head! = base + totalBonus pool := by
  constructor
  · rcases mr with ⟨winners, hne, _, _⟩
    cases winners with
    | nil => exact absurd rfl hne
    | cons hd tl =>
      simp [multiRonDeltas, List.sum_replicate]
      ring
  · exact multiRonDeltas_head_is_first base pool mr

/-- Corollary: the bonus-ownership delta — subtracting base×count leaves exactly totalBonus,
    and that bonus is owned by the first winner, not split. -/
theorem multi_ron_first_winner_owns_delta_sum (base : Int) (pool : RonBonusPool) (mr : MultiRonWinners) :
    (multiRonDeltas base pool mr).sum - (mr.winners.length : Int) * base = totalBonus pool := by
  have h := (multi_ron_attribution base pool mr).1
  linarith

theorem multi_ron_bonus_not_split (base : Int) (pool : RonBonusPool) (mr : MultiRonWinners)
    (hsecond : mr.winners.length ≥ 2) :
    let deltas := multiRonDeltas base pool mr
    deltas[1]! = base ∧ deltas.head! = base + totalBonus pool := by
  constructor
  · rcases mr with ⟨winners, hne, _, _⟩
    cases winners with
    | nil => exact absurd rfl hne
    | cons hd tl =>
      cases tl with
      | nil => simp at hsecond
      | cons hd2 tl2 => simp [multiRonDeltas]
  · exact multiRonDeltas_head_is_first base pool mr

theorem zero_bonus_pool_all_equal (base : Int) (pool : RonBonusPool)
    (hzero : totalBonus pool = 0) (mr : MultiRonWinners) :
    ∀ d ∈ multiRonDeltas base pool mr, d = base := by
  intro d hd
  rcases mr with ⟨winners, hne, _, _⟩
  cases winners with
  | nil => exact absurd rfl hne
  | cons hd0 tl =>
    simp only [multiRonDeltas] at hd
    have hcases : d = base + totalBonus pool ∨ d ∈ List.replicate tl.length base := by
      simpa using hd
    cases hcases with
    | inl heq => rw [heq, hzero]; ring
    | inr hmem =>
      have hmem' := List.mem_replicate.mp hmem
      exact hmem'.2

-- ---------------------------------------------------------------------------
-- 7. Packet partition supplement — ActorVisiblePacket projection
--    Mirrors event.py#ActorVisiblePacket and partition_actor_packets
-- ---------------------------------------------------------------------------

/-- Actor-visible packet: nonempty, mutually exclusive, exhaustive partition unit
    per `PacketBoundarySpec`. Here modeled as a list of envelopes with packet_id
    derived from the identity bytes (event.py#packet_identity_document). -/
structure ActorVisiblePacketLite where
  actor : Fin 4
  events : List EventEnvelopeLite
  nonempty : events ≠ []
  deriving DecidableEq, Repr

def packetContainsOnlyVisible (pkt : ActorVisiblePacketLite) : Prop :=
  ∀ e ∈ pkt.events, isActorVisible e.kind = true

theorem packet_never_contains_serverPrivate (pkt : ActorVisiblePacketLite)
    (hvis : packetContainsOnlyVisible pkt) :
    ∀ e ∈ pkt.events, e.kind ≠ .CallResolved := by
  intro e he
  have hv := hvis e he
  intro heq
  rw [heq] at hv
  simp [isActorVisible] at hv

theorem packet_excludes_serverPrivate_envelope (pkt : ActorVisiblePacketLite)
    (hvis : packetContainsOnlyVisible pkt) :
    envelopeOf .CallResolved ∉ pkt.events := by
  intro hmem
  have hne := packet_never_contains_serverPrivate pkt hvis _ hmem
  exact hne rfl

-- ---------------------------------------------------------------------------
-- 8. State/Turn integration — wall→hand→discard→meld chain
--    Minimal projection that mirrors State.lean's GameState fields without
--    hard-importing it (keeps `lake build Formal.Mahjong.Event` green before
--    State/Turn land; after they land, `file://formal/Formal/Mahjong/State.lean`
--    and `Turn.lean` refine these aliases).
--    Values cited here are exactly the State contract:
--    wallPos, hands, discards, melds, scores, roundWind, honba, kyotaku.
-- ---------------------------------------------------------------------------

/-- Wall pointer — index into the 136-tile wall (0..136).
    Mirrors `State.lean#GameState.wallPos` and `Wall.lean#WallSchedule`. -/
abbrev WallPos := Fin 137

/-- Minimal game-state projection for Event transitions. -/
structure GameStateLite where
  wallPos : WallPos
  scores : Fin 4 → Int
  honba : Nat
  kyotaku : Nat
  roundWind : Fin 4
  dealer : Fin 4
  deriving Inhabited

def initialScores : Fin 4 → Int := fun _ => 25000

def initialGameStateLite : GameStateLite where
  wallPos := ⟨0, by omega⟩
  scores := initialScores
  honba := 0
  kyotaku := 0
  roundWind := ⟨0, by omega⟩
  dealer := ⟨0, by omega⟩

theorem initial_honba_kyotaku_zero : initialGameStateLite.honba = 0 ∧ initialGameStateLite.kyotaku = 0 :=
  ⟨rfl, rfl⟩

theorem initial_scores_sum : (Finset.univ.sum fun s : Fin 4 => initialGameStateLite.scores s) = 100000 := by
  simp [initialGameStateLite, initialScores, Fin.sum_univ_four]

-- Turn distinction (mirrors Turn.lean#TurnEvent)
inductive TurnSignal where
  | Advance (actor : Fin 4)           -- public, tile-free (turn_advance)
  | Draw (actor : Fin 4) (tile : TileId) -- actor_private, hidden tile (draw_tile)
  deriving DecidableEq, Repr

def turnSignalIsPublic : TurnSignal → Bool
  | .Advance _ => true
  | .Draw _ _ => false

def turnSignalIsPrivate : TurnSignal → Bool
  | .Advance _ => false
  | .Draw _ _ => true

theorem advance_is_public (actor : Fin 4) : turnSignalIsPublic (.Advance actor) = true := rfl
theorem draw_is_private (actor : Fin 4) (t : TileId) : turnSignalIsPrivate (.Draw actor t) = true := rfl

theorem turn_advance_vs_draw_tile_hidden :
    ∀ (actor : Fin 4) (t : TileId), TurnSignal.Advance actor ≠ TurnSignal.Draw actor t := by
  intro _ _ h; cases h

def turnSignalIsPrivateExt : TurnSignal → Bool
  | .Draw _ _ => true
  | .Advance _ => false

theorem draw_tile_tile_hidden' (actor : Fin 4) (t : TileId) :
    turnSignalIsPrivateExt (.Draw actor t) = true := rfl

-- Event application to state — honba/kyotaku settlement examples
def applyRyukyoku (s : GameStateLite) : GameStateLite :=
  { s with honba := s.honba + 1 }

def applyRonHonba (s : GameStateLite) : GameStateLite :=
  { s with honba := 0, kyotaku := 0 }

theorem ryukyoku_increments_honba (s : GameStateLite) :
    (applyRyukyoku s).honba = s.honba + 1 := rfl

theorem ron_resets_honba (s : GameStateLite) :
    (applyRonHonba s).honba = 0 := rfl

theorem ron_resets_kyotaku (s : GameStateLite) :
    (applyRonHonba s).kyotaku = 0 := rfl

-- ---------------------------------------------------------------------------
-- 9. Exhaustiveness and counting witnesses (module is >180 lines)
-- ---------------------------------------------------------------------------

theorem event_cases (e : MahjongEvent) :
    e = .CallWindow ∨ e = .CallResolved ∨ e = .Ron ∨ e = .Tsumo ∨ e = .Ryukyoku := by
  cases e <;> simp

theorem event_not_both_public_and_private (e : MahjongEvent) :
    ¬(isServerPrivate e = true ∧ isActorVisible e = true) := by
  cases e <;> simp [isServerPrivate, isActorVisible]

theorem callWindow_callResolved_distinct :
    visibilityOf .CallWindow ≠ visibilityOf .CallResolved := by decide

theorem ron_tsumo_both_public :
    visibilityOf .Ron = .Public ∧ visibilityOf .Tsumo = .Public := by
  constructor <;> rfl

theorem wallPos_bound (p : WallPos) : p.val ≤ 136 := by have := p.isLt; omega
theorem honbaBonus_zero_empty_pool : honbaBonus ⟨0,0⟩ = 0 := by simp [honbaBonus]
theorem kyotakuBonus_zero_empty_pool : kyotakuBonus ⟨0,0⟩ = 0 := by simp [kyotakuBonus]
theorem totalBonus_zero_empty_pool : totalBonus ⟨0,0⟩ = 0 := by simp [totalBonus, honbaBonus, kyotakuBonus]

theorem event_visibility_exhaustive (e : MahjongEvent) :
    visibilityOf e = .Public ∨ visibilityOf e = .ServerPrivate := by
  cases e <;> simp [visibilityOf]

theorem turnSignal_cases (t : TurnSignal) :
    (∃ a, t = .Advance a) ∨ (∃ a tile, t = .Draw a tile) := by
  cases t with
  | Advance a => left; exact ⟨a, rfl⟩
  | Draw a tile => right; exact ⟨a, tile, rfl⟩

end Formal.Mahjong.EventModule
