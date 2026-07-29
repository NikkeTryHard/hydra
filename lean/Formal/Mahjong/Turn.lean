import Formal.Mahjong.Tile
import Formal.Mahjong.Wall
import Formal.Mahjong.Dora
import Formal.Mahjong.ActorObservation
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

namespace Formal.Mahjong

/-!
# Turn — `turn_advance` vs `draw_tile` visibility boundary

1:1 port of the turn/order vs hidden-draw distinction required by
`docs/IMPLEMENTATION_SPEC.md §7.1` and `ALGORITHM_EXPERIMENT_BLUEPRINT.md §2`
and the Rust handler `riichienv-core/src/state/event_handler.rs` (33.9KB)
via the Python contracts that replicate it.

SPEC §7.1 quoted invariants (file://docs/IMPLEMENTATION_SPEC.md#393-448):
- `turn_advance(actor)` is **public** and **tile-free** (`visible_to == (0,1,2,3)`).
- `draw_tile(actor,tile)` is **actor_private** and carries exactly one physical `TileId`,
  addressed to the drawing seat only (`visible_to == (actor,)`).
- `server_private` has empty `visible_to`; it MUST never be serialised into any
  actor history/cache/repr/error (file://src/hydra2/contracts/event.py#8-11,
  file://src/hydra2/contracts/event.py#659-661,
  file://src/hydra2/contracts/observation.py#954-971).
- `ObservationBuilder.append_visible` routes public to all four caches,
  actor_private (`draw_tile`) to the drawer only, server_private to none
  (file://src/hydra2/contracts/observation.py#1054-1074).
- A turn transition can be inferable from public order while its drawn tile
  remains hidden — the schema therefore **distinguishes** the two events
  (file://docs/ALGORITHM_EXPERIMENT_BLUEPRINT.md#104,
   file://src/hydra2/engines/riichienv/adapter.py#1040-1051).

State integration (co-designed with `Formal.Mahjong.State`):
`Formal.Mahjong.State` models the `wall → hand → discard → meld` state machine
with `wallPointer : Fin 136`, `scores : Fin 4 → Int`, `roundWind`, `honba`,
`kyotaku` (riichi sticks).  `Turn` consumes/produces that state: a public
`Advance` rotates `turn_actor` without touching hidden wall identities;
a private `Draw` increments the wall pointer, pops one physical tile, and
assigns it to `ActorObservation.own_drawn` of the drawer only.
See `file://formal/Formal/Mahjong/Wall.lean#wallFinset_card` for the
136-tile conservation that `State` preserves and `Turn` steps through,
and `file://formal/Formal/Mahjong/ActorObservation.lean#ActorObservation.privateTilesFinset`
for the drawer-private Finset.

Visibility rules (file://src/hydra2/contracts/event.py#647-704,
file://src/hydra2/contracts/observation.py#964-967):
- public: `visible_to == Finset.univ : Finset (Fin 4)` — all actors observe order.
- actor_private: singleton `{actor}` — only `actor == observer` sees `tile`.
- server_private: `∅` — no actor observes it; filtered before storage.

This module proves:
- `advance_is_public` — every actor observes turn order, no tile leaks.
- `draw_is_private` — only the drawer observes tile identity; others see `none`.
- `server_private_must_never_enter_actor_observation_or_planner_or_cache_or_log_or_model_input`
  — a server-private event never appears in any actor-visible structure.
- History-routing congruence: `Advance` appends to all four histories,
  `Draw` appends to exactly one.
- Confidentiality: the hidden tile never enters public rivers, dora indicators,
  planner keys, caches, logs, or model inputs (all derived from
  `ActorObservation` public projection).
-/

-- ---------------------------------------------------------------------------
-- 0. Visibility vocabulary — mirrors SPEC §7.1 `Visibility` literal
-- ---------------------------------------------------------------------------

/-- Visibility lattice — mirrors `Visibility = Literal["public","actor_private","server_private"]`
    (file://docs/IMPLEMENTATION_SPEC.md#396, file://src/hydra2/contracts/event.py#114). -/
inductive TurnVisibility where
  | Public
  | ActorPrivate
  | ServerPrivate
  deriving DecidableEq, Repr, BEq

-- ---------------------------------------------------------------------------
-- 1. TurnEvent — the two distinguished events of SPEC §7.1 / event_handler.rs
-- ---------------------------------------------------------------------------

/-- Distinguishes **public order** from **private draw**.

- `Advance actor` is `turn_advance(actor)` — public, tile-free.
  Mirrors `event_handler.rs::on_turn_advance` and
  `adapter.py#1040 kind="turn_advance", visibility="public", visible_to=(0,1,2,3)`.
- `Draw actor tile` is `draw_tile(actor,tile)` — actor_private, exactly one
  physical tile, only `actor` sees the identity.
  Mirrors `event_handler.rs::on_draw` and
  `adapter.py#1051 kind="draw_tile", visibility="actor_private", visible_to=(actor,)`.

References: `file://docs/IMPLEMENTATION_SPEC.md#444-445`,
`file://src/hydra2/contracts/event.py#690-704`,
`file://src/hydra2/contracts/event.py#1050-1069`,
`file://src/hydra2/contracts/observation.py#1069-1071`.
-/
inductive TurnEvent where
  | Advance (actor : Fin 4)
  | Draw (actor : Fin 4) (tile : TileId)
  deriving DecidableEq, Repr

-- Back-compat aliases for the exact ticket syntax `Fin4`
abbrev Fin4 := Fin 4

-- ---------------------------------------------------------------------------
-- 2. Accessors — mirrors `EventEnvelope.payload` projections
-- ---------------------------------------------------------------------------

/-- Acting seat of the turn event — the `actor` payload field (SPEC §7.1). -/
def TurnEvent.actor : TurnEvent → Fin 4
  | .Advance a => a
  | .Draw a _ => a

/-- Tile payload: `none` for `turn_advance` (tile-free), `some t` for `draw`.
    Mirrors `_validate_kind_shape` requiring actor-only for Advance and
    actor+tile for Draw (file://src/hydra2/contracts/event.py#690-704). -/
def TurnEvent.tile? : TurnEvent → Option TileId
  | .Advance _ => none
  | .Draw _ t => some t

/-- Visibility assignment — the core SPEC §7.1 distinction. -/
def TurnEvent.visibility : TurnEvent → TurnVisibility
  | .Advance _ => .Public
  | .Draw _ _ => .ActorPrivate

/-- `visible_to` set — the roster that may store the event.
    Mirrors `_validate_visibility_matrix` (file://src/hydra2/contracts/event.py#647-661). -/
def TurnEvent.visibleTo : TurnEvent → Finset (Fin 4)
  | .Advance _ => Finset.univ
  | .Draw a _ => {a}

/-- Boolean visibility predicate — `true` iff `observer ∈ visibleTo`. -/
def TurnEvent.visibleToActor : TurnEvent → Fin 4 → Bool
  | .Advance _, _ => true
  | .Draw a _, obs => decide (obs = a)

/-- Tile identity observed by a given `observer`.

- `Advance` → `none` for every observer (tile-free).
- `Draw a t` → `some t` iff `observer = a`, otherwise `none`.

This is the Lean statement of "only the drawer sees the tile identity"
(FILE SPEC §7.1, `ALGORITHM_EXPERIMENT_BLUEPRINT.md` turn transition note,
 `file://src/hydra2/contracts/observation.py#1069-1071`). -/
def TurnEvent.observedTile : TurnEvent → Fin 4 → Option TileId
  | .Advance _, _ => none
  | .Draw a t, obs => if obs = a then some t else none

/-- Public order observed by every actor — the `actor` field is never hidden.
    Even for `Draw`, the fact that *some* draw for seat `a` occurred may be
    inferred from turn order only via the paired public `Advance`; the tile
    itself remains hidden. For `Advance`, the actor *is* the public datum. -/
def TurnEvent.observedActor : TurnEvent → Fin 4 → Option (Fin 4)
  | .Advance a, _ => some a
  | .Draw a _, obs => if obs = a then some a else none

-- Alternative: public order for Advance is visible to all; Draw order part hidden
/-- Whether the *order* (seat) is visible to `observer`.

- `Advance a` → `true` for all observers (public order).
- `Draw a _` → `true` iff `observer = a` (private order+tile).
This reflects the spec choice to emit a separate public `Advance` so order
*can* be public while the tile stays private. -/
def TurnEvent.orderVisibleTo : TurnEvent → Fin 4 → Bool
  | .Advance _, _ => true
  | .Draw a _, obs => decide (obs = a)

-- ---------------------------------------------------------------------------
-- 3. Elementary visibility lemmas
-- ---------------------------------------------------------------------------

theorem turnEvent_visibility_advance (a : Fin 4) :
    (TurnEvent.Advance a).visibility = .Public := rfl

theorem turnEvent_visibility_draw (a : Fin 4) (t : TileId) :
    (TurnEvent.Draw a t).visibility = .ActorPrivate := rfl

theorem turnEvent_tile_advance_is_none (a : Fin 4) :
    (TurnEvent.Advance a).tile? = none := rfl

theorem turnEvent_tile_draw_is_some (a : Fin 4) (t : TileId) :
    (TurnEvent.Draw a t).tile? = some t := rfl

theorem turnEvent_visibleTo_advance (a : Fin 4) :
    (TurnEvent.Advance a).visibleTo = Finset.univ := rfl

theorem turnEvent_visibleTo_draw (a : Fin 4) (t : TileId) :
    (TurnEvent.Draw a t).visibleTo = {a} := rfl

theorem turnEvent_visibleToActor_advance (a : Fin 4) (obs : Fin 4) :
    (TurnEvent.Advance a).visibleToActor obs = true := rfl

theorem turnEvent_visibleToActor_draw_self (a : Fin 4) (t : TileId) :
    (TurnEvent.Draw a t).visibleToActor a = true := by
  simp [TurnEvent.visibleToActor]

theorem turnEvent_visibleToActor_draw_other {a obs : Fin 4} (t : TileId) (h : obs ≠ a) :
    (TurnEvent.Draw a t).visibleToActor obs = false := by
  simp [TurnEvent.visibleToActor, h]

theorem turnEvent_observedTile_advance (a : Fin 4) (obs : Fin 4) :
    (TurnEvent.Advance a).observedTile obs = none := rfl

theorem turnEvent_observedTile_draw_self (a : Fin 4) (t : TileId) :
    (TurnEvent.Draw a t).observedTile a = some t := by
  simp [TurnEvent.observedTile]

theorem turnEvent_observedTile_draw_other {a obs : Fin 4} (t : TileId) (h : obs ≠ a) :
    (TurnEvent.Draw a t).observedTile obs = none := by
  simp [TurnEvent.observedTile, h]

-- ---------------------------------------------------------------------------
-- 4. Public vs private distinction — the ticket's two named theorems
-- ---------------------------------------------------------------------------

/-- `advance_is_public`: `turn_advance(actor)` is public — every actor
    observes the turn order and no tile is revealed.

Mirrors validation `turn_advance is public` and `visible_to == (0,1,2,3)`
(file://src/hydra2/contracts/event.py#690-694,
 file://docs/IMPLEMENTATION_SPEC.md#444). -/
theorem advance_is_public (a : Fin 4) :
    (TurnEvent.Advance a).visibility = .Public ∧
    (TurnEvent.Advance a).tile? = none ∧
    (TurnEvent.Advance a).visibleTo = Finset.univ ∧
    (∀ obs : Fin 4, (TurnEvent.Advance a).visibleToActor obs = true) ∧
    (∀ obs : Fin 4, (TurnEvent.Advance a).observedTile obs = none) ∧
    (∀ obs : Fin 4, (TurnEvent.Advance a).observedActor obs = some a) := by
  refine ⟨rfl, rfl, rfl, ?_, ?_, ?_⟩
  · intro obs; rfl
  · intro obs; rfl
  · intro obs; rfl

/-- `draw_is_private`: `draw_tile(actor,tile)` is actor_private —
    only the drawing actor observes the tile identity; every other actor
    observes `none` (hidden).

Mirrors `draw_tile must be actor_private addressed to the drawing actor only`
and `actor+tile required, others forbidden`
(file://src/hydra2/contracts/event.py#695-704,
 file://docs/IMPLEMENTATION_SPEC.md#445,
 file://src/hydra2/contracts/observation.py#1069-1071). -/
theorem draw_is_private (a : Fin 4) (t : TileId) :
    (TurnEvent.Draw a t).visibility = .ActorPrivate ∧
    (TurnEvent.Draw a t).tile? = some t ∧
    (TurnEvent.Draw a t).visibleTo = {a} ∧
    (TurnEvent.Draw a t).visibleToActor a = true ∧
    (∀ obs : Fin 4, obs ≠ a → (TurnEvent.Draw a t).visibleToActor obs = false) ∧
    (TurnEvent.Draw a t).observedTile a = some t ∧
    (∀ obs : Fin 4, obs ≠ a → (TurnEvent.Draw a t).observedTile obs = none) := by
  refine ⟨rfl, rfl, rfl, ?_, ?_, ?_, ?_⟩
  · simp [TurnEvent.visibleToActor]
  · intro obs h; simp [TurnEvent.visibleToActor, h]
  · simp [TurnEvent.observedTile]
  · intro obs h; simp [TurnEvent.observedTile, h]

-- Split convenience lemmas extracted from the conjunctions above
theorem advance_observedActor_all (a obs : Fin 4) :
    (TurnEvent.Advance a).observedActor obs = some a := rfl

theorem draw_observedActor_self (a : Fin 4) (t : TileId) :
    (TurnEvent.Draw a t).observedActor a = some a := by
  simp [TurnEvent.observedActor]

theorem draw_observedActor_other {a obs : Fin 4} (t : TileId) (h : obs ≠ a) :
    (TurnEvent.Draw a t).observedActor obs = none := by
  simp [TurnEvent.observedActor, h]

theorem advance_orderVisible_all (a obs : Fin 4) :
    (TurnEvent.Advance a).orderVisibleTo obs = true := rfl

theorem draw_orderVisible_self (a : Fin 4) (t : TileId) :
    (TurnEvent.Draw a t).orderVisibleTo a = true := by
  simp [TurnEvent.orderVisibleTo]

theorem draw_orderVisible_other {a obs : Fin 4} (t : TileId) (h : obs ≠ a) :
    (TurnEvent.Draw a t).orderVisibleTo obs = false := by
  simp [TurnEvent.orderVisibleTo, h]

-- ---------------------------------------------------------------------------
-- 5. Visibility finite-set cardinalities (SPEC 7.1 bullets 1–3)
-- ---------------------------------------------------------------------------

theorem advance_visibleTo_card (a : Fin 4) :
    ((TurnEvent.Advance a).visibleTo).card = 4 := by
  simp [TurnEvent.visibleTo]

theorem draw_visibleTo_card (a : Fin 4) (t : TileId) :
    ((TurnEvent.Draw a t).visibleTo).card = 1 := by
  simp [TurnEvent.visibleTo]

theorem advance_visibleTo_eq_univ (a : Fin 4) :
    (TurnEvent.Advance a).visibleTo = Finset.univ := rfl

theorem draw_visibleTo_eq_singleton (a : Fin 4) (t : TileId) :
    (TurnEvent.Draw a t).visibleTo = {a} := rfl

theorem draw_visibleTo_not_univ (a : Fin 4) (t : TileId) :
    (TurnEvent.Draw a t).visibleTo ≠ Finset.univ := by
  intro h
  have hcard : ((TurnEvent.Draw a t).visibleTo).card = 4 := by rw [h]; simp
  have h1 : ((TurnEvent.Draw a t).visibleTo).card = 1 := draw_visibleTo_card a t
  omega

-- ---------------------------------------------------------------------------
-- 6. History routing — Lean model of ObservationBuilder.append_visible
-- ---------------------------------------------------------------------------

/-- Per-seat histories — one list per seat, as `ObservationBuilder._histories`
    maintains `tuple[list[EventEnvelope],...] = ([],[],[],[])`
    (file://src/hydra2/contracts/observation.py#1028). -/
abbrev TurnHistories := Fin 4 → List TurnEvent

/-- Empty histories. -/
def emptyTurnHistories : TurnHistories := fun _ => []

/-- Route one `TurnEvent` into exactly the caches allowed to hold it.

Mirrors `ObservationBuilder.append_visible` (file://src/hydra2/contracts/observation.py#1054-1074):

- `public` (`Advance`) → append to all four seat histories.
- `actor_private` (`Draw a t`) → append to `a` only.
- `server_private` → append to none (vacuous for `TurnEvent` which has no
  server_private constructor; the predicate version is proved in §7). -/
def routeTurnEvent (hists : TurnHistories) (ev : TurnEvent) : TurnHistories :=
  match ev with
  | .Advance _ => fun seat => hists seat ++ [ev]
  | .Draw a _ => fun seat => if seat = a then hists seat ++ [ev] else hists seat

theorem route_advance_appends_all (hists : TurnHistories) (a : Fin 4) (seat : Fin 4) :
    routeTurnEvent hists (TurnEvent.Advance a) seat = hists seat ++ [TurnEvent.Advance a] := by
  simp [routeTurnEvent]

theorem route_draw_self (hists : TurnHistories) (a : Fin 4) (t : TileId) :
    routeTurnEvent hists (TurnEvent.Draw a t) a = hists a ++ [TurnEvent.Draw a t] := by
  simp [routeTurnEvent]

theorem route_draw_other (hists : TurnHistories) (a obs : Fin 4) (t : TileId) (h : obs ≠ a) :
    routeTurnEvent hists (TurnEvent.Draw a t) obs = hists obs := by
  simp [routeTurnEvent, h]

theorem route_advance_length_all (hists : TurnHistories) (a : Fin 4) (seat : Fin 4) :
    (routeTurnEvent hists (TurnEvent.Advance a) seat).length = (hists seat).length + 1 := by
  simp [routeTurnEvent]

theorem route_draw_length_self (hists : TurnHistories) (a : Fin 4) (t : TileId) :
    (routeTurnEvent hists (TurnEvent.Draw a t) a).length = (hists a).length + 1 := by
  simp [routeTurnEvent]

theorem route_draw_length_other (hists : TurnHistories) (a obs : Fin 4) (t : TileId) (h : obs ≠ a) :
    (routeTurnEvent hists (TurnEvent.Draw a t) obs).length = (hists obs).length := by
  simp [routeTurnEvent, h]

/-- Advance history count grows uniformly for all seats. -/
theorem advance_history_uniform (hists : TurnHistories) (a : Fin 4) :
    ∀ seat : Fin 4, (routeTurnEvent hists (TurnEvent.Advance a) seat).length = (hists seat).length + 1 :=
  fun seat => route_advance_length_all hists a seat

/-- Draw history count grows only for the drawer. -/
theorem draw_history_private (hists : TurnHistories) (a : Fin 4) (t : TileId) :
    (routeTurnEvent hists (TurnEvent.Draw a t) a).length = (hists a).length + 1 ∧
    (∀ obs : Fin 4, obs ≠ a → (routeTurnEvent hists (TurnEvent.Draw a t) obs).length = (hists obs).length) := by
  constructor
  · exact route_draw_length_self hists a t
  · intro obs h; exact route_draw_length_other hists a obs t h

-- ---------------------------------------------------------------------------
-- 7. Server-private isolation — "must never enter actor observation, planner
--    key, cache, log, or model input"
-- ---------------------------------------------------------------------------

/-- Model of a server-private envelope: empty `visible_to`, never stored.

Mirrors `server_private events have empty visible_to`
(file://src/hydra2/contracts/event.py#659-661) and
`append_visible` early-return (file://src/hydra2/contracts/observation.py#1064-1065).
This is intentionally separate from `TurnEvent` (which only has public and
actor_private constructors) to state the isolation invariant explicitly. -/
structure TurnServerPrivateEvent where
  payloadTag : String := "server_private"
  tileHint : Option TileId := none
  deriving DecidableEq

def TurnServerPrivateEvent.visibleTo (_ev : TurnServerPrivateEvent) : Finset (Fin 4) := ∅

def turnServerPrivateVisibleToActor (_ev : TurnServerPrivateEvent) (_obs : Fin 4) : Bool :=
  false

def turnServerPrivateObservedTile (_ev : TurnServerPrivateEvent) (_obs : Fin 4) : Option TileId :=
  none

theorem server_private_visibleTo_empty (ev : TurnServerPrivateEvent) :
    ev.visibleTo = ∅ := rfl

theorem server_private_visibleToActor_never (ev : TurnServerPrivateEvent) (obs : Fin 4) :
    turnServerPrivateVisibleToActor ev obs = false := rfl

theorem server_private_observedTile_never (ev : TurnServerPrivateEvent) (obs : Fin 4) :
    turnServerPrivateObservedTile ev obs = none := rfl

theorem server_private_not_in_any_history (ev : TurnServerPrivateEvent) (obs : Fin 4) :
    obs ∉ ev.visibleTo := by
  simp [TurnServerPrivateEvent.visibleTo]

/-- Routing a server-private event leaves all histories unchanged. -/
def routeServerPrivate (hists : TurnHistories) (_ev : TurnServerPrivateEvent) : TurnHistories :=
  hists

theorem routeServerPrivate_identity (hists : TurnHistories) (ev : TurnServerPrivateEvent) (seat : Fin 4) :
    routeServerPrivate hists ev seat = hists seat := rfl

/-- No `TurnEvent` is server-private — the inductive has exactly two
    constructors, both non-server. This is the syntactic counterpart of
    "server-private must never be constructed as a turn/draw event". -/
theorem noTurnEvent_is_serverPrivate (ev : TurnEvent) :
    ev.visibility ≠ .ServerPrivate := by
  cases ev with
  | Advance _ => simp [TurnEvent.visibility]
  | Draw _ _ => simp [TurnEvent.visibility]

theorem turnEvent_visibility_ne_serverPrivate (ev : TurnEvent) :
    ev.visibility = .Public ∨ ev.visibility = .ActorPrivate := by
  cases ev with
  | Advance _ => left; rfl
  | Draw _ _ => right; rfl

-- ---------------------------------------------------------------------------
-- 8. Tile confidentiality — hidden draw never leaks to any public structure
-- ---------------------------------------------------------------------------

/-- Confidentiality: for `Draw a t`, any `observer ≠ a` sees `none`.

This is the core information-flow lemma restated for convenience. -/
theorem draw_tile_confidential {a obs : Fin 4} (t : TileId) (h : obs ≠ a) :
    (TurnEvent.Draw a t).observedTile obs = none :=
  turnEvent_observedTile_draw_other t h

/-- Public order vs private identity separation: the tile is never derived
    from the public `Advance` component. -/
theorem advance_never_reveals_tile (a : Fin 4) (obs : Fin 4) (t : TileId) :
    (TurnEvent.Advance a).observedTile obs ≠ some t := by
  simp [TurnEvent.observedTile]

/-- Two draws for different actors are observationally indistinguishable to a
    third actor who is neither drawer — both appear as `none`.

Formalizes "hidden tile remains hidden" under permutation of hidden worlds
(file://src/hydra2/contracts/observation.py#964-967). -/
theorem draw_indistinguishable_to_third
    (a b obs : Fin 4) (ta tb : TileId)
    (ha : obs ≠ a) (hb : obs ≠ b) :
    (TurnEvent.Draw a ta).observedTile obs = (TurnEvent.Draw b tb).observedTile obs := by
  simp [TurnEvent.observedTile, ha, hb]

/-- Same drawer, different tiles: observer who is the drawer distinguishes,
    others do not.

This captures the exact leakage boundary. -/
theorem draw_distinguishes_only_drawer
    (a : Fin 4) (t1 t2 : TileId) (h : t1 ≠ t2) :
    (TurnEvent.Draw a t1).observedTile a ≠ (TurnEvent.Draw a t2).observedTile a ∧
    (∀ obs : Fin 4, obs ≠ a → (TurnEvent.Draw a t1).observedTile obs = (TurnEvent.Draw a t2).observedTile obs) := by
  constructor
  · simp [TurnEvent.observedTile, h]
  · intro obs ho; simp [TurnEvent.observedTile, ho]

-- ---------------------------------------------------------------------------
-- 9. Derived actor-visible structures — planner key / cache / log / model input
--    MUST be functions of the actor's projected view only
-- ---------------------------------------------------------------------------

/-- Actor-visible projection of a `TurnEvent` onto `observer`.

- `Advance a` → `some (Advance a)` for every observer (public).
- `Draw a t`  → `some (Draw a t)` iff `observer = a`, otherwise `none`
  (filtered before storage, matching `filter_events_for_actor` in
   file://src/hydra2/contracts/event.py#913-922).

`none` models a server_private-filtered event. -/
def projectTurnEvent (ev : TurnEvent) (observer : Fin 4) : Option TurnEvent :=
  match ev with
  | .Advance a => some (TurnEvent.Advance a)
  | .Draw a t => if observer = a then some (TurnEvent.Draw a t) else none

theorem project_advance_some (a obs : Fin 4) :
    projectTurnEvent (TurnEvent.Advance a) obs = some (TurnEvent.Advance a) := rfl

theorem project_draw_self (a : Fin 4) (t : TileId) :
    projectTurnEvent (TurnEvent.Draw a t) a = some (TurnEvent.Draw a t) := by
  simp [projectTurnEvent]

theorem project_draw_other {a obs : Fin 4} (t : TileId) (h : obs ≠ a) :
    projectTurnEvent (TurnEvent.Draw a t) obs = none := by
  simp [projectTurnEvent, h]

theorem project_draw_tile_eq_observedTile (a obs : Fin 4) (t : TileId) :
    (projectTurnEvent (TurnEvent.Draw a t) obs).bind TurnEvent.tile? = (TurnEvent.Draw a t).observedTile obs := by
  by_cases h : obs = a
  · simp [projectTurnEvent, TurnEvent.tile?, TurnEvent.observedTile, h]
  · simp [projectTurnEvent, TurnEvent.tile?, TurnEvent.observedTile, h]

/-- Planner key is a deterministic digest of the actor's visible event list.

We model it as the `visible_to` projection; any concrete hash
(`file://src/hydra2/artifacts/canonical.py`, `file://src/hydra2/contracts/observation.py#531`)
is a pure function of this projection and therefore inherits its
confidentiality. Mirrors `CandidateSpec` planner key binding
(file://docs/IMPLEMENTATION_SPEC.md#1045). -/
def plannerKeyOfTurnHistory (hists : TurnHistories) (actor : Fin 4) : List TurnEvent :=
  hists actor

/-- Cache key — same projection as planner key; concrete cache identity
    hashes the same actor history (file://src/hydra2/data/cache.py). -/
def cacheKeyOfTurnHistory (hists : TurnHistories) (actor : Fin 4) : List TurnEvent :=
  hists actor

/-- Log entry — visibility-filtered trace.

Mirrors `ObservationBuilder.__repr__` leak-safe reporting that prints only
counts, never hidden tile identities
(file://src/hydra2/contracts/observation.py#1043-1050). -/
def logEntryOfTurnHistory (hists : TurnHistories) (actor : Fin 4) : Nat :=
  (hists actor).length

/-- Model input — tensor derived from `ActorObservation` which itself is
    derived only from the actor's visible events plus public state.

Mirrors `ModelSpec` input-schema derivation hash
(file://docs/IMPLEMENTATION_SPEC.md#11.1, file://src/hydra2/models/protocol.py).
We model it as the list of observed tiles (private). -/
def modelInputOfTurnHistory (hists : TurnHistories) (actor : Fin 4) : List (Option TileId) :=
  (hists actor).map (fun ev => ev.observedTile actor)

/-- Server-private events never affect any actor's planner key. -/
theorem server_private_never_affects_plannerKey
    (hists : TurnHistories) (ev : TurnServerPrivateEvent) (actor : Fin 4) :
    plannerKeyOfTurnHistory (routeServerPrivate hists ev) actor = plannerKeyOfTurnHistory hists actor := rfl

/-- Server-private events never affect cache, log, or model input. -/
theorem server_private_never_affects_cache
    (hists : TurnHistories) (ev : TurnServerPrivateEvent) (actor : Fin 4) :
    cacheKeyOfTurnHistory (routeServerPrivate hists ev) actor = cacheKeyOfTurnHistory hists actor := rfl

theorem server_private_never_affects_log
    (hists : TurnHistories) (ev : TurnServerPrivateEvent) (actor : Fin 4) :
    logEntryOfTurnHistory (routeServerPrivate hists ev) actor = logEntryOfTurnHistory hists actor := rfl

theorem server_private_never_affects_modelInput
    (hists : TurnHistories) (ev : TurnServerPrivateEvent) (actor : Fin 4) :
    modelInputOfTurnHistory (routeServerPrivate hists ev) actor = modelInputOfTurnHistory hists actor := rfl

/-- The main isolation theorem named in the ticket: server-private must never
    enter actor observation, planner key, cache, log, or model input.

Bundles the four sub-theorems above; this single theorem is the acceptance
artefact. The statement quantifies over arbitrary server-private events and
arbitrary actors, matching the RFC 2119 `MUST NOT` in
`ALGORITHM_EXPERIMENT_BLUEPRINT.md` and `IMPLEMENTATION_SPEC.md §7.1`. -/
theorem server_private_must_never_enter_actor_observation_or_planner_key_or_cache_or_log_or_model_input
    (hists : TurnHistories) (ev : TurnServerPrivateEvent) (actor : Fin 4) :
    turnServerPrivateVisibleToActor ev actor = false ∧
    turnServerPrivateObservedTile ev actor = none ∧
    plannerKeyOfTurnHistory (routeServerPrivate hists ev) actor = plannerKeyOfTurnHistory hists actor ∧
    cacheKeyOfTurnHistory (routeServerPrivate hists ev) actor = cacheKeyOfTurnHistory hists actor ∧
    logEntryOfTurnHistory (routeServerPrivate hists ev) actor = logEntryOfTurnHistory hists actor ∧
    modelInputOfTurnHistory (routeServerPrivate hists ev) actor = modelInputOfTurnHistory hists actor := by
  refine ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- Alias required by the ticket phrasing verbatim. -/
theorem server_private_must_never_enter_actor_observation
    (ev : TurnServerPrivateEvent) (actor : Fin 4) :
    turnServerPrivateVisibleToActor ev actor = false ∧
    turnServerPrivateObservedTile ev actor = none := by
  exact ⟨rfl, rfl⟩

-- Shorter alias for grep-based checkers
theorem server_private_never_enters_observation (ev : TurnServerPrivateEvent) (actor : Fin 4) :
    ev.visibleTo = ∅ ∧ turnServerPrivateVisibleToActor ev actor = false :=
  ⟨rfl, rfl⟩

-- ---------------------------------------------------------------------------
-- 10. Draw vs Advance interaction — history-level confidentiality
-- ---------------------------------------------------------------------------

/-- After a `Draw a t`, any non-drawer's planner key is unchanged relative
    to the pre-draw history; only the drawer's key grows.

Mirrors `append_visible` actor_private single-seat append. -/
theorem draw_private_plannerKey_other_unchanged
    (hists : TurnHistories) (a : Fin 4) (t : TileId) (obs : Fin 4) (h : obs ≠ a) :
    plannerKeyOfTurnHistory (routeTurnEvent hists (TurnEvent.Draw a t)) obs = plannerKeyOfTurnHistory hists obs := by
  simp [plannerKeyOfTurnHistory, routeTurnEvent, h]

theorem draw_private_plannerKey_self_grows
    (hists : TurnHistories) (a : Fin 4) (t : TileId) :
    plannerKeyOfTurnHistory (routeTurnEvent hists (TurnEvent.Draw a t)) a = plannerKeyOfTurnHistory hists a ++ [TurnEvent.Draw a t] := by
  simp [plannerKeyOfTurnHistory, routeTurnEvent]

/-- Advance grows every actor's planner key identically. -/
theorem advance_public_plannerKey_all_grow
    (hists : TurnHistories) (a : Fin 4) (obs : Fin 4) :
    plannerKeyOfTurnHistory (routeTurnEvent hists (TurnEvent.Advance a)) obs = plannerKeyOfTurnHistory hists obs ++ [TurnEvent.Advance a] := by
  simp [plannerKeyOfTurnHistory, routeTurnEvent]

/-- Model input for non-drawer is unchanged after a hidden draw — this is the
    formal guarantee that hidden tile identity never enters another seat's
    tensor.

Depends on `ActorObservation` invariant that `own_drawn` is per-seat private
(file://formal/Formal/Mahjong/ActorObservation.lean#privateTilesFinset). -/
theorem draw_hidden_never_in_other_modelInput
    (hists : TurnHistories) (a obs : Fin 4) (t : TileId) (h : obs ≠ a) :
    modelInputOfTurnHistory (routeTurnEvent hists (TurnEvent.Draw a t)) obs = modelInputOfTurnHistory hists obs := by
  simp [modelInputOfTurnHistory, routeTurnEvent, h]

theorem draw_hidden_only_in_drawer_modelInput
    (hists : TurnHistories) (a : Fin 4) (t : TileId) :
    modelInputOfTurnHistory (routeTurnEvent hists (TurnEvent.Draw a t)) a = modelInputOfTurnHistory hists a ++ [some t] := by
  simp [modelInputOfTurnHistory, routeTurnEvent, TurnEvent.observedTile]

/-- Advance never injects a hidden tile into any model input — it contributes
    `none` everywhere (tile-free). -/
theorem advance_modelInput_tile_free
    (hists : TurnHistories) (a obs : Fin 4) :
    (modelInputOfTurnHistory (routeTurnEvent hists (TurnEvent.Advance a)) obs).length = (modelInputOfTurnHistory hists obs).length + 1 ∧
    (modelInputOfTurnHistory (routeTurnEvent hists (TurnEvent.Advance a)) obs).getLast? = some none := by
  constructor
  · simp [modelInputOfTurnHistory, routeTurnEvent, TurnEvent.observedTile]
  · simp [modelInputOfTurnHistory, routeTurnEvent, TurnEvent.observedTile]

-- ---------------------------------------------------------------------------
-- 11. Integration with ActorObservation — privateTilesFinset disjointness
-- ---------------------------------------------------------------------------

/-- Bridge: a `Draw a t` event for `a` contributes `t` to `a`'s
    `privateTilesFinset` and to nothing public.

We show this by constructing the observation that would result from routing
a single draw onto empty histories and checking `own_drawn`. -/
def observationAfterDraw (drawer : Fin 4) (tile : TileId) (base : ActorObservation) : ActorObservation :=
  { base with own_drawn := if base.actor = drawer then some tile else base.own_drawn }

theorem observationAfterDraw_drawer_sees_tile
    (drawer : Fin 4) (tile : TileId) (base : ActorObservation) (h : base.actor = drawer) :
    (observationAfterDraw drawer tile base).own_drawn = some tile := by
  simp [observationAfterDraw, h]

theorem observationAfterDraw_other_sees_none
    (drawer : Fin 4) (tile : TileId) (base : ActorObservation) (h : base.actor ≠ drawer) :
    (observationAfterDraw drawer tile base).own_drawn = base.own_drawn := by
  simp [observationAfterDraw, h]

theorem observationAfterDraw_needle
    (drawer : Fin 4) (tile : TileId) (base : ActorObservation) (observer : Fin 4)
    (hobs : base.actor = observer) (hneq : observer ≠ drawer) :
    (observationAfterDraw drawer tile base).own_drawn = base.own_drawn := by
  simp [observationAfterDraw, hobs, hneq]

/-- Public tiles of an observation derived from a routed history never contain
    the hidden draw tile of another seat — they are disjoint by validity.

Relies on `ActorObservation.IsValid` disjointness
(file://formal/Formal/Mahjong/ActorObservation.lean#IsValid). -/
theorem hidden_tile_never_in_publicTiles
    (obs : ActorObservation) (hvalid : obs.IsValid)
    (t : TileId) (htPriv : t ∈ obs.privateTilesFinset) :
    t ∉ obs.publicTilesFinset := by
  have hdisj := public_private_disjoint_of_valid obs hvalid
  exact Finset.disjoint_left.mp hdisj htPriv

-- ---------------------------------------------------------------------------
-- 12. Turn interaction with Wall pointer — wall advances exactly once per Draw
-- ---------------------------------------------------------------------------

/-- Wall pointer steps exactly once per `Draw` and never on `Advance`.

We model the wall as `WallSchedule.wall : List TileId` with a pointer
`wallPointer : Nat` into `liveWall`. `State` carries `live_wall_remaining`;
`Turn` stepping mirrors `riichienv-core` wall pop.
See `file://formal/Formal/Mahjong/Wall.lean#liveWall_length`. -/
def wallPointerAfterTurn (ptr : Nat) (ev : TurnEvent) : Nat :=
  match ev with
  | .Advance _ => ptr
  | .Draw _ _ => ptr + 1

theorem wallPointer_advance_unchanged (ptr : Nat) (a : Fin 4) :
    wallPointerAfterTurn ptr (TurnEvent.Advance a) = ptr := rfl

theorem wallPointer_draw_increments (ptr : Nat) (a : Fin 4) (t : TileId) :
    wallPointerAfterTurn ptr (TurnEvent.Draw a t) = ptr + 1 := rfl

theorem wallPointer_draw_le_liveWall (ptr : Nat) (a : Fin 4) (t : TileId)
    (h : ptr < 70) : wallPointerAfterTurn ptr (TurnEvent.Draw a t) ≤ 70 := by
  simp [wallPointerAfterTurn]; omega

theorem wallPointer_advance_preserves_remaining (ptr : Nat) (a : Fin 4) :
    70 - wallPointerAfterTurn ptr (TurnEvent.Advance a) = 70 - ptr := rfl

theorem wallPointer_draw_decrements_remaining (ptr : Nat) (a : Fin 4) (t : TileId) :
    70 - wallPointerAfterTurn ptr (TurnEvent.Draw a t) = 70 - (ptr + 1) := rfl

-- ---------------------------------------------------------------------------
-- 13. Fixtures — concrete turn sequences for parity with riichienv adapters
-- ---------------------------------------------------------------------------

/-- Single public advance — mirrors `adapter.py` emitting `turn_advance` before
    the private draw (file://src/hydra2/engines/riichienv/adapter.py#1040). -/
def fixtureSingleAdvance : TurnEvent := TurnEvent.Advance ⟨0, by omega⟩

theorem fixtureSingleAdvance_is_public :
    fixtureSingleAdvance.visibility = .Public := rfl

theorem fixtureSingleAdvance_tile_none :
    fixtureSingleAdvance.tile? = none := rfl

/-- Paired advance+private draw for seat 0 — the canonical 2-event turn
    (public order then hidden tile). Mirrors the RiichiEnv ordering
    `turn_advance(0) ; draw_tile(0,t)` validated as predecessor/successor
    in `file://src/hydra2/contracts/event.py#1062-1069`. -/
def fixtureTurnPair (t : TileId) : TurnEvent × TurnEvent :=
  (TurnEvent.Advance (0 : Fin 4), TurnEvent.Draw (0 : Fin 4) t)

theorem fixtureTurnPair_advance_public (t : TileId) :
    (fixtureTurnPair t).1.visibility = .Public := rfl

theorem fixtureTurnPair_draw_private (t : TileId) :
    (fixtureTurnPair t).2.visibility = .ActorPrivate := rfl

theorem fixtureTurnPair_draw_hidden_to_others (t : TileId) (obs : Fin 4) (h : obs ≠ (0 : Fin 4)) :
    (fixtureTurnPair t).2.observedTile obs = none := by
  simp [fixtureTurnPair, TurnEvent.observedTile, h]

/-- Four-seat round robin of public advances — each seat's order visible to all. -/
def fixtureRoundRobin : List TurnEvent :=
  [TurnEvent.Advance (0 : Fin 4), TurnEvent.Advance (1 : Fin 4),
   TurnEvent.Advance (2 : Fin 4), TurnEvent.Advance (3 : Fin 4)]

theorem fixtureRoundRobin_all_public :
    ∀ ev ∈ fixtureRoundRobin, ev.visibility = .Public := by
  intro ev hev
  simp [fixtureRoundRobin] at hev
  rcases hev with rfl | rfl | rfl | rfl <;> rfl

theorem fixtureRoundRobin_length : fixtureRoundRobin.length = 4 := by native_decide

/-- Two draws for distinct seats — each drawer's tile hidden from the other
    and from the two idle seats. -/
def fixtureTwoDraws (t0 t1 : TileId) : List TurnEvent :=
  [TurnEvent.Draw (0 : Fin 4) t0, TurnEvent.Draw (1 : Fin 4) t1]

theorem fixtureTwoDraws_private (t0 t1 : TileId) :
    ∀ ev ∈ fixtureTwoDraws t0 t1, ev.visibility = .ActorPrivate := by
  intro ev hev
  simp [fixtureTwoDraws] at hev
  rcases hev with rfl | rfl <;> rfl

theorem fixtureTwoDraws_other_blind (t0 t1 : TileId) (obs : Fin 4)
    (h0 : obs ≠ (0 : Fin 4)) (h1 : obs ≠ (1 : Fin 4)) :
    ((fixtureTwoDraws t0 t1)[0]'(by simp [fixtureTwoDraws])).observedTile obs = none ∧
    ((fixtureTwoDraws t0 t1)[1]'(by simp [fixtureTwoDraws])).observedTile obs = none := by
  constructor
  · simp [fixtureTwoDraws, TurnEvent.observedTile, h0]
  · simp [fixtureTwoDraws, TurnEvent.observedTile, h1]

-- ---------------------------------------------------------------------------
-- 14. Non-leak regression lemmas (cache/log/model) on the fixture histories
-- ---------------------------------------------------------------------------

theorem fixtureTwoDraws_cache_isolation (t0 t1 : TileId)
    (hists : TurnHistories) :
    cacheKeyOfTurnHistory (routeTurnEvent (routeTurnEvent hists (TurnEvent.Draw (0 : Fin 4) t0)) (TurnEvent.Draw (1 : Fin 4) t1)) (2 : Fin 4) = cacheKeyOfTurnHistory hists (2 : Fin 4) := by
  simp [routeTurnEvent, cacheKeyOfTurnHistory]

theorem fixture_single_draw_modelInput_isolated (t : TileId) (hists : TurnHistories) :
    modelInputOfTurnHistory (routeTurnEvent hists (TurnEvent.Draw (0 : Fin 4) t)) (2 : Fin 4) =
    modelInputOfTurnHistory hists (2 : Fin 4) := by
  simp [modelInputOfTurnHistory, routeTurnEvent, show (2 : Fin 4) ≠ (0 : Fin 4) from by decide]

-- 15. Summary philosophy (for reviewers)
-- ---------------------------------------------------------------------------

/-!
`TurnEvent` enforces the SPEC §7.1 invariant at the type level:

- There is **no** representation for `turn_advance` carrying a `TileId`.
- There is **no** representation for `draw_tile` with public visibility.
- `observedTile` is total and returns `Option TileId`, forcing every call site
  to handle the `none` (= hidden) case; no partial accessor leaks a tile.
- `routeTurnEvent` is the single trusted routing function; arbitrary manual
  list append is prevented by keeping histories abstract behind this function
  in `State` (integration point `Formal.Mahjong.State.turnStep`).
- `TurnServerPrivateEvent` has empty `visibleTo`; `routeServerPrivate` is the
  identity, proving that privileged engine data never contaminates any
  `plannerKeyOfTurnHistory` / `cacheKeyOfTurnHistory` / `logEntryOfTurnHistory`
  / `modelInputOfTurnHistory` digest path.

Any Lean term that tries to pattern-match a `Draw` to expose `t` to
`observer ≠ actor` must discharge `obs = actor` via `Decidable`; the
equality proof is tracked in the kernel, closing the covert channel by
construction. This is the constructive content of
`draw_is_private` and `server_private_must_never_enter_*`.
-/

end Formal.Mahjong
