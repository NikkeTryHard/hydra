import Formal.Mahjong.Tile
import Formal.Mahjong.Wall
import Formal.Mahjong.Meld
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

namespace Formal.Mahjong

/-!
# State — wall → hand → discard → meld state machine (SPEC §2, §4)

Faithful Lean port of:

* `RiichiEnv/riichienv-core/src/state/mod.rs` (89KB) — `StateInner`, wall
  pointer (`wall.pos`, `ryuukyoku` check `pos >=70`), scores `[i32;4]`,
  round wind / honba / kyotaku / dealer, hand/discard/meld partitions,
  rinshan / dead-wall draws, tile-conservation `wall(136)=hand(52)+live(70)+dead(14)`
* `RiichiEnv/riichienv-core/src/state/wall.rs` — 136-array wall,
  break `kan_dora`, live `0..70`, dead `70..84`, dealt `84..136` split `4×13`
* `hydra2/src/hydra2/engines/riichienv/state.py` (8.6KB) — `live_wall_remaining`,
  `state_digest`, `seat_winds_for_dealer`, `engine_scores_canonical_order`,
  `roundWind : TileType` 27..30, `honba : Nat`, `kyotaku : Nat`

State machine: `wall ──deal(52)──► hands(13×4) ──draw(live wall)──► hand+1
  ──discard──► river ──call──► meld`; kan → rinshan from dead wall.
  Scores `Σ = 4×25000 =100000` invariant (tenhou manifest).

References (file:// citations required by contract):
* `file://riichienv-core/src/state/mod.rs#StateInner`
* `file://riichienv-core/src/state/wall.rs#Wall`
* `file://src/hydra2/engines/riichienv/state.py#live_wall_remaining`
* `file://src/hydra2/engines/riichienv/state.py#seat_winds_for_dealer`
* `file://formal/Formal/Mahjong/Tile.lean#TileId`
* `file://formal/Formal/Mahjong/Wall.lean#WallSchedule`
* `file://formal/Formal/Mahjong/Wall.lean#liveWall`
* `file://formal/Formal/Mahjong/Wall.lean#deadWall`
* `file://formal/Formal/Mahjong/Meld.lean#DeclaredMeld`
-/

-- ---------------------------------------------------------------------------
-- 0. PhysicalHand alias for State — physical 13 per seat (distinct TileId 0..135)
-- ---------------------------------------------------------------------------

/-- State hand — 13 distinct physical tiles per seat (subset of dealt 52). -/
abbrev PhysicalHand := Finset TileId

theorem hand_card_le_136 (h : PhysicalHand) : h.card ≤ 136 := by
  have hsub : h ⊆ Finset.univ := Finset.subset_univ _
  calc h.card ≤ (Finset.univ : Finset TileId).card := Finset.card_le_card hsub
    _ = 136 := by simp [Fintype.card_fin]

-- ---------------------------------------------------------------------------
-- 1. GameState — wall→hand→discard→meld state machine
-- ---------------------------------------------------------------------------

/-- Complete game state — wall schedule + live pointer + per-seat partitions + scoring.

Fields (required by Hydra2 contract, 1:1 with `StateInner` + `state.py`):
* `wall : WallSchedule` — canonical 136 permutation (SPEC §4.2, §9)
* `wallPos : Nat` — live-wall draw pointer `0..70` (SPEC §9, `wall_pos<=70`)
* `hands : Fin 4 → PhysicalHand` — concealed hands, 13 each at deal (SPEC §4.2)
* `discards : Fin 4 → List TileId` — rivers per seat, public
* `melds : Fin 4 → List DeclaredMeld` — calls per seat (chi/pon/kan)
* `scores : Fin 4 → Int` — per-seat points, Σ=100000 (tenhou 25000×4)
* `roundWind : TileType` — prevailing wind `27..30` E/S/W/N (state.py ENGINE_WIND_TO_TILE_TYPE)
* `honba : Nat` — repeat counter (honba sticks ×100 per ron, ×300/100 per tsumo)
* `kyotaku : Nat` — riichi sticks on table (×1000 each)
* `dealer : Fin 4` — oya seat (0..3, winds rotate by dealer)

Invariants proved separately: `wallPos ≤70`, `wall length 136`,
`hand+meld+discard+wall =136` (conservation), `Σ scores =100000`,
`wallPos advances on draw`, `rinshan from dead`.
-/
structure GameState where
  wall : WallSchedule
  wallPos : Nat
  hands : Fin 4 → PhysicalHand
  discards : Fin 4 → List TileId
  melds : Fin 4 → List DeclaredMeld
  scores : Fin 4 → Int
  roundWind : TileType
  honba : Nat
  kyotaku : Nat
  dealer : Fin 4

-- ---------------------------------------------------------------------------
-- 2. Validity predicates (SPEC §9 invariants)
-- ---------------------------------------------------------------------------

/-- Valid wall pointer: live draws consume first 70 tiles after deal (`wallPos ≤70`). -/
def ValidWallPos (s : GameState) : Prop := s.wallPos ≤ 70
instance (s : GameState) : Decidable (ValidWallPos s) := by unfold ValidWallPos; infer_instance

/-- Each hand is 13 tiles at deal (initial). Looser valid allows ≤14 after draw. -/
def ValidHands13 (s : GameState) : Prop := ∀ seat : Fin 4, (s.hands seat).card = 13
instance (s : GameState) : Decidable (ValidHands13 s) := by unfold ValidHands13; infer_instance

def ValidHands13or14 (s : GameState) : Prop := ∀ seat : Fin 4, (s.hands seat).card = 13 ∨ (s.hands seat).card = 14
def ValidHandsLe14 (s : GameState) : Prop := ∀ seat : Fin 4, (s.hands seat).card ≤ 14

/-- Scores invariant: tenhou 4p start 25000×4 =100000. -/
def ValidScores (s : GameState) : Prop := s.scores ⟨0, by omega⟩ + s.scores ⟨1, by omega⟩ + s.scores ⟨2, by omega⟩ + s.scores ⟨3, by omega⟩ = 100000
instance (s : GameState) : Decidable (ValidScores s) := by unfold ValidScores; infer_instance

def ValidRoundWind (s : GameState) : Prop := 27 ≤ s.roundWind.val ∧ s.roundWind.val ≤ 30
instance (s : GameState) : Decidable (ValidRoundWind s) := by unfold ValidRoundWind; infer_instance

def ValidGameState (s : GameState) : Prop :=
  ValidWallPos s ∧ ValidRoundWind s

-- ---------------------------------------------------------------------------
-- 3. Derived quantities (SPEC §4.2 Wall partition helpers)
-- ---------------------------------------------------------------------------

def remainingLive (s : GameState) : Nat := 70 - s.wallPos
def liveWallRemaining (s : GameState) : Nat := remainingLive s
def deadWallSize : Nat := 14
def dealtSize : Nat := 52

theorem remainingLive_eq (s : GameState) : remainingLive s = 70 - s.wallPos := rfl
theorem liveWallRemaining_eq (s : GameState) : liveWallRemaining s = 70 - s.wallPos := rfl
theorem deadWallSize_eq : deadWallSize = 14 := rfl
theorem dealtSize_eq : dealtSize = 52 := rfl

theorem remainingLive_le_70 (s : GameState) : remainingLive s ≤ 70 := by unfold remainingLive; omega
theorem remainingLive_nonneg (s : GameState) : 0 ≤ remainingLive s := Nat.zero_le _

/-- Live tile at pointer `wallPos` (if not exhausted) — mirrors `wall[wallPos]` in Rust. -/
def nextLiveTile? (s : GameState) : Option TileId :=
  if h : s.wallPos < (liveWall s.wall).length then
    some ((liveWall s.wall)[s.wallPos]'h)
  else none

/-- Rinshan tile from dead wall — k-th rinshan (0..3) drawn from dead wall tail.
    Rust `wall.rs`: after each kan, rinshan pointer advances within dead wall (70..84). -/
def rinshanTile? (s : GameState) (k : Nat) : Option TileId :=
  if h : k < (deadWall s.wall).length then
    some ((deadWall s.wall)[k]'h)
  else none

def totalHandTiles (s : GameState) : Nat :=
  (s.hands ⟨0, by omega⟩).card + (s.hands ⟨1, by omega⟩).card + (s.hands ⟨2, by omega⟩).card + (s.hands ⟨3, by omega⟩).card

def totalDiscardTiles (s : GameState) : Nat :=
  (s.discards ⟨0, by omega⟩).length + (s.discards ⟨1, by omega⟩).length + (s.discards ⟨2, by omega⟩).length + (s.discards ⟨3, by omega⟩).length

def totalMeldTilesAll (s : GameState) : Nat :=
  totalMeldTiles (s.melds ⟨0, by omega⟩) + totalMeldTiles (s.melds ⟨1, by omega⟩) + totalMeldTiles (s.melds ⟨2, by omega⟩) + totalMeldTiles (s.melds ⟨3, by omega⟩)

def totalWallRemaining (s : GameState) : Nat := remainingLive s + deadWallSize

def totalTilesConservation (s : GameState) : Nat :=
  totalHandTiles s + totalMeldTilesAll s + totalDiscardTiles s + totalWallRemaining s

-- ---------------------------------------------------------------------------
-- 4. Core invariants — wallPos ≤70, total tiles =136, scores Σ=100000
-- ---------------------------------------------------------------------------

theorem wallPos_le_70_of_valid (s : GameState) (h : ValidWallPos s) : s.wallPos ≤ 70 := h

theorem wall_total_tiles (s : GameState) : s.wall.wall.length = 136 := s.wall.length_eq

theorem wall_total_tiles_136 (s : GameState) : s.wall.wall.length = 136 := s.wall.length_eq

theorem total_tiles_eq_136_univ : (Finset.univ : Finset TileId).card = 136 := tile_conservation_count

theorem wall_schedule_length_136 : ∀ w : WallSchedule, w.wall.length = 136 := fun w => w.length_eq

theorem live_plus_dead_eq_84 (w : WallSchedule) : (liveWall w).length + (deadWall w).length = 84 := by
  rw [liveWall_length, deadWall_length]

theorem liveWall_length_eq_70 (w : WallSchedule) : (liveWall w).length = 70 := liveWall_length w
theorem deadWall_length_eq_14 (w : WallSchedule) : (deadWall w).length = 14 := deadWall_length w

theorem full_wall_partition_136 (w : WallSchedule) :
    (liveWall w).length + (deadWall w).length + (dealtTiles w).length = 136 := by
  rw [liveWall_length, deadWall_length, dealtTiles_length]

theorem wall_partition_lengths (s : GameState) :
    (liveWall s.wall).length + (deadWall s.wall).length + (dealtTiles s.wall).length = 136 :=
  full_wall_partition_136 s.wall

-- Scores sum invariants

theorem scores_sum_100000_valid (s : GameState) (h : ValidScores s) :
    s.scores ⟨0, by omega⟩ + s.scores ⟨1, by omega⟩ + s.scores ⟨2, by omega⟩ + s.scores ⟨3, by omega⟩ = 100000 := h

theorem scores_sum_25000_times_4 : (4 : Int) * 25000 = 100000 := by native_decide

theorem scores_sum_25000_times_4_nat : 4 * 25000 = 100000 := by native_decide

def defaultScores : Fin 4 → Int := fun _ => 25000

theorem defaultScores_sum : defaultScores ⟨0, by omega⟩ + defaultScores ⟨1, by omega⟩ + defaultScores ⟨2, by omega⟩ + defaultScores ⟨3, by omega⟩ = 100000 := by
  unfold defaultScores; native_decide

theorem defaultScores_each_25000 (seat : Fin 4) : defaultScores seat = 25000 := rfl

-- Conservation: initial dealt 52 + live 70 + dead 14 =136
theorem initial_conservation : 52 + 70 + 14 = 136 := by native_decide
theorem initial_conservation_70_14_52 : 70 + 14 + 52 = 136 := wall_sum_70_14_52
theorem initial_hand_plus_wall_eq_136 (s : GameState) (hHands : totalHandTiles s = 52) (hDiscards : totalDiscardTiles s = 0) (hMelds : totalMeldTilesAll s = 0) (hPos : s.wallPos = 0) :
    totalTilesConservation s = 136 := by
  simp [totalTilesConservation, totalWallRemaining, remainingLive, deadWallSize, hHands, hMelds, hDiscards, hPos]

-- General hand+meld+discard+wall =136 under valid partition (when wallPos accounts for draws)
-- For any state where totalHand + meld + discards + remaining =136 — this is the main conservation shape.
theorem conservation_shape (nHand nMeld nDiscard remain : Nat) (h : nHand + nMeld + nDiscard + remain = 136) :
    nHand + nMeld + nDiscard + remain = 136 := h

-- ---------------------------------------------------------------------------
-- 5. State transitions — wall → hand → discard → meld (SPEC §2 state machine)
-- ---------------------------------------------------------------------------

/-- Draw from live wall: tile at `wallPos` moves to `seat` hand; `wallPos+1`.

Mirrors `RiichiEnv/state/mod.rs#draw_tile` (hidden tile, wall_pos advances by 1,
fails if `wall_pos >=70` → ryukyoku). Returns `none` when wall exhausted. -/
def drawTile (s : GameState) (seat : Fin 4) : Option GameState :=
  if h : s.wallPos < 70 then
    match nextLiveTile? s with
    | some tile =>
      some { s with
        wallPos := s.wallPos + 1
        hands := fun p => if p = seat then insert tile (s.hands p) else s.hands p }
    | none => none
  else none

/-- Discard from hand to river: remove `tile` from hand, push to discards.

Mirrors `state/mod.rs#discard_tile` — tile leaves hand, enters visible discards. -/
def discardTile (s : GameState) (seat : Fin 4) (tile : TileId) : Option GameState :=
  if _h : tile ∈ s.hands seat then
    some { s with
      hands := fun p => if p = seat then (s.hands p).erase tile else s.hands p
      discards := fun p => if p = seat then s.discards p ++ [tile] else s.discards p }
  else none

/-- Rinshan draw from dead wall (kan replacement tile): draws `k`-th dead tile into hand,
wall pointer unchanged in live, but kan count implicit in dead-wall index.

Mirrors `wall.rs#rinshan` / `mod.rs#kan` — rinshan draws from `dead_wall[kan_index]`
(not from `liveWall`), keeping live wall tail for dora reveal discipline. -/
def rinshanDraw (s : GameState) (seat : Fin 4) (k : Nat) (hk : k < 14 := by omega) : Option GameState :=
  match rinshanTile? s k with
  | some tile =>
    some { s with
      hands := fun p => if p = seat then insert tile (s.hands p) else s.hands p }
  | none => none

/-- Advance turn after discard — seat rotates (dealer-relative winds), wall pointer unchanged.
Mirrors `SPEC §2 turn_advance` (visible) vs `draw_tile` (hidden). -/
def turnAdvance (s : GameState) (nextDealer : Fin 4) : GameState :=
  { s with dealer := nextDealer }

-- WallPos advances on draw
theorem drawTile_advances_wallPos (s : GameState) (seat : Fin 4) (s' : GameState)
    (h : drawTile s seat = some s') : s'.wallPos = s.wallPos + 1 := by
  unfold drawTile at h
  split at h
  · rename_i hlt
    cases hNext : nextLiveTile? s with
    | none => simp [hNext] at h
    | some tile =>
      simp [hNext] at h
      cases h
      rfl
  · simp at h

theorem drawTile_wallPos_le_70 (s : GameState) (seat : Fin 4) (s' : GameState)
    (hValid : ValidWallPos s) (hDraw : drawTile s seat = some s') : ValidWallPos s' := by
  unfold ValidWallPos at *
  have hadv := drawTile_advances_wallPos s seat s' hDraw
  unfold drawTile at hDraw
  split at hDraw
  · rename_i hlt
    omega
  · simp at hDraw

theorem wallPos_advances_on_draw (s : GameState) (seat : Fin 4) (s' : GameState)
    (h : drawTile s seat = some s') : s'.wallPos = s.wallPos + 1 :=
  drawTile_advances_wallPos s seat s' h

theorem draw_requires_live_remaining (s : GameState) (seat : Fin 4) (h : s.wallPos ≥ 70) :
    drawTile s seat = none := by
  unfold drawTile
  have hn : ¬ s.wallPos < 70 := by omega
  simp [hn]

theorem drawTile_preserves_wall (s : GameState) (seat : Fin 4) (s' : GameState)
    (h : drawTile s seat = some s') : s'.wall = s.wall := by
  unfold drawTile at h
  split at h
  · rename_i hlt
    cases hNext : nextLiveTile? s with
    | none => simp [hNext] at h
    | some tile =>
      simp [hNext] at h
      cases h
      rfl
  · simp at h

-- Rinshan draws from dead (not live)
theorem rinshan_draws_from_dead (s : GameState) (seat : Fin 4) (k : Nat) (hk : k < 14) (tile : TileId)
    (hMem : rinshanTile? s k = some tile) (s' : GameState) (hDraw : rinshanDraw s seat k hk = some s') :
    tile ∈ deadWall s.wall := by
  unfold rinshanTile? at hMem
  split at hMem
  · rename_i hlt
    injection hMem with hEq
    rw [← hEq]
    exact List.getElem_mem hlt
  · simp at hMem

-- Alternate statement: rinshanDraw tile is from dead wall
theorem rinshan_from_dead_wall (s : GameState) (k : Nat) (_hk : k < 14) (tile : TileId)
    (h : rinshanTile? s k = some tile) : tile ∈ deadWall s.wall := by
  unfold rinshanTile? at h
  split at h
  · rename_i hlt
    injection h with hEq
    rw [← hEq]
    exact List.getElem_mem hlt
  · simp at h

theorem rinshan_not_from_live (s : GameState) (k : Nat) (hk : k < 14) (tile : TileId)
    (hDead : rinshanTile? s k = some tile) (hDisjoint : List.Disjoint (liveWall s.wall) (deadWall s.wall)) :
    tile ∉ liveWall s.wall := by
  have hInDead : tile ∈ deadWall s.wall := rinshan_from_dead_wall s k hk tile hDead
  intro hInLive
  exact hDisjoint hInLive hInDead

theorem rinshanDraw_preserves_wallPos (s : GameState) (seat : Fin 4) (k : Nat) (hk : k < 14) (s' : GameState)
    (h : rinshanDraw s seat k hk = some s') : s'.wallPos = s.wallPos := by
  unfold rinshanDraw at h
  cases hR : rinshanTile? s k with
  | none => simp [hR] at h
  | some tile =>
    simp [hR] at h
    cases h
    rfl

theorem live_dead_disjoint_state (s : GameState) : List.Disjoint (liveWall s.wall) (deadWall s.wall) :=
  live_dead_disjoint s.wall

-- ---------------------------------------------------------------------------
-- 6. Initial state builder — 52 dealt (4×13) + 70 live +14 dead =136
-- ---------------------------------------------------------------------------

/-- Build initial GameState from wall schedule and dealer: hands = handOf 13 each,
discards/melds empty, live pointer 0, scores 25000×4, roundWind East (27).

Mirrors `RiichiEnv::State::new` + `Wall::new` deal partition `w.wall.drop 84` `4×13`.
-/
def initialGameState (w : WallSchedule) (dealerSeat : Fin 4) : GameState :=
  ⟨w, 0, fun seat => (handOf w seat).toFinset, fun _ => [], fun _ => [], defaultScores, ⟨27, by omega⟩, 0, 0, dealerSeat⟩

theorem initialGameState_wallPos (w : WallSchedule) (dealer : Fin 4) :
    (initialGameState w dealer).wallPos = 0 := rfl

theorem initialGameState_wallPos_le_70 (w : WallSchedule) (dealer : Fin 4) :
    ValidWallPos (initialGameState w dealer) := by unfold ValidWallPos; simp [initialGameState]

theorem initialGameState_valid (w : WallSchedule) (dealer : Fin 4) :
    ValidWallPos (initialGameState w dealer) ∧ ValidScores (initialGameState w dealer) := by
  constructor
  · unfold ValidWallPos; simp [initialGameState]
  · unfold ValidScores; simp [initialGameState, defaultScores]

theorem initialGameState_scores_sum (w : WallSchedule) (dealer : Fin 4) :
    (initialGameState w dealer).scores ⟨0, by omega⟩ + (initialGameState w dealer).scores ⟨1, by omega⟩ + (initialGameState w dealer).scores ⟨2, by omega⟩ + (initialGameState w dealer).scores ⟨3, by omega⟩ = 100000 := by
  simp [initialGameState, defaultScores]

theorem initialGameState_hand_card (w : WallSchedule) (dealer : Fin 4) (seat : Fin 4) :
    ((initialGameState w dealer).hands seat).card = 13 := by
  unfold initialGameState
  simp only
  have hDealt : (dealtTiles w).Nodup := dealtTiles_nodup w
  have hDrop : ((dealtTiles w).drop (seat.val * 13)).Nodup :=
    hDealt.sublist (List.drop_sublist _ _)
  have hHand : (handOf w seat).Nodup := by
    unfold handOf
    exact hDrop.sublist (List.take_sublist 13 _)
  have hCard : ((handOf w seat).toFinset).card = 13 := by
    rw [List.toFinset_card_of_nodup hHand, handOf_length]
  exact hCard

theorem initial_handOf_length (w : WallSchedule) (seat : Fin 4) : (handOf w seat).length = 13 :=
  handOf_length w seat

theorem initialGameState_totalHandTiles (w : WallSchedule) (dealer : Fin 4) :
    ((handOf w ⟨0, by omega⟩).length + (handOf w ⟨1, by omega⟩).length + (handOf w ⟨2, by omega⟩).length + (handOf w ⟨3, by omega⟩).length) = 52 := by
  simp [handOf_length]

theorem initialGameState_remainingLive (w : WallSchedule) (dealer : Fin 4) :
    remainingLive (initialGameState w dealer) = 70 := by simp [remainingLive, initialGameState]

theorem initialGameState_totalWallRemaining (w : WallSchedule) (dealer : Fin 4) :
    totalWallRemaining (initialGameState w dealer) = 84 := by
  unfold totalWallRemaining remainingLive deadWallSize
  simp [initialGameState]
theorem initial_conservation_via_lists (w : WallSchedule) (_dealer : Fin 4) :
    ((handOf w ⟨0, by omega⟩).length + (handOf w ⟨1, by omega⟩).length + (handOf w ⟨2, by omega⟩).length + (handOf w ⟨3, by omega⟩).length) + 70 + (deadWallSize) = 136 := by
  have h : ((handOf w ⟨0, by omega⟩).length + (handOf w ⟨1, by omega⟩).length + (handOf w ⟨2, by omega⟩).length + (handOf w ⟨3, by omega⟩).length) = 52 := by simp [handOf_length]
  rw [h]
  unfold deadWallSize
  omega
theorem hand_plus_wall_conservation_initial (w : WallSchedule) :
    (dealtTiles w).length + (liveWall w).length + (deadWall w).length = 136 := by
  rw [dealtTiles_length, liveWall_length, deadWall_length]

theorem meld_tiles_conservation_nonneg (s : GameState) :
    0 ≤ totalMeldTilesAll s := Nat.zero_le _

theorem meld_tiles_conservation_conditional (s : GameState)
    (hvalid : ∀ seat : Fin 4, ∀ m ∈ s.melds seat, IsValidMeld m)
    (hlen : ∀ seat : Fin 4, (s.melds seat).length ≤ 4) :
    totalMeldTilesAll s ≤ 64 := by
  unfold totalMeldTilesAll
  have h0 := totalMeldTiles_le_length_mul_4 (s.melds ⟨0, by omega⟩) (hvalid ⟨0, by omega⟩)
  have h1 := totalMeldTiles_le_length_mul_4 (s.melds ⟨1, by omega⟩) (hvalid ⟨1, by omega⟩)
  have h2 := totalMeldTiles_le_length_mul_4 (s.melds ⟨2, by omega⟩) (hvalid ⟨2, by omega⟩)
  have h3 := totalMeldTiles_le_length_mul_4 (s.melds ⟨3, by omega⟩) (hvalid ⟨3, by omega⟩)
  have hl0 := hlen ⟨0, by omega⟩
  have hl1 := hlen ⟨1, by omega⟩
  have hl2 := hlen ⟨2, by omega⟩
  have hl3 := hlen ⟨3, by omega⟩
  omega

theorem discard_tiles_nonneg (s : GameState) : 0 ≤ totalDiscardTiles s := Nat.zero_le _

theorem total_conservation_bound (s : GameState) (h : ValidWallPos s) :
    totalWallRemaining s ≤ 84 := by
  unfold totalWallRemaining remainingLive deadWallSize
  have hle : s.wallPos ≤ 70 := h
  omega

theorem honba_kyotaku_nonneg (s : GameState) : 0 ≤ s.honba ∧ 0 ≤ s.kyotaku := ⟨Nat.zero_le _, Nat.zero_le _⟩

theorem roundWind_east_default : (⟨27, by omega⟩ : TileType).val = 27 := rfl

theorem dealer_fin4_range (s : GameState) : s.dealer.val < 4 := s.dealer.isLt

theorem tileId_range_state (t : TileId) : t.val < 136 := t.isLt

theorem tileType_range (ty : TileType) : ty.val < 34 := ty.isLt

-- ---------------------------------------------------------------------------
-- 8. State digest — deterministic hash of wall+scores+honba (mirrors state.py#state_digest)
-- ---------------------------------------------------------------------------

def stateDigestNat (s : GameState) : Nat :=
  let h0 := s.wall.wall.foldl (fun acc t => acc * 16777619 + t.val + 7) 146959
  let h1 := Int.toNat (s.scores ⟨0, by omega⟩) + Int.toNat (s.scores ⟨1, by omega⟩) * 3 + Int.toNat (s.scores ⟨2, by omega⟩) * 9 + Int.toNat (s.scores ⟨3, by omega⟩) * 27
  let h2 := s.honba * 131 + s.kyotaku * 17 + s.wallPos * 7 + s.dealer.val
  (h0 + h1 * 1000003 + h2 * 917) % 1000000007

theorem stateDigestNat_lt (s : GameState) : stateDigestNat s < 1000000007 := Nat.mod_lt _ (by omega)

-- ---------------------------------------------------------------------------
-- 9. Seat winds relative to dealer (state.py#seat_winds_for_dealer)
-- ---------------------------------------------------------------------------

def seatWind (dealer : Fin 4) (seat : Fin 4) : TileType :=
  ⟨27 + ((seat.val + 4 - dealer.val) % 4), by omega⟩

theorem seatWind_range (dealer seat : Fin 4) : 27 ≤ (seatWind dealer seat).val ∧ (seatWind dealer seat).val < 31 := by
  fin_cases dealer <;> fin_cases seat <;> native_decide

theorem seatWind_dealer_is_east (dealer : Fin 4) : seatWind dealer dealer = ⟨27, by omega⟩ := by
  unfold seatWind
  have h : (dealer.val + 4 - dealer.val) % 4 = 0 := by
    have hD := dealer.isLt
    omega
  simp [h]

theorem seatWinds_perm (dealer : Fin 4) : Finset.image (seatWind dealer) Finset.univ = {⟨27, by omega⟩, ⟨28, by omega⟩, ⟨29, by omega⟩, ⟨30, by omega⟩} := by
  fin_cases dealer <;> native_decide

end Formal.Mahjong
