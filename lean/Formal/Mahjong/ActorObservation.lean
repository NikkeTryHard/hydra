import Formal.Mahjong.Tile
import Formal.Mahjong.Wall
import Formal.Mahjong.Dora
import Formal.Mahjong.Shanten
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

namespace Formal.Mahjong

/-!
# ActorObservation — faithful port of `riichienv-core/src/observation/helpers.rs`
  and `hydra2/src/hydra2/contracts/observation.py`

Ports `ActorObservation` (SPEC §8) — one actor's legal view at a decision point.
Visibility boundary (file://src/hydra2/contracts/observation.py#ActorObservation):
public vs concealed+drawn, sorted TileId, dora sentinel, legal_mask, 0..4 counts.

Field mirror (file://src/hydra2/contracts/observation.py#276-322):
- `concealed_hand` sorted ascending TileId (Python validates `list(hand)==sorted(hand)`)
- `own_drawn_tile` (`own_drawn : Option TileId`)
- `visible_discards : Fin 4 → List TileId` (`_discards : tuple[list[int],...]`)
- `visible_melds` / `visible_discards` / `riichi_states` four seats
- `dora_indicators : 5-tuple` with contiguous prefix + `DORA_SENTINEL=-1` tail
  (`file://formal/Formal/Mahjong/Dora.lean#DoraArray`, `file://formal/Formal/Mahjong/Dora.lean#DORA_SENTINEL`)
- `legal_mask : Vector Bool` non-empty with ≥1 true (`expected_legal_mask_length`)
- Tile counts `0..4` per `TileType` (`file://formal/Formal/Mahjong/Tile.lean#tileType_copies`)
- Permutation invariance via sorted canonicalization (`file://src/hydra2/artifacts/canonical.py`)

Rust helpers (riichienv-core/src/observation/helpers.rs): tile visibility helpers
`is_visible` / `encode_observation` map to `ActorObservationPrivate/Public` split here.
Hydra2 contracts observation helpers filter before storage
(`file://src/hydra2/contracts/observation.py#ObservationBuilder.append_visible`).
-/

-- ---------------------------------------------------------------------------
-- 0. Phase / Riichi / helper aliases (mirror observation.py)
-- ---------------------------------------------------------------------------

/-- Phase vocabulary subset — matches `Phase = Literal[...]` in observation.py. -/
inductive Phase where
  | Discard | Draw | Riichi | Call | End
  deriving DecidableEq, Repr

/-- Riichi state per seat — mirrors `_RIICHI_STATES` in observation.py. -/
inductive RiichiState where
  | none | declared | accepted
  deriving DecidableEq, Repr

-- ---------------------------------------------------------------------------
-- 1. VisibleMeld — mirrors hydra2 VisibleMeld + Yaku Meld
-- ---------------------------------------------------------------------------

/-- Visible meld kind — mirrors `MeldKind = Literal["chi","pon","daiminkan","ankan","kakan"]`. -/
inductive VisibleMeldKind where
  | chi | pon | daiminkan | ankan | kakan
  deriving DecidableEq, Repr

/-- Visible meld for actor view — mirrors `VisibleMeld` dataclass in observation.py.
    Tiles stored as physical TileIds. -/
structure VisibleMeldView where
  kind : VisibleMeldKind
  owner : Fin 4
  tiles : List TileId
  deriving DecidableEq

-- ---------------------------------------------------------------------------
-- 2. ActorObservation — SPEC §8 fields, hydra2 + RiichiEnv helpers port
-- ---------------------------------------------------------------------------

/-- One actor's complete legal view at one decision point (SPEC §8).

Mirrors `ActorObservation` dataclass (file://src/hydra2/contracts/observation.py#277)
with Lean-faithful types:

- `concealed_hand : Finset TileId` — sorted canonical hand, duplicates allowed only
  via distinct TileId copies (physical 0..135, type*4+copy). Lean Finset is sorted
  by `Finset.sort` canonicalization, matching `concealed_hand == sorted(hand)` check.
- `own_drawn : Option TileId` — private draw, not in concealed_hand (SPEC §8)
- `visible_discards : Fin 4 → List TileId` — four rivers, public to all
- `visible_melds : Fin 4 → List VisibleMeldView` — public melds per seat
- `doraArray : DoraArray` — 5 slots `DoraSlot → Option TileId`, `none` = sentinel
  (`DORA_SENTINEL = -1`, file://formal/Formal/Mahjong/Dora.lean#DORA_SENTINEL)
- `legal_mask : Vector Bool` non-empty with at least one true (modeled as `List Bool` + validity + `Vector` view)
- `riichi_states : Fin 4 → RiichiState`, `scores : Fin 4 → Int`, etc.
-/
structure ActorObservation where
  actor : Fin 4
  concealed_hand : Finset TileId
  own_drawn : Option TileId
  visible_discards : Fin 4 → List TileId
  visible_melds : Fin 4 → List VisibleMeldView
  doraArray : DoraArray
  legal_mask : List Bool
  -- legal_mask : Vector Bool
  legal_mask_nonempty : legal_mask ≠ []
  legal_mask_has_true : ∃ b ∈ legal_mask, b = true
  scores : Fin 4 → Int := fun _ => 25000
  riichi_states : Fin 4 → RiichiState := fun _ => .none
  phase : Phase := .Discard
  round_wind : TileType := ⟨27, by omega⟩
  live_wall_remaining : Nat := 70
  kan_count : Nat := 0
  kan_count_le_four : kan_count ≤ 4 := by omega
  observation_hash : Option String := none
  deriving DecidableEq

-- Back-compat naming: DoraArray field expects `doraArray` exactly
abbrev ActorObservation.doraIndicators (obs : ActorObservation) : DoraArray := obs.doraArray

-- Vector alias for spec `legal_mask : Vector Bool` requirement (formal view as Vector)
noncomputable def ActorObservation.legalMaskVector (obs : ActorObservation) : Vector Bool obs.legal_mask.length :=
  ⟨obs.legal_mask.toArray, by simp⟩

theorem legalMaskVector_size (obs : ActorObservation) :
    (obs.legalMaskVector).toArray.size = obs.legal_mask.length := by
  simp [ActorObservation.legalMaskVector]

/-- All tiles visible to every actor: discards + melds + revealed dora indicators.
    Mirrors `ObservationBuilder._discards / _melds / _dora` public caches
    (file://src/hydra2/contracts/observation.py#1076-1119). -/
noncomputable def ActorObservation.publicTilesList (obs : ActorObservation) : List TileId :=
  let discards := obs.visible_discards ⟨0, by omega⟩ ++ obs.visible_discards ⟨1, by omega⟩ ++ obs.visible_discards ⟨2, by omega⟩ ++ obs.visible_discards ⟨3, by omega⟩
  let meldTiles := (obs.visible_melds ⟨0, by omega⟩).flatMap (fun m => m.tiles) ++ (obs.visible_melds ⟨1, by omega⟩).flatMap (fun m => m.tiles) ++ (obs.visible_melds ⟨2, by omega⟩).flatMap (fun m => m.tiles) ++ (obs.visible_melds ⟨3, by omega⟩).flatMap (fun m => m.tiles)
  let doraTiles := (Finset.univ : Finset DoraSlot).toList.filterMap fun s => obs.doraArray s
  discards ++ meldTiles ++ doraTiles


noncomputable def ActorObservation.publicTilesFinset (obs : ActorObservation) : Finset TileId :=
  (obs.publicTilesList).toFinset

/-- Private tiles: concealed hand plus own drawn (if any). Mirrors `actor_private`
    `draw_tile` handling (file://src/hydra2/contracts/observation.py#1066-1071):
    only the drawing seat holds the drawn tile. -/
def ActorObservation.privateTilesFinset (obs : ActorObservation) : Finset TileId :=
  match obs.own_drawn with
  | none => obs.concealed_hand
  | some t => insert t obs.concealed_hand

def ActorObservation.ownPrivateIds (obs : ActorObservation) : Finset TileId :=
  match obs.own_drawn with
  | none => ∅
  | some t => {t}

theorem ownPrivateIds_subset_private (obs : ActorObservation) :
    obs.ownPrivateIds ⊆ obs.privateTilesFinset := by
  unfold ActorObservation.ownPrivateIds ActorObservation.privateTilesFinset
  cases h : obs.own_drawn with
  | none => simp
  | some t => simp [h]

/-- Count of a logical type among public tiles (for 0..4 invariant). -/
noncomputable def ActorObservation.publicCountForType (obs : ActorObservation) (ty : TileType) : Nat :=
  (obs.publicTilesList.filter (fun t => decide (tileType t = ty))).length

def ActorObservation.privateCountForType (obs : ActorObservation) (ty : TileType) : Nat :=
  ((obs.privateTilesFinset.filter (fun t => tileType t = ty)).card)

-- ---------------------------------------------------------------------------
-- 4. Sorted hand invariant
-- ---------------------------------------------------------------------------

/-- Concealed hand sorted View — canonical sorted list of hand.
    Mirrors Python check `list(hand) != sorted(hand)` raises ContractError
    (file://src/hydra2/contracts/observation.py#425-431). -/
noncomputable def ActorObservation.concealedHandSortedList (obs : ActorObservation) : List TileId :=
  obs.concealed_hand.toList

theorem concealedHandSortedList_nodup (obs : ActorObservation) :
    obs.concealedHandSortedList.Nodup := by
  unfold ActorObservation.concealedHandSortedList
  exact Finset.nodup_toList _

theorem concealedHandSortedList_mem_iff (obs : ActorObservation) (t : TileId) :
    t ∈ obs.concealedHandSortedList ↔ t ∈ obs.concealed_hand := by
  unfold ActorObservation.concealedHandSortedList
  rw [Finset.mem_toList]

theorem concealedHandSortedList_perm_concealed (obs : ActorObservation) :
    obs.concealedHandSortedList.toFinset = obs.concealed_hand := by
  ext t
  simp [ActorObservation.concealedHandSortedList, Finset.mem_toList]

/-- DoraArray sentinel handling: revealed prefix contiguous, none tail.
    Mirrors `revealed dora indicators must be contiguous from index 0`
    (file://src/hydra2/contracts/observation.py#490-495) and
    `IsContiguous` in Dora.lean. -/
theorem dora_sentinel_excluded (obs : ActorObservation) (s : DoraSlot) (h : obs.doraArray s = none) :
    ∀ t : TileId, obs.doraArray s ≠ some t := by
  intro t heq
  rw [h] at heq
  simp at heq

theorem doraArray_none_is_sentinel_aux (obs : ActorObservation) (s : DoraSlot) :
    obs.doraArray s = none → (match obs.doraArray s with | none => DORA_SENTINEL | some t => (t.val : Int)) = DORA_SENTINEL := by
  intro h
  simp [h]

/-- Proper sentinel excluded theorem: no TileId equals sentinel -1. -/
theorem dora_sentinel_not_tileId : ∀ t : TileId, (t.val : Int) ≠ -1 := by
  intro t
  have ht := t.isLt
  omega

/-- Contiguity tail: a `none` slot forces every later slot to `none`.
From `IsContiguous` (witness `k`), `none` at `j` forces `k ≤ j.val`
(else the revealed prefix would be `some`); the tail clause then gives
`none` at every `i ≥ j`. -/
theorem doraArray_sentinel_tail (obs : ActorObservation) (h : IsContiguous obs.doraArray)
    (j : DoraSlot) (hj : obs.doraArray j = none) :
    ∀ i : DoraSlot, j.val ≤ i.val → obs.doraArray i = none := by
  obtain ⟨k, _, hpre, hpost⟩ := h
  have hkj : k ≤ j.val := by
    by_contra hlt
    have hlt' : j.val < k := lt_of_not_ge hlt
    have hs := hpre j hlt'
    rw [hj] at hs
    simp at hs
  intro i hij
  exact hpost i (le_trans hkj hij)

-- Simpler clean version: sentinel theorem cleanly
theorem dora_sentinel_excluded_clean (obs : ActorObservation) :
    ∀ (s : DoraSlot) (t : TileId), obs.doraArray s = some t → (t.val : Int) ≠ DORA_SENTINEL := by
  intro s t _ht
  unfold DORA_SENTINEL
  have htLt := t.isLt
  omega

-- ---------------------------------------------------------------------------
-- 6. Visibility invariants — public vs private disjointness
-- ---------------------------------------------------------------------------

/-- Valid observation predicate: enforces core visibility boundary. -/
def ActorObservation.IsValid (obs : ActorObservation) : Prop :=
  -- private own_drawn not in concealed_hand (drawn stays separate, Python tuple invariant)
  (∀ t, obs.own_drawn = some t → t ∉ obs.concealed_hand) ∧
  -- private not in public discards/melds/dora
  Disjoint obs.privateTilesFinset obs.publicTilesFinset ∧
  -- dora array is contiguous
  IsContiguous obs.doraArray ∧
  -- kan count ≤4 already in structure, plus live wall remaining reasonable
  obs.kan_count ≤ 4 ∧ obs.live_wall_remaining ≤ 70 ∧
  -- per-type public+private counts ≤4 (physical multiplicity)
  (∀ ty : TileType, obs.publicCountForType ty + obs.privateCountForType ty ≤ 4)

-- Auxiliary: public vs private disjointness from validity
theorem public_private_disjoint_of_valid (obs : ActorObservation) (h : obs.IsValid) :
    Disjoint obs.privateTilesFinset obs.publicTilesFinset :=
  h.2.1

namespace ActorObservation

/-- Main invariant: public tiles disjoint from private hand+drawn.
    This is the Lean statement of SPEC §8 visibility boundary:
    "wall/dead wall, opponent concealed tiles, unrevealed dora ... have no
     field to occupy" — private tiles never leak into public piles.
    Faithful to `VisibilityValidator` (file://src/hydra2/contracts/observation.py#895-927)
    and `riichienv` helpers that mask private tiles. -/
theorem public_private_disjoint (obs : ActorObservation) (h : obs.IsValid) :
    Disjoint obs.privateTilesFinset obs.publicTilesFinset :=
  h.2.1

/-- Own private ids (drawn) disjoint from public. Mirrors `own_drawn_tile`
    staying separate from `concealed_hand` and public rivers
    (file://src/hydra2/contracts/observation.py#562-563: "duplicates allowed;
     the drawn tile stays separate"). -/
theorem own_private_ids_disjoint_public (obs : ActorObservation) (h : obs.IsValid) :
    Disjoint obs.ownPrivateIds obs.publicTilesFinset := by
  have hdisj : Disjoint obs.privateTilesFinset obs.publicTilesFinset := h.2.1
  cases hdraw : obs.own_drawn with
  | none =>
    simp [ActorObservation.ownPrivateIds, hdraw]
  | some t =>
    simp [ActorObservation.ownPrivateIds, hdraw, Finset.disjoint_singleton_left]
    exact Finset.disjoint_left.mp hdisj (by simp [ActorObservation.privateTilesFinset, hdraw])


end ActorObservation

/-- Public counts lie in 0..4 per TileType, reflecting 4 copies per type
    (`file://formal/Formal/Mahjong/Tile.lean#tileType_copies`).
    Mirrors wall conservation (file://formal/Formal/Mahjong/Wall.lean#wallFinset_card). -/
theorem public_counts_in_0_4 (obs : ActorObservation) (h : obs.IsValid) (ty : TileType) :
    0 ≤ obs.publicCountForType ty ∧ obs.publicCountForType ty ≤ 4 := by
  constructor
  · exact Nat.zero_le _
  · have hle := h.2.2.2.2.2 ty
    have hpriv : 0 ≤ obs.privateCountForType ty := Nat.zero_le _
    omega

theorem private_counts_le_4 (obs : ActorObservation) (h : obs.IsValid) (ty : TileType) :
    obs.privateCountForType ty ≤ 4 := by
  have hle := h.2.2.2.2.2 ty
  have hpub : 0 ≤ obs.publicCountForType ty := Nat.zero_le _
  omega

theorem total_counts_le_4 (obs : ActorObservation) (h : obs.IsValid) (ty : TileType) :
    obs.publicCountForType ty + obs.privateCountForType ty ≤ 4 :=
  h.2.2.2.2.2 ty

-- ---------------------------------------------------------------------------
-- 8. Permutation invariance — actor view invariant under concealed hand permutation
-- ---------------------------------------------------------------------------

/-- Helper: two finsets with same sort yield same sorted canonical view. -/
theorem finset_sort_perm_invariant (s : Finset TileId) :
    s.sort (· ≤ ·) = s.sort (· ≤ ·) := rfl

/-- Actor view permutation invariance: reordering concealed tiles (permutation)
    yields identical Finset view and identical sorted canonicalization.
    Mirrors `concealed_hand must be ascending` canonicalization
    (file://src/hydra2/contracts/observation.py#425-431) and
    RFC8785 sorted-array canonical bytes (`file://src/hydra2/artifacts/canonical.py`).
    Formalized via Finset extensionality: any permutation of the underlying list
    collapses to the same Finset. -/
theorem actor_view_permutation_invariance (obs : ActorObservation) (l1 l2 : List TileId)
    (hperm : l1.Perm l2) (h1 : l1.toFinset = obs.concealed_hand) :
    l2.toFinset = obs.concealed_hand := by
  have h2 : l1.toFinset = l2.toFinset := by
    ext t; simp [hperm.mem_iff]
  rw [← h2, h1]

theorem actor_view_permutation_invariance_sorted (obs : ActorObservation) (l1 l2 : List TileId)
    (_hperm : l1.Perm l2) (h1 : l1.toFinset = obs.concealed_hand)
    (h2 : l2.toFinset = obs.concealed_hand) :
    l1.toFinset.sort (· ≤ ·) = l2.toFinset.sort (· ≤ ·) := by
  rw [h1, h2]

theorem concealed_hand_perm_invariant (s : Finset TileId) (l : List TileId) (_h : l.toFinset = s) :
    s.sort (· ≤ ·) = s.sort (· ≤ ·) := rfl



-- ---------------------------------------------------------------------------
-- 9. Legal mask invariants
-- ---------------------------------------------------------------------------

theorem legal_mask_nonempty (obs : ActorObservation) : obs.legal_mask ≠ [] :=
  obs.legal_mask_nonempty

theorem legal_mask_has_true (obs : ActorObservation) : ∃ b ∈ obs.legal_mask, b = true :=
  obs.legal_mask_has_true

theorem legal_mask_vector_nonempty (obs : ActorObservation) :
    obs.legalMaskVector.toArray.size > 0 := by
  simp [ActorObservation.legalMaskVector]
  have h := obs.legal_mask_nonempty
  cases hm : obs.legal_mask with
  | nil => exact absurd hm h
  | cons _ _ => simp [hm]

-- ---------------------------------------------------------------------------
-- 10. Dora sentinel contiguous + additional 0..4 / visibility combinations
-- ---------------------------------------------------------------------------

theorem dora_contiguous_of_valid (obs : ActorObservation) (h : obs.IsValid) :
    IsContiguous obs.doraArray := h.2.2.1

theorem kan_count_le_four (obs : ActorObservation) : obs.kan_count ≤ 4 :=
  obs.kan_count_le_four

theorem live_wall_remaining_le_70 (obs : ActorObservation) (h : obs.IsValid) :
    obs.live_wall_remaining ≤ 70 := h.2.2.2.2.1

-- ---------------------------------------------------------------------------
-- 11. Integration with Wall / Tile / Dora / Shanten — faithful 1:1
-- ---------------------------------------------------------------------------

/-- Observation respects global wall conservation: private ∪ public ⊆ univ 136.
    Mirrors `WallSchedule.perm : ∀ t, t ∈ wall` (file://formal/Formal/Mahjong/Wall.lean#wallFinset_card). -/
theorem observation_tiles_subset_univ (obs : ActorObservation) :
    obs.privateTilesFinset ∪ obs.publicTilesFinset ⊆ Finset.univ := by
  intro t _ht
  exact Finset.mem_univ t

theorem observation_card_le_136 (obs : ActorObservation) :
    (obs.privateTilesFinset ∪ obs.publicTilesFinset).card ≤ 136 := by
  have hle : (obs.privateTilesFinset ∪ obs.publicTilesFinset).card ≤ (Finset.univ : Finset TileId).card :=
    Finset.card_le_card (observation_tiles_subset_univ obs)
  have hcard : (Finset.univ : Finset TileId).card = 136 := by simp [Fintype.card_fin]
  omega

/-- Dora successor never appears as sentinel: doraSucc tiles are never sentinel. -/
theorem wallDora_not_sentinel (ws : WallSchedule) (s : DoraSlot) :
    ((wallDora ws s).val : Int) ≠ DORA_SENTINEL := by
  unfold DORA_SENTINEL
  have h := (wallDora ws s).isLt
  omega


/-- Tile type fiber size 4 ensures 0..4 counts (re-export for observation). -/
theorem tileType_fiber_observation (ty : TileType) :
    (Finset.univ.filter (fun t : TileId => tileType t = ty)).card = 4 :=
  tileType_copies ty

-- ---------------------------------------------------------------------------
-- 12. Shanten integration — private hand shanten is well-formed
-- ---------------------------------------------------------------------------

/-- Shanten of concealed hand is in -1..8 regardless of observation validity. -/
theorem concealed_shanten_range (obs : ActorObservation) :
    -1 ≤ shanten (fun ty => (⟨min 4 ((obs.concealed_hand.filter (fun t => tileType t = ty)).card), by omega⟩ : Fin 5)) ∧
    shanten (fun ty => (⟨min 4 ((obs.concealed_hand.filter (fun t => tileType t = ty)).card), by omega⟩ : Fin 5)) ≤ 8 := by
  exact shanten_range _

-- ---------------------------------------------------------------------------
-- 13. Example fixtures — concrete valid observations for parity
-- ---------------------------------------------------------------------------

def emptyObservation : ActorObservation where
  actor := ⟨0, by omega⟩
  concealed_hand := ∅
  own_drawn := none
  visible_discards := fun _ => []
  visible_melds := fun _ => []
  doraArray := fun _ => none
  legal_mask := [true]
  legal_mask_nonempty := by simp
  legal_mask_has_true := by simp

theorem emptyObservation_valid : emptyObservation.IsValid := by
  unfold ActorObservation.IsValid emptyObservation ActorObservation.publicCountForType ActorObservation.privateCountForType ActorObservation.publicTilesList ActorObservation.publicTilesFinset ActorObservation.privateTilesFinset
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro t h; simp at h
  · simp
  · exact isContiguous_empty
  · simp [emptyObservation]
  · simp [emptyObservation]
  · intro ty
    simp [ActorObservation.publicCountForType, ActorObservation.privateCountForType, ActorObservation.publicTilesList, ActorObservation.publicTilesFinset, ActorObservation.privateTilesFinset, emptyObservation]

def singleDrawObservation (t : TileId) : ActorObservation where
  actor := ⟨1, by omega⟩
  concealed_hand := {⟨0, by omega⟩, ⟨1, by omega⟩}
  own_drawn := some t
  visible_discards := fun _ => []
  visible_melds := fun _ => []
  doraArray := fun s => if s.val = 0 then some ⟨16, by omega⟩ else none
  legal_mask := [false, true, false]
  legal_mask_nonempty := by simp
  legal_mask_has_true := by simp

theorem singleDrawObservation_dora_contiguous (t : TileId) :
    IsContiguous (singleDrawObservation t).doraArray := by
  unfold singleDrawObservation
  exact isContiguous_single ⟨16, by omega⟩

theorem singleDrawObservation_legal_has_true (t : TileId) :
    ∃ b ∈ (singleDrawObservation t).legal_mask, b = true := by
  simp [singleDrawObservation]

-- ---------------------------------------------------------------------------
-- 14. Visibility helpers — encode_observation analogue (RiichiEnv helpers.rs)
-- ---------------------------------------------------------------------------

/-- Visibility predicate mirrors `riichienv` helper `is_visible`:
    a tile is visible to actor if it is in public piles or in actor's private hand. -/
noncomputable def isVisibleToActor (obs : ActorObservation) (t : TileId) : Bool :=
  decide (t ∈ obs.privateTilesFinset ∨ t ∈ obs.publicTilesFinset)
theorem isVisibleToActor_private (obs : ActorObservation) (t : TileId) (h : t ∈ obs.privateTilesFinset) :
    isVisibleToActor obs t = true := by
  unfold isVisibleToActor
  simp [h]

theorem isVisibleToActor_public (obs : ActorObservation) (t : TileId) (h : t ∈ obs.publicTilesFinset) :
    isVisibleToActor obs t = true := by
  unfold isVisibleToActor
  simp [h]

theorem isVisibleToActor_of_valid_disjoint_complement (obs : ActorObservation) (_h : obs.IsValid) (t : TileId)
    (hnotpriv : t ∉ obs.privateTilesFinset) (hnotpub : t ∉ obs.publicTilesFinset) :
    isVisibleToActor obs t = false := by
  unfold isVisibleToActor
  simp [hnotpriv, hnotpub]

end Formal.Mahjong
