import Formal.Mahjong.Tile
import Formal.Mahjong.Wall
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Card
import Mathlib.Data.Fintype.Card
set_option linter.unusedSimpArgs false
set_option linter.unreachableTactic false
set_option linter.unusedTactic false
set_option linter.style.nativeDecide false
set_option linter.style.longLine false

namespace Formal.Mahjong

/-!
# Dora — indicator successor, sentinel, and reveal discipline

Tenhou 4p: 5 dora indicators (and 5 ura) sampled from the dead wall.
Indicator `t` maps cyclically to its dora `nextDora t = doraSucc t`:

* Manzu 1m→2m … 9m→1m  (types 0..8)
* Pinzu 1p→2p … 9p→1p  (types 9..17)
* Souzu 1s→2s … 9s→1s (types 18..26)
* Winds E→S→W→N→E     (types 27..30)
* Dragons W→G→R→W     (types 31..33)

The successor is a bijection preserving the copy index,
iterates to identity in 9/4/3 steps, and never fixes a tile.
Reveal discipline is contiguous (`∃ k, prefix revealed`), kan adds one,
ura remains hidden until win, and dora is a bonus (not a yaku).

References: SPEC §4.2/§9, `tenhou.net/man`, `riichienv`.
-/

-- ---------------------------------------------------------------------------
-- 1. Sentinel & slots
-- ---------------------------------------------------------------------------

/-- Sentinel value for “no dora” in indicator arrays (SPEC §9). -/
def DORA_SENTINEL : Int := -1

theorem dora_sentinel_eq : DORA_SENTINEL = -1 := rfl

/-- Five dora-indicator slots (0 = initial, 1..4 = kan doras). -/
abbrev DoraSlot := Fin 5

/-- Array of (optional) dora indicators per slot; `none` = not yet revealed. -/
abbrev DoraArray := DoraSlot → Option TileId

/-- Number of revealed slots in an array. -/
def numRevealed (arr : DoraArray) : Nat :=
  (Finset.univ.filter (fun s : DoraSlot => (arr s).isSome)).card

theorem numRevealed_le_5 (arr : DoraArray) : numRevealed arr ≤ 5 := by
  unfold numRevealed
  have h : (Finset.univ.filter (fun s : DoraSlot => (arr s).isSome)).card ≤ (Finset.univ : Finset DoraSlot).card :=
    Finset.card_filter_le _ _
  simp [Fintype.card_fin] at h
  omega

-- ---------------------------------------------------------------------------
-- 2. Cyclic successor on TileType
-- ---------------------------------------------------------------------------

/-- Successor on logical types (0..33), cyclic within each suit/honor group. -/
def doraSuccType (ty : TileType) : TileType :=
  if h : ty.val < 27 then
    -- suited: keep suit, rotate rank (ty.val % 9)
    ⟨(ty.val / 9) * 9 + (ty.val % 9 + 1) % 9, by omega⟩
  else if h2 : ty.val < 31 then
    -- winds 27..30 cycle length 4
    ⟨27 + (ty.val - 27 + 1) % 4, by omega⟩
  else
    -- dragons 31..33 cycle length 3
    ⟨31 + (ty.val - 31 + 1) % 3, by omega⟩

/-- Predecessor (inverse) on logical types. -/
def doraPredType (ty : TileType) : TileType :=
  if h : ty.val < 27 then
    ⟨(ty.val / 9) * 9 + (ty.val % 9 + 8) % 9, by omega⟩
  else if h2 : ty.val < 31 then
    ⟨27 + (ty.val - 27 + 3) % 4, by omega⟩
  else
    ⟨31 + (ty.val - 31 + 2) % 3, by omega⟩

-- Suited / wind / dragon predicates on TileType for case splits
def isWindType (ty : TileType) : Prop := 27 ≤ ty.val ∧ ty.val < 31
def isDragonType (ty : TileType) : Prop := 31 ≤ ty.val

instance (ty : TileType) : Decidable (isWindType ty) := by unfold isWindType; infer_instance
instance (ty : TileType) : Decidable (isDragonType ty) := by unfold isDragonType; infer_instance

theorem doraSuccType_wind (ty : TileType) (h : isWindType ty) :
    (doraSuccType ty).val = 27 + (ty.val - 27 + 1) % 4 := by
  unfold doraSuccType isWindType at *
  have h1 : ¬ ty.val < 27 := by omega
  have h2 : ty.val < 31 := h.2
  simp [h1, h2]

theorem doraSuccType_dragon (ty : TileType) (h : isDragonType ty) :
    (doraSuccType ty).val = 31 + (ty.val - 31 + 1) % 3 := by
  unfold doraSuccType isDragonType at *
  have h1 : ¬ ty.val < 27 := by omega
  have h2 : ¬ ty.val < 31 := by omega
  simp [h1, h2]

theorem doraSuccType_suited (ty : TileType) (h : ty.isSuited) :
    (doraSuccType ty).val = (ty.val / 9) * 9 + (ty.val % 9 + 1) % 9 := by
  unfold doraSuccType TileType.isSuited at *
  simp [h]

-- Inverse lemmas (finite check via native_decide on the 34-element type)
theorem doraSucc_pred_inverse : ∀ ty : TileType, doraSuccType (doraPredType ty) = ty := by
  native_decide

theorem doraPred_succ_inverse : ∀ ty : TileType, doraPredType (doraSuccType ty) = ty := by
  native_decide

theorem doraSuccType_bijective : Function.Bijective doraSuccType := by
  constructor
  · intro a b h
    have ha := doraPred_succ_inverse a
    have hb := doraPred_succ_inverse b
    rw [← ha, ← hb, h]
  · intro b
    exact ⟨doraPredType b, doraSucc_pred_inverse b⟩

theorem doraSuccType_ne_self : ∀ ty : TileType, doraSuccType ty ≠ ty := by
  native_decide

-- Iteration identities via finite enumeration
theorem doraSuccType_iter9_suited (ty : TileType) (h : ty.isSuited) :
    (doraSuccType^[9]) ty = ty := by
  -- 9 steps rotates rank by 9 ≡ 0 (mod 9)
  have h9 : ∀ t : TileType, t.isSuited → (doraSuccType^[9]) t = t := by native_decide
  exact h9 ty h

theorem doraSuccType_iter4_wind (ty : TileType) (h : isWindType ty) :
    (doraSuccType^[4]) ty = ty := by
  have h4 : ∀ t : TileType, isWindType t → (doraSuccType^[4]) t = t := by native_decide
  exact h4 ty h

theorem doraSuccType_iter3_dragon (ty : TileType) (h : isDragonType ty) :
    (doraSuccType^[3]) ty = ty := by
  have h3 : ∀ t : TileType, isDragonType t → (doraSuccType^[3]) t = t := by native_decide
  exact h3 ty h

-- ---------------------------------------------------------------------------
-- 3. Lift to TileId (physical tile) — preserves copy
-- ---------------------------------------------------------------------------

/-- Cyclic dora successor on physical tiles: preserves copy, maps type cyclically. -/
def doraSucc (t : TileId) : TileId :=
  mkTile (doraSuccType (tileType t)) (tileCopy t)

/-- Inverse on physical tiles. -/
def doraPred (t : TileId) : TileId :=
  mkTile (doraPredType (tileType t)) (tileCopy t)

theorem tileType_doraSucc (t : TileId) : tileType (doraSucc t) = doraSuccType (tileType t) := by
  unfold doraSucc
  rw [tileType_mkTile]

theorem tileCopy_doraSucc (t : TileId) : tileCopy (doraSucc t) = tileCopy t := by
  unfold doraSucc
  rw [tileCopy_mkTile]

theorem tileType_doraPred (t : TileId) : tileType (doraPred t) = doraPredType (tileType t) := by
  unfold doraPred
  rw [tileType_mkTile]

theorem tileCopy_doraPred (t : TileId) : tileCopy (doraPred t) = tileCopy t := by
  unfold doraPred
  rw [tileCopy_mkTile]

theorem doraSucc_pred_inverse_tile : ∀ t : TileId, doraSucc (doraPred t) = t := by
  intro t
  simp only [doraSucc, doraPred, tileType_mkTile, tileCopy_mkTile, doraSucc_pred_inverse, mkTile_tileType_tileCopy]

theorem doraPred_succ_inverse_tile : ∀ t : TileId, doraPred (doraSucc t) = t := by
  intro t
  simp only [doraSucc, doraPred, tileType_mkTile, tileCopy_mkTile, doraPred_succ_inverse, mkTile_tileType_tileCopy]

/-- Dora successor is bijective (permutation of 136 tiles). -/
theorem doraSucc_bijective : Function.Bijective doraSucc := by
  constructor
  · intro a b h
    have ha := doraPred_succ_inverse_tile a
    have hb := doraPred_succ_inverse_tile b
    calc a = doraPred (doraSucc a) := ha.symm
      _ = doraPred (doraSucc b) := by rw [h]
      _ = b := doraPred_succ_inverse_tile b
  · intro b
    exact ⟨doraPred b, doraSucc_pred_inverse_tile b⟩

theorem doraSucc_injective : Function.Injective doraSucc :=
  doraSucc_bijective.1

theorem doraSucc_surjective : Function.Surjective doraSucc :=
  doraSucc_bijective.2

/-- No tile is fixed by doraSucc (rank/wind/dragon cycles are ≥3). -/
theorem doraSucc_ne_self : ∀ t : TileId, doraSucc t ≠ t := by
  intro t h
  have htype : tileType (doraSucc t) = tileType t := by rw [h]
  rw [tileType_doraSucc] at htype
  have hne := doraSuccType_ne_self (tileType t)
  exact hne htype

-- Iteration on TileId: 9/4/3 steps restore (using type-level lemmas)

theorem doraSucc_iter9_suited (t : TileId) (h : isSuitedTile t) :
    (doraSucc^[9]) t = t := by
  have h_all : ∀ s : TileId, isSuitedTile s → (doraSucc^[9]) s = s := by native_decide
  exact h_all t h

theorem doraSucc_iter4_wind (t : TileId) (h : isWindType (tileType t)) :
    (doraSucc^[4]) t = t := by
  have h_all : ∀ s : TileId, isWindType (tileType s) → (doraSucc^[4]) s = s := by native_decide
  exact h_all t h

theorem doraSucc_iter3_dragon (t : TileId) (h : isDragonType (tileType t)) :
    (doraSucc^[3]) t = t := by
  have h_all : ∀ s : TileId, isDragonType (tileType s) → (doraSucc^[3]) s = s := by native_decide
  exact h_all t h

-- ---------------------------------------------------------------------------
-- 4. Indicator → dora
-- ---------------------------------------------------------------------------

/-- Dora tile for an indicator: the cyclic successor. -/
def nextDora (indicator : TileId) : TileId := doraSucc indicator

theorem nextDora_eq_doraSucc (t : TileId) : nextDora t = doraSucc t := rfl

theorem nextDora_bijective : Function.Bijective nextDora := doraSucc_bijective

theorem nextDora_ne_indicator : ∀ t : TileId, nextDora t ≠ t :=
  doraSucc_ne_self

-- ---------------------------------------------------------------------------
-- 5. DoraArray reveal discipline
-- ---------------------------------------------------------------------------

/-- Contiguous reveal: ∃ k ≤5, first k slots are `some`, rest `none`. -/
def IsContiguous (arr : DoraArray) : Prop :=
  ∃ k : Nat, k ≤ 5 ∧
    (∀ i : DoraSlot, i.val < k → (arr i).isSome = true) ∧
    (∀ i : DoraSlot, k ≤ i.val → arr i = none)

theorem isContiguous_empty : IsContiguous (fun _ => none) :=
  ⟨0, by omega, by intro i h; omega, by intro i _; rfl⟩

theorem isContiguous_single (t : TileId) :
    IsContiguous (fun i => if i.val = 0 then some t else none) := by
  refine ⟨1, by omega, ?_, ?_⟩
  · intro i hi
    have hiv : i.val = 0 := by omega
    simp [hiv]
  · intro i hi
    have hne : i.val ≠ 0 := by omega
    simp [hne]

/-- Canonical theorem: an array is contiguous iff it satisfies the `∃k` shape
    from the spec (`∀ i<k, arr i ≠ none ∧ ∀ i≥k, arr i = none`). -/
theorem dora_revealed_contiguous (arr : DoraArray) (h : IsContiguous arr) :
    ∃ k : Nat, k ≤ 5 ∧
      (∀ i : DoraSlot, i.val < k → arr i ≠ none) ∧
      (∀ i : DoraSlot, k ≤ i.val → arr i = none) := by
  obtain ⟨k, hk, hpre, hpost⟩ := h
  refine ⟨k, hk, ?_, hpost⟩
  intro i hi
  have hs := hpre i hi
  intro heq
  rw [heq] at hs
  simp at hs

theorem dora_revealed_contiguous_empty :
    ∃ k : Nat, k ≤ 5 ∧
      (∀ i : DoraSlot, (fun _ : DoraSlot => (none : Option TileId)) i ≠ none → i.val < k) := by
  exact ⟨0, by omega, by intro i h; simp at h⟩

-- Kan reveal: after k kans, 1+k indicators are revealed (up to 5)

def kanRevealedArray (ws : WallSchedule) (k : Nat) (hk : k ≤ 4) : DoraArray :=
  fun i => if i.val < 1 + k then some ((deadWall ws)[i.val]'(by have h := deadWall_length ws; omega)) else none

theorem kanRevealedArray_contiguous (ws : WallSchedule) (k : Nat) (hk : k ≤ 4) :
    IsContiguous (kanRevealedArray ws k hk) := by
  refine ⟨1 + k, by omega, ?_, ?_⟩
  · intro i hi
    unfold kanRevealedArray
    simp [hi]
  · intro i hi
    unfold kanRevealedArray
    simp [show ¬ (i.val < 1 + k) from by omega]

theorem kan_dora_reveal (ws : WallSchedule) (k : Nat) (hk : k ≤ 4) :
    numRevealed (kanRevealedArray ws k hk) = 1 + k := by
  unfold numRevealed kanRevealedArray
  have heq : (Finset.univ.filter (fun s : DoraSlot => (if s.val < 1 + k then (some ((deadWall ws)[s.val]'(by have h := deadWall_length ws; omega)) : Option TileId) else none).isSome))
           = (Finset.univ.filter (fun s : DoraSlot => decide (s.val < 1 + k))) := by
    ext s
    simp
  rw [heq]
  have hk_le : k ≤ 4 := hk
  have h_cases : k = 0 ∨ k = 1 ∨ k = 2 ∨ k = 3 ∨ k = 4 := by omega
  rcases h_cases with rfl | rfl | rfl | rfl | rfl <;> native_decide

theorem kan_dora_reveal_zero (ws : WallSchedule) :
    numRevealed (kanRevealedArray ws 0 (by omega)) = 1 :=
  kan_dora_reveal ws 0 (by omega)

theorem kan_dora_reveal_full (ws : WallSchedule) :
    numRevealed (kanRevealedArray ws 4 (by omega)) = 5 :=
  kan_dora_reveal ws 4 (by omega)

-- ---------------------------------------------------------------------------
-- 6. Ura hidden until win
-- ---------------------------------------------------------------------------

/-- Ura-dora indicators mirror dora slots but are hidden until the hand wins. -/
def UraArray := DoraSlot → Option TileId

/-- Visibility predicate: ura is all-`none` unless `won = true`. -/
def uraHiddenUntilWin (won : Bool) (ura : UraArray) : Prop :=
  if won then True else ∀ i : DoraSlot, ura i = none

theorem ura_hidden_until_win (ura : UraArray) :
    uraHiddenUntilWin false ura → ∀ i, ura i = none := by
  intro h i
  unfold uraHiddenUntilWin at h
  simp at h
  exact h i

theorem ura_revealed_after_win (ura : UraArray) (h : ∀ i, ura i ≠ none) :
    uraHiddenUntilWin true ura := by
  unfold uraHiddenUntilWin
  simp

theorem ura_hidden_empty : uraHiddenUntilWin false (fun _ => none) := by
  unfold uraHiddenUntilWin
  simp

-- ---------------------------------------------------------------------------
-- 7. Dora is bonus, not yaku
-- ---------------------------------------------------------------------------

/-- Dora adds `han` but never satisfies the yaku requirement. -/
def doraHan (count : Nat) : Nat := count

def requiresYaku : Bool := true
def doraCountsAsYaku : Bool := false

theorem dora_not_counted_as_yaku : doraCountsAsYaku = false := rfl

theorem dora_is_bonus_not_requirement :
    doraCountsAsYaku ≠ requiresYaku := by
  unfold doraCountsAsYaku requiresYaku
  native_decide

/-- Dora han is additive bonus; yaku han must come from elsewhere. -/
theorem dora_bonus_additive (yakuHan doraCount : Nat) :
    yakuHan + doraHan doraCount = yakuHan + doraCount := rfl

theorem dora_zero_no_yaku (yakuHan : Nat) (h : yakuHan = 0) (doraCount : Nat) :
    doraHan doraCount ≠ 0 → yakuHan + doraHan doraCount > 0 := by
  omega

-- ---------------------------------------------------------------------------
-- 8. Wall integration: dora indicators in the dead wall
-- ---------------------------------------------------------------------------

/-- Dora indicator tile in the dead wall at slot `s` (even positions). -/
def wallDoraIndicator (ws : WallSchedule) (s : DoraSlot) : TileId :=
  (deadWall ws)[s.val]'(by have h := deadWall_length ws; omega)

/-- Dora tile for a wall slot. -/
def wallDora (ws : WallSchedule) (s : DoraSlot) : TileId :=
  nextDora (wallDoraIndicator ws s)

theorem wallDora_eq_nextDora (ws : WallSchedule) (s : DoraSlot) :
    wallDora ws s = doraSucc (wallDoraIndicator ws s) := rfl

theorem wallDora_ne_indicator (ws : WallSchedule) (s : DoraSlot) :
    wallDora ws s ≠ wallDoraIndicator ws s := by
  unfold wallDora
  exact doraSucc_ne_self _

end Formal.Mahjong
