import Mathlib.Data.Fin.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Card
import Mathlib.Data.Fintype.Card
import Mathlib.Tactic

set_option linter.unusedSimpArgs false
set_option linter.unreachableTactic false
set_option linter.unusedTactic false
set_option linter.style.nativeDecide false
set_option linter.style.longLine false

/-!
# Hydra2 Mahjong Tile Foundations — 4-player Tenhou

Physical tiles `0..135` (`TileId := Fin 136`), logical types `0..33`
(`TileType := Fin 34`), per-type copy `0..3` (`Copy := Fin 4`).

The encoding `id = type*4 + copy` is the Tenhou / mjlog
`136`-array representation. Suit/Honor partition and red-dora
sentinels `{16,52,88}` are formalized with `Finset` cardinality
lemmas. This file is the root for `Wall`, `Dora`, `Shanten`, `Yaku`.

References: `SPEC §4.1`, `tenhou.net/man`, `riichienv` tile table.
-/

namespace Formal.Mahjong

-- ---------------------------------------------------------------------------
-- 1. Core types: physical id, logical type, copy
-- ---------------------------------------------------------------------------

/-- Physical tile id — exactly the 136 tiles in a 4-player wall. -/
abbrev TileId := Fin 136

/-- Logical tile type — 34 distinct faces (27 suited + 7 honors). -/
abbrev TileType := Fin 34

/-- Copy index within a tile type — 4 identical copies. -/
abbrev Copy := Fin 4

/-- Suit for numbered tiles: 0 = manzu (萬子), 1 = pinzu (筒子), 2 = souzu (索子). -/
abbrev Suit := Fin 3

/-- Honor kind: 0-3 winds (ton/nan/sha/pei), 4-6 dragons (haku/hatsu/chun). -/
abbrev Honor := Fin 7

-- ---------------------------------------------------------------------------
-- 2. Encoding / decoding: id ↔ (type, copy)
-- ---------------------------------------------------------------------------

/-- Logical type of a physical tile: `type = id / 4` (integer division). -/
def tileType (t : TileId) : TileType :=
  ⟨t.val / 4, by have ht := t.isLt; omega⟩

/-- Copy index of a physical tile: `copy = id % 4`. -/
def tileCopy (t : TileId) : Copy :=
  ⟨t.val % 4, Nat.mod_lt _ (by omega)⟩

/-- Construct a physical tile from a logical type and copy index.
    Inverse of `(tileType, tileCopy)`. -/
def mkTile (ty : TileType) (c : Copy) : TileId :=
  ⟨ty.val * 4 + c.val, by have h1 := ty.isLt; have h2 := c.isLt; omega⟩

-- ---------------------------------------------------------------------------
-- 3. Round-trip lemmas
-- ---------------------------------------------------------------------------

theorem tileType_mkTile (ty : TileType) (c : Copy) :
    tileType (mkTile ty c) = ty := by
  apply Fin.ext
  simp only [tileType, mkTile]
  have hc := c.isLt
  omega

theorem tileCopy_mkTile (ty : TileType) (c : Copy) :
    tileCopy (mkTile ty c) = c := by
  apply Fin.ext
  simp only [tileCopy, mkTile]
  have hty := ty.isLt
  have hc := c.isLt
  omega

theorem mkTile_tileType_tileCopy (t : TileId) :
    mkTile (tileType t) (tileCopy t) = t := by
  apply Fin.ext
  simp only [mkTile, tileType, tileCopy]
  have ht := t.isLt
  omega

theorem mkTile_tileType_eq (ty : TileType) (c : Copy) :
    tileType (mkTile ty c) = ty :=
  tileType_mkTile ty c

theorem mkTile_tileCopy_eq (ty : TileType) (c : Copy) :
    tileCopy (mkTile ty c) = c :=
  tileCopy_mkTile ty c

theorem tileType_mkTile_tileCopy_roundtrip (t : TileId) :
    mkTile (tileType t) (tileCopy t) = t :=
  mkTile_tileType_tileCopy t

theorem mkTile_injective_type_copy :
    ∀ (ty1 ty2 : TileType) (c1 c2 : Copy),
      mkTile ty1 c1 = mkTile ty2 c2 → ty1 = ty2 ∧ c1 = c2 := by
  intro ty1 ty2 c1 c2 h
  have hval : ty1.val * 4 + c1.val = ty2.val * 4 + c2.val := by
    have := congrArg Fin.val h
    simp only [mkTile] at this
    exact this
  have hty1 := ty1.isLt
  have hty2 := ty2.isLt
  have hc1 := c1.isLt
  have hc2 := c2.isLt
  constructor
  · apply Fin.ext; omega
  · apply Fin.ext; omega

theorem mkTile_injective (ty : TileType) : Function.Injective (mkTile ty) := by
  intro c1 c2 h
  have := mkTile_injective_type_copy ty ty c1 c2 h
  exact this.2

theorem tileType_surjective : Function.Surjective tileType := by
  intro ty
  exact ⟨mkTile ty ⟨0, by omega⟩, tileType_mkTile ty _⟩

theorem tileCopy_surjective_for_type (ty : TileType) :
    ∀ c : Copy, ∃ t : TileId, tileType t = ty ∧ tileCopy t = c := by
  intro c
  exact ⟨mkTile ty c, tileType_mkTile ty c, tileCopy_mkTile ty c⟩

-- ---------------------------------------------------------------------------
-- 4. Suit / honor partition on logical types
-- ---------------------------------------------------------------------------

/-- Suited tile types are `0..26` (27 tiles: 3 suits × 9 ranks). -/
def TileType.isSuited (ty : TileType) : Prop :=
  ty.val < 27

/-- Honor tile types are `27..33` (7 tiles: 4 winds + 3 dragons). -/
def TileType.isHonor (ty : TileType) : Prop :=
  27 ≤ ty.val

instance (ty : TileType) : Decidable (ty.isSuited) :=
  inferInstanceAs (Decidable (ty.val < 27))

instance (ty : TileType) : Decidable (ty.isHonor) :=
  inferInstanceAs (Decidable (27 ≤ ty.val))

/-- Suit index for suited types with proof obligation. -/
def TileType.suitOf (ty : TileType) (h : ty.isSuited) : Suit :=
  ⟨ty.val / 9, by unfold TileType.isSuited at h; omega⟩

/-- Total suit projection (`%3` to stay in bounds for honors): matches
    `suitOf` on suited types, clamped via modulo. -/
def TileType.suit (ty : TileType) : Suit :=
  ⟨(ty.val / 9) % 3, Nat.mod_lt _ (by omega)⟩

theorem TileType.suit_of_suited (ty : TileType) (h : ty.isSuited) :
    (ty.suit).val = ty.val / 9 := by
  unfold TileType.suit
  have : ty.val / 9 < 3 := by unfold TileType.isSuited at h; omega
  have : (ty.val / 9) % 3 = ty.val / 9 := Nat.mod_eq_of_lt this
  simp [this]

/-- Rank within suit: `rank = type % 9` (0 = 1, ..., 8 = 9). -/
def TileType.rank (ty : TileType) : Fin 9 :=
  ⟨ty.val % 9, Nat.mod_lt _ (by omega)⟩

/-- Honor index for honor types: `honor = type - 27`. -/
def TileType.honorIndex (ty : TileType) : Honor :=
  ⟨ty.val - 27, by have := ty.isLt; omega⟩

def isSuitedTile (t : TileId) : Prop := (tileType t).isSuited
def isHonorTile (t : TileId) : Prop := (tileType t).isHonor

instance (t : TileId) : Decidable (isSuitedTile t) := by
  unfold isSuitedTile; infer_instance
instance (t : TileId) : Decidable (isHonorTile t) := by
  unfold isHonorTile; infer_instance

theorem suited_or_honor (ty : TileType) : ty.isSuited ∨ ty.isHonor := by
  unfold TileType.isSuited TileType.isHonor
  omega

theorem not_suited_and_honor (ty : TileType) : ¬(ty.isSuited ∧ ty.isHonor) := by
  unfold TileType.isSuited TileType.isHonor
  omega

theorem suited_iff_not_honor (ty : TileType) : ty.isSuited ↔ ¬ty.isHonor := by
  constructor
  · intro h hh; exact not_suited_and_honor ty ⟨h, hh⟩
  · intro h; have := suited_or_honor ty; tauto

theorem honor_iff_not_suited (ty : TileType) : ty.isHonor ↔ ¬ty.isSuited := by
  constructor
  · intro h hh; exact not_suited_and_honor ty ⟨hh, h⟩
  · intro h; have := suited_or_honor ty; tauto

theorem suited_em (ty : TileType) : ty.isSuited ∨ ¬ty.isSuited := Decidable.em _

theorem honor_em (ty : TileType) : ty.isHonor ∨ ¬ty.isHonor := Decidable.em _

def both_decidable (ty : TileType) :
    Decidable (ty.isSuited) × Decidable (ty.isHonor) :=
  ⟨inferInstance, inferInstance⟩

-- Suited / honor cardinalities on TileType

theorem suited_tileTypes_card :
    (Finset.univ.filter (fun ty : TileType => ty.isSuited)).card = 27 := by
  native_decide

theorem honor_tileTypes_card :
    (Finset.univ.filter (fun ty : TileType => ty.isHonor)).card = 7 := by
  native_decide

theorem suited_plus_honor_card :
    (Finset.univ.filter (fun ty : TileType => ty.isSuited)).card +
    (Finset.univ.filter (fun ty : TileType => ty.isHonor)).card = 34 := by
  rw [suited_tileTypes_card, honor_tileTypes_card]

-- ---------------------------------------------------------------------------
-- 5. Red dora sentinels — the three aka tiles (5mr, 5pr, 5sr)
-- ---------------------------------------------------------------------------

/-- Red tile ids: 5-man second copy = 16, 5-pin = 52, 5-sou = 88.
    Each is `type*4 + 0` for types `4, 13, 22` (the marked aka copy).
    Tenhou replaces one regular copy per 5-suit with an aka. -/
def redTileIds : Finset TileId :=
  {⟨16, by omega⟩, ⟨52, by omega⟩, ⟨88, by omega⟩}

theorem redTileIds_card : redTileIds.card = 3 := by
  native_decide

theorem redTileIds_nonempty : redTileIds.Nonempty := by
  rw [Finset.nonempty_iff_ne_empty]
  intro h; have := redTileIds_card; rw [h] at this; simp at this

theorem red_mem_16 : (⟨16, by omega⟩ : TileId) ∈ redTileIds := by
  native_decide

theorem red_mem_52 : (⟨52, by omega⟩ : TileId) ∈ redTileIds := by
  native_decide

theorem red_mem_88 : (⟨88, by omega⟩ : TileId) ∈ redTileIds := by
  native_decide

theorem red_ids_distinct :
    (⟨16, by omega⟩ : TileId) ≠ (⟨52, by omega⟩ : TileId) ∧
    (⟨16, by omega⟩ : TileId) ≠ (⟨88, by omega⟩ : TileId) ∧
    (⟨52, by omega⟩ : TileId) ≠ (⟨88, by omega⟩ : TileId) := by
  refine ⟨?_, ?_, ?_⟩ <;> native_decide

theorem red_ids_in_range :
    ∀ t ∈ redTileIds, t.val < 136 := by
  intro t ht
  have := t.isLt
  exact this

theorem red_ids_tileType_values :
    tileType ⟨16, by omega⟩ = (⟨4, by omega⟩ : TileType) ∧
    tileType ⟨52, by omega⟩ = (⟨13, by omega⟩ : TileType) ∧
    tileType ⟨88, by omega⟩ = (⟨22, by omega⟩ : TileType) := by
  refine ⟨?_, ?_, ?_⟩ <;> native_decide

theorem red_ids_are_suited :
    isSuitedTile ⟨16, by omega⟩ ∧
    isSuitedTile ⟨52, by omega⟩ ∧
    isSuitedTile ⟨88, by omega⟩ := by
  refine ⟨?_, ?_, ?_⟩ <;> native_decide

-- ---------------------------------------------------------------------------
-- 6. Core counting theorems: conservation & per-type copies
-- ---------------------------------------------------------------------------

theorem tileId_range (t : TileId) : t.val < 136 := t.isLt

theorem tile_conservation_count :
    (Finset.univ : Finset TileId).card = 136 := by
  simp [Fintype.card_fin]

theorem tileType_card : Fintype.card TileType = 34 := by
  simp [Fintype.card_fin]

theorem copy_card : Fintype.card Copy = 4 := by
  simp [Fintype.card_fin]

theorem tile_conservation_via_types :
    Fintype.card TileType * Fintype.card Copy = 136 := by
  rw [tileType_card, copy_card]

theorem logical_count : (Finset.univ : Finset TileType).card = 34 := by
  simp [Fintype.card_fin]

theorem physical_count : (Finset.univ : Finset TileId).card = 136 :=
  tile_conservation_count

theorem logical_times_copies_eq_physical :
    (Finset.univ : Finset TileType).card * 4 = (Finset.univ : Finset TileId).card := by
  rw [logical_count, physical_count]

-- Per-type fiber size = 4

theorem tileType_copies (ty : TileType) :
    (Finset.univ.filter (fun t : TileId => tileType t = ty)).card = 4 := by
  have h_eq : (Finset.univ.filter (fun t : TileId => tileType t = ty)) =
      (Finset.univ : Finset Copy).image (mkTile ty) := by
    ext t
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_image]
    constructor
    · intro h
      refine ⟨tileCopy t, ?_⟩
      have hmk := mkTile_tileType_tileCopy t
      rw [h] at hmk
      exact hmk
    · intro ⟨c, hc⟩
      rw [← hc, tileType_mkTile]
  rw [h_eq, Finset.card_image_of_injective _ (mkTile_injective ty)]
  simp [Fintype.card_fin]

theorem tileType_copies_univ_sum :
    ∑ _ty : TileType, (Finset.univ.filter (fun t : TileId => tileType t = _ty)).card = 136 := by
  simp only [tileType_copies]
  native_decide

-- ---------------------------------------------------------------------------
-- 7. Red dedup via Finset image //4  (physical 3 vs logical 3)
-- ---------------------------------------------------------------------------

theorem redDedup_card :
    (redTileIds.image tileType).card = 3 := by
  native_decide

theorem redDedup_subset_logical :
    redTileIds.image tileType ⊆ (Finset.univ : Finset TileType) := by
  intro ty _
  exact Finset.mem_univ ty

theorem redDedup_types_are_4_13_22 :
    redTileIds.image tileType = ({⟨4, by omega⟩, ⟨13, by omega⟩, ⟨22, by omega⟩} : Finset TileType) := by
  native_decide

theorem red_physical_vs_logical_card :
    redTileIds.card = (redTileIds.image tileType).card := by
  rw [redTileIds_card, redDedup_card]

-- ---------------------------------------------------------------------------
-- 8. Fintype instances & image covering
-- ---------------------------------------------------------------------------

-- `TileId`, `TileType`, `Copy` already have `Fintype` via `Fin n`.
-- We record the card lemmas explicitly for downstream `Wall`/`Dora`.

theorem tileType_image_univ :
    (Finset.univ : Finset TileId).image tileType = Finset.univ := by
  ext ty
  simp only [Finset.mem_image, Finset.mem_univ, true_and]
  constructor
  · intro _; trivial
  · intro _
    exact ⟨mkTile ty ⟨0, by omega⟩, tileType_mkTile ty _⟩

theorem tileType_image_card :
    ((Finset.univ : Finset TileId).image tileType).card = 34 := by
  rw [tileType_image_univ]
  simp [Fintype.card_fin]

theorem copiesOf_tileType_injective :
    ∀ ty : TileType, Set.InjOn (fun c : Copy => mkTile ty c) Set.univ := by
  intro ty c1 _ c2 _ h
  exact mkTile_injective ty h

-- ---------------------------------------------------------------------------
-- 9. Tile constants for Wall / Dora / Shanten
-- ---------------------------------------------------------------------------

/-- Number of tiles per wall in 4p: 136. -/
def wallSize : Nat := 136

theorem wallSize_eq : wallSize = 136 := rfl

theorem wallSize_eq_card_univ : wallSize = (Finset.univ : Finset TileId).card := by
  simp [wallSize, tile_conservation_count]

/-- Logical types: 34. -/
def logicalSize : Nat := 34

theorem logicalSize_eq : logicalSize = 34 := rfl

theorem logicalSize_eq_card : logicalSize = (Finset.univ : Finset TileType).card := by
  simp [logicalSize, logical_count]

end Formal.Mahjong
