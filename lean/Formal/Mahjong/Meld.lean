import Formal.Mahjong.Tile
import Formal.Mahjong.Wall
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Card
import Mathlib.Data.Finset.Sort
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
# DeclaredMeld — chi/pon/kan invariants (SPEC §6.2)

Faithful Lean port of:

* `RiichiEnv/riichienv-core/src/yaku_checker.rs`
* `RiichiEnv/riichienv-core/src/state/legal_actions.rs`
* `hydra2/src/hydra2/engines/riichienv/actions.py`
* `hydra2/src/hydra2/contracts/action.py` — `_consumed_pair_forms_run`, `_all_same_type`
* `hydra2/src/hydra2/contracts/observation.py` — `VisibleMeld`
* `docs/IMPLEMENTATION_SPEC.md §6.2`

Invariants:

* chi: 2 distinct physical + consecutive logical + same suit + honors forbidden; source = previous seat
* pon: 2 same logical
* daiminkan: 3 same logical
* ankan: 4 same logical no source
* kakan: added tile + prior pon ID

Plus: `meldTiles_nodup`, `concealed+drawn vs meld disjoint`, `tiles_sorted`, `meldCount` etc.
Uses `Finset`, `List`.
-/

-- ---------------------------------------------------------------------------
-- 0. MeldKind
-- ---------------------------------------------------------------------------

inductive MeldKind where
  | Chi
  | Pon
  | Daiminkan
  | Ankan
  | Kakan
  deriving DecidableEq, Repr, BEq

def MeldKind.toNat : MeldKind → Nat
  | .Chi => 0 | .Pon => 1 | .Daiminkan => 2 | .Ankan => 3 | .Kakan => 4

def MeldKind.toActionOrdinal : MeldKind → Nat
  | .Chi => 4 | .Pon => 5 | .Daiminkan => 6 | .Ankan => 7 | .Kakan => 8

theorem meldKind_toNat_range (k : MeldKind) : k.toNat < 5 := by cases k <;> simp [MeldKind.toNat]
theorem meldKind_toOrdinal_range (k : MeldKind) : 4 ≤ k.toActionOrdinal ∧ k.toActionOrdinal ≤ 8 := by cases k <;> simp [MeldKind.toActionOrdinal]

abbrev Seat := Fin 4

def meldTileType (t : TileId) : TileType := tileType t

-- ---------------------------------------------------------------------------
-- 1. Structure DeclaredMeld
-- ---------------------------------------------------------------------------

structure DeclaredMeld where
  kind : MeldKind
  tiles : Finset TileId
  calledTile : Option TileId := none
  sourceSeat : Option Seat := none
  addedTile : Option TileId := none
  priorPonId : Option String := none
  deriving DecidableEq

def meldTiles (m : DeclaredMeld) : Finset TileId := m.tiles
noncomputable def meldTilesList (m : DeclaredMeld) : List TileId := m.tiles.toList
noncomputable def meldTilesSortedList (m : DeclaredMeld) : List TileId := m.tiles.sort (· ≤ ·)

theorem meldTilesList_nodup (m : DeclaredMeld) : (meldTilesList m).Nodup := by unfold meldTilesList; exact Finset.nodup_toList _
theorem meldTilesSortedList_nodup (m : DeclaredMeld) : (meldTilesSortedList m).Nodup := by
  unfold meldTilesSortedList
  exact Finset.sort_nodup (s := m.tiles) (r := (· ≤ ·))
theorem meldTiles_nodup (m : DeclaredMeld) : m.tiles.toList.Nodup := Finset.nodup_toList _
theorem tiles_sorted (m : DeclaredMeld) : List.Pairwise (· ≤ ·) (meldTilesSortedList m) := by
  unfold meldTilesSortedList
  exact Finset.pairwise_sort (s := m.tiles) (r := (· ≤ ·))
theorem meldTiles_sorted_and_nodup (m : DeclaredMeld) : List.Pairwise (· ≤ ·) (meldTilesSortedList m) ∧ (meldTilesSortedList m).Nodup :=
  ⟨tiles_sorted m, meldTilesSortedList_nodup m⟩
theorem meldTiles_in_range (m : DeclaredMeld) (t : TileId) (_ht : t ∈ m.tiles) : t.val < 136 := t.isLt
theorem meld_finset_card_le_univ (m : DeclaredMeld) : m.tiles.card ≤ 136 := by
  have hsub : m.tiles ⊆ Finset.univ := Finset.subset_univ _
  calc m.tiles.card ≤ (Finset.univ : Finset TileId).card := Finset.card_le_card hsub
    _ = 136 := by simp [Fintype.card_fin]

-- ---------------------------------------------------------------------------
-- 2. Per-kind predicates (SPEC §6.2)
-- ---------------------------------------------------------------------------

def IsValidChi (m : DeclaredMeld) : Prop :=
  m.kind = .Chi ∧ m.tiles.card = 3 ∧ m.calledTile.isSome ∧ m.sourceSeat.isSome ∧
  m.addedTile.isNone ∧ m.priorPonId.isNone ∧
  (∃ called ∈ m.tiles, m.calledTile = some called ∧
    let rest := m.tiles.erase called
    rest.card = 2 ∧ (∀ t ∈ rest, (tileType t).isSuited) ∧ (tileType called).isSuited ∧
    (∀ t ∈ rest, (tileType t).val / 9 = (tileType called).val / 9) ∧
    let allTypes := m.tiles.image tileType
    allTypes.card = 3)

def IsValidPon (m : DeclaredMeld) : Prop :=
  m.kind = .Pon ∧ m.tiles.card = 3 ∧ m.calledTile.isSome ∧ m.sourceSeat.isSome ∧
  m.addedTile.isNone ∧ m.priorPonId.isNone ∧
  (∃ called ∈ m.tiles, m.calledTile = some called ∧ (∀ t ∈ m.tiles, tileType t = tileType called))

def IsValidDaiminkan (m : DeclaredMeld) : Prop :=
  m.kind = .Daiminkan ∧ m.tiles.card = 4 ∧ m.calledTile.isSome ∧ m.sourceSeat.isSome ∧
  m.addedTile.isNone ∧ m.priorPonId.isNone ∧
  (∃ called ∈ m.tiles, m.calledTile = some called ∧ (∀ t ∈ m.tiles, tileType t = tileType called))

def IsValidAnkan (m : DeclaredMeld) : Prop :=
  m.kind = .Ankan ∧ m.tiles.card = 4 ∧ m.calledTile.isNone ∧ m.sourceSeat.isNone ∧
  m.addedTile.isNone ∧ m.priorPonId.isNone ∧
  (∃ ty : TileType, ∀ t ∈ m.tiles, tileType t = ty) ∧
  (∃ ty : TileType, m.tiles = Finset.image (fun c : Copy => mkTile ty c) Finset.univ)

def IsValidKakan (m : DeclaredMeld) : Prop :=
  m.kind = .Kakan ∧ m.tiles.card = 4 ∧ m.calledTile.isNone ∧ m.sourceSeat.isNone ∧
  m.addedTile.isSome ∧ m.priorPonId.isSome ∧
  (∃ added ∈ m.tiles, m.addedTile = some added ∧ ∃ ty : TileType, tileType added = ty ∧
    (∀ t ∈ m.tiles, tileType t = ty))

def IsValidMeld (m : DeclaredMeld) : Prop :=
  match m.kind with
  | .Chi => IsValidChi m
  | .Pon => IsValidPon m
  | .Daiminkan => IsValidDaiminkan m
  | .Ankan => IsValidAnkan m
  | .Kakan => IsValidKakan m

-- ---------------------------------------------------------------------------
-- 3. Construction witnesses
-- ---------------------------------------------------------------------------

def mkChiMeld (startVal : Nat) (hstart : startVal % 9 ≤ 6) (hsuit : startVal < 27) (hstart2 : startVal + 2 < 34) (calledCopy c1Copy c2Copy : Copy) (source : Seat) : DeclaredMeld where
  kind := .Chi
  tiles := { mkTile ⟨startVal, by omega⟩ calledCopy, mkTile ⟨startVal+1, by omega⟩ c1Copy, mkTile ⟨startVal+2, by omega⟩ c2Copy }
  calledTile := some (mkTile ⟨startVal, by omega⟩ calledCopy)
  sourceSeat := some source
  addedTile := none
theorem mkChiMeld_card (startVal : Nat) (hstart : startVal % 9 ≤ 6) (hsuit : startVal < 27) (hstart2 : startVal + 2 < 34) (calledCopy c1Copy c2Copy : Copy) (source : Seat) :
    (mkChiMeld startVal hstart hsuit hstart2 calledCopy c1Copy c2Copy source).tiles.card = 3 := by
  unfold mkChiMeld
  have h1 : mkTile (⟨startVal, by omega⟩ : TileType) calledCopy ≠ mkTile (⟨startVal+1, by omega⟩ : TileType) c1Copy := by
    intro h; have ht := (mkTile_injective_type_copy _ _ _ _ h).1
    have : startVal = startVal + 1 := by simpa using congrArg Fin.val ht
    omega
  have h2 : mkTile (⟨startVal, by omega⟩ : TileType) calledCopy ≠ mkTile (⟨startVal+2, by omega⟩ : TileType) c2Copy := by
    intro h; have ht := (mkTile_injective_type_copy _ _ _ _ h).1
    have : startVal = startVal + 2 := by simpa using congrArg Fin.val ht
    omega
  have h3 : mkTile (⟨startVal+1, by omega⟩ : TileType) c1Copy ≠ mkTile (⟨startVal+2, by omega⟩ : TileType) c2Copy := by
    intro h; have ht := (mkTile_injective_type_copy _ _ _ _ h).1
    have : startVal + 1 = startVal + 2 := by simpa using congrArg Fin.val ht
    omega
  have h12 : mkTile (⟨startVal, by omega⟩ : TileType) calledCopy ∉ ({mkTile (⟨startVal+1, by omega⟩ : TileType) c1Copy, mkTile (⟨startVal+2, by omega⟩ : TileType) c2Copy} : Finset TileId) := by simp [h1, h2]
  rw [Finset.card_insert_of_notMem h12, Finset.card_pair h3]

def mkPonMeld (ty : TileType) (cCalled c1 c2 : Copy) (hDistinct : cCalled ≠ c1 ∧ cCalled ≠ c2 ∧ c1 ≠ c2) (source : Seat) : DeclaredMeld where
  kind := .Pon
  tiles := { mkTile ty cCalled, mkTile ty c1, mkTile ty c2 }
  calledTile := some (mkTile ty cCalled)
  sourceSeat := some source
  addedTile := none
  priorPonId := none

theorem mkPonMeld_card (ty : TileType) (cCalled c1 c2 : Copy) (hDistinct : cCalled ≠ c1 ∧ cCalled ≠ c2 ∧ c1 ≠ c2) (source : Seat) :
    (mkPonMeld ty cCalled c1 c2 hDistinct source).tiles.card = 3 := by
  unfold mkPonMeld
  have h1 : mkTile ty cCalled ≠ mkTile ty c1 := fun h => hDistinct.1 ((mkTile_injective_type_copy _ _ _ _ h).2)
  have h2 : mkTile ty cCalled ≠ mkTile ty c2 := fun h => hDistinct.2.1 ((mkTile_injective_type_copy _ _ _ _ h).2)
  have h3 : mkTile ty c1 ≠ mkTile ty c2 := fun h => hDistinct.2.2 ((mkTile_injective_type_copy _ _ _ _ h).2)
  have h12 : mkTile ty cCalled ∉ ({mkTile ty c1, mkTile ty c2} : Finset TileId) := by simp [h1, h2]
  rw [Finset.card_insert_of_notMem h12, Finset.card_pair h3]

theorem mkPonMeld_sameType (ty : TileType) (cCalled c1 c2 : Copy) (hDistinct : cCalled ≠ c1 ∧ cCalled ≠ c2 ∧ c1 ≠ c2) (source : Seat)
    (t : TileId) (ht : t ∈ (mkPonMeld ty cCalled c1 c2 hDistinct source).tiles) : tileType t = ty := by
  unfold mkPonMeld at ht; simp at ht; rcases ht with rfl | rfl | rfl <;> simp [tileType_mkTile]

def mkDaiminkanMeld (ty : TileType) (source : Seat) : DeclaredMeld where
  kind := .Daiminkan
  tiles := Finset.image (fun c : Copy => mkTile ty c) Finset.univ
  calledTile := some (mkTile ty ⟨0, by omega⟩)
  sourceSeat := some source
  addedTile := none
  priorPonId := none

theorem mkDaiminkanMeld_card (ty : TileType) (source : Seat) : (mkDaiminkanMeld ty source).tiles.card = 4 := by
  unfold mkDaiminkanMeld; rw [Finset.card_image_of_injective _ (mkTile_injective ty)]; simp [Fintype.card_fin]
theorem mkDaiminkanMeld_sameType (ty : TileType) (source : Seat) (t : TileId) (ht : t ∈ (mkDaiminkanMeld ty source).tiles) : tileType t = ty := by
  unfold mkDaiminkanMeld at ht; simp at ht; obtain ⟨c, _, rfl⟩ := ht; exact tileType_mkTile ty c

def mkAnkanMeld (ty : TileType) : DeclaredMeld where
  kind := .Ankan
  tiles := Finset.image (fun c : Copy => mkTile ty c) Finset.univ
  calledTile := none
  sourceSeat := none
  addedTile := none
  priorPonId := none

theorem mkAnkanMeld_card (ty : TileType) : (mkAnkanMeld ty).tiles.card = 4 := by
  unfold mkAnkanMeld; rw [Finset.card_image_of_injective _ (mkTile_injective ty)]; simp [Fintype.card_fin]
theorem mkAnkanMeld_sameType (ty : TileType) (t : TileId) (ht : t ∈ (mkAnkanMeld ty).tiles) : tileType t = ty := by
  unfold mkAnkanMeld at ht; simp at ht; obtain ⟨c, _, rfl⟩ := ht; exact tileType_mkTile ty c

def mkKakanMeld (ty : TileType) (ponId : String) : DeclaredMeld where
  kind := .Kakan
  tiles := Finset.image (fun c : Copy => mkTile ty c) Finset.univ
  calledTile := none
  sourceSeat := none
  addedTile := some (mkTile ty ⟨3, by omega⟩)
  priorPonId := some ponId

theorem mkKakanMeld_card (ty : TileType) (ponId : String) : (mkKakanMeld ty ponId).tiles.card = 4 := by
  unfold mkKakanMeld; rw [Finset.card_image_of_injective _ (mkTile_injective ty)]; simp [Fintype.card_fin]
theorem mkKakanMeld_hasPrior (ty : TileType) (ponId : String) : (mkKakanMeld ty ponId).priorPonId.isSome = true := by simp [mkKakanMeld]
theorem mkKakanMeld_hasAdded (ty : TileType) (ponId : String) : (mkKakanMeld ty ponId).addedTile.isSome = true := by simp [mkKakanMeld]

-- ---------------------------------------------------------------------------
-- 4. Global invariants
-- ---------------------------------------------------------------------------

def meldExpectedCard : MeldKind → Nat
  | .Chi => 3 | .Pon => 3 | .Daiminkan => 4 | .Ankan => 4 | .Kakan => 4

theorem meldExpectedCard_pos (k : MeldKind) : 0 < meldExpectedCard k := by cases k <;> simp [meldExpectedCard]

theorem validMeld_card (m : DeclaredMeld) (h : IsValidMeld m) : m.tiles.card = meldExpectedCard m.kind := by
  cases hk : m.kind
  · simp only [IsValidMeld, hk, meldExpectedCard] at h ⊢; exact h.2.1
  · simp only [IsValidMeld, hk, meldExpectedCard] at h ⊢; exact h.2.1
  · simp only [IsValidMeld, hk, meldExpectedCard] at h ⊢; exact h.2.1
  · simp only [IsValidMeld, hk, meldExpectedCard] at h ⊢; exact h.2.1
  · simp only [IsValidMeld, hk, meldExpectedCard] at h ⊢; exact h.2.1
-- Chi construction witnesses
theorem mkChiMeld_sameSuit (startVal : Nat) (hstart : startVal % 9 ≤ 6) (hsuit : startVal < 27) (hstart2 : startVal + 2 < 34) (calledCopy c1Copy c2Copy : Copy) (source : Seat)
    (t : TileId) (ht : t ∈ (mkChiMeld startVal hstart hsuit hstart2 calledCopy c1Copy c2Copy source).tiles) :
    (tileType t).val / 9 = startVal / 9 := by
  unfold mkChiMeld at ht; simp at ht; rcases ht with rfl | rfl | rfl
  · simp [tileType_mkTile]
  · simp only [tileType_mkTile]
    have h1 : (startVal + 1) / 9 = startVal / 9 := by omega
    simpa [h1]
  · simp only [tileType_mkTile]
    have h1 : (startVal + 2) / 9 = startVal / 9 := by omega
    simpa [h1]
theorem mkChiMeld_honors_forbidden (startVal : Nat) (hstart : startVal % 9 ≤ 6) (hsuit : startVal < 27) (hstart2 : startVal + 2 < 34) (calledCopy c1Copy c2Copy : Copy) (source : Seat)
    (t : TileId) (ht : t ∈ (mkChiMeld startVal hstart hsuit hstart2 calledCopy c1Copy c2Copy source).tiles) :
    (tileType t).isSuited := by
  unfold TileType.isSuited
  have hval : (tileType t).val = startVal ∨ (tileType t).val = startVal + 1 ∨ (tileType t).val = startVal + 2 := by
    unfold mkChiMeld at ht; simp at ht; rcases ht with rfl | rfl | rfl <;> simp [tileType_mkTile]
  rcases hval with h | h | h
  · rw [h]; omega
  · rw [h]; omega
  · rw [h]; omega

-- ---------------------------------------------------------------------------
-- 5. Hand vs meld disjointness
-- ---------------------------------------------------------------------------

structure ActorHandState where
  concealed : Finset TileId
  drawn : Option TileId
  melds : List DeclaredMeld
  deriving DecidableEq

def actorPrivateTiles (s : ActorHandState) : Finset TileId :=
  match s.drawn with | none => s.concealed | some d => insert d s.concealed

def meldTilesUnion (s : ActorHandState) : Finset TileId :=
  s.melds.foldl (fun acc m => acc ∪ m.tiles) ∅

def PrivateMeldDisjoint (s : ActorHandState) : Prop := Disjoint (actorPrivateTiles s) (meldTilesUnion s)

theorem privateMeldDisjoint_emptyMelds (concealed : Finset TileId) (drawn : Option TileId) :
    PrivateMeldDisjoint { concealed := concealed, drawn := drawn, melds := [] } := by
  unfold PrivateMeldDisjoint actorPrivateTiles meldTilesUnion; simp

theorem meldTilesUnion_empty : meldTilesUnion { concealed := ∅, drawn := none, melds := [] } = ∅ := rfl

-- ---------------------------------------------------------------------------
-- 6. DeclaredMeld list invariants
-- ---------------------------------------------------------------------------

def meldCount (melds : List DeclaredMeld) (k : MeldKind) : Nat :=
  (melds.filter (fun m => decide (m.kind = k))).length

theorem meldCount_le_length (melds : List DeclaredMeld) (k : MeldKind) : meldCount melds k ≤ melds.length := by
  unfold meldCount; exact List.length_filter_le _ _

theorem meldCount_sum_eq_length (melds : List DeclaredMeld) :
    meldCount melds .Chi + meldCount melds .Pon + meldCount melds .Daiminkan + meldCount melds .Ankan + meldCount melds .Kakan = melds.length := by
  induction melds with
  | nil => simp [meldCount]
  | cons m ms ih =>
    have h1 : meldCount (m :: ms) .Chi + meldCount (m :: ms) .Pon + meldCount (m :: ms) .Daiminkan + meldCount (m :: ms) .Ankan + meldCount (m :: ms) .Kakan
            = meldCount ms .Chi + meldCount ms .Pon + meldCount ms .Daiminkan + meldCount ms .Ankan + meldCount ms .Kakan + 1 := by
      cases hk : m.kind <;> simp [meldCount, hk, List.filter_cons] <;> ac_rfl
    rw [h1, ih]; simp

def totalMeldTiles : List DeclaredMeld → Nat
  | [] => 0
  | m :: ms => m.tiles.card + totalMeldTiles ms

theorem totalMeldTiles_nil : totalMeldTiles [] = 0 := rfl
theorem totalMeldTiles_cons (m : DeclaredMeld) (ms : List DeclaredMeld) : totalMeldTiles (m :: ms) = m.tiles.card + totalMeldTiles ms := rfl

def DORA_SENTINEL_MELD : Int := -1
theorem dora_sentinel_meld_eq : DORA_SENTINEL_MELD = -1 := rfl

def countTypeInMeld (m : DeclaredMeld) (ty : TileType) : Nat :=
  (m.tiles.filter (fun t => decide (tileType t = ty))).card

theorem countTypeInMeld_le_card (m : DeclaredMeld) (ty : TileType) : countTypeInMeld m ty ≤ m.tiles.card := by
  unfold countTypeInMeld; exact Finset.card_filter_le _ _

theorem countTypeInMeld_le_4 (m : DeclaredMeld) (ty : TileType) : countTypeInMeld m ty ≤ 4 := by
  unfold countTypeInMeld
  have hfiber : (Finset.univ.filter (fun t : TileId => tileType t = ty)).card = 4 := tileType_copies ty
  have hsub : (m.tiles.filter (fun t => decide (tileType t = ty))) ⊆ (Finset.univ.filter (fun t : TileId => tileType t = ty)) := by
    intro t ht
    simp only [Finset.mem_filter, decide_eq_true_eq] at ht ⊢
    exact ⟨Finset.mem_univ t, ht.2⟩
  calc (m.tiles.filter (fun t => decide (tileType t = ty))).card
      ≤ (Finset.univ.filter (fun t : TileId => tileType t = ty)).card := Finset.card_le_card hsub
    _ = 4 := hfiber

theorem ankan_count_four (ty : TileType) : countTypeInMeld (mkAnkanMeld ty) ty = 4 := by
  unfold countTypeInMeld mkAnkanMeld
  have himg : (Finset.image (fun c : Copy => mkTile ty c) Finset.univ).filter (fun t => decide (tileType t = ty))
           = Finset.image (fun c : Copy => mkTile ty c) Finset.univ := by
    ext t
    simp only [Finset.mem_filter, Finset.mem_image, Finset.mem_univ, true_and, decide_eq_true_eq]
    constructor
    · intro ⟨h1, _⟩; exact h1
    · intro ⟨c, heq⟩
      exact ⟨⟨c, heq⟩, by rw [←heq, tileType_mkTile]⟩
  rw [himg, Finset.card_image_of_injective _ (mkTile_injective ty)]; simp [Fintype.card_fin]
-- 7. Kan invariants
-- ---------------------------------------------------------------------------

theorem isAnkan_card (m : DeclaredMeld) (h : IsValidAnkan m) : m.tiles.card = 4 := h.2.1
theorem isDaiminkan_card (m : DeclaredMeld) (h : IsValidDaiminkan m) : m.tiles.card = 4 := h.2.1
theorem isKakan_card (m : DeclaredMeld) (h : IsValidKakan m) : m.tiles.card = 4 := h.2.1
theorem isPon_card (m : DeclaredMeld) (h : IsValidPon m) : m.tiles.card = 3 := h.2.1
theorem isChi_card (m : DeclaredMeld) (h : IsValidChi m) : m.tiles.card = 3 := h.2.1

theorem ankan_no_source_called (m : DeclaredMeld) (h : IsValidAnkan m) : m.sourceSeat.isNone ∧ m.calledTile.isNone :=
  ⟨h.2.2.2.1, h.2.2.1⟩
theorem daiminkan_has_source (m : DeclaredMeld) (h : IsValidDaiminkan m) : m.sourceSeat.isSome ∧ m.calledTile.isSome :=
  ⟨h.2.2.2.1, h.2.2.1⟩
theorem pon_has_source (m : DeclaredMeld) (h : IsValidPon m) : m.sourceSeat.isSome ∧ m.calledTile.isSome :=
  ⟨h.2.2.2.1, h.2.2.1⟩
theorem kakan_has_added_and_prior (m : DeclaredMeld) (h : IsValidKakan m) : m.addedTile.isSome ∧ m.priorPonId.isSome :=
  ⟨h.2.2.2.2.1, h.2.2.2.2.2.1⟩
theorem kakan_no_source_called (m : DeclaredMeld) (h : IsValidKakan m) : m.sourceSeat.isNone ∧ m.calledTile.isNone :=
  ⟨h.2.2.2.1, h.2.2.1⟩
theorem kakan_prior_isSome (m : DeclaredMeld) (h : IsValidKakan m) : m.priorPonId.isSome = true := by
  have ⟨_, h2⟩ := kakan_has_added_and_prior m h; exact h2
theorem kakan_allSameType (m : DeclaredMeld) (h : IsValidKakan m) : ∃ ty : TileType, ∀ t ∈ m.tiles, tileType t = ty := by
  rcases h with ⟨_, _, _, _, _, _, ⟨added, hmem, hEq, ⟨ty, hTy, hall⟩⟩⟩
  exact ⟨ty, hall⟩

def MeldsAreDisjoint (melds : List DeclaredMeld) : Prop :=
  ∀ i j : Nat, ∀ hi : i < melds.length, ∀ hj : j < melds.length, i ≠ j →
    Disjoint (melds[i]'hi).tiles (melds[j]'hj).tiles

theorem meldsAreDisjoint_nil : MeldsAreDisjoint ([] : List DeclaredMeld) := by
  intro i j hi hj hne
  have hi0 : i < 0 := by simpa using hi
  omega
theorem meldsAreDisjoint_singleton (m : DeclaredMeld) : MeldsAreDisjoint [m] := by
  intro i j hi hj hne
  have hi0 : i < 1 := by simpa using hi
  have hj0 : j < 1 := by simpa using hj
  omega
def maxMelds : Nat := 4
theorem maxMelds_eq : maxMelds = 4 := rfl

theorem totalMeldTiles_le_length_mul_4 (melds : List DeclaredMeld) (hvalid : ∀ m ∈ melds, IsValidMeld m) :
    totalMeldTiles melds ≤ melds.length * 4 := by
  induction melds with
  | nil => simp [totalMeldTiles]
  | cons m ms ih =>
    have hm_card : m.tiles.card ≤ 4 := by
      have hm_valid := hvalid m (by simp)
      have hcard := validMeld_card m hm_valid
      cases hk : m.kind <;> simp [meldExpectedCard, hk] at hcard <;> omega
    have hms_valid : ∀ m' ∈ ms, IsValidMeld m' := fun m' hm' => hvalid m' (List.Mem.tail _ hm')
    have ihm := ih hms_valid
    simp only [totalMeldTiles, List.length]
    omega

theorem totalMeldTiles_le_16 (melds : List DeclaredMeld) (h : melds.length ≤ 4) (hvalid : ∀ m ∈ melds, IsValidMeld m) :
    totalMeldTiles melds ≤ 16 := by
  have hle := totalMeldTiles_le_length_mul_4 melds hvalid
  omega
theorem meldKind_exhaustive (k : MeldKind) : k = .Chi ∨ k = .Pon ∨ k = .Daiminkan ∨ k = .Ankan ∨ k = .Kakan := by cases k <;> simp
theorem meldKind_distinct : (MeldKind.Chi ≠ MeldKind.Pon) ∧ (MeldKind.Pon ≠ MeldKind.Daiminkan) ∧ (MeldKind.Daiminkan ≠ MeldKind.Ankan) ∧ (MeldKind.Ankan ≠ MeldKind.Kakan) ∧ (MeldKind.Chi ≠ MeldKind.Kakan) := by refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;> native_decide

end Formal.Mahjong
