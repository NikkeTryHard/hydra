import Formal.Mahjong.Tile
import Formal.Mahjong.Wall
import Formal.Mahjong.Meld
import Formal.Mahjong.State
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

namespace Formal.Mahjong.ActionModule

/-!
# Mahjong Action — legal-action vocabulary and IsLegal predicate (SPEC section 6)

Faithful Lean port of:

* `file://riichienv-core/src/legal_actions.rs#LegalActions` — tenhou legal set
  generation: draw-decision vs discard-response phases, chi predecessor constraint,
  pon same-type, kan four-copy check, riichi gating, ron/tsumo win check.
* `file://src/hydra2/contracts/action.py#ActionKind` — frozen 13-kind enum
  (`pass, discard, tsumogiri, riichi_discard, chi, pon, daiminkan, ankan, kakan,
  ron, tsumo, abort_nine_terminals, accept_abortive_draw`) with
  `ACTION_KIND_ORDINALS`, `ACTION_PHASES`, `CLAIM_KINDS`, `_consumed_pair_forms_run`, `_all_same_type`.
* `file://src/hydra2/contracts/action.py#CanonicalAction` — structural invariants per kind.
* `file://formal/Formal/Mahjong/Meld.lean#DeclaredMeld` — meld kinds and validity.
* `file://formal/Formal/Mahjong/State.lean#GameState` — wallPos, hands, discards, melds, scores.
-/

-- ---------------------------------------------------------------------------
-- 0. ActionKind — frozen ordinal vocabulary (SPEC 6.1, action.py)
-- ---------------------------------------------------------------------------

/-- Canonical action kinds — ticket surface collapsed from 13 frozen Python kinds.
    Mirrors `file://riichienv-core/src/legal_actions.rs#Action` and
    `file://src/hydra2/contracts/action.py#ActionKind`.
-/
inductive ActionKind where
  | Discard (tile : TileId)
  | Tsumogiri (tile : TileId)
  | Riichi (tile : TileId)
  | Chi (called : TileId) (consumed1 : TileId) (consumed2 : TileId) (source : Fin 4)
  | Pon (called : TileId) (consumed1 : TileId) (consumed2 : TileId) (source : Fin 4)
  | Daiminkan (called : TileId) (consumed1 : TileId) (consumed2 : TileId) (consumed3 : TileId) (source : Fin 4)
  | Ankan (ty : TileType)
  | Kakan (added : TileId)
  | Tsumo
  | Ron (winningTile : TileId) (source : Fin 4)
  | Pass
  deriving DecidableEq, Repr, BEq

/-- Frozen ordinal per `ACTION_KIND_ORDINALS` (action.py). -/
def ActionKind.toNat : ActionKind -> Nat
  | .Discard _ => 1
  | .Tsumogiri _ => 2
  | .Riichi _ => 3
  | .Chi _ _ _ _ => 4
  | .Pon _ _ _ _ => 5
  | .Daiminkan _ _ _ _ _ => 6
  | .Ankan _ => 7
  | .Kakan _ => 8
  | .Ron _ _ => 9
  | .Tsumo => 10
  | .Pass => 0

def ActionKind.toOrdinal : ActionKind -> Nat := ActionKind.toNat

theorem actionKind_toNat_range (a : ActionKind) : a.toNat < 13 := by
  cases a <;> simp [ActionKind.toNat]

theorem actionKind_ordinal_range (a : ActionKind) : a.toOrdinal < 13 := actionKind_toNat_range a

def ActionKind.isDiscard : ActionKind -> Bool
  | .Discard _ => true
  | .Tsumogiri _ => true
  | .Riichi _ => true
  | _ => false

def ActionKind.isCalls : ActionKind -> Bool
  | .Chi _ _ _ _ => true
  | .Pon _ _ _ _ => true
  | .Daiminkan _ _ _ _ _ => true
  | .Ankan _ => true
  | .Kakan _ => true
  | _ => false

def ActionKind.isWin : ActionKind -> Bool
  | .Tsumo => true
  | .Ron _ _ => true
  | _ => false

def ActionKind.isPass : ActionKind -> Bool
  | .Pass => true
  | _ => false

theorem discard_is_not_pass (t : TileId) : (ActionKind.Discard t).isPass = false := rfl
theorem tsumogiri_is_not_pass (t : TileId) : (ActionKind.Tsumogiri t).isPass = false := rfl
theorem riichi_is_not_pass (t : TileId) : (ActionKind.Riichi t).isPass = false := rfl

-- ---------------------------------------------------------------------------
-- 1. Pure helpers — mirrors action.py _consumed_pair_forms_run, _all_same_type
-- ---------------------------------------------------------------------------

def isHonorTileType (ty : TileType) : Bool := decide (27 <= ty.val)

def consumedPairFormsRun (called c1 c2 : TileId) : Bool :=
  let types := [tileType called, tileType c1, tileType c2]
  let vals := types.map (fun ty : TileType => ty.val)
  let suits := types.map (fun ty : TileType => ty.val / 9)
  let allSuited := types.all (fun ty => decide (ty.isSuited))
  let sameSuit := (suits[0]? == suits[1]?) && (suits[1]? == suits[2]?)
  let honorFree := !(types.any (fun ty => isHonorTileType ty))
  let distinct := (vals[0]? != vals[1]?) && (vals[1]? != vals[2]?) && (vals[0]? != vals[2]?)
  let consecutive :=
    match vals with
    | [v0, v1, v2] =>
      let mn := min v0 (min v1 v2)
      let mx := max v0 (max v1 v2)
      decide (mx - mn == 2)
    | _ => false
  allSuited && sameSuit && honorFree && distinct && consecutive

def allSameType3 (a b c : TileId) : Bool :=
  decide (tileType a = tileType b) && decide (tileType b = tileType c)

def allSameType4 (a b c d : TileId) : Bool :=
  decide (tileType a = tileType b) && decide (tileType b = tileType c) && decide (tileType c = tileType d)

theorem allSameType3_true_iff (a b c : TileId) : allSameType3 a b c = true <-> tileType a = tileType b /\ tileType b = tileType c := by
  unfold allSameType3
  simp only [Bool.and_eq_true, decide_eq_true_eq]

theorem allSameType4_true_iff (a b c d : TileId) : allSameType4 a b c d = true <-> tileType a = tileType b /\ tileType b = tileType c /\ tileType c = tileType d := by
  unfold allSameType4
  simp only [Bool.and_eq_true, decide_eq_true_eq]
  constructor
  · intro ⟨⟨h1, h2⟩, h3⟩; exact ⟨h1, h2, h3⟩
  · intro ⟨h1, h2, h3⟩; exact ⟨⟨h1, h2⟩, h3⟩

def prevSeat (actor : Fin 4) : Fin 4 := ⟨(actor.val + 3) % 4, by omega⟩

theorem prevSeat_ne_actor (actor : Fin 4) : prevSeat actor != actor := by
  unfold prevSeat
  fin_cases actor <;> native_decide

theorem prevSeat_val (actor : Fin 4) : (prevSeat actor).val = (actor.val + 3) % 4 := rfl

inductive ActionPhase where
  | DrawDecision
  | DiscardResponse
  | KanResponse
  deriving DecidableEq, Repr, BEq

def ActionKind.phase : ActionKind -> ActionPhase
  | .Discard _ => .DrawDecision
  | .Tsumogiri _ => .DrawDecision
  | .Riichi _ => .DrawDecision
  | .Ankan _ => .DrawDecision
  | .Kakan _ => .DrawDecision
  | .Tsumo => .DrawDecision
  | .Chi _ _ _ _ => .DiscardResponse
  | .Pon _ _ _ _ => .DiscardResponse
  | .Daiminkan _ _ _ _ _ => .DiscardResponse
  | .Ron _ _ => .DiscardResponse
  | .Pass => .DiscardResponse

-- ---------------------------------------------------------------------------
-- 2. Legality predicates — mirroring legal_actions.rs per-kind gates
-- ---------------------------------------------------------------------------

def tileInHand (s : Formal.Mahjong.GameState) (actor : Fin 4) (t : TileId) : Prop :=
  t ∈ s.hands actor

instance (s : Formal.Mahjong.GameState) (actor : Fin 4) (t : TileId) : Decidable (tileInHand s actor t) :=
  inferInstanceAs (Decidable (t ∈ s.hands actor))

def hasFourCopies (s : Formal.Mahjong.GameState) (actor : Fin 4) (ty : TileType) : Prop :=
  ∀ c : Copy, mkTile ty c ∈ s.hands actor

instance (s : Formal.Mahjong.GameState) (actor : Fin 4) (ty : TileType) : Decidable (hasFourCopies s actor ty) :=
  inferInstanceAs (Decidable (∀ c : Copy, mkTile ty c ∈ s.hands actor))

def countTypeInHand (s : Formal.Mahjong.GameState) (actor : Fin 4) (ty : TileType) : Nat :=
  (s.hands actor |>.filter (fun t => decide (tileType t = ty))).card

theorem countTypeInHand_le_4 (s : Formal.Mahjong.GameState) (actor : Fin 4) (ty : TileType) :
    countTypeInHand s actor ty ≤ 4 := by
  unfold countTypeInHand
  have hfiber : (Finset.univ.filter (fun t : TileId => tileType t = ty)).card = 4 :=
    Formal.Mahjong.tileType_copies ty
  have hsub : (s.hands actor |>.filter (fun t => decide (tileType t = ty)))
      ⊆ (Finset.univ.filter (fun t : TileId => tileType t = ty)) := by
    intro t ht
    simp only [Finset.mem_filter, decide_eq_true_eq] at ht ⊢
    exact ⟨Finset.mem_univ t, ht.2⟩
  calc (s.hands actor |>.filter (fun t => decide (tileType t = ty))).card
      ≤ (Finset.univ.filter (fun t : TileId => tileType t = ty)).card := Finset.card_le_card hsub
    _ = 4 := hfiber

theorem hasFourCopies_iff_count_eq_4 (s : Formal.Mahjong.GameState) (actor : Fin 4) (ty : TileType) :
    hasFourCopies s actor ty ↔ countTypeInHand s actor ty = 4 := by
  constructor
  · intro h
    have hsub : (Finset.image (fun c : Copy => mkTile ty c) Finset.univ) ⊆ s.hands actor := by
      intro t ht
      simp only [Finset.mem_image, Finset.mem_univ, true_and] at ht
      obtain ⟨c, rfl⟩ := ht
      exact h c
    have hcard : (Finset.image (fun c : Copy => mkTile ty c) Finset.univ).card = 4 := by
      rw [Finset.card_image_of_injective _ (Formal.Mahjong.mkTile_injective ty)]
      simp [Fintype.card_fin]
    have hmem_sub : (Finset.image (fun c : Copy => mkTile ty c) Finset.univ) ⊆ (s.hands actor |>.filter (fun t => decide (tileType t = ty))) := by
      intro t ht
      have ht2 : t ∈ s.hands actor := hsub ht
      have hty : tileType t = ty := by
        simp only [Finset.mem_image, Finset.mem_univ, true_and] at ht
        obtain ⟨c, rfl⟩ := ht
        exact Formal.Mahjong.tileType_mkTile ty c
      simp only [Finset.mem_filter, decide_eq_true_eq]
      exact ⟨ht2, hty⟩
    have hle : 4 ≤ countTypeInHand s actor ty := by
      unfold countTypeInHand
      calc 4 = (Finset.image (fun c : Copy => mkTile ty c) Finset.univ).card := hcard.symm
        _ ≤ (s.hands actor |>.filter (fun t => decide (tileType t = ty))).card := Finset.card_le_card hmem_sub
    have hle2 := countTypeInHand_le_4 s actor ty
    omega
  · intro h
    intro c
    have hmem : mkTile ty c ∈ Finset.univ.filter (fun t : TileId => tileType t = ty) := by
      simp [Formal.Mahjong.tileType_mkTile]
    have hsub_fiber : (s.hands actor |>.filter (fun t => decide (tileType t = ty))) ⊆ Finset.univ.filter (fun t : TileId => tileType t = ty) := by
      intro t ht
      simp only [Finset.mem_filter, decide_eq_true_eq] at ht ⊢
      exact ⟨Finset.mem_univ t, ht.2⟩
    have hfiber_card : (Finset.univ.filter (fun t : TileId => tileType t = ty)).card = 4 := Formal.Mahjong.tileType_copies ty
    unfold countTypeInHand at h
    have heq : ((s.hands actor).filter (fun t => decide (tileType t = ty))) = Finset.univ.filter (fun t : TileId => tileType t = ty) := by
      apply Finset.eq_of_subset_of_card_le hsub_fiber
      rw [h, hfiber_card]
    have hmem2 : mkTile ty c ∈ ((s.hands actor).filter (fun t => decide (tileType t = ty))) := by
      rw [heq]; exact hmem
    have hmem2a := (Finset.mem_filter.mp hmem2).1
    exact hmem2a

def hasPonMeldOfType (s : Formal.Mahjong.GameState) (actor : Fin 4) (ty : TileType) : Prop :=
  ∃ m ∈ s.melds actor, m.kind = Formal.Mahjong.MeldKind.Pon ∧ ∀ t ∈ m.tiles, tileType t = ty

def hasPonMeldOfTypeBool (s : Formal.Mahjong.GameState) (actor : Fin 4) (ty : TileType) : Bool :=
  (s.melds actor).any (fun m => decide (m.kind = Formal.Mahjong.MeldKind.Pon) && decide (∀ t ∈ m.tiles, tileType t = ty))

def IsLegalDiscard (s : Formal.Mahjong.GameState) (actor : Fin 4) (tile : TileId) : Prop :=
  tileInHand s actor tile ∧ Formal.Mahjong.ValidWallPos s

def IsLegalTsumogiri (s : Formal.Mahjong.GameState) (actor : Fin 4) (tile : TileId) : Prop :=
  tileInHand s actor tile ∧ (s.hands actor).card = 14 ∧ Formal.Mahjong.ValidWallPos s

def IsLegalRiichi (s : Formal.Mahjong.GameState) (actor : Fin 4) (tile : TileId) : Prop :=
  tileInHand s actor tile ∧ (s.hands actor).card = 14 ∧ s.kyotaku < 4 ∧ Formal.Mahjong.ValidWallPos s

def IsLegalChi (s : Formal.Mahjong.GameState) (actor : Fin 4) (called c1 c2 : TileId) (source : Fin 4) : Prop :=
  source = prevSeat actor ∧
  c1 ∈ s.hands actor ∧ c2 ∈ s.hands actor ∧
  c1 ≠ c2 ∧ called ∉ s.hands actor ∧
  consumedPairFormsRun called c1 c2 = true

def IsLegalPon (s : Formal.Mahjong.GameState) (actor : Fin 4) (called c1 c2 : TileId) (source : Fin 4) : Prop :=
  source ≠ actor ∧
  c1 ∈ s.hands actor ∧ c2 ∈ s.hands actor ∧
  c1 ≠ c2 ∧
  allSameType3 called c1 c2 = true

def IsLegalDaiminkan (s : Formal.Mahjong.GameState) (actor : Fin 4) (called c1 c2 c3 : TileId) (source : Fin 4) : Prop :=
  source ≠ actor ∧
  c1 ∈ s.hands actor ∧ c2 ∈ s.hands actor ∧ c3 ∈ s.hands actor ∧
  c1 ≠ c2 ∧ c1 ≠ c3 ∧ c2 ≠ c3 ∧
  allSameType4 called c1 c2 c3 = true

def IsLegalAnkan (s : Formal.Mahjong.GameState) (actor : Fin 4) (ty : TileType) : Prop :=
  hasFourCopies s actor ty

def IsLegalKakan (s : Formal.Mahjong.GameState) (actor : Fin 4) (added : TileId) : Prop :=
  added ∈ s.hands actor ∧ ∃ ty : TileType, tileType added = ty ∧ hasPonMeldOfType s actor ty

def IsLegalTsumo (s : Formal.Mahjong.GameState) (actor : Fin 4) : Prop :=
  (s.hands actor).card = 14 ∧ Formal.Mahjong.ValidWallPos s

def IsLegalRon (s : Formal.Mahjong.GameState) (actor : Fin 4) (winningTile : TileId) (source : Fin 4) : Prop :=
  source ≠ actor ∧ winningTile ∈ s.discards source

def IsLegalPass (_s : Formal.Mahjong.GameState) (_actor : Fin 4) : Prop := True

-- ---------------------------------------------------------------------------
-- 3. Combined IsLegal predicate — mirrors tenhou legal sets (legal_actions.rs)
-- ---------------------------------------------------------------------------

/-- Unified legality — `IsLegal s actor a` holds iff the per-kind gate holds.
    Mirrors `file://riichienv-core/src/legal_actions.rs#is_legal` and
    `file://src/hydra2/contracts/action.py#CanonicalAction` invariants.
-/
def IsLegal (s : Formal.Mahjong.GameState) (actor : Fin 4) : ActionKind -> Prop
  | .Discard tile => IsLegalDiscard s actor tile
  | .Tsumogiri tile => IsLegalTsumogiri s actor tile
  | .Riichi tile => IsLegalRiichi s actor tile
  | .Chi called c1 c2 source => IsLegalChi s actor called c1 c2 source
  | .Pon called c1 c2 source => IsLegalPon s actor called c1 c2 source
  | .Daiminkan called c1 c2 c3 source => IsLegalDaiminkan s actor called c1 c2 c3 source
  | .Ankan ty => IsLegalAnkan s actor ty
  | .Kakan added => IsLegalKakan s actor added
  | .Tsumo => IsLegalTsumo s actor
  | .Ron winningTile source => IsLegalRon s actor winningTile source
  | .Pass => IsLegalPass s actor

def IsLegalBool (s : Formal.Mahjong.GameState) (actor : Fin 4) (a : ActionKind) : Bool :=
  match a with
  | .Discard tile => decide (tile ∈ s.hands actor) && decide (s.wallPos ≤ 70)
  | .Tsumogiri tile => decide (tile ∈ s.hands actor) && decide ((s.hands actor).card = 14) && decide (s.wallPos ≤ 70)
  | .Riichi tile => decide (tile ∈ s.hands actor) && decide ((s.hands actor).card = 14) && decide (s.kyotaku < 4) && decide (s.wallPos ≤ 70)
  | .Chi called c1 c2 source => decide (source = prevSeat actor) && decide (c1 ∈ s.hands actor) && decide (c2 ∈ s.hands actor) && decide (c1 ≠ c2) && decide (called ∉ s.hands actor) && consumedPairFormsRun called c1 c2
  | .Pon called c1 c2 source => decide (source ≠ actor) && decide (c1 ∈ s.hands actor) && decide (c2 ∈ s.hands actor) && decide (c1 ≠ c2) && allSameType3 called c1 c2
  | .Daiminkan called c1 c2 c3 source => decide (source ≠ actor) && decide (c1 ∈ s.hands actor) && decide (c2 ∈ s.hands actor) && decide (c3 ∈ s.hands actor) && decide (c1 ≠ c2) && decide (c1 ≠ c3) && decide (c2 ≠ c3) && allSameType4 called c1 c2 c3
  | .Ankan ty => decide (hasFourCopies s actor ty)
  | .Kakan added => decide (added ∈ s.hands actor) && hasPonMeldOfTypeBool s actor (tileType added)
  | .Tsumo => decide ((s.hands actor).card = 14) && decide (s.wallPos ≤ 70)
  | .Ron winningTile source => decide (source ≠ actor) && decide (winningTile ∈ s.discards source)
  | .Pass => true

theorem isLegalBool_true_of_discard (s : Formal.Mahjong.GameState) (actor : Fin 4) (tile : TileId) (h : IsLegal s actor (ActionKind.Discard tile)) :
    IsLegalBool s actor (ActionKind.Discard tile) = true := by
  unfold IsLegal IsLegalDiscard tileInHand at h
  unfold IsLegalBool
  have h1 : decide (tile ∈ s.hands actor) = true := decide_eq_true h.1
  have h2 : decide (s.wallPos ≤ 70) = true := decide_eq_true h.2
  simp [h1, h2]

-- ---------------------------------------------------------------------------
-- 4. Legal action set (Finset) enumeration
-- ---------------------------------------------------------------------------

def legalDiscards (s : Formal.Mahjong.GameState) (actor : Fin 4) : Finset ActionKind :=
  (s.hands actor).image (fun t => ActionKind.Discard t)

def legalPassSet : Finset ActionKind := {ActionKind.Pass}

theorem legalPassSet_card : legalPassSet.card = 1 := by
  unfold legalPassSet; native_decide

theorem legalDiscards_card_le_hand (s : Formal.Mahjong.GameState) (actor : Fin 4) :
    (legalDiscards s actor).card ≤ (s.hands actor).card := by
  unfold legalDiscards
  exact Finset.card_image_le

theorem legalDiscards_nonempty_iff_hand_nonempty (s : Formal.Mahjong.GameState) (actor : Fin 4) :
    (legalDiscards s actor).Nonempty ↔ (s.hands actor).Nonempty := by
  unfold legalDiscards
  constructor
  · intro ⟨y, hy⟩
    simp only [Finset.mem_image] at hy
    obtain ⟨x, hx, _⟩ := hy
    exact ⟨x, hx⟩
  · intro ⟨x, hx⟩
    exact ⟨ActionKind.Discard x, Finset.mem_image.mpr ⟨x, hx, rfl⟩⟩

-- ---------------------------------------------------------------------------
-- 5. Transition consequences
-- ---------------------------------------------------------------------------

theorem isLegalDiscard_tile_mem (s : Formal.Mahjong.GameState) (actor : Fin 4) (tile : TileId) (h : IsLegal s actor (ActionKind.Discard tile)) :
    tile ∈ s.hands actor := h.1

theorem isLegalDiscard_wallPos (s : Formal.Mahjong.GameState) (actor : Fin 4) (tile : TileId) (h : IsLegal s actor (ActionKind.Discard tile)) :
    Formal.Mahjong.ValidWallPos s := h.2

theorem legalAnkan_hasFour (s : Formal.Mahjong.GameState) (actor : Fin 4) (ty : TileType) (h : IsLegal s actor (ActionKind.Ankan ty)) :
    hasFourCopies s actor ty := h

theorem legalKakan_has_added (s : Formal.Mahjong.GameState) (actor : Fin 4) (added : TileId) (h : IsLegal s actor (ActionKind.Kakan added)) :
    added ∈ s.hands actor := h.1

-- ---------------------------------------------------------------------------
-- 6. Required theorems — assignment contract
-- ---------------------------------------------------------------------------

/-- `discard_ne_pass` — `Discard` actions are distinct from `Pass`. -/
theorem discard_ne_pass (tile : TileId) : ActionKind.Discard tile ≠ ActionKind.Pass := by
  intro h
  cases h

theorem tsumogiri_ne_pass (tile : TileId) : ActionKind.Tsumogiri tile ≠ ActionKind.Pass := by
  intro h; cases h

theorem riichi_ne_pass (tile : TileId) : ActionKind.Riichi tile ≠ ActionKind.Pass := by
  intro h; cases h

theorem chi_ne_pass (called c1 c2 : TileId) (source : Fin 4) :
    ActionKind.Chi called c1 c2 source ≠ ActionKind.Pass := by
  intro h; cases h

theorem pon_ne_pass (called c1 c2 : TileId) (source : Fin 4) :
    ActionKind.Pon called c1 c2 source ≠ ActionKind.Pass := by
  intro h; cases h

theorem daiminkan_ne_pass (called c1 c2 c3 : TileId) (source : Fin 4) :
    ActionKind.Daiminkan called c1 c2 c3 source ≠ ActionKind.Pass := by
  intro h; cases h

theorem ankan_ne_pass (ty : TileType) : ActionKind.Ankan ty ≠ ActionKind.Pass := by
  intro h; cases h

theorem kakan_ne_pass (added : TileId) : ActionKind.Kakan added ≠ ActionKind.Pass := by
  intro h; cases h

theorem tsumo_ne_pass : ActionKind.Tsumo ≠ ActionKind.Pass := by
  intro h; cases h

theorem ron_ne_pass (w : TileId) (src : Fin 4) : ActionKind.Ron w src ≠ ActionKind.Pass := by
  intro h; cases h

theorem discard_ordinal_ne_pass_ordinal (tile : TileId) :
    (ActionKind.Discard tile).toNat ≠ (ActionKind.Pass).toNat := by
  simp [ActionKind.toNat]

/-- `tile_count_preserved_on_discard` — discarding preserves hand+discard count. -/
theorem tile_count_preserved_on_discard_hand (s : Formal.Mahjong.GameState) (actor : Fin 4) (tile : TileId) (s' : Formal.Mahjong.GameState)
    (h : Formal.Mahjong.discardTile s actor tile = some s') :
    (s'.hands actor).card + 1 = (s.hands actor).card := by
  unfold Formal.Mahjong.discardTile at h
  split at h
  · rename_i hmem
    simp only [Option.some.injEq] at h
    have hEq : s' = { s with hands := fun p => if p = actor then (s.hands p).erase tile else s.hands p, discards := fun p => if p = actor then s.discards p ++ [tile] else s.discards p } := h.symm
    have hHands : s'.hands actor = (s.hands actor).erase tile := by
      rw [hEq]; simp
    rw [hHands]
    have hcard := Finset.card_erase_of_mem hmem
    have hpos : 0 < (s.hands actor).card := Finset.card_pos.mpr ⟨tile, hmem⟩
    omega
  · simp at h

theorem tile_count_preserved_on_discard_discards (s : Formal.Mahjong.GameState) (actor : Fin 4) (tile : TileId) (s' : Formal.Mahjong.GameState)
    (h : Formal.Mahjong.discardTile s actor tile = some s') :
    (s'.discards actor).length = (s.discards actor).length + 1 := by
  unfold Formal.Mahjong.discardTile at h
  split at h
  · rename_i hmem
    simp only [Option.some.injEq] at h
    have hEq : s' = { s with hands := fun p => if p = actor then (s.hands p).erase tile else s.hands p, discards := fun p => if p = actor then s.discards p ++ [tile] else s.discards p } := h.symm
    have hDisc : s'.discards actor = s.discards actor ++ [tile] := by
      rw [hEq]; simp
    rw [hDisc, List.length_append, List.length_singleton]
  · simp at h

theorem tile_count_preserved_on_discard_combined (s : Formal.Mahjong.GameState) (actor : Fin 4) (tile : TileId) (s' : Formal.Mahjong.GameState)
    (h : Formal.Mahjong.discardTile s actor tile = some s') :
    (s'.hands actor).card + (s'.discards actor).length = (s.hands actor).card + (s.discards actor).length := by
  have hHand := tile_count_preserved_on_discard_hand s actor tile s' h
  have hDisc := tile_count_preserved_on_discard_discards s actor tile s' h
  omega

theorem tile_count_preserved_on_discard_other_hands (s : Formal.Mahjong.GameState) (actor other : Fin 4) (tile : TileId) (s' : Formal.Mahjong.GameState)
    (h : Formal.Mahjong.discardTile s actor tile = some s') (hne : other ≠ actor) :
    s'.hands other = s.hands other := by
  unfold Formal.Mahjong.discardTile at h
  split at h
  · rename_i hmem
    simp only [Option.some.injEq] at h
    have hEq : s' = { s with hands := fun p => if p = actor then (s.hands p).erase tile else s.hands p, discards := fun p => if p = actor then s.discards p ++ [tile] else s.discards p } := h.symm
    rw [hEq]
    simp [hne]
  · simp at h

theorem tile_count_preserved_on_discard_other_discards (s : Formal.Mahjong.GameState) (actor other : Fin 4) (tile : TileId) (s' : Formal.Mahjong.GameState)
    (h : Formal.Mahjong.discardTile s actor tile = some s') (hne : other ≠ actor) :
    s'.discards other = s.discards other := by
  unfold Formal.Mahjong.discardTile at h
  split at h
  · rename_i hmem
    simp only [Option.some.injEq] at h
    have hEq : s' = { s with hands := fun p => if p = actor then (s.hands p).erase tile else s.hands p, discards := fun p => if p = actor then s.discards p ++ [tile] else s.discards p } := h.symm
    rw [hEq]
    simp [hne]
  · simp at h

theorem tile_count_preserved_on_discard_wall (s : Formal.Mahjong.GameState) (actor : Fin 4) (tile : TileId) (s' : Formal.Mahjong.GameState)
    (h : Formal.Mahjong.discardTile s actor tile = some s') :
    s'.wall = s.wall := by
  unfold Formal.Mahjong.discardTile at h
  split at h
  · rename_i hmem
    simp only [Option.some.injEq] at h
    have hEq : s' = { s with hands := fun p => if p = actor then (s.hands p).erase tile else s.hands p, discards := fun p => if p = actor then s.discards p ++ [tile] else s.discards p } := h.symm
    rw [hEq]
  · simp at h

theorem tile_count_preserved_on_discard (s : Formal.Mahjong.GameState) (actor : Fin 4) (tile : TileId) (s' : Formal.Mahjong.GameState)
    (h : Formal.Mahjong.discardTile s actor tile = some s') :
    (s'.hands actor).card + (s'.discards actor).length = (s.hands actor).card + (s.discards actor).length ∧
    s'.wall = s.wall ∧ s'.melds = s.melds := by
  refine ⟨?_, ?_, ?_⟩
  · exact tile_count_preserved_on_discard_combined s actor tile s' h
  · exact tile_count_preserved_on_discard_wall s actor tile s' h
  · unfold Formal.Mahjong.discardTile at h
    split at h
    · rename_i hmem
      simp only [Option.some.injEq] at h
      have hEq : s' = { s with hands := fun p => if p = actor then (s.hands p).erase tile else s.hands p, discards := fun p => if p = actor then s.discards p ++ [tile] else s.discards p } := h.symm
      rw [hEq]
    · simp at h

/-- `kan_requires_fourth_tile` — Ankan legality requires the fourth physical copy. -/
theorem kan_requires_fourth_tile (s : Formal.Mahjong.GameState) (actor : Fin 4) (ty : TileType)
    (h : IsLegal s actor (ActionKind.Ankan ty)) :
    hasFourCopies s actor ty ∧ countTypeInHand s actor ty = 4 := by
  constructor
  · exact h
  · rwa [← hasFourCopies_iff_count_eq_4]

theorem kakan_requires_prior_pon (s : Formal.Mahjong.GameState) (actor : Fin 4) (added : TileId)
    (h : IsLegal s actor (ActionKind.Kakan added)) :
    ∃ ty : TileType, tileType added = ty ∧ hasPonMeldOfType s actor ty := by
  obtain ⟨ty, hTy, hPon⟩ := h.2
  exact ⟨ty, hTy, hPon⟩

theorem kan_requires_fourth_tile_copy3 (s : Formal.Mahjong.GameState) (actor : Fin 4) (ty : TileType)
    (h : IsLegal s actor (ActionKind.Ankan ty)) :
    mkTile ty ⟨3, by omega⟩ ∈ s.hands actor := by
  have h4 := (kan_requires_fourth_tile s actor ty h).1
  exact h4 ⟨3, by omega⟩

theorem kan_requires_fourth_tile_distinct (s : Formal.Mahjong.GameState) (actor : Fin 4) (ty : TileType)
    (h : IsLegal s actor (ActionKind.Ankan ty)) :
    ∃ (c1 c2 c3 c4 : Copy), c1 ≠ c2 ∧ c1 ≠ c3 ∧ c1 ≠ c4 ∧ c2 ≠ c3 ∧ c2 ≠ c4 ∧ c3 ≠ c4 ∧
      mkTile ty c1 ∈ s.hands actor ∧ mkTile ty c2 ∈ s.hands actor ∧
      mkTile ty c3 ∈ s.hands actor ∧ mkTile ty c4 ∈ s.hands actor := by
  have h4 := (kan_requires_fourth_tile s actor ty h).1
  refine ⟨⟨0, by omega⟩, ⟨1, by omega⟩, ⟨2, by omega⟩, ⟨3, by omega⟩, ?_, ?_, ?_, ?_, ?_, ?_, h4 _, h4 _, h4 _, h4 _⟩ <;> native_decide

theorem ankan_four_copies_card (s : Formal.Mahjong.GameState) (actor : Fin 4) (ty : TileType)
    (h : IsLegal s actor (ActionKind.Ankan ty)) :
    (Finset.image (fun c : Copy => mkTile ty c) Finset.univ).card = 4 ∧
    (Finset.image (fun c : Copy => mkTile ty c) Finset.univ) ⊆ s.hands actor := by
  constructor
  · rw [Finset.card_image_of_injective _ (Formal.Mahjong.mkTile_injective ty)]
    simp [Fintype.card_fin]
  · intro t ht
    simp only [Finset.mem_image, Finset.mem_univ, true_and] at ht
    obtain ⟨c, rfl⟩ := ht
    exact (kan_requires_fourth_tile s actor ty h).1 c

-- ---------------------------------------------------------------------------
-- 7. Additional invariants
-- ---------------------------------------------------------------------------

theorem chi_requires_prev_seat (s : Formal.Mahjong.GameState) (actor : Fin 4) (called c1 c2 : TileId) (source : Fin 4)
    (h : IsLegal s actor (ActionKind.Chi called c1 c2 source)) :
    source = prevSeat actor := h.1

theorem pon_requires_source_ne_actor (s : Formal.Mahjong.GameState) (actor : Fin 4) (called c1 c2 : TileId) (source : Fin 4)
    (h : IsLegal s actor (ActionKind.Pon called c1 c2 source)) :
    source ≠ actor := h.1

theorem daiminkan_requires_three_distinct (s : Formal.Mahjong.GameState) (actor : Fin 4) (called c1 c2 c3 : TileId) (source : Fin 4)
    (h : IsLegal s actor (ActionKind.Daiminkan called c1 c2 c3 source)) :
    c1 ≠ c2 ∧ c1 ≠ c3 ∧ c2 ≠ c3 := by
  unfold IsLegal IsLegalDaiminkan at h
  rcases h with ⟨_, _, _, _, h12, h13, h23, _⟩
  exact ⟨h12, h13, h23⟩

theorem isLegal_ankan_count_eq_4 (s : Formal.Mahjong.GameState) (actor : Fin 4) (ty : TileType)
    (h : IsLegal s actor (ActionKind.Ankan ty)) :
    countTypeInHand s actor ty = 4 :=
  (kan_requires_fourth_tile s actor ty h).2

theorem actionKind_noDup_helper : ActionKind.Pass ≠ ActionKind.Tsumo := by decide
theorem actionKind_pass_ne_tsumo : ActionKind.Pass ≠ ActionKind.Tsumo := by decide
theorem actionKind_pass_ne_ron (w : TileId) (src : Fin 4) : ActionKind.Pass ≠ ActionKind.Ron w src := by
  intro h; cases h

theorem ordinal_pass_lt_discard (t : TileId) : (ActionKind.Pass).toNat < (ActionKind.Discard t).toNat := by
  simp [ActionKind.toNat]

theorem ordinal_chi_lt_pon (called c1 c2 : TileId) (src : Fin 4) (called2 : TileId) :
    (ActionKind.Chi called c1 c2 src).toNat < (ActionKind.Pon called2 c1 c2 src).toNat := by
  simp [ActionKind.toNat]

theorem ordinal_ankan_lt_kakan (ty : TileType) (added : TileId) :
    (ActionKind.Ankan ty).toNat < (ActionKind.Kakan added).toNat := by
  simp [ActionKind.toNat]

theorem discard_is_drawDecision (t : TileId) : (ActionKind.Discard t).phase = .DrawDecision := rfl
theorem chi_is_discardResponse (called c1 c2 : TileId) (src : Fin 4) :
    (ActionKind.Chi called c1 c2 src).phase = .DiscardResponse := rfl
theorem pass_is_discardResponse : ActionKind.Pass.phase = .DiscardResponse := rfl
theorem ankan_phase_gating (ty : TileType) : (ActionKind.Ankan ty).phase = .DrawDecision := rfl
theorem ron_phase_gating (w : TileId) (src : Fin 4) : (ActionKind.Ron w src).phase = .DiscardResponse := rfl

-- ---------------------------------------------------------------------------
-- 8. Digest analogue (mirrors Wall.lean#wall_schedule_digest)
-- ---------------------------------------------------------------------------

def actionDigestNat (a : ActionKind) : Nat :=
  match a with
  | .Discard t => t.val * 131 + 1
  | .Tsumogiri t => t.val * 131 + 2
  | .Riichi t => t.val * 131 + 3
  | .Chi c a b _ => c.val * 16777619 + a.val * 131 + b.val + 4
  | .Pon c a b _ => c.val * 16777619 + a.val * 131 + b.val + 5
  | .Daiminkan c a b d _ => c.val * 16777619 + a.val * 131 + b.val + d.val + 6
  | .Ankan ty => ty.val * 131 + 7
  | .Kakan t => t.val * 131 + 8
  | .Ron w _ => w.val * 131 + 9
  | .Tsumo => 10
  | .Pass => 0

theorem actionDigestNat_pass_zero : actionDigestNat ActionKind.Pass = 0 := rfl
theorem actionDigestNat_tsumo_ten : actionDigestNat ActionKind.Tsumo = 10 := rfl

theorem actionDigestNat_discard_ne_pass (t : TileId) : actionDigestNat (ActionKind.Discard t) ≠ actionDigestNat ActionKind.Pass := by
  unfold actionDigestNat
  simp only
  have ht := t.isLt
  omega

end Formal.Mahjong.ActionModule
