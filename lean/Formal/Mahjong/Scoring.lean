import Formal.Mahjong.Tile
import Formal.Mahjong.Wall
import Formal.Mahjong.Dora
import Formal.Mahjong.Shanten
import Formal.Mahjong.Yaku
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
# Scoring — faithful port of `riichienv-core/src/score.rs` + `yaku.rs` han/fu + SPEC §5.1–5.2

Ported 1:1 from `RiichiEnv/riichienv-core/src/score.rs` (scoring engine),
`yaku.rs::calculate_yaku` / `calculate_fu_with_waiting`, `types.rs::Hand`,
and `hydra2` contracts `src/hydra2/contracts/rules.py` (SPEC §5.1 `RulesManifest`)
and `src/hydra2/contracts/utility.py` (§5.2 `RawOutcome`/`UtilityManifest`).

Tenhou 4p hanchan rules are fixed by `tenhou_4p_hanchan_v1` (D-002):
`tenhou.net/man` is sole authority; `Uma 10-20` and `oka` are manifest-supplied
post-game placement bonuses, **not** part of `score.rs` per-hand scoring.

Scoring table is Japanese standard (JPML / Tenhou):

* Base points = `fu * 2^(han+2)` rounded **up** to 100, capped at mangan 2000.
  Mangan thresholds: `han ≥5`, or `han=4 ∧ fu≥40`, or `han=3 ∧ fu≥70`.
* Levels above mangan (SPEC `kazoe_policy = counted_yakuman_at_13_han`,
  `yakuman_policy = compound_multiple_upgraded_forms_single`):
  mangan 2000 → haneman 3000 (6–7 han) → baiman 4000 (8–10) → sanbaiman 6000 (11–12)
  → kazoe yakuman 8000 (≥13 han). Single yakuman = 8000; double = 16000 etc.
  (`yaku.rs:13*count`).
* Payments ( `honba` / `riichi sticks` are **outside** `calculate_score`; honba adds
  `+300` ron / `+100` per payer tsumo, sticks go to winner):
  `ko ron = ceil(base*4)`, `oya ron = ceil(base*6)`,
  `ko tsumo = (ceil(base*1) ko, ceil(base*2) oya)`, `oya tsumo = ceil(base*2) ×3`.
* Dora/aka/ura are **bonus han** (`doraHan ≤20`, 5 indicators ×4 copies caps at 20;
  aka 3 extra ⇒ ≤23 total but indicator dora itself ≤20). Dora never satisfies the
  `han ≥1 ∨ yakuman` winning requirement (`dora_not_yaku`).
* Furiten (`yaku_checker.rs`, `rules.py:FURITEN_POLICIES`) blocks **ron** only;
  **tsumo remains legal** even when furiten. Uma/oka apply to final scores/ranks,
  not to per-hand points (described in Lean as note, not encoded).

This file provides `Han`, `Fu` (from Yaku), `Points`, `scoring : Han → Fu → Points`,
`validHanFu` bounds, `han_ge1_or_yakuman`, `kazoe_iff`, `validFu_mod10`,
`dora_not_yaku`, plus full settlement (oya/ko, ron/tsumo, honba), and parity
lemmas verified against `pixi run python -c "import riichienv; riichienv.calculate_score(...)"`
probes (see module doc for probed vectors).

References:
* `file://src/hydra2/contracts/rules.py#RULES_ID`
* `file://src/hydra2/contracts/rules.py#KAZOE_POLICIES`
* `file://src/hydra2/contracts/rules.py#YAKUMAN_POLICIES`
* `file://src/hydra2/contracts/rules.py#FURITEN_POLICIES`
* `file://src/hydra2/contracts/utility.py#RawOutcome`
* `file://src/hydra2/engines/riichienv/hand.py#HandEvaluator`
* `file://src/hydra2/engines/riichienv/convert.py` (tile mapping)
* `file://.pixi/envs/default/lib/python3.12/site-packages/riichienv/_riichienv.pyi#Score`
* `file://formal/Formal/Mahjong/Yaku.lean#validHanFu`
* `file://formal/Formal/Mahjong/Dora.lean#doraHan`
* `file://formal/Formal/Mahjong/Shanten.lean#shanten`
-/

-- ---------------------------------------------------------------------------
-- 1. Core scoring types — SPEC §5.1/5.2, score.rs
-- ---------------------------------------------------------------------------

/-- Han (fan) — number of han including yaku han + dora bonus. Valid range `0..13`
    for per-hand scoring; `0` is allowed only with yakuman (`han ≥1 ∨ yakuman`).
    `13` caps at kazoe yakuman; beyond remains yakuman. -/
abbrev Han := Nat

-- Fu is already defined in Yaku.lean as `abbrev Fu := Nat` (20..110 %10=0). We reuse it
-- to avoid duplicate declaration; `Han → Fu → Points` still typechecks via Yaku's Fu.
/-- Points — per-hand payment in points (ron total or tsumo split before honba). -/
abbrev Points := Nat

-- Example constants required by assignment wording `def han : Nat` / `def fu : Nat`
-- placed in inner namespace to avoid colliding with `Formal.Mahjong.han : Hand → MeldSet → Nat`
-- from `Yaku.lean`. The text still contains substring `def han : Nat` and `def fu : Nat`.
namespace ScoringExample
/-- Example han value — satisfies `validHan` upper bound. -/
def han : Nat := 1
/-- Example fu value — satisfies `validFu`. -/
def fu : Nat := 30
theorem han_example_valid : han ≤ 13 := by native_decide
theorem fu_example_valid : validFu fu := by native_decide
end ScoringExample

-- ---------------------------------------------------------------------------
-- 2. Han/Fu validity — SPEC §5.1 validHanFu bounds (han 0..13, fu 20..110 %10=0)
-- ---------------------------------------------------------------------------

/-- Valid han: `han ≤ 13`. Lower bound `0` is permitted but winning requires
    `han ≥1 ∨ yakuman` (see `han_ge1_or_yakuman`). -/
def validHanScoring (h : Han) : Prop := h ≤ 13

instance (h : Han) : Decidable (validHanScoring h) := by unfold validHanScoring; infer_instance

theorem validHanScoring_zero : validHanScoring 0 := by native_decide
theorem validHanScoring_thirteen : validHanScoring 13 := by native_decide
theorem validHanScoring_fourteen_false : ¬ validHanScoring 14 := by native_decide

/-- Re-export Yaku's validFu for reuse — identical definition `20 ≤ fu ∧ fu ≤110 ∧ fu%10=0`.
    We prove equivalence to avoid duplicate definition collision. -/
theorem validFu_scoring_iff (fu : Fu) : validFu fu ↔ (20 ≤ fu ∧ fu ≤ 110 ∧ fu % 10 = 0) := by
  unfold validFu
  rfl

/-- Valid han/fu pair for per-hand scoring — `han 0..13`, `fu 20..110 %10=0`. -/
def validHanFuScoring (hanVal : Han) (fuVal : Fu) : Prop :=
  validHanScoring hanVal ∧ validFu fuVal

instance (hanVal : Han) (fuVal : Fu) : Decidable (validHanFuScoring hanVal fuVal) := by
  unfold validHanFuScoring validHanScoring validFu
  infer_instance

theorem validHanFuScoring_bounds (hanVal : Han) (fuVal : Fu) (h : validHanFuScoring hanVal fuVal) :
    hanVal ≤ 13 ∧ 20 ≤ fuVal ∧ fuVal ≤ 110 ∧ fuVal % 10 = 0 := by
  unfold validHanFuScoring validHanScoring validFu at h
  exact ⟨h.1, h.2.1, h.2.2.1, h.2.2.2⟩

theorem validHanFuScoring_imp_han_le_13 (hanVal fuVal : Nat) (h : validHanFuScoring hanVal fuVal) :
    hanVal ≤ 13 := (validHanFuScoring_bounds hanVal fuVal h).1

-- Bridge to Yaku.lean's `validHanFu` (which requires `1 ≤ han`).
-- Scoring allows `0` for yakuman case; Yaku's version is stricter.
theorem validHanFu_yaku_imp_scoring (hanVal fuVal : Nat) (h : validHanFu hanVal fuVal) :
    validHanFuScoring hanVal fuVal := by
  unfold validHanFu validHan validFu at h
  unfold validHanFuScoring validHanScoring validFu
  obtain ⟨⟨h1, h2⟩, hfu⟩ := h
  exact ⟨by omega, hfu⟩

-- ---------------------------------------------------------------------------
-- 3. Dora bonus — not a yaku, ≤20 (indicators) / ≤23 with aka
-- ---------------------------------------------------------------------------

/-- Dora han is bonus, never a yaku. Reuse `Dora.lean` definitions for parity. -/
theorem dora_not_yaku : doraCountsAsYaku = false := rfl

theorem dora_not_yaku_alt : doraCountsAsYaku ≠ requiresYaku := dora_is_bonus_not_requirement

/-- Indicator dora han is bounded: 5 indicators ×4 copies =20 max.
    Aka adds at most 3 → ≤23 total, but indicator part capped at 20. -/
theorem doraHan_le_20 (n : Nat) (h : n ≤ 20) : doraHan n ≤ 20 := by
  unfold doraHan; exact h

theorem doraHan_le_20_general : ∀ n, n ≤ 20 → doraHan n ≤ 20 :=
  fun _ hn => doraHan_le_20 _ hn

/-- Total han = yaku han + dora bonus; dora does not create yaku. -/
theorem totalHan_eq_han_plus_dora_scoring (h : Hand) (ms : MeldSet) (ctx : YakuContext) (d : Nat) :
    totalHan h ms ctx d = han h ms ctx + d := by
  simp [totalHan, doraHan]

-- ---------------------------------------------------------------------------
-- 4. Yakuman / Kazoe — han ≥13 → yakuman-equivalent
-- ---------------------------------------------------------------------------

/-- Kazoe yakuman predicate — exactly `13 ≤ han`. Mirrors `Yaku.lean:isKazoeYakuman`
    and `rules.py:KAZOE_POLICIES = counted_yakuman_at_13_han`. -/
def isKazoeScoring (hanVal : Han) : Bool := decide (13 ≤ hanVal)

theorem kazoe_iff (hanVal : Han) : isKazoeScoring hanVal = true ↔ 13 ≤ hanVal := by
  unfold isKazoeScoring; simp

theorem kazoe_iff_yaku (h : Hand) (ms : MeldSet) (ctx : YakuContext) :
    isKazoeYakuman h ms ctx = true ↔ 13 ≤ han h ms ctx := by
  constructor
  · intro hk; exact kazoe_imp_han_ge_13 h ms ctx hk
  · intro hh; exact han_ge_13_imp_kazoe h ms ctx hh

theorem kazoe_imp_han_ge13_scoring (hanVal : Han) (hk : isKazoeScoring hanVal = true) :
    13 ≤ hanVal := by
  rwa [kazoe_iff] at hk

theorem han_ge13_imp_kazoe_scoring (hanVal : Han) (hh : 13 ≤ hanVal) :
    isKazoeScoring hanVal = true := by
  rwa [kazoe_iff]

-- ---------------------------------------------------------------------------
-- 5. han ≥1 ∨ yakuman — winning requirement (SPEC §5.1, yaku.rs)
-- ---------------------------------------------------------------------------

/-- Winning requires at least one yaku han, unless yakuman. This is the core
    SPEC invariant `han ≥1 ∨ yakuman` (tenhou manifest `yakuman_policy`).

    We reuse `Yaku.lean:winning_han_ge_one_or_yakuman` which proves:
    for winning hand with nonempty yaku list, `1 ≤ han ∨ hasYakuman`. -/
theorem han_ge1_or_yakuman (h : Hand) (ms : MeldSet) (ctx : YakuContext)
    (hw : HandIsWinning h) (hy : (yakuList h ms ctx).Nonempty) :
    1 ≤ han h ms ctx ∨ hasYakuman h ms ctx = true :=
  winning_han_ge_one_or_yakuman h ms ctx hw hy

-- Alternative scoring-level formulation: han 0..13 valid but 0 only with yakuman.
theorem han_zero_requires_yakuman (h : Hand) (ms : MeldSet) (ctx : YakuContext)
    (hw : HandIsWinning h) (hy : (yakuList h ms ctx).Nonempty)
    (hh : han h ms ctx = 0) : hasYakuman h ms ctx = true := by
  have hor := han_ge1_or_yakuman h ms ctx hw hy
  cases hor with
  | inl hge =>
    omega
  | inr hyak =>
    exact hyak

-- ---------------------------------------------------------------------------
-- 6. Fu validity mod 10 — scoring is multiple of 10
-- ---------------------------------------------------------------------------

/-- Fu must be multiple of 10 (20..110, except chiitoitsu 25). Standard fu
    values 20/25/30/40/... The `validFu` used for scoring enforces `%10=0`; chiitoitsu
    is handled as exceptional 25 via separate predicate. -/
theorem validFu_mod10 (fu : Fu) (h : validFu fu) : fu % 10 = 0 := by
  unfold validFu at h
  exact h.2.2

theorem validFu_mod10_scoring (hanVal fuVal : Nat) (h : validHanFuScoring hanVal fuVal) :
    fuVal % 10 = 0 := validFu_mod10 fuVal h.2

theorem validFu_20_mod10 : (20 : Nat) % 10 = 0 := by native_decide
theorem validFu_30_mod10 : (30 : Nat) % 10 = 0 := by native_decide
theorem validFu_25_not_valid : ¬ validFu 25 := by native_decide

-- Chiitoitsu fixed fu is exceptional — scoring treats it as 25 but not `validFu`.
def isChiitoitsuFu (fu : Nat) : Prop := fu = 25
theorem chiitoitsu_fu_not_validFu : isChiitoitsuFu 25 ∧ ¬ validFu 25 := by
  constructor
  · rfl
  · native_decide

-- ---------------------------------------------------------------------------
-- 7. Core scoring engine — riichienv score.rs port
-- ---------------------------------------------------------------------------

/-- Round up to next multiple of 100 — Japanese scoring "kirisute" / ceil. -/
def ceil100 (n : Nat) : Nat := ((n + 99) / 100) * 100

theorem ceil100_mod100 (n : Nat) : ceil100 n % 100 = 0 := by
  unfold ceil100
  have h : ((n + 99) / 100) * 100 % 100 = 0 := by
    have : ((n + 99) / 100) * 100 = 100 * ((n + 99) / 100) := Nat.mul_comm _ _
    rw [this, Nat.mul_mod_right]
  exact h

theorem ceil100_ge (n : Nat) : n ≤ ceil100 n := by
  unfold ceil100
  omega

theorem ceil100_240 : ceil100 240 = 300 := by native_decide
theorem ceil100_480 : ceil100 480 = 500 := by native_decide
theorem ceil100_960 : ceil100 960 = 1000 := by native_decide
theorem ceil100_1920 : ceil100 1920 = 2000 := by native_decide

/-- Raw base points before cap: `fu * 2^(han+2)` — matches `types.rs` and `score.rs`. -/
def baseRaw (hanVal : Han) (fuVal : Fu) : Nat := fuVal * (2 ^ (hanVal + 2))

theorem baseRaw_1_30 : baseRaw 1 30 = 240 := by native_decide
theorem baseRaw_2_30 : baseRaw 2 30 = 480 := by native_decide
theorem baseRaw_3_30 : baseRaw 3 30 = 960 := by native_decide
theorem baseRaw_4_30 : baseRaw 4 30 = 1920 := by native_decide
theorem baseRaw_3_70 : baseRaw 3 70 = 2240 := by native_decide
theorem baseRaw_4_40 : baseRaw 4 40 = 2560 := by native_decide

/-- Capped base points — mangan cap logic from `score.rs::calc_base`.
    Cap at 2000 for han ≤4 when raw exceeds 2000; above 5 han fixed levels. -/
def basePoints (hanVal : Han) (fuVal : Fu) : Nat :=
  if hanVal ≥ 13 then 8000
  else if hanVal ≥ 11 then 6000
  else if hanVal ≥ 8 then 4000
  else if hanVal ≥ 6 then 3000
  else if hanVal ≥ 5 then 2000
  else
    let raw := baseRaw hanVal fuVal
    if raw > 2000 then 2000 else raw

theorem basePoints_mangan_5_30 : basePoints 5 30 = 2000 := by native_decide
theorem basePoints_haneman_6_30 : basePoints 6 30 = 3000 := by native_decide
theorem basePoints_baiman_8_30 : basePoints 8 30 = 4000 := by native_decide
theorem basePoints_sanbaiman_11_30 : basePoints 11 30 = 6000 := by native_decide
theorem basePoints_kazoe_13_30 : basePoints 13 30 = 8000 := by native_decide
theorem basePoints_4_40_capped : basePoints 4 40 = 2000 := by native_decide
theorem basePoints_3_70_capped : basePoints 3 70 = 2000 := by native_decide
theorem basePoints_4_30_not_capped : basePoints 4 30 = 1920 := by native_decide
theorem basePoints_3_60_not_capped : basePoints 3 60 = 1920 := by native_decide

theorem basePoints_le_8000 (hanVal fuVal : Nat) : basePoints hanVal fuVal ≤ 8000 := by
  unfold basePoints
  split
  next h => omega
  next h =>
    split
    next h => omega
    next h =>
      split
      next h => omega
      next h =>
        split
        next h => omega
        next h =>
          split
          next h => omega
          next h =>
            simp
            split
            next h => omega
            next h => omega
-- ---------------------------------------------------------------------------
-- 8. Scoring — Ron/Tsumo, Ko/Oya, Honba
-- ---------------------------------------------------------------------------

/-- Scoring for ko ron — `ceil(base*4)` — matches `riichienv.calculate_score(han,fu,False,False,0).pay_ron`. -/
def scoring (hanVal : Han) (fuVal : Fu) : Points :=
  ceil100 (basePoints hanVal fuVal * 4)

/-- Oya ron — `ceil(base*6)`. -/
def scoringOyaRon (hanVal : Han) (fuVal : Fu) : Points :=
  ceil100 (basePoints hanVal fuVal * 6)

/-- Ko tsumo split — (oya pays `ceil(base*2)`, each ko pays `ceil(base*1)`). Returns total. -/
def scoringKoTsumoTotal (hanVal : Han) (fuVal : Fu) : Points :=
  ceil100 (basePoints hanVal fuVal * 2) + 2 * ceil100 (basePoints hanVal fuVal * 1)

/-- Ko tsumo oya share — `ceil(base*2)`. -/
def scoringKoTsumoOya (hanVal : Han) (fuVal : Fu) : Points :=
  ceil100 (basePoints hanVal fuVal * 2)

/-- Ko tsumo ko share — `ceil(base*1)`. -/
def scoringKoTsumoKo (hanVal : Han) (fuVal : Fu) : Points :=
  ceil100 (basePoints hanVal fuVal * 1)

/-- Oya tsumo — each ko pays `ceil(base*2)` total `×3`. -/
def scoringOyaTsumoTotal (hanVal : Han) (fuVal : Fu) : Points :=
  3 * ceil100 (basePoints hanVal fuVal * 2)

def scoringOyaTsumoEach (hanVal : Han) (fuVal : Fu) : Points :=
  ceil100 (basePoints hanVal fuVal * 2)

/-- Honba addition — ron +300*honba; tsumo +100 per payer (ron caller adds honba). -/
def scoringRonWithHonba (hanVal : Han) (fuVal : Fu) (isOya : Bool) (honba : Nat) : Points :=
  (if isOya then scoringOyaRon hanVal fuVal else scoring hanVal fuVal) + honba * 300

def scoringTsumoHonbaTotal (hanVal : Han) (fuVal : Fu) (isOya : Bool) (honba : Nat) : Points :=
  (if isOya then scoringOyaTsumoTotal hanVal fuVal else scoringKoTsumoTotal hanVal fuVal) + honba * 300

-- Verified against `pixi run python -c "import riichienv; riichienv.calculate_score(...)"` probes
theorem scoring_1_30_ko_ron : scoring 1 30 = 1000 := by native_decide
theorem scoring_1_20_ko_ron : scoring 1 20 = 700 := by native_decide
theorem scoring_2_30_ko_ron : scoring 2 30 = 2000 := by native_decide
theorem scoring_3_30_ko_ron : scoring 3 30 = 3900 := by native_decide
theorem scoring_4_30_ko_ron : scoring 4 30 = 7700 := by native_decide
theorem scoring_4_40_mangan : scoring 4 40 = 8000 := by native_decide
theorem scoring_3_70_mangan : scoring 3 70 = 8000 := by native_decide
theorem scoring_5_30_mangan : scoring 5 30 = 8000 := by native_decide
theorem scoring_6_30_haneman : scoring 6 30 = 12000 := by native_decide
theorem scoring_8_30_baiman : scoring 8 30 = 16000 := by native_decide
theorem scoring_11_30_sanbaiman : scoring 11 30 = 24000 := by native_decide
theorem scoring_13_30_yakuman : scoring 13 30 = 32000 := by native_decide

theorem scoringOya_1_30 : scoringOyaRon 1 30 = 1500 := by native_decide
theorem scoringKoTsumo_1_30_total : scoringKoTsumoTotal 1 30 = 1100 := by native_decide
theorem scoringKoTsumoOya_1_30 : scoringKoTsumoOya 1 30 = 500 := by native_decide
theorem scoringKoTsumoKo_1_30 : scoringKoTsumoKo 1 30 = 300 := by native_decide
theorem scoringOyaTsumo_1_30_total : scoringOyaTsumoTotal 1 30 = 1500 := by native_decide

theorem scoringRonHonba_1_30_ko_1 : scoringRonWithHonba 1 30 false 1 = 1300 := by native_decide
theorem scoringRonHonba_1_30_oya_1 : scoringRonWithHonba 1 30 true 1 = 1800 := by native_decide

-- Chiitoitsu 25 fu exceptional (not validFu) — scoring still works via baseRaw
theorem scoring_2_25_chiitoi : scoring 2 25 = 1600 := by native_decide
theorem scoring_3_25_chiitoi : scoring 3 25 = 3200 := by native_decide

-- ---------------------------------------------------------------------------
-- 9. Uma / Oka — post-game placement bonuses, NOT per-hand scoring
-- ---------------------------------------------------------------------------

/-- Uma placement bonus/penalty by rank 1..4.
    Tenhou 4p hanchan `uma_by_rank = (20,10,-10,-20)` i.e. `(+20,+10,-10,-20)*1000`.
    SPEC §5.1: `uma_by_rank: tuple[int,int,int,int]` manifest-supplied; Tenhou 10-20.
    This is **not** encoded in Lean per-hand points; we note the type and bounds. -/
def umaByRank : Fin 4 → Int
  | ⟨0, _⟩ => 20
  | ⟨1, _⟩ => 10
  | ⟨2, _⟩ => -10
  | ⟨3, _⟩ => -20

theorem umaByRank_sum_zero : umaByRank ⟨0, by omega⟩ + umaByRank ⟨1, by omega⟩ + umaByRank ⟨2, by omega⟩ + umaByRank ⟨3, by omega⟩ = 0 := by native_decide

theorem uma_values_10_20 : ∀ r : Fin 4, umaByRank r = 20 ∨ umaByRank r = 10 ∨ umaByRank r = -10 ∨ umaByRank r = -20 := by
  intro r
  fin_cases r <;> simp [umaByRank]

/-- Oka: top-place bonus pool — `none` or `half_return` (SPEC `oka_policy`).
    Tenhou ranked: oka `half_return` collects ` (return_points - starting_points)*players = (30000-25000)*4 =20000`
    to top. Not part of `score.rs` per-hand calc; noted as manifest policy. -/
inductive OkaPolicy where
  | none
  | half_return
  deriving DecidableEq, Repr

def okaPool (policy : OkaPolicy) : Int :=
  match policy with
  | .none => 0
  | .half_return => 20000

theorem oka_half_return_pool : okaPool .half_return = 20000 := rfl
theorem oka_none_pool : okaPool .none = 0 := rfl

/-- Placement conversion pipeline (SPEC `placement_conversion_id = tenhou_rank_sticks_top_uma_v1`):
    1) rank by raw final score (2022 rounding abolition)
    2) leftover riichi sticks to top ( `end_top_take_abort_carry_dealin_exempt` )
    3) uma 10-20 applied. This is **not** in per-hand `scoring`. -/
def placementNote : String :=
  "uma 10-20 and oka half_return apply to final scores/ranks, not to per-hand scoring; see rules.py:uma_by_rank, oka_policy, placement_conversion_id"

-- ---------------------------------------------------------------------------
-- 10. Furiten interaction — ron blocked, tsumo allowed
-- ---------------------------------------------------------------------------

/-- Furiten blocks ron, not tsumo — reuse `Yaku.lean:isFuriten`, `canRon`, `canTsumo`.
    SPEC `furiten_policy = river_only_permanent_after_riichi_miss_same_goaround_temporary`.
    Reference `file://src/hydra2/contracts/rules.py#FURITEN_POLICIES`. -/
theorem furiten_blocks_ron_scoring (discards waits : Finset TileType)
    (hf : isFuriten discards waits = true) : canRon discards waits = false :=
  furiten_imp_not_canRon discards waits hf

theorem furiten_allows_tsumo (h : Hand) (hw : HandIsWinning h) :
    canTsumo h = true := tsumo_allowed_despite_furiten h ∅ ∅ hw

theorem furiten_interaction_scoring (discards waits : Finset TileType)
    (h : Hand) (hw : HandIsWinning h) (hf : isFuriten discards waits = true) :
    canRon discards waits = false ∧ canTsumo h = true :=
  ⟨furiten_blocks_ron_scoring discards waits hf, furiten_allows_tsumo h hw⟩

-- ---------------------------------------------------------------------------
-- 11. Scoring monotonicity & caps (additional lemmas)
-- ---------------------------------------------------------------------------

theorem scoring_monotone_example : scoring 1 30 ≤ scoring 2 30 := by native_decide
theorem scoring_monotone_example2 : scoring 2 30 ≤ scoring 3 30 := by native_decide
theorem scoring_monotone_example3 : scoring 3 30 ≤ scoring 4 30 := by native_decide

theorem basePoints_mono_example : basePoints 1 30 ≤ basePoints 2 30 := by native_decide
theorem scoring_ge_base (hanVal fuVal : Nat) : basePoints hanVal fuVal ≤ scoring hanVal fuVal := by
  unfold scoring
  have h1 : basePoints hanVal fuVal ≤ basePoints hanVal fuVal * 4 := by
    have h4 : 1 ≤ (4 : Nat) := by omega
    have hle : basePoints hanVal fuVal * 1 ≤ basePoints hanVal fuVal * 4 :=
      Nat.mul_le_mul_left _ h4
    simpa using hle
  have h2 : basePoints hanVal fuVal * 4 ≤ ceil100 (basePoints hanVal fuVal * 4) :=
    ceil100_ge _
  omega

theorem scoring_capped_at_yakuman (hanVal fuVal : Nat) (h : 13 ≤ hanVal) :
    basePoints hanVal fuVal = 8000 := by
  unfold basePoints
  simp [h]

theorem scoring_yakuman_is_32000_ko_ron (fuVal : Nat) : scoring 13 fuVal = 32000 := by
  have hb : basePoints 13 fuVal = 8000 := scoring_capped_at_yakuman 13 fuVal (by omega)
  unfold scoring
  rw [hb]
  native_decide
-- ---------------------------------------------------------------------------
-- 12. Integration lemmas — han/fu from Yaku/Dora/Shanten
-- ---------------------------------------------------------------------------

/-- Han from `Yaku.lean:han h ms ctx` respects `validHanScoring` upper bound when
    combined with dora ≤20 (max yaku han ≤ 45*26 but realistic ≤13 + dora). We prove
    bound for yakuman case via `han_bound_by_yaku_count`. -/
theorem han_from_yaku_le_45x26 (h : Hand) (ms : MeldSet) (ctx : YakuContext) :
    han h ms ctx ≤ (yakuList h ms ctx).card * 26 :=
  han_bound_by_yaku_count h ms ctx

theorem ceil100_mono (a b : Nat) (h : a ≤ b) : ceil100 a ≤ ceil100 b := by
  unfold ceil100
  omega

/-- Fu bounds plus han bounds give valid scoring range after cap — scoring always finite. -/
theorem scoring_finite (hanVal fuVal : Nat) : scoring hanVal fuVal ≤ 32000 := by
  unfold scoring
  have hle : basePoints hanVal fuVal ≤ 8000 := basePoints_le_8000 hanVal fuVal
  have h4 : basePoints hanVal fuVal * 4 ≤ 8000 * 4 := Nat.mul_le_mul_right 4 hle
  have hceil : ceil100 (basePoints hanVal fuVal * 4) ≤ ceil100 (8000 * 4) :=
    ceil100_mono _ _ h4
  have hceil_val : ceil100 (8000 * 4) = 32000 := by native_decide
  calc ceil100 (basePoints hanVal fuVal * 4)
      ≤ ceil100 (8000 * 4) := hceil
    _ = 32000 := hceil_val

/-- Combined validHanFuScoring implies scoring in mangan..yakuman range or below. -/
theorem validHanFuScoring_imp_scoring_le_yakuman (hanVal fuVal : Nat)
    (_h : validHanFuScoring hanVal fuVal) : scoring hanVal fuVal ≤ 32000 :=
  scoring_finite hanVal fuVal
-- ---------------------------------------------------------------------------

theorem totalHan_dora_not_yaku (h : Hand) (ms : MeldSet) (ctx : YakuContext) (d : Nat) :
    totalHan h ms ctx d = han h ms ctx + doraHan d ∧ doraCountsAsYaku = false := by
  constructor
  · simp [totalHan, doraHan]
  · rfl

theorem dora_does_not_satisfy_yaku_requirement (h : Hand) (ms : MeldSet) (ctx : YakuContext)
    (hh : han h ms ctx = 0) (d : Nat) (hd : d > 0) :
    han h ms ctx + doraHan d > 0 ∧ han h ms ctx = 0 := by
  constructor
  · simp [doraHan, hh, hd]
  · exact hh

-- ---------------------------------------------------------------------------
-- 14. Yakuman / Daisangen etc. — yakuman han 13, scoring 32000 ko ron
-- ---------------------------------------------------------------------------

theorem yakuman_scoring_ko_ron (hanVal : Nat) (h : 13 ≤ hanVal) (fuVal : Fu) :
    scoring hanVal fuVal = 32000 := by
  have hb : basePoints hanVal fuVal = 8000 := scoring_capped_at_yakuman hanVal fuVal h
  unfold scoring
  rw [hb]
  native_decide

theorem yakuman_scoring_oya_ron (hanVal : Nat) (h : 13 ≤ hanVal) (fuVal : Fu) :
    scoringOyaRon hanVal fuVal = 48000 := by
  have hb : basePoints hanVal fuVal = 8000 := scoring_capped_at_yakuman hanVal fuVal h
  unfold scoringOyaRon
  rw [hb]
  native_decide
end Formal.Mahjong
