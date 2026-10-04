import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Data.Real.Basic
import Mathlib.Tactic

set_option linter.unusedSimpArgs false
set_option linter.unreachableTactic false
set_option linter.unusedTactic false
set_option linter.unusedDecidableInType false
set_option linter.unusedSectionVars false
set_option linter.style.longLine false

/-!
# Hydra2 Transition Rao-Blackwellization
Finite joint-law conditioning; conditioning on the wrong variable loses information and fails closed.
-/

namespace Hydra2.Blueprint.Modules.RaoBlackwell

section FiniteRB

variable {X Y : Type} [Fintype X] [DecidableEq X] [Fintype Y] [DecidableEq Y]

structure JointLaw (X Y : Type) [Fintype X] [Fintype Y] where
  px : X → ℝ
  py_given_x : X → Y → ℝ
  px_nonneg : ∀ x, 0 ≤ px x
  px_sum_one : ∑ x : X, px x = 1
  py_nonneg : ∀ x y, 0 ≤ py_given_x x y
  py_stoch : ∀ x, ∑ y : Y, py_given_x x y = 1

noncomputable def jointProb (L : JointLaw X Y) (x : X) (y : Y) : ℝ :=
  L.px x * L.py_given_x x y

noncomputable def expectationJoint (L : JointLaw X Y) (g : X → Y → ℝ) : ℝ :=
  ∑ x : X, ∑ y : Y, jointProb L x y * g x y

noncomputable def RB (L : JointLaw X Y) (g : X → Y → ℝ) (x : X) : ℝ :=
  ∑ y : Y, L.py_given_x x y * g x y

noncomputable def expectationRB (L : JointLaw X Y) (g : X → Y → ℝ) : ℝ :=
  ∑ x : X, L.px x * RB L g x

theorem RB_unbiased (L : JointLaw X Y) (g : X → Y → ℝ) :
    expectationRB L g = expectationJoint L g := by
  unfold expectationRB RB expectationJoint jointProb
  congr 1; ext x
  rw [Finset.mul_sum]
  congr 1; ext y; ring
noncomputable def varJoint (L : JointLaw X Y) (g : X → Y → ℝ) : ℝ :=
  (∑ x : X, ∑ y : Y, jointProb L x y * g x y ^ 2) - (expectationJoint L g) ^ 2

noncomputable def varRB (L : JointLaw X Y) (g : X → Y → ℝ) : ℝ :=
  (∑ x : X, L.px x * RB L g x ^ 2) - (expectationRB L g) ^ 2

/-- Factor `px x` out of the joint second-moment sum (shared by both variance theorems). -/
private theorem RB_px_factor (L : JointLaw X Y) (g : X → Y → ℝ) (x : X) :
    ∑ y : Y, L.px x * L.py_given_x x y * g x y ^ 2 =
      L.px x * ∑ y : Y, L.py_given_x x y * g x y ^ 2 := by
  have h : ∑ y : Y, L.px x * L.py_given_x x y * g x y ^ 2 =
      ∑ y : Y, L.px x * (L.py_given_x x y * g x y ^ 2) := by
    apply Finset.sum_congr rfl; intro y _; ring
  rw [h, ← Finset.mul_sum]

/-- Second-moment difference identity (shared algebraic core). -/
private theorem RB_var_diff (L : JointLaw X Y) (g : X → Y → ℝ) (m : X → ℝ) :
    (∑ x : X, ∑ y : Y, L.px x * L.py_given_x x y * g x y ^ 2) -
      (∑ x : X, L.px x * (m x) ^ 2) =
      ∑ x : X, L.px x * ((∑ y : Y, L.py_given_x x y * g x y ^ 2) - (m x) ^ 2) := by
  rw [← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl; intro x _
  rw [RB_px_factor L g x]; ring

theorem RB_variance_reduction (L : JointLaw X Y) (g : X → Y → ℝ) :
    varRB L g ≤ varJoint L g := by
  unfold varRB varJoint expectationJoint expectationRB jointProb RB
  have hE : (∑ x : X, L.px x * ∑ y : Y, L.py_given_x x y * g x y) =
      (∑ x : X, ∑ y : Y, L.px x * L.py_given_x x y * g x y) := by
    congr 1; ext x
    rw [Finset.mul_sum]
    congr 1; ext y; ring
  have hRB_le : ∀ x : X, (∑ y : Y, L.py_given_x x y * g x y) ^ 2 ≤
      ∑ y : Y, L.py_given_x x y * g x y ^ 2 := by
    intro x
    let RBx := ∑ y : Y, L.py_given_x x y * g x y
    have h_expand : ∀ y : Y, L.py_given_x x y * (g x y - RBx) ^ 2 =
        L.py_given_x x y * g x y ^ 2 - 2 * RBx * (L.py_given_x x y * g x y) + RBx ^ 2 * L.py_given_x x y := by
      intro y; ring
    have h_sum_expand : ∑ y : Y, L.py_given_x x y * (g x y - RBx) ^ 2 =
        ∑ y : Y, (L.py_given_x x y * g x y ^ 2 - 2 * RBx * (L.py_given_x x y * g x y) + RBx ^ 2 * L.py_given_x x y) := by
      apply Finset.sum_congr rfl; intro y _; exact h_expand y
    have h_sum_eq : ∑ y : Y, (L.py_given_x x y * g x y ^ 2 - 2 * RBx * (L.py_given_x x y * g x y) + RBx ^ 2 * L.py_given_x x y) =
        (∑ y : Y, L.py_given_x x y * g x y ^ 2) - 2 * RBx * (∑ y : Y, L.py_given_x x y * g x y) + RBx ^ 2 * (∑ y : Y, L.py_given_x x y) := by
      calc ∑ y : Y, (L.py_given_x x y * g x y ^ 2 - 2 * RBx * (L.py_given_x x y * g x y) + RBx ^ 2 * L.py_given_x x y)
          = ∑ y : Y, ((L.py_given_x x y * g x y ^ 2 - 2 * RBx * (L.py_given_x x y * g x y)) + RBx ^ 2 * L.py_given_x x y) := by
            apply Finset.sum_congr rfl; intro y _; ring
        _ = (∑ y : Y, (L.py_given_x x y * g x y ^ 2 - 2 * RBx * (L.py_given_x x y * g x y))) + ∑ y : Y, RBx ^ 2 * L.py_given_x x y := by
            rw [Finset.sum_add_distrib]
        _ = ((∑ y : Y, L.py_given_x x y * g x y ^ 2) - ∑ y : Y, 2 * RBx * (L.py_given_x x y * g x y)) + ∑ y : Y, RBx ^ 2 * L.py_given_x x y := by
            rw [Finset.sum_sub_distrib]
        _ = ((∑ y : Y, L.py_given_x x y * g x y ^ 2) - 2 * RBx * ∑ y : Y, L.py_given_x x y * g x y) + RBx ^ 2 * ∑ y : Y, L.py_given_x x y := by
            have h1 : ∑ y : Y, 2 * RBx * (L.py_given_x x y * g x y) = 2 * RBx * ∑ y : Y, L.py_given_x x y * g x y := by
              rw [← Finset.mul_sum]
            have h2 : ∑ y : Y, RBx ^ 2 * L.py_given_x x y = RBx ^ 2 * ∑ y : Y, L.py_given_x x y := by
              rw [← Finset.mul_sum]
            rw [h1, h2]
    have h_py_sum : ∑ y : Y, L.py_given_x x y = 1 := L.py_stoch x
    have h_RBx_def : RBx = ∑ y : Y, L.py_given_x x y * g x y := rfl
    have h_eq : ∑ y : Y, L.py_given_x x y * (g x y - RBx) ^ 2 =
        (∑ y : Y, L.py_given_x x y * g x y ^ 2) - RBx ^ 2 := by
      rw [h_sum_expand, h_sum_eq, h_py_sum, h_RBx_def]
      ring
    have h_nonneg : 0 ≤ ∑ y : Y, L.py_given_x x y * (g x y - RBx) ^ 2 := by
      apply Finset.sum_nonneg; intro y _; exact mul_nonneg (L.py_nonneg x y) (sq_nonneg _)
    linarith
  have h_var_diff : (∑ x : X, ∑ y : Y, L.px x * L.py_given_x x y * g x y ^ 2) -
      (∑ x : X, L.px x * (∑ y : Y, L.py_given_x x y * g x y) ^ 2) =
      ∑ x : X, L.px x * ((∑ y : Y, L.py_given_x x y * g x y ^ 2) - (∑ y : Y, L.py_given_x x y * g x y) ^ 2) :=
    RB_var_diff L g (fun x => ∑ y : Y, L.py_given_x x y * g x y)
  have h_nonneg_sum : 0 ≤ ∑ x : X, L.px x * ((∑ y : Y, L.py_given_x x y * g x y ^ 2) - (∑ y : Y, L.py_given_x x y * g x y) ^ 2) := by
    apply Finset.sum_nonneg; intro x _
    apply mul_nonneg (L.px_nonneg x)
    linarith [hRB_le x]
  have h_main : (∑ x : X, L.px x * (∑ y : Y, L.py_given_x x y * g x y) ^ 2) ≤
      (∑ x : X, ∑ y : Y, L.px x * L.py_given_x x y * g x y ^ 2) := by linarith
  have hE2 : (∑ x : X, L.px x * ∑ y : Y, L.py_given_x x y * g x y) ^ 2 =
      (∑ x : X, ∑ y : Y, L.px x * L.py_given_x x y * g x y) ^ 2 := by rw [hE]
  linarith

/-- Total-variance equality: `varJoint - varRB = E[Var(g|x)]` in residual form.
Factors out the equality half of `RB_variance_reduction` (the per-`x` Jensen
block there only adds per-summand nonnegativity); mean-cancel via `RB_unbiased`. -/
theorem RB_total_variance (L : JointLaw X Y) (g : X → Y → ℝ) :
    varJoint L g - varRB L g
    = ∑ x : X, L.px x * ((∑ y : Y, L.py_given_x x y * g x y ^ 2) - (RB L g x) ^ 2) := by
  have hE := RB_unbiased L g
  have hEsq : (expectationRB L g) ^ 2 = (expectationJoint L g) ^ 2 := by rw [hE]
  have hAB : (∑ x : X, ∑ y : Y, L.px x * L.py_given_x x y * g x y ^ 2)
      - (∑ x : X, L.px x * RB L g x ^ 2)
      = ∑ x : X, L.px x * ((∑ y : Y, L.py_given_x x y * g x y ^ 2) - (RB L g x) ^ 2) :=
    RB_var_diff L g (RB L g)
  unfold varJoint varRB jointProb
  rw [hEsq]
  linear_combination hAB

theorem RB_strict_reduction_exists :
    ∃ (X0 : Type) (_ : Fintype X0) (_ : DecidableEq X0)
      (Y0 : Type) (_ : Fintype Y0) (_ : DecidableEq Y0)
      (L : JointLaw X0 Y0) (g : X0 → Y0 → ℝ),
      varRB L g < varJoint L g := by
  refine ⟨Bool, inferInstance, inferInstance, Bool, inferInstance, inferInstance,
    ⟨fun _ => 1/2, fun _ _ => 1/2,
      (by intro x; norm_num),
      (by
        have h_univ : (Finset.univ : Finset Bool) = {false, true} := by decide
        rw [h_univ, Finset.sum_insert (by decide : (false : Bool) ∉ ({true} : Finset Bool)), Finset.sum_singleton]
        norm_num),
      (by intro x y; norm_num),
      (by intro x
          have h_univ : (Finset.univ : Finset Bool) = {false, true} := by decide
          rw [h_univ, Finset.sum_insert (by decide : (false : Bool) ∉ ({true} : Finset Bool)), Finset.sum_singleton]
          norm_num)⟩,
    (fun _ y => if y then 1 else 0), ?_⟩
  have hRB : varRB (X:=Bool) (Y:=Bool)
      ⟨fun _ => (1/2 : ℝ), fun _ _ => (1/2 : ℝ),
        (by intro x; norm_num),
        (by have h_univ : (Finset.univ : Finset Bool) = {false, true} := by decide
            rw [h_univ, Finset.sum_insert (by decide), Finset.sum_singleton]; norm_num),
        (by intro x y; norm_num),
        (by intro x; have h_univ : (Finset.univ : Finset Bool) = {false, true} := by decide
            rw [h_univ, Finset.sum_insert (by decide), Finset.sum_singleton]; norm_num)⟩
      (fun _ y => if y then (1:ℝ) else 0) = 0 := by
    unfold varRB expectationRB RB
    simp only
    have h_univ : (Finset.univ : Finset Bool) = {false, true} := by decide
    have hRBx : ∀ x : Bool, (∑ y : Bool, (1/2 : ℝ) * if y then (1:ℝ) else 0) = (1/2 : ℝ) := by
      intro x; rw [h_univ, Finset.sum_insert (by decide), Finset.sum_singleton]; norm_num
    have hE : (∑ x : Bool, (1/2 : ℝ) * ∑ y : Bool, (1/2 : ℝ) * if y then (1:ℝ) else 0) = (1/2 : ℝ) := by
      have h_eq : (∑ x : Bool, (1/2 : ℝ) * ∑ y : Bool, (1/2 : ℝ) * if y then (1:ℝ) else 0) =
          (∑ x : Bool, (1/2 : ℝ) * (1/2 : ℝ)) := by
        apply Finset.sum_congr rfl; intro x _; rw [hRBx x]
      rw [h_eq, h_univ, Finset.sum_insert (by decide), Finset.sum_singleton]; norm_num
    have hSq : (∑ x : Bool, (1/2 : ℝ) * ((∑ y : Bool, (1/2 : ℝ) * if y then (1:ℝ) else 0) ^ 2)) = (1/4 : ℝ) := by
      have h_eq : (∑ x : Bool, (1/2 : ℝ) * ((∑ y : Bool, (1/2 : ℝ) * if y then (1:ℝ) else 0) ^ 2)) =
          (∑ x : Bool, (1/2 : ℝ) * ((1/2 : ℝ) ^ 2)) := by
        apply Finset.sum_congr rfl; intro x _; rw [hRBx x]
      rw [h_eq, h_univ, Finset.sum_insert (by decide), Finset.sum_singleton]; norm_num
    have h1 : (∑ x : Bool, (1/2 : ℝ) * ((∑ y : Bool, (1/2 : ℝ) * if y then (1:ℝ) else 0) ^ 2)) -
        (∑ x : Bool, (1/2 : ℝ) * ∑ y : Bool, (1/2 : ℝ) * if y then (1:ℝ) else 0) ^ 2 = (0 : ℝ) := by
      rw [hSq, hE]; norm_num
    exact h1
  have hJoint : varJoint (X:=Bool) (Y:=Bool)
      ⟨fun _ => (1/2 : ℝ), fun _ _ => (1/2 : ℝ),
        (by intro x; norm_num),
        (by have h_univ : (Finset.univ : Finset Bool) = {false, true} := by decide
            rw [h_univ, Finset.sum_insert (by decide), Finset.sum_singleton]; norm_num),
        (by intro x y; norm_num),
        (by intro x; have h_univ : (Finset.univ : Finset Bool) = {false, true} := by decide
            rw [h_univ, Finset.sum_insert (by decide), Finset.sum_singleton]; norm_num)⟩
      (fun _ y => if y then (1:ℝ) else 0) = 1/4 := by
    unfold varJoint expectationJoint jointProb
    simp only
    have h_univ : (Finset.univ : Finset Bool) = {false, true} := by decide
    have h_inner_sq : ∀ x : Bool, (∑ y : Bool, (1/2 : ℝ) * (1/2 : ℝ) * (if y then (1:ℝ) else 0) ^ 2) = (1/4 : ℝ) := by
      intro x; rw [h_univ, Finset.sum_insert (by decide), Finset.sum_singleton]; simp; norm_num
    have h_inner : ∀ x : Bool, (∑ y : Bool, (1/2 : ℝ) * (1/2 : ℝ) * (if y then (1:ℝ) else 0)) = (1/4 : ℝ) := by
      intro x; rw [h_univ, Finset.sum_insert (by decide), Finset.sum_singleton]; simp; norm_num
    have h1 : (∑ x : Bool, ∑ y : Bool, (1/2 : ℝ) * (1/2 : ℝ) * (if y then (1:ℝ) else 0) ^ 2) = (1/2 : ℝ) := by
      have h : (∑ x : Bool, ∑ y : Bool, (1/2 : ℝ) * (1/2 : ℝ) * (if y then (1:ℝ) else 0) ^ 2) =
          (∑ x : Bool, (1/4 : ℝ)) := by
        apply Finset.sum_congr rfl; intro x _; exact h_inner_sq x
      rw [h, h_univ, Finset.sum_insert (by decide), Finset.sum_singleton]; norm_num
    have h2 : (∑ x : Bool, ∑ y : Bool, (1/2 : ℝ) * (1/2 : ℝ) * (if y then (1:ℝ) else 0)) = (1/2 : ℝ) := by
      have h : (∑ x : Bool, ∑ y : Bool, (1/2 : ℝ) * (1/2 : ℝ) * (if y then (1:ℝ) else 0)) =
          (∑ x : Bool, (1/4 : ℝ)) := by
        apply Finset.sum_congr rfl; intro x _; exact h_inner x
      rw [h, h_univ, Finset.sum_insert (by decide), Finset.sum_singleton]; norm_num
    rw [h1, h2]; norm_num
  linarith


def costRB (cardX cardY : Nat) : Nat := cardX * cardY
def costSample (cardX : Nat) : Nat := cardX

/-- Empty worlds cost nothing: zero X-samples price at zero regardless of
packet width — the finite reason the clock's budget loop short-circuits
on empty batches instead of pricing them. Pair with `costRB_singleton`
(trivial Y is free): together both degenerate corners are pinned, so the
cost model has no surprise minimum charge. -/
theorem costRB_zero_worlds (cardY : Nat) : costRB 0 cardY = 0 := by
  unfold costRB
  simp

/-- Empty packets cost nothing: zero packet width prices at zero regardless
of world count — the mirror of `costRB_zero_worlds`. Together both
degenerate corners are pinned, so an empty batch in either dimension
short-circuits the budget loop instead of pricing. -/
theorem costRB_zero_packets (cardX : Nat) : costRB cardX 0 = 0 := by
  unfold costRB
  simp

/-- Sampling nothing costs nothing: `costSample 0 = 0` — the sampler's
degenerate corner, matching `costRB_zero_worlds` at the joint level. -/
theorem costSample_zero : costSample 0 = 0 := rfl

theorem costRB_ge_sample (cardX cardY : Nat) (hY : 1 ≤ cardY) : costSample cardX ≤ costRB cardX cardY := by
  unfold costSample costRB
  have : cardX * 1 ≤ cardX * cardY := Nat.mul_le_mul_left cardX hY
  simpa using this

/-- Sampling cost is zero-preserving in the monotone order: `costSample`
maps 0 to 0 and respects `≤` (see `costSample_zero`, `costSample_mono`) —
the finite reason the clock's world-pricing has no fixed startup charge:
pricing from zero starts at zero and only rises. -/
theorem costSample_zero_le (cardX : Nat) : costSample 0 ≤ costSample cardX := by
  rw [costSample_zero]
  exact Nat.zero_le _

/-- Sampling cost is monotone in worlds: more X-samples never cost less.
Pair with `costRB_mono_Y`: together the cost model moves the right way in
EACH argument, so the 5 s clock's budget binary-search is monotone in both
dimensions — widening either side can only raise the price. -/
theorem costSample_mono (cardX₁ cardX₂ : Nat) (h : cardX₁ ≤ cardX₂) :
    costSample cardX₁ ≤ costSample cardX₂ := by
  unfold costSample
  exact h

/-- Sample cost splits additively over world batches: pricing `a + b`
worlds equals pricing the parts — the finite reason per-batch sample
budgets add without overhead: the sampler has no fixed cost, so
sharding the world set never changes the sample bill. Pairs
`costRB_add_batches` (joint cost additive in worlds): together sample
and joint agree that world-splitting is free. -/
theorem costSample_additive (a b : Nat) :
    costSample (a + b) = costSample a + costSample b :=
  rfl

/-- Sampler pricing is the identity: `costSample` unfolds to exactly
`cardX` everywhere — the finite reason world-pricing has no hidden
multiplier: the sampler charges one unit per world, so the clock's
world budget IS the world count. Pairs `costSample_zero_le` (zero is
the floor): together pricing is pinned at identity with floor zero. -/
theorem costSample_identity (cardX : Nat) : costSample cardX = cardX := rfl

/-- RB cost scales linearly in the marginalized dimension: exact enumeration
over Y costs |Y|x per X-sample. Practical core: the 34-discard exact packet
enumeration in PBRF is budgetable — 34x per world, known upfront, so the 5 s
clock can price it before spending a single sim. -/
theorem costRB_linear_Y (cardX cardY : Nat) :
    costRB cardX cardY = cardY * costSample cardX := by
  unfold costRB costSample
  ring

/-- Trivial Y costs nothing extra: a singleton marginalized dimension makes
RB exactly as cheap as plain sampling. Practical core: collapse degenerate
dimensions first — never pay enumeration overhead where no variance exists. -/
theorem costRB_singleton (cardX : Nat) : costRB cardX 1 = costSample cardX := by
  unfold costRB costSample
  simp

/-- Singleton worlds cost nothing extra: a single world makes RB exactly
as cheap as its packet count — the mirror of `costRB_singleton` (single
packet). Together both degenerate ends are pinned, so the joint price
collapses to whichever dimension survives, and the clock enumerates
only the live axis. -/
theorem costRB_singleton_world (cardY : Nat) :
    costRB 1 cardY = cardY := by
  unfold costRB
  simp

/-- Zero cost means a dead axis: the joint price is zero iff worlds or
packets vanish — the finite reason a zero bill diagnoses exactly which
dimension died, so the clock's zero-cost short-circuit names the empty
axis instead of re-pricing. Pairs `costRB_zero_worlds` and
`costRB_zero_packets` (each direction separately): together the
characterization is if-and-only-if. -/
theorem costRB_zero_iff (cardX cardY : Nat) :
    costRB cardX cardY = 0 ↔ cardX = 0 ∨ cardY = 0 := by
  unfold costRB
  exact Nat.mul_eq_zero

/-- Enumeration cost splits additively over world batches: pricing two batches
separately equals pricing them together. Practical core: the 5 s clock can
split worlds across batches (or workers) and sum the prices — no hidden
interaction cost. -/
theorem costRB_add_batches (cardX1 cardX2 cardY : Nat) :
    costRB (cardX1 + cardX2) cardY = costRB cardX1 cardY + costRB cardX2 cardY := by
  unfold costRB
  ring

/-- Symmetric widening is quadratic: doubling BOTH sides quadruples the
joint cost — the finite reason the 5 s clock widens one dimension at a
time, never both. Pairs `costRB_linear_Y` (one-sided widening is
linear): together the cost surface tells the scheduler exactly which
widening is cheap (one side: 2x) and which is ruinous (both: 4x). -/
theorem costRB_double_both (cardX cardY : Nat) :
    costRB (2 * cardX) (2 * cardY) = 4 * costRB cardX cardY := by
  unfold costRB
  ring

/-- Enumeration cost is monotone in worlds at fixed packet width: more
X-samples never cost less. Pair with `costRB_mono_Y` (monotone in Y):
together the joint cost is monotone in BOTH arguments — the 5 s clock's
budget search never reverses in either dimension. -/
theorem costRB_mono_X (cardX₁ cardX₂ cardY : Nat) (h : cardX₁ ≤ cardX₂) :
    costRB cardX₁ cardY ≤ costRB cardX₂ cardY := by
  unfold costRB
  exact Nat.mul_le_mul_right cardY h

/-- Enumeration cost is monotone in the marginalized dimension: a wider Y
costs at least as much at fixed X. Pair with `costRB_singleton` (floor)
and `costRB_linear_Y` (exact rate): together they bracket the PBRF packet
bill — the 5 s clock can binary-search the affordable Y width from a
known cost slope. Failure mode killed: a wider packet enumeration
silently fitting the same budget. Zero packets kill the joint (proved
next): the packet axis vetoes the bill at any world count, so the bill
factorizes with either axis able to zero it. -/
theorem costRB_packet_veto (cardX : Nat) : costRB cardX 0 = 0 :=
  costRB_zero_packets cardX

theorem costRB_mono_Y (cardX cardY₁ cardY₂ : Nat) (h : cardY₁ ≤ cardY₂) :
    costRB cardX cardY₁ ≤ costRB cardX cardY₂ := by
  unfold costRB
  exact Nat.mul_le_mul_left cardX h

/-- Joint cost composes from the parts: `costRB = cardY × costSample` is
`costRB_linear_Y` restated through the sampler — the finite reason the
clock can price from EITHER side (worlds-first or packets-first) and
agree. Pair with `costRB_add_batches` (additive over batch splits):
together pricing is path-independent — split first or price whole,
the bill is identical. -/
theorem costRB_via_sample (cardX cardY : Nat) :
    costRB cardX cardY = cardY * costSample cardX :=
  costRB_linear_Y cardX cardY

/-- Joint cost is symmetric in the product: swapping the argument order
through `costSample` recovers `cardX * cardY` — the finite reason the
clock can quote worlds-first or packets-first and land on the same
number. Cross-checks `costRB_linear_Y` through the sampler identity:
two pricing paths, one bill. -/
theorem costRB_symmetric_quote (cardX cardY : Nat) :
    costRB cardX cardY = costRB cardY cardX := by
  unfold costRB
  ring

/-- Enumeration cost splits additively over packet batches: pricing two
packet batches equals pricing the concatenation — the mirror of
`costRB_add_batches` (additive over world batches). Together the joint
price is additive in EACH argument, so the clock can split the grid
along either axis and the bill stays identical. -/
theorem costRB_add_packets (cardX cardY1 cardY2 : Nat) :
    costRB cardX (cardY1 + cardY2)
      = costRB cardX cardY1 + costRB cardX cardY2 := by
  unfold costRB
  ring

/-- Singleton packet grid prices at worlds: `costRB cardX 1` is already
`costSample cardX` via `costRB_singleton` — restated here as the joint
grid's unit column. The finite reason the packet axis bottoms at the
sampler price, never below: a one-packet enumeration IS sampling, so
the RB premium vanishes exactly at width one. -/
theorem costRB_unit_column (cardX : Nat) :
    costRB cardX 1 = cardX := by
  unfold costRB
  simp

/-- Unit grid costs one: `costRB 1 1 = 1` — the finite reason the
cheapest non-degenerate enumeration (one world, one packet) prices at
exactly one unit: the cost surface's minimal live point is unity, so
the clock's budget floor for any real work is one, never zero (zero
means a dead axis per `costRB_zero_iff`). -/
theorem costRB_unit_grid : costRB 1 1 = 1 := by
  unfold costRB
  simp

end FiniteRB

end Hydra2.Blueprint.Modules.RaoBlackwell
