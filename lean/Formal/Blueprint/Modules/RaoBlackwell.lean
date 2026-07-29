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
# Hydra2 §11.1 Transition Rao-Blackwellization
Blueprint §11.1, sources in header.
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

theorem costRB_ge_sample (cardX cardY : Nat) (hY : 1 ≤ cardY) : costSample cardX ≤ costRB cardX cardY := by
  unfold costSample costRB
  have : cardX * 1 ≤ cardX * cardY := Nat.mul_le_mul_left cardX hY
  simpa using this

end FiniteRB

end Hydra2.Blueprint.Modules.RaoBlackwell
