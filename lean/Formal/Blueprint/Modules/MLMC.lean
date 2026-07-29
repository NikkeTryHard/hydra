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
# Hydra2 §11.4 Fixed MLMC — finite `Fintype` case proven: `mlmc_telescope` (`induction` `Icc` `sum_insert`) + `mlmc_missing_correction_bias` (`sum_erase` omit-k failure). HARD skip (prose/spec refs only, no Lean defs here): `3-level` `signed` `deterministic` `tiny test`, `SPEC MLMC-TELESCOPE-001`, `pilot-frozen` `ladder`+`counts`, `independent groups` `ProductMeasure`, `paired levels`, `residual bias zero only when L exact`, `reject outcome-dependent allocation`, `Lagrange` `Nℓ*∝√(Vℓ/Cℓ)` `PyApprox`, `Giles 2008/2015` `αβγ` `O(ε⁻²)` + general `MeasureTheory` `HasFiniteIntegral` `tower law`.
-/
namespace Hydra2.Blueprint.Modules.MLMC

section Telescoping

variable {Sample : Type} [Fintype Sample] [DecidableEq Sample]

noncomputable def mlmcExpectation (levels : Nat → Sample → ℝ) (prob : Sample → ℝ) (ℓ : Nat) : ℝ :=
  ∑ u : Sample, prob u * levels ℓ u

noncomputable def mlmcDiff (levels : Nat → Sample → ℝ) (prob : Sample → ℝ) (ℓ : Nat) : ℝ :=
  mlmcExpectation levels prob ℓ - mlmcExpectation levels prob (ℓ - 1)

theorem mlmc_telescope (levels : Nat → Sample → ℝ) (prob : Sample → ℝ) (L : Nat) :
    mlmcExpectation levels prob L =
    mlmcExpectation levels prob 0 + ∑ ℓ ∈ Finset.Icc 1 L, mlmcDiff levels prob ℓ := by
  induction L with
  | zero =>
    have hEmpty : Finset.Icc 1 0 = (∅ : Finset Nat) := by
      simp
    simp [hEmpty]
  | succ n ih =>
    have h_le : (1 : Nat) ≤ n + 1 := by omega
    have h_mem : (n + 1) ∉ Finset.Icc 1 n := by
      simp [Finset.mem_Icc]
    have h_Icc_succ : Finset.Icc 1 (n + 1) = insert (n + 1) (Finset.Icc 1 n) := by
      have h_eq : Finset.Icc 1 (n + 1) = Finset.Icc 1 (n.succ) := by rfl
      rw [h_eq]
      have h2 : insert n.succ (Finset.Icc 1 n) = Finset.Icc 1 n.succ :=
        Finset.insert_Icc_right_eq_Icc_succ h_le
      rw [h2]
    have h_sum_insert : ∑ ℓ ∈ Finset.Icc 1 (n + 1), mlmcDiff levels prob ℓ =
        mlmcDiff levels prob (n + 1) + ∑ ℓ ∈ Finset.Icc 1 n, mlmcDiff levels prob ℓ := by
      rw [h_Icc_succ, Finset.sum_insert h_mem]
    have h_diff : mlmcDiff levels prob (n + 1) = mlmcExpectation levels prob (n + 1) - mlmcExpectation levels prob n := by
      unfold mlmcDiff
      have h_sub : n + 1 - 1 = n := by omega
      rw [h_sub]
    calc mlmcExpectation levels prob (n + 1)
        = mlmcExpectation levels prob n + (mlmcExpectation levels prob (n + 1) - mlmcExpectation levels prob n) := by ring
      _ = mlmcExpectation levels prob n + mlmcDiff levels prob (n + 1) := by rw [← h_diff]
      _ = (mlmcExpectation levels prob 0 + ∑ ℓ ∈ Finset.Icc 1 n, mlmcDiff levels prob ℓ) + mlmcDiff levels prob (n + 1) := by rw [ih]
      _ = mlmcExpectation levels prob 0 + (∑ ℓ ∈ Finset.Icc 1 n, mlmcDiff levels prob ℓ + mlmcDiff levels prob (n + 1)) := by ring
      _ = mlmcExpectation levels prob 0 + (mlmcDiff levels prob (n + 1) + ∑ ℓ ∈ Finset.Icc 1 n, mlmcDiff levels prob ℓ) := by ring
      _ = mlmcExpectation levels prob 0 + ∑ ℓ ∈ Finset.Icc 1 (n + 1), mlmcDiff levels prob ℓ := by rw [← h_sum_insert]

theorem mlmc_residual_bias_zero_iff_exact
    (levels : Nat → Sample → ℝ) (prob : Sample → ℝ) (L : Nat) (trueValue : ℝ)
    (hExact : mlmcExpectation levels prob L = trueValue) :
    mlmcExpectation levels prob L - trueValue = 0 := by
  linarith

theorem mlmc_missing_correction_bias
    (levels : Nat → Sample → ℝ) (prob : Sample → ℝ) (L k : Nat) (hk : 1 ≤ k ∧ k ≤ L)
    (hNonzero : mlmcDiff levels prob k ≠ 0) :
    (mlmcExpectation levels prob 0 + ∑ ℓ ∈ (Finset.Icc 1 L).erase k, mlmcDiff levels prob ℓ)
    ≠ mlmcExpectation levels prob L := by
  intro hEq
  have h_mem : k ∈ Finset.Icc 1 L := Finset.mem_Icc.mpr hk
  have h_sum_erase : ∑ ℓ ∈ (Finset.Icc 1 L).erase k, mlmcDiff levels prob ℓ =
      (∑ ℓ ∈ Finset.Icc 1 L, mlmcDiff levels prob ℓ) - mlmcDiff levels prob k := by
    rw [Finset.sum_erase_eq_sub h_mem]
  have h_tel := mlmc_telescope levels prob L
  have h1 : mlmcExpectation levels prob 0 + ∑ ℓ ∈ (Finset.Icc 1 L).erase k, mlmcDiff levels prob ℓ =
      mlmcExpectation levels prob L - mlmcDiff levels prob k := by
    rw [h_sum_erase]
    have h : mlmcExpectation levels prob 0 + (∑ ℓ ∈ Finset.Icc 1 L, mlmcDiff levels prob ℓ - mlmcDiff levels prob k) =
        (mlmcExpectation levels prob 0 + ∑ ℓ ∈ Finset.Icc 1 L, mlmcDiff levels prob ℓ) - mlmcDiff levels prob k := by ring
    rw [h, ← h_tel]
  have h2 : mlmcDiff levels prob k = 0 := by linarith
  exact hNonzero h2

end Telescoping

end Hydra2.Blueprint.Modules.MLMC
