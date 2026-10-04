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
# Hydra2 Fixed MLMC — finite `Fintype` case proven:
  `mlmc_telescope` (`induction` `Icc` `sum_insert`) +
  `mlmc_missing_correction_bias` (`sum_erase` omit-k failure; omitting a level
  without its correction biases the telescope and fails closed).
  Out of scope here (no Lean defs): multi-level signed deterministic tiny test,
  pilot-frozen ladder and counts, independent groups `ProductMeasure`,
  paired levels, residual bias zero only when L exact,
  rejected outcome-dependent allocation,
  `Lagrange` optimal counts, Giles 2008/2015 rates
  (https://doi.org/10.1287/opre.1080.0496) + general `MeasureTheory` tower law.
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

/-- Single-level telescope is exact: at `L = 1` the estimate is the
coarse expectation plus the one correction — the finite reason the
two-level ladder (coarse + one refinement) needs no deeper homage:
the telescope bottoms at exactly one correction past the base. Pairs
`mlmc_telescope` (the general form): together the ladder is pinned
at both the single-step and the full-sum ends. -/
theorem mlmc_telescope_single (levels : Nat → Sample → ℝ) (prob : Sample → ℝ) :
    mlmcExpectation levels prob 1 =
    mlmcExpectation levels prob 0 + mlmcDiff levels prob 1 := by
  have h := mlmc_telescope levels prob 1
  simpa using h

theorem mlmc_residual_bias_zero_iff_exact
    (levels : Nat → Sample → ℝ) (prob : Sample → ℝ) (L : Nat) (trueValue : ℝ)
    (hExact : mlmcExpectation levels prob L = trueValue) :
    mlmcExpectation levels prob L - trueValue = 0 := by
  linarith

/-- Exact top level kills the residual: when the finest fidelity IS the
truth, the telescope's bias vanishes — the finite reason the top of the
multi-fidelity ladder must be full-fidelity rollouts, never another
approximation. Pair with `mlmc_missing_correction_bias` (dropping a
live level breaks the estimate): together the ladder is exact iff the
top is exact AND every live correction is kept. -/
theorem mlmc_exact_top_zero_bias
    (levels : Nat → Sample → ℝ) (prob : Sample → ℝ) (L : Nat) (trueValue : ℝ)
    (hExact : mlmcExpectation levels prob L = trueValue) :
    (mlmcExpectation levels prob 0 + ∑ ℓ ∈ Finset.Icc 1 L, mlmcDiff levels prob ℓ)
    - trueValue = 0 := by
  rw [← mlmc_telescope levels prob L, hExact, sub_self]

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

/-- Erasing a ZERO correction preserves the estimate: the mirror of
`mlmc_missing_correction_bias` (which says erasing a nonzero level breaks
it). Together they give the exact pruning rule — drop exactly the levels
whose correction vanishes, keep every other. Failure mode killed: a
pruning pass that cannot tell dead levels from live ones. -/
theorem mlmc_erase_zero_preserves
    (levels : Nat → Sample → ℝ) (prob : Sample → ℝ) (L k : Nat) (hk : 1 ≤ k ∧ k ≤ L)
    (hZero : mlmcDiff levels prob k = 0) :
    (mlmcExpectation levels prob 0 + ∑ ℓ ∈ (Finset.Icc 1 L).erase k, mlmcDiff levels prob ℓ)
    = mlmcExpectation levels prob L := by
  have h_mem : k ∈ Finset.Icc 1 L := Finset.mem_Icc.mpr hk
  have h_sum_erase : ∑ ℓ ∈ (Finset.Icc 1 L).erase k, mlmcDiff levels prob ℓ =
      (∑ ℓ ∈ Finset.Icc 1 L, mlmcDiff levels prob ℓ) - mlmcDiff levels prob k := by
    rw [Finset.sum_erase_eq_sub h_mem]
  have h_tel := mlmc_telescope levels prob L
  rw [h_sum_erase, hZero, sub_zero, h_tel]

/-- Level contributions add linearly: the two-level correction sum splits
    across the level boundary, so coarse and fine corrections compose
    without interaction terms. Practical core: multi-fidelity budgets
    (cheap screening games + expensive full games) add — never double-count
    a correction level when splitting the 5 s search budget. Degenerate
    splits are free (proved below): no guard needed against empty splits. -/
theorem mlmc_level_additive (levels : Nat → Sample → ℝ) (prob : Sample → ℝ)
    (a b c : Nat) (hab : a ≤ b) (hbc : b ≤ c) :
    ∑ ℓ ∈ Finset.Icc a c, mlmcDiff levels prob ℓ
      = (∑ ℓ ∈ Finset.Icc a b, mlmcDiff levels prob ℓ)
      + (∑ ℓ ∈ Finset.Ioc b c, mlmcDiff levels prob ℓ) := by
  have hdisj : Disjoint (Finset.Icc a b) (Finset.Ioc b c) := by
    apply Finset.disjoint_left.mpr
    intro x hx1 hx2
    simp [Finset.mem_Icc] at hx1
    simp [Finset.mem_Ioc] at hx2
    omega
  have hunion : Finset.Icc a b ∪ Finset.Ioc b c = Finset.Icc a c := by
    ext x
    simp [Finset.mem_Icc, Finset.mem_Ioc]
    omega
  rw [← hunion, Finset.sum_union hdisj]

/-- Degenerate split is free: splitting at the left endpoint leaves the
whole sum on one side — the finite reason the budget splitter needs no
guard against empty splits: `Icc a a` plus `Ioc a c` is exactly `Icc a c`.
Chains `mlmc_level_additive` at `b = a`: splitting is safe at ANY
boundary including the degenerate one. -/
theorem mlmc_level_additive_left (levels : Nat → Sample → ℝ) (prob : Sample → ℝ)
    (a c : Nat) (hac : a ≤ c) :
    ∑ ℓ ∈ Finset.Icc a c, mlmcDiff levels prob ℓ
      = (∑ ℓ ∈ Finset.Icc a a, mlmcDiff levels prob ℓ)
      + (∑ ℓ ∈ Finset.Ioc a c, mlmcDiff levels prob ℓ) :=
  mlmc_level_additive levels prob a a c (le_refl a) hac

/-- Degenerate split is free (right): splitting at the right endpoint
leaves the whole sum on one side — the mirror of
`mlmc_level_additive_left` (degenerate at the left). Together splits
are safe at BOTH degenerate boundaries, so the budget splitter needs
no guard in either direction: `Ioc c c` is empty and the sum stands. -/
theorem mlmc_level_additive_right (levels : Nat → Sample → ℝ) (prob : Sample → ℝ)
    (a c : Nat) (hac : a ≤ c) :
    ∑ ℓ ∈ Finset.Icc a c, mlmcDiff levels prob ℓ
      = (∑ ℓ ∈ Finset.Icc a c, mlmcDiff levels prob ℓ)
      + (∑ ℓ ∈ Finset.Ioc c c, mlmcDiff levels prob ℓ) :=
  mlmc_level_additive levels prob a c c hac (le_refl c)

/-- Level-0 correction is vacuous: `E₀ − E₀ = 0` since `0 − 1 = 0` in Nat.
Failure mode killed: budgeting (or double-counting) a correction at level
0 — the coarsest level's contribution is its expectation alone, and the
telescope's correction sum over `Icc 1 L` correctly excludes it. -/
theorem mlmcDiff_zero (levels : Nat → Sample → ℝ) (prob : Sample → ℝ) :
    mlmcDiff levels prob 0 = 0 := by
  unfold mlmcDiff
  simp

/-- Identical successive levels correct nothing: when `levels (l+1)` equals
`levels l` pointwise, the correction is zero — the finite reason adaptive
refinement stops where the emulator stops changing: no signal difference,
no correction, no budget spent. Pairs `mlmcDiff_zero` (level 0 vacuous):
together both degenerate refinements vanish. -/
theorem mlmcDiff_eq_zero_of_succ_eq (levels : Nat → Sample → ℝ)
    (prob : Sample → ℝ) (l : ℕ)
    (h : ∀ s, levels (l + 1) s = levels l s) :
    mlmcDiff levels prob (l + 1) = 0 := by
  have hexp : mlmcExpectation levels prob (l + 1)
      = mlmcExpectation levels prob l := by
    unfold mlmcExpectation
    apply Finset.sum_congr rfl
    intro s _
    rw [h s]
  unfold mlmcDiff
  have hsub : l + 1 - 1 = l := by omega
  rw [hsub, hexp, sub_self]

/-- Identical levels correct nothing: when `E_ℓ = E_{ℓ-1}` the diff
vanishes, so the level is prunable by `mlmc_erase_zero_preserves`. Pair
with `mlmcDiff_zero` (level 0 always vacuous): together the only levels
worth budgeting are those where the fidelity step moves the expectation
— the finite reason multi-fidelity budgets pay per level only on
measured correction. -/
theorem mlmcDiff_eq_zero_of_levels_eq (levels : Nat → Sample → ℝ) (prob : Sample → ℝ)
    (ℓ : Nat) (h : mlmcExpectation levels prob ℓ = mlmcExpectation levels prob (ℓ - 1)) :
    mlmcDiff levels prob ℓ = 0 := by
  unfold mlmcDiff
  rw [h, sub_self]

/-- Erasing two dead levels preserves the estimate: the single-erase rule
`mlmc_erase_zero_preserves` composes — drop a second vanishing level and
the telescope still lands. Practical core: the pruning pass can sweep
all dead levels in one pass, not one-at-a-time with re-verification
between — the finite reason the multi-fidelity budget loop prunes the
whole dead set per iteration. -/
theorem mlmc_erase_two_zero_preserves
    (levels : Nat → Sample → ℝ) (prob : Sample → ℝ) (L k₁ k₂ : Nat)
    (hk : 1 ≤ k₁ ∧ k₁ ≤ L) (hk2 : 1 ≤ k₂ ∧ k₂ ≤ L) (hne : k₁ ≠ k₂)
    (hZ1 : mlmcDiff levels prob k₁ = 0) (hZ2 : mlmcDiff levels prob k₂ = 0) :
    (mlmcExpectation levels prob 0
      + ∑ ℓ ∈ (Finset.Icc 1 L).erase k₁ |>.erase k₂, mlmcDiff levels prob ℓ)
    = mlmcExpectation levels prob L := by
  have h_mem1 : k₁ ∈ Finset.Icc 1 L := Finset.mem_Icc.mpr hk
  have h_mem2 : k₂ ∈ (Finset.Icc 1 L).erase k₁ := by
    rw [Finset.mem_erase]
    exact ⟨Ne.symm hne, Finset.mem_Icc.mpr hk2⟩
  have h1 := mlmc_telescope levels prob L
  have hsub : ∑ ℓ ∈ (Finset.Icc 1 L).erase k₁ |>.erase k₂, mlmcDiff levels prob ℓ
      = (∑ ℓ ∈ Finset.Icc 1 L, mlmcDiff levels prob ℓ)
        - mlmcDiff levels prob k₁ - mlmcDiff levels prob k₂ := by
    rw [Finset.sum_erase_eq_sub h_mem2, Finset.sum_erase_eq_sub h_mem1]
  rw [hsub, hZ1, hZ2, sub_zero, sub_zero, h1]

/-- Erasing a live level costs exactly its correction: dropping a NONZERO
correction shifts the estimate by precisely that correction — the mirror
of `mlmc_erase_two_zero_preserves` (dead levels drop free). Together the
pruning pass prices every drop (zero or live), so the budget loop never
drops a level without booking its cost. -/
theorem mlmc_erase_live_costs_correction
    (levels : Nat → Sample → ℝ) (prob : Sample → ℝ) (L k : Nat)
    (hk : k ∈ Finset.Icc 1 L) :
    (mlmcExpectation levels prob 0
      + ∑ ℓ ∈ (Finset.Icc 1 L).erase k, mlmcDiff levels prob ℓ)
    = mlmcExpectation levels prob L - mlmcDiff levels prob k := by
  have h1 := mlmc_telescope levels prob L
  have hsub : ∑ ℓ ∈ (Finset.Icc 1 L).erase k, mlmcDiff levels prob ℓ
      = (∑ ℓ ∈ Finset.Icc 1 L, mlmcDiff levels prob ℓ)
        - mlmcDiff levels prob k :=
    Finset.sum_erase_eq_sub hk
  rw [hsub]
  linarith

end Telescoping

section BudgetOrder

variable {Sample : Type} [Fintype Sample] [DecidableEq Sample]

/-- Budget increases never shrink the ladder count: more levels cost at
least as many corrections — the finite reason the multi-fidelity budget
loop's level sweep is monotone in L. Pair with
`mlmc_erase_two_zero_preserves` (dead levels drop free): together the
loop adds levels monotonically and prunes dead ones freely, so the
ladder only ever grows where corrections are live. -/
theorem mlmc_ladder_mono_levels (levels : Nat → Sample → ℝ) (prob : Sample → ℝ)
    (L₁ L₂ : Nat) (h : L₁ ≤ L₂) :
    (Finset.Icc 1 L₁).card ≤ (Finset.Icc 1 L₂).card := by
  apply Finset.card_le_card
  intro x hx
  simp [Finset.mem_Icc] at hx ⊢
  omega

/-- Strict version of `mlmc_ladder_mono_levels`: strictly more levels
mean strictly more corrections — the finite reason the budget loop
cannot add a level for free: every new level is a new correction to
price, so level-count sweeps translate one-to-one into budget steps
with no flat stretches. The witness is the top level itself: `L₂` sits
in its own range but above `L₁`'s. -/
theorem mlmc_ladder_strict_mono_levels (levels : Nat → Sample → ℝ) (prob : Sample → ℝ)
    (L₁ L₂ : Nat) (h : L₁ < L₂) :
    (Finset.Icc 1 L₁).card < (Finset.Icc 1 L₂).card := by
  apply Finset.card_lt_card
  have hsub : Finset.Icc 1 L₁ ⊆ Finset.Icc 1 L₂ := by
    intro x hx
    simp [Finset.mem_Icc] at hx ⊢
    omega
  have hmem : L₂ ∈ Finset.Icc 1 L₂ := by
    simp [Finset.mem_Icc]
    omega
  have hne : Finset.Icc 1 L₁ ≠ Finset.Icc 1 L₂ := by
    intro heq
    have hcon : L₂ ∈ Finset.Icc 1 L₁ := heq ▸ hmem
    simp [Finset.mem_Icc] at hcon
    omega
  exact Finset.ssubset_iff_subset_ne.mpr ⟨hsub, hne⟩

end BudgetOrder

end Hydra2.Blueprint.Modules.MLMC
