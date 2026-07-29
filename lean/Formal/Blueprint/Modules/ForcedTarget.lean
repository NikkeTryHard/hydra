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

/-! # Hydra2 forced playouts + policy-target pruning (KataGo)

Mirrors `ideas/forced-target-pruning.md` (Wu 1902.10565 §3.2): forced-visit
count `n_forced(c) = √(k·P(c)·ΣN)` with `k = 2` decouples exploration from the
policy target — deep-narrow lines get evaluated for TARGET QUALITY while
pruning keeps that mass out of the teacher (numerator AND denominator).
Ablation NoForcedTP 1276 vs 1329 = 1.25x training-time factor.

Finite core below: `nForced` def + square identity + monotonicity in search
mass, plus the pruned-target distribution invariant. Dirichlet noise, FPU,
gating config, Elo stay harness-side.
-/

namespace Hydra2.Blueprint.ForcedTarget

section ForcedVisits

noncomputable def nForced (k P sumN : ℝ) : ℝ :=
  Real.sqrt (k * P * sumN)

theorem nForced_sq (k P sumN : ℝ) (hk : 0 ≤ k) (hP : 0 ≤ P) (hS : 0 ≤ sumN) :
    (nForced k P sumN) ^ 2 = k * P * sumN := by
  unfold nForced
  exact Real.sq_sqrt (mul_nonneg (mul_nonneg hk hP) hS)

theorem nForced_mono_sumN (k P S1 S2 : ℝ) (hk : 0 ≤ k) (hP : 0 ≤ P)
    (h : S1 ≤ S2) :
    nForced k P S1 ≤ nForced k P S2 := by
  unfold nForced
  apply Real.sqrt_le_sqrt
  exact mul_le_mul_of_nonneg_left h (mul_nonneg hk hP)

end ForcedVisits

section PrunedTarget

variable {A : Type} [Fintype A] [DecidableEq A]

/-- Pruned policy target: visit distribution over the UNPRUNED mass `N'`
(`N'(c) = N(c) - sub(c)` with forced visits subtracted, singletons dropped).
`π(c) = N'(c) / Σ N'` sums to one — pruned mass never corrupts the teacher. -/
noncomputable def prunedTarget (acts : Finset A) (N' : A → ℝ) : A → ℝ :=
  fun c => N' c / ∑ a ∈ acts, N' a

theorem prunedTarget_sum_one (acts : Finset A) (N' : A → ℝ)
    (hne : (∑ a ∈ acts, N' a) ≠ 0) :
    ∑ c ∈ acts, prunedTarget acts N' c = 1 := by
  unfold prunedTarget
  rw [← Finset.sum_div]
  exact div_self hne

end PrunedTarget

section PUCTPrune

variable {A : Type} [DecidableEq A]

/-- KataGo PUCT value with exploration floor
(`Wu 1902.10565 §2`: `c_PUCT = 1.1`, `FPU V(c) = V(n) - c_FPU·√P_explored`
with `c_FPU = 0.2`). `N` is the visit count; forced visits are enforced by
treating `PUCT(c) = ∞` while `N(c) < n_forced(c)` (harness scheduler). -/
noncomputable def puct (V P sumN cPUCT : ℝ) (N : ℕ) : ℝ :=
  V + cPUCT * P * Real.sqrt sumN / (1 + (N : ℝ))

/-- PUCT is antitone in visits: more visits can only lower the exploration
bonus (numerator nonneg). Pruning (which removes visits) therefore raises a
child's PUCT — hence the KataGo rule subtracts only while the pruned child
stays below the best at FINAL utilities (`PruneOK` below). -/
theorem puct_antitone_N (V P sumN cPUCT : ℝ) (hc : 0 ≤ cPUCT) (hP : 0 ≤ P)
    (hS : 0 ≤ sumN) {N1 N2 : ℕ} (h : N1 ≤ N2) :
    puct V P sumN cPUCT N2 ≤ puct V P sumN cPUCT N1 := by
  unfold puct
  have hnum : (0 : ℝ) ≤ cPUCT * P * Real.sqrt sumN :=
    mul_nonneg (mul_nonneg hc hP) (Real.sqrt_nonneg _)
  have hden : (1 : ℝ) + (N1 : ℝ) ≤ 1 + (N2 : ℝ) := by
    have hcast : (N1 : ℝ) ≤ (N2 : ℝ) := Nat.cast_le.mpr h
    linarith
  have hpos : (0 : ℝ) < 1 + (N1 : ℝ) := by
    have hnn : (0 : ℝ) ≤ (N1 : ℝ) := Nat.cast_nonneg _
    linarith
  have hdiv := div_le_div_of_nonneg_left hnum hpos hden
  linarith

/-- Boundary conditions: no prior mass, no search mass, or zero constant
forces zero — nothing to explore from nothing. -/
theorem nForced_zero_sumN (k P : ℝ) : nForced k P 0 = 0 := by
  unfold nForced
  have h0 : k * P * (0 : ℝ) = 0 := by ring
  rw [h0, Real.sqrt_zero]

theorem nForced_zero_prior (k sumN : ℝ) : nForced k 0 sumN = 0 := by
  unfold nForced
  have h0 : k * (0 : ℝ) * sumN = 0 := by ring
  rw [h0, Real.sqrt_zero]

theorem nForced_zero_k (P sumN : ℝ) : nForced 0 P sumN = 0 := by
  unfold nForced
  have h0 : (0 : ℝ) * P * sumN = 0 := by ring
  rw [h0, Real.sqrt_zero]

/-- KataGo prune rule (`Wu §3.2` + katac4 re-derivation): the best child keeps
all visits; from each other child subtract up to `n_forced` (`sub c ≤ N c`,
`(N c - sub c) ≤ nF c`) so long as its PUCT stays below the best at FINAL
utilities; outright drop singletons (`N' c ≠ 1`). -/
def PruneOK (acts : Finset A) (cStar : A) (N N' sub : A → ℕ)
    (Vfin P : A → ℝ) (sumN cPUCT : ℝ) (nF : A → ℝ) : Prop :=
  N' cStar = N cStar ∧ ∀ c ∈ acts, c ≠ cStar →
    sub c ≤ N c ∧ (N c : ℝ) - (sub c : ℝ) ≤ nF c ∧ N' c = N c - sub c ∧
    N' c ≠ 1 ∧
    puct (Vfin c) (P c) sumN cPUCT (N' c)
      < puct (Vfin cStar) (P cStar) sumN cPUCT (N cStar)

/-- Pruning only removes visits from non-best children. -/
theorem prune_mono_visits (acts : Finset A) (cStar : A) (N N' sub : A → ℕ)
    (Vfin P : A → ℝ) (sumN cPUCT : ℝ) (nF : A → ℝ)
    (h : PruneOK acts cStar N N' sub Vfin P sumN cPUCT nF)
    (c : A) (hc : c ∈ acts) (hne : c ≠ cStar) : N' c ≤ N c := by
  have heq := (h.2 c hc hne).2.2.1
  rw [heq]
  exact Nat.sub_le _ _

/-- The best child is untouched by pruning. -/
theorem prune_best_untouched (acts : Finset A) (cStar : A) (N N' sub : A → ℕ)
    (Vfin P : A → ℝ) (sumN cPUCT : ℝ) (nF : A → ℝ)
    (h : PruneOK acts cStar N N' sub Vfin P sumN cPUCT nF) :
    N' cStar = N cStar :=
  h.1

end PUCTPrune

end Hydra2.Blueprint.ForcedTarget
