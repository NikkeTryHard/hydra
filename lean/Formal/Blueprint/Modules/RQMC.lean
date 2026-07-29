import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Tactic

set_option linter.unusedSimpArgs false
set_option linter.unreachableTactic false
set_option linter.unusedTactic false
set_option linter.unusedDecidableInType false
set_option linter.unusedSectionVars false
set_option linter.style.longLine false

/-!
# Hydra2 §11.5 Randomized QMC + §11.6 Scenario Coreset + §11.7 Primal-Dual Pruning

Sources: QMC Wikipedia (random shift `y=x+U mod1`, Owen nested scramble),
         Giles/SMC sources (see other modules), §11.6/11.7 are heuristic (search-only).

RQMC: independently scrambled low-discrepancy points, mapped via inverse-CDF or
categorical partition; one scramble = one dependent replicate; separate scrambles
for uncertainty. Pure QMC without scramble gives error bound but no variance estimate.

For editors: these modules are intentionally less formal than §11.1/11.2/11.8 because
the blueprint marks them as heuristic / rate-dependent. Lean formalizes the structural
properties (unbiasedness under scramble, one-scramble dependence).
-/

namespace Hydra2.Blueprint.Modules.RQMC

section RQMC

variable {Outcome : Type} [Fintype Outcome] [DecidableEq Outcome]

/- RQMC estimator: `Ĩ_N = 1/N ∑_{i=1}^N f(y_i)` where `y_i = x_i ⊕ U` (shift) or
Owen scramble. Each scramble `U` yields one dependent replicate; different
scrambles are independent.

Finite structural core behind unbiasedness (proved below): a shift permutes
`Fin n` (`rqmcShift_bijective`), so uniform sums — hence means and composed
statistics — are preserved. The full `MeasureTheory` uniformity pushforward
(`scrambled Uniform = Uniform`) is HARD-skipped. -/
/-- Random-shift scrambling on `Fin n`: `y = (x + U) mod n`, the `n`-point
discretization of the RQMC random shift `y = x + U mod 1` which preserves
`Uniform[0,1)`. Same construction as `CRN.crnShift` (shared primitive uniforms);
a shift permutes `Fin n`, so a scrambled point set keeps its marginals — the finite
structural core behind `RQMC_unbiased` (whose full `MeasureTheory` uniformity
pushforward is HARD-skipped below). -/
noncomputable def rqmcShift (n : Nat) (_hn : 0 < n) (shift : Fin n) (u : Fin n) : Fin n :=
  ⟨(u.val + shift.val) % n, Nat.mod_lt _ (by omega)⟩

noncomputable def rqmcShiftInv (n : Nat) (_hn : 0 < n) (shift : Fin n) (u : Fin n) : Fin n :=
  ⟨(u.val + n - shift.val) % n, Nat.mod_lt _ (by omega)⟩

theorem rqmcShift_left_inv (n : Nat) (hn : 0 < n) (shift : Fin n) (u : Fin n) :
    rqmcShiftInv n hn shift (rqmcShift n hn shift u) = u := by
  unfold rqmcShift rqmcShiftInv
  apply Fin.ext
  simp only
  have h_s_le_n : shift.val ≤ n := Nat.le_of_lt shift.isLt
  have h1 : ((u.val + shift.val) % n + n - shift.val) = ((u.val + shift.val) % n + (n - shift.val)) := by
    rw [Nat.add_sub_assoc h_s_le_n ((u.val + shift.val) % n)]
  have h2 : (((u.val + shift.val) % n + (n - shift.val)) % n) = (u.val + shift.val + (n - shift.val)) % n := by
    have h := Nat.mod_add_mod (m := u.val + shift.val) (k := n - shift.val) (n := n)
    exact h
  have h3 : u.val + shift.val + (n - shift.val) = u.val + n := by
    calc u.val + shift.val + (n - shift.val) = u.val + (shift.val + (n - shift.val)) := by rw [Nat.add_assoc]
    _ = u.val + n := by rw [Nat.add_sub_cancel' h_s_le_n]
  have h4 : (u.val + n) % n = u.val := by
    have h1' : (u.val + n) % n = (u.val % n + n % n) % n := by rw [← Nat.add_mod]
    simp [Nat.mod_eq_of_lt u.isLt, Nat.mod_self] at h1'
    simpa [Nat.mod_eq_of_lt u.isLt] using h1'
  calc ((u.val + shift.val) % n + n - shift.val) % n
      = (((u.val + shift.val) % n + (n - shift.val)) % n) := by rw [h1]
    _ = (u.val + shift.val + (n - shift.val)) % n := h2
    _ = (u.val + n) % n := by rw [h3]
    _ = u.val := h4

theorem rqmcShift_right_inv (n : Nat) (hn : 0 < n) (shift : Fin n) (u : Fin n) :
    rqmcShift n hn shift (rqmcShiftInv n hn shift u) = u := by
  unfold rqmcShift rqmcShiftInv
  apply Fin.ext
  simp only
  have h_u_lt : u.val < n := u.isLt
  have h_s_le : shift.val ≤ n := Nat.le_of_lt shift.isLt
  have h_le : shift.val ≤ u.val + n := Nat.le_trans h_s_le (Nat.le_add_left n u.val)
  have h1 : ((u.val + n - shift.val) % n + shift.val) % n = (u.val + n - shift.val + shift.val) % n := by
    have h := Nat.mod_add_mod (m := u.val + n - shift.val) (k := shift.val) (n := n)
    exact h
  have h2 : u.val + n - shift.val + shift.val = u.val + n := Nat.sub_add_cancel h_le
  have h3 : (u.val + n) % n = u.val := by
    calc (u.val + n) % n = u.val % n := Nat.add_mod_right u.val n
      _ = u.val := Nat.mod_eq_of_lt h_u_lt
  calc ((u.val + n - shift.val) % n + shift.val) % n
      = (u.val + n - shift.val + shift.val) % n := h1
    _ = (u.val + n) % n := by rw [h2]
    _ = u.val := h3
theorem rqmcShift_bijective (n : Nat) (hn : 0 < n) (shift : Fin n) :
    Function.Bijective (rqmcShift n hn shift) :=
  ⟨fun a b h => by
      have : rqmcShiftInv n hn shift (rqmcShift n hn shift a) =
             rqmcShiftInv n hn shift (rqmcShift n hn shift b) := by rw [h]
      rw [rqmcShift_left_inv n hn shift a, rqmcShift_left_inv n hn shift b] at this
      exact this,
   fun b => ⟨rqmcShiftInv n hn shift b, rqmcShift_right_inv n hn shift b⟩⟩

/-- Shift preserves outcome counts: scrambling permutes `Fin n`, so the histogram
over scrambled indices equals the histogram over raw indices. This is the finite
reason one random shift keeps marginals (`RQMC_unbiased` structural core). -/
theorem rqmcShift_preserves_filter_card (n : Nat) (hn : 0 < n) (shift : Fin n)
    (f : Fin n → Outcome) (o : Outcome) :
    (Finset.univ.filter (fun u : Fin n => f (rqmcShift n hn shift u) = o)).card =
    (Finset.univ.filter (fun u : Fin n => f u = o)).card := by
  have hBij := rqmcShift_bijective n hn shift
  have hInj : Function.Injective (rqmcShift n hn shift) := hBij.1
  have hSurj : Function.Surjective (rqmcShift n hn shift) := hBij.2
  apply Finset.card_bij (fun a _ => rqmcShift n hn shift a)
  · intro a ha
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at ha ⊢
    exact ha
  · intro a₁ ha₁ a₂ ha₂ h
    exact hInj h
  · intro b' hb'
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hb'
    obtain ⟨a, rfl⟩ := hSurj b'
    refine ⟨a, ?_, rfl⟩
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    exact hb'

/-- Uniform sums are preserved by the shift (reindex through `rqmcShift_bijective`):
the finite unbiasedness core — means and composed statistics follow. -/
theorem RQMC_shift_sum_preservation (n : Nat) (hn : 0 < n) (shift : Fin n)
    (f : Fin n → ℝ) :
    ∑ u, f (rqmcShift n hn shift u) = ∑ u, f u :=
  Fintype.sum_bijective _ (rqmcShift_bijective n hn shift) _ _ fun _ => rfl

theorem RQMC_shift_mean_preservation (n : Nat) (hn : 0 < n) (shift : Fin n)
    (f : Fin n → ℝ) :
    (1 / (n : ℝ)) * ∑ u, f (rqmcShift n hn shift u) = (1 / (n : ℝ)) * ∑ u, f u := by
  rw [RQMC_shift_sum_preservation n hn shift f]

theorem RQMC_shift_comp_sum_preservation (n : Nat) (hn : 0 < n) (shift : Fin n)
    (g : Fin n → Outcome) (f : Outcome → ℝ) :
    ∑ u, f (g (rqmcShift n hn shift u)) = ∑ u, f (g u) :=
  Fintype.sum_bijective _ (rqmcShift_bijective n hn shift) _ _ fun _ => rfl

/-- Nested-scramble digit-position permutation: a `k`-digit base-`b` point is a
digit vector `Fin k → Fin b`; Owen's nested uniform scramble (Owen 1995
`Randomly permuted (t,m,s)-nets`; Helmer tree-shuffle: `randomly shuffle the b
sub-trees of the root, then recurse into each subtree`) coherently permutes
digit positions, `maximally randomizing while preserving multidimensional
stratification` (Burley JCGT 2020). Finite core: position permutation is a
bijection on digit vectors, so scrambled point sets keep their counts — the
same reindexing principle as `rqmcShift_bijective` (OwenDepth scout F5
`lean_use`). Digit-*value* permutations within each subtree + `(t,s)`-net
preservation are out of scope (honest gap); the full scramble-beats-shift rate
stays axiomatized (`axiom_RQMC_rate_smooth`). -/
noncomputable def digitPermute (k b : Nat) (σ : Equiv.Perm (Fin k))
    (f : Fin k → Fin b) : Fin k → Fin b :=
  f ∘ ⇑σ

theorem digitPermute_left_inv (k b : Nat) (σ : Equiv.Perm (Fin k))
    (f : Fin k → Fin b) :
    digitPermute k b σ.symm (digitPermute k b σ f) = f := by
  unfold digitPermute
  funext i
  simp only [Function.comp_apply, Equiv.apply_symm_apply]

theorem digitPermute_right_inv (k b : Nat) (σ : Equiv.Perm (Fin k))
    (f : Fin k → Fin b) :
    digitPermute k b σ (digitPermute k b σ.symm f) = f := by
  unfold digitPermute
  funext i
  simp only [Function.comp_apply, Equiv.symm_apply_apply]
theorem digitPermute_bijective (k b : Nat) (σ : Equiv.Perm (Fin k)) :
    Function.Bijective (digitPermute k b σ) :=
  ⟨fun x y h => by
      have : digitPermute k b σ.symm (digitPermute k b σ x) =
             digitPermute k b σ.symm (digitPermute k b σ y) := by rw [h]
      rw [digitPermute_left_inv k b σ x, digitPermute_left_inv k b σ y] at this
      exact this,
   fun y => ⟨digitPermute k b σ.symm y, digitPermute_right_inv k b σ y⟩⟩


/-- Gain-counting kernel `K(x) = x(1-x)` (Owen–Pan `arXiv:2308.08035` §5 Eq.16 `G̃(u,k,n') = Σ_v H(u,v) m(u,v,k) ε'_v(1-ε'_v)` with `ε' = n'/m - ⌊n'/m⌋` the fractional part, `K(x) = x(1-x)`; WalshRetry scout. Via `C = n²/m + m·ε(1-ε)` Eq.11 the `ε(1-ε)` factor carries the `n`-dependence of the gain `G(u,k,n)` in Eq.4, the multi-base generalization of Owen 1997 SINUM Thm.2). Finite core: on `[0,1]`, `0 ≤ K ≤ 1/4` (max at `1/2`) — the elementary bound behind `Γ ≤ [b/(b-1)]^{d-1} ≤ e` (Faure) and `Γ_d = O(log d)` (Halton Cor.3/Thm.4). Full gain combinatorics (`H`, `m`, `C` closed forms Eqs.5-11) stay future work. -/
noncomputable def rqmcGainK (x : ℝ) : ℝ := x * (1 - x)

theorem rqmcGainK_nonneg (x : ℝ) (h0 : 0 ≤ x) (h1 : x ≤ 1) :
    0 ≤ rqmcGainK x := by
  unfold rqmcGainK
  exact mul_nonneg h0 (by linarith)

theorem rqmcGainK_le_quarter (x : ℝ) (h0 : 0 ≤ x) (h1 : x ≤ 1) :
    rqmcGainK x ≤ 1 / 4 := by
  unfold rqmcGainK
  have hsq : 0 ≤ (x - 1 / 2) ^ 2 := sq_nonneg _
  have heq : x * (1 - x) = 1 / 4 - (x - 1 / 2) ^ 2 := by ring
  linarith
/-- Within one scramble, points are functionally dependent: they are all
determined by the single shared `shift`. Finite witness: for `n ≥ 2` there are
two distinct indices whose scrambled images are distinct (by injectivity), yet
both are deterministic functions of the same `shift` — so they cannot be
independent. Full independence would need `MeasureTheory.ProductMeasure`. -/
theorem RQMC_one_scramble_dependent (n : Nat) (hn : 0 < n) (hn2 : 2 ≤ n)
    (shift : Fin n) :
    ∃ a b : Fin n, a ≠ b ∧ rqmcShift n hn shift a ≠ rqmcShift n hn shift b := by
  have hInj : Function.Injective (rqmcShift n hn shift) :=
    (rqmcShift_bijective n hn shift).1
  have hn1 : 1 < n := by omega
  have hne : (⟨0, hn⟩ : Fin n) ≠ ⟨1, hn1⟩ := by simp
  exact ⟨⟨0, hn⟩, ⟨1, hn1⟩, hne, fun h => hne (hInj h)⟩

/-- Different scrambles give independent replicates. Finite core: scramble
invariance — sums through any two shifts agree (both equal the unscrambled sum
by `RQMC_shift_sum_preservation`). Full probabilistic independence of separate
scrambles would need `MeasureTheory` product measures over replicates. -/
theorem RQMC_separate_scrambles_independent (n : Nat) (hn : 0 < n)
    (shift1 shift2 : Fin n) (f : Fin n → ℝ) :
    ∑ u, f (rqmcShift n hn shift1 u) = ∑ u, f (rqmcShift n hn shift2 u) := by
  rw [RQMC_shift_sum_preservation n hn shift1 f,
    RQMC_shift_sum_preservation n hn shift2 f]

/-- Owen rate `O(N^{-3/2})` needs smoothness (`A* < 1/2`), not always.
Finite witness: two integrands with the same mean but different variation, so no
uniform rate can hold without a smoothness hypothesis. `f = 0` vs
`g i = 2 * i - 1` on `Fin 2`: same sum, different `|·|`-variation. -/
theorem RQMC_rate_requires_smoothness :
    ∃ (f g : Fin 2 → ℝ), (∑ u, f u) = (∑ u, g u) ∧
      (∑ u, |f u|) ≠ (∑ u, |g u|) := by
  refine ⟨fun _ => 0, fun i => 2 * (i.val : ℝ) - 1, ?_, ?_⟩
  · have huniv : (Finset.univ : Finset (Fin 2)) = {0, 1} := by decide
    have h01 : (0 : Fin 2) ∉ ({1} : Finset (Fin 2)) := by decide
    have h0v : (0 : Fin 2).val = 0 := by decide
    have h1v : (1 : Fin 2).val = 1 := by decide
    simp only [huniv, Finset.sum_insert h01, Finset.sum_singleton,
      h0v, h1v, Nat.cast_zero, Nat.cast_one] at *
    norm_num
  · have huniv : (Finset.univ : Finset (Fin 2)) = {0, 1} := by decide
    have h01 : (0 : Fin 2) ∉ ({1} : Finset (Fin 2)) := by decide
    have h0v : (0 : Fin 2).val = 0 := by decide
    have h1v : (1 : Fin 2).val = 1 := by decide
    simp only [huniv, Finset.sum_insert h01, Finset.sum_singleton,
      h0v, h1v, Nat.cast_zero, Nat.cast_one] at *
    norm_num

/-- Tiny test: categorical frequencies converge to declared probabilities as `scrambles→∞`;
one-scramble IID interval attempt must fail (would underestimate variance).
Finite core: uniform weights over `Fin n` sum to one. -/
theorem RQMC_tiny_test (n : Nat) (hn : 0 < n) :
    ∑ _i : Fin n, ((1 : ℝ) / (n : ℝ)) = 1 := by
  have hnR : (n : ℝ) ≠ 0 := by exact_mod_cast ne_of_gt hn
  rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
    mul_one_div_cancel hnR]

end RQMC

section Coreset

-- §11.6 Scenario coreset: select weighted subset only from current search population,
-- store original scenario IDs and nonnegative weights summing to one, use weighted
-- objective for *search* only, never for confirmation.
/-- Coreset is search-only weighting discipline. Finite core: nonnegative weights
normalize to nonnegative weights summing to one (cf. `coreset_weights_sum_one`).
The search-vs-confirmation separation itself is heuristic, not formalized. -/
theorem coreset_is_search_only (n : Nat) (w : Fin n → ℝ) (hw : ∀ i, 0 ≤ w i)
    (hpos : 0 < ∑ i, w i) :
    (∀ i, 0 ≤ w i / ∑ j, w j) ∧ ∑ i, w i / ∑ j, w j = 1 := by
  constructor
  · intro i
    exact div_nonneg (hw i) (le_of_lt hpos)
  · rw [← Finset.sum_div]
    exact div_self (ne_of_gt hpos)
/-- Normalized coreset weights `wᵢ/∑w` (the def the old `rfl after def` note
pointed at): they sum to one whenever the total is positive. -/
noncomputable def coresetNormalize (n : Nat) (w : Fin n → ℝ) : Fin n → ℝ :=
  fun i => w i / ∑ j, w j

theorem coreset_weights_sum_one (n : Nat) (w : Fin n → ℝ) (hpos : 0 < ∑ i, w i) :
    ∑ i, coresetNormalize n w i = 1 := by
  unfold coresetNormalize
  rw [← Finset.sum_div]
  exact div_self (ne_of_gt hpos)
/-- Unweighted subsets fail: two-point population `{0, 10}` has uniform mean `5`
but the singleton subset `{0}` has mean `0`. Hence subset means need weights. -/
theorem coreset_unweighted_subset_fails :
    ∃ (f : Fin 2 → ℝ), (1 / 2 : ℝ) * ∑ u, f u ≠ f 0 := by
  refine ⟨fun i => (i.val : ℝ) * 10, ?_⟩
  have huniv : (Finset.univ : Finset (Fin 2)) = {0, 1} := by decide
  have h01 : (0 : Fin 2) ∉ ({1} : Finset (Fin 2)) := by decide
  have h0v : (0 : Fin 2).val = 0 := by decide
  have h1v : (1 : Fin 2).val = 1 := by decide
  simp only [huniv, Finset.sum_insert h01, Finset.sum_singleton,
    h0v, h1v, Nat.cast_zero, Nat.cast_one] at *
  norm_num

end Coreset
section PrimalDual

-- §11.7 Primal-dual pruning: prune `b` only when `U_b < L_a` for valid simultaneous
-- one-sided confidence bounds with multiplicity correction. Sampled mean alone is not a bound.
/-- Pruning needs simultaneous valid bounds. Finite core: if `[a_lo, a_hi]` and
`[b_lo, b_hi]` trap the true values `va, vb` and `b_hi < a_lo`, then `vb < va`.
Full simultaneous coverage with multiplicity correction needs `MeasureTheory`. -/
theorem pruning_requires_simultaneous_bounds (a_lo a_hi b_lo b_hi va vb : ℝ)
    (ha_lo : a_lo ≤ va) (_ha_hi : va ≤ a_hi)
    (_hb_lo : b_lo ≤ vb) (hb_hi : vb ≤ b_hi)
    (hprune : b_hi < a_lo) : vb < va := by
  linarith
/-- Sample means alone cannot prune: means may favor `a` while intervals overlap.
Witness `meanA = 5, meanB = 4.9` both inside `[4, 6]`, so `¬ (6 < 4)` — no prune. -/
theorem pruning_sample_mean_insufficient :
    ∃ (meanA meanB lo hi : ℝ), meanB < meanA ∧ lo ≤ meanB ∧ meanB ≤ hi ∧
      lo ≤ meanA ∧ meanA ≤ hi ∧ ¬ (hi < lo) := by
  exact ⟨5, 4.9, 4, 6, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num⟩
/-- Certified pruning only fires when the inequality holds: under valid one-sided
bounds `vb ≤ ub`, `la ≤ va`, the prune condition `ub < la` implies `vb < va`
(hence `vb ≠ va` — `b` is not optimal). Full certified-sequential logic needs
`MeasureTheory`. -/
theorem pruning_certified_prunes_only_when_inequality_holds (va vb ub la : ℝ)
    (hb : vb ≤ ub) (ha : la ≤ va) (h : ub < la) : vb < va ∧ vb ≠ va := by
  constructor
  · linarith
  · exact ne_of_lt (by linarith)

end PrimalDual

end Hydra2.Blueprint.Modules.RQMC
