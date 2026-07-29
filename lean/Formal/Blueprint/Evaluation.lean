import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.Finset.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Data.Real.Basic
import Mathlib.Tactic

set_option linter.unusedSimpArgs false
set_option linter.unreachableTactic false
set_option linter.unusedTactic false
set_option linter.unusedSectionVars false
set_option linter.style.longLine false
set_option linter.style.whitespace false

/-! # Hydra2 P11 Evaluation Block Independence & Fixed-N Power Formula
Mirrors `ALGORITHM_EXPERIMENT_BLUEPRINT.md` § evaluation: block-independent
scramble/wall-block averaging and the fixed-N power calculation for
promotion gates.
-/

namespace Hydra2.Blueprint.Evaluation

section BlockIndependence

variable {Block : Type} [Fintype Block] [DecidableEq Block]

/-- An evaluation block aggregates a disjoint wall/scramble unit.
    `value b` is the block-level statistic (e.g. mean placement). -/
structure BlockStat (Block : Type) where
  value : Block → ℝ

noncomputable def blockMean (s : BlockStat Block) : ℝ :=
  (∑ b : Block, s.value b) / (Fintype.card Block : ℝ)

theorem blockMean_eq_sum_div_card (s : BlockStat Block) :
    blockMean s = (∑ b : Block, s.value b) / (Fintype.card Block : ℝ) := rfl

/-- Fixed-N variance of the block mean under i.i.d. blocks: Var(mean) = Var(block)/N. -/
noncomputable def blockMeanVariance (varBlock : ℝ) (n : ℕ) (hn : 0 < n) : ℝ :=
  varBlock / (n : ℝ)

theorem blockMeanVariance_pos_of_var_pos (varBlock : ℝ) (n : ℕ) (hn : 0 < n) (hvar : 0 < varBlock) :
    0 < blockMeanVariance varBlock n hn := by
  unfold blockMeanVariance
  exact div_pos hvar (Nat.cast_pos.mpr hn)

theorem blockMeanVariance_mono_n (varBlock : ℝ) (n m : ℕ) (hn : 0 < n) (hm : 0 < m) (hvar : 0 < varBlock) (h_le : (n : ℝ) ≤ (m : ℝ)) (h_n_le_m : n ≤ m) :
    blockMeanVariance varBlock m hm ≤ blockMeanVariance varBlock n hn := by
  unfold blockMeanVariance
  have h_n_pos : (0 : ℝ) < (n : ℝ) := Nat.cast_pos.mpr hn
  have h_m_pos : (0 : ℝ) < (m : ℝ) := Nat.cast_pos.mpr hm
  exact div_le_div_of_nonneg_left (le_of_lt hvar) h_n_pos h_le

/-- Block sums are additive across a partition of blocks. -/
theorem block_sum_partition {α : Type} [DecidableEq α] (s : Finset α) (t : Finset α) (h_disj : Disjoint s t)
    (f : α → ℝ) : (∑ x ∈ s ∪ t, f x) = (∑ x ∈ s, f x) + (∑ x ∈ t, f x) :=
  Finset.sum_union h_disj

/-- Independence proxy: disjoint block sets have zero cross-term in variance
    expansion when covariance is zero. We prove the algebraic identity:
    (a + b)^2 = a^2 + b^2 when cross term 2ab is zero (independence). -/
theorem independent_blocks_variance_add (a b : ℝ) (h_cov_zero : a * b = 0) :
    (a + b) ^ 2 = a ^ 2 + b ^ 2 := by
  have h : 2 * a * b = 0 := by
    calc 2 * a * b = 2 * (a * b) := by ring
    _ = 2 * 0 := by rw [h_cov_zero]
    _ = 0 := by ring
  nlinarith [sq_nonneg a, sq_nonneg b, sq_nonneg (a + b)]

/-- Averaging over N independent blocks reduces variance by 1/N. -/
theorem independent_averaging_variance (varBlock : ℝ) (n : ℕ) (hn : 0 < n) :
    blockMeanVariance varBlock n hn * (n : ℝ) = varBlock := by
  unfold blockMeanVariance
  have h_n_ne : (n : ℝ) ≠ 0 := ne_of_gt (Nat.cast_pos.mpr hn)
  field_simp

abbrev ScrambleBlocks (n : ℕ) := Fin n

theorem scramble_blocks_card (n : ℕ) : Fintype.card (ScrambleBlocks n) = n := by
  simp [ScrambleBlocks]
/-- Finite factorization behind wall/scramble independence: the double sum of
    a product statistic over `Fin n` scrambles × `Fin m` walls factors into the
    product of the marginal sums. This is the finite discrete core of the claim
    that scramble blocks are independent of wall blocks.
    Stochastic extension (not proved here): upgrade to `MeasureTheory`
    `ProductMeasure` IID walls + semantic-RNG sampling units
    (see `EvaluationAxioms.axiom_wallBlock_independent_unit`). -/
theorem scramble_wall_product_mean_factors (n m : ℕ) (f : Fin n → ℝ) (g : Fin m → ℝ) :
    (∑ p : Fin n × Fin m, f p.1 * g p.2) = (∑ s, f s) * (∑ w, g w) := by
  rw [Fintype.sum_prod_type, Fintype.sum_mul_sum]

/-- Cross-term factorization over finite block sets: the product of block sums
    is the double sum of pairwise products (`Finset.sum_mul_sum`). Vanishing of
    this double sum is the finite proxy for zero covariance. -/
theorem wall_block_cross_factors {α β : Type*}
    (s : Finset α) (t : Finset β) (f : α → ℝ) (g : β → ℝ) :
    (∑ x ∈ s, f x) * (∑ y ∈ t, g y) = ∑ x ∈ s, ∑ y ∈ t, f x * g y :=
  Finset.sum_mul_sum s t f g

/-- Wall-block variance is additive over disjoint block sets with zero
    covariance: Var(sum) = sum of Vars. Reuses `independent_blocks_variance_add`
    with `Finset.sum_union` splitting the total over the partition. -/
theorem wall_block_variance_additive {α : Type*} [DecidableEq α]
    (s t : Finset α) (h_disj : Disjoint s t) (f : α → ℝ)
    (h_cov : (∑ x ∈ s, f x) * (∑ x ∈ t, f x) = 0) :
    (∑ x ∈ s ∪ t, f x) ^ 2 = (∑ x ∈ s, f x) ^ 2 + (∑ x ∈ t, f x) ^ 2 := by
  rw [Finset.sum_union h_disj]
  exact independent_blocks_variance_add _ _ h_cov

/-- Three-block case: pairwise zero covariance gives additivity over three
    blocks. Demonstrates the induction step extending
    `independent_blocks_variance_add` to `n` blocks (the full `Finset` induction
    is `wall_block_variance_additive_finset` below). -/
theorem wall_block_variance_additive_three (a b c : ℝ)
    (hab : a * b = 0) (hac : a * c = 0) (hbc : b * c = 0) :
    (a + b + c) ^ 2 = a ^ 2 + b ^ 2 + c ^ 2 := by
  have hab_c : (a + b) * c = 0 := by rw [add_mul, hac, hbc, add_zero]
  calc (a + b + c) ^ 2 = ((a + b) + c) ^ 2 := by rw [add_assoc]
    _ = (a + b) ^ 2 + c ^ 2 := independent_blocks_variance_add _ _ hab_c
    _ = (a ^ 2 + b ^ 2) + c ^ 2 := by rw [independent_blocks_variance_add _ _ hab]
    _ = a ^ 2 + b ^ 2 + c ^ 2 := by ring

/-- `n`-block variance additivity by `Finset` induction: under pairwise zero
    covariance, the square of the block-total equals the sum of block squares.
    The insert step splits the total with `Finset.sum_insert`, kills the cross
    term `a b * ∑` via `Finset.mul_sum` + `Finset.sum_eq_zero` (each summand
    vanishes by the pairwise hypothesis), then applies
    `independent_blocks_variance_add` and the induction hypothesis.
    `Finset.sum_add_distrib` is the companion splitting law used when the
    per-block statistic itself is a sum of two components. -/
theorem wall_block_variance_additive_finset {ι : Type*} [DecidableEq ι]
    (a : ι → ℝ) (s : Finset ι) :
    (∀ i ∈ s, ∀ j ∈ s, i ≠ j → a i * a j = 0) →
    (∑ i ∈ s, a i) ^ 2 = ∑ i ∈ s, (a i) ^ 2 := by
  induction s using Finset.induction with
  | empty => intro _; simp
  | @insert b t hb ih =>
    intro h
    have h' : ∀ i ∈ t, ∀ j ∈ t, i ≠ j → a i * a j = 0 := fun i hi j hj hne =>
      h i (Finset.mem_insert_of_mem hi) j (Finset.mem_insert_of_mem hj) hne
    rw [Finset.sum_insert hb, Finset.sum_insert hb]
    have hcross : a b * (∑ j ∈ t, a j) = 0 := by
      rw [Finset.mul_sum]
      apply Finset.sum_eq_zero
      intro j hj
      exact h b (Finset.mem_insert_self b t) j (Finset.mem_insert_of_mem hj)
        (fun heq => hb (heq ▸ hj))
    calc (a b + ∑ j ∈ t, a j) ^ 2
        = (a b) ^ 2 + (∑ j ∈ t, a j) ^ 2 := independent_blocks_variance_add _ _ hcross
      _ = (a b) ^ 2 + ∑ i ∈ t, (a i) ^ 2 := by rw [ih h']

end BlockIndependence

section FixedNPower

/-- Standardized effect size: δ = (μ1 - μ0) / σ. -/
noncomputable def effectSize (mu0 mu1 sigma : ℝ) (h_sigma_pos : 0 < sigma) : ℝ :=
  (mu1 - mu0) / sigma

theorem effectSize_zero_when_no_difference (mu sigma : ℝ) (h_sigma_pos : 0 < sigma) :
    effectSize mu mu sigma h_sigma_pos = 0 := by
  unfold effectSize; simp

theorem effectSize_sign (mu0 mu1 sigma : ℝ) (h_sigma_pos : 0 < sigma) :
    0 < effectSize mu0 mu1 sigma h_sigma_pos ↔ mu0 < mu1 := by
  unfold effectSize
  rw [div_pos_iff]
  constructor
  · intro h
    cases h with
    | inl h => exact sub_pos.mp h.1
    | inr h => linarith [h.2]
  · intro h
    left; exact ⟨sub_pos.mpr h, h_sigma_pos⟩

/-- Fixed-N two-sided z-test power approximation.
    `alpha` is two-sided significance, `n` is fixed block count,
    `delta` is standardized effect, `z_alpha` and `z_beta` are critical values.
    This is the textbook normal approximation: power ≈ Φ( δ√n - z_{1-α/2} ). -/
noncomputable def zPowerApprox (delta : ℝ) (n : ℕ) (z_alpha : ℝ) : ℝ :=
  delta * Real.sqrt (n : ℝ) - z_alpha

theorem zPowerApprox_mono_n (delta : ℝ) (n m : ℕ) (z_alpha : ℝ) (h_delta_pos : 0 ≤ delta) (h_n_le : n ≤ m) :
    zPowerApprox delta n z_alpha ≤ zPowerApprox delta m z_alpha := by
  unfold zPowerApprox
  have h_sqrt_mono : Real.sqrt (n : ℝ) ≤ Real.sqrt (m : ℝ) := by
    apply Real.sqrt_le_sqrt
    exact Nat.cast_le.mpr h_n_le
  linarith [mul_le_mul_of_nonneg_left h_sqrt_mono h_delta_pos]

/-- Strict version of `zPowerApprox_mono_n`: positive effect and strictly more
blocks give strictly more (proxy) power. Needs `Real.sqrt_lt_sqrt` (which
takes an explicit nonneg hypothesis, unlike the weak form). -/
theorem zPowerApprox_strict_mono_n (delta : ℝ) (n m : ℕ) (z_alpha : ℝ)
    (h_delta_pos : 0 < delta) (h : n < m) :
    zPowerApprox delta n z_alpha < zPowerApprox delta m z_alpha := by
  unfold zPowerApprox
  have h_sqrt_lt : Real.sqrt (n : ℝ) < Real.sqrt (m : ℝ) := by
    exact Real.sqrt_lt_sqrt (Nat.cast_nonneg _) (Nat.cast_lt.mpr h)
  linarith [mul_lt_mul_of_pos_left h_sqrt_lt h_delta_pos]

theorem zPowerApprox_mono_delta (n : ℕ) (z_alpha delta1 delta2 : ℝ) (h_le : delta1 ≤ delta2) :
    zPowerApprox delta1 n z_alpha ≤ zPowerApprox delta2 n z_alpha := by
  unfold zPowerApprox
  have h_n_nonneg : 0 ≤ Real.sqrt (n : ℝ) := Real.sqrt_nonneg _
  linarith [mul_le_mul_of_nonneg_right h_le h_n_nonneg]

theorem zPowerApprox_antitone_alpha (delta : ℝ) (n : ℕ) (z1 z2 : ℝ) (h_le : z1 ≤ z2) :
    zPowerApprox delta n z2 ≤ zPowerApprox delta n z1 := by
  unfold zPowerApprox; linarith

/-- Power is monotone in N and effect, antitone in critical value — the fixed-N
    formula's qualitative behavior that gates rely on. No MeasureTheory needed. -/
theorem fixedN_power_qualitative (delta1 delta2 : ℝ) (n1 n2 : ℕ) (z1 z2 : ℝ)
    (h_delta_le : delta1 ≤ delta2) (h_n_le : n1 ≤ n2) (h_z_le : z2 ≤ z1) (h_delta_nonneg : 0 ≤ delta1) :
    zPowerApprox delta1 n1 z1 ≤ zPowerApprox delta2 n2 z2 := by
  calc zPowerApprox delta1 n1 z1 ≤ zPowerApprox delta1 n2 z1 := zPowerApprox_mono_n delta1 n1 n2 z1 h_delta_nonneg h_n_le
    _ ≤ zPowerApprox delta2 n2 z1 := zPowerApprox_mono_delta n2 z1 delta1 delta2 h_delta_le
    _ ≤ zPowerApprox delta2 n2 z2 := zPowerApprox_antitone_alpha delta2 n2 z2 z1 h_z_le
/-- Fixed-N does not adapt: power is computed at the pre-registered N, not
    at a data-dependent stopping time. We formalize as: if you increase N
    after seeing data, you are computing a different function's value. -/
theorem fixedN_no_peeking (delta : ℝ) (z_alpha : ℝ) (n n' : ℕ) (h_ne : n ≠ n') :
    zPowerApprox delta n z_alpha ≠ zPowerApprox delta n' z_alpha ∨ delta = 0 ∨ n = n' := by
  by_cases h_delta : delta = 0
  · right; left; exact h_delta
  · by_cases h_eq : n = n'
    · right; right; exact h_eq
    · left
      have h_n_ne : (n : ℝ) ≠ (n' : ℝ) := by exact_mod_cast h_ne
      have h_sqrt_ne : Real.sqrt (n : ℝ) ≠ Real.sqrt (n' : ℝ) := by
        intro h_eq_sqrt
        have h_sq : (Real.sqrt (n : ℝ)) ^ 2 = (Real.sqrt (n' : ℝ)) ^ 2 := by rw [h_eq_sqrt]
        rw [Real.sq_sqrt (Nat.cast_nonneg _), Real.sq_sqrt (Nat.cast_nonneg _)] at h_sq
        exact h_n_ne h_sq
      intro h_pow_eq
      unfold zPowerApprox at h_pow_eq
      have h_mul_eq : delta * Real.sqrt (n : ℝ) = delta * Real.sqrt (n' : ℝ) := by linarith
      have h_delta_ne : delta ≠ 0 := h_delta
      have h_sqrt_eq : Real.sqrt (n : ℝ) = Real.sqrt (n' : ℝ) := by
        exact mul_left_cancel₀ h_delta_ne h_mul_eq
      exact h_sqrt_ne h_sqrt_eq

/-- Required N for target z-power (inverting the approximation).
    `n ≥ ((z_alpha + z_beta)/δ)^2`. -/
noncomputable def requiredBlocks (delta z_alpha z_beta : ℝ) (h_delta_pos : 0 < delta) : ℝ :=
  ((z_alpha + z_beta) / delta) ^ 2

theorem requiredBlocks_nonneg (delta z_alpha z_beta : ℝ) (h_delta_pos : 0 < delta) :
    0 ≤ requiredBlocks delta z_alpha z_beta h_delta_pos := by
  unfold requiredBlocks; positivity

theorem requiredBlocks_mono_effect (delta1 delta2 z_alpha z_beta : ℝ)
    (h1 : 0 < delta1) (h2 : 0 < delta2) (h_le : delta1 ≤ delta2) (h_z_nonneg : 0 ≤ z_alpha + z_beta) :
    requiredBlocks delta2 z_alpha z_beta h2 ≤ requiredBlocks delta1 z_alpha z_beta h1 := by
  unfold requiredBlocks
  have h1_pos : 0 < delta1 := h1
  have h2_pos : 0 < delta2 := h2
  have h_div_anti : (z_alpha + z_beta) / delta2 ≤ (z_alpha + z_beta) / delta1 := by
    apply div_le_div_of_nonneg_left h_z_nonneg (by linarith : 0 < delta1) h_le
  exact sq_le_sq' (by linarith [div_nonneg h_z_nonneg (le_of_lt h2_pos), div_nonneg h_z_nonneg (le_of_lt h1_pos)]) h_div_anti

/-- Inverting the approximation: `n` blocks beyond `requiredBlocks` achieve
`z_beta ≤ zPowerApprox` (deterministic proxy inversion; no `Φ` needed). -/
theorem requiredBlocks_sufficient (delta z_alpha z_beta : ℝ) (h_delta_pos : 0 < delta)
    (n : ℕ) (hn : requiredBlocks delta z_alpha z_beta h_delta_pos ≤ (n : ℝ)) :
    z_beta ≤ zPowerApprox delta n z_alpha := by
  unfold requiredBlocks at hn
  unfold zPowerApprox
  have hsqrt : (z_alpha + z_beta) / delta ≤ Real.sqrt (n : ℝ) :=
    Real.le_sqrt_of_sq_le hn
  have hmul : z_alpha + z_beta ≤ delta * Real.sqrt (n : ℝ) :=
    (div_le_iff₀' h_delta_pos).mp hsqrt
  linarith


end FixedNPower

section PromotionGate

/-- Promotion requires block-independent evidence and fixed-N power at the
    pre-registered bound. This bundles the two previous sections. `gatePower` is
    monotone in `n_blocks` and `delta`, antitone in `z_alpha` (`zPowerApprox_mono_n` etc.); `fixedN_power_qualitative` is the qualitative power statement `power>0` when `delta>0`, `fixedN_no_peeking` is frozen `N` before unblinding `SPEC §18.3` vs `timeUniformCS` `Howard` `hedged` `HARD skip`. -/
structure PromotionGate where
  n_blocks : ℕ
  n_pos : 0 < n_blocks
  delta : ℝ
  delta_pos : 0 < delta
  z_alpha : ℝ
  z_beta : ℝ

noncomputable def gatePower (g : PromotionGate) : ℝ :=
  zPowerApprox g.delta g.n_blocks g.z_alpha

noncomputable def gateRequiredBlocks (g : PromotionGate) : ℝ :=
  requiredBlocks g.delta g.z_alpha g.z_beta g.delta_pos

theorem gatePower_mono_blocks (g1 g2 : PromotionGate) (h_n_le : g1.n_blocks ≤ g2.n_blocks)
    (h_delta_eq : g1.delta = g2.delta) (h_z_eq : g1.z_alpha = g2.z_alpha) :
    gatePower g1 ≤ gatePower g2 := by
  unfold gatePower
  rw [h_delta_eq, h_z_eq]
  exact zPowerApprox_mono_n g2.delta g1.n_blocks g2.n_blocks g2.z_alpha (le_of_lt g2.delta_pos) h_n_le

theorem gate_sufficient (g : PromotionGate)
    (hn : gateRequiredBlocks g ≤ (g.n_blocks : ℝ)) : g.z_beta ≤ gatePower g :=
  requiredBlocks_sufficient g.delta g.z_alpha g.z_beta g.delta_pos g.n_blocks hn

end PromotionGate

end Hydra2.Blueprint.Evaluation
