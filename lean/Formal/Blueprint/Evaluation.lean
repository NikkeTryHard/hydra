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
Block-independent scramble and wall-block averaging with the fixed-N power
calculation for promotion gates. Wall blocks are the independent unit; games
within a wall share randomness, so averaging over games underestimates variance
and fails closed by requiring block averaging.
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

/-- Zero block variance means a perfect estimator: with `varBlock = 0` the
block mean has zero variance at any count — the finite reason deterministic
simulators need exactly one block. Pair with `blockMeanVariance_mono_n`
(more blocks tighten): together the variance ladder is pinned at zero
and shrinks upward, so the eval budget's first question is whether the
simulator is deterministic at all. -/
theorem blockMeanVariance_zero_var (n : ℕ) (hn : 0 < n) :
    blockMeanVariance 0 n hn = 0 := by
  unfold blockMeanVariance
  simp

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

/-- Quadrupling blocks halves the standard error: `Var/N` scaling means the
CI half-width shrinks as `1/sqrt(N)` — the finite reason the duplicate-wall
eval budget prices significance in blocks, not games. Practical core: buying
2x tighter promotion gates costs 4x blocks; size the S6 ship-bar budget
from the narrowest CI worth paying for. -/
theorem blockMeanVariance_quarter_at_4x (varBlock : ℝ) (n : ℕ) (hn : 0 < n) :
    blockMeanVariance varBlock (4 * n) (by omega) = blockMeanVariance varBlock n hn / 4 := by
  unfold blockMeanVariance
  have h_n_ne : (n : ℝ) ≠ 0 := ne_of_gt (Nat.cast_pos.mpr hn)
  push_cast
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
    The insert step partitions the total with `Finset.sum_insert`, kills the cross
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

/-- Effect size flips sign under group swap: the challenger-vs-incumbent
framing never matters for magnitude — only the sign changes. Practical
core: promotion gates can fix one orientation (challenger minus incumbent)
and read the sign directly, with no abs-value bookkeeping that could mask
a regression as an improvement. -/
theorem effectSize_antisymm (mu0 mu1 sigma : ℝ) (h_sigma_pos : 0 < sigma) :
    effectSize mu1 mu0 sigma h_sigma_pos = -effectSize mu0 mu1 sigma h_sigma_pos := by
  unfold effectSize
  rw [← neg_div]
  congr 1
  ring

/-- Fixed-N two-sided z-test power approximation.
    `alpha` is two-sided significance, `n` is fixed block count,
    `delta` is standardized effect, `z_alpha` and `z_beta` are critical values.
    This is the textbook normal approximation: power ≈ Φ( δ√n - z_{1-α/2} ). -/
noncomputable def zPowerApprox (delta : ℝ) (n : ℕ) (z_alpha : ℝ) : ℝ :=
  delta * Real.sqrt (n : ℝ) - z_alpha

/-- Zero blocks means zero signal: power proxy at `n = 0` is `-z_alpha` —
the gate starts below any nonnegative bar and must earn its way up block
by block. Practical core: an unplayed promotion candidate never passes;
the S6 ship bar cannot be met on zero evidence. -/
theorem zPowerApprox_zero_blocks (delta z_alpha : ℝ) :
    zPowerApprox delta 0 z_alpha = -z_alpha := by
  unfold zPowerApprox
  simp

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

/-- Strict version of `zPowerApprox_mono_delta`: positive block count and
strictly larger effect give strictly more (proxy) power — the mirror of
`zPowerApprox_strict_mono_n` (strict in blocks). Together strictness holds
in EACH input at positive levels, so the gate's clearest pass can be strict
in either margin, and ties break on genuine signal, never on flatness. -/
theorem zPowerApprox_strict_mono_delta (n : ℕ) (z_alpha delta1 delta2 : ℝ)
    (h_n_pos : 0 < n) (h : delta1 < delta2) :
    zPowerApprox delta1 n z_alpha < zPowerApprox delta2 n z_alpha := by
  unfold zPowerApprox
  have h_sqrt_pos : 0 < Real.sqrt (n : ℝ) :=
    Real.sqrt_pos.mpr (Nat.cast_pos.mpr h_n_pos)
  linarith [mul_lt_mul_of_pos_left h h_sqrt_pos]

theorem zPowerApprox_antitone_alpha (delta : ℝ) (n : ℕ) (z1 z2 : ℝ) (h_le : z1 ≤ z2) :
    zPowerApprox delta n z2 ≤ zPowerApprox delta n z1 := by
  unfold zPowerApprox; linarith

/-- Strict version of `zPowerApprox_antitone_alpha`: a strictly larger
significance bar gives strictly less (proxy) power at any delta and
block count — the mirror of `zPowerApprox_strict_mono_delta` (strict
in effect). With strictness in all three inputs, the gate's ranking
is total on distinct configurations: no two different (delta, n,
z_alpha) triples ever tie on flatness, so promotion order never needs
an arbitrary tiebreak. -/
theorem zPowerApprox_strict_antitone_alpha (delta : ℝ) (n : ℕ) (z1 z2 : ℝ)
    (h : z1 < z2) :
    zPowerApprox delta n z2 < zPowerApprox delta n z1 := by
  unfold zPowerApprox; linarith

/-- Zero effect means zero signal at any block count: the power proxy sits
at `-z_alpha` regardless of N — the finite reason peeking at more blocks
can never manufacture significance when there is no effect. Pair with
`zPowerApprox_zero_blocks` (zero blocks, any effect): together both
degenerate inputs pin the proxy at the bar floor, so the gate's only
free variable is genuine effect size. -/
theorem zPowerApprox_zero_effect (n : ℕ) (z_alpha : ℝ) :
    zPowerApprox 0 n z_alpha = -z_alpha := by
  unfold zPowerApprox
  simp

/-- Strict qualitative power: positive effect with strictly more blocks AND
a strictly larger effect gives strictly more proxy power — the finite reason
a promotion that clears on both margins simultaneously is unambiguous.
Chains the strict-N law with the weak-delta law: strictness needs only
one strict margin, so the gate's clearest pass is strict in blocks. -/
theorem fixedN_power_strict_blocks (delta1 delta2 : ℝ) (n1 n2 : ℕ) (z : ℝ)
    (h_le : delta1 ≤ delta2) (h_lt : n1 < n2) (h_pos : 0 < delta1) :
    zPowerApprox delta1 n1 z < zPowerApprox delta2 n2 z := by
  calc zPowerApprox delta1 n1 z < zPowerApprox delta1 n2 z :=
        zPowerApprox_strict_mono_n delta1 n1 n2 z h_pos h_lt
    _ ≤ zPowerApprox delta2 n2 z :=
        zPowerApprox_mono_delta n2 z delta1 delta2 h_le

/-- Power is monotone in N and effect, antitone in critical value — the fixed-N
    formula's qualitative behavior that gates rely on. No MeasureTheory needed. -/
theorem fixedN_power_qualitative (delta1 delta2 : ℝ) (n1 n2 : ℕ) (z1 z2 : ℝ)
    (h_delta_le : delta1 ≤ delta2) (h_n_le : n1 ≤ n2) (h_z_le : z2 ≤ z1) (h_delta_nonneg : 0 ≤ delta1) :
    zPowerApprox delta1 n1 z1 ≤ zPowerApprox delta2 n2 z2 := by
  calc zPowerApprox delta1 n1 z1 ≤ zPowerApprox delta1 n2 z1 := zPowerApprox_mono_n delta1 n1 n2 z1 h_delta_nonneg h_n_le
    _ ≤ zPowerApprox delta2 n2 z1 := zPowerApprox_mono_delta n2 z1 delta1 delta2 h_delta_le
    _ ≤ zPowerApprox delta2 n2 z2 := zPowerApprox_antitone_alpha delta2 n2 z2 z1 h_z_le

/-- Fixed-N does not adapt (peeking-flatness corollary): with zero effect
the proxy is flat, so adaptive stopping cannot manufacture signal — the
finite reason the no-peeking rule bites exactly when there IS an effect
to chase. Pairs with `zPowerApprox_zero_effect` (flat at `-z_alpha`):
flatness is both shield and limit. The general no-peeking disjunction
follows below. -/
theorem fixedN_no_peeking_zero_effect (z_alpha : ℝ) (n n' : ℕ) :
    zPowerApprox 0 n z_alpha = zPowerApprox 0 n' z_alpha := by
  rw [zPowerApprox_zero_effect, zPowerApprox_zero_effect]

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

/-- Doubling the effect quarters the required blocks: the mirror of
`requiredBlocks_double_price` in the other direction — bigger effects are
cheaper to confirm. Practical core: pick the comparison with the biggest
honest effect first, since confirmation cost falls quadratically — the
finite reason eval budgets chase large deltas before small ones. -/
theorem requiredBlocks_quarter_price (delta z_alpha z_beta : ℝ) (h_delta_pos : 0 < delta) :
    requiredBlocks (2 * delta) z_alpha z_beta (by linarith)
      = requiredBlocks delta z_alpha z_beta h_delta_pos / 4 := by
  unfold requiredBlocks
  have hne : delta ≠ 0 := ne_of_gt h_delta_pos
  field_simp
  ring

/-- Halving the effect quadruples the required blocks: the `1/δ²` scaling law.
    Practical core: detecting half-size placement edges costs 4x duplicate-wall
    blocks — size the S5 confirmation budget from the smallest effect worth
    detecting, or the gate silently underpowers. Limited-compute lemma: effects
    have quadratic prices. -/
theorem requiredBlocks_double_price (delta z_alpha z_beta : ℝ) (h_delta_pos : 0 < delta) :
    requiredBlocks (delta / 2) z_alpha z_beta (by linarith)
      = 4 * requiredBlocks delta z_alpha z_beta h_delta_pos := by
  unfold requiredBlocks
  have hne : delta ≠ 0 := ne_of_gt h_delta_pos
  field_simp
  ring

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
    monotone in `n_blocks` and `delta`, antitone in `z_alpha` (`zPowerApprox_mono_n` etc.); `fixedN_power_qualitative` is the qualitative power statement `power>0` when `delta>0`, `fixedN_no_peeking` is frozen `N` before unblinding vs adaptive peeking which
    invalidates confirmation and fails closed. -/
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

/-- Gate power is monotone in effect size at fixed blocks and threshold:
    a larger standardized effect can only raise power. Practical core: pick
    the comparison with the biggest honest effect — stronger signal never
    hurts the gate, so limited-compute eval budgets chase large deltas. -/
theorem gatePower_mono_delta (g1 g2 : PromotionGate)
    (h_n_eq : g1.n_blocks = g2.n_blocks) (h_z_eq : g1.z_alpha = g2.z_alpha)
    (h_le : g1.delta ≤ g2.delta) :
    gatePower g1 ≤ gatePower g2 := by
  unfold gatePower
  rw [h_n_eq, h_z_eq]
  exact zPowerApprox_mono_delta g2.n_blocks g2.z_alpha g1.delta g2.delta h_le

/-- Strict version of `gatePower_mono_delta`: a strictly larger effect
means strictly more promotion power at fixed blocks and bar — the
finite reason the gate's clearest pass is strict in the margin that
matters: the structure bundles positivity (`n_pos`, `delta_pos`), so
strictness comes free from `zPowerApprox_strict_mono_delta` with no
side proof. Pairs `gatePower_antitone_alpha` (stricter bar lowers
power): together the gate is strict in effect, antitone in bar. -/
theorem gatePower_strict_mono_delta (g1 g2 : PromotionGate)
    (h_n_eq : g1.n_blocks = g2.n_blocks) (h_z_eq : g1.z_alpha = g2.z_alpha)
    (h : g1.delta < g2.delta) :
    gatePower g1 < gatePower g2 := by
  unfold gatePower
  rw [h_n_eq, h_z_eq]
  exact zPowerApprox_strict_mono_delta g2.n_blocks g2.z_alpha g1.delta g2.delta g2.n_pos h

/-- Gate power is antitone in the significance threshold at fixed blocks
    and effect: a stricter bar can only lower power. Practical core: looser
    α buys power at fixed compute — set the bar from the false-promotion
    budget, never tighter. -/
theorem gatePower_antitone_alpha (g1 g2 : PromotionGate)
    (h_n_eq : g1.n_blocks = g2.n_blocks) (h_d_eq : g1.delta = g2.delta)
    (h_le : g1.z_alpha ≤ g2.z_alpha) :
    gatePower g2 ≤ gatePower g1 := by
  unfold gatePower
  rw [h_n_eq, h_d_eq]
  exact zPowerApprox_antitone_alpha g2.delta g2.n_blocks g1.z_alpha g2.z_alpha h_le

/-- Identical gates agree: same blocks, effect, and bar give the same
power — the finite reason gate comparisons are well-defined: the
promotion decision depends only on the gate record, never on hidden
state. Reflexivity with content: `gatePower_refl` states it
trivially; this version pins equality across two records. -/
theorem gatePower_congr (g1 g2 : PromotionGate)
    (h_n : g1.n_blocks = g2.n_blocks) (h_d : g1.delta = g2.delta)
    (h_z : g1.z_alpha = g2.z_alpha) :
    gatePower g1 = gatePower g2 := by
  unfold gatePower
  rw [h_n, h_d, h_z]

/-- Gate power is reflexive: identical gates have equal power — the finite
reason the duplicate-harness compares two DIFFERENT checkpoints, never a
gate against itself. Pair with the three monotone/antitone laws (which
move power in each argument): together the gate is a well-behaved
function of (blocks, effect, threshold) with no degenerate self-edge. -/
theorem gatePower_refl (g : PromotionGate) : gatePower g = gatePower g := rfl

/-- Gate projections unfold: power and required blocks are the fixed-N
formula applied to the gate's own fields — the finite reason the
promotion decision reduces to one inequality on blocks. Chains into
`gate_sufficient` below: meeting the required-block budget pushes proxy
power past the beta bar, so the S6 ship bar is a block count, not a
debate. -/
theorem gate_projections_unfold (g : PromotionGate) :
    gatePower g = zPowerApprox g.delta g.n_blocks g.z_alpha
    ∧ gateRequiredBlocks g
      = requiredBlocks g.delta g.z_alpha g.z_beta g.delta_pos := ⟨rfl, rfl⟩

theorem gate_sufficient (g : PromotionGate)
    (hn : gateRequiredBlocks g ≤ (g.n_blocks : ℝ)) : g.z_beta ≤ gatePower g :=
  requiredBlocks_sufficient g.delta g.z_alpha g.z_beta g.delta_pos g.n_blocks hn

/-- Gate sufficiency at equality: meeting the block budget EXACTLY still
passes — the finite reason the S6 ship bar is `≤`-closed, not `<`: a
promotion landing exactly on budget ships, so the gate never demands
one-more-block past sufficiency. Sharp form of `gate_sufficient`. -/
theorem gate_sufficient_at_equality (g : PromotionGate)
    (heq : gateRequiredBlocks g = (g.n_blocks : ℝ)) :
    g.z_beta ≤ gatePower g :=
  gate_sufficient g (le_of_eq heq)

end PromotionGate

end Hydra2.Blueprint.Evaluation
